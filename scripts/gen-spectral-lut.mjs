// gen-spectral-lut.mjs — build the RGB→reflectance coefficient LUT for the
// spectral path tracer. Clean-room implementation of the sigmoid-of-a-quadratic
// reflectance model (Jakob & Hanika 2019): the reflectance of an RGB colour is
//
//     s(L) = 1/2 + 1/2 * x / sqrt(1 + x*x),   x = c0*L^2 + c1*L + c2
//
// with L the wavelength remapped to [0,1] over the tracer's [380,720] band.
// The three coefficients (c0,c1,c2) are fit per RGB so the reflectance, observed
// through the SAME CIE response the kernel integrates (Wyman 2013 fits) under an
// equal-energy illuminant, reproduces the target colour. We solve them on a 3D
// grid — parameterised, per the paper, by (argmax channel, max component z, and
// the two smaller channels scaled by z) — by damped Gauss-Newton, walking z so
// each solve seeds from its neighbour. The grid is uniform in all three axes so
// the GPU can fetch with hardware trilinear filtering.
//
// Output: js/srgb_lut.js (ES module, base64 float32) — magic, res, then
// res*res*(res*3) RGBA float32 texels (RGB = c0,c1,c2, A = 1) laid out as a
// Data3DTexture of width=res(x), height=res(y), depth=res*3 (z within slab,
// slabs stacked). Run: node scripts/gen-spectral-lut.mjs [res]
//
// The math (sigmoid model, grid parameterisation, Gauss-Newton + continuation)
// is from the published paper; no third-party source was copied.

import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const RES = Math.max(8, Math.min(64, parseInt(process.argv[2] || '36', 10) || 36));
const OUT = resolve(__dirname, '../js/srgb_lut.js');

const LAMBDA_MIN = 380, LAMBDA_MAX = 720;
const STEP = 5;                                       // nm; 5 nm = 69 samples
const NSAMP = Math.floor((LAMBDA_MAX - LAMBDA_MIN) / STEP) + 1;
const COEFFS = 3;

// ── Wyman 2013 single-lobe CIE 1931 fits (must mirror spectral_kernel.js) ──
function wymanG(x, mu, s1, s2) {
    const s = x < mu ? s1 : s2;
    const e = (x - mu) * s;
    return Math.exp(-0.5 * e * e);
}
const cieX = (l) => 1.056 * wymanG(l, 599.8, 0.0264, 0.0323)
    + 0.362 * wymanG(l, 442.0, 0.0624, 0.0374)
    - 0.065 * wymanG(l, 501.1, 0.0490, 0.0382);
const cieY = (l) => 0.821 * wymanG(l, 568.8, 0.0213, 0.0247)
    + 0.286 * wymanG(l, 530.9, 0.0613, 0.0322);
const cieZ = (l) => 1.217 * wymanG(l, 437.0, 0.0845, 0.0278)
    + 0.681 * wymanG(l, 459.0, 0.0385, 0.0725);

// Precompute the observer at each integer wavelength, plus the remapped L.
const LAM = new Float64Array(NSAMP);
const Ln = new Float64Array(NSAMP);   // wavelength remapped to [0,1]
const CX = new Float64Array(NSAMP);
const CY = new Float64Array(NSAMP);
const CZ = new Float64Array(NSAMP);
let Xint = 0, Yint = 0, Zint = 0;
for (let i = 0; i < NSAMP; i++) {
    const l = LAMBDA_MIN + i * STEP;
    LAM[i] = l;
    Ln[i] = (l - LAMBDA_MIN) / (LAMBDA_MAX - LAMBDA_MIN);
    CX[i] = cieX(l); CY[i] = cieY(l); CZ[i] = cieZ(l);
    Xint += CX[i]; Yint += CY[i]; Zint += CZ[i];
}

// sigmoid reflectance from the quadratic value
function sigmoid(x) { return 0.5 + 0.5 * x / Math.sqrt(1 + x * x); }

// Observed normalised XYZ for a coefficient triple (equal-energy illuminant,
// kernel's /Yint normalisation so flat reflectance s=1 → Y=1).
function observe(c0, c1, c2) {
    let X = 0, Y = 0, Z = 0;
    for (let i = 0; i < NSAMP; i++) {
        const L = Ln[i];
        const s = sigmoid((c0 * L + c1) * L + c2);
        X += CX[i] * s; Y += CY[i] * s; Z += CZ[i] * s;
    }
    return [X / Yint, Y / Yint, Z / Yint];
}

// ── sRGB(D65) → XYZ, chromatically adapted to equal-energy so a white RGB maps
//    to a flat (s=1) reflectance under this observer. Rows scaled so M·(1,1,1)
//    equals the equal-energy white the observer yields. ──
// sRGB linear → XYZ (D65), standard matrix.
const M_D65 = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
];
// Bradford adaptation D65 → E (equal energy).
function mul3(A, B) {
    const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++)
        C[r][c] = A[r][0] * B[0][c] + A[r][1] * B[1][c] + A[r][2] * B[2][c];
    return C;
}
function mulV(A, v) {
    return [A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2],
        A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2],
        A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2]];
}
const BRAD = [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]];
const BRAD_INV = [[0.9869929, -0.1470543, 0.1599627], [0.4323053, 0.5183603, 0.0492912], [-0.0085287, 0.0400428, 0.9684867]];
const XYZ_D65 = [0.95047, 1.0, 1.08883];
const XYZ_E = [1.0, 1.0, 1.0];
const csD65 = mulV(BRAD, XYZ_D65);
const csE = mulV(BRAD, XYZ_E);
const DIAG = [[csE[0] / csD65[0], 0, 0], [0, csE[1] / csD65[1], 0], [0, 0, csE[2] / csD65[2]]];
const ADAPT = mul3(BRAD_INV, mul3(DIAG, BRAD)); // D65 → E
let M_E = mul3(ADAPT, M_D65);
// Rescale rows so white RGB(1,1,1) hits the observer's equal-energy white
// (Xint/Yint, 1, Zint/Yint) — guarantees white → flat reflectance.
{
    const w = mulV(M_E, [1, 1, 1]);
    const want = [Xint / Yint, 1, Zint / Yint];
    for (let r = 0; r < 3; r++) {
        const k = want[r] / w[r];
        for (let c = 0; c < 3; c++) M_E[r][c] *= k;
    }
}
function rgbToTargetXYZ(r, g, b) { return mulV(M_E, [r, g, b]); }

// ── Damped Gauss-Newton solve for the 3 coefficients ──
function solve(target, init) {
    let c = init.slice();
    let lambda = 1e-3;
    let prevErr = Infinity;
    for (let iter = 0; iter < 40; iter++) {
        const f = observe(c[0], c[1], c[2]);
        const r = [f[0] - target[0], f[1] - target[1], f[2] - target[2]];
        const err = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        if (err < 1e-10) break;
        // Jacobian via forward differences (reuse the base observation f).
        const J = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
        for (let k = 0; k < 3; k++) {
            const h = 1e-3 * (1 + Math.abs(c[k]));
            const cp = c.slice(); cp[k] += h;
            const fp = observe(cp[0], cp[1], cp[2]);
            for (let row = 0; row < 3; row++) J[row][k] = (fp[row] - f[row]) / h;
        }
        // Levenberg-Marquardt: (JtJ + lambda·diag) dc = -Jt·r
        const JtJ = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
        const Jtr = [0, 0, 0];
        for (let a = 0; a < 3; a++) {
            for (let b2 = 0; b2 < 3; b2++) {
                let s = 0; for (let row = 0; row < 3; row++) s += J[row][a] * J[row][b2];
                JtJ[a][b2] = s;
            }
            let s = 0; for (let row = 0; row < 3; row++) s += J[row][a] * r[row];
            Jtr[a] = s;
        }
        for (let a = 0; a < 3; a++) JtJ[a][a] += lambda * (JtJ[a][a] + 1e-9);
        const dc = solve3(JtJ, [-Jtr[0], -Jtr[1], -Jtr[2]]);
        if (!dc) { lambda *= 10; continue; }
        const cand = [c[0] + dc[0], c[1] + dc[1], c[2] + dc[2]];
        const fc = observe(cand[0], cand[1], cand[2]);
        const ec = (fc[0] - target[0]) ** 2 + (fc[1] - target[1]) ** 2 + (fc[2] - target[2]) ** 2;
        if (ec < err) { c = cand; lambda = Math.max(lambda * 0.5, 1e-7); }
        else { lambda *= 4; }
        if (Math.abs(prevErr - ec) < 1e-12) break;
        prevErr = ec;
    }
    return c;
}

// 3x3 linear solve (Cramer); null if singular.
function solve3(A, b) {
    const det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
        - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
        + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    if (Math.abs(det) < 1e-20) return null;
    const inv = 1 / det;
    const x = [
        (b[0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (b[1] * A[2][2] - A[1][2] * b[2]) + A[0][2] * (b[1] * A[2][1] - A[1][1] * b[2])) * inv,
        (A[0][0] * (b[1] * A[2][2] - A[1][2] * b[2]) - b[0] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) + A[0][2] * (A[1][0] * b[2] - b[1] * A[2][0])) * inv,
        (A[0][0] * (A[1][1] * b[2] - b[1] * A[2][1]) - A[0][1] * (A[1][0] * b[2] - b[1] * A[2][0]) + b[0] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])) * inv,
    ];
    return x;
}

// ── Build the grid ──
// Layout: data[(((slab*RES + zi)*RES + yi)*RES + xi)*COEFFS + k]. zi/yi/xi all
// uniform in [0,1]. slab = argmax channel. For slab i: z=max=rgb[i],
// x=rgb[(i+1)%3]/z, y=rgb[(i+2)%3]/z (the two smaller channels normalised).
const data = new Float64Array(3 * RES * RES * RES * COEFFS);
function gridRGB(slab, z, x, y) {
    // x,y are the smaller channels in [0,1] (× z); reconstruct full RGB.
    const rgb = [0, 0, 0];
    rgb[slab] = z;
    rgb[(slab + 1) % 3] = x * z;
    rgb[(slab + 2) % 3] = y * z;
    return rgb;
}

console.log(`Generating ${RES}^3 × 3-slab reflectance LUT (${(3 * RES * RES * RES * COEFFS * 4 / 1e6).toFixed(2)} MB float32)…`);
let worst = 0, sumErr = 0, nfit = 0;
for (let slab = 0; slab < 3; slab++) {
    for (let yi = 0; yi < RES; yi++) {
        const y = yi / (RES - 1);
        for (let xi = 0; xi < RES; xi++) {
            const x = xi / (RES - 1);
            // z-continuation: start from a mid-bright z and walk outward so each
            // solve seeds from a neighbour that is already in-basin.
            let seed = [0, 0, 0];
            const order = [];
            const mid = Math.floor((RES - 1) * 0.75);
            for (let zi = mid; zi < RES; zi++) order.push(zi);
            for (let zi = mid - 1; zi >= 0; zi--) order.push(zi);
            let upSeed = [0, 0, 0], downSeed = null;
            for (const zi of order) {
                const z = Math.max(1e-4, zi / (RES - 1));
                const rgb = gridRGB(slab, z, x, y);
                const target = rgbToTargetXYZ(rgb[0], rgb[1], rgb[2]);
                const init = (zi >= mid) ? upSeed : (downSeed || upSeed);
                const c = solve(target, init);
                if (zi === mid) { upSeed = c; downSeed = c; }
                else if (zi > mid) upSeed = c;
                else downSeed = c;
                const base = (((slab * RES + zi) * RES + yi) * RES + xi) * COEFFS;
                data[base] = c[0]; data[base + 1] = c[1]; data[base + 2] = c[2];
                // accuracy bookkeeping
                const f = observe(c[0], c[1], c[2]);
                const e = Math.sqrt((f[0] - target[0]) ** 2 + (f[1] - target[1]) ** 2 + (f[2] - target[2]) ** 2);
                worst = Math.max(worst, e); sumErr += e; nfit++;
            }
        }
    }
    process.stdout.write(`  slab ${slab + 1}/3 done\n`);
}
console.log(`Fit complete. mean XYZ error ${(sumErr / nfit).toExponential(2)}, worst ${worst.toExponential(2)} over ${nfit} points.`);

// ── Serialize as RGBA float32, Data3DTexture-ready (w=RES, h=RES, d=RES*3) ──
// depth index = slab*RES + zi; texel(x=xi, y=yi, z=slab*RES+zi).
const texW = RES, texH = RES, texD = RES * 3;
const out = new Float32Array(texW * texH * texD * 4);
for (let slab = 0; slab < 3; slab++) {
    for (let zi = 0; zi < RES; zi++) {
        for (let yi = 0; yi < RES; yi++) {
            for (let xi = 0; xi < RES; xi++) {
                const src = (((slab * RES + zi) * RES + yi) * RES + xi) * COEFFS;
                const d = (slab * RES + zi);
                const dst = (((d * texH) + yi) * texW + xi) * 4;
                out[dst] = data[src];
                out[dst + 1] = data[src + 1];
                out[dst + 2] = data[src + 2];
                out[dst + 3] = 1;
            }
        }
    }
}
mkdirSync(dirname(OUT), { recursive: true });
// ES module (base64 float32) so the tracer imports it synchronously and it
// bundles into standalone snapshots — no async fetch / sidecar needed. Layout:
// RES, then RES×RES×(RES*3) RGBA float32 texels; depth d = slab*RES + zi packs
// the 3 argmax slabs. The loader slices that into 3 Data3DTextures.
const b64 = Buffer.from(out.buffer).toString('base64');
const mod = `// AUTO-GENERATED by scripts/gen-spectral-lut.mjs — do not edit by hand.
// sRGB -> reflectance coefficient LUT (sigmoid-of-quadratic model). RES^3 per
// argmax slab (3 slabs), RGBA float32 (RGB = c0,c1,c2), packed depth = slab*RES+zi.
export const SPECTRAL_LUT_RES = ${RES};
export const SPECTRAL_LUT_LAMBDA_MIN = ${LAMBDA_MIN};
export const SPECTRAL_LUT_LAMBDA_MAX = ${LAMBDA_MAX};
const B64 = '${b64}';
export function decodeSpectralLut() {
    const bin = typeof atob === 'function'
        ? atob(B64)
        : Buffer.from(B64, 'base64').toString('binary');
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
}
`;
writeFileSync(OUT, mod);
console.log(`Wrote ${OUT} (${(mod.length / 1e6).toFixed(2)} MB JS, res=${RES}, ${out.length * 4 / 1e6} MB raw).`);

// ── Round-trip sanity ──
function roundtrip(r, g, b) {
    // direct fit (no LUT quantisation) for a reference
    const t = rgbToTargetXYZ(r, g, b);
    const c = solve(t, [0, 0, 0]);
    const f = observe(c[0], c[1], c[2]);
    return { target: t, got: f };
}
for (const [name, rgb] of [['white', [1, 1, 1]], ['mid-grey', [0.5, 0.5, 0.5]],
    ['red', [0.8, 0.1, 0.1]], ['green', [0.1, 0.7, 0.1]], ['blue', [0.1, 0.1, 0.8]]]) {
    const { target, got } = roundtrip(...rgb);
    const e = Math.hypot(got[0] - target[0], got[1] - target[1], got[2] - target[2]);
    console.log(`  ${name.padEnd(9)} target XYZ [${target.map((v) => v.toFixed(3)).join(', ')}] got [${got.map((v) => v.toFixed(3)).join(', ')}]  err ${e.toExponential(2)}`);
}
