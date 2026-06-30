// gi_oct.js — octahedral direction <-> [0,1]² mapping for HALO-GI probe atlases.
//
// Phase-0 primitive (design spec docs/GI_HALO_design.md §7). r185 TSL ships no
// octahedral helper (equirectUV exists, octahedral = 0), so we hand-roll the
// signNotZero map (Cigolle/Donow/Meyer/McGuire/Luebke 2014, "A Survey of
// Efficient Representations for Independent Unit Vectors").
//
// Two parallel implementations that MUST stay numerically identical:
//   - JS  (octEncode / octDecode): CPU-side packing, relocation rays, and the
//     pure round-trip unit test (scripts/gi-oct-test.mjs).
//   - TSL (octEncodeNode / octDecodeNode): the probe trace/blend/sample kernels.
//     TSL is INJECTED (not imported) so this module also loads under plain node
//     for the unit test — matching the repo's builder convention.
//
// Convention: encode maps a unit vector to uv in [0,1]² (tile-local, BEFORE the
// 1px gutter is added). decode is its inverse. The +Z hemisphere fills the
// centre diamond; the −Z hemisphere folds into the four corners. Seam-correct
// by construction: opposite tile edges map to antipodal directions, which is
// exactly what the border-gutter copy relies on for continuous HW bilinear.

// ── JS (CPU) ────────────────────────────────────────────────────────
function signNotZero(k) { return k >= 0 ? 1 : -1; }

// Encode unit dir (nx,ny,nz) → uv in [0,1]². `out` is a length-≥2 array.
export function octEncode(nx, ny, nz, out = [0, 0]) {
    const inv = 1 / (Math.abs(nx) + Math.abs(ny) + Math.abs(nz) || 1e-20);
    let ox = nx * inv;
    let oy = ny * inv;
    if (nz < 0) {
        // fold the lower hemisphere into the corners. NOTE the .yx swap:
        // both new components read the ORIGINAL ox/oy, so compute then assign.
        const fx = (1 - Math.abs(oy)) * signNotZero(ox);
        const fy = (1 - Math.abs(ox)) * signNotZero(oy);
        ox = fx;
        oy = fy;
    }
    out[0] = ox * 0.5 + 0.5;
    out[1] = oy * 0.5 + 0.5;
    return out;
}

// Decode uv in [0,1]² → unit dir. `out` is a length-≥3 array.
export function octDecode(u, v, out = [0, 0, 0]) {
    const fx = u * 2 - 1;
    const fy = v * 2 - 1;
    let nx = fx;
    let ny = fy;
    const nz = 1 - Math.abs(fx) - Math.abs(fy);
    const t = Math.max(-nz, 0);
    nx += nx >= 0 ? -t : t;
    ny += ny >= 0 ? -t : t;
    const len = Math.hypot(nx, ny, nz) || 1;
    out[0] = nx / len;
    out[1] = ny / len;
    out[2] = nz / len;
    return out;
}

// ── TSL (GPU) — mirrors the JS above exactly. TSL namespace injected. ─
// n: vec3 unit-dir node → vec2 uv node in [0,1]².
export function octEncodeNode(n, TSL) {
    const { float, vec2, abs, select, max } = TSL;
    const sgn = (k) => select(k.greaterThanEqual(float(0)), float(1), float(-1));
    const inv = float(1).div(max(abs(n.x).add(abs(n.y)).add(abs(n.z)), float(1e-20)));
    const ox = n.x.mul(inv);
    const oy = n.y.mul(inv);
    const foldedX = float(1).sub(abs(oy)).mul(sgn(ox));
    const foldedY = float(1).sub(abs(ox)).mul(sgn(oy));
    const lower = n.z.lessThan(float(0));
    const ex = select(lower, foldedX, ox);
    const ey = select(lower, foldedY, oy);
    return vec2(ex.mul(0.5).add(0.5), ey.mul(0.5).add(0.5));
}

// uv: vec2 node in [0,1]² → vec3 unit-dir node.
export function octDecodeNode(uv, TSL) {
    const { float, vec3, abs, select, normalize, max } = TSL;
    const fx = uv.x.mul(2).sub(1);
    const fy = uv.y.mul(2).sub(1);
    const nz = float(1).sub(abs(fx)).sub(abs(fy));
    const t = max(nz.negate(), float(0));
    const nx = fx.add(select(fx.greaterThanEqual(float(0)), t.negate(), t));
    const ny = fy.add(select(fy.greaterThanEqual(float(0)), t.negate(), t));
    return normalize(vec3(nx, ny, nz));
}

export default { octEncode, octDecode, octEncodeNode, octDecodeNode };
