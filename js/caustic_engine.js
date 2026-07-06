// caustic_engine.js — reusable pure-WebGPU realtime metal-caustic engine.
//
// Master home for the GPU photon-caustic source (speedball). It computes a
// reflective metal caustic entirely in TSL compute passes and resolves it into
// a StorageTexture that any receiver plane can sample.
//
// PORTABILITY: this module imports ONLY `three/tsl` (build-agnostic) and takes
// the THREE namespace as a parameter — so the SAME file vendors byte-identical
// into any host build. Pass THREE from whatever your build exposes the WebGPU
// classes under: `import * as THREE from 'three'` when an importmap maps three
// to three.webgpu.js (speedball), or `import * as THREE from 'three/webgpu'`
// under a bundler where bare `three` is the WebGL core (Vite / sigils).
//
// Pipeline (all on the GPU, no CPU readback):
//   emit   : one compute thread per photon; samples the caster, reflects off the
//            light, hits the receiver, atomic-splats fixed-point energy into a
//            u32 density grid (WGSL has no float atomics -> u32 fixed point).
//   convert: u32 grid -> float density buffer.
//   blur   : separable Gaussian density estimation (crisp cusps).
//   max    : atomicMax reduction of the blurred peak -> auto-exposure.
//   bloom  : threshold the bright cores, wide Gaussian -> hot-cusp halo.
//   resolve: gamma crush + bloom + metal tint + HDR tonemap -> RGBA16F StorageTexture.
//   overlay: caller adds `overlayMesh` and samples the texture additively.
//
// Progressive: the grid accumulates across frames (never cleared except on
// markDirty()) so it converges noisy->sharp in a few frames, then holds. A full
// rebake of millions of photons costs <1ms of GPU time.
//
// EMISSION SEAM: `buildEmit()` below is analytic (a curved door panel + two
// torus wheels) — the reference emitter that validates the look. To throw
// caustics off REAL geometry, replace buildEmit with one that traces the caster
// mesh via this repo's BVH tracer (js/spectral_traverse.js buildTraversal +
// spectral_scene.js buildSpectralScene). All photon-tracing lives here in
// speedball; that is the canonical path.

import {
    Fn, If, Loop, Return, instanceIndex, uniform, storage, textureStore, texture, uv,
    atomicAdd, atomicLoad, atomicMax, atomicStore,
    float, int, uint, vec3, vec4, uvec2,
    select, max, min, abs, sqrt, sin, cos, exp, pow, floor, ceil,
    dot, normalize, reflect, clamp,
} from 'three/tsl';

const PI = Math.PI;

// Receiver-plane presets for createCausticEngine({ receiver }).
//  - floor: horizontal y=y plane (the default).
//  - wall:  vertical plane at z=z facing +Z (a backdrop BEHIND a sigil at z>zWall).
export const causticReceiverFloor = ({ y = 0, width = 9, height = 7 } = {}) =>
    ({ origin: [0, y, 0], normal: [0, 1, 0], uAxis: [1, 0, 0], vAxis: [0, 0, 1], width, height });
export const causticReceiverWall = ({ z = -1.2, y = 0, width = 6, height = 6 } = {}) =>
    ({ origin: [0, y, z], normal: [0, 0, 1], uAxis: [1, 0, 0], vAxis: [0, 1, 0], width, height });

// Metal presets: tint (linear rgb 0..1) + a matching roughness.
export const CAUSTIC_METALS = {
    chrome: { rgb: [238 / 255, 247 / 255, 255 / 255], roughness: 0.035 },
    gold: { rgb: [255 / 255, 208 / 255, 105 / 255], roughness: 0.045 },
    copper: { rgb: [255 / 255, 158 / 255, 105 / 255], roughness: 0.055 },
};

/**
 * Create a GPU caustic engine bound to a WebGPURenderer.
 *
 * @param {object}  opts
 * @param {object}  opts.THREE               the three namespace exposing WebGPU classes
 * @param {import('three').WebGPURenderer} opts.renderer
 * @param {number} [opts.grid=768]            caustic grid resolution (WxH)
 * @param {number} [opts.targetPhotons=3e6]   photons accumulated before "converged"
 * @param {object} [opts.receiver]            receiver plane extents (world units)
 * @returns engine handle: { overlayMesh, texture, uniforms, update(), setters, dispose() }
 */
export function createCausticEngine({
    THREE,
    renderer,
    grid = 768,
    targetPhotons = 3_000_000,
    // Receiver plane the caustic lands on. `null` = the classic floor (y=0).
    // Otherwise a plane: { origin, normal (faces the incoming rays), uAxis, vAxis, width, height }.
    // (Not named `floor` — that would shadow the imported TSL floor() used in the kernels.)
    receiver = null,
} = {}) {
    if (!THREE) throw new Error('createCausticEngine requires the THREE namespace (pass { THREE, renderer }).');
    const W = grid, H = grid, cells = W * H;
    const SCALE = 256.0;       // fixed-point scale for atomic energy deposit
    const MAXSCALE = 64.0;     // fixed-point scale for the atomicMax auto-exposure
    const U32_MAX = 4.2e9;     // clamp ceiling below 2^32 to avoid atomic wrap
    // Normalized receiver frame (defaults reproduce the original floor exactly).
    const R = (() => {
        const d = receiver || {};
        const v3 = (a, fb) => new THREE.Vector3(...(a || fb));
        return {
            origin: v3(d.origin, [0, 0, 0]),
            normal: v3(d.normal, [0, 1, 0]).normalize(),
            uAxis: v3(d.uAxis, [1, 0, 0]).normalize(),
            vAxis: v3(d.vAxis, [0, 0, 1]).normalize(),
            width: d.width ?? 9,
            height: d.height ?? 7,
        };
    })();

    const params = {
        photonBudget: 300000,
        strength: 2.2,         // overlay additive opacity
        softness: 0.9,         // streak footprint + density-estimate bandwidth
        bloom: 0.7,
        roughness: 0.035,
        tint: CAUSTIC_METALS.chrome.rgb.slice(),
        light: [-2.2, 3.4, -1.45],
    };

    const makeStorage = (array) => new THREE.StorageBufferAttribute(array, 1);

    // ── storage ──────────────────────────────────────────────────────
    const grid_ = storage(makeStorage(new Uint32Array(cells)), 'uint', cells).toAtomic();
    const maxB = storage(makeStorage(new Uint32Array(1)), 'uint', 1).toAtomic();
    const densF = storage(makeStorage(new Float32Array(cells)), 'float', cells);
    const tmpB = storage(makeStorage(new Float32Array(cells)), 'float', cells);
    const sharpB = storage(makeStorage(new Float32Array(cells)), 'float', cells);
    const brightB = storage(makeStorage(new Float32Array(cells)), 'float', cells);
    const bloomB = storage(makeStorage(new Float32Array(cells)), 'float', cells);
    const storageAttrs = [grid_, maxB, densF, tmpB, sharpB, brightB, bloomB];

    // RGBA16F: storage-capable + HDR. sRGB storage formats are NOT storage-capable
    // in WebGPU, so keep it LINEAR and store linear values directly.
    const causticTex = new THREE.StorageTexture(W, H);
    causticTex.format = THREE.RGBAFormat;
    causticTex.type = THREE.HalfFloatType;
    causticTex.colorSpace = THREE.NoColorSpace;

    const U = {
        frameSeed: uniform(1, 'uint'),
        lightPos: uniform(new THREE.Vector3(...params.light)),
        doorMat: uniform(new THREE.Matrix4()),
        wheel0Mat: uniform(new THREE.Matrix4()),
        wheel1Mat: uniform(new THREE.Matrix4()),
        metalRoughness: uniform(params.roughness),
        causticWidth: uniform(params.softness),
        sharpSigma: uniform(1.0),
        bloomSigma: uniform(12.0),
        bloomGain: uniform(params.bloom * 2.0),
        tint: uniform(new THREE.Vector3(...params.tint)),
    };

    // Caster-mesh emission state (populated by setCasterMesh; when 'mesh',
    // photons are emitted off a real triangle mesh instead of the analytic
    // door/wheels). Geometry is baked to WORLD space on upload.
    let emitMode = 'analytic';
    let meshPos = null, meshNrm = null, meshIdx = null, meshCdf = null;
    let meshTriCount = 0, meshSearchIters = 1;
    const meshAttrs = []; // StorageBufferAttributes to dispose on rebuild

    // ── RNG (PCG) ────────────────────────────────────────────────────
    const INV_U32 = 1.0 / 4294967296.0;
    const pcgHash = (x) => {
        const st = x.mul(uint(747796405)).add(uint(2891336453));
        const word = st.shiftRight(st.shiftRight(uint(28)).add(uint(4))).bitXor(st).mul(uint(277803737));
        return word.shiftRight(uint(22)).bitXor(word);
    };
    const rngState = { node: null };
    const nextRand = () => {
        const s = rngState.node;
        const ns = s.mul(uint(747796405)).add(uint(2891336453));
        s.assign(ns);
        const word = ns.shiftRight(ns.shiftRight(uint(28)).add(uint(4))).bitXor(ns).mul(uint(277803737));
        const res = word.shiftRight(uint(22)).bitXor(word);
        return float(res).mul(INV_U32);
    };

    // bilinear 4-tap atomic deposit
    const addTap = (ix, iy, amt) => {
        const inb = ix.greaterThanEqual(int(0)).and(ix.lessThan(int(W)))
            .and(iy.greaterThanEqual(int(0))).and(iy.lessThan(int(H)));
        If(inb, () => {
            const cell = uint(iy).mul(uint(W)).add(uint(ix));
            const q = uint(clamp(amt.mul(float(SCALE)), float(0), float(U32_MAX)));
            atomicAdd(grid_.element(cell), q);
        });
    };

    // Shared: given a WORLD surface point + normal on the metal, reflect the
    // light off it, hit the floor plane, and splat the anisotropic streak.
    // Culls (Return) on back-face / up-going / off-floor — safe inside the kernel.
    // receiver plane as compile-time constant vectors
    const Ro = vec3(R.origin.x, R.origin.y, R.origin.z);
    const Rn = vec3(R.normal.x, R.normal.y, R.normal.z);
    const Ru = vec3(R.uAxis.x, R.uAxis.y, R.uAxis.z);
    const Rv = vec3(R.vAxis.x, R.vAxis.y, R.vAxis.z);
    const RhalfW = R.width * 0.5, RhalfH = R.height * 0.5;

    const emitTrace = (wp, wn, sourceGain) => {
        const toP = wp.sub(U.lightPos);
        const distSq = max(float(0.5), dot(toP, toP));
        const incident = normalize(toP);
        const ndl = max(float(0), dot(incident, wn).mul(-1));
        If(ndl.lessThanEqual(float(0.0001)), () => { Return(); });
        const reflected = normalize(reflect(incident, wn));
        // intersect the receiver plane; the reflected ray must enter its FRONT face
        const denom = dot(reflected, Rn);
        If(denom.greaterThanEqual(float(-0.012)), () => { Return(); });
        const t = dot(Ro.sub(wp), Rn).div(denom);
        If(t.lessThanEqual(float(0)), () => { Return(); });
        If(t.greaterThan(float(80)), () => { Return(); }); // allow grazing throws to distant receivers
        const hit = wp.add(reflected.mul(t));
        const d = hit.sub(Ro);
        const uu = dot(d, Ru);
        const vv = dot(d, Rv);
        If(abs(uu).greaterThan(float(RhalfW)), () => { Return(); });
        If(abs(vv).greaterThan(float(RhalfH)), () => { Return(); });

        const grazing = float(1).sub(min(float(1), abs(denom)));
        const grazeGain = float(0.55).add(grazing.mul(1.5));
        const roughPenalty = max(float(0.12), float(1).sub(U.metalRoughness.mul(5.5)));
        const energy = ndl.mul(roughPenalty).mul(sourceGain).mul(grazeGain).mul(float(8).div(distSq));

        const cx = uu.div(R.width).add(0.5).mul(float(W));
        const cy = float(0.5).sub(vv.div(R.height)).mul(float(H));
        const cdx0 = dot(reflected, Ru).mul(W / R.width);
        const cdy0 = dot(reflected, Rv).mul(-1).mul(H / R.height);
        const clen = max(sqrt(cdx0.mul(cdx0).add(cdy0.mul(cdy0))), float(1e-6));
        const cdx = cdx0.div(clen);
        const cdy = cdy0.div(clen);
        const streakPx = min(float(18), float(2).add(grazing.mul(15)).mul(U.causticWidth));
        const steps = uint(max(float(1), ceil(streakPx)));
        const stepsF = float(steps);
        const ePer = energy.div(stepsF);

        Loop({ start: uint(0), end: steps, type: 'uint', condition: '<' }, ({ i: s }) => {
            const tt = select(steps.equal(uint(1)), float(0),
                float(s).div(max(stepsF.sub(1), float(1))).sub(0.5));
            const fx = cx.add(cdx.mul(streakPx).mul(tt)).sub(0.5);
            const fy = cy.add(cdy.mul(streakPx).mul(tt)).sub(0.5);
            const x0 = int(floor(fx));
            const y0 = int(floor(fy));
            const txf = fx.sub(float(x0));
            const tyf = fy.sub(float(y0));
            addTap(x0, y0, ePer.mul(float(1).sub(txf)).mul(float(1).sub(tyf)));
            addTap(x0.add(int(1)), y0, ePer.mul(txf).mul(float(1).sub(tyf)));
            addTap(x0, y0.add(int(1)), ePer.mul(float(1).sub(txf)).mul(tyf));
            addTap(x0.add(int(1)), y0.add(int(1)), ePer.mul(txf).mul(tyf));
        });
    };

    // ── ANALYTIC emitter (reference: curved door + torus wheels) ─────
    function buildEmit(count) {
        return Fn(() => {
            const pid = instanceIndex.toVar();
            If(pid.greaterThanEqual(uint(count)), () => { Return(); });
            rngState.node = pcgHash(pid.bitXor(pcgHash(U.frameSeed))).toVar();

            const pick = nextRand();
            const wp = vec3(0, 0, 0).toVar();
            const wn = vec3(0, 0, 1).toVar();
            const sourceGain = float(1).toVar();

            // 44% curved door panel
            If(pick.lessThan(float(0.44)), () => {
                const u = nextRand(); const v = nextRand();
                const x = u.mul(2).sub(1).mul(2.7);
                const y = v.mul(2).sub(1).mul(0.86);
                const arg = x.mul(2.25).add(y.mul(0.55));
                const shoulderAng = x.div(2.7).mul(PI * 0.5);
                const shoulder = cos(shoulderAng).mul(0.18);
                const crease = sin(arg).mul(0.035);
                const crown = x.mul(x).mul(0.05);
                const z = shoulder.add(crease).add(crown);
                const CREASE = 0.12; // low-pass the caster normal so cusps focus
                const dzdx = sin(shoulderAng).mul(-(PI * 0.5 / 2.7) * 0.18)
                    .add(cos(arg).mul(2.25 * 0.035 * CREASE)).add(x.mul(0.1));
                const dzdy = cos(arg).mul(0.55 * 0.035 * CREASE);
                const lp = vec3(x, y, z);
                const ln = normalize(vec3(dzdx.mul(-1), dzdy.mul(-1), float(1)));
                wp.assign(U.doorMat.mul(vec4(lp, 1)).xyz);
                wn.assign(normalize(U.doorMat.mul(vec4(ln, 0)).xyz));
            });
            // 56% torus wheels
            If(pick.greaterThanEqual(float(0.44)), () => {
                const major = select(nextRand().lessThan(float(0.72)), float(0.56), float(0.31));
                const minor = select(major.greaterThan(float(0.4)), float(0.072), float(0.035));
                const theta = nextRand().mul(2 * PI);
                const phi = nextRand().mul(2 * PI);
                const tube = major.add(minor.mul(cos(phi)));
                const lp = vec3(tube.mul(cos(theta)), tube.mul(sin(theta)), minor.mul(sin(phi)));
                const ln = normalize(vec3(cos(phi).mul(cos(theta)), cos(phi).mul(sin(theta)), sin(phi)));
                const which = nextRand();
                If(which.lessThan(float(0.5)), () => {
                    wp.assign(U.wheel0Mat.mul(vec4(lp, 1)).xyz);
                    wn.assign(normalize(U.wheel0Mat.mul(vec4(ln, 0)).xyz));
                });
                If(which.greaterThanEqual(float(0.5)), () => {
                    wp.assign(U.wheel1Mat.mul(vec4(lp, 1)).xyz);
                    wn.assign(normalize(U.wheel1Mat.mul(vec4(ln, 0)).xyz));
                });
                sourceGain.assign(float(1.55));
            });

            emitTrace(wp, wn, sourceGain);
        })().compute(count);
    }

    // ── MESH emitter: sample the caster triangle mesh (world-space) ──
    // Area-weighted triangle pick (binary-search the CDF) + barycentric point +
    // interpolated normal, then the shared reflect/floor/splat. This is the
    // "throw caustics off real geometry" path — emission only (no ray tracing);
    // wiring the reflected ray to buildTraversal is the next upgrade.
    function buildMeshEmit(count) {
        const triCount = meshTriCount;
        const iters = meshSearchIters;
        const fetchP = (i) => { const b = i.mul(uint(3)); return vec3(meshPos.element(b), meshPos.element(b.add(uint(1))), meshPos.element(b.add(uint(2)))); };
        const fetchN = (i) => { const b = i.mul(uint(3)); return vec3(meshNrm.element(b), meshNrm.element(b.add(uint(1))), meshNrm.element(b.add(uint(2)))); };
        return Fn(() => {
            const pid = instanceIndex.toVar();
            If(pid.greaterThanEqual(uint(count)), () => { Return(); });
            rngState.node = pcgHash(pid.bitXor(pcgHash(U.frameSeed))).toVar();

            // area-weighted triangle via binary search over the cumulative CDF
            const r = nextRand();
            const lo = int(0).toVar();
            const hi = int(triCount - 1).toVar();
            Loop({ start: uint(0), end: uint(iters), type: 'uint', condition: '<' }, () => {
                const mid = int(floor(float(lo.add(hi)).mul(0.5)));
                If(meshCdf.element(uint(mid)).lessThan(r), () => { lo.assign(mid.add(int(1))); });
                If(meshCdf.element(uint(mid)).greaterThanEqual(r), () => { hi.assign(mid); });
            });
            const base = uint(lo).mul(uint(3));
            const i0 = meshIdx.element(base);
            const i1 = meshIdx.element(base.add(uint(1)));
            const i2 = meshIdx.element(base.add(uint(2)));

            // uniform barycentric (sqrt trick)
            const su = sqrt(nextRand());
            const r2 = nextRand();
            const b0 = float(1).sub(su);
            const b1 = su.mul(float(1).sub(r2));
            const b2 = su.mul(r2);
            const wp = fetchP(i0).mul(b0).add(fetchP(i1).mul(b1)).add(fetchP(i2).mul(b2)).toVar();
            const wn = normalize(fetchN(i0).mul(b0).add(fetchN(i1).mul(b1)).add(fetchN(i2).mul(b2))).toVar();

            emitTrace(wp, wn, float(1.0));
        })().compute(count);
    }

    // ── resolve chain passes ─────────────────────────────────────────
    const clearGrid = Fn(() => {
        const idx = instanceIndex.toVar();
        If(idx.greaterThanEqual(uint(cells)), () => { Return(); });
        atomicStore(grid_.element(idx), uint(0));
    })().compute(cells);

    const convert = Fn(() => {
        const idx = instanceIndex.toVar();
        If(idx.greaterThanEqual(uint(cells)), () => { Return(); });
        densF.element(idx).assign(float(atomicLoad(grid_.element(idx))).div(float(SCALE)));
    })().compute(cells);

    const makeBlur = (src, dst, horizontal, sigmaU) => Fn(() => {
        const idx = instanceIndex.toVar();
        If(idx.greaterThanEqual(uint(cells)), () => { Return(); });
        const x = int(idx.mod(uint(W)));
        const y = int(idx.div(uint(W)));
        const sigma = max(sigmaU, float(0.35));
        const R = int(clamp(ceil(sigma.mul(3)), float(1), float(64)));
        const inv2s2 = float(-0.5).div(sigma.mul(sigma));
        const span = R.mul(int(2)).add(int(1));
        const sum = float(0).toVar();
        const wsum = float(0).toVar();
        Loop({ start: uint(0), end: uint(span), type: 'uint', condition: '<' }, ({ i: tstep }) => {
            const k = int(tstep).sub(R);
            const w = exp(float(k).mul(float(k)).mul(inv2s2));
            const sx = horizontal ? clamp(x.add(k), int(0), int(W - 1)) : x;
            const sy = horizontal ? y : clamp(y.add(k), int(0), int(H - 1));
            const sidx = uint(sy).mul(uint(W)).add(uint(sx));
            sum.addAssign(src.element(sidx).mul(w));
            wsum.addAssign(w);
        });
        dst.element(idx).assign(sum.div(max(wsum, float(1e-6))));
    })().compute(cells);

    const sharpH = makeBlur(densF, tmpB, true, U.sharpSigma);
    const sharpV = makeBlur(tmpB, sharpB, false, U.sharpSigma);
    const bloomH = makeBlur(brightB, tmpB, true, U.bloomSigma);
    const bloomV = makeBlur(tmpB, bloomB, false, U.bloomSigma);

    const clearMax = Fn(() => { atomicStore(maxB.element(uint(0)), uint(0)); })().compute(1);

    const reduceMax = Fn(() => {
        const idx = instanceIndex.toVar();
        If(idx.greaterThanEqual(uint(cells)), () => { Return(); });
        const q = uint(clamp(sharpB.element(idx).mul(float(MAXSCALE)), float(0), float(U32_MAX)));
        atomicMax(maxB.element(uint(0)), q);
    })().compute(cells);

    const threshold = Fn(() => {
        const idx = instanceIndex.toVar();
        If(idx.greaterThanEqual(uint(cells)), () => { Return(); });
        const invMax = float(MAXSCALE).div(max(float(atomicLoad(maxB.element(uint(0)))), float(1)));
        const nd = sharpB.element(idx).mul(invMax);
        brightB.element(idx).assign(max(float(0), nd.sub(float(0.4))));
    })().compute(cells);

    const resolve = Fn(() => {
        const idx = instanceIndex.toVar();
        If(idx.greaterThanEqual(uint(cells)), () => { Return(); });
        const px = idx.mod(uint(W));
        const py = idx.div(uint(W));
        const invMax = float(MAXSCALE).div(max(float(atomicLoad(maxB.element(uint(0)))), float(1)));
        const nd = max(sharpB.element(idx).mul(invMax), float(0));
        const core = pow(nd, float(2.6));                     // contrast crush -> dark wash, bright cusps
        const I = core.add(bloomB.element(idx).mul(U.bloomGain));
        const v = float(1).sub(exp(I.mul(-1.6)));             // soft HDR saturation
        const hot = max(float(0), v.sub(float(0.75)).div(float(0.25))); // hot cores -> white
        const rr = v.mul(U.tint.x.add(float(1).sub(U.tint.x).mul(hot)));
        const gg = v.mul(U.tint.y.add(float(1).sub(U.tint.y).mul(hot)));
        const bb = v.mul(U.tint.z.add(float(1).sub(U.tint.z).mul(hot)));
        textureStore(causticTex, uvec2(px, py), vec4(rr, gg, bb, float(1)));
    })().compute(cells);

    // ── overlay plane (caller adds this to a scene) ──────────────────
    const overlayMat = new THREE.MeshBasicNodeMaterial();
    overlayMat.colorNode = texture(causticTex, uv());
    overlayMat.transparent = true;
    overlayMat.blending = THREE.AdditiveBlending;
    overlayMat.depthTest = false;
    overlayMat.depthWrite = false;
    overlayMat.toneMapped = false;
    overlayMat.opacity = params.strength;
    const overlayGeo = new THREE.PlaneGeometry(R.width, R.height);
    const overlayMesh = new THREE.Mesh(overlayGeo, overlayMat);
    // Orient the plane to the receiver frame: local +X→uAxis, +Y→vAxis, +Z→normal.
    overlayMesh.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(R.uAxis, R.vAxis, R.normal));
    overlayMesh.position.copy(R.origin).addScaledVector(R.normal, 0.012);
    overlayMesh.renderOrder = 1000;

    // ── progressive dispatch ─────────────────────────────────────────
    let dirty = true, accum = 0, converged = false, emit = null, emitCount = 0;

    function syncUniforms() {
        U.lightPos.value.set(params.light[0], params.light[1], params.light[2]);
        U.metalRoughness.value = params.roughness;
        U.causticWidth.value = params.softness;
        U.sharpSigma.value = Math.max(0.5, 0.5 + params.softness + params.roughness * 45);
        U.bloomSigma.value = 9 + params.softness * 12;
        U.bloomGain.value = params.bloom * 2.0;
        U.tint.value.set(params.tint[0], params.tint[1], params.tint[2]);
        overlayMat.opacity = params.strength;
    }
    syncUniforms();

    function ensureEmit(count) {
        if (emitCount !== count) {
            emit = emitMode === 'mesh' ? buildMeshEmit(count) : buildEmit(count);
            emitCount = count;
        }
        return emit;
    }

    function markDirty() { dirty = true; }

    // Call once per frame. Returns { accum, converged }.
    function update() {
        if (dirty) {
            renderer.compute(clearGrid);
            accum = 0; converged = false; dirty = false;
            U.frameSeed.value = (0x51f15e + Math.floor(params.light[0] * 1000) + Math.floor(params.light[1] * 100)) >>> 0;
        }
        if (converged) return { accum, converged };
        const batch = Math.max(1000, params.photonBudget | 0);
        const e = ensureEmit(batch);
        U.frameSeed.value = (U.frameSeed.value + 1) >>> 0;
        renderer.compute(e);
        renderer.compute(convert);
        renderer.compute(sharpH); renderer.compute(sharpV);
        renderer.compute(clearMax); renderer.compute(reduceMax);
        renderer.compute(threshold);
        renderer.compute(bloomH); renderer.compute(bloomV);
        renderer.compute(resolve);
        accum += batch;
        if (accum >= targetPhotons) converged = true;
        return { accum, converged };
    }

    // ── setters ──────────────────────────────────────────────────────
    function setLight(x, y, z) { params.light = [x, y, z]; U.lightPos.value.set(x, y, z); markDirty(); }
    function setCasterMatrices(doorMat, wheel0Mat, wheel1Mat) {
        if (doorMat) U.doorMat.value.copy(doorMat);
        if (wheel0Mat) U.wheel0Mat.value.copy(wheel0Mat);
        if (wheel1Mat) U.wheel1Mat.value.copy(wheel1Mat);
        markDirty();
    }
    function setMetal(name) {
        const p = CAUSTIC_METALS[name] || CAUSTIC_METALS.chrome;
        params.tint = p.rgb.slice(); params.roughness = p.roughness;
        syncUniforms(); markDirty();
    }
    // Switch to MESH emission: bake `mesh` (a THREE.Mesh) to world space, build a
    // per-triangle area CDF, upload geometry to storage, and emit photons off it.
    // Re-call to update after the mesh moves or its geometry changes.
    function setCasterMesh(mesh) {
        const geo = mesh.geometry;
        const posAttr = geo.getAttribute('position');
        if (!posAttr) throw new Error('setCasterMesh: geometry has no position attribute');
        mesh.updateMatrixWorld(true);
        const m = mesh.matrixWorld;
        const nm = new THREE.Matrix3().getNormalMatrix(m);
        const nAttr = geo.getAttribute('normal');
        const vCount = posAttr.count;
        const wpos = new Float32Array(vCount * 3);
        const wnrm = new Float32Array(vCount * 3);
        const _v = new THREE.Vector3(), _n = new THREE.Vector3();
        for (let i = 0; i < vCount; i++) {
            _v.fromBufferAttribute(posAttr, i).applyMatrix4(m);
            wpos[i * 3] = _v.x; wpos[i * 3 + 1] = _v.y; wpos[i * 3 + 2] = _v.z;
            if (nAttr) _n.fromBufferAttribute(nAttr, i).applyMatrix3(nm).normalize(); else _n.set(0, 1, 0);
            wnrm[i * 3] = _n.x; wnrm[i * 3 + 1] = _n.y; wnrm[i * 3 + 2] = _n.z;
        }
        const idxArr = geo.index
            ? new Uint32Array(geo.index.array)
            : Uint32Array.from({ length: vCount }, (_, i) => i);
        const triCount = Math.floor(idxArr.length / 3);
        if (triCount < 1) throw new Error('setCasterMesh: geometry has no triangles');

        // cumulative, area-normalized CDF over triangles (world space)
        const cdf = new Float32Array(triCount);
        const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3(), e1 = new THREE.Vector3(), e2 = new THREE.Vector3();
        let acc = 0;
        for (let t = 0; t < triCount; t++) {
            const i0 = idxArr[t * 3], i1 = idxArr[t * 3 + 1], i2 = idxArr[t * 3 + 2];
            a.set(wpos[i0 * 3], wpos[i0 * 3 + 1], wpos[i0 * 3 + 2]);
            b.set(wpos[i1 * 3], wpos[i1 * 3 + 1], wpos[i1 * 3 + 2]);
            c.set(wpos[i2 * 3], wpos[i2 * 3 + 1], wpos[i2 * 3 + 2]);
            acc += 0.5 * e1.subVectors(b, a).cross(e2.subVectors(c, a)).length();
            cdf[t] = acc;
        }
        const inv = acc > 0 ? 1 / acc : 0;
        for (let t = 0; t < triCount; t++) cdf[t] *= inv;
        cdf[triCount - 1] = 1.0; // guard the search's upper bound

        for (const at of meshAttrs) at?.dispose?.();
        meshAttrs.length = 0;
        const posA = makeStorage(wpos), nrmA = makeStorage(wnrm), idxA = makeStorage(idxArr), cdfA = makeStorage(cdf);
        meshAttrs.push(posA, nrmA, idxA, cdfA);
        meshPos = storage(posA, 'float', wpos.length);
        meshNrm = storage(nrmA, 'float', wnrm.length);
        meshIdx = storage(idxA, 'uint', idxArr.length);
        meshCdf = storage(cdfA, 'float', triCount);
        meshTriCount = triCount;
        meshSearchIters = Math.max(1, Math.ceil(Math.log2(Math.max(2, triCount))) + 1);
        emitMode = 'mesh';
        emitCount = -1; // force ensureEmit to rebuild with the mesh emitter
        markDirty();
    }
    function setMetalTint(r, g, b) { params.tint = [r, g, b]; U.tint.value.set(r, g, b); }
    function setRoughness(v) { params.roughness = v; syncUniforms(); markDirty(); }
    function setSoftness(v) { params.softness = v; syncUniforms(); markDirty(); }
    function setBloom(v) { params.bloom = v; U.bloomGain.value = v * 2.0; }  // resolve-only; no restart
    function setStrength(v) { params.strength = v; overlayMat.opacity = v; }
    function setPhotonBudget(n) { params.photonBudget = n; }

    function dispose() {
        for (const a of storageAttrs) a?.value?.dispose?.();
        for (const a of meshAttrs) a?.dispose?.();
        causticTex.dispose?.();
        overlayGeo.dispose(); overlayMat.dispose();
    }

    return {
        overlayMesh,
        texture: causticTex,
        uniforms: U,
        update,
        setLight, setCasterMatrices, setCasterMesh, setMetal, setMetalTint,
        setRoughness, setSoftness, setBloom, setStrength, setPhotonBudget,
        markDirty, dispose,
        get accum() { return accum; },
        get converged() { return converged; },
    };
}
