// gi_probes.js — HALO-GI DDGI irradiance field (docs/GI_HALO_design.md §3).
//
// A world-space grid of octahedral irradiance probes traced against the SAME
// stackless BVH the spectral path tracer uses (shared byte-identically via
// spectral_traverse.js — no second acceleration structure). Pure WebGPU/TSL
// compute; nothing reads back to the CPU.
//
// MVP (Phase 1): single grid, trace + cosine-gather blend + temporal hysteresis,
// infinite bounce over frames (trace reads last frame's atlas), miss = BLACK
// (the load-bearing energy invariant — probes carry ONLY surface inter-reflection;
// PMREM IBL owns the sky). Leak-free Chebyshev + relocation/classification and
// SSILVB are Phase 2/3.
//
// Churn-free by construction (fixes the surfel-grid recompile freeze):
//   • irradiance STATE lives in a read_write StorageBufferAttribute (irrBuffer).
//   • a write-only StorageTexture ATLAS is uploaded from it for HW-bilinear
//     sampling; the material's atlas binding is STABLE, so per-tick data writes
//     never change the material cache key. Only grid resize / enable flips it.

import * as THREE from 'three';
import { LightingNode } from 'three/webgpu';
import {
    Fn, If, Loop, Return, instanceIndex, storage, uniform, texture,
    float, int, uint, vec2, vec3, vec4, uvec2,
    max as tslMax, min as tslMin, mix, clamp, floor, normalize, dot, cross, length,
    abs as tslAbs, sqrt, cos, sin, pow, exp, textureStore, positionWorld, normalWorld, select,
} from 'three/tsl';

// buildSpectralScene (pulls three-mesh-bvh) is lazy-loaded in rebuild() so
// importing this module for the GiProbeNode (e.g. from max_lights_node.js) does
// NOT drag the CPU BVH builder into that module graph.
let _buildSpectralScene = null;
let _collectLights = null;       // cheap light re-collect for reactivity (no BVH rebuild)
let _LIGHT_STRIDE = 16;
import { buildTraversal, T_MAX, RAY_EPS, PI } from './spectral_traverse.js';
import { octEncodeNode, octDecodeNode } from './gi_oct.js';

// namespace injected into the octahedral node builders (gi_oct.js).
const TSL = { float, vec2, vec3, abs: tslAbs, select, max: tslMax, normalize };

const OCT_RES = 6;                 // interior octahedral resolution per probe
const BORDER = 1;                  // 1px gutter on every side
const TILE = OCT_RES + 2 * BORDER; // 8×8 atlas tile
const RAYS_PER_PROBE_DEFAULT = 64; // MVP ray budget (doc target 144). LOCKED baseline:
                                   // divisions=12 → 624 probes is tuned at 64 rays. Live via setRays().
const RAYS_MIN = 32, RAYS_MAX = 256;
// DIAGNOSTIC: gate the in-trace albedo/emissive TEXTURE sampling. false → packed-color
// baseline (the proven-working path); true → Lumen-style textured bounce. Flipped to
// isolate whether the texture path is what's blacking out the field.
const GI_SAMPLE_TEXTURES = false;
const CLASSIFY_RAYS = 32;          // fixed full-sphere rays for classification
const BACKFACE_FRACTION = 0.25;    // > this fraction backface hits → probe is buried → INACTIVE
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
const TARGET_PROBES_LONG_AXIS = 12;   // default probes along the longest grid axis (live via setDivisions)
const MAX_PROBES_PER_AXIS = 32;
const ATLAS_DIM_FALLBACK = 8192;      // assumed GPU maxTextureDimension2D when the device limit is unreadable
const MAX_PROBES_PER_TICK = 128;    // cap compute bursts on room-scale grids
const SURFACE_NORMAL_BIAS_CELL = 0.03; // sample 3% of a cell off the shaded wall, not half a cell
const TRACE_SURFACE_BIAS_CELL = 0.005; // shadow/NEE ray origin bias, scaled to scene units
const GI_CHEBY_BIAS_CELL = 0.08;       // Chebyshev SELF-OCCLUSION tolerance as a fraction of the
                                       // min cell. A lit surface must not shadow itself against its
                                       // own low-res depth moments (that's the "leak-free errors on
                                       // triangles" — per-triangle self-occlusion). Big enough to
                                       // absorb the normal-bias offset + oct-depth averaging error,
                                       // small enough to stay UNDER wall thickness so leak-free
                                       // visibility through real walls is preserved.
const MAX_TRIANGLES = 4_000_000;
const MAX_LIGHTS = 64;             // matches spectral_scene LIGHT_STRIDE table
// ── reactivity: respond to live light/geometry edits ──
const REACTIVE_TICKS = 75;         // ticks of EASED re-convergence after a light/geo edit
const REACTIVE_HYSTERESIS = 0.8;   // STARTING hysteresis of the reactive fade; it ramps UP to
                                   // the steady-state value across the burst so an edit fades in
                                   // smoothly. (Was a flat 0.4 → passed 60% of a freshly-rotated
                                   // 64-ray estimate each tick → the "flickers left/right" boil.)
const ROT_MIN_TICKS = 6;           // min ticks between ray-set rotations. Small grids (≤
                                   // MAX_PROBES_PER_TICK probes = a 1-tick full pass) otherwise
                                   // rotate the ray set EVERY tick, injecting fresh per-tick noise
                                   // the temporal blend has to absorb. Large grids (multi-tick
                                   // passes ≥ this) are unaffected — they already rotate per pass.
const LIGHT_CHECK_INTERVAL = 6;    // ticks between light-change checks
const GEO_CHECK_INTERVAL = 24;     // ticks between geometry-change checks (full rebuild = expensive)
const GEO_SETTLE_INTERVALS = 2;    // geo must be stable this many checks before a rebuild fires (debounce)
// ── freeze-proofing: gate the synchronous BVH rebuild + GPU solve on viewport idle ──
// The CPU MeshBVH build in rebuild() blocks the render thread, so it (and the GPU
// solve) must NEVER land while the user is orbiting, the timeline is playing, or a
// delta-sync burst is in flight. GI is world-space, so holding the field static
// during motion is visually lossless; it resumes and converges once the view rests.
const GI_IDLE_MS = 200;            // ms of camera/sync quiet before GI work resumes
const REBUILD_BACKOFF_TICKS = 45;  // ticks to wait after a failed/empty rebuild before retrying
// ── denoise uplift (CORE, docs/GI_HALO_design.md §11) tunables ──
const GI_FILTER_K = 8.0;           // spatial filter: variance→edge-stop bandwidth
const GI_FILTER_EPS = 0.001;       // spatial filter luma² absolute floor (avoids /0 on black)
const GI_FILTER_REL = 0.0225;      // spatial filter RELATIVE floor (~15% luma)²: even a temporally
                                   // converged texel gets a mild edge-PRESERVING bilateral smooth of
                                   // sub-threshold (noise-scale) neighbours, while strong directional
                                   // edges (red↔green) stay sharp. Steady-state splotch reduction.

let _node = null;

// ── injection node: samples the atlas at the shaded surface and adds the
// probe irradiance into builder.context.irradiance (mirrors the GiVolumeNode /
// hemisphere addAssign pattern at max_lights_node.js:224). Stable atlas binding
// → its cacheToken changes ONLY on grid resize / enable, never on data writes.
export class GiProbeNode extends LightingNode {
    static get type() { return 'GiProbeNode'; }

    constructor() {
        super();
        this._atlas = null;
        this._depthAtlas = null;
        this._stateAtlas = null;
        this._enabled = false;
        this._hasData = false;
        this._structGen = 0;     // bumps on grid resize / atlas realloc ONLY
        this.intensity = 1.0;

        this.gridMinNode = uniform(new THREE.Vector3());
        this.gridSizeNode = uniform(new THREE.Vector3(1, 1, 1));
        this.resNode = uniform(new THREE.Vector3(2, 2, 2));
        this.atlasDimNode = uniform(new THREE.Vector2(1, 1));
        this.intensityNode = uniform(1.0);
        // Runtime enable gate. The graph membership below is still the authoritative
        // on/off switch; this uniform remains as a cheap extra guard for compiled graphs.
        this.enabledNode = uniform(1.0);
        this.normalBiasNode = uniform(0.04);
        // 0 → no visibility test = pure trilinear "radiosity" look (THE DEFAULT, by user pref:
        // smoother, no per-triangle self-occlusion); 1 → full Chebyshev leak-free visibility.
        // The Chebyshev term self-occludes on dense/thick geometry and hurt the look more than
        // leaks helped, so it's off by default and has no UI toggle. Reachable via setCheby(1).
        this.chebyStrengthNode = uniform(0.0);
        // Chebyshev self-occlusion tolerance (WORLD units). Surface stays "visible" to its
        // own probes within this depth → stops the leak-free term erroring on triangles.
        // Set per rebuild to minCell * GI_CHEBY_BIAS_CELL; tweak live via this uniform.
        this.chebyBiasNode = uniform(0.0);
        // 0 → classification IGNORED (default — safe for thin 2-sided walls, which
        // a backface test misreads); 1 → drop probes buried in SOLID geometry.
        this.classifyStrengthNode = uniform(0.0);
    }

    // Graph membership is the authoritative GI switch. WebGPU node uniforms can stay
    // cached in already-built light graphs, so enable/disable must change whether this
    // lighting node is pushed at all; the page marks PBR materials dirty after flips.
    // Graph membership = "is there a field to sample". MUST stay decoupled from
    // enabled/intensity: setup() gates those via the enabledNode/intensityNode UNIFORMS,
    // so the node is folded into the shader ONCE (when data appears) and stays put.
    // Gating active on _enabled/intensity reintroduces the original bug — toggling them
    // only flips the cacheToken, which never forces a recompile, so GI silently drops out.
    get active() { return this._hasData; }
    // structure-only token: data writes (textureStore) do NOT change this, so
    // materials never recompile on a probe tick — only on resize / first data.
    get cacheToken() { return `gi-halo-probes:${this._structGen}`; }

    setEnabled(on) { this._enabled = on === true; this.enabledNode.value = this._enabled ? 1.0 : 0.0; }
    setIntensity(v) {
        this.intensity = Number.isFinite(v) ? Math.max(0, v) : 0;
        this.intensityNode.value = this.intensity;
    }
    setChebyStrength(v) { if (Number.isFinite(v)) this.chebyStrengthNode.value = THREE.MathUtils.clamp(v, 0, 1); }
    setClassifyStrength(v) { if (Number.isFinite(v)) this.classifyStrengthNode.value = THREE.MathUtils.clamp(v, 0, 1); }
    setAtlases(atlas, depthAtlas, stateAtlas) {
        this._atlas = atlas || null;
        this._depthAtlas = depthAtlas || null;
        this._stateAtlas = stateAtlas || null;
        this._hasData = !!atlas;
        this._structGen++;
    }
    // Update the grid placement uniforms only. Uniform .value writes do NOT change
    // a material cache key, so this is churn-free — the same-dim rebuild path uses
    // it to re-place probes after a geometry edit WITHOUT a TSL recompile.
    updateGridUniforms(gridMin, gridSize, res, atlasW, atlasH, normalBias, chebyBias) {
        this.gridMinNode.value.copy(gridMin);
        this.gridSizeNode.value.copy(gridSize);
        this.resNode.value.copy(res);
        this.atlasDimNode.value.set(atlasW, atlasH);
        if (Number.isFinite(normalBias)) this.normalBiasNode.value = Math.max(1e-4, normalBias);
        if (Number.isFinite(chebyBias)) this.chebyBiasNode.value = Math.max(0, chebyBias);
    }
    setGrid(gridMin, gridSize, res, atlasW, atlasH, normalBias, chebyBias) {
        this.updateGridUniforms(gridMin, gridSize, res, atlasW, atlasH, normalBias, chebyBias);
        this._structGen++;   // resize/first-enable ONLY → cacheToken moves → one recompile
    }

    // world position of grid probe (px,py,pz).
    _probePos(px, py, pz) {
        const f = vec3(px, py, pz).div(this.resNode.sub(1.0).max(vec3(1.0)));
        return this.gridMinNode.add(f.mul(this.gridSizeNode));
    }

    // tile-local atlas uv for probe (col,row) at octahedral coord octUV.
    _tileUV(col, row, octUV) {
        const ox = col.mul(float(TILE)).add(float(BORDER)).add(octUV.x.mul(float(OCT_RES))).add(0.5);
        const oy = row.mul(float(TILE)).add(float(BORDER)).add(octUV.y.mul(float(OCT_RES))).add(0.5);
        return vec2(ox.div(this.atlasDimNode.x), oy.div(this.atlasDimNode.y));
    }

    // sample the probe field at world (P, N): trilinear over the 8 cage probes,
    // each fetched octahedrally in the shading-normal direction and weighted by
    // a depth-moment Chebyshev visibility test (leak-free through thin walls).
    sampleIrradiance(P, N) {
        const atlas = this._atlas;
        const depthAtlas = this._depthAtlas;
        const res = this.resNode;
        const ry = res.y;
        const cell = this.gridSizeNode.div(res.sub(1.0).max(vec3(1.0)));
        const gridF = P.sub(this.gridMinNode).div(cell.max(vec3(1e-6)));
        const baseF = gridF.floor().clamp(vec3(0.0), res.sub(2.0).max(vec3(0.0)));
        const frac = gridF.sub(baseF).clamp(0.0, 1.0);
        const Nn = N.normalize();
        const octN = octEncodeNode(Nn, TSL); // irradiance dir = shading normal

        // PURE-EXPRESSION accumulation (NO toVar/addAssign): this runs inside a
        // fragment material colorNode (not an Fn-wrapped compute kernel), where
        // var mutation does NOT sequence — toVar/addAssign would silently yield 0.
        // The loop is unrolled (8 taps), so a plain expression tree is correct.
        let acc = vec3(0.0);
        let wsum = float(0.0);
        const bx = baseF.x.toUint(), by = baseF.y.toUint(), bz = baseF.z.toUint();
        for (let i = 0; i < 8; i++) {
            const dx = i & 1, dy = (i >> 1) & 1, dz = (i >> 2) & 1;
            const px = float(bx.add(uint(dx)));
            const py = float(by.add(uint(dy)));
            const pz = float(bz.add(uint(dz)));
            const col = px;
            const row = pz.mul(ry).add(py);
            const wx = dx ? frac.x : float(1.0).sub(frac.x);
            const wy = dy ? frac.y : float(1.0).sub(frac.y);
            const wz = dz ? frac.z : float(1.0).sub(frac.z);
            const wTri = wx.mul(wy).mul(wz).add(1e-4);

            // per-probe meta (NEAREST): R=state, GBA=relocation offset. Gated by
            // classifyStrength (default 0 = ignored) — relocation/classification by a
            // backface test misreads thin 2-sided walls, so it's opt-in for solid scenes.
            const metaUV = vec2(col.add(0.5).div(this.resNode.x), row.add(0.5).div(this.resNode.y.mul(this.resNode.z)));
            const meta = texture(this._stateAtlas, metaUV);
            const stateV = meta.x;
            const reloc = vec3(meta.y, meta.z, meta.w).mul(this.classifyStrengthNode);

            // Chebyshev visibility: relocated probe → surface direction vs stored depth.
            const probePos = this._probePos(px, py, pz).add(reloc);
            const toSurf = P.sub(probePos);
            const dist = length(toSurf);
            const octD = octEncodeNode(toSurf.div(dist.max(float(1e-6))), TSL);
            const m = texture(depthAtlas, this._tileUV(col, row, octD));
            const m1 = m.x; const m2 = m.y;
            const variance = m2.sub(m1.mul(m1)).abs();
            // Self-occlusion tolerance: a lit surface must not shadow ITSELF against its own
            // low-res depth moments. dist (probe→fragment) carries the normal-bias offset and
            // is compared to oct-averaged depth, so on dense/thick geometry it slips just past
            // m1 per triangle → the leak-free term "errors on triangles". A depth bias (db, <
            // wall thickness) treats the surface as visible within tolerance, so real walls
            // still occlude (no leak) but a surface stops fighting its own shadow.
            const db = this.chebyBiasNode;
            const dm = dist.sub(m1).sub(db).max(float(0.0));
            const chebyRaw = variance.div(variance.add(dm.mul(dm)).max(float(1e-6)));
            const cheby = select(dist.lessThanEqual(m1.add(db)), float(1.0), chebyRaw);
            const visW = mix(float(1.0), tslMax(cheby.mul(cheby).mul(cheby), float(0.05)), this.chebyStrengthNode);

            // Smooth backface/wrap weight (standard DDGI; was MISSING). Fades out probes whose
            // hemisphere faces AWAY from the surface, so the Chebyshev term no longer has to
            // HARD-cut them — that hard cut is the splotch that fights normal bias. Gated by
            // chebyStrength so leak control = 0 stays the exact pure-trilinear look.
            const dirToProbe = probePos.sub(P).div(dist.max(float(1e-6)));
            const wrap = dot(dirToProbe, Nn).mul(0.5).add(0.5);
            const wrapW = mix(float(1.0), tslMax(wrap.mul(wrap), float(0.05)), this.chebyStrengthNode);

            const stateEff = mix(float(1.0), stateV, this.classifyStrengthNode);
            const w = wTri.mul(wrapW).mul(visW).mul(stateEff);
            const e = texture(atlas, this._tileUV(col, row, octN)).xyz;
            acc = acc.add(e.mul(w));
            wsum = wsum.add(w);
        }
        return acc.div(wsum.max(float(1e-4)));
    }

    setup(builder) {
        if (!this._hasData || !this._atlas || !this._depthAtlas || !this._stateAtlas) return;
        const P = positionWorld.add(normalWorld.mul(this.normalBiasNode));
        const N = normalWorld;
        const E = this.sampleIrradiance(P, N).max(vec3(0.0)).mul(this.intensityNode).mul(this.enabledNode);
        builder.context.irradiance.addAssign(E);
    }
}

export function getGiProbeNode() {
    if (!_node) _node = new GiProbeNode();
    return _node;
}

function computeGridResolution(size, targetLongAxis = TARGET_PROBES_LONG_AXIS) {
    const longest = Math.max(size.x, size.y, size.z, 1e-3);
    const spacing = longest / Math.max(1, targetLongAxis);
    const axis = (s) => THREE.MathUtils.clamp(Math.round(s / spacing) + 1, 2, MAX_PROBES_PER_AXIS);
    return new THREE.Vector3(axis(size.x), axis(size.y), axis(size.z));
}

export function createProbeField({ renderer, scene, intensity = 1.0, hysteresis = 0.95, onRebuilt = null, divisions = TARGET_PROBES_LONG_AXIS } = {}) {
    const node = getGiProbeNode();
    node.setIntensity(intensity);
    // Live grid density: probes along the longest axis. setDivisions() updates it and
    // requests a (resize) rebuild; per-axis counts derive from it so cells stay ~cubic.
    let targetLongAxis = THREE.MathUtils.clamp(Math.round(divisions) || TARGET_PROBES_LONG_AXIS, 2, MAX_PROBES_PER_AXIS);
    // Live ray budget per probe (structural — changing it re-sizes the ray scratch buffer and
    // rebuilds the trace/blend kernels, so setRays() requests an idle-gated rebuild). Default
    // 64 keeps the locked 624-probe baseline visually-equivalent.
    let raysPerProbe = RAYS_PER_PROBE_DEFAULT;
    // Normal-bias scale (×) over the auto-computed minCell·SURFACE_NORMAL_BIAS_CELL offset, and the
    // most-recent minCell, so setNormalBias() can rewrite the node uniform INSTANTLY (no rebuild)
    // and the scale survives the next rebuild's auto-bias recompute.
    let normalBiasScale = 1.0;
    let curMinCell = 0.1;

    const gridMin = new THREE.Vector3();
    const gridSize = new THREE.Vector3(1, 1, 1);
    const res = new THREE.Vector3(2, 2, 2);
    let probeTotal = 0;
    let atlasW = 1, atlasH = 1;

    let gpu = null;           // { buffers, kernels, atlases, buffers... }
    let dirty = true;
    // Cached CPU build (BVH soup + material textures). The BVH depends ONLY on geometry,
    // so a divisions/rays change must NOT rebuild it — that ~200ms synchronous MeshBVH +
    // soup-flatten is the one remaining main-thread hitch. buildDirty gates a fresh build:
    // it's set by geometry/light-count/volume changes (true at start), and left FALSE by
    // setDivisions/setRays so those resize the grid/kernels off the cached soup (no hitch).
    let cachedBuilt = null;
    let buildDirty = true;
    let manualVolumes = null; // explicit probe volumes (Probe Origin boxes); null = auto-fit scene
    let needsClassify = true; // one-shot probe classification after a rebuild
    let needsClear = true;    // zero the atlas+depthAtlas ONCE on a FULL rebuild (fresh textures
                              //   aren't guaranteed zeroed); the same-dim path reuses live history,
                              //   so it must NOT clear (clearing would black-flash the field).
    let rebuildBackoff = 0;   // ticks remaining before retrying after a failed/empty rebuild (A7)
    let inFlight = false;
    let disposed = false;
    let probeCursor = 0;
    let frameCounter = 0;
    let updatedPerTick = 1;
    let refreshStarted = false; // B1: false until the first full-field pass begins; gates ray-rotation advance
    let ticksSinceRot = 0;      // B1: ticks since the last ray-set rotation (throttles small-grid rotation)
    // same-dim detection: a geometry edit that doesn't resize the grid reuses the
    // existing atlases/buffers and skips the recompile (A4).
    let prevAtlasW = 0, prevAtlasH = 0, prevProbeTotal = 0;
    let quantStep = 1;        // translation deadband (~quarter cell) for the geo signature (A1)
    let lightQuant = 1;       // scene-relative position deadband for the light signature (B4)
    // reactivity: self-detect live light/geometry edits and re-converge fast.
    let baseHysteresis = THREE.MathUtils.clamp(hysteresis, 0, 0.99);
    let lastLightSig = null;
    let lastGeoSig = null;
    let geoStable = -1;       // -1 = no pending geo change; >=0 = stable-check count since a change (debounce, A1)
    let checkCounter = 0;
    let reactiveTicks = 0;
    const _sigVec = new THREE.Vector3();

    const U = {
        gridMin: uniform(new THREE.Vector3()),
        gridSize: uniform(new THREE.Vector3(1, 1, 1)),
        resX: uniform(2, 'uint'), resY: uniform(2, 'uint'), resZ: uniform(2, 'uint'),
        probeTotal: uniform(1, 'uint'),
        probeOffset: uniform(0, 'uint'),
        updatedCount: uniform(1, 'uint'),
        atlasDim: uniform(new THREE.Vector2(1, 1)),
        lightCount: uniform(0, 'uint'),
        frameJitter: uniform(0.0),
        hysteresis: uniform(THREE.MathUtils.clamp(hysteresis, 0, 0.99)),
        maxDist: uniform(100.0),        // miss-ray depth (probe sees "far") = grid diagonal
        depthSharpness: uniform(50.0),  // cosine power → depth tracks nearest occluder crisply
        radianceClamp: uniform(8.0),    // cap the multibounce feedback term (anti-runaway)
        cellMin: uniform(0.1),          // min grid cell spacing (relocation margin)
        relocClamp: uniform(0.045),     // max relocation offset (< 0.45·cell → probe stays in cell)
        classifyStrength: uniform(0.0), // gates relocation APPLY (mirrors node.classifyStrengthNode)
        filterStrength: uniform(1.0),   // CORE denoise: 0 = filter off (harness baseline), 1 = full intra-tile spatial filter
        filterSmooth: uniform(0.5),     // UI "Smoothness": widens the bilateral edge-stop (0 = baseline detail, 1 = very smooth)
    };

    function isSupported() {
        return renderer?.backend?.isWebGPUBackend === true
            && typeof renderer.computeAsync === 'function'
            && typeof THREE.StorageTexture === 'function'
            && typeof THREE.StorageBufferAttribute === 'function';
    }

    // Same-dim rebuild: free ONLY the previous build's BVH storages, ray scratch, and
    // texture maps — the atlases + irr/depth/state buffers are handed to the new kernels
    // via `reuse`, so they must NOT be disposed (preserves live history, keeps the
    // material binding stable → no recompile).
    function disposeBVHOnly(g) {
        if (!g) return;
        for (const k of ['bvhNodes', 'triIndex', 'vertexData', 'triMaterial', 'materials', 'lights']) g.buffers?.[k]?.dispose?.();
        g.rayBuffer?.dispose?.();
        if (g.maps) for (const t of Object.values(g.maps)) t?.dispose?.();
    }

    function disposeGPU() {
        if (!gpu) return;
        for (const k of ['bvhNodes', 'triIndex', 'vertexData', 'triMaterial', 'materials', 'lights']) gpu.buffers[k]?.dispose?.();
        gpu.irrBuffer?.dispose?.();
        gpu.depthBuffer?.dispose?.();
        gpu.stateBuffer?.dispose?.();
        gpu.rayBuffer?.dispose?.();
        gpu.atlas?.dispose?.();
        gpu.depthAtlas?.dispose?.();
        gpu.stateAtlas?.dispose?.();
        if (gpu.maps) for (const t of Object.values(gpu.maps)) t?.dispose?.();
        gpu = null;
        prevAtlasW = prevAtlasH = prevProbeTotal = 0; // next rebuild takes the full (resize) path
        node.setAtlases(null, null, null);
    }

    // spherical-Fibonacci ray k of N, with a per-frame jitter to decorrelate
    // frames. MUST be reproduced identically in the blend gather.
    function rayDir(kNode, jitterNode) {
        // Cranley-Patterson rotation: stratify the cosine-z by index k, then
        // toroidally shift BOTH z and azimuth by the per-frame jitter. Same 64
        // rays, materially lower variance than the old raw index-shift; keeps the
        // (k, jitter) signature so the blend gather reproduces each ray identically.
        const sk = float(kNode).add(0.5).div(float(raysPerProbe));
        const u = sk.add(jitterNode);
        const uw = u.sub(floor(u));                           // wrap to [0,1) (fract is not imported)
        const z = float(1.0).sub(uw.mul(2.0));
        const r = sqrt(tslMax(float(0.0), float(1.0).sub(z.mul(z))));
        const phi = float(kNode).mul(float(GOLDEN_ANGLE)).add(jitterNode.mul(float(2.0 * Math.PI)));
        return vec3(r.mul(cos(phi)), r.mul(sin(phi)), z);
    }

    // fixed full-sphere Fibonacci ray for classification (NO frame jitter).
    function classifyRayDir(kNode) {
        const z = float(1.0).sub(float(kNode).add(0.5).div(float(CLASSIFY_RAYS)).mul(2.0));
        const r = sqrt(tslMax(float(0.0), float(1.0).sub(z.mul(z))));
        const phi = float(kNode).mul(float(GOLDEN_ANGLE));
        return vec3(r.mul(cos(phi)), r.mul(sin(phi)), z);
    }

    function probeWorldPos(pIndexNode) {
        const ix = pIndexNode.mod(U.resX);
        const iy = pIndexNode.div(U.resX).mod(U.resY);
        const iz = pIndexNode.div(U.resX.mul(U.resY));
        const fx = float(ix).div(tslMax(float(1.0), float(U.resX).sub(1.0)));
        const fy = float(iy).div(tslMax(float(1.0), float(U.resY).sub(1.0)));
        const fz = float(iz).div(tslMax(float(1.0), float(U.resZ).sub(1.0)));
        return vec3(
            U.gridMin.x.add(fx.mul(U.gridSize.x)),
            U.gridMin.y.add(fy.mul(U.gridSize.y)),
            U.gridMin.z.add(fz.mul(U.gridSize.z)),
        );
    }

    function buildKernels(built, reuse = null) {
        const buffers = {
            bvhNodes: new THREE.StorageBufferAttribute(built.bvhNodes, 1),
            triIndex: new THREE.StorageBufferAttribute(built.triIndex, 1),
            vertexData: new THREE.StorageBufferAttribute(built.vertexData, 1),
            triMaterial: new THREE.StorageBufferAttribute(built.triMaterial, 1),
            materials: new THREE.StorageBufferAttribute(built.materials, 1),
            lights: new THREE.StorageBufferAttribute(built.lights, 1),
        };
        const bvhNodes = storage(buffers.bvhNodes, 'uint', buffers.bvhNodes.count).toReadOnly();
        const triIndex = storage(buffers.triIndex, 'uint', buffers.triIndex.count).toReadOnly();
        const vertexData = storage(buffers.vertexData, 'float', buffers.vertexData.count).toReadOnly();
        const triMaterial = storage(buffers.triMaterial, 'uint', buffers.triMaterial.count).toReadOnly();
        const materials = storage(buffers.materials, 'float', buffers.materials.count).toReadOnly();
        const lights = storage(buffers.lights, 'float', buffers.lights.count).toReadOnly();

        const Utrav = { nodeCount: uniform(built.nodeCount >>> 0, 'uint'), envRotation: uniform(0.0), envIntensity: uniform(1.0) };
        const trav = buildTraversal({
            storages: { bvhNodes, triIndex, vertexData, triMaterial, materials },
            U: Utrav, env: null, lut: null, lutRes: 0, maps: built.maps,
        });
        const { fetchVert, fetchNorm, traverseClosest, traverseAny, matFloat, triVert, fetchUV, hitUV, sampleLayer, srgbToLinear, albedoTex, emissiveTex, haveAlbedoMap, haveEmissiveMap } = trav;

        // ray scratch: 4 floats per (probe,ray) = rgb + hitT. itemSize-1 'float'
        // scalar storage — the proven in-repo pattern (gi_irradiance_volume), not
        // the unproven vec4 binding.
        const rayBuffer = new THREE.StorageBufferAttribute(new Float32Array(Math.max(4, updatedCap() * raysPerProbe * 4)), 1);
        const rayData = storage(rayBuffer, 'float', rayBuffer.count);

        // irradiance STATE buffer (read_write): 4 floats per probe texel. Reused on a
        // same-dim rebuild so the field keeps converging from its live history (no black flash).
        const irrBuffer = reuse?.irrBuffer || new THREE.StorageBufferAttribute(new Float32Array(Math.max(4, probeTotal * TILE * TILE * 4)), 1);
        const irr = storage(irrBuffer, 'float', irrBuffer.count);
        const irrRead = storage(irrBuffer, 'float', irrBuffer.count).toReadOnly();

        // write-only sampled atlas (HW bilinear) — uploaded from irrBuffer. Reused
        // verbatim on a same-dim rebuild so the material's binding stays stable
        // (churn-free) and the live irradiance history survives the geometry edit.
        const atlas = reuse?.atlas || (() => {
            const t = new THREE.StorageTexture(atlasW, atlasH);
            t.type = THREE.HalfFloatType; t.format = THREE.RGBAFormat;
            t.minFilter = THREE.LinearFilter; t.magFilter = THREE.LinearFilter;
            t.wrapS = THREE.ClampToEdgeWrapping; t.wrapT = THREE.ClampToEdgeWrapping;
            t.generateMipmaps = false; t.mipmapsAutoUpdate = false;
            return t;
        })();

        // depth-moment STATE (read_write): 2 floats per probe texel (meanR, meanR²),
        // + a sampled depth atlas for the Chebyshev visibility test (leak-free).
        const depthBuffer = reuse?.depthBuffer || new THREE.StorageBufferAttribute(new Float32Array(Math.max(2, probeTotal * TILE * TILE * 2)), 1);
        const depthS = storage(depthBuffer, 'float', depthBuffer.count);
        const depthRead = storage(depthBuffer, 'float', depthBuffer.count).toReadOnly();
        const depthAtlas = reuse?.depthAtlas || (() => {
            const t = new THREE.StorageTexture(atlasW, atlasH);
            t.type = THREE.HalfFloatType; t.format = THREE.RGBAFormat;
            t.minFilter = THREE.LinearFilter; t.magFilter = THREE.LinearFilter;
            t.wrapS = THREE.ClampToEdgeWrapping; t.wrapT = THREE.ClampToEdgeWrapping;
            t.generateMipmaps = false; t.mipmapsAutoUpdate = false;
            return t;
        })();

        // probe META: 4 floats/probe = [state(1=active/0=buried), offset.xyz(relocation)].
        // Sampled (NEAREST, per-probe) by the node; atlas packs R=state, GBA=offset.
        const stateBuffer = reuse?.stateBuffer || new THREE.StorageBufferAttribute(new Float32Array(Math.max(4, probeTotal * 4)), 1);
        const stateS = storage(stateBuffer, 'float', stateBuffer.count);
        const stateRead = storage(stateBuffer, 'float', stateBuffer.count).toReadOnly();
        const stateAtlas = reuse?.stateAtlas || (() => {
            const t = new THREE.StorageTexture(Math.max(1, res.x), Math.max(1, res.y * res.z));
            t.type = THREE.HalfFloatType; t.format = THREE.RGBAFormat;
            t.minFilter = THREE.NearestFilter; t.magFilter = THREE.NearestFilter;
            t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
            t.generateMipmaps = false; t.mipmapsAutoUpdate = false;
            return t;
        })();

        const loadLightVec3 = (base, off) => vec3(lights.element(base.add(uint(off))), lights.element(base.add(uint(off + 1))), lights.element(base.add(uint(off + 2))));

        // sample the (last-frame) atlas irradiance at world (P,N) — for multibounce.
        const sampleAtlas = (P, N) => {
            const cell = vec3(
                U.gridSize.x.div(tslMax(float(1.0), float(U.resX).sub(1.0))),
                U.gridSize.y.div(tslMax(float(1.0), float(U.resY).sub(1.0))),
                U.gridSize.z.div(tslMax(float(1.0), float(U.resZ).sub(1.0))),
            );
            const gridF = P.sub(U.gridMin).div(cell.max(vec3(1e-6)));
            const resV = vec3(float(U.resX), float(U.resY), float(U.resZ));
            const baseF = gridF.floor().clamp(vec3(0.0), resV.sub(2.0).max(vec3(0.0)));
            const frac = gridF.sub(baseF).clamp(0.0, 1.0);
            const octUV = octEncodeNode(N.normalize(), TSL);
            const acc = vec3(0.0).toVar();
            const wsum = float(0.0).toVar();
            for (let i = 0; i < 8; i++) {
                const dx = i & 1, dy = (i >> 1) & 1, dz = (i >> 2) & 1;
                const px = baseF.x.add(float(dx));
                const py = baseF.y.add(float(dy));
                const pz = baseF.z.add(float(dz));
                const wx = dx ? frac.x : float(1.0).sub(frac.x);
                const wy = dy ? frac.y : float(1.0).sub(frac.y);
                const wz = dz ? frac.z : float(1.0).sub(frac.z);
                const w = wx.mul(wy).mul(wz).add(1e-4);
                const col = px;
                const row = pz.mul(float(U.resY)).add(py);
                const ox = col.mul(float(TILE)).add(float(BORDER)).add(octUV.x.mul(float(OCT_RES))).add(0.5);
                const oy = row.mul(float(TILE)).add(float(BORDER)).add(octUV.y.mul(float(OCT_RES))).add(0.5);
                const uv = vec2(ox.div(U.atlasDim.x), oy.div(U.atlasDim.y));
                acc.addAssign(texture(atlas, uv).xyz.mul(w));
                wsum.addAssign(w);
            }
            return acc.div(wsum.max(float(1e-4)));
        };

        // ── TRACE: one thread per (updated probe, ray). RGB shade; miss=BLACK ──
        const traceKernel = Fn(() => {
            const gid = instanceIndex.toVar();
            const slot = gid.div(uint(raysPerProbe)).toVar();
            If(slot.greaterThanEqual(U.updatedCount), () => { Return(); });
            const k = gid.mod(uint(raysPerProbe)).toVar();
            const probeIndex = U.probeOffset.add(slot).mod(U.probeTotal).toVar();
            const ro = probeWorldPos(probeIndex).toVar();
            // apply relocation offset (gated by classifyStrength; 0 = no relocation).
            const mbT = probeIndex.mul(uint(4));
            ro.addAssign(vec3(stateRead.element(mbT.add(uint(1))), stateRead.element(mbT.add(uint(2))), stateRead.element(mbT.add(uint(3)))).mul(U.classifyStrength));
            const rd = normalize(rayDir(k, U.frameJitter)).toVar();

            const outRgb = vec3(0.0).toVar();
            const hitT = float(-1.0).toVar();
            const bestT = float(T_MAX).toVar();
            const bestTri = int(-1).toVar();
            traverseClosest(ro, rd, bestT, bestTri);

            If(bestTri.greaterThanEqual(int(0)), () => {
                hitT.assign(bestT);
                const triId = uint(bestTri);
                const matId = triMaterial.element(triId);
                const p0 = fetchVert(triVert(triId, 0));
                const p1 = fetchVert(triVert(triId, 1));
                const p2 = fetchVert(triVert(triId, 2));
                const ngRaw = normalize(cross(p1.sub(p0), p2.sub(p0)));
                const faceFwd = dot(ngRaw, rd).lessThan(float(0.0));
                const ng = ngRaw.mul(select(faceFwd, float(1.0), float(-1.0))).toVar();
                const hitPoint = ro.add(rd.mul(bestT));
                const traceBias = tslMax(U.cellMin.mul(float(TRACE_SURFACE_BIAS_CELL)), float(RAY_EPS));
                const hitPos = hitPoint.add(ng.mul(traceBias)).toVar();

                // Recover hit UV via Möller–Trumbore barycentrics (traverseClosest
                // discards them). Mirror the traversal's op order/float types so the
                // UVs are numerically identical. A real hit guarantees |det|>0.
                const e1 = p1.sub(p0);
                const e2 = p2.sub(p0);
                const pv = cross(rd, e2);
                const det = dot(e1, pv);
                const invDet = float(1.0).div(det);
                const tv = ro.sub(p0);
                const ub = dot(tv, pv).mul(invDet);
                const qv = cross(tv, e1);
                const vb = dot(rd, qv).mul(invDet);
                const uv0 = hitUV(triId, ub, vb);
                const uv = uv0.mul(vec2(matFloat(matId, 18), matFloat(matId, 19))).add(vec2(matFloat(matId, 20), matFloat(matId, 21)));

                const baseColor = vec3(matFloat(matId, 0), matFloat(matId, 1), matFloat(matId, 2)).toVar();
                if (GI_SAMPLE_TEXTURES && haveAlbedoMap) {
                    const aL = matFloat(matId, 12);
                    const texRGB = srgbToLinear(sampleLayer(albedoTex, uv, aL));
                    // factor × texture; layer −1 (no map) falls back to the packed factor.
                    baseColor.assign(select(aL.greaterThan(float(-0.5)), baseColor.mul(texRGB), baseColor));
                }
                const emissive = vec3(matFloat(matId, 7), matFloat(matId, 8), matFloat(matId, 9)).toVar();
                if (GI_SAMPLE_TEXTURES && haveEmissiveMap) {
                    const eL = matFloat(matId, 16);
                    const eTex = srgbToLinear(sampleLayer(emissiveTex, uv, eL));
                    emissive.assign(select(eL.greaterThan(float(-0.5)), emissive.mul(eTex), emissive));
                }
                const radiance = emissive.toVar();

                // energy-weighted diffuse albedo: metals (slot 4) and glass (slot 5)
                // don't bounce Lambert diffuse. Cap ≤0.95 so the temporal multibounce
                // series E = direct/(1−kd) is provably convergent (no runaway).
                const metal = matFloat(matId, 4);
                const transm = matFloat(matId, 5);
                const kd = clamp(baseColor.mul(float(1.0).sub(metal)).mul(float(1.0).sub(transm)), vec3(0.0), vec3(0.95)).toVar();

                // NEE over ALL lights (count small; loop avoids sampling noise).
                Loop({ start: uint(0), end: U.lightCount, type: 'uint', condition: '<' }, ({ i: li }) => {
                    const lb = li.mul(uint(16)).toVar();
                    const ltype = lights.element(lb);
                    const lpos = loadLightVec3(lb, 1);
                    const ldir = loadLightVec3(lb, 4);
                    const lcol = loadLightVec3(lb, 7);
                    const lrange = lights.element(lb.add(uint(10)));
                    const ldecay = lights.element(lb.add(uint(11)));
                    const lcosAngle = lights.element(lb.add(uint(12)));
                    const lcosPen = lights.element(lb.add(uint(13)));
                    const isDir = ltype.lessThan(float(0.5));
                    const isSpot = float(ltype.sub(float(2.0)).abs()).lessThan(float(0.5));
                    const toLight = select(isDir, ldir.mul(-1.0), lpos.sub(hitPos));
                    const dist = select(isDir, float(1e4), tslMax(length(toLight), float(1e-4)));
                    const wi = normalize(toLight);
                    const ndl = tslMax(dot(ng, wi), float(0.0));
                    If(ndl.greaterThan(float(0.0)), () => {
                        const blocked = traverseAny(hitPos, wi, dist.sub(traceBias));
                        If(blocked.lessThan(float(0.5)), () => {
                            const falloff = float(1.0).div(tslMax(pow(dist, ldecay), float(0.01)));
                            const rr = dist.div(tslMax(lrange, float(1e-4)));
                            const rr2 = rr.mul(rr);
                            const win = clamp(float(1.0).sub(rr2.mul(rr2)), float(0.0), float(1.0));
                            const ranged = falloff.mul(win.mul(win));
                            const posAtten = select(lrange.greaterThan(float(0.0)), ranged, falloff);
                            const distAtten = select(isDir, float(1.0), posAtten);
                            const angleCos = dot(wi, ldir).mul(-1.0);
                            const spotAtten = clamp(angleCos.sub(lcosAngle).div(tslMax(lcosPen.sub(lcosAngle), float(1e-4))), float(0.0), float(1.0));
                            const atten = distAtten.mul(select(isSpot, spotAtten, float(1.0)));
                            const diffuse = kd.mul(float(1.0).div(float(PI)));
                            radiance.addAssign(diffuse.mul(ndl).mul(lcol).mul(atten));
                        });
                    });
                });

                // multibounce: add last frame's irradiance × albedo (the atlas is
                // re-uploaded every tick, so this reads a valid prior field; on
                // tick 1 it reads zero, which is correct). Lambert: E·albedo/π.
                const bounce = sampleAtlas(hitPos, ng).mul(kd).mul(float(1.0).div(float(PI)));
                // hue-stable firefly rolloff (vs the old per-channel hard clamp that
                // clipped saturated bounce hues): scale by luminance, preserving chroma.
                const bl = dot(bounce, vec3(0.2126, 0.7152, 0.0722));
                const roll = U.radianceClamp.div(tslMax(U.radianceClamp, bl));
                radiance.addAssign(bounce.mul(roll));
                outRgb.assign(radiance);
            });
            // miss → outRgb stays BLACK (CRITICAL: never sample sky here — IBL
            // owns the sky hemisphere; sampling it would double-count IBL).

            const rb = slot.mul(uint(raysPerProbe)).add(k).mul(uint(4)).toVar();
            rayData.element(rb).assign(outRgb.x);
            rayData.element(rb.add(uint(1))).assign(outRgb.y);
            rayData.element(rb.add(uint(2))).assign(outRgb.z);
            rayData.element(rb.add(uint(3))).assign(hitT);
        })().compute(updatedCap() * raysPerProbe);

        // ── BLEND: one thread per (updated probe, atlas texel). Cosine-gather
        // the probe's rays for this texel's octahedral direction; hysteresis. ──
        const blendKernel = Fn(() => {
            const gid = instanceIndex.toVar();
            const slot = gid.div(uint(TILE * TILE)).toVar();
            If(slot.greaterThanEqual(U.updatedCount), () => { Return(); });
            const local = gid.mod(uint(TILE * TILE)).toVar();
            const lx = local.mod(uint(TILE)).toVar();
            const ly = local.div(uint(TILE)).toVar();
            const probeIndex = U.probeOffset.add(slot).mod(U.probeTotal).toVar();

            // texel direction (border texels get slightly-out-of-range uv → a
            // natural octahedral gutter; continuous under HW bilinear for MVP).
            const u = float(lx).sub(float(BORDER)).add(0.5).div(float(OCT_RES));
            const v = float(ly).sub(float(BORDER)).add(0.5).div(float(OCT_RES));
            const dir = octDecodeNode(vec2(u, v), TSL).toVar();

            const acc = vec3(0.0).toVar();
            const wsum = float(0.0).toVar();
            const dAcc = float(0.0).toVar();   // Σ w·dist  (sharp cosine weight)
            const dAcc2 = float(0.0).toVar();  // Σ w·dist²
            const dwsum = float(0.0).toVar();
            Loop({ start: uint(0), end: uint(raysPerProbe), type: 'uint', condition: '<' }, ({ i: k }) => {
                const rb = slot.mul(uint(raysPerProbe)).add(k).mul(uint(4));
                const rrgb = vec3(rayData.element(rb), rayData.element(rb.add(uint(1))), rayData.element(rb.add(uint(2))));
                const hitT = rayData.element(rb.add(uint(3)));
                const rdir = normalize(rayDir(k, U.frameJitter));
                const cw = tslMax(dot(dir, rdir), float(0.0));
                acc.addAssign(rrgb.mul(cw));
                wsum.addAssign(cw);
                // depth moments: miss → "far" so the probe stays visible that way.
                const rdist = select(hitT.lessThan(float(0.0)), U.maxDist, hitT);
                const dw = pow(cw, U.depthSharpness);
                dAcc.addAssign(rdist.mul(dw));
                dAcc2.addAssign(rdist.mul(rdist).mul(dw));
                dwsum.addAssign(dw);
            });
            const meanRad = acc.div(wsum.max(float(1e-4)));
            const meanR = dAcc.div(dwsum.max(float(1e-4)));
            const meanR2 = dAcc2.div(dwsum.max(float(1e-4)));

            // irradiance: read prev + write blended through ONE read_write 'float'
            // binding (proven surfel-grid pattern).
            const ib = probeIndex.mul(uint(TILE * TILE)).add(local).mul(uint(4)).toVar();
            const prev = vec3(irr.element(ib), irr.element(ib.add(uint(1))), irr.element(ib.add(uint(2))));
            const wasBlack = dot(prev, vec3(1.0)).lessThan(float(1e-6));
            // Stable temporal accumulation: HALO intentionally jitters rays every
            // tick, so luminance disagreement is expected on steady walls. Treating
            // that as disocclusion collapses history and causes visible flicker.
            // Real light/geometry edits are handled by temporarily lowering
            // U.hysteresis from tick(), not by per-texel history rejection here.
            const LUMA = vec3(0.2126, 0.7152, 0.0722);
            const curL = dot(meanRad, LUMA);
            const prevM2 = irr.element(ib.add(uint(3)));
            const h = select(wasBlack, float(0.0), U.hysteresis);
            const blended = mix(meanRad, prev, h);
            irr.element(ib).assign(blended.x);
            irr.element(ib.add(uint(1))).assign(blended.y);
            irr.element(ib.add(uint(2))).assign(blended.z);
            // luminance 2nd moment E[L²] in the FREE 4th slot (buffer-only; the upload
            // keeps atlas.w=1.0 so the fragment sampler never sees it). luma is linear
            // → E[luma]=luma(E[rgb]), so variance = max(0, M2 − luma(rgb)²) anywhere.
            const m2 = mix(curL.mul(curL), prevM2, h);
            irr.element(ib.add(uint(3))).assign(m2);

            // depth moments: same hysteresis; fill instantly when unseeded.
            const db = probeIndex.mul(uint(TILE * TILE)).add(local).mul(uint(2)).toVar();
            const dprev = vec2(depthS.element(db), depthS.element(db.add(uint(1))));
            const dWasZero = dprev.x.lessThan(float(1e-6));
            const dh = select(dWasZero, float(0.0), U.hysteresis);
            const dblended = mix(vec2(meanR, meanR2), dprev, dh);
            depthS.element(db).assign(dblended.x);
            depthS.element(db.add(uint(1))).assign(dblended.y);
        })().compute(updatedCap() * TILE * TILE);

        // ── CLEAR: new StorageTextures are not assumed zeroed. Do this once per
        // rebuild before the round-robin batch uploads start populating live probes.
        const clearAtlasKernel = Fn(() => {
            const gid = instanceIndex.toVar();
            const total = uint(probeTotal * TILE * TILE);
            If(gid.greaterThanEqual(total), () => { Return(); });
            const probeIndex = gid.div(uint(TILE * TILE)).toVar();
            const local = gid.mod(uint(TILE * TILE)).toVar();
            const lx = local.mod(uint(TILE)).toVar();
            const ly = local.div(uint(TILE)).toVar();
            const col = probeIndex.mod(U.resX);
            const row = probeIndex.div(U.resX.mul(U.resY)).mul(U.resY).add(probeIndex.div(U.resX).mod(U.resY));
            const tx = col.mul(uint(TILE)).add(lx);
            const ty = row.mul(uint(TILE)).add(ly);
            textureStore(atlas, uvec2(tx, ty), vec4(0.0, 0.0, 0.0, 1.0)).toWriteOnly();
            textureStore(depthAtlas, uvec2(tx, ty), vec4(0.0, 0.0, 0.0, 1.0)).toWriteOnly();
        })().compute(probeTotal * TILE * TILE);

        // ── UPLOAD: copy the updated probes into the atlas StorageTexture. Unchanged
        // probes keep their previous atlas texels; uploading the full field every tick
        // caused avoidable WebGPU stalls on larger rooms.
        const uploadKernel = Fn(() => {
            const gid = instanceIndex.toVar();
            const slot = gid.div(uint(TILE * TILE)).toVar();
            If(slot.greaterThanEqual(U.updatedCount), () => { Return(); });
            const probeIndex = U.probeOffset.add(slot).mod(U.probeTotal).toVar();
            const local = gid.mod(uint(TILE * TILE)).toVar();
            const lx = local.mod(uint(TILE)).toVar();
            const ly = local.div(uint(TILE)).toVar();
            const col = probeIndex.mod(U.resX);
            const row = probeIndex.div(U.resX.mul(U.resY)).mul(U.resY).add(probeIndex.div(U.resX).mod(U.resY));
            const tx = col.mul(uint(TILE)).add(lx);
            const ty = row.mul(uint(TILE)).add(ly);
            // ── intra-tile variance/edge-stopped spatial filter (CORE splotch killer).
            // Reads the read-only irrBuffer and writes the write-only atlas, so the
            // denoised result never feeds back into history (no over-blur, no RAW
            // hazard). Interior texels only; the 1px octahedral gutter is copied raw.
            // Taps stay inside THIS probe's tile → cannot mix radiance across a wall.
            const LUMA = vec3(0.2126, 0.7152, 0.0722);
            const probeBase = probeIndex.mul(uint(TILE * TILE)).toVar();
            const probeTexel = probeBase.add(local).toVar();
            const ib = probeTexel.mul(uint(4));
            const eC = vec3(irrRead.element(ib), irrRead.element(ib.add(uint(1))), irrRead.element(ib.add(uint(2)))).toVar();
            const lumaC = dot(eC, LUMA);
            const varC = tslMax(irrRead.element(ib.add(uint(3))).sub(lumaC.mul(lumaC)), float(0.0));
            const lxI = int(lx); const lyI = int(ly);
            const facc = vec3(0.0).toVar();
            const fwsum = float(0.0).toVar();
            // UI "Smoothness" (U.filterSmooth) widens the variance-adaptive edge stop so more
            // neighbours are trusted → stronger blur that kills GI splotch. 0 = baseline
            // bandwidth (original directional detail), 1 = ~7× wider (very smooth).
            const smW = float(1.0).add(U.filterSmooth.mul(float(6.0)));
            const kEff = float(GI_FILTER_K).mul(smW);
            const relEff = float(GI_FILTER_REL).mul(smW);
            for (let jy = -1; jy <= 1; jy++) {
                for (let jx = -1; jx <= 1; jx++) {
                    const gw = Math.exp(-(jx * jx + jy * jy) * 0.5); // separable 3×3 gaussian (JS const)
                    const nx = lxI.add(int(jx)).clamp(int(BORDER), int(BORDER + OCT_RES - 1)).toUint();
                    const ny = lyI.add(int(jy)).clamp(int(BORDER), int(BORDER + OCT_RES - 1)).toUint();
                    const nIb = probeBase.add(ny.mul(uint(TILE))).add(nx).mul(uint(4));
                    const en = vec3(irrRead.element(nIb), irrRead.element(nIb.add(uint(1))), irrRead.element(nIb.add(uint(2))));
                    const dLum = dot(en, LUMA).sub(lumaC);
                    // variance-adaptive edge stop: noisy texel (high var) → wide trust →
                    // blur; converged texel (var→0) → narrow → preserve directional detail.
                    const es = exp(dLum.mul(dLum).div(varC.mul(kEff).add(tslMax(float(GI_FILTER_EPS), lumaC.mul(lumaC).mul(relEff))).max(float(1e-8))).mul(-1.0));
                    const w = float(gw).mul(es);
                    facc.addAssign(en.mul(w));
                    fwsum.addAssign(w);
                }
            }
            const filtered = facc.div(fwsum.max(float(1e-4)));
            const isInterior = lx.greaterThanEqual(uint(BORDER)).and(lx.lessThanEqual(uint(BORDER + OCT_RES - 1)))
                .and(ly.greaterThanEqual(uint(BORDER))).and(ly.lessThanEqual(uint(BORDER + OCT_RES - 1)));
            const outE = select(isInterior, mix(eC, filtered, U.filterStrength), eC);
            textureStore(atlas, uvec2(tx, ty), vec4(outE, float(1.0))).toWriteOnly();
            const db = probeTexel.mul(uint(2));
            const dd = vec2(depthRead.element(db), depthRead.element(db.add(uint(1))));
            textureStore(depthAtlas, uvec2(tx, ty), vec4(dd.x, dd.y, float(0.0), float(1.0))).toWriteOnly();
        })().compute(updatedCap() * TILE * TILE);

        // ── CLASSIFY: one thread per probe. Fixed full-sphere rays; if too many
        // hit BACKFACES the probe is buried in geometry → mark INACTIVE. ──
        const classifyKernel = Fn(() => {
            const p = instanceIndex.toVar();
            If(p.greaterThanEqual(U.probeTotal), () => { Return(); });
            const ro = probeWorldPos(p).toVar();
            const back = float(0.0).toVar();
            const hits = float(0.0).toVar();
            const closeBackDist = float(1e30).toVar();
            const closeBackDir = vec3(0.0).toVar();
            const closeFrontDist = float(1e30).toVar();
            Loop({ start: uint(0), end: uint(CLASSIFY_RAYS), type: 'uint', condition: '<' }, ({ i: k }) => {
                const rd = normalize(classifyRayDir(k)).toVar();
                const bestT = float(T_MAX).toVar();
                const bestTri = int(-1).toVar();
                traverseClosest(ro, rd, bestT, bestTri);
                If(bestTri.greaterThanEqual(int(0)), () => {
                    hits.addAssign(float(1.0));
                    const triId = uint(bestTri);
                    const p0 = fetchVert(triVert(triId, 0));
                    const p1 = fetchVert(triVert(triId, 1));
                    const p2 = fetchVert(triVert(triId, 2));
                    const ng = normalize(cross(p1.sub(p0), p2.sub(p0)));
                    If(dot(rd, ng).greaterThan(float(0.0)), () => { // backface → probe is behind this surface
                        back.addAssign(float(1.0));
                        If(bestT.lessThan(closeBackDist), () => { closeBackDist.assign(bestT); closeBackDir.assign(rd); });
                    }).Else(() => {
                        If(bestT.lessThan(closeFrontDist), () => { closeFrontDist.assign(bestT); });
                    });
                });
            });
            const frac = back.div(tslMax(hits, float(1.0)));
            const state = select(frac.greaterThan(float(BACKFACE_FRACTION)), float(0.0), float(1.0)).toVar();

            // RELOCATION: if the probe is behind a surface (closest hit is a backface),
            // push ALONG that ray past the surface into valid space. Clamp < relocClamp
            // so the probe never leaves its own cell. (Applied only when classifyStrength>0.)
            const off = vec3(0.0).toVar();
            If(back.greaterThan(float(0.5)).and(closeBackDist.lessThan(closeFrontDist)), () => {
                const step = closeBackDist.add(float(0.5).mul(tslMin(closeFrontDist, U.cellMin)));
                const raw = closeBackDir.mul(step);
                const len = length(raw);
                off.assign(raw.mul(tslMin(len, U.relocClamp).div(tslMax(len, float(1e-6)))));
            });

            const mb = p.mul(uint(4)).toVar();
            stateS.element(mb).assign(state);
            stateS.element(mb.add(uint(1))).assign(off.x);
            stateS.element(mb.add(uint(2))).assign(off.y);
            stateS.element(mb.add(uint(3))).assign(off.z);
        })().compute(probeTotal);

        // upload per-probe meta → atlas (1 texel/probe): R=state, GBA=relocation offset.
        const uploadStateKernel = Fn(() => {
            const p = instanceIndex.toVar();
            If(p.greaterThanEqual(uint(probeTotal)), () => { Return(); });
            const col = p.mod(U.resX);
            const row = p.div(U.resX.mul(U.resY)).mul(U.resY).add(p.div(U.resX).mod(U.resY));
            const mb = p.mul(uint(4));
            textureStore(stateAtlas, uvec2(col, row), vec4(
                stateRead.element(mb), stateRead.element(mb.add(uint(1))),
                stateRead.element(mb.add(uint(2))), stateRead.element(mb.add(uint(3))),
            )).toWriteOnly();
        })().compute(probeTotal);

        return { buffers, traceKernel, blendKernel, uploadKernel, clearAtlasKernel, classifyKernel, uploadStateKernel, atlas, depthAtlas, stateAtlas, irrBuffer, depthBuffer, stateBuffer, rayBuffer, maps: built.maps, lightCount: built.lightCount };
    }

    function updatedCap() {
        // Round-robin: update ~1/4 of small fields, but cap the batch so large
        // rooms do not hitch the viewport with a huge compute burst.
        return Math.max(1, Math.min(MAX_PROBES_PER_TICK, Math.ceil(Math.max(1, probeTotal) / 4)));
    }

    async function ensureSceneBuilder() {
        if (!_buildSpectralScene) {
            const mod = await import('./spectral_scene.js');
            _buildSpectralScene = mod.buildSpectralScene;
            _collectLights = mod.collectLights || null;
            if (Number.isFinite(mod.LIGHT_STRIDE)) _LIGHT_STRIDE = mod.LIGHT_STRIDE;
        }
        return _buildSpectralScene;
    }

    // Auto-fit the grid to the TRACED geometry only — honour the SAME visibility gate
    // the BVH uses (spectral_scene.objectIsRenderable: userData.maxjsVisible !== false).
    // Box3.setFromObject(scene) would include the sky dome — a unit box scaled to
    // ~camera.far (≈15× the model) — plus probe-helper/gizmo meshes, ballooning the grid
    // so probes land in open air and the injected irradiance washes out to ~nothing.
    // The maxjsVisible flag keeps the sky out of the BVH but NOT out of setFromObject, so
    // the fit has to re-apply it here. Mirrors the per-mesh world-AABB union the soup bakes.
    const _fitBox = new THREE.Box3();
    function autoFitTracedBounds(out) {
        out.makeEmpty();
        scene.updateMatrixWorld(true);
        scene.traverseVisible((o) => {
            if (o.userData?.maxjsVisible === false) return;     // sky / probe helpers / gizmos
            if (o.isInstancedMesh) {                            // instance-aware AABB (per-instance matrices)
                if (!o.boundingBox) o.computeBoundingBox();
                if (o.boundingBox) out.union(_fitBox.copy(o.boundingBox).applyMatrix4(o.matrixWorld));
                return;
            }
            if (!o.isMesh || !o.geometry) return;
            const g = o.geometry;
            if (!g.boundingBox) g.computeBoundingBox();
            if (g.boundingBox) out.union(_fitBox.copy(g.boundingBox).applyMatrix4(o.matrixWorld));
        });
    }

    async function rebuild() {
        // Reuse the cached BVH+texture soup unless geometry actually changed. A
        // divisions/rays change (buildDirty=false) skips the synchronous MeshBVH build
        // entirely → no main-thread hitch; it only resizes the grid/atlas/kernels below.
        let built = cachedBuilt;
        if (!built || buildDirty) {
            const buildSpectralScene = await ensureSceneBuilder();
            built = await buildSpectralScene({ THREE, scene, maxTriangles: MAX_TRIANGLES });
            // On a failed/empty build, keep the EXISTING field as history (don't tear it
            // down) — tick() arms a backoff so we don't re-enter the synchronous build
            // every frame. (A7)
            if (!built || built.error) return false;
            cachedBuilt = built;
            buildDirty = false;
        }

        const box = new THREE.Box3();
        scene.updateMatrixWorld(true);
        // Explicit "Probe Origin" volumes override the whole-scene auto-fit: fewer
        // probes, denser where it matters, faster. Multiple boxes are unioned into
        // one grid for now (perf win = tighter than the whole scene); far-apart
        // boxes will later get separate grids. Auto-fit only when no volume is set.
        const hasVolumes = Array.isArray(manualVolumes) && manualVolumes.length > 0;
        if (hasVolumes) { box.makeEmpty(); for (const v of manualVolumes) if (v.box && !v.box.isEmpty()) box.union(v.box); }
        else if (built.bounds && !built.bounds.isEmpty()) box.copy(built.bounds);  // exact traced-soup AABB (preferred)
        else autoFitTracedBounds(box);  // fallback: scene walk honouring the BVH visibility gate
        if (box.isEmpty()) return false;
        box.getSize(gridSize);
        gridMin.copy(box.min);
        if (!hasVolumes) {
            // pad the AUTO-FIT grid so surfaces sit inside the cage. Explicit boxes
            // are used as drawn (the author owns their extent).
            const pad = gridSize.clone().multiplyScalar(0.06);
            gridMin.sub(pad); gridSize.add(pad.clone().multiplyScalar(2));
        }
        // Manual divisions: a single explicit volume with a res override sets grid
        // resolution directly; otherwise derive from box size.
        const resOverride = (hasVolumes && manualVolumes.length === 1 && manualVolumes[0].res) ? manualVolumes[0].res : null;
        res.copy(resOverride ? resOverride : computeGridResolution(gridSize, targetLongAxis));
        // Keep the octahedral atlas within the GPU's 2D texture limit. atlasH = resY·resZ·TILE
        // is the tall axis, so a dense grid on a cubic scene can blow past 8192. Shrink res
        // uniformly until both atlas dims fit (bounded loop; each step trims ~15%).
        const maxDim = renderer?.backend?.device?.limits?.maxTextureDimension2D || ATLAS_DIM_FALLBACK;
        for (let g = 0; g < 12 && (res.x * TILE > maxDim || res.y * res.z * TILE > maxDim); g++) {
            res.set(Math.max(2, Math.floor(res.x * 0.85)), Math.max(2, Math.floor(res.y * 0.85)), Math.max(2, Math.floor(res.z * 0.85)));
        }
        probeTotal = res.x * res.y * res.z;
        atlasW = res.x * TILE;
        atlasH = res.y * res.z * TILE;

        const minCell = Math.min(gridSize.x / Math.max(1, res.x - 1), gridSize.y / Math.max(1, res.y - 1), gridSize.z / Math.max(1, res.z - 1));
        curMinCell = Math.max(1e-4, minCell);                   // remembered so setNormalBias() can rewrite the node uniform live
        quantStep = Math.max(1e-4, minCell * 0.25);             // A1: geo-signature translation deadband (~¼ cell)
        lightQuant = Math.max(1e-3, gridSize.length() * 0.003); // B4: light-signature position deadband (~0.3% of scene)

        // (A4) Same-dim path: a geometry/light edit that does NOT resize the grid reuses
        // the live atlases + irr/depth/state buffers, rebuilds ONLY the BVH-bound kernels,
        // and updates grid uniforms WITHOUT bumping the cache token → NO full-scene TSL
        // recompile (the per-rebuild half of the freeze) and NO black flash. The reused
        // history re-converges to the new geometry over a reactivity burst.
        const sameDim = !!gpu && atlasW === prevAtlasW && atlasH === prevAtlasH && probeTotal === prevProbeTotal;
        if (sameDim) {
            const prev = gpu;
            const reuse = {
                atlas: prev.atlas, depthAtlas: prev.depthAtlas, stateAtlas: prev.stateAtlas,
                irrBuffer: prev.irrBuffer, depthBuffer: prev.depthBuffer, stateBuffer: prev.stateBuffer,
            };
            gpu = buildKernels(built, reuse);
            disposeBVHOnly(prev);   // free ONLY the old BVH storages + ray scratch + maps
            U.gridMin.value.copy(gridMin);
            U.gridSize.value.copy(gridSize);
            U.lightCount.value = Math.min(MAX_LIGHTS, gpu.lightCount) >>> 0;
            U.maxDist.value = gridSize.length();
            U.cellMin.value = Math.max(1e-4, minCell);
            U.relocClamp.value = 0.45 * minCell;
            // res / probeTotal / atlasDim are unchanged by definition → leave those uniforms.
            // Churn-free: update the NODE's placement uniforms WITHOUT bumping _structGen.
            node.updateGridUniforms(gridMin, gridSize, res, atlasW, atlasH, minCell * SURFACE_NORMAL_BIAS_CELL * normalBiasScale, minCell * GI_CHEBY_BIAS_CELL);
            probeCursor = 0;
            refreshStarted = false;
            needsClear = false;             // reuse the live atlas history (no black flash)
            needsClassify = true;           // refresh per-probe state for the new geometry
            reactiveTicks = REACTIVE_TICKS; // re-converge the reused history to the edit
            dirty = false;
            return true;
        }

        // Resize / first-enable path: full teardown + fresh resources + exactly ONE recompile.
        disposeGPU();
        gpu = buildKernels(built);

        U.gridMin.value.copy(gridMin);
        U.gridSize.value.copy(gridSize);
        U.resX.value = res.x >>> 0; U.resY.value = res.y >>> 0; U.resZ.value = res.z >>> 0;
        U.probeTotal.value = probeTotal >>> 0;
        U.atlasDim.value.set(atlasW, atlasH);
        U.lightCount.value = Math.min(MAX_LIGHTS, gpu.lightCount) >>> 0;
        U.maxDist.value = gridSize.length();
        U.cellMin.value = Math.max(1e-4, minCell);
        U.relocClamp.value = 0.45 * minCell;
        node.setAtlases(gpu.atlas, gpu.depthAtlas, gpu.stateAtlas);
        node.setGrid(gridMin, gridSize, res, atlasW, atlasH, minCell * SURFACE_NORMAL_BIAS_CELL * normalBiasScale, minCell * GI_CHEBY_BIAS_CELL);
        prevAtlasW = atlasW; prevAtlasH = atlasH; prevProbeTotal = probeTotal;
        probeCursor = 0;
        refreshStarted = false;
        dirty = false;
        needsClear = true;      // fresh StorageTextures aren't guaranteed zeroed
        needsClassify = true;
        // The setGrid/setAtlases calls above bumped _structGen → the lights-node
        // cacheToken changed. Force the one-shot material recompile NOW, the frame
        // the probe data first exists, so the field folds into the lights graph
        // immediately on enable / after a resize — not a rebuild late. Fires only on
        // this (resize/first-enable) path, never on the same-dim path or per tick →
        // cannot reintroduce recompile churn.
        if (typeof onRebuilt === 'function') { try { onRebuilt(); } catch (e) { /* non-fatal */ } }
        return true;
    }

    async function tick(opts = {}) {
        if (disposed || inFlight || !node._enabled || !isSupported()) return;

        // (A2) Idle gate — the ONE hard rule. The synchronous CPU BVH rebuild AND the
        // GPU solve are held while the user orbits, the timeline plays, or a delta-sync
        // burst is in flight. GI is world-space, so the field staying static during
        // motion is visually lossless; work resumes the moment the view rests, so a
        // freeze can never land during interaction.
        const idleMs = Number.isFinite(opts.idleMs) ? opts.idleMs : Infinity;
        const playing = opts.playing === true;
        if (idleMs < GI_IDLE_MS || playing) return;

        // (A7) Back off after a failed/empty rebuild instead of re-entering the
        // synchronous build every tick.
        if (rebuildBackoff > 0) rebuildBackoff--;

        if (dirty || !gpu) {
            if (rebuildBackoff > 0) return;
            inFlight = true; let ok = false;
            try { ok = await rebuild(); } finally { inFlight = false; }
            if (!ok) { dirty = false; rebuildBackoff = REBUILD_BACKOFF_TICKS; return; }
        }
        if (!gpu) return;

        // reactivity: detect live light/geometry edits (throttled). Light change →
        // cheap in-place buffer refresh; geometry change → debounced rebuild.
        checkCounter++;
        if (checkCounter % LIGHT_CHECK_INTERVAL === 0) {
            const ls = lightSignature();
            if (lastLightSig !== null && ls !== lastLightSig) refreshLights();
            lastLightSig = ls;
        }
        if (checkCounter % GEO_CHECK_INTERVAL === 0) {
            const gs = geoSignature();
            // (A1) Debounce: rebuild only after the geometry has been STABLE for
            // GEO_SETTLE_INTERVALS consecutive checks, so a continuous drag never
            // thrashes. geoStable: -1 = no pending change; >=0 = stable-checks counted.
            if (lastGeoSig === null) lastGeoSig = gs;
            else if (gs !== lastGeoSig) { lastGeoSig = gs; geoStable = 0; }
            else if (geoStable >= 0) {
                geoStable++;
                if (geoStable >= GEO_SETTLE_INTERVALS) { geoStable = -1; reactiveTicks = REACTIVE_TICKS; requestRebuild(); }
            }
        }
        // Reactive re-converge: EASE the temporal blend from a faster (lower-hysteresis) start
        // UP to the steady-state hysteresis across the burst, so a light/geometry edit FADES
        // smoothly into its new solution over a couple seconds — instead of snapping or boiling.
        // (Was a flat REACTIVE_HYSTERESIS for the whole burst → big per-tick jumps = the
        // "calculating buncha shit, flickers left/right" pop the user reported.)
        if (reactiveTicks > 0) {
            const t = 1 - (reactiveTicks / REACTIVE_TICKS); // 0 at burst start → 1 at burst end
            U.hysteresis.value = REACTIVE_HYSTERESIS + (baseHysteresis - REACTIVE_HYSTERESIS) * t;
            reactiveTicks--;
        } else {
            U.hysteresis.value = baseHysteresis;
        }

        const updated = Math.min(updatedCap(), probeTotal);
        U.probeOffset.value = probeCursor >>> 0;
        U.updatedCount.value = updated >>> 0;
        // (B1) Advance the ray-set rotation only at a full-field pass boundary AND no more
        // often than ROT_MIN_TICKS apart. Within a held rotation every probe is solved with the
        // SAME rays, so re-solving a static probe yields an identical estimate → the temporal
        // blend is a no-op → ZERO per-tick boil (the flicker). The original pass-only gate held
        // for LARGE grids (multi-tick passes) but small grids finish a full pass EVERY tick, so
        // they rotated every tick → fresh 64-ray noise each tick. The tick floor keeps slow
        // multi-rotation averaging (SH-volume smoothness) at any grid size.
        ticksSinceRot++;
        if (probeCursor === 0 && ticksSinceRot >= ROT_MIN_TICKS) {
            if (refreshStarted) { frameCounter = (frameCounter + 1) >>> 0; U.frameJitter.value = (frameCounter * 0.61803398875) % 1; }
            refreshStarted = true;
            ticksSinceRot = 0;
        }
        probeCursor = probeTotal > 0 ? (probeCursor + updated) % probeTotal : 0;

        inFlight = true;
        try {
            // (A5) One-time-per-rebuild prep. clear (full rebuild only) + the cheap
            // uploadState (ALWAYS — it's the sole writer of stateAtlas; a zero-init
            // stateBuffer yields a neutral state, and skipping it would leave the
            // relocation sample reading uninitialised texels → NaN). classify (the
            // 32-ray BVH walk) runs ONLY when Solid-scene mode is on. These WRITE
            // buffers the trace then READS, so let them finish before tracing.
            if (needsClear || needsClassify) {
                const prep = [];
                if (needsClear) { prep.push(renderer.computeAsync(gpu.clearAtlasKernel)); needsClear = false; }
                if (needsClassify) {
                    if (U.classifyStrength.value > 0) prep.push(renderer.computeAsync(gpu.classifyKernel));
                    prep.push(renderer.computeAsync(gpu.uploadStateKernel));
                    needsClassify = false;
                }
                await Promise.all(prep);
            }
            // (A6) Submit trace→blend→upload in order WITHOUT awaiting between them:
            // same-queue submission order preserves the data dependency, so a single
            // trailing await suffices. This removes the per-kernel CPU↔GPU round-trip
            // that serialized GI against the frame (the steady-state "slow").
            await Promise.all([
                renderer.computeAsync(gpu.traceKernel),
                renderer.computeAsync(gpu.blendKernel),
                renderer.computeAsync(gpu.uploadKernel),
            ]);
        } catch (e) {
            console.warn('max.js HALO-GI probe tick failed:', e);
            dirty = true;
        } finally {
            inFlight = false;
        }
    }

    function setEnabled(on) { node.setEnabled(on === true); if (on && (!gpu || !node._hasData)) requestRebuild(); }
    // freshBuild=true (default) invalidates the cached BVH+texture soup so rebuild()
    // rebuilds it (geometry/light-count/volume change, or first build). Grid-only callers
    // (setDivisions/setRays) pass false → the cached soup is reused, no MeshBVH hitch.
    function requestRebuild(freshBuild = true) { dirty = true; rebuildBackoff = 0; if (freshBuild) buildDirty = true; }
    // Set explicit probe volume(s) — e.g. synced "HALO-GI Probe Grid" helpers. Each
    // entry is a world-space THREE.Box3 (auto resolution) OR { box, res } where res is
    // a Vector3/[x,y,z] of MANUAL per-axis divisions. Pass null/empty to revert to
    // whole-scene auto-fit. When set, auto-fit is OFF (the box bounds the field).
    function normalizeRes(r) {
        if (!r) return null;
        const rx = Number.isFinite(r.x) ? r.x : (Array.isArray(r) ? r[0] : NaN);
        const ry = Number.isFinite(r.y) ? r.y : (Array.isArray(r) ? r[1] : NaN);
        const rz = Number.isFinite(r.z) ? r.z : (Array.isArray(r) ? r[2] : NaN);
        if (!Number.isFinite(rx) || !Number.isFinite(ry) || !Number.isFinite(rz)) return null;
        const c = (v) => THREE.MathUtils.clamp(Math.round(v), 2, MAX_PROBES_PER_AXIS);
        return new THREE.Vector3(c(rx), c(ry), c(rz));
    }
    function setVolumes(boxes) {
        const arr = Array.isArray(boxes) ? boxes : [boxes];
        const list = [];
        for (const entry of arr) {
            if (!entry) continue;
            const box = entry.isBox3 ? entry : entry.box;
            if (!box || !box.isBox3 || box.isEmpty()) continue;
            list.push({ box: box.clone(), res: entry.isBox3 ? null : normalizeRes(entry.res) });
        }
        manualVolumes = list.length ? list : null;
        requestRebuild();
    }
    const setBounds = (box) => setVolumes(box ? [box] : null); // single-box convenience

    // ── reactivity helpers ──
    // Cheap scene signatures: a change flags a light refresh (in-place) or a full
    // BVH rebuild. Both kick a low-hysteresis burst so the field re-converges fast.
    function lightSignature() {
        let s = '', n = 0;
        // (B4) Scene-relative deadbands so sub-perceptual delta-sync jitter does NOT
        // arm a reactive burst every check; a genuine edit still changes the signature.
        const q = lightQuant > 1e-6 ? lightQuant : 1;
        scene.traverseVisible((o) => {
            if (!o.isLight || o.isAmbientLight || o.isHemisphereLight) return;
            o.getWorldPosition(_sigVec);
            const c = o.color;
            s += `${Math.round(_sigVec.x / q)},${Math.round(_sigVec.y / q)},${Math.round(_sigVec.z / q)}|`
               + `${c ? Math.round((c.r * 7 + c.g * 11 + c.b * 13) * 16) : 0}|`
               + `${Math.round((o.intensity || 0) * 4)}|${Math.round((o.angle || 0) * 50)};`;
            n++;
        });
        return n + ':' + s;
    }
    function geoSignature() {
        let meshes = 0, prims = 0, hash = 0;
        // (A1) Quantize world translation to ~¼ cell: sub-perceptual sync jitter does
        // NOT re-arm a rebuild, while a genuine move DOES (the BVH bakes world-space
        // verts, so real moves MUST rebuild or the field traces stale geometry).
        const q = quantStep > 1e-6 ? quantStep : 1;
        scene.traverseVisible((o) => {
            if (!o.isMesh && !o.isInstancedMesh) return;
            const p = o.geometry?.attributes?.position; if (!p) return;
            meshes++;
            prims += o.geometry.index ? o.geometry.index.count : p.count;
            const e = o.matrixWorld?.elements;
            if (e) hash += Math.round(e[12] / q) + Math.round(e[13] / q) * 1.7 + Math.round(e[14] / q) * 2.3 + p.count + (o.count || 1);
        });
        return `${meshes}:${prims}:${Math.round(hash)}`;
    }
    // re-collect lights into the existing buffer (no BVH rebuild). Count change → full rebuild.
    function refreshLights() {
        if (!gpu || !_collectLights) return;
        let records;
        try { records = _collectLights(THREE, scene); } catch { return; }
        if (records.length !== gpu.lightCount) { requestRebuild(); return; }
        const arr = gpu.buffers.lights.array;
        arr.fill(0);
        for (let i = 0; i < records.length; i++) arr.set(records[i], i * _LIGHT_STRIDE);
        gpu.buffers.lights.needsUpdate = true;
        reactiveTicks = REACTIVE_TICKS;
    }

    function dispose() { disposed = true; disposeGPU(); cachedBuilt = null; buildDirty = true; node.setEnabled(false); }

    return {
        node,
        tick,
        setEnabled,
        setIntensity: (v) => node.setIntensity(v),
        setChebyStrength: (v) => node.setChebyStrength(v),
        setClassifyStrength: (v) => {
            node.setClassifyStrength(v); // node-side: classification gate + relocation apply
            if (Number.isFinite(v)) {
                U.classifyStrength.value = THREE.MathUtils.clamp(v, 0, 1); // trace-side relocation apply
                // The classify kernel only runs when classifyStrength>0, and the state buffer
                // is zero-init (= "buried"). So turning solid-scene on AFTER load read every
                // probe as buried and killed GI. (Re)run classification now so the state atlas
                // actually reflects which probes are buried vs free.
                if (U.classifyStrength.value > 0) needsClassify = true;
            }
        },
        setDivisions: (n) => {
            const v = THREE.MathUtils.clamp(Math.round(Number(n)) || TARGET_PROBES_LONG_AXIS, 2, MAX_PROBES_PER_AXIS);
            if (v === targetLongAxis) return;
            targetLongAxis = v;
            requestRebuild(false); // grid-only resize → reuse cached BVH+textures (no MeshBVH hitch)
        },
        getDivisions: () => targetLongAxis,
        // ── STRUCTURAL knob: ray budget per probe. Re-sizes the ray scratch + rebuilds the
        // trace/blend kernels, so it goes through the idle-gated rebuild (never a per-tick recompile).
        setRays: (n) => {
            const v = THREE.MathUtils.clamp(Math.round(Number(n) / 16) * 16 || RAYS_PER_PROBE_DEFAULT, RAYS_MIN, RAYS_MAX);
            if (v === raysPerProbe) return;
            raysPerProbe = v;
            requestRebuild(false); // ray budget only → reuse cached BVH+textures (kernel rebuild, no MeshBVH hitch)
        },
        getRays: () => raysPerProbe,
        // ── UNIFORM knobs (apply INSTANTLY — no recompile, no rebuild). ──
        setFilterStrength: (v) => { if (Number.isFinite(v)) U.filterStrength.value = THREE.MathUtils.clamp(v, 0, 1); }, // CORE denoise: 0 = off (harness baseline), 1 = full
        setSmoothness: (v) => { if (Number.isFinite(v)) U.filterSmooth.value = THREE.MathUtils.clamp(v, 0, 1); }, // UI "Smoothness": widen the denoise edge-stop
        setHysteresis: (v) => {
            if (!Number.isFinite(v)) return;
            baseHysteresis = THREE.MathUtils.clamp(v, 0, 0.99); // steady-state temporal blend (higher = more stable/slower)
            U.hysteresis.value = baseHysteresis;                // apply now; tick() re-asserts it when no reactive burst is active
        },
        setNormalBias: (v) => {
            if (!Number.isFinite(v)) return;
            normalBiasScale = THREE.MathUtils.clamp(v, 0, 8);   // × the auto minCell·SURFACE_NORMAL_BIAS_CELL offset
            node.normalBiasNode.value = Math.max(1e-4, curMinCell * SURFACE_NORMAL_BIAS_CELL * normalBiasScale);
        },
        setRadianceClamp: (v) => { if (Number.isFinite(v)) U.radianceClamp.value = Math.max(0, v); },   // cap multibounce feedback (anti-runaway)
        setDepthSharpness: (v) => { if (Number.isFinite(v)) U.depthSharpness.value = THREE.MathUtils.clamp(v, 1, 200); }, // depth-moment cosine power (Chebyshev crispness)
        requestRebuild,
        setBounds,
        setVolumes,
        isSupported,
        hasData: () => node._hasData === true,
        getStats: () => ({ probes: probeTotal, res: res.clone(), atlas: [atlasW, atlasH], rays: raysPerProbe, oct: OCT_RES, tile: TILE, active: node.active }),
        getResolution: () => res.clone(),
        getBounds: () => new THREE.Box3(gridMin.clone(), gridMin.clone().add(gridSize)),
        _debugUpload: async () => { if (gpu && !disposed) { try { await renderer.computeAsync(gpu.uploadKernel); } catch (e) { /* harness-only */ } } },
        _debugAtlas: () => gpu?.atlas || null,
        _debugDepthAtlas: () => gpu?.depthAtlas || null,
        _debugStateAtlas: () => gpu?.stateAtlas || null,
        _debugStateBuffer: () => gpu?.stateBuffer || null,
        _debugRead: async (which) => {
            const buf = which === 'irr' ? gpu?.irrBuffer
                : which === 'mat' ? gpu?.buffers?.materials
                : which === 'lights' ? gpu?.buffers?.lights
                : which === 'state' ? gpu?.stateBuffer : null;
            if (!buf || typeof renderer.getArrayBufferAsync !== 'function') return null;
            try { return new Float32Array(await renderer.getArrayBufferAsync(buf)); } catch (e) { return { error: String(e) }; }
        },
        _debugLightCount: () => gpu?.lightCount,
        dispose,
    };
}

export default createProbeField;
