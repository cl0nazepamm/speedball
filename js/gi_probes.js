// gi_probes.js — SPEEDBALL GI DDGI irradiance field (docs/GI_SPEEDBALL_design.md §3).
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
// In-trace albedo/emissive TEXTURE sampling for the bounce (Lumen-style textured GI).
// Near-free: the BVH traversal already samples the albedo atlas per hit-candidate for
// alpha-testing, so this only adds one RGB fetch at the final hit — the 64-ray walk
// dominates. Skipped entirely when a scene has no albedo/emissive maps. Textured albedo
// reads darker than a flat factor, so compensate with GI intensity/exposure.
const GI_SAMPLE_TEXTURES = true;
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
// ── denoise uplift (CORE, docs/GI_SPEEDBALL_design.md §11) tunables ──
const GI_FILTER_K = 8.0;           // spatial filter: variance→edge-stop bandwidth
const GI_FILTER_EPS = 0.001;       // spatial filter luma² absolute floor (avoids /0 on black)
const GI_FILTER_REL = 0.0225;      // spatial filter RELATIVE floor (~15% luma)²: even a temporally
                                   // converged texel gets a mild edge-PRESERVING bilateral smooth of
                                   // sub-threshold (noise-scale) neighbours, while strong directional
                                   // edges (red↔green) stay sharp. Steady-state splotch reduction.
// ── temporal stabilization: per-texel variance-aware hysteresis ──
const GI_TEMPORAL_NOISE_H_BOOST = 0.25; // steady/noisy samples borrow a little extra history, not a hard floor
const GI_TEMPORAL_CHANGE_H_DROP = 0.30; // significant per-texel changes converge faster than the base slider
const GI_TEMPORAL_MIN_CHANGE_H = 0.55;
const GI_TEMPORAL_VAR_EPS = 0.000025;  // absolute luma variance floor
const GI_TEMPORAL_VAR_REL = 0.0025;    // relative floor: lum^2 * this
const GI_TEMPORAL_CHANGE_SIGMA0 = 0.75;
const GI_TEMPORAL_CHANGE_SIGMA1 = 2.5;
const GI_TEMPORAL_CLAMP_SIGMA = 6.0;

let _node = null;

// ── cascaded probe grid ──
// C0 = coarse full-bounds grid (byte-identical to the old single grid when cascades=1);
// C1 = fine sub-box at ~2× spacing, placed by a CPU triangle-density histogram over the
// SHARED BVH soup. Only ONE cascade is solved per tick (round-robin), so the per-tick GPU
// budget is unchanged. cascades=1 is the byte-identical fallback (C1 never allocated).
const NUM_CASC = 2;
const C1_RES_SCALE = 2.0;             // fine cascade target ≈ 2× the coarse long-axis density
const C1_MIN_AXIS_FRAC = 0.25;        // fine box ≥ 25% of the coarse box per axis (not degenerate)
const C1_MAX_AXIS_FRAC = 0.60;        // fine box ≤ 60% of the coarse box per axis (not the whole scene)
const C1_HIST_G = 16;                 // 16³ = 4096 fixed density bins (constant, cheap)
const C1_HIST_THRESHOLD = 0.4;        // union bins ≥ 0.4·peak into the detail cluster

// ── injection node: samples the atlas at the shaded surface and adds the
// probe irradiance into builder.context.irradiance (mirrors the GiVolumeNode /
// hemisphere addAssign pattern at max_lights_node.js:224). Stable atlas binding
// → its cacheToken changes ONLY on grid resize / enable, never on data writes.
export class GiProbeNode extends LightingNode {
    static get type() { return 'GiProbeNode'; }

    constructor() {
        super();
        // Per-cascade atlas triplets (index 0 = C0 coarse, 1 = C1 fine). SEPARATE atlases
        // per cascade (not a shared pack): each runs the existing atlas-fit math verbatim.
        this._atlas = [null, null];
        this._depthAtlas = [null, null];
        this._stateAtlas = [null, null];
        this._enabled = false;
        this._structGen = 0;     // bumps on grid resize / atlas realloc / cascade-count change ONLY
        this.intensity = 1.0;

        // Per-cascade grid uniforms (2-slot arrays).
        this.gridMinNode = [uniform(new THREE.Vector3()), uniform(new THREE.Vector3())];
        this.gridSizeNode = [uniform(new THREE.Vector3(1, 1, 1)), uniform(new THREE.Vector3(1, 1, 1))];
        this.resNode = [uniform(new THREE.Vector3(2, 2, 2)), uniform(new THREE.Vector3(2, 2, 2))];
        this.atlasDimNode = [uniform(new THREE.Vector2(1, 1)), uniform(new THREE.Vector2(1, 1))];
        // Per-cascade biases: coarse cells are larger → its own normal/cheby bias.
        this.normalBiasNode = [uniform(0.04), uniform(0.04)];
        this.chebyBiasNode = [uniform(0.0), uniform(0.0)];

        // Cascade-invariant look uniforms (single instance).
        this.intensityNode = uniform(1.0);
        // Runtime enable gate. The graph membership below is still the authoritative
        // on/off switch; this uniform remains as a cheap extra guard for compiled graphs.
        this.enabledNode = uniform(1.0);
        // Active cascade count. Defaults to 1 so the very first fold is the single-grid
        // shader (byte-identical fallback); set to `cascades` after the first full build.
        this.cascadeCountNode = uniform(1.0);
        // Fraction of the C1 extent used as the fine→coarse blend band (hides the seam).
        this.borderBandNode = uniform(0.15);
        // 0 → no visibility test = pure trilinear "radiosity" look (THE DEFAULT, by user pref:
        // smoother, no per-triangle self-occlusion); 1 → full Chebyshev leak-free visibility.
        // The Chebyshev term self-occludes on dense/thick geometry and hurt the look more than
        // leaks helped, so it's off by default and has no UI toggle. Reachable via setCheby(1).
        this.chebyStrengthNode = uniform(0.0);
        // 0 → classification IGNORED (default — safe for thin 2-sided walls, which
        // a backface test misreads); 1 → drop probes buried in SOLID geometry.
        this.classifyStrengthNode = uniform(0.0);
    }

    // computed readiness: every cascade in [0..count) has a non-null atlas triplet.
    get _ready() {
        const count = Math.round(this.cascadeCountNode.value) || 1;
        for (let c = 0; c < count; c++) {
            if (!this._atlas[c] || !this._depthAtlas[c] || !this._stateAtlas[c]) return false;
        }
        return true;
    }

    // Graph membership is the authoritative GI switch. WebGPU node uniforms can stay
    // cached in already-built light graphs, so enable/disable must change whether this
    // lighting node is pushed at all; the page marks PBR materials dirty after flips.
    // Graph membership = "is there a field to sample". MUST stay decoupled from
    // enabled/intensity: setup() gates those via the enabledNode/intensityNode UNIFORMS,
    // so the node is folded into the shader ONCE (when data appears) and stays put.
    // Gating active on _enabled/intensity reintroduces the original bug — toggling them
    // only flips the cacheToken, which never forces a recompile, so GI silently drops out.
    get active() { return this._ready; }
    // structure-only token: data writes (textureStore) do NOT change this, so
    // materials never recompile on a probe tick — only on resize / first data.
    get cacheToken() { return `gi-speedball-probes:${this._structGen}`; }

    setEnabled(on) { this._enabled = on === true; this.enabledNode.value = this._enabled ? 1.0 : 0.0; }
    setIntensity(v) {
        this.intensity = Number.isFinite(v) ? Math.max(0, v) : 0;
        this.intensityNode.value = this.intensity;
    }
    setChebyStrength(v) { if (Number.isFinite(v)) this.chebyStrengthNode.value = THREE.MathUtils.clamp(v, 0, 1); }
    setClassifyStrength(v) { if (Number.isFinite(v)) this.classifyStrengthNode.value = THREE.MathUtils.clamp(v, 0, 1); }
    // Active-cascade count (fragment-visible): 1 → wFine≡0 (byte-identical), 2 → blend.
    setCascadeCount(n) {
        const v = (Math.round(Number(n)) === 2) ? 2 : 1;
        if (this.cascadeCountNode.value === v) return;
        this.cascadeCountNode.value = v;
        this._structGen++;   // shader tap count changes → one recompile
    }
    setAtlases(c, atlas, depthAtlas, stateAtlas) {
        this._atlas[c] = atlas || null;
        this._depthAtlas[c] = depthAtlas || null;
        this._stateAtlas[c] = stateAtlas || null;
        this._structGen++;
    }
    // Update the grid placement uniforms only. Uniform .value writes do NOT change
    // a material cache key, so this is churn-free — the same-dim rebuild path uses
    // it to re-place probes after a geometry edit WITHOUT a TSL recompile.
    updateGridUniforms(c, gridMin, gridSize, res, atlasW, atlasH, normalBias, chebyBias) {
        this.gridMinNode[c].value.copy(gridMin);
        this.gridSizeNode[c].value.copy(gridSize);
        this.resNode[c].value.copy(res);
        this.atlasDimNode[c].value.set(atlasW, atlasH);
        if (Number.isFinite(normalBias)) this.normalBiasNode[c].value = Math.max(1e-4, normalBias);
        if (Number.isFinite(chebyBias)) this.chebyBiasNode[c].value = Math.max(0, chebyBias);
    }
    setGrid(c, gridMin, gridSize, res, atlasW, atlasH, normalBias, chebyBias) {
        this.updateGridUniforms(c, gridMin, gridSize, res, atlasW, atlasH, normalBias, chebyBias);
        this._structGen++;   // resize/first-enable ONLY → cacheToken moves → one recompile
    }

    // world position of grid probe (px,py,pz) in cascade c.
    _probePos(px, py, pz, c) {
        const f = vec3(px, py, pz).div(this.resNode[c].sub(1.0).max(vec3(1.0)));
        return this.gridMinNode[c].add(f.mul(this.gridSizeNode[c]));
    }

    // tile-local atlas uv for probe (col,row) at octahedral coord octUV in cascade c.
    _tileUV(col, row, octUV, c) {
        const ox = col.mul(float(TILE)).add(float(BORDER)).add(octUV.x.mul(float(OCT_RES))).add(0.5);
        const oy = row.mul(float(TILE)).add(float(BORDER)).add(octUV.y.mul(float(OCT_RES))).add(0.5);
        return vec2(ox.div(this.atlasDimNode[c].x), oy.div(this.atlasDimNode[c].y));
    }

    // sample the probe field at world (P, N): trilinear over the 8 cage probes,
    // each fetched octahedrally in the shading-normal direction and weighted by
    // a depth-moment Chebyshev visibility test (leak-free through thin walls).
    // Sample ONE cascade c (0=coarse, 1=fine) — the original 8-tap trilinear gather,
    // parameterized per cascade. PURE-EXPRESSION (no toVar/addAssign): fragment colorNode.
    _sampleCascade(P, N, c) {
        const atlas = this._atlas[c];
        const depthAtlas = this._depthAtlas[c];
        const res = this.resNode[c];
        const ry = res.y;
        const cell = this.gridSizeNode[c].div(res.sub(1.0).max(vec3(1.0)));
        const gridF = P.sub(this.gridMinNode[c]).div(cell.max(vec3(1e-6)));
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
            const metaUV = vec2(col.add(0.5).div(this.resNode[c].x), row.add(0.5).div(this.resNode[c].y.mul(this.resNode[c].z)));
            const meta = texture(this._stateAtlas[c], metaUV);
            const stateV = meta.x;
            const reloc = vec3(meta.y, meta.z, meta.w).mul(this.classifyStrengthNode);

            // Chebyshev visibility: relocated probe → surface direction vs stored depth.
            const probePos = this._probePos(px, py, pz, c).add(reloc);
            const toSurf = P.sub(probePos);
            const dist = length(toSurf);
            const octD = octEncodeNode(toSurf.div(dist.max(float(1e-6))), TSL);
            const m = texture(depthAtlas, this._tileUV(col, row, octD, c));
            const m1 = m.x; const m2 = m.y;
            const variance = m2.sub(m1.mul(m1)).abs();
            // Self-occlusion tolerance: a lit surface must not shadow ITSELF against its own
            // low-res depth moments. dist (probe→fragment) carries the normal-bias offset and
            // is compared to oct-averaged depth, so on dense/thick geometry it slips just past
            // m1 per triangle → the leak-free term "errors on triangles". A depth bias (db, <
            // wall thickness) treats the surface as visible within tolerance, so real walls
            // still occlude (no leak) but a surface stops fighting its own shadow.
            const db = this.chebyBiasNode[c];
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
            const e = texture(atlas, this._tileUV(col, row, octN, c)).xyz;
            acc = acc.add(e.mul(w));
            wsum = wsum.add(w);
        }
        return acc.div(wsum.max(float(1e-4)));
    }

    // Cascaded sample: always the coarse cascade; blend toward the fine cascade across a
    // narrow inner border when P is inside C1 and cascades>=2. Fixed tap count (8 or 16),
    // no data-dependent branch → bounded fragment cost. cascades==1 → E0 exactly.
    //
    // COMPILE-TIME cascade selection (invariants #5/#6): a TSL fragment cannot reference a
    // null StorageTexture, and the material RECOMPILES whenever _structGen changes (which
    // includes every cascade-count change). So the E1 (fine) 8-tap subtree is emitted ONLY
    // when the fine cascade is actually bound at setup() time — otherwise the shader is the
    // exact original 8-tap single grid (byte-identical) and never touches _atlas[1]=null.
    sampleIrradiance(P, N) {
        const E0 = this._sampleCascade(P, N, 0); // coarse: valid over full bounds, ALWAYS
        const useFine = Math.round(this.cascadeCountNode.value) >= 2 && !!this._atlas[1] && !!this._depthAtlas[1] && !!this._stateAtlas[1];
        if (!useFine) return E0;                 // single-grid fallback — byte-identical to today
        // Fine inside-test in normalized C1 coords.
        const f = P.sub(this.gridMinNode[1]).div(this.gridSizeNode[1].max(vec3(1e-6))); // 0..1 inside the fine box
        const fLo = tslMin(tslMin(f.x, f.y), f.z);
        const fHi = tslMin(tslMin(float(1.0).sub(f.x), float(1.0).sub(f.y)), float(1.0).sub(f.z));
        const edge = tslMin(fLo, fHi); // dist to nearest face; <0 outside
        const inBand = this.borderBandNode;
        const wIn = clamp(edge.div(inBand.max(float(1e-4))), float(0.0), float(1.0)); // 1 deep inside, ramps to 0 at the border, 0 outside
        const gate = this.cascadeCountNode.sub(1.0).clamp(float(0.0), float(1.0)); // 0 when cascades==1, 1 when 2 (live extra guard)
        const wFine = wIn.mul(gate);
        const E1 = this._sampleCascade(P, N, 1);
        return mix(E0, E1, wFine);
    }

    setup(builder) {
        if (!this._ready) return;
        const P = positionWorld.add(normalWorld.mul(this.normalBiasNode[0]));
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

    // Active cascade count. 1 = byte-identical single-grid fallback (C1 never allocated);
    // 2 = coarse + fine detail cascade. cascadeCountNode defaults to 1 and is set to
    // `cascades` after the first full build so the first fold is the single-grid shader.
    let cascades = 2;
    let solveTurn = 0;        // round-robin cascade index across ticks (C0 even, C1 odd)
    let buildStage = 0;       // staggered build phase machine (0 = build C0, 1 = build C1, 2 = done)
    let buildCascadeCount = 1; // effective cascade count for the CURRENT build (fitFineBox may drop to 1)

    // Per-cascade state (index 0 = C0 coarse full-bounds, 1 = C1 fine sub-box). The BVH
    // soup (cachedBuilt) stays a single shared driver var — never duplicated per cascade.
    function makeCascadeU() {
        return {
            gridMin: uniform(new THREE.Vector3()),
            gridSize: uniform(new THREE.Vector3(1, 1, 1)),
            resX: uniform(2, 'uint'), resY: uniform(2, 'uint'), resZ: uniform(2, 'uint'),
            probeTotal: uniform(1, 'uint'),
            probeOffset: uniform(0, 'uint'),
            updatedCount: uniform(1, 'uint'),
            atlasDim: uniform(new THREE.Vector2(1, 1)),
            maxDist: uniform(100.0),
            cellMin: uniform(0.1),
            relocClamp: uniform(0.045),
        };
    }
    function makeCascade() {
        return {
            gridMin: new THREE.Vector3(),
            gridSize: new THREE.Vector3(1, 1, 1),
            res: new THREE.Vector3(2, 2, 2),
            probeTotal: 0,
            atlasW: 1, atlasH: 1,
            minCell: 0.1,
            gpu: null,
            U: makeCascadeU(),
            probeCursor: 0,
            refreshStarted: false,
            ticksSinceRot: 0,
            prevAtlasW: 0, prevAtlasH: 0, prevProbeTotal: 0,
            needsClear: true, needsClassify: true,
            normalBias: 0.04, chebyBias: 0.0,
        };
    }
    const casc = [makeCascade(), makeCascade()];

    let continuous = false;   // false = idle-gated (default). true = keep the bounded GPU solve
                              // running while the camera moves; heavy build steps still wait for rest.
    let dirty = true;
    // Cached CPU build (BVH soup + material textures). The BVH depends ONLY on geometry,
    // so a divisions/rays change must NOT rebuild it — that ~200ms synchronous MeshBVH +
    // soup-flatten is the one remaining main-thread hitch. buildDirty gates a fresh build:
    // it's set by geometry/light-count/volume changes (true at start), and left FALSE by
    // setDivisions/setRays so those resize the grid/kernels off the cached soup (no hitch).
    let cachedBuilt = null;
    let buildDirty = true;
    let manualVolumes = null; // explicit probe volumes (Probe Origin boxes); null = auto-fit scene
    // needsClassify / needsClear / probeCursor / refreshStarted / ticksSinceRot / prev*
    // are now PER-CASCADE (see makeCascade); frameCounter is SHARED (one ray-set rotation
    // advanced only on C0's pass boundary — both cascades read the same U.frameJitter).
    let rebuildBackoff = 0;   // ticks remaining before retrying after a failed/empty rebuild (A7)
    let inFlight = false;
    let disposed = false;
    let frameCounter = 0;
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

    // SHARED compute uniforms — a single instance, folded into BOTH cascade U blocks by
    // reference (see below), so one GUI knob writes both cascades. Per-cascade uniforms
    // (gridMin/gridSize/res*/probeTotal/probeOffset/updatedCount/atlasDim/maxDist/cellMin/
    // relocClamp) live in each C.U (makeCascadeU).
    const U = {
        lightCount: uniform(0, 'uint'),
        frameJitter: uniform(0.0),
        hysteresis: uniform(THREE.MathUtils.clamp(hysteresis, 0, 0.99)),
        depthSharpness: uniform(50.0),  // cosine power → depth tracks nearest occluder crisply
        radianceClamp: uniform(8.0),    // cap the multibounce feedback term (anti-runaway)
        classifyStrength: uniform(0.0), // gates relocation APPLY (mirrors node.classifyStrengthNode)
        filterStrength: uniform(1.0),   // CORE denoise: 0 = filter off (harness baseline), 1 = full intra-tile spatial filter
        filterSmooth: uniform(0.5),     // UI "Smoothness": widens the bilateral edge-stop (0 = baseline detail, 1 = very smooth)
        // ── live temporal-blend tuning (adaptive hysteresis). Defaults == the old constants
        // → byte-identical until a slider moves. See blendKernel for how each is used.
        tempChangeSigma1: uniform(GI_TEMPORAL_CHANGE_SIGMA1), // delta (in σ) above which a change counts as REAL → snaps. lower = snappier
        tempChangeHDrop: uniform(GI_TEMPORAL_CHANGE_H_DROP),  // how much hysteresis drops on a real change. higher = harder snap
        tempClampSigma: uniform(GI_TEMPORAL_CLAMP_SIGMA),     // firefly clamp band (in σ). lower = tighter/steadier, more lag
    };
    // Fold the shared uniforms into every cascade's U by REFERENCE so buildKernels closes
    // over C.U and reads both shared + per-cascade uniforms uniformly.
    for (const C of casc) Object.assign(C.U, U);

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

    // Free ONE cascade's GPU resources (buffers + its own atlas triplet) and reset its
    // prev-dim trackers. Clears the node's atlas binding for that cascade slot.
    function disposeCascadeGPU(c) {
        const C = casc[c];
        const g = C.gpu;
        if (g) {
            for (const k of ['bvhNodes', 'triIndex', 'vertexData', 'triMaterial', 'materials', 'lights']) g.buffers[k]?.dispose?.();
            g.irrBuffer?.dispose?.();
            g.depthBuffer?.dispose?.();
            g.stateBuffer?.dispose?.();
            g.rayBuffer?.dispose?.();
            g.atlas?.dispose?.();
            g.depthAtlas?.dispose?.();
            g.stateAtlas?.dispose?.();
            if (g.maps) for (const t of Object.values(g.maps)) t?.dispose?.();
        }
        C.gpu = null;
        C.prevAtlasW = C.prevAtlasH = C.prevProbeTotal = 0; // next rebuild takes the full (resize) path
        node.setAtlases(c, null, null, null);
    }

    function disposeGPU() {
        for (let c = 0; c < NUM_CASC; c++) disposeCascadeGPU(c);
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

    function probeWorldPos(pIndexNode, U) {
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

    // Build ONE cascade's GPU kernels+resources, closing over the cascade object C
    // (C.U, C.probeTotal, C.atlasW/H, C.res). The BVH soup (built) is SHARED across
    // cascades — never rebuilt per cascade. reuse carries C's live buffers/atlases on a
    // same-dim rebuild. Returns the gpu object; the caller stores it in C.gpu.
    function buildKernels(built, C, reuse = null) {
        const U = C.U;                       // per-cascade + shared uniforms (folded by reference)
        const probeTotal = C.probeTotal;     // shadow the old flat name → per-cascade
        const atlasW = C.atlasW, atlasH = C.atlasH;
        const res = C.res;
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
            const ro = probeWorldPos(probeIndex, U).toVar();
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
            const lAcc = float(0.0).toVar();
            const lAcc2 = float(0.0).toVar();
            const LUMA = vec3(0.2126, 0.7152, 0.0722);
            Loop({ start: uint(0), end: uint(raysPerProbe), type: 'uint', condition: '<' }, ({ i: k }) => {
                const rb = slot.mul(uint(raysPerProbe)).add(k).mul(uint(4));
                const rrgb = vec3(rayData.element(rb), rayData.element(rb.add(uint(1))), rayData.element(rb.add(uint(2))));
                const hitT = rayData.element(rb.add(uint(3)));
                const rdir = normalize(rayDir(k, U.frameJitter));
                const cw = tslMax(dot(dir, rdir), float(0.0));
                acc.addAssign(rrgb.mul(cw));
                wsum.addAssign(cw);
                const rl = dot(rrgb, LUMA);
                lAcc.addAssign(rl.mul(cw));
                lAcc2.addAssign(rl.mul(rl).mul(cw));
                // depth moments: miss → "far" so the probe stays visible that way.
                const rdist = select(hitT.lessThan(float(0.0)), U.maxDist, hitT);
                const dw = pow(cw, U.depthSharpness);
                dAcc.addAssign(rdist.mul(dw));
                dAcc2.addAssign(rdist.mul(rdist).mul(dw));
                dwsum.addAssign(dw);
            });
            const meanRad = acc.div(wsum.max(float(1e-4)));
            const curL = lAcc.div(wsum.max(float(1e-4)));
            const curM2 = lAcc2.div(wsum.max(float(1e-4)));
            const meanR = dAcc.div(dwsum.max(float(1e-4)));
            const meanR2 = dAcc2.div(dwsum.max(float(1e-4)));

            // irradiance: read prev + write blended through ONE read_write 'float'
            // binding (proven surfel-grid pattern).
            const ib = probeIndex.mul(uint(TILE * TILE)).add(local).mul(uint(4)).toVar();
            const prev = vec3(irr.element(ib), irr.element(ib.add(uint(1))), irr.element(ib.add(uint(2))));
            const wasBlack = dot(prev, vec3(1.0)).lessThan(float(1e-6));
            // Variance-aware temporal accumulation: SPEEDBALL intentionally jitters rays every
            // tick. Keep steady texels slightly steadier, but lower hysteresis for significant
            // luma changes so the probe field does not feel frozen.
            const prevL = dot(prev, LUMA);
            const prevM2 = irr.element(ib.add(uint(3)));
            const prevVar = tslMax(prevM2.sub(prevL.mul(prevL)), float(0.0));
            const curVar = tslMax(curM2.sub(curL.mul(curL)), float(0.0));
            const lumRef = tslMax(tslMax(curL, prevL), float(0.0));
            const varFloor = float(GI_TEMPORAL_VAR_EPS).add(lumRef.mul(lumRef).mul(float(GI_TEMPORAL_VAR_REL)));
            const sigma = sqrt(prevVar.add(curVar).add(varFloor));
            const deltaL = curL.sub(prevL);
            const absDelta = tslAbs(deltaL);
            const s0 = sigma.mul(float(GI_TEMPORAL_CHANGE_SIGMA0));
            const s1 = sigma.mul(U.tempChangeSigma1.max(float(GI_TEMPORAL_CHANGE_SIGMA0 + 0.01)));
            const changeW = clamp(absDelta.sub(s0).div(s1.sub(s0).max(float(1e-6))), float(0.0), float(1.0));
            const noiseH = U.hysteresis.add(float(1.0).sub(U.hysteresis).mul(float(GI_TEMPORAL_NOISE_H_BOOST)));
            const changeH = tslMax(U.hysteresis.sub(U.tempChangeHDrop), float(GI_TEMPORAL_MIN_CHANGE_H));
            const hEff = mix(noiseH, changeH, changeW);
            const h = select(wasBlack, float(0.0), hEff);
            const band = sigma.mul(U.tempClampSigma);
            const clampScale = tslMin(float(1.0), band.div(absDelta.max(float(1e-6))));
            const clipped = prev.add(meanRad.sub(prev).mul(clampScale));
            const candidate0 = mix(clipped, meanRad, changeW);
            const candidate = select(wasBlack, meanRad, candidate0);
            const blended = mix(candidate, prev, h);
            irr.element(ib).assign(blended.x);
            irr.element(ib.add(uint(1))).assign(blended.y);
            irr.element(ib.add(uint(2))).assign(blended.z);
            // luminance 2nd moment E[L²] in the FREE 4th slot (buffer-only; the upload
            // keeps atlas.w=1.0 so the fragment sampler never sees it). luma is linear
            // → E[luma]=luma(E[rgb]), so variance = max(0, M2 − luma(rgb)²) anywhere.
            const m2 = mix(curM2, prevM2, h);
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
            const ro = probeWorldPos(p, U).toVar();
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

    function totalUnionProbes() {
        return casc[0].probeTotal + (cascades > 1 ? casc[1].probeTotal : 0);
    }
    function updatedCap() {
        // ONE shared per-tick budget over the UNION of both cascades, still ceilinged at
        // MAX_PROBES_PER_TICK. Because only ONE cascade dispatches per tick (round-robin),
        // total probes solved/tick == the single-grid count — never both cascades, never
        // over the cap. Also sizes the per-cascade ray scratch (safe upper bound).
        return Math.max(1, Math.min(MAX_PROBES_PER_TICK, Math.ceil(Math.max(1, totalUnionProbes()) / 4)));
    }

    // DETAIL-DRIVEN C1 placement: a cheap CPU triangle-centroid density histogram over the
    // SHARED built soup. Run INSIDE the idle-gated rebuild (never during motion), alongside
    // the ~200ms MeshBVH build → cannot hitch. Deterministic geometry → stable box across
    // rebuilds → same-dim reuse keeps working. Returns {min, size} for C1, or null (fallback
    // to cascades=1 placement) on a flat/degenerate histogram or an all-scene cluster.
    function fitFineBox(built, box0) {
        if (!built || !built.vertexData || !built.triIndex || !(built.triCount > 0)) return null;
        const G = C1_HIST_G;                             // 16³ = 4096 fixed bins (constant, cheap)
        const hist = new Uint32Array(G * G * G);
        const min = box0.min;
        const size = new THREE.Vector3(); box0.getSize(size);
        const inv = [G / Math.max(1e-6, size.x), G / Math.max(1e-6, size.y), G / Math.max(1e-6, size.z)];
        const vd = built.vertexData, ti = built.triIndex, S = 8; // VERTEX_DATA_STRIDE, pos at 0-2
        const triCount = built.triCount;
        for (let t = 0; t < triCount; t++) {             // one linear pass: centroid binning
            let cx = 0, cy = 0, cz = 0;
            for (let k = 0; k < 3; k++) { const v = ti[t * 3 + k] * S; cx += vd[v]; cy += vd[v + 1]; cz += vd[v + 2]; }
            cx /= 3; cy /= 3; cz /= 3;
            const gx = THREE.MathUtils.clamp((cx - min.x) * inv[0] | 0, 0, G - 1);
            const gy = THREE.MathUtils.clamp((cy - min.y) * inv[1] | 0, 0, G - 1);
            const gz = THREE.MathUtils.clamp((cz - min.z) * inv[2] | 0, 0, G - 1);
            hist[(gz * G + gy) * G + gx]++;
        }
        // Peak bin, then union the AABB (in bin coords) of all bins ≥ threshold·peak.
        let peak = 0;
        for (let i = 0; i < hist.length; i++) if (hist[i] > peak) peak = hist[i];
        if (peak === 0) return null;                     // no geometry → fallback
        const thr = peak * C1_HIST_THRESHOLD;
        let bx0 = G, by0 = G, bz0 = G, bx1 = -1, by1 = -1, bz1 = -1, count = 0;
        for (let z = 0; z < G; z++) for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) {
            if (hist[(z * G + y) * G + x] >= thr) {
                if (x < bx0) bx0 = x; if (x > bx1) bx1 = x;
                if (y < by0) by0 = y; if (y > by1) by1 = y;
                if (z < bz0) bz0 = z; if (z > bz1) bz1 = z;
                count++;
            }
        }
        if (count === 0 || bx1 < 0) return null;
        // Flat/degenerate: cluster spans essentially the whole box → no detail region → fallback.
        if (bx0 === 0 && by0 === 0 && bz0 === 0 && bx1 === G - 1 && by1 === G - 1 && bz1 === G - 1) return null;

        const cell = [size.x / G, size.y / G, size.z / G];
        // bin range → world AABB, padded by one coarse cell (box0.minCell approximated by bin cell).
        const fmin = new THREE.Vector3(min.x + bx0 * cell[0], min.y + by0 * cell[1], min.z + bz0 * cell[2]);
        const fmax = new THREE.Vector3(min.x + (bx1 + 1) * cell[0], min.y + (by1 + 1) * cell[1], min.z + (bz1 + 1) * cell[2]);
        const pad = Math.min(cell[0], cell[1], cell[2]);
        fmin.subScalar(pad); fmax.addScalar(pad);
        const fsize = fmax.clone().sub(fmin);
        // Enforce MIN (≥ C1_MIN_AXIS_FRAC of box0) and MAX (≤ C1_MAX_AXIS_FRAC) per axis; re-center
        // and clip to box0 so C1 is neither degenerate nor the whole scene.
        const box0max = min.clone().add(size);
        const axes = ['x', 'y', 'z'];
        for (const a of axes) {
            const lo = C1_MIN_AXIS_FRAC * size[a];
            const hi = C1_MAX_AXIS_FRAC * size[a];
            let s = THREE.MathUtils.clamp(fsize[a], lo, hi);
            let center = (fmin[a] + fmax[a]) * 0.5;
            let mn = center - s * 0.5;
            let mx = center + s * 0.5;
            // clip to box0 (shift the window back inside if it overhangs).
            if (mn < min[a]) { mx += (min[a] - mn); mn = min[a]; }
            if (mx > box0max[a]) { mn -= (mx - box0max[a]); mx = box0max[a]; }
            mn = Math.max(mn, min[a]); mx = Math.min(mx, box0max[a]);
            fmin[a] = mn; fsize[a] = Math.max(1e-4, mx - mn);
        }
        return { min: fmin, size: fsize };
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

    // Fit an atlas within the GPU 2D texture limit by uniformly shrinking res. Mutates
    // resVec; returns the same vector. (The proven single-grid clamp loop, per cascade.)
    const _maxDim = () => (renderer?.backend?.device?.limits?.maxTextureDimension2D || ATLAS_DIM_FALLBACK);
    function fitAtlas(resVec) {
        const maxDim = _maxDim();
        for (let g = 0; g < 12 && (resVec.x * TILE > maxDim || resVec.y * resVec.z * TILE > maxDim); g++) {
            resVec.set(Math.max(2, Math.floor(resVec.x * 0.85)), Math.max(2, Math.floor(resVec.y * 0.85)), Math.max(2, Math.floor(resVec.z * 0.85)));
        }
        return resVec;
    }

    // Build (or same-dim-reuse) ONE cascade's kernels+resources and wire its uniforms +
    // node bindings. Handles both the reuse (churn-free) path and the full recompile path
    // for that cascade independently. Does NOT fire onRebuilt (the caller sequences that).
    function buildOneCascade(built, c) {
        const C = casc[c];
        const gridMin = C.gridMin, gridSize = C.gridSize, res = C.res;
        const minCell = C.minCell;
        const atlasW = C.atlasW, atlasH = C.atlasH, probeTotal = C.probeTotal;
        const normalBias = minCell * SURFACE_NORMAL_BIAS_CELL * normalBiasScale;
        const chebyBias = minCell * GI_CHEBY_BIAS_CELL;

        // Per-cascade same-dim reuse: reuse this cascade's live atlases + buffers if its
        // dims are unchanged → NO recompile for this cascade, NO black flash.
        const sameDim = !!C.gpu && atlasW === C.prevAtlasW && atlasH === C.prevAtlasH && probeTotal === C.prevProbeTotal;
        if (sameDim) {
            const prev = C.gpu;
            const reuse = {
                atlas: prev.atlas, depthAtlas: prev.depthAtlas, stateAtlas: prev.stateAtlas,
                irrBuffer: prev.irrBuffer, depthBuffer: prev.depthBuffer, stateBuffer: prev.stateBuffer,
            };
            C.gpu = buildKernels(built, C, reuse);
            disposeBVHOnly(prev);   // free ONLY the old BVH storages + ray scratch + maps
            C.U.gridMin.value.copy(gridMin);
            C.U.gridSize.value.copy(gridSize);
            C.U.lightCount.value = Math.min(MAX_LIGHTS, C.gpu.lightCount) >>> 0;
            C.U.maxDist.value = gridSize.length();
            C.U.cellMin.value = Math.max(1e-4, minCell);
            C.U.relocClamp.value = 0.45 * minCell;
            // Churn-free: update the NODE's placement uniforms WITHOUT bumping _structGen.
            node.updateGridUniforms(c, gridMin, gridSize, res, atlasW, atlasH, normalBias, chebyBias);
            C.probeCursor = 0;
            C.refreshStarted = false;
            C.needsClear = false;             // reuse the live atlas history (no black flash)
            C.needsClassify = true;           // refresh per-probe state for the new geometry
            return false;                     // no recompile occurred
        }

        // Resize / first-enable path for this cascade: fresh resources + one recompile.
        disposeCascadeGPU(c);
        C.gpu = buildKernels(built, C);
        C.U.gridMin.value.copy(gridMin);
        C.U.gridSize.value.copy(gridSize);
        C.U.resX.value = res.x >>> 0; C.U.resY.value = res.y >>> 0; C.U.resZ.value = res.z >>> 0;
        C.U.probeTotal.value = probeTotal >>> 0;
        C.U.atlasDim.value.set(atlasW, atlasH);
        C.U.lightCount.value = Math.min(MAX_LIGHTS, C.gpu.lightCount) >>> 0;
        C.U.maxDist.value = gridSize.length();
        C.U.cellMin.value = Math.max(1e-4, minCell);
        C.U.relocClamp.value = 0.45 * minCell;
        node.setAtlases(c, C.gpu.atlas, C.gpu.depthAtlas, C.gpu.stateAtlas);
        node.setGrid(c, gridMin, gridSize, res, atlasW, atlasH, normalBias, chebyBias);
        C.prevAtlasW = atlasW; C.prevAtlasH = atlasH; C.prevProbeTotal = probeTotal;
        C.probeCursor = 0;
        C.refreshStarted = false;
        C.needsClear = true;      // fresh StorageTextures aren't guaranteed zeroed
        C.needsClassify = true;
        return true;              // recompile occurred
    }

    // STAGE 0 of the staggered build: (re)compute BOTH cascades' dims off the SHARED soup
    // (C0 = today's path exactly; C1 = fitFineBox when cascades>=2, res ~2× finer), then
    // build C0 only. Sets cascadeCountNode=1 so the fold is the single-grid shader until C1
    // comes online (stage 1). Returns false on a failed/empty build.
    async function rebuild() {
        // Reuse the cached BVH+texture soup unless geometry actually changed. A
        // divisions/rays change (buildDirty=false) skips the synchronous MeshBVH build.
        let built = cachedBuilt;
        if (!built || buildDirty) {
            const buildSpectralScene = await ensureSceneBuilder();
            built = await buildSpectralScene({ THREE, scene, maxTriangles: MAX_TRIANGLES });
            if (!built || built.error) return false;   // keep existing field; tick() arms a backoff (A7)
            cachedBuilt = built;
            buildDirty = false;
        }

        const box = new THREE.Box3();
        scene.updateMatrixWorld(true);
        const hasVolumes = Array.isArray(manualVolumes) && manualVolumes.length > 0;
        if (hasVolumes) { box.makeEmpty(); for (const v of manualVolumes) if (v.box && !v.box.isEmpty()) box.union(v.box); }
        else if (built.bounds && !built.bounds.isEmpty()) box.copy(built.bounds);  // exact traced-soup AABB (preferred)
        else autoFitTracedBounds(box);  // fallback: scene walk honouring the BVH visibility gate
        if (box.isEmpty()) return false;

        // ── C0: coarse full-bounds grid (EXACTLY the old single-grid path). ──
        const C0 = casc[0];
        box.getSize(C0.gridSize);
        C0.gridMin.copy(box.min);
        if (!hasVolumes) {
            const pad = C0.gridSize.clone().multiplyScalar(0.06);
            C0.gridMin.sub(pad); C0.gridSize.add(pad.clone().multiplyScalar(2));
        }
        const resOverride = (hasVolumes && manualVolumes.length === 1 && manualVolumes[0].res) ? manualVolumes[0].res : null;
        C0.res.copy(resOverride ? resOverride : computeGridResolution(C0.gridSize, targetLongAxis));
        fitAtlas(C0.res);
        C0.probeTotal = C0.res.x * C0.res.y * C0.res.z;
        C0.atlasW = C0.res.x * TILE;
        C0.atlasH = C0.res.y * C0.res.z * TILE;
        C0.minCell = Math.max(1e-4, Math.min(C0.gridSize.x / Math.max(1, C0.res.x - 1), C0.gridSize.y / Math.max(1, C0.res.y - 1), C0.gridSize.z / Math.max(1, C0.res.z - 1)));
        curMinCell = C0.minCell;                                 // setNormalBias() lives off C0's cell
        quantStep = Math.max(1e-4, C0.minCell * 0.25);          // A1: geo-signature translation deadband
        lightQuant = Math.max(1e-3, C0.gridSize.length() * 0.003); // B4: light-signature position deadband

        // ── C1 dims: fine sub-box via the CPU triangle-density histogram (idle-gated). ──
        // A flat/degenerate histogram → treat as cascades=1 for placement (safe fallback).
        let wantC1 = cascades >= 2;
        if (wantC1) {
            const fine = fitFineBox(built, new THREE.Box3(C0.gridMin.clone(), C0.gridMin.clone().add(C0.gridSize)));
            if (!fine) { wantC1 = false; }
            else {
                const C1 = casc[1];
                C1.gridMin.copy(fine.min);
                C1.gridSize.copy(fine.size);
                // ~2× the coarse long-axis density over the (smaller) fine box.
                C1.res.copy(computeGridResolution(C1.gridSize, Math.min(MAX_PROBES_PER_AXIS, Math.round(targetLongAxis * C1_RES_SCALE))));
                fitAtlas(C1.res);
                C1.probeTotal = C1.res.x * C1.res.y * C1.res.z;
                C1.atlasW = C1.res.x * TILE;
                C1.atlasH = C1.res.y * C1.res.z * TILE;
                C1.minCell = Math.max(1e-4, Math.min(C1.gridSize.x / Math.max(1, C1.res.x - 1), C1.gridSize.y / Math.max(1, C1.res.y - 1), C1.gridSize.z / Math.max(1, C1.res.z - 1)));
            }
        }
        // effective cascade count for THIS build (may drop to 1 on a uniform scene).
        buildCascadeCount = wantC1 ? 2 : 1;

        // Build C0 only in stage 0. If C1 is NOT already online (first build / resize / a C1
        // dim change), force the shader single-grid (cascadeCountNode=1) so sampleIrradiance
        // never dereferences a null/stale C1 atlas until C1 comes online in stage 1
        // (invariant #6). If C1 IS already live and will be reused SAME-DIM this build, keep
        // count at 2 — dropping it would force a needless 2→1→2 recompile pair (churn) on a
        // same-dim geometry rebuild.
        const genBefore = node._structGen;
        const c1WasReady = !!casc[1].gpu;
        const c1SameDim = c1WasReady && buildCascadeCount >= 2
            && casc[1].atlasW === casc[1].prevAtlasW && casc[1].atlasH === casc[1].prevAtlasH && casc[1].probeTotal === casc[1].prevProbeTotal;
        const recompiled = buildOneCascade(built, 0);
        if (buildCascadeCount < 2) {
            // No fine cascade this build → single-grid shader + dispose C1, finish now.
            node.setCascadeCount(1);
            disposeCascadeGPU(1);
            buildStage = 2;
        } else if (c1SameDim) {
            // C1 stays live at its current dims → keep it bound (no recompile), rebuild its
            // BVH-bound kernels next idle tick without dropping the blend.
            buildStage = 1;
        } else {
            // C1 will be (re)allocated at new dims → hide it (single-grid) until stage 1.
            node.setCascadeCount(1);
            buildStage = 1;   // C1 built next idle tick (staggered — never 2× the build/frame)
        }
        reactiveTicks = REACTIVE_TICKS;
        dirty = false;
        // Fire the one-shot recompile the frame C0's data first exists (resize/first-enable
        // path OR a cascade-count change to 1). The same-dim path with no structGen change
        // needs no recompile.
        if ((recompiled || node._structGen !== genBefore) && typeof onRebuilt === 'function') { try { onRebuilt(); } catch (e) { /* non-fatal */ } }
        return true;
    }

    // STAGE 1: build C1 only (one idle tick after C0), then flip cascadeCountNode to 2 so
    // the fragment starts blending the fine cascade. Runs behind the SAME idle gate as C0.
    function advanceBuildStageC1() {
        const built = cachedBuilt;
        if (!built) { buildStage = 2; return; }
        const genBefore = node._structGen;
        const recompiled = buildOneCascade(built, 1);
        node.setCascadeCount(2);  // C1 online → fragment blends fine cascade
        buildStage = 2;
        // C1's setGrid/setAtlases + setCascadeCount bump _structGen only on the full path;
        // a same-dim C1 reuse changes nothing → no recompile needed (churn-free).
        if ((recompiled || node._structGen !== genBefore) && typeof onRebuilt === 'function') { try { onRebuilt(); } catch (e) { /* non-fatal */ } }
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
        const moving = idleMs < GI_IDLE_MS || playing;
        // Default: fully idle-gated (moving → return). Continuous mode: keep the bounded GPU
        // SOLVE running while moving, but STILL hold every synchronous/compiling step — the
        // ~200ms MeshBVH rebuild, the staggered cascade build, and the per-tick scene-signature
        // scans — for rest. Those are the only real hitch sources; the capped round-robin solve
        // is the same GPU cost that already runs smooth at rest, so it's safe every frame.
        if (moving && !continuous) return;
        const restOnly = !moving;   // gates the heavy paths; the solve runs in both modes

        // (A7) Back off after a failed/empty rebuild instead of re-entering the
        // synchronous build every tick.
        if (rebuildBackoff > 0) rebuildBackoff--;

        if (dirty || !casc[0].gpu) {
            if (!restOnly) return;   // never (re)build the soup/kernels mid-motion (hitch source)
            if (rebuildBackoff > 0) return;
            inFlight = true; let ok = false;
            try { ok = await rebuild(); } finally { inFlight = false; }
            if (!ok) { dirty = false; rebuildBackoff = REBUILD_BACKOFF_TICKS; return; }
            return;   // stage 0 done this tick; the solve waits for the next tick
        }
        if (!casc[0].gpu) return;

        // (A2/#2) Staggered build: advance exactly ONE build stage per idle tick so no
        // single frame does 2× the build. C1 comes online one idle tick after C0. Held for rest.
        if (buildStage < 2) {
            if (!restOnly) return;
            inFlight = true;
            try { advanceBuildStageC1(); } finally { inFlight = false; }
            return;
        }

        // reactivity: detect live light/geometry edits (throttled). Light change →
        // cheap in-place buffer refresh; geometry change → debounced rebuild. These walk the
        // scene graph (CPU), so in continuous mode they run ONLY at rest — never per orbit frame.
        if (restOnly) {
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

        // (#1) ALTERNATE exactly ONE cascade per tick (round-robin C0→C1). Only one
        // cascade dispatches, so total probes solved/tick == the single-grid count — the
        // per-tick GPU budget is UNCHANGED. cascades==1 always picks C0.
        const c = (cascades > 1 && casc[1].gpu) ? (solveTurn % 2) : 0;
        solveTurn = (solveTurn + 1) >>> 0;
        const C = casc[c];
        const gpu = C.gpu;
        if (!gpu) return;

        const updated = Math.min(updatedCap(), C.probeTotal);
        C.U.probeOffset.value = C.probeCursor >>> 0;
        C.U.updatedCount.value = updated >>> 0;
        // (B1) Ray-set rotation shares ONE frameJitter, advanced ONLY on C0's pass boundary
        // (and no more often than ROT_MIN_TICKS apart). Both cascades read the same
        // U.frameJitter, so rayDir(k,jitter) stays byte-identical between trace and blend
        // for whichever cascade runs. Per-cascade ticksSinceRot/refreshStarted are tracked
        // but frameCounter++ happens only at C0's boundary.
        C.ticksSinceRot++;
        if (c === 0 && C.probeCursor === 0 && C.ticksSinceRot >= ROT_MIN_TICKS) {
            if (C.refreshStarted) { frameCounter = (frameCounter + 1) >>> 0; U.frameJitter.value = (frameCounter * 0.61803398875) % 1; }
            C.refreshStarted = true;
            C.ticksSinceRot = 0;
        }
        C.probeCursor = C.probeTotal > 0 ? (C.probeCursor + updated) % C.probeTotal : 0;

        inFlight = true;
        try {
            // (A5) One-time-per-rebuild prep for cascade C ONLY. clear (full rebuild only)
            // + the cheap uploadState (ALWAYS — sole writer of stateAtlas; skipping it
            // leaves the relocation sample reading uninitialised texels → NaN). classify
            // (the 32-ray BVH walk) runs ONLY in Solid-scene mode. These WRITE buffers the
            // trace then READS, so let them finish before tracing.
            if (C.needsClear || C.needsClassify) {
                const prep = [];
                if (C.needsClear) { prep.push(renderer.computeAsync(gpu.clearAtlasKernel)); C.needsClear = false; }
                if (C.needsClassify) {
                    if (U.classifyStrength.value > 0) prep.push(renderer.computeAsync(gpu.classifyKernel));
                    prep.push(renderer.computeAsync(gpu.uploadStateKernel));
                    C.needsClassify = false;
                }
                await Promise.all(prep);
            }
            // (A6/#1) Submit trace→blend→upload for cascade C in order WITHOUT awaiting
            // between them — same-queue order preserves the data dependency, so one trailing
            // await suffices. Per-tick dispatch count == today (single cascade).
            await Promise.all([
                renderer.computeAsync(gpu.traceKernel),
                renderer.computeAsync(gpu.blendKernel),
                renderer.computeAsync(gpu.uploadKernel),
            ]);
        } catch (e) {
            console.warn('max.js SPEEDBALL GI probe tick failed:', e);
            dirty = true;
        } finally {
            inFlight = false;
        }
    }

    function setEnabled(on) { node.setEnabled(on === true); if (on && (!casc[0].gpu || !node._ready)) requestRebuild(); }
    // freshBuild=true (default) invalidates the cached BVH+texture soup so rebuild()
    // rebuilds it (geometry/light-count/volume change, or first build). Grid-only callers
    // (setDivisions/setRays) pass false → the cached soup is reused, no MeshBVH hitch.
    function requestRebuild(freshBuild = true) { dirty = true; rebuildBackoff = 0; if (freshBuild) buildDirty = true; }
    // Set explicit probe volume(s) — e.g. synced "SPEEDBALL GI Probe Grid" helpers. Each
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
    // re-collect lights into EACH cascade's light buffer (no BVH rebuild). Count change →
    // full rebuild. Both cascades bind their own copy of the light soup, so update both.
    function refreshLights() {
        if (!casc[0].gpu || !_collectLights) return;
        let records;
        try { records = _collectLights(THREE, scene); } catch { return; }
        if (records.length !== casc[0].gpu.lightCount) { requestRebuild(); return; }
        for (const C of casc) {
            const g = C.gpu; if (!g) continue;
            const arr = g.buffers.lights.array;
            arr.fill(0);
            for (let i = 0; i < records.length; i++) arr.set(records[i], i * _LIGHT_STRIDE);
            g.buffers.lights.needsUpdate = true;
        }
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
                if (U.classifyStrength.value > 0) for (const C of casc) C.needsClassify = true;
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
            // per-cascade: each cascade scales the offset by ITS OWN minCell (coarse cells larger).
            for (let c = 0; c < NUM_CASC; c++) {
                node.normalBiasNode[c].value = Math.max(1e-4, casc[c].minCell * SURFACE_NORMAL_BIAS_CELL * normalBiasScale);
            }
        },
        setRadianceClamp: (v) => { if (Number.isFinite(v)) U.radianceClamp.value = Math.max(0, v); },   // cap multibounce feedback (anti-runaway)
        setDepthSharpness: (v) => { if (Number.isFinite(v)) U.depthSharpness.value = THREE.MathUtils.clamp(v, 1, 200); }, // depth-moment cosine power (Chebyshev crispness)
        // ── adaptive-blend tuning (live; no recompile). Tune "stable continuous" by feel. ──
        setChangeThreshold: (v) => { if (Number.isFinite(v)) U.tempChangeSigma1.value = THREE.MathUtils.clamp(v, 0.5, 8); },   // σ delta to treat a change as REAL — lower = snappier
        setSnapAmount: (v) => { if (Number.isFinite(v)) U.tempChangeHDrop.value = THREE.MathUtils.clamp(v, 0, 0.9); },         // hysteresis drop on a real change — higher = harder snap
        setFireflyClamp: (v) => { if (Number.isFinite(v)) U.tempClampSigma.value = THREE.MathUtils.clamp(v, 1, 20); },         // clamp band in σ — lower = steadier, more lag
        getChangeThreshold: () => U.tempChangeSigma1.value,
        getSnapAmount: () => U.tempChangeHDrop.value,
        getFireflyClamp: () => U.tempClampSigma.value,
        // ── STRUCTURAL knob: cascade count (single grid vs cascaded). 1↔2 never changes
        // geometry, so requestRebuild(false) reuses the ~200ms BVH soup (invariant #3); the
        // change flows through the idle gate + staggered build. 1 → wFine≡0 (byte-identical).
        setCascades: (n) => {
            const v = (Math.round(Number(n)) === 1) ? 1 : 2;   // default 2
            if (v === cascades) return;
            cascades = v;
            if (v === 1) node.setCascadeCount(1);              // stop blending fine immediately
            requestRebuild(false);                             // re-fit/alloc/free C1 via the idle-gated staggered build
        },
        getCascades: () => cascades,
        // ── Continuous solve: keep updating GI while the camera moves (bounded GPU solve only;
        // heavy rebuilds still wait for rest, so the no-hitch guarantee holds). false = idle-gated.
        setContinuous: (on) => { continuous = on === true; },
        getContinuous: () => continuous,
        requestRebuild,
        setBounds,
        setVolumes,
        isSupported,
        hasData: () => node._ready === true,
        // getStats/getResolution/getBounds/_debug* take an optional cascade index
        // (default 0 = coarse, preserving current callers).
        getStats: (ci = 0) => { const C = casc[ci] || casc[0]; return { probes: C.probeTotal, res: C.res.clone(), atlas: [C.atlasW, C.atlasH], rays: raysPerProbe, oct: OCT_RES, tile: TILE, active: node.active, cascades, cascade: ci }; },
        getResolution: (ci = 0) => (casc[ci] || casc[0]).res.clone(),
        getBounds: (ci = 0) => { const C = casc[ci] || casc[0]; return new THREE.Box3(C.gridMin.clone(), C.gridMin.clone().add(C.gridSize)); },
        _debugUpload: async (ci = 0) => { const g = casc[ci]?.gpu; if (g && !disposed) { try { await renderer.computeAsync(g.uploadKernel); } catch (e) { /* harness-only */ } } },
        _debugAtlas: (ci = 0) => casc[ci]?.gpu?.atlas || null,
        _debugDepthAtlas: (ci = 0) => casc[ci]?.gpu?.depthAtlas || null,
        _debugStateAtlas: (ci = 0) => casc[ci]?.gpu?.stateAtlas || null,
        _debugStateBuffer: (ci = 0) => casc[ci]?.gpu?.stateBuffer || null,
        _debugRead: async (which, ci = 0) => {
            const g = casc[ci]?.gpu;
            const buf = which === 'irr' ? g?.irrBuffer
                : which === 'mat' ? g?.buffers?.materials
                : which === 'lights' ? g?.buffers?.lights
                : which === 'state' ? g?.stateBuffer : null;
            if (!buf || typeof renderer.getArrayBufferAsync !== 'function') return null;
            try { return new Float32Array(await renderer.getArrayBufferAsync(buf)); } catch (e) { return { error: String(e) }; }
        },
        _debugLightCount: (ci = 0) => casc[ci]?.gpu?.lightCount,
        dispose,
    };
}

export default createProbeField;
