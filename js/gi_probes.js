// gi_probes.js — SPEEDBALL GI DDGI irradiance field (docs/GI_SPEEDBALL_design.md §3).
//
// A world-space grid of octahedral irradiance probes traced against the SAME
// stackless BVH the spectral path tracer uses (shared byte-identically via
// spectral_traverse.js — no second acceleration structure). Pure WebGPU/TSL
// compute; nothing reads back to the CPU.
//
// MVP (Phase 1): single grid, trace + cosine-gather blend + temporal hysteresis,
// infinite bounce over frames (trace reads last frame's atlas), miss = SKY
// (radiance from an INJECTED SH-9 via setSky(), × skyIntensity; zero SH = black —
// then probes carry ONLY surface inter-reflection and PMREM IBL owns the sky;
// see "sky → probes" in createProbeField). Leak-free Chebyshev +
// relocation/classification and SSILVB are Phase 2/3.
//
// Churn-free by construction (fixes the surfel-grid recompile freeze):
//   • irradiance STATE lives in a read_write StorageBufferAttribute (irrBuffer).
//   • a write-only StorageTexture ATLAS is uploaded from it for HW-bilinear
//     sampling; the material's atlas binding is STABLE, so per-tick data writes
//     never change the material cache key. Only grid resize / enable flips it.

import * as THREE from 'three/webgpu';
import { LightingNode } from 'three/webgpu';
import {
    Fn, If, Loop, Return, instanceIndex, invocationLocalIndex, workgroupId, workgroupArray, workgroupBarrier,
    storage, uniform, texture, textureLevel, sharedUniformGroup, struct,
    float, int, uint, vec2, vec3, vec4, uvec2,
    max as tslMax, min as tslMin, mix, clamp, floor, normalize, dot, cross, length,
    abs as tslAbs, sqrt, cos, sin, pow, exp, smoothstep, textureStore,
    positionWorld, normalWorld, normalWorldGeometry, normalLocal, modelNormalMatrix, select,
    positionViewDirection, cameraWorldMatrix, roughness,
} from 'three/tsl';

// buildSpectralScene (pulls three-mesh-bvh) is lazy-loaded in rebuild() so
// importing this module for the GiProbeNode (e.g. from max_lights_node.js) does
// NOT drag the CPU BVH builder into that module graph.
let _buildSpectralScene = null;
let _createBlasCache = null;
let _rebindMaterialMapsArenaBuild = null;
let _collectLights = null;       // cheap light/emitter re-collect for reactivity (no BVH rebuild)
let _emissiveScaled = null;
let _LIGHT_STRIDE = 16;
import { buildTraversal, T_MAX, RAY_EPS, PI } from './spectral_traverse.js';
import { octEncodeNode, octDecodeNode } from './gi_oct.js';
import { disposeComputeNodes, disposeStorageAttribute, disposeStorageAttributes } from './webgpu_cleanup.js';

// namespace injected into the octahedral node builders (gi_oct.js).
const TSL = { float, vec2, vec3, abs: tslAbs, select, max: tslMax, normalize };
const ProbeLightingSample = struct({
    irradiance: 'vec3',
    roughRadiance: 'vec4',
}, 'SpeedballProbeLightingSample');

const OCT_RES = 6;                 // interior octahedral resolution per probe
const BORDER = 1;                  // 1px gutter on every side
const TILE = OCT_RES + 2 * BORDER; // 8×8 atlas tile
const PROBE_WORKGROUP_SIZE = 64;
// Reflection quality is structural: disabled tiers allocate/bind/dispatch nothing.
// The legacy boolean remains exact (`true` = ultra, `false` = off), while named
// tiers let applications buy only the angular bandwidth they actually need.
export const REFLECTION_QUALITY_TIERS = Object.freeze({
    off: Object.freeze({ name: 'off', rough: false, glossy: false, glossyOct: 0, glossyUpdateInterval: 0, roughnessLimit: 0 }),
    rough: Object.freeze({ name: 'rough', rough: true, glossy: false, glossyOct: 0, glossyUpdateInterval: 0, roughnessLimit: 1 }),
    high: Object.freeze({ name: 'high', rough: true, glossy: true, glossyOct: 8, glossyUpdateInterval: 2, roughnessLimit: 1 }),
    ultra: Object.freeze({ name: 'ultra', rough: true, glossy: true, glossyOct: 16, glossyUpdateInterval: 1, roughnessLimit: 1 }),
});

export function resolveReflectionQuality(reflectionQuality, roughReflections = false) {
    const fallback = roughReflections === true ? 'ultra' : 'off';
    const name = reflectionQuality == null ? fallback : String(reflectionQuality).trim().toLowerCase();
    const tier = REFLECTION_QUALITY_TIERS[name];
    if (!tier) throw new RangeError(`Unknown reflectionQuality "${reflectionQuality}". Expected off, rough, high, or ultra.`);
    return tier;
}
// Local-reflection lobe. Power 8 keeps enough support for the default 64 rays/probe
// to converge temporally, while cutting the old power-4 half-width from ~33° to
// ~24° so nearby silhouettes survive the octahedral/bilinear reconstruction.
// Evaluated as ((d²)²)² in the blend loop (no dynamic pow).
const ROUGH_SPECULAR_POWER = 8;
// A second lobe from the SAME rays gives smooth materials a useful local signal
// without a screen-space pass. Power 64 remains a probe approximation rather than
// a perfect mirror, but matches the separate 16×16 cache closely enough for metal.
const GLOSSY_SPECULAR_POWER = 64;
const ROUGH_LOBE_MIX_START = 0.22;
const ROUGH_LOBE_MIX_END = 0.58;
// Depth-proxy parallax correction is trusted only where the directional distance
// moments agree. High relative variance marks silhouettes/disocclusions, where a
// spherical proxy would bend the lookup toward an unrelated surface.
const ROUGH_PARALLAX_VAR_START = 0.02;
const ROUGH_PARALLAX_VAR_END = 0.20;
const ROUGH_PARALLAX_INSIDE_FADE = 0.12;
// Negative hitT encoding: -1 remains a true miss; -(distance + 2) marks geometry
// that must occlude DDGI depth but has no Lambert/emissive radiance for the rough
// atlas (pure metal/glass). It therefore cannot black-out the PMREM fallback.
const ROUGH_UNSHADED_T_BIAS = 2.0;
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
const STORAGE_BINDING_FALLBACK = 128 * 1024 * 1024; // WebGPU baseline maxStorageBufferBindingSize
const RAYS_PER_TICK = 98_304;       // MAX per-tick trace budget (÷ rays/probe → probes/tick).
                                    // ≈1.5k probes at 64 rays — covers the whole Sponza/city
                                    // union every tick; huge grids fall back to round-robin.
                                    // AUTO-THROTTLED down when the frame cadence slips (see
                                    // tick()) so the solve never drags the browser below 60.
const RAYS_PER_TICK_MIN = 2_048;    // responsiveness floor (32 probes @64). Weak GPUs must
                                    // be allowed below the old 16k floor instead of pinning
                                    // the viewer near 12 fps forever.
const RAYS_PER_TICK_REST_RESUME = 4_096; // reserve frame headroom for post-interaction
                                         // deform uploads/refits, then grow only if cadence allows.
const MAX_PROBES_PER_TICK = 2048;   // absolute ceiling (bounds dispatch + ray scratch)
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
const GI_EMITTER_INJECT_CAP = 16;
const GI_EMITTER_VIS_RETENTION = 0.8;
const GI_LIGHTS_PER_CELL = 16;
const GI_LIGHT_CELL_STRIDE = GI_LIGHTS_PER_CELL + 1;
const GI_LIGHT_CELL_OVERFLOW = GI_LIGHTS_PER_CELL + 1;
const giLightDataCount = () => MAX_LIGHTS * _LIGHT_STRIDE;
// ── reactivity: respond to live light/geometry edits ──
// There is deliberately NO global reactive/low-hysteresis burst here. Edits
// (lights, sky, transforms, trace-side knobs, rebuilds) flow through the trace
// and re-converge via the per-texel change detector in the blend, whose retention
// is bounded (never below debugTempMinChangeH). A global authority drop read as
// "flicker for a second, then settle" — a visible fade-out/in on every slider
// touch. The temporal policy is CONSTANT at all times.
const JITTER_HYSTERESIS_DEFAULTS = Object.freeze({
    gated: 0.60,                   // held basis: low-latency, mostly flicker-free response
    montecarlo: 0.90,              // fresh basis every solve: history absorbs the sample blast
});
const LIGHT_CHECK_INTERVAL = 6;    // ticks between light-change checks
const XFORM_CHECK_INTERVAL = 2;    // ticks between transform checks (in-place TLAS rewrite = cheap)
const GEO_CHECK_INTERVAL = 24;     // ticks between STRUCTURE checks (topology → full rebuild = expensive)
const GEO_SETTLE_INTERVALS = 2;    // structure must be stable this many checks before a rebuild fires (debounce)
const DEFORM_CHECK_INTERVAL = 12;  // ticks between DEFORM checks (same-topology vertex motion → in-place
                                   // gather+refit via built.updateDeforms, no MeshBVH, no debounce needed —
                                   // so streamed vertex animation tracks GI without EVER arming the rebuild)
// ── freeze-proofing: gate the synchronous BVH rebuild + GPU solve on viewport idle ──
// The CPU MeshBVH build in rebuild() blocks the render thread, so it (and the GPU
// solve) must NEVER land while the user is orbiting, the timeline is playing, or a
// delta-sync burst is in flight. GI is world-space, so holding the field static
// during motion is visually lossless; it resumes and converges once the view rests.
const GI_IDLE_MS = 200;            // ms of camera/sync quiet before GI work resumes
const REBUILD_BACKOFF_TICKS = 45;  // ticks to wait after a failed/empty rebuild before retrying
const TICK_OVERLOAD_MS = 100;      // outside the normal EMA window; require repeated misses
const TICK_PAUSE_MS = 1000;        // tab/debugger/host gaps are pauses, not solve pressure
const TICK_OVERLOAD_STRIKES = 2;   // ignore one unrelated stall; back off if it repeats
const PROBE_COMPUTE_KEYS = [
    'traceKernel', 'emitterVisKernel', 'blendKernel', 'glossyKernel', 'uploadKernel', 'lightGridKernel',
    'clearAtlasKernel', 'clearGlossyAtlasKernel', 'clearEmitterVisKernel', 'classifyKernel', 'uploadStateKernel',
];
const PROBE_SCENE_BUFFER_KEYS = ['bvhNodes', 'triIndex', 'vertexData', 'triMaterial', 'materials'];
const PROBE_SCENE_STORAGE_GROWTH = 1.5;
const PROBE_MAP_KEYS = ['albedo', 'normal', 'roughness', 'metalness', 'emissive', 'alpha'];
const TEXTURE_ARRAY_LAYERS_FALLBACK = 256;
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
export const HYSTERESIS_DT_REF_MS = 1000 / 60; // slider values are reference-rate retentions at 60 Hz

// Average revisit cadence for one probe. Fractional pass lengths are intentional:
// round-robin batches alternate floor/ceil revisit counts, and their long-term temporal
// rate is probeTotal / probesPerTick. Keeping the average continuous also prevents a
// ray-budget change from stepping the history coefficient at every ceil() boundary.
export function probeUpdateIntervalTicks(probeTotal, probesPerTick, solveEveryTicks = 1) {
    const total = Math.max(0, Number(probeTotal) || 0);
    const cap = Math.max(1, Number(probesPerTick) || 1);
    const schedule = Math.max(1, Number(solveEveryTicks) || 1);
    return schedule * Math.max(1, total / cap);
}

// Clamp a stale high solve budget when an interaction becomes idle. This is a
// pure helper so the post-scrub responsiveness contract stays deterministic in
// smoke tests without requiring Three/WebGPU.
export function probeBudgetAfterInteraction(
    currentBudget,
    minBudget = RAYS_PER_TICK_MIN,
    resumeBudget = RAYS_PER_TICK_REST_RESUME,
) {
    const min = Math.max(1, Math.floor(Number(minBudget) || 1));
    const resume = Math.max(min, Math.floor(Number(resumeBudget) || min));
    const current = Math.max(min, Math.floor(Number(currentBudget) || min));
    return Math.min(current, resume);
}

// One accepted solve interval represents real pressure from the work submitted by
// the previous accepted tick. Shrink immediately on a cadence miss instead of
// waiting for an EMA tail; the controller's cooldown keeps the budget from
// bouncing straight back up. Kept pure for source-only smoke coverage.
export function probeBudgetAfterCadenceMiss(
    currentBudget,
    minBudget = RAYS_PER_TICK_MIN,
    shrinkFactor = 0.5,
) {
    const min = Math.max(1, Math.floor(Number(minBudget) || 1));
    const current = Math.max(min, Math.floor(Number(currentBudget) || min));
    const factor = Number.isFinite(shrinkFactor)
        ? Math.min(0.95, Math.max(0.05, shrinkFactor))
        : 0.5;
    return Math.max(min, Math.floor(current * factor));
}

export function hysteresisExponentForInterval(updateDtMs, normalize = true) {
    if (!normalize) return 1;
    const dt = Number.isFinite(updateDtMs) ? Math.max(0, updateDtMs) : HYSTERESIS_DT_REF_MS;
    // Fast side (dt < ref): UNCLAMPED. h^(dt/ref) → 1 as the update rate rises, so
    // 120/240/480 Hz cadences consume the same fresh Monte-Carlo noise per second
    // as 60 Hz instead of boiling harder the faster the machine renders.
    // Slow side (dt > ref): CLAMPED at 1 — the flicker-proof bound. Every per-texel
    // policy is a reference-domain retention r applied as r^exponent; exponent ≤ 1
    // guarantees r^exponent ≥ r, so ONE update can never blend in more fresh noise
    // than the slider admits at 60 Hz, no matter how sparse the service cadence gets
    // (low FPS, throttled ray budgets, round-robin revisits, alternating cascades).
    // Sparse service therefore converges SLOWER in wall-clock, never noisier: an
    // unclamped exponent kept per-second decay exactly rate-invariant, but paid for
    // it by dissolving the field into flicker exactly when the machine struggled.
    // Bounded per-update variance is the invariant; convergence speed is the slack.
    return Math.min(1, dt / HYSTERESIS_DT_REF_MS);
}

let _node = null;
let _activeProbeFieldOwner = null;

const _nowMs = () => (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();

// ── fragment-uniform liveness ──
// The GI term is injected by the LIGHTS node, not the material, so GI-lit materials
// have no custom material nodes → NodeMaterialObserver.needsRefresh() sees a plain
// material and SKIPS binding updates while nothing it tracks (matrices/material
// props/lights) changes. Plain per-object uniforms therefore go stale at rest —
// setIntensity/setEnabled "did nothing" until a recompile or camera move. A SHARED
// uniform group is the designed escape hatch: one bind group cached across all
// render objects (same mechanism as three's renderGroup), re-uploaded when its
// version bumps. Every fragment-side setter bumps it via touchGiUniforms().
const GI_UNIFORM_GROUP = sharedUniformGroup('speedballGi');
const touchGiUniforms = () => { GI_UNIFORM_GROUP.needsUpdate = true; };

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
        // Per-cascade atlas sets (index 0 = C0 coarse, 1 = C1 fine). SEPARATE atlases
        // per cascade (not a shared pack): each runs the existing atlas-fit math verbatim.
        this._atlas = [null, null];
        this._roughSpecularAtlas = [null, null];
        this._glossySpecularAtlas = [null, null];
        this._depthAtlas = [null, null];
        this._stateAtlas = [null, null];
        this._enabled = false;
        this._structGen = 0;     // bumps on grid resize / atlas realloc / cascade-count change ONLY
        this._roughReflectionsConfigured = false;
        this._glossyReflectionsConfigured = false;
        this._glossyOctRes = 1;
        this._glossyTile = 1;
        this.intensity = 1.0;

        // Per-cascade grid uniforms (2-slot arrays).
        this.gridMinNode = [uniform(new THREE.Vector3()), uniform(new THREE.Vector3())];
        this.gridSizeNode = [uniform(new THREE.Vector3(1, 1, 1)), uniform(new THREE.Vector3(1, 1, 1))];
        this.resNode = [uniform(new THREE.Vector3(2, 2, 2)), uniform(new THREE.Vector3(2, 2, 2))];
        this.atlasDimNode = [uniform(new THREE.Vector2(1, 1)), uniform(new THREE.Vector2(1, 1))];
        this.glossyAtlasDimNode = [uniform(new THREE.Vector2(1, 1)), uniform(new THREE.Vector2(1, 1))];
        this.glossyTilesXNode = [uniform(1.0), uniform(1.0)];
        // Per-cascade biases: coarse cells are larger → its own normal/cheby bias.
        this.normalBiasNode = [uniform(0.04), uniform(0.04)];
        this.chebyBiasNode = [uniform(0.0), uniform(0.0)];
        // Debug receiver sampling. debug/city_debug.html can push these around without
        // saving user settings or rebuilding the scene.
        this.samplePositionScaleNode = uniform(1.0);
        this.sampleNormalMixNode = uniform(1.0); // 0 = geometry normal, 1 = shading normal
        this.sampleBiasScaleNode = uniform(1.0);
        // Receiver sampling GEOMETRY (bias point + visibility weights) must be
        // camera-invariant. Three's normalWorld/normalWorldGeometry reconstruct through
        // cameraViewMatrix, which wobbles on rotation and makes the sample bias flicker
        // like GI/wall z-fighting. Applies ONLY to the bias/weight normal — the
        // irradiance FETCH direction always uses the detailed shading normal (see setup).
        this.sampleObjectNormalNode = uniform(1.0); // 1 = object normal matrix, bypass camera-view normal reconstruction
        // How much normal-map detail the GI fetch direction carries on TRUSTED
        // (tangent-TBN) materials: 0 = smooth vertex normal, 1 = full shading normal.
        // Flicker-safe at ANY value — both mix inputs are camera-invariant there, and
        // untrusted materials ignore this entirely (always the stable normal).
        this.detailStrengthNode = uniform(1.0);

        // Cascade-invariant look uniforms (single instance).
        this.intensityNode = uniform(1.0);
        // Local-reflection strength is a coverage/composite weight, not a radiance
        // multiplier. 0 preserves stock PMREM exactly; 1 lets local probe hits replace
        // PMREM in their reflected lobe. The field is opt-in at allocation time.
        this.reflectionIntensityNode = uniform(1.0);
        // Materials rougher than this keep the stock environment and skip all local
        // reflection parallax/atlas work. Pinned at 1 (full range) unless the
        // roughnessLimit API overrides it.
        this.roughnessLimitNode = uniform(1.0);
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

        // ALL fragment-side uniforms live in the shared GI group (see GI_UNIFORM_GROUP):
        // per-object uniforms go stale at rest because the material observer skips
        // binding updates for plain (non-node) materials.
        for (const u of [
            ...this.gridMinNode, ...this.gridSizeNode, ...this.resNode, ...this.atlasDimNode,
            ...this.glossyAtlasDimNode, ...this.glossyTilesXNode,
            ...this.normalBiasNode, ...this.chebyBiasNode,
            this.samplePositionScaleNode, this.sampleNormalMixNode, this.sampleBiasScaleNode, this.sampleObjectNormalNode,
            this.detailStrengthNode,
            this.intensityNode, this.reflectionIntensityNode, this.roughnessLimitNode,
            this.enabledNode, this.cascadeCountNode, this.borderBandNode,
            this.chebyStrengthNode, this.classifyStrengthNode,
        ]) u.setGroup(GI_UNIFORM_GROUP);
    }

    // computed readiness: every cascade in [0..count) has its required atlas set.
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
    get roughReflectionsReady() {
        if (!this._roughReflectionsConfigured) return false;
        const count = Math.round(this.cascadeCountNode.value) || 1;
        for (let c = 0; c < count; c++) {
            if (!this._roughSpecularAtlas[c]) return false;
            if (this._glossyReflectionsConfigured && !this._glossySpecularAtlas[c]) return false;
        }
        return true;
    }
    // structure-only token: data writes (textureStore) do NOT change this, so
    // materials never recompile on a probe tick — only on resize / first data.
    get cacheToken() { return `gi-speedball-probes:${this._structGen}`; }

    setEnabled(on) { this._enabled = on === true; this.enabledNode.value = this._enabled ? 1.0 : 0.0; touchGiUniforms(); }
    setIntensity(v) {
        this.intensity = Number.isFinite(v) ? Math.max(0, v) : 0;
        this.intensityNode.value = this.intensity;
        touchGiUniforms();
    }
    setReflectionIntensity(v) {
        this.reflectionIntensityNode.value = Number.isFinite(v) ? THREE.MathUtils.clamp(v, 0, 1) : 0;
        touchGiUniforms();
    }
    setRoughnessLimit(v) {
        this.roughnessLimitNode.value = Number.isFinite(v) ? THREE.MathUtils.clamp(v, 0, 1) : 1;
        touchGiUniforms();
    }
    setReflectionConfig(config) {
        const rough = config?.rough === true;
        const glossy = rough && config?.glossy === true;
        const glossyOct = glossy ? Math.max(1, Math.round(config.glossyOct) || 1) : 1;
        const glossyTile = glossy ? glossyOct + 2 * BORDER : 1;
        const changed = rough !== this._roughReflectionsConfigured
            || glossy !== this._glossyReflectionsConfigured
            || glossyOct !== this._glossyOctRes;
        this._roughReflectionsConfigured = rough;
        this._glossyReflectionsConfigured = glossy;
        this._glossyOctRes = glossyOct;
        this._glossyTile = glossyTile;
        if (changed) this._structGen++;
    }
    setChebyStrength(v) { if (Number.isFinite(v)) { this.chebyStrengthNode.value = THREE.MathUtils.clamp(v, 0, 1); touchGiUniforms(); } }
    setDetailStrength(v) { if (Number.isFinite(v)) { this.detailStrengthNode.value = THREE.MathUtils.clamp(v, 0, 1); touchGiUniforms(); } }
    setClassifyStrength(v) { if (Number.isFinite(v)) { this.classifyStrengthNode.value = THREE.MathUtils.clamp(v, 0, 1); touchGiUniforms(); } }
    // Active-cascade count (fragment-visible): 1 → wFine≡0 (byte-identical), 2 → blend.
    setCascadeCount(n) {
        const v = (Math.round(Number(n)) === 2) ? 2 : 1;
        if (this.cascadeCountNode.value === v) return;
        this.cascadeCountNode.value = v;
        touchGiUniforms();
        this._structGen++;   // shader tap count changes → one recompile
    }
    setAtlases(c, atlas, depthAtlas, stateAtlas, roughSpecularAtlas = null, glossySpecularAtlas = null) {
        const a = atlas || null, d = depthAtlas || null, s = stateAtlas || null;
        const rs = roughSpecularAtlas || null, gs = glossySpecularAtlas || null;
        // Bump only on a real identity change. A no-op call (e.g. disposing an
        // already-empty cascade on every cascades=1 rebuild) must NOT move the
        // cacheToken — that fired the whole-scene material-dirty pass per
        // rebuild and was the dominant churn hitch.
        const changed = this._atlas[c] !== a || this._depthAtlas[c] !== d || this._stateAtlas[c] !== s
            || this._roughSpecularAtlas[c] !== rs || this._glossySpecularAtlas[c] !== gs;
        this._atlas[c] = a;
        this._roughSpecularAtlas[c] = rs;
        this._glossySpecularAtlas[c] = gs;
        this._depthAtlas[c] = d;
        this._stateAtlas[c] = s;
        if (changed) this._structGen++;
    }
    // Update the grid placement uniforms only. Uniform .value writes do NOT change
    // a material cache key, so this is churn-free — the same-dim rebuild path uses
    // it to re-place probes after a geometry edit WITHOUT a TSL recompile.
    updateGridUniforms(c, gridMin, gridSize, res, atlasW, atlasH, glossyAtlasW, glossyAtlasH, glossyTilesX, normalBias, chebyBias) {
        this.gridMinNode[c].value.copy(gridMin);
        this.gridSizeNode[c].value.copy(gridSize);
        this.resNode[c].value.copy(res);
        this.atlasDimNode[c].value.set(atlasW, atlasH);
        this.glossyAtlasDimNode[c].value.set(glossyAtlasW, glossyAtlasH);
        this.glossyTilesXNode[c].value = Math.max(1, glossyTilesX);
        if (Number.isFinite(normalBias)) this.normalBiasNode[c].value = Math.max(1e-4, normalBias);
        if (Number.isFinite(chebyBias)) this.chebyBiasNode[c].value = Math.max(0, chebyBias);
        touchGiUniforms();
    }
    setGrid(c, gridMin, gridSize, res, atlasW, atlasH, glossyAtlasW, glossyAtlasH, glossyTilesX, normalBias, chebyBias) {
        this.updateGridUniforms(c, gridMin, gridSize, res, atlasW, atlasH, glossyAtlasW, glossyAtlasH, glossyTilesX, normalBias, chebyBias);
        this._structGen++;   // resize/first-enable ONLY → cacheToken moves → one recompile
    }

    // world position of grid probe (px,py,pz) in cascade c.
    _probePos(px, py, pz, c) {
        const f = vec3(px, py, pz).div(this.resNode[c].sub(1.0).max(vec3(1.0)));
        return this.gridMinNode[c].add(f.mul(this.gridSizeNode[c]));
    }

    // tile-local atlas uv for probe (col,row) at octahedral coord octUV in cascade c.
    _tileUV(col, row, octUV, c) {
        // Interior texels are generated at (i + 0.5) / OCT_RES. Mapping that
        // coordinate back to texel space is therefore BORDER + uv * OCT_RES;
        // adding another half texel shifts the fold endpoints asymmetrically into
        // an interior texel on one side and the gutter on the other, producing the
        // world-normal quadrant cross on smooth surfaces.
        const ox = col.mul(float(TILE)).add(float(BORDER)).add(octUV.x.mul(float(OCT_RES)));
        const oy = row.mul(float(TILE)).add(float(BORDER)).add(octUV.y.mul(float(OCT_RES)));
        return vec2(ox.div(this.atlasDimNode[c].x), oy.div(this.atlasDimNode[c].y));
    }

    // The glossy cache is packed independently in near-square tile rows so its
    // larger glossy tiles never inherit the diffuse atlas' tall resY×resZ layout.
    _glossyTileUV(px, py, pz, octUV, c) {
        const probeIndex = px.add(py.mul(this.resNode[c].x)).add(pz.mul(this.resNode[c].x.mul(this.resNode[c].y)));
        const col = probeIndex.mod(this.glossyTilesXNode[c]);
        const row = floor(probeIndex.div(this.glossyTilesXNode[c]));
        const ox = col.mul(float(this._glossyTile)).add(float(BORDER)).add(octUV.x.mul(float(this._glossyOctRes)));
        const oy = row.mul(float(this._glossyTile)).add(float(BORDER)).add(octUV.y.mul(float(this._glossyOctRes)));
        return vec2(ox.div(this.glossyAtlasDimNode[c].x), oy.div(this.glossyAtlasDimNode[c].y));
    }

    // sample the probe field at world (P, Nvis, Ndir): trilinear over the 8 cage
    // probes, each fetched octahedrally in the Ndir direction and weighted by a
    // depth-moment Chebyshev visibility test (leak-free through thin walls).
    // Nvis (camera-invariant) drives the weights — the Chebyshev select is a hard
    // boundary, so any camera-dependent wobble there flickers like z-fighting.
    // Ndir (detailed shading normal) drives ONLY the fetch direction — irradiance
    // is cosine-convolved + bilinear over the oct tile, so wobble there stays smooth.
    // dualDetail: fetch the atlas at BOTH directions per tap and blend the fetched
    // irradiance by detailStrength. Strength in the ANGLE domain saturates perceptually
    // almost immediately (the detail pattern is fully visible after a few degrees of
    // steering on a low-resolution oct tile), so the knob felt like a toggle; blending the
    // IRRADIANCE is linear in the visible contrast by construction — 0.5 really is
    // half the detail. Emitted only for trusted materials (8 extra small-texture taps).
    // ONE trilinear tap of cascade c at probe coords (px,py,pz), trilinear weight wTri:
    // meta/relocation fetch, Chebyshev visibility, wrap weight, irradiance fetch.
    // Shared VERBATIM by the unrolled single-grid gather and the looped cascaded gather
    // so the two paths can never drift. Pure expression — legal in both contexts.
    // lod0: fetch with EXPLICIT level 0 (textureSampleLevel). Required when the gather
    // runs inside a NON-UNIFORM branch (the cascaded receiver skips whole cascades per
    // pixel), where WGSL forbids implicit-derivative sampling. The atlases have no
    // mips, so level 0 is what implicit LOD resolves to anyway — visually identical.
    _tapEW(c, P, Nn, octN, octNs, px, py, pz, wTri, lod0 = false) {
        const fetch = (tex, uv) => (lod0 ? textureLevel(tex, uv, float(0.0)) : texture(tex, uv));
        const col = px;
        const row = pz.mul(this.resNode[c].y).add(py);

        // per-probe meta (NEAREST): R=state, GBA=relocation offset. Gated by
        // classifyStrength (default 0 = ignored) — relocation/classification by a
        // backface test misreads thin 2-sided walls, so it's opt-in for solid scenes.
        const metaUV = vec2(col.add(0.5).div(this.resNode[c].x), row.add(0.5).div(this.resNode[c].y.mul(this.resNode[c].z)));
        const meta = fetch(this._stateAtlas[c], metaUV);
        const stateV = meta.x;
        const reloc = vec3(meta.y, meta.z, meta.w).mul(this.classifyStrengthNode);

        // Chebyshev visibility: relocated probe → surface direction vs stored depth.
        const probePos = this._probePos(px, py, pz, c).add(reloc);
        const toSurf = P.sub(probePos);
        const dist = length(toSurf);
        const octD = octEncodeNode(toSurf.div(dist.max(float(1e-6))), TSL);
        const m = fetch(this._depthAtlas[c], this._tileUV(col, row, octD, c));
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
        const eD = fetch(this._atlas[c], this._tileUV(col, row, octN, c)).xyz;
        const eS = octNs ? fetch(this._atlas[c], this._tileUV(col, row, octNs, c)).xyz : null;
        const e = eS ? mix(eS, eD, this.detailStrengthNode) : eD;
        // probePos is returned for the optional specular depth-proxy correction. It
        // reuses this exact relocated origin instead of fetching probe meta twice.
        return { e, w, probePos };
    }

    // Sample ONE cascade c (0=coarse, 1=fine) — the original 8-tap trilinear gather,
    // parameterized per cascade. PURE-EXPRESSION (no toVar/addAssign): fragment colorNode.
    _sampleCascade(P, Nvis, Ndir, c, dualDetail = false) {
        const res = this.resNode[c];
        const cell = this.gridSizeNode[c].div(res.sub(1.0).max(vec3(1.0)));
        const gridF = P.sub(this.gridMinNode[c]).div(cell.max(vec3(1e-6)));
        const baseF = gridF.floor().clamp(vec3(0.0), res.sub(2.0).max(vec3(0.0)));
        const frac = gridF.sub(baseF).clamp(0.0, 1.0);
        const Nn = Nvis.normalize();
        const octN = octEncodeNode(Ndir.normalize(), TSL); // irradiance dir = detailed shading normal
        const octNs = dualDetail ? octEncodeNode(Nn, TSL) : null; // smooth dir, for the detail-strength blend

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
            const wx = dx ? frac.x : float(1.0).sub(frac.x);
            const wy = dy ? frac.y : float(1.0).sub(frac.y);
            const wz = dz ? frac.z : float(1.0).sub(frac.z);
            const wTri = wx.mul(wy).mul(wz).add(1e-4);
            const { e, w } = this._tapEW(c, P, Nn, octN, octNs, px, py, pz, wTri);
            acc = acc.add(e.mul(w));
            wsum = wsum.add(w);
        }
        return acc.div(wsum.max(float(1e-4)));
    }

    // Looped gather for the CASCADED receiver: the exact per-tap math (_tapEW), but
    // emitted as a real WGSL loop inside an Fn instead of 8 inlined taps. Two inlined
    // 8-tap subtrees (16 taps) overflow WebKit's 8192-byte budget for pipeline-variable
    // storage (Safari: "combined byte size of all variables in the private address
    // space exceeds 8192 bytes" → invalid RenderPipeline on every GI material). A loop
    // keeps ONE tap's variables live regardless of tap count, and the Fn wrapper gives
    // toVar/addAssign proper statement sequencing (the raw fragment expression context
    // does not — see _sampleCascade). The single-grid path intentionally stays on the
    // unrolled pure-expression gather: it is the shipped default, byte-identical.
    _sampleCascadeLooped(P, Nvis, Ndir, c) {
        return Fn(() => {
            const res = this.resNode[c];
            const cell = this.gridSizeNode[c].div(res.sub(1.0).max(vec3(1.0)));
            // Probe interpolation is a spatial lookup and must use the actual surface
            // position. P carries the outward normal bias exclusively for visibility;
            // letting it drive gridF makes the selected cage move when normal bias moves.
            const gridF = reflectionP.sub(this.gridMinNode[c]).div(cell.max(vec3(1e-6)));
            const baseF = gridF.floor().clamp(vec3(0.0), res.sub(2.0).max(vec3(0.0)));
            const frac = gridF.sub(baseF).clamp(0.0, 1.0);
            const Nn = Nvis.normalize();
            const octN = octEncodeNode(Ndir.normalize(), TSL);
            const acc = vec3(0.0).toVar();
            const wsum = float(0.0).toVar();
            Loop({ start: uint(0), end: uint(8), type: 'uint', condition: '<' }, ({ i }) => {
                const dx = i.bitAnd(uint(1)).toFloat();
                const dy = i.shiftRight(uint(1)).bitAnd(uint(1)).toFloat();
                const dz = i.shiftRight(uint(2)).bitAnd(uint(1)).toFloat();
                const px = baseF.x.add(dx);
                const py = baseF.y.add(dy);
                const pz = baseF.z.add(dz);
                const wx = mix(float(1.0).sub(frac.x), frac.x, dx);
                const wy = mix(float(1.0).sub(frac.y), frac.y, dy);
                const wz = mix(float(1.0).sub(frac.z), frac.z, dz);
                const wTri = wx.mul(wy).mul(wz).add(1e-4);
                const { e, w } = this._tapEW(c, P, Nn, octN, null, px, py, pz, wTri, true);
                acc.addAssign(e.mul(w));
                wsum.addAssign(w);
            });
            return acc.div(wsum.max(float(1e-4)));
        })();
    }

    // Opt-in physical receiver: gather diffuse irradiance and local reflection radiance
    // together. State/visibility are evaluated ONCE per probe; reflection pixels reuse
    // the depth atlas for parallax and fetch one or both roughness lobes. Intensity 0
    // dynamically skips those taps, while roughReflections:false emits none of this path.
    _sampleCombinedCascadeLooped(P, reflectionP, Nvis, Ndir, Rdir, reflectionWeight, roughLobeMix, c, dualDetail = false) {
        return Fn(() => {
            const res = this.resNode[c];
            const cell = this.gridSizeNode[c].div(res.sub(1.0).max(vec3(1.0)));
            const gridF = P.sub(this.gridMinNode[c]).div(cell.max(vec3(1e-6)));
            const baseF = gridF.floor().clamp(vec3(0.0), res.sub(2.0).max(vec3(0.0)));
            const frac = gridF.sub(baseF).clamp(0.0, 1.0);
            const Nn = Nvis.normalize();
            const octN = octEncodeNode(Ndir.normalize(), TSL);
            const octNs = dualDetail ? octEncodeNode(Nn, TSL) : null;
            const Rn = Rdir.normalize();
            const octR = octEncodeNode(Rn, TSL);
            const acc = vec3(0.0).toVar();
            const roughAcc = vec4(0.0).toVar();
            const glossyAcc = this._glossyReflectionsConfigured ? vec4(0.0).toVar() : null;
            const wsum = float(0.0).toVar();
            const roughWsum = float(0.0).toVar();
            const glossyWsum = this._glossyReflectionsConfigured ? float(0.0).toVar() : null;
            Loop({ start: uint(0), end: uint(8), type: 'uint', condition: '<' }, ({ i }) => {
                const dx = i.bitAnd(uint(1)).toFloat();
                const dy = i.shiftRight(uint(1)).bitAnd(uint(1)).toFloat();
                const dz = i.shiftRight(uint(2)).bitAnd(uint(1)).toFloat();
                const px = baseF.x.add(dx);
                const py = baseF.y.add(dy);
                const pz = baseF.z.add(dz);
                const wx = mix(float(1.0).sub(frac.x), frac.x, dx);
                const wy = mix(float(1.0).sub(frac.y), frac.y, dy);
                const wz = mix(float(1.0).sub(frac.z), frac.z, dz);
                const wTri = wx.mul(wy).mul(wz).add(1e-4);
                const { e, w, probePos } = this._tapEW(c, P, Nn, octN, octNs, px, py, pz, wTri, true);
                acc.addAssign(e.mul(w));
                wsum.addAssign(w);
                const wantsRough = roughLobeMix.greaterThan(float(0.0));
                const wantsGlossy = this._glossyReflectionsConfigured
                    ? roughLobeMix.lessThan(float(1.0))
                    : null;
                const wantsThisTap = this._glossyReflectionsConfigured ? wantsRough.or(wantsGlossy) : wantsRough;
                If(reflectionWeight.greaterThan(float(0.0)).and(wantsThisTap), () => {
                    const row = pz.mul(res.y).add(py);
                    // Parallax-aware probe lookup. The depth atlas gives a directional
                    // mean radius around this probe. Intersect the fragment reflection
                    // ray reflectionP+tR with that sphere, then fetch radiance in the direction
                    // from the actual probe origin to the shared hit proxy. This makes
                    // neighbouring probes agree on local silhouettes instead of all
                    // sampling the same world direction from different origins.
                    const depthR = textureLevel(this._depthAtlas[c], this._tileUV(px, row, octR, c), float(0.0));
                    const radius = depthR.x.max(float(1e-4));
                    const radius2 = radius.mul(radius);
                    const q = reflectionP.sub(probePos);
                    const q2 = dot(q, q);
                    const qDotR = dot(q, Rn);
                    const disc = radius2.sub(q2).add(qDotR.mul(qDotR));
                    const sqrtDisc = sqrt(disc.max(float(0.0)));
                    const nearT = qDotR.negate().sub(sqrtDisc);
                    const farT = qDotR.negate().add(sqrtDisc);
                    const rayT = select(nearT.greaterThan(float(1e-4)), nearT, farT);
                    const relVar = tslMax(depthR.y.sub(radius2), float(0.0)).div(radius2.max(float(1e-4)));
                    const stableDepth = float(1.0).sub(smoothstep(
                        float(ROUGH_PARALLAX_VAR_START),
                        float(ROUGH_PARALLAX_VAR_END),
                        relVar,
                    ));
                    const validHit = select(
                        disc.greaterThan(float(1e-6)).and(rayT.greaterThan(float(1e-4))),
                        float(1.0),
                        float(0.0),
                    );
                    // The spherical proxy is only valid while it encloses P. Fade the
                    // correction near/outside that boundary instead of letting the far
                    // quadratic root bend a lookup through unrelated geometry.
                    const insideRatio = radius2.sub(q2).div(radius2.max(float(1e-4)));
                    const insideWeight = smoothstep(float(0.0), float(ROUGH_PARALLAX_INSIDE_FADE), insideRatio);
                    const parallaxWeight = stableDepth.mul(validHit).mul(insideWeight);
                    const correctedR = normalize(q.add(Rn.mul(rayT.max(float(0.0)))));
                    const sampleR = normalize(mix(Rn, correctedR, parallaxWeight));
                    const octSampleR = octEncodeNode(sampleR, TSL);
                    // Fetch only the lobe(s) the current roughness needs. Glossy uses
                    // its independent high-resolution packing plus a tighter spatial
                    // weight; rough keeps the stable low-resolution DDGI blend.
                    const reflectionProbeW = w.mul(w);
                    if (this._glossyReflectionsConfigured) {
                        If(wantsGlossy, () => {
                            const glossy = textureLevel(
                                this._glossySpecularAtlas[c],
                                this._glossyTileUV(px, py, pz, octSampleR, c),
                                float(0.0),
                            );
                            // SSR-style confidence idea without screen-space dependence:
                            // prefer probes whose depth proxy supports the reprojected hit.
                            const confidence = mix(float(0.2), float(1.0), parallaxWeight);
                            const glossyProbeW = reflectionProbeW.mul(confidence);
                            glossyAcc.addAssign(glossy.mul(glossyProbeW));
                            glossyWsum.addAssign(glossyProbeW);
                        });
                    }
                    If(roughLobeMix.greaterThan(float(0.0)), () => {
                        const broad = textureLevel(this._roughSpecularAtlas[c], this._tileUV(px, row, octSampleR, c), float(0.0));
                        roughAcc.addAssign(broad.mul(reflectionProbeW));
                        roughWsum.addAssign(reflectionProbeW);
                    });
                });
            });
            const den = wsum.max(float(1e-4));
            const roughResolved = roughAcc.div(roughWsum.max(float(1e-4)));
            const reflectionResolved = this._glossyReflectionsConfigured
                ? mix(glossyAcc.div(glossyWsum.max(float(1e-4))), roughResolved, roughLobeMix)
                : roughResolved;
            return ProbeLightingSample({
                irradiance: acc.div(den),
                roughRadiance: reflectionResolved,
            });
        })();
    }

    sampleIrradianceAndRough(P, reflectionP, Nvis, Ndir, Rdir, reflectionWeight, roughLobeMix, dualDetail = false) {
        const useFine = Math.round(this.cascadeCountNode.value) >= 2
            && !!this._atlas[1] && !!this._roughSpecularAtlas[1]
            && (!this._glossyReflectionsConfigured || !!this._glossySpecularAtlas[1])
            && !!this._depthAtlas[1] && !!this._stateAtlas[1];
        if (!useFine) return this._sampleCombinedCascadeLooped(P, reflectionP, Nvis, Ndir, Rdir, reflectionWeight, roughLobeMix, 0, dualDetail);

        return Fn(() => {
            // Cascade ownership is spatial too; normal bias must not move a receiver
            // across the fine-volume blend band.
            const f = reflectionP.sub(this.gridMinNode[1]).div(this.gridSizeNode[1].max(vec3(1e-6)));
            const fLo = tslMin(tslMin(f.x, f.y), f.z);
            const fHi = tslMin(tslMin(float(1.0).sub(f.x), float(1.0).sub(f.y)), float(1.0).sub(f.z));
            const edge = tslMin(fLo, fHi);
            const wIn = clamp(edge.div(this.borderBandNode.max(float(1e-4))), float(0.0), float(1.0));
            const gate = this.cascadeCountNode.sub(1.0).clamp(float(0.0), float(1.0));
            const wFine = wIn.mul(gate).toVar();
            const E = vec3(0.0).toVar();
            const S = vec4(0.0).toVar();
            If(wFine.lessThan(float(1.0)), () => {
                const coarse = this._sampleCombinedCascadeLooped(P, reflectionP, Nvis, Ndir, Rdir, reflectionWeight, roughLobeMix, 0);
                E.assign(coarse.get('irradiance'));
                S.assign(coarse.get('roughRadiance'));
            });
            If(wFine.greaterThan(float(0.0)), () => {
                const fine = this._sampleCombinedCascadeLooped(P, reflectionP, Nvis, Ndir, Rdir, reflectionWeight, roughLobeMix, 1);
                E.assign(mix(E, fine.get('irradiance'), wFine));
                S.assign(mix(S, fine.get('roughRadiance'), wFine));
            });
            return ProbeLightingSample({ irradiance: E, roughRadiance: S });
        })();
    }

    // Cascaded sample: coarse cascade, blended toward the fine cascade across a narrow
    // inner border when P is inside C1 and cascades>=2. Each cascade is sampled ONLY
    // where its blend weight is live: outside the fine box (wFine=0) the fine gather is
    // skipped, deep inside it (wFine=1) the coarse gather is skipped — so almost every
    // pixel pays the same 8 taps as single-grid mode and only the border band pays 16.
    // Cost stays bounded (worst case 16 taps, confined to the band); the taps inside
    // these NON-UNIFORM branches use explicit LOD-0 fetches (see _tapEW). cascades==1 →
    // E0 exactly. The cascaded path uses the LOOPED gather (_sampleCascadeLooped) so the
    // tap bodies fit WebKit's 8192-byte pipeline-variable budget; normal-detail dual
    // fetch stays single-grid only (kept minimal while the looped path proves out).
    //
    // COMPILE-TIME cascade selection (invariants #5/#6): a TSL fragment cannot reference a
    // null StorageTexture, and the material RECOMPILES whenever _structGen changes (which
    // includes every cascade-count change). So the E1 (fine) 8-tap subtree is emitted ONLY
    // when the fine cascade is actually bound at setup() time — otherwise the shader is the
    // exact original 8-tap single grid (byte-identical) and never touches _atlas[1]=null.
    sampleIrradiance(P, Nvis, Ndir, dualDetail = false) {
        const useFine = Math.round(this.cascadeCountNode.value) >= 2 && !!this._atlas[1] && !!this._depthAtlas[1] && !!this._stateAtlas[1];
        if (!useFine) return this._sampleCascade(P, Nvis, Ndir, 0, dualDetail); // byte-identical single-grid fallback

        return Fn(() => {
            // Fine inside-test in normalized C1 coords.
            const f = P.sub(this.gridMinNode[1]).div(this.gridSizeNode[1].max(vec3(1e-6))); // 0..1 inside the fine box
            const fLo = tslMin(tslMin(f.x, f.y), f.z);
            const fHi = tslMin(tslMin(float(1.0).sub(f.x), float(1.0).sub(f.y)), float(1.0).sub(f.z));
            const edge = tslMin(fLo, fHi); // dist to nearest face; <0 outside
            const inBand = this.borderBandNode;
            const wIn = clamp(edge.div(inBand.max(float(1e-4))), float(0.0), float(1.0)); // 1 deep inside, ramps to 0 at the border, 0 outside
            const gate = this.cascadeCountNode.sub(1.0).clamp(float(0.0), float(1.0)); // 0 when cascades==1, 1 when 2 (live extra guard)
            const wFine = wIn.mul(gate).toVar();
            const E = vec3(0.0).toVar();
            If(wFine.lessThan(float(1.0)), () => {
                E.assign(this._sampleCascadeLooped(P, Nvis, Ndir, 0)); // coarse: valid over full bounds
            });
            If(wFine.greaterThan(float(0.0)), () => {
                E.assign(mix(E, this._sampleCascadeLooped(P, Nvis, Ndir, 1), wFine));
            });
            return E;
        })();
    }

    setup(builder) {
        if (!this._ready) return;
        // The receiver normal has TWO jobs with opposite needs, so it is SPLIT:
        // - Nvis (bias point + Chebyshev/wrap weights): must be camera-INVARIANT.
        //   The visibility test is a hard boundary; a camera-dependent normal moves P
        //   across it every frame and flickers like z-fighting. modelNormalMatrix
        //   never touches the camera, so the weights are rock-stable.
        // - Ndir (fetch direction): wants the detailed shading normal — but ONLY when
        //   that normal is itself camera-invariant. Tangent-attribute normal maps are
        //   (the view round-trip cancels). Screen-derivative normals are NOT: custom
        //   normalNode bumps (e.g. the city generator's surface-gradient relief) and
        //   flat shading come from dFdx/dFdy, which are garbage on 2×2 quads that
        //   straddle triangle edges — on dense facades the affected pixel set changes
        //   with every subpixel camera rotation, so the fetch direction random-walks
        //   and flickers no matter how smooth the irradiance atlas is. The gate is
        //   BINARY and STATIC per material build (a normal source either is invariant
        //   or it isn't — a partial blend only dims the flicker): no uniforms, no
        //   sliders, no per-frame cost. See docs/gi_receiver_normal_flicker.md.
        const objectDerivedNormal = normalize(modelNormalMatrix.mul(normalLocal).toVarying('v_speedballObjectNormal'));
        const viewDerivedNormal = normalize(mix(normalWorldGeometry, normalWorld, this.sampleNormalMixNode));
        const stableNormal = normalize(mix(viewDerivedNormal, objectDerivedNormal, this.sampleObjectNormalNode));
        // KNOWN EDGE: the gate reads builder.geometry, so if one material instance is
        // shared between a mesh WITH tangents and one WITHOUT, whichever geometry builds
        // the shader decides the path for that pairing. Fails SAFE (worst case = less GI
        // detail, never flicker); avoid sharing materials across mixed-tangent geometry.
        const material = builder.material;
        const stableTBN = builder.geometry?.hasAttribute?.('tangent') === true
            && !!material?.normalMap && !material.normalNode && material.flatShading !== true;
        // detailStrength (taste knob, NOT a flicker tradeoff — both fetch directions
        // are camera-invariant on trusted materials) blends the FETCHED IRRADIANCE
        // inside _sampleCascade, not the direction: angle-domain strength saturates
        // perceptually within a few degrees (reads as a toggle), irradiance-domain
        // strength is linear in the visible contrast.
        const detailNormal = stableTBN ? viewDerivedNormal : stableNormal;
        const reflectionP = positionWorld.mul(this.samplePositionScaleNode);
        const P = reflectionP.add(stableNormal.mul(this.normalBiasNode[0]).mul(this.sampleBiasScaleNode));

        // Optional Lumen-style local reflections. The probe is appended AFTER Three's
        // EnvironmentNode (see GiLightsNode), so context.radiance already contains the
        // material/scene PMREM. RGB is premultiplied by coverage and A is coverage:
        // by default that means LOCAL hits only; reflectionSkyFallback can explicitly
        // let true misses carry the injected SH sky in scenes without an environment.
        // Composite rather than add so the prior environment remains visible wherever
        // this layer has no authority and a later SSR pass can overlay its own hits.
        //
        // Only Physical/Standard materials expose a finite roughness scalar. Phong/Lambert
        // keep their exact old path, and the runtime branch skips all reflection atlas
        // taps at reflection intensity 0. `material.userData.speedballReflections = false`
        // is a compile-time per-material opt-out (set material.needsUpdate after changing it).
        const materialAllowsReflections = material?.userData?.speedballReflections !== false;
        if (this.roughReflectionsReady && materialAllowsReflections && Number.isFinite(material?.roughness) && builder.context.radiance) {
            const roughnessParticipation = select(
                roughness.lessThanEqual(this.roughnessLimitNode),
                float(1.0),
                float(0.0),
            );
            const reflectionWeight = this.reflectionIntensityNode.mul(this.enabledNode).mul(roughnessParticipation).clamp(0.0, 1.0).toVar();
            const roughLobeMix = (this._glossyReflectionsConfigured
                ? smoothstep(float(ROUGH_LOBE_MIX_START), float(ROUGH_LOBE_MIX_END), roughness)
                : float(1.0)).toVar();
            // Orthographic-safe view direction, transformed to the same stable world
            // space as detailNormal. Match Three's EnvironmentNode roughness bend so
            // very rough lobes cannot gather radiance from behind the tangent plane.
            const V = positionViewDirection.transformDirection(cameraWorldMatrix);
            const mirror = V.negate().reflect(detailNormal);
            const rough4 = pow(roughness, float(4.0));
            const R = normalize(mix(mirror, detailNormal, rough4));
            const sample = this.sampleIrradianceAndRough(P, reflectionP, stableNormal, detailNormal, R, reflectionWeight, roughLobeMix, stableTBN);
            const E = sample.get('irradiance').max(vec3(0.0)).mul(this.intensityNode).mul(this.enabledNode);
            builder.context.irradiance.addAssign(E);
            If(reflectionWeight.greaterThan(float(0.0)), () => {
                const local = sample.get('roughRadiance').max(vec4(0.0)).toVar();
                const covered = local.w.clamp(0.0, 1.0).mul(reflectionWeight);
                builder.context.radiance.mulAssign(float(1.0).sub(covered));
                builder.context.radiance.addAssign(local.rgb.mul(reflectionWeight));
            });
        } else {
            const E = this.sampleIrradiance(P, stableNormal, detailNormal, stableTBN).max(vec3(0.0)).mul(this.intensityNode).mul(this.enabledNode);
            builder.context.irradiance.addAssign(E);
        }
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

export function createProbeField({
    renderer,
    scene,
    intensity = 1.0,
    hysteresis = null,
    jitterMode: initialJitterMode = 'gated',
    onRebuilt = null,
    divisions = TARGET_PROBES_LONG_AXIS,
    rays: initialRays = RAYS_PER_PROBE_DEFAULT,
    cascades: initialCascades = 2,
    continuous: initialContinuous = true,
    roughReflections = false,
    reflectionQuality = null,
    reflectionIntensity = 1.0,
    roughnessLimit = null,
    reflectionSkyFallback = false,
    clusteredLighting = false,
    autoDetectChanges = true,
} = {}) {
    if (_activeProbeFieldOwner !== null) {
        throw new Error('createProbeField: only one active field is supported per module instance; dispose the existing field first.');
    }
    const fieldOwner = {};
    const node = getGiProbeNode();
    const reflectionConfig = resolveReflectionQuality(reflectionQuality, roughReflections);
    const roughReflectionsEnabled = reflectionConfig.rough;
    const glossyReflectionsEnabled = reflectionConfig.glossy;
    const glossyOctRes = glossyReflectionsEnabled ? reflectionConfig.glossyOct : 1;
    const glossyTile = glossyReflectionsEnabled ? glossyOctRes + 2 * BORDER : 1;
    const glossyUpdateInterval = glossyReflectionsEnabled ? reflectionConfig.glossyUpdateInterval : 0;
    const glossyHistoryBytesPerProbe = glossyTile * glossyTile * 4 * Float32Array.BYTES_PER_ELEMENT;
    node.setReflectionConfig(reflectionConfig);
    node.setIntensity(intensity);
    node.setReflectionIntensity(reflectionIntensity);
    node.setRoughnessLimit(Number.isFinite(roughnessLimit)
        ? roughnessLimit
        : reflectionConfig.roughnessLimit);
    // Live grid density: probes along the longest axis. setDivisions() updates it and
    // requests a (resize) rebuild; per-axis counts derive from it so cells stay ~cubic.
    let targetLongAxis = THREE.MathUtils.clamp(Math.round(divisions) || TARGET_PROBES_LONG_AXIS, 2, MAX_PROBES_PER_AXIS);
    // Live ray budget per probe (structural — changing it re-sizes the ray scratch buffer and
    // rebuilds the trace/blend kernels, so setRays() requests an idle-gated rebuild). Default
    // 64 keeps the locked 624-probe baseline visually-equivalent.
    let raysPerProbe = THREE.MathUtils.clamp(
        Math.round(Number(initialRays) / 16) * 16 || RAYS_PER_PROBE_DEFAULT,
        RAYS_MIN,
        RAYS_MAX,
    );
    // Normal-bias scale (×) over the auto-computed minCell·SURFACE_NORMAL_BIAS_CELL offset, and the
    // most-recent minCell, so setNormalBias() can rewrite the node uniform INSTANTLY (no rebuild)
    // and the scale survives the next rebuild's auto-bias recompute.
    let normalBiasScale = 1.0;
    let curMinCell = 0.1;

    // Active cascade count. 1 = byte-identical single-grid fallback (C1 never allocated);
    // 2 = coarse + fine detail cascade. cascadeCountNode defaults to 1 and is set to
    // `cascades` after the first full build so the first fold is the single-grid shader.
    let cascades = Math.round(Number(initialCascades)) === 1 ? 1 : 2;
    let solveTurn = 0;        // round-robin cascade index across ticks (C0 even, C1 odd)
    let buildStage = 0;       // staggered build phase machine (0 = build C0, 1 = build C1, 2 = done)
    let fieldEverReady = false; // first bring-up completed once — REbuilds are rest-gated, bring-up is not
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
            glossyPhase: uniform(0, 'uint'),
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
            glossyTilesX: 1, glossyTilesY: 1,
            glossyAtlasW: 1, glossyAtlasH: 1,
            minCell: 0.1,
            gpu: null,
            U: makeCascadeU(),
            probeCursor: 0,
            lastSolveAt: 0,
            solveDtEma: 0,
            glossyPhase: 0,
            prevAtlasW: 0, prevAtlasH: 0, prevGlossyAtlasW: 0, prevGlossyAtlasH: 0, prevProbeTotal: 0,
            needsClear: true, needsClassify: true,
            normalBias: 0.04, chebyBias: 0.0,
        };
    }
    const casc = [makeCascade(), makeCascade()];
    const c0LightCellCount = () => Math.max(1, casc[0].res.x - 1) * Math.max(1, casc[0].res.y - 1) * Math.max(1, casc[0].res.z - 1);

    let continuous = initialContinuous === true; // DEFAULT ON: keep the bounded GPU solve running
                              // while the camera moves — heavy build steps still wait for rest, so
                              // the no-hitch guarantee holds. false opts into strict idle-gating.
    let detectSceneChanges = autoDetectChanges !== false;
    let dirty = true;
    // Cached CPU build (BVH soup + material textures). The BVH depends ONLY on geometry,
    // so a divisions/rays change must NOT rebuild it — that ~200ms synchronous MeshBVH +
    // soup-flatten is the one remaining main-thread hitch. buildDirty gates a fresh build:
    // it's set by geometry/light-count/volume changes (true at start), and left FALSE by
    // setDivisions/setRays so those resize the grid/kernels off the cached soup (no hitch).
    let cachedBuilt = null;
    let blasCache = null;    // cross-rebuild BLAS cache (created with the lazy scene builder)
    let lastBuildSceneRewritten = false; // this build's scene landed as an in-place arena rewrite
    // Map extraction stages CPU bytes against this per-field arena. Accepted
    // builds either rewrite the current texture objects layer-by-layer or adopt
    // a fresh capacity generation. The generation is reference-counted through
    // sceneResource so staggered C0/C1 replacement cannot dispose old bindings.
    let mapsArena = {
        current: null,
        nextGeneration: 0,
        growth: PROBE_SCENE_STORAGE_GROWTH,
        maxLayers: Number(renderer?.backend?.device?.limits?.maxTextureArrayLayers)
            || TEXTURE_ARRAY_LAYERS_FALLBACK,
    };
    let mapsArenaRebinds = 0;
    let mapsArenaRewrites = 0;
    let mapsArenaLastUpdate = 'none';
    let kernelResidentReuses = 0;
    let kernelRebuilds = 0;
    // C0/C1 trace the same scene soup. Storage lives in a capacity arena whose
    // BufferAttribute + TSL-node identities survive structural rebuilds. Build
    // generations remain ref-counted separately so their material-map textures
    // stay alive through staggered C0/C1 replacement. On a capacity growth the
    // old generation also keeps its old arena alive until both kernels move over.
    let sceneResource = null;
    let sceneStorageGeneration = 0;
    let sceneStorageRebinds = 0;
    let sceneStorageRewrites = 0;
    let sceneStorageLastUpdate = 'none';
    // Lights have an independent lifetime because clustered-grid resizes can
    // replace giLightArena while the geometry/BVH arrays stay byte-identical.
    let lightResource = null;
    // ── clustered-lighting mode (opt-in, structural at creation) ──
    // The raster side draws EVERY light through GiClusteredLightsNode; the GI lane
    // budgets to a FIXED MAX_LIGHTS-slot arena. Analytic records precede the
    // reserved emitter suffix in both modes; only clustered analytic records use
    // importance selection. Cell-list overflow and out-of-grid hits fall back to
    // the full arena, with type-3 records excluded from per-hit shading.
    const clusteredGi = clusteredLighting === true;
    let giLightArena = null;     // clustered records + C0 cell-list suffix
    let giLegacyLightArena = null; // fixed MAX_LIGHTS records; count edits never rebuild geometry
    let giSelectedCount = 0;
    let giLegacyLightCount = 0;
    let giEmitterBase = 0;
    let giEmitterCount = 0;
    let liveLightRecords = null; // latest collected records survive grid/ray-only kernel rebuilds
    let giLightGridDirty = false;
    let buildDirty = true;
    // Monotonic invalidation token for the CPU soup. A native/browser scene
    // update can arrive while buildSpectralScene is yielding to material-map
    // extraction; the token prevents that request from being cleared by the
    // older build when it resumes.
    let buildGeneration = 0;
    let manualVolumes = null; // explicit probe volumes (Probe Origin boxes); null = auto-fit scene
    // needsClassify / needsClear / probeCursor / prev* are PER-CASCADE (see
    // makeCascade); frameCounter is shared because Monte Carlo advances one basis
    // from C0 and both cascades read the same U.frameJitter.
    let rebuildBackoff = 0;   // ticks remaining before retrying after a failed/empty rebuild (A7)
    // ── auto-throttle (the hard rule: never lag the browser). The per-tick ray budget
    // adapts to the observed tick cadence: halve when frames slip, creep back up when
    // they're comfortably fast. Measures GPU pressure on THIS machine — no tuning knob.
    let tickBudgetRays = RAYS_PER_TICK;
    // Experimentation knob (setRayBudget): the per-tick trace budget TARGET the
    // auto-throttle recovers toward and the kernel build sizes its scratch from.
    // More rays/tick = faster light propagation for more GPU; frame pacing stays
    // owned by the cadence controller either way.
    let rayBudgetCeiling = RAYS_PER_TICK;
    // Ray-jitter regime (setJitterMode): 'gated' (default) holds the basis and
    // pairs with a low-latency 0.60 history; 'montecarlo' re-jitters every C0
    // solve tick and pairs with 0.90 history to absorb the sample blast.
    const normalizeJitterMode = (mode) => {
        const m = String(mode ?? '').toLowerCase().replace(/[\s_-]/g, '');
        return (m === 'montecarlo' || m === 'mc') ? 'montecarlo' : 'gated';
    };
    let jitterMode = normalizeJitterMode(initialJitterMode);
    let lastTickAt = 0;
    let tickDtEma = 0;
    // Temporal cadence must survive auto-throttle's deliberate tickDtEma resets.
    // Otherwise every budget adjustment injects a one-frame 60 Hz history jump.
    let hysteresisTickDtEma = 0;
    let budgetCooldown = 0;   // ticks to hold after a shrink before growing again (damps sawtooth)
    let cadenceOverloadStreak = 0;
    let inFlight = false;
    let disposed = false;

    // A canvas resize/fullscreen transition can produce one slow presentation frame even
    // though GI workload did not change. Discard that interval from the budget controller;
    // keep the learned ray budget and temporal history intact.
    function resetFramePacing() {
        lastTickAt = 0;
        tickDtEma = 0;
        cadenceOverloadStreak = 0;
    }
    let frameCounter = 0;
    let emitterVisSeedCounter = 0;   // advances with the active ray-sampling epoch
    let quantStep = 1;        // translation deadband (~quarter cell) for the geo signature (A1)
    let lightQuant = 1;       // scene-relative position deadband for the light signature (B4)
    // reactivity: self-detect live light/geometry edits and re-converge fast.
    let baseHysteresis = Number.isFinite(hysteresis)
        ? THREE.MathUtils.clamp(hysteresis, 0, 0.99)
        : JITTER_HYSTERESIS_DEFAULTS[jitterMode];
    const jitterHysteresis = {
        gated: JITTER_HYSTERESIS_DEFAULTS.gated,
        montecarlo: JITTER_HYSTERESIS_DEFAULTS.montecarlo,
    };
    jitterHysteresis[jitterMode] = baseHysteresis;
    let hysteresisNormalize = true;
    let chebyBiasScale = 1.0;
    let debugFreezeRayJitter = false;
    let debugFrameJitterOverride = null;
    let lastLightSig = null;
    let lastGeoSig = null;
    let lastXformSig = null;
    let lastDeformSig = null;
    let lastRefitCount = 0;   // BLASes refit by the most recent deform refresh (debug/stats)
    let lastTransformInstanceCount = 0;
    let lastTlasRefitCount = 0;
    const pendingTransformTargets = new Set();
    let pendingAllTransforms = false;
    let pendingDeformRefresh = false;
    const pendingMaterialValueTargets = new Set();
    let pendingAllMaterialValues = false;
    let lastMaterialValueRecords = 0;  // uber records rewritten by the most recent value refresh (debug/stats)
    let geoStable = -1;       // -1 = no pending geo change; >=0 = stable-check count since a change (debounce, A1)
    let checkCounter = 0;
    let lastIdleMs = Infinity;
    let lastMoving = false;
    let lastPlaying = false;
    let lastRestOnly = true;
    let lastSolveList = '';
    let lastUpdatedCount = 0;
    const _sigVec = new THREE.Vector3();
    const _sigScale = new THREE.Vector3();
    const _sigEmissive = [0, 0, 0];

    // SHARED compute uniforms — a single instance, folded into BOTH cascade U blocks by
    // reference (see below), so one GUI knob writes both cascades. Per-cascade uniforms
    // (gridMin/gridSize/res*/probeTotal/probeOffset/updatedCount/atlasDim/maxDist/cellMin/
    // relocClamp) live in each C.U (makeCascadeU).
    const U = {
        lightCount: uniform(0, 'uint'),
        emitterBase: uniform(0, 'uint'),
        emitterCount: uniform(0, 'uint'),
        // Emitter-vis disk target follows the same sampling epoch as frameJitter:
        // stable across a Gated pass, fresh every Monte Carlo solve tick.
        emitterVisSeed: uniform(0.0),
        frameJitter: uniform(0.0),
        // Raw 60 Hz reference-rate retention. Adaptive signal-specific weights are
        // derived from this first, then normalized once with hysteresisExponent.
        hysteresis: uniform(baseHysteresis),
        hysteresisExponent: uniform(1.0),
        // Raw cosine exponent. Seven is already a ~25 degree half-power cone and
        // about 7.5 effective samples at the default 64 rays; the old implicit 50
        // collapsed the estimate to roughly one ray and made visibility brittle.
        depthSharpness: uniform(7.0),
        radianceClamp: uniform(8.0),    // cap the multibounce feedback term (anti-runaway)
        classifyStrength: uniform(0.0), // gates relocation APPLY (mirrors node.classifyStrengthNode)
        filterStrength: uniform(1.0),   // CORE denoise: 0 = filter off (harness baseline), 1 = full intra-tile spatial filter
        filterSmooth: uniform(0.5),     // UI "Smoothness": widens the bilateral edge-stop (0 = baseline detail, 1 = very smooth)
        // ── live temporal-blend tuning (adaptive hysteresis). Defaults == the old constants
        // → byte-identical until a slider moves. See blendKernel for how each is used.
        tempChangeSigma1: uniform(GI_TEMPORAL_CHANGE_SIGMA1), // delta (in σ) above which a change counts as REAL → snaps. lower = snappier
        tempChangeHDrop: uniform(GI_TEMPORAL_CHANGE_H_DROP),  // how much hysteresis drops on a real change. higher = harder snap
        tempClampSigma: uniform(GI_TEMPORAL_CLAMP_SIGMA),     // firefly clamp band (in σ). lower = tighter/steadier, more lag
        // NIR band gate: 0 = visible band (IR illuminators, emitter class 4, are
        // invisible — they emit outside the sampled domain), 1 = NIR sensing
        // (white-phosphor NV): IR lights join NEE at their promoted (k,k,k).
        // Mirrors the PT's emitterAtLambda 850 nm band collapsed to one scalar.
        nirGate: uniform(0.0),
        nirGain: uniform(1.0),
        // ── sky → probes (see the "sky → probes" block below + traceKernel miss branch) ──
        skyIntensity: uniform(1.0),   // scales the injected SH sky (0 = off)
        skySH: Array.from({ length: 9 }, () => uniform(new THREE.Vector3())), // radiance SH-9; all-zero = miss stays BLACK (the old invariant)
        // Explicit layer contract: 0 gives Speedball no miss coverage and leaves
        // prior radiance unchanged; 1 lets an explicitly configured SH sky fill it.
        reflectionSkyFallback: uniform(reflectionSkyFallback === true ? 1.0 : 0.0),
        // Avoid opaque-black reflection coverage when fallback is requested before
        // setSky(). This tracks explicit ownership, not SH energy, so an intentionally
        // configured black sky is still a valid distant layer.
        skyConfigured: uniform(0.0),
        debugTraceBiasScale: uniform(1.0),
        debugRayEpsScale: uniform(1.0),
        debugDirectScale: uniform(1.0),
        debugEmissiveScale: uniform(1.0),
        debugAlbedoScale: uniform(1.0),
        debugBounceScale: uniform(1.0),
        debugCosinePower: uniform(1.0),
        debugTempNoiseHBoost: uniform(GI_TEMPORAL_NOISE_H_BOOST),
        debugTempChangeSigma0: uniform(GI_TEMPORAL_CHANGE_SIGMA0),
        debugTempMinChangeH: uniform(GI_TEMPORAL_MIN_CHANGE_H),
        debugTempVarEps: uniform(GI_TEMPORAL_VAR_EPS),
        debugTempVarRel: uniform(GI_TEMPORAL_VAR_REL),
        debugDepthHistoryScale: uniform(1.0),
        debugFilterKScale: uniform(1.0),
        debugFilterRelScale: uniform(1.0),
        debugFilterEpsScale: uniform(1.0),
    };
    // Fold the shared uniforms into every cascade's U by REFERENCE so buildKernels closes
    // over C.U and reads both shared + per-cascade uniforms uniformly.
    for (const C of casc) Object.assign(C.U, U);

    // ── sky → probes ──────────────────────────────────────────────────────────
    // Miss rays return sky RADIANCE evaluated from an injected SH-9 (9 vec3
    // uniforms = 27 floats). So probes carry occlusion-aware sky light and BOUNCE
    // it (street canyons darken, courtyards glow — flat IBL can't do either).
    // ZERO textures, zero probe memory, and pure ALU in the kernel — the WGSL
    // builder forbids sampled textures with implicit/gradient LOD in compute, so
    // SH is also the only representation that can't break there. Diffuse GI only
    // needs a low-frequency sky; SH-9 IS that signal.
    //
    // Injection API (general-purpose — the library never guesses at the scene):
    //   setSky(null)                          → off (miss = BLACK, the old invariant)
    //   setSky(color | 0xrrggbb)              → uniform dome radiance
    //   setSky({ zenith, horizon, ground })   → vertical gradient dome
    //   setSky(LightProbe | SphericalHarmonics3) → full SH sky (radiance SH)
    // The sun must stay NEE-only: inject a sun-free sky or the sun double-counts.
    function _skyToSH(input) {
        const out = Array.from({ length: 9 }, () => new THREE.Vector3());
        if (!input) return out;
        const src = input.isLightProbe ? input.sh : (input.isSphericalHarmonics3 ? input : null);
        if (src) {
            for (let i = 0; i < 9; i++) out[i].copy(src.coefficients[i]);
            return out;
        }
        // color / gradient dome → numeric SH projection over a Fibonacci sphere
        let radianceAt;
        const _c = new THREE.Color();
        if (typeof input === 'number' || input.isColor) {
            const c = new THREE.Color(input);
            radianceAt = () => c;
        } else if (typeof input === 'object') {
            const zen = new THREE.Color(input.zenith ?? 0);
            const hor = new THREE.Color(input.horizon ?? input.zenith ?? 0);
            const gnd = new THREE.Color(input.ground ?? 0);
            radianceAt = (d) => (d.y >= 0 ? _c.lerpColors(hor, zen, d.y) : _c.lerpColors(hor, gnd, -d.y));
        } else return out;
        const N = 128, dir = new THREE.Vector3(), basis = new Array(9).fill(0);
        const w = (4 * Math.PI) / N;
        for (let i = 0; i < N; i++) {
            const y = 1 - (2 * (i + 0.5)) / N;
            const r = Math.sqrt(Math.max(0, 1 - y * y));
            const phi = i * Math.PI * (3 - Math.sqrt(5));
            dir.set(r * Math.cos(phi), y, r * Math.sin(phi));
            THREE.SphericalHarmonics3.getBasisAt(dir, basis);
            const c = radianceAt(dir);
            for (let j = 0; j < 9; j++) {
                out[j].x += c.r * basis[j] * w;
                out[j].y += c.g * basis[j] * w;
                out[j].z += c.b * basis[j] * w;
            }
        }
        return out;
    }
    function setSky(input) {
        // Trace-side change: newly traced rays carry the new sky and the per-texel
        // change detector re-converges affected texels at the bounded steady rate.
        const sh = _skyToSH(input);
        U.skyConfigured.value = input !== null && input !== undefined ? 1.0 : 0.0;
        for (let i = 0; i < 9; i++) U.skySH[i].value.copy(sh[i]);
    }

    function setReflectionSkyFallback(on) {
        // Reflection caches are temporal by design; converged local history is
        // preserved and the ownership change blends in at the steady rate.
        U.reflectionSkyFallback.value = on === true ? 1.0 : 0.0;
    }

    function isSupported() {
        return renderer?.backend?.isWebGPUBackend === true
            && typeof renderer.computeAsync === 'function'
            && typeof THREE.StorageTexture === 'function'
            && typeof THREE.StorageBufferAttribute === 'function';
    }

    // The compute graph bakes exactly one thing from a specific build: the
    // material map array textures. Identity-equal maps (including the all-null
    // case of an untextured scene) make a live kernel fully reusable across an
    // in-place scene rewrite.
    function mapsCompatible(a, b) {
        const ka = Object.keys(a || {});
        const kb = Object.keys(b || {});
        if (ka.length !== kb.length) return false;
        for (const key of ka) if ((a[key] || null) !== ((b || {})[key] || null)) return false;
        return true;
    }

    function retainSceneStorage(arena) {
        if (arena) arena.refs++;
        return arena;
    }

    function releaseSceneStorage(arena) {
        if (!arena || arena.refs <= 0) return;
        arena.refs--;
        if (arena.refs === 0) {
            disposeStorageAttributes(renderer, arena.buffers, PROBE_SCENE_BUFFER_KEYS);
        }
    }

    function disposeMapsGeneration(generation) {
        if (!generation || generation.disposed) return;
        generation.disposed = true;
        const disposedMaps = new Set();
        for (const texture of Object.values(generation.textures || {})) {
            if (!texture || disposedMaps.has(texture)) continue;
            disposedMaps.add(texture);
            texture.dispose?.();
        }
    }

    function retainMapsGeneration(generation) {
        if (generation) generation.refs++;
        return generation;
    }

    function releaseMapsGeneration(generation) {
        if (!generation || generation.refs <= 0) return;
        generation.refs--;
        if (generation.refs === 0) disposeMapsGeneration(generation);
    }

    function commitMapsArenaPlan(built) {
        const plan = built?.mapsArenaPlan;
        if (!plan || plan.committed) return { mapsRebound: false, mapsRewritten: false };
        const generation = plan.generation;
        if (!generation || generation.disposed || !mapsArena) {
            throw new Error('SPEEDBALL GI material maps arena generation is unavailable');
        }

        if (plan.kind === 'rewrite') {
            if (mapsArena.current !== generation) {
                throw new Error('SPEEDBALL GI material maps arena changed during a staged rewrite');
            }
            const layerBytes = generation.width * generation.height * 4;
            for (const key of PROBE_MAP_KEYS) {
                const live = plan.liveLayers[key] | 0;
                const texture = generation.textures[key];
                const packed = plan.packedLayers?.[key] || null;
                const changedLayers = plan.changedLayers?.[key] || [];
                if (live <= 0 || !texture || !packed || changedLayers.length === 0) continue;
                if (packed.length !== live * layerBytes) {
                    throw new Error(`SPEEDBALL GI invalid ${key} map layer staging length`);
                }
                for (const layer of changedLayers) {
                    const start = layer * layerBytes;
                    texture.image.data.set(packed.subarray(start, start + layerBytes), start);
                }
                // Three WebGPU honors DataArrayTexture layerUpdates. Preserve
                // any earlier pending layers until the renderer consumes them;
                // add only content that actually changed in this build.
                if (typeof texture.addLayerUpdate === 'function') {
                    for (const layer of changedLayers) texture.addLayerUpdate(layer);
                }
                texture.needsUpdate = true;
            }
            generation.liveLayers = { ...plan.liveLayers };
            mapsArenaRewrites++;
            mapsArenaLastUpdate = 'rewrite';
            plan.committed = true;
            plan.packedLayers = null;
            plan.changedLayers = null;
            return { mapsRebound: false, mapsRewritten: true };
        }

        mapsArena.current = generation;
        generation.liveLayers = { ...plan.liveLayers };
        mapsArenaRebinds++;
        mapsArenaLastUpdate = plan.kind;
        plan.committed = true;
        plan.packedLayers = null;
        return { mapsRebound: true, mapsRewritten: false };
    }

    function retainSceneResource(resource) {
        if (resource) resource.refs++;
        return resource;
    }

    function releaseSceneResource(resource) {
        if (!resource || resource.refs <= 0) return;
        resource.refs--;
        if (resource.refs > 0) return;
        releaseMapsGeneration(resource.mapsGeneration);
        releaseSceneStorage(resource.storage);
    }

    function retainLightResource(resource) {
        if (resource) resource.refs++;
        return resource;
    }

    function releaseLightResource(resource) {
        if (!resource || resource.refs <= 0) return;
        resource.refs--;
        if (resource.refs === 0) disposeStorageAttribute(renderer, resource.buffer);
    }

    function sceneStorageMaxElements(ArrayType) {
        const limits = renderer?.backend?.device?.limits;
        const binding = Number(limits?.maxStorageBufferBindingSize) || STORAGE_BINDING_FALLBACK;
        const buffer = Number(limits?.maxBufferSize) || binding;
        return Math.max(1, Math.floor(Math.min(binding, buffer) / ArrayType.BYTES_PER_ELEMENT));
    }

    function sceneStorageCapacity(required, previous, ArrayType) {
        const live = Math.max(1, Math.floor(Number(required) || 0));
        const prior = Math.max(0, Math.floor(Number(previous) || 0));
        const target = Math.max(
            live,
            Math.ceil(live * PROBE_SCENE_STORAGE_GROWTH),
            prior > 0 ? Math.ceil(prior * PROBE_SCENE_STORAGE_GROWTH) : 0,
        );
        const limit = sceneStorageMaxElements(ArrayType);
        // Preserve today's fail-loud behavior if a live scene itself exceeds the
        // device binding limit. Otherwise spend all available geometric headroom.
        return limit >= live ? Math.min(target, limit) : live;
    }

    function updateSceneTraversalUniforms(Utrav, built) {
        Utrav.nodeCount.value = built.nodeCount >>> 0;
        Utrav.tlasNodeCount.value = (built.tlasNodeCount ?? 0) >>> 0;
        Utrav.instBase.value = (built.instBase ?? 0) >>> 0;
        Utrav.tlasBase.value = (built.tlasBase ?? 0) >>> 0;
    }

    function createSceneStorage(built, previous = null) {
        const buffers = {};
        const capacities = {};
        const liveLengths = {};
        for (const key of PROBE_SCENE_BUFFER_KEYS) {
            const source = built[key];
            const ArrayType = source.constructor;
            const capacity = sceneStorageCapacity(source.length, previous?.capacities?.[key], ArrayType);
            const array = new ArrayType(capacity);
            array.set(source);
            buffers[key] = new THREE.StorageBufferAttribute(array, 1);
            capacities[key] = capacity;
            liveLengths[key] = source.length;
        }
        const storages = {
            bvhNodes: storage(buffers.bvhNodes, 'uint', buffers.bvhNodes.count).toReadOnly(),
            triIndex: storage(buffers.triIndex, 'uint', buffers.triIndex.count).toReadOnly(),
            vertexData: storage(buffers.vertexData, 'float', buffers.vertexData.count).toReadOnly(),
            triMaterial: storage(buffers.triMaterial, 'uint', buffers.triMaterial.count).toReadOnly(),
            materials: storage(buffers.materials, 'float', buffers.materials.count).toReadOnly(),
        };
        const traversalUniforms = {
            nodeCount: uniform(0, 'uint'),
            tlasNodeCount: uniform(0, 'uint'),
            instBase: uniform(0, 'uint'),
            tlasBase: uniform(0, 'uint'),
            envRotation: uniform(0.0),
            envIntensity: uniform(1.0),
        };
        updateSceneTraversalUniforms(traversalUniforms, built);
        return {
            refs: 0,
            generation: ++sceneStorageGeneration,
            buffers,
            storages,
            traversalUniforms,
            capacities,
            liveLengths,
        };
    }

    function sceneStorageFits(arena, built, forceRebind = false) {
        return !forceRebind && !!arena
            && PROBE_SCENE_BUFFER_KEYS.every((key) => built[key].length <= arena.capacities[key]);
    }

    function rewriteSceneStorage(arena, built) {
        for (const key of PROBE_SCENE_BUFFER_KEYS) {
            const source = built[key];
            arena.buffers[key].array.set(source, 0);
            arena.liveLengths[key] = source.length;
            // Upload only the live prefix. Stale capacity tail is intentionally
            // left untouched and is unreachable through the exact traversal
            // count/base uniforms updated below.
            markStorageDirty(arena.buffers[key], [[0, source.length]]);
        }
        updateSceneTraversalUniforms(arena.traversalUniforms, built);
    }

    function copySceneStorageRanges(key, source, ranges) {
        const arena = sceneResource?.storage;
        const attr = arena?.buffers?.[key];
        if (!attr || !source || source.length > attr.array.length) return false;
        const uploadRanges = Array.isArray(ranges) && ranges.length > 0
            ? ranges
            : [[0, source.length]];
        for (const [startValue, countValue] of uploadRanges) {
            const start = Math.max(0, Math.floor(Number(startValue) || 0));
            const count = Math.max(0, Math.min(
                Math.floor(Number(countValue) || 0),
                source.length - start,
                attr.array.length - start,
            ));
            if (count > 0) attr.array.set(source.subarray(start, start + count), start);
        }
        markStorageDirty(attr, uploadRanges);
        return true;
    }

    function createSceneResource(built, storageArena) {
        const arena = retainSceneStorage(storageArena);
        const mapsGeneration = retainMapsGeneration(built.mapsArenaGeneration);
        return {
            built,
            refs: 1, // active-root ref
            storage: arena,
            mapsGeneration,
            buffers: arena.buffers,
            storages: arena.storages,
            traversalUniforms: arena.traversalUniforms,
            maps: built.maps,
        };
    }

    function createLightResource(array) {
        const buffer = new THREE.StorageBufferAttribute(array, 1);
        const storageNode = storage(buffer, 'float', buffer.count);
        if (!clusteredGi) storageNode.toReadOnly();
        return {
            array,
            refs: 1, // active-root ref
            buffer,
            storage: storageNode,
        };
    }

    // Install the root owners that future cascade kernels retain. Within capacity,
    // only the live prefixes and traversal uniforms change; the arena's attributes
    // and TSL storage nodes remain resident. A generation wrapper keeps its own map
    // textures alive until staggered C0/C1 replacement drops the last old kernel.
    function prepareSharedResources(built, lightDataChanged = false) {
        let sceneBuffersRebound = false;
        let sceneBuffersRewritten = false;
        if (!sceneResource || sceneResource.built !== built) {
            const previous = sceneResource;
            let arena = previous?.storage || null;
            // A map-generation rebind must carry a matching scene-storage
            // generation. Otherwise staggered C1 would see the new material
            // layer indices through its old map bindings between C0 and C1.
            let mapsGenerationRebound = (previous?.mapsGeneration || null)
                !== (built.mapsArenaGeneration || null);
            // The inverse coupling matters too: if scene storage grows while a
            // staged map rewrite fits, old C1 still owns the old material records.
            // Fork the map generation so its texture contents stay paired with
            // those records until that old cascade releases both resources.
            const rawSceneStorageFits = sceneStorageFits(arena, built, false);
            if (!rawSceneStorageFits && !mapsGenerationRebound && previous
                && typeof _rebindMaterialMapsArenaBuild === 'function'
                && _rebindMaterialMapsArenaBuild(THREE, built, mapsArena)) {
                mapsGenerationRebound = true;
            }
            if (!sceneStorageFits(arena, built, mapsGenerationRebound)) {
                arena = createSceneStorage(built, arena);
                sceneStorageRebinds++;
                sceneStorageLastUpdate = mapsGenerationRebound && previous
                    ? 'maps-rebind'
                    : (previous ? 'grow' : 'allocate');
                sceneBuffersRebound = true;
            } else {
                rewriteSceneStorage(arena, built);
                sceneStorageRewrites++;
                sceneStorageLastUpdate = 'rewrite';
                sceneBuffersRewritten = true;
            }
            commitMapsArenaPlan(built);
            sceneResource = createSceneResource(built, arena);
            releaseSceneResource(previous);
        }

        const lightArray = clusteredGi ? giLightArena : giLegacyLightArena;
        if (!lightArray) throw new Error('SPEEDBALL GI light storage is unavailable');
        if (!lightResource || lightResource.array !== lightArray) {
            const previous = lightResource;
            lightResource = createLightResource(lightArray);
            releaseLightResource(previous);
        } else if (lightDataChanged) {
            // The typed array was refilled in place; every cascade observes the
            // same GPU attribute, so one version bump uploads it for both.
            markStorageDirty(lightResource.buffer, null);
        }
        return { sceneBuffersRebound, sceneBuffersRewritten };
    }

    // Same-dim rebuild: free ONLY the previous compute graph + ray scratch and
    // release its shared-owner refs. Atlases and history buffers were handed to
    // the new kernels via `reuse`, so they must stay alive.
    function disposeKernelOnly(g) {
        if (!g) return;
        disposeComputeNodes(g, PROBE_COMPUTE_KEYS);
        disposeStorageAttribute(renderer, g.rayBuffer);
        releaseLightResource(g.lightResource);
        releaseSceneResource(g.sceneResource);
    }

    // Free ONE cascade's GPU resources (buffers + its own atlas set) and reset its
    // prev-dim trackers. Clears the node's atlas binding for that cascade slot.
    function disposeCascadeGPU(c) {
        const C = casc[c];
        const g = C.gpu;
        if (g) {
            disposeComputeNodes(g, PROBE_COMPUTE_KEYS);
            disposeStorageAttribute(renderer, g.irrBuffer);
            disposeStorageAttribute(renderer, g.roughSpecularBuffer);
            disposeStorageAttribute(renderer, g.glossySpecularBuffer);
            disposeStorageAttribute(renderer, g.glossyWeightBuffer);
            disposeStorageAttribute(renderer, g.depthBuffer);
            disposeStorageAttribute(renderer, g.stateBuffer);
            disposeStorageAttribute(renderer, g.emitterVisBuffer);
            disposeStorageAttribute(renderer, g.rayBuffer);
            g.atlas?.dispose?.();
            g.roughSpecularAtlas?.dispose?.();
            g.glossySpecularAtlas?.dispose?.();
            g.depthAtlas?.dispose?.();
            g.stateAtlas?.dispose?.();
            releaseLightResource(g.lightResource);
            releaseSceneResource(g.sceneResource);
        }
        C.gpu = null;
        C.prevAtlasW = C.prevAtlasH = C.prevGlossyAtlasW = C.prevGlossyAtlasH = C.prevProbeTotal = 0;
        node.setAtlases(c, null, null, null, null, null);
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
        const isC0 = C === casc[0];
        const probeTotal = C.probeTotal;     // shadow the old flat name → per-cascade
        const atlasW = C.atlasW, atlasH = C.atlasH;
        const res = C.res;
        const glossyGroupsPerProbe = glossyReflectionsEnabled
            ? Math.ceil((glossyTile * glossyTile) / PROBE_WORKGROUP_SIZE)
            : 0;
        // Both modes bind a fixed-capacity light arena with a contiguous emitter
        // suffix. Clustered mode appends the C0 cell lists after that fixed arena.
        // rebuild() prepares both root owners before C0; keep these guards so a
        // future call-order change still lands in the shared path.
        if (clusteredGi && !giLightArena) fillGiLightArena(recordsFromBuilt(built));
        if (!clusteredGi && !giLegacyLightArena) fillLegacyLightArena(recordsFromBuilt(built));
        const expectedLightArray = clusteredGi ? giLightArena : giLegacyLightArena;
        if (sceneResource?.built !== built || lightResource?.array !== expectedLightArray) {
            prepareSharedResources(built);
        }
        const sharedScene = sceneResource;
        const sharedLights = lightResource;
        const buffers = {
            ...sharedScene.buffers,
            lights: sharedLights.buffer,
        };
        // These nodes are owned by the capacity arena, not by this kernel build.
        // A structural rebuild that fits therefore presents the exact same TSL
        // storage-node and BufferAttribute identities to Three's binding cache.
        const { bvhNodes, triIndex, vertexData, triMaterial, materials } = sharedScene.storages;
        // The clustered cell lists occupy the float arena's suffix. Keeping light
        // records and exact-small-integer indices in ONE binding is structural:
        // the trace kernel already uses the portable WebGPU baseline of eight
        // storage buffers, so a separate grid binding makes its pipeline invalid.
        const lights = sharedLights.storage;
        const lightGridCellCount = clusteredGi ? c0LightCellCount() : 0;

        // Live traversal sizes/bases are resident uniforms. Capacity may be
        // larger than this build, but no walk can enter the stale tail.
        const Utrav = sharedScene.traversalUniforms;
        const trav = buildTraversal({
            storages: { bvhNodes, triIndex, vertexData, triMaterial, materials },
            U: Utrav, env: null, lut: null, lutRes: 0, maps: sharedScene.maps,
        });
        const { fetchVert, fetchNorm, traverseClosest, traverseAny, instLocalRay, instNormalToWorld, matFloat, triVert, fetchUV, hitUV, sampleLayer, srgbToLinear, albedoTex, emissiveTex, haveAlbedoMap, haveEmissiveMap } = trav;

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

        // Optional rough local-radiance history: RGB is premultiplied by directional
        // coverage, A is coverage. The low-resolution lobe stays in the SAME blend
        // dispatch; the high-resolution glossy companion below adds one resolve
        // dispatch, but neither path adds tracing, BVH traversal, NEE, or rays.
        const roughSpecularBuffer = roughReflectionsEnabled
            ? (reuse?.roughSpecularBuffer || new THREE.StorageBufferAttribute(new Float32Array(Math.max(4, probeTotal * TILE * TILE * 4)), 1))
            : null;
        const roughSpecular = roughSpecularBuffer ? storage(roughSpecularBuffer, 'float', roughSpecularBuffer.count) : null;
        const roughSpecularRead = roughSpecularBuffer ? storage(roughSpecularBuffer, 'float', roughSpecularBuffer.count).toReadOnly() : null;
        // High-resolution companion lobe for smooth/glossy receivers. Store its
        // unnormalized RGBA numerator and scalar angular support separately; weak
        // rotating ray sets must not receive the same history weight as strong ones.
        const glossySpecularBuffer = glossyReflectionsEnabled
            ? (reuse?.glossySpecularBuffer || new THREE.StorageBufferAttribute(new Float32Array(Math.max(4, probeTotal * glossyTile * glossyTile * 4)), 1))
            : null;
        const glossySpecular = glossySpecularBuffer ? storage(glossySpecularBuffer, 'float', glossySpecularBuffer.count) : null;
        const glossyWeightBuffer = glossyReflectionsEnabled
            ? (reuse?.glossyWeightBuffer || new THREE.StorageBufferAttribute(new Float32Array(Math.max(1, probeTotal * glossyTile * glossyTile)), 1))
            : null;
        const glossyWeight = glossyWeightBuffer ? storage(glossyWeightBuffer, 'float', glossyWeightBuffer.count) : null;

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

        // Same tile packing as irradiance so receiver placement/cascade math stays
        // identical. No mips: the power-8 gather is already the roughness filter.
        const roughSpecularAtlas = roughReflectionsEnabled ? (reuse?.roughSpecularAtlas || (() => {
            const t = new THREE.StorageTexture(atlasW, atlasH);
            t.type = THREE.HalfFloatType; t.format = THREE.RGBAFormat;
            t.minFilter = THREE.LinearFilter; t.magFilter = THREE.LinearFilter;
            t.wrapS = THREE.ClampToEdgeWrapping; t.wrapT = THREE.ClampToEdgeWrapping;
            t.generateMipmaps = false; t.mipmapsAutoUpdate = false;
            return t;
        })()) : null;
        const glossySpecularAtlas = glossyReflectionsEnabled ? (reuse?.glossySpecularAtlas || (() => {
            const t = new THREE.StorageTexture(C.glossyAtlasW, C.glossyAtlasH);
            t.type = THREE.HalfFloatType; t.format = THREE.RGBAFormat;
            t.minFilter = THREE.LinearFilter; t.magFilter = THREE.LinearFilter;
            t.wrapS = THREE.ClampToEdgeWrapping; t.wrapT = THREE.ClampToEdgeWrapping;
            t.generateMipmaps = false; t.mipmapsAutoUpdate = false;
            return t;
        })()) : null;

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

        const emitterVisBuffer = reuse?.emitterVisBuffer || new THREE.StorageBufferAttribute(
            new Float32Array(Math.max(1, probeTotal * GI_EMITTER_INJECT_CAP)),
            1,
        );
        const emitterVis = storage(emitterVisBuffer, 'float', emitterVisBuffer.count);
        const emitterVisRead = storage(emitterVisBuffer, 'float', emitterVisBuffer.count).toReadOnly();

        const probeTraceOrigin = (probeIndex) => {
            const ro = probeWorldPos(probeIndex, U).toVar();
            const mbT = probeIndex.mul(uint(4));
            ro.addAssign(vec3(
                stateRead.element(mbT.add(uint(1))),
                stateRead.element(mbT.add(uint(2))),
                stateRead.element(mbT.add(uint(3))),
            ).mul(U.classifyStrength));
            return ro;
        };

        const loadLightVec3 = (base, off) => vec3(lights.element(base.add(uint(off))), lights.element(base.add(uint(off + 1))), lights.element(base.add(uint(off + 2))));
        let lightGridKernel = null;
        // The cell list is derived exclusively from C0 and lives in the shared
        // light buffer. Building the identical list from C1 was duplicate work.
        if (clusteredGi && isC0) {
            const gridU = casc[0].U;
            lightGridKernel = Fn(() => {
                const cellIndex = instanceIndex.toVar();
                If(cellIndex.greaterThanEqual(uint(lightGridCellCount)), () => { Return(); });
                const cellsX = gridU.resX.sub(uint(1));
                const cellsY = gridU.resY.sub(uint(1));
                const ix = cellIndex.mod(cellsX);
                const iy = cellIndex.div(cellsX).mod(cellsY);
                const iz = cellIndex.div(cellsX.mul(cellsY));
                const cellSize = vec3(
                    gridU.gridSize.x.div(float(cellsX)),
                    gridU.gridSize.y.div(float(cellsY)),
                    gridU.gridSize.z.div(float(gridU.resZ.sub(uint(1)))),
                );
                const cellMin = gridU.gridMin.add(vec3(float(ix), float(iy), float(iz)).mul(cellSize));
                const cellMax = cellMin.add(cellSize);
                const listBase = uint(giLightDataCount()).add(cellIndex.mul(uint(GI_LIGHT_CELL_STRIDE)));
                const count = uint(0).toVar();
                Loop({ start: uint(0), end: U.lightCount, type: 'uint', condition: '<' }, ({ i: li }) => {
                    const lb = li.mul(uint(_LIGHT_STRIDE));
                    const ltype = lights.element(lb);
                    const lpos = loadLightVec3(lb, 1);
                    const lrange = lights.element(lb.add(uint(10)));
                    const dx = tslMax(tslMax(cellMin.x.sub(lpos.x), float(0.0)), lpos.x.sub(cellMax.x));
                    const dy = tslMax(tslMax(cellMin.y.sub(lpos.y), float(0.0)), lpos.y.sub(cellMax.y));
                    const dz = tslMax(tslMax(cellMin.z.sub(lpos.z), float(0.0)), lpos.z.sub(cellMax.z));
                    // Membership must be conservative: false positives cost work;
                    // a boundary false negative changes radiance.
                    const paddedRange = lrange.add(tslMax(float(1e-4), lrange.mul(float(1e-5))));
                    const intersects = dx.mul(dx).add(dy.mul(dy)).add(dz.mul(dz)).lessThanEqual(paddedRange.mul(paddedRange));
                    const member = ltype.lessThan(float(2.5)).and(
                        ltype.lessThan(float(0.5)).or(lrange.lessThanEqual(float(0.0))).or(intersects),
                    );
                    If(member, () => {
                        If(count.lessThan(uint(GI_LIGHTS_PER_CELL)), () => {
                            lights.element(listBase.add(uint(1)).add(count)).assign(float(li));
                        });
                        count.addAssign(uint(1));
                    });
                });
                lights.element(listBase).assign(float(select(
                    count.greaterThan(uint(GI_LIGHTS_PER_CELL)),
                    uint(GI_LIGHT_CELL_OVERFLOW),
                    count,
                )));
            })().compute(lightGridCellCount);
        }

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
                // Same centre convention as GiProbeNode._tileUV above. Do not add
                // another 0.5 here: octUV already addresses texel-centred samples.
                const ox = col.mul(float(TILE)).add(float(BORDER)).add(octUV.x.mul(float(OCT_RES)));
                const oy = row.mul(float(TILE)).add(float(BORDER)).add(octUV.y.mul(float(OCT_RES)));
                const uv = vec2(ox.div(U.atlasDim.x), oy.div(U.atlasDim.y));
                acc.addAssign(texture(atlas, uv).level(0).xyz.mul(w));
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
            const ro = probeTraceOrigin(probeIndex);
            const rd = normalize(rayDir(k, U.frameJitter)).toVar();

            const outRgb = vec3(0.0).toVar();
            const hitT = float(-1.0).toVar();
            const bestT = float(T_MAX).toVar();
            const bestTri = int(-1).toVar();
            const bestInst = int(-1).toVar();
            traverseClosest(ro, rd, bestT, bestTri, bestInst);

            If(bestTri.greaterThanEqual(int(0)), () => {
                hitT.assign(bestT);
                const triId = uint(bestTri);
                const instId = uint(bestInst);
                const matId = triMaterial.element(triId);
                // Vertex data is LOCAL space — shade via the hit instance: the
                // local ray reproduces the traversal's barycentrics exactly
                // (affine-invariant), the geometric normal transforms to world
                // through (M⁻¹)ᵀ, and hitPoint stays on the WORLD ray (t is
                // the same parameter in both spaces).
                const L = instLocalRay(instId, ro, rd);
                const lro = L.ro.toVar();
                const lrd = L.rd.toVar();
                const p0 = fetchVert(triVert(triId, 0));
                const p1 = fetchVert(triVert(triId, 1));
                const p2 = fetchVert(triVert(triId, 2));
                const ngRaw = instNormalToWorld(instId, normalize(cross(p1.sub(p0), p2.sub(p0))));
                const faceFwd = dot(ngRaw, rd).lessThan(float(0.0));
                const ng = ngRaw.mul(select(faceFwd, float(1.0), float(-1.0))).toVar();
                const hitPoint = ro.add(rd.mul(bestT));
                const traceBias = tslMax(
                    U.cellMin.mul(float(TRACE_SURFACE_BIAS_CELL)).mul(U.debugTraceBiasScale),
                    float(RAY_EPS).mul(U.debugRayEpsScale)
                );
                const hitPos = hitPoint.add(ng.mul(traceBias)).toVar();

                // Recover hit UV via Möller–Trumbore barycentrics (traverseClosest
                // discards them). Mirror the traversal's op order/float types so the
                // UVs are numerically identical. A real hit guarantees |det|>0.
                const e1 = p1.sub(p0);
                const e2 = p2.sub(p0);
                const pv = cross(lrd, e2);
                const det = dot(e1, pv);
                const invDet = float(1.0).div(det);
                const tv = lro.sub(p0);
                const ub = dot(tv, pv).mul(invDet);
                const qv = cross(tv, e1);
                const vb = dot(lrd, qv).mul(invDet);
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
                const radiance = emissive.mul(U.debugEmissiveScale).toVar();

                // energy-weighted diffuse albedo: metals (slot 4) and glass (slot 5)
                // don't bounce Lambert diffuse. Cap ≤0.95 so the temporal multibounce
                // series E = direct/(1−kd) is provably convergent (no runaway).
                const metal = matFloat(matId, 4);
                const transm = matFloat(matId, 5);
                const kd = clamp(baseColor.mul(U.debugAlbedoScale).mul(float(1.0).sub(metal)).mul(float(1.0).sub(transm)), vec3(0.0), vec3(0.95)).toVar();
                if (roughReflectionsEnabled) {
                    const diffuseEnergy = float(1.0).sub(metal).mul(float(1.0).sub(transm));
                    const emissivePeak = tslMax(tslMax(emissive.x, emissive.y), emissive.z);
                    const roughShadeable = diffuseEnergy.greaterThan(float(0.02)).or(emissivePeak.greaterThan(float(1e-6)));
                    If(roughShadeable.not(), () => {
                        hitT.assign(bestT.add(float(ROUGH_UNSHADED_T_BIAS)).negate());
                    });
                }

                if (clusteredGi) {
                    const shadeLight = (li) => {
                        const lb = li.mul(uint(_LIGHT_STRIDE)).toVar();
                        const ltype = lights.element(lb);
                        If(ltype.lessThan(float(2.5)), () => {
                            const lpos = loadLightVec3(lb, 1);
                            const ldir = loadLightVec3(lb, 4);
                            const lrange = lights.element(lb.add(uint(10)));
                            const isDir = ltype.lessThan(float(0.5));
                            const toLight = select(isDir, ldir.mul(-1.0), lpos.sub(hitPos));
                            const dist = select(isDir, float(1e4), tslMax(length(toLight), float(1e-4)));
                            const wi = normalize(toLight);
                            const ndl = tslMax(dot(ng, wi), float(0.0));
                            const reachesHit = isDir.or(lrange.lessThanEqual(float(0.0))).or(dist.lessThan(lrange));
                            If(ndl.greaterThan(float(0.0)).and(reachesHit), () => {
                                const eclass = lights.element(lb.add(uint(16)));
                                const isIr = tslAbs(eclass.sub(float(4.0))).lessThan(float(0.25));
                                const lcol = loadLightVec3(lb, 7).mul(select(isIr, U.nirGate.mul(U.nirGain), float(1.0)));
                                // Band-gated lights must stay black before the shadow query.
                                const lpeak = tslMax(tslMax(lcol.x, lcol.y), lcol.z);
                                If(ndl.mul(lpeak).greaterThan(float(0.0)), () => {
                                    const ldecay = lights.element(lb.add(uint(11)));
                                    const lcosAngle = lights.element(lb.add(uint(12)));
                                    const lcosPen = lights.element(lb.add(uint(13)));
                                    const isSpot = float(ltype.sub(float(2.0)).abs()).lessThan(float(0.5));
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
                                        radiance.addAssign(diffuse.mul(ndl).mul(lcol).mul(atten).mul(U.debugDirectScale));
                                    });
                                });
                            });
                        });
                    };

                    const gridU = casc[0].U;
                    const cellCount = vec3(float(gridU.resX), float(gridU.resY), float(gridU.resZ)).sub(vec3(1.0));
                    const cellSize = gridU.gridSize.div(cellCount);
                    const cellF = hitPos.sub(gridU.gridMin).div(cellSize).floor();
                    const outside = cellF.x.lessThan(float(0.0)).or(cellF.y.lessThan(float(0.0))).or(cellF.z.lessThan(float(0.0)))
                        .or(cellF.x.greaterThanEqual(cellCount.x)).or(cellF.y.greaterThanEqual(cellCount.y)).or(cellF.z.greaterThanEqual(cellCount.z));
                    const safeCell = cellF.clamp(vec3(0.0), cellCount.sub(vec3(1.0)));
                    const cx = uint(safeCell.x), cy = uint(safeCell.y), cz = uint(safeCell.z);
                    const cellsX = gridU.resX.sub(uint(1)), cellsY = gridU.resY.sub(uint(1));
                    const cellIndex = cz.mul(cellsX.mul(cellsY)).add(cy.mul(cellsX)).add(cx);
                    const listBase = uint(giLightDataCount()).add(cellIndex.mul(uint(GI_LIGHT_CELL_STRIDE)));
                    const listCount = uint(lights.element(listBase));
                    If(outside.or(listCount.greaterThan(uint(GI_LIGHTS_PER_CELL))), () => {
                        Loop({ start: uint(0), end: U.lightCount, type: 'uint', condition: '<' }, ({ i: li }) => { shadeLight(li); });
                    }).Else(() => {
                        Loop({ start: uint(0), end: listCount, type: 'uint', condition: '<' }, ({ i }) => {
                            shadeLight(uint(lights.element(listBase.add(uint(1)).add(i))));
                        });
                    });
                } else {
                    // NEE over ALL lights (count small; loop avoids sampling noise).
                    Loop({ start: uint(0), end: U.lightCount, type: 'uint', condition: '<' }, ({ i: li }) => {
                        // stride matches spectral_scene LIGHT_STRIDE (17: [16] is the
                        // emitter class — probes are RGB-domain except for the band
                        // gate below: class-4 IR illuminators only exist when the
                        // imager senses NIR, so their promoted (k,k,k) is scaled by
                        // U.nirGate (0 in the visible band, 1 under NV).
                        const lb = li.mul(uint(17)).toVar();
                        const ltype = lights.element(lb);
                        If(ltype.lessThan(float(2.5)), () => {
                            const lpos = loadLightVec3(lb, 1);
                            const ldir = loadLightVec3(lb, 4);
                            const eclass = lights.element(lb.add(uint(16)));
                            const isIr = tslAbs(eclass.sub(float(4.0))).lessThan(float(0.25));
                            const lcol = loadLightVec3(lb, 7).mul(select(isIr, U.nirGate.mul(U.nirGain), float(1.0)));
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
                            // fold the light's peak into the gate so band-gated (black) lights
                            // skip the shadow ray entirely, not just shade to zero.
                            const lpeak = tslMax(tslMax(lcol.x, lcol.y), lcol.z);
                            If(ndl.mul(lpeak).greaterThan(float(0.0)).and(isDir.or(lrange.lessThanEqual(float(0.0))).or(dist.lessThan(lrange))), () => {
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
                                    radiance.addAssign(diffuse.mul(ndl).mul(lcol).mul(atten).mul(U.debugDirectScale));
                                });
                            });
                        });
                    });
                }

                // multibounce: add last frame's irradiance × albedo (the atlas is
                // re-uploaded every tick, so this reads a valid prior field; on
                // tick 1 it reads zero, which is correct). Lambert: E·albedo/π.
                const bounce = sampleAtlas(hitPos, ng).mul(kd).mul(float(1.0).div(float(PI)));
                // hue-stable firefly rolloff (vs the old per-channel hard clamp that
                // clipped saturated bounce hues): scale by luminance, preserving chroma.
                const bl = dot(bounce, vec3(0.2126, 0.7152, 0.0722));
                // 1e-6 floor: clamp=0 with a black bounce is otherwise 0/0 = NaN → poisons history.
                const roll = U.radianceClamp.div(tslMax(tslMax(U.radianceClamp, bl), float(1e-6)));
                radiance.addAssign(bounce.mul(roll).mul(U.debugBounceScale));
                outRgb.assign(radiance);
            }).Else(() => {
                // miss → SKY radiance from the injected SH-9 (zero SH = the old
                // miss=BLACK invariant bit-for-bit). hitT stays -1 → depth moments
                // still record "far": sky light never occludes. If the receiver ALSO
                // gets diffuse IBL from scene.environment, that ambient double-counts —
                // turn environmentIntensity down/off and let the probes own the sky.
                // Basis order/constants match THREE.SphericalHarmonics3.getAt.
                const x = rd.x, y = rd.y, z = rd.z;
                const sky = U.skySH[0].mul(0.282095)
                    .add(U.skySH[1].mul(y.mul(0.488603)))
                    .add(U.skySH[2].mul(z.mul(0.488603)))
                    .add(U.skySH[3].mul(x.mul(0.488603)))
                    .add(U.skySH[4].mul(x.mul(y).mul(1.092548)))
                    .add(U.skySH[5].mul(y.mul(z).mul(1.092548)))
                    .add(U.skySH[6].mul(z.mul(z).mul(3.0).sub(1.0).mul(0.315392)))
                    .add(U.skySH[7].mul(x.mul(z).mul(1.092548)))
                    .add(U.skySH[8].mul(x.mul(x).sub(y.mul(y)).mul(0.546274)));
                outRgb.assign(sky.max(vec3(0.0)).mul(U.skyIntensity)); // SH ringing can dip negative → clamp
            });

            const rb = slot.mul(uint(raysPerProbe)).add(k).mul(uint(4)).toVar();
            rayData.element(rb).assign(outRgb.x);
            rayData.element(rb.add(uint(1))).assign(outRgb.y);
            rayData.element(rb.add(uint(2))).assign(outRgb.z);
            rayData.element(rb.add(uint(3))).assign(hitT);
        })().compute(updatedCap() * raysPerProbe);

        const emitterVisKernel = Fn(() => {
            const gid = instanceIndex.toVar();
            const slot = gid.div(uint(GI_EMITTER_INJECT_CAP)).toVar();
            const emitterIndex = gid.mod(uint(GI_EMITTER_INJECT_CAP)).toVar();
            If(slot.greaterThanEqual(U.updatedCount).or(emitterIndex.greaterThanEqual(U.emitterCount)), () => { Return(); });
            const probeIndex = U.probeOffset.add(slot).mod(U.probeTotal).toVar();
            const ro = probeTraceOrigin(probeIndex);
            const lb = U.emitterBase.add(emitterIndex).mul(uint(_LIGHT_STRIDE)).toVar();
            const center = loadLightVec3(lb, 1);
            const sourceRadius = tslMax(lights.element(lb.add(uint(14))), float(0.0));
            const emitterAxis = center.sub(ro);
            const dist = length(emitterAxis).toVar();
            const sample = float(0.0).toVar();
            If(dist.greaterThan(tslMax(sourceRadius, float(1e-3))), () => {
                const d = emitterAxis.div(dist);
                const emitterUp = select(tslAbs(d.y).lessThan(float(0.999)), vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0));
                const emitterTangent = normalize(cross(emitterUp, d));
                const emitterBitangent = cross(d, emitterTangent);
                const emitterSeed = float(probeIndex).mul(12.9898).add(float(emitterIndex).mul(78.233)).add(U.emitterVisSeed.mul(37.719));
                const emitterHash0 = sin(emitterSeed).mul(43758.5453);
                const emitterHash1 = sin(emitterSeed.add(19.19)).mul(24634.6345);
                const emitterU = emitterHash0.sub(floor(emitterHash0));
                const emitterV = emitterHash1.sub(floor(emitterHash1));
                const emitterDiskRadius = sqrt(emitterU).mul(sourceRadius);
                const emitterDiskAngle = emitterV.mul(float(PI).mul(2.0));
                const target = center.add(
                    emitterTangent.mul(cos(emitterDiskAngle).mul(emitterDiskRadius))
                        .add(emitterBitangent.mul(sin(emitterDiskAngle).mul(emitterDiskRadius))),
                );
                const shadowDir = normalize(target.sub(ro));
                const traceBias = tslMax(
                    U.cellMin.mul(float(TRACE_SURFACE_BIAS_CELL)).mul(U.debugTraceBiasScale),
                    float(RAY_EPS).mul(U.debugRayEpsScale),
                );
                const blocked = traverseAny(
                    ro,
                    shadowDir,
                    tslMax(dist.sub(sourceRadius).sub(traceBias), float(1e-4)),
                );
                sample.assign(select(blocked.lessThan(float(0.5)), float(1.0), float(0.0)));
            });
            const visIndex = probeIndex.mul(uint(GI_EMITTER_INJECT_CAP)).add(emitterIndex);
            const vis = emitterVis.element(visIndex);
            emitterVis.element(visIndex).assign(mix(sample, vis, float(GI_EMITTER_VIS_RETENTION)));
        })().compute(updatedCap() * GI_EMITTER_INJECT_CAP);

        const injectEmitterVirtualRays = ({
            probeIndex, texelDir, acc, hit = null, wsum,
            lAcc = null, lAcc2 = null,
            secondaryAcc = null, secondaryHit = null, secondaryWsum = null,
            primaryWeight, secondaryWeight = null,
        }) => {
            If(U.emitterCount.greaterThan(uint(0)), () => {
                const ro = probeTraceOrigin(probeIndex);
                Loop({ start: uint(0), end: U.emitterCount, type: 'uint', condition: '<' }, ({ i: emitterIndex }) => {
                    const lb = U.emitterBase.add(emitterIndex).mul(uint(_LIGHT_STRIDE)).toVar();
                    const center = loadLightVec3(lb, 1);
                    const sourceRadius = tslMax(lights.element(lb.add(uint(14))), float(0.0));
                    const emitterAxis = center.sub(ro);
                    const dist = length(emitterAxis).toVar();
                    If(dist.greaterThan(tslMax(sourceRadius, float(1e-3))), () => {
                        const d = emitterAxis.div(dist);
                        // Solid angle from the record's Cauchy projected area (slot 15):
                        // (A/4)/d^2 equals the sphere solid angle in the far field but
                        // does not blow up to a hemisphere when a long thin emitter's
                        // bounding sphere swallows the probe. Color slots are surface
                        // RADIANCE, so Omega carries ALL of the geometry term.
                        const projArea = tslMax(lights.element(lb.add(uint(15))), float(0.0));
                        const Omega = tslMin(float(2.0 * PI), projArea.div(dist.mul(dist)));
                        const kVirtual = tslMin(
                            float(raysPerProbe),
                            float(raysPerProbe).mul(Omega).div(float(4.0 * PI)),
                        );
                        const visIndex = probeIndex.mul(uint(GI_EMITTER_INJECT_CAP)).add(emitterIndex);
                        const L = loadLightVec3(lb, 7).mul(emitterVisRead.element(visIndex));
                        const w = primaryWeight(texelDir, d);
                        const wk = w.mul(kVirtual);
                        acc.addAssign(L.mul(wk));
                        wsum.addAssign(wk);
                        if (hit) hit.addAssign(wk);
                        if (lAcc && lAcc2) {
                            const lum = dot(L, vec3(0.2126, 0.7152, 0.0722));
                            lAcc.addAssign(lum.mul(wk));
                            lAcc2.addAssign(lum.mul(lum).mul(wk));
                        }
                        if (secondaryAcc && secondaryHit && secondaryWsum && secondaryWeight) {
                            const secondaryWk = secondaryWeight(texelDir, d).mul(kVirtual);
                            secondaryAcc.addAssign(L.mul(secondaryWk));
                            secondaryHit.addAssign(secondaryWk);
                            secondaryWsum.addAssign(secondaryWk);
                        }
                    });
                });
            });
        };

        // ── BLEND: one 64-thread workgroup per updated probe. Every lane first
        // caches its share of ray scratch/directions, then the 36 interior lanes
        // cosine-gather the unique 6×6 oct texels. Upload mirrors those texels into
        // the 28 gutter positions, so evaluating gutters here was pure duplicate
        // work (and repeated every ray load + trig operation 28 extra times).
        const blendKernel = Fn(() => {
            const rayCache = workgroupArray('vec4', raysPerProbe);
            const dirCache = workgroupArray('vec4', raysPerProbe);
            const slot = workgroupId.x;
            const lane = invocationLocalIndex;
            const activeProbe = slot.lessThan(U.updatedCount);
            const loadK = lane.toVar();
            Loop(loadK.lessThan(uint(raysPerProbe)), () => {
                If(activeProbe, () => {
                    const rb = slot.mul(uint(raysPerProbe)).add(loadK).mul(uint(4));
                    rayCache.element(loadK).assign(vec4(
                        rayData.element(rb), rayData.element(rb.add(uint(1))),
                        rayData.element(rb.add(uint(2))), rayData.element(rb.add(uint(3))),
                    ));
                    dirCache.element(loadK).assign(vec4(normalize(rayDir(loadK, U.frameJitter)), 0.0));
                });
                loadK.addAssign(uint(PROBE_WORKGROUP_SIZE));
            });
            workgroupBarrier();

            // A barrier forbids invocation-local early returns. Guard all output
            // work after every lane has reached it.
            If(activeProbe.and(lane.lessThan(uint(OCT_RES * OCT_RES))), () => {
            const lx = lane.mod(uint(OCT_RES)).add(uint(BORDER)).toVar();
            const ly = lane.div(uint(OCT_RES)).add(uint(BORDER)).toVar();
            const local = ly.mul(uint(TILE)).add(lx).toVar();
            const probeIndex = U.probeOffset.add(slot).mod(U.probeTotal).toVar();

            // Interior texel direction; upload builds the canonical mirrored
            // gutter from these unique samples.
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
            const sAcc = roughReflectionsEnabled ? vec3(0.0).toVar() : null;
            const sHit = roughReflectionsEnabled ? float(0.0).toVar() : null;
            const sWsum = roughReflectionsEnabled ? float(0.0).toVar() : null;
            const LUMA = vec3(0.2126, 0.7152, 0.0722);
            const skyEnabled = roughReflectionsEnabled
                ? select(U.skyIntensity.greaterThan(float(0.0)), float(1.0), float(0.0))
                : null;
            const skyValid = roughReflectionsEnabled
                ? U.reflectionSkyFallback.mul(U.skyConfigured).mul(skyEnabled)
                : null;
            Loop({ start: uint(0), end: uint(raysPerProbe), type: 'uint', condition: '<' }, ({ i: k }) => {
                const cachedRay = rayCache.element(k).toVar();
                const rrgb = cachedRay.xyz;
                const hitT = cachedRay.w;
                const rdir = dirCache.element(k).xyz;
                const cd = tslMax(dot(dir, rdir), float(0.0));
                const cw = pow(cd, U.debugCosinePower.max(float(1e-4)));
                acc.addAssign(rrgb.mul(cw));
                wsum.addAssign(cw);
                const rl = dot(rrgb, LUMA);
                lAcc.addAssign(rl.mul(cw));
                lAcc2.addAssign(rl.mul(rl).mul(cw));
                // depth moments: miss → "far" so the probe stays visible that way.
                const roughUnshaded = hitT.lessThan(float(-1.5));
                const decodedHitT = select(roughUnshaded, hitT.negate().sub(float(ROUGH_UNSHADED_T_BIAS)), hitT);
                const rdist = select(hitT.equal(float(-1.0)), U.maxDist, decodedHitT);
                const dw = pow(cw, U.depthSharpness);
                dAcc.addAssign(rdist.mul(dw));
                dAcc2.addAssign(rdist.mul(rdist).mul(dw));
                dwsum.addAssign(dw);
                if (roughReflectionsEnabled) {
                    // Fixed power-8 lobe as ((d²)²)²: materially cheaper than another
                    // dynamic pow and intentionally independent of the diffuse debug knob.
                    const cd2 = cd.mul(cd);
                    const cd4 = cd2.mul(cd2);
                    const sw = cd4.mul(cd4);
                    // Local hits are always valid. A true miss (-1) can explicitly use
                    // the injected SH sky; encoded unshaded metal/glass hits (< -1.5)
                    // leave prior radiance unchanged so other layers can own them.
                    const valid = select(
                        hitT.greaterThanEqual(float(0.0)),
                        float(1.0),
                        select(hitT.equal(float(-1.0)), skyValid, float(0.0)),
                    );
                    sAcc.addAssign(rrgb.mul(sw).mul(valid));
                    sHit.addAssign(sw.mul(valid));
                    sWsum.addAssign(sw);
                }
            });
            injectEmitterVirtualRays({
                probeIndex,
                texelDir: dir,
                acc,
                wsum,
                lAcc,
                lAcc2,
                secondaryAcc: sAcc,
                secondaryHit: sHit,
                secondaryWsum: sWsum,
                primaryWeight: (texelDir, emitterDir) => {
                    const cd = tslMax(dot(texelDir, emitterDir), float(0.0));
                    return pow(cd, U.debugCosinePower.max(float(1e-4)));
                },
                secondaryWeight: roughReflectionsEnabled ? ((texelDir, emitterDir) => {
                    const cd = tslMax(dot(texelDir, emitterDir), float(0.0));
                    const cd2 = cd.mul(cd);
                    const cd4 = cd2.mul(cd2);
                    return cd4.mul(cd4);
                }) : null,
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
            const varFloor = U.debugTempVarEps.add(lumRef.mul(lumRef).mul(U.debugTempVarRel));
            const sigma = sqrt(prevVar.add(curVar).add(varFloor));
            const deltaL = curL.sub(prevL);
            const absDelta = tslAbs(deltaL);
            const s0 = sigma.mul(U.debugTempChangeSigma0);
            const s1 = sigma.mul(U.tempChangeSigma1.max(U.debugTempChangeSigma0.add(0.01)));
            const changeW = clamp(absDelta.sub(s0).div(s1.sub(s0).max(float(1e-6))), float(0.0), float(1.0));
            // Build every signal policy in the slider's 60 Hz reference domain, then
            // normalize the FINAL retention for this probe's real update interval.
            // Normalizing U.hysteresis first and subtracting a fixed change drop after it
            // made that drop happen once per frame: 120/240 Hz therefore admitted much
            // more fresh jitter per second than 60 Hz, which presented as FPS flicker.
            const rawNoiseH = U.hysteresis.add(float(1.0).sub(U.hysteresis).mul(U.debugTempNoiseHBoost));
            const rawChangeH = tslMax(U.hysteresis.sub(U.tempChangeHDrop), U.debugTempMinChangeH);
            const rawHEff = mix(rawNoiseH, rawChangeH, changeW);
            const hEff = pow(rawHEff.clamp(1e-6, 1.0), U.hysteresisExponent);
            // Reflection history must remain a true time semigroup. Reusing the
            // diffuse changeW here made rough reflections frame-rate dependent:
            // changeW is recomputed from evolving diffuse moments on every render
            // substep, so 30 and 240 Hz followed different nonlinear trajectories.
            // Scene edits therefore transition the reflection lobes at the steady /
            // noisy reference rate — smooth by construction, normalized exactly
            // once by elapsed time. (No global reactive ramp: constant policy.)
            const steadyReflectionH = pow(rawNoiseH.clamp(1e-6, 1.0), U.hysteresisExponent);
            const h = select(wasBlack, float(0.0), hEff);
            // Firefly clamp — AUTHORITATIVE, banded by PRIOR variance only. Two
            // dead-knob traps in the old form: (1) the changeW mix escaped to the
            // raw mean, so any spike past σ1 bypassed the clamp entirely and
            // tempClampSigma had no effect above ~σ1; (2) the band folded in
            // curVar, so a single-tick firefly inflated its own clamp gate.
            // changeW keeps its retention role (rawHEff above). Real changes
            // still release fast: the m2 history below blends the RAW current
            // moments, so prevVar inflates on the first clamped update and the
            // band widens geometrically over the next few revisits.
            const band = sqrt(prevVar.add(varFloor)).mul(U.tempClampSigma);
            const clampScale = tslMin(float(1.0), band.div(absDelta.max(float(1e-6))));
            const clipped = prev.add(meanRad.sub(prev).mul(clampScale));
            const candidate = select(wasBlack, meanRad, clipped);
            const blended = mix(candidate, prev, h);
            irr.element(ib).assign(blended.x);
            irr.element(ib.add(uint(1))).assign(blended.y);
            irr.element(ib.add(uint(2))).assign(blended.z);
            // luminance 2nd moment E[L²] in the FREE 4th slot (buffer-only; the upload
            // keeps atlas.w=1.0 so the fragment sampler never sees it). luma is linear
            // → E[luma]=luma(E[rgb]), so variance = max(0, M2 − luma(rgb)²) anywhere.
            const m2 = mix(curM2, prevM2, h);
            irr.element(ib.add(uint(3))).assign(m2);

            // Depth is written for every solved probe texel and is strictly positive
            // after its first update (a miss stores maxDist). It is therefore the
            // existing allocation-free initialization sentinel shared by diffuse,
            // rough reflection, and depth history. Reflection RGBA cannot serve this
            // purpose: zero coverage is a valid converged state when PMREM owns misses.
            const db = probeIndex.mul(uint(TILE * TILE)).add(local).mul(uint(2)).toVar();
            const dprev = vec2(depthS.element(db), depthS.element(db.add(uint(1))));
            const dWasZero = dprev.x.lessThan(float(1e-6));

            if (roughReflectionsEnabled) {
                // Same accepted cadence as diffuse, but deliberately independent of its
                // nonlinear per-texel change detector (see steadyReflectionH above).
                const sb = probeIndex.mul(uint(TILE * TILE)).add(local).mul(uint(4)).toVar();
                const sPrev = vec4(
                    roughSpecular.element(sb), roughSpecular.element(sb.add(uint(1))),
                    roughSpecular.element(sb.add(uint(2))), roughSpecular.element(sb.add(uint(3))),
                );
                const sDen = sWsum.max(float(1e-4));
                const sCur = vec4(sAcc.div(sDen), sHit.div(sDen).clamp(0.0, 1.0));
                // Never infer initialization from sPrev energy. Transparent black is
                // a valid reflection sample and must retain history across sparse hits.
                const sh = select(dWasZero, float(0.0), steadyReflectionH);
                const sBlended = mix(sCur, sPrev, sh);
                roughSpecular.element(sb).assign(sBlended.x);
                roughSpecular.element(sb.add(uint(1))).assign(sBlended.y);
                roughSpecular.element(sb.add(uint(2))).assign(sBlended.z);
                roughSpecular.element(sb.add(uint(3))).assign(sBlended.w);
            }

            // depth moments: same hysteresis; fill instantly when unseeded.
            const rawDepthH = U.hysteresis.mul(U.debugDepthHistoryScale).clamp(0.0, 0.999);
            const depthH = pow(rawDepthH.max(float(1e-6)), U.hysteresisExponent);
            const dh = select(dWasZero, float(0.0), depthH);
            const dblended = mix(vec2(meanR, meanR2), dprev, dh);
            depthS.element(db).assign(dblended.x);
            depthS.element(db.add(uint(1))).assign(dblended.y);
            });
        })().compute(updatedCap() * PROBE_WORKGROUP_SIZE, [PROBE_WORKGROUP_SIZE]);

        // ── GLOSSY: high-angular-resolution resolve from the SAME ray scratch.
        // It is a fourth dispatch but performs no BVH traversal and traces no rays.
        // Numerator/support history is accumulated before division so sparse power-64
        // ray sets converge without giving one weak sample a full frame of authority.
        const glossyKernel = glossyReflectionsEnabled ? Fn(() => {
            const rayCache = workgroupArray('vec4', raysPerProbe);
            const dirCache = workgroupArray('vec4', raysPerProbe);
            const group = workgroupId.x;
            const lane = invocationLocalIndex;
            const slot = group.div(uint(glossyGroupsPerProbe)).toVar();
            const local = group.mod(uint(glossyGroupsPerProbe))
                .mul(uint(PROBE_WORKGROUP_SIZE)).add(lane).toVar();
            const activeProbe = slot.lessThan(U.updatedCount);
            const loadK = lane.toVar();
            Loop(loadK.lessThan(uint(raysPerProbe)), () => {
                If(activeProbe, () => {
                    const rb = slot.mul(uint(raysPerProbe)).add(loadK).mul(uint(4));
                    rayCache.element(loadK).assign(vec4(
                        rayData.element(rb), rayData.element(rb.add(uint(1))),
                        rayData.element(rb.add(uint(2))), rayData.element(rb.add(uint(3))),
                    ));
                    dirCache.element(loadK).assign(vec4(normalize(rayDir(loadK, U.frameJitter)), 0.0));
                });
                loadK.addAssign(uint(PROBE_WORKGROUP_SIZE));
            });
            workgroupBarrier();

            let resolvesTexel = activeProbe.and(local.lessThan(uint(glossyTile * glossyTile)));
            // High quality interleaves directional texels across two solves. Every
            // probe still receives service on every batch (no round-robin starvation),
            // while half of the expensive 64-ray lobe loops stay inactive.
            if (glossyUpdateInterval > 1) {
                resolvesTexel = resolvesTexel.and(local.mod(uint(glossyUpdateInterval)).equal(U.glossyPhase));
            }
            If(resolvesTexel, () => {
            const probeIndex = U.probeOffset.add(slot).mod(U.probeTotal).toVar();
            const lx = local.mod(uint(glossyTile)).toVar();
            const ly = local.div(uint(glossyTile)).toVar();

            // Canonical mirrored oct gutter, resolved before evaluating direction.
            const edge = uint(glossyTile - 1);
            const lo = uint(BORDER);
            const hi = uint(BORDER + glossyOctRes - 1);
            const onLeft = lx.equal(uint(0));
            const onRight = lx.equal(edge);
            const onTop = ly.equal(uint(0));
            const onBottom = ly.equal(edge);
            const onColumnBorder = onLeft.or(onRight);
            const onRowBorder = onTop.or(onBottom);
            const onCorner = onColumnBorder.and(onRowBorder);
            const sx = select(
                onCorner,
                select(onLeft, hi, lo),
                select(onRowBorder, edge.sub(lx), select(onColumnBorder, select(onLeft, lo, hi), lx)),
            ).toVar();
            const sy = select(
                onCorner,
                select(onTop, hi, lo),
                select(onRowBorder, select(onTop, lo, hi), select(onColumnBorder, edge.sub(ly), ly)),
            ).toVar();
            const u = float(sx).sub(float(BORDER)).add(0.5).div(float(glossyOctRes));
            const v = float(sy).sub(float(BORDER)).add(0.5).div(float(glossyOctRes));
            const dir = octDecodeNode(vec2(u, v), TSL).toVar();

            const gAcc = vec3(0.0).toVar();
            const gHit = float(0.0).toVar();
            const gWsum = float(0.0).toVar();
            const skyEnabled = select(U.skyIntensity.greaterThan(float(0.0)), float(1.0), float(0.0));
            const skyValid = U.reflectionSkyFallback.mul(U.skyConfigured).mul(skyEnabled);
            Loop({ start: uint(0), end: uint(raysPerProbe), type: 'uint', condition: '<' }, ({ i: k }) => {
                const cachedRay = rayCache.element(k).toVar();
                const rrgb = cachedRay.xyz;
                const hitT = cachedRay.w;
                const rdir = dirCache.element(k).xyz;
                const cd = tslMax(dot(dir, rdir), float(0.0));
                const cd2 = cd.mul(cd);
                const cd4 = cd2.mul(cd2);
                const cd8 = cd4.mul(cd4);
                const cd16 = cd8.mul(cd8);
                const cd32 = cd16.mul(cd16);
                const gw = cd32.mul(cd32); // power 64
                const valid = select(
                    hitT.greaterThanEqual(float(0.0)),
                    float(1.0),
                    select(hitT.equal(float(-1.0)), skyValid, float(0.0)),
                );
                gAcc.addAssign(rrgb.mul(gw).mul(valid));
                gHit.addAssign(gw.mul(valid));
                gWsum.addAssign(gw);
            });
            injectEmitterVirtualRays({
                probeIndex,
                texelDir: dir,
                acc: gAcc,
                hit: gHit,
                wsum: gWsum,
                primaryWeight: (texelDir, emitterDir) => {
                    const cd = tslMax(dot(texelDir, emitterDir), float(0.0));
                    const cd2 = cd.mul(cd);
                    const cd4 = cd2.mul(cd2);
                    const cd8 = cd4.mul(cd4);
                    const cd16 = cd8.mul(cd8);
                    const cd32 = cd16.mul(cd16);
                    return cd32.mul(cd32);
                },
            });

            const gt = probeIndex.mul(uint(glossyTile * glossyTile)).add(local).toVar();
            const gb = gt.mul(uint(4)).toVar();
            const prevNum = vec4(
                glossySpecular.element(gb), glossySpecular.element(gb.add(uint(1))),
                glossySpecular.element(gb.add(uint(2))), glossySpecular.element(gb.add(uint(3))),
            );
            const prevDen = glossyWeight.element(gt);
            // Make persisted support invariant to setRays(): both numerator and
            // denominator are per-ray means, so a same-dimension 64→256 rebuild does
            // not give the new sample four times the configured temporal authority.
            const invRayCount = float(1.0 / raysPerProbe);
            const curNum = vec4(gAcc, gHit).mul(invRayCount);
            const curDen = gWsum.mul(invRayCount);
            const empty = prevDen.lessThan(float(1e-6));
            // Match the rough cache's steady/noisy reference retention. A dedicated
            // glossy change detector is intentionally avoided: power-64 support is
            // sparse, so rotating ray sets would repeatedly look like real changes.
            // Numerator and support MUST share this exact coefficient or the resolved
            // colour/coverage ratio pumps as cadence changes.
            const glossyReferenceH = U.hysteresis.add(
                float(1.0).sub(U.hysteresis).mul(U.debugTempNoiseHBoost),
            );
            const glossyH = pow(
                glossyReferenceH.clamp(1e-6, 1.0),
                U.hysteresisExponent.mul(float(glossyUpdateInterval)),
            );
            const gh = select(empty, float(0.0), glossyH);
            const num = mix(curNum, prevNum, gh).toVar();
            const den = mix(curDen, prevDen, gh).max(float(1e-6)).toVar();
            glossySpecular.element(gb).assign(num.x);
            glossySpecular.element(gb.add(uint(1))).assign(num.y);
            glossySpecular.element(gb.add(uint(2))).assign(num.z);
            glossySpecular.element(gb.add(uint(3))).assign(num.w);
            glossyWeight.element(gt).assign(den);

            const col = probeIndex.mod(uint(C.glossyTilesX));
            const row = probeIndex.div(uint(C.glossyTilesX));
            const tx = col.mul(uint(glossyTile)).add(lx);
            const ty = row.mul(uint(glossyTile)).add(ly);
            const resolved = vec4(num.xyz.div(den), num.w.div(den).clamp(0.0, 1.0));
            textureStore(glossySpecularAtlas, uvec2(tx, ty), resolved).toWriteOnly();
            });
        })().compute(
            updatedCap() * glossyGroupsPerProbe * PROBE_WORKGROUP_SIZE,
            [PROBE_WORKGROUP_SIZE],
        ) : null;

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
            if (roughReflectionsEnabled) {
                textureStore(roughSpecularAtlas, uvec2(tx, ty), vec4(0.0)).toWriteOnly();
            }
            textureStore(depthAtlas, uvec2(tx, ty), vec4(0.0, 0.0, 0.0, 1.0)).toWriteOnly();
        })().compute(probeTotal * TILE * TILE);

        const clearGlossyAtlasKernel = glossyReflectionsEnabled ? Fn(() => {
            const gid = instanceIndex.toVar();
            const total = uint(probeTotal * glossyTile * glossyTile);
            If(gid.greaterThanEqual(total), () => { Return(); });
            const probeIndex = gid.div(uint(glossyTile * glossyTile)).toVar();
            const local = gid.mod(uint(glossyTile * glossyTile)).toVar();
            const lx = local.mod(uint(glossyTile)).toVar();
            const ly = local.div(uint(glossyTile)).toVar();
            const col = probeIndex.mod(uint(C.glossyTilesX));
            const row = probeIndex.div(uint(C.glossyTilesX));
            textureStore(
                glossySpecularAtlas,
                uvec2(col.mul(uint(glossyTile)).add(lx), row.mul(uint(glossyTile)).add(ly)),
                vec4(0.0),
            ).toWriteOnly();
        })().compute(probeTotal * glossyTile * glossyTile) : null;

        const clearEmitterVisKernel = Fn(() => {
            const gid = instanceIndex.toVar();
            const total = uint(probeTotal * GI_EMITTER_INJECT_CAP);
            If(gid.greaterThanEqual(total), () => { Return(); });
            emitterVis.element(gid).assign(float(0.0));
        })().compute(probeTotal * GI_EMITTER_INJECT_CAP);

        // ── UPLOAD: one workgroup per updated probe. The 36 interior lanes run
        // the expensive 3×3 bilateral filter once, cache the final values, then all
        // 64 lanes write the canonical interior/gutter destinations. Previously the
        // 28 gutter lanes repeated an identical filter for their mirrored source.
        const uploadKernel = Fn(() => {
            const irradianceCache = workgroupArray('vec4', OCT_RES * OCT_RES);
            const roughCache = roughReflectionsEnabled
                ? workgroupArray('vec4', OCT_RES * OCT_RES)
                : null;
            const depthCache = workgroupArray('vec4', OCT_RES * OCT_RES);
            const slot = workgroupId.x;
            const lane = invocationLocalIndex;
            const activeProbe = slot.lessThan(U.updatedCount);
            const probeIndex = U.probeOffset.add(slot).mod(U.probeTotal).toVar();
            const probeBase = probeIndex.mul(uint(TILE * TILE)).toVar();

            If(activeProbe.and(lane.lessThan(uint(OCT_RES * OCT_RES))), () => {
                const sx = lane.mod(uint(OCT_RES)).add(uint(BORDER)).toVar();
                const sy = lane.div(uint(OCT_RES)).add(uint(BORDER)).toVar();
                const probeTexel = probeBase.add(sy.mul(uint(TILE))).add(sx).toVar();
                // Reads the read-only history and writes only workgroup memory, so
                // denoising never feeds back into temporal accumulation.
                const LUMA = vec3(0.2126, 0.7152, 0.0722);
                const ib = probeTexel.mul(uint(4));
                const eC = vec3(
                    irrRead.element(ib), irrRead.element(ib.add(uint(1))), irrRead.element(ib.add(uint(2))),
                ).toVar();
                const lumaC = dot(eC, LUMA);
                const varC = tslMax(irrRead.element(ib.add(uint(3))).sub(lumaC.mul(lumaC)), float(0.0));
                const sxI = int(sx); const syI = int(sy);
                const facc = vec3(0.0).toVar();
                const fwsum = float(0.0).toVar();
                const smW = float(1.0).add(U.filterSmooth.mul(float(6.0)));
                const kEff = float(GI_FILTER_K).mul(smW).mul(U.debugFilterKScale);
                const relEff = float(GI_FILTER_REL).mul(smW).mul(U.debugFilterRelScale);
                for (let jy = -1; jy <= 1; jy++) {
                    for (let jx = -1; jx <= 1; jx++) {
                        const gw = Math.exp(-(jx * jx + jy * jy) * 0.5);
                        const nx = sxI.add(int(jx)).clamp(int(BORDER), int(BORDER + OCT_RES - 1)).toUint();
                        const ny = syI.add(int(jy)).clamp(int(BORDER), int(BORDER + OCT_RES - 1)).toUint();
                        const nIb = probeBase.add(ny.mul(uint(TILE))).add(nx).mul(uint(4));
                        const en = vec3(
                            irrRead.element(nIb), irrRead.element(nIb.add(uint(1))), irrRead.element(nIb.add(uint(2))),
                        );
                        const dLum = dot(en, LUMA).sub(lumaC);
                        const es = exp(dLum.mul(dLum).div(
                            varC.mul(kEff).add(tslMax(
                                float(GI_FILTER_EPS).mul(U.debugFilterEpsScale),
                                lumaC.mul(lumaC).mul(relEff),
                            )).max(float(1e-8)),
                        ).mul(-1.0));
                        const w = float(gw).mul(es);
                        facc.addAssign(en.mul(w));
                        fwsum.addAssign(w);
                    }
                }
                const filtered = facc.div(fwsum.max(float(1e-4)));
                irradianceCache.element(lane).assign(vec4(mix(eC, filtered, U.filterStrength), 1.0));

                if (roughReflectionsEnabled) {
                    const sb = probeTexel.mul(uint(4));
                    roughCache.element(lane).assign(vec4(
                        roughSpecularRead.element(sb), roughSpecularRead.element(sb.add(uint(1))),
                        roughSpecularRead.element(sb.add(uint(2))), roughSpecularRead.element(sb.add(uint(3))),
                    ));
                }
                const db = probeTexel.mul(uint(2));
                depthCache.element(lane).assign(vec4(
                    depthRead.element(db), depthRead.element(db.add(uint(1))), 0.0, 1.0,
                ));
            });
            workgroupBarrier();

            If(activeProbe, () => {
                const local = lane;
                const lx = local.mod(uint(TILE)).toVar();
                const ly = local.div(uint(TILE)).toVar();
                const col = probeIndex.mod(U.resX);
                const row = probeIndex.div(U.resX.mul(U.resY)).mul(U.resY)
                    .add(probeIndex.div(U.resX).mod(U.resY));
                const tx = col.mul(uint(TILE)).add(lx);
                const ty = row.mul(uint(TILE)).add(ly);

                // Canonical octahedral gutter: mirror final interior edge/corner
                // values into the border, now by indexing the workgroup cache.
                const edge = uint(TILE - 1);
                const lo = uint(BORDER);
                const hi = uint(BORDER + OCT_RES - 1);
                const onLeft = lx.equal(uint(0));
                const onRight = lx.equal(edge);
                const onTop = ly.equal(uint(0));
                const onBottom = ly.equal(edge);
                const onColumnBorder = onLeft.or(onRight);
                const onRowBorder = onTop.or(onBottom);
                const onCorner = onColumnBorder.and(onRowBorder);
                const sx = select(
                    onCorner,
                    select(onLeft, hi, lo),
                    select(onRowBorder, edge.sub(lx), select(onColumnBorder, select(onLeft, lo, hi), lx)),
                ).toVar();
                const sy = select(
                    onCorner,
                    select(onTop, hi, lo),
                    select(onRowBorder, select(onTop, lo, hi), select(onColumnBorder, edge.sub(ly), ly)),
                ).toVar();
                const sourceLane = sy.sub(uint(BORDER)).mul(uint(OCT_RES))
                    .add(sx.sub(uint(BORDER))).toVar();
                textureStore(atlas, uvec2(tx, ty), irradianceCache.element(sourceLane)).toWriteOnly();
                if (roughReflectionsEnabled) {
                    textureStore(roughSpecularAtlas, uvec2(tx, ty), roughCache.element(sourceLane)).toWriteOnly();
                }
                textureStore(depthAtlas, uvec2(tx, ty), depthCache.element(sourceLane)).toWriteOnly();
            });
        })().compute(updatedCap() * PROBE_WORKGROUP_SIZE, [PROBE_WORKGROUP_SIZE]);

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
                const bestInst = int(-1).toVar();
                traverseClosest(ro, rd, bestT, bestTri, bestInst);
                If(bestTri.greaterThanEqual(int(0)), () => {
                    hits.addAssign(float(1.0));
                    const triId = uint(bestTri);
                    const p0 = fetchVert(triVert(triId, 0));
                    const p1 = fetchVert(triVert(triId, 1));
                    const p2 = fetchVert(triVert(triId, 2));
                    // local verts → world geometric normal via the hit instance
                    const ng = instNormalToWorld(uint(bestInst), normalize(cross(p1.sub(p0), p2.sub(p0))));
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

        const solveKernels = [traceKernel, emitterVisKernel, blendKernel];
        if (glossyKernel) solveKernels.push(glossyKernel);
        solveKernels.push(uploadKernel);

        const gpu = {
            buffers, traceKernel, emitterVisKernel, blendKernel, glossyKernel, uploadKernel,
            solveKernels,
            clearAtlasKernel, clearGlossyAtlasKernel, clearEmitterVisKernel, classifyKernel, uploadStateKernel, lightGridKernel,
            atlas, roughSpecularAtlas, glossySpecularAtlas, depthAtlas, stateAtlas,
            irrBuffer, roughSpecularBuffer, glossySpecularBuffer, glossyWeightBuffer,
            depthBuffer, stateBuffer, emitterVisBuffer, rayBuffer,
            lightGridCellCount,
            glossyGroupsPerProbe,
            sceneResource: sharedScene,
            lightResource: sharedLights,
            maps: sharedScene.maps,
            lightCount: clusteredGi ? giSelectedCount : giLegacyLightCount,
            // Scratch + compute grids above were sized from updatedCap() at THIS
            // moment; the tick clamps its dispatch to this snapshot so a raised
            // budget ceiling cannot overrun a held (not-yet-rebuilt) kernel set.
            probeCapBuilt: updatedCap(),
        };
        // Retain only after the complete graph exists. If TSL construction throws,
        // the active roots remain the sole owners and no half-built cascade leaks a ref.
        retainSceneResource(sharedScene);
        retainLightResource(sharedLights);
        return gpu;
    }

    function totalUnionProbes() {
        return casc[0].probeTotal + (cascades > 1 ? casc[1].probeTotal : 0);
    }
    function updatedCap() {
        // RAY-budget-based per-tick probe count: a fixed trace budget (RAYS_PER_TICK)
        // divided by the live ray count, ceilinged at MAX_PROBES_PER_TICK. GPU cost per
        // tick is ~constant regardless of rays/probe, and the ray scratch stays bounded
        // (≈ RAYS_PER_TICK × 16 B). Small/medium grids fit whole → every probe updates
        // EVERY tick; the cadence exponent keeps their wall-time stability unchanged. The old
        // union/4-with-128-ceiling cap made each texel wait ~10 frames per update, which
        // is why re-convergence took ~10 s and low hysteresis was the only way to speed
        // it up... at the price of flicker). Also sizes the per-cascade ray scratch —
        // this is the BUILD-TIME ceiling; the live per-tick count is tickCap() below.
        return Math.max(1, Math.min(MAX_PROBES_PER_TICK, Math.floor(rayBudgetCeiling / Math.max(1, raysPerProbe))));
    }
    // Live per-tick probe count under the AUTO-THROTTLED ray budget (≤ updatedCap()).
    function tickCap() {
        return Math.max(1, Math.min(updatedCap(), Math.floor(tickBudgetRays / Math.max(1, raysPerProbe))));
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
            _createBlasCache = mod.createBlasCache || null;
            _rebindMaterialMapsArenaBuild = mod.rebindMaterialMapsArenaBuild || null;
            const collectAnalyticLights = mod.collectLights || null;
            const collectEmitterRecords = mod.collectEmitterRecords || null;
            _collectLights = collectAnalyticLights ? (three, root, renderCamera = null) => {
                const records = collectAnalyticLights(three, root, renderCamera);
                if (collectEmitterRecords) records.push(...collectEmitterRecords(three, root, renderCamera));
                return records;
            } : null;
            _emissiveScaled = mod.emissiveScaled || null;
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

    // Fit atlas dimensions and the largest optional history binding by uniformly
    // shrinking res. WebGPU's common storage-binding baseline is 128 MiB: a 32³
    // glossy numerator would be ~162 MiB even though its near-square texture fits.
    const _maxDim = () => (renderer?.backend?.device?.limits?.maxTextureDimension2D || ATLAS_DIM_FALLBACK);
    const _maxGlossyHistoryBytes = () => {
        const limits = renderer?.backend?.device?.limits;
        const binding = Number(limits?.maxStorageBufferBindingSize) || STORAGE_BINDING_FALLBACK;
        const buffer = Number(limits?.maxBufferSize) || binding;
        return Math.min(binding, buffer);
    };
    function fitAtlas(resVec) {
        const maxDim = _maxDim();
        const maxGlossyBytes = _maxGlossyHistoryBytes();
        const exceedsLimits = () => resVec.x * TILE > maxDim
            || resVec.y * resVec.z * TILE > maxDim
            || (glossyReflectionsEnabled
                && resVec.x * resVec.y * resVec.z * glossyHistoryBytesPerProbe > maxGlossyBytes);
        for (let g = 0; g < 32 && exceedsLimits(); g++) {
            resVec.set(Math.max(2, Math.floor(resVec.x * 0.85)), Math.max(2, Math.floor(resVec.y * 0.85)), Math.max(2, Math.floor(resVec.z * 0.85)));
        }
        return resVec;
    }

    function setGlossyLayout(C) {
        // Near-square packing is independent of probe XYZ. Texture dimensions and
        // the larger RGBA32F history binding are both guarded by fitAtlas().
        if (!glossyReflectionsEnabled) {
            C.glossyTilesX = C.glossyTilesY = 1;
            C.glossyAtlasW = C.glossyAtlasH = 1;
            return;
        }
        C.glossyTilesX = Math.max(1, Math.ceil(Math.sqrt(C.probeTotal)));
        C.glossyTilesY = Math.max(1, Math.ceil(C.probeTotal / C.glossyTilesX));
        C.glossyAtlasW = C.glossyTilesX * glossyTile;
        C.glossyAtlasH = C.glossyTilesY * glossyTile;
    }

    // Build (or same-dim-reuse) ONE cascade's kernels+resources and wire its uniforms +
    // node bindings. Handles both the reuse (churn-free) path and the full recompile path
    // for that cascade independently. Does NOT fire onRebuilt (the caller sequences that).
    function buildOneCascade(built, c, opts = {}) {
        const C = casc[c];
        const gridMin = C.gridMin, gridSize = C.gridSize, res = C.res;
        const minCell = C.minCell;
        const atlasW = C.atlasW, atlasH = C.atlasH, probeTotal = C.probeTotal;
        const normalBias = minCell * SURFACE_NORMAL_BIAS_CELL * normalBiasScale;
        const chebyBias = minCell * GI_CHEBY_BIAS_CELL * chebyBiasScale;

        // Per-cascade same-dim reuse: reuse this cascade's live atlases + buffers if its
        // dims are unchanged → NO recompile for this cascade, NO black flash.
        const sameDim = !!C.gpu
            && atlasW === C.prevAtlasW && atlasH === C.prevAtlasH
            && C.glossyAtlasW === C.prevGlossyAtlasW && C.glossyAtlasH === C.prevGlossyAtlasH
            && probeTotal === C.prevProbeTotal
            && (!clusteredGi || C.gpu.lightGridCellCount === c0LightCellCount());
        if (sameDim) {
            const prev = C.gpu;
            // Kernel-resident fast path: a within-capacity in-place scene rewrite
            // leaves every binding of the live compute graph valid — the arena
            // owns the storage nodes and traversal uniforms, the dims are
            // unchanged, and the material map textures are identity-equal.
            // Keeping the graph avoids the GPU-process pipeline recompile that
            // stalls the first dispatch (~190 ms measured on churn.html) even
            // while the JS thread shows no work at all.
            if (opts.sceneRewritten && sceneResource
                && mapsCompatible(prev.sceneResource?.maps, built.maps)) {
                if (prev.sceneResource !== sceneResource) {
                    retainSceneResource(sceneResource);
                    releaseSceneResource(prev.sceneResource);
                    prev.sceneResource = sceneResource;
                }
                C.U.gridMin.value.copy(gridMin);
                C.U.gridSize.value.copy(gridSize);
                C.U.lightCount.value = Math.min(MAX_LIGHTS,
                    clusteredGi ? giSelectedCount : giLegacyLightCount) >>> 0;
                C.U.maxDist.value = gridSize.length();
                C.U.cellMin.value = Math.max(1e-4, minCell);
                C.U.relocClamp.value = 0.45 * minCell;
                node.updateGridUniforms(
                    c, gridMin, gridSize, res, atlasW, atlasH,
                    C.glossyAtlasW, C.glossyAtlasH, C.glossyTilesX,
                    normalBias, chebyBias,
                );
                C.probeCursor = 0;
                C.lastSolveAt = 0;
                C.solveDtEma = 0;
                C.glossyPhase = 0;
                C.needsClear = false;             // keep the live atlas history (no black flash)
                C.needsClassify = true;           // refresh per-probe state for the new geometry
                kernelResidentReuses++;
                return false;                     // graph kept — no recompile anywhere
            }
            const reuse = {
                atlas: prev.atlas, roughSpecularAtlas: prev.roughSpecularAtlas, glossySpecularAtlas: prev.glossySpecularAtlas,
                depthAtlas: prev.depthAtlas, stateAtlas: prev.stateAtlas,
                irrBuffer: prev.irrBuffer, roughSpecularBuffer: prev.roughSpecularBuffer,
                glossySpecularBuffer: prev.glossySpecularBuffer, glossyWeightBuffer: prev.glossyWeightBuffer,
                depthBuffer: prev.depthBuffer, stateBuffer: prev.stateBuffer, emitterVisBuffer: prev.emitterVisBuffer,
            };
            C.gpu = buildKernels(built, C, reuse);
            kernelRebuilds++;
            disposeKernelOnly(prev); // release old graph/scratch; shared scene data stays resident
            C.U.gridMin.value.copy(gridMin);
            C.U.gridSize.value.copy(gridSize);
            C.U.lightCount.value = Math.min(MAX_LIGHTS, C.gpu.lightCount) >>> 0;
            C.U.maxDist.value = gridSize.length();
            C.U.cellMin.value = Math.max(1e-4, minCell);
            C.U.relocClamp.value = 0.45 * minCell;
            // Churn-free: update the NODE's placement uniforms WITHOUT bumping _structGen.
            node.updateGridUniforms(
                c, gridMin, gridSize, res, atlasW, atlasH,
                C.glossyAtlasW, C.glossyAtlasH, C.glossyTilesX,
                normalBias, chebyBias,
            );
            C.probeCursor = 0;
            C.lastSolveAt = 0;
            C.solveDtEma = 0;
            C.glossyPhase = 0;
            C.needsClear = false;             // reuse the live atlas history (no black flash)
            C.needsClassify = true;           // refresh per-probe state for the new geometry
            return false;                     // no recompile occurred
        }

        // Resize / first-enable path for this cascade: fresh resources + one recompile.
        disposeCascadeGPU(c);
        C.gpu = buildKernels(built, C);
        kernelRebuilds++;
        C.U.gridMin.value.copy(gridMin);
        C.U.gridSize.value.copy(gridSize);
        C.U.resX.value = res.x >>> 0; C.U.resY.value = res.y >>> 0; C.U.resZ.value = res.z >>> 0;
        C.U.probeTotal.value = probeTotal >>> 0;
        C.U.atlasDim.value.set(atlasW, atlasH);
        C.U.lightCount.value = Math.min(MAX_LIGHTS, C.gpu.lightCount) >>> 0;
        C.U.maxDist.value = gridSize.length();
        C.U.cellMin.value = Math.max(1e-4, minCell);
        C.U.relocClamp.value = 0.45 * minCell;
        node.setAtlases(c, C.gpu.atlas, C.gpu.depthAtlas, C.gpu.stateAtlas, C.gpu.roughSpecularAtlas, C.gpu.glossySpecularAtlas);
        node.setGrid(
            c, gridMin, gridSize, res, atlasW, atlasH,
            C.glossyAtlasW, C.glossyAtlasH, C.glossyTilesX,
            normalBias, chebyBias,
        );
        C.prevAtlasW = atlasW; C.prevAtlasH = atlasH;
        C.prevGlossyAtlasW = C.glossyAtlasW; C.prevGlossyAtlasH = C.glossyAtlasH;
        C.prevProbeTotal = probeTotal;
        C.probeCursor = 0;
        C.lastSolveAt = 0;
        C.solveDtEma = 0;
        C.glossyPhase = 0;
        C.needsClear = true;      // fresh StorageTextures aren't guaranteed zeroed
        C.needsClassify = true;
        return true;              // recompile occurred
    }

    async function flushGiLightGrid() {
        if (!clusteredGi) return;
        const kernel = casc[0]?.gpu?.lightGridKernel;
        if (!kernel) return;
        // Clear before submission so a concurrent light refill cannot be lost.
        giLightGridDirty = false;
        await renderer.computeAsync(kernel);
    }

    // STAGE 0 of the staggered build: (re)compute BOTH cascades' dims off the SHARED soup
    // (C0 = today's path exactly; C1 = fitFineBox when cascades>=2, res ~2× finer), then
    // build C0 only. Sets cascadeCountNode=1 so the fold is the single-grid shader until C1
    // comes online (stage 1). Returns false on a failed/empty build.
    function disposeUninstalledBuild(built) {
        built?.disposeDeformUpdates?.();
        if (built?.mapsArenaGeneration) {
            const generation = built.mapsArenaGeneration;
            if (generation.refs === 0 && generation !== mapsArena?.current) {
                disposeMapsGeneration(generation);
            }
            return;
        }
        if (!built?.maps) return;
        const disposedMaps = new Set();
        for (const texture of Object.values(built.maps)) {
            if (!texture || disposedMaps.has(texture)) continue;
            disposedMaps.add(texture);
            texture.dispose?.();
        }
    }

    function retryFreshBuild(built) {
        disposeUninstalledBuild(built);
        dirty = true;
        buildDirty = true;
        rebuildBackoff = 0;
        return 'retry';
    }

    async function rebuild() {
        // Reuse the cached BVH+texture soup unless geometry actually changed. A
        // divisions/rays change (buildDirty=false) skips the synchronous MeshBVH build.
        let built = cachedBuilt;
        if (!built || buildDirty) {
            const buildSpectralScene = await ensureSceneBuilder();
            if (disposed) return false;
            // Capture after the lazy import: changes that land while the module
            // loads are included in this build. Changes during the build's own
            // async texture extraction are validated below.
            scene.updateMatrixWorld(true);
            const startedGeneration = buildGeneration;
            const startedGeoSig = detectSceneChanges ? geoSignature() : null;
            if (!blasCache && _createBlasCache) blasCache = _createBlasCache();
            if (mapsArena) {
                mapsArena.maxLayers = Number(renderer?.backend?.device?.limits?.maxTextureArrayLayers)
                    || mapsArena.maxLayers;
            }
            built = await buildSpectralScene({
                THREE,
                scene,
                maxTriangles: MAX_TRIANGLES,
                blasCache,
                mapsArena,
            });
            if (disposed) {
                disposeUninstalledBuild(built);
                return false;
            }
            if (buildGeneration !== startedGeneration) return retryFreshBuild(built);
            if (!built || built.error) {
                disposeUninstalledBuild(built);
                return false;   // keep existing field; tick() arms a backoff (A7)
            }

            scene.updateMatrixWorld(true);
            if (detectSceneChanges && geoSignature() !== startedGeoSig) return retryFreshBuild(built);

            // Vertex buffers and transforms may advance while material images
            // are decoded. Catch the fresh build up in place before publishing
            // it; layout drift fails closed to another idle-gated fresh build.
            let deformResult = null;
            let transformResult = null;
            try {
                deformResult = typeof built.updateDeformsAsync === 'function'
                    ? await built.updateDeformsAsync()
                    : built.updateDeforms?.();
                if (disposed) {
                    disposeUninstalledBuild(built);
                    return false;
                }
                if (buildGeneration !== startedGeneration) return retryFreshBuild(built);
                transformResult = built.updateTransforms?.();
            } catch {
                return retryFreshBuild(built);
            }
            if (!deformResult || !transformResult) return retryFreshBuild(built);
            if (transformResult.bounds && built.bounds?.copy) built.bounds.copy(transformResult.bounds);

            if (cachedBuilt && cachedBuilt !== built) cachedBuilt.disposeDeformUpdates?.();
            cachedBuilt = built;
            buildDirty = false;
            // Establish all signature baselines from the exact scene state that
            // was installed. The first idle checks can now detect—not silently
            // baseline—an edit that arrives immediately after this point.
            lastGeoSig = detectSceneChanges ? geoSignature() : null;
            lastDeformSig = detectSceneChanges ? deformSignature() : null;
            lastXformSig = detectSceneChanges ? xformSignature() : null;
            lastLightSig = detectSceneChanges ? lightSignature() : null;
            geoStable = -1;
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
        setGlossyLayout(C0);
        C0.minCell = Math.max(1e-4, Math.min(C0.gridSize.x / Math.max(1, C0.res.x - 1), C0.gridSize.y / Math.max(1, C0.res.y - 1), C0.gridSize.z / Math.max(1, C0.res.z - 1)));
        curMinCell = C0.minCell;                                 // setNormalBias() lives off C0's cell
        quantStep = Math.max(1e-4, C0.minCell * 0.25);          // A1: geo-signature translation deadband
        lightQuant = Math.max(1e-3, C0.gridSize.length() * 0.003); // B4: light-signature position deadband

        // Clustered mode: budget the GI lane into the fixed arena BEFORE any kernel
        // binds it. Records come from the build snapshot itself (no re-collect), so
        // the arena is byte-consistent with what the legacy path would have bound;
        // the light lane's refreshLights keeps it current from there.
        const sceneGenerationChanged = sceneResource?.built !== built;
        if (sceneGenerationChanged || !liveLightRecords) {
            try {
                liveLightRecords = _collectLights ? _collectLights(THREE, scene) : recordsFromBuilt(built);
            } catch {
                liveLightRecords = recordsFromBuilt(built);
            }
        }
        const lightDataChanged = clusteredGi
            ? fillGiLightArena(liveLightRecords).recordDataChanged
            : fillLegacyLightArena(liveLightRecords).recordDataChanged;
        const { sceneBuffersRebound, sceneBuffersRewritten } = prepareSharedResources(built, lightDataChanged);
        lastBuildSceneRewritten = sceneBuffersRewritten && !sceneBuffersRebound;

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
                // Bound the fine cascade's SIZE relative to the coarse grid. On a big boxy
                // scene the fine box hits its 60%-per-axis cap at 2× density and C1 blows up
                // to many times C0 (city: 7k fine vs 1k coarse) — starving the per-tick ray
                // budget and freezing C1's convergence. Shrink uniformly (cells stay ~cubic).
                const c1Cap = Math.max(64, 2 * C0.probeTotal);
                for (let g = 0; g < 12 && C1.res.x * C1.res.y * C1.res.z > c1Cap; g++) {
                    C1.res.set(Math.max(2, Math.floor(C1.res.x * 0.85)), Math.max(2, Math.floor(C1.res.y * 0.85)), Math.max(2, Math.floor(C1.res.z * 0.85)));
                }
                C1.probeTotal = C1.res.x * C1.res.y * C1.res.z;
                C1.atlasW = C1.res.x * TILE;
                C1.atlasH = C1.res.y * C1.res.z * TILE;
                setGlossyLayout(C1);
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
        const recompiled = buildOneCascade(built, 0, { sceneRewritten: lastBuildSceneRewritten });
        if (clusteredGi) {
            giLightGridDirty = true;
            await flushGiLightGrid();
            if (disposed) return false;
        }
        if (buildCascadeCount < 2) {
            // No fine cascade this build → single-grid shader + dispose C1, finish now.
            node.setCascadeCount(1);
            disposeCascadeGPU(1);
            buildStage = 2;
            fieldEverReady = true;
        } else if (c1SameDim) {
            // C1 stays live at its current dims → keep it bound (no recompile), rebuild its
            // BVH-bound kernels next idle tick without dropping the blend.
            buildStage = 1;
        } else {
            // C1 will be (re)allocated at new dims → hide it (single-grid) until stage 1.
            node.setCascadeCount(1);
            buildStage = 1;   // C1 built next idle tick (staggered — never 2× the build/frame)
        }
        dirty = false;
        // Fire the one-shot recompile the frame C0's data first exists (resize/first-enable
        // path OR a cascade-count change to 1). The same-dim path with no structGen change
        // needs no recompile.
        if ((recompiled || sceneBuffersRebound || node._structGen !== genBefore) && typeof onRebuilt === 'function') { try { onRebuilt(); } catch (e) { /* non-fatal */ } }
        return true;
    }

    // STAGE 1: build C1 only (one idle tick after C0), then flip cascadeCountNode to 2 so
    // the fragment starts blending the fine cascade. Runs behind the SAME idle gate as C0.
    async function advanceBuildStageC1() {
        const built = cachedBuilt;
        if (!built) { buildStage = 2; fieldEverReady = true; return; }
        const genBefore = node._structGen;
        const recompiled = buildOneCascade(built, 1, { sceneRewritten: lastBuildSceneRewritten });
        node.setCascadeCount(2);  // C1 online → fragment blends fine cascade
        buildStage = 2;
        fieldEverReady = true;
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
        const wasMoving = lastMoving;
        const wasPlaying = lastPlaying;
        lastIdleMs = idleMs;
        lastMoving = moving;
        lastPlaying = playing;
        if (!moving && wasMoving && (!continuous || wasPlaying)) {
            // Scrub/playback release can expose a geometry catch-up/refit on
            // the same first idle frames. Never resume a strict-idle field at
            // its stale pre-interaction maximum ray budget and make those
            // bounded CPU slices compete with a full GPU solve. Continuous-mode
            // GUI/camera interaction never paused the solve, though: clamping
            // that live budget on release creates a visible full-pass -> sparse-
            // batch cadence cliff, so leave it untouched unless playback itself
            // just stopped and the rest-only scene scans may need to catch up.
            tickBudgetRays = probeBudgetAfterInteraction(tickBudgetRays);
            tickDtEma = 0;
            lastTickAt = 0;
            budgetCooldown = Math.max(budgetCooldown, 30);
        }
        // Default: fully idle-gated (moving → return). Continuous mode: keep the bounded GPU
        // SOLVE running while moving, but STILL hold every synchronous/compiling step — the
        // ~200ms MeshBVH rebuild, the staggered cascade build, and the per-tick scene-signature
        // scans — for rest. Those are the only real hitch sources; the capped round-robin solve
        // is the same GPU cost that already runs smooth at rest, so it's safe every frame.
        if (moving && !continuous) return;
        const restOnly = !moving;   // gates the heavy paths; the solve runs in both modes
        lastRestOnly = restOnly;

        // (A7) Back off after a failed/empty rebuild instead of re-entering the
        // synchronous build every tick.
        if (rebuildBackoff > 0) rebuildBackoff--;

        // Topology is the RAREST change class, so a pending rest-gated rebuild
        // gets the LOWEST scheduling priority — never a veto over the lanes
        // beneath it. While the host keeps playing, the tick falls through:
        // transform/deform packets and the solve keep working against the
        // current build, and the rebuild lands at the next rest window. The
        // FIRST bring-up still builds immediately even mid-motion: a host
        // playing from frame 0 never gets a rest window, and there is no
        // converged GI to disturb yet, so the one-time boot hitch is the
        // cheaper failure mode by construction.
        const buildHeld = !restOnly && fieldEverReady && !!casc[0].gpu;
        if ((dirty || !casc[0].gpu) && !buildHeld) {
            // A CPU soup/kernel build is not evidence that the previous GPU
            // solve was too expensive. Keep it out of the cadence interval.
            resetFramePacing();
            if (rebuildBackoff > 0) return;
            inFlight = true; let ok = false;
            try {
                ok = await rebuild();
            } catch (e) {
                if (disposed) return;
                console.warn('SPEEDBALL GI rebuild failed:', e);
                dirty = false;
                rebuildBackoff = REBUILD_BACKOFF_TICKS;
                return;
            } finally {
                inFlight = false;
            }
            if (disposed) return;
            if (ok === 'retry') return; // newer scene state stays armed; retry at the next idle tick
            if (!ok) { dirty = false; rebuildBackoff = REBUILD_BACKOFF_TICKS; return; }
            return;   // stage 0 done this tick; the solve waits for the next tick
        }
        if (!casc[0].gpu) return;

        // (A2/#2) Staggered build: advance exactly ONE build stage per idle tick so no
        // single frame does 2× the build. C1 comes online one idle tick after C0. Held for rest.
        if (buildStage < 2 && !buildHeld) {
            resetFramePacing();
            inFlight = true;
            const genBefore = node._structGen;
            try {
                await advanceBuildStageC1();
            } catch (error) {
                if (!disposed) {
                    console.warn('SPEEDBALL GI fine cascade build failed; continuing with the coarse cascade:', error);
                    disposeCascadeGPU(1);
                    node.setCascadeCount(1);
                    buildCascadeCount = 1;
                    buildStage = 2;
                    fieldEverReady = true;
                    if (node._structGen !== genBefore && typeof onRebuilt === 'function') {
                        try { onRebuilt(); } catch (e) { /* non-fatal */ }
                    }
                }
            } finally {
                inFlight = false;
            }
            if (disposed) return;
            return;
        }

        // Measure this accepted solve tick before deriving any temporal coefficient or
        // mutating the live ray budget. Calls dropped by inFlight never reach here, so
        // this is the cadence the history buffers actually see (not raw display FPS).
        // Long idle/motion gaps are discarded: resuming after a pause must not turn one
        // Monte-Carlo sample into a multi-second catch-up jump.
        const tNow = _nowMs();
        if (lastTickAt > 0) {
            const dt = tNow - lastTickAt;
            if (dt > 0 && dt < TICK_OVERLOAD_MS) {
                cadenceOverloadStreak = 0;
                hysteresisTickDtEma = hysteresisTickDtEma > 0 ? hysteresisTickDtEma * 0.8 + dt * 0.2 : dt;
                tickDtEma = tickDtEma > 0 ? tickDtEma * 0.8 + dt * 0.2 : dt;
                if (budgetCooldown > 0) budgetCooldown--;
                if (tickDtEma > 18.5 && tickBudgetRays > RAYS_PER_TICK_MIN) {
                    tickBudgetRays = probeBudgetAfterCadenceMiss(tickBudgetRays);
                    tickDtEma = 0;        // re-measure only the budget controller at the new cap
                    budgetCooldown = 120; // hold ~2 s before growing again — a render-bound
                                          // scene that misses 60 fps at ANY budget otherwise
                                          // saw-tooths between floor and max
                } else if (budgetCooldown === 0 && tickDtEma < 17.2 && tickBudgetRays < rayBudgetCeiling) {
                    tickBudgetRays = Math.min(rayBudgetCeiling, tickBudgetRays + 1024);
                }
            } else if (dt >= TICK_OVERLOAD_MS && dt < TICK_PAUSE_MS) {
                // computeAsync submits without waiting for GPU completion, so this is
                // presentation cadence rather than a direct solve timer. One long gap
                // may be unrelated; repeated accepted gaps still mean the browser is
                // not making progress and must make the bounded GI workload back off.
                tickDtEma = 0;
                cadenceOverloadStreak = Math.min(
                    TICK_OVERLOAD_STRIKES,
                    cadenceOverloadStreak + 1,
                );
                if (cadenceOverloadStreak >= TICK_OVERLOAD_STRIKES && tickBudgetRays > RAYS_PER_TICK_MIN) {
                    tickBudgetRays = probeBudgetAfterCadenceMiss(tickBudgetRays);
                    budgetCooldown = 120;
                }
            } else {
                tickDtEma = 0;
                hysteresisTickDtEma = 0;
                cadenceOverloadStreak = 0;
            }
        }
        lastTickAt = tNow;

        // reactivity: detect live edits (throttled). Transform change → in-place
        // instance/TLAS rewrite (near-instant, no rebuild). Light change → cheap
        // in-place buffer refresh. DEFORM change (same-topology vertex motion,
        // e.g. streamed skinned-mesh vertex buffers) → in-place soup gather +
        // bounds refit (no MeshBVH — never a hitch). STRUCTURE change
        // (topology/instance set) → debounced full rebuild. The heavy lanes
        // walk the scene graph (CPU), so in continuous mode they run ONLY at
        // rest — never per orbit frame.
        checkCounter++;
        // LIGHTS are the exception: refresh-class, not rebuild-class.
        // refreshLights() is an in-place buffer refill (no BVH, no compile)
        // and lightSignature() only reads analytic lights plus opted-in emitters with
        // deadbands, so this lane rides THROUGH motion. Rest-gating it made
        // every host light edit read as "GI stopped solving" while the
        // interactive raster light updated live (maxjs 2026-07-24).
        if (detectSceneChanges && checkCounter % LIGHT_CHECK_INTERVAL === 0) {
            const ls = lightSignature();
            if (lastLightSig !== null && ls !== lastLightSig) refreshLights();
            lastLightSig = ls;
        }
        if (clusteredGi && giLightGridDirty) {
            const dirtyBefore = dirty;
            inFlight = true;
            try { await flushGiLightGrid(); } finally { inFlight = false; }
            // Abort only on state that changed UNDER the await — a rebuild that
            // was already pending (held for rest) must not stall this tick.
            if (disposed || (dirty && !dirtyBefore) || giLightGridDirty) return;
        }
        // Explicit host changes bypass the scene-wide/rest-only signature
        // scans. One moving object rewrites only its stable instance rows and
        // exact TLAS ancestors while the game or DCC viewport is still moving.
        if (pendingAllTransforms || pendingTransformTargets.size > 0) {
            const targets = pendingAllTransforms ? null : Array.from(pendingTransformTargets);
            pendingAllTransforms = false;
            pendingTransformTargets.clear();
            if (!refreshTransforms(targets)) return;
            // The next rest-only signature pass establishes a fresh baseline;
            // it must not replay this already-committed packet as a full refit.
            lastXformSig = null;
        }
        // Material-value packets ride THROUGH motion like transforms; no
        // signature baseline exists to reset (no fallback scan reads values).
        if (pendingAllMaterialValues || pendingMaterialValueTargets.size > 0) {
            const targets = pendingAllMaterialValues ? null : Array.from(pendingMaterialValueTargets);
            pendingAllMaterialValues = false;
            pendingMaterialValueTargets.clear();
            if (!refreshMaterialValues(targets)) return;
        }
        if (pendingDeformRefresh) {
            pendingDeformRefresh = false;
            const dirtyBefore = dirty;
            inFlight = true;
            try { await refreshDeforms(); } finally { inFlight = false; }
            // Same rule as above: a pre-existing held rebuild does not abort.
            if (disposed || (dirty && !dirtyBefore)) return;
            lastDeformSig = null;
        }
        if (restOnly && detectSceneChanges) {
            if (checkCounter % XFORM_CHECK_INTERVAL === 0) {
                const xs = xformSignature();
                if (lastXformSig !== null && xs !== lastXformSig) refreshTransforms();
                lastXformSig = xs;
            }
            if (checkCounter % DEFORM_CHECK_INTERVAL === 0) {
                const ds = deformSignature();
                if (lastDeformSig !== null && ds !== lastDeformSig) {
                    inFlight = true;
                    try { await refreshDeforms(); } finally { inFlight = false; }
                    if (disposed || dirty) return;
                    // updateDeformsAsync coalesces versions that arrive while
                    // it yields, so baseline the state it actually committed.
                    lastDeformSig = deformSignature();
                } else {
                    lastDeformSig = ds;
                }
            }
            if (checkCounter % GEO_CHECK_INTERVAL === 0) {
                const gs = geoSignature();
                // (A1) Debounce: rebuild only after the structure has been STABLE for
                // GEO_SETTLE_INTERVALS consecutive checks, so a continuous edit never
                // thrashes. geoStable: -1 = no pending change; >=0 = stable-checks counted.
                if (lastGeoSig === null) lastGeoSig = gs;
                else if (gs !== lastGeoSig) { lastGeoSig = gs; geoStable = 0; }
                else if (geoStable >= 0) {
                    geoStable++;
                    if (geoStable >= GEO_SETTLE_INTERVALS) { geoStable = -1; requestRebuild(); }
                }
            }
        }
        // CONSTANT temporal policy — no reactive/low-hysteresis burst. A global
        // authority drop after an edit read as "flicker for a second, then settle"
        // (a visible fade-out/in on every slider touch). Edits re-converge through
        // the per-texel change detector in the blend, whose retention is bounded,
        // so a light/sky/geometry change transitions smoothly at the steady rate.
        U.hysteresis.value = baseHysteresis;

        // (#1) Cascade scheduling. When the per-tick ray budget covers the WHOLE union
        // (the common small/medium-grid case) solve BOTH cascades every tick — every
        // probe refreshes every tick, so the field re-converges in ~1 s at unchanged
        // hysteresis. Only when the union exceeds the budget fall back to the old
        // round-robin (one cascade per tick) so the per-tick GPU cost stays bounded.
        const haveC1 = cascades > 1 && !!casc[1].gpu;
        const fullPassPerTick = tickCap() >= totalUnionProbes();
        let solveList;
        if (!haveC1) solveList = [0];
        else if (fullPassPerTick) solveList = [0, 1];
        else { solveList = [solveTurn % 2]; solveTurn = (solveTurn + 1) >>> 0; }
        lastSolveList = solveList.join(',');
        lastUpdatedCount = tickCap();

        inFlight = true;
        try {
            for (const ci of solveList) {
                const C = casc[ci];
                const gpu = C.gpu;
                if (!gpu) continue;

                const updated = Math.min(tickCap(), gpu.probeCapBuilt ?? Infinity, C.probeTotal);
                // Normalize per CASCADE from its accepted service cadence. In partial
                // mode C0/C1 alternate and may have very different sizes, so the old
                // union-wide ceil(total/cap) coefficient could make the fine grid boil
                // while the coarse grid lagged. The fractional pass ratio is the exact
                // long-term rate for circular modulo batches and stays continuous as the
                // auto-throttled cap changes.
                let nextSolveDtEma = C.solveDtEma;
                if (C.lastSolveAt > 0) {
                    const solveDt = tNow - C.lastSolveAt;
                    if (solveDt > 0 && solveDt < 200) {
                        nextSolveDtEma = nextSolveDtEma > 0 ? nextSolveDtEma * 0.8 + solveDt * 0.2 : solveDt;
                    } else {
                        nextSolveDtEma = 0;
                    }
                }
                const fallbackServiceTicks = (!haveC1 || fullPassPerTick) ? 1 : 2;
                const serviceDt = nextSolveDtEma > 0
                    ? nextSolveDtEma
                    : (hysteresisTickDtEma > 0 ? hysteresisTickDtEma : HYSTERESIS_DT_REF_MS) * fallbackServiceTicks;
                const revisitPasses = probeUpdateIntervalTicks(C.probeTotal, updated);
                C.U.hysteresisExponent.value = hysteresisExponentForInterval(
                    serviceDt * revisitPasses,
                    hysteresisNormalize,
                );
                C.U.probeOffset.value = C.probeCursor >>> 0;
                C.U.updatedCount.value = updated >>> 0;
                // (B1) The product contract is binary and independent of idle time,
                // probe count, batch size, or resize cadence:
                //   Gated      — hold the current ray/emitter sampling basis forever.
                //   Monte Carlo — advance it on every accepted C0 solve.
                // Both cascades read the same shared uniforms, so trace and blend
                // remain byte-identical for whichever cascade runs this tick.
                const rotateRaySet = ci === 0 && jitterMode === 'montecarlo';
                // Keep every stochastic input in the selected regime. Gated holds
                // the emitter target indefinitely; Monte Carlo advances it on every
                // C0 solve tick along with the ray basis.
                if (rotateRaySet && !debugFreezeRayJitter) {
                    emitterVisSeedCounter = (emitterVisSeedCounter + 1) >>> 0;
                    U.emitterVisSeed.value = (emitterVisSeedCounter * 0.61803398875) % 1;
                }
                // Monte Carlo changes sampling epoch every C0 solve by definition;
                // this block is unreachable in Gated mode.
                if (rotateRaySet) {
                    if (!debugFreezeRayJitter) {
                        frameCounter = (frameCounter + 1) >>> 0;
                        U.frameJitter.value = debugFrameJitterOverride ?? ((frameCounter * 0.61803398875) % 1);
                    } else if (debugFrameJitterOverride !== null) {
                        U.frameJitter.value = debugFrameJitterOverride;
                    }
                }
                C.probeCursor = C.probeTotal > 0 ? (C.probeCursor + updated) % C.probeTotal : 0;

                // (A5) One-time-per-rebuild prep for cascade C ONLY. clear (full rebuild only)
                // + the cheap uploadState (ALWAYS — sole writer of stateAtlas; skipping it
                // leaves the relocation sample reading uninitialised texels → NaN). classify
                // (the 32-ray BVH walk) runs ONLY in Solid-scene mode. These WRITE buffers the
                // trace then READS, so let them finish before tracing.
                if (C.needsClear || C.needsClassify) {
                    const prep = [];
                    if (C.needsClear) {
                        prep.push(gpu.clearAtlasKernel);
                        if (gpu.clearGlossyAtlasKernel) prep.push(gpu.clearGlossyAtlasKernel);
                        prep.push(gpu.clearEmitterVisKernel);
                        C.needsClear = false;
                    }
                    if (C.needsClassify) {
                        if (U.classifyStrength.value > 0) prep.push(gpu.classifyKernel);
                        prep.push(gpu.uploadStateKernel);
                        C.needsClassify = false;
                    }
                    // One ordered compute pass: classify writes stateBuffer and
                    // uploadState consumes it later in the same pass. Separate
                    // computeAsync calls only created extra encoders/submissions;
                    // they were never GPU-parallel.
                    if (prep.length > 0) await renderer.computeAsync(prep);
                    if (disposed) return;
                }
                // Match the dispatch envelope to the LIVE auto-throttled batch.
                // Count sizes each live dispatch without recompiling. Barrier
                // kernels cannot use Three's generated early-return guard, so their
                // exact 64-aligned counts plus activeProbe protect every access.
                gpu.traceKernel.count = updated * raysPerProbe;
                gpu.emitterVisKernel.count = U.emitterCount.value > 0
                    ? updated * GI_EMITTER_INJECT_CAP
                    : 0;
                gpu.blendKernel.count = updated * PROBE_WORKGROUP_SIZE;
                if (gpu.glossyKernel) {
                    gpu.glossyKernel.count = updated * gpu.glossyGroupsPerProbe * PROBE_WORKGROUP_SIZE;
                }
                gpu.uploadKernel.count = updated * PROBE_WORKGROUP_SIZE;

                // (A6/#1) One ordered compute pass for trace→emitter visibility→blend→glossy→upload.
                // This keeps all storage dependencies on the same queue/pass while
                // removing three command encoders and submissions per solve.
                if (gpu.glossyKernel) {
                    C.U.glossyPhase.value = C.glossyPhase >>> 0;
                    C.glossyPhase = (C.glossyPhase + 1) % glossyUpdateInterval;
                }
                await renderer.computeAsync(gpu.solveKernels);
                if (disposed) return;
                C.lastSolveAt = tNow;
                C.solveDtEma = nextSolveDtEma;
            }
        } catch (e) {
            if (!disposed) {
                console.warn('SPEEDBALL GI probe tick failed:', e);
                dirty = true;
            }
        } finally {
            inFlight = false;
        }
    }

    function setEnabled(on) { node.setEnabled(on === true); if (on && (!casc[0].gpu || !node._ready)) requestRebuild(); }
    // freshBuild=true (default) invalidates the cached BVH+texture soup so rebuild()
    // rebuilds it (geometry/light-count/volume change, or first build). Grid-only callers
    // (setDivisions/setRays) pass false → the cached soup is reused, no MeshBVH stall.
    function requestRebuild(freshBuild = true) {
        dirty = true;
        rebuildBackoff = 0;
        if (freshBuild) {
            cachedBuilt?.cancelDeformUpdates?.();
            buildDirty = true;
            buildGeneration++;
        }
    }

    // Event-driven scene updates for games, realtime DCC bridges, and other
    // hosts that already know what changed. Legacy signatures remain as a
    // compatibility fallback; explicit transform/deform notifications are
    // consumed during continuous motion instead of waiting for an idle scan.
    function markTransformsDirty(targets = null) {
        if (targets == null) {
            pendingAllTransforms = true;
            pendingTransformTargets.clear();
            return;
        }
        if (pendingAllTransforms) return;
        const list = Array.isArray(targets) || targets instanceof Set ? targets : [targets];
        for (const target of list) if (target != null) pendingTransformTargets.add(target);
    }
    function markDeformsDirty() {
        pendingDeformRefresh = true;
    }
    // Value-only material edits (emissive, color, roughness, …). targets may be
    // materials or meshes (expanded to their materials); null refreshes every
    // record. Structural material changes self-escalate inside the refresh.
    function markMaterialValuesDirty(targets = null) {
        if (targets == null) {
            pendingAllMaterialValues = true;
            pendingMaterialValueTargets.clear();
            return;
        }
        if (pendingAllMaterialValues) return;
        const list = Array.isArray(targets) || targets instanceof Set ? targets : [targets];
        for (const target of list) if (target != null) pendingMaterialValueTargets.add(target);
    }
    function markTopologyDirty() {
        requestRebuild(true);
    }
    function notifySceneChange(change) {
        if (Array.isArray(change)) {
            for (const entry of change) notifySceneChange(entry);
            return true;
        }
        const type = typeof change === 'string' ? change : change?.type;
        switch (String(type || '').toLowerCase()) {
            case 'transform':
            case 'xform':
                markTransformsDirty(change?.objects ?? change?.object ?? change?.target ?? null);
                return true;
            case 'deform':
            case 'vertices':
                markDeformsDirty();
                return true;
            case 'light':
            case 'lighting':
                forceLightingRefresh();
                return true;
            // Value lane, not the rebuild lane: the refresh itself detects
            // structural material changes (reassignment, map binding, dedup
            // split) and escalates to the full rebuild, so hosts just say
            // "material changed" and get the cheapest correct path.
            case 'material':
            case 'materials':
                markMaterialValuesDirty(change?.materials ?? change?.material ?? change?.objects ?? change?.target ?? null);
                return true;
            case 'add':
            case 'added':
            case 'remove':
            case 'removed':
            case 'topology':
            case 'geometry':
            case 'structure':
                markTopologyDirty();
                return true;
            default:
                return false;
        }
    }
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
    // BVH rebuild. Re-convergence rides the bounded per-texel change detector.
    const structureIds = new WeakMap();
    let nextStructureId = 1;
    function structureId(value) {
        if (!value || (typeof value !== 'object' && typeof value !== 'function')) return 0;
        let id = structureIds.get(value);
        if (id == null) {
            id = nextStructureId++;
            structureIds.set(value, id);
        }
        return id;
    }
    function lightSignature() {
        // Numeric rolling hash — NO per-light string build. This lane deliberately
        // runs THROUGH motion every LIGHT_CHECK_INTERVAL ticks, so at hundreds of
        // lights (clustered mode) the old concatenated-string signature became a
        // real per-check CPU + GC tax. Two independent 32-bit accumulators make a
        // missed edit a ~2^-64 event; the light count stays explicit in the return
        // value so count↔hash aliasing is impossible. Order-sensitive like the old
        // string (two lights swapping records must re-fill the buffer).
        // (B4) Scene-relative deadbands so sub-perceptual delta-sync jitter does NOT
        // trigger a refresh every check; a genuine edit still changes the signature.
        let n = 0, h1 = 0x9e3779b9 | 0, h2 = 0x85ebca6b | 0;
        const q = lightQuant > 1e-6 ? lightQuant : 1;
        scene.traverseVisible((o) => {
            if (o.isLight && !o.isAmbientLight && !o.isHemisphereLight) {
                // Excluded lights never reach collectLights (objectIsRenderable) → their
                // edits must not churn the lane (same rule 2abc2cd applied to the
                // geo/deform/xform signatures for excluded meshes).
                if (o.userData?.maxjsVisible === false) return;
                o.updateWorldMatrix?.(true, false);
                o.getWorldPosition(_sigVec);
                const c = o.color;
                let v = Math.round(_sigVec.x / q);
                h1 = (Math.imul(h1, 31) + v) | 0; h2 = Math.imul(h2 ^ v, 0x45d9f3b) | 0;
                v = Math.round(_sigVec.y / q);
                h1 = (Math.imul(h1, 31) + v) | 0; h2 = Math.imul(h2 ^ v, 0x45d9f3b) | 0;
                v = Math.round(_sigVec.z / q);
                h1 = (Math.imul(h1, 31) + v) | 0; h2 = Math.imul(h2 ^ v, 0x45d9f3b) | 0;
                v = c ? Math.round((c.r * 7 + c.g * 11 + c.b * 13) * 16) : 0;
                h1 = (Math.imul(h1, 31) + v) | 0; h2 = Math.imul(h2 ^ v, 0x45d9f3b) | 0;
                v = Math.round((o.intensity || 0) * 4);
                h1 = (Math.imul(h1, 31) + v) | 0; h2 = Math.imul(h2 ^ v, 0x45d9f3b) | 0;
                v = Math.round((o.angle || 0) * 50);
                h1 = (Math.imul(h1, 31) + v) | 0; h2 = Math.imul(h2 ^ v, 0x45d9f3b) | 0;
                // range/decay were missing from the old string signature — a distance
                // edit silently never reached the GI records until something else moved.
                v = Math.round((o.distance || 0) * 4);
                h1 = (Math.imul(h1, 31) + v) | 0; h2 = Math.imul(h2 ^ v, 0x45d9f3b) | 0;
                v = Math.round((o.decay || 0) * 8);
                h1 = (Math.imul(h1, 31) + v) | 0; h2 = Math.imul(h2 ^ v, 0x45d9f3b) | 0;
                n++;
                return;
            }
            if (o.userData?.giEmitter !== true || (!o.isMesh && !o.isInstancedMesh)
                || o.isSkinnedMesh || o.name === '__maxjs_sky__' || o.userData?.maxjsVisible === false) return;
            const position = o.geometry?.attributes?.position;
            if (!position || position.count < 3 || position.itemSize < 3 || !position.array) return;
            const materials = Array.isArray(o.material) ? o.material : [o.material];
            if (!materials.some((m) => m && m.visible !== false && (!Number.isFinite(m.opacity) || m.opacity > 1e-4))) return;
            const material = materials[0];
            if (!material || material.visible === false || (Number.isFinite(material.opacity) && material.opacity <= 1e-4)
                || !_emissiveScaled) return;
            const em = _emissiveScaled(material, _sigEmissive);
            if (Math.max(em[0], em[1], em[2]) <= 0) return;
            if (!o.geometry.boundingSphere) {
                try { o.geometry.computeBoundingSphere(); } catch { return; }
            }
            const sphere = o.geometry.boundingSphere;
            if (!sphere || !Number.isFinite(sphere.radius)) return;
            o.updateWorldMatrix?.(true, false);
            _sigVec.copy(sphere.center).applyMatrix4(o.matrixWorld);
            o.getWorldScale(_sigScale);
            const radius = sphere.radius * Math.max(Math.abs(_sigScale.x), Math.abs(_sigScale.y), Math.abs(_sigScale.z));
            if (!Number.isFinite(radius)) return;
            const emitterScale = Number.isFinite(o.userData?.giEmitterScale) ? o.userData.giEmitterScale : 1;
            let v = Math.round(_sigVec.x / q);
            h1 = (Math.imul(h1, 31) + v) | 0; h2 = Math.imul(h2 ^ v, 0x45d9f3b) | 0;
            v = Math.round(_sigVec.y / q);
            h1 = (Math.imul(h1, 31) + v) | 0; h2 = Math.imul(h2 ^ v, 0x45d9f3b) | 0;
            v = Math.round(_sigVec.z / q);
            h1 = (Math.imul(h1, 31) + v) | 0; h2 = Math.imul(h2 ^ v, 0x45d9f3b) | 0;
            v = Math.round(em[0] * 16);
            h1 = (Math.imul(h1, 31) + v) | 0; h2 = Math.imul(h2 ^ v, 0x45d9f3b) | 0;
            v = Math.round(em[1] * 16);
            h1 = (Math.imul(h1, 31) + v) | 0; h2 = Math.imul(h2 ^ v, 0x45d9f3b) | 0;
            v = Math.round(em[2] * 16);
            h1 = (Math.imul(h1, 31) + v) | 0; h2 = Math.imul(h2 ^ v, 0x45d9f3b) | 0;
            v = Math.round(emitterScale * 16);
            h1 = (Math.imul(h1, 31) + v) | 0; h2 = Math.imul(h2 ^ v, 0x45d9f3b) | 0;
            v = Math.round(radius / q);
            h1 = (Math.imul(h1, 31) + v) | 0; h2 = Math.imul(h2 ^ v, 0x45d9f3b) | 0;
            n++;
        });
        return n + ':' + h1 + ':' + h2;
    }
    // STRUCTURE signature: topology / connectivity / instance-set identity —
    // anything that invalidates the pooled BLAS soup and needs a full rebuild.
    // Transforms are deliberately EXCLUDED: moves ride the TLAS fast path.
    // position.version is ALSO excluded: same-count vertex motion is a
    // DEFORM (deformSignature → in-place gather+refit), not a rebuild — this
    // split is what keeps a streamed vertex animation from arming the ~200 ms
    // MeshBVH rebuild the moment the stream pauses. SkinnedMesh is skipped to
    // mirror the BVH build (spectral_scene isTraceableMesh): GPU skinning
    // never lands in the soup, so its churn must not schedule rebuilds.
    // CONSUMER CONTRACT: hosts streaming geometry must not bump identity or
    // version on data that did not change. A byte-identical index re-send
    // that bumps index.version is indistinguishable from a connectivity edit
    // here and arms the debounced FULL rebuild (maxjs 2026-07-23: settle
    // packets after every timeline scrub did exactly that — ~550 ms freeze
    // per release). Change-detect on the host side before rewriting
    // attributes; this lane deliberately stays paranoid.
    function geoSignature() {
        let meshes = 0, prims = 0, hash = 0;
        scene.traverseVisible((o) => {
            if (!o.isMesh && !o.isInstancedMesh) return;
            if (o.userData?.maxjsVisible === false) return;  // excluded from the soup → must not arm a rebuild
            if (o.isSkinnedMesh) return;
            const p = o.geometry?.attributes?.position; if (!p) return;
            const idx = o.geometry.index;
            meshes++;
            prims += idx ? idx.count : p.count;
            // InstancedMesh allocation capacity is structural; its live count
            // is transform-state and rides the reserved-slot TLAS fast path.
            const instanceCapacity = o.isInstancedMesh
                ? Math.max(0, (o.instanceMatrix?.count ?? o.count) | 0)
                : 1;
            hash = ((hash * 31) + structureId(o)) | 0;
            hash = ((hash * 31) + structureId(o.geometry)) | 0;
            // Replacing a same-sized position attribute is still a DEFORM.
            // updateDeforms tracks both attribute identity and version, so the
            // structure lane only needs the vertex count here.
            hash = ((hash * 31) + p.count) | 0;
            hash = ((hash * 31) + structureId(idx) + (idx ? ((idx.version | 0) + idx.count) : 0)) | 0;
            hash = ((hash * 31) + instanceCapacity) | 0;
            const groups = o.geometry.groups || [];
            hash = ((hash * 31) + groups.length) | 0;
            for (const group of groups) {
                hash = ((hash * 31) + (group.start | 0)) | 0;
                hash = ((hash * 31) + (group.count | 0)) | 0;
                hash = ((hash * 31) + (group.materialIndex | 0)) | 0;
            }
            const mats = Array.isArray(o.material) ? o.material : [o.material];
            hash = ((hash * 31) + mats.length) | 0;
            for (const material of mats) hash = ((hash * 31) + structureId(material)) | 0;
        });
        return `${meshes}:${prims}:${hash}`;
    }
    // DEFORM signature: same-topology vertex motion (streamed vertex buffers,
    // morph bakes, CPU skinning). A change routes to refreshDeforms() — the
    // in-place soup gather + bounds refit — NEVER to the full rebuild.
    function deformSignature() {
        let h = 0;
        scene.traverseVisible((o) => {
            if (!o.isMesh && !o.isInstancedMesh) return;
            if (o.userData?.maxjsVisible === false) return;  // excluded from the soup → nothing to re-gather
            if (o.isSkinnedMesh) return;
            const g = o.geometry, p = g?.attributes?.position; if (!p) return;
            const n = g.attributes.normal;
            h = ((h * 31) + structureId(p)) | 0;
            h = ((h * 31) + structureId(n)) | 0;
            h = ((h * 31) + ((p.version | 0) * 7) + (n ? ((n.version | 0) * 13) : 5)) | 0;
            if (p.isInterleavedBufferAttribute) h = ((h * 31) + ((p.data?.version | 0) * 17)) | 0;
            if (n?.isInterleavedBufferAttribute) h = ((h * 31) + ((n.data?.version | 0) * 19)) | 0;
        });
        return h;
    }
    // TRANSFORM signature: quantized affine matrixWorld (rotation/scale at 1e-3,
    // translation deadbanded to ~¼ cell so sub-perceptual sync jitter does not
    // churn the TLAS) + InstancedMesh matrix version.
    function xformSignature() {
        const q = quantStep > 1e-6 ? quantStep : 1;
        let h = 0;
        scene.traverseVisible((o) => {
            if (!o.isMesh && !o.isInstancedMesh) return;
            if (o.userData?.maxjsVisible === false) return; // same reason: not in the soup, no TLAS entry to churn
            if (o.isSkinnedMesh) return; // not in the soup (see geoSignature) → must not churn the TLAS
            if (!o.geometry?.attributes?.position) return;
            const e = o.matrixWorld?.elements;
            if (e) {
                for (let k = 0; k < 12; k++) h = ((h * 33) + Math.round(e[k] * 1000)) | 0;
                h = ((h * 33) + Math.round(e[12] / q)) | 0;
                h = ((h * 33) + Math.round(e[13] / q)) | 0;
                h = ((h * 33) + Math.round(e[14] / q)) | 0;
            }
            if (o.isInstancedMesh && o.instanceMatrix) {
                h = ((h * 33) + (o.instanceMatrix.version | 0)) | 0;
                h = ((h * 33) + Math.max(0, o.count | 0)) | 0;
            }
        });
        return h;
    }
    // Moving-object fast path: rewrite the instance table + TLAS in place from
    // the live matrixWorlds (spectral_scene updateTransforms) and re-upload the
    // small materials buffer. No soup rewrite, no MeshBVH rebuild, no shader
    // recompile — probes re-trace the moved geometry on the next solve pass.
    // A null result means the instance set changed under us → full rebuild.
    function refreshTransforms(objects = null) {
        const built = cachedBuilt;
        if (!built?.updateTransforms || !casc[0].gpu) return false;
        if (sceneResource?.built !== built) { requestRebuild(); return false; }
        let res = null;
        try { res = built.updateTransforms(objects == null ? undefined : { objects }); } catch { res = null; }
        if (!res) { requestRebuild(); return false; }
        if (res.bounds && built.bounds?.copy) built.bounds.copy(res.bounds);
        lastTransformInstanceCount = res.updatedInstances | 0;
        lastTlasRefitCount = res.refittedTlasNodes | 0;
        if (Array.isArray(res.materialRanges) && res.materialRanges.length > 0) {
            copySceneStorageRanges('materials', built.materials, res.materialRanges);
        }
        if (res.emitterTransformsTouched) forceLightingRefresh();
        return true;
    }
    // Material-VALUE fast path: as long as the probes keep tracing, a
    // non-structural material edit is FREE — the affected resident uber records
    // are rewritten in place (spectral_scene updateMaterialValues) and those few
    // floats re-uploaded; the field re-converges through the bounded per-texel
    // change detector, multi-bounce included. No soup rewrite, no MeshBVH, no
    // recompile. Structural drift (map binding, material reassignment, dedup
    // split) fails closed to the full rebuild lane at its usual lowest priority.
    function refreshMaterialValues(targets = null) {
        const built = cachedBuilt;
        if (!built?.updateMaterialValues || !casc[0].gpu) return false;
        if (sceneResource?.built !== built) { requestRebuild(); return false; }
        let res = null;
        try { res = built.updateMaterialValues(targets == null ? undefined : { materials: targets }); } catch { res = null; }
        if (!res) { requestRebuild(); return false; }
        lastMaterialValueRecords = res.updatedRecords | 0;
        if (Array.isArray(res.materialRanges) && res.materialRanges.length > 0) {
            copySceneStorageRanges('materials', built.materials, res.materialRanges);
        }
        if (res.emitterValuesTouched) forceLightingRefresh();
        return true;
    }
    // Deforming-object fast path: re-gather the deformed BLAS vertex slices +
    // refit their node bounds in place (spectral_scene updateDeforms — no
    // MeshBVH, no allocation, no recompile), then re-upload ONLY the touched
    // slices of vertexData/bvhNodes plus the instance/TLAS tail. Probes
    // re-trace the deformed geometry on the next solve pass and re-converge
    // through the bounded per-texel change detector — the temporal policy
    // stays CONSTANT (no reactive burst). A null result means the vertex
    // count/layout changed under us → the soup is invalid → full rebuild.
    async function refreshDeforms() {
        const built = cachedBuilt;
        if ((!built?.updateDeformsAsync && !built?.updateDeforms) || !casc[0].gpu) return false;
        if (sceneResource?.built !== built) { requestRebuild(); return false; }
        let res = null;
        try {
            res = typeof built.updateDeformsAsync === 'function'
                ? await built.updateDeformsAsync()
                : built.updateDeforms();
        } catch { res = null; }
        if (disposed || cachedBuilt !== built) return false;
        if (!res) { requestRebuild(); return false; }
        if (res.bounds && built.bounds?.copy) built.bounds.copy(res.bounds);
        lastRefitCount = res.refitted;
        const updated = (res.updated ?? res.refitted) | 0;
        if (!updated) return true; // version churn outside the traced set — nothing to upload
        const matStart = built.instBase | 0;
        const matCount = Math.max(0, (built.tlasBase | 0)
            + ((built.tlasNodeCount | 0) * (built.strides?.TLAS_STRIDE_F32 || 12)) - matStart);
        copySceneStorageRanges('vertexData', built.vertexData, res.vertRanges);
        if (res.refitted) {
            copySceneStorageRanges('bvhNodes', built.bvhNodes, res.nodeRanges);
            copySceneStorageRanges('materials', built.materials, matCount > 0 ? [[matStart, matCount]] : null);
        }
        return true;
    }
    // needsUpdate re-uploads the WHOLE storage buffer; updateRanges (when the
    // three build supports them) bound the copy to the deformed slices. Keep
    // every disjoint range queued before the renderer consumes this version;
    // an explicit full upload clears the ranged list.
    function markStorageDirty(attr, ranges) {
        if (!attr) return;
        if (typeof attr.addUpdateRange === 'function' && typeof attr.clearUpdateRanges === 'function') {
            if (Array.isArray(ranges) && ranges.length > 0) {
                // Preserve disjoint ranges queued by earlier edits until the
                // renderer consumes this attribute version and clears them.
                for (const [start, count] of ranges) attr.addUpdateRange(start, count);
            } else {
                attr.clearUpdateRanges();
            }
        }
        attr.needsUpdate = true;
    }
    // ── clustered mode: importance top-K into the fixed arena ──
    function giLightImportance(rec, bmin, bmax) {
        if (rec[0] < 0.5) return Infinity;                    // directional: always keep
        const power = Math.max(rec[7], rec[8], rec[9]);       // records store color × intensity
        const cone = Math.abs(rec[0] - 2) < 0.5 ? Math.max(0.02, (1 - rec[12]) * 0.5) : 1;
        const dx = Math.max(bmin.x - rec[1], 0, rec[1] - bmax.x);
        const dy = Math.max(bmin.y - rec[2], 0, rec[2] - bmax.y);
        const dz = Math.max(bmin.z - rec[3], 0, rec[3] - bmax.z);
        return (power * cone) / (1 + dx * dx + dy * dy + dz * dz);
    }
    const _selMax = new THREE.Vector3();
    function selectGiLights(records, budget) {
        if (records.length <= budget) return records;
        const C0 = casc[0];
        const bmin = C0.gridMin;
        const bmax = _selMax.copy(C0.gridMin).add(C0.gridSize);
        const scored = records.map((rec, i) => [giLightImportance(rec, bmin, bmax), i]);
        scored.sort((a, b) => (b[0] - a[0]) || (a[1] - b[1]));   // score desc, index asc (deterministic)
        scored.length = budget;
        return scored.map(([, i]) => records[i]);
    }
    function selectGiLightRecords(records, importanceRanked) {
        const analytic = [];
        const emitters = [];
        for (const rec of records) {
            if (Math.abs(rec[0] - 3) < 0.5) emitters.push(rec);
            else if (rec[0] < 2.5) analytic.push(rec);
        }
        const emitterCount = Math.min(GI_EMITTER_INJECT_CAP, emitters.length);
        const analyticBudget = MAX_LIGHTS - emitterCount;
        const selectedAnalytic = importanceRanked
            ? selectGiLights(analytic, analyticBudget)
            : analytic.slice(0, analyticBudget);
        return {
            records: selectedAnalytic.concat(emitters.slice(0, emitterCount)),
            emitterBase: selectedAnalytic.length,
            emitterCount,
        };
    }
    function updateEmitterRegion(selection) {
        giEmitterBase = selection.emitterBase;
        giEmitterCount = selection.emitterCount;
        U.emitterBase.value = giEmitterBase >>> 0;
        U.emitterCount.value = giEmitterCount >>> 0;
    }
    function fillLegacyLightArena(records) {
        const need = giLightDataCount();
        const selection = selectGiLightRecords(records, false);
        const selected = selection.records;
        const count = selected.length;
        const resized = !giLegacyLightArena || giLegacyLightArena.length !== need;
        if (resized) giLegacyLightArena = new Float32Array(need);
        let recordDataChanged = resized || count !== giLegacyLightCount;
        if (!recordDataChanged) {
            outer: for (let i = 0; i < count; i++) {
                const rec = selected[i];
                const base = i * _LIGHT_STRIDE;
                for (let k = 0; k < _LIGHT_STRIDE; k++) {
                    if (giLegacyLightArena[base + k] !== rec[k]) {
                        recordDataChanged = true;
                        break outer;
                    }
                }
            }
        }
        giLegacyLightArena.fill(0);
        for (let i = 0; i < count; i++) giLegacyLightArena.set(selected[i], i * _LIGHT_STRIDE);
        giLegacyLightCount = count;
        updateEmitterRegion(selection);
        return { resized, recordDataChanged };
    }
    function fillGiLightArena(records) {
        // Float32 represents every small cell count/index exactly. Packing the
        // lists after the fixed record arena avoids a ninth storage binding.
        const need = giLightDataCount() + c0LightCellCount() * GI_LIGHT_CELL_STRIDE;
        const selection = selectGiLightRecords(records, true);
        const selected = selection.records;
        const resized = !giLightArena || giLightArena.length !== need;
        if (resized) giLightArena = new Float32Array(need);
        let recordDataChanged = resized || selected.length !== giSelectedCount;
        if (!recordDataChanged) {
            outer: for (let i = 0; i < selected.length; i++) {
                const rec = selected[i];
                const base = i * _LIGHT_STRIDE;
                for (let k = 0; k < _LIGHT_STRIDE; k++) {
                    if (giLightArena[base + k] !== rec[k]) {
                        recordDataChanged = true;
                        break outer;
                    }
                }
            }
        }
        giLightArena.fill(0);
        for (let i = 0; i < selected.length; i++) giLightArena.set(selected[i], i * _LIGHT_STRIDE);
        giSelectedCount = selected.length;
        updateEmitterRegion(selection);
        return { resized, recordDataChanged };
    }
    // Build-snapshot records (stride slices of the packed buffer) — keeps the arena
    // byte-consistent with the build without a second scene traverse.
    function recordsFromBuilt(built) {
        const n = built.lightCount | 0, out = [];
        for (let i = 0; i < n; i++) out.push(built.lights.subarray(i * _LIGHT_STRIDE, (i + 1) * _LIGHT_STRIDE));
        return out;
    }
    // Re-collect into the one fixed-capacity light buffer shared by both cascades.
    // Count changes update the uniform and buffer in place; they never rebuild BVHs.
    function refreshLights() {
        if (!casc[0].gpu || !_collectLights) return;
        let records;
        // No camera in this scope by design: the build path collects with
        // camera = null too, so refresh and build stay filter-consistent.
        try { records = _collectLights(THREE, scene); } catch { return; }
        liveLightRecords = records;
        if (clusteredGi) {
            // Fixed arena: count drift lands in the count uniform + an in-place
            // refill — never a rebuild. U.lightCount is the SHARED uniform folded
            // into both cascade blocks, so one write serves C0 and C1.
            const previousArena = giLightArena;
            fillGiLightArena(records);
            if (previousArena !== giLightArena || lightResource?.array !== giLightArena) {
                // This can only happen after a C0 grid-size change. Existing
                // kernels cannot rebind a differently-sized storage attribute.
                requestRebuild(false);
                return;
            }
            U.lightCount.value = Math.min(MAX_LIGHTS, giSelectedCount) >>> 0;
            for (const C of casc) {
                const g = C.gpu; if (!g) continue;
                g.lightCount = giSelectedCount;
            }
            markStorageDirty(lightResource.buffer, null);
            giLightGridDirty = true;
            return;
        }
        const previousArena = giLegacyLightArena;
        fillLegacyLightArena(records);
        if (previousArena !== giLegacyLightArena || lightResource?.array !== giLegacyLightArena) {
            requestRebuild(false);
            return;
        }
        U.lightCount.value = giLegacyLightCount >>> 0;
        for (const C of casc) {
            const g = C.gpu; if (!g) continue;
            g.lightCount = giLegacyLightCount;
        }
        markStorageDirty(lightResource.buffer, null);
    }

    function forceLightingRefresh() {
        scene.updateMatrixWorld?.(true);
        refreshLights();
        lastLightSig = detectSceneChanges ? lightSignature() : null;
        touchGiUniforms();
    }

    function setDebugKnobs(opts = {}) {
        const num = (key, fallback = null) => Number.isFinite(opts[key]) ? Number(opts[key]) : fallback;
        const clampNum = (key, min, max, target) => {
            const v = num(key);
            if (v === null) return;
            target.value = THREE.MathUtils.clamp(v, min, max);
        };

        let touchedReceiver = false;
        const samplePositionScale = num('samplePositionScale');
        if (samplePositionScale !== null) {
            node.samplePositionScaleNode.value = THREE.MathUtils.clamp(samplePositionScale, 0.001, 1000);
            touchedReceiver = true;
        }
        const sampleNormalMix = num('sampleNormalMix');
        if (sampleNormalMix !== null) {
            node.sampleNormalMixNode.value = THREE.MathUtils.clamp(sampleNormalMix, 0, 1);
            touchedReceiver = true;
        }
        const sampleBiasScale = num('sampleBiasScale');
        if (sampleBiasScale !== null) {
            node.sampleBiasScaleNode.value = THREE.MathUtils.clamp(sampleBiasScale, -8, 8);
            touchedReceiver = true;
        }
        const sampleObjectNormal = num('sampleObjectNormal');
        if (sampleObjectNormal !== null) {
            node.sampleObjectNormalNode.value = THREE.MathUtils.clamp(sampleObjectNormal, 0, 1);
            touchedReceiver = true;
        }
        const borderBand = num('borderBand');
        if (borderBand !== null) {
            node.borderBandNode.value = THREE.MathUtils.clamp(borderBand, 0.001, 1);
            touchedReceiver = true;
        }
        const chebyBias = num('chebyBiasScale');
        if (chebyBias !== null) {
            chebyBiasScale = THREE.MathUtils.clamp(chebyBias, 0, 16);
            for (let c = 0; c < NUM_CASC; c++) {
                node.chebyBiasNode[c].value = Math.max(0, casc[c].minCell * GI_CHEBY_BIAS_CELL * chebyBiasScale);
            }
            touchedReceiver = true;
        }
        if (touchedReceiver) touchGiUniforms();

        clampNum('traceBiasScale', 0, 32, U.debugTraceBiasScale);
        clampNum('rayEpsScale', 0, 32, U.debugRayEpsScale);
        clampNum('directScale', 0, 16, U.debugDirectScale);
        clampNum('emissiveScale', 0, 16, U.debugEmissiveScale);
        clampNum('albedoScale', 0, 4, U.debugAlbedoScale);
        clampNum('bounceScale', 0, 16, U.debugBounceScale);
        clampNum('cosinePower', 0.01, 16, U.debugCosinePower);
        clampNum('tempNoiseHBoost', 0, 1, U.debugTempNoiseHBoost);
        clampNum('tempChangeSigma0', 0, 8, U.debugTempChangeSigma0);
        clampNum('tempChangeSigma1', 0.01, 16, U.tempChangeSigma1);
        clampNum('tempChangeHDrop', 0, 0.99, U.tempChangeHDrop);
        clampNum('tempMinChangeH', 0, 0.99, U.debugTempMinChangeH);
        clampNum('tempVarEps', 0, 0.1, U.debugTempVarEps);
        clampNum('tempVarRel', 0, 1, U.debugTempVarRel);
        clampNum('tempClampSigma', 0.01, 64, U.tempClampSigma);
        clampNum('depthHistoryScale', 0, 4, U.debugDepthHistoryScale);
        clampNum('filterKScale', 0, 16, U.debugFilterKScale);
        clampNum('filterRelScale', 0, 16, U.debugFilterRelScale);
        clampNum('filterEpsScale', 0, 16, U.debugFilterEpsScale);

        if ('freezeRayJitter' in opts) debugFreezeRayJitter = opts.freezeRayJitter === true;
        if ('frameJitter' in opts) {
            const v = Number(opts.frameJitter);
            if (Number.isFinite(v) && v >= 0) {
                debugFrameJitterOverride = THREE.MathUtils.clamp(v, 0, 1);
                U.frameJitter.value = debugFrameJitterOverride;
            } else {
                debugFrameJitterOverride = null;
            }
        }
    }

    function getDebugKnobs() {
        return {
            samplePositionScale: node.samplePositionScaleNode.value,
            sampleNormalMix: node.sampleNormalMixNode.value,
            sampleBiasScale: node.sampleBiasScaleNode.value,
            sampleObjectNormal: node.sampleObjectNormalNode.value,
            borderBand: node.borderBandNode.value,
            chebyBiasScale,
            traceBiasScale: U.debugTraceBiasScale.value,
            rayEpsScale: U.debugRayEpsScale.value,
            directScale: U.debugDirectScale.value,
            emissiveScale: U.debugEmissiveScale.value,
            albedoScale: U.debugAlbedoScale.value,
            bounceScale: U.debugBounceScale.value,
            cosinePower: U.debugCosinePower.value,
            hysteresisNormalize,
            tempNoiseHBoost: U.debugTempNoiseHBoost.value,
            tempChangeSigma0: U.debugTempChangeSigma0.value,
            tempChangeSigma1: U.tempChangeSigma1.value,
            tempChangeHDrop: U.tempChangeHDrop.value,
            tempMinChangeH: U.debugTempMinChangeH.value,
            tempVarEps: U.debugTempVarEps.value,
            tempVarRel: U.debugTempVarRel.value,
            tempClampSigma: U.tempClampSigma.value,
            depthHistoryScale: U.debugDepthHistoryScale.value,
            filterKScale: U.debugFilterKScale.value,
            filterRelScale: U.debugFilterRelScale.value,
            filterEpsScale: U.debugFilterEpsScale.value,
            freezeRayJitter: debugFreezeRayJitter,
            frameJitter: debugFrameJitterOverride ?? -1,
        };
    }

    function dispose() {
        if (disposed) return;
        disposed = true;
        if (_activeProbeFieldOwner === fieldOwner) _activeProbeFieldOwner = null;
        // Invalidates any build currently awaiting material texture extraction.
        // Its continuation observes disposed before publishing cachedBuilt or
        // allocating cascade resources and disposes its uninstalled maps.
        buildGeneration++;
        cachedBuilt?.disposeDeformUpdates?.();
        disposeGPU();
        releaseLightResource(lightResource);
        releaseSceneResource(sceneResource);
        lightResource = null;
        sceneResource = null;
        liveLightRecords = null;
        cachedBuilt = null;
        blasCache = null;
        if (mapsArena) mapsArena.current = null;
        mapsArena = null;
        pendingTransformTargets.clear();
        pendingAllTransforms = false;
        pendingDeformRefresh = false;
        pendingMaterialValueTargets.clear();
        pendingAllMaterialValues = false;
        buildDirty = true;
        node.setEnabled(false);
        // Old handles retain this node reference, so detach the module singleton.
        // A later field receives a fresh node that a disposed handle cannot mutate.
        if (_node === node) _node = null;
    }

    const api = {
        node,
        tick,
        resetFramePacing,
        setEnabled,
        getEnabled: () => node._enabled,
        setIntensity: (v) => node.setIntensity(v),
        getIntensity: () => node.intensityNode.value,
        // Uniform/live. The structural atlas allocation is chosen once with
        // createProbeField({ reflectionQuality }) to keep the disabled path zero-cost.
        setReflectionIntensity: (v) => node.setReflectionIntensity(v),
        getReflectionIntensity: () => node.reflectionIntensityNode.value,
        setRoughnessLimit: (v) => node.setRoughnessLimit(v),
        getRoughnessLimit: () => node.roughnessLimitNode.value,
        setReflectionSkyFallback,
        getReflectionSkyFallback: () => U.reflectionSkyFallback.value > 0.5,
        hasRoughReflections: () => roughReflectionsEnabled,
        hasGlossyReflections: () => glossyReflectionsEnabled,
        getReflectionQuality: () => reflectionConfig.name,
        setChebyStrength: (v) => node.setChebyStrength(v),
        setNormalDetail: (v) => node.setDetailStrength(v),  // GI normal-map detail on trusted materials (0 = smooth, 1 = full)
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
            requestRebuild(false); // grid-only resize → reuse cached BVH+textures (no MeshBVH stall)
        },
        getDivisions: () => targetLongAxis,
        // ── STRUCTURAL knob: ray budget per probe. Re-sizes the ray scratch + rebuilds the
        // trace/blend kernels, so it goes through the idle-gated rebuild (never a per-tick recompile).
        setRays: (n) => {
            const v = THREE.MathUtils.clamp(Math.round(Number(n) / 16) * 16 || RAYS_PER_PROBE_DEFAULT, RAYS_MIN, RAYS_MAX);
            if (v === raysPerProbe) return;
            raysPerProbe = v;
            requestRebuild(false); // ray budget only → reuse cached BVH+textures (kernel rebuild, no MeshBVH stall)
        },
        getRays: () => raysPerProbe,
        // ── STRUCTURAL knob: per-tick trace budget. More rays/tick = faster light
        // propagation for more GPU; the cadence controller still shrinks the live
        // budget on frame-time pressure, so pacing is protected at any target.
        // Kernel rebuild only (scratch is budget-sized) — cached BVH soup reused.
        // Note MAX_PROBES_PER_TICK (2048) still bounds probes/tick, so at low
        // rays/probe a very high budget saturates early (e.g. 131k rays @64).
        setRayBudget: (v) => {
            if (!Number.isFinite(v)) return;
            const budget = THREE.MathUtils.clamp(Math.round(v), RAYS_PER_TICK_MIN, 524_288);
            if (budget === rayBudgetCeiling) return;
            rayBudgetCeiling = budget;
            tickBudgetRays = budget;   // take effect now; the throttle pulls back if the GPU can't
            requestRebuild(false);
        },
        getRayBudget: () => rayBudgetCeiling,
        // ── UNIFORM knobs (apply INSTANTLY — no recompile, no rebuild). ──
        setFilterStrength: (v) => { if (Number.isFinite(v)) U.filterStrength.value = THREE.MathUtils.clamp(v, 0, 1); }, // CORE denoise: 0 = off (harness baseline), 1 = full
        setSmoothness: (v) => { if (Number.isFinite(v)) U.filterSmooth.value = THREE.MathUtils.clamp(v, 0, 1); }, // UI "Smoothness": widen the denoise edge-stop
        setHysteresis: (v) => {
            if (!Number.isFinite(v)) return;
            baseHysteresis = THREE.MathUtils.clamp(v, 0, 0.99); // steady-state temporal blend (higher = more stable/slower)
            jitterHysteresis[jitterMode] = baseHysteresis;      // each sampling mode remembers its deliberate override
            U.hysteresis.value = baseHysteresis;                // apply now; tick() re-asserts it every solve
        },
        getHysteresis: () => baseHysteresis,
        setHysteresisNormalization: (on) => {
            hysteresisNormalize = on !== false;
            U.hysteresis.value = baseHysteresis;
            if (!hysteresisNormalize) U.hysteresisExponent.value = 1;
        },
        getHysteresisNormalization: () => hysteresisNormalize,
        // Ray-jitter regime: 'gated' (GATED BASIS, default) vs 'montecarlo'.
        // Switching restores that mode's remembered hysteresis profile. Both
        // writes are live uniforms/CPU state — instant, no recompile or rebuild.
        setJitterMode: (mode) => {
            const m = String(mode ?? '').toLowerCase().replace(/[\s_-]/g, '');
            const nextMode = (m === 'gated' || m === 'gatedbasis') ? 'gated'
                : (m === 'montecarlo' || m === 'mc') ? 'montecarlo' : null;
            if (!nextMode || nextMode === jitterMode) return;
            jitterHysteresis[jitterMode] = baseHysteresis;
            jitterMode = nextMode;
            baseHysteresis = jitterHysteresis[jitterMode];
            U.hysteresis.value = baseHysteresis;
        },
        getJitterMode: () => jitterMode,
        setNormalBias: (v) => {
            if (!Number.isFinite(v)) return;
            normalBiasScale = THREE.MathUtils.clamp(v, 0, 8);   // × the auto minCell·SURFACE_NORMAL_BIAS_CELL offset
            // per-cascade: each cascade scales the offset by ITS OWN minCell (coarse cells larger).
            for (let c = 0; c < NUM_CASC; c++) {
                node.normalBiasNode[c].value = Math.max(1e-4, casc[c].minCell * SURFACE_NORMAL_BIAS_CELL * normalBiasScale);
            }
            touchGiUniforms();
        },
        setRadianceClamp: (v) => {
            if (!Number.isFinite(v)) return;
            const next = Math.max(0, v);
            if (U.radianceClamp.value === next) return;
            U.radianceClamp.value = next;   // cap multibounce feedback (anti-runaway)
            // Trace-side knob: newly traced rays carry the new clamp; texels it
            // really moves re-converge through the bounded change detector.
        },
        // depth-moment cosine power (Chebyshev crispness). 0 = uniform-ish weighting;
        // floored at 0.01 because pow(0,0) is indeterminate in WGSL (cw is exactly 0
        // for backfacing rays) and would poison the depth history with NaN.
        setDepthSharpness: (v) => { if (Number.isFinite(v)) U.depthSharpness.value = Math.max(0.01, THREE.MathUtils.clamp(v, 0, 200)); },
        setSky,   // inject the sky: null | Color/hex | {zenith,horizon,ground} | LightProbe/SphericalHarmonics3 (radiance SH)
        setSkyIntensity: (v) => {
            if (!Number.isFinite(v)) return;
            const next = Math.max(0, v);
            if (U.skyIntensity.value === next) return;
            U.skyIntensity.value = next;    // sky radiance on miss rays (0 = off)
        },
        getSkyIntensity: () => U.skyIntensity.value,
        // ── NIR band sensing (white-phosphor NV). Trace-side gate for emitter-class-4
        // IR illuminators: 0 = visible band (IR lights dark, the physical truth),
        // 1 = NIR (IR lights join NEE at white × intensity). Pair with
        // setNirDirectSensing (gi_lights_node) so the DIRECT raster term flips too —
        // installSpeedballGI's setNirSensing does both.
        setNirSensing: (on) => {
            const v = on ? 1.0 : 0.0;
            if (U.nirGate.value === v) return;
            U.nirGate.value = v;
        },
        getNirSensing: () => U.nirGate.value > 0.5,
        // Scalar trim on the gated class-4 NEE term (sensed-band illuminator
        // brightness). Same uniform group as the gate — live, no recompile.
        setNirGain: (gain) => {
            const v = Number.isFinite(gain) ? Math.max(0, gain) : 1;
            if (U.nirGain.value === v) return;
            U.nirGain.value = v;
        },
        getNirGain: () => U.nirGain.value,
        // ── adaptive-blend tuning (live; no recompile). Tune "stable continuous" by feel. ──
        setChangeThreshold: (v) => { if (Number.isFinite(v)) U.tempChangeSigma1.value = THREE.MathUtils.clamp(v, 0.5, 8); },   // σ delta to treat a change as REAL — lower = snappier
        setSnapAmount: (v) => { if (Number.isFinite(v)) U.tempChangeHDrop.value = THREE.MathUtils.clamp(v, 0, 0.9); },         // hysteresis drop on a real change — higher = harder snap
        setFireflyClamp: (v) => { if (Number.isFinite(v)) U.tempClampSigma.value = THREE.MathUtils.clamp(v, 1, 20); },         // clamp band in σ — lower = steadier, more lag
        getChangeThreshold: () => U.tempChangeSigma1.value,
        getSnapAmount: () => U.tempChangeHDrop.value,
        getFireflyClamp: () => U.tempClampSigma.value,
        setDebugKnobs,
        getDebugKnobs,
        forceLightingRefresh,
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
        setAutoDetectChanges: (on) => { detectSceneChanges = on !== false; },
        getAutoDetectChanges: () => detectSceneChanges,
        requestRebuild,
        markTransformsDirty,
        markDeformsDirty,
        markMaterialValuesDirty,
        markTopologyDirty,
        notifySceneChange,
        setBounds,
        setVolumes,
        isSupported,
        hasData: () => node._ready === true,
        // getStats/getResolution/getBounds/_debug* take an optional cascade index
        // (default 0 = coarse, preserving current callers).
        getStats: (ci = 0) => { const C = casc[ci] || casc[0]; return {
            probes: C.probeTotal, res: C.res.clone(), atlas: [C.atlasW, C.atlasH],
            rays: raysPerProbe, oct: OCT_RES, tile: TILE, active: node.active,
            roughReflections: roughReflectionsEnabled,
            reflectionQuality: reflectionConfig.name,
            roughnessLimit: node.roughnessLimitNode.value,
            roughSpecularPower: roughReflectionsEnabled ? ROUGH_SPECULAR_POWER : 0,
            glossySpecularPower: glossyReflectionsEnabled ? GLOSSY_SPECULAR_POWER : 0,
            glossyOct: glossyReflectionsEnabled ? glossyOctRes : 0,
            glossyTile: glossyReflectionsEnabled ? glossyTile : 0,
            glossyUpdateInterval,
            glossyProbeGather: glossyReflectionsEnabled ? 8 : 0,
            glossyAtlas: glossyReflectionsEnabled ? [C.glossyAtlasW, C.glossyAtlasH] : [0, 0],
            cascades, cascade: ci, budgetRays: tickBudgetRays,
        }; },
        getResolution: (ci = 0) => (casc[ci] || casc[0]).res.clone(),
        getBounds: (ci = 0) => { const C = casc[ci] || casc[0]; return new THREE.Box3(C.gridMin.clone(), C.gridMin.clone().add(C.gridSize)); },
        _debugUpload: async (ci = 0) => { const g = casc[ci]?.gpu; if (g && !disposed) { try { await renderer.computeAsync(g.uploadKernel); } catch (e) { /* harness-only */ } } },
        _debugAtlas: (ci = 0) => casc[ci]?.gpu?.atlas || null,
        _debugRoughSpecularAtlas: (ci = 0) => casc[ci]?.gpu?.roughSpecularAtlas || null,
        _debugGlossySpecularAtlas: (ci = 0) => casc[ci]?.gpu?.glossySpecularAtlas || null,
        _debugDepthAtlas: (ci = 0) => casc[ci]?.gpu?.depthAtlas || null,
        _debugStateAtlas: (ci = 0) => casc[ci]?.gpu?.stateAtlas || null,
        _debugStateBuffer: (ci = 0) => casc[ci]?.gpu?.stateBuffer || null,
        _debugBuffers: (ci = 0) => casc[ci]?.gpu?.buffers || null,
        _debugState: () => ({
            idleMs: lastIdleMs,
            moving: lastMoving,
            playing: lastPlaying,
            restOnly: lastRestOnly,
            continuous,
            dirty,
            buildDirty,
            buildStage,
            buildCascadeCount,
            rebuildBackoff,
            inFlight,
            baseHysteresis,
            hysteresis: U.hysteresis.value,
            hysteresisExponent: U.hysteresisExponent.value,
            frameJitter: U.frameJitter.value,
            frameCounter,
            jitterMode,
            jitterHysteresis: { ...jitterHysteresis },
            tickBudgetRays,
            tickDtEma,
            sceneResourceRefs: sceneResource?.refs || 0,
            sceneStorageGeneration: sceneResource?.storage?.generation || 0,
            sceneStorageRebinds,
            sceneStorageRewrites,
            sceneStorageLastUpdate,
            sceneStorageCapacities: sceneResource?.storage
                ? { ...sceneResource.storage.capacities }
                : null,
            sceneStorageLiveLengths: sceneResource?.storage
                ? { ...sceneResource.storage.liveLengths }
                : null,
            mapsArenaGeneration: mapsArena?.current?.generation || 0,
            mapsArenaRebinds,
            mapsArenaRewrites,
            mapsArenaLastUpdate,
            mapsArenaCapacities: mapsArena?.current
                ? { ...mapsArena.current.capacities }
                : null,
            mapsArenaLiveLayers: mapsArena?.current
                ? { ...mapsArena.current.liveLayers }
                : null,
            kernelResidentReuses,
            kernelRebuilds,
            lightResourceRefs: lightResource?.refs || 0,
            sharedSceneBuffers: !!casc[0].gpu && !!casc[1].gpu
                && PROBE_SCENE_BUFFER_KEYS.every((key) => casc[0].gpu.buffers[key] === casc[1].gpu.buffers[key]),
            sharedLightBuffer: !!casc[0].gpu && !!casc[1].gpu
                && casc[0].gpu.buffers.lights === casc[1].gpu.buffers.lights,
            hysteresisTickDtEma,
            budgetCooldown,
            cadenceOverloadStreak,
            checkCounter,
            geoStable,
            lastRefitCount,
            lastTransformInstanceCount,
            lastTlasRefitCount,
            blasCache: blasCache
                ? { hits: blasCache.hits, misses: blasCache.misses, triangles: blasCache.triangles, entries: blasCache.map.size }
                : null,
            pendingTransformCount: pendingAllTransforms ? -1 : pendingTransformTargets.size,
            pendingDeformRefresh,
            pendingMaterialValueCount: pendingAllMaterialValues ? -1 : pendingMaterialValueTargets.size,
            lastMaterialValueRecords,
            solveList: lastSolveList,
            updatedCount: lastUpdatedCount,
            cascades: cascades,
            cascadeState: casc.map((C) => ({
                probes: C.probeTotal,
                cursor: C.probeCursor,
                solveDtEma: C.solveDtEma,
                needsClear: C.needsClear,
                needsClassify: C.needsClassify,
                hasGpu: !!C.gpu,
                minCell: C.minCell,
            })),
        }),
        _debugRead: async (which, ci = 0) => {
            const g = casc[ci]?.gpu;
            const buf = which === 'irr' ? g?.irrBuffer
                : which === 'spec' ? g?.roughSpecularBuffer
                : which === 'gloss' ? g?.glossySpecularBuffer
                : which === 'glossWeight' ? g?.glossyWeightBuffer
                : which === 'mat' ? g?.buffers?.materials
                : which === 'lights' ? g?.buffers?.lights
                : which === 'state' ? g?.stateBuffer : null;
            if (!buf || typeof renderer.getArrayBufferAsync !== 'function') return null;
            try { return new Float32Array(await renderer.getArrayBufferAsync(buf)); } catch (e) { return { error: String(e) }; }
        },
        _debugLightCount: (ci = 0) => casc[ci]?.gpu?.lightCount,
        dispose,
    };
    _activeProbeFieldOwner = fieldOwner;
    return api;
}

export default createProbeField;
