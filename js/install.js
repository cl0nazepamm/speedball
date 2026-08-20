// install.js — one-call setup for SPEEDBALL GI.
//
// Collapses the wiring that the probe field needs into a single call plus one
// update() per frame:
//   1. installs the lights-node factory so GI folds into every PBR material,
//   2. wires the post-rebuild material-dirty pass (the silent-"GI-missing" footgun),
//   3. tracks world-space camera interaction for rebuild gating and solve pacing.
//
// Advanced users can still import createProbeField / giLights directly and wire it
// by hand; this is the batteries-included path.

import * as THREE from 'three/webgpu';
import { createProbeField } from './gi_probes.js';
import { giLights, setNirDirectSensing, setNirIlluminatorGain } from './gi_lights_node.js';
import { giClusteredLights } from './gi_clustered_lights_node.js';

// spectral_scene / gi_probes gate: userData.maxjsVisible === false → kept out of the
// GI BVH and the auto-fit bounds. Wrapped here so callers never touch the raw flag.
const EXCLUDE_FLAG = 'maxjsVisible';

/**
 * Exclude an object (and its subtree) from the GI trace and grid auto-fit. Use for
 * sky domes, gizmos, probe helpers — anything that shouldn't bounce light or inflate
 * the auto-fit bounds.
 *
 * @param {THREE.Object3D} object
 * @returns {THREE.Object3D} the same object, for chaining
 */
export function excludeFromGI(object) {
    object.traverse((o) => { o.userData = o.userData || {}; o.userData[EXCLUDE_FLAG] = false; });
    return object;
}

/**
 * Normalize materials so metals actually bounce diffuse GI. The trace path kills
 * Lambert bounce on metals (metalness → no diffuse), so an all-metal import reads
 * as black GI. Opt-in — this mutates materials in place.
 *
 * @param {THREE.Object3D} scene
 * @param {{ maxMetalness?: number, minRoughness?: number }} [opts]
 */
export function prepareMaterialsForGI(scene, { maxMetalness = 0.5, minRoughness = 0.6 } = {}) {
    const seen = new WeakSet();
    scene.traverse((o) => {
        const list = Array.isArray(o.material) ? o.material : (o.material ? [o.material] : []);
        for (const m of list) {
            if (!m || seen.has(m)) continue; seen.add(m);
            if (Number.isFinite(m.metalness) && m.metalness > maxMetalness) { m.metalness = 0; m.needsUpdate = true; }
            if (Number.isFinite(m.roughness) && m.roughness < minRoughness) { m.roughness = minRoughness; m.needsUpdate = true; }
        }
    });
}

// Force the lights graph to recompile so a newly-active GI node folds into every PBR
// program. In current three WebGPU, material.needsUpdate alone does NOT rebuild the
// node graph — disposing the compiled program does. Runs only on a GI rebuild (first
// data / grid resize), never per frame. (Do NOT compileAsync() here: it bakes a
// variant that can omit the just-activated GI node, silently dropping GI.)
function makeMaterialDirtier(scene) {
    return () => {
        const seen = new WeakSet();
        const mark = (m) => {
            if (!m || seen.has(m)) return; seen.add(m);
            if (m.isMeshBasicMaterial || m.isLineBasicMaterial || m.isLineDashedMaterial) return;
            m.dispose?.();
            m.needsUpdate = true;
        };
        scene.traverse((o) => {
            if (!o.material) return;
            if (Array.isArray(o.material)) o.material.forEach(mark); else mark(o.material);
        });
    };
}

const _now = () => (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();

/**
 * Escape hatch for renderer integrations and diagnostics. Everything here is
 * intentionally outside the stable installer contract and may change between
 * releases. Prefer the methods on {@link SpeedballGIHandle} in application code.
 *
 * @typedef {object} SpeedballGIAdvanced
 * @property {object} node raw probe node used by the material lights graph
 * @property {Function} tick raw field tick; normal render loops should call update()
 * @property {Function} resetFramePacing reset the internal frame-time controller
 * @property {Function} requestRebuild request a structural field rebuild
 * @property {Function} setDebugKnobs set diagnostic-only receiver knobs
 * @property {Function} getDebugKnobs read diagnostic-only receiver knobs
 * @property {object} debug GPU-resource inspection helpers for test harnesses
 */

/**
 * Stable handle returned by {@link installSpeedballGI}.
 *
 * @typedef {object} SpeedballGIHandle
 * @property {(options?: {camera?: THREE.Camera, playing?: boolean}) => (Promise<void>|undefined)} update
 * @property {() => void} markInteraction
 * @property {() => void} dispose
 * @property {(enabled: boolean) => void} setEnabled
 * @property {() => boolean} getEnabled
 * @property {(intensity: number) => void} setIntensity
 * @property {() => number} getIntensity
 * @property {(divisions: number) => void} setDivisions
 * @property {() => number} getDivisions
 * @property {(rays: number) => void} setRays
 * @property {() => number} getRays
 * @property {(rays: number) => void} setRayBudget
 * @property {() => number} getRayBudget
 * @property {(cascades: number) => void} setCascades
 * @property {() => number} getCascades
 * @property {(continuous: boolean) => void} setContinuous
 * @property {() => boolean} getContinuous
 * @property {(enabled: boolean) => void} setAutoDetectChanges
 * @property {() => boolean} getAutoDetectChanges
 * @property {(mode: 'gated'|'montecarlo') => void} setJitterMode
 * @property {() => ('gated'|'montecarlo')} getJitterMode
 * @property {(hysteresis: number) => void} setHysteresis
 * @property {() => number} getHysteresis
 * @property {(enabled: boolean) => void} setHysteresisNormalization
 * @property {() => boolean} getHysteresisNormalization
 * @property {Function} setReflectionIntensity
 * @property {Function} getReflectionIntensity
 * @property {Function} setRoughnessLimit
 * @property {Function} getRoughnessLimit
 * @property {Function} setReflectionSkyFallback
 * @property {Function} getReflectionSkyFallback
 * @property {Function} hasRoughReflections
 * @property {Function} hasGlossyReflections
 * @property {Function} getReflectionQuality
 * @property {Function} setFilterStrength
 * @property {Function} setSmoothness
 * @property {Function} setNormalBias
 * @property {Function} setRadianceClamp
 * @property {Function} setDepthSharpness
 * @property {Function} setChebyStrength
 * @property {Function} setNormalDetail
 * @property {Function} setClassifyStrength
 * @property {Function} setSky
 * @property {Function} setSkyIntensity
 * @property {Function} getSkyIntensity
 * @property {Function} setNirSensing
 * @property {Function} getNirSensing
 * @property {Function} setNirGain
 * @property {Function} getNirGain
 * @property {Function} setChangeThreshold
 * @property {Function} getChangeThreshold
 * @property {Function} setSnapAmount
 * @property {Function} getSnapAmount
 * @property {Function} setFireflyClamp
 * @property {Function} getFireflyClamp
 * @property {Function} markTransformsDirty
 * @property {Function} markDeformsDirty
 * @property {Function} markMaterialValuesDirty
 * @property {Function} markTopologyDirty
 * @property {Function} notifySceneChange
 * @property {Function} forceLightingRefresh
 * @property {Function} markMaterialsDirty
 * @property {Function} setBounds
 * @property {Function} setVolumes
 * @property {() => boolean} isSupported
 * @property {() => boolean} hasData
 * @property {Function} getStats
 * @property {Function} getResolution
 * @property {Function} getBounds
 * @property {SpeedballGIAdvanced} advanced
 */

/**
 * Install SPEEDBALL GI on a WebGPU renderer + scene in one call.
 *
 * IMPORTANT: call this at SETUP, before the first render / before
 * renderer.setAnimationLoop(). It installs the lights-node factory, which must be
 * in place before any lit material compiles — if the render loop has already run,
 * the renderer caches a non-GI lights node and GI never folds in. The scene may be
 * empty at install time; the probe field auto-fits and builds once geometry appears.
 *
 * @example
 *   const gi = installSpeedballGI({ renderer, scene, camera });  // at setup
 *   // render loop:
 *   gi.update();            // one call per render frame
 *
 * @param {object} opts
 * @param {THREE.WebGPURenderer} opts.renderer
 * @param {THREE.Scene} opts.scene
 * @param {THREE.Camera} [opts.camera]         used for idle detection; pass per-frame to update() to override
 * @param {boolean} [opts.enabled=true]
 * @param {number}  [opts.intensity=10]        canonical demo tuning (Sponza)
 * @param {number}  [opts.divisions=16]        probes along the longest grid axis
 * @param {number}  [opts.rays=64]             rays traced per probe (32..256, quantized to 16)
 * @param {1|2}     [opts.cascades=2]           one full field or coarse + fine fields
 * @param {boolean} [opts.continuous=true]      keep the bounded solve running during interaction
 * @param {'gated'|'montecarlo'} [opts.jitterMode='gated'] stable held basis or fresh basis every solve
 * @param {number}  [opts.hysteresis]          mode default is 0.60 Gated / 0.90 Monte Carlo
 * @param {boolean} [opts.roughReflections=false] legacy reflection switch (`true` = ultra, `false` = off)
 * @param {'off'|'rough'|'high'|'ultra'} [opts.reflectionQuality] structural reflection tier; overrides roughReflections
 * @param {number}  [opts.reflectionIntensity=1] local-vs-environment reflection coverage blend, 0..1
 * @param {number}  [opts.roughnessLimit] skip local receiver work above this material roughness; pinned at 1 when omitted
 * @param {boolean} [opts.reflectionSkyFallback=false] fill reflection misses from setSky() SH instead of leaving them for PMREM/SSR
 * @param {object}  [opts.lights]              max light counts for the batched lights node
 * @param {boolean|object} [opts.clusteredLighting=false]  SECONDARY MODE (three r185+): Forward+
 *                  clustered raster lighting — thousands of non-shadowed point lights for
 *                  near-constant direct cost (GiClusteredLightsNode replaces the batched
 *                  DynamicLightsNode; `lights` caps are then ignored), while the GI lane
 *                  budgets itself to the MAX_LIGHTS most important records (importance
 *                  top-K + fixed light arena — light-count changes never trigger a BVH
 *                  rebuild in this mode). Directional/spot/shadow-casting lights keep the
 *                  stock per-light path. Pass an object to tune the cluster grid:
 *                  { maxLights=1024, tileSize=32, zSlices=24, maxLightsPerCluster=64 }.
 *                  Default false = the primary path, byte-identical to previous releases.
 * @param {boolean} [opts.installLightsNode=true]  set false if you install your own GI-aware lights node
 * @param {boolean} [opts.prepareMaterials=false]  run prepareMaterialsForGI(scene) on install
 * @param {boolean} [opts.autoDetectChanges=true]  compatibility scene scans; set false when the host emits dirty events
 * @param {(error: unknown) => void} [opts.onError] called for asynchronous update failures
 * @returns {SpeedballGIHandle}
 */
export function installSpeedballGI({
    renderer,
    scene,
    camera,
    enabled = true,
    intensity = 10,
    divisions = 16,
    rays = 64,
    cascades = 2,
    continuous = true,
    hysteresis = null,
    jitterMode = 'gated',
    roughReflections = false,
    reflectionQuality = null,
    reflectionIntensity = 1,
    roughnessLimit = null,
    reflectionSkyFallback = false,
    lights = { maxDirectionalLights: 4, maxPointLights: 16, maxSpotLights: 16, maxHemisphereLights: 2 },
    clusteredLighting = false,
    installLightsNode = true,
    prepareMaterials = false,
    autoDetectChanges = true,
    onError = null,
} = {}) {
    if (!renderer || !scene) throw new Error('installSpeedballGI: { renderer, scene } are required.');
    if (onError !== null && typeof onError !== 'function') {
        throw new TypeError('installSpeedballGI: onError must be a function when provided.');
    }

    const clustered = clusteredLighting === true
        || (typeof clusteredLighting === 'object' && clusteredLighting !== null);
    const clusteredOpts = clustered && typeof clusteredLighting === 'object' ? clusteredLighting : {};

    // 1. Lights factory — one line folds GI into every PBR material (no per-material wiring).
    let prevCreateNode = null;
    let installedCreateNode = null;
    if (installLightsNode) {
        if (!renderer.lighting) throw new Error('installSpeedballGI: renderer.lighting is missing — needs a WebGPURenderer.');
        // Guard the one sharp edge: the factory must be in place before the first render.
        // If the loop already ran, the renderer has cached a non-GI lights node and GI
        // will silently never fold in — warn loudly rather than fail quietly.
        const framesRendered = renderer.info?.render?.frameCount ?? renderer.info?.render?.frame ?? 0;
        if (framesRendered > 0) {
            console.warn(`installSpeedballGI: called after ${framesRendered} frame(s) already rendered. ` +
                'Install it at setup, BEFORE the first render / renderer.setAnimationLoop(), ' +
                'or GI may never fold into already-compiled materials.');
        }
        prevCreateNode = renderer.lighting.createNode || null;
        installedCreateNode = clustered
            ? (lightList = []) => giClusteredLights(clusteredOpts).setLights(lightList)
            : (lightList = []) => giLights(lights).setLights(lightList);
        renderer.lighting.createNode = installedCreateNode;
    }

    // 2. Probe field, with the material-dirty pass wired as onRebuilt (footgun handled).
    const markMaterialsDirty = makeMaterialDirtier(scene);
    let gi;
    try {
        if (prepareMaterials) prepareMaterialsForGI(scene);
        gi = createProbeField({
            renderer,
            scene,
            intensity,
            hysteresis,
            jitterMode,
            divisions,
            rays,
            cascades,
            continuous,
            roughReflections,
            reflectionQuality,
            reflectionIntensity,
            roughnessLimit,
            reflectionSkyFallback,
            clusteredLighting: clustered,
            autoDetectChanges,
            onRebuilt: markMaterialsDirty,
        });
    } catch (error) {
        if (installLightsNode && renderer.lighting?.createNode === installedCreateNode) {
            renderer.lighting.createNode = prevCreateNode;
        }
        throw error;
    }
    if (enabled) gi.setEnabled(true);

    // 3. Idle tracking off the camera transform → works with any controls (or none):
    //    hold the world-space field static while the view moves, converge when it rests.
    let lastInteraction = _now();
    const _pos = new THREE.Vector3(Infinity, 0, 0);
    const _quat = new THREE.Quaternion(2, 0, 0, 0);
    const _worldPos = new THREE.Vector3();
    const _worldQuat = new THREE.Quaternion();
    const cameraMoved = (cam) => {
        if (!cam) return false;
        cam.getWorldPosition(_worldPos);
        cam.getWorldQuaternion(_worldQuat);
        const moved = _worldPos.distanceToSquared(_pos) > 1e-7
            || Math.abs(_worldQuat.dot(_quat)) < 0.99999995;
        _pos.copy(_worldPos); _quat.copy(_worldQuat);
        return moved;
    };
    const _drawingSize = new THREE.Vector2();
    let drawingWidth = -1;
    let drawingHeight = -1;
    const presentationSizeChanged = () => {
        if (typeof renderer.getDrawingBufferSize !== 'function') return false;
        renderer.getDrawingBufferSize(_drawingSize);
        const width = Math.round(_drawingSize.x);
        const height = Math.round(_drawingSize.y);
        const changed = drawingWidth >= 0 && (width !== drawingWidth || height !== drawingHeight);
        drawingWidth = width;
        drawingHeight = height;
        return changed;
    };
    presentationSizeChanged();
    let disposed = false;
    let updateFailureWarned = false;

    const mutate = (method) => (...args) => {
        if (disposed) return;
        return gi[method](...args);
    };
    const query = (method) => (...args) => gi[method](...args);
    const advanced = Object.freeze({
        node: gi.node,
        tick: mutate('tick'),
        resetFramePacing: mutate('resetFramePacing'),
        requestRebuild: mutate('requestRebuild'),
        setDebugKnobs: mutate('setDebugKnobs'),
        getDebugKnobs: query('getDebugKnobs'),
        debug: Object.freeze({
            upload: mutate('_debugUpload'),
            atlas: query('_debugAtlas'),
            roughSpecularAtlas: query('_debugRoughSpecularAtlas'),
            glossySpecularAtlas: query('_debugGlossySpecularAtlas'),
            depthAtlas: query('_debugDepthAtlas'),
            stateAtlas: query('_debugStateAtlas'),
            stateBuffer: query('_debugStateBuffer'),
            buffers: query('_debugBuffers'),
            state: query('_debugState'),
            read: query('_debugRead'),
            lightCount: query('_debugLightCount'),
        }),
    });

    return {
        /** Call once per frame. Heavy rebuilds stay idle-gated; solve cadence follows setContinuous(). */
        update({ camera: cam = camera, playing = false } = {}) {
            if (disposed) return;
            const now = _now();
            if (cameraMoved(cam)) lastInteraction = now;
            if (presentationSizeChanged()) gi.resetFramePacing();
            return gi.tick({ idleMs: now - lastInteraction, playing }).then((result) => {
                updateFailureWarned = false;
                return result;
            }).catch((error) => {
                if (disposed || updateFailureWarned) return;
                updateFailureWarned = true;
                if (onError) {
                    try {
                        onError(error);
                    } catch (callbackError) {
                        console.warn('SPEEDBALL GI onError callback failed:', callbackError);
                    }
                    return;
                }
                console.warn('SPEEDBALL GI update failed:', error);
            });
        },

        /** Treat "now" as an interaction, deferring the next solve (e.g. after a big edit). */
        markInteraction() {
            if (!disposed) lastInteraction = _now();
        },

        setEnabled: mutate('setEnabled'),
        getEnabled: query('getEnabled'),
        setIntensity: mutate('setIntensity'),
        getIntensity: query('getIntensity'),
        setReflectionIntensity: mutate('setReflectionIntensity'),
        getReflectionIntensity: query('getReflectionIntensity'),
        setRoughnessLimit: mutate('setRoughnessLimit'),
        getRoughnessLimit: query('getRoughnessLimit'),
        setReflectionSkyFallback: mutate('setReflectionSkyFallback'),
        getReflectionSkyFallback: query('getReflectionSkyFallback'),
        hasRoughReflections: query('hasRoughReflections'),
        hasGlossyReflections: query('hasGlossyReflections'),
        getReflectionQuality: query('getReflectionQuality'),
        setChebyStrength: mutate('setChebyStrength'),
        setNormalDetail: mutate('setNormalDetail'),
        setClassifyStrength: mutate('setClassifyStrength'),
        setDivisions: mutate('setDivisions'),
        getDivisions: query('getDivisions'),
        setRays: mutate('setRays'),
        getRays: query('getRays'),
        setRayBudget: mutate('setRayBudget'),
        getRayBudget: query('getRayBudget'),
        setFilterStrength: mutate('setFilterStrength'),
        setSmoothness: mutate('setSmoothness'),
        setHysteresis: mutate('setHysteresis'),
        getHysteresis: query('getHysteresis'),
        setHysteresisNormalization: mutate('setHysteresisNormalization'),
        getHysteresisNormalization: query('getHysteresisNormalization'),
        setJitterMode: mutate('setJitterMode'),
        getJitterMode: query('getJitterMode'),
        setNormalBias: mutate('setNormalBias'),
        setRadianceClamp: mutate('setRadianceClamp'),
        setDepthSharpness: mutate('setDepthSharpness'),
        setSky: mutate('setSky'),
        setSkyIntensity: mutate('setSkyIntensity'),
        getSkyIntensity: query('getSkyIntensity'),

        /**
         * NIR band sensing (white-phosphor NV filter on/off). One switch for BOTH terms
         * of emitter-class-'ir' lights: the probes' NEE gate (GI) and the direct raster
         * gate (gi_lights_node). Uniform writes only — no recompile, no scene mutation;
         * light.color stays black, so the light never exists in the visible band.
         */
        setNirSensing(on) {
            if (disposed) return;
            gi.setNirSensing(on);
            setNirDirectSensing(on);
        },

        /**
         * Sensed-band illuminator gain: one scalar trim for BOTH raster terms of
         * emitter-class-'ir' lights (probes' NEE + direct raster). Uniform writes
         * only. Hosts running the spectral tracer wire its setNirGain alongside.
         */
        setNirGain(gain) {
            if (disposed) return;
            gi.setNirGain?.(gain);
            setNirIlluminatorGain(gain);
        },

        getNirSensing: query('getNirSensing'),
        getNirGain: query('getNirGain'),
        setChangeThreshold: mutate('setChangeThreshold'),
        setSnapAmount: mutate('setSnapAmount'),
        setFireflyClamp: mutate('setFireflyClamp'),
        getChangeThreshold: query('getChangeThreshold'),
        getSnapAmount: query('getSnapAmount'),
        getFireflyClamp: query('getFireflyClamp'),
        forceLightingRefresh: mutate('forceLightingRefresh'),
        setCascades: mutate('setCascades'),
        getCascades: query('getCascades'),
        setContinuous: mutate('setContinuous'),
        getContinuous: query('getContinuous'),
        setAutoDetectChanges: mutate('setAutoDetectChanges'),
        getAutoDetectChanges: query('getAutoDetectChanges'),
        markTransformsDirty: mutate('markTransformsDirty'),
        markDeformsDirty: mutate('markDeformsDirty'),
        markMaterialValuesDirty: mutate('markMaterialValuesDirty'),
        markTopologyDirty: mutate('markTopologyDirty'),
        notifySceneChange: mutate('notifySceneChange'),
        setBounds: mutate('setBounds'),
        setVolumes: mutate('setVolumes'),
        isSupported: query('isSupported'),
        hasData: query('hasData'),
        getStats: query('getStats'),
        getResolution: query('getResolution'),
        getBounds: query('getBounds'),

        /** Recompile lit materials so GI folds in — call if you add meshes after install. */
        markMaterialsDirty() {
            if (!disposed) markMaterialsDirty();
        },

        advanced,

        /** Full teardown: restore the previous lights factory and free GPU resources. */
        dispose() {
            if (disposed) return;
            disposed = true;
            if (installLightsNode && renderer.lighting?.createNode === installedCreateNode) {
                renderer.lighting.createNode = prevCreateNode;
            }
            try {
                gi.dispose();
            } finally {
                // Already-compiled materials retain the old lights graph. Invalidate
                // them after restoring the factory so a later render cannot touch
                // the disposed field, including materials that are currently hidden.
                markMaterialsDirty();
            }
        },
    };
}
