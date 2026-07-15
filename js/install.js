// install.js — one-call setup for SPEEDBALL GI.
//
// Collapses the wiring that the probe field needs into a single call plus one
// update() per frame:
//   1. installs the lights-node factory so GI folds into every PBR material,
//   2. wires the post-rebuild material-dirty pass (the silent-"GI-missing" footgun),
//   3. idle-gates the solve off the camera transform (works with any controls, or none).
//
// Advanced users can still import createProbeField / giLights directly and wire it
// by hand; this is the batteries-included path.

import * as THREE from 'three/webgpu';
import { createProbeField } from './gi_probes.js';
import { giLights, setNirDirectSensing } from './gi_lights_node.js';

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
            if (m.visible === false) return;
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
 *   gi.update();            // idle-gated solve
 *
 * @param {object} opts
 * @param {THREE.WebGPURenderer} opts.renderer
 * @param {THREE.Scene} opts.scene
 * @param {THREE.Camera} [opts.camera]         used for idle detection; pass per-frame to update() to override
 * @param {boolean} [opts.enabled=true]
 * @param {number}  [opts.intensity=10]        canonical demo tuning (Sponza)
 * @param {number}  [opts.divisions=16]        probes along the longest grid axis
 * @param {number}  [opts.hysteresis=0.9]      temporal stability (higher = steadier / slower)
 * @param {boolean} [opts.roughReflections=false] reuse DDGI rays for glossy + rough local reflections; opt-in keeps legacy path allocation-free
 * @param {number}  [opts.reflectionIntensity=1] local-vs-environment reflection coverage blend, 0..1
 * @param {object}  [opts.lights]              max light counts for the batched lights node
 * @param {boolean} [opts.installLightsNode=true]  set false if you install your own GI-aware lights node
 * @param {boolean} [opts.prepareMaterials=false]  run prepareMaterialsForGI(scene) on install
 * @returns {object} the probe field (all setters) augmented with update(), markInteraction(),
 *                   markMaterialsDirty(), and a dispose() that also restores the lights factory.
 */
export function installSpeedballGI({
    renderer,
    scene,
    camera,
    enabled = true,
    intensity = 10,
    divisions = 16,
    hysteresis = 0.9,
    roughReflections = false,
    reflectionIntensity = 1,
    lights = { maxDirectionalLights: 4, maxPointLights: 16, maxSpotLights: 16, maxHemisphereLights: 2 },
    installLightsNode = true,
    prepareMaterials = false,
} = {}) {
    if (!renderer || !scene) throw new Error('installSpeedballGI: { renderer, scene } are required.');

    // 1. Lights factory — one line folds GI into every PBR material (no per-material wiring).
    let prevCreateNode = null;
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
        renderer.lighting.createNode = (lightList = []) => giLights(lights).setLights(lightList);
    }

    if (prepareMaterials) prepareMaterialsForGI(scene);

    // 2. Probe field, with the material-dirty pass wired as onRebuilt (footgun handled).
    const markMaterialsDirty = makeMaterialDirtier(scene);
    const gi = createProbeField({
        renderer,
        scene,
        intensity,
        hysteresis,
        divisions,
        roughReflections,
        reflectionIntensity,
        onRebuilt: markMaterialsDirty,
    });
    if (enabled) gi.setEnabled(true);

    // 3. Idle tracking off the camera transform → works with any controls (or none):
    //    hold the world-space field static while the view moves, converge when it rests.
    let lastInteraction = _now();
    const _pos = new THREE.Vector3(Infinity, 0, 0);
    const _quat = new THREE.Quaternion(2, 0, 0, 0);
    const cameraMoved = (cam) => {
        if (!cam) return false;
        const moved = cam.position.distanceToSquared(_pos) > 1e-7
            || Math.abs(cam.quaternion.dot(_quat)) < 0.99999995;
        _pos.copy(cam.position); _quat.copy(cam.quaternion);
        return moved;
    };

    return {
        ...gi,          // createProbeField returns plain closures (no `this`) — safe to spread
        gi,             // the raw field, if you want the exact object
        node: gi.node,

        /** Call once per frame. Idle-gated: solves only when the view is at rest. */
        update({ camera: cam = camera, playing = false } = {}) {
            const now = _now();
            if (cameraMoved(cam)) lastInteraction = now;
            gi.tick({ idleMs: now - lastInteraction, playing });
        },

        /** Treat "now" as an interaction, deferring the next solve (e.g. after a big edit). */
        markInteraction() { lastInteraction = _now(); },

        /**
         * NIR band sensing (white-phosphor NV filter on/off). One switch for BOTH terms
         * of emitter-class-'ir' lights: the probes' NEE gate (GI) and the direct raster
         * gate (gi_lights_node). Uniform writes only — no recompile, no scene mutation;
         * light.color stays black, so the light never exists in the visible band.
         */
        setNirSensing(on) { gi.setNirSensing(on); setNirDirectSensing(on); },

        /** Recompile lit materials so GI folds in — call if you add meshes after install. */
        markMaterialsDirty,

        /** Full teardown: restore the previous lights factory and free GPU resources. */
        dispose() {
            if (installLightsNode && renderer.lighting) renderer.lighting.createNode = prevCreateNode;
            gi.dispose();
        },
    };
}
