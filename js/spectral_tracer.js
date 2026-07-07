// spectral_tracer.js — WebGPU spectral path tracer controller.
//
// Drop-in replacement for createPathTracingController (the old WebGL
// three-gpu-pathtracer wrapper). Exposes the SAME public surface so the
// existing index.html touchpoints (render-mode branch, env binding, live
// rebuild scheduling, settings/bridge, powershot capture) keep working.
//
// Runs on the SAME WebGPURenderer as a render-loop branch: a compute kernel
// traces 1 sample/pixel/frame into an XYZ accumulation buffer, then a
// fullscreen QuadMesh blits XYZ→sRGB to the canvas (bypassing post-FX, as the
// old PT mode did). Scene/BVH build is CPU-side and debounced; accumulation
// resets on camera move or scene-dirty.

import * as THREE from 'three/webgpu';
import { buildSpectralScene } from './spectral_scene.js';
import { buildKernels } from './spectral_kernel.js';
import { decodeSpectralLut, SPECTRAL_LUT_RES } from './srgb_lut.js';

// RGB→reflectance LUT as 3 Data3DTextures (one per argmax slab), built once and
// shared across scene rebuilds. The packed module stores depth = slab*res+zi,
// so slab s occupies a contiguous res³ RGBA-float slice.
let spectralLutTextures = null;
function getSpectralLutTextures() {
    if (spectralLutTextures) return spectralLutTextures;
    try {
        const res = SPECTRAL_LUT_RES;
        const all = decodeSpectralLut(); // Float32Array, RGBA, 3 slabs packed in depth
        const sliceLen = res * res * res * 4;
        const out = [];
        for (let s = 0; s < 3; s++) {
            const slice = all.slice(s * sliceLen, (s + 1) * sliceLen);
            const tex = new THREE.Data3DTexture(slice, res, res, res);
            tex.format = THREE.RGBAFormat;
            tex.type = THREE.FloatType;
            tex.minFilter = THREE.LinearFilter;
            tex.magFilter = THREE.LinearFilter;
            tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;
            tex.colorSpace = THREE.NoColorSpace; // raw coefficients, not colour
            tex.needsUpdate = true;
            out.push(tex);
        }
        spectralLutTextures = out;
    } catch (e) {
        spectralLutTextures = null; // kernel falls back to the cheap 3-bin upsample
    }
    return spectralLutTextures;
}

const SCENE_REBUILD_COALESCE_MS = 25;
const SCENE_REBUILD_MIN_INTERVAL_MS = 50;
const SCENE_REBUILD_RETRY_MS = 1000;
const CAMERA_MATRIX_EPSILON = 1e-6;
const DEFAULT_SAMPLES_PER_FRAME = 64;
const MIN_SAMPLES_PER_FRAME = 1;
const MAX_SAMPLES_PER_FRAME = 512;
// Live IPR dispatches exactly ONE sample per presented frame so the viewport
// stays maximally interactive and refines progressively; the full
// samplesPerFrame batch is reserved for capture/render-to-image, where
// wall-clock matters more than framerate. (samplesPerFrame is GPU-bound:
// batching >1/frame live just stalls each present without converging faster.)
const LIVE_SAMPLES_PER_FRAME = 1;
const DEFAULT_GI_CLAMP = 8.0;
const MIN_GI_CLAMP = 1.0;
const MAX_GI_CLAMP = 1000.0;
const DEFAULT_SAMPLE_LIMIT = 0;     // 0 = unlimited (keep accumulating forever)
const MAX_SAMPLE_LIMIT = 100000;
const MAX_TRIANGLES = 4_000_000;

function normalizeSamplesPerFrame(value) {
    const n = Math.round(Number(value));
    if (!Number.isFinite(n)) return DEFAULT_SAMPLES_PER_FRAME;
    return Math.max(MIN_SAMPLES_PER_FRAME, Math.min(MAX_SAMPLES_PER_FRAME, n));
}
function normalizeGIClamp(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return DEFAULT_GI_CLAMP;
    return Math.max(MIN_GI_CLAMP, Math.min(MAX_GI_CLAMP, n));
}
function normalizeSampleLimit(value) {
    const n = Math.round(Number(value));
    if (!Number.isFinite(n) || n <= 0) return DEFAULT_SAMPLE_LIMIT;
    return Math.min(MAX_SAMPLE_LIMIT, n);
}
function matrixChanged(matrix, cache, epsilon = CAMERA_MATRIX_EPSILON) {
    const e = matrix?.elements;
    if (!e) return false;
    if (!cache.initialized) { cache.values.set(e); cache.initialized = true; return true; }
    for (let i = 0; i < 16; i++) {
        if (Math.abs(cache.values[i] - e[i]) > epsilon) { cache.values.set(e); return true; }
    }
    return false;
}
function cameraLayersMask(camera) {
    return Number.isFinite(camera?.layers?.mask) ? camera.layers.mask : 0;
}

export function createSpectralTracer({
    renderer,
    scene,
    camera,
    enabled = false,
    onStatus = () => {},
    onError = () => {},
    settings = {},
} = {}) {
    let activeCamera = camera;
    let disposed = false;
    let started = false;

    let sceneDirty = true;
    let sceneDirtyAt = 0;
    let hasSceneBuilt = false;
    let lastSceneRebuildAt = -Infinity;
    let nextRebuildRetryAt = 0;
    let lastRebuildErrorKey = '';
    let renderedSamples = 0;
    let frameSeed = 1;
    let captureMode = false;
    let paused = false;
    let warnedUnsupported = false;

    let samplesPerFrame = normalizeSamplesPerFrame(settings.samplesPerFrame);
    let giClamp = normalizeGIClamp(settings.giClamp);
    let sampleLimit = normalizeSampleLimit(settings.sampleLimit);
    let convergedNotified = false;
    let freezeSync = settings.freezeSync === true;
    // Camera post: thin-lens DOF + where tone mapping happens.
    let dofEnabled = false;
    let dofFocusDistance = 5;
    let dofApertureRadius = 0;
    let toneMapInBlit = true; // false → emit linear HDR for an external post stack
    // 'visible' (XYZ→sRGB, default) or 'nv' (true photocathode flux, linear).
    // NV mode is meant to feed the image-intensifier model: point nvTarget at
    // a HalfFloat RenderTarget and hand its .texture to
    // powershotInfrared.setInputMode('nir') + renderTexture(...). With no
    // nvTarget the raw flux is blitted straight to the canvas (debug view).
    let renderMode = 'visible';
    let nvTarget = null;

    const lastCameraWorld = { initialized: false, values: new Float64Array(16) };
    const lastCameraProj = { initialized: false, values: new Float64Array(16) };
    // Last env intensity/rotation pushed to the kernel — so a live HDRI
    // intensity/rotation tweak (no scene rebuild) resets accumulation.
    let lastEnvIntensity = NaN;
    let lastEnvRotation = NaN;
    let lastCameraLayersMask = NaN;

    // GPU resources (rebuilt on scene change)
    let gpu = null; // { buffers, kernels, quad, width, height, env }

    function isEnabled() { return enabled === true && !disposed; }
    function isStarted() { return started === true && isEnabled(); }
    function isSupported() { return renderer?.backend?.isWebGPUBackend === true; }

    function resetCameraKeys() {
        lastCameraWorld.initialized = false;
        lastCameraProj.initialized = false;
    }

    function almostEqualDof(a, b, absEpsilon = 1e-5, relEpsilon = 1e-5) {
        const scale = Math.max(1, Math.abs(a), Math.abs(b));
        return Math.abs(a - b) <= absEpsilon + relEpsilon * scale;
    }

    function requestSceneRebuild({ immediate = false } = {}) {
        const wasDirty = sceneDirty;
        sceneDirty = true;
        const now = performance.now();
        if (immediate) sceneDirtyAt = now;
        else if (!wasDirty || sceneDirtyAt <= now) sceneDirtyAt = now + SCENE_REBUILD_COALESCE_MS;
    }

    function shouldRebuildSceneNow(now = performance.now()) {
        if (!sceneDirty) return false;
        if (freezeSync && hasSceneBuilt) return false;
        if (now < nextRebuildRetryAt) return false;
        if (now < sceneDirtyAt) return false;
        if (Number.isFinite(lastSceneRebuildAt) && now - lastSceneRebuildAt < SCENE_REBUILD_MIN_INTERVAL_MS) {
            sceneDirtyAt = lastSceneRebuildAt + SCENE_REBUILD_MIN_INTERVAL_MS;
            return false;
        }
        return true;
    }

    function preload() { return Promise.resolve(); }

    function start() {
        if (!isEnabled()) return false;
        if (started) return true;
        started = true;
        requestSceneRebuild({ immediate: true });
        return true;
    }

    function rendererSize() {
        const v = new THREE.Vector2();
        renderer.getDrawingBufferSize(v);
        return { width: Math.max(1, Math.floor(v.x)), height: Math.max(1, Math.floor(v.y)) };
    }

    function disposeGPU() {
        if (!gpu) return;
        for (const key of ['bvhNodes', 'triIndex', 'vertexData', 'triMaterial', 'materials', 'lights', 'accum']) {
            gpu.buffers[key]?.dispose?.();
        }
        // Per-scene PBR map array textures (the shared spectral LUT is NOT
        // disposed here — it persists across rebuilds).
        if (gpu.maps) for (const t of Object.values(gpu.maps)) t?.dispose?.();
        gpu.quad?.material?.dispose?.();
        gpu.quad?.geometry?.dispose?.();
        gpu = null;
    }

    function makeStorage(array) {
        return new THREE.StorageBufferAttribute(array, 1);
    }

    let sceneBuildInFlight = false;
    function rebuildScene() {
        // buildSpectralScene is ASYNC (it time-slices PBR map extraction over
        // rAF). Kick the build and keep presenting the previous scene (or a
        // clear frame) until the new buffers swap in. Never treat the returned
        // PROMISE as the built object: every field reads undefined, makeStorage
        // gets undefined arrays, and the zero-size storage buffers cascade into
        // "Invalid BindGroup bindGroup_object" on every compute submit.
        if (sceneBuildInFlight) return hasSceneBuilt;
        sceneBuildInFlight = true;
        (async () => {
        try {
            const built = await buildSpectralScene({ THREE, scene, camera: activeCamera, maxTriangles: MAX_TRIANGLES });
            if (!built) {
                // Empty scene — nothing to trace. Treat as built so we just clear.
                disposeGPU();
                hasSceneBuilt = false;
                sceneDirty = false;
                sceneDirtyAt = 0;
                lastCameraLayersMask = cameraLayersMask(activeCamera);
                return;
            }
            if (built.error) {
                onStatus(`max.js - Path tracer: ${built.error}`);
                sceneDirty = false; // don't spin; surface the cap and stop
                hasSceneBuilt = false;
                return;
            }

            disposeGPU();
            const { width, height } = rendererSize();
            const buffers = {
                bvhNodes: makeStorage(built.bvhNodes),
                triIndex: makeStorage(built.triIndex),
                vertexData: makeStorage(built.vertexData),
                triMaterial: makeStorage(built.triMaterial),
                materials: makeStorage(built.materials),
                lights: makeStorage(built.lights),
                accum: makeStorage(new Float32Array(width * height * 4)),
                lightCount: built.lightCount,
                nodeCount: built.nodeCount,
                // two-level BVH: instance table + TLAS live in the materials tail
                tlasNodeCount: built.tlasNodeCount,
                instBase: built.instBase,
                tlasBase: built.tlasBase,
            };
            const kernels = buildKernels({
                THREE, buffers, env: built.env,
                lut: getSpectralLutTextures(), lutRes: SPECTRAL_LUT_RES,
                maps: built.maps, width, height,
                mode: renderMode,
            });
            const quad = new THREE.QuadMesh(kernels.blitMaterial);
            gpu = { buffers, kernels, quad, width, height, env: built.env, maps: built.maps };

            applyUniformSettings();
            updateCameraUniforms(true);
            updateEnvUniforms(true);
            renderedSamples = 0;

            sceneDirty = false;
            hasSceneBuilt = true;
            sceneDirtyAt = 0;
            lastCameraLayersMask = cameraLayersMask(activeCamera);
            lastSceneRebuildAt = performance.now();
            nextRebuildRetryAt = 0;
            lastRebuildErrorKey = '';
        } catch (error) {
            sceneDirty = true;
            const key = error?.stack || error?.message || String(error);
            nextRebuildRetryAt = performance.now() + SCENE_REBUILD_RETRY_MS;
            if (key !== lastRebuildErrorKey) { lastRebuildErrorKey = key; onError(error); }
        } finally {
            sceneBuildInFlight = false;
        }
        })();
        return hasSceneBuilt;
    }

    function applyUniformSettings() {
        if (!gpu) return;
        const u = gpu.kernels.uniforms;
        u.radianceClamp.value = giClamp;
        u.bounceCap.value = captureMode ? 6 : 4;
        u.apertureRadius.value = dofEnabled ? Math.max(0, dofApertureRadius) : 0;
        u.focusDistance.value = Math.max(0.01, dofFocusDistance);
        u.toneMapEnabled.value = toneMapInBlit ? 1 : 0;
    }

    // Push the live scene environment intensity/rotation into the kernel so the
    // path-traced env matches the rasterized HDRI 1:1. The env *texture* itself
    // is baked at build time (a swap triggers a rebuild); intensity and rotation
    // are uniforms and can change without one (sliders), so we sync them here.
    function updateEnvUniforms(force = false) {
        if (!gpu) return false;
        const u = gpu.kernels.uniforms;
        const intensity = Number.isFinite(scene?.environmentIntensity) ? scene.environmentIntensity : 1;
        // three's WebGPU env node samples with sampleDir = Rᵀ(θ)·worldDir, where
        // Rᵀ = transpose(makeRotationFromEuler(scene.environmentRotation)) and
        // θ = environmentRotation.y (three.webgpu.js materialEnvRotation, applied
        // as materialEnvRotation.mul(dir)). The kernel's envAtLambda rotation is
        // exactly Rᵀ(t)·dir, so the uniform is +θ (un-negated) to rotate the
        // path-traced env the same way as the rasterized HDRI.
        const rotation = Number.isFinite(scene?.environmentRotation?.y) ? scene.environmentRotation.y : 0;
        if (!force && intensity === lastEnvIntensity && rotation === lastEnvRotation) return false;
        lastEnvIntensity = intensity;
        lastEnvRotation = rotation;
        u.envIntensity.value = intensity;
        u.envRotation.value = rotation;
        return true;
    }

    function updateCameraUniforms(force = false) {
        if (!gpu || !activeCamera) return false;
        activeCamera.updateMatrixWorld();
        const worldChanged = matrixChanged(activeCamera.matrixWorld, lastCameraWorld);
        const projChanged = matrixChanged(activeCamera.projectionMatrix, lastCameraProj);
        if (!force && !worldChanged && !projChanged) return false;
        const u = gpu.kernels.uniforms;
        u.camWorld.value.copy(activeCamera.matrixWorld);
        u.camProjInv.value.copy(activeCamera.projectionMatrixInverse);
        u.camPos.value.setFromMatrixPosition(activeCamera.matrixWorld);
        return true;
    }

    function resetAccumulation() {
        if (!gpu) return;
        renderedSamples = 0;
        convergedNotified = false;
        try { renderer.compute(gpu.kernels.clearKernel); } catch (e) { onError(e); }
    }

    function clearFrame() {
        try { renderer.clear(); } catch {}
        return true;
    }

    // Present the accumulated image: to the canvas, or — NV mode with a flux
    // target bound — into that RenderTarget for the intensifier stage.
    function presentQuad() {
        if (renderMode === 'nv' && nvTarget) {
            const prev = renderer.getRenderTarget?.() ?? null;
            try {
                renderer.setRenderTarget(nvTarget);
                gpu.quad.render(renderer);
            } finally {
                renderer.setRenderTarget(prev);
            }
        } else {
            gpu.quad.render(renderer);
        }
    }

    function ensureSize() {
        if (!gpu) return;
        const { width, height } = rendererSize();
        if (width !== gpu.width || height !== gpu.height) requestSceneRebuild({ immediate: true });
    }

    function render() {
        if (!isStarted()) return false;
        if (!isSupported()) {
            if (!warnedUnsupported) { warnedUnsupported = true; onStatus('max.js - Path tracer requires the WebGPU backend'); }
            return false;
        }

        // Paused: dispatch NO compute and ignore camera/scene changes — just
        // re-present the accumulated image. The 64 trace dispatches/frame are
        // what starve the GPU/compositor, so this frees the UI panels while the
        // last render stays frozen on screen. Resume picks up where it left off.
        if (paused) {
            if (!gpu) return clearFrame();
            try { presentQuad(); } catch (error) { onError(error); }
            return true;
        }

        const currentCameraLayersMask = cameraLayersMask(activeCamera);
        if (hasSceneBuilt && currentCameraLayersMask !== lastCameraLayersMask) {
            requestSceneRebuild({ immediate: true });
        }

        if (sceneDirty) {
            if (shouldRebuildSceneNow()) {
                if (!rebuildScene()) return hasSceneBuilt || clearFrame();
            } else if (!hasSceneBuilt) {
                return clearFrame();
            }
        }
        if (!gpu) return clearFrame();

        ensureSize();
        if (!gpu) return clearFrame();

        if (updateCameraUniforms()) resetAccumulation();
        if (updateEnvUniforms()) resetAccumulation();

        // Sample limit (converge-and-stop): once the target is reached, dispatch
        // no more compute — just hold the converged frame. Frees the GPU exactly
        // like a pause, but automatic. Capture/render-to-image ignores it (it
        // drives its own sample target). A camera move resets and re-traces.
        if (!captureMode && sampleLimit > 0 && renderedSamples >= sampleLimit) {
            if (!convergedNotified) {
                convergedNotified = true;
                onStatus(`max.js - Path tracer converged (${renderedSamples}/${sampleLimit} samples)`);
            }
            try { presentQuad(); } catch (error) { onError(error); }
            return true;
        }

        try {
            const u = gpu.kernels.uniforms;
            // Capture/render-to-image uses the full batch; live IPR caps to a
            // small batch so the panel stays responsive (see LIVE_SAMPLES_PER_FRAME).
            const count = captureMode ? samplesPerFrame : LIVE_SAMPLES_PER_FRAME;
            for (let i = 0; i < count; i++) {
                frameSeed = (frameSeed + 1) >>> 0;
                u.frameSeed.value = frameSeed;
                renderer.compute(gpu.kernels.traceKernel);
                renderedSamples += 1;
            }
            u.sampleCount.value = Math.max(1, renderedSamples);
            // Exposure follows the viewer's tone-mapping exposure (display-time
            // multiply in the blit — no accumulation reset needed).
            u.exposure.value = Number.isFinite(renderer.toneMappingExposure) ? renderer.toneMappingExposure : 1;
            presentQuad();
            return true;
        } catch (error) {
            onError(error);
            return false;
        }
    }

    function markSceneDirty() { requestSceneRebuild(); }
    // Warm the scene/BVH build ahead of time WITHOUT presenting a frame, so a
    // later switch into the path-traced view swaps in with zero rebuild lag.
    // Safe to call once at load while another path (e.g. DDGI) owns the canvas:
    // it kicks the async buildSpectralScene and returns immediately. Returns the
    // current built state (false until the async build lands).
    function prebuild() {
        if (disposed || !isSupported()) return false;
        if (hasSceneBuilt || sceneBuildInFlight) return hasSceneBuilt;
        return rebuildScene();
    }
    function isSceneBuilt() { return hasSceneBuilt === true; }
    function getSampleCount() { return renderedSamples; }
    function isPaused() { return paused === true; }
    function setPaused(next) {
        const v = next === true;
        if (paused === v) return false;
        paused = v;
        // On resume, re-check the camera so a move made while paused triggers a
        // clean restart instead of accumulating onto a stale viewpoint.
        if (!paused) resetCameraKeys();
        onStatus(`max.js - Path tracer ${paused ? 'paused' : 'resumed'}`);
        return true;
    }
    function togglePaused() { setPaused(!paused); return paused; }
    function isCaptureReady(minSamples = 1) {
        return isStarted() && hasSceneBuilt && renderedSamples >= Math.max(1, minSamples | 0);
    }
    function setCaptureMode(next) {
        const v = next === true;
        if (captureMode === v) return false;
        captureMode = v;
        applyUniformSettings();
        resetAccumulation();
        return true;
    }
    function setOptions(options = {}) {
        let changed = false;
        let imageChanged = false;
        if (options.samplesPerFrame != null) {
            const n = normalizeSamplesPerFrame(options.samplesPerFrame);
            changed = changed || n !== samplesPerFrame; samplesPerFrame = n;
        }
        if (options.giClamp != null) {
            const n = normalizeGIClamp(options.giClamp);
            if (n !== giClamp) imageChanged = true;
            changed = changed || n !== giClamp; giClamp = n;
        }
        if (options.freezeSync != null) {
            const n = options.freezeSync === true;
            changed = changed || n !== freezeSync; freezeSync = n;
        }
        if (options.sampleLimit != null) {
            const n = normalizeSampleLimit(options.sampleLimit);
            if (n !== sampleLimit) {
                changed = true;
                sampleLimit = n;
                // raising the cap should resume tracing; re-arm the converged note
                convergedNotified = false;
            }
        }
        if (imageChanged) { applyUniformSettings(); resetAccumulation(); }
        return changed;
    }
    function getSettings() { return { samplesPerFrame, giClamp, freezeSync, sampleLimit }; }
    function setDOF(opts = {}) {
        let changed = false;
        if (typeof opts.enabled === 'boolean' && opts.enabled !== dofEnabled) { dofEnabled = opts.enabled; changed = true; }
        if (Number.isFinite(opts.focusDistance)) {
            const nextFocusDistance = Math.max(0.01, opts.focusDistance);
            if (!almostEqualDof(nextFocusDistance, dofFocusDistance, 1e-4, 1e-5)) {
                dofFocusDistance = nextFocusDistance;
                changed = true;
            }
        }
        if (Number.isFinite(opts.apertureRadius)) {
            const nextApertureRadius = Math.max(0, opts.apertureRadius);
            if (!almostEqualDof(nextApertureRadius, dofApertureRadius, 1e-6, 1e-4)) {
                dofApertureRadius = nextApertureRadius;
                changed = true;
            }
        }
        if (changed) { applyUniformSettings(); resetAccumulation(); }
        return changed;
    }
    function setToneMapInBlit(next) {
        const v = next !== false;
        if (toneMapInBlit === v) return false;
        toneMapInBlit = v;
        applyUniformSettings();
        return true;
    }
    function isSyncFrozen() { return freezeSync === true; }
    // 'visible' | 'nv'. The λ domain, sampling scheme and accumulation channel
    // are baked into the kernels, so a mode change rebuilds them (and restarts
    // accumulation) — cheap next to the BVH build it rides on.
    function setRenderMode(next) {
        const v = next === 'nv' ? 'nv' : 'visible';
        if (renderMode === v) return false;
        renderMode = v;
        requestSceneRebuild({ immediate: true });
        onStatus(`max.js - Path tracer mode: ${v === 'nv' ? 'night-vision (photocathode flux)' : 'visible'}`);
        return true;
    }
    function getRenderMode() { return renderMode; }
    // RenderTarget the NV flux is presented into (null → canvas debug view).
    function setNvTarget(target) {
        nvTarget = target || null;
    }
    function setCamera(nextCamera) {
        if (!nextCamera) return;
        const previousCameraLayersMask = cameraLayersMask(activeCamera);
        activeCamera = nextCamera;
        resetCameraKeys();
        if (cameraLayersMask(activeCamera) !== previousCameraLayersMask) requestSceneRebuild({ immediate: true });
        if (updateCameraUniforms(true)) resetAccumulation();
    }
    function dispose() {
        disposed = true;
        disposeGPU();
        hasSceneBuilt = false;
        sceneDirty = false;
    }

    if (typeof window !== 'undefined' && isEnabled()) {
        window.addEventListener('pagehide', dispose, { once: true });
    }

    return {
        isEnabled,
        isStarted,
        preload,
        clearFrame,
        start,
        isSupported,
        render,
        markSceneDirty,
        prebuild,
        isSceneBuilt,
        getSampleCount,
        isPaused,
        setPaused,
        togglePaused,
        isCaptureReady,
        setCaptureMode,
        setOptions,
        getSettings,
        setDOF,
        setToneMapInBlit,
        setRenderMode,
        getRenderMode,
        setNvTarget,
        isSyncFrozen,
        setCamera,
        dispose,
    };
}
