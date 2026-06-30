// gi_irradiance_volume.js - WebGPU compute local-bounce probe volume.
//
// The GI solve is GPU-owned: CPU code only packs scene surface/light data into
// storage buffers. Direct lighting, surface bounce gather, temporal blend, and
// material shading all stay on the WebGPU/TSL path. No cube-camera bake, no CPU
// raycast/light solve, no readback.

import * as THREE from 'three';
import { LightingNode } from 'three/webgpu';
import {
    Fn, If, Loop, Return,
    instanceIndex, storage, uniform,
    float, uint, vec3, vec4,
    max as tslMax, min as tslMin, mix, smoothstep,
    positionWorld, normalWorld,
} from 'three/tsl';

let _instance = null;

const SAMPLE_STRIDE = 12; // pos.xyz, normal.xyz, albedo.rgb, flags/pad.xyz
const LIGHT_STRIDE = 16;  // type, pos.xyz, dir.xyz, color.rgb, distance, decay, coneCos, penumbraCos, pad
const PROBE_LOBES = 6;    // +X, -X, +Y, -Y, +Z, -Z receiver-normal lobes
const PROBE_STRIDE = PROBE_LOBES * 4; // per lobe: rgb, confidence
const SURFEL_STRIDE = 4;  // rgb, confidence
const TARGET_PROBES_LONG_AXIS = 9;
const MAX_PROBES_PER_AXIS = 14;
const MAX_SURFACE_SAMPLES = 1024;
const MAX_LIGHTS = 64;
const VISIBILITY_SAMPLE_STRIDE = 16;
const SURFEL_SLICE_SIZE = 128;
const PROBE_SLICE_SIZE = 96;

export class GiVolumeNode extends LightingNode {
    static get type() { return 'GiVolumeNode'; }

    constructor() {
        super();
        this._probeNode = null;
        this._sampleNode = null;
        this._surfelNode = null;
        this._probeFloatCount = 0;
        this._sampleFloatCount = 0;
        this._surfelFloatCount = 0;
        this._sampleCount = 0;
        this._generation = 0;
        this._enabled = false;
        this._hasData = false;

        this.boundsMinNode = uniform(new THREE.Vector3());
        this.invSizeNode = uniform(new THREE.Vector3(1, 1, 1));
        this.resNode = uniform(new THREE.Vector3(2, 2, 2));
        this.intensityNode = uniform(0.38);
        this.normalBiasNode = uniform(0.02);
        this.edgeFadeNode = uniform(0.06);
        this.intensity = 0.38;
    }

    get active() {
        return this._enabled === true && this._hasData === true && this.intensity > 0;
    }

    get cacheToken() {
        return `gi-webgpu-probes:${this._generation}`;
    }

    setEnabled(on) { this._enabled = on === true; }

    setIntensity(v) {
        this.intensity = Number.isFinite(v) ? Math.max(0, v) : 0;
        this.intensityNode.value = this.intensity;
    }

    setNormalBias(v) {
        if (Number.isFinite(v)) this.normalBiasNode.value = Math.max(0, v);
    }

    setEdgeFade(v) {
        if (Number.isFinite(v)) this.edgeFadeNode.value = THREE.MathUtils.clamp(v, 1e-4, 0.5);
    }

    setSurfelBuffers({ sampleAttr, sampleFloatCount, surfelAttr, surfelFloatCount, sampleCount } = {}) {
        this._sampleNode = sampleAttr ? storage(sampleAttr, 'float', sampleFloatCount).toReadOnly() : null;
        this._surfelNode = surfelAttr ? storage(surfelAttr, 'float', surfelFloatCount).toReadOnly() : null;
        this._sampleFloatCount = sampleFloatCount || 0;
        this._surfelFloatCount = surfelFloatCount || 0;
        this._sampleCount = sampleCount || 0;
        this._hasData = !!(this._probeNode && this._probeFloatCount > 0);
        this._generation++;
    }

    setProbeBuffer(attribute, floatCount) {
        this._probeNode = attribute ? storage(attribute, 'float', floatCount).toReadOnly() : null;
        this._probeFloatCount = floatCount || 0;
        this._hasData = !!(this._probeNode && this._probeFloatCount > 0);
        this._generation++;
    }

    setVolume(boundsMin, boundsSize, resolution) {
        this.boundsMinNode.value.copy(boundsMin);
        this.invSizeNode.value.set(
            1 / Math.max(1e-6, boundsSize.x),
            1 / Math.max(1e-6, boundsSize.y),
            1 / Math.max(1e-6, boundsSize.z),
        );
        this.resNode.value.copy(resolution);
    }

    setup(builder) {
        if (!this._hasData || !this._probeNode) return;

        const samplePos = positionWorld.add(normalWorld.mul(this.normalBiasNode));
        const localRaw = samplePos.sub(this.boundsMinNode).mul(this.invSizeNode);
        const local = localRaw.clamp(0.0, 1.0);

        const e = this.edgeFadeNode;
        const fadeLo = smoothstep(float(0.0), e, localRaw);
        const fadeHi = smoothstep(float(0.0), e, vec3(1.0).sub(localRaw));
        const fade3 = fadeLo.mul(fadeHi);
        const fade = tslMin(tslMin(fade3.x, fade3.y), fade3.z).clamp(0.0, 1.0);

        const grid = local.mul(this.resNode.sub(1.0));
        const probeF = grid.add(0.5).floor().clamp(vec3(0.0), this.resNode.sub(1.0));
        const rx = this.resNode.x.toUint();
        const ry = this.resNode.y.toUint();
        const id = probeF.x.toUint()
            .add(probeF.y.toUint().mul(rx))
            .add(probeF.z.toUint().mul(rx).mul(ry));

        const loadProbeLobe = (lobe) => {
            const base = id.mul(uint(PROBE_STRIDE)).add(uint(lobe * 4));
            return vec4(
                this._probeNode.element(base),
                this._probeNode.element(base.add(uint(1))),
                this._probeNode.element(base.add(uint(2))),
                this._probeNode.element(base.add(uint(3))),
            );
        };

        const n = normalWorld.normalize();
        const wxp = tslMax(n.x, 0.0);
        const wxn = tslMax(n.x.negate(), 0.0);
        const wyp = tslMax(n.y, 0.0);
        const wyn = tslMax(n.y.negate(), 0.0);
        const wzp = tslMax(n.z, 0.0);
        const wzn = tslMax(n.z.negate(), 0.0);
        const wsum = tslMax(wxp.add(wxn).add(wyp).add(wyn).add(wzp).add(wzn), float(1e-5));
        const gi4 = loadProbeLobe(0).mul(wxp)
            .add(loadProbeLobe(1).mul(wxn))
            .add(loadProbeLobe(2).mul(wyp))
            .add(loadProbeLobe(3).mul(wyn))
            .add(loadProbeLobe(4).mul(wzp))
            .add(loadProbeLobe(5).mul(wzn))
            .div(wsum);
        const gi = gi4.xyz.mul(gi4.w.clamp(0.0, 1.0));

        builder.context.irradiance.addAssign(
            gi.max(vec3(0.0)).mul(this.intensityNode).mul(fade),
        );
    }
}

export function getGiVolumeNode() {
    if (!_instance) _instance = new GiVolumeNode();
    return _instance;
}

function computeResolution(size) {
    const longest = Math.max(size.x, size.y, size.z, 1e-3);
    const spacing = longest / TARGET_PROBES_LONG_AXIS;
    const axis = (s) => THREE.MathUtils.clamp(Math.round(s / spacing) + 1, 2, MAX_PROBES_PER_AXIS);
    return new THREE.Vector3(axis(size.x), axis(size.y), axis(size.z));
}

function materialAlbedo(material, target) {
    target.setRGB(1, 1, 1);
    const mat = Array.isArray(material) ? material[0] : material;
    if (mat?.color?.isColor) target.copy(mat.color);
    return target;
}

function materialAllowsGiSample(material) {
    const materials = Array.isArray(material) ? material : [material];
    for (const mat of materials) {
        if (!mat) continue;
        if (mat.visible === false) continue;
        if (mat.userData?.maxjsExcludeGI === true) continue;
        if (mat.isMeshBasicMaterial || mat.isLineBasicMaterial || mat.isLineDashedMaterial) continue;
        if (mat.transparent === true && Number.isFinite(mat.opacity) && mat.opacity < 0.75) continue;
        return true;
    }
    return false;
}

function shouldSampleObject(object) {
    if (!object?.isMesh || object.visible === false) return false;
    if (object.userData?.maxjsHelper === true) return false;
    if (object.userData?.maxjsExcludeGI === true) return false;
    if (object.userData?.maxjsExcludeFromRuntimeSnapshot === true) return false;
    if (object.userData?.volumetricBoundsBypass === true) return false;
    if (!materialAllowsGiSample(object.material)) return false;
    return object.geometry?.attributes?.position !== undefined;
}

function buildSurfaceSampleArray(scene, maxSamples, rayBias) {
    const meshes = [];
    let totalTriangles = 0;
    scene.updateMatrixWorld(true);
    scene.traverse((object) => {
        if (!shouldSampleObject(object)) return;
        const pos = object.geometry.attributes.position;
        const triCount = object.geometry.index ? Math.floor(object.geometry.index.count / 3) : Math.floor(pos.count / 3);
        if (triCount <= 0) return;
        meshes.push({ object, triCount });
        totalTriangles += triCount;
    });

    const sampleCount = Math.max(0, Math.min(maxSamples, totalTriangles > 0 ? maxSamples : 0));
    const out = new Float32Array(Math.max(1, sampleCount) * SAMPLE_STRIDE);
    if (sampleCount === 0) return { array: out, count: 0 };

    const normalMatrix = new THREE.Matrix3();
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const c = new THREE.Vector3();
    const na = new THREE.Vector3();
    const nb = new THREE.Vector3();
    const nc = new THREE.Vector3();
    const p = new THREE.Vector3();
    const n = new THREE.Vector3();
    const albedo = new THREE.Color();
    let cursor = 0;

    for (const { object, triCount } of meshes) {
        if (cursor >= sampleCount) break;
        const geom = object.geometry;
        const pos = geom.attributes.position;
        const nor = geom.attributes.normal;
        const idx = geom.index;
        const budget = Math.max(1, Math.round(maxSamples * (triCount / Math.max(1, totalTriangles))));
        const stride = Math.max(1, Math.floor(triCount / budget));
        const repsPerTri = triCount < budget ? Math.max(1, Math.ceil(budget / triCount)) : 1;
        normalMatrix.getNormalMatrix(object.matrixWorld);
        materialAlbedo(object.material, albedo);

        for (let t = 0; t < triCount && cursor < sampleCount; t += stride) {
            const i0 = idx ? idx.getX(t * 3 + 0) : t * 3 + 0;
            const i1 = idx ? idx.getX(t * 3 + 1) : t * 3 + 1;
            const i2 = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;
            a.fromBufferAttribute(pos, i0).applyMatrix4(object.matrixWorld);
            b.fromBufferAttribute(pos, i1).applyMatrix4(object.matrixWorld);
            c.fromBufferAttribute(pos, i2).applyMatrix4(object.matrixWorld);

            if (nor) {
                na.fromBufferAttribute(nor, i0);
                nb.fromBufferAttribute(nor, i1);
                nc.fromBufferAttribute(nor, i2);
                n.copy(na).add(nb).add(nc).applyMatrix3(normalMatrix).normalize();
            } else {
                n.copy(c).sub(b);
                na.copy(a).sub(b);
                n.cross(na).normalize();
            }
            if (n.lengthSq() < 1e-8) n.set(0, 1, 0);

            for (let r = 0; r < repsPerTri && cursor < sampleCount; r++) {
                let u = repsPerTri === 1 ? 1 / 3 : (0.5 + r * 0.754877666) % 1;
                let v = repsPerTri === 1 ? 1 / 3 : (0.25 + r * 0.569840296) % 1;
                if (u + v > 1) { u = 1 - u; v = 1 - v; }
                const w = 1 - u - v;
                p.set(0, 0, 0).addScaledVector(a, w).addScaledVector(b, u).addScaledVector(c, v);
                const o = cursor * SAMPLE_STRIDE;
                out[o + 0] = p.x; out[o + 1] = p.y; out[o + 2] = p.z;
                out[o + 3] = n.x; out[o + 4] = n.y; out[o + 5] = n.z;
                out[o + 6] = albedo.r; out[o + 7] = albedo.g; out[o + 8] = albedo.b;
                out[o + 9] = rayBias;
                cursor++;
            }
        }
    }

    return { array: out, count: cursor };
}

function buildLightArray(scene) {
    const out = new Float32Array(Math.max(1, MAX_LIGHTS) * LIGHT_STRIDE);
    const lightPos = new THREE.Vector3();
    const targetPos = new THREE.Vector3();
    const dir = new THREE.Vector3();
    let count = 0;

    scene.updateMatrixWorld(true);
    scene.traverse((object) => {
        if (count >= MAX_LIGHTS) return;
        if (!object?.isLight || object.visible === false || object.intensity <= 0) return;
        if (object.userData?.maxjsVisible === false) return;
        if (object.userData?.maxjsHelper === true) return;
        if (object.userData?.maxjsExcludeGI === true) return;
        if (object.userData?.volumetricBypass === true && object.userData?.maxjsIncludeGI !== true) return;
        if (object.isAmbientLight || object.isHemisphereLight || object.isRectAreaLight) return;

        const o = count * LIGHT_STRIDE;
        const contribution = Number.isFinite(object.userData?.giContrib)
            ? object.userData.giContrib
            : (Number.isFinite(object.userData?.volContrib) ? object.userData.volContrib : 1);
        if (contribution <= 0) return;
        const color = object.color?.isColor
            ? new THREE.Color(object.color.r, object.color.g, object.color.b).multiplyScalar(object.intensity)
            : new THREE.Color(object.intensity, object.intensity, object.intensity);

        object.getWorldPosition(lightPos);
        out[o + 1] = lightPos.x; out[o + 2] = lightPos.y; out[o + 3] = lightPos.z;
        out[o + 7] = color.r; out[o + 8] = color.g; out[o + 9] = color.b;
        out[o + 14] = Math.max(0, contribution);

        if (object.isDirectionalLight) {
            object.target?.getWorldPosition?.(targetPos);
            dir.copy(lightPos).sub(targetPos).normalize(); // Three lightDirection: surface -> light
            out[o + 0] = 0;
            out[o + 4] = dir.x; out[o + 5] = dir.y; out[o + 6] = dir.z;
            count++;
            return;
        }

        if (object.isPointLight) {
            out[o + 0] = 1;
            out[o + 10] = Math.max(0, object.distance || 0);
            out[o + 11] = Number.isFinite(object.decay) ? object.decay : 2;
            count++;
            return;
        }

        if (object.isSpotLight) {
            object.target?.getWorldPosition?.(targetPos);
            dir.copy(lightPos).sub(targetPos).normalize(); // matches SpotLightDataNode's direction
            out[o + 0] = 2;
            out[o + 4] = dir.x; out[o + 5] = dir.y; out[o + 6] = dir.z;
            out[o + 10] = Math.max(0, object.distance || 0);
            out[o + 11] = Number.isFinite(object.decay) ? object.decay : 2;
            out[o + 12] = Math.cos(object.angle || Math.PI / 3);
            out[o + 13] = Math.cos((object.angle || Math.PI / 3) * (1 - (object.penumbra || 0)));
            count++;
        }
    });

    return { array: out, count };
}

export function createIrradianceVolume({
    renderer,
    scene,
    intensity = 0.38,
    hysteresis = 0.08,
    radianceClamp = 7.5,
    surfaceSampleBudget = MAX_SURFACE_SAMPLES,
    surfaceRadiusScale = 2.6,
    realtime = true,
    realtimeHz = 30,
    playbackHz = 12,
    settlePasses = 3,
} = {}) {
    const node = getGiVolumeNode();
    node.setIntensity(intensity);

    const boundsMin = new THREE.Vector3();
    const boundsSize = new THREE.Vector3(1, 1, 1);
    const resolution = new THREE.Vector3(2, 2, 2);
    const stats = {
        mode: 'webgpu-compute',
        probes: 0,
        samples: 0,
        lights: 0,
        lobes: PROBE_LOBES,
        updates: 0,
        lightUpdates: 0,
        realtime: realtime === true,
        realtimeHz,
        playbackHz,
        active: false,
        baking: false,
    };

    let probeCount = 0;
    let sampleCount = 0;
    let lightCount = 0;
    let probeAttr = null;
    let sampleAttr = null;
    let surfelAttr = null;
    let lightAttr = null;
    let surfelComputeNode = null;
    let probeComputeNode = null;
    let dirtyBuffers = true;
    let dirtySolve = false;
    let dirtyLightBuffer = true;
    let dirtySurfel = true;
    let inFlight = false;
    let disposed = false;
    let lastSolveMs = 0;
    let remainingSettlePasses = 0;
    let remainingSurfelSlices = 0;
    let surfelCursor = 0;
    let surfelDispatchCount = 1;
    let probeCursor = 0;
    let probeDispatchCount = 1;
    let nativeSurfacePack = null;
    let nativeLightPack = null;
    let radClamp = Number.isFinite(radianceClamp) && radianceClamp > 0 ? radianceClamp : 0;

    const U = {
        boundsMin: uniform(new THREE.Vector3()),
        boundsSize: uniform(new THREE.Vector3(1, 1, 1)),
        radius: uniform(1.0),
        hysteresis: uniform(THREE.MathUtils.clamp(hysteresis, 0, 0.98)),
        radianceClamp: uniform(radClamp),
        bounceGain: uniform(0.22),
        surfelOffset: uniform(0, 'uint'),
        probeOffset: uniform(0, 'uint'),
    };

    function isSupported() {
        return renderer?.backend?.isWebGPUBackend === true
            && typeof renderer.computeAsync === 'function'
            && typeof THREE.StorageBufferAttribute === 'function';
    }

    function disposeBuffers() {
        probeAttr = null;
        sampleAttr = null;
        surfelAttr = null;
        lightAttr = null;
        surfelComputeNode = null;
        probeComputeNode = null;
        probeCount = 0;
        sampleCount = 0;
        lightCount = 0;
        stats.probes = 0;
        stats.samples = 0;
        stats.lights = 0;
        stats.lobes = PROBE_LOBES;
        stats.lightUpdates = 0;
        surfelCursor = 0;
        surfelDispatchCount = 1;
        probeCursor = 0;
        probeDispatchCount = 1;
        node.setSurfelBuffers();
        node.setProbeBuffer(null, 0);
    }

    function queueSurfelSweep() {
        const slices = Math.max(1, Math.ceil(Math.max(1, sampleCount) / Math.max(1, SURFEL_SLICE_SIZE)));
        remainingSurfelSlices = Math.max(remainingSurfelSlices, slices);
    }

    function queueSettleSweeps(sweeps = settlePasses) {
        const slices = Math.max(1, Math.ceil(Math.max(1, probeCount) / Math.max(1, PROBE_SLICE_SIZE)));
        remainingSettlePasses = Math.max(remainingSettlePasses, slices * Math.max(1, sweeps));
    }

    function updateVolumeState() {
        if (typeof THREE.StorageBufferAttribute !== 'function') {
            console.warn('max.js WebGPU GI unavailable: StorageBufferAttribute missing from THREE build.');
            return;
        }
        probeCount = resolution.x * resolution.y * resolution.z;
        if (!probeAttr || probeAttr.array.length !== Math.max(1, probeCount * PROBE_STRIDE)) {
            probeAttr = new THREE.StorageBufferAttribute(new Float32Array(Math.max(1, probeCount * PROBE_STRIDE)), 1);
            node.setProbeBuffer(probeAttr, probeAttr.array.length);
        }
        node.setVolume(boundsMin, boundsSize, resolution);

        const minCell = Math.min(
            boundsSize.x / Math.max(1, resolution.x - 1),
            boundsSize.y / Math.max(1, resolution.y - 1),
            boundsSize.z / Math.max(1, resolution.z - 1),
        );
        const maxCell = Math.max(
            boundsSize.x / Math.max(1, resolution.x - 1),
            boundsSize.y / Math.max(1, resolution.y - 1),
            boundsSize.z / Math.max(1, resolution.z - 1),
        );
        node.setNormalBias(Math.max(1e-4, minCell * 0.04));
        U.radius.value = Math.max(1e-4, maxCell * surfaceRadiusScale);
        stats.probes = probeCount;
    }

    function rebuildSceneBuffers() {
        let samplePack = null;
        if (nativeSurfacePack?.array && nativeSurfacePack.count > 0) {
            boundsMin.copy(nativeSurfacePack.boundsMin);
            boundsSize.copy(nativeSurfacePack.boundsSize);
            resolution.copy(computeResolution(boundsSize));
            samplePack = {
                array: nativeSurfacePack.array,
                count: nativeSurfacePack.count,
            };
        }
        updateVolumeState();
        const minCell = Math.min(
            boundsSize.x / Math.max(1, resolution.x - 1),
            boundsSize.y / Math.max(1, resolution.y - 1),
            boundsSize.z / Math.max(1, resolution.z - 1),
        );
        if (!samplePack) {
            samplePack = buildSurfaceSampleArray(scene, surfaceSampleBudget, Math.max(1e-4, minCell * 0.025));
        }
        const lightPack = nativeLightPack?.array
            ? { array: nativeLightPack.array, count: nativeLightPack.count }
            : buildLightArray(scene);
        sampleCount = samplePack.count;
        lightCount = lightPack.count;
        sampleAttr = new THREE.StorageBufferAttribute(samplePack.array, 1);
        surfelAttr = new THREE.StorageBufferAttribute(new Float32Array(Math.max(1, samplePack.count * SURFEL_STRIDE)), 1);
        lightAttr = new THREE.StorageBufferAttribute(lightPack.array, 1);
        node.setProbeBuffer(probeAttr, probeAttr.array.length);
        node.setSurfelBuffers({
            sampleAttr,
            sampleFloatCount: sampleAttr.array.length,
            surfelAttr,
            surfelFloatCount: surfelAttr.array.length,
            sampleCount: samplePack.count,
        });
        node.setVolume(boundsMin, boundsSize, resolution);
        stats.samples = samplePack.count;
        stats.lights = lightPack.count;
        stats.lobes = PROBE_LOBES;
        surfelCursor = 0;
        probeCursor = 0;
        buildComputeNodes(sampleCount, lightCount);
        dirtyBuffers = false;
        dirtyLightBuffer = false;
        dirtySurfel = true;
        dirtySolve = true;
        queueSurfelSweep();
        queueSettleSweeps();
    }

    function floatArraysNearlyEqual(a, b, eps = 1e-5) {
        if (!a || !b || a.length !== b.length) return false;
        for (let i = 0; i < a.length; i++) {
            if (Math.abs(a[i] - b[i]) > eps) return false;
        }
        return true;
    }

    function updateLightBuffer() {
        if (!lightAttr || sampleCount <= 0) {
            dirtyBuffers = true;
            return false;
        }
        const lightPack = nativeLightPack?.array
            ? { array: nativeLightPack.array, count: nativeLightPack.count }
            : buildLightArray(scene);
        if (lightPack.count !== lightCount) {
            lightCount = lightPack.count;
            lightAttr = new THREE.StorageBufferAttribute(lightPack.array, 1);
            stats.lights = lightCount;
            buildComputeNodes(sampleCount, lightCount);
            dirtyLightBuffer = false;
            stats.lightUpdates++;
            return true;
        }
        if (floatArraysNearlyEqual(lightAttr.array, lightPack.array)) {
            dirtyLightBuffer = false;
            return false;
        }
        lightAttr.array.set(lightPack.array);
        lightAttr.needsUpdate = true;
        stats.lights = lightCount;
        dirtyLightBuffer = false;
        stats.lightUpdates++;
        return true;
    }

    function buildComputeNodes(sampleCount, lightCount) {
        if (!sampleAttr || !surfelAttr || !probeAttr || sampleCount <= 0) {
            surfelComputeNode = null;
            probeComputeNode = null;
            return;
        }
        const surfels = storage(surfelAttr, 'float', surfelAttr.array.length);
        const surfelsRead = storage(surfelAttr, 'float', surfelAttr.array.length).toReadOnly();
        const probes = storage(probeAttr, 'float', probeAttr.array.length);
        const samples = storage(sampleAttr, 'float', sampleAttr.array.length).toReadOnly();
        const lights = storage(lightAttr, 'float', lightAttr.array.length).toReadOnly();
        const sampleTotal = Math.max(1, sampleCount);
        surfelDispatchCount = Math.max(1, Math.min(SURFEL_SLICE_SIZE, sampleTotal));
        const visibilitySampleCount = Math.max(1, Math.ceil(sampleCount / VISIBILITY_SAMPLE_STRIDE));
        const nx = resolution.x >>> 0;
        const ny = resolution.y >>> 0;
        const nz = resolution.z >>> 0;
        const probeTotal = Math.max(1, nx * ny * nz);
        probeDispatchCount = Math.max(1, Math.min(PROBE_SLICE_SIZE, probeTotal));
        const radiusSq = U.radius.mul(U.radius);

        const loadSampleVec3 = (base, offset) => vec3(
            samples.element(base.add(uint(offset))),
            samples.element(base.add(uint(offset + 1))),
            samples.element(base.add(uint(offset + 2))),
        );
        const loadLightVec3 = (base, offset) => vec3(
            lights.element(base.add(uint(offset))),
            lights.element(base.add(uint(offset + 1))),
            lights.element(base.add(uint(offset + 2))),
        );

        surfelComputeNode = Fn(() => {
            const si = instanceIndex.add(U.surfelOffset).toVar();
            If(si.greaterThanEqual(uint(sampleCount)), () => { Return(); });

            const sb = si.mul(uint(SAMPLE_STRIDE)).toVar();
            const sPos = loadSampleVec3(sb, 0).toVar();
            const sNormal = loadSampleVec3(sb, 3).normalize().toVar();
            const sAlbedo = loadSampleVec3(sb, 6).toVar();
            const direct = vec3(0.0).toVar();

            Loop({ start: uint(0), end: uint(lightCount), type: 'uint', condition: '<' }, ({ i: l }) => {
                const lb = l.mul(uint(LIGHT_STRIDE)).toVar();
                const type = lights.element(lb).toVar();
                const lPos = loadLightVec3(lb, 1).toVar();
                const lDir = loadLightVec3(lb, 4).normalize().toVar();
                const lColor = loadLightVec3(lb, 7).toVar();
                const lDistance = lights.element(lb.add(uint(10))).toVar();
                const lDecay = lights.element(lb.add(uint(11))).toVar();
                const coneCos = lights.element(lb.add(uint(12))).toVar();
                const penumbraCos = lights.element(lb.add(uint(13))).toVar();
                const lContrib = tslMax(lights.element(lb.add(uint(14))), 0.0).toVar();
                const lRadiance = lColor.mul(lContrib).toVar();

                If(type.lessThan(0.5), () => {
                    const ndotl = tslMax(sNormal.dot(lDir), 0.0).toVar();
                    const visibility = float(1.0).toVar();
                    If(ndotl.greaterThan(0.001), () => {
                        const occlusionRadius = U.radius.mul(0.022).toVar();
                        const occlusionRadiusSq = occlusionRadius.mul(occlusionRadius).toVar();
                        const maxOccDist = U.radius.mul(2.0).toVar();
                        Loop({ start: uint(0), end: uint(visibilitySampleCount), type: 'uint', condition: '<' }, ({ i: oi }) => {
                            const bi = oi.mul(uint(VISIBILITY_SAMPLE_STRIDE)).toVar();
                            If(bi.notEqual(si).and(bi.lessThan(uint(sampleCount))), () => {
                                const bb = bi.mul(uint(SAMPLE_STRIDE)).toVar();
                                const bPos = loadSampleVec3(bb, 0).toVar();
                                const bNormal = loadSampleVec3(bb, 3).normalize().toVar();
                                const toBlocker = bPos.sub(sPos).toVar();
                                const along = toBlocker.dot(lDir).toVar();
                                If(along.greaterThan(0.03).and(along.lessThan(maxOccDist)), () => {
                                    const distSq = toBlocker.dot(toBlocker).toVar();
                                    const perpSq = tslMax(distSq.sub(along.mul(along)), 0.0).toVar();
                                    const normalCrossing = bNormal.dot(lDir).toVar();
                                    If(perpSq.lessThan(occlusionRadiusSq)
                                        .and(normalCrossing.mul(normalCrossing).greaterThan(0.035)), () => {
                                        const occ = float(1.0).sub(smoothstep(float(0.0), occlusionRadiusSq, perpSq)).mul(0.72).toVar();
                                        visibility.mulAssign(float(1.0).sub(occ).clamp(0.18, 1.0));
                                    });
                                });
                            });
                        });
                    });
                    direct.addAssign(lRadiance.mul(ndotl).mul(visibility));
                }).ElseIf(type.lessThan(1.5), () => {
                    const lv = lPos.sub(sPos).toVar();
                    const d = lv.length().toVar();
                    If(d.greaterThan(1e-5), () => {
                        const ld = lv.div(d).toVar();
                        const ndotl = tslMax(sNormal.dot(ld), 0.0).toVar();
                        const rangeFade = float(1.0).toVar();
                        If(lDistance.greaterThan(0.0), () => {
                            const x = d.div(lDistance).clamp(0.0, 1.0).toVar();
                            rangeFade.assign(float(1.0).sub(x.mul(x).mul(x).mul(x)).clamp(0.0, 1.0));
                            rangeFade.mulAssign(rangeFade);
                        });
                        const atten = rangeFade.div(tslMax(float(1.0), d.pow(lDecay))).toVar();
                        direct.addAssign(lRadiance.mul(ndotl).mul(atten));
                    });
                }).Else(() => {
                    const lv = lPos.sub(sPos).toVar();
                    const d = lv.length().toVar();
                    If(d.greaterThan(1e-5), () => {
                        const ld = lv.div(d).toVar();
                        const ndotl = tslMax(sNormal.dot(ld), 0.0).toVar();
                        const spot = smoothstep(coneCos, penumbraCos, ld.dot(lDir)).toVar();
                        const rangeFade = float(1.0).toVar();
                        If(lDistance.greaterThan(0.0), () => {
                            const x = d.div(lDistance).clamp(0.0, 1.0).toVar();
                            rangeFade.assign(float(1.0).sub(x.mul(x).mul(x).mul(x)).clamp(0.0, 1.0));
                            rangeFade.mulAssign(rangeFade);
                        });
                        const atten = spot.mul(rangeFade).div(tslMax(float(1.0), d.pow(lDecay))).toVar();
                        direct.addAssign(lRadiance.mul(ndotl).mul(atten));
                    });
                });
            });

            const outColor = direct.mul(sAlbedo).mul(U.bounceGain).toVar();
            const confidence = float(0.0).toVar();
            const lum = outColor.dot(vec3(0.2126, 0.7152, 0.0722)).toVar();
            If(lum.greaterThan(1e-5), () => {
                If(U.radianceClamp.greaterThan(0.0).and(lum.greaterThan(U.radianceClamp)), () => {
                    outColor.mulAssign(U.radianceClamp.div(lum));
                });
                confidence.assign(1.0);
            });

            const base = si.mul(uint(SURFEL_STRIDE)).toVar();
            const prev = vec4(
                surfels.element(base),
                surfels.element(base.add(uint(1))),
                surfels.element(base.add(uint(2))),
                surfels.element(base.add(uint(3))),
            ).toVar();
            const next = vec4(outColor, confidence).toVar();
            const blended = mix(prev, next, float(1.0).sub(U.hysteresis)).toVar();
            surfels.element(base).assign(blended.x);
            surfels.element(base.add(uint(1))).assign(blended.y);
            surfels.element(base.add(uint(2))).assign(blended.z);
            surfels.element(base.add(uint(3))).assign(blended.w);
        })().compute(surfelDispatchCount).setName('max.js GI Surfel Direct Slice');

        probeComputeNode = Fn(() => {
            const p = instanceIndex.add(U.probeOffset).toVar();
            If(p.greaterThanEqual(uint(probeTotal)), () => { Return(); });

            const ix = p.mod(uint(nx)).toVar();
            const iy = p.div(uint(nx)).mod(uint(ny)).toVar();
            const iz = p.div(uint(nx * ny)).toVar();
            const pos = vec3(
                U.boundsMin.x.add(float(ix).div(float(Math.max(1, nx - 1))).mul(U.boundsSize.x)),
                U.boundsMin.y.add(float(iy).div(float(Math.max(1, ny - 1))).mul(U.boundsSize.y)),
                U.boundsMin.z.add(float(iz).div(float(Math.max(1, nz - 1))).mul(U.boundsSize.z)),
            ).toVar();

            const accumXP = vec3(0.0).toVar();
            const accumXN = vec3(0.0).toVar();
            const accumYP = vec3(0.0).toVar();
            const accumYN = vec3(0.0).toVar();
            const accumZP = vec3(0.0).toVar();
            const accumZN = vec3(0.0).toVar();
            const weightXP = float(0.0).toVar();
            const weightXN = float(0.0).toVar();
            const weightYP = float(0.0).toVar();
            const weightYN = float(0.0).toVar();
            const weightZP = float(0.0).toVar();
            const weightZN = float(0.0).toVar();

            Loop({ start: uint(0), end: uint(sampleCount), type: 'uint', condition: '<' }, ({ i }) => {
                const sb = i.mul(uint(SAMPLE_STRIDE)).toVar();
                const sPos = loadSampleVec3(sb, 0).toVar();
                const sNormal = loadSampleVec3(sb, 3).normalize().toVar();
                const surfelBase = i.mul(uint(SURFEL_STRIDE)).toVar();
                const bounce = vec4(
                    surfelsRead.element(surfelBase),
                    surfelsRead.element(surfelBase.add(uint(1))),
                    surfelsRead.element(surfelBase.add(uint(2))),
                    surfelsRead.element(surfelBase.add(uint(3))),
                ).toVar();

                If(bounce.w.greaterThan(0.0), () => {
                    const toProbe = pos.sub(sPos).toVar();
                    const distSq = toProbe.dot(toProbe).toVar();
                    If(distSq.greaterThan(float(1e-8)).and(distSq.lessThan(radiusSq)), () => {
                        const dir = toProbe.mul(distSq.inverseSqrt()).toVar();
                        const facing = tslMax(sNormal.dot(dir), 0.0).toVar();
                        If(facing.greaterThan(0.01), () => {
                            const local = float(1.0).sub(distSq.div(radiusSq)).toVar();
                            const w = facing.mul(local).mul(local).toVar();
                            // Store by receiver normal lobe. dir is source->probe,
                            // so probe->source is the direction a receiver normal
                            // must face to see this bounced surfel.
                            const toSource = dir.negate().toVar();
                            const wxp = tslMax(toSource.x, 0.0).mul(w).toVar();
                            const wxn = tslMax(toSource.x.negate(), 0.0).mul(w).toVar();
                            const wyp = tslMax(toSource.y, 0.0).mul(w).toVar();
                            const wyn = tslMax(toSource.y.negate(), 0.0).mul(w).toVar();
                            const wzp = tslMax(toSource.z, 0.0).mul(w).toVar();
                            const wzn = tslMax(toSource.z.negate(), 0.0).mul(w).toVar();
                            accumXP.addAssign(bounce.xyz.mul(wxp));
                            accumXN.addAssign(bounce.xyz.mul(wxn));
                            accumYP.addAssign(bounce.xyz.mul(wyp));
                            accumYN.addAssign(bounce.xyz.mul(wyn));
                            accumZP.addAssign(bounce.xyz.mul(wzp));
                            accumZN.addAssign(bounce.xyz.mul(wzn));
                            weightXP.addAssign(wxp);
                            weightXN.addAssign(wxn);
                            weightYP.addAssign(wyp);
                            weightYN.addAssign(wyn);
                            weightZP.addAssign(wzp);
                            weightZN.addAssign(wzn);
                        });
                    });
                });
            });

            const base = p.mul(uint(PROBE_STRIDE)).toVar();
            const storeLobe = (offset, accum, weight) => {
                const outColor = vec3(0.0).toVar();
                const confidence = float(0.0).toVar();
                If(weight.greaterThan(1e-5), () => {
                    outColor.assign(accum.div(weight));
                    confidence.assign(tslMin(weight.mul(0.24), float(1.0)));
                });
                const lbase = base.add(uint(offset * 4));
                const prev = vec4(
                    probes.element(lbase),
                    probes.element(lbase.add(uint(1))),
                    probes.element(lbase.add(uint(2))),
                    probes.element(lbase.add(uint(3))),
                ).toVar();
                const next = vec4(outColor, confidence).toVar();
                const blended = mix(prev, next, float(1.0).sub(U.hysteresis)).toVar();
                probes.element(lbase).assign(blended.x);
                probes.element(lbase.add(uint(1))).assign(blended.y);
                probes.element(lbase.add(uint(2))).assign(blended.z);
                probes.element(lbase.add(uint(3))).assign(blended.w);
            };
            storeLobe(0, accumXP, weightXP);
            storeLobe(1, accumXN, weightXN);
            storeLobe(2, accumYP, weightYP);
            storeLobe(3, accumYN, weightYN);
            storeLobe(4, accumZP, weightZP);
            storeLobe(5, accumZN, weightZN);
        })().compute(probeDispatchCount).setName('max.js GI Probe Gather Slice');
    }

    function setBounds(box) {
        if (!box || box.isEmpty?.()) return false;
        box.getSize(boundsSize);
        boundsMin.copy(box.min);
        const nextRes = computeResolution(boundsSize);
        const boundsChanged = !sampleAttr
            || nextRes.x !== resolution.x
            || nextRes.y !== resolution.y
            || nextRes.z !== resolution.z
            || !U.boundsMin.value.equals(boundsMin)
            || !U.boundsSize.value.equals(boundsSize);
        resolution.copy(nextRes);
        U.boundsMin.value.copy(boundsMin);
        U.boundsSize.value.copy(boundsSize);
        updateVolumeState();
        if (boundsChanged) dirtyBuffers = true;
        dirtyLightBuffer = true;
        dirtySolve = true;
        return true;
    }

    function requestRefresh() {
        dirtyBuffers = true;
        dirtyLightBuffer = true;
        dirtySurfel = true;
        dirtySolve = true;
        queueSurfelSweep();
        queueSettleSweeps();
    }

    function requestLightRefresh() {
        dirtyLightBuffer = true;
        dirtySurfel = true;
        dirtySolve = true;
        queueSurfelSweep();
        queueSettleSweeps();
    }

    function setNativeSurface({ array, count, boundsMin: min, boundsSize: size } = {}) {
        if (!(array instanceof Float32Array) || !Number.isFinite(count) || count <= 0) return false;
        if (!Array.isArray(min) || !Array.isArray(size)) return false;
        nativeSurfacePack = {
            array,
            count: Math.max(0, Math.floor(count)),
            boundsMin: new THREE.Vector3(min[0] || 0, min[1] || 0, min[2] || 0),
            boundsSize: new THREE.Vector3(Math.max(1e-4, size[0] || 1), Math.max(1e-4, size[1] || 1), Math.max(1e-4, size[2] || 1)),
        };
        requestRefresh();
        return true;
    }

    function setNativeLights({ array, count } = {}) {
        if (!(array instanceof Float32Array) || !Number.isFinite(count)) return false;
        nativeLightPack = {
            array,
            count: Math.max(0, Math.floor(count)),
        };
        requestLightRefresh();
        return true;
    }

    function scheduleBake() { requestLightRefresh(); }
    function bakeAll() { requestRefresh(); }

    function setEnabled(on) {
        const was = node._enabled;
        node.setEnabled(on === true);
        if (on === true && !was) requestRefresh();
        stats.active = node.active;
    }

    function tick({ playback = false } = {}) {
        if (disposed || inFlight || node._enabled !== true || !isSupported()) return;
        if (dirtyBuffers) rebuildSceneBuffers();
        if (!surfelComputeNode || !probeComputeNode) return;
        const nowMs = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
        const effectiveHz = playback === true
            ? Math.max(1, Number.isFinite(playbackHz) ? playbackHz : 3)
            : Math.max(1, Number.isFinite(realtimeHz) ? realtimeHz : 18);
        const intervalMs = realtime === true
            ? 1000 / effectiveHz
            : Infinity;
        if (!dirtySolve && (realtime !== true || nowMs - lastSolveMs < intervalMs)) return;
        const lightChanged = dirtyLightBuffer ? updateLightBuffer() : false;
        if (lightChanged) {
            dirtySolve = true;
            dirtySurfel = true;
            queueSurfelSweep();
            queueSettleSweeps();
        }
        if (!dirtySolve && remainingSettlePasses <= 0 && remainingSurfelSlices <= 0) return;
        dirtySolve = false;
        if (remainingSettlePasses > 0) remainingSettlePasses--;
        U.surfelOffset.value = surfelCursor >>> 0;
        U.probeOffset.value = probeCursor >>> 0;
        surfelCursor = sampleCount > 0
            ? (surfelCursor + Math.max(1, surfelDispatchCount)) % sampleCount
            : 0;
        probeCursor = probeCount > 0
            ? (probeCursor + Math.max(1, probeDispatchCount)) % probeCount
            : 0;
        const runSurfel = dirtySurfel === true || remainingSurfelSlices > 0;
        dirtySurfel = false;
        if (runSurfel && remainingSurfelSlices > 0) remainingSurfelSlices--;
        inFlight = true;
        stats.baking = true;
        const computePromise = runSurfel
            ? renderer.computeAsync(surfelComputeNode).then(() => renderer.computeAsync(probeComputeNode))
            : renderer.computeAsync(probeComputeNode);
        computePromise
            .then(() => {
                lastSolveMs = nowMs;
                stats.updates++;
            })
            .catch((err) => {
                dirtySolve = true;
                dirtySurfel = runSurfel || dirtySurfel;
                console.warn('max.js WebGPU GI compute failed:', err);
            })
            .finally(() => {
                inFlight = false;
                stats.baking = false;
                stats.active = node.active;
            });
        return computePromise;
    }

    function getWrittenProbeCount() {
        return probeCount;
    }

    function dispose() {
        disposed = true;
        disposeBuffers();
        node.setEnabled(false);
    }

    return {
        node,
        setBounds,
        tick,
        bakeAll,
        scheduleBake,
        requestRefresh,
        requestLightRefresh,
        setNativeSurface,
        setNativeLights,
        setEnabled,
        setIntensity: (v) => node.setIntensity(v),
        setNormalBias: (v) => node.setNormalBias(v),
        setEdgeFade: (v) => node.setEdgeFade(v),
        setRadianceClamp: (v) => {
            radClamp = Number.isFinite(v) && v > 0 ? v : 0;
            U.radianceClamp.value = radClamp;
            dirtySolve = true;
        },
        setRealtime: (on) => { stats.realtime = realtime = on === true; dirtySolve = true; },
        setRealtimeHz: (hz) => {
            if (Number.isFinite(hz) && hz > 0) {
                realtimeHz = Math.max(1, Math.min(240, hz));
                stats.realtimeHz = realtimeHz;
            }
        },
        setPlaybackHz: (hz) => {
            if (Number.isFinite(hz) && hz > 0) {
                playbackHz = Math.max(1, Math.min(60, hz));
                stats.playbackHz = playbackHz;
            }
        },
        getRadianceClamp: () => radClamp,
        getResolution: () => resolution.clone(),
        getProbeCount: () => probeCount,
        getWrittenProbeCount,
        hasData: () => node._hasData === true && probeCount > 0,
        hasPendingWork: () => dirtyBuffers || dirtySolve || dirtyLightBuffer || inFlight || remainingSettlePasses > 0 || remainingSurfelSlices > 0,
        isSeeded: () => false,
        isBaking: () => dirtyBuffers || dirtySolve || dirtyLightBuffer || inFlight || remainingSettlePasses > 0 || remainingSurfelSlices > 0,
        isSupported,
        getStats: () => ({ ...stats, active: node.active, baking: dirtyBuffers || dirtySolve || dirtyLightBuffer || inFlight || remainingSettlePasses > 0 || remainingSurfelSlices > 0 }),
        dispose,
    };
}

export default createIrradianceVolume;
