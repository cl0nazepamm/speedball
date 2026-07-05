// GiLightsNode — stock batched lights (DynamicLightsNode) + opt-in SPEEDBALL GI / surfel
// irradiance injection, for NON-Studio WebGPU modes.
//
// SPEEDBALL GI used to be Studio-only because the GI term is injected by the lights node,
// and only Studio installed a custom lights node (MaxLightsNode). This adds the same
// GI injection to a plain DynamicLightsNode so GI works in every WebGPU mode WITHOUT
// the Studio light-linking overhead. The GI nodes only join the graph (and the cache
// key) when active, so with GI OFF this is byte-identical to plain DynamicLightsNode —
// no extra uniform buffers, no behaviour change. Studio keeps using MaxLightsNode.
//
// ── IR illuminators (emitterClass 'ir', class 4) ───────────────────
// An IR illuminator is legitimately BLACK in RGB — its Three.js color stays (0,0,0)
// and only `intensity` drives the 850 nm band. The spectral path tracer gets this
// right for free (emitterAtLambda evaluates the band per wavelength), and the GI
// probes read the promoted (k,k,k) from the lights buffer, but the stock DIRECT
// raster term reads light.color → black × intensity = nothing. Faking it by painting
// light.color white would make the light exist in RGB — wrong band, wrong philosophy.
//
// Instead the direct term is simulated in-shader: IR-tagged lights are lifted out of
// the batched path onto their own AnalyticLightNodes whose colorNode is the SENSED
// color — white × intensity × nirGate — where nirGate is a global uniform saying
// which band the imager samples (0 = visible, 1 = NIR). light.color is never touched:
// in the visible band the light contributes exactly 0, flip to NIR (white-phosphor
// NV) and the direct term appears, matching the (k,k,k) the GI/PT already use.
// Toggling is a uniform write — no scene mutation, no shader recompile.

import DynamicLightsNode from 'three/addons/tsl/lighting/DynamicLightsNode.js';
import { NodeUtils, warn } from 'three/webgpu';
import { uniform, vec3, renderGroup } from 'three/tsl';
import { getGiProbeNode } from './gi_probes.js';
import { getGiVolumeNode } from './gi_irradiance_volume.js';

// Which band the imager senses: 0 = visible (IR direct term is exactly 0),
// 1 = NIR (IR direct term = white × intensity, same promotion the GI probes and
// the path tracer's 850 nm band apply). renderGroup → re-uploaded every render,
// so the toggle is live even while the camera rests.
const _nirGate = uniform(0.0).setGroup(renderGroup);

/** Flip the DIRECT raster term of IR-tagged lights (NV/white-phosphor filter on/off). */
export function setNirDirectSensing(on) { _nirGate.value = on ? 1.0 : 0.0; }
export function getNirDirectSensing() { return _nirGate.value > 0.5; }

// IR classifier — mirrors spectral_scene.js emitterClassValue's class-4 branch.
// Kept inline (not imported): spectral_scene statically imports three-mesh-bvh and
// this node must stay dependency-light for plain GI consumers.
function isIrEmitter(light) {
    const raw = light.userData?.emitterClass;
    if (raw === undefined || raw === null || light.isNode === true) return false;
    if (typeof raw === 'string') {
        const s = raw.trim().toLowerCase();
        return s === 'ir' || s.startsWith('ir_') || s.startsWith('ir ') || s.includes('illuminator');
    }
    return Number.isFinite(raw) && Math.trunc(raw) >= 4;
}

// Per-light AnalyticLightNodes for IR emitters (mirrors DynamicLightsNode's private
// getOrCreateLightNode cache). The node's colorNode is replaced AFTER construction —
// NOT via light.colorNode, which SpotLightNode interprets as a projector FUNCTION.
const _irLightNodes = new WeakMap();

function getOrCreateIrLightNode(light, nodeLibrary) {
    let node = _irLightNodes.get(light);
    if (node !== undefined) return node;

    const LightNodeClass = nodeLibrary.getLightNodeClass(light.constructor);
    if (LightNodeClass === null || LightNodeClass === undefined) {
        warn(`GiLightsNode: Light node not found for IR light ${light.constructor.name}.`);
        node = null;
    } else {
        node = new LightNodeClass(light);
        // Sensed color: white × intensity × band gate. light.color stays black.
        // Shadows still work — AnalyticLightNode wraps colorNode × shadowNode.
        const intensity = uniform(light.intensity)
            .setGroup(renderGroup)
            .onRenderUpdate(() => light.intensity);
        node.colorNode = vec3(1.0, 1.0, 1.0).mul(intensity).mul(_nirGate);
    }
    _irLightNodes.set(light, node);
    return node;
}

export default class GiLightsNode extends DynamicLightsNode {
    static get type() { return 'GiLightsNode'; }

    customCacheKey() {
        const all = this._lights;
        const ir = all.filter(isIrEmitter);
        let base;
        if (ir.length === 0) {
            base = super.customCacheKey();
        } else {
            // Compute the stock key WITHOUT the IR lights (they leave the batched
            // path), then hash their identity in so tagging/untagging recompiles.
            this._lights = all.filter((l) => !isIrEmitter(l));
            try { base = super.customCacheKey(); } finally { this._lights = all; }
        }
        const arr = [base];
        for (const l of ir) arr.push(l.id, l.castShadow ? 1 : 0);
        const probe = getGiProbeNode();
        if (probe.active) arr.push(NodeUtils.hashString(probe.cacheToken));
        const vol = getGiVolumeNode();
        if (vol.active) arr.push(NodeUtils.hashString(vol.cacheToken));
        if (arr.length === 1) return base; // GI off, no IR → identical key to stock
        return NodeUtils.hashArray(arr);
    }

    setupLightsNode(builder) {
        const all = this._lights;
        const ir = all.filter(isIrEmitter);
        let lightNodes;
        if (ir.length === 0) {
            lightNodes = super.setupLightsNode(builder);
        } else {
            // IR lights must NOT reach the batched data nodes (those read light.color
            // CPU-side and ignore any node override → black). Hide them from the stock
            // pass, then append per-light nodes carrying the sensed-band colorNode.
            this._lights = all.filter((l) => !isIrEmitter(l));
            try { lightNodes = super.setupLightsNode(builder); } finally { this._lights = all; }
            for (const light of ir) {
                const node = getOrCreateIrLightNode(light, builder.renderer.library);
                if (node !== null) lightNodes.push(node);
            }
        }
        // Indirect bounce into builder.context.irradiance — global, unmasked, only
        // when active (same contract as MaxLightsNode's GI push).
        const probe = getGiProbeNode();
        if (probe.active) lightNodes.push(probe);
        const vol = getGiVolumeNode();
        if (vol.active) lightNodes.push(vol);
        return lightNodes;
    }

    // setLights() re-feeds the batched data nodes outside setupLightsNode — keep IR
    // lights out of that path too (they'd occupy batch slots as black no-ops).
    _updateDataNodeLights(lights) {
        if (lights.some(isIrEmitter)) lights = lights.filter((l) => !isIrEmitter(l));
        super._updateDataNodeLights(lights);
    }
}

export const giLights = (options = {}) => new GiLightsNode(options);
