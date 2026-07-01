// GiLightsNode — stock batched lights (DynamicLightsNode) + opt-in SPEEDBALL GI / surfel
// irradiance injection, for NON-Studio WebGPU modes.
//
// SPEEDBALL GI used to be Studio-only because the GI term is injected by the lights node,
// and only Studio installed a custom lights node (MaxLightsNode). This adds the same
// GI injection to a plain DynamicLightsNode so GI works in every WebGPU mode WITHOUT
// the Studio light-linking overhead. The GI nodes only join the graph (and the cache
// key) when active, so with GI OFF this is byte-identical to plain DynamicLightsNode —
// no extra uniform buffers, no behaviour change. Studio keeps using MaxLightsNode.

import DynamicLightsNode from 'three/addons/tsl/lighting/DynamicLightsNode.js';
import { NodeUtils } from 'three/webgpu';
import { getGiProbeNode } from './gi_probes.js?v=speedball4';
import { getGiVolumeNode } from './gi_irradiance_volume.js?v=speedball4';

export default class GiLightsNode extends DynamicLightsNode {
    static get type() { return 'GiLightsNode'; }

    customCacheKey() {
        const base = super.customCacheKey();
        const tokens = [];
        const probe = getGiProbeNode();
        if (probe.active) tokens.push(probe.cacheToken);
        const vol = getGiVolumeNode();
        if (vol.active) tokens.push(vol.cacheToken);
        if (tokens.length === 0) return base; // GI off → identical key to stock
        const arr = [base];
        for (const tk of tokens) arr.push(NodeUtils.hashString(tk));
        return NodeUtils.hashArray(arr);
    }

    setupLightsNode(builder) {
        const lightNodes = super.setupLightsNode(builder);
        // Indirect bounce into builder.context.irradiance — global, unmasked, only
        // when active (same contract as MaxLightsNode's GI push).
        const probe = getGiProbeNode();
        if (probe.active) lightNodes.push(probe);
        const vol = getGiVolumeNode();
        if (vol.active) lightNodes.push(vol);
        return lightNodes;
    }
}

export const giLights = (options = {}) => new GiLightsNode(options);
