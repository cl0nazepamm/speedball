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
import { getGiProbeNode } from './gi_probes.js';
import { getGiVolumeNode } from './gi_irradiance_volume.js';

export default class GiLightsNode extends DynamicLightsNode {
    static get type() { return 'GiLightsNode'; }

    customCacheKey() {
        const base = super.customCacheKey();
        const arr = [base];
        const probe = getGiProbeNode();
        if (probe.active) arr.push(NodeUtils.hashString(probe.cacheToken));
        const vol = getGiVolumeNode();
        if (vol.active) arr.push(NodeUtils.hashString(vol.cacheToken));
        if (arr.length === 1) return base; // GI off → identical key to stock
        return NodeUtils.hashArray(arr);
    }

    setupLightsNode(builder) {
        let lightNodes = super.setupLightsNode(builder);
        const probe = getGiProbeNode();
        // DynamicLightsNode replaces (rather than extends) Three's stock light list.
        // Restore ONLY material-owned EnvironmentNode for the opt-in reflection graph,
        // first so the probe can composite local radiance over PMREM. Pulling unrelated
        // AO/light-map nodes into this feature would widen its image/perf contract.
        // The legacy roughReflections:false graph remains on the exact old path.
        if (probe.roughReflectionsReady) {
            const environments = (builder.context.materialLightings || []).filter((node) => node?.type === 'EnvironmentNode');
            lightNodes = [ ...environments, ...lightNodes ];
        }
        // Indirect bounce into builder.context.irradiance — global, unmasked, only
        // when active (same contract as MaxLightsNode's GI push).
        if (probe.active) lightNodes.push(probe);
        const vol = getGiVolumeNode();
        if (vol.active) lightNodes.push(vol);
        return lightNodes;
    }
}

export const giLights = (options = {}) => new GiLightsNode(options);
