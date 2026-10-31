// GiClusteredLightsNode — three's Forward+ ClusteredLightsNode + SPEEDBALL GI injection.
// The RASTER half of the opt-in "clustered lighting" mode (installSpeedballGI
// { clusteredLighting: true }).
//
// Stock three r185+ ClusteredLightsNode bins every non-shadowed PointLight into a
// view-frustum cluster grid (screen tiles × exponential Z slices, compute-culled), so
// fragments loop only their cluster's lights — thousands of small direct lights for
// near-constant cost. Everything else (directional, spot, shadow-casting points) falls
// through to the stock per-light path unchanged.
//
// This subclass adds the SAME two seams GiLightsNode adds to DynamicLightsNode:
//   1. GI probe / legacy-volume injection into builder.context.irradiance (only when
//      active — with GI off the node is behaviour-identical to stock ClusteredLightsNode),
//   2. cache-key stamping so GI activation recompiles materials.
//
// Unlike GiLightsNode there is NO environment-restore step: core LightsNode (the
// clustered node's base) already folds builder.context.materialLightings into the
// light list — only DynamicLightsNode replaces it.
//
// Note the raster/GI split of the clustered mode: this node draws EVERY light
// directly; the probe field's NEE lane budgets itself to the MAX_LIGHTS most
// important records via gi_probes' importance selection (see createProbeField's
// clusteredLighting option), so thousands of raster lights never blow up the
// per-probe-ray shadow loop.

import ClusteredLightsNode from 'three/addons/tsl/lighting/ClusteredLightsNode.js';
import { NodeUtils } from 'three/webgpu';
import { getGiProbeNode } from './gi_probes.js';
import { getGiVolumeNode } from './gi_irradiance_volume.js';

export default class GiClusteredLightsNode extends ClusteredLightsNode {
    static get type() { return 'GiClusteredLightsNode'; }

    customCacheKey() {
        // super = clustered compute key + core LightsNode key over the non-clustered
        // lights. Hash the GI tokens in so GI activation recompiles.
        const base = super.customCacheKey();
        const arr = [base];
        const probe = getGiProbeNode();
        if (probe.active) arr.push(NodeUtils.hashString(probe.cacheToken));
        const vol = getGiVolumeNode();
        if (vol.active) arr.push(NodeUtils.hashString(vol.cacheToken));
        if (arr.length === 1) return base;   // GI off → identical key to stock
        return NodeUtils.hashArray(arr);
    }

    setupLightsNode(builder) {
        const lightNodes = super.setupLightsNode(builder);
        const probe = getGiProbeNode();
        // Indirect bounce into builder.context.irradiance — global, unmasked, only
        // when active (same contract as GiLightsNode / MaxLightsNode's GI push).
        if (probe.active) lightNodes.push(probe);
        const vol = getGiVolumeNode();
        if (vol.active) lightNodes.push(vol);
        return lightNodes;
    }
}

/**
 * Factory for the clustered-mode lights node (options-object flavour of the stock
 * positional constructor — matches giLights' shape).
 *
 * @param {object} [opts]
 * @param {number} [opts.maxLights=1024]          maximum clustered point lights
 * @param {number} [opts.tileSize=32]             screen tile size in pixels
 * @param {number} [opts.zSlices=24]              exponential depth slices
 * @param {number} [opts.maxLightsPerCluster=64]  per-cluster light-list capacity
 */
export const giClusteredLights = ({ maxLights = 1024, tileSize = 32, zSlices = 24, maxLightsPerCluster = 64 } = {}) =>
    new GiClusteredLightsNode(maxLights, tileSize, zSlices, maxLightsPerCluster);
