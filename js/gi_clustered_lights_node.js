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
// This subclass adds the SAME three seams GiLightsNode adds to DynamicLightsNode:
//   1. GI probe / legacy-volume injection into builder.context.irradiance (only when
//      active — with GI off the node is behaviour-identical to stock ClusteredLightsNode),
//   2. IR-emitter lifting: class-4 IR illuminators are black-RGB point lights, so the
//      stock partition would waste cluster slots on lights that shade to zero and can
//      never see the NIR band gate. They ride per-light AnalyticLightNodes shared with
//      gi_lights_node — setNirDirectSensing stays the single band switch for BOTH modes,
//   3. cache-key stamping so GI activation / IR tagging recompiles materials.
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
import { isIrEmitter, getOrCreateIrLightNode } from './gi_lights_node.js';

export default class GiClusteredLightsNode extends ClusteredLightsNode {
    static get type() { return 'GiClusteredLightsNode'; }

    constructor(maxLights = 1024, tileSize = 32, zSlices = 24, maxLightsPerCluster = 64) {
        super(maxLights, tileSize, zSlices, maxLightsPerCluster);
        this._irLights = [];
        this._restLights = [];
    }

    // Partition IR-tagged emitters OUT before the stock clustered/material split
    // (mirrors ClusteredLightsNode.setLights' reuse-the-arrays pattern — called
    // per frame, so no fresh allocations).
    setLights(lights) {
        const ir = this._irLights, rest = this._restLights;
        let ni = 0, nr = 0;
        for (const light of lights) {
            if (isIrEmitter(light)) ir[ni++] = light; else rest[nr++] = light;
        }
        ir.length = ni; rest.length = nr;
        super.setLights(rest);
        this._allLights = lights;   // keep getLights() faithful to the renderer's list
        return this;
    }

    customCacheKey() {
        // super = clustered compute key + core LightsNode key over the non-clustered
        // lights (IR lights are already filtered out of both). Hash IR identity and
        // the GI tokens in so tagging/untagging or GI activation recompiles.
        const base = super.customCacheKey();
        const arr = [base];
        for (const l of this._irLights) arr.push(l.id, l.castShadow ? 1 : 0);
        const probe = getGiProbeNode();
        if (probe.active) arr.push(NodeUtils.hashString(probe.cacheToken));
        const vol = getGiVolumeNode();
        if (vol.active) arr.push(NodeUtils.hashString(vol.cacheToken));
        if (arr.length === 1) return base;   // GI off, no IR → identical key to stock
        return NodeUtils.hashArray(arr);
    }

    // IR lights live in NEITHER stock list (clustered / material), so without this a
    // scene lit only by IR illuminators would report hasLights false and skip setup.
    get hasLights() {
        return super.hasLights || this._irLights.length > 0;
    }

    setupLightsNode(builder) {
        const lightNodes = super.setupLightsNode(builder);
        for (const light of this._irLights) {
            const node = getOrCreateIrLightNode(light, builder.renderer.library);
            if (node !== null) lightNodes.push(node);
        }
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
