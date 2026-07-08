// SPEEDBALL GI — public entry point.
//
// Easy path (batteries included):
//   import { installSpeedballGI } from 'speedball-gi';
//   const gi = installSpeedballGI({ renderer, scene, camera });
//   // render loop: gi.update();
//
// Advanced path (wire it yourself):
//   import { createProbeField, giLights } from 'speedball-gi';

// One-call setup + scene helpers.
export { installSpeedballGI, excludeFromGI, prepareMaterialsForGI } from './install.js';

// Core BVH-traced DDGI probe field (octahedral, infinite-bounce, idle-gated).
export { createProbeField, getGiProbeNode, GiProbeNode } from './gi_probes.js';

// Lights node that injects the GI term into every PBR material.
// setNirDirectSensing flips the DIRECT term of emitter-class-'ir' lights between
// bands (0 in visible, white × intensity under NV); pair it with the probe field's
// setNirSensing — or use the install handle's setNirSensing, which does both.
// isIrEmitter / getOrCreateIrLightNode are the seam for hosts with a CUSTOM lights
// node: lift IR lights out of your batched path onto these nodes and the same
// setNirDirectSensing switch drives them.
export {
    giLights, default as GiLightsNode,
    setNirDirectSensing, getNirDirectSensing,
    isIrEmitter, getOrCreateIrLightNode,
} from './gi_lights_node.js';

// Legacy surfel/lobe irradiance volume (kept for the non-BVH path).
export { createIrradianceVolume, getGiVolumeNode, GiVolumeNode } from './gi_irradiance_volume.js';
