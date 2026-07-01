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
export { giLights, default as GiLightsNode } from './gi_lights_node.js';

// Legacy surfel/lobe irradiance volume (kept for the non-BVH path).
export { createIrradianceVolume, getGiVolumeNode, GiVolumeNode } from './gi_irradiance_volume.js';
