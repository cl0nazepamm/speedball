// Material-map residency smoke. This repo uses node-only source assertions for
// gi_probes because importing it requires a live Three WebGPU renderer. Prove
// the load-bearing contract: exact material slots, headroom and format checks,
// staged/changed-layer uploads into stable texture identities, independent
// standalone behavior, ref-counted generation overlap, coherent map/storage
// fallback, and live counters for reviewer verification.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const scene = await readFile(new URL('../js/spectral_scene.js', import.meta.url), 'utf8');
const probes = await readFile(new URL('../js/gi_probes.js', import.meta.url), 'utf8');
const tracer = await readFile(new URL('../js/spectral_tracer.js', import.meta.url), 'utf8');

// All six texture keys retain the existing material-record ABI. Fresh records
// initialize every layer slot to -1, so no stale capacity layer is addressable.
for (const [field, recIdx, out] of [
    ['map', 12, 'albedo'],
    ['normalMap', 13, 'normal'],
    ['roughnessMap', 14, 'roughness'],
    ['metalnessMap', 15, 'metalness'],
    ['emissiveMap', 16, 'emissive'],
    ['alphaMap', 24, 'alpha'],
]) {
    assert.match(scene, new RegExp(`field: '${field}', recIdx: ${recIdx}, out: '${out}'`));
    assert.match(scene, new RegExp(`uberList\\[i\\]\\[ty\\.recIdx\\] = layer`));
}
assert.match(scene, /-1, -1, -1, -1, -1,[^\n]*\[12\.\.16\] map layers/);
assert.match(scene, /-1,[^\n]*\[24\] alpha-map layer/);

// Probe builds opt into the arena; the spectral tracer still calls the builder
// without one and receives exact-size, independently disposable textures.
assert.match(scene, /mapsArena = null,/);
assert.match(probes, /buildSpectralScene\(\{[\s\S]*?blasCache,[\s\S]*?mapsArena,[\s\S]*?\}\)/);
assert.match(tracer, /buildSpectralScene\(\{ THREE, scene, camera: activeCamera, maxTriangles: MAX_TRIANGLES \}\)/);
assert.match(scene, /if \(!mapsArena\) \{[\s\S]*?new THREE\.DataArrayTexture\(packedLayers\[out\], size, size, live\)/);

// Capacity is geometric and device-capped. Reuse requires all live counts to
// fit and checks the texture object's dimensions, depth, format, and type.
assert.match(probes, /growth: PROBE_SCENE_STORAGE_GROWTH/);
assert.match(scene, /prior > 0 \? Math\.ceil\(prior \* factor\) : 0/);
assert.match(scene, /return limit >= live \? Math\.min\(target, limit\) : live/);
assert.match(scene, /generation\.width !== size \|\| generation\.height !== size/);
assert.match(scene, /texture\.image\?\.depth === capacity/);
assert.match(scene, /texture\.format === format[\s\S]*?texture\.type === type/);
assert.match(scene, /liveLayers\[out\] <= generation\.capacities\[out\]/);

// A fit returns the exact resident texture map and stages bytes without touching
// it. Commit copies only byte-different live layers and queues per-layer uploads.
assert.match(scene, /maps: previous\.textures,[\s\S]*?kind: 'rewrite'/);
assert.match(scene, /function materialMapLayerEquals\(/);
assert.match(scene, /changedLayers\[ty\.out\]\.push\(layer\)/);
assert.match(probes, /texture\.image\.data\.set\(packed\.subarray\(start, start \+ layerBytes\), start\)/);
assert.match(probes, /for \(const layer of changedLayers\) texture\.addLayerUpdate\(layer\)/);
assert.match(probes, /texture\.needsUpdate = true/);

// Growth/config changes make fresh texture objects. Map-generation changes
// force a fresh scene-storage arena; scene-storage growth inversely forks maps,
// keeping old C1 material records paired with old map contents until release.
assert.match(scene, /new THREE\.DataArrayTexture\(data, size, size, capacity\)/);
assert.match(scene, /kind: previous \? \(configCompatible \? 'grow' : 'reconfigure'\) : 'allocate'/);
assert.match(scene, /export function rebindMaterialMapsArenaBuild/);
assert.match(probes, /const mapsGenerationRebound =|let mapsGenerationRebound =/);
assert.match(probes, /sceneStorageFits\(arena, built, mapsGenerationRebound\)/);
assert.match(probes, /_rebindMaterialMapsArenaBuild\(THREE, built, mapsArena\)/);
assert.match(probes, /sceneStorageLastUpdate = mapsGenerationRebound && previous[\s\S]*?'maps-rebind'/);

// Scene resources retain/release a map generation instead of disposing maps per
// build wrapper. Stale candidates are disposed only when uninstalled and unowned.
assert.match(probes, /const mapsGeneration = retainMapsGeneration\(built\.mapsArenaGeneration\)/);
assert.match(probes, /releaseMapsGeneration\(resource\.mapsGeneration\);\s*releaseSceneStorage\(resource\.storage\)/);
assert.match(probes, /if \(generation\.refs === 0 && generation !== mapsArena\?\.current\)/);
assert.match(probes, /if \(mapsArena\) mapsArena\.current = null;\s*mapsArena = null;/);

// Kernel counters distinguish the identity-stable path from every graph build;
// arena generation/capacity/live counts expose the same review surface as storage.
assert.match(probes, /kernelResidentReuses\+\+;\s*return false;/);
assert.equal((probes.match(/kernelRebuilds\+\+;/g) || []).length, 2);
for (const field of [
    'mapsArenaGeneration', 'mapsArenaRebinds', 'mapsArenaRewrites',
    'mapsArenaLastUpdate', 'mapsArenaCapacities', 'mapsArenaLiveLayers',
    'kernelResidentReuses', 'kernelRebuilds',
]) {
    assert.match(probes, new RegExp(`\\b${field}\\b`));
}

console.log('gi-maps-residency-smoke: OK');
