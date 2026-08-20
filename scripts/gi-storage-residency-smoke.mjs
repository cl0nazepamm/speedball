// Structural-residency smoke. The repo's node-only smoke style uses source
// assertions because importing gi_probes.js would require a live WebGPU/Three
// renderer. Prove the load-bearing wiring: one headroom arena owns stable
// attributes + storage nodes, within-capacity builds rewrite it in place without
// firing the node-material rebuild callback, growth allocates a new arena and
// does fire that one-time rebind, live traversal uniforms exclude stale tail,
// and the existing transform/deform lanes copy their dirty CPU ranges into the
// resident backing arrays before requesting ranged uploads.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const probes = await readFile(new URL('../js/gi_probes.js', import.meta.url), 'utf8');
const traverse = await readFile(new URL('../js/spectral_traverse.js', import.meta.url), 'utf8');

assert.match(probes, /const PROBE_SCENE_STORAGE_GROWTH = 1\.5;/,
    'scene buffers reserve 50% geometric headroom');
assert.match(probes, /function sceneStorageFits\(arena, built, forceRebind = false\) \{\s*return !forceRebind && !!arena\s*&& PROBE_SCENE_BUFFER_KEYS\.every\(\(key\) => built\[key\]\.length <= arena\.capacities\[key\]\);\s*\}/,
    'every live scene array must fit before the arena can be reused');

// StorageBufferAttribute and storage() creation belong exclusively to the arena.
// buildKernels must consume those resident nodes instead of rebuilding five nodes.
for (const key of ['bvhNodes', 'triIndex', 'vertexData', 'triMaterial', 'materials']) {
    assert.match(probes, new RegExp(`${key}: storage\\(buffers\\.${key},`),
        `${key} owns one arena-level storage node`);
}
assert.match(probes, /const \{ bvhNodes, triIndex, vertexData, triMaterial, materials \} = sharedScene\.storages;/,
    'cascade kernels bind the arena-owned storage nodes');
assert.match(probes, /const storageNode = storage\(buffer, 'float', buffer\.count\);/,
    'the fixed light arena also owns one resident storage node');
assert.match(probes, /const lights = sharedLights\.storage;/,
    'cascade kernels reuse the resident light storage node');

// Fit path: same arena, prefix rewrite, stable object identities, no rebind flag.
assert.match(probes, /if \(!sceneStorageFits\(arena, built, mapsGenerationRebound\)\) \{[\s\S]*?sceneBuffersRebound = true;[\s\S]*?\} else \{\s*rewriteSceneStorage\(arena, built\);\s*sceneStorageRewrites\+\+;\s*sceneStorageLastUpdate = 'rewrite';/,
    'within-capacity rebuild rewrites rather than reallocates');
assert.match(probes, /arena\.buffers\[key\]\.array\.set\(source, 0\);/,
    'resident attribute backing arrays receive the new live prefix');
assert.match(probes, /if \(\(recompiled \|\| sceneBuffersRebound \|\| node\._structGen !== genBefore\) && typeof onRebuilt === 'function'\)/,
    'within-capacity storage rewrites do not trigger the node-material rebuild callback');

// Growth path: a new headroom arena means new attributes/storage nodes and exactly
// the structural notification that the fit path avoids.
assert.match(probes, /arena = createSceneStorage\(built, arena\);\s*sceneStorageRebinds\+\+;\s*sceneStorageLastUpdate = mapsGenerationRebound && previous\s*\? 'maps-rebind'\s*:\s*\(previous \? 'grow' : 'allocate'\);\s*sceneBuffersRebound = true;/,
    'capacity overflow takes the explicit rebind path');

// Stale tail is unreachable: all live sizes/bases are uniforms updated on every
// rewrite, TLAS walks end at the live count, and each BLAS also carries its live end.
for (const field of ['nodeCount', 'tlasNodeCount', 'instBase', 'tlasBase']) {
    assert.match(probes, new RegExp(`Utrav\\.${field}\\.value =`),
        `${field} is refreshed for each live build`);
}
assert.match(traverse, /Loop\(\{ start: uint\(0\), end: U\.tlasNodeCount,/,
    'TLAS traversal is bounded by the live node count, not buffer capacity');
assert.match(traverse, /const blasEnd = uint\(instF\(inst, 13\)\)\.toVar\(\);/,
    'BLAS traversal is bounded by each live instance record');
assert.match(traverse, /materials\.element\(U\.instBase\.add/);
assert.match(traverse, /materials\.element\(U\.tlasBase\.add/);

// Resident backing arrays are separate from built.*; keep both fast lanes exact.
assert.match(probes, /copySceneStorageRanges\('materials', built\.materials, res\.materialRanges\)/,
    'transform refits copy material/TLAS ranges into resident storage');
assert.match(probes, /copySceneStorageRanges\('vertexData', built\.vertexData, res\.vertRanges\)/,
    'deform refits copy vertex ranges into resident storage');
assert.match(probes, /copySceneStorageRanges\('bvhNodes', built\.bvhNodes, res\.nodeRanges\)/,
    'deform refits copy BVH ranges into resident storage');

// Debug counters make churn.html review deterministic: rewrites rise while the
// arena generation/rebind count remains flat until a capacity overflow.
for (const field of [
    'sceneStorageGeneration', 'sceneStorageRebinds', 'sceneStorageRewrites',
    'sceneStorageLastUpdate', 'sceneStorageCapacities', 'sceneStorageLiveLengths',
]) {
    assert.match(probes, new RegExp(`\\b${field}\\b`));
}

console.log('gi-storage-residency-smoke: OK');
