import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import {
    createFlatTlasRefitIndex,
    refitFlatTlasDirty,
    refitFlatTlasRange,
} from '../js/gi_refit.js';

// Five-node preorder TLAS:
//   0 interior -> 1,4
//   1 interior -> 2,3
//   2 leaf slot 0, 3 leaf slot 1, 4 leaf slots 2..3
const STRIDE = 12;
const nodes = new Float32Array(5 * STRIDE);
const setNode = (i, miss, off, count) => {
    const b = i * STRIDE;
    nodes[b + 6] = miss;
    nodes[b + 7] = off;
    nodes[b + 8] = count;
};
setNode(0, 5, 0, 0);
setNode(1, 4, 0, 0);
setNode(2, 3, 0, 1);
setNode(3, 4, 1, 1);
setNode(4, 5, 2, 2);
const instanceBounds = new Float32Array([
    0, 0, 0, 1, 1, 1,
    2, 0, 0, 3, 1, 1,
    5, 5, 5, 6, 6, 6,
    7, 7, 7, 8, 8, 8,
]);
assert.equal(refitFlatTlasRange({ nodes, instanceBounds, end: 5 }), true);
const index = createFlatTlasRefitIndex({ nodes, instanceBounds, end: 5 });
assert.ok(index, 'valid threaded TLAS produces a sparse-refit index');
assert.deepEqual(Array.from(index.leafBySlot), [2, 3, 4, 4]);
assert.deepEqual(Array.from(index.parentByNode), [-1, 0, 1, 1, 0]);

const bounds = (i) => Array.from(nodes.subarray(i * STRIDE, i * STRIDE + 6));
const untouchedRight = bounds(4);
instanceBounds.set([20, 20, 20, 21, 21, 21], 0);
assert.deepEqual(
    refitFlatTlasDirty({ nodes, instanceBounds, dirtySlots: [0], index }),
    [0, 1, 2],
    'one dirty instance touches only its leaf and unique ancestor chain',
);
assert.deepEqual(bounds(2), [20, 20, 20, 21, 21, 21]);
assert.deepEqual(bounds(1), [2, 0, 0, 21, 21, 21]);
assert.deepEqual(bounds(4), untouchedRight, 'unrelated TLAS subtree is byte-stable');
assert.deepEqual(bounds(0), [2, 0, 0, 21, 21, 21]);

instanceBounds.set([-5, -5, -5, -4, -4, -4], 2 * 6);
assert.deepEqual(
    refitFlatTlasDirty({ nodes, instanceBounds, dirtySlots: [2], index }),
    [0, 4],
    'a dirty slot in the other partition does not revisit the left subtree',
);
assert.deepEqual(bounds(4), [-5, -5, -5, 8, 8, 8], 'leaf recomputes all of its slots');
assert.equal(refitFlatTlasDirty({ nodes, instanceBounds, dirtySlots: [99], index }), null,
    'invalid host slots fail closed');

const sceneSrc = await readFile(new URL('../js/spectral_scene.js', import.meta.url), 'utf8');
assert.match(sceneSrc, /function updateTransforms\(\{ rewriteInstanceRows = true, objects = null \} = \{\}\)/);
assert.match(sceneSrc, /slotsByObject/);
assert.match(sceneSrc, /refitFlatTlasDirty\(\{/);
assert.match(sceneSrc, /materialRanges/);
assert.match(sceneSrc, /for \(let i = 0; i < Math\.max\(0, capacity \| 0\); i\+\+\)/,
    'InstancedMesh reserves its allocation capacity in the resident TLAS');
assert.match(sceneSrc, /materials\[b \+ 15\] = isActive \? 1 : 0/,
    'live count changes activate and tombstone reserved instance records');

const probesSrc = await readFile(new URL('../js/gi_probes.js', import.meta.url), 'utf8');
const compatibilityScannerStart = probesSrc.indexOf('if (restOnly && detectSceneChanges)');
assert.ok(compatibilityScannerStart >= 0, 'the rest-only compatibility scanner is present');
for (const api of ['markTransformsDirty', 'markDeformsDirty', 'markMaterialValuesDirty', 'markTopologyDirty', 'notifySceneChange']) {
    assert.match(probesSrc, new RegExp(`^\\s*${api},$`, 'm'), `${api} is public`);
}
assert.ok(
    probesSrc.indexOf('if (pendingAllTransforms || pendingTransformTargets.size > 0)')
        < compatibilityScannerStart,
    'host transform packets flush before the rest-only compatibility scanner',
);
assert.match(probesSrc, /built\.updateTransforms\(objects == null \? undefined : \{ objects \}\)/);

// Event-complete integrations can disable the fallback scene traversals without
// disabling explicit transform/material/topology packets.
assert.match(probesSrc, /autoDetectChanges = true,/,
    'compatibility change detection remains enabled by default');
assert.match(probesSrc, /let detectSceneChanges = autoDetectChanges !== false;/);
assert.match(probesSrc, /if \(detectSceneChanges && checkCounter % LIGHT_CHECK_INTERVAL === 0\)/,
    'automatic light signatures are optional');
assert.match(probesSrc, /if \(restOnly && detectSceneChanges\)/,
    'automatic transform, deform, and topology signatures are optional');
assert.match(probesSrc, /const startedGeoSig = detectSceneChanges \? geoSignature\(\) : null;/,
    'event-complete rebuilds rely on explicit generations instead of an extra topology scan');
assert.match(probesSrc, /lastGeoSig = detectSceneChanges \? geoSignature\(\) : null;/,
    'fresh builds skip compatibility signature baselines when detection is disabled');
assert.match(probesSrc, /lastLightSig = detectSceneChanges \? lightSignature\(\) : null;\s*touchGiUniforms\(\);/,
    'an explicit light refresh does not immediately rescan the scene');
assert.match(probesSrc, /setAutoDetectChanges: \(on\) => \{ detectSceneChanges = on !== false; \}/);
assert.match(probesSrc, /getAutoDetectChanges: \(\) => detectSceneChanges/);
assert.match(probesSrc, /fine cascade build failed; continuing with the coarse cascade/,
    'a failed fine cascade degrades to C0 instead of hot-looping every frame');

const installSrc = await readFile(new URL('../js/install.js', import.meta.url), 'utf8');
assert.match(installSrc, /autoDetectChanges = true,/,
    'the one-call API preserves compatibility scanning by default');
assert.match(installSrc, /clusteredLighting: clustered,[\s\S]*autoDetectChanges,[\s\S]*onRebuilt: markMaterialsDirty/,
    'the installer forwards the host detection policy into the field');

const demoSrc = await readFile(new URL('../index.html', import.meta.url), 'utf8');
assert.match(demoSrc, /autoDetectChanges: false/,
    'the event-complete demo opts out of compatibility traversals');
for (const packet of ['markTransformsDirty', 'markMaterialValuesDirty', 'markTopologyDirty', 'forceLightingRefresh']) {
    assert.match(demoSrc, new RegExp(`gi\\?\\.${packet}|gi\\.${packet}`),
        `the demo supplies ${packet} explicitly`);
}

// ── Material-VALUE lane: in-place record rewrite, never a rebuild ──
assert.match(sceneSrc, /function updateMaterialValues\(\{ materials: targets = null \} = \{\}\)/);
assert.match(sceneSrc, /^\s*updateMaterialValues,$/m, 'updateMaterialValues is exported on built');
assert.ok(sceneSrc.includes('rec[12] = materials[b + 12]'),
    'value refresh preserves the buildMaterialTextures-owned layer slots');
assert.match(sceneSrc, /const objectMaterialSnapshot = new Map\(\)/,
    'material reassignment fails closed to the rebuild lane');
assert.ok(
    probesSrc.indexOf('if (pendingAllMaterialValues || pendingMaterialValueTargets.size > 0)')
        < compatibilityScannerStart,
    'material-value packets ride through motion ahead of the rest-only scanner',
);
assert.match(probesSrc, /built\.updateMaterialValues\(targets == null \? undefined : \{ materials: targets \}\)/);
assert.match(probesSrc, /case 'material':\s*\n\s*case 'materials':\s*\n\s*markMaterialValuesDirty/,
    "notifySceneChange('material') routes to the value lane, not straight to a rebuild");
assert.match(probesSrc, /if \(!refreshTransforms\(targets\)\) return;\s*\/\/[\s\S]*?lastXformSig = null;/,
    'an explicit transform packet cannot replay as a full fallback refit at rest');
assert.match(probesSrc, /const instanceCapacity = o\.isInstancedMesh/,
    'InstancedMesh capacity, not active count, owns the structural signature');
assert.match(probesSrc, /Math\.max\(0, o\.count \| 0\)/,
    'active InstancedMesh count remains on the transform signature lane');

// ── Opt-in emissive NEE promotion: split traced energy from the light record ──
assert.match(sceneSrc, /export function collectEmitterRecords\(THREE, scene, camera = null\)/);
assert.match(sceneSrc, /out\.push\(\[3, center\.x, center\.y, center\.z/,
    'promoted emissive meshes create type-3 sphere light records');
assert.ok(sceneSrc.includes("zeroEmissive ? ':E0' : ''"),
    'zero-emissive records cannot dedupe with an unpromoted material record');
const materialValueLane = sceneSrc.slice(
    sceneSrc.indexOf('function updateMaterialValues'),
    sceneSrc.indexOf('let asyncDeformRequestSerial'),
);
assert.match(materialValueLane, /if \(zeroEmissive\) \{ rec\[7\] = 0; rec\[8\] = 0; rec\[9\] = 0; \}/,
    'material-value refresh re-applies promoted-emitter zeroing');
assert.match(probesSrc, /const GI_EMITTER_INJECT_CAP = 16;/);
assert.match(probesSrc, /const GI_EMITTER_VIS_RETENTION = 0\.8;/);
assert.match(probesSrc, /const emitterVisKernel = Fn\(\(\) => \{/);
assert.match(probesSrc, /const solveKernels = \[traceKernel, emitterVisKernel, blendKernel\];/,
    'emitter visibility resolves between trace and probe-field blending');
assert.match(probesSrc, /float\(raysPerProbe\)\.mul\(Omega\)\.div\(float\(4\.0 \* PI\)\)/,
    'emitter injection uses the solid-angle virtual-ray expectation');
assert.equal(
    (probesSrc.match(/If\(ltype\.lessThan\(float\(2\.5\)\), \(\) => \{/g) || []).length,
    2,
    'clustered and legacy per-hit shading both skip type-3 emitter records',
);
assert.match(probesSrc, /const member = ltype\.lessThan\(float\(2\.5\)\)\.and\(/,
    'clustered light-grid membership excludes type-3 emitter records');
assert.doesNotMatch(probesSrc, /const isEmitter =/,
    'per-hit emitter NEE branches are removed');
assert.match(probesSrc, /const analyticBudget = MAX_LIGHTS - emitterCount;/,
    'reserved emitter slots shrink only the analytic-light budget');
assert.match(probesSrc, /emitterBase: selectedAnalytic\.length,/,
    'emitter records occupy one contiguous suffix independent of collection order');

const traversalSrc = await readFile(new URL('../js/spectral_traverse.js', import.meta.url), 'utf8');
assert.match(traversalSrc, /const materialTraversalFlags = \(matId\) => uint\(matFloat\(matId, 26\)\)/);
assert.match(traversalSrc, /If\(flagSet\(flags, 1\), \(\) => \{\s*If\(materialAlphaAccepts/s,
    'alpha UV/texture work is nested behind the packed material flag');
assert.match(traversalSrc, /materialSideAccepts\(matId, det\.mul\(dSign\)\)\.and\(flagSet\(flags, 4\)\)/,
    'non-shadow-blocking materials skip alpha work in any-hit traversal');
assert.equal((traversalSrc.match(/instF\(inst, 15\)\.greaterThan\(float\(0\.5\)\)/g) || []).length, 2,
    'closest and any-hit both mask inactive reserved instance slots');

const sceneMaterialSrc = sceneSrc.match(/const hasAlphaTexture[\s\S]*?const traversalFlags[\s\S]*?;/)?.[0];
assert.ok(sceneMaterialSrc, 'CPU material traversal classification is present');
assert.match(sceneMaterialSrc, /transmissionC < 0\.5/);

console.log('gi-dynamic-tlas-smoke: OK');
