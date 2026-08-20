// Deform fast-path smoke: refitFlatBlasRange must rebuild exact bounds over
// the threaded flat layout (left child = i+1, right child = left's miss),
// address the pool ABSOLUTELY (a BLAS can sit at any base), propagate a
// deform to every ancestor, and fail LOUDLY (false) on layout drift. Plus
// source-text checks that the deform path is actually wired: spectral_scene
// exposes updateDeforms, and gi_probes routes vertex motion to the refit —
// never to the ~200 ms synchronous MeshBVH rebuild.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { refitFlatBlasRange, refitFlatTlasRange } from '../js/gi_refit.js';

const NODE_STRIDE = 8;
const VDATA_STRIDE = 8;

// ── Hand-built 5-node threaded BLAS over 4 triangles ──
//   0 interior (miss 5)
//   ├─ 1 interior (miss 4)          right child of 0 = miss(1) = 4
//   │   ├─ 2 leaf tri0 (miss 3)     right child of 1 = miss(2) = 3
//   │   └─ 3 leaf tri1 (miss 4)
//   └─ 4 leaf tris 2..3 (miss 5)
// Placed at arbitrary pool bases to prove absolute addressing.
function buildFixture({ nodeBase = 0, triBase = 0, vertBase = 0 }) {
    const nodeCount = nodeBase + 5;
    const buf = new ArrayBuffer(nodeCount * NODE_STRIDE * 4);
    const nodesF = new Float32Array(buf);
    const nodesU = new Uint32Array(buf);
    const triIndex = new Uint32Array((triBase + 4) * 3);
    const vertexData = new Float32Array((vertBase + 12) * VDATA_STRIDE);

    const tris = [
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[2, 0, 0], [3, 0, 0], [2, 1, 1]],
        [[5, 5, 5], [6, 5, 5], [5, 6, 5]],
        [[7, 7, 7], [8, 7, 7], [7, 8, 8]],
    ];
    for (let t = 0; t < 4; t++) {
        for (let k = 0; k < 3; k++) {
            const v = vertBase + t * 3 + k;
            triIndex[(triBase + t) * 3 + k] = v;
            vertexData.set(tris[t][k], v * VDATA_STRIDE);
        }
    }
    const node = (i, miss, leaf) => {
        const b = (nodeBase + i) * NODE_STRIDE;
        nodesU[b + 6] = nodeBase + miss;
        nodesU[b + 7] = leaf ? (((leaf.cnt & 0xFF) << 24) | ((triBase + leaf.off) & 0x00FFFFFF)) >>> 0 : 0xFFFFFFFF;
        // bounds left as garbage zeros — refit must overwrite every node
    };
    node(0, 5, null);
    node(1, 4, null);
    node(2, 3, { off: 0, cnt: 1 });
    node(3, 4, { off: 1, cnt: 1 });
    node(4, 5, { off: 2, cnt: 2 });
    return { nodesF, nodesU, triIndex, vertexData, root: nodeBase, end: nodeBase + 5 };
}

function boundsOf(fix, i) {
    const b = i * NODE_STRIDE;
    return Array.from(fix.nodesF.subarray(b, b + 6));
}

for (const bases of [{ nodeBase: 0, triBase: 0, vertBase: 0 }, { nodeBase: 7, triBase: 2, vertBase: 5 }]) {
    const fix = buildFixture(bases);
    assert.equal(refitFlatBlasRange({ ...fix, nodeStrideU32: NODE_STRIDE, vertexDataStride: VDATA_STRIDE }), true);
    const nb = bases.nodeBase;
    assert.deepEqual(boundsOf(fix, nb + 2), [0, 0, 0, 1, 1, 0], 'leaf tri0');
    assert.deepEqual(boundsOf(fix, nb + 3), [2, 0, 0, 3, 1, 1], 'leaf tri1');
    assert.deepEqual(boundsOf(fix, nb + 4), [5, 5, 5, 8, 8, 8], 'leaf tris 2..3');
    assert.deepEqual(boundsOf(fix, nb + 1), [0, 0, 0, 3, 1, 1], 'interior = union of leaves 2,3');
    assert.deepEqual(boundsOf(fix, nb + 0), [0, 0, 0, 8, 8, 8], 'root = union of all');

    // Deform: shift tri3 by +10 on every axis → only its leaf and ancestors move.
    for (let k = 0; k < 3; k++) {
        const v = (bases.vertBase + 9 + k) * VDATA_STRIDE;
        fix.vertexData[v] += 10; fix.vertexData[v + 1] += 10; fix.vertexData[v + 2] += 10;
    }
    assert.equal(refitFlatBlasRange({ ...fix, nodeStrideU32: NODE_STRIDE, vertexDataStride: VDATA_STRIDE }), true);
    assert.deepEqual(boundsOf(fix, nb + 4), [5, 5, 5, 18, 18, 18], 'deformed leaf refit');
    assert.deepEqual(boundsOf(fix, nb + 0), [0, 0, 0, 18, 18, 18], 'deform propagates to root');
    assert.deepEqual(boundsOf(fix, nb + 1), [0, 0, 0, 3, 1, 1], 'untouched subtree stays exact');
}

// Layout drift must abort, not mis-walk: corrupt the left child's miss link
// (the right-child pointer) to land outside the BLAS range.
{
    const fix = buildFixture({ nodeBase: 0, triBase: 0, vertBase: 0 });
    fix.nodesU[1 * NODE_STRIDE + 6] = 99;
    assert.equal(refitFlatBlasRange({ ...fix, nodeStrideU32: NODE_STRIDE, vertexDataStride: VDATA_STRIDE }), false);
}

// ── Fixed-topology TLAS refit ──
// Same five-node shape as the BLAS fixture, but leaves reference permanent
// instance slots. Put it behind a prefix and pass a subarray to prove packed
// materials-tail addressing does not leak into the refit contract.
{
    const STRIDE = 12, prefixNodes = 3;
    const packed = new Float32Array((prefixNodes + 5) * STRIDE);
    const nodes = packed.subarray(prefixNodes * STRIDE);
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
    const topologyBefore = [0, 1, 2, 3, 4].map((i) => Array.from(nodes.subarray(i * STRIDE + 6, i * STRIDE + 9)));
    const instanceBounds = new Float32Array([
        0, 0, 0, 1, 1, 1,
        2, 0, 0, 3, 1, 1,
        5, 5, 5, 6, 6, 6,
        7, 7, 7, 8, 8, 8,
    ]);
    assert.equal(refitFlatTlasRange({ nodes, instanceBounds, end: 5 }), true);
    const bounds = (i) => Array.from(nodes.subarray(i * STRIDE, i * STRIDE + 6));
    assert.deepEqual(bounds(1), [0, 0, 0, 3, 1, 1], 'TLAS left partition');
    assert.deepEqual(bounds(4), [5, 5, 5, 8, 8, 8], 'TLAS right partition');
    assert.deepEqual(bounds(0), [0, 0, 0, 8, 8, 8], 'TLAS root union');

    // Cross the original median partition. Bounds stay exact even though the
    // frozen partition may now be less traversal-optimal.
    instanceBounds.set([20, 20, 20, 21, 21, 21], 0);
    assert.equal(refitFlatTlasRange({ nodes, instanceBounds, end: 5 }), true);
    assert.deepEqual(bounds(1), [2, 0, 0, 21, 21, 21], 'crossed frozen partition refits exactly');
    assert.deepEqual(bounds(0), [2, 0, 0, 21, 21, 21], 'crossed root stays exact');
    assert.deepEqual(
        [0, 1, 2, 3, 4].map((i) => Array.from(nodes.subarray(i * STRIDE + 6, i * STRIDE + 9))),
        topologyBefore,
        'refit never rewrites miss/payload/order fields',
    );

    nodes[1 * STRIDE + 6] = 99;
    assert.equal(refitFlatTlasRange({ nodes, instanceBounds, end: 5 }), false, 'invalid TLAS layout escalates');
}

{
    const nodes = new Float32Array(12);
    nodes[7] = 0; nodes[8] = 1;
    assert.equal(refitFlatTlasRange({ nodes, instanceBounds: new Float32Array([-2, -3, -4, 5, 6, 7]) }), true);
    assert.deepEqual(Array.from(nodes.subarray(0, 6)), [-2, -3, -4, 5, 6, 7], 'single-instance TLAS');
}

// ── Wiring: the deform path must actually be reachable ──
const sceneSrc = await readFile(new URL('../js/spectral_scene.js', import.meta.url), 'utf8');
assert.match(sceneSrc, /function updateDeforms\([^)]*\)/);
assert.match(sceneSrc, /^\s*updateDeforms,$/m, 'updateDeforms exposed on the built scene');
assert.match(sceneSrc, /^\s*updateLights,$/m, 'same-count light edits have a BVH-free update lane');
assert.match(sceneSrc, /nextRecords\.length !== lightRecords\.length\) return null/,
    'light-count drift fails closed to a structural rebuild');
assert.match(sceneSrc, /if \(tagSrc\) tagSrc\[t\] = sourceA;/, 'tag-vertex source map captured for multi-material gathers');
assert.match(sceneSrc, /srcVertCount: vCount/, 'soup keeps the 1:1 source vertex count');
assert.match(sceneSrc, /const posChanged = posA !== b\.srcPosAttr/);
assert.match(sceneSrc, /const normChanged = normA !== b\.srcNormAttr/,
    'same-sized replacement attributes cannot hide behind reset version counters');
assert.match(sceneSrc, /indexA !== b\.srcIndexAttr \|\| iv !== b\.srcIndexVersion/,
    'connectivity edits fail closed before deform refit touches the old triangle soup');
assert.match(sceneSrc, /refitFlatBlasRange\(\{/);
assert.match(sceneSrc, /refitFlatTlasRange\(\{/);
const updateDeformsSrc = sceneSrc.match(/function updateDeforms\([\s\S]*?\n    \}/)?.[0];
assert.ok(updateDeformsSrc, 'updateDeforms body present');
assert.match(updateDeformsSrc, /const posChanged =/);
assert.match(updateDeformsSrc, /if \(posChanged && !refitFlatBlasRange/,
    'a normal-only settle updates shading data without rebuilding BLAS bounds');
assert.match(updateDeformsSrc, /if \(refitted === 0 \|\| !updateTlas\)/,
    'a normal-only settle cannot rewrite the TLAS');
const updateTransformsSrc = sceneSrc.match(/function updateTransforms\([\s\S]*?\n    \}/)?.[0];
assert.ok(updateTransformsSrc, 'updateTransforms present');
assert.doesNotMatch(updateTransformsSrc, /computeDynamic\(/, 'live transforms never rebuild or sort the TLAS');

const probesSrc = await readFile(new URL('../js/gi_probes.js', import.meta.url), 'utf8');
assert.match(probesSrc, /checkCounter % DEFORM_CHECK_INTERVAL === 0/, 'deform check runs in the tick');
assert.match(probesSrc, /function deformSignature\(\)/);
assert.match(probesSrc, /function refreshDeforms\(\)/);
assert.match(probesSrc, /const startedGeneration = buildGeneration;/,
    'probe build snapshots the fresh-build invalidation generation');
assert.match(probesSrc, /geoSignature\(\) !== startedGeoSig/,
    'probe build rejects structural drift across async material extraction');
assert.match(probesSrc, /if \(ok === 'retry'\) return;/,
    'async build drift remains armed instead of entering the failure backoff');
assert.match(probesSrc, /lastDeformSig = deformSignature\(\);/,
    'successful build establishes a deform baseline from the installed pose');
assert.match(probesSrc, /built\.bounds\.copy\(transformResult\.bounds\)/,
    'async build catch-up publishes the final deformed/transformed bounds');
assert.match(probesSrc, /if \(disposed\) \{\s*disposeUninstalledBuild\(built\);\s*return false;/,
    'disposing a field during async texture extraction rejects and frees the old build');
assert.match(probesSrc, /function dispose\(\) \{[\s\S]*?buildGeneration\+\+;/,
    'field disposal invalidates the in-flight build generation');

// STRUCTURE signature must NOT hash position.version anymore — that is the
// exact coupling that armed a full MeshBVH rebuild after every vertex-stream
// pause (the skinned-anim hitch). Vertex motion belongs to deformSignature.
const geoSig = probesSrc.match(/function geoSignature\(\) \{[\s\S]*?\n    \}/)?.[0];
assert.ok(geoSig, 'geoSignature present');
assert.doesNotMatch(geoSig, /p\.version/, 'structure signature no longer hashes position.version');
assert.doesNotMatch(geoSig, /structureId\(p\)/,
    'same-sized position attribute replacement stays on the deform lane');
assert.match(geoSig, /isSkinnedMesh/, 'structure signature mirrors the BVH SkinnedMesh skip');
const deformSig = probesSrc.match(/function deformSignature\(\) \{[\s\S]*?\n    \}/)?.[0];
assert.ok(deformSig, 'deformSignature present');
assert.match(deformSig, /p\.version/, 'deform signature owns position.version');
assert.match(deformSig, /structureId\(p\)/,
    'deform signature owns same-sized replacement attribute identity');

// The refit core must stay dependency-free so this smoke keeps running
// without installed peers (three / three-mesh-bvh).
const refitSrc = await readFile(new URL('../js/gi_refit.js', import.meta.url), 'utf8');
assert.doesNotMatch(refitSrc, /^import /m, 'gi_refit.js must not import anything');

const tracerSrc = await readFile(new URL('../js/spectral_tracer.js', import.meta.url), 'utf8');
assert.match(tracerSrc, /function markTransformsDirty\(\)/, 'path tracer exposes transform refit lane');
assert.match(tracerSrc, /function markDeformsDirty\(/, 'path tracer exposes deform refit lane');
assert.match(tracerSrc, /function markLightsDirty\(\)/, 'path tracer exposes packed-light update lane');
assert.match(tracerSrc, /gpu = \{[^\n]*built \};/, 'path tracer retains the built scene for live refits');
for (const [label, source] of [['probe field', probesSrc], ['path tracer', tracerSrc]]) {
    const dirtyHelper = source.match(/function markStorageDirty\([^)]*\) \{[\s\S]*?\n    \}/)?.[0];
    assert.ok(dirtyHelper, `${label} storage dirty helper present`);
    assert.doesNotMatch(dirtyHelper, /clearUpdateRanges\(\);\s*\n\s*for \(const \[start, count\]/,
        `${label} preserves earlier disjoint ranges until GPU upload`);
    const markStorageDirty = Function(`return (${dirtyHelper});`)();
    const queuedRanges = [];
    const attr = {
        addUpdateRange(start, count) { queuedRanges.push([start, count]); },
        clearUpdateRanges() { queuedRanges.length = 0; },
        set needsUpdate(value) { this.updated = value; },
    };
    markStorageDirty(attr, [[2, 5]]);
    markStorageDirty(attr, [[20, 3]]);
    assert.deepEqual(queuedRanges, [[2, 5], [20, 3]],
        `${label} retains two deform packets' disjoint ranges before render`);
    assert.equal(attr.updated, true);
}

const packageJson = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));
assert.ok(packageJson.files.includes('js/gi_refit.js'), 'published package includes the refit dependency');

console.log('gi-deform-refit-smoke: OK');
