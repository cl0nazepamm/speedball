// blas_worker.js — off-thread three-mesh-bvh build for spectral_scene's BLASes.
//
// three-mesh-bvh's packed-tree builder (src/core/build/buildTree.js and the
// modules it imports) is pure typed-array code with no `three` import, so a
// module worker can load it straight from the package's file layout — no
// import map needed inside the worker. The worker receives ONE local-space
// triangle soup (tightly packed xyz positions + a Uint32 triangle index),
// builds the EXACT tree `new MeshBVH(geometry, { targetLeafSize, indirect:
// false })` would (same buildPackedTree code path over a duck-typed bvh whose
// primitive bounds mirror MeshBVH.computePrimitiveBounds), permutes the index
// in place, and transfers both buffers back with the root node buffer.
// spectral_scene.js then flattens that root exactly as it does for a
// main-thread build, so the two paths are byte-for-byte interchangeable.
//
// Everything the worker runs is also exported, so hosts and smoke tests can
// drive the same code synchronously in Node (no Worker global required).

const FLOAT32_EPSILON = Math.pow(2, -24);

// Duck-typed stand-in for the MeshBVH instance buildPackedTree expects:
// primitiveBuffer/primitiveBufferStride (the triangle index it partitions in
// place), one root range (no groups, full draw range), and the per-triangle
// [center, halfExtent] bounds MeshBVH.computePrimitiveBounds writes for a
// non-normalized, non-interleaved float position attribute.
function createDuckBvh(vertexPos, triIndex, triCount) {
    return {
        _roots: null,
        primitiveBuffer: triIndex,
        primitiveBufferStride: 3,
        getRootRanges() {
            return [{ offset: 0, count: triCount }];
        },
        computePrimitiveBounds(offset, count, targetBuffer) {
            if (offset < 0 || count + offset - targetBuffer.offset > targetBuffer.length / 6) {
                throw new Error('blas_worker: compute triangle bounds range is invalid.');
            }
            const writeOffset = targetBuffer.offset;
            for (let i = offset, l = offset + count; i < l; i++) {
                const tri3 = i * 3;
                const boundsIndexOffset = (i - writeOffset) * 6;
                const ai = triIndex[tri3] * 3;
                const bi = triIndex[tri3 + 1] * 3;
                const ci = triIndex[tri3 + 2] * 3;
                for (let el = 0; el < 3; el++) {
                    const a = vertexPos[ai + el];
                    const b = vertexPos[bi + el];
                    const c = vertexPos[ci + el];
                    let min = a;
                    if (b < min) min = b;
                    if (c < min) min = c;
                    let max = a;
                    if (b > max) max = b;
                    if (c > max) max = c;
                    // Same float32-epsilon inflation as MeshBVH so the split
                    // decisions — and therefore the tree — are identical.
                    const halfExtents = (max - min) / 2;
                    const el2 = el * 2;
                    targetBuffer[boundsIndexOffset + el2 + 0] = min + halfExtents;
                    targetBuffer[boundsIndexOffset + el2 + 1] = halfExtents + (Math.abs(min) + halfExtents) * FLOAT32_EPSILON;
                }
            }
            return targetBuffer;
        },
    };
}

// Resolve three-mesh-bvh's build internals from the URL of its buildTree.js.
// Constants.js sits one directory up (src/core/Constants.js) — it carries
// DEFAULT_OPTIONS so a future upstream default change flows through here too.
export async function loadBuildTools(buildTreeUrl) {
    if (!buildTreeUrl) throw new Error('blas_worker: no three-mesh-bvh buildTree.js URL');
    const constantsUrl = new URL('../Constants.js', buildTreeUrl).href;
    const [tree, constants] = await Promise.all([import(buildTreeUrl), import(constantsUrl)]);
    if (typeof tree.buildPackedTree !== 'function') {
        throw new Error('blas_worker: buildTree.js does not export buildPackedTree');
    }
    if (!constants.DEFAULT_OPTIONS || typeof constants.DEFAULT_OPTIONS !== 'object') {
        throw new Error('blas_worker: Constants.js does not export DEFAULT_OPTIONS');
    }
    return { buildPackedTree: tree.buildPackedTree, DEFAULT_OPTIONS: constants.DEFAULT_OPTIONS };
}

// Build one BLAS tree synchronously with the loaded tools. `triIndex` is
// permuted IN PLACE (exactly like MeshBVH does to geometry.index). Returns the
// packed root node buffer and the root bounds [minX,minY,minZ,maxX,maxY,maxZ].
export function buildBlasTreeSync(tools, { vertexPos, triIndex, triCount, targetLeafSize = 8 }) {
    if (!(vertexPos instanceof Float32Array) || !(triIndex instanceof Uint32Array)) {
        throw new Error('blas_worker: vertexPos must be Float32Array and triIndex Uint32Array');
    }
    if (!Number.isInteger(triCount) || triCount <= 0 || triIndex.length < triCount * 3) {
        throw new Error(`blas_worker: invalid triangle count ${triCount}`);
    }
    const bvh = createDuckBvh(vertexPos, triIndex, triCount);
    tools.buildPackedTree(bvh, {
        ...tools.DEFAULT_OPTIONS,
        targetLeafSize,
        indirect: false,
        useSharedArrayBuffer: false,
        range: null,
        onProgress: null,
    });
    const roots = bvh._roots;
    if (!Array.isArray(roots) || roots.length !== 1) {
        throw new Error(`blas_worker: expected one BVH root, got ${Array.isArray(roots) ? roots.length : 'none'}`);
    }
    const root = roots[0];
    // MeshBVH.getBoundingBox() unions the first node's bounds of every root;
    // with one root that is exactly the root node's [min, max].
    const bounds = Float32Array.from(new Float32Array(root, 0, 6));
    return { root, bounds };
}

// ── Worker entry ───────────────────────────────────────────────────────────
// Messages in:  { type: 'init', buildTreeUrl }
//               { type: 'build', id, vertexPos: ArrayBuffer, triIndex: ArrayBuffer, triCount, targetLeafSize }
// Messages out: { type: 'ready' } | { type: 'init-error', message }
//               { type: 'built', id, vertexPos, triIndex, root, bounds }   (buffers transferred back)
//               { type: 'build-error', id, message, vertexPos, triIndex }  (buffers transferred back)
const isWorkerScope = typeof WorkerGlobalScope !== 'undefined'
    && typeof self !== 'undefined'
    && self instanceof WorkerGlobalScope;

if (isWorkerScope) {
    let tools = null;
    self.onmessage = async (event) => {
        const msg = event.data;
        if (!msg || typeof msg !== 'object') return;
        if (msg.type === 'init') {
            try {
                tools = await loadBuildTools(msg.buildTreeUrl);
                self.postMessage({ type: 'ready' });
            } catch (error) {
                self.postMessage({ type: 'init-error', message: String(error?.message || error) });
            }
            return;
        }
        if (msg.type === 'build') {
            const vertexPosBuffer = msg.vertexPos;
            const triIndexBuffer = msg.triIndex;
            try {
                if (!tools) throw new Error('blas_worker: build requested before init');
                const { root, bounds } = buildBlasTreeSync(tools, {
                    vertexPos: new Float32Array(vertexPosBuffer),
                    triIndex: new Uint32Array(triIndexBuffer),
                    triCount: msg.triCount,
                    targetLeafSize: msg.targetLeafSize,
                });
                self.postMessage(
                    { type: 'built', id: msg.id, vertexPos: vertexPosBuffer, triIndex: triIndexBuffer, root, bounds },
                    [vertexPosBuffer, triIndexBuffer, root, bounds.buffer],
                );
            } catch (error) {
                // Hand the soup back so the caller can fall back to a
                // main-thread build without re-gathering it.
                const transfer = [];
                if (vertexPosBuffer instanceof ArrayBuffer && vertexPosBuffer.byteLength > 0) transfer.push(vertexPosBuffer);
                if (triIndexBuffer instanceof ArrayBuffer && triIndexBuffer.byteLength > 0) transfer.push(triIndexBuffer);
                self.postMessage(
                    {
                        type: 'build-error',
                        id: msg.id,
                        message: String(error?.message || error),
                        vertexPos: vertexPosBuffer,
                        triIndex: triIndexBuffer,
                    },
                    transfer,
                );
            }
        }
    };
}
