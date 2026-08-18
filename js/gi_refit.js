// ── In-place bounds refit for a flattened, threaded BLAS node range ──
//
// spectral_scene.js pools every BLAS into flat bvhNodes/triIndex/vertexData
// buffers (see its header). When a mesh DEFORMS without changing topology
// (streamed vertex buffers, morphs, CPU skinning), the tree STRUCTURE stays
// valid — only the node bounds go stale. This refit rewrites them in place,
// so a deform costs O(verts + nodes) instead of a ~200 ms synchronous
// MeshBVH rebuild on the render thread.
//
// Layout contract (flattenBVHRoot + the pool assembly in spectral_scene.js):
//   • nodes are PRE-ORDER: an interior node's left child is at i+1;
//   • the miss/escape link of the left child IS the right child's index
//     (a node's miss = first slot after its whole subtree, so the left
//     subtree's escape lands exactly on its right sibling);
//   • node stride 8 u32: f32 bounds [minX,minY,minZ,maxX,maxY,maxZ] at +0..5,
//     miss at +6, leaf word at +7 (0xFFFFFFFF = interior, else
//     (triCount << 24) | triOffset with POOL-ABSOLUTE triOffset).
//
// Walking the range in REVERSE therefore visits every child before its
// parent: leaves recompute exact bounds from the (already re-gathered)
// pooled vertices; interiors union their two children. Bounds stay exact
// per refit — they never accumulate slack. Only the PARTITION quality is
// frozen at build-time topology, which for bounded deforms (skinned
// characters, morphs) costs a little traversal efficiency, never
// correctness.
//
// Deliberately dependency-free (no three, no three-mesh-bvh) so node smoke
// tests can exercise the walk without a browser or installed peers.
//
// Returns false — WITHOUT finishing the write — if the threaded-layout
// invariant is violated (right child outside (left, end)); the caller must
// treat that as "layout drifted, full rebuild required", matching
// flattenBVHRoot's fail-loudly policy.
export function createFlatBlasRefitStepper({
    nodesF, nodesU, triIndex, vertexData, root, end,
    nodeStrideU32 = 8, vertexDataStride = 8,
}) {
    const valid = !!nodesF && !!nodesU && !!triIndex && !!vertexData
        && Number.isInteger(root) && Number.isInteger(end)
        && root >= 0 && end > root
        && end * nodeStrideU32 <= nodesU.length
        && end * nodeStrideU32 <= nodesF.length;
    const state = {
        nodesF, nodesU, triIndex, vertexData, root, end,
        nodeStrideU32, vertexDataStride,
        cursor: end - 1,
        leaf: null,
        done: !valid,
        valid,
        processedNodes: 0,
        processedTriangles: 0,
    };

    // Reverse pre-order is naturally resumable: every child has a larger node
    // index than its parent, so any completed suffix remains valid after a
    // yield. A leaf keeps its partial accumulator when the budget expires.
    state.step = ({ deadline = Infinity, now = () => performance.now() } = {}) => {
        if (!state.valid) return false;
        // Reading a high-resolution clock per triangle costs more than the
        // actual bounds math on large soups. Sample once per small work batch;
        // 32 triangles/interiors add only a tiny bounded overshoot while
        // removing nearly all timer overhead from million-triangle refits.
        let workUntilBudgetCheck = 32;
        const budgetReached = () => {
            workUntilBudgetCheck--;
            if (workUntilBudgetCheck > 0) return false;
            workUntilBudgetCheck = 32;
            return now() >= deadline;
        };
        while (state.cursor >= root) {
            const i = state.cursor;
            const base = i * nodeStrideU32;
            const info = nodesU[base + 7];
            if (info !== 0xFFFFFFFF) {
                if (!state.leaf) {
                    const triOff = info & 0x00FFFFFF;
                    const triCnt = info >>> 24;
                    if (triCnt === 0) {
                        // MeshBVH never emits empty leaves. Preserve its stale
                        // finite bounds rather than writing +/-Infinity.
                        state.cursor--;
                        state.processedNodes++;
                        if (budgetReached()) return true;
                        continue;
                    }
                    state.leaf = {
                        next: triOff,
                        end: triOff + triCnt,
                        minX: Infinity, minY: Infinity, minZ: Infinity,
                        maxX: -Infinity, maxY: -Infinity, maxZ: -Infinity,
                    };
                }
                const leaf = state.leaf;
                while (leaf.next < leaf.end) {
                    const t3 = leaf.next * 3;
                    if (t3 < 0 || t3 + 2 >= triIndex.length) {
                        state.valid = false;
                        state.done = true;
                        return false;
                    }
                    for (let k = 0; k < 3; k++) {
                        const dd = triIndex[t3 + k] * vertexDataStride;
                        if (dd < 0 || dd + 2 >= vertexData.length) {
                            state.valid = false;
                            state.done = true;
                            return false;
                        }
                        const x = vertexData[dd], y = vertexData[dd + 1], z = vertexData[dd + 2];
                        if (x < leaf.minX) leaf.minX = x;
                        if (x > leaf.maxX) leaf.maxX = x;
                        if (y < leaf.minY) leaf.minY = y;
                        if (y > leaf.maxY) leaf.maxY = y;
                        if (z < leaf.minZ) leaf.minZ = z;
                        if (z > leaf.maxZ) leaf.maxZ = z;
                    }
                    leaf.next++;
                    state.processedTriangles++;
                    if (leaf.next < leaf.end && budgetReached()) return true;
                }
                nodesF[base] = leaf.minX;
                nodesF[base + 1] = leaf.minY;
                nodesF[base + 2] = leaf.minZ;
                nodesF[base + 3] = leaf.maxX;
                nodesF[base + 4] = leaf.maxY;
                nodesF[base + 5] = leaf.maxZ;
                state.leaf = null;
            } else {
                const l = i + 1;
                if (l >= end) {
                    state.valid = false;
                    state.done = true;
                    return false;
                }
                const r = nodesU[l * nodeStrideU32 + 6]; // left escape = right child
                if (!(r > l && r < end)) {
                    state.valid = false;
                    state.done = true;
                    return false;
                }
                const lb = l * nodeStrideU32, rb = r * nodeStrideU32;
                nodesF[base] = Math.min(nodesF[lb], nodesF[rb]);
                nodesF[base + 1] = Math.min(nodesF[lb + 1], nodesF[rb + 1]);
                nodesF[base + 2] = Math.min(nodesF[lb + 2], nodesF[rb + 2]);
                nodesF[base + 3] = Math.max(nodesF[lb + 3], nodesF[rb + 3]);
                nodesF[base + 4] = Math.max(nodesF[lb + 4], nodesF[rb + 4]);
                nodesF[base + 5] = Math.max(nodesF[lb + 5], nodesF[rb + 5]);
            }
            state.cursor--;
            state.processedNodes++;
            if (state.cursor >= root && budgetReached()) return true;
        }
        state.done = true;
        return true;
    };
    return state;
}

export function refitFlatBlasRange(args) {
    const state = createFlatBlasRefitStepper(args);
    if (!state.valid) return false;
    while (!state.done) {
        if (!state.step({ deadline: Infinity })) return false;
    }
    return true;
}

function defaultBudgetNow() {
    return globalThis.performance?.now?.() ?? Date.now();
}

function defaultBudgetYield() {
    // scheduler.yield() only yields to another task; its continuation may run
    // again before the browser presents a frame. A long refit then obeys the
    // per-slice millisecond cap while still chaining enough slices to turn the
    // post-scrub settle tail into visibly low FPS. A visible viewer gets one
    // slice per animation frame. Hidden/background consumers use a timer so
    // their work cannot stall indefinitely on a throttled rAF.
    if (globalThis.document?.visibilityState !== 'hidden'
        && typeof globalThis.requestAnimationFrame === 'function') {
        return new Promise((resolve) => globalThis.requestAnimationFrame(() => resolve()));
    }
    return new Promise((resolve) => setTimeout(resolve, 0));
}

// Generic latest-state runner used by spectral_scene's large deform path.
// `task.step` may mutate CPU staging arrays, but `commit` is reached only after
// a complete, still-current task. A stale snapshot is discarded and captured
// again after yielding; callers therefore converge to the latest quiet state
// without ever publishing partial GPU ranges.
export async function runLatestBudgetedTask({
    capture,
    createTask,
    validate,
    commit,
    isCancelled = () => false,
    budgetMs = 3,
    now = defaultBudgetNow,
    yieldTask = defaultBudgetYield,
}) {
    const budget = Math.max(0.25, Math.min(3, Number.isFinite(budgetMs) ? budgetMs : 3));
    while (!isCancelled()) {
        const snapshot = capture();
        if (!snapshot || snapshot.invalid) return null;
        const task = createTask(snapshot);
        if (!task) return null;
        let restart = false;
        while (!task.done) {
            const deadline = now() + budget;
            if (task.step({ deadline, now }) === false) return null;
            if (task.done) break;
            await yieldTask();
            if (isCancelled()) return null;
            if (!validate(snapshot)) {
                restart = true;
                break;
            }
        }
        if (restart || !validate(snapshot)) {
            await yieldTask();
            continue;
        }
        const result = commit(snapshot, task);
        if (result?.restart === true) {
            await yieldTask();
            continue;
        }
        return result;
    }
    return null;
}

// Refit a flattened, threaded TLAS without rebuilding or re-sorting it.
//
// `nodes` is the float view of the TLAS only (callers can pass a subarray of a
// larger packed buffer). Its layout is:
//   bounds [0..5], miss [6], instance offset [7], instance count [8].
// A count of zero marks an interior node; leaves cover stable, build-time
// instance slots in `instanceBounds` (six floats per slot by default).
//
// As with the BLAS refit, reverse pre-order visits children before parents.
// Keeping the original partition can make a wildly moved scene less optimal to
// traverse, but the bounds stay exact and correctness never depends on a sort.
// This is the transform hot path: no object arrays, slices, or O(n log n)
// median sorts after the initial build.
export function refitFlatTlasRange({ nodes, instanceBounds, root = 0, end, nodeStrideF32 = 12, instanceBoundsStride = 6 }) {
    if (!nodes || !instanceBounds) return false;
    const nodeEnd = Number.isInteger(end) ? end : Math.floor(nodes.length / nodeStrideF32);
    const instanceCount = Math.floor(instanceBounds.length / instanceBoundsStride);
    if (root < 0 || nodeEnd <= root || nodeEnd * nodeStrideF32 > nodes.length) return false;

    for (let i = nodeEnd - 1; i >= root; i--) {
        const base = i * nodeStrideF32;
        const leafCount = nodes[base + 8];
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        if (leafCount > 0) {
            const off = nodes[base + 7];
            if (!Number.isInteger(off) || !Number.isInteger(leafCount)
                || off < 0 || leafCount < 1 || off + leafCount > instanceCount) return false;
            for (let slot = off, slotEnd = off + leafCount; slot < slotEnd; slot++) {
                const b = slot * instanceBoundsStride;
                const x0 = instanceBounds[b], y0 = instanceBounds[b + 1], z0 = instanceBounds[b + 2];
                const x1 = instanceBounds[b + 3], y1 = instanceBounds[b + 4], z1 = instanceBounds[b + 5];
                if (x0 < minX) minX = x0;
                if (y0 < minY) minY = y0;
                if (z0 < minZ) minZ = z0;
                if (x1 > maxX) maxX = x1;
                if (y1 > maxY) maxY = y1;
                if (z1 > maxZ) maxZ = z1;
            }
        } else {
            const left = i + 1;
            if (left >= nodeEnd) return false;
            const right = nodes[left * nodeStrideF32 + 6];
            if (!Number.isInteger(right) || !(right > left && right < nodeEnd)) return false;
            const lb = left * nodeStrideF32, rb = right * nodeStrideF32;
            minX = Math.min(nodes[lb], nodes[rb]);
            minY = Math.min(nodes[lb + 1], nodes[rb + 1]);
            minZ = Math.min(nodes[lb + 2], nodes[rb + 2]);
            maxX = Math.max(nodes[lb + 3], nodes[rb + 3]);
            maxY = Math.max(nodes[lb + 4], nodes[rb + 4]);
            maxZ = Math.max(nodes[lb + 5], nodes[rb + 5]);
        }
        nodes[base] = minX;
        nodes[base + 1] = minY;
        nodes[base + 2] = minZ;
        nodes[base + 3] = maxX;
        nodes[base + 4] = maxY;
        nodes[base + 5] = maxZ;
    }
    return true;
}

// Build the immutable lookup tables used by the sparse transform lane. TLAS
// topology is frozen between structural rebuilds, so deriving leaf/parent
// ownership once lets a host-provided dirty object update only its instance
// slots and the ancestor chain back to the root.
export function createFlatTlasRefitIndex({
    nodes, instanceBounds, root = 0, end,
    nodeStrideF32 = 12, instanceBoundsStride = 6,
}) {
    if (!nodes || !instanceBounds) return null;
    const nodeEnd = Number.isInteger(end) ? end : Math.floor(nodes.length / nodeStrideF32);
    const instanceCount = Math.floor(instanceBounds.length / instanceBoundsStride);
    if (root < 0 || nodeEnd <= root || nodeEnd * nodeStrideF32 > nodes.length) return null;

    const parentByNode = new Int32Array(nodeEnd);
    const leafBySlot = new Int32Array(instanceCount);
    parentByNode.fill(-1);
    leafBySlot.fill(-1);

    for (let i = root; i < nodeEnd; i++) {
        const base = i * nodeStrideF32;
        const leafCount = nodes[base + 8];
        if (leafCount > 0) {
            const off = nodes[base + 7];
            if (!Number.isInteger(off) || !Number.isInteger(leafCount)
                || off < 0 || leafCount < 1 || off + leafCount > instanceCount) return null;
            for (let slot = off; slot < off + leafCount; slot++) {
                if (leafBySlot[slot] !== -1) return null;
                leafBySlot[slot] = i;
            }
        } else {
            const left = i + 1;
            if (left >= nodeEnd) return null;
            const right = nodes[left * nodeStrideF32 + 6];
            if (!Number.isInteger(right) || !(right > left && right < nodeEnd)) return null;
            if (parentByNode[left] !== -1 || parentByNode[right] !== -1) return null;
            parentByNode[left] = i;
            parentByNode[right] = i;
        }
    }
    for (let slot = 0; slot < instanceCount; slot++) if (leafBySlot[slot] < root) return null;

    return {
        root,
        end: nodeEnd,
        nodeStrideF32,
        instanceBoundsStride,
        parentByNode,
        leafBySlot,
        marks: new Uint32Array(nodeEnd),
        epoch: 0,
        dirtyNodes: [],
    };
}

// Sparse exact TLAS refit. `dirtySlots` are stable build-time TLAS slots whose
// instance bounds have already been rewritten. A dirty leaf is recomputed from
// every slot it owns, then only its unique ancestors are unioned bottom-up.
// Returns the touched node indices in ascending order for ranged GPU uploads.
export function refitFlatTlasDirty({ nodes, instanceBounds, dirtySlots, index }) {
    if (!nodes || !instanceBounds || !index || !dirtySlots) return null;
    const {
        root, end, nodeStrideF32, instanceBoundsStride,
        parentByNode, leafBySlot, marks, dirtyNodes,
    } = index;
    if (end * nodeStrideF32 > nodes.length
        || Math.floor(instanceBounds.length / instanceBoundsStride) !== leafBySlot.length) return null;

    let epoch = (index.epoch + 1) >>> 0;
    if (epoch === 0) {
        marks.fill(0);
        epoch = 1;
    }
    index.epoch = epoch;
    dirtyNodes.length = 0;

    const markNode = (node) => {
        if (node < root || node >= end || marks[node] === epoch) return;
        marks[node] = epoch;
        dirtyNodes.push(node);
    };
    for (const rawSlot of dirtySlots) {
        const slot = Number(rawSlot);
        if (!Number.isInteger(slot) || slot < 0 || slot >= leafBySlot.length) return null;
        let node = leafBySlot[slot];
        if (node < root) return null;
        while (node >= root) {
            markNode(node);
            node = parentByNode[node];
        }
    }
    if (dirtyNodes.length === 0) return [];

    // Pre-order guarantees children have larger indices than parents.
    dirtyNodes.sort((a, b) => b - a);
    for (const i of dirtyNodes) {
        const base = i * nodeStrideF32;
        const leafCount = nodes[base + 8];
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        if (leafCount > 0) {
            const off = nodes[base + 7];
            if (!Number.isInteger(off) || !Number.isInteger(leafCount)
                || off < 0 || off + leafCount > leafBySlot.length) return null;
            for (let slot = off; slot < off + leafCount; slot++) {
                const b = slot * instanceBoundsStride;
                const x0 = instanceBounds[b], y0 = instanceBounds[b + 1], z0 = instanceBounds[b + 2];
                const x1 = instanceBounds[b + 3], y1 = instanceBounds[b + 4], z1 = instanceBounds[b + 5];
                if (x0 < minX) minX = x0;
                if (y0 < minY) minY = y0;
                if (z0 < minZ) minZ = z0;
                if (x1 > maxX) maxX = x1;
                if (y1 > maxY) maxY = y1;
                if (z1 > maxZ) maxZ = z1;
            }
        } else {
            const left = i + 1;
            if (left >= end) return null;
            const right = nodes[left * nodeStrideF32 + 6];
            if (!Number.isInteger(right) || !(right > left && right < end)) return null;
            const lb = left * nodeStrideF32, rb = right * nodeStrideF32;
            minX = Math.min(nodes[lb], nodes[rb]);
            minY = Math.min(nodes[lb + 1], nodes[rb + 1]);
            minZ = Math.min(nodes[lb + 2], nodes[rb + 2]);
            maxX = Math.max(nodes[lb + 3], nodes[rb + 3]);
            maxY = Math.max(nodes[lb + 4], nodes[rb + 4]);
            maxZ = Math.max(nodes[lb + 5], nodes[rb + 5]);
        }
        nodes[base] = minX;
        nodes[base + 1] = minY;
        nodes[base + 2] = minZ;
        nodes[base + 3] = maxX;
        nodes[base + 4] = maxY;
        nodes[base + 5] = maxZ;
    }
    dirtyNodes.sort((a, b) => a - b);
    return dirtyNodes.slice();
}
