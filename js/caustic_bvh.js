// caustic_bvh.js — tiny dependency-free threaded BVH builder for caustic rays.
//
// Nodes are emitted depth-first. Interior nodes descend to `node + 1`; every
// node stores the first index after its subtree (`miss`) so GPU traversal needs
// neither recursion nor a stack.

export const CAUSTIC_BVH_NODE_STRIDE = 9;

export function buildCausticBvh(positions, indices, { leafSize = 8 } = {}) {
    const triCount = Math.floor(indices.length / 3);
    if (triCount < 1) throw new Error('buildCausticBvh: geometry has no triangles');

    const maxLeafSize = Math.min(255, Math.max(1, Math.floor(leafSize)));
    const bounds = new Float32Array(triCount * 6);
    const centroids = new Float32Array(triCount * 3);
    const order = Array.from({ length: triCount }, (_, i) => i);

    for (let t = 0; t < triCount; t++) {
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        for (let k = 0; k < 3; k++) {
            const v = indices[t * 3 + k] * 3;
            const x = positions[v], y = positions[v + 1], z = positions[v + 2];
            if (x < minX) minX = x; if (x > maxX) maxX = x;
            if (y < minY) minY = y; if (y > maxY) maxY = y;
            if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
        }
        const b = t * 6, c = t * 3;
        bounds[b] = minX; bounds[b + 1] = minY; bounds[b + 2] = minZ;
        bounds[b + 3] = maxX; bounds[b + 4] = maxY; bounds[b + 5] = maxZ;
        centroids[c] = (minX + maxX) * 0.5;
        centroids[c + 1] = (minY + maxY) * 0.5;
        centroids[c + 2] = (minZ + maxZ) * 0.5;
    }

    const records = [];
    function emit(lo, hi) {
        const index = records.length;
        const rec = {
            bounds: [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity],
            miss: 0,
            offset: 0,
            count: 0,
        };
        records.push(rec);

        let cminX = Infinity, cminY = Infinity, cminZ = Infinity;
        let cmaxX = -Infinity, cmaxY = -Infinity, cmaxZ = -Infinity;
        for (let i = lo; i < hi; i++) {
            const tri = order[i], b = tri * 6, c = tri * 3;
            rec.bounds[0] = Math.min(rec.bounds[0], bounds[b]);
            rec.bounds[1] = Math.min(rec.bounds[1], bounds[b + 1]);
            rec.bounds[2] = Math.min(rec.bounds[2], bounds[b + 2]);
            rec.bounds[3] = Math.max(rec.bounds[3], bounds[b + 3]);
            rec.bounds[4] = Math.max(rec.bounds[4], bounds[b + 4]);
            rec.bounds[5] = Math.max(rec.bounds[5], bounds[b + 5]);
            cminX = Math.min(cminX, centroids[c]); cmaxX = Math.max(cmaxX, centroids[c]);
            cminY = Math.min(cminY, centroids[c + 1]); cmaxY = Math.max(cmaxY, centroids[c + 1]);
            cminZ = Math.min(cminZ, centroids[c + 2]); cmaxZ = Math.max(cmaxZ, centroids[c + 2]);
        }

        if (hi - lo <= maxLeafSize) {
            rec.offset = lo;
            rec.count = hi - lo;
            rec.miss = records.length;
            return index;
        }

        const ex = cmaxX - cminX, ey = cmaxY - cminY, ez = cmaxZ - cminZ;
        const axis = ex >= ey && ex >= ez ? 0 : (ey >= ez ? 1 : 2);
        const sorted = order.slice(lo, hi).sort((a, b) => (
            centroids[a * 3 + axis] - centroids[b * 3 + axis]
        ));
        for (let i = 0; i < sorted.length; i++) order[lo + i] = sorted[i];

        const mid = (lo + hi) >> 1;
        emit(lo, mid);
        emit(mid, hi);
        rec.miss = records.length;
        return index;
    }
    emit(0, triCount);

    const nodes = new Float32Array(records.length * CAUSTIC_BVH_NODE_STRIDE);
    for (let i = 0; i < records.length; i++) {
        const rec = records[i], base = i * CAUSTIC_BVH_NODE_STRIDE;
        nodes.set(rec.bounds, base);
        nodes[base + 6] = rec.miss;
        nodes[base + 7] = rec.offset;
        nodes[base + 8] = rec.count;
    }

    return {
        nodes,
        triangles: Uint32Array.from(order),
        nodeCount: records.length,
        leafSize: maxLeafSize,
    };
}
