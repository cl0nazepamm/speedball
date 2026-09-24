// ── Cross-rebuild BLAS cache ────────────────────────────────────────────────
// A structural rebuild used to rebuild EVERY BLAS in the scene; with a cache
// installed a topology change pays only for the geometries it actually
// changed. Entries are keyed by the same structural fingerprint as the
// in-build dedup (geometry identity × attribute identity/version × per-tri
// uber mapping — or the host's stable `userData.speedballGeometryKey`, see
// spectral_scene.js), so any content change misses. The cached core is
// immutable build output — records, soup slices, BVH-ordered materials — and
// every build works on a shallow clone (see the reuse site), so per-build pool
// offsets stamped by a newer build can never corrupt an older build that is
// still draining async work against its own pool. Capacity is bounded by
// total cached triangles, evicted least-recently-used first.
//
// Hosts that switch scenes or recreate probe fields should create ONE cache
// with createBlasCache() and hand it to every createProbeField({ blasCache })
// so it outlives the field; a field without one owns a private cache.
export function createBlasCache({ maxTriangles = 2_000_000 } = {}) {
    return { map: new Map(), maxTriangles, triangles: 0, hits: 0, misses: 0 };
}
