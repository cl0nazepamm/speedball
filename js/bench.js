// bench.js — BVH-hitch benchmark harness.
//
// The one remaining main-thread stall in SPEEDBALL GI is the synchronous MeshBVH
// (re)build inside buildSpectralScene: any geometry change re-flattens + rebuilds
// the whole triangle soup on the JS thread, which lands as a long frame. This
// harness makes that hitch MEASURABLE and REPRODUCIBLE by churning an object in and
// out of the scene on a fixed cadence — each toggle forces one full-soup rebuild —
// while recording per-frame times so the spike is captured and quantified.
//
// It changes nothing about the GI itself; it just drives the rebuild path and reads
// the clock. Run only when the GPU is otherwise free.

import * as THREE from 'three';

const _now = () => (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();

export function createBenchmark({ scene, gi, hitchThresholdMs = 50, ringSize = 1024 } = {}) {
    if (!scene) throw new Error('createBenchmark: { scene } required');

    let obj = null;
    let present = false;
    let running = false;
    let timer = 0;
    let cycles = 0;
    let triangles = 40000;
    let intervalMs = 1200;

    // frame-time ring buffer
    const ring = new Float64Array(ringSize);
    let ri = 0, filled = 0, last = 0;

    // hitch tracking + churn↔hitch correlation
    let hitchCount = 0;
    let lastHitchMs = 0;
    let lastHitchAt = 0;
    let lastToggleAt = 0;
    let lastToggleKind = '';
    let rebuildHitchMs = 0;    // worst frame within the window after the last toggle
    let windowOpen = false;

    // Build a test object whose triangle count is roughly `tris`.
    // Icosahedron detail d → 20·4^d faces.
    function makeObject(tris) {
        const d = Math.max(0, Math.min(7, Math.round(Math.log(Math.max(20, tris) / 20) / Math.log(4))));
        const geo = new THREE.IcosahedronGeometry(2.5, d);
        const mat = new THREE.MeshStandardMaterial({ color: 0xff3355, roughness: 0.6, metalness: 0 });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.name = '__bench_obj__';
        mesh.position.set(0, 3, 0);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        return mesh;
    }

    function disposeObj() {
        if (obj) { obj.geometry?.dispose?.(); obj.material?.dispose?.(); obj = null; }
    }

    // Toggle the object in/out and force ONE full-soup rebuild. Do NOT markInteraction —
    // we WANT the idle-gated rebuild to fire (camera is expected at rest during a bench run).
    function toggle() {
        if (present) {
            scene.remove(obj);
            present = false;
            lastToggleKind = 'unload';
        } else {
            if (!obj) obj = makeObject(triangles);
            scene.add(obj);
            present = true;
            lastToggleKind = 'load';
        }
        cycles++;
        lastToggleAt = _now();
        windowOpen = true;
        rebuildHitchMs = 0;
        // freshBuild=true → buildDirty → the ~200ms MeshBVH rebuild path (the hitch we hunt).
        gi?.requestRebuild?.(true);
    }

    // Call once per animation frame with the loop's timestamp.
    function frame(nowMs) {
        const t = Number.isFinite(nowMs) ? nowMs : _now();
        if (last > 0) {
            const dt = t - last;
            ring[ri] = dt; ri = (ri + 1) % ringSize; if (filled < ringSize) filled++;
            if (dt > hitchThresholdMs) { hitchCount++; lastHitchMs = dt; lastHitchAt = t; }
            // attribute the worst frame in the ~2s after a toggle to that rebuild
            if (windowOpen) {
                if (dt > rebuildHitchMs) rebuildHitchMs = dt;
                if (t - lastToggleAt > 2000) windowOpen = false;
            }
        }
        last = t;
    }

    function start(opts = {}) {
        if (Number.isFinite(opts.intervalMs)) intervalMs = Math.max(100, opts.intervalMs);
        if (Number.isFinite(opts.triangles)) { triangles = Math.max(20, Math.round(opts.triangles)); disposeObj(); }
        if (running) clearInterval(timer);
        running = true;
        timer = setInterval(toggle, intervalMs);
        toggle(); // fire the first one immediately
    }

    function stop() {
        running = false;
        clearInterval(timer); timer = 0;
        if (present) { scene.remove(obj); present = false; gi?.requestRebuild?.(true); }
    }

    function reset() {
        ri = 0; filled = 0; last = 0;
        hitchCount = 0; lastHitchMs = 0; lastHitchAt = 0; rebuildHitchMs = 0; cycles = 0;
    }

    function setTriangles(n) { if (Number.isFinite(n)) { triangles = Math.max(20, Math.round(n)); disposeObj(); } }
    function setInterval_(ms) { if (Number.isFinite(ms)) { intervalMs = Math.max(100, ms); if (running) start(); } }

    function getStats() {
        let n = filled, mx = 0, sum = 0;
        const arr = new Array(n);
        for (let i = 0; i < n; i++) { const v = ring[i]; arr[i] = v; sum += v; if (v > mx) mx = v; }
        arr.sort((a, b) => a - b);
        const q = (p) => n ? arr[Math.min(n - 1, Math.floor(p * n))] : 0;
        const meanMs = n ? sum / n : 0;
        return {
            running, present, cycles,
            fps: meanMs > 0 ? 1000 / meanMs : 0,
            meanMs, medianMs: q(0.5), p95Ms: q(0.95), p99Ms: q(0.99), maxMs: mx,
            hitchCount, hitchThresholdMs,
            lastHitchMs, lastHitchAgeMs: lastHitchAt ? (_now() - lastHitchAt) : -1,
            rebuildHitchMs,           // worst frame right after the last load/unload
            lastToggleKind, triangles, intervalMs,
        };
    }

    return { start, stop, reset, frame, toggle, setTriangles, setInterval: setInterval_, getStats, isRunning: () => running };
}

export default createBenchmark;
