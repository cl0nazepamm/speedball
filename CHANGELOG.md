# Changelog

All notable changes to Speedball GI are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-07-03

- Hysteresis is now frame-rate normalized over the real per-probe update
  interval (tick dt × round-robin pass length), bounded so large grids and slow
  machines converge as fast as possible without dissolving into noise. New
  `setHysteresisNormalization(on)` / `getHysteresisNormalization()` to compare
  against the raw per-update value.
- Safari: cascaded mode compiles again — the cascaded receiver gather is now a
  real WGSL loop instead of two inlined 8-tap subtrees, fitting WebKit's
  8192-byte pipeline-variable budget.
- Cascaded receiver cost: each cascade is sampled only where its blend weight
  is live, so most pixels pay single-grid cost and only the border band pays
  both cascades.
- Default hysteresis is 0.6 (was 0.9) — normalization makes it feel consistent
  across machines, so the snappier default wins.

## [0.1.0] — 2026-07-02

Initial public release.

- Real-time, BVH-traced dynamic diffuse GI (DDGI) for three.js WebGPU:
  octahedral irradiance probes, infinite bounce, continuous (Lumen-style)
  convergence.
- One-call setup via `installSpeedballGI({ renderer, scene, camera })` plus a
  per-frame `gi.update()`; idle-gated so it never hitches the frame.
- Scene helpers: `excludeFromGI(object)` and `prepareMaterialsForGI(scene)`.
- Cascaded probe grid (single-grid or two-cascade), live-tunable intensity,
  divisions, rays, and hysteresis.
- Sponza demo scene.
