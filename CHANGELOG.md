# Changelog

All notable changes to Speedball GI are documented here. This project follows
[Semantic Versioning](https://semver.org/).

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
