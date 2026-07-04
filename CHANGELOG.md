# Changelog

All notable changes to Speedball GI are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

- **Two-level BVH (TLAS/BLAS): moving objects now update GI instantly.**
  Each unique geometry builds ONE local-space BLAS (instanced/shared meshes
  are pooled once — the triangle cap now counts unique triangles); a TLAS
  over instance world-AABBs plus the instance table ride in the tail of the
  materials buffer (no extra storage bindings). Dragging an object is an
  in-place instance/TLAS rewrite + tiny buffer re-upload — no soup rewrite,
  no MeshBVH rebuild, no shader recompile, no frame hitch. Zero setup: the
  field detects transform changes itself (checked every 2 ticks).
- `buildTraversal` consumers: vertex data is LOCAL space now — hit shading
  must use the new `instLocalRay` / `instNormalToWorld` helpers and pass a
  `bestInst` out-var to `traverseClosest`. `U` gains `tlasNodeCount`,
  `instBase`, `tlasBase` uint uniforms (all provided by `buildSpectralScene`).
- Continuous solve now defaults ON at the field level — smooth GI while the
  camera moves out of the box; `setContinuous(false)` restores strict
  idle-gating.

- Added a NIR (near-infrared) spectral layer to the shared scene modules:
  `spectral_scene.js` now emits a photocathode-facing `nirAlbedo` field
  (material slot [25], MAT_STRIDE unchanged at 28) and per-light emitter
  classes (light slot [16]), bumping `LIGHT_STRIDE` from 16 to 17. These
  fields are inert for GI and exist for the night-vision render mode of
  downstream consumers (e.g. the maxjs spectral path tracer).
- New `speedball-gi/spectral-scene` and `speedball-gi/spectral-traverse`
  subpath exports so consumers can import the scene foundation directly
  instead of copying files.
- **`three-mesh-bvh` requirement moved to `>=0.9.4 <0.10.0`** (was capped at
  `<0.9.0`). The stackless-BVH flattener reads MeshBVH's internal `_roots`
  byte layout; 0.9.4 changed the interior right-child word from an absolute
  uint32 index to a parent-relative offset in node units, and the flattener
  now decodes that encoding. The flatten walk is also validated end-to-end
  (bounds-checked offsets, exact node/triangle accounting) so any future
  upstream layout drift throws a descriptive error at build time instead of
  rendering black. Builds also use `maxLeafSize` (the 0.9.7+ name for
  `maxLeafTris`), so no deprecation warnings.

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
