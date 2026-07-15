# Changelog

All notable changes to Speedball GI are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

- Added opt-in local DDGI reflections. Power-16 glossy and power-8 rough
  radiance/coverage lobes are filtered from the existing probe rays inside the
  existing blend/upload passes: no extra BVH traversal, ray budget, or compute
  dispatch. Standard/Physical receivers reuse the diffuse visibility gather,
  depth-moment parallax-correct each probe lookup, blend the lobes by material
  roughness, sharpen only the reflection probe-blend weights, and composite local
  hits over Three's PMREM through the native
  `context.radiance` path. Distant environment misses and unsupported pure-metal/
  glass probe hits still fall back to PMREM rather than blacking it out.

## [0.6.4] — 2026-07-13

- Fixed unbounded WebGPU storage growth after settled animated-scene rebuilds.
  HALO-GI now disposes obsolete compute nodes before releasing their bindings,
  and explicitly evicts standalone `StorageBufferAttribute` resources from the
  Three r185 attribute manager so GPU buffers and renderer memory bookkeeping
  are actually released.
- Applied the same compute/storage teardown to the spectral path tracer.
- Added a cleanup smoke test and included the shared teardown helper in the npm
  package.

## [0.6.0] — 2026-07-08

- NIR band sensing for the raster path (`setNirSensing(on)` on the install
  handle; granular: probe field `setNirSensing` + `setNirDirectSensing` from
  `gi_lights_node`). Emitter-class-`'ir'` lights (RGB-black, intensity-driven)
  are now simulated in both raster terms instead of leaking or vanishing:
  - GI probes gate class-4 lights in NEE on a `nirGate` uniform — previously
    the promoted white `(k,k,k)` lit the field even in the visible band.
  - The direct term lifts IR lights off the batched `DynamicLightsNode` path
    onto per-light nodes whose `colorNode` is the sensed color
    (white × intensity × `nirGate`) — previously black × intensity = nothing,
    so NV showed GI but no direct light. `light.color` is never mutated;
    toggling the band is a uniform write (no recompile). Shadows still apply.
- Spectral tracer: native-white → D65 correction. The kernel upsamples
  equal-RGB to a flat spectrum (Illuminant E white), which the plain sRGB
  matrix rendered warm; the exact native white is now Bradford-adapted to D65
  and baked into the XYZ→sRGB blit, so equal-RGB scenes come out neutral.
- Spectral tracer: `envBackground` option (setting + per-`setEnvironment`
  override). When off, primary-miss rays return black — the environment stays
  a light source for bounces but is never seen directly by the camera.
- Spectral tracer: roughness 0 is no longer floored — it's a legal delta
  mirror (the glossy lobe degenerates to the exact reflection direction).
- New `speedball-gi/srgb-lut` subpath export — the spectral sRGB→reflectance
  LUT decode for external NIR band consumers.
- All modules import from `three/webgpu` instead of bare `three`: bundlers
  resolve bare `three` to the WebGL core (no `QuadMesh`/`StorageTexture`),
  which broke non-importmap consumers. Importmap consumers are unaffected
  (both specifiers map to the same build).

## [0.5.0] — 2026-07-07

- **The spectral path tracer and the photon caustic engine now SHIP in the
  npm package.** Speedball is the single source for all GPU light transport;
  downstream apps (maxjs, powershot-threejs, sigils) import instead of
  vendoring. New entry points:
  - `speedball-gi/spectral-tracer` — `createSpectralTracer` (progressive
    BVH-traced spectral path tracing, RGB/NV modes, embedded sRGB→reflectance
    LUT — no sidecar files to host).
  - `speedball-gi/caustics` — `createCausticEngine` + receiver/metal presets
    (pure-WebGPU compute photon caustics).
- Caustic engine upstreams (from the sigils fork):
  - Soft t-cull: `setThrowFalloff(1/reach^2)` fades long grazing throws with
    virtual-source divergence so the caustic hugs the geometry; `0` (default)
    keeps the classic open throw.
  - Overlay plane is double-sided and oriented by the full receiver basis
    matrix (right- AND left-handed receiver frames stay valid).
  - `setCasterMesh(mesh, { shaper })`: optional local-space vertex hook —
    `{ position(v, i), normal(n, i) → bool }` — bakes displacement that the
    render material only applies procedurally (e.g. a TSL height-field), so
    photons emit off the same surface the camera sees. Sigil-specific height
    baking now lives in sigils as a shaper, not in the engine.

## [0.4.0] — 2026-07-04

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
