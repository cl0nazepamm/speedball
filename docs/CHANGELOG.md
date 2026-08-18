# Changelog

All notable changes to Speedball GI are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

- Cross-rebuild BLAS cache: a structural rebuild now pays only for the
  geometries that actually changed. Cached cores (BVH records, soup slices,
  BVH-ordered materials) are keyed by geometry identity × attribute
  identity/version × per-tri material mapping, reused via per-build clones so
  an in-flight older build can never be corrupted, and bounded by a
  least-recently-used triangle budget. Measured on the churn harness at rest:
  worst frame per topology/enter/leave action drops from ~240 ms to ~110 ms
  (the remainder is pool upload + kernel work, not BLAS builds).
- three-mesh-bvh housekeeping: demo CDN pins moved to 0.9.14, the build option
  renamed to `targetLeafSize` (kills the per-BLAS deprecation warning during
  churn), and the peer floor raised to `>=0.9.11` — older 0.9.x silently
  ignored the leaf-size option under its previous names. Local BLAS bounds now
  come from the MeshBVH build itself instead of a separate `computeBoundingBox`
  pass: one less O(V) scan per BLAS per rebuild, and the bounds no longer
  wrongly include the origin when invisible triangles leave zero-filled
  duplicate vertex slots.
- Added an event-driven dynamic-scene lane for games and realtime DCC viewers.
  `markTransformsDirty(object | { object, instanceIndex })` coalesces host
  transform packets and rewrites only the affected stable instance records plus
  their unique TLAS ancestor chains during continuous motion. Unknown/untraced
  transform targets are cheap no-ops; the rest-only full-scene signatures remain
  as compatibility fallback. `InstancedMesh` allocation capacity is now resident
  in the TLAS, so active-count changes and slot recycling within that capacity do
  not rebuild the BVH. Added `markDeformsDirty()`, `markTopologyDirty()`, and the
  generic `notifySceneChange()` event surface.
- Packed material traversal flags in slot 26 and added an exact opaque fast path.
  Closest-hit and shadow traversal now skip candidate-hit UV interpolation and
  alpha texture sampling for constant opaque materials; non-shadow-blocking
  transmission candidates also exit before alpha work.

- Added structural local-reflection quality tiers: `off`, `rough`, `high`, and
  `ultra`. The Sponza demo now uses `high`, which resolves an 8x8 glossy cache
  on interleaved two-phase texels and blends eight glossy receiver probes; legacy
  `roughReflections: true` maps to the unchanged 16x16/every-solve `ultra`
  quality contract. The live receiver cutoff is `roughnessLimit` (pinned at 1,
  hidden from the Quality panel). Added the compile-time
  `material.userData.speedballReflections = false` opt-out. The
  Sponza Quality panel now exposes the four structural tiers and reloads while
  retaining scene/query state when the selection changes.
- Fixed a hard glossy ring in `high`: normal bias now affects visibility only,
  while glossy grid/cascade selection uses the unbiased surface position. Removed
  the discontinuous single-dominant-probe shortcut from the high receiver.
- Light reactivity now rides THROUGH motion: the light-signature check and
  `refreshLights()` moved out of the rest-only gate in `tick()`. Lights are
  refresh-class (in-place buffer refill — no BVH, no compile, bounded
  lights-only traverse with deadbands), so light edits and animated lights
  update the field live during interaction and playback. Explicit host transform
  and deform packets now have their own continuous lane; fallback xform/deform
  scans and structural rebuilds remain rest-only.
- IR illuminator gain: a scalar trim on emitter-class-4 lights in the sensed
  band, on top of the existing on/off gate, uniform-driven (no recompiles).
  Three consumers, one knob per host: `setNirIlluminatorGain` (direct raster
  term, gi_lights_node — also multiplied into custom-lights-node IR nodes),
  `field.setNirGain` (probes' NEE), and the spectral tracer's `setNirGain`
  (multiplies the 850 nm emission band; resets accumulation). The install
  handle's `setNirGain` drives both raster consumers.
- Authored NIR albedo (`userData.nirAlbedo`, material slot [25]) no longer
  paints tagged materials flat past the red edge. The tag now sets the
  material's NIR *level* and the albedo map keeps supplying per-texel spatial
  variation: the kernel passes the pre-map scalar color into `jhReflectance`,
  which scales the authored value by the JH ratio texel/flat (clamped to
  [0, 1], flat guarded at 1e-3). Materials without an albedo map — and all
  untagged materials — shade exactly as before, and scenes with no albedo
  atlas compile an unchanged shader.
- Frame-paced large deform refits so their 2 ms slices now yield through the
  next animation frame instead of chaining `scheduler.yield()` continuations
  ahead of paint. Normal-only settle packets update packed shading data without
  refitting BLAS/TLAS bounds. Speedball also resumes from interaction at a conservative
  ray budget, can throttle down to 2,048 rays on weak GPUs, and grows gradually
  after frame cadence recovers; stopped timeline scrubs therefore cannot turn
  the deferred catch-up lane into a sustained ~12 fps tail.
- Added a deform fast path so same-topology vertex animation (streamed vertex
  buffers, CPU skinning, morph bakes) never triggers the ~200 ms synchronous
  MeshBVH rebuild. `built.updateDeforms()` re-gathers the deformed BLAS's
  slice of the pooled vertex data from the live attributes and refits the
  flattened node bounds in place (`gi_refit.js` — reverse pre-order walk over
  the threaded layout; the tree structure stays build-time), then rewrites the
  instance/TLAS tail via the existing `updateTransforms` path. The probe field
  now splits its reactivity signatures: STRUCTURE (counts + index identity →
  debounced full rebuild) no longer hashes `position.version`; a new DEFORM
  signature (`position`/`normal` identity + versions, checked every 12 idle ticks) routes
  vertex motion to the in-place refit and re-uploads only the touched buffer
  ranges (`updateRanges` when the three build supports them). Previously any
  pause in a vertex stream (scrub stop, frame step, held pose, animation end)
  armed the settle debounce and landed a full-scene MeshBVH rebuild on the
  render thread — a repeating ~200 ms hitch during skinned-mesh workflows. GI
  refreshes the latest deforming pose on its idle cadence after interaction;
  it does not perform scene scans in the playback/input lane. SkinnedMesh
  objects are excluded from all geometry
  signatures, mirroring their exclusion from the BVH build, so GPU-skinned
  meshes can never schedule pointless rebuilds. Temporal policy is untouched:
  deform refreshes re-converge through the bounded per-texel change detector,
  no reactive burst. Async probe-scene builds now validate a monotonic scene
  invalidation generation and the structure signature across material-map
  extraction, catch up vertex/transform refits before install, and retry at
  idle without losing an update that arrived while the build was yielding.
  Deform refits also validate source index identity/version/count before any
  writes, so connectivity edits fail closed to the structural rebuild lane.
  Disposing or replacing a probe field now invalidates an async texture build
  and frees its uninstalled maps before its continuation can publish resources.
- Replaced the moving-instance TLAS rebuild/sort with an exact in-place refit
  of the frozen build-time partition. Live transforms now rewrite stable
  instance slots, refit leaf/interior bounds in reverse pre-order, and upload
  only the dynamic materials tail. Instance-count/layout drift fails closed to
  the structural rebuild lane.
- Path tracing now retains its built scene and exposes typed transform/deform/
  light invalidation. Topology-stable timeline updates use ranged buffer uploads and
  reset accumulation without rebuilding MeshBVHs; untyped/structural edits keep
  the conservative full-build behavior. Same-count light edits rewrite only the
  packed light storage; light-count drift fails closed. Scene revisions preserve
  edits that arrive while material extraction is yielding.
- Included `js/gi_refit.js` in the published package whitelist.

## [0.6.6] — 2026-07-20

- Made hysteresis normalization variance-bounded: the time-normalization exponent
  now clamps at 1, so one update can never blend in more fresh Monte-Carlo noise
  than the slider admits at 60 Hz — for every policy branch (diffuse noise/change,
  depth, rough, glossy) and every service cadence. 0.6.5 removed this bound to
  keep per-second decay exactly rate-invariant; the cost was sparse service (low
  frame rates, throttled ray budgets, large-grid round-robin revisits, alternating
  cascades) blending most of a fresh noisy ray set per revisit — the field
  dissolved into flicker exactly when the machine was struggling. Sparse service
  now converges slower in wall-clock instead of noisier; at and above 60 Hz the
  per-second decay remains rate-invariant and high-refresh exponents stay
  unclamped.
- Removed the global reactive/low-hysteresis burst. Light, sky, transform, and
  trace-side-knob edits no longer drop history authority for ~1.25 s — that read
  as "flicker for a second, then settle" (a visible fade-out/in on every slider
  touch). The temporal policy is constant at all times; edits re-converge through
  the bounded per-texel change detector, so a change transitions smoothly at the
  steady rate instead of pumping. `advanceReactiveTicks` (a debug/test helper
  export, never part of the package entry point) is gone.
- `setDepthSharpness` now accepts 0 (uniform-ish depth weighting; internally
  floored at 0.01 because `pow(0,0)` is indeterminate in WGSL and would poison
  the depth history with NaN). Demo defaults retuned: depth sharpness 0,
  Chebyshev strength 0.8.

## [0.6.5] — 2026-07-20

- Fixed hysteresis normalization across render rates. Adaptive diffuse and depth
  policies are formed in the 60 Hz reference domain before their final retention
  is time-normalized. Rough and glossy reflections now share a dedicated steady /
  noisy reference retention that remains a true elapsed-time semigroup instead of
  inheriting diffuse's nonlinear change detector. Glossy numerator/support use the
  same coefficient, and zero-coverage rough texels no longer re-arm as uninitialized
  and bypass history on the next sparse hit. Each cascade uses its own accepted
  solve cadence and fractional probe revisit rate, removing the near-full-batch
  discontinuity that made high-throughput machines flicker more. The reactive fade
  is elapsed-time based, and normalization no longer clamps either high-refresh or
  sparse-revisit exponents.
- Added opt-in local DDGI reflections without adding reflection rays or BVH
  traversal. The power-8 rough lobe stays in the existing 6x6 probe blend, while
  smooth materials use a separately packed 16x16 power-64 glossy cache. Its
  support-aware numerator/denominator history resolves in one additional compute
  dispatch and converges sparse sharp samples without giving weak ray sets equal
  authority.
- Standard/Physical receivers reuse the diffuse visibility gather,
  depth-moment-parallax-correct each probe lookup, blend the two lobes by material
  roughness, and composite coverage through Three's native `context.radiance`
  path. Three still owns PMREM, metallic F0, Fresnel, and DFG shading, while a
  later SSR pass remains free to overlay its own confident hits.
- Added explicit `reflectionSkyFallback` creation and runtime controls. It defaults
  off so true probe misses leave Three's prior radiance unchanged; scenes with only
  a `setSky()` SH sky can opt in to use that already-traced radiance as the distant
  reflection layer. Runtime ownership changes reconverge temporally.
  Unsupported non-emissive pure-metal/glass hits remain transparent to the prior
  reflection layer instead of becoming black occluders.
- Reflection grids now respect both the device's 2D texture limit and its storage
  buffer binding limit, and glossy history is normalized per ray so changing the
  live ray count cannot skew temporal authority.
- Updated the Sponza diagnostic ball to a receiver-only `metalness: 1`,
  `roughness: 0` target and enabled SH reflection fallback for its procedural-sky,
  no-environment configuration.

## [0.6.4] — 2026-07-13

- Fixed unbounded WebGPU storage growth after settled animated-scene rebuilds.
  Speedball GI now disposes obsolete compute nodes before releasing their bindings,
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
