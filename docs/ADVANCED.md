# Advanced integration

This document covers engine-level Speedball GI integration and the package's
lower-level light-transport surfaces. The normal application path is the concise
`installSpeedballGI()` example in the project README.

## Installer ownership and lifecycle

Call `installSpeedballGI()` before the first render. It installs the renderer's
GI-aware lights-node factory and owns that hook together with the probe field.
Always call the installer handle's `dispose()` before disposing renderer or scene
resources.

The stable handle exposes lifecycle, tuning, scene-dirty notifications, field
queries, reflections, and NIR controls. Raw probe-node access, manual ticking,
rebuild controls, pacing controls, and GPU diagnostics live under `gi.advanced`.
That namespace is intentionally unstable and intended for adapters and test
harnesses. Integrations that need complete ownership can instead import
`createProbeField()` and `giLights` directly.

Asynchronous field failures do not reject the render loop. Pass `onError(error)`
to route them into host telemetry; it fires once per uninterrupted failure streak.

Creation-time structural options include `divisions`, `rays`, `cascades`,
`reflectionQuality`, and `clusteredLighting`. Grid, ray, and cascade changes also
have rebuild-gated setters. Reflection quality and clustered lighting remain
creation-time-only because they change GPU bindings and material graphs.

## Hysteresis and sampling modes

Hysteresis is the main stability/convergence control. It is normalized by default
over each cascade's accepted solve cadence and fractional round-robin revisit
interval. One sparse update never blends in more noise than the selected value
admits at 60 Hz. At higher refresh rates, wall-time decay remains rate-invariant.

The two supported sampling profiles are:

- **Gated** holds its ray and emitter-visibility basis indefinitely. Its default
  hysteresis is `0.60` for low-latency, mostly flicker-free lighting.
- **Monte Carlo** re-jitters every accepted solve. Its default hysteresis is
  `0.90` to absorb the fresh sample set.

`setJitterMode()` remembers explicit hysteresis overrides independently for each
mode. `setHysteresisNormalization(false)` exposes the raw per-update behavior.

## Explicit scene changes

Event-complete hosts should install with `autoDetectChanges: false`. Explicit
packets continue to run during motion while compatibility scene-signature scans
are skipped.

```js
gi.markTransformsDirty(mesh);
gi.markTransformsDirty({ object: crowd, instanceIndex: 37 });
gi.markDeformsDirty(streamedMesh);
gi.markMaterialValuesDirty(material);
gi.notifySceneChange({ type: 'transform', object: mesh });
gi.notifySceneChange({ type: 'deform', object: streamedMesh });
gi.notifySceneChange({ type: 'material', material });
gi.notifySceneChange({ type: 'topology' });
```

Transform packets are coalesced during continuous motion. A traced object rewrites
only its stable instance records and unique TLAS ancestor chain. Cameras, helpers,
excluded objects, and other untraced targets are cheap no-ops.

`InstancedMesh` allocation capacity is reserved in the TLAS at build time. Changes
within `instanceMatrix.count` need only `markTransformsDirty()`. Capacity growth,
new or removed geometry, connectivity changes, and material reassignment are
structural and require `markTopologyDirty()`.

Use `forceLightingRefresh()` for known light changes. Use
`markMaterialValuesDirty()` for scalar/color material edits that keep the same
material and texture bindings.

## Helpers

- `excludeFromGI(object)` excludes an object and its subtree from tracing and
  automatic bounds.
- `prepareMaterialsForGI(scene)` can normalize an imported all-metal scene so it
  contributes diffuse bounce. It mutates materials and is opt-in; the installer
  also accepts `prepareMaterials: true`.
- `setBounds(box)` overrides automatic field bounds.
- `setVolumes(volumes)` supplies curated volume data to the low-level field.

## Local DDGI reflections

Choose `reflectionQuality` at creation. Reflection lobes reuse DDGI rays and add
no reflection BVH traversal:

- `off` allocates no reflection buffers, atlases, compute, or receiver sampling.
- `rough` enables the compact stable rough lobe.
- `high` adds an interleaved 8x8 glossy cache and continuous eight-probe gather.
- `ultra` uses the legacy 16x16, full-rate glossy path.

The legacy `roughReflections: false` and `true` values map to `off` and `ultra`.
A named quality tier takes precedence.

Local reflections use Three's native `context.radiance` path, leaving Standard
and Physical BRDF, metallic F0, Fresnel, and DFG to Three. The receiver applies
probe visibility and depth-moment parallax correction. Runtime controls include
`setReflectionIntensity()` and `setRoughnessLimit()`.

Set `material.userData.speedballReflections = false` before material compilation
for a zero-sampling material opt-out. Recompile the material if this flag changes
later.

Reflection ownership remains composable:

1. `scene.environment` or `material.envMap` supplies distant radiance.
2. Speedball composites local DDGI coverage over it.
3. SSR can overlay screen-space hits afterward.

Probe-ray misses preserve the prior radiance by default. In a scene with an
explicit Speedball sky but no environment map, enable `reflectionSkyFallback` to
fill misses from the same SH-9 sky. Keep it disabled when PMREM or another system
owns distant reflections.

## Clustered lighting

`clusteredLighting: true` replaces the batched dynamic raster lights node with
Three's Forward+ clustered node. It is intended for many non-shadowed point lights;
directional, spot, and shadow-casting lights retain their stock paths.

The GI lane keeps a fixed 64-light arena and selects the most important records by
power, spot solid angle, and proximity to the probe volume. Light-count changes
rewrite the arena and count uniform without rebuilding the BVH.

```js
const gi = installSpeedballGI({
  renderer,
  scene,
  camera,
  clusteredLighting: {
    maxLights: 1024,
    tileSize: 32,
    zSlices: 24,
    maxLightsPerCluster: 64,
  },
});
```

## Additional package surfaces

Speedball also ships lower-level light-transport entry points:

- `speedball-gi/spectral-tracer` exports `createSpectralTracer`, a progressive
  BVH-traced spectral path tracer with RGB and NIR/night-vision modes.
- `speedball-gi/spectral-scene` and `speedball-gi/spectral-traverse` expose the
  shared flattened scene, BVH, materials, lights, and traversal foundation.
- `speedball-gi/srgb-lut` contains the embedded sRGB-to-reflectance LUT.
- `speedball-gi/caustics` exports `createCausticEngine` and receiver/caster presets.

The caustic engine supports analytic and mesh emission, rigid caster updates with
`setCasterTransform()`, and procedural deformation through `setCasterMesh(mesh,
{ shaper })`. Refraction mode uses BVH-accelerated through-mesh Snell traversal
and Cauchy-style chromatic R/G/B grids. Response controls include
`setLightIntensity`, `setLightColor`, `setReceiverAlbedo`, `setStrength`, and
`setLightCone(direction, angleRadians, penumbra)`.

Plain CDN import maps can use:

```text
https://cdn.jsdelivr.net/npm/speedball-gi@0.7.0/js/index.js
```

Packed light records use 17 floats; slot 16 stores emitter class. Material records
use 28 floats; slot 25 stores NIR albedo and slot 26 stores traversal flags.

## Limitations

- WebGPU only.
- Tested against Three `>=0.185.0 <0.186.0`; WebGPU/TSL and cleanup code touches
  revision-specific renderer APIs.
- One active probe field per module instance.
- Standard/Physical-style materials are the target. Exotic node graphs,
  transmission, alpha edge cases, and tiny normal-map detail remain approximate.
- Probe reflections are stable and parallax-corrected, but are not pixel-accurate
  mirrors or transmission. Use SSR for exact specular detail.
- Large worlds and separated islands can waste probes without curated bounds.
- Two-cascade mode currently targets Chromium WebGPU and needs more rays than one
  grid for comparable smoothness.
- Convergence is temporal; scene and light changes fade into the field.
