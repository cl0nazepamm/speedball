# Speedball GI

BVH-traced **dynamic diffuse global illumination** (DDGI) for
[three.js](https://threejs.org/) WebGPU.

Best for medium scale scenes. It is prone to leaking but it is continuous and can be used in timelapse/day night cycle scenarios.


## Hysteresis
Hysteresis is the main slider for radiosity fade in/out.

It is normalized by default over each cascade's accepted solve cadence and fractional round-robin revisit interval. Adaptive diffuse/depth history is normalized only after its final reference-rate retention is formed. Rough and glossy reflection caches use one dedicated steady/noisy retention for their complete stored state (including glossy numerator and support), so their wall-time decay remains identical across refresh rates. A valid zero-coverage rough texel stays initialized instead of accepting the next sparse hit without history.
The normalization is bounded on the slow side: one update never blends in more fresh noise than the slider admits at 60 Hz, no matter how sparse the service cadence gets (low frame rates, throttled ray budgets, large-grid round-robin revisits). Sparse service converges slower in wall-clock instead of flickering. At and above 60 Hz the per-second decay stays rate-invariant; the Hysteresis slider remains the explicit stability/convergence tradeoff.

The live demo exposes a **normalize hysteresis** switch so you can turn the normalization off and compare against the raw per-update value.

## Launch live demos

**[▶ Sponza GI](https://cl0nazepamm.github.io/speedball/)** · **[▶ Glass dispersion](https://cl0nazepamm.github.io/speedball/dispersion.html)** · **[▶ Clustered lighting](https://cl0nazepamm.github.io/speedball/clustered.html)**

Locally: `npm start` then open `http://127.0.0.1:8777/`, `/dispersion.html`, or `/clustered.html`.

Requires a WebGPU-capable browser (Chrome/Edge stable; Safari 26+).

# Installation

```bash
npm install speedball-gi three three-mesh-bvh
```

## Quickstart

```js
import { installSpeedballGI } from 'speedball-gi';

// At SETUP before the first render / renderer.setAnimationLoop():
const gi = installSpeedballGI({
  renderer, scene, camera,
  reflectionQuality: 'high', // off | rough | high | ultra; reuses DDGI rays
  // reflectionSkyFallback: true, // only when setSky() should replace a missing environment map
});

// In your render loop, once per frame:
gi.update();
```

**Install before the first render / `setAnimationLoop()`.** Speedball folds a
GI aware lights node into every lit material at compile time. If the render loop
has already run, three has cached a non-GI lights node and GI will fail.

## Helpers

- **`excludeFromGI(object)`** — keep an object (and its subtree) out of the GI
  trace and the auto-fit bounds. Use it for sky domes, gizmos, and probe
  helpers — anything that shouldn't bounce light or inflate the grid.
- **`prepareMaterialsForGI(scene)`** — normalize materials so metals actually
  bounce diffuse GI. The trace path kills Lambert bounce on metals, so an
  all-metal import reads as black GI. Opt-in; mutates materials in place. You can
  also pass `prepareMaterials: true` to `installSpeedballGI`.

## Local DDGI reflections

Choose a structural **`reflectionQuality`** at creation time. Every tier builds
local-radiance lobes from rays Speedball already traces, adding no reflection rays
or BVH traversal:

- **`off`** — zero reflection buffers, atlases, compute, or material sampling.
- **`rough`** — the stable power-8 lobe in the compact 6x6 cache only.
- **`high`** — rough plus an 8x8 power-64 glossy cache. Glossy texels are
  interleaved over two solves and smooth receivers sample their dominant probe.
- **`ultra`** — the legacy `roughReflections: true` path: 16x16 glossy resolution,
  every texel every solve, and eight glossy receiver probes.

The old boolean remains compatible: `roughReflections: false` maps to `off`, and
`roughReflections: true` maps to `ultra`. A named tier overrides the boolean.
The bundled Sponza demo exposes the four modes under **SPEEDBALL GI → Quality →
reflection quality**. Changing it reloads the page because the tier changes GPU
buffers, bindings, compute kernels, and material graphs.

The physical receiver reuses the diffuse gather's probe visibility, applies
depth-moment parallax correction to each reflection lookup, and samples only the
lobe(s) required by material roughness. Local reflections run through the full
roughness range (`roughnessLimit` 1). Override at creation with `roughnessLimit` or
live with `gi.setRoughnessLimit(0..1)`. Set
`material.userData.speedballReflections = false` before material compilation for a
zero-sampling per-material opt-out (then set `material.needsUpdate = true` if changed
later). The result stays in Three's native
`context.radiance` path, so Standard/Physical BRDF, metallic F0, Fresnel, and DFG
remain Three's responsibility.

Reflection layers stay explicit and composable:

1. `scene.environment` / `material.envMap` supplies distant radiance through
   Three's EnvironmentNode.
2. Speedball composites local DDGI coverage over that radiance.
3. An SSR pass can overlay its screen-space hits afterward using its own confidence.

By default, true probe-ray misses leave the prior radiance unchanged, so PMREM stays
visible there and a later SSR pass can independently overlay its own hits. For a
scene that calls `setSky()` but deliberately has no environment map, pass
**`reflectionSkyFallback: true`** (or call
`gi.setReflectionSkyFallback(true)`) to fill those misses from the same SH-9 sky.
Keep it off when PMREM or another reflection layer owns the distant environment.
Changing this ownership at runtime reconverges through the normal temporal history;
set it at creation when the layer boundary must be established before first solve.

The whole feature is opt-in, so existing integrations keep their allocation,
shader, and image path. `gi.setReflectionIntensity(0..1)` changes its live
contribution; intensity zero skips receiver taps but structural `off` is the mode
that also removes reflection compute and memory.

## Clustered lighting (secondary mode)

Pass **`clusteredLighting: true`** (three r185+) to draw thousands of
non-shadowed point lights cheaply. The batched raster lights node is replaced by
`GiClusteredLightsNode` — three's Forward+ clustered addon (compute-culled
screen-tile × depth-slice light lists) with the same GI injection and IR-emitter
seams — so the direct term stops caring how many small point lights the scene
carries (the `lights` batch caps no longer apply). Directional, spot, and
shadow-casting lights keep the stock per-light path.

The GI lane switches to a fixed importance-budgeted light arena: the probes'
NEE shades the `MAX_LIGHTS` (64) most important records — ranked by peak power ×
spot solid-angle × proximity to the probe volume, directionals always kept —
and light-count changes land in a count uniform + in-place refill, **never** a
BVH rebuild. Pass an object to tune the cluster grid:
`clusteredLighting: { maxLights: 1024, tileSize: 32, zSlices: 24, maxLightsPerCluster: 64 }`.

The default (`false`) keeps the primary batched path byte-identical to previous
releases. See `clustered.html` for the no-sun Sponza demo — 50 lights by
default, with a stress slider to 500 that exercises the importance cut and the
rebuild-free count changes.

## Limitations

- **WebGPU-only**
- **Install timing is sharp** — install before the first render / animation loop. Late install may need an explicit material recompile pass.
- **Material support is approximate(WIP)** — the trace path uses a flattened scene
  representation. Standard PBR-ish materials are the target; exotic node graphs,
  alpha/transmission edge cases, and tiny normal-map detail won't all bounce
  exactly like final shading.
- **Probe reflections are approximate** — the glossy lobe is parallax-corrected and
  stable off-screen, but its angular/probe resolution is not a pixel-accurate mirror
  or transmission path. Non-emissive pure metal/glass *hits inside the traced
  scene* cannot be shaded by the Lambert DDGI ray, so they leave PMREM visible
  instead of becoming black local occluders; use SSR/PT when exact mirror detail
  is required.
- **Best for small to medium scale scenes** — very large worlds or many separated islands
  can waste probes unless bounds and cascades are curated.
- **Cascades:** — Don't even bother with cascades if you are not using Chromium. Additionally cascaded grid will require higher ray count to look as smooth as single grid probes.
- **Convergence is temporal** — loads, light edits, and geometry changes fade in
  over a few frames instead of snapping instantly. Similar to Lumen.

## Beyond DDGI: the full light-transport surface

Speedball is the single source for all of its GPU light transport — downstream
apps (powershot-threejs, sigils) import these entry points rather than
vendoring files:

- **`speedball-gi/spectral-tracer`** — `createSpectralTracer`: progressive
  BVH-traced spectral path tracing (RGB and NV/night-vision modes). The
  sRGB→reflectance LUT ships embedded (`speedball-gi/srgb-lut`), so there are
  no sidecar files to host.
- **`speedball-gi/caustics`** — `createCausticEngine` plus receiver/metal
  presets: pure-WebGPU compute photon caustics with analytic and mesh-emission
  casters, soft t-cull (`setThrowFalloff`), and a `setCasterMesh(mesh,
  { shaper })` hook for baking procedural vertex displacement into photon
  emission. Pass `mode: 'refract'` (plus `ior` / `dispersion` / `thickness`)
  for glass caustics — same splat pipeline, BVH-accelerated through-mesh Snell
  refraction, Cauchy-style chromatic R/G/B grids, high-photon convergence
  controls (`photonBudget`, `resolveInterval`, `setTargetPhotons`), light/floor
  response controls (`setLightIntensity`, `setLightColor`,
  `setReceiverAlbedo`), and optional
  spotlight-cone matching via `setLightCone(direction, angleRadians, penumbra)`.
- **`speedball-gi/spectral-scene`** / **`speedball-gi/spectral-traverse`** —
  the shared scene foundation (scene → flat BVH/material/light buffers; TSL
  traversal + spectral shading emitters).

All of these work from a plain CDN import map too (e.g. jsDelivr:
`https://cdn.jsdelivr.net/npm/speedball-gi@0.5.0/js/index.js`)

Light records are stride 17 floats (slot [16] = emitter class) and material
records stride 28 (slot [25] = NIR albedo); these extra fields are inert for
GI and exist for the night-vision render mode of downstream consumers.

## Changelog

Release history lives in [docs/CHANGELOG.md](docs/CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE).
