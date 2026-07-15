# Speedball GI

BVH-traced **dynamic diffuse global illumination** (DDGI) for
[three.js](https://threejs.org/) WebGPU.

Best for medium scale scenes. It is prone to leaking but it is continuous and can be used in timelapse/day night cycle scenarios.


## Hysteresis
Hysteresis is the main slider for radiosity fade in/out.

It is normalized by default over the real interval between updates of a probe (frame rate × how many ticks the round-robin scan needs to revisit it), so the temporal blend feels similar across machines, browsers, and grid sizes. 
With slow machines or very large grids it converges as fast as it can without dissolving into per-update noise.

The live demo exposes a **normalize hysteresis** switch so you can turn the normalization off and compare against the raw per-update value.

## Launch live demos

**[▶ Sponza GI](https://cl0nazepamm.github.io/speedball/)** · **[▶ Glass dispersion](https://cl0nazepamm.github.io/speedball/dispersion.html)**

Locally: `npm start` then open `http://127.0.0.1:8777/` or `/dispersion.html`.

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
  roughReflections: true, // optional glossy + rough local reflections; reuses DDGI rays
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

Pass **`roughReflections: true`** at creation time to build rough and glossy
local-radiance lobes from the rays Speedball already traces. It adds no reflection
rays or BVH traversal. Diffuse, depth, and the stable power-8 rough lobe keep the
compact 6x6 octahedral cache; smooth materials use a separate 16x16, power-64
glossy cache with support-aware temporal history. That high-resolution resolve is
one additional dispatch per steady probe solve while the opt-in feature is enabled;
new atlas allocations also receive a one-time clear.

The physical receiver reuses the diffuse gather's probe visibility, applies
depth-moment parallax correction to each reflection lookup, and samples only the
lobe(s) required by material roughness. The result stays in Three's native
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
shader, and image path. Its live contribution is
`gi.setReflectionIntensity(0..1)`.

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
apps (maxjs, powershot-threejs, sigils) import these entry points rather than
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
  for glass caustics — same splat pipeline, Snell thin-slab bend, chromatic
  R/G/B grids so dispersion fans on the receiver.
- **`speedball-gi/spectral-scene`** / **`speedball-gi/spectral-traverse`** —
  the shared scene foundation (scene → flat BVH/material/light buffers; TSL
  traversal + spectral shading emitters).

All of these work from a plain CDN import map too (e.g. jsDelivr:
`https://cdn.jsdelivr.net/npm/speedball-gi@0.5.0/js/index.js`) — that is how
maxjs consumes them without being an npm package.

Light records are stride 17 floats (slot [16] = emitter class) and material
records stride 28 (slot [25] = NIR albedo); these extra fields are inert for
GI and exist for the night-vision render mode of downstream consumers.

## License

MIT — see [LICENSE](LICENSE).
