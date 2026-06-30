# HALO-GI — standalone (WebGPU DDGI + Sky)

A self-contained test page for **HALO-GI** (the max.js BVH-traced dynamic diffuse
global illumination) running **outside 3ds Max** — for fast iteration without the
Max round-trip, and as a basis for a potential upstream three.js example.

It loads the **Sponza** scene, a **Preetham sky** with an azimuth/elevation sun, and
bounces the sunlight through the scene with HALO-GI.

## Run

WebGPU requires a secure context, so serve over http (not `file://`):

```
serve.bat
```
(or `python -m http.server 8777 --bind 127.0.0.1`)

then open **http://127.0.0.1:8777/** in a WebGPU browser (Chrome / Edge 121+).

## Controls (top-right panel)

- **Sky / Sun** — azimuth°, elevation°, turbidity, rayleigh, sun intensity
- **HALO-GI** — enabled, intensity, smoothness
- **exposure**

Drag to orbit, scroll to zoom. GI **holds while you move** and **re-converges when
the view rests** (idle-gated — world-space GI is lossless held static, so a frame
hitch can never land mid-interaction).

## How it wires up

```
renderer.lighting.createNode = (lights) => giLights({...}).setLights(lights); // inject GI into every PBR material
const gi = createProbeField({ renderer, scene });  // BVH-traced DDGI probe field
gi.setEnabled(true);
// render loop:
gi.tick({ idleMs, playing: false });               // idle-gated solve
```

The sky is kept out of the GI BVH with `sky.userData.maxjsVisible = false`.

## Layout

- `index.html` — the demo
- `js/` — HALO-GI modules (copied from `maxjs/web/js`): `gi_probes`, `gi_oct`,
  `spectral_traverse`, `spectral_scene`, `gi_lights_node`, `gi_irradiance_volume`
- `vendor/three-r185/` — three.js r185 (WebGPU build + addons)
- `node_modules/three-mesh-bvh/` — BVH builder
- `Sponza/` — glTF test scene
