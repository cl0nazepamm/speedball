# Speedball GI

Real-time BVH-traced dynamic diffuse global illumination for
[three.js](https://threejs.org/) WebGPU. Speedball keeps solving during motion;
structural rebuilds wait for a safe idle window.

**[Launch the Sponza demo](https://cl0nazepamm.github.io/speedball/)**

## Install

```bash
npm install speedball-gi three three-mesh-bvh
```

Speedball currently targets Three r185:

```json
{
  "three": ">=0.185.0 <0.186.0",
  "three-mesh-bvh": ">=0.9.11 <0.10.0"
}
```

## Quick start

Install GI before the first render or `renderer.setAnimationLoop()` call.

```js
import { installSpeedballGI } from 'speedball-gi';

const gi = installSpeedballGI({
  renderer,
  scene,
  camera,
  intensity: 10,
  divisions: 16,
  rays: 64,
  cascades: 1,
  jitterMode: 'gated',
  onError: console.error,
});

renderer.setAnimationLoop(async () => {
  gi.update();
  await renderer.renderAsync(scene, camera);
});

// Dispose GI before renderer and scene resources.
function dispose() {
  gi.dispose();
  renderer.dispose();
}
```

Continuous solving is enabled by default. `gated` holds a stable sampling basis
for low-latency, mostly flicker-free lighting. `montecarlo` refreshes the basis
every solve for maximum discovery and expects more hysteresis.

## Dynamic scenes

The default integration detects scene changes automatically. Engines and editors
that already know what changed can disable those scans and send exact events:

```js
const gi = installSpeedballGI({
  renderer,
  scene,
  camera,
  autoDetectChanges: false,
});

gi.markTransformsDirty(mesh);
gi.markDeformsDirty(streamedMesh);
gi.markMaterialValuesDirty(material);
gi.forceLightingRefresh();
gi.markTopologyDirty();
```

Use `excludeFromGI(object)` for sky domes, helpers, and gizmos. The installer
supports one active probe field per module instance.

## Documentation

- [Advanced integration](docs/ADVANCED.md) — tuning, reflections, clustered
  lighting, explicit dirty lanes, diagnostics, spectral tracing, and caustics.
- [Changelog](docs/CHANGELOG.md)

Repository demo only: clone this repository, run `node scripts/serve.mjs`, then open
`http://127.0.0.1:8777/`. Demo files are not included in the npm package.

## License

MIT — see [LICENSE](LICENSE).
