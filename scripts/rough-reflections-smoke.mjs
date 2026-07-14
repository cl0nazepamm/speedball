import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const probes = await readFile(new URL('../js/gi_probes.js', import.meta.url), 'utf8');
const lights = await readFile(new URL('../js/gi_lights_node.js', import.meta.url), 'utf8');
const install = await readFile(new URL('../js/install.js', import.meta.url), 'utf8');

// Opt-out is the compatibility contract: no rough atlas/buffer allocation and no
// material-lighting graph change unless the creation-time flag is explicitly true.
assert.match(probes, /roughReflections\s*=\s*false/);
assert.match(probes, /roughReflectionsEnabled\s*=\s*roughReflections\s*===\s*true/);
assert.match(probes, /roughReflectionsEnabled\s*\?\s*\(reuse\?\.roughSpecularBuffer/);
assert.match(probes, /roughReflectionsEnabled\s*\?\s*\(reuse\?\.roughSpecularAtlas/);
assert.match(lights, /if \(probe\.roughReflectionsReady\)/);
assert.match(install, /roughReflections\s*=\s*false/);

// Reuse the existing rayData and solve cadence: the blend section must not trace,
// and the steady solve remains exactly trace + blend + upload.
const blend = probes.slice(probes.indexOf('const blendKernel'), probes.indexOf('const clearAtlasKernel'));
assert.ok(blend.includes('rayData.element'));
assert.ok(!blend.includes('traverseClosest('));
assert.ok(!blend.includes('traverseAny('));

const dispatch = probes.match(/await Promise\.all\(\[\s*renderer\.computeAsync\(gpu\.traceKernel\),\s*renderer\.computeAsync\(gpu\.blendKernel\),\s*renderer\.computeAsync\(gpu\.uploadKernel\),\s*\]\);/s);
assert.ok(dispatch, 'steady probe solve must remain exactly three compute dispatches');

// Receiver and history invariants: one shared visibility gather, independent rough
// initialization, and PMREM/environment nodes ordered before the probe composite.
assert.match(probes, /sampleIrradianceAndRough/);
assert.match(probes, /roughAcc\.addAssign\(s\.mul\(w\)\)/);
assert.match(probes, /const sh = select\(sWasEmpty, float\(0\.0\), hEff\)/);
assert.match(probes, /ROUGH_UNSHADED_T_BIAS/);
assert.match(lights, /materialLightings[\s\S]*\.\.\.lightNodes/);
assert.ok(lights.indexOf('materialLightings') < lights.indexOf('lightNodes.push(probe)'));

console.log('rough reflections smoke: ok');
