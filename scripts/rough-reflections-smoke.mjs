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
assert.match(probes, /roughReflectionsEnabled\s*\?\s*\(reuse\?\.glossySpecularBuffer/);
assert.match(probes, /roughReflectionsEnabled\s*\?\s*\(reuse\?\.glossySpecularAtlas/);
assert.match(probes, /!this\._roughSpecularAtlas\[c\] \|\| !this\._glossySpecularAtlas\[c\]/);
assert.match(probes, /glossySpecularAtlas: prev\.glossySpecularAtlas/);
assert.match(probes, /glossySpecularBuffer: prev\.glossySpecularBuffer/);
assert.match(probes, /disposeStorageAttribute\(renderer, g\.glossySpecularBuffer\)/);
assert.match(probes, /g\.glossySpecularAtlas\?\.dispose\?\.\(\)/);
assert.match(probes, /textureStore\(glossySpecularAtlas, uvec2\(tx, ty\), vec4\(0\.0\)\)/);
assert.match(probes, /textureStore\(glossySpecularAtlas, uvec2\(tx, ty\), g\)/);
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

// Receiver and history invariants: one shared visibility gather, independent lobe
// initialization, and PMREM/environment nodes ordered before the probe composite.
assert.match(probes, /sampleIrradianceAndRough/);
assert.match(probes, /roughAcc\.addAssign\(s\.mul\(reflectionProbeW\)\)/);
assert.match(probes, /const sh = select\(sWasEmpty, float\(0\.0\), hEff\)/);
assert.match(probes, /const gh = select\(gWasEmpty, float\(0\.0\), hEff\)/);
assert.match(probes, /ROUGH_UNSHADED_T_BIAS/);
assert.match(lights, /materialLightings[\s\S]*\.\.\.lightNodes/);
assert.ok(lights.indexOf('materialLightings') < lights.indexOf('lightNodes.push(probe)'));

// Fidelity tier: the local lobe is narrower than the original power-4 solve, and
// each probe reprojects that lookup through its existing directional depth moments.
// This must remain receiver-side reuse, not a new trace or compute pass.
assert.match(probes, /const ROUGH_SPECULAR_POWER = 8/);
assert.match(probes, /const GLOSSY_SPECULAR_POWER = 16/);
assert.match(blend, /const cd4 = cd2\.mul\(cd2\);\s*const sw = cd4\.mul\(cd4\);/);
assert.match(blend, /const gw = sw\.mul\(sw\);/);
assert.match(probes, /return \{ e, w, probePos \}/);
assert.match(probes, /const reflectionP = positionWorld\.mul\(this\.samplePositionScaleNode\)/);
assert.match(probes, /const P = reflectionP\.add\(stableNormal\.mul/);
assert.match(probes, /const depthR = textureLevel\(this\._depthAtlas\[c\]/);
assert.match(probes, /const insideWeight = smoothstep\(float\(0\.0\), float\(ROUGH_PARALLAX_INSIDE_FADE\), insideRatio\)/);
assert.match(probes, /const parallaxWeight = stableDepth\.mul\(validHit\)\.mul\(insideWeight\)/);
assert.match(probes, /const correctedR = normalize\(q\.add\(Rn\.mul\(rayT\.max\(float\(0\.0\)\)\)\)\)/);
assert.match(probes, /const sampleR = normalize\(mix\(Rn, correctedR, parallaxWeight\)\)/);
assert.match(probes, /textureLevel\(this\._glossySpecularAtlas\[c\]/);
assert.match(probes, /s\.assign\(mix\(s, broad, roughLobeMix\)\)/);
assert.match(probes, /const reflectionProbeW = w\.mul\(w\)/);
assert.match(probes, /roughRadiance: roughAcc\.div\(reflectionWsum\.max\(float\(1e-4\)\)\)/);
assert.match(probes, /ROUGH_PARALLAX_VAR_START/);

console.log('rough reflections smoke: ok');
