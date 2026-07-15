import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const probes = await readFile(new URL('../js/gi_probes.js', import.meta.url), 'utf8');
const lights = await readFile(new URL('../js/gi_lights_node.js', import.meta.url), 'utf8');
const install = await readFile(new URL('../js/install.js', import.meta.url), 'utf8');
const demo = await readFile(new URL('../index.html', import.meta.url), 'utf8');

// Opt-out is the compatibility contract: no rough atlas/buffer allocation and no
// material-lighting graph change unless the creation-time flag is explicitly true.
assert.match(probes, /roughReflections\s*=\s*false/);
assert.match(probes, /roughReflectionsEnabled\s*=\s*roughReflections\s*===\s*true/);
assert.match(probes, /roughReflectionsEnabled\s*\?\s*\(reuse\?\.roughSpecularBuffer/);
assert.match(probes, /roughReflectionsEnabled\s*\?\s*\(reuse\?\.roughSpecularAtlas/);
assert.match(probes, /roughReflectionsEnabled\s*\?\s*\(reuse\?\.glossySpecularBuffer/);
assert.match(probes, /roughReflectionsEnabled\s*\?\s*\(reuse\?\.glossySpecularAtlas/);
assert.match(probes, /roughReflectionsEnabled\s*\?\s*\(reuse\?\.glossyWeightBuffer/);
assert.match(probes, /!this\._roughSpecularAtlas\[c\] \|\| !this\._glossySpecularAtlas\[c\]/);
assert.match(probes, /glossySpecularAtlas: prev\.glossySpecularAtlas/);
assert.match(probes, /glossySpecularBuffer: prev\.glossySpecularBuffer/);
assert.match(probes, /glossyWeightBuffer: prev\.glossyWeightBuffer/);
assert.match(probes, /disposeStorageAttribute\(renderer, g\.glossySpecularBuffer\)/);
assert.match(probes, /disposeStorageAttribute\(renderer, g\.glossyWeightBuffer\)/);
assert.match(probes, /g\.glossySpecularAtlas\?\.dispose\?\.\(\)/);
assert.match(probes, /const clearGlossyAtlasKernel = roughReflectionsEnabled \? Fn/);
assert.match(probes, /textureStore\(glossySpecularAtlas, uvec2\(tx, ty\), resolved\)/);
assert.match(lights, /if \(probe\.roughReflectionsReady\)/);
assert.match(install, /roughReflections\s*=\s*false/);
assert.match(install, /reflectionSkyFallback\s*=\s*false/);
assert.match(install, /reflectionSkyFallback,/);
assert.match(probes, /reflectionSkyFallback\s*=\s*false/);
assert.match(probes, /function setReflectionSkyFallback\(on\)/);
assert.match(probes, /setReflectionSkyFallback,/);
assert.match(probes, /skyConfigured:\s*uniform\(0\.0\)/);
assert.match(probes, /const configured = input !== null && input !== undefined \? 1\.0 : 0\.0/);
assert.match(demo, /reflectionSkyFallback:\s*true/);
assert.match(demo, /excludeFromGI\(metalBall\)/);
assert.match(demo, /metalness:\s*1,\s*roughness:\s*0/);

// Reuse the existing rayData: neither low-frequency blend nor high-resolution
// glossy resolve may trace. The glossy tier adds one resolve dispatch, not rays.
const blend = probes.slice(probes.indexOf('const blendKernel'), probes.indexOf('const glossyKernel'));
const glossy = probes.slice(probes.indexOf('const glossyKernel'), probes.indexOf('// ── CLEAR:'));
assert.ok(blend.includes('rayData.element'));
assert.ok(!blend.includes('traverseClosest('));
assert.ok(!blend.includes('traverseAny('));
assert.ok(glossy.includes('rayData.element'));
assert.ok(!glossy.includes('traverseClosest('));
assert.ok(!glossy.includes('traverseAny('));
assert.match(probes, /if \(gpu\.glossyKernel\) solve\.push\(renderer\.computeAsync\(gpu\.glossyKernel\)\)/);
assert.match(probes, /solve\.push\(renderer\.computeAsync\(gpu\.uploadKernel\)\)/);
const steadySolve = probes.slice(probes.indexOf('// (A6/#1)'), probes.indexOf('await Promise.all(solve)'));
const dispatchOrder = [
    'renderer.computeAsync(gpu.traceKernel)',
    'renderer.computeAsync(gpu.blendKernel)',
    'renderer.computeAsync(gpu.glossyKernel)',
    'renderer.computeAsync(gpu.uploadKernel)',
].map((call) => steadySolve.indexOf(call));
assert.ok(dispatchOrder.every((at) => at >= 0), 'all steady reflection dispatches must be present');
assert.deepEqual([...dispatchOrder].sort((a, b) => a - b), dispatchOrder, 'steady dispatch order must be trace -> blend -> glossy -> upload');
assert.equal(steadySolve.match(/renderer\.computeAsync/g)?.length, 4, 'opt-out has three calls; glossy opt-in adds exactly one');

// Receiver and history invariants: one shared visibility gather, independent lobe
// initialization, and PMREM/environment nodes ordered before the probe composite.
assert.match(probes, /sampleIrradianceAndRough/);
assert.match(probes, /roughAcc\.addAssign\(broad\.mul\(reflectionProbeW\)\)/);
assert.match(probes, /const sh = select\(sWasEmpty, float\(0\.0\), hEff\)/);
assert.match(glossy, /const gh = select\(empty, float\(0\.0\), U\.hysteresis\)/);
assert.match(glossy, /const num = mix\(curNum, prevNum, gh\)/);
assert.match(glossy, /const den = mix\(curDen, prevDen, gh\)/);
assert.match(probes, /ROUGH_UNSHADED_T_BIAS/);
assert.match(lights, /materialLightings[\s\S]*\.\.\.lightNodes/);
assert.ok(lights.indexOf('materialLightings') < lights.indexOf('lightNodes.push(probe)'));
const composite = probes.slice(probes.indexOf('const local = sample.get'), probes.indexOf('        } else {', probes.indexOf('const local = sample.get')));
assert.match(composite, /const covered = local\.w\.clamp\(0\.0, 1\.0\)\.mul\(reflectionWeight\)/);
assert.match(composite, /builder\.context\.radiance\.mulAssign\(float\(1\.0\)\.sub\(covered\)\)/);
assert.match(composite, /builder\.context\.radiance\.addAssign\(local\.rgb\.mul\(reflectionWeight\)\)/);
assert.ok(composite.indexOf('.mulAssign(') < composite.indexOf('.addAssign('), 'premultiplied local radiance must composite over prior radiance');

// Fidelity tier: rough stays on the proven 6x6 cache while glossy owns a
// support-aware 16x16/power-64 cache and independent near-square packing.
assert.match(probes, /const ROUGH_SPECULAR_POWER = 8/);
assert.match(probes, /const GLOSSY_OCT_RES = 16/);
assert.match(probes, /const GLOSSY_SPECULAR_POWER = 64/);
assert.match(probes, /const STORAGE_BINDING_FALLBACK = 128 \* 1024 \* 1024/);
assert.match(probes, /const GLOSSY_HISTORY_BYTES_PER_PROBE = GLOSSY_TILE \* GLOSSY_TILE \* 4 \* Float32Array\.BYTES_PER_ELEMENT/);
assert.match(probes, /maxStorageBufferBindingSize/);
assert.match(probes, /GLOSSY_HISTORY_BYTES_PER_PROBE > maxGlossyBytes/);
assert.match(probes, /C\.glossyTilesX = Math\.max\(1, Math\.ceil\(Math\.sqrt\(C\.probeTotal\)\)\)/);
assert.match(probes, /const probeIndex = px\.add\(py\.mul\(this\.resNode\[c\]\.x\)\)\.add\(pz\.mul\(this\.resNode\[c\]\.x\.mul\(this\.resNode\[c\]\.y\)\)\)/);
assert.match(probes, /const col = probeIndex\.mod\(this\.glossyTilesXNode\[c\]\)/);
assert.match(probes, /const row = floor\(probeIndex\.div\(this\.glossyTilesXNode\[c\]\)\)/);
assert.match(glossy, /const col = probeIndex\.mod\(uint\(C\.glossyTilesX\)\)/);
assert.match(glossy, /const row = probeIndex\.div\(uint\(C\.glossyTilesX\)\)/);
assert.match(blend, /const cd4 = cd2\.mul\(cd2\);\s*const sw = cd4\.mul\(cd4\);/);
assert.match(blend, /const skyEnabled = select\(U\.skyIntensity\.greaterThan\(float\(0\.0\)\), float\(1\.0\), float\(0\.0\)\)/);
assert.match(blend, /const skyValid = U\.reflectionSkyFallback\.mul\(U\.skyConfigured\)\.mul\(skyEnabled\)/);
assert.match(blend, /hitT\.equal\(float\(-1\.0\)\), skyValid/);
assert.match(glossy, /const cd32 = cd16\.mul\(cd16\);\s*const gw = cd32\.mul\(cd32\)/);
assert.match(glossy, /const skyEnabled = select\(U\.skyIntensity\.greaterThan\(float\(0\.0\)\), float\(1\.0\), float\(0\.0\)\)/);
assert.match(glossy, /const skyValid = U\.reflectionSkyFallback\.mul\(U\.skyConfigured\)\.mul\(skyEnabled\)/);
assert.match(glossy, /hitT\.equal\(float\(-1\.0\)\), skyValid/);
assert.match(glossy, /const invRayCount = float\(1\.0 \/ raysPerProbe\)/);
assert.match(glossy, /const curNum = vec4\(gAcc, gHit\)\.mul\(invRayCount\)/);
assert.match(glossy, /const curDen = gWsum\.mul\(invRayCount\)/);
assert.match(probes, /return \{ e, w, probePos \}/);
assert.match(probes, /const reflectionP = positionWorld\.mul\(this\.samplePositionScaleNode\)/);
assert.match(probes, /const P = reflectionP\.add\(stableNormal\.mul/);
assert.match(probes, /const depthR = textureLevel\(this\._depthAtlas\[c\]/);
assert.match(probes, /const insideWeight = smoothstep\(float\(0\.0\), float\(ROUGH_PARALLAX_INSIDE_FADE\), insideRatio\)/);
assert.match(probes, /const parallaxWeight = stableDepth\.mul\(validHit\)\.mul\(insideWeight\)/);
assert.match(probes, /const correctedR = normalize\(q\.add\(Rn\.mul\(rayT\.max\(float\(0\.0\)\)\)\)\)/);
assert.match(probes, /const sampleR = normalize\(mix\(Rn, correctedR, parallaxWeight\)\)/);
assert.match(probes, /textureLevel\(\s*this\._glossySpecularAtlas\[c\]/);
assert.match(probes, /this\._glossyTileUV\(px, py, pz, octSampleR, c\)/);
assert.match(probes, /const reflectionProbeW = w\.mul\(w\)/);
assert.match(probes, /const confidence = mix\(float\(0\.2\), float\(1\.0\), parallaxWeight\)/);
assert.match(probes, /roughRadiance: mix\(glossyResolved, roughResolved, roughLobeMix\)/);
assert.match(probes, /ROUGH_PARALLAX_VAR_START/);

// Pure arithmetic regressions for limits/history/packing; no WebGPU workload.
const bindingLimit = 128 * 1024 * 1024;
const glossyBytesPerProbe = 18 * 18 * 4 * Float32Array.BYTES_PER_ELEMENT;
const maxGrid = [32, 32, 32];
while (maxGrid[0] * maxGrid[1] * maxGrid[2] * glossyBytesPerProbe > bindingLimit) {
    for (let i = 0; i < 3; i++) maxGrid[i] = Math.max(2, Math.floor(maxGrid[i] * 0.85));
}
assert.deepEqual(maxGrid, [27, 27, 27]);
assert.ok(maxGrid[0] * maxGrid[1] * maxGrid[2] * glossyBytesPerProbe <= bindingLimit);

const historyResolve = (rays) => {
    const h = 0.9;
    const currentNum = (rays * 0.75) / rays;
    const currentDen = rays / rays;
    const num = currentNum * (1 - h) + 0.5 * h;
    const den = currentDen * (1 - h) + 1.0 * h;
    return num / den;
};
assert.equal(historyResolve(64), historyResolve(256), 'history authority must be ray-count invariant');

const compositeOver = (prior, localPremul, coverage, intensity) => prior * (1 - coverage * intensity) + localPremul * intensity;
assert.equal(compositeOver(0.8, 0.2, 1, 0), 0.8);
assert.equal(compositeOver(0.8, 0, 0, 1), 0.8);
assert.equal(compositeOver(0.8, 0.2, 1, 1), 0.2);
const reflectionValidity = (hitT, fallback, skyConfigured, skyIntensity) => hitT >= 0
    ? 1
    : Number(hitT === -1 && fallback && skyConfigured && skyIntensity > 0);
assert.equal(reflectionValidity(2, false, false, 0), 1, 'local hits are always valid');
assert.equal(reflectionValidity(-1, true, true, 1), 1, 'configured SH may own true misses');
assert.equal(reflectionValidity(-1, true, false, 1), 0, 'unconfigured SH cannot cover prior radiance');
assert.equal(reflectionValidity(-1, true, true, 0), 0, 'disabled SH intensity cannot cover prior radiance');
assert.equal(reflectionValidity(-3, true, true, 1), 0, 'encoded unshadeable hits remain transparent');

for (const total of [2, 17, 216, 32 * 32 * 32]) {
    const tilesX = Math.ceil(Math.sqrt(total));
    const tilesY = Math.ceil(total / tilesX);
    for (const probe of new Set([0, Math.min(tilesX - 1, total - 1), Math.min(tilesX, total - 1), total - 1])) {
        const col = probe % tilesX;
        const row = Math.floor(probe / tilesX);
        assert.equal(row * tilesX + col, probe);
        assert.ok(col < tilesX && row < tilesY);
    }
    assert.ok((total - 1) % tilesX < tilesX, 'partial final glossy row must stay in bounds');
}

console.log('rough reflections smoke: ok');
