import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const probes = await readFile(new URL('../js/gi_probes.js', import.meta.url), 'utf8');
const lights = await readFile(new URL('../js/gi_lights_node.js', import.meta.url), 'utf8');
const install = await readFile(new URL('../js/install.js', import.meta.url), 'utf8');
const settings = await readFile(new URL('../js/gi_settings.js', import.meta.url), 'utf8');
const demo = await readFile(new URL('../index.html', import.meta.url), 'utf8');
const readme = await readFile(new URL('../README.md', import.meta.url), 'utf8');

// Structural tier contract. Legacy booleans remain exact and named tiers override them.
for (const [name, values] of Object.entries({
    off: ['rough: false', 'glossy: false', 'glossyOct: 0'],
    rough: ['rough: true', 'glossy: false', 'glossyOct: 0'],
    high: ['rough: true', 'glossy: true', 'glossyOct: 8', 'glossyUpdateInterval: 2', 'roughnessLimit: 1'],
    ultra: ['rough: true', 'glossy: true', 'glossyOct: 16', 'glossyUpdateInterval: 1', 'roughnessLimit: 1'],
})) {
    const tier = probes.match(new RegExp(`${name}: Object\\.freeze\\(\\{([^}]*)\\}`))?.[1] ?? '';
    for (const value of values) assert.ok(tier.includes(value), `${name} must include ${value}`);
}
assert.match(probes, /const fallback = roughReflections === true \? 'ultra' : 'off'/);
assert.match(probes, /const reflectionConfig = resolveReflectionQuality\(reflectionQuality, roughReflections\)/);
assert.match(probes, /const roughReflectionsEnabled = reflectionConfig\.rough/);
assert.match(probes, /const glossyReflectionsEnabled = reflectionConfig\.glossy/);
assert.match(install, /roughReflections\s*=\s*false/);
assert.match(install, /reflectionQuality\s*=\s*null/);
assert.match(install, /reflectionQuality,/);
assert.match(demo, /const REFLECTION_QUALITY_VALUES = new Set\(\['off', 'rough', 'high', 'ultra'\]\)/);
assert.match(demo, /reflectionQuality:\s*params\.giReflectionQuality/);
assert.match(demo, /onReflectionQuality:\s*\(quality\) =>/);
assert.match(demo, /reloadWithStructuralSetting\('reflections', quality, 'high'\)/);
assert.match(demo, /const SCENE_MODE = SCENES\[URL_PARAMS\.get\('scene'\)\]/,
    'the optional clustered comparison remains an install-time URL mode');
assert.doesNotMatch(demo, /reloadWithStructuralSetting\('scene'/,
    'the release deck does not expose a live scene-mode control');

// Off and rough-only do not allocate or dispatch the glossy path.
assert.match(probes, /const roughSpecularBuffer = roughReflectionsEnabled/);
assert.match(probes, /const roughSpecularAtlas = roughReflectionsEnabled/);
assert.match(probes, /const glossySpecularBuffer = glossyReflectionsEnabled/);
assert.match(probes, /const glossyWeightBuffer = glossyReflectionsEnabled/);
assert.match(probes, /const glossySpecularAtlas = glossyReflectionsEnabled/);
assert.match(probes, /const glossyKernel = glossyReflectionsEnabled \? Fn/);
assert.match(probes, /const clearGlossyAtlasKernel = glossyReflectionsEnabled \? Fn/);
assert.match(probes, /if \(!glossyReflectionsEnabled\) \{[\s\S]*C\.glossyAtlasW = C\.glossyAtlasH = 1/);
assert.match(probes, /if \(!this\._roughReflectionsConfigured\) return false/);
assert.match(probes, /this\._glossyReflectionsConfigured && !this\._glossySpecularAtlas\[c\]/);

// Reflections reuse ray scratch; neither lobe resolve may add BVH traversal.
const blend = probes.slice(probes.indexOf('const blendKernel'), probes.indexOf('const glossyKernel'));
const glossy = probes.slice(probes.indexOf('const glossyKernel'), probes.indexOf('// ── CLEAR:'));
for (const source of [blend, glossy]) {
    assert.ok(source.includes('rayData.element'));
    assert.ok(!source.includes('traverseClosest('));
    assert.ok(!source.includes('traverseAny('));
}

// High tier interleaves texels rather than skipping probe batches, and normalizes
// temporal authority for the longer per-texel revisit interval. The workgroup
// cache requires every lane to reach the barrier, so phase selection is a positive
// output guard rather than an invocation-local early return.
assert.match(glossy, /const rayCache = workgroupArray\('vec4', raysPerProbe\)/);
assert.match(glossy, /const dirCache = workgroupArray\('vec4', raysPerProbe\)/);
assert.match(glossy, /loadK\.addAssign\(uint\(PROBE_WORKGROUP_SIZE\)\)/,
    'WG64 lanes cooperatively load the ray cache without overlap');
assert.match(glossy, /workgroupBarrier\(\);[\s\S]*If\(resolvesTexel/);
assert.match(glossy,
    /resolvesTexel = resolvesTexel\.and\(local\.mod\(uint\(glossyUpdateInterval\)\)\.equal\(U\.glossyPhase\)\)/);
assert.doesNotMatch(glossy, /\bReturn\s*\(/,
    'no glossy invocation may return before or after a workgroup barrier');
assert.match(glossy, /U\.hysteresisExponent\.mul\(float\(glossyUpdateInterval\)\)/);
assert.match(probes, /C\.U\.glossyPhase\.value = C\.glossyPhase >>> 0/);
assert.match(probes, /C\.glossyPhase = \(C\.glossyPhase \+ 1\) % glossyUpdateInterval/);
assert.match(probes,
    /const glossyGroupsPerProbe = glossyReflectionsEnabled\s*\? Math\.ceil\(\(glossyTile \* glossyTile\) \/ PROBE_WORKGROUP_SIZE\)/);
assert.match(glossy,
    /updatedCap\(\) \* glossyGroupsPerProbe \* PROBE_WORKGROUP_SIZE,[\s\S]*\[PROBE_WORKGROUP_SIZE\]/,
    'glossy dispatch pads each probe to complete WG64 cache-sharing workgroups');
assert.match(probes,
    /gpu\.glossyKernel\.count = updated \* gpu\.glossyGroupsPerProbe \* PROBE_WORKGROUP_SIZE/,
    'the live glossy dispatch count preserves complete WG64 groups');

// High-tier receiver: continuous glossy trilinear gather, unbiased spatial lookup,
// a live roughness cutoff, and a compile-time per-material opt-out.
assert.doesNotMatch(probes, /dominantGlossy|dominantI/);
assert.match(probes, /const gridF = reflectionP\.sub\(this\.gridMinNode\[c\]\)/);
assert.match(probes, /const f = reflectionP\.sub\(this\.gridMinNode\[1\]\)/);
assert.match(probes, /If\(wantsGlossy/);
assert.match(probes, /If\(roughLobeMix\.greaterThan\(float\(0\.0\)\)/);
assert.match(probes, /material\?\.userData\?\.speedballReflections !== false/);
assert.match(probes, /roughness\.lessThanEqual\(this\.roughnessLimitNode\)/);
assert.match(probes, /setRoughnessLimit/);
assert.match(install, /roughnessLimit,/);
assert.match(demo, /roughnessLimit:\s*1/);
assert.match(settings, /gi\.setRoughnessLimit\?\.\(1\)/);
assert.match(settings, /gi\.setNormalDetail\?\.\(1\)/);
assert.doesNotMatch(settings, /reflection roughness/);
assert.doesNotMatch(settings, /\.name\('normal detail'\)/);
assert.doesNotMatch(demo, /reflection roughness/);
assert.match(readme, /material\.userData\.speedballReflections = false/);

// Existing layer ownership/composite invariants remain intact.
assert.match(lights, /if \(probe\.roughReflectionsReady\)/);
assert.match(probes, /function setReflectionSkyFallback\(on\)/);
assert.match(probes, /const covered = local\.w\.clamp\(0\.0, 1\.0\)\.mul\(reflectionWeight\)/);
assert.match(probes, /builder\.context\.radiance\.mulAssign\(float\(1\.0\)\.sub\(covered\)\)/);
assert.match(probes, /builder\.context\.radiance\.addAssign\(local\.rgb\.mul\(reflectionWeight\)\)/);
assert.match(probes, /reflectionQuality: reflectionConfig\.name/);
assert.match(probes, /glossyProbeGather: glossyReflectionsEnabled \? 8 : 0/);

// Active lobe-loop estimates for one probe: high's 10x10 tile is 69.1% cheaper
// than ultra's 18x18 tile before cadence; two-phase interleaving makes it 84.6%
// cheaper. Dispatch itself is padded to complete WG64 workgroups for barriers.
const ultraTexels = (16 + 2) ** 2;
const highTexels = (8 + 2) ** 2;
assert.equal(ultraTexels, 324);
assert.equal(highTexels, 100);
assert.ok(1 - highTexels / ultraTexels > 0.69);
assert.ok(1 - (highTexels / 2) / ultraTexels > 0.845);
const workgroupSize = 64;
const highDispatchLanes = Math.ceil(highTexels / workgroupSize) * workgroupSize;
const ultraDispatchLanes = Math.ceil(ultraTexels / workgroupSize) * workgroupSize;
assert.equal(highDispatchLanes, 128);
assert.equal(ultraDispatchLanes, 384);
assert.ok(1 - highDispatchLanes / ultraDispatchLanes > 0.66);

console.log('rough reflections smoke: ok');
