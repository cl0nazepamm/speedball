import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { buildCausticBvh, CAUSTIC_BVH_NODE_STRIDE } from '../js/caustic_bvh.js';

const engine = await readFile(new URL('../js/caustic_engine.js', import.meta.url), 'utf8');
const readme = await readFile(new URL('../README.md', import.meta.url), 'utf8');
const demo = await readFile(new URL('../index.html', import.meta.url), 'utf8');
const manifest = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));

// Production glass preset: more convergence than the old demo, but not the
// expensive 1024/240k burst path.
assert.match(engine, /const DEFAULT_GLASS_TARGET_PHOTONS = 3_000_000/);
assert.match(engine, /const DEFAULT_GLASS_PHOTONS_PER_UPDATE = 90_000/);
assert.match(engine, /const DEFAULT_GLASS_RESOLVE_INTERVAL = 2/);
assert.match(engine, /photonBudget = undefined/);
assert.match(engine, /resolveInterval = undefined/);
assert.match(engine, /photonTarget = cleanPhotonCount/);
assert.match(engine, /setTargetPhotons/);
assert.match(engine, /get targetPhotons\(\) \{ return photonTarget; \}/);


// Intensity math is not peak-normalized: density is normalized by photon count
// and receiver area, while display radiance responds to light and floor color.
assert.doesNotMatch(engine, /atomicMax/);
assert.match(engine, /sampleScale: uniform\(1\)/);
assert.match(engine, /lightPower: uniform\(params\.lightPower\)/);
assert.match(engine, /lightColor: uniform\(new THREE\.Vector3\(\.\.\.params\.lightColor\)\)/);
assert.match(engine, /receiverAlbedo: uniform\(new THREE\.Vector3\(\.\.\.params\.receiverAlbedo\)\)/);
assert.match(engine, /densityNormBase \/ Math\.max\(1, totalPhotons\)/);
assert.match(engine, /U\.lightColor\.mul\(U\.receiverAlbedo\)\.mul\(U\.tint\)[\s\S]*\.mul\(U\.lightPower\)\.mul\(float\(CAUSTIC_RESPONSE_SCALE \/ PI\)\)/);
assert.match(engine, /function setLightIntensity\(intensity, referenceIntensity = DEFAULT_LIGHT_INTENSITY\)/);
assert.match(engine, /function setLightColor\(r, g, b\)/);
assert.match(engine, /function setReceiverAlbedo\(r, g, b\)/);
assert.match(engine, /overlayMat\.depthTest = true/);

// Mesh exit rays use a threaded BVH rather than an O(triangle-count) scan.
assert.match(engine, /buildCausticBvh\(lpos, idxArr\)/);
assert.match(engine, /const leafTriCount = uint\(meshAccel\.element/);
assert.match(engine, /cursor\.assign\(miss\)/);
assert.doesNotMatch(engine, /end: uint\(triCount\)[\s\S]{0,200}const base = t\.mul/);

const cubePositions = new Float32Array([
    -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1,
    -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1,
]);
const cubeIndices = new Uint32Array([
    0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
    0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
    0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5,
]);
const cubeBvh = buildCausticBvh(cubePositions, cubeIndices, { leafSize: 2 });
assert.ok(cubeBvh.nodeCount > 1);
assert.deepEqual(Array.from(cubeBvh.nodes.subarray(0, 6)), [-1, -1, -1, 1, 1, 1]);
const visited = [];
function walkBvh(node) {
    const base = node * CAUSTIC_BVH_NODE_STRIDE;
    const miss = cubeBvh.nodes[base + 6];
    const offset = cubeBvh.nodes[base + 7];
    const count = cubeBvh.nodes[base + 8];
    assert.ok(miss > node && miss <= cubeBvh.nodeCount, 'threaded miss link must move forward');
    if (count > 0) {
        assert.ok(count <= 2);
        for (let i = 0; i < count; i++) visited.push(cubeBvh.triangles[offset + i]);
    } else {
        const right = walkBvh(node + 1);
        assert.equal(walkBvh(right), miss, 'children must exactly fill the parent subtree');
    }
    return miss;
}
assert.equal(walkBvh(0), cubeBvh.nodeCount);
assert.deepEqual(visited.sort((a, b) => a - b), Array.from({ length: 12 }, (_, i) => i));

// Spotlight caustics should obey the same cone as the visible SpotLight.
assert.match(engine, /spotDir: uniform\(new THREE\.Vector3\(0, -1, 0\)\)/);
assert.match(engine, /const spotWeight = \(dirFromLight\) =>/);
assert.match(engine, /setLightCone\(direction, angleRad = Math\.PI \/ 2, penumbra = 0\)/);

// Resolve-only controls must still refresh after convergence without retracing.
assert.match(engine, /needsDensityResolve = true, needsBloomResolve = true, needsFinalResolve = true/);
assert.match(engine, /batchesSinceResolve >= params\.resolveInterval/);
assert.match(engine, /function markFinalDirty\(\) \{ needsFinalResolve = true; forceResolve = true; \}/);
assert.match(engine, /function setBloom\(v\) \{[\s\S]*?markFinalDirty\(\);\s*\}/);
assert.match(engine, /function setIor\(v\) \{[\s\S]*?if \(Math\.abs\(params\.ior - next\) < 1e-6\) return;/);

// Bloom is thresholded in physical-response space once, not multiplied by the
// light response a second time during resolve.
assert.doesNotMatch(engine, /bloomB\.element\(idx\)[^\n]*responsePeak/);

// Area-weighted sampling is an integral: preserve the demo calibration while
// scaling photon energy with the world-space caster surface area.
assert.match(engine, /casterAreaScale: uniform\(1\)/);
assert.match(engine, /U\.casterAreaScale\.value = acc \/ REFERENCE_CASTER_AREA/);
assert.match(engine, /ndl\.mul\(cone\)\.mul\(U\.casterAreaScale\)/);

// The RGB band split is Cauchy-like: n grows with 1/lambda^2, so blue bends
// more than red while green stays at the nominal IOR.
assert.match(engine, /const FRAUNHOFER_C_NM = 656\.3/);
assert.match(engine, /const FRAUNHOFER_D_NM = 587\.6/);
assert.match(engine, /const FRAUNHOFER_F_NM = 486\.1/);
assert.match(engine, /const DISPERSION_IOR_SPAN = 0\.09/);
assert.match(engine, /const nRed = max\(float\(1\.01\), nCenter\.add\(iorSpan\.mul\(float\(DISPERSION_RED_WEIGHT\)\)\)\)/);
assert.match(engine, /const nBlue = max\(float\(1\.01\), nCenter\.add\(iorSpan\.mul\(float\(DISPERSION_BLUE_WEIGHT\)\)\)\)/);
assert.match(engine, /const absorption = exp\(pathLen\.mul\(max\(U\.thickness, float\(0\.02\)\)\)\.mul\(float\(-0\.18\)\)\)/);

const C = 656.3;
const D = 587.6;
const F = 486.1;
const invC = 1 / (C * C);
const invD = 1 / (D * D);
const invF = 1 / (F * F);
const denom = invF - invC;
const redWeight = (invC - invD) / denom;
const greenWeight = 0;
const blueWeight = (invF - invD) / denom;
const span = 0.24 * 0.09;
const nRed = 1.52 + span * redWeight;
const nGreen = 1.52 + span * greenWeight;
const nBlue = 1.52 + span * blueWeight;

assert.ok(redWeight < 0);
assert.equal(greenWeight, 0);
assert.ok(blueWeight > 0);
assert.ok(nRed < nGreen && nGreen < nBlue);
assert.ok(nBlue - nGreen > nGreen - nRed, 'Cauchy split should pull blue farther than red');
assert.ok(Math.abs((nBlue - nRed) - span) < 1e-12);

assert.match(readme, /BVH-accelerated\s+through-mesh Snell/);
assert.match(readme, /Cauchy-style chromatic R\/G\/B grids/);
assert.match(readme, /setLightIntensity/);
assert.match(readme, /setReceiverAlbedo/);
assert.match(readme, /setLightCone\(direction, angleRadians, penumbra\)/);

// Caustics remain a supported library subpath, but the release Sponza page is
// deliberately the compact emission + metal GI demo while projection is revised.
assert.equal(manifest.exports['./caustics'], './js/caustic_engine.js');
assert.match(demo, /id="ball-emissive"/);
assert.match(demo, /id="ball-metal"/);
assert.doesNotMatch(demo, /id="ball-glass"/);
assert.doesNotMatch(demo, /createCausticEngine|caustic_engine\.js|causticPass/);

console.log('caustics smoke: ok');
