import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { octEncode, octDecode } from '../js/gi_oct.js';

const probes = await readFile(new URL('../js/gi_probes.js', import.meta.url), 'utf8');

// The blend writes directional texels at (i + 0.5) / OCT_RES, whose inverse is
// BORDER + uv * OCT_RES. A second +0.5 shifts every lookup and opens the four
// lower-hemisphere fold seams on smooth world-space normals.
const correctedLookups = probes.match(/add\(float\(BORDER\)\)\.add\(octUV\.[xy]\.mul\(float\(OCT_RES\)\)\)/g) || [];
assert.equal(correctedLookups.length, 4, 'receiver and multibounce lookups must use the corrected oct-atlas inverse');
assert.doesNotMatch(probes, /octUV\.[xy]\.mul\(float\(OCT_RES\)\)\)\.add\(0\.5\)/);

// The upload must source every gutter texel from its canonical mirrored interior
// edge/corner, after applying the same filter as the interior.
assert.match(probes, /const onCorner = onColumnBorder\.and\(onRowBorder\)/);
assert.match(probes, /const probeTexel = probeBase\.add\(sy\.mul\(uint\(TILE\)\)\)\.add\(sx\)/);
const upload = probes.slice(probes.indexOf('const uploadKernel = Fn'), probes.indexOf('// ── CLASSIFY:'));
assert.match(upload,
    /const irradianceCache = workgroupArray\('vec4', OCT_RES \* OCT_RES\)/,
    'upload caches exactly the 36 filtered interior values');
assert.match(upload,
    /irradianceCache\.element\(lane\)\.assign\(vec4\(mix\(eC, filtered, U\.filterStrength\), 1\.0\)\)/,
    'each interior lane filters once into workgroup memory');
assert.match(upload, /workgroupBarrier\(\)/,
    'all filtered interior values are visible before canonical gutter writes');
assert.match(upload,
    /const sourceLane = sy\.sub\(uint\(BORDER\)\)\.mul\(uint\(OCT_RES\)\)\s*\.add\(sx\.sub\(uint\(BORDER\)\)\)/,
    'canonical mirrored coordinates select an interior cache lane');
assert.match(upload,
    /textureStore\(atlas, uvec2\(tx, ty\), irradianceCache\.element\(sourceLane\)\)/,
    'all 64 atlas writes source the cached interior/filter result');
assert.match(probes, /high: Object\.freeze\([^\n]*glossyOct: 8/);
assert.match(probes, /ultra: Object\.freeze\([^\n]*glossyOct: 16/);
assert.match(probes, /const u = float\(sx\)\.sub\(float\(BORDER\)\)\.add\(0\.5\)\.div\(float\(glossyOctRes\)\)/);
assert.match(probes, /octUV\.x\.mul\(float\(this\._glossyOctRes\)\)/);
assert.match(probes, /octUV\.y\.mul\(float\(this\._glossyOctRes\)\)/);

// CPU analogue of the diffuse 6x6 and high/ultra glossy oct tiles plus their
// canonical mirrored gutters. A smooth asymmetric field exposes either fold.
const BORDER = 1;
const field = ([x, y, z]) => 0.5 + 0.17 * x + 0.23 * y + 0.31 * z;

// CPU analogue of the WG64 upload's canonical sourceLane mapping. Every tile
// destination must stay in the 36-entry cache, and all 36 sources must be used.
const OCT_RES = 6;
const TILE = OCT_RES + 2 * BORDER;
const sourceLaneForUpload = (lane) => {
    const lx = lane % TILE;
    const ly = Math.floor(lane / TILE);
    const edge = TILE - 1;
    const lo = BORDER;
    const hi = BORDER + OCT_RES - 1;
    const onLeft = lx === 0;
    const onRight = lx === edge;
    const onTop = ly === 0;
    const onBottom = ly === edge;
    const onColumnBorder = onLeft || onRight;
    const onRowBorder = onTop || onBottom;
    const onCorner = onColumnBorder && onRowBorder;
    const sx = onCorner
        ? (onLeft ? hi : lo)
        : (onRowBorder ? edge - lx : (onColumnBorder ? (onLeft ? lo : hi) : lx));
    const sy = onCorner
        ? (onTop ? hi : lo)
        : (onRowBorder ? (onTop ? lo : hi) : (onColumnBorder ? edge - ly : ly));
    return (sy - BORDER) * OCT_RES + sx - BORDER;
};
const uploadSources = Array.from({ length: TILE * TILE }, (_, lane) => sourceLaneForUpload(lane));
assert.ok(uploadSources.every((lane) => lane >= 0 && lane < OCT_RES * OCT_RES),
    'every canonical upload destination reads within the 36-entry cache');
assert.deepEqual([...new Set(uploadSources)].sort((a, b) => a - b),
    Array.from({ length: OCT_RES * OCT_RES }, (_, lane) => lane),
    'the 64 canonical writes retain coverage of all 36 unique interior sources');

function makeSampler(OCT) {
    const TILE = OCT + 2 * BORDER;
    const atlas = Array.from({ length: TILE }, () => Array(TILE).fill(0));
    for (let y = BORDER; y < BORDER + OCT; y++) {
        for (let x = BORDER; x < BORDER + OCT; x++) {
            atlas[y][x] = field(octDecode((x - BORDER + 0.5) / OCT, (y - BORDER + 0.5) / OCT));
        }
    }
    for (let x = BORDER; x < BORDER + OCT; x++) {
        atlas[0][x] = atlas[BORDER][TILE - 1 - x];
        atlas[TILE - 1][x] = atlas[BORDER + OCT - 1][TILE - 1 - x];
    }
    for (let y = BORDER; y < BORDER + OCT; y++) {
        atlas[y][0] = atlas[TILE - 1 - y][BORDER];
        atlas[y][TILE - 1] = atlas[TILE - 1 - y][BORDER + OCT - 1];
    }
    atlas[0][0] = atlas[BORDER + OCT - 1][BORDER + OCT - 1];
    atlas[0][TILE - 1] = atlas[BORDER + OCT - 1][BORDER];
    atlas[TILE - 1][0] = atlas[BORDER][BORDER + OCT - 1];
    atlas[TILE - 1][TILE - 1] = atlas[BORDER][BORDER];

    return (direction) => {
        const [u, v] = octEncode(...direction);
        const px = BORDER + u * OCT - 0.5;
        const py = BORDER + v * OCT - 0.5;
        const x0 = Math.floor(px);
        const y0 = Math.floor(py);
        const fx = px - x0;
        const fy = py - y0;
        const at = (x, y) => atlas[Math.max(0, Math.min(TILE - 1, y))][Math.max(0, Math.min(TILE - 1, x))];
        return at(x0, y0) * (1 - fx) * (1 - fy)
            + at(x0 + 1, y0) * fx * (1 - fy)
            + at(x0, y0 + 1) * (1 - fx) * fy
            + at(x0 + 1, y0 + 1) * fx * fy;
    };
}

const epsilon = 1e-6;
for (const oct of [6, 8, 16]) {
    const sample = makeSampler(oct);
    const xFoldJump = Math.abs(sample([-epsilon, 0, -1]) - sample([epsilon, 0, -1]));
    const yFoldJump = Math.abs(sample([0, -epsilon, -1]) - sample([0, epsilon, -1]));
    assert.ok(xFoldJump < 1e-5, `${oct}x${oct} x fold must be continuous, got ${xFoldJump}`);
    assert.ok(yFoldJump < 1e-5, `${oct}x${oct} y fold must be continuous, got ${yFoldJump}`);
}

console.log('GI oct seam smoke: ok');
