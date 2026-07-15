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
assert.match(probes, /const outE = mix\(eC, filtered, U\.filterStrength\)/);
assert.match(probes, /const GLOSSY_OCT_RES = 16/);
assert.match(probes, /const u = float\(sx\)\.sub\(float\(BORDER\)\)\.add\(0\.5\)\.div\(float\(GLOSSY_OCT_RES\)\)/);
assert.match(probes, /octUV\.x\.mul\(float\(GLOSSY_OCT_RES\)\)/);
assert.match(probes, /octUV\.y\.mul\(float\(GLOSSY_OCT_RES\)\)/);

// CPU analogue of the diffuse 6x6 and glossy 16x16 oct tiles plus their
// canonical mirrored gutters. A smooth asymmetric field exposes either fold.
const BORDER = 1;
const field = ([x, y, z]) => 0.5 + 0.17 * x + 0.23 * y + 0.31 * z;

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
for (const oct of [6, 16]) {
    const sample = makeSampler(oct);
    const xFoldJump = Math.abs(sample([-epsilon, 0, -1]) - sample([epsilon, 0, -1]));
    const yFoldJump = Math.abs(sample([0, -epsilon, -1]) - sample([0, epsilon, -1]));
    assert.ok(xFoldJump < 1e-5, `${oct}x${oct} x fold must be continuous, got ${xFoldJump}`);
    assert.ok(yFoldJump < 1e-5, `${oct}x${oct} y fold must be continuous, got ${yFoldJump}`);
}

console.log('GI oct seam smoke: ok');
