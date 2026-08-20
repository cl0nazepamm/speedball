import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(new URL('../js/gi_probes.js', import.meta.url), 'utf8');
const installSource = readFileSync(new URL('../js/install.js', import.meta.url), 'utf8');
const rotationStart = source.indexOf('const rotateRaySet =');
const rotationEnd = source.indexOf('// Keep every stochastic input', rotationStart);
assert.ok(rotationStart >= 0 && rotationEnd > rotationStart, 'sampling rotation gate must be extractable');
const rotationBlock = source.slice(rotationStart, rotationEnd);
assert.match(rotationBlock, /jitterMode === 'montecarlo'/,
    'Monte Carlo remains a fresh basis every accepted C0 solve');
assert.match(rotationBlock, /const rotateRaySet = ci === 0 && jitterMode === 'montecarlo';/,
    'only Monte Carlo can enter the sampling-epoch rotation block');
assert.doesNotMatch(rotationBlock, /gated|probeCursor|probeCoverageSinceRot|ticksSinceRot|hold/i,
    'Gated has no timer, cursor, coverage, or batch-size rotation path');
assert.doesNotMatch(source, /GATED_JITTER_HOLD_TICKS|shouldRotateGatedJitter|probeCoverageSinceRot/,
    'no delayed Gated rotation machinery may remain in core');

assert.match(source, /jitterMode: initialJitterMode = 'gated'/,
    'createProbeField initializes Gated before the first build');
assert.match(installSource, /jitterMode = 'gated'/,
    'installSpeedballGI exposes Gated as the public default');

console.log('GI jitter mode smoke passed');
