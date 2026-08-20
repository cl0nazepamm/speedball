import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(new URL('../js/gi_probes.js', import.meta.url), 'utf8');
const holdLiteral = source.match(/export const GATED_JITTER_HOLD_TICKS\s*=\s*([0-9_]+)/)?.[1];
assert.ok(holdLiteral, 'Gated hold constant must be present');
const holdTicks = Number(holdLiteral.replaceAll('_', ''));
assert.equal(holdTicks, 240, 'Gated holds a basis for four seconds at 60 accepted solves');

const helperMatch = source.match(/export function shouldRotateGatedJitter\([\s\S]*?\n\}/)?.[0];
assert.ok(helperMatch, 'Gated scheduling helper must be present');
const helperSource = helperMatch
    .replace('export function', 'function')
    .replaceAll('GATED_JITTER_HOLD_TICKS', String(holdTicks));
const shouldRotateGatedJitter = Function(
    `${helperSource}; return shouldRotateGatedJitter;`,
)();

function firstRotationTick(probeTotal, probesPerTick) {
    let ticks = 0;
    let coverage = 0;
    for (;;) {
        ticks++;
        if (shouldRotateGatedJitter(ticks, coverage, probeTotal)) return ticks;
        coverage += Math.min(probeTotal, probesPerTick);
    }
}

// Exact regression: boot can fit the whole Sponza field in one batch, while a
// resize-induced cadence backoff or a divisions change makes it partial. Both
// must now retain the same Gated sampling epoch.
assert.equal(firstRotationTick(1_496, 1_536), holdTicks, 'one-batch startup stays Gated');
assert.equal(firstRotationTick(1_496, 768), holdTicks, 'resized partial batch has identical Gated cadence');
assert.equal(firstRotationTick(1_734, 1_536), holdTicks, 'different divisions have identical Gated cadence');
assert.equal(firstRotationTick(10_000, 32), 314, 'very sparse service waits for one full field coverage');

const rotationBlock = source.slice(
    source.indexOf('// (B1) Both cascades share ONE frameJitter.'),
    source.indexOf('// (A5) One-time-per-rebuild prep'),
);
assert.match(rotationBlock, /jitterMode === 'montecarlo'/,
    'Monte Carlo remains a fresh basis every accepted C0 solve');
assert.match(rotationBlock, /shouldRotateGatedJitter\(/,
    'Gated uses the invariant hold helper');
assert.doesNotMatch(rotationBlock, /probeCursor\s*===\s*0/,
    'Gated scheduling cannot depend on circular-batch divisibility');

assert.match(source, /jitterMode: initialJitterMode = 'gated'/,
    'createProbeField initializes Gated before the first build');
assert.match(source, /jitterMode = 'gated'/,
    'installSpeedballGI exposes Gated as the public default');

console.log('GI jitter mode smoke passed');
