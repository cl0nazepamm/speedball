import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(new URL('../js/gi_probes.js', import.meta.url), 'utf8');

const helperSource = source.match(
    /export function probeBudgetAfterInteraction\([\s\S]*?\n\}/,
)?.[0];
assert.ok(helperSource, 'post-interaction budget helper is present');

// Evaluate the pure helper with the production default constants substituted;
// importing gi_probes.js itself would require a live Three/WebGPU environment.
const executable = helperSource
    .replace('export function', 'function')
    .replace('RAYS_PER_TICK_MIN', '2048')
    .replace('RAYS_PER_TICK_REST_RESUME', '4096');
const probeBudgetAfterInteraction = Function(
    `${executable}; return probeBudgetAfterInteraction;`,
)();

assert.equal(probeBudgetAfterInteraction(98_304), 4_096,
    'release cannot resume at the stale maximum ray budget');
assert.equal(probeBudgetAfterInteraction(3_072), 3_072,
    'an already-safe adaptive budget is preserved');
assert.equal(probeBudgetAfterInteraction(512), 2_048,
    'the solve never falls below its bounded progress floor');

assert.match(source, /const wasMoving = lastMoving;/);
assert.match(source, /if \(!moving && wasMoving\) \{[\s\S]*probeBudgetAfterInteraction\(tickBudgetRays\)/,
    'moving-to-rest transition applies the recovery budget');
assert.match(source, /tickBudgetRays \* 0\.5/,
    'bad cadence backs off aggressively instead of spending a long tail above the floor');
assert.match(source, /tickBudgetRays \+ 1024/,
    'budget growth is gradual after frame cadence recovers');

console.log('GI post-scrub budget smoke passed');
