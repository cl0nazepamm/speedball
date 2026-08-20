import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(new URL('../js/gi_probes.js', import.meta.url), 'utf8');

const extractHelper = (name) => {
    const helper = source.match(new RegExp(`export function ${name}\\([\\s\\S]*?\\n\\}`))?.[0];
    assert.ok(helper, `${name} helper is present`);
    return helper;
};
const compileHelper = (name, substitutions = []) => {
    let executable = extractHelper(name).replace('export function', 'function');
    for (const [from, to] of substitutions) executable = executable.replaceAll(from, to);
    return Function(`${executable}; return ${name};`)();
};

// Evaluate the pure helper with the production default constants substituted;
// importing gi_probes.js itself would require a live Three/WebGPU environment.
const probeBudgetAfterInteraction = compileHelper('probeBudgetAfterInteraction', [
    ['RAYS_PER_TICK_MIN', '2048'],
    ['RAYS_PER_TICK_REST_RESUME', '4096'],
]);
const probeBudgetAfterCadenceMiss = compileHelper('probeBudgetAfterCadenceMiss', [
    ['RAYS_PER_TICK_MIN', '2048'],
]);

assert.equal(probeBudgetAfterInteraction(98_304), 4_096,
    'release cannot resume at the stale maximum ray budget');
assert.equal(probeBudgetAfterInteraction(3_072), 3_072,
    'an already-safe adaptive budget is preserved');
assert.equal(probeBudgetAfterInteraction(512), 2_048,
    'the solve never falls below its bounded progress floor');

assert.equal(probeBudgetAfterCadenceMiss(98_304), 49_152,
    'one accepted cadence miss immediately halves the maximum budget');
let backedOffBudget = 98_304;
for (let i = 0; i < 12; i++) backedOffBudget = probeBudgetAfterCadenceMiss(backedOffBudget);
assert.equal(backedOffBudget, 2_048,
    'repeated misses reach the bounded progress floor');
assert.equal(probeBudgetAfterCadenceMiss(2_048), 2_048,
    'cadence backoff cannot cross the bounded progress floor');

const readIntegerConstant = (name) => {
    const literal = source.match(new RegExp(`const ${name}\\s*=\\s*([0-9_]+)`))?.[1];
    assert.ok(literal, `${name} is present`);
    return Number(literal.replaceAll('_', ''));
};
const overloadMs = readIntegerConstant('TICK_OVERLOAD_MS');
const pauseMs = readIntegerConstant('TICK_PAUSE_MS');
const overloadStrikes = readIntegerConstant('TICK_OVERLOAD_STRIKES');
assert.equal(overloadMs, 100, 'accepted solve cadence window remains 100 ms');
assert.equal(pauseMs, 1_000, 'long host gaps begin at 1000 ms');
assert.equal(overloadStrikes, 2, 'one isolated long cadence miss is tolerated');

// Tie the numeric model to the production interval branch. One presentation stall
// may be unrelated to GI, but repeated accepted stalls mean the browser is not
// making progress. Genuine tab/debugger/host pauses must remain budget-neutral.
const cadenceBlock = source.slice(
    source.indexOf('// Measure this accepted solve tick'),
    source.indexOf('// reactivity: detect live edits'),
);
assert.match(cadenceBlock,
    /else if \(dt >= TICK_OVERLOAD_MS && dt < TICK_PAUSE_MS\)/,
    '100-999 ms accepted gaps use the repeated-overload lane');
assert.match(cadenceBlock,
    /cadenceOverloadStreak >= TICK_OVERLOAD_STRIKES[\s\S]*probeBudgetAfterCadenceMiss\(tickBudgetRays\)/,
    'repeated long cadence misses back off the live budget');
assert.match(cadenceBlock,
    /\} else \{\s*tickDtEma = 0;\s*hysteresisTickDtEma = 0;\s*cadenceOverloadStreak = 0;/,
    'pause-sized gaps clear cadence state without changing the learned budget');

const budgetAfterIntervals = (initialBudget, intervals) => {
    let budget = initialBudget;
    let streak = 0;
    for (const dt of intervals) {
        if (dt > 0 && dt < overloadMs) {
            streak = 0;
        } else if (dt >= overloadMs && dt < pauseMs) {
            streak = Math.min(overloadStrikes, streak + 1);
            if (streak >= overloadStrikes) budget = probeBudgetAfterCadenceMiss(budget);
        } else {
            streak = 0;
        }
    }
    return budget;
};
assert.equal(budgetAfterIntervals(98_304, [169]), 98_304,
    'one isolated 169 ms presentation stall preserves the learned budget');
assert.equal(budgetAfterIntervals(98_304, [169, 16, 169]), 98_304,
    'a recovered frame breaks the overload streak');
assert.equal(budgetAfterIntervals(98_304, [169, 169]), 49_152,
    'two consecutive 169 ms stalls halve the budget');
assert.equal(budgetAfterIntervals(98_304, [169, 169, 169]), 24_576,
    'continued overload keeps backing off until cadence recovers');
assert.equal(budgetAfterIntervals(24_576, [169, 1_000, 169]), 24_576,
    'a pause breaks the overload streak');
assert.equal(budgetAfterIntervals(24_576, [5_000]), 24_576,
    'a long host/debugger pause preserves the learned budget');

assert.match(source, /const wasMoving = lastMoving;/);
assert.match(source, /if \(!moving && wasMoving && \(!continuous \|\| wasPlaying\)\) \{[\s\S]*probeBudgetAfterInteraction\(tickBudgetRays\)/,
    'moving-to-rest transition applies the recovery budget');
assert.match(source, /if \(\(dirty \|\| !casc\[0\]\.gpu\) && !buildHeld\) \{\s*[\s\S]*?resetFramePacing\(\)/,
    'a CPU scene/kernel rebuild is excluded from solve cadence');
assert.match(source, /if \(buildStage < 2 && !buildHeld\) \{\s*resetFramePacing\(\)/,
    'a staggered cascade build is excluded from solve cadence');
// Topology gets the LOWEST priority: a rest-held rebuild must fall through to
// the continuous lanes and the solve instead of vetoing the whole tick, and a
// pre-existing held rebuild must not abort the packet-consumption blocks.
assert.match(source, /const buildHeld = !restOnly && fieldEverReady && !!casc\[0\]\.gpu;/,
    'held rebuilds are detected without gating first bring-up');
assert.match(source, /if \(disposed \|\| \(dirty && !dirtyBefore\)\) return;/,
    'deform packets abort only on a rebuild armed under the await');
assert.match(source, /probeBudgetAfterCadenceMiss\(tickBudgetRays\)/,
    'bad cadence backs off aggressively instead of spending a long tail above the floor');
assert.match(source, /tickBudgetRays \+ 1024/,
    'budget growth is gradual after frame cadence recovers');

// r185 accepts an array of ComputeNodes as one ordered compute pass. Per-kernel
// dispatch sizes must live on each node's count; a second computeAsync argument
// would apply one shared dispatch size to every node in the array.
assert.match(source, /const solveKernels = \[traceKernel, emitterVisKernel, blendKernel\];/,
    'the ordered solve group resolves emitter visibility between trace and blend');
assert.match(source, /if \(glossyKernel\) solveKernels\.push\(glossyKernel\);\s*solveKernels\.push\(uploadKernel\);/,
    'optional glossy resolve precedes the final upload in the cached solve group');
assert.match(source, /buffers, traceKernel, emitterVisKernel, blendKernel, glossyKernel, uploadKernel,\s*solveKernels,/,
    'the cached solve group is retained by the cascade GPU owner');

const solveBlock = source.slice(
    source.indexOf('// Match the dispatch envelope to the LIVE auto-throttled batch.'),
    source.indexOf('C.lastSolveAt = tNow;'),
);
assert.match(solveBlock, /gpu\.traceKernel\.count = updated \* raysPerProbe;/,
    'trace dispatch count follows the live probe and ray budget');
assert.match(solveBlock, /gpu\.emitterVisKernel\.count = U\.emitterCount\.value > 0\s*\? updated \* GI_EMITTER_INJECT_CAP\s*: 0;/,
    'emitter visibility scales with the live probe batch and dispatches no work without emitters');
assert.match(solveBlock, /gpu\.blendKernel\.count = updated \* PROBE_WORKGROUP_SIZE;/,
    'blend dispatch count follows one complete cache-sharing workgroup per live probe');
assert.match(solveBlock,
    /gpu\.glossyKernel\.count = updated \* gpu\.glossyGroupsPerProbe \* PROBE_WORKGROUP_SIZE;/,
    'glossy dispatch pads each live probe to complete cache-sharing workgroups');
assert.match(solveBlock, /gpu\.uploadKernel\.count = updated \* PROBE_WORKGROUP_SIZE;/,
    'upload dispatch count follows one complete cache-sharing workgroup per live probe');
assert.match(solveBlock, /await renderer\.computeAsync\(gpu\.solveKernels\);/,
    'the ordered solve group is submitted as one compute pass');
assert.doesNotMatch(solveBlock, /renderer\.computeAsync\(gpu\.(?:trace|emitterVis|blend|glossy|upload)Kernel/,
    'the solve path has no per-kernel compute submissions');
assert.doesNotMatch(solveBlock, /computeAsync\(gpu\.solveKernels\s*,/,
    'the solve batch does not use one shared dispatch-size override');

const workgroupSize = readIntegerConstant('PROBE_WORKGROUP_SIZE');
assert.equal(workgroupSize, 64, 'probe cache-sharing workgroups remain 64 lanes');
const octRes = readIntegerConstant('OCT_RES');
const border = readIntegerConstant('BORDER');
const tile = octRes + 2 * border;
assert.equal(octRes * octRes, 36, 'only the 6x6 interior needs blend/filter work');
assert.equal(tile * tile, workgroupSize, 'one 64-lane workgroup covers the complete 8x8 upload tile');
assert.match(source,
    /const glossyGroupsPerProbe = glossyReflectionsEnabled\s*\? Math\.ceil\(\(glossyTile \* glossyTile\) \/ PROBE_WORKGROUP_SIZE\)\s*:\s*0;/,
    'glossy workgroups per probe cover every tile texel with complete workgroups');
const dispatchCounts = (updated, rays, glossyTile) => {
    const glossyGroupsPerProbe = Math.ceil((glossyTile * glossyTile) / workgroupSize);
    return {
        trace: updated * rays,
        blend: updated * workgroupSize,
        glossy: updated * glossyGroupsPerProbe * workgroupSize,
        upload: updated * workgroupSize,
    };
};
assert.deepEqual(dispatchCounts(32, 64, 10), {
    trace: 2_048,
    blend: 2_048,
    glossy: 4_096,
    upload: 2_048,
}, 'minimum-budget high-reflection solve uses the padded live dispatch envelope');

const blendBlock = source.slice(
    source.indexOf('const blendKernel = Fn'),
    source.indexOf('const glossyKernel ='),
);
const glossyBlock = source.slice(
    source.indexOf('const glossyKernel ='),
    source.indexOf('const uploadKernel = Fn'),
);
const uploadBlock = source.slice(
    source.indexOf('const uploadKernel = Fn'),
    source.indexOf('// ── CLASSIFY:'),
);
for (const [name, block, loaderSize] of [
    ['blend', blendBlock, 'PROBE_WORKGROUP_SIZE'],
    ['glossy', glossyBlock, 'PROBE_WORKGROUP_SIZE'],
]) {
    assert.match(block, /const loadK = lane\.toVar\(\)/,
        `${name} cache loader starts each lane at its invocation index`);
    assert.match(block, /Loop\(loadK\.lessThan\(uint\(raysPerProbe\)\)/,
        `${name} cache loader bounds every workgroup write by the live ray count`);
    assert.match(block, new RegExp(`loadK\\.addAssign\\(uint\\(${loaderSize}\\)\\)`),
        `${name} cache loader advances by one whole workgroup`);
    assert.match(block, /workgroupBarrier\(\)/,
        `${name} consumers wait for the shared ray cache`);
}
assert.match(blendBlock,
    /\.compute\(updatedCap\(\) \* PROBE_WORKGROUP_SIZE, \[PROBE_WORKGROUP_SIZE\]\)/,
    'blend reserves one complete 64-lane workgroup per updated probe');
assert.match(glossyBlock,
    /updatedCap\(\) \* glossyGroupsPerProbe \* PROBE_WORKGROUP_SIZE,[\s\S]*\[PROBE_WORKGROUP_SIZE\]/,
    'glossy reserves complete WG64 groups per updated probe');
assert.match(uploadBlock,
    /\.compute\(updatedCap\(\) \* PROBE_WORKGROUP_SIZE, \[PROBE_WORKGROUP_SIZE\]\)/,
    'upload reserves one complete 64-lane workgroup per updated probe');
assert.doesNotMatch(uploadBlock, /\bReturn\s*\(/,
    'upload guards work instead of returning before its workgroup barrier');

const rayRange = source.match(/const RAYS_MIN = (\d+), RAYS_MAX = (\d+);/);
assert.ok(rayRange, 'supported ray-count range is present');
const [, raysMinText, raysMaxText] = rayRange;
const raysMin = Number(raysMinText);
const raysMax = Number(raysMaxText);
assert.match(source, /Math\.round\(Number\(n\) \/ 16\) \* 16/,
    'setRays continues to quantize supported ray counts to 16-ray steps');
for (const [name, loaderSize] of [['blend', workgroupSize], ['glossy', workgroupSize]]) {
    for (let rays = raysMin; rays <= raysMax; rays += 16) {
        const loads = Array(rays).fill(0);
        for (let lane = 0; lane < loaderSize; lane++) {
            for (let ray = lane; ray < rays; ray += loaderSize) {
                assert.ok(ray >= 0 && ray < rays, `${name} ${rays}-ray loader stays in bounds`);
                loads[ray]++;
            }
        }
        assert.deepEqual(loads, Array(rays).fill(1),
            `${name} WG${loaderSize} covers every element of the ${rays}-ray cache exactly once`);
    }
}

// CPU analogues of the lane mappings catch workgroup-size changes that remain
// numerically equal for today's 8x8 tile but would silently break cache indexing.
const blendLocals = [];
for (let lane = 0; lane < octRes * octRes; lane++) {
    const lx = lane % octRes + border;
    const ly = Math.floor(lane / octRes) + border;
    blendLocals.push(ly * tile + lx);
}
const expectedInteriorLocals = [];
for (let ly = border; ly < border + octRes; ly++) {
    for (let lx = border; lx < border + octRes; lx++) expectedInteriorLocals.push(ly * tile + lx);
}
assert.deepEqual(blendLocals, expectedInteriorLocals,
    'the first 36 blend lanes cover every interior atlas texel in row-major order');
assert.equal(new Set(blendLocals).size, octRes * octRes,
    'blend interior lane mapping has no duplicate destinations');
assert.match(blendBlock, /const lx = lane\.mod\(uint\(OCT_RES\)\)\.add\(uint\(BORDER\)\)/);
assert.match(blendBlock, /const ly = lane\.div\(uint\(OCT_RES\)\)\.add\(uint\(BORDER\)\)/);

const uploadSourceForLane = (lane) => {
    const lx = lane % tile;
    const ly = Math.floor(lane / tile);
    const edge = tile - 1;
    const lo = border;
    const hi = border + octRes - 1;
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
    return { lx, ly, sx, sy, sourceLane: (sy - border) * octRes + sx - border };
};
const uploadMappings = Array.from({ length: workgroupSize }, (_, lane) => uploadSourceForLane(lane));
assert.ok(uploadMappings.every(({ sourceLane }) => sourceLane >= 0 && sourceLane < octRes * octRes),
    'all 64 upload lanes index the 36-element workgroup cache in bounds');
assert.deepEqual([...new Set(uploadMappings.map(({ sourceLane }) => sourceLane))].sort((a, b) => a - b),
    Array.from({ length: octRes * octRes }, (_, lane) => lane),
    'canonical upload mapping reaches all 36 cached interior sources');
for (const { lx, ly, sourceLane } of uploadMappings) {
    if (lx >= border && lx < border + octRes && ly >= border && ly < border + octRes) {
        assert.equal(sourceLane, (ly - border) * octRes + lx - border,
            `interior upload (${lx},${ly}) reads its matching cache lane`);
    }
}
assert.deepEqual([0, tile - 1, tile * (tile - 1), tile * tile - 1]
    .map((lane) => uploadSourceForLane(lane).sourceLane), [35, 30, 5, 0],
    'four upload corners mirror the canonical opposite interior corners');
assert.match(uploadBlock, /If\(activeProbe\.and\(lane\.lessThan\(uint\(OCT_RES \* OCT_RES\)\)\)/,
    'only 36 upload lanes execute the bilateral filter');
assert.match(uploadBlock, /workgroupBarrier\(\)/,
    'all upload lanes synchronize before reading filtered cache values');
assert.match(uploadBlock,
    /const sourceLane = sy\.sub\(uint\(BORDER\)\)\.mul\(uint\(OCT_RES\)\)\s*\.add\(sx\.sub\(uint\(BORDER\)\)\)/,
    'upload maps canonical gutter coordinates into the 36-element cache');
assert.match(uploadBlock, /irradianceCache\.element\(sourceLane\)/);
assert.match(uploadBlock, /depthCache\.element\(sourceLane\)/);

const prepBlock = source.slice(
    source.indexOf('if (C.needsClear || C.needsClassify)'),
    source.indexOf('// Match the dispatch envelope to the LIVE auto-throttled batch.'),
);
assert.doesNotMatch(prepBlock, /prep\.push\(renderer\.computeAsync/,
    'prep batching stores ComputeNodes, not already-started Promises');
assert.match(prepBlock, /await renderer\.computeAsync\(prep\);/,
    'prep dependencies are submitted as one ordered compute pass');

console.log('GI scheduler and dynamic-dispatch smoke passed');
