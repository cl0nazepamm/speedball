// Deterministic coverage for the time-sliced deform refit contract: reverse
// pre-order may yield mid-leaf, stale source versions restart from the latest
// snapshot, and no publication occurs until final exact bounds are complete.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import {
    createFlatBlasRefitStepper,
    runLatestBudgetedTask,
} from '../js/gi_refit.js';

const NODE_STRIDE = 8;
const VERTEX_STRIDE = 8;
const TRI_COUNT = 240; // leaf payload is eight-bit; large enough for several slices
const BUDGET_CHECK_BATCH = 32;

function makeSingleLeafFixture(points) {
    const buffer = new ArrayBuffer(NODE_STRIDE * 4);
    const nodesF = new Float32Array(buffer);
    const nodesU = new Uint32Array(buffer);
    const triIndex = new Uint32Array(TRI_COUNT * 3);
    const vertexData = new Float32Array(points.length * VERTEX_STRIDE);
    for (let i = 0; i < TRI_COUNT; i++) {
        triIndex[i * 3] = 0;
        triIndex[i * 3 + 1] = 1;
        triIndex[i * 3 + 2] = 2;
    }
    nodesU[6] = 1;
    nodesU[7] = ((TRI_COUNT << 24) | 0) >>> 0;
    const copyPoints = (next) => {
        for (let i = 0; i < next.length; i++) {
            const d = i * VERTEX_STRIDE;
            vertexData[d] = next[i][0];
            vertexData[d + 1] = next[i][1];
            vertexData[d + 2] = next[i][2];
        }
    };
    copyPoints(points);
    return { nodesF, nodesU, triIndex, vertexData, copyPoints };
}

// A tiny synthetic clock makes every budget decision reproducible. The leaf
// is deliberately larger than one slice, proving its accumulator survives
// yields and only the completed exact bounds land in the node.
{
    const fixture = makeSingleLeafFixture([[-2, 1, 4], [5, -3, 2], [1, 7, -6]]);
    const stepper = createFlatBlasRefitStepper({
        ...fixture,
        root: 0,
        end: 1,
        nodeStrideU32: NODE_STRIDE,
        vertexDataStride: VERTEX_STRIDE,
    });
    let clock = 0;
    const now = () => ++clock;
    let slices = 0;
    while (!stepper.done) {
        const before = stepper.processedTriangles;
        assert.equal(stepper.step({ deadline: now() + 3, now }), true);
        assert.ok(stepper.processedTriangles - before <= 3 * BUDGET_CHECK_BATCH,
            'one slice stays within the deterministic 3 ms budget plus its batched clock cadence');
        slices++;
    }
    assert.ok(slices > 1, 'large leaf yielded');
    assert.deepEqual(Array.from(fixture.nodesF.subarray(0, 6)), [-2, -3, -6, 5, 7, 4]);
}

// Mutate the source on the first yield. The first partial tree must never be
// committed; the runner re-captures version 2 and eventually publishes only
// its exact bounds.
{
    let sourceVersion = 1;
    let sourcePoints = [[0, 0, 0], [1, 0, 0], [0, 1, 0]];
    const fixture = makeSingleLeafFixture(sourcePoints);
    let captures = 0;
    let yields = 0;
    let commits = 0;
    let maxTrianglesPerSlice = 0;
    let clock = 0;
    const now = () => ++clock;

    const result = await runLatestBudgetedTask({
        budgetMs: 3,
        now,
        capture() {
            captures++;
            fixture.copyPoints(sourcePoints);
            return { version: sourceVersion, invalid: false };
        },
        createTask() {
            const inner = createFlatBlasRefitStepper({
                ...fixture,
                root: 0,
                end: 1,
                nodeStrideU32: NODE_STRIDE,
                vertexDataStride: VERTEX_STRIDE,
            });
            return {
                get done() { return inner.done; },
                step(options) {
                    const before = inner.processedTriangles;
                    const ok = inner.step(options);
                    maxTrianglesPerSlice = Math.max(maxTrianglesPerSlice, inner.processedTriangles - before);
                    return ok;
                },
            };
        },
        validate(snapshot) {
            return snapshot.version === sourceVersion;
        },
        commit() {
            commits++;
            return { bounds: Array.from(fixture.nodesF.subarray(0, 6)), version: sourceVersion };
        },
        async yieldTask() {
            yields++;
            assert.equal(commits, 0, 'partial slices are not published');
            if (yields === 1) {
                sourcePoints = [[10, 20, 30], [12, 19, 31], [11, 25, 28]];
                sourceVersion++;
            }
        },
    });

    assert.ok(yields > 1, 'runner yielded repeatedly');
    assert.equal(captures, 2, 'stale version restarted exactly once');
    assert.equal(commits, 1, 'only latest complete state committed');
    assert.ok(maxTrianglesPerSlice <= 3 * BUDGET_CHECK_BATCH,
        'all deterministic slices honored the 3 ms cap plus batched clock cadence');
    assert.deepEqual(result, { bounds: [10, 19, 28, 12, 25, 31], version: 2 });
}

// The production default must yield to the browser's next presented frame,
// not scheduler.yield(), whose continuation can run before paint and chain
// many individually bounded slices into a post-scrub low-FPS tail.
{
    const savedRaf = globalThis.requestAnimationFrame;
    const savedScheduler = globalThis.scheduler;
    const savedDocument = globalThis.document;
    let rafYields = 0;
    let schedulerYields = 0;
    globalThis.document = { visibilityState: 'visible' };
    globalThis.requestAnimationFrame = (cb) => {
        rafYields++;
        cb(0);
        return rafYields;
    };
    globalThis.scheduler = {
        yield() {
            schedulerYields++;
            return Promise.resolve();
        },
    };
    let steps = 0;
    const result = await runLatestBudgetedTask({
        capture: () => ({ invalid: false }),
        createTask: () => ({
            done: false,
            step() {
                steps++;
                this.done = steps >= 3;
                return true;
            },
        }),
        validate: () => true,
        commit: () => ({ steps }),
    });
    assert.deepEqual(result, { steps: 3 });
    assert.equal(rafYields, 2, 'every unfinished slice yields through rAF');
    assert.equal(schedulerYields, 0, 'scheduler.yield cannot bypass frame presentation');
    if (savedRaf === undefined) delete globalThis.requestAnimationFrame;
    else globalThis.requestAnimationFrame = savedRaf;
    if (savedScheduler === undefined) delete globalThis.scheduler;
    else globalThis.scheduler = savedScheduler;
    if (savedDocument === undefined) delete globalThis.document;
    else globalThis.document = savedDocument;
}

// Wiring/source invariants that cannot be exercised without Three/WebGPU in
// this dependency-free smoke.
const sceneSrc = await readFile(new URL('../js/spectral_scene.js', import.meta.url), 'utf8');
assert.match(sceneSrc, /function updateDeformsAsync\(/);
assert.match(sceneSrc, /instanceof Float32Array/, 'large gather has a direct typed-array lane');
assert.match(sceneSrc, /validateDeformSnapshot\(snapshot\)/, 'snapshot revalidated before commit');
assert.match(sceneSrc, /disposeDeformUpdates,/, 'built scene exposes lifecycle cancellation');

const tracerSrc = await readFile(new URL('../js/spectral_tracer.js', import.meta.url), 'utf8');
assert.match(tracerSrc, /await built\.updateDeformsAsync/);
assert.match(tracerSrc, /gpu\?\.built !== built/, 'late PT completion cannot publish into a replacement scene');

const probesSrc = await readFile(new URL('../js/gi_probes.js', import.meta.url), 'utf8');
assert.match(probesSrc, /await built\.updateDeformsAsync/);
assert.match(probesSrc, /cachedBuilt !== built/, 'late Halo completion cannot publish into a replacement field');

console.log('gi async deform refit smoke: ok');
