import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const probeSource = readFileSync(new URL('../js/gi_probes.js', import.meta.url), 'utf8');
const helperStart = probeSource.indexOf('export const HYSTERESIS_DT_REF_MS');
const helperEnd = probeSource.indexOf('let _node = null;', helperStart);
assert.ok(helperStart >= 0 && helperEnd > helperStart, 'production hysteresis helpers must remain extractable');
const helperSource = probeSource.slice(helperStart, helperEnd).replace(/\bexport\s+/g, '');
const { HYSTERESIS_DT_REF_MS, advanceReactiveTicks, hysteresisExponentForInterval, probeUpdateIntervalTicks } = new Function(
    `${helperSource}\nreturn { HYSTERESIS_DT_REF_MS, advanceReactiveTicks, hysteresisExponentForInterval, probeUpdateIntervalTicks };`,
 )();

const EPS = 1e-12;
const close = (actual, expected, message) => {
    assert.ok(Math.abs(actual - expected) <= EPS, `${message}: expected ${expected}, got ${actual}`);
};
const closeVector = (actual, expected, message) => {
    assert.equal(actual.length, expected.length, `${message}: vector length`);
    for (let i = 0; i < actual.length; i++) close(actual[i], expected[i], `${message}[${i}]`);
};
const mixVector = (current, previous, history) => current.map((value, i) => value * (1 - history) + previous[i] * history);
close(HYSTERESIS_DT_REF_MS, 1000 / 60, 'reference cadence');

// Reference-domain adaptive policy from the compute blend. The significant-change
// branch is the sensitive one: applying its fixed drop after normalization was the
// high-FPS flicker bug.
const base = 0.9;
const noiseBoost = 0.25;
const changeDrop = 0.3;
const minChange = 0.55;
const changeReference = Math.max(base - changeDrop, minChange); // 0.6 at 60 Hz

function squareWavePeakToPeak(fps, seconds = 8) {
    const exponent = hysteresisExponentForInterval(1000 / fps, true);
    const history = Math.pow(changeReference, exponent);
    let value = 0;
    let lo = Infinity;
    let hi = -Infinity;
    const frames = fps * seconds;
    for (let frame = 0; frame < frames; frame++) {
        // Fixed physical 30 Hz square wave: the input is identical in wall time at
        // every tested render rate.
        const candidate = Math.floor((frame * 30) / fps) % 2;
        value = candidate * (1 - history) + value * history;
        if (frame >= frames - fps) {
            lo = Math.min(lo, value);
            hi = Math.max(hi, value);
        }
    }
    return hi - lo;
}

const expectedPeakToPeak = 8 / 17; // 0.470588... for two 60 Hz updates at h=0.6
for (const fps of [30, 60, 120, 240, 480]) {
    close(squareWavePeakToPeak(fps), expectedPeakToPeak, `${fps} Hz adaptive history`);
}

// No high-refresh floor: dt/ref must keep shrinking above 240 Hz or fast displays
// consume too much new Monte-Carlo noise per second.
close(hysteresisExponentForInterval(1000 / 480), 0.125, '480 Hz exponent');
close(hysteresisExponentForInterval(1000 / 120, false), 1, 'normalization-off exponent');

// Sparse revisits must not saturate either. Ten services between updates means
// fewer, stronger blends at low FPS and more, weaker blends at high FPS; their
// one-second product must still equal exactly 60 reference-rate blends.
const sparseReference = 0.95;
for (const fps of [30, 60, 120, 240, 480]) {
    const revisitServices = 10;
    const exponent = hysteresisExponentForInterval((1000 / fps) * revisitServices);
    const updatesPerSecond = fps / revisitServices;
    const retention = Math.pow(Math.pow(sparseReference, exponent), updatesPerSecond);
    close(retention, Math.pow(sparseReference, 60), `${fps} Hz sparse history`);
}

// The reactive fade is 75 reference ticks (=1.25 s), not 75 rendered frames.
// After half a second, every render rate must have consumed exactly 30 units.
for (const fps of [30, 60, 120, 240, 480]) {
    let remaining = 75;
    for (let frame = 0; frame < fps * 0.5; frame++) {
        remaining = advanceReactiveTicks(remaining, 1000 / fps);
    }
    close(remaining, 45, `${fps} Hz reactive ramp`);
}

// Circular modulo batches have a fractional average revisit interval N/K. This
// keeps wall-time decay continuous right up to full coverage instead of ceil(N/K)
// nearly doubling the decay at K=N-1 and snapping back at K=N.
for (let updated = 25; updated <= 100; updated++) {
    const probes = 100;
    const revisit = probeUpdateIntervalTicks(probes, updated);
    close((updated / probes) * revisit, 1, `fractional pass N=${probes} K=${updated}`);
}

// Alternating unequal cascades: each is serviced every two accepted ticks, and its
// own size determines the remaining fractional pass length.
const coarseTicks = probeUpdateIntervalTicks(100, 100, 2);
const fineTicks = probeUpdateIntervalTicks(200, 100, 2);
close(coarseTicks, 2, 'coarse cascade cadence');
close(fineTicks, 4, 'fine cascade cadence');
close((1 / 2) * (100 / 100) * coarseTicks, 1, 'coarse wall-time authority');
close((1 / 2) * (100 / 200) * fineTicks, 1, 'fine wall-time authority');

// Exercise the steady/noisy branch too; reference-first exponentiation must retain
// the same one-second history at every render rate.
const noiseReference = base + (1 - base) * noiseBoost;
const expectedNoiseRetention = Math.pow(noiseReference, 60);
for (const fps of [30, 60, 120, 240, 480]) {
    const exponent = hysteresisExponentForInterval(1000 / fps);
    close(Math.pow(Math.pow(noiseReference, exponent), fps), expectedNoiseRetention, `${fps} Hz noise history`);
}

// Reflections intentionally use the steady/noisy reference retention rather than
// diffuse's nonlinear per-texel change detector. This makes their vec4 history a
// true time semigroup: repeating one physical sample over more render substeps has
// exactly the same wall-time result.
const roughCandidates = [
    [0.08, 0.18, 0.03, 0.20],
    [0.62, 0.12, 0.44, 0.75],
];
function roughReflectionState(fps, seconds = 8) {
    const history = Math.pow(noiseReference, hysteresisExponentForInterval(1000 / fps));
    const blockFrames = fps / 30;
    assert.ok(Number.isInteger(blockFrames), `${fps} Hz rough block alignment`);
    let state = null;
    const trajectory = [];
    for (let frame = 0; frame < fps * seconds; frame++) {
        const candidate = roughCandidates[Math.floor((frame * 30) / fps + 1e-9) % 2];
        state = state === null ? [...candidate] : mixVector(candidate, state, history);
        if ((frame + 1) % blockFrames === 0) trajectory.push([...state]);
    }
    return { state, trajectory };
}
const roughReference = roughReflectionState(60);
for (const fps of [30, 60, 120, 240, 480]) {
    const actual = roughReflectionState(fps);
    closeVector(actual.state, roughReference.state, `${fps} Hz rough reflection history`);
    assert.equal(actual.trajectory.length, roughReference.trajectory.length, `${fps} Hz rough trajectory length`);
    for (let i = 0; i < actual.trajectory.length; i++) {
        closeVector(actual.trajectory[i], roughReference.trajectory[i], `${fps} Hz rough boundary ${i}`);
    }
}

// Transparent black is valid converged reflection history, not an initialization
// marker. Once the parallel depth history is seeded, a later sparse hit must still
// receive the configured hysteresis instead of jumping to 100% authority.
const transparentBlack = [0, 0, 0, 0];
const sparseHit = [0.8, 0.2, 0.1, 1.0];
const retainedSparseHit = mixVector(sparseHit, transparentBlack, noiseReference);
closeVector(retainedSparseHit, sparseHit.map((v) => v * (1 - noiseReference)), 'zero-coverage rough history remains initialized');
assert.notDeepEqual(retainedSparseHit, sparseHit, 'zero-coverage rough history must not accept a sparse hit at full authority');

// Glossy stores an unnormalized numerator and angular support denominator. Both
// hidden states must use the SAME elapsed-time coefficient; otherwise their ratio
// pumps with FPS even when the candidate signal is physically identical.
const sparseCascadeTicks = probeUpdateIntervalTicks(200, 100, 2);
close(sparseCascadeTicks, 4, 'reflection sparse cascade cadence');
const glossyCandidates = [
    { num: [0.04, 0.16, 0.02, 0.20], den: 0.40 },
    { num: [0.48, 0.09, 0.30, 0.60], den: 0.75 },
];
function glossyReflectionState(fps, seconds = 8) {
    const history = Math.pow(noiseReference, hysteresisExponentForInterval((1000 / fps) * sparseCascadeTicks));
    const blockFrames = fps / 7.5;
    assert.ok(Number.isInteger(blockFrames), `${fps} Hz glossy block alignment`);
    let num = null;
    let den = 0;
    const trajectory = [];
    for (let frame = 0; frame < fps * seconds; frame += sparseCascadeTicks) {
        const candidate = glossyCandidates[Math.floor((frame * 7.5) / fps + 1e-9) % 2];
        if (num === null) {
            num = [...candidate.num];
            den = candidate.den;
        } else {
            num = mixVector(candidate.num, num, history);
            den = candidate.den * (1 - history) + den * history;
        }
        if ((frame + sparseCascadeTicks) % blockFrames === 0) {
            trajectory.push({
                num: [...num],
                den,
                resolved: [num[0] / den, num[1] / den, num[2] / den, num[3] / den],
            });
        }
    }
    return {
        num,
        den,
        resolved: [num[0] / den, num[1] / den, num[2] / den, num[3] / den],
        trajectory,
    };
}
const glossyReference = glossyReflectionState(60);
for (const fps of [30, 60, 120, 240, 480]) {
    const actual = glossyReflectionState(fps);
    closeVector(actual.num, glossyReference.num, `${fps} Hz glossy numerator`);
    close(actual.den, glossyReference.den, `${fps} Hz glossy support`);
    closeVector(actual.resolved, glossyReference.resolved, `${fps} Hz glossy resolved reflection`);
    assert.equal(actual.trajectory.length, glossyReference.trajectory.length, `${fps} Hz glossy trajectory length`);
    for (let i = 0; i < actual.trajectory.length; i++) {
        closeVector(actual.trajectory[i].num, glossyReference.trajectory[i].num, `${fps} Hz glossy numerator boundary ${i}`);
        close(actual.trajectory[i].den, glossyReference.trajectory[i].den, `${fps} Hz glossy support boundary ${i}`);
        closeVector(actual.trajectory[i].resolved, glossyReference.trajectory[i].resolved, `${fps} Hz glossy resolved boundary ${i}`);
    }
}

// Tie the numeric models to the production shader wiring. These guards catch the
// two regressions the coefficient-only smoke could not see: rough falling back to
// an energy sentinel/adaptive diffuse hEff, and glossy blending numerator/support
// with different or unnormalized coefficients.
assert.match(probeSource, /const steadyReflectionH\s*=\s*pow\(rawNoiseH\.clamp\(1e-6,\s*1\.0\),\s*U\.hysteresisExponent\);/, 'rough reflection reference must use normalized steady/noisy history');
assert.match(probeSource, /const sh\s*=\s*select\(dWasZero,\s*float\(0\.0\),\s*steadyReflectionH\);/, 'rough reflection must use seeded steady history');
assert.match(probeSource, /const sBlended\s*=\s*mix\(sCur,\s*sPrev,\s*sh\);/, 'rough reflection blend must consume seeded normalized history');
assert.doesNotMatch(probeSource, /const sWasEmpty\b/, 'rough reflection must not use RGBA energy as an initialization sentinel');
assert.match(probeSource, /const glossyReferenceH\s*=\s*U\.hysteresis\.add\(\s*float\(1\.0\)\.sub\(U\.hysteresis\)\.mul\(U\.debugTempNoiseHBoost\),?\s*\);/s, 'glossy reference must use the steady/noisy policy');
assert.match(probeSource, /const glossyH\s*=\s*pow\(glossyReferenceH\.clamp\(1e-6,\s*1\.0\),\s*U\.hysteresisExponent\);/, 'glossy reflection must normalize steady history');
assert.match(probeSource, /const gh\s*=\s*select\(empty,\s*float\(0\.0\),\s*glossyH\);/, 'glossy initialization must feed normalized history');
assert.match(probeSource, /const num\s*=\s*mix\(curNum,\s*prevNum,\s*gh\)\.toVar\(\);/, 'glossy numerator must use normalized history');
assert.match(probeSource, /const den\s*=\s*mix\(curDen,\s*prevDen,\s*gh\)\.max\(float\(1e-6\)\)\.toVar\(\);/, 'glossy support must use the same normalized history');

console.log('GI hysteresis normalization smoke passed');

