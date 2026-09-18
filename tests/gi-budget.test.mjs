import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createProbeBudgetController } from '../js/gi_budget.js';

const MAX = 98304;
function simulate(interval, { seconds = 12, initial = MAX } = {}) {
    const controller = createProbeBudgetController();
    let budget = initial, ms = 0, frame = 0, lowest = budget;
    const history = [];
    while (ms < seconds * 1000) {
        const dt = interval(budget, frame++, ms);
        ms += dt;
        budget = controller.update(dt, budget, MAX);
        lowest = Math.min(lowest, budget);
        history.push({ ms, budget, dt });
    }
    return { budget, lowest, history };
}

test('55 FPS, steady or dropped-frame cadence, preserves full probe coverage', () => {
    for (const interval of [() => 1000 / 55, (_, f) => f % 11 === 10 ? 1000 / 30 : 1000 / 60]) {
        const result = simulate(interval);
        assert.equal(result.budget, MAX);
        assert.ok(result.lowest >= MAX * 0.75, 'cadence jitter cannot cause cumulative cuts');
    }
});

test('presentation caps cannot cause repeated blind reductions', () => {
    for (const fps of [54, 50, 30, 15]) {
        const result = simulate(() => 1000 / fps);
        assert.equal(result.budget, MAX, `${fps} FPS restores coverage`);
        assert.ok(result.lowest >= MAX * 0.75, `${fps} FPS makes at most one unsuccessful cut`);
    }
});

test('real GI pressure backs off and converges near the cadence target', () => {
    const result = simulate(budget => Math.max(1000 / 60, 8 + budget / 4096));
    assert.ok(result.budget < MAX);
    assert.ok(result.budget > 2048);
    assert.ok(result.history.at(-1).dt < 20);
});

test('severe GI pressure can still reach the minimum budget', () => {
    const result = simulate(budget => 15 + budget / 100, { seconds: 25 });
    assert.equal(result.lowest, 2048);
});

test('small interaction-resume budgets recover at 55 and 30 FPS', () => {
    for (const fps of [55, 30]) {
        const result = simulate(() => 1000 / fps, { initial: 4096 });
        assert.equal(result.budget, MAX);
    }
});

test('isolated stalls and pauses do not exhaust the budget', () => {
    const result = simulate((_, f) => f === 45 ? 500 : f === 180 ? 5000 : 1000 / 60);
    assert.equal(result.budget, MAX);
    assert.ok(result.lowest >= MAX * 0.75);
});

test('changed presentation caps and external budget resets are handled', () => {
    const result = simulate((_, f, ms) => ms < 5000 ? 1000 / 30 : 1000 / 60);
    assert.equal(result.budget, MAX);
    const controller = createProbeBudgetController();
    for (let i = 0; i < 40; i++) controller.update(20, MAX, MAX);
    assert.equal(controller.update(1000 / 60, 4096, MAX), 4096);
    controller.reset();
    assert.equal(controller.update(1000 / 60, MAX, MAX), MAX);
});
