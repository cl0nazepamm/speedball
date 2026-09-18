// Presentation cadence includes frame caps and non-GI work. A budget reduction
// is a trial: keep it only when it actually improves cadence.
export function createProbeBudgetController() {
    let elapsed = 0;
    let samples = 0;
    let holdMs = 0;
    let targetMs = 18.5;
    let trial = null;
    let expectedBudget = null;

    function reset() {
        elapsed = samples = holdMs = 0;
        targetMs = 18.5;
        trial = null;
        expectedBudget = null;
    }

    function update(dt, budget, ceiling, minimum = 2048) {
        if (expectedBudget !== null && budget !== expectedBudget) reset();
        expectedBudget = budget;
        if (!(dt > 0) || !Number.isFinite(dt) || dt >= 1000) {
            // A hidden tab/debugger pause is not evidence about GPU throughput.
            elapsed = samples = 0;
            trial = null;
            return budget;
        }
        holdMs = Math.max(0, holdMs - dt);
        elapsed += dt;
        samples++;
        if (elapsed < 500 || samples < 4) return budget;

        const meanMs = elapsed / samples;
        elapsed = samples = 0;
        const apply = (next) => (expectedBudget = Math.max(minimum, Math.min(ceiling, Math.round(next))));

        if (trial) {
            const previous = trial;
            trial = null;
            if (previous.kind === 'shrink') {
                if (meanMs > previous.meanMs * 0.95) {
                    // Less GI did not buy frame time. Restore coverage and learn
                    // this presentation floor instead of starving the probes.
                    targetMs = Math.max(18.5, Math.min(meanMs, previous.meanMs) * 1.1);
                    holdMs = 2000;
                    return apply(previous.budget);
                }
            } else if (meanMs > targetMs && meanMs > previous.meanMs * 1.1) {
                holdMs = 2000;
                return apply(previous.budget);
            }
        }

        // Reconsider a learned cap when presentation itself becomes faster.
        if (meanMs < targetMs * 0.8) targetMs = 18.5;
        if (holdMs > 0) return budget;
        if (meanMs > targetMs && budget > minimum) {
            trial = { kind: 'shrink', budget, meanMs };
            return apply(budget * 0.75);
        }
        if (meanMs <= targetMs && budget < ceiling) {
            trial = { kind: 'grow', budget, meanMs };
            return apply(budget + Math.max(2048, budget * 0.5));
        }
        return budget;
    }

    return { update, reset };
}
