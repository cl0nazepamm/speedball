// Run with a local server on 8778:
// playwright-cli run-code --filename=tests/gi-backends.playwright.js
async page => {
    const base = 'http://127.0.0.1:8778/tests/fixtures/gi-backends.html';
    const assert = (condition, message) => { if (!condition) throw new Error(message); };
    const results = [];
    const cases = [
        { quality: 'off', rays: 32, cascades: 1, steps: 2 },
        { quality: 'rough', rays: 64, cascades: 1, steps: 5 },
        { quality: 'high', rays: 64, cascades: 2, steps: 5 },
        { quality: 'ultra', rays: 32, cascades: 1, steps: 3 },
        { quality: 'high', rays: 32, cascades: 1, steps: 4, features: '&maps&emitter&classify&clustered&beforeinit' },
    ];
    for (const config of cases) {
        const records = [];
        for (const backend of ['webgl', 'webgpu']) {
            const tab = await page.context().newPage();
            const errors = [];
            tab.on('pageerror', error => errors.push(error.message));
            tab.on('console', message => {
                if (['error', 'warning'].includes(message.type())) errors.push(message.text());
            });
            try {
                await tab.goto(`${base}?backend=${backend}&reflections=${config.quality}&rays=${config.rays}&cascades=${config.cascades}&steps=${config.steps}${config.features || ''}`);
                await tab.waitForFunction(() => window.ready, null, { timeout: 60000 });
                const record = await tab.evaluate(async () => {
                    const { renderer, gi } = testGI;
                    const fields = {};
                    for (let c = 0; c < gi.advanced.debug.state().buildCascadeCount; c++) {
                        for (const kind of ['irr', 'depth', 'spec', 'gloss', 'glossWeight', 'state']) {
                            const values = await gi.advanced.debug.read(kind, c);
                            if (values) fields[`${kind}${c}`] = Array.from(values);
                        }
                    }
                    return { fields, cascades: gi.advanced.debug.state().buildCascadeCount, backend: renderer.backend.isWebGLBackend ? 'webgl' : 'webgpu', supported: gi.isSupported(), data: gi.hasData() };
                });
                assert(record.backend === backend && record.supported && record.data, `${backend}: field unavailable`);
                assert(record.cascades === config.cascades && (config.cascades === 1 || record.fields.irr1), `${backend}: fine cascade was not exercised`);
                assert(errors.length === 0, `${backend}/${config.quality}: ${errors.join('\n')}`);
                records.push(record);
            } finally {
                await tab.close();
            }
        }
        const differences = {};
        for (const [name, a] of Object.entries(records[0].fields)) {
            const b = records[1].fields[name];
            assert(b?.length === a.length, `${name}: backend layouts differ`);
            let max = 0, total = 0, scale = 0;
            for (let i = 0; i < a.length; i++) {
                assert(Number.isFinite(a[i]) && Number.isFinite(b[i]), `${name}[${i}]: nonfinite probe value`);
                const difference = Math.abs(a[i] - b[i]);
                max = Math.max(max, difference); total += difference; scale += Math.abs(b[i]);
            }
            const relative = total / Math.max(scale, 1e-6);
            assert(relative < 0.005 && max < 0.02, `${config.quality}/${name}: max ${max}, relative ${relative}`);
            differences[name] = { max, relative };
        }
        results.push({ ...config, differences });
    }

    // Exercise texture/alpha maps, explicit emitters, classification, sparse
    // round-robin updates, live edits, history reuse, and lifecycle on WebGL.
    const tab = await page.context().newPage();
    const errors = [];
    tab.on('pageerror', error => errors.push(error.message));
    tab.on('console', message => { if (['error', 'warning'].includes(message.type())) errors.push(message.text()); });
    try {
        await tab.goto(`${base}?backend=webgl&reflections=high&maps&emitter&classify&clustered&steps=4`);
        await tab.waitForFunction(() => window.ready, null, { timeout: 60000 });
        const lifecycle = await tab.evaluate(async () => {
            const t = testGI, { gi, renderer, THREE } = t;
            const check = (ok, message) => { if (!ok) throw new Error(message); };
            const read = async () => {
                const a = await gi.advanced.debug.read('irr');
                check(a?.length > 0 && a.every(Number.isFinite), 'Invalid solved irradiance');
                return a;
            };
            const before = await read();
            check(before.some(v => v > 0), 'No GI energy');
            const atlas = gi.advanced.debug.atlas();
            const generation = gi.advanced.debug.state().sceneStorageGeneration;
            t.blocker.position.x = 1.7;
            gi.markTransformsDirty(t.blocker);
            t.light.intensity = 30;
            gi.forceLightingRefresh();
            t.emitterMaterial.emissiveIntensity = 8;
            gi.markMaterialValuesDirty(t.emitterMaterial);
            await t.step(4);
            const after = await read();
            check(after.some((v, i) => Math.abs(v - before[i]) > 0.001), 'Live edits did not reach the solver');
            check(atlas === gi.advanced.debug.atlas(), 'Live edits replaced the atlas');
            check(generation === gi.advanced.debug.state().sceneStorageGeneration, 'Live edits rebuilt the scene arena');

            const positions = t.blocker.geometry.attributes.position;
            positions.setY(0, positions.getY(0) + 0.3); positions.needsUpdate = true;
            gi.markDeformsDirty();
            await t.step(3);
            check(gi.advanced.debug.state().lastRefitCount > 0, 'Deform did not refit the BVH');

            gi.setRays(64); await t.step(4);
            check(atlas === gi.advanced.debug.atlas(), 'Ray change lost compatible history');
            gi.setDivisions(8); gi.setRayBudget(2048); await t.step(4);
            check(atlas !== gi.advanced.debug.atlas(), 'Grid resize did not replace its atlas');
            check(gi.advanced.debug.state().cascadeState[0].cursor > 0, 'Sparse updates did not advance');

            // A caller's target, viewport, scissor, output mode and clear state
            // must survive an internal solve, including the copy/render passes.
            const host = new THREE.RenderTarget(16, 16);
            renderer.setRenderTarget(host); renderer.setClearColor(0x123456, 0.3);
            renderer.setViewport(1, 2, 10, 11); renderer.setScissor(2, 3, 7, 8); renderer.setScissorTest(true);
            renderer.autoClear = false;
            await gi.advanced.tick({ idleMs: 1000 });
            check(renderer.getRenderTarget() === host && renderer.autoClear === false, 'Host render state lost');
            check(renderer.getScissorTest() && renderer.getScissor(new THREE.Vector4()).equals(new THREE.Vector4(2, 3, 7, 8)), 'Host scissor lost');
            check(renderer.getViewport(new THREE.Vector4()).equals(new THREE.Vector4(1, 2, 10, 11)), 'Host viewport lost');
            check(renderer.getClearColor(new THREE.Color()).getHex() === 0x123456 && renderer.getClearAlpha() === 0.3, 'Host clear state lost');
            renderer.setRenderTarget(null); renderer.setScissorTest(false); renderer.setViewport(0, 0, 640, 480); renderer.autoClear = true; host.dispose();

            const solved = await read();
            gi.setEnabled(false); await t.step(2);
            check((await read()).every((v, i) => v === solved[i]), 'Disabled GI still solved');
            gi.setEnabled(true); await t.step(2);
            const resources = renderer.info.memory.renderTargets;
            gi.dispose(); gi.dispose();
            check(!gi.hasData(), 'Disposed field still advertises data');
            check(renderer.info.memory.renderTargets < resources, 'Render targets leaked on disposal');
            const second = t.installSpeedballGI({ ...t.options, installLightsNode: false });
            await second.advanced.tick({ idleMs: 1000 }); await second.advanced.tick({ idleMs: 1000 });
            check(second.hasData(), 'Reinstall failed'); second.dispose();
            check(renderer.backend.gl.getError() === 0, 'WebGL error after lifecycle checks');
            return { passed: true, remainingTargets: renderer.info.memory.renderTargets };
        });
        assert(errors.length === 0, `WebGL lifecycle: ${errors.join('\n')}`);
        results.push({ lifecycle });
    } finally {
        await tab.close();
    }

    // Exercise Three's automatic fallback, not just forceWebGL.
    const fallbackContext = await page.context().browser().newContext();
    try {
        await fallbackContext.addInitScript(() => Object.defineProperty(navigator, 'gpu', { value: undefined }));
        const fallbackTab = await fallbackContext.newPage();
        const failures = [];
        fallbackTab.on('pageerror', error => failures.push(error.message));
        fallbackTab.on('console', message => { if (message.type() === 'error') failures.push(message.text()); });
        await fallbackTab.goto(`${base}?backend=webgpu&beforeinit&clustered&steps=2`);
        await fallbackTab.waitForFunction(() => window.ready, null, { timeout: 60000 });
        const fallback = await fallbackTab.evaluate(async () => {
            const { webglProbeSupport } = await import('/js/gi_webgl.js');
            return testGI.renderer.backend.isWebGLBackend && testGI.gi.isSupported() && testGI.gi.hasData()
                && testGI.renderer.backend.gl.getError() === 0
                && !webglProbeSupport({ backend: { isWebGLBackend: true, extensions: null } })
                && !webglProbeSupport({ backend: { isWebGLBackend: true, extensions: { has: () => false } } });
        });
        assert(fallback && failures.length === 0, `Automatic fallback: ${failures.join('\n')}`);
        results.push({ automaticFallback: true });
    } finally {
        await fallbackContext.close();
    }
    return results;
}
