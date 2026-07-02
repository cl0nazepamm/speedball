// gi_settings.js — DEMO helper: ONE shared, persisted set of SPEEDBALL GI settings
// for every example page. Sponza (index.html) is the canonical scene — these defaults
// are its tuning — and the city adopts whatever is saved here. Scene-specific state
// (camera, sun/time-of-day, seed, props) stays per-page. Not part of the library
// (js/index.js does not export it).

const KEY = 'speedball-gi-settings-v2';

// Canonical defaults (= Sponza's tuning).
export const GI_DEFAULTS = {
    giEnabled: true, giIntensity: 10, giDivisions: 16, giRays: 64,
    giCascades: 1, giContinuous: true, showProbes: false,
    giHysteresis: 0.6, giNormalBias: 1.75, giRadianceClamp: 8, giDepthSharpness: 40,
    giLeak: 0.5, giSolid: 0,
    giChangeThreshold: 2.5, giSnapAmount: 0.30, giFireflyClamp: 6.0,
};

const GI_KEYS = Object.keys(GI_DEFAULTS);
const GI_PERSIST_KEYS = GI_KEYS.filter((k) => ![
    'giChangeThreshold',
    'giSnapAmount',
    'giFireflyClamp',
].includes(k));

// Saved settings merged over the defaults. Type-checked per key so a stale or
// hand-edited store can never inject NaN/strings into the setters. Adaptive
// temporal tuning is intentionally not restored: it is pinned to the defaults.
export function loadGiSettings() {
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(KEY)); } catch (e) { /* storage disabled */ }
    const out = { ...GI_DEFAULTS };
    if (saved && typeof saved === 'object') {
        for (const k of GI_PERSIST_KEYS) if (typeof saved[k] === typeof GI_DEFAULTS[k]) out[k] = saved[k];
    }
    return out;
}

// Debounced write of the GI keys (only) out of a page's params object.
let _saveTimer = 0;
export function saveGiSettings(params) {
    clearTimeout(_saveTimer);
    _saveTimer = setTimeout(() => {
        const out = {};
        for (const k of GI_PERSIST_KEYS) if (k in params) out[k] = params[k];
        try { localStorage.setItem(KEY, JSON.stringify(out)); } catch (e) { /* storage disabled */ }
    }, 300);
}

// One GI panel for EVERY demo page — identical controls everywhere, built from the
// same shared params (Sponza = canonical). Trimming a slider here trims it on all
// pages at once. onInteract = the page's markInteraction; onStructure fires for
// knobs that re-place probes so pages can refresh their probe helpers.
export function addGiPanel(gui, gi, params, { onInteract = () => {}, onStructure = () => {} } = {}) {
    const fGI = gui.addFolder('SPEEDBALL GI');
    fGI.add(params, 'giEnabled').name('enabled').onChange((v) => { gi.setEnabled(v); onInteract(); });
    fGI.add(params, 'giIntensity').min(0).step(0.05).name('intensity').onChange((v) => { gi.setIntensity(v); onInteract(); }); // uncapped
    // STRUCTURAL knobs (idle-gated rebuild — never a per-tick recompile).
    fGI.add(params, 'giDivisions', 2, 32, 1).name('divisions').onChange((v) => { gi.setDivisions(v); onStructure(); onInteract(); });
    fGI.add(params, 'giRays', 32, 256, 16).name('rays / probe').onChange((v) => { gi.setRays(v); onInteract(); });
    fGI.add(params, 'giCascades', { 'single grid': 1, 'cascaded (2)': 2 }).name('cascades').onChange((v) => { gi.setCascades(+v); onStructure(); onInteract(); });
    fGI.add(params, 'giContinuous').name('continuous (solve while moving)').onChange((v) => { gi.setContinuous(v); onInteract(); });
    fGI.add(params, 'showProbes').name('show probes').onChange(() => onStructure());
    // Quality — UNIFORM-backed, apply instantly (no recompile/rebuild).
    const fQ = fGI.addFolder('Quality');
    fQ.add(params, 'giHysteresis', 0.5, 0.99, 0.01).name('hysteresis').onChange((v) => { gi.setHysteresis(v); onInteract(); });
    fQ.add(params, 'giNormalBias', 0, 4, 0.05).name('normal bias').onChange((v) => { gi.setNormalBias(v); onInteract(); });
    fQ.add(params, 'giRadianceClamp', 0, 32, 0.5).name('radiance clamp').onChange((v) => { gi.setRadianceClamp(v); onInteract(); });
    fQ.add(params, 'giDepthSharpness', 1, 200, 1).name('depth sharpness').onChange((v) => { gi.setDepthSharpness(v); onInteract(); });
    fQ.add(params, 'giLeak', 0, 1, 0.05).name('chebyshev strength').onChange((v) => { gi.setChebyStrength(v); onInteract(); });
    // solid-scene (classify) stays hidden + pinned to 0 — a backface test misreads thin
    // two-sided geometry (Sponza curtains), so it's opt-in for enclosed solids only.
    // Adaptive temporal blend is pinned below. Hysteresis remains the one public
    // temporal control; the rest should feel like engine behavior, not scene tuning.
    return fGI;
}

// Push every shared setting into a probe field via its live setters.
export function applyGiSettings(gi, s) {
    gi.setEnabled(s.giEnabled);
    gi.setIntensity(s.giIntensity);
    gi.setDivisions(s.giDivisions);
    gi.setRays(s.giRays);
    gi.setCascades(s.giCascades);
    gi.setContinuous(s.giContinuous);
    gi.setHysteresis(s.giHysteresis);
    gi.setNormalBias(s.giNormalBias);
    gi.setRadianceClamp(s.giRadianceClamp);
    gi.setDepthSharpness(s.giDepthSharpness);
    gi.setChebyStrength(s.giLeak);
    gi.setClassifyStrength(s.giSolid);
    gi.setChangeThreshold(s.giChangeThreshold);
    gi.setSnapAmount(s.giSnapAmount);
    gi.setFireflyClamp(s.giFireflyClamp);
}
