// PowerSHOT infrared mode - image-intensifier night-vision emulation.
//
// This is a separate imaging path from the visible-light digital ISP. It models
// a modern Gen-3 image-intensifier tube: a GaAs photocathode (red/NIR-weighted
// spectral response), an MCP gain stage with a constant-output transfer curve,
// resolution-limited optics, sparse photon scintillation, and a phosphor screen.
//
// It works in a single scalar "electron-flux" channel L in scene-linear 0..1+,
// packs adaptation / halo / eye-glow sources into one quarter-resolution
// analysis target, then develops the final phosphor image in one full-resolution
// pass. The shipped look is a high-end white-phosphor tube.

import * as THREE from "three/webgpu";
import {
  vec2, vec3, vec4, float, uniform, texture, screenUV,
  mix, max, min, dot, abs, floor, fract, sin, cos, sqrt, log, exp, step,
  smoothstep,
} from "three/tsl";

const LUM709 = vec3(0.2126, 0.7152, 0.0722);
const TAU = 6.2831853;

function exp2u(x) {
  return exp(x.mul(Math.LN2));
}

function srgbToLinear(c) {
  const lo = c.mul(1.0 / 12.92);
  const hi = c.add(0.055).div(1.055).max(0.0).pow(2.4);
  return mix(lo, hi, step(0.04045, c));
}

function linearToSrgb(c) {
  const v = c.clamp(0.0, 1.0);
  const lo = v.mul(12.92);
  const hi = v.pow(1.0 / 2.4).mul(1.055).sub(0.055);
  return mix(lo, hi, step(0.0031308, v));
}

function hash13(p) {
  const a = fract(p.mul(0.1031));
  const d = dot(a, vec3(a.z, a.y, a.x).add(31.32));
  const b = a.add(d);
  return fract(b.x.add(b.y).mul(b.z));
}

// Box-Muller gaussian keyed to a grain cell and a temporal phase. Used for the
// continuous shot-noise "fizz" on lit surfaces (Poisson sqrt(N) behaviour).
function gaussTemporal(p, t, salt) {
  const u1 = hash13(vec3(p.x, p.y, t.add(salt))).max(1e-6);
  const u2 = hash13(vec3(p.x.add(11.0), p.y.add(3.0), t.add(salt).add(1.7))).max(1e-6);
  return sqrt(log(u1).mul(-2.0)).mul(cos(u2.mul(6.2831853)));
}

// --- Stage 1: GaAs photocathode spectral response -------------------------
// Fake the NIR-weighted response of a Gen-3 photocathode from an sRGB frame
// (which carries no real NIR): rising red>green>blue quantum efficiency, a
// chlorophyll "Wood effect" foliage glow, a waxy skin lift, and suppression of
// blue sky / open water toward black. Returns the scalar electron flux.
function nirFromLinear(lin, ctx) {
  const P = ctx.P;
  const broad = dot(lin, P.spectralMix);
  const redExcess = max(lin.r.sub(lin.g.add(lin.b).mul(0.5)), 0.0);
  const greenDominance = max(lin.g.sub(lin.r.mul(0.58).add(lin.b.mul(0.42))), 0.0);
  const vegExcess = greenDominance.mul(smoothstep(0.035, 0.55, lin.g));
  const skin = max(min(lin.r.mul(0.95), lin.g).sub(lin.b.mul(0.72)), 0.0);
  const skyMask = smoothstep(0.02, 0.38, lin.b.sub(max(lin.r, lin.g).mul(0.86)));
  const waterMask = smoothstep(0.02, 0.34, min(lin.g, lin.b).sub(lin.r.mul(1.12)));

  let sim = broad
    .add(redExcess.mul(P.redReflectance))
    .add(vegExcess.mul(P.greenReflectance))
    .add(skin.mul(P.skinBoost));
  sim = sim
    .mul(float(1.0).sub(P.skySuppress.mul(skyMask)))
    .mul(float(1.0).sub(P.waterSuppress.mul(waterMask)))
    .sub(lin.b.mul(P.blueSuppression));

  const mono = dot(lin, LUM709);
  sim = sim.max(0.0).pow(P.photocathodeGamma);
  return mix(sim, mono, P.nirInput).max(0.0).mul(exp2u(P.exposure));
}

function pseudoNirValue(srcTex, ctx, uv) {
  return nirFromLinear(srgbToLinear(texture(srcTex, uv).rgb), ctx);
}

// --- Stage 7 (folded into read): resolution-limited PSF -------------------
// A real tube is never razor-sharp (cascaded photocathode gap + MCP pore
// sampling + phosphor grain + fibre-optic window). A small separable-ish
// gaussian of the SOURCE keeps the signal soft so scintillation later sits as
// crisp sparkle on a soft base - the cure for "generic grain".
function softNir(srcTex, ctx, uv) {
  const s = ctx.P.psfSigma;
  const ox = vec2(ctx.texel.x.mul(s), 0.0);
  const oy = vec2(0.0, ctx.texel.y.mul(s));
  const wC = 0.4;
  const wN = 0.15;
  return pseudoNirValue(srcTex, ctx, uv).mul(wC)
    .add(pseudoNirValue(srcTex, ctx, uv.add(ox)).mul(wN))
    .add(pseudoNirValue(srcTex, ctx, uv.sub(ox)).mul(wN))
    .add(pseudoNirValue(srcTex, ctx, uv.add(oy)).mul(wN))
    .add(pseudoNirValue(srcTex, ctx, uv.sub(oy)).mul(wN));
}

// --- Stage 11: phosphor screen --------------------------------------------
// One intensity-invariant chroma multiply; only the brightest blooming cores
// desaturate toward white (eye/sensor clip). White phosphor is a cool
// near-neutral, NOT pure (1,1,1).
function phosphorMap(L, ctx) {
  const P = ctx.P;
  let screen = L.sub(P.screenBlack).max(0.0);
  screen = screen.mul(P.screenGain);
  screen = screen.div(screen.add(P.screenShoulder)).mul(P.screenShoulder.add(1.0));
  screen = screen.max(0.0).pow(P.phosphorGamma);

  const base = P.phosphorChroma.mul(screen);
  const white = P.highlightWhite.mul(min(screen, float(1.0)));
  const s = smoothstep(P.bloomStart, P.bloomStart.add(P.bloomRange), screen)
    .mul(P.highlightDesat);
  return mix(base, white, s);
}

// --- Stage 8: chicken-wire fixed pattern ----------------------------------
// Faint hexagonal MCP multifibre-boundary gain modulation, visible only in
// bright uniform fields, near-invisible on a high-end white-phosphor tube. A
// smooth 3-plane-wave honeycomb (no hard SDF) avoids moire against the pixel
// grid. Replaces the old CCD-style row/column gaussian streaks.
function hexGain(ctx, uv, L) {
  const P = ctx.P;
  const aspect = ctx.resolution.x.div(ctx.resolution.y);
  const p = vec2(uv.x.sub(0.5).mul(aspect), uv.y.sub(0.5)).mul(P.chickenFreq);
  const a = p.x;
  const b = p.x.mul(0.5).add(p.y.mul(0.8660254));
  const c = p.x.mul(-0.5).add(p.y.mul(0.8660254));
  const h = cos(a.mul(TAU)).add(cos(b.mul(TAU))).add(cos(c.mul(TAU)));
  const hn = h.add(1.5).mul(1.0 / 4.5).clamp(0.0, 1.0); // 0 at cell borders, 1 at centres
  const line = float(1.0).sub(smoothstep(0.0, P.chickenLine.add(1e-3), hn));
  const gate = smoothstep(P.chickenGateLo, P.chickenGateHi, L);
  return float(1.0).sub(P.chickenAmp.mul(line).mul(gate));
}

// --- Stage 9: scintillation -----------------------------------------------
// The headline fix. Real intensifier noise is NOT symmetric gaussian grain - it
// is a SPARSE field of brief BRIGHT flashes (single amplified photo-electrons),
// each ~one resolution element, whose density RISES as the signal falls (dark
// areas boil, bright areas are clean), riding on the EBI self-glow floor, plus a
// continuous sqrt(signal) shot fizz on lit surfaces.
//
// One frame's sparse sparkle field, a pure function of (cell, frame phase).
function sparkleAt(cell, fphase, L, ctx) {
  const P = ctx.P;
  const u = hash13(vec3(cell.x, cell.y, fphase.add(11.0)));
  const darkness = float(1.0).sub(smoothstep(0.025, 0.58, L));
  const dens = P.scintDensity.mul(
    float(1.0).add(P.scintDarkBoost.mul(darkness)),
  );
  const fire = step(float(1.0).sub(dens), u); // only the top `dens` fraction fire
  const v = hash13(vec3(cell.x.add(7.0), cell.y.add(3.0), fphase.add(23.0)));
  const amp = v.pow(P.scintSharp).mul(P.scintGain); // pow -> rare bright pops, not uniform
  const ride = P.ebi.mul(P.scintFloor).add(sqrt(L.add(1e-4)).mul(0.5));
  return fire.mul(amp).mul(ride);
}

function scintillation(L, ctx) {
  const P = ctx.P;
  const grain = max(P.scintGrain, float(0.5));
  const cell = floor(screenUV.mul(ctx.resolution).div(grain));
  const f = ctx.frame;

  // Continuous shot fizz: absolute noise ~ sqrt(signal), so relative noise falls
  // as the surface brightens (correct Poisson / high-SNR behaviour).
  const shot = gaussTemporal(cell, f, 37.0).mul(sqrt(L.add(P.ebi)).mul(P.shotStrength));

  // Sparse sparkle with a STATELESS phosphor "boil": because each frame's field
  // is a pure function of (cell, frame), we recompute the two previous frames and
  // combine with a decaying max() - each flash turns on sharply then fades over a
  // couple of frames (P45 ~1 ms afterglow), so the field boils rather than
  // blinks like TV snow. Needs no history buffer, and freezes correctly when the
  // frame uniform is held constant.
  const s0 = sparkleAt(cell, f, L, ctx);
  const s1 = sparkleAt(cell, f.sub(1.0), L, ctx).mul(P.persistence);
  const s2 = sparkleAt(cell, f.sub(2.0), L, ctx).mul(P.persistence.mul(P.persistence));
  const sparkle = max(s0, max(s1, s2));

  return shot.add(sparkle).mul(P.noiseAmount);
}

function infraredOutputAlpha(sourceSample, effectColor) {
  const sourceAlpha = sourceSample.a.clamp(0.0, 1.0);
  const effectDelta = dot(abs(effectColor.sub(sourceSample.rgb)), LUM709);
  const effectAlpha = effectDelta.sub(0.002).mul(6.0).clamp(0.0, 0.65);
  return sourceAlpha.add(effectAlpha.mul(sourceAlpha.oneMinus())).clamp(0.0, 1.0);
}

function stAnalysis(srcTex, ctx, eyeMaskTex) {
  const nir = pseudoNirValue(srcTex, ctx, screenUV);
  const glowMask = smoothstep(
    ctx.P.glowThreshold,
    ctx.P.glowThreshold.add(ctx.P.glowSoftness),
    nir,
  );
  const glowSource = nir.mul(glowMask);
  const mask = eyeMaskTex ? texture(eyeMaskTex, screenUV).r.clamp(0.0, 1.0) : float(0.0);
  return vec4(nir, glowSource, glowSource.mul(mask), 1.0);
}

function stAnalysisBlur(tex, ctx, dx, dy) {
  const sigma = 2.55;
  let sum = vec3(0.0);
  let wsum = 0.0;
  for (let i = -6; i <= 6; i += 1) {
    const w = Math.exp(-(i * i) / (2.0 * sigma * sigma));
    const off = ctx.analysisTexel.mul(vec2(dx, dy)).mul(ctx.P.glowRadius.mul(i));
    sum = sum.add(texture(tex, screenUV.add(off)).rgb.mul(w));
    wsum += w;
  }
  return vec4(sum.div(wsum), 1.0);
}

function stDevelop(srcTex, ctx, analysisTex, eyeMaskTex, stages) {
  const P = ctx.P;
  const sourceSample = texture(srcTex, screenUV);
  const nirSharp = pseudoNirValue(srcTex, ctx, screenUV); // sharp, for eye local contrast
  const analysis = analysisTex ? texture(analysisTex, screenUV).rgb : vec3(nirSharp, 0.0, 0.0);

  // 1+7: photocathode signal, resolution-limited (soft).
  let signal = softNir(srcTex, ctx, screenUV);

  // 2: EBI self-glow floor - lifts blacks into a faint glowing grey so noise has
  // something to ride on (what you see with the lens cap on).
  signal = signal.add(P.ebi);

  // 3: local adaptation / ABC - huge gain that auto-dims bright regions and lifts
  // shadows toward a constant output, from the quarter-res blurred local mean.
  if (stages.adaptation) {
    const local = max(analysis.r, float(1e-4));
    const adaptiveGain = P.middleGrey
      .div(local)
      .pow(P.localGain)
      .clamp(P.minGain, P.maxGain);
    signal = signal.mul(adaptiveGain);
  }

  // 4: MCP gain + Naka-Rushton transfer - constant-gain region, saturation knee,
  // and a hard ABC ceiling (maxOutput). The phosphor is linear, so no extra gamma.
  const x = signal.mul(P.gain);
  signal = x.div(x.div(P.maxOutput).add(1.0));

  // 5: halo / bloom - bright sources bloom into a fixed angular disc (charge
  // spreading across the photocathode-MCP gap), constant in screen space.
  if (stages.glow) {
    signal = signal.add(analysis.g.mul(P.glowStrength));
  }

  // 6: eyeshine / retroreflection (animal eyes, retroreflectors under the IR
  // illuminator), with the optional eye-mask input.
  if (stages.eyes) {
    const localContrast = nirSharp.sub(analysis.r.mul(P.eyeLocalRatio)).max(0.0);
    const eyeCore = smoothstep(
      P.eyeThreshold,
      P.eyeThreshold.add(P.eyeSoftness),
      localContrast,
    ).mul(P.eyeStrength);
    signal = signal
      .add(eyeCore.mul(P.eyeCoreStrength))
      .add(eyeCore.mul(analysis.g).mul(P.eyeHaloStrength));

    if (eyeMaskTex) {
      const mask = texture(eyeMaskTex, screenUV).r.clamp(0.0, 1.0);
      signal = signal
        .add(analysis.b.mul(P.maskedEyeHalo).mul(P.eyeStrength))
        .add(eyeCore.mul(mask).mul(P.maskedEyeCore));
    }
  }

  // 8+9: device-locked chicken-wire, then sparse scintillation on top.
  if (stages.noise) {
    signal = signal.mul(hexGain(ctx, screenUV, signal));
    signal = signal.add(scintillation(signal, ctx)).max(P.ebi.mul(0.5));
  }

  // 10: eyepiece shading - circular field-stop vignette + centre hotspot only.
  // No CRT scanlines and no whole-frame flicker: an intensifier is not a raster
  // display and autogating runs at kHz, imperceptibly.
  if (stages.display) {
    const p = screenUV.sub(0.5);
    const aspect = ctx.resolution.x.div(ctx.resolution.y);
    const q = vec2(p.x.mul(aspect), p.y);
    const radius = sqrt(dot(q, q));
    const vignette = float(1.0).sub(smoothstep(0.25, 0.78, radius).mul(P.vignette));
    const hotspot = exp(radius.mul(radius).mul(-8.0)).mul(P.hotspot);
    signal = signal.mul(vignette).add(hotspot);
  }

  // 11: phosphor colour map.
  const phosphor = phosphorMap(signal.clamp(0.0, 1.35), ctx);
  const effectColor = linearToSrgb(phosphor).clamp(0.0, 1.0);
  const finalColor = mix(sourceSample.rgb, effectColor, ctx.power).clamp(0.0, 1.0);
  return vec4(finalColor, infraredOutputAlpha(sourceSample, finalColor));
}

export function makeInfraredUniforms() {
  return {
    resolution: uniform(new THREE.Vector2(1, 1)),
    texel: uniform(new THREE.Vector2(1, 1)),
    analysisTexel: uniform(new THREE.Vector2(1, 1)),
    frame: uniform(0),
    power: uniform(1),
    P: {
      // photocathode spectral response
      exposure: uniform(1.0),
      nirInput: uniform(0.0),
      spectralMix: uniform(new THREE.Vector3(0.50, 0.40, 0.10)),
      redReflectance: uniform(0.25),
      greenReflectance: uniform(0.65),
      blueSuppression: uniform(0.10),
      skySuppress: uniform(0.45),
      waterSuppress: uniform(0.35),
      skinBoost: uniform(0.12),
      photocathodeGamma: uniform(0.88),

      // tube self-glow floor
      ebi: uniform(0.0045),

      // local adaptation / ABC
      middleGrey: uniform(0.18),
      localGain: uniform(0.32),
      minGain: uniform(0.70),
      maxGain: uniform(3.0),

      // MCP gain + Naka-Rushton transfer
      gain: uniform(3.4),
      maxOutput: uniform(1.05),

      // halo / bloom
      glowThreshold: uniform(0.60),
      glowSoftness: uniform(0.22),
      glowStrength: uniform(0.45),
      glowRadius: uniform(1.45),

      // eyeshine
      eyeStrength: uniform(0.90),
      eyeThreshold: uniform(0.30),
      eyeSoftness: uniform(0.12),
      eyeLocalRatio: uniform(1.15),
      eyeCoreStrength: uniform(0.56),
      eyeHaloStrength: uniform(0.50),
      maskedEyeCore: uniform(0.90),
      maskedEyeHalo: uniform(0.80),

      // resolution-limited optics
      psfSigma: uniform(0.75),

      // chicken-wire fixed pattern
      chickenAmp: uniform(0.02),
      chickenFreq: uniform(38.0),
      chickenLine: uniform(0.06),
      chickenGateLo: uniform(0.45),
      chickenGateHi: uniform(0.85),

      // scintillation noise
      noiseAmount: uniform(0.70),
      scintGrain: uniform(1.15),
      scintDensity: uniform(0.055),
      scintGain: uniform(0.55),
      scintSharp: uniform(3.2),
      scintDarkBoost: uniform(1.8),
      shotStrength: uniform(0.035),
      scintFloor: uniform(0.5),

      // phosphor screen
      phosphorChroma: uniform(new THREE.Vector3(0.92, 0.96, 1.00)),
      highlightWhite: uniform(new THREE.Vector3(1.00, 1.00, 1.00)),
      screenBlack: uniform(0.004),
      screenGain: uniform(1.08),
      screenShoulder: uniform(0.92),
      phosphorGamma: uniform(0.94),
      highlightDesat: uniform(0.55),
      bloomStart: uniform(0.75),
      bloomRange: uniform(0.55),

      // eyepiece shading
      vignette: uniform(0.30),
      hotspot: uniform(0.10),

      // phosphor persistence (scintillation boil tail)
      persistence: uniform(0.32),
    },
  };
}

export const INFRARED_PRESETS = {
  white_phosphor: {
    name: "P45 White Phosphor",
    sensor_resolution: [1280, 960],
    exposure: 0.85,
    nir_input: 0.0,
    spectral_mix: [0.58, 0.34, 0.08],
    red_reflectance: 0.24,
    green_reflectance: 0.92,
    blue_suppression: 0.16,
    sky_suppress: 0.66,
    water_suppress: 0.52,
    skin_boost: 0.17,
    photocathode_gamma: 0.86,
    ebi: 0.0065,
    middle_grey: 0.18,
    local_gain: 0.46,
    min_gain: 0.58,
    max_gain: 4.2,
    gain: 3.9,
    max_output: 0.98,
    glow_threshold: 0.44,
    glow_softness: 0.24,
    glow_strength: 0.34,
    glow_radius: 1.90,
    eye_strength: 0.78,
    eye_threshold: 0.28,
    eye_softness: 0.14,
    eye_local_ratio: 1.15,
    eye_core_strength: 0.50,
    eye_halo_strength: 0.44,
    masked_eye_core: 0.82,
    masked_eye_halo: 0.68,
    psf_sigma: 0.92,
    chicken_amp: 0.012,
    chicken_freq: 44.0,
    chicken_line: 0.045,
    chicken_gate_lo: 0.54,
    chicken_gate_hi: 0.92,
    noise_amount: 0.48,
    scint_grain: 1.05,
    scint_density: 0.018,
    scint_gain: 0.74,
    scint_sharp: 4.2,
    scint_dark_boost: 4.0,
    shot_strength: 0.026,
    scint_floor: 0.72,
    phosphor_chroma: [0.78, 0.86, 0.96],
    highlight_white: [0.96, 0.98, 1.00],
    screen_black: 0.006,
    screen_gain: 1.12,
    screen_shoulder: 0.86,
    phosphor_gamma: 0.94,
    highlight_desat: 0.46,
    bloom_start: 0.64,
    bloom_range: 0.58,
    vignette: 0.26,
    hotspot: 0.055,
    persistence: 0.42,
  },
};

export const INFRARED_PRESET_KEYS = Object.keys(INFRARED_PRESETS);

export function applyInfraredPreset(ctx, preset) {
  const P = ctx.P;
  P.exposure.value = preset.exposure;
  P.nirInput.value = preset.nir_input;
  P.spectralMix.value.set(...preset.spectral_mix);
  P.redReflectance.value = preset.red_reflectance;
  P.greenReflectance.value = preset.green_reflectance;
  P.blueSuppression.value = preset.blue_suppression;
  P.skySuppress.value = preset.sky_suppress;
  P.waterSuppress.value = preset.water_suppress ?? 0.35;
  P.skinBoost.value = preset.skin_boost;
  P.photocathodeGamma.value = preset.photocathode_gamma ?? 0.88;
  P.ebi.value = preset.ebi;
  P.middleGrey.value = preset.middle_grey;
  P.localGain.value = preset.local_gain;
  P.minGain.value = preset.min_gain;
  P.maxGain.value = preset.max_gain;
  P.gain.value = preset.gain;
  P.maxOutput.value = preset.max_output;
  P.glowThreshold.value = preset.glow_threshold;
  P.glowSoftness.value = preset.glow_softness;
  P.glowStrength.value = preset.glow_strength;
  P.glowRadius.value = preset.glow_radius;
  P.eyeStrength.value = preset.eye_strength;
  P.eyeThreshold.value = preset.eye_threshold;
  P.eyeSoftness.value = preset.eye_softness;
  P.eyeLocalRatio.value = preset.eye_local_ratio;
  P.eyeCoreStrength.value = preset.eye_core_strength;
  P.eyeHaloStrength.value = preset.eye_halo_strength;
  P.maskedEyeCore.value = preset.masked_eye_core;
  P.maskedEyeHalo.value = preset.masked_eye_halo;
  P.psfSigma.value = preset.psf_sigma;
  P.chickenAmp.value = preset.chicken_amp;
  P.chickenFreq.value = preset.chicken_freq;
  P.chickenLine.value = preset.chicken_line;
  P.chickenGateLo.value = preset.chicken_gate_lo;
  P.chickenGateHi.value = preset.chicken_gate_hi;
  P.noiseAmount.value = preset.noise_amount;
  P.scintGrain.value = preset.scint_grain;
  P.scintDensity.value = preset.scint_density;
  P.scintGain.value = preset.scint_gain;
  P.scintSharp.value = preset.scint_sharp;
  P.scintDarkBoost.value = preset.scint_dark_boost;
  P.shotStrength.value = preset.shot_strength;
  P.scintFloor.value = preset.scint_floor;
  P.phosphorChroma.value.set(...preset.phosphor_chroma);
  P.highlightWhite.value.set(...preset.highlight_white);
  P.screenBlack.value = preset.screen_black ?? 0.004;
  P.screenGain.value = preset.screen_gain ?? 1.08;
  P.screenShoulder.value = preset.screen_shoulder ?? 0.92;
  P.phosphorGamma.value = preset.phosphor_gamma ?? 0.94;
  P.highlightDesat.value = preset.highlight_desat ?? 0.55;
  P.bloomStart.value = preset.bloom_start;
  P.bloomRange.value = preset.bloom_range;
  P.vignette.value = preset.vignette;
  P.hotspot.value = preset.hotspot;
  P.persistence.value = preset.persistence ?? 0.3;
}

export const INFRARED_STAGE_DEFS = [
  { id: "adaptation", label: "Local gain adaptation" },
  { id: "glow", label: "Intensifier halo" },
  { id: "eyes", label: "Retinal flare" },
  { id: "noise", label: "Tube scintillation" },
  { id: "display", label: "Phosphor display" },
];

export class InfraredPipeline {
  constructor(renderer) {
    this.renderer = renderer;
    this.ctx = makeInfraredUniforms();

    const opts = {
      type: THREE.HalfFloatType,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
      colorSpace: THREE.NoColorSpace,
    };
    this.rtAnalysisA = new THREE.RenderTarget(1, 1, opts);
    this.rtAnalysisB = new THREE.RenderTarget(1, 1, { ...opts });

    this.quadScene = new THREE.Scene();
    this.quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), null);
    this.mesh.frustumCulled = false;
    this.quadScene.add(this.mesh);

    this.enabled = new Set(["adaptation", "glow", "eyes", "noise", "display"]);
    this.source = null;
    this.eyeMask = null;
    this.size = { w: 0, h: 0 };
    this.analysisSteps = [];
    this.developMat = null;
    this.dirty = true;
  }

  _mat(colorNode) {
    const m = new THREE.MeshBasicNodeMaterial();
    m.colorNode = colorNode;
    m.depthTest = false;
    m.depthWrite = false;
    m.toneMapped = false;
    return m;
  }

  setSource(tex) {
    if (this.source === tex) return;
    this.source = tex;
    this.clearHistory();
    this.dirty = true;
  }

  setSize(w, h) {
    if (w === this.size.w && h === this.size.h) return;
    const aw = Math.max(1, Math.round(w / 4));
    const ah = Math.max(1, Math.round(h / 4));
    this.rtAnalysisA.setSize(aw, ah);
    this.rtAnalysisB.setSize(aw, ah);
    this.size = { w, h };
    this.ctx.resolution.value.set(w, h);
    this.ctx.texel.value.set(1 / w, 1 / h);
    this.ctx.analysisTexel.value.set(1 / aw, 1 / ah);
    this.clearHistory();
    this.dirty = true;
  }

  setInputMode(mode) {
    this.ctx.P.nirInput.value = mode === "nir" ? 1 : 0;
  }

  setEyeMask(textureObject) {
    if (this.eyeMask === textureObject) return;
    if (textureObject) {
      textureObject.colorSpace = THREE.NoColorSpace;
      textureObject.flipY = false;
      textureObject.generateMipmaps = false;
      textureObject.minFilter = THREE.LinearFilter;
      textureObject.magFilter = THREE.LinearFilter;
    }
    this.eyeMask = textureObject || null;
    this.dirty = true;
  }

  clearEyeMask() {
    this.setEyeMask(null);
  }

  clearHistory() {
    // The phosphor "boil" is stateless (it recomputes prior frames analytically),
    // so the intensifier path keeps no history target.
  }

  setEnabled(id, on) {
    const hasStage = this.enabled.has(id);
    if (on === hasStage) return;
    if (on) this.enabled.add(id);
    else this.enabled.delete(id);
    this.dirty = true;
  }

  _rebuild() {
    for (const s of this.analysisSteps) s.material.dispose();
    if (this.developMat) this.developMat.dispose();
    this.analysisSteps = [];
    this.developMat = null;
    this.dirty = false;
    if (!this.source) return;

    const stages = {
      adaptation: this.enabled.has("adaptation"),
      glow: this.enabled.has("glow"),
      eyes: this.enabled.has("eyes"),
      noise: this.enabled.has("noise"),
      display: this.enabled.has("display"),
    };
    const needsAnalysis = stages.adaptation || stages.glow || stages.eyes;

    if (needsAnalysis) {
      this.analysisSteps.push({
        material: this._mat(stAnalysis(this.source, this.ctx, this.eyeMask)),
        target: this.rtAnalysisA,
      });
      this.analysisSteps.push({
        material: this._mat(stAnalysisBlur(this.rtAnalysisA.texture, this.ctx, 1, 0)),
        target: this.rtAnalysisB,
      });
      this.analysisSteps.push({
        material: this._mat(stAnalysisBlur(this.rtAnalysisB.texture, this.ctx, 0, 1)),
        target: this.rtAnalysisA,
      });
    }

    this.developMat = this._mat(
      stDevelop(
        this.source,
        this.ctx,
        needsAnalysis ? this.rtAnalysisA.texture : null,
        this.eyeMask,
        stages,
      ),
    );
    this.developMat.transparent = true;
    this.developMat.blending = THREE.NoBlending;
  }

  renderTexture(inputTexture, frame = 0, { outputTarget = null } = {}) {
    if (!inputTexture) return false;
    this.setSource(inputTexture);
    if (this.dirty) this._rebuild();
    if (!this.source || !this.developMat) return false;
    this.ctx.frame.value = frame;
    const r = this.renderer;
    const previousTarget = r.getRenderTarget?.() ?? null;

    try {
      for (const step of this.analysisSteps) {
        this.mesh.material = step.material;
        r.setRenderTarget(step.target);
        r.render(this.quadScene, this.quadCam);
      }
      this.mesh.material = this.developMat;
      r.setRenderTarget(outputTarget);
      r.render(this.quadScene, this.quadCam);
      return true;
    } finally {
      r.setRenderTarget(previousTarget);
    }
  }

  async render(frame) {
    this.renderTexture(this.source, frame);
  }

  dispose() {
    for (const s of this.analysisSteps) s.material.dispose();
    this.analysisSteps = [];
    if (this.developMat) this.developMat.dispose();
    this.developMat = null;
    this.rtAnalysisA.dispose();
    this.rtAnalysisB.dispose();
    this.mesh.geometry.dispose();
  }
}
