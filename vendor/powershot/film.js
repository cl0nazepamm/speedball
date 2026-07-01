// PowerSHOT film mode — motion-picture film emulation for three.js WebGPU
// (pure TSL).
//
// Models the classic negative -> print -> projection chain in exposure/density
// space rather than as a display-referred LUT:
//
//   sRGB input -> scene linear -> exposure / flicker / gate weave
//     -> halation backscatter (injected into film-layer exposure)
//     -> digital-to-film 3x3 (spectral sensitivity cross-talk)
//     -> negative H&D curves (per dye layer, logE -> density)
//     -> density-domain grain (amplitude peaks in mid densities)
//     -> printer lights -> print stock H&D curves
//     -> projected transmittance -> print-to-display 3x3 -> sRGB out
//
// Stage budget (rule: only genuinely spatial work gets its own pass):
//   1. halation extract   (quarter res)  — threshold scene-linear highlights
//   2. halation blur H    (quarter res)
//   3. halation blur V    (quarter res)
//   4. develop            (full res)     — every point op above, fused
//
// Everything else (curves, matrices, grain, printer lights, negative view,
// power mix) is a point operation and lives inside the single develop pass.

import * as THREE from "three/webgpu";
import {
  vec2, vec3, vec4, float, uniform, texture, screenUV,
  mix, clamp, max, min, dot, abs, floor, fract, sin, cos, sqrt, log, exp,
  step, smoothstep,
} from "three/tsl";

const LN10 = Math.LN10;
const LUM709 = vec3(0.2126, 0.7152, 0.0722);

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

function log10(x) {
  return log(x).mul(1.0 / LN10);
}

function pow10(x) {
  return exp(x.mul(LN10));
}

function exp2u(x) {
  return exp(x.mul(Math.LN2));
}

// piecewise sRGB <-> linear (component-wise on vec3)
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

// Dave Hoskins hash (same family as the PowerShot ISP — stable at large
// pixel coordinates where fract(sin(dot)) hashes break into grid patterns).
function hash13(p) {
  const a = fract(p.mul(0.1031));
  const d = dot(a, vec3(a.z, a.y, a.x).add(31.32));
  const b = a.add(d);
  return fract(b.x.add(b.y).mul(b.z));
}

// Three independent unit gaussians per grain cell via Box-Muller (one full
// pair from u1/u2, a third from u3/u4 — 4 hashes for 3 channels).
function grainGauss3(cell, t) {
  const u1 = hash13(vec3(cell.x, cell.y, t)).max(1e-6);
  const u2 = hash13(vec3(cell.x.add(7.31), cell.y.add(11.07), t.add(3.71)));
  const u3 = hash13(vec3(cell.x.add(2.97), cell.y.add(5.41), t.add(7.93))).max(1e-6);
  const u4 = hash13(vec3(cell.x.add(9.13), cell.y.add(1.83), t.add(13.57)));
  const r1 = sqrt(log(u1).mul(-2.0));
  const a1 = u2.mul(6.2831853);
  const r2 = sqrt(log(u3).mul(-2.0));
  const a2 = u4.mul(6.2831853);
  return vec3(r1.mul(cos(a1)), r1.mul(sin(a1)), r2.mul(cos(a2)));
}

// ---------------------------------------------------------------------------
// film maths (all vec3 component-wise — per dye layer)
// ---------------------------------------------------------------------------

// H&D characteristic curve: logistic in log10 exposure.
// density = dmin + amp / (1 + 10^(-slope * (logE - center)))
function hdCurve(logE, dmin, amp, slope, center) {
  return dmin.add(amp.div(pow10(logE.sub(center).mul(slope).negate()).add(1.0)));
}

function filmMatrix(c, r0, r1, r2) {
  return vec3(dot(c, r0), dot(c, r1), dot(c, r2));
}

// Slow multi-frequency gate weave + per-frame registration jitter, in pixels.
function weaveOffset(ctx) {
  const f = ctx.frame;
  const slowX = sin(f.mul(0.473)).mul(0.55).add(sin(f.mul(0.211).add(1.7)).mul(0.45));
  const slowY = sin(f.mul(0.379).add(0.8)).mul(0.6).add(sin(f.mul(0.157).add(4.1)).mul(0.4));
  const ff = floor(f);
  const jx = hash13(vec3(ff, 1.7, 9.1)).sub(0.5).mul(0.7);
  const jy = hash13(vec3(ff, 6.3, 2.9)).sub(0.5).mul(0.7);
  return vec2(slowX.add(jx), slowY.add(jy)).mul(ctx.P.weave);
}

function weavedUV(ctx) {
  return screenUV.add(ctx.texel.mul(weaveOffset(ctx)));
}

// scene-linear exposure shared by the extract + develop passes.
// inputGamma is a contrast pre-trim for display-referred sources (JPEGs
// already carry a baked tone curve; the print stock adds its own ~1.75x
// system gamma on top). It is a power pivoted at 18% grey — in logE that
// just rescales the scene around the LAD anchor, like shooting a flatter
// scene, so the fitted film curves, mid-grey neutrality and grain/halation
// calibration are untouched. Applied before the exposure gain so the
// exposure slider stays "stops at the film plane".
function sceneExposure(srcTex, ctx, uv) {
  const lin = srgbToLinear(texture(srcTex, uv).rgb);
  const flat = lin.div(0.18).max(1e-7).pow(ctx.P.inputGamma).mul(0.18);
  return flat.mul(exp2u(ctx.P.exposure));
}

// Density-domain grain: gaussian cell noise bilinearly interpolated at the
// grain pitch, amplitude coupled to negative density (strongest in the mids,
// fading in clear toe and dense shoulder — like real RMS granularity).
function densityGrain(density, ctx) {
  const px = screenUV.mul(ctx.resolution).div(max(ctx.P.grainSize, float(0.5)));
  const cell = floor(px);
  const f = fract(px);
  const w = f.mul(f).mul(f.mul(-2.0).add(3.0));
  const t = ctx.frame;
  const g00 = grainGauss3(cell, t);
  const g10 = grainGauss3(cell.add(vec2(1.0, 0.0)), t);
  const g01 = grainGauss3(cell.add(vec2(0.0, 1.0)), t);
  const g11 = grainGauss3(cell.add(vec2(1.0, 1.0)), t);
  const n = mix(mix(g00, g10, w.x), mix(g01, g11, w.x), w.y);

  // colour vs correlated mono grain (dye layers are independent on real stock)
  const grain = mix(vec3(n.x), n, ctx.P.grainSaturation);

  // base sigma 0.018: Kodak 5219 diffuse rms granularity (sigma-D ~0.0085 for
  // the green layer at a 48-micron aperture) scaled to our ~25-micron-
  // equivalent grain pitch. strength 1.0 = datasheet-accurate.
  const x = density.sub(ctx.P.grainDensityCenter).mul(ctx.P.grainDensityWidth);
  const amp = exp(x.mul(x).negate())
    .mul(ctx.P.grainChannel)
    .mul(ctx.P.grainStrength.mul(0.018));
  return density.add(grain.mul(amp));
}

function filmOutputAlpha(sourceSample, effectColor) {
  const sourceAlpha = sourceSample.a.clamp(0.0, 1.0);
  const effectDelta = dot(abs(effectColor.sub(sourceSample.rgb)), LUM709);
  const effectAlpha = effectDelta.sub(0.002).mul(6.0).clamp(0.0, 0.65);
  return sourceAlpha.add(effectAlpha.mul(sourceAlpha.oneMinus())).clamp(0.0, 1.0);
}

// ---------------------------------------------------------------------------
// passes
// ---------------------------------------------------------------------------

// Pass 1 — halation source: scene-linear highlights above threshold, rendered
// into the quarter-res target (the downsample IS the sampling).
function stHaloExtract(srcTex, ctx) {
  const lin = sceneExposure(srcTex, ctx, weavedUV(ctx));
  const lum = dot(lin, LUM709);
  const m = smoothstep(ctx.P.halThreshold, ctx.P.halThreshold.add(ctx.P.halSoftness), lum);
  return lin.mul(m);
}

// Passes 2/3 — separable gaussian with a wide tail; radius scales tap spacing.
function stHaloBlur(tex, ctx, dx, dy) {
  const sigma = 2.6;
  let sum = vec3(0.0);
  let wsum = 0.0;
  for (let i = -6; i <= 6; i += 1) {
    const w = Math.exp(-(i * i) / (2.0 * sigma * sigma));
    const off = ctx.haloTexel.mul(vec2(dx, dy)).mul(ctx.P.halRadius.mul(i));
    sum = sum.add(texture(tex, screenUV.add(off)).rgb.mul(w));
    wsum += w;
  }
  return sum.div(wsum);
}

// Pass 4 — the whole photochemical chain, fused.
function stDevelop(srcTex, ctx, haloTex) {
  const uv = weavedUV(ctx);
  let E = sceneExposure(srcTex, ctx, uv);

  // halation: light that punched through the emulsion and bounced back off
  // the base re-exposes the layers (red-heavy), BEFORE the H&D response.
  if (haloTex) {
    const halo = texture(haloTex, uv).rgb;
    E = E.add(halo.mul(ctx.P.halColor).mul(ctx.P.halStrength));
  }

  // spectral sensitivity cross-talk (rows sum to 1: neutrals stay neutral)
  E = filmMatrix(E, ctx.P.d2fR, ctx.P.d2fG, ctx.P.d2fB).max(1e-6);

  // negative development
  const logE = log10(E);
  let dNeg = hdCurve(logE, ctx.P.negDmin, ctx.P.negAmp, ctx.P.negSlope, ctx.P.negCenter);
  dNeg = densityGrain(dNeg, ctx);

  // printing: light through the negative exposes the print stock. Printer
  // lights are log10 offsets; flicker is a per-frame printer/lamp wobble.
  const flick = hash13(vec3(floor(ctx.frame), 3.3, 5.7)).sub(0.5)
    .mul(ctx.P.flicker.mul(0.05));
  const lights = ctx.P.printLights
    .add(ctx.P.printExposure)
    .add(vec3(1.0, 0.0, -1.0).mul(ctx.P.printWarmth.mul(0.06)))
    .add(flick);
  const logEp = lights.sub(dNeg);
  const dPrint = hdCurve(logEp, ctx.P.prtDmin, ctx.P.prtAmp, ctx.P.prtSlope, ctx.P.prtCenter);

  // projection: transmittance normalised so clear film (Dmin) hits display
  // white, with a projector gain for bright-surround viewing.
  const tPrint = pow10(ctx.P.prtDmin.sub(dPrint));
  let out = tPrint.mul(ctx.P.displayGain);
  out = filmMatrix(out, ctx.P.p2dR, ctx.P.p2dG, ctx.P.p2dB);
  const printView = linearToSrgb(out);

  // negative inspection view: the orange-masked negative itself
  const tNeg = pow10(dNeg.add(ctx.P.negMask).negate());
  const negView = linearToSrgb(tNeg);

  const film = mix(printView, negView, ctx.P.negativeView);
  const sourceSample = texture(srcTex, screenUV);
  const effectColor = mix(sourceSample.rgb, film, ctx.power).clamp(0.0, 1.0);
  return vec4(effectColor, filmOutputAlpha(sourceSample, effectColor));
}

// ---------------------------------------------------------------------------
// uniforms + presets
// ---------------------------------------------------------------------------

export function makeFilmUniforms() {
  return {
    resolution: uniform(new THREE.Vector2(1, 1)),
    texel: uniform(new THREE.Vector2(1, 1)),
    haloTexel: uniform(new THREE.Vector2(1, 1)),
    frame: uniform(0),
    power: uniform(1),
    P: {
      exposure: uniform(0),
      // source contrast trim (not preset data — describes the input medium,
      // so it survives stock switches). Default 0.65 suits display-referred
      // JPEG/video input, whose baked tone curve would otherwise stack with
      // the print's ~1.75x system gamma; set 1.0 for scene-linear sources.
      inputGamma: uniform(0.65),

      d2fR: uniform(new THREE.Vector3(1, 0, 0)),
      d2fG: uniform(new THREE.Vector3(0, 1, 0)),
      d2fB: uniform(new THREE.Vector3(0, 0, 1)),

      negDmin: uniform(new THREE.Vector3(0, 0, 0)),
      negAmp: uniform(new THREE.Vector3(2.15, 2.55, 2.46)),
      negSlope: uniform(new THREE.Vector3(0.4, 0.39, 0.4)),
      negCenter: uniform(new THREE.Vector3(-0.35, -0.34, -0.35)),
      negMask: uniform(new THREE.Vector3(0.19, 0.56, 0.84)),

      grainStrength: uniform(1.0),
      grainSize: uniform(1.6),
      grainSaturation: uniform(0.8),
      grainDensityCenter: uniform(0.45),
      grainDensityWidth: uniform(0.5),
      grainChannel: uniform(new THREE.Vector3(1.1, 1.0, 3.4)),

      halStrength: uniform(0.35),
      halThreshold: uniform(0.55),
      halSoftness: uniform(0.35),
      halRadius: uniform(1.5),
      halColor: uniform(new THREE.Vector3(1.0, 0.35, 0.08)),

      printLights: uniform(new THREE.Vector3(1.915, 1.73, 1.377)),
      printExposure: uniform(0),
      printWarmth: uniform(0),
      prtDmin: uniform(new THREE.Vector3(0.093, 0.089, 0.126)),
      prtAmp: uniform(new THREE.Vector3(4.023, 4.026, 3.973)),
      prtSlope: uniform(new THREE.Vector3(2.056, 1.84, 2.013)),
      prtCenter: uniform(new THREE.Vector3(1.266, 0.96, 0.633)),

      p2dR: uniform(new THREE.Vector3(1, 0, 0)),
      p2dG: uniform(new THREE.Vector3(0, 1, 0)),
      p2dB: uniform(new THREE.Vector3(0, 0, 1)),

      displayGain: uniform(new THREE.Vector3(1.539, 1.451, 1.242)),
      weave: uniform(0.4),
      flicker: uniform(0.12),
      negativeView: uniform(0),
    },
  };
}

// Negative and print curves are logistic fits to the sensitometric plots in
// the public Kodak datasheets (VISION3 5219 / 5207 technical data, VISION
// 2383 print film data sheet), digitized from the plot rasters with the
// camera-stops axis anchoring 18% grey. Printer lights are solved so the
// fitted mid-grey negative prints to the Kodak LAD aim densities
// (1.09/1.06/1.03 Status A); display_gain is the per-channel projection
// calibration that puts the LAD patch at neutral 0.155 display-linear.
// Grain sigma/channel ratios come from the 5219 diffuse rms granularity
// plot. The two 3x3 matrices are NOT datasheet-derived (that needs spectral
// data) — they are mild hand-set cross-talk, rows summing to 1.
export const FILM_PRESETS = {
  kodak_500t: {
    name: "Vision3 500T → 2383 print",
    exposure: 0,
    digital_to_film: [
      [1.08, -0.05, -0.03],
      [-0.04, 1.10, -0.06],
      [-0.02, -0.07, 1.09],
    ],
    neg_dmin: [0, 0, 0],
    neg_amp: [2.1498, 2.549, 2.4561],
    neg_slope: [0.3977, 0.3939, 0.399],
    neg_center: [-0.3522, -0.335, -0.3486],
    neg_mask: [0.19, 0.56, 0.84],
    grain_strength: 1.0,
    grain_size: 1.6,
    grain_saturation: 0.8,
    grain_density_center: 0.45,
    grain_density_width: 0.5,
    grain_channel: [1.1, 1.0, 3.4],
    halation_strength: 0.35,
    halation_threshold: 0.55,
    halation_softness: 0.35,
    halation_radius: 1.5,
    halation_color: [1.0, 0.35, 0.08],
    print_lights: [1.9147, 1.7296, 1.3765],
    print_dmin: [0.0931, 0.0887, 0.1261],
    print_amp: [4.0227, 4.0264, 3.9732],
    print_slope: [2.0563, 1.8395, 2.0128],
    print_center: [1.2655, 0.9598, 0.6333],
    print_to_display: [
      [1.04, -0.02, -0.02],
      [-0.03, 1.06, -0.03],
      [-0.02, -0.04, 1.06],
    ],
    display_gain: [1.539, 1.451, 1.242],
    weave: 0.4,
    flicker: 0.12,
  },
  kodak_200t: {
    name: "Vision3 200T → 2383 print",
    exposure: 0,
    digital_to_film: [
      [1.08, -0.05, -0.03],
      [-0.04, 1.10, -0.06],
      [-0.02, -0.07, 1.09],
    ],
    neg_dmin: [0, 0, 0],
    neg_amp: [2.0865, 2.4854, 2.3796],
    neg_slope: [0.4331, 0.4264, 0.4261],
    neg_center: [-0.436, -0.3787, -0.3819],
    neg_mask: [0.19, 0.56, 0.84],
    grain_strength: 0.75,
    grain_size: 1.45,
    grain_saturation: 0.8,
    grain_density_center: 0.45,
    grain_density_width: 0.5,
    grain_channel: [1.1, 1.0, 3.4],
    halation_strength: 0.3,
    halation_threshold: 0.58,
    halation_softness: 0.35,
    halation_radius: 1.4,
    halation_color: [1.0, 0.34, 0.07],
    print_lights: [1.9149, 1.711, 1.3498],
    print_dmin: [0.0931, 0.0887, 0.1261],
    print_amp: [4.0227, 4.0264, 3.9732],
    print_slope: [2.0563, 1.8395, 2.0128],
    print_center: [1.2655, 0.9598, 0.6333],
    print_to_display: [
      [1.04, -0.02, -0.02],
      [-0.03, 1.06, -0.03],
      [-0.02, -0.04, 1.06],
    ],
    display_gain: [1.539, 1.451, 1.242],
    weave: 0.35,
    flicker: 0.1,
  },
  kodak_250d: {
    name: "Vision3 250D → 2383 print",
    exposure: 0,
    digital_to_film: [
      [1.05, -0.03, -0.02],
      [-0.02, 1.06, -0.04],
      [-0.01, -0.04, 1.05],
    ],
    neg_dmin: [0, 0, 0],
    neg_amp: [2.1455, 2.5679, 2.4498],
    neg_slope: [0.414, 0.4063, 0.4137],
    neg_center: [-0.4755, -0.4185, -0.4539],
    neg_mask: [0.19, 0.56, 0.84],
    grain_strength: 0.65,
    grain_size: 1.35,
    grain_saturation: 0.8,
    grain_density_center: 0.45,
    grain_density_width: 0.5,
    grain_channel: [1.1, 1.0, 3.4],
    halation_strength: 0.25,
    halation_threshold: 0.6,
    halation_softness: 0.35,
    halation_radius: 1.3,
    halation_color: [1.0, 0.32, 0.06],
    print_lights: [1.9669, 1.7788, 1.4258],
    print_dmin: [0.0931, 0.0887, 0.1261],
    print_amp: [4.0227, 4.0264, 3.9732],
    print_slope: [2.0563, 1.8395, 2.0128],
    print_center: [1.2655, 0.9598, 0.6333],
    print_to_display: [
      [1.04, -0.02, -0.02],
      [-0.03, 1.06, -0.03],
      [-0.02, -0.04, 1.06],
    ],
    display_gain: [1.539, 1.451, 1.242],
    weave: 0.3,
    flicker: 0.08,
  },
  kodak_50d: {
    name: "Vision3 50D → 2383 print",
    exposure: 0,
    digital_to_film: [
      [1.05, -0.03, -0.02],
      [-0.02, 1.06, -0.04],
      [-0.01, -0.04, 1.05],
    ],
    neg_dmin: [0, 0, 0],
    neg_amp: [2.3181, 2.7327, 2.663],
    neg_slope: [0.3806, 0.3831, 0.3784],
    neg_center: [-0.5894, -0.4837, -0.4368],
    neg_mask: [0.19, 0.56, 0.84],
    grain_strength: 0.45,
    grain_size: 1.2,
    grain_saturation: 0.8,
    grain_density_center: 0.45,
    grain_density_width: 0.5,
    grain_channel: [1.1, 1.0, 3.4],
    halation_strength: 0.2,
    halation_threshold: 0.62,
    halation_softness: 0.35,
    halation_radius: 1.2,
    halation_color: [1.0, 0.32, 0.06],
    print_lights: [2.1113, 1.899, 1.5235],
    print_dmin: [0.0931, 0.0887, 0.1261],
    print_amp: [4.0227, 4.0264, 3.9732],
    print_slope: [2.0563, 1.8395, 2.0128],
    print_center: [1.2655, 0.9598, 0.6333],
    print_to_display: [
      [1.04, -0.02, -0.02],
      [-0.03, 1.06, -0.03],
      [-0.02, -0.04, 1.06],
    ],
    display_gain: [1.539, 1.451, 1.242],
    weave: 0.3,
    flicker: 0.08,
  },
};

export const FILM_PRESET_KEYS = Object.keys(FILM_PRESETS);

export function applyFilmPreset(ctx, preset) {
  const P = ctx.P;
  P.exposure.value = preset.exposure;
  P.d2fR.value.set(...preset.digital_to_film[0]);
  P.d2fG.value.set(...preset.digital_to_film[1]);
  P.d2fB.value.set(...preset.digital_to_film[2]);
  P.negDmin.value.set(...preset.neg_dmin);
  P.negAmp.value.set(...preset.neg_amp);
  P.negSlope.value.set(...preset.neg_slope);
  P.negCenter.value.set(...preset.neg_center);
  P.negMask.value.set(...preset.neg_mask);
  P.grainStrength.value = preset.grain_strength;
  P.grainSize.value = preset.grain_size;
  P.grainSaturation.value = preset.grain_saturation;
  P.grainDensityCenter.value = preset.grain_density_center;
  P.grainDensityWidth.value = preset.grain_density_width;
  P.grainChannel.value.set(...preset.grain_channel);
  P.halStrength.value = preset.halation_strength;
  P.halThreshold.value = preset.halation_threshold;
  P.halSoftness.value = preset.halation_softness;
  P.halRadius.value = preset.halation_radius;
  P.halColor.value.set(...preset.halation_color);
  P.printLights.value.set(...preset.print_lights);
  P.prtDmin.value.set(...preset.print_dmin);
  P.prtAmp.value.set(...preset.print_amp);
  P.prtSlope.value.set(...preset.print_slope);
  P.prtCenter.value.set(...preset.print_center);
  P.p2dR.value.set(...preset.print_to_display[0]);
  P.p2dG.value.set(...preset.print_to_display[1]);
  P.p2dB.value.set(...preset.print_to_display[2]);
  P.displayGain.value.set(...preset.display_gain);
  P.weave.value = preset.weave;
  P.flicker.value = preset.flicker;
}

export const FILM_STAGE_DEFS = [
  { id: "halation", label: "Halation backscatter" }, // 3 quarter-res passes
  { id: "develop", label: "Negative + print develop" }, // fused point pass
];

// ---------------------------------------------------------------------------
// runner — mirrors the PowerShot Pipeline contract (persistent material per
// pass, rebuilt only when source / size / enabled set changes).
// ---------------------------------------------------------------------------

export class FilmPipeline {
  constructor(renderer) {
    this.renderer = renderer;
    this.ctx = makeFilmUniforms();

    const opts = {
      type: THREE.HalfFloatType,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
      colorSpace: THREE.NoColorSpace,
    };
    // quarter-res ping-pong pair for the halation halo
    this.rtHaloA = new THREE.RenderTarget(1, 1, opts);
    this.rtHaloB = new THREE.RenderTarget(1, 1, { ...opts });

    this.quadScene = new THREE.Scene();
    this.quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), null);
    this.mesh.frustumCulled = false;
    this.quadScene.add(this.mesh);

    this.enabled = new Set(["halation", "develop"]);
    this.source = null;
    this.size = { w: 0, h: 0 };
    this.haloSteps = []; // [{ material, target }]
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
    this.dirty = true;
  }

  setSize(w, h) {
    if (w === this.size.w && h === this.size.h) return;
    const hw = Math.max(1, Math.round(w / 4));
    const hh = Math.max(1, Math.round(h / 4));
    this.rtHaloA.setSize(hw, hh);
    this.rtHaloB.setSize(hw, hh);
    this.size = { w, h };
    this.ctx.resolution.value.set(w, h);
    this.ctx.texel.value.set(1 / w, 1 / h);
    this.ctx.haloTexel.value.set(1 / hw, 1 / hh);
    this.dirty = true;
  }

  setEnabled(id, on) {
    const hasStage = this.enabled.has(id);
    if (on === hasStage) return;
    if (on) this.enabled.add(id);
    else this.enabled.delete(id);
    this.dirty = true;
  }

  _rebuild() {
    for (const s of this.haloSteps) s.material.dispose();
    if (this.developMat) this.developMat.dispose();
    this.haloSteps = [];
    this.developMat = null;
    this.dirty = false;
    if (!this.source) return;

    const halation = this.enabled.has("halation");
    if (halation) {
      this.haloSteps.push({
        material: this._mat(stHaloExtract(this.source, this.ctx)),
        target: this.rtHaloA,
      });
      this.haloSteps.push({
        material: this._mat(stHaloBlur(this.rtHaloA.texture, this.ctx, 1, 0)),
        target: this.rtHaloB,
      });
      this.haloSteps.push({
        material: this._mat(stHaloBlur(this.rtHaloB.texture, this.ctx, 0, 1)),
        target: this.rtHaloA,
      });
    }

    this.developMat = this._mat(
      stDevelop(this.source, this.ctx, halation ? this.rtHaloA.texture : null),
    );
    this.developMat.transparent = false;
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
      for (const step of this.haloSteps) {
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
    for (const s of this.haloSteps) s.material.dispose();
    this.haloSteps = [];
    if (this.developMat) this.developMat.dispose();
    this.developMat = null;
    this.rtHaloA.dispose();
    this.rtHaloB.dispose();
    this.mesh.geometry.dispose();
  }
}
