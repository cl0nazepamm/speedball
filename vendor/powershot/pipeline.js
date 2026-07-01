// PowerSHOT realtime ISP — TSL stage library + ping-pong runner (three.js WebGPU).
//
// Each stage is a pure function (inputTexture, ctx) -> colorNode. The runner
// renders every enabled stage into its own half-float render target, feeding the
// previous target's texture into the next stage. Everything runs on the GPU; the
// only global statistics we need (white balance) are frozen to the preset's
// nominal gains for now (gray-world-on-GPU can replace that later).
//
// Domain convention: pixel values live in a 0..255 "signal"
// space. We sample textures in 0..1 and multiply up to 255 at the first stage so
// every constant (highlight_clip, shadow_crush, noise*255, thresholds) matches the
// Python reference 1:1, then divide back to 0..1 at the final stage.

import * as THREE from "three/webgpu";
import {
  vec2, vec3, vec4, float, uniform, texture, screenUV,
  mix, clamp, max, min, dot, abs, floor, fract, sin, cos, sqrt, log, mod, step,
  exp,
} from "three/tsl";

const LEVELS = 255.0;

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

// sample input texture at a uv offset given in *pixels*
function tapPx(tex, uvN, texel, dx, dy) {
  return texture(tex, uvN.add(texel.mul(vec2(dx, dy))));
}

// Dave Hoskins hashes — good distribution with NO large-argument sin (the
// classic fract(sin(dot)) hash degenerates into grid/diagonal patterns once
// pixel coordinates get into the hundreds, which reads as "horrible" grain).
function hash12(p) {
  const a = fract(vec3(p.x, p.y, p.x).mul(0.1031));
  const d = dot(a, vec3(a.y, a.z, a.x).add(33.33));
  const b = a.add(d);
  return fract(b.x.add(b.y).mul(b.z));
}

function hash13(p) {
  const a = fract(p.mul(0.1031));
  const d = dot(a, vec3(a.z, a.y, a.x).add(31.32));
  const b = a.add(d);
  return fract(b.x.add(b.y).mul(b.z));
}

// unit-variance gaussian via Box-Muller from two independent uniform hashes.
// `gaussFixed` is time-invariant (fixed-pattern noise); `gaussTemporal` folds
// the frame counter in as a third hash dimension so grain shimmers each frame.
function gaussFixed(p, salt) {
  const q = p.add(salt);
  const u1 = hash12(q).max(1e-6);
  const u2 = hash12(q.add(vec2(19.19, 7.41))).max(1e-6);
  return sqrt(log(u1).mul(-2.0)).mul(cos(u2.mul(6.2831853)));
}

function gaussTemporal(p, t, salt) {
  const u1 = hash13(vec3(p.x, p.y, t.add(salt))).max(1e-6);
  const u2 = hash13(vec3(p.x.add(11.0), p.y.add(3.0), t.add(salt).add(1.7))).max(1e-6);
  return sqrt(log(u1).mul(-2.0)).mul(cos(u2.mul(6.2831853)));
}

// RGGB Bayer phase masks for the current fragment. Returns floats {isR,isGr,isGb,isB}.
function bayerPhase(uvN, ctx) {
  const p = floor(uvN.mul(ctx.resolution));
  const px = mod(p.x, 2.0); // 0 or 1
  const py = mod(p.y, 2.0);
  const xEven = px.oneMinus();
  const yEven = py.oneMinus();
  return {
    isR: yEven.mul(xEven),
    isGr: yEven.mul(px),
    isGb: py.mul(xEven),
    isB: py.mul(px),
  };
}

// ---------------------------------------------------------------------------
// stages
// ---------------------------------------------------------------------------

// Stage 1 also handles the implicit downsample: sampling the (larger) source
// image into the sensor-resolution target IS the downsample.
function stInput(tex, ctx) {
  return texture(tex, screenUV).rgb.mul(LEVELS);
}

function stCopy(tex, ctx) {
  return texture(tex, screenUV).rgb;
}

function stBarrel(tex, ctx) {
  const k = ctx.P.barrel;
  const c = screenUV.sub(0.5).mul(vec2(2.0, 2.0)); // -1..1, aspect-naive (matches py)
  const r2 = dot(c, c);
  const factor = float(1.0).add(k.mul(r2));
  // invert: r_distorted = r*(1+k r^2); sample source along r/r_distorted
  const srcUv = c.div(factor).mul(0.5).add(0.5);
  return texture(tex, srcUv).rgb;
}

function stChromatic(tex, ctx) {
  const ca = ctx.P.ca; // pixels
  const maxDim = max(ctx.resolution.x, ctx.resolution.y);
  const rScale = float(1.0).add(ca.div(maxDim));
  const bScale = float(1.0).sub(ca.mul(0.7).div(maxDim));
  const c = screenUV.sub(0.5);
  const rUv = c.div(rScale).add(0.5);
  const bUv = c.div(bScale).add(0.5);
  const r = texture(tex, rUv).r;
  const g = texture(tex, screenUV).g;
  const b = texture(tex, bUv).b;
  return vec3(r, g, b);
}

// Cheap lens point-spread function. This lives before the sensor path so edges
// get softened before Bayer sampling and sharpening, like small compact optics.
function stLensPsf(tex, ctx) {
  const t = ctx.texel;
  const c = texture(tex, screenUV).rgb;
  let blur = c.mul(0.28);
  blur = blur.add(tapPx(tex, screenUV, t, -1, 0).rgb.mul(0.15));
  blur = blur.add(tapPx(tex, screenUV, t, 1, 0).rgb.mul(0.15));
  blur = blur.add(tapPx(tex, screenUV, t, 0, -1).rgb.mul(0.15));
  blur = blur.add(tapPx(tex, screenUV, t, 0, 1).rgb.mul(0.15));
  blur = blur.add(tapPx(tex, screenUV, t, -1, -1).rgb.mul(0.03));
  blur = blur.add(tapPx(tex, screenUV, t, 1, -1).rgb.mul(0.03));
  blur = blur.add(tapPx(tex, screenUV, t, -1, 1).rgb.mul(0.03));
  blur = blur.add(tapPx(tex, screenUV, t, 1, 1).rgb.mul(0.03));
  return mix(c, blur, ctx.P.lensSoftness).clamp(0.0, LEVELS);
}

function bloomSource(c, ctx) {
  const lum = max(max(c.r, c.g), c.b);
  const denom = max(float(1.0), float(LEVELS).sub(ctx.P.ccdBloomThreshold));
  const mask = lum.sub(ctx.P.ccdBloomThreshold).div(denom).clamp(0.0, 1.0).pow(0.5);
  return c.mul(mask);
}

// CCD column bleed: bright saturated charge smears vertically before CFA
// sampling. Use dense column integration; sparse far taps create repeated
// highlight ghosts instead of a continuous readout smear.
//
// Split into three passes for speed: the smear is band-limited by its own
// exp(-|dy|/24) kernel, so it can be integrated at quarter vertical
// resolution with no visible difference. Pass 1 extracts the thresholded
// bloom source at full width / quarter height (thresholding stays per-pixel:
// each quarter row explicitly averages its 4 masked source rows). Pass 2
// smears vertically in quarter space — every source row participates, which
// is denser integration than the old step-2 tap ladder. Pass 3 composites
// the bilinearly-upsampled smear over the full-res image.
function stCcdBloomExtract(tex, ctx) {
  // runs on the quarter-height target; screenUV.y lands on the exact center
  // of each 4-row group, so dy = ±0.5/±1.5 hits the 4 row centers exactly.
  const t = ctx.texel;
  let sum = vec3(0.0);
  for (const dy of [-1.5, -0.5, 0.5, 1.5]) {
    sum = sum.add(bloomSource(tapPx(tex, screenUV, t, 0, dy).rgb, ctx));
  }
  return sum.mul(0.25);
}

function stCcdBloomSmear(tex, ctx) {
  let smear = vec3(0.0);
  let wsum = 0.0;
  for (let qdy = -16; qdy <= 16; qdy += 1) {
    const ady = Math.abs(qdy * 4);
    const w = Math.exp(-ady / 24.0) * (1.0 - ady / 96.0);
    smear = smear.add(texture(tex, screenUV.add(ctx.bloomTexel.mul(vec2(0.0, qdy)))).rgb.mul(w));
    wsum += w;
  }
  return smear.div(wsum);
}

function stCcdBloomComposite(tex, ctx, smearTex) {
  const c = texture(tex, screenUV).rgb;
  const smear = texture(smearTex, screenUV).rgb;
  return c.add(smear.mul(ctx.P.ccdBloom)).clamp(0.0, LEVELS);
}

// --- Bayer domain ---

function stMosaic(tex, ctx) {
  return mosaicColor(texture(tex, screenUV).rgb, ctx);
}

function mosaicColor(c, ctx) {
  const ph = bayerPhase(screenUV, ctx);
  const bayer = c.r.mul(ph.isR)
    .add(c.g.mul(ph.isGr.add(ph.isGb)))
    .add(c.b.mul(ph.isB));
  return vec3(bayer);
}

function stWhiteBalance(tex, ctx) {
  return whiteBalanceColor(texture(tex, screenUV).rgb, ctx);
}

function whiteBalanceColor(c, ctx) {
  const b = c.r;
  const ph = bayerPhase(screenUV, ctx);
  const gain = ctx.P.wbR.mul(ph.isR)
    .add(ctx.P.wbG.mul(ph.isGr.add(ph.isGb)))
    .add(ctx.P.wbB.mul(ph.isB));
  return vec3(b.mul(gain).clamp(0.0, LEVELS));
}

function stBlackLevel(tex, ctx) {
  return blackLevelColor(texture(tex, screenUV).rgb, ctx);
}

function blackLevelColor(c, ctx) {
  return vec3(blackLevelValue(c.r, ctx));
}

function blackLevelValue(b, ctx) {
  const ph = bayerPhase(screenUV, ctx);
  const off = ctx.P.blR.mul(ph.isR)
    .add(ctx.P.blGr.mul(ph.isGr))
    .add(ctx.P.blGb.mul(ph.isGb))
    .add(ctx.P.blB.mul(ph.isB));
  return b.add(off).clamp(0.0, LEVELS);
}

function samePhase3x3(tex, ctx) {
  const t = ctx.texel;
  const c = texture(tex, screenUV).r;
  let sum = float(0.0);
  sum = sum.add(tapPx(tex, screenUV, t, -2, -2).r);
  sum = sum.add(tapPx(tex, screenUV, t, 0, -2).r);
  sum = sum.add(tapPx(tex, screenUV, t, 2, -2).r);
  sum = sum.add(tapPx(tex, screenUV, t, -2, 0).r);
  sum = sum.add(tapPx(tex, screenUV, t, 2, 0).r);
  sum = sum.add(tapPx(tex, screenUV, t, -2, 2).r);
  sum = sum.add(tapPx(tex, screenUV, t, 0, 2).r);
  sum = sum.add(tapPx(tex, screenUV, t, 2, 2).r);
  return { c, sum, avg: sum.div(8.0) };
}

function stBayerNoise(tex, ctx) {
  return bayerNoiseColor(texture(tex, screenUV).rgb, ctx);
}

function bayerNoiseColor(c, ctx) {
  return vec3(bayerNoiseValue(c.r, ctx));
}

function bayerNoiseValue(b, ctx) {
  const p = floor(screenUV.mul(ctx.resolution));
  const t = ctx.frame;
  const ns = ctx.noiseScale;

  // fixed-pattern noise (time-invariant): per-column, per-row, per-pixel offsets + gain
  const colFpn = gaussFixed(vec2(p.x, 0.0), vec2(2.70, 0.0)).mul(ctx.P.colFpn.mul(LEVELS)).mul(ns);
  const rowFpn = gaussFixed(vec2(0.0, p.y), vec2(0.0, 5.10)).mul(ctx.P.rowFpn.mul(LEVELS)).mul(ns);
  const dsnu = gaussFixed(p, vec2(0.50, 0.50)).mul(ctx.P.dsnu.mul(LEVELS)).mul(ns);
  const prnuGain = float(1.0).add(gaussFixed(p, vec2(3.30, 7.70)).mul(ctx.P.prnu).mul(ns));

  let sig = b.add(colFpn).add(rowFpn).add(dsnu).mul(prnuGain);

  // signal-dependent read + shot noise (temporal): same curve as PowerShot.py.
  const sl = sig.div(LEVELS).clamp(0.0, 1.0);
  const readStd = ctx.P.noise.mul(LEVELS).mul(float(0.9).add(sl.oneMinus().mul(1.8)));
  const read = gaussTemporal(p, t, 0.0).mul(readStd).mul(ns);
  const shotStd = sqrt(sl.add(1e-4)).mul(ctx.P.colorNoise.mul(LEVELS));
  const shot = gaussTemporal(p, t, 37.0).mul(shotStd).mul(ns);

  sig = sig.add(read).add(shot);

  return sig.clamp(0.0, LEVELS);
}

// Remove isolated Bayer-domain hot/dead samples before they demosaic into
// colored plus-shaped dots. This is deliberately same-phase only.
function stDeadPixelCorrection(tex, ctx) {
  const n = samePhase3x3(tex, ctx);
  const isolated = step(ctx.P.dpcThreshold, abs(n.c.sub(n.avg)));
  return vec3(mix(n.c, n.avg, isolated).clamp(0.0, LEVELS));
}

// Optical low-pass filter: phase-aware diamond blur on same-color neighbors.
function stAAF(tex, ctx) {
  const s = ctx.P.aaf;
  const n = samePhase3x3(tex, ctx);
  const filtered = n.c.mul(8.0).add(n.sum).div(16.0);
  return vec3(mix(n.c, filtered, s));
}

function greenGuideAt(tex, ctx, uvN) {
  const t = ctx.texel;
  const b = texture(tex, uvN).r;
  const guide = b.mul(0.50)
    .add(tapPx(tex, uvN, t, 0, -2).r.mul(-0.125))
    .add(tapPx(tex, uvN, t, 0, -1).r.mul(0.25))
    .add(tapPx(tex, uvN, t, -2, 0).r.mul(-0.125))
    .add(tapPx(tex, uvN, t, -1, 0).r.mul(0.25))
    .add(tapPx(tex, uvN, t, 1, 0).r.mul(0.25))
    .add(tapPx(tex, uvN, t, 2, 0).r.mul(-0.125))
    .add(tapPx(tex, uvN, t, 0, 1).r.mul(0.25))
    .add(tapPx(tex, uvN, t, 0, 2).r.mul(-0.125));
  const ph = bayerPhase(uvN, ctx);
  const isGreen = ph.isGr.add(ph.isGb);
  return mix(guide, b, isGreen);
}

// Joint bilateral Bayer NR, split into two passes. The naive single-pass form
// recomputes the 9-tap green guide for all 25 footprint pixels (~250 fetches);
// instead pass 1 computes each pixel's guide once and packs it next to the
// Bayer value (the Bayer-domain image is replicated grey, so .g is free), and
// pass 2 reads value+guide together — one fetch per tap. Same math, ~10x fewer
// fetches.
function stBnrGuide(tex, ctx) {
  const b = texture(tex, screenUV).r;
  const guide = greenGuideAt(tex, ctx, screenUV);
  return vec3(b, guide, b);
}

// Compact realtime stand-in for the Python joint bilateral Bayer NR. It filters
// same-color Bayer samples before demosaic, which prevents chroma speckles from
// becoming stable red/blue dots in the RGB image.
function stBayerDenoise(tex, ctx) {
  const strength = ctx.P.bayerNR;
  const center = texture(tex, screenUV);
  const c = center.r;
  const centerGuide = center.g;
  const spatialSigma = max(ctx.P.bnrSpatial, float(1e-3));
  const rangeSigma = max(ctx.P.bnrRange, float(1e-3));
  let sum = c;
  let wsum = float(1.0);

  for (const [sx, sy] of [
    [-2, -2], [-1, -2], [0, -2], [1, -2], [2, -2],
    [-2, -1], [-1, -1], [0, -1], [1, -1], [2, -1],
    [-2, 0], [-1, 0],            [1, 0], [2, 0],
    [-2, 1], [-1, 1], [0, 1], [1, 1], [2, 1],
    [-2, 2], [-1, 2], [0, 2], [1, 2], [2, 2],
  ]) {
    const dist2 = float(sx * sx + sy * sy);
    const spatialW = exp(dist2.div(spatialSigma.mul(spatialSigma).mul(-2.0)));
    const tap = tapPx(tex, screenUV, ctx.texel, sx * 2, sy * 2);
    const diff = tap.g.sub(centerGuide);
    const rangeW = exp(diff.mul(diff).div(rangeSigma.mul(rangeSigma).mul(-2.0)));
    const w = rangeW.mul(spatialW);
    sum = sum.add(tap.r.mul(w));
    wsum = wsum.add(w);
  }

  const filtered = sum.div(max(wsum, float(1e-5)));
  return vec3(mix(c, filtered, strength).clamp(0.0, LEVELS));
}

function demosaicBilinear(tex, ctx) {
  const t = ctx.texel;
  const c = texture(tex, screenUV).r;
  const N = tapPx(tex, screenUV, t, 0, -1).r;
  const S = tapPx(tex, screenUV, t, 0, 1).r;
  const E = tapPx(tex, screenUV, t, 1, 0).r;
  const W = tapPx(tex, screenUV, t, -1, 0).r;
  const NE = tapPx(tex, screenUV, t, 1, -1).r;
  const NW = tapPx(tex, screenUV, t, -1, -1).r;
  const SE = tapPx(tex, screenUV, t, 1, 1).r;
  const SW = tapPx(tex, screenUV, t, -1, 1).r;

  const ortho4 = N.add(S).add(E).add(W).mul(0.25);
  const diag4 = NE.add(NW).add(SE).add(SW).mul(0.25);
  const horiz2 = E.add(W).mul(0.5);
  const vert2 = N.add(S).mul(0.5);

  const ph = bayerPhase(screenUV, ctx);
  const r = c.mul(ph.isR).add(horiz2.mul(ph.isGr)).add(vert2.mul(ph.isGb)).add(diag4.mul(ph.isB));
  const g = ortho4.mul(ph.isR.add(ph.isB)).add(c.mul(ph.isGr.add(ph.isGb)));
  const b = diag4.mul(ph.isR).add(vert2.mul(ph.isGr)).add(horiz2.mul(ph.isGb)).add(c.mul(ph.isB));
  return vec3(r, g, b);
}

function demosaicEdgeAware(tex, ctx) {
  const t = ctx.texel;
  const c = texture(tex, screenUV).r;
  const N = tapPx(tex, screenUV, t, 0, -1).r;
  const S = tapPx(tex, screenUV, t, 0, 1).r;
  const E = tapPx(tex, screenUV, t, 1, 0).r;
  const W = tapPx(tex, screenUV, t, -1, 0).r;
  const NE = tapPx(tex, screenUV, t, 1, -1).r;
  const NW = tapPx(tex, screenUV, t, -1, -1).r;
  const SE = tapPx(tex, screenUV, t, 1, 1).r;
  const SW = tapPx(tex, screenUV, t, -1, 1).r;

  const horiz2 = E.add(W).mul(0.5);
  const vert2 = N.add(S).mul(0.5);
  const diag4 = NE.add(NW).add(SE).add(SW).mul(0.25);
  const gradH = abs(E.sub(W));
  const gradV = abs(N.sub(S));
  const edgeGreen = mix(horiz2, vert2, step(gradH, gradV));

  const ph = bayerPhase(screenUV, ctx);
  const r = c.mul(ph.isR).add(horiz2.mul(ph.isGr)).add(vert2.mul(ph.isGb)).add(diag4.mul(ph.isB));
  const g = edgeGreen.mul(ph.isR.add(ph.isB)).add(c.mul(ph.isGr.add(ph.isGb)));
  const b = diag4.mul(ph.isR).add(vert2.mul(ph.isGr)).add(horiz2.mul(ph.isGb)).add(c.mul(ph.isB));
  return vec3(r, g, b);
}

function stDemosaic(tex, ctx) {
  return mix(demosaicBilinear(tex, ctx), demosaicEdgeAware(tex, ctx), ctx.P.demosaicSharp).clamp(0.0, LEVELS);
}

// Chroma noise reduction: keep the center pixel's luma (so grain/detail stays
// sharp) but replace its chroma with a 3x3 neighborhood average. This is what
// kills the colored-dot static that Bayer noise + demosaic leaves in shadows,
// exactly like a real camera's chroma NR. Strength blends toward the original.
const LUMA = vec3(0.299, 0.587, 0.114);
function stChromaDenoise(tex, ctx) {
  const t = ctx.texel;
  const c = texture(tex, screenUV).rgb;
  let sum = c;
  sum = sum.add(tapPx(tex, screenUV, t, -1, -1).rgb);
  sum = sum.add(tapPx(tex, screenUV, t, 0, -1).rgb);
  sum = sum.add(tapPx(tex, screenUV, t, 1, -1).rgb);
  sum = sum.add(tapPx(tex, screenUV, t, -1, 0).rgb);
  sum = sum.add(tapPx(tex, screenUV, t, 1, 0).rgb);
  sum = sum.add(tapPx(tex, screenUV, t, -1, 1).rgb);
  sum = sum.add(tapPx(tex, screenUV, t, 0, 1).rgb);
  sum = sum.add(tapPx(tex, screenUV, t, 1, 1).rgb);
  const avg = sum.div(9.0);

  // sharp luma + smoothed chroma
  const y = dot(c, LUMA);
  const denoised = vec3(y).add(avg.sub(dot(avg, LUMA)));
  return mix(c, denoised, ctx.P.chromaNR).clamp(0.0, LEVELS);
}

// --- RGB ISP ---

function stCCM(tex, ctx) {
  return ccmColor(texture(tex, screenUV).rgb, ctx);
}

function ccmColor(c, ctx) {
  const r = dot(c, ctx.P.ccm0);
  const g = dot(c, ctx.P.ccm1);
  const b = dot(c, ctx.P.ccm2);
  return vec3(r, g, b).clamp(0.0, LEVELS);
}

function stTone(tex, ctx) {
  return toneColor(texture(tex, screenUV).rgb, ctx);
}

function toneColor(c, ctx) {
  c = min(c, ctx.P.hiClip);
  c = c.div(ctx.P.hiClip).mul(LEVELS);
  c = c.div(LEVELS).clamp(0.0, 1.0).pow(ctx.P.gamma).mul(LEVELS);
  // crush shadows: anything below threshold -> 0
  c = c.mul(step(ctx.P.shadow, c));
  return c.clamp(0.0, LEVELS);
}

function stSaturation(tex, ctx) {
  return saturationColor(texture(tex, screenUV).rgb, ctx);
}

function saturationColor(c, ctx) {
  const gray = c.r.add(c.g).add(c.b).div(3.0);
  return mix(vec3(gray), c, ctx.P.sat).clamp(0.0, LEVELS);
}

function stVignette(tex, ctx) {
  return vignetteColor(texture(tex, screenUV).rgb, ctx);
}

function vignetteColor(c, ctx) {
  const res = ctx.resolution;
  const cxy = res.mul(0.5);
  const pos = screenUV.mul(res);
  const maxR = sqrt(dot(cxy, cxy));
  const d = pos.sub(cxy);
  const r = sqrt(dot(d, d)).div(maxR);
  const cosT = cos(r.mul(0.7853982)); // r * pi/4
  const falloff = float(1.0).sub(ctx.P.vignette.mul(cosT.pow(4.0).oneMinus()));
  return c.mul(falloff).clamp(0.0, LEVELS);
}

function stDigitalPointStack(tex, ctx, ids) {
  let c = texture(tex, screenUV).rgb;
  return digitalPointStackColor(c, ctx, ids);
}

function stInputDigitalPointStack(tex, ctx, ids) {
  let c = texture(tex, screenUV).rgb.mul(LEVELS);
  return digitalPointStackColor(c, ctx, ids);
}

function digitalPointStackColor(c, ctx, ids) {
  if (ids.has("mosaic")) c = mosaicColor(c, ctx);
  if (ids.has("blacklevel")) c = blackLevelColor(c, ctx);
  if (ids.has("noise")) c = bayerNoiseColor(c, ctx);
  if (ids.has("wb")) c = whiteBalanceColor(c, ctx);
  if (ids.has("ccm")) c = ccmColor(c, ctx);
  if (ids.has("tone")) c = toneColor(c, ctx);
  if (ids.has("saturation")) c = saturationColor(c, ctx);
  if (ids.has("vignette")) c = vignetteColor(c, ctx);
  return c;
}

function stEdgeEnhance(tex, ctx) {
  const t = ctx.texel;
  const lum = (s) => s.r.mul(0.299).add(s.g.mul(0.587)).add(s.b.mul(0.114));
  const c = texture(tex, screenUV);
  const center = lum(c).mul(8.0);
  let sum = center;
  sum = sum.sub(lum(tapPx(tex, screenUV, t, -1, -1)));
  sum = sum.sub(lum(tapPx(tex, screenUV, t, 0, -1)));
  sum = sum.sub(lum(tapPx(tex, screenUV, t, 1, -1)));
  sum = sum.sub(lum(tapPx(tex, screenUV, t, -1, 0)));
  sum = sum.sub(lum(tapPx(tex, screenUV, t, 1, 0)));
  sum = sum.sub(lum(tapPx(tex, screenUV, t, -1, 1)));
  sum = sum.sub(lum(tapPx(tex, screenUV, t, 0, 1)));
  sum = sum.sub(lum(tapPx(tex, screenUV, t, 1, 1)));
  const edge = sum.div(8.0);
  // coring: zero weak edges (noise), then gain + clamp halos
  const cored = edge.mul(step(ctx.P.eeThresh, abs(edge)));
  const enhanced = cored.mul(ctx.P.eeGain).clamp(-40.0, 40.0);
  return c.rgb.add(vec3(enhanced)).clamp(0.0, LEVELS);
}

function quantize(v, q) {
  return floor(v.div(q).add(0.5)).mul(q);
}

function rgbToYCbCr(c) {
  const y = c.r.mul(0.299).add(c.g.mul(0.587)).add(c.b.mul(0.114));
  const cb = c.r.mul(-0.168736).add(c.g.mul(-0.331264)).add(c.b.mul(0.5));
  const cr = c.r.mul(0.5).add(c.g.mul(-0.418688)).add(c.b.mul(-0.081312));
  return vec3(y, cb, cr);
}

function yCbCrToRgb(c) {
  return vec3(
    c.x.add(c.z.mul(1.402)),
    c.x.sub(c.y.mul(0.344136)).sub(c.z.mul(0.714136)),
    c.x.add(c.y.mul(1.772)),
  );
}

function softBand(value, start, end) {
  return value.sub(start).div(end - start).clamp(0.0, 1.0);
}

function smoothBand(value, start, end) {
  const t = softBand(value, start, end);
  return t.mul(t).mul(float(3.0).sub(t.mul(2.0)));
}

function analogSampleYcc(tex, ctx, uv, pxOffset) {
  return rgbToYCbCr(texture(tex, uv.add(ctx.texel.mul(pxOffset))).rgb);
}

function analogCarrier(p, fieldPhase) {
  return sin(p.x.mul(1.5707963).add(fieldPhase));
}

function stAnalogVhs(tex, ctx) {
  const p = floor(screenUV.mul(ctx.resolution));
  const t = ctx.frame.mul(0.037);
  const strength = ctx.P.analogStrength.clamp(0.0, 3.0);
  const tracking = ctx.P.analogTracking.clamp(0.0, 3.0);
  const chromaBleed = ctx.P.analogChromaBleed.clamp(0.0, 3.0);
  const ringing = ctx.P.analogRinging.clamp(0.0, 3.0);
  const tapeNoise = ctx.P.analogTapeNoise.clamp(0.0, 3.0);
  const bandMask = ctx.P.analogBandMask.clamp(0.0, 3.0);
  const edgeWaveAmount = ctx.P.analogEdgeWave.clamp(0.0, 3.0);
  const dropoutAmount = ctx.P.analogDropouts.clamp(0.0, 3.0);
  const scanlines = ctx.P.analogScanlines.clamp(0.0, 3.0);
  const headSwitch = ctx.P.analogHeadSwitch.clamp(0.0, 3.0);
  const active = strength.mul(step(0.001, strength));
  const fieldLine = floor(p.y.div(2.0));

  const rowNoise = hash12(vec2(p.y.mul(0.37), floor(t.mul(29.0))).add(13.7)).sub(0.5);
  const lineWave = sin(p.y.mul(0.073).add(t.mul(5.3))).add(sin(p.y.mul(0.019).sub(t.mul(2.1))).mul(0.55));
  const headDrift = sin(ctx.frame.mul(0.021)).mul(0.018)
    .add(hash12(vec2(floor(ctx.frame.mul(0.037)), 8.1)).sub(0.5).mul(0.014));
  const headY = float(0.84).add(headDrift).clamp(0.78, 0.93);
  const belowHead = screenUV.y.sub(headY);
  const afterHead = step(0.0, belowHead);
  const headAttack = belowHead.div(0.012).clamp(0.0, 1.0);
  const headFalloff = exp(belowHead.mul(-17.0)).mul(afterHead);
  const headTail = float(1.0).sub(belowHead.div(0.22).clamp(0.0, 1.0)).mul(afterHead);
  const headXFeather = smoothBand(screenUV.x, 0.012, 0.055)
    .mul(float(1.0).sub(smoothBand(screenUV.x, 0.945, 0.988)));
  const headMask = headAttack.mul(headFalloff.mul(0.8).add(headTail.mul(0.2)))
    .mul(headXFeather)
    .mul(headSwitch)
    .clamp(0.0, 1.0);
  const headNoise = hash12(vec2(fieldLine.mul(0.73), floor(t.mul(47.0))).add(91.3)).sub(0.5);
  const headLineBreak = step(0.58, hash12(vec2(fieldLine.mul(1.91), floor(ctx.frame.mul(0.19))).add(117.0)));
  const headPhaseShift = headNoise.mul(24.0).add(headLineBreak.mul(headNoise.mul(20.0)));
  const edgeStep = hash12(vec2(fieldLine.mul(0.17), floor(ctx.frame.mul(0.09))).add(41.0)).sub(0.5);
  const edgeWave = sin(fieldLine.mul(0.37).add(t.mul(1.7))).mul(edgeStep).mul(edgeWaveAmount.mul(1.65));
  const xJitterPx = lineWave.mul(tracking.mul(0.85))
    .add(rowNoise.mul(tracking.mul(2.0)))
    .add(edgeWave)
    .add(headPhaseShift.mul(headMask.mul(tracking)))
    .add(headNoise.mul(headMask.mul(tracking.mul(6.0))));
  const uv = screenUV.add(ctx.texel.mul(vec2(xJitterPx, 0.0)));

  const base = rgbToYCbCr(texture(tex, uv).rgb);
  const left = analogSampleYcc(tex, ctx, uv, vec2(-1.0, 0.0));
  const right = analogSampleYcc(tex, ctx, uv, vec2(1.0, 0.0));
  const farLeft = analogSampleYcc(tex, ctx, uv, vec2(-3.0, 0.0));
  const farRight = analogSampleYcc(tex, ctx, uv, vec2(3.0, 0.0));
  const veryLeft = analogSampleYcc(tex, ctx, uv, vec2(-7.0, 0.0));
  const veryRight = analogSampleYcc(tex, ctx, uv, vec2(7.0, 0.0));
  const above = analogSampleYcc(tex, ctx, uv, vec2(0.0, -2.0));
  const below = analogSampleYcc(tex, ctx, uv, vec2(0.0, 2.0));

  const chromaDelayPx = float(1.6).add(chromaBleed.mul(3.2));
  const chromaDelay = analogSampleYcc(tex, ctx, uv, vec2(chromaDelayPx, 0.0));
  const wideChromaDelay = analogSampleYcc(tex, ctx, uv, vec2(chromaDelayPx.add(chromaBleed.mul(5.0)), 0.0));
  const chromaHoriz = left.yz.add(base.yz.mul(2.0)).add(right.yz).add(chromaDelay.yz.mul(2.0)).add(wideChromaDelay.yz).div(7.0);
  const chromaVert = above.yz.add(base.yz.mul(2.0)).add(below.yz).div(4.0);
  const chromaBlur = mix(chromaHoriz, chromaVert, chromaBleed.mul(0.28).clamp(0.0, 0.55));

  const edge = right.x.sub(left.x);
  const lumaBlur = veryLeft.x.add(farLeft.x.mul(2.0)).add(left.x.mul(3.0)).add(base.x.mul(4.0))
    .add(right.x.mul(3.0)).add(farRight.x.mul(2.0)).add(veryRight.x).div(16.0);
  const lumaHigh = base.x.sub(lumaBlur);
  const preemphasis = lumaHigh.mul(ringing.mul(0.95));
  const ring2 = farLeft.x.sub(left.x).add(right.x.sub(farRight.x)).mul(0.16)
    .add(veryLeft.x.sub(farLeft.x).add(farRight.x.sub(veryRight.x)).mul(0.08))
    .add(edge.mul(0.23));
  const freqNoise = hash12(vec2(floor(p.x.mul(0.055)), fieldLine.add(floor(ctx.frame.mul(0.21))))).sub(0.5);
  let y = mix(base.x, lumaBlur.add(preemphasis), strength.mul(0.42).clamp(0.0, 0.74));
  y = y.add(ring2.mul(ringing.mul(1.35).add(freqNoise.mul(tapeNoise.mul(0.52)))));

  const sat = lengthApprox(base.yz);
  const phaseBucket = mod(floor(p.y.div(2.0)).add(ctx.frame.mul(0.25)), 4.0);
  const phaseOffset = phaseBucket.mul(1.5707963).add(ctx.P.analogTracking.mul(0.35));
  const carrier = analogCarrier(p, phaseOffset);
  const carrierQ = analogCarrier(p.add(vec2(1.0, 0.0)), phaseOffset);
  const crawl = carrier.mul(sat).mul(chromaBleed.mul(0.32));
  y = y.add(crawl.mul(0.55));

  let chroma = mix(base.yz, chromaBlur, chromaBleed.mul(0.58).clamp(0.0, 0.9));
  const phaseNoise = rowNoise.mul(chromaBleed.mul(0.22)).add(headNoise.mul(headMask.mul(0.65)));
  const phaseSin = sin(phaseNoise);
  const phaseCos = cos(phaseNoise);
  chroma = vec2(
    chroma.x.mul(phaseCos).sub(chroma.y.mul(phaseSin)),
    chroma.x.mul(phaseSin).add(chroma.y.mul(phaseCos)),
  );
  const chromaNoise = vec2(
    gaussTemporal(p, ctx.frame, 121.0),
    gaussTemporal(p, ctx.frame, 177.0),
  ).mul(tapeNoise.mul(1.35));
  chroma = chroma.add(chromaNoise).add(vec2(carrier.mul(crawl).mul(0.34), carrierQ.mul(crawl).mul(-0.22)));

  const dropoutFrame = floor(ctx.frame.mul(0.13));
  const dropoutLineSeed = hash13(vec3(fieldLine.mul(2.37), dropoutFrame, 73.0));
  const dropoutSegmentWidth = float(88.0).add(hash12(vec2(fieldLine, dropoutFrame).add(19.0)).mul(96.0));
  const dropoutSegment = floor(p.x.div(dropoutSegmentWidth));
  const dropoutSegmentSeed = hash13(vec3(
    dropoutSegment.mul(5.37).add(dropoutLineSeed.mul(17.0)),
    fieldLine.mul(3.11),
    dropoutFrame.add(73.0),
  ));
  const dropoutLineGate = step(float(1.0).sub(dropoutAmount.mul(0.075).clamp(0.0, 0.26)), dropoutLineSeed);
  const dropoutSegmentGate = step(float(1.0).sub(dropoutAmount.mul(0.32).clamp(0.0, 0.92)), dropoutSegmentSeed);
  const dropoutSegmentPos = fract(p.x.div(dropoutSegmentWidth).add(dropoutLineSeed));
  const dropoutSegmentEnvelope = softBand(dropoutSegmentPos, 0.04, 0.18)
    .mul(float(1.0).sub(softBand(dropoutSegmentPos, 0.74, 0.96)));
  const dropoutBand = dropoutLineGate.mul(dropoutSegmentGate).mul(dropoutSegmentEnvelope);
  const bandDamage = dropoutAmount.mul(bandMask);
  const chromaLoss = dropoutBand.mul(bandDamage.mul(0.85)).clamp(0.0, 1.0);
  chroma = mix(chroma, chroma.mul(0.08), chromaLoss);
  const dropoutStatic = hash13(vec3(p.x.mul(0.91), p.y.mul(1.73), ctx.frame.add(12.0))).sub(0.5);
  y = y.add(dropoutBand.mul(dropoutStatic).mul(tapeNoise.mul(bandDamage).mul(34.0)));

  const scan = sin(p.y.mul(3.1415927)).mul(0.5).add(0.5);
  const interlace = step(0.5, mod(p.y.add(ctx.frame), 2.0));
  y = y.mul(float(1.0).sub(scan.mul(scanlines.mul(0.055))).sub(interlace.mul(scanlines.mul(0.025))));
  y = y.add(gaussTemporal(p, ctx.frame, 211.0).mul(tapeNoise.mul(1.6)));

  const headLine = float(1.0).sub(abs(belowHead).div(0.018).clamp(0.0, 1.0))
    .mul(headXFeather)
    .mul(headSwitch.mul(0.34));
  const headStatic = gaussTemporal(p, ctx.frame, 319.0).mul(0.62)
    .add(hash13(vec3(p.x.mul(0.31), p.y.mul(1.7), ctx.frame.mul(0.21))).sub(0.5));
  const headLumaNoise = headStatic.mul(tapeNoise.mul(38.0)).mul(headMask);
  const headLineNoise = headLineBreak.mul(hash12(vec2(p.x.mul(0.043), fieldLine.add(ctx.frame))).sub(0.5)).mul(headMask);
  const headTear = headMask.clamp(0.0, 1.0);
  y = mix(y, y.mul(0.72).add(headLumaNoise).add(headLineNoise.mul(48.0)).add(headLine.mul(42.0)), headTear.mul(0.72));
  chroma = mix(chroma, chroma.mul(0.24).add(vec2(headStatic.mul(18.0), headNoise.mul(12.0))), headTear.mul(0.82));

  const rightBorder = step(0.983, screenUV.x).mul(strength.mul(0.72).clamp(0.0, 1.0));
  y = mix(y, y.mul(0.15), rightBorder);
  chroma = mix(chroma, chroma.mul(0.1), rightBorder);

  const oversat = float(1.0).add(strength.mul(0.18));
  const analog = yCbCrToRgb(vec3(y, chroma.mul(oversat))).clamp(0.0, LEVELS);
  const source = texture(tex, screenUV).rgb;
  return mix(source, analog, active.clamp(0.0, 1.0)).clamp(0.0, LEVELS);
}

function lengthApprox(v) {
  return abs(v.x).add(abs(v.y)).mul(0.5);
}

function dctBasis(pos, freq) {
  return cos(pos.mul(2.0).add(1.0).mul(freq).mul(0.19634954084936207));
}

function dctNorm1D(freq) {
  return mix(float(Math.SQRT1_2), float(1.0), step(0.5, freq)).mul(0.5);
}

// Firefox's WGSL compiler (Naga) rejects compile-time-constant *sub*expressions
// inside runtime shader code ("Abstract types may only appear in constant
// expressions"); Chrome's Tint just folds them. dctBasis/dctNorm1D build exactly
// such subtrees when one argument is a JS loop constant: `(2*pos+1)` when the
// sample position is constant (forward DCT), and the whole `step()/mix()` chain
// when the frequency is constant (inverse DCT). These variants collapse the
// constant part to a single literal up front — identical f32 arithmetic, but no
// all-abstract subtree for Naga to choke on.

// forward transforms: the sample position is the compile-time constant, so
// (2*pos+1) — an exact integer — becomes one literal instead of `x*2.0+1.0`.
function dctBasisConstPos(posConst, freq) {
  return cos(float(2 * posConst + 1).mul(freq).mul(0.19634954084936207));
}

// inverse transforms: the frequency is the compile-time constant, so the entire
// normalization is `mix(SQRT1_2, 1, step(0.5, freq)) * 0.5` evaluated in JS.
function dctNorm1DConst(freq) {
  return (freq < 0.5 ? Math.SQRT1_2 : 1.0) * 0.5;
}

function samplePixel(tex, ctx, p) {
  const samplePos = min(p.add(0.5), ctx.resolution.sub(0.5));
  return texture(tex, samplePos.div(ctx.resolution)).rgb;
}

function jpegChroma420(tex, ctx, p) {
  const chromaBase = floor(p.div(2.0)).mul(2.0);
  const uv00 = min(chromaBase.add(vec2(0.5, 0.5)), ctx.resolution.sub(0.5)).div(ctx.resolution);
  const uv10 = min(chromaBase.add(vec2(1.5, 0.5)), ctx.resolution.sub(0.5)).div(ctx.resolution);
  const uv01 = min(chromaBase.add(vec2(0.5, 1.5)), ctx.resolution.sub(0.5)).div(ctx.resolution);
  const uv11 = min(chromaBase.add(vec2(1.5, 1.5)), ctx.resolution.sub(0.5)).div(ctx.resolution);
  const c00 = rgbToYCbCr(texture(tex, uv00).rgb);
  const c10 = rgbToYCbCr(texture(tex, uv10).rgb);
  const c01 = rgbToYCbCr(texture(tex, uv01).rgb);
  const c11 = rgbToYCbCr(texture(tex, uv11).rgb);
  return c00.add(c10).add(c01).add(c11).mul(0.25).yz;
}

function jpegInput(tex, ctx, p) {
  const c = rgbToYCbCr(samplePixel(tex, ctx, p));
  const chroma = mix(c.yz, jpegChroma420(tex, ctx, p), ctx.P.jpegChroma420.mul(ctx.P.jpegStrength).clamp(0.0, 1.0));
  return vec3(c.x.sub(128.0), chroma.x, chroma.y);
}


function sampleDct(tex, ctx, p) {
  const samplePos = min(p.add(0.5), ctx.resolution.sub(0.5));
  return texture(tex, samplePos.div(ctx.resolution)).rgb;
}

function jpegAmount(ctx) {
  return ctx.P.jpegStrength.clamp(0.0, 1.0);
}

function jpegHighlightAmount(original, ctx) {
  const luma = dot(original, LUMA).div(LEVELS).clamp(0.0, 1.0);
  const midtone = luma.sub(0.08).div(0.42).clamp(0.0, 1.0).mul(ctx.P.jpegMidtone);
  const highlight = luma.sub(0.24).div(0.58).clamp(0.0, 1.0).pow(0.85).mul(ctx.P.jpegHighlight);
  const mask = max(midtone, highlight).clamp(0.0, 1.0);
  return jpegAmount(ctx).pow(0.55).mul(mask);
}

const JPEG_LUMA_Q = [
  [16, 11, 10, 16, 24, 40, 51, 61],
  [12, 12, 14, 19, 26, 58, 60, 55],
  [14, 13, 16, 24, 40, 57, 69, 56],
  [14, 17, 22, 29, 51, 87, 80, 62],
  [18, 22, 37, 56, 68, 109, 103, 77],
  [24, 35, 55, 64, 81, 104, 113, 92],
  [49, 64, 78, 87, 103, 121, 120, 101],
  [72, 92, 95, 98, 112, 100, 103, 99],
];

const JPEG_CHROMA_Q = [
  [17, 18, 24, 47, 99, 99, 99, 99],
  [18, 21, 26, 66, 99, 99, 99, 99],
  [24, 26, 56, 99, 99, 99, 99, 99],
  [47, 66, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
];

// 8x8 quantization tables as a LUT texture (r = luma Q, g = chroma Q). The
// previous in-shader table lookup expanded to a 64-term masked sum per pixel
// per table; a nearest-filtered fetch returns the identical value for free.
let jpegQTexture = null;
function getJpegQTexture() {
  if (!jpegQTexture) {
    // Half-float, not fp32: every quant value is an integer <= 255, which fp16
    // stores exactly, so the sampled numbers are identical. The reason to avoid
    // fp32 here is portability — rgba32float is only sampleable where the WebGPU
    // `float32-filterable` feature exists (Chrome/Dawn). Firefox doesn't expose
    // it, so an fp32 sampled texture made the whole JPEG stage fail (black frame)
    // there. rgba16float is filterable on every backend.
    const half = THREE.DataUtils.toHalfFloat;
    const data = new Uint16Array(64 * 4);
    for (let y = 0; y < 8; y += 1) {
      for (let x = 0; x < 8; x += 1) {
        const i = (y * 8 + x) * 4;
        data[i] = half(JPEG_LUMA_Q[y][x]);
        data[i + 1] = half(JPEG_CHROMA_Q[y][x]);
        data[i + 3] = half(1);
      }
    }
    jpegQTexture = new THREE.DataTexture(data, 8, 8, THREE.RGBAFormat, THREE.HalfFloatType);
    jpegQTexture.minFilter = THREE.NearestFilter;
    jpegQTexture.magFilter = THREE.NearestFilter;
    jpegQTexture.needsUpdate = true;
  }
  return jpegQTexture;
}

function jpegQuantStep(ctx, u, v) {
  const quality = ctx.P.jpegQuality.clamp(1.0, 100.0);
  const lowScale = float(5000.0).div(quality);
  const highScale = float(200.0).sub(quality.mul(2.0));
  const qualityScale = mix(lowScale, highScale, step(50.0, quality));
  const strength = max(ctx.P.jpegStrength, float(0.01));
  const scale = qualityScale.mul(0.01).mul(strength);
  const q = texture(getJpegQTexture(), vec2(u.add(0.5).div(8.0), v.add(0.5).div(8.0)));
  const yQ = q.r.mul(scale).clamp(1.0, 255.0);
  const cQ = q.g.mul(scale).clamp(1.0, 255.0);
  return vec3(yQ, cQ, cQ);
}

function stJpegDctRows(tex, ctx) {
  const p = floor(screenUV.mul(ctx.resolution));
  const block = floor(p.div(8.0)).mul(8.0);
  const local = mod(p, 8.0);
  const u = local.x;
  let sum = vec3(0.0);
  for (let x = 0; x < 8; x += 1) {
    sum = sum.add(jpegInput(tex, ctx, block.add(vec2(x, local.y))).mul(dctBasisConstPos(x, u)));
  }
  return sum.mul(dctNorm1D(u));
}

function stJpegDctColsQuant(tex, ctx) {
  const p = floor(screenUV.mul(ctx.resolution));
  const block = floor(p.div(8.0)).mul(8.0);
  const local = mod(p, 8.0);
  const u = local.x;
  const v = local.y;
  let sum = vec3(0.0);
  for (let y = 0; y < 8; y += 1) {
    sum = sum.add(sampleDct(tex, ctx, block.add(vec2(u, y))).mul(dctBasisConstPos(y, v)));
  }
  const coeff = sum.mul(dctNorm1D(v));
  const q = jpegQuantStep(ctx, u, v);
  return quantize(coeff, q);
}

// Inverse DCT, separable like the forward transform: columns then rows,
// 8+8 taps instead of the 64-tap full-block form. Mathematically identical.
function stJpegIdctCols(tex, ctx) {
  const p = floor(screenUV.mul(ctx.resolution));
  const block = floor(p.div(8.0)).mul(8.0);
  const local = mod(p, 8.0);
  let sum = vec3(0.0);
  for (let v = 0; v < 8; v += 1) {
    const by = dctBasis(local.y, float(v)).mul(dctNorm1DConst(v));
    sum = sum.add(sampleDct(tex, ctx, block.add(vec2(local.x, v))).mul(by));
  }
  return sum;
}

function stJpegIdctRows(tex, ctx, originalTex) {
  const p = floor(screenUV.mul(ctx.resolution));
  const block = floor(p.div(8.0)).mul(8.0);
  const local = mod(p, 8.0);
  let sum = vec3(0.0);
  for (let u = 0; u < 8; u += 1) {
    const bx = dctBasis(local.x, float(u)).mul(dctNorm1DConst(u));
    sum = sum.add(sampleDct(tex, ctx, block.add(vec2(u, local.y))).mul(bx));
  }
  const original = samplePixel(originalTex, ctx, p);
  const decoded = yCbCrToRgb(sum.add(vec3(128.0, 0.0, 0.0))).clamp(0.0, LEVELS);
  return mix(original, decoded, jpegHighlightAmount(original, ctx)).clamp(0.0, LEVELS);
}

// final: bring 0..255 back to 0..1 for display
function stOutput(tex, ctx) {
  return texture(tex, screenUV).rgb.div(LEVELS);
}

function powershotOutputColorGrading(color, ctx) {
  return color.mul(ctx.outputBrightness.add(1.0)).sub(0.5).mul(ctx.outputContrast.add(1.0)).add(0.5);
}

function powershotOutputAlpha(sourceSample, effectColor) {
  const sourceAlpha = sourceSample.a.clamp(0.0, 1.0);
  const effectDelta = dot(abs(effectColor.sub(sourceSample.rgb)), LUMA);
  const effectAlpha = effectDelta.sub(0.002).mul(6.0).clamp(0.0, 0.65);
  return sourceAlpha.add(effectAlpha.mul(sourceAlpha.oneMinus())).clamp(0.0, 1.0);
}

// ---------------------------------------------------------------------------
// stage registry - ordered as a camera-inspired ISP signal path
// ---------------------------------------------------------------------------

export const STAGE_DEFS = [
  { id: "barrel", label: "Barrel distortion", make: stBarrel },
  { id: "ca", label: "Chromatic aberration", make: stChromatic },
  { id: "lens", label: "Lens PSF softness", make: stLensPsf },
  { id: "ccdbloom", label: "CCD bloom / vertical smear" }, // multi-pass, built by id in _rebuild
  { id: "mosaic", label: "Bayer mosaic", make: stMosaic },
  { id: "dpc", label: "Dead pixel correction", make: stDeadPixelCorrection },
  { id: "blacklevel", label: "Black level offset", make: stBlackLevel },
  { id: "noise", label: "CCD sensor noise", make: stBayerNoise },
  { id: "aaf", label: "Anti-alias filter (OLPF)", make: stAAF },
  { id: "bnr", label: "Bayer noise reduction" }, // multi-pass, built by id in _rebuild
  { id: "wb", label: "White balance (Bayer)", make: stWhiteBalance },
  { id: "demosaic", label: "Demosaic", make: stDemosaic },
  { id: "chromanr", label: "Chroma noise reduction", make: stChromaDenoise },
  { id: "ccm", label: "Color correction matrix", make: stCCM },
  { id: "tone", label: "Tone curve", make: stTone },
  { id: "saturation", label: "Saturation boost", make: stSaturation },
  { id: "vignette", label: "Vignette", make: stVignette },
  { id: "edge", label: "Edge enhancement", make: stEdgeEnhance },
  { id: "jpeg", label: "JPEG DCT compression" }, // multi-pass, built by id in _rebuild
];

export const ANALOG_STAGE_DEFS = [
  { id: "analog", label: "Analog VHS / NTSC", make: stAnalogVhs },
];

const DIGITAL_POINT_STAGE_IDS = new Set([
  "mosaic", "blacklevel", "noise", "wb",
  "ccm", "tone", "saturation", "vignette",
]);

// Bayer-domain stages: output is replicated grey and every in-section
// consumer samples .r only, so these can live in single-channel targets.
const GREY_STAGE_IDS = new Set([
  "mosaic", "dpc", "blacklevel", "noise", "aaf", "bnr", "wb",
]);

// ---------------------------------------------------------------------------
// uniforms built per preset (so we can hot-swap without rebuilding nodes)
// ---------------------------------------------------------------------------

export function makeUniforms() {
  return {
    resolution: uniform(new THREE.Vector2(1, 1)),
    texel: uniform(new THREE.Vector2(1, 1)),
    bloomTexel: uniform(new THREE.Vector2(1, 1)),
    frame: uniform(0),
    power: uniform(1),
    // global noise trim: raw Bayer noise is injected, then reduced in later stages
    // down through Bayer/RGB denoising, so we scale the injection to roughly
    // what survives the realtime subset.
    noiseScale: uniform(1.06),
    outputBrightness: uniform(0),
    outputContrast: uniform(0),
    P: {
      barrel: uniform(0), ca: uniform(0),
      lensSoftness: uniform(0.25),
      ccdBloom: uniform(0), ccdBloomThreshold: uniform(200),
      wbR: uniform(1), wbG: uniform(1), wbB: uniform(1),
      blR: uniform(0), blGr: uniform(0), blGb: uniform(0), blB: uniform(0),
      noise: uniform(0), colorNoise: uniform(0), hotRate: uniform(0),
      colFpn: uniform(0), rowFpn: uniform(0), prnu: uniform(0), dsnu: uniform(0),
      dpcThreshold: uniform(30), aaf: uniform(0), bayerNR: uniform(0), bnrSpatial: uniform(1.5), bnrRange: uniform(25),
      demosaicSharp: uniform(0.55),
      chromaNR: uniform(1.0),
      ccm0: uniform(new THREE.Vector3(1, 0, 0)),
      ccm1: uniform(new THREE.Vector3(0, 1, 0)),
      ccm2: uniform(new THREE.Vector3(0, 0, 1)),
      hiClip: uniform(255), gamma: uniform(1), shadow: uniform(0), sat: uniform(1),
      vignette: uniform(0), eeGain: uniform(0), eeThresh: uniform(0),
      jpegQuality: uniform(60), jpegStrength: uniform(0.2), jpegChroma420: uniform(0.75),
      jpegMidtone: uniform(0.45), jpegHighlight: uniform(1.0),
      analogStrength: uniform(0.65), analogTracking: uniform(0.45),
      analogChromaBleed: uniform(0.75), analogRinging: uniform(0.65), analogTapeNoise: uniform(0.75),
      analogBandMask: uniform(0.35), analogEdgeWave: uniform(0.35), analogDropouts: uniform(0.35), analogScanlines: uniform(0.55),
      analogHeadSwitch: uniform(0.45),
    },
  };
}

export function applyPreset(ctx, preset) {
  const P = ctx.P;
  P.barrel.value = preset.barrel_distortion;
  P.ca.value = preset.chromatic_aberration;
  P.lensSoftness.value = preset.lens_softness ?? 0.25;
  P.ccdBloom.value = preset.ccd_bloom_strength;
  P.ccdBloomThreshold.value = preset.ccd_bloom_threshold;
  P.wbR.value = preset.wb_shift[0];
  P.wbG.value = preset.wb_shift[1];
  P.wbB.value = preset.wb_shift[2];
  P.blR.value = preset.black_level[0];
  P.blGr.value = preset.black_level[1];
  P.blGb.value = preset.black_level[2];
  P.blB.value = preset.black_level[3];
  P.noise.value = preset.noise_intensity;
  // Suppress chroma speckles and hot pixels; they read as colored fireflies in motion.
  P.colorNoise.value = 0;
  P.hotRate.value = 0;
  P.colFpn.value = preset.column_fpn;
  P.rowFpn.value = preset.row_fpn;
  P.prnu.value = preset.prnu;
  P.dsnu.value = preset.dsnu;
  P.dpcThreshold.value = preset.dpc_threshold;
  P.aaf.value = preset.aaf_strength;
  P.bayerNR.value = preset.bnr_strength;
  P.bnrSpatial.value = preset.bnr_spatial_sigma ?? 1.5;
  P.bnrRange.value = preset.bnr_range_sigma;
  P.demosaicSharp.value = preset.demosaic_quality === "malvar" ? 0.85 : 0.55;
  P.ccm0.value.set(...preset.ccm[0]);
  P.ccm1.value.set(...preset.ccm[1]);
  P.ccm2.value.set(...preset.ccm[2]);
  P.hiClip.value = preset.highlight_clip;
  P.gamma.value = preset.gamma;
  P.shadow.value = preset.shadow_crush;
  P.sat.value = preset.saturation_boost;
  P.vignette.value = preset.vignette_strength;
  P.eeGain.value = preset.ee_gain;
  P.eeThresh.value = preset.ee_threshold;
  P.jpegQuality.value = preset.jpeg_quality;
  P.jpegStrength.value = 0.2;
  P.jpegChroma420.value = 0.75;
  P.jpegMidtone.value = 0.45;
  P.jpegHighlight.value = 1.0;
  P.analogStrength.value = preset.analog_vhs_strength ?? 0.65;
  P.analogTracking.value = preset.analog_tracking ?? 0.45;
  P.analogChromaBleed.value = preset.analog_chroma_bleed ?? 0.75;
  P.analogRinging.value = preset.analog_ringing ?? 0.65;
  P.analogTapeNoise.value = preset.analog_tape_noise ?? 0.75;
  P.analogBandMask.value = preset.analog_band_mask ?? 0.35;
  P.analogEdgeWave.value = preset.analog_edge_wave ?? 0.35;
  P.analogDropouts.value = preset.analog_dropouts ?? 0.35;
  P.analogScanlines.value = preset.analog_scanlines ?? 0.55;
  P.analogHeadSwitch.value = preset.analog_head_switch ?? 0.45;
}

// ---------------------------------------------------------------------------
// runner — one persistent material PER stage (built once, reused every frame).
//
// A NodeMaterial compiles its shader once and caches it, so we cannot reuse a
// single material and swap `.colorNode` between passes — every pass would run
// the first-compiled graph. Instead we bake a dedicated material per active
// stage (each sampling a fixed ping-pong target) and only rebuild the chain
// when the source image, working size, or enabled-stage set changes. Per frame
// we just issue the draws; uniforms (e.g. `frame`) update by reference.
// ---------------------------------------------------------------------------

export class Pipeline {
  constructor(renderer) {
    this.renderer = renderer;
    this.ctx = makeUniforms();

    const opts = {
      type: THREE.HalfFloatType,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
      colorSpace: THREE.NoColorSpace,
    };
    this.rtA = new THREE.RenderTarget(1, 1, opts);
    this.rtB = new THREE.RenderTarget(1, 1, { ...opts });
    this.rtC = new THREE.RenderTarget(1, 1, { ...opts });
    // quarter-height pair for the CCD bloom smear (full width)
    this.rtBloomA = new THREE.RenderTarget(1, 1, { ...opts });
    this.rtBloomB = new THREE.RenderTarget(1, 1, { ...opts });
    // Scratch targets for split-pass intermediates (BNR guide, JPEG IDCT
    // columns). These values previously lived in registers inside one big pass;
    // storing them at half precision adds rounding the original never had, so
    // fp32 keeps the split passes numerically faithful. Nearest filtering
    // because every consumer samples exact texel centers.
    //
    // fp32 sampled textures only work where the WebGPU `float32-filterable`
    // feature exists (Chrome/Dawn). Firefox doesn't expose it and would reject
    // them, so fall back to half-float there — the same intermediates the rest
    // of the chain already uses, and a working frame beats a faithful black one.
    const preciseType = renderer.hasFeature("float32-filterable")
      ? THREE.FloatType
      : THREE.HalfFloatType;
    const preciseOpts = {
      ...opts,
      type: preciseType,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
    };
    this.rtD = new THREE.RenderTarget(1, 1, preciseOpts);
    this.rtE = new THREE.RenderTarget(1, 1, { ...preciseOpts });
    // single-channel (r16float) pair for the Bayer-domain section — same .r
    // bits as the RGBA pair at a quarter of the bandwidth.
    const greyOpts = { ...opts, format: THREE.RedFormat };
    this.rtGreyA = new THREE.RenderTarget(1, 1, greyOpts);
    this.rtGreyB = new THREE.RenderTarget(1, 1, { ...greyOpts });

    // fullscreen quad — its material is swapped to the current step's material
    this.quadScene = new THREE.Scene();
    this.quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), null);
    this.mesh.frustumCulled = false;
    this.quadScene.add(this.mesh);

    this.enabled = new Set(STAGE_DEFS.map((s) => s.id));
    this.mode = "digital";
    this.source = null;
    this.size = { w: 0, h: 0 };
    this.steps = [];       // [{ material, target }]
    this.outputMat = null; // draws final 0..255 texture to screen as 0..1
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
    this.rtA.setSize(w, h);
    this.rtB.setSize(w, h);
    this.rtC.setSize(w, h);
    this.rtD.setSize(w, h);
    this.rtE.setSize(w, h);
    this.rtGreyA.setSize(w, h);
    this.rtGreyB.setSize(w, h);
    const bh = Math.max(1, Math.round(h / 4));
    this.rtBloomA.setSize(w, bh);
    this.rtBloomB.setSize(w, bh);
    this.size = { w, h };
    this.ctx.resolution.value.set(w, h);
    this.ctx.texel.value.set(1 / w, 1 / h);
    this.ctx.bloomTexel.value.set(1 / w, 1 / bh);
    this.dirty = true;
  }

  setEnabled(id, on) {
    const hasStage = this.enabled.has(id);
    if (on === hasStage) return;
    if (on) this.enabled.add(id);
    else this.enabled.delete(id);
    this.dirty = true;
  }

  setMode(mode) {
    const next = mode === "analog" ? "analog" : "digital";
    if (next === this.mode) return;
    this.mode = next;
    this.dirty = true;
  }

  setOutputColorGrading({ brightness = 0, contrast = 0 } = {}) {
    this.ctx.outputBrightness.value = Number.isFinite(brightness) ? brightness : 0;
    this.ctx.outputContrast.value = Number.isFinite(contrast) ? contrast : 0;
    if (this.outputMat) this.outputMat.needsUpdate = true;
  }

  _rebuild() {
    for (const s of this.steps) s.material.dispose();
    if (this.outputMat) this.outputMat.dispose();
    this.steps = [];
    this.dirty = false;
    if (!this.source) { this.outputMat = null; return; }

    const active = this.mode === "analog"
      ? ANALOG_STAGE_DEFS
      : STAGE_DEFS.filter((s) => this.enabled.has(s.id));
    let startIndex = 0;

    // Bayer-domain stages write replicated grey and their in-section
    // consumers read .r only, so they can ping-pong through the
    // single-channel targets — same bits, a quarter of the bandwidth. Only
    // allowed when the enabled run of them feeds straight into demosaic;
    // any other consumer (chroma NR, CCM stack, edge, JPEG, the output
    // blit) reads .gb too, so those combos stay on the RGBA targets.
    let greyEligible = false;
    if (this.mode === "digital") {
      const first = active.findIndex((s) => GREY_STAGE_IDS.has(s.id));
      if (first >= 0) {
        let last = first;
        while (last + 1 < active.length && GREY_STAGE_IDS.has(active[last + 1].id)) last += 1;
        greyEligible = active[last + 1]?.id === "demosaic";
      }
    }

    let read = this.rtA;
    const targetFor = (domain) => (domain === "grey"
      ? (read === this.rtGreyA ? this.rtGreyB : this.rtGreyA)
      : (read === this.rtA ? this.rtB : this.rtA));
    const push = (material, domain = "rgb") => {
      const target = targetFor(domain);
      this.steps.push({ material, target });
      read = target;
    };
    const stackDomain = (ids) => (greyEligible && ids.every((id) => GREY_STAGE_IDS.has(id)) ? "grey" : "rgb");

    // Mandatory input/downsample is point-only, so fold it into the first
    // digital point-stage run when no earlier resampling stage is active.
    if (this.mode === "digital" && active.length > 0 && DIGITAL_POINT_STAGE_IDS.has(active[0].id)) {
      const ids = [];
      while (startIndex < active.length && DIGITAL_POINT_STAGE_IDS.has(active[startIndex].id)) {
        ids.push(active[startIndex].id);
        startIndex += 1;
      }
      read = stackDomain(ids) === "grey" ? this.rtGreyA : this.rtA;
      this.steps.push({
        material: this._mat(stInputDigitalPointStack(this.source, this.ctx, new Set(ids))),
        target: read,
      });
    } else {
      // mandatory input + downsample pass: 0..1 source -> 0..255 in rtA.
      this.steps.push({ material: this._mat(stInput(this.source, this.ctx)), target: this.rtA });
    }

    // ping-pong the active stages with baked input textures
    for (let i = startIndex; i < active.length; i += 1) {
      const stage = active[i];
      if (DIGITAL_POINT_STAGE_IDS.has(stage.id)) {
        const ids = [];
        let j = i;
        while (j < active.length && DIGITAL_POINT_STAGE_IDS.has(active[j].id)) {
          ids.push(active[j].id);
          j += 1;
        }
        if (ids.length > 1) {
          const idSet = new Set(ids);
          push(this._mat(stDigitalPointStack(read.texture, this.ctx, idSet)), stackDomain(ids));
          i = j - 1;
          continue;
        }
      }
      if (stage.id === "ccdbloom") {
        // extract (quarter height) -> smear (quarter height) -> composite (full)
        this.steps.push({ material: this._mat(stCcdBloomExtract(read.texture, this.ctx)), target: this.rtBloomA });
        this.steps.push({ material: this._mat(stCcdBloomSmear(this.rtBloomA.texture, this.ctx)), target: this.rtBloomB });
        push(this._mat(stCcdBloomComposite(read.texture, this.ctx, this.rtBloomB.texture)));
        continue;
      }
      if (stage.id === "bnr") {
        // guide prepass (fp32) -> joint bilateral reading value+guide per tap
        this.steps.push({ material: this._mat(stBnrGuide(read.texture, this.ctx)), target: this.rtD });
        push(this._mat(stBayerDenoise(this.rtD.texture, this.ctx)), greyEligible ? "grey" : "rgb");
        continue;
      }
      if (stage.id === "jpeg") {
        // original copy -> row DCT -> col DCT + quant -> col IDCT (fp32)
        // -> row IDCT + composite
        this.steps.push({ material: this._mat(stCopy(read.texture, this.ctx)), target: this.rtC });
        push(this._mat(stJpegDctRows(read.texture, this.ctx)));
        push(this._mat(stJpegDctColsQuant(read.texture, this.ctx)));
        this.steps.push({ material: this._mat(stJpegIdctCols(read.texture, this.ctx)), target: this.rtE });
        push(this._mat(stJpegIdctRows(this.rtE.texture, this.ctx, this.rtC.texture)));
        continue;
      }
      const domain = greyEligible && GREY_STAGE_IDS.has(stage.id) ? "grey" : "rgb";
      push(this._mat(stage.make(read.texture, this.ctx)), domain);
    }

    const finalTex = this.steps[this.steps.length - 1].target.texture;
    const sourceSample = texture(this.source, screenUV);
    const effectColor = mix(sourceSample.rgb, stOutput(finalTex, this.ctx), this.ctx.power);
    const finalColor = powershotOutputColorGrading(effectColor, this.ctx);
    this.outputMat = this._mat(vec4(finalColor, powershotOutputAlpha(sourceSample, effectColor)));
    this.outputMat.toneMapped = false;
    this.outputMat.transparent = true;
    this.outputMat.blending = THREE.NoBlending;
  }

  renderTexture(inputTexture, frame = 0, { outputTarget = null } = {}) {
    if (!inputTexture) return false;
    this.setSource(inputTexture);
    if (this.dirty) this._rebuild();
    if (!this.source || !this.outputMat) return false;
    this.ctx.frame.value = frame;
    const r = this.renderer;
    const previousTarget = r.getRenderTarget?.() ?? null;

    try {
      for (const step of this.steps) {
        this.mesh.material = step.material;
        r.setRenderTarget(step.target);
        r.render(this.quadScene, this.quadCam);
      }

      this.mesh.material = this.outputMat;
      r.setRenderTarget(outputTarget);
      r.render(this.quadScene, this.quadCam);
      return true;
    } finally {
      r.setRenderTarget(previousTarget);
    }
  }

  // run the chain and present to screen
  async render(frame) {
    this.renderTexture(this.source, frame);
  }

  dispose() {
    for (const s of this.steps) s.material.dispose();
    this.steps = [];
    if (this.outputMat) this.outputMat.dispose();
    this.outputMat = null;
    this.rtA.dispose();
    this.rtB.dispose();
    this.rtC.dispose();
    this.rtD.dispose();
    this.rtE.dispose();
    this.rtGreyA.dispose();
    this.rtGreyB.dispose();
    this.rtBloomA.dispose();
    this.rtBloomB.dispose();
    this.mesh.geometry.dispose();
  }
}
