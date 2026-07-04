// spectral_traverse.js — shared TSL traversal + shading helpers for the
// spectral path tracer AND the SPEEDBALL GI probe kernel (docs/GI_SPEEDBALL_design.md §3.4).
//
// EXTRACTED VERBATIM from spectral_kernel.js buildKernels() so both kernels
// trace the SAME resident stackless BVH with byte-identical logic — there is no
// second acceleration structure. The path tracer keeps the local closures only
// in that they are now produced by buildTraversal() and destructured back; the
// emitted graph is unchanged (gate #2: pixel-identical A/B of the path tracer).
//
// buildTraversal() returns graph EMITTERS (not values): each call site builds a
// fresh subgraph, exactly as the inline closures did. It closes over the storage
// nodes + uniforms passed in, so the probe kernel binds its own storage()/U over
// the same buffers and imports these instead of copying them.

import * as TSL from 'three/tsl';

const {
    Loop, If, Break, texture, texture3D,
    int, uint, vec2, vec3,
    uintBitsToFloat, equirectUV,
    select, max, min, abs, sqrt, sin, cos, exp, pow, smoothstep,
    dot, cross, normalize, reflect, mix, clamp, float,
} = TSL;

export const PI = 3.141592653589793;
export const LAMBDA_MIN = 380.0;
export const LAMBDA_MAX = 720.0;
export const LAMBDA_RANGE = LAMBDA_MAX - LAMBDA_MIN;
// NV-mode λ domain: photocathode-weighted sampling over the GaAs response
// window. The JH coefficient normalization stays anchored to the VISIBLE
// range the LUT was fitted over (see jhEval); only the sampling domain moves.
export const NV_LAMBDA_MIN = 550.0;
export const NV_LAMBDA_MAX = 900.0;
export const T_MAX = 1e30;
export const RAY_EPS = 1e-3;

// ── Gen-3 GaAs photocathode response S(λ) ──────────────────────────
// Analytic fit (measured curves are out of scope): a window rising from
// ~550 nm, broad peak around 780–830 nm, steep cutoff to ~900 nm. The CPU fit
// and the TSL emitter below MUST stay in exact sync — the CPU side builds the
// λ inverse-CDF and the ∫S(λ)dλ normalization the kernel divides by.
function smoothstepJS(e0, e1, x) {
    const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
    return t * t * (3 - 2 * t);
}

export function photocathodeResponseJS(l) {
    const rise = smoothstepJS(540, 620, l);
    const cut = 1 - smoothstepJS(860, 900, l);
    const peakShape = 0.75 + 0.25 * Math.exp(-(((l - 805) / 70) ** 2));
    return rise * cut * peakShape;
}

// TSL emitter — same fit as photocathodeResponseJS.
export function photocathodeS(lambda) {
    const rise = smoothstep(float(540.0), float(620.0), lambda);
    const cut = float(1.0).sub(smoothstep(float(860.0), float(900.0), lambda));
    const dn = lambda.sub(805.0).div(70.0);
    const peakShape = float(0.75).add(exp(dn.mul(dn).mul(-1.0)).mul(0.25));
    return rise.mul(cut).mul(peakShape);
}

// Precompute the NV λ sampler: a small inverse-CDF LUT mapping u∈[0,1) → λ
// (64 entries, linear interp — plenty for a curve this smooth), plus the
// response integral ∫S(λ)dλ over the domain. pdf(λ) = S(λ)/integral.
export function buildPhotocathodeSampler(entries = 64, lambdaMin = NV_LAMBDA_MIN, lambdaMax = NV_LAMBDA_MAX) {
    const STEPS = 2048;
    const dl = (lambdaMax - lambdaMin) / STEPS;
    const cdf = new Float64Array(STEPS + 1);
    for (let i = 0; i < STEPS; i++) {
        const l = lambdaMin + (i + 0.5) * dl;
        cdf[i + 1] = cdf[i] + photocathodeResponseJS(l) * dl;
    }
    const integral = cdf[STEPS];
    const lut = new Float32Array(entries);
    let cursor = 0;
    for (let e = 0; e < entries; e++) {
        const target = (e / (entries - 1)) * integral;
        while (cursor < STEPS && cdf[cursor + 1] < target) cursor++;
        const c0 = cdf[cursor];
        const c1 = cdf[Math.min(cursor + 1, STEPS)];
        const f = c1 > c0 ? (target - c0) / (c1 - c0) : 0;
        lut[e] = lambdaMin + (cursor + f) * dl;
    }
    lut[entries - 1] = lambdaMax;
    return { lut, integral };
}

// LPS sodium: one 589 nm line, σ ≈ 4 nm. The gaussian's amplitude is chosen so
// the line conserves the light's Y (what jhEmission's smooth spectrum would
// integrate to): A = Y · CIE_Y_INTEGRAL / (√(2π)·σ·ȳ(589)). Precomputed:
// 106.936 / (10.027 · 0.7746) ≈ 13.77.
const SODIUM_Y_SCALE = 13.77;
const SODIUM_SIGMA = 4.0;
const IR_ILLUM_SIGMA = 15.0;

// Linear RGB → scalar reflectance at λ via a smooth 3-bin spectrum
// (blue→green→red). Cheap and good enough for averaged color.
export function rgbToSpectral(rgbNode, lambda) {
    const t = clamp(lambda.sub(LAMBDA_MIN).div(LAMBDA_RANGE), 0.0, 1.0);
    const lo = mix(rgbNode.z, rgbNode.y, clamp(t.mul(2.0), 0.0, 1.0));
    const hi = mix(rgbNode.y, rgbNode.x, clamp(t.sub(0.5).mul(2.0), 0.0, 1.0));
    return select(t.lessThan(0.5), lo, hi);
}

// Build the shared graph emitters bound to a given set of storage nodes +
// uniforms. `storages` = { bvhNodes, triIndex, vertexData, triMaterial,
// materials } (TSL storage nodes). `U` must expose nodeCount (BLAS pool),
// tlasNodeCount, instBase, tlasBase (uint uniforms — instance records + TLAS
// nodes live in the tail of the materials buffer, see spectral_scene.js),
// plus envRotation, envIntensity. `env` (equirect texture | null), `lut`
// (3 Data3D | null) and `maps` (PBR DataArrayTextures | {}) mirror
// buildKernels' inputs. Traversal is TWO-LEVEL: world-ray TLAS walk over
// instance AABBs → local-ray BLAS walk per instance; hit shading transforms
// back via instLocalRay / instNormalToWorld (vertex data is LOCAL space).
export function buildTraversal({ storages, U, env = null, lut = null, lutRes = 0, maps = {} }) {
    const { bvhNodes, triIndex, vertexData, triMaterial, materials } = storages;
    const MSTRIDE = uint(28);

    // ── helpers (graph emitters) ───────────────────────────────────
    const VSTRIDE = uint(8);
    const fetchVert = (vid) => {
        const b = vid.mul(VSTRIDE);
        return vec3(vertexData.element(b), vertexData.element(b.add(uint(1))), vertexData.element(b.add(uint(2))));
    };
    const fetchNorm = (vid) => {
        const b = vid.mul(VSTRIDE).add(uint(3));
        return vec3(vertexData.element(b), vertexData.element(b.add(uint(1))), vertexData.element(b.add(uint(2))));
    };
    const fetchUV = (vid) => {
        const b = vid.mul(VSTRIDE).add(uint(6));
        return vec2(vertexData.element(b), vertexData.element(b.add(uint(1))));
    };
    const triVert = (triId, k) => triIndex.element(triId.mul(uint(3)).add(uint(k)));
    const matFloat = (matId, k) => materials.element(matId.mul(MSTRIDE).add(uint(k)));

    // ── Two-level BVH: instance + TLAS accessors ───────────────────
    // Instance records and TLAS nodes ride in the TAIL of the materials float
    // buffer (zero extra storage bindings — both kernels sit at the 8-buffer
    // budget). U.instBase / U.tlasBase are float-element bases; TLAS bounds
    // are plain floats, miss/payload are bit-cast uints. Instance record
    // (stride 28 = MAT_STRIDE): [0..11] inverse world 3×4 (rows, w =
    // translation), [12] blasRoot, [13] blasEnd (node indices, exact ≤2^24),
    // [14] winding sign (±1, mirrored instances flip Möller–Trumbore's det),
    // [15..27] reserved.
    const INST_STRIDE = uint(28);
    const instF = (inst, k) => materials.element(U.instBase.add(inst.mul(INST_STRIDE)).add(uint(k)));
    const instRow = (inst, r) => vec3(instF(inst, r * 4), instF(inst, r * 4 + 1), instF(inst, r * 4 + 2));
    const instInvPoint = (inst, p) => vec3(
        dot(instRow(inst, 0), p).add(instF(inst, 3)),
        dot(instRow(inst, 1), p).add(instF(inst, 7)),
        dot(instRow(inst, 2), p).add(instF(inst, 11)));
    const instInvDir = (inst, d) => vec3(dot(instRow(inst, 0), d), dot(instRow(inst, 1), d), dot(instRow(inst, 2), d));
    // world normal from a LOCAL normal: (M⁻¹)ᵀ·n = n.x·row0 + n.y·row1 + n.z·row2.
    const instNormalToWorld = (inst, n) => normalize(
        instRow(inst, 0).mul(n.x).add(instRow(inst, 1).mul(n.y)).add(instRow(inst, 2).mul(n.z)));
    const instDetSign = (inst) => instF(inst, 14);
    const instLocalRay = (inst, ro, rd) => ({ ro: instInvPoint(inst, ro), rd: instInvDir(inst, rd) });
    // TLAS node = 12 PLAIN floats: [0..5] bounds, [6] miss, [7] instOffset,
    // [8] instCount (0 = interior → descend), [9..11] reserved. Every value is
    // an exact small integer as f32 — NO bit-casts: uint payloads stored in a
    // float buffer die on some drivers (denormal miss links flush to zero,
    // 0xFFFFFFFF interior markers are NaNs whose bits may canonicalize).
    const TLAS_STRIDE = uint(12);
    const tlasF = (n, k) => materials.element(U.tlasBase.add(n.mul(TLAS_STRIDE)).add(uint(k)));

    // ── PBR map array textures ─────────────────────────────────────
    const albedoTex = maps.albedo || null;
    const normalTex = maps.normal || null;
    const roughTex = maps.roughness || null;
    const metalTex = maps.metalness || null;
    const emissiveTex = maps.emissive || null;
    const alphaTex = maps.alpha || null;
    const haveAlbedoMap = !!albedoTex;
    const haveNormalMap = !!normalTex;
    const haveRoughMap = !!roughTex;
    const haveMetalMap = !!metalTex;
    const haveEmissiveMap = !!emissiveTex;
    const haveAlphaMap = !!alphaTex;

    // sRGB → linear (exact piecewise), component-wise.
    const srgbToLinear = (c) => select(
        c.lessThanEqual(vec3(0.04045)),
        c.div(12.92),
        pow(max(c.add(0.055).div(1.055), vec3(0)), vec3(2.4)),
    );
    // Sample a map-array layer at the transformed UV.
    const sampleLayer = (tex, uv, layerF) =>
        texture(tex, uv).depth(int(max(layerF, float(0)))).level(0).xyz;
    const sampleLayerRGBA = (tex, uv, layerF) =>
        texture(tex, uv).depth(int(max(layerF, float(0)))).level(0);

    const materialSideAccepts = (matId, det) => {
        const side = matFloat(matId, 22);
        const frontFace = det.greaterThan(float(0));
        const frontSide = side.lessThan(float(0.5));
        const backSide = side.greaterThanEqual(float(0.5)).and(side.lessThan(float(1.5)));
        const doubleSide = side.greaterThanEqual(float(1.5));
        return doubleSide
            .or(frontSide.and(frontFace))
            .or(backSide.and(det.lessThan(float(0))));
    };

    const materialAlphaAccepts = (matId, uv) => {
        const alpha = matFloat(matId, 10).toVar();
        if (haveAlbedoMap) {
            const aL = matFloat(matId, 12);
            const rgba = sampleLayerRGBA(albedoTex, uv, aL);
            alpha.assign(alpha.mul(select(aL.greaterThan(float(-0.5)), rgba.w, float(1))));
        }
        if (haveAlphaMap) {
            const aL = matFloat(matId, 24);
            const rgba = sampleLayerRGBA(alphaTex, uv, aL);
            // three.js alphaMap samples the green channel.
            alpha.assign(alpha.mul(select(aL.greaterThan(float(-0.5)), rgba.y, float(1))));
        }
        const alphaTest = matFloat(matId, 23);
        return alpha.greaterThan(float(1.0e-4))
            .and(alphaTest.lessThanEqual(float(0)).or(alpha.greaterThanEqual(alphaTest)));
    };

    const hitUV = (triId, u, vbar) => {
        const w = float(1).sub(u).sub(vbar);
        return fetchUV(triVert(triId, 0)).mul(w)
            .add(fetchUV(triVert(triId, 1)).mul(u))
            .add(fetchUV(triVert(triId, 2)).mul(vbar));
    };

    // ── Jakob–Hanika RGB → reflectance upsampling ──────────────────
    const haveLut = !!(lut && lut.length === 3 && lutRes > 1);
    const LUTN = float(lutRes || 2);
    const lutUV = (g) => clamp(g, 0.0, 1.0).mul(LUTN.sub(1)).add(0.5).div(LUTN);
    const jhCoeffs = (rgb) => {
        const r = rgb.x, g = rgb.y, b = rgb.z;
        const fetch = (lt, zc, a, c) => {
            const z = max(zc, float(1e-4));
            return texture3D(lt, vec3(lutUV(a.div(z)), lutUV(c.div(z)), lutUV(clamp(zc, 0.0, 1.0)))).xyz;
        };
        const c0 = fetch(lut[0], r, g, b); // r is max: x=g/z, y=b/z
        const c1 = fetch(lut[1], g, b, r); // g is max: x=b/z, y=r/z
        const c2 = fetch(lut[2], b, r, g); // b is max: x=r/z, y=g/z
        const isB = b.greaterThanEqual(g).and(b.greaterThanEqual(r));
        const isG = g.greaterThanEqual(r);
        return select(isB, c2, select(isG, c1, c0));
    };
    const jhEval = (c, lambda) => {
        // Ln normalization stays anchored to the VISIBLE range the LUT was
        // fitted over, but is NOT clamped above 1: past 720 nm the
        // sigmoid-quadratic extrapolates smoothly into NIR (the old clamp
        // froze the spectrum at its red-edge value). The sigmoid is in [0,1]
        // by construction; the outer clamp is only an fp guard.
        const Ln = max(lambda.sub(LAMBDA_MIN).div(LAMBDA_RANGE), 0.0);
        const x = c.x.mul(Ln).add(c.y).mul(Ln).add(c.z);
        return clamp(float(0.5).add(x.mul(0.5).div(sqrt(x.mul(x).add(1.0)))), 0.0, 1.0);
    };
    const jhReflectanceBase = haveLut
        ? (rgb, lambda) => jhEval(jhCoeffs(clamp(rgb, 0.0, 1.0)), lambda)
        : (rgb, lambda) => rgbToSpectral(rgb, lambda);
    // nirAlbedo (material slot [25]): −1 = untagged → JH extrapolation is the
    // default PRIOR; ≥0 = authored NIR reflectance, blended in across the red
    // edge. RGB carries zero information about NIR (metamerism) — foliage
    // red-edge, water absorption etc. are injected as data, never inferred.
    const jhReflectance = (rgb, lambda, nirAlbedo = null) => {
        const base = jhReflectanceBase(rgb, lambda);
        if (!nirAlbedo) return base;
        const blend = smoothstep(float(700.0), float(740.0), lambda);
        return select(
            nirAlbedo.lessThan(float(0.0)),
            base,
            mix(base, clamp(nirAlbedo, 0.0, 1.0), blend),
        );
    };
    const jhEmission = haveLut
        ? (rgb, lambda) => {
            const m = max(max(rgb.x, rgb.y), rgb.z);
            return jhReflectanceBase(rgb.div(max(m, float(1e-6))), lambda).mul(m);
        }
        : (rgb, lambda) => rgbToSpectral(rgb, lambda);

    // Light emission at λ for a tagged emitter class (lights buffer slot [16]).
    // Packed encoding (see spectral_scene.js collectLights): 0 = untagged
    // (JH prior), 2 = LED, 3 = sodium (LPS), 4 = IR illuminator; any value
    // ≥ 500 = incandescent/halogen with that colour temperature in Kelvin.
    // Env emission stays JH (envAtLambda is untouched).
    const emitterAtLambda = (rgb, lambda, eclass) => {
        const base = jhEmission(rgb, lambda);
        const lum = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
        // incandescent: RGB-derived intensity scaled by the Planck ratio
        // B(λ,T)/B(560,T) — the NIR tail that dominates through a tube.
        const T = max(eclass, float(500.0));
        const c2 = float(1.4388e7); // second radiation constant, nm·K
        const planckRatio = pow(float(560.0).div(lambda), float(5.0))
            .mul(exp(c2.div(T.mul(560.0))).sub(1.0))
            .div(max(exp(c2.div(T.mul(lambda))).sub(1.0), float(1e-6)));
        const incandescent = lum.mul(planckRatio);
        // LED: phosphor-white spectrum dies past ~700 nm (no NIR tail).
        const led = base.mul(float(1.0).sub(smoothstep(float(690.0), float(725.0), lambda)));
        // sodium: single 589 nm line, Y-conserving amplitude (constant is
        // precomputed CPU-side, see SODIUM_Y_SCALE).
        const dn = lambda.sub(589.0).div(SODIUM_SIGMA);
        const sodium = lum.mul(SODIUM_Y_SCALE).mul(exp(dn.mul(dn).mul(-0.5)));
        // IR illuminator: 850 nm band; its RGB is meaningless, so the light's
        // intensity (max channel) sets the band's PEAK spectral radiance.
        // Contributes ~nothing in visible mode — correct and expected.
        const di = lambda.sub(850.0).div(IR_ILLUM_SIGMA);
        const ir = max(max(rgb.x, rgb.y), rgb.z).mul(exp(di.mul(di).mul(-0.5)));
        const isLed = abs(eclass.sub(float(2.0))).lessThan(float(0.25));
        const isNa = abs(eclass.sub(float(3.0))).lessThan(float(0.25));
        const isIr = abs(eclass.sub(float(4.0))).lessThan(float(0.25));
        const isInc = eclass.greaterThanEqual(float(500.0));
        return select(isInc, incandescent,
            select(isIr, ir, select(isNa, sodium, select(isLed, led, base))));
    };

    // env radiance at λ for a world direction (or sky fallback)
    const envAtLambda = (dir, lambda) => {
        const out = float(0).toVar();
        if (env) {
            const cr = cos(U.envRotation), sr = sin(U.envRotation);
            const rdir = vec3(
                dir.x.mul(cr).sub(dir.z.mul(sr)),
                dir.y,
                dir.x.mul(sr).add(dir.z.mul(cr)),
            );
            const uv = equirectUV(normalize(rdir));
            const rgb = texture(env, uv).level(0).xyz.mul(U.envIntensity);
            out.assign(jhEmission(rgb, lambda));
        } else {
            out.assign(float(0));
        }
        return out;
    };

    // ── BLAS closest-hit for ONE instance (LOCAL space) ─────────────
    // The local ray keeps the direction UNNORMALIZED, so the ray parameter t
    // is IDENTICAL in local and world space (affine invariance): best-t
    // comparisons stay valid across instances and non-uniform scale works.
    // Barycentrics (u,v) are affine-invariant too, so hitUV/alpha tests are
    // exact. Only the det SIGN flips under mirroring — dSign corrects it.
    const blasClosest = (inst, wro, wrd, bestTVar, bestTriVar, bestInstVar) => {
        const ro = instInvPoint(inst, wro).toVar();
        const rd = instInvDir(inst, wrd).toVar();
        const dSign = instDetSign(inst);
        const invD = vec3(float(1).div(rd.x), float(1).div(rd.y), float(1).div(rd.z));
        const blasEnd = uint(instF(inst, 13)).toVar();
        const cursor = uint(instF(inst, 12)).toVar();
        Loop({ start: uint(0), end: U.nodeCount, type: 'uint', condition: '<' }, () => {
            If(cursor.greaterThanEqual(blasEnd), () => { Break(); });
            const base = cursor.mul(uint(8));
            const bmin = vec3(
                uintBitsToFloat(bvhNodes.element(base)),
                uintBitsToFloat(bvhNodes.element(base.add(uint(1)))),
                uintBitsToFloat(bvhNodes.element(base.add(uint(2)))));
            const bmax = vec3(
                uintBitsToFloat(bvhNodes.element(base.add(uint(3)))),
                uintBitsToFloat(bvhNodes.element(base.add(uint(4)))),
                uintBitsToFloat(bvhNodes.element(base.add(uint(5)))));
            const miss = bvhNodes.element(base.add(uint(6)));
            const payload = bvhNodes.element(base.add(uint(7)));

            const t0 = bmin.sub(ro).mul(invD);
            const t1 = bmax.sub(ro).mul(invD);
            const tsmall = min(t0, t1);
            const tbig = max(t0, t1);
            const tNear = max(max(tsmall.x, tsmall.y), tsmall.z);
            const tFar = min(min(tbig.x, tbig.y), tbig.z);

            If(tFar.greaterThanEqual(max(tNear, float(0))).and(tNear.lessThan(bestTVar)), () => {
                If(payload.equal(uint(0xFFFFFFFF)), () => {
                    cursor.assign(cursor.add(uint(1)));
                }).Else(() => {
                    const triOffset = payload.bitAnd(uint(0x00FFFFFF));
                    const triCount = payload.shiftRight(uint(24));
                    Loop({ start: uint(0), end: triCount, type: 'uint', condition: '<' }, ({ i }) => {
                        const triId = triOffset.add(i);
                        const i0 = triVert(triId, 0);
                        const i1 = triVert(triId, 1);
                        const i2 = triVert(triId, 2);
                        const p0 = fetchVert(i0);
                        const p1 = fetchVert(i1);
                        const p2 = fetchVert(i2);
                        const e1 = p1.sub(p0);
                        const e2 = p2.sub(p0);
                        const pv = cross(rd, e2);
                        const det = dot(e1, pv);
                        If(abs(det).greaterThan(float(1e-12)), () => {
                            const invDet = float(1).div(det);
                            const tv = ro.sub(p0);
                            const u = dot(tv, pv).mul(invDet);
                            If(u.greaterThanEqual(float(0)).and(u.lessThanEqual(float(1))), () => {
                                const qv = cross(tv, e1);
                                const vbar = dot(rd, qv).mul(invDet);
                                If(vbar.greaterThanEqual(float(0)).and(u.add(vbar).lessThanEqual(float(1))), () => {
                                    const tHit = dot(e2, qv).mul(invDet);
                                    If(tHit.greaterThan(float(RAY_EPS)).and(tHit.lessThan(bestTVar)), () => {
                                        const matId = triMaterial.element(triId);
                                        const acceptsSide = materialSideAccepts(matId, det.mul(dSign));
                                        const acceptsAlpha = materialAlphaAccepts(matId, hitUV(triId, u, vbar));
                                        If(acceptsSide.and(acceptsAlpha), () => {
                                            bestTVar.assign(tHit);
                                            bestTriVar.assign(int(triId));
                                            bestInstVar.assign(int(inst));
                                        });
                                    });
                                });
                            });
                        });
                    });
                    cursor.assign(miss);
                });
            }).Else(() => {
                cursor.assign(miss);
            });
        });
    };

    // ── Two-level closest-hit: returns {t, tri, inst} via out-vars ──
    // Outer walk over the TLAS (instance world-AABBs, same skip-link node
    // encoding); each TLAS leaf descends into its instances' BLAS ranges.
    // bestInstVar is optional for callers that only need occlusion distance.
    const traverseClosest = (ro, rd, bestTVar, bestTriVar, bestInstVar = null) => {
        const bestInst = bestInstVar || int(-1).toVar();
        const invD = vec3(float(1).div(rd.x), float(1).div(rd.y), float(1).div(rd.z));
        const cursor = uint(0).toVar();
        Loop({ start: uint(0), end: U.tlasNodeCount, type: 'uint', condition: '<' }, () => {
            If(cursor.greaterThanEqual(U.tlasNodeCount), () => { Break(); });
            const bmin = vec3(tlasF(cursor, 0), tlasF(cursor, 1), tlasF(cursor, 2));
            const bmax = vec3(tlasF(cursor, 3), tlasF(cursor, 4), tlasF(cursor, 5));
            const miss = uint(tlasF(cursor, 6));
            const instCountF = tlasF(cursor, 8);

            const t0 = bmin.sub(ro).mul(invD);
            const t1 = bmax.sub(ro).mul(invD);
            const tNear = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
            const tFar = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));

            If(tFar.greaterThanEqual(max(tNear, float(0))).and(tNear.lessThan(bestTVar)), () => {
                If(instCountF.lessThan(float(0.5)), () => {
                    cursor.assign(cursor.add(uint(1)));
                }).Else(() => {
                    const instOffset = uint(tlasF(cursor, 7));
                    Loop({ start: uint(0), end: uint(instCountF), type: 'uint', condition: '<' }, ({ i }) => {
                        blasClosest(instOffset.add(i), ro, rd, bestTVar, bestTriVar, bestInst);
                    });
                    cursor.assign(miss);
                });
            }).Else(() => {
                cursor.assign(miss);
            });
        });
    };

    // BLAS any-hit for ONE instance (LOCAL space, early-out via blockedVar).
    const blasAny = (inst, wro, wrd, maxDist, blockedVar) => {
        const ro = instInvPoint(inst, wro).toVar();
        const rd = instInvDir(inst, wrd).toVar();
        const dSign = instDetSign(inst);
        const invD = vec3(float(1).div(rd.x), float(1).div(rd.y), float(1).div(rd.z));
        const blasEnd = uint(instF(inst, 13)).toVar();
        const cursor = uint(instF(inst, 12)).toVar();
        Loop({ start: uint(0), end: U.nodeCount, type: 'uint', condition: '<' }, () => {
            If(cursor.greaterThanEqual(blasEnd).or(blockedVar.greaterThan(float(0.5))), () => { Break(); });
            const base = cursor.mul(uint(8));
            const bmin = vec3(uintBitsToFloat(bvhNodes.element(base)), uintBitsToFloat(bvhNodes.element(base.add(uint(1)))), uintBitsToFloat(bvhNodes.element(base.add(uint(2)))));
            const bmax = vec3(uintBitsToFloat(bvhNodes.element(base.add(uint(3)))), uintBitsToFloat(bvhNodes.element(base.add(uint(4)))), uintBitsToFloat(bvhNodes.element(base.add(uint(5)))));
            const miss = bvhNodes.element(base.add(uint(6)));
            const payload = bvhNodes.element(base.add(uint(7)));
            const t0 = bmin.sub(ro).mul(invD);
            const t1 = bmax.sub(ro).mul(invD);
            const tNear = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
            const tFar = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
            If(tFar.greaterThanEqual(max(tNear, float(0))).and(tNear.lessThan(maxDist)), () => {
                If(payload.equal(uint(0xFFFFFFFF)), () => {
                    cursor.assign(cursor.add(uint(1)));
                }).Else(() => {
                    const triOffset = payload.bitAnd(uint(0x00FFFFFF));
                    const triCount = payload.shiftRight(uint(24));
                    Loop({ start: uint(0), end: triCount, type: 'uint', condition: '<' }, ({ i }) => {
                        const triId = triOffset.add(i);
                        const p0 = fetchVert(triVert(triId, 0));
                        const p1 = fetchVert(triVert(triId, 1));
                        const p2 = fetchVert(triVert(triId, 2));
                        const e1 = p1.sub(p0); const e2 = p2.sub(p0);
                        const pv = cross(rd, e2); const det = dot(e1, pv);
                        If(abs(det).greaterThan(float(1e-12)), () => {
                            const invDet = float(1).div(det);
                            const tv = ro.sub(p0);
                            const u = dot(tv, pv).mul(invDet);
                            If(u.greaterThanEqual(float(0)).and(u.lessThanEqual(float(1))), () => {
                                const qv = cross(tv, e1);
                                const vb = dot(rd, qv).mul(invDet);
                                If(vb.greaterThanEqual(float(0)).and(u.add(vb).lessThanEqual(float(1))), () => {
                                    const tHit = dot(e2, qv).mul(invDet);
                                    If(tHit.greaterThan(float(RAY_EPS)).and(tHit.lessThan(maxDist)), () => {
                                        const matId = triMaterial.element(triId);
                                        const acceptsSide = materialSideAccepts(matId, det.mul(dSign));
                                        const acceptsAlpha = materialAlphaAccepts(matId, hitUV(triId, u, vb));
                                        const occTrans = matFloat(matId, 5);
                                        If(acceptsSide.and(acceptsAlpha).and(occTrans.lessThan(float(0.5))), () => { blockedVar.assign(float(1)); });
                                    });
                                });
                            });
                        });
                    });
                    cursor.assign(miss);
                });
            }).Else(() => { cursor.assign(miss); });
        });
    };

    // any-hit (shadow) traversal: returns 1.0 if blocked within maxDist
    const traverseAny = (ro, rd, maxDist) => {
        const invD = vec3(float(1).div(rd.x), float(1).div(rd.y), float(1).div(rd.z));
        const cursor = uint(0).toVar();
        const blocked = float(0).toVar();
        Loop({ start: uint(0), end: U.tlasNodeCount, type: 'uint', condition: '<' }, () => {
            If(cursor.greaterThanEqual(U.tlasNodeCount).or(blocked.greaterThan(float(0.5))), () => { Break(); });
            const bmin = vec3(tlasF(cursor, 0), tlasF(cursor, 1), tlasF(cursor, 2));
            const bmax = vec3(tlasF(cursor, 3), tlasF(cursor, 4), tlasF(cursor, 5));
            const miss = uint(tlasF(cursor, 6));
            const instCountF = tlasF(cursor, 8);
            const t0 = bmin.sub(ro).mul(invD);
            const t1 = bmax.sub(ro).mul(invD);
            const tNear = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
            const tFar = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
            If(tFar.greaterThanEqual(max(tNear, float(0))).and(tNear.lessThan(maxDist)), () => {
                If(instCountF.lessThan(float(0.5)), () => {
                    cursor.assign(cursor.add(uint(1)));
                }).Else(() => {
                    const instOffset = uint(tlasF(cursor, 7));
                    Loop({ start: uint(0), end: uint(instCountF), type: 'uint', condition: '<' }, ({ i }) => {
                        blasAny(instOffset.add(i), ro, rd, maxDist, blocked);
                    });
                    cursor.assign(miss);
                });
            }).Else(() => { cursor.assign(miss); });
        });
        return blocked;
    };

    // cosine-weighted hemisphere sample around n
    const cosineSample = (n, r1, r2) => {
        const phi = r1.mul(2 * PI);
        const cosT = sqrt(float(1).sub(r2));
        const sinT = sqrt(r2);
        const a = select(abs(n.y).lessThan(float(0.999)), vec3(0, 1, 0), vec3(1, 0, 0));
        const t = normalize(cross(a, n));
        const b = cross(n, t);
        return normalize(
            t.mul(cos(phi).mul(sinT))
                .add(b.mul(sin(phi).mul(sinT)))
                .add(n.mul(cosT)));
    };

    return {
        fetchVert, fetchNorm, fetchUV, hitUV, triVert, matFloat,
        srgbToLinear, sampleLayer,
        jhReflectance, jhEmission, emitterAtLambda, envAtLambda, cosineSample,
        traverseClosest, traverseAny,
        instLocalRay, instNormalToWorld, instDetSign,
        haveAlbedoMap, haveNormalMap, haveRoughMap, haveMetalMap, haveEmissiveMap, haveAlphaMap,
        albedoTex, normalTex, roughTex, metalTex, emissiveTex, alphaTex,
    };
}
