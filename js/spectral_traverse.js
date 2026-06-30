// spectral_traverse.js — shared TSL traversal + shading helpers for the
// spectral path tracer AND the HALO-GI probe kernel (docs/GI_HALO_design.md §3.4).
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
    select, max, min, abs, sqrt, sin, cos, exp, pow,
    dot, cross, normalize, reflect, mix, clamp, float,
} = TSL;

export const PI = 3.141592653589793;
export const LAMBDA_MIN = 380.0;
export const LAMBDA_MAX = 720.0;
export const LAMBDA_RANGE = LAMBDA_MAX - LAMBDA_MIN;
export const T_MAX = 1e30;
export const RAY_EPS = 1e-3;

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
// materials } (TSL storage nodes). `U` must expose nodeCount, envRotation,
// envIntensity uniforms. `env` (equirect texture | null), `lut` (3 Data3D | null)
// and `maps` (PBR DataArrayTextures | {}) mirror buildKernels' inputs.
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
        texture(tex, uv).depth(int(max(layerF, float(0)))).xyz;
    const sampleLayerRGBA = (tex, uv, layerF) =>
        texture(tex, uv).depth(int(max(layerF, float(0))));

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
        const Ln = clamp(lambda.sub(LAMBDA_MIN).div(LAMBDA_RANGE), 0.0, 1.0);
        const x = c.x.mul(Ln).add(c.y).mul(Ln).add(c.z);
        return float(0.5).add(x.mul(0.5).div(sqrt(x.mul(x).add(1.0))));
    };
    const jhReflectance = haveLut
        ? (rgb, lambda) => jhEval(jhCoeffs(clamp(rgb, 0.0, 1.0)), lambda)
        : (rgb, lambda) => rgbToSpectral(rgb, lambda);
    const jhEmission = haveLut
        ? (rgb, lambda) => {
            const m = max(max(rgb.x, rgb.y), rgb.z);
            return jhReflectance(rgb.div(max(m, float(1e-6))), lambda).mul(m);
        }
        : (rgb, lambda) => rgbToSpectral(rgb, lambda);

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

    // ── BVH closest-hit: returns {t, tri} via out-vars ──────────────
    const traverseClosest = (ro, rd, bestTVar, bestTriVar) => {
        const invD = vec3(float(1).div(rd.x), float(1).div(rd.y), float(1).div(rd.z));
        const cursor = uint(0).toVar();
        Loop({ start: uint(0), end: U.nodeCount, type: 'uint', condition: '<' }, () => {
            If(cursor.greaterThanEqual(U.nodeCount), () => { Break(); });
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
                                        const acceptsSide = materialSideAccepts(matId, det);
                                        const acceptsAlpha = materialAlphaAccepts(matId, hitUV(triId, u, vbar));
                                        If(acceptsSide.and(acceptsAlpha), () => {
                                            bestTVar.assign(tHit);
                                            bestTriVar.assign(int(triId));
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

    // any-hit (shadow) traversal: returns 1.0 if blocked within maxDist
    const traverseAny = (ro, rd, maxDist) => {
        const invD = vec3(float(1).div(rd.x), float(1).div(rd.y), float(1).div(rd.z));
        const cursor = uint(0).toVar();
        const blocked = float(0).toVar();
        Loop({ start: uint(0), end: U.nodeCount, type: 'uint', condition: '<' }, () => {
            If(cursor.greaterThanEqual(U.nodeCount).or(blocked.greaterThan(float(0.5))), () => { Break(); });
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
                                        const acceptsSide = materialSideAccepts(matId, det);
                                        const acceptsAlpha = materialAlphaAccepts(matId, hitUV(triId, u, vb));
                                        const occTrans = matFloat(matId, 5);
                                        If(acceptsSide.and(acceptsAlpha).and(occTrans.lessThan(float(0.5))), () => { blocked.assign(float(1)); });
                                    });
                                });
                            });
                        });
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
        fetchVert, fetchNorm, fetchUV, triVert, matFloat,
        srgbToLinear, sampleLayer,
        jhReflectance, jhEmission, envAtLambda, cosineSample,
        traverseClosest, traverseAny,
        haveAlbedoMap, haveNormalMap, haveRoughMap, haveMetalMap, haveEmissiveMap, haveAlphaMap,
        albedoTex, normalTex, roughTex, metalTex, emissiveTex, alphaTex,
    };
}
