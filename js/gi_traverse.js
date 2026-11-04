// gi_traverse.js — TSL BVH traversal + material helpers for the
// SPEEDBALL GI probe kernel.
//
// buildTraversal() returns graph EMITTERS (not values): each call site builds a
// fresh subgraph. It closes over the storage nodes + uniforms passed in, so the
// probe kernel binds its own storage()/U over the shared scene buffers.

import * as TSL from 'three/tsl';

const {
    Loop, If, Break, texture,
    int, uint, vec2, vec3,
    uintBitsToFloat,
    select, max, min, abs, pow,
    dot, cross, normalize, float,
} = TSL;

export const PI = 3.141592653589793;
export const T_MAX = 1e30;
export const RAY_EPS = 1e-3;

// Build the shared graph emitters bound to a given set of storage nodes +
// uniforms. `storages` = { bvhNodes, triIndex, vertexData, triMaterial,
// materials } (TSL storage nodes). `U` must expose nodeCount (BLAS pool),
// tlasNodeCount, instBase, tlasBase (uint uniforms — instance records + TLAS
// nodes live in the tail of the materials buffer, see gi_scene.js).
// `maps` holds the scene's PBR DataArrayTextures ({} when none). Traversal is
// TWO-LEVEL: world-ray TLAS walk over
// instance AABBs → local-ray BLAS walk per instance; hit shading transforms
// back via instLocalRay / instNormalToWorld (vertex data is LOCAL space).
export function buildTraversal({ storages, U, maps = {} }) {
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
    // [15] active flag (reserved InstancedMesh capacity), [16..27] reserved.
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
    const materialTraversalFlags = (matId) => uint(matFloat(matId, 26));
    const flagSet = (flags, bit) => flags.bitAnd(uint(bit)).notEqual(uint(0));

    const hitUV = (triId, u, vbar) => {
        const w = float(1).sub(u).sub(vbar);
        return fetchUV(triVert(triId, 0)).mul(w)
            .add(fetchUV(triVert(triId, 1)).mul(u))
            .add(fetchUV(triVert(triId, 2)).mul(vbar));
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
                                        const flags = materialTraversalFlags(matId);
                                        const commitHit = () => {
                                            bestTVar.assign(tHit);
                                            bestTriVar.assign(int(triId));
                                            bestInstVar.assign(int(inst));
                                        };
                                        If(materialSideAccepts(matId, det.mul(dSign)), () => {
                                            If(flagSet(flags, 1), () => {
                                                If(materialAlphaAccepts(matId, hitUV(triId, u, vbar)), commitHit);
                                            }).Else(() => {
                                                If(flagSet(flags, 2), commitHit);
                                            });
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
                        const inst = instOffset.add(i);
                        If(instF(inst, 15).greaterThan(float(0.5)), () => {
                            blasClosest(inst, ro, rd, bestTVar, bestTriVar, bestInst);
                        });
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
                                        const flags = materialTraversalFlags(matId);
                                        If(materialSideAccepts(matId, det.mul(dSign)).and(flagSet(flags, 4)), () => {
                                            If(flagSet(flags, 1), () => {
                                                If(materialAlphaAccepts(matId, hitUV(triId, u, vb)), () => { blockedVar.assign(float(1)); });
                                            }).Else(() => {
                                                If(flagSet(flags, 2), () => { blockedVar.assign(float(1)); });
                                            });
                                        });
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
                        const inst = instOffset.add(i);
                        If(instF(inst, 15).greaterThan(float(0.5)), () => {
                            blasAny(inst, ro, rd, maxDist, blocked);
                        });
                    });
                    cursor.assign(miss);
                });
            }).Else(() => { cursor.assign(miss); });
        });
        return blocked;
    };

    return {
        fetchVert, fetchNorm, fetchUV, hitUV, triVert, matFloat,
        srgbToLinear, sampleLayer,
        traverseClosest, traverseAny,
        instLocalRay, instNormalToWorld, instDetSign,
        haveAlbedoMap, haveNormalMap, haveRoughMap, haveMetalMap, haveEmissiveMap, haveAlphaMap,
        albedoTex, normalTex, roughTex, metalTex, emissiveTex, alphaTex,
    };
}
