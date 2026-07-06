// spectral_kernel.js — TSL compute kernel + blit material for the spectral
// path tracer. Pure-TSL (no raw WGSL); mirrors the compute idioms proven in
// web/js/layer_deform.js and web/js/gpu_normals.js on this r185 build.
//
// One invocation = one pixel = one full path (1 spp/frame, accumulated). The
// path carries a single hero wavelength λ; material reflectance is a 3-bin
// spectrum sampled at λ; the path's scalar radiance is converted to CIE XYZ
// (Wyman 2013 analytic fits) and summed in a storage buffer. The blit divides
// by the global sample count and maps XYZ→sRGB. "Fast not accurate" by design.
//
// Buffer strides come from spectral_scene.js: bvhNodes u32×8, triIndex u32×3,
// vertexData f32×8 (pos+normal+uv), triMaterial u32×1, materials f32×28,
// lights f32×17, accum f32×4 (XYZ + NIR photocathode flux). PBR maps arrive
// as DataArrayTextures (one per type) sampled at the hit UV.
//
// Render modes (buildKernels `mode`):
//   'visible' — λ ∈ [380,720] uniform, Wyman XYZ accumulation (unchanged).
//   'nv'      — λ ∈ [550,900] importance-sampled against the Gen-3 GaAs
//               photocathode response; the path's radiance × S(λ)/pdf(λ),
//               normalized by ∫S dλ, accumulates in the 4th channel as true
//               electron flux. The blit then emits LINEAR flux (no tone map,
//               no sRGB) for the image-intensifier stage
//               (powershot_infrared.js, setInputMode("nir")).
//
// The BVH traversal + JH-spectral/env/PBR shading emitters live in
// spectral_traverse.js (shared byte-identically with the HALO-GI probe kernel).

import * as TSL from 'three/tsl';

const {
    Fn, Loop, If, Break, Return, instanceIndex, uniform, uniformArray, storage, texture, texture3D,
    float, int, uint, vec2, vec3, vec4,
    uintBitsToFloat, equirectUV,
    select, max, min, abs, sqrt, sin, cos, exp, pow, floor, fract,
    dot, cross, normalize, reflect, mix, clamp, length, smoothstep,
} = TSL;

import {
    buildTraversal,
    photocathodeS, buildPhotocathodeSampler,
    PI, LAMBDA_MIN, LAMBDA_MAX, LAMBDA_RANGE,
    NV_LAMBDA_MIN, NV_LAMBDA_MAX, T_MAX, RAY_EPS,
} from './spectral_traverse.js';

// ∫ȳ(λ)dλ over [380,720] for the Wyman fits below. Dividing the XYZ Monte
// Carlo estimate by this maps a unit flat spectrum to Y=1 (neutral white);
// without it a fully-lit white surface lands at Y≈107 and blows out to white.
const CIE_Y_INTEGRAL = 106.936;
const INV_U32 = 1.0 / 4294967296.0;

// ── Wyman 2013 single-lobe CIE 1931 fits ───────────────────────────
function wymanG(x, mu, s1, s2) {
    const t = x.sub(mu);
    const s = select(x.lessThan(mu), float(s1), float(s2));
    const e = t.mul(s);
    return exp(e.mul(e).mul(-0.5));
}
function cieX(l) {
    return wymanG(l, 599.8, 0.0264, 0.0323).mul(1.056)
        .add(wymanG(l, 442.0, 0.0624, 0.0374).mul(0.362))
        .sub(wymanG(l, 501.1, 0.0490, 0.0382).mul(0.065));
}
function cieY(l) {
    return wymanG(l, 568.8, 0.0213, 0.0247).mul(0.821)
        .add(wymanG(l, 530.9, 0.0613, 0.0322).mul(0.286));
}
function cieZ(l) {
    return wymanG(l, 437.0, 0.0845, 0.0278).mul(1.217)
        .add(wymanG(l, 459.0, 0.0385, 0.0725).mul(0.681));
}

export function buildKernels({
    THREE, buffers, env, lut = null, lutRes = 0, maps = {}, width, height,
    // 'visible' | 'nv'. nvSampling 'uniform' exists only for A/B-validating
    // the importance estimator (means must agree within Monte Carlo noise).
    mode = 'visible', nvSampling = 'importance',
}) {
    const isNV = mode === 'nv';
    const nvImportance = isNV && nvSampling !== 'uniform';
    // Per-mode λ domain. Visible keeps [380,720] — byte-identical behaviour.
    const lambdaMin = isNV ? NV_LAMBDA_MIN : LAMBDA_MIN;
    const lambdaRange = (isNV ? NV_LAMBDA_MAX : LAMBDA_MAX) - lambdaMin;
    // NV λ sampler: 64-entry inverse-CDF LUT + ∫S dλ (flux normalization —
    // the analogue of CIE_Y_INTEGRAL: a unit-radiance flat-spectrum scene
    // lands near flux 1.0).
    const pcSampler = isNV ? buildPhotocathodeSampler() : null;
    const pcLutNode = nvImportance ? uniformArray(Array.from(pcSampler.lut), 'float') : null;
    const pcLutN = pcSampler ? pcSampler.lut.length : 0;
    // Storage nodes
    const bvhNodes = storage(buffers.bvhNodes, 'uint', buffers.bvhNodes.count);
    const triIndex = storage(buffers.triIndex, 'uint', buffers.triIndex.count);
    // Interleaved per-vertex: [px,py,pz, nx,ny,nz, u,v] (stride 8) — packed to
    // stay within the 8 storage-buffer budget.
    const vertexData = storage(buffers.vertexData, 'float', buffers.vertexData.count);
    const triMaterial = storage(buffers.triMaterial, 'uint', buffers.triMaterial.count);
    const materials = storage(buffers.materials, 'float', buffers.materials.count);
    const lights = storage(buffers.lights, 'float', buffers.lights.count);
    const accum = storage(buffers.accum, 'float', buffers.accum.count);

    // Uniforms (driven each frame by the tracer)
    const U = {
        camWorld: uniform(new THREE.Matrix4()),
        camProjInv: uniform(new THREE.Matrix4()),
        camPos: uniform(new THREE.Vector3()),
        resolution: uniform(new THREE.Vector2(width, height)),
        frameSeed: uniform(0, 'uint'),
        sampleCount: uniform(1),
        bounceCap: uniform(4, 'uint'),
        rrStart: uniform(2, 'uint'),
        radianceClamp: uniform(8.0),
        envIntensity: uniform(1.0),
        envRotation: uniform(0.0),
        hasEnv: uniform(env ? 1 : 0, 'uint'),
        lightCount: uniform(buffers.lightCount >>> 0, 'uint'),
        nodeCount: uniform(buffers.nodeCount >>> 0, 'uint'),
        tlasNodeCount: uniform((buffers.tlasNodeCount ?? 0) >>> 0, 'uint'),
        instBase: uniform((buffers.instBase ?? 0) >>> 0, 'uint'),
        tlasBase: uniform((buffers.tlasBase ?? 0) >>> 0, 'uint'),
        exposure: uniform(1.0),
        // Thin-lens depth of field. apertureRadius 0 = pinhole (DOF off).
        apertureRadius: uniform(0.0),
        focusDistance: uniform(5.0),
        // 1 = tone-map + sRGB encode in the blit (direct-to-canvas);
        // 0 = emit LINEAR HDR for an external post stack to tone-map.
        toneMapEnabled: uniform(1, 'uint'),
    };

    const W = U.resolution.x;
    const H = U.resolution.y;

    // ── shared traversal + shading emitters (BVH closest/any-hit, JH spectral,
    // env, PBR map sampling) bound to this kernel's storage + uniforms. Extracted
    // to spectral_traverse.js so the HALO-GI probe kernel reuses identical logic
    // against the same resident BVH (no second acceleration structure).
    const trav = buildTraversal({
        storages: { bvhNodes, triIndex, vertexData, triMaterial, materials },
        U, env, lut, lutRes, maps,
    });
    const {
        fetchVert, fetchNorm, fetchUV, triVert, matFloat,
        srgbToLinear, sampleLayer,
        jhReflectance, jhEmission, emitterAtLambda, envAtLambda, cosineSample,
        traverseClosest, traverseAny, instLocalRay, instNormalToWorld,
        haveAlbedoMap, haveNormalMap, haveRoughMap, haveMetalMap, haveEmissiveMap,
        albedoTex, normalTex, roughTex, metalTex, emissiveTex,
    } = trav;

    // PCG integer finalizer (full avalanche) — used to derive a well-mixed
    // initial state from (pixel, frame). The OLD seed `(pix ^ frameSeed)*prime`
    // only perturbed the low bits with frameSeed, so a pixel's mean over the
    // accumulated frames was dominated by its high bits (= its row) → each row
    // converged to a biased wavelength tint → horizontal colored banding.
    const pcgHash = (x) => {
        const st = x.mul(uint(747796405)).add(uint(2891336453));
        const word = st.shiftRight(st.shiftRight(uint(28)).add(uint(4))).bitXor(st).mul(uint(277803737));
        return word.shiftRight(uint(22)).bitXor(word);
    };

    // PCG-style RNG over a uint var; returns float in [0,1) and advances state.
    const rngState = { node: null };
    const nextRand = () => {
        const s = rngState.node;
        const ns = s.mul(uint(747796405)).add(uint(2891336453));
        s.assign(ns);
        const word = ns.shiftRight(ns.shiftRight(uint(28)).add(uint(4))).bitXor(ns).mul(uint(277803737));
        const res = word.shiftRight(uint(22)).bitXor(word);
        return float(res).mul(INV_U32);
    };

    // ── main trace kernel ──────────────────────────────────────────
    const traceKernel = Fn(() => {
        const pix = instanceIndex.toVar();
        const total = uint(width * height);
        If(pix.greaterThanEqual(total), () => { Return(); });

        // Fully hash both pixel index and frame into the initial RNG state so
        // every pixel samples wavelengths uniformly over the accumulation (no
        // per-row bias). See pcgHash note above.
        rngState.node = pcgHash(pix.bitXor(pcgHash(U.frameSeed))).toVar();

        const px = float(pix.mod(uint(width)));
        const py = float(pix.div(uint(width)));
        const jx = nextRand();
        const jy = nextRand();
        // NDC (-1..1), flip Y so image matches the rasterized camera
        const ndcX = px.add(jx).div(W).mul(2).sub(1);
        const ndcY = float(1).sub(py.add(jy).div(H).mul(2));

        // primary ray from inverse projection * camera world
        const clip = vec4(ndcX, ndcY, float(-1), float(1));
        const viewPos = U.camProjInv.mul(clip);
        const viewDir = vec3(viewPos.x, viewPos.y, viewPos.z).div(viewPos.w);
        const ro = vec3(U.camPos).toVar();
        const rdWorld = U.camWorld.mul(vec4(viewDir, 0)).xyz;
        const rd = normalize(rdWorld).toVar();

        // Thin-lens DOF: jitter the ray origin across the aperture disk and
        // re-aim at the point where the pinhole ray crosses the focal plane.
        // Physically-correct bokeh; converges with the accumulation. Pinhole
        // when apertureRadius == 0. (Uniform branch → consistent RNG draw.)
        If(U.apertureRadius.greaterThan(float(0)), () => {
            const camRight = U.camWorld.mul(vec4(1, 0, 0, 0)).xyz;
            const camUp = U.camWorld.mul(vec4(0, 1, 0, 0)).xyz;
            const camFwd = U.camWorld.mul(vec4(0, 0, -1, 0)).xyz;
            const tFocus = U.focusDistance.div(max(dot(rd, camFwd), float(1e-3)));
            const focusPoint = ro.add(rd.mul(tFocus));
            const ar = sqrt(nextRand()).mul(U.apertureRadius);
            const ang = nextRand().mul(2 * PI);
            const lensOff = camRight.mul(cos(ang).mul(ar)).add(camUp.mul(sin(ang).mul(ar)));
            ro.assign(ro.add(lensOff));
            rd.assign(normalize(focusPoint.sub(ro)));
        });

        // Hero wavelength. Visible: uniform over the domain. NV: drawn from
        // the photocathode inverse-CDF LUT, so λ lands where the tube actually
        // responds (pdf ∝ S(λ)); the pdf then cancels against S(λ) in the
        // accumulation weight below.
        const lambda = float(lambdaMin).add(nextRand().mul(lambdaRange)).toVar();
        if (nvImportance) {
            const u = nextRand().mul(pcLutN - 1);
            const i0 = uint(floor(u));
            const i1 = min(i0.add(uint(1)), uint(pcLutN - 1));
            lambda.assign(mix(pcLutNode.element(i0), pcLutNode.element(i1), fract(u)));
        }
        const throughput = float(1).toVar();
        const radiance = float(0).toVar();

        Loop({ start: uint(0), end: U.bounceCap, type: 'uint', condition: '<' }, ({ i }) => {
            // GI/firefly clamp ("GI Clamp" control): cap each INDIRECT bounce's
            // radiance contribution at radianceClamp. Direct contributions
            // (i==0 — a directly visible emitter/HDRI, or first-hit NEE) stay
            // exact, so only multi-bounce spikes (a bright emitter or HDRI sun
            // reached through a low-probability bounce) get bounded. Clamping
            // the throughput-weighted contribution is what actually suppresses
            // fireflies — clamping throughput alone never fires, since for
            // energy-conserving materials throughput ≤ 1 ≤ radianceClamp.
            const clampGI = (contrib) => select(i.greaterThan(uint(0)), min(contrib, U.radianceClamp), contrib);

            const bestT = float(T_MAX).toVar();
            const bestTri = int(-1).toVar();
            const bestInst = int(-1).toVar();
            traverseClosest(ro, rd, bestT, bestTri, bestInst);

            If(bestTri.lessThan(int(0)), () => {
                radiance.addAssign(clampGI(throughput.mul(envAtLambda(rd, lambda))));
                Break();
            });

            const triId = uint(bestTri);
            const instId = uint(bestInst);
            const matId = triMaterial.element(triId);
            // Vertex data is LOCAL space (two-level BVH). Shade in the hit
            // instance's local frame — dot(nLocal, lrd) ≡ dot(nWorld, rd) and
            // barycentrics are affine-invariant, so every test below is exact —
            // then transform the FINAL normals to world through (M⁻¹)ᵀ.
            const Lray = instLocalRay(instId, ro, rd);
            const lro = Lray.ro.toVar();
            const lrd = Lray.rd.toVar();
            const vi0 = triVert(triId, 0);
            const vi1 = triVert(triId, 1);
            const vi2 = triVert(triId, 2);
            const p0 = fetchVert(vi0);
            const p1 = fetchVert(vi1);
            const p2 = fetchVert(vi2);
            const ngLocalRaw = normalize(cross(p1.sub(p0), p2.sub(p0)));
            const entering = dot(ngLocalRaw, lrd).lessThan(float(0));         // front-face hit
            const ngLocal = ngLocalRaw.mul(select(entering, float(1), float(-1))); // geometric, faces ray

            // Smooth shading normal: re-derive the hit barycentrics (Möller-
            // Trumbore, LOCAL ray) and interpolate the per-vertex normals. Falls
            // back to the geometric normal when no vertex normals were synced.
            const e1n = p1.sub(p0);
            const e2n = p2.sub(p0);
            const pvn = cross(lrd, e2n);
            const invDetN = float(1).div(dot(e1n, pvn));
            const tvn = lro.sub(p0);
            const bU = dot(tvn, pvn).mul(invDetN);
            const bV = dot(lrd, cross(tvn, e1n)).mul(invDetN);
            const bW = float(1).sub(bU).sub(bV);
            const nInterp = bW.mul(fetchNorm(vi0)).add(bU.mul(fetchNorm(vi1))).add(bV.mul(fetchNorm(vi2)));
            const nLen = length(nInterp);
            const Ns = select(nLen.greaterThan(float(0.01)), nInterp.div(max(nLen, float(1e-6))), ngLocal).toVar();

            // Interpolated + transformed hit UV — shared by every map sample.
            const uvHit = fetchUV(vi0).mul(bW).add(fetchUV(vi1).mul(bU)).add(fetchUV(vi2).mul(bV));
            const uvRep = vec2(matFloat(matId, 18), matFloat(matId, 19));
            const uvOff = vec2(matFloat(matId, 20), matFloat(matId, 21));
            const uv = uvHit.mul(uvRep).add(uvOff).toVar();

            // Normal map: perturb Ns in a UV-derived tangent frame (applied
            // before the face-flip so back-faces are corrected post-perturbation).
            if (haveNormalMap) {
                const nmL = matFloat(matId, 13);
                If(nmL.greaterThan(float(-0.5)), () => {
                    const duv1 = fetchUV(vi1).sub(fetchUV(vi0));
                    const duv2 = fetchUV(vi2).sub(fetchUV(vi0));
                    const denom = duv1.x.mul(duv2.y).sub(duv2.x.mul(duv1.y));
                    If(abs(denom).greaterThan(float(1e-10)), () => {
                        const r = float(1).div(denom);
                        const tRaw = e1n.mul(duv2.y).sub(e2n.mul(duv1.y)).mul(r);
                        // Gram-Schmidt T against Ns; bitangent from the cross.
                        const T = normalize(tRaw.sub(Ns.mul(dot(Ns, tRaw))));
                        const B = cross(Ns, T);
                        const ns = matFloat(matId, 17);
                        const tn = sampleLayer(normalTex, uv, nmL).mul(2).sub(1); // [0,1]→[-1,1]
                        const perturbed = T.mul(tn.x.mul(ns)).add(B.mul(tn.y.mul(ns))).add(Ns.mul(tn.z));
                        Ns.assign(normalize(perturbed));
                    });
                });
            }
            // keep the shading normal on the geometric side that faces the ray
            Ns.assign(Ns.mul(select(dot(Ns, ngLocal).lessThan(float(0)), float(-1), float(1))));
            // local frame done — everything below shades in WORLD space
            const ng = instNormalToWorld(instId, ngLocal);
            Ns.assign(instNormalToWorld(instId, Ns));

            const hitPoint = ro.add(rd.mul(bestT));
            const hitPos = hitPoint.add(ng.mul(float(RAY_EPS)));        // +ng offset for NEE

            // material fields (scalar PBR × optional maps sampled at the hit UV)
            const baseColor = vec3(matFloat(matId, 0), matFloat(matId, 1), matFloat(matId, 2)).toVar();
            const roughness = matFloat(matId, 3).toVar();
            const metalness = matFloat(matId, 4).toVar();
            const transmission = matFloat(matId, 5);
            const ior = matFloat(matId, 6);
            const emissive = vec3(matFloat(matId, 7), matFloat(matId, 8), matFloat(matId, 9)).toVar();
            const dispersionB = matFloat(matId, 11);

            // map mul-semantics mirror three (diffuse*=map, rough*=g, metal*=b,
            // emissive*=map); −1 layer → multiply by 1 (no-op).
            if (haveAlbedoMap) {
                const aL = matFloat(matId, 12);
                const s = srgbToLinear(sampleLayer(albedoTex, uv, aL));
                baseColor.assign(baseColor.mul(select(aL.greaterThan(float(-0.5)), s, vec3(1))));
            }
            if (haveRoughMap) {
                const rL = matFloat(matId, 14);
                const s = sampleLayer(roughTex, uv, rL).y; // green channel (glTF metallic-roughness)
                roughness.assign(roughness.mul(select(rL.greaterThan(float(-0.5)), s, float(1))));
            }
            if (haveMetalMap) {
                const mL = matFloat(matId, 15);
                const s = sampleLayer(metalTex, uv, mL).z; // blue channel (glTF metallic-roughness)
                metalness.assign(metalness.mul(select(mL.greaterThan(float(-0.5)), s, float(1))));
            }
            if (haveEmissiveMap) {
                const eL = matFloat(matId, 16);
                const s = srgbToLinear(sampleLayer(emissiveTex, uv, eL));
                emissive.assign(emissive.mul(select(eL.greaterThan(float(-0.5)), s, vec3(1))));
            }
            roughness.assign(clamp(roughness, float(0.02), float(1)));
            metalness.assign(clamp(metalness, float(0), float(1)));

            const isGlass = transmission.greaterThan(float(0.5));
            const notGlass = select(isGlass, float(0), float(1));

            // emissive contribution at λ
            radiance.addAssign(clampGI(throughput.mul(jhEmission(emissive, lambda))));

            // material slot [25]: authored NIR albedo (−1 = untagged → JH
            // extrapolation prior). Blended in above 700 nm; visible-mode
            // output is effectively untouched.
            const nirAlbedo = matFloat(matId, 25);
            const albedoL = jhReflectance(baseColor, lambda, nirAlbedo);

            // NEE: one light sample (diffuse only — fast). Mirrors the raster
            // punctual-light model (three getDistanceAttenuation +
            // getSpotAttenuation) so spot cones, decay and range agree with the
            // viewport. Reference: web/js/max_lights_node.js Masked*LightDataNode.
            If(U.lightCount.greaterThan(uint(0)), () => {
                const li = uint(min(float(U.lightCount).sub(1), floor(nextRand().mul(float(U.lightCount)))));
                const lb = li.mul(uint(17));
                const ltype = lights.element(lb);
                const lpos = vec3(lights.element(lb.add(uint(1))), lights.element(lb.add(uint(2))), lights.element(lb.add(uint(3))));
                const ldir = vec3(lights.element(lb.add(uint(4))), lights.element(lb.add(uint(5))), lights.element(lb.add(uint(6))));
                const lcol = vec3(lights.element(lb.add(uint(7))), lights.element(lb.add(uint(8))), lights.element(lb.add(uint(9))));
                const lrange = lights.element(lb.add(uint(10)));
                const ldecay = lights.element(lb.add(uint(11)));
                const lcosAngle = lights.element(lb.add(uint(12)));
                const lcosPen = lights.element(lb.add(uint(13)));
                // [16] emitter class (packed; ≥500 = incandescent colour temp)
                const leclass = lights.element(lb.add(uint(16)));

                // type: 0 directional, 1 point, 2 spot. ldir is the beam forward
                // (normalize(target-pos)); three's spotDirection is its negation.
                const isDir = ltype.lessThan(float(0.5));
                const isSpot = abs(ltype.sub(float(2))).lessThan(float(0.5));
                const toLight = select(isDir, ldir.mul(-1), lpos.sub(hitPos));
                const dist = select(isDir, float(1e4), max(length(toLight), float(1e-4)));
                const wi = normalize(toLight);
                const ndl = max(dot(Ns, wi), float(0));
                If(ndl.greaterThan(float(0)), () => {
                    const blocked = traverseAny(hitPos, wi, dist.sub(float(RAY_EPS)));
                    If(blocked.lessThan(float(0.5)), () => {
                        // distance attenuation — three getDistanceAttenuation:
                        // 1/max(dist^decay, 0.01), windowed by (1-(dist/range)^4)^2
                        // when range>0. Directional lights take no distance falloff.
                        const falloff = float(1).div(max(pow(dist, ldecay), float(0.01)));
                        const rr = dist.div(max(lrange, float(1e-4)));
                        const rr2 = rr.mul(rr);
                        const win = clamp(float(1).sub(rr2.mul(rr2)), float(0), float(1));
                        const ranged = falloff.mul(win.mul(win));
                        const posAtten = select(lrange.greaterThan(float(0)), ranged, falloff);
                        const distAtten = select(isDir, float(1), posAtten);
                        // spot cone — three getSpotAttenuation:
                        // smoothstep(coneCos, penumbraCos, dot(lightDir, spotDir)).
                        // angleCos = dot(wi, spotDir) = -dot(wi, ldir).
                        const angleCos = dot(wi, ldir).mul(-1);
                        const spotAtten = smoothstep(lcosAngle, lcosPen, angleCos);
                        const atten = distAtten.mul(select(isSpot, spotAtten, float(1)));
                        const lrad = emitterAtLambda(lcol, lambda, leclass).mul(atten);
                        const diffuse = albedoL.mul(float(1).sub(metalness)).mul(1.0 / PI);
                        // glass is specular — skip the diffuse direct-light term for it
                        radiance.addAssign(clampGI(throughput.mul(diffuse).mul(ndl).mul(lrad).mul(float(U.lightCount)).mul(notGlass)));
                    });
                });
            });

            // Russian roulette after rrStart bounces
            If(i.greaterThanEqual(U.rrStart), () => {
                const pSurv = clamp(throughput, float(0.05), float(1));
                If(nextRand().greaterThan(pSurv), () => { Break(); });
                throughput.assign(throughput.div(pSurv));
            });

            // ── opaque BSDF: metal = rough mirror, else cosine diffuse ──
            const doMetal = nextRand().lessThan(metalness);
            const diffuseDir = cosineSample(Ns, nextRand(), nextRand());
            const mirror = reflect(rd, Ns);
            const glossy = normalize(mirror.add(cosineSample(Ns, nextRand(), nextRand()).sub(Ns).mul(roughness)));
            const opaqueDir = select(doMetal, glossy, diffuseDir);

            // ── dielectric (glass) BSDF with WAVELENGTH-DEPENDENT IOR ──
            // n(λ): shorter wavelengths refract more (normal dispersion), so a
            // prism fans white light into a spectrum. This is the path's
            // wavelength driving GEOMETRY (Snell), not just the final color.
            // The linear fit extrapolates fine to 900 nm (NV mode) — real
            // dispersion flattens out in NIR, so linear slightly overshoots,
            // imperceptibly through an intensifier. No change needed.
            const nLambda = ior.add(dispersionB.mul(float(550).sub(lambda)).div(float(170)));
            // Reflect/refract against the smooth shading normal Ns (same as the
            // opaque lobe) so curved glass bends light smoothly instead of per
            // triangle facet. entering/eta stay on the geometric normal — which
            // medium we're crossing into is topological, and Ns can lie about it.
            const eta = select(entering, float(1).div(nLambda), nLambda);   // n_in / n_out
            const cosI = clamp(dot(rd, Ns).mul(float(-1)), float(0), float(1));
            const sin2T = eta.mul(eta).mul(float(1).sub(cosI.mul(cosI)));
            const tir = sin2T.greaterThan(float(1));                        // total internal reflection
            const cosT = sqrt(max(float(0), float(1).sub(sin2T)));
            const r0 = nLambda.sub(float(1)).div(nLambda.add(float(1)));
            const R0 = r0.mul(r0);
            const fcos = select(entering, cosI, cosT);                     // cosine in the denser medium
            const om = float(1).sub(fcos);
            const fres = R0.add(float(1).sub(R0).mul(om.mul(om).mul(om).mul(om).mul(om)));
            const refractDir = normalize(rd.mul(eta).add(Ns.mul(eta.mul(cosI).sub(cosT))));
            const reflectDir = reflect(rd, Ns);
            const doReflect = tir.or(nextRand().lessThan(fres));           // Fresnel importance sample
            const glassDir = select(doReflect, reflectDir, refractDir);

            // pick lobe; glass is clear (weight 1, Fresnel already sampled)
            const newDir = select(isGlass, glassDir, opaqueDir);
            const throughputMul = select(isGlass, float(1), albedoL);
            // GI Clamp now bounds radiance contributions (see clampGI above);
            // throughput just carries the path weight (≤1 for physical BSDFs).
            throughput.assign(throughput.mul(throughputMul));

            // offset the next ray's origin onto whichever side it actually leaves
            const sideN = ng.mul(select(dot(newDir, ng).greaterThan(float(0)), float(1), float(-1)));
            ro.assign(hitPoint.add(sideN.mul(float(RAY_EPS))));
            rd.assign(normalize(newDir));
        });

        const o = pix.mul(uint(4));
        if (isNV) {
            // NV mode: true photocathode electron flux in the 4th channel.
            // Estimator: radiance · S(λ) / (pdf(λ) · ∫S dλ). With importance
            // sampling pdf(λ) = S(λ)/∫S, so the weight collapses to exactly 1
            // — treating the inverse-CDF draw as exact, standard practice.
            // With uniform sampling (A/B validation) pdf = 1/range instead.
            // XYZ is skipped entirely: photocathode-weighted λ barely lands
            // below 720 where x̄ȳz̄ are meaningful, and the ALU is wasted.
            const fluxW = nvImportance
                ? float(1.0)
                : photocathodeS(lambda).mul(lambdaRange / pcSampler.integral);
            accum.element(o.add(uint(3))).addAssign(radiance.mul(fluxW));
        } else {
            // hero λ radiance → XYZ. ×LAMBDA_RANGE undoes the uniform pdf (1/range);
            // ÷CIE_Y_INTEGRAL normalizes luminance so unit radiance → Y≈1.
            const w = LAMBDA_RANGE / CIE_Y_INTEGRAL;
            accum.element(o).addAssign(cieX(lambda).mul(radiance).mul(w));
            accum.element(o.add(uint(1))).addAssign(cieY(lambda).mul(radiance).mul(w));
            accum.element(o.add(uint(2))).addAssign(cieZ(lambda).mul(radiance).mul(w));
        }
    })().compute(width * height);

    // clear kernel (reset accumulation)
    const clearKernel = Fn(() => {
        const pix = instanceIndex.toVar();
        const total = uint(width * height);
        If(pix.greaterThanEqual(total), () => { Return(); });
        const o = pix.mul(uint(4));
        accum.element(o).assign(float(0));
        accum.element(o.add(uint(1))).assign(float(0));
        accum.element(o.add(uint(2))).assign(float(0));
        accum.element(o.add(uint(3))).assign(float(0));
    })().compute(width * height);

    // ── blit material: XYZ accum → sRGB ────────────────────────────
    const blitMaterial = new THREE.MeshBasicNodeMaterial();
    blitMaterial.depthTest = false;
    blitMaterial.depthWrite = false;
    // The blit owns its tone map (neutralToneMapping when toneMapEnabled, else
    // linear pass-through for an external post stack). The renderer must not
    // apply its own on top — true would double-tone-map the direct path and
    // corrupt the linear handoff into the post pipeline.
    blitMaterial.toneMapped = false;
    blitMaterial.colorNode = Fn(() => {
        const sc = TSL.screenCoordinate;
        const ix = uint(sc.x);
        const iy = uint(sc.y);
        const idx = iy.mul(uint(width)).add(ix);
        const o = idx.mul(uint(4));
        const inv = float(1).div(max(U.sampleCount, float(1)));
        if (isNV) {
            // LINEAR electron flux for the intensifier stage: sample-count
            // divide + exposure only. NO tone map, NO sRGB — the tube model
            // (powershot_infrared.js, inputMode "nir") owns the transfer
            // curve, and it reads the red channel raw.
            const flux = accum.element(o.add(uint(3))).mul(inv).mul(U.exposure).max(0.0);
            return vec4(flux, flux, flux, 1);
        }
        const xyz = vec3(accum.element(o), accum.element(o.add(uint(1))), accum.element(o.add(uint(2))));
        const c = xyz.mul(inv);
        // XYZ (D65) → linear sRGB
        const rgb = vec3(
            c.x.mul(3.2406).sub(c.y.mul(1.5372)).sub(c.z.mul(0.4986)),
            c.x.mul(-0.9689).add(c.y.mul(1.8758)).add(c.z.mul(0.0415)),
            c.x.mul(0.0557).sub(c.y.mul(0.2040)).add(c.z.mul(1.0570)),
        );
        // neutralToneMapping is Fn([color, exposure]) — BOTH inputs are
        // required; omitting exposure makes it multiply by an undefined node
        // and the whole blit resolves to NaN (black screen).
        const lin = max(rgb, vec3(0));
        // Direct-to-canvas: exposure + neutral tone map (display-referred).
        // External post stack: emit LINEAR HDR (exposure only) and let the
        // stack's bloom/grade run in linear before it tone-maps at output.
        const tone = TSL.neutralToneMapping(lin, U.exposure);
        const out = select(U.toneMapEnabled.equal(uint(1)), tone, lin.mul(U.exposure));
        return vec4(out, 1);
    })();

    return { traceKernel, clearKernel, blitMaterial, uniforms: U };
}

export { LAMBDA_MIN, LAMBDA_MAX };
