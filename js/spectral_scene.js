// spectral_scene.js — CPU build pipeline for the WebGPU spectral path tracer.
//
// TWO-LEVEL BVH: one LOCAL-space BLAS per unique geometry (deduped, so
// shared/instanced geometry is built once) pooled into a single stackless
// threaded node buffer, plus an instance table + TLAS over instance
// world-AABBs packed into the tail of the materials buffer. Materials are
// deduped into a flat uber table. Moving objects ride
// built.updateTransforms() — an in-place instance/TLAS rewrite that needs no
// soup rewrite, no MeshBVH rebuild, and no shader recompile. DEFORMING
// objects (streamed vertex buffers, morphs, CPU skinning — same topology)
// ride built.updateDeforms() — re-gather the deformed BLAS's pooled vertex
// slice + in-place bounds refit (gi_refit.js) + the same TLAS rewrite, so a
// vertex-animated mesh never forces the synchronous MeshBVH rebuild.
//
// Output is plain typed arrays + counts; spectral_kernel turns them into
// StorageBufferAttributes. Nothing here touches the GPU.
//
// Materials are read as scalar PBR fields off the live material (works for
// every Standard/Physical/Lambert/Phong/Basic node material; TSL/MaterialX
// graphs fall back to their scalar values, same limitation as the old WebGL
// tracer). PBR *maps* (albedo/normal/roughness/metalness/emissive) are also
// extracted to GPU array textures and sampled at the hit UV — see
// buildMaterialTextures below.

import { MeshBVH } from 'three-mesh-bvh';
import {
    createFlatBlasRefitStepper,
    createFlatTlasRefitIndex,
    refitFlatBlasRange,
    refitFlatTlasDirty,
    refitFlatTlasRange,
    runLatestBudgetedTask,
} from './gi_refit.js';

const NODE_STRIDE_U32 = 8;     // bvhNodes: 6 aabb floats + miss + payload
// materials stride (floats). Layout:
//   [0..2] baseColor RGB (linear) · [3] roughness · [4] metalness
//   [5] transmission · [6] ior · [7..9] emissive RGB (linear×intensity)
//   [10] opacity · [11] dispersionB
//   [12] albedo-map layer · [13] normal-map layer · [14] roughness-map layer
//   [15] metalness-map layer · [16] emissive-map layer   (all −1 = none)
//   [17] normalScale · [18..19] uv repeat xy · [20..21] uv offset xy
//   [22] side (0 front, 1 back, 2 double) · [23] alphaTest
//   [24] alpha-map layer
//   [25] nirAlbedo: −1 = untagged (kernel uses JH extrapolation as prior),
//        else authored [0,1] NIR reflectance (userData.nirAlbedo wins over
//        the classifyNir heuristics) · [26] traversal flags · [27] pad
const MAT_STRIDE = 28;
const MAT_FLAG_ALPHA_TEXTURE = 1; // alpha acceptance requires UV/map sampling
const MAT_FLAG_ALPHA_ACCEPT = 2;  // constant alpha path is guaranteed accepted
const MAT_FLAG_SHADOW_BLOCK = 4;  // transmission < 0.5 (matches traverseAny)
const LIGHT_STRIDE = 17;        // lights: see layout below
const VERT_STRIDE = 3;          // vertexPos: tightly packed xyz (flat f32 storage, read by offset; also fed to MeshBVH which assumes stride 3)
const VERTEX_DATA_STRIDE = 8;   // GPU interleaved per-vertex: pos(3) + normal(3) + uv(2)
const BYTES_PER_BVH_NODE = 32;  // three-mesh-bvh BYTES_PER_NODE (8 x u32)
const TEXTURE_ATLAS_SIZE = 256; // every material map is resampled to this square size and stacked into a DataArrayTexture layer
const SKIP_TRIANGLE_MATERIAL = 0xFFFFFFFF;
// Keep only genuinely tiny edits synchronous. A few thousand gathered/refit
// items stay below one frame slice even on slower hosts; anything larger
// enters the budgeted lane before it can turn into a scrub hitch.
const ASYNC_DEFORM_WORK_THRESHOLD = 4096;
// Target 2 ms internally so clock batching, JIT/GC jitter, and the task handoff
// still fit under the public 3 ms main-thread slice ceiling in real scenes.
const ASYNC_DEFORM_MAX_BUDGET_MS = 2;

const SKY_NAMES = new Set(['__maxjs_sky__']);

function meshMaterials(mesh) {
    if (!mesh?.material) return [];
    return Array.isArray(mesh.material) ? mesh.material : [mesh.material];
}

function materialIsRenderable(material) {
    return !!material
        && material.visible !== false
        && (!Number.isFinite(material.opacity) || material.opacity > 1.0e-4);
}

function objectLayersMatchCamera(object, camera) {
    if (!camera?.layers?.test || !object?.layers) return true;
    return object.layers.test(camera.layers);
}

function objectIsRenderable(object, camera) {
    return object?.visible !== false
        && object.userData?.maxjsVisible !== false
        && objectLayersMatchCamera(object, camera);
}

function isTraceableGeometry(geometry) {
    const pos = geometry?.attributes?.position;
    return !!(pos && pos.count >= 3 && pos.itemSize >= 3 && pos.array);
}

function isTraceableMesh(object, camera) {
    if (!object?.isMesh && !object?.isInstancedMesh) return false;
    if (object.isSkinnedMesh) return false; // bind-pose only; skip for now
    if (SKY_NAMES.has(object.name)) return false;
    if (!objectIsRenderable(object, camera)) return false;
    if (!isTraceableGeometry(object.geometry)) return false;
    return meshMaterials(object).some(materialIsRenderable);
}

function buildTriangleMaterialMap(geometry, materials, triCount, internMaterial, materialIsArray = false) {
    const triMat = new Uint32Array(triCount);
    triMat.fill(SKIP_TRIANGLE_MATERIAL);

    const groups = Array.isArray(geometry?.groups) ? geometry.groups : [];
    if (materialIsArray) {
        for (const grp of groups) {
            const slot = grp.materialIndex == null ? 0 : Math.trunc(Number(grp.materialIndex));
            if (!Number.isFinite(slot) || slot < 0 || slot >= materials.length) continue;
            const material = materials[slot] || null;
            const uber = materialIsRenderable(material)
                ? internMaterial(material)
                : SKIP_TRIANGLE_MATERIAL;
            const start = Math.max(0, Math.trunc(Number(grp.start) || 0));
            const count = Math.max(0, Math.trunc(Number(grp.count) || 0));
            const t0 = Math.floor(start / 3);
            const t1 = Math.min(triCount, t0 + Math.floor(count / 3));
            for (let t = t0; t < t1; t++) triMat[t] = uber;
        }
    } else {
        const baseMaterial = materials[0] || null;
        const baseUber = materialIsRenderable(baseMaterial)
            ? internMaterial(baseMaterial)
            : SKIP_TRIANGLE_MATERIAL;
        triMat.fill(baseUber);
    }

    let visibleTriCount = 0;
    for (let t = 0; t < triCount; t++) {
        if (triMat[t] !== SKIP_TRIANGLE_MATERIAL) visibleTriCount++;
    }
    if (visibleTriCount === triCount) {
        return { triMat, visibleTriCount, visibleTriIndices: null };
    }
    if (visibleTriCount === 0) {
        return { triMat, visibleTriCount, visibleTriIndices: new Uint32Array(0) };
    }
    const visibleTriIndices = new Uint32Array(visibleTriCount);
    let cursor = 0;
    for (let t = 0; t < triCount; t++) {
        if (triMat[t] !== SKIP_TRIANGLE_MATERIAL) visibleTriIndices[cursor++] = t;
    }
    return { triMat, visibleTriCount, visibleTriIndices };
}

// ── Uber material mapping ──────────────────────────────────────────
// One flat record maps every material model to: baseColor, roughness,
// metalness, transmission, ior, emissive*intensity, opacity, dispersionB.
// Reads linear-space color components directly (Three stores working-space).

// userData.giColor / userData.giEmissive hint → [r,g,b] linear, or null. Accepts a
// THREE.Color (working-space components used as-is) or a 0xRRGGBB hex (sRGB → linear).
function giColorHint(v) {
    if (v == null) return null;
    if (v.isColor) return [v.r, v.g, v.b];
    if (Number.isFinite(v)) {
        const s = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
        return [s((v >> 16) & 255), s((v >> 8) & 255), s(v & 255)];
    }
    return null;
}

function emissiveScaled(material, out) {
    // Node-driven emissive (emissiveNode) is invisible to the packer; an explicit
    // userData.giEmissive hint (absolute linear energy, NOT scaled by
    // emissiveIntensity) is the only way to feed it to the trace.
    const hint = giColorHint(material.userData?.giEmissive);
    if (hint) { out[0] = hint[0]; out[1] = hint[1]; out[2] = hint[2]; return out; }
    const e = material.emissive;
    const k = Number.isFinite(material.emissiveIntensity) ? material.emissiveIntensity : 1;
    if (e?.isColor) { out[0] = e.r * k; out[1] = e.g * k; out[2] = e.b * k; }
    else { out[0] = 0; out[1] = 0; out[2] = 0; }
    return out;
}

// ── NIR albedo authoring ───────────────────────────────────────────
// RGB carries zero information about NIR (metamerism): two identical greens
// can differ 10× at 850 nm (chlorophyll red edge vs green paint). Truth is
// injected as DATA — one scalar NIR reflectance per material — with cheap
// name/colour heuristics providing defaults for the big classes. −1 leaves
// the material on the JH-extrapolation prior. Colour inputs are LINEAR.
export function classifyNir(name, r, g, b, roughness, metalness, transmission) {
    const n = String(name || '');
    const maxc = Math.max(r, g, b);
    const minc = Math.min(r, g, b);
    const sat = maxc > 1e-6 ? (maxc - minc) / maxc : 0;
    // chlorophyll red edge: foliage is DARK green visibly, ~0.5 in NIR
    if (/leaf|foliage|grass|tree|plant|veg/i.test(n)) return 0.55;
    if (g > 1.15 * Math.max(r, b) && roughness > 0.5 && metalness < 0.2) return 0.55;
    // water absorbs strongly past ~750 nm → near-black through a tube
    if (/water|ocean|pool/i.test(n)) return 0.04;
    if (transmission > 0.5 && b > r) return 0.04;
    // asphalt / very dark low-saturation dielectric
    if (/asphalt|tarmac|road/i.test(n)) return 0.06;
    if (maxc < 0.08 && sat < 0.15 && metalness < 0.2 && transmission < 0.5) return 0.06;
    // skin: waxy NIR lift (sub-surface scattering)
    const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    if (r > g && g > b && b > 1e-6 && r / b >= 1.3 && r / b <= 2.5
        && lum > 0.12 && lum < 0.7 && metalness < 0.2) return 0.62;
    return -1; // untagged → JH extrapolation prior
}

function materialToUber(material) {
    const color = material.color;
    let r = color?.isColor ? color.r : 1;
    let g = color?.isColor ? color.g : 1;
    let b = color?.isColor ? color.b : 1;
    // Node-driven materials (colorNode) render via TSL, so the scalar color the packer
    // reads is usually the untouched DEFAULT WHITE — the trace then bounces at the
    // 0.95-albedo cap while the screen shows dark stone (a closed street canyon turns
    // into an integrating sphere: broad energy wash on unlit surfaces). Honor a
    // userData.giColor hint first; otherwise a colorNode material whose scalar is
    // still pure white falls back to mid-grey. A deliberately-set scalar is kept.
    const hint = giColorHint(material.userData?.giColor);
    if (hint) { [r, g, b] = hint; }
    else if (material.colorNode && r === 1 && g === 1 && b === 1) { r = g = b = 0.5; }

    // Roughness/metalness exist on Standard/Physical; derive sane values for
    // the rest. Phong shininess → roughness; Basic/Lambert → fully diffuse.
    let roughness = Number.isFinite(material.roughness) ? material.roughness : null;
    let metalness = Number.isFinite(material.metalness) ? material.metalness : 0;
    if (roughness == null) {
        if (Number.isFinite(material.shininess)) {
            const s = Math.max(0, material.shininess);
            roughness = Math.sqrt(2 / (s + 2)); // Phong exponent → roughness
        } else {
            roughness = 1;
        }
    }
    const transmission = Number.isFinite(material.transmission) ? material.transmission : 0;
    const ior = Number.isFinite(material.ior) ? material.ior : 1.5;
    const opacity = Number.isFinite(material.opacity) ? material.opacity : 1;
    const alphaTest = Number.isFinite(material.alphaTest) ? material.alphaTest : 0;
    const em = emissiveScaled(material, [0, 0, 0]);
    // Dispersion strength: the kernel reads this as the per-wavelength IOR
    // spread n(λ) = ior ± dispersionB across the visible band. three's
    // MeshPhysicalMaterial.dispersion drives it (0 = none).
    const dispersionB = Number.isFinite(material.dispersion) ? material.dispersion : 0;

    // UV transform + normalScale: read from the primary present map (all maps in
    // a material almost always share one transform). Fast-not-accurate: a single
    // repeat/offset is applied to every map; per-map transforms and rotation are
    // intentionally ignored.
    const primary = material.map || material.normalMap || material.roughnessMap
        || material.metalnessMap || material.emissiveMap || material.alphaMap || null;
    const rep = primary?.repeat;
    const off = primary?.offset;
    const repX = rep && Number.isFinite(rep.x) ? rep.x : 1;
    const repY = rep && Number.isFinite(rep.y) ? rep.y : 1;
    const offX = off && Number.isFinite(off.x) ? off.x : 0;
    const offY = off && Number.isFinite(off.y) ? off.y : 0;
    const ns = material.normalScale;
    const normalScale = ns && Number.isFinite(ns.x) ? ns.x : 1;

    const side = Number.isFinite(material.side)
        ? Math.min(2, Math.max(0, material.side | 0))
        : 0;

    // NIR reflectance (slot [25]): explicit userData.nirAlbedo wins, else the
    // name/colour classifier, else −1 (JH prior). Clamped roughness/metalness
    // values match what the classifier expects.
    const udNir = material.userData?.nirAlbedo;
    const roughnessC = Math.min(1, Math.max(0, roughness)); // 0 = delta mirror, legal
    const metalnessC = Math.min(1, Math.max(0, metalness));
    const transmissionC = Math.min(1, Math.max(0, transmission));
    const nirAlbedo = Number.isFinite(udNir)
        ? Math.min(1, Math.max(0, udNir))
        : classifyNir(material.name, r, g, b, roughnessC, metalnessC, transmissionC);

    // Traversal classification is packed once instead of rediscovered for
    // every candidate triangle. A bound color/alpha map stays on the exact
    // texture path; otherwise scalar alpha acceptance is a build-time fact.
    const hasAlphaTexture = !!(material.map || material.alphaMap);
    const constantAlphaAccepted = opacity > 1.0e-4
        && (alphaTest <= 0 || opacity >= alphaTest);
    const traversalFlags = (hasAlphaTexture ? MAT_FLAG_ALPHA_TEXTURE : 0)
        | (!hasAlphaTexture && constantAlphaAccepted ? MAT_FLAG_ALPHA_ACCEPT : 0)
        | (transmissionC < 0.5 ? MAT_FLAG_SHADOW_BLOCK : 0);

    return [r, g, b,
        Math.min(1, Math.max(0, roughness)), // 0 = delta mirror, legal
        Math.min(1, Math.max(0, metalness)),
        Math.min(1, Math.max(0, transmission)),
        Math.max(1, ior),
        em[0], em[1], em[2],
        Math.min(1, Math.max(0, opacity)),
        dispersionB,
        -1, -1, -1, -1, -1,   // [12..16] map layers, filled by buildMaterialTextures
        normalScale,
        repX, repY, offX, offY,
        side,
        Math.min(1, Math.max(0, alphaTest)),
        -1,                     // [24] alpha-map layer, filled by buildMaterialTextures
        nirAlbedo,              // [25] NIR reflectance (−1 = JH prior)
        traversalFlags, 0];     // [26] traversal flags, [27] pad
}

const texUuid = (t) => (t && t.isTexture ? t.uuid : '-');

// Materials sharing scalar params but differing in any bound map must NOT
// collapse to one uber index, else they'd share a texture layer. Fold the map
// identities into the dedup key.
function uberKey(rec, material) {
    let k = '';
    for (let i = 0; i < MAT_STRIDE; i++) k += Math.round(rec[i] * 1000) + ':';
    k += texUuid(material.map) + '|' + texUuid(material.normalMap) + '|'
        + texUuid(material.roughnessMap) + '|' + texUuid(material.metalnessMap)
        + '|' + texUuid(material.emissiveMap) + '|' + texUuid(material.alphaMap);
    return k;
}

// ── Material map extraction ────────────────────────────────────────
// Resample every bound PBR map to a fixed square and stack same-typed maps
// into one DataArrayTexture (one binding per map type, layer = material's
// assigned slot). RGBA8, NoColorSpace — the kernel decodes sRGB for
// colour maps itself. Textures are deduped by uuid within a type so a reused
// map costs one layer. Unreadable sources (compressed, no decoded image)
// leave the layer at −1 and the material keeps its scalar field.

function toDrawable(image) {
    if (!image) return null;
    if (typeof HTMLImageElement !== 'undefined' && image instanceof HTMLImageElement) {
        return (image.complete && image.naturalWidth > 0) ? image : null;
    }
    if (typeof HTMLCanvasElement !== 'undefined' && image instanceof HTMLCanvasElement) return image;
    if (typeof ImageBitmap !== 'undefined' && image instanceof ImageBitmap) return image;
    if (typeof OffscreenCanvas !== 'undefined' && image instanceof OffscreenCanvas) return image;
    if (typeof HTMLVideoElement !== 'undefined' && image instanceof HTMLVideoElement) {
        return image.readyState >= 2 ? image : null;
    }
    // DataTexture-style { data, width, height }: rebuild a canvas from raw RGBA8.
    const { data, width, height } = image;
    if (data && width > 0 && height > 0
        && (data instanceof Uint8Array || data instanceof Uint8ClampedArray)
        && data.length >= width * height * 4) {
        const c = document.createElement('canvas');
        c.width = width; c.height = height;
        const ctx = c.getContext('2d');
        const id = ctx.createImageData(width, height);
        id.data.set(data.subarray(0, width * height * 4));
        ctx.putImageData(id, 0, 0);
        return c;
    }
    return null;
}

function extractTextureRGBA(tex, size) {
    if (typeof document === 'undefined') return null;
    if (!tex || tex.isCompressedTexture) return null;
    const drawable = toDrawable(tex.image);
    if (!drawable) return null;
    const c = document.createElement('canvas');
    c.width = size; c.height = size;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    if (!ctx) return null;
    // Bake the texture's flipY into the layer so a kernel sample at the mesh UV
    // lands on the same texel three's rasterizer would read (default image
    // textures use flipY = true). DataTextures default flipY = false.
    if (tex.flipY) { ctx.translate(0, size); ctx.scale(1, -1); }
    try { ctx.drawImage(drawable, 0, 0, size, size); }
    catch { return null; }
    let pixels;
    try { pixels = ctx.getImageData(0, 0, size, size).data; }
    catch { return null; } // tainted canvas (cross-origin without CORS)
    return new Uint8Array(pixels);
}

// Walk every uber material, extract each map type, build the DataArrayTextures,
// and write the assigned layer index back into each uber record. Returns
// { albedo, normal, roughness, metalness, emissive, alpha } (DataArrayTexture | null).
async function buildMaterialTextures(THREE, uberList, uberMaterials, size) {
    const TYPES = [
        { field: 'map', recIdx: 12, out: 'albedo' },
        { field: 'normalMap', recIdx: 13, out: 'normal' },
        { field: 'roughnessMap', recIdx: 14, out: 'roughness' },
        { field: 'metalnessMap', recIdx: 15, out: 'metalness' },
        { field: 'emissiveMap', recIdx: 16, out: 'emissive' },
        { field: 'alphaMap', recIdx: 24, out: 'alpha' },
    ];
    const layerBytes = size * size * 4;
    const result = { albedo: null, normal: null, roughness: null, metalness: null, emissive: null, alpha: null };

    for (const ty of TYPES) {
        const layers = [];
        const byUuid = new Map();
        for (let i = 0; i < uberList.length; i++) {
            const tex = uberMaterials[i]?.[ty.field];
            if (!tex || !tex.isTexture) continue;
            let layer = byUuid.get(tex.uuid);
            if (layer === undefined) {
                // Time-slice: yield a frame before each unique-texture canvas
                // resample so the per-texture drawImage+getImageData cost is
                // spread across frames instead of one synchronous stall. Cheap
                // (already-extracted) repeats of the same uuid don't yield.
                if (typeof requestAnimationFrame === 'function') {
                    await new Promise((r) => requestAnimationFrame(r));
                }
                const data = extractTextureRGBA(tex, size);
                if (!data) continue; // unreadable → leave record layer at −1
                layer = layers.length;
                layers.push(data);
                byUuid.set(tex.uuid, layer);
            }
            uberList[i][ty.recIdx] = layer;
        }
        if (layers.length === 0) continue;
        const merged = new Uint8Array(layerBytes * layers.length);
        for (let l = 0; l < layers.length; l++) merged.set(layers[l], l * layerBytes);
        const arr = new THREE.DataArrayTexture(merged, size, size, layers.length);
        arr.format = THREE.RGBAFormat;
        arr.type = THREE.UnsignedByteType;
        arr.minFilter = THREE.LinearFilter;
        arr.magFilter = THREE.LinearFilter;
        arr.wrapS = THREE.RepeatWrapping;   // honour uv repeat > 1
        arr.wrapT = THREE.RepeatWrapping;
        arr.colorSpace = THREE.NoColorSpace; // raw bytes; kernel decodes sRGB
        arr.generateMipmaps = false;
        arr.needsUpdate = true;
        result[ty.out] = arr;
    }
    return result;
}

// ── Threaded (stackless) BVH re-flatten ────────────────────────────
// three-mesh-bvh 0.9.4+ node layout (BYTES_PER_NODE = 32, addressed in
// 32-bit words): bounds = float bits [n32+0..5]; leaf test
// u16[n32*2+15]===0xFFFF; leaf: triOffset=u32[n32+6] (tri units),
// triCount=u16[n32*2+14]; internal: left child = n32+8 (contiguous in
// source), right child = n32 + u32[n32+6]*8 (PARENT-RELATIVE offset in
// node units — 0.8.x stored an absolute u32 index in that word, so this
// decode REQUIRES >=0.9.4). We re-emit DFS so the LEFT child is contiguous
// (idx+1) in the output and every node gets a single escape/miss index =
// first node after its subtree.
//
// The decode reads MeshBVH._roots private bytes, so layout drift in a
// future upstream bump must fail LOUDLY here (a mis-walk otherwise renders
// as silent black): every visited offset is bounds-checked, and the walk
// must consume exactly the whole buffer and account for every triangle.

function flattenBVHRoot(rootBuffer, expectedTriCount) {
    const f32 = new Float32Array(rootBuffer);
    const u16 = new Uint16Array(rootBuffer);
    const u32 = new Uint32Array(rootBuffer);

    const records = []; // { bx[6], leaf, triOffset, triCount, miss }

    // Iterative DFS with an explicit worklist so deep/unbalanced trees can't
    // overflow the JS call stack. We need each node's miss index to equal the
    // output position immediately AFTER its whole subtree, so we record a
    // node, push a "finalize" marker, then push its children (right first so
    // left is processed next → contiguous idx+1).
    const work = [{ n32: 0, kind: 'visit' }];
    const nodeCapacity = u32.length >>> 3; // buffer holds exactly this many 32-byte nodes
    let flattenedTris = 0;

    while (work.length > 0) {
        const item = work.pop();
        if (item.kind === 'finalize') {
            records[item.idx].miss = records.length;
            continue;
        }
        const n32 = item.n32;
        if (n32 < 0 || (n32 & 7) !== 0 || (n32 >>> 3) >= nodeCapacity || records.length >= nodeCapacity) {
            throw new Error(`spectral_scene: BVH flatten walked outside the node buffer (n32=${n32}, visited=${records.length}, nodes=${nodeCapacity}) — three-mesh-bvh internal layout drifted; flattenBVHRoot expects the 0.9.4+ parent-relative child encoding.`);
        }
        const isLeaf = u16[n32 * 2 + 15] === 0xFFFF;
        const idx = records.length;
        const rec = {
            bx: [f32[n32], f32[n32 + 1], f32[n32 + 2], f32[n32 + 3], f32[n32 + 4], f32[n32 + 5]],
            leaf: isLeaf,
            triOffset: 0,
            triCount: 0,
            miss: 0,
        };
        records.push(rec);
        if (isLeaf) {
            rec.triOffset = u32[n32 + 6];
            rec.triCount = u16[n32 * 2 + 14];
            rec.miss = records.length; // escape = next slot
            flattenedTris += rec.triCount;
        } else {
            const leftN32 = n32 + 8;
            // 0.9.4+: parent-relative offset in NODE units (8 u32 words/node)
            const rightN32 = n32 + u32[n32 + 6] * 8;
            // finalize sets miss AFTER both children are flattened
            work.push({ kind: 'finalize', idx });
            work.push({ kind: 'visit', n32: rightN32 });
            work.push({ kind: 'visit', n32: leftN32 });
        }
    }

    if (records.length !== nodeCapacity) {
        throw new Error(`spectral_scene: BVH flatten visited ${records.length} nodes but the buffer holds ${nodeCapacity} — three-mesh-bvh internal layout drifted.`);
    }
    if (Number.isFinite(expectedTriCount) && flattenedTris !== expectedTriCount) {
        throw new Error(`spectral_scene: BVH flatten accounted for ${flattenedTris} triangles, expected ${expectedTriCount} — three-mesh-bvh internal layout drifted.`);
    }

    // Records are serialized at POOL-assembly time (buildSpectralScene) so a
    // per-geometry BLAS can be placed at any base offset: miss links and leaf
    // triOffsets get the pool bases added there.
    return records;
}

// ── Light extraction ───────────────────────────────────────────────
// Layout (stride 17 floats): [0] type(0 dir/1 point/2 spot/3 rect),
// [1..3] worldPos, [4..6] worldDir (toward target), [7..9] color*intensity,
// [10] range, [11] decay, [12] cosAngle, [13] cosPenumbra, [14] w, [15] h,
// [16] emitter class (packed): 0 untagged (JH emission), 2 LED,
//      3 sodium (LPS 589 nm), 4 IR illuminator (850 nm band); any value
//      ≥ 500 = incandescent/halogen with that colour temperature in Kelvin
//      (class 1 + userData.colorTemp ?? 2856 collapse to this at pack time).
// Source: light.userData.emitterClass — string ('incandescent', 'halogen',
// 'led', 'sodium', 'ir') or number 0..4. Untagged lights keep today's
// JH-emission behaviour exactly.

function emitterClassValue(obj) {
    const raw = obj.userData?.emitterClass;
    let cls = 0;
    if (typeof raw === 'string') {
        const s = raw.trim().toLowerCase();
        if (s.startsWith('incan') || s === 'halogen' || s === 'tungsten') cls = 1;
        else if (s === 'led') cls = 2;
        else if (s.startsWith('sodium') || s === 'lps') cls = 3;
        else if (s === 'ir' || s.startsWith('ir_') || s.startsWith('ir ') || s.includes('illuminator')) cls = 4;
    } else if (Number.isFinite(raw)) {
        cls = Math.min(4, Math.max(0, Math.trunc(raw)));
    }
    if (cls === 1) {
        const t = Number.isFinite(obj.userData?.colorTemp) ? obj.userData.colorTemp : 2856;
        return Math.min(20000, Math.max(500, t));
    }
    return cls;
}

export function collectLights(THREE, scene, camera = null) {
    const out = [];
    const pos = new THREE.Vector3();
    const dir = new THREE.Vector3();
    const tgt = new THREE.Vector3();
    scene.traverseVisible((obj) => {
        if (!obj.isLight) return;
        if (!objectIsRenderable(obj, camera)) return;
        if (obj.isAmbientLight || obj.isHemisphereLight) return; // folded into env
        obj.updateWorldMatrix(true, false);
        obj.getWorldPosition(pos);
        const c = obj.color, k = Number.isFinite(obj.intensity) ? obj.intensity : 1;
        if (k <= 0) return;
        const eclass = emitterClassValue(obj);
        let cr = (c?.isColor ? c.r : 1) * k;
        let cg = (c?.isColor ? c.g : 1) * k;
        let cb = (c?.isColor ? c.b : 1) * k;
        if (cr <= 0 && cg <= 0 && cb <= 0) {
            // An IR illuminator is legitimately BLACK in RGB — intensity alone
            // drives its 850 nm band (and it stays invisible in visible mode
            // because the band is far outside the sampled domain). Everything
            // else with a black colour genuinely emits nothing.
            // NOTE the promotion below is for the PT's emitterAtLambda (max
            // channel = the band's peak radiance). RGB-domain consumers (the GI
            // probes' NEE) must NOT shade this white as-is — they gate class-4
            // lights on the nirGate uniform (see gi_probes.js), so the light
            // stays dark in the visible band and appears under NV.
            if (eclass !== 4) return;
            cr = cg = cb = k;
        }
        let type = 1, range = 0, decay = 2, cosAngle = -1, cosPen = -1, w = 0, h = 0;
        dir.set(0, 0, -1);
        if (obj.isDirectionalLight || obj.isSpotLight) {
            type = obj.isSpotLight ? 2 : 0;
            obj.target?.updateWorldMatrix?.(true, false);
            obj.target?.getWorldPosition(tgt);
            dir.copy(tgt).sub(pos).normalize();
            if (obj.isSpotLight) {
                const angle = Number.isFinite(obj.angle) ? obj.angle : Math.PI / 4;
                const pen = Number.isFinite(obj.penumbra) ? obj.penumbra : 0;
                cosAngle = Math.cos(angle);
                cosPen = Math.cos(angle * (1 - pen));
                range = obj.distance || 0;
                decay = Number.isFinite(obj.decay) ? obj.decay : 2;
            }
        } else if (obj.isPointLight) {
            type = 1;
            range = obj.distance || 0;
            decay = Number.isFinite(obj.decay) ? obj.decay : 2;
        } else if (obj.isRectAreaLight) {
            // Phase 1: treat as a point light at its center (orientation deferred).
            type = 1;
            w = obj.width || 0; h = obj.height || 0;
        } else {
            return;
        }
        out.push([type, pos.x, pos.y, pos.z, dir.x, dir.y, dir.z, cr, cg, cb, range, decay, cosAngle, cosPen, w, h, eclass]);
    });
    return out;
}

// ── Main build ─────────────────────────────────────────────────────
// TWO-LEVEL layout:
//   • one BLAS per unique (geometry × per-tri material mapping), built in
//     LOCAL space and pooled into the classic bvhNodes/triIndex/vertexData/
//     triMaterial buffers (node encoding unchanged — miss links and leaf
//     triOffsets are pre-offset to pool-absolute indices at assembly),
//   • an instance table (inverse world 3×4 rows + winding sign + BLAS range)
//     and a TLAS over instance world-AABBs, both packed into the TAIL of the
//     materials float buffer so NO extra storage binding is needed,
//   • built.updateTransforms() — the moving-object fast path: recompute the
//     instance records + TLAS IN PLACE (identical array sizes) from the live
//     matrixWorlds. Microseconds of CPU; the caller re-uploads the (small)
//     materials buffer. No soup rewrite, no MeshBVH rebuild, no recompile.

// Local-space soup + BVH for ONE unique geometry/material mapping.
// vertexMaterial is stamped so the per-triangle material survives the BVH
// index permutation; material-array draws get a unique first vertex per
// visible triangle (preserves Multi/Sub assignment across shared vertices).
function buildLocalBlas(THREE, d) {
    const { pos, index, normal, uv } = d;
    const vCount = pos.count;
    const extra = d.uniqueTriMaterial ? d.visibleTriCount : 0;
    const totalV = vCount + extra;
    const vertexPos = new Float32Array(totalV * VERT_STRIDE);
    const vertexNormal = new Float32Array(totalV * 3); // LOCAL space, 0 = none → flat
    const vertexUV = new Float32Array(totalV * 2);
    const vertexMaterial = new Uint32Array(totalV);
    const triIndex = new Uint32Array(d.visibleTriCount * 3);
    // tag vertex t duplicates SOURCE vertex tagSrc[t] (the triangle's first
    // index) — kept so updateDeforms can re-copy the duplicates after a
    // deform gather. 0xFFFFFFFF = skipped triangle (never referenced).
    const tagSrc = d.uniqueTriMaterial ? new Uint32Array(d.visibleTriCount).fill(0xFFFFFFFF) : null;
    for (let i = 0; i < vCount; i++) {
        const o = i * VERT_STRIDE;
        vertexPos[o] = pos.getX(i); vertexPos[o + 1] = pos.getY(i); vertexPos[o + 2] = pos.getZ(i);
        if (normal) {
            const no = i * 3;
            vertexNormal[no] = normal.getX(i); vertexNormal[no + 1] = normal.getY(i); vertexNormal[no + 2] = normal.getZ(i);
        }
        if (uv) {
            const uo = i * 2;
            vertexUV[uo] = uv.getX(i); vertexUV[uo + 1] = uv.getY(i);
        }
    }
    const tagBase = vCount;
    for (let t = 0; t < d.visibleTriCount; t++) {
        const sourceTri = d.visibleTriIndices ? d.visibleTriIndices[t] : t;
        if (sourceTri < 0 || sourceTri >= d.triCount) continue;
        const sourceA = index ? index.getX(sourceTri * 3) : sourceTri * 3;
        if (tagSrc) tagSrc[t] = sourceA;
        const a = d.uniqueTriMaterial ? tagBase + t : sourceA;
        const b = index ? index.getX(sourceTri * 3 + 1) : sourceTri * 3 + 1;
        const c = index ? index.getX(sourceTri * 3 + 2) : sourceTri * 3 + 2;
        if (d.uniqueTriMaterial) {
            const po = a * VERT_STRIDE, ps = sourceA * VERT_STRIDE;
            vertexPos[po] = vertexPos[ps]; vertexPos[po + 1] = vertexPos[ps + 1]; vertexPos[po + 2] = vertexPos[ps + 2];
            const no = a * 3, ns = sourceA * 3;
            vertexNormal[no] = vertexNormal[ns]; vertexNormal[no + 1] = vertexNormal[ns + 1]; vertexNormal[no + 2] = vertexNormal[ns + 2];
            const uo = a * 2, us = sourceA * 2;
            vertexUV[uo] = vertexUV[us]; vertexUV[uo + 1] = vertexUV[us + 1];
        }
        const to = t * 3;
        triIndex[to] = a; triIndex[to + 1] = b; triIndex[to + 2] = c;
        const um = d.triMat[sourceTri];
        vertexMaterial[a] = um;
        if (!d.uniqueTriMaterial) { vertexMaterial[b] = um; vertexMaterial[c] = um; }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(vertexPos, VERT_STRIDE));
    geometry.setIndex(new THREE.BufferAttribute(triIndex, 1));
    geometry.clearGroups();
    const bvh = new MeshBVH(geometry, { targetLeafSize: 8, indirect: false });
    // MeshBVH writes geometry.boundingBox during the build (setBoundingBox
    // default) from the indexed triangles only — unlike computeBoundingBox,
    // which also scanned the zero-filled slots of skipped triangles and could
    // wrongly pull the bounds toward the origin.
    const localBounds = geometry.boundingBox ? geometry.boundingBox.clone() : new THREE.Box3();
    const roots = bvh._roots;
    geometry.dispose?.();
    if (!Array.isArray(roots) || roots.length === 0) return null;
    const records = flattenBVHRoot(roots[0], d.visibleTriCount);
    // Per-triangle material in BVH order (MeshBVH permuted triIndex in place;
    // the material rides the triangle's first vertex through the permutation).
    const triMaterial = new Uint32Array(d.visibleTriCount);
    for (let t = 0; t < d.visibleTriCount; t++) triMaterial[t] = vertexMaterial[triIndex[t * 3]] >>> 0;
    return { vertexPos, vertexNormal, vertexUV, triIndex, triMaterial, records, localBounds, vertCount: totalV, triCount: d.visibleTriCount, srcVertCount: vCount, tagSrc };
}

// Threaded TLAS over instance world-AABBs. Median split by index midpoint on
// the longest axis. The initial partition and `order` are frozen after build;
// live transforms only rewrite instance rows and refit exact bounds bottom-up.
// Leaves reference PERMUTED instance slots [off, off+cnt); `order` maps
// slot → source instance index.
function buildTlasRecords(aabbs, leafSize = 2) {
    const order = aabbs.map((_, i) => i);
    const records = [];
    function emit(lo, hi) {
        const rec = { bx: [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity], leaf: false, off: 0, cnt: 0, miss: 0 };
        records.push(rec);
        for (let i = lo; i < hi; i++) {
            const a = aabbs[order[i]];
            if (a.min[0] < rec.bx[0]) rec.bx[0] = a.min[0];
            if (a.min[1] < rec.bx[1]) rec.bx[1] = a.min[1];
            if (a.min[2] < rec.bx[2]) rec.bx[2] = a.min[2];
            if (a.max[0] > rec.bx[3]) rec.bx[3] = a.max[0];
            if (a.max[1] > rec.bx[4]) rec.bx[4] = a.max[1];
            if (a.max[2] > rec.bx[5]) rec.bx[5] = a.max[2];
        }
        if (hi - lo <= leafSize) {
            rec.leaf = true; rec.off = lo; rec.cnt = hi - lo;
        } else {
            const ex = rec.bx[3] - rec.bx[0], ey = rec.bx[4] - rec.bx[1], ez = rec.bx[5] - rec.bx[2];
            const axis = (ex >= ey && ex >= ez) ? 0 : (ey >= ez ? 1 : 2);
            const sub = order.slice(lo, hi).sort((a, b) => aabbs[a].c[axis] - aabbs[b].c[axis]);
            for (let i = 0; i < sub.length; i++) order[lo + i] = sub[i];
            const mid = (lo + hi) >> 1;
            emit(lo, mid);
            emit(mid, hi);
        }
        rec.miss = records.length; // escape = first slot after this subtree
    }
    emit(0, order.length);
    return { records, order };
}

export async function buildSpectralScene({ THREE, scene, camera = null, maxTriangles = 4_000_000 } = {}) {
    if (!scene) return null;
    scene.updateMatrixWorld(true);

    const uberList = [];
    const uberMaterials = []; // parallel to uberList: source THREE material (for map extraction)
    const uberMap = new Map();
    function internMaterial(material) {
        const mat = material || {};
        const rec = materialToUber(mat);
        const key = uberKey(rec, mat);
        let idx = uberMap.get(key);
        if (idx === undefined) {
            idx = uberList.length;
            uberList.push(rec);
            uberMaterials.push(mat);
            uberMap.set(key, idx);
        }
        return idx;
    }

    // Pass 1: gather draws; dedupe BLAS by (geometry identity × per-tri uber
    // mapping) so shared and instanced geometry costs ONE local soup + BVH.
    const blasList = [];
    const blasByKey = new Map();
    const instances = []; // { blas, object, instanceIndex } — matrices re-read on update
    scene.traverseVisible((obj) => {
        if (!isTraceableMesh(obj, camera)) return;
        const geom = obj.geometry;
        const pos = geom.attributes.position;
        const index = geom.index;
        const triCount = index ? Math.floor(index.count / 3) : Math.floor(pos.count / 3);
        if (triCount <= 0) return;

        const mats = meshMaterials(obj);
        const { triMat, visibleTriCount, visibleTriIndices } =
            buildTriangleMaterialMap(geom, mats, triCount, internMaterial, Array.isArray(obj.material));
        if (visibleTriCount <= 0) return;

        const uniqueTriMaterial = Array.isArray(obj.material);
        let key = `${geom.uuid}:${index ? index.version : -1}:${pos.version}:${uniqueTriMaterial ? 1 : 0}`;
        if (uniqueTriMaterial) {
            let h = 0;
            for (let t = 0; t < triCount; t++) h = ((h * 31) + triMat[t] + 1) >>> 0;
            key += ':' + h;
        } else {
            key += ':' + triMat[0];
        }

        let blasIdx = blasByKey.get(key);
        if (blasIdx === undefined) {
            const blas = buildLocalBlas(THREE, {
                pos, index, triCount, visibleTriCount, visibleTriIndices, triMat, uniqueTriMaterial,
                normal: geom.attributes.normal || null,
                uv: geom.attributes.uv || null,
            });
            if (!blas) return;
            // Deform tracking: soup vertices [0, srcVertCount) map 1:1 onto the
            // source geometry's vertices, so updateDeforms can re-gather this
            // BLAS's pooled slice straight from the live attributes.
            blas.srcGeom = geom;
            blas.srcPosAttr = pos;
            blas.srcNormAttr = geom.attributes.normal || null;
            blas.srcIndexAttr = index || null;
            blas.srcPosVersion = pos.version | 0;
            blas.srcNormVersion = geom.attributes.normal ? (geom.attributes.normal.version | 0) : -1;
            blas.srcIndexVersion = index ? (index.version | 0) : -1;
            blas.srcPosDataVersion = attributeDataVersion(pos);
            blas.srcNormDataVersion = attributeDataVersion(geom.attributes.normal || null);
            blas.srcIndexDataVersion = attributeDataVersion(index || null);
            blas.srcIndexCount = index ? index.count : -1;
            blasIdx = blasList.length;
            blasList.push(blas);
            blasByKey.set(key, blasIdx);
        }
        if (obj.isInstancedMesh) {
            const capacity = Number.isFinite(obj.instanceMatrix?.count) ? obj.instanceMatrix.count : obj.count;
            // Reserve every allocated matrix slot in the TLAS. Games can then
            // grow/shrink `count` or recycle a slot without reallocating the
            // soup; inactive records are masked in traversal until activated.
            for (let i = 0; i < Math.max(0, capacity | 0); i++) {
                instances.push({ blas: blasIdx, object: obj, instanceIndex: i });
            }
        } else {
            instances.push({ blas: blasIdx, object: obj, instanceIndex: -1 });
        }
    });

    if (blasList.length === 0 || instances.length === 0) return null;
    const instCount = instances.length;
    // Stable object list for live transform refreshes. Build it once so the
    // timeline lane does not allocate/dedupe a Set on every update.
    const instanceObjects = Array.from(new Set(instances.map((ins) => ins.object)));
    const instanceObjectCounts = instanceObjects.map((object) => ({
        object,
        count: object.isInstancedMesh
            ? Math.max(0, (object.instanceMatrix?.count ?? object.count) | 0)
            : 1,
    }));
    const instanceCountByObject = new Map(instanceObjectCounts.map((entry) => [entry.object, entry.count]));
    if (instCount >= (1 << 24)) return { error: `too many instances for the TLAS leaf payload: ${instCount}` };

    // Pool assembly: place every BLAS at its base offsets, then serialize
    // nodes with pool-absolute miss links / leaf triOffsets.
    let poolVerts = 0, poolTris = 0, poolNodes = 0;
    for (const b of blasList) {
        b.vertBase = poolVerts; b.triBase = poolTris; b.nodeBase = poolNodes;
        poolVerts += b.vertCount; poolTris += b.triCount; poolNodes += b.records.length;
    }
    if (poolTris === 0) return null;
    if (poolTris > maxTriangles) {
        // Hard cap: refuse to build rather than blow the storage-buffer ceiling.
        // NOTE: the cap now applies to POOLED (unique) triangles — instances are free.
        return { error: `scene too large for path tracer: ${poolTris} tris > ${maxTriangles} cap` };
    }
    if (poolTris >= (1 << 24) || poolNodes >= (1 << 24)) {
        return { error: `BVH pool exceeds the 24-bit index ceiling (${poolTris} tris / ${poolNodes} nodes)` };
    }

    const triIndex = new Uint32Array(poolTris * 3);
    const triMaterial = new Uint32Array(poolTris);
    const vertexData = new Float32Array(poolVerts * VERTEX_DATA_STRIDE);
    const nodeBuf = new ArrayBuffer(poolNodes * NODE_STRIDE_U32 * 4);
    const nodesF = new Float32Array(nodeBuf);
    const nodesU = new Uint32Array(nodeBuf);
    for (const b of blasList) {
        for (let i = 0; i < b.vertCount; i++) {
            const dd = (b.vertBase + i) * VERTEX_DATA_STRIDE, p = i * VERT_STRIDE, n = i * 3, u = i * 2;
            vertexData[dd] = b.vertexPos[p]; vertexData[dd + 1] = b.vertexPos[p + 1]; vertexData[dd + 2] = b.vertexPos[p + 2];
            vertexData[dd + 3] = b.vertexNormal[n]; vertexData[dd + 4] = b.vertexNormal[n + 1]; vertexData[dd + 5] = b.vertexNormal[n + 2];
            vertexData[dd + 6] = b.vertexUV[u]; vertexData[dd + 7] = b.vertexUV[u + 1];
        }
        for (let t = 0; t < b.triCount; t++) {
            const to = (b.triBase + t) * 3, ts = t * 3;
            triIndex[to] = b.triIndex[ts] + b.vertBase;
            triIndex[to + 1] = b.triIndex[ts + 1] + b.vertBase;
            triIndex[to + 2] = b.triIndex[ts + 2] + b.vertBase;
            triMaterial[b.triBase + t] = b.triMaterial[t];
        }
        for (let i = 0; i < b.records.length; i++) {
            const base = (b.nodeBase + i) * NODE_STRIDE_U32;
            const rec = b.records[i];
            nodesF[base] = rec.bx[0]; nodesF[base + 1] = rec.bx[1]; nodesF[base + 2] = rec.bx[2];
            nodesF[base + 3] = rec.bx[3]; nodesF[base + 4] = rec.bx[4]; nodesF[base + 5] = rec.bx[5];
            nodesU[base + 6] = (rec.miss + b.nodeBase) >>> 0;
            nodesU[base + 7] = rec.leaf
                ? (((rec.triCount & 0xFF) << 24) | ((rec.triOffset + b.triBase) & 0x00FFFFFF)) >>> 0
                : 0xFFFFFFFF;
        }
        b.blasRoot = b.nodeBase;
        b.blasEnd = b.nodeBase + b.records.length;
        // pooled now — drop the per-BLAS copies so `built` holds one copy of the soup
        b.vertexPos = null; b.vertexNormal = null; b.vertexUV = null;
        b.triIndex = null; b.triMaterial = null; b.records = null;
    }

    // ── Dynamic tail: instance records + TLAS ──────────────────────
    const _m4 = new THREE.Matrix4();
    const _inv = new THREE.Matrix4();
    const _box = new THREE.Box3();
    function instanceWorldMatrix(ins, out) {
        const o = ins.object;
        if (ins.instanceIndex >= 0) {
            o.getMatrixAt(ins.instanceIndex, out);
            out.premultiply(o.matrixWorld);
        } else {
            out.copy(o.matrixWorld);
        }
        return out;
    }
    function computeDynamic() {
        const aabbs = new Array(instCount);
        const invRows = new Float32Array(instCount * 12);
        const detSign = new Float32Array(instCount);
        const active = new Uint8Array(instCount);
        const worldBounds = new THREE.Box3();
        worldBounds.makeEmpty();
        for (let i = 0; i < instCount; i++) {
            const ins = instances[i];
            const isActive = ins.instanceIndex < 0 || ins.instanceIndex < Math.max(0, ins.object.count | 0);
            active[i] = isActive ? 1 : 0;
            if (isActive) instanceWorldMatrix(ins, _m4);
            else _m4.copy(ins.object.matrixWorld);
            _inv.copy(_m4).invert();
            const e = _inv.elements; // column-major → store ROWS (w = translation term)
            const o = i * 12;
            invRows[o] = e[0]; invRows[o + 1] = e[4]; invRows[o + 2] = e[8]; invRows[o + 3] = e[12];
            invRows[o + 4] = e[1]; invRows[o + 5] = e[5]; invRows[o + 6] = e[9]; invRows[o + 7] = e[13];
            invRows[o + 8] = e[2]; invRows[o + 9] = e[6]; invRows[o + 10] = e[10]; invRows[o + 11] = e[14];
            detSign[i] = _m4.determinant() < 0 ? -1 : 1;
            if (isActive) {
                _box.copy(blasList[ins.blas].localBounds).applyMatrix4(_m4);
                worldBounds.union(_box);
            } else {
                // A finite point keeps the threaded TLAS numerically valid;
                // the instance active flag prevents its BLAS from being entered.
                const eWorld = ins.object.matrixWorld.elements;
                _box.min.set(eWorld[12], eWorld[13], eWorld[14]);
                _box.max.copy(_box.min);
            }
            aabbs[i] = {
                min: [_box.min.x, _box.min.y, _box.min.z],
                max: [_box.max.x, _box.max.y, _box.max.z],
                c: [(_box.min.x + _box.max.x) * 0.5, (_box.min.y + _box.max.y) * 0.5, (_box.min.z + _box.max.z) * 0.5],
            };
        }
        if (worldBounds.isEmpty() && aabbs.length > 0) {
            worldBounds.min.fromArray(aabbs[0].min);
            worldBounds.max.fromArray(aabbs[0].max);
        }
        const { records, order } = buildTlasRecords(aabbs);
        return { aabbs, invRows, detSign, active, records, order, worldBounds };
    }

    // Extract PBR maps into array textures FIRST — this writes each material's
    // assigned layer index into its uber record ([12..16]) before we pack the
    // materials buffer below.
    const maps = await buildMaterialTextures(THREE, uberList, uberMaterials, TEXTURE_ATLAS_SIZE);

    // Materials buffer with the dynamic tail:
    //   [ ubers (uberCount×MAT_STRIDE) | instances (instCount×MAT_STRIDE) | TLAS (tlasNodes×12) ]
    // TLAS nodes are PLAIN floats (bounds, miss, instOffset, instCount — all
    // exact small integers as f32). Never bit-cast uints into a float buffer:
    // denormal miss links flush to zero on some drivers and NaN interior
    // markers lose their payload bits.
    const TLAS_STRIDE_F32 = 12;
    const dyn0 = computeDynamic();
    const instBase = uberList.length * MAT_STRIDE;     // float-element index
    const tlasBase = instBase + instCount * MAT_STRIDE;
    const tlasNodeCount = dyn0.records.length;
    const materials = new Float32Array(tlasBase + tlasNodeCount * TLAS_STRIDE_F32);
    const tlasNodes = materials.subarray(tlasBase, tlasBase + tlasNodeCount * TLAS_STRIDE_F32);
    const liveInstanceBounds = new Float32Array(instCount * 6);
    const tlasOrder = dyn0.order;
    const slotsByObject = new Map();
    const objectByUuid = new Map();
    for (let slot = 0; slot < instCount; slot++) {
        const object = instances[tlasOrder[slot]].object;
        let slots = slotsByObject.get(object);
        if (!slots) { slots = []; slotsByObject.set(object, slots); }
        slots.push(slot);
        if (typeof object.uuid === 'string') objectByUuid.set(object.uuid, object);
    }
    for (let i = 0; i < uberList.length; i++) materials.set(uberList[i], i * MAT_STRIDE);

    function writeDynamic(dyn) {
        for (let slot = 0; slot < instCount; slot++) {
            const src = dyn.order[slot];
            const ins = instances[src];
            const blas = blasList[ins.blas];
            const b = instBase + slot * MAT_STRIDE;
            materials.set(dyn.invRows.subarray(src * 12, src * 12 + 12), b);
            materials[b + 12] = blas.blasRoot;   // exact ≤ 2^24 (guarded above)
            materials[b + 13] = blas.blasEnd;
            materials[b + 14] = dyn.detSign[src];
            materials[b + 15] = dyn.active[src];
            materials.fill(0, b + 16, b + MAT_STRIDE);
            const aabb = dyn.aabbs[src];
            const bb = slot * 6;
            liveInstanceBounds[bb] = aabb.min[0]; liveInstanceBounds[bb + 1] = aabb.min[1]; liveInstanceBounds[bb + 2] = aabb.min[2];
            liveInstanceBounds[bb + 3] = aabb.max[0]; liveInstanceBounds[bb + 4] = aabb.max[1]; liveInstanceBounds[bb + 5] = aabb.max[2];
        }
        for (let i = 0; i < dyn.records.length; i++) {
            const rec = dyn.records[i];
            const b = tlasBase + i * TLAS_STRIDE_F32;
            materials[b] = rec.bx[0]; materials[b + 1] = rec.bx[1]; materials[b + 2] = rec.bx[2];
            materials[b + 3] = rec.bx[3]; materials[b + 4] = rec.bx[4]; materials[b + 5] = rec.bx[5];
            materials[b + 6] = rec.miss;                    // exact ints ≤ 2^24 as f32
            materials[b + 7] = rec.leaf ? rec.off : 0;
            materials[b + 8] = rec.leaf ? rec.cnt : 0;      // 0 = interior → descend
            materials[b + 9] = 0; materials[b + 10] = 0; materials[b + 11] = 0;
        }
    }
    writeDynamic(dyn0);
    const tlasRefitIndex = createFlatTlasRefitIndex({
        nodes: tlasNodes,
        instanceBounds: liveInstanceBounds,
        end: tlasNodeCount,
    });
    if (!tlasRefitIndex) return { error: 'invalid flattened TLAS topology' };

    function currentInstanceCapacity(object) {
        return object.isInstancedMesh
            ? Math.max(0, (object.instanceMatrix?.count ?? object.count) | 0)
            : 1;
    }

    function recordRanges(indices, base, stride) {
        if (!indices.length) return [];
        const sorted = Array.from(new Set(indices)).sort((a, b) => a - b);
        const ranges = [];
        let start = sorted[0], end = start;
        for (let i = 1; i < sorted.length; i++) {
            const value = sorted[i];
            if (value === end + 1) { end = value; continue; }
            ranges.push([base + start * stride, (end - start + 1) * stride]);
            start = end = value;
        }
        ranges.push([base + start * stride, (end - start + 1) * stride]);
        return ranges;
    }

    // Moving-object fast path. With no object list this preserves the legacy
    // compatibility scan/refit. Event-driven hosts pass the dirty Object3D(s),
    // rewriting only their stable instance slots and exact TLAS ancestors.
    // A wildly rearranged scene can make the frozen partition less efficient
    // to traverse, but its exact refit bounds stay correct. Returns null only
    // if the stored threaded layout is invalid (caller should full-rebuild).
    function updateTransforms({ rewriteInstanceRows = true, objects = null } = {}) {
        const dirtySlots = [];
        const dirtyObjects = new Set();
        const full = objects == null;
        if (full) {
            for (const entry of instanceObjectCounts) {
                if (currentInstanceCapacity(entry.object) !== entry.count) return null;
                dirtyObjects.add(entry.object);
            }
            for (let slot = 0; slot < instCount; slot++) dirtySlots.push(slot);
        } else {
            const targets = Array.isArray(objects) || objects instanceof Set ? objects : [objects];
            for (const target of targets) {
                const rawObject = target?.object || target;
                const object = typeof rawObject === 'string' ? objectByUuid.get(rawObject) : rawObject;
                const slots = slotsByObject.get(object);
                // Hosts commonly report transforms for helpers, cameras, newly
                // spawned objects awaiting a topology packet, or meshes excluded
                // from GI. They are not evidence that the resident soup is bad.
                if (!object || !slots) continue;
                if (currentInstanceCapacity(object) !== (instanceCountByObject.get(object) ?? -1)) return null;
                dirtyObjects.add(object);
                const instanceIndex = Number.isInteger(target?.instanceIndex) ? target.instanceIndex : -1;
                if (instanceIndex < 0) {
                    dirtySlots.push(...slots);
                } else {
                    let found = false;
                    for (const slot of slots) {
                        if (instances[tlasOrder[slot]].instanceIndex !== instanceIndex) continue;
                        dirtySlots.push(slot);
                        found = true;
                    }
                    if (!found) continue;
                }
            }
        }
        if (dirtySlots.length > 1) {
            dirtySlots.sort((a, b) => a - b);
            let write = 1;
            for (let read = 1; read < dirtySlots.length; read++) {
                if (dirtySlots[read] !== dirtySlots[write - 1]) dirtySlots[write++] = dirtySlots[read];
            }
            dirtySlots.length = write;
        }
        if (dirtySlots.length === 0) return {
            bounds: null, materialRanges: [], updatedInstances: 0, refittedTlasNodes: 0, full,
        };
        for (const object of dirtyObjects) object.updateWorldMatrix?.(true, false);
        for (const slot of dirtySlots) {
            const src = tlasOrder[slot];
            const ins = instances[src];
            const isActive = ins.instanceIndex < 0 || ins.instanceIndex < Math.max(0, ins.object.count | 0);
            if (isActive) instanceWorldMatrix(ins, _m4);
            else _m4.copy(ins.object.matrixWorld);
            if (rewriteInstanceRows) {
                const detSign = _m4.determinant() < 0 ? -1 : 1;
                _inv.copy(_m4).invert();
                const e = _inv.elements;
                const b = instBase + slot * MAT_STRIDE;
                materials[b] = e[0]; materials[b + 1] = e[4]; materials[b + 2] = e[8]; materials[b + 3] = e[12];
                materials[b + 4] = e[1]; materials[b + 5] = e[5]; materials[b + 6] = e[9]; materials[b + 7] = e[13];
                materials[b + 8] = e[2]; materials[b + 9] = e[6]; materials[b + 10] = e[10]; materials[b + 11] = e[14];
                materials[b + 14] = detSign;
                materials[b + 15] = isActive ? 1 : 0;
            }
            if (isActive) {
                _box.copy(blasList[ins.blas].localBounds).applyMatrix4(_m4);
            } else {
                const eWorld = ins.object.matrixWorld.elements;
                _box.min.set(eWorld[12], eWorld[13], eWorld[14]);
                _box.max.copy(_box.min);
            }
            const bb = slot * 6;
            liveInstanceBounds[bb] = _box.min.x;
            liveInstanceBounds[bb + 1] = _box.min.y;
            liveInstanceBounds[bb + 2] = _box.min.z;
            liveInstanceBounds[bb + 3] = _box.max.x;
            liveInstanceBounds[bb + 4] = _box.max.y;
            liveInstanceBounds[bb + 5] = _box.max.z;
        }
        let touchedNodes;
        if (full) {
            if (!refitFlatTlasRange({ nodes: tlasNodes, instanceBounds: liveInstanceBounds, end: tlasNodeCount })) return null;
            touchedNodes = Array.from({ length: tlasNodeCount }, (_, i) => i);
        } else {
            touchedNodes = refitFlatTlasDirty({
                nodes: tlasNodes,
                instanceBounds: liveInstanceBounds,
                dirtySlots,
                index: tlasRefitIndex,
            });
            if (!touchedNodes) return null;
        }
        const bounds = new THREE.Box3();
        bounds.min.set(tlasNodes[0], tlasNodes[1], tlasNodes[2]);
        bounds.max.set(tlasNodes[3], tlasNodes[4], tlasNodes[5]);
        const materialRanges = [];
        if (rewriteInstanceRows) materialRanges.push(...recordRanges(dirtySlots, instBase, MAT_STRIDE));
        materialRanges.push(...recordRanges(touchedNodes, tlasBase, TLAS_STRIDE_F32));
        return {
            bounds,
            materialRanges,
            updatedInstances: dirtySlots.length,
            refittedTlasNodes: touchedNodes.length,
            full,
        };
    }

    let asyncDeformRequestSerial = 0;
    let asyncDeformLifecycle = 0;
    let asyncDeformDisposed = false;
    let asyncDeformLatestOptions = null;
    let asyncDeformPump = null;
    let asyncDeformPumpLifecycle = -1;

    function attributeArrayIdentity(attr) {
        return attr?.isInterleavedBufferAttribute ? attr.data?.array : attr?.array;
    }

    function attributeDataVersion(attr) {
        return attr?.isInterleavedBufferAttribute ? (attr.data?.version | 0) : -1;
    }

    function directFloatVec3(attr) {
        if (!attr || attr.normalized === true) return null;
        if (attr.isInterleavedBufferAttribute) {
            const array = attr.data?.array;
            const stride = attr.data?.stride | 0;
            const offset = attr.offset | 0;
            if (array instanceof Float32Array && stride >= offset + 3) {
                return { array, stride, offset };
            }
            return null;
        }
        if (attr.array instanceof Float32Array && (attr.itemSize | 0) >= 3) {
            return { array: attr.array, stride: attr.itemSize | 0, offset: 0 };
        }
        return null;
    }

    function stampAttribute(attr) {
        return {
            attr: attr || null,
            count: attr ? attr.count : -1,
            version: attr ? (attr.version | 0) : -1,
            dataVersion: attributeDataVersion(attr),
            array: attributeArrayIdentity(attr),
            direct: directFloatVec3(attr),
        };
    }

    function attributeMatches(attr, stamp) {
        const current = attr || null;
        return current === stamp.attr
            && (current ? current.count : -1) === stamp.count
            && (current ? (current.version | 0) : -1) === stamp.version
            && attributeDataVersion(current) === stamp.dataVersion
            && attributeArrayIdentity(current) === stamp.array;
    }

    function captureDeformSnapshot(requestSerial, lifecycle, options) {
        const records = [];
        let work = 0;
        for (const b of blasList) {
            const geom = b.srcGeom;
            const posA = geom?.attributes?.position || null;
            if (!posA) return { invalid: true };
            const normA = geom.attributes.normal || null;
            const indexA = geom.index || null;
            const indexVersion = indexA ? (indexA.version | 0) : -1;
            const indexDataVersion = attributeDataVersion(indexA);
            const indexCount = indexA ? indexA.count : -1;
            // Connectivity is structural. Reject the entire snapshot before
            // touching pooled arrays so no partially updated soup is exposed.
            if (indexA !== b.srcIndexAttr || indexVersion !== b.srcIndexVersion ||
                indexDataVersion !== b.srcIndexDataVersion ||
                indexCount !== b.srcIndexCount || posA.count !== b.srcVertCount ||
                (normA && normA.count !== b.srcVertCount)) {
                return { invalid: true };
            }
            const pos = stampAttribute(posA);
            const norm = stampAttribute(normA);
            const index = stampAttribute(indexA);
            const posChanged = posA !== b.srcPosAttr || pos.version !== b.srcPosVersion ||
                pos.dataVersion !== b.srcPosDataVersion;
            const normChanged = normA !== b.srcNormAttr || norm.version !== b.srcNormVersion ||
                norm.dataVersion !== b.srcNormDataVersion;
            const changed = posChanged || normChanged;
            const rec = { b, geom, pos, norm, index, posChanged, normChanged, changed };
            records.push(rec);
            if (changed) {
                work += b.srcVertCount + (b.tagSrc?.length || 0);
                if (posChanged) work += b.blasEnd - b.blasRoot;
            }
        }
        return { requestSerial, lifecycle, options, records, work, invalid: false };
    }

    function validateDeformSnapshot(snapshot) {
        if (!snapshot || snapshot.invalid || asyncDeformDisposed ||
            snapshot.lifecycle !== asyncDeformLifecycle ||
            snapshot.requestSerial !== asyncDeformRequestSerial) return false;
        for (const rec of snapshot.records) {
            const geom = rec.b.srcGeom;
            if (geom !== rec.geom ||
                !attributeMatches(geom?.attributes?.position, rec.pos) ||
                !attributeMatches(geom?.attributes?.normal || null, rec.norm) ||
                !attributeMatches(geom?.index || null, rec.index)) return false;
        }
        return true;
    }

    function createAsyncDeformTask(snapshot) {
        const changed = snapshot.records.filter((rec) => rec.changed);
        const task = {
            changed,
            recordIndex: 0,
            phase: 'gather',
            sourceIndex: 0,
            tagIndex: 0,
            refit: null,
            done: changed.length === 0,
        };
        const copySourceRange = (rec, begin, end) => {
            const b = rec.b;
            const posDirect = rec.pos.direct;
            const normDirect = rec.norm.direct;
            for (let i = begin; i < end; i++) {
                const dd = (b.vertBase + i) * VERTEX_DATA_STRIDE;
                if (rec.posChanged) {
                    if (posDirect) {
                        const s = posDirect.offset + i * posDirect.stride;
                        vertexData[dd] = posDirect.array[s];
                        vertexData[dd + 1] = posDirect.array[s + 1];
                        vertexData[dd + 2] = posDirect.array[s + 2];
                    } else {
                        vertexData[dd] = rec.pos.attr.getX(i);
                        vertexData[dd + 1] = rec.pos.attr.getY(i);
                        vertexData[dd + 2] = rec.pos.attr.getZ(i);
                    }
                }
                if (rec.normChanged && rec.norm.attr) {
                    if (normDirect) {
                        const s = normDirect.offset + i * normDirect.stride;
                        vertexData[dd + 3] = normDirect.array[s];
                        vertexData[dd + 4] = normDirect.array[s + 1];
                        vertexData[dd + 5] = normDirect.array[s + 2];
                    } else {
                        vertexData[dd + 3] = rec.norm.attr.getX(i);
                        vertexData[dd + 4] = rec.norm.attr.getY(i);
                        vertexData[dd + 5] = rec.norm.attr.getZ(i);
                    }
                }
            }
        };
        task.step = ({ deadline, now }) => {
            while (task.recordIndex < changed.length) {
                const rec = changed[task.recordIndex];
                const b = rec.b;
                if (task.phase === 'gather') {
                    const end = Math.min(b.srcVertCount, task.sourceIndex + 256);
                    copySourceRange(rec, task.sourceIndex, end);
                    task.sourceIndex = end;
                    if (task.sourceIndex < b.srcVertCount) {
                        if (now() >= deadline) return true;
                        continue;
                    }
                    task.phase = 'tags';
                }
                if (task.phase === 'tags') {
                    const tags = b.tagSrc;
                    if (tags) {
                        const end = Math.min(tags.length, task.tagIndex + 256);
                        const tagBase = b.vertBase + b.srcVertCount;
                        for (let t = task.tagIndex; t < end; t++) {
                            const s = tags[t];
                            if (s === 0xFFFFFFFF) continue;
                            if (s >= b.srcVertCount) return false;
                            const dd = (tagBase + t) * VERTEX_DATA_STRIDE;
                            const ds = (b.vertBase + s) * VERTEX_DATA_STRIDE;
                            vertexData[dd] = vertexData[ds];
                            vertexData[dd + 1] = vertexData[ds + 1];
                            vertexData[dd + 2] = vertexData[ds + 2];
                            vertexData[dd + 3] = vertexData[ds + 3];
                            vertexData[dd + 4] = vertexData[ds + 4];
                            vertexData[dd + 5] = vertexData[ds + 5];
                        }
                        task.tagIndex = end;
                        if (task.tagIndex < tags.length) {
                            if (now() >= deadline) return true;
                            continue;
                        }
                    }
                    task.phase = rec.posChanged ? 'refit' : 'next';
                }
                if (task.phase === 'refit') {
                    task.refit ||= createFlatBlasRefitStepper({
                        nodesF, nodesU, triIndex, vertexData,
                        root: b.blasRoot, end: b.blasEnd,
                        nodeStrideU32: NODE_STRIDE_U32,
                        vertexDataStride: VERTEX_DATA_STRIDE,
                    });
                    if (!task.refit.valid || task.refit.step({ deadline, now }) === false) return false;
                    if (!task.refit.done) return true;
                    task.phase = 'next';
                }
                if (task.phase === 'next') {
                    task.recordIndex++;
                    task.phase = 'gather';
                    task.sourceIndex = 0;
                    task.tagIndex = 0;
                    task.refit = null;
                    if (task.recordIndex < changed.length && now() >= deadline) return true;
                }
            }
            task.done = true;
            return true;
        };
        return task;
    }

    function commitAsyncDeform(snapshot, task) {
        if (!validateDeformSnapshot(snapshot)) return { restart: true };
        const vertRanges = [], nodeRanges = [];
        let refitted = 0;
        for (const rec of task.changed) {
            const b = rec.b;
            b.srcPosAttr = rec.pos.attr;
            b.srcNormAttr = rec.norm.attr;
            b.srcPosVersion = rec.pos.version;
            b.srcNormVersion = rec.norm.version;
            b.srcPosDataVersion = rec.pos.dataVersion;
            b.srcNormDataVersion = rec.norm.dataVersion;
            vertRanges.push([b.vertBase * VERTEX_DATA_STRIDE, b.vertCount * VERTEX_DATA_STRIDE]);
            if (rec.posChanged) {
                const rb = b.blasRoot * NODE_STRIDE_U32;
                b.localBounds.min.set(nodesF[rb], nodesF[rb + 1], nodesF[rb + 2]);
                b.localBounds.max.set(nodesF[rb + 3], nodesF[rb + 4], nodesF[rb + 5]);
                nodeRanges.push([b.blasRoot * NODE_STRIDE_U32, (b.blasEnd - b.blasRoot) * NODE_STRIDE_U32]);
                refitted++;
            }
        }
        if (task.changed.length === 0) {
            return { updated: 0, refitted: 0, bounds: null, vertRanges, nodeRanges };
        }
        if (refitted === 0 || snapshot.options.updateTlas === false) {
            return { updated: task.changed.length, refitted, bounds: null, vertRanges, nodeRanges };
        }
        const res = updateTransforms({ rewriteInstanceRows: false });
        if (!res) return null;
        return { updated: task.changed.length, refitted, bounds: res.bounds, vertRanges, nodeRanges };
    }

    // Deform fast path: SAME topology, moved vertices (streamed vertex
    // buffers, morphs, CPU skinning). Small edits retain the synchronous API;
    // large edits use updateDeformsAsync below so the UI never pays one
    // unbounded gather/refit task.
    function updateDeforms({ updateTlas = true } = {}) {
        if (asyncDeformPump) {
            // A direct synchronous caller owns the buffers now. Invalidate the
            // yielded job; this complete pass overwrites every uncommitted
            // changed slice before publishing its ranges.
            asyncDeformLifecycle++;
            asyncDeformRequestSerial++;
        }
        let updated = 0;
        let refitted = 0;
        const vertRanges = [], nodeRanges = [];
        for (const b of blasList) {
            const geom = b.srcGeom;
            const posA = geom?.attributes?.position;
            if (!posA) return null;
            const normA = geom.attributes.normal || null;
            const indexA = geom.index || null;
            const pv = posA.version | 0;
            const nv = normA ? (normA.version | 0) : -1;
            const iv = indexA ? (indexA.version | 0) : -1;
            const pdv = attributeDataVersion(posA);
            const ndv = attributeDataVersion(normA);
            const idv = attributeDataVersion(indexA);
            // Connectivity is structural, never a deformation. Fail before
            // touching the pooled vertices so an index edit cannot refit the
            // old triangle soup during the structural debounce window.
            if (indexA !== b.srcIndexAttr || iv !== b.srcIndexVersion ||
                idv !== b.srcIndexDataVersion ||
                (indexA ? indexA.count : -1) !== b.srcIndexCount) return null;
            const posChanged = posA !== b.srcPosAttr ||
                pv !== b.srcPosVersion || pdv !== b.srcPosDataVersion;
            const normChanged = normA !== b.srcNormAttr ||
                nv !== b.srcNormVersion || ndv !== b.srcNormDataVersion;
            if (!posChanged && !normChanged) continue;
            if (posA.count !== b.srcVertCount || (normA && normA.count !== b.srcVertCount)) return null;
            for (let i = 0; i < b.srcVertCount; i++) {
                const dd = (b.vertBase + i) * VERTEX_DATA_STRIDE;
                if (posChanged) {
                    vertexData[dd] = posA.getX(i);
                    vertexData[dd + 1] = posA.getY(i);
                    vertexData[dd + 2] = posA.getZ(i);
                }
                if (normChanged && normA) {
                    vertexData[dd + 3] = normA.getX(i);
                    vertexData[dd + 4] = normA.getY(i);
                    vertexData[dd + 5] = normA.getZ(i);
                }
                // normal attribute gone since build → keep the build-time
                // normals (stale beats zero/flat for a low-frequency field)
            }
            if (b.tagSrc) {
                const tagBase = b.vertBase + b.srcVertCount;
                for (let t = 0; t < b.tagSrc.length; t++) {
                    const s = b.tagSrc[t];
                    if (s === 0xFFFFFFFF) continue;
                    const dd = (tagBase + t) * VERTEX_DATA_STRIDE, ds = (b.vertBase + s) * VERTEX_DATA_STRIDE;
                    if (posChanged) {
                        vertexData[dd] = vertexData[ds];
                        vertexData[dd + 1] = vertexData[ds + 1];
                        vertexData[dd + 2] = vertexData[ds + 2];
                    }
                    if (normChanged) {
                        vertexData[dd + 3] = vertexData[ds + 3];
                        vertexData[dd + 4] = vertexData[ds + 4];
                        vertexData[dd + 5] = vertexData[ds + 5];
                    }
                }
            }
            if (posChanged && !refitFlatBlasRange({
                    nodesF, nodesU, triIndex, vertexData,
                    root: b.blasRoot, end: b.blasEnd,
                    nodeStrideU32: NODE_STRIDE_U32, vertexDataStride: VERTEX_DATA_STRIDE,
                })) return null;
            b.srcPosAttr = posA;
            b.srcNormAttr = normA;
            b.srcPosVersion = pv;
            b.srcNormVersion = nv;
            b.srcPosDataVersion = pdv;
            b.srcNormDataVersion = ndv;
            vertRanges.push([b.vertBase * VERTEX_DATA_STRIDE, b.vertCount * VERTEX_DATA_STRIDE]);
            updated++;
            if (posChanged) {
                const rb = b.blasRoot * NODE_STRIDE_U32;
                b.localBounds.min.set(nodesF[rb], nodesF[rb + 1], nodesF[rb + 2]);
                b.localBounds.max.set(nodesF[rb + 3], nodesF[rb + 4], nodesF[rb + 5]);
                nodeRanges.push([b.blasRoot * NODE_STRIDE_U32, (b.blasEnd - b.blasRoot) * NODE_STRIDE_U32]);
                refitted++;
            }
        }
        if (updated === 0) return { updated: 0, refitted: 0, bounds: null, vertRanges, nodeRanges };
        if (refitted === 0 || !updateTlas) {
            return { updated, refitted, bounds: null, vertRanges, nodeRanges };
        }
        const res = updateTransforms({ rewriteInstanceRows: false }); // TLAS bounds from the refit local bounds
        if (!res) return null;
        return { updated, refitted, bounds: res.bounds, vertRanges, nodeRanges };
    }

    function updateDeformsAsync(options = {}) {
        if (asyncDeformDisposed) return Promise.resolve(null);
        asyncDeformRequestSerial++;
        asyncDeformLatestOptions = {
            updateTlas: options.updateTlas !== false,
            budgetMs: Math.min(ASYNC_DEFORM_MAX_BUDGET_MS,
                Math.max(0.25, Number.isFinite(options.budgetMs) ? options.budgetMs : ASYNC_DEFORM_MAX_BUDGET_MS)),
            workThreshold: Math.max(0, Number.isFinite(options.workThreshold)
                ? options.workThreshold : ASYNC_DEFORM_WORK_THRESHOLD),
            now: typeof options.now === 'function' ? options.now : undefined,
            yieldTask: typeof options.yieldTask === 'function' ? options.yieldTask : undefined,
        };
        if (asyncDeformPump && asyncDeformPumpLifecycle === asyncDeformLifecycle) {
            return asyncDeformPump;
        }

        const preview = captureDeformSnapshot(
            asyncDeformRequestSerial, asyncDeformLifecycle, asyncDeformLatestOptions);
        if (preview.invalid) return Promise.resolve(null);
        if (preview.work <= asyncDeformLatestOptions.workThreshold) {
            return Promise.resolve(updateDeforms({ updateTlas: asyncDeformLatestOptions.updateTlas }));
        }

        const pumpLifecycle = asyncDeformLifecycle;
        const running = runLatestBudgetedTask({
            capture: () => captureDeformSnapshot(
                asyncDeformRequestSerial, asyncDeformLifecycle, asyncDeformLatestOptions),
            createTask: createAsyncDeformTask,
            validate: validateDeformSnapshot,
            commit: commitAsyncDeform,
            isCancelled: () => asyncDeformDisposed || asyncDeformLifecycle !== pumpLifecycle,
            budgetMs: asyncDeformLatestOptions.budgetMs,
            now: asyncDeformLatestOptions.now,
            yieldTask: asyncDeformLatestOptions.yieldTask,
        });
        let wrapped = null;
        wrapped = running.finally(() => {
            if (asyncDeformPump === wrapped) {
                asyncDeformPump = null;
                asyncDeformPumpLifecycle = -1;
            }
        });
        asyncDeformPump = wrapped;
        asyncDeformPumpLifecycle = pumpLifecycle;
        return wrapped;
    }

    function cancelDeformUpdates() {
        asyncDeformLifecycle++;
        asyncDeformRequestSerial++;
    }

    function disposeDeformUpdates() {
        asyncDeformDisposed = true;
        cancelDeformUpdates();
    }

    // Lights
    const lightRecords = collectLights(THREE, scene, camera);
    const lights = new Float32Array(Math.max(1, lightRecords.length) * LIGHT_STRIDE);
    for (let i = 0; i < lightRecords.length; i++) lights.set(lightRecords[i], i * LIGHT_STRIDE);

    // Same-layout light fast path. Light motion/color/intensity does not touch
    // either BVH; rewrite the retained packed buffer in place. A count change
    // changes storage layout and therefore fails closed to a full scene build.
    function updateLights() {
        const nextRecords = collectLights(THREE, scene, camera);
        if (nextRecords.length !== lightRecords.length) return null;
        lights.fill(0);
        for (let i = 0; i < nextRecords.length; i++) {
            lights.set(nextRecords[i], i * LIGHT_STRIDE);
        }
        return { lightCount: nextRecords.length };
    }

    // Environment (equirect)
    const env = scene.userData?.maxjsPathTraceEnvironment?.isTexture
        ? scene.userData.maxjsPathTraceEnvironment
        : (scene.environment?.isTexture ? scene.environment : null);

    return {
        error: null,
        bounds: dyn0.worldBounds.clone(), // world-space AABB of the traced instances
        bvhNodes: nodesU, nodeCount: poolNodes,
        triIndex, triCount: poolTris,
        vertexData, vertexCount: poolVerts,
        triMaterial,
        materials, materialCount: uberList.length,
        instBase, instCount, tlasBase, tlasNodeCount,
        updateTransforms,
        updateDeforms,
        updateDeformsAsync,
        cancelDeformUpdates,
        disposeDeformUpdates,
        updateLights,
        lights, lightCount: lightRecords.length,
        env,
        maps, // { albedo, normal, roughness, metalness, emissive, alpha } DataArrayTexture | null
        strides: { NODE_STRIDE_U32, MAT_STRIDE, LIGHT_STRIDE, VERT_STRIDE, VERTEX_DATA_STRIDE, TLAS_STRIDE_F32 },
    };
}

export { NODE_STRIDE_U32, MAT_STRIDE, LIGHT_STRIDE, VERT_STRIDE, VERTEX_DATA_STRIDE, BYTES_PER_BVH_NODE };
