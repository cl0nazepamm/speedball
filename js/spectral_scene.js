// spectral_scene.js — CPU build pipeline for the WebGPU spectral path tracer.
//
// Flattens the visible scene (meshes + InstancedMesh instances) into ONE
// world-space indexed triangle soup, dedupes materials into a flat uber
// table, builds a CPU MeshBVH (three-mesh-bvh), and re-flattens the BVH to a
// STACKLESS threaded layout (each node carries a single "miss"/escape index)
// so the GPU traversal is one loop with no traversal stack.
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
//        the classifyNir heuristics) · [26..27] pad
const MAT_STRIDE = 28;
const LIGHT_STRIDE = 17;        // lights: see layout below
const VERT_STRIDE = 3;          // vertexPos: tightly packed xyz (flat f32 storage, read by offset; also fed to MeshBVH which assumes stride 3)
const VERTEX_DATA_STRIDE = 8;   // GPU interleaved per-vertex: pos(3) + normal(3) + uv(2)
const BYTES_PER_BVH_NODE = 32;  // three-mesh-bvh BYTES_PER_NODE (8 x u32)
const TEXTURE_ATLAS_SIZE = 256; // every material map is resampled to this square size and stacked into a DataArrayTexture layer
const SKIP_TRIANGLE_MATERIAL = 0xFFFFFFFF;

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
    const roughnessC = Math.min(1, Math.max(0.02, roughness));
    const metalnessC = Math.min(1, Math.max(0, metalness));
    const transmissionC = Math.min(1, Math.max(0, transmission));
    const nirAlbedo = Number.isFinite(udNir)
        ? Math.min(1, Math.max(0, udNir))
        : classifyNir(material.name, r, g, b, roughnessC, metalnessC, transmissionC);

    return [r, g, b,
        Math.min(1, Math.max(0.02, roughness)),
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
        0, 0];                  // [26..27] pad
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
// three-mesh-bvh node layout (BYTES_PER_NODE = 32, addressed in 32-bit
// words): bounds = float bits [n32+0..5]; leaf test u16[n32*2+15]===0xFFFF;
// leaf: triOffset=u32[n32+6] (tri units), triCount=u16[n32*2+14]; internal:
// left child = n32+8 (contiguous in source), right child word = u32[n32+6].
// We re-emit DFS so the LEFT child is contiguous (idx+1) in the output and
// every node gets a single escape/miss index = first node after its subtree.

function flattenBVHRoot(rootBuffer) {
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

    while (work.length > 0) {
        const item = work.pop();
        if (item.kind === 'finalize') {
            records[item.idx].miss = records.length;
            continue;
        }
        const n32 = item.n32;
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
        } else {
            const leftN32 = n32 + 8;
            const rightN32 = u32[n32 + 6];
            // finalize sets miss AFTER both children are flattened
            work.push({ kind: 'finalize', idx });
            work.push({ kind: 'visit', n32: rightN32 });
            work.push({ kind: 'visit', n32: leftN32 });
        }
    }

    // Serialize to one Uint32Array (bounds stored as float bits).
    const nodeCount = records.length;
    const buf = new ArrayBuffer(nodeCount * NODE_STRIDE_U32 * 4);
    const outF = new Float32Array(buf);
    const outU = new Uint32Array(buf);
    for (let i = 0; i < nodeCount; i++) {
        const base = i * NODE_STRIDE_U32;
        const rec = records[i];
        outF[base + 0] = rec.bx[0]; outF[base + 1] = rec.bx[1]; outF[base + 2] = rec.bx[2];
        outF[base + 3] = rec.bx[3]; outF[base + 4] = rec.bx[4]; outF[base + 5] = rec.bx[5];
        outU[base + 6] = rec.miss >>> 0;
        outU[base + 7] = rec.leaf
            ? (((rec.triCount & 0xFF) << 24) | (rec.triOffset & 0x00FFFFFF)) >>> 0
            : 0xFFFFFFFF;
    }
    return { nodes: outU, nodeCount };
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

export async function buildSpectralScene({ THREE, scene, camera = null, maxTriangles = 4_000_000 } = {}) {
    if (!scene) return null;
    scene.updateMatrixWorld(true);

    // Pass 1: gather eligible (mesh, matrixWorld, materialUberIndex-per-group)
    // and count verts/tris so we can allocate once.
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

    const draws = []; // { geometry, matrices: Mat4[], triMat: Uint32 per-source-tri material }
    let totalVerts = 0;
    let totalTris = 0;

    const mat4Scratch = new THREE.Matrix4();
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

        const matrices = [];
        if (obj.isInstancedMesh) {
            const capacity = Number.isFinite(obj.instanceMatrix?.count) ? obj.instanceMatrix.count : obj.count;
            const count = Math.max(0, Math.min(obj.count | 0, capacity | 0));
            for (let i = 0; i < count; i++) {
                const m = new THREE.Matrix4();
                obj.getMatrixAt(i, m);
                m.premultiply(obj.matrixWorld);
                matrices.push(m);
            }
        } else {
            matrices.push(obj.matrixWorld.clone());
        }
        if (matrices.length === 0) return;

        const uniqueTriMaterial = Array.isArray(obj.material);
        totalVerts += (pos.count + (uniqueTriMaterial ? visibleTriCount : 0)) * matrices.length;
        totalTris += visibleTriCount * matrices.length;
        draws.push({
            geom, pos, index, triCount, visibleTriCount, visibleTriIndices, matrices, triMat, uniqueTriMaterial,
            normal: geom.attributes.normal || null,
            uv: geom.attributes.uv || null,
        });
    });

    if (totalTris === 0 || totalVerts === 0) return null;
    if (totalTris > maxTriangles) {
        // Hard cap: refuse to build rather than blow the storage-buffer ceiling.
        return { error: `scene too large for path tracer: ${totalTris} tris > ${maxTriangles} cap` };
    }

    // Pass 2: allocate and fill the world-space indexed soup. vertexMaterial
    // is stamped so the final per-triangle material survives the BVH index
    // permutation. Material-array draws get a unique first vertex per visible
    // triangle, which preserves exact Multi/Sub group material assignment even
    // when source triangles share vertices.
    const vertexPos = new Float32Array(totalVerts * VERT_STRIDE);
    const vertexNormal = new Float32Array(totalVerts * 3); // world-space, 0 = none → flat
    const vertexUV = new Float32Array(totalVerts * 2);
    const triIndex = new Uint32Array(totalTris * 3);
    const vertexMaterial = new Uint32Array(totalVerts);
    let vCursor = 0; // vertex count written
    let tCursor = 0; // triangle count written

    const v = new THREE.Vector3();
    const nrm = new THREE.Vector3();
    const normalMat = new THREE.Matrix3();
    for (const d of draws) {
        const { pos, index, triCount, visibleTriCount, visibleTriIndices, matrices, triMat, uniqueTriMaterial, normal, uv } = d;
        const vCount = pos.count;
        for (const m of matrices) {
            const vBase = vCursor;
            const tagBase = vBase + vCount;
            normalMat.getNormalMatrix(m); // inverse-transpose of the model 3x3
            for (let i = 0; i < vCount; i++) {
                v.fromBufferAttribute(pos, i).applyMatrix4(m);
                const o = (vBase + i) * VERT_STRIDE;
                vertexPos[o] = v.x; vertexPos[o + 1] = v.y; vertexPos[o + 2] = v.z;
                if (normal) {
                    nrm.fromBufferAttribute(normal, i).applyMatrix3(normalMat).normalize();
                    const no = (vBase + i) * 3;
                    vertexNormal[no] = nrm.x; vertexNormal[no + 1] = nrm.y; vertexNormal[no + 2] = nrm.z;
                }
                if (uv) {
                    const uo = (vBase + i) * 2;
                    vertexUV[uo] = uv.getX(i); vertexUV[uo + 1] = uv.getY(i);
                }
            }
            for (let t = 0; t < visibleTriCount; t++) {
                const sourceTri = visibleTriIndices ? visibleTriIndices[t] : t;
                if (sourceTri < 0 || sourceTri >= triCount) continue;
                const sourceA = vBase + (index ? index.getX(sourceTri * 3) : sourceTri * 3);
                const a = uniqueTriMaterial ? tagBase + t : sourceA;
                const b = vBase + (index ? index.getX(sourceTri * 3 + 1) : sourceTri * 3 + 1);
                const c = vBase + (index ? index.getX(sourceTri * 3 + 2) : sourceTri * 3 + 2);
                if (uniqueTriMaterial) {
                    const po = a * VERT_STRIDE;
                    const ps = sourceA * VERT_STRIDE;
                    vertexPos[po] = vertexPos[ps];
                    vertexPos[po + 1] = vertexPos[ps + 1];
                    vertexPos[po + 2] = vertexPos[ps + 2];
                    const no = a * 3;
                    const ns = sourceA * 3;
                    vertexNormal[no] = vertexNormal[ns];
                    vertexNormal[no + 1] = vertexNormal[ns + 1];
                    vertexNormal[no + 2] = vertexNormal[ns + 2];
                    const uo = a * 2;
                    const us = sourceA * 2;
                    vertexUV[uo] = vertexUV[us];
                    vertexUV[uo + 1] = vertexUV[us + 1];
                }
                const to = (tCursor + t) * 3;
                triIndex[to] = a;
                triIndex[to + 1] = b;
                triIndex[to + 2] = c;
                const um = triMat[sourceTri];
                vertexMaterial[a] = um;
                if (!uniqueTriMaterial) {
                    vertexMaterial[b] = um;
                    vertexMaterial[c] = um;
                }
            }
            vCursor += vCount + (uniqueTriMaterial ? visibleTriCount : 0);
            tCursor += visibleTriCount;
        }
    }

    // Build a real BufferGeometry for MeshBVH (single root → clear groups).
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(vertexPos, VERT_STRIDE));
    geometry.setIndex(new THREE.BufferAttribute(triIndex, 1));
    geometry.clearGroups();
    geometry.computeBoundingBox();
    // World-space AABB of the EXACT traced triangle soup. Returned so consumers
    // (SPEEDBALL GI grid auto-fit) bound to what the BVH actually contains — no second
    // scene walk, no dependency on per-object visibility flags. Cloned because the
    // geometry is disposed below.
    const bounds = geometry.boundingBox ? geometry.boundingBox.clone() : null;

    const bvh = new MeshBVH(geometry, { maxLeafTris: 8, indirect: false });
    const roots = bvh._roots;
    if (!Array.isArray(roots) || roots.length === 0) return { error: 'BVH build produced no root' };
    const { nodes: bvhNodes, nodeCount } = flattenBVHRoot(roots[0]);

    // MeshBVH permuted triIndex in place (whole triangles). Per-triangle
    // material in BVH order = material of the triangle's first vertex, which
    // vertexMaterial carries through the permutation.
    const triMaterial = new Uint32Array(totalTris);
    for (let t = 0; t < totalTris; t++) {
        triMaterial[t] = vertexMaterial[triIndex[t * 3]] >>> 0;
    }

    // Extract PBR maps into array textures FIRST — this writes each material's
    // assigned layer index into its uber record ([12..16]) before we pack the
    // materials buffer below.
    const maps = await buildMaterialTextures(THREE, uberList, uberMaterials, TEXTURE_ATLAS_SIZE);

    // Materials buffer
    const materials = new Float32Array(uberList.length * MAT_STRIDE);
    for (let i = 0; i < uberList.length; i++) {
        materials.set(uberList[i], i * MAT_STRIDE);
    }

    // Lights
    const lightRecords = collectLights(THREE, scene, camera);
    const lights = new Float32Array(Math.max(1, lightRecords.length) * LIGHT_STRIDE);
    for (let i = 0; i < lightRecords.length; i++) lights.set(lightRecords[i], i * LIGHT_STRIDE);

    // Environment (equirect)
    const env = scene.userData?.maxjsPathTraceEnvironment?.isTexture
        ? scene.userData.maxjsPathTraceEnvironment
        : (scene.environment?.isTexture ? scene.environment : null);

    geometry.dispose?.();

    // Interleave pos+normal+uv into one GPU storage buffer (8 floats/vertex) so
    // the kernel stays within the 8 storage-buffer budget. Layout per vertex:
    // [px,py,pz, nx,ny,nz, u,v]. Original vertex order (BVH only permuted tris).
    const vertexData = new Float32Array(totalVerts * VERTEX_DATA_STRIDE);
    for (let i = 0; i < totalVerts; i++) {
        const d = i * VERTEX_DATA_STRIDE, p = i * VERT_STRIDE, n = i * 3, u = i * 2;
        vertexData[d] = vertexPos[p]; vertexData[d + 1] = vertexPos[p + 1]; vertexData[d + 2] = vertexPos[p + 2];
        vertexData[d + 3] = vertexNormal[n]; vertexData[d + 4] = vertexNormal[n + 1]; vertexData[d + 5] = vertexNormal[n + 2];
        vertexData[d + 6] = vertexUV[u]; vertexData[d + 7] = vertexUV[u + 1];
    }

    return {
        error: null,
        bounds, // THREE.Box3 | null — world-space AABB of the traced soup
        bvhNodes, nodeCount,
        triIndex, triCount: totalTris,
        vertexData, vertexCount: totalVerts,
        triMaterial,
        materials, materialCount: uberList.length,
        lights, lightCount: lightRecords.length,
        env,
        maps, // { albedo, normal, roughness, metalness, emissive, alpha } DataArrayTexture | null
        strides: { NODE_STRIDE_U32, MAT_STRIDE, LIGHT_STRIDE, VERT_STRIDE, VERTEX_DATA_STRIDE },
    };
}

export { NODE_STRIDE_U32, MAT_STRIDE, LIGHT_STRIDE, VERT_STRIDE, VERTEX_DATA_STRIDE, BYTES_PER_BVH_NODE };
