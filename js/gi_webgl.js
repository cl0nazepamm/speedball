// Texture-backed execution for WebGPURenderer's WebGL2 backend. The GI math
// stays in gi_probes; this module owns only storage, render passes and state.
import * as THREE from 'three/webgpu';
import { Fn, glslFn, uint, uvec2, ivec2, vec4, texture, textureLoad, screenCoordinate, mrt, struct } from 'three/tsl';

const owners = new WeakMap();

// Compact scalar loads avoid expanding address calculations and conditional
// texture flips at every BVH read. CPU data textures are always stored unflipped.
const loadFloat = glslFn(`float giLoadFloat(sampler2D data, uint index) {
    uint pixel = index / 4u;
    uint width = uint(textureSize(data, 0).x);
    return texelFetch(data, ivec2(pixel % width, pixel / width), 0)[index % 4u];
}`);
const loadUint = glslFn(`uint giLoadUint(usampler2D data, uint index) {
    uint pixel = index / 4u;
    uint width = uint(textureSize(data, 0).x);
    return texelFetch(data, ivec2(pixel % width, pixel / width), 0)[index % 4u];
}`);

export function webglProbeSupport(renderer) {
    const backend = renderer?.backend;
    return backend?.isWebGLBackend === true
        && backend.extensions?.has('EXT_color_buffer_float') === true;
}

export function disposeProbeTexture(texture) {
    const owner = owners.get(texture);
    if (owner) owner.dispose();
    else texture?.dispose();
}

export function createWebGLProbeBackend(renderer) {
    const gl = renderer.backend.gl;
    const maxSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);

    function dimensions(width, height) {
        if (width > maxSize || height > maxSize) {
            throw new RangeError(`SPEEDBALL GI: texture ${width}x${height} exceeds WebGL limit ${maxSize}.`);
        }
    }

    function data(array) {
        const width = Math.min(2048, maxSize, Math.max(1, Math.ceil(array.length / 4)));
        const height = Math.max(1, Math.ceil(array.length / (width * 4)));
        dimensions(width, height);
        const padded = new array.constructor(width * height * 4);
        padded.set(array);
        const integer = array instanceof Uint32Array;
        const tex = new THREE.DataTexture(padded, width, height,
            integer ? THREE.RGBAIntegerFormat : THREE.RGBAFormat,
            integer ? THREE.UnsignedIntType : THREE.FloatType);
        tex.needsUpdate = true;
        const sampler = texture(tex).convert('property');
        const fetch = integer ? loadUint : loadFloat;
        const node = {
            element(index) { return fetch(sampler, uint(index)); },
            toReadOnly() { return this; },
        };
        return {
            isGITextureBuffer: true, array, count: array.length, node,
            set needsUpdate(value) { if (value) { padded.set(array); tex.needsUpdate = true; } },
            async read() { return array.slice(); },
            dispose() { tex.dispose(); },
        };
    }

    // Every history group has fixed output and read textures. Copying before a
    // sparse update avoids feedback and keeps all material bindings stable.
    function group(width, height, names, { history = false, half = false, linear = false } = {}) {
        dimensions(width, height);
        const opts = {
            count: names.length, depthBuffer: false, stencilBuffer: false,
            type: half ? THREE.HalfFloatType : THREE.FloatType,
            minFilter: linear ? THREE.LinearFilter : THREE.NearestFilter,
            magFilter: linear ? THREE.LinearFilter : THREE.NearestFilter,
            generateMipmaps: false,
        };
        const target = new THREE.RenderTarget(width, height, opts);
        const previous = history ? new THREE.RenderTarget(width, height, opts) : null;
        // Three r185 clears secondary MRT attachments to (0,0,0,1), even
        // when clearAlpha is zero. Reflection coverage/history needs real zero.
        const clearMaterial = names.length > 1 ? new THREE.NodeMaterial() : null;
        if (clearMaterial) {
            clearMaterial.depthTest = clearMaterial.depthWrite = clearMaterial.toneMapped = false;
            clearMaterial.blending = THREE.NoBlending;
            clearMaterial.fragmentNode = mrt(Object.fromEntries(names.map(name => [name, vec4(0)])));
        }
        let disposed = false;
        const owner = {
            target, previous, initialized: false,
            clearQuad: clearMaterial ? new THREE.QuadMesh(clearMaterial) : null,
            dispose() {
                if (disposed) return;
                disposed = true;
                target.dispose();
                previous?.dispose();
                clearMaterial?.dispose();
            },
        };
        target.textures.forEach((tex, i) => { tex.name = names[i]; owners.set(tex, owner); });
        previous?.textures.forEach((tex, i) => { tex.name = names[i]; });
        // Allocate as render targets before a receiver can sample them. Otherwise
        // Three first allocates a standalone texture, then replaces it on the
        // first target clear, leaving WebGL's already-cached sampler stale.
        const state = THREE.RendererUtils.resetRendererState(renderer);
        try {
            renderer.setClearColor(0, 0);
            initialize(owner);
        } finally {
            THREE.RendererUtils.restoreRendererState(renderer, state);
        }
        return owner;
    }

    function buffer(owner, attachment, count, components, tile = 1, tilesX = owner.target.width) {
        const texture = owner.target.textures[attachment];
        const readTexture = owner.previous?.textures[attachment] || texture;
        function coord(index) {
            const pixel = uint(index).div(uint(components));
            const probe = pixel.div(uint(tile * tile));
            const local = pixel.mod(uint(tile * tile));
            return ivec2(probe.mod(uint(tilesX)).mul(uint(tile)).add(local.mod(uint(tile))),
                probe.div(uint(tilesX)).mul(uint(tile)).add(local.div(uint(tile))));
        }
        const nodeFor = tex => ({
            element(index) { return textureLoad(tex, coord(index)).element(uint(index).mod(uint(components))); },
            toReadOnly() { return this; },
        });
        return {
            isGITextureBuffer: true, owner, texture, count,
            node: nodeFor(readTexture), readNode: nodeFor(texture),
            async read() {
                const { width, height } = owner.target;
                const pixels = await renderer.readRenderTargetPixelsAsync(owner.target, 0, 0, width, height, attachment);
                const values = new Float32Array(count);
                for (let i = 0; i < count; i++) {
                    const pixel = Math.floor(i / components);
                    const probe = Math.floor(pixel / (tile * tile)), local = pixel % (tile * tile);
                    const x = (probe % tilesX) * tile + local % tile;
                    const y = Math.floor(probe / tilesX) * tile + Math.floor(local / tile);
                    values[i] = pixels[((height - 1 - y) * width + x) * 4 + i % components];
                }
                return values;
            },
            dispose() { owner.dispose(); },
        };
    }

    function pass(owner, shader, { history = false } = {}) {
        const material = new THREE.NodeMaterial();
        material.name = `SPEEDBALL GI ${owner.target.textures.map(t => t.name).join('/')}`;
        material.depthTest = material.depthWrite = false;
        material.blending = THREE.NoBlending;
        material.toneMapped = false;
        const names = owner.target.textures.map(tex => tex.name);
        const Result = struct(Object.fromEntries(names.map(name => [name, 'vec4'])));
        const result = Fn(() => {
            // TSL screenCoordinate uses top-left coordinates on both backends;
            // textureLoad handles the WebGL render-target Y flip in the same way.
            const pixel = uvec2(screenCoordinate.xy).toVar();
            return Result(shader(pixel));
        })();
        material.fragmentNode = mrt(Object.fromEntries(names.map(name => [name, result.get(name)])));
        const quad = new THREE.QuadMesh(material);
        return {
            count: 1,
            run() {
                if (this.count === 0) return;
                initialize(owner);
                if (history) {
                    owner.target.textures.forEach((tex, i) => {
                        renderer.copyTextureToTexture(tex, owner.previous.textures[i]);
                    });
                }
                renderer.setRenderTarget(owner.target);
                quad.render(renderer);
            },
            dispose() { material.dispose(); },
        };
    }

    function initialize(owner) {
        if (owner.initialized) return;
        renderer.setRenderTarget(owner.target);
        renderer.clear();
        owner.clearQuad?.render(renderer);
        if (owner.previous) {
            renderer.setRenderTarget(owner.previous);
            renderer.clear();
            owner.clearQuad?.render(renderer);
        }
        owner.initialized = true;
    }

    function clear(...groups) {
        return {
            run() {
                for (const owner of new Set(groups.filter(Boolean))) {
                    owner.initialized = false;
                    initialize(owner);
                }
            },
            dispose() {},
        };
    }

    async function dispatch(passes) {
        const state = THREE.RendererUtils.resetRendererState(renderer);
        try {
            renderer.autoClear = false;
            renderer.setClearColor(0, 0);
            for (const p of Array.isArray(passes) ? passes : [passes]) p?.run();
        } finally {
            THREE.RendererUtils.restoreRendererState(renderer, state);
        }
    }

    return { data, group, buffer, pass, clear, dispatch, initialize, owner: texture => owners.get(texture), maxSize };
}
