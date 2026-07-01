import * as THREE from "three/webgpu";
import {
  NodeUpdateType,
  convertToTexture,
  nodeObject,
  screenUV,
  texture,
} from "three/tsl";

import { FilmPipeline } from "./film.js";
import { InfraredPipeline } from "./infrared.js";
import { Pipeline } from "./pipeline.js";

const DEFAULT_TARGET_OPTIONS = {
  type: THREE.HalfFloatType,
  minFilter: THREE.LinearFilter,
  magFilter: THREE.LinearFilter,
  depthBuffer: false,
  colorSpace: THREE.NoColorSpace,
};

const _size = new THREE.Vector2();

function finitePositive(value, fallback) {
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function textureSize(tex, fallback) {
  const image = tex?.image || tex?.source?.data || null;
  return {
    width: finitePositive(image?.width || image?.videoWidth, fallback.width),
    height: finitePositive(image?.height || image?.videoHeight, fallback.height),
  };
}

function sourceRenderTarget(node) {
  return node?.renderTarget || node?.passNode?.renderTarget || null;
}

function rendererDrawingBufferSize(renderer) {
  if (typeof renderer.getDrawingBufferSize === "function") {
    renderer.getDrawingBufferSize(_size);
  } else {
    renderer.getSize(_size);
    const pixelRatio = typeof renderer.getPixelRatio === "function"
      ? renderer.getPixelRatio()
      : 1;
    _size.multiplyScalar(pixelRatio);
  }
  return {
    width: finitePositive(Math.round(_size.x), 1),
    height: finitePositive(Math.round(_size.y), 1),
  };
}

function isEffectInstance(value) {
  return value && typeof value.renderTexture === "function";
}

function normalizeEffectArgs(effectOrOptions, options) {
  if (
    effectOrOptions === null
    || effectOrOptions === undefined
    || isEffectInstance(effectOrOptions)
  ) {
    return {
      effect: effectOrOptions || null,
      options: options || {},
    };
  }

  return {
    effect: effectOrOptions.effect || effectOrOptions.pipeline || null,
    options: effectOrOptions,
  };
}

function defaultRenderOptions(options, frame, node, effect) {
  if (typeof options === "function") return options(frame, node, effect) || {};
  return options || {};
}

/**
 * TSL adapter for running a PowerShot-compatible effect as a THREE.RenderPipeline stage.
 *
 * Pass the output node you already had, plus any configured effect instance
 * with setSize() and renderTexture(). The node renders the input graph to a
 * texture, runs effect.renderTexture(), and exposes the result as a normal vec4
 * output node.
 */
export class EffectPassNode extends THREE.Node {
  constructor(inputNode, effect = null, options = {}) {
    super("vec4");

    this.inputNode = nodeObject(inputNode);
    this.inputTextureNode = convertToTexture(this.inputNode);
    this.outputTarget = new THREE.RenderTarget(1, 1, {
      ...DEFAULT_TARGET_OPTIONS,
      ...(options.targetOptions || {}),
    });
    this.outputTextureNode = texture(this.outputTarget.texture, screenUV);

    this.effect = effect;
    this.createEffect = typeof options.createEffect === "function"
      ? options.createEffect
      : ((renderer) => new Pipeline(renderer));
    this.configureEffect = typeof options.configureEffect === "function"
      ? options.configureEffect
      : (typeof options.configure === "function" ? options.configure : null);
    this.renderOptions = options.renderOptions || null;
    this.autoSize = options.autoSize !== false;
    this.resolutionScale = finitePositive(options.resolutionScale, 1);
    this.frame = 0;
    this.frameProvider = typeof options.frame === "function" ? options.frame : null;
    this.ownsEffect = false;
    this.effectConfigured = false;
    this.updateBeforeType = NodeUpdateType.RENDER;
  }

  get pipeline() {
    return this.effect;
  }

  set pipeline(effect) {
    this.setEffect(effect);
  }

  get ownsPipeline() {
    return this.ownsEffect;
  }

  set ownsPipeline(value) {
    this.ownsEffect = Boolean(value);
  }

  setEffect(effect) {
    if (this.ownsEffect) this.effect?.dispose?.();
    this.effect = effect;
    this.ownsEffect = false;
    this.effectConfigured = false;
    return this;
  }

  setPipeline(effect) {
    return this.setEffect(effect);
  }

  setResolutionScale(scale) {
    this.resolutionScale = finitePositive(scale, 1);
    return this;
  }

  setup(builder) {
    const props = builder.getNodeProperties(this);
    props.inputTextureNode = this.inputTextureNode;
    props.outputTextureNode = this.outputTextureNode;
    return this.outputTextureNode;
  }

  _ensureEffect(renderer) {
    if (!this.effect) {
      this.effect = this.createEffect(renderer, this);
      this.ownsEffect = true;
      this.effectConfigured = false;
    }
    if (!isEffectInstance(this.effect)) {
      throw new Error("EffectPassNode requires an effect with renderTexture(inputTexture, frame, options).");
    }
    if (!this.effectConfigured) {
      this.configureEffect?.(this.effect, renderer, this);
      this.effectConfigured = true;
    }
    return this.effect;
  }

  _sourceTexture(renderer) {
    this.inputTextureNode.updateTexture?.();

    const target = sourceRenderTarget(this.inputTextureNode) || sourceRenderTarget(this.inputNode);
    return this.inputTextureNode.value
      || target?.texture
      || this.inputNode.value
      || null;
  }

  _sourceSize(renderer, sourceTexture) {
    const fallback = rendererDrawingBufferSize(renderer);
    const target = sourceRenderTarget(this.inputTextureNode) || sourceRenderTarget(this.inputNode);
    if (target) {
      return {
        width: finitePositive(target.width, fallback.width),
        height: finitePositive(target.height, fallback.height),
      };
    }
    return textureSize(sourceTexture, fallback);
  }

  _setWorkSize(renderer, sourceTexture, effect) {
    const sourceSize = this._sourceSize(renderer, sourceTexture);
    const scale = finitePositive(this.resolutionScale, 1);
    const width = Math.max(1, Math.round(sourceSize.width * scale));
    const height = Math.max(1, Math.round(sourceSize.height * scale));

    if (this.outputTarget.width !== width || this.outputTarget.height !== height) {
      this.outputTarget.setSize(width, height);
    }
    if (this.autoSize && typeof effect.setSize === "function") effect.setSize(width, height);
  }

  updateBefore(frame) {
    const renderer = frame?.renderer;
    if (!renderer) return false;

    const effect = this._ensureEffect(renderer);
    const sourceTexture = this._sourceTexture(renderer);
    if (!sourceTexture) return false;

    this._setWorkSize(renderer, sourceTexture, effect);

    const previousTarget = renderer.getRenderTarget?.() ?? null;
    const previousToneMapping = renderer.toneMapping;
    const previousExposure = renderer.toneMappingExposure;
    const previousOutputColorSpace = renderer.outputColorSpace;

    try {
      renderer.toneMapping = THREE.NoToneMapping;
      renderer.toneMappingExposure = 1.0;
      renderer.outputColorSpace = previousOutputColorSpace;

      const pipelineFrame = this.frameProvider
        ? this.frameProvider(frame, this)
        : frame.frameId ?? this.frame;
      const rendered = effect.renderTexture(sourceTexture, pipelineFrame, {
        ...defaultRenderOptions(this.renderOptions, frame, this, effect),
        outputTarget: this.outputTarget,
      });
      this.frame += 1;
      return rendered;
    } finally {
      renderer.setRenderTarget(previousTarget);
      renderer.toneMapping = previousToneMapping;
      renderer.toneMappingExposure = previousExposure;
      renderer.outputColorSpace = previousOutputColorSpace;
    }
  }

  dispose() {
    this.outputTarget.dispose();
    if (this.ownsEffect) this.effect?.dispose?.();
    this.effect = null;
  }
}

export class PowerShotPassNode extends EffectPassNode {
  constructor(inputNode, pipeline = null, options = {}) {
    super(inputNode, pipeline, {
      createEffect: (renderer) => new Pipeline(renderer),
      ...options,
    });
  }
}

export class FilmPassNode extends EffectPassNode {
  constructor(inputNode, pipeline = null, options = {}) {
    super(inputNode, pipeline, {
      createEffect: (renderer) => new FilmPipeline(renderer),
      ...options,
    });
  }
}

export class InfraredPassNode extends EffectPassNode {
  constructor(inputNode, pipeline = null, options = {}) {
    super(inputNode, pipeline, {
      createEffect: (renderer) => new InfraredPipeline(renderer),
      ...options,
    });
  }
}

export function effectPass(inputNode, effectOrOptions = null, options = {}) {
  const normalized = normalizeEffectArgs(effectOrOptions, options);
  return new EffectPassNode(inputNode, normalized.effect, normalized.options);
}

export function powerShotPass(inputNode, pipelineOrOptions = null, options = {}) {
  const normalized = normalizeEffectArgs(pipelineOrOptions, options);
  return new PowerShotPassNode(inputNode, normalized.effect, normalized.options);
}

export function filmPass(inputNode, pipelineOrOptions = null, options = {}) {
  const normalized = normalizeEffectArgs(pipelineOrOptions, options);
  return new FilmPassNode(inputNode, normalized.effect, normalized.options);
}

export function infraredPass(inputNode, pipelineOrOptions = null, options = {}) {
  const normalized = normalizeEffectArgs(pipelineOrOptions, options);
  return new InfraredPassNode(inputNode, normalized.effect, normalized.options);
}
