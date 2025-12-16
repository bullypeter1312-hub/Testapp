import sharp from 'sharp';

import type { PipelineType, pipeline as pipelineFactory } from '@xenova/transformers';

import type {
  ImageUpscaleOptions,
  ImageUpscaleResult,
  ImageUpscaler,
  UpscaleFactor,
} from '../pipeline/types';

type PipelineFactory = typeof pipelineFactory;
type PipelineFactoryOptions = Parameters<PipelineFactory>[2];

type TransformersPipeline = (input: unknown, options?: Record<string, unknown>) => Promise<unknown>;

type RealEsrganModelVariant = 'standard' | 'face';

export interface RealEsrganUpscalerOptions {
  /**
   * Maximum number of input pixels accepted by the upscaler.
   *
   * Real-ESRGAN is memory hungry: output pixels grow by scale^2.
   * We cap input pixels to avoid runaway memory usage.
   */
  maxInputPixels?: number;
}

interface RealEsrganPipelineConfig {
  task: PipelineType;
  modelId: string;
  scale: UpscaleFactor;
  faceEnhance: boolean;
  pipelineOptions: PipelineFactoryOptions;
  inferenceOptions: Record<string, unknown>;
}

const DEFAULT_MAX_INPUT_PIXELS = 4096 * 4096;

const TASK: PipelineType = 'image-to-image';

// These model IDs are intentionally centralized so we can add/replace checkpoints
// without touching the rest of the upscaler implementation.
const MODEL_IDS: Record<
  UpscaleFactor,
  Record<RealEsrganModelVariant, { modelId: string; inferenceOptions?: Record<string, unknown> }>
> = {
  2: {
    standard: { modelId: 'Xenova/Real-ESRGAN-x2plus' },
    face: {
      modelId: 'Xenova/Real-ESRGAN-x2plus',
      inferenceOptions: { face_enhance: true },
    },
  },
  4: {
    standard: { modelId: 'Xenova/Real-ESRGAN-x4plus' },
    face: {
      modelId: 'Xenova/Real-ESRGAN-x4plus',
      inferenceOptions: { face_enhance: true },
    },
  },
};

const pipelineCache = new Map<string, Promise<TransformersPipeline>>();

function buildConfig(options: ImageUpscaleOptions): RealEsrganPipelineConfig {
  const faceEnhance = Boolean(options.faceEnhance);
  const variant: RealEsrganModelVariant = faceEnhance ? 'face' : 'standard';
  const entry = MODEL_IDS[options.scale][variant];

  return {
    task: TASK,
    modelId: entry.modelId,
    scale: options.scale,
    faceEnhance,
    pipelineOptions: {
      // Loading quantized weights significantly reduces memory usage.
      quantized: true,
    },
    inferenceOptions: {
      ...(entry.inferenceOptions ?? {}),
    },
  };
}

async function getPipeline(config: RealEsrganPipelineConfig): Promise<TransformersPipeline> {
  const cacheKey = `${config.task}:${config.modelId}:${config.scale}x:${config.faceEnhance}`;
  const cached = pipelineCache.get(cacheKey);
  if (cached) return cached;

  const pipelinePromise = (async () => {
    const { pipeline } = await import('@xenova/transformers');

    // NOTE: @xenova/transformers caches model files on disk by default.
    // This upscaler additionally caches the in-memory pipeline so weights are
    // loaded once per process and reused across requests.
    return pipeline(config.task, config.modelId, config.pipelineOptions) as unknown as TransformersPipeline;
  })();

  pipelineCache.set(cacheKey, pipelinePromise);

  try {
    return await pipelinePromise;
  } catch (err) {
    pipelineCache.delete(cacheKey);
    throw err;
  }
}

export function resetRealEsrganModelCacheForTests() {
  pipelineCache.clear();
}

async function normalizeInput(buffer: Buffer, maxInputPixels: number) {
  try {
    // We first read metadata without a pixel limit so we can return a nicer error
    // message (and avoid starting any expensive processing).
    const metaProbe = sharp(buffer, { failOnError: true, limitInputPixels: false });
    const meta = await metaProbe.metadata();

    if (!meta.width || !meta.height) {
      throw new Error('Unable to determine input image dimensions');
    }

    const inputPixels = meta.width * meta.height;

    if (inputPixels > maxInputPixels) {
      throw new Error(
        `Input image too large: ${meta.width}x${meta.height} (${inputPixels} pixels) exceeds limit (${maxInputPixels} pixels)`,
      );
    }

    // Recreate a new sharp pipeline with the pixel limit enforced.
    const img = sharp(buffer, { failOnError: true, limitInputPixels: maxInputPixels });

    return {
      width: meta.width,
      height: meta.height,
      // Normalise orientation + colorspace + output format for deterministic inference.
      buffer: await img
        .rotate()
        .toColorspace('srgb')
        .withIccProfile('srgb')
        .png({ compressionLevel: 9, force: true })
        .toBuffer(),
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    throw new Error(`Failed to decode/normalise input image: ${message}`, { cause: err });
  }
}

async function normalizeOutput(buffer: Buffer) {
  // Even though the model output should already be sRGB-ish, we enforce a stable
  // output format + colorspace for downstream consumers.
  return sharp(buffer, { failOnError: true })
    .toColorspace('srgb')
    .withIccProfile('srgb')
    .png({ compressionLevel: 9, force: true })
    .toBuffer();
}

async function coercePipelineOutputToBuffer(output: unknown): Promise<Buffer> {
  if (Buffer.isBuffer(output)) return output;

  if (output instanceof Uint8Array) {
    return Buffer.from(output);
  }

  if (Array.isArray(output) && output.length > 0) {
    return coercePipelineOutputToBuffer(output[0]);
  }

  if (output && typeof output === 'object') {
    const maybeWithImage = output as { image?: unknown; toBuffer?: unknown };

    if (maybeWithImage.image) {
      return coercePipelineOutputToBuffer(maybeWithImage.image);
    }

    if (typeof maybeWithImage.toBuffer === 'function') {
      const result = await (maybeWithImage.toBuffer as () => Promise<unknown>)();
      return coercePipelineOutputToBuffer(result);
    }
  }

  throw new Error('Unsupported Real-ESRGAN pipeline output type');
}

export class RealEsrganUpscaler implements ImageUpscaler {
  private readonly maxInputPixels: number;

  constructor(options: RealEsrganUpscalerOptions = {}) {
    this.maxInputPixels = options.maxInputPixels ?? DEFAULT_MAX_INPUT_PIXELS;
  }

  async upscale(input: Buffer, options: ImageUpscaleOptions): Promise<ImageUpscaleResult> {
    const config = buildConfig(options);

    const startedAt = performance.now();

    try {
      const normalizedInput = await normalizeInput(input, this.maxInputPixels);
      const srPipeline = await getPipeline(config);

      const rawOutput = await srPipeline(normalizedInput.buffer, config.inferenceOptions);
      const outputBuffer = await coercePipelineOutputToBuffer(rawOutput);
      const normalizedOutput = await normalizeOutput(outputBuffer);

      const finishedAt = performance.now();

      return {
        buffer: normalizedOutput,
        metadata: {
          scale: config.scale,
          faceEnhance: config.faceEnhance,
          runtimeMs: Math.round(finishedAt - startedAt),
          modelId: config.modelId,
          inputWidth: normalizedInput.width,
          inputHeight: normalizedInput.height,
        },
      };
    } catch (err) {
      const finishedAt = performance.now();
      const message = err instanceof Error ? err.message : String(err);

      throw new Error(
        `Real-ESRGAN upscale failed (scale=${config.scale}x, faceEnhance=${config.faceEnhance}, runtimeMs=${Math.round(
          finishedAt - startedAt,
        )}): ${message}`,
        { cause: err },
      );
    }
  }
}
