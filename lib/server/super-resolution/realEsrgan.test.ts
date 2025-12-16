import { beforeEach, describe, expect, it, vi } from 'vitest';

import { RealEsrganUpscaler, resetRealEsrganModelCacheForTests } from './realEsrgan';

const pipelineMock = vi.fn();

vi.mock('@xenova/transformers', () => ({
  pipeline: pipelineMock,
}));

const ONE_BY_ONE_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/6X+0QAAAABJRU5ErkJggg==',
  'base64',
);

describe('RealEsrganUpscaler', () => {
  beforeEach(() => {
    resetRealEsrganModelCacheForTests();
    pipelineMock.mockReset();
  });

  it('selects the 2x checkpoint and caches the pipeline', async () => {
    const srFn = vi.fn().mockResolvedValue(ONE_BY_ONE_PNG);
    pipelineMock.mockResolvedValue(srFn);

    const upscaler = new RealEsrganUpscaler({ maxInputPixels: 1000 });

    const result1 = await upscaler.upscale(ONE_BY_ONE_PNG, { scale: 2 });
    const result2 = await upscaler.upscale(ONE_BY_ONE_PNG, { scale: 2 });

    expect(pipelineMock).toHaveBeenCalledTimes(1);
    expect(pipelineMock).toHaveBeenCalledWith('image-to-image', 'Xenova/Real-ESRGAN-x2plus', {
      quantized: true,
    });

    expect(srFn).toHaveBeenCalledTimes(2);
    expect(srFn).toHaveBeenCalledWith(expect.any(Buffer), {});

    expect(result1.metadata.scale).toBe(2);
    expect(result1.metadata.faceEnhance).toBe(false);
    expect(result1.metadata.runtimeMs).toEqual(expect.any(Number));
    expect(result1.buffer).toBeInstanceOf(Buffer);

    expect(result2.metadata.scale).toBe(2);
  });

  it('branches into faceEnhance mode for 4x', async () => {
    const srFn = vi.fn().mockResolvedValue(ONE_BY_ONE_PNG);
    pipelineMock.mockResolvedValue(srFn);

    const upscaler = new RealEsrganUpscaler({ maxInputPixels: 1000 });

    await upscaler.upscale(ONE_BY_ONE_PNG, { scale: 4, faceEnhance: true });

    expect(pipelineMock).toHaveBeenCalledTimes(1);
    expect(pipelineMock).toHaveBeenCalledWith('image-to-image', 'Xenova/Real-ESRGAN-x4plus', {
      quantized: true,
    });

    expect(srFn).toHaveBeenCalledWith(expect.any(Buffer), { face_enhance: true });
  });

  it('wraps and rethrows inference errors with useful context', async () => {
    const innerError = new Error('boom');
    const srFn = vi.fn().mockRejectedValue(innerError);
    pipelineMock.mockResolvedValue(srFn);

    const upscaler = new RealEsrganUpscaler({ maxInputPixels: 1000 });

    try {
      await upscaler.upscale(ONE_BY_ONE_PNG, { scale: 2 });
      throw new Error('Expected upscale() to throw');
    } catch (err) {
      expect(err).toBeInstanceOf(Error);
      const e = err as Error & { cause?: unknown };
      expect(e.message).toMatch(/Real-ESRGAN upscale failed/);
      expect(e.message).toMatch(/scale=2x/);
      expect(e.message).toMatch(/faceEnhance=false/);
      expect(e.message).toMatch(/boom/);
      expect(e.cause).toBe(innerError);
    }
  });
});
