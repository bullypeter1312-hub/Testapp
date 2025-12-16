export type UpscaleFactor = 2 | 4;

export interface ImageUpscaleOptions {
  scale: UpscaleFactor;
  faceEnhance?: boolean;
}

export interface ImageUpscaleMetadata {
  scale: UpscaleFactor;
  faceEnhance: boolean;
  runtimeMs: number;

  // Optional fields for debugging/observability.
  modelId?: string;
  inputWidth?: number;
  inputHeight?: number;
}

export interface ImageUpscaleResult {
  buffer: Buffer;
  metadata: ImageUpscaleMetadata;
}

export interface ImageUpscaler {
  upscale(input: Buffer, options: ImageUpscaleOptions): Promise<ImageUpscaleResult>;
}
