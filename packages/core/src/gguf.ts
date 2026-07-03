import type { Fetcher, HfAuth } from "./types.js";
import { authHeaders, isUrl } from "./hf.js";

export type GgufMetadata = {
  url: string;
  fileSize?: number | undefined;
  version: number;
  tensorCount: number;
  metadata: Record<string, unknown>;
  tensorBytes?: number | undefined;
  unknownTensorTypes: number[];
  sources: string[];
};

type TensorTypeSpec = { blockSize: number; typeSize: number; name: string };

const GGML_TYPES: Record<number, TensorTypeSpec> = {
  0: { name: "F32", blockSize: 1, typeSize: 4 },
  1: { name: "F16", blockSize: 1, typeSize: 2 },
  2: { name: "Q4_0", blockSize: 32, typeSize: 18 },
  3: { name: "Q4_1", blockSize: 32, typeSize: 20 },
  6: { name: "Q5_0", blockSize: 32, typeSize: 22 },
  7: { name: "Q5_1", blockSize: 32, typeSize: 24 },
  8: { name: "Q8_0", blockSize: 32, typeSize: 34 },
  9: { name: "Q8_1", blockSize: 32, typeSize: 40 },
  10: { name: "Q2_K", blockSize: 256, typeSize: 84 },
  11: { name: "Q3_K", blockSize: 256, typeSize: 110 },
  12: { name: "Q4_K", blockSize: 256, typeSize: 144 },
  13: { name: "Q5_K", blockSize: 256, typeSize: 176 },
  14: { name: "Q6_K", blockSize: 256, typeSize: 210 },
  15: { name: "Q8_K", blockSize: 256, typeSize: 292 },
  16: { name: "IQ2_XXS", blockSize: 256, typeSize: 66 },
  17: { name: "IQ2_XS", blockSize: 256, typeSize: 74 },
  18: { name: "IQ3_XXS", blockSize: 256, typeSize: 98 },
  19: { name: "IQ1_S", blockSize: 256, typeSize: 50 },
  20: { name: "IQ4_NL", blockSize: 32, typeSize: 18 },
  21: { name: "IQ3_S", blockSize: 256, typeSize: 110 },
  22: { name: "IQ2_S", blockSize: 256, typeSize: 82 },
  23: { name: "IQ4_XS", blockSize: 256, typeSize: 136 },
  24: { name: "I8", blockSize: 1, typeSize: 1 },
  25: { name: "I16", blockSize: 1, typeSize: 2 },
  26: { name: "I32", blockSize: 1, typeSize: 4 },
  27: { name: "I64", blockSize: 1, typeSize: 8 },
  28: { name: "F64", blockSize: 1, typeSize: 8 },
  29: { name: "IQ1_M", blockSize: 256, typeSize: 56 },
  30: { name: "BF16", blockSize: 1, typeSize: 2 },
  31: { name: "Q4_0_4_4", blockSize: 32, typeSize: 18 },
  32: { name: "Q4_0_4_8", blockSize: 32, typeSize: 18 },
  33: { name: "Q4_0_8_8", blockSize: 32, typeSize: 18 },
  34: { name: "TQ1_0", blockSize: 256, typeSize: 54 },
  35: { name: "TQ2_0", blockSize: 256, typeSize: 66 }
};

const GGML_TYPE_BY_NAME: Record<string, TensorTypeSpec> = Object.fromEntries(
  Object.values(GGML_TYPES).map((spec) => [spec.name.toLowerCase(), spec])
);

export function resolveGgufUrl(source: string): string {
  if (isUrl(source)) return source;
  const [repo, file] = source.split("::");
  if (!repo || !file) {
    throw new Error("GGUF Hugging Face file sources must use repo::path/to/model.gguf");
  }
  return `https://huggingface.co/${repo}/resolve/main/${file}`;
}

export async function fetchGgufMetadata(
  source: string,
  auth?: HfAuth,
  fetcher: Fetcher = fetch
): Promise<GgufMetadata> {
  const url = resolveGgufUrl(source);
  const fileSize = await headSize(fetcher, url, auth);
  const sizes = [4 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024];
  let lastError: unknown;
  for (const size of sizes) {
    try {
      const bytes = await rangeBytes(fetcher, url, size, auth);
      return parseGguf(bytes, url, fileSize);
    } catch (error) {
      lastError = error;
      if (!(error instanceof RangeTooSmallError)) break;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Failed to parse GGUF metadata");
}

async function headSize(fetcher: Fetcher, url: string, auth?: HfAuth): Promise<number | undefined> {
  const res = await fetcher(url, { method: "HEAD", headers: authHeaders(auth) });
  if (!res.ok) return undefined;
  const raw = res.headers.get("x-linked-size") ?? res.headers.get("content-length");
  const size = raw ? Number(raw) : undefined;
  return size && Number.isFinite(size) ? size : undefined;
}

async function rangeBytes(
  fetcher: Fetcher,
  url: string,
  size: number,
  auth?: HfAuth
): Promise<ArrayBuffer> {
  const headers = { ...authHeaders(auth), range: `bytes=0-${size - 1}` };
  const res = await fetcher(url, { headers });
  if (res.status !== 206) {
    throw new Error("GGUF source must support HTTP range requests.");
  }
  return res.arrayBuffer();
}

class RangeTooSmallError extends Error {}

class Reader {
  private offset = 0;

  constructor(private readonly view: DataView) {}

  position(): number {
    return this.offset;
  }

  private ensure(bytes: number): void {
    if (this.offset + bytes > this.view.byteLength) {
      throw new RangeTooSmallError("GGUF metadata exceeds current range read.");
    }
  }

  u8(): number {
    this.ensure(1);
    return this.view.getUint8(this.offset++);
  }

  i8(): number {
    this.ensure(1);
    return this.view.getInt8(this.offset++);
  }

  u16(): number {
    this.ensure(2);
    const value = this.view.getUint16(this.offset, true);
    this.offset += 2;
    return value;
  }

  i16(): number {
    this.ensure(2);
    const value = this.view.getInt16(this.offset, true);
    this.offset += 2;
    return value;
  }

  u32(): number {
    this.ensure(4);
    const value = this.view.getUint32(this.offset, true);
    this.offset += 4;
    return value;
  }

  i32(): number {
    this.ensure(4);
    const value = this.view.getInt32(this.offset, true);
    this.offset += 4;
    return value;
  }

  f32(): number {
    this.ensure(4);
    const value = this.view.getFloat32(this.offset, true);
    this.offset += 4;
    return value;
  }

  u64(): number {
    this.ensure(8);
    const value = this.view.getBigUint64(this.offset, true);
    this.offset += 8;
    const asNumber = Number(value);
    if (!Number.isSafeInteger(asNumber)) throw new Error("GGUF integer exceeds JS safe range.");
    return asNumber;
  }

  i64(): number {
    this.ensure(8);
    const value = this.view.getBigInt64(this.offset, true);
    this.offset += 8;
    const asNumber = Number(value);
    if (!Number.isSafeInteger(asNumber)) throw new Error("GGUF integer exceeds JS safe range.");
    return asNumber;
  }

  f64(): number {
    this.ensure(8);
    const value = this.view.getFloat64(this.offset, true);
    this.offset += 8;
    return value;
  }

  string(): string {
    const length = this.u64();
    this.ensure(length);
    const bytes = new Uint8Array(this.view.buffer, this.view.byteOffset + this.offset, length);
    this.offset += length;
    return new TextDecoder().decode(bytes);
  }
}

export function parseGguf(buffer: ArrayBuffer, url = "memory://gguf", fileSize?: number): GgufMetadata {
  const reader = new Reader(new DataView(buffer));
  const magic = String.fromCharCode(reader.u8(), reader.u8(), reader.u8(), reader.u8());
  if (magic !== "GGUF") throw new Error("Invalid GGUF magic.");
  const version = reader.u32();
  const tensorCount = reader.u64();
  const metadataCount = reader.u64();
  const metadata: Record<string, unknown> = {};

  for (let i = 0; i < metadataCount; i++) {
    const key = reader.string();
    const type = reader.u32();
    metadata[key] = readValue(reader, type);
  }

  let tensorBytes = 0;
  const unknownTensorTypes = new Set<number>();
  for (let i = 0; i < tensorCount; i++) {
    reader.string();
    const dimCount = reader.u32();
    const dims: number[] = [];
    for (let dim = 0; dim < dimCount; dim++) dims.push(reader.u64());
    const type = reader.u32();
    reader.u64();
    const spec = GGML_TYPES[type];
    if (!spec) {
      unknownTensorTypes.add(type);
      continue;
    }
    const elements = dims.reduce((product, dim) => product * dim, 1);
    tensorBytes += Math.ceil(elements / spec.blockSize) * spec.typeSize;
  }

  return {
    url,
    fileSize,
    version,
    tensorCount,
    metadata,
    tensorBytes: unknownTensorTypes.size === 0 ? tensorBytes : undefined,
    unknownTensorTypes: [...unknownTensorTypes],
    sources: [url]
  };
}

function readValue(reader: Reader, type: number): unknown {
  switch (type) {
    case 0:
      return reader.u8();
    case 1:
      return reader.i8();
    case 2:
      return reader.u16();
    case 3:
      return reader.i16();
    case 4:
      return reader.u32();
    case 5:
      return reader.i32();
    case 6:
      return reader.f32();
    case 7:
      return reader.u8() !== 0;
    case 8:
      return reader.string();
    case 9: {
      const itemType = reader.u32();
      const length = reader.u64();
      const values: unknown[] = [];
      for (let i = 0; i < length; i++) values.push(readValue(reader, itemType));
      return values;
    }
    case 10:
      return reader.u64();
    case 11:
      return reader.i64();
    case 12:
      return reader.f64();
    default:
      throw new Error(`Unsupported GGUF metadata type: ${type}`);
  }
}

export function ggufNumber(metadata: GgufMetadata, key: string): number | undefined {
  const value = metadata.metadata[key];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

export function ggufString(metadata: GgufMetadata, key: string): string | undefined {
  const value = metadata.metadata[key];
  return typeof value === "string" ? value : undefined;
}

// llama.cpp's llama_ftype enum, stored in GGUF as general.file_type. Describes the overall weight
// quantization of the file (weights are per-tensor quantized, so this is the representative label
// rather than a single dtype). Deprecated/removed values fall back to a generic label.
const GGUF_FILE_TYPES: Record<number, string> = {
  0: "F32",
  1: "F16",
  2: "Q4_0",
  3: "Q4_1",
  7: "Q8_0",
  8: "Q5_0",
  9: "Q5_1",
  10: "Q2_K",
  11: "Q3_K_S",
  12: "Q3_K_M",
  13: "Q3_K_L",
  14: "Q4_K_S",
  15: "Q4_K_M",
  16: "Q5_K_S",
  17: "Q5_K_M",
  18: "Q6_K",
  19: "IQ2_XXS",
  20: "IQ2_XS",
  21: "Q2_K_S",
  22: "IQ3_XS",
  23: "IQ3_XXS",
  24: "IQ1_S",
  25: "IQ4_NL",
  26: "IQ3_S",
  27: "IQ3_M",
  28: "IQ2_S",
  29: "IQ2_M",
  30: "IQ4_XS",
  31: "IQ1_M",
  32: "BF16",
  33: "Q4_0_4_4",
  34: "Q4_0_4_8",
  35: "Q4_0_8_8",
  36: "TQ1_0",
  37: "TQ2_0"
};

export function ggufFileTypeName(metadata: GgufMetadata): string | undefined {
  const value = ggufNumber(metadata, "general.file_type");
  if (value === undefined) return undefined;
  return GGUF_FILE_TYPES[value] ?? `file_type ${value}`;
}

export function cacheTypeBytes(cacheType: string): number | undefined {
  const normalized = cacheType.toLowerCase();
  const spec = GGML_TYPE_BY_NAME[normalized];
  if (!spec) return undefined;
  // Exact bytes-per-element for block-quantized types (e.g. Q4_0 is 18 bytes per 32-element
  // block = 0.5625 B/elem, not a flat 0.5), matching the same table used for tensor byte sizes.
  return spec.typeSize / spec.blockSize;
}
