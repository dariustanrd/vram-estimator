import { describe, expect, it } from "vitest";
import {
  calculateEstimate,
  estimateLlamaCpp,
  estimateVllm,
  getVllmDefaults,
  memory,
  parseGguf,
  resolveHfToken
} from "../src/index.js";

describe("units", () => {
  it("converts bytes to GB and GiB", () => {
    const result = memory(1_073_741_824);
    expect(result.gb).toBeCloseTo(1.073741824);
    expect(result.gib).toBe(1);
  });
});

describe("calculator", () => {
  it("calculates weights, kv cache, overhead, and total", () => {
    const result = calculateEstimate({
      mode: "vllm",
      weightBytes: { value: 16_000_000_000, source: "fixture", providedBy: "metadata" },
      layers: { value: 32, source: "fixture", providedBy: "metadata" },
      kvGroupWidth: { value: 1024, source: "fixture", providedBy: "metadata" },
      context: { value: 8192, source: "fixture", providedBy: "metadata" },
      batch: { value: 1, source: "fixture", providedBy: "metadata" },
      kvBytes: { value: 2, source: "fixture", providedBy: "metadata" },
      utilization: { value: 0.9, source: "fixture", providedBy: "runtime-default" },
      userOverrides: {},
      modelSources: [],
      runtimeSources: []
    });
    expect(result.ok).toBe(true);
    expect(result.memory?.kvCache.bytes).toBe(1_073_741_824);
    expect(result.memory?.total.bytes).toBeCloseTo(18_970_824_248.888);
  });

  it("reports missing required values", () => {
    const result = calculateEstimate({
      mode: "vllm",
      userOverrides: {},
      modelSources: [],
      runtimeSources: []
    });
    expect(result.ok).toBe(false);
    expect(result.missing.map((item) => item.field)).toContain("weightBytes");
  });

  it("estimates hardware KV capacity from supplied GPU VRAM", () => {
    const result = calculateEstimate({
      mode: "vllm",
      weightBytes: { value: 16_000_000_000, source: "fixture", providedBy: "metadata" },
      layers: { value: 32, source: "fixture", providedBy: "metadata" },
      kvGroupWidth: { value: 1024, source: "fixture", providedBy: "metadata" },
      context: { value: 8192, source: "fixture", providedBy: "metadata" },
      batch: { value: 1, source: "fixture", providedBy: "metadata" },
      kvBytes: { value: 2, source: "fixture", providedBy: "metadata" },
      utilization: { value: 0.9, source: "fixture", providedBy: "runtime-default" },
      hardware: { gpuVramGb: 24, numGpus: 1 },
      userOverrides: {},
      modelSources: [],
      runtimeSources: []
    });
    expect(result.ok).toBe(true);
    expect(result.hardware?.inferred).toBe(false);
    expect(result.hardware?.availableKvCache.bytes).toBe(5_600_000_000);
    expect(result.hardware?.gpuKvCacheBlocks).toBe(2670);
    expect(result.hardware?.blocksPerFullContext).toBe(512);
    expect(result.hardware?.gpuKvCacheTokens).toBe(42_720);
  });

  it("matches vLLM's block-rounded logged KV token capacity", () => {
    const result = calculateEstimate({
      mode: "vllm",
      weightBytes: { value: 0, source: "fixture", providedBy: "metadata" },
      layers: { value: 1, source: "fixture", providedBy: "metadata" },
      kvGroupWidth: { value: 1, source: "fixture", providedBy: "metadata" },
      context: { value: 17, source: "fixture", providedBy: "metadata" },
      batch: { value: 1, source: "fixture", providedBy: "metadata" },
      kvBytes: { value: 1, source: "fixture", providedBy: "metadata" },
      utilization: { value: 1, source: "fixture", providedBy: "runtime-default" },
      hardware: { gpuVramGb: 0.00000032, numGpus: 1 },
      userOverrides: {},
      modelSources: [],
      runtimeSources: []
    });
    expect(result.ok).toBe(true);
    expect(result.hardware?.gpuKvCacheBlocks).toBe(10);
    expect(result.hardware?.blocksPerFullContext).toBe(2);
    expect(result.hardware?.maxFullContextConcurrency).toBe(5);
    expect(result.hardware?.gpuKvCacheTokens).toBe(85);
  });
});

describe("runtime defaults", () => {
  it("loads vLLM defaults with provenance", () => {
    const defaults = getVllmDefaults("v0.24.0");
    expect(defaults.defaults.gpu_memory_utilization?.value).toBe(0.92);
    expect(defaults.defaults.gpu_memory_utilization?.source.file).toBe("vllm/config/cache.py");
  });
});

describe("auth", () => {
  it("prefers user token over environment token", () => {
    expect(resolveHfToken("user", "env")).toBe("user");
    expect(resolveHfToken(undefined, "env")).toBe("env");
  });
});

describe("Hugging Face exact metadata", () => {
  it("uses a user-provided vLLM gpu memory utilization", async () => {
    const fetcher = async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/models/")) {
        return jsonResponse({
          id: "org/tiny",
          sha: "abc",
          siblings: [{ rfilename: "model.safetensors", size: 1000 }]
        });
      }
      if (url.endsWith("/config.json")) {
        return jsonResponse({
          torch_dtype: "float16",
          num_hidden_layers: 1,
          hidden_size: 4,
          max_position_embeddings: 8,
          num_attention_heads: 2,
          num_key_value_heads: 1,
          head_dim: 2
        });
      }
      throw new Error(`Unexpected URL ${url}`);
    };

    const result = await estimateVllm({ model: "org/tiny", gpuMemoryUtilization: 0.5 }, fetcher as typeof fetch);

    expect(result.ok).toBe(true);
    expect(result.resolvedInputs.utilization).toMatchObject({
      value: 0.5,
      source: "user.gpuMemoryUtilization",
      providedBy: "user"
    });
    expect(result.userOverrides.gpuMemoryUtilization).toBe(0.5);
  });

  it("does not infer params or architecture from a model name", async () => {
    const fetcher = async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/models/")) {
        return jsonResponse({
          id: "org/Model-8B",
          sha: "abc",
          siblings: [{ rfilename: "model.safetensors", size: 1000 }]
        });
      }
      if (url.endsWith("/config.json")) {
        return jsonResponse({ torch_dtype: "float16" });
      }
      throw new Error(`Unexpected URL ${url}`);
    };
    const result = await estimateVllm({ model: "org/Model-8B" }, fetcher as typeof fetch);
    expect(result.ok).toBe(false);
    expect(result.missing.map((item) => item.field)).toContain("layers");
    expect(result.resolvedInputs.weightBytes?.value).toBe(1000);
  });
});

describe("GGUF parser", () => {
  it("parses metadata and exact tensor bytes", () => {
    const parsed = parseGguf(makeTinyGguf());
    expect(parsed.version).toBe(3);
    expect(parsed.metadata["general.architecture"]).toBe("llama");
    expect(parsed.tensorBytes).toBe(64);
  });

  it("estimates BERT-style encoder GGUFs without requiring KV-head metadata", async () => {
    const result = await estimateLlamaCpp(
      { source: "https://example.test/bert.gguf" },
      ggufFetcher(makeTinyBertGguf())
    );

    expect(result.ok).toBe(true);
    expect(result.missing).toEqual([]);
    expect(result.resolvedInputs.kvGroupWidth).toMatchObject({ value: 0, providedBy: "metadata" });
    expect(result.resolvedInputs.kvBytes).toMatchObject({ value: 0, providedBy: "metadata" });
    expect(result.memory?.kvCache.bytes).toBe(0);
    expect(result.memory?.weights.bytes).toBe(64);
    expect(result.formula?.kvCache).toContain("has no persistent autoregressive KV cache");
    expect(result.notes.join("\n")).toContain("KV-cache term in this estimate is 0");
    expect(result.notes.join("\n")).not.toContain("total persistent KV cache reserved");
    expect(result.notes.join("\n")).not.toContain("assumed head_dim");
  });
});

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json" }
  });
}

function makeTinyGguf(): ArrayBuffer {
  const writer = new Writer();
  writer.bytes([0x47, 0x47, 0x55, 0x46]);
  writer.u32(3);
  writer.u64(1);
  writer.u64(6);
  writer.kvString("general.architecture", "llama");
  writer.kvU32("llama.block_count", 1);
  writer.kvU32("llama.embedding_length", 4);
  writer.kvU32("llama.context_length", 8);
  writer.kvU32("llama.attention.head_count", 2);
  writer.kvU32("llama.attention.head_count_kv", 1);
  writer.string("token_embd.weight");
  writer.u32(2);
  writer.u64(4);
  writer.u64(4);
  writer.u32(0);
  writer.u64(0);
  return writer.buffer();
}

function makeTinyBertGguf(): ArrayBuffer {
  const writer = new Writer();
  writer.bytes([0x47, 0x47, 0x55, 0x46]);
  writer.u32(3);
  writer.u64(1);
  writer.u64(6);
  writer.kvString("general.architecture", "bert");
  writer.kvU32("general.file_type", 2);
  writer.kvU32("bert.block_count", 6);
  writer.kvU32("bert.embedding_length", 384);
  writer.kvU32("bert.context_length", 512);
  writer.kvU32("bert.attention.head_count", 12);
  writer.string("token_embd.weight");
  writer.u32(2);
  writer.u64(4);
  writer.u64(4);
  writer.u32(0);
  writer.u64(0);
  return writer.buffer();
}

function ggufFetcher(buffer: ArrayBuffer): typeof fetch {
  return (async (_input: RequestInfo | URL, init?: RequestInit) => {
    if (init?.method === "HEAD") {
      return new Response(null, { status: 200, headers: { "content-length": String(buffer.byteLength) } });
    }
    return new Response(buffer, {
      status: 206,
      headers: {
        "content-range": `bytes 0-${buffer.byteLength - 1}/${buffer.byteLength}`,
        "content-length": String(buffer.byteLength)
      }
    });
  }) as typeof fetch;
}

class Writer {
  private chunks: number[] = [];

  bytes(values: number[]): void {
    this.chunks.push(...values);
  }

  u32(value: number): void {
    this.bytes([value & 255, (value >> 8) & 255, (value >> 16) & 255, (value >> 24) & 255]);
  }

  u64(value: number): void {
    let big = BigInt(value);
    for (let i = 0; i < 8; i++) {
      this.chunks.push(Number(big & 255n));
      big >>= 8n;
    }
  }

  string(value: string): void {
    const bytes = new TextEncoder().encode(value);
    this.u64(bytes.length);
    this.bytes([...bytes]);
  }

  kvString(key: string, value: string): void {
    this.string(key);
    this.u32(8);
    this.string(value);
  }

  kvU32(key: string, value: number): void {
    this.string(key);
    this.u32(4);
    this.u32(value);
  }

  buffer(): ArrayBuffer {
    return new Uint8Array(this.chunks).buffer;
  }
}
