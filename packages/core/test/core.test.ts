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

  it("displays quantized HF weight dtype instead of torch_dtype", async () => {
    const fetcher = async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/models/")) {
        return jsonResponse({
          id: "org/fp8-model",
          sha: "abc",
          siblings: [{ rfilename: "model.safetensors", size: 1000 }]
        });
      }
      if (url.endsWith("/config.json")) {
        return jsonResponse({
          torch_dtype: "bfloat16",
          quantization_config: { quant_method: "fp8" },
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

    const result = await estimateVllm({ model: "org/fp8-model", kvDtype: "auto" }, fetcher as typeof fetch);

    expect(result.ok).toBe(true);
    expect(result.resolvedInputs.weightDtype).toMatchObject({
      value: "fp8",
      source: "config.quantization_config.quant_method",
      providedBy: "metadata"
    });
    expect(result.resolvedInputs.kvBytes).toMatchObject({ value: 2, source: "dtype.bfloat16" });
  });

  it("reads vLLM architecture fields from nested text_config", async () => {
    const result = await estimateVllm(
      { model: "org/nested", batch: 1 },
      hfModelFetcher({
        model_type: "wrapper",
        text_config: {
          dtype: "bfloat16",
          num_hidden_layers: 1,
          hidden_size: 4,
          max_position_embeddings: 8,
          num_attention_heads: 2,
          num_key_value_heads: 1,
          head_dim: 2
        }
      })
    );

    expect(result.ok).toBe(true);
    expect(result.resolvedInputs.layers).toMatchObject({
      value: 1,
      source: "config.text_config.num_hidden_layers",
      providedBy: "metadata"
    });
    expect(result.resolvedInputs.kvBytes).toMatchObject({ value: 2, source: "dtype.bfloat16" });
    expect(result.notes.join("\n")).toContain("Resolved architecture fields from config.text_config");
  });

  it("reads vLLM architecture fields from alternate nested config keys", async () => {
    const result = await estimateVllm(
      { model: "org/llm-config", batch: 1 },
      hfModelFetcher({
        llm_config: {
          dtype: "float16",
          num_hidden_layers: 1,
          hidden_size: 4,
          max_position_embeddings: 8,
          num_attention_heads: 2,
          num_key_value_heads: 1,
          head_dim: 2
        }
      })
    );

    expect(result.ok).toBe(true);
    expect(result.resolvedInputs.layers).toMatchObject({ source: "config.llm_config.num_hidden_layers" });
  });

  it("displays compressed-tensors weight dtype from config groups", async () => {
    const result = await estimateVllm(
      { model: "org/compressed", batch: 1 },
      hfModelFetcher({
        torch_dtype: "bfloat16",
        quantization_config: {
          quant_method: "compressed-tensors",
          config_groups: {
            group_0: { weights: { type: "float8_e4m3fn" } }
          }
        },
        num_hidden_layers: 1,
        hidden_size: 4,
        max_position_embeddings: 8,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 2
      })
    );

    expect(result.ok).toBe(true);
    expect(result.resolvedInputs.weightDtype).toMatchObject({
      value: "float8_e4m3fn",
      source: "config.quantization_config.config_groups.*.weights.type"
    });
  });

  it("uses scalar HF sliding_window for all layers when no per-layer pattern exists", async () => {
    const result = await estimateVllm(
      { model: "org/sliding", context: 8, batch: 1 },
      hfModelFetcher({
        torch_dtype: "float16",
        num_hidden_layers: 2,
        hidden_size: 8,
        max_position_embeddings: 8,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        sliding_window: 4
      })
    );

    expect(result.ok).toBe(true);
    expect(result.memory?.kvCache.bytes).toBe(128);
    expect(result.resolvedInputs.attentionLayerContexts).toMatchObject({ value: "4,4" });
  });

  it("reports unsupported DeepSeek-V4 HF cache metadata", async () => {
    const result = await estimateVllm(
      { model: "deepseek/v4", context: 8, batch: 1 },
      hfModelFetcher({
        model_type: "deepseek_v4",
        torch_dtype: "bfloat16",
        num_hidden_layers: 1,
        hidden_size: 8,
        max_position_embeddings: 8,
        num_attention_heads: 2,
        kv_lora_rank: 3,
        qk_rope_head_dim: 1,
        qk_nope_head_dim: 2,
        v_head_dim: 3
      })
    );

    expect(result.ok).toBe(false);
    expect(result.memory).toBeNull();
    expect(result.missing.map((item) => item.field)).toContain("kvCache");
  });

  it("uses safetensors dtype metadata when HF config omits dtype", async () => {
    const result = await estimateVllm(
      { model: "org/bf16", batch: 1 },
      hfModelFetcher(
        {
          num_hidden_layers: 1,
          hidden_size: 4,
          max_position_embeddings: 8,
          num_attention_heads: 2,
          num_key_value_heads: 1,
          head_dim: 2
        },
        { safetensors: { parameters: { BF16: 1000 }, total: 1000 } }
      )
    );

    expect(result.ok).toBe(true);
    expect(result.resolvedInputs.kvBytes).toMatchObject({ value: 2, source: "dtype.bfloat16" });
    expect(result.resolvedInputs.weightDtype).toMatchObject({
      value: "bfloat16",
      source: "hf.safetensors.parameters.BF16",
      providedBy: "metadata"
    });
  });

  it("calculates vLLM MLA latent KV cache directly", async () => {
    const result = await estimateVllm(
      { model: "org/mla", context: 4, batch: 1 },
      hfModelFetcher({
        torch_dtype: "bfloat16",
        num_hidden_layers: 2,
        hidden_size: 8,
        max_position_embeddings: 16,
        num_attention_heads: 2,
        kv_lora_rank: 3,
        qk_rope_head_dim: 1,
        qk_nope_head_dim: 2,
        v_head_dim: 3
      })
    );

    expect(result.ok).toBe(true);
    expect(result.memory?.kvCache.bytes).toBe(64);
    expect(result.resolvedInputs.kvCacheBytes).toMatchObject({
      value: 64,
      source: "hf.mla_kv_cache_formula",
      providedBy: "metadata"
    });
    expect(result.formula?.kvCache).toContain("mla_cache = sum(layer_contexts)");
  });

  it("excludes HF mlp_only_layers from direct KV cache", async () => {
    const result = await estimateVllm(
      { model: "org/mlp-only", context: 8, batch: 1 },
      hfModelFetcher({
        torch_dtype: "float16",
        num_hidden_layers: 4,
        hidden_size: 8,
        max_position_embeddings: 8,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        mlp_only_layers: [1, 3]
      })
    );

    expect(result.ok).toBe(true);
    expect(result.memory?.kvCache.bytes).toBe(256);
    expect(result.resolvedInputs.attentionLayerContexts).toMatchObject({ value: "8,0,8,0" });
    expect(result.formula?.kvCache).toContain("sum_over_layers");
  });

  it("reports unsupported HF recurrent state cache instead of returning KV-only totals", async () => {
    const result = await estimateVllm(
      { model: "org/recurrent", context: 8, batch: 1 },
      hfModelFetcher({
        torch_dtype: "float16",
        num_hidden_layers: 2,
        hidden_size: 8,
        max_position_embeddings: 8,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        layer_types: ["full_attention", "mamba"]
      })
    );

    expect(result.ok).toBe(false);
    expect(result.memory).toBeNull();
    expect(result.missing.map((item) => item.field)).toContain("stateCache");
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
    expect(result.formula?.overhead).toContain("GGUF-only calculation cannot determine that overhead");
    expect(result.notes.join("\n")).toContain("KV-cache term in this estimate is 0");
    expect(result.notes.join("\n")).not.toContain("total persistent KV cache reserved");
    expect(result.notes.join("\n")).not.toContain("assumed head_dim");
  });

  it("sums tensor bytes across inferable split GGUF files", async () => {
    const firstUrl = "https://example.test/model-00001-of-00002.gguf";
    const secondUrl = "https://example.test/model-00002-of-00002.gguf";
    const result = await estimateLlamaCpp(
      { source: firstUrl },
      ggufFetcherByUrl({
        [firstUrl]: makeSplitTinyGguf(0, [4, 4]),
        [secondUrl]: makeSplitTinyGguf(1, [8, 4])
      })
    );

    expect(result.ok).toBe(true);
    expect(result.resolvedInputs.weightBytes).toMatchObject({ value: 192, source: "gguf.tensor_table" });
    expect(result.memory?.weights.bytes).toBe(192);
    expect(result.modelSources).toEqual([firstUrl, secondUrl]);
    expect(result.notes.join("\n")).toContain("summed tensor tables across 2 files");
  });

  it("reports missing weights for mismatched split GGUF metadata", async () => {
    const firstUrl = "https://example.test/bad-00001-of-00002.gguf";
    const secondUrl = "https://example.test/bad-00002-of-00002.gguf";
    const result = await estimateLlamaCpp(
      { source: firstUrl },
      ggufFetcherByUrl({
        [firstUrl]: makeSplitTinyGguf(0, [4, 4]),
        [secondUrl]: makeSplitTinyGguf(0, [8, 4])
      })
    );

    expect(result.ok).toBe(false);
    expect(result.resolvedInputs.weightBytes).toBeUndefined();
    expect(result.missing.map((item) => item.field)).toContain("weightBytes");
  });

  it("reports missing weights for split GGUF files with non-inferable names", async () => {
    const result = await estimateLlamaCpp(
      { source: "https://example.test/model.gguf" },
      ggufFetcherByUrl({ "https://example.test/model.gguf": makeSplitTinyGguf(0, [4, 4]) })
    );

    expect(result.ok).toBe(false);
    expect(result.resolvedInputs.weightBytes).toBeUndefined();
    expect(result.missing.map((item) => item.field)).toContain("weightBytes");
  });

  it("recognizes new GPT-OSS-era GGML tensor types", () => {
    expect(parseGguf(makeGgufFixture(0, () => {}, { dims: [32], tensorType: 39 })).tensorBytes).toBe(17);
    expect(parseGguf(makeGgufFixture(0, () => {}, { dims: [64], tensorType: 40 })).tensorBytes).toBe(36);
    expect(parseGguf(makeGgufFixture(0, () => {}, { dims: [128], tensorType: 41 })).tensorBytes).toBe(18);
    expect(parseGguf(makeGgufFixture(0, () => {}, { dims: [64], tensorType: 42 })).tensorBytes).toBe(18);
  });

  it("uses scalar GGUF sliding_window for all layers when no pattern exists", async () => {
    const result = await estimateLlamaCpp(
      { source: "https://example.test/sliding.gguf" },
      ggufFetcher(makeSlidingGguf())
    );

    expect(result.ok).toBe(true);
    expect(result.memory?.kvCache.bytes).toBe(128);
    expect(result.resolvedInputs.attentionLayerContexts).toMatchObject({ value: "4,4" });
  });

  it("calculates llama.cpp KV cache from per-layer GGUF KV-head arrays", async () => {
    const result = await estimateLlamaCpp(
      { source: "https://example.test/per-layer.gguf" },
      ggufFetcher(makePerLayerKvGguf())
    );

    expect(result.ok).toBe(true);
    expect(result.memory?.kvCache.bytes).toBe(384);
    expect(result.resolvedInputs.kvCacheBytes).toMatchObject({
      value: 384,
      source: "gguf.per_layer_kv_cache_formula",
      providedBy: "metadata"
    });
    expect(result.resolvedInputs.kvHeadsPerLayer).toMatchObject({ value: "1,0,2" });
    expect(result.formula?.kvCache).toContain("sum_over_layers");
  });

  it("rejects ambiguous GGUF per-layer arrays whose length does not match layer count", async () => {
    const result = await estimateLlamaCpp(
      { source: "https://example.test/short-array.gguf" },
      ggufFetcher(makeShortKvArrayGguf())
    );

    expect(result.ok).toBe(false);
    expect(result.memory).toBeNull();
    expect(result.missing.map((item) => item.field)).toContain("kvHeads");
  });

  it("reports unsupported llama.cpp DeepSeek-V4 cache metadata", async () => {
    const result = await estimateLlamaCpp(
      { source: "https://example.test/deepseek4.gguf" },
      ggufFetcher(makeDeepseek4Gguf())
    );

    expect(result.ok).toBe(false);
    expect(result.memory).toBeNull();
    expect(result.missing.map((item) => item.field)).toContain("kvCache");
  });

  it("reports unsupported llama.cpp recurrent state cache", async () => {
    const result = await estimateLlamaCpp(
      { source: "https://example.test/recurrent.gguf" },
      ggufFetcher(makeRecurrentGguf())
    );

    expect(result.ok).toBe(false);
    expect(result.memory).toBeNull();
    expect(result.missing.map((item) => item.field)).toContain("stateCache");
  });
});

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json" }
  });
}

function hfModelFetcher(config: Record<string, unknown>, api: Record<string, unknown> = {}): typeof fetch {
  return (async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("/api/models/")) {
      return jsonResponse({
        id: "org/test",
        sha: "abc",
        siblings: [{ rfilename: "model.safetensors", size: 1000 }],
        ...api
      });
    }
    if (url.endsWith("/config.json")) return jsonResponse(config);
    throw new Error(`Unexpected URL ${url}`);
  }) as typeof fetch;
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

function makeSplitTinyGguf(splitNo: number, dims: number[]): ArrayBuffer {
  return makeGgufFixture(
    8,
    (writer) => {
      writer.kvString("general.architecture", "llama");
      writer.kvU32("llama.block_count", 1);
      writer.kvU32("llama.embedding_length", 4);
      writer.kvU32("llama.context_length", 8);
      writer.kvU32("llama.attention.head_count", 2);
      writer.kvU32("llama.attention.head_count_kv", 1);
      writer.kvU32("split.count", 2);
      writer.kvU32("split.no", splitNo);
    },
    { dims }
  );
}

function makePerLayerKvGguf(): ArrayBuffer {
  return makeGgufFixture(8, (writer) => {
    writer.kvString("general.architecture", "llama");
    writer.kvU32("llama.block_count", 3);
    writer.kvU32("llama.embedding_length", 8);
    writer.kvU32("llama.context_length", 8);
    writer.kvU32("llama.attention.head_count", 2);
    writer.kvArrayU32("llama.attention.head_count_kv", [1, 0, 2]);
    writer.kvU32("llama.attention.key_length", 4);
    writer.kvU32("llama.attention.value_length", 4);
  });
}

function makeSlidingGguf(): ArrayBuffer {
  return makeGgufFixture(9, (writer) => {
    writer.kvString("general.architecture", "llama");
    writer.kvU32("llama.block_count", 2);
    writer.kvU32("llama.embedding_length", 8);
    writer.kvU32("llama.context_length", 8);
    writer.kvU32("llama.attention.head_count", 2);
    writer.kvU32("llama.attention.head_count_kv", 1);
    writer.kvU32("llama.attention.key_length", 4);
    writer.kvU32("llama.attention.value_length", 4);
    writer.kvU32("llama.attention.sliding_window", 4);
  });
}

function makeShortKvArrayGguf(): ArrayBuffer {
  return makeGgufFixture(8, (writer) => {
    writer.kvString("general.architecture", "llama");
    writer.kvU32("llama.block_count", 3);
    writer.kvU32("llama.embedding_length", 8);
    writer.kvU32("llama.context_length", 8);
    writer.kvU32("llama.attention.head_count", 2);
    writer.kvArrayU32("llama.attention.head_count_kv", [1, 2]);
    writer.kvU32("llama.attention.key_length", 4);
    writer.kvU32("llama.attention.value_length", 4);
  });
}

function makeDeepseek4Gguf(): ArrayBuffer {
  return makeGgufFixture(8, (writer) => {
    writer.kvString("general.architecture", "deepseek4");
    writer.kvU32("deepseek4.block_count", 1);
    writer.kvU32("deepseek4.embedding_length", 8);
    writer.kvU32("deepseek4.context_length", 8);
    writer.kvU32("deepseek4.attention.head_count", 2);
    writer.kvU32("deepseek4.attention.head_count_kv", 1);
    writer.kvU32("deepseek4.attention.key_length", 4);
    writer.kvU32("deepseek4.attention.value_length", 4);
  });
}

function makeRecurrentGguf(): ArrayBuffer {
  return makeGgufFixture(9, (writer) => {
    writer.kvString("general.architecture", "nemotron-h");
    writer.kvU32("nemotron-h.block_count", 2);
    writer.kvU32("nemotron-h.embedding_length", 8);
    writer.kvU32("nemotron-h.context_length", 8);
    writer.kvU32("nemotron-h.attention.head_count", 2);
    writer.kvU32("nemotron-h.attention.head_count_kv", 1);
    writer.kvU32("nemotron-h.attention.key_length", 4);
    writer.kvU32("nemotron-h.attention.value_length", 4);
    writer.kvU32("nemotron-h.recurrent_pattern", 2);
  });
}

function makeGgufFixture(
  metadataCount: number,
  writeMetadata: (writer: Writer) => void,
  options: { dims?: number[]; tensorType?: number } = {}
): ArrayBuffer {
  const writer = new Writer();
  const dims = options.dims ?? [4, 4];
  writer.bytes([0x47, 0x47, 0x55, 0x46]);
  writer.u32(3);
  writer.u64(1);
  writer.u64(metadataCount);
  writeMetadata(writer);
  writer.string("token_embd.weight");
  writer.u32(dims.length);
  for (const dim of dims) writer.u64(dim);
  writer.u32(options.tensorType ?? 0);
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

function ggufFetcherByUrl(buffers: Record<string, ArrayBuffer>): typeof fetch {
  return (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const buffer = buffers[url];
    if (!buffer) throw new Error(`Unexpected GGUF URL ${url}`);
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

  kvArrayU32(key: string, values: number[]): void {
    this.string(key);
    this.u32(9);
    this.u32(4);
    this.u64(values.length);
    for (const value of values) this.u32(value);
  }

  buffer(): ArrayBuffer {
    return new Uint8Array(this.chunks).buffer;
  }
}
