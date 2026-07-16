import { describe, expect, it } from "vitest";
import { runCli } from "../src/index.js";

describe("CLI snapshots", () => {
  it("prints successful vLLM estimate output", async () => {
    const output = await invoke(["vllm", "org/tiny"], hfFetcher({ completeConfig: true }));
    const parsed = JSON.parse(output);

    expect(parsed.modelSourceDetails).toMatchObject({
      provider: "huggingface",
      modelId: "org/tiny",
      revision: "abc123",
      parsedFiles: [
        {
          path: "config.json",
          fields: {
            num_hidden_layers: 1,
            hidden_size: 4,
            max_position_embeddings: 8,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            torch_dtype: "float16"
          }
        }
      ],
      weightBytes: 1000,
      weightFiles: [{ path: "model.safetensors", size: 1000 }]
    });

    expect(snapshot(parsed)).toMatchInlineSnapshot(`
      {
        "formula": {
          "kvCache": "kv_cache = 2 x layers x kv_group_width x context x batch x kv_bytes_per_element 
      	= 2 x 1 x 2 x 8 x 128 x 2 
      	= 8192",
          "total": "total = (weights + kv_cache) / utilization 
      	= (1000 + 8192) / 0.92 
      	= 9991.304347826086",
        },
        "missing": [],
        "mode": "vllm",
        "ok": true,
        "resolved": [
          "weightBytes:1000:metadata",
          "layers:1:metadata",
          "kvGroupWidth:2:metadata",
          "context:8:metadata",
          "batch:128:runtime-default",
          "kvBytes:2:metadata",
          "utilization:0.92:runtime-default",
          "hiddenSize:4:metadata",
          "attentionHeads:2:metadata",
          "kvHeads:1:metadata",
          "headDim:2:metadata",
          "headDimK:2:metadata",
          "headDimV:2:metadata",
          "gqa:0.5:metadata",
          "weightDtype:float16:metadata",
        ],
        "totalBytes": 9991.304347826086,
      }
    `);
  });

  it("prints successful llama.cpp estimate output", async () => {
    const gguf = makeTinyGguf();
    const output = await invoke(["llamacpp", "https://example.test/tiny.gguf"], ggufFetcher(gguf));
    const parsed = JSON.parse(output);

    expect(snapshot(parsed)).toMatchInlineSnapshot(`
      {
        "formula": {
          "kvCache": "kv_cache = layers x kv_group_width x context x parallel x (cache_bytes_k + cache_bytes_v) 
      	= 1 x 2 x 8 x 1 x (2 + 2) 
      	= 64",
          "total": "total = (weights + kv_cache) / utilization 
      	= (64 + 64) / 1 
      	= 128",
        },
        "missing": [],
        "mode": "llamacpp",
        "ok": true,
        "resolved": [
          "weightBytes:64:metadata",
          "layers:1:metadata",
          "kvGroupWidth:2:metadata",
          "context:8:metadata",
          "batch:1:runtime-default",
          "kvBytes:2:runtime-default",
          "utilization:1:runtime-default",
          "hiddenSize:4:metadata",
          "attentionHeads:2:metadata",
          "kvHeads:1:metadata",
          "headDimK:2:metadata",
          "headDimV:2:metadata",
          "gqa:0.5:metadata",
          "cacheBytesK:2:runtime-default",
          "cacheBytesV:2:runtime-default",
        ],
        "totalBytes": 128,
      }
    `);
  });

  it("prints cacheless llama.cpp KV formula for BERT-style GGUFs", async () => {
    const gguf = makeTinyBertGguf();
    const output = await invoke(
      ["llamacpp", "https://example.test/bert.gguf", "--cache-type-k", "not-a-cache-type"],
      ggufFetcher(gguf)
    );
    const parsed = JSON.parse(output);

    expect(snapshot(parsed)).toMatchInlineSnapshot(`
      {
        "formula": {
          "kvCache": "kv_cache = 0
      	bert has no persistent autoregressive KV cache in llama.cpp",
          "overhead": "modeled_overhead = total - weights - kv_cache 
      	= 64 - 64 - 0 
      	= 0
      
      Note: this 0 is only the estimator's residual after weights + persistent KV cache. llama.cpp still needs runtime memory for temporary activations, graph buffers, backend workspaces, allocator padding, tokenizer/model structures, and possibly mmap/accounting effects; this GGUF-only calculation cannot determine that overhead.",
          "total": "total = (weights + kv_cache) / utilization 
      	= (64 + 0) / 1 
      	= 64",
        },
        "missing": [],
        "mode": "llamacpp",
        "ok": true,
        "resolved": [
          "weightBytes:64:metadata",
          "layers:6:metadata",
          "kvGroupWidth:0:metadata",
          "context:512:metadata",
          "batch:1:runtime-default",
          "kvBytes:0:metadata",
          "utilization:1:runtime-default",
          "hiddenSize:384:metadata",
          "attentionHeads:12:metadata",
          "weightDtype:Q4_0:metadata",
          "modelType:Encoder · BERT:metadata",
        ],
        "totalBytes": 64,
      }
    `);
  });

  it("prints missing metadata with required override fields", async () => {
    const output = await invoke(["vllm", "org/incomplete"], hfFetcher({ completeConfig: false }));
    const parsed = JSON.parse(output);

    expect(snapshot(parsed)).toMatchInlineSnapshot(`
      {
        "formula": null,
        "missing": [
          "kvBytes",
          "layers",
          "kvGroupWidth",
          "context",
        ],
        "mode": "vllm",
        "ok": false,
        "resolved": [
          "weightBytes:1000:metadata",
          "batch:128:runtime-default",
          "utilization:0.92:runtime-default",
        ],
        "totalBytes": null,
      }
    `);
  });

  it("prints JSON output", async () => {
    const output = await invoke(["runtime-defaults"], async () => {
      throw new Error("runtime-defaults should not fetch");
    });

    expect(() => JSON.parse(output)).not.toThrow();
  });
});

async function invoke(argv: string[], fetcher: typeof fetch): Promise<string> {
  const writes: string[] = [];
  await runCli(argv, {
    env: {},
    fetcher,
    stdout: (text) => writes.push(text)
  });
  return writes.join("\n");
}

function snapshot(result: any): Record<string, unknown> {
  const resolved = Object.entries(result.resolvedInputs ?? {}).map(
    ([field, value]: [string, any]) => `${field}:${value.value}:${value.providedBy}`
  );
  return {
    mode: result.mode,
    ok: result.ok,
    missing: result.missing.map((item: any) => item.field),
    totalBytes: result.memory?.total.bytes ?? null,
    formula: result.formula
      ? {
          kvCache: result.formula.kvCache,
          ...(result.formula.overhead.includes("GGUF-only calculation cannot determine")
            ? { overhead: result.formula.overhead }
            : {}),
          total: result.formula.total
        }
      : null,
    resolved
  };
}

function hfFetcher(options: { completeConfig: boolean }): typeof fetch {
  return async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("/api/models/")) {
      return jsonResponse({
        id: "org/tiny",
        sha: "abc123",
        siblings: [{ rfilename: "model.safetensors", size: 1000 }]
      });
    }
    if (url.endsWith("/config.json")) {
      return jsonResponse(
        options.completeConfig
          ? {
              num_hidden_layers: 1,
              hidden_size: 4,
              max_position_embeddings: 8,
              num_attention_heads: 2,
              num_key_value_heads: 1,
              torch_dtype: "float16"
            }
          : {}
      );
    }
    throw new Error(`Unexpected fetch ${url}`);
  };
}

function ggufFetcher(buffer: ArrayBuffer): typeof fetch {
  return async (_input: RequestInfo | URL, init?: RequestInit) => {
    if (init?.method === "HEAD") {
      return new Response(null, {
        status: 200,
        headers: { "content-length": String(buffer.byteLength) }
      });
    }
    return new Response(buffer, { status: 206 });
  };
}

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
