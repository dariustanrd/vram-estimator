#!/usr/bin/env node
import { pathToFileURL } from "node:url";
import {
  estimateLlamaCpp,
  estimateVllm,
  listRuntimeDefaults,
  resolveHfToken,
  type Fetcher,
  type LlamaCppEstimateInput,
  type VllmEstimateInput
} from "@vram-estimator/core";

type Parsed = {
  command?: string | undefined;
  positional: string[];
  flags: Record<string, string | boolean>;
};

export type CliOptions = {
  env?: Record<string, string | undefined> | undefined;
  stdout?: ((text: string) => void) | undefined;
  fetcher?: Fetcher | undefined;
};

export async function runCli(argv: string[], options: CliOptions = {}): Promise<void> {
  const parsed = parse(argv);
  const env = options.env ?? process.env;
  const write = options.stdout ?? console.log;
  const fetcher = options.fetcher ?? fetch;
  if (!parsed.command || parsed.flags.help) {
    printHelp(write);
    return;
  }

  if (parsed.command === "runtime-defaults") {
    printJson(listRuntimeDefaults(), write);
    return;
  }

  const target = parsed.positional[0];
  if (!target) throw new Error(`Missing target for ${parsed.command}`);
  const hfToken = resolveHfToken(stringFlag(parsed, "hf-token"), env.HF_TOKEN);

  if (parsed.command === "vllm") {
    const input: VllmEstimateInput = {
      model: target,
      hfToken,
      context: numberFlag(parsed, "max-model-len", "context"),
      batch: numberFlag(parsed, "max-num-seqs", "batch"),
      kvDtype: stringFlag(parsed, "kv-cache-dtype", "kv-dtype"),
      runtimeVersion: stringFlag(parsed, "runtime-version"),
      gpuVramGb: numberFlag(parsed, "gpu-vram-gb"),
      numGpus: numberFlag(parsed, "num-gpus"),
      overrides: parseOverrides(parsed)
    };
    printJson(await estimateVllm(input, fetcher), write);
    return;
  }

  if (parsed.command === "llamacpp") {
    const input: LlamaCppEstimateInput = {
      source: target,
      hfToken,
      context: numberFlag(parsed, "ctx-size", "context"),
      parallel: numberFlag(parsed, "parallel"),
      cacheTypeK: stringFlag(parsed, "cache-type-k"),
      cacheTypeV: stringFlag(parsed, "cache-type-v"),
      runtimeVersion: stringFlag(parsed, "runtime-version"),
      gpuVramGb: numberFlag(parsed, "gpu-vram-gb"),
      numGpus: numberFlag(parsed, "num-gpus"),
      overrides: parseOverrides(parsed)
    };
    printJson(await estimateLlamaCpp(input, fetcher), write);
    return;
  }

  throw new Error(`Unknown command: ${parsed.command}`);
}

function parse(argv: string[]): Parsed {
  const [command, ...rest] = argv;
  const flags: Record<string, string | boolean> = {};
  const positional: string[] = [];
  for (let i = 0; i < rest.length; i++) {
    const arg = rest[i]!;
    if (!arg.startsWith("--")) {
      positional.push(arg);
      continue;
    }
    const withoutPrefix = arg.slice(2);
    const [inlineKey, inlineValue] = withoutPrefix.split("=", 2);
    if (inlineValue !== undefined) {
      flags[inlineKey!] = inlineValue;
      continue;
    }
    const next = rest[i + 1];
    if (next && !next.startsWith("--")) {
      flags[inlineKey!] = next;
      i++;
    } else {
      flags[inlineKey!] = true;
    }
  }
  return { command, positional, flags };
}

// Override flags follow each runtime's own naming convention: vLLM/Hugging Face config.json field
// names and llama.cpp GGUF metadata key names. Legacy short flags remain as aliases so existing
// scripts keep working. Multiple flags may map to the same internal field; a user only passes one.
const OVERRIDE_FLAGS: Record<string, string> = {
  "weight-bytes": "weightBytes",
  gqa: "gqa",
  // layers
  "num-hidden-layers": "layers",
  "block-count": "layers",
  layers: "layers",
  // hidden size
  "hidden-size": "hiddenSize",
  "embedding-length": "hiddenSize",
  // attention heads
  "num-attention-heads": "attentionHeads",
  "attention-head-count": "attentionHeads",
  "attention-heads": "attentionHeads",
  // key/value heads
  "num-key-value-heads": "kvHeads",
  "attention-head-count-kv": "kvHeads",
  "kv-heads": "kvHeads",
  // head dimension
  "head-dim": "headDim",
  "attention-key-length": "headDim",
  // model dtype (vLLM)
  "torch-dtype": "modelDtype",
  "model-dtype": "modelDtype",
  // kv cache element bytes (vLLM)
  "kv-bytes": "kvBytes",
  // cache element bytes (llama.cpp)
  "cache-bytes-k": "cacheBytesK",
  "cache-bytes-v": "cacheBytesV"
};

function parseOverrides(parsed: Parsed): Record<string, number | string> {
  const overrides: Record<string, number | string> = {};
  for (const [flag, field] of Object.entries(OVERRIDE_FLAGS)) {
    const value = parsed.flags[flag];
    if (typeof value === "string") {
      const numeric = Number(value);
      overrides[field] = Number.isFinite(numeric) && field !== "modelDtype" ? numeric : value;
    }
  }
  return overrides;
}

function stringFlag(parsed: Parsed, ...names: string[]): string | undefined {
  for (const name of names) {
    const value = parsed.flags[name];
    if (typeof value === "string") return value;
  }
  return undefined;
}

function numberFlag(parsed: Parsed, ...names: string[]): number | undefined {
  for (const name of names) {
    const value = stringFlag(parsed, name);
    if (value === undefined) continue;
    const parsedValue = Number(value);
    if (!Number.isFinite(parsedValue)) throw new Error(`--${name} must be a number`);
    return parsedValue;
  }
  return undefined;
}

function printJson(value: unknown, write: (text: string) => void): void {
  write(JSON.stringify(value, null, 2));
}

function printHelp(write: (text: string) => void): void {
  write(`vram-estimator

Commands:
  vllm <hf-model-id-or-config-url> [--max-model-len N] [--max-num-seqs N] [--kv-cache-dtype DTYPE] [--gpu-vram-gb N] [--num-gpus N] [--hf-token TOKEN] [--runtime-version VERSION]
  llamacpp <gguf-url-or-repo::file> [--ctx-size N] [--parallel N] [--cache-type-k TYPE] [--cache-type-v TYPE] [--gpu-vram-gb N] [--num-gpus N] [--hf-token TOKEN] [--runtime-version VERSION]
  runtime-defaults

vLLM model overrides (Hugging Face config.json fields):
  --num-hidden-layers N --hidden-size N --num-attention-heads N --num-key-value-heads N
  --head-dim N --torch-dtype DTYPE --kv-bytes N --gqa N --weight-bytes N

llama.cpp model overrides (GGUF metadata keys):
  --block-count N --embedding-length N --attention-head-count N --attention-head-count-kv N
  --attention-key-length N --cache-bytes-k N --cache-bytes-v N --gqa N --weight-bytes N
`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  runCli(process.argv.slice(2)).catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
