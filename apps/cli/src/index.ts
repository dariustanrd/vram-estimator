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
      context: numberFlag(parsed, "context"),
      batch: numberFlag(parsed, "batch"),
      kvDtype: stringFlag(parsed, "kv-dtype"),
      runtimeVersion: stringFlag(parsed, "runtime-version"),
      overrides: parseOverrides(parsed)
    };
    printJson(await estimateVllm(input, fetcher), write);
    return;
  }

  if (parsed.command === "llamacpp") {
    const input: LlamaCppEstimateInput = {
      source: target,
      hfToken,
      context: numberFlag(parsed, "context"),
      parallel: numberFlag(parsed, "parallel"),
      cacheTypeK: stringFlag(parsed, "cache-type-k"),
      cacheTypeV: stringFlag(parsed, "cache-type-v"),
      runtimeVersion: stringFlag(parsed, "runtime-version"),
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

function parseOverrides(parsed: Parsed): Record<string, number | string> {
  const mapping: Record<string, string> = {
    "weight-bytes": "weightBytes",
    layers: "layers",
    "hidden-size": "hiddenSize",
    "kv-bytes": "kvBytes",
    gqa: "gqa",
    "model-dtype": "modelDtype",
    "attention-heads": "attentionHeads",
    "kv-heads": "kvHeads",
    "cache-bytes-k": "cacheBytesK",
    "cache-bytes-v": "cacheBytesV"
  };
  const overrides: Record<string, number | string> = {};
  for (const [flag, field] of Object.entries(mapping)) {
    const value = parsed.flags[flag];
    if (typeof value === "string") {
      const numeric = Number(value);
      overrides[field] = Number.isFinite(numeric) && flag !== "model-dtype" ? numeric : value;
    }
  }
  return overrides;
}

function stringFlag(parsed: Parsed, name: string): string | undefined {
  const value = parsed.flags[name];
  return typeof value === "string" ? value : undefined;
}

function numberFlag(parsed: Parsed, name: string): number | undefined {
  const value = stringFlag(parsed, name);
  if (value === undefined) return undefined;
  const parsedValue = Number(value);
  if (!Number.isFinite(parsedValue)) throw new Error(`--${name} must be a number`);
  return parsedValue;
}

function printJson(value: unknown, write: (text: string) => void): void {
  write(JSON.stringify(value, null, 2));
}

function printHelp(write: (text: string) => void): void {
  write(`vram-estimator

Commands:
  vllm <hf-model-id-or-config-url> [--hf-token TOKEN] [--context N] [--batch N] [--kv-dtype DTYPE] [--runtime-version VERSION]
  llamacpp <gguf-url-or-repo::file> [--hf-token TOKEN] [--context N] [--parallel N] [--cache-type-k TYPE] [--cache-type-v TYPE] [--runtime-version VERSION]
  runtime-defaults

Explicit override flags:
  --weight-bytes N --layers N --hidden-size N --gqa N --attention-heads N --kv-heads N
  --kv-bytes N --model-dtype DTYPE --cache-bytes-k N --cache-bytes-v N
`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  runCli(process.argv.slice(2)).catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
