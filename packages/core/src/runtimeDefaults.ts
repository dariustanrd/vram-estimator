import { llamaCppB9860, vllmV0240 } from "./generated/runtimeDefaults.generated.js";
import type { RuntimeSnapshot } from "./types.js";

const VLLM: Record<string, RuntimeSnapshot> = {
  "v0.24.0": vllmV0240
};

const LLAMA_CPP: Record<string, RuntimeSnapshot> = {
  b9860: llamaCppB9860
};

export function listRuntimeDefaults(): { vllm: RuntimeSnapshot[]; llamacpp: RuntimeSnapshot[] } {
  return {
    vllm: Object.values(VLLM),
    llamacpp: Object.values(LLAMA_CPP)
  };
}

export function getVllmDefaults(version = "v0.24.0"): RuntimeSnapshot {
  const snapshot = VLLM[version];
  if (!snapshot) throw new Error(`Unsupported vLLM runtime version: ${version}`);
  return snapshot;
}

export function getLlamaCppDefaults(version = "b9860"): RuntimeSnapshot {
  const snapshot = LLAMA_CPP[version];
  if (!snapshot) throw new Error(`Unsupported llama.cpp runtime version: ${version}`);
  return snapshot;
}
