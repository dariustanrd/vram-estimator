import llamaCppB9860 from "../../../runtime-defaults/llama-cpp/b9860.json" with { type: "json" };
import vllmV0240 from "../../../runtime-defaults/vllm/v0.24.0.json" with { type: "json" };
import type { RuntimeSnapshot } from "./types.js";

const VLLM: Record<string, RuntimeSnapshot> = {
  "v0.24.0": vllmV0240 as RuntimeSnapshot
};

const LLAMA_CPP: Record<string, RuntimeSnapshot> = {
  b9860: llamaCppB9860 as RuntimeSnapshot
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
