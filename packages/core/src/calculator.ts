import { memory } from "./units.js";
import type { EstimateResult, GroundTruthValue, MissingValue, RuntimeDefault } from "./types.js";

export type CoreCalcInput = {
  mode: "vllm" | "llamacpp";
  weightBytes?: GroundTruthValue<number> | undefined;
  layers?: GroundTruthValue<number> | undefined;
  hiddenSize?: GroundTruthValue<number> | undefined;
  context?: GroundTruthValue<number> | undefined;
  batch?: GroundTruthValue<number> | undefined;
  kvBytes?: GroundTruthValue<number> | undefined;
  gqa?: GroundTruthValue<number> | undefined;
  utilization?: GroundTruthValue<number> | undefined;
  userOverrides: Record<string, unknown>;
  modelSources: string[];
  runtimeSources: RuntimeDefault[];
  notes?: string[] | undefined;
  missing?: MissingValue[] | undefined;
  kvFormulaLabel?: string | undefined;
};

export function calculateEstimate(input: CoreCalcInput): EstimateResult {
  const missing = [...(input.missing ?? [])];
  const required = {
    weightBytes: input.weightBytes,
    layers: input.layers,
    hiddenSize: input.hiddenSize,
    context: input.context,
    batch: input.batch,
    kvBytes: input.kvBytes,
    gqa: input.gqa,
    utilization: input.utilization
  };

  for (const [field, value] of Object.entries(required)) {
    if (!value && !missing.some((item) => item.field === field)) {
      missing.push({ field, reason: "Required exact value is missing." });
    }
  }

  const resolvedInputs: Record<string, GroundTruthValue<unknown>> = {};
  for (const [field, value] of Object.entries(required)) {
    if (value) resolvedInputs[field] = value;
  }

  if (missing.length > 0) {
    return {
      mode: input.mode,
      ok: false,
      missing,
      resolvedInputs,
      userOverrides: input.userOverrides,
      memory: null,
      formula: null,
      modelSources: input.modelSources,
      runtimeSources: input.runtimeSources,
      notes: input.notes ?? []
    };
  }

  const weightBytes = input.weightBytes!.value;
  const kvBytes =
    2 *
    input.layers!.value *
    input.hiddenSize!.value *
    input.context!.value *
    input.batch!.value *
    input.kvBytes!.value *
    input.gqa!.value;
  const totalBytes = (weightBytes + kvBytes) / input.utilization!.value;
  const overheadBytes = totalBytes - weightBytes - kvBytes;

  return {
    mode: input.mode,
    ok: true,
    missing: [],
    resolvedInputs,
    userOverrides: input.userOverrides,
    memory: {
      weights: memory(weightBytes),
      kvCache: memory(kvBytes),
      overhead: memory(overheadBytes),
      total: memory(totalBytes)
    },
    formula: {
      weights: `weights = exact_weight_bytes \n\t= ${weightBytes}`,
      kvCache:
        input.kvFormulaLabel ??
          `kv_cache = 2 x layers x hiddenSize x context x batch x kvBytes x gqa \n\t= 2 x ${input.layers!.value} x ${input.hiddenSize!.value} x ${input.context!.value} x ${input.batch!.value} x ${input.kvBytes!.value} x ${input.gqa!.value} \n\t= ${kvBytes}`,
        total: `total = (weights + kv_cache) / utilization \n\t= (${weightBytes} + ${kvBytes}) / ${input.utilization!.value} \n\t= ${totalBytes}`,
      overhead: `overhead = total - weights - kv_cache \n\t= ${totalBytes} - ${weightBytes} - ${kvBytes} \n\t= ${overheadBytes}`
    },
    modelSources: input.modelSources,
    runtimeSources: input.runtimeSources,
    notes: input.notes ?? []
  };
}

export function gt<T>(
  value: T | undefined,
  source: string,
  providedBy: GroundTruthValue<T>["providedBy"]
): GroundTruthValue<T> | undefined {
  if (value === undefined || value === null) return undefined;
  return { value, source, providedBy };
}
