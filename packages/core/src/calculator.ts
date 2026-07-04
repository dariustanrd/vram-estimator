import { memory } from "./units.js";
import type { EstimateResult, GroundTruthValue, HardwareEstimate, HardwareInput, MissingValue, RuntimeDefault } from "./types.js";

const DECIMAL_GB = 1_000_000_000;
const DEFAULT_VLLM_BLOCK_SIZE = 16;
const COMMON_GPU_VRAM_GB = [4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 120, 144, 192];

export type CoreCalcInput = {
  mode: "vllm" | "llamacpp";
  weightBytes?: GroundTruthValue<number> | undefined;
  layers?: GroundTruthValue<number> | undefined;
  /**
   * Per-token, per-layer element width used for ONE of the K or V tensors, i.e.
   * num_key_value_heads x head_dim. This replaces the older hiddenSize x gqa approximation:
   * that shortcut is only correct when head_dim happens to equal hiddenSize / num_attention_heads,
   * which is false for architectures with an explicit, different head_dim (Gemma2/3, Qwen2, etc).
   */
  kvGroupWidth?: GroundTruthValue<number> | undefined;
  context?: GroundTruthValue<number> | undefined;
  batch?: GroundTruthValue<number> | undefined;
  kvBytes?: GroundTruthValue<number> | undefined;
  utilization?: GroundTruthValue<number> | undefined;
  userOverrides: Record<string, unknown>;
  modelSources: string[];
  modelSourceDetails?: Record<string, unknown> | undefined;
  runtimeSources: RuntimeDefault[];
  notes?: string[] | undefined;
  missing?: MissingValue[] | undefined;
  hardware?: HardwareInput | undefined;
  kvFormulaLabel?: string | undefined;
  /**
   * Additional values that are useful for transparency (raw hidden_size, attention heads,
   * gqa ratio, head_dim, etc.) but are not themselves part of the calculation. Merged into
   * resolvedInputs for display purposes only; never gates the missing-value check.
   */
  displayValues?: Record<string, GroundTruthValue<unknown> | undefined> | undefined;
};

export function calculateEstimate(input: CoreCalcInput): EstimateResult {
  const missing = [...(input.missing ?? [])];
  const required = {
    weightBytes: input.weightBytes,
    layers: input.layers,
    kvGroupWidth: input.kvGroupWidth,
    context: input.context,
    batch: input.batch,
    kvBytes: input.kvBytes,
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
  for (const [field, value] of Object.entries(input.displayValues ?? {})) {
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
      hardware: null,
      formula: null,
      modelSources: input.modelSources,
      modelSourceDetails: input.modelSourceDetails,
      runtimeSources: input.runtimeSources,
      notes: input.notes ?? []
    };
  }

  const weightBytes = input.weightBytes!.value;
  const kvBytesPerToken = 2 * input.layers!.value * input.kvGroupWidth!.value * input.kvBytes!.value;
  const kvBytesPerFullContext = kvBytesPerToken * input.context!.value;
  const kvBytes = kvBytesPerFullContext * input.batch!.value;
  const totalBytes = (weightBytes + kvBytes) / input.utilization!.value;
  const overheadBytes = totalBytes - weightBytes - kvBytes;
  const hardware = calculateHardwareEstimate({
    hardware: input.hardware,
    weightBytes,
    kvBytesPerToken,
    context: input.context!.value,
    utilization: input.utilization!.value
  });

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
    hardware,
    formula: {
      weights: `weights = exact_weight_bytes \n\t= ${weightBytes}`,
      kvCache:
        input.kvFormulaLabel ??
          `kv_cache = 2 x layers x kv_group_width x context x batch x kv_bytes_per_element \n\t= 2 x ${input.layers!.value} x ${input.kvGroupWidth!.value} x ${input.context!.value} x ${input.batch!.value} x ${input.kvBytes!.value} \n\t= ${kvBytes}`,
        total: `total = (weights + kv_cache) / utilization \n\t= (${weightBytes} + ${kvBytes}) / ${input.utilization!.value} \n\t= ${totalBytes}`,
      overhead: `overhead (modeled residual, not measured activation/workspace memory) = total - weights - kv_cache \n\t= ${totalBytes} - ${weightBytes} - ${kvBytes} \n\t= ${overheadBytes}`
    },
    modelSources: input.modelSources,
    modelSourceDetails: input.modelSourceDetails,
    runtimeSources: input.runtimeSources,
    notes: input.notes ?? []
  };
}

export function gt<T>(
  value: T | undefined,
  source: string,
  providedBy: GroundTruthValue<T>["providedBy"],
  assumed?: boolean
): GroundTruthValue<T> | undefined {
  if (value === undefined || value === null) return undefined;
  return { value, source, providedBy, assumed };
}

type HardwareCalcInput = {
  hardware?: HardwareInput | undefined;
  weightBytes: number;
  kvBytesPerToken: number;
  context: number;
  utilization: number;
};

function calculateHardwareEstimate(input: HardwareCalcInput): HardwareEstimate {
  const numGpus = normalizeGpuCount(input.hardware?.numGpus);
  const blockSize = DEFAULT_VLLM_BLOCK_SIZE;
  const kvBlockBytes = input.kvBytesPerToken * blockSize;
  const blocksPerFullContext = blockSize > 0 ? Math.ceil(input.context / blockSize) : 0;
  const kvBytesPerFullContextBlockRounded = blocksPerFullContext * kvBlockBytes;
  const minimumRequiredTotalBytes = (input.weightBytes + kvBytesPerFullContextBlockRounded) / input.utilization;
  const minimumRequiredPerGpuBytes = minimumRequiredTotalBytes / numGpus;
  const userGpuVramGb = positiveNumber(input.hardware?.gpuVramGb);
  const inferred = userGpuVramGb === undefined;
  const commonGpuVramGb = inferred ? inferCommonGpuVramGb(minimumRequiredPerGpuBytes) : undefined;
  const gpuVramPerGpuBytes = (userGpuVramGb ?? commonGpuVramGb ?? Math.ceil(minimumRequiredPerGpuBytes / DECIMAL_GB)) * DECIMAL_GB;
  const totalGpuVramBytes = gpuVramPerGpuBytes * numGpus;
  const gpuMemoryBudgetBytes = totalGpuVramBytes * input.utilization;
  const availableKvCacheBytes = Math.max(0, gpuMemoryBudgetBytes - input.weightBytes);
  const gpuKvCacheBlocks = kvBlockBytes > 0 ? Math.floor(availableKvCacheBytes / kvBlockBytes) : 0;
  const maxFullContextConcurrency = blocksPerFullContext > 0 ? gpuKvCacheBlocks / blocksPerFullContext : 0;
  const gpuKvCacheTokens = input.context > 0 ? Math.floor(maxFullContextConcurrency * input.context) : 0;
  const allocatedKvCacheBytes = gpuKvCacheBlocks * kvBlockBytes;
  const totalUsedBytes = input.weightBytes + allocatedKvCacheBytes;
  const unusedHeadroomBytes = Math.max(0, totalGpuVramBytes - totalUsedBytes);

  return {
    gpuVramPerGpu: memory(gpuVramPerGpuBytes),
    numGpus,
    totalGpuVram: memory(totalGpuVramBytes),
    inferred,
    commonGpuVramGb,
    minimumRequiredGpuVram: memory(minimumRequiredTotalBytes),
    gpuMemoryBudget: memory(gpuMemoryBudgetBytes),
    availableKvCache: memory(availableKvCacheBytes),
    allocatedKvCache: memory(allocatedKvCacheBytes),
    totalUsed: memory(totalUsedBytes),
    unusedHeadroom: memory(unusedHeadroomBytes),
    kvBytesPerToken: input.kvBytesPerToken,
    blockSize,
    kvBlockBytes,
    gpuKvCacheBlocks,
    blocksPerFullContext,
    gpuKvCacheTokens,
    maxFullContextConcurrency,
    fitsFullContext: gpuKvCacheBlocks >= blocksPerFullContext
  };
}

function normalizeGpuCount(value: number | undefined): number {
  const parsed = positiveNumber(value);
  return parsed === undefined ? 1 : Math.max(1, Math.floor(parsed));
}

function positiveNumber(value: number | undefined): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : undefined;
}

function inferCommonGpuVramGb(minimumRequiredPerGpuBytes: number): number | undefined {
  const requiredGb = minimumRequiredPerGpuBytes / DECIMAL_GB;
  return COMMON_GPU_VRAM_GB.find((size) => size >= requiredGb);
}
