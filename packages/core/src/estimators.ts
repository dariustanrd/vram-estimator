import { calculateEstimate, gt } from "./calculator.js";
import { bytesForDtype } from "./dtypes.js";
import { fetchGgufMetadata, ggufNumber, ggufString, cacheTypeBytes } from "./gguf.js";
import {
  fetchHfModelMetadata,
  kvBytesFromDtype,
  numberField,
  stringField
} from "./hf.js";
import { getLlamaCppDefaults, getVllmDefaults } from "./runtimeDefaults.js";
import type {
  EstimateResult,
  Fetcher,
  GroundTruthValue,
  LlamaCppEstimateInput,
  MissingValue,
  RuntimeDefault,
  VllmEstimateInput
} from "./types.js";

export async function estimateVllm(
  input: VllmEstimateInput,
  fetcher: Fetcher = fetch
): Promise<EstimateResult> {
  const runtime = getVllmDefaults(input.runtimeVersion);
  const defaults = runtime.defaults;
  const metadata = await fetchHfModelMetadata(input.model, { token: input.hfToken }, fetcher);
  const config = metadata.config;
  const userOverrides = compact({
    ...input.overrides,
    context: input.context,
    batch: input.batch,
    kvDtype: input.kvDtype
  });

  const layers = valueWithOverride(
    input.overrides?.layers,
    numberField(config, "num_hidden_layers"),
    "override.layers",
    "config.num_hidden_layers"
  );
  const hiddenSize = valueWithOverride(
    input.overrides?.hiddenSize,
    numberField(config, "hidden_size"),
    "override.hiddenSize",
    "config.hidden_size"
  );
  const context = valueWithOverride(
    input.context ?? input.overrides?.context,
    numberField(config, "max_position_embeddings"),
    input.context !== undefined || input.overrides?.context !== undefined ? "user.context" : "config.max_position_embeddings",
    "config.max_position_embeddings"
  );
  const batch = valueWithOverride(
    input.batch ?? input.overrides?.batch,
    defaults.max_num_seqs?.value as number | undefined,
    input.batch !== undefined || input.overrides?.batch !== undefined ? "user.batch" : "runtime.max_num_seqs",
    "runtime.max_num_seqs"
  );
  const attentionHeads = valueWithOverride(
    input.overrides?.attentionHeads,
    numberField(config, "num_attention_heads"),
    "override.attentionHeads",
    "config.num_attention_heads"
  );
  const kvHeads = valueWithOverride(
    input.overrides?.kvHeads,
    numberField(config, "num_key_value_heads"),
    "override.kvHeads",
    "config.num_key_value_heads"
  );
  const gqa = exactGqa(input.overrides?.gqa, attentionHeads, kvHeads);
  const modelDtype = input.overrides?.modelDtype ?? stringField(config, "torch_dtype");
  const defaultKvDtype = defaults.kv_cache_dtype?.value as string | undefined;
  const selectedKvDtype = input.kvDtype ?? defaultKvDtype;
  const resolvedKvDtype = selectedKvDtype === "auto" ? modelDtype : selectedKvDtype;
  const kvBytesValue = input.overrides?.kvBytes ?? kvBytesFromDtype(resolvedKvDtype);
  const weightBytes = valueWithOverride(
    input.overrides?.weightBytes,
    metadata.weightBytes,
    "override.weightBytes",
    metadata.weightFiles.length > 0 ? "hf.weight_file_sizes" : "hf.metadata"
  );
  const utilization = gt(
    defaults.gpu_memory_utilization?.value as number | undefined,
    "runtime.gpu_memory_utilization",
    "runtime-default"
  );

  const missing: MissingValue[] = [];
  if (selectedKvDtype === "auto" && !modelDtype && input.overrides?.kvBytes === undefined) {
    missing.push({
      field: "kvBytes",
      reason: "vLLM kv_cache_dtype=auto requires an explicit model torch_dtype or user kvBytes override."
    });
  }
  if (attentionHeads && !kvHeads && input.overrides?.gqa === undefined) {
    missing.push({
      field: "gqa",
      reason: "num_key_value_heads is absent; exact GQA ratio cannot be derived."
    });
  }
  if (!metadata.weightBytes && input.overrides?.weightBytes === undefined) {
    missing.push({
      field: "weightBytes",
      reason: "Hugging Face metadata did not expose exact model weight file sizes."
    });
  }

  return calculateEstimate({
    mode: "vllm",
    weightBytes,
    layers,
    hiddenSize,
    context,
    batch,
    kvBytes: gt(kvBytesValue, input.overrides?.kvBytes !== undefined ? "override.kvBytes" : `dtype.${resolvedKvDtype}`, input.overrides?.kvBytes !== undefined ? "user" : "metadata"),
    gqa,
    utilization,
    userOverrides,
    modelSources: metadata.sources,
    runtimeSources: Object.values(defaults),
    notes: [
      "vLLM v1 estimate is single GPU only with tensor_parallel_size=1.",
      "This is a deterministic estimate from exact metadata and selected runtime defaults, not observed runtime allocation."
    ],
    missing
  });
}

export async function estimateLlamaCpp(
  input: LlamaCppEstimateInput,
  fetcher: Fetcher = fetch
): Promise<EstimateResult> {
  const runtime = getLlamaCppDefaults(input.runtimeVersion);
  const defaults = runtime.defaults;
  const metadata = await fetchGgufMetadata(input.source, { token: input.hfToken }, fetcher);
  const arch = ggufString(metadata, "general.architecture");
  const prefix = arch ? `${arch}.` : "";
  const userOverrides = compact({
    ...input.overrides,
    context: input.context,
    parallel: input.parallel,
    cacheTypeK: input.cacheTypeK,
    cacheTypeV: input.cacheTypeV
  });

  const layers = valueWithOverride(
    input.overrides?.layers,
    arch ? ggufNumber(metadata, `${prefix}block_count`) : undefined,
    "override.layers",
    `${prefix}block_count`
  );
  const hiddenSize = valueWithOverride(
    input.overrides?.hiddenSize,
    arch ? ggufNumber(metadata, `${prefix}embedding_length`) : undefined,
    "override.hiddenSize",
    `${prefix}embedding_length`
  );
  const context = valueWithOverride(
    input.context ?? input.overrides?.context,
    arch ? ggufNumber(metadata, `${prefix}context_length`) : undefined,
    input.context !== undefined || input.overrides?.context !== undefined ? "user.context" : `${prefix}context_length`,
    `${prefix}context_length`
  );
  const parallel = valueWithOverride(
    input.parallel ?? input.overrides?.parallel,
    defaults.parallel?.value as number | undefined,
    input.parallel !== undefined || input.overrides?.parallel !== undefined ? "user.parallel" : "runtime.parallel",
    "runtime.parallel"
  );
  const attentionHeads = valueWithOverride(
    input.overrides?.attentionHeads,
    arch ? ggufNumber(metadata, `${prefix}attention.head_count`) : undefined,
    "override.attentionHeads",
    `${prefix}attention.head_count`
  );
  const kvHeads = valueWithOverride(
    input.overrides?.kvHeads,
    arch ? ggufNumber(metadata, `${prefix}attention.head_count_kv`) : undefined,
    "override.kvHeads",
    `${prefix}attention.head_count_kv`
  );
  const gqa = exactGqa(input.overrides?.gqa, attentionHeads, kvHeads);
  const cacheTypeK = input.cacheTypeK ?? (defaults.cache_type_k?.value as string | undefined);
  const cacheTypeV = input.cacheTypeV ?? (defaults.cache_type_v?.value as string | undefined);
  const cacheBytesK = input.overrides?.cacheBytesK ?? (cacheTypeK ? cacheTypeBytes(cacheTypeK) : undefined);
  const cacheBytesV = input.overrides?.cacheBytesV ?? (cacheTypeV ? cacheTypeBytes(cacheTypeV) : undefined);
  const combinedKvBytes =
    cacheBytesK !== undefined && cacheBytesV !== undefined ? (cacheBytesK + cacheBytesV) / 2 : undefined;
  const weightBytes = valueWithOverride(
    input.overrides?.weightBytes,
    metadata.tensorBytes,
    "override.weightBytes",
    "gguf.tensor_table"
  );
  const utilization = gt(
    defaults.runtime_utilization?.value as number | undefined,
    "runtime.runtime_utilization",
    "runtime-default"
  );

  const missing: MissingValue[] = [];
  if (!arch) {
    missing.push({
      field: "architecture",
      reason: "GGUF general.architecture is absent; architecture-specific metadata keys cannot be resolved."
    });
  }
  if (metadata.unknownTensorTypes.length > 0 && input.overrides?.weightBytes === undefined) {
    missing.push({
      field: "weightBytes",
      reason: `GGUF tensor table contains unsupported tensor types: ${metadata.unknownTensorTypes.join(", ")}.`
    });
  }
  if (attentionHeads && !kvHeads && input.overrides?.gqa === undefined) {
    missing.push({
      field: "gqa",
      reason: "GGUF attention.head_count_kv is absent; exact GQA ratio cannot be derived."
    });
  }

  return calculateEstimate({
    mode: "llamacpp",
    weightBytes,
    layers,
    hiddenSize,
    context,
    batch: parallel,
    kvBytes: gt(
      combinedKvBytes,
      cacheBytesK === input.overrides?.cacheBytesK || cacheBytesV === input.overrides?.cacheBytesV
        ? "override.cacheBytesK/cacheBytesV"
        : `cache-types.${cacheTypeK}/${cacheTypeV}`,
      input.overrides?.cacheBytesK !== undefined || input.overrides?.cacheBytesV !== undefined ? "user" : "runtime-default"
    ),
    gqa,
    utilization,
    userOverrides,
    modelSources: metadata.sources,
    runtimeSources: Object.values(defaults),
    notes: [
      "llama.cpp v1 estimate assumes full GPU offload equivalent to --gpu-layers 999.",
      "This is a deterministic estimate from exact GGUF metadata and selected runtime defaults, not observed runtime allocation."
    ],
    missing,
    kvFormulaLabel: `kv_cache = layers x hidden x context x parallel x (cache_bytes_k + cache_bytes_v) x gqa = ${layers?.value ?? "?"} x ${hiddenSize?.value ?? "?"} x ${context?.value ?? "?"} x ${parallel?.value ?? "?"} x (${cacheBytesK ?? "?"} + ${cacheBytesV ?? "?"}) x ${gqa?.value ?? "?"}`
  });
}

function valueWithOverride(
  override: number | string | undefined,
  metadata: number | string | undefined,
  overrideSource: string,
  metadataSource: string
): GroundTruthValue<number> | undefined {
  if (typeof override === "number" && Number.isFinite(override)) return gt(override, overrideSource, "user");
  if (typeof metadata === "number" && Number.isFinite(metadata)) return gt(metadata, metadataSource, "metadata");
  return undefined;
}

function exactGqa(
  override: number | undefined,
  attentionHeads: GroundTruthValue<number> | undefined,
  kvHeads: GroundTruthValue<number> | undefined
): GroundTruthValue<number> | undefined {
  if (override !== undefined) return gt(override, "override.gqa", "user");
  if (!attentionHeads || !kvHeads) return undefined;
  return gt(kvHeads.value / attentionHeads.value, `${kvHeads.source} / ${attentionHeads.source}`, "metadata");
}

function compact(values: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(values).filter(([, value]) => value !== undefined));
}
