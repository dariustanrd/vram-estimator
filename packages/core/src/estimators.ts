import { calculateEstimate, gt } from "./calculator.js";
import { fetchGgufMetadata, ggufNumber, ggufNumberArray, ggufString, cacheTypeBytes, ggufFileTypeName } from "./gguf.js";
import type { GgufMetadata } from "./gguf.js";
import {
  fetchHfModelMetadata,
  hfWeightDtype,
  kvBytesFromDtype,
  numberField,
  stringField
} from "./hf.js";
import type { HfConfig } from "./hf.js";
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
  const configSources = hfConfigSources(config);
  const userOverrides = compact({
    ...input.overrides,
    context: input.context,
    batch: input.batch,
    kvDtype: input.kvDtype,
    gpuMemoryUtilization: input.gpuMemoryUtilization,
    gpuVramGb: input.gpuVramGb,
    numGpus: input.numGpus
  });

  const layers = input.overrides?.layers !== undefined
    ? gt(input.overrides.layers, "override.layers", "user")
    : numberFromConfigs(configSources, ["num_hidden_layers", "n_layer", "num_layers"]);
  const hiddenSize = input.overrides?.hiddenSize !== undefined
    ? gt(input.overrides.hiddenSize, "override.hiddenSize", "user")
    : numberFromConfigs(configSources, ["hidden_size", "n_embd", "d_model"]);
  const context = input.context !== undefined || input.overrides?.context !== undefined
    ? gt(input.context ?? input.overrides?.context, "user.context", "user")
    : numberFromConfigs(configSources, ["max_position_embeddings", "max_seq_len", "seq_length", "model_max_length", "n_positions"], isUsableContextLength);
  const batch = valueWithOverride(
    input.batch ?? input.overrides?.batch,
    defaults.max_num_seqs?.value as number | undefined,
    input.batch !== undefined || input.overrides?.batch !== undefined ? "user.batch" : "runtime.max_num_seqs",
    "runtime.max_num_seqs",
    "runtime-default"
  );
  const attentionHeads = input.overrides?.attentionHeads !== undefined
    ? gt(input.overrides.attentionHeads, "override.attentionHeads", "user")
    : numberFromConfigs(configSources, ["num_attention_heads", "n_head", "num_heads"]);
  const kvHeads = input.overrides?.kvHeads !== undefined
    ? gt(input.overrides.kvHeads, "override.kvHeads", "user")
    : numberFromConfigs(configSources, ["num_key_value_heads", "num_kv_heads", "multi_query_group_num", "num_attention_groups"]);
  const gqa = exactGqa(input.overrides?.gqa, attentionHeads, kvHeads);

  const qkNopeHeadDim = numberFromConfigs(configSources, ["qk_nope_head_dim"]);
  const qkRopeHeadDim = numberFromConfigs(configSources, ["qk_rope_head_dim"]);
  const kvLoraRank = numberFromConfigs(configSources, ["kv_lora_rank"]);
  const hasMlaCache = Boolean(kvLoraRank && qkRopeHeadDim);
  const headDimExact = input.overrides?.headDim !== undefined
    ? gt(input.overrides.headDim, "override.headDim", "user")
    : numberFromConfigs(configSources, ["head_dim", "qk_head_dim"]);
  const derivedHeadDim = deriveHeadDim(hiddenSize, attentionHeads);
  const keyHeadDim = input.overrides?.headDim !== undefined
    ? headDimExact
    : qkNopeHeadDim && qkRopeHeadDim
      ? gt(
          qkNopeHeadDim.value + qkRopeHeadDim.value,
          `${qkNopeHeadDim.source} + ${qkRopeHeadDim.source}`,
          "metadata"
        )
      : headDimExact ?? derivedHeadDim;
  const valueHeadDim = input.overrides?.headDim !== undefined
    ? headDimExact
    : numberFromConfigs(configSources, ["v_head_dim"]) ?? headDimExact ?? derivedHeadDim;
  const headDim = headDimExact ?? derivedHeadDim;
  const headDimAssumed = !headDimExact && Boolean(derivedHeadDim);
  const kvHeadsResolved = kvHeads ?? deriveKvHeadsFromGqaOverride(input.overrides?.gqa, attentionHeads);
  const kvGroupWidth =
    !hasMlaCache && kvHeadsResolved && keyHeadDim && valueHeadDim && keyHeadDim.value === valueHeadDim.value
      ? gt(
          kvHeadsResolved.value * keyHeadDim.value,
          `${kvHeadsResolved.source} x ${keyHeadDim.source}`,
          kvHeadsResolved.providedBy === "user" || keyHeadDim.providedBy === "user" ? "user" : "metadata",
          headDimAssumed
        )
      : undefined;

  const configDtype = stringFromConfigs(configSources, ["torch_dtype", "dtype"]);
  const modelDtypeInfo = input.overrides?.modelDtype !== undefined
    ? { value: input.overrides.modelDtype, source: "override.modelDtype", providedBy: "user" as const }
    : configDtype
      ? { ...configDtype, providedBy: "metadata" as const }
      : metadata.safetensorsDtype
        ? { ...metadata.safetensorsDtype, providedBy: "metadata" as const }
        : undefined;
  const modelDtype = modelDtypeInfo?.value;
  const metadataWeightDtype = hfWeightDtypeFromConfigs(configSources, metadata.safetensorsDtype);
  const weightDtype = input.overrides?.modelDtype
    ? { value: input.overrides.modelDtype, source: "override.modelDtype", providedBy: "user" as const }
    : metadataWeightDtype
      ? { ...metadataWeightDtype, providedBy: "metadata" as const }
      : undefined;
  const defaultKvDtype = defaults.kv_cache_dtype?.value as string | undefined;
  const selectedKvDtype = input.kvDtype ?? defaultKvDtype;
  const resolvedKvDtype = selectedKvDtype === "auto" ? modelDtype : selectedKvDtype;
  const kvBytes = gt(
    input.overrides?.kvBytes ?? kvBytesFromDtype(resolvedKvDtype),
    input.overrides?.kvBytes !== undefined ? "override.kvBytes" : `dtype.${resolvedKvDtype}`,
    input.overrides?.kvBytes !== undefined ? "user" : "metadata"
  );
  const weightBytes = valueWithOverride(
    input.overrides?.weightBytes,
    metadata.weightBytes,
    "override.weightBytes",
    metadata.weightFiles.length > 0 ? "hf.weight_file_sizes" : "hf.metadata"
  );
  const utilization = valueWithOverride(
    input.gpuMemoryUtilization,
    defaults.gpu_memory_utilization?.value as number | undefined,
    "user.gpuMemoryUtilization",
    "runtime.gpu_memory_utilization",
    "runtime-default"
  );

  const missing: MissingValue[] = [];
  if (selectedKvDtype === "auto" && !modelDtype && input.overrides?.kvBytes === undefined) {
    missing.push({
      field: "kvBytes",
      reason: "vLLM kv_cache_dtype=auto requires a model torch_dtype/dtype, unambiguous safetensors dtype, or user kvBytes override."
    });
  }
  if (!hasMlaCache && attentionHeads && !kvHeadsResolved) {
    missing.push({
      field: "gqa",
      reason: "num_key_value_heads is absent; provide a kvHeads or gqa override to derive the exact GQA ratio."
    });
  }
  if (isDeepSeek4HfConfig(configSources)) {
    missing.push({
      field: "kvCache",
      reason: "DeepSeek-V4-style HF metadata indicates specialized compressed/indexer/cache state that is not represented by the current vLLM KV-cache model."
    });
  }
  if (!metadata.weightBytes && input.overrides?.weightBytes === undefined) {
    missing.push({
      field: "weightBytes",
      reason: "Hugging Face metadata did not expose exact model weight file sizes."
    });
  }

  const notes = [
    "vLLM v1 estimate is single GPU only with tensor_parallel_size=1.",
    "This is a deterministic estimate from exact metadata and selected runtime defaults, not observed runtime allocation.",
    `KV cache assumes ${batch?.value ?? "batch"} concurrent sequences each holding the configured ${context?.value ?? "context"}-token context simultaneously unless model metadata marks some layers as sliding-window or cacheless. Override context/batch to model your expected concurrency instead of relying on these defaults.`,
    "The hardware capacity card uses supply-side budgeting: available_kv_cache = gpu_vram x gpu_memory_utilization - weights. vLLM's runtime profiler can reserve additional non-KV memory for CUDA graphs, activations, and non-torch allocations, so observed server logs may report a smaller KV cache than this deterministic budget unless those reserves are modeled separately.",
    "\"Overhead\" is a modeled residual computed as (weights + kv_cache) / gpu_memory_utilization - weights - kv_cache. It approximates activation memory, CUDA graphs, and allocator overhead as a fixed proportion of the utilization setting; it is not vLLM's actual memory-profiler output, which depends on max_num_batched_tokens and intermediate size."
  ];
  if (configSources[0]?.prefix !== "config") {
    notes.push(`Resolved architecture fields from ${configSources[0]?.prefix}; top-level config.json appears to be a wrapper around the text model config.`);
  }
  if (headDimAssumed) {
    notes.push(
      "config.json does not expose head_dim explicitly; assumed head_dim = hidden_size / num_attention_heads. This assumption is wrong for architectures with a non-standard head_dim; use the headDim override if you know the model's real value."
    );
  }
  const slidingWindow = numberFromConfigs(configSources, ["sliding_window", "sliding_window_size"]);
  if (typeof slidingWindow?.value === "number" && context && slidingWindow.value < context.value) {
    notes.push(
      `${slidingWindow.source}=${slidingWindow.value}, smaller than the ${context.value}-token context used here. When layer metadata identifies sliding-window layers, the KV cache formula uses ${slidingWindow.value} tokens for those layers; otherwise verify the intended context/window behavior.`
    );
  }
  if (config.rope_scaling || configSources.some((source) => source.config.rope_scaling)) {
    notes.push(
      "config.json includes rope_scaling. max_position_embeddings may reflect either the base or the RoPE-scaled context length depending on how the checkpoint was published; verify the intended context length and use the context override if needed."
    );
  }

  const kvPlan = buildHfDirectKvPlan({
    configSources,
    layers,
    context,
    batch,
    kvBytes,
    kvHeads: kvHeadsResolved,
    keyHeadDim,
    valueHeadDim,
    kvLoraRank,
    qkRopeHeadDim,
    missing,
    notes
  });

  return calculateEstimate({
    mode: "vllm",
    weightBytes,
    layers: kvPlan ? undefined : layers,
    kvGroupWidth: kvPlan ? undefined : kvGroupWidth,
    context,
    batch,
    kvBytes: kvPlan ? undefined : kvBytes,
    kvCacheBytes: kvPlan?.kvCacheBytes,
    kvBytesPerToken: kvPlan?.kvBytesPerToken,
    utilization,
    userOverrides,
    modelSources: metadata.sources,
    modelSourceDetails: metadata.sourceDetails,
    runtimeSources: Object.values(defaults),
    notes,
    missing,
    hardware: {
      gpuVramGb: input.gpuVramGb,
      numGpus: input.numGpus
    },
    kvFormulaLabel: kvPlan?.formula,
    displayValues: {
      layers,
      hiddenSize,
      attentionHeads,
      kvHeads: kvHeadsResolved,
      headDim,
      headDimK: keyHeadDim,
      headDimV: valueHeadDim,
      qkNopeHeadDim,
      qkRopeHeadDim,
      kvLoraRank,
      gqa,
      kvBytes,
      weightDtype,
      ...kvPlan?.displayValues
    }
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
    cacheTypeV: input.cacheTypeV,
    gpuVramGb: input.gpuVramGb,
    numGpus: input.numGpus
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
    "runtime.parallel",
    "runtime-default"
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
  const attentionHeadsArray = arch ? ggufNumberArray(metadata, `${prefix}attention.head_count`) : undefined;
  const kvHeadsArray = input.overrides?.kvHeads === undefined && arch ? ggufNumberArray(metadata, `${prefix}attention.head_count_kv`) : undefined;
  const gqa = exactGqa(input.overrides?.gqa, attentionHeads, kvHeads);
  const kvHeadsResolved = kvHeads ?? deriveKvHeadsFromGqaOverride(input.overrides?.gqa, attentionHeads);
  const kvCachelessArchitecture = isKvCachelessGgufArchitecture(arch);

  // llama.cpp's GGUF writer stores the true per-head K/V dimensions as attention.key_length /
  // attention.value_length for most architectures. Prefer those over the hidden_size / head_count
  // ratio, which silently breaks for models with a non-standard head_dim (Gemma2/3, etc).
  const keyHeadDim = valueWithOverride(
    input.overrides?.headDim,
    arch ? ggufNumber(metadata, `${prefix}attention.key_length`) : undefined,
    "override.headDim",
    `${prefix}attention.key_length`
  );
  const valueHeadDim = valueWithOverride(
    input.overrides?.headDim,
    arch ? ggufNumber(metadata, `${prefix}attention.value_length`) : undefined,
    "override.headDim",
    `${prefix}attention.value_length`
  );
  const fallbackHeadDim = kvCachelessArchitecture ? undefined : deriveHeadDim(hiddenSize, attentionHeads);
  const headDimK = keyHeadDim ?? fallbackHeadDim;
  const headDimV = valueHeadDim ?? fallbackHeadDim;
  const headDimAssumed = !keyHeadDim && !valueHeadDim && Boolean(fallbackHeadDim);
  const asymmetricHeadDim = Boolean(keyHeadDim && valueHeadDim && keyHeadDim.value !== valueHeadDim.value);
  // The shared kv_cache formula (calculator.ts) uses a single averaged K/V element width. This is
  // exact whenever key_length === value_length (true for essentially every published llama.cpp
  // model) and only approximate for architectures that report asymmetric K/V dimensions, flagged
  // via a note below.
  const avgHeadDim =
    headDimK && headDimV
      ? gt(
          (headDimK.value + headDimV.value) / 2,
          asymmetricHeadDim ? `avg(${headDimK.source}, ${headDimV.source})` : headDimK.source,
          "metadata",
          headDimAssumed
        )
      : undefined;
  const kvGroupWidth = kvCachelessArchitecture
    ? gt(0, `${arch} architecture has no persistent autoregressive KV cache in llama.cpp`, "metadata")
    : kvHeadsResolved && avgHeadDim
      ? gt(
          kvHeadsResolved.value * avgHeadDim.value,
          `${kvHeadsResolved.source} x ${avgHeadDim.source}`,
          kvHeadsResolved.providedBy === "user" ? "user" : "metadata",
          headDimAssumed
        )
      : undefined;

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
  if (metadata.split && !metadata.split.complete && input.overrides?.weightBytes === undefined) {
    missing.push({
      field: "weightBytes",
      reason: `GGUF source is split into ${metadata.split.count} files, but the estimator could not fetch and validate all sibling split metadata to sum tensor tables.`
    });
  }
  if (arch === "deepseek4") {
    missing.push({
      field: "kvCache",
      reason: "llama.cpp DeepSeek-V4 uses specialized DSV4 compressed/indexer/cache state that is not represented by the generic GGUF KV metadata yet."
    });
  }
  if (kvHeadsArray && layers && kvHeadsArray.length !== layers.value) {
    missing.push({
      field: "kvHeads",
      reason: `GGUF ${prefix}attention.head_count_kv array has ${kvHeadsArray.length} entries for ${layers.value} layers; the estimator will not repeat or truncate a per-layer array silently.`
    });
  }
  if (!kvCachelessArchitecture && (attentionHeads || attentionHeadsArray) && !kvHeadsResolved && !kvHeadsArray) {
    missing.push({
      field: "gqa",
      reason: "GGUF attention.head_count_kv is absent; provide a kvHeads or gqa override to derive the exact GQA ratio."
    });
  }

  const notes = [
    "llama.cpp v1 estimate assumes full GPU offload equivalent to --gpu-layers 999.",
    "This is a deterministic estimate from exact GGUF metadata and selected runtime defaults, not observed runtime allocation.",
    kvCachelessArchitecture
      ? `${arch} does not allocate a persistent autoregressive KV cache in llama.cpp. context still limits input tokens per slot, but the KV-cache term in this estimate is 0.`
      : "context is treated as the desired context PER PARALLEL SLOT; total persistent KV cache reserved = context x parallel, matching how llama.cpp's --ctx-size total budget is divided across --parallel slots. If you already have a fixed total --ctx-size in mind, set parallel=1 and pass that total directly as context."
  ];
  if (defaults.runtime_utilization?.value === 1) {
    notes.push(
      "The default runtime_utilization=1 assumes no extra reservation, so overhead will show as ~0. This does NOT include llama.cpp's ggml compute/graph buffer, which typically adds hundreds of MB to a few GB depending on context size, batch size, and flash-attention settings. Lower the utilization or add manual headroom for a safer real-world estimate."
    );
  }
  if (!kvCachelessArchitecture && headDimAssumed) {
    notes.push(
      `${prefix}attention.key_length / ${prefix}attention.value_length are absent from this GGUF file; assumed head_dim = embedding_length / attention.head_count. This assumption is wrong for architectures with a non-standard head_dim (e.g. Gemma2/3); use the headDim override if you know the model's real value.`
    );
  }
  if (!kvCachelessArchitecture && asymmetricHeadDim) {
    notes.push(
      "This architecture reports different key_length and value_length (asymmetric K/V head dimensions); the estimator uses K and V byte sizes separately when a direct per-layer KV formula is available."
    );
  }
  if (kvCachelessArchitecture) {
    notes.push(
      "Only persistent weights and persistent KV cache are counted here. Temporary activations, graph buffers, backend workspaces, allocator padding, tokenizer/model structures, and mmap/accounting effects are runtime-dependent and are not currently modeled, so real RAM/VRAM usage should be higher than this persistent-memory total."
    );
  }
  const slidingWindow = arch ? ggufNumber(metadata, `${prefix}attention.sliding_window`) : undefined;
  if (!kvCachelessArchitecture && typeof slidingWindow === "number" && context && slidingWindow < context.value) {
    notes.push(
      `GGUF metadata reports ${prefix}attention.sliding_window=${slidingWindow}, smaller than the ${context.value}-token context used here. When GGUF metadata identifies SWA layers, the KV formula uses the sliding-window length for those layers.`
    );
  }
  if (metadata.split?.complete) {
    notes.push(`Detected split GGUF metadata and summed tensor tables across ${metadata.split.count} files for exact weight bytes.`);
  }

  const directKvPlan = buildGgufDirectKvPlan({
    metadata,
    arch,
    prefix,
    layers,
    context,
    parallel,
    cacheBytesK,
    cacheBytesV,
    kvHeads: kvHeadsResolved,
    kvHeadsArray,
    headDimK,
    headDimV,
    kvCachelessArchitecture,
    missing,
    notes
  });

  return calculateEstimate({
    mode: "llamacpp",
    weightBytes,
    layers: directKvPlan ? undefined : layers,
    kvGroupWidth: directKvPlan ? undefined : kvGroupWidth,
    context,
    batch: parallel,
    kvBytes: directKvPlan
      ? undefined
      : kvCachelessArchitecture
        ? gt(0, `${arch} architecture has no persistent autoregressive KV cache in llama.cpp`, "metadata")
        : gt(
            combinedKvBytes,
            cacheBytesK === input.overrides?.cacheBytesK || cacheBytesV === input.overrides?.cacheBytesV
              ? "override.cacheBytesK/cacheBytesV"
              : `cache-types.${cacheTypeK}/${cacheTypeV}`,
            input.overrides?.cacheBytesK !== undefined || input.overrides?.cacheBytesV !== undefined ? "user" : "runtime-default"
          ),
    kvCacheBytes: directKvPlan?.kvCacheBytes,
    kvBytesPerToken: directKvPlan?.kvBytesPerToken,
    utilization,
    userOverrides,
    modelSources: metadata.sources,
    runtimeSources: Object.values(defaults),
    notes,
    missing,
    overheadCaveat: kvCachelessArchitecture
      ? "Note: this 0 is only the estimator's residual after weights + persistent KV cache. llama.cpp still needs runtime memory for temporary activations, graph buffers, backend workspaces, allocator padding, tokenizer/model structures, and possibly mmap/accounting effects; this GGUF-only calculation cannot determine that overhead."
      : undefined,
    hardware: {
      gpuVramGb: input.gpuVramGb,
      numGpus: input.numGpus
    },
    kvFormulaLabel: directKvPlan?.formula ?? (kvCachelessArchitecture
      ? `kv_cache = 0\n\t${arch} has no persistent autoregressive KV cache in llama.cpp`
      : `kv_cache = layers x kv_group_width x context x parallel x (cache_bytes_k + cache_bytes_v) \n\t= ${layers?.value ?? "?"} x ${kvGroupWidth?.value ?? "?"} x ${context?.value ?? "?"} x ${parallel?.value ?? "?"} x (${cacheBytesK ?? "?"} + ${cacheBytesV ?? "?"}) \n\t= ${llamaCppKvCacheBytes(layers, kvGroupWidth, context, parallel, cacheBytesK, cacheBytesV) ?? "?"}`),
    displayValues: {
      layers,
      hiddenSize,
      attentionHeads,
      kvHeads: kvHeadsResolved,
      headDimK,
      headDimV,
      gqa,
      cacheBytesK: kvCachelessArchitecture ? undefined : gt(cacheBytesK, input.overrides?.cacheBytesK !== undefined ? "override.cacheBytesK" : `cache-type.${cacheTypeK}`, input.overrides?.cacheBytesK !== undefined ? "user" : "runtime-default"),
      cacheBytesV: kvCachelessArchitecture ? undefined : gt(cacheBytesV, input.overrides?.cacheBytesV !== undefined ? "override.cacheBytesV" : `cache-type.${cacheTypeV}`, input.overrides?.cacheBytesV !== undefined ? "user" : "runtime-default"),
      weightDtype: gt(arch ? ggufFileTypeName(metadata) : undefined, "general.file_type", "metadata"),
      ...directKvPlan?.displayValues
    }
  });
}

type HfConfigSource = { config: HfConfig; prefix: string };

type DirectKvPlan = {
  kvCacheBytes: GroundTruthValue<number>;
  kvBytesPerToken: GroundTruthValue<number>;
  formula: string;
  displayValues?: Record<string, GroundTruthValue<unknown> | undefined> | undefined;
};

type LayerContextPlan = {
  contexts: number[];
  source: string;
  hasSpecialContexts: boolean;
  recurrentLayers: number;
};

function hfConfigSources(config: HfConfig): HfConfigSource[] {
  const sources: HfConfigSource[] = [];
  const seen = new Set<HfConfig>();
  for (const field of ["text_config", "language_config", "llm_config", "model_config"]) {
    const nested = objectConfigField(config, field);
    if (nested && !seen.has(nested)) {
      seen.add(nested);
      sources.push({ config: nested, prefix: `config.${field}` });
    }
  }
  if (!seen.has(config)) sources.push({ config, prefix: "config" });
  return sources;
}

function hfWeightDtypeFromConfigs(
  sources: HfConfigSource[],
  fallback?: { value: string; source: string } | undefined
): { value: string; source: string } | undefined {
  for (const source of sources) {
    const dtype = hfWeightDtype(source.config);
    if (dtype) {
      return {
        value: dtype.value,
        source: rebaseConfigSource(dtype.source, source.prefix)
      };
    }
  }
  return fallback;
}

function rebaseConfigSource(source: string, prefix: string): string {
  return source.startsWith("config.") ? `${prefix}${source.slice("config".length)}` : source;
}

function objectConfigField(config: HfConfig, field: string): HfConfig | undefined {
  const value = config[field];
  return value && typeof value === "object" && !Array.isArray(value) ? (value as HfConfig) : undefined;
}

function numberFromConfigs(
  sources: HfConfigSource[],
  fields: string[],
  predicate: (value: number) => boolean = () => true
): GroundTruthValue<number> | undefined {
  for (const source of sources) {
    for (const field of fields) {
      const value = numberField(source.config, field);
      if (value !== undefined && predicate(value)) return gt(value, `${source.prefix}.${field}`, "metadata");
    }
  }
  return undefined;
}

function stringFromConfigs(
  sources: HfConfigSource[],
  fields: string[]
): { value: string; source: string } | undefined {
  for (const source of sources) {
    for (const field of fields) {
      const value = stringField(source.config, field);
      if (value !== undefined) return { value, source: `${source.prefix}.${field}` };
    }
  }
  return undefined;
}

function numberArrayFromConfigs(sources: HfConfigSource[], fields: string[]): { value: number[]; source: string } | undefined {
  for (const source of sources) {
    for (const field of fields) {
      const value = source.config[field];
      if (Array.isArray(value) && value.length > 0 && value.every((item) => typeof item === "number" && Number.isFinite(item))) {
        return { value: value as number[], source: `${source.prefix}.${field}` };
      }
    }
  }
  return undefined;
}

function stringArrayFromConfigs(sources: HfConfigSource[], fields: string[]): { value: string[]; source: string } | undefined {
  for (const source of sources) {
    for (const field of fields) {
      const value = source.config[field];
      if (Array.isArray(value) && value.length > 0 && value.every((item) => typeof item === "string")) {
        return { value: value as string[], source: `${source.prefix}.${field}` };
      }
    }
  }
  return undefined;
}

function isUsableContextLength(value: number): boolean {
  // Tokenizer/model configs sometimes use enormous sentinel values (e.g. 1e30)
  // to mean "unset"; those should not become VRAM requirements.
  return value > 0 && value <= 10_000_000;
}

function isDeepSeek4HfConfig(sources: HfConfigSource[]): boolean {
  for (const source of sources) {
    const modelType = stringField(source.config, "model_type")?.toLowerCase();
    if (modelType && /deepseek[-_]?v?4/.test(modelType)) return true;
    const architectures = source.config.architectures;
    if (Array.isArray(architectures)) {
      for (const architecture of architectures) {
        if (typeof architecture === "string" && /deepseek[-_]?v?4/i.test(architecture)) return true;
      }
    }
  }
  return false;
}

function buildHfDirectKvPlan(input: {
  configSources: HfConfigSource[];
  layers?: GroundTruthValue<number> | undefined;
  context?: GroundTruthValue<number> | undefined;
  batch?: GroundTruthValue<number> | undefined;
  kvBytes?: GroundTruthValue<number> | undefined;
  kvHeads?: GroundTruthValue<number> | undefined;
  keyHeadDim?: GroundTruthValue<number> | undefined;
  valueHeadDim?: GroundTruthValue<number> | undefined;
  kvLoraRank?: GroundTruthValue<number> | undefined;
  qkRopeHeadDim?: GroundTruthValue<number> | undefined;
  missing: MissingValue[];
  notes: string[];
}): DirectKvPlan | undefined {
  if (!input.layers || !input.context || !input.batch || !input.kvBytes) return undefined;

  const layerContexts = resolveHfLayerContexts(input.configSources, input.layers.value, input.context.value);
  if (layerContexts.recurrentLayers > 0) {
    input.missing.push({
      field: "stateCache",
      reason: `${layerContexts.source} identifies ${layerContexts.recurrentLayers} recurrent/SSM layers. vLLM state-cache memory for hybrid recurrent models is not yet modeled, so the estimator will not return a partial KV-only total.`
    });
  }

  if (input.kvLoraRank && input.qkRopeHeadDim) {
    const mlaWidth = input.kvLoraRank.value + input.qkRopeHeadDim.value;
    const contextSum = layerContexts.contexts.reduce((sum, value) => sum + value, 0);
    let kvCacheBytes = contextSum * input.batch.value * mlaWidth * input.kvBytes.value;
    const formulaParts = [
      `mla_cache = sum(layer_contexts) x batch x (kv_lora_rank + qk_rope_head_dim) x kv_bytes_per_element`,
      `\t= ${contextSum} x ${input.batch.value} x (${input.kvLoraRank.value} + ${input.qkRopeHeadDim.value}) x ${input.kvBytes.value}`,
      `\t= ${kvCacheBytes}`
    ];

    const indexer = hfIndexerCacheBytes(input.configSources, input.layers.value, input.context.value, input.batch.value);
    if (indexer) {
      kvCacheBytes += indexer.bytes;
      formulaParts.push(
        `indexer_cache = ${indexer.activeLayers} layers x ${input.context.value} context x ${input.batch.value} batch x ${indexer.bytesPerTokenPerLayer} bytes/token/layer`,
        `\t= ${indexer.bytes}`,
        `kv_cache = mla_cache + indexer_cache = ${kvCacheBytes}`
      );
      input.notes.push(
        `Detected sparse-attention indexer metadata; included ${indexer.activeLayers} uint8 indexer cache layers in addition to the MLA latent cache.`
      );
    } else {
      formulaParts.push(`kv_cache = ${kvCacheBytes}`);
    }

    if (layerContexts.hasSpecialContexts) {
      input.notes.push(`KV cache uses per-layer effective contexts from ${layerContexts.source}.`);
    }

    return {
      kvCacheBytes: gt(kvCacheBytes, "hf.mla_kv_cache_formula", "metadata")!,
      kvBytesPerToken: gt(kvCacheBytes / (input.context.value * input.batch.value), "hf.mla_kv_cache_formula / context / batch", "metadata")!,
      formula: formulaParts.join(" \n"),
      displayValues: {
        kvCacheWidth: gt(mlaWidth, `${input.kvLoraRank.source} + ${input.qkRopeHeadDim.source}`, "metadata"),
        attentionLayerContexts: gt(layerContexts.contexts.join(","), layerContexts.source, "metadata")
      }
    };
  }

  if (!input.kvHeads || !input.keyHeadDim || !input.valueHeadDim) return undefined;

  const globalKvHeads = numberFromConfigs(input.configSources, ["num_global_key_value_heads"]);
  const globalHeadDim = numberFromConfigs(input.configSources, ["global_head_dim"]);
  const layerTypes = stringArrayFromConfigs(input.configSources, ["layer_types", "hybrid_layer_pattern", "layers_block_type"]);
  const hasAsymmetricHeadDims = input.keyHeadDim.value !== input.valueHeadDim.value;
  const hasGlobalAttentionDims = Boolean(globalKvHeads && globalHeadDim && layerTypes);
  if (!layerContexts.hasSpecialContexts && !hasAsymmetricHeadDims && !hasGlobalAttentionDims) return undefined;

  let elementContextSum = 0;
  for (let layer = 0; layer < input.layers.value; layer++) {
    const type = layerTypes ? repeatedAt(layerTypes.value, layer).toLowerCase() : "";
    const useGlobal = hasGlobalAttentionDims && (type.includes("global") || type.includes("full"));
    const kvHeads = useGlobal ? globalKvHeads!.value : input.kvHeads.value;
    const keyDim = useGlobal ? globalHeadDim!.value : input.keyHeadDim.value;
    const valueDim = useGlobal ? globalHeadDim!.value : input.valueHeadDim.value;
    elementContextSum += layerContexts.contexts[layer]! * kvHeads * (keyDim + valueDim);
  }

  const kvCacheBytes = elementContextSum * input.batch.value * input.kvBytes.value;
  if (layerContexts.hasSpecialContexts) {
    input.notes.push(`KV cache uses per-layer effective contexts from ${layerContexts.source}.`);
  }
  if (hasGlobalAttentionDims) {
    input.notes.push("Detected separate local/global attention KV metadata; full/global layers use global KV heads and head_dim, local/sliding layers use local KV heads and head_dim.");
  }

  return {
    kvCacheBytes: gt(kvCacheBytes, "hf.per_layer_kv_cache_formula", "metadata")!,
    kvBytesPerToken: gt(kvCacheBytes / (input.context.value * input.batch.value), "hf.per_layer_kv_cache_formula / context / batch", "metadata")!,
    formula: `kv_cache = sum_over_layers(effective_context_layer x kv_heads_layer x (key_head_dim_layer + value_head_dim_layer)) x batch x kv_bytes_per_element \n\t= ${elementContextSum} x ${input.batch.value} x ${input.kvBytes.value} \n\t= ${kvCacheBytes}`,
    displayValues: {
      attentionLayerContexts: gt(layerContexts.contexts.join(","), layerContexts.source, "metadata")
    }
  };
}

function resolveHfLayerContexts(sources: HfConfigSource[], layerCount: number, context: number): LayerContextPlan {
  const mlpOnlyLayers = numberArrayFromConfigs(sources, ["mlp_only_layers"]);
  const layerTypes = stringArrayFromConfigs(sources, ["layer_types", "hybrid_layer_pattern", "layers_block_type"]);
  const slidingWindow = numberFromConfigs(sources, ["sliding_window", "sliding_window_size", "attention_window"]);
  const slidingPattern = numberFromConfigs(sources, ["sliding_window_pattern", "swa_pattern"]);
  const contexts = Array.from({ length: layerCount }, () => context);
  const mlpOnly = new Set((mlpOnlyLayers?.value ?? []).map((value) => Math.trunc(value)));
  let recurrentLayers = 0;
  let hasSpecialContexts = false;

  for (let layer = 0; layer < layerCount; layer++) {
    if (mlpOnly.has(layer)) {
      contexts[layer] = 0;
      hasSpecialContexts = true;
      continue;
    }

    const rawType = layerTypes ? repeatedAt(layerTypes.value, layer).toLowerCase() : "";
    if (rawType.includes("mamba") || rawType.includes("recurrent") || rawType.includes("ssm")) {
      contexts[layer] = 0;
      recurrentLayers++;
      hasSpecialContexts = true;
      continue;
    }
    if (rawType.includes("mlp") && !rawType.includes("attention") && !rawType.includes("attn")) {
      contexts[layer] = 0;
      hasSpecialContexts = true;
      continue;
    }
    const usesSlidingWindow = slidingWindow && (
      !layerTypes && !slidingPattern ||
        slidingPattern && isPatternSwaOrRecurrent(layer, slidingPattern.value, false) ||
        rawType.includes("sliding") ||
        rawType.includes("local") ||
        rawType.includes("swa")
    );
    if (usesSlidingWindow) {
      contexts[layer] = Math.min(context, slidingWindow.value);
      hasSpecialContexts = hasSpecialContexts || contexts[layer] !== context;
    }
  }

  const sourceParts = compactStrings([layerTypes?.source, mlpOnlyLayers?.source, slidingWindow?.source, slidingPattern?.source]);
  return {
    contexts,
    source: sourceParts.length > 0 ? sourceParts.join(" + ") : "uniform full-context layers",
    hasSpecialContexts,
    recurrentLayers
  };
}

function hfIndexerCacheBytes(
  sources: HfConfigSource[],
  layerCount: number,
  context: number,
  batch: number
): { bytes: number; activeLayers: number; bytesPerTokenPerLayer: number } | undefined {
  const indexTopk = numberFromConfigs(sources, ["index_topk"]);
  const indexHeadDim = numberFromConfigs(sources, ["index_head_dim"]);
  if (!indexTopk || !indexHeadDim) return undefined;
  const quantBlockSize = numberFromConfigs(sources, ["index_quant_block_size"])?.value ?? 128;
  const pattern = stringArrayFromConfigs(sources, ["index_topk_pattern"]);
  const frequency = numberFromConfigs(sources, ["index_topk_freq"])?.value ?? 1;
  const skipOffset = numberFromConfigs(sources, ["index_skip_topk_offset"])?.value ?? 2;
  let activeLayers = 0;
  for (let layer = 0; layer < layerCount; layer++) {
    const skip = pattern
      ? repeatedAt(pattern.value, layer).toLowerCase() === "s"
      : Math.max(layer - skipOffset + 1, 0) % frequency !== 0;
    if (!skip) activeLayers++;
  }
  const bytesPerTokenPerLayer = indexHeadDim.value + Math.floor(indexHeadDim.value / quantBlockSize) * 4;
  return {
    bytes: activeLayers * context * batch * bytesPerTokenPerLayer,
    activeLayers,
    bytesPerTokenPerLayer
  };
}

function buildGgufDirectKvPlan(input: {
  metadata: GgufMetadata;
  arch: string | undefined;
  prefix: string;
  layers?: GroundTruthValue<number> | undefined;
  context?: GroundTruthValue<number> | undefined;
  parallel?: GroundTruthValue<number> | undefined;
  cacheBytesK?: number | undefined;
  cacheBytesV?: number | undefined;
  kvHeads?: GroundTruthValue<number> | undefined;
  kvHeadsArray?: number[] | undefined;
  headDimK?: GroundTruthValue<number> | undefined;
  headDimV?: GroundTruthValue<number> | undefined;
  kvCachelessArchitecture: boolean;
  missing: MissingValue[];
  notes: string[];
}): DirectKvPlan | undefined {
  if (input.kvCachelessArchitecture) return undefined;
  if (!input.layers || !input.context || !input.parallel) return undefined;
  if (input.cacheBytesK === undefined || input.cacheBytesV === undefined) return undefined;
  if (!input.headDimK || !input.headDimV) return undefined;

  const layerContexts = resolveGgufLayerContexts(input.metadata, input.prefix, input.layers.value, input.context.value);
  if (layerContexts.recurrentLayers > 0) {
    input.missing.push({
      field: "stateCache",
      reason: `${layerContexts.source} identifies ${layerContexts.recurrentLayers} recurrent/SSM layers. llama.cpp recurrent state memory is not yet modeled, so the estimator will not return a partial KV-only total.`
    });
  }

  const kvHeadsArray = input.kvHeadsArray;
  if (kvHeadsArray && kvHeadsArray.length !== input.layers.value) return undefined;
  const keyDimSwa = ggufNumber(input.metadata, `${input.prefix}attention.key_length_swa`);
  const valueDimSwa = ggufNumber(input.metadata, `${input.prefix}attention.value_length_swa`);
  const needsDirect = Boolean(
    kvHeadsArray ||
      layerContexts.hasSpecialContexts ||
      keyDimSwa !== undefined ||
      valueDimSwa !== undefined ||
      input.headDimK.value !== input.headDimV.value
  );
  if (!needsDirect) return undefined;
  if (!input.kvHeads && !kvHeadsArray) return undefined;

  let kvCacheBytes = 0;
  let elementBytesPerFullContext = 0;
  for (let layer = 0; layer < input.layers.value; layer++) {
    const kvHeads = kvHeadsArray ? repeatedAt(kvHeadsArray, layer) : input.kvHeads!.value;
    if (kvHeads === 0) continue;
    const isSwa = layerContexts.swaLayers.has(layer);
    const keyDim = isSwa && keyDimSwa !== undefined ? keyDimSwa : input.headDimK.value;
    const valueDim = isSwa && valueDimSwa !== undefined ? valueDimSwa : input.headDimV.value;
    const context = layerContexts.contexts[layer]!;
    kvCacheBytes += context * input.parallel.value * kvHeads * (keyDim * input.cacheBytesK + valueDim * input.cacheBytesV);
    elementBytesPerFullContext += context * kvHeads * (keyDim * input.cacheBytesK + valueDim * input.cacheBytesV);
  }

  if (kvHeadsArray) {
    input.notes.push(`GGUF ${input.prefix}attention.head_count_kv is per-layer; KV cache sums each layer instead of assuming one scalar KV-head count.`);
  }
  if (layerContexts.hasSpecialContexts) {
    input.notes.push(`KV cache uses per-layer effective contexts from ${layerContexts.source}.`);
  }
  if (keyDimSwa !== undefined || valueDimSwa !== undefined) {
    input.notes.push("Detected separate SWA key/value head dimensions; SWA layers use attention.key_length_swa/value_length_swa when present.");
  }

  return {
    kvCacheBytes: gt(kvCacheBytes, "gguf.per_layer_kv_cache_formula", "metadata")!,
    kvBytesPerToken: gt(kvCacheBytes / (input.context.value * input.parallel.value), "gguf.per_layer_kv_cache_formula / context / parallel", "metadata")!,
    formula: `kv_cache = sum_over_layers(effective_context_layer x kv_heads_layer x (key_head_dim_layer x cache_bytes_k + value_head_dim_layer x cache_bytes_v)) x parallel \n\t= ${elementBytesPerFullContext} x ${input.parallel.value} \n\t= ${kvCacheBytes}`,
    displayValues: {
      attentionLayerContexts: gt(layerContexts.contexts.join(","), layerContexts.source, "metadata"),
      kvHeadsPerLayer: kvHeadsArray ? gt(kvHeadsArray.join(","), `${input.prefix}attention.head_count_kv`, "metadata") : undefined
    }
  };
}

function resolveGgufLayerContexts(
  metadata: GgufMetadata,
  prefix: string,
  layerCount: number,
  context: number
): LayerContextPlan & { swaLayers: Set<number> } {
  const contexts = Array.from({ length: layerCount }, () => context);
  const swaLayers = new Set<number>();
  const slidingWindow = ggufNumber(metadata, `${prefix}attention.sliding_window`);
  const slidingPattern = ggufNumber(metadata, `${prefix}attention.sliding_window_pattern`);
  const recurrentPattern = ggufNumber(metadata, `${prefix}recurrent_pattern`) ?? ggufNumber(metadata, `${prefix}ssm.recurrent_pattern`);
  let recurrentLayers = 0;
  let hasSpecialContexts = false;

  for (let layer = 0; layer < layerCount; layer++) {
    if (recurrentPattern !== undefined && isPatternSwaOrRecurrent(layer, recurrentPattern, false)) {
      contexts[layer] = 0;
      recurrentLayers++;
      hasSpecialContexts = true;
      continue;
    }
    const usesSlidingWindow = slidingWindow !== undefined && (
      slidingPattern === undefined || isPatternSwaOrRecurrent(layer, slidingPattern, false)
    );
    if (usesSlidingWindow) {
      contexts[layer] = Math.min(context, slidingWindow);
      swaLayers.add(layer);
      hasSpecialContexts = hasSpecialContexts || contexts[layer] !== context;
    }
  }

  const kvHeadsArray = ggufNumberArray(metadata, `${prefix}attention.head_count_kv`);
  if (kvHeadsArray) {
    for (let layer = 0; layer < Math.min(layerCount, kvHeadsArray.length); layer++) {
      if (kvHeadsArray[layer] === 0) {
        contexts[layer] = 0;
        hasSpecialContexts = true;
      }
    }
  }

  const sourceParts = compactStrings([
    slidingWindow !== undefined ? `${prefix}attention.sliding_window` : undefined,
    slidingPattern !== undefined ? `${prefix}attention.sliding_window_pattern` : undefined,
    recurrentPattern !== undefined ? `${prefix}recurrent_pattern` : undefined,
    kvHeadsArray ? `${prefix}attention.head_count_kv` : undefined
  ]);
  return {
    contexts,
    source: sourceParts.length > 0 ? sourceParts.join(" + ") : "uniform full-context layers",
    hasSpecialContexts,
    recurrentLayers,
    swaLayers
  };
}

function isPatternSwaOrRecurrent(layer: number, pattern: number, denseFirst: boolean): boolean {
  if (pattern <= 0) return true;
  if (pattern === 1) return false;
  return denseFirst ? layer % pattern !== 0 : layer % pattern < pattern - 1;
}

function repeatedAt<T>(values: T[], index: number): T {
  return values[index % values.length]!;
}

function compactStrings(values: Array<string | undefined>): string[] {
  return values.filter((value): value is string => Boolean(value));
}

function valueWithOverride(
  override: number | string | undefined,
  fallback: number | string | undefined,
  overrideSource: string,
  fallbackSource: string,
  fallbackProvidedBy: GroundTruthValue<number>["providedBy"] = "metadata"
): GroundTruthValue<number> | undefined {
  if (typeof override === "number" && Number.isFinite(override)) return gt(override, overrideSource, "user");
  if (typeof fallback === "number" && Number.isFinite(fallback)) return gt(fallback, fallbackSource, fallbackProvidedBy);
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

function deriveHeadDim(
  hiddenSize: GroundTruthValue<number> | undefined,
  attentionHeads: GroundTruthValue<number> | undefined
): GroundTruthValue<number> | undefined {
  if (!hiddenSize || !attentionHeads) return undefined;
  return gt(
    hiddenSize.value / attentionHeads.value,
    `${hiddenSize.source} / ${attentionHeads.source} (assumed head_dim)`,
    "metadata",
    true
  );
}

function deriveKvHeadsFromGqaOverride(
  gqaOverride: number | undefined,
  attentionHeads: GroundTruthValue<number> | undefined
): GroundTruthValue<number> | undefined {
  if (gqaOverride === undefined || !attentionHeads) return undefined;
  return gt(attentionHeads.value * gqaOverride, `override.gqa x ${attentionHeads.source}`, "user");
}

function llamaCppKvCacheBytes(
  layers: GroundTruthValue<number> | undefined,
  kvGroupWidth: GroundTruthValue<number> | undefined,
  context: GroundTruthValue<number> | undefined,
  parallel: GroundTruthValue<number> | undefined,
  cacheBytesK: number | undefined,
  cacheBytesV: number | undefined
): number | undefined {
  if (!layers || !kvGroupWidth || !context || !parallel || cacheBytesK === undefined || cacheBytesV === undefined) {
    return undefined;
  }
  return layers.value * kvGroupWidth.value * context.value * parallel.value * (cacheBytesK + cacheBytesV);
}

const LLAMA_CPP_CACHELESS_GGUF_ARCHITECTURES = new Set([
  "bert",
  "dream",
  "eurobert",
  "gemma-embedding",
  "jina-bert-v2",
  "jina-bert-v3",
  "llada",
  "llada-moe",
  "modern-bert",
  "neo-bert",
  "nomic-bert",
  "nomic-bert-moe",
  "rnd1",
  "wavtokenizer-dec"
]);

function isKvCachelessGgufArchitecture(arch: string | undefined): boolean {
  return arch !== undefined && LLAMA_CPP_CACHELESS_GGUF_ARCHITECTURES.has(arch);
}

function compact(values: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(values).filter(([, value]) => value !== undefined));
}
