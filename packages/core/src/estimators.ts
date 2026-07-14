import { calculateEstimate, gt } from "./calculator.js";
import { bytesForDtype } from "./dtypes.js";
import { fetchGgufMetadata, ggufNumber, ggufString, cacheTypeBytes, ggufFileTypeName } from "./gguf.js";
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
    kvDtype: input.kvDtype,
    gpuMemoryUtilization: input.gpuMemoryUtilization,
    gpuVramGb: input.gpuVramGb,
    numGpus: input.numGpus
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
    "runtime.max_num_seqs",
    "runtime-default"
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

  // head_dim is only equal to hidden_size / num_attention_heads for "standard" MHA/GQA layouts.
  // Several current architectures (Gemma2/3, Qwen2/2.5, etc.) publish an explicit head_dim that
  // differs from that ratio; prefer it whenever config.json exposes it.
  const headDimExact = valueWithOverride(
    input.overrides?.headDim,
    numberField(config, "head_dim"),
    "override.headDim",
    "config.head_dim"
  );
  const headDim = headDimExact ?? deriveHeadDim(hiddenSize, attentionHeads);
  const headDimAssumed = !headDimExact && Boolean(headDim);
  const kvHeadsResolved = kvHeads ?? deriveKvHeadsFromGqaOverride(input.overrides?.gqa, attentionHeads);
  const kvGroupWidth =
    kvHeadsResolved && headDim
      ? gt(
          kvHeadsResolved.value * headDim.value,
          `${kvHeadsResolved.source} x ${headDim.source}`,
          kvHeadsResolved.providedBy === "user" || headDimExact?.providedBy === "user" ? "user" : "metadata",
          headDimAssumed
        )
      : undefined;

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
      reason: "vLLM kv_cache_dtype=auto requires an explicit model torch_dtype or user kvBytes override."
    });
  }
  if (attentionHeads && !kvHeadsResolved) {
    missing.push({
      field: "gqa",
      reason: "num_key_value_heads is absent; provide a kvHeads or gqa override to derive the exact GQA ratio."
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
    `KV cache assumes ${batch?.value ?? "batch"} concurrent sequences each holding the full ${context?.value ?? "context"}-token context simultaneously (a worst-case capacity bound, defaulted from runtime.max_num_seqs and config.max_position_embeddings). Real vLLM allocates KV cache blocks dynamically from whatever memory is available and typically needs far less in normal traffic. Override context/batch to model your expected concurrency instead of relying on these defaults.`,
    "The hardware capacity card uses supply-side budgeting: available_kv_cache = gpu_vram x gpu_memory_utilization - weights. vLLM's runtime profiler can reserve additional non-KV memory for CUDA graphs, activations, and non-torch allocations, so observed server logs may report a smaller KV cache than this deterministic budget unless those reserves are modeled separately.",
    "\"Overhead\" is a modeled residual computed as (weights + kv_cache) / gpu_memory_utilization - weights - kv_cache. It approximates activation memory, CUDA graphs, and allocator overhead as a fixed proportion of the utilization setting; it is not vLLM's actual memory-profiler output, which depends on max_num_batched_tokens and intermediate size."
  ];
  if (headDimAssumed) {
    notes.push(
      "config.json does not expose head_dim explicitly; assumed head_dim = hidden_size / num_attention_heads. This assumption is wrong for architectures with a non-standard head_dim (e.g. Gemma2/3, Qwen2/2.5); use the headDim override if you know the model's real value."
    );
  }
  const slidingWindow = numberField(config, "sliding_window");
  if (typeof slidingWindow === "number" && context && slidingWindow < context.value) {
    notes.push(
      `config.json reports sliding_window=${slidingWindow}, smaller than the ${context.value}-token context used here. The KV cache formula assumes every layer caches the full context; sliding-window layers actually only need up to ${slidingWindow} tokens, so real KV cache usage is likely lower than shown.`
    );
  }
  if (config.rope_scaling) {
    notes.push(
      "config.json includes rope_scaling. max_position_embeddings may reflect either the base or the RoPE-scaled context length depending on how the checkpoint was published; verify the intended context length and use the context override if needed."
    );
  }

  return calculateEstimate({
    mode: "vllm",
    weightBytes,
    layers,
    kvGroupWidth,
    context,
    batch,
    kvBytes: gt(kvBytesValue, input.overrides?.kvBytes !== undefined ? "override.kvBytes" : `dtype.${resolvedKvDtype}`, input.overrides?.kvBytes !== undefined ? "user" : "metadata"),
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
    displayValues: {
      hiddenSize,
      attentionHeads,
      kvHeads: kvHeadsResolved,
      headDim,
      gqa,
      weightDtype: gt(
        modelDtype,
        input.overrides?.modelDtype !== undefined ? "override.modelDtype" : "config.torch_dtype",
        input.overrides?.modelDtype !== undefined ? "user" : "metadata"
      )
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
  if (!kvCachelessArchitecture && attentionHeads && !kvHeadsResolved) {
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
      "This architecture reports different key_length and value_length (asymmetric K/V head dimensions, e.g. some latent-attention designs); the KV cache formula averages them and may be inaccurate for this model."
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
      `GGUF metadata reports ${prefix}attention.sliding_window=${slidingWindow}, smaller than the ${context.value}-token context used here. Sliding-window layers only need up to ${slidingWindow} cached tokens, so real KV cache usage is likely lower than shown.`
    );
  }

  return calculateEstimate({
    mode: "llamacpp",
    weightBytes,
    layers,
    kvGroupWidth,
    context,
    batch: parallel,
    kvBytes: kvCachelessArchitecture
      ? gt(0, `${arch} architecture has no persistent autoregressive KV cache in llama.cpp`, "metadata")
      : gt(
          combinedKvBytes,
          cacheBytesK === input.overrides?.cacheBytesK || cacheBytesV === input.overrides?.cacheBytesV
            ? "override.cacheBytesK/cacheBytesV"
            : `cache-types.${cacheTypeK}/${cacheTypeV}`,
          input.overrides?.cacheBytesK !== undefined || input.overrides?.cacheBytesV !== undefined ? "user" : "runtime-default"
        ),
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
    kvFormulaLabel: kvCachelessArchitecture
      ? `kv_cache = 0\n\t${arch} has no persistent autoregressive KV cache in llama.cpp`
      : `kv_cache = layers x kv_group_width x context x parallel x (cache_bytes_k + cache_bytes_v) \n\t= ${layers?.value ?? "?"} x ${kvGroupWidth?.value ?? "?"} x ${context?.value ?? "?"} x ${parallel?.value ?? "?"} x (${cacheBytesK ?? "?"} + ${cacheBytesV ?? "?"}) \n\t= ${llamaCppKvCacheBytes(layers, kvGroupWidth, context, parallel, cacheBytesK, cacheBytesV) ?? "?"}`,
    displayValues: {
      hiddenSize,
      attentionHeads,
      kvHeads: kvHeadsResolved,
      headDimK,
      headDimV,
      gqa,
      weightDtype: gt(arch ? ggufFileTypeName(metadata) : undefined, "general.file_type", "metadata")
    }
  });
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
