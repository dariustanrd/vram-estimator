export type RuntimeName = "vllm" | "llama.cpp";

export type SourceRef = {
  repo: string;
  version: string;
  file: string;
  symbol: string;
  url: string;
};

export type RuntimeDefault<T = unknown> = {
  value: T;
  source: SourceRef;
};

export type RuntimeSnapshot = {
  runtime: string;
  version: string;
  repo: string;
  defaults: Record<string, RuntimeDefault>;
};

export type MemoryUnit = "gb" | "gib";

export type MemoryAmount = {
  bytes: number;
  gb: number;
  gib: number;
};

export type GroundTruthValue<T> = {
  value: T;
  source: string;
  providedBy: "metadata" | "user" | "runtime-default";
};

export type MissingValue = {
  field: string;
  reason: string;
};

export type FormulaBreakdown = {
  weights: string;
  kvCache: string;
  total: string;
  overhead: string;
};

export type EstimateResult = {
  mode: "vllm" | "llamacpp";
  ok: boolean;
  missing: MissingValue[];
  resolvedInputs: Record<string, GroundTruthValue<unknown>>;
  userOverrides: Record<string, unknown>;
  memory: {
    weights: MemoryAmount;
    kvCache: MemoryAmount;
    overhead: MemoryAmount;
    total: MemoryAmount;
  } | null;
  formula: FormulaBreakdown | null;
  modelSources: string[];
  runtimeSources: RuntimeDefault[];
  notes: string[];
};

export type Fetcher = typeof fetch;

export type HfAuth = {
  token?: string | undefined;
};

export type VllmEstimateInput = {
  model: string;
  hfToken?: string | undefined;
  context?: number | undefined;
  batch?: number | undefined;
  kvDtype?: string | undefined;
  runtimeVersion?: string | undefined;
  overrides?: Partial<{
    weightBytes: number;
    layers: number;
    hiddenSize: number;
    context: number;
    batch: number;
    kvBytes: number;
    gqa: number;
    modelDtype: string;
    attentionHeads: number;
    kvHeads: number;
  }> | undefined;
};

export type LlamaCppEstimateInput = {
  source: string;
  hfToken?: string | undefined;
  context?: number | undefined;
  parallel?: number | undefined;
  cacheTypeK?: string | undefined;
  cacheTypeV?: string | undefined;
  runtimeVersion?: string | undefined;
  overrides?: Partial<{
    weightBytes: number;
    layers: number;
    hiddenSize: number;
    context: number;
    parallel: number;
    cacheBytesK: number;
    cacheBytesV: number;
    gqa: number;
    attentionHeads: number;
    kvHeads: number;
  }> | undefined;
};
