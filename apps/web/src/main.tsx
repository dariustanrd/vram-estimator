import { StrictMode, useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { createRoot } from "react-dom/client";
import type { EstimateResult, GroundTruthValue, MemoryUnit } from "@vram-estimator/core";
import { formatMemory, memory } from "@vram-estimator/core";
import "./styles.css";

type Mode = "vllm" | "llamacpp";
type Origin = "you" | "default" | "model" | "hardware" | "assumed" | "missing";
type ResultSelection = { kind: "concurrency"; seqs: number } | { kind: "hardware" };
type HfModelSuggestion = {
  id: string;
  value?: string | undefined;
  repoId?: string | undefined;
  file?: string | undefined;
  sizeBytes?: number | undefined;
  downloads?: number | undefined;
  likes?: number | undefined;
  pipelineTag?: string | undefined;
};
type HfModelSearchResponse = { models?: HfModelSuggestion[] | undefined };

/**
 * Per-tab we stick to ONE naming convention so the UI matches what the user reads and types
 * elsewhere: the vLLM tab uses `vllm serve` CLI flags + Hugging Face config.json field names,
 * the llama.cpp tab uses `llama-server` CLI flags + GGUF metadata key names. The wire keys sent
 * to the API stay camelCase (see OVERRIDE_FIELDS[].key); only the labels differ.
 */
const RUNTIME_LABEL: Record<Mode, string> = {
  vllm: "vllm serve",
  llamacpp: "llama-server"
};

// Model-architecture override fields, labeled in each tab's own convention.
const OVERRIDE_FIELDS: Record<Mode, Array<{ key: string; label: string }>> = {
  vllm: [
    { key: "layers", label: "num_hidden_layers" },
    { key: "hiddenSize", label: "hidden_size" },
    { key: "attentionHeads", label: "num_attention_heads" },
    { key: "kvHeads", label: "num_key_value_heads" },
    { key: "headDim", label: "head_dim" },
    { key: "modelDtype", label: "torch_dtype" },
    { key: "kvBytes", label: "kv_cache bytes/elem" },
    { key: "gqa", label: "gqa ratio" },
    { key: "weightBytes", label: "weight bytes" }
  ],
  llamacpp: [
    { key: "layers", label: "block_count" },
    { key: "hiddenSize", label: "embedding_length" },
    { key: "attentionHeads", label: "attention.head_count" },
    { key: "kvHeads", label: "attention.head_count_kv" },
    { key: "headDim", label: "attention.key_length" },
    { key: "cacheBytesK", label: "cache K bytes/elem" },
    { key: "cacheBytesV", label: "cache V bytes/elem" },
    { key: "gqa", label: "gqa ratio" },
    { key: "weightBytes", label: "tensor bytes" }
  ]
};

// resolvedInputs key -> how it is displayed in each tab's convention.
const VARIABLES: Record<Mode, Array<{ key: string; label: string; role: string }>> = {
  vllm: [
    { key: "weightBytes", label: "weight_bytes", role: "Total safetensors size" },
    { key: "layers", label: "num_hidden_layers", role: "Transformer blocks" },
    { key: "kvGroupWidth", label: "kv_group_width", role: "num_key_value_heads x head_dim" },
    { key: "context", label: "max_model_len", role: "--max-model-len (model default if blank)" },
    { key: "batch", label: "max_num_seqs", role: "--max-num-seqs \u00b7 concurrent sequences (vLLM default, worst case)" },
    { key: "kvBytes", label: "kv_bytes", role: "Bytes/elem from --kv-cache-dtype" },
    { key: "utilization", label: "gpu_memory_utilization", role: "--gpu-memory-utilization" },
    { key: "hiddenSize", label: "hidden_size", role: "Reference only" },
    { key: "attentionHeads", label: "num_attention_heads", role: "Query heads" },
    { key: "kvHeads", label: "num_key_value_heads", role: "K/V heads" },
    { key: "headDim", label: "head_dim", role: "Per-head dimension" },
    { key: "gqa", label: "gqa", role: "kv_heads / attention_heads (reference)" }
  ],
  llamacpp: [
    { key: "weightBytes", label: "tensor_bytes", role: "GGUF tensor table total" },
    { key: "layers", label: "block_count", role: "Transformer blocks" },
    { key: "kvGroupWidth", label: "kv_group_width", role: "K/V elements per token per layer; 0 when llama.cpp has no persistent KV cache" },
    { key: "context", label: "ctx_size", role: "--ctx-size (per slot)" },
    { key: "batch", label: "parallel", role: "--parallel (slots)" },
    { key: "kvBytes", label: "cache_bytes", role: "Avg bytes/elem from --cache-type-k/v; 0 when no persistent KV cache" },
    { key: "cacheBytesK", label: "cache_bytes_k", role: "Bytes/elem from --cache-type-k" },
    { key: "cacheBytesV", label: "cache_bytes_v", role: "Bytes/elem from --cache-type-v" },
    { key: "utilization", label: "runtime_utilization", role: "Headroom divisor" },
    { key: "hiddenSize", label: "embedding_length", role: "Reference only" },
    { key: "attentionHeads", label: "attention.head_count", role: "Query heads" },
    { key: "kvHeads", label: "attention.head_count_kv", role: "K/V heads" },
    { key: "headDimK", label: "attention.key_length", role: "Key head dim" },
    { key: "headDimV", label: "attention.value_length", role: "Value head dim" },
    { key: "gqa", label: "gqa", role: "head_count_kv / head_count (reference)" }
  ]
};

function App() {
  const [mode, setMode] = useState<Mode>("vllm");
  const [memoryUnit, setMemoryUnit] = useState<MemoryUnit>("gb");
  const [target, setTarget] = useState("");
  const [hfToken, setHfToken] = useState("");
  const [context, setContext] = useState("");
  const [batch, setBatch] = useState("");
  const [kvDtype, setKvDtype] = useState("auto");
  const [gpuMemoryUtilization, setGpuMemoryUtilization] = useState("");
  const [cacheTypeK, setCacheTypeK] = useState("f16");
  const [cacheTypeV, setCacheTypeV] = useState("f16");
  const [gpuVramGb, setGpuVramGb] = useState("");
  const [numGpus, setNumGpus] = useState("1");
  const [overrides, setOverrides] = useState<Record<string, string>>({});
  // Field keys whose current value was auto-filled from a runtime/model default after an estimate
  // (not typed by the user). These are shown for transparency but are NOT sent back as explicit
  // inputs, so re-running keeps the original provenance ("runtime default"/"model") instead of
  // flipping every value to "you".
  const [defaulted, setDefaulted] = useState<Set<string>>(new Set());
  const [result, setResult] = useState<EstimateResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [modelSuggestions, setModelSuggestions] = useState<HfModelSuggestion[]>([]);
  const [modelSuggestionsOpen, setModelSuggestionsOpen] = useState(false);
  const [modelSuggestionsLoading, setModelSuggestionsLoading] = useState(false);
  const [activeModelSuggestion, setActiveModelSuggestion] = useState(-1);
  const modelSuggestionRefs = useRef<Array<HTMLButtonElement | null>>([]);

  const endpoint = mode === "vllm" ? "/api/estimate/vllm" : "/api/estimate/llamacpp";
  const placeholder =
    mode === "vllm"
      ? "Search Hugging Face <model> or <repo>/<model>"
      : "Search Hugging Face GGUF <model> or <repo>/<model>::<file>";

  useEffect(() => {
    if (!modelSuggestionsOpen || activeModelSuggestion < 0) return;
    modelSuggestionRefs.current[activeModelSuggestion]?.scrollIntoView({ block: "nearest" });
  }, [activeModelSuggestion, modelSuggestionsOpen, modelSuggestions.length]);

  useEffect(() => {
    const query = target.trim();
    const shouldSearch = query.length >= 2 && !/^https?:\/\//i.test(query) && !query.includes("::");
    if (!shouldSearch) {
      setModelSuggestions([]);
      setModelSuggestionsOpen(false);
      setModelSuggestionsLoading(false);
      setActiveModelSuggestion(-1);
      return;
    }

    const controller = new AbortController();
    const timeout = window.setTimeout(async () => {
      setModelSuggestionsLoading(true);
      try {
        const searchEndpoint = mode === "vllm" ? "/api/hf/models" : "/api/hf/gguf";
        const requestInit: RequestInit = { signal: controller.signal };
        if (hfToken) requestInit.headers = { authorization: `Bearer ${hfToken}` };
        const response = await fetch(`${searchEndpoint}?q=${encodeURIComponent(query)}`, requestInit);
        if (!response.ok) throw new Error(`Model search failed: ${response.status}`);
        const data = (await response.json()) as HfModelSearchResponse;
        const models = data.models ?? [];
        setModelSuggestions(models);
        setModelSuggestionsOpen(models.length > 0);
        setActiveModelSuggestion(models.length > 0 ? 0 : -1);
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setModelSuggestions([]);
        setModelSuggestionsOpen(false);
        setActiveModelSuggestion(-1);
      } finally {
        if (!controller.signal.aborted) setModelSuggestionsLoading(false);
      }
    }, 250);

    return () => {
      window.clearTimeout(timeout);
      controller.abort();
    };
  }, [hfToken, mode, target]);

  const parsedOverrides = useMemo(() => {
    const parsed: Record<string, number | string> = {};
    for (const [key, value] of Object.entries(overrides)) {
      if (defaulted.has(key)) continue; // auto-filled default: let the backend resolve it again
      if (!value.trim()) continue;
      const numeric = Number(value);
      parsed[key] = Number.isFinite(numeric) && key !== "modelDtype" ? numeric : value;
    }
    return parsed;
  }, [overrides, defaulted]);

  function unmarkDefault(key: string) {
    setDefaulted((prev) => {
      if (!prev.has(key)) return prev;
      const next = new Set(prev);
      next.delete(key);
      return next;
    });
  }

  function updateOverride(key: string, value: string) {
    setOverrides((prev) => ({ ...prev, [key]: value }));
    unmarkDefault(key);
  }

  function changeMode(next: Mode) {
    if (next === mode) return;
    // Drop values that were auto-filled from defaults so they don't leak across tabs; keep
    // anything the user actually typed.
    setOverrides((prev) => {
      const cleaned = { ...prev };
      for (const key of defaulted) delete cleaned[key];
      return cleaned;
    });
    if (defaulted.has("context")) setContext("");
    if (defaulted.has("batch")) setBatch("");
    if (defaulted.has("gpuMemoryUtilization")) setGpuMemoryUtilization("");
    if (defaulted.has("gpuVramGb")) setGpuVramGb("");
    setDefaulted(new Set());
    setResult(null);
    setError("");
    setMode(next);
  }

  // After an estimate, reflect the values that were actually used back into the optional inputs so
  // the user can see and tweak them. Values that came from a runtime/model default are also tracked
  // in `defaulted` so they are shown but not re-sent as explicit overrides (preserving provenance).
  function syncFieldsFromResult(data: EstimateResult) {
    if (!data.ok) return;
    const ri = data.resolvedInputs;
    const nextDefaulted = new Set(defaulted);
    let nextContext = context;
    let nextBatch = batch;
    let nextGpuMemoryUtilization = gpuMemoryUtilization;
    let nextGpuVramGb = gpuVramGb;
    const nextOverrides = { ...overrides };

    const apply = (
      fieldKey: string,
      resolved: GroundTruthValue<unknown> | undefined,
      setLocal: (value: string) => void
    ) => {
      if (!resolved || resolved.value === undefined || resolved.value === null) return;
      if (resolved.providedBy === "user") {
        nextDefaulted.delete(fieldKey); // the user owns this value; leave it as typed
        return;
      }
      setLocal(String(resolved.value));
      nextDefaulted.add(fieldKey);
    };

    apply("context", ri.context, (value) => {
      nextContext = value;
    });
    apply("batch", ri.batch, (value) => {
      nextBatch = value;
    });
    if (mode === "vllm") {
      apply("gpuMemoryUtilization", ri.utilization, (value) => {
        nextGpuMemoryUtilization = value;
      });
    }
    for (const field of OVERRIDE_FIELDS[mode]) {
      const key = field.key === "headDim" && mode === "llamacpp" ? "headDimK" : field.key;
      apply(field.key, ri[key], (value) => {
        nextOverrides[field.key] = value;
      });
    }
    if (data.hardware?.inferred) {
      nextGpuVramGb = String(data.hardware.commonGpuVramGb ?? data.hardware.gpuVramPerGpu.gb);
      nextDefaulted.add("gpuVramGb");
    } else if (data.hardware) {
      nextDefaulted.delete("gpuVramGb");
    }

    setContext(nextContext);
    setBatch(nextBatch);
    setGpuMemoryUtilization(nextGpuMemoryUtilization);
    setGpuVramGb(nextGpuVramGb);
    setOverrides(nextOverrides);
    setDefaulted(nextDefaulted);
  }

  function chooseModelSuggestion(suggestionValue: string) {
    setTarget(suggestionValue);
    setModelSuggestionsOpen(false);
    setActiveModelSuggestion(-1);
  }

  function handleModelKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (modelSuggestions.length === 0) return;

    if (event.key === "ArrowDown") {
      event.preventDefault();
      setModelSuggestionsOpen(true);
      setActiveModelSuggestion((prev) => (prev + 1) % modelSuggestions.length);
      return;
    }

    if (event.key === "ArrowUp") {
      event.preventDefault();
      setModelSuggestionsOpen(true);
      setActiveModelSuggestion((prev) => (prev <= 0 ? modelSuggestions.length - 1 : prev - 1));
      return;
    }

    if (event.key === "Enter" && modelSuggestionsOpen && activeModelSuggestion >= 0) {
      const suggestion = modelSuggestions[activeModelSuggestion];
      if (!suggestion) return;
      event.preventDefault();
      chooseModelSuggestion(suggestion.value ?? suggestion.id);
      return;
    }

    if (event.key === "Escape") {
      setModelSuggestionsOpen(false);
      setActiveModelSuggestion(-1);
    }
  }

  async function submit() {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const body =
        mode === "vllm"
          ? {
              model: target,
              hfToken: hfToken || undefined,
              context: defaulted.has("context") ? undefined : numberOrUndefined(context),
              batch: defaulted.has("batch") ? undefined : numberOrUndefined(batch),
              kvDtype: kvDtype || undefined,
              gpuMemoryUtilization: defaulted.has("gpuMemoryUtilization") ? undefined : numberOrUndefined(gpuMemoryUtilization),
              gpuVramGb: defaulted.has("gpuVramGb") ? undefined : numberOrUndefined(gpuVramGb),
              numGpus: numberOrUndefined(numGpus),
              overrides: parsedOverrides
            }
          : {
              source: target,
              hfToken: hfToken || undefined,
              context: defaulted.has("context") ? undefined : numberOrUndefined(context),
              parallel: defaulted.has("batch") ? undefined : numberOrUndefined(batch),
              cacheTypeK,
              cacheTypeV,
              gpuVramGb: defaulted.has("gpuVramGb") ? undefined : numberOrUndefined(gpuVramGb),
              numGpus: numberOrUndefined(numGpus),
              overrides: parsedOverrides
            };
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body)
      });
      const data = await readEstimateResponse(response);
      setResult(data);
      syncFieldsFromResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="topbar-brand">
          <img className="topbar-logo" src="/icon-192.png" alt="" aria-hidden="true" />
          <div>
            <h1>Model VRAM Estimator</h1>
            <p>Model + optional metadata in, inference VRAM memory math out.</p>
          </div>
        </div>
        <div className="topbar-actions">
          <div className="segmented" role="tablist" aria-label="Estimator mode">
            <button className={mode === "vllm" ? "active" : ""} onClick={() => changeMode("vllm")}>
              vLLM
            </button>
            <button className={mode === "llamacpp" ? "active" : ""} onClick={() => changeMode("llamacpp")}>
              llama.cpp
            </button>
          </div>
          <div className="segmented unit-toggle" role="group" aria-label="Memory unit">
            <button className={memoryUnit === "gb" ? "active" : ""} onClick={() => setMemoryUnit("gb")}>
              GB
            </button>
            <button className={memoryUnit === "gib" ? "active" : ""} onClick={() => setMemoryUnit("gib")}>
              GiB
            </button>
          </div>
        </div>
      </header>

      <section className="workspace">
        <form
          className="controls"
          onSubmit={(event) => {
            event.preventDefault();
            void submit();
          }}
        >
          <section className="control-card">
            <div className="control-group">
              <span className="group-label">Model</span>
            <div className="model-search">
              <label htmlFor="model-target-input">{mode === "vllm" ? "Hugging Face model" : "GGUF source"}</label>
              <div className="model-combobox">
                <input
                  id="model-target-input"
                  value={target}
                  onChange={(event) => {
                    setTarget(event.target.value);
                    setModelSuggestionsOpen(true);
                  }}
                  onFocus={() => {
                    if (modelSuggestions.length > 0) setModelSuggestionsOpen(true);
                  }}
                  onBlur={() => window.setTimeout(() => setModelSuggestionsOpen(false), 120)}
                  onKeyDown={handleModelKeyDown}
                  placeholder={placeholder}
                  autoComplete="off"
                  role="combobox"
                  aria-autocomplete="list"
                  aria-expanded={modelSuggestionsOpen}
                  aria-controls="hf-model-suggestions"
                  aria-activedescendant={
                    activeModelSuggestion >= 0 ? `hf-model-suggestion-${activeModelSuggestion}` : undefined
                  }
                />
                {(modelSuggestionsOpen || modelSuggestionsLoading) && (
                  <div className="model-suggestions" id="hf-model-suggestions" role="listbox">
                    {modelSuggestionsLoading && (
                      <div className="model-suggestion-status">
                        Searching Hugging Face {mode === "vllm" ? "models" : "GGUF files"}…
                      </div>
                    )}
                    {!modelSuggestionsLoading && modelSuggestions.length === 0 && target.trim().length >= 2 && (
                      <div className="model-suggestion-status">
                        {mode === "vllm"
                          ? "No public models found. You can still paste a model id."
                          : "No GGUF files found. You can still paste a URL or repo::file."}
                      </div>
                    )}
                    {modelSuggestions.map((suggestion, index) => (
                      <button
                        type="button"
                        id={`hf-model-suggestion-${index}`}
                        role="option"
                        aria-selected={index === activeModelSuggestion}
                        className={`model-suggestion${index === activeModelSuggestion ? " active" : ""}`}
                        key={suggestion.id}
                        ref={(element) => {
                          modelSuggestionRefs.current[index] = element;
                        }}
                        onMouseEnter={() => setActiveModelSuggestion(index)}
                        onMouseDown={(event) => {
                          event.preventDefault();
                          chooseModelSuggestion(suggestion.value ?? suggestion.id);
                        }}
                      >
                        <span className="model-suggestion-id">{suggestion.file ?? suggestion.id}</span>
                        <span className="model-suggestion-meta">
                          {suggestion.repoId && <span>{suggestion.repoId}</span>}
                          {suggestion.sizeBytes !== undefined && <span>{formatMemory(memory(suggestion.sizeBytes), memoryUnit)}</span>}
                          {suggestion.pipelineTag && <span>{suggestion.pipelineTag}</span>}
                          {suggestion.downloads !== undefined && <span>{formatCompactNumber(suggestion.downloads)} downloads</span>}
                          {suggestion.likes !== undefined && <span>{formatCompactNumber(suggestion.likes)} likes</span>}
                        </span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
            <label>
              HF token
              <input
                type="password"
                value={hfToken}
                onChange={(event) => setHfToken(event.target.value)}
                placeholder="Optional, used only for this lookup"
              />
            </label>
              <p className="token-note">Tokens are sent for the current lookup only and are not stored by this app.</p>
            </div>
          </section>

          <section className="control-card">
            <div className="control-group">
              <span className="group-label">
                {RUNTIME_LABEL[mode]} flags <em>&mdash; blank uses the runtime default</em>
              </span>
            <div className="grid2">
              <label>
                <span className="field-label">
                  <code className="flag">{mode === "vllm" ? "--max-model-len" : "--ctx-size"}</code>
                  {defaulted.has("context") && <DefaultTag />}
                </span>
                <input
                  value={context}
                  onChange={(event) => {
                    setContext(event.target.value);
                    unmarkDefault("context");
                  }}
                  placeholder="model default"
                />
              </label>
              <label>
                <span className="field-label">
                  <code className="flag">{mode === "vllm" ? "--max-num-seqs" : "--parallel"}</code>
                  {defaulted.has("batch") && <DefaultTag />}
                </span>
                <input
                  value={batch}
                  onChange={(event) => {
                    setBatch(event.target.value);
                    unmarkDefault("batch");
                  }}
                  placeholder="runtime default"
                />
              </label>
            </div>
            {mode === "vllm" ? (
              <div className="grid2">
                <label>
                  <code className="flag">--kv-cache-dtype</code>
                  <select value={kvDtype} onChange={(event) => setKvDtype(event.target.value)}>
                    <option value="auto">auto</option>
                    <option value="fp8">fp8</option>
                    <option value="fp8_e5m2">fp8_e5m2</option>
                    <option value="fp8_e4m3">fp8_e4m3</option>
                  </select>
                </label>
                <label>
                  <span className="field-label">
                    <code className="flag">--gpu-memory-utilization</code>
                    {defaulted.has("gpuMemoryUtilization") && <DefaultTag />}
                  </span>
                  <input
                    type="number"
                    min="0.01"
                    max="1"
                    step="0.01"
                    value={gpuMemoryUtilization}
                    onChange={(event) => {
                      setGpuMemoryUtilization(event.target.value);
                      unmarkDefault("gpuMemoryUtilization");
                    }}
                    placeholder="runtime default"
                  />
                </label>
              </div>
            ) : (
              <div className="grid2">
                <label>
                  <code className="flag">--cache-type-k</code>
                  <select value={cacheTypeK} onChange={(event) => setCacheTypeK(event.target.value)}>
                    <CacheTypeOptions />
                  </select>
                </label>
                <label>
                  <code className="flag">--cache-type-v</code>
                  <select value={cacheTypeV} onChange={(event) => setCacheTypeV(event.target.value)}>
                    <CacheTypeOptions />
                  </select>
                </label>
              </div>
              )}
            </div>

            <OverrideFields mode={mode} overrides={overrides} defaulted={defaulted} updateOverride={updateOverride} />
          </section>

          <section className="control-card">
            <div className="control-group">
              <span className="group-label">
                GPU hardware <em>&mdash; blank VRAM infers the smallest common GPU that fits one full context</em>
              </span>
              <div className="grid2">
                <label>
                  <span className="field-label">
                    VRAM amount per GPU (GB)
                    {defaulted.has("gpuVramGb") && <DefaultTag />}
                  </span>
                  <input
                    type="number"
                    min="0"
                    step="0.1"
                    value={gpuVramGb}
                    onChange={(event) => {
                      setGpuVramGb(event.target.value);
                      unmarkDefault("gpuVramGb");
                    }}
                    placeholder="infer"
                  />
                </label>
                <label>
                  Num GPUs
                  <input
                    type="number"
                    min="1"
                    max="1"
                    step="1"
                    value={numGpus}
                    onChange={(event) => setNumGpus(event.target.value)}
                    placeholder="1"
                  />
                </label>
              </div>
              <p className="token-note">Multi-GPU placement is not modeled yet; use 1 GPU for vLLM tensor_parallel_size=1 estimates.</p>
            </div>
          </section>

          <button className="primary" disabled={loading || !target.trim()}>
            {loading ? "Estimating\u2026" : "Estimate"}
          </button>

          <div className="site-footer site-footer-desktop">
            <BuiltByLink />
          </div>
        </form>

        <section className="results">
          {error && <div className="notice error">{error}</div>}
          {!result && !error && (
            <div className="empty">Enter a model source to calculate weights, KV cache, overhead, and total memory.</div>
          )}
          {result && (
            <ResultView
              result={result}
              memoryUnit={memoryUnit}
              target={target}
              kvDtype={kvDtype}
              cacheTypeK={cacheTypeK}
              cacheTypeV={cacheTypeV}
            />
          )}
        </section>

        <div className="site-footer site-footer-mobile">
          <BuiltByLink />
        </div>
      </section>
    </main>
  );
}

function BuiltByLink() {
  return (
    <>
      <a href="https://github.com/dariustanrd/vram-estimator" target="_blank" rel="noreferrer">
        <GithubIcon />
        <span className="handle">vram-estimator</span>
      </a>
      <span className="footer-label">built by</span>
      <a href="https://x.com/dariustan_" target="_blank" rel="noreferrer">
        <span className="handle">@dariustan_</span>
      </a>
    </>
  );
}

function GithubIcon() {
  // https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/github.svg
  return (
    <svg className="footer-icon" viewBox="0 0 24 24" aria-hidden="true">
      <path
        fill="currentColor"
        d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"
      />
    </svg>
  );
}

function CacheTypeOptions() {
  return (
    <>
      <option value="f16">f16</option>
      <option value="f32">f32</option>
      <option value="bf16">bf16</option>
      <option value="q8_0">q8_0</option>
      <option value="q5_1">q5_1</option>
      <option value="q5_0">q5_0</option>
      <option value="q4_1">q4_1</option>
      <option value="q4_0">q4_0</option>
    </>
  );
}

function OverrideFields({
  mode,
  overrides,
  defaulted,
  updateOverride
}: {
  mode: Mode;
  overrides: Record<string, string>;
  defaulted: Set<string>;
  updateOverride: (key: string, value: string) => void;
}) {
  return (
    <details>
      <summary>Model overrides ({mode === "vllm" ? "config.json fields" : "GGUF metadata keys"})</summary>
      <div className="override-grid">
        {OVERRIDE_FIELDS[mode].map((field) => (
          <label key={field.key}>
            <span className="field-label">
              <code className="flag">{field.label}</code>
              {defaulted.has(field.key) && <DefaultTag />}
            </span>
            <input
              value={overrides[field.key] ?? ""}
              onChange={(event) => updateOverride(field.key, event.target.value)}
            />
          </label>
        ))}
      </div>
    </details>
  );
}

function DefaultTag() {
  return (
    <span className="default-tag" title="Auto-filled from the value that was used (a runtime or model default, not typed by you). Editing it makes it your own explicit value.">
      default
    </span>
  );
}

function ResultView({
  result,
  memoryUnit,
  target,
  kvDtype,
  cacheTypeK,
  cacheTypeV
}: {
  result: EstimateResult;
  memoryUnit: MemoryUnit;
  target: string;
  kvDtype: string;
  cacheTypeK: string;
  cacheTypeV: string;
}) {
  const configuredSeqs = numberValue(result.resolvedInputs.batch) ?? 1;
  const modelDetails = parseModelSourceDetails(result.modelSourceDetails);
  const [selection, setSelection] = useState<ResultSelection>(() => defaultSelection(result, configuredSeqs));

  useEffect(() => {
    setSelection(defaultSelection(result, configuredSeqs));
  }, [configuredSeqs, result]);

  return (
    <div className="stack">
      {!result.ok && (
        <div className="notice error">
          <strong>Missing exact metadata</strong>
          <ul>
            {result.missing.map((item) => (
              <li key={`${item.field}-${item.reason}`}>
                <code>{item.field}</code> {item.reason}
              </li>
            ))}
          </ul>
        </div>
      )}
      {result.memory && (
        <SizesPanel result={result} memoryUnit={memoryUnit} selection={selection} onSelect={setSelection} />
      )}
      {result.ok && (
        <CommandView
          result={result}
          target={target}
          kvDtype={kvDtype}
          cacheTypeK={cacheTypeK}
          cacheTypeV={cacheTypeV}
          selection={selection}
        />
      )}
      {result.notes.length > 0 && <AssumptionsView notes={result.notes} />}
      {result.formula && <CalculationView result={result} memoryUnit={memoryUnit} selection={selection} />}
      {modelDetails && <ModelDetailsView details={modelDetails} memoryUnit={memoryUnit} />}
      <details className="panel">
        <summary>Raw resolved inputs &amp; sources</summary>
        <div className="panel-body">
          <pre>{JSON.stringify(result.resolvedInputs, null, 2)}</pre>
          <pre>{JSON.stringify({ model: result.modelSources, runtime: result.runtimeSources, hardware: result.hardware }, null, 2)}</pre>
        </div>
      </details>
    </div>
  );
}

type FlagRow = { flag: string; value: string; origin: Origin };

type ParsedFile = {
  path: string;
  url?: string | undefined;
  fields: Record<string, unknown>;
};

type ModelSourceDetails = {
  provider?: string | undefined;
  modelId?: string | undefined;
  revision?: string | undefined;
  repoApiUrl?: string | undefined;
  parsedFiles?: ParsedFile[] | undefined;
  weightBytes?: number | undefined;
  weightFiles?: Array<{ path: string; size: number; source: string }> | undefined;
  safetensorsDtype?: { value: string; source: string } | undefined;
  weightDtype?: { value: string; source: string } | undefined;
};

function parseModelSourceDetails(raw: Record<string, unknown> | undefined): ModelSourceDetails | null {
  if (!raw || typeof raw !== "object") return null;
  const details = raw as ModelSourceDetails;
  const hasParsedFiles = Array.isArray(details.parsedFiles) && details.parsedFiles.length > 0;
  const hasWeightFiles = Array.isArray(details.weightFiles) && details.weightFiles.length > 0;
  if (!hasParsedFiles && !hasWeightFiles && details.modelId === undefined) return null;
  return details;
}

function formatFieldValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function ModelDetailsView({ details, memoryUnit }: { details: ModelSourceDetails; memoryUnit: MemoryUnit }) {
  const parsedFiles = details.parsedFiles ?? [];
  const weightFiles = details.weightFiles ?? [];
  const dtype = details.safetensorsDtype ?? details.weightDtype;
  const hfLink =
    details.provider === "huggingface" && details.modelId
      ? `https://huggingface.co/${details.modelId}${details.revision ? `/tree/${details.revision}` : ""}`
      : undefined;

  return (
    <details className="panel">
      <summary>Model metadata &amp; parsed files</summary>
      <div className="panel-body">
        <div className="model-detail-grid">
          {details.modelId && (
            <div className="model-detail">
              <span>Model</span>
              {hfLink ? (
                <a href={hfLink} target="_blank" rel="noreferrer">
                  {details.modelId}
                </a>
              ) : (
                <strong>{details.modelId}</strong>
              )}
            </div>
          )}
          {details.revision && (
            <div className="model-detail">
              <span>Revision</span>
              <code className="inline">{details.revision}</code>
            </div>
          )}
          {dtype && (
            <div className="model-detail">
              <span>Datatype</span>
              <strong>{dtype.value}</strong>
              <em>{dtype.source}</em>
            </div>
          )}
          {details.weightBytes !== undefined && (
            <div className="model-detail">
              <span>Weights</span>
              <strong>{formatMemory(memory(details.weightBytes), memoryUnit)}</strong>
            </div>
          )}
        </div>

        {parsedFiles.map((file) => (
          <div className="parsed-file" key={file.path}>
            <div className="group-label parsed-file-head">
              <span>{file.path}</span>
              {file.url && (
                <a href={file.url} target="_blank" rel="noreferrer">
                  source
                </a>
              )}
            </div>
            <div className="config-table">
              {Object.entries(file.fields).map(([key, value]) => (
                <div className="config-row" key={key}>
                  <code className="config-key">{key}</code>
                  <span className="config-value">{formatFieldValue(value)}</span>
                </div>
              ))}
            </div>
          </div>
        ))}

        {weightFiles.length > 0 && (
          <div className="parsed-file">
            <div className="group-label">Weight files</div>
            <div className="config-table">
              {weightFiles.map((file) => (
                <div className="config-row" key={file.path}>
                  <code className="config-key">{file.path}</code>
                  <span className="config-value">
                    {formatMemory(memory(file.size), memoryUnit)} · {file.source}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </details>
  );
}

function SizesPanel({
  result,
  memoryUnit,
  selection,
  onSelect
}: {
  result: EstimateResult;
  memoryUnit: MemoryUnit;
  selection: ResultSelection;
  onSelect: (selection: ResultSelection) => void;
}) {
  const mem = result.memory!;
  const util = numberValue(result.resolvedInputs.utilization) ?? 1;
  const configuredSeqs = numberValue(result.resolvedInputs.batch) ?? 1;
  const weightsBytes = mem.weights.bytes;
  const kvPerSeqBytes = configuredSeqs > 0 ? mem.kvCache.bytes / configuredSeqs : mem.kvCache.bytes;
  const dtype = result.resolvedInputs.weightDtype?.value;
  const modelType = describeModelType(result);
  const hardware = result.hardware;
  const showHardwareCapacity = Boolean(hardware && result.mode === "vllm" && hardware.kvBytesPerToken > 0);

  const unitWord = result.mode === "vllm" ? "sequence" : "slot";
  const batchProvidedBy = result.resolvedInputs.batch?.providedBy;
  const configuredNote =
    batchProvidedBy === "user" ? "your value" : result.mode === "vllm" ? "vLLM default max_num_seqs" : "llama.cpp default parallel";

  const scenarioAt = (seqs: number) => {
    const kv = kvPerSeqBytes * seqs;
    const total = (weightsBytes + kv) / util;
    const overhead = total - weightsBytes - kv;
    return { kv: memory(kv), overhead: memory(overhead), total: memory(total) };
  };

  const scenarios: Array<{ title: string; sublabel: string; seqs: number }> = [
    {
      title: `Single ${unitWord}`,
      sublabel: result.mode === "vllm" ? "max_num_seqs = 1" : "parallel = 1",
      seqs: 1
    }
  ];
  if (configuredSeqs !== 1) {
    scenarios.push({
      title: batchProvidedBy === "user" ? "Your concurrency" : "Default concurrency",
      sublabel: configuredNote,
      seqs: configuredSeqs
    });
  }

  return (
    <div className="sizes">
      <div className="sizes-fixed">
        <span className="group-label">Model (fixed)</span>
        <Metric label="Weights" value={formatMemory(mem.weights, memoryUnit)} />
        <div className="metric">
          <span>Datatype</span>
          <strong>{dtype ? String(dtype) : "\u2014"}</strong>
        </div>
        <div className="metric model-type" tabIndex={modelType.links.length > 0 ? 0 : undefined}>
          <span>Model type</span>
          <strong>{modelType.label}</strong>
          <em>{modelType.detail}</em>
          {modelType.links.length > 0 && (
            <div className="model-type-links" aria-label="Model type references">
              <span>Learn more</span>
              <ul>
                {modelType.links.map((link) => (
                  <li key={link.url}>
                    <a href={link.url} target="_blank" rel="noreferrer">
                      {link.label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
      <div className="sizes-scenarios">
        <span className="group-label">VRAM required per concurrency</span>
        {showHardwareCapacity && hardware && (
          <button
            type="button"
            className={`scenario-card${selection.kind === "hardware" ? " active" : ""}${hardware.fitsFullContext ? "" : " warning"}`}
            onClick={() => onSelect({ kind: "hardware" })}
            aria-pressed={selection.kind === "hardware"}
          >
            <div className="scenario-head">
              <span className="scenario-count">
                {formatConcurrency(hardware.maxFullContextConcurrency)}
                <small>× {unitWord}</small>
              </span>
              <div className="scenario-title">
                <strong>Hardware capacity</strong>
                <span>
                  {hardware.inferred ? "inferred" : "your"} {formatMemory(hardware.gpuVramPerGpu, memoryUnit)} × {hardware.numGpus} GPU · no --max-num-seqs
                </span>
              </div>
            </div>
            <div className="scenario-stats">
              <div className="scenario-stat">
                <span>KV cache</span>
                <strong>{formatMemory(hardware.allocatedKvCache, memoryUnit)}</strong>
                <em>{formatInteger(hardware.gpuKvCacheTokens)} vLLM tokens / {formatInteger(hardware.gpuKvCacheBlocks)} blocks</em>
              </div>
              <div className="scenario-stat total">
                <span>Total used</span>
                <strong>{formatMemory(hardware.totalUsed, memoryUnit)}</strong>
                <em>{formatMemory(hardware.unusedHeadroom, memoryUnit)} reserved/free</em>
              </div>
            </div>
          </button>
        )}
        {scenarios.map((scenario) => {
          const s = scenarioAt(scenario.seqs);
          return (
            <button
              type="button"
              className={`scenario-card${selection.kind === "concurrency" && selection.seqs === scenario.seqs ? " active" : ""}`}
              key={scenario.title}
              onClick={() => onSelect({ kind: "concurrency", seqs: scenario.seqs })}
              aria-pressed={selection.kind === "concurrency" && selection.seqs === scenario.seqs}
            >
              <div className="scenario-head">
                <span className="scenario-count">
                  {scenario.seqs}
                  <small>× {unitWord}</small>
                </span>
                <div className="scenario-title">
                  <strong>{scenario.title}</strong>
                  <span>{scenario.sublabel}</span>
                </div>
              </div>
              <div className="scenario-stats">
                <div className="scenario-stat">
                  <span>KV cache</span>
                  <strong>{formatMemory(s.kv, memoryUnit)}</strong>
                </div>
                <div className="scenario-stat total">
                  <span>Total</span>
                  <strong>{formatMemory(s.total, memoryUnit)}</strong>
                  <em>incl. {formatMemory(s.overhead, memoryUnit)} overhead</em>
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

type ModelTypeLink = { label: string; url: string };

type ModelTypeDescription = { label: string; detail: string; links: ModelTypeLink[] };

const MODEL_TYPE_LINKS = {
  attnVariants: {label: "Attention Variants", url: "https://magazine.sebastianraschka.com/p/visual-attention-variants"},
  transformer: { label: "Transformer", url: "https://arxiv.org/abs/1706.03762" },
  illustratedTransformer: { label: "Illustrated Transformer", url: "https://jalammar.github.io/illustrated-transformer/" },
  mqaPaper: { label: "MQA paper", url: "https://arxiv.org/abs/1911.02150" },
  mqaBlog: { label: "MQA blog", url: "https://fireworks.ai/blog/multi-query-attention-is-all-you-need" },
  gqaPaper: { label: "GQA paper", url: "https://arxiv.org/abs/2305.13245" },
  gqaExplainer: { label: "GQA explainer", url: "https://sebastianraschka.com/llms-from-scratch/ch04/04_gqa/" },
  mlaPaper: { label: "DeepSeek-V2 MLA", url: "https://arxiv.org/abs/2405.04434" },
  mlaExplainer: { label: "MLA explainer", url: "https://sebastianraschka.com/llms-from-scratch/ch04/05_mla/" },
  bertPaper: { label: "BERT encoder", url: "https://arxiv.org/abs/1810.04805" },
  longformerPaper: { label: "Longformer", url: "https://arxiv.org/abs/2004.05150" },
  swaExplainer: {label: "SWA explainer", url: "https://sebastianraschka.com/llm-architecture-gallery/swa/" },
  gemmaLocalGlobal: { label: "Gemma 2", url: "https://arxiv.org/abs/2408.00118" },
  pagedAttention: { label: "PagedAttention", url: "https://arxiv.org/abs/2309.06180" },
  ggufSpec: { label: "GGUF metadata", url: "https://github.com/ggml-org/ggml/blob/master/docs/gguf.md" }
} satisfies Record<string, ModelTypeLink>;

function describeModelType(result: EstimateResult): ModelTypeDescription {
  const ri = result.resolvedInputs;
  const metadataModelType = stringValue(ri.modelType);
  const attentionHeads = numberValue(ri.attentionHeads);
  const kvHeads = numberValue(ri.kvHeads);
  const gqa = numberValue(ri.gqa);
  const usesDirectKv = numberValue(ri.kvCacheBytes) !== undefined && numberValue(ri.kvBytesPerToken) !== undefined;
  const attentionLayout = describeAttentionLayout(attentionHeads, kvHeads, gqa);
  const layerContexts = parseNumberList(stringValue(ri.attentionLayerContexts));
  const context = numberValue(ri.context);
  const hasSpecialLayerContexts = layerContexts !== undefined && context !== undefined && layerContexts.some((value) => value !== context);

  if (metadataModelType) {
    return {
      label: metadataModelType,
      detail: result.memory?.kvCache.bytes === 0 ? "No persistent autoregressive KV cache" : "From model metadata",
      links: result.memory?.kvCache.bytes === 0
        ? [MODEL_TYPE_LINKS.bertPaper, MODEL_TYPE_LINKS.longformerPaper]
        : []
    };
  }

  if (numberValue(ri.kvGroupWidth) === 0 && result.memory?.kvCache.bytes === 0) {
    return {
      label: "Cacheless · No KV cache",
      detail: "No persistent autoregressive KV cache",
      links: [MODEL_TYPE_LINKS.bertPaper, MODEL_TYPE_LINKS.longformerPaper]
    };
  }

  if (usesDirectKv) {
    if (numberValue(ri.kvLoraRank) !== undefined) {
      return {
        label: "Decoder · MLA",
        detail: "Direct cache formula from kv_lora_rank + qk_rope_head_dim",
        links: [MODEL_TYPE_LINKS.attnVariants, MODEL_TYPE_LINKS.mlaPaper, MODEL_TYPE_LINKS.mlaExplainer]
      };
    }
    if (stringValue(ri.kvHeadsPerLayer)) {
      return {
        label: "Decoder · Defined per-layer KV cache",
        detail: "GGUF reports per-layer KV-head metadata",
        links: [MODEL_TYPE_LINKS.ggufSpec, MODEL_TYPE_LINKS.pagedAttention]
      };
    }
    if (hasSpecialLayerContexts) {
      return attentionLayout
        ? {
            label: `Decoder · ${attentionLayout.label} + Non-uniform KV`,
            detail: `${attentionLayout.detail}; Per-layer KV cache size is non-uniform, e.g. from sliding-window/local attention, local/global patterns, or zero-KV layers.`,
            links: uniqueModelTypeLinks([
              ...attentionLayout.links,
              MODEL_TYPE_LINKS.swaExplainer,
              MODEL_TYPE_LINKS.gemmaLocalGlobal
            ])
          }
        : {
            label: "Decoder · Non-uniform KV",
            detail: "Per-layer KV cache size is non-uniform, e.g. from sliding-window/local attention, local/global patterns, or zero-KV layers.",
            links: [MODEL_TYPE_LINKS.attnVariants, MODEL_TYPE_LINKS.swaExplainer, MODEL_TYPE_LINKS.gemmaLocalGlobal]
          };
    }
    return {
      label: "Decoder · Direct KV formula",
      detail: "Cache cannot be represented by one uniform scalar formula",
      links: [MODEL_TYPE_LINKS.pagedAttention, MODEL_TYPE_LINKS.ggufSpec]
    };
  }

  if (attentionLayout) {
    return {
      label: `Decoder · ${attentionLayout.label}`,
      detail: attentionLayout.detail,
      links: attentionLayout.links
    };
  }

  return {
    label: result.mode === "vllm" ? "Decoder · HF" : "Decoder · GGUF",
    detail: "Model type inferred from available metadata",
    links: result.mode === "vllm" ? [MODEL_TYPE_LINKS.pagedAttention] : [MODEL_TYPE_LINKS.ggufSpec]
  };
}

function describeAttentionLayout(
  attentionHeads: number | undefined,
  kvHeads: number | undefined,
  gqa: number | undefined
): { label: string; detail: string; links: ModelTypeLink[] } | undefined {
  if (attentionHeads === undefined || kvHeads === undefined) return undefined;
  if (kvHeads === attentionHeads) {
    return {
      label: "MHA",
      detail: `${attentionHeads} query heads, ${kvHeads} KV heads`,
      links: [MODEL_TYPE_LINKS.transformer, MODEL_TYPE_LINKS.illustratedTransformer]
    };
  }
  if (kvHeads === 1 && attentionHeads > 1) {
    return {
      label: "MQA",
      detail: `${attentionHeads} query heads share 1 KV head`,
      links: [MODEL_TYPE_LINKS.attnVariants, MODEL_TYPE_LINKS.mqaPaper, MODEL_TYPE_LINKS.mqaBlog]
    };
  }
  return {
    label: "GQA",
    detail: `${attentionHeads} query heads, ${kvHeads} KV heads${gqa !== undefined ? ` (${formatNumber(gqa)} ratio)` : ""}`,
    links: [MODEL_TYPE_LINKS.attnVariants, MODEL_TYPE_LINKS.gqaPaper, MODEL_TYPE_LINKS.gqaExplainer]
  };
}

function uniqueModelTypeLinks(links: ModelTypeLink[]): ModelTypeLink[] {
  const seen = new Set<string>();
  return links.filter((link) => {
    if (seen.has(link.url)) return false;
    seen.add(link.url);
    return true;
  });
}

function CommandView({
  result,
  target,
  kvDtype,
  cacheTypeK,
  cacheTypeV,
  selection
}: {
  result: EstimateResult;
  target: string;
  kvDtype: string;
  cacheTypeK: string;
  cacheTypeV: string;
  selection: ResultSelection;
}) {
  const [copied, setCopied] = useState(false);
  const ri = result.resolvedInputs;
  const bin = RUNTIME_LABEL[result.mode];

  const modelArg =
    result.mode === "vllm"
      ? target || "<model>"
      : target.includes("::")
        ? `-hf ${target.replace("::", ":")}`
        : `-m ${target || "<model.gguf>"}`;

  const selectedSeqs = selection.kind === "concurrency" ? selection.seqs : numberValue(ri.batch) ?? 1;
  const selectedOrigin: Origin = selectedSeqs === numberValue(ri.batch) ? originOf(ri.batch) : "you";
  const flags: FlagRow[] =
    result.mode === "vllm"
      ? selection.kind === "hardware"
        ? [
            { flag: "--max-model-len", value: valueStr(ri.context), origin: originOf(ri.context) },
            { flag: "--kv-cache-dtype", value: kvDtype, origin: kvDtype === "auto" ? "default" : "you" },
            { flag: "--gpu-memory-utilization", value: valueStr(ri.utilization), origin: originOf(ri.utilization) },
            { flag: "--tensor-parallel-size", value: "1", origin: "default" }
          ]
        : [
            { flag: "--max-model-len", value: valueStr(ri.context), origin: originOf(ri.context) },
            { flag: "--max-num-seqs", value: String(selectedSeqs), origin: selectedOrigin },
            { flag: "--kv-cache-dtype", value: kvDtype, origin: kvDtype === "auto" ? "default" : "you" },
            { flag: "--gpu-memory-utilization", value: valueStr(ri.utilization), origin: originOf(ri.utilization) },
            { flag: "--tensor-parallel-size", value: "1", origin: "default" }
          ]
      : [
          { flag: "--ctx-size", value: valueStr(ri.context), origin: originOf(ri.context) },
          { flag: "--parallel", value: String(selectedSeqs), origin: selectedOrigin },
          { flag: "--cache-type-k", value: cacheTypeK, origin: cacheTypeK === "f16" ? "default" : "you" },
          { flag: "--cache-type-v", value: cacheTypeV, origin: cacheTypeV === "f16" ? "default" : "you" },
          { flag: "--n-gpu-layers", value: "999", origin: "default" }
        ];

  const command = `${bin} ${modelArg} ${flags.map((f) => `${f.flag} ${f.value}`).join(" ")}`;

  async function copy() {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Equivalent command</h2>
        <button type="button" className="ghost" onClick={() => void copy()}>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <div className="panel-body">
        <p className="hint">
          {selection.kind === "hardware"
            ? `Hardware capacity is not configured with --max-num-seqs. Start vLLM with the memory/model flags below; vLLM profiles memory, allocates KV cache blocks, and reports the maximum concurrency in the startup logs.`
            : `These are the ${bin} flags for the selected ${result.mode === "vllm" ? "sequence" : "slot"} concurrency.`}
        </p>
        <pre className="command">{formatCommand(bin, modelArg, flags)}</pre>
        <div className="flag-table">
          {flags.map((row) => (
            <div className="flag-row" key={row.flag}>
              <code className="flag">{row.flag}</code>
              <span className="flag-value">{row.value}</span>
              <Badge origin={row.origin} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function AssumptionsView({ notes }: { notes: string[] }) {
  return (
    <details className="notice assumptions">
      <summary>
        Assumptions &amp; caveats <span className="count">{notes.length}</span>
      </summary>
      <ul>
        {notes.map((note) => (
          <li key={note}>{note}</li>
        ))}
      </ul>
    </details>
  );
}

function CalculationView({ result, memoryUnit, selection }: { result: EstimateResult; memoryUnit: MemoryUnit; selection: ResultSelection }) {
  if (selection.kind === "hardware" && result.hardware) {
    return <HardwareCalculationView result={result} memoryUnit={memoryUnit} />;
  }

  const selectedSeqs = selection.kind === "concurrency" ? selection.seqs : selectedCommandSeqs(result, selection);
  const variables = VARIABLES[result.mode];
  const mem = result.memory!;
  const configuredSeqs = numberValue(result.resolvedInputs.batch) ?? 1;
  const util = numberValue(result.resolvedInputs.utilization) ?? 1;
  const weightsBytes = mem.weights.bytes;
  const kvPerSeqBytes = configuredSeqs > 0 ? mem.kvCache.bytes / configuredSeqs : mem.kvCache.bytes;
  const kvBytes = kvPerSeqBytes * selectedSeqs;
  const totalBytes = (weightsBytes + kvBytes) / util;
  const overheadBytes = totalBytes - weightsBytes - kvBytes;
  const selectedMemory = {
    weights: memory(weightsBytes),
    kvCache: memory(kvBytes),
    total: memory(totalBytes),
    overhead: memory(overheadBytes)
  };
  const batchLabel = result.mode === "vllm" ? "batch" : "parallel";
  const configuredBatchLabel = `configured_${batchLabel}`;
  const selectedBatchLabel = `selected_${batchLabel}`;
  const contextValue = numberValue(result.resolvedInputs.context) ?? 0;
  const directKvCacheBytes = numberValue(result.resolvedInputs.kvCacheBytes);
  const directKvBytesPerToken = numberValue(result.resolvedInputs.kvBytesPerToken);
  const usesDirectKvFormula = directKvBytesPerToken !== undefined && directKvCacheBytes !== undefined;
  const averageCacheBytesPerToken = usesDirectKvFormula
    ? directKvBytesPerToken
    : contextValue > 0 && configuredSeqs > 0
      ? mem.kvCache.bytes / (contextValue * configuredSeqs)
      : 0;
  const standardAverageFormula = result.mode === "llamacpp" && numberValue(result.resolvedInputs.cacheBytesK) !== undefined && numberValue(result.resolvedInputs.cacheBytesV) !== undefined
    ? `average_cache_bytes_per_token = layers x kv_group_width x (cache_bytes_k + cache_bytes_v) \n\t= ${valueStr(result.resolvedInputs.layers)} x ${valueStr(result.resolvedInputs.kvGroupWidth)} x (${valueStr(result.resolvedInputs.cacheBytesK)} + ${valueStr(result.resolvedInputs.cacheBytesV)}) \n\t= ${averageCacheBytesPerToken}`
    : `average_cache_bytes_per_token = 2 x layers x kv_group_width x kv_bytes_per_element \n\t= 2 x ${valueStr(result.resolvedInputs.layers)} x ${valueStr(result.resolvedInputs.kvGroupWidth)} x ${valueStr(result.resolvedInputs.kvBytes)} \n\t= ${averageCacheBytesPerToken}`;
  const averageCacheBytesPerTokenFormula = usesDirectKvFormula
    ? `average_cache_bytes_per_token = direct_kv_cache_bytes / (context x ${configuredBatchLabel}) \n\t= ${directKvCacheBytes} / (${contextValue} x ${configuredSeqs}) \n\t= ${directKvBytesPerToken}`
    : standardAverageFormula;
  const isCachelessLlamaCpp = result.mode === "llamacpp" && kvBytes === 0 && numberValue(result.resolvedInputs.kvGroupWidth) === 0;
  const kvFormula = isCachelessLlamaCpp
    ? result.formula!.kvCache
    : `kv_cache = average_cache_bytes_per_token x context x ${selectedBatchLabel} \n\t= ${averageCacheBytesPerToken} x ${valueStr(result.resolvedInputs.context)} x ${selectedSeqs} \n\t= ${kvBytes}`;
  const totalFormula = `total = (weights + kv_cache) / utilization \n\t= (${weightsBytes} + ${kvBytes}) / ${util} \n\t= ${totalBytes}`;
  const overheadCaveat = isCachelessLlamaCpp
    ? "Note: this 0 is only the estimator's residual after weights + persistent KV cache. llama.cpp still needs runtime memory for temporary activations, graph buffers, backend workspaces, allocator padding, tokenizer/model structures, and possibly mmap/accounting effects; this GGUF-only calculation cannot determine that overhead."
    : "";
  const overheadFormula = `modeled_overhead = total - weights - kv_cache \n\t= ${totalBytes} - ${weightsBytes} - ${kvBytes} \n\t= ${overheadBytes}${overheadCaveat ? `\n\n${overheadCaveat}` : ""}`;
  const selectedBatch = {
    ...result.resolvedInputs.batch,
    value: selectedSeqs,
    providedBy: selectedSeqs === configuredSeqs ? result.resolvedInputs.batch?.providedBy : "user",
    source: selectedSeqs === configuredSeqs ? result.resolvedInputs.batch?.source : "selected concurrency"
  } as GroundTruthValue<unknown>;
  return (
    <div className="panel">
      <h2>How this was calculated</h2>
      <div className="panel-body">
        <div className="formula-list">
          <FormulaLine label="Weights" formula={result.formula!.weights} value={formatMemory(selectedMemory.weights, memoryUnit)} />
          {usesDirectKvFormula ? (
            <FormulaLine
              label="Direct KV cache bytes"
              formula={`${result.formula!.kvCache}\n\nThis is the exact cache bytes for the configured context x ${batchLabel}. The selected concurrency calculation normalizes it to average bytes/token below.`}
              value={formatMemory(memory(directKvCacheBytes), memoryUnit)}
            />
          ) : null}
          {!isCachelessLlamaCpp ? (
            <FormulaLine
              label="Average cache bytes/token"
              formula={averageCacheBytesPerTokenFormula}
              value={`${formatNumber(averageCacheBytesPerToken)} bytes/token`}
            />
          ) : null}
          <FormulaLine label="KV cache" formula={kvFormula} value={formatMemory(selectedMemory.kvCache, memoryUnit)} />
          <FormulaLine label="Total" formula={totalFormula} value={formatMemory(selectedMemory.total, memoryUnit)} />
          <FormulaLine label="Overhead" formula={overheadFormula} value={formatMemory(selectedMemory.overhead, memoryUnit)} />
        </div>
        <div className="var-table">
          <div className="var-head">
            <span>Input</span>
            <span>Value</span>
          </div>
          {variables.map((variable) => (
            <VariableRow key={variable.key} label={variable.label} role={variable.role} value={variable.key === "batch" ? selectedBatch : result.resolvedInputs[variable.key]} />
          ))}
        </div>
      </div>
    </div>
  );
}

function HardwareCalculationView({ result, memoryUnit }: { result: EstimateResult; memoryUnit: MemoryUnit }) {
  const hardware = result.hardware!;
  const ri = result.resolvedInputs;
  const util = numberValue(ri.utilization) ?? 1;
  const weightsBytes = result.memory!.weights.bytes;
  const layers = numberValue(ri.layers) ?? 0;
  const kvGroupWidth = numberValue(ri.kvGroupWidth) ?? 0;
  const kvBytes = numberValue(ri.kvBytes) ?? 0;
  const context = numberValue(ri.context) ?? 0;
  const batch = numberValue(ri.batch) ?? 1;
  const batchLabel = result.mode === "vllm" ? "batch" : "parallel";
  const directKvCacheBytes = numberValue(ri.kvCacheBytes);
  const directKvBytesPerToken = numberValue(ri.kvBytesPerToken);
  const usesDirectKvFormula = directKvCacheBytes !== undefined && directKvBytesPerToken !== undefined;
  const perTokenPerLayerBytes = 2 * kvGroupWidth * kvBytes;
  const requestedBytes = hardware.gpuMemoryBudget.bytes;
  const nonKvBytes = weightsBytes;
  const kvBytesPerTokenFormula = usesDirectKvFormula
    ? `bytes_per_token_all_layers = average_cache_bytes_per_token \n\t= kv_cache_bytes / (context x ${batchLabel}) \n\t= ${directKvCacheBytes} / (${context} x ${batch}) \n\t= ${hardware.kvBytesPerToken}`
    : `per_token_per_layer = 2 x kv_group_width x kv_bytes_per_element \n\t= 2 x ${kvGroupWidth} x ${kvBytes} \n\t= ${perTokenPerLayerBytes}\nbytes_per_token_all_layers = per_token_per_layer x layers \n\t= ${perTokenPerLayerBytes} x ${layers} \n\t= ${hardware.kvBytesPerToken}`;
  const kvBlocksFormula = hardware.kvBlockBytes === 0
    ? `kv_block_bytes = bytes_per_token_all_layers x block_size \n\t= ${hardware.kvBytesPerToken} x ${hardware.blockSize} \n\t= 0\ngpu_blocks = 0 because this model/runtime has no persistent KV cache blocks`
    : `kv_block_bytes = bytes_per_token_all_layers x block_size \n\t= ${hardware.kvBytesPerToken} x ${hardware.blockSize} \n\t= ${hardware.kvBlockBytes}\ngpu_blocks = floor(available_kv_cache_memory / kv_block_bytes) \n\t= floor(${hardware.availableKvCache.bytes} / ${hardware.kvBlockBytes}) \n\t= ${hardware.gpuKvCacheBlocks}`;

  return (
    <div className="panel">
      <h2>How hardware capacity was calculated</h2>
      <div className="panel-body">
        <p className="hint">
          Simplified vLLM-style supply-side calculation. vLLM first budgets GPU memory with{" "}
          <code className="inline-code">gpu_memory_utilization</code>, profiles non-KV memory, then allocates KV blocks from the remainder.
          This estimator uses model weights as the known non-KV component; runtime profiling overhead can reduce the real log value.
        </p>
        <div className="formula-list">
          <FormulaLine
            label="Requested GPU budget"
            formula={`requested_memory = total_gpu_vram x gpu_memory_utilization \n\t= ${hardware.totalGpuVram.bytes} x ${util} \n\t= ${requestedBytes}`}
            value={formatMemory(hardware.gpuMemoryBudget, memoryUnit)}
          />
          <FormulaLine
            label="Available KV memory"
            formula={`available_kv_cache_memory = requested_memory - non_kv_cache_memory \n\t≈ ${requestedBytes} - ${nonKvBytes} \n\t= ${hardware.availableKvCache.bytes}`}
            value={formatMemory(hardware.availableKvCache, memoryUnit)}
          />
          {usesDirectKvFormula ? (
            <FormulaLine
              label="Direct KV cache bytes"
              formula={`${result.formula!.kvCache}\n\nThis is the exact cache bytes for the configured context x ${batchLabel}. The hardware capacity calculation normalizes it to bytes/token below; ${batchLabel} is not used directly when computing available GPU KV blocks.`}
              value={formatMemory(result.memory!.kvCache, memoryUnit)}
            />
          ) : null}
          <FormulaLine
            label="KV bytes per token"
            formula={kvBytesPerTokenFormula}
            value={`${formatInteger(hardware.kvBytesPerToken)} bytes/token`}
          />
          <FormulaLine
            label="KV blocks"
            formula={kvBlocksFormula}
            value={`${formatInteger(hardware.gpuKvCacheBlocks)} blocks`}
          />
          <FormulaLine
            label="Full-context concurrency"
            formula={`blocks_per_full_context = ceil(max_model_len / block_size) \n\t= ceil(${context} / ${hardware.blockSize}) \n\t= ${hardware.blocksPerFullContext}\nmax_concurrency = gpu_blocks / blocks_per_full_context \n\t= ${hardware.gpuKvCacheBlocks} / ${hardware.blocksPerFullContext} \n\t= ${hardware.maxFullContextConcurrency} = ${formatConcurrency(hardware.maxFullContextConcurrency)}x\ngpu_kv_cache_tokens = int(max_concurrency x max_model_len) \n\t= int(${hardware.maxFullContextConcurrency} x ${context}) \n\t= ${hardware.gpuKvCacheTokens}`}
            value={`${formatInteger(hardware.gpuKvCacheTokens)} tokens`}
          />
          <FormulaLine
            label="Total used"
            formula={`Note: --max-num-seqs is not part of this capacity calculation. vLLM computes GPU KV blocks from available memory, then reports max concurrency as gpu_blocks / blocks_per_full_context and GPU KV cache size as int(max_concurrency x max_model_len).\n\nallocated_kv_cache = gpu_blocks x kv_block_bytes \n\t= ${hardware.gpuKvCacheBlocks} x ${hardware.kvBlockBytes} \n\t= ${hardware.allocatedKvCache.bytes}\ntotal_used = weights + allocated_kv_cache \n\t= ${weightsBytes} + ${hardware.allocatedKvCache.bytes} \n\t= ${hardware.totalUsed.bytes}`}
            value={formatMemory(hardware.totalUsed, memoryUnit)}
          />
        </div>
      </div>
    </div>
  );
}

function FormulaLine({ label, formula, value }: { label: string; formula: string; value?: string | undefined }) {
  return (
    <div className="formula-line">
      <span>{label}</span>
      <code>
        {formula}
        {value ? ` = ${value}` : ""}
      </code>
    </div>
  );
}

function VariableRow({ label, role, value }: { label: string; role: string; value: GroundTruthValue<unknown> | undefined }) {
  return (
    <div className="var-row">
      <div className="var-name" title={value ? `${role} \u2014 source: ${value.source}` : role}>
        <code className="flag">{label}</code>
        <span className="var-role">{role}</span>
      </div>
      <div className="var-val">
        <span>{value ? String(value.value) : "\u2014"}</span>
        <Badge origin={originOf(value)} />
      </div>
    </div>
  );
}

function Badge({ origin }: { origin: Origin }) {
  const meta: Record<Origin, { label: string; title: string }> = {
    you: { label: "you", title: "You set this value explicitly." },
    default: {
      label: "runtime default",
      title:
        "Default baked into the selected runtime version (vLLM / llama.cpp) and captured in this app's runtime-defaults snapshot. It does NOT come from the model."
    },
    model: { label: "model", title: "Read from the model's Hugging Face config.json / GGUF metadata." },
    hardware: { label: "hardware", title: "Derived from the selected GPU hardware capacity estimate." },
    assumed: {
      label: "assumed",
      title: "Derived from a simplifying assumption (e.g. head_dim = hidden_size / heads); may be wrong for some architectures."
    },
    missing: { label: "missing", title: "No value available for this input." }
  };
  const { label, title } = meta[origin];
  return (
    <span className={`badge badge-${origin}`} title={title}>
      {label}
    </span>
  );
}

function Metric({ label, value, primary }: { label: string; value: string; primary?: boolean }) {
  return (
    <div className={primary ? "metric primary" : "metric"}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function originOf(value: GroundTruthValue<unknown> | undefined): Origin {
  if (!value) return "missing";
  if (value.assumed) return "assumed";
  if (value.providedBy === "user") return "you";
  if (value.providedBy === "metadata") return "model";
  return "default";
}

function valueStr(value: GroundTruthValue<unknown> | undefined): string {
  return value ? String(value.value) : "?";
}

function numberValue(value: GroundTruthValue<unknown> | undefined): number | undefined {
  return value && typeof value.value === "number" ? value.value : undefined;
}

function stringValue(value: GroundTruthValue<unknown> | undefined): string | undefined {
  return value && typeof value.value === "string" ? value.value : undefined;
}

function parseNumberList(value: string | undefined): number[] | undefined {
  if (!value) return undefined;
  const numbers = value.split(",").map((part) => Number(part.trim()));
  return numbers.every((number) => Number.isFinite(number)) ? numbers : undefined;
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return "0";
  return Number.isInteger(value) ? String(value) : value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function formatConcurrency(value: number): string {
  if (!Number.isFinite(value)) return "0";
  return value >= 10 ? value.toFixed(1) : value.toFixed(2);
}

function formatInteger(value: number): string {
  if (!Number.isFinite(value)) return "0";
  return Math.floor(value).toLocaleString();
}

function formatCompactNumber(value: number): string {
  if (!Number.isFinite(value)) return "0";
  return Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

function defaultSelection(result: EstimateResult, configuredSeqs: number): ResultSelection {
  if (result.hardware && result.mode === "vllm" && result.hardware.kvBytesPerToken > 0) return { kind: "hardware" };
  return { kind: "concurrency", seqs: configuredSeqs };
}

function selectedCommandSeqs(result: EstimateResult, selection: ResultSelection): number {
  if (selection.kind === "concurrency") return selection.seqs;
  const maxWholeConcurrency = Math.floor(result.hardware?.maxFullContextConcurrency ?? 0);
  return Math.max(1, maxWholeConcurrency);
}

function formatCommand(bin: string, modelArg: string, flags: FlagRow[]): string {
  const lines = [`${bin} ${modelArg} \\`];
  flags.forEach((flag, index) => {
    const suffix = index === flags.length - 1 ? "" : " \\";
    lines.push(`  ${flag.flag} ${flag.value}${suffix}`);
  });
  return lines.join("\n");
}

async function readEstimateResponse(response: Response): Promise<EstimateResult> {
  const text = await response.text();
  const contentType = response.headers.get("content-type") ?? "";

  if (!text.trim()) {
    throw new Error(
      response.ok
        ? "API returned an empty response."
        : `API request failed with HTTP ${response.status}. In local development, make sure the API is running with npm run dev:api or use npm run dev to start both servers.`
    );
  }

  if (!contentType.includes("application/json")) {
    throw new Error(`API returned non-JSON response (${response.status}): ${text.slice(0, 200)}`);
  }

  const data = JSON.parse(text) as EstimateResult | { error?: string };
  if (!response.ok && "error" in data && data.error) {
    throw new Error(data.error);
  }
  return data as EstimateResult;
}

function numberOrUndefined(value: string): number | undefined {
  if (!value.trim()) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
