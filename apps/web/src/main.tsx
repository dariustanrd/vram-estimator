import { StrictMode, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import type { EstimateResult, GroundTruthValue, MemoryUnit } from "@vram-estimator/core";
import { formatMemory, memory } from "@vram-estimator/core";
import "./styles.css";

type Mode = "vllm" | "llamacpp";
type Origin = "you" | "default" | "model" | "assumed" | "missing";

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
    { key: "kvGroupWidth", label: "kv_group_width", role: "head_count_kv x head_dim" },
    { key: "context", label: "ctx_size", role: "--ctx-size (per slot)" },
    { key: "batch", label: "parallel", role: "--parallel (slots)" },
    { key: "kvBytes", label: "cache_bytes", role: "Avg bytes/elem from --cache-type-k/v" },
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
  const [cacheTypeK, setCacheTypeK] = useState("f16");
  const [cacheTypeV, setCacheTypeV] = useState("f16");
  const [overrides, setOverrides] = useState<Record<string, string>>({});
  // Field keys whose current value was auto-filled from a runtime/model default after an estimate
  // (not typed by the user). These are shown for transparency but are NOT sent back as explicit
  // inputs, so re-running keeps the original provenance ("runtime default"/"model") instead of
  // flipping every value to "you".
  const [defaulted, setDefaulted] = useState<Set<string>>(new Set());
  const [result, setResult] = useState<EstimateResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const endpoint = mode === "vllm" ? "/api/estimate/vllm" : "/api/estimate/llamacpp";
  const placeholder =
    mode === "vllm"
      ? "meta-llama/Llama-3.1-8B-Instruct"
      : "https://huggingface.co/user/repo/resolve/main/model.gguf or repo::model.gguf";

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
    for (const field of OVERRIDE_FIELDS[mode]) {
      const key = field.key === "headDim" && mode === "llamacpp" ? "headDimK" : field.key;
      apply(field.key, ri[key], (value) => {
        nextOverrides[field.key] = value;
      });
    }

    setContext(nextContext);
    setBatch(nextBatch);
    setOverrides(nextOverrides);
    setDefaulted(nextDefaulted);
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
              overrides: parsedOverrides
            }
          : {
              source: target,
              hfToken: hfToken || undefined,
              context: defaulted.has("context") ? undefined : numberOrUndefined(context),
              parallel: defaulted.has("batch") ? undefined : numberOrUndefined(batch),
              cacheTypeK,
              cacheTypeV,
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
        <div>
          <h1>VRAM Estimator</h1>
          <p>Exact metadata in, transparent inference memory math out.</p>
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
          className="panel controls"
          onSubmit={(event) => {
            event.preventDefault();
            void submit();
          }}
        >
          <div className="control-group">
            <span className="group-label">Model</span>
            <label>
              {mode === "vllm" ? "Hugging Face model" : "GGUF source"}
              <input value={target} onChange={(event) => setTarget(event.target.value)} placeholder={placeholder} />
            </label>
            <label>
              HF token
              <input
                type="password"
                value={hfToken}
                onChange={(event) => setHfToken(event.target.value)}
                placeholder="Optional, used only for this lookup"
              />
            </label>
          </div>

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
              <label>
                <code className="flag">--kv-cache-dtype</code>
                <select value={kvDtype} onChange={(event) => setKvDtype(event.target.value)}>
                  <option value="auto">auto</option>
                  <option value="fp8">fp8</option>
                  <option value="fp8_e5m2">fp8_e5m2</option>
                  <option value="fp8_e4m3">fp8_e4m3</option>
                </select>
              </label>
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
          <button className="primary" disabled={loading || !target.trim()}>
            {loading ? "Estimating\u2026" : "Estimate"}
          </button>
          <p className="token-note">Tokens are sent for the current lookup only and are not stored by this app.</p>
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
      </section>
    </main>
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
      {result.memory && <SizesPanel result={result} memoryUnit={memoryUnit} />}
      {result.ok && (
        <CommandView
          result={result}
          target={target}
          kvDtype={kvDtype}
          cacheTypeK={cacheTypeK}
          cacheTypeV={cacheTypeV}
        />
      )}
      {result.notes.length > 0 && <AssumptionsView notes={result.notes} />}
      {result.formula && <CalculationView result={result} memoryUnit={memoryUnit} />}
      <details className="panel">
        <summary>Raw resolved inputs &amp; sources</summary>
        <div className="panel-body">
          <pre>{JSON.stringify(result.resolvedInputs, null, 2)}</pre>
          <pre>{JSON.stringify({ model: result.modelSources, runtime: result.runtimeSources }, null, 2)}</pre>
        </div>
      </details>
    </div>
  );
}

type FlagRow = { flag: string; value: string; origin: Origin };

function SizesPanel({ result, memoryUnit }: { result: EstimateResult; memoryUnit: MemoryUnit }) {
  const mem = result.memory!;
  const util = numberValue(result.resolvedInputs.utilization) ?? 1;
  const configuredSeqs = numberValue(result.resolvedInputs.batch) ?? 1;
  const weightsBytes = mem.weights.bytes;
  const kvPerSeqBytes = configuredSeqs > 0 ? mem.kvCache.bytes / configuredSeqs : mem.kvCache.bytes;
  const dtype = result.resolvedInputs.weightDtype?.value;

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
      </div>
      <div className="sizes-scenarios">
        <span className="group-label">Per concurrency</span>
        {scenarios.map((scenario) => {
          const s = scenarioAt(scenario.seqs);
          return (
            <div className="scenario-card" key={scenario.title}>
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
            </div>
          );
        })}
      </div>
    </div>
  );
}

function CommandView({
  result,
  target,
  kvDtype,
  cacheTypeK,
  cacheTypeV
}: {
  result: EstimateResult;
  target: string;
  kvDtype: string;
  cacheTypeK: string;
  cacheTypeV: string;
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

  const flags: FlagRow[] =
    result.mode === "vllm"
      ? [
          { flag: "--max-model-len", value: valueStr(ri.context), origin: originOf(ri.context) },
          { flag: "--max-num-seqs", value: valueStr(ri.batch), origin: originOf(ri.batch) },
          { flag: "--kv-cache-dtype", value: kvDtype, origin: kvDtype === "auto" ? "default" : "you" },
          { flag: "--gpu-memory-utilization", value: valueStr(ri.utilization), origin: originOf(ri.utilization) },
          { flag: "--tensor-parallel-size", value: "1", origin: "default" }
        ]
      : [
          { flag: "--ctx-size", value: valueStr(ri.context), origin: originOf(ri.context) },
          { flag: "--parallel", value: valueStr(ri.batch), origin: originOf(ri.batch) },
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
        <p className="hint">These are the {bin} flags whose values drive the estimate above.</p>
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

function CalculationView({ result, memoryUnit }: { result: EstimateResult; memoryUnit: MemoryUnit }) {
  const variables = VARIABLES[result.mode];
  return (
    <div className="panel">
      <h2>How this was calculated</h2>
      <div className="panel-body">
        <div className="formula-list">
          <FormulaLine label="Weights" formula={result.formula!.weights} value={result.memory ? formatMemory(result.memory.weights, memoryUnit) : undefined} />
          <FormulaLine label="KV cache" formula={result.formula!.kvCache} value={result.memory ? formatMemory(result.memory.kvCache, memoryUnit) : undefined} />
          <FormulaLine label="Total" formula={result.formula!.total} value={result.memory ? formatMemory(result.memory.total, memoryUnit) : undefined} />
          <FormulaLine label="Overhead" formula={result.formula!.overhead} value={result.memory ? formatMemory(result.memory.overhead, memoryUnit) : undefined} />
        </div>
        <div className="var-table">
          <div className="var-head">
            <span>Input</span>
            <span>Value</span>
          </div>
          {variables.map((variable) => (
            <VariableRow key={variable.key} label={variable.label} role={variable.role} value={result.resolvedInputs[variable.key]} />
          ))}
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
