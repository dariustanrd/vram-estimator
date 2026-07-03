import { StrictMode, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import type { EstimateResult, GroundTruthValue, MemoryUnit } from "@vram-estimator/core";
import { formatMemory } from "@vram-estimator/core";
import "./styles.css";

type Mode = "vllm" | "llamacpp";

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
  const [result, setResult] = useState<EstimateResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const endpoint = mode === "vllm" ? "/api/estimate/vllm" : "/api/estimate/llamacpp";
  const title = mode === "vllm" ? "vLLM / Hugging Face" : "llama.cpp / GGUF";
  const placeholder =
    mode === "vllm"
      ? "meta-llama/Llama-3.1-8B-Instruct"
      : "https://huggingface.co/user/repo/resolve/main/model.gguf or repo::model.gguf";

  const parsedOverrides = useMemo(() => {
    const parsed: Record<string, number | string> = {};
    for (const [key, value] of Object.entries(overrides)) {
      if (!value.trim()) continue;
      const numeric = Number(value);
      parsed[key] = Number.isFinite(numeric) && key !== "modelDtype" ? numeric : value;
    }
    return parsed;
  }, [overrides]);

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
              context: numberOrUndefined(context),
              batch: numberOrUndefined(batch),
              kvDtype: kvDtype || undefined,
              overrides: parsedOverrides
            }
          : {
              source: target,
              hfToken: hfToken || undefined,
              context: numberOrUndefined(context),
              parallel: numberOrUndefined(batch),
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
            <button className={mode === "vllm" ? "active" : ""} onClick={() => setMode("vllm")}>
              vLLM
            </button>
            <button className={mode === "llamacpp" ? "active" : ""} onClick={() => setMode("llamacpp")}>
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
          <h2>{title}</h2>
          <label>
            Source
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
          <div className="grid2">
            <label>
              Context
              <input value={context} onChange={(event) => setContext(event.target.value)} placeholder="metadata default" />
            </label>
            <label>
              {mode === "vllm" ? "Batch" : "Parallel"}
              <input value={batch} onChange={(event) => setBatch(event.target.value)} placeholder="runtime default" />
            </label>
          </div>
          {mode === "vllm" ? (
            <label>
              KV dtype
              <select value={kvDtype} onChange={(event) => setKvDtype(event.target.value)}>
                <option value="auto">auto</option>
                <option value="float16">float16</option>
                <option value="bfloat16">bfloat16</option>
                <option value="fp8">fp8</option>
              </select>
            </label>
          ) : (
            <div className="grid2">
              <label>
                Cache K
                <select value={cacheTypeK} onChange={(event) => setCacheTypeK(event.target.value)}>
                  <option value="f16">f16</option>
                  <option value="f32">f32</option>
                  <option value="q8_0">q8_0</option>
                  <option value="q4_0">q4_0</option>
                </select>
              </label>
              <label>
                Cache V
                <select value={cacheTypeV} onChange={(event) => setCacheTypeV(event.target.value)}>
                  <option value="f16">f16</option>
                  <option value="f32">f32</option>
                  <option value="q8_0">q8_0</option>
                  <option value="q4_0">q4_0</option>
                </select>
              </label>
            </div>
          )}
          <OverrideFields mode={mode} overrides={overrides} setOverrides={setOverrides} />
          <button className="primary" disabled={loading || !target.trim()}>
            {loading ? "Estimating" : "Estimate"}
          </button>
          <p className="token-note">Frontend tokens are sent for the current lookup only and are not stored by this app.</p>
        </form>

        <section className="results">
          {error && <div className="notice error">{error}</div>}
          {!result && !error && <div className="empty">Enter a model source to calculate weights, KV cache, overhead, and total memory.</div>}
          {result && <ResultView result={result} memoryUnit={memoryUnit} />}
        </section>
      </section>
    </main>
  );
}

function OverrideFields({
  mode,
  overrides,
  setOverrides
}: {
  mode: Mode;
  overrides: Record<string, string>;
  setOverrides: (next: Record<string, string>) => void;
}) {
  const fields =
    mode === "vllm"
      ? ["weightBytes", "layers", "hiddenSize", "attentionHeads", "kvHeads", "headDim", "gqa", "kvBytes", "modelDtype"]
      : ["weightBytes", "layers", "hiddenSize", "attentionHeads", "kvHeads", "headDim", "gqa", "cacheBytesK", "cacheBytesV"];

  return (
    <details>
      <summary>Explicit overrides</summary>
      <div className="override-grid">
        {fields.map((field) => (
          <label key={field}>
            {field}
            <input
              value={overrides[field] ?? ""}
              onChange={(event) => setOverrides({ ...overrides, [field]: event.target.value })}
            />
          </label>
        ))}
      </div>
    </details>
  );
}

function ResultView({ result, memoryUnit }: { result: EstimateResult; memoryUnit: MemoryUnit }) {
  return (
    <div className="stack">
      {!result.ok && (
        <div className="notice">
          <strong>Missing exact metadata</strong>
          <ul>
            {result.missing.map((item) => (
              <li key={`${item.field}-${item.reason}`}>
                {item.field}: {item.reason}
              </li>
            ))}
          </ul>
        </div>
      )}
      {result.memory && (
        <div className="metrics">
          <Metric label="Weights" value={formatMemory(result.memory.weights, memoryUnit)} />
          <Metric label="KV cache" value={formatMemory(result.memory.kvCache, memoryUnit)} />
          <Metric label="Overhead" value={formatMemory(result.memory.overhead, memoryUnit)} />
          <Metric label="Total" value={formatMemory(result.memory.total, memoryUnit)} />
        </div>
      )}
      {result.notes.length > 0 && (
        <div className="notice assumptions">
          <strong>Assumptions &amp; caveats</strong>
          <p>This estimate uses simplifying assumptions. Read these before trusting the total for capacity planning.</p>
          <ul>
            {result.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </div>
      )}
      {result.formula && <CalculationView result={result} memoryUnit={memoryUnit} />}
      <Panel title="Resolved Inputs">
        <pre>{JSON.stringify(result.resolvedInputs, null, 2)}</pre>
      </Panel>
      <Panel title="Sources">
        <pre>{JSON.stringify({ model: result.modelSources, runtime: result.runtimeSources }, null, 2)}</pre>
      </Panel>
    </div>
  );
}

function CalculationView({ result, memoryUnit }: { result: EstimateResult; memoryUnit: MemoryUnit }) {
  return (
    <Panel title="Calculation">
      <div className="formula-list">
        <FormulaLine label="Weights" formula={result.formula!.weights} value={result.memory ? formatMemory(result.memory.weights, memoryUnit) : undefined} />
        <FormulaLine label="KV cache" formula={result.formula!.kvCache} value={result.memory ? formatMemory(result.memory.kvCache, memoryUnit) : undefined} />
        <FormulaLine label="Total" formula={result.formula!.total} value={result.memory ? formatMemory(result.memory.total, memoryUnit) : undefined} />
        <FormulaLine label="Overhead" formula={result.formula!.overhead} value={result.memory ? formatMemory(result.memory.overhead, memoryUnit) : undefined} />
      </div>
      <div className="legend">
        <h3>What the numbers mean</h3>
        <VariableRow name="exact_weight_bytes" description="Exact model weight bytes from Hugging Face file metadata or the GGUF tensor table." value={result.resolvedInputs.weightBytes} />
        <VariableRow name="layers" description="Transformer block count." value={result.resolvedInputs.layers} />
        <VariableRow name="kv_group_width" description="num_key_value_heads x head_dim: the exact per-token, per-layer element width of one K or V tensor. Replaces the old hidden_size x gqa shortcut, which is wrong whenever head_dim differs from hidden_size / num_attention_heads." value={result.resolvedInputs.kvGroupWidth} />
        <VariableRow name="hidden_size" description="Raw model hidden size / embedding length, shown for reference only (no longer used directly in the KV cache formula)." value={result.resolvedInputs.hiddenSize} />
        <VariableRow name="attention_heads" description="Number of query attention heads." value={result.resolvedInputs.attentionHeads} />
        <VariableRow name="kv_heads" description="Number of key/value heads (equal to attention_heads unless the model uses grouped-query attention)." value={result.resolvedInputs.kvHeads} />
        {result.mode === "vllm" ? (
          <VariableRow name="head_dim" description="Dimension of each attention head. Read from an explicit config field when available; otherwise assumed from hidden_size / attention_heads (flagged under Assumptions above when used)." value={result.resolvedInputs.headDim} />
        ) : (
          <>
            <VariableRow name="head_dim_k" description="Key head dimension. Read from GGUF attention.key_length when available; otherwise assumed from embedding_length / attention.head_count (flagged under Assumptions above when used)." value={result.resolvedInputs.headDimK} />
            <VariableRow name="head_dim_v" description="Value head dimension. Read from GGUF attention.value_length when available; otherwise assumed from embedding_length / attention.head_count (flagged under Assumptions above when used)." value={result.resolvedInputs.headDimV} />
          </>
        )}
        <VariableRow name="context" description="Tokens in the requested or metadata context window." value={result.resolvedInputs.context} />
        <VariableRow name={result.mode === "vllm" ? "batch" : "parallel"} description={result.mode === "vllm" ? "Concurrent sequences used for the estimate (worst case: each assumed to use the full context)." : "llama.cpp parallel slot count (context is reserved per slot; total reserved = context x parallel)."} value={result.resolvedInputs.batch} />
        <VariableRow name="kv_bytes" description={result.mode === "vllm" ? "Bytes per KV cache element from explicit KV dtype/model dtype." : "Average bytes per K/V cache element from selected cache types, using exact GGUF block/type sizes."} value={result.resolvedInputs.kvBytes} />
        <VariableRow name="gqa" description="Grouped-query attention ratio (kv_heads / attention_heads), shown for reference; the KV cache formula uses kv_group_width directly." value={result.resolvedInputs.gqa} />
        <VariableRow name="utilization" description="Runtime memory utilization divisor from the selected runtime defaults. Overhead is a modeled residual, not measured activation memory - see Assumptions above." value={result.resolvedInputs.utilization} />
      </div>
    </Panel>
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

function VariableRow({
  name,
  description,
  value
}: {
  name: string;
  description: string;
  value: GroundTruthValue<unknown> | undefined;
}) {
  return (
    <div className="variable-row">
      <div>
        <strong>{name}</strong>
        <p>{description}</p>
      </div>
      <code className={value?.assumed ? "assumed-value" : undefined}>
        {value ? `${String(value.value)} (${value.providedBy}; ${value.source})` : "missing"}
        {value?.assumed ? " \u26a0 assumed" : ""}
      </code>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="panel">
      <h2>{title}</h2>
      <div className="panel-body">{children}</div>
    </div>
  );
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
