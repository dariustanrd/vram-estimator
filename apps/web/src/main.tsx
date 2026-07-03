import { StrictMode, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import type { EstimateResult, GroundTruthValue } from "@vram-estimator/core";
import { formatMemory } from "@vram-estimator/core";
import "./styles.css";

type Mode = "vllm" | "llamacpp";

function App() {
  const [mode, setMode] = useState<Mode>("vllm");
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
        <div className="segmented" role="tablist">
          <button className={mode === "vllm" ? "active" : ""} onClick={() => setMode("vllm")}>
            vLLM
          </button>
          <button className={mode === "llamacpp" ? "active" : ""} onClick={() => setMode("llamacpp")}>
            llama.cpp
          </button>
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
          {result && <ResultView result={result} />}
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
      ? ["weightBytes", "layers", "hiddenSize", "attentionHeads", "kvHeads", "gqa", "kvBytes", "modelDtype"]
      : ["weightBytes", "layers", "hiddenSize", "attentionHeads", "kvHeads", "gqa", "cacheBytesK", "cacheBytesV"];

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

function ResultView({ result }: { result: EstimateResult }) {
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
          <Metric label="Weights" value={formatMemory(result.memory.weights)} />
          <Metric label="KV cache" value={formatMemory(result.memory.kvCache)} />
          <Metric label="Overhead" value={formatMemory(result.memory.overhead)} />
          <Metric label="Total" value={formatMemory(result.memory.total)} />
        </div>
      )}
      {result.formula && <CalculationView result={result} />}
      <Panel title="Resolved Inputs">
        <pre>{JSON.stringify(result.resolvedInputs, null, 2)}</pre>
      </Panel>
      <Panel title="Sources">
        <pre>{JSON.stringify({ model: result.modelSources, runtime: result.runtimeSources }, null, 2)}</pre>
      </Panel>
      {result.notes.length > 0 && (
        <Panel title="Notes">
          <ul>
            {result.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </Panel>
      )}
    </div>
  );
}

function CalculationView({ result }: { result: EstimateResult }) {
  return (
    <Panel title="Calculation">
      <div className="formula-list">
        <FormulaLine label="Weights" formula={result.formula!.weights} />
        <FormulaLine label="KV cache" formula={result.formula!.kvCache} />
        <FormulaLine label="Total" formula={result.formula!.total} />
        <FormulaLine label="Overhead" formula={result.formula!.overhead} />
      </div>
      <div className="legend">
        <h3>What the numbers mean</h3>
        <VariableRow name="exact_weight_bytes" description="Exact model weight bytes from Hugging Face file metadata or the GGUF tensor table." value={result.resolvedInputs.weightBytes} />
        <VariableRow name="layers" description="Transformer block count." value={result.resolvedInputs.layers} />
        <VariableRow name="hidden" description="Model hidden size / embedding length." value={result.resolvedInputs.hiddenSize} />
        <VariableRow name="context" description="Tokens in the requested or metadata context window." value={result.resolvedInputs.context} />
        <VariableRow name={result.mode === "vllm" ? "batch" : "parallel"} description={result.mode === "vllm" ? "Concurrent sequences used for the estimate." : "llama.cpp parallel sequence count."} value={result.resolvedInputs.batch} />
        <VariableRow name="kv_bytes" description={result.mode === "vllm" ? "Bytes per KV cache element from explicit KV dtype/model dtype." : "Average bytes per K/V cache element from selected cache types."} value={result.resolvedInputs.kvBytes} />
        <VariableRow name="gqa" description="Grouped-query attention ratio: key/value heads divided by attention heads." value={result.resolvedInputs.gqa} />
        <VariableRow name="utilization" description="Runtime memory utilization divisor from the selected runtime defaults." value={result.resolvedInputs.utilization} />
      </div>
    </Panel>
  );
}

function FormulaLine({ label, formula }: { label: string; formula: string }) {
  return (
    <div className="formula-line">
      <span>{label}</span>
      <code>{formula}</code>
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
      <code>{value ? `${String(value.value)} (${value.providedBy}; ${value.source})` : "missing"}</code>
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
