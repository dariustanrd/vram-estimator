import { Hono } from "hono";
import { cors } from "hono/cors";
import {
  estimateLlamaCpp,
  estimateVllm,
  listRuntimeDefaults,
  resolveHfToken,
  type LlamaCppEstimateInput,
  type VllmEstimateInput
} from "@vram-estimator/core";

type Env = {
  Bindings: {
    HF_TOKEN?: string;
  };
};

type HfSearchApiModel = {
  id?: unknown;
  modelId?: unknown;
  downloads?: unknown;
  likes?: unknown;
  pipeline_tag?: unknown;
};

type HfModelApiModel = HfSearchApiModel & {
  siblings?: Array<{
    rfilename?: unknown;
    size?: unknown;
  }> | undefined;
};

async function searchHfModels(query: string, hfToken: string | undefined, filter?: string): Promise<HfSearchApiModel[]> {
  const url = new URL("https://huggingface.co/api/models");
  url.searchParams.set("search", query);
  url.searchParams.set("sort", "downloads");
  url.searchParams.set("direction", "-1");
  url.searchParams.set("limit", "8");
  url.searchParams.set("full", "false");
  if (filter) url.searchParams.set("filter", filter);

  const response = await fetch(url, hfToken ? { headers: { authorization: `Bearer ${hfToken}` } } : undefined);
  if (!response.ok) return [];
  const data = await response.json();
  return Array.isArray(data) ? (data as HfSearchApiModel[]) : [];
}

function hfModelId(model: HfSearchApiModel): string | undefined {
  return typeof model.id === "string" ? model.id : typeof model.modelId === "string" ? model.modelId : undefined;
}

function hfModelApiUrl(modelId: string): string {
  return `https://huggingface.co/api/models/${modelId.split("/").map(encodeURIComponent).join("/")}?blobs=true`;
}

function hfSearchResult(model: HfSearchApiModel) {
  const id = hfModelId(model);
  if (!id) return undefined;
  return {
    id,
    downloads: typeof model.downloads === "number" ? model.downloads : undefined,
    likes: typeof model.likes === "number" ? model.likes : undefined,
    pipelineTag: typeof model.pipeline_tag === "string" ? model.pipeline_tag : undefined
  };
}

export const app = new Hono<Env>();

app.onError((error, c) => {
  return c.json({ error: error.message }, 500);
});

app.use(
  "*",
  cors({
    origin: "*",
    allowHeaders: ["content-type", "authorization"],
    allowMethods: ["GET", "POST", "OPTIONS"]
  })
);

app.get("/api/runtime-defaults", (c) => c.json(listRuntimeDefaults()));

app.get("/api/hf/models", async (c) => {
  const query = c.req.query("q")?.trim();
  if (!query || query.length < 2) return c.json({ models: [] });

  const authHeader = c.req.header("authorization");
  const bearer = authHeader?.toLowerCase().startsWith("bearer ") ? authHeader.slice(7) : undefined;
  const hfToken = resolveHfToken(bearer, c.env?.HF_TOKEN);
  const models = (await searchHfModels(query, hfToken))
    .map(hfSearchResult)
    .filter((model): model is NonNullable<typeof model> => Boolean(model));

  c.header("cache-control", hfToken ? "private, no-store" : "public, max-age=60, s-maxage=300");
  return c.json({ models });
});

app.get("/api/hf/gguf", async (c) => {
  const query = c.req.query("q")?.trim();
  if (!query || query.length < 2) return c.json({ models: [] });
  const isDirectRepoQuery = query.includes("/") && !/\s/.test(query);
  const directRepoId = isDirectRepoQuery ? query : undefined;

  const authHeader = c.req.header("authorization");
  const bearer = authHeader?.toLowerCase().startsWith("bearer ") ? authHeader.slice(7) : undefined;
  const hfToken = resolveHfToken(bearer, c.env?.HF_TOKEN);
  const searchResults = await searchHfModels(query, hfToken, "gguf");
  const fallbackResults = searchResults.length > 0 ? [] : await searchHfModels(`${query} gguf`, hfToken);
  const directRepoResult: HfSearchApiModel[] = isDirectRepoQuery ? [{ id: query }] : [];
  const seen = new Set<string>();
  const modelResults = [...directRepoResult, ...searchResults, ...fallbackResults].filter((model) => {
    const id = hfModelId(model);
    if (!id || seen.has(id)) return false;
    seen.add(id);
    return true;
  });

  const details = await Promise.all(
    modelResults.slice(0, isDirectRepoQuery ? 7 : 6).map(async (model) => {
      const id = hfModelId(model);
      if (!id) return undefined;
      const response = await fetch(hfModelApiUrl(id), hfToken ? { headers: { authorization: `Bearer ${hfToken}` } } : undefined);
      if (!response.ok) return undefined;
      const detail = (await response.json()) as HfModelApiModel;
      return { fallback: model, detail };
    })
  );

  const allModels = details.flatMap((entry) => {
    if (!entry) return [];
    const repoId = hfModelId(entry.detail) ?? hfModelId(entry.fallback);
    if (!repoId) return [];
    const downloads =
      typeof entry.detail.downloads === "number"
        ? entry.detail.downloads
        : typeof entry.fallback.downloads === "number"
          ? entry.fallback.downloads
          : undefined;
    const likes =
      typeof entry.detail.likes === "number"
        ? entry.detail.likes
        : typeof entry.fallback.likes === "number"
          ? entry.fallback.likes
          : undefined;
    const pipelineTag =
      typeof entry.detail.pipeline_tag === "string"
        ? entry.detail.pipeline_tag
        : typeof entry.fallback.pipeline_tag === "string"
          ? entry.fallback.pipeline_tag
          : undefined;

    const ggufFiles = (entry.detail.siblings ?? []).filter(
      (file) => typeof file.rfilename === "string" && /\.gguf$/i.test(file.rfilename) && !/mmproj/i.test(file.rfilename)
    );
    const visibleFiles = repoId === directRepoId ? ggufFiles : ggufFiles.slice(0, 4);

    return visibleFiles.map((file) => {
      const filename = file.rfilename as string;
      return {
        id: `${repoId}::${filename}`,
        value: `${repoId}::${filename}`,
        repoId,
        file: filename,
        sizeBytes: typeof file.size === "number" ? file.size : undefined,
        downloads,
        likes,
        pipelineTag
      };
    });
  });
  const directRepoModels = directRepoId ? allModels.filter((model) => model.repoId === directRepoId) : [];
  const models = directRepoModels.length > 0 ? directRepoModels : allModels.slice(0, 16);

  c.header("cache-control", hfToken ? "private, no-store" : "public, max-age=60, s-maxage=300");
  return c.json({ models });
});

app.post("/api/estimate/vllm", async (c) => {
  const body = (await c.req.json()) as VllmEstimateInput;
  const authHeader = c.req.header("authorization");
  const bearer = authHeader?.toLowerCase().startsWith("bearer ") ? authHeader.slice(7) : undefined;
  const hfToken = resolveHfToken(body.hfToken ?? bearer, c.env?.HF_TOKEN);
  const result = await estimateVllm({ ...body, hfToken });
  return c.json(result, result.ok ? 200 : 422);
});

app.post("/api/estimate/llamacpp", async (c) => {
  const body = (await c.req.json()) as LlamaCppEstimateInput;
  const authHeader = c.req.header("authorization");
  const bearer = authHeader?.toLowerCase().startsWith("bearer ") ? authHeader.slice(7) : undefined;
  const hfToken = resolveHfToken(body.hfToken ?? bearer, c.env?.HF_TOKEN);
  const result = await estimateLlamaCpp({ ...body, hfToken });
  return c.json(result, result.ok ? 200 : 422);
});

export default app;
