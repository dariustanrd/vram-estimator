import { bytesForDtype } from "./dtypes.js";
import type { Fetcher, HfAuth } from "./types.js";

export type HfConfig = Record<string, unknown>;

export type HfModelMetadata = {
  modelId?: string | undefined;
  revision?: string | undefined;
  config: HfConfig;
  configUrl: string;
  repoApiUrl?: string | undefined;
  weightBytes?: number | undefined;
  weightFiles: Array<{ path: string; size: number; source: string }>;
  sourceDetails: HfModelSourceDetails;
  sources: string[];
};

export type HfModelSourceDetails = {
  provider: "huggingface";
  modelId?: string | undefined;
  revision?: string | undefined;
  repoApiUrl?: string | undefined;
  parsedFiles: Array<{
    path: string;
    url: string;
    fields: HfConfig;
  }>;
  weightBytes?: number | undefined;
  weightFiles: Array<{ path: string; size: number; source: string }>;
};

type HfSibling = {
  rfilename: string;
  size?: number | undefined;
};

type HfApiModel = {
  id?: string | undefined;
  sha?: string | undefined;
  siblings?: HfSibling[] | undefined;
};

export function authHeaders(auth?: HfAuth): HeadersInit {
  return auth?.token ? { authorization: `Bearer ${auth.token}` } : {};
}

export function isUrl(value: string): boolean {
  try {
    const parsed = new URL(value);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

export function parseHfConfigUrl(value: string): { modelId?: string | undefined; revision?: string | undefined; path?: string | undefined } {
  if (!isUrl(value)) return {};
  const url = new URL(value);
  if (url.hostname !== "huggingface.co") return {};
  const parts = url.pathname.split("/").filter(Boolean);
  const markerIndex = parts.findIndex((part) => part === "resolve" || part === "raw");
  if (markerIndex < 1 || markerIndex + 2 >= parts.length) return {};
  const modelId = parts.slice(0, markerIndex).join("/");
  return {
    modelId,
    revision: parts[markerIndex + 1],
    path: parts.slice(markerIndex + 2).join("/")
  };
}

export function resolveHfToken(userToken?: string, envToken?: string): string | undefined {
  return userToken || envToken || undefined;
}

export async function fetchHfModelMetadata(
  input: string,
  auth?: HfAuth,
  fetcher: Fetcher = fetch
): Promise<HfModelMetadata> {
  if (isUrl(input)) {
    const configRes = await fetcher(input, { headers: authHeaders(auth) });
    if (!configRes.ok) throw new Error(`Failed to fetch config URL: ${configRes.status}`);
    const config = (await configRes.json()) as HfConfig;
    const parsed = parseHfConfigUrl(input);
    if (!parsed.modelId) {
      return {
        config,
        configUrl: input,
        weightFiles: [],
        sourceDetails: hfSourceDetails({
          config,
          configUrl: input,
          configPath: parsed.path ?? "config.json",
          weightFiles: []
        }),
        sources: [input]
      };
    }
    return fetchHfModelMetadataFromRepo(parsed.modelId, parsed.revision, auth, fetcher, config, input);
  }

  return fetchHfModelMetadataFromRepo(input, undefined, auth, fetcher);
}

async function fetchHfModelMetadataFromRepo(
  modelId: string,
  revision: string | undefined,
  auth: HfAuth | undefined,
  fetcher: Fetcher,
  providedConfig?: HfConfig,
  providedConfigUrl?: string
): Promise<HfModelMetadata> {
  const repoApiUrl = `https://huggingface.co/api/models/${modelId}${revision ? `/revision/${revision}` : ""}?blobs=true`;
  const apiRes = await fetcher(repoApiUrl, { headers: authHeaders(auth) });
  if (!apiRes.ok) throw new Error(`Failed to fetch Hugging Face model metadata: ${apiRes.status}`);
  const api = (await apiRes.json()) as HfApiModel;
  const resolvedRevision = api.sha ?? revision ?? "main";
  const configUrl =
    providedConfigUrl ?? `https://huggingface.co/${modelId}/raw/${resolvedRevision}/config.json`;
  const config =
    providedConfig ??
    ((await checkedJson(fetcher, configUrl, auth)) as HfConfig);

  const siblings = api.siblings ?? [];
  const weightCandidates = siblings
    .filter((file) => /\.(safetensors|bin)$/i.test(file.rfilename))
    .filter((file) => !/optimizer|training_args|scheduler|tokenizer/i.test(file.rfilename));
  const preferred = weightCandidates.some((file) => /\.safetensors$/i.test(file.rfilename))
    ? weightCandidates.filter((file) => /\.safetensors$/i.test(file.rfilename))
    : weightCandidates;
  const weightFiles = await Promise.all(
    preferred.map(async (file) => {
      const source = `https://huggingface.co/${modelId}/resolve/${resolvedRevision}/${file.rfilename}`;
      const size = file.size ?? (await headContentLength(fetcher, source, auth));
      return size === undefined ? undefined : { path: file.rfilename, size, source };
    })
  );
  const exactWeightFiles = weightFiles.filter((file): file is NonNullable<typeof file> => Boolean(file));
  const weightBytes =
    exactWeightFiles.length > 0
      ? exactWeightFiles.reduce((sum, file) => sum + file.size, 0)
      : undefined;

  return {
    modelId,
    revision: resolvedRevision,
    config,
    configUrl,
    repoApiUrl,
    weightBytes,
    weightFiles: exactWeightFiles,
    sourceDetails: hfSourceDetails({
      modelId,
      revision: resolvedRevision,
      repoApiUrl,
      config,
      configUrl,
      configPath: parseHfConfigUrl(configUrl).path ?? "config.json",
      weightBytes,
      weightFiles: exactWeightFiles
    }),
    sources: [
      repoApiUrl,
      configUrl,
      ...exactWeightFiles.map((file) => `${file.path}: ${file.source}`)
    ]
  };
}

function hfSourceDetails(input: {
  modelId?: string | undefined;
  revision?: string | undefined;
  repoApiUrl?: string | undefined;
  config: HfConfig;
  configUrl: string;
  configPath: string;
  weightBytes?: number | undefined;
  weightFiles: Array<{ path: string; size: number; source: string }>;
}): HfModelSourceDetails {
  return {
    provider: "huggingface",
    modelId: input.modelId,
    revision: input.revision,
    repoApiUrl: input.repoApiUrl,
    parsedFiles: [
      {
        path: input.configPath,
        url: input.configUrl,
        fields: input.config
      }
    ],
    weightBytes: input.weightBytes,
    weightFiles: input.weightFiles
  };
}

async function checkedJson(fetcher: Fetcher, url: string, auth?: HfAuth): Promise<unknown> {
  const res = await fetcher(url, { headers: authHeaders(auth) });
  if (!res.ok) throw new Error(`Failed to fetch JSON ${url}: ${res.status}`);
  return res.json();
}

async function headContentLength(
  fetcher: Fetcher,
  url: string,
  auth?: HfAuth
): Promise<number | undefined> {
  const res = await fetcher(url, { method: "HEAD", headers: authHeaders(auth) });
  if (!res.ok) return undefined;
  const linked = res.headers.get("x-linked-size");
  const content = res.headers.get("content-length");
  const raw = linked ?? content;
  if (!raw) return undefined;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : undefined;
}

export function numberField(config: HfConfig, field: string): number | undefined {
  const value = config[field];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

export function stringField(config: HfConfig, field: string): string | undefined {
  const value = config[field];
  return typeof value === "string" ? value : undefined;
}

export function kvBytesFromDtype(dtype: string | undefined): number | undefined {
  return bytesForDtype(dtype) ?? undefined;
}
