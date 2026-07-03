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
