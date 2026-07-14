import { afterEach, describe, expect, it, vi } from "vitest";
import { app } from "../src/index.js";

describe("API GGUF search", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns every GGUF file for an exact repo query", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = new URL(String(input));
      if (url.pathname === "/api/models") return jsonResponse([]);
      if (url.pathname === "/api/models/org/many") {
        return jsonResponse({
          id: "org/many",
          downloads: 123,
          likes: 4,
          siblings: [
            { rfilename: "model-q2.gguf", size: 2 },
            { rfilename: "model-q3.gguf", size: 3 },
            { rfilename: "model-q4.gguf", size: 4 },
            { rfilename: "model-q5.gguf", size: 5 },
            { rfilename: "model-q6.gguf", size: 6 },
            { rfilename: "model-q8.gguf", size: 8 },
            { rfilename: "mmproj-model.gguf", size: 1 },
            { rfilename: "README.md", size: 1 }
          ]
        });
      }
      throw new Error(`Unexpected URL ${url}`);
    });

    const response = await app.request("/api/hf/gguf?q=org%2Fmany");
    const body = (await response.json()) as { models: Array<{ file: string }> };

    expect(body.models.map((model) => model.file)).toEqual([
      "model-q2.gguf",
      "model-q3.gguf",
      "model-q4.gguf",
      "model-q5.gguf",
      "model-q6.gguf",
      "model-q8.gguf"
    ]);
  });

  it("falls back to GGUF search when an exact repo has no GGUF files", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = new URL(String(input));
      if (url.pathname === "/api/models") return jsonResponse([{ id: "org/missing-gguf" }]);
      if (url.pathname === "/api/models/org/missing") {
        return jsonResponse({
          id: "org/missing",
          siblings: [{ rfilename: "README.md", size: 1 }]
        });
      }
      if (url.pathname === "/api/models/org/missing-gguf") {
        return jsonResponse({
          id: "org/missing-gguf",
          siblings: [{ rfilename: "model-q4.gguf", size: 4 }]
        });
      }
      throw new Error(`Unexpected URL ${url}`);
    });

    const response = await app.request("/api/hf/gguf?q=org%2Fmissing");
    const body = (await response.json()) as { models: Array<{ repoId: string; file: string }> };

    expect(body.models).toEqual([{ id: "org/missing-gguf::model-q4.gguf", value: "org/missing-gguf::model-q4.gguf", repoId: "org/missing-gguf", file: "model-q4.gguf", sizeBytes: 4 }]);
  });
});

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json" }
  });
}