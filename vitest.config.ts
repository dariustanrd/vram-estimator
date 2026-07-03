import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["packages/**/*.test.ts", "apps/**/*.test.ts"],
    environment: "node"
  },
  resolve: {
    alias: {
      "@vram-estimator/core": new URL("./packages/core/src/index.ts", import.meta.url).pathname
    }
  }
});
