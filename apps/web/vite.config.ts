import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@vram-estimator/core": new URL("../../packages/core/src/index.ts", import.meta.url).pathname
    }
  },
  server: {
    proxy: {
      "/api": "http://localhost:8787"
    }
  }
});
