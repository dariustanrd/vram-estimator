# VRAM Estimator

Inference-only VRAM estimator for vLLM and llama.cpp.

The tool uses exact Hugging Face or GGUF metadata where available and reports
missing required values instead of inferring them from model names.

## Development

```bash
npm install
npm test
npm run typecheck
npm run dev:web
npm run dev:api
```

## CLI

```bash
npm run cli -- vllm meta-llama/Llama-3.1-8B-Instruct --context 8192
npm run cli -- llamacpp https://example.com/model.gguf --context 8192
```
