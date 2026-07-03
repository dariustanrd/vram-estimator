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

Runtime knobs use each backend's own CLI flag names, and model overrides use that
backend's metadata field names (Hugging Face `config.json` for vLLM, GGUF metadata
keys for llama.cpp):

```bash
# vLLM (vllm serve flags + config.json field overrides)
npm run cli -- vllm meta-llama/Llama-3.1-8B-Instruct --max-model-len 8192 --max-num-seqs 32
npm run cli -- vllm meta-llama/Llama-3.1-8B-Instruct --kv-cache-dtype fp8 --num-key-value-heads 8

# llama.cpp (llama-server flags + GGUF metadata key overrides)
npm run cli -- llamacpp https://example.com/model.gguf --ctx-size 8192 --parallel 4
npm run cli -- llamacpp repo::model.gguf --cache-type-k q8_0 --cache-type-v q8_0
```

Run `npm run cli` with no arguments for the full flag list.
