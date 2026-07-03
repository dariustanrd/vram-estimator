// Bytes-per-element for the dtype strings that actually appear in Hugging Face config.json
// (torch_dtype) and vLLM's kv_cache_dtype option. GGUF/llama.cpp quantized cache types (q4_0,
// q4_k, etc.) are intentionally NOT included here: they never appear in these fields, and their
// real bytes-per-element depends on block/type size, not a flat fraction (see gguf.ts
// cacheTypeBytes, which derives it exactly from the GGUF type table).
const DTYPE_BYTES: Record<string, number> = {
  float64: 8,
  double: 8,
  float32: 4,
  f32: 4,
  fp32: 4,
  bfloat16: 2,
  bf16: 2,
  float16: 2,
  fp16: 2,
  f16: 2,
  half: 2,
  fp8: 1,
  fp8_e4m3: 1,
  fp8_e5m2: 1,
  int8: 1,
  i8: 1
};

export function bytesForDtype(dtype: string | undefined | null): number | null {
  if (!dtype) return null;
  const normalized = dtype.toLowerCase().replace(/^torch\./, "");
  return DTYPE_BYTES[normalized] ?? null;
}
