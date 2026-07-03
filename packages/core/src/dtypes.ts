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
  i8: 1,
  q8_0: 1,
  int4: 0.5,
  uint4: 0.5,
  q4_0: 0.5,
  q4_1: 0.5,
  q4_k: 0.5,
  q4_k_m: 0.5,
  q4_k_s: 0.5
};

export function bytesForDtype(dtype: string | undefined | null): number | null {
  if (!dtype) return null;
  const normalized = dtype.toLowerCase().replace(/^torch\./, "");
  return DTYPE_BYTES[normalized] ?? null;
}
