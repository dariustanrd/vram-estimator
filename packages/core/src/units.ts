import type { MemoryAmount, MemoryUnit } from "./types.js";

export const DECIMAL_GB = 1_000_000_000;
export const BINARY_GIB = 1024 ** 3;

export function memory(bytes: number): MemoryAmount {
  return {
    bytes,
    gb: bytes / DECIMAL_GB,
    gib: bytes / BINARY_GIB
  };
}

export function formatMemory(amount: MemoryAmount, unit: MemoryUnit | "both" = "both"): string {
  if (unit === "gb") return `${amount.gb.toFixed(2)} GB`;
  if (unit === "gib") return `${amount.gib.toFixed(2)} GiB`;
  return `${amount.gb.toFixed(2)} GB / ${amount.gib.toFixed(2)} GiB`;
}
