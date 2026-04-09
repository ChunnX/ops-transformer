# Ascend NPU Chip Specifications

## Ascend 950

| Parameter | Value |
|-----------|-------|
| AI Core (Cube) count | 32 |
| AI Vector count | 64 (2 per AIC) |
| Frequency | 1.65 GHz |

### Compute Throughput (per core, per cycle)

| Unit | BF16 | INT8 | FP8 |
|------|------|------|-----|
| MMAD (MACs/cycle) | 4096 | 4096 | 8192 |
| MMAD (FLOPS/cycle, ×2) | 8192 | 8192 | 16384 |

Chip total: BF16 = 432.5 TFLOPS, FP8 = 865.1 TFLOPS

### Bandwidth

**Chip-total bandwidth caps**:

| Bus | Chip Total |
|-----|-----------|
| HBM | 1.6 TB/s (1600 GB/s) |
| L2 | 5.2 TB/s (5200 GB/s) |

**Per-core bandwidth (bytes/cycle)**:

| Bus | B/cycle | GB/s @1.65GHz | Contention |
|-----|---------|---------------|-----------|
| HBM→L1 (MTE1/MTE2) | 33.3 | 54.9 (theoretical) | Yes: min(theoretical, chip_total/active_cores) |
| L2→Core | 108.0 | 178.2 (theoretical) | Yes: min(theoretical, chip_total/active_cores) |
| L1→L0A (MTE1) | 256.0 | 422.4 | No (per-core local bus) |
| L1→L0B (MTE1) | 256.0 | 422.4 | No |
| L0C→FixPipe | 256.0 | 422.4 | No |
| UB→L1 | 245.0 | 404.2 | No |
| UB→Reg | 512.0 | 844.8 | No |

**Effective per-core bandwidth at 32 cores**:
- HBM: min(54.9, 1600/32) = **50.0 GB/s**
- L2: min(178.2, 5200/32) = **162.5 GB/s**

### Cache Sizes (per core)

| Cache | Size | Purpose |
|-------|------|---------|
| UB (Unified Buffer) | 256 KB | Vector pipeline working memory |
| L1 | 512 KB | Cube pipeline, A+B tile staging |
| L0A (input A) | 64 KB | Left matrix tile for MMAD |
| L0B (input B) | 256 KB | Right matrix tile for MMAD (transposed) |
| L0C (accumulator) | 256 KB | MMAD output accumulator |

### Contention Rule

When `active_cores × per_core_bw > chip_total_bw`:
```
effective_per_core_bw = chip_total_bw / active_cores
```

### Mixed HBM/L2 Bandwidth

- First access to data: HBM bandwidth (cold path)
- Reused data (e.g., shared A matrix in split-N): L2 bandwidth (warm path)
- Cross-matmul reuse: if matmul B uses the same input as matmul A, it hits L2

### Synchronization Overhead

| Event | Cost |
|-------|------|
| CrossCoreSetFlag + WaitFlag (per pair) | 0.8 µs |
| Intra-core HardEvent (FIX_V, V_MTE3, etc.) | ~0 (not modeled, <2 cycles) |

---

## Ascend 950 — Vector Compute Model (Instruction-Level)

### Core Principle

Unlike the chip-total scalar model (910B), the 950 vector model works at **instruction granularity**:

```
每条指令耗时 (cycles) = data_volume (bytes) / 256 (bytes/cycle)
基本块耗时 (cycles)   = Σ(每条指令耗时) / IPC
wall-clock time (s)   = 基本块耗时 / freq_hz
```

Where:
- **256 B/cycle**: Each AIV core processes 256 bytes per vector instruction
- **IPC = 1.3**: Pipeline overlap between independent instructions (empirical)
- **All vector ops execute on float32 (4 bytes/element)**, even if input is BF16/INT8 — the cast happens first
- `data_volume = per_core_elements × 4 bytes`

### Complete Formula

```python
def vector_compute_time(per_core_elements, n_instructions, freq_ghz=1.65):
    """
    per_core_elements: number of elements processed by one AIV core
    n_instructions: number of vector instructions (from table below)
    returns: time in seconds
    """
    data_bytes = per_core_elements * 4          # float32
    per_instr_cycles = data_bytes / 256         # 256 B/cycle
    total_cycles = n_instructions * per_instr_cycles / 1.3  # IPC
    return total_cycles / (freq_ghz * 1e9)
```

### Three-Stage Pipeline (MTE2 → Vector → MTE3)

Vector operations follow a load-compute-store pipeline:

```
mte2_time = load_bytes / per_core_bw("HBM", active_aiv_cores)
vec_time  = vector_compute_time(elements, n_instructions)
mte3_time = store_bytes / per_core_bw("HBM", active_aiv_cores)

total = max(mte2_time, vec_time, mte3_time)    # pipelined overlap
```

For **scatter** (indexed write): `mte3_bw *= 0.4` (scatter_efficiency).

### Vector Instruction Count Table

Each "instruction" processes 256 bytes of float32 data per cycle per AIV core. The instruction count depends on the **operation type** and **quantization mode**.

#### BF16 Mode (no weight quantization on MM1/MM2)

| Operation | Instructions | Breakdown |
|-----------|:-----------:|-----------|
| **RmsNorm** (standard) | 6 | cast(bf16→f32) + mul(x²) + reducesum + rsqrt_mul + mul(γ) + cast(f32→bf16) |
| **RoPE** (rotary embedding) | 6 | split_even_odd + mul(cos) + mul(sin) + sub + add + interleave |
| **Scatter** (indexed cache write) | 1 | Data movement dominated (MTE3 bound, not compute) |
| **MulScale** (scalar multiply) | 2 | mul(scale) + cast |
| **DynamicQuant** (BF16→INT8) | 5 | abs + reducemax + mul(scale) + cast(f32→i8) + cast(extra) |

#### INT8/FP8 Quantization Mode (MM1/MM2 output is INT32/FP32)

| Operation | Instructions | Delta vs BF16 | Reason |
|-----------|:-----------:|:-------------:|--------|
| **RmsNorm** (with dequant) | 7 | +1 | Additional cast(i32→f32) + mul(deqScale) at input |
| **RmsNorm + KV Quant** | 10 | +4 | Above + abs + reducemax + mul(kvScale) + cast for KV cache quantization |
| **Dequant** (standalone) | 3 | N/A | cast(i32→f32) + mul(rowScale) + mul(colScale) |
| **RoPE** | 6 | same | Input already dequantized to BF16 |
| **Scatter** | 1 | same | — |

#### Generalizing to Other Operators

For operations not in the table, estimate instruction count by analyzing the kernel's vector compute:

| Operation Pattern | Typical Instructions | Examples |
|------------------|:-------------------:|---------|
| Element-wise unary | 1–2 | Abs, Neg, Cast, Relu |
| Element-wise binary | 1 | Add, Mul, Sub |
| Normalize (reduce + scale) | 5–7 | RmsNorm, LayerNorm, Softmax |
| Reduction (full) | 3–4 | ReduceSum, ReduceMax |
| Complex transform | 6–8 | RoPE, GeLU, SiLU |
| Quantize (with scale compute) | 4–5 | DynamicQuant, PerTileQuant |
| Dequantize | 2–3 | Dequant, DequantPerChannel |

### Worked Example: RopeQr (Decode BS=1, N=128, Dr=64, BF16)

```
Per AIV core (64 cores total):
  elements = 1 × 128 × 64 / 64 = 128 elements per core
  (but for MLA Prolog, RopeQr runs on 1 AIV core, so elements = 1 × 128 × 64 = 8192)

  data_bytes = 8192 × 4 = 32,768 bytes
  per_instr_cycles = 32,768 / 256 = 128 cycles
  n_instructions = 6 (RoPE)
  total_cycles = 6 × 128 / 1.3 = 590.8 cycles
  vec_time = 590.8 / 1.65e9 = 0.358 µs

  Load: 8192 × 2 + 64 × 2 = 16,512 bytes
  mte2_time = 16,512 / 50e9 = 0.33 µs

  Store: 8192 × 2 = 16,384 bytes
  mte3_time = 16,384 / 50e9 = 0.33 µs

  total = max(0.33, 0.358, 0.33) = 0.358 µs → VECTOR bound
```

Note: actual perf model reports ~0.105 µs because it uses the legacy `vec_ops_per_sec` path for some operations. The instruction-level model is more accurate for 950.

---

## Ascend 910B

| Parameter | Value |
|-----------|-------|
| AI Core (Cube) count | 24 |
| AI Vector count | 48 |
| BF16 compute | 320.0 TFLOPS |
| INT8 compute | 640.0 TOPS |
| FP8 compute | 640.0 TFLOPS |
| HBM bandwidth | 1.2 TB/s |
| L2 bandwidth | 6.4 TB/s |
| Vector BF16 | 10.0 TOPS (chip-total, not per-instruction) |

### Cache Sizes (per core)

| Cache | Size |
|-------|------|
| UB | 192 KB |
| L1 | 512 KB |
| L0A | 64 KB |
| L0B | 64 KB |
| L0C | 128 KB |

### 910B Vector Model (Legacy Chip-Total)

The 910B model does NOT have per-instruction vector modeling. It uses a scalar chip-total throughput:

```
vec_time = total_ops / (vec_ops_per_sec_bf16 / aiv_num)
```

Where `vec_ops_per_sec_bf16 = 10.0 × 10¹²` ops/sec (chip total across all 48 AIV cores).

This is less accurate than the 950 instruction-level model — it does not distinguish between operation types.

---

## Data Type Sizes

| Type | Bytes | Used In |
|------|-------|---------|
| BF16 / FP16 | 2 | Default activation/weight |
| INT8 | 1 | Quantized weight |
| FP8 (E4M3) | 1 | MXFP8 activation/weight |
| FP32 / float | 4 | Matmul output accumulator (INT8/FP8 mode), vector compute |
| INT32 | 4 | INT8 matmul output accumulator |

### Quantization Impact on Matmul

| Quant Mode | dtype_a (activation) | dtype_b (weight) | dtype_c (output) | MMAD throughput |
|------------|---------------------|------------------|-----------------|-----------------|
| BF16 (no quant) | 2B | 2B | 2B | 4096 MACs/cycle |
| INT8 partial | 2B | 1B | 4B | 4096 MACs/cycle |
| INT8 full | 1B | 1B | 4B | 4096 MACs/cycle |
| MXFP8 | 1B | 1B | 4B | 8192 MACs/cycle |

### Roofline Ridge Points (950, 32 cores)

| Dtype | Peak Compute (per core) | HBM BW (per core) | Ridge Point |
|-------|------------------------|-------------------|-------------|
| BF16 | 13,517 GFLOPS | 50 GB/s | **270.3 FLOPs/B** |
| FP8 | 27,034 GFLOPS | 50 GB/s | **540.7 FLOPs/B** |
