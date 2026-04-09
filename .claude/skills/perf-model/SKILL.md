---
name: perf-model
description: Perform theoretical performance modeling for an Ascend NPU operator. For MLA Prolog V3, use the existing perf model toolkit (14 CLI modes, 6 timing models, 3 optimization analysis modes). For other operators, build a model from the operator's kernel code and tiling logic.
argument-hint: <operator_path> [--chip 950|910b] [--scenario decode|prefill] [--bs <N>] [--heads <N>] [--quant bf16|int8|fp8] [--He <N>] [--mode <mode>]
---

# Ascend NPU Operator Performance Modeling

You are performing **theoretical performance modeling** for an operator on Ascend NPU hardware.

## Decision: Use Existing Toolkit or Build From Scratch?

**Check first**: if `<operator_path>` is `attention/mla_prolog_v3` (or `attention/mla_prolog`), a comprehensive perf model toolkit already exists at `attention/mla_prolog_v3/tests/perf/`. **Use it directly** — do NOT rebuild from scratch.

For any other operator, follow the "Build From Scratch" workflow in Phase 2 below.

---

## MLA Prolog V3: Use the Existing Perf Model Toolkit

### Quick Start

```bash
cd attention/mla_prolog_v3/tests/perf

# Full analysis (all modes at once)
python perf_analyzer.py --chip 950 --batch-size $BS --seq-len 1 --head-num $HEADS --mode full

# Single analysis mode
python perf_analyzer.py --chip 950 --batch-size 1 --head-num 128 --mode <mode>

# Cross-testcase report
python perf_analyzer.py --chip 950 --mode report
```

### 14 Available Analysis Modes

| Mode | Purpose | 950 only |
|------|---------|:--------:|
| `bound` | Per-stage MTE1/L0A/L0B/MMAD/FixPipe breakdown | |
| `pipeline` | AIC/AIV timeline + critical path | |
| `estimate` | Total kernel duration | |
| `advice` | Automated bottleneck detection + suggestions | |
| `search` | stepBatchSize search | |
| `report` | All testcases comparison | |
| `roofline` | Per-matmul roofline (AI, ridge point, efficiency) | ✓ |
| `block-search` | baseM/baseN/baseK exhaustive search | ✓ |
| `split-kn` | 2D N×K core split search for MM1/MM2 | ✓ |
| `split-kn-fused` | Fused K-accum + RmsNorm vs separate (supplement to split-kn) | ✓ |
| `split-m` | M-axis split for prefill | ✓ |
| `prefill` | K-outer loop cube-bound optimization | ✓ |
| `dequant-opt` | L0C→UB→L1 intra-core pipeline for MM3→MM4 (FP8/INT8) | ✓ |
| `full` | All above except dequant-opt and split-kn-fused | |

### Three Landed Optimization Analyses

These optimizations have perf model validation but kernel implementation is pending:

1. **Split-KN** (`--mode split-kn`): N×K 2D cube core splitting for MM1/MM2. Best config: MM1:N4×K7 + MM2:N1×K28 (BF16); MM1:N8×K4 + MM2:N1×K28 (FP8/INT8). Savings: −16 µs BF16 / −9.5 µs FP8 for decode bs1 hd128.

2. **Split-KN-Fused** (`--mode split-kn-fused`): Fuses K-accumulation into M-split RmsNorm step, avoiding a new barrier + GM round-trip. Savings: ~1.5 µs/call.

3. **DequantOpt** (`--mode dequant-opt`): L0C→UB direct (with VDEQ16 inline dequant) → UB compute (cast + RoPE) → UB→L1 direct, all intra-core. Eliminates 3 cross-core syncs + 3 GM round-trips. Savings: ~3.3 µs FP8/INT8.

Combined projection: FP8 decode bs1 hd128 from 48.7 → ~34.4 µs (−29%, 1.42× speedup).

### He Sweep (different model sizes)

The optimal split-kn config varies by He (hidden_size). Use:
```bash
python perf_analyzer.py --chip 950 --batch-size 1 --head-num 128 --He <value> --mode split-kn
```

Recommended configs per He are documented in `docs/MlaPrologV3_950_综合优化设计.md` §2.1.3.

### Key Design Documents

| Document | Path |
|----------|------|
| Comprehensive 3-optimization design | `docs/MlaPrologV3_950_综合优化设计.md` |
| DequantOpt detailed design | `docs/MlaPrologV3_DequantOpt设计.md` |
| Split-KN-Fused detailed design | `docs/MlaPrologV3_SplitKN_FusedAccum设计.md` |
| Split-KN corrected sweep report | `tests/perf/report_split_kn.md` |
| Perf modeling methodology | `tests/perf/methodology.md` |
| Perf module guide | `tests/perf/CLAUDE.md` |

### Testing the Perf Model

```bash
cd attention/mla_prolog_v3/tests/perf
python test_regression.py     # 24/24 golden snapshots must pass (bit-identical)
python test_perf_unit.py      # 35/35 unit tests must pass
```

After any numerical formula change, run `test_regression.py`. If the change is intentional (bug fix), inspect each `.diff` and reset with `python test_regression.py --reset`.

### 6 Timing Models (chip-total / contention convention)

All share the same convention: pass chip-total dimensions + `active_cores = total cores`, divide bandwidth by `active_cores` internally.

| Model | File | Purpose |
|-------|------|---------|
| `estimate_matmul_detailed` | `perf/model/matmul_detailed.py` | Per-cycle inner-loop (950, split-N) |
| `estimate_matmul_legacy` | `perf/model/matmul_legacy.py` | Chip-total (910B) |
| `estimate_matmul_split_kn` | `perf/model/matmul_split_kn.py` | 2D N×K split + K-accum |
| `estimate_matmul_split_m` | `perf/model/matmul_split_m.py` | M-axis split (prefill) |
| `estimate_matmul_k_outer` | `perf/model/matmul_k_outer.py` | K-outer (cube-bound prefill) |
| `estimate_vector_op` | `perf/model/vector_ops.py` | Vector MTE2→compute→MTE3 |

### Programmatic Usage

```python
from perf.model import OperatorParams, ASCEND_950, estimate_all_stages
from perf.tiling import compute_tiling
from perf.pipeline import build_pipeline_dag

params = OperatorParams(batch_size=1, seq_len=1, head_num=128)
tiling = compute_tiling(params, ASCEND_950)
stages = estimate_all_stages(params, tiling, ASCEND_950)
result = build_pipeline_dag(params, tiling, ASCEND_950, stages)
print(f"Pipeline total: {result.total_us:.2f} us")
print(f"Critical path: {' → '.join(result.critical_path)}")
```

---

## Other Operators: Build From Scratch

For operators without an existing perf model toolkit, follow this workflow.

### Phase 1: Read Operator Source

1. **Kernel code**: Read all `.h` files in `<operator_path>/op_kernel/` — look for:
   - Matmul calls (Mmad, MatmulSplitK, MatmulFullLoad, etc.)
   - Buffer size constants (L1_A_SIZE, L0A_PP_SIZE, etc.)
   - baseM, baseN, baseK, stepK parameters per matmul
   - Data types and quantization handling
   - Pipeline stages and sync points (AIC/AIV events)

2. **Tiling code**: Read `<operator_path>/op_host/*_tiling.cpp` and `*_tiling.h` — look for:
   - Core allocation per matmul (FillMatmul*Tiling functions)
   - N-axis split vs K-axis split strategy
   - stepBatchSize calculation
   - Workspace size computation

3. **Docs**: Read `<operator_path>/docs/` and `README.md` for:
   - Operator computation graph
   - Input/output tensor dimensions
   - Supported quantization modes

### Phase 2: Extract Matmul Parameters

For each matmul found, record:

| Field | Description |
|-------|-------------|
| Name | e.g., MM1_Cq, MM2_CkvKr |
| M | Batch dimension (typically T = BS × SeqLen) |
| K | Reduction dimension |
| N | Output dimension (total, before core split) |
| dtype_a | Left matrix element size in bytes |
| dtype_b | Right matrix element size in bytes |
| dtype_c | Output element size in bytes |
| split_mode | "split_n", "split_k", or "full_load" |
| active_cores | Number of AIC cores used |
| N_per_core | N / active_cores |
| a_reused | Whether A matrix is reused from a prior matmul (L2 warm) |

### Phase 3: Build Performance Model

Follow the methodology in `methodology.md` (in this skill directory):

1. Load chip specs from `chip_specs.md`
2. For each matmul, compute the inner-loop timing model:
   - Outer: HBM/L2 → L1 (MTE1), max(load_A, load_B)
   - Inner: L1 → L0A/L0B + MMAD, stepK iterations, double-buffered
   - Output: L0C → FixPipe, overlaps with outer loop
   - Per kL1 iter: `max(mte1_time, total_inner, fixpipe_time)`
   - Total: `kL1_loops × per_kl1`
3. Use **chip-total / contention** convention: `per_core_bw = chip_bw / active_cores`
4. Compute roofline metrics (AI, ridge point, efficiency)
5. If possible, build AIC/AIV pipeline DAG and find critical path

### Phase 4: Generate Report

Output a markdown report with these sections:

1. **Chip Specs Summary** — key hardware numbers
2. **Operator Overview** — matmul table with dimensions
3. **Roofline Analysis** — per-matmul AI vs ridge point, bound classification
4. **Detailed Timing** — per-matmul breakdown (MTE1, L0A, L0B, MMAD, FixPipe)
5. **Pipeline Analysis** — AIC/AIV timeline, critical path (if DAG built)
6. **Optimization Recommendations** — based on bottleneck analysis

Save the report to `<operator_path>/tests/perf/report_<chip>.md`.

### Phase 5: Summarize

Present key findings to the user:
- Which matmuls dominate execution time
- Whether the operator is compute-bound or memory-bound
- Top optimization opportunities with expected impact
