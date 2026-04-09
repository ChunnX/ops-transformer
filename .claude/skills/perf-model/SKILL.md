---
name: perf-model
description: Perform theoretical performance modeling for an Ascend NPU operator. For MLA Prolog V3, use the existing perf model toolkit (14 CLI modes, 6 timing models, 3 optimization analysis modes). For other operators, build a model from the operator's kernel code and tiling logic. Optionally generates optimization recommendations with projected performance gains.
argument-hint: <operator_path> [--chip 950|910b] [--scenario decode|prefill] [--bs <N>] [--heads <N>] [--quant bf16|int8|fp8] [--He <N>] [--mode <mode>] [--optimize]
---

# Ascend NPU Operator Performance Modeling

You are performing **theoretical performance modeling** for an operator on Ascend NPU hardware.

---

## Step 0: Ask the User

Before starting, confirm with the user:

1. **Target chip**: 950 (per-cycle model) or 910B (chip-total model)? Default: 950.
2. **Scenario**: Decode (small T=1~32) or Prefill (large T≥128)?
3. **Quantization**: BF16, INT8, or FP8?
4. **Optimization suggestions?** Ask: "是否需要性能优化建议？我可以在建模同时分析瓶颈并提出可行的优化方案及预期收益。" If yes, follow the optimization workflow in `optimization_guide.md`.

---

## Step 1: Route — Existing Toolkit or Build From Scratch?

**Check first**: If `<operator_path>` is `attention/mla_prolog_v3` (or `attention/mla_prolog`), a comprehensive perf model toolkit already exists at `attention/mla_prolog_v3/tests/perf/`. **Use it directly** — do NOT rebuild from scratch. Skip to [MLA Prolog V3 Fast Path](#mla-prolog-v3-fast-path).

For **any other operator**, follow the [Generic Operator Workflow](#generic-operator-workflow) below.

---

## MLA Prolog V3 Fast Path

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

1. **Split-KN** (`--mode split-kn`): N×K 2D cube core splitting for MM1/MM2. Savings: −16 µs BF16 / −9.5 µs FP8 for decode bs1 hd128.
2. **Split-KN-Fused** (`--mode split-kn-fused`): Fuses K-accumulation into M-split RmsNorm step. Savings: ~1.5 µs/call.
3. **DequantOpt** (`--mode dequant-opt`): L0C→UB direct path, eliminates 3 cross-core syncs + 3 GM round-trips. Savings: ~3.3 µs FP8/INT8.

### Key Design Documents

| Document | Path |
|----------|------|
| Comprehensive 3-optimization design | `docs/MlaPrologV3_950_综合优化设计.md` |
| Perf modeling methodology | `tests/perf/methodology.md` |
| Perf module guide | `tests/perf/CLAUDE.md` |

### Testing the Perf Model

```bash
cd attention/mla_prolog_v3/tests/perf
python test_regression.py     # golden snapshots must pass (bit-identical)
python test_perf_unit.py      # unit tests must pass
```

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
```

---

## Generic Operator Workflow

For any CANN operator without an existing perf model toolkit.

### Phase 1: Read Operator Source & Identify Patterns

Read the operator source code and classify its computational pattern:

```
<operator_path>/
├── op_host/                      ← Tiling logic, shape inference
│   ├── *_tiling.h / *_tiling.cpp ← Core allocation, block sizes, stepK
│   ├── *_tiling_data.h           ← TilingData struct
│   └── *_def.cpp                 ← Operator registration
├── op_kernel/                    ← Device-side kernel
│   ├── *.cpp                     ← Kernel entry (__global__ function)
│   ├── *.h                       ← Core compute template class
│   └── arch*/ or service_*.h     ← Arch-specific or modular implementations
└── docs/                         ← Design docs
```

**What to extract from kernel code** (`op_kernel/*.h`):

| Pattern | Look for | Example |
|---------|----------|---------|
| **Matmul calls** | `Mmad`, `MatmulSplitN`, `MatmulFullLoad`, `matmulObj.Iterate()` | Each call = one matmul stage |
| **Buffer constants** | `L1_A_SIZE`, `L0A_PP_SIZE`, `UB_SIZE`, `BLOCK_SIZE` | Determines tile sizes |
| **Block parameters** | `baseM`, `baseN`, `baseK`, `stepK`, `nSplitSize` | Per-matmul tiling |
| **Data types** | Template params `<bf16, int8, fp8>`, `sizeof(T)` | Affects bandwidth & compute |
| **AIC/AIV split** | `ASCEND_IS_AIC`, `ASCEND_IS_AIV`, `if constexpr (CvMode)` | Pipeline structure |
| **Sync points** | `CrossCoreSetFlag`, `CrossCoreWaitFlag`, `SYNC_*` | Pipeline dependencies |
| **Vector ops** | `RmsNorm`, `Softmax`, `RoPE`, `Scatter`, `Dequant`, `DynamicQuant` | Vector stages |
| **Loop structure** | `for (step = 0; step < totalSteps; step++)` | Main iteration |

**What to extract from tiling code** (`op_host/*_tiling.cpp`):

| Field | Description |
|-------|-------------|
| `coreNum` / `blockNum` | Number of AIC cores per matmul |
| `singleCoreN` | N-axis work per core |
| `stepBatchSize` | M-axis batch per kernel step |
| `mm*BlockNum` | Per-matmul core allocation |
| Buffer factor | Single (1×) vs double (2×) buffer |

**Common CANN operator patterns:**

| Pattern | Operators | Characteristics |
|---------|-----------|-----------------|
| **Multi-matmul fusion** | MlaProlog, FFN, FlashAttention | Multiple sequential matmuls with vector ops between them; AIC/AIV pipeline |
| **Single matmul + vector** | GroupedMatmul, MatmulAdd | One matmul with pre/post-processing; simpler pipeline |
| **Pure vector** | MoeTokenPermute, RoPE operators | No cube ops; UB-based, MTE2/Vector/MTE3 pipeline |
| **Batched independent** | GroupedMatmul (GMM) | Multiple independent matmuls with per-group tiling |
| **Communication + compute** | MC2 operators | Cross-device data movement + local compute |

### Phase 2: Extract Computation Graph

For each operator, build a **stage table**:

```markdown
| Stage | Type | M | K | N | dtype_a | dtype_b | dtype_c | Pipeline | Active Cores |
|-------|------|---|---|---|---------|---------|---------|----------|-------------|
| MM1   | matmul | T | He | Hcq | 2B | 2B | 2B | AIC | 24 |
| RmsNorm | vector | T×Hcq | — | — | 4B | — | 2B | AIV | 64 |
| MM2   | matmul | T | Hcq | N×D | 2B | 2B | 2B | AIC | 32 |
| ...   | ... | ... | ... | ... | ... | ... | ... | ... | ... |
```

Record **data dependencies** between stages (which stage's output feeds into which stage's input) — these determine the pipeline DAG.

### Phase 3: Compute Per-Stage Timing

Follow the methodology in `methodology.md`:

#### For Matmul stages (AIC/Cube):

Use the **three-stage inner-loop model** (see `methodology.md` §3):

```
1. MTE1 (HBM/L2 → L1): time = max(A_load_time, B_load_time)
2. Inner loop (L1 → L0 + MMAD): time = stepK × max(l0a, l0b, mmad)
3. FixPipe (L0C → output): time = C_bytes / fixpipe_bw

per_kl1 = max(mte1, inner, fixpipe)
total = kL1Loops × per_kl1
```

Key parameters from `chip_specs.md`:
- Bandwidth: use `per_core_bw = min(per_cycle × freq, chip_total / active_cores)` (contention model)
- MMAD throughput: BF16 = 4096 MACs/cycle, FP8 = 8192 MACs/cycle
- Cache sizes: L0A=64KB, L0B=256KB, L0C=256KB, L1=512KB, UB=256KB (per core, 950)

#### For Vector stages (AIV):

Use the **instruction-level model** (see `chip_specs.md` §Vector Compute):

```
per_instr_cycles = (per_core_elements × 4 bytes) / 256    # 256 B/cycle
total_cycles = n_instructions × per_instr_cycles / 1.3    # IPC=1.3
vec_time = total_cycles / (1.65 × 10⁹)                    # freq=1.65 GHz

total = max(mte2_load_time, vec_time, mte3_store_time)     # pipelined
```

Refer to `chip_specs.md` §Vector Instruction Count Table for `n_instructions` by operation type.

#### For Sync overhead:

Each `CrossCoreSetFlag`/`WaitFlag` pair = **0.8 µs** on 950.

### Phase 4: Build Pipeline DAG & Find Critical Path

If the operator has AIC/AIV stages:

1. **Nodes**: Each matmul and vector stage
2. **Edges**: Data dependencies + sync events (CrossCore flags)
3. **Solve**: Topological sort → forward pass (earliest start/end) → backtrack critical path
4. **Total time** = max(end time of all leaf nodes)

Key insight: **Never naively sum AIC + AIV times**. Vector stages often overlap with the next matmul:
- AIV accumulation can hide behind the next AIC matmul
- Only the critical path determines end-to-end latency

### Phase 5: Generate Report

Output a **markdown report** with these sections:

1. **Chip Specs Summary** — key hardware numbers used
2. **Operator Overview** — stage table with dimensions
3. **Roofline Analysis** — per-matmul AI vs ridge point, bound classification
4. **Detailed Timing** — per-stage breakdown (MTE1, L0A, L0B, MMAD, FixPipe for matmuls; MTE2, Vec, MTE3 for vectors)
5. **Pipeline Analysis** — AIC/AIV timeline with critical path highlighted
6. **Bottleneck Summary** — which stages dominate, what resource limits them
7. **Optimization Recommendations** (if `--optimize` or user opted in) — see `optimization_guide.md`

Save the report to `<operator_path>/tests/perf/report_<chip>.md`.

### Phase 5b: Generate Mermaid Diagrams

Use the **Mermaid MCP** (`mcp__mermaidchart__validate_and_render_mermaid_diagram`) to create visual diagrams for the report. Generate at minimum:

1. **Hardware dataflow diagram** (flowchart) — HBM → L2 → AIC/AIV cache hierarchy with bandwidth annotations
2. **Pipeline timeline** (sequence diagram) — AIC/AIV execution with sync points, timing data, and critical path highlighted
3. **Optimization decision tree** (flowchart, if `--optimize`) — bottleneck classification → recommended techniques → projected gains

Validate each diagram via the MCP tool. Include `[在线编辑此图](link)` links from the MCP response. See `methodology.md` §7.2 for Mermaid templates.

### Phase 6: Present to User

Summarize key findings:
- Which stages dominate execution time (and why)
- Whether the operator is compute-bound or memory-bound
- Top optimization opportunities with expected impact (if optimization enabled)
- Actionable next steps (what to try first)

---

## Reference Files

| File | Content |
|------|---------|
| `chip_specs.md` | Ascend 950/910B hardware constants, bandwidth, cache, vector compute model |
| `methodology.md` | Matmul timing model, vector op model, roofline, pipeline DAG methodology |
| `optimization_guide.md` | Performance optimization decision tree, techniques by bottleneck type, projected gains |
