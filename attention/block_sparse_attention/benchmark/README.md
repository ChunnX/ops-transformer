# `block_sparse_attention` benchmark

`benchmark.py` is a latency benchmark for the
`attention/block_sparse_attention` kernel, measuring its speedup over an
`npu_fusion_attention` dense baseline across a per-row block-sparse pattern
swept through sparsities 0..0.9. All workload generators (`gen_pfa_inputs`,
`create_attention_mask`, `sabi_to_block_sparse_mask`, frame/sparsity policy,
timing harness) live in the sibling `util.py` so this directory ships as a
fully self-contained feature — no cross-tree imports, no `sys.path` patching.

## Prerequisites

Refer to the prerequisites of the [pytest](../tests/pytest/README.md)

## What the table measures

- **Workload**: `B=1, H=3, S_q=S_kv=118 806, D=128`, BF16, BNSD layout. Random
  Q/K/V tensors; per-row block-level sparsity pattern with `FrameWidths(29,1,29,1)`
  by default (see [Frame pattern](#frame-pattern-sink--text-prompt-tokens) below).
- **Each row**: a different sparsity level (fraction of KV blocks dropped per
  Q tile). The frame forces a minimum density, so high sparsities can become
  infeasible — those rows print `INFEASIBLE`.
- **Timing**: `N_REPEATS=10`, `N_WARMUP=2` runs per sparsity, latency averaged
  with `torch.npu.Event`.
- **Correctness column** (`Outputs_equal`): only computed at sparsity 0 by
  default (force-dense reference path). Set `RUN_REFERENCE = True` in the
  script to enable per-sparsity correctness comparison; the full dense
  `[B, H, S, S]` reference mask costs ~42 GB and OOMs at this S — see
  `attention/block_sparse_attention/tests/pytest/` (`test_attn.py`, `test_lse.py`,
  `test_joint.py`) for a smaller-S pytest correctness suite.

## Frame pattern (sink + text prompt tokens)

The frame forces a fixed set of blocks to always be kept, regardless of the
random sparse selection: the left/right border columns are kept in every
query row, and the top/bottom border rows are kept fully dense.
`FrameWidths(left_cols, right_cols, top_rows, bottom_rows)` widths are in
BLOCK units (cols in kv-block lengths, rows in q-block lengths). Setting any
width to `0` disables that side of the frame.

Why this matters in practice: multimodal video-generation workloads commonly
use a **sink + text** prompt layout — a handful of leading tokens
(BOS / system / attention-sink anchors) and a chunk of text-prompt tokens
that every video query token must attend to. Those tokens live in the
leftmost KV columns, so the left-frame columns model them. Their attention
scores are non-negligible for output quality — dropping them in the random
sparse selection causes visible regressions in the generated frames. The
right column mirrors this for any trailing anchor tokens; the top/bottom
dense rows reserve a small set of query positions (e.g. boundary frames or
register tokens) that need full context.

The benchmark default `FrameWidths(29,1,29,1)` is calibrated for that
workload class at 128×128 granularity. When changing the KV block size, the
left/right widths should be scaled inversely to keep token-coverage matched
(see Run 2 below: at 128×512 the left frame is `8` instead of `29`, since
`8*512 ≈ 29*128`).

## Running

```bash
cd attention/block_sparse_attention/benchmark

# Run 1 — default: Q block size = KV block size 128, default frame FrameWidths(29,1,29,1)
python benchmark.py

# Run 2 — override: KV block size 512, token-matched frame FrameWidths(8,1,29,1)
BSA_BLOCK_SIZE_KV=512 BSA_FRAME=8,1,29,1 python benchmark.py
```

Defaults — shape (`B`/`H`/`S`/`D`), sparsity sweep, `FRAME`, `BLOCK_SIZE_Q`,
`BLOCK_SIZE_KV`, dtype, layout — are defined at the top of `util.py`
in this directory. To change them, edit `util.py` (or override at
the env-var level — see the "Environment variables" section below).

## Results — 2026-05-12, Ascend 910B2 / CANN 8.5.0-beta.1

Two runs at different KV block granularity. Every row shows latency in µs;
the speedup column reports `(dense baseline) / (our latency)` where the dense
baseline is `npu_fusion_attention` at sparsity = 0 (printed in the
`Ref_Latency_[usec]` column of the first row of each run).

### Speedup summary vs dense baseline

Dense baseline = **157,261 µs** (`npu_fusion_attention` at sparsity 0, from
Run 1's `Ref_Latency_[usec]`; Run 2's baseline reading of 157,378 µs is within
NPU-clock noise of this).

| Sparsity | Run 1 (128×128) µs | Run 2 (128×512) µs | Speedup vs dense |
|---------:|-------------------:|-------------------:|-----------------:|
| 0.00 | 149,960 | 150,151 | 1.05× |
| 0.10 | 136,602 | 136,249 | 1.15× |
| 0.20 | 121,843 | 121,634 | 1.29× |
| 0.30 | 107,140 | 106,928 | 1.47× |
| 0.40 |  92,073 |  92,044 | 1.71× |
| 0.50 |  77,249 |  77,263 | **2.04×** |
| 0.60 |  61,088 |  61,112 | **2.57×** |
| 0.70 |  45,480 |  45,626 | **3.46×** |
| 0.80 |  29,936 |  29,911 | **5.25×** |
| 0.90 |  14,823 |  14,681 | **10.61×** |

Speedup column uses Run 1 latencies (the two runs agree within ~1% — see raw
dumps below).

### Run 1 — `block_shape=[128, 128]`

```
==========================================================================================
  KERNEL=block_sparse_attention  DTYPE=torch.bfloat16  INPUT_LAYOUT='BNSD'
  BLOCK_SIZE_Q=128  BLOCK_SIZE_KV=128
  FrameWidths(left_cols=29, right_cols=1, top_rows=29, bottom_rows=1)
  TORCH_REFERENCE='npu_fusion_attention'
==========================================================================================
  H   B    s_q   s_kv    D  sparsity   Outputs_equal Ref_Latency_[usec] Our_Latency_[usec]
--------------------------------------------------------------------------------------------
  3   1 118806 118806  128      0.00             yes          157260.96          149960.13
  3   1 118806 118806  128      0.05             N/A                N/A          144000.87
  3   1 118806 118806  128      0.10             N/A                N/A          136601.68
  3   1 118806 118806  128      0.20             N/A                N/A          121842.59
  3   1 118806 118806  128      0.30             N/A                N/A          107139.64
  3   1 118806 118806  128      0.40             N/A                N/A           92073.33
  3   1 118806 118806  128      0.50             N/A                N/A           77249.27
  3   1 118806 118806  128      0.60             N/A                N/A           61087.95
  3   1 118806 118806  128      0.70             N/A                N/A           45480.47
  3   1 118806 118806  128      0.80             N/A                N/A           29936.18
  3   1 118806 118806  128      0.90             N/A                N/A           14822.55
============================================================================================
```

### Run 2 — `block_shape=[128, 512]` (coarser KV granularity, token-matched frame)

KV block size = 512 means each frame "column" covers 4× more KV tokens than at
128×128. To keep the LEFT-frame coverage roughly token-matched
(`29*128 = 3712` vs `8*512 = 4096`), the frame is set to
`FrameWidths(8, 1, 29, 1)` for this run via the `BSA_FRAME` env var.
(`top_rows`/`bottom_rows` are Q-axis blocks at fixed 128, so they don't
scale.) This also brings the minimum forced density down enough to make
sparsity 0.9 feasible (was infeasible with `FrameWidths(29,1,29,1)` at
this granularity).

```
==========================================================================================
  KERNEL=block_sparse_attention  DTYPE=torch.bfloat16  INPUT_LAYOUT='BNSD'
  BLOCK_SIZE_Q=128  BLOCK_SIZE_KV=512
  FrameWidths(left_cols=29, right_cols=1, top_rows=29, bottom_rows=1)
  TORCH_REFERENCE='npu_fusion_attention'
==========================================================================================
  H   B    s_q   s_kv    D  sparsity   Outputs_equal Ref_Latency_[usec] Our_Latency_[usec]
--------------------------------------------------------------------------------------------
  3   1 118806 118806  128      0.00             yes          157377.81          150151.43
  3   1 118806 118806  128      0.05             N/A                N/A          143405.07
  3   1 118806 118806  128      0.10             N/A                N/A          136249.34
  3   1 118806 118806  128      0.20             N/A                N/A          121633.67
  3   1 118806 118806  128      0.30             N/A                N/A          106927.53
  3   1 118806 118806  128      0.40             N/A                N/A           92043.54
  3   1 118806 118806  128      0.50             N/A                N/A           77262.93
  3   1 118806 118806  128      0.60             N/A                N/A           61111.86
  3   1 118806 118806  128      0.70             N/A                N/A           45625.84
  3   1 118806 118806  128      0.80             N/A                N/A           29911.24
  3   1 118806 118806  128      0.90             N/A                N/A           14681.41
============================================================================================
```

## File layout (autonomous, no cross-tree imports)

| File | Purpose |
|---|---|
| `benchmark.py` | The benchmark entry point. Builds the per-row sparse pattern, converts sabi → int8 block mask, times `cann_ops_transformer.ops.block_sparse_attention(…)` over the sparsity sweep, prints the table. |
| `util.py` | All shared helpers — `FrameWidths`, constants (`B_VALS, H_VALS, S_VALS, D_VALS, SPARSITY_VALS, BLOCK_SIZE_Q, BLOCK_SIZE_KV, FRAME, …`), pattern generators (`generate_sparse_blocks_by_row{,_per_head}`, `make_block_mask{,_per_head}`), reference implementations (`ref_attention_fp32`, `ref_attention_launcher`), timing harness (`_run_timed`, `_check_correctness`), and the sabi→int8 mask converter (`sabi_to_block_sparse_mask`). |
| `README.md` | This file. |

The `tests/pytest/` sibling directory imports from this `util.py` via a
local `sys.path.insert(0, ../../benchmark)` — that's the only intra-package
path manipulation.

## Environment variables

- `ASCEND_RT_VISIBLE_DEVICES` — pick the NPU (commonly `5` on this machine).
- `BSA_BLOCK_SIZE_KV` — override `BLOCK_SIZE_KV` (default `128`). Affects
  both the mask shape passed to the kernel and the `block_shape` value.
  `BLOCK_SIZE_Q` is currently NOT overridable (the mask generator hardcodes
  it to 128).
- `BSA_FRAME` — override `FrameWidths` as `left,right,top,bottom` (block
  units, e.g. `BSA_FRAME=8,1,29,1`). Default is `FrameWidths(29,1,29,1)`.
  Useful for keeping token-coverage matched across different `BSA_BLOCK_SIZE_KV`
  runs, and for avoiding infeasibility at high sparsity + coarse KV granularity.

