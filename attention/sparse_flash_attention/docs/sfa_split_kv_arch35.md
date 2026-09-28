# A5 SparseFlashAttention Split-KV decode

## Scope

The Ascend 950 kernel previously assigned a complete sparse-index row to one
query task. For a small total query count, a 2048-entry row leaves most AICs idle.
This path partitions each row across AICs inside the existing SFA launch. It
does not change the operator signature, duplicate queries or indices, launch
an additional merge kernel, or enable an unimplemented FlashDecode tiling key.

Host tiling selects this path only on A5 for BF16 output and inputs, TND query,
PA_BSND KV, one KV head, eight query heads, 512-dimensional values, 64-dimensional
RoPE, sparse block size 1, 256 to 2048 sparse indices, KV sequence length at
least the index count, sparse mode 0, no sinks, and no requested softmax LSE.
Multi-batch splitting is initially restricted to TopK=2048; smaller TopK
multi-batch inputs continue through the original SFA path.
For TND, `Q` below means total query tokens across all batches. The kernel
uses `actual_seq_lengths_query` (cumulative per-batch lengths) to map each
global query token back to its batch and batch-local position. This keeps the
batch-specific KV length and page table, while partial results are indexed by
the global query token.

Other shapes retain the existing unsplit SFA tiling and kernel, not dense
attention. The same original path is selected when there are too many queries
for at least two partitions, including large multi-batch workloads. Each row
whose last index is `-1` uses the original unsplit row computation at runtime;
this preserves the existing handling of trailing padding. Interior invalid
indices are not newly supported.

The tiler reads the available AIC count from the platform. It assigns the
largest power-of-two partition count no greater than
`min(16, floor(AICs / Q), ceil(TopK / 128))`. The cap of 16 matches the
reduction UB capacity, and each partition receives at least one 128-index
block. For example, 24 AICs with B=2 and two queries per batch (Q=4 total)
yield four partitions per query and 16 tasks. The 36-AIC benchmark device
selects 16 for Q=1 and 8 for Q=4 at TopK=2048. With fewer than two partitions,
the original path remains selected.

Each AIC processes one `(query, partition)` pair with the existing KV gather,
QK/softmax/PV pipeline. Its two AIV subblocks stage FP32 partial output and
local softmax max/sum in a dedicated workspace. After cross-core synchronization,
the first partition merges the statistics using the stable online-softmax
identity `w_i = sum_i * exp(max_i - max(max_j))`, then writes the final BF16
output. The merge workspace size accounts for both AIV subblocks per AIC.

## Verification

Build: `bash build.sh --pkg --soc=ascend950 --ops=sparse_flash_attention`
with an isolated vendor name and install prefix; no system CANN files were
modified. Host tiling UT:
`bash build.sh --ophost_test --ops=sparse_flash_attention --soc=ascend950`.
Device regression: `pytest -q attention/sparse_flash_attention/tests/pytest/test_sfa_split_kv_arch35.py`
on Ascend950DT_9582 with the isolated package selected: 52 passed, 2 skipped.
The test covers
single- and multi-batch TND input, various total Q counts and TopK values,
full index rows, trailing `-1` padding, mixed rows, eager execution and NPU
graph replay. Full rows are compared with an independent PyTorch softmax
reference (relative L2 < 0.02). Single-batch padded rows are finite-output
smoke checks; for multi-batch mixed input only the full-index row is checked.
The installed baseline operator itself differs from that naive reference for
padded rows and may produce non-finite values for multi-batch padded rows, so
no padded-row numerical equivalence is claimed here.

Single-op timing on Ascend950DT_9592: B=1, N2=1, G=8, BF16, K=2048,
PA_BSND, physical KV length 2048, page size 128, no indexer or remap.
The same benchmark uses an NPU graph with 16 SFA calls, warmup, 11 event-timed
repeats; values below are per call in microseconds.

| Q | Original SFA median / p95 | Split-KV median / p95 | Speedup | Reference relative L2 |
| - | ------------------------- | -------------------------- | ------- | --------------------- |
| 1 | 75.36 / 76.09 | 17.20 / 17.90 | 4.38x | 0.00192 |
| 4 | 75.70 / 75.95 | 19.43 / 19.57 | 3.90x | 0.00188 |

Profiler confirms 16 MIX_AIC blocks (32 MIX_AIV) for Q=1 and 32 MIX_AIC
blocks (64 MIX_AIV) for Q=4. Measurements are specific to this device, shape,
package, and graph benchmark. They do not imply an end-to-end inference gain.

With physical KV length 4096 on Ascend950DT_9582, the same benchmark uses
16 graph calls and 11 event-timed repeats. The table compares the system SFA
with this change; Q is the total number of query tokens across B batches.

| B | Q | Original median (us) | Split-KV median (us) | Speedup |
| - | - | -------------------- | -------------------- | ------- |
| 1 | 1 | 74.55 | 18.07 | 4.13x |
| 1 | 4 | 74.64 | 20.16 | 3.70x |
| 2 | 4 | 77.98 | 20.49 | 3.80x |

On the same device and benchmark, the previous single-batch split-KV package
measured 17.50 us for Q=1 and 19.56 us for Q=4. The multi-batch mapping
adds about 3% to these two single-batch measurements.
