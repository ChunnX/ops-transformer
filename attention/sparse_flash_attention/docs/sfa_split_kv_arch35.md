# A5 SparseFlashAttention Split-KV decode

## Scope

The Ascend 950 kernel previously assigned a complete sparse-index row to one
query task. For a small total query count, a 2048-entry row leaves most AICs idle.
This path partitions each row across AICs inside the existing SFA launch. It
does not change the operator signature, duplicate queries or indices, launch
an additional merge kernel, or change dense attention.

Host tiling selects this path only on A5 for matching FP16 or BF16 inputs and output, TND query,
PA_BSND KV, one KV head, 1–128 query heads, 512-dimensional values, absent or
64-dimensional RoPE, sparse block size 1, 256 to 2048 sparse indices, KV sequence length at
least the index count, sparse mode 0 or 3, and no sinks. Returned softmax max/sum
are supported with and without KV partitioning.
Mode 3 additionally requires each query's causal window to cover the entire
index row; otherwise that query uses the existing unsplit path. Single- and
multi-batch inputs share the same partition selection.
The existing OpDef signature enforces matching Q/K/V/output dtypes, and common
tiling validation enforces A5 sparse block size 1 and, when supplied, RoPE dimension 64.
Both `HAS_ROPE=0` and `HAS_ROPE=1` FD instances are registered. The Split-KV selector does
not duplicate these checks; mixed dtypes remain unsupported by the interface.
For TND, `Q` below means total query tokens across all batches. The kernel
uses `actual_seq_lengths_query` (cumulative per-batch lengths) to map each
global query token back to its batch and batch-local position. This keeps the
batch-specific KV length and page table, while partial results are indexed by
the global query token.

Other shapes retain the existing unsplit SFA tiling and kernel, not dense
attention. The same original path is selected when there are too many queries
for at least two partitions, including large multi-batch workloads. Each row
whose last index is `-1` uses the original unsplit row computation at runtime;
only its valid prefix is passed to QK/softmax/PV. Interior invalid indices
followed by valid indices remain outside the documented input contract.

The tiler reads the available AIC count from the platform. It assigns the
largest power-of-two partition count no greater than
`min(16, floor(AICs / (Q * headParts)), ceil(TopK / 128))`, where `headParts`
is 1 for G ≤ 64 and 2 for G > 64. The cap of 16 matches the
reduction UB capacity, and each partition receives at least one 128-index
block. For example, 24 AICs with B=2 and two queries per batch (Q=4 total)
yield four partitions per query and 16 tasks for G ≤ 64. The 36-AIC benchmark device
selects 16 for Q=1 and 8 for Q=4 at TopK=2048 when G ≤ 64; G > 64 selects
16 and 4, respectively, with two head tiles per partition. With fewer than two partitions,
the original path remains selected.

Each AIC processes one `(query, partition, headTile)` task with the existing KV gather,
QK/softmax/PV pipeline. Its two AIV subblocks stage FP32 partial output and
local softmax max/sum in a dedicated workspace. After cross-core synchronization,
the first partition merges the statistics using the stable online-softmax
identities:

```text
M = max_i(max_i)
w_i = sum_i * exp(max_i - M)
S = sum_i(w_i)
O = sum_i(w_i * O_i) / S
```

It writes the final FP16 or BF16 output and, when requested, FP32 `M`/`S` in
the existing output layout. These statistics reuse the merge's max/sum
calculation; no additional logarithm or kernel launch is needed.

## Generic head tiling

As in dense FlashDecode, workspace strides and reduction row counts use the
actual head tile, and the merge processes bounded chunks of eight rows.
The original A5 SFA head split is retained: G ≤ 64 uses one AIC head tile;
G > 64 uses two tiles of `ceil(G/2)` and `floor(G/2)` heads. Each tile's two
AIVs handle its ceiling/floor halves. Empty halves (G=1) are not staged or
merged. Odd head counts use a uniform padded workspace stride but write
only their valid rows.

Physical AIC task IDs are `((query * partitions + partition) * headParts + headTile)`.
Workspace slots are ordered by `(query, headTile, AIV subblock, partition)`.
This keeps a tile's partition results contiguous for reduction and gives
each output head exactly one final writer.

FD head tiles gather KV into separate AIC scratch regions. They do not use
the original paired-head shared-gather barrier: partitioned and fallback
rows can have different loop counts, so that all-AIV barrier would not be
appropriate here. The original non-FD shared-gather path is unchanged.

## Compile-time FD isolation

Host tiling selects the existing `FLASH_DECODE` template parameter, propagated
as `IS_FD` into arch35 services. `if constexpr (IS_FD)` isolates partition
scheduling, workspace staging, the final synchronization, merge and extra UB
allocation. `FLASH_DECODE=0` instances compile without those operations.
The new template combinations are limited to TND/PA_BSND, with or without RoPE;
all original template combinations retain `FLASH_DECODE=0`.

Within an FD instance, a runtime per-query check still handles trailing
`-1` padding, short KV and short causal rows. Only partition zero computes
such a row, without the final FD merge. These tensor-content decisions
cannot be made by host tiling.

## Trailing sparse padding

The interface requires valid indices before a contiguous `-1` suffix. A padded
row is computed by partition zero without an FD merge. Before scheduling its
QK/softmax/PV loops, the kernel finds the valid prefix length and clamps both
the final tile width and the number of S2 iterations. The index tensor's
original row stride is unchanged. A row with no valid indices writes zero
attention output and zero max/sum, matching the existing golden reference.
The same length handling applies to unsplit arch35 rows without sinks.

This addresses the unwritten-gather-scratch failure described in
[Issue #5208](https://gitcode.com/cann/ops-transformer/issues/5208).
It does not implement interleaved valid/invalid indices, add sinks support to
FD, or change the sparse-mode contract. Full FD rows bypass the prefix search.

## Verification

See [A5 validation results](sfa_split_kv_arch35_validation.md) for the
single-operator B-by-Q grid, one-factor-at-a-time generalization sweeps and
padding regressions.
Build: `bash build.sh --pkg --soc=ascend950 --ops=sparse_flash_attention`
with a separate vendor name and install prefix. Host tiling UT:
`bash build.sh --ophost_test --ops=sparse_flash_attention --soc=ascend950`.
Device regression: `pytest -q attention/sparse_flash_attention/tests/pytest/test_sfa_split_kv_arch35.py`
with the isolated package selected.
The test covers
both FP16 and BF16, every head count 1–128, absent/64-dimensional RoPE, LSE on/off, single- and multi-batch
TND input, various total Q counts and TopK values,
full index rows, trailing `-1` padding, all-empty rows, mixed rows, eager execution
and NPU graph replay. Every row is compared with an independent PyTorch softmax
reference (relative L2 < 0.02 for BF16 and < 0.005 for FP16). Returned FP32
softmax max/sum are checked separately. Mixed Q/KV
dtype combinations must be rejected by the existing OpDef signature check.

`tests/pytest/bench_sfa_split_kv_arch35.py` provides FP16/BF16 NPU-graph timing
with configurable heads, sparse mode, LSE and RoPE.
`tests/pytest/bench_sfa_split_kv_sweeps_arch35.py` adds a B/Q grid and fixed-B/Q
sweeps of every G in [1,128], dtype, mode, LSE, RoPE, TopK, page size and KV length.
Each round contains 32 calls per graph and 101 event-timed replays. The validation
table reports the median of three interleaved rounds' per-call medians against
the target master baseline.
