# A5 SparseFlashAttention Split-KV decode

## Scope

The Ascend 950 kernel previously assigned a complete sparse-index row to one
query task. For a small total query count, a 2048-entry row leaves most AICs idle.
This path partitions each row across AICs inside the existing SFA launch. It
does not change the operator signature, duplicate queries or indices, launch
an additional merge kernel, or enable an unimplemented FlashDecode tiling key.

Host tiling selects this path only on A5 for matching FP16 or BF16 inputs and output, TND query,
PA_BSND KV, one KV head, eight query heads, 512-dimensional values, 64-dimensional
RoPE, sparse block size 1, 256 to 2048 sparse indices, KV sequence length at
least the index count, sparse mode 0 or 3, no sinks, and no requested softmax LSE.
Mode 3 additionally requires each query's causal window to cover the entire
index row; otherwise that query uses the existing unsplit path. Single- and
multi-batch inputs share the same partition selection.
The existing OpDef signature enforces matching Q/K/V/output dtypes, and common
tiling validation enforces A5 sparse block size 1. The Split-KV selector does
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
identity `w_i = sum_i * exp(max_i - max(max_j))`, then writes the final FP16 or BF16
output. The merge workspace size accounts for both AIV subblocks per AIC.

## Verification

See [A5 validation results](sfa_split_kv_arch35_validation.md) for the
single-operator B-by-Q grid and the A5 64K end-to-end measurements.
This change does not modify arch22 kernels, C8 operators, or A2 LSE support.

Build: `bash build.sh --pkg --soc=ascend950 --ops=sparse_flash_attention`
with an isolated vendor name and install prefix; no system CANN files were
modified. Host tiling UT:
`bash build.sh --ophost_test --ops=sparse_flash_attention --soc=ascend950`.
Device regression: `pytest -q attention/sparse_flash_attention/tests/pytest/test_sfa_split_kv_arch35.py`
with the isolated package selected.
The test covers
both FP16 and BF16, single- and multi-batch TND input, various total Q counts and TopK values,
full index rows, trailing `-1` padding, mixed rows, eager execution and NPU
graph replay. Full rows are compared with an independent PyTorch softmax
reference (relative L2 < 0.02 for BF16 and < 0.005 for FP16). Mixed Q/KV
dtype combinations must be rejected by the existing OpDef signature check.
Single-batch padded rows are finite-output
smoke checks; for multi-batch mixed input only the full-index row is checked.
The installed baseline operator itself differs from that naive reference for
padded rows and may produce non-finite values for multi-batch padded rows, so
no padded-row numerical equivalence is claimed here.

`tests/pytest/bench_sfa_split_kv_arch35.py` provides a BF16 NPU-graph timing
entry point. The historical grid uses the minimum of 102 event-timed samples
per shape on a shared device; it is not a median-latency table.
