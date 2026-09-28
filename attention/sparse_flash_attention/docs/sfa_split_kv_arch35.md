# A5 SparseFlashAttention Split-KV decode

## Scope

The Ascend 950 kernel previously assigned a complete sparse-index row to one
query task. For B=1 and Q=1 or Q=4, a 2048-entry row leaves most AICs idle.
This path partitions each row across AICs inside the existing SFA launch. It
does not change the operator signature, duplicate queries or indices, launch
an additional merge kernel, or enable an unimplemented FlashDecode tiling key.

Host tiling selects this path only for BF16 output and inputs, TND query,
PA_BSND KV, one KV head, eight query heads, 512-dimensional values, 64-dimensional
RoPE, sparse block size 1, exactly 2048 sparse indices, KV sequence length at
least 2048, sparse mode 0, no sinks, and no requested softmax LSE. Other shapes
retain their previous tiling. Each row whose last index is `-1` falls back at
runtime to the original unsplit row path; this preserves the existing handling
of trailing padding. Interior invalid indices are not newly supported.

The tiler reads the available AIC count from the platform. It assigns the
largest power-of-two partition count no greater than `min(16, AICs / Q)`.
The cap of 16 matches the reduction UB and keeps each partition at least 128
indices long. For example, 8 AICs with Q=1 yield 8 partitions, while 24 AICs
with Q=4 yield 4. The 36-AIC benchmark device selects 16 for Q=1 and 8 for
Q=4. With fewer than two partitions, the original path remains selected.

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
on Ascend950DT_9592 with the isolated package selected. The test covers
Q=1/4, full 2048 entries, trailing `-1` padding, mixed Q=4 rows, eager and
NPU graph replay. Full rows are compared with an independent PyTorch softmax
reference (relative L2 < 0.02); padded rows are finite-output smoke checks.
The installed baseline operator itself differs from that naive reference for
padded rows, so no padded-row numerical equivalence is claimed here.

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
