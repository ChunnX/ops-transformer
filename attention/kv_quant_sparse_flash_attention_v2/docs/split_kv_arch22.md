# Ascend 910B G8/G64 C8 partitioning

The arch22 implementation supports the V2 output/LSE interface for TND query,
PA_BSND KV, one KV head, 8 or 64 query heads, and no sinks. The existing V1
single-output ABI remains separate. A caller requiring DCP statistics must
use V2 rather than passing three outputs to the V1 ACLNN interface.

Partitioning applies to BF16 query/output, packed int8 C8 KV (656 bytes per
token), per-tile quantization, tile size 128, sparse block size 1, sparse mode
0, and TopK 1024 through 2048. Other supported shapes use the original compute
path. The tiler obtains the available AIC count from the platform.

For G8, and G64 without requested LSE, a query's KV row is divided into up to
16 power-of-two partitions. Each partition contains at least one 512-index
inner block. The kernel stages partial output, max and sum in user workspace,
synchronizes active cores, and combines the partial outputs with stable
softmax weights. The staging offset excludes the CANN system workspace already
skipped by GetUserWorkspace. Rows with incomplete/padded index lists retain
the original unsplit row computation.

For G64 with requested LSE, DCP commonly supplies a short local KV prefix,
followed by -1 padding up to TopK=2048. The tiler partitions the 64 heads
instead: each head has one owner, so output and LSE require no cross-core
reduction. The number of head groups is the largest power of two at most
min(8, AICs / totalQueryTokens). Cumulative query lengths map each global query
to its own batch, page table and KV length. All query/output/LSE offsets retain
the original 64-head stride.

The documented valid-prefix/-1-suffix index contract permits binary search
for the local valid length. G64 LSE calls bound inner loops by this prefix,
including when the query count is too large for head partitioning. An empty
local shard writes zero output, zero softmax sum and a finite negative softmax
max; max + log(sum) is negative infinity. Sequential initialization of these
buffers drains DMA before reusing scratch memory.

G64 KV reduction processes at most 16 heads at a time. After the compute
pipeline drains, the upper 32 KiB of the input buffer holds merge statistics,
while its lower region holds partial output. This avoids allocating another
12 KiB of UB, which would exceed the arch22 C8 kernel's 192 KiB UB budget.

Host verification:

```bash
bash build.sh --ophost_test --soc=ascend910b \
    --ops=sparse_flash_attention,kv_quant_sparse_flash_attention,kv_quant_sparse_flash_attention_v2
```

Build and install an isolated vendor package, then run
tests/pytest/test_split_kv_arch22.py. The device regression compares output and
LSE with independent CPU dequantization/softmax, including G8/G64, multiple
batches, TopK fallback, padding, empty local shards, disjoint DCP shard merging,
eager execution and NPU graph replay. The empty-shard producer is additionally
verified with uninitialized output/LSE allocations matching the vLLM wrapper.

Operator graph timings measure this kernel in isolation. End-to-end TTFT,
PIL and TPOT require a separate paired model benchmark with the same weights,
datasets, TP/DCP settings and speculative decoding configuration.
See [GLM-5.2 validation](../../sparse_flash_attention/docs/split_kv_glm52_validation.md)
for the completed paired model measurements and their version limitations.
