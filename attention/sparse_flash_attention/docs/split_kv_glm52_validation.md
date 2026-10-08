# GLM-5.2 sparse-attention partitioning validation

## Kernel scope

- Ascend 950 (A5): G8, BF16, TND query / PA_BSND KV, sparse mode 0 or 3,
  no sinks or requested LSE. A mode-3 query only partitions its KV row when
  the causal window covers the row; short or padded rows retain the original
  computation. Multi-batch selection is no longer restricted to TopK=2048.
- Ascend 910B (A2), ordinary SFA: G8/G64, BF16, TND / PA_BSND, sparse mode 0,
  with optional softmax max/sum output. Partial output/statistics are merged
  using stable softmax weights in user workspace.
- Ascend 910B (A2), packed C8 V2: G8/G64, BF16 query/output, int8 KV with
  per-tile quantization and tile size 128. G64 calls requesting LSE partition
  heads, not short DCP KV shards; each head writes its own output/statistics.
  Valid-prefix / -1-suffix rows are bounded by their valid length. An empty
  local shard writes zero output and sum, yielding negative-infinite LSE.

The tilers separate kernel capability checks from partition-count selection.
Unsupported configurations continue to use the original operator path.
The A5 path does not gain G64 or LSE support merely because A2 supports them.
The V1 single-output C8 ABI and V2 output/statistics ABI remain distinct.

GLM-5.2 has 64 attention heads and 32 indexer heads. The A5 TP8/no-DCP model
uses local G8 SFA calls. A2 TP8/DCP8 decode gathers queries and uses actual
G64 C8/LSE calls; the latent KV cache has one head. These are full-model
measurements, not a 32-head indexer benchmark.

## Historical paired AISBench results (2026-10-06)

Each platform/context has one warmup request per version followed by ten
serial successful requests per version, each producing 1024 output tokens.
Both arms use the same prompts, weights and serving configuration for that
platform/context. Temperature is zero, EOS is ignored, and prefix caching is
disabled. The context sets are natural-text prompts from the existing
successful benchmark configuration; the 128K prompts preserve the body ends
and full question from the 256K set. Actual chat input lengths are 131084
tokens for 128K and 254595--262081 for 256K.

Configuration:

- A5 64K: TP8, DCP1, DSpark with 7 speculative tokens, max sequences 1,
  max model length 73728, batched-token limit 8192, memory utilization 0.95,
  FULL_DECODE_ONLY graphs, SFA C8 / indexer C8 / Pivot / DSA CP disabled.
- A2 128K and 256K: TP8, DCP8, MTP with 3 speculative tokens, int8 KV cache,
  SFA C8 enabled and indexer C8 / Pivot / DSA CP disabled, max sequences 4,
  max model length 268288, batched-token limit 8192, memory utilization 0.92,
  CP interleave 128, FULL_DECODE_ONLY graphs, expert parallelism,
  Flashcomm1 and fused MC2 enabled. Setting only the SFA C8 additional-config
  flag is insufficient: cache dtype must also be int8 to exercise C8.

Arithmetic means, milliseconds; positive reduction means lower latency:

| Platform/context | Metric | Baseline | Candidate | Reduction |
| --- | --- | ---: | ---: | ---: |
| A5 64K | TTFT | 16811.4100 | 16905.2367 | -0.5581% |
| A5 64K | TPOT | 14.0847 | 12.4244 | 11.7879% |
| A5 64K | PIL | 37.8170 | 34.3151 | 9.2602% |
| A2 128K | TTFT | 63251.5795 | 64297.7660 | -1.6540% |
| A2 128K | TPOT | 34.0203 | 29.5747 | 13.0674% |
| A2 128K | PIL | 106.8301 | 92.0726 | 13.8140% |
| A2 256K | TTFT | 155710.4327 | 157817.1658 | -1.3530% |
| A2 256K | TPOT | 35.1042 | 30.3121 | 13.6509% |
| A2 256K | PIL | 108.7180 | 94.0873 | 13.4574% |

TTFT/PIL/TPOT are derived from AISBench raw time points. TPOT is
`(last_token_time - first_token_time) / 1023`; PIL is the average streaming
chunk interval after the first token, not the same quantity as TPOT with
speculative decoding. Accepted/drafted counter deltas give MTP acceptance
rates of 23.2152% / 24.3453% for A5, 71.8763% / 70.8791% for A2 128K and
70.1212% / 70.3906% for A2 256K (baseline / candidate).

These results show TPOT/PIL improvements and small TTFT regressions. They
are comparisons of the recorded package versions, not a repeat against
latest master, and predate the master synchronization and eligibility
refactoring in this PR. Do not attribute these percentages to the earlier
A5 mode-0-only PR, or claim that the current merged source has been
end-to-end remeasured against latest master. Prompt-pair variation and
speculative acceptance also do not establish cross-run statistical stability.

The measured A2 runtime used a native V2 bridge because the old runtime's
historical three-output C8 V1 call does not match this repository's V1 ABI.
Its C++ wrapper allocated statistics with `empty`, matching the old runtime;
the repository V2 wrapper retains full/zero allocation. Both baseline and
candidate performed the same query concatenation. Use the same bridge and
allocation policy when reproducing the recorded comparison.

Candidate package SHA256 identifiers:

- A2 complete SFA / C8 V1 / C8 V2 package:
  `f44d7769f0b3011ed7f4eb890fcbbb598c9e56c420ebe12824ac8e26ae5730d8`.
- The previously verified cumulative source patch, based on `c9561d1d2`,
  before master synchronization/refactoring:
  `45212d0e29ed008136322242f8337c3516aaa4c553797b6c9ac6469eafd349db`.

Do not deploy a V2-only operator package for this model: legacy SFA calls
also require matching host tiling and kernel metadata for the other operators.

## Correctness coverage

The pre-synchronization candidate passed A5 mode0/3 numerical/graph tests
(115 passed, 4 skipped, 1 expected failure for an existing baseline padding
issue), A2 C8 numerical/LSE tests with uninitialized output/statistics
(210 passed, 30 skipped), and ordinary A2 SFA G8/G64 tests, including the
final UB-scratch G64 reduction implementation. The C8 reference independently
dequantizes KV and computes softmax on CPU; coverage includes multi-batch,
padding, empty local shards, disjoint-shard merges and graph replay.

Host regression commands:

```bash
bash build.sh --ophost_test --soc=ascend950 --ops=sparse_flash_attention
bash build.sh --ophost_test --soc=ascend910b \
    --ops=sparse_flash_attention,kv_quant_sparse_flash_attention,kv_quant_sparse_flash_attention_v2
```

Install a complete isolated vendor package for the selected architecture,
then run the SFA arch35 / SFA arch22 / C8 V2 arch22 pytest suites. Do not
replace system CANN packages or alter unrelated running services.
