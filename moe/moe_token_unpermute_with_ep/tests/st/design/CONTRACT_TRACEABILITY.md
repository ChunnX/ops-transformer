# ACLNN test-contract traceability

The executable case generator derives only the constraints below. Paths are relative to the operator directory.

| ID | Contract | Source |
|---|---|---|
| DT-TOKENS | `permutedTokens`: FP16/BF16/FP32, rank 2, ND | `docs/aclnnMoeTokenUnpermuteWithEp.md`; `op_host/moe_token_unpermute_with_ep_def.cpp` |
| DT-INDICES | `sortedIndices`: INT32, rank 1, ND | same sources |
| DT-PROBS | optional `probsOptional`: FP16/BF16/FP32, rank 2, ND; dtype independent of tokens | same sources and nine dtype profiles in `_def.cpp` |
| SH-TOPK | With probs, `sortedIndices.numel() == probs.shape[0] * probs.shape[1]`; non-fallback `numTopk == probs.shape[1]`; TopK ≤ 512 | `op_host/moe_token_unpermute_with_ep_tiling.cpp` input checks |
| SH-NOPROBS | Without probs and with a non-empty EP range, output rows are `sortedIndices.numel()/numTopk`; fallback fixes TopK to one | `op_host/moe_token_unpermute_with_ep_tiling.cpp` `Init` |
| EP-RANGE | Non-empty `[start,end)` uses local token row `index-start`, with outside-range contributions zero; absent/default `[0,0]` is direct-gather fallback | `docs/aclnnMoeTokenUnpermuteWithEp.md`; tiling and kernel implementation |
| ATTR-PAD | `paddedMode` supports only `false` | ACLNN interface document |
| ATTR-RESTORE | `restoreShapeOptional` supports only null/empty | ACLNN interface document |
| CONTIG | Non-contiguous tensor inputs and output are accepted through `AutoContiguous`/ACLNN view descriptors | ACLNN interface document; `_def.cpp` |
| EMPTY | Empty inputs are outside the documented rank/shape contract and yield zero-core/division hazards; L2 expects rejection | ACLNN interface document and tiling `SetCoreNum`/`TilingToken` |

The upstream canndev 200-case JSON is consumed only as a seed corpus. Its legacy non-empty `restore_shape` rows are not treated as positive contract evidence.
