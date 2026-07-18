# ScaledCosineAttentionScore

## Product Support

| Product | Supported |
|---|:---:|
| Ascend 910B | ? |
| Ascend 910_93 | ? |

## Function Description

`ScaledCosineAttentionScore` fuses L2 statistics along head dimension, batch cosine scoring and per-head logit scaling into one operator, outputting attention scores before softmax:

```text
inv_q[b,h,i] = 1 / sqrt(sum_t(query[b,h,i,t]^2) + eps)
inv_k[b,h,j] = 1 / sqrt(sum_t(key[b,h,j,t]^2) + eps)
score[b,h,i,j] = sum_t(query[b,h,i,t] * key[b,h,j,t])
                   * inv_q[b,h,i] * inv_k[b,h,j]
                   * exp(min(scale[h], clamp_max))
```

Square sum, dot product, sqrt, division and scale are computed in FLOAT32; result is cast back to query dtype.
No intermediate normalized tensors are created, no user workspace is allocated, and key is not explicitly transposed.

## Parameter Description

| Parameter | Input/Output/Attr | shape | Data Type | Format |
|---|---|---|---|---|
| query | Input | `[B,H,N,d]` | FLOAT16, BFLOAT16, FLOAT32 | ND |
| key | Input | `[B,H,N,d]`, same shape/dtype as query | same as query | ND |
| scale | Input | `[H]` or `[H,1,1]` | FLOAT32 | ND |
| clamp_max | Attr | scalar, default `4.6052` | FLOAT | - |
| eps | Attr | positive scalar, default `1e-12` | FLOAT | - |
| attn_score | Output | `[B,H,N,N]` | same as query | ND |

## Constraints

- query/key rank must be 4, and shapes must be identical; all dimensions must be positive at runtime.
- scale must be FLOAT32, shape `[H]` or `[H,1,1]`.
- `clamp_max` must be finite; `eps` must be finite and positive.
- Currently registered only for `ascend910b` and `ascend910_93`.
- Current kernel is a Vector Core fusion baseline; final throughput needs to be measured against combined ops on target NPU.
- This operator uses `sqrt(sum(x^2) + eps)`, do not confuse with `max(norm(x), eps)`.

## ops-transformer Integration

This directory is a single-operator module within the ops-transformer repository, not a standalone CMake project.
Main files:
```text
??? CMakeLists.txt
??? docs/aclnnScaledCosineAttentionScore.md
??? examples/test_aclnn_scaled_cosine_attention_score.cpp
??? op_host
?   ??? config/ascend910b/scaled_cosine_attention_score_binary.json
?   ??? scaled_cosine_attention_score_def.cpp
?   ??? scaled_cosine_attention_score_infershape.cpp
?   ??? scaled_cosine_attention_score_tiling.cpp
?   ??? scaled_cosine_attention_score_tiling.h
??? op_kernel
?   ??? scaled_cosine_attention_score.cpp
?   ??? scaled_cosine_attention_score_impl.hpp
?   ??? scaled_cosine_attention_score_tiling_def.h
??? tests/ut
```

For calling instructions, see [`docs/aclnnScaledCosineAttentionScore.md`](docs/aclnnScaledCosineAttentionScore.md).
Algorithm and tiling details see [`docs/design.md`](docs/design.md).
