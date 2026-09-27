# TurboQuantAttentionScore

## 产品支持情况

| 产品 | 支持情况 |
|---|---|
| Atlas 800T A2 | √ |

## 功能说明

TurboQuant 解码侧 attention score 算子：在标准 MHA/GQA attention 中，直接基于 TurboQuant 编码的 Paged Key cache 计算 attention score——不完整反量化 Key 向量到原空间，而是在旋转空间中按需解码主量化分支（查表反量化后点积），并叠加 QJL 残差修正项。配套编码侧算子为 KvCacheTurboQuant。

对每个 query token 的每个 q_head，与所有 cached key 计算：

```
score[i] = score_mse[i] + score_qjl[i]
score_mse[i] = x_norm[i] * <Hq, centroid[idx[i]]>
score_qjl[i] = (sqrt(pi/2) / qjl_dim) * gamma[i] * <S(Hq), sign(qjl[i])>
```

其中 `Hq = H @ q`、`S(Hq) = S @ Hq` 由调用方预计算（作为输入 query_rotated / query_qjl 给出）；`idx` 为 mse_bits 位宽的主量化码（bit-packed，8 维/组、小端字节序），`qjl` 为 1-bit 残差符号。超出各请求 seq_len 的位置输出 -inf。

本算子只计算 score，不包含 softmax 与 Value cache 的解码/聚合；不支持 MLA。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 | shape |
|---|---|---|---|---|---|
| query_rotated | 输入 | 预旋转后的 query（Hq） | bfloat16 | ND | [num_q_tokens, num_q_heads, 128] |
| query_qjl | 输入 | 旋转后 query 的 QJL 投影（S·Hq） | bfloat16 | ND | [num_q_tokens, num_q_heads, 128] |
| key_cache_idx | 输入 | Key cache 主量化编码（bit-packed） | uint8 | ND | [num_blocks, 128, num_kv_heads, 128·mse_bits/8] |
| key_cache_qjl | 输入 | Key cache 残差 QJL 符号编码（bit-packed） | uint8 | ND | [num_blocks, 128, num_kv_heads, 16] |
| key_cache_norm | 输入 | Key 向量 L2 范数 | bfloat16 | ND | [num_blocks, 128, num_kv_heads] |
| key_cache_gamma | 输入 | Key 原尺度残差范数 | bfloat16 | ND | [num_blocks, 128, num_kv_heads] |
| block_table | 输入 | Paged block 映射 | int32 | ND | [max_batch, max_blocks_per_seq] |
| seq_lens | 输入 | 各请求 KV 序列长度 | int32 | ND | [max_batch] |
| attn_scores | 输出 | Attention 分数（无效位置为 -inf） | float32 | ND | [num_q_tokens, num_q_heads, max_blocks_per_seq·128] |
| mse_bits | 属性 | MSE 主量化位宽，可选 2/3/4，默认 3 | int | - | - |

## 约束说明

- head_dim 与 qjl_dim 固定 128，block_size 固定 128（与 vllm-ascend 当前配置一致）；
- 仅支持标准 MHA/GQA：num_q_heads / num_kv_heads 为整数，num_kv_heads ≤ 16 且 group 大小 ≤ 8；
- batch > 1 时按 decode 语义处理（num_q_tokens = batch，每个请求 1 个 query token）；
- block_table 总条目数（max_batch × max_blocks_per_seq）≤ 8192；
- 仅支持 ND 连续 tensor；mse_bits 仅支持 2/3/4，其余取值返回参数错误；
- 仅支持静态 shape（-1 等动态维返回参数错误）；key_cache_norm / key_cache_gamma 仅支持 bfloat16（与官方测试件一致）；
- query 旋转与 QJL 投影由调用方预计算，不属于本算子。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| aclnn 调用 | [aclnnTurboQuantAttentionScore](docs/aclnnTurboQuantAttentionScore.md) | 两段式接口 `aclnnTurboQuantAttentionScoreGetWorkspaceSize` / `aclnnTurboQuantAttentionScore` |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
|---|---|---|---|---|
| wisdom_qq | 社区开发者 | TurboQuantAttentionScore | 2026-09 | 算子设计、Ascend C 实现、精度/性能自验 |
