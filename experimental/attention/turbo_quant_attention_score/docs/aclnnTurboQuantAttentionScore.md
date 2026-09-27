# aclnnTurboQuantAttentionScore

## 产品支持情况

| 产品 | 支持情况 |
|---|---|
| Atlas 800T A2 | √ |

## 功能说明

- **接口功能**：直接基于 TurboQuant 编码（mse_bits 主量化码 + 1-bit QJL 残差符号 + 范数/残差范数标量）的 Paged Key cache 计算 attention score，不完整反量化 Key 向量。输出 fp32 score，超出各请求 seq_len 的位置为 -inf。
- **计算公式**（对每个 q_token、q_head，kv_head k = q_head / group）：

```
score[i] = x_norm[i] * <query_rotated, centroid[unpack(idx[i])]>
         + (sqrt(pi/2) / 128) * gamma[i] * <query_qjl, unpack_sign(qjl[i])>
score[i >= seq_lens[b]] = -inf
```

- bit 解包格式（与编码侧 KvCacheTurboQuant / golden 逐字节一致）：
  - idx：每 8 维一组打包为 8×mse_bits 位小端字，拆 mse_bits 个字节；128 维共 128·mse_bits/8 字节；
  - qjl：每 8 维 1 字节，bit→±1。

## 函数原型

每个算子分为两段式接口，两段之间通过 `workspace` 和 `executor` 传递信息：

```
aclnnStatus aclnnTurboQuantAttentionScoreGetWorkspaceSize(
    const aclTensor *queryRotated,
    const aclTensor *queryQjl,
    const aclTensor *keyCacheIdx,
    const aclTensor *keyCacheQjl,
    const aclTensor *keyCacheNorm,
    const aclTensor *keyCacheGamma,
    const aclTensor *blockTable,
    const aclTensor *seqLens,
    int64_t mseBits,
    const aclTensor *attnScores,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)

aclnnStatus aclnnTurboQuantAttentionScore(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnTurboQuantAttentionScoreGetWorkspaceSize

- **参数说明**：

| 参数 | 维度 | 数据类型 | 输入/输出 | 说明 |
|---|---|---|---|---|
| queryRotated | 3 | bfloat16 | 输入 | [num_q_tokens, num_q_heads, 128]，ND 连续 |
| queryQjl | 3 | bfloat16 | 输入 | [num_q_tokens, num_q_heads, 128]，ND 连续 |
| keyCacheIdx | 4 | uint8 | 输入 | [num_blocks, 128, num_kv_heads, 128·mseBits/8] |
| keyCacheQjl | 4 | uint8 | 输入 | [num_blocks, 128, num_kv_heads, 16] |
| keyCacheNorm | 3 | bfloat16 | 输入 | [num_blocks, 128, num_kv_heads] |
| keyCacheGamma | 3 | bfloat16 | 输入 | [num_blocks, 128, num_kv_heads] |
| blockTable | 2 | int32 | 输入 | [max_batch, max_blocks_per_seq] |
| seqLens | 1 | int32 | 输入 | [max_batch] |
| mseBits | - | int64_t | 输入属性 | 主量化位宽，仅支持 2/3/4 |
| attnScores | 3 | float32 | 输出 | [num_q_tokens, num_q_heads, max_blocks_per_seq·128] |
| workspaceSize | - | uint64_t | 输出 | 需要用户申请的 workspace 大小 |
| executor | - | aclOpExecutor* | 输出 | executor 指针 |

- **返回值**：`ACLNN_SUCCESS` / `ACLNN_ERR_PARAM_NULLPTR` / `ACLNN_ERR_PARAM_INVALID` 等，见 aclnn 错误码说明。

## aclnnTurboQuantAttentionScore

- **参数说明**：

| 参数 | 说明 |
|---|---|
| workspace | GetWorkspaceSize 申请的内存地址 |
| workspaceSize | GetWorkspaceSize 返回的大小 |
| executor | GetWorkspaceSize 返回的 executor |
| stream | 任务执行流 |

- **返回值**：`ACLNN_SUCCESS` / `ACLNN_ERR_INTERNAL_ERROR`。

## 约束说明

- head_dim / qjl_dim 固定 128，block_size 固定 128；num_kv_heads ≤ 16，GQA group ≤ 8；
- batch > 1 时按 decode 语义（num_q_tokens = batch）；
- block_table 总条目数 ≤ 8192；
- 仅支持 ND 连续 tensor；不支持 MLA；输出不含 softmax 与 V 聚合。

## 调用示例

```cpp
// 1. 申请 workspace 并获取 executor
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnTurboQuantAttentionScoreGetWorkspaceSize(queryRotated, queryQjl, keyCacheIdx, keyCacheQjl,
                                              keyCacheNorm, keyCacheGamma, blockTable, seqLens, 3,
                                              attnScores, &workspaceSize, &executor);
// 2. 按 workspaceSize 申请设备内存
void *workspace = nullptr;
aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL);
// 3. 下发执行
aclnnTurboQuantAttentionScore(workspace, workspaceSize, executor, stream);
```
