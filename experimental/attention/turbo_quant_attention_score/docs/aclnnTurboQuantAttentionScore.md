# aclnnTurboQuantAttentionScore

## 功能说明

基于 TurboQuant 编码的 Paged Key Cache 计算 attention score：

```text
score = x_norm * <Hq, dequant(idx)>
      + (sqrt(pi/2)/D) * gamma * <S(Hq), qjl_signs>
```

## 接口原型

```cpp
aclnnStatus aclnnTurboQuantAttentionScoreGetWorkspaceSize(
    const aclTensor* queryRotated,
    const aclTensor* queryQjl,
    const aclTensor* keyCacheIdx,
    const aclTensor* keyCacheQjl,
    const aclTensor* keyCacheNorm,
    const aclTensor* keyCacheGamma,
    const aclTensor* blockTable,
    const aclTensor* seqLens,
    int64_t mseBits,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

aclnnStatus aclnnTurboQuantAttentionScore(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

参数顺序为输入 → 属性 → 输出。当前用户 workspace 为 0，但调用方仍应使用
第一段接口返回的 `workspaceSize`。

## 参数

| 参数 | 说明 |
|---|---|
| queryRotated | `[T,H,D]`，BFLOAT16 |
| queryQjl | `[T,H,D]`，BFLOAT16 |
| keyCacheIdx | `[num_blocks,128,KvH,idxBytes]`，UINT8 |
| keyCacheQjl | `[num_blocks,128,KvH,qjlBytes]`，UINT8 |
| keyCacheNorm | `[num_blocks,128,KvH]`，BFLOAT16 / FLOAT16 |
| keyCacheGamma | `[num_blocks,128,KvH]`，BFLOAT16 / FLOAT16 |
| blockTable | `[B,maxBlocks]`，INT32 |
| seqLens | `[B]`，INT32 |
| mseBits | `{2,3,4}`，默认 3 |
| out | `[T,H,maxKvLen]`，FLOAT32 |

## 调用样例

参见
[`examples/test_aclnn_turbo_quant_attention_score.cpp`](../examples/test_aclnn_turbo_quant_attention_score.cpp)。
