# aclnnQkvRmsNormRopeCacheWithKScale

## 接口原型

```cpp
aclnnStatus aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize(
    const aclTensor *qkv,
    const aclTensor *qGamma,
    const aclTensor *kGamma,
    const aclTensor *cosSin,
    const aclTensor *slotMapping,
    aclTensor *kCache,
    aclTensor *vCache,
    aclTensor *kScaleCache,
    const aclTensor *cuSeqLen,
    const aclTensor *actualSeqLens,
    const aclTensor *qRotation,
    const aclTensor *kRotation,
    const aclTensor *vScale,
    const aclIntArray *headNums,
    const char *inputLayout,
    float epsilon,
    aclTensor *qOut,
    aclTensor *qScale,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnQkvRmsNormRopeCacheWithKScale(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## 功能描述

QkvRmsNormRopeCacheWithKScale 面向 Transformer decode/prefill 流程，融合 Q/K/V 拆分、Q/K RMSNorm、RoPE、Q/K 分支 rotation matmul、FP8 动态量化以及 KV cache 原地写回。

## 参数说明

### 输入参数

| 参数名 | 必选/可选 | 数据类型 | 数据格式 | Shape | 说明 |
|--------|----------|---------|---------|-------|------|
| qkv | 必选 | BFLOAT16 | ND | 见 inputLayout | Q/K/V 融合输入 |
| qGamma | 必选 | FLOAT | ND | [D] | Q 分支 RMSNorm gamma |
| kGamma | 必选 | FLOAT | ND | [D] | K 分支 RMSNorm gamma |
| cosSin | 必选 | FLOAT | ND | [MaxSeqLen, D] | RoPE cos/sin 数据 |
| slotMapping | 必选 | INT32 | ND | [T] | token 到 cache slot 的映射 |
| kCache | 必选 | FLOAT8_E4M3FN | ND | [BlockNum, Nk, BlockSize, D] | K cache，输入输出复用 |
| vCache | 必选 | FLOAT8_E4M3FN | ND | [BlockNum, Nv, BlockSize, D] | V cache，输入输出复用 |
| kScaleCache | 必选 | FLOAT | ND | [BlockNum, Nk, BlockSize, 1] | K scale cache，输入输出复用 |
| cuSeqLen | 必选 | INT32 | ND | [Batch+1] | batch 前缀和 |
| actualSeqLens | 必选 | INT32 | ND | [Batch] | 每个 batch 追加当前 token 后的实际序列长度 |
| qRotation | 可选 | BFLOAT16 | ND | [D, D] | Q 分支 rotation 矩阵，当前场景必须非空 |
| kRotation | 可选 | BFLOAT16 | ND | [D, D] | K 分支 rotation 矩阵，当前场景必须非空 |
| vScale | 可选 | FLOAT | ND | [Nv] | V 分支量化 scale，当前场景必须非空 |

### 属性参数

| 参数名 | 数据类型 | 必选/可选 | 默认值 | 说明 |
|--------|---------|----------|--------|------|
| headNums | INT64[] | 必选 | - | 头数配置 [Nq, Nk, Nv] |
| inputLayout | STRING | 必选 | - | qkv 与 qOut/qScale 的 N/T 轴顺序，仅支持大小写敏感的 NTD 和 TND |
| epsilon | FLOAT | 可选 | 1e-6 | RMSNorm epsilon |

### 输出参数

| 参数名 | 数据类型 | 数据格式 | Shape | 说明 |
|--------|---------|---------|-------|------|
| qOut | FLOAT8_E4M3FN | ND | 见 inputLayout | Q 分支量化输出 |
| qScale | FLOAT | ND | 见 inputLayout | Q 分支量化 scale |
| kCache | FLOAT8_E4M3FN | ND | [BlockNum, Nk, BlockSize, D] | K cache 原地更新输出 |
| vCache | FLOAT8_E4M3FN | ND | [BlockNum, Nv, BlockSize, D] | V cache 原地更新输出 |
| kScaleCache | FLOAT | ND | [BlockNum, Nk, BlockSize, 1] | K scale cache 原地更新输出 |

## 约束条件

- 仅支持 Ascend 950 (arch35) 平台。
- `inputLayout == "NTD"` 时，`qkv/qOut/qScale` 分别为 `[Nq+Nk+Nv, T, D]`、`[Nq, T, D]`、`[Nq, T]`。
- `inputLayout == "TND"` 时，`qkv/qOut/qScale` 分别为 `[T, Nq+Nk+Nv, D]`、`[T, Nq, D]`、`[T, Nq]`。
- `actualSeqLens.shape == [cuSeqLen.shape[0] - 1]`。
- 首期 fast path 支持 `D == 128` 且 `Nv == Nk`。
- `D` 必须为偶数。
- `qRotation.shape == [D, D]`，`kRotation.shape == [D, D]`，均为非转置 ND 矩阵。
- `vScale.shape[0] == Nv`。
- `headNums` 必须包含恰好 3 个正整数。
- `kCache`、`vCache`、`kScaleCache` 为 in-place 参数，ACLNN 调用不再额外传入 cache 输出 tensor。

## 计算公式

Q/K rotation 分支分别使用独立矩阵：

```text
q_after_cube[t,nq,:] = rope_q[t,nq,:] @ q_rotation
k_after_cube[t,nk,:] = rope_k[t,nk,:] @ k_rotation
```

cache 写回：

```text
block_id = slot_mapping[t] / BlockSize
block_offset = slot_mapping[t] % BlockSize
k_cache[block_id,nk,block_offset,:] = k_quant[t,nk,:]
v_cache[block_id,nv,block_offset,:] = v_out[t,nv,:]
k_scale_cache[block_id,nk,block_offset,0] = k_scale[t,nk]
```

## 调用示例

```cpp
aclTensor *qRotation = ...;
aclTensor *kRotation = ...;
aclTensor *vScale = ...;
aclTensor *qOut = ...;
aclTensor *qScale = ...;

uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnStatus ret = aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize(
    qkv, qGamma, kGamma, cosSin, slotMapping,
    kCache, vCache, kScaleCache, cuSeqLen, actualSeqLens,
    qRotation, kRotation, vScale, headNums, inputLayout, epsilon,
    qOut, qScale, &workspaceSize, &executor);
```

## 支持平台

- Ascend 950 (arch35)
