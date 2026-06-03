# aclnnQkvRmsNormRopeCacheWithKScale

## 接口原型

```cpp
aclnnStatus aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize(
    const aclTensor *qkv,
    const aclTensor *qGamma,
    const aclTensor *kGamma,
    const aclTensor *cosSin,
    const aclTensor *slotMapping,
    const aclTensor *kCache,
    const aclTensor *vCache,
    const aclTensor *kScaleCache,
    const aclTensor *cuSeqLen,
    const aclTensor *rotation,
    const aclTensor *vScale,
    const aclIntArray *headNums,
    float epsilon,
    aclTensor *qOut,
    aclTensor *qScale,
    aclTensor *kCacheOut,
    aclTensor *vCacheOut,
    aclTensor *kScaleCacheOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnQkvRmsNormRopeCacheWithKScale(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## 功能描述

QkvRmsNormRopeCacheWithKScale 是一个面向 Transformer decode/prefill 流程的融合算子，将以下操作融合在一个算子中：

1. **Q/K/V 拆分**：从融合输入中分离 Q、K、V
2. **RMSNorm**：对 Q 和 K 分别应用 RMSNorm（使用独立的 gamma 参数）
3. **RoPE**：应用旋转位置编码
4. **Rotation Matmul**：与 rotation 矩阵进行矩阵乘法
5. **FP8 动态量化**：对 Q/K 进行动态量化，对 V 进行简单量化
6. **KV Cache 写回**：将量化后的 K/V 写入 cache

## 参数说明

### 输入参数

| 参数名 | 必选/可选 | 数据类型 | 数据格式 | Shape | 说明 |
|--------|----------|---------|---------|-------|------|
| qkv | 必选 | BFLOAT16 | ND | [T, Nq+Nk+Nv, D] | Q/K/V 融合输入 |
| qGamma | 必选 | FLOAT | ND | [D] | Q 分支 RMSNorm gamma |
| kGamma | 必选 | FLOAT | ND | [D] | K 分支 RMSNorm gamma |
| cosSin | 必选 | FLOAT | ND | [MaxSeqLen, D] | RoPE cos/sin 数据 |
| slotMapping | 必选 | INT32 | ND | [T] | token 到 cache slot 的映射 |
| kCache | 必选 | FLOAT8_E4M3FN | ND | [BlockNum, Nk, BlockSize, D] | K cache（输入输出复用） |
| vCache | 必选 | FLOAT8_E4M3FN | ND | [BlockNum, Nv, BlockSize, D] | V cache（输入输出复用） |
| kScaleCache | 必选 | FLOAT | ND | [BlockNum, Nk, BlockSize, 1] | K scale cache（输入输出复用） |
| cuSeqLen | 必选 | INT32 | ND | [Batch+1] | batch 前缀和 |
| rotation | 可选 | BFLOAT16 | ND | [D, D] | rotation 矩阵（当前场景必须非空） |
| vScale | 可选 | FLOAT | ND | [Nv] | V 分支量化 scale（当前场景必须非空） |

### 属性参数

| 参数名 | 数据类型 | 必选/可选 | 默认值 | 说明 |
|--------|---------|----------|--------|------|
| headNums | INT64[] | 必选 | - | 头数配置 [Nq, Nk, Nv] |
| epsilon | FLOAT | 可选 | 1e-6 | RMSNorm epsilon |

### 输出参数

| 参数名 | 数据类型 | 数据格式 | Shape | 说明 |
|--------|---------|---------|-------|------|
| qOut | FLOAT8_E4M3FN | ND | [T, Nq, D] | Q 分支量化输出 |
| qScale | FLOAT | ND | [T, Nq] | Q 分支量化 scale |
| kCacheOut | FLOAT8_E4M3FN | ND | [BlockNum, Nk, BlockSize, D] | K cache 更新输出 |
| vCacheOut | FLOAT8_E4M3FN | ND | [BlockNum, Nv, BlockSize, D] | V cache 更新输出 |
| kScaleCacheOut | FLOAT | ND | [BlockNum, Nk, BlockSize, 1] | K scale cache 更新输出 |

## 约束条件

### 平台约束
- 仅支持 Ascend 950 (arch35) 平台

### Shape 约束
- `qkv.shape[1] == Nq + Nk + Nv`
- `D == 128`（首期 fast path）
- `D % 2 == 0`（RoPE 要求）
- `Nv == Nk`（首期 fast path）
- `kCache.shape[1] == Nk`
- `vCache.shape[1] == Nv`
- `kScaleCache.shape[1] == Nk`
- `cosSin.shape[1] == D`
- `rotation.shape == [D, D]`
- `vScale.shape[0] == Nv`

### 数据类型约束
- qkv: BFLOAT16
- qGamma, kGamma: FLOAT
- cosSin: FLOAT
- slotMapping: INT32
- kCache, vCache: FLOAT8_E4M3FN
- kScaleCache: FLOAT
- cuSeqLen: INT32
- rotation: BFLOAT16
- vScale: FLOAT
- qOut: FLOAT8_E4M3FN
- qScale: FLOAT

### 特殊约束
- `rotation` 和 `vScale` 虽然标记为可选，但当前场景必须非空
- `kCache`、`vCache`、`kScaleCache` 为输入输出复用（in-place 更新）
- `headNums` 必须包含恰好 3 个正整数

## 计算公式

### 1. Q/K/V 拆分
```
Q = qkv[:, 0:Nq, :]
K = qkv[:, Nq:Nq+Nk, :]
V = qkv[:, Nq+Nk:Nq+Nk+Nv, :]
```

### 2. RMSNorm
```
S[t,h] = sum(X[t,h,:]^2)
R[t,h] = sqrt(S[t,h] / D + epsilon)
Y[t,h,:] = X[t,h,:] / R[t,h] * gamma[h,:]
```

### 3. RoPE
```
out_low = y_low * cos - y_high * sin
out_high = y_high * cos + y_low * sin
```

### 4. Rotation Matmul
```
qk_after_cube[t,h,:] = rope_out[t,h,:] @ rotation
```

### 5. V 量化
```
v_out = cast_to_fp8(V * v_scale)
```

### 6. Q/K 动态量化
```
max_abs = max(abs(qk_after_cube))
scale = max_abs / 448
qk_quant = cast_to_fp8(qk_after_cube / scale)
```

### 7. Cache 写回
```
block_id = slot_mapping[t] / BlockSize
block_offset = slot_mapping[t] % BlockSize
k_cache[block_id, nk, block_offset, :] = qk_quant[t, Nq+nk, :]
v_cache[block_id, nv, block_offset, :] = v_out[t, nv, :]
k_scale_cache[block_id, nk, block_offset, 0] = scale[t, Nq+nk]
```

## 调用示例

```cpp
#include "aclnn_qkv_rms_norm_rope_cache_with_kscale.h"

// 准备输入输出 tensor
aclTensor *qkv = ...;
aclTensor *qGamma = ...;
aclTensor *kGamma = ...;
aclTensor *cosSin = ...;
aclTensor *slotMapping = ...;
aclTensor *kCache = ...;
aclTensor *vCache = ...;
aclTensor *kScaleCache = ...;
aclTensor *cuSeqLen = ...;
aclTensor *rotation = ...;
aclTensor *vScale = ...;
aclIntArray *headNums = ...;
float epsilon = 1e-6f;

aclTensor *qOut = ...;
aclTensor *qScale = ...;
aclTensor *kCacheOut = ...;
aclTensor *vCacheOut = ...;
aclTensor *kScaleCacheOut = ...;

// 获取 workspace 大小
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnStatus ret = aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize(
    qkv, qGamma, kGamma, cosSin, slotMapping,
    kCache, vCache, kScaleCache, cuSeqLen,
    rotation, vScale, headNums, epsilon,
    qOut, qScale, kCacheOut, vCacheOut, kScaleCacheOut,
    &workspaceSize, &executor);

// 分配 workspace
void *workspace = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}

// 执行算子
aclrtStream stream;
aclrtCreateStream(&stream);
ret = aclnnQkvRmsNormRopeCacheWithKScale(workspace, workspaceSize, executor, stream);
aclrtSynchronizeStream(stream);

// 清理
if (workspace) aclrtFree(workspace);
aclDestroyAclOpExecutor(executor);
aclrtDestroyStream(stream);
```

## 支持平台

- Ascend 950 (arch35)

## 注意事项

1. 当前为首期实现，仅支持 D=128 和 Nv==Nk 的 fast path
2. rotation 和 vScale 虽然接口标记为可选，但必须传入非空指针
3. kCache、vCache、kScaleCache 为 in-place 更新，输入输出共享物理内存
4. 不支持 D!=128 或 Nv!=Nk 的配置，会返回 GRAPH_FAILED
