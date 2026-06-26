# aclnnMoeTokenPermuteV2

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_token_permute)

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- **接口功能**：MoE的permute计算，根据索引indices将tokens广播并排序。
- **扩展能力**：相比`aclnnMoeTokenPermute`，本接口新增`quantMode`和`expandedScaleOut`，用于Ascend 950平台下对接`MoeInitRoutingV3`，支持MXFP8和MXFP4量化输出。
- **平台行为**：
  - Ascend 950：调用`MoeInitRoutingV3`，支持`quantMode=-1/2/3/9`。
  - 非Ascend 950：量化参数静默忽略，按非量化兼容路径处理。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMoeTokenPermuteV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeTokenPermuteV2”接口执行计算。

```cpp
aclnnStatus aclnnMoeTokenPermuteV2GetWorkspaceSize(
    const aclTensor  *tokens,
    const aclTensor  *indices,
    int64_t           numOutTokens,
    bool              paddedMode,
    int64_t           quantMode,
    const aclTensor  *permuteTokensOut,
    const aclTensor  *sortedIndicesOut,
    const aclTensor  *expandedScaleOut,
    uint64_t         *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMoeTokenPermuteV2(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

## aclnnMoeTokenPermuteV2GetWorkspaceSize

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度 |
| --- | --- | --- | --- | --- | --- | --- |
| `tokens` | 输入 | 输入token特征。 | 支持空tensor；要求维度大于等于2，第一维为`num_tokens`。Ascend 950量化模式仅支持FLOAT16、BFLOAT16。 | 非量化：FLOAT16、BFLOAT16、FLOAT32、INT8；量化：FLOAT16、BFLOAT16 | ND | ≥2 |
| `indices` | 输入 | 输入indices索引。 | 支持空tensor；shape为1D或2D。`paddedMode=false`时表示每个输入token对应的topK个专家索引。 | INT32、INT64 | ND | 1或2 |
| `numOutTokens` | 输入 | 有效输出token数。 | `0`表示不删除token；大于0时保留排序后的前`numOutTokens`个token；小于0时按负切片索引处理。 | - | - | - |
| `paddedMode` | 输入 | 是否为填充模式。 | 当前不支持`true`，建议固定为`false`。 | - | - | - |
| `quantMode` | 输入 | 量化模式。 | `-1`为非量化；`2`为MXFP8 E5M2；`3`为MXFP8 E4M3FN；`9`为MXFP4 E2M1。非Ascend 950平台静默忽略量化参数。 | - | - | - |
| `permuteTokensOut` | 输出 | 根据indices扩展并排序后的tokens。 | 非量化时数据类型同`tokens`；`quantMode=2/3/9`时分别为FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1。 | FLOAT16、BFLOAT16、FLOAT32、INT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1 | ND | ≥2 |
| `sortedIndicesOut` | 输出 | `permuteTokensOut`和`tokens`的映射关系。 | 1D Tensor，shape为`[indices.numel()]`。 | INT32 | ND | 1 |
| `expandedScaleOut` | 输出 | 量化后的per-token分块scale。 | 量化场景必须提供有效输出；非量化场景可传shape为`[0]`、dtype为FLOAT32的空Tensor。 | FLOAT8_E8M0、FLOAT32 | ND | 1、2或3 |
| `workspaceSize` | 输出 | 返回需要在Device侧申请的workspace大小。 | - | - | - | - |
| `executor` | 输出 | 返回op执行器，包含算子计算流程。 | - | - | - | - |

## 输出Shape

先计算：

```text
flatten_size = indices.numel()

if numOutTokens > 0:
    M = min(numOutTokens, flatten_size)
else:
    M = max(numOutTokens + flatten_size, 0)

H = tokens.shape[1]
```

各模式输出shape如下：

| quantMode | permuteTokensOut shape | sortedIndicesOut shape | expandedScaleOut shape |
| --- | --- | --- | --- |
| `-1` | `[M, H]` | `[flatten_size]` | `[0]` |
| `2` | `[M, H]` | `[flatten_size]` | `[M, AlignUp(CeilDiv(H, 32), 2)]` |
| `3` | `[M, H]` | `[flatten_size]` | `[M, AlignUp(CeilDiv(H, 32), 2)]` |
| `9` | `[M, H / 2]` | `[flatten_size]` | `[M, CeilDiv(H, 64), 2]` |

其中：

```text
CeilDiv(a, b) = (a + b - 1) // b
AlignUp(a, b) = ((a + b - 1) // b) * b
```

## 返回值

aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

常见错误场景：

| 返回值 | 错误码 | 描述 |
| --- | --- | --- |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 输入或输出Tensor为空指针。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 输入、输出数据类型或`quantMode`不在支持范围内。 |
| ACLNN_ERR_INNER_TILING_ERROR | 561002 | shape不满足算子约束，或`paddedMode=true`。 |

## aclnnMoeTokenPermuteV2

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `workspace` | 输入 | 在Device侧申请的workspace内存地址。 |
| `workspaceSize` | 输入 | 在Device侧申请的workspace大小，由第一段接口`aclnnMoeTokenPermuteV2GetWorkspaceSize`获取。 |
| `executor` | 输入 | op执行器，包含算子计算流程。 |
| `stream` | 输入 | 指定执行任务的Stream流。 |

## 约束说明

- `indices`要求元素个数小于`16777215`，值大于等于`0`且小于`16777215`。
- 不支持`paddedMode=true`。
- Ascend 950量化模式下，`tokens` dtype与`MoeInitRoutingV3`量化输入类型一致，当前支持FLOAT16、BFLOAT16。
- `quantMode=9`时，`tokens`隐藏维`H`需要为偶数，输出`permuteTokensOut`隐藏维为`H / 2`。
- 非Ascend 950平台下，`quantMode=2/3/9`采用静默忽略策略，实际按非量化兼容路径处理。

## 调用示例

下面示例展示非量化调用。量化调用时需要按`quantMode`调整`permuteTokensOut`和`expandedScaleOut`的数据类型及shape。

```cpp
#include "aclnnop/aclnn_moe_token_permute_v2.h"

// 调用aclnnMoeTokenPermuteV2第一段接口
ret = aclnnMoeTokenPermuteV2GetWorkspaceSize(
    x,
    indices,
    numTokenOut,
    padMode,
    quantMode,
    expandedXOut,
    sortedIndicesOut,
    expandedScaleOut,
    &workspaceSize,
    &executor);
CHECK_RET(ret == ACL_SUCCESS,
          LOG_PRINT("aclnnMoeTokenPermuteV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

// 根据第一段接口计算出的workspaceSize申请device内存
void* workspaceAddr = nullptr;
if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
}

// 调用aclnnMoeTokenPermuteV2第二段接口
ret = aclnnMoeTokenPermuteV2(workspaceAddr, workspaceSize, executor, stream);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteV2 failed. ERROR: %d\n", ret); return ret);
```
