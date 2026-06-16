# aclnnCausalConv1dFn

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mamba/causal_conv1d)

## 功能说明

`aclnnCausalConv1dFn` 是因果一维卷积算子的前向计算（prefill / chunk-prefill）接口，用于处理长序列的初始计算阶段。

### 计算公式

$$
y_t = \text{Activation}\left(\sum_{j=0}^{W-1} w_j \cdot x_{t-j} + b\right)
$$

其中：
- $W$ 为卷积核宽度（支持2、3、4）
- $w_j$ 为卷积权重
- $b$ 为偏置（可选）
- $\text{Activation}$ 为激活函数（可选，SiLU）

### 运行模式

该接口专用于 **prefill 模式**，支持以下两种输入格式：

1. **Batch 模式**：x 为 3D 张量 `[B, L, D]`，其中 L > 1
2. **Varlen 模式**：x 为 2D 张量 `[T, D]`，配合 `queryStartLoc` 使用

### 特性支持

- ✅ APC（Automatic Prefix Caching）
- ✅ 初始状态模式（initialStateMode）
- ✅ 残差连接
- ❌ 投机解码（MTP）— 请使用 `aclnnCausalConv1dUpdate`

## 接口定义

### 第一段接口：计算workspace大小

```cpp
aclnnStatus aclnnCausalConv1dFnGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    aclTensor       *convStatesRef,
    const aclTensor *biasOptional,
    const aclTensor *queryStartLocOptional,
    const aclTensor *cacheIndicesOptional,
    const aclTensor *initialStateModeOptional,
    const aclTensor *initialStateIdxOptional,
    const aclTensor *numComputedTokensOptional,
    const aclTensor *blockIdxFirstScheduledTokenOptional,
    const aclTensor *blockIdxLastScheduledTokenOptional,
    int64_t         activationMode,
    int64_t         nullBlockId,
    int64_t         blockSizeToAlign,
    aclTensor       *y,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

### 第二段接口：执行计算

```cpp
aclnnStatus aclnnCausalConv1dFn(
    void          *workspace,
    uint64_t      workspaceSize,
    aclOpExecutor *executor,
    aclrtStream   stream)
```

## 参数说明

<table>
  <thead>
    <tr>
      <th>参数名</th>
      <th>类型</th>
      <th>说明</th>
      <th>约束</th>
      <th>数据类型</th>
      <th>格式</th>
      <th>维度</th>
      <th>非连续</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>输入序列张量</td>
      <td><ul><li>不支持空tensor</li><li>Batch模式：shape为[B, L, D]，L > 1</li><li>Varlen模式：shape为[T, D]</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>✓</td>
    </tr>
    <tr>
      <td>weight（aclTensor*）</td>
      <td>输入</td>
      <td>卷积权重张量</td>
      <td><ul><li>不支持空tensor</li><li>shape为[W, D]，W∈{2,3,4}</li></ul></td>
      <td>与x相同</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>convStatesRef（aclTensor*）</td>
      <td>输入/输出</td>
      <td>卷积状态张量，存储历史token数据</td>
      <td><ul><li>不支持空tensor</li><li>shape为[num_cache_lines, state_len, D]</li><li>state_len ≥ W-1</li><li>计算完成后原地更新</li></ul></td>
      <td>与x相同</td>
      <td>ND</td>
      <td>3</td>
      <td>✓</td>
    </tr>
    <tr>
      <td>biasOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>偏置张量</td>
      <td><ul><li>可为nullptr</li><li>shape为[D]</li></ul></td>
      <td>与x相同</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>queryStartLocOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>变长序列起始位置</td>
      <td><ul><li>Varlen模式必须提供</li><li>shape为[batch+1]</li><li>queryStartLoc[0]必须为0</li><li>值必须非递减</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cacheIndicesOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>缓存索引，指定每个序列对应的缓存状态</td>
      <td><ul><li>可为nullptr（使用恒等映射）</li><li>shape为[batch]</li><li>值∈[0, num_cache_lines)</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>initialStateModeOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>初始状态模式标志</td>
      <td><ul><li>仅prefill支持</li><li>可为nullptr（全部零初始化）</li><li>shape为[batch]</li><li>值：0=零初始化，1=使用缓存状态</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>initialStateIdxOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>初始状态索引</td>
      <td><ul><li>可为nullptr</li><li>shape为[batch]</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>numComputedTokensOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>已计算token数</td>
      <td><ul><li>可为nullptr</li><li>shape为[batch]</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>blockIdxFirstScheduledTokenOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>首个调度token块索引</td>
      <td><ul><li>可为nullptr</li><li>shape为[batch]</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>blockIdxLastScheduledTokenOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>最后调度token块索引</td>
      <td><ul><li>可为nullptr</li><li>shape为[batch]</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>activationMode（int64_t）</td>
      <td>属性</td>
      <td>激活函数类型</td>
      <td><ul><li>0：Identity（无激活）</li><li>1：SiLU</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nullBlockId（int64_t）</td>
      <td>属性</td>
      <td>空块ID，用于跳过不需要计算的序列</td>
      <td><ul><li>当cacheIndices[i]==nullBlockId时跳过</li><li>默认值0</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockSizeToAlign（int64_t）</td>
      <td>属性</td>
      <td>块大小对齐参数</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y（aclTensor*）</td>
      <td>输出</td>
      <td>输出张量</td>
      <td><ul><li>shape与x相同</li></ul></td>
      <td>与x相同</td>
      <td>ND</td>
      <td>2-3</td>
      <td>✓</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>workspace大小（字节）</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>执行器指针</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 约束条件

### 数据类型约束
- x、weight、convStatesRef、biasOptional、y 的数据类型必须一致（FLOAT16 或 BFLOAT16）
- queryStartLocOptional、cacheIndicesOptional、initialStateModeOptional、initialStateIdxOptional、numComputedTokensOptional、blockIdxFirstScheduledTokenOptional、blockIdxLastScheduledTokenOptional 的数据类型必须为 INT32

### Shape约束
- dim 必须满足 32 字节对齐：`(dim * dtypeSize) % 32 == 0`
- dim 范围：[64, 16384]
- batch 范围：[1, 1024]
- seqLen 范围：[1, 16384]（Batch模式）
- cuSeqlen 范围：[batch, 1024×1024]（Varlen模式）
- kernelWidth W ∈ {2, 3, 4}
- stateLen ≥ W - 1

### 模式约束
- Batch模式：x 为 3D `[B, L, D]`，L > 1
- Varlen模式：x 为 2D `[T, D]`，必须提供 queryStartLoc

## 错误码

| 错误码 | 数值 | 说明 |
|--------|------|------|
| ACLNN_SUCCESS | 0 | 成功 |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 参数为空指针 |
| ACLNN_ERR_INNER_TILING_ERROR | 561002 | 内部tiling错误，包括：<br>- 数据类型不在支持范围内<br>- 数据类型不一致<br>- shape不在支持范围内<br>- dim不满足对齐要求<br>- W不在[2,4]范围内<br>- Varlen模式未提供queryStartLoc |

## 调用示例

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_causal_conv1d_fn.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

int main() {
    // 初始化
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtContext context;
    aclrtCreateContext(&context, 0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // 参数设置
    int32_t batchSize = 2;
    int32_t seqLen = 4;
    int32_t dim = 64;
    int32_t width = 3;
    int32_t stateLen = width;

    // 创建tensor
    std::vector<int64_t> xShape = {batchSize, seqLen, dim};
    std::vector<int64_t> weightShape = {width, dim};
    std::vector<int64_t> convStatesShape = {batchSize, stateLen, dim};
    std::vector<int64_t> biasShape = {dim};
    std::vector<int64_t> yShape = {batchSize, seqLen, dim};

    aclTensor *x = aclCreateTensor(xShape.data(), xShape.size(), ACL_BF16, nullptr, 0, ACL_FORMAT_ND, xShape.data(), xShape.size(), nullptr);
    aclTensor *weight = aclCreateTensor(weightShape.data(), weightShape.size(), ACL_BF16, nullptr, 0, ACL_FORMAT_ND, weightShape.data(), weightShape.size(), nullptr);
    aclTensor *convStates = aclCreateTensor(convStatesShape.data(), convStatesShape.size(), ACL_BF16, nullptr, 0, ACL_FORMAT_ND, convStatesShape.data(), convStatesShape.size(), nullptr);
    aclTensor *bias = aclCreateTensor(biasShape.data(), biasShape.size(), ACL_BF16, nullptr, 0, ACL_FORMAT_ND, biasShape.data(), biasShape.size(), nullptr);
    aclTensor *y = aclCreateTensor(yShape.data(), yShape.size(), ACL_BF16, nullptr, 0, ACL_FORMAT_ND, yShape.data(), yShape.size(), nullptr);

    // 第一段：计算workspace
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    int64_t activationMode = 0;  // Identity
    int64_t nullBlockId = 0;

    auto ret = aclnnCausalConv1dFnGetWorkspaceSize(
        x, weight, convStates, bias,
        nullptr,  // queryStartLoc
        nullptr,  // cacheIndices
        nullptr,  // initialStateMode
        nullptr,  // initialStateIdx
        nullptr,  // numComputedTokens
        nullptr,  // blockIdxFirstScheduledToken
        nullptr,  // blockIdxLastScheduledToken
        activationMode, nullBlockId,
        0,  // blockSizeToAlign
        y, &workspaceSize, &executor);
    CHECK_RET(ret == ACLNN_SUCCESS, return -1);

    // 分配workspace
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 第二段：执行计算
    ret = aclnnCausalConv1dFn(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACLNN_SUCCESS, return -1);

    // 同步
    aclrtSynchronizeStream(stream);

    // 清理
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(convStates);
    aclDestroyTensor(bias);
    aclDestroyTensor(y);
    if (workspace) aclrtFree(workspace);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclFinalize();

    return 0;
}
```

## 性能建议

1. **使用Batch模式**：当所有序列长度相同时，使用 3D 输入 `[B, L, D]` 性能更优
2. **合理设置stateLen**：stateLen 应等于 W-1，过大会浪费内存
3. **启用APC**：通过 cacheIndices 复用历史状态，减少重复计算
4. **避免频繁创建/销毁executor**：在循环中复用 executor 对象

## 相关接口

- `aclnnCausalConv1dUpdate`：用于 decode/update 模式
- `aclnnCausalConv1d`：通用接口（自动推断模式）
