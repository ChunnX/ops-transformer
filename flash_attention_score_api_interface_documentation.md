# FlashAttentionScore 算子对外接口完整文档

## 概述

FlashAttentionScore 算子提供了多层次的外部接口，包括 ACLNN（Ascend Compute Language Neural Network）原生接口和 PyTorch 框架接口。本文档详细说明了这些接口的定义、参数支持、使用方法和约束条件。

---

## 一、接口版本演进

### 1.1 ACLNN 接口版本

| 版本 | 接口名称 | 主要特性 | 适用场景 |
|------|----------|----------|----------|
| V1 | `aclnnFlashAttentionScore` | 基础 FlashAttention 功能 | 标准自注意力计算 |
| V2 | `aclnnFlashAttentionScoreV2` | 新增 `pseType` 参数 | 支持 PSE 计算模式选择 |
| V3 | `aclnnFlashAttentionScoreV3` | 新增 `sink` 参数 | 支持 GPT-OSS 模型 sink 功能 |
| V4 | `aclnnFlashAttentionScoreV4` | 完整功能版本 | 支持量化、RoPE、Dropout 优化等 |
| Quant | `aclnnQuantFlashAttentionScore` | 量化专用版本 | FP8 量化场景 |

### 1.2 变长序列接口版本

| 版本 | 接口名称 | 主要特性 | 适用场景 |
|------|----------|----------|----------|
| V1 | `aclnnFlashAttentionVarLenScore` | 基础变长序列支持 | 变长序列处理 |
| V2 | `aclnnFlashAttentionVarLenScoreV2` | 新增 `pseType` 参数 | 变长 + PSE 模式 |
| V3 | `aclnnFlashAttentionVarLenScoreV3` | 新增 RoPE 支持 | 变长 + RoPE |
| V4 | `aclnnFlashAttentionVarLenScoreV4` | 新增 `softmaxOutLayout` | 变长 + 输出布局控制 |
| V5 | `aclnnFlashAttentionVarLenScoreV5` | 完整功能版本 | 变长 + sink + 完整特性 |

---

## 二、ACLNN 接口详细说明

### 2.1 V4 接口（功能最完整）

#### 接口原型

**文件位置**: `op_api/aclnn_flash_attention_score.h:144-190`

```c++
// 第一段接口：获取工作空间大小
aclnnStatus aclnnFlashAttentionScoreV4GetWorkspaceSize(
    const aclTensor *query,                          // Query 输入
    const aclTensor *key,                            // Key 输入
    const aclTensor *value,                          // Value 输入
    const aclTensor *realShiftOptional,              // PSE 输入（可选）
    const aclTensor *dropMaskOptional,               // Dropout 掩码（可选）
    const aclTensor *paddingMaskOptional,            // Padding 掩码（可选）
    const aclTensor *attenMaskOptional,              // 注意力掩码（可选）
    const aclTensor *queryRopeOptional,               // Query RoPE（可选）
    const aclTensor *keyRopeOptional,                // Key RoPE（可选）
    const aclTensor *dScaleQOptional,                // Query 量化参数（可选）
    const aclTensor *dScaleKOptional,                // Key 量化参数（可选）
    const aclTensor *dScaleVOptional,                // Value 量化参数（可选）
    const aclTensor *sinkOptional,                   // Sink 输入（可选）
    const aclIntArray *prefixOptional,               // Prefix 稀疏参数（可选）
    const aclIntArray *actualSeqQLenOptional,        // 实际 Query 序列长度（可选）
    const aclIntArray *actualSeqKvLenOptional,        // 实际 KV 序列长度（可选）
    const aclIntArray *qStartIdxOptional,            // Query 起始索引（可选）
    const aclIntArray *kvStartIdxOptional,           // KV 起始索引（可选）
    double scaleValue,                               // 缩放系数
    double keepProb,                                 // Dropout 保持概率
    int64_t preTokens,                               // 向前 Token 数（滑窗左边界）
    int64_t nextTokens,                              // 向后 Token 数（滑窗右边界）
    int64_t headNum,                                 // 注意力头数
    char *inputLayout,                               // 输入数据布局
    int64_t innerPrecise,                            // 内部精度模式
    int64_t sparseMode,                              // 稀疏计算模式
    int64_t outDtype,                                // 输出数据类型
    int64_t pseType,                                 // PSE 计算类型
    char *softmaxOutLayout,                          // Softmax 输出布局
    int64_t seed,                                    // Dropout 随机种子
    int64_t offset,                                  // Dropout 偏移量
    const aclTensor *softmaxMaxOut,                  // Softmax 最大值输出
    const aclTensor *softmaxSumOut,                  // Softmax 求和输出
    const aclTensor *softmaxOutOut,                  // Softmax 输出
    const aclTensor *attentionOutOut,                // 注意力输出
    uint64_t *workspaceSize,                         // 工作空间大小输出
    aclOpExecutor **executor);                       // 执行器输出

// 第二段接口：执行计算
aclnnStatus aclnnFlashAttentionScoreV4(
    void *workspace,                                 // 工作空间内存
    uint64_t workspaceSize,                          // 工作空间大小
    aclOpExecutor *executor,                         // 执行器
    const aclrtStream stream);                       // 执行流
```

### 2.2 参数详细说明

#### 必需输入参数

| 参数名 | 类型 | 维度 | 数据类型 | 说明 |
|--------|------|------|----------|------|
| `query` | Tensor | [B,S,H] 或 [B,N,S,D] | FLOAT16/BFLOAT16/FLOAT32 | 查询向量 |
| `key` | Tensor | [B,S,H] 或 [B,N,S,D] | FLOAT16/BFLOAT16/FLOAT32 | 键向量 |
| `value` | Tensor | [B,S,H] 或 [B,N,S,D] | FLOAT16/BFLOAT16/FLOAT32 | 值向量 |
| `scaleValue` | double | - | - | 缩放系数，通常为 1/√D |
| `keepProb` | double | - | - | Dropout 保持概率，范围 (0,1] |
| `preTokens` | int64_t | - | - | 滑窗左边界，稀疏计算使用 |
| `nextTokens` | int64_t | - | - | 滑窗右边界，稀疏计算使用 |
| `headNum` | int64_t | - | - | 注意力头数 N1 |
| `inputLayout` | string | - | - | 数据布局：BSH/SBH/BSND/BNSD/TND |
| `innerPrecise` | int64_t | - | - | 内部精度：0/1(保留), 2(无效行处理) |
| `sparseMode` | int64_t | - | - | 稀疏模式：0-9 |
| `outDtype` | int64_t | - | - | 输出类型：0(fp16), 1(bf16) |
| `pseType` | int64_t | - | - | PSE 类型：0-3 |

#### 可选输入参数

| 参数名 | 类型 | 维度 | 数据类型 | 说明 |
|--------|------|------|----------|------|
| `realShiftOptional` | Tensor | [B,N,S,S]/[B,N,1,Skv]/[1,N,S,S] | FLOAT16/BFLOAT16/FLOAT32 | 位置编码偏移 |
| `dropMaskOptional` | Tensor | [0-1] | UINT8 | 外部 Dropout 掩码 |
| `attenMaskOptional` | Tensor | [B,N,S,S]/[B,1,S,S]/[1,1,S,S]/[S,S] | BOOL/UINT8 | 注意力掩码 |
| `queryRopeOptional` | Tensor | [B,S,H] 或 [B,N,S,D_rope] | FLOAT16/BFLOAT16 | Query RoPE 编码 |
| `keyRopeOptional` | Tensor | [B,S,H] 或 [B,N,S,D_rope] | FLOAT16/BFLOAT16 | Key RoPE 编码 |
| `dScaleQOptional` | Tensor | [B,N1,Ceil(Sq/128),1] | FLOAT32 | Query 反量化参数 |
| `dScaleKOptional` | Tensor | [B,N2,Ceil(Skv/256),1] | FLOAT32 | Key 反量化参数 |
| `dScaleVOptional` | Tensor | [B,N2,Ceil(Skv/512),1] | FLOAT32 | Value 反量化参数 |
| `sinkOptional` | Tensor | [N1] | FLOAT32 | Sink 偏置值 |
| `prefixOptional` | IntArray | [B] | INT64 | Prefix 模式的 N 值 |
| `actualSeqQLenOptional` | IntArray | [B] | INT64 | 实际 Query 序列长度 |
| `actualSeqKvLenOptional` | IntArray | [B] | INT64 | 实际 KV 序列长度 |
| `qStartIdxOptional` | IntArray | [1] | INT64 | Query 起始索引 |
| `kvStartIdxOptional` | IntArray | [1] | INT64 | KV 起始索引 |
| `softmaxOutLayout` | string | - | - | Softmax 输出布局：""/"same_as_input" |
| `seed` | int64_t | - | - | Dropout 随机种子 |
| `offset` | int64_t | - | - | Dropout 偏移量 |

#### 输出参数

| 参数名 | 类型 | 维度 | 数据类型 | 说明 |
|--------|------|------|----------|------|
| `softmaxMaxOut` | Tensor | 与 query 相同或由 layout 决定 | FLOAT32 | Softmax 最大值（反向用） |
| `softmaxSumOut` | Tensor | 与 query 相同或由 layout 决定 | FLOAT32 | Softmax 求和（反向用） |
| `softmaxOutOut` | Tensor | 与 query 相同 | FLOAT16/BFLOAT16/FLOAT32 | Softmax 输出 |
| `attentionOutOut` | Tensor | 与 query 相同 | FLOAT16/BFLOAT16/FLOAT32 | 注意力输出 |
| `workspaceSize` | uint64_t | - | - | 所需工作空间大小 |
| `executor` | aclOpExecutor | - | - | 算子执行器 |

---

## 三、PyTorch 接口详细说明

### 3.1 接口函数

**文件位置**: `tests/pytest/npu_impl.py:24-87`

```python
def fa_npu(q, k, v, q_rope, k_rope, atten_mask, pse, **kwargs):
    """
    NPU 侧 FlashAttention 实现

    参数:
        q: Query tensor, shape [B, N, S, D] (BNSD layout)
        k: Key tensor, shape [B, N, S, D]
        v: Value tensor, shape [B, N, S, D]
        q_rope: Query RoPE tensor (可选)
        k_rope: Key RoPE tensor (可选)
        atten_mask: Attention mask tensor (可选)
        pse: Positional encoding tensor (可选)

    kwargs 参数:
        input_layout: 数据布局，默认 "BNSD"
        N1: 注意力头数
        D: 头维度
        DRope: RoPE 维度，默认 0
        sparse_mode: 稀疏模式，默认 0
        prefix: Prefix 参数（可选）
        pre_tokens: 向前 token 数，默认 2147483647
        next_tokens: 向后 token 数，默认 2147483647
        actual_seq_qlen: 实际 Query 序列长度（可选）
        actual_seq_kvlen: 实际 KV 序列长度（可选）
        scale: 缩放系数，默认 1/√D
        pse_type: PSE 类型，默认 1
        keep_prob: Dropout 保持概率，默认 1.0
        q_start_idx: Query 起始索引（可选）
        kv_start_idx: KV 起始索引（可选）
        seed: 随机种子，默认 0

    返回:
        tuple: (attention_out, softmax_max, softmax_sum)
    """
```

### 3.2 PyTorch 调用示例

```python
import torch
import torch_npu

# 基本调用
output = torch_npu.npu_fusion_attention_v2(
    q1, k1, v1, n1, input_layout,
    # 可选参数
    pse=pse1,
    atten_mask=atten_mask1,
    actual_seq_qlen=actual_seq_qlen,
    actual_seq_kvlen=actual_seq_kvlen,
    query_rope=query_rope,
    key_rope=key_rope,
    scale=scale,
    keep_prob=keep_prob,
    pre_tokens=pre_tokens,
    next_tokens=next_tokens,
    prefix=prefix,
    pse_type=pse_type,
    q_start_idx=tuple([q_start_idx]) if q_start_idx is not None else None,
    kv_start_idx=tuple([kv_start_idx]) if kv_start_idx is not None else None,
    sparse_mode=sparse_mode
)

# 返回值: (attention_out, softmax_max, softmax_sum)
npu_out = output[0].cpu()
npu_max = output[1].cpu()
npu_sum = output[2].cpu()
```

---

## 四、核心参数详解

### 4.1 稀疏模式 (sparseMode)

**代码位置**: `op_kernel/arch32/flash_attention_score_common.h:75-86`

| 值 | 模式名称 | 说明 | 适用场景 |
|----|----------|------|----------|
| 0 | NO_MASK | 无掩码 | 全注意力计算 |
| 1 | ALL_MASK | 全掩码 | 使用外部 atten_mask |
| 2 | LEFT_UP_CAUSAL | 左上因果 | decoder 模型 |
| 3 | RIGHT_DOWN_CAUSAL | 右下因果 | encoder 模型 |
| 4 | BAND | 滑窗模式 | 滑窗注意力 |
| 5 | PREFIX | Prefix 模式 | prefix LM |
| 6 | PREFIX_COMPRESS | Prefix 压缩 | 压缩 prefix |
| 7 | RIGHT_DOWN_CAUSAL | 右下因果（别名） | - |
| 8 | RIGHT_DOWN_CAUSAL_BAND | 右下因果+滑窗 | 复合模式 |
| 9 | BAND_LEFT_UP_CAUSAL | 滑窗+左上因果 | 复合模式 |

### 4.2 数据布局 (inputLayout)

| 布局 | 格式 | 维度说明 | 使用场景 |
|------|------|----------|----------|
| BSH | [Batch, Seq, Head] | H = N × D | 传统布局 |
| SBH | [Seq, Batch, Head] | H = N × D | 序列优先布局 |
| BSND | [Batch, Seq, Num, Dim] | 标准四维 | 最常用布局 |
| BNSD | [Batch, Num, Seq, Dim] | Num 优先 | NPU 优化布局 |
| TND | [Total, Num, Dim] | 变长序列 | 变长序列专用 |

### 4.3 PSE 类型 (pseType)

| 值 | 含义 | 计算顺序 | 备注 |
|----|------|----------|------|
| 0 | 外部 PSE，先 mul 后 add | score × pse + bias | 自定义 PSE |
| 1 | 外部 PSE，先 add 后 mul | (score + pse) × scale | 默认模式，与 V1 兼容 |
| 2 | 内部生成 PSE，先 mul 后 add | 自动生成 | 当前仅支持 Sq=Skv |
| 3 | 内部生成 PSE，先 mul 后 add 再 sqrt | 自动生成 ALiBi | ALiBi 位置编码 |

### 4.4 内部精度模式 (innerPrecise)

| 值 | 模式 | 说明 | 性能影响 |
|----|------|------|----------|
| 0 | 保留值 | - | - |
| 1 | 保留值 | - | - |
| 2 | AA_INVALID_LINE_HIGH_PRECISION | 无效行高精度处理 | 性能下降，精度提升 |

**触发条件**：
- 手动设置 `innerPrecise=2`
- 算子自动判断（如 BAND 模式下 preTokens < s1Size - s2Size 或 nextTokens < 0）

---

## 五、参数约束与使用建议

### 5.1 维度约束

#### 基本约束
```cpp
// B: Batch Size
// S: Sequence Length
// N: Number of Heads
// D: Head Dimension
// H: Head Size = N × D

B: 1 ~ 2M (带 prefix 时最大 2K)
S: 1 ~ 1M (TND 格式最大 20000)
N: 1 ~ 256
D: 1 ~ 768
H: N × D
```

#### 数据类型约束
```cpp
// 输入支持的数据类型
query/key/value: FLOAT16 | BFLOAT16 | FLOAT32 | FLOAT8_E5M2 | FLOAT8_E4M3FN
atten_mask: BOOL | UINT8
drop_mask: UINT8
real_shift/pse: FLOAT16 | BFLOAT16 | FLOAT32

// 输出数据类型
softmax_max/sum: FLOAT32
softmax_out: FLOAT16 | BFLOAT16 | FLOAT32
attention_out: FLOAT16 | BFLOAT16 | FLOAT32
```

### 5.2 稀疏模式约束

#### BAND 模式约束
```cpp
// sparseMode = 4 (BAND)
// 1. preTokens 和 nextTokens 必须有交集
// 2. attenMask 与 preTokens/nextTokens 范围必须一致
// 3. 支持负值，表示向对角线外扩展
```

#### PREFIX 模式约束
```cpp
// sparseMode = 5/6 (PREFIX)
// 等长场景 (Sq == Skv):
//   - prefix N 取值范围: [0, Skv]
// 不等长场景 (Sq > Skv):
//   - prefix N 取值范围: [Skv-Sq, Skv]
// 不等长场景 (Sq <= Skv):
//   - prefix N 取值范围: [0, Skv]
```

### 5.3 性能优化建议

#### 1. 稀疏模式选择
```cpp
// 当 attenMaskOptional.shape < 2048 时
// 建议使用 sparseMode = 0 (NO_MASK)
// 可减少内存使用量

// 用户不特意指定时建议传入 sparseMode = 0
```

#### 2. 数据布局选择
```cpp
// 性能优先: BNSD (NPU 优化布局)
// 内存优先: BSND (标准布局)
// 变长序列: TND (专用布局)
```

#### 3. 序列长度对齐
```cpp
// 建议按 128 字节对齐序列长度
// 可提升内存访问效率
const int64_t ALIGN_SIZE = 128;
S2_aligned = ((S2 + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
```

---

## 六、完整调用流程

### 6.1 ACLNN 两段式接口流程

```cpp
// 第一阶段：获取工作空间大小
aclnnStatus aclnnFlashAttentionScoreV4GetWorkspaceSize(
    // 输入参数
    query, key, value,
    realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional,
    queryRopeOptional, keyRopeOptional, dScaleQOptional, dScaleKOptional, dScaleVOptional,
    sinkOptional, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional,
    qStartIdxOptional, kvStartIdxOptional,
    scaleValue, keepProb, preTokens, nextTokens, headNum,
    inputLayout, innerPrecise, sparseMode, outDtype, pseType,
    softmaxOutLayout, seed, offset,
    // 输出参数
    softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut,
    &workspaceSize, &executor
);

// 第二阶段：执行计算
aclnnStatus aclnnFlashAttentionScoreV4(
    workspace,      // 工作空间内存
    workspaceSize,  // 工作空间大小
    executor,       // 执行器
    stream          // 执行流
);
```

### 6.2 完整代码示例

```cpp
#include "aclnnop/aclnn_flash_attention_score.h"

// 1. 初始化
int32_t deviceId = 0;
aclrtContext context;
aclrtStream stream;
aclInit(nullptr);
aclrtSetDevice(deviceId);
aclrtCreateContext(&context, deviceId);
aclrtCreateStream(&stream);

// 2. 构造输入
int64_t B = 1, N1 = 8, S1 = 512, S2 = 512, D = 128;
std::vector<int64_t> qShape = {B, N1, S1, D};  // BNSD 格式
std::vector<int64_t> kShape = {B, N1, S2, D};
std::vector<int64_t> vShape = {B, N1, S2, D};

// 创建 tensor
aclTensor* query = CreateTensor(qShape, ACL_FLOAT16);
aclTensor* key = CreateTensor(kShape, ACL_FLOAT16);
aclTensor* value = CreateTensor(vShape, ACL_FLOAT16);

// 设置参数
double scaleValue = 1.0 / sqrt(D);
double keepProb = 1.0;
int64_t preTokens = 2147483647;  // INT_MAX
int64_t nextTokens = 2147483647; // INT_MAX
int64_t headNum = N1;
char inputLayout[] = "BNSD";
int64_t innerPrecise = 0;
int64_t sparseMode = 0;  // NO_MASK
int64_t outDtype = 0;
int64_t pseType = 1;
char softmaxOutLayout[] = "";

// 3. 获取 workspace
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclnnFlashAttentionScoreV4GetWorkspaceSize(
    query, key, value, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
    scaleValue, keepProb, preTokens, nextTokens, headNum,
    inputLayout, innerPrecise, sparseMode, outDtype, pseType,
    softmaxOutLayout, 0, 0,
    softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut,
    &workspaceSize, &executor
);

// 4. 分配 workspace
void* workspace = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}

// 5. 执行计算
aclnnFlashAttentionScoreV4(workspace, workspaceSize, executor, stream);

// 6. 同步并获取结果
aclrtSynchronizeStream(stream);
CopyOutToHost(attentionOutOut);
```

---

## 七、错误码与调试

### 7.1 常见错误码

| 错误码 | 错误名称 | 说明 | 解决方法 |
|--------|----------|------|----------|
| 161001 | ACLNN_ERR_PARAM_NULLPTR | 必需参数为空指针 | 检查所有必需输入参数 |
| 161002 | ACLNN_ERR_PARAM_INVALID | 参数无效 | 检查数据类型、维度范围 |
| 161003 | ACLNN_ERR_PARAM_SHAPE | 形状不匹配 | 检查 query/key/value 形状一致性 |
| 161004 | ACLNN_ERR_UNSUPPORTED | 不支持的操作 | 检查数据类型和布局组合 |

### 7.2 调试建议

#### 1. 参数校验
```cpp
// 检查输入参数
assert(query != nullptr && key != nullptr && value != nullptr);
assert(query->dtype == key->dtype && key->dtype == value->dtype);
assert(headNum > 0 && headNum <= 256);
assert(keepProb > 0.0 && keepProb <= 1.0);
```

#### 2. 日志输出
```cpp
// 启用详细日志
setenv("ASCEND_GLOBAL_LOG_LEVEL", "1", 1);
setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", 1);

// 查看算子选择日志
// 会输出 sparseMode、inputLayout 等关键参数
```

#### 3. 性能分析
```cpp
// 使用 NPU 性能分析工具
// 1. 开启 Profiling
aclprofCreateConfig(...);
aclprofStart(...);

// 2. 执行算子
aclnnFlashAttentionScoreV4(...);

// 3. 停止 Profiling
aclprofStop(...);
aclprofDestroy(...);
```

---

## 八、版本兼容性说明

### 8.1 接口兼容性

| 版本 | 向后兼容 | 新特性 | 弃用特性 |
|------|----------|--------|----------|
| V1 → V2 | ✓ | pseType 参数 | - |
| V2 → V3 | ✓ | sink 参数 | - |
| V3 → V4 | ✓ | 量化、RoPE、seed/offset | - |
| V4 | - | 当前版本 | - |

### 8.2 迁移指南

#### 从 V1 升级到 V4
```cpp
// V1 接口
aclnnFlashAttentionScoreGetWorkspaceSize(
    query, key, value, realShift, dropMask, paddingMask, attenMask, prefix,
    scaleValue, keepProb, preTokens, nextTokens, headNum, inputLayout,
    innerPrecise, sparseMode,
    softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut,
    &workspaceSize, &executor
);

// V4 接口（新增参数）
aclnnFlashAttentionScoreV4GetWorkspaceSize(
    query, key, value, realShift, dropMask, paddingMask, attenMask,
    queryRope, keyRope,                    // 新增 RoPE
    dScaleQ, dScaleK, dScaleV,             // 新增量化参数
    sink,                                 // 新增 sink
    prefix,
    actualSeqQLen, actualSeqKvLen,         // 新增变长支持
    qStartIdx, kvStartIdx,                 // 新增起始索引
    scaleValue, keepProb, preTokens, nextTokens, headNum,
    inputLayout, innerPrecise, sparseMode,
    outDtype,                              // 新增输出类型
    pseType,                               // 新增 PSE 类型
    softmaxOutLayout,                      // 新增输出布局
    seed, offset,                          // 新增随机种子
    softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut,
    &workspaceSize, &executor
);
```

---

## 九、总结

### 9.1 接口选择建议

| 场景 | 推荐接口 | 理由 |
|------|----------|------|
| 标准自注意力 | V1/V2 | 兼容性好，参数简单 |
| 需要 PSE 控制 | V2+ | 支持 pseType 参数 |
| GPT-OSS 模型 | V3+ | 支持 sink 功能 |
| 量化推理 | V4 | 支持 FP8 量化 |
| 变长序列 | VarLenV5 | 完整变长支持 |

### 9.2 关键参数配置建议

```cpp
// 基础配置
inputLayout = "BNSD";           // 推荐 NPU 优化布局
sparseMode = 0;                 // 默认无掩码模式
pseType = 1;                    // 默认兼容模式
innerPrecise = 0;               // 自动选择精度模式

// 性能优化
preTokens = INT_MAX;            // 不限制向前范围
nextTokens = INT_MAX;           // 不限制向后范围
keepProb = 1.0;                 // 不使用 dropout（推理时）

// 精度优化
innerPrecise = 2;               // 启用无效行高精度（必要时）
```

### 9.3 注意事项

1. **内存管理**: 两段式接口需要手动管理 workspace 内存
2. **流同步**: 需要调用 `aclrtSynchronizeStream` 确保计算完成
3. **错误处理**: 必须检查每一阶段返回状态码
4. **资源释放**: 记得释放 tensor 和 workspace 内存
5. **版本匹配**: 确保 CANN 版本与 PyTorch 版本匹配

**相关文档位置**：
- ACLNN 接口定义：`op_api/aclnn_flash_attention_score.h`
- PyTorch 接口实现：`tests/pytest/npu_impl.py`
- 算子原型定义：`op_graph/flash_attention_score_proto.h`
- 完整参数文档：`docs/aclnnFlashAttentionScoreV4.md`