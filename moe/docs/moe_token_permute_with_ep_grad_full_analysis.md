# moe_token_permute_with_ep_grad 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeTokenPermuteWithEpGrad`
**算子类**: MoE Token 重排反向算子，Expert Parallelism (EP) 增强变体
**功能描述**: `moe_token_permute_with_ep` 的反向算子。将专家处理后的梯度（含 token 梯度和 probs 梯度）映射回原始 token 顺序，支持 EP 场景下的 range 区间约束。计算分为两阶段：先计算 token_grad（ISTOKEN=true），再计算 probs_grad（ISTOKEN=false）。

**计算公式**:
- token_grad: 当 `range[0] <= sortedIndices[i] < range[1]` 时，`tokenGradOut[i] = permutedTokensOutputGrad[sortedIndices[i] - range[0]]`，否则为 0；然后 reshape 为 `(-1, topK, hiddenSize)` 并沿 topK 维度求和
- probs_grad: 类似地，按 range 约束映射 probs 梯度，范围外填 0

**支持平台**: ascend910b (Atlas A2), ascend910_93 (Atlas A3)
**支持数据类型**: BF16/FP16/FP32

---

## 2. 目录结构与文件职责

```
moe_token_permute_with_ep_grad/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_token_permute_with_ep_grad_def.cpp            # 算子定义
│   ├── moe_token_permute_with_ep_grad_infershape.cpp     # Shape推导
│   ├── moe_token_permute_with_ep_grad_tiling.h           # Tiling数据结构定义 + 公共参数结构体
│   ├── moe_token_permute_with_ep_grad_tiling.cpp         # Tiling策略实现
│   └── config/
│       ├── ascend910_93/ {binary.json, simplified_key.ini}
│       └── ascend910b/  {binary.json, simplified_key.ini}
├── op_kernel/
│   ├── moe_token_permute_with_ep_grad.cpp                # Kernel入口（TilingKey分发）
│   └── moe_token_permute_with_ep_grad.h                  # Kernel核心实现类
├── docs/ aclnnMoeTokenPermuteWithEpGrad.md               # ACLNN API文档
├── examples/ test_aclnn_moe_token_permute_with_ep_grad.cpp
└── tests/ut/
    ├── op_host/ {infershape UT, tiling UT}
    └── op_kernel/ {UT + gen_data.py, gen_tiling.py}
```

---

## 3. Host 侧分析

### 3.1 算子定义 (op_host/moe_token_permute_with_ep_grad_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | permuted_tokens_output_grad | REQUIRED | BF16/FP16/FP32 | 2D: [numOutTokens, H] |
| 输入 | sorted_indices | REQUIRED | INT32 | 1D: [totalLength] |
| 输入 | permuted_probs_output_grad | OPTIONAL | BF16/FP16/FP32 | 1D: [numOutTokens] |
| 输出 | token_grad | REQUIRED | 与输入同类型 | 2D: [tokensNum, H] |
| 输出 | probs_grad | REQUIRED | 与输入同类型 | 2D: [tokensNum, topK] |

**属性**:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| num_topk | Int | 1 | 每个 token 的专家数 |
| range | ListInt | {0, 0} | EP 场景的专家范围 [start, end) |
| padded_mode | Bool | false | padding 模式（当前仅支持 false） |

**与 moe_token_permute_grad 的关键差异**:
1. 新增 probs 梯度输入和输出
2. range 属性支持 EP 场景
3. 两阶段计算（token_grad + probs_grad）
4. ISTOKEN 模板参数区分两种计算模式

### 3.2 Tiling 策略 (op_host/moe_token_permute_with_ep_grad_tiling.cpp)

**公共 Tiling 函数**: `PermuteWithEpGradTilingCompute()`（第369行），被本算子和 `moe_token_permute_with_routing_map_grad` 复用，通过 `isUnpermute` 参数区分调用场景。

**Tiling 流水线**（第369-387行）:
```
MoeTokenUnpermuteWithEpInputParamCheck()  →  校验输入参数
Init()            →  解析输入、获取 range 属性、计算 tokensNum/topK
SetCoreNum()      →  核数分配（tokensNum 与物理核数取较小值）
TilingHiddenSize() →  hidden 维度切分（按 UB 容量确定每次处理的 H 长度）
SetBufferNum()    →  buffer 数调整（2~4，UB 装不下时减少 buffer）
ComputeRemainMemerySpace() →  计算剩余 UB 空间
TilingToken()     →  token 维度切分（按剩余空间确定每次处理的 token 数）
SetTilingKey()    →  按 probs 是否存在及数据类型设置 Key
SetTilingData()   →  保存 TilingData + SetBlockDim
```

**TilingKey 编码**（tiling.cpp 第283-306行，本算子 isUnpermute=false）:
- Key=0: 无 probs
- Key=1: 有 probs

**核心参数结构**（tiling.h 第70-76行）:
```
MoeTokenUnpermuteWithEpParam
├── InputParam: tokensNum, topK, hiddenSize, totalLength, start, end, numOutTokens
│               tokensDtypeSize, indicesDtypeSize, probsDtypeSize, haveProbs, isUnpermute
├── TilingParam: hiddenTiling  {length, num, remain}
├── TilingParam: tokenTiling   {length, num, remain}
├── TilingParam: tokenPerCore  {length, num, remain}
└── CoreParam:   maxCoreMemery, maxCoreNum, usedCoreNum, remainMemerySpace, bufferNum, tilingKey
```

**UB 容量计算**（tiling.cpp 第86-94行 `ComputeMaxHiddenSize`）:
- 每个 hidden 单元占用: `inputDtypeSize * (QUE_NUM + bufferNum - 1) + FLOAT_DATA_SIZE * castNum`
- 其中 castNum = 非 float 类型时需要额外的 float 累加缓冲（2 份）
- 预留 5120 字节给 indices 和 probs 空间（topK 最大 512 时所需）

### 3.3 Shape 推导 (op_host/moe_token_permute_with_ep_grad_infershape.cpp)

**推导逻辑**（第21-37行）:
```
输入:
  permuted_tokens_output_grad: [numOutTokens, H]
  topK (attr)

推导:
  tokens_num = permuted_tokens_output_grad.GetDim(0) / topK
  token_grad = [tokens_num, H]
```

注意: `probs_grad` 的形状不由 infershape 推导（由调用者提供）。

**数据类型推导**（第39-42行）:
```
token_grad.dtype = permuted_tokens_output_grad.dtype
```

### 3.4 ACLNN API (docs/aclnnMoeTokenPermuteWithEpGrad.md)

采用 CANN 标准两段式接口:

**第一阶段 — 获取 Workspace 大小**:
```cpp
aclnnStatus aclnnMoeTokenPermuteWithEpGradGetWorkspaceSize(
    const aclTensor       *permutedTokensOutputGrad,
    const aclTensor       *sortedIndices,
    const aclTensor       *permutedProbsOutputGradOptional,
    int64_t                numTopk,
    const aclIntArray     *rangeOptional,
    bool                   paddedMode,
    const aclTensor       *tokenGradOut,
    const aclTensor       *probsGradOut,
    uint64_t              *workspaceSize,
    aclOpExecutor         **executor)
```

**第二阶段 — 执行计算**:
```cpp
aclnnStatus aclnnMoeTokenPermuteWithEpGrad(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

**错误码**: `ACLNN_ERR_PARAM_NULLPTR`(161001) — 空指针; `ACLNN_ERR_PARAM_INVALID`(161002) — 类型/格式不支持; `ACLNN_ERR_INNER_TILING_ERROR`(561002) — shape 维度或属性校验失败。

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化 (op_kernel/moe_token_permute_with_ep_grad.cpp)

**入口函数**（第40-60行）: `extern "C" __global__ __aicore__ void moe_token_permute_with_ep_grad(...)`

**初始化流程**:
1. `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` 声明仅使用 AIV 核
2. 通过 `GET_TILING_DATA` 获取 tiling 数据
3. 根据 `TILING_KEY_IS(0/1)` 分发到有无 probs 两种路径

**GENERAL_OP_IMPL 宏**（第26-38行）执行两阶段:
```
阶段1: 计算 token_grad (ISTOKEN=true)
    KernelMoeTokenUnpermuteWithEp<tokenDtype, int32_t, tokenDtype, false, true>
    op1.Init(permuted_output_grad, sorted_indices, nullptr, token_grad, ...)
    op1.Process()
    t_pipe.Destroy()  // 销毁阶段1的 pipe

阶段2: 计算 probs_grad (ISTOKEN=false, 仅当 haveProbs=true)
    TPipe t_probs_pipe  // 新建阶段2的 pipe
    KernelMoeTokenUnpermuteWithEp<probsDtype, int32_t, probsDtype, false, false>
    op2.Init(permuted_probs_output_grad, sorted_indices, nullptr, probs_grad, ...)
    op2.Process()
```

**为什么阶段间需 Destroy pipe**: 两个阶段共享 UB 空间，阶段1的 pipe 占用的 UB buffer 需释放后才能给阶段2使用。

### 4.2 核心计算流程

**模板参数**:
```
T1: 数据类型 (bfloat16_t/half/float)
T2: 索引类型 (int32_t)
T3: probs 类型 (同 T1)
PROBS: bool (始终为 false，不处理 probs 乘法)
ISTOKEN: bool — true: 处理 token 梯度(按 hidden 维度); false: 处理 prob 梯度(逐个标量)
```

**Process() 主循环**（moe_token_permute_with_ep_grad.h 第203-212行）:
```
for i in [0, tokens_splited_num):
    CalMultiOutToken(i * tokens_splited_length, tokens_splited_length)
if tokens_splited_remain > 0:
    CalMultiOutToken(tokens_splited_num * tokens_splited_length, tokens_splited_remain)
```

**CalMultiOutToken**（第215-249行）: 一次搬运多组 sorted_indices，然后用 `#pragma bisheng auto_sync parallel` 并行处理每个 token。

**ISTOKEN=true 时的 CalPartOutToken**（第269-329行）:
1. 从 indicesLocal 读取 `acl_token_idx = indicesLocal.GetValue(start_token)`
2. 判断 `acl_token_idx >= start && acl_token_idx < end`
3. 若在范围内: `CopyTokenIn` → `CalFirstToken` / `CalToken`（Cast 到 float 累加，乘 prob_value）
4. 若不在范围内: `Duplicate(0)` 填零
5. 循环处理 topK 个 token 后 `CopyOut`

**ISTOKEN=false 时的 CalPartOutProb**（第422-440行）:
1. 逐 prob 遍历 topK 个索引
2. 判断 `acl_prob_idx >= start && acl_prob_idx < end`
3. 在范围内: 从 `tokensGM[(idx-start)*hidden_size]` 读取 probs 梯度
4. 不在范围内: `Duplicate(0)` 填零
5. `CopyProbOut` 写到 probs_grad 输出

**关键: EP 偏移计算**: 所有读取偏移使用 `(idx - start)` 而非直接 `idx`，因为 `permuted_tokens_output_grad` 仅包含 [start, end) 范围的 token。

### 4.3 数据结构与内存布局

**ISTOKEN=true 时的 UB 布局**（Init 第168-181行）:
```
UB 空间划分:
+---------------------------------------------------+
| tokens_inque (buffer_num份, hidden_splited_length) |  ← token 输入（双/多缓冲）
+---------------------------------------------------+
| indices_inque (1份, block_splited_length * 4B)      |  ← sorted indices
+---------------------------------------------------+
| outque (1份, hidden_splited_length)                 |  ← 输出缓冲
+---------------------------------------------------+
| temp_buffer0 (hidden_splited_length * 4B + 256)     |  ← float 累加器0（非float时）
| temp_buffer1 (hidden_splited_length * 4B)           |  ← float 累加器1（非float时）
+---------------------------------------------------+
| probs_inque (1份, 仅 PROBS=true 时)                 |  ← probs 缓冲
| temp_buffer2 (block_splited_length * 4B)            |  ← float probs（非float时）
+---------------------------------------------------+
```

**ISTOKEN=false 时**: `tokens_inque` 替换为 `probs_inoutque`（单缓冲），hidden_size 被 tiling 设为 1。

### 4.4 多核并行策略

**核数分配**（tiling.cpp 第210-217行）:
- `usedCoreNum = min(tokensNum, maxCoreNum)`，按 token 数均匀分配核
- 尾核处理: 若 `tokensNum % usedCoreNum > 0`，前 `tokens_core_remain` 个核各多处理 1 个 token（kernel Init 第123-126行）

**hidden 维度切分**: 当 hidden_size 超过 UB 一次可容纳的长度时，按 `hidden_splited_length` 分多次处理，最后处理 `hidden_splited_remain` 尾块。

**token 维度切分**: 当 indices 和 probs 空间超出 UB 剩余空间时，按 `tokens_splited_length` 分批处理。

**无核间同步**: 各核独立处理各自的 token 子集，输出写到不同位置，无需 SyncAll。

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

**MoeTokenPermuteWithEpGradTilingData**（tiling.h 第80-99行）:

```
MoeTokenPermuteWithEpGradTilingData
├── hidden_size                  : int64_t   # 原始 H 维度（ISTOKEN=false 时 tiling 设为 1）
├── permuted_probs_grad_length   : int64_t   # probs梯度标记（0=有probs, 1=无probs）
├── top_k                        : int64_t   # topK
├── start                        : int64_t   # range 起始
├── end                          : int64_t   # range 结束
├── num_out_tokens               : int64_t   # 输出 token 数
├── hidden_splited_length        : int64_t   # hidden 维度每次处理长度
├── hidden_splited_num           : int64_t   # hidden 分片数
├── hidden_splited_remain        : int64_t   # hidden 尾块长度
├── tokens_core_length           : int64_t   # 每核分配的 token 数
├── tokens_core_remain           : int64_t   # 尾核数
├── tokens_splited_length        : int64_t   # token 每次处理长度
├── tokens_splited_num           : int64_t   # token 分片数
├── tokens_splited_remain        : int64_t   # token 尾块长度
└── buffer_num                   : int64_t   # UB buffer 数（2~4）
```

### 5.2 Workspace 布局

本算子不使用 Workspace（tiling.cpp 中未计算 workspaceSize）。所有中间计算在 UB 中完成，使用 float 类型累加避免精度损失。

---

## 6. 完整数据流

```
输入阶段:
  permuted_tokens_output_grad [numOut, H]
  sorted_indices [totalLength]
  permuted_probs_output_grad [numOut] (可选)

Host: Tiling计算
  - 按 tokensNum 分核
  - 切分 hidden 维度和 token 维度
  - 设置 TilingKey (0=无probs, 1=有probs)

Kernel 阶段1 — token_grad (ISTOKEN=true):
  permuted_tokens_output_grad   sorted_indices
       [numOut, H]              [totalLength]
            |                        |
            +-----------+------------+
                        |
              per core, per token_batch:
              搬入 sorted_indices[topK个]
              |
              for each token:
                idx = indicesLocal[token_idx]
                |
                +--[start <= idx < end]--→ CopyTokenIn((idx-start)*H + h_part)
                |                             Cast → float
                |                             CalFirstToken / CalToken (累加)
                +--[else]----------------→ Duplicate(0)
                |
              CopyOut (Cast 回原类型)
                        |
                 token_grad [tokensNum, H]

Kernel 阶段2 — probs_grad (ISTOKEN=false, 仅 Key=1):
  permuted_probs_output_grad   sorted_indices
       [numOut]                 [totalLength]
            |                        |
            +-----------+------------+
                        |
              per core, per token_batch:
              for each prob (topK个):
                idx = indicesLocal[prob_idx]
                |
                +--[start <= idx < end]--→ CopyProbIn((idx-start))
                +--[else]----------------→ Duplicate(0)
                CopyProbOut
                        |
                 probs_grad [tokensNum, topK]
```

---

## 7. 性能相关设计

1. **bisheng 并行编译指示**: `CalMultiOutToken` 中的 token 循环使用 `#pragma bisheng auto_sync parallel`（kernel 第240行），让编译器自动并行化多个 token 的处理。

2. **动态 buffer 数调整**: `SetBufferNum()` 在 UB 不足以容纳多缓冲时，从 4 逐步降低到 2（tiling.cpp 第233-241行），牺牲流水重叠换取更大的单次处理量。

3. **两阶段 pipe 分离**: 阶段1 结束后 `Destroy()` pipe，阶段2 新建 `TPipe`，确保 UB 空间完全释放后再用（kernel 第32-33行）。

4. **显式硬件事件同步**: 使用预分配的 8 个 event_t（MTE2↔S, S↔V, MTE2↔V, MTE3↔V），比 PipeBarrier 更细粒度，减少不必要的等待（kernel 第192-199行）。

5. **非 float 类型的 Cast 累加**: BF16/FP16 输入先 Cast 到 float 累加，再 Cast 回原类型输出，避免低精度累加误差（kernel 第354-367行 CalFirstToken）。

6. **range 外快速跳过**: 不在 [start, end) 范围的 token 直接 `Duplicate(0)` 填零，跳过 CopyTokenIn 和计算步骤（kernel 第300-302行）。

---

## 8. 依赖关系

```
op_host/moe_token_permute_with_ep_grad_tiling.cpp
└── moe_token_permute_with_ep_grad_tiling.h
    ├── register/op_def_registry.h               # 算子定义注册
    ├── register/tilingdata_base.h                # Tiling 数据基类
    ├── tiling/platform/platform_ascendc.h        # 平台信息（核数/UB大小）
    ├── register/op_impl_registry.h               # 算子实现注册
    ├── log/log.h                                 # 日志宏
    ├── platform/platform_info.h                  # 平台信息辅助
    └── tiling/tiling_api.h                       # Tiling API

op_kernel/moe_token_permute_with_ep_grad.cpp
├── moe_token_permute_with_ep_grad.h
│   └── kernel_operator.h                         # AscendC Kernel 框架
└── kernel_operator.h
```

---

## 9. 与同族变体的核心差异

### 9.1 vs moe_token_permute_grad

| 维度 | moe_token_permute_grad | moe_token_permute_with_ep_grad |
|------|------------------------|-------------------------------|
| probs 处理 | 无 | 有（可选输入 + 必选输出） |
| 计算阶段 | 单阶段 | 两阶段（token_grad + probs_grad） |
| ISTOKEN 参数 | 无 | true/false 区分两种模式 |
| range 支持 | 无 | 支持 [start, end) |
| 索引范围判断 | `idx < numOutTokens` | `idx >= start && idx < end` |
| 偏移计算 | `idx * hidden_size` | `(idx - start) * hidden_size` |
| 硬件同步 | PipeBarrier | 显式 SetFlag/WaitFlag |
| TilingKey | 0/2/4 | 0/1 |
| 共享 Tiling 代码 | 独立 | 复用 PermuteWithEpGradTilingCompute |
| 空 probs 场景 | 无 probs 概念 | Duplicate(0) 填充范围外 prob |

### 9.2 vs moe_token_permute_with_routing_map_grad

| 维度 | moe_token_permute_with_ep_grad | moe_token_permute_with_routing_map_grad |
|------|-------------------------------|----------------------------------------|
| 输入 | 3 个（梯度+索引+可选 probs 梯度） | 4 个（梯度+probs 梯度+索引+可选 routing_map） |
| dropAndPad | 不支持 | 支持（TilingKey=1000） |
| probs_grad 计算 | ISTOKEN=false 模式 | KernelMoeTokenPermuteWithRoutingMap |
| TilingData 结构 | 单一结构 | 两个子结构（unpermute + dropPad） |

---

## 10. 关键约束与注意事项

| 约束 | 说明 | 来源 |
|------|------|------|
| topK 范围 | (0, 512] | tiling.cpp 第141行 |
| tokens 必须为 2D | shape 维度校验 | tiling.cpp 第116-117行 |
| paddedMode | 当前仅支持 false | def.cpp |
| range 大小 | 必须为 2 或为空 | tiling.cpp 第177-178行 |
| 确定性计算 | 默认确定性实现 | ACLNN API 文档 |
| 无核间同步 | 各核独立处理，输出无重叠 | Kernel 设计 |
| probs 可选 | 为空时 TilingKey=0，跳过 probs_grad 计算 | tiling.cpp 第302-304行 |
