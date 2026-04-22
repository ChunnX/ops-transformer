# moe_fused_topk 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeFusedTopk`
**算子类**: MoE 融合分组 TopK
**功能描述**: MoE 融合分组 TopK 算子。将 Sigmoid 激活 + 偏置相加 + 分组 TopK 选择 + Sigmoid 权重提取 + L1 归一化融合为单一算子。支持物理专家到逻辑专家的映射表（mapping_table）机制。输出权重固定为 FLOAT 类型。

**支持平台**: ascend910b, ascend910_93
**支持数据类型**: FP16/BF16/FP32（输入），FLOAT（输出权重），INT32（索引）

---

## 2. 目录结构与文件职责

```
moe_fused_topk/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_fused_topk_def.cpp             # 算子定义
│   ├── moe_fused_topk_tiling.h            # TilingData结构（22字段）
│   ├── moe_fused_topk_tiling.cpp          # Tiling策略（2 TilingKey）
│   ├── op_api/
│   │   ├── aclnn_moe_fused_topk.h         # ACLNN API声明
│   │   ├── aclnn_moe_fused_topk.cpp       # ACLNN参数校验+执行
│   │   ├── moe_fused_topk.h               # 内部接口声明
│   │   └── moe_fused_topk.cpp             # 内部实现
│   └── config/ {ascend910b, ascend910_93}
├── op_kernel/
│   ├── moe_fused_topk.cpp                 # Kernel入口（6分支）
│   └── moe_fused_topk.h                   # MoeFusedTopk模板类
├── examples/
└── tests/ut/ {op_host, op_kernel}
```

---

## 3. Host 侧分析

### 3.1 算子定义 (moe_fused_topk_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | x | REQUIRED | FP16/BF16/FP32 | 2D: [numToken, expertNum] |
| 输入 | add_num | REQUIRED | 同 x | 1D: [expertNum] |
| 输入 | mapping_num | OPTIONAL | INT32 | 1D: [expertNum] |
| 输入 | mapping_table | OPTIONAL | INT32 | 2D: [expertNum, maxMappingNum] |
| 输出 | y | REQUIRED | FLOAT | 2D: [numToken, topK] |
| 输出 | indices | REQUIRED | INT32 | 2D: [numToken, topK] |

**属性**:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| group_num | Int | 必填 | 分组数量（<=256） |
| group_topk | Int | 必填 | 选择的分组数量 |
| top_n | Int | 必填 | 组内用于计算组得分的专家数 |
| top_k | Int | 必填 | 最终选取的专家数量 |
| activate_type | Int | 必填 | 激活类型（仅 0=Sigmoid） |
| is_norm | Bool | true | 是否对输出归一化 |
| scale | Float | 1.0 | 归一化后的缩放系数 |
| enable_expert_mapping | Bool | false | 是否使能专家映射 |

### 3.2 Tiling 策略 (moe_fused_topk_tiling.cpp)

**TilingData**（tiling.h）: 22 个字段，含 `firstDimSize/secondDimSize/addNumDimSize`（维度）、`groupNum/groupTopk/topN/topK`（分组参数）、`activateType/isNorm/scale`（控制参数）、`groupEles`（每组专家数）、`blockNum/batchPerCore/tailBatch`（核分配）、`ubFactorElement`（UB 因子）、`expertNum/tableDim`（专家映射）、`topkMaxValue/topkMinValue`（TopK 临时空间）、`workspacePerCore`（每核 workspace）、`topkTilingData`（高阶 TopK API Tiling）。

**TilingKey 编码**:

| Key | 策略 |
|-----|------|
| 0 | 不使能专家映射（enableExpertMapping=false） |
| 1 | 使能专家映射（enableExpertMapping=true） |

**行切分策略**: numToken<=coreNum 时 batchPerCore=1, usedCoreNum=numToken；否则 batchPerCore=numToken/coreNum, 尾核处理余数。

**UB 空间计算**: xInBuf + addNumInBuf + yOutBuf + indicesOutBuf + sigmoidBuf + sigmoidAddBuf + sortedBuf + topkValueBuf + assistBuf + mappingNumBuf + tempBuf。

### 3.3 Shape 推导

本算子通过 ACLNN API 层直接分配输出 tensor，无需独立 InferShape 文件:

```cpp
outShape = x->GetViewShape();
outShape.SetDim(1, topK);
y = executor->AllocTensor(outShape, DT_FLOAT, FORMAT_ND);
indices = executor->AllocTensor(outShape, DT_INT32, FORMAT_ND);
```

### 3.4 ACLNN API (aclnn_moe_fused_topk.h)

**两段式接口**:

```cpp
// 第一段
aclnnStatus aclnnMoeFusedTopkGetWorkspaceSize(
    const aclTensor* x, const aclTensor* addNum,
    const aclTensor* mappingNum, const aclTensor* mappingTable,
    int64_t groupNum, int64_t groupTopk, int64_t topN, int64_t topK,
    int64_t activateType, bool isNorm, double scale,
    bool enableExpertMapping,
    const aclTensor* y, const aclTensor* indices,
    uint64_t* workspaceSize, aclOpExecutor** executor);

// 第二段
aclnnStatus aclnnMoeFusedTopk(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);
```

**参数校验**: groupNum > 0 且 <= 256，groupTopk <= groupNum，topK <= expertNum 且 expertNum <= 1024，topN <= expertNum/groupNum，expertNum % groupNum == 0。若 enableExpertMapping: maxMappingNum <= 128。

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化

**Kernel 入口**（moe_fused_topk.cpp）: 3 dtype × 2 Key = 6 个分支。

```
moe_fused_topk() 入口
  +-- DT_FLOAT:   Key=0 → MoeFusedTopk<float, float, 0>
  |               Key=1 → MoeFusedTopk<float, float, 1>
  +-- DT_FLOAT16: Key=0 → MoeFusedTopk<half, float, 0>
  |               Key=1 → MoeFusedTopk<half, float, 1>
  +-- DT_BF16:    Key=0 → MoeFusedTopk<bfloat16_t, float, 0>
                  Key=1 → MoeFusedTopk<bfloat16_t, float, 1>
```

模板参数: `inputT`（输入类型），`calT`（计算类型，固定 float），`enableExpertMapping`（编译时常量）。

### 4.2 核心计算流程

**数学公式**:
```
1. sigmoid = Sigmoid(x) + addNum                  (激活+偏置)
2. groupScore[g] = ReduceSum(TopK(sigmoid[g], topN))  (组内TopN求和)
3. topKGroups = TopK(groupScore, groupTopk)        (组间TopK)
4. 未选中组置0: sigmoid[mask==0] = 0
5. y = TopK(sigmoid, topK)                         (全局TopK)
6. 若 isNorm: y = y / sum(y) * scale               (L1归一化)
7. 若 enableExpertMapping: indices = mappingTable[indices][offset]
```

**Process 流程**:
```
CopyInAddNum() [+ CopyInMappingNum() 若启用映射]
for loop in 0..loopBatch:
    CopyInX(loop)
    ActivateAndAdd()           ← Sigmoid + Add
    CopyToWorkspace()          ← UB不足时暂存到GM
    CopyFromWorkspace()        ← 按分组加载回UB（带对齐填充）
    GroupReduceSumInternelImpl()  ← 组内TopN + ReduceSum
    GroupTopkImpl()            ← 组间TopK + 广播掩码
    GatherSigmoidImpl()        ← 全局TopK
    [isNorm] NormImpl()        ← L1归一化
    CopyOut(loop)              ← [映射查表]
```

**Workspace 使用机制**: 由于 UB 空间有限，Sigmoid+Add 结果先写入 workspace，再按分组加载回 UB（每组对齐到 CeilAlign(groupEles, 32)，不足的用 floatNegativeInf 填充）。

**专家映射**: 当 enableExpertMapping=true 时，CopyOut 阶段查 mappingTable 将物理专家 ID 映射为逻辑专家 ID: `indices[kI] = mappingTable[expertId * tableDim + (batchOffset + loop) % expertMappingNum]`。

### 4.3 数据结构与内存布局

**UB 内存布局**:
```
┌────────────────────────────────────────────────┐
│ xInBuf        | groupElesAlign*groupNum*FP32   │  ← 按分组存储
│ addNumInBuf   | expertNumAlign*FP32            │  ← 偏置
│ yOutBuf       | topkPad*FP32                   │  ← 输出权重
│ indicesOutBuf | topkSortCount*INT32            │  ← 输出索引
│ sigmoidBuf    | expertNumAlign*FP32            │  ← Sigmoid结果
│ sigmoidAddBuf | expertNumAlign*FP32            │  ← Sigmoid+Add
│ tempBuf       | topkMinValue                   │  ← 临时空间
│ sortedBuf     | groupNum*groupElesAlign*INT64  │  ← 排序结果
│ topkValueBuf  | expertNumAlign*FP32            │  ← TopK中间值
│ assistBuf     | expertNumAlign*UINT32          │  ← 索引辅助
│ mappingNumBuf | expertNum*INT32               │  ← 映射数量表
└────────────────────────────────────────────────┘
```

### 4.4 多核并行策略

按行均匀分核: `batchPerCore = numToken / coreNum`，前 tailBatch 核多处理 1 个 token。

### 4.5 通信模式

无 SyncAll。各核独立按行处理。

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

22 个字段的 MoeFusedTopkTilingData（见 3.2 节）。CompileInfo 含 `totalCoreNum`, `ubSize`。

### 5.2 Workspace 布局

```
┌──────────────────────────────────────────────────────────────┐
│ workspacePerCore = expertNum * sizeof(float)                  │
│ 总 workspace = usedCoreNum * workspacePerCore                 │
├──────────────────────────────────────────────────────────────┤
│ System Workspace = 16 MB                                      │
└──────────────────────────────────────────────────────────────┘
```

### 5.3 完整数据流

```
  x [numToken, expertNum]   addNum [expertNum]
       │                         │
       └─────────────────────────┘
                   │
         对每行 loop (核间并行):
           sigmoidAdd = Sigmoid(x[loop]) + addNum   (Cast到FP32)
                   │
         CopyToWorkspace → CopyFromWorkspace(按分组对齐)
                   │
         GroupReduceSum: TopK(组内, topN) → ReduceSum → 组得分
                   │
         GroupTopk: Sort(组得分) → 选groupTopk组 → 广播掩码置0
                   │
         GatherSigmoid: Sort(全局) → Extract前topK → Gather Sigmoid值
                   │
         [isNorm] Norm: ReduceSum → 1/sum → Muls(sum) → Muls(scale)
                   │
         CopyOut: 输出 y[loop, k], indices[loop, k]
         [enableMapping] 查 mappingTable 替换索引
                   │
           ┌───────┴───────┐
           │               │
       y [numToken, topK] (FLOAT)
       indices [numToken, topK] (INT32)
```

---

## 7. 性能相关设计

1. **Workspace 中转**: UB 空间不足时，Sigmoid+Add 结果暂存到 workspace，按分组加载回 UB 并对齐，避免 UB 溢出。

2. **组内 TopN + ReduceSum**: 使用高阶 TopK API 选前 topN 个，再 ReduceSum 作为组得分，减少排序范围。

3. **广播掩码**: 组间 TopK 后，通过广播掩码 Mul 将未选中组置 0，替代逐元素条件判断。

4. **专家映射**: 物理专家到逻辑专家的映射查表在 CopyOut 阶段完成，不影响计算路径。

5. **输出固定 FLOAT**: 权重输出固定为 FLOAT 类型，避免低精度累加误差。

---

## 8. 依赖关系

```
op_host/moe_fused_topk_def.cpp
└── register/op_def_registry.h

op_host/moe_fused_topk_tiling.cpp
├── moe_fused_topk_tiling.h
│   ├── register/op_impl_registry.h
│   └── tiling_base/tiling_base.h
└── platform_ascendc::PlatformAscendC

op_host/op_api/aclnn_moe_fused_topk.cpp
├── aclnn_moe_fused_topk.h
├── moe_fused_topk.h
├── moe_fused_topk.cpp
└── aclnn_executor.h

op_kernel/moe_fused_topk.cpp
├── moe_fused_topk.h
└── kernel_operator.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| group_num | 必须 > 0 且 <= 256 |
| group_topk | 必须 <= groupNum |
| top_k | 必须 <= expertNum 且 expertNum <= 1024 |
| top_n | 必须 <= expertNum / groupNum |
| expertNum % groupNum | 必须整除 |
| activate_type | 仅支持 0=Sigmoid |
| maxMappingNum | 若 enableExpertMapping: <= 128 |
| 输出类型 | y 固定 FLOAT，不受输入类型影响 |
| 平台 | 仅 ascend910b/910_93，无 regbase |

---

## 10. 与同族变体的核心差异

| 维度 | moe_gating_top_k | moe_fused_topk（本算子） |
|------|-------------------|--------------------------|
| 归一化方式 | Sigmoid/Softmax 可选 | 仅 Sigmoid |
| 分组选择 | 组内排序→选组→多路归并 | 组内 TopN 求和→选组→广播掩码 |
| 偏置输入 | 可选 bias | 必选 addNum |
| 输出类型 | y 与 x 同类型 | y 固定 FLOAT |
| expert 映射 | 无 | 有（mappingTable） |
| TilingKey | 0/1/2（场景） | 0/1（映射开关） |
| Workspace | 固定 16MB | 按需分配（perCore * expertNum * 4） |
| 典型应用 | 训练+推理 | 推理（需专家映射场景） |
