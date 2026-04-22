# moe_gating_top_k 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeGatingTopK`
**算子类**: MoE 门控 TopK 选择
**功能描述**: MoE 模型中的门控 TopK 选择算子。对输入的 gating 分数执行归一化（Sigmoid 或 Softmax）、偏置相加、分组 TopK 选择及重归一化，输出选中的专家权重和索引。支持 DeepSeekV3 专用高性能路径、广义分组选择和不分组选择三种模式。

**支持平台**: ascend910b (arch32), ascend910_93 (arch35), ascend950 (regbase)
**支持数据类型**: FP16/BF16/FP32（输入），INT32（索引），FLOAT（归一化输出）

---

## 2. 目录结构与文件职责

```
moe_gating_top_k/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── moe_gating_top_k_def.cpp             # 算子定义
│   ├── moe_gating_top_k_tiling.h            # TilingData结构（19字段）
│   ├── moe_gating_top_k_tiling.cpp          # Tiling策略（3 TilingKey）
│   ├── moe_gating_top_k_tiling_base.cpp     # Tiling基类公共逻辑
│   ├── moe_gating_top_k_tiling_arch35.cpp   # arch35(regbase) Tiling策略
│   └── config/ {ascend910b, ascend910_93, ascend950}
├── op_kernel/
│   ├── moe_gating_top_k.cpp                 # membase Kernel入口（3分支）
│   ├── moe_gating_top_k_apt.cpp             # regbase Kernel入口
│   ├── common.h                             # 公共常量和工具
│   ├── moe_gating_top_k_e_k_fullload.h      # DeepSeekV3高性能路径
│   ├── moe_gating_top_k_without_group.h     # 不分组路径
│   ├── moe_gating_top_k_generalized.h       # 广义分组路径
│   └── arch35/
│       ├── common.h                         # arch35公共定义
│       └── moe_gating_top_k_regbase.h       # regbase kernel实现
├── op_graph/                                # 图模式算子原型
├── examples/
└── tests/ut/ {op_host, op_kernel}
```

注：本算子暂无独立 ACLNN API 发布接口，通过 GE 框架直接调用。

---

## 3. Host 侧分析

### 3.1 算子定义 (moe_gating_top_k_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | x | REQUIRED | FP16/BF16/FP32 | 2D: [rows, expertCount] |
| 输入 | bias | OPTIONAL | 同 x | 1D: [expertCount] |
| 输出 | y | REQUIRED | 同 x | 2D: [rows, k] |
| 输出 | expert_idx | REQUIRED | INT32 | 2D: [rows, k] |
| 输出 | out | REQUIRED | FLOAT | 2D: [rows, expertCount] |

**属性**:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| k | Int | 必填 | 选取的专家数量 |
| k_group | Int | 1 | 每组选取的组数 |
| group_count | Int | 1 | 专家分组数 |
| group_select_mode | Int | 0 | 组选择模式: 0=MAX, 1=SUM |
| renorm | Int | 0 | 是否重归一化: 0=否, 1=L1 |
| norm_type | Int | 0 | 归一化类型: 0=Softmax, 1=Sigmoid |
| out_flag | Bool | false | 是否输出完整归一化结果 |
| routed_scaling_factor | Float | 1.0 | 路由缩放因子 |
| eps | Float | 1e-20 | 防除零小量 |

### 3.2 Tiling 策略 (moe_gating_top_k_tiling.cpp)

**TilingData**（tiling.h）: 19 个字段，含核分配（`needCoreNum`）、行切分（`rowCount/perCoreRowCount/lastCoreRowCount`）、专家配置（`expertCount/k/kGroup/groupCount/perGroupExpertCount`）、控制标记（`addBias/renorm/normType/outFlag/groupSelectMode`）、归并参数（`vmsCount`）、缩放因子（`routedScalingFactor/eps`）、UB 临时空间（`calTmpBufUbSize`）。

**TilingKey 编码**（3 个 Key）:

| Key | 策略 | 条件 |
|-----|------|------|
| 0 | DeepSeekV3 高性能 | expertCount=256, groupCount=8, kGroup=4, k<=32, 有bias, groupSelectMode=SUM, renorm=NO, normType=SIGMOID, outFlag=false |
| 1 | 不分组 | groupCount==1 或 groupCount==expertCount 或 kGroup==groupCount |
| 2 | 广义分组 | 其余情况 |

**Tiling 注册优先级**: Membase=2000（高），Regbase=1000（低）。

**行切分策略**: 按行均匀分核，`perCoreRows = CeilDiv(rows, numBlocks)`，尾核处理余数。

### 3.3 Shape 推导 (moe_gating_top_k_infershape.cpp)

```
y          = [rows, k]          (dtype 同 x)
expert_idx = [rows, k]          (dtype INT32)
out        = [rows, expertCount] (dtype FLOAT)
```

校验: k 必须 > 0 且 <= expertCount。

### 3.4 ACLNN API

本算子无独立 ACLNN 发布接口，通过 GE 框架直接调用。

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化

**Membase 入口**（moe_gating_top_k.cpp）: 3 个 TilingKey 分支。

```
moe_gating_top_k() 入口
  +-- Key=0: MoeGatingTopKEKFullload   (DeepSeekV3高性能路径)
  +-- Key=1: MoeGatingTopKWithoutGroup (不分组路径)
  +-- Key=2: MoeGatingTopKGenerlized   (广义分组路径)
```

所有 Kernel 类将输入 Cast 为 FP32 计算。

### 4.2 核心计算流程

**数学公式**:
```
1. norm = Sigmoid(x) 或 Softmax(x)    (由 normType 决定)
2. score = norm + bias                 (可选)
3. topK_indices = GroupedTopK(score)   (分组TopK选择)
4. y = score[topK_indices]
5. 若 renorm=1: y = y / sum(y) * routedScalingFactor
```

**不分组路径**（MoeGatingTopKWithoutGroup）:
```
对每行 row:
  Cast2FP32 → Sigmoid/Softmax → Add(bias)
  → Sort32+Sort(全专家排序) → GatherMask(提取TopK)
  → Gather权重+Renorm+缩放
  → CopyOut
```

**广义分组路径**（MoeGatingTopKGenerlized）:
```
对每行 row:
  Cast2FP32 → Sigmoid/Softmax → Add(bias)
  → SortInGroup(组内Sort32排序)
  → SelectTopKGroupIndex(组选择: MAX取最大/SUM取前2之和)
  → SelectTopKExpertIdx(MrgSort多路归并)
  → CumputeActualTopKExpertId(修正专家索引,去除对齐偏移)
  → SelectTopKExpertScore+CopyOut
```

**DeepSeekV3 高性能路径**（MoeGatingTopKEKFullload）:
针对 expertCount=256, groupCount=8, kGroup=4 固定配置优化:
- 全量加载专家分数到 UB
- Sigmoid + Add Bias 固定流水线
- Sort32 组内排序 + PairReduceSum 计算组得分
- MrgSort4 四路归并（最多一轮）

### 4.3 数据结构与内存布局

**公共常量**（common.h）: 全 FP32 计算空间。

**UB 内存布局**（以 MoeGatingTopKWithoutGroup 为例）:
```
┌────────────────────────────────────────┐
│ xInQueue     (E*FP32*2)               │  <- 输入数据(双缓冲)
├────────────────────────────────────────┤
│ yOutQueue    (k*FP32)                  │  <- TopK权重输出
├────────────────────────────────────────┤
│ expertIdxOut (k*FP32)                  │  <- TopK索引输出
├────────────────────────────────────────┤
│ outOutQueue  (E*FP32)                  │  <- 完整归一化输出
├────────────────────────────────────────┤
│ biasBuf      (E*FP32*2)               │  <- 偏置数据
├────────────────────────────────────────┤
│ expertIdBuf  (E*INT32)                 │  <- 专家编号序列
├────────────────────────────────────────┤
│ xNormBuf     (E*FP32)                  │  <- Sigmoid/Softmax结果
├────────────────────────────────────────┤
│ xNormWithBias(E*FP32)                  │  <- 加偏置后结果
├────────────────────────────────────────┤
│ topKExpertId (k*INT32)                 │  <- TopK专家ID
├────────────────────────────────────────┤
│ calcTmpBuf   (E*FP32*8)               │  <- 计算临时空间
└────────────────────────────────────────┘
```

### 4.4 多核并行策略

按行均匀分核: `perCoreRows = CeilDiv(rows, numBlocks)`。各核独立处理自己的行段，无需核间通信。

### 4.5 通信模式

无 SyncAll。各核独立按行处理，无核间依赖。

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

19 个字段的 MoeGatingTopKTilingData（见 3.2 节）。CompileInfo 含 `totalCoreNum`, `ubSize`。

### 5.2 Workspace 布局

```
┌──────────────────────────────────────────────────────────────┐
│ System Workspace = 16 MB (固定大小)                            │
└──────────────────────────────────────────────────────────────┘
```

无额外算子级 workspace。

---

## 6. 完整数据流

```
  x [rows, expertCount]   bias [expertCount](可选)
       │                        │
       └────────────────────────┘
                   │
         对每行 i (核间并行):
           score = Sigmoid/Softmax(x[i]) + bias   (归一化+偏置)
                   │
         ┌─────────┴──────────┐
         │ 不分组(Key=1):      │ 分组(Key=0/2):
         │ Sort(全专家)        │ SortInGroup(组内排序)
         │ → TopK             │ → SelectTopKGroupIndex(组选择)
         │                    │ → MrgSort(多路归并选TopK)
         └─────────┬──────────┘
                   │
         y[i] = score[topK_indices]
         若 renorm: y[i] = y[i] / sum(y[i]) * routedScalingFactor
                   │
           ┌───────┴───────┐
           │               │
       y [rows, k]    expert_idx [rows, k]
                    out [rows, E](可选)
```

---

## 7. 性能相关设计

1. **DeepSeekV3 专用路径**: 针对固定配置（E=256, G=8, kG=4）全量加载 UB，MrgSort4 单轮归并，消除通用路径的分支开销。

2. **全 FP32 计算**: 所有路径将输入 Cast 为 FP32 执行 Sigmoid/Softmax 和排序，保证数值稳定性，最终 Cast 回原始精度。

3. **多路归并优化**: MrgSort4 四路归并，归并轮数 `vmsCount = ceil(log4(ceil(kGroup/4)))`，减少比较次数。

4. **组选择 SUM 模式**: PairReduceSum 取组内 Top2 之和作为组得分，避免全量 ReduceSum 开销。

5. **双缓冲 UB**: 不分组路径 xInQueue 为双缓冲（E*FP32*2），实现数据搬运与计算重叠。

---

## 8. 依赖关系

```
op_host/moe_gating_top_k_def.cpp
└── register/op_def_registry.h

op_host/moe_gating_top_k_tiling.cpp
├── moe_gating_top_k_tiling.h
│   ├── register/op_impl_registry.h
│   └── tiling_base/tiling_base.h
└── platform_ascendc::PlatformAscendC

op_host/moe_gating_top_k_tiling_arch35.cpp
├── moe_gating_top_k_tiling.h
└── tiling_base/tiling_base.h

op_kernel/moe_gating_top_k.cpp (membase)
├── moe_gating_top_k_e_k_fullload.h
├── moe_gating_top_k_without_group.h
├── moe_gating_top_k_generalized.h
├── common.h
└── kernel_operator.h

op_kernel/moe_gating_top_k_apt.cpp (regbase)
├── arch35/common.h
├── arch35/moe_gating_top_k_regbase.h
└── kernel_operator.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| k | 必须 > 0 且 <= expertCount |
| expertCount | 必须 <= 1024 |
| groupCount | 必须 > 0 且整除 expertCount |
| kGroup | 必须 <= groupCount |
| DeepSeekV3 路径 | 仅限 expertCount=256, groupCount=8, kGroup=4 的固定配置 |
| regbase | ascend950 使用 SoftMaxTiling 辅助 |
| 输出 out | dtype 固定为 FLOAT，不受输入类型影响 |
| norm_type | 0=Softmax, 1=Sigmoid |
| group_select_mode | 0=MAX（取每组最大值）, 1=SUM（取前2之和） |

---

## 10. 与同族变体的核心差异

| 维度 | moe_gating_top_k（本算子） | moe_gating_top_k_softmax | moe_gating_top_k_softmax_v2 |
|------|---------------------------|--------------------------|------------------------------|
| 归一化 | Sigmoid/Softmax 可选 | 仅 Softmax | 仅 Softmax |
| 分组选择 | 有（group 机制） | 无（全局 TopK） | 无（全局 TopK） |
| finished 支持 | 无 | 有（推理 EOS） | 有（推理 EOS） |
| renorm | 有（L1） | 无 | 有（Softmax 重归一化） |
| 输出 | y, expert_idx, out | y, expert_idx, row_idx | y, expert_idx, softmax_result |
| row_idx 输出 | 无 | 有 | 无 |
| 偏置输入 | 有 bias（可选） | 无 | 无 |
| 输入维度 | 仅 2D | 2D/3D | 2D/3D |
| TilingKey 范围 | 0/1/2（3个） | 0-17（18个） | 101010-103133（36+个） |
| 典型应用 | 训练（DeepSeekV3 等） | 推理（基础版） | 推理（增强版） |
