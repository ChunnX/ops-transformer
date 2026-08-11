# GMM Add 与非量化 GMM Kernel 实现差异分析

## 修订记录

| 版本 | 日期 | 修改人 | 修改内容 |
| --- | --- | --- | --- |
| V1.0 | 2026-08-11 | Codex | 首次创建，分析 Ascend 950 上 GMM Add 与非量化 GMM 的 Kernel 实现差异 |

## 1. 概述

### 1.1 文档目的

本文以 Ascend 950（DAV_3510，代码目录为 `arch35`）上的非量化 GMM 为基线，分析 GMM Add 的 Kernel 调用链、算子语义、动态 Shape、GM 地址管理、Scheduler 调度和 BlockMmad 写回方式，说明两者在复用相同 Block Scheduler 和 BlockMmad 时仍然需要不同内层 Kernel 的原因。

本文重点回答以下问题：

1. GMM Add 与非量化 GMM 的数学语义和输入组织有什么区别；
2. 两者的外层 Kernel 如何组装相同的 Scheduler 和 BlockMmad；
3. 两者的内层 Kernel 分别承担哪些动态 group 管理职责；
4. 相同 Scheduler 在两类算子中接收到的问题形态有何不同；
5. 相同 BlockMmad 如何通过不同写回参数实现普通输出与原地累加；
6. 后续切换 Tensor API 实现时，GMM Add 需要保留哪些专有逻辑。

本文聚焦主计算路径，不展开分析非量化 GMM 的 `EmptyTensor` 特殊处理。

### 1.2 基本信息

| 项目 | 非量化 GMM | GMM Add |
| --- | --- | --- |
| 算子类别 | Grouped MatMul | Grouped MatMul + In-place Add |
| 目标芯片 | Ascend 950PR / Ascend 950DT | Ascend 950PR / Ascend 950DT |
| 目标架构 | DAV_3510 / `arch35` | DAV_3510 / `arch35` |
| 输入类型 | 已注册的 FP16、BF16、FP32 等非量化组合 | X/Weight 为 FP16 或 BF16 |
| 输出类型 | 由算子规格决定 | YRef 固定为 FP32 |
| 分组类型 | 不分组、M 轴分组、K 轴分组 | 仅 K 轴分组 |
| Weight 格式 | ND、FRACTAL_NZ | 仅 ND |
| Bias | 可选 | 不支持，使用 YRef 原地累加 |

### 1.3 涉及文件

GMM Add 调用链：

| 层级 | 文件 | 主要职责 |
| --- | --- | --- |
| Kernel 入口 | [grouped_matmul_add.cpp](../op_kernel/grouped_matmul_add.cpp) | 根据架构和 TilingKey 选择 Kernel 实现 |
| 外层 Kernel | [grouped_matmul_add_basic_cgmct.h](../op_kernel/arch35/grouped_matmul_add_basic_cgmct.h) | 读取 tiling、固定 Layout、组装 Scheduler、BlockMmad 和内层 Kernel |
| 内层 Kernel | [kernel_grouped_matmul_add.h](../../common/cgmct/kernel/kernel_grouped_matmul_add.h) | 解析 K 分组、维护连续 GM 偏移、调度 tile 并执行原地累加 |

非量化 GMM 对照实现：

| 层级 | 文件 | 主要职责 |
| --- | --- | --- |
| 分析文档 | [非量化GMM高阶APIKernel实现分析.md](../../grouped_matmul/docs/非量化GMM高阶APIKernel实现分析.md) | 说明非量化 GMM 的组件组装、动态 group 管理和高阶 API 数据通路 |
| 外层 Kernel | [grouped_matmul_basic_kernel.h](../../grouped_matmul/op_kernel/arch35/non_quant/grouped_matmul_basic_kernel.h) | 读取非量化 GMM tiling 并组装通用 Kernel |
| 内层 Kernel | [kernel_grouped_matmul.h](../../common/cgmct/kernel/kernel_grouped_matmul.h) | 处理 TensorList、动态 M/N/K、Bias、ND/NZ 和多种分组方式 |

共享组件：

| 组件 | 文件 |
| --- | --- |
| Block Scheduler | [block_scheduler_grouped_matmul_aswt.h](../../common/cgmct/block/block_scheduler_grouped_matmul_aswt.h) |
| BlockMmad Builder | [block_grouped_matmul_builder.h](../../common/cgmct/block/block_grouped_matmul_builder.h) |
| BlockMmad | [block_mmad_multi_block_bias.h](../../common/cgmct/block/block_mmad_multi_block_bias.h) |

## 2. 核心结论

虽然两者使用相同的 `GroupedMatmulAswtScheduler`、`BlockGroupedMatmulBuilder`、`MatmulMultiBlockBias<>` 和 `BlockEpilogueEmpty`，但 Kernel 层不能直接共用，原因是两者的 group 语义和输出语义不同：

```text
非量化 GMM：
    通用动态 group 管理
    + TensorList/单 Tensor 地址解释
    + 动态 M/N/K
    + 可选 Bias
    + 覆盖写回 Y

GMM Add：
    固定 TN/ND/s-s-s/split-K 场景
    + 固定 M/N、动态 K
    + 连续 GM 地址推进
    + 无 Bias
    + 原地 atomic add 到 YRef
```

两者共享组件的职责边界为：

```text
内层 Kernel：解释算子语义，生成当前 group 和当前 tile 的 Shape/地址
Scheduler：把当前 group 的 M/N tile 映射到物理 AIC
BlockMmad：完成单个 tile 内的 K 循环、MMAD、同步和 Fixpipe 写回
```

Scheduler 和 BlockMmad 只处理已经被 Kernel 规范化后的局部问题，因此能够复用；动态 group 语义和最终写回语义仍然必须由不同 Kernel 表达。

## 3. 数学语义对比

### 3.1 非量化 GMM

非量化 GMM 每组计算为：

```text
Y_i = X_i × W_i + Bias_i
```

其中 Bias 可选。不同 group 的 M、N、K 可以不同，具体维度由 TensorDesc 和 groupList 共同确定。

### 3.2 GMM Add

GMM Add 每组计算为：

```text
YRef_i(out) = X_i × Weight_i + YRef_i(in)
```

当前 Ascend 950 路径仅支持 K 轴分组：

```text
M_i = M
N_i = N
K_i 随 group 变化
```

典型物理存储为：

```text
X      : [K_total, M]，transposeX = true
Weight : [K_total, N]，transposeWeight = false
YRef   : [groupNum, M, N]

K_total = K_0 + K_1 + ... + K_(groupNum-1)
```

执行结果为：

```text
YRef[0] += X[0:K_0]^T × Weight[0:K_0]
YRef[1] += X[K_0:K_0+K_1]^T × Weight[K_0:K_0+K_1]
...
```

不同 group 写入各自的 `YRef_i`，不是把所有 group 的结果归约到同一个 `[M,N]` 输出矩阵。

## 4. Kernel 入口差异

### 4.1 非量化 GMM 入口

非量化 GMM 根据 tiling key、A/B 转置属性和 Weight 格式派发多种 Layout 组合：

| transposeX | transposeWeight | Weight 格式 | LayoutA | LayoutB |
| --- | --- | --- | --- | --- |
| false | false | ND | `RowMajor` | `RowMajor` |
| false | false | FRACTAL_NZ | `RowMajor` | `Nz` |
| true | false | ND | `ColumnMajor` | `RowMajor` |
| true | false | FRACTAL_NZ | `ColumnMajor` | `Nz` |
| false | true | ND | `RowMajor` | `ColumnMajor` |
| false | true | FRACTAL_NZ | `RowMajor` | `Zn` |

### 4.2 GMM Add 入口

GMM Add 的 Ascend 950 路径只处理一个 TilingKey：

```cpp
if (TILING_KEY_IS(10000900009000090001UL)) { // split_k
    GroupedMatmulAdd::GmmAddCgmct<layout::ColumnMajor, layout::RowMajor>(
        x, weight, groupList, y, tiling);
}
```

该模板实例固定表达：

```text
LayoutA        = ColumnMajor
LayoutB        = RowMajor
transposeX     = true
transposeWeight= false
groupType      = 2（split-K）
A/B format     = ND
```

Host tiling 同样会拒绝其他组合。因此，GMM Add 可以视为通用非量化 GMM 的固定 TN、ND、split-K 子场景，再叠加 YRef 原地累加语义。

### 4.3 输入/输出地址契约

顶层 Kernel 参数同时包含输入 `y` 和输出 `yRef`：

```cpp
grouped_matmul_add(x, weight, groupList, y, yRef, workspace, tiling)
```

Ascend 950 路径实际把输入参数 `y` 传给 `GmmAddCgmct`，输出参数 `yRef` 没有直接传入内层 Kernel。内层 Kernel 随后在该地址上执行 atomic add。

因此当前实现依赖算子的原地/同址契约：输入 YRef 与最终输出引用相同的 GM 数据。后续重构或切换 Tensor API 时，需要继续明确并验证该地址别名关系。

## 5. 外层 Kernel 组装差异

### 5.1 相同部分

两者都组装以下组件：

```cpp
using BlockScheduler = GroupedMatmulAswtScheduler;

using BlockMmad = Block::BlockGroupedMatmulBuilder<
    ...,
    BlockScheduler,
    MatmulMultiBlockBias<>>;

using BlockEpilogue = Block::BlockEpilogueEmpty;
```

其共同含义为：

- Scheduler 使用 GMM ASWT 多核调度；
- BlockMmad 使用 Matmul 高阶 API 的多基本块流水；
- `MatmulMultiBlockBias<>` 具备设置 Bias 的能力；
- 空 Epilogue 表示不使用 Vector 后处理，结果由 Cube/Fixpipe 直接写回 GM；
- L1/L0 TileShape 使用动态 `_0`，实际参数来自 `TCubeTiling`。

### 5.2 不同部分

| 项目 | 非量化 GMM | GMM Add |
| --- | --- | --- |
| 内层 Kernel | `KernelGroupedMatmul` | `KernelGroupedMatmulAdd` |
| LayoutA | RowMajor/ColumnMajor | 固定 ColumnMajor |
| LayoutB | RowMajor/ColumnMajor/Nz/Zn | 固定 RowMajor |
| Bias 类型 | `DTYPE_BIAS` | 使用 `CType` 作为模板占位 |
| Bias GM 地址 | 实际 Bias 地址 | `nullptr` |
| GMM 参数 | groupType、singleX/W/Y、Bias、Cache、尾块等 | groupNum、groupListType、尾块参数 |
| 输出语义 | 覆盖写回 | 原地累加 |

GMM Add 构造 BlockMmad 参数时使用：

```cpp
{x, weight, y, nullptr, groupList}
```

虽然 Builder 仍选用具有 Bias 能力的 `MatmulMultiBlockBias<>`，但 Bias 地址为空且 Host 设置 `TCubeTiling::isBias=0`，运行时不会执行 Bias 分支。`BiasType=CType` 仅用于满足 Builder 模板参数要求。

## 6. 动态 Shape 管理差异

### 6.1 非量化 GMM：每组恢复 M/N/K

非量化 GMM 的每个 group 可能具有不同 Shape。内层 Kernel 需要：

1. 解析 `ListTensorDesc`；
2. 获取当前 X/Weight 的 TensorDesc；
3. 读取最后两个逻辑维度；
4. 根据 `transA/transB` 恢复 M/N/K；
5. 根据 `groupType` 使用 groupList 覆盖 M 或 K。

其逻辑为：

```text
M = transA ? xShape[1] : xShape[0]
K = transB ? wShape[1] : wShape[0]
N = transB ? wShape[0] : wShape[1]

groupType = -1：不覆盖
groupType =  0：M = splitValue
groupType =  2：K = splitValue
```

### 6.2 GMM Add：固定 M/N，仅更新 K

GMM Add 不解析 TensorDesc，而是在 Kernel 初始化时直接读取：

```cpp
M = matmulTiling->M;
N = matmulTiling->N;
```

每个 group 只执行：

```cpp
K = GetSplitValueFromGroupList(groupIdx, groupListType);
```

这是因为 Host 已保证：

- X 和 Weight 均为单个二维 Tensor；
- M、N 对所有 group 相同；
- 仅 K 根据 groupList 变化。

这种专用化避免了所有 AIC 对 TensorDesc 的重复解析。

## 7. groupList 解析差异

两者都支持两种 groupList 语义：

```text
groupListType = 0：保存分组轴结束位置的累积和
groupListType = 1：保存每组长度
```

当 `groupListType=0` 时：

```cpp
splitValue = groupList[groupIdx] - preOffset;
preOffset = groupList[groupIdx];
```

当 `groupListType=1` 时：

```cpp
splitValue = groupList[groupIdx];
```

主要区别为：

- 非量化 GMM 在 `groupType=-1` 时不使用 groupList，并可用 splitValue 覆盖 M 或 K；
- GMM Add 始终读取 groupList，且 splitValue 只能表示当前组 K。

当 GMM Add 的 `K_i<=0` 时，当前组跳过 Matmul。下一组仍会推进 `C offset += M×N`，因此零 K group 对应的 YRef 分片保持输入值，符合：

```text
X_i × Weight_i = 0
YRef_i(out) = YRef_i(in)
```

## 8. GM 地址管理差异

### 8.1 非量化 GMM

非量化 GMM 同时支持 TensorList 和连续单 Tensor：

```text
singleX/Weight/Y = 0：使用 ListTensorDesc.GetDataPtr(groupIdx)
singleX/Weight/Y = 1：使用 group 0 基址 + 累计偏移
```

`baseOffset_` 包含五类状态：

| 下标 | 含义 | 每组增量 |
| --- | --- | --- |
| 0 | X 元素偏移 | `M×K` |
| 1 | Weight 元素偏移 | ND 为 `K×N`，NZ 为对齐后的物理大小 |
| 2 | M 轴累计偏移 | `M` |
| 3 | Bias 元素偏移 | `N` |
| 4 | Y 元素偏移 | `M×N` |

Weight 为 NZ 时还需要根据 C0 和转置状态计算物理空间大小。

### 8.2 GMM Add

GMM Add 只支持连续单 Tensor，直接把 GM 地址解释为普通数组：

```cpp
aGlobal = x      + aBaseOffset;
bGlobal = weight + bBaseOffset;
cGlobal = y      + cBaseOffset;
```

`baseOffset_` 只有三个字段：

| 下标 | 含义 | 每组增量 |
| --- | --- | --- |
| 0 | A 元素偏移 | `M×K_i` |
| 1 | B 元素偏移 | `N×K_i` |
| 2 | YRef 元素偏移 | `M×N` |

对于 group `i+1`：

```text
aBaseOffset += M × K_i
bBaseOffset += N × K_i
cBaseOffset += M × N
```

A 的物理布局为 `[K_total,M]`，但每组占用元素数仍为 `M×K_i`。group 内部的转置 tile 地址由 `Coordinate<transA=true,...>` 计算。

## 9. Scheduler 复用及调用差异

### 9.1 共享调度能力

两者使用相同的 `BlockSchedulerGroupedMatmulAswt`，共享以下逻辑：

- `ceil(M/baseM) × ceil(N/baseN)` 基本 tile 网格；
- 长度为 4 的 M 方向滑窗；
- 奇偶窗口反转 N 的蛇形遍历；
- M/N 几何尾块；
- ASWT 最后一波拆分；
- 使用 `count` 保留跨 group 的物理核轮转相位。

Scheduler 只负责 M/N 分核，K 由 BlockMmad 内部循环处理。因此，GMM Add 每组 K 不同不需要修改 Scheduler。

### 9.2 输入问题形态不同

| 项目 | 非量化 GMM | GMM Add |
| --- | --- | --- |
| 每组 M | 可变化 | 固定 |
| 每组 N | 可变化 | 固定 |
| 每组 K | 可变化 | 可变化 |
| 每组 tile 网格 | 可能变化 | 通常相同 |
| Weight NZ | 支持 | 不支持 |
| N 尾块 C0 对齐 | NZ 时需要 | 不需要 |
| tailSplit | 特定 split-M 场景可能关闭 | 固定传入 `true` |

两者都采用：

```cpp
count = (count + currentGroupTileNum) % blockNum;
```

下一组根据 count 轮转起始核，避免 many-small-groups 始终集中在低编号 AIC。GMM Add 的 M/N 固定，因此各组基础 tile 数通常一致；非量化 GMM 的 tile 数则可能随每组 Shape 变化。

Host 侧两者均只在 `groupNum==1` 时计算大于 1 的 `mTailCnt/nTailCnt`，避免多 group 场景中尾波拆分破坏跨组 count 语义。

## 10. BlockMmad 复用及写回差异

### 10.1 共同调用流程

两者均执行：

```text
BlockMmad.Init()          每个 AIC 一次
for each group:
    SetOrgShape()         每组一次
    for each tile:
        SetSingleShape()
        SetTensorA()
        SetTensorB()
        IterateAll()
```

其中：

- `SetOrgShape(M,N,K)` 刷新当前 group 的原始 Shape 和 leading dimension；
- `SetSingleShape(curM,curN,K)` 设置当前核处理的 M/N 子块和完整当前组 K；
- BlockMmad/`MatmulImpl` 负责 K 方向 L1/L0 循环、MMAD、Buffer 和事件同步；
- `BlockEpilogueEmpty` 下结果由 Fixpipe 直接写回 GM。

### 10.2 非量化 GMM：覆盖写回和可选 Bias

非量化 GMM 每个 tile 调用：

```cpp
blockMmadOp.SetSingleShape(curM, curN, K);
blockMmadOp.SetTensorA(aGlobal[aOffset], transA);
blockMmadOp.SetTensorB(bGlobal[bOffset], transB);
if (hasBias) {
    blockMmadOp.SetBias(biasGlobal[biasOffset]);
}
blockMmadOp.IterateAll(cGlobal[cOffset]);
```

`IterateAll` 使用默认 `enAtomic=0`，Fixpipe 覆盖写回当前 Y tile。

### 10.3 GMM Add：atomic add 写回

GMM Add 每个 tile 调用：

```cpp
blockMmadOp.SetSingleShape(curM, curN, K);
blockMmadOp.SetTensorA(aGlobal[aOffset], transA);
blockMmadOp.SetTensorB(bGlobal[bOffset], transB);
blockMmadOp.IterateAll(cGlobal[cOffset], 1);
```

第二个参数 `1` 开启 atomic add，其目标语义为：

```text
GM[YRef tile] = GM[YRef tile] + 当前 L0C Matmul 结果
```

需要特别说明：

- atomic add 不是因为不同 group 写同一个输出；
- atomic add 也不是因为 Scheduler 把一个输出元素分给多个核；
- ASWT 切分后的 M/N 子块仍然互不重叠；
- atomic add 的作用是保留 YRef 原值并加入 Matmul 结果。

GMM Add 不能把 YRef 当作普通 Bias 处理。Bias 通常是 `[1,N]` 广播数据，而 YRef 是完整的 `[M,N]` 矩阵。

## 11. Cache 策略差异

非量化 GMM 根据：

```text
weightNoL2Cache == 1 && baseM > currentM
```

动态设置 B 的 L2 Cache Hint，避免小 M group 的低复用 Weight 污染 Cache。

GMM Add 没有 `weightNoL2Cache` 参数，而是在初始化时执行：

```cpp
cGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
```

该策略作用于 FP32 原地累加输出。后续切换实现时，需要分别评估 Weight 复用策略和 atomic 输出的 Cache 行为，不能直接套用普通非量化 GMM 的 B Cache Hint。

## 12. 数据类型与格式范围差异

| 能力 | 非量化 GMM | GMM Add |
| --- | --- | --- |
| X/Weight dtype | FP16、BF16、FP32 等已注册组合 | FP16、BF16 |
| 输出 dtype | 随算子规格 | FP32 |
| X 格式 | ND | ND |
| Weight 格式 | ND、FRACTAL_NZ | ND |
| X 转置 | 支持 true/false 组合 | 固定 true |
| Weight 转置 | 支持 true/false 组合 | 固定 false |
| Bias | 可选 | 无 |
| TensorList | 支持 | 不支持 |
| groupType | -1、0、2 | 仅 2 |

因此 GMM Add 不涉及：

- NZ TensorDesc Shape 恢复；
- NZ Weight 物理偏移和 C0 对齐；
- Bias 地址与 Bias dtype；
- FP32 输入的 HF32 计算模式；
- 多 Tensor列表的地址选择。

## 13. 共享风险与专有风险

### 13.1 共享风险

由于两者仍复用同一套 CGMCT Matmul 高阶 API 封装，以下风险同时存在：

1. `blockMmadOp.Init()` 后没有显式调用 `blockMmadOp.End()`；
2. 没有在 `Init()` 前显式调用 `SetSubBlockIdx(0)`；
3. 每个 AIC、每个 group 都通过 GM `GetValue()` 读取 groupList；
4. 外层 Params 中的 `{1,1,1,1}` 是占位 ProblemShape；
5. 正确性依赖 Host 保证 baseM/baseN/baseK、groupList 和 tail count 合法。

### 13.2 GMM Add 已规避的通用 GMM 开销或风险

- 不解析 TensorDesc，减少每核、每组的控制元数据访问；
- 不支持 NZ，避免分形 Shape 和物理偏移计算；
- splitValue 直接以 `int64_t` 写入 K，没有通用 GMM `int64_t` 到 `int32_t SetMKN()` 的收窄问题；
- 固定 M/N，简化每组 Shape 状态更新。

### 13.3 GMM Add 专有风险

1. 正确性依赖输入 Y 和输出 YRef 的原地地址契约；
2. 写回必须保持 atomic add 语义，不能退化为普通覆盖写；
3. YRef 是完整 M×N 矩阵，不能使用 Bias 接口替代；
4. atomic 输出的 Cache 策略需要与目标平台的 Fixpipe/GM 行为匹配；
5. 后续若扩展非原地输出，需要显式处理输入 Y 到输出 YRef 的初始化或融合读写。

## 14. 对 Tensor API 切换的启示

GMM Add 可以复用非量化 GMM Tensor API 切换方案中的以下能力：

```text
Tensor API BlockMmad 主计算流水
GMM SWAT Scheduler
跨 group 的 count/物理核轮转
groupList cumsum/count 解析
M/N Tensor Slice
```

但不能直接复用当前非量化 GMM Tensor API Kernel，必须保留以下专有逻辑：

1. M/N 从 tiling 固定获取，只让 K 随 groupList 变化；
2. A/B/Y 是连续 GM Tensor，不需要 TensorList 描述解析；
3. 每组输出基址固定推进 `M×N`；
4. A 固定使用转置 ND Layout，B 固定使用非转置 ND Layout；
5. BlockMmad 写回必须支持 atomic add，完成 `YRef += Matmul`；
6. 保留或重新评估 C 的 L2 Cache Hint 策略。

其中 atomic add 是切换工作的关键边界。如果新的 Tensor API `BlockMmadMatmulBasic` 只支持覆盖写回，则需要扩展其写回接口、CopyOut Policy 或新增 GMM Add 专用 DispatchPolicy。仅在外层 Kernel 中构造 Tensor Slice，无法表达 YRef 原地累加语义。

## 15. 最终总结

GMM Add 可以概括为：

```text
非量化 GMM 的固定 TN/ND/s-s-s/split-K 子集
+ 完整 YRef 矩阵的原地 atomic add
- TensorList、动态 M/N、Bias、NZ 和多 Layout 能力
```

二者能够复用同一 Scheduler，是因为 Scheduler 只关心当前 group 的 M/N tile 与物理核映射；能够复用同一 BlockMmad，是因为 BlockMmad 只关心当前 tile 的 Matmul 流水，并通过 `enAtomic` 参数选择覆盖写回或累加写回。

真正决定两类算子差异的是内层 Kernel：非量化 GMM Kernel 负责通用动态 group 解释，GMM Add Kernel 则负责固定 split-K 连续存储和 YRef 原地累加。因此，后续 Tensor API 切换应继续复用调度和单 tile 计算能力，同时为 GMM Add 保留专用 Kernel 语义。
