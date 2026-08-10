声明：本文使用 [Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode) 许可协议，转载、引用或修改等操作请遵循此许可协议。

# 1 非量化 GMM 高阶 API Kernel 实现分析

## 1.1 文档目的

本文分析 Ascend 950（DAV_3510，代码目录为 `arch35`）上非量化 GroupedMatmul（以下简称 GMM）基于 Matmul 高阶 API 的 Kernel 实现，重点说明：

1. 外层 Kernel 如何通过模板组装 Scheduler、BlockMmad 和 Epilogue；
2. 内层 Kernel 如何解析每个 group 的 shape，并管理单 Tensor、多 Tensor 的 GM 地址；
3. `GroupedMatmulAswtScheduler` 如何完成跨 group 负载轮转、蛇形遍历和尾波拆分；
4. `MatmulMultiBlockBias<>` 如何落到 `AscendC::MatmulImpl`，以及实际的数据搬运与写回路径；
5. 当前实现满足的高阶 API 最佳实践、存在的风险和建议验证项。

本文聚焦 `GmmNoQuantAswt` 主计算路径，不分析 `EmptyTensor` 空 Tensor 处理逻辑。

## 1.2 涉及文件

主要分析以下四个文件：

| 层级 | 文件 | 主要职责 |
| --- | --- | --- |
| 外层 Kernel | [grouped_matmul_basic_kernel.h](../op_kernel/arch35/non_quant/grouped_matmul_basic_kernel.h) | 读取 tiling、选择数据类型/Layout，组装完整 Kernel 类型 |
| 内层 Kernel | [kernel_grouped_matmul.h](../../common/cgmct/kernel/kernel_grouped_matmul.h) | 遍历 group、恢复 M/N/K、管理 GM 地址并驱动 Scheduler 和 Matmul |
| Block Scheduler | [block_scheduler_grouped_matmul_aswt.h](../../common/cgmct/block/block_scheduler_grouped_matmul_aswt.h) | 基本块映射、蛇形遍历、跨核分配及 ASWT 尾波拆分 |
| Block MMAD | [block_mmad_multi_block_bias.h](../../common/cgmct/block/block_mmad_multi_block_bias.h) | 封装 `MatmulImpl` 的输入设置、Bias 设置与 `IterateAll` 写回 |

为还原完整模板实例化过程，本文还参考：

- [block_grouped_matmul_builder.h](../../common/cgmct/block/block_grouped_matmul_builder.h)
- [block_mmad_with_params.h](../../common/cgmct/block/block_mmad_with_params.h)
- [matmul_impl_traits.h](../../common/cgmct/block/matmul_impl_traits.h)
- [coord_utils.h](../../common/cgmct/utils/coord_utils.h)
- [copy_in_with_params.h](../../common/cgmct/tile/copy_in/copy_in_with_params.h)
- [copy_out_to_gm_with_params_v310.h](../../common/cgmct/tile/copy_out/copy_out_to_gm_with_params_v310.h)

# 2 总体架构

当前实现不是在 GMM Kernel 中手写完整的 GM→L1→L0→MMAD→Fixpipe 流水，而是采用“自定义 GMM Kernel/Scheduler + `MatmulImpl` 高阶 API + DAV3510 Copy Policy”的组合方式。

```text
GmmNoQuantAswt
  │
  ├─ 读取 GMMNoQuantBaseParams / TCubeTiling
  ├─ 根据 Layout 实例化 BlockGroupedMatmulBuilder
  ├─ 组装 KernelGroupedMatmul
  └─ op(params)
       │
       ├─ MatmulImpl.Init()                    一核一次
       ├─ for each group
       │    ├─ 解析 groupList
       │    ├─ 获取当前组 M/N/K
       │    ├─ 绑定当前组 A/B/Bias/C 地址
       │    └─ IterateOneGroup
       │          ├─ Scheduler 生成 (mTile,nTile)
       │          ├─ 计算普通尾块或 ASWT 拆分子块
       │          ├─ 计算 A/B/Bias/C offset
       │          ├─ SetOrgShape / SetSingleShape
       │          ├─ SetTensorA/B/Bias
       │          └─ IterateAll → Fixpipe 写回 Y
       └─ 当前实现未显式调用 MatmulImpl.End()
```

各层的职责边界为：

- 外层 Kernel 负责静态组件组装；
- 内层 Kernel 负责 GMM 动态 shape、地址和 group 循环；
- Scheduler 负责把每组 `(M,N)` 网格映射给物理 AIC；
- BlockMmad/`MatmulImpl` 负责单个 tile 内的 K 循环、片上 Buffer、流水同步和结果写回。

# 3 外层 Kernel：`GmmNoQuantAswt`

## 3.1 Tiling 数据读取

`GmmNoQuantAswt` 首先通过：

```cpp
GET_TILING_DATA_MEMBER(GMMNoQuantTilingData, gmmNoQuantParam, gmmBaseParams_, tiling);
GET_TILING_DATA_MEMBER(GMMNoQuantTilingData, mmTilingData, mmTilingData_, tiling);
```

分别读取：

- `GMMNoQuantBaseParams`：group 数量、分组方式、单/多 Tensor 标志、Bias 标志、ASWT 尾波参数、L2 Cache 策略等；
- `TCubeTiling`：`baseM/baseN/baseK`、L1 深度、step、double buffer 等 Matmul 高阶 API 参数。

`mmTilingData_` 是 Kernel 栈上的局部对象，其地址随后传给 `MatmulImpl::Init`，符合高阶 API 对 tiling 数据生命周期的要求，不存在将 tiling header 放入 UB/L1 后再强制转换指针的问题。

## 3.2 类型与 Layout 选择

外层定义：

```cpp
using AType = DTYPE_X;
using BType = DTYPE_X;
using CType = DTYPE_Y;
using BiasType = DTYPE_BIAS;
using LayoutC = layout::RowMajor;
using LayoutBias = layout::RowMajor;
```

A/B 的 Layout 由 kernel tiling key 和 weight format 决定。入口派发位于 [grouped_matmul_apt.cpp](../op_kernel/grouped_matmul_apt.cpp)：

| transpose_x | transpose_weight | weight format | LayoutA | LayoutB |
| --- | --- | --- | --- | --- |
| false | false | ND | `RowMajor` | `RowMajor` |
| false | false | FRACTAL_NZ | `RowMajor` | `Nz` |
| true | false | ND | `ColumnMajor` | `RowMajor` |
| true | false | FRACTAL_NZ | `ColumnMajor` | `Nz` |
| false | true | ND | `RowMajor` | `ColumnMajor` |
| false | true | FRACTAL_NZ | `RowMajor` | `Zn` |

`BlockGroupedMatmulBuilder` 进一步将这些 Layout 转换为：

```cpp
MatmulType<GM, formatA, AType, transA>
MatmulType<GM, formatB, BType, transB>
MatmulType<GM, ND, CType>
MatmulType<GM, ND, BiasType>
```

其中 `RowMajor/Nz` 对应非转置，`ColumnMajor/Zn` 对应转置；B 为 NZ 时仍使用 `CubeFormat::NZ`，转置语义由模板参数 `transB` 表达。

## 3.3 组件组装

外层组装的核心类型为：

```cpp
using BlockScheduler = GroupedMatmulAswtScheduler;

using BlockMmad = Block::BlockGroupedMatmulBuilder<
    AType, LayoutA,
    BType, LayoutB,
    CType, LayoutC,
    BiasType, LayoutBias,
    L1TileShape, L0TileShape,
    BlockScheduler,
    MatmulMultiBlockBias<>>;

using BlockEpilogue = Block::BlockEpilogueEmpty;

using GroupedMatmulKernel = Kernel::KernelGroupedMatmul<
    MatmulShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

组件选择的含义为：

- `GroupedMatmulAswtScheduler`：使用 GMM ASWT 分核策略；
- `MatmulMultiBlockBias<>`：使用 Matmul 高阶 API 的多基本块流水，并具备 Bias 能力；
- `BlockEpilogueEmpty`：没有 Vector 后处理，结果由 Cube/Fixpipe 直接写入 Y；
- `L1TileShape/L0TileShape` 均为 `_0`：实际 tile 参数由运行时 `TCubeTiling` 提供。

## 3.4 占位 `ProblemShape`

外层 `Params` 中传入：

```cpp
{1, 1, 1, 1}
```

这不是实际 GMM shape，而是通用接口要求的占位值。当前 `KernelGroupedMatmul::operator()` 不会将它写入成员 `problemShape_`，真实 M/N/K 在每个 group 中由 `SetMKN()` 重建。

因此该占位值在当前执行路径中不会影响计算，但存在接口脆弱性：如果后续重构开始使用通用 `CanImplement()`、`GetBlockNum()` 或 `params.problemShape`，就必须改为真实、可验证的 shape 描述。

# 4 内层 Kernel：动态 Group 管理

## 4.1 AIC/AIV 分工与 Matmul 初始化

`KernelGroupedMatmul::operator()` 首先执行：

```cpp
if ASCEND_IS_AIV {
    return;
}
```

说明主计算为纯 AIC 路径。随后：

1. 创建 `BlockMmadOp`；
2. 获取 `GetBlockIdx()` 和 `GetBlockNum()`；
3. 绑定 groupList GM 地址；
4. 调用一次 `blockMmadOp.Init(...)`；
5. 遍历所有 group。

一个 AIC 内的全部 group 复用同一个 `MatmulImpl` 实例，各组只刷新原始 shape、当前 tile shape 和 GM Tensor，这是合理且高效的使用方式。

## 4.2 groupList 解析

`GetSplitValueFromGroupList()` 支持两种 groupList 语义：

```text
groupListType = 0：groupList 保存累积结束位置（cumsum）
groupListType = 1：groupList 保存每组元素个数（count）
```

当为 cumsum 时：

```cpp
splitValue = groupList[groupIdx] - previousOffset;
previousOffset = groupList[groupIdx];
```

当为 count 时，直接读取当前值。

`groupType == -1` 表示不使用 groupList 覆盖 M/K，此时 `splitValue` 保持为 0。

## 4.3 每组 M/N/K 恢复

`SetMKN()` 从 x 和 weight 的 `ListTensorDesc/TensorDesc` 中获取最后两个逻辑维度，并结合 Layout 的转置属性恢复当前组 shape：

```cpp
M = transA ? xShape[1] : xShape[0];
K = transB ? wShape[1] : wShape[0];
N = transB ? wShape[0] : wShape[1];
```

随后按 groupType 覆盖分组轴：

```text
groupType = -1：不覆盖，使用各 TensorDesc 的实际 M/N/K
groupType =  0：M = splitValue
groupType =  2：K = splitValue
```

当 x、weight、y 全部是单 Tensor 时，`singleTensor=true`，只有 group 0 读取 TensorDesc；后续 group 复用非分组轴，只更新 M 或 K。这适用于 s-s-s 场景中“一个物理 Tensor 存放多个 group”的语义。

## 4.4 单 Tensor与多 Tensor的地址绑定

`InitGlobalBuffer()` 根据 `singleX/singleWeight/singleY` 选择两套地址语义：

- 标志为 0：输入是 TensorList，使用 `GetDataPtr(groupIdx)`；
- 标志为 1：所有 group 位于同一个 Tensor 中，使用 group 0 基址加累计偏移。

`baseOffset_` 五个字段的含义为：

| 下标 | 含义 | 每组增量 |
| --- | --- | --- |
| 0 | X 元素偏移 | `M × K` |
| 1 | Weight 元素偏移 | ND：`K × N`；NZ：对齐后的物理大小 |
| 2 | 累计 M 轴偏移 | `M`，当前主路径未直接使用 |
| 3 | Bias 元素偏移 | `N` |
| 4 | Y 元素偏移 | `M × N` |

Weight 为 NZ 时，其单 Tensor物理偏移为：

```text
transB = false：Align(N, C0) × Align(K, 16)
transB = true ：Align(N, 16) × Align(K, C0)

C0 = 32 / sizeof(BType)
```

这与 NZ 分形的物理存储粒度匹配。

## 4.5 `UpdateOffset()` 的执行顺序

每组 `Init()` 的顺序为：

```text
UpdateOffset() → 读取当前 groupList → SetMKN() → InitGlobalBuffer()
```

`UpdateOffset()` 在 `SetMKN()` 之前执行，实际含义是“使用 `problemShape_` 中保存的上一组 shape，提交上一组占用的物理空间”。成员 `problemShape_` 使用 `{}` 零初始化，因此：

- group 0：偏移增量为 0；
- group 1 开始：先累加 group 0；
- 后续依次累加前一组。

即使当前组因某一维为 0 而跳过计算，其 M/N/K 仍保存在 `problemShape_` 中；下一组开始时会按当前组真实存储大小推进相应的 X、Weight、Bias、Y 偏移。

# 5 跨 Group 任务轮转

## 5.1 `count` 的语义

每个 group 调度完成后返回：

```cpp
count = (count + currentGroupTileNum) % blockNum;
```

这里的 `count` 不是已处理 tile 总数，而是“所有前序 group 的 tile 总数对 blockNum 的余数”。下一组计算起始任务时使用：

```cpp
curBlock = curBlockIdx >= count ? curBlockIdx : curBlockIdx + blockNum;
```

从而把下一组的第一个 tile 轮转到 `count` 对应的物理核，避免每组都从 core 0 开始。

## 5.2 轮转示例

假设 `blockNum=4`：

```text
group 0 有 5 个 tile：
  core0 → tile0、tile4
  core1 → tile1
  core2 → tile2
  core3 → tile3
  count = 5 % 4 = 1

group 1 有 3 个 tile：
  core1 → tile0
  core2 → tile1
  core3 → tile2
```

该机制保留了跨 group 的调度相位，适合 group 数量多、单组 tile 数量少的场景。

# 6 ASWT Block Scheduler

## 6.1 基本块数量

每组调度的基本网格为：

```cpp
mTileNum = CeilDiv(M, baseM);
nTileNum = CeilDiv(N, baseN);
totalTileNum = mTileNum * nTileNum;
```

各物理核以 `blockNum` 为步长遍历逻辑 tile，因此在未开启 ASWT 拆分时，同一 tile 只会分配给一个核。

## 6.2 长度为 4 的蛇形窗口

Scheduler 设置 `WINDOW_LEN=4`，让 M tile 在一个长度最多为 4 的窗口中快速变化，N tile 在窗口之间推进：

```text
窗口 0：N 从左向右
窗口 1：N 从右向左
窗口 2：N 从左向右
……
```

其目标是让相邻任务更可能访问相同或邻近的 B 数据，改善 Cache 局部性。代码中“Reverse computation for even-numbered rows”的注释与实际条件不一致：实际为奇数 `rowIdx` 反转 N，仅属于注释问题，不影响映射正确性。

## 6.3 普通矩阵尾块

`GetBlockShape()` 首先处理 M/N 维的几何尾块：

```cpp
tailM = M % baseM == 0 ? baseM : M % baseM;
tailN = N % baseN == 0 ? baseN : N % baseN;
```

非边界 tile 返回完整 `baseM/baseN`，最后一个 M/N tile 返回实际剩余尺寸。

## 6.4 ASWT 最后一波拆分

当总 tile 数不能填满最后一个 AIC wave 时，Host tiling 可以给出：

```text
mTailCnt × nTailCnt
```

Scheduler 将最后一波中的每个大 tile 进一步拆成多个 M/N 子块：

```cpp
singleCoreMSplit = CeilDiv(blockShapeM, mTailCnt);
singleCoreNSplit = CeilDiv(blockShapeN, nTailCnt);
```

并通过：

```cpp
mSplitIdx = (blockIdx % tailCnt) % mTailCnt;
nSplitIdx = (blockIdx % tailCnt) / mTailCnt;
```

确定当前核负责的子块。

返回的 `singleShape` 同时携带：

```text
{实际子块 M, 实际子块 N, M 地址偏移, N 地址偏移}
```

后两个偏移被 `Coordinate` 用于计算 A/B/C/Bias 起始地址。因此拆分后的各核写入互不重叠的 Y 子区域，`IterateAll` 不需要 atomic add。

当 B 为 NZ 时，N 方向子块尺寸额外按 C0 对齐，以保证 NZ 起始地址和分形块边界合法。

## 6.5 Scheduler依赖的Host约束

Scheduler 隐含依赖：

```text
当 totalTileNum % blockNum == 0 时，mTailCnt × nTailCnt 必须等于 1。
```

否则最后一个完整 wave 也会进入尾波映射，造成部分 tile 未被覆盖。

当前 Host `CalcTailBasicBlock()` 只在 remainder 非 0 时增大 `mTailCnt/nTailCnt`，因此现有主路径满足该约束。但若该 Scheduler 被其他 Kernel 独立复用，建议在构造函数中增加防御性归一化。

# 7 BlockMmad与 Matmul 高阶 API

## 7.1 Matmul Policy

`MatmulMultiBlockBias<>` 的关键编译期参数为：

```text
ScheduleType              = KernelMultiBlock
SingleShape               = Shape<_0, _0, _0, _0>
ENABLE_INTRINSICS_CHECK   = false
ENABLE_SET_BIAS           = true
CONFIG                    = CONFIG_MDL
```

`ENABLE_SET_BIAS=true` 表示模板具有 Bias 能力，并不意味着每个算例都必须有 Bias。运行时是否调用 `SetBias()` 由 `gmmParams.hasBias` 控制，`TCubeTiling::isBias` 由 Host 设置，两者应保持一致。

## 7.2 实际 `MatmulImpl` 类型

`MatmulImplTraits` 先基于动态 Shape 和 `CONFIG_MDL` 生成 `STATIC_TILING`，再实例化：

```cpp
AscendC::MatmulImpl<
    AType,
    BType,
    CType,
    BiasType,
    STATIC_TILING,
    MatmulCallBackFunc<...>,
    MatmulPolicyCustom>
```

`MatmulPolicyCustom` 将默认的 CopyInA、CopyInB 和 CopyOut 替换为 Builder 指定的：

```cpp
Tile::TileCopy<Arch::DAV3510, Tile::CopyWithParams>
```

因此该路径虽然由 `MatmulImpl` 控制主流水，但 GM↔L1 和 L0C→GM 使用了 DAV3510 的自定义搬运实现。

## 7.3 API调用顺序

Matmul 对象一核初始化一次：

```cpp
blockMmadOp.Init(matmulTiling, GetTPipePtr());
```

每个 group 调用：

```cpp
blockMmadOp.SetOrgShape(M, N, K);
```

每个 tile 调用：

```cpp
blockMmadOp.SetSingleShape(curM, curN, K);
blockMmadOp.SetTensorA(aGlobal[aOffset], transA);
blockMmadOp.SetTensorB(bGlobal[bOffset], transB);
if (hasBias) {
    blockMmadOp.SetBias(biasGlobal[biasOffset]);
}
blockMmadOp.IterateAll(cGlobal[cOffset]);
```

调用语义为：

- `SetOrgShape`：刷新当前 group 的原始 leading dimension 和 NZ 地址解释；
- `SetSingleShape`：设置当前核负责的 M/N 子块和完整当前组 K；
- `IterateAll`：由高阶 API 在当前 tile 内完成 K 分段、L0 基本块遍历和 Fixpipe 写回。

## 7.4 实际数据通路

```text
A/B ND GM
  └─ DataCopy(Nd2NzParams) → L1 fractal

B NZ GM
  └─ NZ2NZ DataCopy → L1

L1
  └─ MatmulImpl 内部 L1→L0A/L0B
       └─ MMAD K 累加 → L0C
            └─ DAV3510 Fixpipe → RowMajor Y
```

ND 输入由 `CopyND2NZ()` 使用 `Nd2NzParams` 完成 GM→L1 转换；NZ weight 由 `CopyNZ2NZ()` 完成分形块搬运；非量化结果由 `FixpipeParamsC310<ROW_MAJOR>` 直接写回 Y，并在 FP32 累加输出为 FP16/BF16 时设置相应转换模式。

这里使用的是 Cube 专用的参数化 `DataCopy`/Fixpipe 路径，不是普通 GM↔UB 的简单长度版 `DataCopy`，因此不能机械套用“非 32 字节必须使用 `DataCopyPad`”的规则。尾块、分形对齐和 stride 已由专用 Copy Policy 处理。

## 7.5 Buffer与同步

四个主文件中没有显式声明 TQue/TBuf，也没有手写 `SetFlag/WaitFlag`，原因是：

- 当前为纯 AIC/Cube 路径，没有 Vector Epilogue 的 UB 流水；
- L1/L0A/L0B/L0C Buffer 由 `MatmulImpl` 管理；
- MTE、MMAD、Fixpipe 之间的事件同步由高阶 API 内部实现；
- L1 depth、step 和 double buffer 由 `TCubeTiling` 控制。

因此不应在外层随意增加 `PipeBarrier<PIPE_ALL>()`，否则可能破坏高阶 API 流水并带来明显性能损失。

# 8 实现优点

当前设计具有以下优点：

1. `MatmulImpl` 每核只初始化一次，跨 group 复用，避免重复构造高阶 API 状态；
2. 每个 group 都调用 `SetOrgShape()`，能够正确刷新动态 M/N/K 和 leading dimension；
3. 单 Tensor拼接与 TensorList 两种 GM 地址语义清晰分离；
4. ND/NZ 和 A/B 转置通过编译期 Layout 统一推导，运行时分支较少；
5. `count` 保留跨 group 调度相位，缓解 many-small-groups 对低编号核的集中使用；
6. ASWT 利用最后一波空闲核进一步拆分 tile，子块写回区域互不重叠；
7. 空 Epilogue 下直接由 Fixpipe 写回 Y，不需要 workspace 和 AIC/AIV 通信；
8. 小 M group 可按 tiling 指示关闭 B 的 L2 Cache，避免低复用 weight 污染缓存。

# 9 风险与改进建议

## 9.1 高优先级：未显式调用 `End()`

`BlockMmadWithParams` 已经提供：

```cpp
void End()
{
    matmul_.End();
}
```

但 `KernelGroupedMatmul::operator()` 在 group 循环结束后直接返回，没有调用 `blockMmadOp.End()`。

Matmul 高阶 API 的标准生命周期应为：

```text
Init → Set/Iterate → End
```

缺少 `End()` 可能导致内部 event/queue 未完全排空或资源状态残留，风险可能在重复 Kernel 调用、压力场景或 SDK 版本切换时表现为挂起或不稳定。即使某个版本的析构函数能够兜底，也不建议依赖隐式行为。

建议在所有 group 完成后增加：

```cpp
blockMmadOp.End();
```

## 9.2 中优先级：groupList 的 `GlobalTensor::GetValue()` 热点

当前每个 AIC、每个 group 都通过 `GetValue()` 读取 groupList。读取次数近似为：

```text
blockNum × groupNum
```

它读取的是少量控制元数据，通常不会造成直接正确性问题，但在 many-small-groups 场景下会形成所有 AIC 对相同 GM 标量地址的重复访问。此外，每核还可能重复解析 TensorDesc。

可评估以下优化方向：

1. groupNum 较小时随 tiling 下发 per-group shape/offset；
2. 预生成 per-group M/N/K 和 GM offset 表，避免 Kernel 重复解析；
3. 由少量核读取元数据后，通过适合 DAV3510 AIC 的共享机制分发。

不能简单将其替换为 UB `DataCopyPad`，因为当前是 Cube-only 路径，需要同时考虑 AIC 内存权限和同步成本。

## 9.3 中优先级：未显式调用 `SetSubBlockIdx(0)`

封装层已经暴露 `SetSubBlockIdx()`，但当前调用者在 `Init()` 前没有显式设置。默认值通常为 0，因此当前场景大概率可以工作；从高阶 API 合约和跨版本稳健性考虑，建议显式调用：

```cpp
blockMmadOp.SetSubBlockIdx(0);
blockMmadOp.Init(...);
```

## 9.4 中优先级：FP32 精度模式不显式

当前未调用 `SetHF32()`。FP16/BF16 路径不依赖 HF32 开关；FP32 ND 路径则依赖 `MatmulImpl` 的默认精度状态。

建议明确算子的 FP32 精度合同：

- 要求完整 FP32：`SetHF32(false, 0)`；
- 允许 HF32：Host tiling 的 mad type 与 Kernel `SetHF32` 必须配对。

如果后续显式开启 HF32，还应在结束前恢复状态。

## 9.5 Host约束依赖较强

外层直接构造 `Params` 并调用 Kernel，没有走通用 `CanImplement()`；而 `CanImplement()` 接收的仍是占位 `ProblemShape`，不能覆盖每个 group 的动态 shape。

运行时实际依赖 Host 保证：

- `groupNum` 位于合法范围；
- `baseM/baseN/baseK` 非 0；
- cumsum groupList 单调，count groupList 非负；
- M/N/K 能安全转换为高阶 API 使用的 `int32_t`；
- `gmmParams.hasBias` 与 `TCubeTiling::isBias` 一致；
- ASWT tail count 只在最后一轮不满时大于 1；
- NZ shape、C0 和物理地址满足对齐要求。

其中 `GetSplitValueFromGroupList()` 返回 `int64_t`，但 `SetMKN()` 的 `splitValue` 参数为 `int32_t`。若输入规范未限制 groupList 值不超过 `INT32_MAX`，存在收窄溢出风险。

建议将这些依赖整理为 Host tiling 与 Kernel 的明确接口合同，并为关键字段增加 Host 侧校验或调试断言。

## 9.6 扩展约束：FP32 FRACTAL_NZ

当前 NZ TensorDesc shape 恢复逻辑将两个分形因子都按 16 处理。对当前已注册主路径影响有限，因为 FP32 weight 对应 ND 格式；如果未来开放 FP32 FRACTAL_NZ，需要注意：

```text
FP32 C0 = 32 / sizeof(float) = 8
```

因此：

- B 非转置时，N 的恢复因子应为 C0；
- B 转置时，K 的恢复因子应为 C0；
- 另一个 Cube 外层分形因子仍为 16。

扩展 FP32 NZ 前，应将硬编码 16 改为按数据类型和物理轴分别计算。

# 10 建议验证矩阵

建议至少覆盖以下测试组合：

| 维度 | 测试项 |
| --- | --- |
| 数据类型 | FP16、BF16、FP32 ND |
| Weight格式 | ND、FRACTAL_NZ（已支持的数据类型） |
| 转置 | NN、TN、NT |
| Bias | 有 Bias、无 Bias |
| groupType | `-1`、切 M（0）、切 K（2） |
| Tensor组合 | s-s-s、s-m-s、m-m-s、m-m-m，以及 Host 声明支持的其他组合 |
| groupListType | cumsum、count |
| 调度 | 单 group、多 group、many-small-groups |
| 基本块 | M/N 整除、M 尾块、N 尾块、MN 双尾块 |
| ASWT | `tileNum < blockNum`、有 remainder、恰好整除 blockNum |
| 生命周期 | 同一 stream 连续多次启动 Kernel，验证缺少/补充 `End()` 的差异 |
| Cache | `weightNoL2Cache` 开启与关闭路径 |

跨 group 调度应特别检查：

1. 每个逻辑 tile 恰好执行一次；
2. `count != 0` 时下一组的起始核正确轮转；
3. ASWT 拆分后的各子块无重叠、无遗漏；
4. 单 Tensor模式下各 group 的 X/Weight/Bias/Y 累计偏移正确；
5. NZ weight 的 group 基址和 N 方向 split offset 满足 C0 对齐。

# 11 结论

非量化 GMM 当前实现的主体架构是合理的：自定义 Kernel 负责 GMM 特有的动态 group 管理，自定义 Scheduler 负责跨 group 和尾波负载均衡，而 K 方向流水、片上 Buffer 和 Fixpipe 写回交由 `MatmulImpl` 高阶 API 完成。

当前最需要优先确认和修正的是 `MatmulImpl::End()` 生命周期闭环；其次是 groupList/TensorDesc 的每核重复元数据访问，以及 `SetSubBlockIdx`、FP32 精度模式和 Host→Kernel 约束的显式化。完成这些补强后，该实现的可维护性、跨版本稳定性和 many-small-groups 性能都会更可控。
