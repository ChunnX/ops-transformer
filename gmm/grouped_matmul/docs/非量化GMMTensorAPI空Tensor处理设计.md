# 非量化 GMM Tensor API 空 Tensor 处理设计

## 修订记录

| 版本 | 日期 | 修改内容 |
|---|---|---|
| 1.0 | 2026-08-11 | 分析非量化 GMM 切换 Tensor API 后，K 分组空 Tensor 的职责归属与实现方案 |

## 1. 概述

### 1.1 背景

非量化 GroupedMatmul（以下简称 GMM）正在将 arch35 Kernel 从原有实现切换为基于 Tensor API 和 Blaze 公共组件的实现。相关改动分别位于：

- [ops-transformer PR 9806](https://gitcode.com/cann/ops-transformer/pull/9806/diffs)：增加算子侧 Tensor API 适配入口，负责解析算子 TilingData、选择数据类型和 Layout，并组装公共 GMM Kernel 参数。
- [ops-tensor PR 227](https://gitcode.com/cann/ops-tensor/pull/227/diffs)：增加非量化 GMM 公共 Kernel、GMM Scheduler 适配和 DispatchPolicy，复用普通 MatMul 的 BlockMmad 实现。

上述改动已经覆盖 `M > 0 && N > 0 && K > 0` 的正常矩阵乘场景，暂未覆盖 K 分组中的空 Tensor 场景。

K 分组时，每个 group 的 K 由 `groupList` 决定。当某组 `K == 0` 时，其数学结果为：

```text
C[M, N] = A[M, 0] * B[0, N] = 0[M, N]
```

由于该组没有有效的 K tile，Cube 不会产生或写回输出，因此 Kernel 必须显式把该组对应的 Y 区域置零。

### 1.2 设计目标

本文档解决以下问题：

1. 说明旧实现中 `EmptyTensor` 虽然只在 Kernel 入口调用，但如何取得每个 group 的 shape 并完成处理。
2. 梳理 Tensor API 版本的代码复用边界。
3. 确定空 K 逻辑在 Tensor API Kernel 中的正确落点。
4. 给出可分阶段落地的输出清零方案。

### 1.3 适用范围

- 非量化 GMM。
- arch35 Kernel。
- `groupType == 2` 的 K 分组场景。
- 当前 K 分组不支持 bias，因此 `K == 0` 时输出为全零。

本文档不涉及量化 GMM 的空 Tensor 处理。

## 2. 旧版 EmptyTensor 调用机制

### 2.1 Kernel 入口分工

旧版入口位于：

```text
gmm/grouped_matmul/op_kernel/grouped_matmul_apt.cpp
```

在 x 转置、weight 不转置的模板分支中，入口按物理核类型执行不同逻辑：

```cpp
if ASCEND_IS_AIV {
    EmptyTensor<DTYPE_Y>(x, weight, groupList, y, tiling);
}
if ASCEND_IS_AIC {
    GmmNoQuantAswt<...>(x, weight, bias, groupList, y, tiling);
}
```

`EmptyTensor` 定义在：

```text
gmm/grouped_matmul/op_kernel/arch35/non_quant/grouped_matmul_basic_kernel.h
```

它是被 Kernel 入口内联调用的 `__aicore__` 函数，并不是另外启动的独立 Kernel。

因此一次 MIX Kernel 启动后的执行模型为：

```text
AIC 实例：进入 GmmNoQuantAswt，遍历并计算正常 group
AIV 实例：进入 EmptyTensor，再次遍历全部 group，查找 K == 0 的 group
```

### 2.2 为什么入口调用一次即可处理任意 group

`EmptyTensor` 内部包含完整的 group 循环：

```cpp
for (uint32_t groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
    splitValue = GetSplitValue(...);
    GetTensorShape(...);
    ...
    if (k == 0) {
        InitOutput(..., 0);
    }
}
```

它在循环中执行以下操作：

1. 根据 `groupListType` 解析当前组的 K。
   - 累计值模式：`K_i = groupList[i] - groupList[i - 1]`。
   - 分组大小模式：`K_i = groupList[i]`。
2. 从 x 和 weight 的 TensorList descriptor 中取得 M、N 等 shape 信息。
3. 判断当前组是否满足 `K == 0`。
4. 按 AIV 核切分当前组的 `M * N` 输出区域。
5. 调用 `InitOutput` 将该组输出写零。
6. 无论 K 是否为空，都累加 `yBaseOffset += M * N`，定位下一组输出。

所以旧实现不是“GMM 处理到某个 group 时回调 EmptyTensor”，而是 AIC 和 AIV 分别独立完成一次 group 遍历。

### 2.3 旧实现的局限

旧方案能够满足功能要求，但存在两套重复逻辑：

- `GmmNoQuantAswt` 解析一次 `groupList` 和各组 shape。
- `EmptyTensor` 再解析一次 `groupList` 和各组 shape。
- 正常计算和空组处理各自维护 Y 偏移。

切换为公共 Tensor API Kernel 后，如果仍在 `grouped_matmul_apt.cpp` 中保留 `EmptyTensor`，会造成算子适配层和公共 Kernel 各自维护一套 group 语义，增加 shape、offset 和单/多 Tensor 组合不一致的风险。

## 3. Tensor API 版本代码复用逻辑

### 3.1 分层职责

| 层次 | 主要职责 | 代表文件 |
|---|---|---|
| ops-transformer 适配层 | 解析算子 TilingData，选择类型和 Layout，组装公共 Kernel 参数 | `grouped_matmul_tensor_api_kernel.h` |
| ops-tensor GMM Kernel | 遍历 group、解析 groupList、取得当前 M/N/K、组织当前组计算 | `kernel_grouped_matmul_no_quant.h` |
| GMM Scheduler | 维护各组 A/B/Bias/C 偏移、跨组核游标以及 M/N block 映射 | `block_scheduler_gmm_swat_with_tail_split.h` |
| BlockMmad | 将当前 group 的一个普通 MatMul block 交给 Tensor API MatMul 实现 | `block_mmad_matmul_basic.h` |
| BlockEpilogue | MMAD 完成后的附加后处理；当前为 `BlockEpilogueEmpty` | Blaze Epilogue |

### 3.2 ops-transformer 适配层

`GmmNoQuantTensorApi` 负责：

- 从 `GMMNoQuantTilingData` 中取得 GMM 和 MatMul tiling 参数。
- 根据模板选择 `LayoutA`、`LayoutB`。
- 组装 `BlockMmad`、`BlockScheduler` 和 `GemmUniversal` 类型。
- 把 x、weight、bias、groupList、y 地址和 tiling 参数转换为公共 Kernel 的 `Params`。
- 调用 `GroupedMatmulKernel kernel(params)`。

该层不应该重新解析每个 group 的 shape，也不应该维护 group 输出偏移。

### 3.3 ops-tensor 公共 GMM Kernel

公共 Kernel 的 `operator()` 已经包含逐 group 处理过程：

```cpp
for (uint32_t groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
    const int64_t splitValue = GetSplitValue(groupIdx, gmmParams);
    SetProblemShape(params, groupIdx, splitValue);

    const int64_t problemM = Get<MNK_M>(problemShape_);
    const int64_t problemN = Get<MNK_N>(problemShape_);
    const int64_t problemK = Get<MNK_K>(problemShape_);

    scheduler.UpdateNoQuantGroup(groupParams);
    if (problemM > 0 && problemN > 0 && problemK > 0) {
        scheduler.UpdateNextProblem(problemShape_);
        ProcessSingleGroup(...);
    }
}
```

其中：

- `GetSplitValue` 已经使用 `int64_t` 解析 groupList。
- `SetProblemShape` 是首次完整获知当前 group 的 M、N、K 的位置。
- `UpdateNoQuantGroup` 在正常计算条件判断之前执行。
- `ProcessSingleGroup` 只处理三个维度均大于零的普通 MatMul。

这正是插入空 K 处理的自然位置。

### 3.4 Scheduler 的偏移复用

`UpdateNoQuantGroup` 为当前组生成 `NoQuantGroupInfo`，并更新下一组偏移：

```cpp
noQuantGroupInfo_.cOffset = params.singleY ? noQuantNextCOffset_ : 0;
...
noQuantNextAOffset_ += problemM * problemK;
noQuantNextBOffset_ += GetNoQuantWeightSize(problemN, problemK, params);
noQuantNextCOffset_ += problemM * problemN;
```

当 `K == 0` 时：

- A、B 的连续 Tensor 偏移增量为零。
- C 的偏移仍增加 `M * N`。
- `GetNoQuantGroupInfo().cOffset` 能正确指向当前组输出。

空 Tensor 处理应复用该偏移结果，避免继续维护旧版 `yBaseOffset`。

## 4. 推荐架构

### 4.1 空 K 逻辑的归属

空 K 逻辑应放在：

```text
ops-tensor_gmm_add/include/blaze/gemm/kernel/kernel_grouped_matmul_no_quant.h
```

即 `KernelGroupedMmadNoQuant` 对应的 `GemmUniversal` 特化中。

原因如下：

1. 该层已经拥有当前 `groupIdx`、M、N、K 和 groupList 解析结果。
2. 该层能够复用 Scheduler 计算出的 C 偏移。
3. 该层同时了解 AIC/AIV 角色，可以确保两者写入区域互斥。
4. ops-transformer 适配层仍只承担参数转换，便于其他算子复用 ops-tensor 公共实现。

### 4.2 不建议放置的位置

| 位置 | 不建议原因 |
|---|---|
| `grouped_matmul_apt.cpp` | 会继续保留第二套 groupList、shape 和 Y offset 解析逻辑 |
| `grouped_matmul_tensor_api_kernel.h` | 该文件是算子 tiling 到公共 Kernel 参数的适配层，不应承载逐组数学语义 |
| GMM Scheduler | Scheduler 可以维护偏移和 block 映射，但不应直接读写输出 Tensor |
| BlockMmad | `K == 0` 时不存在有效 MatMul tile，且 BlockMmad 应保持普通 MatMul 复用能力 |
| `BlockEpilogueEmpty` | Epilogue 只在 MMAD tile 完成后执行，无法覆盖没有 MMAD tile 的空 K 组 |

### 4.3 AIC/AIV 执行流程

推荐执行流程如下：

```text
GmmNoQuantTensorApi
  |
  +-- 组装 Params（AIC 和 AIV 都执行）
  |
  +-- GemmUniversal::operator()
        |
        +-- Init：初始化 groupList、初始 shape 等公共状态
        |
        +-- AIV：ProcessEmptyGroups
        |     |
        |     +-- 遍历全部 group
        |     +-- 解析当前 M/N/K
        |     +-- 更新 group A/B/C offset
        |     +-- 仅对 K == 0 的 C 区域写零
        |
        +-- AIC：ProcessNonEmptyGroups
              |
              +-- 遍历全部 group
              +-- 解析当前 M/N/K
              +-- 更新 group A/B/C offset
              +-- 仅对 M/N/K > 0 的 group 调用 BlockMmad
```

AIC 不写 K 为空的组，AIV 不写 K 非空的组，因此两者操作的输出区域互斥，不需要额外的 AIC/AIV 核间同步。Kernel 完成事件作为外部同步边界。

## 5. Kernel 修改方案

### 5.1 调整两层 AIV 提前返回

当前 Tensor API 代码存在两处 AIV 直接返回：

1. `grouped_matmul_tensor_api_kernel.h::GmmNoQuantTensorApi`。
2. `kernel_grouped_matmul_no_quant.h::GemmUniversal::operator()`。

仅修改其中一处不能使空 K 路径执行。推荐让外层适配函数在 AIC 和 AIV 上都完成 `Params` 组装，然后在公共 Kernel 内分流：

```cpp
__aicore__ inline void operator()(const Params& params)
{
    const uint32_t blockNum = AscendC::GetBlockNum();
    if (blockNum == 0 || params.gmmParams.groupNum == 0) {
        return;
    }

    Init(params);
    BlockScheduler scheduler(...);
    SetSchedulerTailAlign(scheduler);

    if ASCEND_IS_AIV {
        ProcessEmptyGroups(params, scheduler);
        return;
    }

    BlockMmad blockMmad;
    blockMmad.Init(params.mmParams);
    ProcessNonEmptyGroups(params, scheduler, blockMmad);
}
```

AIV 分支不应构造或初始化 `BlockMmad`。

### 5.2 抽取公共 group 元信息准备逻辑

AIC 和 AIV 都需要执行以下步骤：

```cpp
splitValue = GetSplitValue(...);
SetProblemShape(...);
groupParams = BuildNoQuantGroupParams(...);
scheduler.UpdateNoQuantGroup(groupParams);
```

建议将其抽取为内部辅助函数，例如 `PrepareGroup`，避免今后两个分支的 group shape 或 offset 语义出现差异。

可以返回以下元信息：

```cpp
struct GroupMeta {
    int64_t m;
    int64_t n;
    int64_t k;
    typename BlockScheduler::NoQuantGroupInfo groupInfo;
};
```

如果保留现有单循环结构，也可在循环中先完成公共元信息准备，再通过核类型选择对应处理函数。

### 5.3 ProcessEmptyGroups 参考实现

```cpp
__aicore__ inline void ProcessEmptyGroups(const Params& params, BlockScheduler& scheduler)
{
    if (params.gmmParams.groupType != GROUP_TYPE_SPLIT_K ||
        params.mmParams.groupListGmAddr == nullptr) {
        return;
    }

    for (uint32_t groupIdx = 0; groupIdx < params.gmmParams.groupNum; ++groupIdx) {
        const int64_t splitValue = GetSplitValue(groupIdx, params.gmmParams);
        SetProblemShape(params, groupIdx, splitValue);

        const int64_t m = AscendC::Te::Get<MNK_M>(problemShape_);
        const int64_t n = AscendC::Te::Get<MNK_N>(problemShape_);
        const int64_t k = AscendC::Te::Get<MNK_K>(problemShape_);

        const typename BlockScheduler::NoQuantGroupParams groupParams{
            problemShape_,
            params.gmmParams.singleX == 1,
            params.gmmParams.singleWeight == 1,
            params.gmmParams.singleY == 1,
            TRANS_B,
            WEIGHT_NZ,
            static_cast<uint32_t>(sizeof(BType))};

        // 必须在 K == 0 判断前执行，保证当前组及后续组 C offset 正确。
        scheduler.UpdateNoQuantGroup(groupParams);

        if (m <= 0 || n <= 0 || k != 0) {
            continue;
        }

        const typename BlockScheduler::NoQuantGroupInfo& groupInfo =
            scheduler.GetNoQuantGroupInfo();

        __gm__ CType* cPtr =
            GetTensorAddr<CType>(params.gmmParams.singleY == 0 ? groupIdx : 0,
                                 params.mmParams.cGmAddr) +
            groupInfo.cOffset;

        ZeroOutputByAiv(cPtr, m, n);
    }
}
```

上述代码为结构示意，`groupListGmAddr` 等字段名应以最终 `BlockMmad::Params` 定义为准。

### 5.4 正常计算分支

AIC 分支保持现有行为：

```cpp
scheduler.UpdateNoQuantGroup(groupParams);
if (problemM > 0 && problemN > 0 && problemK > 0) {
    scheduler.UpdateNextProblem(problemShape_);
    ...
    ProcessSingleGroup(params, scheduler, blockMmad, groupIdx);
}
```

空 K 组不要调用：

- `scheduler.UpdateNextProblem`。
- `ProcessSingleGroup`。
- `BlockMmad`。

`UpdateNextProblem` 负责正常 MatMul block 的物理核调度。空 K 清零采用独立的 AIV 输出切分，避免改变 AIC 正常计算的跨组物理核游标。

## 6. 输出清零实现

### 6.1 多核切分

建议沿 M 轴切分输出，保持与旧实现相同的连续行写回方式：

```cpp
const int64_t alignElements = 32 / sizeof(CType);
const int64_t rowsPerCore = AlignUp(CeilDiv(m, logicalCoreNum), alignElements);
const int64_t rowBegin = logicalCoreIdx * rowsPerCore;

if (rowBegin >= m) {
    return;
}

const int64_t currentRows = Min(rowsPerCore, m - rowBegin);
__gm__ CType* coreOutput = cPtr + rowBegin * n;
ZeroOutput(coreOutput, currentRows, n);
```

实现时应保留旧 MIX Kernel 已验证的逻辑 AIV 核编号映射，例如结合 `GetBlockIdx()` 和 `GetTaskRation()`，避免把物理 AIV 子核编号直接当作逻辑核编号。

如果未来 TilingKey 支持一个 AIC 对应多个 AIV，需要：

- 使用 `GetSubBlockIdx()` 只选择一个 AIV 执行清零；或
- 把 AIV sub-block 编入不重叠的数据切分。

即使重复写零在数值上幂等，也会造成重复带宽开销，不应依赖同址并发写入。

### 6.2 第一阶段：复用 InitOutput

第一阶段建议把已有且经过验证的 `InitOutput` 迁移到公共 Kernel 的 AIV 分支：

```cpp
InitOutput<CType>(outputGm, currentRows * n, static_cast<CType>(0));
```

优点：

- 与旧版行为一致。
- 迁移风险较低。
- 可以先完成职责下沉和 group offset 复用。
- 不需要在同一个修改中同时引入新的 UB 分块与尾块搬运逻辑。

该阶段的关键不是强制所有指令都改写为 Tensor API，而是保证空 K 语义由公共 GMM Kernel 负责。

### 6.3 第二阶段：UB 零块加 Tensor API CopyUB2GM

在当前 PR 固定的 Tensor API 版本中，没有发现直接对 GM Tensor 执行 scalar fill 的通用原语。因此完整 Tensor API 搬运方案建议采用：

```text
Vector Duplicate/寄存器 Duplicate：在 UB 中生成零块
                            |
                            v
                    Vector -> MTE3 同步
                            |
                            v
Tensor API MakeTensor + MakeCopy(CopyUB2GM) + Copy：UB -> GM
```

具体步骤：

1. 每个参与清零的 AIV 准备一个固定容量、32 字节对齐的 UB 零缓冲。
2. 使用 `AscendC::Duplicate` 或寄存器 `Duplicate + DataCopy` 把 UB 缓冲初始化为零。
3. 零缓冲只初始化一次，在多个空 group 之间复用。
4. 在 Vector 写 UB 与 MTE3 读取 UB 之间设置正确的 event/barrier。
5. 按 UB 容量循环构造 UB Tensor 和 GM Tensor。
6. 使用 `MakeCopy(CopyUB2GM{})` 和 `Copy` 写回。
7. 最后一个 chunk 以真实有效行数和列数构造 Tensor shape，不能覆盖相邻 group。

不建议使用以下方式生成零：

```cpp
Transform<MulScalar>(uninitializedUb, uninitializedUb, 0);
```

未初始化 UB 中可能包含 NaN，浮点 NaN 乘零仍可能得到 NaN，不能保证生成全零。

### 6.4 对齐和尾块要求

- UB 起始地址必须满足 32 字节对齐。
- UB 行 stride 应按 32 字节对齐，不能简单使用非对齐的 `N * sizeof(CType)` 作为 UB 下一行起点。
- GM 侧当前组输出仍按紧凑的 `M * N` 布局写回。
- 非对齐 N 和最后一个 chunk 必须通过 Tensor shape/有效长度准确描述。
- 所有 shape、GM offset 和数据长度继续使用 `int64_t`，避免从 Tensor descriptor 取得 shape 后过早窄化。
- 单次搬运的行数和字节数不得超过底层 Copy 指令字段范围，必要时继续分批。

## 7. 算子入口调整

Tensor API 版本最终接入 `grouped_matmul_apt.cpp` 后，入口只负责根据 A/B transpose 和 weight format 选择 Layout，例如：

```cpp
if constexpr (A_NOT_TRANS && B_NOT_TRANS) {
    GmmNoQuantTensorApi<NDExtLayoutPtn, WeightLayout>(...);
} else if constexpr (A_TRANS && B_NOT_TRANS) {
    GmmNoQuantTensorApi<DNExtLayoutPtn, WeightLayout>(...);
} else if constexpr (A_NOT_TRANS && B_TRANS) {
    GmmNoQuantTensorApi<NDExtLayoutPtn, TransWeightLayout>(...);
}
```

入口不再保留：

```cpp
if ASCEND_IS_AIV {
    EmptyTensor(...);
}
```

需要注意：K 分组模板必须继续使用能够启动 AIV 的 MIX TilingKey。若未来该路径变成纯 AIC Kernel，公共 Kernel 中的 AIV 空组分支不会执行，此时必须改为 AIC 清零或额外启动清零 Kernel。

## 8. 正确性约束

### 8.1 Bias 语义

当前 K 分组场景不支持 bias，所以：

```text
K == 0 => Y = 0
```

如果后续开放 K 分组 bias，数学语义应变为：

```text
Y[M, N] = Broadcast(Bias[N])
```

届时不能继续无条件调用清零函数，建议将内部接口命名为 `ProcessDegenerateGroup`，再根据 bias 能力选择清零或 bias broadcast。

### 8.2 AIC/AIV 写入范围

- AIC 只写 `M > 0 && N > 0 && K > 0` 的 group。
- AIV 只写 `M > 0 && N > 0 && K == 0` 的 group。
- `M == 0` 或 `N == 0` 时不形成任何 GM 写操作。
- AIV 多核之间必须按行或连续元素划分不重叠区间。

### 8.3 单 Tensor 与 TensorList

输出地址必须统一使用：

```cpp
GetTensorAddr<CType>(singleY == 0 ? groupIdx : 0, cGmAddr) + groupInfo.cOffset
```

- `singleY == 0`：每组 Y 是独立 Tensor，Scheduler 的组内 C 偏移为零。
- `singleY == 1`：所有组共享连续 Y，Scheduler 提供累计 C 偏移。

不能同时使用 TensorList 的分组地址和手工累计 `yBaseOffset`，否则可能重复偏移。

## 9. 验证方案

### 9.1 功能用例

| 场景 | groupList 示例 | 预期结果 |
|---|---|---|
| 累计值模式，中间组为空 | `[K0, K0, K0 + K2]` | 第二组输出全零，前后组正常计算 |
| 分组大小模式，中间组为空 | `[K0, 0, K2]` | 第二组输出全零 |
| 首组为空 | `[0, K1, K1 + K2]` | 首组输出全零，后续 offset 正确 |
| 尾组为空 | `[K0, K0 + K1, K0 + K1]` | 尾组输出全零 |
| 连续多个空组 | `[K0, K0, K0, K0 + K3]` | 多个空组分别写零且互不覆盖 |
| 全部组为空 | 所有组 K 均为零 | 全部有效 Y 区域为零 |
| M 为零 | `M == 0, K == 0` | 不写 GM，不越界 |
| N 为零 | `N == 0, K == 0` | 不写 GM，不越界 |

### 9.2 组合覆盖

- `groupListType`：累计值模式、分组大小模式。
- `singleX/singleWeight/singleY` 的有效组合。
- Y 为单个连续 Tensor 和 TensorList。
- weight 为 ND/NZ 等当前 K 分组支持的格式。
- 输出 dtype：fp16、bf16、fp32 等非量化路径支持类型。
- N 对 32 字节对齐和不对齐。
- `M < coreNum`、M 尾块和大 M 多轮 UB 搬运。
- 空组位于首部、中间、尾部以及连续出现。
- 空组相邻的非空组输出保持正确，验证 C offset 没有错位。

### 9.3 并行覆盖

- 当前 AIC:AIV=1:1 模板。
- 不同 launch block 数。
- 若后续支持 AIC:AIV=1:2，验证没有重复写同一输出区间。

## 10. 推荐落地顺序

1. 在 ops-tensor 公共非量化 GMM Kernel 中加入 AIV 空 group 分支。
2. 复用 `GetSplitValue`、`SetProblemShape` 和 `UpdateNoQuantGroup`，删除独立的 `yBaseOffset` 计算。
3. 第一阶段使用现有 `InitOutput` 完成功能迁移。
4. 将 `grouped_matmul_apt.cpp` 切换到 `GmmNoQuantTensorApi`，删除入口侧旧 `EmptyTensor` 调用。
5. 补齐空 K 白盒、UT 和 ST 用例。
6. 功能稳定后，将清零实现替换为 UB 零块加 Tensor API `CopyUB2GM`，并验证对齐、尾块和性能。

## 11. 结论

旧版 `EmptyTensor` 能在 Kernel 入口调用一次完成任意空 group 的处理，是因为它在每个 AIV 上独立遍历了全部 group。

Tensor API 版本已经把 groupList 解析、当前组 M/N/K 获取以及分组地址计算统一下沉到 ops-tensor 公共 GMM Kernel。空 K 是 GMM 的分组数学语义，也依赖当前 group 的 shape 和 C offset，因此应在该公共 Kernel 的逐 group 循环中处理。

最终职责应保持为：

```text
ops-transformer：类型、Layout 和 Tiling 参数适配
ops-tensor GMM Kernel：group 语义、空 K 判断和输出处理
GMM Scheduler：group offset 和 block 调度
BlockMmad：复用普通 Tensor API MatMul
```

这一方案能够消除入口侧的重复 group 遍历和偏移维护，使非量化 GMM 的正常计算与空 Tensor 语义都由同一个可复用公共 Kernel 统一保证。
