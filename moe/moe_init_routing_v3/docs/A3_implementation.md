# MoeInitRoutingV3 — A3 实现梳理

> 适用产品：Atlas A2 训练/推理、Atlas A3 训练/推理系列产品（非 950PR/950DT）
> 代码入口：
> - tiling：`op_host/moe_init_routing_v3_tiling.cpp`
> - kernel：`op_kernel/moe_init_routing_v3.cpp`
> 注册：`REGISTER_OPS_TILING_TEMPLATE(MoeInitRoutingV3, MoeInitRountingV3TilingBase, 10000)`（默认模板，非 950 芯片走此分支）

## 1. 算子功能回顾

aclnnMoeInitRoutingV3 完成 MoE 的 routing 计算：

1. 对 `expertIdx`（shape `[N, K]`，int32）做 key-value sort，得到 `sortedExpertIdx` 与 `sortedRowIdx`。
2. 根据 `rowIdxType` 输出 `expandedRowIdxOut`：
   - `rowIdxType=1`（scatter）：直接输出 `sortedRowIdx`。
   - `rowIdxType=0`（gather）：输出其逆映射 `expandedRowIdxOut[sortedRowIdx[i]]=i`。
3. 在排序后的 expert 维度上统计直方图 `expertTokensCountOrCumsumOut`，支持 `cumsum / count / key_value` 三种模式。
4. 根据 `quantMode` 做可选量化（-1 非量化 / 0 静态 / 1 动态 INT8 量化；A3 模板只支持这三种）。
5. 按 `expandedRowIdxOut` 搬运 `x` 得到 `expandedXOut`，并按需输出 `expandedScaleOut`。
6. DropPad 模式（`dropPadMode=1`）下，每个 expert 处理的 token 数对齐到 `expertCapacity`：超出 drop，不足 pad 0。

## 2. Tiling 设计（`MoeInitRountingV3TilingBase`）

继承 `TilingBaseClass`，按 7 步流水实现。

### 2.1 平台与输入信息获取

- `GetPlatformInfo`：从 `MoeInitRoutingV3CompileInfo` 读取 `aivNum` 与 `ubSize`，写入 `aicoreParams_`。
- `GetShapeAttrsInfo`：读取 `x / expertIdx / 可选 scale / 可选 offset`，以及 9 个属性（`activeNum / expertCapacity / expertNum / dropPadMode / expertTokensNumType / expertTokensNumFlag / quantMode / activeExpertRangeList / rowIdxType`）。

### 2.2 校验

| 校验函数 | 关键约束 |
|---------|---------|
| `CheckAttr` | `dropPadMode ∈ {0,1}`；`expertNum>0`；`expert_range` 长度为 2 或 0；`expertStart<expertEnd≤expertNum`；`quantMode ∈ {-1,0,1}`；`rowIdxType ∈ {0,1}`；DropPad 强约束 `rowIdxType=0`、`expert_range=[0,expertNum]` |
| `CheckInputShape` | `x / expertIdx` 必须 2D 且 dim0 相等；`n = expertIdx.dim0`、`k = expertIdx.dim1`、`cols = x.dim1`；空 tensor（`n==0`）走特化路径；按 `quantMode` 校验 `scale / offset` shape |
| `CheckOutShape` | 按 DropPad/非 DropPad 校验 `expandedX`（2D vs 3D）、`expandedRowIdx`、`expertTokensCountOrCumsum`（1D 或 2D）、`expandedScale` |
| `CheckDtype` | `x ∈ {INT8,FP16,BF16,FLOAT}`；`expertIdx=INT32`；`expandedRowIdx=INT32`；`expertTokensCountOrCumsum=INT64`；`INT8 x` 不支持量化 |

### 2.3 核心切分策略

| 阶段 | 函数 | 作用 |
|------|------|------|
| 排序切分 | `Tiling4VBSCompute` | 单圈最多 `sortLoopMaxElement` 元素；超过则多核排序，核数取 `4^ceil(log4(needCoreNum))`，再约束到 `aivNum` |
| 中间归并 | `Tiling4VMSMiddleCompute` | 多核排序后用 4 路归并，核数 > 4 时需要中间归并 |
| SortOut | `Tiling4SortOutCompute` | 最终一次 `MrgSort` 输出，单次最多 `mrgSortListMaxElement=1504` |
| 直方图 | `Tiling4ExpertTokensCountCompute` | 按 `n*k / aivNum` 切分，按 UB 预算算 `maxElementsPerLoop` |
| DropPad SrcToDst | `Tiling4SrcToDstDropPadCompute` | 按 `totalLength_/aivNum` 分核，列方向按 UB 自适应 `perLoopCols/perLoopRows`；动态量化场景走 `Tiling4SrcToDstDropPadDynamicCompute` |
| 非量化 SrcToDst | `Tiling4SrcToDstCompute` | Dropless 场景的索引搬运切分 |
| GatherOut | `Tiling4GatherOutCompute` | 按 `totalLength_/aivNum` 分核；`perLoopCols` 自适应 UB，列循环数 `colsLoops` |

### 2.4 性能模板（特化）

A3 提供三类性能模板，通过 `IsFullLoad / IsPerformanceMode_X_1_7168_* / GetPerformanceMode / ComputeCountingSortTiling` 判定：

| 模板 | 触发条件（摘要） | TilingKey |
|------|----------------|-----------|
| **低时延性能模板**（PERFORMANCE） | x=`(1,7168)` BF16、expertIdx=`(1,8)`、scale=`(256,7168)`、`quantMode=1`、`expertNum=256`、`expert_range=[0,256]`、`expertTokensNumType=2` | `2000000` |
| **全载模板**（FullLoad，非量化/静态/动态） | `totalLength_ ≤ sortLoopMaxElement` 且非 DropPad，UB 装得下所有 `sort+rowIdx+expert+gather` buffer | 非量化 `2100000`、静态 `2200000`、动态 `2300000+ep*10000+smoothType*1000` |
| **大 batch 性能模板**（Gather+Sort） | `expertNum=256`、`expertEnd-expertStart≤32`、`N∈[384,8192]`、`k=8`、`quantMode=-1`、`rowIdxType=1`、`expertTokensNumType=1` | `1200000/1201000/1300000/1301000`（单核/多核 × gather/scatter） |
| **Counting Sort 优化** | 非全载、特定 shape 范围（`COUNTING_SORT_MIN_N ≤ n ≤ COUNTING_SORT_MAX_N`）且非动态量化 | `1100000/1100100/...` 复用大 batch tilingkey 体系 |

### 2.5 TilingKey 编码

通用模板：`TILINGKEY_BASE(1e6) + sortMode*1e5 + (quantMode+1)*1e4 + rowIdxType*1e3 + dropPadMode*1e2`，再叠加性能模板（2e6 段、3e6 空_tensor）。

### 2.6 Workspace

排序空间 + 多核同步空间 + scatter workspace + expertTokensCount workspace + 16KB 常驻；动态量化追加 `aivNum * cols * sizeof(float)`；DropPad 追加 `expertIdxValueWorkspace`；Counting Sort 按需覆盖。

## 3. Kernel 实现（`op_kernel/moe_init_routing_v3.cpp`）

### 3.1 任务类型

`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0)`；AIC 核直接 `return`，全部计算在 AIV 完成。

### 3.2 分发逻辑

按 `TILING_KEY_IS(...)` 串行匹配模板，命中即 `return`：

```
EMPTY_TENSOR(3e6)                     空 tensor：单核刷零 expertTokensCount
│
├─ 全载性能模板（2.x e6 段）
│   ├─ PERFORMANCE(2e6)               MoeV3FullLoad
│   ├─ DYNAMIC_QUANT_*_FULLLOAD       MoeV3FullLoadDynamicQuant<...,GATHER/SCATTER,NO_SCALE/SCALE_1H/SCALE_EH>
│   ├─ UNQUANTIZED_FULLLOAD(2.1e6)    MoeV3FullLoadUnquantized
│   └─ STATIC_QUANT_FULLLOAD(2.2e6)   MoeV3FullLoadStaticQuant
│
├─ Counting Sort 优化（拦截 GATHER_SORT 路径）
│   ├─ MoeV3FullLoadCutOriginT        FullLoad 计数排序（decode 小 T）
│   └─ MoeV3CutOriginT                非 FullLoad 计数排序（prefill 大 T）
│
├─ Gather→Sort 多核性能模板（1.2e6/1.3e6）
│   ├─ MoeSortActualExpert            单核 Gather→Sort
│   ├─ MoeGatherSortMultiCore         多核 Gather→Sort
│   └─ MoeSortMultiCorePerformance    最终归并
│
├─ 通用排序（1.0e6/1.1e6 段）
│   ├─ MoeSortOneCore
│   └─ MoeSortMultiCore
│
├─ ExpertTokensCount<CUMSUM/COUNT/KEY_VALUE_MODE>
│
├─ RowIdxGather / SrcToDst DropPad
│
└─ GatherOut 系列（按量化×DropPad×ep 组合）
    MoeGatherOut / MoeGatherOutDynamicQuant / MoeGatherOutQuant
    MoeGatherOutDroppad / MoeV3SrcToDstWithCapacity / MoeV3SrcToDstAndGather ...
```

### 3.3 关键头文件依赖

```
moe_v3_sort_one_core.h / moe_v3_sort_multi_core.h
moe_v3_mrgsort.h / moe_v3_mrgsort_out.h
moe_v3_full_load*.h（5 个全载模板）
moe_v3_gather_*.h / moe_v3_gather_out*.h / moe_v3_gather_out_droppad.h
moe_v3_cut_origin_t.h / moe_v3_full_load_cut_origin_t.h（计数排序）
moe_v3_row_idx_gather*.h / moe_v3_row_idx_gather_droppad*.h
moe_v3_expert_tokens_count.h
moe_v3_sort_actual_expert.h / moe_v3_gather_sort_multi_core.h / moe_v3_sort_multi_core_performance.h
```

## 4. 数据流总览

```
expertIdx ─► Sort(OneCore/MultiCore) ─► sortedExpertIdx + sortedRowIdx
                                          │
                                          ├─► (scatter 直出 / gather 反映射) ─► expandedRowIdxOut
                                          │
                                          ├─► ExpertTokensCount ─► expertTokensCountOrCumsumOut
                                          │
                                          └─► RowIdxGather / SrcToDst
                                                  │
x ─(可选 quant)─► expandedXOut ──────────────────┘
scale ─(可选透传/dynamicQuant)─► expandedScaleOut
```

## 5. 适用场景与限制

- 仅支持 `quantMode ∈ {-1, 0, 1}`。
- DropPad 仅支持 `rowIdxType=0`（gather）。
- 空输入（`n==0`）：仅刷零 `expertTokensCountOrCumsumOut`，其它输出为空 tensor。
- 性能模板有严格准入条件，未命中则进入通用模板（排序→直方图→搬运），性能可预期但延迟较高。
