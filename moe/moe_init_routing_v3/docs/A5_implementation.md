# MoeInitRoutingV3 — A5 实现梳理

> 适用产品：Ascend 950PR / Ascend 950DT（RegBase 3510 架构，`__NPU_ARCH__ == 3510`）
> 代码入口：
> - tiling：`op_host/moe_init_routing_v3_tiling_arch35.cpp`
> - kernel：`op_kernel/moe_init_routing_v3_apt.cpp`
> 注册：`IsCapable()` 判定 `Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)`，仅 3510 系列芯片进入此模板。

## 1. 算子功能（与 A3 一致）

参见 `aclnnMoeInitRoutingV3.md`。A5 与 A3 实现的是同一个算子接口，**差异主要在于**：

1. 适配 RegBase 3510 架构（新增大量量化模式）。
2. TilingKey 编码体系完全不同（1e7 起 vs 1e6 起）。
3. Kernel 子模块路径前缀 `arch35/`，模块数量显著增加。
4. 多了一层 SIMT DCache 预留与 overflow 寄存器保存恢复逻辑。

## 2. 支持的量化模式（关键扩展）

A5 在 A3 的 `{-1, 0, 1}` 基础上扩展到 14 种量化模式：

| quantMode | 含义 | expandedXOut dtype |
|-----------|------|-------------------|
| -1 | 非量化 | 同 x（含 FP8/MXFP4 输入直转） |
| 0 | 静态量化 | INT8 |
| 1 | 动态量化 INT8 | INT8 |
| 2 / 3 | MXFP8 量化 | FLOAT8_E5M2 / FLOAT8_E4M3FN |
| 4 / 5 | FP8 PerGroup（GroupSize=128，RoundScale） | FLOAT8_E5M2 / FLOAT8_E4M3FN |
| 6 | HIF8 直转 | HIFLOAT8 |
| 7 | HIF8 PerTensor | HIFLOAT8 |
| 8 | HIF8 PerToken | HIFLOAT8 |
| 9 | MXFP4 量化 | FLOAT4_E2M1 |
| 11 / 12 | FP8 PerBlock（BlockSize=128） | FLOAT8_E5M2 / FLOAT8_E4M3FN |
| 13 | INT4 动态量化 | INT4 |
| 14 / 15 | FP8 PerGroup + Amax clamp | FLOAT8_E5M2 / FLOAT8_E4M3FN |
| 16 / 17 | MXFP8 RoundScale + Amax | FLOAT8_E5M2 / FLOAT8_E4M3FN |

## 3. Tiling 设计（`MoeInitRoutingV3Arch35TilingClass`）

### 3.1 与 A3 的主要结构差异

- 重写 `GetShapeAttrsInfo` 为空实现，**将 shape/attr 获取延后到 `DoOpTiling` 内部**，以便先校验 `IsCapable`。
- 将原本庞大的 `CheckAttr/CheckInputShape/CheckOutShape/CheckDtype` 拆分为 20+ 个细粒度子函数（`ValidateExpertTokensNumType` / `ValidateExpertNum` / `ValidateDropPadMode` / `CheckInputX` / `CheckOutputExpandedX` ...），便于维护。
- 新增 `CheckSetEmptyTensor`：除 `n==0` 外还处理 `cols==0 且 n*k>0` 的子场景（A3 仅处理 `n==0`）。
- 新增 `MXFPX` / `MXFP8` / `FP8 PerBlock` / `FP8 PerGroup` / `HIF8` 等专用 tiling 入口：`Tiling4GatherOutMxQuant / Tiling4GatherOutFP8Quant / Tiling4GatherOutMxFP8NoQuantCompute`。

### 3.2 平台参数

- `aivCoreNum_` / `totalUbSize_` / `socVersion_` 来自 compile info。
- **关键**：`availUbSize_ = totalUbSize_ - SIMT_DCACHE_SIZE(64KB)`，需要给 SIMT 预留 DCache 空间，并配合 `SetLocalMemSize()` 使用。这是 3510 架构独有的约束。
- `sortLoopMaxElement_` 同时受 `AscendC::Sort` API 能力约束（`SORT_API_MAX_ELEM = 32*255`），A3 没有这条硬上限。

### 3.3 切分流程

```
DoOpTiling:
  ├─ CheckSetPlatformInfo()
  ├─ DoGetShapeAttrsInfo()         GetInputTensorsInfo / GetOutputTensorsInfo / GetInputAttrsInfo
  ├─ CheckSetAttrs()               Validate*Series 子函数
  ├─ CheckSetListAttrs()           activeExpertRangeList 校验
  ├─ CheckSetInputs()              CheckInputX / CheckInputExpertIdx / CheckInputScale / CheckInputOffset
  ├─ CheckSetEmptyTensor()         n==0 或 cols==0 特化
  ├─ CheckOutputs()                CheckOutputExpandedX / RowIdx / ExpertTokensCount / ExpandedScale
  │
  ├─ sortLoopMaxElement_ 计算
  ├─ Tiling4VBSCompute             排序切分（沿用 4^x 核数策略，但受 SORT_API_MAX_ELEM 限制）
  ├─ Tiling4VMSMiddleCompute
  ├─ Tiling4SortOutCompute
  ├─ Tiling4ExpertTokensCountCompute
  │
  ├─ isFullload_ = IsFullLoad()
  │
  └─ GatherOut 分支选择（按量化模式分流）
      ├─ MXFP8 / MXFP8 RS+Amax / MXFP4   → Tiling4GatherOutMxQuant
      ├─ FP8 PerBlock / PerGroup          → Tiling4GatherOutFP8Quant
      ├─ DropPad 且非全载                 → Tiling4SrcToDstDropPadCompute + Tiling4GatherOutDropPadCompute
      ├─ MXFP8 NoQuant (x is fp8)         → Tiling4GatherOutMxFP8NoQuantCompute
      └─ 其它（含 INT4/INT8 动态、静态、非量化）→ Tiling4GatherOutCompute
```

### 3.4 FullLoad 判定与 A3 的差异

- `IsSupportFullloadQuantMode()`：A5 仅在 `quantMode ∈ {-1(non-MXFPX), 0, 1(含 INT4)}` 时支持全载，与 A3 一致；MXFP8/MXFP4/FP8 系列**不走全载**。
- `IsFullLoad` 的 UB 预算使用 `availUbSize_`（已扣除 SIMT DCache），A3 直接使用 `aicoreParams_.ubSize`。

### 3.5 TilingKey 编码（与 A3 完全不同）

- **通用模板**：`TILINGKEY_BASE(1e7) + sortMode*1e6 + quantModeFactor*1e4 + rowIdxType*1e3 + dropPadMode*1e2`
  - `quantModeFactor`：大部分为 `quantMode_+1`；MXFP8 系列重映射为 3、MXFP8RS_AMAX 重映射为 17、FP8 PerBlock 为 12、FP8 PerGroup 为 5、FP8 PerGroup+Amax 为 15。
- **全载模板**：`FULLLOAD_TILINGKEY_BASE(2e5) + quantModeFactor*1e4`（INT4 复用 INT8 的 key，kernel 通过 `quantMode` 字段区分）。
- **空 tensor**：`EMPTY_TENSOR_TILINGKEY(3e6)`。

### 3.6 Workspace

`sortWorkspace + coreSyncWorkspace + scatterWorkspace + expertTokensCountWorkspace + 16KB`，结构同 A3，但 DropPad / MXFPX / FP8 / HIF8 等新量化模式按需扩容。

## 4. Kernel 实现（`op_kernel/moe_init_routing_v3_apt.cpp`）

### 4.1 入口差异

- 不再使用 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0)`，而是简单判 `g_coreType == AIC` 后 return。
- **新增 overflow 寄存器保存恢复**：
  ```cpp
  #if (__NPU_ARCH__ == 3510)
      int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
  #endif
  ...
  #if (__NPU_ARCH__ == 3510)
      SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
  #endif
  ```
  每个 tiling 分支退出前都要恢复 overflow mode，这是 RegBase 架构的硬约束。
- 使用 `REGISTER_TILING_DEFAULT + GET_TILING_DATA_WITH_STRUCT`（A3 是 `GET_TILING_DATA`）。

### 4.2 分发逻辑（与 A3 结构相似，但分支更多）

```
EMPTY_TENSOR(3e6)                    n==0 或 cols==0：InitGlobalMemory 刷零
│
├─ 全载模板（2.x e5 段）
│   ├─ FULLLOAD_UNQUANTIZED(2e5)     MoeV3FullLoadUnquantized
│   ├─ FULLLOAD_DYNAMIC_QUANT(2.2e5) MoeV3FullLoadDynamicQuant（INT8 / INT4 通过 quantMode 切换）
│   └─ FULLLOAD_STATIC_QUANT(2.1e5)  MoeV3FullLoadStaticQuant
│
├─ 1. 排序阶段
│   MoeSortOneCore / MoeSortMultiCore（覆盖 14 种量化模式 × 单/多核 = 28+ key）
│
├─ 2. ExpertTokensCount
│
├─ 3. RowIdxGather（或 DropPad 专用 MoeV3RowIdxGatherDropPad）
│
└─ 4. GatherOut（按量化模式分流）
    ├─ MoeV3GatherOutMxfp8         x∈{fp8_e4m3,fp8_e5m2} 非量化直转
    ├─ MoeV3GatherOutMxFp4         x∈fp4x2_e2m1 非量化直转
    ├─ MoeGatherOut<DTYPE_X>       标准非量化
    ├─ MoeV3GatherOutDropPad       非量化 DropPad
    ├─ MoeV3GatherStaticQuant      静态量化
    ├─ MoeGatherOutDynamicQuant    INT8 / INT4 动态量化（INT4 通过模板特化 int4b_t）
    ├─ MoeGatherOutMxfp8Quant      MXFP8（含 RS+Amax，第三模板参数区分）
    ├─ MoeV3GatherFP8GroupQuant    FP8 PerGroup（含 Amax clamp）
    ├─ MoeGatherOutHif8Quant       HIF8 直转
    ├─ MoeGatherOutHif8PertensorQuant
    ├─ MoeGatherOutHif8PertokenQuant
    ├─ MoeV3GatherMxfp4Quant       MXFP4
    ├─ MoeGatherOutFP8PerBlockQuant
    └─ ...
```

### 4.3 模块依赖（全部位于 `arch35/`）

`arch35/moe_v3_sort_one_core.h / moe_v3_sort_multi_core.h / moe_v3_mrgsort*.h / moe_v3_full_load_*.h / moe_v3_gather_*.h / moe_v3_gather_out_mxfp8.h / moe_v3_gather_out_mxfp4.h / moe_v3_gather_mxfp4_quant.h / moe_v3_gather_mxfp8_quant.h / moe_v3_gather_fp8_perblock_quant.h / moe_v3_gather_fp8_group_quant.h / moe_v3_gather_hif8_*.h / moe_v3_row_idx_gather*.h / moe_v3_expert_tokens_count.h`

## 5. 关键设计要点

1. **量化模式扩展**：14 种量化模式覆盖 FP8/MXFP8/MXFP4/HIF8/INT4/INT8 全场景，是 A5 相对 A3 的核心增量。
2. **RegBase 架构适配**：SIMT DCache 预留、overflow 寄存器保存恢复、Sort API 元素上限，都是 3510 硬件约束的直接体现。
3. **Tiling 类拆分**：20+ 子校验函数 + 多个量化专用 Tiling 入口，便于扩展新量化模式。
4. **全载复用**：INT4 动态量化全载复用 INT8 的 tilingkey，kernel 通过 `quantMode` 字段区分，减少 key 空间膨胀。
5. **空 tensor 处理增强**：除 `n==0` 外，处理 `cols==0 且 n*k>0` 时仍需初始化 `expandedRowIdx/expandedScale`。
