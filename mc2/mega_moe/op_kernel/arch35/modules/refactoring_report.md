# MegaMoE Arch35 模块化重构报告

## 1. 结构性变化

### 原始代码结构

原始 `mega_moe.h` 是一个 1376 行的单体类 `MegaMoe`，包含全部逻辑：
- 多核编排（dispatch、通信、量化）
- 单核 per-expert GMM 计算
- 全部以 private 方法实现，隐式访问 20+ 成员变量

### 重构后结构

per-expert GMM 计算逻辑被抽取为两个自包含的模块 struct：

| 抽取逻辑 | 新文件 | 新 Struct |
|---|---|---|
| UpdateGroupParams (kGmm1) + flag-wait/sendCnt 分支 | `gmm1_op.h` | `Gmm1Op` |
| UpdateGlobalBuffer (kGmm1) | `gmm1_op.h` | `Gmm1Op` |
| GroupMatmulWithSwigluQuant | `gmm1_op.h` | `Gmm1Op` |
| UpdateGroupParams (kGmm2) | `gmm2_op.h` | `Gmm2Op` |
| UpdateGlobalBuffer (kGmm2) | `gmm2_op.h` | `Gmm2Op` |
| GroupMatmulWithCombine | `gmm2_op.h` | `Gmm2Op` |
| ProcessCombine → ProcessCombineInternal | `gmm2_op.h` | `Gmm2Op` |
| InitCombineBuffers | `gmm2_op.h` | `Gmm2Op` |
| ResetGmm2CombineSyncCounters | `gmm2_op.h` | `Gmm2Op` |

共享类型定义被抽取到 `mega_moe_tile_info.h`：

| 原始位置 | 新位置 |
|---|---|
| `TupleShape`、`BlockOffset` (MegaMoeImpl namespace 内 typedef) | `mega_moe_tile_info.h` (MegaMoeModules namespace) |
| `ExpertLoopState` (MegaMoe class 内嵌套 struct) | `mega_moe_tile_info.h` (MegaMoeModules namespace) |
| `TileInfo` (**新增**) + `TilingByTile` (**新增**) | `mega_moe_tile_info.h` |

---

## 2. 接口变化 — 参数数量对比

**设计约束：process() 中每个模块调用 ≤5 参数**。

原始方法作为 private class method 隐式访问成员变量，"有效依赖数"远超 5。新模块方法为 static function，仅使用显式参数。

| 方法 | 原始显式参数 | 原始有效依赖 (含隐式成员) | 新签名 | 新显式参数 |
|---|---|---|---|---|
| GMM1 UpdateGroupParams | 3 | ~8 (state, expertIdx, sendCnt, params_, aicNum_, blockIdx_, subBlockIdx_, expertRevNumsGlobalTensor_) | `UpdateGroupParams(ctx, expertIdx, sendCnt, tile, cfg)` | **5** |
| GMM1 UpdateGlobalBuffer | 2 | ~5 (gmmAddrInfo, state, params_, dispatchFlagSlotsPerExpert_, epilogueOp_) | `UpdateGlobalBuffer(ctx, tile, cfg)` | **3** |
| GMM1 Compute | 2 | ~7 (gmmAddrInfo, state, params_, startBlockIdx_, vecSetSyncCom_, epilogueOp_, ENABLE_A8W4) | `Compute(ctx, cfg)` | **2** |
| GMM2 UpdateGroupParams | 2 | ~5 (state, expertIdx, params_, blockIdx_, aicNum_, expertRevNumsGlobalTensor_) | `UpdateGroupParams(ctx, expertIdx, tile, cfg)` | **4** |
| GMM2 UpdateGlobalBuffer | 2 | ~5 (gmmAddrInfo, state, params_, dispatchFlagSlotsPerExpert_, CombineQuantMode) | `UpdateGlobalBuffer(ctx, tile, cfg)` | **3** |
| GMM2 Compute | 3 | ~9 (addrInfo, state, expertIdx, params_, startBlockIdx_, vecSetSyncCom_, gmm2PingPongIdx_, ENABLE_A8W4, CombineQuantMode) | `Compute(ctx, expertIdx, tile, cfg)` | **4** |
| ProcessCombine | 3 | ~8 (addrInfo, gmm2State, expertIdx, k_, aivCoreIdx_, blockAivNum_, subBlockIdx_, rankId_, combineUbTensorSize_) | `ProcessCombineInternal(ctx, expertIdx, tile, cfg)` | **4** |
| InitCombineBuffers | 0 | ~2 (k_, combineUbTensorSize_) | `InitCombineBuffers(cfg)` | **1** |
| ResetGmm2CombineSyncCounters | 0 | ~5 (params_, expertPerRank_, blockAivNum_, blockIdx_, resetTensor_) | `ResetGmm2CombineSyncCounters(tile, cfg, resetTensor)` | **3** |

**所有原始有效依赖 >5 的方法，重构后显式参数均 ≤5**。

---

## 3. 核标识变化 — blockIdx → TileInfo

### 原始：硬件 API 作为成员变量

```cpp
uint32_t blockNum_     = GetBlockNum();
uint32_t blockIdx_     = GetBlockIdx() / GetTaskRation();
uint32_t aivCoreIdx_   = GetBlockIdx();
uint32_t subBlockIdx_  = GetSubBlockIdx();
uint32_t blockAivNum_  = GetBlockNum() * 2;
```

GMM 方法中使用位置：

| 原始代码位置 | 用途 |
|---|---|
| UpdateGroupParams kGmm1 (line 864) | `subBlockIdx_ == 0` 选择 flag-wait vs. sendCnt 分支 |
| UpdateGroupParams kGmm1 (line 866-867) | `blockIdx_` 计算 flag 地址 |
| UpdateGroupParams kGmm1 (line 874) | `blockIdx_` 计算 expertRevNumsGlobalTensor 偏移 |
| UpdateGroupParams kGmm2 (line 882) | `blockIdx_` 计算 expertRevNumsGlobalTensor 偏移 |
| ProcessCombine (line 1002-1003) | `aivCoreIdx_`/`blockAivNum_` 用于核分组 |
| ProcessCombine (line 1005-1009) | `subBlockIdx_` A8W4 门控 |
| ProcessCombine (line 1021-1022) | `aivCoreIdx_`/`blockAivNum_` 计数器地址 |
| ResetGmm2CombineSyncCounters | `blockAivNum_` 总计数器计算；`GetBlockIdx()/GetBlockNum()` 内嵌于 TilingByCore |

### 重构后：TileInfo 显式传入

```cpp
struct TileInfo {
    uint32_t tileIdx;        // 替代 blockIdx_
    uint32_t totalTiles;     // 替代 blockNum_
    uint32_t aivTileIdx;     // 替代 aivCoreIdx_
    uint32_t aivTotalTiles;  // 替代 blockAivNum_
    uint32_t subTileIdx;     // 替代 subBlockIdx_
};
```

编排器在 Process() 中构造：

```cpp
MegaMoeModules::TileInfo tile;
tile.tileIdx       = blockIdx_;
tile.totalTiles    = blockNum_;
tile.aivTileIdx    = aivCoreIdx_;
tile.aivTotalTiles = blockAivNum_;
tile.subTileIdx    = subBlockIdx_;
```

替换映射：

| 原始引用 | 新引用 | 所在模块 |
|---|---|---|
| `subBlockIdx_ == 0` | `tile.subTileIdx == 0` | Gmm1Op::UpdateGroupParams |
| `blockIdx_` (flag 地址) | `tile.tileIdx` | Gmm1Op::UpdateGroupParams |
| `blockIdx_` (expertRev 偏移) | `tile.tileIdx` | Gmm1Op/Gmm2Op::UpdateGroupParams |
| `aivCoreIdx_` (分组) | `tile.aivTileIdx` | Gmm2Op::ProcessCombineInternal |
| `blockAivNum_` (总核数) | `tile.aivTotalTiles` | Gmm2Op::ProcessCombineInternal |
| `subBlockIdx_` (A8W4 门控) | `tile.subTileIdx` | Gmm2Op::ProcessCombineInternal |
| `GetBlockIdx()/GetBlockNum()` in TilingByCore | `tile.aivTileIdx`/`tile.aivTotalTiles` via TilingByTile | Gmm2Op::ResetGmm2CombineSyncCounters |

新增 `TilingByTile` 函数替代 `TilingByCore`：

```cpp
// 原始：TilingByCore(totalLen, coreLen, coreOffset) — 内部调用 GetBlockIdx()/GetBlockNum()
// 新增：TilingByTile(totalLen, tileLen, tileOffset, tileIdx, totalTiles) — 使用显式参数
```

---

## 4. 状态捆绑 — Config 与 Context

### Gmm1Op::Config（共享，Process() 中一次性设置）

```cpp
struct Config {
    Params params;                             // 替代 params_ (workspace/peermem/tiling)
    uint32_t aicNum;                           // 替代 aicNum_
    int32_t dispatchFlagSlotsPerExpert;        // 替代 dispatchFlagSlotsPerExpert_
    GlobalTensor<int32_t> expertRevNumsGlobalTensor; // 替代 expertRevNumsGlobalTensor_
    BlockEpilogue* epilogueOp;                 // 替代 epilogueOp_ 成员
};
```

捆绑 5 个原始成员变量。编排器 Process() 中设置：
```cpp
gmm1Cfg.params = params_;
gmm1Cfg.aicNum = aicNum_;
gmm1Cfg.dispatchFlagSlotsPerExpert = dispatchFlagSlotsPerExpert_;
gmm1Cfg.expertRevNumsGlobalTensor = expertRevNumsGlobalTensor_;
gmm1Cfg.epilogueOp = &epilogueOp_;
```

### Gmm1Op::Context（per-loop，per-expert 迭代）

```cpp
struct Context {
    ExpertLoopState state;         // 替代局部 gmm1State
    GMMAddrInfo addrInfo;          // 替代局部 gmm1AddrInfo
    uint32_t startBlockIdx = 0;    // 替代 startBlockIdx_ 成员
    int32_t vecSetSyncCom = 0;     // 替代 vecSetSyncCom_ 成员
};
```

通过 `Gmm1Op::CreateContext(hiddenDim_, k_)` 创建。

### Gmm2Op::Config（共享，Process() 中一次性设置）

```cpp
struct Config {
    Params params;                             // 替代 params_
    uint32_t aicNum;                           // 替代 aicNum_
    int32_t dispatchFlagSlotsPerExpert;        // 替代 dispatchFlagSlotsPerExpert_
    GlobalTensor<int32_t> expertRevNumsGlobalTensor; // 替代 expertRevNumsGlobalTensor_
    uint32_t rankId;                           // 替代 rankId_
    uint32_t k;                                // 替代 k_
    int64_t combineUbTensorSize = 0;           // 替代 combineUbTensorSize_ (由 InitCombineBuffers 计算)
};
```

捆绑 7 个原始成员变量。

### Gmm2Op::Context（per-loop，per-expert 迭代）

```cpp
struct Context {
    ExpertLoopState state;         // 替代局部 gmm2State
    GMMAddrInfo addrInfo;          // 替代局部 gmm2AddrInfo
    uint32_t startBlockIdx = 0;    // 替代 startBlockIdx_ 成员
    int32_t vecSetSyncCom = 0;     // 替代 vecSetSyncCom_ 成员
    uint16_t gmm2PingPongIdx = 0;  // 替代 gmm2PingPongIdx_ 成员
};
```

### 捆绑如何减少参数

无捆绑时，`UpdateGroupParams` 需要 `(state, expertIdx, sendCnt, params, aicNum, blockIdx, subBlockIdx, expertRevNumsGlobalTensor)` = 8 参数。通过 Config (5字段→1引用) + Context (4字段→1引用) + TileInfo (5字段→1引用)，变为 `(ctx, expertIdx, sendCnt, tile, cfg)` = **5 参数**。

---

## 5. 模板参数变化

### 原始 MegaMoe 模板参数

```cpp
template <typename XType, typename OutputType, typename TopkWeightsType,
          typename Weight1Type, int32_t QuantMode, int32_t CombineQuantMode>
```
6 个模板参数。派生类型在类体内计算：
- `QuantOutType`, `SwigluQuantOutType`, `QuantScaleOutType`, `ActivationType`
- `A_ELEMS_PER_BYTE`, `B_ELEMS_PER_BYTE`, `C_ELEMS_PER_BYTE`
- `ENABLE_A8W4`, `ENABLE_A4W4` (static constexpr bool)

### Gmm1Op 模板参数

```cpp
template <typename QuantOutType, typename Weight1Type, typename SwigluQuantOutType,
          typename QuantScaleOutType, int32_t QuantMode, bool ENABLE_A8W4, bool ENABLE_A4W4>
```
7 个模板参数。

变化：
- **移除**: `XType`, `OutputType`, `TopkWeightsType` (GMM1 不需要)
- **移除**: `CombineQuantMode` (GMM1 不涉及)
- **提升为显式参数**: `QuantOutType`, `SwigluQuantOutType`, `QuantScaleOutType` (原始为类内派生)
- **提升为模板 bool**: `ENABLE_A8W4`, `ENABLE_A4W4` (原始为 static constexpr bool)

### Gmm2Op 模板参数

```cpp
template <typename QuantOutType, typename Weight1Type, typename SwigluQuantOutType,
          typename QuantScaleOutType, int32_t QuantMode, int32_t CombineQuantMode,
          bool ENABLE_A8W4, bool ENABLE_A4W4>
```
8 个模板参数。与 Gmm1Op 相同的提升方式，额外保留 `CombineQuantMode`。

### 编排器类型别名

编排器仍计算派生类型并传递：
```cpp
using Gmm1Op = MegaMoeModules::GMM1::Gmm1Op<
    QuantOutType, Weight1Type, SwigluQuantOutType, QuantScaleOutType,
    QuantMode, ENABLE_A8W4, ENABLE_A4W4>;
using Gmm2Op = MegaMoeModules::GMM2::Gmm2Op<
    QuantOutType, Weight1Type, SwigluQuantOutType, QuantScaleOutType,
    QuantMode, CombineQuantMode, ENABLE_A8W4, ENABLE_A4W4>;
```

---

## 6. 逻辑等价性验证

| 方法 | 等价性 | 差异说明 |
|---|---|---|
| UpdateGroupParams kGmm1 | **等价** | offset 计算块逐行相同；`tile.subTileIdx` 替代 `subBlockIdx_`；`tile.tileIdx` 替代 `blockIdx_`；原始单模板函数按 Mode 分支 → 新代码拆为两个独立函数 |
| UpdateGlobalBuffer kGmm1 | **等价** | 地址计算公式相同；`*cfg.epilogueOp` 指针解引用替代 `epilogueOp_` 成员访问 |
| GroupMatmulWithSwigluQuant | **等价** | 三分支 dispatch (A8W4/NZ/generic) 完全保留 |
| UpdateGroupParams kGmm2 | **等价** | offset 计算与 Gmm1Op 中相同（代码复制的合理取舍）；M_VALUE 获取用 `tile.tileIdx` 替代 `blockIdx_` |
| UpdateGlobalBuffer kGmm2 | **等价** | 所有地址计算公式保留，包括 constexpr guard |
| GroupMatmulWithCombine | **等价** | A8W4/A4W4/A8W8_NZ/generic 分支保留；ProcessCombineInternal 调用保留 |
| ProcessCombineInternal | **等价** | `cfg.k` 替代 `k_`；`tile.aivTileIdx/aivTotalTiles/subTileIdx` 替代对应成员；ComputeCoreGrouping/counter wait/CombineTokenGroup 调用不变 |
| InitCombineBuffers | **等价** | 公式 `(singleTokenBytes * 2) / sizeof(bfloat16_t)` 相同；结果存入 `cfg.combineUbTensorSize` 替代成员 |
| ResetGmm2CombineSyncCounters | **等价** | `TilingByTile` 替代 `TilingByCore`（显式参数 vs. 内部调用 GetBlockIdx/GetBlockNum）；`resetTensor` 由参数传入替代成员变量 |

---

## 7. 保留在编排器中的方法

以下方法 **未抽取** 到模块，保留在 `modules/mega_moe.h`：

| 方法 | 保留原因 |
|---|---|
| Init | 拥有全部成员变量初始化；设置 GM tensors、workspace、peermem、epilogue |
| DispatchBuffInit | 分配 LocalTensor 成员 buffer (expertTokenCntTensor_, cumsumInfoTensor_, gatherMaskTensor_ 等) |
| SendAndQuantBuffInit | 分配 LocalTensor 成员 buffer (topkIdsTensor_, resetTensor_, xIn/Out, sendMaskTensor_ 等) |
| UnpermuteBuffInit | 分配 LocalTensor 成员 buffer (dataResTensor_, dataResFp32Tensor_, topKWeightsTensor_, bf16/fp32ScaleTensor_) |
| SendMaskCal | 多核跨卡通信，使用成员 tensors，按 `aivCoreIdx_` stride 跨 `blockAivNum_` 迭代全部全局专家 |
| SendCntCal | per-local-expert 计数计算，使用成员 tensors，写 GM AtomicAdd flag |
| TripleInfoCalAndDispatch | 多核 dispatch，`blockIdx_/blockNum_` stride，跨卡通信 |
| CopyGMToGMPerToken | 6-buffer 软流水，使用成员 tensors |
| ResetFlagList | workspace flag 清理，委托 Gmm2Op::ResetGmm2CombineSyncCounters 但自身保留 swigluToGmm2/dispatchToGmm1 flag 重置 |
| QuantProcessInRank | Pre-GMM 本卡 token 量化，使用成员 tensors |
| ExpertTokenNumCopyOut | 使用成员 tensors |
| CrossRankSyncInWorldSize | 跨卡同步，`aivCoreIdx_/blockAivNum_` |
| Unpermute | Post-GMM unpermute/reduction，使用成员 tensors |

**共性**：保留的方法要么 (a) 分配/持有 LocalTensor 成员 buffer，(b) 执行多核 stride 通信，或 (c) 是不属于 per-expert GMM 计算循环的前/后处理阶段。

---

## 8. 依赖变化

### 原始 mega_moe.h includes

```
kernel_operator.h, kernel_tiling/kernel_tiling.h, mc2_kernel_utils.h (conditional),
kernel_operator_list_tensor_intf.h, mega_moe_base.h, mega_moe_workspace_info.h,
block_epilogue_swiglu_mx_quant.h, mega_moe_impl.h, quantize_functions.h (conditional)
```

### 新编排器 modules/mega_moe.h includes

同原始 + **新增**:
```
mega_moe_combine_send.h         (原始通过 mega_moe_impl.h 间接引用，现在显式引用)
modules/mega_moe_tile_info.h    (新增)
modules/gmm1_op.h               (新增)
modules/gmm2_op.h               (新增)
```

### gmm1_op.h includes

```
mega_moe_tile_info.h, mega_moe_base.h, mega_moe_workspace_info.h,
block_epilogue_swiglu_mx_quant.h, mega_moe_impl.h
```

**不包含** (vs 原始编排器): `mc2_kernel_utils.h` (无量化), `quantize_functions.h` (无 QuantProcessInRank), `mega_moe_combine_send.h` (GMM1 无 combine)

### gmm2_op.h includes

```
mega_moe_tile_info.h, mega_moe_base.h, mega_moe_workspace_info.h,
block_epilogue_swiglu_mx_quant.h, mega_moe_impl.h, mega_moe_combine_send.h
```

**额外包含**: `mega_moe_combine_send.h` (ProcessCombineInternal 的 CombineTokenGroup 调用需要)

### 命名空间变化

| 类型 | 原始命名空间 | 新命名空间 |
|---|---|---|
| TupleShape | `MegaMoeImpl` (class 内 typedef) | `MegaMoeModules` (mega_moe_tile_info.h) |
| BlockOffset | `MegaMoeImpl` (class 内 typedef) | `MegaMoeModules` (mega_moe_tile_info.h) |
| ExpertLoopState | `MegaMoeImpl::MegaMoe` (nested struct) | `MegaMoeModules` (mega_moe_tile_info.h) |
| TileInfo | 不存在 | `MegaMoeModules` (新增) |
| TilingByTile | 不存在 (使用 TilingByCore) | `MegaMoeModules` (新增) |

结构相同（均为 `Shape<int64_t, ...>`），仅命名空间不同，无功能影响。

---

## 总结

本次重构将原始单体 `MegaMoe` 类的 **per-expert GMM 计算逻辑** 拆分为两个独立的单核算子模块（Gmm1Op、Gmm2Op），同时保留多核编排器负责 buffer 管理、通信和前后处理。核心变化：

1. **GMM1/GMM2 内部零 blockIdx** — 所有核标识通过 TileInfo 显式传入
2. **接口 ≤5 参数** — 通过 Config/Context/TileInfo 捆绑替代隐式成员访问
3. **逻辑完全等价** — 每个抽取方法的计算公式和分支逻辑与原始代码一致
4. **模板参数提升** — 派生类型和 constexpr bool 从类内派生变为显式模板参数
5. **新增 TilingByTile** — 替代 TilingByCore，不依赖硬件 API
