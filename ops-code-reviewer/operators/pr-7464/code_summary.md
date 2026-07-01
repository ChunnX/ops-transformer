# 代码概要

算子: FusedInferAttentionScore | 功能: Fused Infer Attention Score Tiling 侧参数初始化、切分策略、TilingData 填充 | 侧别: Tiling

## 变更文件概览

| # | 文件 | 侧别 | 变更类型 | 说明 |
|---|------|------|---------|------|
| 1 | attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.cpp | Tiling | 修改 | 核心重构：提取子函数、表驱动化、规范化代码结构 |
| 2 | attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.h | Tiling | 修改 | 新增重构子函数声明、新增类成员方法声明 |

---

## 文件1: fused_infer_attention_score_tiling_impl.cpp（Tiling侧）

### 变更概要

本次 PR 是一次**代码整洁度重构（cleancode）**，不改变算子整体功能逻辑。主要动作：
1. **大函数分解** — 将 `InitImplParam`、`SplitPolicy`、`ComputeTilingData` 等巨函数拆分为多个职责单一的私有方法
2. **条件提取** — 将内联的条件判断逻辑提取为独立的方法（如 `CheckPFAMergeSupport`、`CheckSmallHeadOptimization`）
3. **表驱动化** — 将 `UpdateTilingKeyConfig` 中的长 if-else 链替换为 `TilingConfigRule` 结构体 + `std::vector` 查表
4. **工具函数复用** — 引入 `IsTensorValid`、`ParseSeqLength`、`GetMapValueOrDefault`、`ReadFirstTensorValue`、`SnapDimToBucket` 等 static 工具函数消除重复代码
5. **空指针加固** — 在多处添加了原本缺失的 nullptr 检查

### 代码脉络

**入口**: `FusedInferAttentionScoreTilingImpl::DoOpTiling(gert::TilingContext *context, const FiaTilingInfo &fiaInfo)` → 文件:2146-2181 → 被 `fused_infer_attention_score_tiling_v4.cpp:51` 调用

**数据流**:
```
FiaTilingInfo (外部传入的解析后算子参数)
  → InitImplParam (初始化 Tiling flags + 计算 loop 参数 + 解析实际序列长度)
    → InitTilingFlags (设置 actualSeqLenQFlag_/actualSeqLenKVFlag_/actualSharedPrefixLenFlag_)
    → ComputeLoopParams (调用 SetGSMerge + SetIsIFA → 计算 nLoopTimes_/gsSize_/headDimAlign_)
    → ParseActualSeqLengths (填充 actualSeqLengthsQ_/actualSeqLengthsKV_ 向量)
  → SplitPolicy (根据反量化模式选择切分策略)
    → ApplySinnerSouterOverrides (DN/量化场景覆盖 sInnerFactor_/sOuterFactor_)
    → ApplySplitStrategy (S1外切分 / NB切分 / FlashDecode)
    → ApplyRowInvalidCheck (检测行无效条件)
  → ComputeTilingData (填充 FA/PA Tiling 数据结构)
    → SetMaskTilingData + SetSparseCompressMode + ComputeSparseType
    → SetAlibiStartIdx + SetLayoutType + SetPaLayoutTilingData
    → SetTransposeLayout + SetAntiQuantInfo + ComputeNeedInit
  → GenTilingKey + SetBlockDim + GetWorkspace + SetTilingData (写回 context)
```

**计算核心**: `DoOpTiling` → 顺序调用 7 个子阶段，每个子阶段负责 TilingData 结构的一个维度

**分支覆盖**:
| 分支条件 | 位置(行) | 触发场景 | 处理逻辑 |
|---------|---------|---------|---------|
| `fiaInfo.emptyTensorFlag == true` | 2152 | Empty tensor 场景 | 走 `SetEmptyTensor` 快速路径，直接写回 |
| `quantMode == ANTI_QUANT` | 1211 | 反量化模式 | 调用 `SplitDequant`，使用 Cube 视角分核 |
| `quantMode != ANTI_QUANT` | 1215-1221 | 非量化/全量化 | `AdjustSinnerAndSouter` + `ApplySinnerSouterOverrides` + `ApplySplitStrategy` |
| `enableS1OutSplit == true` | 1248-1250 | S1 外切分核优化 | 调用 `SplitOutSeq` |
| `isIFAFlag_ && !pfaMergeFlag_ && CheckFlashDecode` | 1254-1258 | FlashDecode 场景 | `flashDecodeFlag_=true` + `SplitS2` |
| `gsMergeFlag_ == true` | 260 | G/S1 合轴 | `nLoopTimes_=n2Size`, `gsSize_=gSize*s1Size` |
| `antiQuantFlag` 内各种条件 | 122-186 | 反量化合轴判断 | `CheckPFAMergeSupport`/`CheckActualSeqLenAligned`/`CheckAntiQuantGSMerge` |
| attenMask 相关 | 1744-1769 | Mask 存在时 | 从 tensor shape 提取 mask 尺寸并填充 |
| `fiaInfo.kvStorageMode == PAGE_ATTENTION` | 1826 | PA 场景 | 设置 paLayoutType |
| `fiaInfo.antiQuantFlag` (in ComputeNeedInit) | 1889-1890 | 伪量化 | 跳过 needInit 计算 |

### 新增工具函数

| 函数 | 作用 | 复用点 |
|------|------|--------|
| `IsTensorValid(tensor, dims)` | 三重检查(dims>0, tensor!=null, data!=null) | `InitTilingFlags` |
| `ParseSeqLength(tensor, dims, bIdx, layout)` | 从 tensor 解析第bIdx个batch的实际序列长度，处理累积型 layout | `ParseActualSeqLengths`, `CheckActualSeqLenAligned` |
| `IsSupportedQLayout(qLayout)` | 判断是否为 BSH/BSND/TND 优化 layout | `CheckPfaMergeLayout`, `GetPreNextTokensLeftUp` |
| `GetMapValueOrDefault(map, key, defaultVal)` | 模板函数，safe map 查找 | `SetSparseCompressMode`, `SetLayoutType`, `SetTransposeLayout` |
| `ReadFirstTensorValue(tensor)` | 安全读取 tensor 第一个 int64 值 | `SetAlibiStartIdx` |
| `SnapDimToBucket(dim)` | 将维度值归入最近的 bucket (64/128/256/512/576) | `UpdateTilingKeyConfig` |
| `SetSparseCompressMode(fiaInfo, inputParams)` | static 函数，设置 sparse mode → compress mode 映射 | `ComputeTilingData` |

### 核心 API: 
- `gert::TilingContext` — Ascend Tiling 框架上下文，用于获取平台信息、设置 TilingKey/BlockDim/Workspace/TilingData
- `platform_ascendc::PlatformAscendC` — 平台信息查询（GetCoreNumAiv/Aic, GetCoreMemSize 等）
- `gert::Tensor::GetStorageShape()/GetShapeSize()/GetData<T>()` — 从框架传入的 tensor 获取 shape 和数据指针
- `faRunTilingAdapter_.inputParamsRegbase.set_xxx()` — 向 TilingData 结构体填充参数
- `OP_CHECK_IF` / `OP_LOGI` / `OP_LOGE` — Ascend C 框架的错误检查和日志宏

### 输出:
- `faRunTilingAdapter_` (FlashAttentionScoreSimplifiedTilingData) 通过 `SetTilingData(context, fiaInfo)` 写入 context
- 同步机制：使用 `context->SetScheduleMode(BATCH_MODE_SCHEDULE)` (行:2175)

---

## 文件2: fused_infer_attention_score_tiling_impl.h（Tiling侧）

### 变更概要

新增方法声明以匹配 .cpp 中的函数分解重构：

| 新增方法 | 职责 |
|---------|------|
| `ApplySinnerSouterOverrides` | DN/量化场景覆盖 sInner/sOuter |
| `ApplySplitStrategy` | 选择并执行切分策略 |
| `ApplyRowInvalidCheck` | 检查行无效条件 |
| `SetMaskTilingData` | 填充 attenMask 相关 TilingData |
| `ComputeSparseType` | 计算 sparse 压缩类型 |
| `SetAlibiStartIdx` | 设置 alibi 起始索引 |
| `SetLayoutType` | 设置 layout 类型映射 |
| `SetPaLayoutTilingData` | 设置 PA 场景 layout |
| `SetTransposeLayout` | 设置转置 layout |
| `SetAntiQuantInfo` | 设置反量化 per-tensor flag |
| `ComputeNeedInit` | 计算是否需要初始化输出 |
| `CheckPfaMergeLayout` | 检查 PFA 合轴 layout 条件 |
| `CheckSmallHeadOptimization` | 检查小头优化条件 |
| `InitTilingFlags` | 初始化 tiling 标志位 |
| `ComputeLoopParams` | 计算循环参数 |
| `ParseActualSeqLengths` | 解析实际序列长度 |
| `CheckPFAMergeSupport` | 检查 PFA 合轴是否支持 |
| `CheckActualSeqLenAligned` | 检查实际序列长度是否对齐 |
| `CheckAntiQuantGSMerge` | 反量化 GS 合轴检查 |

---

## 变量溯源（关键类成员）

| 变量 | 声明(文件:行) | 初始化(文件:行) | 来源类型 |
|------|-------------|----------------|---------|
| `actualSeqLenQFlag_` | .h:173 `bool false` | InitTilingFlags:.cpp:234 `IsTensorValid(...)` | 外部输入(Tensor检查) |
| `actualSeqLenKVFlag_` | .h:174 `bool false` | InitTilingFlags:.cpp:235 `IsTensorValid(...)` | 外部输入(Tensor检查) |
| `actualSharedPrefixLenFlag_` | .h:175 `bool false` | InitTilingFlags:.cpp:244-248 | 外部输入(Tensor检查) |
| `gsMergeFlag_` | .h:170 `bool false` | ComputeLoopParams:.cpp:254-258 | 外部输入派生 |
| `pfaMergeFlag_` | .h:171 `bool false` | SetGSMerge:.cpp:115 `CheckPFAMergeSupport(...)` | 外部输入派生 |
| `nLoopTimes_` | .h:160 `uint32_t` | ComputeLoopParams:.cpp:260 | TilingInfo派生 |
| `gsSize_` | .h:161 `uint32_t` | ComputeLoopParams:.cpp:261 | TilingInfo派生 |
| `sOuterFactor_` | .h:162 `uint32_t` | AdjustSinnerAndSouter:.cpp:345 `SOUTER_64` | 外部输入派生+编译期常量 |
| `sInnerFactor_` | .h:163 `uint32_t` | AdjustSinnerAndSouter:.cpp:346 `SINNER_128` | 外部输入派生+编译期常量 |
| `needInit_` | .h:181 `bool false` | ComputeNeedInit:.cpp:1888 `false` | 外部输入派生 |
| `enableS1OutSplit` | .h:182 `bool false` | ApplySplitStrategy:.cpp:1247 `CheckS1OutSplit(...)` | 外部输入派生 |
| `isRowInvalid_` | .h:183 `bool false` | ApplyRowInvalidCheck:.cpp:1265 `true` | 外部输入派生 |
| `isPFAFlag_` | .h:179 `bool false` | ComputeLoopParams:.cpp:264-265 | 外部输入派生 |
| `isIFAFlag_` | .h:180 `bool false` | SetIsIFA:.cpp:99 | 外部输入派生 |

> 来源类型说明：所有关键标志位均派生自 `FiaTilingInfo`（外部传入的已解析参数），`fiaInfo` 内部的字段在上游（`fused_infer_attention_score_tiling_info_parser.h`）已完成校验。数字常量（`SOUTER_*/SINNER_*/DSIZE_*/NUM_*`）定义在 `fused_infer_attention_score_tiling_constants.h` 中作为编译期 `constexpr`。

### ⚠️ 行为变更点

| 位置 | 旧行为 | 新行为 | 影响 |
|------|--------|--------|------|
| `ComputeNeedInit` 循环内变量初始化 (1893-1894) | `preTokensPerbatch`/`nextTokensPerbatch` 在循环外声明，在 `RIGHT_DOWN`/`BAND` 之外的分支才重置为 fiaInfo 值，否则保持上一次赋值 | 每次循环重新初始化为 `fiaInfo.preToken`/`fiaInfo.nextToken` | 对非 RIGHT_DOWN/BAND 的连续批次行为一致；对 RIGHT_DOWN/BAND 场景，旧代码中前一次迭代的修改可能泄漏到下一批次 |
| `ComputeNeedInit` antiQuant 提前返回 (1889-1890) | 循环条件包含 `quantMode != ANTI_QUANT` | 函数开头 early return | 等价，逻辑无变化 |
| `CheckActualSeqLenAligned` nullptr 检查 (149) | 未显式检查 `actSeqLenQ == nullptr`，依赖 `GetShapeSize() == 0` 做防护 | 显式检查 nullptr 并 return true | 增加安全性，原代码在 dims > 0 但 tensor 为 nullptr 时有风险 |
| `SetMaskTilingData` nullptr 检查 (1754-1757) | `attenMaskFlag==true` 时直接访问 tensor 属性，未检查 nullptr | 添加 `attenMaskTensor == nullptr` 错误日志和 GRAPH_FAILED 返回 | 增加了防御性编程 |
| `UpdateTilingKeyConfig` 中的 `sOuter`/`sInner` 使用 (1298-1303) | 伪量化分支内使用 `sOuterFactor_`/`sInnerFactorSize_` 直接比较；非量化使用 `sOuterFactor_*2`/`sInnerFactor_` | 通过局部变量 `sOuter`/`sInner` 统一设置，伪量化时用 `sOuterFactor_`/`sInnerFactorSize_`，非量化用 `sOuterFactor_*2`/`sInnerFactor_` | 等价重构，无行为变化 |
| `AdjustSinnerAndSouter` 的 `sOuterFactor_`/`sInnerFactor_` 默认值 (362-367) | 旧代码在 `ROPE_SPLIT_D512` 分支中显式设置 `sInnerFactor_ = SINNER_128` 和 `softmaxSOuterFactor = SOUTER_64` | 新代码使用三元表达式 `sOuterFactor_ = ...? SOUTER_32 : SOUTER_64`，**不再设置 `sInnerFactor_` 和 `softmaxSOuterFactor`** | `sInnerFactor_` 保持初始值 `SINNER_128` (等价)，但 `softmaxSOuterFactor` 保持初始值 `SOUTER_64`（旧代码在该分支中也设为 SOUTER_64，等价） |

---

## 跨文件关联

### Tiling → Kernel 数据流

Tiling 侧填充的数据结构：
- `faRunTilingAdapter_` (`FlashAttentionScoreSimplifiedTilingData`) — 包含 `inputParamsRegbase`, `multiCoreParamsRegbase`, `initOutputParams` 三个子结构
- `pfaTilingData_` (`PromptFlashAttentionTilingDataV2`) — 包含 `promptAttentionBaseParams`, `promptAttentionSingleCoreParams`, `promptAttentionSeqParams`
- `ifaTilingData_` (`IncreFlashAttentionTilingDataRegbase`) — IFA 场景专用

这些数据结构通过 `SetTilingData`（.cpp:2027-2046）写入 `gert::TilingContext`，再由框架传递到 Kernel 侧的 AI Core 上执行计算。

### Include / 调用链

| 上游文件 | 关联方式 | 依据 |
|----------|----------|------|
| `fused_infer_attention_score_tiling_impl.h` | #include | .cpp:16 |
| `../fused_infer_attention_score_tiling_info_parser.h` | #include | .cpp:17 — `FiaTilingInfo` 结构定义，包含 opParamInfo 等所有输入参数 |
| `../fused_infer_attention_score_tiling_constants.h` | #include | .cpp:24 — 所有编译期常量定义（SOUTER_*/SINNER_*/DSIZE_*/NUM_* 等） |
| `../fused_infer_attention_score_tiling_utils.h` | #include | .cpp:23 — 工具函数（AlignUp, CeilDivision, CalcTailSize 等） |
| `tiling/tiling_api.h` | #include | .cpp:21 — Ascend C Tiling API |
| `../../common/op_host/fia_tiling_base.h` | #include (via .h:19) | 基类 `FiaTilingBase` 定义 |
| `../../common/op_host/fia_tiling_info.h` | #include (via .h:20) | `FiaTilingInfo` 声明 |
| `fused_infer_attention_score_tiling_v4.cpp` | 调用 `DoOpTiling` | .cpp:51 — 上层调度入口 |

---

## 高性能设计（Tiling侧，非Kernel侧）

Tiling 侧不直接涉及 AI Core 上的流水线/Buffer 管理，但通过以下策略影响 Kernel 性能：

**多核切分策略**:
| 维度 | 切分方式 | 说明 |
|------|----------|------|
| S1 外切分 (`SplitOutSeq`) | 按 Q 序列长度 S1 在 batch/head 维度外切分 | 当 L2 需求超过一半容量时启用，复用 L2 Cache |
| NB 切分 (`SplitNBSeq`) | 按 batch/head 切分，负载均衡 | 基于每个 (batch,head) 的 innerBlock 计算量分配核 |
| FlashDecode (`SplitS2`) | KV 长度方向切分 | 长 KV 序列场景，按 kvSplitPart 切分 |

**切分粒度控制**:
- `sOuterFactor_`（S1/Souter 方向最小块）：32/64/128，由 `AdjustSinnerAndSouter` 决定
- `sInnerFactor_`（S2/Sinner 方向最小块）：64/128/256/512/1024，由 `AdjustSinnerAndSouter` 和 `ApplySinnerSouterOverrides` 决定
- `sOuterFactor_ * CV_RATIO (2)` 为实际切分块大小

**表驱动 TilingKey**:
- 重构前：长 if-else 链（~70 行）
- 重构后：`configRules` 表（30 条规则，.cpp:1306-1338），遍历匹配
- 优势：规则集中管理，新增规则只需追加一行
