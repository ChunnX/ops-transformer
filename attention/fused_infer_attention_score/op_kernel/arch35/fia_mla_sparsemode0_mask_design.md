# arch35 fia\*mla 模板支持 sparseMode=0 + 自定义 mask 方案

> 目标目录：`attention/fused_infer_attention_score/op_kernel/arch35/`
> 适用模板：`fia_kernel_noquant_mla.h`、`fia_block_cube_noquant_mla.h`、`fia_block_vec_noquant_mla.h`、`fia_block_vec_flashdecode_mla.h`（以 `fia` 开头、`mla` 结尾）
> 参考模板：同目录下 `fia_*_gqa.h`（noquant/fullquant 重构模板）、`fia_block_vec_flashdecode_fullquant.h`，以及 `op_kernel/arch22/fia_*_mla.h` 老模板

---

## 1. 背景与目标

MLA D512（Q: D=576 = 512 + rope 64，V: D=512）非量化场景，现有两条模板链路：

| 链路 | 模板 | 状态 |
|---|---|---|
| 老模板（线上在用） | `flash_attention_kernel_noquant_mla.h` / `flash_attention_block_cube_noquant_mla.h` + `FANoQuantBlockVecInfer`，经 `fia_arch35_noquant.h` 非 reconstruct 分支 `INVOKE_FA_OP_IMPL_ASCEND950_KVSAME_BASEAPI` 调用 | tiling key 已开（含 `HasAttenMask`），自带老式 mask 机制（`AttenMaskInfo` + compressMode） |
| **新模板（fia\*mla，重构版）** | `fia_kernel_noquant_mla.h` / `fia_block_cube_noquant_mla.h` / `fia_block_vec_noquant_mla.h` / `fia_block_vec_flashdecode_mla.h`，经 `run_fia_noquant_mla_kernel`（`fia_template_dispatcher.h:77`）调用 | **当前被整体禁用**（见 §2.2），但 mask 机制已按 GQA 重构模板同步实现 |

**目标**：把 fia\*mla 新模板链路打开，支持 **sparseMode=0（`SPARSE_MODE_NO_MASK` / kernel 侧 `DEFAULT_MASK`）且传入自定义 attenMask** 的场景（即全量 mask 参与计算）。

---

## 2. 现状盘点

### 2.1 sparseMode 枚举语义（kernel 侧，`attention/common/op_kernel/vector_common.h:45`）

```cpp
enum SparseMode : uint8_t {
    DEFAULT_MASK = 0,   // sparseMode=0：pre/next 决定 S2 遍历范围，attenMask 全量参与
    ALL_MASK,           // 1
    LEFT_UP_CAUSAL,     // 2（因果）
    RIGHT_DOWN_CAUSAL,  // 3
    BAND,               // 4
    TREE = 9,
};
```

host 侧对应 `op_host/fia_tiling_info.h:69`：`SPARSE_MODE_NO_MASK=0 / ALL_MASK=1 / LEFT_UP=2 / RIGHT_DOWN=3 / BAND=4`。

**sparseMode=0 且带 mask 的执行语义**（与 fullquant GQA 参照模板一致）：
- S2 遍历范围由 `preTokens/nextTokens` 经 `GetSafeActToken`（`vector_common.h:71`）钳制后决定；API 默认 `preTokens=nextTokens=INT_MAX` 时钳制到 `[actSeqLensKv, actSeqLensQ]`，等价于遍历**全量 S2**；
- mask 拷贝不跳过：`IsSkipAttentionmask`（`vector_common.h:953`）对 `DEFAULT_MASK/ALL_MASK` 恒返回 `false`；
- mask 通过共享 VF（`vf_mul_sel_softmaxflashv2_cast_nz.h` 的 `ProcessVec1Vf`）在 softmax 前叠加到 QK 结果。

### 2.2 fia\*mla 链路被禁用的两个开关（现状）

| # | 位置 | 现状 |
|---|---|---|
| 1 | `op_host/arch35/fia_tiling_nonquant_mla.cpp:99` `IsCapable()` | **`return false;` 硬禁用**（commit `ad4f53b8f` "disable nonquant MLA IsCapable"），host 永远不会路由到 MLA 模板 |
| 2 | `op_kernel/arch35/fused_infer_attention_score_template_tiling_key.h` 约 1457-1473 行 "重构MLA模板tilingkey" 段 | **整体被注释**（commit `d4e1e3e15` "disable reconstruct MLA template tilingkey"），kernel 侧无实例化 |

除此之外，**kernel 与 host 的 mask 机制均已就绪**（见 §3、§4），因此本方案的核心是"解锁 + 核对 + 验证"，而非新写 mask 逻辑。

### 2.3 各层 mask 能力现状（fia\*mla 模板自身）

**kernel 主调度 `fia_kernel_noquant_mla.h`**
- `InitConstInfo` 已读取 `sparseMode / preTokens / nextTokens`，且 `HAS_MASK`（= `VecFaBlockType::HAS_MASK`）时读取 `attenMaskBatch / attenMaskS1Size / attenMaskS2Size`（216-223 行）；
- `GetTaskDealMode`：`HAS_MASK` 时走 `CalcCurS2StartEndWithSparse`，否则 `CalcCurS2StartEndNoSparse`（425-430 行）；
- `CalcCurS2StartEndWithSparse` + `GetPreNextTokenLeftUp`（454-545 行）与 GQA noquant/fullquant 重构模板逻辑一致（含 `GetSafeActToken` 钳制、BAND/RIGHT_DOWN 偏移）。

**vec 块 `fia_block_vec_noquant_mla.h`**（与 `fia_block_vec_noquant_gqa.h` 逐行对齐）
- `HAS_MASK = hasAtten`（59 行）、`attenMaskGmInt`（89 行）、`attenMaskInQue[2]` ping-pong（98 行）；
- `ProcessVec1Nd` 中 `AttenMaskCopyIn(attenMaskUb, 0, runInfo.actVecMSize, runInfo)`（311-316 行）→ `ProcessVec1Vf<..., hasAtten, ...>` 应用 mask；
- `AttenMaskCopyIn`（717-762 行）：填充 `MaskInfo`（含 `sparseMode`、`attenMaskBatch`、S1/S2 stride、`maskValue=negativeIntScalar`、`maskFormat=MASK_LAYOUT`），`IsSkipAttentionmask`/`IsSkipAttentionmaskForPre` 判定后经共享 `AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSize>` 拷贝，必要时 `MergeMask`（BAND 场景）；
- `RowInvalid`（522-548 行）：`isRowInvalidOpen` 或非 LEFT_UP_CAUSAL 且 pre/next 为负时对无效行置 `-inf`。

**FD 块 `fia_block_vec_flashdecode_mla.h`**
- `attenMaskFlag = hasAtten`（76 行）；
- `CalcPreNextTokens`（350-369 行）：`DEFAULT_MASK && attenMaskFlag` 分支已存在；
- `DealInvalidRows`（371-395 行）、`DealInvalidMaskRows`（397-415 行，`isRowInvalidOpen && sparseMode∈{0,1}`）均已实现。FD 不再重复加载 mask（prefill 阶段已作用于累加结果），仅做行无效处理，语义正确。

**cube 块 `fia_block_cube_noquant_mla.h`**：与 mask 无关，无需改动。

**host `op_host/arch35/fia_tiling_nonquant_mla.cpp`**
- `IsCapableSparseLayoutCheckMla`（77-97 行）：已允许 `sparseMode ∈ {0, 3, 4}`，**没有** GQA 那样的 `sparseMode==0 && attenMaskFlag` 拒绝分支（GQA 在 `fia_tiling_nonquant_gqa.cpp:138` 拒绝了该组合，MLA 不需要）；
- `SplitPolicy` → 共享 `split_core_v2::SplitCore`，已支持 `DEFAULT_MASK`；
- `IsExistRowInvalid`（363-408 行）：已处理 `DEFAULT_MASK` 分支；
- `SetAttenMaskTilingData`（551-571 行）：填充 `attenMaskBatch / S1Size / S2Size / sparseMode`；`preTokens/nextTokens`（612-613 行）、`isRowInvalidOpen / isExistRowInvalid`（614-615 行）均已写入；
- `GenTilingKey`（487-503 行）：`hasAttenMask = fiaInfo_->attenMaskFlag`（468 行）、`isReconstructTemp = true`（485 行）——tiling key 位已绑定 mask。

**kernel 路由 `fia_arch35_noquant.h:88-97`**（reconstruct 分支）
```cpp
if constexpr (config == Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512 && hasRope) {
    run_fia_noquant_mla_kernel<half, half, inOutLayoutType, config, hasAttenMask, KvLayoutType,
        isFd, emptyTensor, enableS1OutSplit>(...);
} else {
    run_fia_noquant_gqa_kernel<...>(...);
}
```
`run_fia_noquant_mla_kernel`（`fia_template_dispatcher.h:77-125`）已把 `hasAttenMask` 透传给 `FANoQuantMlaBlockVec<..., hasAttenMask, ...>` 与 `FiaBlockVecFlashDecodeMla<..., hasAttenMask, ...>`。

---

## 3. 参照模板对照（结论：机制同源，可直接复用）

| 参照模板 | 可复用的做法 |
|---|---|
| `fia_block_vec_noquant_gqa.h` / `fia_kernel_noquant_gqa.h`（重构 GQA，**线上已开**，tiling key 含 `HasAttenMask: false, true`） | `AttenMaskCopyIn` / `MaskInfo` / `RowInvalid` / `CalcCurS2StartEndWithSparse` 与 MLA 模板**逐行一致**，是 fia\*mla 的直接模板参照 |
| `fia_kernel_fullquant_gqa.h` / `fia_block_vec_fullquant_gqa.h` | host `IsCapable()` **无 sparse/mask 检查**，sparseMode=0+mask、LEFT_UP、BAND 全场景放行；`GetPreNextTokenLeftUp` 的 BAND/RIGHT_DOWN 偏移写法与 MLA 模板相同 |
| `arch22/fia_block_vec_nonquant_mla.h` / `arch22/fia_kernel_nonquant_mla.h` | arch22 MLA 老模板已支持 mask：`MaskInfo` 填充 + `IsSkipAttentionmask` + `AttentionmaskCopyIn` + `AttentionMaskCompute`（445-510 行）、`DealInvalidRows/DealInvalidMaskRows`（646-677 行）——证明 MLA 算法带 mask 的可行性与需求 |
| 共享层 `attention/common/op_kernel/vector_common.h` | `SparseMode` 枚举、`GetSafeActToken`、`IsSkipAttentionmask`、`AttentionmaskCopyIn`（GS/SG 布局）、`MergeMask` —— 全部为 MLA/GQA 共用，无 MLA 专属缺口 |

---

## 4. 差距分析

**kernel 侧：无需新增 mask 逻辑**。逐项核对结论：

1. sparseMode=0 + mask 的 S2 范围：`CalcCurS2StartEndWithSparse` + `GetSafeActToken` 处理正确（API 默认 pre/next=INT_MAX → 钳制后等价全量 S2）；
2. mask 加载：`IsSkipAttentionmask(DEFAULT_MASK)=false` → 全量拷贝，`AttenMaskCopyIn` 字段（batch/S1/S2 stride、左右 padding、`maskFormat=SG/GS`）已按 layout 正确填充；
3. mask 应用：共享 `ProcessVec1Vf` 的 `hasAtten` 分支；
4. 行无效：`RowInvalid`、FD 的 `DealInvalidRows/DealInvalidMaskRows` 已实现；
5. FD：`CalcPreNextTokens` 的 `DEFAULT_MASK && attenMaskFlag` 分支已存在。

**host 侧：两个开关未打开**（§2.2），是本方案要动的核心。

**唯一语义差异需确认**：MLA host `IsCapableSparseLayoutCheckMla` 对 `attenMaskFlag` 无约束（与 GQA 的拒绝策略相反）。这是**有利**差异——sparseMode=0 + mask 天然通过，无需新增放行分支；但要注意与老模板/checker 的路由一致性（见 §5.4）。

---

## 5. 方案设计（改动点清单）

### 5.1 【必改】Host：解除 `IsCapable()` 硬禁用（✅ 已实现）

`op_host/arch35/fia_tiling_nonquant_mla.cpp:99-112`：

```cpp
bool FiaTilingNonQuantMlaArch35::IsCapable()
{
-   return false;   // 删除此硬开关
    if (!IsCapableBasicCheckMla()) { return false; }
    if (!IsCapableFeatureCheckMla()) { return false; }
    if (!IsCapableSparseLayoutCheckMla()) { return false; }
    return true;
}
```

同时核对 `IsCapableSparseLayoutCheckMla`（77-97 行）保持对 `sparseMode=0`（含 `attenMaskFlag=true`）放行：
- `SPARSE_MODE_NO_MASK` 已在允许集合内；
- 无需像 GQA 那样新增 `sparseMode==0 && attenMaskFlag → false`（MLA kernel 已支持，否则会误杀目标场景）；
- 维度约束（qkD=512、ropeD=64、vD=512、n2=1、n1∈[1,128]）保持不变。

### 5.2 【必改】Tiling key：恢复"重构MLA模板tilingkey"段（✅ 已实现，全量恢复未裁剪）

`op_kernel/arch35/fused_infer_attention_score_template_tiling_key.h` 约 1457-1473 行，取消注释：

```cpp
ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD, InOutLayoutType_BSH_BSH, InOutLayoutType_TND_TND, InOutLayoutType_NTD_NTD),
    ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512),
    ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
    ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode),
    ASCENDC_TPL_BOOL_SEL(HasAttenMask, false, true),   // ← sparseMode=0 + mask 走 true
    ASCENDC_TPL_BOOL_SEL(HasRope, true),
    ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
    ASCENDC_TPL_BOOL_SEL(IsFd, false, true),
    ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0),
    ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
    ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false),
    ASCENDC_TPL_BOOL_SEL(IsReconstructTemp, true),
    ASCENDC_TPL_TILING_STRUCT_SEL(FusedInferAttentionScoreTilingData)
),
```

注意事项：
- 该段是唯一 `IsReconstructTemp=true && HasRope=true` 的 config，与 `fia_arch35_noquant.h:89` 的 reconstruct 分支条件（`config==DAligned576 && hasRope`）精确匹配，恢复后**不会**误入 GQA kernel 分支；
- 全量恢复 = 4 layout × 2 mask × 4 KV 布局 × 2 isFd = 64 个实例 × 2 种 dtype（half/bf16），编译压力大。**建议首期裁剪**：先只开目标布局（如 `BNSD_BNSD` + `BSH_BSH`）与 `KvLayoutType_NO_PA`，mask 与 isFd 按需（isFd 涉及 FD 场景，见 §5.5）。验证通过后再逐步扩展；
- 与 `GenTilingKey`（host 侧绑定 `hasAttenMask`、`isReconstructTemp=true`、`hasRope=true`）必须一一对应，任何 host/kernel 枚举不一致会导致 tilingkey 查表失败。

### 5.3 【核对】Kernel 侧核对项（预期无需改动）

恢复后按 §4 逐项验证，重点关注：

1. **vec UB mask 缓冲容量**（`fia_block_vec_noquant_mla.h:681-682`）：`attenMaskInQue[0/1]` 各 4096B = 32 行（mBaseSize=64，vec1 半行 32）× 128 列（s2BaseSize），当前 config 恰好容纳全量 mask tile；**若未来扩展 S2=256 需扩到 8192B**（GQA 模板即为 8192B）；
2. **`MASK_LAYOUT` 推导**（66-68 行）：BSH/TND→`SG`，BNSD→`GS`，与 GQA 一致；NTD 场景注意确认 mask 形状语义；
3. **`maskInfo.batchIdx`**（732 行）：`attenMaskBatch==1 ? 0 : bIdx`，与 host `SetAttenMaskTilingData` 的 `attenMaskBatch` 推导一致（dimNum!=2 或 s1Size==1 时取 dim0）；
4. **FD + mask 组合**（isFd=true && hasAttenMask=true）：`CalcPreNextTokens` 的 `DEFAULT_MASK && attenMaskFlag` 分支、`DealInvalidRows`、`DealInvalidMaskRows` 已在；验证 FD 场景下 `isExistRowInvalid` 由 `FLASH_DECODE ? HAS_MASK : ...` 兜底（166-183 行）；
5. **workspace**：host `CalcMaxWorkspaceSize` 与 kernel `InitMMResBuf`（`fia_kernel_noquant_mla.h:165-176`）在 mask 场景不变，无需调整；
6. **`negativeIntScalar`**（`BOOL_ATTEN_MASK_SCALAR_VALUE = -1e12`）为 mask 的加性惩罚值，bool/int8/uint8 mask 兼容，与参照模板一致。

### 5.4 【核对】Checker 与路由一致性（✅ 已核对/已放宽，结论见 §6）

- `op_host/checkers/common_checker.cpp` 当前对 MLA D512 的 G 轴/head 数按 {1,2,4,...128} 2 的幂集合检查（commit `ad4f53b8f` 已 revert 掉 arch35 非量化的放宽分支）。**已恢复放宽**：arch35 非量化（`npuArch==DAV_3510`）时 G/N1 ∈ [1,128] 任意整数（见 §6.4）；
- 确认无 checker 对 `sparseMode=0 && attenMask 存在` 的拒绝逻辑（MLA 场景下该组合合法）；
- 老模板（`flash_attention_*_mla`）与 fia\*mla 的路由优先级：`FiaTilingNonQuantMlaArch35` 注册顺序/模板注册表（`common/op_host/fia_tiling_templates_registry.h`）决定谁先命中，解锁后需确认 MLA D512 场景稳定路由到 fia\*mla（或按计划灰度切换）。

### 5.5 【必做】测试用例

1. **扩展示例**：`examples/test_aclnn_fused_infer_attention_score_v4_mla_noquant.cpp` 当前 `sparseMode=0` 且未传 mask（200-217 行）。新增用例：
   - sparseMode=0 + attenMask（形状 `S1×S2` 与 `B×S1×S2` 各一），布局 BNSD / BSH / TND 各一；
   - preTokens/nextTokens 默认 `INT_MAX`（全量 S2）与显式 `0/0`（对角窗口 + mask）各一，验证 `CalcCurS2StartEndWithSparse` 两种边界；
   - `innerPrecise` 打开行无效（`isRowInvalidOpen`）时 mask 行无效行为（`RowInvalid` / FD `DealInvalidMaskRows`）；
   - isFd=true 组合（decode 场景 + mask）回归 `DealInvalidRows`。
2. **精度对照**：与老模板（`flash_attention_*_mla`）或 fullquant GQA 同参数结果对齐（mask 生效后 softmax 分母/分子一致）。
3. **编译**：验证裁剪后的 tiling key 段编译时间与二进制增量可控（重点 isFd/mask 组合数）。

---

## 6. 自定义 mask 与 pre/next 特性约束（checker 支持范围核对结论）

> 本节是 2026-08 对 `op_host/checkers/` 逐条核对的结论，供实现与测试对照。行号以当前工作区为准。

### 6.1 场景判定与量化模式边界

- MLA decode D512 判定（`fused_infer_attention_score_tiling_info_parser.cpp:1018`）：`ropeMode==ROPE_SPLIT`（query_rope/key_rope 均存在且 D 相等）且 `qkHeadDim==512` → `mlaMode=ROPE_SPLIT_D512`。实际形状：Q: D=512 + query_rope 64，K: D=512 + key_rope 64，V: D=512。
- **sparseMode=0 + 自定义 mask 仅在非量化（fp16/bf16）MLA decode 放行**：
  - 全量化被拦：`CheckFullQuantIFAMLA`（`mask_checker.cpp:97`）——Q dtype 为 fp8/hifloat8/int8 且 mask 非空时必须 sparseMode=3；INT8 Q + 非 TND/TND_NTD layout 且 QS==1（decode）时 mask 必须为空。即全量化下 sparse0+mask 组合 checker 直接拒绝。
  - antiquant 与 MLA decode 互斥：`CheckFeatureAntiQuant`（`rope_checker.cpp:469`）要求 query_rope/key_rope 必须为空，而 ROPE_SPLIT 判定依赖 rope tensor 存在 → MLA decode 不存在 antiquant 分支。

### 6.2 mask 约束（sparseMode=0）

- dtype：bool / int8 / uint8；format：ND / NCHW / NHWC / NCDHW（`mask_checker.cpp:33`）。
- 维数：**3D `[B或1, >=Q_S, >=KV_S+sysPrefix]` 或 4D `[B或1, 1, >=Q_S, >=KV_S+sysPrefix]`**；**2D 在 arch35 被拒**（`mask_checker.cpp:453`，仅 910B + no-rope + 特定 layout 支持 2D）——测试/用户侧必须传 3D/4D，设计文档 §5.5 规划中的 "S1×S2 二维 mask 用例" 需改为 3D。
- 下限（`mask_checker.cpp:538`）：maskQ_S >= s1Size（decode 即 >=1）；maskKV_S >= maxActualseq + systemPrefixLen（非 PA）或 s2Size + systemPrefixLen（PA）；batch ∈ {1, B}；4D 的 N 维 ==1。
- 对照：sparseMode=3/4 时 mask 必须**精确** `[2048,2048]`（`SPARSE_OPTIMIZE_ATTENTION_SIZE`，`mask_checker.cpp:542`）；sparse0 是 `>=` 下限，宽松得多。
- sparseMode=0 不传 mask：合法，跳过全部 mask shape 校验（`mask_checker.cpp:586`）。

### 6.3 pre/next token 约束

- checker 硬约束（`mask_checker.cpp:308`）：`nextToken*(-1) > preToken` 时拒绝，等价 **`preToken + nextToken >= 0`**；默认 INT_MAX/INT_MAX（parser 未传时置 `SPARSE_MODE_INT_MAX`）通过。报错文案 "next_tokens * (-1) > pre_tokens" 写的是非法条件，注意别误读。
- **只传 pre/next 不传 mask 不生效**：`HAS_MASK=false` 时 `GetTaskDealMode`（`fia_kernel_noquant_mla.h:426`）走 `CalcCurS2StartEndNoSparse`（`:473`），S2 范围固定全量 `[0, actSeqLensKv]`，pre/next **完全不消费**；老模板同语义（`flash_attention_kernel_noquant_mla.h:335` 只写入 `attenMaskInfo`）。即 pre/next 是 mask 的配套参数，不能单独作为窗口参数。
- 带 mask 生效路径：`CalcCurS2StartEndWithSparse` → `GetPreNextTokenLeftUp` + `GetSafeActToken` 钳制（pre ∈ [-actSeqLensKv, actSeqLensQ]，next ∈ [-actSeqLensQ, actSeqLensKv]）。
- 窗口语义：mask 只作用于窗口内 S2 列，窗口外 KV 不参与计算、不做 -inf 置位；默认 INT_MAX → 全量 S2 + 全量 mask 生效。
- 行无效：host `IsExistRowInvalid`（`fia_tiling_nonquant_mla.cpp:363`）无 mask 直接 false；带 mask 且 `preTokenLeftUp = s2Size - s1Size + safePre` < 0 时置 `isExistRowInvalid`；`isRowInvalidOpen = (innerPrecise >> 1) & 1`；kernel `RowInvalid`（`fia_block_vec_noquant_mla.h:522`）、FD `DealInvalidRows/DealInvalidMaskRows`。

### 6.4 N1/G 轴支持范围放宽（arch35 非量化，已实现）

- 放宽条件：`enableNonQuant_ && fiaInfo.npuArch == NpuArch::DAV_3510`（**正向判定**，不用 `socVersion != 910B`，与 checkers 既有惯例一致，如 `common_checker.cpp:1001`）。
- N1 ∈ **[1, 128] 任意整数**，G ∈ **[1, 128] 任意整数**（MLA decode 下 N2==1，故 G==N1）；其他 arch/量化模式保持 2 的幂集合 {1,2,4,8,16,32,64,128}。
- 位置：`common_checker.cpp` `CheckAxis`（G 轴）/ `CheckHeadNum`（num_heads）、`rope_checker.cpp` `CheckNSize`（N 轴），三处同步放宽。
- 与 host `IsCapableSparseLayoutCheckMla`（`fia_tiling_nonquant_mla.cpp:93`，n1 ∈ [1,128] 任意值）对齐，消除此前"checker 拦 2 的幂、host 放行任意值"的矛盾。

### 6.5 其他边界（与 mask 正交但影响路由）

- sparseMode=9（TREE）：checker 对 MLA 非量化放行，但 host `IsCapableSparseLayoutCheckMla` 只放行 0/3/4 → sparse9 到不了 fia\*mla 模板（走老模板）。
- PA 场景：blockSize ∈ [16, 1024] 且 16 对齐；mask KV_S 下限为 s2Size + systemPrefixLen（需覆盖完整 S2）。
- B ∈ (0, 65536]；QS >= 1（fullquant 时 [1, 16]）；N2 == 1；布局白名单 BSH/BSND/BNSD/TND/BNSD_NBSD/BSND_NBSD/BSH_NBSD/TND_NTD。

## 7. 关键语义说明（sparseMode=0 + mask 的执行路径）

以 `Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512`、`hasAttenMask=true`、`isFd=false` 为例：

```
host: IsCapable 放行(sparseMode=0) 
  → SplitCore 分核(DEFAULT_MASK 已支持) → IsExistRowInvalid 置位
  → FillTiling(sparseMode=0, pre/next=INT_MAX, mask 维度, isRowInvalidOpen)
  → GenTilingKey(hasAttenMask=1, hasRope=1, isReconstructTemp=1)
kernel: run_fia_noquant_mla_kernel<..., hasAttenMask=true, ...>
  → FlashAttentionNoQuantMlaKernel::InitConstInfo(sparseMode=0, mask 维度)
  → GetTaskDealMode: HAS_MASK → CalcCurS2StartEndWithSparse
      pre/next 经 GetSafeActToken 钳制 → S2 范围≈全量
  → ProcessVec1: AttenMaskCopyIn(IsSkipAttentionmask=false → 全量拷贝)
      → ProcessVec1Vf<hasAtten=true> 叠加 mask(-1e12) 后 softmax
  → ProcessVec2 → RowInvalid(必要时) → CopyOut
```

---

## 8. 风险与注意事项

| 风险 | 说明 | 缓解 |
|---|---|---|
| tiling key 枚举漂移 | host `GenTilingKey` 与 kernel 模板表必须一一对应 | 恢复时逐位核对 11 个字段；建议写单测断言 tilingkey 数值 |
| 编译时长/产物膨胀 | 全量恢复 64 实例 × 2 dtype | 首期裁剪 layout/KV 布局/isFd 组合 |
| mask 缓冲容量 | 当前 4096B 仅适配 32×128 | 不做 S2=256 扩展即无风险；扩展需同步放大 |
| 与老模板路由竞争 | 解锁后 fia\*mla 与 `flash_attention_*_mla` 并存 | 核对模板注册表顺序与 IsCapable 优先级，按灰度策略切换 |
| checker 前置拦截 | G/head 数的 2 的幂限制可能挡掉目标 MLA 场景 | **已消除**：arch35 非量化 N1/G ∈ [1,128]（`npuArch==DAV_3510` 正向判定，见 §6.4）；全量化场景若需放宽另行评估 |
| FD 场景回归 | mask + isFd 组合涉及行无效与 LSE 写回 | §5.5 测试用例覆盖，对照老模板结果 |

---

## 9. 实施步骤（Checklist）

**当前状态（2026-08-10）：解锁类改动已全部落地（工作区，未提交）；验证类未开始。**

- [x] 1. `op_host/arch35/fia_tiling_nonquant_mla.cpp`：删除 `IsCapable()` 中的 `return false;`——**已实现**，`IsCapableSparseLayoutCheckMla` 放行 sparseMode=0（含 mask）
- [x] 2. `op_kernel/arch35/fused_infer_attention_score_template_tiling_key.h`：恢复"重构MLA模板tilingkey"段——**已实现（全量恢复：4 layout × 2 mask × 4 KV 布局 × 2 isFd，未按 §5.2 建议裁剪）**，与 `fia_arch35_noquant.h:89` 分支条件、host `GenTilingKey` 三处一致
- [x] 3. `op_host/checkers/` 对 MLA D512 的 G/head 检查放宽——**已实现**：arch35 非量化（`npuArch == DAV_3510` 正向判定）N1/G ∈ [1, 128] 任意整数，其他场景保持 2 的幂集合（见 §6.4）
- [x] 4. Kernel 侧按 §5.3 逐项核对——**已核对**：git diff 无 kernel 改动，mask 机制（UB 缓冲、FD 组合、workspace）均已就绪，符合"预期无改动"
- [ ] 5. 新增/扩展测试用例（§5.5），对照老模板与 fullquant GQA 精度——**未实现**：现有 pytest（`tests/pytest/test_mla_d512_runner.py`、`gen_mla_cases.py`）仍为 sparse=0 无 mask（`atten_mask=None`、pre/next 写死 INT_MAX）；sparse0+mask、显式 pre/next（如 0/0 对角窗口）、innerPrecise 行无效、isFd+mask 用例均缺失；注意 §6.2：mask 用例只能传 3D/4D
- [ ] 6. 回归：sparseMode=0 无 mask（保持现状行为）、sparseMode=3/4（已有放行）不劣化——**未做**
