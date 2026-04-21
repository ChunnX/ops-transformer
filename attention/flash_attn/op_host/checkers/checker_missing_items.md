# npu_flash_attn Checkers 遗漏项分析与命名修正报告

## 一、遗漏项分析

### 1. cu_seqlens_q/cu_seqlens_kv 基本约束未通用检查

**文档约束位置**: 文档行 617-630 (SeqLengths参数组 - 单参数约束)

**✅ 已修复**: 在 `actual_seq_len_checker.cpp` 的 CheckSinglePara 中添加了 cu_seqlens_q/cu_seqlens_kv 的通用检查逻辑。

---

### 2. metadata 参数缺少检查

**文档约束位置**: 文档行 206-213 (参数说明表)

**✅ 已修复**: 新增 `metadata_checker.h` 和 `metadata_checker.cpp`，检查 dtype=INT32，shape=1维，dim0>0。

---

### 3. win_left/win_right 参数缺少检查

**文档约束位置**: 文档行 233-249, 592-597

**✅ 已修复**: 在 `mask_checker.cpp` 添加：
- win_left >= -1 检查
- win_right >= -1 检查
- band模式时 win_left <= win_right 检查
- causal模式(mask_mode=3)时 win_left=-1 和 win_right=-1 检查

---

### 4. attn_mask 参数缺少检查（新增）

**文档约束位置**: 文档行 234-241 (参数说明表), 618-630 (Mask参数组存在性约束)

**✅ 已修复**: 在 `mask_checker.cpp` 添加：
- 单参数校验: dtype=INT8, shape=(2048, 2048)
- 存在性校验:
  - mask_mode=1时不支持传入attn_mask
  - mask_mode=3时必须传入attn_mask
  - mask_mode=4时必须传入attn_mask
- win_left/win_right存在性校验:
  - mask_mode=1时不支持传入win_left/win_right
  - mask_mode=3时不支持传入win_left/win_right

---

### 5. seqused_q/seqused_kv 值范围检查

**文档约束位置**: 文档行 662-678 (SeqLengths参数组 - 单参数约束)

**✅ 已修复**: 在 `seq_len_checker.cpp` 添加：
- seqused_q值需 <= Q_S 检查
- seqused_kv值需 <= KV_S 检查

---

### 6. max_seqlen_q/max_seqlen_kv 值校验

**文档约束位置**: 文档行 710-724 (SeqLengths参数组 - 单参数约束)

**✅ 已修复**: 在 `seq_len_checker.cpp` 添加：
- max_seqlen_q仅支持-1和非负整数检查
- max_seqlen_kv仅支持-1和非负整数检查

---

### 7. block_table 值校验

**文档约束位置**: 文档行 748-764 (Paged Attention参数组 - 单参数约束)

**✅ 已修复**: 在 `paged_attention_checker.cpp` 添加：
- block_table值只能为正整数检查（遍历tensor数据）

---

### 8. PagedAttention layout_kv 限制检查

**文档约束位置**: 文档行 760-763 (Paged Attention参数组 - 特性交叉约束)

**✅ 已修复**: 在 `paged_attention_checker.cpp` CheckMultiPara 中添加：
- PagedAttention使能时layout_kv仅支持PA_BBND/PA_BNBD/PA_Nz检查

---

### 9. return_softmax_lse 值校验

**文档约束位置**: 文档行 826-829 (SoftmaxLSE参数组 - 单参数约束)

**✅ 已修复**: 在 `softmax_lse_checker.cpp` CheckSinglePara 中添加：
- return_softmax_lse仅支持0和1检查

---

### 10. dtype 修正（文档已改为 INT32）

**文档约束位置**: 参数说明表

**✅ 已修复**:
| 参数 | 旧dtype | 新dtype | 文件 |
|------|---------|---------|------|
| block_table | INT64 | INT32 | paged_attention_checker.cpp |
| metadata | INT64 | INT32 | metadata_checker.cpp |
| cu_seqlens_q/kv | 未检查 | INT32检查 | actual_seq_len_checker.cpp |
| seqused_q/kv | 未检查 | INT32检查 | actual_seq_len_checker.cpp |

---

### 11. flash_attn_def.cpp 语法错误

**✅ 已修复**: 第116行 `.UnknownShapeFormat({...});` 后的分号删除，与 `.AutoContiguous()` 合并。

---

## 二、命名修正（Checker文件名与文档参数名对应）

### 修正对照表

| 文档参数名 | 原Checker名 | 修正后Checker名 | 状态 |
|-----------|-------------|-----------------|------|
| sinks | learnable_sink_checker | sinks_checker | ✅ 已修正 |
| metadata | (新增) | metadata_checker | ✅ 已创建 |

---

## 三、空函数清理（根据文档约束类型）

文档中每个参数组有四种约束类型，根据文档分析，删除不需要的空函数：

### 约束类型与Checker方法对应

| 方法 | 文档约束类型 |
|------|-------------|
| CheckSinglePara | 单参数约束 |
| CheckParaExistence | 存在性约束 |
| CheckFeature | 特性交叉约束 |
| CheckMultiPara | 一致性约束（多参数） |

### 各Checker保留的方法

| Checker | 文档参数组 | CheckSinglePara | CheckParaExistence | CheckFeature | CheckMultiPara |
|---------|-----------|-----------------|--------------------|--------------|----------------|
| sinks_checker | LearnableSink | ✅ dtype/shape | ❌ 无约束 | ❌ 无约束 | ❌ 无约束 |
| metadata_checker | - | ✅ dtype/shape | ❌ 无约束 | ❌ 无约束 | ❌ 无约束 |
| mask_checker | Mask | ✅ mask_mode值/win_left/win_right | ✅ attn_mask/win存在性 | ❌ 无约束 | ❌ 无约束 |
| layout_checker | layout匹配 | ✅ layout值范围 | ❌ 无约束 | ❌ 无约束 | ✅ layout匹配关系 |
| paged_attention_checker | PagedAttention | ✅ blockTable/block_size | ✅ seqused_kv | ❌ 无约束 | ✅ layout_kv限制/shape匹配 |
| softmax_lse_checker | SoftmaxLSE | ✅ dtype/return_softmax_lse值 | ✅ lseOut非空 | ❌ 无约束 | ✅ shape匹配 |
| empty_tensor_checker | 特殊场景 | ❌ | ❌ | ❌ | ❌ (仅保留IsEmptyTensorScenario) |

### 修改说明

**base_checker.h**: 将纯虚函数改为虚函数并提供默认实现，允许子类只override需要的方法。

---

## 四、修改文件清单

### 遗漏项修复

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `actual_seq_len_checker.cpp` | 修改 | 添加 cu_seqlens_q/kv 基本约束检查、seqused值范围检查、max_seqlen值校验 |
| `metadata_checker.h/cpp` | 新增 | Metadata 参数检查器 |
| `mask_checker.cpp` | 修改 | 添加 win_left/win_right >= -1检查、causal模式win必须为-1检查 |
| `paged_attention_checker.cpp` | 修改 | 添加 block_table值正整数检查、layout_kv限制检查 |
| `softmax_lse_checker.cpp` | 修改 | 添加 return_softmax_lse值校验(仅0和1) |

### 命名修正

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `learnable_sink_checker` → `sinks_checker` | 重命名 | 与文档参数名 `sinks` 对应 |
| `fa_tiling_info.h` | 修改 | learnableSink → sinks, learnableSinkFlag → sinksFlag |
| `fa_checker.h/cpp` | 修改 | LearnableSinkChecker → SinksChecker |

### 空函数清理

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `base_checker.h` | 修改 | 纯虚函数改为虚函数+默认实现 |
| `sinks_checker.h/cpp` | 修改 | 仅保留 CheckSinglePara |
| `metadata_checker.h/cpp` | 修改 | 仅保留 CheckSinglePara |
| `mask_checker.h/cpp` | 修改 | 仅保留 CheckSinglePara、CheckParaExistence |
| `layout_checker.h/cpp` | 修改 | 仅保留 CheckSinglePara、CheckMultiPara |
| `paged_attention_checker.h/cpp` | 修改 | 仅保留 CheckSinglePara、CheckParaExistence、CheckMultiPara |
| `softmax_lse_checker.h/cpp` | 修改 | 仅保留 CheckSinglePara、CheckParaExistence、CheckMultiPara |
| `empty_tensor_checker.h/cpp` | 修改 | 仅保留 IsEmptyTensorScenario |

---

## 五、单元测试覆盖

### 测试文件位置
`attention/flash_attn/tests/ut/op_host/checkers/test_checkers.cpp`

### 测试覆盖矩阵

| Checker | 测试用例 | 覆盖的文档约束 |
|---------|---------|--------------|
| MaskChecker | MaskChecker_ValidMaskMode_1/3/4 | mask_mode支持范围(1/3/4) |
| MaskChecker | MaskChecker_InvalidMaskMode_0/2 | mask_mode非法值拦截 |
| MaskChecker | MaskChecker_BandMode_WinLeftGreaterThanWinRight | band模式win_left<=win_right |
| MaskChecker | MaskChecker_WinLeftLessThanMinusOne | win_left >= -1 |
| MaskChecker | MaskChecker_WinRightLessThanMinusOne | win_right >= -1 |
| MaskChecker | MaskChecker_CausalMode_WinNotMinusOne | causal模式win必须为-1 |
| ShapeChecker | ShapeChecker_BSizeValid/InvalidZero/TooLarge | B轴范围(0, 65536] |
| ShapeChecker | ShapeChecker_DSizeValid/Not16Aligned/TooLarge | D轴范围、16对齐 |
| ShapeChecker | ShapeChecker_NHeadRatioValid/Invalid | N_Q是N_KV整数倍 |
| LayoutChecker | LayoutChecker_ValidBSND/BNSD/TND | layout值范围验证 |
| LayoutChecker | LayoutChecker_BNSD_WithBSNDOut/PagedAttentionKV | layout匹配关系 |
| LayoutChecker | LayoutChecker_InvalidBSNDWithBNSDOut/TNDWithBSNDOut | 非法layout组合拦截 |
| SoftmaxLSEChecker | SoftmaxLSEChecker_EnabledBSNDLayout/TNDLayout | lseOut shape匹配(B,N,Q_S)/(N,T_Q) |
| SoftmaxLSEChecker | SoftmaxLSEChecker_ReturnSoftmaxLseValue | return_softmax_lse仅支持0/1 |
| PagedAttentionChecker | PagedAttentionChecker_Enabled | blockTable shape验证 |
| PagedAttentionChecker | PagedAttentionChecker_BlockTableValuePositive | blockTable值必须为正整数 |
| PagedAttentionChecker | PagedAttentionChecker_LayoutKvRestriction | PagedAttention时layout_kv限制 |
| ActualSeqLenChecker | ActualSeqLenChecker_SequsedValid/Invalid/Negative | seqused_q/kv约束 |
| ActualSeqLenChecker | ActualSeqLenChecker_SequsedValueRange | seqused值<=Q_S/KV_S |
| ActualSeqLenChecker | ActualSeqLenChecker_CuSeqlensValid/FirstNotZero/Negative | cu_seqlens_q/kv约束 |
| ActualSeqLenChecker | ActualSeqLenChecker_MaxSeqlenValue | max_seqlen仅支持-1或非负整数 |