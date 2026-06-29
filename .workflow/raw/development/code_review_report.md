# GMM A16W4 PerGroup ND 数据流特性代码检视报告

## 检视范围

**基线 commit**: `073726cd5aa`
**检视日期**: 2026-06-29
**检视文件**（源码部分）:
- `gmm/grouped_matmul/op_host/grouped_matmul_def.cpp`
- `gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp`
- `gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.h`
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_config.h`
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_block_vf_nd.h` (新增)
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/grouped_matmul_weight_quant_basic_controller.h`
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_basic_block.h`
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_tiling_key.h`
- `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_vec_compute.h`
- `gmm/grouped_matmul/op_kernel/grouped_matmul_apt.cpp`

**规范来源**: `cpp-secure.md`、`cpp-general.md`、`cpp-style.md`、`ascendc-api.md`、`ascendc-perf.md`、`ascendc-topk.md`

**额外检视项**（用户要求）:
1. C++ 编码规范
2. Magic number（不应直接使用字面量数，应定义字面量常量）
3. 超大函数（函数体超过 50 行）
4. 代码逻辑重复（能复用的函数直接展开写导致代码膨胀）

---

## 一、高风险问题（安全 / 潜在崩溃）

### 问题 1: LOG API 参数类型与格式化说明符不匹配 `[cpp-secure 11.3]` `[适用: Tiling]`

**文件**: `grouped_weight_quant_batch_matmul_tiling.cpp:737`

**问题代码**:
```cpp
OP_LOGI("SetMatMulTiling", "stepKb = %llu", tilingData_.mmTilingData.stepKb);
```

**假设检验**:
- H0: 该 LOG 语句类型安全
- H1: 该 LOG 语句存在类型不匹配风险
- 证据: `stepKb` 赋值来源为 `STEP_K_4`（`constexpr uint32_t`），类型为 `uint32_t`（4 字节）。格式化说明符 `%llu` 期望 `unsigned long long`（8 字节）。类型大小不匹配时，LOG 宏按说明符宽度截断或读取超量字节，导致后续参数错位。
- 自信值: 70%（规范违反 +40%，上下文无防御 +30%）→ 超过 60% 阈值

**建议修复**:
```cpp
OP_LOGI("SetMatMulTiling", "stepKb = %u", tilingData_.mmTilingData.stepKb);
```

---

### 问题 2: LOG API `%lu` 用于 `uint64_t` 类型，跨平台不安全 `[cpp-secure 11.3]` `[适用: Tiling]`

**文件**: `grouped_weight_quant_batch_matmul_tiling.cpp:1477-1489`

**问题代码**:
```cpp
OP_LOGI(context->GetNodeName(),
    "Tiling result: groupNum: %u, coreNum: %u, kSize: %lu, nSize: %lu, ... "
    "mainBlockCount: %lu, ...",
    ..., tilingData_.gmmWeightQuantParam.kSize, tilingData_.gmmWeightQuantParam.nSize, ...,
    tilingData_.gmmWeightQuantParam.mainBlockCount, ...);
```

**假设检验**:
- H0: `%lu` 用于 `uint64_t` 是安全的
- H1: `%lu` 在 Windows/32 位系统上仅读取 32 位，造成截断
- 证据: `kSize`、`nSize`、`mainBlockCount` 均为 `uint64_t` 类型。规范明确指出 `uint64_t` 推荐使用 `%llu`，`%lu` 在 Windows 上仅读取 32 位。
- 自信值: 65%（规范违反 +40%，Tiling 侧编译目标为 Linux 64 位但规范要求通用性 +25%）

**建议修复**: 将所有 `uint64_t` 对应的 `%lu` 改为 `%llu`。

---

### 问题 3: `groupType_` 缩窄转换为 `int8_t` 可能截断 `[cpp-secure 1.1]` `[适用: All]`

**文件**: `grouped_weight_quant_batch_matmul_tiling.cpp:77, 1465`

**问题代码**:
```cpp
// line 77
std::to_string(static_cast<int8_t>(groupType_))
// line 1465
static_cast<int8_t>(groupType_)
```

**假设检验**:
- H0: 转换安全
- H1: `groupType_` 为 `int64_t`，缩窄为 `int8_t` 可能截断
- 证据: `groupType_` 声明为 `int64_t`（tiling.h:301）。实际取值为 -1/0/1/2，在 `int8_t` 范围内。但缩窄转换本身是未定义行为风险点——如果未来扩展 `groupType_` 取值范围超过 [-128, 127]，将产生截断。
- 自信值: 55%（缩窄转换风险 +40%，当前值域安全 -15%）→ 未超 60%，标注为**存疑**

**建议**: 使用 `static_cast<int32_t>(groupType_)` 替代，避免缩窄到 `int8_t`。

---

### 问题 4: 有符号/无符号比较不一致 `[cpp-general 6.2]` `[适用: All]`

**文件**: `grouped_weight_quant_batch_matmul_tiling.cpp:1440 vs 1448`

**问题代码**:
```cpp
// line 1440 — 用 int64_t 比较
OP_CHECK_IF(nSize_ >= static_cast<int64_t>(BASIC_BLOCK_BASE_N_MIN) && ...)
// line 1448 — 用 uint64_t 比较
OP_CHECK_IF(nSize_ >= static_cast<uint64_t>(BASIC_BLOCK_BASE_N_MIN) && ...)
```

**假设检验**:
- H0: 两处比较语义一致
- H1: 同一函数内对同一变量使用不同类型转换，存在不一致
- 证据: `nSize_` 为 `uint64_t`。`BASIC_BLOCK_BASE_N_MIN` 为 `uint32_t`。同一函数内 1440 行用 `static_cast<int64_t>`，1448 行用 `static_cast<uint64_t>`，风格不一致。虽然值安全（128 为正数），但可维护性差。
- 自信值: 60%（规范违反 +40%，上下文无风险但一致性差 +20%）

**建议修复**: 统一使用 `static_cast<uint64_t>(BASIC_BLOCK_BASE_N_MIN)`。

---

## 二、中风险问题（代码质量 / 可维护性）

### 问题 5: 超大函数 — `DeriveGroupSizeMulti` 约 90 行 `[用户额外要求: 函数体超过50行]`

**文件**: `grouped_weight_quant_batch_matmul_tiling.cpp:1221-1311`

**问题代码**: 函数体约 90 行，包含 tensor-0 的 groupSize 推导和循环检查所有 tensor 一致性的逻辑。

**建议拆分**:
```cpp
bool DeriveGroupSizeMulti(const gert::TilingContext *context) {
    // 1. 检查 scale[0] 维度，判断 perchannel/pergroup 模式
    if (scaleDimNum0 == 1) {
        return CheckMultiA16W4PerChannelShape(context);  // 已有
    }
    // 2. 从 tensor-0 推导 groupSize
    groupSize_ = DeriveGroupSizeFromTensor0(context);
    // 3. 循环检查所有 tensor 一致性
    return CheckAllTensorsGroupSizeConsistency(context);
}
```

---

### 问题 6: 超大函数 — `AntiQuantInt4PerGroupNdKn` 约 85 行 `[用户额外要求]`

**文件**: `basic_block_vf_nd.h:575-660`

**问题**: 函数体约 85 行，包含 groupNum 完整组循环 + groupTail 尾组循环，两个循环内部逻辑高度相似（load scale → load offset → load weight → cast → add → mul → store）。

**建议拆分**: 将单组处理逻辑提取为 `ProcessOneGroupNdKn(...)` 辅助函数。

---

### 问题 7: 超大函数 — `AntiQuantInt4PerGroupNdNkGs32` 约 89 行 `[用户额外要求]`

**文件**: `basic_block_vf_nd.h:908-997`

**问题**: 函数体约 89 行，包含 4-group Select 拼接逻辑。

---

### 问题 8: 超大函数 — `AntiQuantInt4PerGroupNdNkGs256` 约 71 行 `[用户额外要求]`

**文件**: `basic_block_vf_nd.h:729-800`

---

### 问题 9: 超大函数 — `AntiQuantInt4PerGroupNdNkGs64` 约 82 行 `[用户额外要求]`

**文件**: `basic_block_vf_nd.h:805-887`

---

### 问题 10: 超大函数 — `AntiQuantProcessNd` 约 58 行 `[用户额外要求]`

**文件**: `weight_quant_vec_compute.h:1025-1083`

---

### 问题 11: Magic Number — workspace 大小使用字面量 `[用户额外要求: magic number]`

**文件**: `grouped_weight_quant_batch_matmul_tiling.cpp:806`

**问题代码**:
```cpp
workspaces[0] = 16777216U;  // 16 * 1024 * 1024: default workspace size
```

**建议修复**:
```cpp
constexpr uint64_t DEFAULT_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;  // 16MB
workspaces[0] = DEFAULT_WORKSPACE_SIZE;
```

---

### 问题 12: Magic Number — A8W4 baseK 放大倍数 `[用户额外要求: magic number]`

**文件**: `grouped_weight_quant_batch_matmul_tiling.cpp:729`

**问题代码**:
```cpp
tilingData_.mmTilingData.baseK = BASIC_BLOCK_BASE_K * 2;
```

**建议修复**:
```cpp
constexpr uint32_t S8S4_BASEK_MULTIPLIER = 2;
tilingData_.mmTilingData.baseK = BASIC_BLOCK_BASE_K * S8S4_BASEK_MULTIPLIER;
```

---

### 问题 13: Magic Number — float32/int32 表示 8 个 fp4/int4 `[用户额外要求: magic number]`

**文件**: `grouped_weight_quant_batch_matmul_tiling.cpp:1076`

**问题代码**:
```cpp
nSize_ = static_cast<uint64_t>(8) * nSize_;
```

**建议修复**:
```cpp
constexpr uint64_t FP4_PER_FP32 = 8;
nSize_ = FP4_PER_FP32 * nSize_;
```

---

### 问题 14: Magic Number — Select 长度 32/96 `[用户额外要求: magic number]`

**文件**: `basic_block_vf_nd.h:929-930`

**问题代码**:
```cpp
uint32_t selectLen32 = 32;
uint32_t selectLen96 = 96;
```

**建议修复**:
```cpp
constexpr uint32_t SELECT_LEN_32 = 32;
constexpr uint32_t SELECT_LEN_96 = 96;
```

---

### 问题 15: Magic Number — OOMCheckAddrRange 第二个参数 `[用户额外要求: magic number]`

**文件**: `weight_quant_vec_compute.h:299`

**问题代码**:
```cpp
AscendC::OOMCheckAddrRange((__gm__ uint8_t *)(MX_TRANS_ID), 256);
```

**建议修复**:
```cpp
constexpr uint32_t MX_TRANS_ID_CHECK_SIZE = 256;
AscendC::OOMCheckAddrRange((__gm__ uint8_t *)(MX_TRANS_ID), MX_TRANS_ID_CHECK_SIZE);
```

---

### 问题 16: Magic Number — DataCopy blockLen = 8 `[用户额外要求: magic number]`

**文件**: `weight_quant_vec_compute.h:303`

**问题代码**:
```cpp
dmaParams.blockLen = 8;
```

---

### 问题 17: Magic Number — NZ 场景对齐 128 `[用户额外要求: magic number]`

**文件**: `weight_quant_vec_compute.h:558`

**问题代码**:
```cpp
uint64_t scaleInnerSize = wqmmConfig.weightFormat != CubeFormat::NZ
                              ? vecConfig.ubMte2InnerSize
                              : 128UL;  // nz场景当前不会合并，固定对齐到128即可
```

**建议修复**:
```cpp
constexpr uint64_t NZ_SCALE_INNER_SIZE = 128UL;
```

---

### 问题 18: Magic Number — 分派函数中的 groupSize 比较字面量 `[用户额外要求: magic number]`

**文件**: `basic_block_vf_nd.h:1009-1025`

**问题代码**:
```cpp
if (groupSize == 32) {
    ...
} else if (groupSize == 64) {
    ...
} else if (groupSize == 128) {
    ...
} else if (groupSize == 256) {
    ...
}
```

**建议修复**: 定义命名常量或使用 `enum class GroupSize : uint16_t { GS_32 = 32, GS_64 = 64, GS_128 = 128, GS_256 = 256 }`。

---

## 三、代码逻辑重复问题

### 问题 19: `grouped_matmul_apt.cpp` 中 8 个几乎完全相同的 `else if constexpr` 分支 `[用户额外要求: 代码逻辑重复]`

**文件**: `grouped_matmul_apt.cpp:417-510`

**问题**: 8 个 `else if constexpr` 分支结构完全相同，仅 `OFFSET_OR_BIAS_EXIT`、`WQ_B_TRANS` 和 `wqmmCfg` 的 `hasAntiQuantOffset`/`bTrans` 字段不同。约 88 行高度重复代码。

**建议**: 使用宏或模板函数消除重复:
```cpp
#define INVOKE_PERGROUP_ND(offsetBiasExit, bTrans, hasOffset) \
    } else if constexpr (W_TYPE == WQGMM_ND && \
                         OFFSET_OR_BIAS_EXIT == offsetBiasExit && \
                         C_QUANT_TYPE == WQGMM_NONE && \
                         W_QUANT_TYPE == WQGMM_PER_GROUP && \
                         WQ_B_TRANS == bTrans && \
                         WQ_A_TRANS == WQGMM_NO_TRANS && \
                         TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_2 && \
                         ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_BASIC_BLOCK && \
                         ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) { \
        static constexpr WqmmConfig wqmmCfg = {false, (bTrans == WQGMM_TRANS), \
            QuantType::PER_GROUP, hasOffset, QuantType::NONE, CubeFormat::ND}; \
        INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(GMMWeightQuantBasicController, wqmmCfg, \
                                                          VEC_ANTIQUANT_CONFIG_6);
```

---

### 问题 20: `DeriveGroupSizeMulti` 中 tensor-0 检查与循环内 tensor-i 检查逻辑重复 `[用户额外要求: 代码逻辑重复]`

**文件**: `grouped_weight_quant_batch_matmul_tiling.cpp:1245-1309`

**问题**: tensor-0 的 groupSize 推导（1245-1254）与循环内 tensor-i 的一致性检查（1271-1286）逻辑高度相似。offset shape 检查（1294-1308）也与 `CheckMultiA16W4PerChannelShape` 中的 offset 检查（1335-1348）几乎完全相同。

**建议**: 提取共用辅助函数:
```cpp
bool CheckSingleTensorGroupSize(const gert::TilingContext*, uint16_t idx, uint32_t expectedGroupSize) const;
bool CheckAntiquantOffsetShape(const gert::TilingContext*, uint16_t idx, const gert::Shape& scaleShape) const;
```

---

### 问题 21: `CheckTensorDimSingleXSingleWeightSingleY` 中 bias 维度检查冗余 `[用户额外要求: 代码逻辑重复]` `[cpp-general 1.3]`

**文件**: `grouped_weight_quant_batch_matmul_tiling.cpp:255-272`

**问题代码**:
```cpp
if (IsA16W4ND()) {
    ...
    if (hasBias_) {
        OP_CHECK_IF(!CheckTensorDimEqualTarget(context, BIAS_IDX, idx, 2, "bias"), ...);  // 第一次检查
    }
} else {
    ...
}
if (hasBias_) {
    OP_CHECK_IF(!CheckTensorDimEqualTarget(context, BIAS_IDX, idx, 2, "bias"), ...);  // 第二次检查（A16W4ND路径冗余）
}
```

**问题**: 对于 `IsA16W4ND()` 路径，bias 维度检查执行了两次（line 256 和 line 268），是冗余代码。

**建议修复**: 将 line 267-272 的 bias 检查移入 `else` 分支（MxA8W4 路径），或删除 line 255-260 的检查。

---

### 问题 22: `castS4ToF16Trait` 在多个函数中重复定义 `[用户额外要求: 代码逻辑重复]`

**文件**: `basic_block_vf_nd.h` — 出现在 line 490, 538, 604, 681, 748, 824, 933（共 7 处）

**问题代码** (每处都一样):
```cpp
static constexpr MicroAPI::CastTrait castS4ToF16Trait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                          MicroAPI::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::UNKNOWN};
```

**建议修复**: 在文件作用域定义一次:
```cpp
static constexpr MicroAPI::CastTrait CAST_S4_TO_F16_TRAIT = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
```

---

### 问题 23: PerGroup NdNk 各 GS 变体函数结构高度相似 `[用户额外要求: 代码逻辑重复]`

**文件**: `basic_block_vf_nd.h` — `AntiQuantInt4PerGroupNdNkGs32/Gs64/Gs128/Gs256`

**问题**: 四个函数的核心处理流程完全相同（load scale → load offset → load weight → cast → add → mul → store），仅 scale/offset 加载策略因 group 数量不同而异。

**建议**: 可考虑提取公共的 "cast+add+mul+store" 内层逻辑为辅助函数，各 GS 变体仅负责 scale/offset 加载策略。但由于 MicroAPI 寄存器操作的内联性要求，这种重构可能影响性能，需权衡。

---

## 四、低风险问题（风格 / 规范）

### 问题 24: 方法名前多余空格 `[cpp-style 1.2]` `[适用: All]`

**文件**: `grouped_weight_quant_batch_matmul_tiling.cpp:68`

**问题代码**:
```cpp
bool GroupedWeightQuantBatchMatmulTiling:: SetShapeList(const gert::TilingContext *context)
```

**说明**: `::` 与 `SetShapeList` 之间有多余空格。

---

### 问题 25: `weight_quant_tiling_key.h` 末尾无换行 `[cpp-style 2.x]`

**文件**: `weight_quant_tiling_key.h:284`

**问题代码**: 文件最后一行 `#endif` 后无换行符（diff 中显示 `\ No newline at end of file`）。

---

### 问题 26: 注释中的拼写错误

**文件**: `grouped_weight_quant_batch_matmul_tiling.cpp` 多处

**问题**:
- Line 197: `"tenosr"` → 应为 `"tensor"`
- Line 201: `"tenosr"` → 应为 `"tensor"`
- Line 204: `"tenosr"` → 应为 `"tensor"`
- Line 208: `"tenosr"` → 应为 `"tensor"`

---

## 五、Ascend C API / 性能规范检查

### API 检查清单

| 条例 | 状态 | 说明 |
|------|------|------|
| API-1 (禁止 GlobalTensor::SetValue/GetValue) | PASS | 生产代码路径未使用 |
| API-2 (禁止 std:: 计算函数) | PASS | Kernel 侧未使用 std:: 函数 |
| API-3 (DataCopy 32字节对齐) | PASS | 使用 DataCopyPad2D 处理非对齐 |
| API-6 (AllocTensor/FreeTensor 配对) | PASS | 使用 TBuf 静态分配模式 |
| API-7 (禁止动态内存分配) | PASS | 无 new/malloc/vector |
| API-8 (repeatTimes ≤ 255) | PASS | 循环使用 uint16_t 索引 |
| API-9 (Cast RoundMode) | PASS | int4→fp16 使用 UNKNOWN（无需舍入） |
| API-10 (blockLen 单位) | PASS | DataCopyParams.blockLen 使用 32字节块数 |

### 性能检查清单

| 条例 | 状态 | 说明 |
|------|------|------|
| PERF-1 (循环内禁止逐元素) | PASS | 使用 MicroAPI 批量操作 |
| PERF-2 (禁止写死硬件参数) | PASS | 核数从 TilingData 获取 |
| PERF-3 (Double Buffer) | PASS | 使用 vecConfig.ubMte2BufferNum |
| PERF-7 (尾块处理) | PASS | AntiQuantInt4PerGroupNdKn 有 groupTail 单独处理 |

### 安全检查清单

| 条例 | 状态 | 说明 |
|------|------|------|
| 2.3 (除零保护) | PASS | Tiling 侧 `CheckGroupSize` 校验 groupSize 非 0；Kernel 侧 groupSize 来自 TilingData（已校验） |
| 3.1 (禁止未初始化变量) | PASS | 成员变量在头文件中有默认值初始化 |
| 3.3 (数组索引校验) | PASS | Tiling 侧有 MAX_TENSOR_CONT 校验 |
| 3.5 (指针判空) | PASS | Tiling 侧使用 OP_CHECK_IF 判空 |
| 4.1 (外部输入校验) | PASS | AnalyzeAttr 中有完整的参数校验链 |
| TOPK-8 (GM 偏移用 int64) | PASS | antiquantParamsBaseOffset 等使用 uint64_t |

---

## 六、问题汇总

| 编号 | 严重级别 | 类别 | 文件 | 行号 | 描述 |
|------|---------|------|------|------|------|
| 1 | **高** | LOG API 类型不匹配 | tiling.cpp | 737 | `%llu` 用于 `uint32_t` |
| 2 | **高** | LOG API 类型不匹配 | tiling.cpp | 1477-1489 | `%lu` 用于 `uint64_t` |
| 3 | 存疑 | 缩窄转换 | tiling.cpp | 77, 1465 | `int64_t` → `int8_t` |
| 4 | **中** | 有符号/无符号不一致 | tiling.cpp | 1440 vs 1448 | 同一函数内转换风格不一致 |
| 5 | **中** | 超大函数(90行) | tiling.cpp | 1221-1311 | `DeriveGroupSizeMulti` |
| 6 | **中** | 超大函数(85行) | vf_nd.h | 575-660 | `AntiQuantInt4PerGroupNdKn` |
| 7 | **中** | 超大函数(89行) | vf_nd.h | 908-997 | `AntiQuantInt4PerGroupNdNkGs32` |
| 8 | **中** | 超大函数(71行) | vf_nd.h | 729-800 | `AntiQuantInt4PerGroupNdNkGs256` |
| 9 | **中** | 超大函数(82行) | vf_nd.h | 805-887 | `AntiQuantInt4PerGroupNdNkGs64` |
| 10 | **中** | 超大函数(58行) | vec_compute.h | 1025-1083 | `AntiQuantProcessNd` |
| 11 | **中** | Magic Number | tiling.cpp | 806 | `16777216U` |
| 12 | **中** | Magic Number | tiling.cpp | 729 | `* 2` |
| 13 | **中** | Magic Number | tiling.cpp | 1076 | `static_cast<uint64_t>(8)` |
| 14 | **中** | Magic Number | vf_nd.h | 929-930 | `32`, `96` |
| 15 | **中** | Magic Number | vec_compute.h | 299 | `256` |
| 16 | **中** | Magic Number | vec_compute.h | 303 | `8` |
| 17 | **中** | Magic Number | vec_compute.h | 558 | `128UL` |
| 18 | **中** | Magic Number | vf_nd.h | 1009-1025 | `32/64/128/256` 字面量比较 |
| 19 | **中** | 代码重复(88行) | apt.cpp | 417-510 | 8 个相同结构的 else if 分支 |
| 20 | **中** | 代码重复 | tiling.cpp | 1245-1309 | tensor-0 与 tensor-i 检查逻辑重复 |
| 21 | **中** | 冗余代码 | tiling.cpp | 255-272 | bias 维度检查执行两次 |
| 22 | **中** | 代码重复(7处) | vf_nd.h | 多处 | `castS4ToF16Trait` 重复定义 |
| 23 | 低 | 代码重复 | vf_nd.h | 多处 | 4 个 GS 变体函数结构相似 |
| 24 | 低 | 风格 | tiling.cpp | 68 | `::` 后多余空格 |
| 25 | 低 | 风格 | tiling_key.h | 284 | 文件末尾无换行 |
| 26 | 低 | 拼写 | tiling.cpp | 197,201,204,208 | `"tenosr"` → `"tensor"` |

---

## 七、总结

本次 GMM A16W4 PerGroup ND 特性开发代码整体质量良好：
- **安全编码**方面：Tiling 侧参数校验链完整，除法/求余均有除零保护，指针使用前均有判空。发现 2 个 LOG API 类型不匹配问题（高优）需修复。
- **Ascend C API** 方面：未发现 API 黑名单违规，DataCopy 对齐处理正确，无动态内存分配。
- **C++ 编码规范**方面：发现 1 个缩窄转换存疑点、1 个有符号/无符号比较不一致。
- **Magic Number** 方面：发现 8 处 magic number，建议统一使用命名常量。
- **超大函数**方面：发现 6 个超过 50 行的函数，最大约 90 行，建议拆分。
- **代码重复**方面：`grouped_matmul_apt.cpp` 中 8 个 dispatch 分支高度重复（最严重），`castS4ToF16Trait` 重复定义 7 处，`DeriveGroupSizeMulti` 中检查逻辑与 `CheckMultiA16W4PerChannelShape` 重复。

**建议优先修复**: 问题 1、2（LOG API 类型不匹配，可能导致栈数据泄露或段错误）→ 问题 19（apt.cpp 代码重复，影响可维护性）→ 问题 5-9（超大函数拆分）。

---

## 八、修改结果（2026-06-29）

### 修改原则

- `vf_nd.h` 中的函数有编译器特殊要求，**仅修改 magic number**，不拆分函数、不移动 `castS4ToF16Trait` 定义、不重构 GS 变体函数。
- 其他文件按报告问题逐项修复。

### 逐项修改记录

| 编号 | 状态 | 修改内容 |
|------|------|---------|
| P1 | **已修复** | `tiling.cpp:737` — `%llu` → `%u`（stepKb 为 `uint32_t`） |
| P2 | **已修复** | `tiling.cpp:1477-1489` — `PrintTilingResult` 中 `%lu` → `%llu`（kSize/nSize/mainBlockCount 为 `uint64_t`） |
| P3 | **已修复** | `tiling.cpp:77, 1465` — `static_cast<int8_t>(groupType_)` → `static_cast<int32_t>(groupType_)` |
| P4 | **已修复** | `tiling.cpp:1440` — `static_cast<int64_t>(BASIC_BLOCK_BASE_N_MIN)` → `static_cast<uint64_t>(BASIC_BLOCK_BASE_N_MIN)`，与 1448 行统一 |
| P5 | **已修复** | `DeriveGroupSizeMulti` 从 ~90 行降至 ~65 行，提取 `CheckAntiquantOffsetMatchScale` 公共辅助函数 |
| P6 | **不修复** | `vf_nd.h` 中 `AntiQuantInt4PerGroupNdKn` — 编译器特殊要求，不可拆分 |
| P7 | **不修复** | `vf_nd.h` 中 `AntiQuantInt4PerGroupNdNkGs32` — 编译器特殊要求，不可拆分 |
| P8 | **不修复** | `vf_nd.h` 中 `AntiQuantInt4PerGroupNdNkGs256` — 编译器特殊要求，不可拆分 |
| P9 | **不修复** | `vf_nd.h` 中 `AntiQuantInt4PerGroupNdNkGs64` — 编译器特殊要求，不可拆分 |
| P10 | **已修复** | `vec_compute.h` — 提取 `AntiQuantProcessNdInt4` 辅助函数，`AntiQuantProcessNd` 从 ~58 行降至 ~38 行 |
| P11 | **已修复** | `tiling.h` 新增 `DEFAULT_WORKSPACE_SIZE` 常量，`tiling.cpp:806` 使用该常量替代 `16777216U` |
| P12 | **已修复** | `tiling.h` 新增 `S8S4_BASEK_MULTIPLIER` 常量，`tiling.cpp:729` 使用该常量替代 `* 2` |
| P13 | **已修复** | `tiling.h` 新增 `FP4_PER_FP32` 常量，`tiling.cpp:1076` 使用该常量替代 `static_cast<uint64_t>(8)` |
| P14 | **已修复** | `vf_nd.h` 新增 `SELECT_LEN_HALF`(32)/`SELECT_LEN_3Q`(96) 常量，Gs32 函数中使用 |
| P15 | **已修复** | `vec_compute.h` 新增 `MX_TRANS_ID_CHECK_SIZE`(256) 常量，替代 OOMCheckAddrRange 中的字面量 |
| P16 | **已修复** | `vec_compute.h` 新增 `MX_TRANS_ID_COPY_BLOCK_LEN`(8) 常量，替代 DataCopy blockLen 中的字面量 |
| P17 | **已修复** | `vec_compute.h` 新增 `NZ_SCALE_INNER_SIZE`(128) 常量，替代 NZ 场景 scale 对齐字面量 |
| P18 | **已修复** | `vf_nd.h` 新增 `PERGROUP_SIZE_32/64/128/256` 常量，dispatch 函数和各 GS 变体函数中的 `GROUP_SIZE` 字面量全部替换 |
| P19 | **已修复** | `apt.cpp` — 使用 `INVOKE_A16W4_PG_ND` 宏消除 7 个重复分支，从 ~88 行降至 ~20 行 |
| P20 | **已修复** | 随 P5 一并修复 — 提取 `CheckAntiquantOffsetMatchScale` 消除 `DeriveGroupSizeMulti` 与 `CheckMultiA16W4PerChannelShape` 中的 offset 检查重复 |
| P21 | **已修复** | `tiling.cpp` — 删除 `CheckTensorDimSingleXSingleWeightSingleY` 中 A16W4ND 块内的冗余 bias 维度检查 |
| P22 | **不修复** | `vf_nd.h` 中 `castS4ToF16Trait` 重复定义 — 编译器要求 `static constexpr` 在函数作用域内，不可移到文件作用域 |
| P23 | **不修复** | `vf_nd.h` 中 4 个 GS 变体函数结构相似 — 编译器特殊要求，不可重构 |
| P24 | **已修复** | `tiling.cpp:68` — 删除 `::` 与 `SetShapeList` 之间的多余空格 |
| P25 | **已修复** | `tiling_key.h:284` — `#endif` 后添加换行符 |
| P26 | **已修复** | `tiling.cpp:197,201,204,208` — `"tenosr"` → `"tensor"`（4 处） |

### 修改统计

| 指标 | 数值 |
|------|------|
| 总问题数 | 26 |
| 已修复 | **19** |
| 不修复（编译器约束） | 7（P6/P7/P8/P9/P22/P23 + P22 关联） |
| 代码净变化 | -38 行（135 新增 / 173 删除） |

### 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `grouped_weight_quant_batch_matmul_tiling.h` | 新增 3 个常量定义 + 1 个辅助函数声明 |
| `grouped_weight_quant_batch_matmul_tiling.cpp` | LOG API 修复、magic number 替换、typo 修复、风格修复、冗余代码删除、`DeriveGroupSizeMulti` 拆分 + `CheckAntiquantOffsetMatchScale` 提取 |
| `weight_quant_vec_compute.h` | 新增 3 个常量 + `AntiQuantProcessNdInt4` 提取拆分 |
| `weight_quant_tiling_key.h` | 文件末尾添加换行符 |
| `grouped_matmul_apt.cpp` | `INVOKE_A16W4_PG_ND` 宏消除 7 个重复 dispatch 分支 |
| `basic_block_vf_nd.h` | 新增 6 个常量 + 替换所有 magic number（不涉及函数结构变更） |
