# PR #10056 第二轮检视报告 — AttentionToFFNV2、FFNToAttentionV2 算子

**审查范围**：第二轮深度审查，不重复第一轮的 5 个问题。

## 高 — Bug

### 1. MX 量化路径缺少 `receiveDataCastFloatBuf_` 和 `smoothScalesBuf_` 的初始化

**文件**：`mc2/attention_to_ffn/op_kernel/attention_to_ffn.h` — `QuantInit` 函数

在 `QuantInit` 的 MX/MX_CLIP 分支中，只初始化了 `xInQueue_` 和 `xOutQueue_`，但未初始化 `receiveDataCastFloatBuf_` 和 `smoothScalesBuf_`。然而在 `Init` 中，无论量化模式如何，都会执行：

```cpp
castTempBuf_ = receiveDataCastFloatBuf_;
sumOutBuf_ = smoothScalesBuf_;
```

而在 `QuantProcess` 的 arch3510 MX 路径中，直接使用：

```cpp
LocalTensor<float> receiveDataCastFloat = receiveDataCastFloatBuf_.Get<float>();
__ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)receiveDataCastFloat.GetPhyAddr();
__ubuf__ uint16_t *halfScaleLocalAddr =
    (__ubuf__ uint16_t *)receiveDataCastFloat[Ceil(mxScaleNum, 32U) * 32U].GetPhyAddr();
```

**触发条件**：MX 量化模式（quantMode 3/4/5/6/7）在 arch3510 上运行时。

**影响**：访问未初始化的 UB buffer，可能导致随机数据或 NPU 异常。

**建议修复**：在 `QuantInit` 的 `IsMxOrMxClipQuant` 分支中添加 `receiveDataCastFloatBuf_` 和 `smoothScalesBuf_` 的 `InitBuffer` 调用，确保 buffer 大小足够容纳 `maxExpAddr` 和 `halfScaleLocalAddr`（至少 `Ceil(mxScaleNum, 32) * 32 * sizeof(float) + mxScaleNum * sizeof(uint16_t)` 字节）。

### 2. MX 量化路径缺少流水同步屏障

**文件**：`mc2/attention_to_ffn/op_kernel/attention_to_ffn.h` — `QuantProcess` 函数 arch3510 分支

在 MX_QUANT 路径中：

```cpp
Quant::ComputeMaxExp(srcAddr, maxExpAddr, axisH_);
Quant::ComputeScale<XOutType>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum);
Quant::ComputeFp8Data<...>(srcAddr, halfScaleLocalAddr, outLocalAddr, axisH_);
```

`ComputeMaxExp` 写入 `maxExpAddr`，`ComputeScale` 读取 `maxExpAddr` 并写入 `halfScaleLocalAddr`，`ComputeFp8Data` 读取 `halfScaleLocalAddr`。这三步之间存在 RAW 数据依赖，但中间没有 `PipeBarrier<PIPE_V>()` 同步。

MX_CLIP 路径存在同样的问题。

**触发条件**：MX/MX_CLIP 量化模式在 arch3510 上运行时。

**影响**：Vector 流水线 RAW 冒险，可能导致量化结果错误（读到未完成写入的数据）。

**建议修复**：在 `ComputeMaxExp` 和 `ComputeScale` 之间、`ComputeScale` 和 `ComputeFp8Data/ComputeFp4Data` 之间各添加 `PipeBarrier<PIPE_V>()`。

### 3. V1/V2 tiling data 结构体字段顺序不一致，reinterpret_cast 不安全

**文件**：
- `mc2/attention_to_ffn_v2/op_kernel/arch35/attention_to_ffn_v2.cpp:2536`
- `mc2/ffn_to_attention_v2/op_kernel/arch35/ffn_to_attention_v2.cpp:5635`

V2 kernel 使用 `reinterpret_cast` 将 V2 tiling data 强转为 V1 类型：

```cpp
const auto *v1TilingData = reinterpret_cast<const AttentionToFFNTilingData *>(&tilingData);
```

仅有 `static_assert(sizeof(...) == sizeof(...))` 校验总大小，但未校验字段顺序。

对比 V1 `PrintTilingDataInfo` 打印的字段顺序：`X, BS, H, K, L, expertNum, moeExpertNum, HS, ...`（无 `sharedExpertNum`）。
V2 `AttentionToFfnV2Info` 定义：`X, BS, H, K, L, expertNum, moeExpertNum, sharedExpertNum, HS, ...`（有 `sharedExpertNum` 在 `moeExpertNum` 和 `HS` 之间）。

如果 V1 结构体中 `sharedExpertNum` 不在 `moeExpertNum` 之后，则 `reinterpret_cast` 后所有字段偏移错位。

**触发条件**：V2 legacy 路径（quantMode 0 或 2），kernel 读取 V1 tiling data 时。

**影响**：kernel 读取到错误的 tiling 参数值（如错误的 HS、worldSize 等），导致地址计算越界或数据损坏。

**建议修复**：添加 `static_assert(offsetof(AttentionToFfnV2Info, X) == offsetof(AttentionToFFNInfo, X))` 等逐字段 offset 校验，或确保 V1/V2 结构体字段顺序完全一致。

## 中 — 设计/正确性问题

### 4. FP4_E2M1 的 H 维度需为偶数的约束未在 tiling 中校验

**文件**：`mc2/attention_to_ffn_v2/op_host/op_tiling/attention_to_ffn_v2_tiling.cpp` — `TilingNewQuantMode` 函数

README 明确约束"FP4输出（quantMode=5）时H维度需为偶数"，但 tiling 校验代码中没有针对 quantMode=5 检查 `h % 2 == 0`。

**触发条件**：用户传入 quantMode=5 且 H 为奇数时。

**影响**：FP4 打包时 `Ceil(axisH_, FP4_ELEMS_PER_BYTE)` 会多打包半个元素，可能导致数据对齐问题和量化结果错误。

**建议修复**：在 `TilingNewQuantMode` 中添加：
```cpp
if (quantMode == USER_QUANT_MODE_MX_E2M1) {
    OP_TILING_CHECK(h % 2 != 0, OP_LOGE(nodeName, "H must be even for FP4_E2M1 quantization"), return ge::GRAPH_FAILED);
}
```

### 5. `FFNToAttentionV2` OpDef 中 `MC2().HcclGroup` 被注释掉

**文件**：`mc2/ffn_to_attention_v2/op_host/ffn_to_attention_v2_def.cpp:5402`

```cpp
//this->MC2().HcclGroup({"group"});
```

V1 的 `FFNToAttention` OpDef 应该有此注册（用于声明 `group` 为 MC2 通信域属性）。V2 注释掉后，MC2 框架可能无法正确识别 `group` 属性的通信域语义，导致通信初始化失败。

**触发条件**：V2 算子执行时，MC2 框架需要通过 HcclGroup 属性建立通信域。

**影响**：通信域可能无法正确初始化，或算子无法正确加入 MC2 调度图。

**建议修复**：取消注释，或确认 V2 是否通过其他机制（如 context tensor）处理通信域注册。

### 6. kernel 中 `epHcclBuffer_[rankId]` 无边界检查

**文件**：`mc2/attention_to_ffn/op_kernel/attention_to_ffn.h` — `GetWindowAddrByRankId` 函数

```cpp
if constexpr (contextType == AttentionToFFNContextType::MC2_CONTEXT) {
    return (GM_ADDR)mc2Context_->epHcclBuffer_[rankId];
}
```

`rankId` 来自 `mc2Context_->epRankId`，`epHcclBuffer_` 大小为 `HCCL_MAX_RANK_SIZE`。如果 `epRankId >= HCCL_MAX_RANK_SIZE`，将导致数组越界。`FFNToAttention` 的 `GetWindowAddr` 也有同样问题。

**触发条件**：context tensor 数据损坏，或 `epRankId`/`curAttenWorkRank` 超出预期范围。

**影响**：GM 地址越界读取，可能导致 NPU 访存异常或数据损坏。

**建议修复**：在 kernel 中添加 `rankId < HCCL_MAX_RANK_SIZE` 的断言或 clamp 保护。

### 7. `DataCopyPadExtParams<XType>` 改为非模板 `DataCopyPadParams` 可能影响 pad 行为

**文件**：`mc2/attention_to_ffn/op_kernel/attention_to_ffn.h` — `SendTokenData` 函数

```cpp
-    DataCopyPadExtParams<XType> copyPadExtParams{false, 0U, 0U, 0U};
+    DataCopyPadParams copyPadExtParams{false, 0U, 0U, 0U};
```

`DataCopyPadExtParams` 是模板参数化的，包含数据类型信息，可能影响 DataCopyPad 的 padding 计算。改为非模板的 `DataCopyPadParams` 后，API 可能无法正确推断元素大小，导致 pad 计算错误。同时 `xCopyParams` 从 `DataCopyExtParams`（`uint32_t` blockLen）改为 `DataCopyParams`（`uint16_t` blockLen），类型收窄是潜在风险。

**触发条件**：量化路径下执行 `DataCopyPad` 搬运输入数据时。

**影响**：可能导致 DataCopyPad 的 padding 行为不正确，数据对齐错误。

**建议修复**：确认 `DataCopyPadParams` 是否与 `DataCopyPadExtParams` 在所有支持的数据类型下行为一致；如不一致，保持使用模板版本。

### 8. V2 tiling 测试缺少 `ccl_buffer_size` 属性或值不正确

**文件**：
- `mc2/attention_to_ffn_v2/tests/ut/op_host/arch35/test_attention_to_ffn_v2_tiling.cpp:3065-3081`
- `mc2/ffn_to_attention_v2/tests/ut/op_host/arch35/test_ffn_to_attention_v2_tiling.cpp:6091-6104`

- AttentionToFfnV2 tiling 测试的 `BuildAttrs` 函数没有提供 `ccl_buffer_size` 属性，但 OpDef 声明为 REQUIRED，tiling 代码会在 legacy 路径中校验该属性非空。测试期望 GRAPH_SUCCESS，但实际应失败。
- FFNToAttentionV2 tiling 测试的 `BuildAttrs` 设置 `ccl_buffer_size = 0`，但 tiling 校验 `*cclBufferSizePtr <= 0` 会拒绝。测试同样期望 GRAPH_SUCCESS。

**触发条件**：运行这些 tiling 单测时。

**影响**：单测预期与实际行为不一致，测试无法通过或无法覆盖真实场景。

**建议修复**：
- AttentionToFfnV2 测试：在 `BuildAttrs` 中添加 `{"ccl_buffer_size", AnyValue::CreateFrom<int64_t>(1LL << 30)}`
- FFNToAttentionV2 测试：将 `ccl_buffer_size` 值从 0 改为有效正值（如 `1LL << 30`）

### 9. V2 新量化路径的 `ccl_buffer_size` 校验与 legacy 路径不一致

**文件**：`mc2/attention_to_ffn_v2/op_host/op_tiling/attention_to_ffn_v2_tiling.cpp` — `TilingNewQuantMode` vs `AttentionToFFNTilingFuncBase`

- Legacy 路径校验 `ccl_buffer_size >= neededSize * MB_SIZE`（MB 对齐后的值）
- 新量化路径校验 `ccl_buffer_size >= requiredWindowBytes`（精确字节数，未对齐到 2MB）

README 要求 `ccl_buffer_size` 按 2MB 对齐。新路径允许传入未对齐的值，可能导致 buffer 分配不足。

**触发条件**：MX 量化模式下传入未对齐到 2MB 的 `ccl_buffer_size` 时。

**影响**：buffer 大小可能不满足 2MB 对齐要求，导致通信 buffer 分配异常。

**建议修复**：新路径也应使用 `CeilAlign(requiredWindowBytes, MB_ALIGN)` 校验，与 legacy 路径和 README 约束保持一致。

## 低 — 代码质量

### 10. Python 端 `_get_hccl_comm_name` 和 `_get_npu_backend` 在两个文件中重复定义

**文件**：
- `torch_extension/cann_ops_transformer/ops/attention_to_ffn_v2.py:7489-7500`
- `torch_extension/cann_ops_transformer/ops/ffn_to_attention_v2.py:8027-8038`

两个 V2 Python 文件各自定义了完全相同的 `_get_hccl_comm_name` 和 `_get_npu_backend` 函数。代码重复增加维护成本。

**建议修复**：将公共函数提取到 `comm_context.py` 或新建的 `utils.py` 中，两个文件 import 使用。

### 11. `get_window_addr` 使用硬编码偏移读取 context struct

**文件**：
- `torch_extension/cann_ops_transformer/ops/attention_to_ffn_v2.py:7585-7588`
- `torch_extension/cann_ops_transformer/ops/ffn_to_attention_v2.py:8116-8122`

```python
offset_epHcclBuffer = 2 * 4 + 8
offset_target = offset_epHcclBuffer + self.rank_id * 8
addr_bytes = ctypes.c_uint64.from_address(context_ptr + offset_target)
```

硬编码偏移 `2 * 4 + 8 = 16` 对应 `AttentionFFNContext` 的 `epHcclBuffer_` 起始位置。如果 C++ 结构体添加字段或调整对齐，Python 端不会自动同步，导致读取错误地址。

**建议修复**：在 C++ 侧提供一个 `get_window_addr(rank_id)` 的导出函数，Python 端调用而非手动计算偏移。

### 12. MX_CLIP 量化路径在 kernel 中缺少 E2M1 分支的编译期保护

**文件**：`mc2/attention_to_ffn_v2/op_kernel/arch35/attention_to_ffn_v2.cpp:2565-2579`

MX_CLIP 分支只处理 E5M2 和 E4M3，没有 `else` 分支或 `static_assert` 防止 E2M1：

```cpp
} else if constexpr (QuantMode == ATTN_FFN_TILINGKEY_MX_CLIP) {
    if constexpr (OutDtype == ATTN_FFN_TILINGKEY_OUT_E5M2) { ... }
    else if constexpr (OutDtype == ATTN_FFN_TILINGKEY_OUT_E4M3) { ... }
    // 无 else 分支：如果 OutDtype == E2M1，kernel 静默不做任何操作
}
```

虽然 tiling 层阻止了 MX_CLIP+E2M1 的组合，但 kernel 缺少编译期保护。

**建议修复**：添加 `else static_assert(false, "MX_CLIP does not support E2M1")` 或等价的编译期错误。
