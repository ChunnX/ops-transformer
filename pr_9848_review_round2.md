# PR #9848 第二轮检视报告 — allto_all_matmul_v2 operator

**审查范围**：第二轮深度审查，不重复第一轮的 9 个问题。

## 高 — Bug

### 1. graph_convert 目标算子名称错误

**文件**：`torch_extension/cann_ops_transformer/ops/graph_convert/graph_convert_all_to_all_quant_matmul.py:3556`

graph converter 创建的 GE 算子类型是 `"AlltoAllMatmul"`（旧名称），而本 PR 注册的算子是 `"AlltoAllMatmulV2"`（`allto_all_matmul_v2_proto.h` 中 `REG_OP(AlltoAllMatmulV2)`）。IrDef 也使用 `"AlltoAllMatmul"`（line 3561）。图模式下会找不到算子或匹配到错误的旧算子定义。

**触发条件**：任何图模式（torchair/GE）调用 `all_to_all_quant_matmul`。

**建议修复**：将 `op_type="AlltoAllMatmul"` 和 `IrDef("AlltoAllMatmul")` 改为 `"AlltoAllMatmulV2"`。

### 2. graph_convert group_size 编码错误

**文件**：`graph_convert_all_to_all_quant_matmul.py:3633`

```python
group_size=group_sizes[0] if group_sizes else 0,
```

仅取 `group_sizes[0]`（即 groupM=1）作为 `group_size` 属性值。但 C++ 的 `CheckAndGetGroupSize` 将 `[M, N, K]` 编码为 `(M << 32) | (N << 16) | K`。tiling 验证逻辑解码后期望 `gsM=1, gsN=1, gsK=32`，但传入值 `1` 会被解码为 `gsK=1, gsN=0, gsM=0`，导致验证失败。

**触发条件**：图模式调用且 `group_sizes` 非空。

**建议修复**：改为调用编码逻辑：`group_sizes[0] << 32 | group_sizes[1] << 16 | group_sizes[2]`，或直接复用 C++ 侧的 `CheckAndGetGroupSize`。

### 3. graph_convert 包含 V2 proto 中不存在的输入和属性

**文件**：`graph_convert_all_to_all_quant_matmul.py:3526-3548`

graph converter 构建的 inputs 包含 `"scales"`（line 3533），attrs 包含 `"alltoall_out_flag"`（line 3546），IrDef 也声明了 `optional_input("scales", ...)` 和 `attr("alltoall_out_flag", ...)`。但 `AlltoAllMatmulV2` 的 proto/OpDef 中没有这些字段。说明 graph converter 从旧版 `AlltoAllMatmul` 复制而来，未适配 V2。

**触发条件**：图模式调用。

**建议修复**：移除 `scales` 输入和 `alltoall_out_flag` 属性，或重新基于 V2 proto 生成 graph converter。

### 4. Python comm_mode=None 传入 C++ std::string 导致 TypeError

**文件**：`torch_extension/cann_ops_transformer/ops/all_to_all_quant_matmul.py:3059`

Python 公共 API 默认 `comm_mode: Optional[str] = None`，然后直接传递 `comm_mode=comm_mode` 到 `torch.ops`。但 schema 声明为 `str comm_mode="ai_cpu"`（非 `str?`），传入 None 会导致 PyTorch dispatch 层 TypeError。

**触发条件**：用户不显式传 `comm_mode` 参数（使用默认值 None）。

**建议修复**：将 Python 默认改为 `comm_mode="urma"`（与 tiling 要求一致），或在传参前做 `comm_mode = comm_mode or "urma"` 转换。

### 5. Python y_dtype=-1 默认产生 float32 输出，tiling 必拒

**文件**：`all_to_all_quant_matmul.py:3015-3017` + `all_to_all_quant_matmul.cpp:3209-3212`

Python 默认 `y_dtype=-1`，经转换后传 None 到 C++。C++ `GetOutputScalarType` 在 `yDtype` 无值时返回 `at::kFloat`，分配 float32 输出 tensor。但 tiling 验证要求 `yDtype == BF16 || yDtype == FP16`，float32 会被拒绝。即默认路径必然失败。

**触发条件**：用户不显式传 `y_dtype` 参数。

**建议修复**：将默认输出类型改为 BF16，或在 Python 层强制要求 `y_dtype` 参数。

## 中 — 设计/内存安全问题

### 6. TransX2Tensor 创建的 aclTensor 未销毁（内存泄漏）

**文件**：`mc2/allto_all_matmul_v2/op_api/aclnn_allto_all_matmul_v2.cpp:151-155`

`TransX2Tensor` 调用 `aclCreateTensor()` 分配新 tensor 对象，传给 inner API 后从未调用 `aclDestroyTensor()` 释放。每次 x2 非连续时（DAV_3510 arch）都会泄漏。

**触发条件**：DAV_3510 架构下 x2 为非连续 tensor（转置场景）。

**建议修复**：在 inner API 返回后添加 `aclDestroyTensor(transX2)`（需确保 inner API 不会持有该 tensor 引用）。

### 7. InferShape 未设置 all2all_out 输出 shape

**文件**：`mc2/allto_all_matmul_v2/op_host/allto_all_matmul_v2_infershape.cpp:183-190`

`AlltoAllMatmulV2InferShape` 只设置了 output[0]（y）的 shape，从未设置 output[1]（all2all_out）的 shape。框架可能使用未初始化的 shape 值，导致下游算子拿到错误 shape。

**触发条件**：每次 InferShape 调用。

**建议修复**：在设置 y shape 后，也设置 all2all_out 的 shape（应为 `[M/world_size, K_total]`）。

### 8. at::from_blob 未持有设备内存所有权（潜在泄漏）

**文件**：`torch_extension/cann_ops_transformer/ops/csrc/comm_channel_builder_manager.cpp:3383`

```cpp
at::Tensor context = at::from_blob(devCtx, {sizeof(CommContext) / sizeof(int32_t)}, at::kInt);
```

`at::from_blob` 默认不获取内存所有权。`devCtx` 由 `CreateDeviceContext` 分配的设备指针在 tensor 析构后不会释放。

**触发条件**：每次创建通信 context。

**建议修复**：使用带 deleter 的 `at::from_blob` 重载，或记录生命周期管理策略确保设备内存最终被释放。

### 9. InferShape 未验证 world_size 范围和整除性

**文件**：`allto_all_matmul_v2_infershape.cpp:170-176`

InferShape 直接用 `m / worldSize` 计算输出 M 维，但未检查：1) worldSize 是否为 2 的幂且在 [2,16] 范围内；2) m 是否能被 worldSize 整除。如果 worldSize=3 且 m=100，输出 shape 会是 `[33, N]`（截断），而 tiling 阶段才会报错。导致 InferShape 返回错误 shape 给框架。

**触发条件**：worldSize 非 2 的幂或 m 不能被 worldSize 整除。

**建议修复**：在 InferShape 中添加与 tiling 一致的 worldSize 范围检查和整除性检查。

## 低 — 代码质量

### 10. Python meta 缺少 K 对齐和 scale shape 校验

**文件**：`all_to_all_quant_matmul.py:2899-2922`（meta 函数）

meta 函数仅检查 `x1.dim()==2` 和 `x2.dim()==2`，但未检查 Ka 是否能被 64 整除、K_total = Ka × world_size、scale tensor shape 是否匹配。这些检查推迟到 tiling 阶段，导致 meta 阶段可能返回不合规的 output shape。

**触发条件**：传入 K 不对齐 64 或 scale shape 错误的输入。

**建议修复**：在 meta 中添加关键约束检查。

### 11. C++ extension is_w4 死代码路径

**文件**：`all_to_all_quant_matmul.cpp:3243, 3249`

`bool is_w4 = x2.dtype() == at::kInt` 及相关 INT4 解包逻辑在 V2 算子中不可达，因为 OpDef 仅支持 FP8/FP4 dtype，不支持 INT4。这是从旧版代码残留的死代码。

**建议修复**：移除 is_w4 相关逻辑。

### 12. NnopbaseSetHcclServerType weak 符号未判空调用

**文件**：`aclnn_allto_all_matmul_v2.cpp:182`

`NnopbaseSetHcclServerType` 声明为 `__attribute__((weak))`，若运行时未提供该符号则函数指针为 null。代码直接调用而不检查 null，在符号缺失的环境中会崩溃。

**触发条件**：运行环境未提供 NnopbaseSetHcclServerType 实现。

**建议修复**：添加 `if (NnopbaseSetHcclServerType != nullptr)` 守卫。

### 13. C++ x1/x2_quant_mode fallback 默认值与 tiling 要求不一致

**文件**：`all_to_all_quant_matmul.cpp:3263-3264`

当 Python 传 None（x1_quant_mode=0 时转换），C++ 默认 `x1QuantModeVal = DYN_PERTOKEN_QUANT_MODE (7)` 和 `x2QuantModeVal = PERCHANNEL_QUANT_MODE (2)`。但 tiling 要求两者均为 `MX_QUANT_MODE (6)`。即使 Python 层修复了默认值，C++ 层的 fallback 仍然错误。

**建议修复**：C++ fallback 也应改为 `MX_QUANT_MODE (6)`。
