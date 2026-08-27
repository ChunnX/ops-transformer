# PR #9848 检视报告 — add allto_all_matmul_v2 operator

## 高 — Bug

### 1. `DoOpTiling` 空指针解引用

`allto_all_matmul_v2_tiling_base.h` 的 `DoOpTiling` 方法中：

```cpp
auto *rt = context_->GetRawTilingData();
auto cap = rt->GetCapacity();      // rt 在此处被解引用
if (!rt || cap < sizeof(...)) {    // 判空逻辑来得太晚
```

如果 `GetRawTilingData()` 返回 nullptr，会在判空之前崩溃。应交换两行顺序，先判空再访问。

### 2. Python API 默认参数与 Tiling 验证不兼容

`all_to_all_quant_matmul.py` 暴露 `x1_quant_mode: int = 0` 和 `x2_quant_mode: int = 0`，值为 0 时 Python 层转为 `None`。C++ 扩展 (`all_to_all_quant_matmul.cpp`) 将 `None` 默认为 `DYN_PERTOKEN_QUANT_MODE` (7) 和 `PERCHANNEL_QUANT_MODE` (2)。但 Tiling 代码 (`CheckQuantAndYDtypeAttr`) 要求两者均为 `MX_QUANT_MODE` (6)。使用默认参数调用 Python API 的用户会收到 Tiling 失败错误。

`comm_mode` 同样不匹配：Python 默认 `"ai_cpu"`（或公共 API 中为 `None`），但 Tiling 要求 `"urma"`。公共 API 的默认值实际无法使用。

### 3. `CheckNullParams` 遗漏 `context` 判空

`aclnn_allto_all_matmul_v2.cpp` 中的 `CheckNullParams` 验证了 x1, x2, scales, output, workspaceSize, executor，但没有验证 `context`。如果调用时 `context` 为 nullptr，会转发给 `aclnnInnerAlltoAllMatmulV2GetWorkspaceSize`，可能导致崩溃或未定义行为。

## 中 — 设计/一致性问题

### 4. `all2all_out` 声明 REQUIRED 但 kernel 中丢弃

`allto_all_matmul_v2_def.cpp` 将 `all2all_out` 声明为 `ParamType(REQUIRED)`，`allto_all_matmul_v2_proto.h` 声明为 `.OUTPUT(all2all_out, ...)`（非可选）。但 kernel (`allto_all_matmul_v2_apt.cpp`) 中第一行可执行代码是 `(void)all2all_out;` — 该输出从未被写入。Python 文档字符串也确认该输出"尚未支持"。

如果该输出确实未使用，应在 OpDef 和 proto 中标记为 OPTIONAL；否则用户将收到未初始化的内存作为输出。

### 5. `TransX2Tensor` 仅交换前两维，但 `IsTransposeLastTwoDims` 允许 2-6D

`aclnn_allto_all_matmul_v2.cpp`:

```cpp
viewDim[0] = x2->GetViewShape().GetDim(1);
viewDim[1] = x2->GetViewShape().GetDim(0);
```

硬编码仅交换 dim 0 和 dim 1。但守卫条件 `IsTransposeLastTwoDims` 接受多达 6D 张量并检查最后两个维度的步长。对于 3D 及以上的转置张量，存储形状的复制是正确的，但视图形状仅交换了 dim 0/1 而非最后两个维度，导致视图与存储不匹配。

### 6. InferShape 在未检查维度数量时访问 scale 形状维度

`allto_all_matmul_v2_infershape.cpp` — `ValidateInferScaleShapes`:

```cpp
if (x1ScaleShape->GetDim(1U) != ...)   // 未检查 GetDimNum() >= 3 即访问
```

Tiling 代码验证了 `x1ScaleDimNum != SCALE_DIM_NUM`，但 InferShape 代码没有。如果提供 1D 或 2D 的 scale 张量，`GetDim(1U)` / `GetDim(2U)` 可能越界或返回垃圾数据。

## 低 — 代码质量

### 7. apt.cpp 中 FP4_E2M1 分支是死代码

`#elif (ORIG_DTYPE_X1 == DT_FLOAT4_E2M1 && ORIG_DTYPE_X2 == DT_FLOAT4_E2M1)` 分支与 `#else` 默认分支生成的代码完全相同。应移除或使其产生不同的实际类型别名（例如使用 `fp4x2_e2m1_t` 而非 `DTYPE_X1`）。

### 8. `comm_tiling_base.h` 潜在除零风险

`CommTilingBase::GetCommTilingData` 计算 `nTile = (n + baseN - 1) / baseN`，然后 `headMSize = ((usedCoreNum + nTile - 1) / nTile) * baseM`。如果 matmul Tiling 产生 `baseN == 0` 或 `nTile == 0`，将导致除零异常。虽不太可能发生，但作为共享工具类增加防御性检查是值得的。

### 9. Python 中上下文名称拼写不一致

`all_to_all_quant_matmul.py`:

```python
context = channel_builder.create_context(group_name + "allto_all_matmul")
```

缺少下划线，应为 `"all_to_all_matmul"` 以与算子名称保持一致。
