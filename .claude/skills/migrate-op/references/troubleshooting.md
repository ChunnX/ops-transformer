# 迁移排障指南

本文档汇总算子迁移过程中遇到的常见问题、错误特征及其解决方案，按编译、运行、精度验证三个阶段组织。迁移规则与适配方法见各专项指南，本文仅覆盖问题诊断与修复。

---

## 1. 编译阶段

### 1.1 `undefined reference to TilingRegistry`

**原因**：registry 命名空间不一致或 CMake 未收录 tiling_base 源码。

**解决**：检查 `tiling_base/*.cpp` 是否被编进 `cust_optiling`，确认注册宏和入口调用使用同一个命名空间。

### 1.2 `OP_LOGE` / `OPS_ERR_IF` / `OP_TILING_CHECK` 未定义

**原因**：日志/错误宏来自开源仓公共头，单算子工程中不可用。

**解决**：复制本地 `common/log.h`，或将宏替换为本地已有宏。

### 1.3 `nlohmann/json.hpp not found`

**原因**：旧接口遗留，开源仓用 JSON 解析 compileInfo。

**解决**：不要引入 JSON 依赖，改用 `PlatformAscendC` 或 `TilingParse` 填充的 `CompileInfo`。

### 1.4 `.SetTiling()` 和 `IMPL_OP_OPTILING` 重复注册

**原因**：同一个算子同时用两种方式注册 tiling，导致运行时注册冲突。

**解决**：简单算子只用 `.SetTiling()`；复杂算子 `_def.cpp` 不写 `.SetTiling()`，在 `_tiling.cpp` 里用 `IMPL_OP_OPTILING`。

---

## 2. 运行阶段

### 2.1 所有输出为 0

**原因**：tiling key 没匹配到有效分支，或 TilingData 字段错位。

**解决**：
- 检查 host 侧 `context->SetTilingKey(key)` 是否与 kernel 中 `TILING_KEY_IS(...)` / `GetTilingKey()` 分支一致
- 确认 kernel 读取的 TilingData 字段顺序与 `_tiling.h` 定义一致

### 2.2 随机大误差

**原因**：block dim、分块长度、workspace 大小或 dtype 分支与 host 不一致。

**解决**：对比 host tiling 计算的 block dim、分块长度与 kernel 预期，检查 dtype 分支是否遗漏。

### 2.3 运行崩溃

**原因**：GM_ADDR 参数顺序错误、workspace 未分配、tiling data size 未写入。

**解决**：
- 检查 kernel 入口 GM_ADDR 参数顺序与 OpDef 输入/输出声明顺序匹配
- 确认 host 设置了 `SetBlockDim()` 且 workspace 指针判空
- 检查 tiling data size 是否通过 `SetDataSize()` 写入

### 2.4 workspace 崩溃

**原因**：`context->GetWorkspaceSizes(1)` 返回值未判空。

**解决**：即使 workspace 为 0，也应判空后设置 `workspace[0] = 0`。

### 2.5 `GetCompileInfo()` 为空

**原因**：缺少 `TilingParse<CompileInfoType>(...)` 注册，或 `TilingPrepare` 没有写入字段。

**解决**：保留/补齐 CompileInfo 结构和 parse 注册。

### 2.6 `compileInfoPtr is nullptr`（CANN 同名算子冲突）

**特征**：TilingParse 和打包的 `liboptiling.so` 都正确，但 precision 仍报同名内置算子的 `compileInfoPtr is nullptr`。

**原因**：运行期未优先加载自定义 tiling 库，CANN 使用了内置同名算子。

**解决**：在 `validation/pybind.cpp` 中基于 `ASCEND_CUSTOM_OPP_PATH` 显式 `dlopen(.../op_tiling/liboptiling.so, RTLD_NOW | RTLD_GLOBAL)` 后再调用 `EXEC_NPU_CMD`。显式加载前先用 `ctypes.CDLL` 或 `ldd` 检查该 so 没有未解析符号。

常见未解析符号来自迁移时保留了开源仓调试函数声明但没有实现，例如 `Ops::NN::DebugTilingContext(...)`。这类调试打印不属于 tiling 语义，单算子工程中应删除对应 include 和日志调用，或复制完整实现。

### 2.7 动态输入数量错误

**特征**：`ParamType(DYNAMIC)` 的输入遍历数量不正确。

**原因**：用固定 `GetInputTensor(i)` 循环到所有张量。

**解决**：使用 `GetDynamicInputTensor(dynamicInputIndex, j)`，遇到 `nullptr` 停止。

### 2.8 tiling key 不匹配

**特征**：kernel 中按 `TILING_KEY_IS(...)` 或 `GetTilingKey()` 分支，但 host 未设置对应 key。

**解决**：host 必须调用 `context->SetTilingKey(key)`，且 key 值与 kernel 编译分支一致。

### 2.9 `Check out != nullptr`

**原因**：`EXEC_NPU_CMD` 参数顺序错误，输出张量不在最后。

**解决**：输出张量必须放在 `EXEC_NPU_CMD` 参数列表的最后位置。

### 2.10 标量参数恒为 0（类型不匹配）

**特征**：tiling 日志显示属性值恒为 0（如 `scale: 0.000000`），但测试传入非零值。

**原因**：pybind.cpp 中标量参数类型（如 `float`）与 autogen 生成的 `aclnnXxxGetWorkspaceSize` 函数签名中对应参数类型（如 `double`）不一致，导致 ABI 不匹配。

**解决**：
1. 打开编译目录下 `build_out/autogen/aclnn_xxx.h`，查看 `GetWorkspaceSize` 签名中每个参数的类型
2. pybind.cpp 中标量参数的类型必须与 autogen 签名**完全一致**

查找 autogen 头文件：
```bash
find runs -path "*<OpName>*" -name "aclnn_*.h" -print
```

### 2.11 `std::bad_alloc` / `MemoryError`（ListFloat 未拆分）

**原因**：`pytorch_npu_helper.hpp` 的 `EXEC_NPU_CMD` 宏没有 `std::vector<float>` → `aclFloatArray*` 的转换重载。

**解决**：将 `ListFloat` 属性拆分为独立的 `Float` 属性。详见 `validation_guide.md` 第 5 节。

---

## 3. 精度验证阶段

### 3.1 Model 未复制精度提升逻辑

**特征**：精度测试误差过大。

**原因**：`Model`（golden）未复制算子的精度提升逻辑（如 int16→float32→计算→转回）。

**解决**：`Model.forward()` 中必须复制与算子一致的精度提升逻辑。

### 3.2 推导参数泄露到 `get_inputs()`

**特征**：运行时报参数数量不匹配。

**原因**：`get_inputs()` 返回了推导参数。

**解决**：`get_inputs()` 只返回用户直接指定的参数；推导参数由 `ModelNew.forward()` 计算。

### 3.3 使用 `custom_pybind_api` 而非 `<opname>`

**特征**：调用自定义算子失败。

**原因**：module.py 中使用了模板占位符名 `custom_pybind_api` 而非渲染后的实际函数名。

**解决**：使用 `kernel_gen_ops.<opname>(...)` 调用。

### 3.4 `all_abs_diff` / `all_rel_diff` 类型错误

**原因**：`custom_check_precision` 返回了 Python `float` 而非 `torch.Tensor`。

**解决**：确保 `all_abs_diff` 和 `all_rel_diff` 是 `torch.Tensor`。

### 3.5 CSV bool 字段解析错误

**特征**：bool 参数未正确解析。

**原因**：CSV bool 字段可能被解析为 `int`（1/0）。

**解决**：
```python
ac = param['align_corners']
align_corners = bool(ac) if not isinstance(ac, str) else (ac.lower() == 'true')
```
