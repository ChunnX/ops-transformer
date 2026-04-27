# 迁移排障

先读当前失败日志，再按症状定位。连续两轮仍无法定位时，才搜索一个相似已迁移算子作对照。

## 编译失败

### `undefined reference to TilingRegistry`

registry 相关 `.cpp` 未编入 `cust_optiling`，或注册命名空间不一致。检查 `op_host/CMakeLists.txt` 是否收录 `tiling_base`/`common` 实现文件，注册宏和入口调用是否使用同一 namespace。

### `OP_LOGE` / `OPS_ERR_IF` / `OP_TILING_CHECK` 未定义

日志或错误宏来自开源仓公共头。优先本地化最小公共头；若只是调试日志，可删除日志调用或换成当前工程可用日志，不改变 tiling 分支。

### `nlohmann/json.hpp not found`

旧 tiling 接口残留。用 `gert::TilingContext` 和 `PlatformAscendC` 获取平台信息，不引入 JSON 依赖。

### `.SetTiling()` 与 `IMPL_OP_OPTILING` 冲突

同一 op 同时注册两套 tiling。简单算子保留 `.SetTiling()`；复杂 registry 算子用 `IMPL_OP_OPTILING`，并从 `_def.cpp` 移除 `.SetTiling()`。

### 找不到 `aclnn_*.h` 或 pybind 链接失败

OpDef 未被 `opbuild` 正确处理，或 `OP_ADD(OpName)`、类名、文件收录不一致。检查 `op_host` 编译源列表和 `build_out/autogen/`；不要通过手写或复制 `aclnn_*` 源码解决。

## 运行失败

### `Check out != nullptr`

`EXEC_NPU_CMD` 参数顺序错误。输出 tensor 放在参数列表最后，并与 OpDef 输出顺序一致。

### 所有输出为 0

常见原因是 tiling key 未命中、TilingData 字段错位或 block dim 为 0。检查 `SetTilingKey()`、kernel `TILING_KEY_IS`、TilingData 字段顺序和 `SetBlockDim()`。

### 随机大误差

优先检查 GM_ADDR 参数顺序、dtype 分支、workspace 大小、block 分块长度、layout 转换和原算子精度提升逻辑。

### workspace 崩溃

确认 `context->GetWorkspaceSizes(1)` 判空后设置，即使 workspace 为 0 也显式写 `workspace[0] = 0`。

### `GetCompileInfo()` 为空

缺少 `TilingParse<CompileInfoType>` 或 prepare 阶段没有填充 compile info。保留原结构，并确认注册与读取类型一致。

### `compileInfoPtr is nullptr`

可能命中了 CANN 内置同名算子的 tiling。先确认本次产物存在：

```text
op_api/lib/libcust_opapi.so
op_impl/ai_core/tbe/op_tiling/liboptiling.so
```

再确认 precision 环境中的 `CUSTOM_OPTILING_LIB_PATH` 指向本次产物。不要在单个 `pybind.cpp` 中重复手写全局 `dlopen`。

### 标量参数恒为 0

pybind 标量类型与 `build_out/autogen/aclnn_*.h` 的 `GetWorkspaceSize` 签名不一致。查看 autogen 头，按签名精确修改 pybind 参数类型。

### `std::bad_alloc` / `MemoryError`

常见于 `ListFloat` 属性桥接。按 `references/validation.md` 将固定长度 ListFloat 拆成多个 Float 属性，并在 `ModelNew` 中保持用户语义。

## 精度失败

### golden 与算子语义不一致

检查 `Model.forward()` 是否复现原算子的 dtype cast、舍入、clamp、layout、广播、边界处理和特殊值行为。

### 参数数量不匹配

`get_inputs()` 只返回用户可见参数。推导参数在 `ModelNew.forward()` 中计算。

### 调用了 `custom_pybind_api`

module 中应调用 `kernel_gen_ops.<kernel_name>(...)`。`custom_pybind_api` 只作为 pybind 源文件占位符。

### diff 返回类型错误

`custom_check_precision()` 返回 `(bool, torch.Tensor, torch.Tensor)`。

## 查看相似算子的时机

满足以下条件之一再看：

- 当前错误已经按日志修了两轮仍无进展
- 报错指向框架接口差异，当前 skill 未覆盖
- 需要确认某类复杂 registry 的本地 CMake 写法

查看时只打开一个最相似算子的相关文件，并记录借鉴点来自哪个错误症状。
