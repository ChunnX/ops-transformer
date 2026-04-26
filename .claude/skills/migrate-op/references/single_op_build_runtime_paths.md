# 单算子编译与运行路径指南

这份指南用于迁移算子时快速判断：源码应该放哪里、默认 msopgen CMake 会编什么、运行时会加载哪些 so、什么时候必须补自定义 CMake。遇到路径或 CMake 判断分歧时，以当前仓库代码和本文为准。

## 1. 先建立心智模型

一个 NPUKernelBench 单算子不是只有 kernel。完整链路至少包含四类东西：

```text
op_host/OpDef          -> 生成 proto、aclnn API 和注册信息
op_host/tiling         -> 生成自定义 liboptiling.so，注册 host tiling
op_kernel/AscendC      -> 生成 kernel .o / binary
validation/pybind.cpp  -> 生成 Python 扩展 kernel_gen_ops.*.so
```

运行时也要分清两条查找：

```text
aclnn{OpName} API 查找      -> libcust_opapi.so
OpType={OpName} tiling 查找 -> liboptiling.so 注册表
```

`libcust_opapi.so` 命中自定义库，不代表 tiling 一定命中自定义库。当前框架已经在 `EXEC_NPU_CMD` 前强制加载自定义 `liboptiling.so`，但前提是自定义 OPP 产物路径完整。

## 2. 任务目录和运行目录

任务目录：

```text
tasks/level{N}/{Category}/{OpName}/
  api_desc.md
  question/
    op_host/
    op_kernel/
  answer/0/
    op_host/
    op_kernel/
  validation/
    module.py
    prepare_inputs.py
    pybind.cpp
    test_cases.csv
```

框架从目录名解析：

```text
OpName      = 目录名去掉 "_" 后缀前的部分，例如 ForeachPowScalarList
kernel_name = OpName 转 snake_case，例如 foreach_pow_scalar_list
```

编译运行目录默认在：

```text
runs/lvl{N}/{Category}/{OpName}/fixed_case_{case_id}/sample{sample_id}/
  src/          # 本次实际编译源码
  build/        # kernel_gen_ops.*.so 的构建目录
  log/          # compile 日志
  case_{i}/log/ # precision/perf 日志
```

不要直接根据 `tasks/.../answer/0` 判断编译结果；实际编译看 `runs/.../src`。

## 3. 源码是怎么进入 src 的

默认配置：

```yaml
compile:
  mode: "msopgen"
  common_dir: "libs/common_include"
kernel_only_mode: True
```

源码准备顺序：

```text
1. 复制 libs/common_include/msopgen/ 到 runs/.../src/
2. 复制 tasks/.../validation/ 到 runs/.../src/validation/
3. kernel_only_mode=True 时，把 answer/{sample_id}/ 整体覆盖到 runs/.../src/
4. 渲染 validation/pybind.cpp 中的占位符
```

占位符渲染：

```text
aclnnCustomOp     -> aclnn{OpName}
custom_pybind_api -> {kernel_name}
```

重要判断：

- `answer/0/CMakeLists.txt` 会覆盖 msopgen 根 CMake，默认不要创建。
- `answer/0/op_host/CMakeLists.txt` 会覆盖 msopgen 默认 host CMake，只有需要改 host 编译规则时创建。
- `answer/0/op_kernel/CMakeLists.txt` 会覆盖默认 kernel CMake，只有 kernel 编译规则特殊时创建。

## 4. msopgen 默认 CMake 做了什么

### 4.1 根 CMake

`libs/common_include/msopgen/CMakeLists.txt` 会按目录存在情况加入：

```text
op_host/   -> host 侧 OpDef、tiling、opapi
op_kernel/ -> AscendC kernel
```

`validation/` 由 `build.sh` 在 OPP 编译后单独进入编译，用于生成 Python 扩展。

### 4.2 op_host 默认规则

默认 `op_host/CMakeLists.txt` 使用：

```cmake
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ops_srcs)
```

这只收集 `op_host` 当前目录下的 `.cpp`，不会递归收集：

```text
op_host/common/*.cpp
op_host/tiling_base/*.cpp
op_host/op_api/*.cpp
```

默认 host 编译会生成三类库：

```text
cust_op_proto
  输出: libcust_opsproto_rt2.0.so
  来源: OpDef、InferShape、autogen op_proto

cust_optiling
  输出: libcust_opmaster_rt2.0.so
  兼容入口: liboptiling.so
  来源: tiling 源码、IMPL_OP_OPTILING 或 .SetTiling 相关注册

cust_opapi
  输出: libcust_opapi.so
  来源: build_out/autogen/aclnn_*.cpp，或自定义 CMake 选择的本地 aclnn_*.cpp
```

`opbuild(OPS_SRC ... OUT_DIR build_out/autogen)` 会从 OpDef 生成：

```text
build_out/autogen/op_proto.*
build_out/autogen/aclnn_*.h
build_out/autogen/aclnn_*.cpp
build_out/autogen/fallback_*.cpp
```

### 4.3 op_kernel 默认规则

默认 `op_kernel/CMakeLists.txt` 调用：

```cmake
add_kernels_compile()
```

正常情况下，`op_kernel/{kernel_name}.cpp` 和相关头文件足够。编译成功后 verifier 会检查自定义 OPP 目录下是否存在与 `OpName` 相关的 kernel `.o`。

### 4.4 validation 默认规则

`validation/CMakeLists.txt` 生成：

```text
runs/.../build/kernel_gen_ops.*.so
```

它链接本次 `build_out/op_host` 中的 `cust_opapi`，以及 torch、torch_npu、tiling_api、ascendcl、acl_op_compiler、nnopbase 等运行依赖。

## 5. 编译产物必须长什么样

一次成功编译后，重点看：

```text
runs/.../src/build_out/op_host/
  op_api/lib/libcust_opapi.so
  op_api/include/aclnn_*.h
  op_impl/ai_core/tbe/op_tiling/liboptiling.so
  op_impl/ai_core/tbe/op_tiling/lib/linux/{arch}/libcust_opmaster_rt2.0.so
  op_proto/lib/linux/{arch}/libcust_opsproto_rt2.0.so
```

Python 扩展：

```text
runs/.../build/kernel_gen_ops.*.so
```

`liboptiling.so` 通常是兼容软链，真实库一般是：

```text
op_impl/ai_core/tbe/op_tiling/lib/linux/{arch}/libcust_opmaster_rt2.0.so
```

如果自定义 CMake 忘了 install 或忘了 `optiling_compat`，precision 会在运行阶段提前暴露为自定义 `liboptiling.so` 缺失或 `dlopen` 失败。

## 6. precision 运行时加载路径

precision worker 会找到包含下列文件的自定义 OPP root：

```text
op_api/lib/libcust_opapi.so
```

然后设置：

```text
ASCEND_CUSTOM_OPP_PATH=<custom_opp_root>/
CUST_OPAPI_LIB_PATH=<custom_opp_root>/op_api/lib/libcust_opapi.so
CUSTOM_OPTILING_LIB_PATH=<custom_opp_root>/op_impl/ai_core/tbe/op_tiling/liboptiling.so
LD_LIBRARY_PATH=<custom_opp_root>/op_api/lib:${LD_LIBRARY_PATH}
```

调用链：

```text
ModelNew.forward()
  -> kernel_gen_ops.{kernel_name}()
    -> EXEC_NPU_CMD(aclnn{OpName}, ...)
      -> dlopen(CUSTOM_OPTILING_LIB_PATH, RTLD_NOW | RTLD_GLOBAL)
      -> dlopen(CUST_OPAPI_LIB_PATH)
      -> dlsym(aclnn{OpName}GetWorkspaceSize)
      -> dlsym(aclnn{OpName})
      -> aclnn{OpName}GetWorkspaceSize(...)
         -> executor 按 OpType 查 tiling 注册表
      -> aclnn{OpName}(...)
```

迁移时不要在单个 `validation/pybind.cpp` 中重复写 `dlopen`。如果仍然出现同名内置 tiling 相关问题，优先检查：

```text
CUSTOM_OPTILING_LIB_PATH 是否存在
liboptiling.so 软链是否有效
ldd liboptiling.so 是否有 not found
自定义 CMake 是否把 tiling 源码编进 cust_optiling
```

## 7. 默认 msopgen CMake 什么时候够用

满足这些条件时，通常不需要写自定义 CMake：

- `op_host` 所有 `.cpp` 都在当前目录，不需要递归子目录。
- OpDef 可以驱动 `opbuild` 生成正确的 `aclnn_*.cpp`。
- 不需要手写 `aclnn_*.cpp`。
- 不需要额外 include 目录。
- 不需要额外 link library。
- 不需要特殊 compile definitions。
- kernel 可由默认 `add_kernels_compile()` 发现。

这种情况下 `answer/0` 可以只放：

```text
answer/0/
  op_host/
    {kernel_name}_def.cpp
    {kernel_name}_tiling.cpp
    {kernel_name}_tiling.h
    {kernel_name}_infershape.cpp   # 需要时
  op_kernel/
    {kernel_name}.cpp
    {kernel_name}.h                # 需要时
```

## 8. 什么时候必须增加或修改 CMake

### 8.1 op_host 有子目录源码

如果迁移后有：

```text
op_host/common/*.cpp
op_host/tiling_base/*.cpp
op_host/op_tiling/*.cpp
```

默认 CMake 不会编译这些文件。必须写 `op_host/CMakeLists.txt`，显式 `file(GLOB_RECURSE ...)` 或 `target_sources(...)`。

### 8.2 复杂 CANN 开源仓 tiling

出现这些特征时，通常要自定义 host CMake：

```text
TilingBaseClass
REGISTER_TILING_TEMPLATE
REGISTER_OPS_TILING_TEMPLATE
tiling_templates_registry
多个 *_tiling*.cpp
公共 common/log.h、common_dtype.h、tiling_base/*.cpp
```

要确保：

```text
*_def.cpp             -> opbuild 的 OPS_SRC
*_infershape.cpp      -> cust_op_proto
*_tiling*.cpp/.h      -> cust_optiling
common/tiling_base    -> cust_optiling 需要的源码和 include
```

### 8.3 需要手写 op_api

如果 autogen 的 `aclnn_*.cpp` 不满足原算子调用语义，或者原算子依赖手写 op_api：

```text
op_host/aclnn_xxx.cpp
op_host/aclnn_xxx.h
op_host/op_api/*.cpp
```

必须在 CMake 中选择这些本地源文件生成 `cust_opapi`，并确认导出：

```text
aclnn{OpName}GetWorkspaceSize
aclnn{OpName}
```

### 8.4 需要额外 include、宏或链接库

常见信号：

```text
找不到本地公共头
undefined reference
需要 OP_TILING_LIB / OP_PROTO_LIB 下不同代码分支
需要 link ops_base / opapi / 特定 CANN 库
```

这类问题不要靠复制头文件蒙混过编译，要确认目标源码被正确编进对应 target。

### 8.5 覆盖根 CMake 要非常谨慎

`answer/0/CMakeLists.txt` 会覆盖 msopgen 根 CMake。只有这些情况才考虑：

- 需要使用 `cann-ops` 风格工程组织。
- 需要特殊 binary package / cross compile 逻辑。
- 默认 msopgen 根 CMake 无法表达工程结构。

多数算子只需要覆盖 `op_host/CMakeLists.txt`，不要覆盖根 CMake。

## 9. 常用检查命令

检查 opapi：

```bash
find runs -path "*op_api/lib/libcust_opapi.so" -print
```

检查 tiling：

```bash
find runs -path "*op_tiling/liboptiling.so" -print
find runs -name "libcust_opmaster_rt2.0.so" -print
```

检查 kernel：

```bash
find runs -name "*<OpName>*.o" -print
```

检查 pybind：

```bash
find runs -name "kernel_gen_ops*.so" -print
```

检查 so 依赖：

```bash
ldd <custom_opp_root>/op_impl/ai_core/tbe/op_tiling/liboptiling.so
ldd <custom_opp_root>/op_api/lib/libcust_opapi.so
```

## 10. Agent 决策清单

迁移时按这个顺序判断：

1. OpDef 是否足够生成正确 autogen opapi？不够就准备手写 op_api 和 CMake。
2. `op_host` 是否只有当前目录 `.cpp`？不是就准备自定义 `op_host/CMakeLists.txt`。
3. tiling 注册是 `.SetTiling` 还是 `IMPL_OP_OPTILING`？不要两种混用。
4. `TilingParse<CompileInfo>` 和 `GetCompileInfo()` 是否成对存在？
5. TilingData 字段、顺序、类型是否和 kernel 读取一致？
6. 自定义 CMake 是否保留 `cust_op_proto`、`cust_optiling`、`cust_opapi`、`optiling_compat` 和 install 规则？
7. 编译产物是否包含 `libcust_opapi.so`、`liboptiling.so`、kernel `.o`、`kernel_gen_ops.*.so`？
8. precision 报加载错误时，先查 OPP 产物和 `ldd`，再查算子逻辑。
