# NPUKernelBench msopgen 契约

需要重新判断框架复制、CMake、产物校验或运行时加载路径时读取本文。本文中的“复制”专指 NPUKernelBench 编译前的框架行为；迁移 agent 维护 `tasks/.../answer/0`，不直接改 `runs/.../src`。

## 任务与命名

任务目录为：

```text
tasks/level{N}/{Category}/{OpName}/
```

框架从目录名解析：

```text
OpName      = 目录名中 "_" 之前的部分
kernel_name = OpName 转 snake_case
short_id    = lvl{N}_category{Category}_{OpName}
```

`validation/test_cases.csv` 决定 case 数；`answer/{sample_id}/test_cases.csv` 存在时优先于 validation 侧 CSV。

## 框架编译源码准备

默认配置：

```yaml
kernel_only_mode: True
compile:
  mode: msopgen
  common_dir: libs/common_include
```

`prepare_source_files()` 的顺序由框架执行：

```text
1. copy libs/common_include/msopgen/ -> runs/.../src/
2. copy tasks/.../validation/       -> runs/.../src/validation/
3. copy answer/{sample_id}/         -> runs/.../src/
4. render validation/pybind.cpp placeholders
```

因此 `answer/0` 的文件会落在 msopgen 工程根：

```text
answer/0/op_host/*   -> runs/.../src/op_host/*
answer/0/op_kernel/* -> runs/.../src/op_kernel/*
answer/0/CMakeLists.txt -> runs/.../src/CMakeLists.txt
```

创建 `answer/0/CMakeLists.txt` 等于覆盖 msopgen 根 CMake。只在确实需要改变根编译结构时添加。

## 默认 CMake 行为

根 CMake 会加入存在的 `op_host/` 和 `op_kernel/`。

默认 `op_host/CMakeLists.txt`：

```cmake
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ops_srcs)
opbuild(OPS_SRC ${ops_srcs} OUT_DIR ${ASCEND_AUTOGEN_PATH})
```

它只收集 `op_host` 当前目录 `.cpp`，不递归子目录。host 构建会从 OpDef 生成 autogen aclnn API，并产出：

```text
libcust_opsproto_rt2.0.so
libcust_opmaster_rt2.0.so / liboptiling.so
libcust_opapi.so
build_out/autogen/aclnn_*.h
build_out/autogen/aclnn_*.cpp
```

迁移时不从开源仓导入 `op_api/`，不手写 `aclnn_*.cpp` 或 `aclnn_*.h`。如果 autogen 头缺失，修 OpDef、`OP_ADD` 或 host CMake 收录，而不是补本地 aclnn 源码。

默认 `op_kernel/CMakeLists.txt` 只调用 `add_kernels_compile()`，适合常规 AscendC kernel。

`validation/CMakeLists.txt` 生成 Python 扩展：

```text
runs/.../build/kernel_gen_ops.*.so
```

## 成功产物

编译成功至少应存在：

```text
runs/.../build/kernel_gen_ops.*.so
runs/.../src/build_out/op_host/op_api/lib/libcust_opapi.so
runs/.../src/build_out/op_host/op_impl/ai_core/tbe/op_tiling/liboptiling.so
```

verifier 还会在自定义 OPP root 下查找与 `OpName` 相关的 kernel `.o`。

## precision 加载

precision worker 根据 `libcust_opapi.so` 找到 custom OPP root，并设置：

```text
ASCEND_CUSTOM_OPP_PATH=<root>/
CUST_OPAPI_LIB_PATH=<root>/op_api/lib/libcust_opapi.so
CUSTOM_OPTILING_LIB_PATH=<root>/op_impl/ai_core/tbe/op_tiling/liboptiling.so
LD_LIBRARY_PATH=<root>/op_api/lib:${LD_LIBRARY_PATH}
```

`kernel_gen_ops` 从 `runs/.../build` 导入。`EXEC_NPU_CMD` 通过渲染后的占位符调用自动生成的 `aclnn{OpName}`，并依赖本次 custom opapi 和 optiling 库。

## `-skip_op_build` 语义

`python scripts/run_test.py -task_name <OpName> -stages compile` 默认执行完整算子编译。已有成功 OPP 产物后，再运行不带 `-skip_op_build` 的 `-stages compile precision` 会重新编译算子。

`-skip_op_build` 的行为：

```text
复用已有 op_host/op_kernel OPP 产物
重新编译 validation/pybind Python 扩展
继续执行 precision 阶段
```

适用场景：

- 编译已经通过，准备跑少量或全量 precision
- 只修改了 `validation/`、`pybind.cpp`、`module.py`、`prepare_inputs.py` 或 `test_cases.csv`
- 需要重新生成 `kernel_gen_ops.*.so`，但不需要重编 OPP

修改 `op_host/` 或 `op_kernel/` 后，删除当前算子的 `runs/lvl{N}/{Category}/{OpName}`，重新执行完整 compile，再用 `-skip_op_build` 跑 precision。

## 常用检查

```bash
find runs/lvl{N}/{Category}/{OpName} -name 'aclnn_*.h' -o -name 'libcust_opapi.so' -o -name 'liboptiling.so' -o -name 'kernel_gen_ops*.so'
rg -n 'aclnn|custom_pybind_api|kernel_gen_ops|EXEC_NPU_CMD' tasks/level{N}/{Category}/{OpName}/validation
```
