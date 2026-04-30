# 两种编译框架的完整说明

本文帮助你理解 CANN 开源仓（"源"）和 NPUKernelBench msopgen（"目标"）之间的所有关键差异。

## 1. 编译基础设施

### CANN 开源仓（源）

开源仓是一个多算子的大型 CMake 工程。根 `CMakeLists.txt` 提供：
- 公共编译函数：`add_modules_sources()`、`add_onnx_plugin_sources()`
- 公共 target：`nnopbase`、`exe_graph`、`register`、`tiling_api`
- 公共 include 路径：`src/common/inc/`（含 `tiling/`、`log/`、`error/`、`kernel/` 等）
- 公共 CMake 选项：`TILING_DIR`、`COMPUTE_UNIT`、`DEPENDENCIES`

每个算子 `CMakeLists.txt` 调用这些公共函数来编译，例如：
```cmake
add_modules_sources(ops/operator_name
    DEPENDENCIES common_tiling
    TILING_DIR membase
)
```

### NPUKernelBench msopgen（目标）

msopgen 是一个**单算子独立编译单元**。编译脚手架位于 `libs/common_include/msopgen/`：

```
libs/common_include/msopgen/
├── CMakeLists.txt          # 根 CMake（按序 include op_host/ 和 op_kernel/）
├── CMakePresets.json       # 预设 ASCEND_CANN_PACKAGE_PATH 和芯片配置
├── build.sh                # 编译脚本（两步：算子编译 + pybind 扩展编译）
├── cmake/
│   ├── config.cmake        # 项目配置、vendor 名称等
│   ├── func.cmake          # opbuild()、add_kernels_compile() 等函数定义
│   ├── intf.cmake          # 接口库定义（intf_pub 等）
│   ├── device_task.cmake   # device 侧编译任务
│   ├── makeself.cmake      # CPack 打包
│   └── util/               # Python 辅助脚本和 shell 工具
├── op_host/
│   └── CMakeLists.txt      # 默认 host 编译规则（收集当前目录 .cpp）
├── op_kernel/
│   └── CMakeLists.txt      # 默认 kernel 编译规则
└── validation/
    ├── CMakeLists.txt      # pybind Python 扩展编译
    └── pytorch_npu_helper.hpp  # EXEC_NPU_CMD 宏定义
```

#### 编译流程

`build.sh` 执行两步编译：

**第一步：算子编译**
```bash
cmake -S . -B build_out --preset=default
cmake --build build_out --target binary -j$(nproc)
cmake --build build_out --target package -j$(nproc)
```

这一步产生三个核心库：
- `libcust_opsproto_rt2.0.so` — 算子 proto（OpDef → protobuf → 序列化）
- `libcust_opmaster_rt2.0.so` / `liboptiling.so` — tiling 逻辑
- `libcust_opapi.so` — aclnn API（从 OpDef 自动生成）

注意：aclnn API **始终由 `opbuild` 从 OpDef 自动生成**。这些 autogen 文件位于 `build_out/autogen/`：
- `aclnn_*.h` / `aclnn_*.cpp` — aclnn API
- `op_proto.h` / `op_proto.cc` — proto 序列化
- `group_proto/*` — 如果有 group proto

**第二步：Python 扩展编译**
```bash
cd validation
cmake --no-warn-unused-cli -B ../../build \
    -DSHARED_LIB_DIR=${script_path}/build_out/op_host/ \
    -DPython3_EXECUTABLE=$(which python3)
cmake --build ../../build -j16
```

产生 `runs/.../build/kernel_gen_ops.*.so`，这是一个 Python C 扩展，可通过 `import kernel_gen_ops` 导入。

#### 默认 op_host CMake 行为

```cmake
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ops_srcs)
opbuild(OPS_SRC ${ops_srcs} OUT_DIR ${ASCEND_AUTOGEN_PATH})
```

**`aux_source_directory` 只收集当前目录的 `.cpp`，不递归子目录。**如果有子目录（`common/`、`tiling_base/`）中的 `.cpp` 需要编译，必须写自定义 `op_host/CMakeLists.txt`（见 `04-cmake-guide.md`）。

`opbuild()` 从 `ops_srcs` 中解析 OpDef（通过 `OP_ADD(OpName)` 宏），生成 autogen aclnn API 代码。

#### 默认 op_kernel CMake 行为

```cmake
add_kernels_compile()
```

编译所有 `.cpp` 为 kernel `.o` 文件。

## 2. 产物与加载

### 编译成功必需的产物

```
runs/.../src/build_out/op_host/
├── op_api/lib/libcust_opapi.so
└── op_impl/ai_core/tbe/op_tiling/
    ├── liboptiling.so
    └── lib/linux/<arch>/libcust_opmaster_rt2.0.so

runs/.../build/
└── kernel_gen_ops.*.so
```

### 精度验证时的加载路径

`verifier.py` 中的 `custom_opp_runtime_env()` 搜索 libcust_opapi.so 时有多个候选路径：

```python
candidates = [
    os.path.join(search_dir, "build_out", "op_host"),           # primary
    os.path.join(search_dir, "build_out", "packages", "vendors"), # fallback
    os.path.join(search_dir, "build_out", "op_host", "packages", "vendors"),
    os.path.join(search_dir, "output", "packages", "vendors"),
]
```

然后设置环境变量：

```python
ASCEND_CUSTOM_OPP_PATH = <root>/
CUST_OPAPI_LIB_PATH = <root>/op_api/lib/libcust_opapi.so
CUSTOM_OPTILING_LIB_PATH = <root>/op_impl/ai_core/tbe/op_tiling/liboptiling.so
LD_LIBRARY_PATH = <root>/op_api/lib:${LD_LIBRARY_PATH}
```

Python 通过 `sys.path.append(build_dir)` 导入 `kernel_gen_ops.*.so`。

### `-skip_op_build` 模式

当 `op_host/` 和 `op_kernel/` 代码未修改，只修改了 `validation/` 目录下的文件时使用：

```bash
python scripts/run_test.py -task_name <OpName> -stages compile precision -skip_op_build
```

行为：复用已有 OPP 产物，仅重新编译 `validation/` pybind 扩展。

注意：修改 `op_host/` 或 `op_kernel/` 后必须先执行完整 compile（不带 `-skip_op_build`）。

## 3. 源与目标的关键差异速查表

| 维度 | CANN 开源仓（源） | NPUKernelBench msopgen（目标） |
|------|-------------------|-------------------------------|
| **工程范围** | 多算子统一编译 | 单算子独立编译 |
| **CMake 函数** | `add_modules_sources()` 等仓库级函数 | `opbuild()`、`add_kernels_compile()` |
| **aclnn API** | autogen 或手写 `op_api/` | 仅 `opbuild` 从 OpDef 自动生成，禁止手写 |
| **公共依赖** | `src/common/inc/` 全局 include | 必须本地化到 `answer/0/op_host/` |
| **芯片配置** | 多种芯片 | 仅 `ascend910b` 和 `ascend910_93` |
| **编译产物路径** | 仓库级 build 目录 | 每个算子独立的 `runs/` 子目录 |
| **TilingParse** | 图编译路径下触发 | **aclnn 直调路径（pybind）不会调用，禁止依赖** |
| **CompileInfo** | 图编译路径下通过 TilingParse 填充 | **aclnn 直调路径下 GetCompileInfo() 始终返回 nullptr，改用 PlatformAscendC 直接获取** |

## 4. 目录命名规则

框架从任务目录名自动解析：

```
tasks/level{N}/{Category}/{OpName}/
```

- `OpName` = 目录名（PascalCase，如 `ScatterElementsV2`）
- `kernel_name` = `OpName` 转 snake_case（如 `scatter_elements_v2`）
- `short_id` = `lvl{N}_category{Category}_{OpName}`

`kernel_name` 用于：pybind 注册函数名、host/kernel 文件名中的前缀。

## 5. 快速对比命令

```bash
# 在开源仓源算子目录运行，了解源结构
find . -maxdepth 3 -type f | sort
rg -n 'TilingContext|SetTiling|REGISTER_TILING|IMPL_OP_OPTILING|TilingBaseClass' op_host
rg -n '__global__|GM_ADDR|TILING_KEY_IS' op_kernel
find op_kernel -type d -name "arch35" -o -name "membase"
find op_kernel -name "*_apt.cpp"

# 在已有成功迁移的同类算子目录运行，参考目标结构
find tasks/level{N}/{Category}/ -name "answer" -path "*/0/*" | head -5
```
