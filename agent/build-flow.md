# 编译环节静态分析

## 总入口

仓库编译入口是根目录 `build.sh`，核心 CMake 入口是根目录 `CMakeLists.txt`。

常用命令：

```bash
# 构建指定算子的 custom package
bash build.sh --pkg --soc=ascend910b --ops=<op_name> --vendor_name=custom -j16

# 构建所有默认算子 custom package
bash build.sh --pkg --soc=ascend910b --vendor_name=custom -j16

# 只生成算子源码或框架相关目标时，可组合 --ophost、--opapi、--opgraph、--opkernel
bash build.sh --opkernel --soc=ascend910b --ops=<op_name>
```

`build.sh --help` 暴露的主要编译参数包括：

- `--pkg`：生成安装包。
- `--jit`：打开 JIT 相关构建路径。
- `--soc`：指定 SoC，默认静态值为 `ascend910b`。
- `--vendor_name`：custom package 的 vendor 名称。
- `--ops`：限定编译的算子名，逗号分隔。
- `--experimental`：纳入 `experimental/` 下算子。
- `--ophost`、`--opapi`、`--opgraph`、`--onnxplugin`、`--opkernel`：选择库或子模块。
- `--kernel_template_input`、`--bisheng_flags`、`--oom`：影响 kernel 编译和资源约束。
- `-u`、`--ophost_test`、`--opapi_test`、`--opgraph_test`、`--opkernel_test`：进入测试编译链路。

## 默认输出目录

| 路径 | 作用 |
| --- | --- |
| `build/` | CMake build 目录、autogen、binary 中间产物、test 中间产物。 |
| `output/` | 部分构建输出或中间输出目录。 |
| `build_out/` | `CMAKE_INSTALL_PREFIX`，安装包和 install 产物输出目录。 |
| `build/autogen/` | `op_build` 生成的 aclnn、proto、register 等自动代码。 |
| `build/binary/<soc>/` | kernel 编译脚本、复制后的源码、bin 和算子配置生成目录。 |

注意：`build.sh` 的 clean 阶段会删除 `build/` 和 `output/`。不要把手工资料放进这两个目录。

## 端到端编译流程

### 1. 参数解析和环境定位

模块：`build.sh`

输入：

- 用户命令行参数，例如 `--pkg --soc=ascend910b --ops=flash_attention_score`。
- CANN 路径参数 `-p`，或环境变量 `ASCEND_HOME_PATH`、`ASCEND_OPP_PATH`。
- 默认候选路径 `/usr/local/Ascend/ascend-toolkit/latest`、`~/Ascend/ascend-toolkit/latest`、`/usr/local/Ascend/latest`。

处理：

- 设置 `ASCEND_SOC_UNITS`，默认 `ascend910b`。
- 校验 SoC 是否在支持列表中。
- 定位 CANN package path。
- `source $ASCEND_CANN_PACKAGE_PATH/bin/setenv.bash`。
- 非 `--jit --noaicpu` 场景检查 `bisheng` 是否可用。

输出：

- CMake 配置参数 `CUSTOM_OPTION`。
- 环境变量和路径变量。
- 清理后的 `build/`、`output/`。

### 2. CMake 顶层配置

模块：`CMakeLists.txt`

输入：

- `build.sh` 传入的 `-D` 参数。
- `version.cmake` 中 package 版本和 CANN 依赖版本。
- `cmake/variables.cmake`、`cmake/func.cmake`、`cmake/custom_build.cmake`、`cmake/opbuild.cmake`。

处理：

- 设置 `BUILD_OPEN_PROJECT`、`ENABLE_TILING_SINK`、`ENABLE_AICPU`、`ENABLE_EXPERIMENTAL`、`ENABLE_TEST` 等选项。
- 将 `ASCEND_COMPUTE_UNIT` 映射到架构目录：
  - `ascend310p` -> `arch20`
  - `ascend910b` -> `arch22`
  - `ascend910_93` -> `arch22`
  - `ascend950` -> `arch35`
  - `mc62cm12a` -> `arch38`
  - `kirinx90`、`kirin9030` -> `arch22`
- 加载 `common`、`mc2`、`posembedding`、`moe`、`ffn`、`attention` 等子目录。
- 根据 `ASCEND_OP_NAME` 和 `ENABLE_EXPERIMENTAL` 决定参与编译的算子集合。

输出：

- CMake target 列表。
- `COMPILED_OPS`、`COMPILED_OP_DIRS` 等被选中的算子集合变量。
- 自动生成目录和 install 规则。

### 3. 算子发现和依赖展开

模块：`cmake/func.cmake`

输入：

- 主线算子域下的 `CMakeLists.txt`、`op_host/CMakeLists.txt`。
- `ASCEND_OP_NAME`，即 `--ops` 传入的算子过滤条件。
- `ENABLE_EXPERIMENTAL`。

处理：

- `op_add_subdirectory()` 用 `file(GLOB_RECURSE ...)` 扫描算子目录。
- 对匹配目录执行 `add_subdirectory()`。
- 通过 `add_op_to_compiled_list()` 记录参与编译的算子。
- 通过 `op_add_depend_directory()` 引入 `${op_name}_depends` 声明的依赖。

输出：

- 每个算子的 CMake 子目标。
- 供后续 `opbuild`、binary compile 和 package 使用的算子列表。

### 4. Host/API/Graph 代码构建和自动生成

模块：

- 算子目录下 `op_host/`
- 算子目录下 `op_host_aclnn/`、`op_host_aclnnInner/`、`op_host_aclnnExc/`
- `cmake/opbuild.cmake`
- `cmake/custom_build.cmake`

输入：

- `op_host` 中的算子定义、shape 推导、tiling 逻辑、注册文件。
- `op_host_aclnn*` 中的 aclnn API 定义。
- CANN 的 `opbuild` 工具。

处理：

- `gen_aclnn_with_opdef()` 收集 `op_host_aclnn*` 下的 opdef 源码。
- 调用 `OP_BUILD_TOOL` 生成 aclnn stub、proto 信息和统一头文件。
- 构建 `cust_opapi`、`cust_proto`、`cust_opmaster`、`optiling_compat` 等 custom package 需要的库或产物。

输出：

- `build/autogen/` 下的自动生成源码。
- `aclnn_ops_transformer*.h`。
- custom package 需要安装的 host/API/graph 相关库和配置。

### 5. AscendC kernel 编译脚本生成

模块：

- `cmake/func.cmake`
- `cmake/scripts/util/ascendc_bin_param_build.py`
- `cmake/scripts/util/parse_ini_to_json.py`

输入：

- 每个算子 `op_kernel/` 下源码。
- op info ini，例如 `aic-<soc>-ops-info.ini`。
- tiling key、kernel name、输入输出 dtype/format 等配置。

处理：

- `add_compile_cmd_target()` 调用 `ascendc_bin_param_build.py`。
- 根据 op info 生成每个 kernel 的编译 shell 脚本。
- `add_ops_info_target()` 将 ini 转为 json 配置。

输出：

- `build/binary/<soc>/gen/*.sh`：kernel 编译脚本。
- `build/binary/<soc>/gen/*ops-info.json`：算子配置 json。

补充：

- `cmake/custom_build.cmake` 会调用 `cmake/scripts/util/ascendc_impl_build.py` 生成 AscendC 动态实现相关 py 文件。
- `cmake/func.cmake` 后续还会调用 `cmake/scripts/util/ascendc_ops_config.py` 处理 binary 输出目录中的算子配置。

### 6. AscendC kernel binary 构建

模块：`cmake/func.cmake` 中 `add_bin_compile_target()`

输入：

- 上一步生成的 kernel 编译脚本。
- 复制到 `build/binary/<soc>/src/<op>/` 的算子源码。
- 生成的 dynamic py 文件。
- CANN AscendC compiler 和 `bisheng`。

处理：

- `add_ops_src_copy()` 将 `op_kernel` 相关源码复制到 binary build 源码目录。
- 对每个生成脚本执行 `bash <bin_script> <dynamic_py> <bin_out>`。
- 按 SoC 编译出 `.o`、`.json`、`.bin` 等 kernel 相关产物。

输出：

- `build/binary/<soc>/bin/` 下的 kernel binary。
- package install 规则引用的 AI Core 实现和配置。

### 7. package 生成

模块：

- `cmake/custom_build.cmake`
- CPack / makeself 相关 CMake 规则

输入：

- host/API/graph 库。
- AI Core binary 和 config。
- vendor 名称。
- `version.cmake` 中版本和依赖元信息。

处理：

- 按 custom package 路径组织安装目录。
- custom package 路径形态通常为 `packages/vendors/<vendor_name>_transformer/...`。
- built-in package 和 custom package 路径不同，当前后续优化通常更关注 custom package。

输出：

- `build_out/` 下的 `cann-ops-transformer-<vendor>_linux-<arch>.run` 或相关安装产物。

## 编译模块输入输出总表

| 模块 | 主要输入 | 主要输出 |
| --- | --- | --- |
| `build.sh` | 命令行参数、CANN 路径、环境变量 | CMake 参数、清理后的 build 目录、执行入口 |
| `CMakeLists.txt` | CMake 选项、SoC、算子过滤条件 | target 图、算子集合、install 规则 |
| `op_add_subdirectory()` | 算子目录、`--ops`、`--experimental` | 被纳入构建的算子子目录 |
| `op_host` | op def、infer shape、tiling、注册代码 | host/tiling/proto 目标和配置 |
| `op_host_aclnn*` | aclnn API 定义 | aclnn stub、统一头文件、opapi 产物 |
| `op_kernel` | AscendC 源码、tiling data、arch 源码 | kernel 编译输入源码 |
| `cmake/scripts/util/ascendc_impl_build.py` | op build 输出、算子实现信息 | dynamic py |
| `cmake/scripts/util/ascendc_bin_param_build.py` | op info ini、kernel 元数据 | binary 编译脚本 |
| `cmake/scripts/util/parse_ini_to_json.py` | op info ini | ops-info json |
| `cmake/scripts/util/ascendc_ops_config.py` | binary 输出目录、SoC | binary config |
| `add_bin_compile_target()` | 编译脚本、dynamic py、复制后的源码 | AI Core kernel binary |
| package 规则 | host/API/graph 库、kernel binary、config | custom package runfile |

## B3/910B 编译关注点

- 用户当前 B3 环境应优先使用 `--soc=ascend910b`。
- `ascend910b` 在 CMake 中对应 `arch22`，后续优化要优先查看 `op_kernel/arch22` 或条件编译分支。
- 如果只优化单个算子，使用 `--ops=<op_name>` 可以缩短 build 和验证范围。
- 若修改公共目录，例如 `attention/common/` 或 `mc2/3rd/`，影响面可能跨多个算子，不能只看单算子 build。
- `--bisheng_flags`、`--kernel_template_input`、`--oom` 可能影响 kernel 编译和资源使用，后续性能调优时需要记录每次配置。
