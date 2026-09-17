# MoeTutelCombineX 样例构建与运行说明

本目录提供算子的 aclnn 调用样例 [test_aclnn_moe_tutel_combine_x.cpp](test_aclnn_moe_tutel_combine_x.cpp)（2D 无 batch 形态：`y_grad [4, 2]`、`gates/indices/locations [4]`、输出 `x_grad [4, 2]`）。样例为独立 CMake 工程，不属于算子仓主构建目标（主构建仅产出算子 ophost/opapi/opgraph 交付物），按下述步骤单独编译运行。

## 前提条件

- 开发与运行环境合设：已安装驱动、固件、CANN 软件包（`ascend-toolkit`），并完成环境变量注入（如 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，路径以实际安装位置为准）。
- 算子已通过仓根 `build.sh` 完成构建并安装（opapi 交付物随算子包装入运行环境），样例依赖其头文件 `aclnnop/aclnn_moe_tutel_combine_x.h` 与算子符号。
- `ASCEND_INSTALL_PATH` 环境变量指向 CANN 安装路径（默认 `/usr/local/Ascend/ascend-toolkit/latest`）。

## 编译步骤

在任意工作目录（可为本目录）创建 `CMakeLists.txt`：

```bash
cmake_minimum_required(VERSION 3.18.4)
project(MOE_TUTEL_COMBINE_X_EXAMPLE)
add_compile_options(-std=c++11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./bin")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

if(NOT DEFINED ENV{ASCEND_INSTALL_PATH})
    set(ASCEND_INSTALL_PATH "/usr/local/Ascend/ascend-toolkit/latest")
else()
    set(ASCEND_INSTALL_PATH $ENV{ASCEND_INSTALL_PATH})
endif()

add_executable(test_aclnn_moe_tutel_combine_x
               test_aclnn_moe_tutel_combine_x.cpp)
target_include_directories(test_aclnn_moe_tutel_combine_x PRIVATE
                           ${ASCEND_INSTALL_PATH}/include)
target_link_directories(test_aclnn_moe_tutel_combine_x PRIVATE
                        ${ASCEND_INSTALL_PATH}/lib64)
target_link_libraries(test_aclnn_moe_tutel_combine_x
                      ascendcl opapi)
```

> 链接库名以运行环境实际为准：CANN 标准布局为 `ascendcl`（ACL 基础接口）与 `opapi`（aclnn 算子接口入口，`libopapi.so`）。若链接报未定义 `aclnnMoeTutelCombineX*` 符号，请确认算子包已安装且 `LD_LIBRARY_PATH` 包含其 `lib64`。

编译：

```bash
mkdir -p build && cd build
cmake .. && make
```

## 运行步骤

```bash
# 设备 ID 默认为 0，可通过样例内 deviceId 修改
./bin/test_aclnn_moe_tutel_combine_x
```

预期输出（E=2、C=2、EC=4、H=2、S=4 的微型数值例）：

```text
xGrad[0] is: 0.200000
xGrad[1] is: 2.000000
xGrad[2] is: 0.600000
xGrad[3] is: 6.000000
xGrad[4] is: 1.200000
xGrad[5] is: 12.000000
xGrad[6] is: 0.400000
xGrad[7] is: 4.000000
```

即 `x_grad = [[0.2, 2], [0.6, 6], [1.2, 12], [0.4, 4]]`（`x_grad[t, :] = gates[t] * y_grad[indices[t]*C + locations[t], :]`，`gates = [0.1, 0.2, 0.3, 0.4]`）。

## 参考

- 通用编译运行流程：[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)
- 两段式接口说明：[两段式调用接口](../../../docs/zh/context/two_phase_api.md)
- 返回码说明：[aclnn 返回码](../../../docs/zh/context/aclnn_return_code.md)
- 算子接口详情：[aclnnMoeTutelCombineX](../docs/aclnnMoeTutelCombineX.md)
