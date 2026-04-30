# CMake 适配指南

本文解释什么时候需要自定义 CMake，以及如何编写。

## 1. 默认 CMake 行为回顾

msopgen 脚手架提供以下默认 CMake 规则（位于 `libs/common_include/msopgen/`）：

**根 CMakeLists.txt：**
```cmake
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_host)
    add_subdirectory(op_host)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)
    add_subdirectory(op_kernel)
endif()
```

**默认 op_host/CMakeLists.txt：**
```cmake
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ops_srcs)
opbuild(OPS_SRC ${ops_srcs} OUT_DIR ${ASCEND_AUTOGEN_PATH})
# ... 编译 cust_op_proto, cust_optiling, cust_opapi ...
```

关键限制：**`aux_source_directory` 只收集当前目录的 `.cpp`，不递归子目录。**

aclnn API **始终由 `opbuild` 从 OpDef 自动生成**。不需要、也不应该手写 `aclnn_*.cpp` 或 `aclnn_*.h`。

**默认 op_kernel/CMakeLists.txt：**
```cmake
add_kernels_compile()
```

## 2. 决策：是否需要自定义 CMake

```
answer/0/op_host/ 下有子目录且包含 .cpp 文件？
  ├── 是 → 需要自定义 op_host/CMakeLists.txt
  └── 否 → 默认 CMake 足够
```

| 场景 | 需要的 CMake 文件 |
|------|-------------------|
| 只有 `op_host/*.cpp` + `op_host/*.h`（无子目录） | 不需要任何自定义 CMake |
| `op_host/` 有子目录（`common/`、`tiling_base/` 等）含 `.cpp` | 需要 `answer/0/op_host/CMakeLists.txt` |
| 需要覆盖整个根 CMake 结构 | 需要 `answer/0/CMakeLists.txt`（极度罕见） |

## 3. 自定义 op_host/CMakeLists.txt

### 3.1 从成功样例复制

最好的方法是从已迁移成功的同类算子复制 `op_host/CMakeLists.txt`：

```bash
# 找已迁移且使用了 IMPL_OP_OPTILING 或复杂 CMake 的算子
rg -l "file.*GLOB_RECURSE" tasks/ --include="CMakeLists.txt" | head -5
```

然后：
```bash
# 复制同类型算子的 CMakeLists.txt
cp tasks/level3/<Category>/<SimilarOp>/answer/0/op_host/CMakeLists.txt \
   tasks/level{N}/{Category}/{OpName}/answer/0/op_host/CMakeLists.txt
```

在复制的 CMakeLists.txt 中替换文件路径中的算子名。

### 3.2 多文件 tiling 编译的核心结构

当 `op_host/` 下有子目录需要编译时，CMake 的核心是分离 `DEF_CPP`（opbuild 处理的 OpDef 源文件）、`TILING_SRC`（tiling 逻辑源文件）和 `INFERSHAPE_SRC`（shape 推导源文件）：

```cmake
file(GLOB_RECURSE OPS_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE DEF_CPP "${CMAKE_CURRENT_SOURCE_DIR}/*def.cpp")
file(GLOB_RECURSE TILING_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*tiling*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*tiling*.h")
file(GLOB_RECURSE INFERSHAPE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*infershape*.cpp")

opbuild(OPS_SRC ${DEF_CPP}
        OUT_DIR ${ASCEND_AUTOGEN_PATH}
        INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
)

# proto 库
file(GLOB group_proto_src ${ASCEND_AUTOGEN_PATH}/group_proto/*.cc)
add_library(cust_op_proto SHARED
    $<$<TARGET_EXISTS:group_proto_src>:${group_proto_src}>
    ${INFERSHAPE_SRC}
    ${ASCEND_AUTOGEN_PATH}/op_proto.cc
)
target_include_directories(cust_op_proto PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/external
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB)
target_link_libraries(cust_op_proto PRIVATE
    intf_pub exe_graph register tiling_api
    -Wl,--whole-archive rt2_registry -Wl,--no-whole-archive
)
set_target_properties(cust_op_proto PROPERTIES OUTPUT_NAME cust_opsproto_rt2.0)

# tiling 库：选中所有 tiling 相关的 .cpp
add_library(cust_optiling SHARED ${TILING_SRC} ${INFERSHAPE_SRC})
target_include_directories(cust_optiling PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/external
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)
target_link_libraries(cust_optiling PRIVATE
    nnopbase intf_pub exe_graph register tiling_api ops_base
    -Wl,--whole-archive rt2_registry -Wl,--no-whole-archive
)
set_target_properties(cust_optiling PROPERTIES OUTPUT_NAME cust_opmaster_rt2.0)
```

上述片段展示了 proto/tiling 两个核心库的选源和 include/link 规则。完整 CMake 还需保留 `cust_opapi`、`optiling_compat`、install 等打包逻辑。实际迁移时**优先参考本仓已通过编译的同类算子**的完整 `op_host/CMakeLists.txt`。

### 3.3 确认打包产物

编译后确认产物包含运行时所需的 tiling 库：

```bash
find runs -path "*<OpName>*" -name "libcust_opmaster_rt2.0.so"
find runs -path "*<OpName>*" -name "liboptiling.so"
```

## 4. 覆盖根 CMakeLists.txt（`answer/0/CMakeLists.txt`）

**极度罕见**。仅在需要完全自定义编译流程时使用。如果确实需要，从 `libs/common_include/msopgen/CMakeLists.txt` 复制然后在基础上修改。

## 5. 常见问题

### `aux_source_directory` 没收集到 `.cpp`

子目录中的 `.cpp` 不会被收集。使用 `file(GLOB_RECURSE ...)` 或显式列出。

### `opbuild` 没生成 autogen 文件

检查：`_def.cpp` 中是否有 `OP_ADD(OpName)`，`_def.cpp` 是否被 `opbuild` 的 `OPS_SRC` 参数包含，OpDef 类是否在 `namespace ops { }` 中且构造函数为 public。

### 编译成功但运行时找不到 tiling

检查：
1. `liboptiling.so` 是否在正确的打包路径下
2. `ldd runs/.../liboptiling.so` 是否有未解析符号
3. `ASCEND_CUSTOM_OPP_PATH` 是否指向正确的 custom OPP 根目录

如果 CANN 中有同名内置算子，自定义 tiling 可能被内置覆盖。可在 `validation/pybind.cpp` 中显式 `dlopen` 自定义 `liboptiling.so`：
```cpp
#include <dlfcn.h>
const char* opp_path = std::getenv("ASCEND_CUSTOM_OPP_PATH");
if (opp_path) {
    std::string lib_path = std::string(opp_path)
        + "/op_impl/ai_core/tbe/op_tiling/liboptiling.so";
    dlopen(lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
}
```

### 多文件 tiling 编译后有未解析符号

符号可能来自保留的调试日志调用。删除对应的 include 和调试日志调用即可。不要复制调试函数实现——它们不属于 tiling 语义。
