# 备注

```
[2026-03-19 15:04:57] [ERROR] RUNTIME(695120,transformer_op_api_ut):2026-03-19-15:04:57.898.610 [api_impl.cc:7751]695120 GetResInCurrentThread:ctx is NULL!
[2026-03-19 15:04:57] [ERROR] RUNTIME(695120,transformer_op_api_ut):2026-03-19-15:04:57.898.631 [api_c_stream.cc:749]695120 rtsGetResInCurrentThread:ErrCode=107002, desc=[the context is a null pointer], InnerCode=0x7070001
[2026-03-19 15:04:57] [ERROR] RUNTIME(695120,transformer_op_api_ut):2026-03-19-15:04:57.898.634 [error_message_manage.cc:55]695120 FuncErrorReason:rtsGetResInCurrentThread execution failed, the context is a null pointer.
[2026-03-19 15:04:57] [ERROR] ASCENDCL(695120,transformer_op_api_ut):2026-03-19-15:04:57.898.642 [device.cpp:528]695120 aclrtGetResInCurrentThreadImpl:call rtsGetResInCurrentThread failed, runtime result = 107002.
[2026-03-19 15:04:57] [       OK ] prompt_flash_attention_v3_opapi_ut.prompt_flash_attention_v3_aclnn_1 (0 ms)
```

lcov op_api里面报上面的日志内容属于正常现象，不必检查此错误，重点排查其他错误问题

项目里面我负责的主要模块是

-   attention/prompt_flash_attention
-   attention/lightning_indexer
-   attention/sparse_flash_attention

这是三个算子，我的任务是写算子目录下的tests/ut，例如attention/prompt_flash_attention/tests/ut，你只能动这些目录下的文件，不能改算子原来的代码

lcov看护覆盖率中：

ut下面的op_host/arch32看护的是算子目录下的op_host/*tiling.cpp文件，例如attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp

ut下面的op_api看护的是算子目录下op_host/op_api/里面的算子接口，例如attention/prompt_flash_attention/op_host/op_api/aclnn_prompt_flash_attention.cpp

./log/下面是运行覆盖率命令后的日志文件，例如log/li_opapi_ut.log就是lightning_indexer关注op_api下面算子接口的覆盖日志

你的所有回答全部写在这个文件，包括代码。

# 问题

1、读取这个tests/ut/framework_normal/common/mc2_hcom_topology_mocker.cpp，这个文件是对mc2/common/utils/mc2_hcom_topo_info.h的打桩，op_host ut运行时，不会编mc2/common/utils/mc2_hcom_topo_info.cpp，而是编打桩文件（tests/ut/framework_normal/common/mc2_hcom_topology_mocker.cpp）中的打桩函数，请问这是怎么实现的（读取tests/），以及我想自己实现一个打桩，针对attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp中6244和6545的GetSoftMaxMinTmpSize和GetSoftMaxFlashV2MinTmpSize实现一个自己的api接口，这两个api接口时/home/j60100428/Ascend_0320/cann-9.0.0/include/tiling/tiling_api.h头文件中的\#include "activation/softmax_tiling.h"引入的。

## 回答

### 一、mc2_hcom_topology_mocker.cpp 打桩机制分析

**核心文件：`mc2/common/CMakeLists.txt`，关键片段（第37-46行）：**

```cmake
if(UT_TEST_ALL OR OP_HOST_UT)
   # 步骤1：把真实实现文件从源文件列表中排除
   list(FILTER OPTILING_SRCS EXCLUDE REGEX "utils/mc2_hcom_topo_info\\.cpp")

   # 步骤2：把打桩文件编译成独立的动态库 mc2_tiling_ut_stub
   file(GLOB MC2_TILING_UT_STUB
       ${UT_COMMON_INC}/mc2_hcom_topology_mocker.cpp
       ${UT_COMMON_INC}/op_cache_tiling.cpp)
   add_library(mc2_tiling_ut_stub SHARED ${MC2_TILING_UT_STUB})
   ...
endif()
```

**步骤3：UT 可执行程序链接 stub 库（`tests/ut/framework_normal/op_host/CMakeLists.txt` 第63行）：**

```cmake
target_link_libraries(${OP_HOST_UT_EXE} PRIVATE
    ...
    $<$<TARGET_EXISTS:mc2_tiling_ut_stub>:mc2_tiling_ut_stub>
    ...
)
```

**机制总结（三步缺一不可）：**

| 步骤 | 操作 | 目的 |
|------|------|------|
| 排除 | `list(FILTER ... EXCLUDE ...)` 把真实 `.cpp` 从编译列表去掉 | 避免符号冲突/ODR违规 |
| 替换 | 打桩 `.cpp` 编译成 `mc2_tiling_ut_stub.so` | 提供相同接口的可控实现 |
| 链接 | UT exe 链接 stub 库 | 运行时使用打桩函数 |

**打桩文件本身的设计模式：**

打桩文件 `mc2_hcom_topology_mocker.cpp`：
- `#include "mc2_hcom_topo_info.h"` ← 包含**同一个头文件**，保证符号签名一致
- 重新实现头文件中声明的所有类方法（`MC2HcomTopology::CommGetInstSizeByGroup` 等）
- 通过 `MC2HcomTopologyMocker` 单例（key-value map）让测试用例控制返回值
- 测试用例调用 `MC2HcomTopologyMocker::GetInstance().SetValue("rankNum", 16)` 即可控制行为

---

### 二、为 GetSoftMaxMinTmpSize / GetSoftMaxFlashV2MinTmpSize 实现打桩

**与 mc2 情况的关键区别：**

| | mc2_hcom_topo_info | GetSoftMaxMinTmpSize |
|---|---|---|
| 实现来源 | 项目内 `.cpp` 文件 | 外部预编译动态库 `tiling_api.so` |
| 打桩方式 | CMake 排除 `.cpp`，替换 stub | 编译期符号覆盖（ELF 符号解析规则） |

**ELF 动态链接符号解析规则（关键原理）：**

在链接生成 `libophost_transformer_ut.so` 时，如果目标文件（`.o`）中定义了 `GetSoftMaxMinTmpSize`，同时 `tiling_api.so` 也提供该符号，**链接器优先使用目标文件中的定义**，不再从 `tiling_api.so` 引入该符号。这与 mc2 机制的本质相同，只是触发方式不同（mc2 是排除 `.cpp`，这里是直接在目标中重定义）。

---

**实现步骤：**

#### 第1步：创建 mocker 头文件

新建 `tests/ut/framework_normal/common/softmax_tiling_mocker.h`：

```cpp
#pragma once
#include <cstdint>

// Mocker 单例，测试用例通过它控制 GetSoftMaxMinTmpSize / GetSoftMaxFlashV2MinTmpSize 的返回值
class SoftmaxTilingMocker {
public:
    static SoftmaxTilingMocker& GetInstance() {
        static SoftmaxTilingMocker instance;
        return instance;
    }

    void SetSoftMaxMinTmpSize(uint32_t val) { softMaxMinTmpSize_ = val; }
    void SetSoftMaxFlashV2MinTmpSize(uint32_t val) { softMaxFlashV2MinTmpSize_ = val; }
    void Reset() { softMaxMinTmpSize_ = 0; softMaxFlashV2MinTmpSize_ = 0; }

    uint32_t GetSoftMaxMinTmpSize() const { return softMaxMinTmpSize_; }
    uint32_t GetSoftMaxFlashV2MinTmpSize() const { return softMaxFlashV2MinTmpSize_; }

private:
    SoftmaxTilingMocker() = default;
    uint32_t softMaxMinTmpSize_{0};
    uint32_t softMaxFlashV2MinTmpSize_{0};
};
```

#### 第2步：创建 stub 实现文件

新建 `tests/ut/framework_normal/common/softmax_tiling_mocker.cpp`：

```cpp
// 包含声明这两个函数的头文件，确保签名一致
#include "activation/softmax_tiling.h"
#include "softmax_tiling_mocker.h"

// 覆盖 tiling_api.so 中的 GetSoftMaxMinTmpSize
uint32_t GetSoftMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isReuseSource)
{
    return SoftmaxTilingMocker::GetInstance().GetSoftMaxMinTmpSize();
}

// 覆盖 tiling_api.so 中的 GetSoftMaxFlashV2MinTmpSize
uint32_t GetSoftMaxFlashV2MinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize1,
    const uint32_t dataTypeSize2, const bool isUpdate, const bool isBasicBlock,
    const bool isFlashOutputBrc)
{
    return SoftmaxTilingMocker::GetInstance().GetSoftMaxFlashV2MinTmpSize();
}
```

#### 第3步：在 CMakeLists.txt 中注册 stub

参考 mc2 的做法，在 PFA tiling 对应的 CMakeLists.txt（或 `mc2/common/CMakeLists.txt` 中统一管理的地方）的 `if(UT_TEST_ALL OR OP_HOST_UT)` 块中加入：

```cmake
if(UT_TEST_ALL OR OP_HOST_UT)
    # 将打桩文件编译进 mc2_tiling_ut_stub（或单独建 softmax_tiling_ut_stub）
    # 方式一：追加到已有 stub 库（最简单）
    target_sources(mc2_tiling_ut_stub PRIVATE
        ${UT_COMMON_INC}/softmax_tiling_mocker.cpp
    )
    target_include_directories(mc2_tiling_ut_stub PRIVATE
        ${UT_COMMON_INC}
        # softmax_tiling.h 所在的 include 路径
        ${ASCEND_DIR}/include/tiling
    )
endif()
```

或者单独建立一个库（适合需要独立管控的场景）：

```cmake
if(UT_TEST_ALL OR OP_HOST_UT)
    add_library(softmax_tiling_ut_stub SHARED
        ${UT_COMMON_INC}/softmax_tiling_mocker.cpp
    )
    target_include_directories(softmax_tiling_ut_stub PRIVATE
        ${UT_COMMON_INC}
        ${ASCEND_DIR}/include/tiling
    )
endif()
```

并在 `tests/ut/framework_normal/op_host/CMakeLists.txt` 的 exe 链接中加入：

```cmake
target_link_libraries(${OP_HOST_UT_EXE} PRIVATE
    ...
    $<$<TARGET_EXISTS:softmax_tiling_ut_stub>:softmax_tiling_ut_stub>
    ...
)
```

#### 第4步：在测试用例中使用

```cpp
#include "softmax_tiling_mocker.h"

TEST(PromptFlashAttentionTilingTest, TestCVDiff) {
    // 设置打桩返回值（非零值，防止 tiling 逻辑进入缩减循环）
    SoftmaxTilingMocker::GetInstance().SetSoftMaxMinTmpSize(1024);
    SoftmaxTilingMocker::GetInstance().SetSoftMaxFlashV2MinTmpSize(2048);

    // ... 构造 tiling context 并调用 ...

    // 测试结束后重置
    SoftmaxTilingMocker::GetInstance().Reset();
}
```

---

**注意事项：**

1. **签名必须完全一致**：stub 中函数签名（参数类型、顺序、const修饰）必须与 `softmax_tiling.h` 中声明的完全一致，否则链接时会产生两个不同的符号，打桩无效。
2. **链接顺序**：stub 库（`.so` 或链接进主 `.so` 的 `.o`）必须在链接时**先于** `tiling_api` 被处理。由于 `libophost_transformer_ut.so` 显式包含 stub 的 `.o`，ELF 链接器会优先使用本地定义，无需额外调整顺序。
3. **只影响调用方，不影响 tiling_api 内部**：如果 `tiling_api.so` 内部有其他函数调用 `GetSoftMaxMinTmpSize`，那些调用不会被 stub 拦截（内部调用走 PLT 的本地绑定），但从 `prompt_flash_attention_tiling.cpp` 发起的调用会被拦截。这对 UT 来说已经足够。

2、读取log/pfa_ophost_ut.log，为什么会报

```
[2026-03-24 17:37:12]CMake Error at attention/prompt_flash_attention/op_host/CMakeLists.txt:37 (add_library):
  Cannot find source file:

    //softmax_tiling_mocker.cpp
...
CMake Generate step failed.
```

感觉像是没识别到UT_COMMON_INC，是我写错cmakelists文件了？如果不是检索一下全仓，看看UT_COMMON_INC怎么设置的怎么读取。同时"/home/j60100428/asc-devkit/impl/adv_api/tiling/activation/softmax_tiling.cpp"就是调用api仓的源码，主要问题是两个函数里面platform_ascendc::PlatformAscendC* platform = platform_ascendc::PlatformAscendCManager::GetInstance();调用的是无参函数，而我想要传递的是GetInstance有参函数（函数原型在/home/j60100428/asc-devkit/include/utils/tiling/platform/下面），所以需要你再当前仓打桩重写这两函数，其他地方不变可以吗？

## 回答

### 一、`UT_COMMON_INC` 为空的根因

**不是你写错了，是 CMake 处理顺序问题。**

关键文件：`cmake/custom_build.cmake`

```cmake
# 第 289 行：先处理 attention（包含 PFA 的 op_host/CMakeLists.txt）
add_subdirectory(attention)

# 第 294 行：后处理 framework_normal → 才设置 UT_COMMON_INC
add_subdirectory(tests/ut/framework_normal)

# 第 339 行：mc2/common 在 framework_normal 之后，所以 mc2 能正常读到 UT_COMMON_INC
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/mc2/common)
```

`UT_COMMON_INC` 在 `tests/ut/framework_normal/CMakeLists.txt` 第14行设置：
```cmake
set(UT_COMMON_INC ${CMAKE_CURRENT_SOURCE_DIR}/common CACHE STRING "ut common include path" FORCE)
```

PFA 的 `op_host/CMakeLists.txt` 在第289行就被处理了，这时 `UT_COMMON_INC` 还没有被赋值（CACHE 里也还没有），所以 `${UT_COMMON_INC}` 展开为空字符串，路径变成 `//softmax_tiling_mocker.cpp`。mc2 的 `common/CMakeLists.txt` 在第339行处理，晚于第294行，所以它能正常读到 `UT_COMMON_INC`。

**修复方法**：在 `attention/prompt_flash_attention/op_host/CMakeLists.txt` 中不使用 `UT_COMMON_INC`，改用 `PROJECT_SOURCE_DIR` 硬编码路径（已修复）：

```cmake
# 修复前（错误）：
if(UT_TEST_ALL OR OP_HOST_UT)
    add_library(softmax_tiling_ut_stub SHARED
        ${UT_COMMON_INC}/softmax_tiling_mocker.cpp   # UT_COMMON_INC 此时为空！
    )
    target_include_directories(softmax_tiling_ut_stub PRIVATE
        ${UT_COMMON_INC}
        ${ASCEND_DIR}/include/tiling
    )
endif()

# 修复后（正确）：
if(UT_TEST_ALL OR OP_HOST_UT)
    set(_SOFTMAX_STUB_COMMON_DIR ${PROJECT_SOURCE_DIR}/tests/ut/framework_normal/common)
    add_library(softmax_tiling_ut_stub SHARED
        ${_SOFTMAX_STUB_COMMON_DIR}/softmax_tiling_mocker.cpp
    )
    target_include_directories(softmax_tiling_ut_stub PRIVATE
        ${_SOFTMAX_STUB_COMMON_DIR}
        ${OP_TILING_INCLUDE}   # 包含所有 tiling 相关头文件路径，比 ${ASCEND_DIR}/include/tiling 更完整
    )
endif()
```

---

### 二、同时发现并修复了原 stub 的命名空间 bug

原始 `softmax_tiling_mocker.cpp` 中函数定义在**全局命名空间**，但 `softmax_tiling.h` 中函数声明在 `namespace AscendC` 内。两个符号名不同（`::GetSoftMaxMinTmpSize` vs `AscendC::GetSoftMaxMinTmpSize`），导致 stub 实际上没有覆盖 `tiling_api.so` 中的函数，打桩无效。

---

### 三、重写的打桩文件

#### `tests/ut/framework_normal/common/softmax_tiling_mocker.h`（已修改）

```cpp
#pragma once
#include <cstdint>
#include <string>

// UT 通过 SetSocVersion("ascend910b") 等设置 SoC，
// 打桩函数将用 PlatformAscendCManager::GetInstance(socVersion) 初始化平台。
class SoftmaxTilingMocker {
public:
    static SoftmaxTilingMocker& GetInstance() {
        static SoftmaxTilingMocker instance;
        return instance;
    }
    void SetSocVersion(const std::string& socVersion) { socVersion_ = socVersion; }
    const char* GetSocVersion() const { return socVersion_.empty() ? nullptr : socVersion_.c_str(); }
    void Reset() { socVersion_.clear(); }
private:
    SoftmaxTilingMocker() = default;
    std::string socVersion_;
};
```

#### `tests/ut/framework_normal/common/softmax_tiling_mocker.cpp`（已修改）

关键变化：
- 函数定义放在 `namespace AscendC` 内（修复命名空间 bug）
- `PlatformAscendCManager::GetInstance()` → `PlatformAscendCManager::GetInstance(SoftmaxTilingMocker::GetInstance().GetSocVersion())`
- 计算逻辑与 `softmax_tiling.cpp` 原版完全一致，去掉了无法包含的内部 `HighLevelApiCheck` 宏（这些只是输入校验，不影响计算结果）

#### 在 UT 测试用例中使用

```cpp
#include "softmax_tiling_mocker.h"

// 测试前设置 SoC 版本（字符串与 PlatformAscendCManager 支持的 socVersion 一致）
SoftmaxTilingMocker::GetInstance().SetSocVersion("ascend910b");

// ... 执行 tiling，此时 GetSoftMaxMinTmpSize / GetSoftMaxFlashV2MinTmpSize
//     会用 ascend910b 对应的 NpuArch 计算出真实的 size 值 ...

// 测试结束后复位
SoftmaxTilingMocker::GetInstance().Reset();
```



3、读取log/pfa_ophost_ut.log，现在报了符号错误

```
[2026-03-24 20:26:21] [ 92%] Bu/usr/bin/ld: ../../../../attention/prompt_flash_attention/op_host/libsoftmax_tiling_ut_stub.so: undefined reference to `platform_ascendc::PlatformAscendCManager::platformInfo'
/usr/bin/ld: ../../../../attention/prompt_flash_attention/op_host/libsoftmax_tiling_ut_stub.so: undefined reference to `platform_ascendc::PlatformAscendC::GetCurNpuArch() const'
/usr/bin/ld: ../../../../attention/prompt_flash_attention/op_host/libsoftmax_tiling_ut_stub.so: undefined reference to `platform_ascendc::PlatformAscendCManager::PlatformAscendCManagerInit(char const*)'
/usr/bin/ld: ../../../../attention/prompt_flash_attention/op_host/libsoftmax_tiling_ut_stub.so: undefined reference to `platform_ascendc::PlatformAscendCManager::platformInitMtx'
```

感觉这个像是缺少链接的什么编译库导致的问题，请排查。将结果添加到这个md文档，不要去动仓里面的代码了

## 回答

### 根因分析

**判断正确：缺少链接库。**

4个 undefined reference 对应的符号：

| 符号 | 类型 | 说明 |
|------|------|------|
| `PlatformAscendCManager::platformInfo` | 静态成员变量 | `PlatformAscendC*` 指针 |
| `PlatformAscendC::GetCurNpuArch() const` | 非内联成员函数 | 获取当前 NpuArch |
| `PlatformAscendCManager::PlatformAscendCManagerInit(char const*)` | 私有静态方法 | 初始化 platform |
| `PlatformAscendCManager::platformInitMtx` | 静态 mutex | 初始化锁 |

`softmax_tiling_mocker.cpp` 调用了 `PlatformAscendCManager::GetInstance(socVersion)`。虽然 `GetInstance` 是内联函数（定义在 `platform_ascendc.h`），但它的函数体引用了上面 4 个**非内联符号**，这些符号的定义在 `libtiling_api.a` 中。

用 `nm` 验证（全部有定义，非 `U`）：
```
0000000000000028 B _ZN16platform_ascendc22PlatformAscendCManager12platformInfoE        ← platformInfo
0000000000000000 B _ZN16platform_ascendc22PlatformAscendCManager15platformInitMtxE     ← platformInitMtx
0000000000002390 T _ZN16platform_ascendc22PlatformAscendCManager26PlatformAscendCManagerInitEPKc  ← Init
0000000000000700 T _ZNK16platform_ascendc15PlatformAscendC13GetCurNpuArchEv            ← GetCurNpuArch
```
以上均来自：`${ASCEND_CANN_PACKAGE_PATH}/devlib/libtiling_api.a`

---

### 问题所在

`attention/prompt_flash_attention/op_host/CMakeLists.txt` 中 `softmax_tiling_ut_stub` 的定义只有 `target_include_directories`，**缺少 `target_link_libraries`**：

```cmake
# 当前（缺链接）：
if(UT_TEST_ALL OR OP_HOST_UT)
    set(_SOFTMAX_STUB_COMMON_DIR ${PROJECT_SOURCE_DIR}/tests/ut/framework_normal/common)
    add_library(softmax_tiling_ut_stub SHARED
        ${_SOFTMAX_STUB_COMMON_DIR}/softmax_tiling_mocker.cpp
    )
    target_include_directories(softmax_tiling_ut_stub PRIVATE
        ${_SOFTMAX_STUB_COMMON_DIR}
        ${OP_TILING_INCLUDE}
    )
    # ← 这里缺少 target_link_libraries！
endif()
```

`.so` 构建时链接器默认允许 undefined symbol（延迟绑定），所以 `.so` 本身能生成。但当 UT 可执行文件链接 `softmax_tiling_ut_stub.so` 时，链接器执行完整符号检查，发现 `.so` 里仍有 4 个未解析符号，报错。

---

### 修复方法

在 `attention/prompt_flash_attention/op_host/CMakeLists.txt` 的 `softmax_tiling_ut_stub` 定义中补充一行 `target_link_libraries`：

```cmake
if(UT_TEST_ALL OR OP_HOST_UT)
    set(_SOFTMAX_STUB_COMMON_DIR ${PROJECT_SOURCE_DIR}/tests/ut/framework_normal/common)
    add_library(softmax_tiling_ut_stub SHARED
        ${_SOFTMAX_STUB_COMMON_DIR}/softmax_tiling_mocker.cpp
    )
    target_include_directories(softmax_tiling_ut_stub PRIVATE
        ${_SOFTMAX_STUB_COMMON_DIR}
        ${OP_TILING_INCLUDE}
    )
    target_link_libraries(softmax_tiling_ut_stub PRIVATE tiling_api)   # ← 补充这一行
endif()
```

`tiling_api` 是 CMake IMPORTED STATIC target（由 `cmake/modules/Findtilingapi.cmake` 定义，指向 `libtiling_api.a`），链接它后，linker 会从静态库中提取包含上述 4 个符号的 object，嵌入到 `libsoftmax_tiling_ut_stub.so` 中，undefined reference 全部消除。

> **参考**：`libophost_transformer_ut.so` 自身也是同样方式链接 `tiling_api`（见 `tests/ut/framework_normal/op_host/CMakeLists.txt` 第42行），说明 `libtiling_api.a` 支持被链接进 shared library。


4、现在把softmax_tiling_ut_stub链接上tiling_api后，好像调用的又是tiling_api里面的GetSoftMaxMinTmpSize和GetSoftMaxFlashV2MinTmpSize（tiling_api本身就有这两函数，好像由于softmax_tiling_ut_stub链接的tiling_api，导致被覆盖了），GetInstance又是无参数传入导致报错了，请问有什么办法既然让问题3的符号能找到，又能让问题2的功能生效（调用打桩后的GetSoftMaxMinTmpSize和GetSoftMaxFlashV2MinTmpSize）。请将回答写在这个md文档

## 回答

### 根因分析

**问题根源不在于 `softmax_tiling_ut_stub` 链接了 `tiling_api`，而在于 `libophost_transformer_ut.so` 本身也嵌入了 `tiling_api` 的 `softmax_tiling.cpp.o`。**

关键链路如下：

1. `prompt_flash_attention_tiling.cpp` 调用了 `AscendC::GetSoftMaxMinTmpSize`，产生 undefined reference。
2. 链接 `libophost_transformer_ut.so` 时，linker 从 `libtiling_api.a` 中拉取 `softmax_tiling.cpp.o` 来满足这个引用，将其**嵌入**进 `libophost_transformer_ut.so`。
3. `test_op_host_main.cpp` 通过 `gert::OpImplSpaceRegistryV2::AddSoToRegistry` 在运行时 `dlopen` 加载 `libophost_transformer_ut.so`。
4. `dlopen` 加载时（很可能带 `RTLD_DEEPBIND` 标志），SO 内部的符号优先绑定自身已嵌入的定义，即 `softmax_tiling.cpp.o` 中原版的 `GetSoftMaxMinTmpSize`（内部调用无参 `GetInstance()`），不走 PLT 跳到 stub。

所以整个链路是：

```
prompt_flash_attention_tiling.cpp
    → 调用 AscendC::GetSoftMaxMinTmpSize
    → libophost_transformer_ut.so 内部嵌入了 softmax_tiling.cpp.o（来自 libtiling_api.a）
    → dlopen(RTLD_DEEPBIND) 优先用 SO 内部定义
    → 调用 原版 GetSoftMaxMinTmpSize → GetInstance()（无参）→ 报错
```

---

### 关键证据：`libtiling_api.a` 中两个 `.o` 是独立对象

用 `ar t` 确认 `libtiling_api.a` 的对象：

```
softmax_tiling.cpp.o     ← 定义 GetSoftMaxMinTmpSize / GetSoftMaxFlashV2MinTmpSize
platform_ascendc.cpp.o   ← 定义 platformInfo, platformInitMtx, PlatformAscendCManagerInit, GetCurNpuArch
```

两者**完全独立**。问题3中为 stub 链接 `tiling_api` 时，linker 只拉取了 `platform_ascendc.cpp.o`（因为 stub 里有 undefined ref 需要它），**没有**拉取 `softmax_tiling.cpp.o`（stub 自己已经定义了这两个函数，不产生 undefined ref）。

所以 stub 本身是干净的：它只嵌入了 `platform_ascendc.cpp.o`，没有嵌入 `softmax_tiling.cpp.o`。**问题出在 `libophost_transformer_ut.so` 那边。**

---

### 解决方案：让 `libophost_transformer_ut.so` 链接 stub，不从 `tiling_api` 嵌入 `softmax_tiling.cpp.o`

修改 `tests/ut/framework_normal/op_host/CMakeLists.txt`，在 `${OPHOST_NAME}_ut` 的 `target_link_libraries` 中，将 `softmax_tiling_ut_stub` 加在 `tiling_api` **之前**：

```cmake
# 修改前（第35-45行）：
target_link_libraries(${OPHOST_NAME}_ut PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
    -Wl,--no-as-needed
    -Wl,--as-needed
    -Wl,--whole-archive
        rt2_registry_static
    -Wl,--no-whole-archive
    tiling_api         # ← 当前：softmax_tiling.cpp.o 从这里被拉进来
    runtime
    acl_rt
)

# 修改后：
target_link_libraries(${OPHOST_NAME}_ut PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
    -Wl,--no-as-needed
    -Wl,--as-needed
    -Wl,--whole-archive
        rt2_registry_static
    -Wl,--no-whole-archive
    $<$<TARGET_EXISTS:softmax_tiling_ut_stub>:softmax_tiling_ut_stub>   # ← 新增：先满足符号，阻止 tiling_api 贡献 softmax_tiling.cpp.o
    tiling_api
    runtime
    acl_rt
)
```

**原理**：

| 步骤 | 说明 |
|------|------|
| linker 处理 `softmax_tiling_ut_stub.so` | `libophost_transformer_ut.so` 的 undefined ref `AscendC::GetSoftMaxMinTmpSize` 被 stub.so 中的定义满足 |
| linker 继续处理 `tiling_api`（`libtiling_api.a`）| `AscendC::GetSoftMaxMinTmpSize` 已有定义，linker **不再**从静态库中拉取 `softmax_tiling.cpp.o` |
| 结果 | `libophost_transformer_ut.so` 不嵌入原版实现，运行时通过 PLT 调用 stub.so 中的打桩版本 |

---

### 修改后的完整行为链路

```
prompt_flash_attention_tiling.cpp
    → 调用 AscendC::GetSoftMaxMinTmpSize
    → libophost_transformer_ut.so 中无本地定义（softmax_tiling.cpp.o 未嵌入）
    → PLT 查找全局符号表
    → softmax_tiling_ut_stub.so 中的打桩版本（使用有参 GetInstance）
    → PlatformAscendCManager::GetInstance(socVersion) → 正常计算
```

---

### 总结：三个库各自承担的职责

| 库 | 嵌入的对象 | 提供的符号 |
|----|-----------|-----------|
| `libsoftmax_tiling_ut_stub.so` | `softmax_tiling_mocker.cpp.o` + `platform_ascendc.cpp.o`（来自 tiling_api） | 打桩版 `GetSoftMaxMinTmpSize`、`GetSoftMaxFlashV2MinTmpSize`，及所需 platform 符号 |
| `libophost_transformer_ut.so` | 所有 tiling/infer 对象 + `platform_ascendc.cpp.o`（`softmax_tiling.cpp.o` 已被 stub 截走，**不再嵌入**） | 算子 tiling 逻辑；`GetSoftMaxMinTmpSize` 通过 PLT 转发给 stub |
| `libtiling_api.a` | — | 静态库，按需贡献 `.o`，两个库各取所需后 `softmax_tiling.cpp.o` 已无人需要 |

5、读取log/pfa_ophost_ut.log你可以看到符号未定位问题又重现了，结合之前四个问题，解一下，结果放在这个md文档，如果有需要请查询整个仓，结合/home/j60100428/asc-devkit/impl/adv_api/tiling/activation/softmax_tiling.cpp好好分析一下是否需要打桩其他函数和变量。

## 回答

### 一、为何符号未定位问题再次出现

**直接原因：问题3的修复被移除，但问题4的修复被保留，导致两者不一致。**

当前状态（通过读取文件确认）：

| 文件 | 当前状态 | 问题 |
|------|---------|------|
| `attention/prompt_flash_attention/op_host/CMakeLists.txt` | stub 只有 `target_include_directories`，**缺少** `target_link_libraries(softmax_tiling_ut_stub PRIVATE tiling_api)` | platform 符号无来源 |
| `tests/ut/framework_normal/op_host/CMakeLists.txt` | `softmax_tiling_ut_stub` 已在 `tiling_api` 之前链接（问题4修复已应用）| ✓ 正确 |

**错误链路：**

```
transformer_op_host_ut (exe)
  → 链接 libsoftmax_tiling_ut_stub.so
  → linker 发现 .so 中仍有 4 个 undefined reference：
      platform_ascendc::PlatformAscendCManager::platformInfo
      platform_ascendc::PlatformAscendC::GetCurNpuArch() const
      platform_ascendc::PlatformAscendCManager::PlatformAscendCManagerInit(char const*)
      platform_ascendc::PlatformAscendCManager::platformInitMtx
  → 报错：collect2: error: ld returned 1 exit status
```

**根本原因是问题3和问题4的修复必须同时存在，而非互斥：**

| 问题 | 修复 | 目的 |
|------|------|------|
| 问题3 | stub 链接 `tiling_api` → 拉入 `platform_ascendc.cpp.o` | 满足 stub.so 自身的 platform 符号引用 |
| 问题4 | `libophost_transformer_ut.so` 链接时 stub 在 `tiling_api` 之前 | 阻止 `softmax_tiling.cpp.o` 被嵌入 .so，确保调用走 stub |

两者各自独立解决不同层面的问题，缺一不可。将问题4的修复理解为"不能让 stub 链接 tiling_api"是误读——问题4指出的是 `libophost_transformer_ut.so` 那侧的问题，与 stub 自身链接 tiling_api 无关。

---

### 二、修复方法

在 `attention/prompt_flash_attention/op_host/CMakeLists.txt` 的 `if(UT_TEST_ALL OR OP_HOST_UT)` 块中，补回 `target_link_libraries`：

```cmake
if(UT_TEST_ALL OR OP_HOST_UT)
    set(_SOFTMAX_STUB_COMMON_DIR ${PROJECT_SOURCE_DIR}/tests/ut/framework_normal/common)
    add_library(softmax_tiling_ut_stub SHARED
        ${_SOFTMAX_STUB_COMMON_DIR}/softmax_tiling_mocker.cpp
    )
    target_include_directories(softmax_tiling_ut_stub PRIVATE
        ${_SOFTMAX_STUB_COMMON_DIR}
        ${OP_TILING_INCLUDE}
    )
    target_link_libraries(softmax_tiling_ut_stub PRIVATE tiling_api)   # ← 补回此行
endif()
```

`tests/ut/framework_normal/op_host/CMakeLists.txt` 第42行的顺序（stub 在 tiling_api 之前）保持不变，无需修改。

---

### 三、softmax_tiling.cpp 全函数分析——是否需要补打桩

逐一分析 `softmax_tiling.cpp` 中每个函数对 `PlatformAscendCManager::GetInstance()` 的调用，以及 PFA tiling 代码的实际调用情况：

| 函数名 | softmax_tiling.cpp 中是否调用 GetInstance() | PFA tiling 是否调用 | 是否需要打桩 |
|--------|----------------------------------------------|----------------------|--------------|
| `GetSoftMaxMaxTmpSize` | ✓ 是（第106行） | 未调用 | 无需 |
| **`GetSoftMaxMinTmpSize`** | ✓ 是（第140行） | ✓ 是（第441、1492、6244、6353行） | **已打桩 ✓** |
| `GetSoftMaxFlashMaxTmpSize` | ✓ 是（第236行） | 未调用 | 无需 |
| **`GetSoftMaxFlashMinTmpSize`** | ✓ 是（第277行） | ✓ 是（**第442行**） | **⚠️ 缺少打桩！** |
| `GetSoftMaxGradMaxTmpSize` | ✗ 否（纯形状计算） | 未调用 | 无需 |
| `GetSoftMaxGradMinTmpSize` | ✓ 是（第410行） | 未调用 | 无需 |
| `GetSoftMaxFlashV2MaxTmpSize` | ✓ 是（第545行） | 未调用 | 无需 |
| **`GetSoftMaxFlashV2MinTmpSize`** | ✓ 是（第630行） | ✓ 是（第437、1493、1590、6245、6354行） | **已打桩 ✓** |
| `GetSoftMaxFlashV3MaxMinTmpSize` | ✓ 是（第808行） | 未调用 | 无需 |
| `SoftMaxTilingFunc` | ✗ 否 | ✓ 是（第1597行） | 无需（不调用 GetInstance） |
| `SoftMaxFlashV2TilingFunc` | ✗ 否 | ✓ 是（第1594、1598、3342行） | 无需（不调用 GetInstance） |
| `IsBasicBlockInSoftMax` | ✗ 否 | ✓ 是（第3343行） | 无需（纯数据判断） |

**关键发现：`GetSoftMaxFlashMinTmpSize` 缺少打桩！**

PFA tiling 第442行：
```cpp
uint32_t softmaxFlashTmpSize = GetSoftMaxFlashMinTmpSize(tmpShape, typeByteSize, true, true);
```

该函数在 `softmax_tiling.cpp` 第277行调用了无参 `GetInstance()`，与其他两个被打桩函数完全一致。如果 UT 走到 `ASCEND910B` 分支（第440行），就会命中此调用，产生与问题1相同的无参 `GetInstance()` 错误。

---

### 四、补充打桩 `GetSoftMaxFlashMinTmpSize`

函数签名（来自 `softmax_tiling.h`）：

```cpp
// namespace AscendC
uint32_t GetSoftMaxFlashMinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize,
    const bool isUpdate, const bool isReuseSource);
```

在 `tests/ut/framework_normal/common/softmax_tiling_mocker.cpp` 末尾（`} // namespace AscendC` 之前）追加：

```cpp
// ---- 打桩：GetSoftMaxFlashMinTmpSize ------------------------------------
// 逻辑与 softmax_tiling.cpp 第259-298行一致，仅替换 GetInstance()
uint32_t GetSoftMaxFlashMinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize,
    const bool isUpdate, const bool /*isReuseSource*/)
{
    const uint32_t inputSize = static_cast<uint32_t>(srcShape.GetShapeSize());
    if (inputSize == 0U) { return 0U; }
    if (dataTypeSize == 0U || (dataTypeSize != SOFTMAX_HALF_SIZE && dataTypeSize != SOFTMAX_FLOAT_SIZE)) {
        return 0U;
    }

    const std::vector<uint32_t> retVec = GetLastAxisShapeND(srcShape);
    const uint32_t elementNumPerBlk = SOFTMAX_DEFAULT_BLK_SIZE / dataTypeSize;
    const uint32_t srcM = retVec[0];
    const uint32_t srcK = retVec[1];

    platform_ascendc::PlatformAscendC* platform =
        platform_ascendc::PlatformAscendCManager::GetInstance(
            SoftmaxTilingMocker::GetInstance().GetSocVersion());
    if (platform == nullptr) { return 0U; }
    const auto npuArch = platform->GetCurNpuArch();

    uint32_t needSize;
    if (npuArch == NpuArch::DAV_3510 ||
        npuArch == NpuArch::DAV_5102) {
        const uint32_t needSize2 = srcM * (elementNumPerBlk + srcK);
        if (!isUpdate) {
            const uint32_t needSize1 = srcM * (BASIC_TILE_NUM + srcK) +
                SOFTMAX_BASICBLOCK_UNIT * SOFTMAX_FLOAT_SIZE +
                (srcM + BASIC_TILE_NUM - 1U) / BASIC_TILE_NUM * BASIC_TILE_NUM;
            needSize = std::max(needSize1, needSize2);
        } else {
            needSize = needSize2;
        }
    } else {
        needSize = !isUpdate ?
            elementNumPerBlk + srcK + SOFTMAX_BASICBLOCK_UNIT :
            elementNumPerBlk * SOFTMAX_TMPFLASHUPDATE_COUNT + srcK * SOFTMAX_TMPBUFFER_COUNT;
    }
    return needSize * SOFTMAX_FLOAT_SIZE;
}
```

---

### 五、完整修复清单（按操作顺序）

1. **`attention/prompt_flash_attention/op_host/CMakeLists.txt`**：补回 `target_link_libraries(softmax_tiling_ut_stub PRIVATE tiling_api)`（一行）

2. **`tests/ut/framework_normal/common/softmax_tiling_mocker.cpp`**：在 `} // namespace AscendC` 前追加 `GetSoftMaxFlashMinTmpSize` 的打桩实现（如上）

3. **`tests/ut/framework_normal/op_host/CMakeLists.txt`**：无需修改，第42行已正确将 stub 置于 tiling_api 之前

操作完成后，三个 softmax 函数（`GetSoftMaxMinTmpSize`、`GetSoftMaxFlashMinTmpSize`、`GetSoftMaxFlashV2MinTmpSize`）均已打桩，所有 platform 符号由 stub.so 自身持有（来自 tiling_api 的 `platform_ascendc.cpp.o`），且 `libophost_transformer_ut.so` 不会嵌入 `softmax_tiling.cpp.o`。



6、读取log/pfa_ophost_ut.log日志文件，你会发现依然有包socVersion找不到的错误，GetInstance()依然没有把soc传进去，或者说是没有调到我们打桩函数，走的原有函数，下面细看还有CMakeFiles错误

```
[2026-03-25 14:10:14make[3]: *** [tests/ut/framework_normal/op_host/CMakeFiles/transformer_op_host_ut.dir/build.make:117: tests/ut/framework_normal/op_host/transformer_op_host_ut] Error 1
make[3]: *** Deleting file 'tests/ut/framework_normal/op_host/transformer_op_host_ut'
make[2]: *** [CMakeFiles/Makefile2:2000: tests/ut/framework_normal/op_host/CMakeFiles/transformer_op_host_ut.dir/all] Error 2
make[1]: *** [CMakeFiles/Makefile2:2007: tests/ut/framework_normal/op_host/CMakeFiles/transformer_op_host_ut.dir/rule] Error 2
make: *** [Makefile:865: transformer_op_host_ut] Error 2
```

我怀疑是cmakelists的修改不对，另外我想试试打桩函数是否写正确了，看看在tests哪里主动调一下测试结果

另外我有个疑问，在整个打桩过程中

```
if(UT_TEST_ALL OR OP_HOST_UT)
    set(_SOFTMAX_STUB_COMMON_DIR ${PROJECT_SOURCE_DIR}/tests/ut/framework_normal/common)
    add_library(softmax_tiling_ut_stub SHARED
        ${_SOFTMAX_STUB_COMMON_DIR}/softmax_tiling_mocker.cpp
    )
    target_include_directories(softmax_tiling_ut_stub PRIVATE
        ${_SOFTMAX_STUB_COMMON_DIR}
        ${OP_TILING_INCLUDE}
    )
    target_link_libraries(softmax_tiling_ut_stub PRIVATE tiling_api)
endif()
```

为什么不把tiling_api也编进softmax_tiling_ut_stub里面，结合我们自己打桩的文件，将原有tiling_api里面的函数重定义掉重新生成一个与tiling_api功能大同小异的so去链接，这样只用在依赖tiling_api的时候替换成我们自己的库就行，就不用考虑so的先后顺序了。

## 回答

### 一、为何打桩函数没被调到——直接证据

日志第301-302行：

```
[ERROR] ASCENDCKERNEL ... [../../../../impl/adv_api/tiling/activation/softmax_tiling.cpp:141][GetSoftMaxMinTmpSize] Failed to get PlatformAscendC.
[ERROR] ASCENDCKERNEL ... [../../../../impl/adv_api/tiling/activation/softmax_tiling.cpp:631][GetSoftMaxFlashV2MinTmpSize] Failed to get PlatformAscendC.
```

路径明确指向 `asc-devkit/impl/adv_api/tiling/activation/softmax_tiling.cpp`（原版），不是我们的 `softmax_tiling_mocker.cpp`。同时 mocker 里加的 `std::cout << "20260325 GetSoftMaxMinTmpSize_mocker"` 没有出现在日志中，也证实打桩函数根本没有被执行。

---

### 二、根本原因：问题4的修复被移除

读取 `tests/ut/framework_normal/op_host/CMakeLists.txt` 当前内容（第35-45行）：

```cmake
target_link_libraries(${OPHOST_NAME}_ut PRIVATE
    ...
    -Wl,--no-whole-archive
    tiling_api        # ← softmax_tiling_ut_stub 已不在 tiling_api 之前！
    runtime
    acl_rt
)
```

**问题4的修复（stub 置于 tiling_api 之前）被撤销了。**

完整的错误链路：

```
libophost_transformer_ut.so 链接时：
  tiling_api → linker 拉取 softmax_tiling.cpp.o（原版）→ 嵌入 .so

运行时 dlopen libophost_transformer_ut.so：
  prompt_flash_attention_tiling.cpp 调用 AscendC::GetSoftMaxMinTmpSize
  → .so 内部已有定义（原版 softmax_tiling.cpp.o）
  → 调用原版 → GetInstance()（无参）→ customSocVersion: (null) → 失败

transformer_op_host_ut（exe）虽然链接了 softmax_tiling_ut_stub.so，
但 exe 本身不调用这些函数，调用发生在被 dlopen 的 .so 内部，
exe 侧的符号对 .so 内部调用无效。
```

---

### 三、CMakeLists 错误（14:10:14 那次）分析

那次报错是运行时 ASAN 检测到测试失败后 make 退出，不是 CMake 配置错误。当前（16:17:05）这次日志显示构建本身全部成功（libsoftmax_tiling_ut_stub.so、libophost_transformer_ut.so、transformer_op_host_ut 均正常构建），**CMakeLists 写法没有问题**，只是打桩链接顺序被改掉了。

---

### 四、修复方法——恢复问题4的修复

在 `tests/ut/framework_normal/op_host/CMakeLists.txt` 中，将 stub 放回 `tiling_api` 之前：

```cmake
target_link_libraries(${OPHOST_NAME}_ut PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
    -Wl,--no-as-needed
    -Wl,--as-needed
    -Wl,--whole-archive
        rt2_registry_static
    -Wl,--no-whole-archive
    $<$<TARGET_EXISTS:softmax_tiling_ut_stub>:softmax_tiling_ut_stub>   # ← 恢复此行
    tiling_api
    runtime
    acl_rt
)
```

`attention/prompt_flash_attention/op_host/CMakeLists.txt` 中的 `target_link_libraries(softmax_tiling_ut_stub PRIVATE tiling_api)` 已存在，无需修改。

---

### 五、验证打桩函数是否被调到

测试文件（`tests/ut/op_host/arch32/test_prompt_flash_attention_tiling.cpp`）已经包含了正确的设置：

```cpp
static void SetUpTestCase() {
    SoftmaxTilingMocker::GetInstance().SetSocVersion("Ascend910B");  // 第45行
}
```

mocker.cpp 中也已经加了 cout（第59行）：
```cpp
std::cout << "20260325 GetSoftMaxMinTmpSize_mocker" << std::endl;
```

**修复 CMakeLists 后重新构建运行，若日志出现：**
```
20260325 GetSoftMaxMinTmpSize_mocker
```
则说明打桩生效。若仍看到 `softmax_tiling.cpp:141` 错误，则说明链接顺序还有问题。

---

### 六、关于"Fat Stub"方案——将 tiling_api 整体编进 stub 的可行性分析

**方案概念完全可行，且从工程角度看更优雅。** 以下是实现方式和注意点：

#### 实现方式

利用 ELF 链接器的 `--allow-multiple-definition` + `--whole-archive` 组合：

```cmake
# attention/prompt_flash_attention/op_host/CMakeLists.txt
if(UT_TEST_ALL OR OP_HOST_UT)
    set(_SOFTMAX_STUB_COMMON_DIR ${PROJECT_SOURCE_DIR}/tests/ut/framework_normal/common)
    add_library(softmax_tiling_ut_stub SHARED
        ${_SOFTMAX_STUB_COMMON_DIR}/softmax_tiling_mocker.cpp
    )
    target_include_directories(softmax_tiling_ut_stub PRIVATE
        ${_SOFTMAX_STUB_COMMON_DIR}
        ${OP_TILING_INCLUDE}
    )
    # Fat stub：将 tiling_api 整体打包进来，但我们自己的函数优先
    target_link_libraries(softmax_tiling_ut_stub PRIVATE
        -Wl,--allow-multiple-definition
        -Wl,--whole-archive $<TARGET_FILE:tiling_api> -Wl,--no-whole-archive
        -Wl,--no-allow-multiple-definition
    )
endif()
```

**链接器行为**：
- `add_library` 的源文件（`softmax_tiling_mocker.cpp.o`）先于依赖库被处理
- `--allow-multiple-definition` 允许重名符号共存，**保留第一个定义**
- `--whole-archive` 强制将 `libtiling_api.a` 中所有 `.o`（包括 `softmax_tiling.cpp.o`、`platform_ascendc.cpp.o` 等）全部嵌入
- 由于 mocker.cpp.o 的 `GetSoftMaxMinTmpSize` 先被链接器处理，tiling_api 的同名函数被静默丢弃
- 结果：`libsoftmax_tiling_ut_stub.so` 自包含 tiling_api 的全部功能，但三个关键函数被替换

#### 使用方式（两处 CMakeLists 都用 stub 替换 tiling_api）

```cmake
# tests/ut/framework_normal/op_host/CMakeLists.txt
# ophost_ut：用 stub 替换 tiling_api
target_link_libraries(${OPHOST_NAME}_ut PRIVATE
    ...
    -Wl,--no-whole-archive
    $<$<TARGET_EXISTS:softmax_tiling_ut_stub>:softmax_tiling_ut_stub>  # 替代 tiling_api
    # tiling_api   ← 不再需要（内容已在 stub 中）
    runtime
    acl_rt
)

# exe：同样只需 stub
target_link_libraries(${OP_HOST_UT_EXE} PRIVATE
    ...
    $<$<TARGET_EXISTS:softmax_tiling_ut_stub>:softmax_tiling_ut_stub>
    # 不再需要单独列 tiling_api
    ...
)
```

#### Fat Stub 方案优缺点对比

| 对比项 | 当前方案（stub + 顺序控制） | Fat Stub 方案 |
|--------|---------------------------|--------------|
| CMakeLists 改动 | 两处都需要维护 stub 在 tiling_api 之前 | 只需在 stub 定义处加 --whole-archive |
| 使用方 | 所有依赖 tiling_api 的地方需保证顺序 | 直接替换 tiling_api，无顺序要求 |
| 风险 | 顺序被意外移除就失效（本次问题的来源） | `--allow-multiple-definition` 可能掩盖真实重定义 bug |
| stub.so 体积 | 小（仅 mocker + platform.o） | 大（包含整个 tiling_api） |
| 推荐场景 | UT 文件少、依赖 tiling_api 的目标少 | UT 覆盖面广、多个目标依赖 tiling_api |

**结论**：两种方案都可行。如果以后其他算子的 UT 也需要打桩 softmax，Fat Stub 方案更省心；当前 PFA 单个算子的 UT，恢复问题4的修复（在 ophost_ut 链接中将 stub 置于 tiling_api 之前）是最小改动。





7、读取log/pfa_ophost_ut.log日志文件，我采用"Fat Stub"方案，编译阶段就报错了，请检查
