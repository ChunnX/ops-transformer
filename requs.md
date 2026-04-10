# 需求

## 需求一

​	当前仓有如下修改 attention/prompt_flash_attention/op_host/CMakeLists.txt

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
     target_link_libraries(softmax_tiling_ut_stub PRIVATE
         -Wl,--allow-multiple-definition
         -Wl,--whole-archive $<TARGET_FILE:tiling_api> -Wl,--no-whole-archive
         -Wl,--no-allow-multiple-definition
     )
 endif()
```

​	添加如下文件 attention/prompt_flash_attention/tests/ut/op_host/arch32/CMakeLists.txt attention/prompt_flash_attention/tests/ut/op_host/arch32/test_prompt_flash_attention_tiling.cpp tests/ut/framework_normal/common/softmax_tiling_mocker.cpp 
tests/ut/framework_normal/common/softmax_tiling_mocker.h

​	这些修改是为了lcov覆盖/home/j60100428/ops-transformer_0408/attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp和/home/j60100428/ops-transformer_0408/attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.h文件的，但当前有一个阻塞问题：就是prompt_flash_attention_tiling.cpp的6242~6249行调了两个高级api函数GetSoftMaxMinTmpSize和GetSoftMaxFlashV2MinTmpSize，

```c
while ((softmaxTmpSize == 0 || softmaxFlashTmpSize == 0) && (softmaxSOuterFactorTmp > 0)) {
            tmpShape = Shape({softmaxSOuterFactorTmp, sInnerFactorTmp});
            softmaxTmpSize = GetSoftMaxMinTmpSize(tmpShape, typeByteSize, true);
            softmaxFlashTmpSize = GetSoftMaxFlashV2MinTmpSize(tmpShape, typeByteSize, sizeof(float), true, true);
            if (softmaxTmpSize == 0 || softmaxFlashTmpSize == 0) {
                softmaxSOuterFactorTmp -= softmaxSOuterFactorStep;
            }
        }
```

这两个函数是另一个仓的（/home/j60100428/asc-devkit/impl/adv_api/tiling/activation/softmax_tiling.cpp），在编译后引在tiling/tiling_api.h中，通过tiling_api库文件引进来

```
(py310) root@kvm-j13684215094:/home/j60100428/ops-transformer_0408# find ../asc-devkit/ -name "tiling_api.h"
../asc-devkit/include/adv_api/tiling_api.h
(py310) root@kvm-j13684215094:/home/j60100428/ops-transformer_0408# find ../Ascend_0401/ -name "tiling_api.h"
../Ascend_0401/cann-9.0.0/x86_64-linux/asc/include/tiling/tiling_api.h
../Ascend_0401/cann-9.0.0/x86_64-linux/asc/include/adv_api/tiling_api.h
```

但现在发现GetSoftMaxMinTmpSize和GetSoftMaxFlashV2MinTmpSize两个函数的实现有问题，并不是我们预期想要的，所以我采取打桩策略，去重定义这些函数，也就有了softmax_tiling_mocker.cpp 文件，我想法是在跑op_host测的UT查看lcov覆盖率时，结合原来的tiling_api库加上softmax_tiling_mocker.cpp、softmax_tiling_mocker.h重编一个tiling_api，并引用，实现对softmax_tiling_mocker.cpp的函数重定义的效果，但目前缺乏了cmake工程支持，我不知道该如何改cmake工程

​	我不确定我这个策略能否成功，你帮我判别一下，如果可以请帮我实现，并在本文档内详细解释注解。

### 实现结论

- 目前可以把方案分成两个方向：
  - 方案一：前置依赖 stub 动态库，在运行时尝试让同名符号优先解析到 stub。
  - 方案二：UT/lcov 场景下重编一个“带 mock 实现的 `tiling_api`”，从源头替换高级 API 实现。
- 结合你的补充说明，方案一你已经试过，实际运行时仍然命中了原始 `tiling_api`，因此它不能作为当前问题的可靠解法。
- 所以更适合作为正式落地方案的是方案二：仅在 `OP_HOST_UT` / lcov 覆盖场景下，重编 `tiling_api`，把 `softmax_tiling_mocker.cpp` 编译进去，用链接阶段的同名符号重定义来替换原实现。

### 为什么方案一不适合作为最终方案

原写法如下：

```cmake
target_link_libraries(softmax_tiling_ut_stub PRIVATE
    -Wl,--allow-multiple-definition
    -Wl,--whole-archive $<TARGET_FILE:tiling_api> -Wl,--no-whole-archive
    -Wl,--no-allow-multiple-definition
)
```

如果把这段理解为“前置依赖 stub”的一种变体，它的问题在于：

1. `softmax_tiling_ut_stub` 本身只是一个“提供重定义符号的桩库”，但高级 API 的实际解析顺序受 `dlopen`、DT_NEEDED 顺序、符号可见性、链接器选项等多因素影响。
2. 即使 `libophost_transformer_ut.so` 前面挂了 stub，也不代表运行时一定优先命中 stub；你已经验证过，实际还是进了原始 `tiling_api`。
3. 这个问题不是 `PromptFlashAttention` 专属，只要 op_host tiling 代码里直接调用高级 API，都可能遇到同样的问题。
4. 因此方案一最多只能算“尝试性手段”，不能作为稳定覆盖 lcov 的基础设施。

### 方案二：重编 tiling_api，作为 op_host UT/lcov 的专用高级 API 实现

这个方案更符合你当前的诉求：不依赖运行时“谁先解析到谁”，而是在构建阶段直接生成一份带 mock 实现的 `tiling_api`，让所有 op_host UT 在链接时统一使用它。

#### 核心思路

1. 保留原 `tiling_api` 的大部分源码和依赖。
2. 在 UT/lcov 专用构建分支中，把 `softmax_tiling_mocker.cpp` 一起编进新的 `tiling_api`。
3. 通过 `--allow-multiple-definition` 或源码排除原 `softmax_tiling.cpp` 中对应实现的方式，让最终导出的符号以 mock 版本为准。
4. op_host UT 不再额外挂 stub，也不赌运行时解析顺序，而是直接链接这份“UT 专用 tiling_api”。

#### 方案二为什么更稳

- 它把问题从“运行时符号抢占”转成了“构建期确定唯一导出实现”。
- 对所有 op_host tiling UT 都生效，而不是只对 `PromptFlashAttention` 单独做补丁。
- 更适合做 lcov 基础能力，因为所有调用高级 API 的 tiling 代码最终都统一落到这份 UT 版本 `tiling_api` 上。

#### 方案二推荐的工程落点

建议不要继续在 `attention/prompt_flash_attention/op_host/CMakeLists.txt` 下做文章。  
你提到这里的改动是之前方案一的残留，这个判断是合理的；如果采用方案二，这里的 stub 逻辑应该舍弃。

更合理的落点应在“UT 框架层”或“tiling_api 的引入层”，也就是：

- `tests/ut/framework_normal/op_host/CMakeLists.txt`
- 或者更上游的 UT 公共 CMake
- 或者专门新增一个 `tiling_api_ut` / `tiling_api_lcov` 目标，再在 `OP_HOST_UT` 下替换掉默认 `tiling_api`

#### 方案二的推荐 CMake 组织方式

推荐新增一个专用目标，例如：

```cmake
if(UT_TEST_ALL OR OP_HOST_UT)
    add_library(tiling_api_ut SHARED
        ${TILING_API_ORIGINAL_SRCS}
        ${PROJECT_SOURCE_DIR}/tests/ut/framework_normal/common/softmax_tiling_mocker.cpp
    )

    target_include_directories(tiling_api_ut PRIVATE
        ${OP_TILING_INCLUDE}
        ${PROJECT_SOURCE_DIR}/tests/ut/framework_normal/common
        ...原 tiling_api 所需头文件...
    )

    target_link_libraries(tiling_api_ut PRIVATE
        ...原 tiling_api 所需依赖...
        -Wl,--allow-multiple-definition
    )
endif()
```

然后在 op_host UT 侧切换链接目标：

```cmake
target_link_libraries(${OPHOST_NAME}_ut PRIVATE
    ...
    $<$<BOOL:${OP_HOST_UT}>:tiling_api_ut>
    $<$<NOT:$<BOOL:${OP_HOST_UT}>>:tiling_api>
    ...
)
```

#### 方案二有两种实现细节

##### 写法 A：保留原 softmax_tiling.cpp，再用 mock 同名符号覆盖

特点：

- 改动小。
- 适合你当前已有 `softmax_tiling_mocker.cpp` 的状态。
- 需要链接器允许多重定义，并确认最终导出的符号是 mock 版本。

风险：

- 仍然要关注链接顺序。
- 不同链接器/不同构建参数下，最终保留哪一个定义可能存在差异，需要通过 `nm -D` / `readelf -Ws` 验证。

##### 写法 B：UT 构建时排除原 softmax_tiling.cpp 中对应实现，再只编 mock

特点：

- 最稳。
- 最终只有一个实现，不存在多重定义冲突。

风险：

- 需要你能控制 `tiling_api` 源码列表。
- 如果 `softmax_tiling.cpp` 里除了这两个函数还有其他必须保留的实现，就要做更细颗粒度拆分，或者从原仓抽出公共辅助逻辑。

从可靠性角度看，**写法 B 优于写法 A**；  
从接入成本角度看，**写法 A 更容易先落地验证**。

#### 方案二与 `softmax_tiling_mocker.cpp` 的关系

`softmax_tiling_mocker.cpp` 当前做的事，本质上不是普通 testcase stub，而是“UT 版高级 API 实现”。  
所以如果采用方案二，它的定位应该从：

- “PromptFlashAttention 的临时桩文件”

升级为：

- “op_host UT/lcov 通用的高级 API 替代实现”

这也更符合你说的：  
**这件事不是 PromptFlashAttention 专属，而是所有在 op_host tiling 文件里调用高级 API 的算子都会遇到的问题。**

### 这套方案为什么能覆盖到 lcov

你要覆盖的是：

- `attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp`
- `attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.h`

阻塞点在于 6242~6249 行的高级 API 调用会返回不符合预期的结果，导致后续分支跑不到。

现在改成“UT 专用 tiling_api”之后：

1. `PromptFlashAttention` 的 tiling 逻辑仍然是原仓源码在执行；
2. 只是它内部调用的 softmax tmp size 计算函数，被 UT 版 `tiling_api` 接管；
3. stub 内部通过 `SoftmaxTilingMocker::SetSocVersion()` 显式指定 SoC，规避了原实现依赖环境态 `PlatformAscendCManager::GetInstance()` 的问题；
4. 这样 6242~6249 行对应的循环和后续分支就有机会稳定走通，lcov 才能打到目标源码。

也就是说：被覆盖的仍然是 `prompt_flash_attention_tiling.cpp` 的代码路径，不是仅仅覆盖到 `softmax_tiling_mocker.cpp`。

### 相关测试文件为什么也成立

`attention/prompt_flash_attention/tests/ut/op_host/arch32/test_prompt_flash_attention_tiling.cpp` 里面已经有这段逻辑：

```cpp
SoftmaxTilingMocker::GetInstance().SetSocVersion("Ascend910B");
```

它的作用是给 stub 提供显式 SoC。

这样 stub 内部的：

```cpp
platform_ascendc::PlatformAscendCManager::GetInstance(
    SoftmaxTilingMocker::GetInstance().GetSocVersion())
```

就不再依赖外部环境，而是由 UT 自己控制平台上下文，因此更适合稳定覆盖。

### 方案二需要注意的边界

1. 这套方案必须严格限定在 `UT_TEST_ALL OR OP_HOST_UT` 或 lcov 构建场景下，不能污染正式产品构建。
2. 如果后续 `tiling_api` 的函数签名变化，`softmax_tiling_mocker.cpp` 必须同步更新。
3. 如果采用“多重定义覆盖”的写法，必须通过符号检查确认最终导出的是 mock 版本。
4. 如果采用“排除原实现”的写法，需要确认不会误删 `softmax_tiling.cpp` 中其他仍被依赖的函数。
5. 由于这是 op_host 侧通用问题，后续建议把 `softmax_tiling_mocker.cpp/.h` 放在更明确的“UT 高级 API 替代实现”位置，而不是挂靠某个单算子语义。

### 本次落地文件

- 当前文档已明确：
  - `attention/prompt_flash_attention/op_host/CMakeLists.txt` 中那部分 stub 逻辑属于方案一残留，可舍弃；
  - 方案二应放在 UT 框架层或 `tiling_api` 的 UT 专用目标上；
  - `tests/ut/framework_normal/common/softmax_tiling_mocker.cpp`
  - `tests/ut/framework_normal/common/softmax_tiling_mocker.h`
    应作为通用高级 API mock 实现继续复用。

### 最后判断

- 你的“打桩覆盖 lcov”方向是对的。
- 结合你已经验证过的结果，方案一不能作为可靠解法。
- 现阶段更值得推进的是方案二：**在 op_host UT/lcov 场景下重编 `tiling_api`，把 `softmax_tiling_mocker.cpp/.h` 纳入进去，形成 UT 专用高级 API 实现。**

### 建议你下一步验证

1. 先确认当前工程里 `tiling_api` 是由哪个 CMake 目标或外部包引入的。
2. 判断你是否能直接拿到 `tiling_api` 的源码列表；如果能，优先做“tiling_api_ut”目标。
3. 先用写法 A 快速验证可行性，再决定是否升级到写法 B。
4. 编译后用 `nm -D` / `readelf -Ws` 检查：
   - `GetSoftMaxMinTmpSize`
   - `GetSoftMaxFlashV2MinTmpSize`
   是否已经由 UT 版 `tiling_api` 导出。
5. 再跑任意依赖这些高级 API 的 op_host UT 和 lcov，确认不是只对 `PromptFlashAttention` 生效，而是对同类算子都生效。

