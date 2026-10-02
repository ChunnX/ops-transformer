# ==============================================================================
# SageAttention2 op_kernel 独立编译验证目标（M2b 构建接线收口）
# 目标架构：Ascend 950PR —— bisheng 架构串 dav-c310（内部 __NPU_ARCH__=3510）
# ==============================================================================
#
# 【接线结论（2026-10-02 容器内实测，依据见下）】
# 标准构建不由本文件驱动：cann-ops-transformer 仓的 op_kernel 由 CANN 算子框架在
# 算子包 binary 阶段编译——op_host/CMakeLists 经 add_modules_sources_with_soc /
# add_ops_compile_options 挂接（参考仓内 fused_infer_attention_score），op_kernel
# 源码经 cmake/custom_build.cmake 安装进算子包（ascendc/<op>/op_kernel），最终由
# CANN 工具链的 ascendc_kernel_cmake（ascendc_library → precompile/preprocess →
# AIC(dav-c310-cube)/AIV(dav-c310-vec) 拆分编译 → merge_mix_obj.sh 合并 device.o）
# 完成设备侧编译。故本文件定位为独立快速编译验证目标（standalone target）：
# 不接进顶层构建（顶层 CMakeLists 仅 add op_host/op_api），仅用于命令行快速验证
# 内核源码在 CANN 9.2.0-beta.1 工具链下的设备可编译性。
#
# 【架构串依据（全部为容器内实测/工具链内证）】
# 1. bisheng/ccec 报错 "Unsupported CCE architecture: dav-3510"——dav-3510 不是合法架构串；
#    合法串枚举于 ccec 二进制内（dav-c310 / dav-c310-cube / dav-c310-vec 等）。
# 2. tilelang-ascend 3rdparty/pto-isa README："Ascend A5 (Ascend 950)"；其 a5 算子
#    CMakeLists 统一使用 -xcce + --cce-aicore-arch=dav-c310（MIX）/ -cube / -vec。
# 3. 工具链内证：--cce-aicore-arch=dav-c310 预处理得 __NPU_ARCH__=3510；
#    asc/impl/basic_api/reg_compute/dav_3510/ 即该代内核实现目录；
#    bisheng_intf.cmake（ascendc_kernel_cmake）对 c310 定义 -D__DAV_C310__ + dav-c310-cube/-vec。
#    故 CATLASS_ARCH=3510（-D 宏）与 catlass ArchTag（catlass/arch/arch.hpp Ascend950、
#    gemm/tile/ascend950/）口径一致，保持不变；bisheng 架构串改用 dav-c310。
#
# 【M2b-0 结果（2026-10-02，容器 cann_dev_950 实测）】
# 62 处 API 形状错误全部清零（0 前端错误，-ferror-limit=0 实测）。拆分编译口径（对齐
# ascendc_kernel_cmake 的 c310_aiv/aic_intf_pub）：
#   AIC pass：bisheng -xcce --cce-aicore-arch=dav-c310-cube --cce-aicore-only --cce-auto-sync
#             --cce-mask-opt → EXIT=0（对象 .text 48,576B + .ascendc_tiling 节）。
#   AIV pass：同形 -vec → EXIT=0（对象 .text 232,032B + .ascendc_tiling 节；迭代 2 打通
#             attention loop 三 VF 后，Q 量化/KV 预处理标量浮点核心与 P NZ 分形打包/GM
#             块写回亦已全向量化）。
# 【AIV 阻塞根因（迭代 2 控制变量探针矩阵钉死，存 tests/aiv_backend_probe.cpp）】
# 原注记「调用链 ≤1 层 / 入口形态」两条根因均被探针否证（2 层链 + 类层次包装探针全 PASS；
# 同一函数体 __global__/__aicore__ 形态差异实为死代码伪 PASS——无 __global__ 引用时后端
# 根本不降低该函数）。被证实的判别条件（AIV 编译单元内，含 __global__ 入口内联链全部代码）：
#   R1 任何标量浮点 ALU/fcmp/浮点库调用（fdiv、fabs、rintf、expf、标量 half/bf16 cast 即
#      fpext/fptrunc）⇒ 后端 "Unsupported scalar instruction in AIV loop"（-O0/-O3 同现）；
#   R2 「计算值/装载值 → float」的标量位强转 ⇒ ISel "Cannot select: f32 = bitcast"
#      （p33f/p33g；u32→f32 的 bitcast(load) 可选中，p33e）；循环内带标量参的向量指令
#      （Muls/Duplicate 的标量形参）必须循环不变（p33d）或改向量×向量（p33b/p33e）。
#   合法通路 = 状态数学全向量 1-lane 掩码 + 寄存器 Duplicate 广播（CANN softmax_flash_impl.h
#   同款）、half/bf16 展开「纯整数位型 + u32 通道过桥」、float 侧只允许直接标量装载。
# 迭代 2 已按 R1/R2 完成 attention loop 三 VF（softmax 状态数学/fold alpha 条件乘）向量化
# 改写 + wrapper qh 位型展开，两 pass EXIT=0（证据 = 上方两对象非空 .text）。
# 【merge 结果（2026-10-02 容器实测，llvm-objcopy localize + ld.lld 合并 EXIT=0）】
# 双 pass 对象各持引擎私有入口副本与 g_tilingKey（拆分编译预期形态而非冲突）：先以
# llvm-objcopy --localize-symbol=g_tilingKey 双对象本地化，再按 merge_obj.sh 的 AIC 先序
# ld.lld -x -m aicorelinux --allow-multiple-definition -Ttext=0 <aic.o> <aiv.o> -static 合并
# → EXIT=0，合并对象 .text 280,672B + .ascendc_tiling 节，nm -u 未定义符号面 = 0——旧
# 「undefined rintf / ldexpf（各 30 次引用级）」阻塞面已随 Q 量化/KV 预处理
#（StepVectorQQuantDevice 的 aicNearbyInt=__builtin_rintf 与 Sa2KvPreprocessor 的
# __builtin_ldexpf 次正规支路）Vector 通路整体向量化清零。M2b-0 出口判据推进：两 pass 0 错
#（达成）→ merge 0 错（达成）→ 含非空代码节的可合并对象（达成）；双 pass 入口/g_tilingKey
# 仲裁与各引擎私有副本的真机语义验证挂 M3（真机执行仍未验证）。
#
cmake_minimum_required(VERSION 3.16)
project(sa2_kernel_build LANGUAGES CXX)

# 1. CANN Toolkit 路径探测
if(NOT DEFINED ASCEND_HOME_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
        set(ASCEND_HOME_PATH $ENV{ASCEND_TOOLKIT_HOME})
    else()
        set(ASCEND_HOME_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    endif()
endif()

# 2. 目标架构配置：CATLASS_ARCH 为 -D 宏口径（3510 = dav-c310 内部 __NPU_ARCH__，见头部依据）；
#    CATLASS_CCE_ARCH 为 bisheng --cce-aicore-arch 实际架构串（dav-3510 非法，实测确定 dav-c310）。
if(NOT DEFINED CATLASS_ARCH OR CATLASS_ARCH STREQUAL "")
    if(DEFINED ENV{CATLASS_ARCH})
        set(CATLASS_ARCH $ENV{CATLASS_ARCH})
    else()
        set(CATLASS_ARCH "3510")
    endif()
endif()
if(NOT DEFINED CATLASS_CCE_ARCH OR CATLASS_CCE_ARCH STREQUAL "")
    if(DEFINED ENV{CATLASS_CCE_ARCH})
        set(CATLASS_CCE_ARCH $ENV{CATLASS_CCE_ARCH})
    else()
        set(CATLASS_CCE_ARCH "dav-c310")
    endif()
endif()

# 3. 包含路径解析（集成 catlass/include, tla, op_host, op_kernel）
set(SA2_KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SA2_HOST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_host)

# 探测 CATLASS / TLA 路径
set(CATLASS_CANDIDATE_PATHS
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/catlass/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../catlass/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../catlass/include"
    "${ASCEND_HOME_PATH}/include/experiment/catlass"
    "${ASCEND_HOME_PATH}/include"
)

set(CATLASS_INC_DIR "")
foreach(p ${CATLASS_CANDIDATE_PATHS})
    if(EXISTS "${p}/catlass/catlass.hpp")
        get_filename_component(CATLASS_INC_DIR "${p}" ABSOLUTE)
        break()
    endif()
endforeach()

if(CATLASS_INC_DIR STREQUAL "")
    message(STATUS "[sa2_kernel] Warning: catlass/catlass.hpp not found in candidate paths, using repo fallback.")
    set(CATLASS_INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../catlass/include")
endif()

message(STATUS "[sa2_kernel] CANN Toolkit Home: ${ASCEND_HOME_PATH}")
message(STATUS "[sa2_kernel] Target Architecture: bisheng=${CATLASS_CCE_ARCH} (CATLASS_ARCH=${CATLASS_ARCH})")
message(STATUS "[sa2_kernel] CATLASS Include: ${CATLASS_INC_DIR}")

# CANN 设备侧头（kernel_operator.h 等）所在 devkit 子目录（对齐 ascendc_kernel_cmake/
# bisheng_config.cmake 的 devkit 探测：aarch64-linux 优先，x86_64-linux 次之，兜底 tools/compiler）
if(EXISTS "${ASCEND_HOME_PATH}/aarch64-linux/asc/include/kernel_operator.h")
    set(SA2_DEVKIT_DIR "${ASCEND_HOME_PATH}/aarch64-linux")
elseif(EXISTS "${ASCEND_HOME_PATH}/x86_64-linux/asc/include/kernel_operator.h")
    set(SA2_DEVKIT_DIR "${ASCEND_HOME_PATH}/x86_64-linux")
else()
    set(SA2_DEVKIT_DIR "${ASCEND_HOME_PATH}")
endif()

set(SA2_KERNEL_INCLUDES
    ${SA2_KERNEL_DIR}
    ${SA2_HOST_DIR}
    ${CATLASS_INC_DIR}
    ${CATLASS_INC_DIR}/tla
    ${SA2_DEVKIT_DIR}/asc
    ${SA2_DEVKIT_DIR}/asc/include
    ${SA2_DEVKIT_DIR}/asc/include/adv_api
    ${SA2_DEVKIT_DIR}/asc/include/basic_api
    ${SA2_DEVKIT_DIR}/asc/include/c_api
    ${SA2_DEVKIT_DIR}/asc/include/simt_api
    ${SA2_DEVKIT_DIR}/asc/include/utils
    ${SA2_DEVKIT_DIR}/asc/impl/adv_api
    ${SA2_DEVKIT_DIR}/asc/impl/basic_api
    ${SA2_DEVKIT_DIR}/asc/impl/c_api
    ${SA2_DEVKIT_DIR}/asc/impl/simt_api
    ${SA2_DEVKIT_DIR}/asc/impl/utils
    ${SA2_DEVKIT_DIR}/tikcpp/tikcfw
    ${SA2_DEVKIT_DIR}/tikcpp/tikcfw/interface
    ${SA2_DEVKIT_DIR}/tikcpp/tikcfw/impl
)

set(SA2_KERNEL_SRCS
    ${SA2_KERNEL_DIR}/sa2_kernel.cpp
)

# 4. 编译器检查：优先使用 BiSheng / CCEC (CANN 9.2.0-beta.1 官方编译器)
find_program(BISHENG_COMPILER NAMES bisheng ccec
    HINTS
    ${ASCEND_HOME_PATH}/bin
    ${ASCEND_HOME_PATH}/tools/ccec_compiler/bin
    ${SA2_DEVKIT_DIR}/ccec_compiler/bin
    ENV PATH
)

# 5. AscendC 设备编译标志（对齐工具链 a5 MIX 内核标准用法与 ascendc_kernel_cmake 的
#    device_intf_pub）：
#    -xcce                       混编模式（宿主 pass + 设备 pass；无标注函数视为 __host__，
#                                设备侧函数/包装须带 __aicore__，内核源码已按此标注）
#    --cce-aicore-arch=dav-c310  MIX 架构串（dav-3510 非法，见头部依据）
#    -D__AICORE__                内核源码设备执行路径选择宏（与宿主仿真路径区分）
#    -D__DAV_C310__              CANN 框架对 c310 一代的约定宏（bisheng_intf.cmake 同款）
set(SA2_ASC_FLAGS
    "-xcce"
    "-std=c++17"
    "-O3"
    "--cce-aicore-arch=${CATLASS_CCE_ARCH}"
    "-DCATLASS_ARCH=${CATLASS_ARCH}"
    "-D__AICORE__"
    "-D__DAV_C310__"
    "-Wno-macro-redefined"
    "-Wno-ignored-attributes"
)

# 6. 生成目标：sa2_kernel 算子对象文件库
if(CMAKE_ASC_COMPILER_LOADED)
    # CANN 官方 ASC 语言工具链（ascendc_kernel_cmake 提供）——架构旗标由 CANN intf_pub 注入，
    # 此处仅声明语言与包含路径；独立验证路径推荐走下方 bisheng 分支。
    set_source_files_properties(${SA2_KERNEL_SRCS} PROPERTIES LANGUAGE ASC)
    add_library(sa2_kernel OBJECT ${SA2_KERNEL_SRCS})
    target_include_directories(sa2_kernel PRIVATE ${SA2_KERNEL_INCLUDES})
    target_compile_definitions(sa2_kernel PRIVATE CATLASS_ARCH=${CATLASS_ARCH} __AICORE__ __DAV_C310__)
    message(STATUS "[sa2_kernel] Configured via standard CMake ASC language toolchain.")
elseif(BISHENG_COMPILER)
    # M2b-0 拆分编译（对齐 ascendc_kernel_cmake/legacy_modules/bisheng_intf.cmake 的
    # c310_aiv/aic_intf_pub：-vec / -cube 各编译一次 + ld.lld 合并）。
    set(INC_FLAGS "")
    foreach(inc ${SA2_KERNEL_INCLUDES})
        list(APPEND INC_FLAGS "-I${inc}")
    endforeach()

    set(SA2_SPLIT_FLAGS
        "-xcce"
        "-std=c++17"
        "-O3"
        "-D__DAV_C310__"
        "--cce-aicore-only"
        "--cce-auto-sync"
        "--cce-mask-opt"
        "-DCATLASS_ARCH=${CATLASS_ARCH}"
        "-D__AICORE__"
        "-Wno-macro-redefined"
        "-Wno-ignored-attributes"
    )

    set(SA2_AIV_OBJ ${CMAKE_CURRENT_BINARY_DIR}/sa2_kernel_aiv.o)
    set(SA2_AIC_OBJ ${CMAKE_CURRENT_BINARY_DIR}/sa2_kernel_aic.o)
    set(SA2_KERNEL_OBJ ${CMAKE_CURRENT_BINARY_DIR}/sa2_kernel.o)

    add_custom_command(
        OUTPUT ${SA2_AIV_OBJ}
        COMMAND ${BISHENG_COMPILER} ${SA2_SPLIT_FLAGS} --cce-aicore-arch=dav-c310-vec ${INC_FLAGS}
                -c ${SA2_KERNEL_SRCS} -o ${SA2_AIV_OBJ}
        DEPENDS ${SA2_KERNEL_SRCS}
        COMMENT "[sa2_kernel] AIV pass (dav-c310-vec)"
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${SA2_AIC_OBJ}
        COMMAND ${BISHENG_COMPILER} ${SA2_SPLIT_FLAGS} --cce-aicore-arch=dav-c310-cube ${INC_FLAGS}
                -c ${SA2_KERNEL_SRCS} -o ${SA2_AIC_OBJ}
        DEPENDS ${SA2_KERNEL_SRCS}
        COMMENT "[sa2_kernel] AIC pass (dav-c310-cube)"
        VERBATIM
    )
    # 合并：MIX 拆分编译下双 pass 对象各持引擎私有副本（入口 sa2_attention2 与
    # g_tilingKey 均重名——入口两份 = 各引擎自身的 lowering 产物 + .ascend.meta 引擎
    # 类型元数据区分，属拆分编译预期形态而非冲突；对齐框架 merge_obj.sh 的 AIC 先序，
    # --allow-multiple-definition 仲裁保留首份全局符号）。语义仲裁的真机验证挂 M3。
    find_program(LLVM_OBJCOPY NAMES llvm-objcopy
        HINTS ${ASCEND_HOME_PATH}/tools/ccec_compiler/bin ENV PATH)
    set(SA2_AIV_OBJ_L ${CMAKE_CURRENT_BINARY_DIR}/sa2_kernel_aiv_l.o)
    set(SA2_AIC_OBJ_L ${CMAKE_CURRENT_BINARY_DIR}/sa2_kernel_aic_l.o)
    set(LD_LLD "${ASCEND_HOME_PATH}/tools/ccec_compiler/bin/ld.lld")
    add_custom_command(
        OUTPUT ${SA2_KERNEL_OBJ}
        COMMAND ${CMAKE_COMMAND} -E copy ${SA2_AIV_OBJ} ${SA2_AIV_OBJ_L}
        COMMAND ${CMAKE_COMMAND} -E copy ${SA2_AIC_OBJ} ${SA2_AIC_OBJ_L}
        COMMAND ${LLVM_OBJCOPY} --localize-symbol=g_tilingKey ${SA2_AIV_OBJ_L}
        COMMAND ${LLVM_OBJCOPY} --localize-symbol=g_tilingKey ${SA2_AIC_OBJ_L}
        COMMAND ${LD_LLD} -x -m aicorelinux --allow-multiple-definition -Ttext=0
                ${SA2_AIC_OBJ_L} ${SA2_AIV_OBJ_L} -static
                -o ${SA2_KERNEL_OBJ}
        DEPENDS ${SA2_AIV_OBJ} ${SA2_AIC_OBJ}
        COMMENT "[sa2_kernel] merge AIC+AIV -> sa2_kernel.o (ld.lld -m aicorelinux)"
        VERBATIM
    )
    add_custom_target(sa2_kernel ALL DEPENDS ${SA2_KERNEL_OBJ})
    message(STATUS "[sa2_kernel] Configured via BiSheng split compile (vec+cube+merge).")
else()
    # 宿主机 / 仿真模式 Fallback (以 CXX 语法检查与桩实现构建对象)
    add_library(sa2_kernel OBJECT ${SA2_KERNEL_SRCS})
    target_include_directories(sa2_kernel PRIVATE ${SA2_KERNEL_INCLUDES})
    target_compile_options(sa2_kernel PRIVATE -std=c++17 -O3 -Wall -Wextra)
    target_compile_definitions(sa2_kernel PRIVATE CATLASS_ARCH=${CATLASS_ARCH} __CCE_KT_TEST__)
    message(STATUS "[sa2_kernel] BiSheng not found; configured host syntax/simulation object fallback.")
endif()
