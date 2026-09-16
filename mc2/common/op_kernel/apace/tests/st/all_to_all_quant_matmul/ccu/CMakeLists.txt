# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# ============================================================
# CMakeLists.txt — apace AllToAllQuantMatmul ST（CCU 直调变体）
#
# 顶层 tests/st/CMakeLists.txt 已设置 ASCEND_DIR / SYSTEM_PREFIX /
# BISHENG / ops-tensor（apace_blaze_api INTERFACE target）
# ============================================================

set(APACE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")          # mc2/common/op_kernel/apace
set(OP_KERNEL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../..")    # mc2/common/op_kernel
set(BLAZE_COMMON "${OPS_TENSOR_INCLUDE_DIR}")

# kernel 模式（-xasc）include 路径：apace kernel + Blaze + hcomm/ccu（CCU 直调增量）。
# ops-tensor 三个根必须置于 CANN 目录之前：本机 CANN 9.2.0 的 asc/include 下自带旧版
# tensor_api（无 asc::te 命名空间），会抢先劫持 blaze 的 "tensor_api/tensor.h" 解析。
set(KERNEL_INCLUDES
    ${BLAZE_COMMON}                                                         # blaze/ + ops-tensor 根
    ${BLAZE_COMMON}/tensor_api/include                                      # tensor_api/...
    ${BLAZE_COMMON}/tensor_api
    ${OP_KERNEL_ROOT}                                                       # apace/ 头文件
    ${APACE_ROOT}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../utils                                  # tests/st/utils
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include                              # basic_api/ adv_api/ comm_api/
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include/basic_api                     # kernel_basic_intf.h（裸名 include）
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include/interface
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include/tiling                        # platform/platform_ascendc.h
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include/comm_api/ccu                  # ccu_host_launch.h
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/ascendc/include/highlevel_api             # kernel_tiling/kernel_tiling.h
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/include                                   # hccl/ acl/
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/include/hccl
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/include/hcomm
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/include/hcomm/ccu                         # ccu/ccu_*.hpp
    ${ASCEND_DIR}/pkg_inc
    ${ASCEND_DIR}/pkg_inc/hcomm
    ${ASCEND_DIR}/pkg_inc/hcomm/ccu                                          # hccl_ccu_res.h / ccu_res.h
    ${ASCEND_DIR}/compiler/tikcpp/tikcfw
    ${ASCEND_DIR}/include
)

set(HOST_INCLUDES
    ${APACE_ROOT}
    ${APACE_ROOT}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include/tiling
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/include
)

set(KERNEL_OPTS "-xasc" "--npu-arch=dav-3510" "-DASC_DEVKIT_MAJOR=9" "-w" "-O3")
set(HOST_OPTS "-std=c++17" "-w" "-O2")

# 链接库：兄弟目录基础组合 + CCU 直调增量（hcomm/asccomm_ccu/c_sec/acl_rt/ascendc_runtime/register）
set(COMMON_LIBS dl platform tiling_api ascendcl acl_rt runtime hccl stdc++ hcomm asccomm_ccu c_sec
    ascendc_runtime register pthread)

add_executable(apace_a2a_qmm_ccu_st
    src/main.cpp
)

# main.cpp: kernel 模式编译（单翻译单元，经 include 引入 .asc 链）
set_source_files_properties(src/main.cpp PROPERTIES
    COMPILE_OPTIONS "${KERNEL_OPTS}"
    INCLUDE_DIRECTORIES "${KERNEL_INCLUDES}"
)

target_link_directories(apace_a2a_qmm_ccu_st PRIVATE
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
    ${ASCEND_DIR}/lib64
)

target_link_libraries(apace_a2a_qmm_ccu_st PRIVATE ${COMMON_LIBS} apace_blaze_api)
