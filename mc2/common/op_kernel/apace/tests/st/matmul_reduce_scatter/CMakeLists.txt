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
# CMakeLists.txt — apace QuantMatmulReduceScatter ST (urma/UDMA PUT)
# ============================================================

set(APACE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(OP_KERNEL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")
set(BLAZE_COMMON "${OPS_TENSOR_INCLUDE_DIR}")

set(KERNEL_INCLUDES
    ${OP_KERNEL_ROOT}
    ${APACE_ROOT}
    ${APACE_ROOT}/kernel/matmul/quant_batch_matmul
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils
    ${BLAZE_COMMON}
    # 子模块 tensor_api 两个根（-iquote 传递见 KERNEL_OPTS；此处 -I 兜底 angle 包含）
    ${BLAZE_COMMON}/tensor_api/include
    ${BLAZE_COMMON}/tensor_api
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/ascendc/include/highlevel_api
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

# kernel 模式（-xasc）编译选项。
# 注意：bisheng 会把 $ASCEND_TOOLKIT_HOME 下的 asc/include 注入到用户 -I 之前，
# 导致同名 tensor_api/*.h 命中 CANN 内置旧副本（无 asc::te），blaze layout_utils.h 报
# "no member named 'te' in namespace 'asc'"。-iquote 目录在 quote 包含解析中优先级最高，
# 可压过注入路径，故子模块 tensor_api 两个根必须以 -iquote 传入。
set(KERNEL_OPTS "-xasc" "--npu-arch=dav-3510" "-DASC_DEVKIT_MAJOR=9" "-w" "-O3"
    "-iquote" "${BLAZE_COMMON}/tensor_api/include"
    "-iquote" "${BLAZE_COMMON}/tensor_api")
set(HOST_OPTS "-std=c++17" "-w" "-O2")

set(COMMON_LIBS dl platform tiling_api ascendcl runtime hccl stdc++ pthread)

add_executable(apace_qmm_rs_urma_st
    src/main.cpp
)

set_source_files_properties(src/main.cpp PROPERTIES
    COMPILE_OPTIONS "${KERNEL_OPTS}"
    INCLUDE_DIRECTORIES "${KERNEL_INCLUDES}"
)

target_link_directories(apace_qmm_rs_urma_st PRIVATE
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
    ${ASCEND_DIR}/lib64
)

target_link_libraries(apace_qmm_rs_urma_st PRIVATE ${COMMON_LIBS} apace_blaze_api)
target_link_libraries(apace_qmm_rs_urma_st PRIVATE "-Wl,--no-as-needed" hccl_fwk "-Wl,--as-needed")
