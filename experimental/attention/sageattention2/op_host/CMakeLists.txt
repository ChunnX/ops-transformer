# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(COMMAND add_op_to_compiled_list)
    add_op_to_compiled_list()
endif()

if (BUILD_OPEN_PROJECT)
    if(TARGET op_host_aclnn)
        # 注意：aclnn/sage_attn_host.cpp 属 op_api 层实现（引用 libnnopbase 的
        # UniqueExecutor 符号），不参与 op_host_proto 库，否则 op_build dlopen
        # 本库时会因未定义符号失败；它由本地 sageattention2_host 目标与
        # benchmarks/build_aclnn_benchmark.sh 单独编译
        target_sources(op_host_aclnn PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/sage_attention2_def.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/sage_attention2_infershape.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/sage_attention2_tiling.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../quant/quant_int8.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../quant/quant_fp8.cpp
        )
        target_include_directories(op_host_aclnn PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/../aclnn
            ${CMAKE_CURRENT_SOURCE_DIR}/../sageattn_fp8/ascendc
            ${CMAKE_CURRENT_SOURCE_DIR}/../sageattn_fp8/catlass
            ${CMAKE_CURRENT_SOURCE_DIR}/../quant
            # aclnn 轻量执行器头 (aclnn/opdev/op_executor.h) 位于 CANN include/aclnn 子树
            ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
        )
        # TilingImpl 经 platform_ascendc::PlatformAscendC 动态探测核数
        #（GetCoreNumAic/GetCoreNumAiv），符号唯一定义在静态库 libtiling_api.a；
        # 框架默认不给 op_host_aclnn 链接 tiling_api，缺失时 op_build dlopen 报
        # "undefined symbol: GetCoreNumAiv"，此处显式补链。
        if(TARGET tiling_api)
            target_link_libraries(op_host_aclnn PRIVATE tiling_api)
        endif()
    endif()
    if(TARGET op_host_aclnnExc)
        target_sources(op_host_aclnnExc PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/sage_attention2_def.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/sage_attention2_infershape.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/sage_attention2_tiling.cpp
        )
        target_include_directories(op_host_aclnnExc PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/../aclnn
            ${CMAKE_CURRENT_SOURCE_DIR}/../sageattn_fp8/ascendc
            ${CMAKE_CURRENT_SOURCE_DIR}/../quant
            # aclnn 错误码 (aclnn/opdev/op_errno.h) 位于 CANN include/aclnn 子树
            ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
        )
        if(TARGET tiling_api)
            target_link_libraries(op_host_aclnnExc PRIVATE tiling_api)
        endif()
    endif()
endif()

if(COMMAND add_ops_compile_options)
    add_ops_compile_options(
        OP_NAME SageAttention2
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-vf-remove-membar=false
                -mllvm -cce-aicore-hoist-movemask=false
    )
endif()

if (NOT BUILD_OPS_RTY_KERNEL AND COMMAND add_modules_sources)
    # op_api 与 op_host 同级布局
    add_modules_sources(OPTYPE sage_attention2 ACLNNTYPE aclnn
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api)
endif()

# 支持独立构建 Host 共享库以供本地单测与集成测试
if(NOT TARGET op_host_aclnn AND NOT TARGET sageattention2_host)
    add_library(sageattention2_host SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/sage_attention2_def.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sage_attention2_infershape.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sage_attention2_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../aclnn/sage_attn_host.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../quant/quant_int8.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../quant/quant_fp8.cpp
    )
    target_include_directories(sageattention2_host PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../aclnn
        ${CMAKE_CURRENT_SOURCE_DIR}/../sageattn_fp8/ascendc
        ${CMAKE_CURRENT_SOURCE_DIR}/../sageattn_fp8/catlass
        ${CMAKE_CURRENT_SOURCE_DIR}/../quant
    )
    target_compile_features(sageattention2_host PUBLIC cxx_std_17)
endif()
