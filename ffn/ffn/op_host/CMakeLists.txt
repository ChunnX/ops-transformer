# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        ffn_def.cpp
    )
endif()

if (BUILD_OPS_RTY_KERNEL) # 回黄kernel
    add_ops_compile_options(
            OP_NAME FFN
            OPTIONS --cce-auto-sync=off
                    -Wno-deprecated-declarations
                    -Werror
    )
else()
    add_modules_sources(OPTYPE "ffn" ACLNNTYPE "aclnn_exclude")
    # 官方 ops-nn mat_mul_v3 host tiling：只编 Basic ASWT 需要的 arch35 源
    set(FFN_OFFICIAL_MM_TILING_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_asw_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_aswt_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_streamk_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_helper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_key.cpp
    )
    file(GLOB FFN_OFFICIAL_COMMON_TILING_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/matmul/common/op_host/op_tiling/*.cpp
    )
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
        ${FFN_OFFICIAL_MM_TILING_SRCS}
        ${FFN_OFFICIAL_COMMON_TILING_SRCS}
        # mc2 的 mat_mul_v3 host 源仍被 MC2_OPT 编入本 target，其引用
        # Ops::Transformer::DebugTilingContext / TilingPrepareForOpCache / LegacyCommonMgr 等，
        # 补编 mc2 对应实现使其可加载。
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../mc2/3rd/common/op_host/op_tiling/debug_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../mc2/3rd/ops_legacy/op_tiling/op_cache_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../mc2/3rd/ops_legacy/op_tiling/legacy_common_manager.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../mc2/3rd/ops_legacy/op_tiling/runtime_kb_api.cpp
    )
    target_include_directories(${OPHOST_NAME}_tiling_obj BEFORE PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/op_host
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/common/inc
    )
endif()
