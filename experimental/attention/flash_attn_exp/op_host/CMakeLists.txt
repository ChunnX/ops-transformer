# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()

set(FLASH_ATTN_SKIP_HEADER TRUE CACHE INTERNAL "Skip packaging header for this operator")
set(FLASH_ATTN_INNER_SKIP_HEADER TRUE CACHE INTERNAL "Skip packaging inner header for this operator")

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        flash_attn_def.cpp
    )
endif()

set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    # ENABLE_EXPERIMENTAL 构建时依赖路径会自动加 experimental/ 前缀，
    # "../attention/common" 映射为仓级 attention/common（kernel 侧代码位于
    # op_kernel 下，经 staged src/common 引用该依赖）
    set(flash_attn_exp_depends ../attention/common CACHE INTERNAL "Dependencies for flash_attn_exp")
    add_ops_compile_options(
        OP_NAME FlashAttn
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
            OP_NAME FlashAttn
            OPTIONS --cce-auto-sync=off
                    -Wno-deprecated-declarations
                    -Werror
    )
else()
    add_ops_compile_options(
            OP_NAME FlashAttn
            OPTIONS --cce-auto-sync=off
                    -Wno-deprecated-declarations
                    -Werror
    )
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
    flash_attn_tiling_info_parser.cpp
    ../../../../attention/common/op_host/fia_tiling_shape.cpp
    )
    add_modules_sources_with_soc(
        # 该变量用于判断op_api目录是否从op_host移出
        OP_API_INDEPENDENT ON
        # op_api新的地址
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OPTYPE flash_attn
        ACLNNTYPE aclnn_inner
    )
endif()
