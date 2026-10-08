# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

# CompressorGrad 双栈：arch35(ascend950) 为 PyPTO 实现（依赖 pypto_pro codegen），
# arch22(ascend910b/ascend910_93) 为原生 AscendC 实现。pypto_pro 仅在 950 参与构建时
# 才是硬依赖，避免无 pypto_pro 的 arch22-only 环境被 require_pypto_pro 整体跳过。
set(_COMPRESSOR_GRAD_WITH_ARCH35 OFF)
if ("ascend950" IN_LIST ASCEND_COMPUTE_UNIT)
    set(_COMPRESSOR_GRAD_WITH_ARCH35 ON)
endif()

if (_COMPRESSOR_GRAD_WITH_ARCH35)
    require_pypto_pro(compressor_grad)
endif()
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
    compressor_grad_def.cpp
    )
    if (_COMPRESSOR_GRAD_WITH_ARCH35)
        # PyPTO kernel: runs codegen at configure time and exposes PYPTO_GEN_DIR for the host tiling -I below.
        # Must be called BEFORE add_modules_sources(...) (which reads PYPTO_GEN_DIR).
        enable_pypto_kernel(compressor_grad)
    endif()
endif()

set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME CompressorGrad
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
        OP_NAME CompressorGrad
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
    )
else()
    add_ops_compile_options(
        OP_NAME CompressorGrad
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE compressor_grad ACLNNTYPE aclnn)
    add_tiling_modules()
    # arch22 tiling 不在 add_modules_sources 的 glob 范围（op_host/arch22/ 而非 op_tiling/），
    # 显式收编；也借此避开 pypto 生成头对 glob 命中的 tiling 源的 force-include（obj_func.cmake）。
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/arch22/compressor_grad_tiling.cpp
    )
    if (_COMPRESSOR_GRAD_WITH_ARCH35)
        # arch35(PyPTO) tiling 与 register 中的 arch35 分支仅在 950 构建时编译
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/compressor_grad_tiling.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/compressor_grad_tiling_register.cpp
            TARGET_DIRECTORY ${OPHOST_NAME}_tiling_obj
            PROPERTIES COMPILE_DEFINITIONS COMPRESSOR_GRAD_HAS_ARCH35)
    endif()
endif()
