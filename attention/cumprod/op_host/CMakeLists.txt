# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.18.4)
require_pypto_pro(cumprod)
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    # op_host_aclnnExc (not op_host_aclnn): with hand-written op_api the def must stay out of
    # the aclnn autogen library, otherwise the generated aclnn_cumprod.cpp (op-def-derived
    # signature x/axisTensor/exclusive/reverse/out) collides with our official-signature
    # implementation in ../op_api.
    target_sources(op_host_aclnnExc PRIVATE
    cumprod_def.cpp
    )
    # Hand-written op_api with the official aclnnCumprod signatures (aclScalar dim +
    # aclDataType dtype): torch_npu's torch.cumprod adapter calls these symbols, so the
    # op-def-derived autogen aclnn must not be generated nor exported.
    add_modules_sources(OP_API_INDEPENDENT ON
                        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
                        OPTYPE cumprod
                        ACLNNTYPE aclnn_exclude)
    enable_pypto_kernel(cumprod)
endif()

set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    # PyPTO kernel: runs codegen at configure time and exposes PYPTO_GEN_DIR for the host tiling -I below.
    # Must be called BEFORE add_modules_sources(...) (which reads PYPTO_GEN_DIR).
    add_ops_compile_options(
        OP_NAME Cumprod
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
        OP_NAME Cumprod
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
    )
else()
    add_ops_compile_options(
        OP_NAME Cumprod
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OP_API_INDEPENDENT ON
                        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
                        OPTYPE cumprod
                        ACLNNTYPE aclnn_exclude)
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_tiling_modules()
endif()
