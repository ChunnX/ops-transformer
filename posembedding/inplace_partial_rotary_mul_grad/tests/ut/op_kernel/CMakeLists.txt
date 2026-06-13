# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    set(inplace_partial_rotary_mul_grad_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/inplace_partial_rotary_mul_grad_tiling_base_arch35.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/inplace_partial_rotary_mul_grad_tiling_ab_arch35.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/inplace_partial_rotary_mul_grad_tiling_a_and_b_arch35.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/inplace_partial_rotary_mul_grad_tiling_aba_and_ba_arch35.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/inplace_partial_rotary_mul_grad_tiling_bab_arch35.cpp
        )
    AddOpTestCase(inplace_partial_rotary_mul_grad "ascend950" "-DDTYPE_DY=float -DDTYPE_COS=float -DDTYPE_SIN=float" "${inplace_partial_rotary_mul_grad_tiling_files}")
endif()
