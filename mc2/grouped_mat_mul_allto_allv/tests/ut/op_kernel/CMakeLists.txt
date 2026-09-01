# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    if("ascend910b" IN_LIST ASCEND_COMPUTE_UNIT)
        set(GMMA2AV_UT_SOC "Ascend910B1")
    else()
        set(GMMA2AV_UT_SOC "Ascend910_9391")
    endif()
    AddOpTestCase(grouped_mat_mul_allto_allv "${GMMA2AV_UT_SOC}" "-DDTYPE_X_REF=half" ""
        "${CMAKE_CURRENT_SOURCE_DIR}/grouped_mat_mul_allto_allv_mte_kt_stub.cpp")
endif()
