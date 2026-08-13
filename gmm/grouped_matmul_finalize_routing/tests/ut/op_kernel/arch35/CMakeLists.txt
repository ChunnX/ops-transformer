# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    set(grouped_matmul_finalize_routing_arch35_kernel
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_kernel/grouped_matmul_finalize_routing_apt.cpp)

    AddOpTestCase(grouped_matmul_finalize_routing "Ascend950PR_9599"
        "-I${PROJECT_SOURCE_DIR}/gmm/common -DDTYPE_X=fp8_e5m2_t -DDTYPE_W=fp8_e5m2_t -DDTYPE_SCALE=fp8_e8m0_t -DDTYPE_PERTOKEN_SCALE=fp8_e8m0_t -DDTYPE_LOGIT=bfloat16_t -DDTYPE_Y=bfloat16_t -DORIG_DTYPE_X=DT_FLOAT8_E5M2 -DORIG_DTYPE_W=DT_FLOAT8_E5M2 -DORIG_DTYPE_Y=DT_BF16"
        ""
        "${grouped_matmul_finalize_routing_arch35_kernel}")
endif()
