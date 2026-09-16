# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_HOST_UT)
    add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    list(FIND ARCH_DIRECTORY "arch35" QBSA_ARCH35_INDEX)
    if(NOT QBSA_ARCH35_INDEX EQUAL -1 AND TARGET ${OP_TILING_MODULE_NAME}_cases_obj)
        file(GLOB_RECURSE QBSA_ARCH35_TILING_CASES CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/qbsa_fp8/test_*_tiling.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/qbsa_mxfp8/test_*_tiling.cpp
        )
        target_sources(${OP_TILING_MODULE_NAME}_cases_obj PRIVATE ${QBSA_ARCH35_TILING_CASES})
    endif()
endif()
