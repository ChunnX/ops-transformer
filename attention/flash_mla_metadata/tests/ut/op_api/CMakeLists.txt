# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_API_UT)
    if(NOT "${ASCEND_OP_NAME}" STREQUAL "ALL" AND NOT "flash_mla_metadata" IN_LIST ASCEND_OP_NAME)
        return()
    endif()
    add_modules_ut_sources(UT_NAME ${OP_API_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    # Keep platform fixtures and the API references they wrap in the same UT library.
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/flash_mla_metadata_platform_fixture.cpp)
    foreach(FLASH_MLA_METADATA_UT_SYMBOL IN ITEMS
        _ZNK2op12PlatformInfo16GetPlatformInfosEv
        _ZNK2op12PlatformInfo14GetCubeCoreNumEv
        _ZNK2op12PlatformInfo16GetVectorCoreNumEv)
        target_link_options(${OPHOST_NAME}_opapi_obj INTERFACE
            "LINKER:--wrap=${FLASH_MLA_METADATA_UT_SYMBOL}")
    endforeach()
endif()
