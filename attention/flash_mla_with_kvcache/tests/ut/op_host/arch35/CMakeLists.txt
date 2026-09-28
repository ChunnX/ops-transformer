# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


if(UT_TEST_ALL OR OP_HOST_UT)
    add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()
if(TARGET ${OP_TILING_MODULE_NAME}_cases_obj AND "arch35" IN_LIST ARCH_DIRECTORY)
    # Host shared libraries hide checker symbols. Compile their real sources into
    # the UT archive as well, following sparse_flash_mla's checker UT convention.
    file(GLOB MLA_CHECKER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/checkers/*checker*.cpp)
    target_sources(${OP_TILING_MODULE_NAME}_cases_obj PRIVATE
        ${MLA_CHECKER_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/flash_mla_with_kvcache_tiling_info_parser.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../common/op_host/fia_tiling_shape.cpp)
    # Limit access-control relaxation to the checker regression translation unit.
    set_source_files_properties(test_flash_mla_checkers_tiling.cpp
        TARGET_DIRECTORY ${OP_TILING_MODULE_NAME}_cases_obj
        PROPERTIES COMPILE_OPTIONS "-fno-access-control")
endif()
