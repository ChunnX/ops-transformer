# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# find all source files

set(COMPOSITE_OP_PATH ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE NNOPBASE_COMPOSITE_SRCS CONFIGURE_DEPENDS "${COMPOSITE_OP_PATH}/*.cpp")

MESSAGE(STATUS "COMPOSITE_OP_PATH: ${COMPOSITE_OP_PATH}")


# set include path
set(NNOPBASE_COMPOSITE_INCLUDING_PATH
    ${COMPOSITE_OP_PATH}/
    ${OPS_BASE_PATH}/include/nnopbase/

    ${ASCEND_INCLUDE_DIR}/transformer/
    ${ASCEND_INCLUDE_DIR}/exe_graph/
)

target_sources(nnopbase PRIVATE ${NNOPBASE_COMPOSITE_SRCS})
target_include_directories(nnopbase PRIVATE ${NNOPBASE_COMPOSITE_INCLUDING_PATH})