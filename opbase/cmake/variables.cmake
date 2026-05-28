# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OPS_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(OPS_BASE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/pkg_inc)
if(NOT CANN_3RD_LIB_PATH)
    set(CANN_3RD_LIB_PATH ${PROJECT_SOURCE_DIR}/third_party)
endif()

if (NOT CANN_3RD_PKG_PATH)
    set(CANN_3RD_PKG_PATH ${PROJECT_SOURCE_DIR}/third_party/pkg)
endif ()

file(GLOB_RECURSE OPS_BASE_SRC CACHE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op_common/op_host/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op_common/op_host/util/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op_common/log/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op_common/atvoss/elewise/elewise_tiling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op_common/atvoss/broadcast/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op_common/atvoss/reduce/reduce_tiling.cpp
)

file(GLOB_RECURSE OPS_BASE_INFER_SRC CACHE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op_common/op_host/infershape_broadcast_util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op_common/op_host/infershape_elewise_util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op_common/op_host/infershape_reduce_util.cpp
)

file(GLOB_RECURSE OPS_BASE_TILING_SRC CACHE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op_common/atvoss/elewise/elewise_tiling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op_common/atvoss/broadcast/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op_common/atvoss/reduce/reduce_tiling.cpp
)

file(GLOB_RECURSE OPS_BASE_UTIL_SRC CACHE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op_common/op_host/util/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/op_common/log/*.cpp
)