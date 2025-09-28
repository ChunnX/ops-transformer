/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

cmake_minimum_required(VERSION 3.14)
message(STATUS "Built ops.mc2.communication_and_computation.all_gather_matmul_v2.op_host.op_api")

file(GLOB _opapi_files_all_gather_matmul_v2 aclnn_all_gather_matmul_v2.cpp)
set(all_gather_matmul_v2_include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OPAPI_INCLUDE}
    ${OP_PROTO_INCLUDE}
    ${OP_TILING_INCLUDE}
    ${OP_HOST_INCLUDE}
    ${OP_FALLBACK}
)

if(ENABLE_TEST STREQUAL "")
  if(NOT (("${COMPILER_TYPE}" STREQUAL "llvm") AND ("${PRODUCT_SIDE}" STREQUAL "device")))
    target_sources(opapi PRIVATE ${_opapi_files_all_gather_matmul_v2})
    target_include_directories(opapi PRIVATE ${all_gather_matmul_v2_include})
  endif()
elseif(UT_TEST_ALL OR OP_API_UT)
  target_sources(opapi_llt PRIVATE ${_opapi_files_all_gather_matmul_v2})
  target_include_directories(opapi_llt PRIVATE ${all_gather_matmul_v2_include})
endif()
