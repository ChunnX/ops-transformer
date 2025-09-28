# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#/

cmake_minimum_required(VERSION 3.14)
message(STATUS "Built ops.mc2.communication_and_computation.matmul_reduce_scatter_v2.op_host.op_api")

file(GLOB _opapi_files_matmul_reduce_scatter_v2 aclnn_matmul_reduce_scatter_v2.cpp)
set(matmul_reduce_scatter_v2_include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OPAPI_INCLUDE}
    ${OP_PROTO_INCLUDE}
    ${OP_TILING_INCLUDE}
    ${OP_HOST_INCLUDE}
    ${OP_FALLBACK}
)

if(ENABLE_TEST STREQUAL "")
  if(NOT (("${COMPILER_TYPE}" STREQUAL "llvm") AND ("${PRODUCT_SIDE}" STREQUAL "device")))
    target_sources(opapi PRIVATE ${_opapi_files_matmul_reduce_scatter_v2})
    target_include_directories(opapi PRIVATE ${matmul_reduce_scatter_v2_include})
  endif()
elseif(UT_TEST_ALL OR OP_API_UT)
  target_sources(opapi_llt PRIVATE
    ${_opapi_files_matmul_reduce_scatter_v2}
  )
  target_include_directories(opapi_llt PRIVATE
    ${matmul_reduce_scatter_v2_include}
  )
endif()