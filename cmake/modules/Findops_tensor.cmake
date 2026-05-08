# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(ops_tensor_FOUND)
  message(STATUS "ops_tensor has been found")
  return()
endif()

include(FindPackageHandleStandardArgs)

set(OPS_TENSOR_HEAD_SEARCH_PATHS
  ${OPS_TENSOR_SOURCE_PATH}/include
  ${ASCEND_CANN_PACKAGE_PATH}/ops_tensor/include
  ${ASCEND_DIR}/ops_tensor/include
  ${TOP_DIR}/ops-tensor/include
  ${CANN_3RD_LIB_PATH}/ops-tensor/include
)

find_path(OPS_TENSOR_INC_DIR
  NAMES blaze/kernel/kernel_qgmm_mx.h
  PATHS ${OPS_TENSOR_HEAD_SEARCH_PATHS}
  NO_CMAKE_SYSTEM_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

find_package_handle_standard_args(ops_tensor REQUIRED_VARS OPS_TENSOR_INC_DIR)

if(OPS_TENSOR_INC_DIR)
  get_filename_component(OPS_TENSOR_INC_DIR ${OPS_TENSOR_INC_DIR} REALPATH)
  set(OPS_TENSOR_INCLUDE_DIRS
    ${OPS_TENSOR_INC_DIR}
    ${OPS_TENSOR_INC_DIR}/tensor_api
  )
endif()

if(ops_tensor_FOUND AND NOT ops_tensor_FIND_QUIETLY)
  message(STATUS "Found ops_tensor include:${OPS_TENSOR_INC_DIR}")
endif()
