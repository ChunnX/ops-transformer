# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

include_guard(GLOBAL)

set(OPS_TENSOR_TAG_ID "CHANGE_ID_PLACEHOLDER")
set(OPS_TENSOR_PRIVATE_REPO "PRIVATE_REPO_URL_PLACEHOLDER")

if(EXISTS "${PROJECT_SOURCE_DIR}/../../ops-tensor")
  get_filename_component(OPS_TENSOR_SOURCE_PATH
                         ${PROJECT_SOURCE_DIR}/../../ops-tensor REALPATH)
  message(STATUS "Find ops-tensor source dir: ${OPS_TENSOR_SOURCE_PATH}")
elseif(EXISTS "${CANN_3RD_LIB_PATH}/ops-tensor")
  get_filename_component(OPS_TENSOR_SOURCE_PATH
                         ${CANN_3RD_LIB_PATH}/ops-tensor REALPATH)
  message(STATUS "Find ops-tensor source dir: ${OPS_TENSOR_SOURCE_PATH}")
elseif(DEFINED ENV{OPS_TENSOR_SOURCE_PATH} AND EXISTS "$ENV{OPS_TENSOR_SOURCE_PATH}")
  get_filename_component(OPS_TENSOR_SOURCE_PATH
                         $ENV{OPS_TENSOR_SOURCE_PATH} REALPATH)
  message(STATUS "Find ops-tensor source dir from env: ${OPS_TENSOR_SOURCE_PATH}")
else()
  if(EXISTS "${PROJECT_SOURCE_DIR}/build/_deps/ops-tensor-subbuild")
    file(REMOVE_RECURSE ${PROJECT_SOURCE_DIR}/build/_deps/ops-tensor-subbuild)
  endif()

  include(FetchContent)
  FetchContent_Declare(
    ops_tensor
    GIT_REPOSITORY ${OPS_TENSOR_PRIVATE_REPO}
    GIT_TAG ${OPS_TENSOR_TAG_ID}
    GIT_PROGRESS TRUE
    SOURCE_DIR ${CANN_3RD_LIB_PATH}/ops-tensor)
  FetchContent_Populate(ops_tensor)

  set(OPS_TENSOR_SOURCE_PATH ${CANN_3RD_LIB_PATH}/ops-tensor)
  message(STATUS "Fetch ops-tensor source dir: ${OPS_TENSOR_SOURCE_PATH}")
endif()

