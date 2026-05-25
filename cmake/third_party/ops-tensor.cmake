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

set(OPTENSOR_TAG_ID gmmtensor)
set(OPTENSOR_GIT_URL "https://gitcode.com/zhangzhizhuo/ops-tensor.git")

if(EXISTS "${PROJECT_SOURCE_DIR}/../ops-tensor")
  get_filename_component(OPTENSOR_SOURCE_PATH
                         ${PROJECT_SOURCE_DIR}/../ops-tensor REALPATH)
  message(STATUS "Find ops-tensor source dir: ${OPTENSOR_SOURCE_PATH}")
elseif(EXISTS "${CANN_3RD_LIB_PATH}/ops-tensor")
  get_filename_component(OPTENSOR_SOURCE_PATH
                         ${CANN_3RD_LIB_PATH}/ops-tensor REALPATH)
  message(STATUS "Find ops-tensor source dir: ${OPTENSOR_SOURCE_PATH}")

  execute_process(
    COMMAND git -c pull.ff=only pull origin ${OPTENSOR_TAG_ID} --depth=1
    WORKING_DIRECTORY ${OPTENSOR_SOURCE_PATH}
    TIMEOUT 10
    RESULT_VARIABLE EXEC_RESULT
    OUTPUT_VARIABLE EXEC_INFO
    ERROR_VARIABLE EXEC_ERROR
  )
  if("${EXEC_RESULT}" STREQUAL "0")
    message(STATUS "ops-tensor updated")
  else()
    message(WARNING "Git pull failed (network unavailable?), using cached ops-tensor code. ${EXEC_ERROR}")
  endif()
else()
  if(EXISTS "${PROJECT_SOURCE_DIR}/build/_deps/ops-tensor-subbuild")
    file(REMOVE_RECURSE ${PROJECT_SOURCE_DIR}/build/_deps/ops-tensor-subbuild)
  endif()

  include(FetchContent)
  FetchContent_Declare(
    ops-tensor
    GIT_REPOSITORY ${OPTENSOR_GIT_URL}
    GIT_TAG ${OPTENSOR_TAG_ID}
    GIT_PROGRESS TRUE
    SOURCE_DIR ${CANN_3RD_LIB_PATH}/ops-tensor)
  FetchContent_Populate(ops-tensor)

  set(OPTENSOR_SOURCE_PATH ${CANN_3RD_LIB_PATH}/ops-tensor)
  message(STATUS "Fetch ops-tensor source dir: ${OPTENSOR_SOURCE_PATH}")
endif()
