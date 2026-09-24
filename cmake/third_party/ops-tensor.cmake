# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
set(OPTENSOR_TAG_ID 88970a817ccd23653c69e7fc06a648dbf81dfecb)

if(EXISTS "${PROJECT_SOURCE_DIR}/../ops-tensor")
  get_filename_component(OPTENSOR_SOURCE_PATH
                         ${PROJECT_SOURCE_DIR}/../ops-tensor REALPATH)
  message(STATUS "Find ops-tensor source dir: ${OPTENSOR_SOURCE_PATH}")
  get_filename_component(TENSOR_API
                         ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc REALPATH)
elseif(EXISTS "${CANN_3RD_LIB_PATH}/ops-tensor")
  get_filename_component(OPTENSOR_SOURCE_PATH
                         ${CANN_3RD_LIB_PATH}/ops-tensor REALPATH)
  message(STATUS "Find ops-tensor source dir: ${OPTENSOR_SOURCE_PATH}")

  # Use the existing local checkout as-is; do not fetch or switch commits.
  # Submodule: local cache only, no network fetch.
  execute_process(
    COMMAND git submodule update --init --recursive --no-fetch
    WORKING_DIRECTORY ${OPTENSOR_SOURCE_PATH}
    RESULT_VARIABLE SUBMODULE_RESULT
    OUTPUT_QUIET ERROR_QUIET
  )
  if(NOT SUBMODULE_RESULT EQUAL 0)
    message(WARNING "ops-tensor submodule local cache insufficient; network fetch disabled")
  endif()
  get_filename_component(TENSOR_API
                         ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc REALPATH)
else()
  set(OPTENSOR_GIT_URL "git@gitcode.com:cann/ops-tensor.git")
  set(OPTENSOR_FALLBACK_GIT_URL "https://gitcode.com/cann/ops-tensor.git")

  execute_process(
    COMMAND git ls-remote ${OPTENSOR_GIT_URL} HEAD
    TIMEOUT 20
    RESULT_VARIABLE GIT_RESULT
    ERROR_VARIABLE GIT_ERROR
    OUTPUT_QUIET
  )
  if(NOT "${GIT_RESULT}" STREQUAL "0")
    message(WARNING
            "[ThirdPartyLib][ops-tensor] failed to access ${OPTENSOR_GIT_URL}, "
            "trying fallback protocol: ${OPTENSOR_FALLBACK_GIT_URL}")
    execute_process(
      COMMAND git ls-remote ${OPTENSOR_FALLBACK_GIT_URL} HEAD
      TIMEOUT 20
      RESULT_VARIABLE FALLBACK_GIT_RESULT
      ERROR_VARIABLE FALLBACK_GIT_ERROR
      OUTPUT_QUIET
    )
    if(NOT "${FALLBACK_GIT_RESULT}" STREQUAL "0")
      message(FATAL_ERROR
              "[ThirdPartyLib][ops-tensor] failed to access ops-tensor with both protocols.\n"
              "Primary (${OPTENSOR_GIT_URL}): ${GIT_ERROR}\n"
              "Fallback (${OPTENSOR_FALLBACK_GIT_URL}): ${FALLBACK_GIT_ERROR}")
    endif()
    set(OPTENSOR_GIT_URL "${OPTENSOR_FALLBACK_GIT_URL}")
  endif()
  message(STATUS "[ThirdPartyLib][ops-tensor] using repository: ${OPTENSOR_GIT_URL}")

  include(FetchContent)

  FetchContent_Declare(
    ops-tensor
    GIT_REPOSITORY ${OPTENSOR_GIT_URL}
    GIT_TAG ${OPTENSOR_TAG_ID}
    GIT_PROGRESS TRUE
    SOURCE_DIR ${CANN_3RD_LIB_PATH}/ops-tensor)

  FetchContent_Populate(ops-tensor)

  set(OPTENSOR_SOURCE_PATH ${CANN_3RD_LIB_PATH}/ops-tensor)

  execute_process(
    COMMAND git submodule update --init --recursive
    WORKING_DIRECTORY ${OPTENSOR_SOURCE_PATH}
    RESULT_VARIABLE SUBMODULE_RESULT
    ERROR_VARIABLE SUBMODULE_ERROR
    OUTPUT_QUIET
  )
  if(NOT SUBMODULE_RESULT EQUAL 0)
    message(FATAL_ERROR "[ThirdPartyLib][ops-tensor] Git submodule update failed: ${SUBMODULE_ERROR}")
  else()
    message(STATUS "[ThirdPartyLib][ops-tensor] Git submodule update")
    get_filename_component(TENSOR_API
                           ${OPTENSOR_SOURCE_PATH}/include/tensor_api REALPATH)
  endif()

endif()
