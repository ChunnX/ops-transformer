# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(CATLASS_TAG_ID 769cd40a8716b28650b6bebb08db4834eea4462f)

if(EXISTS "${PROJECT_SOURCE_DIR}/../catlass")
  get_filename_component(CATLASS_SOURCE_PATH
                         ${PROJECT_SOURCE_DIR}/../catlass REALPATH)
  message(STATUS "Find catlass source dir: ${CATLASS_SOURCE_PATH}")
elseif(EXISTS "${CANN_3RD_LIB_PATH}/catlass")
  get_filename_component(CATLASS_SOURCE_PATH
                         ${CANN_3RD_LIB_PATH}/catlass REALPATH)
  message(STATUS "Find catlass source dir: ${CATLASS_SOURCE_PATH}")

  execute_process(
    COMMAND git fetch origin ${CATLASS_TAG_ID} --depth=1
    WORKING_DIRECTORY ${CATLASS_SOURCE_PATH}
    TIMEOUT 20
    RESULT_VARIABLE FETCH_RESULT
    ERROR_VARIABLE FETCH_ERROR
    OUTPUT_QUIET
  )

  if(FETCH_RESULT EQUAL 0)
    set(CHECKOUT_REF "FETCH_HEAD")
    message(STATUS "catlass fetched ${CATLASS_TAG_ID}")
  else()
    set(CHECKOUT_REF "${CATLASS_TAG_ID}")
    message(WARNING "Git fetch failed (network unavailable?), checkout to cached ${CATLASS_TAG_ID}")
  endif()

  execute_process(
    COMMAND git checkout ${CHECKOUT_REF}
    WORKING_DIRECTORY ${CATLASS_SOURCE_PATH}
    RESULT_VARIABLE CHECKOUT_RESULT
    ERROR_VARIABLE CHECKOUT_ERROR
    OUTPUT_QUIET
  )
  if(NOT CHECKOUT_RESULT EQUAL 0)
    message(FATAL_ERROR "Git checkout failed: ${CHECKOUT_ERROR}")
  endif()
else()
  set(CATLASS_GIT_URL "git@gitcode.com:cann/catlass.git")
  set(CATLASS_FALLBACK_GIT_URL "https://gitcode.com/cann/catlass.git")

  execute_process(
    COMMAND git ls-remote ${CATLASS_GIT_URL} HEAD
    TIMEOUT 20
    RESULT_VARIABLE GIT_RESULT
    ERROR_VARIABLE GIT_ERROR
    OUTPUT_QUIET
  )
  if(NOT "${GIT_RESULT}" STREQUAL "0")
    message(WARNING
            "[ThirdPartyLib][catlass] failed to access ${CATLASS_GIT_URL}, "
            "trying fallback protocol: ${CATLASS_FALLBACK_GIT_URL}")
    execute_process(
      COMMAND git ls-remote ${CATLASS_FALLBACK_GIT_URL} HEAD
      TIMEOUT 20
      RESULT_VARIABLE FALLBACK_GIT_RESULT
      ERROR_VARIABLE FALLBACK_GIT_ERROR
      OUTPUT_QUIET
    )
    if(NOT "${FALLBACK_GIT_RESULT}" STREQUAL "0")
      message(FATAL_ERROR
              "[ThirdPartyLib][catlass] failed to access catlass with both protocols.\n"
              "Primary (${CATLASS_GIT_URL}): ${GIT_ERROR}\n"
              "Fallback (${CATLASS_FALLBACK_GIT_URL}): ${FALLBACK_GIT_ERROR}")
    endif()
    set(CATLASS_GIT_URL "${CATLASS_FALLBACK_GIT_URL}")
  endif()
  message(STATUS "[ThirdPartyLib][catlass] using repository: ${CATLASS_GIT_URL}")

  include(FetchContent)

  FetchContent_Declare(
    catlass
    GIT_REPOSITORY ${CATLASS_GIT_URL}
    GIT_TAG ${CATLASS_TAG_ID}
    GIT_PROGRESS TRUE
    SOURCE_DIR ${CANN_3RD_LIB_PATH}/catlass)

  FetchContent_Populate(catlass)

  set(CATLASS_SOURCE_PATH ${CANN_3RD_LIB_PATH}/catlass)
endif()

# catlass 源码根目录（包含 include/catlass 头文件），供 kernel 编译拷贝与安装使用
set(CATLASS ${CATLASS_SOURCE_PATH})
set(CATLASS_DIR ${CATLASS_SOURCE_PATH})
