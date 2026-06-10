# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
set(OPBASE_TAG_ID cd48c600c4b37146f504217eab2bf3956f7fee8b)
message(STATUS "[opbase] OPBASE_TAG_ID: ${OPBASE_TAG_ID}")
message(STATUS "[opbase] PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message(STATUS "[opbase] CANN_3RD_LIB_PATH: ${CANN_3RD_LIB_PATH}")

if(EXISTS "${PROJECT_SOURCE_DIR}/../../ops-base")
  message(STATUS "[opbase] Found local ops-base at: ${PROJECT_SOURCE_DIR}/../../ops-base")
  get_filename_component(OPBASE_SOURCE_PATH
                         ${PROJECT_SOURCE_DIR}/../../ops-base REALPATH)
  message(STATUS "[opbase] Find opbase source dir: ${OPBASE_SOURCE_PATH}")
elseif(EXISTS "${CANN_3RD_LIB_PATH}/opbase")
  message(STATUS "[opbase] Found cached opbase at: ${CANN_3RD_LIB_PATH}/opbase")
  get_filename_component(OPBASE_SOURCE_PATH
                         ${CANN_3RD_LIB_PATH}/opbase REALPATH)
  message(STATUS "[opbase] Find opbase source dir: ${OPBASE_SOURCE_PATH}")
  message(STATUS "[opbase] Checking out tag: ${OPBASE_TAG_ID}")
  execute_process(
    COMMAND git checkout ${OPBASE_TAG_ID}
    WORKING_DIRECTORY ${OPBASE_SOURCE_PATH}
    RESULT_VARIABLE OPBASE_CHECKOUT_RESULT
    OUTPUT_VARIABLE OPBASE_CHECKOUT_OUTPUT
    ERROR_VARIABLE OPBASE_CHECKOUT_ERROR
  )
  message(STATUS "[opbase] git checkout result: ${OPBASE_CHECKOUT_RESULT}")
  if(NOT OPBASE_CHECKOUT_RESULT EQUAL 0)
    message(WARNING "[opbase] git checkout failed: ${OPBASE_CHECKOUT_ERROR}")
  endif()
else()
  message(STATUS "[opbase] No local opbase found, will fetch from remote")
  message(STATUS "[opbase] Fetch source: https://gitcode.com/cann/opbase.git")
  message(STATUS "[opbase] Fetch destination: ${CANN_3RD_LIB_PATH}/opbase")
  if(EXISTS "${PROJECT_SOURCE_DIR}/build/_deps/opbase-subbuild")
    message(STATUS "[opbase] Removing stale subbuild dir: ${PROJECT_SOURCE_DIR}/build/_deps/opbase-subbuild")
    file(REMOVE_RECURSE ${PROJECT_SOURCE_DIR}/build/_deps/opbase-subbuild)
  endif()
  include(FetchContent)

  FetchContent_Declare(
    opbase
    GIT_REPOSITORY https://gitcode.com/cann/opbase.git
    GIT_TAG ${OPBASE_TAG_ID}
    GIT_PROGRESS TRUE
    SOURCE_DIR ${CANN_3RD_LIB_PATH}/opbase)

  FetchContent_Populate(opbase)

  set(OPBASE_SOURCE_PATH ${CANN_3RD_LIB_PATH}/opbase)
  message(STATUS "[opbase] FetchContent populated, OPBASE_SOURCE_PATH: ${OPBASE_SOURCE_PATH}")

  if(EXISTS ${OPBASE_SOURCE_PATH}/aicpu_common)
    message(STATUS "[opbase] Removing redundant dir: ${OPBASE_SOURCE_PATH}/aicpu_common")
    file(REMOVE_RECURSE ${OPBASE_SOURCE_PATH}/aicpu_common)
  endif()
  if(EXISTS ${OPBASE_SOURCE_PATH}/pkg_inc/op_common/log)
    message(STATUS "[opbase] Removing redundant dir: ${OPBASE_SOURCE_PATH}/pkg_inc/op_common/log")
    file(REMOVE_RECURSE ${OPBASE_SOURCE_PATH}/pkg_inc/op_common/log)
  endif()
  if(EXISTS ${OPBASE_SOURCE_PATH}/pkg_inc/op_common/op_host)
    message(STATUS "[opbase] Removing redundant dir: ${OPBASE_SOURCE_PATH}/pkg_inc/op_common/op_host")
    file(REMOVE_RECURSE ${OPBASE_SOURCE_PATH}/pkg_inc/op_common/op_host)
  endif()
  if(EXISTS ${OPBASE_SOURCE_PATH}/pkg_inc/op_common/op_kernel)
    message(STATUS "[opbase] Removing redundant dir: ${OPBASE_SOURCE_PATH}/pkg_inc/op_common/op_kernel")
    file(REMOVE_RECURSE ${OPBASE_SOURCE_PATH}/pkg_inc/op_common/op_kernel)
  endif()
endif()

if(EXISTS "${OPBASE_SOURCE_PATH}/include")
  message(STATUS "[opbase] include dir exists: ${OPBASE_SOURCE_PATH}/include")
else()
  message(WARNING "[opbase] include dir NOT found: ${OPBASE_SOURCE_PATH}/include")
endif()
if(EXISTS "${OPBASE_SOURCE_PATH}/include/op_common/log/log.h")
  message(STATUS "[opbase] log.h found: ${OPBASE_SOURCE_PATH}/include/op_common/log/log.h")
else()
  message(WARNING "[opbase] log.h NOT found at: ${OPBASE_SOURCE_PATH}/include/op_common/log/log.h")
endif()
