# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# 引入 opbase（cann/opbase）：向前兼容旧版本 CANN toolkit 的公共头（重点为 op_common/log/log.h
# 中的 OP_LOGE_FOR_* / OP_LOGE_LIBOPAPI_REPORT 等新增日志宏，如 EZ0011 OP_LOGE_FOR_INVALID_SHAPEDIM
# 为 9.1.0 后新增接口，9.0.0 基线 toolkit 下编译会失败）。
# 机制与 ops-transformer 主仓 cmake/third_party/opbase.cmake 保持一致：
#   opbase 的 include 路径置于 CANN 包 include 之前，保证新版 log.h 在任意 toolkit 版本下生效。
# 查找优先级（与上游一致）：
#   1) ${PROJECT_SOURCE_DIR}/../../ops-base（上游仓并列检出布局）
#   2) ${CANN_3RD_LIB_PATH}/opbase（预置缓存，存在则 checkout 固定 tag）
#   3) FetchContent 拉取 https://gitcode.com/cann/opbase.git 至 ${CANN_3RD_LIB_PATH}/opbase
# 离线环境可预先手工克隆：
#   git clone https://gitcode.com/cann/opbase.git <工程>/third_party/opbase
#   git -C <工程>/third_party/opbase checkout ec8c387b251655664f4679f1bc5fd30a69efc5a7
# -----------------------------------------------------------------------------------------------------------

if(DEFINED OPBASE_SOURCE_PATH AND EXISTS "${OPBASE_SOURCE_PATH}")
  message(STATUS "opbase already provisioned: ${OPBASE_SOURCE_PATH}")
  return()
endif()

set(OPBASE_TAG_ID ec8c387b251655664f4679f1bc5fd30a69efc5a7)

if(NOT CANN_3RD_LIB_PATH)
  set(CANN_3RD_LIB_PATH ${PROJECT_SOURCE_DIR}/third_party)
endif()

if(EXISTS "${PROJECT_SOURCE_DIR}/../../ops-base")
  get_filename_component(OPBASE_SOURCE_PATH
                         ${PROJECT_SOURCE_DIR}/../../ops-base REALPATH)
  message(STATUS "Find opbase source dir: ${OPBASE_SOURCE_PATH}")
elseif(EXISTS "${CANN_3RD_LIB_PATH}/opbase")
  get_filename_component(OPBASE_SOURCE_PATH
                         ${CANN_3RD_LIB_PATH}/opbase REALPATH)
  message(STATUS "Find opbase source dir: ${OPBASE_SOURCE_PATH}")
  execute_process(
    COMMAND git checkout ${OPBASE_TAG_ID}
    WORKING_DIRECTORY ${OPBASE_SOURCE_PATH}
  )
else()
  if(EXISTS "${PROJECT_SOURCE_DIR}/build/_deps/opbase-subbuild")
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

  # 与上游一致：清理 opbase 中与 CANN 包重复/冲突的内容
  if(EXISTS "${OPBASE_SOURCE_PATH}/aicpu_common")
    file(REMOVE_RECURSE ${OPBASE_SOURCE_PATH}/aicpu_common)
  endif()
  if(EXISTS "${OPBASE_SOURCE_PATH}/pkg_inc/op_common/log")
    file(REMOVE_RECURSE ${OPBASE_SOURCE_PATH}/pkg_inc/op_common/log)
  endif()
  if(EXISTS "${OPBASE_SOURCE_PATH}/pkg_inc/op_common/op_host")
    file(REMOVE_RECURSE ${OPBASE_SOURCE_PATH}/pkg_inc/op_common/op_host)
  endif()
  if(EXISTS "${OPBASE_SOURCE_PATH}/pkg_inc/op_common/op_kernel")
    file(REMOVE_RECURSE ${OPBASE_SOURCE_PATH}/pkg_inc/op_common/op_kernel)
  endif()
endif()

message(STATUS "opbase (log 宏向前兼容) enabled: ${OPBASE_SOURCE_PATH}")
