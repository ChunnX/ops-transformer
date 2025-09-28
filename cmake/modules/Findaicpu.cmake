# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(aicpu_FOUND)
  message(STATUS "aicpu has been found")
  return()
endif()

include(FindPackageHandleStandardArgs)
set(MSPROF_HEAD_SEARCH_PATHS
  ${ASCEND_DIR}/${SYSTEM_PREFIX}/include/experiment/msprof/
  ${TOP_DIR}/abl/msprof/inc            # compile with ci
)

find_path(MSPROF_INC_DIR
  NAMES toolchain/prof_api.h
  PATHS ${MSPROF_HEAD_SEARCH_PATHS}
)

set(CCE_HEAD_SEARCH_PATHS
  ${ASCEND_DIR}/${SYSTEM_PREFIX}/include/experiment/
  ${TOP_DIR}/ace/comop/inc            # compile with ci
)

find_path(CCE_INC_DIR
  NAMES cce/aicpu_engine_struct.h
  PATHS ${CCE_HEAD_SEARCH_PATHS}
)

message(STATUS "Found aicpu include:${MSPROF_INC_DIR}, ${CCE_INC_DIR}")
set(AICPU_INC_DIRS ${MSPROF_INC_DIR} ${CCE_INC_DIR})