# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include_guard(GLOBAL)

if(NOT CANN_3RD_LIB_PATH)
  set(CANN_3RD_LIB_PATH "${PROJECT_SOURCE_DIR}/third_party")
endif()

set(CATLASS_VENDOR_GIT_TAG
    "41bf90da655bba3c66d0acd7e00abe33960ecfd6"
    CACHE STRING "Catlass vendor git tag")

set(_CATLASS_SOURCE_DIR "${CANN_3RD_LIB_PATH}/catlass")
set(_CATLASS_MARK "${_CATLASS_SOURCE_DIR}/include/catlass/catlass.hpp")

if(NOT EXISTS "${_CATLASS_MARK}")
  include(FetchContent)
  FetchContent_Declare(
    catlass_vendor
    GIT_REPOSITORY https://gitcode.com/cann/catlass.git
    GIT_TAG ${CATLASS_VENDOR_GIT_TAG}
    GIT_PROGRESS TRUE
    SOURCE_DIR "${_CATLASS_SOURCE_DIR}")
  FetchContent_Populate(catlass_vendor)
endif()

if(NOT EXISTS "${_CATLASS_MARK}")
  message(FATAL_ERROR "Catlass headers are missing: ${_CATLASS_MARK}")
endif()

get_filename_component(OPS_ADV_CATLASS_INC "${_CATLASS_SOURCE_DIR}/include" REALPATH)
