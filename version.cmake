# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set_cann_package(ops-transformer VERSION "9.1.0")

set_cann_build_dependencies(runtime ">=8.5")
set_cann_build_dependencies(opbase ">=8.5")
set_cann_build_dependencies(hcomm ">=8.5")
set_cann_build_dependencies(ge-executor ">=8.5")
set_cann_build_dependencies(metadef ">=8.5")
set_cann_build_dependencies(ge-compiler ">=8.5")
set_cann_build_dependencies(asc-devkit ">=8.5")
set_cann_build_dependencies(bisheng-compiler ">=8.5")

set_cann_run_dependencies(runtime ">=8.5")
set_cann_run_dependencies(opbase ">=8.5")
set_cann_run_dependencies(hcomm ">=8.5")
set_cann_run_dependencies(ge-executor ">=8.5")
set_cann_run_dependencies(metadef ">=8.5")
set_cann_run_dependencies(ge-compiler ">=8.5")
set_cann_run_dependencies(bisheng-compiler ">=8.5")
set_cann_run_dependencies(asc-tools ">=8.5")
set_cann_run_dependencies(bisheng-compiler ">=8.5")
set_cann_run_dependencies(ops-nn ">=8.5")
set_cann_run_dependencies(hccl ">=8.5")

# -----------------------------------------------------------------------------------------------------------
# 读取已安装 CANN 包版本（toolkit/version.info 的 Version= 行），供算子级 min_cann_version 过滤使用
# -----------------------------------------------------------------------------------------------------------
set(CANN_TOOLKIT_VERSION "" CACHE STRING "Installed CANN toolkit version")
set(_cann_ver_file "${ASCEND_CANN_PACKAGE_PATH}/opp/version.info")
if(EXISTS "${_cann_ver_file}")
    file(STRINGS "${_cann_ver_file}" _cann_ver_lines)
    foreach(_ver_line IN LISTS _cann_ver_lines)
        if(_ver_line MATCHES "^Version=([0-9]+\\.[0-9]+\\.[0-9]+)")
            set(CANN_TOOLKIT_VERSION "${CMAKE_MATCH_1}" CACHE STRING "Installed CANN toolkit version" FORCE)
            break()
        endif()
    endforeach()
endif()
if(CANN_TOOLKIT_VERSION)
    message(STATUS "Detected CANN toolkit version: ${CANN_TOOLKIT_VERSION}")
else()
    message(WARNING "CANN toolkit version not detected from ${_cann_ver_file}; ops with min_cann_version will be skipped.")
endif()
