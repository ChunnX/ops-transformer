# Copyright (c) 2026, edconeone
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Adapted from DeepEP; original copyright and license notices follow.

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
foreach(sdk ASCEND_HOME_PATH SHMEM_ROOT)
    if(NOT IS_DIRECTORY "${${sdk}}")
        message(FATAL_ERROR "${sdk} must identify an installed SDK")
    endif()
endforeach()
set(cann_library_paths "${ASCEND_HOME_PATH}/lib64"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64")
foreach(name MC2_ASCENDCL_LIBRARY MC2_ASCEND_RUNTIME_LIBRARY MC2_SHMEM_LIBRARY)
    unset(${name} CACHE)
endforeach()
find_library(MC2_ASCENDCL_LIBRARY ascendcl PATHS ${cann_library_paths} NO_DEFAULT_PATH REQUIRED)
find_library(MC2_ASCEND_RUNTIME_LIBRARY runtime PATHS ${cann_library_paths} NO_DEFAULT_PATH REQUIRED)
find_library(MC2_SHMEM_LIBRARY shmem PATHS "${SHMEM_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
add_library(mc2_ascendcl SHARED IMPORTED GLOBAL)
set_target_properties(mc2_ascendcl PROPERTIES IMPORTED_LOCATION "${MC2_ASCENDCL_LIBRARY}")
add_library(mc2_shmem SHARED IMPORTED GLOBAL)
set_target_properties(mc2_shmem PROPERTIES IMPORTED_LOCATION "${MC2_SHMEM_LIBRARY}")
