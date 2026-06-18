# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)
project(dispatch_setup_ccu)

set(HCCL_CUSTOM_OPS_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(HCCL_CUSTOM_OPS_INCLUDE_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel_ccu

    # CANN Toolkit头文件
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/include/hcomm
    ${ASCEND_CANN_PACKAGE_PATH}/include/hcomm/ccu

    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu
)

# Host 动态链接库
add_library(dispatch_setup_ccu SHARED)
target_include_directories(dispatch_setup_ccu PRIVATE
    ${HCCL_CUSTOM_OPS_INCLUDE_LIST}
)
target_compile_options(dispatch_setup_ccu PRIVATE
    -U_FORTIFY_SOURCE
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g -O0
    -std=c++14
    -fstack-protector-all
)
target_compile_definitions(dispatch_setup_ccu PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    -DHOST_COMPILE
)
target_link_options(dispatch_setup_ccu PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
)
target_link_directories(dispatch_setup_ccu PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)
target_link_libraries(dispatch_setup_ccu PRIVATE
    hcomm
    c_sec
    ascendcl
)
install(TARGETS dispatch_setup_ccu 
    LIBRARY DESTINATION ${CUSTOM_OPS_OPP_LIB_PATH}
    COMPONENT hccl
)
install(FILES
    inc/dispatch_setup.h
    DESTINATION ${CUSTOM_OPS_OPP_INC_PATH}
    COMPONENT hccl
)

# 添加子目录
add_subdirectory(op_host)
add_subdirectory(op_kernel_ccu)
