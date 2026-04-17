# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB TEST_SRC_FILES "*.cpp")

add_executable(test_mega_moe_kernel ${TEST_SRC_FILES})

target_link_libraries(test_mega_moe_kernel PRIVATE
    gtest
    gtest_main
    tikicpulib
    ${ASCEND_CANN_PACKAGE_PATH}/lib64/libascendcl.so
)

target_include_directories(test_mega_moe_kernel PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../moe_init_routing_v3/op_kernel/arch35
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../moe_finalize_routing_v2/op_kernel/arch35
    ${ASCEND_CANN_PACKAGE_PATH}/include
)

add_test(NAME test_mega_moe_kernel COMMAND test_mega_moe_kernel)