# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 fk2323
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# -----------------------------------------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    set(fmm016_streaming_top_k_blocked_v2_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/fmm016_streaming_top_k_blocked_v2_tiling.cpp)
    AddOpTestCase(fmm016_streaming_top_k_blocked_v2 "Ascend910B1" "" "${fmm016_streaming_top_k_blocked_v2_tiling_files}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel/fmm016_streaming_top_k_blocked_v2.cpp")
endif()
