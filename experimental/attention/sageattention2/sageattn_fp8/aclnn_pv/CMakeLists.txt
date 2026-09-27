# ---------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------

# aclnn 骨架：统一 FP8 PV 入口（与 PyTorch sageattn_qk_int8_pv_fp8_asc 语义对齐）
set(ACLNN_PV_SOURCES
    op_host/sageattn_pv.cpp
    op_kernel/sageattn_pv_kernel.cpp
)
catlass_example_add_executable(aclnn_sageattn_pv host ${ACLNN_PV_SOURCES})
target_include_directories(aclnn_sageattn_pv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
