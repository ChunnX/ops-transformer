# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------------------------------------
# 对齐 dispatch_teardown/op_host/CMakeLists.txt：纯 AIV、非 MC2 算子，带编 op_tiling/op_kernel/op_api。
if (BUILD_OPEN_PROJECT)  # custom
    target_sources(op_host_aclnnInner PRIVATE
        combine_teardown_def.cpp
    )
    add_modules_sources(
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OP_MC2_ENABLE OFF
        OPTYPE combine_teardown combine_teardown ACLNNTYPE aclnn aclnn_inner)
endif()
