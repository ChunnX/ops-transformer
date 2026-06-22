# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------------------------------------
# 对齐 all_gather_matmul_v2/op_host/CMakeLists.txt，裁剪为纯 AIV、非 MC2 算子：
#   - target_sources：把 op 定义加入 op_host_aclnnInner；
#   - add_modules_sources：按 OPTYPE 约定带编 op_tiling / op_kernel / op_api，并从 op 定义
#     生成内层 aclnnInnerDispatchTeardown*（OP_API_DIR 指向 ../op_api 的对外薄壳）；
#   - OP_MC2_ENABLE OFF：本算子不走通信引擎、不依赖 mc2/common（与 all_gather 的 ON 区别）。

if (BUILD_OPEN_PROJECT)  # custom
    target_sources(op_host_aclnnInner PRIVATE
        dispatch_teardown_def.cpp
    )
    add_modules_sources(
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OP_MC2_ENABLE OFF
        OPTYPE dispatch_teardown dispatch_teardown ACLNNTYPE aclnn aclnn_inner)

    # a6-TODO[T3]：状态字跨流可见性对 cache 敏感。all_gather 用了下列编译项做优化；
    #   teardown 既要 done 落盘、又要观测 CCU 的写，dcci 行为需上板核实，故此处先不强加，
    #   用框架默认（如需对齐 all_gather，再放开下面注释）。
    # set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
    # if("${CONDITION_UNIT}" STREQUAL "ascend950")
    #     add_ops_compile_options(
    #         OP_NAME DispatchTeardown
    #         OPTIONS --cce-auto-sync=off
    #                 -mllvm -cce-aicore-dcci-before-kernel-end=false
    #     )
    # endif()
endif()
