# Copyright (c) 2026 fk2323
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the repository for details.

add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        fmm016_streaming_top_k_blocked_v2_def.cpp
    )
endif()

if(BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
        OP_NAME Fmm016StreamingTopKBlockedV2
        OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -Werror
    )
else()
    add_modules_sources(
        OPTYPE fmm016_streaming_top_k_blocked_v2
        ACLNNTYPE aclnn_exclude
    )
endif()
