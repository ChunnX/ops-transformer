# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        sliding_tile_attention_def.cpp
    )
endif()

add_modules_sources(OPTYPE sliding_tile_attention ACLNNTYPE aclnn)
