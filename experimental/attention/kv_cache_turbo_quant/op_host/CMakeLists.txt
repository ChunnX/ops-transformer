# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        kv_cache_turbo_quant_def.cpp
    )
endif()

add_modules_sources(OPTYPE kv_cache_turbo_quant ACLNNTYPE aclnn)
