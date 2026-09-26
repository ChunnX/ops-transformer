# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# -----------------------------------------------------------------------------------------------------------
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE ai_infra_mega_mhc_def.cpp)
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_op_to_compiled_list()
    add_modules_sources(OPTYPE ai_infra_mega_mhc ACLNNTYPE aclnn_exclude)
endif()
