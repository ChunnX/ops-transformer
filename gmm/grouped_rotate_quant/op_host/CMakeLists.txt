# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is licensed under the CANN Open Software License Agreement Version 2.0.
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE grouped_rotate_quant_def.cpp)
endif()

if(BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(OP_NAME GroupedRotateQuant OPTIONS --cce-auto-sync=on -Werror)
else()
    add_modules_sources(
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OPTYPE grouped_rotate_quant ACLNNTYPE aclnn_exclude)
endif()
