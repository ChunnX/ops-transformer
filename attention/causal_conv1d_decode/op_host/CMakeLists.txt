# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------

add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        causal_conv1d_decode_def.cpp
        causal_conv1d_decode_infershape.cpp
        causal_conv1d_decode_tiling.cpp
    )
    add_ops_compile_options(
        OP_NAME CausalConv1dDecode
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_opbase_modules()
    add_modules_sources_with_soc(
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OPTYPE causal_conv1d_decode
        ACLNNTYPE aclnn_exclude
    )
endif()
