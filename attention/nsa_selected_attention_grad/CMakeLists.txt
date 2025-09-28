# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME NsaSelectedAttentionGrad
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnnExc PRIVATE
        op_host/nsa_selected_attention_grad_def.cpp
)

target_sources(opapi PRIVATE
        op_host/nsa_selected_attention_grad.cpp
        op_host/aclnn_nsa_selected_attention_grad.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(aclnn_ops_train PRIVATE
        op_host/nsa_selected_attention_grad.cpp
        op_host/aclnn_nsa_selected_attention_grad.cpp
    )

    target_sources(aclnn_ops_infer PRIVATE
        op_host/nsa_selected_attention_grad.cpp
        op_host/aclnn_nsa_selected_attention_grad.cpp
    )
endif ()

target_sources(optiling PRIVATE
        op_host/nsa_selected_attention_grad_tiling.cpp
        op_host/nsa_selected_attention_grad_tiling_common.cpp
        op_host/nsa_selected_attention_grad_tiling_bs1.cpp
        op_host/nsa_selected_attention_grad_tiling_bs1_basic.cpp
)

target_sources(opsproto PRIVATE
        op_host/nsa_selected_attention_grad_infershape.cpp
)

file(GLOB _NsaSelectedAttentionGrad_Aclnn_header "${CMAKE_CURRENT_SOURCE_DIR}/op_host/aclnn_*.h")

install(FILES ${_NsaSelectedAttentionGrad_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)