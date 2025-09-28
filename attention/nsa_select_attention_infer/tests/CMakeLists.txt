# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


########################################################################################################################
# 调用编译方法, 生成对应编译目标
########################################################################################################################

set(_NSA_SELECT_ATTENTION_INFER_TilingSourcesExt
        ${OPS_TRANSFORMER_DIR}/attention/nsa_select_attention_infer/op_host/nsa_select_attention_infer_tiling.cpp
        ${OPS_TRANSFORMER_DIR}/attention/nsa_select_attention_infer/op_host/nsa_select_attention_infer_tiling_check.cpp
        ${OPS_TRANSFORMER_DIR}/attention/nsa_select_attention_infer/op_host/nsa_select_attention_infer_tiling_register.cpp
)

set(_NSA_SELECT_ATTENTION_INFER_KernelTilingDataDefH
        ${OPS_TRANSFORMER_DIR}/common/include/tiling_base/data_copy_transpose_tiling_def.h
        ${CMAKE_SOURCE_DIR}/attention/nsa_select_attention_infer/op_host/nsa_select_attention_infer_tiling.h
)

set(_NsaSelectedAttention_TilingPrivateIncludesExt
        ${CMAKE_SOURCE_DIR}/attention/nsa_select_attention_infer/op_host/
        ${CMAKE_SOURCE_DIR}/attention/nsa_select_attention_infer/op_host/op_api/
)

set(_NSA_SELECT_ATTENTION_INFER_UTestCommonPrivateIncludeExt
        ${OPBASE_INC_DIRS}
        ${OPS_TRANSFORMER_DIR}/common/include
)

set(_NSA_SELECT_ATTENTION_INFER_TargetPrivateLinkLibrariesExt)
set(_NSA_SELECT_ATTENTION_INFER_CompileDefintions
        ORIG_DTYPE_QUERY=DT_FLOAT16
)
aux_source_directory(${CMAKE_SOURCE_DIR}/attention/nsa_select_attention_infer/op_kernel _NSA_SELECT_ATTENTION_INFER_KernelSourcesExt)
OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   NsaSelectAttentionInfer
        SNAKE                                   nsa_select_attention_infer
        TILING_SOURCES_EXT                      ${_NSA_SELECT_ATTENTION_INFER_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_NsaSelectedAttention_TilingPrivateIncludesExt}
        KERNEL_SOURCES_EXT                      ${_NSA_SELECT_ATTENTION_INFER_KernelSourcesExt}
        KERNEL_TILING_DATA_DEF_H                ${_NSA_SELECT_ATTENTION_INFER_KernelTilingDataDefH}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_NSA_SELECT_ATTENTION_INFER_TargetPrivateLinkLibrariesExt}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_NSA_SELECT_ATTENTION_INFER_CompileDefintions}
        UTEST_COMMON_PRIVATE_INCLUDES_EXT       ${_NSA_SELECT_ATTENTION_INFER_UTestCommonPrivateIncludeExt}
)
