# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


########################################################################################################################
# 调用编译方法, 生成对应编译目标
########################################################################################################################

set(_MlaPreprocessV2_TilingSourcesExt
        ${OPS_TRANSFORMER_DIR}/attention/mla_preprocess_v2/op_host/mla_preprocess_v2_tiling.cpp
)

set(_MlaPreprocessV2_KernelTilingDataDefH
       ${OPS_TRANSFORMER_DIR}/common/include/tiling_base/data_copy_transpose_tiling_def.h
        ${CMAKE_SOURCE_DIR}/attention/mla_preprocess/op_host/mla_preprocess_tilingdata.h
)

set(_MlaPreprocessV2_TargetPrivateLinkLibrariesExt
        ${UTest_NamePrefix}_OpTiling
        error_manager
)

set(_MlaPreprocessV2_KernelPrivateCompileDefinitionsExt
        KernelCtrlParam mla_preprocess_v2,mla_preprocess_v2_FIAS fp16 ORIG_DTYPE_INPUT=DT_FLOAT16
        KernelCtrlParam mla_preprocess_v2,mla_preprocess_v2_FIAS bf16 ORIG_DTYPE_INPUT=DT_BF16
)

aux_source_directory(${CMAKE_SOURCE_DIR}/attention/mla_preprocess/op_kernel _MlaPreprocessV2_KernelSourcesExt)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   MlaPreprocessV2
        SNAKE                                   mla_preprocess_v2
        KERNEL_TILING_DATA_DEF_H                ${_MlaPreprocessV2_KernelTilingDataDefH}
        KERNEL_SOURCES_EXT                      ${_MlaPreprocessV2_KernelSourcesExt}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_MlaPreprocessV2_KernelPrivateCompileDefinitionsExt}
        TILING_SOURCES_EXT                      ${_MlaPreprocessV2_TilingSourcesExt}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_MlaPreprocessV2_TargetPrivateLinkLibrariesExt}
)
