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

set(_MlaPreprocess_TilingSourcesExt
        ${OPS_TRANSFORMER_DIR}/attention/mla_preprocess/op_host/mla_preprocess_tiling.cpp
)

set(_MlaPreprocess_KernelTilingDataDefH
       ${OPS_TRANSFORMER_DIR}/common/include/tiling_base/data_copy_transpose_tiling_def.h
        ${CMAKE_SOURCE_DIR}/attention/mla_preprocess/op_host/mla_preprocess_tilingdata.h
)

set(_MlaPreprocess_TargetPrivateLinkLibrariesExt
        ${UTest_NamePrefix}_OpTiling
        error_manager
)

set(_MlaPreprocess_KernelPrivateCompileDefinitionsExt
        KernelCtrlParam mla_preprocess,mla_preprocess_FIAS fp16 ORIG_DTYPE_INPUT=DT_FLOAT16
        KernelCtrlParam mla_preprocess,mla_preprocess_FIAS bf16 ORIG_DTYPE_INPUT=DT_BF16
)

aux_source_directory(${CMAKE_SOURCE_DIR}/attention/mla_preprocess/op_kernel _MlaPreprocess_KernelSourcesExt)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   MlaPreprocess
        SNAKE                                   mla_preprocess
        KERNEL_TILING_DATA_DEF_H                ${_MlaPreprocess_KernelTilingDataDefH}
        KERNEL_SOURCES_EXT                      ${_MlaPreprocess_KernelSourcesExt}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_MlaPreprocess_KernelPrivateCompileDefinitionsExt}
        TILING_SOURCES_EXT                      ${_MlaPreprocess_TilingSourcesExt}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_MlaPreprocess_TargetPrivateLinkLibrariesExt}
)
