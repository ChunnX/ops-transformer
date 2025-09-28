# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

########################################################################################################################
# 调用编译方法, 生成对应编译目标
########################################################################################################################

set(_Mla_Prolog_TilingSourcesExt
        ${OPS_TRANSFORMER_DIR}/attention/mla_prolog/op_host/mla_prolog_tiling.cpp
        ${OPS_TRANSFORMER_DIR}/attention/mla_prolog/op_host/mla_prolog_tiling_check.cpp
        ${OPS_TRANSFORMER_DIR}/attention/mla_prolog/op_host/mla_prolog_tiling_register.cpp
)

set(_Mla_Prolog_KernelTilingDataDefH
        ${OPS_TRANSFORMER_DIR}/common/include/tiling_base/data_copy_transpose_tiling_def.h
        ${CMAKE_SOURCE_DIR}/attention/mla_prolog/op_host/mla_prolog_tiling.h
)

set(_Mla_Prolog_TargetPrivateLinkLibrariesExt
        ${UTest_NamePrefix}_OpTiling
        error_manager
)

aux_source_directory(${CMAKE_SOURCE_DIR}/attention/mla_prolog/op_kernel _Mla_Prolog_KernelSourcesExt)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   MlaProlog
        SNAKE                                   mla_prolog
        KERNEL_TILING_DATA_DEF_H                ${_Mla_Prolog_KernelTilingDataDefH}
        KERNEL_SOURCES_EXT                      ${_Mla_Prolog_KernelSourcesExt}
        KERNEL_PRIVATE_INCLUDES_EXT             ${OPS_ADV_DIR}/common/include/kernel
        TILING_SOURCES_EXT                      ${_Mla_Prolog_TilingSourcesExt}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_Mla_Prolog_TargetPrivateLinkLibrariesExt}
)
