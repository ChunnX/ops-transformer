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

set(_GmmSwigluQuant_TilingSourcesExt
        ${CMAKE_SOURCE_DIR}/gmm/grouped_matmul_swiglu_quant/op_host/grouped_matmul_swiglu_quant_tiling.cpp
)

set(_GmmSwigluQuant_KernelTilingDataDefH
        ${CMAKE_SOURCE_DIR}/gmm/grouped_matmul_swiglu_quant/op_host/grouped_matmul_swiglu_quant_tiling.h
)

aux_source_directory(${CMAKE_SOURCE_DIR}/gmm/grouped_matmul_swiglu_quant/op_kernel/ _GmmSwigluQuant_KernelSourcesExt)

set(_GmmSwigluQuant_KernelPrivateCompileDefinitionsExt
        KernelCtrlParam grouped_matmul_swiglu_quant,grouped_matmul_swiglu_quant_FIAS fp32 DTYPE_WEIGHT_SCALE=float DTYPE_X=int8_t DTYPE_WEIGHT=int8_t
        KernelCtrlParam grouped_matmul_swiglu_quant,grouped_matmul_swiglu_quant_FIAS fp16 DTYPE_WEIGHT_SCALE=half DTYPE_X=int8_t DTYPE_WEIGHT=int8_t
        KernelCtrlParam grouped_matmul_swiglu_quant,grouped_matmul_swiglu_quant_FIAS bf16 DTYPE_WEIGHT_SCALE=bfloat16_t DTYPE_X=int8_t DTYPE_WEIGHT=int8_t
)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              gmm
        BRIEF                                   GMMSwigluQuant
        SNAKE                                   grouped_matmul_swiglu_quant
        KERNEL_TILING_DATA_DEF_H                ${_GmmSwigluQuant_KernelTilingDataDefH}
        KERNEL_SOURCES_EXT                      ${_GmmSwigluQuant_KernelSourcesExt}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_GmmSwigluQuant_KernelPrivateCompileDefinitionsExt}
        TILING_SOURCES_EXT                      ${_GmmSwigluQuant_TilingSourcesExt}
)
