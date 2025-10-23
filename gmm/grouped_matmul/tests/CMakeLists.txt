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
set(_GMM_TilingSourcesExt
        ${CMAKE_SOURCE_DIR}/gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.cpp
)

set(_GMM_TilingPrivateIncludesExt
        ${CMAKE_SOURCE_DIR}/gmm/grouped_matmul/op_host
)

set(_GMM_KernelTilingDataDefH
        ${CMAKE_SOURCE_DIR}/gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.h
)

set(_GMM_OpApiSourcesExt
        ${CMAKE_SOURCE_DIR}/gmm/grouped_matmul/op_host/op_api/grouped_matmul.cpp
        ${CMAKE_SOURCE_DIR}/gmm/grouped_matmul/op_host/op_api/aclnn_grouped_matmul_util.cpp
        ${CMAKE_SOURCE_DIR}/gmm/grouped_matmul/op_host/op_api/aclnn_grouped_matmul.cpp
)

set(_GMM_OpProtoSourcesExt
        ${CMAKE_SOURCE_DIR}/gmm/grouped_matmul/op_host/grouped_matmul_infershape.cpp
)

aux_source_directory(${CMAKE_SOURCE_DIR}/gmm/grouped_matmul/op_kernel/ _GMM_KernelSourcesExt)
set(_GMM_KernelPrivateCompileDefinitionsExt
        KernelCtrlParam grouped_matmul bf16 DTYPE_X=bfloat16_t DTYPE_WEIGHT=bfloat16_t DTYPE_BIAS=float DTYPE_SCALE=uint64_t DTYPE_Y=bfloat16_t ORIG_DTYPE_X=DT_BF16 ORIG_DTYPE_WEIGHT=DT_BF16 ORIG_DTYPE_Y=DT_BF16
        KernelCtrlParam grouped_matmul fp16 DTYPE_X=half DTYPE_WEIGHT=half DTYPE_BIAS=half DTYPE_SCALE=uint64_t DTYPE_Y=half ORIG_DTYPE_X=DT_FLOAT16 ORIG_DTYPE_WEIGHT=DT_FLOAT16 ORIG_DTYPE_Y=DT_FLOAT16
        KernelCtrlParam grouped_matmul fp32 DTYPE_X=float DTYPE_WEIGHT=float DTYPE_BIAS=float DTYPE_SCALE=uint64_t DTYPE_Y=float  ORIG_DTYPE_X=DT_FLOAT ORIG_DTYPE_WEIGHT=DT_FLOAT ORIG_DTYPE_Y=DT_FLOAT
        KernelCtrlParam grouped_matmul quant_int8 DTYPE_X=int8_t DTYPE_WEIGHT=int8_t DTYPE_BIAS=int32_t DTYPE_SCALE=uint64_t DTYPE_Y=int8_t ORIG_DTYPE_X=DT_INT8 ORIG_DTYPE_WEIGHT=DT_INT8 ORIG_DTYPE_Y=DT_INT8
        KernelCtrlParam grouped_matmul quant_bf16 DTYPE_X=int8_t DTYPE_WEIGHT=int8_t DTYPE_BIAS=int32_t DTYPE_SCALE=bfloat16_t DTYPE_Y=bfloat16_t ORIG_DTYPE_X=DT_INT8 ORIG_DTYPE_WEIGHT=DT_INT8 ORIG_DTYPE_Y=DT_BF16
        KernelCtrlParam grouped_matmul quant_fp16 DTYPE_X=int8_t DTYPE_WEIGHT=int8_t DTYPE_BIAS=int32_t DTYPE_SCALE=float DTYPE_Y=half ORIG_DTYPE_X=DT_INT8 ORIG_DTYPE_WEIGHT=DT_INT8 ORIG_DTYPE_Y=DT_FLOAT16
        KernelCtrlParam grouped_matmul a16w8_bf16 DTYPE_X=bfloat16_t DTYPE_WEIGHT=int8_t DTYPE_BIAS=float DTYPE_SCALE=uint64_t DTYPE_Y=bfloat16_t ORIG_DTYPE_X=DT_BF16 ORIG_DTYPE_WEIGHT=DT_INT8 ORIG_DTYPE_Y=DT_BF16
        KernelCtrlParam grouped_matmul a16w8_fp16 DTYPE_X=half DTYPE_WEIGHT=int8_t DTYPE_BIAS=half DTYPE_SCALE=uint64_t DTYPE_Y=half ORIG_DTYPE_X=DT_FLOAT16 ORIG_DTYPE_WEIGHT=DT_INT8 ORIG_DTYPE_Y=DT_FLOAT16
        KernelCtrlParam grouped_matmul a16w4_bf16 DTYPE_X=bfloat16_t DTYPE_WEIGHT=int4b_t DTYPE_BIAS=float DTYPE_SCALE=uint64_t DTYPE_Y=bfloat16_t ORIG_DTYPE_X=DT_BF16 ORIG_DTYPE_WEIGHT=DT_INT4 ORIG_DTYPE_Y=DT_BF16
        KernelCtrlParam grouped_matmul a16w4_fp16 DTYPE_X=half DTYPE_WEIGHT=int4b_t DTYPE_BIAS=half DTYPE_SCALE=uint64_t DTYPE_Y=half ORIG_DTYPE_X=DT_FLOAT16 ORIG_DTYPE_WEIGHT=DT_INT4 ORIG_DTYPE_Y=DT_FLOAT16
        KernelCtrlParam grouped_matmul a8w4_msd DTYPE_X=int8_t DTYPE_WEIGHT=int4b_t DTYPE_BIAS=float DTYPE_SCALE=uint64_t DTYPE_Y=bfloat16_t ORIG_DTYPE_X=DT_INT8 ORIG_DTYPE_WEIGHT=DT_INT4 ORIG_DTYPE_Y=DT_BF16
        KernelCtrlParam grouped_matmul a4w4 DTYPE_X=int4b_t DTYPE_WEIGHT=int4b_t DTYPE_BIAS=half DTYPE_SCALE=uint64_t DTYPE_Y=half ORIG_DTYPE_X=DT_INT4 ORIG_DTYPE_WEIGHT=DT_INT4 ORIG_DTYPE_Y=DT_FLOAT16
        KernelCtrlParam grouped_matmul quant_int32 DTYPE_X=int8_t DTYPE_WEIGHT=int8_t DTYPE_BIAS=int32_t DTYPE_SCALE=uint64_t DTYPE_Y=int32_t ORIG_DTYPE_X=DT_INT8 ORIG_DTYPE_WEIGHT=DT_INT8 ORIG_DTYPE_Y=DT_INT32
)

set(_GMM_TargetPrivateIncludeExt
        ${CMAKE_SOURCE_DIR}/gmm/grouped_matmul/op_host
        ${CMAKE_SOURCE_DIR}/gmm/grouped_matmul/op_host/op_api
        ${CMAKE_SOURCE_DIR}/common/include
        ${OPBASE_INC_DIRS}
)

set(_GMM_TillingPrivateLinkLibrariesExt
)

set(_GMM_TargetPrivateLinkLibrariesExt
        ${UTest_NamePrefix}_OpTiling 
        error_manager
)

if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
        message(STATUS "The UTest of grouped_matmul don't supported clang compiler yet.")
else ()
        OpsTest_Level2_AddOp(
                SUB_SYSTEM                              transformer
                BRIEF                                   GMM
                SNAKE                                   grouped_matmul
                OPAPI_SOURCES_EXT                       ${_GMM_OpApiSourcesExt}
                PROTO_SOURCES_EXT                       ${_GMM_OpProtoSourcesExt}
                TILING_SOURCES_EXT                      ${_GMM_TilingSourcesExt}
                TILING_PRIVATE_INCLUDES_EXT             ${_GMM_TilingPrivateIncludesExt}
                TILING_PRIVATE_LINK_LIBRARIES_EXT       ${_GMM_TillingPrivateLinkLibrariesExt}
                KERNEL_SOURCES_EXT                      ${_GMM_KernelSourcesExt}
                KERNEL_TILING_DATA_DEF_H                ${_GMM_KernelTilingDataDefH}
                KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_GMM_KernelPrivateCompileDefinitionsExt}
                UTEST_COMMON_PRIVATE_INCLUDES_EXT       ${_GMM_TargetPrivateIncludeExt}
                UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_GMM_TargetPrivateLinkLibrariesExt}
        )
endif()