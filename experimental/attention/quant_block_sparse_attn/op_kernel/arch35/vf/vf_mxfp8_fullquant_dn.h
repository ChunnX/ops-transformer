/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vf_mxfp8_fullquant_dn.h
 * \brief MXFP8 full-quant DN VF wrapper for QuantBlockSparseAttn.
 */
#ifndef QUANT_BLOCK_SPARSE_ATTN_VF_MXFP8_FULLQUANT_DN_H_
#define QUANT_BLOCK_SPARSE_ATTN_VF_MXFP8_FULLQUANT_DN_H_

#include "kernel_tensor.h"
#include "common/util_regbase.h"

using namespace regbaseutil;

/*
 * Reuse the common FlashAttention MXFP8 DN VF implementation without replacing BSA's existing FP8 DN VF.
 * The common header uses the same include guard and FaVectorApi namespace as the local FP8 header, so it is
 * included once into an internal namespace and called explicitly from the BSA vector block.
 */
#ifdef MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_H_
#define BSA_RESTORE_LOCAL_DN_VF_GUARD
#endif
#undef MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_H_
#define FaVectorApi FaVectorApiMxFullQuantRef
#include "../../../../../../attention/common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#undef FaVectorApi
#ifdef BSA_RESTORE_LOCAL_DN_VF_GUARD
#define MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_H_
#undef BSA_RESTORE_LOCAL_DN_VF_GUARD
#else
#undef MUL_SEL_SOFTMAXFLASHV2_CAST_NZ_DN_H_
#endif

#endif // QUANT_BLOCK_SPARSE_ATTN_VF_MXFP8_FULLQUANT_DN_H_
