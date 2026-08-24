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
 * \file mem_format.h
 * \brief 公共内存格式适配模板（与算子逻辑无关的布局 → 格式/序列模式编译期映射）。
 *        模板族原散落于各算子的 memory_copy_arch35.h（与 fused_infer_attention_score 的
 *        fia_block_cube_nonquant_mla.h 等逐字一致），与 gm_format.h（GM 格式推导）同性质：
 *        下沉 common 供各 attention 算子复用，算子侧只 include 本头。
 */
#ifndef MEM_FORMAT_H
#define MEM_FORMAT_H

#include "kernel_operator_list_tensor_intf.h"
#include "gm_layout.h"
#include "fa_ub_tensor.h"
#include "parser.h"

namespace BaseApi {

template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr ActualSeqLensMode GetQActSeqMode()
{
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND || LAYOUT == LayOutTypeEnum::LAYOUT_NTD) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}

template <LayOutTypeEnum LAYOUT, const bool PAGE_ATTENTION>
__aicore__ inline constexpr ActualSeqLensMode GetKvActSeqMode()
{
    if constexpr (PAGE_ATTENTION) {
        return ActualSeqLensMode::BY_BATCH;
    }
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND || LAYOUT == LayOutTypeEnum::LAYOUT_NTD) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}

template <LayOutTypeEnum LAYOUT, bool useDn = false, bool isPerTokenHead = false>
__aicore__ inline constexpr GmFormat GetQueryScaleGmFormat()
{
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH || LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND || LAYOUT == LayOutTypeEnum::LAYOUT_NTD) {
        if constexpr (isPerTokenHead) {
            return GmFormat::NGT;
        }
        if constexpr (!useDn) {
            return GmFormat::NTGD;
        } else {
            return GmFormat::TNGD;
        }
    } else {
        return GmFormat::TNGD;
    }
}

template <LayOutTypeEnum LAYOUT, uint8_t kvLayoutType = 0, bool isPa = false>
__aicore__ inline constexpr GmFormat GetKeyScaleGmFormat()
{
    if constexpr (kvLayoutType == 0) { // KvLayoutType_NO_PA
        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
            return GmFormat::SBND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
            return GmFormat::TND;
        } else {
            return GmFormat::NTD;
        }
    } else if constexpr (kvLayoutType == 1) { // KvLayoutType_PA_BBH
        return GmFormat::PA_BnBsND;
    } else if constexpr (kvLayoutType == 2) { // KvLayoutType_PA_BNBD
        return GmFormat::PA_BnNBsD;
    } else {
        return GmFormat::PA_NZ_K_SCALE;
    }
}

template <LayOutTypeEnum LAYOUT, uint8_t kvLayoutType = 0, bool isPa = false>
__aicore__ inline constexpr GmFormat GetValueScaleGmFormat()
{
    if constexpr (kvLayoutType == 0) { // KvLayoutType_NO_PA
        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
            return GmFormat::SBND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
            return GmFormat::TND2;
        } else {
            return GmFormat::NTD;
        }
    } else if constexpr (kvLayoutType == 1) { // KvLayoutType_PA_BBH
        return GmFormat::PA_BnBsND;
    } else if constexpr (kvLayoutType == 2) { // KvLayoutType_PA_BNBD
        return GmFormat::PA_BnNBsD;
    } else {
        return GmFormat::PA_NZ;
    }
}

template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr UbFormat GetOutUbFormat()
{
    static_assert((LAYOUT == LayOutTypeEnum::LAYOUT_BSH) || (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) ||
                      (LAYOUT == LayOutTypeEnum::LAYOUT_TND) || (LAYOUT == LayOutTypeEnum::LAYOUT_NTD),
                  "Get OutAttention UB GmFormat fail, LAYOUT is incorrect");
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH || LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
        return UbFormat::S1G;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD || LAYOUT == LayOutTypeEnum::LAYOUT_NTD) {
        return UbFormat::GS1;
    }
}

template <LayOutTypeEnum LAYOUT>
__aicore__ inline uint64_t SeqLenFromTensorList(__gm__ uint8_t *keyPtr, uint32_t bIndex)
{
    uint64_t dimInfo[4]; // this mem is used to set shapeinfo, BSH(3) or BNSD(4)
    AscendC::TensorDesc<__gm__ uint8_t> keyTensorDesc;
    ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
    keyTensorDesc.SetShapeAddr(&dimInfo[0]);
    keyListTensorDesc.GetDesc(keyTensorDesc, bIndex);
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) { // v100 没有bsnd
        return keyTensorDesc.GetShape(1);                 // BSH, idx of s is 1
    } else {
        return keyTensorDesc.GetShape(2); // BNSD, idx of s is 2
    }
}

} // namespace BaseApi

#endif // MEM_FORMAT_H
