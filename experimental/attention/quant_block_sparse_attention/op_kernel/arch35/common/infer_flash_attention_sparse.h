/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file infer_flash_attention_sparse.h
 * \brief
 */
#ifndef INFER_FLASH_ATTENTION_SPARSE_H
#define INFER_FLASH_ATTENTION_SPARSE_H

#include "infer_flash_attention_comm.h"

TEMPLATE_INTF
__aicore__ inline void GetSparseParam(const ConstInfo<isInfer, hasRope> &constInfo,
    const AttenMaskInfo &attenMaskInfo, RunParamStr<isInfer> &runParam)
{
    if constexpr (hasAtten) {
        if constexpr (hasRope && dTemplateType == DTemplateType::Aligned576) {
            runParam.preTokensOfMlaPerBatch = attenMaskInfo.preTokens;
            runParam.nextTokensOfMlaPerBatch = attenMaskInfo.nextTokens;
            if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
                runParam.preTokensPerBatch = SPARSE_MODE_INT_DEFAULT;
                runParam.nextTokensPerBatch = SPARSE_MODE_INT_DEFAULT;
            } else {
                runParam.preTokensPerBatch = runParam.preTokensOfMlaPerBatch * constInfo.gSize;
                runParam.nextTokensPerBatch = runParam.nextTokensOfMlaPerBatch * constInfo.gSize;
            }
        } else {
            runParam.preTokensPerBatch = attenMaskInfo.preTokens;
            runParam.nextTokensPerBatch = attenMaskInfo.nextTokens;
        }
        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
            runParam.preTokensPerBatch = SPARSE_MODE_INT_DEFAULT;
            if constexpr (!(hasRope && (dTemplateType == DTemplateType::Aligned576))) {
                runParam.nextTokensPerBatch = runParam.actualS2Size - runParam.actualS1Size;
                if constexpr (enableKVPrefix) {
                    runParam.nextTokensPerBatch += constInfo.actualKVPrefixSize;
                }
            } else {
                if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
                    runParam.nextTokensPerBatch = SPARSE_MODE_INT_DEFAULT;
                    runParam.nextTokensOfMlaPerBatch = runParam.actualS2Size - runParam.actualSeqLengthOfMlaPerBatch;
                    if constexpr (enableKVPrefix) {
                        runParam.nextTokensOfMlaPerBatch += constInfo.actualKVPrefixSize;
                    }
                } else {
                    runParam.nextTokensOfMlaPerBatch = runParam.actualS2Size - runParam.actualSeqLengthOfMlaPerBatch;
                    if constexpr (enableKVPrefix) {
                        runParam.nextTokensOfMlaPerBatch += constInfo.actualKVPrefixSize;
                    }
                    runParam.nextTokensPerBatch = runParam.nextTokensOfMlaPerBatch * constInfo.gSize;
                }
            }
        }
        if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
            if constexpr (hasRope && dTemplateType == DTemplateType::Aligned576) {
                runParam.preTokensOfMlaPerBatch = attenMaskInfo.preTokens - runParam.actualS2Size + runParam.actualSeqLengthOfMlaPerBatch;
                runParam.nextTokensOfMlaPerBatch = attenMaskInfo.nextTokens + runParam.actualS2Size - runParam.actualSeqLengthOfMlaPerBatch;
                if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
                    runParam.preTokensPerBatch = SPARSE_MODE_INT_DEFAULT;
                    runParam.nextTokensPerBatch = SPARSE_MODE_INT_DEFAULT;
                } else {
                    runParam.preTokensPerBatch = runParam.preTokensOfMlaPerBatch * constInfo.gSize;
                    runParam.nextTokensPerBatch = runParam.nextTokensOfMlaPerBatch * constInfo.gSize;
                }
            } else {
                runParam.preTokensPerBatch = attenMaskInfo.preTokens - runParam.actualS2Size +
                    runParam.actualS1Size;
                runParam.nextTokensPerBatch = attenMaskInfo.nextTokens + runParam.actualS2Size -
                    runParam.actualS1Size;
                if constexpr (enableKVPrefix) {
                    runParam.preTokensPerBatch -= constInfo.actualKVPrefixSize;
                    runParam.nextTokensPerBatch += constInfo.actualKVPrefixSize;
                }
            }
        }
    }
}

#endif  // INFER_FLASH_ATTENTION_SPARSE_H