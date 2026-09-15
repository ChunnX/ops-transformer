/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_MLA_STATIC_TILING_H
#define FLASH_MLA_STATIC_TILING_H

#include "../../../fused_infer_attention_score/op_host/fia_tiling_info.h"
#include "../../../fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.h"

namespace FlashMla {
inline void FillArch22MlaStaticTiling(const optiling::FiaTilingInfo &info,
                                      optiling::FusedInferAttentionScoreTilingData &tiling)
{
    tiling.baseParams.set_bSize(info.bSize);
    tiling.baseParams.set_n2Size(info.n2Size);
    tiling.baseParams.set_headDim(info.vHeadDim);
    tiling.baseParams.set_headDimRope(info.ropeHeadDim);
    tiling.baseParams.set_scaleValue(info.scaleValue);
    tiling.baseParams.set_gSize(info.n1Size / info.n2Size);
    tiling.baseParams.set_batchContinuous((info.kvStorageMode == optiling::KvStorageMode::TENSOR_LIST) ? 0 : 1);
    tiling.baseParams.set_actualSeqS1Dims(info.actualLenQDims);
    tiling.baseParams.set_actualSeqS2Dims(info.actualLenKvDims);
    tiling.baseParams.set_accumQSeqFlag(info.isAccumQSeq ? 1 : 0);
    tiling.baseParams.set_accumKVSeqFlag(info.isAccumKVSeq ? 1 : 0);
    tiling.baseParams.set_outputLayout(static_cast<uint32_t>(info.outputLayout));
    tiling.baseParams.set_softmaxLseFlag(info.softmaxLseFlag ? 1 : 0);
    tiling.baseParams.set_isEmptyBatchOverHalf(false);
    tiling.kvStrideParams.set_keyBnStride(info.keyNonContigDim == 0 ? info.keyBnStride : 0);
    tiling.kvStrideParams.set_valueBnStride(info.valueNonContigDim == 0 ? info.valueBnStride : 0);
    tiling.kvStrideParams.set_keyRopeBnStride(info.keyRopeNonContigDim == 0 ? info.kRopeBnStride : 0);
    tiling.pageAttenParams.set_blockSize(info.blockSize);
    tiling.maskParams.set_attenMaskFlag(info.attenMaskFlag ? 1 : 0);
    tiling.maskParams.set_attenMaskBatchStride(info.attenMaskBatchStride);
    tiling.maskParams.set_attenMaskStride(info.attenMaskStride);
    tiling.maskParams.set_sparseMode(info.sparseMode);
    tiling.maskParams.set_preToken(info.preToken);
    tiling.maskParams.set_nextToken(info.nextToken);
    tiling.baseParams.set_slidingFlag(info.slidingFlag);
    tiling.maskParams.set_isRowInvalid(static_cast<uint32_t>(info.innerPrecise) >> 1);
}
} // namespace FlashMla
#endif
