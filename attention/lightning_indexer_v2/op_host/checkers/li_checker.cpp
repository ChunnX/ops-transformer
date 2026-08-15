/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "common_checker.h"
#include "compression_checker.h"
#include "li_checker.h"
#include "mask_checker.h"
#include "metadata_checker.h"
#include "non_quant_checker.h"
#include "paged_attention_checker.h"
#include "seq_len_checker.h"
namespace optiling {
namespace lightning_indexer_v2 {
void LIChecker::BuildInfo(const LIV2TilingInfo &source)
{
    const auto &p = source.opParamInfo;
    info_ = {};
    info_.opName = source.opName;
    info_.isArch35 = platform_ascendc::PlatformAscendC(source.platformInfo).GetCurNpuArch() == NpuArch::DAV_3510;
    info_.query = MakeRequiredTensor(p.query.desc, p.query.shape);
    info_.key = MakeRequiredTensor(p.key.desc, p.key.shape);
    info_.weights = MakeRequiredTensor(p.weights.desc, p.weights.shape);
    info_.cuSeqlensQ = MakeOptionalTensor(p.cuSeqlensQ.desc, p.cuSeqlensQ.tensor);
    info_.cuSeqlensK = MakeOptionalTensor(p.cuSeqlensK.desc, p.cuSeqlensK.tensor);
    info_.sequsedQ = MakeOptionalTensor(p.sequsedQ.desc, p.sequsedQ.tensor);
    info_.sequsedK = MakeOptionalTensor(p.sequsedK.desc, p.sequsedK.tensor);
    info_.cmpResidualK = MakeOptionalTensor(p.cmpResidualK.desc, p.cmpResidualK.tensor);
    info_.blockTable = MakeOptionalTensor(p.blockTable.desc, p.blockTable.tensor);
    info_.outputIdxOffset = MakeOptionalTensor(p.outputIdxOffset.desc, p.outputIdxOffset.tensor);
    info_.metadata = MakeOptionalTensor(p.metadata.desc, p.metadata.tensor);
    info_.sparseIndices = MakeRequiredTensor(p.attenOut.desc, p.attenOut.shape);
    info_.sparseValues = MakeRequiredTensor(p.valuesOut.desc, p.valuesOut.shape);
    info_.queryLayout = ConvertIndexerLayout(static_cast<uint32_t>(source.inputQLayout));
    info_.keyLayout = ConvertIndexerLayout(static_cast<uint32_t>(source.inputKLayout));
    info_.batchSize = source.bSize;
    info_.queryHeadNum = source.n1Size;
    info_.keyHeadNum = source.n2Size;
    info_.querySeqLen = info_.queryLayout == IndexerLayout::TND && !info_.query.shape.empty() ?
                            info_.query.shape[0] :
                            source.s1Size;
    info_.keySeqLen = source.s2Size;
    info_.headDim = GetIndexerHeadDim(info_.query, info_.queryLayout);
    info_.groupSize = source.gSize;
    info_.blockSize = source.blockSize;
    info_.maxBlockNumPerBatch = source.maxBlockNumPerBatch;
    info_.topk = source.topk;
    info_.maxSeqlenQ = source.maxSeqlenQ;
    info_.maskMode = source.maskMode;
    info_.cmpRatio = source.cmpRatio;
    info_.returnValue = p.returnValue == nullptr ? static_cast<int64_t>(source.returnValue) : *p.returnValue;
    info_.keyStrides = source.keyStridesVec;
}
void LIChecker::RegisterCheckers()
{
    ClearCheckers();
    AddChecker<CommonChecker>();
    AddChecker<NonQuantChecker>();
    AddChecker<SeqLenChecker>();
    AddChecker<PagedAttentionChecker>();
    AddChecker<MaskChecker>();
    AddChecker<CompressionChecker>();
    AddChecker<MetadataChecker>();
}
ge::graphStatus LIChecker::Init(const LIV2TilingInfo &source)
{
    BuildInfo(source);
    RegisterCheckers();
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus LIChecker::Process(const LIV2TilingInfo &source)
{
    (void)source;
    return LICompositeChecker::Process(info_);
}
} // namespace lightning_indexer_v2
} // namespace optiling
