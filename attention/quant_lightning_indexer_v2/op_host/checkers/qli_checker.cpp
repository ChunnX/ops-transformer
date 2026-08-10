/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../../../lightning_indexer_v2/op_host/checkers/common_checker.h"
#include "../../../lightning_indexer_v2/op_host/checkers/compression_checker.h"
#include "../../../lightning_indexer_v2/op_host/checkers/mask_checker.h"
#include "../../../lightning_indexer_v2/op_host/checkers/metadata_checker.h"
#include "../../../lightning_indexer_v2/op_host/checkers/paged_attention_checker.h"
#include "../../../lightning_indexer_v2/op_host/checkers/seq_len_checker.h"
#include "qli_checker.h"
#include "quant_checker.h"
namespace optiling {
namespace lightning_indexer_v2 {
namespace {
IndexerLayout ConvertQuantLayout(DataLayout layout)
{
    switch (layout) {
        case DataLayout::BSND:
            return IndexerLayout::BSND;
        case DataLayout::TND:
            return IndexerLayout::TND;
        case DataLayout::PA_BBND:
            return IndexerLayout::PA_BBND;
        default:
            return IndexerLayout::UNKNOWN;
    }
}
int64_t QuantHeadDim(const CheckerTensor &query, IndexerLayout layout)
{
    const size_t index = layout == IndexerLayout::BSND ? 3 : 2;
    return query.shape.size() > index ? query.shape[index] : 0;
}
} // namespace
void QLIChecker::BuildInfo(const QLIV2TilingInfo &source)
{
    const auto &p = source.opParamInfo;
    info_ = {};
    info_.opName = source.opName;
    info_.quantized = true;
    info_.isArch35 = source.npuArch == NpuArch::DAV_3510;
    info_.query = MakeRequiredTensor(p.query.desc, p.query.shape);
    info_.key = MakeRequiredTensor(p.key.desc, p.key.shape);
    info_.weights = MakeRequiredTensor(p.weights.desc, p.weights.shape);
    info_.queryDescale = MakeRequiredTensor(p.query_dequant_scale.desc, p.query_dequant_scale.shape);
    info_.keyDescale = MakeRequiredTensor(p.key_dequant_scale.desc, p.key_dequant_scale.shape);
    info_.cuSeqlensQ = MakeOptionalTensor(p.cuSeqLensQ.desc, p.cuSeqLensQ.tensor);
    info_.cuSeqlensK = MakeOptionalTensor(p.cuSeqLensK.desc, p.cuSeqLensK.tensor);
    info_.sequsedQ = MakeOptionalTensor(p.sequsedQ.desc, p.sequsedQ.tensor);
    info_.sequsedK = MakeOptionalTensor(p.sequsedK.desc, p.sequsedK.tensor);
    info_.cmpResidualK = MakeOptionalTensor(p.cmpResidualK.desc, p.cmpResidualK.tensor);
    info_.blockTable = MakeOptionalTensor(p.blockTable.desc, p.blockTable.tensor);
    info_.outputIdxOffset = MakeOptionalTensor(p.outputIdxOffset.desc, p.outputIdxOffset.tensor);
    info_.metadata = MakeOptionalTensor(p.metadata.desc, p.metadata.tensor);
    info_.sparseIndices = MakeRequiredTensor(p.attenOut.desc, p.attenOut.shape);
    info_.sparseValues = MakeRequiredTensor(p.sparseValues.desc, p.sparseValues.shape);
    info_.queryLayout = ConvertQuantLayout(source.inputQLayout);
    info_.keyLayout = ConvertQuantLayout(source.inputKLayout);
    info_.batchSize = source.bSize;
    info_.queryHeadNum = source.n1Size;
    info_.keyHeadNum = source.n2Size;
    info_.querySeqLen = info_.queryLayout == IndexerLayout::TND && !info_.query.shape.empty() ?
                            info_.query.shape[0] :
                            source.s1Size;
    info_.keySeqLen = source.s2Size;
    info_.headDim = QuantHeadDim(info_.query, info_.queryLayout);
    info_.groupSize = source.gSize;
    info_.blockSize = source.blockSize;
    info_.maxBlockNumPerBatch = source.maxBlockNumPerBatch;
    info_.topk = source.sparseCount;
    info_.quantMode = p.quantMode == nullptr ? 0 : *p.quantMode;
    info_.maxSeqlenQ = source.maxSeqlenQ;
    info_.maskMode = source.sparseMode;
    info_.cmpRatio = source.cmpRatio;
    info_.returnValue = source.returnValue ? 1 : 0;
    info_.keyStrides = source.keyStridesVec;
    info_.keyDescaleStrides = source.keyDequantScaleStridesVec;
}
void QLIChecker::RegisterCheckers()
{
    checkers_.clear();
    checkers_.push_back(std::make_unique<CommonChecker>());
    checkers_.push_back(std::make_unique<QuantChecker>());
    checkers_.push_back(std::make_unique<SeqLenChecker>());
    checkers_.push_back(std::make_unique<PagedAttentionChecker>());
    checkers_.push_back(std::make_unique<MaskChecker>());
    checkers_.push_back(std::make_unique<CompressionChecker>());
    checkers_.push_back(std::make_unique<MetadataChecker>());
}
ge::graphStatus QLIChecker::Init(const QLIV2TilingInfo &source)
{
    BuildInfo(source);
    RegisterCheckers();
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus QLIChecker::RunCheck(
    const std::function<ge::graphStatus(LIBaseChecker *, const IndexerCheckerInfo &)> &method)
{
    for (const auto &checker : checkers_) {
        if (method(checker.get(), info_) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus QLIChecker::Process(const QLIV2TilingInfo &source)
{
    (void)source;
    if (RunCheck([](LIBaseChecker *c, const IndexerCheckerInfo &i) { return c->CheckSinglePara(i); }) !=
            ge::GRAPH_SUCCESS ||
        RunCheck([](LIBaseChecker *c, const IndexerCheckerInfo &i) { return c->CheckParaExistence(i); }) !=
            ge::GRAPH_SUCCESS ||
        RunCheck([](LIBaseChecker *c, const IndexerCheckerInfo &i) { return c->CheckFeature(i); }) != ge::GRAPH_SUCCESS ||
        RunCheck([](LIBaseChecker *c, const IndexerCheckerInfo &i) { return c->CheckMultiPara(i); }) !=
            ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace lightning_indexer_v2
} // namespace optiling
