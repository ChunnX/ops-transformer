/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "liv2_checker.h"

#include <memory>
#include <vector>
#include "base_checker_indexer_v2.h"
#include "common_checker_indexer_v2.h"
#include "paged_attention_checker_indexer_v2.h"
#include "seq_len_checker_indexer_v2.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace indexer_v2_checker {
namespace {
constexpr uint32_t QUERY = 0;
constexpr uint32_t KEY = 1;
constexpr uint32_t WEIGHTS = 2;
constexpr uint32_t CU_Q = 3;
constexpr uint32_t CU_K = 4;
constexpr uint32_t SEQ_Q = 5;
constexpr uint32_t SEQ_K = 6;
constexpr uint32_t RESIDUAL_K = 7;
constexpr uint32_t BLOCK_TABLE = 8;
constexpr uint32_t OUTPUT_OFFSET = 9;
constexpr uint32_t METADATA = 10;
constexpr uint32_t INDICES = 0;
constexpr uint32_t VALUES = 1;
constexpr uint32_t ATTR_TOPK = 0;
constexpr uint32_t ATTR_MAX_SEQLEN_Q = 1;
constexpr uint32_t ATTR_LAYOUT_Q = 2;
constexpr uint32_t ATTR_LAYOUT_K = 3;
constexpr uint32_t ATTR_MASK_MODE = 4;
constexpr uint32_t ATTR_CMP_RATIO = 5;
constexpr uint32_t ATTR_RETURN_VALUE = 6;

CheckerTensor RequiredInput(gert::TilingContext *context, uint32_t index)
{
    return {context->GetInputDesc(index), context->GetInputShape(index), nullptr,
            context->GetDynamicInputStride(index, 0)};
}

CheckerTensor OptionalInput(gert::TilingContext *context, uint32_t index)
{
    return {context->GetOptionalInputDesc(index), nullptr, context->GetOptionalInputTensor(index),
            context->GetDynamicInputStride(index, 0)};
}

CheckerTensor Output(gert::TilingContext *context, uint32_t index)
{
    return {context->GetOutputDesc(index), context->GetOutputShape(index), nullptr, nullptr};
}

class LightningIndexerV2TypeChecker : public IndexerV2BaseChecker {
public:
    ge::graphStatus CheckSinglePara(const IndexerV2CheckerInfo &info) const override
    {
        if (info.maxSeqlenQ != -1) {
            const std::string actual = std::to_string(info.maxSeqlenQ);
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                info.opName, "max_seqlen_q", actual.c_str(),
                "max_seqlen_q is not effective for LightningIndexerV2 and only -1 is supported");
            return ge::GRAPH_FAILED;
        }
        if (CheckDtype(info, info.query, "q", {ge::DT_FLOAT16, ge::DT_BF16}) != ge::GRAPH_SUCCESS ||
            CheckDtype(info, info.key, "k", {ge::DT_FLOAT16, ge::DT_BF16}) != ge::GRAPH_SUCCESS ||
            CheckDtype(info, info.weights, "w", {ge::DT_FLOAT}) != ge::GRAPH_SUCCESS ||
            CheckDtype(info, info.sparseValues, "sparse_values", {ge::DT_FLOAT}) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        const ge::DataType qType = info.query.desc->GetDataType();
        const ge::DataType kType = info.key.desc->GetDataType();
        if (qType != kType) {
            const std::string actual = Ops::Base::ToString(qType) + " and " + Ops::Base::ToString(kType);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(info.opName, "q and k", actual.c_str(),
                                                   "q and k must have the same dtype");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
};

ge::graphStatus BuildInfo(gert::TilingContext *context, IndexerV2CheckerInfo &info)
{
    info.opName = context->GetNodeName() == nullptr ? "LightningIndexerV2" : context->GetNodeName();
    info.query = RequiredInput(context, QUERY);
    info.key = RequiredInput(context, KEY);
    info.weights = RequiredInput(context, WEIGHTS);
    info.cuSeqlensQ = OptionalInput(context, CU_Q);
    info.cuSeqlensK = OptionalInput(context, CU_K);
    info.sequsedQ = OptionalInput(context, SEQ_Q);
    info.sequsedK = OptionalInput(context, SEQ_K);
    info.cmpResidualK = OptionalInput(context, RESIDUAL_K);
    info.blockTable = OptionalInput(context, BLOCK_TABLE);
    info.outputIdxOffset = OptionalInput(context, OUTPUT_OFFSET);
    info.metadata = OptionalInput(context, METADATA);
    info.sparseIndices = Output(context, INDICES);
    info.sparseValues = Output(context, VALUES);

    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "attrs", "Operator attributes must be provided");
        return ge::GRAPH_FAILED;
    }
    const int32_t *topk = attrs->GetAttrPointer<int32_t>(ATTR_TOPK);
    const int32_t *maxSeqlenQ = attrs->GetAttrPointer<int32_t>(ATTR_MAX_SEQLEN_Q);
    const char *layoutQ = attrs->GetStr(ATTR_LAYOUT_Q);
    const char *layoutK = attrs->GetStr(ATTR_LAYOUT_K);
    const int32_t *maskMode = attrs->GetAttrPointer<int32_t>(ATTR_MASK_MODE);
    const int64_t *cmpRatio = attrs->GetAttrPointer<int64_t>(ATTR_CMP_RATIO);
    const int32_t *returnValue = attrs->GetAttrPointer<int32_t>(ATTR_RETURN_VALUE);
    if (topk == nullptr || maxSeqlenQ == nullptr || layoutQ == nullptr || layoutK == nullptr || maskMode == nullptr ||
        cmpRatio == nullptr || returnValue == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "attrs", "Required/defaulted attributes are missing");
        return ge::GRAPH_FAILED;
    }
    info.topk = *topk;
    info.maxSeqlenQ = *maxSeqlenQ;
    info.layoutQText = layoutQ;
    info.layoutKText = layoutK;
    info.maskMode = *maskMode;
    info.cmpRatio = *cmpRatio;
    info.returnValue = *returnValue;

    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(info.opName, "platform_info", "Platform information must be provided");
        return ge::GRAPH_FAILED;
    }
    const auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const NpuArch npuArch = platform.GetCurNpuArch();
    if (npuArch != NpuArch::DAV_2201 && npuArch != NpuArch::DAV_3510) {
        const std::string actual = std::to_string(static_cast<int32_t>(npuArch));
        OP_LOGE_FOR_INVALID_VALUE(info.opName, "npu_arch", actual.c_str(), "DAV_2201 or DAV_3510");
        return ge::GRAPH_FAILED;
    }
    info.arch35 = npuArch == NpuArch::DAV_3510;
    return FinalizeCommonInfo(info);
}

class CheckerPipeline {
public:
    CheckerPipeline()
    {
        checkers_.push_back(std::make_unique<CommonChecker>());
        checkers_.push_back(std::make_unique<SeqLenChecker>());
        checkers_.push_back(std::make_unique<PagedAttentionChecker>());
        checkers_.push_back(std::make_unique<LightningIndexerV2TypeChecker>());
    }

    ge::graphStatus Process(const IndexerV2CheckerInfo &info) const
    {
        using Method = ge::graphStatus (IndexerV2BaseChecker::*)(const IndexerV2CheckerInfo &) const;
        const Method methods[] = {&IndexerV2BaseChecker::CheckParaExistence, &IndexerV2BaseChecker::CheckSinglePara,
                                  &IndexerV2BaseChecker::CheckFeature, &IndexerV2BaseChecker::CheckMultiPara};
        for (Method method : methods) {
            for (const auto &checker : checkers_) {
                if ((checker.get()->*method)(info) != ge::GRAPH_SUCCESS) {
                    return ge::GRAPH_FAILED;
                }
            }
        }
        return ge::GRAPH_SUCCESS;
    }

private:
    std::vector<std::unique_ptr<IndexerV2BaseChecker>> checkers_;
};
} // namespace
} // namespace indexer_v2_checker

ge::graphStatus CheckLightningIndexerV2(gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("LightningIndexerV2", "context", "Tiling context must not be null");
        return ge::GRAPH_FAILED;
    }
    indexer_v2_checker::IndexerV2CheckerInfo info;
    if (indexer_v2_checker::BuildInfo(context, info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return indexer_v2_checker::CheckerPipeline().Process(info);
}

} // namespace optiling
