/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_tiling_v3.cpp
 * \brief
 */

#include "fused_infer_attention_score_tiling_v3.h"
#include "fused_infer_attention_score_tiling_check.h"
#include "fused_infer_attention_score_tiling_info_parser.h"
#include "../../common/op_host/arch32/fia_tiling_nonquant_mla.h"
#include "../../common/op_host/fia_tiling_templates_registry.h"

using namespace AscendC;
namespace optiling {
// FIA新TilingKey, 18位编码, IFA原有TilingKey是17位, 新的TilingKey只是把最高位从1X->10X
// MLA dtype: Q=BF16 KV=BF16 OUT=BF16
// bfl6 7buf_nz
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000020222221, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000020322221, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000020222222, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000020322222, FusedInferAttentionScoreTilingData)
// 7buf_nd
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000010222220, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000010322220, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000000222220, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000000322220, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000010022221, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000010122221, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000010222221, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000010322221, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000010222222, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000010322222, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000000222222, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000000322222, FusedInferAttentionScoreTilingData)

// MLA dtype: Q=FP16 KV=FP16 OUT=FP16
// fp16 7buf_nz
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000020200001, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000020300001, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000020200002, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000020300002, FusedInferAttentionScoreTilingData)
// fp16 7buf_nd
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000010200000, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000010300000, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000000200000, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000000300000, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000010000001, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000010100001, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000010200001, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000010300001, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000010200002, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000010300002, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000000200002, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_105000000000300002, FusedInferAttentionScoreTilingData)

FIA_EXTERN_C ge::graphStatus TilingFusedInferAttentionScoreV3(gert::TilingContext *context)
{
    FiaTilingInfo fiaInfo;
    FiaInfoParser fiaInfoParser(context);
    if (fiaInfoParser.Parse(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Check函数只做校验，不能修改fiaInfo中的信息
    if (TilingCheck::Check(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return FiaTilingRegistry::GetInstance().DoTilingImpl(context, &fiaInfo);
}

bool RouteToFia(gert::TilingContext *context, IncreFlashAttentionContext &ifaContext)
{
    if ((context == nullptr) || (ifaContext.query.desc == nullptr) || (ifaContext.key.desc == nullptr)) {
        return false;
    }

    auto platformInfoPtr = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        return false;
    }

    ge::DataType qDataType = ifaContext.query.desc->GetDataType();
    ge::DataType kDataType = ifaContext.key.desc->GetDataType();
    bool isMla = (ifaContext.keyRope.tensor != nullptr && ifaContext.queryRope.tensor != nullptr);
    if (isMla) {
        // MLA非量化
        if ((qDataType == ge::DT_FLOAT16 || qDataType == ge::DT_BF16) && (qDataType == kDataType)) {
            return true;
        }
        // MLA BF16伪量化
        if ((qDataType == ge::DT_BF16) && (kDataType == ge::DT_INT8)) {
            return false;
        }
        // MLA FP16伪量化
        if ((qDataType == ge::DT_FLOAT16) && (kDataType == ge::DT_INT8)) {
            return false;
        }
        // MLA全量化
        if ((qDataType == ge::DT_INT8) && (kDataType == ge::DT_INT8)) {
            return false;
        }
    }
    return false;
}

} // namespace optiling
