/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <climits>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "turbo_quant_attention_score_tiling.h"

namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext *context)
{
    const uint32_t ranks[] = {3, 3, 4, 4, 3, 3, 2, 1};
    const gert::Shape *s[8];
    for (size_t i = 0; i < 8; ++i) {
        const auto *shape = context->GetInputShape(i);
        if (!shape || !context->GetInputDesc(i)) {
            return ge::GRAPH_FAILED;
        }
        s[i] = &shape->GetStorageShape();
        if (s[i]->GetDimNum() != ranks[i]) {
            return ge::GRAPH_FAILED;
        }
        for (size_t d = 0; d < ranks[i]; ++d) {
            if (s[i]->GetDim(d) <= 0 || s[i]->GetDim(d) > INT_MAX) {
                return ge::GRAPH_FAILED;
            }
        }
    }
    const auto *attrs = context->GetAttrs();
    const auto *bits = attrs ? attrs->GetAttrPointer<int64_t>(0) : nullptr;
    if (!bits || *bits < 2 || *bits > 4) {
        return ge::GRAPH_FAILED;
    }
    const int64_t tokens = s[0]->GetDim(0), qHeads = s[0]->GetDim(1);
    const int64_t kvHeads = s[2]->GetDim(2), batch = s[6]->GetDim(0);
    if (*s[0] != *s[1] || s[0]->GetDim(2) != 128 || s[2]->GetDim(1) != 128 || s[2]->GetDim(3) != 16 * *bits ||
        s[3]->GetDim(3) != 16 || qHeads % kvHeads != 0 || (batch != 1 && tokens != batch) || s[7]->GetDim(0) != batch) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 3; i <= 5; ++i) {
        for (size_t d = 0; d < 3; ++d) {
            if (s[i]->GetDim(d) != s[2]->GetDim(d)) {
                return ge::GRAPH_FAILED;
            }
        }
    }
    const auto normType = context->GetInputDesc(4)->GetDataType();
    const auto lengthType = context->GetInputDesc(7)->GetDataType();
    const ge::DataType types[] = {ge::DT_BF16, ge::DT_BF16, ge::DT_UINT8, ge::DT_UINT8,
                                  normType,    normType,    ge::DT_INT32, lengthType};
    if ((normType != ge::DT_BF16 && normType != ge::DT_FLOAT16) ||
        (lengthType != ge::DT_INT32 && lengthType != ge::DT_INT64)) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < 8; ++i) {
        if (context->GetInputDesc(i)->GetDataType() != types[i]) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto *out = context->GetOutputShape(0);
    if (!out || out->GetStorageShape().GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    int64_t maxLen = out->GetStorageShape().GetDim(2);
    if (maxLen < 0 || maxLen > INT_MAX || maxLen > s[6]->GetDim(1) * 128) {
        return ge::GRAPH_FAILED;
    }
    // Optional value-dependent Tensor inputs may remain on device in aclnn.
    // In that case the caller supplies the output shape, as for other attention
    // operators. Only inspect length values when the runtime provides host data.
    const auto *lengths = context->GetInputTensor(7);
    if (lengths && (lengths->GetPlacement() == gert::kOnHost || lengths->GetPlacement() == gert::kFollowing)) {
        if (!lengths->GetData<int32_t>()) {
            return ge::GRAPH_FAILED;
        }
        int64_t actualMax = 0;
        for (int64_t i = 0; i < batch; ++i) {
            const int64_t len =
                lengthType == ge::DT_INT32 ? lengths->GetData<int32_t>()[i] : lengths->GetData<int64_t>()[i];
            if (len < 0 || len > INT_MAX || len > s[6]->GetDim(1) * 128) {
                return ge::GRAPH_FAILED;
            }
            actualMax = std::max(actualMax, len);
        }
        if (actualMax != maxLen) {
            return ge::GRAPH_FAILED;
        }
    }
    if (out->GetStorageShape() != gert::Shape({tokens, qHeads, maxLen}) ||
        context->GetOutputDesc(0)->GetDataType() != ge::DT_FLOAT || !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t cores = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (!cores || ubSize < (tokens == 1 ? 187U : 185U) * 1024U) {
        return ge::GRAPH_FAILED;
    }
    TurboQuantAttentionScoreTilingData tiling;
    tiling.set_tokens(tokens);
    tiling.set_qHeads(qHeads);
    tiling.set_kvHeads(kvHeads);
    tiling.set_batch(batch);
    tiling.set_blocks(s[2]->GetDim(0));
    tiling.set_tableWidth(s[6]->GetDim(1));
    tiling.set_maxLen(maxLen);
    tiling.set_bits(*bits);
    context->SetBlockDim(cores);
    context->SetTilingKey(normType == ge::DT_BF16 ? 0 : 1);
    context->GetWorkspaceSizes(1)[0] = 0;
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(TurboQuantAttentionScore).Tiling(Tiling).TilingInputsDataDependency({7});
} // namespace optiling
