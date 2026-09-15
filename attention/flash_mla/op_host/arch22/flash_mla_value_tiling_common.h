/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_MLA_VALUE_TILING_COMMON_H
#define FLASH_MLA_VALUE_TILING_COMMON_H
#include "../flash_mla_value_tiling.h"
#include "../flash_mla_sequence_policy.h"
// Conversion and validation at the device-sequence adapter boundary.
#include "../../../fused_infer_attention_score/op_host/fia_tiling_info.h"
#include <algorithm>
#include <numeric>
#include <cstring>
#include <limits>
#include <new>

namespace FlashMla {
inline gert::Tensor MlaSeqTensor(MlaSeqView seq)
{
    const int64_t count = static_cast<int64_t>(seq.count);
    gert::Tensor tensor(gert::StorageShape({count}, {count}), gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {}),
                        ge::DT_INT64);
    tensor.SetData(gert::TensorData(const_cast<int64_t *>(seq.data), nullptr,
                                    static_cast<size_t>(seq.count * sizeof(int64_t)), gert::kOnHost));
    return tensor;
}
inline optiling::FiaLayout MlaFiaLayout(MlaLayout layout)
{
    switch (layout) {
        case MlaLayout::BSH:
            return optiling::FiaLayout::BSH;
        case MlaLayout::BSND:
            return optiling::FiaLayout::BSND;
        case MlaLayout::BNSD:
            return optiling::FiaLayout::BNSD;
        case MlaLayout::TND:
            return optiling::FiaLayout::TND;
        default:
            return optiling::FiaLayout::NZ;
    }
}
inline optiling::TilingKeyLayout MlaTilingLayout(MlaLayout layout)
{
    switch (layout) {
        case MlaLayout::BNSD:
            return optiling::TilingKeyLayout::BNSD;
        case MlaLayout::TND:
            return optiling::TilingKeyLayout::TND;
        case MlaLayout::NZ:
            return optiling::TilingKeyLayout::NZ;
        default:
            return optiling::TilingKeyLayout::BSH_BSND;
    }
}
inline MlaStatus ValidateMlaValueDimensions(const MlaScheduleConfig &c)
{
    // FIA and IFA bound logical B, independently of padded sequence item counts.
    if (c.numHeads == 0 || c.numHeads > 128 || c.batchSize > 65536U || c.qSeqSize > UINT32_MAX ||
        c.kvSeqSize > INT32_MAX) {
        return MlaStatus::INVALID_ARGUMENT;
    }
    // The legacy MLA splitter assumes non-TND Q <= 32. TND stores total T in
    // qSeqSize and checks each batch's Q <= 32 in QKVPreProcess4TND; INT8 total
    // T is limited to 1 MiB by the original preprocessing path.
    const uint64_t maxLegacyQ = c.layoutQ == MlaLayout::TND ? 1024U * 1024U : 32U;
    if (c.queryDataType == kMlaInt8 &&
        (c.qSeqSize > maxLegacyQ ||
         (c.layoutQ != MlaLayout::BSH && c.layoutQ != MlaLayout::BSND && c.layoutQ != MlaLayout::TND))) {
        return MlaStatus::INVALID_ARGUMENT;
    }
    return ValidateMlaSparseConfig(c);
}
inline MlaStatus ValidateMlaSeqViews(const MlaScheduleConfig &c, const MlaPlatformInfo &p, MlaSeqView q, MlaSeqView kv)
{
    if (p.architecture != kMlaArch22 || p.aicNum == 0 || p.aicNum > kMlaMaxAicNum ||
        (p.aivNum != p.aicNum && p.aivNum != 2 * p.aicNum) || ValidateMlaValueDimensions(c) != MlaStatus::SUCCESS ||
        q.count != c.seqQCount || kv.count != c.seqKvCount || (c.seqQPresent && q.count && q.data == nullptr) ||
        (c.seqKvPresent && kv.count && kv.data == nullptr) || (!c.seqQPresent && q.count) ||
        (!c.seqKvPresent && kv.count)) {
        return MlaStatus::INVALID_ARGUMENT;
    }
    if ((!IsMlaQuerySeqIgnored(c.queryDataType, c.layoutQ == MlaLayout::TND) && q.count != 0 && q.count != 1 &&
         q.count < c.batchSize) ||
        (kv.count != 0 && kv.count != 1 && kv.count < c.batchSize) ||
        (c.layoutQ == MlaLayout::TND && ((c.batchSize != 0 && !c.seqQPresent) || q.count != c.batchSize)) ||
        ((c.layoutQ == MlaLayout::TND || c.layoutKv == MlaLayout::TND) &&
         ((c.batchSize != 0 && !c.seqKvPresent) || kv.count != c.batchSize)) ||
        (c.blockSize && c.batchSize != 0 && (!c.seqKvPresent || kv.count == 0))) {
        return MlaStatus::INVALID_SEQ;
    }
    return MlaStatus::SUCCESS;
}

inline MlaStatus ValidateMlaSeqValues(const MlaScheduleConfig &c, MlaSeqView q, MlaSeqView kv)
{
    // The original parsers subtract cumulative values before their detailed
    // checks. Reject invalid values first so INT64_MIN/MAX cannot overflow that
    // subtraction or be narrowed to an apparently valid sequence length.
    const MlaSeqView views[] = {q, kv};
    const uint64_t storage[] = {c.qSeqSize, c.kvSeqSize};
    const MlaLayout layouts[] = {c.layoutQ, c.layoutKv};
    for (uint32_t input = 0; input < 2; ++input) {
        if (input == 0 && IsMlaQuerySeqIgnored(c.queryDataType, c.layoutQ == MlaLayout::TND)) {
            continue;
        }
        const auto &view = views[input];
        const uint64_t count = std::min(view.count, static_cast<uint64_t>(c.batchSize));
        for (uint64_t i = 0; i < count; ++i) {
            const int64_t value = view.data[i];
            if (value < 0 || static_cast<uint64_t>(value) > storage[input] ||
                (layouts[input] == MlaLayout::TND && i != 0 && value < view.data[i - 1])) {
                return MlaStatus::INVALID_SEQ;
            }
        }
    }
    return MlaStatus::SUCCESS;
}
} // namespace FlashMla
#endif
