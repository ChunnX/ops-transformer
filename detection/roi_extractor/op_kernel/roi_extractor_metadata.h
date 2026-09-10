/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the LICENSE file in the root of the software repository for the full text of the License.
 */

/*!
 * \file roi_extractor_metadata.h
 * \brief Shared constants between the PyPTO kernel and the host side.
 */
#ifndef ROI_EXTRACTOR_METADATA_H_
#define ROI_EXTRACTOR_METADATA_H_

#include <cstdint>

namespace optiling {

// Channel lanes per work item (one 256B fp32 VF register).
constexpr uint32_t ROI_EXTRACTOR_CB = 64;
// Max columns of one row-load window (UB tile [CB, SPAN]).
constexpr uint32_t ROI_EXTRACTOR_SPAN = 128;
// Max pooled width (accumulator tile dim).
constexpr uint32_t ROI_EXTRACTOR_PW_MAX = 56;
// Max FPN levels (feats0..feats4 slots).
constexpr uint32_t ROI_EXTRACTOR_MAX_LEVELS = 5;
// TilingKey bit0: 0 = avg, 1 = max (matches RoiExtractorTilingKey).
constexpr uint64_t ROI_EXTRACTOR_TILING_KEY_MAX = 1ULL;

} // namespace optiling

#endif // ROI_EXTRACTOR_METADATA_H_
