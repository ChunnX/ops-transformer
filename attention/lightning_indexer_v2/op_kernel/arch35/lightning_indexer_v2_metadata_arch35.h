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
 * \file lightning_indexer_v2_metadata_arch35.h
 * \brief LightningIndexerV2 metadata definitions for arch35.
 */

#ifndef LIGHTNING_INDEXER_V2_METADATA_ARCH35_H
#define LIGHTNING_INDEXER_V2_METADATA_ARCH35_H

#include "common/lightning_indexer_v2_metadata_common.h"

namespace optiling {

constexpr uint32_t LI_V2_METADATA_TOTAL_SIZE = 1024;
using LI_V2_METADATA_T = int32_t;

namespace detail {
struct LiV2Metadata {
    uint32_t liV2Metadata[AIC_CORE_MAX_NUM][LI_V2_METADATA_SIZE];
    uint32_t ldV2Metadata[AIV_CORE_MAX_NUM][LD_V2_METADATA_SIZE];
};
}; // namespace detail

static_assert(LI_V2_METADATA_TOTAL_SIZE * sizeof(LI_V2_METADATA_T) >= sizeof(detail::LiV2Metadata));
}; // namespace optiling

#endif // LIGHTNING_INDEXER_V2_METADATA_ARCH35_H
