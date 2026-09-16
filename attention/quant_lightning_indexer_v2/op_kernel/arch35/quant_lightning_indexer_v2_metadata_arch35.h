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
 * \file quant_lightning_indexer_v2_metadata_arch35.h
 * \brief QuantLightningIndexerV2 metadata definitions for arch35.
 */

#ifndef QUANT_LIGHTNING_INDEXER_V2_METADATA_ARCH35_H
#define QUANT_LIGHTNING_INDEXER_V2_METADATA_ARCH35_H

#include "../../../lightning_indexer_v2/op_kernel/arch35/common/lightning_indexer_v2_metadata_common.h"

namespace optiling {

constexpr uint32_t QLI_V2_METADATA_TOTAL_SIZE = 1024;
using QLI_V2_METADATA_T = int32_t;

namespace detail {
struct QliV2Metadata {
    uint32_t qliV2Metadata[AIC_CORE_MAX_NUM][LI_V2_METADATA_SIZE];
    uint32_t qldV2Metadata[AIV_CORE_MAX_NUM][LD_V2_METADATA_SIZE];
};
}; // namespace detail

static_assert(QLI_V2_METADATA_TOTAL_SIZE * sizeof(QLI_V2_METADATA_T) >= sizeof(detail::QliV2Metadata));
}; // namespace optiling

#endif // QUANT_LIGHTNING_INDEXER_V2_METADATA_ARCH35_H
