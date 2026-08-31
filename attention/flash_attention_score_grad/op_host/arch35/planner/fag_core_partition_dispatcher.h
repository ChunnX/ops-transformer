/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "../model/fag_route_model.h"

namespace optiling {
namespace fag {

enum class CorePartitionStrategy : uint8_t {
    UNAVAILABLE = 0,
    DENSE = 1,
    BN2S2_LOAD_BALANCE = 2,
    BN2_MULTI_BLOCK_TND = 3,
    BN2_MULTI_BLOCK_SPARSE = 4,
    TND_UNPAD = 5,
    SPARSE_PREFIX = 6,
    SPARSE_GENERAL = 7,
};

struct CorePartitionDispatchInput {
    VariantId variantId = VariantId::UNKNOWN;
    bool isTnd = false;
    bool isSparse = false;
    bool isPrefixSparse = false;
    bool bn2s2Eligible = false;
};

class CorePartitionDispatcher {
public:
    static CorePartitionStrategy Select(const CorePartitionDispatchInput &input)
    {
        switch (input.variantId) {
            case VariantId::BN2S2:
                if (!input.bn2s2Eligible) {
                    return CorePartitionStrategy::UNAVAILABLE;
                }
                return input.isTnd || input.isSparse ? CorePartitionStrategy::BN2S2_LOAD_BALANCE :
                                                       CorePartitionStrategy::DENSE;
            case VariantId::BN2_MULTI_BLOCK:
                if (input.isTnd) {
                    return CorePartitionStrategy::BN2_MULTI_BLOCK_TND;
                }
                return input.isSparse ? CorePartitionStrategy::BN2_MULTI_BLOCK_SPARSE : CorePartitionStrategy::DENSE;
            case VariantId::BN2:
            case VariantId::BN2GS1S2:
                if (input.isTnd) {
                    return CorePartitionStrategy::TND_UNPAD;
                }
                if (input.isSparse) {
                    return input.isPrefixSparse ? CorePartitionStrategy::SPARSE_PREFIX :
                                                  CorePartitionStrategy::SPARSE_GENERAL;
                }
                return CorePartitionStrategy::DENSE;
            default:
                return CorePartitionStrategy::UNAVAILABLE;
        }
    }
};

} // namespace fag
} // namespace optiling
