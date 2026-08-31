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

#include <cstdint>

namespace optiling {
namespace fag {

struct NzOutputCapabilityInput {
    bool isBn2Gs1S2 = false;
    bool queryTypeSupported = false;
    bool isLegacyDeterministic = false;
    int64_t headDim = 0;
    int64_t minimumHeadDimExclusive = 0;
    int64_t maximumHeadDimExclusive = 0;
    int64_t headDimAlignment = 0;
};

class KernelCapabilityVerifier {
public:
    static bool SupportsNzOutput(const NzOutputCapabilityInput &input)
    {
        return input.isBn2Gs1S2 && input.queryTypeSupported && !input.isLegacyDeterministic &&
               input.headDimAlignment > 0 && input.headDim > input.minimumHeadDimExclusive &&
               input.headDim < input.maximumHeadDimExclusive && input.headDim % input.headDimAlignment != 0;
    }
};

} // namespace fag
} // namespace optiling
