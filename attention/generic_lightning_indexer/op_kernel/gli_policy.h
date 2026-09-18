/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GLI_POLICY_H
#define GLI_POLICY_H

#include <cstdint>

namespace gli {
enum class GliAct : uint32_t {
    Relu = 0,
    Identity = 1
};
enum class GliHeadReduce : uint32_t {
    WeightedSum = 0,
    Sum = 1,
    None = 2
};
enum class GliKeyReduce : uint32_t {
    None = 0,
    MaxPool = 1
};
enum class GliSelect : uint32_t {
    TopK = 0,
    None = 1
};
enum class GliExpand : uint32_t {
    Off = 0,
    Expand = 1,
    ExpandAppend = 2
};
enum class GliMask : uint32_t {
    None = 0,
    Causal = 3,
    BlockComplete = 4
};

template <GliAct Act = GliAct::Relu, GliHeadReduce HeadReduce = GliHeadReduce::WeightedSum,
          GliKeyReduce KeyReduce = GliKeyReduce::None, GliSelect Select = GliSelect::TopK,
          GliExpand Expand = GliExpand::Off, GliMask Mask = GliMask::None>
struct GliPolicy {
    static constexpr GliAct act = Act;
    static constexpr GliHeadReduce headReduce = HeadReduce;
    static constexpr GliKeyReduce keyReduce = KeyReduce;
    static constexpr GliSelect select = Select;
    static constexpr GliExpand expand = Expand;
    static constexpr GliMask mask = Mask;
    static constexpr bool applyRelu = (Act == GliAct::Relu);
    static constexpr bool useWeights = (HeadReduce == GliHeadReduce::WeightedSum);
    static constexpr bool doExpand = (Expand != GliExpand::Off);
    static constexpr bool doAppend = (Expand == GliExpand::ExpandAppend);
    static constexpr bool doTopK = (Select == GliSelect::TopK);
};
} // namespace gli

#endif
