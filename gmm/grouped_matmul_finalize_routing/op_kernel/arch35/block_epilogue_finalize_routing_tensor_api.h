/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "cgmct/epilogue/block_epilogue_finalize_routing.h"
#include "tensor_api/tensor.h"

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeLogit_>
class BlockEpilogueFinalizeRoutingTensorApi {
public:
    using LegacyEpilogue = Cgmct::Gemm::Block::BlockEpilogueFinalizeRouting<DataTypeOut_, DataTypeIn_, DataTypeLogit_>;
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using DataTypeLogit = DataTypeLogit_;
    using Params = typename LegacyEpilogue::Params;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t>;

    struct OutputOffsets {
        int64_t yOffset{0};
        int64_t routeOffset{0};
    };

    __aicore__ inline void Init(const Params &params)
    {
        epilogue_.Init(params);
    }

    __aicore__ inline auto GetL0c2UbTensor(int64_t rows, int64_t cols)
    {
        constexpr uint64_t c0 = AscendC::Te::C0_ELEMENT<DataTypeIn>;
        const int64_t copyRows = (rows + 1) & ~1L;
        const auto layout =
            AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn, AscendC::Std::Int<c0>>(copyRows, cols);
        return AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, DataTypeIn>(0), layout);
    }

    __aicore__ inline void UpdateNextProblem(const ProblemShape &shape)
    {
        epilogue_.UpdateNextProblem(typename LegacyEpilogue::ProblemShape{
            AscendC::Te::Get<MNK_M>(shape), AscendC::Te::Get<MNK_N>(shape), AscendC::Te::Get<MNK_K>(shape)});
    }

    __aicore__ inline void UpdateGlobalAddr(const OutputOffsets &offsets)
    {
        epilogue_.UpdateGlobalAddr(
            typename LegacyEpilogue::BlockCoord{offsets.yOffset, 0, 0, 0, offsets.routeOffset, offsets.routeOffset});
    }

    __aicore__ inline void operator()(const BlockShape &shape, const OutputOffsets &offsets)
    {
        epilogue_(
            typename LegacyEpilogue::BlockShape{AscendC::Te::Get<MNK_M>(shape), AscendC::Te::Get<MNK_N>(shape),
                                                AscendC::Te::Get<MNK_K>(shape), 0},
            typename LegacyEpilogue::BlockCoord{offsets.yOffset, 0, 0, 0, offsets.routeOffset, offsets.routeOffset});
    }

private:
    LegacyEpilogue epilogue_;
};
