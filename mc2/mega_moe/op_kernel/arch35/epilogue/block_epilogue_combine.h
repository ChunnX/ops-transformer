/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_EPILOGUE_COMBINE_H
#define BLOCK_EPILOGUE_COMBINE_H

#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "lib/matmul_intf.h"
#include "../common/mega_moe_constants.h"
#include "../common/mega_moe_utils.h"

namespace MegaMoeImpl {

constexpr uint32_t META_INFO_TENSOR_ADDR = 200U * 1024U;

namespace CombineImpl {
template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTokens(uint32_t nLoc, uint32_t n, LocalTensor<int32_t> &metaInfoTensor,
                                     LocalTensor<ElementMMadOut2> &l0cOutUbGMM2, const BlockShape &actualBlockShape,
                                     uint32_t ubTileN, const Params &params);
} // namespace CombineImpl

template <typename ElementC>
class BlockEpilogueCombine {
public:
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutC = Te::NDExtLayoutPtn;
    static constexpr uint32_t C0_SIZE_C = AuxGetC0Size<ElementC>();
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<C0_SIZE_C>>;

    __aicore__ inline BlockEpilogueCombine() {}

    __aicore__ inline void Init(const Params &params, GM_ADDR metaInfoGlobal, uint32_t n, uint32_t tileM)
    {
        params_ = &params;
        metaInfoGlobal_ = metaInfoGlobal;
        n_ = n;
        tileM_ = tileM;
    }

    template <typename SyncPolicy, typename TensorC, typename ActualShape>
    __aicore__ inline void operator()(TensorC &gmC, const ActualShape &actualShape,
                                       uint32_t mLoc, uint32_t nLoc, SyncPolicy &sync, uint32_t loopIdx)
    {
        int32_t mLen = static_cast<int32_t>(Get<M_VALUE>(actualShape));
        if (mLen <= 0) {
            return;
        }

        auto tensorBlockGm = gmC.Slice(
            AscendC::Te::MakeCoord(static_cast<int64_t>(mLoc), static_cast<int64_t>(nLoc)),
            AscendC::Te::MakeShape(static_cast<int64_t>(mLen),
                                    static_cast<int64_t>(Get<N_VALUE>(actualShape))));

        auto layoutL0cUB = MakeLayoutC{}(tileM_, L1_TILE_N);
        int64_t ubOffset = 0;
        auto tensorBlockUb = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, ElementC>(ubOffset), layoutL0cUB);
        AscendC::LocalTensor<ElementC> l0cOutUbGMM2 =
            AscendC::LocalTensor<ElementC>(AscendC::TPosition::VECIN, ubOffset, tileM_ * L1_TILE_N);
        auto copyGM2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
        AscendC::Te::Copy(copyGM2UB, tensorBlockUb, tensorBlockGm);

        AscendC::GlobalTensor<int32_t> metaInfoGm;
        AscendC::LocalTensor<int32_t> metaInfoTensor =
            AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, META_INFO_TENSOR_ADDR,
                                           static_cast<uint32_t>(mLen) * META_INFO_SIZE);
        metaInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            metaInfoGlobal_ + static_cast<uint64_t>(mLoc) * META_INFO_SIZE * sizeof(int32_t)));
        AscendC::DataCopy(metaInfoTensor, metaInfoGm, static_cast<uint32_t>(mLen) * META_INFO_SIZE);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
        CombineImpl::CombineTokens<ElementC, ActualShape>(
            nLoc, n_, metaInfoTensor, l0cOutUbGMM2, actualShape, L1_TILE_N, *params_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    }

private:
    const Params *params_{nullptr};
    GM_ADDR metaInfoGlobal_{0};
    uint32_t n_{0};
    uint32_t tileM_{0};
};

} // namespace MegaMoeImpl

#endif // BLOCK_EPILOGUE_COMBINE_H
