/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe_gmm1_epilogue_tile.h
 * \brief GMM1 epilogue tile 适配器
 */

#ifndef MEGA_MOE_GMM1_EPILOGUE_TILE_H
#define MEGA_MOE_GMM1_EPILOGUE_TILE_H

#include "../epilogue/block_epilogue_activation_mx_quant.h"

namespace MegaMoeImpl {

using namespace AscendC;

template <typename BlockEpilogue, bool TopkWeightsPrefetch, bool IsInterleaved = false>
class Gmm1ActivationQuantEpilogue {
public:
    __aicore__ inline Gmm1ActivationQuantEpilogue(BlockEpilogue &blockEpilogue) : blockEpilogue_(blockEpilogue) {}

    __aicore__ inline void SetMetaInfo(GM_ADDR metaInfoGm, uint32_t expertBeforeCnt);

    __aicore__ inline void UpdateNextProblem(const typename BlockEpilogue::ProblemShape &problemShape);

    // AIV0 epilogue tile 统一入口。
    template <typename MakeLayoutC, typename Config, typename TensorC, typename ActualShape, typename SyncPolicy>
    __aicore__ inline void ProcessActivationTile(const Config &config, TensorC &gmC, const ActualShape &actualShape,
                                                 uint32_t mLoc, uint32_t nLoc, SyncPolicy &sync);

    // A8W4 AIV1 epilogue tile 入口。
    template <typename MakeLayoutC, typename Config, typename TensorC, typename ActualShape, typename SyncPolicy>
    __aicore__ inline void ProcessActivationTileA8W4(const Config &config, TensorC &l0cOutGm,
                                                     const ActualShape &actualShape, uint32_t mLoc, uint32_t nLoc,
                                                     SyncPolicy &sync);

private:
    template <typename MakeLayoutC, typename Config, typename TensorC, typename ActualShape, typename SyncPolicy>
    __aicore__ inline void Gmm1Aiv0PrefetchInterleavedEpilogueTileGeneric(const Config &config, TensorC &gmC,
                                                                          const ActualShape &actualShape,
                                                                          uint32_t mLoc, uint32_t nLoc,
                                                                          SyncPolicy &sync);

    template <typename MakeLayoutC, typename Config, typename TensorC, typename ActualShape, typename SyncPolicy>
    __aicore__ inline void Gmm1Aiv0PrefetchEpilogueTileGeneric(const Config &config, TensorC &gmC,
                                                               const ActualShape &actualShape, uint32_t mLoc,
                                                               uint32_t nLoc, SyncPolicy &sync);

    template <typename Config, typename ActualShape, typename SyncPolicy>
    __aicore__ inline void Gmm1Aiv0InterleavedEpilogueTileGeneric(const Config &config,
                                                                  const ActualShape &actualShape, uint32_t mLoc,
                                                                  uint32_t nLoc, uint16_t pingpongIdx,
                                                                  SyncPolicy &sync);

    template <typename Config, typename ActualShape, typename SyncPolicy>
    __aicore__ inline void Gmm1Aiv0EpilogueTileGeneric(const Config &config, const ActualShape &actualShape,
                                                       uint32_t mLoc, uint32_t nLoc, SyncPolicy &sync);

    template <typename MakeLayoutC, typename Config, typename TensorC, typename ActualShape, typename SyncPolicy>
    __aicore__ inline void Gmm1Aiv1PrefetchEpilogueTileA8W4(const Config &config, TensorC &l0cOutGm,
                                                            const ActualShape &actualShape, uint32_t mLoc,
                                                            uint32_t nLoc, SyncPolicy &sync);

    template <typename MakeLayoutC, typename Config, typename TensorC, typename ActualShape, typename SyncPolicy>
    __aicore__ inline void Gmm1Aiv1EpilogueTileA8W4(const Config &config, TensorC &l0cOutGm,
                                                    const ActualShape &actualShape, uint32_t mLoc, uint32_t nLoc,
                                                    SyncPolicy &sync);

    BlockEpilogue &blockEpilogue_;
    GlobalTensor<float> metaInfoGm_;
    uint32_t expertBeforeCnt_ = 0;
};

template <typename BlockEpilogue, bool TopkWeightsPrefetch, bool IsInterleaved>
__aicore__ inline void
Gmm1ActivationQuantEpilogue<BlockEpilogue, TopkWeightsPrefetch, IsInterleaved>::
    SetMetaInfo(GM_ADDR metaInfoGm, uint32_t expertBeforeCnt)
{
    metaInfoGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(metaInfoGm));
    expertBeforeCnt_ = expertBeforeCnt;
}

template <typename BlockEpilogue, bool TopkWeightsPrefetch, bool IsInterleaved>
__aicore__ inline void
Gmm1ActivationQuantEpilogue<BlockEpilogue, TopkWeightsPrefetch, IsInterleaved>::
    UpdateNextProblem(const typename BlockEpilogue::ProblemShape &problemShape)
{
    blockEpilogue_.UpdateNextProblem(problemShape);
}

template <typename BlockEpilogue, bool TopkWeightsPrefetch, bool IsInterleaved>
template <typename MakeLayoutC, typename Config, typename TensorC, typename ActualShape, typename SyncPolicy>
__aicore__ inline void
Gmm1ActivationQuantEpilogue<BlockEpilogue, TopkWeightsPrefetch, IsInterleaved>::
    ProcessActivationTile(const Config &config, TensorC &gmC, const ActualShape &actualShape, uint32_t mLoc,
                          uint32_t nLoc, SyncPolicy &sync)
{
    if constexpr (TopkWeightsPrefetch) {
        if constexpr (IsInterleaved) {
            Gmm1Aiv0PrefetchInterleavedEpilogueTileGeneric<MakeLayoutC>(config, gmC, actualShape, mLoc, nLoc, sync);
        } else {
            Gmm1Aiv0PrefetchEpilogueTileGeneric<MakeLayoutC>(config, gmC, actualShape, mLoc, nLoc, sync);
        }
    } else {
        if constexpr (IsInterleaved) {
            Gmm1Aiv0InterleavedEpilogueTileGeneric(config, actualShape, mLoc, nLoc, sync.GetPingpongIdx(), sync);
        } else {
            Gmm1Aiv0EpilogueTileGeneric(config, actualShape, mLoc, nLoc, sync);
        }
    }
}

template <typename BlockEpilogue, bool TopkWeightsPrefetch, bool IsInterleaved>
template <typename MakeLayoutC, typename Config, typename TensorC, typename ActualShape, typename SyncPolicy>
__aicore__ inline void
Gmm1ActivationQuantEpilogue<BlockEpilogue, TopkWeightsPrefetch, IsInterleaved>::
    ProcessActivationTileA8W4(const Config &config, TensorC &l0cOutGm, const ActualShape &actualShape, uint32_t mLoc,
                              uint32_t nLoc, SyncPolicy &sync)
{
    if constexpr (TopkWeightsPrefetch) {
        Gmm1Aiv1PrefetchEpilogueTileA8W4<MakeLayoutC>(config, l0cOutGm, actualShape, mLoc, nLoc, sync);
    } else {
        Gmm1Aiv1EpilogueTileA8W4<MakeLayoutC>(config, l0cOutGm, actualShape, mLoc, nLoc, sync);
    }
}

template <typename BlockEpilogue, bool TopkWeightsPrefetch, bool IsInterleaved>
template <typename MakeLayoutC, typename Config, typename TensorC, typename ActualShape, typename SyncPolicy>
__aicore__ inline void
Gmm1ActivationQuantEpilogue<BlockEpilogue, TopkWeightsPrefetch, IsInterleaved>::
    Gmm1Aiv0PrefetchInterleavedEpilogueTileGeneric(const Config &config, TensorC &gmC, const ActualShape &actualShape,
                                                   uint32_t mLoc, uint32_t nLoc, SyncPolicy &sync)
{
    uint32_t mLen = Get<M_VALUE>(actualShape);
    if (mLen == 0U) {
        return;
    }
    constexpr uint32_t subTileM = L1_TILE_M_128;
    auto layoutL0cUb = MakeLayoutC{}(subTileM, L1_TILE_N);
    auto tensorBlockUb = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, typename BlockEpilogue::DataTypeIn>(0),
                                        layoutL0cUb);
    auto copyGm2Ub = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
    auto topkWeightTensor = blockEpilogue_.GetTopkWeightTensor();

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    for (uint32_t subOffset = 0; subOffset < mLen; subOffset += subTileM) {
        uint32_t subM = mLen - subOffset < subTileM ? mLen - subOffset : subTileM;
        uint32_t subMLoc = mLoc + subOffset;
        uint64_t metaInfoRow = expertBeforeCnt_ + subMLoc;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::DataCopy(topkWeightTensor, metaInfoGm_[metaInfoRow * INT32_PER_256B], subM * INT32_PER_256B);
        auto tensorBlockGm =
            gmC.Slice(Te::MakeCoord(subMLoc, nLoc), Te::MakeShape(subM, Get<N_VALUE>(actualShape)));
        AscendC::Te::Copy(copyGm2Ub, tensorBlockUb, tensorBlockGm);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

        uint32_t epilogueN = Get<N_VALUE>(actualShape) / ACTIVATION_N_HALF;
        uint32_t epilogueNLoc = nLoc / ACTIVATION_N_HALF;
        Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{subM, epilogueN, 0, 0};
        Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
            subMLoc * config.outputN + epilogueNLoc,
            subMLoc * Ops::Base::CeilDiv<uint32_t>(config.outputN, static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE +
                Ops::Base::CeilDiv<uint32_t>(epilogueNLoc, static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE,
            static_cast<int64_t>(subMLoc / L1_TILE_M_256),
            0,
            static_cast<int64_t>(metaInfoRow),
            0};
        AscendC::SetCtrlSpr<60, 60>(0);
        blockEpilogue_(epilogueShape, epilogueOffset, 0U);
        sync.NotifyDownstreamReady(subMLoc);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
}

template <typename BlockEpilogue, bool TopkWeightsPrefetch, bool IsInterleaved>
template <typename MakeLayoutC, typename Config, typename TensorC, typename ActualShape, typename SyncPolicy>
__aicore__ inline void
Gmm1ActivationQuantEpilogue<BlockEpilogue, TopkWeightsPrefetch, IsInterleaved>::
    Gmm1Aiv0PrefetchEpilogueTileGeneric(const Config &config, TensorC &gmC, const ActualShape &actualShape,
                                        uint32_t mLoc, uint32_t nLoc, SyncPolicy &sync)
{
    uint32_t mLen = Get<M_VALUE>(actualShape);
    if (mLen == 0U) {
        return;
    }
    constexpr uint32_t subTileM = L1_TILE_M_128;
    uint64_t firstMetaInfoRow = expertBeforeCnt_ + mLoc;
    auto topkWeightTensor = blockEpilogue_.GetTopkWeightTensor();
    AscendC::DataCopy(topkWeightTensor, metaInfoGm_[firstMetaInfoRow * INT32_PER_256B],
                      static_cast<uint32_t>(mLen < subTileM ? mLen : subTileM) * INT32_PER_256B);

    auto layoutL0cUb = MakeLayoutC{}(subTileM, L1_TILE_N);
    int64_t ubOffsetFirst = 0;
    int64_t ubOffsetSecond = static_cast<int64_t>(MAX_SINGLE_MN_ALIGN32_NUM_128) * sizeof(typename BlockEpilogue::DataTypeIn);
    auto tensorBlockUbFirst =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, typename BlockEpilogue::DataTypeIn>(ubOffsetFirst), layoutL0cUb);
    auto tensorBlockUbSecond =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, typename BlockEpilogue::DataTypeIn>(ubOffsetSecond), layoutL0cUb);
    auto copyGm2Ub = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    for (uint32_t subOffset = 0; subOffset < mLen; subOffset += subTileM) {
        uint32_t subM = mLen - subOffset < subTileM ? mLen - subOffset : subTileM;
        uint32_t subMLoc = mLoc + subOffset;
        uint64_t metaInfoRow = expertBeforeCnt_ + subMLoc;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        if (subOffset != 0) {
            AscendC::DataCopy(topkWeightTensor, metaInfoGm_[metaInfoRow * INT32_PER_256B],
                              subM * INT32_PER_256B);
        }

        auto tensorBlockGmFirst =
            gmC.Slice(Te::MakeCoord(subMLoc, nLoc), Te::MakeShape(subM, Get<N_VALUE>(actualShape)));
        auto tensorBlockGmSecond =
            gmC.Slice(Te::MakeCoord(subMLoc, nLoc + config.outputN), Te::MakeShape(subM, Get<N_VALUE>(actualShape)));
        AscendC::Te::Copy(copyGm2Ub, tensorBlockUbFirst, tensorBlockGmFirst);
        AscendC::Te::Copy(copyGm2Ub, tensorBlockUbSecond, tensorBlockGmSecond);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

        Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{subM, Get<N_VALUE>(actualShape), 0, 0};
        Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
            subMLoc * config.outputN + nLoc,
            subMLoc * Ops::Base::CeilDiv<uint32_t>(config.outputN, static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE +
                Ops::Base::CeilDiv<uint32_t>(nLoc, static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE,
            0,
            0,
            static_cast<int64_t>(metaInfoRow),
            0};
        AscendC::SetCtrlSpr<60, 60>(0);
        blockEpilogue_(epilogueShape, epilogueOffset);
        sync.NotifyDownstreamReady(subMLoc);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
}

template <typename BlockEpilogue, bool TopkWeightsPrefetch, bool IsInterleaved>
template <typename Config, typename ActualShape, typename SyncPolicy>
__aicore__ inline void
Gmm1ActivationQuantEpilogue<BlockEpilogue, TopkWeightsPrefetch, IsInterleaved>::
    Gmm1Aiv0InterleavedEpilogueTileGeneric(const Config &config, const ActualShape &actualShape, uint32_t mLoc,
                                           uint32_t nLoc, uint16_t pingpongIdx, SyncPolicy &sync)
{
    if (Get<M_VALUE>(actualShape) == 0U) {
        return;
    }
    uint32_t epilogueN = Get<N_VALUE>(actualShape) / ACTIVATION_N_HALF;
    uint32_t epilogueNLoc = nLoc / ACTIVATION_N_HALF;
    Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{Get<M_VALUE>(actualShape), epilogueN, 0, 0};
    Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
        mLoc * config.outputN + epilogueNLoc,
        mLoc * Ops::Base::CeilDiv<uint32_t>(config.outputN, static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE +
            Ops::Base::CeilDiv<uint32_t>(epilogueNLoc, static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE,
        static_cast<int64_t>(mLoc / L1_TILE_M_256),
        0,
        0,
        0};
    AscendC::SetCtrlSpr<60, 60>(0);
    blockEpilogue_(epilogueShape, epilogueOffset, pingpongIdx);
    sync.NotifyDownstreamReady(mLoc);
}

template <typename BlockEpilogue, bool TopkWeightsPrefetch, bool IsInterleaved>
template <typename Config, typename ActualShape, typename SyncPolicy>
__aicore__ inline void
Gmm1ActivationQuantEpilogue<BlockEpilogue, TopkWeightsPrefetch, IsInterleaved>::
    Gmm1Aiv0EpilogueTileGeneric(const Config &config, const ActualShape &actualShape, uint32_t mLoc, uint32_t nLoc,
                                SyncPolicy &sync)
{
    if (Get<M_VALUE>(actualShape) == 0U) {
        return;
    }
    Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{Get<M_VALUE>(actualShape),
                                                                 Get<N_VALUE>(actualShape), 0, 0};
    Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
        mLoc * config.outputN + nLoc,
        mLoc * Ops::Base::CeilDiv<uint32_t>(config.outputN, static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE +
            Ops::Base::CeilDiv<uint32_t>(nLoc, static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE,
        0,
        0,
        0,
        0};
    AscendC::SetCtrlSpr<60, 60>(0);
    blockEpilogue_(epilogueShape, epilogueOffset);
    sync.NotifyDownstreamReady(mLoc);
}

template <typename BlockEpilogue, bool TopkWeightsPrefetch, bool IsInterleaved>
template <typename MakeLayoutC, typename Config, typename TensorC, typename ActualShape, typename SyncPolicy>
__aicore__ inline void
Gmm1ActivationQuantEpilogue<BlockEpilogue, TopkWeightsPrefetch, IsInterleaved>::
    Gmm1Aiv1PrefetchEpilogueTileA8W4(const Config &config, TensorC &l0cOutGm, const ActualShape &actualShape,
                                     uint32_t mLoc, uint32_t nLoc, SyncPolicy &sync)
{
    uint32_t mLen = Get<M_VALUE>(actualShape);
    if (mLen == 0U) {
        return;
    }
    constexpr uint32_t subTileM = L1_TILE_M_128;
    uint64_t firstMetaInfoRow = expertBeforeCnt_ + mLoc;
    auto topkWeightTensor = blockEpilogue_.GetTopkWeightTensor();
    AscendC::DataCopy(topkWeightTensor, metaInfoGm_[firstMetaInfoRow * INT32_PER_256B],
                      static_cast<uint32_t>(mLen < subTileM ? mLen : subTileM) * INT32_PER_256B);
    auto layoutL0cUb = MakeLayoutC{}(subTileM, L1_TILE_N);
    int64_t ubOffsetFirst = 0;
    int64_t ubOffsetSecond = static_cast<int64_t>(MAX_SINGLE_MN_ALIGN32_NUM_128) * sizeof(typename BlockEpilogue::DataTypeIn);
    auto tensorBlockUbFirst =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, typename BlockEpilogue::DataTypeIn>(ubOffsetFirst), layoutL0cUb);
    auto tensorBlockUbSecond =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, typename BlockEpilogue::DataTypeIn>(ubOffsetSecond), layoutL0cUb);
    auto copyGm2Ub = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    for (uint32_t subOffset = 0; subOffset < mLen; subOffset += subTileM) {
        uint32_t subM = mLen - subOffset < subTileM ? mLen - subOffset : subTileM;
        uint32_t subMLoc = mLoc + subOffset;
        uint64_t metaInfoRow = expertBeforeCnt_ + subMLoc;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        if (subOffset != 0) {
            AscendC::DataCopy(topkWeightTensor, metaInfoGm_[metaInfoRow * INT32_PER_256B],
                              subM * INT32_PER_256B);
        }
        auto tensorBlockGmFirst =
            l0cOutGm.Slice(Te::MakeCoord(subMLoc, nLoc), Te::MakeShape(subM, Get<N_VALUE>(actualShape)));
        auto tensorBlockGmSecond = l0cOutGm.Slice(Te::MakeCoord(subMLoc, nLoc + config.outputN),
                                                  Te::MakeShape(subM, Get<N_VALUE>(actualShape)));
        AscendC::Te::Copy(copyGm2Ub, tensorBlockUbFirst, tensorBlockGmFirst);
        AscendC::Te::Copy(copyGm2Ub, tensorBlockUbSecond, tensorBlockGmSecond);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{subM, Get<N_VALUE>(actualShape), 0, 0};
        Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
            subMLoc * config.outputN + nLoc,
            subMLoc * Ops::Base::CeilDiv<uint32_t>(config.outputN, static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE +
                Ops::Base::CeilDiv<uint32_t>(nLoc, static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE,
            0,
            0,
            static_cast<int64_t>(metaInfoRow),
            0};
        AscendC::SetCtrlSpr<60, 60>(0);
        blockEpilogue_(epilogueShape, epilogueOffset);
        sync.NotifyDownstreamReady(subMLoc);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
}

template <typename BlockEpilogue, bool TopkWeightsPrefetch, bool IsInterleaved>
template <typename MakeLayoutC, typename Config, typename TensorC, typename ActualShape, typename SyncPolicy>
__aicore__ inline void
Gmm1ActivationQuantEpilogue<BlockEpilogue, TopkWeightsPrefetch, IsInterleaved>::
    Gmm1Aiv1EpilogueTileA8W4(const Config &config, TensorC &l0cOutGm, const ActualShape &actualShape, uint32_t mLoc,
                             uint32_t nLoc, SyncPolicy &sync)
{
    if (Get<M_VALUE>(actualShape) == 0U) {
        return;
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
    auto tensorBlockGmFirst = l0cOutGm.Slice(Te::MakeCoord(mLoc, nLoc),
                                             Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
    auto tensorBlockGmSecond = l0cOutGm.Slice(Te::MakeCoord(mLoc, nLoc + config.outputN),
                                              Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
    auto layoutL0cUb = MakeLayoutC{}(config.tileM, L1_TILE_N);
    int64_t ubOffsetFirst = 0;
    uint32_t ubBufferSize =
        config.tileM == L1_TILE_M_128 ? MAX_SINGLE_MN_ALIGN32_NUM_128 : MAX_SINGLE_MN_ALIGN32_NUM_256;
    int64_t ubOffsetSecond = static_cast<int64_t>(ubBufferSize) * sizeof(typename BlockEpilogue::DataTypeIn);
    auto tensorBlockUbFirst =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, typename BlockEpilogue::DataTypeIn>(ubOffsetFirst), layoutL0cUb);
    auto tensorBlockUbSecond =
        Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, typename BlockEpilogue::DataTypeIn>(ubOffsetSecond), layoutL0cUb);
    auto copyGm2Ub = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
    AscendC::Te::Copy(copyGm2Ub, tensorBlockUbFirst, tensorBlockGmFirst);
    AscendC::Te::Copy(copyGm2Ub, tensorBlockUbSecond, tensorBlockGmSecond);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{Get<M_VALUE>(actualShape),
                                                                 Get<N_VALUE>(actualShape), 0, 0};
    Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
        mLoc * config.outputN + nLoc,
        mLoc * Ops::Base::CeilDiv<uint32_t>(config.outputN, static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE +
            Ops::Base::CeilDiv<uint32_t>(nLoc, static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE,
        0,
        0,
        0,
        0};
    AscendC::SetCtrlSpr<60, 60>(0);
    blockEpilogue_(epilogueShape, epilogueOffset);
    sync.NotifyDownstreamReady(mLoc);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_GMM1_EPILOGUE_TILE_H
