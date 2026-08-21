/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GMM1_ACTIVATION_H
#define MEGA_MOE_GMM1_ACTIVATION_H

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "tensor_api/tensor.h"
#include "apace/block/epilogue/block_epilogue_activation_mx_quant.h"
#include "apace/kernel/matmul/moe_grouped_matmul/mega_moe_gmm1_kernel.h"
#include "apace/kernel/matmul/moe_grouped_matmul/mega_moe_gmm_common.h"

namespace MegaMoeImpl {

using namespace AscendC;
using Apace::Gemm::Block::BlockEpilogueActivationMxQuant;
using Apace::Gemm::GmmParams;

constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM_256 = 256U * 256U;
constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM_128 = 128U * 256U;

struct Gmm1ActivationState {
    uint32_t &startBlockIdx;
    int32_t &vecSetSyncCom;
    int32_t &gmTileSequence;
    uint16_t &pingpongIdx;
};

using SharedExpertGmm1ActivationState = Gmm1ActivationState;

// 等待当前 GMM1 tile 对应的 Dispatch 输入就绪。
template <bool IsShared, typename Config>
__aicore__ inline void WaitForGmm1InputReady(const GMMAddrInfo &gmmAddrInfo, const Config &config, uint32_t mLoc)
{
    if constexpr (IsShared) {
        return;
    }
    uint32_t waveIdx = mLoc / config.tileM;
    uint32_t targetValue = (mLoc + config.tileM > config.m) ? (config.m - mLoc) : config.tileM;
    uint64_t flagOffset = static_cast<uint64_t>(waveIdx) * INT_CACHELINE;
    __gm__ int32_t *flagValueAddr = gmmAddrInfo.dispatchToGmm1Flag + flagOffset;
    while (targetValue != AscendC::ReadGmByPassDCache(flagValueAddr)) {
        int64_t st = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - st < 100) {
        }
    }
}

// 通知 GMM2 当前 M 分组的 activation + SwiGLU 已完成。
template <bool IsWaveFlagGrained>
__aicore__ inline void NotifyGmm2InputReady(__gm__ int32_t *flagBase, uint32_t flagSlotIdx = 0U)
{
    if (flagBase == nullptr) {
        return;
    }
    if constexpr (IsWaveFlagGrained) {
        flagBase += static_cast<uint64_t>(flagSlotIdx) * INT_CACHELINE;
    }
    AscendC::AtomicAdd(flagBase, 1);
}

// Prefetch 路径中 AIC 写完 GM tile 后通知 AIV（写 gmm1TileStatus）。
__aicore__ inline void NotifyGmm1TileStatus(const GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx, uint32_t loopIdx)
{
    __gm__ int32_t *statusAddr = gmmAddrInfo.gmm1TileStatus + static_cast<uint64_t>(loopIdx) * INT_CACHELINE;
    AscendC::WriteGmByPassDCache(statusAddr, static_cast<int32_t>(expertIdx + 1));
}

// Prefetch 路径中 AIV 等待 AIC 写完对应 GM tile（轮询 gmm1TileStatus）。
__aicore__ inline void WaitGmm1TileStatus(const GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx, uint32_t loopIdx)
{
    __gm__ int32_t *statusAddr = gmmAddrInfo.gmm1TileStatus + static_cast<uint64_t>(loopIdx) * INT_CACHELINE;
    int32_t roundTag = static_cast<int32_t>(expertIdx + 1);
    while (AscendC::ReadGmByPassDCache(statusAddr) != roundTag) {
        int64_t startCycle = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - startCycle < 100) {
        }
    }
}


// PipelineSync 适配器：将 MoE 同步函数封装为 apace GMM kernel 期望的 PipelineSync 接口。
template <bool IsShared, bool IsWaveFlagGrained>
struct MegaMoeGmm1PipelineSync {
    const GMMAddrInfo &gmmAddrInfo_;
    uint32_t tileM_;
    uint32_t m_;

    // GMM1 等待 Dispatch 输入就绪。
    // 共享专家无 Dispatch 依赖，编译期直接返回。
    __aicore__ inline void WaitGmm1Input(uint32_t mLoc)
    {
        if constexpr (IsShared) {
            return;
        }
        uint32_t waveIdx = mLoc / tileM_;
        uint32_t targetValue = (mLoc + tileM_ > m_) ? (m_ - mLoc) : tileM_;
        uint64_t flagOffset = static_cast<uint64_t>(waveIdx) * INT_CACHELINE;
        __gm__ int32_t *flagValueAddr = gmmAddrInfo_.dispatchToGmm1Flag + flagOffset;
        while (targetValue != AscendC::ReadGmByPassDCache(flagValueAddr)) {
            int64_t st = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - st < 100) {
            }
        }
    }

    // GMM1→epilogue，机制1 — CrossCore（非 Prefetch 路径）。
    // value=0 为非 interleaved 路径，value=pingpongIdx 为 interleaved 路径。
    __aicore__ inline void NotifyGmm1ToEpilogue(uint16_t value = 0)
    {
        NotifyVector(value);
    }

    __aicore__ inline void WaitGmm1ToEpilogue(uint16_t value = 0)
    {
        WaitForCube(value);
    }

    __aicore__ inline void NotifyEpilogueToGmm1(uint16_t value = 0)
    {
        NotifyCube(value);
    }

    __aicore__ inline void WaitEpilogueToGmm1(uint16_t value = 0)
    {
        WaitForVector(value);
    }

    // GMM1→epilogue，机制2 — GM tile-status（Prefetch 路径）。
    __aicore__ inline void NotifyGmm1ToEpilogue(uint32_t expertIdx, uint32_t loopIdx)
    {
        NotifyGmm1TileStatus(gmmAddrInfo_, expertIdx, loopIdx);
    }

    __aicore__ inline void WaitGmm1ToEpilogue(uint32_t expertIdx, uint32_t loopIdx)
    {
        WaitGmm1TileStatus(gmmAddrInfo_, expertIdx, loopIdx);
    }

    // GMM1→epilogue，机制3 — GM sequence（A8W4 非 Prefetch 路径）。
    __aicore__ inline void NotifyGmm1ToEpilogue(int32_t &gmTileSequence)
    {
        AscendC::WriteGmByPassDCache(gmmAddrInfo_.gmmToEpilogueFlag, ++gmTileSequence);
    }

    __aicore__ inline void WaitGmm1ToEpilogue(int32_t expectedReadySequence)
    {
        while (AscendC::ReadGmByPassDCache(gmmAddrInfo_.gmmToEpilogueFlag) < expectedReadySequence) {
            int64_t startCycle = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - startCycle < 100) {
            }
        }
    }

    // epilogue→GMM2，通知 GMM2 activation + SwiGLU 完成（写 activationToGmm2Flag）。
    __aicore__ inline void NotifyGmm2Input(uint32_t mLoc)
    {
        NotifyGmm2InputReady<IsWaveFlagGrained>(gmmAddrInfo_.activationToGmm2Flag, mLoc / L1_TILE_M_256);
    }
};

// GMM1 epilogue 的 moe 适配器
template <typename BlockEpilogue, typename PipelineSync, uint32_t Gmm1TileM, typename GmmElementC = bfloat16_t>
class Gmm1ActivationEpilogueOp {
public:
    using ElementC = GmmElementC;
    static constexpr uint32_t C0_SIZE_C = AscendC::AuxGetC0Size<GmmElementC>();
    using MakeLayoutC = AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn, AscendC::Std::Int<C0_SIZE_C>>;
    using ProblemShape = typename BlockEpilogue::ProblemShape;

    __aicore__ inline Gmm1ActivationEpilogueOp(BlockEpilogue &activationQuantOp, PipelineSync &pipelineSync,
                                               uint32_t outputN, uint32_t expertBeforeCnt, uint16_t &pingpongIdx)
        : activationQuantOp_(activationQuantOp), pipelineSync_(pipelineSync), outputN_(outputN),
          expertBeforeCnt_(expertBeforeCnt), pingpongIdx_(pingpongIdx) {}

    // A8W4 非 Prefetch 路径
    template <bool EnableA8W4, bool IsPrefetch, bool IsInterleaved, typename GmmOutTensor>
    __aicore__ inline void Run(uint32_t mPos, uint32_t nPos, uint32_t mLen, uint32_t nLen,
                               GmmOutTensor &gmC)
    {
        AscendC::GlobalTensor<float> emptyMetaInfoGm;
        Run<EnableA8W4, IsPrefetch, IsInterleaved>(mPos, nPos, mLen, nLen, gmC, emptyMetaInfoGm);
    }

    template <bool EnableA8W4, bool IsPrefetch, bool IsInterleaved, typename GmmOutTensor, typename MetaInfoTensor>
    __aicore__ inline void Run(uint32_t mPos, uint32_t nPos, uint32_t mLen, uint32_t nLen,
                               GmmOutTensor &gmC, MetaInfoTensor &metaInfoGm)
    {
        if constexpr (EnableA8W4) {
            if constexpr (IsPrefetch) {
                Gmm1Aiv1PrefetchEpilogueTileA8W4(mPos, nPos, mLen, nLen, gmC, metaInfoGm);
            } else {
                Gmm1Aiv1EpilogueTileA8W4(mPos, nPos, mLen, nLen, gmC);
            }
        } else {
            if constexpr (IsPrefetch) {
                if constexpr (IsInterleaved) {
                    Gmm1Aiv0PrefetchInterleavedEpilogueTileGeneric(mPos, nPos, mLen, nLen, gmC, metaInfoGm);
                } else {
                    Gmm1Aiv0PrefetchEpilogueTileGeneric(mPos, nPos, mLen, nLen, gmC, metaInfoGm);
                }
            } else {
                if constexpr (IsInterleaved) {
                    Gmm1Aiv0InterleavedEpilogueTileGeneric(mPos, nPos, mLen, nLen);
                } else {
                    Gmm1Aiv0EpilogueTileGeneric(mPos, nPos, mLen, nLen);
                }
            }
        }
    }

    __aicore__ inline void UpdateNextProblem(const ProblemShape &problemShape)
    {
        activationQuantOp_.UpdateNextProblem(problemShape);
    }

private:
    // Prefetch 交错路径从 GM 取回一个子 tile，执行 SwiGLU/量化并通知 GMM2。
    template <typename GmmOutTensor, typename MetaInfoTensor>
    __aicore__ inline void Gmm1Aiv0PrefetchInterleavedEpilogueTileGeneric(uint32_t mPos, uint32_t nPos, uint32_t mLen, uint32_t nLen,
                                                  GmmOutTensor &gmC, MetaInfoTensor &metaInfoGm)
    {
        if (mLen == 0U) {
            return;
        }
        constexpr uint32_t subTileM = L1_TILE_M_128;
        auto layoutL0cUb = MakeLayoutC{}(subTileM, L1_TILE_N);
        auto tensorBlockUb = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(0), layoutL0cUb);
        auto copyGm2Ub = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
        auto topkWeightTensor = activationQuantOp_.GetTopkWeightTensor();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        for (uint32_t subOffset = 0; subOffset < mLen; subOffset += subTileM) {
            uint32_t subM = mLen - subOffset < subTileM ? mLen - subOffset : subTileM;
            uint32_t subMLoc = mPos + subOffset;
            uint64_t metaInfoRow = expertBeforeCnt_ + subMLoc;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::DataCopy(topkWeightTensor, metaInfoGm[metaInfoRow * INT32_PER_256B], subM * INT32_PER_256B);
            auto tensorBlockGm = gmC.Slice(Te::MakeCoord(subMLoc, nPos), Te::MakeShape(subM, nLen));
            AscendC::Te::Copy(copyGm2Ub, tensorBlockUb, tensorBlockGm);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

            uint32_t epilogueN = nLen / ACTIVATION_N_HALF;
            uint32_t epilogueNLoc = nPos / ACTIVATION_N_HALF;
            Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{subM, epilogueN, 0, 0};
            Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                subMLoc * outputN_ + epilogueNLoc,
                subMLoc * CeilDiv(outputN_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                    CeilDiv(epilogueNLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
                static_cast<int64_t>(subMLoc / L1_TILE_M_256),
                0,
                static_cast<int64_t>(metaInfoRow),
                0};
            AscendC::SetCtrlSpr<60, 60>(0);
            activationQuantOp_(epilogueShape, epilogueOffset, 0U);
            pipelineSync_.NotifyGmm2Input(subMLoc);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    }

    // 非 Prefetch 交错路径直接消费 AIC 的 UB ping-pong tile。
    __aicore__ inline void Gmm1Aiv0InterleavedEpilogueTileGeneric(uint32_t mPos, uint32_t nPos, uint32_t mLen, uint32_t nLen)
    {
        if (mLen == 0U) {
            return;
        }
        uint32_t epilogueN = nLen / ACTIVATION_N_HALF;
        uint32_t epilogueNLoc = nPos / ACTIVATION_N_HALF;
        Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{mLen, epilogueN, 0, 0};
        Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
            mPos * outputN_ + epilogueNLoc,
            mPos * CeilDiv(outputN_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                CeilDiv(epilogueNLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
            static_cast<int64_t>(mPos / L1_TILE_M_256),
            0,
            0,
            0};
        AscendC::SetCtrlSpr<60, 60>(0);
        activationQuantOp_(epilogueShape, epilogueOffset, pingpongIdx_);
        pipelineSync_.NotifyGmm2Input(mPos);
    }

    // Prefetch 非交错路径从 GM 取回两路 GMM1 输出，再执行 SwiGLU/量化。
    template <typename GmmOutTensor, typename MetaInfoTensor>
    __aicore__ inline void Gmm1Aiv0PrefetchEpilogueTileGeneric(uint32_t mPos, uint32_t nPos, uint32_t mLen, uint32_t nLen,
                                       GmmOutTensor &gmC, MetaInfoTensor &metaInfoGm)
    {
        if (mLen == 0U) {
            return;
        }
        constexpr uint32_t subTileM = L1_TILE_M_128;
        uint64_t firstMetaInfoRow = expertBeforeCnt_ + mPos;
        auto topkWeightTensor = activationQuantOp_.GetTopkWeightTensor();
        AscendC::DataCopy(topkWeightTensor, metaInfoGm[firstMetaInfoRow * INT32_PER_256B],
                          static_cast<uint32_t>(mLen < subTileM ? mLen : subTileM) * INT32_PER_256B);

        auto layoutL0cUb = MakeLayoutC{}(subTileM, L1_TILE_N);
        int64_t ubOffsetFirst = 0;
        int64_t ubOffsetSecond = static_cast<int64_t>(MAX_SINGLE_MN_ALIGN32_NUM_128) * sizeof(ElementC);
        auto tensorBlockUbFirst = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetFirst), layoutL0cUb);
        auto tensorBlockUbSecond =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetSecond), layoutL0cUb);
        auto copyGm2Ub = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        for (uint32_t subOffset = 0; subOffset < mLen; subOffset += subTileM) {
            uint32_t subM = mLen - subOffset < subTileM ? mLen - subOffset : subTileM;
            uint32_t subMLoc = mPos + subOffset;
            uint64_t metaInfoRow = expertBeforeCnt_ + subMLoc;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            if (subOffset != 0) {
                AscendC::DataCopy(topkWeightTensor, metaInfoGm[metaInfoRow * INT32_PER_256B], subM * INT32_PER_256B);
            }

            auto tensorBlockGmFirst = gmC.Slice(Te::MakeCoord(subMLoc, nPos), Te::MakeShape(subM, nLen));
            auto tensorBlockGmSecond = gmC.Slice(Te::MakeCoord(subMLoc, nPos + outputN_), Te::MakeShape(subM, nLen));
            AscendC::Te::Copy(copyGm2Ub, tensorBlockUbFirst, tensorBlockGmFirst);
            AscendC::Te::Copy(copyGm2Ub, tensorBlockUbSecond, tensorBlockGmSecond);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

            Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{subM, nLen, 0, 0};
            Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                subMLoc * outputN_ + nPos,
                subMLoc * CeilDiv(outputN_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                    CeilDiv(nPos, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
                0,
                0,
                static_cast<int64_t>(metaInfoRow),
                0};
            AscendC::SetCtrlSpr<60, 60>(0);
            activationQuantOp_(epilogueShape, epilogueOffset);
            pipelineSync_.NotifyGmm2Input(subMLoc);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    }

    // 非 Prefetch 非交错路径直接消费 AIC 的 UB tile。
    __aicore__ inline void Gmm1Aiv0EpilogueTileGeneric(uint32_t mPos, uint32_t nPos, uint32_t mLen, uint32_t nLen)
    {
        if (mLen == 0U) {
            return;
        }
        Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{mLen, nLen, 0, 0};
        Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
            mPos * outputN_ + nPos,
            mPos * CeilDiv(outputN_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                CeilDiv(nPos, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
            0,
            0,
            0,
            0};
        AscendC::SetCtrlSpr<60, 60>(0);
        activationQuantOp_(epilogueShape, epilogueOffset);
        pipelineSync_.NotifyGmm2Input(mPos);
    }

    // Prefetch 路径的 AIV1 从 GM 取回 A8W4 GMM1 子 tile，执行 SwiGLU/量化并发布完成状态。
    template <typename GmmOutTensor, typename MetaInfoTensor>
    __aicore__ inline void Gmm1Aiv1PrefetchEpilogueTileA8W4(uint32_t mPos, uint32_t nPos, uint32_t mLen, uint32_t nLen,
                                           GmmOutTensor &gmC, MetaInfoTensor &metaInfoGm)
    {
        if (mLen == 0U) {
            return;
        }

        constexpr uint32_t subTileM = L1_TILE_M_128;
        uint64_t firstMetaInfoRow = expertBeforeCnt_ + mPos;
        auto topkWeightTensor = activationQuantOp_.GetTopkWeightTensor();
        AscendC::DataCopy(topkWeightTensor, metaInfoGm[firstMetaInfoRow * INT32_PER_256B],
                          static_cast<uint32_t>(mLen < subTileM ? mLen : subTileM) * INT32_PER_256B);
        auto layoutL0cUb = MakeLayoutC{}(subTileM, L1_TILE_N);
        int64_t ubOffsetFirst = 0;
        int64_t ubOffsetSecond = static_cast<int64_t>(MAX_SINGLE_MN_ALIGN32_NUM_128) * sizeof(ElementC);
        auto tensorBlockUbFirst =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetFirst), layoutL0cUb);
        auto tensorBlockUbSecond =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetSecond), layoutL0cUb);
        auto copyGm2Ub = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        for (uint32_t subOffset = 0; subOffset < mLen; subOffset += subTileM) {
            uint32_t subM = mLen - subOffset < subTileM ? mLen - subOffset : subTileM;
            uint32_t subMLoc = mPos + subOffset;
            uint64_t metaInfoRow = expertBeforeCnt_ + subMLoc;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            if (subOffset != 0) {
                AscendC::DataCopy(topkWeightTensor, metaInfoGm[metaInfoRow * INT32_PER_256B], subM * INT32_PER_256B);
            }
            auto tensorBlockGmFirst = gmC.Slice(Te::MakeCoord(subMLoc, nPos), Te::MakeShape(subM, nLen));
            auto tensorBlockGmSecond = gmC.Slice(Te::MakeCoord(subMLoc, nPos + outputN_), Te::MakeShape(subM, nLen));
            AscendC::Te::Copy(copyGm2Ub, tensorBlockUbFirst, tensorBlockGmFirst);
            AscendC::Te::Copy(copyGm2Ub, tensorBlockUbSecond, tensorBlockGmSecond);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{subM, nLen, 0, 0};
            Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                subMLoc * outputN_ + nPos,
                subMLoc * CeilDiv(outputN_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                    CeilDiv(nPos, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
                0,
                0,
                static_cast<int64_t>(metaInfoRow),
                0};
            AscendC::SetCtrlSpr<60, 60>(0);
            activationQuantOp_(epilogueShape, epilogueOffset);
            pipelineSync_.NotifyGmm2Input(subMLoc);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    }

    // 非 Prefetch 路径的 AIV1 从 GM 取回完整 A8W4 GMM1 tile 并执行 SwiGLU/量化。
    template <typename GmmOutTensor>
    __aicore__ inline void Gmm1Aiv1EpilogueTileA8W4(uint32_t mPos, uint32_t nPos, uint32_t mLen, uint32_t nLen,
                                         GmmOutTensor &gmC)
    {
        if (mLen == 0U) {
            return;
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
        auto tensorBlockGmFirst = gmC.Slice(Te::MakeCoord(mPos, nPos), Te::MakeShape(mLen, nLen));
        auto tensorBlockGmSecond = gmC.Slice(Te::MakeCoord(mPos, nPos + outputN_), Te::MakeShape(mLen, nLen));
        auto layoutL0cUb = MakeLayoutC{}(Gmm1TileM, L1_TILE_N);
        int64_t ubOffsetFirst = 0;
        uint32_t ubBufferSize =
            Gmm1TileM == L1_TILE_M_128 ? MAX_SINGLE_MN_ALIGN32_NUM_128 : MAX_SINGLE_MN_ALIGN32_NUM_256;
        int64_t ubOffsetSecond = static_cast<int64_t>(ubBufferSize) * sizeof(ElementC);
        auto tensorBlockUbFirst =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetFirst), layoutL0cUb);
        auto tensorBlockUbSecond =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetSecond), layoutL0cUb);
        auto copyGm2Ub = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
        AscendC::Te::Copy(copyGm2Ub, tensorBlockUbFirst, tensorBlockGmFirst);
        AscendC::Te::Copy(copyGm2Ub, tensorBlockUbSecond, tensorBlockGmSecond);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{mLen, nLen, 0, 0};
        Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
            mPos * outputN_ + nPos,
            mPos * CeilDiv(outputN_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                CeilDiv(nPos, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
            0,
            0,
            0,
            0};
        AscendC::SetCtrlSpr<60, 60>(0);
        activationQuantOp_(epilogueShape, epilogueOffset);
        pipelineSync_.NotifyGmm2Input(mPos);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
    }

private:
    BlockEpilogue &activationQuantOp_;
    PipelineSync &pipelineSync_;
    uint32_t outputN_;
    uint32_t expertBeforeCnt_;
    uint16_t &pingpongIdx_;
};

template <typename ActivationType, typename WeightType, typename ActivationOutType, typename QuantScaleType,
          uint32_t ActivationElementsPerByte, bool EnableA8W4, bool TopkWeightsPrefetch, typename BlockEpilogue>
__aicore__ inline void UpdateMoeExpertGmm1GlobalBuffer(
    const GmmExecutionConfig &gmmConfig, const MoeSyncWorkspaceLayout &syncLayout, const WorkspaceInfo &workspace,
    const ExpertWeightTensorListAddrs &weights, BlockEpilogue &epilogueOp, GMMAddrInfo &gmmAddrInfo,
    const ExpertLoopState &state, uint32_t rowOffsetInExpert = 0U, uint32_t gmm1TilesPerMGroup = 0U)
{
    if constexpr (g_coreType == AIV && !EnableA8W4) {
        if (GetSubBlockIdx() != 0) {
            return;
        }
    }

    constexpr uint32_t weightElementsPerByte = PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE;
    constexpr uint32_t outputElementsPerByte = PackedElementTraits<ActivationOutType>::ELEMENTS_PER_BYTE;
    int64_t n = Get<N_VALUE>(state.problemShape);
    int64_t k = Get<K_VALUE>(state.problemShape);
    int64_t expertOffset = state.expertIdx;
    int64_t globalTokenStartIndex = state.globalTokenStartIndex + rowOffsetInExpert;
    uint32_t expertMGroupOffset = rowOffsetInExpert / L1_TILE_M_256;
    int64_t scaleK = Ops::Base::CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    if constexpr (EnableA8W4 || TopkWeightsPrefetch) {
        gmmAddrInfo.gmm1OutGlobal = workspace.gmm1MmadResPtr + globalTokenStartIndex * n * sizeof(bfloat16_t);
    }
    if constexpr (TopkWeightsPrefetch) {
        gmmAddrInfo.gmm1TileStatus = reinterpret_cast<__gm__ int32_t *>(workspace.gmm1TileStatusPtr) +
                                     (static_cast<uint64_t>(state.expertIdx) * syncLayout.gmm1TileStatusCountPerExpert +
                                      static_cast<uint64_t>(expertMGroupOffset) * gmm1TilesPerMGroup) *
                                         INT_CACHELINE;
    }
    gmmAddrInfo.metaInfoGlobal = workspace.metaInfoPtr;
    gmmAddrInfo.activationToGmm2Flag = reinterpret_cast<__gm__ int32_t *>(workspace.flagActivationToGmm2Ptr) +
                                       expertOffset * syncLayout.activationFlagSlotCountPerExpert +
                                       static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
    gmmAddrInfo.aGlobal =
        workspace.dispatchRevDataPtr + globalTokenStartIndex * k / ActivationElementsPerByte * sizeof(ActivationType);
    gmmAddrInfo.aScaleGlobal = workspace.dispatchRevScalePtr + globalTokenStartIndex * scaleK * sizeof(QuantScaleType);
    gmmAddrInfo.bGlobal =
        GetExpertWeightAddr<ActivationType>(weights.weight1, gmmConfig.isPerExpertWeightTensor, state.expertIdx,
                                            static_cast<uint64_t>(state.expertIdx) * n * k / weightElementsPerByte);
    gmmAddrInfo.bScaleGlobal =
        GetExpertWeightAddr<QuantScaleType>(weights.weightScales1, gmmConfig.isPerExpertWeightTensor, state.expertIdx,
                                            static_cast<uint64_t>(state.expertIdx) * n * scaleK);
    if constexpr (g_coreType == AIV) {
        bool runsActivation = true;
        if constexpr (EnableA8W4) {
            runsActivation = GetSubBlockIdx() == 1;
        }
        if (runsActivation) {
            int64_t scaleN = Ops::Base::CeilDiv(n / ACTIVATION_N_HALF, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                             MXFP_MULTI_BASE_SIZE;
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                globalTokenStartIndex * n / ACTIVATION_N_HALF / outputElementsPerByte,
                globalTokenStartIndex * scaleN,
                expertOffset * syncLayout.activationFlagSlotCountPerExpert / INT_CACHELINE + expertMGroupOffset,
                0L,
                0L,
                0L};
            epilogueOp.UpdateGlobalAddr(vecBaseOffset);
        }
    }
    if constexpr (g_coreType == AIC) {
        gmmAddrInfo.dispatchToGmm1Flag = reinterpret_cast<__gm__ int32_t *>(workspace.flagDispatchToGmm1Ptr) +
                                         expertOffset * syncLayout.dispatchFlagSlotCountPerExpert +
                                         static_cast<uint64_t>(expertMGroupOffset) * INT_CACHELINE;
    }
    if constexpr (EnableA8W4) {
        gmmAddrInfo.gmmToEpilogueFlag = nullptr;
        if (workspace.flagGmmToEpiloguePtr != nullptr) {
            gmmAddrInfo.gmmToEpilogueFlag = reinterpret_cast<__gm__ int32_t *>(workspace.flagGmmToEpiloguePtr) +
                                            static_cast<uint64_t>(gmmConfig.blockJob.jobIndex) * INT_CACHELINE;
        }
    }
}

// RunGmm1ActivationByMode: 构造 PipelineSync 和 EpilogueOp 适配器，委托 apace 执行。
// IsShared=true 供共享专家路径使用；EpilogueTileM 保留在签名中以兼容调用方；
template <typename QuantOutType, typename WeightType, typename ActivationOutType, typename QuantScaleType,
          bool EnableA8W4, uint32_t Gmm1TileM, uint32_t EpilogueTileM, bool TopkWeightsPrefetch,
          bool IsShared = false, bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false,
          typename BlockEpilogue>
__aicore__ inline void RunGmm1ActivationByMode(const GmmExecutionConfig &gmmConfig, const Params &gmmParams,
                                               BlockEpilogue &epilogueOp, const GMMAddrInfo &gmmAddrInfo,
                                               const ProblemShape &problemShape, uint32_t globalTokenStartIndex,
                                               Gmm1ActivationState &runtimeState, uint32_t expertIdx,
                                               void *persistentBlockMmadContext = nullptr,
                                               bool allowWeightL2Bypass = false)
{
    Apace::Gemm::GmmParams apaceGmmParams;
    apaceGmmParams.aGlobal = gmmAddrInfo.aGlobal;
    apaceGmmParams.bGlobal = gmmAddrInfo.bGlobal;
    apaceGmmParams.aScaleGlobal = gmmAddrInfo.aScaleGlobal;
    apaceGmmParams.bScaleGlobal = gmmAddrInfo.bScaleGlobal;
    apaceGmmParams.cGlobal = gmmAddrInfo.gmm1OutGlobal;
    apaceGmmParams.metaInfoGlobal = gmmAddrInfo.metaInfoGlobal;

    MegaMoeGmm1PipelineSync<IsShared, IsWaveFlagGrained> pipelineSync{
        gmmAddrInfo, Gmm1TileM, static_cast<uint32_t>(Get<M_VALUE>(problemShape))};

    uint32_t outputN = static_cast<uint32_t>(Get<N_VALUE>(problemShape)) / ACTIVATION_N_HALF;
    using Gmm1EpilogueOp =
        Gmm1ActivationEpilogueOp<BlockEpilogue, MegaMoeGmm1PipelineSync<IsShared, IsWaveFlagGrained>, Gmm1TileM>;
    Gmm1EpilogueOp gmm1Epilogue{epilogueOp, pipelineSync, outputN, globalTokenStartIndex, runtimeState.pingpongIdx};

    if constexpr (EnableA8W4) {
        Apace::Gemm::RunGmm1A8W4<QuantOutType, WeightType, bfloat16_t, QuantScaleType, QuantScaleType, Gmm1EpilogueOp,
                                  Gmm1TileM, TopkWeightsPrefetch, IsShared, IsWaveFlagGrained,
                                  MegaMoeGmm1PipelineSync<IsShared, IsWaveFlagGrained>>(
            gmm1Epilogue, apaceGmmParams, problemShape, pipelineSync, runtimeState.startBlockIdx,
            runtimeState.gmTileSequence, gmmConfig.blockJob, globalTokenStartIndex, expertIdx);
    } else if (gmmConfig.groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
               gmmConfig.groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
        Apace::Gemm::RunGmm1Generic<QuantOutType, QuantOutType, bfloat16_t, QuantScaleType, QuantScaleType,
                                    Gmm1EpilogueOp, true, Gmm1TileM, TopkWeightsPrefetch, IsShared, IsGmm1Interleaved,
                                    IsWaveFlagGrained, MegaMoeGmm1PipelineSync<IsShared, IsWaveFlagGrained>>(
            gmm1Epilogue, apaceGmmParams, problemShape, pipelineSync, runtimeState.startBlockIdx,
            runtimeState.vecSetSyncCom, gmmConfig.blockJob, globalTokenStartIndex, expertIdx,
            runtimeState.pingpongIdx, persistentBlockMmadContext, allowWeightL2Bypass);
    } else {
        Apace::Gemm::RunGmm1Generic<QuantOutType, QuantOutType, bfloat16_t, QuantScaleType, QuantScaleType,
                                    Gmm1EpilogueOp, false, Gmm1TileM, TopkWeightsPrefetch, IsShared, IsGmm1Interleaved,
                                    IsWaveFlagGrained, MegaMoeGmm1PipelineSync<IsShared, IsWaveFlagGrained>>(
            gmm1Epilogue, apaceGmmParams, problemShape, pipelineSync, runtimeState.startBlockIdx,
            runtimeState.vecSetSyncCom, gmmConfig.blockJob, globalTokenStartIndex, expertIdx,
            runtimeState.pingpongIdx, persistentBlockMmadContext, allowWeightL2Bypass);
    }
}

// 原型：MegaMoe::UpdateSharedGlobalBuffer<GMM1>。构造一个共享专家的 GMM1 和 SwiGLU GM 视图。
template <typename ActivationType, typename WeightType, typename ActivationOutType, typename QuantScaleType,
          bool EnableA8W4, typename BlockEpilogue>
__aicore__ inline void UpdateSharedExpertGmm1GlobalBuffer(const MoeStageCommonConfig &commonConfig,
                                                          const GmmExecutionConfig &gmmConfig,
                                                          const WorkspaceInfo &workspace,
                                                          const ExpertWeightTensorListAddrs &weights,
                                                          BlockEpilogue &epilogueOp, GMMAddrInfo &gmmAddrInfo,
                                                          uint32_t sharedExpertIdx)
{
    constexpr uint32_t weightElemsPerByte = PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE;
    constexpr uint32_t outputElemsPerByte = PackedElementTraits<ActivationOutType>::ELEMENTS_PER_BYTE;
    uint64_t m = commonConfig.tokenNum;
    uint64_t n = commonConfig.gmm1OutputDim;
    uint64_t k = commonConfig.tokenHiddenDim;
    uint64_t scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    uint64_t activationN = n / ACTIVATION_N_HALF;
    uint64_t scaleN = Ops::Base::CeilDiv(activationN, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    uint64_t expertIdx = sharedExpertIdx;

    gmmAddrInfo.aGlobal = workspace.sharedExpertInputDataPtr;
    gmmAddrInfo.aScaleGlobal = workspace.sharedExpertInputScalePtr;
    if constexpr (EnableA8W4) {
        gmmAddrInfo.gmm1OutGlobal = workspace.sharedExpertGmm1OutPtr + expertIdx * m * n * sizeof(bfloat16_t);
    }
    gmmAddrInfo.bGlobal = GetExpertWeightAddr<ActivationType>(weights.weight1, gmmConfig.isPerExpertWeightTensor,
                                                              sharedExpertIdx, expertIdx * n * k / weightElemsPerByte);
    gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleType>(
        weights.weightScales1, gmmConfig.isPerExpertWeightTensor, sharedExpertIdx, expertIdx * n * scaleK);
    gmmAddrInfo.activationToGmm2Flag = nullptr;
    gmmAddrInfo.metaInfoGlobal = nullptr;
    gmmAddrInfo.dispatchToGmm1Flag = nullptr;
    if constexpr (EnableA8W4) {
        gmmAddrInfo.gmmToEpilogueFlag = nullptr;
        if (workspace.flagGmmToEpiloguePtr != nullptr) {
            gmmAddrInfo.gmmToEpilogueFlag = reinterpret_cast<__gm__ int32_t *>(workspace.flagGmmToEpiloguePtr) +
                                            static_cast<uint64_t>(gmmConfig.blockJob.jobIndex) * INT_CACHELINE;
        }
    }

    if constexpr (g_coreType == AIV) {
        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
            static_cast<int64_t>(expertIdx * m * activationN / outputElemsPerByte),
            static_cast<int64_t>(expertIdx * m * scaleN),
            0L,
            0L,
            0L,
            0L};
        epilogueOp.UpdateGlobalAddr(vecBaseOffset);
    }
}

// 完成 MoE GMM1/SwiGLU 阶段的末尾同步。
template <bool EnableA8W4, bool TopkWeightsPrefetch>
__aicore__ inline void FinishMoeGmm1ActivationStage(const MoeStageCommonConfig &commonConfig,
                                                    const MoeSyncWorkspaceLayout &syncLayout,
                                                    const WorkspaceInfo &workspace, Gmm1ActivationState &runtimeState)
{
    if constexpr (TopkWeightsPrefetch) {
        int32_t allDoneTag = static_cast<int32_t>(commonConfig.moeExpertPerRank + 1U);
        __gm__ int32_t *allDoneAddr = reinterpret_cast<__gm__ int32_t *>(workspace.gmm1TileStatusPtr) +
                                      static_cast<uint64_t>(commonConfig.moeExpertPerRank) *
                                          syncLayout.gmm1TileStatusCountPerExpert * INT_CACHELINE;
        if constexpr (g_coreType == AIV) {
            constexpr uint32_t epilogueSubBlockIdx = EnableA8W4 ? 1U : 0U;
            if (GetSubBlockIdx() == epilogueSubBlockIdx) {
                AscendC::WriteGmByPassDCache(allDoneAddr, allDoneTag);
            }
        } else {
            while (AscendC::ReadGmByPassDCache(allDoneAddr) != allDoneTag) {
                int64_t startCycle = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - startCycle < 100) {
                }
            }
        }
    } else {
        EndSync(runtimeState.vecSetSyncCom);
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_GMM1_ACTIVATION_H
