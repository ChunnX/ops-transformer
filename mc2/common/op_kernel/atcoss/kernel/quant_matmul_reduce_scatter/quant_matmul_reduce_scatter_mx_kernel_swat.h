/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_mx_kernel_swat.h
 * \brief Custom GemmUniversal specialization for QuantMatmul + ReduceScatter.
 *
 * Based on Blaze's kernel_qbmm_mx.h, with the following QMMRS-specific customizations:
 *   - Params extended with yGM, hcclContext for ReduceScatter post-processing
 *   - Run() executes matmul+epilogue (AlltoAll) via ProcessSingleBatch/ProcessWithBatch,
 *     then AIV performs ReduceAdd (status sync + remote read + add + cast)
 *   - All ReduceAdd helpers (WriteStatus, ReadStatus, ReadRemoteDataAdd) are members of this class
 */

#pragma once

#include "adv_api/hccl/hccl.h"
#include "inc/kernel/moe_distribute_base.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "block_epilogue_all_to_all.h"
#include "include/tensor_api/tensor.h"
#include "quant_matmul_reduce_scatter_tiling_data.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {
#define QBMM_MX_KERNEL_CLASS_TEM_PARAMS                                                                                \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QBMM_MX_KERNEL_TEM_PARAMS                                                                                      \
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler,                                                            \
        AscendC::Std::enable_if_t<                                                                                     \
            AscendC::Std::is_same_v<KernelMmadWithScaleMx, typename BlockMmad::DispatchPolicy::ScheduleType>>

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
class GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS> {
public:
    __aicore__ inline GemmUniversal()
    {
    }
    __aicore__ inline ~GemmUniversal()
    {
    }

    static constexpr bool weightNz = BlockMmad::weightNz;
    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;
    static constexpr bool isAtomicAdd = BlockMmad::DispatchPolicy::isAtomicAdd;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutStatus = AscendC::Te::NDExtLayoutPtn;
    static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    static constexpr int64_t kCacheLineAlignMask = IsFp4<AType>() ? 0xff : 0x7f;
    static constexpr int32_t SCALE_C0 = 2;
    static constexpr uint8_t CROSS_CORE_MODE_ID_TWO = 0x2;
    static constexpr uint16_t CROSS_CORE_FLAG_ID_NINE = 9U;

    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;

    using BlockSchedulerParams = typename BlockScheduler::Params;
    using EpilogueOp = BlockEpilogue;

    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutC = AscendC::Te::FrameLayoutFormat<LayoutC, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<CType>>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        transA, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleADNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        transB, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBDNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBNDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;

    using MakeLayoutUB =
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<CType>>;
    using CopyMakerGM2UB = decltype(AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{}));
    using CopyMakerUB2GM = decltype(AscendC::Te::MakeCopy(AscendC::Te::CopyUB2GM{}));

    struct QBMMTiling {
        uint32_t batchA1;
        uint32_t batchA2;
        uint32_t batchA3;
        uint32_t batchA4;
        uint32_t batchB1;
        uint32_t batchB2;
        uint32_t batchB3;
        uint32_t batchB4;
        uint32_t batchC1;
        uint32_t batchC2;
        uint32_t batchC3;
        uint32_t batchC4;
        uint32_t biasThreeDim;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t isBias;
        uint32_t dbL0C;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        L1Params l1Params;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
        typename EpilogueOp::Params epilogueParams;
        GM_ADDR yGM{nullptr};
        GM_ADDR hcclContext{nullptr};
    };

    __aicore__ inline void Init(const Params &params);
    __aicore__ inline void Run(const Params &params);
    __aicore__ inline void operator()(const Params &params)
    {
        Run(params);
    }

private:
    __aicore__ inline void ResetGmAddr(const Params &params);
    __aicore__ inline void ProcessSingleBatch(const Params &params, BlockScheduler &bs, uint64_t batchCnt,
                                              bool isTailRound);
    __aicore__ inline void ProcessWithBatch(const Params &params, BlockScheduler &bs);
    __aicore__ inline void AddBatchOffset(const Params &params);

    template <typename TensorB, typename TensorScaleB, typename TensorC>
    __aicore__ inline void SetL2Cache(const ProblemShape &problemShape, uint64_t curBaseM, uint64_t baseN,
                                      uint64_t scaleKL1, TensorB &gmB, TensorScaleB &gmScaleB, TensorC &gmC);

    template <typename TensorScaleB>
    __aicore__ inline void SetScaleL2Cache(const ProblemShape &problemShape, uint64_t baseN, uint64_t scaleKL1,
                                           TensorScaleB &gmScaleB);

    __aicore__ inline void ReduceAddProcess();
    __aicore__ inline void WriteStatusToWin();
    __aicore__ inline void ReadStatus();
    __aicore__ inline void ReadRemoteDataAdd();
    __aicore__ inline void SplitToCore(const uint32_t curSendCnt, const uint32_t curUseAivNum, const uint32_t coreId,
                                       uint32_t &startId, uint32_t &endId, uint32_t &sendNum);

private:
    BlockMmad mmadOp_;
    EpilogueOp epilogueOp_;

    __gm__ AType *aGmAddr_ = nullptr;
    __gm__ BType *bGmAddr_ = nullptr;
    __gm__ CType *cGmAddr_ = nullptr;
    __gm__ BiasType *biasGmAddr_ = nullptr;
    __gm__ fp8_e8m0_t *scaleAGmAddr_ = nullptr;
    __gm__ fp8_e8m0_t *scaleBGmAddr_ = nullptr;

    uint64_t batchCOffset_{0};
    uint64_t batchAOffset_{0};
    uint64_t batchBOffset_{0};
    bool isBiasThreeDim_{false};
    bool isBias_{false};
    bool isSameBatch_{false};
    bool needUpdateTail_{false};

    __gm__ Mc2Kernel::HcclOpParam *winContext_{nullptr};
    uint32_t rankId_{0};
    uint32_t coreVid_{0};
    uint32_t tpWorldSize_{0};

    uint64_t m_{0};
    uint64_t n_{0};
    GM_ADDR yGM_{0};

    CopyMakerGM2UB copyGM2UB_ = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
    CopyMakerUB2GM copyUB2GM_ = AscendC::Te::MakeCopy(AscendC::Te::CopyUB2GM{});
};

constexpr static uint64_t UB_ALIGN_BYTES = 32UL;
constexpr uint32_t FLOAT_UB_ALIGN_NUM = 8U;
constexpr int32_t FLAG_ID_NUM_ONE = 1;
constexpr float STATUS_FLAG_VAL = -1.0f;
constexpr float STATUS_FLAG_THREDSHOLD = 0.5f;
constexpr uint32_t STATUS_POLL_MAX_RETRY = 1000000UL;

template <typename RoundMode, typename SatMode, typename IndexPos>
struct CastTrait {
    using roundMode = RoundMode;
    using satMode = SatMode;
    using indexPos = IndexPos;
};

using CastTraitF322Bf16 =
    CastTrait<AscendC::Te::CastRoundMode::RN, AscendC::Te::CastSatMode::NoSat, AscendC::Te::CastIndexPos::Even>;


QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::Run(const Params &params)
{
    m_ = static_cast<uint64_t>(AscendC::Te::Get<MNK_M>(params.problemShape));
    n_ = static_cast<uint64_t>(AscendC::Te::Get<MNK_N>(params.problemShape));
    tpWorldSize_ = params.epilogueParams.tpWorldSize;
    yGM_ = params.yGM;

    if ASCEND_IS_AIV {
        winContext_ = (__gm__ Mc2Kernel::HcclOpParam *)params.hcclContext;
        rankId_ = Mc2Kernel::GetRankId(winContext_);
        coreVid_ = AscendC::GetBlockIdx();
    }

    Init(params);
    BlockScheduler bs(params.problemShape, params.schParams);

    BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
    bool enableL0CPingPong = (params.qbmmParams.dbL0C > 1);
    if ASCEND_IS_AIC {
        mmadOp_.Init(params.problemShape, l0TileShape, params.l1Params, isBias_, enableL0CPingPong);
    }

    if (AscendC::Te::Get<MNK_B>(params.problemShape) == 1) {
        ProcessSingleBatch(params, bs, 0, true);
    } else {
        ProcessWithBatch(params, bs);
    }

    if ASCEND_IS_AIV {
        ReduceAddProcess();
    }
}


QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::Init(const Params &params)
{
    if ASCEND_IS_AIV {
        epilogueOp_.Init(params.epilogueParams);
        return;
    }
    if (params.qbmmParams.isBias == 1) {
        if (params.qbmmParams.biasThreeDim == 1) {
            isBiasThreeDim_ = true;
        }
        isBias_ = true;
    }
    if (params.qbmmParams.batchA1 == params.qbmmParams.batchB1 &&
        params.qbmmParams.batchA2 == params.qbmmParams.batchB2 &&
        params.qbmmParams.batchA3 == params.qbmmParams.batchB3 &&
        params.qbmmParams.batchA4 == params.qbmmParams.batchB4) {
        isSameBatch_ = true;
    }
    ResetGmAddr(params);
}


QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::ResetGmAddr(const Params &params)
{
    if ASCEND_IS_AIV {
        return;
    }

    aGmAddr_ = reinterpret_cast<__gm__ AType *>(params.mmadParams.aGmAddr);
    bGmAddr_ = reinterpret_cast<__gm__ BType *>(params.mmadParams.bGmAddr);
    cGmAddr_ = reinterpret_cast<__gm__ CType *>(params.mmadParams.cGmAddr);
    scaleAGmAddr_ = reinterpret_cast<__gm__ fp8_e8m0_t *>(params.mmadParams.scaleAGmAddr);
    scaleBGmAddr_ = reinterpret_cast<__gm__ fp8_e8m0_t *>(params.mmadParams.scaleBGmAddr);
    if (isBias_) {
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType *>(params.mmadParams.biasGmAddr);
    }
}


QBMM_MX_KERNEL_CLASS_TEM_PARAMS
template <typename TensorScaleB>
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::SetScaleL2Cache(const ProblemShape &problemShape,
                                                                                 uint64_t baseN, uint64_t scaleKL1,
                                                                                 TensorScaleB &gmScaleB)
{
    if (AscendC::Te::Get<MNK_B>(problemShape) != 1) {
        return;
    }
    if constexpr (transB) {
        const int64_t scaleKRowBytes =
            Blaze::Gemm::CeilDiv(AscendC::Te::Get<MNK_K>(problemShape), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
            MXFP_MULTI_BASE_SIZE;
        const int64_t scaleKL1RowBytes = Blaze::Gemm::CeilDiv(scaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        const bool scaleAlignForL2Stream = (scaleKRowBytes & 0x7f) == 0 && (scaleKL1RowBytes & 0x7f) == 0;
        gmScaleB.SetL2CacheHint(scaleAlignForL2Stream ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE :
                                                        AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
    } else {
        const int64_t scaleNStrideBytes = AscendC::Te::Get<MNK_N>(problemShape) * MXFP_MULTI_BASE_SIZE;
        const int64_t scaleBaseNStrideBytes = baseN * MXFP_MULTI_BASE_SIZE;
        const bool scaleAlignForL2Stream = (scaleNStrideBytes & 0x7f) == 0 && (scaleBaseNStrideBytes & 0x7f) == 0;
        gmScaleB.SetL2CacheHint(scaleAlignForL2Stream ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE :
                                                        AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
template <typename TensorB, typename TensorScaleB, typename TensorC>
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::SetL2Cache(const ProblemShape &problemShape,
                                                                            uint64_t curBaseM, uint64_t baseN,
                                                                            uint64_t scaleKL1, TensorB &gmB,
                                                                            TensorScaleB &gmScaleB, TensorC &gmC)
{
    if constexpr (isAtomicAdd) {
        gmC.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
    }

    const bool fullMTile = curBaseM >= AscendC::Te::Get<MNK_M>(problemShape);
    if (!(isSameBatch_ && fullMTile)) {
        return;
    }

    SetScaleL2Cache(problemShape, baseN, scaleKL1, gmScaleB);

    if constexpr (weightNz) {
        gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
    } else {
        if constexpr (transB) {
            bool bAlignForL2Stream = (AscendC::Te::Get<MNK_K>(problemShape) & kCacheLineAlignMask) == 0;
            gmB.SetL2CacheHint(bAlignForL2Stream ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE :
                                                   AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
        } else {
            bool bAlignForL2Stream = (AscendC::Te::Get<MNK_N>(problemShape) & kCacheLineAlignMask) == 0 &&
                                     (baseN & kCacheLineAlignMask) == 0;
            gmB.SetL2CacheHint(bAlignForL2Stream ? AscendC::Te::CacheMode::CACHE_MODE_DISABLE :
                                                   AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
        }
    }
}


QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::ProcessWithBatch(const Params &params,
                                                                                  BlockScheduler &bs)
{
    uint64_t batchC3C4 = static_cast<uint64_t>(params.qbmmParams.batchC3) * params.qbmmParams.batchC4;
    uint64_t batchC2C3C4 = params.qbmmParams.batchC2 * batchC3C4;
    uint64_t batchB3B4 = static_cast<uint64_t>(params.qbmmParams.batchB3) * params.qbmmParams.batchB4;
    uint64_t batchB2B3B4 = params.qbmmParams.batchB2 * batchB3B4;
    uint64_t batchA3A4 = static_cast<uint64_t>(params.qbmmParams.batchA3) * params.qbmmParams.batchA4;
    uint64_t batchA2A3A4 = params.qbmmParams.batchA2 * batchA3A4;
    uint32_t multiA1C1 = params.qbmmParams.batchA1 / params.qbmmParams.batchC1;
    uint32_t multiA2C2 = params.qbmmParams.batchA2 / params.qbmmParams.batchC2;
    uint32_t multiA3C3 = params.qbmmParams.batchA3 / params.qbmmParams.batchC3;
    uint32_t multiA4C4 = params.qbmmParams.batchA4 / params.qbmmParams.batchC4;
    uint32_t multiB1C1 = params.qbmmParams.batchB1 / params.qbmmParams.batchC1;
    uint32_t multiB2C2 = params.qbmmParams.batchB2 / params.qbmmParams.batchC2;
    uint32_t multiB3C3 = params.qbmmParams.batchB3 / params.qbmmParams.batchC3;
    uint32_t multiB4C4 = params.qbmmParams.batchB4 / params.qbmmParams.batchC4;

    uint64_t batchC1Offset = 0;
    uint64_t batchA1Offset = 0;
    uint64_t batchB1Offset = 0;
    uint64_t curBatchC = 1UL;
    uint64_t singleBatchTileCnt = bs.GetTotalCnt();
    uint64_t tailRoundStart =
        (singleBatchTileCnt * AscendC::Te::Get<MNK_B>(params.problemShape) / AscendC::GetBlockNum()) *
        AscendC::GetBlockNum();
    for (uint64_t b1Index = 0; b1Index < params.qbmmParams.batchC1; ++b1Index) {
        uint64_t batchC2Offset = batchC1Offset;
        uint64_t batchA2Offset = batchA1Offset;
        uint64_t batchB2Offset = batchB1Offset;
        for (uint64_t b2Index = 0; b2Index < params.qbmmParams.batchC2; ++b2Index) {
            uint64_t batchC3Offset = batchC2Offset;
            uint64_t batchA3Offset = batchA2Offset;
            uint64_t batchB3Offset = batchB2Offset;
            for (uint64_t b3Index = 0; b3Index < params.qbmmParams.batchC3; ++b3Index) {
                batchCOffset_ = batchC3Offset;
                batchAOffset_ = batchA3Offset;
                batchBOffset_ = batchB3Offset;
                for (uint64_t b4Index = 0; b4Index < params.qbmmParams.batchC4; ++b4Index) {
                    bool isTailRound = curBatchC * singleBatchTileCnt > tailRoundStart;
                    AddBatchOffset(params);
                    ProcessSingleBatch(params, bs, (AscendC::Te::Get<MNK_B>(params.problemShape) - curBatchC),
                                       isTailRound);
                    curBatchC++;
                    batchCOffset_ += 1;
                    batchAOffset_ += multiA4C4;
                    batchBOffset_ += multiB4C4;
                }
                batchC3Offset += params.qbmmParams.batchC4;
                batchA3Offset += params.qbmmParams.batchA4 * static_cast<uint64_t>(multiA3C3);
                batchB3Offset += params.qbmmParams.batchB4 * static_cast<uint64_t>(multiB3C3);
            }
            batchC2Offset += batchC3C4;
            batchA2Offset += batchA3A4 * multiA2C2;
            batchB2Offset += batchB3B4 * multiB2C2;
        }
        batchC1Offset += batchC2C3C4;
        batchA1Offset += batchA2A3A4 * multiA1C1;
        batchB1Offset += batchB2B3B4 * multiB1C1;
    }
}

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::AddBatchOffset(const Params &params)
{
    ResetGmAddr(params);
    constexpr uint64_t sizeShift = IsFp4<AType>() ? 1 : 0;
    aGmAddr_ +=
        (batchAOffset_ * AscendC::Te::Get<MNK_M>(params.problemShape) * AscendC::Te::Get<MNK_K>(params.problemShape)) >>
        sizeShift;
    if constexpr (weightNz) {
        if constexpr (transB) {
            bGmAddr_ +=
                (batchBOffset_ * Blaze::Gemm::CeilDiv(AscendC::Te::Get<MNK_K>(params.problemShape), C0_SIZE) *
                 Blaze::Gemm::CeilDiv(AscendC::Te::Get<MNK_N>(params.problemShape), static_cast<int64_t>(BLOCK_CUBE)) *
                 BLOCK_CUBE * C0_SIZE) >>
                sizeShift;
        } else {
            bGmAddr_ +=
                (batchBOffset_ * Blaze::Gemm::CeilDiv(AscendC::Te::Get<MNK_N>(params.problemShape), C0_SIZE) *
                 Blaze::Gemm::CeilDiv(AscendC::Te::Get<MNK_K>(params.problemShape), static_cast<int64_t>(BLOCK_CUBE)) *
                 BLOCK_CUBE * C0_SIZE) >>
                sizeShift;
        }
    } else {
        bGmAddr_ += (batchBOffset_ * AscendC::Te::Get<MNK_N>(params.problemShape) *
                     AscendC::Te::Get<MNK_K>(params.problemShape)) >>
                    sizeShift;
    }
    cGmAddr_ +=
        batchCOffset_ * AscendC::Te::Get<MNK_M>(params.problemShape) * AscendC::Te::Get<MNK_N>(params.problemShape);
    if (isBiasThreeDim_) {
        biasGmAddr_ += batchCOffset_ * AscendC::Te::Get<MNK_N>(params.problemShape);
    }
}


QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::ProcessSingleBatch(const Params &params,
                                                                                    BlockScheduler &bs,
                                                                                    uint64_t restBatch,
                                                                                    bool isTailRound)
{
    auto scaleKLen =
        Blaze::Gemm::CeilDiv(AscendC::Te::Get<MNK_K>(params.problemShape), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;
    auto layoutA =
        MakeLayoutA{}(AscendC::Te::Get<MNK_M>(params.problemShape), AscendC::Te::Get<MNK_K>(params.problemShape));
    auto layoutScaleA = MakeLayoutScaleA{}(AscendC::Te::Get<MNK_M>(params.problemShape), scaleKLen);
    auto layoutB =
        MakeLayoutB{}(AscendC::Te::Get<MNK_K>(params.problemShape), AscendC::Te::Get<MNK_N>(params.problemShape));
    auto layoutScaleB = MakeLayoutScaleB{}(scaleKLen, AscendC::Te::Get<MNK_N>(params.problemShape));
    auto layoutBias =
        AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1L, AscendC::Te::Get<MNK_N>(params.problemShape));
    auto layoutC =
        MakeLayoutC{}(AscendC::Te::Get<MNK_M>(params.problemShape), AscendC::Te::Get<MNK_N>(params.problemShape));

    auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aGmAddr_), layoutA);
    auto gmScaleA =
        AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleAGmAddr_), layoutScaleA);
    auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bGmAddr_), layoutB);
    auto gmScaleB =
        AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleBGmAddr_), layoutScaleB);
    auto gmBias = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(biasGmAddr_), layoutBias);
    auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cGmAddr_), layoutC);

    BlockCoord blockIdx;
    auto &mTailTile = params.schParams.mTailTile;
    auto &nTailTile = params.schParams.nTailTile;
    if (needUpdateTail_ ||
        (isTailRound && ((bs.GetEndBlockIdx() + 1) + (restBatch * bs.GetTotalCnt())) * mTailTile * nTailTile <=
                            AscendC::GetBlockNum())) {
        needUpdateTail_ = true;
        bs.UpdateTailTile(mTailTile, nTailTile);
    }
    SetL2Cache(params.problemShape, params.qbmmParams.baseM, params.qbmmParams.baseN, params.l1Params.scaleKL1, gmB,
               gmScaleB, gmC);

    int64_t mPos = 0L;
    int64_t nPos = 0L;
    constexpr int64_t kPos = 0L;
    while (bs.GetTileIdx(blockIdx)) {
        BlockShape singleShape =
            bs.template GetBlockShape<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE, weightNz>(blockIdx);
        if (AscendC::Te::Get<IDX_M_TILEIDX>(singleShape) <= 0 || AscendC::Te::Get<IDX_N_TILEIDX>(singleShape) <= 0) {
            return;
        }
        bs.GetTileCoord(blockIdx, mPos, nPos);
        if ASCEND_IS_AIC {
            auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(mPos, kPos),
                                      AscendC::Te::MakeShape(AscendC::Te::Get<IDX_M_TILEIDX>(singleShape),
                                                             AscendC::Te::Get<MNK_K>(params.problemShape)));
            auto gmBlockScaleA =
                gmScaleA.Slice(AscendC::Te::MakeCoord(mPos, kPos),
                               AscendC::Te::MakeShape(AscendC::Te::Get<IDX_M_TILEIDX>(singleShape), scaleKLen));
            auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(kPos, nPos),
                                      AscendC::Te::MakeShape(AscendC::Te::Get<MNK_K>(params.problemShape),
                                                             AscendC::Te::Get<IDX_N_TILEIDX>(singleShape)));
            auto gmBlockScaleB =
                gmScaleB.Slice(AscendC::Te::MakeCoord(kPos, nPos),
                               AscendC::Te::MakeShape(scaleKLen, AscendC::Te::Get<IDX_N_TILEIDX>(singleShape)));
            auto gmBlockBias = gmBias.Slice(AscendC::Te::MakeCoord(0L, nPos),
                                            AscendC::Te::MakeShape(1L, AscendC::Te::Get<IDX_N_TILEIDX>(singleShape)));
            auto gmBlockC = gmC.Slice(AscendC::Te::MakeCoord(mPos, nPos),
                                      AscendC::Te::MakeShape(AscendC::Te::Get<IDX_M_TILEIDX>(singleShape),
                                                             AscendC::Te::Get<IDX_N_TILEIDX>(singleShape)));
            mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleShape);
            AscendC::CrossCoreSetFlag<CROSS_CORE_MODE_ID_TWO, PIPE_FIX>(CROSS_CORE_FLAG_ID_NINE);
        }
        if ASCEND_IS_AIV {
            AscendC::CrossCoreWaitFlag(CROSS_CORE_FLAG_ID_NINE);
            epilogueOp_.Run(mPos, nPos, singleShape);
        }
    }
    bs.UpdateNextBatchBlockRoundParams();
}


QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::ReduceAddProcess()
{
    AscendC::SyncAll<true>();
    WriteStatusToWin();
    ReadStatus();
    AscendC::SyncAll<true>();

    ReadRemoteDataAdd();
}


QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::SplitToCore(const uint32_t curSendCnt,
                                                                             const uint32_t curUseCoreNum,
                                                                             const uint32_t coreId, uint32_t &startId,
                                                                             uint32_t &endId, uint32_t &sendNum)
{
    sendNum = curSendCnt / curUseCoreNum;
    uint32_t remainderNum = curSendCnt % curUseCoreNum;
    startId = sendNum * coreId;
    if (coreId < remainderNum) {
        sendNum += 1;
        startId += coreId;
    } else {
        startId += remainderNum;
    }
    endId = startId + sendNum;
}


QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::ReadRemoteDataAdd()
{
    uint32_t startRowId = 0;
    uint32_t endRowId = 0;
    uint32_t rowNum = 0;
    uint32_t tokenIndex = 0;
    uint64_t tpRankM = m_ / tpWorldSize_;
    SplitToCore(tpRankM, AscendC::GetBlockNum() * 2, coreVid_, startRowId, endRowId, rowNum);

    auto layoutTensorC = AscendC::Te::FrameLayoutFormat<LayoutC, CType>{}(m_, n_);
    auto layoutTensorY = AscendC::Te::FrameLayoutFormat<LayoutC, CType>{}(tpRankM, n_);
    auto gmC =
        AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
                                    (__gm__ CType *)Mc2Kernel::GetBaseWindAddrByRankId(winContext_, rankId_, rankId_)),
                                layoutTensorC);
    auto gmY = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>((__gm__ bfloat16_t *)yGM_),
                                       layoutTensorY);

    uint64_t rowPaddedBytes =
        Blaze::Gemm::CeilDiv(static_cast<uint64_t>(n_ * sizeof(CType)), UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
    uint64_t paddingN = rowPaddedBytes / sizeof(CType);

    uint64_t localbf16Offset = paddingN * sizeof(float);
    uint64_t localfp32Offset = localbf16Offset + paddingN * sizeof(CType);
    uint64_t sumbf16Offset = localfp32Offset + paddingN * sizeof(float);

    auto sumfp32UbTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(0),
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<float>>{}(
            1, paddingN));
    for (uint32_t tileIdx = 0; tileIdx < rowNum; ++tileIdx) {
        AscendC::Te::Transform<AscendC::Te::Inst::MulScalar>(sumfp32UbTensor, sumfp32UbTensor, 0.0f);
        tokenIndex = startRowId + tileIdx;
        for (int tpIndex = 0; tpIndex < tpWorldSize_; ++tpIndex) {
            uint32_t mPos = tokenIndex + tpIndex * (tpRankM);
            uint32_t nPos = 0;

            auto layoutPaddingUB = MakeLayoutUB{}(1, paddingN);
            auto ubTensor = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, CType>(localbf16Offset), layoutPaddingUB);
            auto gmTensor = gmC.Slice(AscendC::Te::MakeCoord(mPos, nPos), AscendC::Te::MakeShape(1, n_));
            AscendC::Te::Copy(copyGM2UB_, ubTensor, gmTensor);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(FLAG_ID_NUM_ONE);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(FLAG_ID_NUM_ONE);

            auto fp32UbTensor = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(localfp32Offset),
                AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<float>>{}(
                    1, paddingN));
            AscendC::Te::Transform<AscendC::Te::Inst::Cast>(fp32UbTensor, ubTensor);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Te::Transform<AscendC::Te::Inst::Add>(sumfp32UbTensor, sumfp32UbTensor, fp32UbTensor);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);
        }
        auto sumbf16UbTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, CType>(sumbf16Offset), MakeLayoutUB{}(1, paddingN));
        auto gmOutTensor = gmY.Slice(AscendC::Te::MakeCoord(tokenIndex, 0), AscendC::Te::MakeShape(1, n_));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Te::Transform<AscendC::Te::Inst::Cast, CastTraitF322Bf16>(sumbf16UbTensor, sumfp32UbTensor);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);
        AscendC::Te::Copy(copyUB2GM_, gmOutTensor, sumbf16UbTensor);
    }
}


QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::WriteStatusToWin()
{
    if (coreVid_ >= tpWorldSize_) {
        return;
    }
    uint32_t curOffset = rankId_ * FLOAT_UB_ALIGN_NUM * sizeof(float);
    auto layoutStatus = AscendC::Te::FrameLayoutFormat<LayoutStatus, float>{}(1, FLOAT_UB_ALIGN_NUM);
    auto ubStatusTensor =
        AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(0), layoutStatus);
    auto gmStatusTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
            (__gm__ float *)(Mc2Kernel::GetBaseWindStateAddrByRankId(winContext_, coreVid_, rankId_) + curOffset)),
        layoutStatus);

    ubStatusTensor[AscendC::Te::MakeCoord(0, 0)] = (float)1;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(FLAG_ID_NUM_ONE);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(FLAG_ID_NUM_ONE);
    AscendC::Te::Copy(copyUB2GM_, gmStatusTensor, ubStatusTensor);
}


QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MX_KERNEL_TEM_PARAMS>::ReadStatus()
{
    if (coreVid_ >= tpWorldSize_) {
        return;
    }

    uint32_t ubStatusOffset = UB_ALIGN_BYTES * 1;
    uint32_t ubResetOffset = ubStatusOffset + UB_ALIGN_BYTES;
    uint32_t gmStatusoffset = coreVid_ * FLOAT_UB_ALIGN_NUM * sizeof(float);
    auto layoutStatus = AscendC::Te::FrameLayoutFormat<LayoutStatus, float>{}(1, FLOAT_UB_ALIGN_NUM);
    auto ubStatusTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(ubStatusOffset), layoutStatus);
    auto ubResetTensor =
        AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(ubResetOffset), layoutStatus);
    auto gmStatusTensor = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
            (__gm__ float *)(Mc2Kernel::GetBaseWindStateAddrByRankId(winContext_, rankId_, rankId_) + gmStatusoffset)),
        layoutStatus);

    float flag = STATUS_FLAG_VAL;
    float minTarget = 1.0f - STATUS_FLAG_THREDSHOLD;
    float maxTarget = 1.0f + STATUS_FLAG_THREDSHOLD;
    uint32_t retryCnt = 0;
    while ((flag < minTarget) || (flag > maxTarget)) {
        if (retryCnt > STATUS_POLL_MAX_RETRY) {
            break;
        }
        retryCnt++;
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(FLAG_ID_NUM_ONE);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(FLAG_ID_NUM_ONE);
        AscendC::Te::Copy(copyGM2UB_, ubStatusTensor, gmStatusTensor);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(FLAG_ID_NUM_ONE);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(FLAG_ID_NUM_ONE);
        flag = ubStatusTensor[AscendC::Te::MakeCoord(0, 0)];
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(FLAG_ID_NUM_ONE);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(FLAG_ID_NUM_ONE);
    ubResetTensor[AscendC::Te::MakeCoord(0, 0)] = 0.0f;
    AscendC::Te::Copy(copyUB2GM_, gmStatusTensor, ubResetTensor);
}

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze

#undef QBMM_MX_KERNEL_CLASS_TEM_PARAMS
#undef QBMM_MX_KERNEL_TEM_PARAMS
