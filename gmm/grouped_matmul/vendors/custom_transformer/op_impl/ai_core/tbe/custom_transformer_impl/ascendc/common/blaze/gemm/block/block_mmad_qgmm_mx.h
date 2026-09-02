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
 * \file block_mmad_qgmm_mx.h
 * \brief QGMM MX tensor_api block mmad implementation with independent kAL1 / kBL1 scheduling.
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "tensor_api/tensor.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/tile/copy_gm_to_l1.h"
#include "blaze/gemm/tile/tile_trait.h"
#include "blaze/gemm/tile/pad_mx_kl1.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <uint64_t FullLoadMode_, bool AtomicAdd_, class ScheduleType_, class AType_, class LayoutA_, class BType_,
          class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<GroupedMatmulWithScaleMx<FullLoadMode_, AtomicAdd_, ScheduleType_>, AType_, LayoutA_, BType_, LayoutB_,
                CType_, LayoutC_, BiasType_, LayoutBias_> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using BiasType = BiasType_;
    using LayoutBias = LayoutBias_;
    using DispatchPolicy = GroupedMatmulWithScaleMx<FullLoadMode_, AtomicAdd_, ScheduleType_>;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool IS_FP4_TYPE = IsFp4<AType>();
    static_assert(DOUBLE_BUFFER_COUNT == 2, "QGMM MX block mmad only supports double buffer.");
    static constexpr uint64_t SPLIT_M_ALIGN = 2;
    static constexpr uint64_t CONCAT_N_FACTOR = 2UL;
    static constexpr uint16_t CONCAT_MATRIX_NUM = 2U;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kAL1;     // A matrix L1 K-axis split size.
        uint64_t kBL1;     // B matrix L1 K-axis split size.
        uint64_t scaleKL1; // Shared ScaleA/ScaleB L1 K-axis split size.
    };

    struct MmadParams {
        BlockShape tileShapeL0;
        L1Params l1Params;
        bool isBias;
        bool enableL0cPingPong;
        uint8_t l1BufferStage{DOUBLE_BUFFER_COUNT};
    };

    template <typename TensorScaleAL1, typename TensorScaleBL1>
    struct ScalePair {
        TensorScaleAL1 scaleA;
        TensorScaleBL1 scaleB;
    };

    __aicore__ inline BlockMmad()
    {
#pragma unroll
        for (uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_1);
        AscendC::SetMMLayoutTransform(true);
    }

    __aicore__ inline ~BlockMmad()
    {
#pragma unroll
        for (uint8_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_1);
        AscendC::SetMMLayoutTransform(false);
    }

    __aicore__ inline void Init(const ProblemShape& problemShape, const MmadParams& params)
    {
        m_ = AscendC::Te::Get<MNK_M>(problemShape);
        n_ = AscendC::Te::Get<MNK_N>(problemShape);
        k_ = AscendC::Te::Get<MNK_K>(problemShape);
        kAL1_ = params.l1Params.kAL1;
        kBL1_ = params.l1Params.kBL1;
        baseM_ = AscendC::Te::Get<MNK_M>(params.tileShapeL0);
        baseN_ = AscendC::Te::Get<MNK_N>(params.tileShapeL0);
        baseK_ = AscendC::Te::Get<MNK_K>(params.tileShapeL0);
        isBias_ = params.isBias;
        enableL0cPingPong_ = params.enableL0cPingPong;
        l1BufferStage_ = params.l1BufferStage == TRIPLE_BUFFER_COUNT ? TRIPLE_BUFFER_COUNT : DOUBLE_BUFFER_COUNT;
        orderAL1BL1_ = kAL1_ >= kBL1_;
        scaleKL1_ = Max(params.l1Params.scaleKL1, 1UL);

        constexpr uint64_t sizeShift = IS_FP4_TYPE ? 1UL : 0UL;
        scaleKL1Span_ = CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        aL1OneBuffer_ = (baseM_ * CeilAlign(kAL1_, MXFP_DIVISOR_SIZE)) >> sizeShift;
        bL1OneBuffer_ = (baseN_ * CeilAlign(kBL1_, MXFP_DIVISOR_SIZE)) >> sizeShift;
        concatBL1OneBuffer_ = (baseN_ * CONCAT_N_FACTOR * CeilAlign(kBL1_, MXFP_DIVISOR_SIZE)) >> sizeShift;
        scaleAL1OneBuffer_ = baseM_ * scaleKL1Span_;
        scaleBL1OneBuffer_ = baseN_ * scaleKL1Span_;
        if (isBias_) {
            biasL1OneBuffer_ = baseN_ * sizeof(BiasType);
        }

        InitL1BufferOffsets();
    }

    __aicore__ inline void UpdateParamsForNextProblem(const ProblemShape& problemShape)
    {
        m_ = AscendC::Te::Get<MNK_M>(problemShape);
        n_ = AscendC::Te::Get<MNK_N>(problemShape);
        k_ = AscendC::Te::Get<MNK_K>(problemShape);
    }

private:
    template <typename TensorAL1, typename TensorScaleAL1, typename CopyL12L0A, typename CopyL12L0ScaleA>
    __aicore__ inline auto CopyNoBiasAToL0(TensorAL1 tensorAL1, TensorScaleAL1 tensorBlockScaleAL1,
                                           CopyL12L0A copyL12L0A, CopyL12L0ScaleA copyL12L0ScaleA, uint64_t curM,
                                           uint64_t curKL0, uint64_t curKL0ScaleSpan, uint64_t scaleOffsetA,
                                           uint64_t kaL1Offset, uint64_t kL0Offset, uint64_t l0Offset)
    {
        auto tensorAL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(l0Offset),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>(curM, curKL0));
        auto tensorBlockAL1 = tensorAL1.Slice(AscendC::Te::MakeCoord(0, kaL1Offset + kL0Offset),
                                              AscendC::Te::MakeShape(curM, curKL0));
        AscendC::Te::Copy(copyL12L0A, tensorAL0, tensorBlockAL1);

        auto tensorScaleAL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0ScaleA, fp8_e8m0_t>(l0Offset >> 4),
            AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(curM, curKL0ScaleSpan));
        AscendC::Te::Copy(copyL12L0ScaleA, tensorScaleAL0,
                          tensorBlockScaleAL1.Slice(AscendC::Te::MakeCoord(0, scaleOffsetA),
                                                    AscendC::Te::MakeShape(curM, curKL0ScaleSpan)));
        return tensorAL0;
    }

    template <typename TensorBL1, typename TensorScaleBL1, typename CopyL12L0B, typename CopyL12L0ScaleB>
    __aicore__ inline auto CopyNoBiasBToL0(TensorBL1 tensorBL1, TensorScaleBL1 tensorBlockScaleBL1,
                                           CopyL12L0B copyL12L0B, CopyL12L0ScaleB copyL12L0ScaleB, uint64_t curN,
                                           uint64_t curKL0, uint64_t curKL0ScaleSpan, uint64_t scaleOffsetB,
                                           uint64_t kbL1Offset, uint64_t kL0Offset, uint64_t l0Offset)
    {
        auto tensorBL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(l0Offset),
            AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>(curKL0, curN));
        auto tensorBlockBL1 = tensorBL1.Slice(AscendC::Te::MakeCoord(kbL1Offset + kL0Offset, 0),
                                              AscendC::Te::MakeShape(curKL0, curN));
        AscendC::Te::Copy(copyL12L0B, tensorBL0, tensorBlockBL1);

        auto tensorScaleBL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0ScaleB, fp8_e8m0_t>(l0Offset >> 4),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(curKL0ScaleSpan, curN));
        AscendC::Te::Copy(copyL12L0ScaleB, tensorScaleBL0,
                          tensorBlockScaleBL1.Slice(AscendC::Te::MakeCoord(scaleOffsetB, 0),
                                                    AscendC::Te::MakeShape(curKL0ScaleSpan, curN)));
        return tensorBL0;
    }

    template <typename TensorL0C, typename TensorAL0, typename TensorBL0, typename MmadAtom>
    __aicore__ inline void RunNoBiasMmad(TensorL0C tensorL0C, TensorAL0 tensorAL0, TensorBL0 tensorBL0,
                                         MmadAtom mmADAtom, uint64_t curM, uint64_t curN, uint64_t curKL0,
                                         uint64_t absKOffset)
    {
        AscendC::Te::MmadParams mmADParams;
        mmADParams.m = static_cast<uint16_t>(curM);
        mmADParams.k = static_cast<uint16_t>(CeilAlign(curKL0, MXFP_DIVISOR_SIZE));
        mmADParams.n = static_cast<uint16_t>(curN);
        mmADParams.unitFlag = (absKOffset + curKL0 >= k_) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
        mmADParams.cmatrixInitVal = (absKOffset == 0);
        AscendC::Te::Mmad(mmADAtom.with(mmADParams), tensorL0C, tensorAL0, tensorBL0);
    }

    template <typename TensorL0C, typename TensorAL1, typename TensorBL1, typename TensorScaleAL1,
              typename TensorScaleBL1>
    __aicore__ inline void IterateNoBias(TensorL0C tensorL0C, TensorAL1 tensorAL1, TensorBL1 tensorBL1,
                                         TensorScaleAL1 tensorScaleAL1, TensorScaleBL1 tensorScaleBL1, uint64_t curM,
                                         uint64_t curN, uint64_t curGmAKL1, uint64_t curGmBKL1, uint64_t absKStartA,
                                         uint64_t absKStartB, uint64_t kaL1Offset, uint64_t kbL1Offset)
    {
        uint64_t l1Ka = NEED_A_SET_L1_K_ZERO ? CeilAlign(curGmAKL1, MXFP_DIVISOR_SIZE) : curGmAKL1;
        uint64_t l1Kb = NEED_B_SET_L1_K_ZERO ? CeilAlign(curGmBKL1, MXFP_DIVISOR_SIZE) : curGmBKL1;
        uint64_t minPadKL1 = Min(l1Ka, l1Kb);
        uint64_t minGmKL1 = Min(curGmAKL1, curGmBKL1);
        uint64_t scaleBaseA = GetScaleOffset(absKStartA % scaleKL1_);
        uint64_t scaleBaseB = GetScaleOffset(absKStartB % scaleKL1_);
        auto tensorBlockScaleAL1 = tensorScaleAL1.Slice(AscendC::Te::MakeCoord(0, scaleBaseA),
                                                        AscendC::Te::MakeShape(curM, GetScaleSpan(curGmAKL1)));
        auto tensorBlockScaleBL1 = tensorScaleBL1.Slice(AscendC::Te::MakeCoord(scaleBaseB, 0),
                                                        AscendC::Te::MakeShape(GetScaleSpan(curGmBKL1), curN));
        auto copyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
        auto copyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
        auto copyL12L0ScaleA = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0ScaleA{});
        auto copyL12L0ScaleB = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0ScaleB{});
        constexpr auto
            mmadAtom = AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, Tile::MmadTraitMX>>{};
        const uint64_t halfL0Size = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;

        for (uint64_t kL0Offset = 0; kL0Offset < minGmKL1; kL0Offset += baseK_) {
            uint64_t curKL0 = (kL0Offset + baseK_ > minPadKL1) ? (minPadKL1 - kL0Offset) : baseK_;
            uint64_t absKOffset = absKStartA + kaL1Offset + kL0Offset;
            uint64_t scaleOffsetA = GetScaleOffset(kaL1Offset + kL0Offset);
            uint64_t scaleOffsetB = GetScaleOffset(kbL1Offset + kL0Offset);
            uint64_t curKL0ScaleSpan = GetScaleSpan(curKL0);
            uint64_t l0Offset = halfL0Size * (l0PingPong_ & 1UL);
            uint16_t mte1WaitMFlag = static_cast<uint16_t>((l0PingPong_ & 1UL) + M_MTE1_FLAG_0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);

            auto tensorAL0 = CopyNoBiasAToL0(tensorAL1, tensorBlockScaleAL1, copyL12L0A, copyL12L0ScaleA, curM, curKL0,
                                             curKL0ScaleSpan, scaleOffsetA, kaL1Offset, kL0Offset, l0Offset);
            auto tensorBL0 = CopyNoBiasBToL0(tensorBL1, tensorBlockScaleBL1, copyL12L0B, copyL12L0ScaleB, curN, curKL0,
                                             curKL0ScaleSpan, scaleOffsetB, kbL1Offset, kL0Offset, l0Offset);

            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 1UL);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 1UL);

            RunNoBiasMmad(tensorL0C, tensorAL0, tensorBL0, mmadAtom, curM, curN, curKL0, absKOffset);

            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);
            l0PingPong_++;
        }
    }

public:
    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorBias,
              typename TensorC>
    __aicore__ inline void operator()(const TensorA& gmA, const TensorB& gmB, const TensorScaleA& gmScaleA,
                                      const TensorScaleB& gmScaleB, const TensorBias& gmBias, const TensorC& gmC,
                                      const BlockShape& blockShape)
    {
        RunImpl(gmA, gmB, gmScaleA, gmScaleB, gmBias, gmC, blockShape);
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorBias,
              typename TensorC>
    __aicore__ inline void operator()(const TensorA& gmA, const TensorB& gmBFirst, const TensorB& gmBSecond,
                                      const TensorScaleA& gmScaleA, const TensorScaleB& gmScaleBFirst,
                                      const TensorScaleB& gmScaleBSecond, const TensorBias& gmBias,
                                      const TensorC& ubCFirst, const TensorC& ubCSecond, const BlockShape& blockShape)
    {
        static_assert(AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<TensorC>, AscendC::Te::Location::UB>,
                      "Dual MMAD output tensors must be located in UB.");
        RunImpl(gmA, gmBFirst, gmScaleA, gmScaleBFirst, gmBias, ubCFirst, blockShape);
        RunImpl(gmA, gmBSecond, gmScaleA, gmScaleBSecond, gmBias, ubCSecond, blockShape);
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorUbC>
    __aicore__ inline void operator()(TensorA gmA, TensorB gmBLeft, TensorScaleA gmScaleA, TensorScaleB gmScaleBLeft,
                                      TensorUbC& ubC, const BlockShape& singleShape)
    {
        RunConcatN(gmA, gmBLeft, gmScaleA, gmScaleBLeft, ubC, singleShape);
    }

private:
    static constexpr bool NEED_A_SET_L1_K_ZERO = AscendC::Std::is_one_of_v<AType, fp8_e5m2_t, fp8_e4m3fn_t>;
    static constexpr bool NEED_B_SET_L1_K_ZERO = AscendC::Std::is_one_of_v<BType, fp8_e5m2_t, fp8_e4m3fn_t> ||
                                                 (AscendC::Std::is_one_of_v<BType, fp4x2_e2m1_t, fp4x2_e1m2_t> &&
                                                  !TRANS_B);
    static constexpr uint64_t C0_SIZE = IS_FP4_TYPE ? C0_SIZE_B4 : C0_SIZE_B8;
    static constexpr uint8_t L1_OPERAND_NUM = 2;
    static constexpr uint8_t MTE1_MTE2_EVENT_ID_NUM = TRIPLE_BUFFER_COUNT * L1_OPERAND_NUM + SCALE_BUFFER_NUM;

    template <typename ElementType>
    __aicore__ inline uint64_t PackFp4Size(uint64_t value) const
    {
        if constexpr (IsFp4<ElementType>()) {
            return value >> 1;
        }
        return value;
    }

    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        TRANS_A, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>>;
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        TRANS_B, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>>;

    __aicore__ inline void InitL1BufferOffsets()
    {
        constexpr uint64_t halfL1Offset = AscendC::TOTAL_L1_SIZE >> 1;
        l1BufferAOffset_[0] = 0UL;
        l1BufferBOffset_[0] = aL1OneBuffer_;
        l1BufferAOffset_[1] = halfL1Offset;
        l1BufferBOffset_[1] = halfL1Offset + aL1OneBuffer_;
#pragma unroll
        for (int32_t bufferId = 0; bufferId < static_cast<int32_t>(SCALE_BUFFER_NUM); ++bufferId) {
            l1BufferScaleAOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_;
            l1BufferScaleBOffset_[bufferId] = l1BufferScaleAOffset_[bufferId] + scaleAL1OneBuffer_;
            l1BufferBiasOffset_[bufferId] = l1BufferScaleBOffset_[bufferId] + scaleBL1OneBuffer_;
        }
        if (l1BufferStage_ == TRIPLE_BUFFER_COUNT) {
            l1BufferAOffset_[2] = l1BufferBiasOffset_[0] + biasL1OneBuffer_;
            l1BufferBOffset_[2] = l1BufferBiasOffset_[1] + biasL1OneBuffer_;
        }
    }

    __aicore__ inline uint64_t GetL1BufId(uint64_t loopCnt) const
    {
        return l1BufferStage_ == TRIPLE_BUFFER_COUNT ? loopCnt % TRIPLE_BUFFER_COUNT : (loopCnt & 1UL);
    }

    __aicore__ inline uint64_t GetScaleSpan(uint64_t kSpan) const
    {
        return CeilDiv(kSpan, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    }

    __aicore__ inline uint64_t GetScaleOffset(uint64_t kOffset) const
    {
        return (kOffset >> MXFP_DIVISOR_SHIFT) << MXFP_MULTI_BASE_SHIFT;
    }

    __aicore__ inline bool NeedBias(uint64_t absKOffset) const { return isBias_ && absKOffset == 0; }

    template <typename ElementType, typename TensorL1, typename TensorGm>
    __aicore__ inline void CopyConcatBByStride(TensorL1 tensorL1, TensorGm gmLeftBlock, uint64_t singleN,
                                               uint64_t curGmBKL1, uint64_t l1K)
    {
        using CopyType = AscendC::Std::conditional_t<(sizeof(ElementType) == 1), int8_t, ElementType>;
        constexpr uint16_t loop2DstStride = 1;
        const uint64_t halfN = n_ >> 1;
        const uint64_t srcDValue = PackFp4Size<ElementType>(TRANS_B ? k_ : n_);
        const uint64_t srcMatrixStride = PackFp4Size<ElementType>(TRANS_B ? (halfN * k_) : halfN);
        const uint16_t nValue = static_cast<uint16_t>(TRANS_B ? singleN : curGmBKL1);
        const uint32_t dValue = static_cast<uint32_t>(PackFp4Size<ElementType>(TRANS_B ? curGmBKL1 : singleN));
        const uint16_t loop3DstStride = static_cast<uint16_t>(
            TRANS_B ? (CeilAlign(singleN, C0_SIZE) * CONCAT_N_FACTOR) : l1K);
        const uint16_t loop4DstStride = static_cast<uint16_t>(TRANS_B ? CeilAlign(singleN, C0_SIZE) :
                                                                        (l1K * CeilAlign(singleN, C0_SIZE) / C0_SIZE));
        const uint64_t loop1SrcStride = srcDValue * sizeof(ElementType);
        const uint64_t loop4SrcStride = srcMatrixStride * sizeof(ElementType);
        const uint8_t cacheMode = gmLeftBlock.Engine().GetCacheMode();

        Blaze::Gemm::Tile::CopySliceGM2L1::CopyGmToCbufMultiNd2nz(
            reinterpret_cast<__cbuf__ CopyType*>(tensorL1.Data().Get()),
            reinterpret_cast<__gm__ CopyType*>(gmLeftBlock.Data().Get()), CONCAT_MATRIX_NUM, loop2DstStride,
            loop3DstStride, loop4DstStride, loop1SrcStride, cacheMode, nValue, dValue, loop4SrcStride, false);
    }

    template <typename TensorScaleBL1, typename TensorScaleB>
    __aicore__ inline void CopyConcatScaleBByStride(TensorScaleBL1 tensorScaleBL1, TensorScaleB gmScaleBLeft,
                                                    uint64_t singleN, uint64_t scaleKBlockSpan, uint64_t scaleL1BufId)
    {
        (void)tensorScaleBL1;
        (void)scaleL1BufId;
        const uint64_t fullScaleKBlock = CeilDiv(k_, MXFP_DIVISOR_SIZE);
        const uint64_t scaleKL1Block = CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE);
        const uint64_t halfN = n_ >> 1;
        const uint64_t concatScaleBOffset = l1BufferBOffset_[scaleL1BufId] + concatBL1OneBuffer_ + scaleAL1OneBuffer_;

        if constexpr (TRANS_B) {
            Blaze::Gemm::Tile::CopySliceGM2L1::CopyGmToCbufMultiDn2nz(
                concatScaleBOffset, reinterpret_cast<__gm__ half*>(gmScaleBLeft.Data().Get()), CONCAT_MATRIX_NUM,
                static_cast<uint32_t>(singleN), static_cast<uint16_t>(scaleKBlockSpan), halfN * fullScaleKBlock,
                fullScaleKBlock, static_cast<uint16_t>(scaleKL1Block), 1,
                static_cast<uint32_t>(Align16(singleN) * scaleKL1Block));
        } else {
            constexpr uint16_t loop2DstStride = 1;
            constexpr uint64_t halfC0Element = AscendC::ONE_BLK_SIZE / sizeof(half);
            const uint16_t nValue = static_cast<uint16_t>(scaleKBlockSpan);
            const uint32_t dValue = static_cast<uint32_t>(singleN);
            const uint16_t loop3DstStride = static_cast<uint16_t>(scaleKL1Block);
            const uint16_t loop4DstStride = static_cast<uint16_t>(CeilAlign(singleN, BLOCK_CUBE) * scaleKL1Block /
                                                                  halfC0Element);
            const uint64_t loop1SrcStride = n_ * sizeof(half);
            const uint64_t loop4SrcStride = halfN * sizeof(half);
            const uint8_t cacheMode = gmScaleBLeft.Engine().GetCacheMode();
            Blaze::Gemm::Tile::CopySliceGM2L1::CopyGmToCbufMultiNd2nz(
                reinterpret_cast<__cbuf__ half*>(tensorScaleBL1.Data().Get()),
                reinterpret_cast<__gm__ half*>(gmScaleBLeft.Data().Get()), CONCAT_MATRIX_NUM, loop2DstStride,
                loop3DstStride, loop4DstStride, loop1SrcStride, cacheMode, nValue, dValue, loop4SrcStride, false);
        }
    }

    template <typename TensorScaleA, typename TensorScaleB>
    __aicore__ inline auto CopyScalesInL1(const TensorScaleA& gmScaleA, const TensorScaleB& gmScaleB, uint64_t curM,
                                          uint64_t curN, uint64_t kL1Offset, uint64_t scaleL1BufId)
    {
        auto layoutScaleAL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
            curM, scaleKL1Span_);
        auto tensorScaleAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(l1BufferScaleAOffset_[scaleL1BufId]),
            layoutScaleAL1);
        auto layoutScaleBL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
            scaleKL1Span_, curN);
        auto tensorScaleBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(l1BufferScaleBOffset_[scaleL1BufId]),
            layoutScaleBL1);

        if (kL1Offset % scaleKL1_ == 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufferStage_ * SCALE_BUFFER_NUM) +
                                                             scaleL1BufId);
            uint64_t curScaleKL1 = scaleKL1_;
            if (kL1Offset + curScaleKL1 > k_) {
                curScaleKL1 = k_ - kL1Offset;
            }
            auto copyScaleGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            auto gmBlockScaleA = gmScaleA.Slice(AscendC::Te::MakeCoord(0, GetScaleOffset(kL1Offset)),
                                                AscendC::Te::MakeShape(curM, GetScaleSpan(curScaleKL1)));
            AscendC::Te::Copy(copyScaleGM2L1, tensorScaleAL1, gmBlockScaleA);

            auto gmBlockScaleB = gmScaleB.Slice(AscendC::Te::MakeCoord(GetScaleOffset(kL1Offset), 0),
                                                AscendC::Te::MakeShape(GetScaleSpan(curScaleKL1), curN));
            AscendC::Te::Copy(copyScaleGM2L1, tensorScaleBL1, gmBlockScaleB);
        }
        return ScalePair<decltype(tensorScaleAL1), decltype(tensorScaleBL1)>{tensorScaleAL1, tensorScaleBL1};
    }

    template <typename TensorScaleA, typename TensorScaleB>
    __aicore__ inline auto CopyConcatScalesInL1(TensorScaleA gmScaleA, TensorScaleB gmScaleBLeft, uint64_t curM,
                                                uint64_t curN, uint64_t kL1Offset, uint64_t scaleL1BufId)
    {
        const uint64_t concatN = curN * CONCAT_N_FACTOR;
        const uint64_t concatScaleAOffset = l1BufferBOffset_[scaleL1BufId] + concatBL1OneBuffer_;
        const uint64_t concatScaleBOffset = concatScaleAOffset + scaleAL1OneBuffer_;
        auto layoutScaleAL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
            curM, scaleKL1Span_);
        auto tensorScaleAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(concatScaleAOffset), layoutScaleAL1);
        auto layoutScaleBL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
            scaleKL1Span_, concatN);
        auto tensorScaleBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(concatScaleBOffset), layoutScaleBL1);

        if (kL1Offset % scaleKL1_ == 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufferStage_ * SCALE_BUFFER_NUM) +
                                                             scaleL1BufId);
            uint64_t curScaleKL1 = scaleKL1_;
            if (kL1Offset + curScaleKL1 > k_) {
                curScaleKL1 = k_ - kL1Offset;
            }
            const uint64_t scaleOffset = GetScaleOffset(kL1Offset);
            const uint64_t scaleSpan = GetScaleSpan(curScaleKL1);
            auto copyScaleGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            auto gmBlockScaleA = gmScaleA.Slice(AscendC::Te::MakeCoord(0, scaleOffset),
                                                AscendC::Te::MakeShape(curM, scaleSpan));
            AscendC::Te::Copy(copyScaleGM2L1, tensorScaleAL1, gmBlockScaleA);

            auto gmBlockScaleBLeft = gmScaleBLeft.Slice(AscendC::Te::MakeCoord(scaleOffset, 0),
                                                        AscendC::Te::MakeShape(scaleSpan, curN));
            CopyConcatScaleBByStride(tensorScaleBL1, gmBlockScaleBLeft, curN, CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE),
                                     scaleL1BufId);
        }
        return ScalePair<decltype(tensorScaleAL1), decltype(tensorScaleBL1)>{tensorScaleAL1, tensorScaleBL1};
    }

    template <typename TensorA>
    __aicore__ inline auto CopyAInL1(const TensorA& gmA, uint64_t curM, uint64_t curGmAKL1, uint64_t aL1BufId,
                                     uint64_t kL1Offset)
    {
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        uint64_t l1K = curGmAKL1;
        if constexpr (NEED_A_SET_L1_K_ZERO) {
            l1K = CeilAlign(curGmAKL1, MXFP_DIVISOR_SIZE);
        }
        auto layoutAL1 = MakeLayoutAL1{}(curM, l1K);
        auto tensorAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(l1BufferAOffset_[aL1BufId]), layoutAL1);
        auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(0, kL1Offset), AscendC::Te::MakeShape(curM, curGmAKL1));
        if constexpr (NEED_A_SET_L1_K_ZERO) {
            Blaze::Gemm::Tile::PadMxKAL1::PadZero(tensorAL1, gmBlockA);
        }
        AscendC::Te::Copy(copyGM2L1, tensorAL1, gmBlockA);
        return tensorAL1;
    }

    template <typename TensorB>
    __aicore__ inline auto CopyBInL1(const TensorB& gmB, uint64_t curN, uint64_t curGmBKL1, uint64_t bL1BufId,
                                     uint64_t kL1Offset)
    {
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        uint64_t l1K = curGmBKL1;
        if constexpr (NEED_B_SET_L1_K_ZERO) {
            l1K = CeilAlign(curGmBKL1, MXFP_DIVISOR_SIZE);
        }
        auto layoutBL1 = MakeLayoutBL1{}(l1K, curN);
        auto tensorBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(l1BufferBOffset_[bL1BufId]), layoutBL1);
        auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(kL1Offset, 0), AscendC::Te::MakeShape(curGmBKL1, curN));
        if constexpr (NEED_B_SET_L1_K_ZERO) {
            Blaze::Gemm::Tile::PadMxKBL1::PadZero(tensorBL1, gmBlockB);
        }
        AscendC::Te::Copy(copyGM2L1, tensorBL1, gmBlockB);
        return tensorBL1;
    }

    template <typename TensorB>
    __aicore__ inline auto CopyConcatBInL1(TensorB gmBLeft, uint64_t curN, uint64_t curGmBKL1, uint64_t bL1BufId,
                                           uint64_t kL1Offset)
    {
        uint64_t l1K = curGmBKL1;
        if constexpr (NEED_B_SET_L1_K_ZERO) {
            l1K = CeilAlign(curGmBKL1, MXFP_DIVISOR_SIZE);
        }
        const uint64_t concatN = curN * CONCAT_N_FACTOR;
        auto layoutBL1 = MakeLayoutBL1{}(l1K, concatN);
        auto tensorBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(l1BufferBOffset_[bL1BufId]), layoutBL1);
        auto gmBlockBLeft = gmBLeft.Slice(AscendC::Te::MakeCoord(kL1Offset, 0),
                                          AscendC::Te::MakeShape(curGmBKL1, curN));
        if constexpr (NEED_B_SET_L1_K_ZERO) {
            Blaze::Gemm::Tile::PadMxKBL1::PadZero(tensorBL1, gmBlockBLeft);
        }
        CopyConcatBByStride<BType>(tensorBL1, gmBlockBLeft, curN, curGmBKL1, l1K);
        return tensorBL1;
    }

    template <typename TensorBias>
    __aicore__ inline auto CopyBiasInL1(const TensorBias& gmBias, uint64_t curN, uint64_t biasBufId, bool needCopyBias)
    {
        uint64_t biasNAlign = CeilAlign(curN, BLOCK_CUBE);
        auto layoutBiasL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, biasNAlign);
        auto tensorBiasL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BiasType>(l1BufferBiasOffset_[biasBufId]), layoutBiasL1);
        if (isBias_ && needCopyBias) {
            auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
            AscendC::Te::Copy(copyGM2L1, tensorBiasL1, gmBias);
        }
        return tensorBiasL1;
    }

    template <typename TensorL0C, typename TensorAL1, typename TensorBL1, typename TensorScaleAL1,
              typename TensorScaleBL1, typename TensorBiasL1>
    __aicore__ inline void Iterate(TensorL0C tensorL0C, TensorAL1 tensorAL1, TensorBL1 tensorBL1,
                                   TensorScaleAL1 tensorScaleAL1, TensorScaleBL1 tensorScaleBL1,
                                   TensorBiasL1 tensorBiasL1, uint64_t curM, uint64_t curN, uint64_t curGmAKL1,
                                   uint64_t curGmBKL1, uint64_t scaleL1BufId, uint64_t absKStartA, uint64_t absKStartB,
                                   uint64_t kaL1Offset, uint64_t kbL1Offset)
    {
        uint64_t l1Ka = NEED_A_SET_L1_K_ZERO ? CeilAlign(curGmAKL1, MXFP_DIVISOR_SIZE) : curGmAKL1;
        uint64_t l1Kb = NEED_B_SET_L1_K_ZERO ? CeilAlign(curGmBKL1, MXFP_DIVISOR_SIZE) : curGmBKL1;

        // Match the actual A/B L1 layouts built by conditional MX K padding, especially for split-K tails.
        uint64_t minPadKL1 = Min(l1Ka, l1Kb);
        uint64_t minGmKL1 = Min(curGmAKL1, curGmBKL1);
        uint64_t scaleBaseA = GetScaleOffset(absKStartA % scaleKL1_);
        uint64_t scaleBaseB = GetScaleOffset(absKStartB % scaleKL1_);
        auto tensorBlockScaleAL1 = tensorScaleAL1.Slice(AscendC::Te::MakeCoord(0, scaleBaseA),
                                                        AscendC::Te::MakeShape(curM, GetScaleSpan(curGmAKL1)));
        auto tensorBlockScaleBL1 = tensorScaleBL1.Slice(AscendC::Te::MakeCoord(scaleBaseB, 0),
                                                        AscendC::Te::MakeShape(GetScaleSpan(curGmBKL1), curN));
        uint64_t biasNAlign = CeilAlign(curN, BLOCK_CUBE);
        auto layoutBt = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, biasNAlign);
        auto tensorBt = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::BIAS, float>(baseN_ * scaleL1BufId * sizeof(float)),
            layoutBt);

        auto copyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
        auto copyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
        auto copyL12L0ScaleA = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0ScaleA{});
        auto copyL12L0ScaleB = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0ScaleB{});
        auto copyL12BT = AscendC::Te::MakeCopy(AscendC::Te::CopyL12BT{});
        constexpr auto
            mmadAtom = AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, Tile::MmadTraitMX>>{};
        const uint64_t halfL0Size = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;

        // Walk the current L1 K strip in baseK L0 tiles, copying A/B/scale to L0 before each MMAD.
        for (uint64_t kL0Offset = 0; kL0Offset < minGmKL1; kL0Offset += baseK_) {
            uint64_t curKL0 = (kL0Offset + baseK_ > minPadKL1) ? (minPadKL1 - kL0Offset) : baseK_;
            uint64_t absKOffset = absKStartA + kaL1Offset + kL0Offset;
            uint64_t scaleOffsetA = GetScaleOffset(kaL1Offset + kL0Offset);
            uint64_t scaleOffsetB = GetScaleOffset(kbL1Offset + kL0Offset);
            uint64_t curKL0ScaleSpan = GetScaleSpan(curKL0);
            bool needBiasThisIter = NeedBias(absKOffset);
            uint64_t l0Offset = halfL0Size * (l0PingPong_ & 1UL);
            uint16_t mte1WaitMFlag = static_cast<uint16_t>((l0PingPong_ & 1UL) + M_MTE1_FLAG_0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);

            auto tensorAL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(l0Offset),
                AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>(curM, curKL0));
            auto tensorBlockAL1 = tensorAL1.Slice(AscendC::Te::MakeCoord(0, kaL1Offset + kL0Offset),
                                                  AscendC::Te::MakeShape(curM, curKL0));
            AscendC::Te::Copy(copyL12L0A, tensorAL0, tensorBlockAL1);

            auto tensorScaleAL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0ScaleA, fp8_e8m0_t>(l0Offset >> 4),
                AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(curM,
                                                                                                    curKL0ScaleSpan));
            AscendC::Te::Copy(copyL12L0ScaleA, tensorScaleAL0,
                              tensorBlockScaleAL1.Slice(AscendC::Te::MakeCoord(0, scaleOffsetA),
                                                        AscendC::Te::MakeShape(curM, curKL0ScaleSpan)));

            if (needBiasThisIter) {
                AscendC::Te::Copy(copyL12BT, tensorBt, tensorBiasL1);
            }

            auto tensorBL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(l0Offset),
                AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>(curKL0, curN));
            auto tensorBlockBL1 = tensorBL1.Slice(AscendC::Te::MakeCoord(kbL1Offset + kL0Offset, 0),
                                                  AscendC::Te::MakeShape(curKL0, curN));
            AscendC::Te::Copy(copyL12L0B, tensorBL0, tensorBlockBL1);

            auto tensorScaleBL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0ScaleB, fp8_e8m0_t>(l0Offset >> 4),
                AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(curKL0ScaleSpan,
                                                                                                    curN));
            AscendC::Te::Copy(copyL12L0ScaleB, tensorScaleBL0,
                              tensorBlockScaleBL1.Slice(AscendC::Te::MakeCoord(scaleOffsetB, 0),
                                                        AscendC::Te::MakeShape(curKL0ScaleSpan, curN)));

            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 1UL);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 1UL);

            AscendC::Te::MmadParams mmadParams;
            mmadParams.m = static_cast<uint16_t>(curM);
            mmadParams.k = static_cast<uint16_t>(CeilAlign(curKL0, MXFP_DIVISOR_SIZE));
            mmadParams.n = static_cast<uint16_t>(curN);
            mmadParams.unitFlag = (absKOffset + curKL0 >= k_) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
            mmadParams.cmatrixInitVal = !needBiasThisIter && (absKOffset == 0);
            if (needBiasThisIter) {
                AscendC::Te::Mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0, tensorBt);
            } else {
                AscendC::Te::Mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0);
            }

            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);
            l0PingPong_++;
        }
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorBias,
              typename TensorC>
    __aicore__ inline void RunImpl(const TensorA& gmA, const TensorB& gmB, const TensorScaleA& gmScaleA,
                                   const TensorScaleB& gmScaleB, const TensorBias& gmBias, const TensorC& tensorC,
                                   const BlockShape& blockShape)
    {
        constexpr bool
            isUbOutput = AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<TensorC>, AscendC::Te::Location::UB>;
        constexpr bool
            isGmOutput = AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<TensorC>, AscendC::Te::Location::GM>;
        static_assert(isUbOutput || isGmOutput, "Block MMAD output tensor must be located in GM or UB.");

        // Current output tile shape.
        uint64_t curM = AscendC::Te::Get<MNK_M>(blockShape);
        uint64_t curN = AscendC::Te::Get<MNK_N>(blockShape);
        const uint64_t halfL0CSize = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT;
        uint64_t l0cOffset = (l0cPingPong_ & 1UL) * halfL0CSize;
        uint64_t l0cLayoutM = curM;
        if constexpr (isUbOutput) {
            // DUAL_DST_SPLIT_M requires an even Fixpipe mSize. Keep curM unchanged for GM slices and MMAD,
            // and expose one padding row only through the L0C-to-UB copy layout when the logical M is odd.
            l0cLayoutM = CeilAlign(curM, SPLIT_M_ALIGN);
        }
        auto layoutL0C = AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE_L0C>>{}(
            l0cLayoutM, curN);
        auto tensorL0C = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(l0cOffset),
                                                 layoutL0C);

        if (orderAL1BL1_) {
            // A-major: kAL1 >= kBL1, so A is reused across the inner B K strips.
            for (uint64_t kOuter = 0; kOuter < k_; kOuter += kAL1_) {
                uint64_t scaleL1BufId = scaleLoopCnt_ & 1UL;
                uint64_t aL1BufId = GetL1BufId(aL1LoopCnt_);
                uint64_t nextKOuter = kOuter + kAL1_;
                uint64_t curGmAKL1 = (nextKOuter > k_) ? (k_ - kOuter) : kAL1_;
                // Copy scales once per scaleKL1 window; this may be a no-op inside the window.
                auto scalePair = CopyScalesInL1(gmScaleA, gmScaleB, curM, curN, kOuter, scaleL1BufId);
                auto& tensorScaleAL1 = scalePair.scaleA;
                auto& tensorScaleBL1 = scalePair.scaleB;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
                auto tensorAL1 = CopyAInL1(gmA, curM, curGmAKL1, aL1BufId, kOuter);
                // Bias is consumed only by the first effective K tile.
                auto tensorBiasL1 = CopyBiasInL1(gmBias, curN, scaleL1BufId, NeedBias(kOuter));

                // Walk B K strips under the current A strip, then stage A/B/scale to L0 and MMAD.
                for (uint64_t kInner = kOuter; kInner < Min(kOuter + kAL1_, k_); kInner += kBL1_) {
                    uint64_t bL1BufId = GetL1BufId(bL1LoopCnt_);
                    uint64_t curGmBKL1 = (kInner + kBL1_ > k_) ? (k_ - kInner) : kBL1_;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufferStage_) + bL1BufId);
                    auto tensorBL1 = CopyBInL1(gmB, curN, curGmBKL1, bL1BufId, kInner);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufId);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufId);
                    Iterate(tensorL0C, tensorAL1, tensorBL1, tensorScaleAL1, tensorScaleBL1, tensorBiasL1, curM, curN,
                            curGmAKL1, curGmBKL1, scaleL1BufId, kOuter, kInner, kInner - kOuter, 0);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufferStage_) + bL1BufId);
                    bL1LoopCnt_++;
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
                // Release the scale buffer when the copied scale window has been fully consumed.
                if ((nextKOuter % scaleKL1_) == 0 || nextKOuter >= k_) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
                        static_cast<uint16_t>(l1BufferStage_ * SCALE_BUFFER_NUM) + scaleL1BufId);
                    scaleLoopCnt_++;
                }
                aL1LoopCnt_++;
            }
        } else {
            // B-major: kBL1 > kAL1, so B is reused across the inner A K strips.
            for (uint64_t kOuter = 0; kOuter < k_; kOuter += kBL1_) {
                uint64_t scaleL1BufId = scaleLoopCnt_ & 1UL;
                uint64_t bL1BufId = GetL1BufId(bL1LoopCnt_);
                uint64_t nextKOuter = kOuter + kBL1_;
                uint64_t curGmBKL1 = (nextKOuter > k_) ? (k_ - kOuter) : kBL1_;
                // Copy scales once per scaleKL1 window; this may be a no-op inside the window.
                auto scalePair = CopyScalesInL1(gmScaleA, gmScaleB, curM, curN, kOuter, scaleL1BufId);
                auto& tensorScaleAL1 = scalePair.scaleA;
                auto& tensorScaleBL1 = scalePair.scaleB;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufferStage_) + bL1BufId);
                auto tensorBL1 = CopyBInL1(gmB, curN, curGmBKL1, bL1BufId, kOuter);
                // Bias is consumed only by the first effective K tile.
                auto tensorBiasL1 = CopyBiasInL1(gmBias, curN, scaleL1BufId, NeedBias(kOuter));

                // Walk A K strips under the current B strip, then stage A/B/scale to L0 and MMAD.
                for (uint64_t kInner = kOuter; kInner < Min(kOuter + kBL1_, k_); kInner += kAL1_) {
                    uint64_t aL1BufId = GetL1BufId(aL1LoopCnt_);
                    uint64_t curGmAKL1 = (kInner + kAL1_ > k_) ? (k_ - kInner) : kAL1_;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
                    auto tensorAL1 = CopyAInL1(gmA, curM, curGmAKL1, aL1BufId, kInner);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(aL1BufId);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(aL1BufId);
                    Iterate(tensorL0C, tensorAL1, tensorBL1, tensorScaleAL1, tensorScaleBL1, tensorBiasL1, curM, curN,
                            curGmAKL1, curGmBKL1, scaleL1BufId, kInner, kOuter, 0, kInner - kOuter);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
                    aL1LoopCnt_++;
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufferStage_) + bL1BufId);
                // Release the scale buffer when the copied scale window has been fully consumed.
                if ((nextKOuter % scaleKL1_) == 0 || nextKOuter >= k_) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
                        static_cast<uint16_t>(l1BufferStage_ * SCALE_BUFFER_NUM) + scaleL1BufId);
                    scaleLoopCnt_++;
                }
                bL1LoopCnt_++;
            }
        }

        if constexpr (isUbOutput) {
            auto copyL0C2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2UB{},
                                                    Blaze::Gemm::Tile::CopyL0C2UBTraitMixSplitM{});
            AscendC::Te::Copy(copyL0C2UB.with(AscendC::Te::FixpipeParams(FINAL_ACCUMULATION)), tensorC, tensorL0C);
        } else {
            auto copyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
            AscendC::Te::Copy(copyL0C2GM.with(AscendC::Te::FixpipeParams(FINAL_ACCUMULATION)), tensorC, tensorL0C);
        }
        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
    }

    template <typename TensorUbC, typename TensorL0C>
    __aicore__ inline void CopyConcatL0C2SingleUb(TensorUbC& ubC, TensorL0C tensorL0C, uint64_t, uint64_t)
    {
        auto copyL0C2UBSplitM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2UB{},
                                                      Blaze::Gemm::Tile::CopyL0C2UBTraitSplitM{});
        AscendC::Te::Copy(copyL0C2UBSplitM.with(AscendC::Te::FixpipeParams(FINAL_ACCUMULATION)), ubC, tensorL0C);
    }

    template <typename TensorL0C, typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB>
    __aicore__ inline void RunConcatNWithAOuter(TensorL0C tensorL0C, TensorA gmA, TensorB gmBLeft,
                                                TensorScaleA gmScaleA, TensorScaleB gmScaleBLeft, uint64_t curM,
                                                uint64_t curN, uint64_t concatN)
    {
        for (uint64_t kOuter = 0; kOuter < k_; kOuter += kAL1_) {
            uint64_t scaleL1BufId = scaleLoopCnt_ & 1UL;
            uint64_t aL1BufId = GetL1BufId(aL1LoopCnt_);
            uint64_t nextKOuter = kOuter + kAL1_;
            uint64_t curGmAKL1 = (nextKOuter > k_) ? (k_ - kOuter) : kAL1_;
            auto scalePair = CopyConcatScalesInL1(gmScaleA, gmScaleBLeft, curM, curN, kOuter, scaleL1BufId);
            auto& tensorScaleAL1 = scalePair.scaleA;
            auto& tensorScaleBL1 = scalePair.scaleB;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
            auto tensorAL1 = CopyAInL1(gmA, curM, curGmAKL1, aL1BufId, kOuter);

            for (uint64_t kInner = kOuter; kInner < Min(kOuter + kAL1_, k_); kInner += kBL1_) {
                uint64_t bL1BufId = GetL1BufId(bL1LoopCnt_);
                uint64_t curGmBKL1 = (kInner + kBL1_ > k_) ? (k_ - kInner) : kBL1_;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufferStage_) + bL1BufId);
                auto tensorBL1 = CopyConcatBInL1(gmBLeft, curN, curGmBKL1, bL1BufId, kInner);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufId);
                IterateNoBias(tensorL0C, tensorAL1, tensorBL1, tensorScaleAL1, tensorScaleBL1, curM, concatN, curGmAKL1,
                              curGmBKL1, kOuter, kInner, kInner - kOuter, 0);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufferStage_) + bL1BufId);
                bL1LoopCnt_++;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
            if ((nextKOuter % scaleKL1_) == 0 || nextKOuter >= k_) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
                    static_cast<uint16_t>(l1BufferStage_ * SCALE_BUFFER_NUM) + scaleL1BufId);
                scaleLoopCnt_++;
            }
            aL1LoopCnt_++;
        }
    }

    template <typename TensorL0C, typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB>
    __aicore__ inline void RunConcatNWithBOuter(TensorL0C tensorL0C, TensorA gmA, TensorB gmBLeft,
                                                TensorScaleA gmScaleA, TensorScaleB gmScaleBLeft, uint64_t curM,
                                                uint64_t curN, uint64_t concatN)
    {
        for (uint64_t kOuter = 0; kOuter < k_; kOuter += kBL1_) {
            uint64_t scaleL1BufId = scaleLoopCnt_ & 1UL;
            uint64_t bL1BufId = GetL1BufId(bL1LoopCnt_);
            uint64_t nextKOuter = kOuter + kBL1_;
            uint64_t curGmBKL1 = (nextKOuter > k_) ? (k_ - kOuter) : kBL1_;
            auto scalePair = CopyConcatScalesInL1(gmScaleA, gmScaleBLeft, curM, curN, kOuter, scaleL1BufId);
            auto& tensorScaleAL1 = scalePair.scaleA;
            auto& tensorScaleBL1 = scalePair.scaleB;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufferStage_) + bL1BufId);
            auto tensorBL1 = CopyConcatBInL1(gmBLeft, curN, curGmBKL1, bL1BufId, kOuter);

            for (uint64_t kInner = kOuter; kInner < Min(kOuter + kBL1_, k_); kInner += kAL1_) {
                uint64_t aL1BufId = GetL1BufId(aL1LoopCnt_);
                uint64_t curGmAKL1 = (kInner + kAL1_ > k_) ? (k_ - kInner) : kAL1_;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
                auto tensorAL1 = CopyAInL1(gmA, curM, curGmAKL1, aL1BufId, kInner);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(aL1BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(aL1BufId);
                IterateNoBias(tensorL0C, tensorAL1, tensorBL1, tensorScaleAL1, tensorScaleBL1, curM, concatN, curGmAKL1,
                              curGmBKL1, kInner, kOuter, 0, kInner - kOuter);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
                aL1LoopCnt_++;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufferStage_) + bL1BufId);
            if ((nextKOuter % scaleKL1_) == 0 || nextKOuter >= k_) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
                    static_cast<uint16_t>(l1BufferStage_ * SCALE_BUFFER_NUM) + scaleL1BufId);
                scaleLoopCnt_++;
            }
            bL1LoopCnt_++;
        }
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorUbC>
    __aicore__ inline void RunConcatN(TensorA gmA, TensorB gmBLeft, TensorScaleA gmScaleA, TensorScaleB gmScaleBLeft,
                                      TensorUbC& ubC, const BlockShape& singleShape)
    {
        uint64_t curM = AscendC::Te::Get<MNK_M>(singleShape);
        uint64_t curN = AscendC::Te::Get<MNK_N>(singleShape);
        const uint64_t concatN = curN * CONCAT_N_FACTOR;
        const uint64_t halfL0CSize = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT;
        uint64_t l0cOffset = (l0cPingPong_ & 1UL) * halfL0CSize;
        uint64_t l0cLayoutM = curM;
        l0cLayoutM = CeilAlign(curM, SPLIT_M_ALIGN);
        auto layoutL0C = AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE_L0C>>{}(
            l0cLayoutM, concatN);
        auto tensorL0C = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(l0cOffset),
                                                 layoutL0C);

        if (orderAL1BL1_) {
            RunConcatNWithAOuter(tensorL0C, gmA, gmBLeft, gmScaleA, gmScaleBLeft, curM, curN, concatN);
        } else {
            RunConcatNWithBOuter(tensorL0C, gmA, gmBLeft, gmScaleA, gmScaleBLeft, curM, curN, concatN);
        }

        CopyConcatL0C2SingleUb(ubC, tensorL0C, curM, curN);
        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
    }

private:
    uint64_t m_{0};
    uint64_t n_{0};
    uint64_t k_{0};
    uint64_t kAL1_{1};
    uint64_t kBL1_{1};
    uint64_t scaleKL1_{1};
    uint64_t scaleKL1Span_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    uint64_t aL1LoopCnt_{0};
    uint64_t bL1LoopCnt_{0};
    uint64_t scaleLoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    uint64_t aL1OneBuffer_{0};
    uint64_t bL1OneBuffer_{0};
    uint64_t concatBL1OneBuffer_{0};
    uint64_t scaleAL1OneBuffer_{0};
    uint64_t scaleBL1OneBuffer_{0};
    uint64_t biasL1OneBuffer_{0};
    uint64_t l1BufferStage_{DOUBLE_BUFFER_COUNT};
    uint64_t l1BufferAOffset_[TRIPLE_BUFFER_COUNT] = {0, 0, 0};
    uint64_t l1BufferBOffset_[TRIPLE_BUFFER_COUNT] = {0, 0, 0};
    uint64_t l1BufferScaleAOffset_[2] = {0, 0};
    uint64_t l1BufferScaleBOffset_[2] = {0, 0};
    uint64_t l1BufferBiasOffset_[2] = {0, 0};
    bool orderAL1BL1_{false};
    bool isBias_{false};
    bool enableL0cPingPong_{false};
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
