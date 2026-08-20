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
 * \file mega_moe_block_mmad_mx_fp8.h
 * \brief arch92 repository-local MX FP8 BlockMmad implementation.
 */
#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/tile/tile_trait.h"
#include "blaze/gemm/utils/common_utils.h"
#include "c_api/asc_simd.h"
#include "c_api/cube_datamove/cube_datamove.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

using AscendC::HardEvent;
using AscendC::SetFlag;
using AscendC::TEventID;
using AscendC::WaitFlag;

// arch92 CANN does not provide this MatmulWithScaleMx BlockMmad binding. Keep the
// public contract identical to the generic Blaze implementation used by MegaMoe.
#define MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_TEMPLATE_PARAMS                                                        \
    template <class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_,          \
              class BiasType_, class LayoutBias_>

#define MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_SPECIALIZATION                                                         \
    BlockMmad<MatmulWithScaleMx<>, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_, LayoutBias_>

MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_TEMPLATE_PARAMS
class MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_SPECIALIZATION {
public:
    using DispatchPolicy = MatmulWithScaleMx<>;
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    static_assert(AscendC::Std::is_same_v<AType, fp8_e4m3fn_t> ||
                      AscendC::Std::is_same_v<AType, fp8_e5m2_t>,
                  "arch92 MegaMoe BlockMmad only supports FP8 A");
    static_assert(AscendC::Std::is_same_v<BType, fp8_e4m3fn_t> ||
                      AscendC::Std::is_same_v<BType, fp8_e5m2_t>,
                  "arch92 MegaMoe BlockMmad only supports FP8 B");
    static_assert(AscendC::Std::is_same_v<AType, BType>,
                  "arch92 MegaMoe BlockMmad requires matching FP8 A/B types");

    struct L1Params {
        uint64_t kL1;
        uint64_t scaleKL1;
        uint64_t l1BufNum;
    };

    __aicore__ inline BlockMmad();
    __aicore__ inline ~BlockMmad();

    __aicore__ inline void Init(const ProblemShape &problemShape, const BlockShape &l0TileShape,
                                const L1Params &l1Params, bool isBias, bool enableL0cPingPong);

    __aicore__ inline void EnableOneShotDebug(uint32_t markerBase, bool dumpPartialAccumulations)
    {
        if (!debugUsed_) {
            debugEnabled_ = true;
            debugMarkerBase_ = markerBase;
            debugPartialAccumulations_ = dumpPartialAccumulations;
        }
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB,
              typename TensorBias, typename TensorC>
    __aicore__ inline void operator()(const TensorA &tensorA, const TensorB &tensorB,
                                      const TensorScaleA &tensorScaleA, const TensorScaleB &tensorScaleB,
                                      const TensorBias &tensorBias, const TensorC &tensorC,
                                      const BlockShape &singleShape);

private:
    struct TileParams {
        uint64_t m;
        uint64_t n;
        uint64_t kOffset;
        uint64_t realK;
        uint64_t scaleWindowStart;
        uint64_t scaleWindowLength;
        uint64_t scaleKOffsetInWindow;
    };

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB,
              typename TensorC>
    __aicore__ inline void ProcessKBlock(const TensorA &tensorA, const TensorB &tensorB,
                                         const TensorScaleA &tensorScaleA,
                                         const TensorScaleB &tensorScaleB, const TensorC &tensorC,
                                         const TileParams &params, bool debugThisTile);

    template <typename TensorAL1, typename TensorBL1, typename TensorScaleAL1, typename TensorScaleBL1,
              typename TensorC>
    __aicore__ inline void ProcessL0Tiles(const TensorAL1 &tensorAL1, const TensorBL1 &tensorBL1,
                                          const TensorScaleAL1 &tensorScaleAL1,
                                          const TensorScaleBL1 &tensorScaleBL1, const TensorC &tensorC,
                                          const TileParams &params, bool debugThisTile);

    template <typename TensorC>
    __aicore__ inline void CopyOut(const TensorC &tensorC, uint64_t m, uint64_t n);

    template <typename TensorC, typename TensorL0C>
    __aicore__ inline void DebugDumpPartialL0C(
        const TensorC &tensorC, const TensorL0C &tensorL0C, uint64_t completedK,
        uint32_t marker);

    template <typename TensorL0C>
    __aicore__ inline void DebugDumpDirectL0C2UB(
        const TensorL0C &tensorL0C, uint64_t m, uint64_t n, bool quantPreRnd,
        uint32_t marker);

    __aicore__ inline static void DebugSetCondUint(uint64_t value)
    {
        set_cond(DEBUG_UINT_BIAS + static_cast<unsigned long>(value));
    }

    __aicore__ inline uint64_t SelectKL1(uint64_t requestedKL1) const;
    __aicore__ inline uint64_t CalcL1BufferBytes(uint64_t kL1, uint64_t scaleKL1) const;
    __aicore__ inline void UpdateL1Offsets();

    __aicore__ inline static uint64_t AlignK(uint64_t value)
    {
        return Blaze::Gemm::CeilDiv(value, MX_SCALE_GROUP_K) * MX_SCALE_GROUP_K;
    }

    template <typename TensorL1>
    __aicore__ inline static void FillL1Zero(const TensorL1 &tensorL1, uint64_t repeatTimes,
                                             uint64_t blockNum, uint64_t dstGap)
    {
        asc_fill_value_config config;
        config.repeat = repeatTimes;
        config.blk_num = blockNum;
        config.dst_gap = dstGap;
        asc_fill_l1((__cbuf__ half *)tensorL1.Data().Get(), half(0), config);
    }

    template <typename TensorL1, typename TensorGm>
    __aicore__ inline static void PadAKTail(const TensorL1 &tensorL1, const TensorGm &tensorGm)
    {
        auto layoutL1 = tensorL1.Layout();
        auto layoutGm = tensorGm.Layout();
        uint64_t realK = AscendC::Std::get<1>(AscendC::Std::get<1>(layoutGm.Shape()));
        uint64_t alignedK = AscendC::Std::get<0>(AscendC::Std::get<1>(layoutL1.Shape())) *
                            AscendC::Std::get<1>(AscendC::Std::get<1>(layoutL1.Shape()));
        if (alignedK - realK < C0_SIZE_A) {
            return;
        }

        uint64_t alignedM = AscendC::Std::get<0>(AscendC::Std::get<0>(layoutL1.Shape())) *
                            AscendC::Std::get<1>(AscendC::Std::get<0>(layoutL1.Shape()));
        uint64_t copyAlignedK = Blaze::Gemm::CeilDiv(realK, C0_SIZE_A) * C0_SIZE_A;
        auto tailTensor = tensorL1.Slice(AscendC::Te::MakeCoord(0U, copyAlignedK),
                                         AscendC::Te::MakeShape(alignedM, alignedK - copyAlignedK));
        FillL1Zero(tailTensor, 1U, alignedM, 0U);
    }

    template <typename TensorL1, typename TensorGm>
    __aicore__ inline static void PadBKTail(const TensorL1 &tensorL1, const TensorGm &tensorGm)
    {
        auto layoutL1 = tensorL1.Layout();
        auto layoutGm = tensorGm.Layout();
        uint64_t realK = AscendC::Std::get<0>(AscendC::Std::get<0>(layoutGm.Shape())) *
                         AscendC::Std::get<1>(AscendC::Std::get<0>(layoutGm.Shape()));
        uint64_t alignedK = AscendC::Std::get<0>(AscendC::Std::get<0>(layoutL1.Shape())) *
                            AscendC::Std::get<1>(AscendC::Std::get<0>(layoutL1.Shape()));
        if (alignedK - realK < C0_SIZE_B) {
            return;
        }

        uint64_t alignedN = AscendC::Std::get<0>(AscendC::Std::get<1>(layoutL1.Shape())) *
                            AscendC::Std::get<1>(AscendC::Std::get<1>(layoutL1.Shape()));
        uint64_t copyAlignedK = Blaze::Gemm::CeilDiv(realK, C0_SIZE_B) * C0_SIZE_B;
        auto tailTensor = tensorL1.Slice(AscendC::Te::MakeCoord(copyAlignedK, 0U),
                                         AscendC::Te::MakeShape(alignedK - copyAlignedK, alignedN));
        FillL1Zero(tailTensor, 1U, alignedN, 0U);
    }

    static constexpr uint64_t M_INDEX = 0U;
    static constexpr uint64_t N_INDEX = 1U;
    static constexpr uint64_t K_INDEX = 2U;
    static constexpr uint64_t BUFFER_COUNT = 2U;
    static constexpr uint64_t DEFAULT_K_L1 = 256U;
    static constexpr uint64_t MX_SCALE_GROUP_K = 64U;
    static constexpr uint64_t MX_SCALE_ELEMENTS_PER_GROUP = 2U;
    static constexpr uint64_t L0_BUFFER_BYTES = 32U * 1024U;
    static constexpr uint64_t DEBUG_UB_OFFSET = 0U;
    static constexpr uint32_t DEBUG_VALUE_COUNT = 20U;
    static constexpr unsigned long DEBUG_UINT_BIAS = 1000000UL;
    static constexpr uint64_t SCALE_C0 = 2U;
    static constexpr uint64_t L0C_C0 = 16U;
    static constexpr TEventID INPUT_BUFFER_EVENT_BASE = 0;
    // Match the native QBMM MX event partition: input buffers use 0-3, scales and MMAD release use 4-5.
    static constexpr TEventID SCALE_BUFFER_EVENT_BASE = 4;
    static constexpr TEventID MTE2_TO_MTE1_EVENT_BASE = 0;
    static constexpr TEventID MMAD_TO_MTE1_EVENT_BASE = 4;
    static constexpr TEventID MTE1_TO_MMAD_EVENT_BASE = 0;

    static constexpr uint64_t C0_SIZE_A = AscendC::AuxGetC0Size<AType>();
    static constexpr uint64_t C0_SIZE_B = AscendC::AuxGetC0Size<BType>();

    using MakeLayoutAL1 =
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE_A>>;
    using MakeLayoutBL1 =
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE_B>>;
    using MakeLayoutScaleAL1 =
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>;
    using MakeLayoutScaleBL1 =
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>;
    using MakeLayoutAL0 = MakeLayoutAL1;
    using MakeLayoutBL0 = MakeLayoutBL1;
    using MakeLayoutScaleAL0 = MakeLayoutScaleAL1;
    using MakeLayoutScaleBL0 = MakeLayoutScaleBL1;

    uint64_t k_{1U};
    uint64_t kL1_{DEFAULT_K_L1};
    uint64_t scaleKL1_{DEFAULT_K_L1};
    uint64_t baseM_{256U};
    uint64_t baseN_{256U};
    uint64_t baseK_{128U};
    uint64_t l1BufferIndex_{0U};
    uint64_t scaleL1BufferIndex_{0U};
    uint64_t l0BufferIndex_{0U};
    bool debugEnabled_{false};
    bool debugUsed_{false};
    bool debugPartialAccumulations_{false};
    uint32_t debugMarkerBase_{0U};
    uint32_t debugStep_{0U};
    uint64_t l1AOffset_[BUFFER_COUNT]{};
    uint64_t l1BOffset_[BUFFER_COUNT]{};
    uint64_t l1ScaleAOffset_[BUFFER_COUNT]{};
    uint64_t l1ScaleBOffset_[BUFFER_COUNT]{};
};

MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_TEMPLATE_PARAMS
__aicore__ inline MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_SPECIALIZATION::
    BlockMmad()
{
    for (uint64_t i = 0U; i < BUFFER_COUNT; ++i) {
        SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_EVENT_BASE + i);
        SetFlag<HardEvent::MTE1_MTE2>(SCALE_BUFFER_EVENT_BASE + i);
        SetFlag<HardEvent::M_MTE1>(MMAD_TO_MTE1_EVENT_BASE + i);
    }
    AscendC::SetMMLayoutTransform(true);
}

MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_TEMPLATE_PARAMS
__aicore__ inline MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_SPECIALIZATION::
    ~BlockMmad()
{
    for (uint64_t i = 0U; i < BUFFER_COUNT; ++i) {
        WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_EVENT_BASE + i);
        WaitFlag<HardEvent::MTE1_MTE2>(SCALE_BUFFER_EVENT_BASE + i);
        WaitFlag<HardEvent::M_MTE1>(MMAD_TO_MTE1_EVENT_BASE + i);
    }
    AscendC::SetMMLayoutTransform(false);
}

MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_TEMPLATE_PARAMS
__aicore__ inline uint64_t MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_SPECIALIZATION::CalcL1BufferBytes(
    uint64_t kL1, uint64_t scaleKL1) const
{
    uint64_t scaleKLength = Blaze::Gemm::CeilDiv(scaleKL1, MX_SCALE_GROUP_K) * MX_SCALE_ELEMENTS_PER_GROUP;
    return baseM_ * kL1 * sizeof(AType) + baseN_ * kL1 * sizeof(BType) +
           baseM_ * scaleKLength * sizeof(fp8_e8m0_t) +
           baseN_ * scaleKLength * sizeof(fp8_e8m0_t);
}

MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_TEMPLATE_PARAMS
__aicore__ inline uint64_t MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_SPECIALIZATION::SelectKL1(
    uint64_t requestedKL1) const
{
    uint64_t selectedKL1 = AlignK(requestedKL1 == 0U ? DEFAULT_K_L1 : requestedKL1);
    constexpr uint64_t halfL1Size = AscendC::TOTAL_L1_SIZE / BUFFER_COUNT;
    if (CalcL1BufferBytes(selectedKL1, scaleKL1_) > halfL1Size) {
        selectedKL1 = DEFAULT_K_L1;
    }
    return selectedKL1;
}

MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_TEMPLATE_PARAMS
__aicore__ inline void MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_SPECIALIZATION::UpdateL1Offsets()
{
    uint64_t scaleKLength = Blaze::Gemm::CeilDiv(scaleKL1_, MX_SCALE_GROUP_K) * MX_SCALE_ELEMENTS_PER_GROUP;
    uint64_t aBytes = baseM_ * kL1_ * sizeof(AType);
    uint64_t bBytes = baseN_ * kL1_ * sizeof(BType);
    uint64_t scaleABytes = baseM_ * scaleKLength * sizeof(fp8_e8m0_t);
    constexpr uint64_t halfL1Size = AscendC::TOTAL_L1_SIZE / BUFFER_COUNT;
    for (uint64_t i = 0U; i < BUFFER_COUNT; ++i) {
        uint64_t bufferBase = i * halfL1Size;
        l1AOffset_[i] = bufferBase;
        l1BOffset_[i] = l1AOffset_[i] + aBytes;
        l1ScaleAOffset_[i] = l1BOffset_[i] + bBytes;
        l1ScaleBOffset_[i] = l1ScaleAOffset_[i] + scaleABytes;
    }
}

MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_TEMPLATE_PARAMS
__aicore__ inline void MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_SPECIALIZATION::Init(
    const ProblemShape &problemShape, const BlockShape &l0TileShape, const L1Params &l1Params,
    bool isBias, bool enableL0cPingPong)
{
    k_ = AscendC::Te::Get<K_INDEX>(problemShape);
    baseM_ = AscendC::Te::Get<M_INDEX>(l0TileShape);
    baseN_ = AscendC::Te::Get<N_INDEX>(l0TileShape);
    baseK_ = AscendC::Te::Get<K_INDEX>(l0TileShape);
    scaleKL1_ = AlignK(l1Params.scaleKL1 == 0U ? DEFAULT_K_L1 : l1Params.scaleKL1);
    kL1_ = SelectKL1(l1Params.kL1);
    if (scaleKL1_ < kL1_ || scaleKL1_ % kL1_ != 0U) {
        scaleKL1_ = kL1_;
    }
    constexpr uint64_t halfL1Size = AscendC::TOTAL_L1_SIZE / BUFFER_COUNT;
    if (CalcL1BufferBytes(kL1_, scaleKL1_) > halfL1Size) {
        kL1_ = DEFAULT_K_L1;
        scaleKL1_ = kL1_;
    }
    l1BufferIndex_ = 0U;
    scaleL1BufferIndex_ = 0U;
    l0BufferIndex_ = 0U;
    UpdateL1Offsets();

    // MegaMoe does not use bias or L0C ping-pong on this path. Keep the generic
    // Init signature so callers remain identical to the A5 generic BlockMmad.
    (void)isBias;
    (void)enableL0cPingPong;
    (void)l1Params.l1BufNum;
}

MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_TEMPLATE_PARAMS
template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB,
          typename TensorBias, typename TensorC>
__aicore__ inline void MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_SPECIALIZATION::operator()(
    const TensorA &tensorA, const TensorB &tensorB, const TensorScaleA &tensorScaleA,
    const TensorScaleB &tensorScaleB, const TensorBias &tensorBias, const TensorC &tensorC,
    const BlockShape &singleShape)
{
    uint64_t m = AscendC::Te::Get<M_INDEX>(singleShape);
    uint64_t n = AscendC::Te::Get<N_INDEX>(singleShape);
    bool debugThisTile = debugEnabled_ && !debugUsed_;
    if (debugThisTile) {
        debugEnabled_ = false;
        debugUsed_ = true;
        debugStep_ = 0U;
    }
    for (uint64_t kOffset = 0U; kOffset < k_; kOffset += kL1_) {
        uint64_t realK = Blaze::Gemm::Min(kL1_, k_ - kOffset);
        uint64_t scaleWindowStart = (kOffset / scaleKL1_) * scaleKL1_;
        uint64_t realScaleWindowK = Blaze::Gemm::Min(scaleKL1_, k_ - scaleWindowStart);
        TileParams params{
            m,
            n,
            kOffset,
            realK,
            scaleWindowStart,
            Blaze::Gemm::CeilDiv(realScaleWindowK, MX_SCALE_GROUP_K) * MX_SCALE_ELEMENTS_PER_GROUP,
            Blaze::Gemm::CeilDiv(kOffset - scaleWindowStart, MX_SCALE_GROUP_K) *
                MX_SCALE_ELEMENTS_PER_GROUP};
        ProcessKBlock(tensorA, tensorB, tensorScaleA, tensorScaleB, tensorC, params,
                      debugThisTile);
    }
    CopyOut(tensorC, m, n);
    (void)tensorBias;
}

MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_TEMPLATE_PARAMS
template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB,
          typename TensorC>
__aicore__ inline void MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_SPECIALIZATION::ProcessKBlock(
    const TensorA &tensorA, const TensorB &tensorB, const TensorScaleA &tensorScaleA,
    const TensorScaleB &tensorScaleB, const TensorC &tensorC, const TileParams &params,
    bool debugThisTile)
{
    uint64_t bufferIndex = l1BufferIndex_ & (BUFFER_COUNT - 1U);
    uint64_t scaleBufferIndex = scaleL1BufferIndex_ & (BUFFER_COUNT - 1U);
    uint64_t alignedK = AlignK(params.realK);
    WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_EVENT_BASE + bufferIndex);

    auto tensorAL1 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(l1AOffset_[bufferIndex]),
        MakeLayoutAL1{}(params.m, alignedK));
    auto tensorBL1 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(l1BOffset_[bufferIndex]),
        MakeLayoutBL1{}(alignedK, params.n));
    auto tensorScaleAL1 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(
            l1ScaleAOffset_[scaleBufferIndex]),
        MakeLayoutScaleAL1{}(params.m, params.scaleWindowLength));
    auto tensorScaleBL1 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(
            l1ScaleBOffset_[scaleBufferIndex]),
        MakeLayoutScaleBL1{}(params.scaleWindowLength, params.n));

    auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
    auto gmBlockA = tensorA.Slice(AscendC::Te::MakeCoord(0U, params.kOffset),
                                 AscendC::Te::MakeShape(params.m, params.realK));
    auto gmBlockB = tensorB.Slice(AscendC::Te::MakeCoord(params.kOffset, 0U),
                                  AscendC::Te::MakeShape(params.realK, params.n));
    PadAKTail(tensorAL1, gmBlockA);
    PadBKTail(tensorBL1, gmBlockB);
    AscendC::Te::Copy(copyGM2L1, tensorAL1, gmBlockA);
    AscendC::Te::Copy(copyGM2L1, tensorBL1, gmBlockB);
    if (params.kOffset == params.scaleWindowStart) {
        WaitFlag<HardEvent::MTE1_MTE2>(SCALE_BUFFER_EVENT_BASE + scaleBufferIndex);
        uint64_t scaleWindowOffset =
            Blaze::Gemm::CeilDiv(params.scaleWindowStart, MX_SCALE_GROUP_K) * MX_SCALE_ELEMENTS_PER_GROUP;
        auto gmBlockScaleA = tensorScaleA.Slice(
            AscendC::Te::MakeCoord(0U, scaleWindowOffset),
            AscendC::Te::MakeShape(params.m, params.scaleWindowLength));
        auto gmBlockScaleB = tensorScaleB.Slice(
            AscendC::Te::MakeCoord(scaleWindowOffset, 0U),
            AscendC::Te::MakeShape(params.scaleWindowLength, params.n));
        AscendC::Te::Copy(copyGM2L1, tensorScaleAL1, gmBlockScaleA);
        AscendC::Te::Copy(copyGM2L1, tensorScaleBL1, gmBlockScaleB);
    }

    SetFlag<HardEvent::MTE2_MTE1>(MTE2_TO_MTE1_EVENT_BASE + bufferIndex);
    WaitFlag<HardEvent::MTE2_MTE1>(MTE2_TO_MTE1_EVENT_BASE + bufferIndex);
    ProcessL0Tiles(tensorAL1, tensorBL1, tensorScaleAL1, tensorScaleBL1, tensorC, params,
                   debugThisTile);
    SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_EVENT_BASE + bufferIndex);
    ++l1BufferIndex_;
    bool isScaleWindowEnd = params.kOffset + params.realK >= params.scaleWindowStart + scaleKL1_ ||
                            params.kOffset + params.realK >= k_;
    if (isScaleWindowEnd) {
        SetFlag<HardEvent::MTE1_MTE2>(SCALE_BUFFER_EVENT_BASE + scaleBufferIndex);
        ++scaleL1BufferIndex_;
    }
}

MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_TEMPLATE_PARAMS
template <typename TensorAL1, typename TensorBL1, typename TensorScaleAL1, typename TensorScaleBL1,
          typename TensorC>
__aicore__ inline void MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_SPECIALIZATION::ProcessL0Tiles(
    const TensorAL1 &tensorAL1, const TensorBL1 &tensorBL1, const TensorScaleAL1 &tensorScaleAL1,
    const TensorScaleBL1 &tensorScaleBL1, const TensorC &tensorC, const TileParams &params,
    bool debugThisTile)
{
    for (uint64_t l0KOffset = 0U; l0KOffset < params.realK; l0KOffset += baseK_) {
        uint64_t realL0K = Blaze::Gemm::Min(baseK_, params.realK - l0KOffset);
        uint64_t alignedL0K = AlignK(realL0K);
        uint64_t realL0ScaleK =
            Blaze::Gemm::CeilDiv(realL0K, MX_SCALE_GROUP_K) * MX_SCALE_ELEMENTS_PER_GROUP;
        uint64_t l0ScaleKOffset = params.scaleKOffsetInWindow +
                                  Blaze::Gemm::CeilDiv(l0KOffset, MX_SCALE_GROUP_K) *
                                      MX_SCALE_ELEMENTS_PER_GROUP;
        uint64_t bufferIndex = l0BufferIndex_ & (BUFFER_COUNT - 1U);
        uint64_t l0Offset = bufferIndex * L0_BUFFER_BYTES;
        WaitFlag<HardEvent::M_MTE1>(MMAD_TO_MTE1_EVENT_BASE + bufferIndex);

        auto tensorAL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(l0Offset),
            MakeLayoutAL0{}(params.m, alignedL0K));
        auto tensorBL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(l0Offset),
            MakeLayoutBL0{}(alignedL0K, params.n));
        auto tensorScaleAL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0ScaleA, fp8_e8m0_t>(l0Offset >> 4U),
            MakeLayoutScaleAL0{}(params.m, realL0ScaleK));
        auto tensorScaleBL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0ScaleB, fp8_e8m0_t>(l0Offset >> 4U),
            MakeLayoutScaleBL0{}(realL0ScaleK, params.n));

        auto tensorBlockAL1 = tensorAL1.Slice(
            AscendC::Te::MakeCoord(0U, l0KOffset), AscendC::Te::MakeShape(params.m, alignedL0K));
        auto tensorBlockBL1 = tensorBL1.Slice(
            AscendC::Te::MakeCoord(l0KOffset, 0U), AscendC::Te::MakeShape(alignedL0K, params.n));
        auto tensorBlockScaleAL1 = tensorScaleAL1.Slice(
            AscendC::Te::MakeCoord(0U, l0ScaleKOffset),
            AscendC::Te::MakeShape(params.m, realL0ScaleK));
        auto tensorBlockScaleBL1 = tensorScaleBL1.Slice(
            AscendC::Te::MakeCoord(l0ScaleKOffset, 0U),
            AscendC::Te::MakeShape(realL0ScaleK, params.n));

        AscendC::Te::Copy(AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{}), tensorAL0, tensorBlockAL1);
        AscendC::Te::Copy(AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{}), tensorBL0, tensorBlockBL1);
        AscendC::Te::Copy(AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0ScaleA{}), tensorScaleAL0,
                          tensorBlockScaleAL1);
        AscendC::Te::Copy(AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0ScaleB{}), tensorScaleBL0,
                          tensorBlockScaleBL1);

        SetFlag<HardEvent::MTE1_M>(MTE1_TO_MMAD_EVENT_BASE + bufferIndex);
        WaitFlag<HardEvent::MTE1_M>(MTE1_TO_MMAD_EVENT_BASE + bufferIndex);

        auto tensorL0C = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(0U),
            AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<L0C_C0>>(
                params.m, params.n));
        bool isFirstK = params.kOffset == 0U && l0KOffset == 0U;
        bool isLastK = params.kOffset + params.realK >= k_ && l0KOffset + realL0K >= params.realK;
        AscendC::Te::MmadParams mmadParams;
        mmadParams.m = static_cast<uint16_t>(params.m);
        mmadParams.k = static_cast<uint16_t>(alignedL0K);
        mmadParams.n = static_cast<uint16_t>(params.n);
        mmadParams.unitFlag = isLastK ? Blaze::Gemm::FINAL_ACCUMULATION : Blaze::Gemm::NON_FINAL_ACCUMULATION;
        mmadParams.cmatrixInitVal = isFirstK;
        AscendC::Te::Mmad(
            AscendC::Te::MmadAtom<
                AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, Blaze::Gemm::Tile::MmadTraitMX>>{}
                .with(mmadParams),
            tensorL0C, tensorAL0, tensorBL0);

        SetFlag<HardEvent::M_MTE1>(MMAD_TO_MTE1_EVENT_BASE + bufferIndex);
        if (debugThisTile && (debugPartialAccumulations_ || isLastK)) {
            uint64_t completedK = params.kOffset + l0KOffset + realL0K;
            uint32_t partialMarker = debugMarkerBase_ + debugStep_;
            DebugDumpPartialL0C(tensorC, tensorL0C, completedK, partialMarker);
            ++debugStep_;
            if (isLastK) {
                DebugDumpDirectL0C2UB(tensorL0C, params.m, params.n, false,
                                      debugMarkerBase_ + debugStep_++);
                DebugDumpDirectL0C2UB(tensorL0C, params.m, params.n, true,
                                      debugMarkerBase_ + debugStep_++);
            }
        }
        ++l0BufferIndex_;
    }
}

MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_TEMPLATE_PARAMS
template <typename TensorC, typename TensorL0C>
__aicore__ inline void MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_SPECIALIZATION::DebugDumpPartialL0C(
    const TensorC &tensorC, const TensorL0C &tensorL0C, uint64_t completedK, uint32_t marker)
{
    if constexpr (AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<TensorC>,
                                          AscendC::Te::Location::GM>) {
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        uint32_t unitFlag = completedK >= k_ ? Blaze::Gemm::FINAL_ACCUMULATION :
                                               Blaze::Gemm::NON_FINAL_ACCUMULATION;
        AscendC::Te::Copy(
            AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{})
                .with(AscendC::Te::FixpipeParams(unitFlag)),
            tensorC, tensorL0C);
        SetFlag<HardEvent::FIX_S>(0);
        WaitFlag<HardEvent::FIX_S>(0);

        AscendC::GlobalTensor<uint16_t> debugGmBits;
        debugGmBits.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint16_t *>(tensorC.Data().Get()));
        set_cond(static_cast<unsigned long>(marker));
        DebugSetCondUint(static_cast<uint64_t>(AscendC::GetBlockIdx()));
        DebugSetCondUint(completedK);
        for (uint32_t valueIdx = 0U; valueIdx < DEBUG_VALUE_COUNT; ++valueIdx) {
            set_cond(static_cast<unsigned long>(debugGmBits.GetValue(valueIdx)));
        }

        SetFlag<HardEvent::FIX_M>(0);
        WaitFlag<HardEvent::FIX_M>(0);
    }
}

MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_TEMPLATE_PARAMS
template <typename TensorL0C>
__aicore__ inline void MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_SPECIALIZATION::DebugDumpDirectL0C2UB(
    const TensorL0C &tensorL0C, uint64_t m, uint64_t n, bool quantPreRnd, uint32_t marker)
{
    static_assert(AscendC::Std::is_same_v<CType, bfloat16_t>,
                  "direct arch92 L0C diagnostic only supports BF16 output");

    auto tensorCUb = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, CType>(DEBUG_UB_OFFSET),
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn,
                                        AscendC::Std::Int<AscendC::AuxGetC0Size<CType>()>>{}(m, n));
    auto copyParams = AscendC::Te::MakeL0COutCopyParams<decltype(tensorCUb), TensorL0C>(
        tensorCUb.Layout(), tensorL0C.Layout());

    // The active ascend910_96 C API exposes the 21-argument form. Calling it
    // directly bypasses the incompatible 22-argument Tensor API wrapper.
    SetFlag<HardEvent::M_FIX>(0);
    WaitFlag<HardEvent::M_FIX>(0);
    asc_copy_l0c2ub_sync(
        (__ubuf__ bfloat16_t *)tensorCUb.Data().Get(),
        (__cc__ float *)tensorL0C.Data().Get(),
        static_cast<uint16_t>(copyParams.nSize), static_cast<uint16_t>(copyParams.mSize),
        copyParams.dstStride, static_cast<uint16_t>(copyParams.srcStride),
        quantPreRnd, 0U, static_cast<uint8_t>(Blaze::Gemm::FINAL_ACCUMULATION),
        static_cast<uint64_t>(QuantMode_t::F322BF16), 0U, false, true,
        0U, 0U, false, 0U, false, false, false, false);

    SetFlag<HardEvent::FIX_S>(0);
    WaitFlag<HardEvent::FIX_S>(0);
    AscendC::LocalTensor<uint16_t> debugUbBits(
        AscendC::TPosition::VECCALC, DEBUG_UB_OFFSET, m * n);
    set_cond(static_cast<unsigned long>(marker));
    DebugSetCondUint(static_cast<uint64_t>(AscendC::GetBlockIdx()));
    DebugSetCondUint(static_cast<uint64_t>(quantPreRnd));
    DebugSetCondUint(m);
    DebugSetCondUint(n);
    DebugSetCondUint(copyParams.nSize);
    DebugSetCondUint(copyParams.mSize);
    DebugSetCondUint(copyParams.srcStride);
    DebugSetCondUint(copyParams.dstStride);
    for (uint32_t valueIdx = 0U; valueIdx < DEBUG_VALUE_COUNT; ++valueIdx) {
        set_cond(static_cast<unsigned long>(debugUbBits.GetValue(valueIdx)));
    }
    SetFlag<HardEvent::FIX_M>(0);
    WaitFlag<HardEvent::FIX_M>(0);
}

MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_TEMPLATE_PARAMS
template <typename TensorC>
__aicore__ inline void MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_SPECIALIZATION::CopyOut(
    const TensorC &tensorC, uint64_t m, uint64_t n)
{
    auto tensorL0C = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(0U),
        AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<L0C_C0>>(m, n));
    if constexpr (AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<TensorC>, AscendC::Te::Location::UB>) {
        AscendC::Te::Copy(
            AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2UB{})
                .with(AscendC::Te::FixpipeParams(Blaze::Gemm::FINAL_ACCUMULATION)),
            tensorC, tensorL0C);
    } else {
        AscendC::Te::Copy(
            AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{})
                .with(AscendC::Te::FixpipeParams(Blaze::Gemm::FINAL_ACCUMULATION)),
            tensorC, tensorL0C);
    }
}

#undef MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_SPECIALIZATION
#undef MEGA_MOE_ARCH92_BLOCK_MMAD_MX_FP8_TEMPLATE_PARAMS

} // namespace Block
} // namespace Gemm
} // namespace Blaze
