/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software and is licensed under the CANN Open Software License Agreement
 * Version 2.0. THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND.
 */

#ifndef SAGE_ATTENTION2_CATLASS_H_
#define SAGE_ATTENTION2_CATLASS_H_

#ifndef SAGE_ATTENTION2_PROFILE_CUT
#define SAGE_ATTENTION2_PROFILE_CUT 0
#endif

#ifndef SAGE_ATTENTION2_RESCALE_PROBE
#define SAGE_ATTENTION2_RESCALE_PROBE 0
#endif

#ifndef SAGE_ATTENTION2_SOFTMAX_PROBE
#define SAGE_ATTENTION2_SOFTMAX_PROBE 0
#endif

#ifndef SAGE_ATTENTION2_QK_PROFILE
#define SAGE_ATTENTION2_QK_PROFILE 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "sage_attention2_common.h"
#include "sage_attention2_mmad.h"
#include "sage_attention2_softmax.h"

namespace SageAttention2Kernel {

template <typename OutputT, uint32_t DPAD, uint32_t EVENT_ID, uint32_t ACCUM_MODE>
class DeferredVScaleRescale {
public:
    using ArchTag = Catlass::Arch::Ascend950;
    static constexpr uint32_t RESCALE_PROBE = SAGE_ATTENTION2_RESCALE_PROBE;
    static_assert(RESCALE_PROBE <= 1U, "unsupported SageAttention2 rescale probe");

    __aicore__ inline DeferredVScaleRescale(Catlass::Arch::Resource<ArchTag> &resource, uint32_t &ubOffset)
    {
        outputUb_ = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += (Q_TILE / 2U) * DPAD * sizeof(float);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID);
    }

    __aicore__ inline ~DeferredVScaleRescale()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID);
    }

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void operator()(TensorDst &outputGm, const AscendC::LocalTensor<float> &expMax,
        const AscendC::LocalTensor<float> &sum, TensorSrc &current, const AscendC::LocalTensor<float> &vScale,
        bool first, bool last, uint32_t kvBlock, uint64_t freeFlag)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID);
        const uint32_t rows = tla::get<0>(current.shape());
        const uint32_t cols = tla::get<1>(current.shape());
        constexpr uint32_t VL = 64U;
        const uint16_t loopsBeforeTail = AscendC::CeilDivision(cols, VL) - 1U;
        uint32_t tailCols = (cols - 1U) % VL + 1U;
        auto runningAddr = reinterpret_cast<__ubuf__ float *>(outputUb_.GetPhyAddr());
        auto currentAddr = reinterpret_cast<__ubuf__ float *>(current.data().GetPhyAddr());
        auto expAddr = reinterpret_cast<__ubuf__ float *>(expMax.GetPhyAddr());
        auto sumAddr = reinterpret_cast<__ubuf__ float *>(sum.GetPhyAddr());
        auto scaleAddr = reinterpret_cast<__ubuf__ float *>(vScale.GetPhyAddr());

        // Host tiling fixes the two-level accumulation interval to four KV tiles. Keep the test compile-time
        // visible so the hot path uses one mask instead of a runtime integer division.
        const bool flush = !last && (kvBlock & (FP16_FLUSH_INTERVAL - 1U)) == FP16_FLUSH_INTERVAL - 1U;
        // The fast path directs the first PV result into outputUb_ itself. Later blocks still arrive through
        // the ping-pong PV buffers and are folded into this running accumulator.
        if (!first) {
            if (!last) {
                if constexpr (ACCUM_MODE == static_cast<uint32_t>(AccumMode::FP32_FP16)) {
                    if (flush) {
                        Update<true>(runningAddr, currentAddr, expAddr,
                            static_cast<uint16_t>(rows), loopsBeforeTail, tailCols);
                    } else {
                        Update<false>(runningAddr, currentAddr, expAddr,
                            static_cast<uint16_t>(rows), loopsBeforeTail, tailCols);
                    }
                } else {
                    Update<false>(runningAddr, currentAddr, expAddr,
                        static_cast<uint16_t>(rows), loopsBeforeTail, tailCols);
                }
            } else if constexpr (RESCALE_PROBE == 0U) {
                if constexpr (AscendC::IsSameType<OutputT, half>::value) {
                    UpdateLastScaleCast(reinterpret_cast<__ubuf__ OutputT *>(runningAddr), runningAddr,
                        currentAddr, expAddr, sumAddr, scaleAddr, static_cast<uint16_t>(rows));
                } else {
                    UpdateLastScale(runningAddr, currentAddr, expAddr, sumAddr, scaleAddr,
                        static_cast<uint16_t>(rows), loopsBeforeTail, tailCols);
                }
            }
        }
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_V>(freeFlag);

        // Probe 1 retains every non-final online update while skipping only final combine/normalize/copy.
        // It is compile-time only and leaves the default full-output path unchanged.
        if constexpr (RESCALE_PROBE == 0U) {
            if (first && last) {
                if constexpr (AscendC::IsSameType<OutputT, half>::value) {
                    FirstLastScaleCast(reinterpret_cast<__ubuf__ OutputT *>(runningAddr), currentAddr,
                        sumAddr, scaleAddr, static_cast<uint16_t>(rows));
                } else {
                    FirstLastScale(runningAddr, currentAddr, sumAddr, scaleAddr,
                        static_cast<uint16_t>(rows), loopsBeforeTail, tailCols);
                }
            }
            if (last) {
                AscendC::LocalTensor<OutputT> output16;
                output16.SetAddr(outputUb_.address_);
                if constexpr (!AscendC::IsSameType<OutputT, half>::value) {
                    AscendC::Cast(output16, outputUb_, AscendC::RoundMode::CAST_ROUND, rows * DPAD);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID);
                auto layout = tla::MakeLayout(tla::MakeShape(rows, cols), tla::MakeStride(DPAD, tla::Int<1>{}));
                auto outputTensor = tla::MakeTensor(output16, layout, Catlass::Arch::PositionUB{});
                using CopyOut = Catlass::Epilogue::Tile::CopyUb2GmTla<ArchTag, decltype(outputTensor), TensorDst>;
                CopyOut copyOut;
                copyOut(outputGm, outputTensor);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID);
    }

private:
    template <bool FLUSH_FP16>
    __simd_callee__ static inline void UpdateD128(__ubuf__ float *running, __ubuf__ float *current,
        __ubuf__ float *expMax, uint16_t rows)
    {
        using namespace AscendC::MicroAPI;
        static constexpr CastTrait FP32_TO_FP16 = {
            RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr CastTrait FP16_TO_FP32 = {
            RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_NONE};
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        for (uint16_t r = 0U; r < rows; ++r) {
            RegTensor<float> expReg;
            RegTensor<float> old0;
            RegTensor<float> old1;
            RegTensor<float> cur0;
            RegTensor<float> cur1;
            RegTensor<float> out0;
            RegTensor<float> out1;
            const uint32_t offset = static_cast<uint32_t>(r) * DPAD;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(expReg, expMax + r);
            LoadAlign(old0, running + offset);
            LoadAlign(cur0, current + offset);
            LoadAlign(old1, running + offset + 64U);
            LoadAlign(cur1, current + offset + 64U);
            out0 = cur0;
            out1 = cur1;
            Mula(out0, old0, expReg, full);
            Mula(out1, old1, expReg, full);
            if constexpr (FLUSH_FP16) {
                RegTensor<half> rounded16;
                RegTensor<float> rounded32;
                Cast<half, float, FP32_TO_FP16>(rounded16, out0, full);
                Cast<float, half, FP16_TO_FP32>(rounded32, rounded16, full);
                out0 = rounded32;
                Cast<half, float, FP32_TO_FP16>(rounded16, out1, full);
                Cast<float, half, FP16_TO_FP32>(rounded32, rounded16, full);
                out1 = rounded32;
            }
            StoreAlign<float, StoreDist::DIST_NORM_B32>(running + offset, out0, full);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(running + offset + 64U, out1, full);
        }
    }

    template <bool FLUSH_FP16>
    __simd_vf__ static inline void Update(__ubuf__ float *running, __ubuf__ float *current,
        __ubuf__ float *expMax, uint16_t rows, uint16_t loopsBeforeTail, uint32_t tailCols)
    {
        if constexpr (DPAD == 128U) {
            UpdateD128<FLUSH_FP16>(running, current, expMax, rows);
            return;
        }
        using namespace AscendC::MicroAPI;
        static constexpr CastTrait FP32_TO_FP16 = {
            RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr CastTrait FP16_TO_FP32 = {
            RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_NONE};
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        MaskReg tail = UpdateMask<float>(tailCols);
        for (uint16_t r = 0U; r < rows; ++r) {
            RegTensor<float> expReg;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(expReg, expMax + r);
            for (uint16_t j = 0U; j < loopsBeforeTail; ++j) {
                RegTensor<float> oldReg;
                RegTensor<float> curReg;
                RegTensor<float> outReg;
                const uint32_t offset = static_cast<uint32_t>(r) * DPAD + j * 64U;
                LoadAlign(oldReg, running + offset);
                LoadAlign(curReg, current + offset);
                outReg = curReg;
                Mula(outReg, oldReg, expReg, full);
                if constexpr (FLUSH_FP16) {
                    RegTensor<half> rounded16;
                    RegTensor<float> rounded32;
                    Cast<half, float, FP32_TO_FP16>(rounded16, outReg, full);
                    Cast<float, half, FP16_TO_FP32>(rounded32, rounded16, full);
                    outReg = rounded32;
                }
                StoreAlign<float, StoreDist::DIST_NORM_B32>(running + offset, outReg, full);
            }
            const uint32_t offset = static_cast<uint32_t>(r) * DPAD + loopsBeforeTail * 64U;
            RegTensor<float> oldReg;
            RegTensor<float> curReg;
            RegTensor<float> outReg;
            LoadAlign(oldReg, running + offset);
            LoadAlign(curReg, current + offset);
            outReg = curReg;
            Mula(outReg, oldReg, expReg, tail);
            if constexpr (FLUSH_FP16) {
                RegTensor<half> rounded16;
                RegTensor<float> rounded32;
                Cast<half, float, FP32_TO_FP16>(rounded16, outReg, tail);
                Cast<float, half, FP16_TO_FP32>(rounded32, rounded16, tail);
                outReg = rounded32;
            }
            StoreAlign<float, StoreDist::DIST_NORM_B32>(running + offset, outReg, tail);
        }
    }

    __simd_vf__ static inline void UpdateLastScale(__ubuf__ float *running, __ubuf__ float *current,
        __ubuf__ float *expMax, __ubuf__ float *sum, __ubuf__ float *scale, uint16_t rows,
        uint16_t loopsBeforeTail, uint32_t tailCols)
    {
        using namespace AscendC::MicroAPI;
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        MaskReg tail = UpdateMask<float>(tailCols);
        RegTensor<float> scale0;
        RegTensor<float> scale1;
        LoadAlign(scale0, scale);
        if constexpr (DPAD > 64U) {
            LoadAlign(scale1, scale + 64U);
        }
        for (uint16_t r = 0U; r < rows; ++r) {
            RegTensor<float> expReg;
            RegTensor<float> sumReg;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(expReg, expMax + r);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(sumReg, sum + r);
            for (uint16_t j = 0U; j < loopsBeforeTail; ++j) {
                RegTensor<float> oldReg;
                RegTensor<float> curReg;
                RegTensor<float> outReg;
                const uint32_t offset = static_cast<uint32_t>(r) * DPAD + j * 64U;
                LoadAlign(oldReg, running + offset);
                LoadAlign(curReg, current + offset);
                Mul(outReg, oldReg, expReg, full);
                Add(outReg, outReg, curReg, full);
                Div(outReg, outReg, sumReg, full);
                Mul(outReg, outReg, scale0, full);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(running + offset, outReg, full);
            }
            const uint32_t offset = static_cast<uint32_t>(r) * DPAD + loopsBeforeTail * 64U;
            RegTensor<float> oldReg;
            RegTensor<float> curReg;
            RegTensor<float> outReg;
            LoadAlign(oldReg, running + offset);
            LoadAlign(curReg, current + offset);
            Mul(outReg, oldReg, expReg, tail);
            Add(outReg, outReg, curReg, tail);
            Div(outReg, outReg, sumReg, tail);
            if constexpr (DPAD > 64U) {
                Mul(outReg, outReg, scale1, tail);
            } else {
                Mul(outReg, outReg, scale0, tail);
            }
            StoreAlign<float, StoreDist::DIST_NORM_B32>(running + offset, outReg, tail);
        }
    }

    __simd_vf__ static inline void FirstLastScale(__ubuf__ float *running, __ubuf__ float *current,
        __ubuf__ float *sum, __ubuf__ float *scale, uint16_t rows, uint16_t loopsBeforeTail, uint32_t tailCols)
    {
        using namespace AscendC::MicroAPI;
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        MaskReg tail = UpdateMask<float>(tailCols);
        RegTensor<float> scale0;
        RegTensor<float> scale1;
        LoadAlign(scale0, scale);
        if constexpr (DPAD > 64U) {
            LoadAlign(scale1, scale + 64U);
        }
        for (uint16_t r = 0U; r < rows; ++r) {
            RegTensor<float> sumReg;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(sumReg, sum + r);
            for (uint16_t j = 0U; j < loopsBeforeTail; ++j) {
                RegTensor<float> curReg;
                RegTensor<float> outReg;
                const uint32_t offset = static_cast<uint32_t>(r) * DPAD + j * 64U;
                LoadAlign(curReg, current + offset);
                Div(outReg, curReg, sumReg, full);
                Mul(outReg, outReg, scale0, full);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(running + offset, outReg, full);
            }
            const uint32_t offset = static_cast<uint32_t>(r) * DPAD + loopsBeforeTail * 64U;
            RegTensor<float> curReg;
            RegTensor<float> outReg;
            LoadAlign(curReg, current + offset);
            Div(outReg, curReg, sumReg, tail);
            if constexpr (DPAD > 64U) {
                Mul(outReg, outReg, scale1, tail);
            } else {
                Mul(outReg, outReg, scale0, tail);
            }
            StoreAlign<float, StoreDist::DIST_NORM_B32>(running + offset, outReg, tail);
        }
    }

    __simd_vf__ static inline void UpdateLastScaleCast(__ubuf__ OutputT *output, __ubuf__ float *running,
        __ubuf__ float *current, __ubuf__ float *expMax, __ubuf__ float *sum, __ubuf__ float *scale,
        uint16_t rows)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait CAST_ZERO = {
            RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        MaskReg full16 = CreateMask<OutputT, MaskPattern::ALL>();
        constexpr uint16_t COL_BLOCKS = DPAD / 64U;
        RegTensor<float> scaleRegs[COL_BLOCKS];
        for (uint16_t j = 0U; j < COL_BLOCKS; ++j) {
            LoadAlign(scaleRegs[j], scale + static_cast<uint32_t>(j) * 64U);
        }
        for (uint16_t r = 0U; r < rows; ++r) {
            RegTensor<float> expReg;
            RegTensor<float> sumReg;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(expReg, expMax + r);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(sumReg, sum + r);
            for (uint16_t j = 0U; j < COL_BLOCKS; ++j) {
                RegTensor<float> oldReg;
                RegTensor<float> curReg;
                RegTensor<float> outReg;
                RegTensor<OutputT> outputReg;
                const uint32_t offset = static_cast<uint32_t>(r) * DPAD + static_cast<uint32_t>(j) * 64U;
                LoadAlign(oldReg, running + offset);
                LoadAlign(curReg, current + offset);
                Mul(outReg, oldReg, expReg, full);
                Add(outReg, outReg, curReg, full);
                Div(outReg, outReg, sumReg, full);
                Mul(outReg, outReg, scaleRegs[j], full);
                Cast<OutputT, float, CAST_ZERO>(outputReg, outReg, full16);
                StoreAlign<OutputT, StoreDist::DIST_PACK_B32>(output + offset, outputReg, full);
            }
        }
    }

    __simd_vf__ static inline void FirstLastScaleCast(__ubuf__ OutputT *output, __ubuf__ float *current,
        __ubuf__ float *sum, __ubuf__ float *scale, uint16_t rows)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait CAST_ZERO = {
            RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        MaskReg full16 = CreateMask<OutputT, MaskPattern::ALL>();
        constexpr uint16_t COL_BLOCKS = DPAD / 64U;
        RegTensor<float> scaleRegs[COL_BLOCKS];
        for (uint16_t j = 0U; j < COL_BLOCKS; ++j) {
            LoadAlign(scaleRegs[j], scale + static_cast<uint32_t>(j) * 64U);
        }
        for (uint16_t r = 0U; r < rows; ++r) {
            RegTensor<float> sumReg;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(sumReg, sum + r);
            for (uint16_t j = 0U; j < COL_BLOCKS; ++j) {
                RegTensor<float> curReg;
                RegTensor<float> outReg;
                RegTensor<OutputT> outputReg;
                const uint32_t offset = static_cast<uint32_t>(r) * DPAD + static_cast<uint32_t>(j) * 64U;
                LoadAlign(curReg, current + offset);
                Div(outReg, curReg, sumReg, full);
                Mul(outReg, outReg, scaleRegs[j], full);
                Cast<OutputT, float, CAST_ZERO>(outputReg, outReg, full16);
                StoreAlign<OutputT, StoreDist::DIST_PACK_B32>(output + offset, outputReg, full);
            }
        }
    }

    AscendC::LocalTensor<float> outputUb_;
    static constexpr uint32_t SYNC_MODE = 4U;
    static constexpr uint32_t FP16_FLUSH_INTERVAL = 4U;
};

template <typename InputT, uint32_t DPAD, bool LAYOUT_NHD, bool IS_CAUSAL, bool PER_THREAD, uint32_t ACCUM_MODE,
    bool SMOOTH_K, bool SMOOTH_V, bool RETURN_LSE>
class CatlassAttention {
public:
    // Performance-probe-only cumulative cuts. Zero preserves the delivery path; 1 drains QK, 2 executes the
    // score dequant/max prefix, 3 drains P after softmax, and 4 drains PV before output rescale. Every produced
    // cross-core token still has a consumer.
    static constexpr uint32_t PROFILE_CUT = SAGE_ATTENTION2_PROFILE_CUT;
    static constexpr uint32_t SOFTMAX_PROBE = SAGE_ATTENTION2_SOFTMAX_PROBE;
    // QK modes 1-3 are cumulative through MMAD/MTE1/MTE2. Modes 4, 5, and 6 retain the MTE2-only token
    // protocol while copying only K, only Q, or neither operand. Mode 7 is protocol-only with half as many
    // synthetic KV work items, used to test whether the handshake cost scales with item count.
    static constexpr uint32_t QK_PROFILE = SAGE_ATTENTION2_QK_PROFILE;
    static_assert(PROFILE_CUT <= 4U, "unsupported SageAttention2 profile cut");
    static_assert(SOFTMAX_PROBE <= 2U, "unsupported SageAttention2 softmax probe");
    static_assert(QK_PROFILE <= 8U, "unsupported SageAttention2 QK profile mode");
    static_assert(SOFTMAX_PROBE == 0U || PROFILE_CUT == 3U,
        "SageAttention2 softmax probe is valid only with profile cut 3");
    static_assert(QK_PROFILE == 0U || PROFILE_CUT == 1U,
        "SageAttention2 QK profile modes are valid only with profile cut 1");

    using ArchTag = Catlass::Arch::Ascend950;
    using ElementQ = int8_t;
    using ElementK = int8_t;
    using ElementQKAccum = half;
    using ElementScore = half;
    using ElementS = float;
    using ElementP = float8_e4m3_t;
    using ElementV = float8_e4m3_t;
    using ElementOTmp = float;
    using LayoutTagQ = Catlass::layout::ColumnMajor;
    using LayoutTagK = Catlass::layout::RowMajor;
    using LayoutTagS = Catlass::layout::RowMajor;
    using LayoutTagP = Catlass::layout::zN;
    using LayoutTagV = Catlass::layout::RowMajor;
    using LayoutTagOTmp = Catlass::layout::RowMajor;
    using LayoutTagMask = Catlass::layout::RowMajor;
    using L1TileShape = tla::Shape<tla::_128, tla::_256, tla::_128>;
    using L0TileShape = tla::Shape<tla::_128, tla::_128, tla::_128>;
    using L0TileShapePV = tla::Shape<tla::_128, tla::_256, tla::_128>;
    using L1TileShapeQK = tla::Shape<tla::_256, tla::_128, tla::_128>;
    using L0TileShapeQK = tla::Shape<tla::_128, tla::_128, tla::_128>;

    using DispatchPolicyQK = Catlass::Gemm::MmadFAIQK<ArchTag, false>;
    using TileCopyQK = Catlass::Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementK, LayoutTagK, ElementQ,
        LayoutTagQ, ElementQKAccum, LayoutTagS, void, Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT, false,
        Catlass::Gemm::Tile::ScaleGranularity::PER_TENSOR>;
    using TileMmadQK = Catlass::Gemm::Tile::TileMmadTla<ArchTag, ElementK, typename TileCopyQK::LayoutTagL1A>;
    using BlockMmadQK = SageMmadQKDn<L1TileShapeQK, L0TileShapeQK, ElementK, ElementQ, ElementQKAccum,
        TileCopyQK, TileMmadQK>;

    using DispatchPolicySoftmax = Catlass::Epilogue::EpilogueAscend950FASoftmax<false, false>;
    using PType = Catlass::Gemm::GemmType<ElementP, LayoutTagP>;
    using SType = Catlass::Gemm::GemmType<ElementS, LayoutTagS>;
    using MaskType = Catlass::Gemm::GemmType<uint8_t, LayoutTagMask>;
    using OnlineSoftmax = Catlass::Epilogue::Block::BlockEpilogue<
        DispatchPolicySoftmax, L1TileShape, PType, SType, MaskType>;

    using DispatchPolicyPV = Catlass::Gemm::MmadFAIPV<ArchTag, false>;
    using TileCopyPV = Catlass::Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutTagP, ElementV,
        LayoutTagV, ElementOTmp, LayoutTagV, void, Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using TileMmadPV = Catlass::Gemm::Tile::TileMmadTla<ArchTag, ElementP, typename TileCopyPV::LayoutTagL1A>;
    using BlockMmadPV = SageMmadPV<L1TileShape, L0TileShapePV, ElementP, ElementV, ElementOTmp,
        TileCopyPV, TileMmadPV, true>;

    static_assert(BlockMmadQK::BLOCK_L0C_SIZE + BlockMmadPV::BLOCK_L0C_SIZE <= ArchTag::L0C_SIZE,
        "combined QK/PV L0C allocation exceeds Ascend950 capacity");
    static_assert(3U * Q_TILE * KV_TILE * sizeof(ElementP) +
        BlockMmadQK::BLOCK_L1_SIZE + BlockMmadPV::BLOCK_L1_SIZE <= ArchTag::L1_SIZE,
        "combined P/QK/PV L1 allocation exceeds Ascend950 capacity");

    using DispatchPolicyRescaleO = Catlass::Epilogue::EpilogueAscend950FARescaleO;
    using OType = Catlass::Gemm::GemmType<InputT, LayoutTagOTmp>;
    using OTmpType = Catlass::Gemm::GemmType<ElementOTmp, LayoutTagOTmp>;
    using RescaleD = std::conditional_t<DPAD == 128U, tla::_128, tla::_64>;
    using RescaleTileShape = tla::Shape<tla::_64, tla::_128, RescaleD>;
    using OnlineRescale = Catlass::Epilogue::Block::BlockEpilogue<
        DispatchPolicyRescaleO, RescaleTileShape, OType, OTmpType>;

    __aicore__ inline Catlass::Arch::Resource<ArchTag> &GetResource()
    {
        return resource_;
    }

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR attentionOut, GM_ADDR softmaxLse, GM_ADDR userWorkspace,
        const SageAttention2TilingData *tiling)
    {
        tiling_ = tiling;
        qOriginal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(query));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(attentionOut));
        outputTyped_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(attentionOut));
        if constexpr (RETURN_LSE) {
            lse_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(softmaxLse));
        }
        q8_.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementQ *>(userWorkspace + tiling->workspace.q8Offset));
        k8_.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementK *>(userWorkspace + tiling->workspace.k8Offset));
        v8_.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementV *>(userWorkspace + tiling->workspace.v8Offset));
        qScale_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(userWorkspace + tiling->workspace.qScaleOffset));
        kScale_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(userWorkspace + tiling->workspace.kScaleOffset));
        vScale_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(userWorkspace + tiling->workspace.vScaleOffset));
        if constexpr (SMOOTH_K) {
            kMean_.SetGlobalBuffer(
                reinterpret_cast<__gm__ uint16_t *>(userWorkspace + tiling->workspace.kMeanOffset));
        }
        if constexpr (SMOOTH_V) {
            vMean_.SetGlobalBuffer(
                reinterpret_cast<__gm__ float *>(userWorkspace + tiling->workspace.vMeanOffset));
        }
        batch_ = tiling->shape.batch;
        qHeads_ = tiling->shape.qHeads;
        kvHeads_ = tiling->shape.kvHeads;
        qSeq_ = tiling->shape.qSeq;
        kvSeq_ = tiling->shape.kvSeq;
        kvSeqPad_ = tiling->shape.kvSeqPad;
        dim_ = tiling->shape.headDim;
        qBlocks_ = tiling->shape.qBlocks;
        kvBlocks_ = tiling->shape.kvBlocks;
        gqaRatio_ = tiling->shape.gqaRatio;
        coreNum_ = tiling->cores.coreNum;
        totalTasks_ = tiling->cores.totalAttentionTasks;
        qGroups_ = tiling->cores.qScaleGroupsPerBlock;
        kGroups_ = tiling->cores.kScaleGroupsPerBlock;
        smScale_ = tiling->attrs.smScale;
        flushInterval_ = tiling->attrs.accumFlushInterval;
        if ASCEND_IS_AIC {
            blockIdx_ = AscendC::GetBlockIdx();
        } else {
            blockIdx_ = AscendC::GetBlockIdx() >> 1U;
        }
        subBlockIdx_ = AscendC::GetSubBlockIdx();

        uint32_t ubOffset = 0U;
        for (uint32_t i = 0U; i < PIPELINE_SLOTS_2; ++i) {
            qkAccumUb_[i] = resource_.ubBuf.template GetBufferByByte<ElementQKAccum>(ubOffset);
            ubOffset += HALF_QK_ACCUM_BYTES;
        }
        scoreUb_[0] = resource_.ubBuf.template GetBufferByByte<ElementScore>(ubOffset);
        ubOffset += HALF_SCORE_BYTES;
        for (uint32_t i = 0U; i < PIPELINE_SLOTS_2; ++i) {
            pvUb_[i] = resource_.ubBuf.template GetBufferByByte<ElementOTmp>(ubOffset);
            ubOffset += HALF_OUTPUT_BYTES;
        }
        for (uint32_t i = 0U; i < Q_SUBTILES; ++i) {
            sumUb_[i] = resource_.ubBuf.template GetBufferByByte<ElementS>(ubOffset);
            ubOffset += HALF_ROW_BYTES;
        }
        for (uint32_t i = 0U; i < PIPELINE_SLOTS_3; ++i) {
            expMaxUb_[i] = resource_.ubBuf.template GetBufferByByte<ElementS>(ubOffset);
            ubOffset += HALF_ROW_BYTES;
        }
        for (uint32_t i = 0U; i < Q_SUBTILES; ++i) {
            maxUb_[i] = resource_.ubBuf.template GetBufferByByte<ElementS>(ubOffset);
            ubOffset += HALF_ROW_BYTES;
        }
        vScaleUb_ = resource_.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += DPAD * sizeof(float);
        kScaleUb_ = resource_.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += MAX_K_SCALE_ELEMENTS * sizeof(float);
        qScaleUb_ = resource_.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += 32U * sizeof(float);
        // OnlineRescale is constructed after OnlineSoftmax in Process. Alias its two output accumulators for
        // the generic fallback path so both implementations use the same storage.
        for (uint32_t i = 0U; i < Q_SUBTILES; ++i) {
            runningOUb_[i] = resource_.ubBuf.template GetBufferByByte<ElementOTmp>(
                ubOffset + SOFTMAX_SCRATCH_BYTES + i * HALF_OUTPUT_BYTES);
        }

        for (uint32_t i = 0U; i < PIPELINE_SLOTS_3; ++i) {
            pL1_[i] = resource_.l1Buf.template GetBufferByByte<ElementP>(i * FULL_P_BYTES);
        }
        l1Offset_ = PIPELINE_SLOTS_3 * FULL_P_BYTES;
        ubOffset_ = ubOffset;

        // Both vector sub-blocks publish the initial buffer-free tokens expected by the CATLASS QK/PV blocks.
        for (uint32_t i = 0U; i < PIPELINE_SLOTS_2; ++i) {
            AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_V>(MM1_FREE_FLAG[i]);
            AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_V>(MM2_FREE_FLAG[i]);
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t l0COffset = 0U;
        BlockMmadQK qk(resource_, l1Offset_, l0COffset);
        BlockMmadPV pv(resource_, l1Offset_, l0COffset);
        // Scores are explicitly dequantized before this epilogue, hence its score scale is one. The epilogue
        // applies the same P scale to both the row sum and P, so it cancels during the final normalization.
        FusedDnSoftmax<PER_THREAD, IS_CAUSAL> softmax(resource_, ubOffset_);
        DeferredVScaleRescale<InputT, DPAD, 3U, ACCUM_MODE> rescale0(resource_, ubOffset_);

        auto qLayout = tla::MakeLayout<ElementQ, LayoutTagQ>(qHeads_ * DPAD, batch_ * qSeq_);
        auto qTensor = tla::MakeTensor(q8_, qLayout, Catlass::Arch::PositionGM{});
        auto kLayout = tla::MakeLayout<ElementK, LayoutTagK>(batch_ * kvSeq_, kvHeads_ * DPAD);
        auto kTensor = tla::MakeTensor(k8_, kLayout, Catlass::Arch::PositionGM{});
        auto vLayout = tla::MakeLayout<ElementV, LayoutTagV>(batch_ * kvSeqPad_, kvHeads_ * DPAD);
        auto vTensor = tla::MakeTensor(v8_, vLayout, Catlass::Arch::PositionGM{});
        AscendC::GlobalTensor<int32_t> unusedTable;
        AscendC::GlobalTensor<uint8_t> unusedMask;
        auto unusedMaskLayout = tla::MakeLayout<uint8_t, LayoutTagMask>(1, 1);
        auto unusedMaskTensor = tla::MakeTensor(unusedMask, unusedMaskLayout, Catlass::Arch::PositionGM{});

        // Profile mode 8 keeps the protocol-only QK body but samples one half of the attention tasks.
        // It is a diagnostic upper-bound probe, never a functional kernel: comparing it with mode 6
        // distinguishes a per-task phase floor from a per-KV-work-item floor before investing in a
        // cross-qBlock drain/reset redesign. The production mode keeps the original scheduler exactly.
        const uint32_t taskStride = QK_PROFILE == 8U ? coreNum_ * 2U : coreNum_;
        for (uint32_t task = blockIdx_; task < totalTasks_; task += taskStride) {
            uint32_t linear = task;
            const uint32_t qBlock = linear % qBlocks_;
            linear /= qBlocks_;
            const uint32_t hq = linear % qHeads_;
            const uint32_t b = linear / qHeads_;
            const uint32_t hkv = hq / gqaRatio_;
            if ASCEND_IS_AIV {
                if constexpr (PROFILE_CUT == 0U) {
                    const uint64_t vScaleBase = (static_cast<uint64_t>(b) * kvHeads_ + hkv) * DPAD;
                    AscendC::DataCopy(vScaleUb_, vScale_[vScaleBase], DPAD);
                }
                if constexpr (PROFILE_CUT == 0U || PROFILE_CUT >= 2U) {
                    if constexpr (PER_THREAD) {
                        AscendC::DataCopy(qScaleUb_, qScale_[QScaleOffset(b, hq, qBlock, 0U)], qGroups_);
                        const uint32_t kScaleElements =
                            ((kvSeq_ + K_QUANT_BLOCK - 1U) / K_QUANT_BLOCK) * kGroups_;
                        AscendC::DataCopy(kScaleUb_, kScale_[KScaleOffset(b, hkv, 0U, 0U)], kScaleElements);
                    } else {
                        // MTE copies whole 32-byte blocks. The compact per-warp scale arrays contain only four
                        // Q floats and can contain fewer than eight K floats, so include the workspace padding
                        // up to the next block instead of issuing a sub-block transfer.
                        constexpr uint32_t SCALE_COPY_ELEMENTS = 32U / sizeof(float);
                        AscendC::DataCopy(
                            qScaleUb_, qScale_[QScaleOffset(b, hq, qBlock, 0U)], SCALE_COPY_ELEMENTS);
                        const uint32_t kBlocks = (kvSeq_ + K_QUANT_BLOCK - 1U) / K_QUANT_BLOCK;
                        const uint32_t kCopyElements =
                            AscendC::CeilDivision(kBlocks, SCALE_COPY_ELEMENTS) * SCALE_COPY_ELEMENTS;
                        AscendC::DataCopy(
                            kScaleUb_, kScale_[KScaleOffset(b, hkv, 0U, 0U)], kCopyElements);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::Muls(qScaleUb_, qScaleUb_, smScale_ * QK_FIXPIPE_COMPENSATION, qGroups_);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }

            // One 128-row attention task matches one quantization block. The two AIV sub-blocks process
            // 64 rows each while QK and PV retain independent two-slot pipelines.
            static_assert(Q_SUBTILES == 1U, "the Q128 scheduler requires one quantization subtile");
            uint32_t subKvBlocks[Q_SUBTILES] = {};
            for (uint32_t qSub = 0U; qSub < Q_SUBTILES; ++qSub) {
                const uint32_t qStart = qBlock * Q_QUANT_BLOCK + qSub * Q_TILE;
                if (qStart < qSeq_) {
                    const uint32_t qReal = Min(Q_TILE, qSeq_ - qStart);
                    subKvBlocks[qSub] = kvBlocks_;
                    if constexpr (IS_CAUSAL) {
                        const uint32_t causalEnd = Min(kvSeq_, qStart + qReal);
                        subKvBlocks[qSub] = (causalEnd + KV_TILE - 1U) / KV_TILE;
                    }
                }
            }
            const uint32_t totalWork = QK_PROFILE == 7U ? (subKvBlocks[0] + 1U) / 2U : subKvBlocks[0];
            if ASCEND_IS_AIV {
                if constexpr (PROFILE_CUT == 0U || PROFILE_CUT >= 2U) {
                    const uint32_t qStart = qBlock * Q_QUANT_BLOCK;
                    const uint32_t qReal = Min(Q_TILE, qSeq_ - qStart);
                    const uint32_t firstHalf = (qReal + 1U) >> 1U;
                    const uint32_t rowOffset = subBlockIdx_ * firstHalf;
                    softmax.PrepareQScale(qScaleUb_, rowOffset);
                }
            }

            // Four-stage C1/V1/C2/V2 software pipeline. CATLASS exposes two result slots for each cube
            // matmul and three P/exp slots specifically for this schedule.
            constexpr uint32_t PIPELINE_DRAIN =
                (PROFILE_CUT == 1U || PROFILE_CUT == 2U) ? 1U : (PROFILE_CUT == 3U ? 2U : 3U);
            for (uint32_t stage = 0U; stage < totalWork + PIPELINE_DRAIN; ++stage) {
                if ASCEND_IS_AIC {
                    if (stage < totalWork) {
                        const uint32_t work = stage;
                        constexpr uint32_t qSub = 0U;
                        const uint32_t kvBlock = work;
                        const uint32_t taskKvBlocks = subKvBlocks[qSub];
                        const uint32_t qStart = qBlock * Q_QUANT_BLOCK + qSub * Q_TILE;
                        const uint32_t qReal = Min(Q_TILE, qSeq_ - qStart);
                        const uint32_t pipelineSlot2 = work & 1U;
                        const uint32_t kvStart = kvBlock * KV_TILE;
                        const uint32_t kvReal = Min(KV_TILE, kvSeq_ - kvStart);
                        const bool first = kvBlock == 0U;
                        const bool last = kvBlock + 1U == taskKvBlocks;
                        auto qkAccumLayout = tla::MakeLayout<ElementQKAccum, LayoutTagS>(KV_TILE, HALF_ROWS);
                        auto qkAccumTensor =
                            tla::MakeTensor(qkAccumUb_[pipelineSlot2], qkAccumLayout, Catlass::Arch::PositionUB{});
                        auto qTile = GetTile(qTensor, tla::MakeCoord(hq * DPAD, b * qSeq_ + qStart),
                            tla::MakeShape(DPAD, qReal));
                        auto kTile = GetTile(kTensor, tla::MakeCoord(b * kvSeq_ + kvStart, hkv * DPAD),
                            tla::MakeShape(kvReal, DPAD));
                        auto qkShape = tla::MakeShape(kvReal, qReal, DPAD);
                        qk(kTile, qTile, qkAccumTensor, unusedTable, qkShape, pipelineSlot2, 0, first, last,
                            QK_FIXPIPE_DEQ_BITS);
                        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(QK_READY_FLAG[pipelineSlot2]);
                        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16U + QK_READY_FLAG[pipelineSlot2]);
                    }
                    if constexpr (PROFILE_CUT == 0U || PROFILE_CUT >= 3U) {
                        if (stage >= 2U && stage - 2U < totalWork) {
                        const uint32_t work = stage - 2U;
                        constexpr uint32_t qSub = 0U;
                        const uint32_t kvBlock = work;
                        const uint32_t qStart = qBlock * Q_QUANT_BLOCK + qSub * Q_TILE;
                        const uint32_t qReal = Min(Q_TILE, qSeq_ - qStart);
                        const uint32_t slot2 = work & 1U;
                        const uint32_t slot3 = work % PIPELINE_SLOTS_3;
                        const uint32_t kvStart = kvBlock * KV_TILE;
                        const uint32_t kvReal = Min(KV_TILE, kvSeq_ - kvStart);
                        const bool first = kvBlock == 0U;
                        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(P_READY_FLAG[slot3]);
                        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16U + P_READY_FLAG[slot3]);
                            if constexpr (PROFILE_CUT == 0U || PROFILE_CUT >= 4U) {
                                auto pLayout = tla::MakeLayout<ElementP, LayoutTagP>(Q_TILE, KV_TILE);
                                auto pTensor = tla::MakeTensor(pL1_[slot3], pLayout, Catlass::Arch::PositionL1{});
                                auto vTile = GetTile(vTensor, tla::MakeCoord(b * kvSeqPad_ + kvStart, hkv * DPAD),
                                    tla::MakeShape(kvReal, DPAD));
                                auto pvLayout = tla::MakeLayout<ElementOTmp, LayoutTagOTmp>(qReal, DPAD);
                                auto pvDst = pvUb_[slot2];
                                if constexpr (!RETURN_LSE && !SMOOTH_V) {
                                    if (dim_ == DPAD && first) {
                                        pvDst = runningOUb_[qSub];
                                    }
                                }
                                auto pvTensor = tla::MakeTensor(pvDst, pvLayout, Catlass::Arch::PositionUB{});
                                auto pvShape = tla::MakeShape(qReal, DPAD, kvReal);
                                pv(pTensor, vTile, pvTensor, unusedTable, pvShape, slot2, 0);
                                AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(PV_READY_FLAG[slot2]);
                                AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16U + PV_READY_FLAG[slot2]);
                            }
                        }
                    }
                }

                if ASCEND_IS_AIV {
                    if (stage >= 1U && stage - 1U < totalWork) {
                        const uint32_t work = stage - 1U;
                        constexpr uint32_t qSub = 0U;
                        const uint32_t kvBlock = work;
                        const uint32_t qStart = qBlock * Q_QUANT_BLOCK + qSub * Q_TILE;
                        const uint32_t qReal = Min(Q_TILE, qSeq_ - qStart);
                        const uint32_t firstHalf = (qReal + 1U) >> 1U;
                        const uint32_t halfRows = subBlockIdx_ == 0U ? firstHalf : qReal - firstHalf;
                        const uint32_t rowOffset = subBlockIdx_ * firstHalf;
                        const uint32_t slot2 = work & 1U;
                        const uint32_t slot3 = work % PIPELINE_SLOTS_3;
                        const uint32_t kvStart = kvBlock * KV_TILE;
                        const uint32_t kvReal = Min(KV_TILE, kvSeq_ - kvStart);
                        const int32_t validBase = static_cast<int32_t>(qStart + rowOffset + 1U) -
                            static_cast<int32_t>(kvStart);
                        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(QK_READY_FLAG[slot2]);
                        if constexpr (PROFILE_CUT == 0U || PROFILE_CUT >= 3U) {
                            const uint32_t localKScaleOffset = (kvStart / K_QUANT_BLOCK) * kGroups_;
                            auto pLayout = tla::MakeLayout<ElementP, LayoutTagP>(Q_TILE, KV_TILE);
                            auto pTensor = tla::MakeTensor(pL1_[slot3], pLayout, Catlass::Arch::PositionL1{});
                            softmax(pTensor, sumUb_[qSub], maxUb_[qSub], expMaxUb_[slot3],
                                qkAccumUb_[slot2], scoreUb_[0], kScaleUb_[localKScaleOffset], kvReal, validBase,
                                kvBlock != 0U, slot2, slot3, halfRows, rowOffset, MM1_FREE_FLAG[slot2],
                                P_READY_FLAG[slot3]);
                        } else if constexpr (PROFILE_CUT == 2U) {
                            const uint32_t localKScaleOffset = (kvStart / K_QUANT_BLOCK) * kGroups_;
                            softmax.ProfileMaxOnly(maxUb_[qSub], expMaxUb_[slot3], qkAccumUb_[slot2],
                                scoreUb_[0], kScaleUb_[localKScaleOffset], kvReal, validBase, kvBlock != 0U,
                                MM1_FREE_FLAG[slot2]);
                        } else {
                            AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_V>(MM1_FREE_FLAG[slot2]);
                        }
                    }
                    if constexpr (PROFILE_CUT == 0U || PROFILE_CUT >= 4U) {
                        if (stage >= 3U && stage - 3U < totalWork) {
                        const uint32_t work = stage - 3U;
                        constexpr uint32_t qSub = 0U;
                        const uint32_t kvBlock = work;
                        const uint32_t taskKvBlocks = subKvBlocks[qSub];
                        const uint32_t qStart = qBlock * Q_QUANT_BLOCK + qSub * Q_TILE;
                        const uint32_t qReal = Min(Q_TILE, qSeq_ - qStart);
                        const uint32_t firstHalf = (qReal + 1U) >> 1U;
                        const uint32_t halfRows = subBlockIdx_ == 0U ? firstHalf : qReal - firstHalf;
                        const uint32_t rowOffset = subBlockIdx_ * firstHalf;
                        const uint32_t slot2 = work & 1U;
                        const uint32_t slot3 = work % PIPELINE_SLOTS_3;
                        const bool first = kvBlock == 0U;
                        const bool last = kvBlock + 1U == taskKvBlocks;
                        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(PV_READY_FLAG[slot2]);
                        if constexpr (PROFILE_CUT == 0U) {
                            if constexpr (!RETURN_LSE && !SMOOTH_V) {
                            if (dim_ == DPAD) {
                                auto pvLayout = tla::MakeLayout<ElementOTmp, LayoutTagOTmp>(halfRows, DPAD);
                                auto pvSrc = first ? runningOUb_[qSub] : pvUb_[slot2];
                                auto pvTensor = tla::MakeTensor(pvSrc, pvLayout, Catlass::Arch::PositionUB{});
                                if constexpr (LAYOUT_NHD) {
                                    auto outLayout =
                                        tla::MakeLayout<InputT, LayoutTagOTmp>(batch_ * qSeq_, qHeads_ * dim_);
                                    auto outTensor = tla::MakeTensor(outputTyped_, outLayout, Catlass::Arch::PositionGM{});
                                    auto outTile = GetTile(outTensor,
                                        tla::MakeCoord(b * qSeq_ + qStart + rowOffset, hq * dim_),
                                        tla::MakeShape(halfRows, dim_));
                                    rescale0(outTile, expMaxUb_[slot3], sumUb_[qSub], pvTensor, vScaleUb_,
                                        first, last, kvBlock, MM2_FREE_FLAG[slot2]);
                                } else {
                                    auto outLayout = tla::MakeLayout<InputT, LayoutTagOTmp>(
                                        batch_ * qHeads_ * qSeq_, dim_);
                                    auto outTensor = tla::MakeTensor(outputTyped_, outLayout, Catlass::Arch::PositionGM{});
                                    auto outTile = GetTile(outTensor,
                                        tla::MakeCoord((b * qHeads_ + hq) * qSeq_ + qStart + rowOffset, 0U),
                                        tla::MakeShape(halfRows, dim_));
                                    rescale0(outTile, expMaxUb_[slot3], sumUb_[qSub], pvTensor, vScaleUb_,
                                        first, last, kvBlock, MM2_FREE_FLAG[slot2]);
                                }
                            } else {
                                ApplyVScale(halfRows, slot2);
                                UpdateOutput(b, hq, hkv, qStart, halfRows, rowOffset, first, last, kvBlock,
                                    slot2, slot3, qSub);
                            }
                            } else {
                                ApplyVScale(halfRows, slot2);
                                UpdateOutput(b, hq, hkv, qStart, halfRows, rowOffset, first, last, kvBlock, slot2,
                                    slot3, qSub);
                            }
                        } else {
                            AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_V>(MM2_FREE_FLAG[slot2]);
                        }
                        }
                    }
                }
            }
        }
    }

private:
    template <bool APPLY_CAUSAL>
    __simd_vf__ static inline void DequantScoresPerThreadVf(__ubuf__ int32_t *src, __ubuf__ float *dst,
        __ubuf__ float *kScale, __ubuf__ float *qScale, uint16_t rows, uint16_t rowOffset, uint16_t cols,
        float smScale, int32_t validBase)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait INT32_TO_FP32 = {
            RegLayout::UNKNOWN, SatMode::UNKNOWN, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        RegTensor<int32_t> lanes;
        RegTensor<int32_t> scaleIndex;
        RegTensor<int32_t> scaleMask;
        Arange(lanes, 0);
        ShiftRights(scaleIndex, lanes, static_cast<int16_t>(1), full);
        Duplicate(scaleMask, 3, full);
        And(scaleIndex, scaleIndex, scaleMask, full);

        RegTensor<float> kScale0;
        RegTensor<float> kScale1;
        Gather(kScale0, kScale, reinterpret_cast<RegTensor<uint32_t> &>(scaleIndex), full);
        if (cols > 64U) {
            Gather(kScale1, kScale + 4U, reinterpret_cast<RegTensor<uint32_t> &>(scaleIndex), full);
        }
        RegTensor<float> minScoreReg;
        if constexpr (APPLY_CAUSAL) {
            Duplicate(minScoreReg, MIN_SCORE, full);
        }

        if (rows == 64U && cols == 128U) {
            const uint16_t baseGroup = (rowOffset / WARP_ROWS) * THREAD_ROWS;
            for (uint16_t group = 0U; group < 16U; ++group) {
                RegTensor<float> qScaleReg;
                RegTensor<float> scoreScale0;
                RegTensor<float> scoreScale1;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(qScaleReg, qScale + baseGroup + group);
                Muls(qScaleReg, qScaleReg, smScale, full);
                Mul(scoreScale0, kScale0, qScaleReg, full);
                Mul(scoreScale1, kScale1, qScaleReg, full);
                uint16_t r = (group / THREAD_ROWS) * WARP_ROWS + group % THREAD_ROWS;
#define SAGE_DEQUANT_FULL_ROW() \
                { \
                    RegTensor<int32_t> accum0; \
                    RegTensor<float> score0; \
                    RegTensor<float> scaled0; \
                    RegTensor<int32_t> accum1; \
                    RegTensor<float> score1; \
                    RegTensor<float> scaled1; \
                    LoadAlign(accum0, src + static_cast<uint32_t>(r) * KV_TILE); \
                    LoadAlign(accum1, src + static_cast<uint32_t>(r) * KV_TILE + 64U); \
                    Cast<float, int32_t, INT32_TO_FP32>(score0, accum0, full); \
                    Cast<float, int32_t, INT32_TO_FP32>(score1, accum1, full); \
                    Mul(scaled0, score0, scoreScale0, full); \
                    Mul(scaled1, score1, scoreScale1, full); \
                    if constexpr (APPLY_CAUSAL) { \
                        int32_t validCols = validBase + r; \
                        int32_t valid0 = validCols < 64 ? validCols : 64; \
                        int32_t valid1 = validCols > 64 ? validCols - 64 : 0; \
                        MaskReg validMask0; \
                        MaskReg validMask1; \
                        CompareScalar<int32_t, AscendC::CMPMODE::LT>(validMask0, lanes, valid0, full); \
                        CompareScalar<int32_t, AscendC::CMPMODE::LT>(validMask1, lanes, valid1, full); \
                        Select(scaled0, scaled0, minScoreReg, validMask0); \
                        Select(scaled1, scaled1, minScoreReg, validMask1); \
                    } \
                    StoreAlign<float, StoreDist::DIST_NORM_B32>( \
                        dst + static_cast<uint32_t>(r) * KV_TILE, scaled0, full); \
                    StoreAlign<float, StoreDist::DIST_NORM_B32>( \
                        dst + static_cast<uint32_t>(r) * KV_TILE + 64U, scaled1, full); \
                }
                SAGE_DEQUANT_FULL_ROW();
                r += THREAD_ROWS;
                SAGE_DEQUANT_FULL_ROW();
                r += THREAD_ROWS;
                SAGE_DEQUANT_FULL_ROW();
                r += THREAD_ROWS;
                SAGE_DEQUANT_FULL_ROW();
#undef SAGE_DEQUANT_FULL_ROW
            }
        } else {
          for (uint16_t r = 0U; r < rows; ++r) {
            const uint16_t qRow = rowOffset + r;
            const uint16_t qGroup = (qRow / WARP_ROWS) * THREAD_ROWS + qRow % THREAD_ROWS;
            RegTensor<float> qScaleReg;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(qScaleReg, qScale + qGroup);
            Muls(qScaleReg, qScaleReg, smScale, full);

            RegTensor<int32_t> accum0;
            RegTensor<float> score0;
            RegTensor<float> scaled0;
            uint32_t firstCols = cols < 64U ? cols : 64U;
            MaskReg firstMask = UpdateMask<float>(firstCols);
            LoadAlign(accum0, src + static_cast<uint32_t>(r) * KV_TILE);
            Cast<float, int32_t, INT32_TO_FP32>(score0, accum0, firstMask);
            Mul(scaled0, score0, kScale0, firstMask);
            Mul(scaled0, scaled0, qScaleReg, firstMask);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(
                dst + static_cast<uint32_t>(r) * KV_TILE, scaled0, firstMask);

            if (cols > 64U) {
                RegTensor<int32_t> accum1;
                RegTensor<float> score1;
                RegTensor<float> scaled1;
                uint32_t tailCols = cols - 64U;
                MaskReg tailMask = UpdateMask<float>(tailCols);
                LoadAlign(accum1, src + static_cast<uint32_t>(r) * KV_TILE + 64U);
                Cast<float, int32_t, INT32_TO_FP32>(score1, accum1, tailMask);
                Mul(scaled1, score1, kScale1, tailMask);
                Mul(scaled1, scaled1, qScaleReg, tailMask);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(
                    dst + static_cast<uint32_t>(r) * KV_TILE + 64U, scaled1, tailMask);
            }
          }
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    template <bool APPLY_CAUSAL>
    __aicore__ inline void DequantScoresPerThread(const AscendC::LocalTensor<int32_t> &src,
        const AscendC::LocalTensor<float> &dst, const AscendC::LocalTensor<float> &kScale,
        const AscendC::LocalTensor<float> &qScale, uint32_t rows, uint32_t rowOffset, uint32_t cols,
        float smScale, int32_t validBase = 0)
    {
        auto srcAddr = reinterpret_cast<__ubuf__ int32_t *>(src.GetPhyAddr());
        auto dstAddr = reinterpret_cast<__ubuf__ float *>(dst.GetPhyAddr());
        auto kScaleAddr = reinterpret_cast<__ubuf__ float *>(kScale.GetPhyAddr());
        auto qScaleAddr = reinterpret_cast<__ubuf__ float *>(qScale.GetPhyAddr());
        DequantScoresPerThreadVf<APPLY_CAUSAL>(srcAddr, dstAddr, kScaleAddr, qScaleAddr,
            static_cast<uint16_t>(rows), static_cast<uint16_t>(rowOffset), static_cast<uint16_t>(cols),
            smScale, validBase);
    }

    __aicore__ inline uint64_t QScaleOffset(uint32_t b, uint32_t hq, uint32_t block, uint32_t group) const
    {
        return (((static_cast<uint64_t>(b) * qHeads_ + hq) * qBlocks_ + block) * qGroups_) + group;
    }

    __aicore__ inline uint64_t KScaleOffset(uint32_t b, uint32_t hkv, uint32_t row, uint32_t group) const
    {
        const uint32_t kBlocks64 = (kvSeq_ + K_QUANT_BLOCK - 1U) / K_QUANT_BLOCK;
        return (((static_cast<uint64_t>(b) * kvHeads_ + hkv) * kBlocks64 + row / K_QUANT_BLOCK) * kGroups_) +
            group;
    }

    // Obsolete non-fused fallback retained only as design history. The active path uses
    // FusedDequantSoftmax and FP16 score storage, so compiling the old FP32 tensor operations is invalid.
#if 0
    __aicore__ inline void ScaleScores(uint32_t b, uint32_t hq, uint32_t hkv, uint32_t qBlock,
        uint32_t qStart, uint32_t kvStart, uint32_t halfRows, uint32_t rowOffset, uint32_t kvReal, uint32_t slot)
    {
        bool causalFused = false;
        if constexpr (PER_THREAD) {
            const uint32_t localKScaleOffset = (kvStart / K_QUANT_BLOCK) * kGroups_;
            if constexpr (IS_CAUSAL) {
                if (halfRows == 64U && kvReal == 128U &&
                    kvStart + kvReal > qStart + rowOffset + 1U) {
                    const int32_t validBase = static_cast<int32_t>(qStart + rowOffset + 1U - kvStart);
                    DequantScoresPerThread<true>(qkAccumUb_[slot], scoreUb_[slot],
                        kScaleUb_[localKScaleOffset], qScaleUb_, halfRows, rowOffset, kvReal, smScale_, validBase);
                    causalFused = true;
                } else {
                    DequantScoresPerThread<false>(qkAccumUb_[slot], scoreUb_[slot],
                        kScaleUb_[localKScaleOffset], qScaleUb_, halfRows, rowOffset, kvReal, smScale_);
                }
            } else {
                DequantScoresPerThread<false>(qkAccumUb_[slot], scoreUb_[slot],
                    kScaleUb_[localKScaleOffset], qScaleUb_, halfRows, rowOffset, kvReal, smScale_);
            }
        } else {
            const QuantGran gran = QuantGran::PER_WARP;
            for (uint32_t c = 0U; c < kvReal; ++c) {
                const uint32_t kRow = kvStart + c;
                const uint32_t kGroup = KScaleGroup(kRow % K_QUANT_BLOCK, gran);
                kScaleUb_.SetValue(c, kScale_.GetValue(KScaleOffset(b, hkv, kRow, kGroup)));
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::Cast(scoreUb_[slot], qkAccumUb_[slot], AscendC::RoundMode::CAST_NONE, halfRows * KV_TILE);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t r = 0U; r < halfRows; ++r) {
                AscendC::Mul(scoreUb_[slot][static_cast<uint64_t>(r) * KV_TILE],
                    scoreUb_[slot][static_cast<uint64_t>(r) * KV_TILE], kScaleUb_, kvReal);
            }
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t r = 0U; r < halfRows; ++r) {
                const uint32_t qRow = qStart + rowOffset + r;
                const uint32_t qGroup = QScaleGroup(qRow % Q_QUANT_BLOCK, gran);
                const float qScale = qScale_.GetValue(QScaleOffset(b, hq, qBlock, qGroup));
                AscendC::Muls(scoreUb_[slot][static_cast<uint64_t>(r) * KV_TILE],
                    scoreUb_[slot][static_cast<uint64_t>(r) * KV_TILE], qScale * smScale_, kvReal);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        if constexpr (IS_CAUSAL) {
          if (!causalFused && kvStart + kvReal > qStart + rowOffset + 1U) {
            // Duplicate requires a 32-byte-aligned UB address. Fill only the short unaligned prefix with scalar
            // stores, then mask the remaining suffix a whole vector at a time.
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            for (uint32_t r = 0U; r < halfRows; ++r) {
                const uint32_t qRow = qStart + rowOffset + r;
                const uint32_t invalidBegin = qRow + 1U > kvStart ? qRow + 1U - kvStart : 0U;
                const uint32_t alignedBegin = Min(kvReal, (invalidBegin + 7U) & ~7U);
                for (uint32_t c = invalidBegin; c < alignedBegin; ++c) {
                    scoreUb_[slot].SetValue(static_cast<uint64_t>(r) * KV_TILE + c, MIN_SCORE);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            for (uint32_t r = 0U; r < halfRows; ++r) {
                const uint32_t qRow = qStart + rowOffset + r;
                const uint32_t invalidBegin = qRow + 1U > kvStart ? qRow + 1U - kvStart : 0U;
                const uint32_t alignedBegin = Min(kvReal, (invalidBegin + 7U) & ~7U);
                if (alignedBegin < kvReal) {
                    AscendC::Duplicate(scoreUb_[slot][static_cast<uint64_t>(r) * KV_TILE + alignedBegin],
                        MIN_SCORE, kvReal - alignedBegin);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
          }
        }
    }
#endif

    __aicore__ inline void ApplyVScale(uint32_t halfRows, uint32_t slot)
    {
        auto pvAddr = reinterpret_cast<__ubuf__ float *>(pvUb_[slot].GetPhyAddr());
        auto scaleAddr = reinterpret_cast<__ubuf__ float *>(vScaleUb_.GetPhyAddr());
        ApplyVScaleVf(pvAddr, scaleAddr, static_cast<uint16_t>(halfRows));
        // The padded-dimension output path consumes the scaled PV tile with scalar GetValue calls.
        // Make the VF stores visible to scalar before UpdateOutput starts reading the same UB region.
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
    }

    __simd_vf__ static inline void ApplyVScaleVf(
        __ubuf__ float *pv, __ubuf__ float *scale, uint16_t rows)
    {
        using namespace AscendC::MicroAPI;
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> scale0;
        LoadAlign(scale0, scale);
        RegTensor<float> scale1;
        if constexpr (DPAD > 64U) {
            LoadAlign(scale1, scale + 64U);
        }
        for (uint16_t r = 0U; r < rows; ++r) {
            __ubuf__ float *row = pv + static_cast<uint32_t>(r) * DPAD;
            RegTensor<float> value0;
            RegTensor<float> scaled0;
            LoadAlign(value0, row);
            Mul(scaled0, value0, scale0, full);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(row, scaled0, full);
            if constexpr (DPAD > 64U) {
                RegTensor<float> value1;
                RegTensor<float> scaled1;
                LoadAlign(value1, row + 64U);
                Mul(scaled1, value1, scale1, full);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(row + 64U, scaled1, full);
            }
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    __aicore__ inline void UpdateOutput(uint32_t b, uint32_t hq, uint32_t hkv, uint32_t qStart,
        uint32_t halfRows, uint32_t rowOffset, bool first, bool last, uint32_t kvBlock, uint32_t slot2,
        uint32_t slot3, uint32_t qSub)
    {
        AscendC::LocalTensor<ElementOTmp> &runningO = runningOUb_[qSub];
        AscendC::LocalTensor<ElementS> &sum = sumUb_[qSub];
        AscendC::LocalTensor<ElementS> &lastMax = maxUb_[qSub];
        if constexpr (!RETURN_LSE && !SMOOTH_V) {
            const uint32_t outputElements = halfRows * DPAD;
            if (first) {
                AscendC::DataCopy(runningO, pvUb_[slot2], outputElements);
            } else {
                for (uint32_t r = 0U; r < halfRows; ++r) {
                    AscendC::Muls(runningO[static_cast<uint64_t>(r) * DPAD],
                        runningO[static_cast<uint64_t>(r) * DPAD], expMaxUb_[slot3].GetValue(r), DPAD);
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(runningO, runningO, pvUb_[slot2], outputElements);
            }
            AscendC::PipeBarrier<PIPE_V>();
            // QK accumulator aliases the score buffer in Q128 mode; use the PV slot as the temporary
            // FP16/BF16 flush destination after its FP32 value has already been accumulated.
            auto output16Ub = pvUb_[slot2].template ReinterpretCast<InputT>();
            if (!last && ACCUM_MODE == static_cast<uint32_t>(AccumMode::FP32_FP16) &&
                (kvBlock + 1U) % flushInterval_ == 0U) {
                AscendC::Cast(output16Ub, runningO, AscendC::RoundMode::CAST_RINT, outputElements);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(runningO, output16Ub, AscendC::RoundMode::CAST_NONE, outputElements);
                AscendC::PipeBarrier<PIPE_V>();
            }
            if (last) {
                for (uint32_t r = 0U; r < halfRows; ++r) {
                    AscendC::Muls(runningO[static_cast<uint64_t>(r) * DPAD],
                        runningO[static_cast<uint64_t>(r) * DPAD], 1.0F / sum.GetValue(r), DPAD);
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(output16Ub, runningO, AscendC::RoundMode::CAST_RINT, outputElements);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                const Layout outputLayout = LAYOUT_NHD ? Layout::NHD : Layout::HND;
                for (uint32_t r = 0U; r < halfRows; ++r) {
                    const uint32_t qRow = qStart + rowOffset + r;
                    AscendC::DataCopy(
                        outputTyped_[SourceOffset<InputT>(outputLayout, b, hq, qRow, 0U, qHeads_, qSeq_, dim_)],
                        output16Ub[static_cast<uint64_t>(r) * DPAD], dim_);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            }
            AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_V>(MM2_FREE_FLAG[slot2]);
            return;
        }
        if constexpr (RETURN_LSE) {
            if (last) {
                __local_mem__ float *sumAddr =
                    reinterpret_cast<__local_mem__ float *>(sum.GetPhyAddr());
                __local_mem__ float *maxAddr =
                    reinterpret_cast<__local_mem__ float *>(lastMax.GetPhyAddr());
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::RegTensor<float> sumReg;
                    AscendC::MicroAPI::RegTensor<float> maxReg;
                    AscendC::MicroAPI::RegTensor<float> logReg;
                    AscendC::MicroAPI::RegTensor<float> lseReg;
                    AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::UpdateMask<float>(halfRows);
                    AscendC::MicroAPI::DataCopy(sumReg, sumAddr);
                    AscendC::MicroAPI::DataCopy(maxReg, maxAddr);
                    AscendC::MicroAPI::Ln(logReg, sumReg, mask);
                    AscendC::MicroAPI::Add(lseReg, maxReg, logReg, mask);
                    // Preserve sum for output normalization and cache max+log(sum) in lastMax for LSE.
                    AscendC::MicroAPI::DataCopy(maxAddr, lseReg, mask);
                    AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                        AscendC::MicroAPI::MemType::VEC_LOAD>();
                }
            }
        }
        for (uint32_t r = 0U; r < halfRows; ++r) {
            const float expMax = expMaxUb_[slot3].GetValue(r);
            const float rowSum = sum.GetValue(r);
            for (uint32_t d = 0U; d < DPAD; ++d) {
                const uint64_t offset = static_cast<uint64_t>(r) * DPAD + d;
                const float current = pvUb_[slot2].GetValue(offset);
                float running = first ? current : runningO.GetValue(offset) * expMax + current;
                if constexpr (ACCUM_MODE == static_cast<uint32_t>(AccumMode::FP32_FP16)) {
                    if (!last && (kvBlock + 1U) % flushInterval_ == 0U) {
                        running = static_cast<float>(static_cast<half>(running));
                    }
                }
                runningO.SetValue(offset, running);
                if (last && d < dim_) {
                    float result = running / rowSum;
#if SAGE_ATTENTION2_DEBUG_STAGE == 8
                    result = 1.0F;
#elif SAGE_ATTENTION2_DEBUG_STAGE == 9
                    if (d == 0U) {
                        result = rowSum;
                    } else if (d == 1U) {
                        result = current;
                    } else if (d == 2U) {
                        result = expMax;
                    } else if (d == 3U) {
                        result = vScale_.GetValue((static_cast<uint64_t>(b) * kvHeads_ + hkv) * DPAD + d);
                    } else if (d == 4U) {
                        result = running;
                    }
#endif
                    if constexpr (SMOOTH_V) {
                        const uint64_t meanOffset = (static_cast<uint64_t>(b) * kvHeads_ + hkv) * DPAD + d;
                        result += vMean_.GetValue(meanOffset);
                    }
                    const uint32_t qRow = qStart + rowOffset + r;
                    const Layout outputLayout = LAYOUT_NHD ? Layout::NHD : Layout::HND;
                    output_.SetValue(SourceOffset<InputT>(outputLayout, b, hq, qRow, d, qHeads_, qSeq_, dim_),
                        EncodeInput16<InputT>(result));
                }
            }
            if constexpr (RETURN_LSE) {
                if (last) {
                    const uint32_t qRow = qStart + rowOffset + r;
                    float lseValue = lastMax.GetValue(r);
                    if constexpr (SMOOTH_K) {
                        float correction = 0.0F;
                        const Layout inputLayout = LAYOUT_NHD ? Layout::NHD : Layout::HND;
                        const uint64_t meanBase = (static_cast<uint64_t>(b) * kvHeads_ + hkv) * DPAD;
                        for (uint32_t d = 0U; d < dim_; ++d) {
                            const float qValue = DecodeInput16<InputT>(qOriginal_.GetValue(
                                SourceOffset<InputT>(inputLayout, b, hq, qRow, d, qHeads_, qSeq_, dim_)));
                            correction += qValue * DecodeInput16<InputT>(kMean_.GetValue(meanBase + d));
                        }
                        lseValue += correction * smScale_;
                    }
                    lse_.SetValue((static_cast<uint64_t>(b) * qHeads_ + hq) * qSeq_ + qRow, lseValue);
                }
            }
        }
        // UpdateOutput uses scalar UB/GM accesses. Publish the cube-buffer-free token only after those
        // accesses complete; otherwise the next task can overwrite pvUb_ while scalar is still reading it.
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_V>(MM2_FREE_FLAG[slot2]);
    }

    static constexpr uint32_t HALF_ROWS = Q_TILE / 2U;
    static_assert(Q_QUANT_BLOCK % Q_TILE == 0U, "attention Q tile must divide the quantization block");
    static constexpr uint32_t Q_SUBTILES = Q_QUANT_BLOCK / Q_TILE;
    static constexpr uint32_t HALF_QK_ACCUM_BYTES = HALF_ROWS * KV_TILE * sizeof(ElementQKAccum);
    static constexpr uint32_t HALF_SCORE_BYTES = HALF_ROWS * KV_TILE * sizeof(ElementScore);
    static constexpr uint32_t HALF_OUTPUT_BYTES = HALF_ROWS * DPAD * sizeof(ElementOTmp);
    static constexpr uint32_t HALF_ROW_BYTES = HALF_ROWS * sizeof(ElementS);
    static constexpr uint32_t FULL_P_BYTES = Q_TILE * KV_TILE * sizeof(ElementP);
    static constexpr uint32_t SOFTMAX_SCRATCH_BYTES =
        2U * HALF_ROW_BYTES + 2U * (HALF_ROWS * KV_TILE * sizeof(ElementP) + 256U) + 256U +
        Q_TILE * sizeof(uint32_t) + HALF_ROW_BYTES;
    static constexpr uint32_t PIPELINE_SLOTS_2 = 2U;
    static constexpr uint32_t PIPELINE_SLOTS_3 = 3U;
    static constexpr float P_SCALE = FP8_MAX_VALUE;
    static constexpr uint32_t MAX_K_SCALE_ELEMENTS = 1024U;
    static constexpr uint64_t SYNC_MODE = 4U;
    static constexpr uint64_t QK_READY_FLAG[PIPELINE_SLOTS_2] = {0U, 1U};
    static constexpr uint64_t P_READY_FLAG[PIPELINE_SLOTS_3] = {2U, 3U, 4U};
    static constexpr uint64_t PV_READY_FLAG[PIPELINE_SLOTS_2] = {5U, 6U};
    static constexpr uint64_t MM2_FREE_FLAG[PIPELINE_SLOTS_2] = {7U, 8U};
    static constexpr uint64_t MM1_FREE_FLAG[PIPELINE_SLOTS_2] = {9U, 10U};
    static constexpr float QK_FIXPIPE_DEQ = 1.0F / 256.0F;
    static constexpr float QK_FIXPIPE_COMPENSATION = 256.0F;
    static constexpr uint64_t QK_FIXPIPE_DEQ_BITS =
        static_cast<uint64_t>(static_cast<uint32_t>(0x3B800000U));
    static constexpr uint32_t ATTENTION_UB_BYTES =
        PIPELINE_SLOTS_2 * HALF_QK_ACCUM_BYTES + HALF_SCORE_BYTES +
        PIPELINE_SLOTS_2 * HALF_OUTPUT_BYTES +
        (2U * Q_SUBTILES + PIPELINE_SLOTS_3) * HALF_ROW_BYTES + DPAD * sizeof(float) +
        MAX_K_SCALE_ELEMENTS * sizeof(float) + 32U * sizeof(float) + SOFTMAX_SCRATCH_BYTES +
        Q_SUBTILES * HALF_OUTPUT_BYTES;
    static_assert(ATTENTION_UB_BYTES <= ArchTag::UB_SIZE,
        "Q128/KV256 FP16-score dual-P/dual-PV attention exceeds Ascend950 UB capacity");

    const SageAttention2TilingData *tiling_ = nullptr;
    AscendC::GlobalTensor<uint16_t> qOriginal_;
    AscendC::GlobalTensor<uint16_t> output_;
    AscendC::GlobalTensor<InputT> outputTyped_;
    AscendC::GlobalTensor<float> lse_;
    AscendC::GlobalTensor<ElementQ> q8_;
    AscendC::GlobalTensor<ElementK> k8_;
    AscendC::GlobalTensor<ElementV> v8_;
    AscendC::GlobalTensor<float> qScale_;
    AscendC::GlobalTensor<float> kScale_;
    AscendC::GlobalTensor<float> vScale_;
    AscendC::GlobalTensor<uint16_t> kMean_;
    AscendC::GlobalTensor<float> vMean_;
    AscendC::LocalTensor<ElementQKAccum> qkAccumUb_[PIPELINE_SLOTS_2];
    AscendC::LocalTensor<ElementScore> scoreUb_[1];
    AscendC::LocalTensor<ElementOTmp> pvUb_[PIPELINE_SLOTS_2];
    AscendC::LocalTensor<ElementOTmp> runningOUb_[Q_SUBTILES];
    AscendC::LocalTensor<ElementS> sumUb_[Q_SUBTILES];
    AscendC::LocalTensor<ElementS> expMaxUb_[PIPELINE_SLOTS_3];
    AscendC::LocalTensor<ElementS> maxUb_[Q_SUBTILES];
    AscendC::LocalTensor<float> vScaleUb_;
    AscendC::LocalTensor<float> kScaleUb_;
    AscendC::LocalTensor<float> qScaleUb_;
    AscendC::LocalTensor<ElementP> pL1_[PIPELINE_SLOTS_3];
    Catlass::Arch::Resource<ArchTag> resource_;
    uint32_t batch_ = 0U;
    uint32_t qHeads_ = 0U;
    uint32_t kvHeads_ = 0U;
    uint32_t qSeq_ = 0U;
    uint32_t kvSeq_ = 0U;
    uint32_t kvSeqPad_ = 0U;
    uint32_t dim_ = 0U;
    uint32_t qBlocks_ = 0U;
    uint32_t kvBlocks_ = 0U;
    uint32_t gqaRatio_ = 0U;
    uint32_t coreNum_ = 0U;
    uint32_t totalTasks_ = 0U;
    uint32_t qGroups_ = 0U;
    uint32_t kGroups_ = 0U;
    uint32_t blockIdx_ = 0U;
    uint32_t subBlockIdx_ = 0U;
    uint32_t flushInterval_ = 0U;
    uint32_t l1Offset_ = 0U;
    uint32_t ubOffset_ = 0U;
    float smScale_ = 1.0F;
};

} // namespace SageAttention2Kernel
#endif // SAGE_ATTENTION2_CATLASS_H_
