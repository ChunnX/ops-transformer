#pragma once

#include "platform_traits.h"
#include "ring_buffer_layout.h"
#include "credit_handshake.h"

#if defined(__CCE__)
#include "kernel_operator.h"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm/tile/copy_l0c_to_ub.hpp"
#endif

namespace sage::microkernel {

#if defined(__CCE__)

struct MxScaleCache {
    GM_ADDR base = nullptr;
};

template <AscendC::CO2Layout OutLayout = AscendC::CO2Layout::NZ>
__aicore__ static inline AscendC::FixpipeParamsArch3510<OutLayout> BuildDequantFixpipeParams(
    const uint16_t rowCount, const uint16_t colCount, const bool isSubBlock, const uint16_t srcStride = 0U)
{
    const uint16_t stride = (srcStride == 0U) ? rowCount : srcStride;
    AscendC::FixpipeParamsArch3510<OutLayout> fp(colCount, rowCount, stride, rowCount * 16U);
    fp.quantPre = QuantMode_t::DEQF16;
    fp.deqScalar = sage::core::DEQ_SCALAR_DEQF16;
    fp.dualDstCtl = 0U;
    fp.subBlockId = isSubBlock;
    return fp;
}

template <AscendC::CO2Layout OutLayout = AscendC::CO2Layout::NZ>
__aicore__ static inline void DispatchDequantFixpipe(
    AscendC::LocalTensor<half> ubDst,
    AscendC::LocalTensor<int32_t> l0cSrc,
    const uint16_t rowCount,
    const uint16_t colCount,
    const bool isSubBlock,
    const uint16_t srcStride = 0U)
{
    static constexpr AscendC::FixpipeConfig pipeCfg{OutLayout, true};
    const auto fpParams = BuildDequantFixpipeParams<OutLayout>(rowCount, colCount, isSubBlock, srcStride);
    AscendC::Fixpipe<half, int32_t, pipeCfg>(ubDst, l0cSrc, fpParams);
}

template <AscendC::CO2Layout OutLayout = AscendC::CO2Layout::NZ>
__aicore__ static inline void DispatchDequantFixpipe(
    AscendC::LocalTensor<half> ubDst,
    AscendC::LocalTensor<int32_t> l0cSrc,
    const uint16_t rowCount,
    const uint16_t colCount,
    const bool isSubBlock,
    const uint16_t srcStride,
    const event_t syncEvt,
    const uint32_t aivSubBlock,
    const orchestration::TokenSignal signal = orchestration::TokenSignal::QK_READY)
{
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(syncEvt);
    DispatchDequantFixpipe<OutLayout>(ubDst, l0cSrc, rowCount, colCount, isSubBlock, srcStride);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(syncEvt);
    orchestration::EventHandshake::NotifyAiv<PIPE_FIX>(signal, aivSubBlock);
}

// Architectural Invariant Verification:
// P_NZ_PAD_ROWS = 1 is strictly required by the Catlass / Ascend 950PR microarchitecture:
// 1. Physically breaks the 16-bank stride resonance across fractal planes in UB/L1.
// 2. Provides the 32B padding row required by StoreNZPlanePair so secondary plane
//    stores do not clobber active rows of the preceding plane.
static_assert(sage::core::P_NZ_PAD_ROWS == 1, "P_NZ_PAD_ROWS must be 1 to guarantee 16-bank conflict elimination and avoid plane clobbering");

template <uint32_t BOffset>
class SplitQkGemmPipeline {
public:
    using ArchTag = memory::Arch;
    using Element = int8_t;
    using Accumulator = int32_t;
    using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, Element, Catlass::layout::RowMajor, Element, Catlass::layout::ColumnMajor, float, Catlass::layout::RowMajor>;

    static constexpr uint32_t K_DIM = 128U;
    static constexpr uint32_t N_DIM = sage::core::N;
    static constexpr uint32_t SUB_M = sage::core::HALF;

    template <uint32_t SubPartId, typename L0BType>
    __aicore__ static inline void ExecuteSubMmad(memory::FixedResource &res, const L0BType &l0TensorB, event_t syncEvt) {
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(syncEvt);
        constexpr uint32_t aOffset = SubPartId * (SUB_M * K_DIM);
        constexpr uint32_t cOffset = SubPartId * (SUB_M * N_DIM * sizeof(Accumulator));
        auto l0SubA = tla::MakeTensor(res.l0ABuf.template GetBufferByByte<Element>(aOffset),
            tla::MakeLayout<Element, typename TileCopy::LayoutTagL0A>(SUB_M, K_DIM), Catlass::Arch::PositionL0A{});
        auto l0SubC = tla::MakeTensor(res.l0CBuf.template GetBufferByByte<Accumulator>(cOffset),
            tla::MakeLayout<Accumulator, Catlass::layout::L0C>(SUB_M, N_DIM), Catlass::Arch::PositionL0C{});
        Catlass::Gemm::Tile::TileMmadTla<ArchTag, Element, typename TileCopy::LayoutTagL1A>()(
            l0SubC, l0SubA, l0TensorB, SUB_M, N_DIM, K_DIM, true);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(syncEvt);
    }

    template <uint32_t SubPartId>
    __aicore__ static inline void ExecuteSubFixpipe(memory::FixedResource &res, uint32_t ubOffset, event_t syncEvt, bool notifyAiv = true) {
        constexpr uint32_t cOffset = SubPartId * (SUB_M * N_DIM * sizeof(Accumulator));
        auto ubDestination = res.ubBuf.template GetBufferByByte<half>(ubOffset);
        auto l0cSource = res.l0CBuf.template GetBufferByByte<Accumulator>(cOffset);
        if (notifyAiv) {
            DispatchDequantFixpipe(
                ubDestination, l0cSource, SUB_M, N_DIM, SubPartId != 0U, SUB_M, syncEvt, SubPartId);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(syncEvt);
            DispatchDequantFixpipe(
                ubDestination, l0cSource, SUB_M, N_DIM, SubPartId != 0U, SUB_M);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(syncEvt);
        }
    }

    // ========================================================================
    // DeepEP & DeepGEMM Asynchronous Pipelining (Fixpipe 0 || MMAD 1 Concurrency):
    // On Atlas 950PR (C310/A5), MMAD operates on PIPE_M while Fixpipe operates
    // on PIPE_FIX. Because SubPart 0 (L0C [0..32KB]) and SubPart 1 (L0C [32..64KB])
    // occupy physically disjoint memory regions in L0A and L0C, issuing Fixpipe
    // for SubPart 0 followed immediately by MMAD for SubPart 1 achieves complete
    // hardware concurrency. The Fixpipe dequantization latency (~1024 cycles) is
    // fully hidden beneath the compute latency of SubPart 1 MMAD.
    // ========================================================================
    template <typename L0BType>
    __aicore__ static inline void ExecutePipelinedMmadAndFixpipe(memory::FixedResource &res, const L0BType &l0TensorB, uint32_t ubOffset, bool notifyAiv = true) {
        constexpr auto syncEvt0 = static_cast<event_t>(orchestration::EventHandshake::QK_EVENT);
        constexpr auto syncEvt1 = static_cast<event_t>(orchestration::EventHandshake::QK_SECOND_FIX_EVENT);

        // Stage 0: MMAD for SubPart 0 on Cube core (PIPE_M)
        ExecuteSubMmad<0U>(res, l0TensorB, syncEvt0);

        // Stage 1: Dual-Issue Async Overlap — Fixpipe (SubPart 0 on PIPE_FIX) || MMAD (SubPart 1 on PIPE_M)
        ExecuteSubFixpipe<0U>(res, ubOffset, syncEvt0, notifyAiv);
        ExecuteSubMmad<1U>(res, l0TensorB, syncEvt1);

        // Stage 2: Fixpipe for SubPart 1 on PIPE_FIX upon MMAD 1 completion
        ExecuteSubFixpipe<1U>(res, ubOffset, syncEvt1, notifyAiv);
    }

    __aicore__ static inline void Execute(memory::FixedResource &res, uint32_t ubOffset, bool loadA, uint32_t kParity, bool notifyAiv = true) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(orchestration::EventHandshake::QK_EVENT);
        auto l1TensorB = tla::MakeTensor(res.l1Buf.template GetBufferByByte<Element>(BOffset + kParity * 65536U), tla::MakeLayout<Element, typename TileCopy::LayoutTagL1B>(K_DIM, N_DIM), Catlass::Arch::PositionL1{});

        if (loadA) orchestration::EventHandshake::WaitFromBothAiv<PIPE_MTE1>(orchestration::TokenSignal::Q_READY);

        const auto kFetchEvt = static_cast<event_t>(orchestration::EventHandshake::K_PREFETCH_EVENT_BASE + kParity);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kFetchEvt);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(orchestration::EventHandshake::QK_EVENT);

        auto l0TensorB = tla::MakeTensor(res.l0BBuf.template GetBufferByByte<Element>(0U), tla::MakeLayout<Element, typename TileCopy::LayoutTagL0B>(K_DIM, N_DIM), Catlass::Arch::PositionL0B{});
        typename TileCopy::CopyL1ToL0B()(l0TensorB, tla::GetTile(l1TensorB, tla::MakeCoord(0U, 0U), tla::MakeShape(K_DIM, N_DIM)));

        if (loadA) {
            auto l1TensorA = tla::MakeTensor(res.l1Buf.template GetBufferByByte<Element>(0U), tla::MakeLayout<Element, typename TileCopy::LayoutTagL1A>(sage::core::M, K_DIM), Catlass::Arch::PositionL1{});
            #pragma unroll
            for (uint32_t part = 0U; part < 2U; ++part) {
                auto l0A_Part = tla::MakeTensor(res.l0ABuf.template GetBufferByByte<Element>(part * (SUB_M * K_DIM)), tla::MakeLayout<Element, typename TileCopy::LayoutTagL0A>(SUB_M, K_DIM), Catlass::Arch::PositionL0A{});
                typename TileCopy::CopyL1ToL0A()(l0A_Part, tla::GetTile(l1TensorA, tla::MakeCoord(part * SUB_M, 0U), tla::MakeShape(SUB_M, K_DIM)));
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kFetchEvt);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(orchestration::EventHandshake::QK_EVENT);
        memory::Sync<AscendC::HardEvent::MTE1_M>();

        ExecutePipelinedMmadAndFixpipe(res, l0TensorB, ubOffset, notifyAiv);
    }
};

template <class ElementT, class MatrixBLayout, bool IsLocalA = false, uint32_t BufferBOffset = sage::core::WIDE_B_L1, bool EmitToUB = true, bool InitAccumulator = true, bool IsPreloadedB = false>
class DelayedGemmPipeline {
public:
    using ArchTag = memory::Arch;
    using Element = ElementT;
    using LayoutB = MatrixBLayout;
    using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, Element, Catlass::layout::RowMajor, Element, LayoutB, float, Catlass::layout::RowMajor>;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;
    using L0OperandTypeA = std::conditional_t<IsLocalA, AscendC::mx_fp8_e4m3_t, Element>;

    static constexpr event_t SYNC_SLOT = static_cast<event_t>(IsLocalA ? EventHandshake::PV_EVENT : EventHandshake::QK_EVENT);
    static constexpr uint32_t L0_OPERAND_OFFSET = IsLocalA ? 32768U : 0U;
    static constexpr uint32_t L0C_ACCUMULATOR_OFFSET = IsLocalA ? 131072U : 0U;

    template <typename TensorLayoutTag>
    __aicore__ static inline auto BindGlobalMatrix(GM_ADDR gmPtr, uint32_t rows, uint32_t cols) {
        AscendC::GlobalTensor<Element> gt;
        gt.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(gmPtr));
        return tla::MakeTensor(gt, tla::MakeLayout<Element, TensorLayoutTag>(rows, cols), Catlass::Arch::PositionGM{});
    }

    __aicore__ static inline void Execute(
        memory::FixedResource &res, GM_ADDR addrA, GM_ADDR addrB, uint32_t kDim, uint32_t nDim,
        uint32_t ubOffset, uint32_t parity = 0U, bool loadA = true, GM_ADDR mxScale = nullptr,
        MxScaleCache *mxCache = nullptr, uint32_t mxK = 0U, uint32_t mxStart = 0U, uint32_t vParity = 0U)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_SLOT);
        const uint32_t l1AOffset = IsLocalA ? ((BufferBOffset == sage::core::WIDE_B_L1 ? 0U : sage::core::P_L1) + parity * sage::core::M * sage::core::N) : 0U;
        auto l1TensorA = tla::MakeTensor(res.l1Buf.template GetBufferByByte<Element>(l1AOffset), tla::MakeLayout<Element, typename TileCopy::LayoutTagL1A>(sage::core::M, kDim), Catlass::Arch::PositionL1{});

        const uint32_t l1BOffset = BufferBOffset + (IsLocalA ? sage::core::N * 128U : 0U) + (IsPreloadedB ? vParity * 65536U : 0U);
        auto l1TensorB = tla::MakeTensor(res.l1Buf.template GetBufferByByte<Element>(l1BOffset), tla::MakeLayout<Element, typename TileCopy::LayoutTagL1B>(kDim, nDim), Catlass::Arch::PositionL1{});

        const bool isCacheValid = (mxCache != nullptr) && (static_cast<uint64_t>(mxK) / 32U * nDim <= sage::core::MX_P_L1 - sage::core::MX_V_L1);
        const uint32_t scaleDimK = (isCacheValid) ? mxK : kDim;
        const uint32_t scaleStartPos = (isCacheValid) ? mxStart : 0U;
        GM_ADDR scaleBaseAddr = (isCacheValid) ? (mxScale - static_cast<uint64_t>(mxStart) / 32U * nDim) : mxScale;
        const uint32_t scaleBufferOffset = IsPreloadedB ? ((vParity & 1U) ? sage::core::V_SCALE_L1_PONG : sage::core::V_SCALE_L1_PING) : sage::core::MX_V_L1;

        auto l1ScaleTensor = tla::MakeTensor(res.l1Buf.template GetBufferByByte<float8_e8m0_t>(scaleBufferOffset), tla::MakeMxScaleLayout<float8_e8m0_t, Catlass::layout::nN, true>(IsPreloadedB ? kDim / 32U : scaleDimK / 32U, nDim), tla::MakeCoord(0U, 0U), Catlass::Arch::PositionL1{});

        if constexpr (IsLocalA == true) {
            if constexpr (IsPreloadedB == false) {
                if (!isCacheValid || mxCache->base != scaleBaseAddr) {
                    AscendC::GlobalTensor<float8_e8m0_t> globalScaleTensor;
                    globalScaleTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float8_e8m0_t *>(scaleBaseAddr));
                    auto gmScaleLayout = tla::MakeTensor(globalScaleTensor, tla::MakeMxScaleLayout<float8_e8m0_t, Catlass::layout::RowMajor, true>(scaleDimK / 32U, nDim), Catlass::Arch::PositionGM{});
                    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(gmScaleLayout), decltype(l1ScaleTensor)>()(l1ScaleTensor, gmScaleLayout);
                    if (isCacheValid) mxCache->base = scaleBaseAddr;
                }
            }
        }

        if constexpr (IsLocalA == false) {
            if (loadA) {
                auto gmTensorA = BindGlobalMatrix<Catlass::layout::RowMajor>(addrA, sage::core::M, kDim);
                typename TileCopy::template CopyGmToL1A<decltype(gmTensorA)>()(l1TensorA, gmTensorA);
            }
            auto gmTensorB = BindGlobalMatrix<LayoutB>(addrB, kDim, nDim);
            typename TileCopy::template CopyGmToL1B<decltype(gmTensorB)>()(l1TensorB, gmTensorB);
        } else if constexpr (IsPreloadedB == false) {
            auto gmTensorV = BindGlobalMatrix<LayoutB>(addrB, kDim, nDim);
            typename TileCopy::template CopyGmToL1B<decltype(gmTensorV)>()(l1TensorB, gmTensorV);
        }

        if constexpr (IsPreloadedB == true) {
            const auto prefetchEvt = static_cast<event_t>(EventHandshake::V_PREFETCH_EVENT_BASE + (vParity & 1U));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(prefetchEvt);
        } else {
            memory::Sync<AscendC::HardEvent::MTE2_MTE1>();
        }

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(SYNC_SLOT);
        auto l0TensorA = tla::MakeTensor(res.l0ABuf.template GetBufferByByte<L0OperandTypeA>(L0_OPERAND_OFFSET), tla::MakeLayout<Element, typename TileCopy::LayoutTagL0A>(sage::core::M, kDim), Catlass::Arch::PositionL0A{});
        auto l0TensorB = tla::MakeTensor(res.l0BBuf.template GetBufferByByte<L0OperandTypeA>(L0_OPERAND_OFFSET), tla::MakeLayout<Element, typename TileCopy::LayoutTagL0B>(kDim, nDim), Catlass::Arch::PositionL0B{});

        if constexpr (IsLocalA == true) {
            auto reinterpretedL0A = tla::MakeTensor(l0TensorA.data().template ReinterpretCast<Element>(), l0TensorA.layout(), Catlass::Arch::PositionL0A{});
            typename TileCopy::CopyL1ToL0A()(reinterpretedL0A, tla::GetTile(l1TensorA, tla::MakeCoord(0U, 0U), tla::MakeShape(sage::core::M, kDim)));
            typename TileCopy::CopyL1ToL0B()(l0TensorB, tla::GetTile(l1TensorB, tla::MakeCoord(0U, 0U), tla::MakeShape(kDim, nDim)), tla::GetTile(l1ScaleTensor, tla::MakeCoord(IsPreloadedB ? 0U : scaleStartPos / 32U, 0U), tla::MakeShape(kDim / 32U, nDim)));
        } else {
            if (loadA) typename TileCopy::CopyL1ToL0A()(l0TensorA, tla::GetTile(l1TensorA, tla::MakeCoord(0U, 0U), tla::MakeShape(sage::core::M, kDim)));
            typename TileCopy::CopyL1ToL0B()(l0TensorB, tla::GetTile(l1TensorB, tla::MakeCoord(0U, 0U), tla::MakeShape(kDim, nDim)));
        }

        if constexpr (IsPreloadedB == true) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<event_t>(EventHandshake::V_PREFETCH_EVENT_BASE + (vParity & 1U)));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_SLOT);
        memory::Sync<AscendC::HardEvent::MTE1_M>();

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(SYNC_SLOT);
        auto l0AccumulatorC = tla::MakeTensor(res.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_ACCUMULATOR_OFFSET), tla::MakeLayout<ElementAccumulator, Catlass::layout::L0C>(sage::core::M, nDim), Catlass::Arch::PositionL0C{});

        Catlass::Gemm::Tile::TileMmadTla<ArchTag, Element, typename TileCopy::LayoutTagL1A>()(
            l0AccumulatorC, l0TensorA, l0TensorB, sage::core::M, nDim, kDim, InitAccumulator);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(SYNC_SLOT);
        memory::Sync<AscendC::HardEvent::M_FIX>();

        if constexpr (EmitToUB == false) {
            // Retain partial sum in L0C
        } else if constexpr (std::is_same_v<Element, int8_t>) {
            auto ubDestination = res.ubBuf.template GetBufferByByte<half>(ubOffset);
            #pragma unroll
            for (uint32_t subBlock = 0U; subBlock < 2U; ++subBlock) {
                const uint32_t l0cByteOffset = L0C_ACCUMULATOR_OFFSET + subBlock * (1024U * sizeof(int32_t));
                auto l0cSource = res.l0CBuf.template GetBufferByByte<int32_t>(l0cByteOffset);
                DispatchDequantFixpipe(ubDestination, l0cSource, sage::core::HALF, nDim, subBlock != 0U, sage::core::M);
            }
        } else {
            auto ubFloatTensor = tla::MakeTensor(res.ubBuf.template GetBufferByByte<float>(ubOffset), tla::MakeLayout<float, Catlass::layout::RowMajor>(sage::core::M, nDim), Catlass::Arch::PositionUB{});
            Catlass::Gemm::Tile::CopyL0CToUBTla<ArchTag, decltype(l0AccumulatorC), decltype(ubFloatTensor), Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>()(ubFloatTensor, l0AccumulatorC);
        }

        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(SYNC_SLOT);
    }
};

template <uint32_t BOffset>
__aicore__ inline __attribute__((always_inline)) void GemmQkSplit(memory::FixedResource &r, uint32_t ubOffset, bool loadA, uint32_t kParity, bool notifyAiv = true) {
    SplitQkGemmPipeline<BOffset>::Execute(r, ubOffset, loadA, kParity, notifyAiv);
}

template <class T, class BLayout, bool LocalA = false, uint32_t BOffset = sage::core::WIDE_B_L1, bool Emit = true, bool Initialize = true, bool PreloadedB = false>
__aicore__ inline __attribute__((always_inline)) void Gemm(memory::FixedResource &r, GM_ADDR a, GM_ADDR b, uint32_t k, uint32_t n, uint32_t ubOffset, uint32_t parity = 0U, bool loadA = true, GM_ADDR mxScale = nullptr, MxScaleCache *mxCache = nullptr, uint32_t mxK = 0U, uint32_t mxStart = 0U, uint32_t vParity = 0U) {
    DelayedGemmPipeline<T, BLayout, LocalA, BOffset, Emit, Initialize, PreloadedB>::Execute(r, a, b, k, n, ubOffset, parity, loadA, mxScale, mxCache, mxK, mxStart, vParity);
}

#endif // __CCE__
} // namespace sage::microkernel

