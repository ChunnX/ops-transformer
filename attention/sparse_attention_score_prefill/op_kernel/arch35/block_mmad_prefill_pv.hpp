/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Prefill PV matmul: V block resident in L1, P from UB → L1 → L0A.
 * Mirrors decode block_mmad_pv_arch35_ABf16_C_to_UB.hpp but simplified
 * for single-core (no cross-core sync needed).
 *
 * Layout: P [groupSize, validSize] zN (A), V [validSize, D] RowMajor (B)
 * Output: O [groupSize, D] → UB via FixPipe (SPLIT_M mode)
 */
#ifndef BLOCK_MMAD_PREFILL_PV_HPP
#define BLOCK_MMAD_PREFILL_PV_HPP

#include "kernel_utils.hpp"

namespace SparseAttnPrefill {

using namespace NpuArch;

template <class ElementP_, class ElementV_, class ElementO_>
struct PrefillBlockMmadPV {
    using ArchTag = Arch::AtlasA5;
    using ElementP = ElementP_;
    using ElementV = ElementV_;
    using ElementO = ElementO_;
    using ElementAccumulator = float;

    using LayoutTagP = layout::zN;
    using LayoutTagV = layout::RowMajor;
    using LayoutTagO = layout::RowMajor;

    using TileCopy = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementP, LayoutTagP, ElementV, LayoutTagV,
        ElementO, LayoutTagO, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    using TileMmad = Gemm::Tile::TileMmadTla<ArchTag, ElementP, LayoutTagL1A>;

    static constexpr uint32_t L0_STAGES = 2;
    static constexpr uint32_t L0_TILE_M = 128;
    static constexpr uint32_t L0_TILE_N = 128;
    static constexpr uint32_t L0_TILE_K = 128;
    static constexpr uint32_t L0A_BUF_SIZE = ArchTag::L0A_SIZE / L0_STAGES;
    static constexpr uint32_t L0B_BUF_SIZE = ArchTag::L0B_SIZE / L0_STAGES;
    static constexpr uint32_t L0C_HALF_SIZE = ArchTag::L0C_SIZE / 2;
    static constexpr uint32_t L0C_BUF_SIZE = L0C_HALF_SIZE / L0_STAGES;

    AscendC::LocalTensor<ElementV> l1VTensor_;
    AscendC::LocalTensor<ElementP> l1PTensor_;
    AscendC::LocalTensor<ElementP> l0ATensor_[L0_STAGES];
    AscendC::LocalTensor<ElementV> l0BTensor_[L0_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor_[L0_STAGES];

    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    TileMmad tileMmad;

    __aicore__ inline
    PrefillBlockMmadPV() {}

    __aicore__ inline
    void Init(Arch::Resource<ArchTag> &resource, uint32_t l1StartAddr,
              uint32_t blockSize, uint32_t D, uint32_t groupSize)
    {
        uint32_t vBufBytes = blockSize * D * sizeof(ElementV);
        uint32_t pBufBytes = groupSize * blockSize * sizeof(ElementP);

        l1VTensor_ = resource.l1Buf.template GetBufferByByte<ElementV>(l1StartAddr);
        l1PTensor_ = resource.l1Buf.template GetBufferByByte<ElementP>(l1StartAddr + vBufBytes);

        for (uint32_t i = 0; i < L0_STAGES; i++) {
            l0ATensor_[i] = resource.l0ABuf.template GetBufferByByte<ElementP>(L0A_BUF_SIZE * i);
            l0BTensor_[i] = resource.l0BBuf.template GetBufferByByte<ElementV>(L0B_BUF_SIZE * i);
            l0CTensor_[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(
                L0C_HALF_SIZE + L0C_BUF_SIZE * i);
        }
    }

    template <class TensorV>
    __aicore__ inline
    void LoadVResident(TensorV &gmVTensor, uint32_t validSize, uint32_t D)
    {
        using CopyGmToL1V = typename TileCopy::template CopyGmToL1B<TensorV>;
        CopyGmToL1V copyGmToL1V;

        auto l1VLayout = tla::MakeLayout<ElementV, LayoutTagL1B>(validSize, D);
        auto l1VTensorTla = tla::MakeTensor(l1VTensor_, l1VLayout, Arch::PositionL1{});
        auto l1VTile = GetTile(l1VTensorTla,
            tla::MakeCoord(0, 0), tla::MakeShape(validSize, D));
        auto gmVTile = GetTile(gmVTensor,
            tla::MakeCoord(0, 0), tla::MakeShape(validSize, D));

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        copyGmToL1V(l1VTile, gmVTile);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID2);
    }

    template <class TensorP, class TensorO>
    __aicore__ inline
    void ComputePV(TensorP &ubPTensor, TensorO &ubOTensor,
                   uint32_t groupSize, uint32_t validSize, uint32_t D)
    {
        uint32_t M = groupSize;
        uint32_t N = D;
        uint32_t K = validSize;

        static constexpr uint32_t ELE_PER_C0 = 32 / sizeof(ElementP);

        // UB → L1 for P using raw DataCopy with Nd2NzParams
        uint32_t pRows = M;
        uint32_t pCols = K;
        uint32_t pColsAligned = RoundUp(pCols, ELE_PER_C0);
        AscendC::Nd2NzParams pCopyParams;
        pCopyParams.ndNum = 1;
        pCopyParams.nValue = pRows;
        pCopyParams.dValue = pCols;
        pCopyParams.srcNdMatrixStride = 0;
        pCopyParams.srcDValue = pColsAligned;
        pCopyParams.dstNzC0Stride = RoundUp(pRows, (uint32_t)16);
        pCopyParams.dstNzNStride = 1;
        pCopyParams.dstNzMatrixStride = 0;

        AscendC::DataCopy(l1PTensor_, ubPTensor.data(), pCopyParams);
        AscendC::PipeBarrier<PIPE_ALL>();

        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorO>;
        CopyL0CToDst copyL0CToUB;

        auto l1PLayout = tla::MakeLayout<ElementP, LayoutTagL1A>(M, K);
        auto l1PTensorTla = tla::MakeTensor(l1PTensor_, l1PLayout, Arch::PositionL1{});
        auto l1VLayout = tla::MakeLayout<ElementV, LayoutTagL1B>(K, N);
        auto l1VTensorTla = tla::MakeTensor(l1VTensor_, l1VLayout, Arch::PositionL1{});

        uint32_t nL0Num = CeilDiv(N, L0_TILE_N);
        uint32_t mL0Num = CeilDiv(M, L0_TILE_M);
        uint32_t kL0Num = CeilDiv(K, L0_TILE_K);

        for (uint32_t nL0Itr = 0; nL0Itr < nL0Num; nL0Itr++) {
            uint32_t nAct = (nL0Itr == nL0Num - 1) ? (N - nL0Itr * L0_TILE_N) : L0_TILE_N;
            uint32_t l0CBufId = nL0Itr % L0_STAGES;
            uint32_t l0CEventId = l0CBufId;

            auto l0CLayout = tla::MakeLayoutL0C(M, nAct);
            auto l0CTensorTla = tla::MakeTensor(l0CTensor_[l0CBufId], l0CLayout, Arch::PositionL0C{});

            for (uint32_t mL0Itr = 0; mL0Itr < mL0Num; mL0Itr++) {
                uint32_t mAct = (mL0Itr == mL0Num - 1) ? (M - mL0Itr * L0_TILE_M) : L0_TILE_M;
                auto l0CTile = GetTile(l0CTensorTla,
                    tla::MakeCoord(mL0Itr * L0_TILE_M, 0), tla::MakeShape(mAct, nAct));

                for (uint32_t kL0Itr = 0; kL0Itr < kL0Num; kL0Itr++) {
                    uint32_t kAct = (kL0Itr == kL0Num - 1) ? (K - kL0Itr * L0_TILE_K) : L0_TILE_K;
                    uint32_t l0ABufId = kL0Itr % L0_STAGES;
                    uint32_t l0BBufId = kL0Itr % L0_STAGES;
                    uint32_t l0AEventId = l0ABufId;
                    uint32_t l0BEventId = l0BBufId + 2;

                    auto l1PSubTile = GetTile(l1PTensorTla,
                        tla::MakeCoord(mL0Itr * L0_TILE_M, kL0Itr * L0_TILE_K),
                        tla::MakeShape(mAct, kAct));
                    auto l0ALayout = tla::MakeLayout<ElementP, LayoutTagL0A>(mAct, kAct);
                    auto l0ATensorTla = tla::MakeTensor(
                        l0ATensor_[l0ABufId], l0ALayout, Arch::PositionL0A{});

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    copyL1ToL0A(l0ATensorTla, l1PSubTile);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);

                    auto l1VSubTile = GetTile(l1VTensorTla,
                        tla::MakeCoord(kL0Itr * L0_TILE_K, nL0Itr * L0_TILE_N),
                        tla::MakeShape(kAct, nAct));
                    auto l0BLayout = tla::MakeLayout<ElementV, LayoutTagL0B>(kAct, nAct);
                    auto l0BTensorTla = tla::MakeTensor(
                        l0BTensor_[l0BBufId], l0BLayout, Arch::PositionL0B{});

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                    copyL1ToL0B(l0BTensorTla, l1VSubTile);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);

                    bool initMmad = (kL0Itr == 0);
                    uint32_t mAligned = RoundUp(mAct, 16);

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                    if (mL0Itr == 0 && kL0Itr == 0) {
                        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
                    }
                    tileMmad(l0CTile, l0ATensorTla, l0BTensorTla,
                             mAligned, nAct, kAct, initMmad);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                }
            }

            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);

            uint32_t mAligned = RoundUp(M, 16);
            uint32_t nAligned = RoundUp(nAct, 16);
            auto ubOTile = GetTile(ubOTensor,
                tla::MakeCoord(0, nL0Itr * L0_TILE_N), tla::MakeShape(mAligned, nAligned));
            copyL0CToUB(ubOTile, l0CTensorTla);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
    }
};

}  // namespace SparseAttnPrefill

#endif  // BLOCK_MMAD_PREFILL_PV_HPP
