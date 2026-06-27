/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Prefill QK matmul (KV-centric). FixPipe / cross-core sync aligned with
 * block_mmad_qk_arch35_ABf16_C_to_UB.hpp: S is split to two AIV UB buffers.
 */
#ifndef GEMM_BLOCK_MMAD_QK_PREFILL_ARCH35_HPP
#define GEMM_BLOCK_MMAD_QK_PREFILL_ARCH35_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/arch/cross_core_sync.hpp"
#include "../../../attn_infra/coord.hpp"
#include "../../../attn_infra/gemm/dispatch_policy.hpp"
#include "../../../attn_infra/gemm/tile_common/tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/tile_mmad.hpp"
#include "../../../attn_infra/layout/layout.hpp"
#include "../../../tla/layout.hpp"
#include "../../../tla/tensor.hpp"

namespace NpuArch::Gemm::Block {

template <class ElementQ_, class ElementK_, class ElementS_>
struct BlockMmadQKPrefillArch35 {
    using DispatchPolicy = MmadAtlasA5PrefillQK;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementQ = ElementQ_;
    using ElementK = ElementK_;
    using ElementS = ElementS_;
    using ElementAccumulator = float;

    using LayoutTagQ = layout::RowMajor;
    using LayoutTagK = layout::ColumnMajor;
    using LayoutTagS = layout::RowMajor;

    using TileCopy = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementQ, LayoutTagQ, ElementK, LayoutTagK,
        ElementS, LayoutTagS, void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    using TileMmad = Gemm::Tile::TileMmadTla<ArchTag, ElementQ, LayoutTagL1A>;

    static constexpr uint32_t L0_STAGES = 2;
    static constexpr uint32_t L0_TILE_M = 128;
    static constexpr uint32_t L0_TILE_N = 128;
    static constexpr uint32_t L0_TILE_K = 128;
    static constexpr uint32_t L0A_BUF_SIZE = ArchTag::L0A_SIZE / L0_STAGES;
    static constexpr uint32_t L0B_BUF_SIZE = ArchTag::L0B_SIZE / L0_STAGES;
    static constexpr uint32_t L0C_HALF_SIZE = ArchTag::L0C_SIZE / 2;
    static constexpr uint32_t L0C_BUF_SIZE = L0C_HALF_SIZE / L0_STAGES;
    static constexpr uint32_t L1_Q_STAGES = 1;
    static constexpr uint32_t V0_V1_FLAG_ID_OFFSET = 16;

    AscendC::LocalTensor<ElementK> l1KTensor_;
    AscendC::LocalTensor<ElementQ> l1QTensor_[L1_Q_STAGES];
    AscendC::LocalTensor<ElementQ> l0ATensor_[L0_STAGES];
    AscendC::LocalTensor<ElementK> l0BTensor_[L0_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor_[L0_STAGES];

    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    TileMmad tileMmad;

    uint32_t l1KBufBytes_;

    __aicore__ inline
    BlockMmadQKPrefillArch35() {}

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        if constexpr (MODE == 4U) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        if constexpr (MODE == 4U) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }

    __aicore__ inline
    void Init(Arch::Resource<ArchTag> &resource, uint32_t blockSize, uint32_t D)
    {
        uint32_t kBufBytes = blockSize * D * sizeof(ElementK);
        uint32_t qBufBytes = L0_TILE_M * D * sizeof(ElementQ);
        l1KBufBytes_ = kBufBytes;

        l1KTensor_ = resource.l1Buf.template GetBufferByByte<ElementK>(0);
        for (uint32_t i = 0; i < L1_Q_STAGES; i++) {
            l1QTensor_[i] = resource.l1Buf.template GetBufferByByte<ElementQ>(
                kBufBytes + qBufBytes * i);
        }
        for (uint32_t i = 0; i < L0_STAGES; i++) {
            l0ATensor_[i] = resource.l0ABuf.template GetBufferByByte<ElementQ>(L0A_BUF_SIZE * i);
            l0BTensor_[i] = resource.l0BBuf.template GetBufferByByte<ElementK>(L0B_BUF_SIZE * i);
            l0CTensor_[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_BUF_SIZE * i);
        }
    }

    template <class TensorK>
    __aicore__ inline
    void LoadKResident(TensorK &gmKTensor, uint32_t validSize, uint32_t D)
    {
        using CopyGmToL1K = typename TileCopy::template CopyGmToL1B<TensorK>;
        CopyGmToL1K copyGmToL1K;

        auto l1KLayout = tla::MakeLayout<ElementK, LayoutTagL1B>(D, validSize);
        auto l1KTensorTla = tla::MakeTensor(l1KTensor_, l1KLayout, Arch::PositionL1{});
        auto l1KTile = GetTile(l1KTensorTla,
            tla::MakeCoord(0, 0), tla::MakeShape(D, validSize));
        auto gmKTile = GetTile(gmKTensor,
            tla::MakeCoord(0, 0), tla::MakeShape(D, validSize));

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        copyGmToL1K(l1KTile, gmKTile);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    template <class TensorQ, class TensorS>
    __aicore__ inline
    void operator()(TensorQ &gmQTensor, TensorS &ubSTensor,
                    uint32_t groupSize, uint32_t validSize, uint32_t D,
                    Arch::CrossCoreFlag &mm1ToSmFlag)
    {
        uint32_t M = groupSize;
        uint32_t N = validSize;
        uint32_t K = D;

        using CopyGmToL1Q = typename TileCopy::template CopyGmToL1A<TensorQ>;
        CopyGmToL1Q copyGmToL1Q;
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorS>;
        CopyL0CToDst copyL0CToDstSub0;
        CopyL0CToDst copyL0CToDstSub1;

        auto l1QLayout = tla::MakeLayout<ElementQ, LayoutTagL1A>(M, K);
        auto l1QTensorTla = tla::MakeTensor(l1QTensor_[0], l1QLayout, Arch::PositionL1{});
        auto l1KLayout = tla::MakeLayout<ElementK, LayoutTagL1B>(K, N);
        auto l1KTensorTla = tla::MakeTensor(l1KTensor_, l1KLayout, Arch::PositionL1{});

        auto gmQTile = GetTile(gmQTensor,
            tla::MakeCoord(0, 0), tla::MakeShape(M, K));
        auto l1QTile = GetTile(l1QTensorTla,
            tla::MakeCoord(0, 0), tla::MakeShape(M, K));

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        copyGmToL1Q(l1QTile, gmQTile);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID1);

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

                    auto l1QSubTile = GetTile(l1QTensorTla,
                        tla::MakeCoord(mL0Itr * L0_TILE_M, kL0Itr * L0_TILE_K),
                        tla::MakeShape(mAct, kAct));
                    auto l0ALayout = tla::MakeLayout<ElementQ, LayoutTagL0A>(mAct, kAct);
                    auto l0ATensorTla = tla::MakeTensor(l0ATensor_[l0ABufId], l0ALayout, Arch::PositionL0A{});

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    copyL1ToL0A(l0ATensorTla, l1QSubTile);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);

                    auto l1KSubTile = GetTile(l1KTensorTla,
                        tla::MakeCoord(kL0Itr * L0_TILE_K, nL0Itr * L0_TILE_N),
                        tla::MakeShape(kAct, nAct));
                    auto l0BLayout = tla::MakeLayout<ElementK, LayoutTagL0B>(kAct, nAct);
                    auto l0BTensorTla = tla::MakeTensor(l0BTensor_[l0BBufId], l0BLayout, Arch::PositionL0B{});

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                    copyL1ToL0B(l0BTensorTla, l1KSubTile);
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

            if (nL0Itr == 0) {
                WaitCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);
            }
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);

            uint32_t mFixPAligned8 = RoundUp(M, 8U);
            uint32_t mPerSubCore = mFixPAligned8 / 2U;
            uint32_t nFixPAligned16 = RoundUp(nAct, 16U);
            auto ubSTile = GetTile(ubSTensor,
                tla::MakeCoord(0, nL0Itr * L0_TILE_N), tla::MakeShape(mPerSubCore, nFixPAligned16));
            auto l0CTileSub0 = GetTile(l0CTensorTla,
                tla::MakeCoord(0, 0), tla::MakeShape(mPerSubCore, nAct));
            auto l0CTileSub1 = GetTile(l0CTensorTla,
                tla::MakeCoord(mPerSubCore, 0), tla::MakeShape(mPerSubCore, nAct));
            copyL0CToDstSub0(ubSTile, l0CTileSub0, false);
            copyL0CToDstSub1(ubSTile, l0CTileSub1, true);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
        }
        SetCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
    }
};

}  // namespace NpuArch::Gemm::Block

#endif  // GEMM_BLOCK_MMAD_QK_PREFILL_ARCH35_HPP
