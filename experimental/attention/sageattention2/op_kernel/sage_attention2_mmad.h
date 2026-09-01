/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software and is licensed under the CANN Open Software License Agreement
 * Version 2.0. THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND.
 */

#ifndef SAGE_ATTENTION2_MMAD_H_
#define SAGE_ATTENTION2_MMAD_H_

// CATLASS 3510 does not provide the split-M, per-tensor L0C(int32)->UB(fp16) specialization even though
// CANN 9.2 exposes the corresponding C310 Fixpipe mode. Keep the extension local to SageAttention2 while
// the final architecture experiment is isolated.
namespace Catlass::Gemm::Tile {

template <class TensorSrc_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<Catlass::Arch::Ascend950, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<half>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    CopyL0CToUBMode::SPLIT_M, ScaleGranularity::PER_TENSOR, ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ElementSrc = typename TensorSrc_::Element;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor,
        uint64_t deqScalar, uint8_t unitFlag = 0)
    {
        const uint32_t rows = tla::get<0>(dstTensor.originShape());
        const uint32_t firstHalf = (rows + 1U) >> 1U;
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> params;
        params.nSize = RoundUp(tla::get<1>(dstTensor.originShape()), 16U);
        params.mSize = firstHalf;
        params.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        params.dstStride = tla::get<0>(dstTensor.stride());
        params.quantPre = QuantMode_t::DEQF16;
        params.deqScalar = deqScalar;
        params.reluEn = ReluEnable_;
        params.unitFlag = unitFlag;
        params.dualDstCtl = 0;
        params.subBlockId = 0;
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::Fixpipe<half, ElementSrc, CFG_ROW_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], params);

        const uint32_t secondHalf = rows - firstHalf;
        if (secondHalf != 0U) {
            params.mSize = secondHalf;
            params.subBlockId = 1;
            auto secondSrcOffset = srcTensor.layout()(tla::MakeCoord(firstHalf, 0U));
            AscendC::Fixpipe<half, ElementSrc, CFG_ROW_MAJOR_UB>(
                dstTensor.data()[dstOffset], srcTensor.data()[secondSrcOffset], params);
        }
    }
};

} // namespace Catlass::Gemm::Tile

namespace SageAttention2Kernel {

// The stock CATLASS FAI blocks require identical L1/L0 tile shapes. Sage uses a 256-column KV tile to
// amortize the vector-side online-softmax work, while retaining 128-column L0 operands so QK and PV do not
// each consume all of L0A/L0B. These two blocks are intentionally local to SageAttention2.
template <class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_,
    class TileCopy_, class TileMmad_>
class SageMmadQK {
public:
    using ArchTag = Catlass::Arch::Ascend950;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;
    using TileCopy = TileCopy_;
    using TileMmad = TileMmad_;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static constexpr uint32_t STAGES = 2U;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});
    static constexpr uint32_t L1A_TILE_SIZE = L1_TILE_M * L1_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L1B_TILE_SIZE = L0_TILE_N * L0_TILE_K * sizeof(ElementB);
    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L0_TILE_M * L0_TILE_N * sizeof(ElementAccumulator);
    static constexpr uint32_t L0A_STAGE_STRIDE = ArchTag::L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_STAGE_STRIDE = ArchTag::L0B_SIZE / STAGES;
    static constexpr uint32_t BLOCK_L1_SIZE = (L1A_TILE_SIZE + L1B_TILE_SIZE) * STAGES;
    static constexpr uint32_t BLOCK_L0C_SIZE = L0C_TILE_SIZE * STAGES;

    static_assert(L1_TILE_M == L0_TILE_M && L1_TILE_K == L0_TILE_K && L1_TILE_N % L0_TILE_N == 0U,
        "Sage QK requires only the N axis to be split in L0");
    static_assert(L0A_TILE_SIZE * STAGES <= ArchTag::L0A_SIZE, "Sage QK exceeds L0A");
    static_assert(L0B_TILE_SIZE * STAGES <= ArchTag::L0B_SIZE, "Sage QK exceeds L0B");

    __aicore__ inline SageMmadQK(
        Catlass::Arch::Resource<ArchTag> &resource, uint32_t &l1Offset, uint32_t &l0COffset)
    {
        const uint32_t l1AOffset = l1Offset;
        const uint32_t l1BOffset = l1AOffset + L1A_TILE_SIZE * STAGES;
        for (uint32_t i = 0U; i < STAGES; ++i) {
            l1A_[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_TILE_SIZE * i);
            l1B_[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_TILE_SIZE * i);
            // QK and PV time-share L0A/L0B and use the same event id for each pipeline stage. Keep the
            // physical stage regions identical even though a QK tile is smaller; otherwise PV stage 0
            // overlaps QK stage 1 without being protected by stage 1's M_MTE1 token.
            l0A_[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_STAGE_STRIDE * i);
            l0B_[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_STAGE_STRIDE * i);
            l0C_[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(l0COffset + L0C_TILE_SIZE * i);
            l1AEvent_[i] = i;
            l1BEvent_[i] = i + STAGES;
            l0AEvent_[i] = i;
            l0BEvent_[i] = i + STAGES;
            l0CEvent_[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEvent_[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEvent_[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEvent_[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEvent_[i]);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEvent_[i]);
        }
        l1Offset += BLOCK_L1_SIZE;
        l0COffset += BLOCK_L0C_SIZE;
    }

    __aicore__ inline ~SageMmadQK()
    {
        for (uint32_t i = 0U; i < STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEvent_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEvent_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEvent_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEvent_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEvent_[i]);
        }
    }

    template <class TensorA, class TensorB, class TensorC, class Shape>
    __aicore__ inline void operator()(TensorA &a, TensorB &b, TensorC &c,
        AscendC::GlobalTensor<int32_t> &, Shape &shape, int64_t taskId, int32_t, bool first, bool last,
        uint64_t deqScalar)
    {
        const int32_t blockM = tla::get<0>(shape);
        const int32_t blockN = tla::get<1>(shape);
        const int32_t blockK = tla::get<2>(shape);
        const uint32_t stage = l0CId_;

        if (first) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEvent_[l1AId_]);
            auto l1ALayout = tla::MakeLayout<ElementA, LayoutTagL1A>(blockM, blockK);
            auto l1ATensor = tla::MakeTensor(l1A_[l1AId_], l1ALayout, Catlass::Arch::PositionL1{});
            using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<TensorA>;
            CopyGmToL1A copyGmToL1A;
            copyGmToL1A(l1ATensor, a);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEvent_[l1AId_]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEvent_[l1AId_]);
        }
        auto l1ALayout = tla::MakeLayout<ElementA, LayoutTagL1A>(blockM, blockK);
        auto l1ATensor = tla::MakeTensor(l1A_[l1AId_], l1ALayout, Catlass::Arch::PositionL1{});

        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<TensorB>;
        CopyGmToL1B copyGmToL1B;

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEvent_[stage]);
        auto l0ALayout = tla::MakeLayout<ElementA, LayoutTagL0A>(blockM, blockK);
        auto l0ATensor = tla::MakeTensor(l0A_[stage], l0ALayout, Catlass::Arch::PositionL0A{});
        const uint32_t zero = 0U;
        auto l1ATile = GetTile(l1ATensor, tla::MakeCoord(zero, zero), tla::MakeShape(blockM, blockK));
        copyL1ToL0A_(l0ATensor, l1ATile);

        const uint32_t nLoops = (blockN + L0_TILE_N - 1U) / L0_TILE_N;
        for (uint32_t n = 0U; n < nLoops; ++n) {
            const uint32_t tileN = n + 1U == nLoops ? blockN - n * L0_TILE_N : L0_TILE_N;
            // Keep low/high N subtiles on fixed L0C stages. The low stage starts Fixpipe first and is
            // therefore normally released before the high stage; the next work item can reuse it without
            // being forced to wait for the previous high-subtile Fixpipe tail.
            const uint32_t cStage = n & 1U;
            const uint32_t bStage = l1BId_;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEvent_[bStage]);
            auto l1BLayout = tla::MakeLayout<ElementB, LayoutTagL1B>(blockK, tileN);
            auto l1BTensor = tla::MakeTensor(l1B_[bStage], l1BLayout, Catlass::Arch::PositionL1{});
            auto bTile = GetTile(b, tla::MakeCoord(zero, n * L0_TILE_N), tla::MakeShape(blockK, tileN));
            copyGmToL1B(l1BTensor, bTile);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEvent_[bStage]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEvent_[bStage]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEvent_[stage]);
            auto l0BLayout = tla::MakeLayout<ElementB, LayoutTagL0B>(blockK, tileN);
            auto l0BTensor = tla::MakeTensor(l0B_[stage], l0BLayout, Catlass::Arch::PositionL0B{});
            auto l1BTile = GetTile(l1BTensor, tla::MakeCoord(zero, zero), tla::MakeShape(blockK, tileN));
            copyL1ToL0B_(l0BTensor, l1BTile);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEvent_[bStage]);
            l1BId_ ^= 1U;

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEvent_[cStage]);
            auto l0CLayout = tla::MakeLayoutL0C(blockM, tileN);
            auto l0CTensor = tla::MakeTensor(l0C_[cStage], l0CLayout, Catlass::Arch::PositionL0C{});
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0CEvent_[cStage]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0CEvent_[cStage]);
            tileMmad_(l0CTensor, l0ATensor, l0BTensor, blockM == 1 ? 16U : blockM, tileN, blockK, true);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEvent_[stage]);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEvent_[cStage]);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEvent_[cStage]);
            if (n == 0U) {
                AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(MM1_FREE[taskId]);
                AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16U + MM1_FREE[taskId]);
            }
            auto cTile = GetTile(c, tla::MakeCoord(0, n * L0_TILE_N), tla::MakeShape(blockM, tileN));
            using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<decltype(cTile)>;
            CopyL0CToDst copyL0CToDst;
            copyL0CToDst(cTile, l0CTensor, deqScalar);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEvent_[cStage]);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEvent_[stage]);
        if (last) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEvent_[l1AId_]);
            l1AId_ ^= 1U;
        }
        l0CId_ ^= 1U;
    }

private:
    static constexpr uint32_t SYNC_MODE = 4U;
    static constexpr uint32_t MM1_FREE[2] = {9U, 10U};
    AscendC::LocalTensor<ElementA> l1A_[STAGES];
    AscendC::LocalTensor<ElementB> l1B_[STAGES];
    AscendC::LocalTensor<ElementA> l0A_[STAGES];
    AscendC::LocalTensor<ElementB> l0B_[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0C_[STAGES];
    int32_t l1AEvent_[STAGES];
    int32_t l1BEvent_[STAGES];
    int32_t l0AEvent_[STAGES];
    int32_t l0BEvent_[STAGES];
    int32_t l0CEvent_[STAGES];
    uint32_t l1AId_ = 0U;
    uint32_t l1BId_ = 0U;
    uint32_t l0CId_ = 0U;
    CopyL1ToL0A copyL1ToL0A_;
    CopyL1ToL0B copyL1ToL0B_;
    TileMmad tileMmad_;
};

// Dn QK computes K * Q^T. Splitting the N(Q) axis gives each AIV every K row and half of the Q rows,
// so softmax reductions become lane-wise Max/Add operations instead of horizontal vector reductions.
// DEQF16 cannot legally use dual-destination Fixpipe on 950, therefore the two Q halves are routed with
// two single-destination Fixpipe instructions.
template <class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_,
    class TileCopy_, class TileMmad_>
class SageMmadQKDn {
public:
    using ArchTag = Catlass::Arch::Ascend950;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;
    using TileCopy = TileCopy_;
    using TileMmad = TileMmad_;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static constexpr uint32_t STAGES = 2U;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});
    static constexpr uint32_t L1A_TILE_SIZE = L1_TILE_M * L1_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L1B_TILE_SIZE = L1_TILE_K * L1_TILE_N * sizeof(ElementB);
    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L0_TILE_M * L0_TILE_N * sizeof(ElementAccumulator);
    static constexpr uint32_t L0A_STAGE_STRIDE = ArchTag::L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_STAGE_STRIDE = ArchTag::L0B_SIZE / STAGES;
    static constexpr uint32_t BLOCK_L1_SIZE = (L1A_TILE_SIZE + L1B_TILE_SIZE) * STAGES;
    static constexpr uint32_t BLOCK_L0C_SIZE = L0C_TILE_SIZE * STAGES;

    static_assert(L1_TILE_M % L0_TILE_M == 0U && L1_TILE_N == L0_TILE_N && L1_TILE_K == L0_TILE_K,
        "Sage Dn QK requires only the K-token/M axis to be split in L0");
    static_assert(L0A_TILE_SIZE * STAGES <= ArchTag::L0A_SIZE, "Sage Dn QK exceeds L0A");
    static_assert(L0B_TILE_SIZE * STAGES <= ArchTag::L0B_SIZE, "Sage Dn QK exceeds L0B");

    __aicore__ inline SageMmadQKDn(
        Catlass::Arch::Resource<ArchTag> &resource, uint32_t &l1Offset, uint32_t &l0COffset)
    {
        const uint32_t l1AOffset = l1Offset;
        const uint32_t l1BOffset = l1AOffset + L1A_TILE_SIZE * STAGES;
        for (uint32_t i = 0U; i < STAGES; ++i) {
            l1A_[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_TILE_SIZE * i);
            l1B_[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_TILE_SIZE * i);
            // Match PV's full per-stage footprint so a shared event id always owns one disjoint physical
            // half of L0A/L0B. Compact QK packing aliases PV stage 0 with QK stage 1 and races MTE1 vs Cube.
            l0A_[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_STAGE_STRIDE * i);
            l0B_[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_STAGE_STRIDE * i);
            l0C_[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(l0COffset + L0C_TILE_SIZE * i);
            l1AEvent_[i] = i;
            l1BEvent_[i] = i + STAGES;
            l0AEvent_[i] = i;
            l0BEvent_[i] = i + STAGES;
            l0CEvent_[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEvent_[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEvent_[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEvent_[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEvent_[i]);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEvent_[i]);
        }
        l1Offset += BLOCK_L1_SIZE;
        l0COffset += BLOCK_L0C_SIZE;
    }

    __aicore__ inline ~SageMmadQKDn()
    {
        for (uint32_t i = 0U; i < STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEvent_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEvent_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEvent_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEvent_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEvent_[i]);
        }
    }

    template <class TensorA, class TensorB, class TensorC, class Shape>
    __aicore__ inline void operator()(TensorA &k, TensorB &q, TensorC &c,
        AscendC::GlobalTensor<int32_t> &, Shape &shape, int64_t taskId, int32_t, bool first, bool last,
        uint64_t deqScalar)
    {
        const uint32_t blockM = tla::get<0>(shape); // K rows
        const uint32_t blockN = tla::get<1>(shape); // Q rows
        const uint32_t blockK = tla::get<2>(shape);
        const uint32_t halfN = (blockN + 1U) >> 1U;
        const uint32_t workStage = l1AId_;

        if constexpr (SAGE_ATTENTION2_QK_PROFILE != 4U && SAGE_ATTENTION2_QK_PROFILE != 6U &&
            SAGE_ATTENTION2_QK_PROFILE != 7U && SAGE_ATTENTION2_QK_PROFILE != 8U) {
            if (first) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEvent_[l1BId_]);
                auto l1BLayout = tla::MakeLayout<ElementB, LayoutTagL1B>(blockK, blockN);
                auto l1BTensor = tla::MakeTensor(l1B_[l1BId_], l1BLayout, Catlass::Arch::PositionL1{});
                using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<TensorB>;
                CopyGmToL1B copyGmToL1B;
                copyGmToL1B(l1BTensor, q);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEvent_[l1BId_]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEvent_[l1BId_]);
            }
        }
        auto l1BLayout = tla::MakeLayout<ElementB, LayoutTagL1B>(blockK, blockN);
        auto l1BTensor = tla::MakeTensor(l1B_[l1BId_], l1BLayout, Catlass::Arch::PositionL1{});

        auto l1ALayout = tla::MakeLayout<ElementA, LayoutTagL1A>(blockM, blockK);
        auto l1ATensor = tla::MakeTensor(l1A_[workStage], l1ALayout, Catlass::Arch::PositionL1{});
        if constexpr (SAGE_ATTENTION2_QK_PROFILE != 5U && SAGE_ATTENTION2_QK_PROFILE != 6U &&
            SAGE_ATTENTION2_QK_PROFILE != 7U && SAGE_ATTENTION2_QK_PROFILE != 8U) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEvent_[workStage]);
            using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<TensorA>;
            CopyGmToL1A copyGmToL1A;
            copyGmToL1A(l1ATensor, k);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEvent_[workStage]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEvent_[workStage]);
        }

        const uint32_t mLoops = (blockM + L0_TILE_M - 1U) / L0_TILE_M;
        if constexpr (SAGE_ATTENTION2_QK_PROFILE < 3U) {
        for (uint32_t m = 0U; m < mLoops; ++m) {
            const uint32_t tileM = m + 1U == mLoops ? blockM - m * L0_TILE_M : L0_TILE_M;
            const uint32_t stage = m & 1U;
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEvent_[stage]);
            auto l0ALayout = tla::MakeLayout<ElementA, LayoutTagL0A>(tileM, blockK);
            auto l0ATensor = tla::MakeTensor(l0A_[stage], l0ALayout, Catlass::Arch::PositionL0A{});
            auto l1ATile = GetTile(l1ATensor, tla::MakeCoord(m * L0_TILE_M, 0U),
                tla::MakeShape(tileM, blockK));
            copyL1ToL0A_(l0ATensor, l1ATile);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEvent_[stage]);

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEvent_[stage]);
            auto l0BLayout = tla::MakeLayout<ElementB, LayoutTagL0B>(blockK, blockN);
            auto l0BTensor = tla::MakeTensor(l0B_[stage], l0BLayout, Catlass::Arch::PositionL0B{});
            auto l1BTile = GetTile(l1BTensor, tla::MakeCoord(0U, 0U), tla::MakeShape(blockK, blockN));
            copyL1ToL0B_(l0BTensor, l1BTile);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEvent_[stage]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEvent_[stage]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEvent_[stage]);
            if constexpr (SAGE_ATTENTION2_QK_PROFILE == 2U) {
                // Cumulative MTE2+MTE1 cut: close both operand events without issuing Cube/Fixpipe.
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEvent_[stage]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEvent_[stage]);
                if (m == 0U) {
                    AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(MM1_FREE[taskId]);
                    AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16U + MM1_FREE[taskId]);
                }
                continue;
            }

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEvent_[stage]);
            auto l0CLayout = tla::MakeLayoutL0C(tileM, blockN);
            auto l0CTensor = tla::MakeTensor(l0C_[stage], l0CLayout, Catlass::Arch::PositionL0C{});
            tileMmad_(l0CTensor, l0ATensor, l0BTensor, tileM, blockN, blockK, true);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEvent_[stage]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEvent_[stage]);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEvent_[stage]);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEvent_[stage]);
            if (m == 0U) {
                AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(MM1_FREE[taskId]);
                AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16U + MM1_FREE[taskId]);
            }

            if constexpr (SAGE_ATTENTION2_QK_PROFILE == 1U) {
                // Cumulative through-MMAD cut. M_FIX above drains Cube; publish the free L0C event from
                // the Fixpipe side without moving accumulator data to UB.
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEvent_[stage]);
                continue;
            }

            AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> params;
            params.nSize = AscendC::CeilDivision(halfN, 16U) * 16U;
            params.mSize = tileM;
            params.srcStride = tla::get<1, 1>(l0CTensor.stride()) / tla::get<0, 0>(l0CTensor.stride());
            // Dn softmax always walks one 64-lane vector per AIV. Keep the K-row stride fixed at 64
            // even when the final Q tile contains only 64 rows (32 rows per AIV); otherwise adjacent
            // K rows are packed with stride 32 and the vector path reads every other row as Q data.
            constexpr uint32_t DN_HALF_ROWS = 64U;
            params.dstStride = DN_HALF_ROWS;
            params.quantPre = QuantMode_t::DEQF16;
            params.deqScalar = deqScalar;
            params.dualDstCtl = 0;
            params.subBlockId = 0;
            auto dstOffset = static_cast<uint64_t>(m) * L0_TILE_M * DN_HALF_ROWS;
            AscendC::Fixpipe<ElementC, ElementAccumulator, CFG_ROW_MAJOR_UB>(
                c.data()[dstOffset], l0CTensor.data()[0], params);
            params.subBlockId = 1;
            auto secondSrcOffset = l0CTensor.layout()(tla::MakeCoord(0U, halfN));
            AscendC::Fixpipe<ElementC, ElementAccumulator, CFG_ROW_MAJOR_UB>(
                c.data()[dstOffset], l0CTensor.data()[secondSrcOffset], params);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEvent_[stage]);
        }
        } else {
            // MTE2-only cumulative cut. Preserve one free-token consume per work item so repeated stages
            // exercise the same AIC/AIV producer-consumer protocol as the full QK path.
            AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(MM1_FREE[taskId]);
            AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16U + MM1_FREE[taskId]);
        }

        if constexpr (SAGE_ATTENTION2_QK_PROFILE != 5U && SAGE_ATTENTION2_QK_PROFILE != 6U &&
            SAGE_ATTENTION2_QK_PROFILE != 7U && SAGE_ATTENTION2_QK_PROFILE != 8U) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEvent_[workStage]);
            l1AId_ ^= 1U;
        }
        if constexpr (SAGE_ATTENTION2_QK_PROFILE != 4U && SAGE_ATTENTION2_QK_PROFILE != 6U &&
            SAGE_ATTENTION2_QK_PROFILE != 7U && SAGE_ATTENTION2_QK_PROFILE != 8U) {
            if (last) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEvent_[l1BId_]);
                l1BId_ ^= 1U;
            }
        }
    }

private:
    static constexpr uint32_t SYNC_MODE = 4U;
    static constexpr uint32_t MM1_FREE[2] = {9U, 10U};
    AscendC::LocalTensor<ElementA> l1A_[STAGES];
    AscendC::LocalTensor<ElementB> l1B_[STAGES];
    AscendC::LocalTensor<ElementA> l0A_[STAGES];
    AscendC::LocalTensor<ElementB> l0B_[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0C_[STAGES];
    int32_t l1AEvent_[STAGES];
    int32_t l1BEvent_[STAGES];
    int32_t l0AEvent_[STAGES];
    int32_t l0BEvent_[STAGES];
    int32_t l0CEvent_[STAGES];
    uint32_t l1AId_ = 0U;
    uint32_t l1BId_ = 0U;
    CopyL1ToL0A copyL1ToL0A_;
    CopyL1ToL0B copyL1ToL0B_;
    TileMmad tileMmad_;
};

template <class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_,
    class TileCopy_, class TileMmad_, bool ENABLE_DN_ = false>
class SageMmadPV {
public:
    using ArchTag = Catlass::Arch::Ascend950;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;
    using TileCopy = TileCopy_;
    using TileMmad = TileMmad_;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static constexpr uint32_t STAGES = 2U;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<2>(L0TileShape{});
    static constexpr uint32_t L1B_TILE_SIZE = L0_TILE_K * L0_TILE_N * sizeof(ElementB);
    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L0_TILE_M * L0_TILE_N * sizeof(ElementAccumulator);
    static constexpr uint32_t BLOCK_L1_SIZE = L1B_TILE_SIZE * STAGES;
    static constexpr uint32_t BLOCK_L0C_SIZE = L0C_TILE_SIZE * STAGES;

    static_assert(L1_TILE_M == L0_TILE_M && L1_TILE_N == L0_TILE_N && L1_TILE_K % L0_TILE_K == 0U,
        "Sage PV requires only the K axis to be split in L0");
    static_assert(L0A_TILE_SIZE * STAGES <= ArchTag::L0A_SIZE, "Sage PV exceeds L0A");
    static_assert(L0B_TILE_SIZE * STAGES <= ArchTag::L0B_SIZE, "Sage PV exceeds L0B");

    __aicore__ inline SageMmadPV(
        Catlass::Arch::Resource<ArchTag> &resource, uint32_t &l1Offset, uint32_t &l0COffset)
    {
        for (uint32_t i = 0U; i < STAGES; ++i) {
            l1B_[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1Offset + L1B_TILE_SIZE * i);
            l0A_[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_TILE_SIZE * i);
            l0B_[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_TILE_SIZE * i);
            l0C_[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(l0COffset + L0C_TILE_SIZE * i);
            l1BEvent_[i] = 4U + i + STAGES;
            l0AEvent_[i] = i;
            l0BEvent_[i] = i + STAGES;
            l0CEvent_[i] = 4U + i;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEvent_[i]);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEvent_[i]);
        }
        l1Offset += BLOCK_L1_SIZE;
        l0COffset += BLOCK_L0C_SIZE;
    }

    __aicore__ inline ~SageMmadPV()
    {
        for (uint32_t i = 0U; i < STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEvent_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEvent_[i]);
        }
    }

    template <class TensorA, class TensorB, class TensorC, class Shape>
    __aicore__ inline void operator()(TensorA &a, TensorB &b, TensorC &c,
        AscendC::GlobalTensor<int32_t> &, Shape &shape, int64_t taskId, int32_t)
    {
        const int32_t blockM = tla::get<0>(shape);
        const int32_t blockN = tla::get<1>(shape);
        const int32_t blockK = tla::get<2>(shape);
        const uint32_t stage = l0CId_;

        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<TensorB>;
        CopyGmToL1B copyGmToL1B;

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEvent_[stage]);
        auto l0CLayout = tla::MakeLayoutL0C(blockM, blockN);
        auto l0CTensor = tla::MakeTensor(l0C_[stage], l0CLayout, Catlass::Arch::PositionL0C{});
        const uint32_t kLoops = (blockK + L0_TILE_K - 1U) / L0_TILE_K;
        const uint32_t zero = 0U;
        for (uint32_t k = 0U; k < kLoops; ++k) {
            const uint32_t tileK = k + 1U == kLoops ? blockK - k * L0_TILE_K : L0_TILE_K;
            const uint32_t bStage = l1BId_;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEvent_[bStage]);
            auto l1BLayout = tla::MakeLayout<ElementB, LayoutTagL1B>(tileK, blockN);
            auto l1BTensor = tla::MakeTensor(l1B_[bStage], l1BLayout, Catlass::Arch::PositionL1{});
            auto gmBTile = GetTile(b, tla::MakeCoord(k * L0_TILE_K, zero), tla::MakeShape(tileK, blockN));
            copyGmToL1B(l1BTensor, gmBTile);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEvent_[bStage]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEvent_[bStage]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEvent_[stage]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEvent_[stage]);
            auto l0ALayout = tla::MakeLayout<ElementA, LayoutTagL0A>(blockM, tileK);
            auto l0ATensor = tla::MakeTensor(l0A_[stage], l0ALayout, Catlass::Arch::PositionL0A{});
            if constexpr (ENABLE_DN_) {
                // Product Dn writes P in a deconflicted K-major layout. The transpose bit is the contract
                // between that vector pack and MM2; for B8, kStep counts 32-element C0 blocks.
                AscendC::LoadData2DParamsV2 params;
                params.mStartPosition = 0U;
                params.kStartPosition = 0U;
                params.ifTranspose = true;
                params.mStep = AscendC::CeilDivision(tileK, 16U);
                params.mStep = AscendC::CeilDivision(params.mStep, 2U) * 2U;
                params.kStep = AscendC::CeilDivision(blockM, 32U);
                // Dn vector output is padded to the full L1 K tile even for a K tail, matching the
                // product fullquant contract. Load only tileK rows while retaining the fixed source stride.
                params.srcStride = AscendC::CeilDivision(L1_TILE_K, 64U) * 4U;
                params.dstStride = AscendC::CeilDivision(blockM, 16U);
                AscendC::LoadData(l0ATensor.data(), a.data()[k * L0_TILE_K * L1_TILE_M], params);
            } else {
                // zN stores each 128-column chunk contiguously as M*128 bytes. Construct the chunk tensor
                // from its physical base so the high half does not inherit the 256-column origin stride.
                auto aChunkLayout = tla::MakeLayout<ElementA, LayoutTagL1A>(blockM, tileK);
                auto aBase = tla::MakeTensor(a.data()[k * L0_TILE_K * L1_TILE_M], aChunkLayout,
                    Catlass::Arch::PositionL1{});
                auto aTile = GetTile(aBase, tla::MakeCoord(zero, zero), tla::MakeShape(blockM, tileK));
                copyL1ToL0A_(l0ATensor, aTile);
            }
            auto l0BLayout = tla::MakeLayout<ElementB, LayoutTagL0B>(tileK, blockN);
            auto l0BTensor = tla::MakeTensor(l0B_[stage], l0BLayout, Catlass::Arch::PositionL0B{});
            auto l1BTile = GetTile(l1BTensor, tla::MakeCoord(zero, zero), tla::MakeShape(tileK, blockN));
            copyL1ToL0B_(l0BTensor, l1BTile);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEvent_[bStage]);
            l1BId_ ^= 1U;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0CEvent_[stage]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0CEvent_[stage]);
            tileMmad_(l0CTensor, l0ATensor, l0BTensor, blockM == 1 ? 16U : blockM, blockN, tileK, k == 0U);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEvent_[stage]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEvent_[stage]);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEvent_[stage]);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEvent_[stage]);
        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(MM2_FREE[taskId]);
        AscendC::CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16U + MM2_FREE[taskId]);
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorC>;
        CopyL0CToDst copyL0CToDst;
        copyL0CToDst(c, l0CTensor);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEvent_[stage]);
        l0CId_ ^= 1U;
    }

private:
    static constexpr uint32_t SYNC_MODE = 4U;
    static constexpr uint32_t MM2_FREE[2] = {7U, 8U};
    AscendC::LocalTensor<ElementB> l1B_[STAGES];
    AscendC::LocalTensor<ElementA> l0A_[STAGES];
    AscendC::LocalTensor<ElementB> l0B_[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0C_[STAGES];
    int32_t l1BEvent_[STAGES];
    int32_t l0AEvent_[STAGES];
    int32_t l0BEvent_[STAGES];
    int32_t l0CEvent_[STAGES];
    uint32_t l1BId_ = 0U;
    uint32_t l0CId_ = 0U;
    CopyL1ToL0A copyL1ToL0A_;
    CopyL1ToL0B copyL1ToL0B_;
    TileMmad tileMmad_;
};

} // namespace SageAttention2Kernel

#endif // SAGE_ATTENTION2_MMAD_H_
