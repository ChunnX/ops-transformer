/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @brief matmul implementation for single transposed p&v base tile, with rescale fused on CUBE
 * This implementation is designed for the following senario (DN + FP8, rescaleOnCube on cube):
 * A full p base tile is loaded to L1 from UB(softmax), no workspace transit
 * A full v base tile is loaded to L1 from GM，relavent instructions launched before p base tile crossCore wait
 * PV is computed in transposed form: C' = (V)^T * (P)^T, C' is [D, qS] on L0C (split n by subcore: [D, qS/2])
 * rescale is fused as: C' += (GO_prev)^T * diag(dm/vScale), accumulated on the same L0C tile
 * FixPipe paths by tile type:
 *   first/middle: CFG_NZ -> L1 (GO^T keeps transposed & Nz, no conversion, deqScalar=vScale applied)
 *   last:         CFG_COLUMN_MAJOR (NZ2DN) -> UB (output is row-major [qS, D], deqScalar=vScale applied)
 * The whole module keeps the transposed view: A = (GO_i)^T is exactly the previous C output [D, qS].
 */
#ifndef GEMM_BLOCK_PV_TRANSPOSED_RESCALE_ARCH35_HPP
#define GEMM_BLOCK_PV_TRANSPOSED_RESCALE_ARCH35_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/arch/bsa_cross_core_sync.hpp"
#include "../../../attn_infra/bsa_coord.hpp"
#include "../../../attn_infra/gemm/bsa_gemm_dispatch_policy.hpp"
#include "../../../attn_infra/gemm/bsa_helper.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/gemm/block/block_mmad_arch35_utils.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_gemm_tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_tile_mmad.hpp"
#include "../../../tla/layout.hpp"
#include "../../../tla/tensor.hpp"

////////////////////////////////////////////////////////////////////

namespace NpuArch::Gemm::Block {
////////////////////////////////////////////////////////////////////

// partial specialization for performing (V)^T * (P)^T = C' [D, qS], fused with rescale (diag matmul)
template <
    class L1TileShape_,
    class L0TileShape_,
    class ElementA_,
    class ElementB_,
    class ElementC_,
    class ElementS_,
    class ElementBias_,
    class TileCopy_,
    class TileMmad_>
struct BlockMmadTla<
    MmadAtlasA5BsaPV<true, true>,  // transposedMm1 && rescaleOnCube, only for DN + rescale-on-cube
    L1TileShape_,
    L0TileShape_,
    ElementA_,
    ElementB_,
    ElementC_,
    ElementS_,
    ElementBias_,
    TileCopy_,
    TileMmad_>
{
public:
    using DispatchPolicy = MmadAtlasA5BsaPV<true, true>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;
    using ElementS = ElementS_;

    using TileMmad = TileMmad_;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;      // fp8（PV^T 用）
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;      // fp8（PV^T 用）

    // RE 段：GO^T/diag 为 fp32，PV 的 TileCopy（fp8）的 CopyL1ToL0A/L0B 不匹配 float 实参；
    // 用 float 的 PackedTileCopyTlaToUB 独立实例化（参考 interface 的 TileCopyPV 写法）；
    // LayoutTagA/B = zN（GO^T/diag 的 L1 物理布局，源已在 L1，无 GM→L1 环节）；
    // CopyMode/DEQUANT 仅影响 CopyL0CToDst（RE 不用），占位
    using TileCopyRescale = Gemm::Tile::PackedTileCopyTlaToUB<
        NpuArch::Arch::AtlasA5, float, NpuArch::layout::zN, float, NpuArch::layout::zN,
        float, NpuArch::layout::RowMajor, void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT, false,
        Gemm::Tile::ScaleGranularity::NO_QUANT>;
    using CopyL1ToL0ARescale = typename TileCopyRescale::CopyL1ToL0A;  // float zN→zN 恒等特化
    using CopyL1ToL0BRescale = typename TileCopyRescale::CopyL1ToL0B;  // float zN→nZ 转置特化

    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static constexpr uint32_t L0_STAGES = DispatchPolicy::L0_STAGES;
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape_{});
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / L0_STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / L0_STAGES;
    static constexpr uint32_t L0C_HALF_BUF_SIZE = ArchTag::L0C_SIZE / 2;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_HALF_BUF_SIZE / L0_STAGES;

    // rescale on cube: A = GO^T [D, qS], B = diag(dm/vScale) 块 [qS/2, qS/2]，fp32
    static constexpr uint32_t DIAG_BUF_NUM = 3;  // aligned with PRE_LAUNCH pipeline
    static constexpr uint32_t SUB_BLOCK_NUM = 2;      // diag 每 stage 双子核各一块（L1 共享，需分块排布）
    // 注：双子核块宽 = qS/2 为运行期值（qPerSubCore_ 成员），qS ≤ 128 且为偶数（tiling 约束）
    static constexpr bool rescaleOnCube = DispatchPolicy::rescaleOnCube;
    // GO^T 的 L1 写完成事件（FIX_M 的空闲 id，l0C 释放用 0-3）：
    // 中间块 FixPipe(CFG_NZ→L1) 后 set、非首块 RE 读 GO^T 前 wait——严格交替，单 id 即可
    static constexpr uint32_t GO_READY_EVENT_ID = 4;

    static constexpr bool FULL_QUANT_FP8 = AscendC::IsSameType<ElementA, fp8_e4m3fn_t>::value;

    __aicore__ inline
    BlockMmadTla(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart,
                 uint32_t l1DiagAddrStart,
                 Mm2L1TileHelper &mm2L1TileHelper)
    {
        l1ATileM = mm2L1TileHelper.mm2L1TileN;   // A = (V)^T: M = D
        l1BTileN = mm2L1TileHelper.mm2L1TileM;   // B = (P)^T: N = qS
        l1ATileK = mm2L1TileHelper.mm2L1TileKRight;  // A = (V)^T: K = kvS
        l1BTileK = mm2L1TileHelper.mm2L1TileKLeft;   // B = (P)^T: K = kvS
        qPerSubCore_ = l1BTileN / 2;   // diag 双子核块宽 = qS/2（运行期，与 S_trans 的 diagDmRound 一致）
        l1ABufNum = mm2L1TileHelper.vL1BufNum;   // A 读 V 段（gather 写入）
        l1BBufNum = mm2L1TileHelper.pL1BufNum;   // B 读 P 段（softmax 写入）
        // V 段在 P 段之后：l1BufAddrStart 指向 P 段，P 段后为 V 段
        uint32_t l1VAddrStart = l1BufAddrStart +
            mm2L1TileHelper.mm2L1TileM * mm2L1TileHelper.mm2L1TileKLeft * sizeof(ElementB) * mm2L1TileHelper.pL1BufNum;
        for (uint32_t i = 0; i < l1ABufNum; i++) {
            // A = (V)^T [D, kvS]：读 V 段（GM 为 RowMajor [kvS,D]，gather 经 ColumnMajor→nZ 特化落 L1，L1 内为 nZ 存储）
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(
                l1VAddrStart + l1ATileK * l1ATileM * sizeof(ElementA) * i);
        }
        for (uint32_t i = 0; i < l1BBufNum; i++) {
            // B = (P)^T [kvS, qS]：读 P 段（zN [qS,kvS] 存储）
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(
                l1BufAddrStart + l1BTileK * l1BTileN * sizeof(ElementB) * i);
        }
        // diag 块 fp32：每 stage 双子核各一块（L1 共享物理），3 份轮转。
        // 注：排布固定 64×64（与 kernel 分配/S_trans 写侧一致，full_quant.h:723），
        // 有效区 [qS/2, qS/2] 在块内前部（qS<128 时后部为预留，RE 的 B 子块不消费）
        for (uint32_t i = 0; i < DIAG_BUF_NUM; i++) {
            l1DiagTensor[i] = resource.l1Buf.template GetBufferByByte<float>(
                l1DiagAddrStart +
                SUB_BLOCK_NUM * 64 * 64 * sizeof(float) * i);
        }
        for (uint32_t i = 0; i < L0_STAGES; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(
                L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(
                L0B_PINGPONG_BUF_SIZE * i);
            // RE 段 float buffer：复用同一偏移（float [128,64]=32KB = L0A_PINGPONG 满；双子核块 [64,64]×2=32KB = L0B_PINGPONG 满；
            // qS<128 时按运行期 qS/2 缩小，均 ≤ 32KB；与 PV^T 段 fp8（16KB）同 stage 不同时使用）
            l0ARescaleTensor[i] = resource.l0ABuf.template GetBufferByByte<float>(
                L0A_PINGPONG_BUF_SIZE * i);
            l0BRescaleTensor[i] = resource.l0BBuf.template GetBufferByByte<float>(
                L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(
                L0C_HALF_BUF_SIZE + L0C_PINGPONG_BUF_SIZE * i);
        }
    }

    /// Destructor
    __aicore__ inline
    ~BlockMmadTla() {}

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC set for 2 AIVs seperately
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
        // in mode 4, AIC wait for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }

    __aicore__ inline
    uint32_t GetCurLoopCounter(uint32_t outterLoopItr, uint32_t curLoopNum, uint32_t curLoopItr)
    {
        return outterLoopItr * curLoopNum + curLoopItr;
    }

    template <class TensorB, class TensorL1B>
    __aicore__ inline
    void SparseVBaseTileL1FullLoad(TensorB &gBTensor, TensorL1B &l1BTensorTla,
                                   AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
                                   uint32_t gatheredKvSTileIdx, uint32_t kvSeqlen,
                                   uint32_t kvSBaseTile, uint32_t blockShapeY,
                                   uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
                                   uint32_t curBaseTileSize, uint32_t embed)
    {
        // V 作 A 矩阵 gather：用 A 角色的 copy（dst = TensorL1A = nZ，与 l1ATensorTla 的 nZ 声明一致；
        // 不能用 CopyGmToL1B——其 dst 固定为 TensorL1B（zN），类型不匹配）
        using CopyGmToL1A = typename TileCopy_::template CopyGmToL1A<TensorB>;
        CopyGmToL1A copyGmToL1A;
        uint32_t baseTileStartOffset = gatheredKvSTileIdx * kvSBaseTile;
        uint32_t baseTileEndOffset = baseTileStartOffset + curBaseTileSize;
        // 稀疏情况下对实际选中的部分gather后进行基本块切分
        uint32_t gatheredStartOffset = baseTileStartOffset;
        uint32_t gatheredYBlockIdx = gatheredStartOffset / blockShapeY;
        uint32_t yBlockInnerStartOffset = gatheredStartOffset % blockShapeY;
        uint32_t oriYBlockIdx = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
        uint32_t oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStartOffset;
        uint32_t dealtLenAccum = 0;

        while (dealtLenAccum < curBaseTileSize && gatheredYBlockIdx < yBlockNumRsvd &&
               oriYBlockIdx < yBlockNumAval && oriStartOffset < kvSeqlen) {
            uint32_t curYBlockSize = blockShapeY;
            if (oriYBlockIdx == yBlockNumAval - 1) {
                curYBlockSize = kvSeqlen - oriYBlockIdx * blockShapeY;
            }
            uint32_t gatheredEndOffset =
                min(gatheredYBlockIdx * blockShapeY + curYBlockSize,
                    baseTileEndOffset);
            uint32_t curDealtLen = gatheredEndOffset - gatheredStartOffset;
            if (curDealtLen == 0) {
                break;
            }
            // ColumnMajor [D, kvS] 视图：稀疏 gather 沿列（kvS 方向）切分
            auto l1BTensorTlaTile = GetTile(l1BTensorTla,
                tla::MakeCoord(0, dealtLenAccum), tla::MakeShape(embed, curDealtLen));
            auto gBTensorTlaTile = GetTile(gBTensor,
                tla::MakeCoord(0, oriStartOffset), tla::MakeShape(embed, curDealtLen));
            copyGmToL1A(l1BTensorTlaTile, gBTensorTlaTile);
            dealtLenAccum += curDealtLen;
            gatheredStartOffset += curDealtLen;
            gatheredYBlockIdx = gatheredStartOffset / blockShapeY;
            yBlockInnerStartOffset = gatheredStartOffset % blockShapeY;
            if (dealtLenAccum < curBaseTileSize) {
                // 防止最后一次循环之后对GM的访问越界引发硬件error
                oriYBlockIdx = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
                oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStartOffset;
            }
        }
    }

    template <class TensorB, class TensorC, class TensorGo>
    __aicore__ inline
    void operator()(TensorB &gBTensor, TensorC &ubCTensor, TensorGo &l1GoTensorTla,
                    AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
                    GemmCoord actualOriShape,
                    uint32_t gatheredKvSTileIdx, uint32_t kvSeqlen,
                    uint32_t kvSBaseTile, uint32_t blockShapeY,
                    uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
                    uint64_t prefixSumL0AStages, uint64_t prefixSumL0BStages,
                    Arch::CrossCoreFlag smToMm2Flag, Arch::CrossCoreFlag mm2ToReFlag,
                    bool isFirstKvSTile, bool isLastKvSTile,
                    uint64_t deqScalar = 0)
    {
        uint32_t rowNum = actualOriShape[0];        // qS
        uint32_t embed = actualOriShape[1];         // D
        uint32_t curBaseTileSize = actualOriShape[2];  // kvS
        // CUBE 单实例执行：不使用 GetSubBlockIdx（与现有 PV 同风格——mmad 全量、FixPipe 显式 Sub0/Sub1 两次调用）

        uint32_t l1ABufId = gatheredKvSTileIdx % l1ABufNum;
        uint32_t l1BBufId = gatheredKvSTileIdx % l1BBufNum;
        uint32_t l1VEventId = l1ABufId + 3;
        uint32_t l1DiagBufId = gatheredKvSTileIdx % DIAG_BUF_NUM;

        // V gather 落 L1：ColumnMajor [D, kvS]（nZ@A1，GM->L1 用 ColumnMajor->nZ 特化，
        // 与 block_mmad_k_qt_st_arch35.hpp 的 Q 加载同构）；A=(V)^T 直接读同一 nZ 视图
        // 注：文件内布局 tag 全部硬编码，不依赖 TileCopy 的 L1A/L1B 推导（interface 保持原样）
        auto l1ALayoutTla = tla::MakeLayout<ElementA, NpuArch::layout::nZ>(embed, curBaseTileSize);
        auto l1ATensorTla = tla::MakeTensor(l1ATensor[l1ABufId], l1ALayoutTla, Arch::PositionL1{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1VEventId);
        SparseVBaseTileL1FullLoad(
            gBTensor, l1ATensorTla, gSparseBlockIdx, gatheredKvSTileIdx, kvSeqlen, kvSBaseTile, blockShapeY,
            yBlockNumAval, yBlockNumRsvd, curBaseTileSize, embed);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1VEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1VEventId);
        // fwd crossCoreSync from online sm to mm2
        WaitCrossCoreSync<4, PIPE_MTE1>(smToMm2Flag);
        // B = (P)^T [kvS, qS]：P 在 L1 从 [kvS,qS] 视角就是 zN，声明 zN 走 copyL1ToL0B zN→nZ 特化（与 B=V 同构）
        auto l1BLayoutTla = tla::MakeLayout<ElementB, NpuArch::layout::zN>(curBaseTileSize, rowNum);
        auto l1BTensorTla = tla::MakeTensor(l1BTensor[l1BBufId], l1BLayoutTla, Arch::PositionL1{});

        // GO^T [D, qS] fp32（kernel 构造的 zN TLA 传入，与末块 go 的 TLA 传入模式一致）
        // diag 双子核块（每块 zN [64,64] 固定排布，与 S_trans 写侧一致：块 0 在 0、块 1 在 16KB），
        // 有效区 [qS/2, qS/2] 在块内前部——RE 的 B 子块按 qHalf 取对角块
        auto l1DiagLayoutTla = tla::MakeLayout<float, NpuArch::layout::zN>(64, 64);
        auto l1DiagTensorTlaBlk0 = tla::MakeTensor(
            l1DiagTensor[l1DiagBufId], l1DiagLayoutTla, Arch::PositionL1{});
        auto l1DiagTensorTlaBlk1 = tla::MakeTensor(
            l1DiagTensor[l1DiagBufId][64 * 64], l1DiagLayoutTla, Arch::PositionL1{});

        // PV^T 与 RE 共用的 L0 分块数：M=D、N=qS、K=kvS（RE 的 GO 沿 M 方向同 D 切分）
        uint32_t mL0LoopNum = CeilDiv(embed, L0_TILE_M);
        uint32_t nL0LoopNum = CeilDiv(rowNum, L0_TILE_N);
        uint32_t kL0LoopNum = CeilDiv(curBaseTileSize, L0_TILE_K);
        uint32_t l0CBufId = gatheredKvSTileIdx % L0_STAGES;
        uint32_t l0CEventId = l0CBufId + 2;
        // L0C：声明为 [D, qS] 整块（CUBE 单实例 mmad 全量计算）
        auto l0CLayoutTla = tla::MakeLayoutL0C(embed, rowNum);
        auto l0CTensorTla = tla::MakeTensor(l0CTensor[l0CBufId], l0CLayoutTla, Arch::PositionL0C{});

        // PV^T：C [D, qS] = (V)^T [D, kvS] * (P)^T [kvS, qS]，M=D、N=qS、K=kvS 三重 L0 分块（全量，无子核分摊）
        for (uint32_t mL0Itr = 0; mL0Itr < mL0LoopNum; mL0Itr++) {
            uint32_t l0TileMAct = (mL0Itr == mL0LoopNum - 1) ? (embed - mL0Itr * L0_TILE_M) : L0_TILE_M;
            for (uint32_t nL0Itr = 0; nL0Itr < nL0LoopNum; nL0Itr++) {
                uint32_t l0TileNAct = (nL0Itr == nL0LoopNum - 1) ? (rowNum - nL0Itr * L0_TILE_N) : L0_TILE_N;
                // l0C 同一 buffer 内子块（bias 保留给 RE 累加，不按 chunk 换 buffer）
                auto l0CTensorTlaTile = GetTile(l0CTensorTla,
                    tla::MakeCoord(mL0Itr * L0_TILE_M, nL0Itr * L0_TILE_N), tla::MakeShape(l0TileMAct, l0TileNAct));
                for (uint32_t kL0Itr = 0; kL0Itr < kL0LoopNum; kL0Itr++) {
                    uint32_t l0TileKAct = (kL0Itr == kL0LoopNum - 1) ?
                        (curBaseTileSize - kL0Itr * L0_TILE_K) : L0_TILE_K;
                    uint32_t l0ALoopCounter = prefixSumL0AStages + GetCurLoopCounter(mL0Itr, kL0LoopNum, kL0Itr);
                    uint32_t l0BLoopCounter = prefixSumL0BStages + GetCurLoopCounter(nL0Itr, kL0LoopNum, kL0Itr);
                    uint32_t l0ABufId = l0ALoopCounter % L0_STAGES;
                    uint32_t l0BBufId = l0BLoopCounter % L0_STAGES;
                    uint32_t l0AEventId = l0ABufId;
                    uint32_t l0BEventId = l0BBufId + 2;

                    // A 块 = (V)^T [D, kvS] 的 [D 块, kvS 块]：ColumnMajor 视图的 M/K 方向分块
                    auto l1ATensorTlaTile = GetTile(l1ATensorTla,
                        tla::MakeCoord(mL0Itr * L0_TILE_M, kL0Itr * L0_TILE_K), tla::MakeShape(l0TileMAct, l0TileKAct));
                    auto l0ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL0A>(l0TileMAct, l0TileKAct);
                    auto l0ATensorTla = tla::MakeTensor(l0ATensor[l0ABufId], l0ALayoutTla, Arch::PositionL0A{});

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    copyL1ToL0A(l0ATensorTla, l1ATensorTlaTile);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                    if ((mL0Itr == mL0LoopNum - 1) && (nL0Itr == nL0LoopNum - 1) && (kL0Itr == kL0LoopNum - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1VEventId);
                    }

                    // B 块 = (P)^T [kvS, qS] 的 [kvS 块, qS 块]：zN 视图的 K/N 方向分块
                    auto l1BTensorTlaTile = GetTile(l1BTensorTla,
                        tla::MakeCoord(kL0Itr * L0_TILE_K, nL0Itr * L0_TILE_N), tla::MakeShape(l0TileKAct, l0TileNAct));
                    auto l0BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL0B>(l0TileKAct, l0TileNAct);
                    auto l0BTensorTla = tla::MakeTensor(l0BTensor[l0BBufId], l0BLayoutTla, Arch::PositionL0B{});

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                    copyL1ToL0B(l0BTensorTla, l1BTensorTlaTile);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                    // reverse crossCoreSync for P：最后一次 copyL1ToL0B 已消费 L1 P 缓冲，VEC 才能覆写
                    // （与旧 PV 的 "L1 P buffer free" 方向对齐；softmax epilogue 在 P→L1 DataCopy 前
                    // wait 同一 flag，缺此 set 会死锁）
                    if ((mL0Itr == mL0LoopNum - 1) && (nL0Itr == nL0LoopNum - 1) && (kL0Itr == kL0LoopNum - 1)) {
                        SetCrossCoreSync<4, PIPE_MTE1>(smToMm2Flag);
                    }

                    // M 侧：等 MTE1 数据就绪（MTE1_M）再 mmad——与旧 PV/QK 惯例一致
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                    if ((mL0Itr == 0) && (nL0Itr == 0) && (kL0Itr == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
                    }
                    tileMmad(
                        l0CTensorTlaTile,
                        l0ATensorTla,
                        l0BTensorTla,
                        RoundUp(l0TileMAct, 16),
                        l0TileNAct,
                        l0TileKAct,
                        (kL0Itr == 0));
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                }
            }
        }

        // rescale：GO^T 每批搬 [D 块, 64]（32KB 满一个 L0A buf，双缓冲 2 批装完 GO 全量 64KB）；
        // diag 双子核块 [64,64]×2 一次搬满 L0B（32KB，批 0 前发，地址 0/16KB 隔离并行）；
        // mmad 按 16 列块分开乘（K=16 不浪费计算；diag 块对角 → 批间交叉项为 0）
        // 首块（含单块）不涉及 rescale
        if (!isFirstKvSTile) {
            // GO^T 由上一块 FixPipe(CFG_NZ→L1) 写入：等 L1 写完成后再读（独立 event id，与 l0C 释放的 FIX_M 区分）
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(GO_READY_EVENT_ID);
            // diag 就绪（双向 token 的 ready 方向）：等 VEC 把本轮 diag(dm/vScale) 写完再读，
            // 避免读到旧一轮或半成品；同一 flag 在 RE 末尾再 set 回 consumed 给 VEC 覆写前 wait
            Arch::CrossCoreFlag diagFlag(6 + l1DiagBufId);
            WaitCrossCoreSync<4, PIPE_MTE1>(diagFlag);
            // 尾块 qS 非 16 倍数时向上取整：最后一块用实际列数 nAct（mmad 的 k/n 用实际值，与 PV^T 的
            // K 尾块同惯例；k 对齐填充读的 B 行在 C 的列区内为 0——对角矩阵保证填充贡献恒为 0，无污染）
            // 批宽 = qS/2（双子核，运行期）：批 0 列 [0, qHalf)、批 1 列 [qHalf, rowNum)——与 diag 双子核块一一对应
            uint32_t qHalf = qPerSubCore_;
            uint32_t frGrpNum = CeilDiv(rowNum, qHalf);  // rowNum 偶数时恒 2
            // diag 的 L0B stage 固定 1 次（双子核块全量一次搬入，两批共用；与 PV^T 最后用的 stage 相差 1 无冲突）
            uint32_t l0BLoopCounter = prefixSumL0BStages + nL0LoopNum * kL0LoopNum;
            uint32_t l0BBufId = l0BLoopCounter % L0_STAGES;
            uint32_t l0BEventId = l0BBufId + 2;
            // B 大搬：diag 双子核块 [64,64] → L0B 偏移 0/16KB（float zN→nZ 转置，2 次并行发，地址隔离；
            // 排布固定 64×64 与写侧一致，qS<128 时有效区在块内前部，多搬部分不被 mmad 消费）
            auto l0BBlkLayoutTla = tla::MakeLayout<float, LayoutTagL0B>(64, 64);
            auto l0BBlk0TensorTla = tla::MakeTensor(l0BRescaleTensor[l0BBufId],
                l0BBlkLayoutTla, Arch::PositionL0B{});
            auto l0BBlk1TensorTla = tla::MakeTensor(
                l0BRescaleTensor[l0BBufId][64 * 64],
                l0BBlkLayoutTla, Arch::PositionL0B{});
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
            CopyL1ToL0BRescale copyL1ToL0BRescale;
            copyL1ToL0BRescale(l0BBlk0TensorTla, l1DiagTensorTlaBlk0);
            copyL1ToL0BRescale(l0BBlk1TensorTla, l1DiagTensorTlaBlk1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);

            for (uint32_t frGrp = 0; frGrp < frGrpNum; frGrp++) {
                // 批列区间 [grpColStart, grpColEnd)：批 0 = [0, qHalf)、尾批 = [qHalf, rowNum)
                uint32_t grpColStart = frGrp * qHalf;
                uint32_t grpColEnd = (frGrp == frGrpNum - 1) ? rowNum : (frGrp + 1) * qHalf;
                uint32_t grpCols = grpColEnd - grpColStart;
                uint32_t nGrpAct = CeilDiv(grpCols, 16);  // 批内 16 列块数（尾块可能 <16 列）
                // A 大搬（先行循环，mL0 流水预取）：(GO_prev)^T 的 [D 块, 批列宽] 连续 → L0A zN 恒等（qS=128 时 32KB 满）
                for (uint32_t mL0Itr = 0; mL0Itr < mL0LoopNum; mL0Itr++) {
                    uint32_t l0TileMAct = (mL0Itr == mL0LoopNum - 1) ? (embed - mL0Itr * L0_TILE_M) : L0_TILE_M;
                    // RE 的 stage 接在 PV^T 之后：A 每 (frGrp, mL0) 一次（PV^T A 总 = mL0×kL0）
                    uint32_t l0ALoopCounter = prefixSumL0AStages +
                        mL0LoopNum * kL0LoopNum + frGrp * mL0LoopNum + mL0Itr;
                    uint32_t l0ABufId = l0ALoopCounter % L0_STAGES;
                    uint32_t l0AEventId = l0ABufId;
                    auto l1GoTensorTlaTile = GetTile(l1GoTensorTla,
                        tla::MakeCoord(mL0Itr * L0_TILE_M, grpColStart), tla::MakeShape(l0TileMAct, grpCols));
                    auto l0ALayoutTla = tla::MakeLayout<float, LayoutTagL0A>(l0TileMAct, grpCols);
                    auto l0ATensorTla = tla::MakeTensor(l0ARescaleTensor[l0ABufId],
                        l0ALayoutTla, Arch::PositionL0A{});
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    CopyL1ToL0ARescale copyL1ToL0ARescale;
                    copyL1ToL0ARescale(l0ATensorTla, l1GoTensorTlaTile);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                }
                // mmad：批内 nGrpAct 个 16 列子块分开乘（K=16 不浪费；B 子块为 L0B [qS/2,qS/2] 的对角块）
                for (uint32_t mL0Itr = 0; mL0Itr < mL0LoopNum; mL0Itr++) {
                    uint32_t l0TileMAct = (mL0Itr == mL0LoopNum - 1) ? (embed - mL0Itr * L0_TILE_M) : L0_TILE_M;
                    uint32_t l0ALoopCounter = prefixSumL0AStages +
                        mL0LoopNum * kL0LoopNum + frGrp * mL0LoopNum + mL0Itr;
                    uint32_t l0ABufId = l0ALoopCounter % L0_STAGES;
                    uint32_t l0AEventId = l0ABufId;
                    auto l0ALayoutTla = tla::MakeLayout<float, LayoutTagL0A>(l0TileMAct, grpCols);
                    auto l0ATensorTla = tla::MakeTensor(l0ARescaleTensor[l0ABufId],
                        l0ALayoutTla, Arch::PositionL0A{});
                    // B 子块源：frGrp 选双子核块（批 0 → 块 0 偏移 0、批 1 → 块 1 偏移 (qS/2)²）
                    auto l0BTensorTla = (frGrp == 0) ? l0BBlk0TensorTla : l0BBlk1TensorTla;
                    // M 侧：等 MTE1 数据就绪（MTE1_M）再 mmad——与旧 PV/QK 惯例一致
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                    if ((frGrp == 0) && (mL0Itr == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                    }
                    for (uint32_t j = 0; j < nGrpAct; j++) {
                        // 批尾块实际列（批宽非 16 倍数时最后一列块 <16；mmad 的 k/n 用实际值，
                        // B 的 16 行中 k 方向只消费前 nAct 行——块宽 ≥ 批宽，不越界）
                        uint32_t nAct = (j == nGrpAct - 1) ? (grpColEnd - grpColStart - j * 16) : 16;
                        // mmad 子块：C 列 grpColStart + j*16（全局）、A 列 j*16（批内）、B 对角 (16j,16j)——偏移由
                        // tileMmad 的 layout()(coord()) 处理（与 copy 类同款机制）
                        auto l0CTensorTlaTileFr = GetTile(l0CTensorTla,
                            tla::MakeCoord(mL0Itr * L0_TILE_M, grpColStart + j * 16),
                            tla::MakeShape(l0TileMAct, nAct));
                        auto l0ATensorTlaTile = GetTile(l0ATensorTla,
                            tla::MakeCoord(mL0Itr * L0_TILE_M, j * 16), tla::MakeShape(l0TileMAct, nAct));
                        auto l0BTensorTlaTile = GetTile(l0BTensorTla,
                            tla::MakeCoord(j * 16, j * 16), tla::MakeShape(16, nAct));
                        tileMmad(
                            l0CTensorTlaTileFr,
                            l0ATensorTlaTile,
                            l0BTensorTlaTile,
                            RoundUp(l0TileMAct, 16),
                            nAct,
                            nAct,
                            false);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    if ((frGrp == frGrpNum - 1) && (mL0Itr == mL0LoopNum - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                    }
                    // diag 消费完成（softmax 覆写同 buffer 前 wait）：最后一块 mmad 后 set（读已完成，单实例单发）
                    if ((frGrp == frGrpNum - 1) && (mL0Itr == mL0LoopNum - 1)) {
                        // 同一 flag set 回 consumed：VEC 下一轮覆写同 buffer 前 wait
                        SetCrossCoreSync<4, PIPE_MTE1>(diagFlag);
                    }
                }
            }
        }

        // fixpipe
        // 注：FIX_M 的 L0C 释放等待已在 mmad 循环（首个 chunk (mL0Itr==0 && nL0Itr==0 && kL0Itr==0)）执行，
        // 此处不重复 wait（双 wait 会消费两个 token 而死锁）
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
        if (isLastKvSTile) {
            // 末块（含单块）：NZ2DN 反量化搬 UB go，倒置再倒置回 ND [qS, D]，VEC 做 ÷gl/cast/写 GM
            // 单目标（dualDstCtl=0）：每子核手动调一次，搬自己的列半区 [D, qS/2]（双目标模式不支持随路量化）
            // ubCTensor 由 kernel 构造为 ColumnMajor（DN）布局的 TLA tensor，
            // CopyL0CToDst<TensorC> 按 dst 类型里的 layout tag 匹配 NZ2DN 特化（与现有 PV 同模式）
            using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorC>;
            // reverse crossCoreSync, do fixPipe only after ubCTensor is fully released
            WaitCrossCoreSync<4, PIPE_FIX>(mm2ToReFlag);
            // 单目标（dualDstCtl=0）显式两次调用（Sub0/Sub1，与现有 MM1/PV 的 FixPipe 模式一致）：
            // 各搬 [D, qS/2] 半区到对应 AIV 的 UB（subBlockId 0/1 路由），双目标模式不支持随路量化
            // 半区 = 列总对齐后 /2（与旧 PV/QK/KQ 的 FixPipe 惯例一致）：FP8 对齐 64（consistent with
            // K*Q^t splitN）、非 FP8 对齐 32；qS=128 时 = 64（与 kernel go UB [embed_, 64] 一致）；
            // 尾块实际列 < qHalf 时多搬的部分为 L0C dummy / go 缓冲固定分配区，无害
            uint32_t qHalfAligned = FULL_QUANT_FP8 ? RoundUp(rowNum, 64) : RoundUp(rowNum, 32);
            uint32_t qHalf = qHalfAligned / 2;
            auto l0CTensorTlaTileSub0 = GetTile(l0CTensorTla,
                tla::MakeCoord(0, 0), tla::MakeShape(embed, qHalf));
            auto l0CTensorTlaTileSub1 = GetTile(l0CTensorTla,
                tla::MakeCoord(0, qHalf), tla::MakeShape(embed, qHalf));
            CopyL0CToDst copyL0CToDst;
            copyL0CToDst(ubCTensor, l0CTensorTlaTileSub0, deqScalar, 0);
            copyL0CToDst(ubCTensor, l0CTensorTlaTileSub1, deqScalar, 1);
            // crossCoreSync after all fixPipe move
            SetCrossCoreSync<4, PIPE_FIX>(mm2ToReFlag);
        } else {
            // 首/中间块：NZ 保形反量化落 L1 作为下一块 GO^T（倒置状态保持，零转换）
            // dst 直接用模板参数 TensorGo（kernel 构造的 GO^T TLA，与末块 go 的 TensorC 模式一致）
            using CopyL0CToL1 = typename TileCopy::template CopyL0CToL1<TensorGo>;
            CopyL0CToL1 copyL0CToL1;
            copyL0CToL1(l1GoTensorTla, l0CTensorTla, deqScalar);
            // GO^T L1 写完成 → 非首块 RE 读前 wait（只能在此分支 set：末块 NZ2DN→UB 与 GO^T 无关，
            // 若统一在尾部 set，跨 task 时 RE 会 wait 到无关的 UB 写完成而提前通过）
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(GO_READY_EVENT_ID);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensor[MAX_L1_STAGES];   // V 段（gather 写入，A=(V)^T 读）
    AscendC::LocalTensor<ElementB> l1BTensor[MAX_L1_STAGES];   // P 段（softmax 写入，B=(P)^T 读）
    AscendC::LocalTensor<float> l1DiagTensor[DIAG_BUF_NUM];    // diag 块 [qS/2, qS/2]
    AscendC::LocalTensor<ElementA> l0ATensor[L0_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[L0_STAGES];
    // RE 段：GO^T/diag 为 fp32，需 float 的 L0 buffer（PV^T 的 fp8 buffer 元素类型不匹配 float 布局）；
    // 物理复用同一 L0A/L0B 内存（RE 块 ≤ PV^T 块大小，同 stage 不同时使用）
    AscendC::LocalTensor<float> l0ARescaleTensor[L0_STAGES];
    AscendC::LocalTensor<float> l0BRescaleTensor[L0_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[L0_STAGES];

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;

    uint32_t l1ATileM;
    uint32_t l1BTileN;
    uint32_t l1ATileK;
    uint32_t l1BTileK;
    uint32_t l1ABufNum;
    uint32_t l1BBufNum;
    uint32_t qPerSubCore_;   // diag 双子核块宽 = qS/2（运行期，qS ≤ 128 且偶数）
};
////////////////////////////////////////////////////////////////////

}  // namespace NpuArch::Gemm::Block
#endif
