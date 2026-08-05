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

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;

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
    static constexpr uint32_t GO_Q_PER_SUBCORE = 64;  // qS/2
    static constexpr uint32_t SUB_BLOCK_NUM = 2;      // diag 每 stage 双子核各一块（L1 共享，需分块排布）
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
        // diag 块 [qS/2, qS/2] fp32：每 stage 双子核各一块（L1 共享物理，子核 s 的块在 stage 内偏移 s×16KB），3 份轮转
        for (uint32_t i = 0; i < DIAG_BUF_NUM; i++) {
            l1DiagTensor[i] = resource.l1Buf.template GetBufferByByte<float>(
                l1DiagAddrStart +
                SUB_BLOCK_NUM * GO_Q_PER_SUBCORE * GO_Q_PER_SUBCORE * sizeof(float) * i);
        }
        for (uint32_t i = 0; i < L0_STAGES; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(
                L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(
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
        // diag 双子核块（每块 zN [64,64]，L1 顺序排布：块 0 在 0、块 1 在 16KB）——RE 全量时按 fr/4 选块
        auto l1DiagLayoutTla = tla::MakeLayout<float, NpuArch::layout::zN>(GO_Q_PER_SUBCORE, GO_Q_PER_SUBCORE);
        auto l1DiagTensorTlaBlk0 = tla::MakeTensor(
            l1DiagTensor[l1DiagBufId], l1DiagLayoutTla, Arch::PositionL1{});
        auto l1DiagTensorTlaBlk1 = tla::MakeTensor(
            l1DiagTensor[l1DiagBufId][GO_Q_PER_SUBCORE * GO_Q_PER_SUBCORE], l1DiagLayoutTla, Arch::PositionL1{});

        uint32_t kL0LoopNum = CeilDiv(curBaseTileSize, L0_TILE_K);
        uint32_t l0CBufId = gatheredKvSTileIdx % L0_STAGES;
        uint32_t l0CEventId = l0CBufId + 2;
        // L0C：声明为 [D, qS] 整块（CUBE 单实例 mmad 全量计算）
        auto l0CLayoutTla = tla::MakeLayoutL0C(embed, rowNum);
        auto l0CTensorTla = tla::MakeTensor(l0CTensor[l0CBufId], l0CLayoutTla, Arch::PositionL0C{});

        // PV^T：C [D, qS] = (V)^T [D, kvS] * (P)^T [kvS, qS]，K=kvS 分块（全量，无子核分摊）
        for (uint32_t kL0Itr = 0; kL0Itr < kL0LoopNum; kL0Itr++) {
            uint32_t l0TileKAct = (kL0Itr == kL0LoopNum - 1) ?
                (curBaseTileSize - kL0Itr * L0_TILE_K) : L0_TILE_K;
            uint32_t l0ALoopCounter = prefixSumL0AStages + GetCurLoopCounter(0, kL0LoopNum, kL0Itr);
            uint32_t l0BLoopCounter = prefixSumL0BStages + GetCurLoopCounter(0, kL0LoopNum, kL0Itr);
            uint32_t l0ABufId = l0ALoopCounter % L0_STAGES;
            uint32_t l0BBufId = l0BLoopCounter % L0_STAGES;
            uint32_t l0AEventId = l0ABufId;
            uint32_t l0BEventId = l0BBufId + 2;

            // A 块 [D, kvS块]：ColumnMajor [D,kvS] 的 K 方向分块
            auto l1ATensorTlaTile = GetTile(l1ATensorTla,
                tla::MakeCoord(0, kL0Itr * L0_TILE_K), tla::MakeShape(embed, l0TileKAct));
            auto l0ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL0A>(embed, l0TileKAct);
            auto l0ATensorTla = tla::MakeTensor(l0ATensor[l0ABufId], l0ALayoutTla, Arch::PositionL0A{});

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
            copyL1ToL0A(l0ATensorTla, l1ATensorTlaTile);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
            if (kL0Itr == kL0LoopNum - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1VEventId);
            }

            // B 块 [kvS块, qS]：zN [kvS,qS] 的 K 行 + 全 N 列
            auto l1BTensorTlaTile = GetTile(l1BTensorTla,
                tla::MakeCoord(kL0Itr * L0_TILE_K, 0),
                tla::MakeShape(l0TileKAct, rowNum));
            auto l0BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL0B>(l0TileKAct, rowNum);
            auto l0BTensorTla = tla::MakeTensor(l0BTensor[l0BBufId], l0BLayoutTla, Arch::PositionL0B{});

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
            copyL1ToL0B(l0BTensorTla, l1BTensorTlaTile);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);

            // M 侧：等 MTE1 数据就绪（MTE1_M）再 mmad——与旧 PV/QK 惯例一致
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
            if (kL0Itr == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
            }
            tileMmad(
                l0CTensorTla,
                l0ATensorTla,
                l0BTensorTla,
                embed,
                rowNum,
                l0TileKAct,
                (kL0Itr == 0));
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
        }

        // rescale：C 按 16 列块拼接累加（8 次 [D,16]×[16,16] 小 mmad，K=16，全量覆盖 qS）
        // diag 双子核块各 [64,64] fp32 = 4×8 个 16×8 fractal，非零仅对角斜带（fractal 行 fr 的列 2fr/2fr+1）
        // 按 16×16 块（=2 个 fractal）合并为 4×4 网格，仅 4 个对角块 (fr,fr) 非零
        // → 只对非零块发 copy（GetTile 子块，copy 特化按子块 coord 自动推导 LoadData 参数）；fr/4 选双子核块
        // 首块（含单块）不涉及 rescale
        if (!isFirstKvSTile) {
            // GO^T 由上一块 FixPipe(CFG_NZ→L1) 写入：等 L1 写完成后再读（独立 event id，与 l0C 释放的 FIX_M 区分）
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(GO_READY_EVENT_ID);
            // 尾块 qS 非 16 倍数时向上取整：最后一块用实际列数 nAct（mmad 的 k/n 用实际值，与 PV^T 的
            // K 尾块同惯例；k 对齐填充读的 B 行在 C 的列区内为 0——对角矩阵保证填充贡献恒为 0，无污染）
            uint32_t blockNum = CeilDiv(rowNum, 16);   // 8
            for (uint32_t fr = 0; fr < blockNum; fr++) {
                uint32_t nAct = min(16, rowNum - fr * 16);
                uint32_t l0ALoopCounter = prefixSumL0AStages + kL0LoopNum + fr;
                uint32_t l0BLoopCounter = prefixSumL0BStages + kL0LoopNum + fr;
                uint32_t l0ABufId = l0ALoopCounter % L0_STAGES;
                uint32_t l0BBufId = l0BLoopCounter % L0_STAGES;
                uint32_t l0AEventId = l0ABufId;
                uint32_t l0BEventId = l0BBufId + 2;

                // A 块 = (GO_prev)^T 的 [D, 16] 列块：zN -> L0A zN 恒等
                auto l1GoTensorTlaTile = GetTile(l1GoTensorTla,
                    tla::MakeCoord(0, fr * 16), tla::MakeShape(embed, 16));
                auto l0ALayoutTla = tla::MakeLayout<float, LayoutTagL0A>(embed, 16);
                auto l0ATensorTla = tla::MakeTensor(l0ATensor[l0ABufId], l0ALayoutTla, Arch::PositionL0A{});
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                copyL1ToL0A(l0ATensorTla, l1GoTensorTlaTile);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);

                // B 块 = diag 的 [16, 16] 对角块：fr/4 选双子核块，块内偏移 (fr%4)×16（对角块 (fr,fr)）
                uint32_t frInBlk = fr % 4;
                auto l1DiagTensorTla = (fr < 4) ? l1DiagTensorTlaBlk0 : l1DiagTensorTlaBlk1;
                auto l1DiagTensorTlaTile = GetTile(l1DiagTensorTla,
                    tla::MakeCoord(frInBlk * 16, frInBlk * 16), tla::MakeShape(16, 16));
                auto l0BLayoutTla = tla::MakeLayout<float, LayoutTagL0B>(16, 16);
                auto l0BTensorTla = tla::MakeTensor(l0BTensor[l0BBufId], l0BLayoutTla, Arch::PositionL0B{});
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                copyL1ToL0B(l0BTensorTla, l1DiagTensorTlaTile);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);

                // mmad：C[:, 16fr:16fr+nAct] += A [D,16] × B [16,16]，K=nAct，累加到 PV^T 的 bias
                // A/B 块固定 [16] 取（物理够），k=nAct 只算有效部分
                // M 侧：等 MTE1 数据就绪（MTE1_M）再 mmad——与旧 PV/QK 惯例一致
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                auto l0CTensorTlaTileFr = GetTile(l0CTensorTla,
                    tla::MakeCoord(0, fr * 16), tla::MakeShape(embed, nAct));
                tileMmad(
                    l0CTensorTlaTileFr,
                    l0ATensorTla,
                    l0BTensorTla,
                    embed,
                    nAct,
                    nAct,
                    false);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                // diag 消费完成（softmax 覆写同 buffer 前 wait）：最后一块 mmad 后 set（读已完成，单实例单发）
                if (fr == blockNum - 1) {
                    Arch::CrossCoreFlag diagConsumedFlag(6 + l1DiagBufId);
                    SetCrossCoreSync<4, PIPE_MTE1>(diagConsumedFlag);
                }
            }
        }

        // fixpipe
        // 注：FIX_M 的 L0C 释放等待已在 mmad 循环（kL0Itr==0）执行，此处不重复 wait（双 wait 会消费两个 token 而死锁）
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
            uint32_t qHalf = rowNum / 2;
            auto l0CTensorTlaTileSub0 = GetTile(l0CTensorTla,
                tla::MakeCoord(0, 0), tla::MakeShape(embed, qHalf));
            auto l0CTensorTlaTileSub1 = GetTile(l0CTensorTla,
                tla::MakeCoord(0, qHalf), tla::MakeShape(embed, rowNum - qHalf));
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
};
////////////////////////////////////////////////////////////////////

}  // namespace NpuArch::Gemm::Block
#endif
