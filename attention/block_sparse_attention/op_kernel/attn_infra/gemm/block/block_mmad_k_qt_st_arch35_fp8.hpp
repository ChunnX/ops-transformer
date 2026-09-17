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
 * @brief matmul implementation for single q&k^t base tile
 * This implementation is designed for the following senario:
 * A full q base tile is loaded to L1 from GM at the very beginning,
 * and it remains persistent until each k base tile is dealt
 * A full q*k^t base tile is loaded to UB from l0C, no workspace transit
 */
#ifndef GEMM_BLOCK_K_QT_ST_ARCH35_HIF8_HPP
#define GEMM_BLOCK_K_QT_ST_ARCH35_HIF8_HPP

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
#include "../../../tla/layout_bsa.hpp"
#include "../../../tla/tensor_bsa.hpp"

////////////////////////////////////////////////////////////////////

namespace NpuArch::Gemm::Block {
////////////////////////////////////////////////////////////////////

template <class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_, class ElementBias_,
          class TileCopy_,
          class TileMmad_>
struct BlockMmadTla<MmadAtlasA5BsaQKHif8<true>, // HIF8 specialization for performing K * Q^t = S^t
                    L1TileShape_, L0TileShape_, ElementA_, ElementB_, ElementC_, ElementBias_, TileCopy_, TileMmad_> {
public:
    using DispatchPolicy = MmadAtlasA5BsaQKHif8<true>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using ElementQ = ElementA_;
    using ElementK = ElementB_;
    using ElementS = ElementC_;

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
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = ArchTag::L0C_SIZE / L0_STAGES;
    static constexpr bool FULL_QUANT_HIF8 = AscendC::IsSameType<ElementQ, hifloat8_t>::value;

    // ---- Plan B: 外层大 db 里再切一层小 db ----
    // 外层大 db 不变: L0A/L0B 每槽 32 KiB、L0C 每槽 128 KiB, 按 l0Stage % 2 交替。
    // 每个大槽内部再等分成 L0_MICRO_STAGES 个小槽:
    //   L0A 小槽 16 KiB = 128 x 128 x sizeof(HIF8)
    //   L0C 小槽 64 KiB = 128 x 128 x sizeof(float)
    // MM1 的 M=256 沿 M 拆成两个 128 微块, 各自写进自己的 L0C 小槽 —— 这样
    // Cube(micro 1) 不必等 FixPipe(micro 0) 把整个 L0C 大槽腾空, 两条流水可以穿插。
    static constexpr uint32_t L0_MICRO_STAGES = 2;
    static constexpr uint32_t L0_TILE_M_MICRO = L0_TILE_M / L0_MICRO_STAGES;
    static constexpr uint32_t L0A_MICRO_BUF_SIZE = ArchTag::L0A_SIZE / (L0_STAGES * L0_MICRO_STAGES);
    static constexpr uint32_t L0B_MICRO_BUF_SIZE = ArchTag::L0B_SIZE / (L0_STAGES * L0_MICRO_STAGES);
    static constexpr uint32_t L0C_MICRO_BUF_SIZE = ArchTag::L0C_SIZE / (L0_STAGES * L0_MICRO_STAGES);
    static_assert(L0_TILE_M % L0_MICRO_STAGES == 0, "L0 tile M must split evenly into micro slots");
    static_assert(L0A_MICRO_BUF_SIZE >= L0_TILE_M_MICRO * L0_TILE_K * sizeof(ElementK),
                  "L0A micro slot too small for one micro M block");
    static_assert(L0C_MICRO_BUF_SIZE >= L0_TILE_M_MICRO * L0_TILE_N * sizeof(ElementAccumulator),
                  "L0C micro slot too small for one micro M block");
    // Q 由两个微块共用, 一次搬完, 只落在外层槽的第 0 片上 —— 所以一片必须装得下
    // 整个 Q tile。这条同时把 "qSTile == L0_TILE_N" 这个隐含前提钉住: 一旦
    // qBaseTile 超过 128, Q 会静默压到第 1 片上。
    static_assert(L0B_MICRO_BUF_SIZE >= L0_TILE_K * L0_TILE_N * sizeof(ElementQ),
                  "L0B micro slice must hold one whole Q tile (qSTile == L0_TILE_N)");

    // MTE1_M/M_MTE1 是同一条队列, L0A 和 L0B 的事件号必须错开:
    // L0A 按内层粒度吃满 0..3 (两个大槽 x 两个小槽), L0B 让到 4..7。
    // MM1 的 Q 只占第 0 片, 所以只碰 4 和 6; 5/7 留给 MM2 的 V —— 同一个 id 池。
    // M_FIX/FIX_M 是另一条队列, L0C 自己用 0..3, 跟上面不冲突。
    static constexpr uint32_t L0A_EVENT_BASE = 0;
    static constexpr uint32_t L0B_EVENT_BASE = L0_STAGES * L0_MICRO_STAGES; // 4

    // 事件号是运行时算出来的, 越界编译不报错只会静默跑飞, 所以在这里把预算钉死。
    // MM1 用到的最大的号是 L0B_EVENT_BASE + L0_STAGES * L0_MICRO_STAGES - 1 = 7。
    static_assert(L0B_EVENT_BASE + L0_STAGES * L0_MICRO_STAGES <= AscendC::QUE_MAX_EVENT,
                  "Plan B MM1 event ids exceed the MTE1<->M queue depth (QUE_MAX_EVENT)");

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag> &resource, Mm1L1TileHelper &mm1L1TileHelper)
    {
        l1QBufNum = mm1L1TileHelper.qL1BufNum;
        l1KBufNum = mm1L1TileHelper.kL1BufNum;
        l1ATileM = mm1L1TileHelper.mm1L1TileN;
        l1BTileN = mm1L1TileHelper.mm1L1TileM;
        l1ATileK = mm1L1TileHelper.mm1L1TileKRight;
        l1BTileK = mm1L1TileHelper.mm1L1TileKLeft;
        for (uint32_t i = 0; i < l1QBufNum; i++) {
            l1QTensor[i] =
                resource.l1Buf.template GetBufferByByte<ElementQ>(l1BTileK * l1BTileN * sizeof(ElementQ) * i);
        }
        for (uint32_t i = 0; i < l1KBufNum; i++) {
            l1KTensor[i] = resource.l1Buf.template GetBufferByByte<ElementK>(
                l1BTileK * l1BTileN * sizeof(ElementQ) * l1QBufNum + l1ATileM * l1ATileK * sizeof(ElementK) * i);
        }
        // L0A/L0B/L0C 三块统一按小槽切: 下标 = outerId * L0_MICRO_STAGES + microIdx,
        // 偏移即 outerId * 大槽 + microIdx * 小槽, 正好等于小槽大小的线性递推。
        // 外层 DB 仍只在 {0,1} 两个 outer 上翻。
        for (uint32_t i = 0; i < L0_STAGES * L0_MICRO_STAGES; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementQ>(L0A_MICRO_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementK>(L0B_MICRO_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_MICRO_BUF_SIZE * i);
        }
    }

    /// Destructor
    __aicore__ inline ~BlockMmadTla() {}

    template <class TensorQ>
    __aicore__ inline void loadQGM(TensorQ &gQTensor, GemmCoord actualOriShape)
    {
        uint32_t qSTile = actualOriShape[0];
        uint32_t embed = actualOriShape[1];
        // Q copied to L1 as B matrix
        using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorQ>;
        CopyGmToL1B copyGmToL1B;
        auto l1BLayoutTla = tla::MakeLayout<ElementQ, LayoutTagL1B>(embed, qSTile);
        auto l1BTensorTla = tla::MakeTensor(l1QTensor[0], l1BLayoutTla, Arch::PositionL1{});
        auto l1BTensorTlaTile = GetTile(l1BTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(embed, qSTile));
        auto gQTensorTlaTile = GetTile(gQTensor, tla::MakeCoord(0, 0), tla::MakeShape(embed, qSTile));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        copyGmToL1B(l1BTensorTlaTile, gQTensorTlaTile);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    template <class TensorK, class TensorL1K>
    __aicore__ inline void SparseKL1TileMLoad(TensorK &gKTensor, TensorL1K &l1KTensorTla,
                                              AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
                                              uint32_t gatheredKvSTileIdx, uint32_t kvSeqlen, uint32_t kvSBaseTile,
                                              uint32_t blockShapeY, uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
                                              uint32_t l1KTileMAct, uint32_t embed, uint32_t kvSBaseTileInnerOffset)
    {
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<TensorK>;
        CopyGmToL1A copyGmToL1A;
        uint32_t baseTileStartOffset = gatheredKvSTileIdx * kvSBaseTile + kvSBaseTileInnerOffset;
        uint32_t baseTileEndOffset = baseTileStartOffset + l1KTileMAct;
        // 稀疏情况下对实际选中的部分gather后进行基本块切分
        // 当前处理的Yblock在gather的序列中的起始偏移，初始值为当前基块的起始偏移
        uint32_t gatheredStartOffset = baseTileStartOffset;
        // 当前处理的Yblock gather后的下标，初始值为基本块起始偏移对应的按Y方向稀疏block的gather后起始下标
        uint32_t gatheredYBlockIdx = gatheredStartOffset / blockShapeY;
        // 当前基本块起始偏移对应的按Y方向稀疏后的block内起始偏移
        uint32_t yBlockInnerStartOffset = gatheredStartOffset % blockShapeY;
        // 当前处理的Yblock原始的下标，初始值为基本块对应的按Y方向稀疏block的原始起始下标
        uint32_t oriYBlockIdx = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
        // 当前处理的Yblock起始位置在原始序列中的偏移，初始值为基本块在原始序列中的起始偏移
        uint32_t oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStartOffset;
        // 逐稀疏block搬移填充基本块过程中，已处理的累积序列长度
        uint32_t dealtLenAccum = 0;

        while (dealtLenAccum < l1KTileMAct && gatheredYBlockIdx < yBlockNumRsvd && oriYBlockIdx < yBlockNumAval &&
               oriStartOffset < kvSeqlen) {
            uint32_t curYBlockSize = blockShapeY;
            if (oriYBlockIdx == yBlockNumAval - 1) {
                curYBlockSize = kvSeqlen - oriYBlockIdx * blockShapeY;
            }
            uint32_t gatheredEndOffset = min(gatheredYBlockIdx * blockShapeY + curYBlockSize, baseTileEndOffset);
            // 当前循环处理的序列长度
            uint32_t curDealtLen = gatheredEndOffset - gatheredStartOffset;
            if (curDealtLen == 0) {
                break;
            }
            // K copied to L1 as A matrix
            auto l1KTensorTlaTile =
                GetTile(l1KTensorTla, tla::MakeCoord(dealtLenAccum, 0), tla::MakeShape(curDealtLen, embed));
            auto gKTensorTlaTile =
                GetTile(gKTensor, tla::MakeCoord(oriStartOffset, 0), tla::MakeShape(curDealtLen, embed));
            copyGmToL1A(l1KTensorTlaTile, gKTensorTlaTile);
            // 为下一次循环刷新循环变量
            dealtLenAccum += curDealtLen;
            gatheredStartOffset += curDealtLen;
            gatheredYBlockIdx = gatheredStartOffset / blockShapeY;
            yBlockInnerStartOffset = gatheredStartOffset % blockShapeY;
            if (dealtLenAccum < l1KTileMAct) {
                oriYBlockIdx = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
                oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStartOffset;
            }
        }
    }

    template <class TensorK, class TensorS>
    __aicore__ inline void operator()(TensorK &gKTensor, TensorS &ubSTensor,
                                      AscendC::GlobalTensor<int32_t> gSparseBlockIdx, GemmCoord actualOriShape,
                                      uint32_t gatheredKvSTileIdx, uint32_t kvSeqlen, uint32_t kvSBaseTile,
                                      uint32_t blockShapeY, uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
                                      uint64_t &l0Stage, Arch::CrossCoreFlag mm1ToSmFlag)
    {
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorS>;
        CopyL0CToDst copyL0CToDst;

        const uint32_t bufId = static_cast<uint32_t>(l0Stage++ % L0_STAGES);
        // 外层大槽 id -> 小槽下标基址。K 沿 M 切成两个微块后各自独立搬移, 所以
        // L0A 的事件也下沉到内层。Q 两个微块共用一次搬运、只占第 0 片, 所以 L0B
        // 的事件号固定取本 outer 的第 0 片 —— Wait/Set 成对落在同一个号上。
        //
        // L0A/L0C 的事件基址是 0, 所以下面对它们 "事件号 == 缓冲片下标", 一个变量
        // 两用是成立的; L0B 的基址是 4, 两者不是同一个数, 必须分开。
        const uint32_t l0AMicroBase = L0A_EVENT_BASE + bufId * L0_MICRO_STAGES;
        const uint32_t l0BSlot = bufId * L0_MICRO_STAGES;     // 缓冲片: 0 / 2
        const uint32_t l0BEventId = L0B_EVENT_BASE + l0BSlot; // 事件号: 4 / 6
        const uint32_t l0CMicroBase = bufId * L0_MICRO_STAGES;

        uint32_t qSTile = actualOriShape[0];
        uint32_t embed = actualOriShape[2];
        uint32_t kSTile = actualOriShape[1];

        // Q as Matrix B on L1, has shape [D, S1] in nZ
        auto l1QLayoutTla = tla::MakeLayout<ElementQ, LayoutTagL1B>(embed, qSTile);
        auto l1QTensorTla = tla::MakeTensor(l1QTensor[0], l1QLayoutTla, Arch::PositionL1{});

        uint32_t mL1LoopNum = CeilDiv(kSTile, l1ATileM);
        uint32_t nL0LoopNum = CeilDiv(qSTile, L0_TILE_N);
        uint32_t kL0LoopNum = CeilDiv(embed, L0_TILE_K);

        // while splitting the base tile S to 2 AIVs,
        // the order of the elements in each column is expected to be preserved,
        // which means a column in l0C cannot be chunked and processed by dualMode FixPipe seperately.
        // therefore, FixPipe won't launch until each portion(chunked only by columns, based on nbuffer strategy)
        // of the base tile is ready on l0C
        for (uint32_t mL1Itr = 0; mL1Itr < mL1LoopNum; mL1Itr++) {
            uint32_t l1TileMAct = (mL1Itr == mL1LoopNum - 1) ? (kSTile - mL1Itr * l1ATileM) : l1ATileM;
            uint32_t mLoopCounterL1 = GetCurLoopCounter(gatheredKvSTileIdx, mL1LoopNum, mL1Itr);
            uint32_t l1KBufId = mLoopCounterL1 % l1KBufNum;
            uint32_t l1AEventId = l1KBufId + 1;
            auto l1KLayoutTla = tla::MakeLayout<ElementK, LayoutTagL1A>(l1TileMAct, embed);
            auto l1KTensorTla = tla::MakeTensor(l1KTensor[l1KBufId], l1KLayoutTla, Arch::PositionL1{});
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventId);
            SparseKL1TileMLoad(gKTensor, l1KTensorTla, gSparseBlockIdx, gatheredKvSTileIdx, kvSeqlen, kvSBaseTile,
                               blockShapeY, yBlockNumAval, yBlockNumRsvd, l1TileMAct, embed, mL1Itr * l1ATileM);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventId);
            uint32_t mL0LoopNum = CeilDiv(l1TileMAct, L0_TILE_M);
            bool l1ToL0BNoRepeatFlag = (nL0LoopNum == 1) && (kL0LoopNum <= L0_STAGES);
            for (uint32_t mL0Itr = 0; mL0Itr < mL0LoopNum; mL0Itr++) {
                uint32_t l0TileMAct = (mL0Itr == mL0LoopNum - 1) ? (l1TileMAct - mL0Itr * L0_TILE_M) : L0_TILE_M;
                uint32_t microLoopNum = CeilDiv(l0TileMAct, L0_TILE_M_MICRO);

                // ---- 第一遍: 微块逐个发 MMAD, 各写自己的 L0C 小槽 ----
                // 与单块版本唯一的差别是循环末尾只 SetFlag<M_FIX>, 不 WaitFlag:
                // 两个微块的 MMAD 都发完才进第二遍收 FixPipe。这样 Cube(micro 1) 不必
                // 等 FixPipe(micro 0) 把整个 L0C 大槽腾空, Cube 和 FixPipe 才可能穿插。
                for (uint32_t microIdx = 0; microIdx < microLoopNum; microIdx++) {
                    uint32_t microMAct = (microIdx == microLoopNum - 1) ? (l0TileMAct - microIdx * L0_TILE_M_MICRO)
                                                                        : L0_TILE_M_MICRO;
                    uint32_t microMOffset = mL0Itr * L0_TILE_M + microIdx * L0_TILE_M_MICRO;
                    const uint32_t l0AEventId = l0AMicroBase + microIdx;
                    const uint32_t l0CMicroEventId = l0CMicroBase + microIdx;
                    auto l0CLayoutTla = tla::MakeLayoutL0C(microMAct, qSTile);
                    auto l0CTensorTla =
                        tla::MakeTensor(l0CTensor[l0CMicroEventId], l0CLayoutTla, Arch::PositionL0C{});
                    for (uint32_t nL0Itr = 0; nL0Itr < nL0LoopNum; nL0Itr++) {
                        uint32_t l0TileNAct = (nL0Itr == nL0LoopNum - 1) ? (qSTile - nL0Itr * L0_TILE_N) : L0_TILE_N;
                        // different n chunks will be concated in the same piece of l0C buffer
                        auto l0CTensorTlaTile = GetTile(l0CTensorTla, tla::MakeCoord(0, nL0Itr * L0_TILE_N),
                                                        tla::MakeShape(microMAct, l0TileNAct));
                        for (uint32_t kL0Itr = 0; kL0Itr < kL0LoopNum; kL0Itr++) {
                            uint32_t l0TileKAct =
                                (kL0Itr == kL0LoopNum - 1) ? (embed - kL0Itr * L0_TILE_K) : L0_TILE_K;

                            auto l1KTensorTlaTile =
                                GetTile(l1KTensorTla, tla::MakeCoord(microMOffset, kL0Itr * L0_TILE_K),
                                        tla::MakeShape(microMAct, l0TileKAct));
                            auto l0ALayoutTla = tla::MakeLayout<ElementK, LayoutTagL0A>(microMAct, l0TileKAct);
                            auto l0ATensorTla =
                                tla::MakeTensor(l0ATensor[l0AEventId], l0ALayoutTla, Arch::PositionL0A{});

                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                            copyL1ToL0A(l0ATensorTla, l1KTensorTlaTile);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                            if ((nL0Itr == nL0LoopNum - 1) && (microIdx == microLoopNum - 1) &&
                                (mL0Itr == mL0LoopNum - 1) && (kL0Itr == kL0LoopNum - 1)) {
                                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventId);
                            }

                            auto l1QTensorTlaTile =
                                GetTile(l1QTensorTla, tla::MakeCoord(kL0Itr * L0_TILE_K, nL0Itr * L0_TILE_N),
                                        tla::MakeShape(l0TileKAct, l0TileNAct));
                            auto l0BLayoutTla = tla::MakeLayout<ElementQ, LayoutTagL0B>(l0TileKAct, l0TileNAct);
                            auto l0BTensorTla =
                                tla::MakeTensor(l0BTensor[l0BSlot], l0BLayoutTla, Arch::PositionL0B{});

                            // Q 两个微块共用, 只在第一个微块搬一次。第二个微块不再等
                            // MTE1_M: 标志已被第一个微块消费掉, 重复 Wait 会死等; 而两个
                            // MMAD 同在 M pipe 上, 程序序已保证 Q 就位。
                            bool loadQ =
                                l1ToL0BNoRepeatFlag && (mL1Itr == 0) && (mL0Itr == 0) && (microIdx == 0);
                            if (loadQ) {
                                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                                copyL1ToL0B(l0BTensorTla, l1QTensorTlaTile);
                                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                            }

                            bool initMmad = (kL0Itr == 0);
                            uint32_t microMAligned = RoundUp(microMAct, 16);
                            if (loadQ) {
                                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                            }
                            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                            if (nL0Itr == 0 && kL0Itr == 0) {
                                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CMicroEventId);
                            }
                            tileMmad(l0CTensorTlaTile, l0ATensorTla, l0BTensorTla, microMAligned, l0TileNAct,
                                     l0TileKAct, initMmad);
                            if (l1ToL0BNoRepeatFlag && (mL1Itr == mL1LoopNum - 1) && (mL0Itr == mL0LoopNum - 1) &&
                                (microIdx == microLoopNum - 1)) {
                                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                            }
                            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                        }
                    }
                    // 只登记"这个微块的 MMAD 完了", 不等 FixPipe —— 等挪到第二遍。
                    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CMicroEventId);
                }

                // ---- 第二遍: 逐个微块收 FixPipe, 收完立刻把小槽还给 Cube ----
                if (mL0Itr == 0) {
                    // reverse crossCoreSync, do fixPipe only after ubSTensor is fully released
                    WaitCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);
                }
                for (uint32_t microIdx = 0; microIdx < microLoopNum; microIdx++) {
                    uint32_t microMAct = (microIdx == microLoopNum - 1) ? (l0TileMAct - microIdx * L0_TILE_M_MICRO)
                                                                        : L0_TILE_M_MICRO;
                    uint32_t microMOffset = mL0Itr * L0_TILE_M + microIdx * L0_TILE_M_MICRO;
                    const uint32_t l0CMicroEventId = l0CMicroBase + microIdx;
                    auto l0CLayoutTla = tla::MakeLayoutL0C(microMAct, qSTile);
                    auto l0CTensorTla =
                        tla::MakeTensor(l0CTensor[l0CMicroEventId], l0CLayoutTla, Arch::PositionL0C{});
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CMicroEventId);

                    uint32_t mFixPAligned8 = RoundUp(microMAct, 8);
                    if constexpr (FULL_QUANT_HIF8) {
                        // NO_QUANT SPLIT_N: one dual-destination Fixpipe serves both AIVs. K * Q^T
                        // stores Q on the N axis of L0C, so the split is along N: the hardware writes
                        // columns 0..n/2-1 into AIV0 and n/2..n-1 into AIV1, both at the same UB
                        // offsets given by ubSTensor's per-AIV stride. nSize is the full per-block Q
                        // width, which is what ubSTensor now declares, and the dequantization is done
                        // by the online-softmax epilogue instead of by the Fixpipe.
                        uint32_t nFixPAligned32 = RoundUp(qSTile, 32);
                        auto ubSTensorTlaTile = GetTile(ubSTensor, tla::MakeCoord(microMOffset, 0),
                                                        tla::MakeShape(mFixPAligned8, nFixPAligned32));
                        copyL0CToDst(ubSTensorTlaTile, l0CTensorTla);
                    } else {
                        uint32_t nFixPAligned8 = RoundUp(qSTile, 8);
                        auto ubSTensorTlaTile = GetTile(ubSTensor, tla::MakeCoord(microMOffset, 0),
                                                        tla::MakeShape(mFixPAligned8, nFixPAligned8));
                        copyL0CToDst(ubSTensorTlaTile, l0CTensorTla);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CMicroEventId);
                }
            }
        }
        // crossCoreSync after all fixPipe move
        SetCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementQ> l1QTensor[MAX_L1_STAGES];
    AscendC::LocalTensor<ElementK> l1KTensor[MAX_L1_STAGES];
    AscendC::LocalTensor<ElementQ> l0ATensor[L0_STAGES * L0_MICRO_STAGES];
    AscendC::LocalTensor<ElementK> l0BTensor[L0_STAGES * L0_MICRO_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[L0_STAGES * L0_MICRO_STAGES];

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;

    uint32_t l1ATileM;
    uint32_t l1BTileN;
    uint32_t l1ATileK;
    uint32_t l1BTileK;
    uint32_t l1QBufNum;
    uint32_t l1KBufNum;

    uint32_t l1PPingPongFlag = 0;
    uint32_t l0CPingPongFlag = 0;
    uint32_t l0ABPingPongFlag = 0;
};
////////////////////////////////////////////////////////////////////

} // namespace NpuArch::Gemm::Block
#endif // GEMM_BLOCK_K_QT_ST_ARCH35_HIF8_HPP
