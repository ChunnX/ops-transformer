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
 * @brief matmul implementation for single p&v base tile
 * This implementation is designed for the following senario:
 * A full p base tile is loaded to L1 from UB, no workspace transit
 * A full v base tile is loaded to L1 from GM，relavent instructions launched before p base tile crossCore wait
 * A full p*v base tile is loaded to UB from l0C, no workspace transit
 */
#ifndef GEMM_BLOCK_PV_ARCH35_ABF16_C2UB_HIF8_HPP
#define GEMM_BLOCK_PV_ARCH35_ABF16_C2UB_HIF8_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/arch/bsa_cross_core_sync.hpp"
#include "../../../attn_infra/bsa_coord.hpp"
#include "../../../attn_infra/gemm/bsa_gemm_dispatch_policy.hpp"
#include "../../../attn_infra/gemm/bsa_helper.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/gemm/block/block_mmad_arch35_utils.hpp"
#include "../../../attn_infra/gemm/block/block_mmad_pv_arch35_ABf16_C_to_UB.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_gemm_tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_tile_mmad.hpp"
#include "../../../tla/layout_bsa.hpp"
#include "../../../tla/tensor_bsa.hpp"

////////////////////////////////////////////////////////////////////

namespace NpuArch::Gemm::Block {
////////////////////////////////////////////////////////////////////

template <class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_, class ElementBias_,
          class TileCopy_, class TileMmad_, bool transposedMm1>
struct BlockMmadTla<
    // if transposedMm1, P is actually nZ in UB, and nZ in L1,
    // and is transposed to zN while loaded to L0A.
    MmadAtlasA5BsaPVHif8<transposedMm1>, L1TileShape_, L0TileShape_, ElementA_, ElementB_, ElementC_, ElementBias_,
    TileCopy_, TileMmad_> {
public:
    using DispatchPolicy = MmadAtlasA5BsaPVHif8<transposedMm1>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;

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

    // ---- Plan C: QK 微切, PV 整块 ----
    //
    // K 是 PV 的**归约轴**: 沿 K 切开的两块是往同一块 L0C 累加的部分和, 第二条 MMAD
    // 必须等第一条 (同址累加), FixPipe 又必须等两块都完 —— 所以切 K 拿不到任何
    // Cube/FixPipe 穿插, 只会多一条 MMAD 和一次 MTE1 握手。故 Plan C 不再切:
    // 一条 MMAD 吃满 K=256, P[128,256] / V[256,128] 各 32 KiB 连续占一个 outer 槽。
    // (对照 MM1: 它切的是 M=kv 这条**自由轴**, 两块落进两个独立 L0C 片, 因此能穿插,
    //  Plan C 原样保留它的微切。)
    //
    // L0_MICRO_STAGES 保持 2 不动 —— 它管的是**物理槽粒度**(16 KiB), 不是 K 的切分。
    // 保持它才使得 l0ATensor[]/l0BTensor[] 的下标->偏移映射与 MM1 逐字节一致:
    //   槽 0,1 = outer 0 = MM1 的两个微块;  槽 2,3 = outer 1 = PV 的整块。
    // 事件号也因此天然错开 (MM1 占 0,1; PV 取 bufId*2 = 2), 无需重排。
    static constexpr uint32_t L0_MICRO_STAGES = 2;
    // Plan C: K 不再切分。整块 tile 一次进 L0, kMicroLoopNum 退化为 1,
    // microK 恒 0, initMmad 里 (microK == 0) 恒真 -> 等价于无累加。
    static constexpr uint32_t L0_TILE_K_MICRO = L0_TILE_K;
    static constexpr uint32_t L0A_MICRO_BUF_SIZE = ArchTag::L0A_SIZE / (L0_STAGES * L0_MICRO_STAGES);
    static constexpr uint32_t L0B_MICRO_BUF_SIZE = ArchTag::L0B_SIZE / (L0_STAGES * L0_MICRO_STAGES);
    static constexpr uint32_t L0C_MICRO_BUF_SIZE = ArchTag::L0C_SIZE / (L0_STAGES * L0_MICRO_STAGES);
    // 整块 K tile 现在要连续占满**一整个 outer 槽** (32 KiB), 不再是半个。
    // 这也是 Plan C 成立的前提: 槽里不再有"只放一个 16 KiB tile"的那半浪费。
    static_assert(L0A_PINGPONG_BUF_SIZE >= L0_TILE_M * L0_TILE_K_MICRO * sizeof(ElementA),
                  "L0 outer slot too small for the whole K tile of A (Plan C)");
    static_assert(L0B_PINGPONG_BUF_SIZE >= L0_TILE_K_MICRO * L0_TILE_N * sizeof(ElementB),
                  "L0 outer slot too small for the whole K tile of B (Plan C)");
    // L0C 不拆: 不切 K 之后只有一块 partial-sum tile, FixPipe 一次走完。
    static_assert(L0C_MICRO_BUF_SIZE >= L0_TILE_M * L0_TILE_N * sizeof(ElementAccumulator),
                  "L0C micro slot too small for the OTmp tile");
    static constexpr uint32_t L0A_EVENT_BASE = 0;
    static constexpr uint32_t L0B_EVENT_BASE = L0_STAGES * L0_MICRO_STAGES; // 4

    // 事件号是运行时算出来的, 越界编译不报错只会静默跑飞, 所以在这里把预算钉死。
    // 用到的最大的号是 L0B_EVENT_BASE + L0_STAGES * L0_MICRO_STAGES - 1 = 7,
    // 要求 MTE1<->M 这条队列至少有 8 面旗 (Ascend950PR: __NPU_ARCH__ == 3510 时
    // QUE_MAX_EVENT == 8)。不够就必须改设计, 不是调大数字能解决的。
    static_assert(L0B_EVENT_BASE + L0_STAGES * L0_MICRO_STAGES <= AscendC::QUE_MAX_EVENT,
                  "Plan B MM2 needs 8 event ids on the MTE1<->M queue; QUE_MAX_EVENT is smaller");

    static constexpr bool FULL_QUANT_FP8 = AscendC::IsSameType<ElementA, fp8_e4m3fn_t>::value;
    static constexpr bool FULL_QUANT_HIF8 = AscendC::IsSameType<ElementA, hifloat8_t>::value;

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart,
                                   Mm2L1TileHelper &mm2L1TileHelper)
    {
        l1ATileM = mm2L1TileHelper.mm2L1TileM;
        l1BTileN = mm2L1TileHelper.mm2L1TileN;
        l1ATileK = mm2L1TileHelper.mm2L1TileKLeft;
        l1BTileK = mm2L1TileHelper.mm2L1TileKRight;
        l1ABufNum = mm2L1TileHelper.pL1BufNum;
        l1BBufNum = mm2L1TileHelper.vL1BufNum;
        for (uint32_t i = 0; i < l1ABufNum; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart + l1ATileM * l1ATileK *
                                                                                                  sizeof(ElementA) * i);
        }
        for (uint32_t i = 0; i < l1BBufNum; i++) {
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(
                l1BufAddrStart + l1ATileM * l1ATileK * sizeof(ElementA) * l1ABufNum +
                l1BTileK * l1BTileN * sizeof(ElementB) * i);
        }
        // L0A/L0B 按小槽切: 下标 = outerId * L0_MICRO_STAGES + microK, 偏移即
        // outerId * 大槽 + microK * 小槽, 正好等于小槽大小的线性递推。
        for (uint32_t i = 0; i < L0_STAGES * L0_MICRO_STAGES; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_MICRO_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_MICRO_BUF_SIZE * i);
        }
        // L0C 也切成 4 片。Plan C 不切 K, 所以 MM2 只用每 outer 的**第 0 片**就够了
        // (整块 OTmp 不再是两个部分和的累加, 一次 MMAD + 一次 FixPipe 走完)。
        // 外层槽 stride 从 128 KiB 收到 64 KiB, 与 MM1 的 L0C 布局对齐 ——
        // 否则两个 MM 会用两套粒度盖同一片物理空间。
        for (uint32_t i = 0; i < L0_STAGES * L0_MICRO_STAGES; i++) {
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_MICRO_BUF_SIZE * i);
        }
    }

    /// Destructor
    __aicore__ inline ~BlockMmadTla() {}

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
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
    __aicore__ inline void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
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

    __aicore__ inline uint32_t GetCurLoopCounter(uint32_t outterLoopItr, uint32_t curLoopNum, uint32_t curLoopItr)
    {
        return outterLoopItr * curLoopNum + curLoopItr;
    }

    template <class TensorB, class TensorL1B>
    __aicore__ inline void SparseVBaseTileL1FullLoad(TensorB &gBTensor, TensorL1B &l1BTensorTla,
                                                     AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
                                                     uint32_t gatheredKvSTileIdx, uint32_t kvSeqlen,
                                                     uint32_t kvSBaseTile, uint32_t blockShapeY, uint32_t yBlockNumAval,
                                                     uint32_t yBlockNumRsvd, uint32_t curBaseTileSize, uint32_t embed)
    {
        using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorB>;
        CopyGmToL1B copyGmToL1B;
        uint32_t baseTileStartOffset = gatheredKvSTileIdx * kvSBaseTile;
        uint32_t baseTileEndOffset = baseTileStartOffset + curBaseTileSize;
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

        while (dealtLenAccum < curBaseTileSize && gatheredYBlockIdx < yBlockNumRsvd && oriYBlockIdx < yBlockNumAval &&
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
            auto l1BTensorTlaTile =
                GetTile(l1BTensorTla, tla::MakeCoord(dealtLenAccum, 0), tla::MakeShape(curDealtLen, embed));
            auto gBTensorTlaTile =
                GetTile(gBTensor, tla::MakeCoord(oriStartOffset, 0), tla::MakeShape(curDealtLen, embed));
            copyGmToL1B(l1BTensorTlaTile, gBTensorTlaTile);
            // 为下一次循环刷新循环变量
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

    template <class TensorB, class TensorC>
    __aicore__ inline void operator()(TensorB &gBTensor, TensorC &ubCTensor,
                                      AscendC::GlobalTensor<int32_t> gSparseBlockIdx, GemmCoord actualOriShape,
                                      uint32_t gatheredKvSTileIdx, uint32_t kvSeqlen, uint32_t kvSBaseTile,
                                      uint32_t blockShapeY, uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
                                      uint64_t &l0Stage,
                                      Arch::CrossCoreFlag smToMm2Flag, Arch::CrossCoreFlag mm2ToReFlag)
    {
        using CopyL0CToDst = typename TileCopy_::template CopyL0CToDst<TensorC>;

        const uint32_t bufId = static_cast<uint32_t>(l0Stage++ % L0_STAGES);
        // 槽的事件: 下标 = 基址 + bufId * L0_MICRO_STAGES + microK。
        // Plan C 不切 K, microK 恒 0, 所以 MM2 在 MTE1_M 这条队列上只用 L0A 的 2 号
        // 和 L0B 的 6 号各一个 id (MM1 恒 outer 0, 占 L0A 0/1 与 L0B 4 号, 天然错开)。
        // L0C 走 M_FIX/FIX_M 另一条队列, 只取每 outer 的第 0 片, 号是 0 和 2。
        const uint32_t l0AMicroBase = L0A_EVENT_BASE + bufId * L0_MICRO_STAGES;
        const uint32_t l0BMicroBase = L0B_EVENT_BASE + bufId * L0_MICRO_STAGES;
        const uint32_t l0CEventId = bufId * L0_MICRO_STAGES;

        uint32_t rowNum = actualOriShape[0];
        uint32_t embed = actualOriShape[1];
        uint32_t curBaseTileSize = actualOriShape[2];

        uint32_t l1BBufId = gatheredKvSTileIdx % l1BBufNum;
        uint32_t l1BEventId = l1BBufId + 3;
        uint32_t l1ABufId = gatheredKvSTileIdx % l1ABufNum;

        auto l1BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL1B>(curBaseTileSize, embed);
        auto l1BTensorTla = tla::MakeTensor(l1BTensor[l1BBufId], l1BLayoutTla, Arch::PositionL1{});
        auto l1ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL1A>(rowNum, curBaseTileSize);
        auto l1ATensorTla = tla::MakeTensor(l1ATensor[l1ABufId], l1ALayoutTla, Arch::PositionL1{});
        // load V full base tile to L1 before crossCoreSync
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);
        SparseVBaseTileL1FullLoad(gBTensor, l1BTensorTla, gSparseBlockIdx, gatheredKvSTileIdx, kvSeqlen, kvSBaseTile,
                                  blockShapeY, yBlockNumAval, yBlockNumRsvd, curBaseTileSize, embed);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);
        // fwd crossCoreSync from online sm to mm2
        WaitCrossCoreSync<4, PIPE_MTE1>(smToMm2Flag);
        // P full base tile already on L1
        uint32_t mL0LoopNum = CeilDiv(rowNum, L0_TILE_M);
        uint32_t nL0LoopNum = CeilDiv(embed, L0_TILE_N);
        uint32_t kL0LoopNum = CeilDiv(curBaseTileSize, L0_TILE_K);
        // while splitting the base tile OTmp to 2 AIVs,
        // the order of the elements in each column is expected to be preserved,
        // which means a column in l0C cannot be chunked and processed by dualMode FixPipe seperately.
        // therefore, FixPipe won't launch until each portion(chunked only by columns, based on nbuffer strategy)
        // of the base tile is ready on l0C
        for (uint32_t nL0Itr = 0; nL0Itr < nL0LoopNum; nL0Itr++) {
            uint32_t l0TileNAct = (nL0Itr == nL0LoopNum - 1) ? (embed - nL0Itr * L0_TILE_N) : L0_TILE_N;
            auto l0CLayoutTla = tla::MakeLayoutL0C(rowNum, l0TileNAct);
            auto l0CTensorTla = tla::MakeTensor(l0CTensor[l0CEventId], l0CLayoutTla, Arch::PositionL0C{});
            for (uint32_t mL0Itr = 0; mL0Itr < mL0LoopNum; mL0Itr++) {
                uint32_t l0TileMAct = (mL0Itr == mL0LoopNum - 1) ? (rowNum - mL0Itr * L0_TILE_M) : L0_TILE_M;
                // uint32_t mLoopCounter = GetCurLoopCounter(gatheredKvSTileIdx, mL0LoopNum, mL0Itr);
                // different m chunks will be concated in the same piece of l0C buffer
                auto l0CTensorTlaTile = GetTile(l0CTensorTla, tla::MakeCoord(mL0Itr * L0_TILE_M, 0),
                                                tla::MakeShape(l0TileMAct, l0TileNAct));
                for (uint32_t kL0Itr = 0; kL0Itr < kL0LoopNum; kL0Itr++) {
                    uint32_t l0TileKAct =
                        (kL0Itr == kL0LoopNum - 1) ? (curBaseTileSize - kL0Itr * L0_TILE_K) : L0_TILE_K;
                    uint32_t kMicroLoopNum = CeilDiv(l0TileKAct, L0_TILE_K_MICRO);
                    for (uint32_t microK = 0; microK < kMicroLoopNum; microK++) {
                        uint32_t microKAct = (microK == kMicroLoopNum - 1)
                                                 ? (l0TileKAct - microK * L0_TILE_K_MICRO)
                                                 : L0_TILE_K_MICRO;
                        uint32_t microKOffset = kL0Itr * L0_TILE_K + microK * L0_TILE_K_MICRO;
                        // 小槽按下标取。取模是为了 curBaseTileSize 超过一个 L0_TILE_K 时
                        // 仍只在已分配的小槽里轮转, 不越界; 槽内的 M_MTE1/MTE1_M 握手序
                        // 保证复用同一小槽时不会写读打架。
                        // Plan C 下 kMicroLoopNum 恒 1、microK 恒 0, 取模是恒等操作,
                        // 保留是为了 KV 尾块 (curBaseTileSize < L0_TILE_K) 时公式不特判。
                        const uint32_t l0AEventId = l0AMicroBase + microK % L0_MICRO_STAGES;
                        // L0A 事件基址 0, 事件号即缓冲片下标; L0B 基址 4, 两者要分开。
                        const uint32_t l0BSlot = bufId * L0_MICRO_STAGES + microK % L0_MICRO_STAGES;
                        const uint32_t l0BEventId = l0BMicroBase + microK % L0_MICRO_STAGES;
                        bool lastK = (kL0Itr == kL0LoopNum - 1) && (microK == kMicroLoopNum - 1);

                        // when L0B buffers wouldn't be reused across the k loop
                        // redundant L0B load caused by m loop can be avoided
                        auto l1BTensorTlaTile =
                            GetTile(l1BTensorTla, tla::MakeCoord(microKOffset, nL0Itr * L0_TILE_N),
                                    tla::MakeShape(microKAct, l0TileNAct));
                        auto l0BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL0B>(microKAct, l0TileNAct);
                        auto l0BTensorTla =
                            tla::MakeTensor(l0BTensor[l0BSlot], l0BLayoutTla, Arch::PositionL0B{});

                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                        copyL1ToL0B(l0BTensorTla, l1BTensorTlaTile);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);

                        if ((mL0Itr == mL0LoopNum - 1) && (nL0Itr == nL0LoopNum - 1) && lastK) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);
                        }

                        auto l1ATensorTlaTile =
                            GetTile(l1ATensorTla, tla::MakeCoord(mL0Itr * L0_TILE_M, microKOffset),
                                    tla::MakeShape(l0TileMAct, microKAct));
                        auto l0ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL0A>(l0TileMAct, microKAct);
                        auto l0ATensorTla =
                            tla::MakeTensor(l0ATensor[l0AEventId], l0ALayoutTla, Arch::PositionL0A{});

                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                        copyL1ToL0A(l0ATensorTla, l1ATensorTlaTile);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                        // reverse crossCoreSync for P
                        if ((mL0Itr == mL0LoopNum - 1) && (nL0Itr == nL0LoopNum - 1) && lastK) {
                            SetCrossCoreSync<4, PIPE_MTE1>(smToMm2Flag);
                        }

                        // Plan C 不切 K, microK 恒 0, 所以这一条退化成
                        // "每个 KV tile 的第一条 MMAD 覆盖写" —— **没有部分和累加**。
                        // (Plan B 的 (microK == 0) 条件在这里恒真, 保留是为了与
                        //  curBaseTileSize > L0_TILE_K 的多趟 K 循环共用同一套公式。)
                        bool initMmad = (kL0Itr == 0) && (microK == 0);
                        uint32_t l0TileMAligned = RoundUp(l0TileMAct, 16);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);

                        if (mL0Itr == 0 && kL0Itr == 0 && microK == 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
                        }
                        tileMmad(l0CTensorTlaTile, l0ATensorTla, l0BTensorTla, l0TileMAligned, l0TileNAct, microKAct,
                                 initMmad);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                    }
                }
            }
            // fixpipe
            if (nL0Itr == 0) {
                // reverse crossCoreSync, do fixPipe only after ubCTensor is fully released
                WaitCrossCoreSync<4, PIPE_FIX>(mm2ToReFlag);
            }
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);

            uint32_t mFixPAligned = 0;
            if constexpr (DispatchPolicy::transposedMm1) {
                mFixPAligned = RoundUp(rowNum, 32);
            } else {
                mFixPAligned = RoundUp(rowNum, 8);
            }
            uint32_t nFixPAligned8 = RoundUp(l0TileNAct, 8);
            // SPLIT_M + NO_QUANT: one dual-destination Fixpipe serves both AIVs, the hardware
            // writing rows 0..m/2-1 into AIV0 and m/2..m-1 into AIV1 at the same UB offsets
            // given by ubCTensor's per-AIV stride. The dst shape therefore carries the full M
            // (it becomes mSize) and the full N of the chunk; the dequantization is done by the
            // rescale-O epilogue instead of by the Fixpipe.
            CopyL0CToDst copyL0CToDst;
            auto ubCTensorTlaTile = GetTile(ubCTensor, tla::MakeCoord(0, nL0Itr * L0_TILE_N),
                                            tla::MakeShape(mFixPAligned, nFixPAligned8));
            copyL0CToDst(ubCTensorTlaTile, l0CTensorTla);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
        }
        // crossCoreSync after all fixPipe move
        SetCrossCoreSync<4, PIPE_FIX>(mm2ToReFlag);
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensor[MAX_L1_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor[MAX_L1_STAGES];
    AscendC::LocalTensor<ElementA> l0ATensor[L0_STAGES * L0_MICRO_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[L0_STAGES * L0_MICRO_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[L0_STAGES * L0_MICRO_STAGES];

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

} // namespace NpuArch::Gemm::Block
#endif // GEMM_BLOCK_PV_ARCH35_ABF16_C2UB_HIF8_HPP
