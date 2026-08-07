/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_BLOCK_QK_ARCH35_MXFP4_HPP
#define GEMM_BLOCK_QK_ARCH35_MXFP4_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/arch/bsa_cross_core_sync.hpp"
#include "../../../attn_infra/bsa_coord.hpp"
#include "../../../attn_infra/gemm/bsa_gemm_dispatch_policy.hpp"
#include "../../../attn_infra/gemm/bsa_helper.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_gemm_tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_tile_mmad.hpp"
#include "block_mmad_arch35_utils.hpp"
// [Step2] QK 专用 mx L1→L0（非转置）；与 PV 的 CopyL1ToL0*MxA5 同文件但不同 functor
#include "../../../attn_infra/gemm/tile_common/copy_l1_to_l0_mx_a5.hpp"
// [Step5] Q/K-scale GM→L1：沿 D Dn2Nz（本步先接 Q-scale；K-scale Step6 复用；勿用 PV MxScale）
//   Q 数据本身走 TileCopyTla RowMajor→zN（见 CopyQGmToL1，头文件已由 bsa_gemm_tile_copy 引入）
#include "../../../attn_infra/gemm/tile_common/copy_gm_to_l1_mx_scale_dn2nz_a5.hpp"
#include "../../../tla/layout.hpp"
#include "../../../tla/tensor.hpp"

namespace NpuArch::Gemm::Block {

template <
    class L1TileShape_,
    class L0TileShape_,
    class ElementA_,
    class ElementB_,
    class ElementC_,
    class ElementBias_,
    class TileCopy_,
    class TileMmad_>
struct BlockMmadTla<
    MmadAtlasA5BsaQK<true>,
    L1TileShape_,
    L0TileShape_,
    ElementA_,
    ElementB_,
    ElementC_,
    ElementBias_,
    TileCopy_,
    TileMmad_>
{
public:
    using DispatchPolicy = MmadAtlasA5BsaQK<true>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using ElementA = ElementA_;  // Q
    using ElementB = ElementB_;  // K
    using ElementC = ElementC_;  // S

    using TileMmad = TileMmad_;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;


    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape_{});

    // [QFA 对齐] Q(M) 方向全程 pad 到 128（QFA actMSizeAlign128 语义）：
    //   L1 画布 / L0B 加载 mStep/xStep / Mmad n / Fixpipe nSize 统一用 128；
    //   有效行仅 rowNumQ，其余为残留垃圾列，vec 侧只读有效列（与 QFA 相同）。
    //   原实现对齐到 CeilAlign(m,16)，S1<128 时 Fixpipe nSize=m 违反 32B 倍数约束导致全错。
    static constexpr uint32_t Q_M_PAD = 128;

    static constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};

    // L1 Q（MTE1_MTE2 / MTE2_MTE1）
    static constexpr uint32_t Q_EVENT0 = 2;
    static constexpr uint32_t Q_EVENT1 = 3;
    // [QK/PV 共享] L1 K 复用 MXFP4::KV_EVENT0..3(4/5/6/7)——与 PV(V) 同组 MTE1_MTE2 门，不再独占 6/7

    // L0A/B（M_MTE1 / MTE1_M）
    static constexpr uint32_t QK_L0AB_EVENT0 = 2;
    static constexpr uint32_t QK_L0AB_EVENT1 = 3;
    // L0C（FIX_M / M_FIX）
    static constexpr uint32_t QK_L0C_EVENT0 = 2;
    static constexpr uint32_t QK_L0C_EVENT1 = 3;

    
    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag> &resource, uint32_t& kvBufId) : kBufId(kvBufId) {
        // L1 区域顺序（与 QFA 一致）：P 数据 | P-scale | Q 数据 | Q-scale | KV(=V) 数据 | KV(=V)-scale

        // Q 数据 占 L1_Q_SIZE * L1_Q_BUFCNT = 16K
        for (uint32_t i = 0; i < MXFP4::L1_Q_BUF_CNT; i++) {
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(MXFP4::L1_Q_BUF_OFFSET + MXFP4::L1_Q_BUF_SIZE * i);
        }

        // Q-scale：数据 占 L1_Q_DESCALE_SIZE * L1_Q_BUFCNT = 1K
        for (uint32_t i = 0; i < MXFP4::L1_Q_BUF_CNT; i++) {
            l1BScaleTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(MXFP4::L1_Q_DESCALE_BUF_OFFSET + MXFP4::L1_Q_DESCALE_BUF_SIZE * i);
        }

        // [QK/PV 共享] K 数据：l1BTensor(B=K) 与 PV 的 V(l1ATensor) 别名同一组 4 槽
        //   直取 L1_KV_BUF_OFFSET（不再 +CNT 跳过 V 区），与 block_mmad_pv ctor :121-123 同式
        for (uint32_t i = 0; i < MXFP4::L1_KV_BUF_CNT; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(MXFP4::L1_KV_BUF_OFFSET + MXFP4::L1_KV_BUF_SIZE * i);
        }

        // [QK/PV 共享] K-scale 与 PV V-scale 别名同一组 4 槽（直取 L1_KV_DESCALE_BUF_OFFSET，不再用 addrL1Start）
        for (uint32_t i = 0; i < MXFP4::L1_KV_DESCALE_BUF_CNT; i++) {
            l1AScaleTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(MXFP4::L1_KV_DESCALE_BUF_OFFSET + MXFP4::L1_KV_DESCALE_BUF_SIZE * i);
        }

        // L0A 区域顺序  QK 数据 | PV 数据
        for (uint32_t i = 0; i < MXFP4::L0A_QK_BUF_CNT; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementB>(MXFP4::L0A_QK_BUF_OFFSET + MXFP4::L0A_QK_BUF_SIZE * i);
        }

        // L0B 区域顺序  QK 数据 | PV 数据
        for (uint32_t i = 0; i < MXFP4::L0B_QK_BUF_CNT; i++) {
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementA>(MXFP4::L0B_QK_BUF_OFFSET + MXFP4::L0B_QK_BUF_SIZE * i);
        }

        // L0C：QK 双缓冲 [0,128K)=2×64K；PV 在 MXFP4::L0C_PV_BUF_OFFSET=128K（勿再叠到 64K）
        for (uint32_t i = 0; i < MXFP4::L0C_QK_BUF_CNT; i++) {
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<float>(
                MXFP4::L0C_QK_BUF_OFFSET + MXFP4::L0C_QK_BUF_SIZE * i);
        }

        // ====================== Event 预置（对应 QFA / PV AllocEventID）======================
        // Step0：只灌令牌，operator() 仍走单缓冲串行；Step1+ 再 Wait 消费这些 flag。
        AllocEventID();
    }
    
    // ============================================================================
    // AllocEventID：预置所有缓冲槽为“可写”（冷启动令牌），对应 QFA/PV AllocEventID
    // ============================================================================
    __aicore__ inline void AllocEventID()
    {
        // L1 Q 双槽（MTE1_MTE2：LoadQ→MTE2 可覆盖写入）
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT1);
        // [QK/PV 共享] KV 池 4 路全预置（单一预置点；PV 的 AllocEventID 不再重复预置 KV）
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT3);
        // L0A/B 双缓冲（M_MTE1：Mmad→MTE1 可覆盖写入）
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT1);
        // L0C 双缓冲（FIX_M：Fixpipe→M 可覆盖写入）
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT1);
    }

    // ============================================================================
    // FreeEventID：等所有预置/归还的 event 排空，对应 QFA/PV FreeEventID
    //   Step5 由 kernel AIC 退出前调用；Step0 仅提供接口，暂不强制挂钩。
    // ============================================================================
    __aicore__ inline void FreeEventID()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT1);
        // [QK/PV 共享] KV 池 4 路全排空（PV 的 FreeEventID 不再排空 KV）
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MXFP4::KV_EVENT3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT1);
    }


    __aicore__ inline ~BlockMmadTla() {}

    // template <class TensorA>
    // __aicore__ inline void loadQGM(TensorA &gATensor, GemmCoord actualOriShape) {}

    __aicore__ inline uint16_t CeilDivision(uint16_t numerator, uint16_t denominator)
    {
        return (numerator + denominator - 1) / denominator;
    }

    __aicore__ inline uint16_t CeilAlign(uint16_t numerator, uint16_t denominator)
    {
        if (denominator == 0)
            return numerator;
        return (numerator + denominator - 1) / denominator * denominator;
    }

    // [Step5] Q 稠密 GM→L1：TileCopyTla RowMajor→zN（内部仍是 Nd2Nz DataCopy）
    //   shape cols = k(fp4 元素个数)；特化内 dValue/srcDValue /=2 → 与裸 packedK Nd2Nz 对齐
    //   [QFA 对齐] L1 画布恒为 Q_M_PAD=128 行（dstNzC0Stride=128，对齐 QFA rowCount=128）；
    //   GetTile(m,k) 只搬 m 行有效数据，m..127 行为残留垃圾（与 QFA 语义一致）
    template <class TensorQ, class TensorL1Q>
    __aicore__ inline void CopyQGmToL1(uint32_t qsActBaseTile,
                                       uint32_t embed,   // embed：fp4 元素个数（非 packedK）
                                       TensorQ gQTensorTla,
                                       TensorL1Q l1QTensorTla)  
    {
        // [BSND stride 修复] Q GM row-stride(头内 token-stride)=qHeadMul*k(BNSD=k); RowMajor cols→row-stride
        auto l1QTileTla = GetTile(l1QTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(qsActBaseTile, embed));
        // [BSND stride 修复] GetTile src 到 (m,k): tile shape 给 dValue=k(正解, 否则整个 gQTensorTla
        //   shape (m, qHeadMul*k) 会让 RowMajor→zN 的 dValue=get<1>(shape)=qHeadMul*k 放大→每 token
        //   多读→Q 错→S 全错 1.36%)。tile 保留 parent 的 row-stride=qHeadMul*k(正解 token-stride)。
        //   V/K 都 GetTile src, 唯 Q 漏了, 故 Q 须补此 GetTile(对 BNSD qHeadMul=1 无变化)。
        auto gQTileTla = GetTile(gQTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(qsActBaseTile, embed));

        // [BSND verify] Q gather 期望值: 对照紧跟的 [CPY-RM2ZN] 实际 dVal(应=k)/srcDVal(应=qHeadMul*k)
        // AscendC::printf("[Q-GATHER] qHeadMul=%u m=%u k=%u expDVal=%u expSrcDVal=%u\n",
        //     (unsigned)qHeadMul, (unsigned)m, (unsigned)k, (unsigned)k, (unsigned)(qHeadMul * k));
        using CopyGmToL1Q = Tile::TileCopyTla<ArchTag, decltype(gQTileTla), decltype(l1QTensorTla)>;
        CopyGmToL1Q copyGmToL1Q;
        copyGmToL1Q(l1QTileTla, gQTileTla);
    }

    // [Step5] Q-scale 稠密 GM→L1：CopyGmToL1MxScaleDn2NzA5（Host 沿 D；禁止 PV MxScale）
    // Host BNSD：[B,N1,S1,D//64,2]；每 token scaleK=Ceil(D/64)*2 字节，行主序 [S, scaleK]
    __aicore__ inline void CopyQScaleGmToL1(uint32_t qsActBaseTile,
                                            uint32_t scaleK,   // = Ceil(D/64)*2
                                            AscendC::GlobalTensor<uint8_t> &gQDequantScale,
                                            uint32_t qBufId,
                                            uint32_t qHeadMul)
    {
        copyGmToL1MxScaleDn2Nz(l1BScaleTensor[qBufId], gQDequantScale, qsActBaseTile, scaleK,
                               0, 0, qHeadMul);
    }


    // Step3：device 侧无 std::min，gather 段长截断用
    __aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b)
    {
        return (a < b) ? a : b;
    }

    // ========== Step6：稀疏 K gather（GetTile + TileCopyTla）==========
    // 依赖公共 fp4 粒度修正：
    //   layout/copy_gm_to_l1 按 SizeOfBits 计算 ELE_NUM_PER_C0(=64 for fp4)。
    // 分段映射保持与 baseline 一致：gathered->ori Y-block，段内 RowMajor->zN 搬运。
    template <class TensorK, class TensorL1K>
    __aicore__ inline
    void SparseKBaseTileL1FullLoad(
        TensorK gKTensorTla, 
        TensorL1K l1KTensorTla, 
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx, // 本 task 行：有效 KV 列下标表
        uint32_t gatheredKvSTileIdx,                   // gathered 序列上第几个 kvBaseTile
        int64_t kvSeqlen,                              // 原始 KV 序列长度（末块截断）
        uint32_t kvSBaseTile,                          // tiling 的 kvBaseTile_
        uint32_t blockShapeY,                          // 稀疏 Y 块高
        uint32_t yBlockNumAval,                        // 原始有效 Y-block 数
        uint32_t yBlockNumRsvd,                        // gathered 行宽（含 reserved）
        uint32_t kvsActBaseTile,                      // 本 tile 实际 K 行数
        uint32_t embed)                            
    {
        uint32_t baseTileStartOffset = gatheredKvSTileIdx * kvSBaseTile;
        uint32_t baseTileEndOffset   = baseTileStartOffset + kvsActBaseTile;
        uint32_t gatheredStartOffset = baseTileStartOffset;
        uint32_t gatheredYBlockIdx   = gatheredStartOffset / blockShapeY;
        uint32_t yBlockInnerStart    = gatheredStartOffset % blockShapeY;
        uint32_t oriYBlockIdx   = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
        uint32_t oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStart;
        uint32_t dealtLenAccum  = 0;
        using CopyGmToL1K = Tile::TileCopyTla<ArchTag, decltype(gKTensorTla), decltype(l1KTensorTla)>;
        CopyGmToL1K copyGmToL1K;

        while (dealtLenAccum < kvsActBaseTile &&
               gatheredYBlockIdx < yBlockNumRsvd &&
               oriYBlockIdx < yBlockNumAval &&
               oriStartOffset < static_cast<uint32_t>(kvSeqlen)) {
            uint32_t curYBlockSize = (oriYBlockIdx == yBlockNumAval - 1)
                                     ? static_cast<uint32_t>(kvSeqlen - oriYBlockIdx * blockShapeY)
                                     : blockShapeY;
            uint32_t gatheredEndOffset = MinU32(gatheredYBlockIdx * blockShapeY + curYBlockSize,
                                                baseTileEndOffset);
            uint32_t curDealtLen = gatheredEndOffset - gatheredStartOffset;
            if (curDealtLen == 0) { break; }

            // AscendC::printf(
            //     "[K-GATHER] seg: oriYBlock=%u oriStart=%u curDealtLen=%u dealtAccum=%u "
            //     "srcOff=%u dstOff=%u c0Stride=%u\n",
            //     (unsigned)oriYBlockIdx, (unsigned)oriStartOffset, (unsigned)curDealtLen,
            //     (unsigned)dealtLenAccum,
            //     (unsigned)(oriStartOffset * embed), (unsigned)(dealtLenAccum * 64),
            //     (unsigned)s2Align64);
            auto gKTile = GetTile(
                gKTensorTla, tla::MakeCoord(oriStartOffset, 0), tla::MakeShape(curDealtLen, embed));
            auto l1KTile = GetTile(
                l1KTensorTla, tla::MakeCoord(dealtLenAccum, 0), tla::MakeShape(curDealtLen, embed));
            copyGmToL1K(l1KTile, gKTile);

            dealtLenAccum += curDealtLen;
            gatheredStartOffset += curDealtLen;
            gatheredYBlockIdx = gatheredStartOffset / blockShapeY;
            yBlockInnerStart   = gatheredStartOffset % blockShapeY;
            if (dealtLenAccum < kvsActBaseTile) {
                oriYBlockIdx   = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
                oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStart;
            }
        }
        // [STEP0 golden] L1 K 数据（gather 完成后；NZ 布局，前 256B 采样）
        // AscendC::DumpTensor(l1BTensor[0].template ReinterpretCast<uint8_t>(), 2201, 128);
        // AscendC::DumpTensor(l1BTensor[0].template ReinterpretCast<uint8_t>()[128], 2202, 128);
    }

    // ========== Step6：稀疏 K-scale gather → CopyGmToL1MxScaleDn2NzA5（沿 D Dn2Nz）==========
    // 与 SparseK 同一套 gathered→ori 行映射；每段 Dn2Nz，B16 偏移拼接。
    __aicore__ inline
    void SparseKScaleBaseTileL1FullLoad(
        AscendC::GlobalTensor<uint8_t> gKScale,       // 已带 batch+head 偏移
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
        uint32_t gatheredKvSTileIdx, int64_t kvSeqlen,
        uint32_t kvSBaseTile, uint32_t blockShapeY,
        uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
        uint32_t kvsActBaseTile, uint32_t embed,
        uint32_t kBufId, uint32_t kvHeadMul)
    {
        uint32_t scaleK = CeilDivision(embed, 64) * 2;
        uint32_t scaleRowHalf = scaleK / 2;

        uint32_t baseTileStartOffset = gatheredKvSTileIdx * kvSBaseTile;
        uint32_t baseTileEndOffset   = baseTileStartOffset + kvsActBaseTile;
        uint32_t gatheredStartOffset = baseTileStartOffset;
        uint32_t gatheredYBlockIdx   = gatheredStartOffset / blockShapeY;
        uint32_t yBlockInnerStart    = gatheredStartOffset % blockShapeY;
        uint32_t oriYBlockIdx   = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
        uint32_t oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStart;
        uint32_t dealtLenAccum  = 0;

        while (dealtLenAccum < kvsActBaseTile && gatheredYBlockIdx < yBlockNumRsvd &&
               oriYBlockIdx < yBlockNumAval && oriStartOffset < static_cast<uint32_t>(kvSeqlen)) {
            uint32_t curYBlockSize = (oriYBlockIdx == yBlockNumAval - 1)
                                     ? static_cast<uint32_t>(kvSeqlen - oriYBlockIdx * blockShapeY)
                                     : blockShapeY;
            uint32_t gatheredEndOffset = MinU32(gatheredYBlockIdx * blockShapeY + curYBlockSize,
                                                baseTileEndOffset);
            uint32_t curDealtLen = gatheredEndOffset - gatheredStartOffset;
            if (curDealtLen == 0) { break; }

            uint32_t srcB16Off = oriStartOffset * kvHeadMul * scaleRowHalf;  // [BSND stride 修复] K-scale 头内 token stride 乘 kvHeadMul
            uint32_t dstB16Off = dealtLenAccum * scaleRowHalf;
            // AscendC::printf(
            //     "[KS-GATHER] seg: oriStart=%u curDealtLen=%u dealtAccum=%u scaleK=%u "
            //     "srcB16Off=%u dstB16Off=%u\n",
            //     (unsigned)oriStartOffset, (unsigned)curDealtLen, (unsigned)dealtLenAccum,
            //     (unsigned)scaleK, (unsigned)srcB16Off, (unsigned)dstB16Off);
            // src 起点已乘 kvHeadMul；段内逐行步长也须乘 kvHeadMul（TND/BSND），
            // 否则段内第 2 行起读到邻头/错 token 的 scale（与 Q-scale 同构缺陷）。
            copyGmToL1MxScaleDn2Nz(
                l1AScaleTensor[kBufId], gKScale, curDealtLen, scaleK, srcB16Off, dstB16Off,
                kvHeadMul);

            dealtLenAccum += curDealtLen;
            gatheredStartOffset += curDealtLen;
            gatheredYBlockIdx = gatheredStartOffset / blockShapeY;
            yBlockInnerStart   = gatheredStartOffset % blockShapeY;
            if (dealtLenAccum < kvsActBaseTile) {
                oriYBlockIdx   = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
                oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStart;
            }
        }
        // AscendC::DumpTensor(l1BScaleTensor[0], 2221, 128);
        // AscendC::DumpTensor(l1BScaleTensor[0][128], 2222, 128);
    }
    
    // [Step2] Q→L0B 改走 mx functor（原裸 5 参 LoadData 下沉到 CopyL1ToL0BMxQKA5）
    // Q-scale 沿 D（Host [S,D//64,2]）；ifTranspose=false 由 functor 内部设定。
    // [QFA 对齐] m 方向三个对齐参数调用处统一传 Q_M_PAD=128（mStep=8/xStep=8）
    __aicore__ inline void LoadQToL0B(uint32_t qsActBaseTileAlign16,
                                      uint32_t embed,
                                      uint32_t qsActBaseTileAlign16L0,
                                      uint32_t scaleMAlign16,
                                      uint32_t scaleK,
                                      uint32_t qBufId, 
                                      uint32_t qkL0abBufId)   // = Ceil(D/64)*2，沿 D
    {
        copyL1ToL0BMxQk(
            l0BTensor[qkL0abBufId].template ReinterpretCast<fp4x2_e2m1_t>(),
            l1BTensor[qBufId].template ReinterpretCast<fp4x2_e2m1_t>(),
            l1BScaleTensor[qBufId].template ReinterpretCast<AscendC::fp8_e8m0_t>(),
            qsActBaseTileAlign16, embed, qsActBaseTileAlign16L0, scaleMAlign16, scaleK);
    }

    // [Step2] K→L0A 改走 mx functor（原裸 LoadKToL0A 下沉到 CopyL1ToL0AMxQKA5）
    // n 方向子切分：L1 已 gather 满整 tile，本次只搬 [nSubRowStart, +nCur)；scale 沿 D。
    __aicore__ inline void LoadKToL0A(uint32_t nSubRowStart,     // = nSub * L0_TILE_N
                                      uint32_t nCur,             // 本子块实际 K 行数(≤L0_TILE_N)
                                      uint32_t embed,
                                      uint32_t kvsActBaseTileAlign16,    // 全 tile 行 16 对齐(K 数据源 stride，与 zN 画布/QFA 一致)
                                      uint32_t scaleK,
                                      uint32_t kBufId,
                                      uint32_t qkL0abBufId)           // = Ceil(D/64)*2，沿 D
    {
        // AscendC::printf(
        //     "[K-L0A] nSubStart=%u nCur=%u mStart=%u mStep=%u kStep=%u srcStr=%u | "
        //     "mx xStart=%u xStep=%u yStep=%u scaleK=%u\n",
        //     (unsigned)nSubRowStart, (unsigned)nCur,
        //     (unsigned)(nSubRowStart / 16), (unsigned)(CeilAlign(nCur, 16) / 16),
        //     (unsigned)CeilDivision(alignK, 64), (unsigned)(s2Align16Full / 16),
        //     (unsigned)(nSubRowStart / 16), (unsigned)(CeilAlign(nCur, 16) / 16),
        //     (unsigned)(scaleK / 2), (unsigned)scaleK);
        copyL1ToL0AMxQk(
            l0ATensor[qkL0abBufId].template ReinterpretCast<fp4x2_e2m1_t>(),
            l1ATensor[kBufId].template ReinterpretCast<fp4x2_e2m1_t>(),
            l1AScaleTensor[kBufId].template ReinterpretCast<AscendC::fp8_e8m0_t>(),
            nSubRowStart, nCur, embed, kvsActBaseTileAlign16, scaleK);
    }

    // [Step3] Mmad 改走 tileMmad（原裸 AscendC::Mmad 下沉到 TileMmadTla）
    // C=[n,m]=[K行,Q行(pad Q_M_PAD)]；A=K@L0A，B=Q@L0B；每 nSub 独立计算，initC 恒 true。
    // tileMmad 只读 .data()（不消费 layout stride），与 PV::MatmulPV 同构。
    __aicore__ inline void MatmulQK(uint32_t n,
                                    uint32_t qsActBaseTileAlign16,
                                    uint32_t embed, 
                                    uint32_t qkL0abBufId, 
                                    uint32_t qkL0cBufId)
    {
        auto l0ALayout = tla::MakeLayout<ElementB, LayoutTagL0A>(n, embed);
        auto l0ATensorTla = tla::MakeTensor(l0ATensor[qkL0abBufId], l0ALayout, Arch::PositionL0A{});
        auto l0BLayout = tla::MakeLayout<ElementA, LayoutTagL0B>(embed, qsActBaseTileAlign16);
        auto l0BTensorTla = tla::MakeTensor(l0BTensor[qkL0abBufId], l0BLayout, Arch::PositionL0B{});
        auto l0CLayout = tla::MakeLayoutL0C(n, qsActBaseTileAlign16);
        auto l0CTensorTla = tla::MakeTensor(l0CTensor[qkL0cBufId], l0CLayout, Arch::PositionL0C{});
        tileMmad(l0CTensorTla, l0ATensorTla, l0BTensorTla, n, qsActBaseTileAlign16, embed, true);
    }

    // [Step4] L0C→UB fixpipe 改走 CopyL0CToUBTla NO_SPLIT/NO_QUANT
    //   nSize=m(Q行, pad Q_M_PAD=128，满足 nSize*sizeof(half) 32B 倍数约束) /
    //   mSize=n(K子块行, 调用处已偶数对齐) / srcStride=CeilAlign(n,16) /
    //   dstStride=256 / quantPre=F322F16 / dualDstCtl=0
    //   subBlockId：选目标 AIV（与 kernel loop%2 / Vec GetSubBlockIdx 对齐）
    //   UB 由调用方构好 TLA（非标 RowMajor：shape=(nEven,Q_M_PAD)、row-stride=256），直传不再 .data() 重包。
    template <class TensorUB>
    __aicore__ inline void FixpipeMm1(uint32_t qsActBaseTileAlign16, uint32_t n, 
                                      TensorUB &ubSTensorTla, bool subBlockId, uint32_t qkL0cBufId)
    {
        auto l0CTensorTla = tla::MakeTensor(
            l0CTensor[qkL0cBufId], tla::MakeLayoutL0C(n, qsActBaseTileAlign16), Arch::PositionL0C{});

        using CopyL0CToUB = Tile::CopyL0CToUBTla<
            ArchTag, decltype(l0CTensorTla), TensorUB,
            Tile::CopyL0CToUBMode::NO_SPLIT,
            Tile::ScaleGranularity::NO_QUANT, false>;
        CopyL0CToUB copyL0CToUB;
        // dualDstCtl=0：整块写到单个 AIV；subBlockId 选 AIV0/AIV1
        copyL0CToUB(ubSTensorTla, l0CTensorTla, subBlockId);
    }

    // Step1：入参对齐 PV —— 已带 GM 偏移的 Q/K/Scale + sparse 元数据 + Task/Tile 形状。
    // Step2：isFirstKvsTile 为 true 时才搬 Q/QScale；后续 tile 复用 L1 中的 Q。
    // Step3：K 按 gSparseBlockIdx 稀疏 gather；KScale 按 Host 沿 D gather（与 SparseVScale 不同）。
    //        n 侧 s2Align64 仅用于 K 数据 NZ；LoadKToL0A 的 MX 轴向见 Step3 改造。
    // UB S：kernel 传入当前 ping-pong 槽的 TLA 视图（行 stride=256）；nSub 再 GetTile。
    // subBlockIdx：Fixpipe 目标 AIV（= loop%2，与 Vec 消费核一致）。
    template <class TensorQ, class TensorK, class TensorUB, class TaskInfoT, class TileInfoT>
    __aicore__ inline
    void operator()(TensorQ &gQTensorTla,
                    TensorK &gKTensorTla, 
                    AscendC::GlobalTensor<uint8_t> gQDequantScale,   // 已带 Q-scale task 偏移（Host 沿 D）
                    AscendC::GlobalTensor<uint8_t> gKDequantScale,   // 已带 K-scale batch+head（Host 沿 D）
                    AscendC::GlobalTensor<int32_t> gSparseBlockIdx,  // 本 task 行有效 KV 列下标（Step3 gather 用）
                    TensorUB &ubSTensorTla,
                    TaskInfoT &curTaskInfo, 
                    TileInfoT &curTileInfo, 
                    uint32_t embed,              // head dim D（→ mmad k）
                    uint32_t kvSBaseTile, uint32_t blockShapeY,  // tiling：kvBaseTile_ / Y 块高
                    uint32_t subBlockIdx,        // Fixpipe → AIV0/AIV1
                    uint32_t kvHeadMul,          // [BSND stride 修复] K/K-scale 头内 token stride 因子(BNSD=1)
                    uint32_t qHeadMul)          // [BSND stride 修复] Q 头内 token stride 因子(BNSD=1)
    {
        // AscendC::printf(
        //     "\n MmadAtlasA5Bsa[QK0725]Mxfp4 Step3 | m(Q)=%u n(K)=%u k(D)=%u gatheredIdx=%u "
        //     "firstQ=%u kvSeq=%lld yAval=%u yRsvd=%u\n",
        //     (unsigned)rowNumQ, (unsigned)curBaseTileSize, (unsigned)embed,
        //     (unsigned)gatheredKvSTileIdx, (unsigned)(isFirstKvsTile ? 1 : 0),
        //     (long long)kvSeqlen, (unsigned)yBlockNumAval, (unsigned)yBlockNumRsvd);

        // K·Qᵀ：A=K[n,D], B=Q[m,D], C[n,m]；m=Q 行，n=K 行(整 tile)，k=D
        uint32_t qsActBaseTile = curTaskInfo.qsActBaseTile;
        int64_t kvSeqlen = curTaskInfo.kvSeqlen;
        uint32_t yBlockNumAval = curTaskInfo.yBlockNumAval;
        uint32_t yBlockNumRsvd = curTaskInfo.yBlockNumRsvd;
        uint32_t qsActBaseTileAlign16 = curTaskInfo.qsActBaseTileAlign16;
        uint32_t qsActBaseTileAlign128 = curTaskInfo.qsActBaseTileAlign128;

        uint32_t kvsActBaseTile = curTileInfo.kvsActBaseTile;
        uint32_t gatheredKvSTileIdx = curTileInfo.pvGatheredKvSTileIdx;
        bool isFirstKvsTile = curTileInfo.isFirstKvsTile;
        bool isLastKvsTile = curTileInfo.isLastKvsTile;
        uint32_t kvsActBaseTileAlign16 = curTileInfo.kvsActBaseTileAlign16;

        // Q-scale / K-scale：均按 D 分组；scaleK = Ceil(D/64)*2（= Host 每 token 的 D//64*2）
        uint32_t scaleK = CeilDivision(embed, 64) * 2;
        uint32_t l1KEventId = MXFP4::KV_EVENT0 + kBufId;
        uint32_t l1QEventId = Q_EVENT0 + qBufId;

        // Step5：Q/QScale 仅在 task 首 KV tile 稠密搬 GM→L1（TileCopyTla + Dn2Nz；后续 tile 复用）
        if (isFirstKvsTile) {
            // AscendC::printf("[Q-LOAD] gatheredIdx=%u m=%u k=%u (task first tile)\n",
            //     (unsigned)gatheredKvSTileIdx, (unsigned)m, (unsigned)k);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1QEventId); 
            auto l1QLayout = tla::MakeLayout<ElementA, layout::zN>(qsActBaseTileAlign16, embed);
            auto l1QTensorTla = tla::MakeTensor(l1BTensor[qBufId], l1QLayout, Arch::PositionL1{});
            CopyQGmToL1(qsActBaseTile, embed, gQTensorTla, l1QTensorTla);                       // k=embed（fp4 元素），非 packedK
            CopyQScaleGmToL1(qsActBaseTile, scaleK, gQDequantScale, qBufId, qHeadMul);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1QEventId);  // 通知 MTE1: Q 就绪
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1QEventId);
            // [STEP0 golden] L1 Q 数据 / Q-scale（仅首 tile；前 256B 采样）
            // AscendC::DumpTensor(l1ATensor[0].template ReinterpretCast<uint8_t>(), 1101, 128);
            // AscendC::DumpTensor(l1ATensor[0].template ReinterpretCast<uint8_t>()[128], 1102, 128);
            // AscendC::DumpTensor(l1AScaleTensor[0], 1111, 128);
            // // Q-scale 单份 0.5K；第二段从 offset 128 起再采 128B（仍在缓冲内）
            // AscendC::DumpTensor(l1AScaleTensor[0][128], 1112, 128);
        }
        // Step3：每 tile 按 gSparseBlockIdx 稀疏 gather 整个 K / KScale（≤512 行）到 L1（替代稠密 CopyK*）
        // KScale：Host 沿 D [S,D//64,2]；与 SparseK 同行映射（Step2）
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KEventId);
        auto l1KLayout = tla::MakeLayout<ElementA, layout::zN>(kvsActBaseTile, embed);
        auto l1KTensorTla = tla::MakeTensor(l1ATensor[kBufId], l1KLayout, Arch::PositionL1{});
        SparseKBaseTileL1FullLoad(
            gKTensorTla, l1KTensorTla, gSparseBlockIdx, gatheredKvSTileIdx, kvSeqlen, kvSBaseTile, blockShapeY,
            yBlockNumAval, yBlockNumRsvd, kvsActBaseTile, embed);
        SparseKScaleBaseTileL1FullLoad(
            gKDequantScale, gSparseBlockIdx, gatheredKvSTileIdx, kvSeqlen, kvSBaseTile, blockShapeY,
            yBlockNumAval, yBlockNumRsvd, kvsActBaseTile, embed, kBufId, kvHeadMul);
        // MTE2(gather) 完成后再 MTE1(L1→L0)
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1KEventId);

        // ===== Step4.5：n 方向按 L0_TILE_N 子切分 =====
        // kvBaseTile=256 时 nLoopNum=2，每个 L0 tile 为 [nCur≤128, m]，不溢出 QK_L0A/QK_L0C。
        // 单 buffer：每个 nSub 末尾 PipeBarrier<PIPE_ALL> 全序列化，避免 L0A/L0C 复用冒险（DB 属 Step6）。
        uint32_t nLoopNum = CeilDivision(kvsActBaseTile, L0_TILE_N);
        for (uint32_t nSub = 0; nSub < nLoopNum; ++nSub) {
            uint32_t l0ABEventId = QK_L0AB_EVENT0 + qkL0abBufId;
            uint32_t l0CEventId = QK_L0C_EVENT0 + qkL0cBufId;
            uint32_t nSubRowStart = nSub * L0_TILE_N;
            uint32_t nCur = MinU32(L0_TILE_N, kvsActBaseTile - nSubRowStart);
            // AscendC::printf("[MM1] nSub=%u nCur=%u rowStart=%u m=%u k=%u\n",
            //     (unsigned)nSub, (unsigned)nCur, (unsigned)nSubRowStart, (unsigned)m, (unsigned)k);
            // Q 常驻 L0B：循环外只搬一次（Step2 成果），n 子循环内复用（Q 不随 nSub 变）
            // [QFA 对齐] m 方向按 Q_M_PAD=128 加载（mStep=8/xStep=8，与 QFA LoadQToL0 一致）
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);

            {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABEventId); 
                LoadQToL0B(qsActBaseTileAlign16, embed, qsActBaseTileAlign16, qsActBaseTileAlign16, scaleK, qBufId, qkL0abBufId);         
                LoadKToL0A(nSubRowStart, nCur, embed, kvsActBaseTileAlign16, scaleK, kBufId, qkL0abBufId);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ABEventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ABEventId);
                // [QFA 对齐] C=[nCur, Q_M_PAD]：Mmad n 恒 128（QFA mmadParams.n=actMSizeAlign128 语义）
                // 等该 L0C 槽空闲（Alloc 预置 / 上次 Fixpipe 归还）→ Mmad 写入
                MatmulQK(nCur, qsActBaseTileAlign16, embed, qkL0cBufId, qkL0abBufId); 
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABEventId); 
                qkL0abBufId = (qkL0abBufId + 1) % MXFP4::L0A_QK_BUF_CNT;
            }
            
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            // AscendC::DumpTensor(l0CTensor[0], 6660 + nSub, 128);
            // AscendC::PipeBarrier<PIPE_ALL>();
            // [QFA 对齐] Fixpipe nSize=Q_M_PAD(满足 32B 倍数约束)、mSize=nCur 偶数对齐
            // （QFA (actS2Size+1)>>1<<1 语义）；写到 ubS 第 nSubRowStart 行起，行 stride=256
            uint32_t nCurEven = (nCur + 1) >> 1 << 1;
            auto ubSSubTla = GetTile(
                ubSTensorTla, tla::MakeCoord(nSubRowStart, 0), tla::MakeShape(nCurEven, qsActBaseTileAlign16));
            FixpipeMm1(qsActBaseTileAlign16, nCur, ubSSubTla, subBlockIdx != 0, qkL0cBufId);
            // Fixpipe 完成，归还 L0C 槽并翻转 ping-pong
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
            qkL0cBufId = (qkL0cBufId + 1) % MXFP4::L0C_QK_BUF_CNT;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1KEventId);
        kBufId = (kBufId + 1) % MXFP4::L1_KV_BUF_CNT;

        if (unlikely(isLastKvsTile)) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1QEventId); 
            qBufId = (qBufId + 1) % MXFP4::L1_Q_BUF_CNT;
        }
    }
    

protected:
    // L1：Q 用 2 份，k 用 4 份
    AscendC::LocalTensor<ElementB> l1ATensor[MXFP4::L1_KV_BUF_CNT];
    AscendC::LocalTensor<ElementA> l1BTensor[MXFP4::L1_Q_BUF_CNT];

    // L0：A/B 各 2 份 ，C 2 份
    AscendC::LocalTensor<ElementB> l0ATensor[MXFP4::L0A_QK_BUF_CNT];
    AscendC::LocalTensor<ElementA> l0BTensor[MXFP4::L0A_QK_BUF_CNT];
    AscendC::LocalTensor<float> l0CTensor[MXFP4::L0C_QK_BUF_CNT];

    // MXFP4 专用：E8M0 scale 的 L1 buffer（Q-scale 2 份，K-scale 4 份）+ MX 版 L1->L0 拷贝器
    AscendC::LocalTensor<uint8_t> l1AScaleTensor[MXFP4::L1_KV_BUF_CNT];
    AscendC::LocalTensor<uint8_t> l1BScaleTensor[MXFP4::L1_Q_BUF_CNT];

    // [Step2] QK 非转置 mx L1→L0 functor（勿与 PV 的 CopyL1ToL0*MxA5 混淆）
    Tile::CopyL1ToL0AMxQKA5 copyL1ToL0AMxQk;  // K: L1→L0A，ifTranspose=false
    Tile::CopyL1ToL0BMxQKA5 copyL1ToL0BMxQk;  // Q: L1→L0B，ifTranspose=false
    // [Step3] fp4 Mmad（scale 已在 L1→L0 烘焙）
    TileMmad tileMmad;
    // [Step5/6] Q/K-scale 沿 D Dn2Nz（Q 稠密 + K 稀疏段共用）
    Tile::CopyGmToL1MxScaleDn2NzA5 copyGmToL1MxScaleDn2Nz;
    // Double-buffer 槽位索引（跨 operator() 持久）
    uint32_t qBufId = 0;        // L1 Q 槽 [0, L1_Q_BUFCNT) —— Step4 轮转
    uint32_t& kBufId;          // [QK/PV 共享] KV 槽游标——引用 kernel 的 kvBufId，与 PV l1ABufId 同址
    uint32_t qkL0abBufId = 0;   // L0A/B 槽 [0, QK_L0AB_BUFCNT) —— Step2 轮转
    uint32_t qkL0cBufId = 0;    // L0C 槽 [0, QK_L0C_BUFCNT) —— Step1 已轮转
};

}  // namespace NpuArch::Gemm::Block

#endif  // GEMM_BLOCK_QK_ARCH35_MXFP4_HPP
