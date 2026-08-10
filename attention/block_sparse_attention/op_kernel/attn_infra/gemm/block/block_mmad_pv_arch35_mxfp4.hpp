/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_BLOCK_PV_ARCH35_MXFP4_HPP
#define GEMM_BLOCK_PV_ARCH35_MXFP4_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/arch/bsa_cross_core_sync.hpp"
#include "../../../attn_infra/bsa_coord.hpp"
#include "../../../attn_infra/gemm/bsa_gemm_dispatch_policy.hpp"
#include "../../../attn_infra/gemm/bsa_helper.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_gemm_tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_tile_mmad.hpp"
// [改造-新增] mx 专用 L1->L0 搬运 functor（带 E8M0 scale 的 5 参 mx LoadData）
#include "../../../attn_infra/gemm/tile_common/copy_l1_to_l0_mx_a5.hpp"
#include "../../../tla/layout.hpp"
#include "../../../tla/tensor.hpp"
#include "block_mmad_arch35_utils.hpp"

namespace NpuArch::Gemm::Block {
template <
    bool transposedMm1_,
    class L1TileShape_,
    class L0TileShape_,
    class ElementA_,
    class ElementB_,
    class ElementC_,
    class ElementBias_,
    class TileCopy_,
    class TileMmad_>
struct BlockMmadTla<
    MmadAtlasA5BsaPVMxfp4<transposedMm1_>,
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
    using DispatchPolicy = MmadAtlasA5BsaPVMxfp4<transposedMm1_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;

    using TileMmad = TileMmad_;

    // [改造-新增] Copy policy 别名（原文件缺失，仅持 TileCopy/TileMmad）。
    //   GM->L1 数据搬运走 TileCopy 模板别名；mx L1->L0 走下方独立 functor。
    //   [TODO] CopyL0CToDst(L0C->UB fixpipe)：PackedTileCopyTla 的 CopyL0CToDst 实为 L0C->GM（语义不符），
    //          UB 版在 PackedTileCopyTlaToUB；待 TileCopy 切到 ToUB 后再别名化，FixpipeMm2 暂保留手写。
    template <class TensorA> using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<TensorA>;
    template <class TensorB> using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<TensorB>;

    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;   // V: nZ（由 ColumnMajor GM 经 L1ATypeSelector 推出）
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;   // P: zN（由 RowMajor GM 经 L1BTypeSelector 推出）
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;   // Vᵀ: zN（A5 arch 固定）
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;   // Pᵀ: nZ（PackedTileCopyTla 固定）

    // [改造-新增] mxfp4 标志，供 if constexpr 分支（仿 ABf16 FULL_QUANT_FP8）。
    static constexpr bool IS_MXFP4 = AscendC::IsSameType<ElementA, fp4x2_e2m1_t>::value;
    static_assert(IS_MXFP4, "BlockMmadTla<MmadAtlasA5BsaPVMxfp4> requires ElementA = fp4x2_e2m1_t");

    static constexpr uint32_t L0_STAGES = DispatchPolicy::L0_STAGES;
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape_{});



    // mode4 核间同步：AIC 侧对两个 AIV 各用一个 flagId，第二个 = 第一个 + 16
    static constexpr uint32_t V0_V1_FLAG_ID_OFFSET = 16;

    // ====================== Event ID 命名常量（与 QFA 完全一致）======================
    // QFA 的 ID 分配策略：QK 用 2-3，PV 用 4-7。同 ID 不同 HardEvent 类型不冲突。
    // 2+2 方案：PV 用 2 槽 ping-pong（KV_EVENT0/1），QK 后续用 ID 2-3 各自独立。
    //
    // V L1 双路（MTE1_MTE2 / MTE2_MTE1：GM→L1 gather ↔ L1→L0 LoadV）
    static constexpr uint32_t KV_EVENT0 = 4;
    static constexpr uint32_t KV_EVENT1 = 5;
    // PV L0A/B 双缓冲（M_MTE1 / MTE1_M：LoadV/P ↔ Mmad）
    //   与 KV_EVENT 复用同一 ID——不同 HardEvent 类型，硬件 flag 独立存储，不冲突
    static constexpr uint32_t PV_L0AB_EVENT0 = KV_EVENT0;  // = 4
    static constexpr uint32_t PV_L0AB_EVENT1 = KV_EVENT1;  // = 5
    // PV L0C 单缓冲（M_FIX / FIX_M：Mmad ↔ Fixpipe）
    //   同样与 KV_EVENT0 复用——M_FIX / FIX_M 与 MTE1_MTE2 / M_MTE1 是不同 HardEvent 类型
    static constexpr uint32_t PV_L0C_EVENT0 = KV_EVENT0;   // = 4

    // L1 各区单份字节数（fp4 两元素占 1 字节；scale 为 E8M0，每 32 元素 1 字节）

    
    static constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag> &resource, uint32_t& kvBufId) : l1ABufId(kvBufId) {
        // L1 区域顺序（与 QFA 一致）：P 数据 | P-scale | Q 数据 | Q-scale | KV(=V) 数据 | KV(=V)-scale
        uint32_t addrL1Start = 0;
        // P 数据：l1BTensor(B=P)，MXFP4::L1_P_BUF_SIZE * 20 = 320K
        for (uint32_t i = 0; i < MXFP4::L1_P_BUF_CNT; i++) {
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(MXFP4::L1_P_BUF_OFFSET + MXFP4::L1_P_BUF_SIZE * i);
        }

        for (uint32_t i = 0; i < MXFP4::L1_P_SCALE_BUF_CNT; i++) {
            l1BScaleTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(MXFP4::L1_P_SCALE_BUF_OFFSET + MXFP4::L1_P_SCALE_BUF_SIZE * i);
        }
        
        for (uint32_t i = 0; i < MXFP4::L1_KV_BUF_CNT; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(MXFP4::L1_KV_BUF_OFFSET + MXFP4::L1_KV_BUF_SIZE * i);
        }

        for (uint32_t i = 0; i < MXFP4::L1_KV_DESCALE_BUF_CNT; i++) {
            l1AScaleTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(MXFP4::L1_KV_DESCALE_BUF_OFFSET + MXFP4::L1_KV_DESCALE_BUF_SIZE * i);
        }

        for (uint32_t i = 0; i < MXFP4::L0A_PV_BUF_CNT; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(MXFP4::L0A_PV_BUF_OFFSET + MXFP4::L0A_PV_BUF_SIZE * i);
        }

        for (uint32_t i = 0; i < MXFP4::L0B_PV_BUF_CNT; i++) {
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(MXFP4::L0B_PV_BUF_OFFSET + MXFP4::L0B_PV_BUF_SIZE * i);
        }

        l0CTensor[0] = resource.l0CBuf.template GetBufferByByte<float>(MXFP4::L0C_PV_BUF_OFFSET);

        // ====================== Event 预置（对应 QFA AllocEventID）======================
        AllocEventID();

        // 初始化L0A中v pad的16行（循环 seed 两个 L0A 槽，配合 ping-pong）
        InitL0BufferForReduceSum();
    }
    __aicore__ inline ~BlockMmadTla() {}

    // ============================================================================
    // AllocEventID：预置所有 event flag（让首轮不用等），对应 QFA AllocEventID
    // ============================================================================
    __aicore__ inline void AllocEventID()
    {
        // PV L0A/B 双缓冲（M_MTE1：Mmad→MTE1，两个槽都预置）
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(PV_L0AB_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(PV_L0AB_EVENT1);
        // PV L0C 单缓冲（FIX_M：Fixpipe→M，预置可写）
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(PV_L0C_EVENT0);
        // [QK/PV 共享] KV 池 MTE1_MTE2 4 路由 QK 的 AllocEventID 统一预置（本处不重复）

    }

    // ============================================================================
    // FreeEventID：等所有 event 排空（释放），对应 QFA FreeEventID
    // ============================================================================
    __aicore__ inline void FreeEventID()
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(PV_L0AB_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(PV_L0AB_EVENT1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(PV_L0C_EVENT0);
        // [QK/PV 共享] KV 池 MTE1_MTE2 4 路由 QK 的 FreeEventID 统一排空（本处不重复）

    }

    // [改造] V 的 L1->L0A 转置搬运改走 mx functor（原裸 LoadVToL0 的 5 参 mx LoadData 下沉到 policy）。
    //   nZ L1(V) + V-scale -> zN L0A(Vᵀ)，ifTranspose=true 由 functor 内部设定。
    __aicore__ inline void LoadVToL0(uint32_t l1ABufId, uint32_t l0ABufId,
                                     uint32_t embedReal, uint32_t s2Align64)
    {
        // [改造] 删除 ReinterpretCast<fp4x2_e2m1_t / fp8_e8m0_t> 散点：
        //   l1ATensor/l0ATensor 已是 ElementA=fp4x2_e2m1_t；scale 由 uint8 reinterpret 为 fp8_e8m0_t 传入 functor。
        copyL1ToL0AMx(
            l0ATensor[l0ABufId],
            l1ATensor[l1ABufId],
            l1AScaleTensor[l1ABufId].template ReinterpretCast<AscendC::fp8_e8m0_t>(),
            embedReal, s2Align64);
    }

    // [改造] P 的 L1->L0B 转置搬运改走 mx functor（原裸 LoadPToL0 下沉到 policy）。
    //   zN L1(P) + P-scale -> nZ L0B(Pᵀ)，ifTranspose=true 由 functor 内部设定。
        __aicore__ inline void LoadPToL0(uint32_t l1BBufId, uint32_t l0BBufId, uint32_t s2Align64,
                                     uint32_t embedReal, uint32_t s1Align64)
    {
        // s2Base 取整 tile S2Base(=256)，用于推导 P-scale srcStride（原硬编码 5 的来源）。
        // [TODO] s2Base 应由调用方/TileShape 传入，此处暂以 256 占位（与 MXFP4::L1_P_BUF_SIZE 排布一致）。
        constexpr uint32_t s2Base = 256;
        copyL1ToL0BMx(
                    l0BTensor[l0BBufId],
                    l1BTensor[l1BBufId],
                    l1BScaleTensor[l1BBufId].template ReinterpretCast<AscendC::fp8_e8m0_t>(),
                    s2Align64, embedReal, s2Base, 5, s1Align64);
    }

    // [改造] Mmad 改走 tileMmad functor（原裸 MatmulPV 的 AscendC::Mmad 下沉到 TileMmadTla）。
    //   C=Vᵀ@P=Oᵀ，m=D+16（含 rowsum seed 行）、n=M、k=S2(对齐64)。
    //   tileMmad 只读 .data()（不消费 layout stride），故 fp4 粒度不影响；这里包 tla 张量传入。
    //   [#1] initC 由 isTileGoupFirstTile 控制：TileGroup 首 tile 重置 L0C、tile 内累加（cmatrixInitVal 语义）。
    __aicore__ inline void MatmulPV(uint32_t l0ABufId, uint32_t l0BBufId,
                                    uint32_t m, uint32_t n, uint32_t k, bool initC)
    {
        auto l0ALayout = tla::MakeLayout<ElementA, LayoutTagL0A>(m, k);
        auto l0ATensorTla = tla::MakeTensor(l0ATensor[l0ABufId], l0ALayout, Arch::PositionL0A{});
        auto l0BLayout = tla::MakeLayout<ElementB, LayoutTagL0B>(k, n);
        auto l0BTensorTla = tla::MakeTensor(l0BTensor[l0BBufId], l0BLayout, Arch::PositionL0B{});
        auto l0CLayout = tla::MakeLayoutL0C(m, n);
        auto l0CTensorTla = tla::MakeTensor(l0CTensor[0], l0CLayout, Arch::PositionL0C{});
        tileMmad(l0CTensorTla, l0ATensorTla, l0BTensorTla, m, n, k, initC);
    }


    // [Clean-1] CopyInA1/CopyInScaleA1 已删除——operator() 走 tla gather（SparseVBaseTileL1FullLoad +
    //   SparseVScaleBaseTileL1FullLoad），这两个 raw Nd2Nz 方法不再使用。

    // [step2-tla化] V 数据 gather 改 ABf16 风格：GetTile(gV段) + GetTile(l1A段) + copyGmToL1A。
    //   gV=ColumnMajor(D,S2)、l1A=nZ(D,s2Align64)，两者 dim0=embed、dim1=seq，
    //   coord/shape 同序 (0, seq_off)/(embed, curDealtLen)。
    //   Nd2Nz 参数（nValue=curDealtLen / dValue=D/2 / srcDValue=D/2 / dstNzC0Stride=s2Align64 /
    //   dstNzNStride=1 / dstOffset=dealtAccum*64）由 copyGmToL1A 的 ColumnMajor→nZ 特化 +
    //   A2 的 bit-granular ELE_PER_C0(fp4→64) + step1 的 /2 分支自动给出，与原 raw gather 1:1 对齐。
    //   [注] 直接用 Tile::TileCopyTla<ArchTag,TensorV,TensorL1A>（按 gather 实参类型派发），
    //        不用 TileCopy::CopyGmToL1A 别名（后者用 Packed 的 TensorL1A/LayoutL1A，
    //        若主模板默认 LayoutTagA=RowMajor→zN 会派发 ColumnMajor→zN 无特化）。
    template <class TensorV, class TensorL1A>
    __aicore__ inline
    void SparseVBaseTileL1FullLoad(
        TensorV &gVTensor, TensorL1A &l1ATensorTla,
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
        uint32_t gatheredKvSTileIdx, int64_t kvSeqlen,
        uint32_t kvSBaseTile, uint32_t blockShapeY,
        uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
        uint32_t curBaseTileSize, uint32_t embed, uint32_t s2Align64)
    {
        using CopyGmToL1A = Tile::TileCopyTla<ArchTag, TensorV, TensorL1A>;
        CopyGmToL1A copyGmToL1A;
        uint32_t baseTileStartOffset = gatheredKvSTileIdx * kvSBaseTile;
        uint32_t baseTileEndOffset   = baseTileStartOffset + curBaseTileSize;
        uint32_t gatheredStartOffset = baseTileStartOffset;
        uint32_t gatheredYBlockIdx   = gatheredStartOffset / blockShapeY;
        uint32_t yBlockInnerStart    = gatheredStartOffset % blockShapeY;
        uint32_t oriYBlockIdx   = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
        uint32_t oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStart;
        uint32_t dealtLenAccum  = 0;

        while (dealtLenAccum < curBaseTileSize && gatheredYBlockIdx < yBlockNumRsvd &&
               oriYBlockIdx < yBlockNumAval && oriStartOffset < kvSeqlen) {
            uint32_t curYBlockSize = (oriYBlockIdx == yBlockNumAval - 1)
                                     ? static_cast<uint32_t>(kvSeqlen - oriYBlockIdx * blockShapeY)
                                     : blockShapeY;
            uint32_t gatheredEndOffset = min(gatheredYBlockIdx * blockShapeY + curYBlockSize,
                                             baseTileEndOffset);
            uint32_t curDealtLen = gatheredEndOffset - gatheredStartOffset;
            if (curDealtLen == 0) { break; }

            // [step2] 段切片：gV/l1A 维序均 (embed, seq)，coord(0, seq_off)/shape(embed, curDealtLen)
            auto gVTensorTile = GetTile(gVTensor,
                tla::MakeCoord(0, oriStartOffset), tla::MakeShape(embed, curDealtLen));
            auto l1ATensorTile = GetTile(l1ATensorTla,
                tla::MakeCoord(0, dealtLenAccum), tla::MakeShape(embed, curDealtLen));
            // [验证] catlass offset vs raw：srcOff 应=oriStartOffset*embed，dstOff 应=dealtAccum*64
            // AscendC::printf("[V-CHK] oriStart=%u dealtAccum=%u curDealtLen=%u srcOff=%u(want %u) dstOff=%u(want %u)\n",
            //     (unsigned)oriStartOffset, (unsigned)dealtLenAccum, (unsigned)curDealtLen,
            //     (unsigned)static_cast<uint32_t>(gVTensorTile.layout()(gVTensorTile.coord())),
            //     (unsigned)(oriStartOffset * embed),
            //     (unsigned)static_cast<uint32_t>(l1ATensorTile.layout()(l1ATensorTile.coord())),
            //     (unsigned)(dealtLenAccum * 64));
            copyGmToL1A(l1ATensorTile, gVTensorTile);

            dealtLenAccum += curDealtLen;
            gatheredStartOffset += curDealtLen;
            gatheredYBlockIdx = gatheredStartOffset / blockShapeY;
            yBlockInnerStart   = gatheredStartOffset % blockShapeY;
            if (dealtLenAccum < curBaseTileSize) {
                oriYBlockIdx   = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
                oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStart;
            }
        }
    }

    // [step2-tla化] V-scale gather 改 ABf16 风格：GetTile(gVScale段) + GetTile(l1AScale段) + mx functor。
    //   gVScale=MxScaleA(ColumnMajor,trans)、l1AScale=zZ，均 (rows=embed, cols=s2Align64/32) 维序 (embed, seq-scale)。
    //   coord(0, seqOff/32)/shape(embed, curDealtLen/32)；nValue=CeilDiv(curDealtLen/32,2)=curDealtLen/64=scaleRows。
    //   Nd2Nz 参数 + B16-view reinterpret + offset 由 step1 移植的 MxScaleA→zZ 特化自动给出，与 raw 1:1 对齐：
    //     nValue=scaleRows / dValue=D / srcDValue=D / dstNzC0Stride=s2Align64/64 / dstNzNStride=1
    //     srcOffset=(oriStartOffset/32)*embed fp8 → half srcB16Off=(oriStartOffset/64)*embed
    //     dstOffset=dealtAccum/2 fp8 → half dstB16Off=(dealtAccum/64)*16
    template <class TensorVScale, class TensorL1AScale>
    __aicore__ inline
    void SparseVScaleBaseTileL1FullLoad(
        TensorVScale &gVScaleTensor, TensorL1AScale &l1AScaleTensorTla,
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
        uint32_t gatheredKvSTileIdx, int64_t kvSeqlen,
        uint32_t kvSBaseTile, uint32_t blockShapeY,
        uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
        uint32_t curBaseTileSize, uint32_t embed, uint32_t s2Align64)
    {
        using CopyGmToL1AScale = Tile::TileCopyTla<ArchTag, TensorVScale, TensorL1AScale>;
        CopyGmToL1AScale copyGmToL1AScale;
        uint32_t baseTileStartOffset = gatheredKvSTileIdx * kvSBaseTile;
        uint32_t baseTileEndOffset   = baseTileStartOffset + curBaseTileSize;
        uint32_t gatheredStartOffset = baseTileStartOffset;
        uint32_t gatheredYBlockIdx   = gatheredStartOffset / blockShapeY;
        uint32_t yBlockInnerStart    = gatheredStartOffset % blockShapeY;
        uint32_t oriYBlockIdx   = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
        uint32_t oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStart;
        uint32_t dealtLenAccum  = 0;

        while (dealtLenAccum < curBaseTileSize && gatheredYBlockIdx < yBlockNumRsvd &&
               oriYBlockIdx < yBlockNumAval && oriStartOffset < kvSeqlen) {
            uint32_t curYBlockSize = (oriYBlockIdx == yBlockNumAval - 1)
                                     ? static_cast<uint32_t>(kvSeqlen - oriYBlockIdx * blockShapeY)
                                     : blockShapeY;
            uint32_t gatheredEndOffset = min(gatheredYBlockIdx * blockShapeY + curYBlockSize,
                                             baseTileEndOffset);
            uint32_t curDealtLen = gatheredEndOffset - gatheredStartOffset;
            if (curDealtLen == 0) { break; }

            // [step2] 段切片：gVScale/l1AScale 维序 (embed, seq-scale)，col=32 fp4（1 quant group=MX_SCALE_GROUP_NUM）。
            //   coord dim1 = seqOff/32（flat col），crd2offset 内部 %2//2 拆 inner/outer → nValue=cols/2=seqOff/64。
            auto gVScaleTile = GetTile(gVScaleTensor,
                tla::MakeCoord(0, oriStartOffset / MX_SCALE_GROUP_NUM),
                tla::MakeShape(embed, CeilDiv(curDealtLen, MX_SCALE_GROUP_NUM)));
            auto l1AScaleTile = GetTile(l1AScaleTensorTla,
                tla::MakeCoord(0, dealtLenAccum / MX_SCALE_GROUP_NUM),
                tla::MakeShape(embed, CeilDiv(curDealtLen, MX_SCALE_GROUP_NUM)));
            // [验证] catlass offset vs raw：raw 用 B16/half 视图，catlass 用 fp8——
            //   srcOff(fp8) 应=2*srcB16Off=(oriStart/64)*embed*2，dstOff(fp8) 应=2*dstB16Off=(dealtAccum/64)*16*2
            uint32_t rawSrcB16Off = (oriStartOffset / 64) * embed;
            uint32_t rawDstB16Off = (dealtLenAccum / 64) * 16;
            // AscendC::printf("[VS-CHK] oriStart=%u dealtAccum=%u curDealtLen=%u srcOff=%u(want %u=2*srcB16) dstOff=%u(want %u=2*dstB16)\n",
            //     (unsigned)oriStartOffset, (unsigned)dealtLenAccum, (unsigned)curDealtLen,
            //     (unsigned)static_cast<uint32_t>(gVScaleTile.layout()(gVScaleTile.coord())),
            //     (unsigned)(rawSrcB16Off * 2),
            //     (unsigned)static_cast<uint32_t>(l1AScaleTile.layout()(l1AScaleTile.coord())),
            //     (unsigned)(rawDstB16Off * 2));
            copyGmToL1AScale(l1AScaleTile, gVScaleTile);

            dealtLenAccum += curDealtLen;
            gatheredStartOffset += curDealtLen;
            gatheredYBlockIdx = gatheredStartOffset / blockShapeY;
            yBlockInnerStart   = gatheredStartOffset % blockShapeY;
            if (dealtLenAccum < curBaseTileSize) {
                oriYBlockIdx   = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
                oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStart;
            }
        }
    }

    // [catlass化] L0C→UB fixpipe 走 CopyL0CToUBTla SPLIT_N/NO_QUANT（dualDstCtl=2），删手写 FixpipeParams。
    //   catlass 从 L0C+UB 张量布局算全部参数，与 raw 1:1：
    //     nSize=RoundUp(M,32)=s1Align128 / mSize=D+1 / srcStride=RoundUp(D+16,16)=D+16 /
    //     dstStride=64 / dualDstCtl=2 / quantPre=NoQuant。
    //   L0C tla 本地构：rows=D+16（srcStride=D+16，含 rowsum seed +16 行，与 MatmulPV 一致）。
    //   UB tla 由调用处（kernel）按 Oᵀ 实际存储布局构好直传（rows=D+1、cols=s1Align128、stride=64），
    //     这里不再 .data() 重包——直收 tla。
    //   [注] 直接实例化 Tile::CopyL0CToUBTla（block 的 TileCopy::CopyL0CToDst 实为 L0C→GM，非 UB）。
    template <class TensorUB>
    __aicore__ inline void FixpipeMm2(uint32_t s1Align64, TensorUB &ubOTmpTensor)
    {
        // L0C tla：rows=D+16（与 MatmulPV 的 mMmad=D+16 一致）→ srcStride=D+16
        auto l0CTensorTla = tla::MakeTensor(l0CTensor[0],
            tla::MakeLayoutL0C(MXFP4::ROW_SUM_NUM + MXFP4::ROW_SUM_PAD_NUM , s1Align64), Arch::PositionL0C{});
        // 模板参序 (ArchTag, TensorSrc=L0C, TensorDst=UB, SPLIT_N, NO_QUANT, ReluEnable=false)
        using CopyL0CToDst = Tile::CopyL0CToUBTla<ArchTag, decltype(l0CTensorTla), TensorUB,
                                                   Tile::CopyL0CToUBMode::SPLIT_N,
                                                   Tile::ScaleGranularity::NO_QUANT, false>;
        CopyL0CToDst copyL0CToDst;
        // [验证] catlass 从布局算出的 fixpipe 参数 vs raw 期望
        uint32_t fpMSize = static_cast<uint32_t>(tla::get<0>(ubOTmpTensor.shape()));
        uint32_t fpNSizeBase = static_cast<uint32_t>(tla::get<1>(ubOTmpTensor.shape()));
        uint32_t fpDstStride = static_cast<uint32_t>(tla::get<0>(ubOTmpTensor.stride()));
        uint32_t fpSrcStride = static_cast<uint32_t>(tla::get<1, 1>(l0CTensorTla.stride())) /
                               static_cast<uint32_t>(tla::get<0, 0>(l0CTensorTla.shape()));
        // AscendC::printf("[FP-CHK] mSize=%u(want %u) nSizeBase=%u(want %u) srcStride=%u(want %u) dstStride=%u(want 64) dualDstCtl=2\n",
        //     (unsigned)fpMSize, (unsigned)(embedReal + 1),
        //     (unsigned)fpNSizeBase, (unsigned)s1Align128,
        //     (unsigned)fpSrcStride, (unsigned)(embedReal + 16),
        //     (unsigned)fpDstStride);
        copyL0CToDst(ubOTmpTensor, l0CTensorTla);   // operator()(dst=UB, src=L0C)
    }

    template <class TensorUB>
    __aicore__ inline void FixpipeMm2SingleVect(uint32_t s1Align64, TensorUB &ubOTmpTensor)
    {
        // L0C tla：rows=D+16（与 MatmulPV 的 mMmad=D+16 一致）→ srcStride=D+16
        auto l0CTensorTla = tla::MakeTensor(l0CTensor[0],
            tla::MakeLayoutL0C(MXFP4::ROW_SUM_NUM + MXFP4::ROW_SUM_PAD_NUM , s1Align64), Arch::PositionL0C{});
        // 模板参序 (ArchTag, TensorSrc=L0C, TensorDst=UB, SPLIT_N, NO_QUANT, ReluEnable=false)
        using CopyL0CToDst = Tile::CopyL0CToUBTla<ArchTag, decltype(l0CTensorTla), TensorUB,
                                                   Tile::CopyL0CToUBMode::NO_SPLIT,
                                                   Tile::ScaleGranularity::NO_QUANT, false>;
        CopyL0CToDst copyL0CToDst;
        // [验证] catlass 从布局算出的 fixpipe 参数 vs raw 期望
        uint32_t fpMSize = static_cast<uint32_t>(tla::get<0>(ubOTmpTensor.shape()));
        uint32_t fpNSizeBase = static_cast<uint32_t>(tla::get<1>(ubOTmpTensor.shape()));
        uint32_t fpDstStride = static_cast<uint32_t>(tla::get<0>(ubOTmpTensor.stride()));
        uint32_t fpSrcStride = static_cast<uint32_t>(tla::get<1, 1>(l0CTensorTla.stride())) /
                               static_cast<uint32_t>(tla::get<0, 0>(l0CTensorTla.shape()));
        // AscendC::printf("[FP-CHK] mSize=%u(want %u) nSizeBase=%u(want %u) srcStride=%u(want %u) dstStride=%u(want 64) dualDstCtl=2\n",
        //     (unsigned)fpMSize, (unsigned)(embedReal + 1),
        //     (unsigned)fpNSizeBase, (unsigned)s1Align128,
        //     (unsigned)fpSrcStride, (unsigned)(embedReal + 16),
        //     (unsigned)fpDstStride);
        copyL0CToDst(ubOTmpTensor, l0CTensorTla);   // operator()(dst=UB, src=L0C)
    }

    __aicore__ inline uint16_t CeilDivision(uint16_t numerator, uint16_t denominator)
    {
        return (numerator + denominator - 1) / denominator;
    }

    __aicore__ inline void InitL0BufferForReduceSum()
    {
        // [#10b] 等 V L1 当前 slot 空闲（上次 LoadV 释放了它）
        //   ctor 调用时：MTE1_MTE2(KV_EVENT0+l1ABufId) 已预置，Wait 直接通过
        //   re-seed 调用时：l1ABufId 已轮转到 NEW slot（operator() 先 Set+rotate 再调这里），
        //     Wait 等 NEW slot 的上次 LoadV 完成
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(KV_EVENT0 + l1ABufId);
        // [#10] seed fill 值与 QFA 一致：
        //   V 数据 = 0x6666（fp4 4.0）、V scale = 0x7d7d（e8m0 0x25→2^(125-127)=0.25）→ 有效 4.0×0.25=1.0
        //   原 0x2222(1.0)×0x7f7f(1.0)=1.0 也是 1.0，但 QFA 用 4.0×0.25（规避 fp4 0x2 编码问题）
        //   对应 QFA Fill kvL1[kvBufId]（当前 slot，非固定 slot 0）
        AscendC::InitConstValueParams<uint16_t> vFillParams(1, static_cast<uint16_t>(MXFP4::L0A_PV_BUF_SIZE / MXFP4::BLOCK_SIZE ), 0, 0x6666);
        AscendC::Fill(l1ATensor[l1ABufId].template ReinterpretCast<uint16_t>(), vFillParams);          // V 数据
        AscendC::PipeBarrier<PIPE_MTE2>();
        AscendC::InitConstValueParams<uint16_t> vscaleFillParams(1, static_cast<uint16_t>(MXFP4::V_SCALE_L0A_SIZE / MXFP4::BLOCK_SIZE), 0, 0x7d7d);
        AscendC::Fill(l1AScaleTensor[l1ABufId].template ReinterpretCast<uint16_t>(), vscaleFillParams); // V scale
        AscendC::PipeBarrier<PIPE_MTE2>();

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(KV_EVENT0 + l1ABufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(KV_EVENT0 + l1ABufId);
        // Fill→LoadData 的 MTE2→MTE1 同步，复用 L1 pipeline event（对应 QFA KV_EVENT0+kvBufId）

            // seed 装载：把 V 的全部 D+16 行(含 rowsum seed)非转置装入 L0A。
            // --- 对照 QFA InitL0BufferForReduceSum() (quant_flash_attn_block_cube_dn.h:675) ---
            //   QFA: Fill kvL1(0x6666)+kvDescale(0x7d7d) → 循环 2 次 LoadData pvL0A（两份 ping-pong L0A）。
            //   [#3] 本处改为循环 MXFP4::L0A_PV_BUF_CNT(=2) 次，seed 两个 L0A 槽，配合 ping-pong。
        AscendC::LoadData2DParamsV2 loadData2DParamsA;
        loadData2DParamsA.mStartPosition = 0;
        loadData2DParamsA.kStartPosition = 0;
        loadData2DParamsA.mStep = (128 + MXFP4::ROW_SUM_PAD_NUM) / 16;   // m = D+16
        loadData2DParamsA.kStep = 256 / 64;              // k = S2（对齐 64），与 mmad k 一致
        loadData2DParamsA.srcStride = loadData2DParamsA.mStep;
        loadData2DParamsA.dstStride = loadData2DParamsA.mStep;
        loadData2DParamsA.ifTranspose = false;             // 不转置

        AscendC::LoadData2DMxParams loadData2DMXParamsA;
        loadData2DMXParamsA.xStartPosition = 0;
        loadData2DMXParamsA.yStartPosition = 0;
        loadData2DMXParamsA.xStep = (128 + MXFP4::ROW_SUM_PAD_NUM) / 16; // x = m = D+16
        loadData2DMXParamsA.yStep = 256 / 64;            // y = k = S2（对齐 64）
        loadData2DMXParamsA.srcStride = loadData2DMXParamsA.yStep;
        loadData2DMXParamsA.dstStride = loadData2DMXParamsA.yStep;

        // [#3] 循环 seed 两个 L0A 槽（对应 QFA for(i<2)），Wait/Set M_MTE1 配合 ping-pong
        //   src 用 l1ATensor[l1ABufId]（当前 slot，与 QFA kvL1[kvBufId] 一致）
        for (uint32_t i = 0; i < MXFP4::L0A_PV_BUF_CNT; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(PV_L0AB_EVENT0 + pvL0abBufId);
            LoadData(l0ATensor[pvL0abBufId].template ReinterpretCast<fp4x2_e2m1_t>(),
                    l1ATensor[l1ABufId].template ReinterpretCast<fp4x2_e2m1_t>(),
                    l1AScaleTensor[l1ABufId].template ReinterpretCast<AscendC::fp8_e8m0_t>(),
                    loadData2DParamsA, loadData2DMXParamsA);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(PV_L0AB_EVENT0 + pvL0abBufId);
            pvL0abBufId = (pvL0abBufId + 1) % MXFP4::L0A_PV_BUF_CNT;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        // [#10b] 释放 V L1 当前 slot 给下次 gather（与开头的 Wait 配对），轮转
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(KV_EVENT0 + l1ABufId);
        l1ABufId = (l1ABufId + 1) % MXFP4::L1_KV_BUF_CNT;
    }

    __aicore__ inline void PadPS2NotAlign64(uint32_t s2Align64, uint32_t s2Align32, uint32_t embedReal, uint32_t pSlot)
    {
        // 补零量 = align64 - align32
        AscendC::InitConstValueParams<uint16_t> PL1InitParams(1, static_cast<uint16_t>(s2Align64 - s2Align32), 0, 0x0000);
        AscendC::Fill(l1BTensor[pSlot][s2Align32 * 64].template ReinterpretCast<uint16_t>(), PL1InitParams);
        AscendC::Fill(l1BTensor[pSlot][2 * s2Align32 * 64 + (s2Align64 - s2Align32) * 64].template ReinterpretCast<uint16_t>(), PL1InitParams);
    }

    __aicore__ inline void PadVS2NotAlign64(uint32_t s2Align64,uint32_t curBaseTileSize,uint32_t embedReal)
    {
        // 补零量 = align64 - align32
        AscendC::InitConstValueParams<uint16_t> kvL1InitParams(1, static_cast<uint16_t>((s2Align64 - curBaseTileSize)), 0, 0x0000);
        // 两个 plane 各补零（fp4 2-element 打包使 L1 NZ fractal 尾部分两段）
        AscendC::Fill(l1ATensor[l1ABufId][MXFP4::ROW_SUM_NUM / 2 * curBaseTileSize].template ReinterpretCast<uint16_t>(), kvL1InitParams);
        // AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Fill(l1ATensor[l1ABufId][MXFP4::ROW_SUM_NUM / 2 * curBaseTileSize * 2 + (s2Align64 - curBaseTileSize) * MXFP4::ROW_SUM_NUM / 2].template ReinterpretCast<uint16_t>(), kvL1InitParams);
    }


    template <class TensorV, class TensorC, class TaskInfoT, class TileInfoT>
    __aicore__ inline
    void operator()(
        TensorV &gV,                  // tla GM 张量（V，ColumnMajor(D,S2)，已在调用处按 gmOffsetV 子缓冲化）
        AscendC::GlobalTensor<uint8_t> gVDequantScale,   // V-scale 基址；block 按 delay20TaskInfo.gmOffsetVScale 取子缓冲
        AscendC::GlobalTensor<int32_t> gSparseIdx,        // sparseIdx 基址；block 按 delay20TaskInfo.gmOffsetSparseIdx 取子缓冲
        TensorC &ubOTmpTensor,        // tla UB 张量（Oᵀ，非标 RowMajor：D+1×s1Align128、stride 64，由 kernel 构）
        GemmCoord actualBlockShapePV,
        TileInfoT &delay20TileInfo,
        TaskInfoT &delay20TaskInfo,   // block 解析 gmOffsetVScale/gmOffsetSparseIdx/kvSeqlen/yBlockNumAval/yBlockNumRsvd
        uint32_t kvSBaseTile, uint32_t blockShapeY, uint32_t kvHeadMul)  // [BSND stride 修复]V-scale 头内 group-stride 因子
    {
        uint32_t rowNum = actualBlockShapePV[0];
        uint32_t embed = actualBlockShapePV[1];
        uint32_t curBaseTileSize = actualBlockShapePV[2];

        uint32_t embedReal = embed;
        // [改造] 在 block 内解析 delay20TaskInfo：取 V-scale/sparseIdx GM 子缓冲 + gather 标量
        AscendC::GlobalTensor<uint8_t> gVScale = gVDequantScale[delay20TaskInfo.gmOffsetVScale];
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx = gSparseIdx[delay20TaskInfo.gmOffsetSparseIdx];
        int64_t  kvSeqlen      = delay20TaskInfo.kvSeqlen;
        uint32_t yBlockNumAval = delay20TaskInfo.yBlockNumAval;
        uint32_t yBlockNumRsvd = delay20TaskInfo.yBlockNumRsvd;
        uint32_t gatheredKvSTileIdx = delay20TileInfo.pvGatheredKvSTileIdx;
        // [#4] V L1 四路轮转：l1ABufId 是成员变量（0..3）
        // P L1 用 delay20TileInfo.loop % 20 轮转（对应 QFA info.loop % 20），
        //   P 由 vector 推送到此槽，block 从此槽读。P L1 不走 event 同步（靠 delay20 调度）
        uint32_t pSlot = delay20TileInfo.loop % MXFP4::L1_P_BUF_CNT;
        // [验证] P L1 槽位轮转
        // AscendC::printf("[P-CHK] loop=%u pSlot=%u\n", (unsigned)delay20TileInfo.loop, (unsigned)pSlot);
        // l1ABufId 是成员变量，不再 local 声明
        // mmad 维度：C = Vᵀ@P = Oᵀ
        
        // AscendC::printf("[P-CHK] s2Align64=%u s2Align32=%u s1Align128=%u\n", (unsigned)delay20TileInfo.kvsActBaseTileAlign64, (unsigned)delay20TileInfo.kvsActBaseTileAlign32, (unsigned)delay20TaskInfo.qsActBaseTileAlign128);

        uint32_t mMmad = MXFP4::ROW_SUM_NUM + MXFP4::ROW_SUM_PAD_NUM;   // 完全按照144的m方向做矩阵乘

        // --- P 尾部补零（S2 对齐 32 但没到 64 → P pad 到 align64）---
        // if (curBaseTileSize != delay20TileInfo.kvsActBaseTileAlign64) {
        //     PadPS2NotAlign64(delay20TileInfo.kvsActBaseTileAlign64, curBaseTileSize, embedReal, pSlot);
        // }
        // [TODO] 目前为了单体测试正确性，正确内容如下
        if (delay20TileInfo.kvsActBaseTileAlign32 != delay20TileInfo.kvsActBaseTileAlign64) {
            PadPS2NotAlign64(delay20TileInfo.kvsActBaseTileAlign64, delay20TileInfo.kvsActBaseTileAlign32, embedReal, pSlot);
        }

        // AscendC::PipeBarrier<PIPE_ALL>();
        // // --- V 尾部补零（S2 没到 64 对齐 → V pad 到 align64）---
        if (curBaseTileSize != delay20TileInfo.kvsActBaseTileAlign64) {
            PadVS2NotAlign64(delay20TileInfo.kvsActBaseTileAlign64,curBaseTileSize,embedReal);
        }

        AscendC::PipeBarrier<PIPE_MTE2>();

        // [#4] 等 V L1 槽空闲（上次 LoadV 释放了它），对应 QFA WaitFlag<MTE1_MTE2>(KV_EVENT0+kvBufId)
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(KV_EVENT0 + l1ABufId);
        // [验证] V L1 槽轮转
        // AscendC::printf("[L1-CHK] l1ABufId=%u\n", (unsigned)l1ABufId);

        // ===== 稀疏 gather：按 gSparseBlockIdx 把稀疏 V / V-scale 逐段搬到 L1 整 tile =====
        // [step2] V 数据 gather 走 tla：gV(tla ColumnMajor) + l1A(tla nZ) + copyGmToL1A(GetTile 段)。
        //   l1A 直接构 nZ(D, s2Align64)——V GM=ColumnMajor→L1 nZ 是布局定论（恒成立），
        //   不用 LayoutTagL1A（主模板默认 LayoutTagA=RowMajor→zN，与 V 不一致）。
        auto l1ATensorTla = tla::MakeTensor(l1ATensor[l1ABufId],
            tla::MakeLayout<ElementA, layout::nZ>(embedReal, delay20TileInfo.kvsActBaseTileAlign64), Arch::PositionL1{});
        // AscendC::printf("[CPY-G2L1] copy v \n"); 

        SparseVBaseTileL1FullLoad(
            gV, l1ATensorTla, gSparseBlockIdx, gatheredKvSTileIdx, kvSeqlen, kvSBaseTile, blockShapeY,
            yBlockNumAval, yBlockNumRsvd, curBaseTileSize, embedReal, delay20TileInfo.kvsActBaseTileAlign64);
        // [step2] V-scale gather 走 tla：gVScale MxScaleA(ColumnMajor,trans) + l1AScale zZ。
        //   实参 (rows=embed, cols=s2Align64/32) 反推自 raw（nValue=s2Align64/64, dValue=D,
        //   srcDValue=D, dstNzC0Stride=s2Align64/64）。gVScale（line 500 已取 gVScale=base[gmOffsetVScale]）
        //   重解释为 fp8_e8m0_t；mx 特化内部再 reinterpret 成 half 做 B16 Nd2Nz（与 raw 一致）。
        auto gVScaleTensorTla = tla::MakeTensor(
            gVScale.ReinterpretCast<AscendC::fp8_e8m0_t>(),
            tla::MakeMxScaleLayout<AscendC::fp8_e8m0_t, layout::ColumnMajor, false>(kvHeadMul * embedReal, delay20TileInfo.kvsActBaseTileAlign64 / 32),  // [BSND stride 修复] rows 乘 kvHeadMul→group-stride=kvHeadMul*embed*2(BNSD=embed*2)
            Arch::PositionGM{});
        auto l1AScaleTensorTla = tla::MakeTensor(l1AScaleTensor[l1ABufId].ReinterpretCast<AscendC::fp8_e8m0_t>(),
            tla::MakeMxScaleLayout<AscendC::fp8_e8m0_t, layout::zZ, false>(embedReal, delay20TileInfo.kvsActBaseTileAlign64 / 32),
            Arch::PositionL1{});
        // AscendC::printf("[CPY-G2L1] copy v scale \n"); 
        SparseVScaleBaseTileL1FullLoad(
            gVScaleTensorTla, l1AScaleTensorTla, gSparseBlockIdx, gatheredKvSTileIdx, kvSeqlen, kvSBaseTile, blockShapeY,
            yBlockNumAval, yBlockNumRsvd, curBaseTileSize, embedReal, delay20TileInfo.kvsActBaseTileAlign64);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(KV_EVENT0 + l1ABufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(KV_EVENT0 + l1ABufId);
        // 验证用：gather 后 V 经 LoadVToL0→Mmad 得 L0C = Vᵀ@P。
        // stub 下 P=fill(0x22)=fp4 1.0、P-scale/V-scale=fill(0x7f)=e8m0 1.0，
        // 故 L0C[d,m] = Σ_s V[s,d] = (gather 行数) × (V 块值)，可用 desc=123 dump 核对。

        uint32_t l0CEventId = PV_L0C_EVENT0;

        // [#3] L0A/B 双缓冲 ping-pong：pvL0abBufId 轮转 0/1（成员变量，跨 operator() 持久化）
        uint32_t l0ABufId = pvL0abBufId;
        uint32_t l0BBufId = pvL0abBufId;
        uint32_t l0ABEventId = PV_L0AB_EVENT0 + pvL0abBufId;

        // [#5] tile 开头（isTileGoupFirstTile）等上一 TileGroup 的 fixpipe 完成（L0C 可复用）
        if (delay20TileInfo.isTileGoupFirstTile) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
        }

        // [#3] 等上次 Mmad 释放 L0A/B 槽（M_MTE1：M→MTE1，Mmad 完成后 L0A/B 可复用）
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABEventId);


        // [验证] L0A/B bufId 轮转
        // AscendC::printf("[L0-CHK] pvL0abBufId=%u\n", (unsigned)pvL0abBufId);
        // AscendC::printf("[CPY-G2L1] LoadPToL0 \n"); 
        LoadPToL0(pSlot, l0BBufId,delay20TileInfo.kvsActBaseTileAlign64, embedReal, delay20TaskInfo.qsActBaseTileAlign64);
        // AscendC::printf("[CPY-G2L1] LoadVToL0 \n"); 
        LoadVToL0(l1ABufId, l0ABufId, embedReal, delay20TileInfo.kvsActBaseTileAlign64);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ABEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ABEventId);
        // AscendC::printf("[CPY-G2L1] MatmulPV \n"); 
        // [#1] initC=isTileGoupFirstTile：TileGroup 首 tile 重置 L0C、tile 内累加

        MatmulPV(l0ABufId, l0BBufId, mMmad, delay20TaskInfo.qsActBaseTileAlign64, delay20TileInfo.kvsActBaseTileAlign64, delay20TileInfo.isTileGoupFirstTile);

        // [#3] Mmad 完成 → 释放 L0A/B 槽给下次（M_MTE1：M→MTE1），轮转
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABEventId);
        pvL0abBufId = (pvL0abBufId + 1) % MXFP4::L0A_PV_BUF_CNT;
        // AscendC::DumpTensor(l0CTensor[0], 123, 128);
        // AscendC::DumpTensor(l0CTensor[1024], 124, 128);
        // AscendC::DumpTensor(l0CTensor[0][1024 * 2], 125, 128);

        if (delay20TileInfo.isUpdatePScale) {  // tile 末尾 → Fixpipe L0C→UB
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            // AscendC::printf("[CPY-G2L1] FixpipeMm2 \n"); 
            // 如果在s1不满64的时候，全部由某一个子vector核处理。如果fixpipe也要缩小基块到64的话，在vect的处理不变的情况下，就需要在s1不满64但是>32的时候，切换搬运模式到单目标搬出
            if (delay20TaskInfo.qsActBaseTileAlign8 <= 64) {
                FixpipeMm2SingleVect(delay20TaskInfo.qsActBaseTileAlign64,ubOTmpTensor);
            } else {
                FixpipeMm2(delay20TaskInfo.qsActBaseTileAlign64,ubOTmpTensor);
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);  // [#5] FIX→M: fixpipe 完成，L0C 释放给下一 TileGroup
        }

        // [#4] V L1→L0 完成（LoadVToL0 已过）→ 释放 V L1 槽给下次 gather，轮转
        //   必须在 isLastSecondKvsTile 之前：re-seed 的 InitL0BufferForReduceSum 会 Wait 新 slot 的
        //   MTE1_MTE2，需要这里先 Set 当前 slot + 轮转到新 slot
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(KV_EVENT0 + l1ABufId);
        l1ABufId = (l1ABufId + 1) % MXFP4::L1_KV_BUF_CNT;

        if (delay20TileInfo.isLastSecondKvsTile) {
            InitL0BufferForReduceSum();  // 倒数第二块 → re-seed L0A +16=1.0 给最后块 rowsum
        }

        // [诊断] 确认轮转是否执行
        // AscendC::printf("[L1-ROT] l1ABufId after rotate=%u\n", (unsigned)l1ABufId);
    }


protected:
    // L1：V 用 KV 区 4 份，P 用 P 区 20 份
    AscendC::LocalTensor<ElementA> l1ATensor[MXFP4::L1_KV_BUF_CNT];
    AscendC::LocalTensor<ElementB> l1BTensor[MXFP4::L1_P_BUF_CNT];
    // L0：A/B 各 L0_STAGES 份 ping-pong，C 只用 1 份
    AscendC::LocalTensor<ElementA> l0ATensor[MXFP4::L0A_PV_BUF_CNT];
    AscendC::LocalTensor<ElementB> l0BTensor[MXFP4::L0A_PV_BUF_CNT];
    AscendC::LocalTensor<float> l0CTensor[MXFP4::L0C_PV_BUF_CNT];

    // MXFP4 专用：E8M0 scale 的 L1 buffer（V-scale 4 份，P-scale 20 份）+ MX 版 L1->L0 拷贝器
    AscendC::LocalTensor<uint8_t> l1AScaleTensor[MXFP4::L1_KV_BUF_CNT];
    AscendC::LocalTensor<uint8_t> l1BScaleTensor[MXFP4::L1_P_BUF_CNT];

    // [改造-新增] policy functor 成员（原文件只有裸 tensor）。
    TileMmad tileMmad;                 // fp4 Mmad（3 参，scale 已在 L1->L0 烘焙）
    Tile::CopyL1ToL0AMxA5 copyL1ToL0AMx;   // V: nZ L1 -> zN L0A 转置（带 V-scale）
    Tile::CopyL1ToL0BMxA5 copyL1ToL0BMx;   // P: zN L1 -> nZ L0B 转置（带 P-scale）

    uint32_t pvL0abBufId = 0;   // [#3] L0A/B 双缓冲轮转索引（0/1），跨 operator() 调用持久化
    uint32_t& l1ABufId;        // [QK/PV 共享] KV 槽游标——引用 kernel 的 kvBufId，与 QK kBufId 同址（0..3）
    
    uint32_t l1ATileM;
    uint32_t l1BTileN;
    uint32_t l1ATileK;
    uint32_t l1BTileK;
    uint32_t l1ABufNum;
    uint32_t l1BBufNum;
};

}  // namespace NpuArch::Gemm::Block

#endif  // GEMM_BLOCK_PV_ARCH35_MXFP4_HPP
