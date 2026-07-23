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

    // L1 各区单份字节数（fp4 两元素占 1 字节；scale 为 E8M0，每 32 元素 1 字节）
    static constexpr uint32_t L1_Q_SIZE = 128 * 128 / 2;                                          // Q 数据：8K
    static constexpr uint32_t L1_Q_DESCALE_SIZE = 128 * (128 / 32) * sizeof(AscendC::fp8_e8m0_t); // Q-scale：0.5K
    static constexpr uint32_t L1_KV_SIZE = 512 * 128 / 2;                                         // KV/V 数据：32K
    static constexpr uint32_t L1_KV_DESCALE_SIZE = 512 * (128 / 32) * sizeof(AscendC::fp8_e8m0_t);// KV/V-scale：2K
    static constexpr uint32_t L1_P_SIZE = 128 * 256 / 2;                                          // P 数据：16K
    static constexpr uint32_t L1_P_DESCALE_SIZE = 128 * 2 * (256 / 64 + 1) * sizeof(AscendC::fp8_e8m0_t); // P-scale：1.25K
    static constexpr uint32_t L1_Q_BUFCNT = 2;    // Q 区份数
    static constexpr uint32_t L1_KV_BUFCNT = 4;   // KV/V 区份数
    static constexpr uint32_t L1_P_BUFCNT = 20;   // P 区份数

    static constexpr uint32_t L1_SINGLE_GLOBAL_MAX_SIZE = 128;

    // L0 各区单份字节数
    static constexpr uint32_t QK_L0A_SIZE = 128 * 128 / 2;              // QK 用 L0A：8K
    static constexpr uint32_t QK_L0B_SIZE = 128 * 128 / 2;              // QK 用 L0B：8K
    static constexpr uint32_t QK_L0C_SIZE = 128 * 128 * sizeof(float);  // QK 用 L0C：64K
    static constexpr uint32_t QK_L0AB_BUFCNT = 2;
    static constexpr uint32_t QK_L0C_BUFCNT = 2;
    static constexpr uint32_t PV_L0A_SIZE = 256 * (128 + 16) / 2;             // PV 用 L0A(=V, D+16 行)：18K
    static constexpr uint32_t PV_L0B_SIZE = 256 * (128 + 16) / 2;            // PV 用 L0B(=P)：18K
    static constexpr uint32_t PV_L0C_SIZE = (128 + 16) * 128 * sizeof(float); // PV 用 L0C(=Oᵀ)：72K
    static constexpr uint32_t V_SCALE_L0A_SIZE = (256 / 64) * ((128 + 16) / 16) * 32 ;
    static constexpr uint32_t P_SCALE_L0B_SIZE = (256 / 64) * ((128 + 16) / 16) * 32 ;
    
    static constexpr uint32_t PV_L0AB_BUFCNT = 2;                    // mm2 L0A/B 双缓冲
    static constexpr uint32_t PV_L0C_BUFCNT = 1;                     // mm2 L0C 单缓冲（原地累加）
    static constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag> &resource) {
        // L1 区域顺序（与 QFA 一致）：P 数据 | P-scale | Q 数据 | Q-scale | KV(=V) 数据 | KV(=V)-scale
        uint32_t addrL1Start = 0;
        // P 数据：l1BTensor(B=P)，L1_P_SIZE * 20 = 320K
        for (uint32_t i = 0; i < L1_P_BUFCNT; i++) {
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(addrL1Start + L1_P_SIZE * i);
        }
        addrL1Start += L1_P_SIZE * L1_P_BUFCNT;

        // P-scale：l1BScaleTensor，起始记为 pScaleBase，占 L1_P_DESCALE_SIZE * 20 = 25K
        uint32_t pScaleBase = addrL1Start;
        addrL1Start += L1_P_DESCALE_SIZE * L1_P_BUFCNT;

        // Q 数据：PV 不用，仅占位以对齐后续偏移（16K）
        addrL1Start += L1_Q_SIZE * L1_Q_BUFCNT;

        // Q-scale：PV 不用，仅占位（1K）
        addrL1Start += L1_Q_DESCALE_SIZE * L1_Q_BUFCNT;

        // KV/V 数据：l1ATensor(A=V)，L1_KV_SIZE * 4 = 128K
        for (uint32_t i = 0; i < L1_KV_BUFCNT; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(addrL1Start + L1_KV_SIZE * i);
        }
        addrL1Start += L1_KV_SIZE * L1_KV_BUFCNT;

        // KV/V-scale：l1AScaleTensor(V-scale)，起始记为 vScaleBase，占 L1_KV_DESCALE_SIZE * 4 = 8K
        uint32_t vScaleBase = addrL1Start;
        addrL1Start += L1_KV_DESCALE_SIZE * L1_KV_BUFCNT;
        // （QFA 在此之后还有 localGlobalMaxL1 = 256*4，PV 用不到，不再申请）

        for (uint32_t i = 0; i < L1_P_BUFCNT; i++) {
            l1BScaleTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(pScaleBase + L1_P_DESCALE_SIZE * i);
        }
        for (uint32_t i = 0; i < L1_KV_BUFCNT; i++) {
            l1AScaleTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(vScaleBase + L1_KV_DESCALE_SIZE * i);
        }
        
        // L0A：跳过 QK 区(8K*2)，PV 区取 L0_STAGES 份 ping-pong —— l0ATensor(A=V)
        uint32_t addrL0AStart = QK_L0A_SIZE * QK_L0AB_BUFCNT;
        for (uint32_t i = 0; i < L0_STAGES; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(addrL0AStart + PV_L0A_SIZE * i);
        }

        // L0B：跳过 QK 区(8K*2)，PV 区取 L0_STAGES 份 ping-pong —— l0BTensor(B=P)
        uint32_t addrL0BStart = QK_L0B_SIZE * QK_L0AB_BUFCNT;
        for (uint32_t i = 0; i < L0_STAGES; i++) {
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(addrL0BStart + PV_L0B_SIZE * i);
        }

        // L0C：跳过 QK 区后取 1 份（不 ping-pong）。字节偏移 = QK_L0C_SIZE*QK_L0C_BUFCNT = 128K，
        //      与 QFA addrL0CStart 一致（累加的就是字节地址，不能再 /sizeof(float)）。
        uint32_t addrL0CStart = QK_L0C_SIZE * QK_L0C_BUFCNT;
        l0CTensor[0] = resource.l0CBuf.template GetBufferByByte<float>(addrL0CStart);

        // 初始化L0A中v pad的16行
        InitL0BufferForReduceSum();
    }
    __aicore__ inline ~BlockMmadTla() {}

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
    __aicore__ inline void LoadPToL0(uint32_t l1BBufId, uint32_t l0BBufId,
                                     uint32_t s1Align128, uint32_t s2Align64,
                                     uint32_t embedReal)
    {
        // s2Base 取整 tile S2Base(=256)，用于推导 P-scale srcStride（原硬编码 5 的来源）。
        // [TODO] s2Base 应由调用方/TileShape 传入，此处暂以 256 占位（与 L1_P_SIZE 排布一致）。
        constexpr uint32_t s2Base = 256;
        copyL1ToL0BMx(
            l0BTensor[l0BBufId],
            l1BTensor[l1BBufId],
            l1BScaleTensor[l1BBufId].template ReinterpretCast<AscendC::fp8_e8m0_t>(),
            s1Align128, s2Align64, embedReal, s2Base, 5);
    }

    // [改造] Mmad 改走 tileMmad functor（原裸 MatmulPV 的 AscendC::Mmad 下沉到 TileMmadTla）。
    //   C=Vᵀ@P=Oᵀ，m=D+16（含 rowsum seed 行）、n=M、k=S2(对齐64)。
    //   tileMmad 只读 .data()（不消费 layout stride），故 fp4 粒度不影响；这里包 tla 张量传入。
    __aicore__ inline void MatmulPV(uint32_t l0ABufId, uint32_t l0BBufId,
                                    uint32_t m, uint32_t n, uint32_t k)
    {
        // [改造] 裸 LocalTensor 包成 tla 张量（布局对象仅用于满足 MakeTensor 签名，tileMmad 不读其 stride）。
        auto l0ALayout = tla::MakeLayout<ElementA, LayoutTagL0A>(m, k);
        auto l0ATensorTla = tla::MakeTensor(l0ATensor[l0ABufId], l0ALayout, Arch::PositionL0A{});
        auto l0BLayout = tla::MakeLayout<ElementB, LayoutTagL0B>(k, n);
        auto l0BTensorTla = tla::MakeTensor(l0BTensor[l0BBufId], l0BLayout, Arch::PositionL0B{});
        auto l0CLayout = tla::MakeLayoutL0C(m, n);
        auto l0CTensorTla = tla::MakeTensor(l0CTensor[0], l0CLayout, Arch::PositionL0C{});
        tileMmad(l0CTensorTla, l0ATensorTla, l0BTensorTla, m, n, k, /*initC=*/true);
    }


    __aicore__ inline void CopyInA1(uint32_t m,uint32_t k,uint32_t alignK,AscendC::GlobalTensor<fp4x2_e2m1_t> gV)
    {
        AscendC::Nd2NzParams nd2nzA1Params;

        uint16_t aColValue = CeilDivision(m, 2);
        nd2nzA1Params.ndNum = 1;  // ND矩阵的数目
        nd2nzA1Params.nValue = k; // 源操作ND矩阵的行数
        nd2nzA1Params.dValue = aColValue; // 源操作ND矩阵的列数，FP4类型下2个连续数据合并为1个FP8数据进行搬运
        nd2nzA1Params.srcNdMatrixStride = 0; // 源操作数相邻ND矩阵间的起始地址偏移
        nd2nzA1Params.srcDValue = aColValue; // 源操作数同一ND矩阵内相邻行的起始地址偏移
        nd2nzA1Params.dstNzC0Stride = alignK; // ND转Nz后，同一行数据切分后各段的起始地址间隔，单位32B
        nd2nzA1Params.dstNzNStride = 1; // ND矩阵第x行与第x+1行转Nz后在dst中的偏移
        nd2nzA1Params.dstNzMatrixStride = 0; // 目的Nz矩阵中，相邻Nz矩阵起始地址的偏移，配置为0时，无意义
        
        AscendC::DataCopy(l1ATensor[0], gV, nd2nzA1Params);
    }

    __aicore__ inline void CopyInScaleA1(uint32_t m,uint32_t scaleK,AscendC::GlobalTensor<uint8_t> gVScale )
    {
        AscendC::GlobalTensor<half> scaleAGMB16;
        scaleAGMB16.SetGlobalBuffer((__gm__ half*)(gVScale.GetPhyAddr()), m * scaleK / 2);
        auto l1AScaleTensorB16 = l1AScaleTensor[0].ReinterpretCast<half>();
        // 转置输入时scaleA的GM shape为[scaleK, m, 2]，按B16视图使用Nd2NzParams搬运
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;           // 源操作数中ND矩阵的个数
        nd2nzParams.nValue = scaleK / 2; // 源操作ND矩阵的行数，按B16视图后2个fp8 scale合并为1个half
        nd2nzParams.dValue = m;          // 源操作ND矩阵的列数
        nd2nzParams.srcDValue = m;       // 源操作数同一ND矩阵内相邻行的起始地址偏移
        nd2nzParams.dstNzC0Stride = scaleK / 2; // ND转Nz后，同一行数据切分后各段的起始地址间隔，单位32B
        nd2nzParams.dstNzNStride = 1;      // ND矩阵第x行与第x+1行转Nz后在dst中的偏移
        nd2nzParams.dstNzMatrixStride = 0; // 相邻Nz矩阵间的起始地址偏移

        AscendC::DataCopy(l1AScaleTensorB16, scaleAGMB16, nd2nzParams);
    }

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
            AscendC::printf("[V-CHK] oriStart=%u dealtAccum=%u curDealtLen=%u srcOff=%u(want %u) dstOff=%u(want %u)\n",
                (unsigned)oriStartOffset, (unsigned)dealtLenAccum, (unsigned)curDealtLen,
                (unsigned)static_cast<uint32_t>(gVTensorTile.layout()(gVTensorTile.coord())),
                (unsigned)(oriStartOffset * embed),
                (unsigned)static_cast<uint32_t>(l1ATensorTile.layout()(l1ATensorTile.coord())),
                (unsigned)(dealtLenAccum * 64));

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
            AscendC::printf("[VS-CHK] oriStart=%u dealtAccum=%u curDealtLen=%u srcOff=%u(want %u=2*srcB16) dstOff=%u(want %u=2*dstB16)\n",
                (unsigned)oriStartOffset, (unsigned)dealtLenAccum, (unsigned)curDealtLen,
                (unsigned)static_cast<uint32_t>(gVScaleTile.layout()(gVScaleTile.coord())),
                (unsigned)(rawSrcB16Off * 2),
                (unsigned)static_cast<uint32_t>(l1AScaleTile.layout()(l1AScaleTile.coord())),
                (unsigned)(rawDstB16Off * 2));
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
    __aicore__ inline void FixpipeMm2(uint32_t s1Align128, uint32_t embedReal, TensorUB &ubOTmpTensor)
    {
        // L0C tla：rows=D+16（与 MatmulPV 的 mMmad=D+16 一致）→ srcStride=D+16
        auto l0CTensorTla = tla::MakeTensor(l0CTensor[0],
            tla::MakeLayoutL0C(embedReal + 16, s1Align128), Arch::PositionL0C{});
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
        AscendC::printf("[FP-CHK] mSize=%u(want %u) nSizeBase=%u(want %u) srcStride=%u(want %u) dstStride=%u(want 64) dualDstCtl=2\n",
            (unsigned)fpMSize, (unsigned)(embedReal + 1),
            (unsigned)fpNSizeBase, (unsigned)s1Align128,
            (unsigned)fpSrcStride, (unsigned)(embedReal + 16),
            (unsigned)fpDstStride);
        copyL0CToDst(ubOTmpTensor, l0CTensorTla);   // operator()(dst=UB, src=L0C)
    }

    __aicore__ inline uint16_t CeilDivision(uint16_t numerator, uint16_t denominator)
    {
        return (numerator + denominator - 1) / denominator;
    }

    __aicore__ inline void InitL0BufferForReduceSum()
    {
          // 用 Fill 假数据填 L1：l1ATensor=V、l1BTensor=P 及各自 scale，
            // 份数按各自 L1 buffer 字节数 /32（Fill 单位是 32B block），不越界。
            AscendC::InitConstValueParams<uint16_t> vFillParams(1, static_cast<uint16_t>(PV_L0A_SIZE / 32 ), 0, 0x2222);
            AscendC::Fill(l1ATensor[0].template ReinterpretCast<uint16_t>(), vFillParams);          // V 数据
            AscendC::PipeBarrier<PIPE_MTE2>();
            AscendC::InitConstValueParams<uint16_t> vscaleFillParams(1, static_cast<uint16_t>(V_SCALE_L0A_SIZE / 32), 0, 0x7f7f);
            AscendC::Fill(l1AScaleTensor[0].template ReinterpretCast<uint16_t>(), vscaleFillParams); // V scale
            AscendC::PipeBarrier<PIPE_MTE2>();
            // P 数据：只填 S2*M(=L1_P_SIZE=16K)。【不可用 PV_L0B_SIZE(18K)】——
            //   P 无 +16 seed 行(只有 L0A/V 才有)，L1 P 单槽=L1_P_SIZE=16K；
            //   PV_L0B_SIZE(18K) 里的 +16 是 L0B 侧对称 padding，不来自 L1，填 18K 会越界 2K 到 l1BTensor[1]。
            AscendC::InitConstValueParams<uint16_t> pFillParams(1, static_cast<uint16_t>(L1_P_SIZE / 32), 0, 0x2222);
            AscendC::Fill(l1BTensor[0].template ReinterpretCast<uint16_t>(), pFillParams);           // P 数据
            AscendC::PipeBarrier<PIPE_MTE2>();
            AscendC::InitConstValueParams<uint16_t> pscaleFillParams(1, static_cast<uint16_t>(L1_P_DESCALE_SIZE / 32), 0, 0x7f7f);
            AscendC::Fill(l1BScaleTensor[0].template ReinterpretCast<uint16_t>(), pscaleFillParams); // P scale
            AscendC::PipeBarrier<PIPE_MTE2>();

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(7);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(7);
            // [gap G6] QFA 用 KV_EVENT0+kvBufId(=4) 做 Fill->LoadData 的 MTE2->MTE1 同步；
            //   此处用 event=7(=KV_EVENT3 槽)，单缓冲假数据下不冲突，但与隐式 kvBufId=0 不一致

            // seed 装载：把 V 的全部 D+16 行(含 rowsum seed)非转置装入 L0A。
            // --- 对照 QFA InitL0BufferForReduceSum() (quant_flash_attn_block_cube_dn.h:627) ---
            //   QFA: Fill kvL1(0x6666)+kvDescale(0x7d7d) → LoadData pvL0A，m=(128+16)/16, k=256/64, ifTranspose=false，
            //        循环 2 次填两份 ping-pong L0A。本处：假数据已由上方 Fill 填好，此处只 LoadData 一次到 L0A[0]。
            // k 必须用真实 S2(=kMmad)：否则 +16 seed 行按 256 排布、真实 V 行/mmad 按 S2 排布，
            // 两者在 L0A 的 k 方向错位 → mmad 尾部 16 行(dump 里那 256 个)乱值。
            // [与 QFA 差异] QFA 硬编码 k=256/64(因 s2BaseSize=256)；本处用 kMmad/64 适配真实 S2(已修)。
            AscendC::LoadData2DParamsV2 loadData2DParamsA;
            loadData2DParamsA.mStartPosition = 0;
            loadData2DParamsA.kStartPosition = 0;
            loadData2DParamsA.mStep = (128 + 16) / 16;   // m = D+16
            loadData2DParamsA.kStep = 256 / 64;              // k = S2（对齐 64），与 mmad k 一致
            loadData2DParamsA.srcStride = loadData2DParamsA.mStep;
            loadData2DParamsA.dstStride = loadData2DParamsA.mStep;
            loadData2DParamsA.ifTranspose = false;             // 不转置

            AscendC::LoadData2DMxParams loadData2DMXParamsA;
            loadData2DMXParamsA.xStartPosition = 0;
            loadData2DMXParamsA.yStartPosition = 0;
            loadData2DMXParamsA.xStep = (128 + 16) / 16; // x = m = D+16
            loadData2DMXParamsA.yStep = 256 / 64;            // y = k = S2（对齐 64）
            loadData2DMXParamsA.srcStride = loadData2DMXParamsA.yStep;
            loadData2DMXParamsA.dstStride = loadData2DMXParamsA.yStep;

            LoadData(l0ATensor[0].template ReinterpretCast<fp4x2_e2m1_t>(),
                    l1ATensor[0].template ReinterpretCast<fp4x2_e2m1_t>(),
                    l1AScaleTensor[0].template ReinterpretCast<AscendC::fp8_e8m0_t>(),
                    loadData2DParamsA, loadData2DMXParamsA);
            AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void PadPS2NotAlign64(uint32_t s2Align64,uint32_t s2Align32,uint32_t embedReal)
    {
        // 补零量 = align64 - align32
        AscendC::InitConstValueParams<uint16_t> PL1InitParams(1, static_cast<uint16_t>(s2Align64 - s2Align32), 0, 0x0000);
        AscendC::Fill(l1BTensor[0][s2Align32 * 64].template ReinterpretCast<uint16_t>(), PL1InitParams);
        if (embedReal == 128) {  // 满块 M 才补第二个 plane（sub1 的 P 半块）
            AscendC::Fill(l1BTensor[0][2 * s2Align32 * 64 + 32 * 64].template ReinterpretCast<uint16_t>(), PL1InitParams);
        }
    }

    __aicore__ inline void PadVS2NotAlign64(uint32_t s2Align64,uint32_t curBaseTileSize,uint32_t embedReal)
    {
        // 补零量 = align64 - align32
        AscendC::InitConstValueParams<uint16_t> kvL1InitParams(1, static_cast<uint16_t>((s2Align64 - curBaseTileSize)), 0, 0x0000);
        // 两个 plane 各补零（fp4 2-element 打包使 L1 NZ fractal 尾部分两段）
        AscendC::Fill(l1ATensor[0][embedReal / 2 * curBaseTileSize].template ReinterpretCast<uint16_t>(), kvL1InitParams);
        // AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Fill(l1ATensor[0][embedReal / 2 * curBaseTileSize * 2 + (s2Align64 - curBaseTileSize) * embedReal / 2].template ReinterpretCast<uint16_t>(), kvL1InitParams);
    }


    template <class TensorV, class TensorC, class TaskInfoT, class TileInfoT>
    __aicore__ inline
    void operator()(
        TensorV &gV,                  // tla GM 张量（V，ColumnMajor(D,S2)，已在调用处按 gmOffsetV 子缓冲化）
        AscendC::GlobalTensor<uint8_t> gVDequantScale,   // V-scale 基址；block 按 delay20TaskInfo.gmOffsetVScale 取子缓冲
        AscendC::GlobalTensor<int32_t> gSparseIdx,        // sparseIdx 基址；block 按 delay20TaskInfo.gmOffsetSparseIdx 取子缓冲
        TensorC &ubOTmpTensor,        // tla UB 张量（Oᵀ，RowMajor）
        GemmCoord actualBlockShapePV,
        TileInfoT &delay20TileInfo,
        TaskInfoT &delay20TaskInfo,   // block 解析 gmOffsetVScale/gmOffsetSparseIdx/kvSeqlen/yBlockNumAval/yBlockNumRsvd
        uint32_t kvSBaseTile, uint32_t blockShapeY)
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
        // 验证阶段只用 0 号 buffer（l1ABufNum/l1BBufNum 未初始化，不能拿来取模）
        uint32_t l1BBufId = 0;              // P (l1BTensor)
        uint32_t l1BEventId = l1BBufId + 3;
        uint32_t l1ABufId = 0;              // V (l1ATensor)
        // mmad 维度：C = Vᵀ@P = Oᵀ
        uint32_t s2Align64 = RoundUp(curBaseTileSize, 64);
        uint32_t s2Align32 = RoundUp(curBaseTileSize, 32);
        uint32_t s1Align128 = RoundUp(rowNum, 128);
        
 

        uint32_t mMmad = embedReal + 16;   // m = D+16（含 rowsum seed 行）
        uint32_t nMmad = s1Align128;           // n = M
        uint32_t kMmad = s2Align64;        // k = S2（对齐 64）

        // todo 规避行为，qk占用了pv的buffer
        // InitL0BufferForReduceSum();

        // --- P 尾部补零（S2 对齐 32 但没到 64 → P pad 到 align64）---
        if (s2Align32 != s2Align64) {
            PadPS2NotAlign64(s2Align64,s2Align32,embedReal);
        }

        // AscendC::PipeBarrier<PIPE_ALL>();
        // // --- V 尾部补零（S2 没到 64 对齐 → V pad 到 align64）---
        if (curBaseTileSize != s2Align64) {
            PadVS2NotAlign64(s2Align64,curBaseTileSize,embedReal);
        }

        AscendC::PipeBarrier<PIPE_MTE2>();

        // ===== 稀疏 gather：按 gSparseBlockIdx 把稀疏 V / V-scale 逐段搬到 L1 整 tile =====
        // [step2] V 数据 gather 走 tla：gV(tla ColumnMajor) + l1A(tla nZ) + copyGmToL1A(GetTile 段)。
        //   l1A 直接构 nZ(D, s2Align64)——V GM=ColumnMajor→L1 nZ 是布局定论（恒成立），
        //   不用 LayoutTagL1A（主模板默认 LayoutTagA=RowMajor→zN，与 V 不一致）。
        auto l1ATensorTla = tla::MakeTensor(l1ATensor[l1ABufId],
            tla::MakeLayout<ElementA, layout::nZ>(embedReal, s2Align64), Arch::PositionL1{});
        SparseVBaseTileL1FullLoad(
            gV, l1ATensorTla, gSparseBlockIdx, gatheredKvSTileIdx, kvSeqlen, kvSBaseTile, blockShapeY,
            yBlockNumAval, yBlockNumRsvd, curBaseTileSize, embedReal, s2Align64);
        // [step2] V-scale gather 走 tla：gVScale MxScaleA(ColumnMajor,trans) + l1AScale zZ。
        //   实参 (rows=embed, cols=s2Align64/32) 反推自 raw（nValue=s2Align64/64, dValue=D,
        //   srcDValue=D, dstNzC0Stride=s2Align64/64）。gVScale（line 500 已取 gVScale=base[gmOffsetVScale]）
        //   重解释为 fp8_e8m0_t；mx 特化内部再 reinterpret 成 half 做 B16 Nd2Nz（与 raw 一致）。
        auto gVScaleTensorTla = tla::MakeTensor(
            gVScale.ReinterpretCast<AscendC::fp8_e8m0_t>(),
            tla::MakeMxScaleLayout<AscendC::fp8_e8m0_t, layout::ColumnMajor, false>(embedReal, s2Align64 / 32),
            Arch::PositionGM{});
        auto l1AScaleTensorTla = tla::MakeTensor(l1AScaleTensor[l1ABufId].ReinterpretCast<AscendC::fp8_e8m0_t>(),
            tla::MakeMxScaleLayout<AscendC::fp8_e8m0_t, layout::zZ, false>(embedReal, s2Align64 / 32),
            Arch::PositionL1{});
        SparseVScaleBaseTileL1FullLoad(
            gVScaleTensorTla, l1AScaleTensorTla, gSparseBlockIdx, gatheredKvSTileIdx, kvSeqlen, kvSBaseTile, blockShapeY,
            yBlockNumAval, yBlockNumRsvd, curBaseTileSize, embedReal, s2Align64);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);
        // 验证用：gather 后 V 经 LoadVToL0→Mmad 得 L0C = Vᵀ@P。
        // stub 下 P=fill(0x22)=fp4 1.0、P-scale/V-scale=fill(0x7f)=e8m0 1.0，
        // 故 L0C[d,m] = Σ_s V[s,d] = (gather 行数) × (V 块值)，可用 desc=123 dump 核对。

        uint32_t l0CBufId = 0;
        uint32_t l0CEventId = l0CBufId + 2;

        uint32_t l0ABufId = 0;
        uint32_t l0BBufId = 0;
        uint32_t l0AEventId = l0ABufId;
        uint32_t l0BEventId = l0BBufId + 2;

        LoadPToL0(l1BBufId,l0BBufId,s1Align128,s2Align64,embedReal);

        LoadVToL0(l1ABufId,l0ABufId,embedReal,s2Align64);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);

        MatmulPV(l0ABufId, l0BBufId, mMmad, nMmad, kMmad);
        AscendC::DumpTensor(l0CTensor[0], 123, 128);
        // AscendC::DumpTensor(l0CTensor[1024], 124, 128);
        AscendC::DumpTensor(l0CTensor[0][1024 * 2], 125, 128);

        if (delay20TileInfo.isUpdatePScale) {  // tile 末尾 → Fixpipe L0C→UB
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            FixpipeMm2(s1Align128,embedReal,ubOTmpTensor);
        }

        if (delay20TileInfo.isLastSecondKvsTile) {
            InitL0BufferForReduceSum();  // 倒数第二块 → re-seed L0A +16=1.0 给最后块 rowsum
        }
    }

protected:
    // L1：V 用 KV 区 4 份，P 用 P 区 20 份
    AscendC::LocalTensor<ElementA> l1ATensor[L1_KV_BUFCNT];
    AscendC::LocalTensor<ElementB> l1BTensor[L1_P_BUFCNT];
    // L0：A/B 各 L0_STAGES 份 ping-pong，C 只用 1 份
    AscendC::LocalTensor<ElementA> l0ATensor[PV_L0AB_BUFCNT];
    AscendC::LocalTensor<ElementB> l0BTensor[PV_L0AB_BUFCNT];
    AscendC::LocalTensor<float> l0CTensor[PV_L0C_BUFCNT];

    // MXFP4 专用：E8M0 scale 的 L1 buffer（V-scale 4 份，P-scale 20 份）+ MX 版 L1->L0 拷贝器
    AscendC::LocalTensor<uint8_t> l1AScaleTensor[L1_KV_BUFCNT];
    AscendC::LocalTensor<uint8_t> l1BScaleTensor[L1_P_BUFCNT];

    // [改造-新增] policy functor 成员（原文件只有裸 tensor）。
    TileMmad tileMmad;                 // fp4 Mmad（3 参，scale 已在 L1->L0 烘焙）
    Tile::CopyL1ToL0AMxA5 copyL1ToL0AMx;   // V: nZ L1 -> zN L0A 转置（带 V-scale）
    Tile::CopyL1ToL0BMxA5 copyL1ToL0BMx;   // P: zN L1 -> nZ L0B 转置（带 P-scale）

    uint32_t l1ATileM;
    uint32_t l1BTileN;
    uint32_t l1ATileK;
    uint32_t l1BTileK;
    uint32_t l1ABufNum;
    uint32_t l1BBufNum;
};

}  // namespace NpuArch::Gemm::Block

#endif  // GEMM_BLOCK_PV_ARCH35_MXFP4_HPP
