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
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;

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

    __aicore__ inline void LoadVToL0(uint32_t l1ABufId,uint32_t l0ABufId,uint32_t embedReal,uint32_t s2Align64)
    {
        AscendC::LoadData2DParamsV2 loadData2DParamsA;
        loadData2DParamsA.mStartPosition = 0;
        loadData2DParamsA.kStartPosition = 0;
        loadData2DParamsA.mStep = s2Align64 / 16;                  // 源 m = S2（16-块单位）[已参数化]
        loadData2DParamsA.kStep = embedReal / 64;                  // 源 k = D（64-块单位）[已参数化]
        loadData2DParamsA.srcStride = s2Align64 / 16;
        loadData2DParamsA.dstStride = (embedReal + 15) / 16 + 1;   // L0A dst stride（+1 给 D+16 留间隔）
        loadData2DParamsA.ifTranspose = true;                // 转置：源 (S2,D) → L0A (D,S2)

        AscendC::LoadData2DMxParams load2DMxParamsA;
        load2DMxParamsA.xStartPosition = 0;
        load2DMxParamsA.yStartPosition = 0;
        load2DMxParamsA.xStep = (embedReal + 15) / 16;             // 转置后 L0A m = D  [已参数化]
        load2DMxParamsA.yStep = (s2Align64 + 64 - 1) / 64;         // 转置后 L0A k = S2（V 量化轴）[已参数化]
        load2DMxParamsA.srcStride = load2DMxParamsA.yStep;
        load2DMxParamsA.dstStride = load2DMxParamsA.yStep;

        AscendC::LoadData(l0ATensor[0].template ReinterpretCast<fp4x2_e2m1_t>(),          // dst = L0A  [恒[0], gap G5]
                 l1ATensor[0].template ReinterpretCast<fp4x2_e2m1_t>(),                   // src = V(L1)
                 l1AScaleTensor[0].template ReinterpretCast<AscendC::fp8_e8m0_t>(),       // scale = V-scale(L1)
                 loadData2DParamsA, load2DMxParamsA);
    }

    __aicore__ inline void LoadPToL0(uint32_t l1BBufId,uint32_t l0BBufId,uint32_t s1Align128,uint32_t s2Align64,uint32_t embedReal)
    {
        AscendC::LoadData2DParamsV2 loadData2DParamsB;
        loadData2DParamsB.mStartPosition = 0;
        loadData2DParamsB.kStartPosition = 0;
        loadData2DParamsB.mStep = (s2Align64 + 15) / 16;   // 源 m = S2（16-块单位）[已参数化]
        loadData2DParamsB.kStep = s1Align128 / 64;          // 源 k = M（64-块单位）[残留: raw rowNum, 尾块需 align128]
        loadData2DParamsB.srcStride = loadData2DParamsB.mStep;
        loadData2DParamsB.dstStride = s1Align128 / 16;      // L0B dst stride [残留: raw rowNum, 尾块需 align128]
        loadData2DParamsB.ifTranspose = true;        // 转置 → P 在 L0B 直通 (k=S2, n=M)

        AscendC::LoadData2DMxParams load2DMxParamsB;
        load2DMxParamsB.xStartPosition = 0;
        load2DMxParamsB.yStartPosition = 0;
        load2DMxParamsB.xStep = embedReal / 16;            // D=M=128 时 = M/16 [已参数化]
        load2DMxParamsB.yStep = (s2Align64 + 63) / 64;     // P 量化轴 S2（64-块）[已参数化]
        load2DMxParamsB.srcStride = 5;               // [残留硬编码] = S2Base/64+1 = 256/64+1（仅 S2=256 正确）
        load2DMxParamsB.dstStride = load2DMxParamsB.yStep;

        AscendC::LoadData(l0BTensor[0].template ReinterpretCast<fp4x2_e2m1_t>(),          // dst = L0B  [恒[0], gap G5]
                 l1BTensor[0].template ReinterpretCast<fp4x2_e2m1_t>(),                   // src = P(L1)
                 l1BScaleTensor[0].template ReinterpretCast<AscendC::fp8_e8m0_t>(),       // scale = P-scale(L1)
                 loadData2DParamsB, load2DMxParamsB);
    }

    __aicore__ inline void MatmulPV(uint32_t l0ABufId, uint32_t l0BBufId,
                                    uint32_t m, uint32_t n, uint32_t k)
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;   // D+16（含 rowsum seed 行）
        mmadParams.n = n;   // M
        mmadParams.k = k;   // S2（对齐 64）
        mmadParams.cmatrixInitVal = true;
        mmadParams.cmatrixSource = false;
        mmadParams.disableGemv = true;
        AscendC::Mmad(l0CTensor[0],                                          // C = L0C  [恒[0], gap G5]
             l0ATensor[0].template ReinterpretCast<fp4x2_e2m1_t>(),          // A = V(L0A，含 +16 seed)
             l0BTensor[0].template ReinterpretCast<fp4x2_e2m1_t>(),          // B = P(L0B)
             mmadParams);
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

    // 稀疏 V 数据 gather：按 gSparseBlockIdx 把稀疏 Y-block 逐段 Nd2Nz 到 L1 整 tile 的对应位置。
    // 复用 CopyInA1 的 Nd2Nz 参数语义，仅 nValue 改为段长、src/dst 加段偏移。
    // NZ 布局推导：dstNzNStride=1（每行 32B），故 gather 行 r 的 dst 字节偏移 = r*32；
    // dstNzC0Stride=s2Align64 保持与整 tile 一致，使 K-block 跨段对齐。
    // （V 是 fp4，sizeof(fp4x2_e2m1_t)=1，元素偏移即字节偏移。）
    __aicore__ inline
    void SparseVBaseTileL1FullLoad(
        AscendC::GlobalTensor<fp4x2_e2m1_t> gV,         // 已带 batch+head 偏移
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
        uint32_t gatheredKvSTileIdx, int64_t kvSeqlen,
        uint32_t kvSBaseTile, uint32_t blockShapeY,
        uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
        uint32_t curBaseTileSize, uint32_t embed, uint32_t s2Align64)
    {
        uint32_t baseTileStartOffset = gatheredKvSTileIdx * kvSBaseTile;
        uint32_t baseTileEndOffset   = baseTileStartOffset + curBaseTileSize;
        uint32_t gatheredStartOffset = baseTileStartOffset;
        uint32_t gatheredYBlockIdx   = gatheredStartOffset / blockShapeY;
        uint32_t yBlockInnerStart    = gatheredStartOffset % blockShapeY;
        uint32_t oriYBlockIdx   = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
        uint32_t oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStart;
        uint32_t dealtLenAccum  = 0;

        uint16_t aColValue = CeilDivision(embed, 2);   // D/2（fp4x2 元素 = 字节）
        while (dealtLenAccum < curBaseTileSize && gatheredYBlockIdx < yBlockNumRsvd &&
               oriYBlockIdx < yBlockNumAval && oriStartOffset < kvSeqlen) {
            uint32_t curYBlockSize = (oriYBlockIdx == yBlockNumAval - 1)
                                     ? static_cast<uint32_t>(kvSeqlen - oriYBlockIdx * blockShapeY)
                                     : blockShapeY;
            uint32_t gatheredEndOffset = min(gatheredYBlockIdx * blockShapeY + curYBlockSize,
                                             baseTileEndOffset);
            uint32_t curDealtLen = gatheredEndOffset - gatheredStartOffset;
            if (curDealtLen == 0) { break; }

            AscendC::Nd2NzParams p;
            p.ndNum = 1;
            p.nValue = curDealtLen;
            p.dValue = aColValue;
            p.srcNdMatrixStride = 0;
            p.srcDValue = aColValue;          // 源行 stride = D/2 字节
            p.dstNzC0Stride = s2Align64;      // 与整 tile 一致，K-block 跨段对齐
            p.dstNzNStride = 1;               // 每行 32B
            p.dstNzMatrixStride = 0;
            // gV 是 GlobalTensor<fp4x2_e2m1_t>，其索引单位是 fp4 元素（4-bit），gV[i] 内部按 i/2 字节寻址。
            // 故行 stride 在索引单位下 = embed（每行 D 个 fp4 = D 索引 = D/2 字节）。
            uint32_t srcByteOff = oriStartOffset * embed;
            // l1ATensor 是 LocalTensor<fp4x2_e2m1_t>，索引单位同样是 fp4 元素（[i]→字节 i/2，与 gV 一致）。
            // NZ 每行 32 字节 = 64 fp4 元素，故段落到 gather 行 dealtLenAccum 的 dst 索引 = dealtLenAccum*64。
            // （单段 dealtAccum=0 时与 *32 同值，不生效；仅多段才考验。）
            uint32_t dstByteOff = dealtLenAccum * 64;             // NZ 每行 32B=64 fp4 元素
            AscendC::printf(
                "[V-GATHER] seg: oriYBlock=%u oriStart=%u curDealtLen=%u dealtAccum=%u "
                "srcByteOff=%u dstByteOff=%u nVal=%u dVal=%u c0Stride=%u nStride=1\n",
                (unsigned)oriYBlockIdx, (unsigned)oriStartOffset, (unsigned)curDealtLen,
                (unsigned)dealtLenAccum, (unsigned)srcByteOff, (unsigned)dstByteOff,
                (unsigned)curDealtLen, (unsigned)aColValue, (unsigned)s2Align64);
            AscendC::DataCopy(l1ATensor[0][dstByteOff], gV[srcByteOff], p);

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

    // 稀疏 V-scale gather：V-scale GM [BT2//64,N2,D,2]，S2 方向按 64 分组。
    // 一个 sparse Y-block（blockShapeY 行 V，须为 64 倍数）对应 blockShapeY/64 个连续 scale-group。
    // B16 视图 Nd2Nz（与 CopyInScaleA1 同语义），段内 scaleRows=ceil(curDealtLen,64)/64 组。
    // NZ 布局：dstNzNStride=1（每组 32B=16 个 half），故 gather 组 g 的 dst half 偏移 = g*16。
    __aicore__ inline
    void SparseVScaleBaseTileL1FullLoad(
        AscendC::GlobalTensor<uint8_t> gVScale,       // 已带 batch+head 偏移
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
        uint32_t gatheredKvSTileIdx, int64_t kvSeqlen,
        uint32_t kvSBaseTile, uint32_t blockShapeY,
        uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
        uint32_t curBaseTileSize, uint32_t embed, uint32_t s2Align64)
    {
        constexpr uint32_t SCALE_GROUP = 64;
        uint32_t baseTileStartOffset = gatheredKvSTileIdx * kvSBaseTile;
        uint32_t baseTileEndOffset   = baseTileStartOffset + curBaseTileSize;
        uint32_t gatheredStartOffset = baseTileStartOffset;
        uint32_t gatheredYBlockIdx   = gatheredStartOffset / blockShapeY;
        uint32_t yBlockInnerStart    = gatheredStartOffset % blockShapeY;
        uint32_t oriYBlockIdx   = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
        uint32_t oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStart;
        uint32_t dealtLenAccum  = 0;     // 已 gather 的 V 行数

        uint32_t scaleK = s2Align64 / SCALE_GROUP * 2;   // 与 CopyInScaleA1 一致
        AscendC::GlobalTensor<half> gVScaleB16;
        gVScaleB16.SetGlobalBuffer((__gm__ half*)(gVScale.GetPhyAddr()), 0x7FFFFFFFu);
        auto l1ScaleB16 = l1AScaleTensor[0].ReinterpretCast<half>();

        while (dealtLenAccum < curBaseTileSize && gatheredYBlockIdx < yBlockNumRsvd &&
               oriYBlockIdx < yBlockNumAval && oriStartOffset < kvSeqlen) {
            uint32_t curYBlockSize = (oriYBlockIdx == yBlockNumAval - 1)
                                     ? static_cast<uint32_t>(kvSeqlen - oriYBlockIdx * blockShapeY)
                                     : blockShapeY;
            uint32_t gatheredEndOffset = min(gatheredYBlockIdx * blockShapeY + curYBlockSize,
                                             baseTileEndOffset);
            uint32_t curDealtLen = gatheredEndOffset - gatheredStartOffset;
            if (curDealtLen == 0) { break; }

            uint32_t scaleRows = (curDealtLen + SCALE_GROUP - 1) / SCALE_GROUP;  // 组数向上取整
            AscendC::Nd2NzParams p;
            p.ndNum = 1;
            p.nValue = scaleRows;
            p.dValue = embed;                // D
            p.srcNdMatrixStride = 0;
            p.srcDValue = embed;             // 每组 D 个 B16 = 2D 字节
            p.dstNzC0Stride = scaleK / 2;    // = s2Align64/64，与整 tile 一致
            p.dstNzNStride = 1;              // 每组 32B
            p.dstNzMatrixStride = 0;
            uint32_t srcGroupIdx = oriStartOffset / SCALE_GROUP;
            uint32_t srcB16Off   = srcGroupIdx * embed;          // B16 单位
            uint32_t dstGroupIdx = dealtLenAccum / SCALE_GROUP;
            uint32_t dstB16Off   = dstGroupIdx * 16;             // 每组 32B = 16 个 half
            AscendC::printf(
                "[VS-GATHER] seg: oriStart=%u curDealtLen=%u dealtAccum=%u scaleRows=%u "
                "srcB16Off=%u dstB16Off=%u nVal=%u dVal=%u c0Stride=%u\n",
                (unsigned)oriStartOffset, (unsigned)curDealtLen, (unsigned)dealtLenAccum,
                (unsigned)scaleRows, (unsigned)srcB16Off, (unsigned)dstB16Off,
                (unsigned)scaleRows, (unsigned)embed, (unsigned)(scaleK / 2));
            AscendC::DataCopy(l1ScaleB16[dstB16Off], gVScaleB16[srcB16Off], p);

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

    __aicore__ inline void FixpipeMm2(uint32_t s1Align128,uint32_t embedReal,AscendC::LocalTensor<ElementC> ubOTmpTensor)
    {
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (s1Align128 + 7) >> 3 << 3;   // n = M 对齐到 8
        fixpipeParams.mSize = embedReal + 1;                       // m = D+1（D 行 Oᵀ + 1 行 rowsum）
        fixpipeParams.srcStride = embedReal + 16;                 // L0C 源行 stride = D+16
        fixpipeParams.dstStride = 64;                                    // UB 目标行 stride = 64 元素
        fixpipeParams.quantPre = QuantMode_t::NoQuant;                  // 不量化（保持 fp32，V2 反量化/归一）
        fixpipeParams.dualDstCtl = 2;                                    // 按 N 维拆分给 2 个 UB

        // uint32_t pvL0COffset = pvL0cBufId * (QK_L0C_SIZE / sizeof(COMPUTE_T));  // L0C 槽
        AscendC::Fixpipe<float, float, CFG_ROW_MAJOR_UB>(ubOTmpTensor, l0CTensor[0], fixpipeParams);
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


    // template <class TensorB, class TensorC>
    __aicore__ inline
    void operator()(
        AscendC::GlobalTensor<fp4x2_e2m1_t> gV,          // 已带 batch+head 偏移
        AscendC::GlobalTensor<uint8_t> gVScale,          // 已带 batch+head 偏移
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
        AscendC::LocalTensor<float> ubOTmpTensor,
        uint32_t rowNum, uint32_t embed, uint32_t curBaseTileSize,
        uint32_t gatheredKvSTileIdx, int64_t kvSeqlen,
        uint32_t kvSBaseTile, uint32_t blockShapeY,
        uint32_t yBlockNumAval, uint32_t yBlockNumRsvd)
    {
        AscendC::printf("MmadAtlasA5BsaPVMxfp4\n");

        uint32_t embedReal = embed;
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
        InitL0BufferForReduceSum();

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
        SparseVBaseTileL1FullLoad(
            gV, gSparseBlockIdx, gatheredKvSTileIdx, kvSeqlen, kvSBaseTile, blockShapeY,
            yBlockNumAval, yBlockNumRsvd, curBaseTileSize, embedReal, s2Align64);
        SparseVScaleBaseTileL1FullLoad(
            gVScale, gSparseBlockIdx, gatheredKvSTileIdx, kvSeqlen, kvSBaseTile, blockShapeY,
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

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
        AscendC::DumpTensor(l0CTensor[0], 123, 128);
        AscendC::PipeBarrier<PIPE_ALL>();
        FixpipeMm2(s1Align128,embedReal,ubOTmpTensor);
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

    uint32_t l1ATileM;
    uint32_t l1BTileN;
    uint32_t l1ATileK;
    uint32_t l1BTileK;
    uint32_t l1ABufNum;
    uint32_t l1BBufNum;
};

}  // namespace NpuArch::Gemm::Block

#endif  // GEMM_BLOCK_PV_ARCH35_MXFP4_HPP
