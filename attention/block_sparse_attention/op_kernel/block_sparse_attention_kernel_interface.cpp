/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_sparse_attention_interface.cpp
 * \brief Block Sparse Attention Interface
 */
#include "kernel_operator.h"
#if (__CCE_AICORE__ == 220)
#include "block_sparse_attention_kernel_regular_arch32.h"
#endif
#if (__CCE_AICORE__ == 310)
#include "arch35/block_sparse_attention_kernel_arch35_regular.h"
#endif

using namespace NpuArch;

#if (__CCE_AICORE__ == 220)
namespace BlockSparse {
template <
    typename InputDtype = half,
    typename SoftmaxDtype = float,
    Epilogue::LseMode lseMode = Epilogue::LseMode::NONE,
    uint32_t QueryLayout = 0,      // 0=TND, 1=BNSD
    uint32_t KvCacheLayout = 0>    // 0=TND, 1=BNSD
__global__ __aicore__ void BlockSparseAttentionInfer(
    GM_ADDR q,
    GM_ADDR k,
    GM_ADDR v,
    GM_ADDR blockSparseMask,
    GM_ADDR mask,
    GM_ADDR blockTables,
    GM_ADDR o,
    GM_ADDR actualQseqlen,
    GM_ADDR actualKvseqlen,
    GM_ADDR blockShape,
    GM_ADDR workspace,                // S matrix (QK^T result)
    GM_ADDR lse,
    GM_ADDR tiling)
{
    using ArchTag = Arch::AtlasA2;
    using ElementQ = InputDtype;
    using LayoutQ = layout::RowMajor;
    using ElementK = InputDtype;
    using LayoutK = layout::ColumnMajor;
    using ElementV = InputDtype;
    using LayoutV = layout::RowMajor;
    using ElementS = SoftmaxDtype;
    using LayoutS = layout::RowMajor;
    using ElementP = InputDtype;
    using LayoutP = layout::RowMajor;
    using ElementO = InputDtype;
    using LayoutO = layout::RowMajor;
    using ElementLse = float;
    using LayoutLse = layout::RowMajor;
    using ElementMask = int8_t;
    using LayoutMask = layout::RowMajor;
    using ElementOTmp = SoftmaxDtype;
    using LayoutOTmp = layout::RowMajor;
    using ElementUpdate = SoftmaxDtype;
    using LayoutUpdate = layout::RowMajor;

    // Use sparse-specific dispatch policies
    using L1TileShapeQK = GemmShape<Q_TILE_CEIL, 128, 128>;
    using L0TileShapeQK = GemmShape<128, 128, 128>;
    using DispatchPolicyQK = Gemm::MmadAtlasA2SFAIQK<false, false>;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK,
                                            QType, KType, SType>;

    using L1TileShapePV = GemmShape<128, 128, 256>;
    using L0TileShapePV = GemmShape<128, 128, 128>;
    using DispatchPolicyPV = Gemm::MmadAtlasA2SFAIPV<false, false>;
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV,
                                            PType, VType, OTmpType>;

    // Epilogue policies for sparse attention
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmax<lseMode, SoftmaxDtype>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax,
        PType, SType, MaskType>;
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleO<lseMode, SoftmaxDtype>;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using LseType = Gemm::GemmType<ElementLse, LayoutLse>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO,
                                                            OType, OTmpType, OUpdateType, LseType>;

    // Kernel instantiation
    using BlockSparseAttentionKernelType = BlockSparseAttentionKernel<BlockMmadQK, BlockMmadPV,
                                                                    EpilogueOnlineSoftmax,
                                                                    EpilogueRescaleO,
                                                                    false,           // PAGED_CACHE_FLAG
                                                                    QueryLayout,     // QUERY_LAYOUT
                                                                    KvCacheLayout>;  // KV_CACHE_LAYOUT
    BlockSparseAttentionKernelParams params{q, k, v, blockSparseMask, mask, blockTables, actualQseqlen, actualKvseqlen,
                                o, lse, workspace, tiling};

    // Call block sparse attention kernel
    BlockSparseAttentionKernelType blockSparseAttenInfer;
    blockSparseAttenInfer(params);
}
}
#endif
#if (__CCE_AICORE__ == 310)

using namespace BsaKernelArch35;

template <class InDtype, class SMDtype, class REDtype, Format qFormat, Format kvFormat>
__global__ __aicore__ void BsaInferIntfRegular(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables,
    GM_ADDR o, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
    GM_ADDR blockSparseMask, GM_ADDR workspace,
    GM_ADDR tiling
) {
    using ArchTag = Arch::AtlasA5;
    using ElementSparseMask = uint8_t;
    using ElementSparseIdx = int32_t;
    using ElementSparseCount = int32_t;
    using ElementQ = InDtype;
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = SMDtype;
    using ElementP = InDtype;
    using ElementO = InDtype;
    using ElementOTmp = REDtype;
    // layout tags
    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    // S is rowMajor on UB(dst)
    using LayoutS = layout::RowMajor;
    // P is actually zN on UB(src), since there is no nd2nz in MTE1
    using LayoutPDummy = layout::zN;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    // OTmp is rowMajor on UB(dst)
    using LayoutOTmp = layout::RowMajor;
    using LayoutSparseIdx = layout::RowMajor;
    using LayoutSparseCount = layout::RowMajor;
    // block mask pre-process
    using DispatchPolicyMask2Idx = Epilogue::EpilogueBsaMask2Idx;
    using EpilogueMask2Idx = Epilogue::Block::BlockEpilogue<
        DispatchPolicyMask2Idx, ElementSparseMask, ElementSparseIdx, ElementSparseCount>;
    // 处理单个tile内Q和K的matmul
    // DN 路径下 A/B 槽角色互换：A=K(ColumnMajor)、B=Q(RowMajor)，等效 SetTensorA(K)+SetTensorB(Q, transpose=true)
    // 第一阶段按 dtype 临时限定 bf16 走 DN（fp16 仍走 ND），三个 Stage 全部落地后应改由 tilingKey 控制
    constexpr bool USE_DN = std::is_same_v<InDtype, bfloat16_t>;
    using ElementAQK = std::conditional_t<USE_DN, ElementK, ElementQ>;
    using LayoutAQK = std::conditional_t<USE_DN, LayoutK, LayoutQ>;
    using ElementBQK = std::conditional_t<USE_DN, ElementQ, ElementK>;
    using LayoutBQK = std::conditional_t<USE_DN, LayoutQ, LayoutK>;
    using L1TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyQK = Gemm::MmadAtlasA5BsaQK;
    // DN 下 fixpipe 沿 N(=S1) 把 [S2, S1] 切给两个 AIV，各得紧凑 [S2, S1/2]（Stage 2 列归约「2 行=1 寄存器」前提）；
    // ND 仍用 NO_SPLIT，由 vec 自己在 UB 上切 qS
    constexpr Gemm::Tile::CopyL0CToUBMode COPY_MODE_QK =
        USE_DN ? Gemm::Tile::CopyL0CToUBMode::SPLIT_N : Gemm::Tile::CopyL0CToUBMode::NO_SPLIT;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementAQK, LayoutAQK, ElementBQK, LayoutBQK, ElementS, LayoutS,
        void, COPY_MODE_QK>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<
        DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementAQK, ElementBQK, ElementS, void, TileCopyQK>;
   // online softmax：DN 走列归约 epilogue（沿 S2 in-lane Max/Sum，S 存 BF16、计算中升 FP32），ND 走原行归约
    using DispatchPolicyOnlineSoftmax = std::conditional_t<USE_DN,
        Epilogue::EpilogueOnlineSoftmaxBsaDn, Epilogue::EpilogueOnlineSoftmaxBsa>;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType>;
    // 处理单个tile内P和Value的matmul
    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyPV = Gemm::MmadAtlasA5BsaPV;
    // DN 下 P 在 L1 上是解折叠后的 P'[S2,S1] 自然 NZ，把 LayoutTag 由 zN 改成 nZ；
    // 这样 CopyL1ToL0A 自动选中 nZ→zN 的 ifTranspose=true 特化，一条 load 完成 P'[S2,S1]→P[S1,S2]
    using LayoutPVPDummy = std::conditional_t<USE_DN, layout::nZ, LayoutPDummy>;
    using TileCopyPV = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementP, LayoutPVPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
        void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<
        DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV, ElementOTmp, void, TileCopyPV>;
    // rescale O
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA5BsaRescaleO;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<
        ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<
        DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS, TileCopyRescaleO, Arch::PositionL0C>;

    using BsaRegularKernelArch35 = BsaRegularKernelArch35<
        EpilogueMask2Idx, BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO, qFormat, kvFormat>;
    BsaKernelParamsArch35 params{q, k, v, mask, blockTables,
        actualQseqlen, actualKvseqlen, blockSparseMask, o, workspace, tiling};
    BsaRegularKernelArch35 bsaRegularKernelArch35;
    bsaRegularKernelArch35(params);
}
#endif