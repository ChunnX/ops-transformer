/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// matmul member definitions for AlltoAllMatmul.
#pragma once

#include "allto_all_matmul.h"

namespace Mc2Kernel {

template <TemplateA2AMMClass>
template <uint32_t TileM, uint32_t TileN>
struct AlltoAllMatmul<TemplateA2AMMFunc>::CatlassMatmulTypes {
    using ArchTag = Arch::AtlasA2;

    static constexpr bool ENABLE_UNIT_FLAG = false;
    static constexpr bool ENABLE_SHUFFLE_K = false;
    static constexpr bool aicCalBias =
        (QuantType == MC2_NON_QUANT) && hasBias; // 计算量化后的矩阵乘，bias不由CatlassMatmul负责

    using ElementA = BType; // 非量化场景、量化场景，A、B的入参类型一致；伪量化场景，A需要动态量化成BType
    using ElementB = BType;
    using ElementC = std::conditional_t<QuantType != MC2_NON_QUANT, int32_t,
                                        CType>; // 非量化场景，Btype和CType一致；量化场景计算结果为int32_t
    using ElementBias = BiasType;
    using LayoutA = layout::RowMajor;
    // B转置
    using LayoutB = std::conditional_t<transB, layout::ColumnMajor, layout::RowMajor>;
    using LayoutC = layout::RowMajor;
    using LayoutBias = layout::VectorLayout;

    using DispatchPolicy = std::conditional_t<aicCalBias, Gemm::MmadAtlasA2PingpongBias<ENABLE_UNIT_FLAG>,
                                              Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG, ENABLE_SHUFFLE_K>>;

    using AType_ = Gemm::GemmType<ElementA, LayoutA>;
    using BType_ = Gemm::GemmType<ElementB, LayoutB>;
    using CType_ = Gemm::GemmType<ElementC, LayoutC>;
    using BiasType_ = std::conditional_t<aicCalBias, Gemm::GemmType<ElementBias, LayoutBias>, void>;

    struct TileCopyOpt : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, BiasType_> {
        using Base = Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, BiasType_>;
        using ElementA = typename Base::ElementA;
        using ElementB = typename Base::ElementB;
        using ElementAccumulator = typename Base::ElementAccumulator;

        using CopyGmToL1A = typename Base::CopyGmToL1A;
        using CopyGmToL1B = typename Base::CopyGmToL1B;

        using CopyL1ToL0A = typename Base::CopyL1ToL0A;
        using CopyL1ToL0B = typename Base::CopyL1ToL0B;

        using CopyL0CToGm = typename Base::CopyL0CToGm;
        using BiasTypeSelector = typename Base::BiasTypeSelector;
        using CopyGmToL1Bias = typename Base::CopyGmToL1Bias;
        using CopyL1ToBT = typename Base::CopyL1ToBT;
    };

    using TileCopy = TileCopyOpt;

    using BlockScheduler30 = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    static constexpr uint32_t L1TileShapeK =
        std::is_same<BType, int4b_t>::value ? 1024 :
        std::is_same<BType, int8_t>::value  ? 512 :
                                              256; // 不同的matmul数据类型对应的L1TileShape不同
    static constexpr uint32_t L0TileShapeK =
        std::is_same<BType, int4b_t>::value ? 256 :
        std::is_same<BType, int8_t>::value  ? 128 :
                                              64; // 不同的matmul数据类型对应的L0TileShape不同

    using L1TileShape = GemmShape<TileM, TileN, L1TileShapeK>;
    using L0TileShape = GemmShape<TileM, TileN, L0TileShapeK>;
    using BlockMmadOpt =
        Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType_, BType_, CType_, BiasType_, TileCopy>;
    using MatmulKernel =
        Gemm::Kernel::AlltoAllMatmulKernel<void, void, BlockMmadOpt, void, BlockScheduler30, aicCalBias>;
};

template <TemplateA2AMMClass>
template <uint32_t TileM, uint32_t TileN>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::CatlassMatmulImpl()
{
    using Types = CatlassMatmulTypes<TileM, TileN>;
    using LayoutA = typename Types::LayoutA;
    using LayoutB = typename Types::LayoutB;
    using LayoutC = typename Types::LayoutC;
    using LayoutBias = typename Types::LayoutBias;
    using MatmulKernel = typename Types::MatmulKernel;

    uint32_t realM = m / rankSize;
    uint32_t realK = k * rankSize;
    LayoutA layoutA{static_cast<uint32_t>(realM), static_cast<uint32_t>(realK)};
    LayoutB layoutB{static_cast<uint32_t>(realK), static_cast<uint32_t>(n)};
    LayoutC layoutC{static_cast<uint32_t>(realM), static_cast<uint32_t>(n)};
    LayoutBias layoutBias{static_cast<uint32_t>(n)};

    GemmCoord processSize{static_cast<uint32_t>(realM), static_cast<uint32_t>(n), static_cast<uint32_t>(realK)};

    GM_ADDR srcGM = (QuantType == MC2_DYNAMIC_QUANT) ?
                        reinterpret_cast<GM_ADDR>(quantAGM_) :
                        reinterpret_cast<GM_ADDR>(gmPeerMem_); // 动态量化时，需要更改左矩阵读取位置
    GM_ADDR matmulResultGM = (QuantType == MC2_NON_QUANT) ?
                                 cGM_ :
                                 reinterpret_cast<GM_ADDR>(dequantCGM_); // 量化矩阵乘法时，需要修改c矩阵存放地址
    MatmulKernel matmul_op;
    typename MatmulKernel::Params params{processSize,
                                         reinterpret_cast<GM_ADDR>(srcGM),
                                         layoutA,
                                         reinterpret_cast<GM_ADDR>(bGM_),
                                         layoutB,
                                         reinterpret_cast<GM_ADDR>(biasGM_),
                                         reinterpret_cast<GM_ADDR>(matmulResultGM),
                                         layoutC,
                                         pValue,
                                         3,
                                         0,
                                         static_cast<int32_t>(rankSize),
                                         MAX_BLOCK_COUNT};
    matmul_op(params);
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::CatlassMatmul()
{
    if ASCEND_IS_AIC {
        if (m0 == 128) {
            CatlassMatmulImpl<128, 256>();
        } else {
            CatlassMatmulImpl<256, 128>();
        }
    }
}

template <TemplateA2AMMClass>
template <typename EpilogueTileShape>
struct AlltoAllMatmul<TemplateA2AMMFunc>::DequantTypes {
    using ArchTag = Arch::AtlasA2;

    using ElementC = int32_t;
    using ElementBias = BiasType;
    using LayoutD = layout::RowMajor;

    using CType_ = Gemm::GemmType<ElementC, layout::RowMajor>;

    static constexpr uint32_t ubStages = 2;
    using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
    using ScaleGType = Gemm::GemmType<ScaleType, layout::VectorLayout>;
    using PerTokenScaleGType = Gemm::GemmType<float, layout::VectorLayout>; // 全量化时，也限定为float
    using BiasGType = Gemm::GemmType<BiasType, layout::VectorLayout>;
    using DType = Gemm::GemmType<CType, layout::RowMajor>;

    using LayoutScale = layout::VectorLayout;
    using LayoutPerTokenScale = layout::VectorLayout;
    using ElementD = CType;

    using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using RowBroadcastAddType = Gemm::GemmType<float, layout::RowMajor>;
    using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;

    struct TileCopyDequant
        : public Catlass::Epilogue::Tile::TileCopy<ArchTag, CType_, ScaleGType, PerTokenScaleGType, DType> {
        using Base = Catlass::Epilogue::Tile::TileCopy<ArchTag, CType_, ScaleGType, PerTokenScaleGType, DType>;
        using ElementC = typename Base::ElementC;
        using ElementScale = typename Base::ElementX;
        using ElementPerTokenScale = typename Base::ElementY;
        using ElementBias = typename BiasGType::Element;
        using ElementD = typename Base::ElementD;

        using CopyGmToUbC = typename Base::CopyGmToUbC;
        using CopyGmToUbScale = typename Base::CopyGmToUbX;
        using CopyGmToUbPerTokenScale = typename Base::CopyGmToUbY;
        using CopyGmToUbBias = Catlass::Epilogue::Tile::CopyGm2Ub<ArchTag, BiasGType>;
        using CopyUbToGmD = typename Base::CopyUbToGmD;
    };

    using EpilogueTileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
    using TileRowBroadcastAdd = Epilogue::Tile::TileRowBroadcastAdd<ArchTag, RowBroadcastAddType, EpilogueTileShape>;
    using TileBroadcastOneBlk =
        Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul =
        Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag, OneBlkColumnBroadcastMulType, EpilogueTileShape>;
    using QuantBlockEpilogue =
        Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType_, ScaleGType, PerTokenScaleGType, BiasGType, DType,
                                       TileRowBroadcastMul, TileRowBroadcastAdd, TileBroadcastOneBlk,
                                       TileOneBlkColumnBroadcastMul, TileCopyDequant, EpilogueTileScheduler>;

    using EpilogueParams = typename QuantBlockEpilogue::Params;
};

template <TemplateA2AMMClass>
__aicore__ inline uint32_t AlltoAllMatmul<TemplateA2AMMFunc>::GetDequantRows(uint32_t totalRows, uint32_t &stRowPerCore)
{
    uint32_t rowsPerCore = DivCeil(totalRows, blockNum);
    uint32_t rowsThisCore = rowsPerCore;
    stRowPerCore = aicIdx * rowsPerCore;
    if (stRowPerCore < totalRows) {
        if (rowsThisCore + stRowPerCore > totalRows) {
            rowsThisCore = totalRows - stRowPerCore;
        }
    } else {
        rowsThisCore = 0;
    }
    return rowsThisCore;
}

template <TemplateA2AMMClass>
template <typename EpilogueTileShape>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::DequantImpl()
{
    using Types = DequantTypes<EpilogueTileShape>;
    using LayoutD = typename Types::LayoutD;
    using ElementD = typename Types::ElementD;
    using ElementC = typename Types::ElementC;
    using ElementBias = typename Types::ElementBias;
    using QuantBlockEpilogue = typename Types::QuantBlockEpilogue;
    using EpilogueParams = typename Types::EpilogueParams;

    uint32_t realM = m / rankSize;
    uint32_t realK = k * rankSize;
    LayoutD layoutD{static_cast<uint32_t>(realM), static_cast<uint32_t>(n)};

    layout::VectorLayout layoutScale{static_cast<uint32_t>(n)};
    layout::VectorLayout layoutPerTokenScale{static_cast<uint32_t>(realM)};
    layout::VectorLayout layoutBias{static_cast<uint32_t>(n)};

    GemmCoord problemShape{static_cast<uint32_t>(realM), static_cast<uint32_t>(n), static_cast<uint32_t>(realK)};

    AscendC::GlobalTensor<ElementD> gmD;
    gmD.SetGlobalBuffer((__gm__ ElementD *)cGM_);
    AscendC::GlobalTensor<ElementC> gmC;
    gmC.SetGlobalBuffer((__gm__ ElementC *)dequantCGM_);

    AscendC::GlobalTensor<ElementBias> gmBias;
    if (biasGM_ != nullptr) {
        gmBias.SetGlobalBuffer((__gm__ ElementBias *)biasGM_);
    }

    uint32_t stRowPerCore;
    uint32_t rowsThisCore = GetDequantRows(problemShape.m(), stRowPerCore);
    MatrixCoord coreOffset(stRowPerCore, 0u);
    auto layoutC = layout::RowMajor{problemShape.m(), n};
    int64_t gmOffsetC = layoutC.GetOffset(coreOffset);
    GemmCoord actualBlockShape{rowsThisCore, n, 1};

    GM_ADDR x1ScaleGM =
        (QuantType == MC2_DYNAMIC_QUANT) ? quantScaleGM_ : x1ScaleGM_; // 根据是否是全量化，决定从哪里获得pertokenScale

    QuantBlockEpilogue blockEpilogue(resource);

    EpilogueParams epilogueParams{
        scaleGM_, layoutScale, x1ScaleGM, layoutPerTokenScale.GetTileLayout(problemShape.template GetCoordByAxis<0>()),
        biasGM_,  layoutBias};
    blockEpilogue.UpdateParams(epilogueParams);

    blockEpilogue(coreOffset, actualBlockShape, gmC[gmOffsetC], layoutC, gmD[gmOffsetC], layoutC);
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::Dequant()
{
    if (m0 == 128) {
        DequantImpl<MatrixShape<32, 256>>();
    } else {
        DequantImpl<MatrixShape<64, 128>>();
    }
}

} // namespace Mc2Kernel
