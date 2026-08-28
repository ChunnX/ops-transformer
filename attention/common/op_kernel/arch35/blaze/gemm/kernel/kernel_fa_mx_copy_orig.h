/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_qbmm_mix.h
 * \brief MIX template kernel: AIC cube compute + AIV dequant epilogue (Tensor API).
 *        Replicates scheduling of original QuantBmmPertokenRegbaseKernel/AL1FullLoad.
 */

#pragma once

#include "kernel_universal.h"
#if ASC_DEVKIT_MAJOR >= 9 
#include "kernel_basic_intf.h" 
#else 
#include "kernel_operator.h" 
#include "kernel_operator_intf.h" 
#endif 
#include "blaze/gemm/utils/common_utils.h" 
#include "blaze/gemm/block/block_scheduler_qbmm.h" 
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

#define QBMM_MIX_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QBMM_MIX_KERNEL_TEM_PARAMS                            \
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler,   \
        AscendC::Std::enable_if_t<                            \
            AscendC::Std::is_same_v<KernelFaTNDPAMx, typename BlockMmad::DispatchPolicy::ScheduleType>>

#define QBMM_MIX_KERNEL_CLASS_TEMPLATE_DEF_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QBMM_MIX_KERNEL_FUNC_TEMPLATE_PARAMS ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler

QBMM_MIX_KERNEL_CLASS_TEM_PARAMS
class GemmUniversal<QBMM_MIX_KERNEL_TEM_PARAMS> {
public:
    __aicore__ inline GemmUniversal()
    {}
    __aicore__ inline ~GemmUniversal()
    {}

    using QType = typename BlockMmad::QType;
    using KType = typename BlockMmad::KType;
    using VType = typename BlockMmad::VType;
    using CType = typename BlockMmad::CType;
    using LayoutQ = typename BlockMmad::LayoutQ;
    using LayoutK = typename BlockMmad::LayoutK;
    using LayoutV = typename BlockMmad::LayoutV;
    using LayoutC = typename BlockMmad::LayoutC;
    using ScaleType = AscendC::fp8_e8m0_t;
    using BlockMmadParams = typename BlockMmad::Params;
    using EpilogueParams = typename BlockEpilogue::Params;
//    using BlockSchedulerParams = typename BlockScheduler::Params;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;
    uint32_t mLoop{0};
    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        //BlockSchedulerParams schParams;
        EpilogueParams epilogueParams;
    };

    __aicore__ inline void operator()(const Params& params)
    {
        Run(params);
    }
    __aicore__ inline void Run(const Params& params);

private:
    __gm__ QType* qGM_;
    __gm__ KType* kGM_;
    __gm__ VType* vGM_;
    __gm__ ScaleType* qScaleGM_;
    __gm__ ScaleType* kScaleGM_;
    __gm__ ScaleType* vScaleGM_;
    __gm__ int32_t* blockTableGM_;
    __gm__ CType* outGM_;
};
QBMM_MIX_KERNEL_CLASS_TEMPLATE_DEF_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MIX_KERNEL_TEM_PARAMS>::Run(const Params& params)
{
    // 输入信息从tiling中获取
    uint32_t bSize = 4; // 48
    uint32_t realN2Size = 8;
    uint32_t gSize = 8;
    uint32_t s1Size = 4;
    uint32_t s2Size = 1024; // 10240;
    uint32_t t1Size = bSize * s1Size; // B*S1
    uint32_t t2Size = bSize * s2Size; // 491520; // B*S2
    uint32_t dSize = 128;
    uint32_t blockSize = 512;
    uint32_t mBaseSize = 128;
    uint32_t s2BaseSize = 512;
    uint32_t maxBlockNumPerBatch = s2Size / blockSize;
    uint32_t actualSeqlenQSize = bSize;
    uint32_t actualSeqlenKVSize = bSize;

    // SeqlenQ
    auto layoutSeqlen = AscendC::Te::MakeFrameLayout<AscendC::Te::NDLayoutPtn>(1, bSize);
    __gm__ uint64_t* seqlenQ = reinterpret_cast<__gm__ uint64_t*>(params.mmadParams.actualSeqlenQAddr);
    auto tensorSeqlenQ = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(seqlenQ), layoutSeqlen);
    // SeqlenKV
    __gm__ uint64_t* seqlenKV = reinterpret_cast<__gm__ uint64_t*>(params.mmadParams.actualSeqlenKVAddr);
    auto tensorSeqlenKV =
        AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(seqlenKV), layoutSeqlen);

    // OUT [B*S1, N2, G, D]   也可以用NDSliceLayoutPtn
    auto totalS1Size = tensorSeqlenQ[AscendC::Te::MakeCoord(0, bSize - 1)];
    t1Size = totalS1Size;
    auto layoutC = AscendC::Te::MakePatternLayout<NDSliceLayoutPtn, AscendC::Te::LayoutTraitDefault<>>(
            AscendC::Te::MakeShape(t1Size, realN2Size, AscendC::Te::MakeShape(gSize, dSize)),
            AscendC::Te::MakeStride(realN2Size * gSize * dSize, gSize * dSize, AscendC::Te::MakeStride(dSize, 1)));
    outGM_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cAddr);
    auto tensorOut = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(outGM_), layoutC);

    // TODO tensorAPI需要新增2维batch的处理
    // Q [B*S1, N2, G, D]   也可以用NDSliceLayoutPtn
    auto layoutQ = AscendC::Te::MakePatternLayout<NDSliceLayoutPtn, AscendC::Te::LayoutTraitDefault<>>(
            AscendC::Te::MakeShape(t1Size, realN2Size, AscendC::Te::MakeShape(gSize, dSize)),
            AscendC::Te::MakeStride(realN2Size * gSize * dSize, gSize * dSize, AscendC::Te::MakeStride(dSize, 1)));
    qGM_ = reinterpret_cast<__gm__ QType*>(params.mmadParams.queryAddr);
    auto tensorQ = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(qGM_), layoutQ);

    auto batchTuple = AscendC::Std::make_tuple(bSize * maxBlockNumPerBatch, realN2Size);
    // K ND [B*S2/blockSize, N2, blockSize, D]
    // K NZ [B*S2/blockSize, N2, D/D0, blockSize, D0]    TODO 需要处理不同的dtype，封装成函数，简写方式
    // 总的blockNum有多少，不知道，使用maxBlockNumPerBatch来构造shape，因为是最外轴，不影响后续的处理
    constexpr auto c0Size = AscendC::Te::C0_ELEMENT<KType>; // reduce轴的内轴
    auto layoutK = MakeZNLayout<KType>(batchTuple, dSize, blockSize);

    ListTensorDesc keyListTensorDesc((__gm__ void *)(params.mmadParams.keyAddr));
    kGM_ = (__gm__ KType*)keyListTensorDesc.GetDataPtr<__gm__ KType>(0);
    auto tensorK = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(kGM_), layoutK);

    // V [B*S2/blockSize, N2, D/D0, blockSize, D0] blockSize是reduce轴
    auto layoutV = MakeNZLayout<VType>(batchTuple, blockSize, dSize);
    ListTensorDesc valueListTensorDesc((__gm__ void *)(params.mmadParams.valueAddr));
    vGM_ = (__gm__ VType*)valueListTensorDesc.GetDataPtr<__gm__ VType>(0);
    auto tensorV = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(vGM_), layoutV);

    // ==============Scale====================
    // QScale 实际排布：[N2, B*S1, G, D//64, 2]，view表达：[B*S1, N2, G, D//64, 2] 这个需要新的Ptn
    const auto scaleDSize = Blaze::Gemm::CeilDiv(dSize, static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    auto layoutQScale = AscendC::Te::MakePatternLayout<AscendC::Te::ScaleANDLayoutPtn,
        AscendC::Te::LayoutTrait<ScaleType, AscendC::Std::Int<SCALE_C0>>>(
            AscendC::Te::MakeShape(t1Size, realN2Size, AscendC::Te::MakeShape(gSize, scaleDSize)),
            AscendC::Te::MakeStride(gSize * scaleDSize, t1Size * gSize * scaleDSize, AscendC::Te::MakeStride(scaleDSize, 1)));
    qScaleGM_ = reinterpret_cast<__gm__ ScaleType*>(params.mmadParams.queryScaleAddr);
    auto tensorQScale = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(qScaleGM_), layoutQScale);
    // KScale NZ [B*S2/blockSize, N2, blockSize/16, D/64, 16, 2]
    // KScale ND [B*S2/blockSize, N2, blockSize, D]  [b, k, n]
    auto layoutKScale = Gemm::MakeScaleNNLayout(batchTuple, scaleDSize, blockSize);
    kScaleGM_ = reinterpret_cast<__gm__ ScaleType*>(params.mmadParams.keyScaleAddr);
    auto tensorKScale = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(kScaleGM_), layoutKScale);
    // VScale  [B*S2/blockSize, N2, D/16, blockSize/64, 16, 2]  blockSize是reduce轴
    auto reduceSize = Blaze::Gemm::CeilDiv(blockSize, static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    auto layoutVScale = Gemm::MakeScaleNNLayout(batchTuple, reduceSize, dSize);
    vScaleGM_ = reinterpret_cast<__gm__ ScaleType*>(params.mmadParams.valueScaleAddr);
    auto tensorVScale = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(vScaleGM_), layoutVScale);

    // blockTable [B, maxBlockNumPerBatch]
    auto layoutTable = AscendC::Te::FrameLayoutFormat<AscendC::Te::NDLayoutPtn, AscendC::Te::_16>{}(bSize, maxBlockNumPerBatch);
    blockTableGM_ = reinterpret_cast<__gm__ int32_t*>(params.mmadParams.blockTableAddr);
    auto tensorBlockTable =
        AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(blockTableGM_), layoutTable);
    
    uint32_t bN2Cnt = bSize * realN2Size;
    uint32_t aicCoreIdx = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        aicCoreIdx = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
    }
    
    uint32_t aicCoreNum = AscendC::GetBlockNum();
    BlockMmad blockMmad;
    blockMmad.Init();
    BlockEpilogue epilogue;
    epilogue.Init(params.epilogueParams);
    for (auto bN2Idx = aicCoreIdx; bN2Idx < bN2Cnt; bN2Idx += aicCoreNum) {
        auto bIdx = bN2Idx / realN2Size;
        auto n2Idx = bN2Idx % realN2Size;
        auto bs1Offset = 0;
        auto actS1Size = tensorSeqlenQ[AscendC::Te::MakeCoord(0, bIdx)];
        if (bIdx > 0) {
            bs1Offset = tensorSeqlenQ[AscendC::Te::MakeCoord(0, bIdx - 1)]; 
            actS1Size = actS1Size - bs1Offset; 
        }
        auto actS2Size = tensorSeqlenKV[AscendC::Te::MakeCoord(0, bIdx)];
        // 对Q矩阵切出B和N2轴  [B*S1, N2, G, D] --> [S1, 1, G, D]
        auto subTensorQ = tensorQ.Slice(
            AscendC::Te::MakeCoord(bs1Offset, n2Idx, AscendC::Te::MakeCoord(0, 0)),
            AscendC::Te::MakeShape(actS1Size, 1, AscendC::Te::MakeShape(gSize, dSize)));
        auto blockTensorQ = AscendC::Te::Squeeze<1>(subTensorQ);
        
        // 对QScale矩阵切出B和N2轴，实际排布：[N2, B*S1, G, D//64, 2]，view表达：[B*S1, N2, G, D//64, 2]
        auto subTensorQScale = tensorQScale.Slice(
            AscendC::Te::MakeCoord(bs1Offset, n2Idx, AscendC::Te::MakeCoord(0, 0)),
            AscendC::Te::MakeShape(actS1Size, 1, AscendC::Te::MakeShape(gSize, scaleDSize)));
        auto blockTensorQScale = AscendC::Te::Squeeze<1>(subTensorQScale);

        // OUT进行切分，一个基本块最多需要分三次搬出，和Q矩阵的搬入一致
        auto subTensorOut = tensorOut.Slice(
            AscendC::Te::MakeCoord(bs1Offset, n2Idx, AscendC::Te::MakeCoord(0, 0)),
            AscendC::Te::MakeShape(actS1Size, 1, AscendC::Te::MakeShape(gSize, dSize)));
        auto blockTensorOut = AscendC::Te::Squeeze<1>(subTensorOut);
        
        auto s1gLoopNum = CeilDiv(actS1Size * gSize, static_cast<uint64_t>(mBaseSize));
        auto s2LoopNum = CeilDiv(actS2Size, static_cast<uint64_t>(s2BaseSize));
        for (auto s1gIdx = 0UL; s1gIdx < s1gLoopNum; ++s1gIdx) {
            auto s1gBlockSize = (s1gIdx + 1 == s1gLoopNum) ? (actS1Size * gSize - mBaseSize * s1gIdx) : mBaseSize;
            // 这里无法对Q进行slice，因为可能有多个不连续的块，无法表达，直接传入subTensorQ
            // [s1Start, gStart] -> [s1Eng, gEnd] 头、中间、尾巴
            // coord = [s1Start, gStart] shape = [1, gSize - gStart] 头
            // coord = [s1Start + 1, 0] shape = [s1End - s1Start - 1, gSize] 中间
            // coord = [s1End, 0] shape = [1, gEnd] 尾巴
            auto s1Start = s1gIdx * mBaseSize / gSize;
            auto s1End = Min(actS1Size * gSize, (s1gIdx + 1) * mBaseSize) / gSize;
            auto gStart = s1gIdx * mBaseSize % gSize;
            auto gEnd = Min(actS1Size * gSize, (s1gIdx + 1) * mBaseSize) % gSize;
            BlockCoord s1gCoord{s1Start, s1End, gStart, gEnd};
            for (auto s2Idx = 0; s2Idx < s2LoopNum; ++s2Idx) {
                auto s2BlockSize = (s2Idx + 1 == s2LoopNum) ? (actS2Size - s2BaseSize * s2Idx) : s2BaseSize;
                auto blockIdxInBatch = s2Idx * s2BaseSize / blockSize;
                auto bsIdx = s2Idx * s2BaseSize % blockSize;   // s2BaseSize == blockSize 这里一定是0，不用管
                auto blockIdx = tensorBlockTable[AscendC::Te::MakeCoord(bIdx, blockIdxInBatch)];
                // K NZ [B*S2/blockSize, N2, D/D0, blockSize/16, 16, D0]
                // 只取一个[blockSize, D]做计算，NZ格式，但是只有3维，blockSize在L0B一定是16的倍数，L1预留空间
                auto subTensorK = tensorK.Slice(
                    AscendC::Te::MakeCoord(
                        blockIdx, n2Idx,
                        AscendC::Te::MakeCoord(AscendC::Te::_0{}, 0), AscendC::Te::MakeCoord(AscendC::Te::_0{}, 0)),
                    AscendC::Te::MakeShape(
                        1, 1,
                        AscendC::Te::MakeShape(AscendC::Std::Int<c0Size>{}, dSize / c0Size),
                        AscendC::Te::MakeShape(AscendC::Te::_16{}, blockSize / AscendC::Te::_16{})));

                auto blockTensorK = AscendC::Te::Squeeze<0, 1>(subTensorK);

                // KScale NZ [B*S2/blockSize, N2, blockSize/16, D/64, 16, 2]
                auto subTensorKScale = tensorKScale.Slice(
                    AscendC::Te::MakeCoord(
                        blockIdx, n2Idx,
                        AscendC::Te::MakeCoord(AscendC::Te::_0{}, 0), AscendC::Te::MakeCoord(AscendC::Te::_0{}, 0)),
                    AscendC::Te::MakeShape(
                        1, 1,
                        AscendC::Te::MakeShape(AscendC::Std::Int<MXFP_MULTI_BASE_SIZE>{}, scaleDSize / MXFP_MULTI_BASE_SIZE),
                        AscendC::Te::MakeShape(AscendC::Te::_16{}, blockSize / AscendC::Te::_16{})));
                auto blockTensorKScale = AscendC::Te::Squeeze<0, 1>(subTensorKScale);
                // V [B*S2/blockSize, N2, D/D0, blockSize, D0] blockSize是reduce轴
                auto subTensorV = tensorV.Slice(
                    AscendC::Te::MakeCoord(
                        blockIdx, n2Idx,
                        AscendC::Te::MakeCoord(AscendC::Te::_0{}, 0), AscendC::Te::MakeCoord(AscendC::Te::_0{}, 0)),
                    AscendC::Te::MakeShape(
                        1, 1,
                        AscendC::Te::MakeShape(AscendC::Te::_16{}, blockSize / AscendC::Te::_16{}),
                        AscendC::Te::MakeShape(AscendC::Std::Int<c0Size>{}, dSize / c0Size)));
                auto blockTensorV = AscendC::Te::Squeeze<0, 1>(subTensorV);
                // VScale  [B*S2/blockSize, N2, D/16, blockSize/64, 16, 2]  blockSize是reduce轴
                auto subTensorVScale = tensorVScale.Slice(
                    AscendC::Te::MakeCoord(
                        blockIdx, n2Idx,
                        AscendC::Te::MakeCoord(AscendC::Te::_0{}, 0), AscendC::Te::MakeCoord(AscendC::Te::_0{}, 0)),
                    AscendC::Te::MakeShape(
                        1, 1,
                        AscendC::Te::MakeShape(AscendC::Std::Int<MXFP_MULTI_BASE_SIZE>{}, reduceSize / MXFP_MULTI_BASE_SIZE),
                        AscendC::Te::MakeShape(AscendC::Te::_16{}, dSize / AscendC::Te::_16{})));
                auto blockTensorVScale = AscendC::Te::Squeeze<0, 1>(subTensorVScale);

                if ASCEND_IS_AIC {
                    blockMmad(
                        blockTensorQ, blockTensorK, blockTensorV, blockTensorQScale, blockTensorKScale,
                        blockTensorVScale, s1gCoord, s1gBlockSize, s2BlockSize);
                } else {
                    // vector处理
                    BlockShape shape = {actS1Size, actS2Size, s1gBlockSize, s2BlockSize};
                    BlockCoord coord = {bIdx, n2Idx, s1gIdx, s2Idx};
                    epilogue(shape, coord, mLoop);
                }
            }
            mLoop++;
        }
    }

}

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
