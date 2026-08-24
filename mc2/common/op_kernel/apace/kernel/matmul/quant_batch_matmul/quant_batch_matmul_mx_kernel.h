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
 * \file quant_batch_matmul_mx_kernel.h
 * \brief MatmulAllToAll MX 场景 matmul 内核（纯计算，不含通信）
 */

#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"

#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

#define QBMM_MX_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockScheduler>
#define QBMM_MX_KERNEL_FUNC_TEM_PARAMS ProblemShape, BlockMmad, BlockScheduler

using namespace AscendC;
using AscendC::Te::Get;

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
class QuantBatchMatmulMxKernel {
public:
    __aicore__ inline QuantBatchMatmulMxKernel() {};
    __aicore__ inline ~QuantBatchMatmulMxKernel() {}

    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    static constexpr int64_t kCacheLineAlignMask = IsFp4<AType>() ? 0xff : 0x7f;
    static constexpr int32_t SCALE_C0 = 2;

    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<C0_SIZE>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<C0_SIZE>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<C0_SIZE>>;
    using MakeLayoutScaleA = Std::conditional_t<IsFp4<AType>(),
        Te::FrameLayoutFormat<Te::NDExtLayoutPtn, Std::Int<SCALE_C0>>,
        Te::FrameLayoutFormat<Te::NDExtLayoutPtn, Std::Int<C0_SIZE>>>;
    using MakeLayoutScaleB = Std::conditional_t<IsFp4<AType>(),
        Te::FrameLayoutFormat<Te::DNExtLayoutPtn, Std::Int<SCALE_C0>>,
        Te::FrameLayoutFormat<Te::DNExtLayoutPtn, Std::Int<C0_SIZE>>>;

    using BlockShape = Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Te::Coord<int64_t, int64_t, int64_t, int64_t>;

    // 类型提升为kernel类直接成员(与all_to_all_qbmm_mx_kernel.h一致)
    struct BlockMmadParams {
        GM_ADDR aGmAddr;
        GM_ADDR bGmAddr;
        GM_ADDR cGmAddr;
        GM_ADDR biasGmAddr = nullptr;
        GM_ADDR scaleAGmAddr;
        GM_ADDR scaleBGmAddr;
    };
    struct L1Params {
        uint64_t kL1Size;
        uint32_t scaleKL1;
        uint8_t nBufferNum;
    };
    struct BlockSchedulerParams {
        uint64_t baseM, baseN;
        uint64_t mTailTile, nTailTile;
        uint64_t mBaseTailSplitCnt, nBaseTailSplitCnt;
        uint64_t mTailMain, nTailMain;
    };
    struct QBMMTiling {
        static constexpr uint8_t BIAS_DISABLED = 0;
        static constexpr uint8_t BIAS_ENABLED = 1;
        uint32_t batchCount;
        uint32_t baseM, baseN, baseK;
        uint8_t isBias;
        uint8_t dbL0c;
        bool skipSelfBatch;
    };
    struct LocalParams {
        uint32_t rankId;
        uint32_t rankSize;
        uint64_t originalM;
        uint32_t selfRankId;
        GM_ADDR cGmSelfAddr;
        bool skipSelfBatch;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        L1Params l1Params;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
        LocalParams localParams;
    };

    __aicore__ inline void InitMmad(const Params &params)
    {
        isBias_ = (params.qbmmParams.isBias != QBMMTiling::BIAS_DISABLED);
        isAtomicAdd_ = false;
        mmadOp_.Init(params.problemShape,
                     Te::MakeShape(static_cast<int64_t>(params.schParams.baseM), static_cast<int64_t>(params.schParams.baseN),
                                   Te::Get<MNK_K>(params.problemShape), C0_SIZE),
                     params.l1Params, isBias_, params.qbmmParams.dbL0c > 1,
                     params.localParams.rankSize);
    }

    __aicore__ inline void operator()(const Params &params) const
    {
        auto rankId = params.localParams.rankId;
        auto rankSize = params.localParams.rankSize;
        auto oriM = params.localParams.originalM;
        bool localFirst = false;
        bool deferredSync = false;

        auto layoutA = MakeLayoutA{}(rankSize * params.localParams.originalM, Te::Get<MNK_K>(params.problemShape));
        auto layoutB = MakeLayoutB{}(rankSize * Te::Get<MNK_K>(params.problemShape), Te::Get<MNK_N>(params.problemShape));

        auto scaleKLen =
            Blaze::Gemm::CeilDiv(static_cast<int64_t>(Te::Get<MNK_K>(params.problemShape)),
                                 static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        auto layoutScaleA = MakeLayoutScaleA{}(rankSize * oriM, scaleKLen);
        auto layoutScaleB = MakeLayoutScaleB{}(rankSize * scaleKLen, Te::Get<MNK_N>(params.problemShape));
        auto layoutBias = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, Te::Get<MNK_N>(params.problemShape));
        auto layoutC = MakeLayoutC{}(Te::Get<MNK_M>(params.problemShape), Te::Get<MNK_N>(params.problemShape));

        __gm__ AType *aGmAddr_ = reinterpret_cast<__gm__ AType *>(params.mmadParams.aGmAddr);
        __gm__ BType *bGmAddr_ = reinterpret_cast<__gm__ BType *>(params.mmadParams.bGmAddr);
        __gm__ CType *cGmAddr_ = reinterpret_cast<__gm__ CType *>(params.mmadParams.cGmAddr);
        __gm__ BiasType *biasGmAddr_ = nullptr;
        __gm__ ::fp8_e8m0_t *scaleAGmAddr_ = reinterpret_cast<__gm__ ::fp8_e8m0_t *>(params.mmadParams.scaleAGmAddr);
        __gm__ ::fp8_e8m0_t *scaleBGmAddr_ = reinterpret_cast<__gm__ ::fp8_e8m0_t *>(params.mmadParams.scaleBGmAddr);
        if (isBias_) {
            biasGmAddr_ = reinterpret_cast<__gm__ BiasType *>(params.mmadParams.biasGmAddr);
        }

        auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(aGmAddr_), layoutA);
        auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(bGmAddr_), layoutB);
        auto gmScaleA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(scaleAGmAddr_), layoutScaleA);
        auto gmScaleB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(scaleBGmAddr_), layoutScaleB);
        auto gmBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(biasGmAddr_), layoutBias);
        auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(cGmAddr_), layoutC);

        BlockScheduler bs;
        bs.Init(params.schParams.baseM, params.schParams.baseN,
                Te::Get<MNK_M>(params.problemShape), Te::Get<MNK_N>(params.problemShape),
                params.schParams.mTailTile, params.schParams.nTailTile,
                params.schParams.mBaseTailSplitCnt, params.schParams.nBaseTailSplitCnt,
                params.schParams.mTailMain, params.schParams.nTailMain,
                static_cast<uint32_t>(GetBlockIdx()));

        BlockCoord blockIdx;
        int64_t mPos = 0L;
        int64_t nPos = 0L;
        constexpr int64_t kPos = 0L;

        while (bs.GetTileIdx(blockIdx)) {
            BlockShape singleShape =
                bs.template GetBlockShape<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE,
                                          BlockMmad::WEIGHT_NZ>(blockIdx);
            if ((Te::Get<IDX_M_TILEIDX>(singleShape) <= 0) || (Te::Get<IDX_N_TILEIDX>(singleShape) <= 0)) {
                return;
            }
            bs.GetTileCoord(blockIdx, mPos, nPos);

            auto gmBlockC =
                gmC.Slice(AscendC::Te::MakeCoord(mPos, nPos),
                          AscendC::Te::MakeShape(Te::Get<MNK_M>(singleShape), Te::Get<MNK_N>(singleShape)));
            auto gmBlockBias =
                gmBias.Slice(Te::MakeCoord(0L, nPos), Te::MakeShape(1L, Te::Get<IDX_N_TILEIDX>(singleShape)));

            for (uint64_t rank = 0; rank < rankSize; rank++) {
                auto actualMPos = rank * oriM + mPos;
                auto gmBlockA =
                    gmA.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                              AscendC::Te::MakeShape(Te::Get<MNK_M>(singleShape), Te::Get<MNK_K>(params.problemShape)));
                auto gmBlockScaleA = gmScaleA.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                                                     AscendC::Te::MakeShape(Te::Get<MNK_M>(singleShape), scaleKLen));
                auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(rank * Te::Get<MNK_K>(params.problemShape), nPos),
                                          AscendC::Te::MakeShape(Te::Get<MNK_K>(params.problemShape),
                                                                  Te::Get<MNK_N>(singleShape)));
                auto gmBlockScaleB = gmScaleB.Slice(AscendC::Te::MakeCoord(rank * scaleKLen, nPos),
                                                     AscendC::Te::MakeShape(scaleKLen, Te::Get<MNK_N>(singleShape)));

                bool isSelfRank = (rank == params.localParams.selfRankId);
                bool skipThis = isSelfRank && params.qbmmParams.skipSelfBatch;
                if (skipThis) {
                    continue;
                }

                if (isSelfRank && (params.localParams.cGmSelfAddr != nullptr) && !params.qbmmParams.skipSelfBatch) {
                    __gm__ CType *selfCAddr = reinterpret_cast<__gm__ CType *>(params.localParams.cGmSelfAddr);
                    auto selfLayoutC = MakeLayoutC{}(Te::Get<MNK_M>(singleShape), Te::Get<MNK_N>(singleShape));
                    auto gmSelfC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(selfCAddr + mPos * Te::Get<MNK_N>(params.problemShape)), selfLayoutC);
                    auto gmBlockSelfC = gmSelfC.Slice(Te::MakeCoord(0L, nPos),
                                                       Te::MakeShape(Te::Get<MNK_M>(singleShape), Te::Get<MNK_N>(singleShape)));
                    mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias,
                            gmBlockSelfC, singleShape, static_cast<int32_t>(rank));
                } else {
                    mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias,
                            gmBlockC, singleShape, static_cast<int32_t>(rank));
                }
            }
        }
    }

private:
    mutable BlockMmad mmadOp_;
    bool isBias_{false};
    bool isAtomicAdd_{false};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
