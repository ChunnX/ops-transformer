/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_reduce_scatter_fp16_bf16.h
 * \brief
 */

#ifndef MATMUL_REDUCE_SCATTER_FP16_BF16_H
#define MATMUL_REDUCE_SCATTER_FP16_BF16_H

#include "lib/hccl/hccl.h"
#include "../common_def.h"
#include "../../common/op_kernel/mc2_common_def.h"
#include "../../common/op_kernel/mc2_mat_mul_asw_kernel.h"
#include "../../common/op_kernel/mc2_mat_mul_asw_block.h"
#include "../../3rd/mat_mul_v3/op_kernel/mat_mul_v3_common.h"
#include "../../3rd/mat_mul_v3/op_kernel/arch35/mat_mul_asw_kernel.h"
#include "../../3rd/mat_mul_v3/op_kernel/arch35/mat_mul_tiling_data.h"
#include "matmul_reduce_scatter_v2_c_tiling.h"

namespace MatmulReduceScatterV2Impl {
using namespace AscendC;

template <typename AType, typename BType, typename BiasType, typename CType>
class MatmulReduceScatterFP16BF16 {
public:
    __aicore__ inline MatmulReduceScatterFP16BF16() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR contextGM,
                                GM_ADDR workspaceGM, Mc2Tiling::MatmulReduceScatterV2TilingData* tilingData, 
                                __gm__ void* mc2InitTiling, __gm__ void* mc2CcTiling, TPipe* tpipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InnerProcess();
    __aicore__ inline void Compute(GM_ADDR cGM, Mc2MatMulV3TilingData& tiling, uint32_t count,
                                   GM_ADDR gmToFloat, bool isLast, bool isTail);
    __aicore__ inline void MatMulV3Compute(GM_ADDR cGM, Mc2MatMulV3TilingData& tiling, uint32_t count,
                                           GM_ADDR gmToFloat, bool isLast, bool isTail);
    __aicore__ inline void PostProcess();
    __aicore__ inline void CastBiasBFtoFloat(GM_ADDR dst, GM_ADDR src, int size);

private:
    Mc2Tiling::MatmulReduceScatterV2TilingData* tilingData_;
    TPipe* tPipe_;
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR cGM_;
    GM_ADDR biasGM_;
    GM_ADDR gmToFloat_;
    __gm__ HcclCombinOpParam* context_;
    uint32_t rankId_;
    AscendC::HcclDataType dataType_;
    uint8_t debugMode_;
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl_;
    AscendC::HcclHandle handles_[MAX_HANDLE];
    TBuf<TPosition::VECCALC> biasCastBuf_;
};

template <typename AType, typename BType, typename BiasType, typename CType>
__aicore__ inline void MatmulReduceScatterFP16BF16<AType, BType, BiasType, CType>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR contextGM, GM_ADDR workspaceGM,
    Mc2Tiling::MatmulReduceScatterV2TilingData* tilingData, __gm__ void* mc2InitTiling, __gm__ void* mc2CcTiling, TPipe* tPipe) {
    tilingData_ = tilingData;
    auto&& cfg = tilingData_->param;
    hccl_.Init(contextGM, mc2InitTiling);
    hccl_.SetCcTiling(mc2CcTiling);
    context_ = (__gm__ HcclCombinOpParam *)(contextGM);
    tPipe_ = tPipe;
    dataType_ = static_cast<AscendC::HcclDataType>(tilingData_->dataType);
    debugMode_ = tilingData_->debugMode;
    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    biasGM_ = biasGM;
    rankId_ = context_->rankId;

    GM_ADDR biasToFloatAddr = workspaceGM;
    AscendC::printf("[LBH_DBG] Init: biasGM=%lx, workspaceGM=%lx, biasLen=%d, rankN=%d, BiasType_is_bf16=%d\n",
                    biasGM, workspaceGM, cfg.biasLen, cfg.rankN, std::is_same_v<BiasType, bfloat16_t>);
    if constexpr (std::is_same_v<BiasType, bfloat16_t>) {
        if (cfg.biasLen > 0) {
            AscendC::printf("[LBH_DBG] Before CastBias: src=%lx, dst=%lx, size=%d\n", biasGM, biasToFloatAddr, cfg.rankN);
            CastBiasBFtoFloat(biasToFloatAddr, biasGM, cfg.rankN);
            AscendC::printf("[LBH_DBG] After CastBias: biasGM_=%lx\n", biasToFloatAddr);
            biasGM_ = biasToFloatAddr;
            SyncAll<true>();
            gmToFloat_ = workspaceGM + cfg.biasLen;
        } else {
            gmToFloat_ = workspaceGM;
        }
    } else {
        gmToFloat_ = workspaceGM;
    }
    AscendC::printf("[LBH_DBG] Init done: biasGM_=%lx, gmToFloat_=%lx\n", biasGM_, gmToFloat_);
}

template <typename AType, typename BType, typename BiasType, typename CType>
__aicore__ inline void MatmulReduceScatterFP16BF16<AType, BType, BiasType, CType>::PostProcess()
{
    auto&& cfg = tilingData_->param;
    // 等待reducescatter执行完成
    if ((GetBlockIdx() == 0) && (g_coreType == AIC)) {
        for (uint32_t i = 0; i < cfg.tileCnt + cfg.tailCnt; i++) {
            hccl_.Wait(handles_[i]);
        }
        // 终止hcclserver
        hccl_.Finalize();
    }
}

template <typename AType, typename BType, typename BiasType, typename CType>
__aicore__ inline void MatmulReduceScatterFP16BF16<AType, BType, BiasType, CType>::Process()
{
    InnerProcess();
    PostProcess();
}

template <typename AType, typename BType, typename BiasType, typename CType>
__aicore__ inline void MatmulReduceScatterFP16BF16<AType, BType, BiasType, CType>::InnerProcess()
{
    auto&& tiling = tilingData_->mC2Mmv3TileTilingData.tCubeTiling;
    auto&& cfg = tilingData_->param;

    // fullmesh算法
    // 计算主块
    Compute(cGM_, tilingData_->mC2Mmv3TileTilingData, cfg.tileCnt, gmToFloat_, cfg.tailM ? false : true, false);
    // 计算尾块
    if (cfg.tailM) {
        uint64_t tileSize = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N) / cfg.rankDim;
        uint64_t tileOffset = tileSize * static_cast<uint64_t>(cfg.tileCnt) * sizeof(C_DTYPE);
        auto cGMTail = cGM_ + tileOffset;
        auto gmToFloatTail = gmToFloat_ + tileOffset;
        Compute(cGMTail, tilingData_->mC2Mmv3TailTilingData, cfg.tailCnt, gmToFloatTail, true, true);
    }
}

template <typename AType, typename BType, typename BiasType, typename CType>
__aicore__ inline void MatmulReduceScatterFP16BF16<AType, BType, BiasType, CType>::Compute(
                                                   GM_ADDR cGM, Mc2MatMulV3TilingData& tiling,
                                                   uint32_t count, GM_ADDR gmToFloat, bool isLast, bool isTail)
{
    if ASCEND_IS_AIV {
        return;
    }

    if (block_idx >= tiling.tCubeTiling.usedCoreNum) {
        for (uint32_t i = 0; i < count; i++) {
            AscendC::CrossCoreSetFlag<0, PIPE_FIX>(3);
            AscendC::CrossCoreWaitFlag(3);
        }
        return;
    }
    // matmul的一次计算流程
    MatMulV3Compute(cGM, tiling, count, gmToFloat, isLast, isTail);
}

template <typename AType, typename BType, typename BiasType, typename CType>
__aicore__ inline void MatmulReduceScatterFP16BF16<AType, BType, BiasType, CType>::MatMulV3Compute(GM_ADDR cGM,
                                                   Mc2MatMulV3TilingData& tiling, 
                                                   uint32_t count, GM_ADDR gmToFloat,
                                                   bool isLast, bool isTail)
{
    using cDataType = typename CType::T;
    using Mmv3BiasType = std::conditional_t<std::is_same_v<BiasType, bfloat16_t>, float, BiasType>;
    AscendC::printf("[LBH_DBG] MatMulV3Compute: biasGM_=%lx, Mmv3BiasType_is_float=%d, is_bf16=%d\n",
                    biasGM_, std::is_same_v<Mmv3BiasType, float>, std::is_same_v<BiasType, bfloat16_t>);
    auto&& cfg = tilingData_->param;
    cfg.rankID = rankId_;
    MC2MatmulV3::MC2MatmulAswKernelDerive<AType, BType, CType, Mmv3BiasType, MC2MatmulV3::MC2MatmulAswBlockDerive> mmv3;
    auto tempGM = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? cGM : gmToFloat_;
    mmv3.Init(aGM_, bGM_, tempGM, biasGM_, nullptr, nullptr, &tiling, GetTPipePtr(), cfg, isTail, false);
    uint64_t sliceM = static_cast<uint64_t>(tiling.tCubeTiling.M) / cfg.rankDim;
    auto recvCount = sliceM * static_cast<uint64_t>(tiling.tCubeTiling.N);
    auto cOffset = recvCount * sizeof(cDataType);
    auto cWork = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? cGM : gmToFloat;
    auto recvBuffer = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? gmToFloat : cGM;
    auto shift = isTail ? cfg.tileCnt : 0;
    uint64_t stride = static_cast<uint64_t>(cfg.rankM / cfg.rankDim) * static_cast<uint64_t>(cfg.rankN);
    uint8_t repeat = 1;
    for (uint32_t i = 0; i < count; i++) {
        mmv3.UpdateSlice(i, isTail);
        mmv3.Process(isLast && (i == (count - 1)));
        AscendC::CrossCoreSetFlag<0, PIPE_FIX>(3);
        AscendC::CrossCoreWaitFlag(3);
        recvCount = (debugMode_ == MC2_DEBUG_ONLY_CUBE) ? 1 : recvCount;
        handles_[i + shift] = hccl_.ReduceScatter<true>(
            cWork, recvBuffer, recvCount, dataType_, HcclReduceOp::HCCL_REDUCE_SUM, stride, repeat);
        cWork += cOffset;
        recvBuffer += cOffset;
    }
    mmv3.End();
}

template <typename AType, typename BType, typename BiasType, typename CType>
__aicore__ inline void MatmulReduceScatterFP16BF16<AType, BType, BiasType, CType>::CastBiasBFtoFloat(
    GM_ADDR dst, GM_ADDR src, int size)
{
    static constexpr uint16_t SYNC_AIV_DONE_FLAG = 8;
    static constexpr uint16_t SYNC_AIC_ACK_FLAG = 10;
    static constexpr uint64_t SYNC_MODE0 = 0;
    static constexpr uint64_t SYNC_MODE2 = 2;

    AscendC::printf("[LBH_DBG] CastBias: g_coreType=%d, blockIdx=%d, src=%lx, dst=%lx, size=%d\n",
                    g_coreType, GetBlockIdx(), src, dst, size);

    if (g_coreType == AIC) {
        AscendC::printf("[LBH_DBG] CastBias AIC wait\n");
        CrossCoreWaitFlag(SYNC_AIV_DONE_FLAG);
        CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE3>(SYNC_AIC_ACK_FLAG);
        CrossCoreWaitFlag(SYNC_AIC_ACK_FLAG);
        return;
    }
    if (GetBlockIdx() != 0) {
        AscendC::printf("[LBH_DBG] CastBias AIV non-zero blockIdx set flag\n");
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_DONE_FLAG);
        return;
    }

    AscendC::printf("[LBH_DBG] CastBias AIV blockIdx=0 start cast\n");
    GlobalTensor<bfloat16_t> gmSrc;
    GlobalTensor<float> gmDst;
    gmSrc.SetGlobalBuffer((__gm__ bfloat16_t*)(src), size);
    gmDst.SetGlobalBuffer((__gm__ float*)(dst), size);

    LocalTensor<bfloat16_t> bf16Local = biasCastBuf_.Get<bfloat16_t>();
    LocalTensor<float> floatLocal = bf16Local[Ceil(size, 32) * 32].template ReinterpretCast<float>();

    uint32_t cpInLen = size * sizeof(bfloat16_t);
    DataCopyExtParams cpInExtParams{1, cpInLen, 0, 0, 0};
    DataCopyPadExtParams<bfloat16_t> padExtParams{false, 0, 0, 0};
    DataCopyPad(bf16Local, gmSrc, cpInExtParams, padExtParams);
    PipeBarrier<PIPE_MTE2>();

    AscendC::DumpTensor("bias_bf16_before_cast", bf16Local, size);

    Cast(floatLocal, bf16Local, AscendC::RoundMode::CAST_NONE, size);
    PipeBarrier<PIPE_V>();

    AscendC::DumpTensor("bias_float_after_cast", floatLocal, size);

    uint32_t cpOutLen = size * sizeof(float);
    DataCopyExtParams cpOutExtParams{1, cpOutLen, 0, 0, 0};
    DataCopyPad(gmDst, floatLocal, cpOutExtParams);
    PipeBarrier<PIPE_MTE3>();

    AscendC::printf("[LBH_DBG] CastBias done, dump GM dst\n");
    AscendC::DumpTensor("bias_float_gm_dst", gmDst, size);

    CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_DONE_FLAG);
}
} // namespace MatmulReduceScatterImpl

#endif  // MATMUL_REDUCE_SCATTER_FP16_BF16_H