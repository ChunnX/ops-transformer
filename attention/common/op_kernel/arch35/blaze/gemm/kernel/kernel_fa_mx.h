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
 * \file kernel_fa_mx.h
 * \brief FA-MX kernel：Blaze 写法 + FIA 框架
 *        写法对齐 kernel_fa_mx_backup.h —— 局部形状变量、GM tensor 全量创建（MakeTensor +
 *        MakePatternLayout + ListTensorDesc）、blockMmad/epilogue 局部对象、全部内联 Run()。
 *        框架对齐 fia_kernel_fullquant_mx.h —— while 软件流水（dispatch/execute 分离、
 *        PRELOAD_N=2、3 槽环形缓冲）、IterateBmm1/2 + ProcessVec1/2 分步接口、
 *        set_intra_block/wait_intra_block 跨核同步、prefree/postalloc。
 *        原始 for 循环版本备份见 kernel_fa_mx_backup.h。
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
    __aicore__ inline GemmUniversal() {}
    __aicore__ inline ~GemmUniversal() {}

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
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;

    // FIA 框架：软件流水深度
    static constexpr uint32_t PRELOAD_N = 2;

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        EpilogueParams epilogueParams;
    };

    __aicore__ inline void operator()(const Params& params) { Run(params); }
    __aicore__ inline void Run(const Params& params);

private:
    // GM 指针成员（Blaze 写法，对齐 backup）
    __gm__ QType* qGM_;
    __gm__ KType* kGM_;
    __gm__ VType* vGM_;
    __gm__ ScaleType* qScaleGM_;
    __gm__ ScaleType* kScaleGM_;
    __gm__ ScaleType* vScaleGM_;
    __gm__ int32_t* blockTableGM_;
    __gm__ CType* outGM_;
    uint32_t mLoop{0};
};

QBMM_MIX_KERNEL_CLASS_TEMPLATE_DEF_PARAMS
__aicore__ inline void GemmUniversal<QBMM_MIX_KERNEL_TEM_PARAMS>::Run(const Params& params)
{
    // ============== 形状参数（局部变量，从 tiling 获取）==============
    uint32_t bSize = 1;                      //修改
    uint32_t realN2Size = 8;
    uint32_t gSize = 8;
    uint32_t s1Size = 4;
    uint32_t s2Size = 1024;                  //修改
    uint32_t t1Size = bSize * s1Size;        // B*S1，后续从 seqlen 覆盖
    uint32_t dSize = 128;
    uint32_t blockSize = 512;
    uint32_t mBaseSize = 128;
    uint32_t s2BaseSize = 512;
    uint32_t maxBlockNumPerBatch = s2Size / blockSize;
    uint32_t actMSizeAlign32 = 32;

    // ============== Seqlen GM tensor（Blaze tensor API）==============
    auto layoutSeqlen = AscendC::Te::MakeFrameLayout<AscendC::Te::NDLayoutPtn>(1, bSize);
    __gm__ uint64_t* seqlenQ =
        reinterpret_cast<__gm__ uint64_t*>(params.mmadParams.actualSeqlenQAddr);
    auto tensorSeqlenQ = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(seqlenQ), layoutSeqlen);
    __gm__ uint64_t* seqlenKV =
        reinterpret_cast<__gm__ uint64_t*>(params.mmadParams.actualSeqlenKVAddr);
    auto tensorSeqlenKV = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(seqlenKV), layoutSeqlen);

    // 从 seqlen 获取总 S1 大小（用于 Q/Out layout）
    auto totalS1Size = tensorSeqlenQ[AscendC::Te::MakeCoord(0, bSize - 1)];
    t1Size = totalS1Size;

    // ============== GM tensor 创建（Blaze tensor API，对齐 backup 写法）==============
    // OUT [B*S1, N2, G, D]
    auto layoutC = AscendC::Te::MakePatternLayout<NDSliceLayoutPtn, AscendC::Te::LayoutTraitDefault<>>(
            AscendC::Te::MakeShape(t1Size, realN2Size, AscendC::Te::MakeShape(gSize, dSize)),
            AscendC::Te::MakeStride(realN2Size * gSize * dSize, gSize * dSize, AscendC::Te::MakeStride(dSize, 1)));
    outGM_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cAddr);
    auto tensorOut = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(outGM_), layoutC);

    // Q [B*S1, N2, G, D]
    auto layoutQ = AscendC::Te::MakePatternLayout<NDSliceLayoutPtn, AscendC::Te::LayoutTraitDefault<>>(
            AscendC::Te::MakeShape(t1Size, realN2Size, AscendC::Te::MakeShape(gSize, dSize)),
            AscendC::Te::MakeStride(realN2Size * gSize * dSize, gSize * dSize, AscendC::Te::MakeStride(dSize, 1)));
    qGM_ = reinterpret_cast<__gm__ QType*>(params.mmadParams.queryAddr);
    auto tensorQ = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(qGM_), layoutQ);

    // K NZ [B*S2/blockSize, N2, D/D0, blockSize, D0]
    auto batchTuple = AscendC::Std::make_tuple(bSize * maxBlockNumPerBatch, realN2Size);
    auto layoutK = MakeZNLayout<KType>(batchTuple, dSize, blockSize);
    ListTensorDesc keyListTensorDesc((__gm__ void*)(params.mmadParams.keyAddr));
    kGM_ = (__gm__ KType*)keyListTensorDesc.GetDataPtr<__gm__ KType>(0);
    auto tensorK = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(kGM_), layoutK);

    // V [B*S2/blockSize, N2, D/D0, blockSize, D0]
    auto layoutV = MakeNZLayout<VType>(batchTuple, blockSize, dSize);
    ListTensorDesc valueListTensorDesc((__gm__ void*)(params.mmadParams.valueAddr));
    vGM_ = (__gm__ VType*)valueListTensorDesc.GetDataPtr<__gm__ VType>(0);
    auto tensorV = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(vGM_), layoutV);

    // ============== Scale GM tensor ==============
    // QScale 实际排布：[N2, B*S1, G, D//64, 2]，view 表达：[B*S1, N2, G, D//64, 2]
    const auto scaleDSize = Blaze::Gemm::CeilDiv(dSize, static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    auto layoutQScale = AscendC::Te::MakePatternLayout<AscendC::Te::ScaleANDLayoutPtn,
        AscendC::Te::LayoutTrait<ScaleType, AscendC::Std::Int<SCALE_C0>>>(
            AscendC::Te::MakeShape(t1Size, realN2Size, AscendC::Te::MakeShape(gSize, scaleDSize)),
            AscendC::Te::MakeStride(gSize * scaleDSize, t1Size * gSize * scaleDSize, AscendC::Te::MakeStride(scaleDSize, 1)));
    qScaleGM_ = reinterpret_cast<__gm__ ScaleType*>(params.mmadParams.queryScaleAddr);
    auto tensorQScale = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(qScaleGM_), layoutQScale);
    // KScale NN [B*S2/blockSize, N2, blockSize, D]
    auto layoutKScale = Gemm::MakeScaleNNLayout(batchTuple, scaleDSize, blockSize);
    kScaleGM_ = reinterpret_cast<__gm__ ScaleType*>(params.mmadParams.keyScaleAddr);
    auto tensorKScale = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(kScaleGM_), layoutKScale);
    // VScale [B*S2/blockSize, N2, D/16, blockSize/64, 16, 2]
    auto reduceSize = Blaze::Gemm::CeilDiv(blockSize, static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    auto layoutVScale = Gemm::MakeScaleNNLayout(batchTuple, reduceSize, dSize);
    vScaleGM_ = reinterpret_cast<__gm__ ScaleType*>(params.mmadParams.valueScaleAddr);
    auto tensorVScale = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(vScaleGM_), layoutVScale);

    // blockTable [B, maxBlockNumPerBatch]
    auto layoutTable =
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NDLayoutPtn, AscendC::Te::_16>{}(
            bSize, maxBlockNumPerBatch);
    blockTableGM_ = reinterpret_cast<__gm__ int32_t*>(params.mmadParams.blockTableAddr);
    auto tensorBlockTable = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(blockTableGM_), layoutTable);

    // ============== 核索引 / bN2 分配（FIA 框架：每核 1 个 bN2）==============
    uint32_t bN2Cnt = bSize * realN2Size;
    uint32_t aicCoreIdx = AscendC::GetBlockIdx();
    uint8_t subBlockIdx = 0;
    if ASCEND_IS_AIV {
        aicCoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        subBlockIdx = AscendC::GetSubBlockIdx();
    }
    uint32_t bN2Start = aicCoreIdx;          //修改
    uint32_t bN2End = aicCoreIdx + 1;
    if (bN2Start >= bN2Cnt) {
        return;
    }

    // 读取当前 bN2 对应 batch 的 seqlen（Blaze tensor 索引）
    uint32_t bIdx = bN2Start / realN2Size;
    auto actS1Size = tensorSeqlenQ[AscendC::Te::MakeCoord(0, bIdx)];
    if (bIdx > 0) {
        auto bs1Offset = tensorSeqlenQ[AscendC::Te::MakeCoord(0, bIdx - 1)];
        actS1Size = actS1Size - bs1Offset;
    }
    auto actS2Size = tensorSeqlenKV[AscendC::Te::MakeCoord(0, bIdx)];

    uint32_t s2LoopTimes = CeilDiv(actS2Size, s2BaseSize);
    uint32_t gS1LoopTimes = CeilDiv(static_cast<uint64_t>(actS1Size) * gSize,
                                     static_cast<uint64_t>(mBaseSize));

    // ============== 创建 block 对象（Blaze 写法：局部变量）==============
    BlockMmad blockMmad;
    blockMmad.Init(params.mmadParams);
    BlockEpilogue epilogue;
    epilogue.Init(params.epilogueParams);

    // ============== FIA 框架：AIC↔AIV 共享缓冲分配 ==============
    // p1(L1): P + PScale，3 槽环形缓冲（AIC/AIV 共享同一 L1 地址）
    constexpr uint32_t mm2LeftSize = 128 * 512 + 128 * 512 / 32;
    __cbuf__ void* p1[3];
    p1[0] = (__cbuf__ void*)get_imm(0);
    p1[1] = (__cbuf__ void*)get_imm(mm2LeftSize);
    p1[2] = (__cbuf__ void*)get_imm(mm2LeftSize * 2);

    // mm1Res/mm2Res(UB): MM1/MM2 结果（AIC 写 → AIV 读）
    constexpr uint32_t mm1ResultSize = 128 / 2 * 512 * 4 / 2;
    __ubuf__ void* mm1Res[2];
    mm1Res[0] = (__ubuf__ void*)get_imm(0);
    mm1Res[1] = (__ubuf__ void*)get_imm(mm1ResultSize);
    __ubuf__ void* mm2Res = (__ubuf__ void*)get_imm(mm1ResultSize * 2);

    // ============== FIA 框架：InitBuffers + prefree_l0c + CrossCoreBufferInit ==============
    if ASCEND_IS_AIC {
        blockMmad.InitBuffers();
        // prefree_l0c: 预释放 L0C 标志，供 IterateBmm1 首轮 wait_flag(PIPE_FIX, PIPE_M) 使用
        set_flag(PIPE_FIX, PIPE_M, 0);
        set_flag(PIPE_FIX, PIPE_M, 1);
    } else {
        epilogue.InitBuffers();
        epilogue.AllocEventID();
    }

    int bmm1_id = 0;
    int bmm2_id = 4;

    // CrossCoreBufferInit: AIV 预置跨核标志，避免 AIC 首轮 wait_intra_block 死锁
    if ASCEND_IS_AIV {
        set_intra_block(PIPE_V, bmm2_id + 1);
        set_intra_block(PIPE_V, bmm2_id + 5);
        set_intra_block(PIPE_V, bmm2_id + 3);
    }

    // ============== FIA 框架：while 软件流水主循环 ==============
    //   dispatch:  按 s2 → gS1 → bN2 顺序登记任务到 3 槽环形缓冲
    //   execute:   当前槽执行 MM1/vec1（2 subLoop），PRELOAD_N 拍前的槽执行 MM2/vec2
    uint32_t bN2Cur = bN2Start;
    uint32_t gS1Cur = 0;
    uint32_t s2Cur = 0;
    uint32_t createdTaskCount = 0;
    uint32_t executedTaskCount = 0;
    mLoop = 0;

    uint32_t bIdxp[3], n2Idxp[3], gS1Idxp[3], s2Idxp[3];
    uint32_t loopIdxp[3], mloopIdxp[3], vecMbaseIdxp[3];
    uint32_t curS2Loop[3];
    bool runInfo[3] = {false, false, false};

    bool shouldDispatchTask = true;
    bool shouldExecuteTask = false;

    while (shouldDispatchTask || shouldExecuteTask) {
        // ---- dispatch：登记任务到环形缓冲 ----
        shouldDispatchTask = (bN2Cur != bN2End) || (gS1Cur != 0) || (s2Cur != 0);
        if (shouldDispatchTask) {
            if (s2Cur == 0) {
                mLoop++;
            }
            uint32_t slot = createdTaskCount % 3;
            bIdxp[slot] = bN2Cur / realN2Size;
            n2Idxp[slot] = bN2Cur % realN2Size;
            gS1Idxp[slot] = gS1Cur * mBaseSize;
            s2Idxp[slot] = s2Cur * s2BaseSize;
            loopIdxp[slot] = createdTaskCount;
            mloopIdxp[slot] = mLoop;
            curS2Loop[slot] = s2Cur;
            runInfo[slot] = true;
            vecMbaseIdxp[slot] = 0;
            if (subBlockIdx == 1) {
                vecMbaseIdxp[slot] = actMSizeAlign32 >> 1;
            }
            createdTaskCount++;

            // UpdateAxisInfo: s2（最内层）→ gS1 → bN2
            if (s2Cur + 1 < s2LoopTimes) {
                s2Cur++;
            } else {
                s2Cur = 0;
                if (gS1Cur + 1 < gS1LoopTimes) {
                    gS1Cur++;
                } else {
                    gS1Cur = 0;
                    bN2Cur++;
                }
            }
        }

        // ---- execute：MM1/vec1（当前槽）+ MM2/vec2（PRELOAD_N 拍前）----
        shouldExecuteTask = false;
        for (uint32_t i = 0; i < 3; i++) {
            if (runInfo[i]) {
                shouldExecuteTask = true;
                break;
            }
        }
        if (shouldExecuteTask) {
            uint64_t loop = executedTaskCount;

            // ===== MM1 / vec1（当前槽，2 个 subLoop）=====
            if (runInfo[loop % 3]) {
                for (uint32_t subLoop = 0; subLoop < 2; ++subLoop) {
                    bmm1_id++;
                    if ASCEND_IS_AIC {
                        blockMmad.IterateBmm1(mm1Res[subLoop], bmm1_id % 2,
                                              bIdxp[loop % 3], n2Idxp[loop % 3],
                                              gS1Idxp[loop % 3], s2Idxp[loop % 3],
                                              curS2Loop[loop % 3], subLoop);
                    } else {
                        epilogue.ProcessVec1(
                            p1[loop % 3], mm1Res[subLoop], bmm1_id % 2, loop % 3,
                            bIdxp[loop % 3], gS1Idxp[loop % 3], s2Idxp[loop % 3],
                            loopIdxp[loop % 3], mloopIdxp[loop % 3],
                            vecMbaseIdxp[loop % 3], curS2Loop[loop % 3], subLoop);
                    }
                }
            }

            // ===== MM2 / vec2（PRELOAD_N 拍前的槽）=====
            if (loop >= PRELOAD_N) {
                uint32_t mm2Slot = (loop - PRELOAD_N) % 3;
                if (runInfo[mm2Slot]) {
                    if ASCEND_IS_AIC {
                        blockMmad.IterateBmm2(mm2Slot, mm2Res, p1[mm2Slot], bmm2_id,
                                              bIdxp[mm2Slot], n2Idxp[mm2Slot],
                                              s2Idxp[mm2Slot]);
                    } else {
                        epilogue.ProcessVec2(
                            mm2Res, bmm2_id,
                            loopIdxp[mm2Slot], mloopIdxp[mm2Slot],
                            bIdxp[mm2Slot], n2Idxp[mm2Slot],
                            gS1Idxp[mm2Slot], vecMbaseIdxp[mm2Slot],
                            curS2Loop[mm2Slot]);
                    }
                    runInfo[mm2Slot] = false;
                }
            }

            executedTaskCount++;
        }
    }

    // ============== FIA 框架：清理 ==============
    if ASCEND_IS_AIV {
        epilogue.FreeEventID();
    } else {
        // postalloc_l0c: 等待预释放的 L0C 标志被消费完毕
        wait_flag(PIPE_FIX, PIPE_M, 0);
        wait_flag(PIPE_FIX, PIPE_M, 1);
    }

    // CrossCoreBufferUnInit: AIC 等待 AIV 最后几个未消费的跨核标志
    if ASCEND_IS_AIC {
        wait_intra_block(PIPE_FIX, bmm2_id + 5);
        wait_intra_block(PIPE_FIX, bmm2_id + 5 + 16);
        wait_intra_block(PIPE_FIX, bmm2_id + 3);
        wait_intra_block(PIPE_FIX, bmm2_id + 3 + 16);
        wait_intra_block(PIPE_FIX, bmm2_id + 1);
        wait_intra_block(PIPE_FIX, bmm2_id + 1 + 16);
    }
}

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
