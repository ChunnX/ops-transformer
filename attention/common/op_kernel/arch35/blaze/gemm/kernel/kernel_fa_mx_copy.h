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
 *
 * NOTE: 本文件由原始 for 嵌套循环改写为 fia_kernel_fullquant_mx.h 的 while 软件流水结构
 *       （dispatch/execute 分离、PRELOAD_N=2、runInfo[3] 环形缓冲、mm1 预计 2 拍 / mm2 滞后 2 拍）。
 *       原始版本备份见同目录 kernel_fa_mx copy_orig.h。
 *
 * 配套的 block 层拆分（均已备份 *_orig.h）：
 *   - block_mmad_fa_mx.h     : operator() 拆为 mm1Process(Q·K) / mm2Process(P·V)，operator() 保持兼容。
 *   - block_epilogue_fa_mx.h : operator() 拆为 Prepare / vec1Process(softmax) / vec2Process(输出)，operator() 保持兼容。
 *
 * TODO(流水缓冲/同步): 当前 mm1/mm2 与 vec1/vec2 已可独立分拍调用，但 mm1Res[2] / mm2Res / p1[3] /
 *       softmax*UB 这些 3 级缓冲的地址与跨核同步 id（CROSS_CORE_SYNC_*、PIPE_FIX/V/MTE1/MTE3）
 *       仍需按 (loop % 3) 流水槽在 ExecuteTask 层重新组织，才能实现真正重叠。当前为结构骨架，
 *       需在硬件/编译环境下验证时序与 buffer 复用。
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
    int bmm1Id_{0};
    int bmm2Id_{4};
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

    // =====================================================================================
    // ============================  while 软件流水（复刻 fia_kernel_fullquant_mx.h）=========
    // =====================================================================================
    // 说明：本文件原为三成嵌套 for（bN2 / s1g / s2 串行），现改写为"发任务 + 执行任务"分离的
    //       while 软件流水。核心目的是让 MM1 提前 2 拍发出、MM2 滞后 2 拍执行，从而与
    //       softmax/vec1 重叠，减少硬件空闲等待。整体对应 fullquant 的 FlashAttention()。

    // ---- 核索引：本核在任务分配表里排第几 ----
    // 注意：AIC(cube)和 AIV(vec) 是配对的一组核，它们要处理同一批任务，所以
    //       AIV 要通过 GetTaskRation() 把核号折算回"对应的 AIC 组号"，保证 == 同一 aicCoreIdx。
    uint32_t aicCoreIdx = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        aicCoreIdx = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
    }
    uint32_t aicCoreNum = AscendC::GetBlockNum();

    // ---- bN2 连续区间分配（每核各分一段连续的 bN2）----
    // bN2 = 一个 (batch=bIdx, head=n2Idx) 组合，是任务的最小并行单位。
    // bN2Cnt = bSize(4) * realN2Size(8) = 32 个 (batch, head) 任务。
    // 每个核分 bN2PerCore 个（由 tiling 决定，当前用总数/核数向上取整兜底），
    // 本核负责 [bN2Start, bN2End) 这一段连续区间（与 fullquant 的 bN2Start/bN2End 一致）。
    uint32_t bN2Cnt = bSize * realN2Size;
    uint32_t bN2PerCore = CeilDiv(bN2Cnt, aicCoreNum);
    uint32_t bN2Start = aicCoreIdx * bN2PerCore;
    uint32_t bN2End = Min(bN2Start + bN2PerCore, bN2Cnt);
    if (bN2Start >= bN2End) {
        return; // 本核没分到任何 bN2（核数 > 任务数时可能发生），直接退出
    }

    // blockMmad / epilogue 是两个 block 层对象：
    //   blockMmad = AIC 侧的矩阵计算单元（MM1:Q·K、MM2:P·V）
    //   epilogue  = AIV 侧的向量处理单元（vec1:softmax、vec2:输出）
    BlockMmad blockMmad;
    blockMmad.Init();
    BlockEpilogue epilogue;
    epilogue.Init(params.epilogueParams);

    // ---- 软件流水状态（映射 fullquant 的 FlashAttention）----
    // PRELOAD_N=2：MM1 提前 2 拍发，MM2 滞后 2 拍执行。
    // 因为"错开 2 拍"，所以最多同时有 3 个不同块的任务在管线里周转，
    // 因此环形缓冲槽数 = PRELOAD_N + 1 = 3。
    constexpr uint32_t PRELOAD_N = 2;
    uint32_t bN2Cur = bN2Start;       // 正在"分发登记"的 bN2 下标（从本核区间起点开始）
    uint32_t gS1Cur = 0;              // 正在分发的 s1g 循环下标（S1 方向第几块）
    uint32_t s2Cur = 0;               // 正在分发的 s2 循环下标（KV 方向第几块）
    uint64_t createdTaskCount = 0;    // 已登记(发出)的任务总数
    uint64_t executedTaskCount = 0;   // 已执行完成的任务总数
    uint32_t mloopLocal = 0;          // 每个 (bN2, s1g) 组的编号（用于 softmax 运行时状态区分）


    // ---- 3 槽环形缓冲 ----
    // 这是软件流水的核心：把每个待处理任务的全部信息存进 runInfo[] 里的一个槽。
    // 因为 MM1 提前 2 拍、MM2 滞后 2 拍，同一管线里最多 3 个任务，故用 %3 轮流。
    // 每往管线里登记一个任务，就占一个新槽；槽被 MM2 消费(执行)后清空复用。
    //
    // 每个槽保存该任务需要的"坐标和尺寸"：
    //   bIdxp/n2Idxp  : 该任务的 (batch, head)
    //   gS1Idxp/s2Idxp: 该任务的 S1 块号 和 KV 块号
    //   bs1Offp       : 该 batch 在拼接 Q 中的起始行偏移（用于从大 Q 里切出这个 batch）
    //   actS1p/actS2p : 该 batch 实际的 S1 / S2 长度（每个 batch 可能不同）
    //   s1gBlockSizep : 该 s1g 块实际处理的行数（最后一块可能不满 128）
    //   s2BlockSizep  : 该 s2 块实际处理的列数（最后一块可能不满 512）
    //   s1gCoordp[]   : Q 切片的头/中间/尾巴坐标（{s1Start,s1End,gStart,gEnd}）
    //   blockIdxp     : 该 s2 块在 K/V 的 blockTable 里的逻辑块号
    //   mloopIdxp     : 该任务所属 (bN2,s1g) 组的编号
    //   loopIdxp      : 全局登记序号
    //   curS2Loop     : 该任务的 s2 下标（保存一份，供执行时用）
    //   runInfo[slot] : 该槽是否还有未消费的任务（true=有）
    uint32_t bIdxp[3], n2Idxp[3], gS1Idxp[3], s2Idxp[3];
    uint32_t mloopIdxp[3], loopIdxp[3], curS2Loop[3];
    uint32_t bs1Offp[3], actS1p[3], actS2p[3];
    uint32_t s1gBlockSizep[3], s2BlockSizep[3];
    uint32_t s1gCoordp[3][4]; // s1Start, s1End, gStart, gEnd
    uint32_t blockIdxp[3];
    bool runInfo[3] = {false, false, false};

    // 当前 bN2 的 s2 / gS1 循环次数（actS1/actS2 为动态值，逐 bN2 读取）
    uint32_t curS2LoopNum = 0;
    uint32_t curGS1LoopNum = 0;

    bool shouldDispatchTask = true;    // 还有任务要"发"(登记)吗
    bool shouldExecuteTask = false;    // 还有任务要"执行"吗

    // =================== 主循环（对应 fullquant 的 FlashAttention）===================
    // 循环反复做两件事，直到全部任务既发完又执行完：
    //   1. dispatch（发任务）：登记下一个块的信息（只登记，不真算）
    //   2. execute（执行）：  真正调 MM1/vec1/MM2/vec2 去算
    while (shouldDispatchTask || shouldExecuteTask) {
        // ------------------------- 阶段1：发任务（仅登记，不执行） -------------------------
        // 对应 fullquant 的 ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur)：
        //   return (bN2Cur != bN2End) || (gS1Cur != gS1OEnd) || (s2Cur != s2OEnd);
        // 其中 gS1OEnd == s2OEnd == 0（见 fia_kernel_fullquant_mx.h:57-58）。
        // 即：只要 (bN2, gS1, s2) 三元组还没走到结束态（三者都到末尾），就还有任务可分。
        // 之所以不能只看 bN2：轴推进 UpdateAxisInfo 可能会停在"bN2 未变但 gS1/s2 已将尽"的中间态，
        // 因此要三元组一起判断，忠实于 fullquant。
        shouldDispatchTask = (bN2Cur != bN2End) || (gS1Cur != 0) || (s2Cur != 0);
        if (shouldDispatchTask) {
            // 每当进入一个新的 (bN2, s1g) 组（即 s2 回到 0 = 该组的第一个 KV 块），
            // mloop 就 +1。mloop 用来区分不同的 s1g 组，softmax 的累加状态是按它存槽的。
            if (s2Cur == 0) {
                mloopLocal++;
            }
            // 计算本次要登记的任务分到哪个槽（%3 环形）
            auto slot = createdTaskCount % 3;
            auto bIdx = bN2Cur / realN2Size;   // 从 bN2 拆出 batch 序号
            auto n2Idx = bN2Cur % realN2Size;  // 从 bN2 拆出 head 序号
            // --- 读取该 batch 的实际长度（每个 batch 的句子长度可能不一样）---
            // Q 是按 [B*S1] 顺序拼接存储的：第 b 个 batch 的 Q 从 bs1Offset 开始，长度 actS1
            uint32_t bs1Offset = 0;
            uint32_t actS1 = tensorSeqlenQ[AscendC::Te::MakeCoord(0, bIdx)];
            if (bIdx > 0) {
                bs1Offset = tensorSeqlenQ[AscendC::Te::MakeCoord(0, bIdx - 1)];
                actS1 -= bs1Offset;
            }
            // KV 方向也有实际长度 actS2
            uint32_t actS2 = tensorSeqlenKV[AscendC::Te::MakeCoord(0, bIdx)];
            // 当这是该 bN2 的第一个任务（s1g=0 且 s2=0）时，算出该 batch 需要循环几轮，
            // 供后面的"轴推进"判断什么时候该换 s1g / 换 bN2
            if (gS1Cur == 0 && s2Cur == 0) {
                curS2LoopNum = CeilDiv(actS2, static_cast<uint64_t>(s2BaseSize));
                curGS1LoopNum = CeilDiv(actS1 * gSize, static_cast<uint64_t>(mBaseSize));
            }

            // --- 计算这个任务实际要处理的块大小 ---
            // s1g 块：默认 128 行一整块，最后一块可能不满（用剩余行数）
            uint32_t s1gBlockSize = mBaseSize;
            uint32_t gS1LoopNum = CeilDiv(actS1 * gSize, static_cast<uint64_t>(mBaseSize));
            if (gS1Cur + 1 == gS1LoopNum) {
                s1gBlockSize = actS1 * gSize - mBaseSize * gS1Cur;
            }
            // s2 块：默认 512 列一整块，最后一块可能不满
            uint32_t s2LoopNum = CeilDiv(actS2, static_cast<uint64_t>(s2BaseSize));
            uint32_t s2BlockSize = s2BaseSize;
            if (s2Cur + 1 == s2LoopNum) {
                s2BlockSize = actS2 - s2BaseSize * s2Cur;
            }
            // Q 切片坐标：Q 是 [S1, G, D] 排布，128 行可能横跨多个 (S1,G)，需要拆出头/中/尾三段坐标
            uint32_t s1Start = gS1Cur * mBaseSize / gSize;
            uint32_t s1End = Min(actS1 * gSize, (gS1Cur + 1) * mBaseSize) / gSize;
            uint32_t gStart = gS1Cur * mBaseSize % gSize;
            uint32_t gEnd = Min(actS1 * gSize, (gS1Cur + 1) * mBaseSize) % gSize;
            // K/V 的逻辑块号：因为 s2BaseSize==blockSize(512)，所以 KV 块号 = s2 下标
            uint32_t blockIdx = tensorBlockTable[AscendC::Te::MakeCoord(bIdx, s2Cur)];

            // --- 把上面算好的所有信息写进当前槽 ---
            bIdxp[slot] = bIdx;
            n2Idxp[slot] = n2Idx;
            gS1Idxp[slot] = gS1Cur;
            s2Idxp[slot] = s2Cur;
            bs1Offp[slot] = bs1Offset;
            actS1p[slot] = actS1;
            actS2p[slot] = actS2;
            s1gBlockSizep[slot] = s1gBlockSize;
            s2BlockSizep[slot] = s2BlockSize;
            s1gCoordp[slot][0] = s1Start;
            s1gCoordp[slot][1] = s1End;
            s1gCoordp[slot][2] = gStart;
            s1gCoordp[slot][3] = gEnd;
            blockIdxp[slot] = blockIdx;
            mloopIdxp[slot] = mloopLocal;
            loopIdxp[slot] = createdTaskCount;
            curS2Loop[slot] = s2Cur;
            runInfo[slot] = true;   // 标记该槽有未消费的任务
            createdTaskCount++;     // 登记数 +1

            // ---- 轴推进（对应 fullquant 的 UpdateAxisInfo）----
            // 遍历顺序和原来 for 一样：先尽量遍历 s2（最内层），s2 走完换 s1g，
            // 最后才换 bN2。这样从"三元组坐标"看，就是按 (s2, s1g, bN2) 字典序推进。
            if (s2Cur + 1 < curS2LoopNum) {
                s2Cur++;                                  // 1) 同一 s1g 内：s2 前进一块
            } else {
                s2Cur = 0;                                // 2) s2 走完，回到 0
                if (gS1Cur + 1 < curGS1LoopNum) {
                    gS1Cur++;                             //    同一 bN2 内：s1g 前进一块
                } else {
                    gS1Cur = 0;                           // 3) s1g 也走完
                    bN2Cur++;                             //    换下一个 bN2
                }
            }
        }

        // ------------------------- 阶段2：执行任务（真正去算） -------------------------
        // 只要任意槽里还有未消费的任务，就进入执行
        shouldExecuteTask = runInfo[0] || runInfo[1] || runInfo[2];
        if (shouldExecuteTask) {
            uint64_t loop = executedTaskCount;  // 本轮到第几个执行任务
            // ===== 执行 MM1 / vec1（用"刚登记进管线"的槽，即当前任务）=====
            // 注意：这里读的是 runInfo[loop % 3]，也就是最近才 register 的那个槽的任务
            if (runInfo[loop % 3]) {
                // ---- 重建该任务的 slice tensor ----
                // dispatch 阶段只存了"坐标/尺寸"数字，没真正切片；
                // 真正执行时，才根据这些数字从大 GM tensor 里现切出 Q/K/QScale/KScale 小块。
                // Q：从大 Q[.., n2Idx, ..] 里按 bs1Offset/actS1 切出本 batch 的 S1 行，再 Squeeze 掉 head 维
                auto subTensorQ = tensorQ.Slice(
                    AscendC::Te::MakeCoord(bs1Offp[loop % 3], n2Idxp[loop % 3], AscendC::Te::MakeCoord(0, 0)),
                    AscendC::Te::MakeShape(actS1p[loop % 3], 1, AscendC::Te::MakeShape(gSize, dSize)));
                auto blockTensorQ = AscendC::Te::Squeeze<1>(subTensorQ);
                // QScale：和 Q 一样切本 batch 的 scale 块
                auto subTensorQScale = tensorQScale.Slice(
                    AscendC::Te::MakeCoord(bs1Offp[loop % 3], n2Idxp[loop % 3], AscendC::Te::MakeCoord(0, 0)),
                    AscendC::Te::MakeShape(actS1p[loop % 3], 1, AscendC::Te::MakeShape(gSize, scaleDSize)));
                auto blockTensorQScale = AscendC::Te::Squeeze<1>(subTensorQScale);
                // K：按 blockIdx/n2Idx 切出该 KV 块（NZ 排布）
                auto subTensorK = tensorK.Slice(
                    AscendC::Te::MakeCoord(
                        blockIdxp[loop % 3], n2Idxp[loop % 3],
                        AscendC::Te::MakeCoord(AscendC::Te::_0{}, 0), AscendC::Te::MakeCoord(AscendC::Te::_0{}, 0)),
                    AscendC::Te::MakeShape(
                        1, 1,
                        AscendC::Te::MakeShape(AscendC::Std::Int<c0Size>{}, dSize / c0Size),
                        AscendC::Te::MakeShape(AscendC::Te::_16{}, blockSize / AscendC::Te::_16{})));
                auto blockTensorK = AscendC::Te::Squeeze<0, 1>(subTensorK);
                // KScale：同理切该 KV 块的 scale
                auto subTensorKScale = tensorKScale.Slice(
                    AscendC::Te::MakeCoord(
                        blockIdxp[loop % 3], n2Idxp[loop % 3],
                        AscendC::Te::MakeCoord(AscendC::Te::_0{}, 0), AscendC::Te::MakeCoord(AscendC::Te::_0{}, 0)),
                    AscendC::Te::MakeShape(
                        1, 1,
                        AscendC::Te::MakeShape(AscendC::Std::Int<MXFP_MULTI_BASE_SIZE>{}, scaleDSize / MXFP_MULTI_BASE_SIZE),
                        AscendC::Te::MakeShape(AscendC::Te::_16{}, blockSize / AscendC::Te::_16{})));
                auto blockTensorKScale = AscendC::Te::Squeeze<0, 1>(subTensorKScale);
                // Q 切片的头/中间/尾巴坐标（blockMmad 内部用它来正确搬 Q）
                BlockCoord s1gCoord{s1gCoordp[loop % 3][0], s1gCoordp[loop % 3][1],
                                    s1gCoordp[loop % 3][2], s1gCoordp[loop % 3][3]};
                uint32_t s1gSize = s1gBlockSizep[loop % 3];
                uint32_t s2Size = s2BlockSizep[loop % 3];

                // 注：blockMmad.mm1Process / epilogue.vec1Process 内部已对 s2Block 按 256 切分循环
                //     （s2TileSize=256），因此每个任务只需调用一次（不再按 subLoop 拆 2 次）。
                bmm1Id_++;
                if ASCEND_IS_AIC {
                    // AIC 核：只做第一段矩阵乘法 MM1 = Q · K^T
                    blockMmad.mm1Process(blockTensorQ, blockTensorK, blockTensorQScale, blockTensorKScale,
                                         s1gCoord, s1gSize, s2Size);
                } else {
                    // AIV 核：先 Prepare（把 blockShape/coord 解析缓存，vec1/vec2 共用），
                    //        再执行 vec1(softmax)。此时 MM1 的 QK 结果已由 AIC 写到 UB。
                    // TODO(流水缓冲): mm1Res/mm2Res/p1/softmax*UB 的 3 级缓冲与跨核同步 id 需按
                    //                (loop % 3 槽) 重新分配，当前沿用 fullquant 的 ProcessVec1 思路。
                    BlockShape shape = {actS1p[loop % 3], actS2p[loop % 3], s1gSize, s2Size};
                    BlockCoord coord = {bIdxp[loop % 3], n2Idxp[loop % 3], gS1Idxp[loop % 3], s2Idxp[loop % 3]};
                    epilogue.Prepare(shape, coord, mloopIdxp[loop % 3]);
                    epilogue.vec1Process();
                }
            }
            // ===== 执行 MM2 / vec2（用"2 拍之前"登记进管线的槽）=====
            // 只有当执行序号 loop >= PRELOAD_N(=2) 时，才有"2 拍前登记的任务"可以消费。
            // 这样 MM2/P·V 就会比 MM1 晚 2 个任务执行 —— 这正是软件流水的精髓：
            // 在算第 loop 个任务的 MM2 时，第 loop 个任务的 vec1 早就完成、且第 loop+2 个的 MM1 已开始。
            if (loop >= PRELOAD_N) {
                uint32_t mm2Slot = (loop - PRELOAD_N) % 3;  // 2 拍前登记的槽
                if (runInfo[mm2Slot]) {
                    if ASCEND_IS_AIC {
                        // AIC：重建该任务的 V / VScale slice，做 MM2 = P · V
                        auto subTensorV = tensorV.Slice(
                            AscendC::Te::MakeCoord(
                                blockIdxp[mm2Slot], n2Idxp[mm2Slot],
                                AscendC::Te::MakeCoord(AscendC::Te::_0{}, 0), AscendC::Te::MakeCoord(AscendC::Te::_0{}, 0)),
                            AscendC::Te::MakeShape(
                                1, 1,
                                AscendC::Te::MakeShape(AscendC::Te::_16{}, blockSize / AscendC::Te::_16{}),
                                AscendC::Te::MakeShape(AscendC::Std::Int<c0Size>{}, dSize / c0Size)));
                        auto blockTensorV = AscendC::Te::Squeeze<0, 1>(subTensorV);
                        auto subTensorVScale = tensorVScale.Slice(
                            AscendC::Te::MakeCoord(
                                blockIdxp[mm2Slot], n2Idxp[mm2Slot],
                                AscendC::Te::MakeCoord(AscendC::Te::_0{}, 0), AscendC::Te::MakeCoord(AscendC::Te::_0{}, 0)),
                            AscendC::Te::MakeShape(
                                1, 1,
                                AscendC::Te::MakeShape(AscendC::Std::Int<MXFP_MULTI_BASE_SIZE>{}, reduceSize / MXFP_MULTI_BASE_SIZE),
                                AscendC::Te::MakeShape(AscendC::Te::_16{}, dSize / AscendC::Te::_16{})));
                        auto blockTensorVScale = AscendC::Te::Squeeze<0, 1>(subTensorVScale);
                        blockMmad.mm2Process(blockTensorV, blockTensorVScale, s1gBlockSizep[mm2Slot], s2BlockSizep[mm2Slot]);
                    } else {
                        // AIV：对滞后任务执行 vec2(输出汇总)
                        // TODO(流水缓冲): 缓冲/同步 id 需按流水槽重分配，当前沿用 fullquant 思路。
                        epilogue.vec2Process();
                    }
                    runInfo[mm2Slot] = false;  // 该槽任务已全部消费（MM1+MM2/vec1+vec2 都做完），清空复用
                }
            }
            executedTaskCount++;  // 执行任务数 +1
        }
    }
}

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
