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
 * \file grouped_matmul_swiglu_quant_v2_a4w4_post.h
 * \brief
 */

#ifndef OP_KERNEL_GROUPED_MATMUL_SWIGLU_QUANT_V2_A4W4_POST_H
#define OP_KERNEL_GROUPED_MATMUL_SWIGLU_QUANT_V2_A4W4_POST_H

#include "grouped_matmul_swiglu_quant_v2_utils_kernel.h"
#include "kernel_operator.h"

#ifdef GMM_SWIGLU_QUANT_V2_A4W4

namespace GroupedMatmulDequantSwigluQuant {
using namespace AscendC;
#define DOUBLE_BUFFER 2
constexpr float DEFAULT_MUL_SCALE = 16.0f;
class GMMA4W4PostProcess {
public:
    __aicore__ inline GMMA4W4PostProcess(){};
    __aicore__ inline void Init(const GMAddrParams gmAddrParams,
                                const GMMSwigluQuantV2BaseParams *__restrict gmmSwigluQuantV2BaseParamsIN,
                                const GMMSwigluQuantV2 *__restrict gmmSwigluIN);

    __aicore__ inline void Process(WorkSpaceSplitConfig &workspaceSplitConfig, int64_t workspaceSplitLoopIdx,
                                   TPipe *pipe);

private:
    __aicore__ inline void UpdateVecConfig(uint32_t blockIdx, VecConfig &vecConfig,
                                           WorkSpaceSplitConfig &workspaceSplitConfig, int64_t workspaceSplitLoopIdx,
                                           TPipe *pipe);

    __aicore__ inline void VectorCompute(uint32_t loopIdx, VecConfig &vecConfig,
                                         WorkSpaceSplitConfig &workspaceSplitConfig);

    __aicore__ inline void customDataCopyIn(uint32_t outLoopIdx, GlobalTensor<half> &mmOutGM, VecConfig &vecConfig,
                                            WorkSpaceSplitConfig &workspaceSplitConfig);

    __aicore__ inline void customDataCopyOut(VecConfig &vecConfig, WorkSpaceSplitConfig &workspaceSplitConfig);

    __aicore__ inline void Quant(uint32_t loopIdx, VecConfig &vecConfig);

    __aicore__ inline void Swiglu(uint32_t loopIdx, VecConfig &vecConfig);

    __aicore__ inline void MulPertokenScale(uint32_t loopIdx, VecConfig &vecConfig,
                                            WorkSpaceSplitConfig &workspaceSplitConfig);

    __aicore__ inline void ApplySmoothScale(uint32_t loopIdx, VecConfig &vecConfig,
                                            WorkSpaceSplitConfig &workspaceSplitConfig);

    __aicore__ inline int64_t SeekExpertGroup(int64_t tokenRow, VecConfig &vecConfig,
                                              WorkSpaceSplitConfig &workspaceSplitConfig);

    const GMMSwigluQuantV2 *__restrict gmmSwigluQuantV2;
    const GMMSwigluQuantV2BaseParams *__restrict gmmSwigluQuantV2BaseParams;
    GlobalTensor<float> perTokenScaleGM;
    GlobalTensor<int64_t> groupListGM;
    GlobalTensor<float> smoothScaleGM;
    GlobalTensor<int8_t> quantOutputGM;
    GlobalTensor<float> quantScaleOutputGM;
    GlobalTensor<half> mmOutGM1;
    GlobalTensor<half> mmOutGM2;
    GlobalTensor<half> mmOutGM;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> mmOutQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> quantOutQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> quantScaleOutQueue;
    TBuf<TPosition::VECCALC> reduceWorkspace;
    LocalTensor<float> curFp32;
    LocalTensor<int8_t> curQuant;
    LocalTensor<float> curScale;
    uint32_t blockIdx = 0;
    int64_t aicCoreNum = 0;
    int64_t aivCoreNum = 0;
};

__aicore__ inline void GMMA4W4PostProcess::Init(
    const GMAddrParams gmAddrParams, const GMMSwigluQuantV2BaseParams *__restrict gmmSwigluQuantV2BaseParamsIN,
    const GMMSwigluQuantV2 *__restrict gmmSwigluIN)
{
    if ASCEND_IS_AIV {
        aicCoreNum = GetBlockNum();
        aivCoreNum = aicCoreNum * NUM_2;
        blockIdx = GetBlockIdx();
        gmmSwigluQuantV2BaseParams = gmmSwigluQuantV2BaseParamsIN;
        gmmSwigluQuantV2 = gmmSwigluIN;
        groupListGM.SetGlobalBuffer((__gm__ int64_t *)gmAddrParams.groupListGM, gmmSwigluQuantV2->groupListLen);
        mmOutGM1.SetGlobalBuffer((__gm__ half *)((__gm__ int8_t *)gmAddrParams.workSpaceGM));
        mmOutGM2.SetGlobalBuffer(
            (__gm__ half *)((__gm__ int8_t *)gmAddrParams.workSpaceGM + gmAddrParams.workSpaceOffset1));
        perTokenScaleGM.SetGlobalBuffer((__gm__ float *)gmAddrParams.xScaleGM, gmmSwigluQuantV2BaseParams->M);
        smoothScaleGM.SetGlobalBuffer((__gm__ float *)gmAddrParams.smoothScaleGM);
        quantOutputGM.SetGlobalBuffer(
            (__gm__ int8_t *)gmAddrParams.yGM,
            gmmSwigluQuantV2BaseParams->M * gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR);
        quantScaleOutputGM.SetGlobalBuffer((__gm__ float *)gmAddrParams.yScaleGM, gmmSwigluQuantV2BaseParams->M);
    }
}

// MTE2: 将 outLoopIdx 对应 tile 的 fp16 中间结果搬入 mmOutQueue 空闲 buffer 并入队(触发 MTE2->V)。
// double buffer 下本函数只做搬入, 不内联 Cast(Cast 由 Process 在持有该 buffer 时执行, 使本 tile 的
// Cast/计算能与下一 tile 的搬入重叠)。outLoopIdx 决定本 tile 行数(末 tile 取 tailLoopNum)。
__aicore__ inline void GMMA4W4PostProcess::customDataCopyIn(uint32_t outLoopIdx, GlobalTensor<half> &mmOutGM,
                                                            VecConfig &vecConfig,
                                                            WorkSpaceSplitConfig &workspaceSplitConfig)
{
    (void)workspaceSplitConfig;
    const int64_t maxRow = gmmSwigluQuantV2->maxProcessRowNum;
    int64_t rows = (outLoopIdx == vecConfig.outLoopNum - 1) ? vecConfig.tailLoopNum : maxRow;
    int64_t processNum = rows * gmmSwigluQuantV2->tokenLen;
    LocalTensor<half> inHalf = mmOutQueue.AllocTensor<half>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(processNum * SIZE_OF_HALF_2), 0, 0, 0};
    DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
    DataCopyPad(inHalf[processNum], mmOutGM[vecConfig.curOffset], copyParams, padParams);
    mmOutQueue.EnQue(inHalf);
    vecConfig.curIdx += rows;
    vecConfig.curOffset = vecConfig.curIdx * gmmSwigluQuantV2->tokenLen;
}

__aicore__ inline void GMMA4W4PostProcess::VectorCompute(uint32_t loopIdx, VecConfig &vecConfig,
                                                         WorkSpaceSplitConfig &workspaceSplitConfig)
{
    // 1.perToken反量化
    MulPertokenScale(loopIdx, vecConfig, workspaceSplitConfig);
    // 2.Swiglu
    Swiglu(loopIdx, vecConfig);
    // 3.ApplySmoothScale（smoothScaleDimNum为0时跳过，表示smoothScale为空指针）
    if (gmmSwigluQuantV2BaseParams->smoothScaleDimNum != 0) {
        ApplySmoothScale(loopIdx, vecConfig, workspaceSplitConfig);
    }
    // 4.Quant
    Quant(loopIdx, vecConfig);
}

__aicore__ inline void GMMA4W4PostProcess::MulPertokenScale(uint32_t loopIdx, VecConfig &vecConfig,
                                                            WorkSpaceSplitConfig &workspaceSplitConfig)
{
    float scale = perTokenScaleGM.GetValue(loopIdx + workspaceSplitConfig.leftMatrixStartIndex + vecConfig.startIdx);
    GmmsqSetWaitFlag<HardEvent::S_V>();
    Muls(curFp32[loopIdx * gmmSwigluQuantV2->tokenLen], curFp32[loopIdx * gmmSwigluQuantV2->tokenLen], scale,
         gmmSwigluQuantV2->tokenLen);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void GMMA4W4PostProcess::Swiglu(uint32_t loopIdx, VecConfig &vecConfig)
{
    // 高阶API swiglu
    float beta = 1.0f;
    LocalTensor<float> workspaceLocal = reduceWorkspace.Get<float>();
    LocalTensor<float> src0Local =
        curFp32[loopIdx * gmmSwigluQuantV2->tokenLen + gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR];
    LocalTensor<float> src1Local = curFp32[loopIdx * gmmSwigluQuantV2->tokenLen];

    SwiGLU<float, false>(workspaceLocal, src0Local, src1Local, beta, gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR);
    PipeBarrier<PIPE_V>();
    DataCopyParams repeatParams{
        1, static_cast<uint16_t>((gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR) / ALIGN_8_ELE), 0, 0};
    DataCopy(curFp32[loopIdx * gmmSwigluQuantV2->tokenLen], workspaceLocal, repeatParams);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void GMMA4W4PostProcess::ApplySmoothScale(uint32_t loopIdx, VecConfig &vecConfig,
                                                            WorkSpaceSplitConfig &workspaceSplitConfig)
{
    const int64_t smoothScaleDimNum = gmmSwigluQuantV2BaseParams->smoothScaleDimNum;
    const int64_t halfTokenLen = gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR;
    const int64_t tokenRow = workspaceSplitConfig.leftMatrixStartIndex + vecConfig.startIdx + loopIdx;
    // 游标: 由单调递增的全局行号派生所属专家(只在跨边界时前进, 跳过空专家), 均摊 O(1)/行。
    const int64_t groupIdx = SeekExpertGroup(tokenRow, vecConfig, workspaceSplitConfig);

    const uint64_t preOffset = static_cast<uint64_t>(loopIdx) * gmmSwigluQuantV2->tokenLen;

    if (smoothScaleDimNum == NUM_2) {
        // smoothScale (E, N/2): 用游标得到的 groupIdx 取该专家那一行, 逐元素乘(维持基线写法)。
        // 注: dim==2 的每元素标量循环是独立瓶颈, 整行载入 UB + 向量 Mul 的优化需配 dim==2 harness
        //     与 UB 预算专项验证, 未纳入本次游标改动。
        GmmsqSetWaitFlag<HardEvent::V_S>();
        for (uint32_t j = 0; j < halfTokenLen; j++) {
            float scale = smoothScaleGM.GetValue(groupIdx * halfTokenLen + j);
            float val = curFp32.GetValue(preOffset + j);
            curFp32.SetValue(preOffset + j, val * scale);
        }
    } else if (smoothScaleDimNum == 1) {
        // smoothScale (E,): 单标量广播到 N/2
        float scale = smoothScaleGM.GetValue(groupIdx);
        GmmsqSetWaitFlag<HardEvent::S_V>();
        Muls(curFp32[preOffset], curFp32[preOffset], scale, halfTokenLen);
        PipeBarrier<PIPE_V>();
    }
}

// 专家游标: tokenRow 单调递增, 只在越过当前专家末时前进(while 天然跳过 0 长度空专家)。
// 无独立倒计时计数器 => 不会与行进度脱节; 一次性初始化用 chunk 起始专家定位, 之后均摊 O(1)。
__aicore__ inline int64_t GMMA4W4PostProcess::SeekExpertGroup(int64_t tokenRow, VecConfig &vecConfig,
                                                              WorkSpaceSplitConfig &workspaceSplitConfig)
{
    const int64_t groupNum = gmmSwigluQuantV2->groupListLen;
    const bool isCumsum = (gmmSwigluQuantV2BaseParams->groupListType == 0);
    if (vecConfig.expertCursorEnd < 0) {
        // 一次性定位: 从本 chunk 的起始专家扫到覆盖 tokenRow 的专家(每个 chunk 只做一次)。
        int64_t g = workspaceSplitConfig.rightMatrixExpertStartIndex;
        int64_t end;
        if (isCumsum) {
            end = groupListGM.GetValue(g);
            while (g < groupNum - 1 && tokenRow >= end) {
                g++;
                end = groupListGM.GetValue(g);
            }
        } else {
            int64_t prefix = 0;
            for (int64_t i = 0; i <= g; i++) {
                prefix += groupListGM.GetValue(i);
            }
            end = prefix;
            while (g < groupNum - 1 && tokenRow >= end) {
                g++;
                end += groupListGM.GetValue(g);
            }
        }
        vecConfig.expertCursorIdx = g;
        vecConfig.expertCursorEnd = end;
        return g;
    }
    // 快路径: 行号单调, 仅跨边界才前进(跳过空专家)。若已到最后专家仍越界, 属数据/切分异常,
    // 保守钳在最后一个专家(不越界读), 不静默产出错误 scale 之外行为。
    while (vecConfig.expertCursorIdx < groupNum - 1 && tokenRow >= vecConfig.expertCursorEnd) {
        vecConfig.expertCursorIdx++;
        if (isCumsum) {
            vecConfig.expertCursorEnd = groupListGM.GetValue(vecConfig.expertCursorIdx);
        } else {
            vecConfig.expertCursorEnd += groupListGM.GetValue(vecConfig.expertCursorIdx);
        }
    }
    return vecConfig.expertCursorIdx;
}

__aicore__ inline void GMMA4W4PostProcess::Quant(uint32_t loopIdx, VecConfig &vecConfig)
{
    uint64_t preOffset = loopIdx * gmmSwigluQuantV2->tokenLen;
    uint64_t halfTokenLen = gmmSwigluQuantV2->tokenLen / BISECT;
    if (gmmSwigluQuantV2BaseParams->smoothScaleDimNum == NUM_2) {
        GmmsqSetWaitFlag<HardEvent::S_V>();
    }
    Abs(curFp32[preOffset + gmmSwigluQuantV2->tokenLen / BISECT], curFp32[preOffset], halfTokenLen);
    PipeBarrier<PIPE_V>();
    // reduceMax
    LocalTensor<float> workLocal = reduceWorkspace.Get<float>(halfTokenLen);
    LocalTensor<float> reduceResLocal =
        reduceWorkspace.GetWithOffset<float>(FLOAT_UB_BLOCK_UNIT_SIZE, halfTokenLen * sizeof(float));
    LocalTensor<float> reduceTmpLocal = reduceWorkspace.GetWithOffset<float>(
        FLOAT_UB_BLOCK_UNIT_SIZE, halfTokenLen * sizeof(float) + UB_BLOCK_UNIT_SIZE);
    ReduceMaxTemplate(reduceResLocal, workLocal, curFp32[preOffset + gmmSwigluQuantV2->tokenLen / BISECT],
                      reduceTmpLocal, static_cast<uint32_t>(halfTokenLen));
    GmmsqSetWaitFlag<HardEvent::V_S>();
    float quantScale = reduceResLocal.GetValue(0) / QUANT_SCALE_INT8;
    curScale.SetValue(loopIdx, quantScale);
    quantScale = QUANT_SCALE_INT8 / reduceResLocal.GetValue(0);
    GmmsqSetWaitFlag<HardEvent::S_V>();
    Muls(curFp32[preOffset], curFp32[preOffset], quantScale, halfTokenLen);
    PipeBarrier<PIPE_V>();
    int32_t dstTempOffset = static_cast<int32_t>(preOffset / BISECT);
    int32_t srcTempOffset = static_cast<int32_t>(preOffset);
    int32_t tempCount = static_cast<int32_t>(halfTokenLen);
    LocalTensor<int8_t> castSpace = reduceWorkspace.Get<int8_t>(UB_BLOCK_UNIT_SIZE);
    CastFp32ToInt8Template(curQuant, curFp32, castSpace, dstTempOffset, srcTempOffset, tempCount);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void GMMA4W4PostProcess::UpdateVecConfig(uint32_t blockIdx, VecConfig &vecConfig,
                                                           WorkSpaceSplitConfig &workspaceSplitConfig,
                                                           int64_t workspaceSplitLoopIdx, TPipe *pipe)
{
    // 第一步 读取grouplist reduceSum 计算总数据个数
    vecConfig.M = workspaceSplitLoopIdx < workspaceSplitConfig.loopCount - 1 ? workspaceSplitConfig.notLastTaskSize :
                                                                               workspaceSplitConfig.lastLoopTaskSize;
    // 第二步 计算分核
    uint32_t eachCoreTaskNum = (vecConfig.M + aivCoreNum - 1) / aivCoreNum;
    vecConfig.usedCoreNum = vecConfig.M >= aivCoreNum ? aivCoreNum : vecConfig.M;
    uint32_t tailCoreIdx = vecConfig.M - (eachCoreTaskNum - 1) * vecConfig.usedCoreNum;
    vecConfig.taskNum = blockIdx < tailCoreIdx ? eachCoreTaskNum : eachCoreTaskNum - 1;
    vecConfig.startIdx =
        blockIdx < tailCoreIdx ? eachCoreTaskNum * blockIdx : ((eachCoreTaskNum - 1) * blockIdx + tailCoreIdx);
    vecConfig.curIdx = vecConfig.startIdx;
    vecConfig.startOffset = vecConfig.startIdx * gmmSwigluQuantV2->tokenLen;
    vecConfig.curOffset = vecConfig.startOffset;
    int64_t curStartIdx = vecConfig.startIdx;
    int64_t prevM = workspaceSplitLoopIdx * workspaceSplitConfig.notLastTaskSize;
    int64_t totalTmp = 0;
    if (gmmSwigluQuantV2BaseParams->groupListType == 1) {
        for (uint32_t i = 0; i < workspaceSplitConfig.rightMatrixExpertStartIndex; i++) {
            totalTmp += groupListGM.GetValue(i);
        }
    }
    for (uint32_t groupIdx = workspaceSplitConfig.rightMatrixExpertStartIndex;
         groupIdx <= workspaceSplitConfig.rightMatrixExpertEndIndex; groupIdx++) {
        int64_t currM = 0;
        if (gmmSwigluQuantV2BaseParams->groupListType == 0) {
            currM = groupListGM.GetValue(groupIdx);
        } else {
            totalTmp += groupListGM.GetValue(groupIdx);
            currM = totalTmp;
        }
        int64_t tempM = currM - prevM;
        prevM = currM;
        curStartIdx -= tempM;
    }
    // 游标状态按 chunk 重置(expertCursorEnd<0 触发 SeekExpertGroup 一次性定位)
    vecConfig.expertCursorEnd = -1;
    // 第三步 计算总数据量
    vecConfig.outLoopNum =
        (vecConfig.taskNum + gmmSwigluQuantV2->maxProcessRowNum - 1) / gmmSwigluQuantV2->maxProcessRowNum;
    vecConfig.tailLoopNum = vecConfig.taskNum % gmmSwigluQuantV2->maxProcessRowNum ?
                                vecConfig.taskNum % gmmSwigluQuantV2->maxProcessRowNum :
                                gmmSwigluQuantV2->maxProcessRowNum;

    // 第四步 申请空间 (vector double buffer, 每队列 2 个 buffer)
    // A4W4 double buffer UB 实际分配(与 host CalMaxRowInUbA4W4 的预算严格一致):
    //   9 * row * n + 8 * AlignUp(row, 8) + 2 * n + 64 <= ubSize
    //   mmOutQueue(fp32, depth2)        = 8 * row * n    ) 9 * row * n
    //   quantOutQueue(int8, depth2)    =     row * n    )
    //   quantScaleOutQueue(fp32, depth2)= 8 * AlignUp(row,8)
    //   reduceWorkspace(SwiGLU tmp n/2*fp32 + ReduceMax res/tmp 各 32B) = 2*n + 64
    pipe->InitBuffer(mmOutQueue, DOUBLE_BUFFER,
                     gmmSwigluQuantV2->maxProcessRowNum * gmmSwigluQuantV2->tokenLen * sizeof(float));
    pipe->InitBuffer(
        quantOutQueue, DOUBLE_BUFFER,
        gmmSwigluQuantV2->maxProcessRowNum * gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR * sizeof(int8_t));
    pipe->InitBuffer(quantScaleOutQueue, DOUBLE_BUFFER,
                     AlignUp<int32_t>(gmmSwigluQuantV2->maxProcessRowNum, ALIGN_8_ELE) * sizeof(float));
    // two 32 byte buffer for reduceMax calculation in Quant.
    pipe->InitBuffer(reduceWorkspace, gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR * sizeof(float) +
                                          UB_BLOCK_UNIT_SIZE + UB_BLOCK_UNIT_SIZE);
}

__aicore__ inline void GMMA4W4PostProcess::Process(WorkSpaceSplitConfig &workspaceSplitConfig,
                                                   int64_t workspaceSplitLoopIdx, TPipe *pipe)
{
    if ASCEND_IS_AIV {
        if (workspaceSplitLoopIdx >= workspaceSplitConfig.loopCount || workspaceSplitLoopIdx < 0) {
            return;
        }
        VecConfig vecConfig;
        UpdateVecConfig(blockIdx, vecConfig, workspaceSplitConfig, workspaceSplitLoopIdx, pipe);

        if (blockIdx < vecConfig.usedCoreNum) {
            mmOutGM = (workspaceSplitLoopIdx % NUM_2 == 0 ? mmOutGM1 : mmOutGM2);
            const int64_t maxRow = gmmSwigluQuantV2->maxProcessRowNum;
            const uint32_t outLoopNum = vecConfig.outLoopNum;
            const int64_t tailLoopNum = vecConfig.tailLoopNum;

            // prologue: 预取 tile 0(MTE2), 使 tile1 的搬入能与 tile0 的计算重叠
            customDataCopyIn(0, mmOutGM, vecConfig, workspaceSplitConfig);

            for (uint32_t outLoopIdx = 0; outLoopIdx < outLoopNum; outLoopIdx++) {
                const bool isLast = (outLoopIdx == outLoopNum - 1);
                const int64_t curRows = isLast ? tailLoopNum : maxRow;
                const int64_t processNum = curRows * gmmSwigluQuantV2->tokenLen;
                vecConfig.innerLoopNum = curRows;

                // 1. DeQue 当前 tile(自动 MTE2->V 同步), fp16 -> fp32
                LocalTensor<half> inHalf = mmOutQueue.DeQue<half>();
                curFp32 = inHalf.ReinterpretCast<float>();
                Cast(curFp32, inHalf[processNum], RoundMode::CAST_NONE, processNum);
                PipeBarrier<PIPE_V>();

                // 2. double buffer: 预取下一个 tile 的中间结果(MTE2), 与当前 tile 的 V 计算重叠
                if (!isLast) {
                    customDataCopyIn(outLoopIdx + 1, mmOutGM, vecConfig, workspaceSplitConfig);
                }

                // 3. 当前 tile 输出 buffer 预留
                curQuant = quantOutQueue.AllocTensor<int8_t>();
                curScale = quantScaleOutQueue.AllocTensor<float>();

                // 4. 四步 vector 计算(perToken反量化、Swiglu、SmoothScale、Quant) -> curQuant/curScale
                for (uint32_t innerLoopIdx = 0; innerLoopIdx < vecConfig.innerLoopNum; innerLoopIdx++) {
                    VectorCompute(innerLoopIdx, vecConfig, workspaceSplitConfig);
                }

                quantOutQueue.EnQue(curQuant);
                quantScaleOutQueue.EnQue(curScale);
                // 归还当前 tile 的 fp32 buffer, 供更后面的 copy-in 复用
                mmOutQueue.FreeTensor(inHalf);

                // 5. MTE3 搬出当前 tile 结果(DeQue 自动 V->MTE3 同步), 与下一个 tile 计算重叠
                customDataCopyOut(vecConfig, workspaceSplitConfig);
            }
        }
    }
}

__aicore__ inline void GMMA4W4PostProcess::customDataCopyOut(VecConfig &vecConfig,
                                                              WorkSpaceSplitConfig &workspaceSplitConfig)
{
    LocalTensor<int8_t> quantLocal = quantOutQueue.DeQue<int8_t>();
    LocalTensor<float> quantScaleLocal = quantScaleOutQueue.DeQue<float>();
    DataCopyParams copyParamsScale{1, (uint16_t)(vecConfig.innerLoopNum * sizeof(float)), 0, 0};
    DataCopyPad(quantScaleOutputGM[workspaceSplitConfig.leftMatrixStartIndex + vecConfig.startIdx], quantScaleLocal,
                copyParamsScale);
    DataCopyParams copyParamsQuant{
        1, (uint16_t)(vecConfig.innerLoopNum * gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR * sizeof(int8_t)), 0,
        0};
    DataCopyPad(quantOutputGM[(workspaceSplitConfig.leftMatrixStartIndex + vecConfig.startIdx) *
                              gmmSwigluQuantV2->tokenLen / SWIGLU_REDUCE_FACTOR],
                quantLocal, copyParamsQuant);
    quantOutQueue.FreeTensor(quantLocal);
    quantScaleOutQueue.FreeTensor(quantScaleLocal);

    vecConfig.startIdx += vecConfig.innerLoopNum;
    vecConfig.startOffset = vecConfig.startIdx * gmmSwigluQuantV2->tokenLen;
}

} // namespace GroupedMatmulDequantSwigluQuant
#endif // GMM_SWIGLU_QUANT_V2_A4W4
#endif // OP_KERNEL_GROUPED_MATMUL_SWIGLU_QUANT_V2_A4W4_POST_H
