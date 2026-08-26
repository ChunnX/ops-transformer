/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_rotary_pos_emb_grad_ab.h
 * \brief AB template kernel (SBND layout / BSND+cosb>1, DxTilingKey=204)
 *
 * 核间: BS 合并轴 × N 轴 二维分核 (blockIdx -> (blockDimBS, blockDimN))
 * 核内: 外层 BS 循环搬 cos/sin(Q/K 双路复用), 内层 N 循环搬 grad、算 dx
 *   - cosb>1: bs=b*s, 每行 N=nQ/nK
 *   - cosb==1: bs=s, B 合并进 N, 每行 N=b*nQ/b*nK (tiling 已把统一后的 totalN 填入 nQ/nK)
 * 数据以扁平 bs_idx 线性访问, BSND/SBND 自动一致 (grad 与 cos 同布局)。
 * 搬运统一按 D/2 拆分：每个半行独立 32B padding 对齐，
 * cos/sin/grad/x/dx/部分积均通过一次 DataCopyPad 按半行搬运，不走整行连续搬运。
 * 计算风格与 A/单路 AB 一致: HalfGradAlignVF 使用半行对齐步长。
 * dCosFlag=1: 部分积 SET/ACCUMULATE 择路、UB 行距固定 ubFactorN_、写回走队列同步 (与 BAB 对齐)。
 */

#ifndef APPLY_ROTARY_POS_EMB_GRAD_AB_H
#define APPLY_ROTARY_POS_EMB_GRAD_AB_H

#include "apply_rotary_pos_emb_grad_common.h"
#include "apply_rotary_pos_emb_grad_tiling_data.h"

namespace ApplyRotaryPosEmbGrad {
using namespace AscendC;

// kDcosFlag: 编译期是否内联计算 grad_cos/grad_sin 部分积 (apt.cpp 由模板参数 DcosFlag 传入)。
// ApplyRopeGradRegbaseABParams 不含 dCosFlag 字段, 故用模板参数, 与 A 模板 <T, IsDcos> 模式一致。
template <typename T, bool kDcosFlag>
class ApplyRotaryPosEmbGradAB
{
public:
    __aicore__ inline ApplyRotaryPosEmbGradAB(TPipe* pipe, const ApplyRopeGradRegbaseABParams* tiling)
        : pipe_(pipe), tilingData_(tiling) {}
    // cosWs/sinWs: grad_cos/grad_sin 部分积 workspace 基址 (DcosFlag=1 时由 apt.cpp 传入,
    // Phase2 reduce 从 workspace 读取, 与 BAB 模板一致)
    __aicore__ inline void Init(
        GM_ADDR gradQueryEmbed, GM_ADDR gradKeyEmbed,
        GM_ADDR cos, GM_ADDR sin,
        GM_ADDR query, GM_ADDR key,
        GM_ADDR gradQueryOut, GM_ADDR gradKeyOut,
        GM_ADDR cosWs, GM_ADDR sinWs);
    __aicore__ inline void Process();

private:
    constexpr static int32_t bufferNum = 2;

    const ApplyRopeGradRegbaseABParams* tilingData_;
    TPipe* pipe_;

    GlobalTensor<T> gradQGm_;
    GlobalTensor<T> gradKGm_;
    GlobalTensor<T> cosGm_;
    GlobalTensor<T> sinGm_;
    GlobalTensor<T> queryGm_;
    GlobalTensor<T> keyGm_;
    GlobalTensor<T> gradQOutGm_;
    GlobalTensor<T> gradKOutGm_;
    GlobalTensor<float> cosWsGm_;
    GlobalTensor<float> sinWsGm_;

    TQue<QuePosition::VECIN, bufferNum> gradQInQue_;
    TQue<QuePosition::VECIN, bufferNum> gradKInQue_;
    TQue<QuePosition::VECIN, bufferNum> cosInQue_;
    TQue<QuePosition::VECIN, bufferNum> sinInQue_;
    TQue<QuePosition::VECOUT, bufferNum> gradQOutQue_;
    TQue<QuePosition::VECOUT, bufferNum> gradKOutQue_;

    TQue<QuePosition::VECIN, bufferNum> queryInQue_;
    TQue<QuePosition::VECIN, bufferNum> keyInQue_;
    TQue<QuePosition::VECOUT, bufferNum> gradCosOutQue_;
    TQue<QuePosition::VECOUT, bufferNum> gradSinOutQue_;

    int64_t dAlign_ = 0;
    int64_t partialDAlign_ = 0; // float partial 的 UB 行距
    int64_t ubFactorBS_ = 0;
    int64_t ubFactorN_ = 0;

private:
    __aicore__ inline void ProcessNLoop(int64_t bsAbsIdx, int64_t nStart, uint32_t currBS, int64_t nBlockCount);
    __aicore__ inline void ProcessN(
        const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor,
        GlobalTensor<T>& gradXGm, GlobalTensor<T>& gradXOutGm, GlobalTensor<T>& xGm,
        TQue<QuePosition::VECIN, bufferNum>& gradXInQue, TQue<QuePosition::VECOUT, bufferNum>& gradXOutQue,
        TQue<QuePosition::VECIN, bufferNum>& xInQue,
        int64_t gradGmOffset, int64_t currXN, uint32_t currBS,
        LocalTensor<float>& cosPartial, LocalTensor<float>& sinPartial, bool isAccumulate);
    __aicore__ inline void ComputeOnePath(
        const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor,
        const LocalTensor<T>& inTensor, const LocalTensor<T>& outTensor,
        uint32_t currBS, uint32_t currDNum);
    __aicore__ inline void ComputeGradWithCosSinVF(
        const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor,
        const LocalTensor<T>& gradIn, const LocalTensor<T>& gradOut, const LocalTensor<T>& xIn,
        const LocalTensor<float>& cosPartial, const LocalTensor<float>& sinPartial,
        uint16_t currSNum, uint16_t currDNum, bool isAccumulate);
};

// =================================================================
// Init
// =================================================================
template <typename T, bool kDcosFlag>
__aicore__ inline void ApplyRotaryPosEmbGradAB<T, kDcosFlag>::Init(
    GM_ADDR gradQueryEmbed, GM_ADDR gradKeyEmbed,
    GM_ADDR cos, GM_ADDR sin,
    GM_ADDR query, GM_ADDR key,
    GM_ADDR gradQueryOut, GM_ADDR gradKeyOut,
    GM_ADDR cosWs, GM_ADDR sinWs)
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    // 按 D/2 搬运，每个半行独立 32B 对齐，UB 行步长由 host 侧统一下发。
    dAlign_ = tilingData_->dAlign;
    ubFactorN_ = tilingData_->ubFactorN;
    ubFactorBS_ = tilingData_->ubFactorBS;

    gradQGm_.SetGlobalBuffer((__gm__ T*)gradQueryEmbed);
    gradKGm_.SetGlobalBuffer((__gm__ T*)gradKeyEmbed);
    cosGm_.SetGlobalBuffer((__gm__ T*)cos);
    sinGm_.SetGlobalBuffer((__gm__ T*)sin);
    queryGm_.SetGlobalBuffer((__gm__ T*)query);
    keyGm_.SetGlobalBuffer((__gm__ T*)key);
    gradQOutGm_.SetGlobalBuffer((__gm__ T*)gradQueryOut);
    gradKOutGm_.SetGlobalBuffer((__gm__ T*)gradKeyOut);
    cosWsGm_.SetGlobalBuffer((__gm__ float*)cosWs);
    sinWsGm_.SetGlobalBuffer((__gm__ float*)sinWs);

    // UB: cos/sin 各 ubFactorBS 行; grad/dx 各 ubFactorBS*ubFactorN 行 (双缓冲)
    pipe_->InitBuffer(cosInQue_, bufferNum, ubFactorBS_ * dAlign_ * sizeof(T));
    pipe_->InitBuffer(sinInQue_, bufferNum, ubFactorBS_ * dAlign_ * sizeof(T));
    pipe_->InitBuffer(gradQInQue_, bufferNum, ubFactorBS_ * ubFactorN_ * dAlign_ * sizeof(T));
    pipe_->InitBuffer(gradKInQue_, bufferNum, ubFactorBS_ * ubFactorN_ * dAlign_ * sizeof(T));
    pipe_->InitBuffer(gradQOutQue_, bufferNum, ubFactorBS_ * ubFactorN_ * dAlign_ * sizeof(T));
    pipe_->InitBuffer(gradKOutQue_, bufferNum, ubFactorBS_ * ubFactorN_ * dAlign_ * sizeof(T));

    if constexpr (kDcosFlag) {
        partialDAlign_ = Ops::Base::CeilAlign<int64_t>(
            tilingData_->d / HALF_COEF, BLOCK_TYPE_SIZE / sizeof(float)) * HALF_COEF;
        // query/key/gradCos/gradSin 与 grad 同形 (ubFactorBS*ubFactorN*dAlign), 仅 dCosFlag=1 时分配
        // 注意: nQ/nK 已是统一后的 totalN, 这里按 maxN 分配保证两路都够用
        pipe_->InitBuffer(queryInQue_, bufferNum, ubFactorBS_ * ubFactorN_ * dAlign_ * sizeof(T));
        pipe_->InitBuffer(keyInQue_, bufferNum, ubFactorBS_ * ubFactorN_ * dAlign_ * sizeof(T));
        pipe_->InitBuffer(gradCosOutQue_, bufferNum, ubFactorBS_ * ubFactorN_ * partialDAlign_ * sizeof(float));
        pipe_->InitBuffer(gradSinOutQue_, bufferNum, ubFactorBS_ * ubFactorN_ * partialDAlign_ * sizeof(float));
    }
}

// =================================================================
// Process — 外层 BS 循环, 每段搬一次 cos/sin, 内层交给 ProcessNLoop
// =================================================================
template <typename T, bool kDcosFlag>
__aicore__ inline void ApplyRotaryPosEmbGradAB<T, kDcosFlag>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) {
        return;
    }
    int64_t blockDimN = GetBlockIdx() % tilingData_->blockNumN;
    int64_t blockDimBS = GetBlockIdx() / tilingData_->blockNumN;
    int64_t bsBlockCount = (blockDimBS == tilingData_->blockNumBS - 1) ? tilingData_->blockTailBS
                                                                       : tilingData_->blockFactorBS;
    int64_t nBlockCount = (blockDimN == tilingData_->blockNumN - 1) ? tilingData_->blockTailN
                                                                    : tilingData_->blockFactorN;
    int64_t bsStart = blockDimBS * tilingData_->blockFactorBS;
    int64_t nStart = blockDimN * tilingData_->blockFactorN;
    uint32_t bsLoopCnt = Ops::Base::CeilDiv(bsBlockCount, ubFactorBS_);
    for (uint32_t bsLoop = 0; bsLoop < bsLoopCnt; bsLoop++) {
        uint32_t currBS = (bsLoop != bsLoopCnt - 1)
                          ? static_cast<uint32_t>(ubFactorBS_)
                          : static_cast<uint32_t>(bsBlockCount - bsLoop * ubFactorBS_);
        int64_t bsAbsIdx = bsStart + bsLoop * ubFactorBS_;
        ProcessNLoop(bsAbsIdx, nStart, currBS, nBlockCount);
    }
}

// =================================================================
// ProcessNLoop — 搬 cos/sin (被 Q/K 复用), 内层 N 循环 tile-interleaved Q/K
// =================================================================
template <typename T, bool kDcosFlag>
__aicore__ inline void ApplyRotaryPosEmbGradAB<T, kDcosFlag>::ProcessNLoop(
    int64_t bsAbsIdx, int64_t nStart, uint32_t currBS, int64_t nBlockCount)
{
    int64_t d = tilingData_->d;
    int64_t dHalf = d / HALF_COEF;
    int64_t dHalfByteSize = dHalf * sizeof(T);

    // 搬入本 BS 段的 cos/sin (currBS 行 × d)
    LocalTensor<T> sinTensor = sinInQue_.AllocTensor<T>();
    LocalTensor<T> cosTensor = cosInQue_.AllocTensor<T>();
    DataCopyExtParams copyCosParams = {static_cast<uint16_t>(currBS * HALF_COEF),
                                       static_cast<uint32_t>(dHalfByteSize), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(sinTensor, sinGm_[bsAbsIdx * d], copyCosParams, padParams);
    DataCopyPad(cosTensor, cosGm_[bsAbsIdx * d], copyCosParams, padParams);
    sinInQue_.EnQue(sinTensor);
    cosInQue_.EnQue(cosTensor);
    sinTensor = sinInQue_.DeQue<T>();
    cosTensor = cosInQue_.DeQue<T>();

    uint32_t nLoopCnt = Ops::Base::CeilDiv(nBlockCount, ubFactorN_);
    for (uint32_t nLoop = 0; nLoop < nLoopCnt; nLoop++) {
        int64_t absN = nStart + nLoop * ubFactorN_;
        // Q/K 各自按统一后的 totalN(nQ/nK) 裁尾
        int64_t currQN = (absN < tilingData_->nQ)
            ? min(ubFactorN_, tilingData_->nQ - absN) : 0;
        int64_t currKN = (absN < tilingData_->nK)
            ? min(ubFactorN_, tilingData_->nK - absN) : 0;
        if (currQN <= 0 && currKN <= 0) {
            break;
        }

        LocalTensor<float> cosPartial, sinPartial;
        if constexpr (kDcosFlag) {
            cosPartial = gradCosOutQue_.AllocTensor<float>();
            sinPartial = gradSinOutQue_.AllocTensor<float>();
        }

        // Q/K tile：先 SET head 数更多的一路（覆盖整个 cos/sin 部分积范围），
        // 再 ACCUMULATE 较小的一路，避免 nQ<nK 时 K 累加读到未初始化区域 (与 BAB 一致)
        if (currQN >= currKN) {
            if (currQN > 0) {
                int64_t qGmOffset = bsAbsIdx * tilingData_->nQ * d + absN * d;
                ProcessN(sinTensor, cosTensor, gradQGm_, gradQOutGm_, queryGm_,
                         gradQInQue_, gradQOutQue_, queryInQue_,
                         qGmOffset, currQN, currBS, cosPartial, sinPartial, false);
            }
            if (currKN > 0) {
                int64_t kGmOffset = bsAbsIdx * tilingData_->nK * d + absN * d;
                ProcessN(sinTensor, cosTensor, gradKGm_, gradKOutGm_, keyGm_,
                         gradKInQue_, gradKOutQue_, keyInQue_,
                         kGmOffset, currKN, currBS, cosPartial, sinPartial, true);
            }
        } else {
            if (currKN > 0) {
                int64_t kGmOffset = bsAbsIdx * tilingData_->nK * d + absN * d;
                ProcessN(sinTensor, cosTensor, gradKGm_, gradKOutGm_, keyGm_,
                         gradKInQue_, gradKOutQue_, keyInQue_,
                         kGmOffset, currKN, currBS, cosPartial, sinPartial, false);
            }
            if (currQN > 0) {
                int64_t qGmOffset = bsAbsIdx * tilingData_->nQ * d + absN * d;
                ProcessN(sinTensor, cosTensor, gradQGm_, gradQOutGm_, queryGm_,
                         gradQInQue_, gradQOutQue_, queryInQue_,
                         qGmOffset, currQN, currBS, cosPartial, sinPartial, true);
            }
        }

        // 写出 cos/sin 部分积到 workspace (Q+K 已在 UB 内累加), 供 Phase2 reduce
        if constexpr (kDcosFlag) {
            if (currQN > 0 || currKN > 0) {
                gradCosOutQue_.EnQue(cosPartial);
                gradSinOutQue_.EnQue(sinPartial);
                cosPartial = gradCosOutQue_.DeQue<float>();
                sinPartial = gradSinOutQue_.DeQue<float>();
                // 写回范围须覆盖 SET 路的完整区间 (nQ<nK 时 K 先 SET 覆盖 [0,currKN)), 与 BAB 一致
                uint32_t wsCurrN = static_cast<uint32_t>(max(currQN, currKN));
                // workspace 部分积偏移（按统一 maxN 线性，与 Phase2 reduce 对齐）
                int64_t maxN = max(tilingData_->nQ, tilingData_->nK);
                int64_t wsOffset = bsAbsIdx * maxN * d + absN * d;
                DataCopyExtParams copyWsParams = {static_cast<uint16_t>(currBS * wsCurrN * HALF_COEF),
                                                  static_cast<uint32_t>(dHalf * sizeof(float)), 0, 0, 0};
                DataCopyPad(cosWsGm_[wsOffset], cosPartial, copyWsParams);
                DataCopyPad(sinWsGm_[wsOffset], sinPartial, copyWsParams);
                gradCosOutQue_.FreeTensor(cosPartial);
                gradSinOutQue_.FreeTensor(sinPartial);
            }
        }
    }

    sinInQue_.FreeTensor(sinTensor);
    cosInQue_.FreeTensor(cosTensor);
}

// =================================================================
// ProcessN — 单 tile 单路径: CopyIn grad (+x) -> Compute -> CopyOut dx
//   isAccumulate: true=cosPartial/sinPartial 累加模式, false=SET 初始化模式 (与 BAB 一致)
// =================================================================
template <typename T, bool kDcosFlag>
__aicore__ inline void ApplyRotaryPosEmbGradAB<T, kDcosFlag>::ProcessN(
    const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor,
    GlobalTensor<T>& gradXGm, GlobalTensor<T>& gradXOutGm, GlobalTensor<T>& xGm,
    TQue<QuePosition::VECIN, bufferNum>& gradXInQue, TQue<QuePosition::VECOUT, bufferNum>& gradXOutQue,
    TQue<QuePosition::VECIN, bufferNum>& xInQue,
    int64_t gradGmOffset, int64_t currXN, uint32_t currBS,
    LocalTensor<float>& cosPartial, LocalTensor<float>& sinPartial, bool isAccumulate)
{
    uint32_t dHalfByteSize = static_cast<uint32_t>(tilingData_->d / HALF_COEF * sizeof(T));

    // CopyIn: gradX，一次 DataCopyPad 按 D/2 拆分搬运。
    LocalTensor<T> gradXIn = gradXInQue.AllocTensor<T>();
    DataCopyExtParams copyParams = {static_cast<uint16_t>(currBS * currXN * HALF_COEF),
                                    dHalfByteSize, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(gradXIn, gradXGm[gradGmOffset], copyParams, padParams);
    gradXInQue.EnQue(gradXIn);
    gradXIn = gradXInQue.DeQue<T>();
    LocalTensor<T> gradXOut = gradXOutQue.AllocTensor<T>();

    // CopyIn: x (query/key, 仅 dCosFlag) + Compute
    LocalTensor<T> xIn;
    if constexpr (kDcosFlag) {
        xIn = xInQue.AllocTensor<T>();
        DataCopyPad(xIn, xGm[gradGmOffset], copyParams, padParams);
        xInQue.EnQue(xIn);
        xIn = xInQue.DeQue<T>();
        ComputeGradWithCosSinVF(sinTensor, cosTensor, gradXIn, gradXOut, xIn,
                                cosPartial, sinPartial, currBS, currXN, isAccumulate);
    } else {
        ComputeOnePath(sinTensor, cosTensor, gradXIn, gradXOut, currBS, currXN);
    }

    // CopyOut: dx (currBS 行 × currXN 列)
    // 注意: UB→GM 方向的 DataCopyPad 无 (dst, src, ExtParams, PadParams) 四参重载,
    // 仅支持 (dst, src, ExtParams) 三参 (与 BAB CopyOut 写法一致)
    gradXOutQue.EnQue(gradXOut);
    gradXOut = gradXOutQue.DeQue<T>();
    DataCopyPad(gradXOutGm[gradGmOffset], gradXOut, copyParams);
    gradXOutQue.FreeTensor(gradXOut);

    gradXInQue.FreeTensor(gradXIn);
    if constexpr (kDcosFlag) {
        xInQue.FreeTensor(xIn);
    }
}

// =================================================================
// ComputeOnePath — 仅 dx (dCosFlag=0)
// =================================================================
template <typename T, bool kDcosFlag>
__aicore__ inline void ApplyRotaryPosEmbGradAB<T, kDcosFlag>::ComputeOnePath(
    const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor,
    const LocalTensor<T>& inTensor, const LocalTensor<T>& outTensor,
    uint32_t currBS, uint32_t currDNum)
{
    // HalfGradAlignVF: currSNum=currBS(cos 行数), currDNum=currN(每 cos 行下的 n 行)
    HalfGradAlignVF<T>(sinTensor, cosTensor, inTensor, outTensor,
                       tilingData_->d, dAlign_, currBS, currDNum);
}

// =================================================================
// ComputeGradWithCosSinVF — 合并 dx + grad_cos/grad_sin 部分积 (dCosFlag=1)
//   isAccumulate=false (SET):  cosPartial = grad*x, sinPartial = grad*rot(x)
//   isAccumulate=true  (ADD):  cosPartial += grad*x, sinPartial += grad*rot(x)
//   (逻辑与 BAB 完全一致, 仅入参为 AB 的 currBS/currN)
// =================================================================
template <typename T, bool kDcosFlag>
__aicore__ inline void ApplyRotaryPosEmbGradAB<T, kDcosFlag>::ComputeGradWithCosSinVF(
    const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor,
    const LocalTensor<T>& gradIn, const LocalTensor<T>& gradOut, const LocalTensor<T>& xIn,
    const LocalTensor<float>& cosPartial, const LocalTensor<float>& sinPartial,
    uint16_t currSNum, uint16_t currDNum, bool isAccumulate)
{
    __local_mem__ T* sinUb = (__local_mem__ T*)sinTensor.GetPhyAddr();
    __local_mem__ T* cosUb = (__local_mem__ T*)cosTensor.GetPhyAddr();
    __local_mem__ T* gradUb = (__local_mem__ T*)gradIn.GetPhyAddr();
    __local_mem__ T* gradOutUb = (__local_mem__ T*)gradOut.GetPhyAddr();
    __local_mem__ T* xUb = (__local_mem__ T*)xIn.GetPhyAddr();
    __local_mem__ float* cosPartialUb = (__local_mem__ float*)cosPartial.GetPhyAddr();
    __local_mem__ float* sinPartialUb = (__local_mem__ float*)sinPartial.GetPhyAddr();
    uint32_t halfD = tilingData_->d / HALF_COEF;
    uint32_t halfDAlign = dAlign_ / HALF_COEF;
    uint32_t halfPartialDAlign = partialDAlign_ / HALF_COEF;
    uint16_t repeatTimes = Ops::Base::CeilDiv(halfD, VL_FLOAT32_SIZE);
    __local_mem__ T *currGradUb, *currGradOutUb, *currSinUb, *currCosUb, *currXUb;
    __local_mem__ float *currCosPartialUb, *currSinPartialUb;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vregGrad1, vregGrad2, vregCos1, vregCos2, vregSin1, vregSin2, vregX1, vregX2;
        Reg::RegTensor<float> vregTmp1, vregTmp2, vregTmp3, vregTmp4, vregTmp5, vregTmp6, vregTmp7, vregTmp8;
        Reg::MaskReg preg;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * dAlign_;
            currCosUb = cosUb + sIdx * dAlign_;
            for (uint16_t row = 0; row < currDNum; row++) {
                currGradUb = gradUb + (sIdx * currDNum + row) * dAlign_;
                currGradOutUb = gradOutUb + (sIdx * currDNum + row) * dAlign_;
                currXUb = xUb + (sIdx * currDNum + row) * dAlign_;
                // cosPartial/sinPartial 为 Q/K 共享缓冲，行距必须用固定的 ubFactorN_（而非 currDNum），
                // 否则 nQ != nK 时 Q/K 两路的行偏移不一致，累加/写回会错位 (与 BAB 一致)
                currCosPartialUb = cosPartialUb + (sIdx * ubFactorN_ + row) * partialDAlign_;
                currSinPartialUb = sinPartialUb + (sIdx * ubFactorN_ + row) * partialDAlign_;
                uint32_t updateCnt = halfD;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = Reg::UpdateMask<float>(updateCnt);
                    int32_t offset = i * VL_FLOAT32_SIZE;
                    int32_t halfOffset = offset + halfDAlign;

                    ops::LoadTwoTensorForDtypeT<T>(
                        currGradUb, currGradUb, vregGrad1, vregGrad2, preg, preg, offset, halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(
                        currCosUb, currCosUb, vregCos1, vregCos2, preg, preg, offset, halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(
                        currSinUb, currSinUb, vregSin1, vregSin2, preg, preg, offset, halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(
                        currXUb, currXUb, vregX1, vregX2, preg, preg, offset, halfOffset);

                    // dx[0:d/2] = grad[0:d/2]*cos[0:d/2] + grad[d/2:d]*sin[d/2:d]
                    Mul(vregTmp1, vregCos1, vregGrad1, preg);
                    Mul(vregTmp2, vregSin2, vregGrad2, preg);
                    Add(vregTmp1, vregTmp1, vregTmp2, preg);
                    // dx[d/2:d] = grad[d/2:d]*cos[d/2:d] - grad[0:d/2]*sin[0:d/2]
                    Mul(vregTmp3, vregCos2, vregGrad2, preg);
                    Mul(vregTmp4, vregSin1, vregGrad1, preg);
                    Sub(vregTmp3, vregTmp3, vregTmp4, preg);

                    // grad_cos = grad*x (前/后半)
                    Mul(vregTmp5, vregGrad1, vregX1, preg);
                    Mul(vregTmp6, vregGrad2, vregX2, preg);
                    // grad_sin = -(grad1*x2) + grad2*x1  (对应 rotate(x)=cat(-x2,x1))
                    Mul(vregTmp7, vregGrad1, vregX2, preg);
                    Muls(vregTmp7, vregTmp7, -1.0f, preg);
                    Mul(vregTmp8, vregGrad2, vregX1, preg);

                    if (!isAccumulate) {
                        ops::StoreOneTensorForDtypeT<T>(currGradOutUb, vregTmp1, preg, offset);
                        ops::StoreOneTensorForDtypeT<T>(currGradOutUb, vregTmp3, preg, halfOffset);
                        ops::StoreOneTensorForDtypeT<float>(currCosPartialUb, vregTmp5, preg, offset);
                        ops::StoreOneTensorForDtypeT<float>(currCosPartialUb, vregTmp6, preg, halfPartialDAlign + offset);
                        ops::StoreOneTensorForDtypeT<float>(currSinPartialUb, vregTmp7, preg, offset);
                        ops::StoreOneTensorForDtypeT<float>(currSinPartialUb, vregTmp8, preg, halfPartialDAlign + offset);
                    } else {
                        ops::StoreOneTensorForDtypeT<T>(currGradOutUb, vregTmp1, preg, offset);
                        ops::StoreOneTensorForDtypeT<T>(currGradOutUb, vregTmp3, preg, halfOffset);

                        ops::LoadTwoTensorForDtypeT<float>(
                            currCosPartialUb, currCosPartialUb, vregCos1, vregCos2, preg, preg,
                            offset, halfPartialDAlign + offset);
                        Add(vregCos1, vregCos1, vregTmp5, preg);
                        Add(vregCos2, vregCos2, vregTmp6, preg);
                        ops::StoreOneTensorForDtypeT<float>(currCosPartialUb, vregCos1, preg, offset);
                        ops::StoreOneTensorForDtypeT<float>(currCosPartialUb, vregCos2, preg, halfPartialDAlign + offset);

                        ops::LoadTwoTensorForDtypeT<float>(
                            currSinPartialUb, currSinPartialUb, vregSin1, vregSin2, preg, preg,
                            offset, halfPartialDAlign + offset);
                        Add(vregSin1, vregSin1, vregTmp7, preg);
                        Add(vregSin2, vregSin2, vregTmp8, preg);
                        ops::StoreOneTensorForDtypeT<float>(currSinPartialUb, vregSin1, preg, offset);
                        ops::StoreOneTensorForDtypeT<float>(currSinPartialUb, vregSin2, preg, halfPartialDAlign + offset);
                    }
                }
            }
        }
    }
}
} // namespace ApplyRotaryPosEmbGrad
#endif // APPLY_ROTARY_POS_EMB_GRAD_AB_H
