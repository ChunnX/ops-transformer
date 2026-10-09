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
 * \file causal_conv1d_backward.h
 * \brief CausalConv1dBackward kernel 实现（arch35，RegBase/Reg 向量核）
 *
 * 是 CausalConv1d（prefill 模式，bias=None，冷启动）的 VJP。语义（与正向 golden/vllm
 * 三方一致的权重序）：
 *   前向： pre[t] = Σ_{k=0..K-1} weight[k]·x[t-(K-1-k)]（weight[K-1] 配当前 token）
 *   反向： dout[t]        = grad_y[t] · Act'(pre[t])（Act' 内部重算 pre；none 时 ≡1）
 *          grad_x[s]      = Σ_{j=0..K-1} weight[K-1-j]·dout[s+j]（右边界截断）
 *          grad_weight[k] = Σ_t dout[t]·x[t-(K-1-k)]（左侧冷启动 pad 0 自然消亡）
 * 中间计算与累加全程 fp32，输出 cast 回输入 dtype。
 *
 * ============================ 切分与流程 ============================
 * 沿 D（通道）维切 tile（baseDim=128），核间按 tile 轮转；每个核独占若干完整
 * 通道，grad_weight 在核内完整累加（无需跨核 reduce，确定性）。
 * token 轴按 query_start_loc 划分的序列段独立处理（None 时整轴一段）：
 * 段内 T_CHUNK=64 行分块 + tail 机制（跨 chunk 边界的 dx 行延迟补算），
 * 段边界即冷启动左边界（左 pad 0），段尾右补 0。
 *
 * ============================ Reg 形态 ============================
 * 向量核位于 vf/causal_conv1d_backward_vf_compute.h（Reg::RegTensor /
 * MaskReg / LoadAlign / StoreAlign / MulAddDst，文件组织与
 * masked_causal_conv1d_backward 的 vf 同构）：
 *   PassA（逐 token 行）: z 重算（Mul + MulAddDst 4-tap 融合）+ silu'(z) 全寄存器
 *                         完成（z 不落 UB）→ grad_z 存 fp32 平面；
 *                         grad_weight 部分和寄存器内按行累加（无乘积平面、无树归并）
 *   PassB（逐 dx 行）  : dx[s] = Σ_j w[K-1-j]·grad_z[s+j]（Mul + MulAddDst 链）
 *   Tail               : gradZ 尾部 hist 行搬移（fp32 行拷贝）
 * d0 在 UB 内按 64-lane 补齐为 d0Round 行距，全 mask 计算（垃圾 lane 逐元素隔离、
 * 不出 GM）；GM 搬运只取 actualDim 有效字节。tbe 权重序修复点与 membase 版一致。
 *
 * ⚠️ 同步约束（arch35 跨流水陷阱，不可回退）：MTE2↔VEC↔MTE3 边界数据传递必须
 * TQue 队列模式（AllocTensor→DataCopyPad→EnQue→DeQue→FreeTensor）；TBuf+PipeBarrier
 * 在 Ascend950 AIV 上不保证跨流水可见性（会读到旧值，静默错位）。
 */
#ifndef CAUSAL_CONV1D_BACKWARD_H
#define CAUSAL_CONV1D_BACKWARD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "causal_conv1d_backward_tiling_data.h"
#include "causal_conv1d_backward_tiling_key.h"
#include "causal_conv1d_backward_vf_compute.h"

namespace NsCausalConv1dBackward {

using namespace AscendC;
using namespace NsCausalConv1dBackwardVF;

template <typename T>
class CausalConv1dBackwardKernel {
public:
    __aicore__ inline CausalConv1dBackwardKernel() {}
    __aicore__ inline void Init(GM_ADDR grad_y, GM_ADDR x, GM_ADDR weight, GM_ADDR query_start_loc,
                                GM_ADDR grad_x, GM_ADDR grad_weight,
                                const CausalConv1dBackwardTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessTile(int64_t dimOffset, int64_t actualDim);
    // 一个序列段 [segStart, segStart+segLen) 内的独立反向：段内 T_CHUNK 分块 + tail 机制
    __aicore__ inline void ProcessSegment(int64_t segStart, int64_t segLen, int64_t dimOffset,
                                          int64_t actualDim);
    // 2D 整段搬运：一次 DataCopyPad 搬 (rows × actualDim) 平面（GM 行距 dim_，UB 紧排）
    __aicore__ inline void CopyInTile2D(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                                        int64_t rows, int64_t actualDim);
    __aicore__ inline void CopyOutTile2D(const GlobalTensor<T>& dst, const LocalTensor<T>& src,
                                         int64_t rows, int64_t actualDim);
    // PassB 分段调度：dx 行 [tStart, tEnd) 写入 outT 行 [0, n)（含右边界按行截断）
    __aicore__ inline void ComputeDxRows(const LocalTensor<float>& gradZ,
                                         const LocalTensor<float>& wF32,
                                         const LocalTensor<float>& dxF32,
                                         int64_t tStart, int64_t tEnd, int64_t chunkStart,
                                         int64_t T0, int64_t d0, int64_t K, int64_t hist);
    // dx 分段写出（每段 ≤ OUT_ROWS 行）
    __aicore__ inline void CopyOutDx(const LocalTensor<T>& dxOut, int64_t baseGm,
                                     int64_t tStart, int64_t rows, int64_t d0);

private:
    // T 维分块行数：单 chunk 计算/搬运块 64×128 元素（bf16 16KB/段）
    static constexpr int64_t T_CHUNK = 64;
    // dx 写出分段行数
    static constexpr int64_t OUT_ROWS = 32;

    GlobalTensor<T> gradYGM_, xGM_, weightGM_;
    GlobalTensor<int32_t> qslGM_;
    GlobalTensor<T> gradXGM_, gradWeightGM_;

    TPipe pipe_;
    // fp16/bf16 staging —— 队列模式：MTE2→VEC / VEC→MTE3 边界必须走 EnQue/DeQue 事件同步。
    TQue<QuePosition::VECIN, 1> gyQue_, xQue_, wQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    // fp32 compute 平面
    TBuf<TPosition::VECCALC> xF32Buf_, gyF32Buf_, dxF32Buf_, gradZBuf_, dwAccBuf_, wF32Buf_;
    // bank 相位垫片：相邻 buffer 间错开 32B，避免跨 buffer 双源大块操作的
    // src0/src1 恒落同一 bank（实测 vec_bank_cflt 77.7% 根因）
    TBuf<TPosition::VECCALC> padA_, padB_, padC_, padD_, padE_, padF_, padG_, padH_;

    int64_t tokenCnt_ = 0;    // 展平 token 轴长度
    int64_t seqCnt_ = 0;      // 序列段数（1 = 整轴一段）
    int64_t dim_ = 0;
    int64_t kW_ = 0;
    int64_t activationMode_ = 0; // 0 none / 1 silu
    int64_t baseDim_ = 0;
    int64_t dimTileCnt_ = 0;
    int64_t usedCores_ = 0;
    int64_t blockIdx_ = 0;
};

template <typename T>
__aicore__ inline void CausalConv1dBackwardKernel<T>::Init(
    GM_ADDR grad_y, GM_ADDR x, GM_ADDR weight, GM_ADDR query_start_loc,
    GM_ADDR grad_x, GM_ADDR grad_weight,
    const CausalConv1dBackwardTilingData* tilingData)
{
    tokenCnt_ = tilingData->tokenCnt;
    seqCnt_ = tilingData->seqCnt;
    dim_ = tilingData->dim;
    kW_ = tilingData->kW;
    activationMode_ = tilingData->activationMode;
    baseDim_ = tilingData->baseDim;
    dimTileCnt_ = tilingData->dimTileCnt;
    usedCores_ = tilingData->usedCores;
    blockIdx_ = GetBlockIdx();

    gradYGM_.SetGlobalBuffer((__gm__ T*)grad_y);
    xGM_.SetGlobalBuffer((__gm__ T*)x);
    weightGM_.SetGlobalBuffer((__gm__ T*)weight);
    qslGM_.SetGlobalBuffer((__gm__ int32_t*)query_start_loc);
    gradXGM_.SetGlobalBuffer((__gm__ T*)grad_x);
    gradWeightGM_.SetGlobalBuffer((__gm__ T*)grad_weight);

    // UB 预算（bd=128, T_CHUNK=64, hist≤3, bf16）：
    //   staging fp16 ~52KB + fp32 平面 ~137KB + outQue 16KB + 垫片 ≈ 210KB < 248KB
    //   各 fp32 平面尾部加 V_LENGTH lane，承接 64-lane 分组装载的尾组越界读
    int64_t bd = baseDim_;
    int64_t hist = kW_ - 1;                    // x 历史/左 pad 行数（1~3）
    int64_t xRowsMax = T_CHUNK + hist;         // x 每 chunk 行数（历史 + 本 chunk）
    int64_t planeF16 = T_CHUNK * bd * sizeof(T);
    int64_t xPlaneF16 = xRowsMax * bd * sizeof(T);
    int64_t wF16Sz = kW_ * bd * sizeof(T);
    int64_t tailF32 = V_LENGTH * sizeof(float);
    int64_t planeF32 = T_CHUNK * bd * sizeof(float) + tailF32;
    int64_t xPlaneF32 = xRowsMax * bd * sizeof(float) + tailF32;
    int64_t gzRowsMax = T_CHUNK + 2 * (kW_ - 1) + 1;  // gradZ：tail + 当前 + 段末扩展（K-1 行）+ 1
    int64_t gzPlaneF32 = gzRowsMax * bd * sizeof(float) + tailF32;
    int64_t wF32Sz = kW_ * bd * sizeof(float) + tailF32;
    int64_t dwAccSz = kW_ * bd * sizeof(float) + tailF32;

    // bank 相位垫片布局：相邻计算 buffer 间错开 32B
    pipe_.InitBuffer(gyQue_, 1, planeF16);
    pipe_.InitBuffer(padA_, 32);
    pipe_.InitBuffer(xQue_, 1, xPlaneF16);
    pipe_.InitBuffer(padB_, 32);
    pipe_.InitBuffer(wQue_, 1, wF16Sz);
    pipe_.InitBuffer(padC_, 32);
    pipe_.InitBuffer(outQue_, 1, planeF16);

    pipe_.InitBuffer(xF32Buf_, xPlaneF32);
    pipe_.InitBuffer(padD_, 32);
    pipe_.InitBuffer(gyF32Buf_, planeF32);
    pipe_.InitBuffer(padE_, 32);
    pipe_.InitBuffer(dxF32Buf_, planeF32);
    pipe_.InitBuffer(padF_, 32);
    pipe_.InitBuffer(gradZBuf_, gzPlaneF32);
    pipe_.InitBuffer(padG_, 32);
    pipe_.InitBuffer(dwAccBuf_, dwAccSz);
    pipe_.InitBuffer(padH_, 32);
    pipe_.InitBuffer(wF32Buf_, wF32Sz);
}

template <typename T>
__aicore__ inline void CausalConv1dBackwardKernel<T>::CopyInTile2D(
    const LocalTensor<T>& dst, const GlobalTensor<T>& src, int64_t rows, int64_t actualDim)
{
    // 一次 2D DataCopyPad 搬整段平面（替代逐行 DataCopy 循环）。
    // 参数单位均为字节：blockLen=每行有效字节，srcStride=GM 每行后额外跳过，dstStride=0（UB 紧排）。
    DataCopyExtParams cp(static_cast<uint16_t>(rows),
                         static_cast<uint32_t>(actualDim * static_cast<int64_t>(sizeof(T))),
                         (dim_ - actualDim) * static_cast<int64_t>(sizeof(T)),
                         0, 0);
    DataCopyPadExtParams<T> pad(false, 0, 0, static_cast<T>(0));
    DataCopyPad(dst, src, cp, pad);
}

template <typename T>
__aicore__ inline void CausalConv1dBackwardKernel<T>::CopyOutTile2D(
    const GlobalTensor<T>& dst, const LocalTensor<T>& src, int64_t rows, int64_t actualDim)
{
    // UB→GM：srcStride=0（UB 紧排），dstStride=GM 每行后额外跳过
    DataCopyExtParams cp(static_cast<uint16_t>(rows),
                         static_cast<uint32_t>(actualDim * static_cast<int64_t>(sizeof(T))),
                         0,
                         (dim_ - actualDim) * static_cast<int64_t>(sizeof(T)), 0);
    DataCopyPad(dst, src, cp);
}

template <typename T>
__aicore__ inline void CausalConv1dBackwardKernel<T>::Process()
{
    for (int64_t tile = blockIdx_; tile < dimTileCnt_; tile += usedCores_) {
        int64_t dimOffset = tile * baseDim_;
        int64_t actualDim = baseDim_;
        if (dimOffset + actualDim > dim_) {
            actualDim = dim_ - dimOffset;
        }
        if (actualDim <= 0) {
            continue;
        }
        ProcessTile(dimOffset, actualDim);
    }
}

template <typename T>
__aicore__ inline void CausalConv1dBackwardKernel<T>::ProcessTile(int64_t dimOffset, int64_t actualDim)
{
    LocalTensor<float> dwAcc = dwAccBuf_.Get<float>();
    LocalTensor<float> wF32 = wF32Buf_.Get<float>();
    int64_t D0 = actualDim;
    int64_t K = kW_;

    // ========== 阶段 0: 加载常驻参数 weight（队列模式：EnQue/DeQue 保证 MTE2→VEC 可见性） ==========
    LocalTensor<T> wF16 = wQue_.template AllocTensor<T>();
    CopyInTile2D(wF16, weightGM_[dimOffset], K, D0);
    wQue_.EnQue(wF16);
    LocalTensor<T> wF16v = wQue_.template DeQue<T>();
    Cast(wF32, wF16v, RoundMode::CAST_NONE, static_cast<int32_t>(K * D0));
    PipeBarrier<PIPE_V>();
    wQue_.FreeTensor(wF16v);

    // 初始化 grad_weight 累加区（跨段累加）
    Duplicate(dwAcc, 0.0f, static_cast<int32_t>(K * D0));
    PipeBarrier<PIPE_V>();

    // ========== 逐序列段处理（qsl 划段；None / 单段时整轴一段） ==========
    if (seqCnt_ <= 1) {
        ProcessSegment(0, tokenCnt_, dimOffset, actualDim);
    } else {
        int64_t lo = qslGM_.GetValue(0);
        for (int64_t i = 1; i <= seqCnt_; ++i) {
            int64_t hi = qslGM_.GetValue(i);
            if (hi > lo) {
                ProcessSegment(lo, hi - lo, dimOffset, actualDim);
            }
            lo = hi;
        }
    }

    // ========== 写出 grad_weight（跨段累加结果，队列模式） ==========
    LocalTensor<T> outDw = outQue_.template AllocTensor<T>();
    Cast(outDw, dwAcc, RoundMode::CAST_RINT, static_cast<int32_t>(K * D0));
    PipeBarrier<PIPE_V>();
    outQue_.EnQue(outDw);
    LocalTensor<T> outDwV = outQue_.template DeQue<T>();
    CopyOutTile2D(gradWeightGM_[dimOffset], outDwV, K, D0);
    outQue_.FreeTensor(outDwV);
}

template <typename T>
__aicore__ inline void CausalConv1dBackwardKernel<T>::ProcessSegment(
    int64_t segStart, int64_t segLen, int64_t dimOffset, int64_t actualDim)
{
    LocalTensor<float> xF32 = xF32Buf_.Get<float>();
    LocalTensor<float> gyF32 = gyF32Buf_.Get<float>();
    LocalTensor<float> dxF32 = dxF32Buf_.Get<float>();
    LocalTensor<float> gradZ = gradZBuf_.Get<float>();
    LocalTensor<float> dwAcc = dwAccBuf_.Get<float>();
    LocalTensor<float> wF32 = wF32Buf_.Get<float>();

    int64_t T0 = segLen;
    int64_t D0 = actualDim;
    int64_t K = kW_;
    int64_t hist = K - 1;  // 左侧冷启动 pad 行数
    int64_t baseGm = segStart * dim_ + dimOffset;
    uint32_t groupCnt = static_cast<uint32_t>(D0 / static_cast<int64_t>(V_LENGTH));
    uint32_t tailLanes = static_cast<uint32_t>(D0 - static_cast<int64_t>(groupCnt) * V_LENGTH);

    // grad_z tail 区行 [0, hist) 仅在 s >= hist 的 chunk 被读（补算行 t >= s-hist），
    // 段首 chunk（s=0）不读 tail → 段边界天然冷启动，无跨段污染。
    for (int64_t s = 0; s < T0; s += T_CHUNK) {
        int64_t e = s + T_CHUNK;
        if (e > T0) {
            e = T0;
        }
        int64_t rows = e - s;              // 本 chunk 行数

        // --- 搬入 grad_y[s..e) 与 x[s-h..e)（队列模式，一次 2D DataCopyPad/平面） ---
        LocalTensor<T> gyT = gyQue_.template AllocTensor<T>();
        LocalTensor<T> xT = xQue_.template AllocTensor<T>();
        CopyInTile2D(gyT, gradYGM_[baseGm + s * dim_], rows, D0);
        if (s >= hist) {
            CopyInTile2D(xT, xGM_[baseGm + (s - hist) * dim_], rows + hist, D0);
        } else {
            // 段首 chunk：GM 无历史行，当前 chunk 行搬入 staging 的 [hist, hist+rows)，
            // cast 前将 xT 前 hist 行清零（左 pad 语义 = 0，使 t-j<0 的乘项自然为 0）
            CopyInTile2D(xT[hist * D0], xGM_[baseGm], rows, D0);
        }
        gyQue_.EnQue(gyT);
        xQue_.EnQue(xT);
        LocalTensor<T> gyV = gyQue_.template DeQue<T>();
        LocalTensor<T> xV = xQue_.template DeQue<T>();

        if (s < hist) {
            Duplicate(xV, static_cast<T>(0), static_cast<int32_t>(hist * D0));
            PipeBarrier<PIPE_V>();
        }

        // ========== B16 → fp32 扩位（UB 层 membase Cast，行紧排 D0） ==========
        Cast(gyF32, gyV, RoundMode::CAST_NONE, static_cast<int32_t>(rows * D0));
        Cast(xF32, xV, RoundMode::CAST_NONE, static_cast<int32_t>((rows + hist) * D0));
        PipeBarrier<PIPE_V>();
        if (s < hist) {
            // 冷启动：xF32 前 hist 行清零（左 pad 语义 = 0）
            Duplicate(xF32, 0.0f, static_cast<int32_t>(hist * D0));
            PipeBarrier<PIPE_V>();
        }

        // ========== PassA: z 重算 + silu' + grad_z + grad_weight 寄存器累加（fp32 平面） ==========
        __ubuf__ float *xP = reinterpret_cast<__ubuf__ float *>(xF32.GetPhyAddr());
        __ubuf__ float *gyP = reinterpret_cast<__ubuf__ float *>(gyF32.GetPhyAddr());
        __ubuf__ float *wP = reinterpret_cast<__ubuf__ float *>(wF32.GetPhyAddr());
        __ubuf__ float *gzP = reinterpret_cast<__ubuf__ float *>(gradZ.GetPhyAddr());
        __ubuf__ float *dwP = reinterpret_cast<__ubuf__ float *>(dwAcc.GetPhyAddr());
        uint16_t rowsU16 = static_cast<uint16_t>(rows);
        uint16_t groupCntU16 = groupCnt;
        uint32_t d0U32 = static_cast<uint32_t>(D0);
        int32_t histI32 = static_cast<int32_t>(hist);
        const bool act = (activationMode_ != 0);
        if (K == 2) {
            if (act) {
                asc_vf_call<NsCausalConv1dBackwardVF::ConvBwdGradZVfImpl<T, 2, true>>(
                    xP, gyP, wP, gzP, dwP, rowsU16, groupCntU16, d0U32, histI32);
            } else {
                asc_vf_call<NsCausalConv1dBackwardVF::ConvBwdGradZVfImpl<T, 2, false>>(
                    xP, gyP, wP, gzP, dwP, rowsU16, groupCntU16, d0U32, histI32);
            }
        } else if (K == 3) {
            if (act) {
                asc_vf_call<NsCausalConv1dBackwardVF::ConvBwdGradZVfImpl<T, 3, true>>(
                    xP, gyP, wP, gzP, dwP, rowsU16, groupCntU16, d0U32, histI32);
            } else {
                asc_vf_call<NsCausalConv1dBackwardVF::ConvBwdGradZVfImpl<T, 3, false>>(
                    xP, gyP, wP, gzP, dwP, rowsU16, groupCntU16, d0U32, histI32);
            }
        } else {
            if (act) {
                asc_vf_call<NsCausalConv1dBackwardVF::ConvBwdGradZVfImpl<T, 4, true>>(
                    xP, gyP, wP, gzP, dwP, rowsU16, groupCntU16, d0U32, histI32);
            } else {
                asc_vf_call<NsCausalConv1dBackwardVF::ConvBwdGradZVfImpl<T, 4, false>>(
                    xP, gyP, wP, gzP, dwP, rowsU16, groupCntU16, d0U32, histI32);
            }
        }

        // 段末扩展行清零（PassB 全 K tap 无分支读取：行 [hist+rows, hist+rows+K-1)）
        Duplicate(gradZ[(hist + rows) * D0], 0.0f, static_cast<int32_t>((K - 1) * D0));
        PipeBarrier<PIPE_V>();

        gyQue_.FreeTensor(gyV);
        xQue_.FreeTensor(xV);

        // ========== PassB + 写出：dx（大块形态，队列模式写出） ==========
        // 4.1 补算上一 chunk 延迟的 dx 行 [s-hist, s)（仅 s >= hist 的 chunk）：
        //     依赖 grad_z 行 [s-hist, s)（tail 区）∪ [s, s+hist)（当前 chunk 头部）
        if (s >= hist && hist > 0) {
            ComputeDxRows(gradZ, wF32, dxF32, s - hist, s, s, T0, D0, K, hist);
            PipeBarrier<PIPE_V>();
            LocalTensor<T> dxOut = outQue_.template AllocTensor<T>();
            Cast(dxOut, dxF32, RoundMode::CAST_RINT, static_cast<int32_t>(hist * D0));
            PipeBarrier<PIPE_V>();
            outQue_.EnQue(dxOut);
            LocalTensor<T> dxV = outQue_.template DeQue<T>();
            CopyOutDx(dxV, baseGm, s - hist, hist, D0);
            outQue_.FreeTensor(dxV);
        }

        // 4.2 本 chunk 常规 dx 行 [s, e-hist)（依赖 grad_z 至 e-1，全部已在当前 chunk 区）
        if (rows > hist) {
            ComputeDxRows(gradZ, wF32, dxF32, s, e - hist, s, T0, D0, K, hist);
            PipeBarrier<PIPE_V>();
            LocalTensor<T> dxOut = outQue_.template AllocTensor<T>();
            Cast(dxOut, dxF32, RoundMode::CAST_RINT, static_cast<int32_t>((rows - hist) * D0));
            PipeBarrier<PIPE_V>();
            outQue_.EnQue(dxOut);
            LocalTensor<T> dxV = outQue_.template DeQue<T>();
            CopyOutDx(dxV, baseGm, s, rows - hist, D0);
            outQue_.FreeTensor(dxV);
        }

        // 4.3 段终局：最后一个 chunk 的延迟行 [max(s, e-hist), e)（t+j >= T0 截断）
        if (e == T0) {
            int64_t pendStart = (e - hist > s) ? (e - hist) : s;
            ComputeDxRows(gradZ, wF32, dxF32, pendStart, e, s, T0, D0, K, hist);
            PipeBarrier<PIPE_V>();
            LocalTensor<T> dxOut = outQue_.template AllocTensor<T>();
            Cast(dxOut, dxF32, RoundMode::CAST_RINT, static_cast<int32_t>((e - pendStart) * D0));
            PipeBarrier<PIPE_V>();
            outQue_.EnQue(dxOut);
            LocalTensor<T> dxV = outQue_.template DeQue<T>();
            CopyOutDx(dxV, baseGm, pendStart, e - pendStart, D0);
            outQue_.FreeTensor(dxV);
        }

        // ========== 阶段 5: 更新 tail：本 chunk 尾部 hist 行 grad_z 拷到行 [0, hist) ==========
        // （rows < hist 仅出现在段尾短 chunk，此后无下一 chunk，tail 不再被访问）
        if (rows >= hist && hist > 0) {
            asc_vf_call<NsCausalConv1dBackwardVF::ConvBwdCopyGzTailVfImpl>(
                gzP, static_cast<uint16_t>(rows), static_cast<uint16_t>(hist),
                static_cast<uint32_t>(D0), static_cast<uint16_t>(groupCnt));
        }
    }
}

template <typename T>
__aicore__ inline void CausalConv1dBackwardKernel<T>::ComputeDxRows(
    const LocalTensor<float>& gradZ, const LocalTensor<float>& wF32, const LocalTensor<float>& dxF32,
    int64_t tStart, int64_t tEnd, int64_t chunkStart, int64_t T0, int64_t d0, int64_t K,
    int64_t hist)
{
    // 计算 dx 行 [tStart, tEnd)（n 行）写到 dxF32 行 [0, n)。
    // dx[s] = Σ_{j: s+j<T0} grad_z[s+j] · weight[K-1-j]（PassB 内逐行截断）。
    // grad_z 源行定位：绝对行 t+j → gradZ 行 (t+j-chunkStart+hist)，
    //   tail 区 [0,hist) ∪ 当前区 [hist, hist+rows) 连续。
    __ubuf__ float *gzP = reinterpret_cast<__ubuf__ float *>(gradZ.GetPhyAddr());
    __ubuf__ float *wP = reinterpret_cast<__ubuf__ float *>(wF32.GetPhyAddr());
    __ubuf__ float *dxP = reinterpret_cast<__ubuf__ float *>(dxF32.GetPhyAddr());
    int32_t srcBaseI32 = static_cast<int32_t>(tStart - chunkStart + hist);
    uint16_t rowsU16 = static_cast<uint16_t>(tEnd - tStart);
    uint16_t groupCntU16 = static_cast<uint16_t>(d0 / static_cast<int64_t>(V_LENGTH));
    uint32_t d0U32 = static_cast<uint32_t>(d0);
    if (K == 2) {
        asc_vf_call<NsCausalConv1dBackwardVF::ConvBwdGradXVfImpl<T, 2>>(
            gzP, wP, dxP, srcBaseI32, rowsU16, groupCntU16, d0U32);
    } else if (K == 3) {
        asc_vf_call<NsCausalConv1dBackwardVF::ConvBwdGradXVfImpl<T, 3>>(
            gzP, wP, dxP, srcBaseI32, rowsU16, groupCntU16, d0U32);
    } else {
        asc_vf_call<NsCausalConv1dBackwardVF::ConvBwdGradXVfImpl<T, 4>>(
            gzP, wP, dxP, srcBaseI32, rowsU16, groupCntU16, d0U32);
    }
}

template <typename T>
__aicore__ inline void CausalConv1dBackwardKernel<T>::CopyOutDx(
    const LocalTensor<T>& dxOut, int64_t baseGm, int64_t tStart, int64_t rows, int64_t D0)
{
    if (rows <= 0) {
        return;
    }
    // 分段写出：每段 ≤ OUT_ROWS 行（dxOut 行紧排 D0）
    for (int64_t seg = 0; seg < rows; seg += OUT_ROWS) {
        int64_t segRows = rows - seg;
        if (segRows > OUT_ROWS) {
            segRows = OUT_ROWS;
        }
        CopyOutTile2D(gradXGM_[baseGm + (tStart + seg) * dim_], dxOut[seg * D0], segRows, D0);
    }
}

} // namespace NsCausalConv1dBackward
#endif // CAUSAL_CONV1D_BACKWARD_H
