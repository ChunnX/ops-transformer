/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_grad.h
 * \brief CausalConv1dGrad kernel 实现（arch35）—— P0 形态重构版
 *
 * 切分策略：沿 D（通道）维切 tile（baseDim），核间按 tile 轮转。
 * 每个核独占若干完整通道，核内即可完整累加 dweight/dbias（无需跨核 reduce）。
 *
 * ============================ P0 形态重构（目标 2x 吞吐） ============================
 * msprof 实测（重构前）：vec 88% 为"低效忙"——scalar 54.4% + vec_resc_cflt 47.3%，
 * 根因是逐 token 128 元素短指令 + 对同目标的串行 Add 累加链（dbias/dweight/dx）。
 * 本版本四个形态变化：
 *
 * 1. 计算整块化（消 scalar 54.4%）：逐 token 的 (1×D0) 短向量调用全部提升为
 *    (T_CHUNK×D0) 整块大块调用（count 形式跨行连续发射，单条 API 64 repeats）。
 * 2. 消串行 Add 累加链（消 vec_resc_cflt 47.3%）：dbias/dweight 的沿 token 轴累加改为
 *    "平面大块 Mul + 原地半平面折叠树归并"（层数 log2(rows)≈6，替代 rows 条 RAW 串行链）。
 * 3. T_CHUNK 32→64：单 chunk 计算块 4096→8192 元素；搬运粒度 8KB→16KB/段。
 * 4. 批量搬运：逐行 256B DataCopy → 一次 2D DataCopyPad 搬整个 (rows×D0) 平面
 *    （blockCount=rows，blockLen=D0*sizeof(T) 字节，srcStride=(dim_-D0)*sizeof(T)）。
 *
 * ============================ P1 消 bank 冲突（w/bias 平面预展开） ============================
 * P0 遗留瓶颈：w[j]/bias 行广播（src1RepStride=0）的 D0 向量被 64 repeat 整块乘反复
 * 重读同一固定地址 → vec_bank_cflt 5.1%→77.7% 成新主导（吞吐卡 1.32x）。
 * P1-R2（生效，2.53x）：行广播全部替换为"倍增预展开 + 纯平面 count Mul/Add"——
 *   ExpandRowToPlane 把 w[j]/bias 的 1 行倍增复制成 (rows×D0) 同形平面
 *   （6 条 count 大块调用，源/目标连续不重叠 → 纯流式访问），随后 Mul/Add 双源同形。
 *   展开 plane 不新增 UB：z 阶段复用 sigBuf（silu' 生命周期从阶段 2.2 起）、
 *   dx 阶段复用 gyF32（grad_z 计算后 gy 死亡）；每 tap 重展开（开销可忽略）。
 *   实测：bank 事件 -84%（89.1M→14.1M），B1_T2048 1448.9→573.7us（2.53x，
 *   每 token 0.280us），15 shape geomean 1.332→3.364 全反超。
 * 已排除（消融）：广播源 src1→src0 交换（±0.05% 无效，读端口对称）；
 *   三倍增错相展开（+9% 更慢，残余 bank 31.2% 非倍增链 2 幂偏移所致）。
 *
 * 数学语义逐行保持（与重构前完全一致）：
 *   z = Σ_j w_j·x_{t-j} + b（左 pad 为 0：首 chunk 将 xF32 前 hist 行清零，乘零项自然消亡）
 *   silu'(z) = σ(z)(1 + z(1-σ(z)))，grad_z = grad_y·silu'(z)
 *   dbias = Σ_t grad_z_t；dweight[j] = Σ_t grad_z_t·x_{t-j}；dx_t = Σ_j grad_z_{t+j}·w_j
 *   dx 右边界：t+j ≥ T0 的项截断（每 j 按有效行数 valid 递减，行 [valid,n) 保持 j=0 结果）
 *
 * Buffer 复用（UB 248KB 预算下容纳 T_CHUNK=64 的 7 个 fp32 平面）：
 *   zBuf 兼 dxMat（silu' 计算完成后 z 死亡，dx 才开始写）
 *   sigBuf 兼 z 阶段 w/bias 展开平面（阶段 2.2 前）兼 dx 的临时乘积区（grad_z 计算后）
 *   gyF32 兼 dx 阶段 w 展开平面（grad_z 计算完成后 gy 死亡）
 *   dweight/dbias 树归并均原地（tmp / gradZ 当前区），不新增 scratch
 *
 * 跨 chunk 边界（与重构前一致的 tail 机制）：
 *   grad_z 行 [0,hist) 为上一 chunk 尾部；dx 每 chunk 末尾 (kW-1) 行延迟到下一 chunk
 *   头部补算；最后一个 chunk 的延迟行在终局按右边界截断补算。
 *
 * chunk 内阶段顺序（dbias 归约移到 dx/tail 之后，使 gradZ 原地归约合法）：
 *   0 搬入+cast → 1 z 重算 → 2 silu'/grad_z → 3 dweight（tmp 原地树归并）
 *   → 4 dx（5.1 补算/5.2 常规/5.3 终局，dxMat=zBuf、dxTmp=sigBuf）
 *   → 5 tail 更新 → 6 dbias（gradZ 当前区原地树归并）
 *
 * ⚠️ 同步约束（arch35 跨流水陷阱，不可回退）：MTE2↔VEC↔MTE3 边界数据传递必须
 * TQue 队列模式（AllocTensor→DataCopyPad→EnQue→DeQue→FreeTensor）；TBuf+PipeBarrier
 * 在 Ascend950 AIV 上不保证跨流水可见性（会读到旧值，静默错位）。
 */
#ifndef CAUSAL_CONV1D_GRAD_H
#define CAUSAL_CONV1D_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "causal_conv1d_grad_tiling_data.h"
#include "causal_conv1d_grad_tiling_key.h"

namespace NsCausalConv1dGrad {

using namespace AscendC;

template <typename T>
class CausalConv1dGradKernel {
public:
    __aicore__ inline CausalConv1dGradKernel() {}
    __aicore__ inline void Init(GM_ADDR grad_y, GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                                GM_ADDR dx, GM_ADDR dweight, GM_ADDR dbias,
                                const CausalConv1dGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessTile(int64_t dimOffset, int64_t actualDim);
    // 2D 整段搬运：一次 DataCopyPad 搬 (rows × actualDim) 平面（GM 行距 dim_，UB 紧排）
    __aicore__ inline void CopyInTile2D(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                                        int64_t rows, int64_t actualDim);
    __aicore__ inline void CopyOutTile2D(const GlobalTensor<T>& dst, const LocalTensor<T>& src,
                                         int64_t rows, int64_t actualDim);
    // 行 → 平面倍增展开（P1-R2）：把 row[0..cols) 复制到 plane 行 [0,rows) 的每一行。
    // 倍增复制：先写第 0 行，随后每步把 [0,filled) 复制到 [filled, filled+k)，
    // ceil(log2(rows))≈6 条 count 大块调用；源/目标区间连续不重叠 → 纯流式访问，
    // 无固定地址重读（消除 srcRepStride=0 广播导致的 vec_bank_cflt，P0 实测 77.7%）。
    __aicore__ inline void ExpandRowToPlane(const LocalTensor<float>& plane,
                                            const LocalTensor<float>& row,
                                            int64_t rows, int64_t cols);
    // 行广播乘/加：dst[r][*] = src[r][*] (op) row[*]，r ∈ [0,rows)。
    // fast（D0%64==0）：row 先倍增展开到 plane（同形平面，生命周期错开复用：z 阶段
    // sigBuf / dx 阶段 gyF32），再平面 count Mul/Add（双源同形流式）；
    // 否则退化为逐行（正确性兜底）。
    __aicore__ inline void BcastMulRows(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                        const LocalTensor<float>& row,
                                        const LocalTensor<float>& plane,
                                        int64_t rows, int64_t cols, bool fast);
    __aicore__ inline void BcastAddRows(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                        const LocalTensor<float>& row,
                                        const LocalTensor<float>& plane,
                                        int64_t rows, int64_t cols, bool fast);
    // 沿行轴原地半平面折叠树归并：v[0..cols) = Σ_{r<n} v[r]（层间 PipeBarrier，破坏 v）
    __aicore__ inline void TreeReduceRowsInPlace(const LocalTensor<float>& v, int64_t n, int64_t cols);
    // 计算一段 dx 行并写出（大块形态）：j=0 Mul 覆盖全 n 行，j>0 Mul→平面 Add 累加，
    // 每 j 按 t+j<T0 截断有效行数；dxTmp 为临时乘积区（sigBuf 复用）；
    // wPlane 为 w 行展开平面（gyF32 复用，grad_z 计算后 gy 已死）
    __aicore__ inline void ComputeDxRows(const LocalTensor<float>& gradZF32,
                                         const LocalTensor<float>& wF32,
                                         const LocalTensor<float>& dxF32,
                                         const LocalTensor<float>& dxTmp,
                                         const LocalTensor<float>& wPlane,
                                         int64_t tStart, int64_t tEnd, int64_t chunkStart,
                                         int64_t T0, int64_t D0, int64_t K, int64_t hist,
                                         bool fast);
    // dx 分段写出（每段 ≤ OUT_ROWS 行，outQue 容量约束）
    __aicore__ inline void CopyOutDx(const LocalTensor<float>& dxF32,
                                     int64_t baseGm, int64_t tStart, int64_t rows, int64_t D0);

private:
    // T 维分块行数（P0: 32→64）：单 chunk 计算/搬运块 64×128 元素（bf16 16KB/段）
    static constexpr int64_t T_CHUNK = 64;
    // dx 写出分段行数（outQue 容量 = OUT_ROWS×baseDim×sizeof(T)）
    static constexpr int64_t OUT_ROWS = 32;

    GlobalTensor<T> gradYGM_, xGM_, weightGM_, biasGM_;
    GlobalTensor<T> dxGM_, dweightGM_, dbiasGM_;

    TPipe pipe_;
    // fp16/bf16 staging —— 队列模式：MTE2→VEC / VEC→MTE3 边界必须走 EnQue/DeQue 事件同步。
    TQue<QuePosition::VECIN, 1> gyQue_, xQue_, wQue_, bQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    // fp32 compute
    TBuf<TPosition::VECCALC> gyF32Buf_, xF32Buf_, wF32Buf_, bF32Buf_;
    TBuf<TPosition::VECCALC> zBuf_, sigBuf_, tmpBuf_, gradZBuf_;
    TBuf<TPosition::VECCALC> dweightBuf_, dbiasBuf_;
    // bank 相位垫片（P0 第 3 轮）：所有计算 plane 均为 2 的幂尺寸，顺序分配下任意两
    // buffer 偏移都是 bank 交错周期（2 的幂）的整数倍 → 跨 buffer 双源大块操作
    // （Mul(tmp,gradZ,x)、Add(z,z,tmp)、Add(dx,dx,dxTmp) 等）src0/src1 恒落同一 bank
    // （实测 vec_bank_cflt 77.7%）。每个计算 buffer 间垫 32B 使相位错开。
    TBuf<TPosition::VECCALC> padA_, padB_, padC_, padD_, padE_, padF_, padG_, padH_, padI_;

    int64_t batch_ = 0;
    int64_t seqLen_ = 0;
    int64_t dim_ = 0;
    int64_t kW_ = 0;
    int64_t baseDim_ = 0;
    int64_t dimTileCnt_ = 0;
    int64_t usedCores_ = 0;
    int64_t blockIdx_ = 0;
};

template <typename T>
__aicore__ inline void CausalConv1dGradKernel<T>::Init(
    GM_ADDR grad_y, GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
    GM_ADDR dx, GM_ADDR dweight, GM_ADDR dbias,
    const CausalConv1dGradTilingData* tilingData)
{
    batch_ = tilingData->batch;
    seqLen_ = tilingData->seqLen;
    dim_ = tilingData->dim;
    kW_ = tilingData->kW;
    baseDim_ = tilingData->baseDim;
    dimTileCnt_ = tilingData->dimTileCnt;
    usedCores_ = tilingData->usedCores;
    blockIdx_ = GetBlockIdx();

    gradYGM_.SetGlobalBuffer((__gm__ T*)grad_y);
    xGM_.SetGlobalBuffer((__gm__ T*)x);
    weightGM_.SetGlobalBuffer((__gm__ T*)weight);
    biasGM_.SetGlobalBuffer((__gm__ T*)bias);
    dxGM_.SetGlobalBuffer((__gm__ T*)dx);
    dweightGM_.SetGlobalBuffer((__gm__ T*)dweight);
    dbiasGM_.SetGlobalBuffer((__gm__ T*)dbias);

    // UB 预算（bd=128, T_CHUNK=64, hist=3, bf16）：
    //   fp16 队列 43008B + fp32 平面 205800B + bank 垫片 288B = 249096B < 248KB(253952B)
    int64_t bd = baseDim_;
    int64_t hist = kW_ - 1;                    // x 历史/左 pad 行数
    int64_t xRowsMax = T_CHUNK + hist;         // x 每 chunk 行数（历史 + 本 chunk）
    int64_t planeF16 = T_CHUNK * bd * sizeof(T);
    int64_t xPlaneF16 = xRowsMax * bd * sizeof(T);
    int64_t outF16 = OUT_ROWS * bd * sizeof(T);
    int64_t wF16Sz = kW_ * bd * sizeof(T);
    int64_t bF16Sz = bd * sizeof(T);
    int64_t planeF32 = T_CHUNK * bd * sizeof(float);
    int64_t xPlaneF32 = xRowsMax * bd * sizeof(float);
    int64_t wF32Sz = kW_ * bd * sizeof(float);
    int64_t bF32Sz = bd * sizeof(float);

    // bank 相位垫片布局（总 9×32B = 288B）：相邻计算 buffer 间错开 32B，
    // 使 src0/src1 的跨 buffer 偏移不再是 bank 交错周期的整数倍
    pipe_.InitBuffer(gyQue_, 1, planeF16);
    pipe_.InitBuffer(padA_, 32);
    pipe_.InitBuffer(xQue_, 1, xPlaneF16);
    pipe_.InitBuffer(padB_, 32);
    pipe_.InitBuffer(wQue_, 1, wF16Sz);
    pipe_.InitBuffer(padC_, 32);
    pipe_.InitBuffer(bQue_, 1, bF16Sz);
    pipe_.InitBuffer(padD_, 32);
    pipe_.InitBuffer(outQue_, 1, outF16);

    pipe_.InitBuffer(gyF32Buf_, planeF32);
    pipe_.InitBuffer(padE_, 32);
    pipe_.InitBuffer(xF32Buf_, xPlaneF32);
    pipe_.InitBuffer(padF_, 32);
    // zBuf 生命周期：阶段 1 z 重算 ~ 阶段 2 silu'；此后复用为 dxMat（阶段 4）
    pipe_.InitBuffer(zBuf_, planeF32);
    pipe_.InitBuffer(padG_, 32);
    // sigBuf 生命周期：阶段 2 silu' ~ grad_z；此后复用为 dx 临时乘积区（阶段 4）
    pipe_.InitBuffer(sigBuf_, planeF32);
    pipe_.InitBuffer(padH_, 32);
    pipe_.InitBuffer(tmpBuf_, planeF32);
    pipe_.InitBuffer(padI_, 32);
    // grad_z 布局：行 [0, hist) 为上一 chunk 尾部（tail），行 [hist, hist+rows) 为当前 chunk
    pipe_.InitBuffer(gradZBuf_, xPlaneF32);

    pipe_.InitBuffer(wF32Buf_, wF32Sz);
    pipe_.InitBuffer(bF32Buf_, bF32Sz);
    pipe_.InitBuffer(dweightBuf_, wF32Sz);
    pipe_.InitBuffer(dbiasBuf_, bF32Sz);
}

template <typename T>
__aicore__ inline void CausalConv1dGradKernel<T>::CopyInTile2D(
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
__aicore__ inline void CausalConv1dGradKernel<T>::CopyOutTile2D(
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
__aicore__ inline void CausalConv1dGradKernel<T>::ExpandRowToPlane(
    const LocalTensor<float>& plane, const LocalTensor<float>& row,
    int64_t rows, int64_t cols)
{
    // 第 0 行：plane[0..cols) = row（Adds 标量 0 直通拷贝）
    Adds(plane, row, 0.0f, static_cast<int32_t>(cols));
    PipeBarrier<PIPE_V>();
    // 倍增：每步把已填充的 [0,filled) 行复制到 [filled, filled+k)（k=min(filled, rows-filled)），
    // count=k*cols 跨行连续，一条大块调用；filled 到 rows 共 ceil(log2(rows)) 步（rows=64 → 6 步）
    int64_t filled = 1;
    while (filled < rows) {
        int64_t k = (rows - filled < filled) ? (rows - filled) : filled;
        Adds(plane[filled * cols], plane, 0.0f, static_cast<int32_t>(k * cols));
        PipeBarrier<PIPE_V>();
        filled += k;
    }
}

template <typename T>
__aicore__ inline void CausalConv1dGradKernel<T>::BcastMulRows(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, const LocalTensor<float>& row,
    const LocalTensor<float>& plane, int64_t rows, int64_t cols, bool fast)
{
    if (fast) {
        // P1-R2：row 倍增展开成 (rows×cols) 平面（plane 为生命周期错开的复用 buffer），
        // 再平面 count Mul——双源同形、地址流式前进，消除 src1RepStride=0 广播的
        // 固定地址重读（P0/R1 实测 bank_cflt 77.7~79.8%，R1 交换广播源位置无效）
        ExpandRowToPlane(plane, row, rows, cols);
        Mul(dst, src, plane, static_cast<int32_t>(rows * cols));
    } else {
        // 兜底：逐行 count 形式（不依赖 stride 语义，任意行距安全）
        for (int64_t r = 0; r < rows; ++r) {
            Mul(dst[r * cols], src[r * cols], row, cols);
        }
    }
}

template <typename T>
__aicore__ inline void CausalConv1dGradKernel<T>::BcastAddRows(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, const LocalTensor<float>& row,
    const LocalTensor<float>& plane, int64_t rows, int64_t cols, bool fast)
{
    if (fast) {
        // P1-R2：同 BcastMulRows，展开后平面 count Add（dst==src 同址读写与 P0 平面 Add 同模式）
        ExpandRowToPlane(plane, row, rows, cols);
        Add(dst, src, plane, static_cast<int32_t>(rows * cols));
    } else {
        for (int64_t r = 0; r < rows; ++r) {
            Adds(dst[r * cols], row, 0.0f, cols);
        }
    }
}

template <typename T>
__aicore__ inline void CausalConv1dGradKernel<T>::TreeReduceRowsInPlace(
    const LocalTensor<float>& v, int64_t n, int64_t cols)
{
    // 半平面折叠：v[r] += v[r+half]（r < half），奇数尾行下移至 v[half]。
    // 层数 ceil(log2(n))（n=64 时 6 层），每层一条大块 Add（count=half*cols 跨行连续），
    // 替代 n 条对同目标的串行 RAW Add 链（msprof vec_resc_cflt 47.3% 的根因）。
    // dst 与 src0 完全同址（逐元素读改写，与 dst==src0 的串行累加同理，安全）。
    // （曾试奇偶配对 v[i]=v[2i]+v[2i+1]：实测更慢（访问跨度大），已回退）
    int64_t active = n;
    while (active > 1) {
        int64_t half = active >> 1;
        Add(v[0], v[0], v[half * cols], static_cast<int32_t>(half * cols));
        if (active & 1) {
            Adds(v[half * cols], v[(active - 1) * cols], 0.0f, static_cast<int32_t>(cols));
        }
        PipeBarrier<PIPE_V>();
        active = half + (active & 1);
    }
}

template <typename T>
__aicore__ inline void CausalConv1dGradKernel<T>::ComputeDxRows(
    const LocalTensor<float>& gradZF32, const LocalTensor<float>& wF32,
    const LocalTensor<float>& dxF32, const LocalTensor<float>& dxTmp,
    const LocalTensor<float>& wPlane,
    int64_t tStart, int64_t tEnd, int64_t chunkStart,
    int64_t T0, int64_t D0, int64_t K, int64_t hist, bool fast)
{
    // 计算 dx 行 [tStart, tEnd)（n 行）写到 dxF32 行 [0, n)。
    // dx_t = Σ_{j: t+j<T0} grad_z_{t+j} · w_j：
    //   j=0 Mul 直接覆盖全 n 行（tEnd ≤ T0 保证 j=0 项恒有效）；
    //   j>0 Mul 写 dxTmp → 平面 Add 累加前 valid=min(n, T0-tStart-j) 行，
    //   行 [valid, n) 的 j 项贡献为 0（t+j ≥ T0 右边界截断），保持 j=0 结果即可。
    // grad_z 源行定位：绝对行 t+j → gradZF32 行 (t+j-chunkStart+hist)，
    //   tail 区 [0,hist) ∪ 当前区 [hist, hist+rows) 连续。
    int64_t n = tEnd - tStart;
    for (int64_t j = 0; j < K; ++j) {
        int64_t valid = T0 - (tStart + j);
        if (valid > n) {
            valid = n;
        }
        if (valid <= 0) {
            break;  // 后续 j 的 valid 更小
        }
        int64_t srcOff = (tStart + j - chunkStart + hist) * D0;
        if (j == 0) {
            BcastMulRows(dxF32, gradZF32[srcOff], wF32, wPlane, valid, D0, fast);
            PipeBarrier<PIPE_V>();
        } else {
            BcastMulRows(dxTmp, gradZF32[srcOff], wF32[j * D0], wPlane, valid, D0, fast);
            PipeBarrier<PIPE_V>();
            Add(dxF32, dxF32, dxTmp, static_cast<int32_t>(valid * D0));
            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename T>
__aicore__ inline void CausalConv1dGradKernel<T>::CopyOutDx(
    const LocalTensor<float>& dxF32, int64_t baseGm, int64_t tStart, int64_t rows, int64_t D0)
{
    if (rows <= 0) {
        return;
    }
    // 分段写出：每段 ≤ OUT_ROWS 行（outQue 容量）
    for (int64_t seg = 0; seg < rows; seg += OUT_ROWS) {
        int64_t segRows = rows - seg;
        if (segRows > OUT_ROWS) {
            segRows = OUT_ROWS;
        }
        LocalTensor<T> outT = outQue_.template AllocTensor<T>();
        Cast(outT, dxF32[seg * D0], RoundMode::CAST_RINT, static_cast<int32_t>(segRows * D0));
        PipeBarrier<PIPE_V>();  // 等待 Cast 完成
        outQue_.EnQue(outT);
        LocalTensor<T> outV = outQue_.template DeQue<T>();
        CopyOutTile2D(dxGM_[baseGm + (tStart + seg) * dim_], outV, segRows, D0);
        outQue_.FreeTensor(outV);
    }
}

template <typename T>
__aicore__ inline void CausalConv1dGradKernel<T>::Process()
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
__aicore__ inline void CausalConv1dGradKernel<T>::ProcessTile(int64_t dimOffset, int64_t actualDim)
{
    LocalTensor<float> gyF32 = gyF32Buf_.Get<float>();
    LocalTensor<float> xF32 = xF32Buf_.Get<float>();
    LocalTensor<float> wF32 = wF32Buf_.Get<float>();
    LocalTensor<float> bF32 = bF32Buf_.Get<float>();
    LocalTensor<float> zF32 = zBuf_.Get<float>();
    LocalTensor<float> sigF32 = sigBuf_.Get<float>();
    LocalTensor<float> tmpF32 = tmpBuf_.Get<float>();
    LocalTensor<float> gradZF32 = gradZBuf_.Get<float>();
    LocalTensor<float> dxF32 = zF32;     // dxMat 复用 zBuf：silu' 计算后 z 死亡
    LocalTensor<float> dxTmpF32 = sigF32; // dx 临时乘积区复用 sigBuf：grad_z 计算后 silu' 死亡
    LocalTensor<float> dweightF32 = dweightBuf_.Get<float>();
    LocalTensor<float> dbiasF32 = dbiasBuf_.Get<float>();

    int64_t T0 = seqLen_;
    int64_t D0 = actualDim;
    int64_t K = kW_;
    int64_t hist = K - 1;  // kW-1，kimik3 场景为 3
    // 行广播 mask 形式要求：cols 为 64 整除（mask≤64 且 RepStride 为整数 block）。
    // 主场景 D=12288/baseDim=128 恒满足；尾 tile 非 64 整除时退化为逐行（仅保正确性）
    bool fast = (D0 % 64 == 0);

    // ========== 阶段 0: 加载常驻参数 weight/bias（队列模式：EnQue/DeQue 保证 MTE2→VEC 可见性） ==========
    LocalTensor<T> wF16 = wQue_.template AllocTensor<T>();
    LocalTensor<T> bF16 = bQue_.template AllocTensor<T>();
    CopyInTile2D(wF16, weightGM_[dimOffset], K, D0);
    CopyInTile2D(bF16, biasGM_[dimOffset], 1, D0);
    wQue_.EnQue(wF16);
    bQue_.EnQue(bF16);
    LocalTensor<T> wF16v = wQue_.template DeQue<T>();
    LocalTensor<T> bF16v = bQue_.template DeQue<T>();

    Cast(wF32, wF16v, RoundMode::CAST_NONE, static_cast<int32_t>(K * D0));
    Cast(bF32, bF16v, RoundMode::CAST_NONE, static_cast<int32_t>(D0));
    PipeBarrier<PIPE_V>();  // 等待 Vector Cast 完成

    wQue_.FreeTensor(wF16v);
    bQue_.FreeTensor(bF16v);

    // 初始化 dweight/dbias 累加区（跨 batch 累加）
    Duplicate(dweightF32, 0.0f, static_cast<int32_t>(K * D0));
    Duplicate(dbiasF32, 0.0f, static_cast<int32_t>(D0));
    PipeBarrier<PIPE_V>();  // 等待初始化完成

    // ========== 逐 batch 处理 ==========
    for (int64_t b = 0; b < batch_; ++b) {
        int64_t baseGm = b * T0 * dim_ + dimOffset;

        // grad_z tail 区行 [0, hist) 首个 chunk 无效（chunkStart=0 时 tail 区行对应绝对行 <0，
        // 不会被访问：补算行 t >= chunkStart-hist 仅在 chunkStart >= hist 的 chunk 发生）
        for (int64_t s = 0; s < T0; s += T_CHUNK) {
            int64_t e = s + T_CHUNK;
            if (e > T0) {
                e = T0;
            }
            int64_t rows = e - s;              // 本 chunk 行数
            int64_t total = rows * D0;         // 本 chunk 计算块元素数

            // --- 搬入 grad_y[s..e) 与 x[s-h..e)（队列模式，一次 2D DataCopyPad/平面） ---
            LocalTensor<T> gyF16 = gyQue_.template AllocTensor<T>();
            LocalTensor<T> xF16 = xQue_.template AllocTensor<T>();
            CopyInTile2D(gyF16, gradYGM_[baseGm + s * dim_], rows, D0);
            if (s >= hist) {
                CopyInTile2D(xF16, xGM_[baseGm + (s - hist) * dim_], rows + hist, D0);
            } else {
                // 首 chunk：GM 无历史行，当前 chunk 行搬入 staging 的 [hist, hist+rows)，
                // cast 后将 xF32 前 hist 行清零（左 pad 语义 = 0，使 t-j<0 的乘项自然为 0）
                CopyInTile2D(xF16[hist * D0], xGM_[baseGm], rows, D0);
            }
            gyQue_.EnQue(gyF16);
            xQue_.EnQue(xF16);
            LocalTensor<T> gyF16v = gyQue_.template DeQue<T>();
            LocalTensor<T> xF16v = xQue_.template DeQue<T>();

            Cast(gyF32, gyF16v, RoundMode::CAST_NONE, static_cast<int32_t>(total));
            Cast(xF32, xF16v, RoundMode::CAST_NONE,
                 static_cast<int32_t>((rows + hist) * D0));
            PipeBarrier<PIPE_V>();  // 等待 Cast 完成

            gyQue_.FreeTensor(gyF16v);
            xQue_.FreeTensor(xF16v);

            if (s < hist) {
                Duplicate(xF32, 0.0f, static_cast<int32_t>(hist * D0));
                PipeBarrier<PIPE_V>();
            }

            // ========== 阶段 1: 重算 z = Σ_j w_j·x_{t-j} + b（t ∈ [s,e)，大块 + w/b 平面展开） ==========
            // x_{t-j} 是 xF32 行 (hist-j) 起的连续 rows 行（错位 j 行 → 仍是大块连续地址）；
            // 首 chunk 的 pad 零行使 t-j<0 的项自然为 0，无需逐行判断左边界。
            // z 初始化：j=0 Mul 直接覆盖写（无需先置 bias）。
            // P1-R2：w/b 展开平面复用 sigF32（silu' 生命周期从阶段 2.2 才开始，此处安全）。
            BcastMulRows(zF32, xF32[hist * D0], wF32, sigF32, rows, D0, fast);
            PipeBarrier<PIPE_V>();
            for (int64_t j = 1; j < K; ++j) {
                BcastMulRows(tmpF32, xF32[(hist - j) * D0], wF32[j * D0], sigF32, rows, D0, fast);
                PipeBarrier<PIPE_V>();
                Add(zF32, zF32, tmpF32, static_cast<int32_t>(total));
                PipeBarrier<PIPE_V>();
            }
            // z += bias（平面展开）。实证：tensor 版 Duplicate(dst, srcTensor, count) 在 arch35 上
            // 未按预期广播（z 缺失 bias → silu'(z) 乘性偏差，dbias/dweight 误差随 T 线性放大）；
            // P1-R2：bias 倍增展开成平面后 count Add，语义可靠且无广播 bank 冲突。
            BcastAddRows(zF32, zF32, bF32, sigF32, rows, D0, fast);
            PipeBarrier<PIPE_V>();

            // ========== 阶段 2: silu'(z) 与 grad_z（行 [0,rows)，total = rows*D0，整块） ==========
            // 2.1 σ(z) = 1/(1+exp(-z))
            Muls(tmpF32, zF32, -1.0f, static_cast<int32_t>(total));     // -z
            PipeBarrier<PIPE_V>();
            Exp(tmpF32, tmpF32, static_cast<int32_t>(total));            // exp(-z)
            PipeBarrier<PIPE_V>();
            Adds(tmpF32, tmpF32, 1.0f, static_cast<int32_t>(total));     // 1+exp(-z)
            PipeBarrier<PIPE_V>();
            // 2.1.1 σ(z) 先写入 gradZF32 当前 chunk 区行 [hist, hist+rows)
            Reciprocal(gradZF32[hist * D0], tmpF32, static_cast<int32_t>(total));
            PipeBarrier<PIPE_V>();

            // 2.2 silu'(z) = σ(z) * (1 + z * (1 - σ(z)))，结果存 sigF32
            Muls(tmpF32, gradZF32[hist * D0], -1.0f, static_cast<int32_t>(total));  // -σ
            Adds(tmpF32, tmpF32, 1.0f, static_cast<int32_t>(total));                 // 1-σ
            PipeBarrier<PIPE_V>();
            Mul(tmpF32, tmpF32, zF32, static_cast<int32_t>(total));                  // z(1-σ)
            PipeBarrier<PIPE_V>();
            Adds(tmpF32, tmpF32, 1.0f, static_cast<int32_t>(total));                 // 1+z(1-σ)
            PipeBarrier<PIPE_V>();
            Mul(sigF32, gradZF32[hist * D0], tmpF32, static_cast<int32_t>(total));   // silu'
            PipeBarrier<PIPE_V>();

            // 2.3 grad_z = grad_y * silu' -> gradZF32 当前 chunk 区（此后 sig/gy/z 生命周期结束）
            Mul(gradZF32[hist * D0], gyF32, sigF32, static_cast<int32_t>(total));
            PipeBarrier<PIPE_V>();

            // ========== 阶段 3: dweight[j] += Σ_t grad_z_t · x_{t-j}（平面大块 Mul + 原地树归并） ==========
            // gradZ 当前区 [hist, hist+rows) 与 x 错位 j 行区间 [(hist-j), (hist-j)+rows) 均
            // 连续 → 一条 count=total 的平面 Mul；随后 tmp 原地树归并出 [D0] 部分和累加。
            for (int64_t j = 0; j < K; ++j) {
                Mul(tmpF32, gradZF32[hist * D0], xF32[(hist - j) * D0],
                    static_cast<int32_t>(total));
                PipeBarrier<PIPE_V>();
                TreeReduceRowsInPlace(tmpF32, rows, D0);
                Add(dweightF32[j * D0], dweightF32[j * D0], tmpF32, static_cast<int32_t>(D0));
                PipeBarrier<PIPE_V>();
            }

            // ========== 阶段 4: dx（大块形态，队列模式写出） ==========
            // w 展开平面复用 gyF32（grad_z 计算完成后 gy 已死，阶段 4 全程无其他读者/写者）
            // 4.1 补算上一 chunk 延迟的 dx 行 [s-hist, s)（仅 s >= hist 的 chunk）：
            //     依赖 grad_z 行 [s-hist, s)（tail 区）∪ [s, s+hist)（当前 chunk 头部）
            if (s >= hist && hist > 0) {
                ComputeDxRows(gradZF32, wF32, dxF32, dxTmpF32, gyF32, s - hist, s, s,
                              T0, D0, K, hist, fast);
                CopyOutDx(dxF32, baseGm, s - hist, hist, D0);
            }

            // 4.2 本 chunk 常规 dx 行 [s, e-hist)（依赖 grad_z 至 e-1，全部已在当前 chunk 区）
            if (rows > hist) {
                ComputeDxRows(gradZF32, wF32, dxF32, dxTmpF32, gyF32, s, e - hist, s,
                              T0, D0, K, hist, fast);
                CopyOutDx(dxF32, baseGm, s, rows - hist, D0);
            }

            // 4.3 终局：最后一个 chunk 的延迟行 [max(s, e-hist), e)（t+j >= T0 截断）
            if (e == T0) {
                int64_t pendStart = (e - hist > s) ? (e - hist) : s;
                ComputeDxRows(gradZF32, wF32, dxF32, dxTmpF32, gyF32, pendStart, e, s,
                              T0, D0, K, hist, fast);
                CopyOutDx(dxF32, baseGm, pendStart, e - pendStart, D0);
            }

            // ========== 阶段 5: 更新 tail：本 chunk 尾部 hist 行 grad_z 拷到行 [0, hist) ==========
            // （rows < hist 仅出现在最后 chunk，此后无下一 chunk，tail 不再被访问）
            if (rows >= hist && hist > 0) {
                for (int64_t r = 0; r < hist; ++r) {
                    Adds(gradZF32[r * D0], gradZF32[(rows + r) * D0], 0.0f, static_cast<int32_t>(D0));
                }
                PipeBarrier<PIPE_V>();
            }

            // ========== 阶段 6: dbias += Σ_t grad_z_t（gradZ 当前区原地树归并 + 累加） ==========
            // 放在 dx/tail 之后：归约会破坏 gradZ 当前区，此前所有读者（dweight/dx/tail）已完成。
            TreeReduceRowsInPlace(gradZF32[hist * D0], rows, D0);
            Add(dbiasF32, dbiasF32, gradZF32[hist * D0], static_cast<int32_t>(D0));
            PipeBarrier<PIPE_V>();
        }
    }

    // ========== 写出 dweight / dbias（跨 batch 累加结果，队列模式） ==========
    LocalTensor<T> outDw = outQue_.template AllocTensor<T>();
    Cast(outDw, dweightF32, RoundMode::CAST_RINT, static_cast<int32_t>(K * D0));
    PipeBarrier<PIPE_V>();
    outQue_.EnQue(outDw);
    LocalTensor<T> outDwV = outQue_.template DeQue<T>();
    CopyOutTile2D(dweightGM_[dimOffset], outDwV, K, D0);
    outQue_.FreeTensor(outDwV);

    LocalTensor<T> outDb = outQue_.template AllocTensor<T>();
    Cast(outDb, dbiasF32, RoundMode::CAST_RINT, static_cast<int32_t>(D0));
    PipeBarrier<PIPE_V>();
    outQue_.EnQue(outDb);
    LocalTensor<T> outDbV = outQue_.template DeQue<T>();
    CopyOutTile2D(dbiasGM_[dimOffset], outDbV, 1, D0);
    outQue_.FreeTensor(outDbV);
}

} // namespace NsCausalConv1dGrad
#endif // CAUSAL_CONV1D_GRAD_H
