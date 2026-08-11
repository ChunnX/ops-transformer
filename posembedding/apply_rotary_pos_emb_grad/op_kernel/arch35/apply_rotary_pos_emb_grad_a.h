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
 * \file apply_rotary_pos_emb_grad_a.h
 * \brief A template kernel (NO_BROADCAST, N=1, cos==dy 全同)
 * MergeDim 将 B×N×S 合并为 b_, 逐 b_ 分核处理 Q 和 K
 * 每个输入/输出独立 buffer（全部双缓冲）：
 *   IsDcos=false: gradQ/gradK/cos/sin/gradQOut/gradKOut 共 6 队列
 *   IsDcos=true : + query/key/gradCosOut/gradSinOut 共 10 队列
 * IsDcos=true 时融合计算 grad_cos/grad_sin（DcosFlag=1），grad 只搬一次
 */
#ifndef APPLY_ROTARY_POS_EMB_GRAD_A_H
#define APPLY_ROTARY_POS_EMB_GRAD_A_H

#include "apply_rotary_pos_emb_grad_common.h"
#include "apply_rotary_pos_emb_grad_tiling_data.h"

namespace ApplyRotaryPosEmbGrad {
using namespace AscendC;

template <typename T, bool IsDcos = false>
class ApplyRotaryPosEmbGradA
{
public:
    __aicore__ inline ApplyRotaryPosEmbGradA(TPipe* pipe, const ApplyRopeGradRegbaseParams* td)
        : pipe_(pipe), tilingData_(td) {}
    __aicore__ inline ~ApplyRotaryPosEmbGradA() {}

    __aicore__ inline void Init(
        GM_ADDR grad_query_embed, GM_ADDR grad_key_embed, GM_ADDR cos, GM_ADDR sin,
        GM_ADDR grad_query, GM_ADDR grad_key,
        GM_ADDR query = nullptr, GM_ADDR key = nullptr,
        GM_ADDR grad_cos = nullptr, GM_ADDR grad_sin = nullptr);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitAllGlobalBuffer(
        GM_ADDR grad_query_embed, GM_ADDR grad_key_embed, GM_ADDR cos, GM_ADDR sin,
        GM_ADDR grad_query, GM_ADDR grad_key,
        GM_ADDR query, GM_ADDR key, GM_ADDR grad_cos, GM_ADDR grad_sin);
    __aicore__ inline void InitAllBuffer();
    __aicore__ inline void ProcessInLoop(TQue<QuePosition::VECIN, 1>& gInQ, TQue<QuePosition::VECIN, 1>& xInQ,
        TQue<QuePosition::VECOUT, 1>& oOutQ, GlobalTensor<T>& gGm, GlobalTensor<T>& oGm, GlobalTensor<T>& xGm,
        LocalTensor<T>& cU, LocalTensor<T>& sU, LocalTensor<T>& gradCos, LocalTensor<T>& gradSin,
        int64_t pos, int64_t len, bool isFirstAccum);

    __aicore__ inline void CopyInCosAndSin(int64_t pos, int64_t len);
    __aicore__ inline void CopyIn(TQue<QuePosition::VECIN, 1>& inQ, GlobalTensor<T>& src, int64_t pos, int64_t len);
    __aicore__ inline void CopyOut(TQue<QuePosition::VECOUT, 1>& outQ, GlobalTensor<T>& dst, int64_t pos, int64_t len);
    __aicore__ inline void CopyAccumOut(GlobalTensor<T>& ocGm, GlobalTensor<T>& osGm, int64_t pos, int64_t len);
    __aicore__ inline void Compute(TQue<QuePosition::VECIN, 1>& gInQ, TQue<QuePosition::VECIN, 1>& xInQ,
        TQue<QuePosition::VECOUT, 1>& oOutQ, LocalTensor<T>& cU, LocalTensor<T>& sU,
        LocalTensor<T>& gradCos, LocalTensor<T>& gradSin, int64_t bL, int64_t nL, bool isFirstAccum);

private:
    TPipe* pipe_;
    const ApplyRopeGradRegbaseParams* tilingData_;
    GlobalTensor<T> gqGm_, gkGm_, cosGm_, sinGm_, oqGm_, okGm_;
    GlobalTensor<T> qGm_, kGm_, ocGm_, osGm_;
    // 每个输入/输出独立 buffer（全部双缓冲）
    TQue<QuePosition::VECIN, 1> gradQInQ_, gradKInQ_, cosInQ_, sinInQ_;
    TQue<QuePosition::VECOUT, 1> gradQOutQ_, gradKOutQ_;
    // 以下 4 个队列仅 IsDcos=true 时 InitBuffer（分配 UB）并使用，IsDcos=false 时不分配、零占用
    TQue<QuePosition::VECIN, 1> queryInQ_, keyInQ_;
    TQue<QuePosition::VECOUT, 1> gradCosOutQ_, gradSinOutQ_;
    int64_t blkIdx_, bStart_, bLen_, ubB_, ubN_, D_, dHalf_, dBuffer_;
};

template <typename T, bool IsDcos>
__aicore__ inline void ApplyRotaryPosEmbGradA<T, IsDcos>::Init(
    GM_ADDR gq, GM_ADDR gk, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR oq, GM_ADDR ok, GM_ADDR query, GM_ADDR key,
    GM_ADDR gradCos, GM_ADDR gradSin)
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) return;
    InitAllGlobalBuffer(gq, gk, cos, sin, oq, ok, query, key, gradCos, gradSin);
    InitAllBuffer();
    blkIdx_ = GetBlockIdx();
    bLen_ = tilingData_->blockFactorB;
    bStart_ = blkIdx_ * tilingData_->blockFactorB;
    if (blkIdx_ == static_cast<int64_t>(tilingData_->blockNumB) - 1 &&
        tilingData_->b % tilingData_->blockFactorB != 0)
        bLen_ = tilingData_->b % tilingData_->blockFactorB;
}

template <typename T, bool IsDcos>
__aicore__ inline void ApplyRotaryPosEmbGradA<T, IsDcos>::InitAllGlobalBuffer(
    GM_ADDR gq, GM_ADDR gk, GM_ADDR cos, GM_ADDR sin,
    GM_ADDR oq, GM_ADDR ok, GM_ADDR query, GM_ADDR key,
    GM_ADDR gradCos, GM_ADDR gradSin)
{
    gqGm_.SetGlobalBuffer((__gm__ T*)gq);
    gkGm_.SetGlobalBuffer((__gm__ T*)gk);
    cosGm_.SetGlobalBuffer((__gm__ T*)cos);
    sinGm_.SetGlobalBuffer((__gm__ T*)sin);
    oqGm_.SetGlobalBuffer((__gm__ T*)oq);
    okGm_.SetGlobalBuffer((__gm__ T*)ok);
    if constexpr (IsDcos) {
        qGm_.SetGlobalBuffer((__gm__ T*)query);
        kGm_.SetGlobalBuffer((__gm__ T*)key);
        ocGm_.SetGlobalBuffer((__gm__ T*)gradCos);
        osGm_.SetGlobalBuffer((__gm__ T*)gradSin);
    }
}

template <typename T, bool IsDcos>
__aicore__ inline void ApplyRotaryPosEmbGradA<T, IsDcos>::InitAllBuffer()
{
    ubB_ = tilingData_->ubFactorS; ubN_ = tilingData_->ubFactorN;
    D_ = tilingData_->d;
    dHalf_ = D_ / HALF_COEF;
    // D 搬运方式：一次搬 D/2、搬 2 次（blockCount 翻倍）。
    // 每个半行独立按 32B 对齐落位，因此 dBuffer_ = CeilAlign(D/2, BLOCK/sizeof(T)) * 2，
    // 半行偏移 dBuffer_/2 天然 32B 对齐，D=100 这类非 16 对齐场景不再触发对齐违例。
    dBuffer_ = Ops::Base::CeilAlign<int64_t>(dHalf_, BLOCK_TYPE_SIZE / sizeof(T)) * HALF_COEF;
    // 每个输入/输出独立 buffer，全部双缓冲
    pipe_->InitBuffer(gradQInQ_, DOUBLE_BUFFER, ubB_ * ubN_ * dBuffer_ * sizeof(T));
    pipe_->InitBuffer(gradKInQ_, DOUBLE_BUFFER, ubB_ * ubN_ * dBuffer_ * sizeof(T));
    pipe_->InitBuffer(cosInQ_, DOUBLE_BUFFER, ubB_ * dBuffer_ * sizeof(T));
    pipe_->InitBuffer(sinInQ_, DOUBLE_BUFFER, ubB_ * dBuffer_ * sizeof(T));
    pipe_->InitBuffer(gradQOutQ_, DOUBLE_BUFFER, ubB_ * ubN_ * dBuffer_ * sizeof(T));
    pipe_->InitBuffer(gradKOutQ_, DOUBLE_BUFFER, ubB_ * ubN_ * dBuffer_ * sizeof(T));
    if constexpr (IsDcos) {
        pipe_->InitBuffer(queryInQ_, DOUBLE_BUFFER, ubB_ * ubN_ * dBuffer_ * sizeof(T));
        pipe_->InitBuffer(keyInQ_, DOUBLE_BUFFER, ubB_ * ubN_ * dBuffer_ * sizeof(T));
        pipe_->InitBuffer(gradCosOutQ_, DOUBLE_BUFFER, ubB_ * ubN_ * dBuffer_ * sizeof(T));
        pipe_->InitBuffer(gradSinOutQ_, DOUBLE_BUFFER, ubB_ * ubN_ * dBuffer_ * sizeof(T));
    }
}

template <typename T, bool IsDcos>
__aicore__ inline void ApplyRotaryPosEmbGradA<T, IsDcos>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCoreNum) return;
    int64_t ulc = Ops::Base::CeilDiv(bLen_, ubB_);
    for (int64_t i = 0; i < ulc; i++) {
        int64_t pos = bStart_ + i * ubB_;
        int64_t len = (i != ulc - 1) ? ubB_ : bLen_ - i * ubB_;

        CopyInCosAndSin(pos, len);
        LocalTensor<T> cU = cosInQ_.DeQue<T>();
        LocalTensor<T> sU = sinInQ_.DeQue<T>();

        if constexpr (IsDcos) {
            // grad_cos/grad_sin 输出缓冲：Q 路写部分积、K 路累加、轮末写回 GM
            LocalTensor<T> gcU = gradCosOutQ_.AllocTensor<T>();
            LocalTensor<T> gsU = gradSinOutQ_.AllocTensor<T>();
            ProcessInLoop(gradQInQ_, queryInQ_, gradQOutQ_, gqGm_, oqGm_, qGm_,
                          cU, sU, gcU, gsU, pos, len, true);    // Q 路：写
            ProcessInLoop(gradKInQ_, keyInQ_, gradKOutQ_, gkGm_, okGm_, kGm_,
                          cU, sU, gcU, gsU, pos, len, false);   // K 路：累加
            gradCosOutQ_.EnQue(gcU);
            gradSinOutQ_.EnQue(gsU);
            CopyAccumOut(ocGm_, osGm_, pos, len);
        } else {
            ProcessInLoop(gradQInQ_, queryInQ_, gradQOutQ_, gqGm_, oqGm_, qGm_,
                          cU, sU, cU, sU, pos, len, true);
            ProcessInLoop(gradKInQ_, keyInQ_, gradKOutQ_, gkGm_, okGm_, kGm_,
                          cU, sU, cU, sU, pos, len, false);
        }

        cosInQ_.FreeTensor(cU);
        sinInQ_.FreeTensor(sU);
    }
}

template <typename T, bool IsDcos>
__aicore__ inline void ApplyRotaryPosEmbGradA<T, IsDcos>::ProcessInLoop(
    TQue<QuePosition::VECIN, 1>& gInQ, TQue<QuePosition::VECIN, 1>& xInQ,
    TQue<QuePosition::VECOUT, 1>& oOutQ,
    GlobalTensor<T>& gGm, GlobalTensor<T>& oGm, GlobalTensor<T>& xGm,
    LocalTensor<T>& cU, LocalTensor<T>& sU, LocalTensor<T>& gradCos, LocalTensor<T>& gradSin,
    int64_t pos, int64_t len, bool isFirstAccum)
{
    CopyIn(gInQ, gGm, pos, len);
    if constexpr (IsDcos) {
        // 搬入 query/key（与 grad 同 pos/len 对齐）
        CopyIn(xInQ, xGm, pos, len);
    }
    Compute(gInQ, xInQ, oOutQ, cU, sU, gradCos, gradSin, len, 1, isFirstAccum);
    CopyOut(oOutQ, oGm, pos, len);
}

template <typename T, bool IsDcos>
__aicore__ inline void ApplyRotaryPosEmbGradA<T, IsDcos>::CopyAccumOut(
    GlobalTensor<T>& ocGm, GlobalTensor<T>& osGm, int64_t pos, int64_t len)
{
    LocalTensor<T> ac = gradCosOutQ_.DeQue<T>();
    LocalTensor<T> as = gradSinOutQ_.DeQue<T>();
    // 一次搬 D/2、搬 len*2 次：UB 中半行按 dBuffer_/2 对齐落位，GM 侧 D 连续
    DataCopyExtParams ep{static_cast<uint16_t>(len * HALF_COEF),
                         static_cast<uint32_t>(dHalf_ * sizeof(T)), 0, 0, 0};
    DataCopyPad(ocGm[pos * D_], ac, ep);
    DataCopyPad(osGm[pos * D_], as, ep);
    gradCosOutQ_.FreeTensor(ac);
    gradSinOutQ_.FreeTensor(as);
}

template <typename T, bool IsDcos>
__aicore__ inline void ApplyRotaryPosEmbGradA<T, IsDcos>::CopyInCosAndSin(int64_t pos, int64_t len)
{
    LocalTensor<T> cU = cosInQ_.AllocTensor<T>();
    LocalTensor<T> sU = sinInQ_.AllocTensor<T>();
    DataCopyExtParams ep{static_cast<uint16_t>(len * HALF_COEF),
                         static_cast<uint32_t>(dHalf_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pp{false, 0, 0, 0};
    DataCopyPad(cU, cosGm_[pos * D_], ep, pp);
    DataCopyPad(sU, sinGm_[pos * D_], ep, pp);
    cosInQ_.EnQue(cU); sinInQ_.EnQue(sU);
}

template <typename T, bool IsDcos>
__aicore__ inline void ApplyRotaryPosEmbGradA<T, IsDcos>::CopyIn(
    TQue<QuePosition::VECIN, 1>& inQ, GlobalTensor<T>& src, int64_t pos, int64_t len)
{
    LocalTensor<T> t = inQ.AllocTensor<T>();
    DataCopyExtParams ep{static_cast<uint16_t>(len * HALF_COEF),
                         static_cast<uint32_t>(dHalf_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pp{false, 0, 0, 0};
    DataCopyPad(t, src[pos * D_], ep, pp);
    inQ.EnQue(t);
}

template <typename T, bool IsDcos>
__aicore__ inline void ApplyRotaryPosEmbGradA<T, IsDcos>::CopyOut(
    TQue<QuePosition::VECOUT, 1>& outQ, GlobalTensor<T>& dst, int64_t pos, int64_t len)
{
    LocalTensor<T> s = outQ.DeQue<T>();
    DataCopyExtParams ep{static_cast<uint16_t>(len * HALF_COEF),
                         static_cast<uint32_t>(dHalf_ * sizeof(T)), 0, 0, 0};
    DataCopyPad(dst[pos * D_], s, ep);
    outQ.FreeTensor(s);
}

template <typename T, bool IsDcos>
__aicore__ inline void ApplyRotaryPosEmbGradA<T, IsDcos>::Compute(
    TQue<QuePosition::VECIN, 1>& gInQ, TQue<QuePosition::VECIN, 1>& xInQ,
    TQue<QuePosition::VECOUT, 1>& oOutQ,
    LocalTensor<T>& cU, LocalTensor<T>& sU, LocalTensor<T>& gradCos, LocalTensor<T>& gradSin,
    int64_t bL, int64_t nL, bool isFirstAccum)
{
    LocalTensor<T> in = gInQ.DeQue<T>();
    LocalTensor<T> out = oOutQ.AllocTensor<T>();
    BatchHalfGradContiguousVF<T, false>(
        (__local_mem__ T*)in.GetPhyAddr(), (__local_mem__ T*)cU.GetPhyAddr(),
        (__local_mem__ T*)sU.GetPhyAddr(), (__local_mem__ T*)out.GetPhyAddr(),
        // A 模板 MergeDim 后 B×N×S 已合并为一维行，UB 中 bL 行按 dBuffer_ 步长排布（行 i 位于 i*dBuffer_）。
        // VF 内 sIdx 的步长恰为 dAlign，而 bIdx/nIdx 步长为 ubFactorN*ubFactorS*dAlign，
        // 因此这里必须把行数传给 sLength（最内层轴），bLength/nLength 传 1，
        // 否则 ubFactorS>1 时行间步长会被放大 ubFactorS 倍，越界写坏 UB。
        bL, 1, nL, D_, dBuffer_, ubB_, ubN_);
    if constexpr (IsDcos) {
        // 融合：query/key 与 grad 同 pos/len 对齐，Q 路写部分积、K 路累加
        LocalTensor<T> xU = xInQ.DeQue<T>();
        if (isFirstAccum) {
            AccumDcosDsinVF<T, true>(
                (__local_mem__ T*)in.GetPhyAddr(), (__local_mem__ T*)xU.GetPhyAddr(),
                (__local_mem__ T*)gradCos.GetPhyAddr(), (__local_mem__ T*)gradSin.GetPhyAddr(),
                static_cast<uint16_t>(bL * nL), D_, dBuffer_);
        } else {
            AccumDcosDsinVF<T, false>(
                (__local_mem__ T*)in.GetPhyAddr(), (__local_mem__ T*)xU.GetPhyAddr(),
                (__local_mem__ T*)gradCos.GetPhyAddr(), (__local_mem__ T*)gradSin.GetPhyAddr(),
                static_cast<uint16_t>(bL * nL), D_, dBuffer_);
        }
        xInQ.FreeTensor(xU);
    }
    gInQ.FreeTensor(in);
    oOutQ.EnQue(out);
}

} // namespace ApplyRotaryPosEmbGrad
#endif
