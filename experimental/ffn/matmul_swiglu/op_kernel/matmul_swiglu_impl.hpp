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
 * \file matmul_swiglu_impl.hpp
 * \brief MatmulSwiglu kernel(方案一: 标准 matmul -> GM 中间 -> 向量 SwiGLU)
 *
 * y = SiLU(gate) * up, [gate|up] = x @ weight (+bias), weight 打包为 [K, 2N]。
 * 方案一(稳): 用一次标准 matmul 算完整 C=[M,2N] 写到 GM 中间缓冲(user workspace),
 * 跨核同步后, 向量侧按行读回 C, 切分 gate=C[:,0:N]/up=C[:,N:2N], 就地做 SwiGLU 写出 y。
 * 不用取巧的 packed 双 matmul + SetOrgShape 子块寻址(该写法真机实测输出恒 0)。
 */
#ifndef MATMUL_SWIGLU_IMPL_HPP
#define MATMUL_SWIGLU_IMPL_HPP

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace NsMatmulSwiglu {
using namespace AscendC;
using namespace matmul;

template <typename xType, typename cType /*fp32 累加*/>
class MatmulSwigluKernel {
public:
    using AType    = MatmulType<TPosition::GM, CubeFormat::ND, xType, false>;
    using BType    = MatmulType<TPosition::GM, CubeFormat::ND, xType, false>;
    using CTypeMm  = MatmulType<TPosition::GM, CubeFormat::ND, cType>;
    using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, cType>;

    Matmul<AType, BType, CTypeMm, BiasType> mm;  // 单 matmul: x[M,K] @ weight[K,2N] -> [M,2N]
    TPipe pipe;

    __aicore__ inline MatmulSwigluKernel() {}

    // userWs: GetUserWorkspace(workspace) 得到的用户 workspace 指针(存 [M,2N] fp32 中间结果)
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y, GM_ADDR userWs,
                                const optiling::MatmulSwigluTilingData* tilingData)
    {
        m_           = tilingData->m;
        k_           = tilingData->k;
        n_           = tilingData->n;
        twoN_        = tilingData->twoN;
        hasBias_     = (tilingData->hasBias == 1U);
        transposeW_  = (tilingData->transposeWeight == 1U);
        usedCoreNum_ = tilingData->usedCoreNum;

        xGm_.SetGlobalBuffer((__gm__ xType*)x, m_ * k_);
        wGm_.SetGlobalBuffer((__gm__ xType*)weight, k_ * twoN_);
        if (hasBias_) {
            biasGm_.SetGlobalBuffer((__gm__ cType*)bias, twoN_);
        }
        yGm_.SetGlobalBuffer((__gm__ xType*)y, m_ * n_);
        cGm_.SetGlobalBuffer((__gm__ cType*)userWs, m_ * twoN_);  // [M, 2N] fp32 中间

        // 向量阶段每次处理一行的 N 个元素: gate/up/sig(fp32) + y(xType)
        pipe.InitBuffer(inQueGate_, 2, n_ * sizeof(cType));
        pipe.InitBuffer(inQueUp_, 2, n_ * sizeof(cType));
        pipe.InitBuffer(sigBuf_, n_ * sizeof(cType));
        pipe.InitBuffer(outQueY_, 2, n_ * sizeof(xType));
    }

    __aicore__ inline void Process()
    {
        // ---- Phase 1: 标准 matmul, 完整 [M,2N] 写到 GM 中间 ----
        mm.SetTensorA(xGm_, false);
        mm.SetTensorB(wGm_, transposeW_);
        if (hasBias_) {
            mm.SetBias(biasGm_);
        }
        mm.IterateAll(cGm_);
        mm.End();

        SyncAll();  // 跨核屏障: 确保 matmul(Cube)写完 cGm, 向量核再读

        // ---- Phase 2: 向量 SwiGLU, 按行切分到各核 ----
        const int64_t coreNum = GetBlockNum();
        const int64_t blockIdx = GetBlockIdx();
        const int64_t rowsPerCore = (static_cast<int64_t>(m_) + coreNum - 1) / coreNum;
        const int64_t rowStart = blockIdx * rowsPerCore;
        int64_t rowEnd = rowStart + rowsPerCore;
        rowEnd = rowEnd < static_cast<int64_t>(m_) ? rowEnd : static_cast<int64_t>(m_);
        for (int64_t row = rowStart; row < rowEnd; ++row) {
            ComputeRow(row);
        }
    }

private:
    // 处理第 row 行: gate=cGm[row,0:N], up=cGm[row,N:2N], y=SiLU(gate)*up
    __aicore__ inline void ComputeRow(int64_t row)
    {
        DataCopyExtParams copyN{1, static_cast<uint32_t>(n_ * sizeof(cType)), 0, 0, 0};
        DataCopyPadExtParams<cType> padN{false, 0, 0, 0};

        LocalTensor<cType> gate = inQueGate_.AllocTensor<cType>();
        LocalTensor<cType> up = inQueUp_.AllocTensor<cType>();
        DataCopyPad(gate, cGm_[row * twoN_], copyN, padN);
        DataCopyPad(up, cGm_[row * twoN_ + n_], copyN, padN);
        inQueGate_.EnQue(gate);
        inQueUp_.EnQue(up);

        gate = inQueGate_.DeQue<cType>();
        up = inQueUp_.DeQue<cType>();
        LocalTensor<cType> sig = sigBuf_.Get<cType>();
        const int32_t len = static_cast<int32_t>(n_);
        Sigmoid(sig, gate, len);
        PipeBarrier<PIPE_V>();
        Mul(gate, gate, sig, len);   // SiLU(gate) = gate * sigmoid(gate)
        PipeBarrier<PIPE_V>();
        Mul(gate, gate, up, len);    // * up
        PipeBarrier<PIPE_V>();

        LocalTensor<xType> yUb = outQueY_.AllocTensor<xType>();
        if constexpr (std::is_same<xType, cType>::value) {
            DataCopy(yUb, gate, len);
        } else {
            Cast(yUb, gate, RoundMode::CAST_RINT, len);
        }
        inQueGate_.FreeTensor(gate);
        inQueUp_.FreeTensor(up);
        outQueY_.EnQue(yUb);

        yUb = outQueY_.DeQue<xType>();
        DataCopyExtParams copyY{1, static_cast<uint32_t>(n_ * sizeof(xType)), 0, 0, 0};
        DataCopyPad(yGm_[row * n_], yUb, copyY);
        outQueY_.FreeTensor(yUb);
    }

private:
    TQue<QuePosition::VECIN, 2> inQueGate_;
    TQue<QuePosition::VECIN, 2> inQueUp_;
    TQue<QuePosition::VECOUT, 2> outQueY_;
    TBuf<TPosition::VECCALC> sigBuf_;

    GlobalTensor<xType> xGm_;
    GlobalTensor<xType> wGm_;
    GlobalTensor<cType> biasGm_;
    GlobalTensor<xType> yGm_;
    GlobalTensor<cType> cGm_;

    uint64_t m_ = 0, k_ = 0, n_ = 0, twoN_ = 0;
    uint32_t usedCoreNum_ = 0;
    bool hasBias_ = false;
    bool transposeW_ = false;
};

}  // namespace NsMatmulSwiglu
#endif  // MATMUL_SWIGLU_IMPL_HPP
