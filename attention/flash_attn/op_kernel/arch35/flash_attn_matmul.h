/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Flash Attention arch35: FP16/BF16 L1 -> L0 and cube instructions.
#ifndef FLASH_ATTN_MATMUL_H_
#define FLASH_ATTN_MATMUL_H_
#include "kernel_operator.h"

namespace FlashAttnMatmul {
using namespace AscendC;
constexpr uint32_t FP16_ONE_FRACTAL_ELEMENT = 16;
constexpr uint32_t AIV_FLAG_OFFSET = 16;

constexpr uint32_t L0_SLOT_BYTES = 32U * 1024U;
constexpr uint32_t L0A_MUTEX_BASE = 6U;
constexpr uint32_t L0B_MUTEX_BASE = 8U;

struct MMParam {
    uint32_t singleM;
    uint32_t singleN;
    uint32_t singleK;
    uint32_t realM = 0;
    uint32_t loadK = 0;
    uint32_t loadN = 0;
};

__aicore__ inline uint32_t GetLoadK(const MMParam &param)
{
    return param.loadK != 0 ? param.loadK : param.singleK;
}

__aicore__ inline uint32_t GetLoadN(const MMParam &param)
{
    return param.loadN != 0 ? param.loadN : param.singleN;
}

template <bool leftTranspose, typename T>
__aicore__ inline void LoadDataToL0A(LocalTensor<T> &dst, const LocalTensor<T> &src, const MMParam &param,
                                     uint64_t offset, uint32_t k, uint32_t m)
{
    static_assert(IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value);
    LoadData2DParamsV2 load;
    load.mStartPosition = 0;
    load.kStartPosition = 0;
    load.ifTranspose = leftTranspose;
    load.mStep = ((leftTranspose ? k : m) + 15) >> 4;
    load.kStep = ((leftTranspose ? m : k) + 15) >> 4;
    load.srcStride = leftTranspose ? (param.singleK + 15) >> 4 : load.mStep;
    if (param.realM != 0) {
        load.mStep = (param.realM + 15) >> 4;
    }
    load.dstStride = leftTranspose ? (m + 15) >> 4 : load.mStep;
    LoadData(dst, src[offset], load);
}

template <bool rightTranspose, typename T>
__aicore__ inline void LoadDataToL0B(LocalTensor<T> &dst, const LocalTensor<T> &src, const MMParam &param,
                                     uint64_t offset, uint32_t k, uint32_t n)
{
    static_assert(IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value);
    LoadData2DParamsV2 load;
    load.mStartPosition = 0;
    load.kStartPosition = 0;
    load.ifTranspose = !rightTranspose;
    load.mStep = ((load.ifTranspose ? k : n) + 15) >> 4;
    load.kStep = ((load.ifTranspose ? n : k) + 15) >> 4;
    load.srcStride = ((load.ifTranspose ? param.singleK : param.singleN) + 15) >> 4;
    load.dstStride = load.ifTranspose ? (n + 15) >> 4 : load.mStep;
    LoadData(dst, src[offset], load);
}

template <bool leftTranspose, bool rightTranspose, bool singleL0B = false, typename A, typename B, typename C>
__aicore__ inline void MatmulFull(const LocalTensor<A> &aL1Tensor, const LocalTensor<B> &bL1Tensor, uint32_t &l0aIdx,
                                  uint32_t &l0bIdx, const LocalTensor<C> &cL0Tensor, struct MMParam &param)
{
    l0aIdx ^= 1U;
    uint32_t aMutex = L0A_MUTEX_BASE + l0aIdx;
    Mutex::Lock<PIPE_MTE1>(aMutex);
    LocalTensor<A> L0ATensor = LocalTensor<A>(TPosition::A2, l0aIdx * L0_SLOT_BYTES, L0_SLOT_BYTES / sizeof(A));
    LoadDataToL0A<leftTranspose>(L0ATensor, aL1Tensor, param, 0, GetLoadK(param), param.singleM); // s2*d,d,s2
    Mutex::Unlock<PIPE_MTE1>(aMutex);

    if constexpr (!singleL0B) {
        l0bIdx ^= 1U;
    }
    const uint32_t bIdx = singleL0B ? 0U : l0bIdx;
    uint32_t bMutex = L0B_MUTEX_BASE + bIdx;
    Mutex::Lock<PIPE_MTE1>(bMutex);
    LocalTensor<B> L0BTensor = LocalTensor<B>(TPosition::B2, bIdx * L0_SLOT_BYTES, L0_SLOT_BYTES / sizeof(B));
    LoadDataToL0B<rightTranspose>(L0BTensor, bL1Tensor, param, 0, GetLoadK(param), GetLoadN(param));
    Mutex::Unlock<PIPE_MTE1>(bMutex);

    Mutex::Lock<PIPE_M>(aMutex);
    Mutex::Lock<PIPE_M>(bMutex);

    MmadParams mmadParams;
    mmadParams.m = param.singleM;
    if (param.realM != 0) {
        mmadParams.m = param.realM;
    }
    mmadParams.n = param.singleN;
    mmadParams.k = param.singleK;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = 0;
    if (mmadParams.m == 1) {
        mmadParams.m = 16;
    }

    Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);

    Mutex::Unlock<PIPE_M>(aMutex);
    Mutex::Unlock<PIPE_M>(bMutex);
}

template <uint32_t baseK, bool leftTranspose, bool rightTranspose, bool singleL0B = false, typename A, typename B,
          typename C>
__aicore__ inline void MatmulK(const LocalTensor<A> &aL1Tensor, const LocalTensor<B> &bL1Tensor, uint32_t &l0aIdx,
                               uint32_t &l0bIdx, const LocalTensor<C> &cL0Tensor, const MMParam &param)
{
    uint32_t kLoops = (param.singleK + baseK - 1) / baseK;
    uint32_t tailSize = param.singleK % baseK;
    uint32_t tailK = tailSize ? tailSize : baseK;
    uint64_t L1Aoffset = leftTranspose ? baseK << 4 : ((param.singleM + 15) >> 4 << 4) * baseK;
    uint64_t L1Boffset = rightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK : baseK << 4;

    for (uint32_t k = 0; k < kLoops; k++) {
        uint32_t tileK = (k == (kLoops - 1)) ? tailK : baseK;
        l0aIdx ^= 1U;
        uint32_t aMutex = L0A_MUTEX_BASE + l0aIdx;
        Mutex::Lock<PIPE_MTE1>(aMutex); // mte1等Matmul：上一轮matmul完成后才能搬运新数据到L0A
        LocalTensor<A> L0ATensor = LocalTensor<A>(TPosition::A2, l0aIdx * L0_SLOT_BYTES, L0_SLOT_BYTES / sizeof(A));
        // 尾块搬运宽度按分形粒度对齐(对齐D时与legacy原值等价; 非对齐D如72→80, 与Nd2Nz写入组数一致)
        LoadDataToL0A<leftTranspose>(L0ATensor, aL1Tensor, param, k * L1Aoffset,
                                     (k == (kLoops - 1) && param.loadK != 0) ? (((tileK + 15) >> 4) << 4) : tileK,
                                     param.singleM); // s2*d,d,s2

        if constexpr (!singleL0B) {
            l0bIdx ^= 1U;
        }
        const uint32_t bIdx = singleL0B ? 0U : l0bIdx;
        uint32_t bMutex = L0B_MUTEX_BASE + bIdx;
        Mutex::Lock<PIPE_MTE1>(bMutex); // mte1等Matmul：上一轮matmul完成后才能搬运新数据到L0B
        LocalTensor<B> L0BTensor = LocalTensor<B>(TPosition::B2, bIdx * L0_SLOT_BYTES, L0_SLOT_BYTES / sizeof(B));
        LoadDataToL0B<rightTranspose>(L0BTensor, bL1Tensor, param, k * L1Boffset,
                                      (k == (kLoops - 1) && param.loadK != 0) ? (((tileK + 15) >> 4) << 4) : tileK,
                                      GetLoadN(param));
        Mutex::Unlock<PIPE_MTE1>(bMutex); // mte1搬运完后，通知可以开始matmul
        // L0B mutex covers both loads on PIPE_MTE1, as in the common split path.
        Mutex::Lock<PIPE_M>(bMutex); // matmul等mte1：L0B数据搬运完成后才能开始matmul

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) {
            mmadParams.m = param.realM;
        }
        mmadParams.n = param.singleN;
        mmadParams.k = tileK;
        if (mmadParams.m == 1) { // m等于1或默认开GEMV模式，文档上没有写怎么关闭GEMV，所以规避当做矩阵运算
            mmadParams.m = 16;
        }
        mmadParams.cmatrixInitVal = (k == 0);
        mmadParams.cmatrixSource = false;

        Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);

        Mutex::Unlock<PIPE_M>(aMutex); // matmul完成后，通知mte1可以开始搬运新数据到L0A
        Mutex::Unlock<PIPE_M>(bMutex); // matmul完成后，通知mte1可以开始搬运新数据到L0B
    }
}

template <uint32_t baseN, bool leftTranspose, bool rightTranspose, bool singleL0B = false, typename A, typename B,
          typename C>
__aicore__ inline void MatmulN(const LocalTensor<A> &aL1Tensor, const LocalTensor<B> &bL1Tensor, uint32_t &l0aIdx,
                               uint32_t &l0bIdx, const LocalTensor<C> &cL0Tensor, const MMParam &param)
{
    uint32_t nLoops = (param.singleN + baseN - 1) / baseN; // 尾块处理
    uint32_t tailSize = param.singleN % baseN;
    uint32_t tailN = tailSize ? tailSize : baseN;
    uint64_t L1Boffset = rightTranspose ? (baseN << 4) : ((param.singleK + 15) >> 4 << 4) * baseN;
    uint64_t L0Coffset = ((param.singleM + 15) >> 4 << 4) * baseN;
    if (param.realM != 0) {
        L0Coffset = ((param.realM + 15) >> 4 << 4) * baseN;
    }

    l0aIdx ^= 1U;
    uint32_t aMutex = L0A_MUTEX_BASE + l0aIdx;
    Mutex::Lock<PIPE_MTE1>(aMutex); // mte1等Matmul：上一轮matmul完成后才能搬运新数据到L0A
    LocalTensor<A> L0ATensor = LocalTensor<A>(TPosition::A2, l0aIdx * L0_SLOT_BYTES, L0_SLOT_BYTES / sizeof(A));
    LoadDataToL0A<leftTranspose>(L0ATensor, aL1Tensor, param, 0, GetLoadK(param), param.singleM); // s2*d,d,s2
    for (uint32_t n = 0; n < nLoops; n++) {
        uint32_t tileN = (n == (nLoops - 1)) ? tailN : baseN;
        uint32_t loadTileN = (n == (nLoops - 1) && param.loadN != 0) ? (((tileN + 15) >> 4) << 4) : tileN;

        if constexpr (!singleL0B) {
            l0bIdx ^= 1U;
        }
        const uint32_t bIdx = singleL0B ? 0U : l0bIdx;
        uint32_t bMutex = L0B_MUTEX_BASE + bIdx;
        Mutex::Lock<PIPE_MTE1>(bMutex); // mte1等Matmul：上一轮matmul完成后才能搬运新数据到L0B
        LocalTensor<B> L0BTensor = LocalTensor<B>(TPosition::B2, bIdx * L0_SLOT_BYTES, L0_SLOT_BYTES / sizeof(B));
        LoadDataToL0B<rightTranspose>(L0BTensor, bL1Tensor, param, n * L1Boffset, GetLoadK(param), loadTileN);
        Mutex::Unlock<PIPE_MTE1>(bMutex); // mte1搬运完后，通知可以开始matmul
        // L0B mutex covers both loads on PIPE_MTE1, as in the common split path.
        Mutex::Lock<PIPE_M>(bMutex); // matmul等mte1：L0B数据搬运完成后才能开始matmul

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) {
            mmadParams.m = param.realM;
        }
        mmadParams.n = tileN;
        mmadParams.k = param.singleK;
        if (mmadParams.m == 1) {
            mmadParams.m = FP16_ONE_FRACTAL_ELEMENT;
        }
        mmadParams.cmatrixInitVal = true;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = 0;
        Mmad(cL0Tensor[n * L0Coffset], L0ATensor, L0BTensor, mmadParams);

        Mutex::Unlock<PIPE_M>(bMutex); // matmul完成后，通知mte1可以开始搬运新数据到L0B
    }
    Mutex::Unlock<PIPE_M>(aMutex); // matmul完成后，通知mte1可以开始搬运新数据到L0A
}

} // namespace FlashAttnMatmul
#endif
