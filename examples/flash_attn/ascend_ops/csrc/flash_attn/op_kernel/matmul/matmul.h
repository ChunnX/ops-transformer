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
 * \file matmul.h
 * \brief
 */
#ifndef MATMUL_H
#define MATMUL_H
#include "../buffer_manager/buffer_policy.h"
using namespace AscendC;

namespace fa_base_matmul {
constexpr uint32_t UNITFLAG_ENABLE = 2;
constexpr uint32_t UNITFLAG_EN_OUTER_LAST = 3;

struct MMParam {
    uint32_t singleM;
    uint32_t singleN;
    uint32_t singleK;
    bool isLeftTranspose;
    bool isRightTranspose;
    bool cmatrixInitVal = true;
    bool isOutKFisrt = true; // 默认值为true，true：在L1切K轴的场景中，表示首轮K
    uint32_t unitFlag = 0;  // 0：disable: 不配置unitFlag
                            // 2：enable: 行为在切K接口中（MatmulK），会将mmadParams.unitFlag设置为 0b10
                            // 3：enable: 行为在切K接口中（MatmulK），在k的最后一轮循环，会将mmadParams.unitFlag设置为 0b11
                            // 外部使用时，在外层k循环的最后一轮将该参数配置为3
    uint32_t realM = 0; // bmm2以s1realsize为M轴，不赋值时不影响现有代码逻辑
};

__aicore__ inline MMParam MakeMMParam(uint32_t singleM, uint32_t singleN, uint32_t singleK, bool isLeftTranspose,
                                      bool isRightTranspose, bool cmatrixInitVal = true, bool isOutKFisrt = true,
                                      uint32_t unitFlag = 0, uint32_t realM = 0)
{
    return {.singleM = singleM,
            .singleN = singleN,
            .singleK = singleK,
            .isLeftTranspose = isLeftTranspose,
            .isRightTranspose = isRightTranspose,
            .cmatrixInitVal = cmatrixInitVal,
            .isOutKFisrt = isOutKFisrt,
            .unitFlag = unitFlag,
            .realM = realM
            };
}

enum class ABLayout {
    MK = 0,
    KM = 1,
    KN = 2,
    NK = 3,
};

template <typename T>
__aicore__ inline T AlignUp(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

#if ((__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102))
template <typename T>
__aicore__ inline uint32_t GetBlockNum(uint32_t size) {
    return ((size + 15) >> 4 << 4) >> 4;
}
// L1->L0A + 切k/切M/全载
template <typename T>
__aicore__ inline void LoadDataToL0A(LocalTensor<T>& aL0Tensor, const LocalTensor<T>& aL1Tensor, 
                                    const MMParam& mmParam, uint64_t L1Aoffset, uint32_t kSplitSize, uint32_t mSplitSize)
{
    LoadData2DParamsV2 loadData2DParamsA; // 基础API LoadData的参数结构体
    loadData2DParamsA.mStartPosition = 0; // 以M*K矩阵为例，源矩阵M轴方向的起始位置，单位为16 element
    loadData2DParamsA.kStartPosition = 0; // 以M*K矩阵为例，源矩阵K轴方向的起始位置，单位为32B
    loadData2DParamsA.ifTranspose = mmParam.isLeftTranspose; // 是否启用转置功能，对每个分型矩阵进行转置
    if (loadData2DParamsA.ifTranspose) {
        loadData2DParamsA.mStep = ((kSplitSize + 15) >> 4 << 4) >> 4; // 以M*K矩阵为例,源矩阵M轴方向搬运长度(S1向上对齐分形(512B),16*16个f16->向上对齐16)，单位为16 element,取值范围：mStep属于[0,255]
        loadData2DParamsA.kStep = GetBlockNum<T>(mSplitSize); // 以M*K矩阵为例,源矩阵K轴方向搬运长度(qkD个f16)，单位为32B,取值范围：nStep属于[0,255]
    } else {
        loadData2DParamsA.mStep = ((mSplitSize + 15) >> 4 << 4) >> 4; // 以M*K矩阵为例,源矩阵M轴方向搬运长度(S1向上对齐分形(512B),16*16个f16->向上对齐16)，单位为16 element,取值范围：mStep属于[0,255]
        loadData2DParamsA.kStep = GetBlockNum<T>(kSplitSize); // 以M*K矩阵为例,源矩阵K轴方向搬运长度(qkD个f16)，单位为32B,取值范围：nStep属于[0,255]
    }
    loadData2DParamsA.srcStride = loadData2DParamsA.ifTranspose ? ((mmParam.singleK + 15) >> 4 << 4) >> 4 : loadData2DParamsA.mStep;
    if (mmParam.realM != 0) {
        loadData2DParamsA.mStep = ((mmParam.realM + 15) >> 4 << 4) >> 4;
    }
    loadData2DParamsA.dstStride = loadData2DParamsA.ifTranspose ? (mSplitSize + 15) >> 4 : loadData2DParamsA.mStep;
    LoadData(aL0Tensor, aL1Tensor[L1Aoffset], loadData2DParamsA);
}

// L1->L0B + 切k/切M/全载
template <typename T>
__aicore__ inline void LoadDataToL0B(LocalTensor<T>& bL0Tensor, const LocalTensor<T>& bL1Tensor,
                                    const MMParam& mmParam, uint64_t L1Boffset, uint32_t kSplitSize, uint32_t nSplitSize, int nLoops = 1)
{
    LoadData2DParamsV2 loadData2DParamsB; // 基础API LoadData的参数结构体
    loadData2DParamsB.mStartPosition = 0; // 以M*K矩阵为例，源矩阵M轴方向的起始位置，单位为16 element
    loadData2DParamsB.kStartPosition = 0; // 以M*K矩阵为例，源矩阵K轴方向的起始位置，单位为32B
    loadData2DParamsB.ifTranspose = !mmParam.isRightTranspose; // 是否启用转置功能，对每个分型矩阵进行转置
    if (loadData2DParamsB.ifTranspose) {
        loadData2DParamsB.mStep = ((kSplitSize + 15) >> 4 << 4) >> 4; // 以M*K矩阵为例,源矩阵M轴方向搬运长度(S1向上对齐分形(512B),16*16个f16->向上对齐16)，单位为16 element,取值范围：mStep属于[0,255]
        loadData2DParamsB.kStep = GetBlockNum<T>(nSplitSize); // 以M*K矩阵为例,源矩阵K轴方向搬运长度(qkD个f16)，单位为32B,取值范围：nStep属于[0,255]
    } else {
        loadData2DParamsB.mStep = ((nSplitSize + 15) >> 4 << 4) >> 4; // 以M*K矩阵为例,源矩阵M轴方向搬运长度(S1向上对齐分形(512B),16*16个f16->向上对齐16)，单位为16 element,取值范围：mStep属于[0,255]
        loadData2DParamsB.kStep = GetBlockNum<T>(kSplitSize); // 以M*K矩阵为例,源矩阵K轴方向搬运长度(qkD个f16)，单位为32B,取值范围：nStep属于[0,255]
    }
    loadData2DParamsB.srcStride = loadData2DParamsB.ifTranspose ? (((mmParam.singleK + 15) >> 4 << 4) >> 4) : (((mmParam.singleN + 15 ) >> 4 << 4) >> 4); // 以M*K矩阵为例，源矩阵K方向前一个分形起始地址与后一个分形起始地址的间隔，单位：512B
    loadData2DParamsB.dstStride = loadData2DParamsB.ifTranspose ? (nSplitSize + 15) >> 4 : loadData2DParamsB.mStep; // 以M*K矩阵为例，目标矩阵K方向前一个分形起始地址与后一个分形起始地址的间隔，单位：512B
    LoadData(bL0Tensor, bL1Tensor[L1Boffset], loadData2DParamsB);
}

#endif
// 全载
// 外部L1切入K时，需要传入cmatrixInitVal的标记
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          typename L0AType, typename L0BType, typename AScaleType = fp8_e8m0_t, typename BScaleType = fp8_e8m0_t,
          typename L0ADType = A, typename L0BDType = B>
__aicore__ inline void MatmulFull(const LocalTensor<A> &aL1Tensor,
                                  const LocalTensor<B> &bL1Tensor,
                                  L0AType &aL0BuffsDb,
                                  L0BType &bL0BuffsDb,
                                  const LocalTensor<C> &cL0Tensor,
                                  struct MMParam &param,
                                  const LocalTensor<AScaleType> &aScaleL1Tensor = LocalTensor<AScaleType>(),
                                  const LocalTensor<BScaleType> &bScaleL1Tensor = LocalTensor<AScaleType>())
{
    Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
    l0aBuffer.Wait<HardEvent::M_MTE1>();
    LocalTensor<L0ADType> L0ATensor = l0aBuffer.GetTensor<L0ADType>();
    LoadDataToL0A(L0ATensor, aL1Tensor, param, 0, param.singleK, param.singleM); // s2*d,d,s2
    l0aBuffer.Set<HardEvent::MTE1_M>();

    Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
    l0bBuffer.Wait<HardEvent::M_MTE1>();
    LocalTensor<L0BDType> L0BTensor = l0bBuffer.GetTensor<L0BDType>();
    LoadDataToL0B(L0BTensor, bL1Tensor, param, 0, param.singleK, param.singleN);
    l0bBuffer.Set<HardEvent::MTE1_M>();

    l0aBuffer.Wait<HardEvent::MTE1_M>();
    l0bBuffer.Wait<HardEvent::MTE1_M>();

    MmadParams mmadParams;
    mmadParams.m = param.singleM;
    if (param.realM != 0) {
        mmadParams.m = param.realM;
    }
    mmadParams.n = param.singleN;
    mmadParams.k = param.singleK;
    mmadParams.cmatrixInitVal = param.isOutKFisrt;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = param.unitFlag;
    if (mmadParams.m == 1) {
        mmadParams.m = 16;
    }
 
    Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);
 
    l0aBuffer.Set<HardEvent::M_MTE1>();
    l0bBuffer.Set<HardEvent::M_MTE1>();
}

// 切K
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          typename L0AType, typename L0BType, typename AScaleType = fp8_e8m0_t, typename BScaleType = fp8_e8m0_t,
          typename L0ADType = A, typename L0BDType = B>
__aicore__ inline void MatmulK(const LocalTensor<A> &aL1Tensor,
                              const LocalTensor<B> &bL1Tensor,
                              L0AType &aL0BuffsDb,
                              L0BType &bL0BuffsDb,
                              const LocalTensor<C> &cL0Tensor,
                              const MMParam &param,
                              const LocalTensor<AScaleType> &aScaleL1Tensor = LocalTensor<AScaleType>(),
                              const LocalTensor<BScaleType> &bScaleL1Tensor = LocalTensor<AScaleType>())
{
    uint32_t kLoops = (param.singleK + baseK - 1) / baseK;
    uint32_t tailSize = param.singleK % baseK;
    uint32_t tailK = tailSize ? tailSize : baseK;
    uint64_t L1Aoffset = param.isLeftTranspose ? baseK << 4 : ((param.singleM + 15) >> 4 << 4) * baseK;
    uint64_t L1Boffset = param.isRightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK : baseK << 4; 
 
    for (uint32_t k = 0; k < kLoops; k++) {
        uint32_t tileK = (k == (kLoops - 1)) ? tailK : baseK;
        Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
        l0aBuffer.Wait<HardEvent::M_MTE1>(); // mte1等Matmul：上一轮matmul完成后才能搬运新数据到L0A
        LocalTensor<L0ADType> L0ATensor = l0aBuffer.GetTensor<L0ADType>();
        LoadDataToL0A(L0ATensor, aL1Tensor, param, k * L1Aoffset, tileK, param.singleM); // s2*d,d,s2

        Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
        l0bBuffer.Wait<HardEvent::M_MTE1>(); // mte1等Matmul：上一轮matmul完成后才能搬运新数据到L0B
        LocalTensor<L0BDType> L0BTensor = l0bBuffer.GetTensor<L0BDType>();
        uint64_t loopNum = param.isRightTranspose ? 1 : kLoops;
        LoadDataToL0B(L0BTensor, bL1Tensor, param, k * L1Boffset, tileK, param.singleN, loopNum);
        l0bBuffer.Set<HardEvent::MTE1_M>(); // mte1搬运完后，通知可以开始matmul
        // l0aBuffer和l0bBuffer共用MTE1_M，在D=512场景减少同步指令数量，提升性能
        l0bBuffer.Wait<HardEvent::MTE1_M>(); // matmul等mte1：L0B数据搬运完成后才能开始matmul
 
        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) {
            mmadParams.m = param.realM;
        }
        mmadParams.n = param.singleN;
        mmadParams.k = tileK;
        if (mmadParams.m == 1) {  // m等于1或默认开GEMV模式，文档上没有写怎么关闭GEMV，所以规避当做矩阵运算
            mmadParams.m = 16;
        }
        mmadParams.cmatrixInitVal = param.isOutKFisrt && (k == 0);
        mmadParams.cmatrixSource = false;
        if (param.unitFlag != 0) {
            mmadParams.unitFlag = (param.unitFlag == UNITFLAG_EN_OUTER_LAST) && (k == kLoops - 1) ?
                                  UNITFLAG_EN_OUTER_LAST : UNITFLAG_ENABLE;
        }
        Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);
 
        l0aBuffer.Set<HardEvent::M_MTE1>(); // matmul完成后，通知mte1可以开始搬运新数据到L0A
        l0bBuffer.Set<HardEvent::M_MTE1>(); // matmul完成后，通知mte1可以开始搬运新数据到L0B
    }
}

// 切N
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          typename L0AType, typename L0BType, typename AScaleType = fp8_e8m0_t, typename BScaleType = fp8_e8m0_t,
          typename L0ADType = A, typename L0BDType = B>
__aicore__ inline void MatmulN(const LocalTensor<A> &aL1Tensor,
                              const LocalTensor<B> &bL1Tensor,
                              L0AType &aL0BuffsDb,
                              L0BType &bL0BuffsDb,
                              const LocalTensor<C> &cL0Tensor,
                              const MMParam &param,
                              const LocalTensor<AScaleType> &aScaleL1Tensor = LocalTensor<AScaleType>(),
                              const LocalTensor<BScaleType> &bScaleL1Tensor = LocalTensor<AScaleType>())
{
    uint32_t nLoops = (param.singleN + baseN - 1) / baseN; // 尾块处理
    uint32_t tailSize = param.singleN % baseN;
    uint32_t tailN = tailSize ? tailSize : baseN;
    uint64_t L1Boffset = param.isRightTranspose ? (baseN << 4) : ((param.singleK + 15) >> 4 << 4) * baseN;
    uint64_t L0Coffset = ((param.singleM + 15) >> 4 << 4) * baseN;
    if (param.realM != 0) {
        L0Coffset = ((param.realM + 15) >> 4 << 4) * baseN;
    }
 
    Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
    l0aBuffer.Wait<HardEvent::M_MTE1>(); // mte1等Matmul：上一轮matmul完成后才能搬运新数据到L0A
    LocalTensor<L0ADType> L0ATensor = l0aBuffer.GetTensor<L0ADType>();
    LoadDataToL0A(L0ATensor, aL1Tensor, param, 0, param.singleK, param.singleM); // s2*d,d,s2
    for (uint32_t n = 0; n < nLoops; n++) {
        uint32_t tileN = (n == (nLoops - 1)) ? tailN : baseN;
        
        Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
        l0bBuffer.Wait<HardEvent::M_MTE1>(); // mte1等Matmul：上一轮matmul完成后才能搬运新数据到L0B
        LocalTensor<L0BDType> L0BTensor = l0bBuffer.GetTensor<L0BDType>();
        uint64_t loopNum = param.isRightTranspose ? nLoops : 1;
        LoadDataToL0B(L0BTensor, bL1Tensor, param, n * L1Boffset, param.singleK, tileN, loopNum);
        l0bBuffer.Set<HardEvent::MTE1_M>(); // mte1搬运完后，通知可以开始matmul
        // l0aBuffer和l0bBuffer共用MTE1_M，在D=512场景减少同步指令数量，提升性能
        l0bBuffer.Wait<HardEvent::MTE1_M>(); // matmul等mte1：L0B数据搬运完成后才能开始matmul
 
        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) {
            mmadParams.m = param.realM;
        }
        mmadParams.n = tileN;
        mmadParams.k = param.singleK;
        if (mmadParams.m == 1) {
            mmadParams.m = 16;
        }
        mmadParams.cmatrixInitVal = param.isOutKFisrt;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = param.unitFlag;
        Mmad(cL0Tensor[n * L0Coffset], L0ATensor, L0BTensor, mmadParams);

        l0bBuffer.Set<HardEvent::M_MTE1>(); // matmul完成后，通知mte1可以开始搬运新数据到L0B
    }
    l0aBuffer.Set<HardEvent::M_MTE1>(); // matmul完成后，通知mte1可以开始搬运新数据到L0A
}

template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL, typename L0AType, typename L0BType>
__aicore__ inline void MatmulBase(const LocalTensor<A> &aL1Tensor,
                                  const LocalTensor<B> &bL1Tensor,
                                  L0AType &aL0BuffsDb,
                                  L0BType &bL0BuffsDb,
                                  const LocalTensor<C> &cL0Tensor,
                                  struct MMParam &param)
{
    if ((param.singleK + baseK - 1) / baseK > 1) {
        MatmulK<A, B, C, baseM, baseN, baseK, AL, BL>(aL1Tensor, bL1Tensor, aL0BuffsDb, bL0BuffsDb, cL0Tensor, param);
    } else if ((param.singleN + baseN - 1) / baseN > 1) {
        MatmulN<A, B, C, baseM, baseN, baseK, AL, BL>(aL1Tensor, bL1Tensor, aL0BuffsDb, bL0BuffsDb, cL0Tensor, param);
    } else {
        MatmulFull<A, B, C, baseM, baseN, baseK, AL, BL>(aL1Tensor, bL1Tensor, aL0BuffsDb, bL0BuffsDb, cL0Tensor, param);
    }
}

}
#endif
