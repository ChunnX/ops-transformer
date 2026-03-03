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
 * \file vf_mul.h
 * \brief
 */

#ifndef VF_MUL_H
#define VF_MUL_H
#include "kernel_operator.h"
#include "../../compressor_comm.h"
using namespace AscendC;

constexpr uint32_t FLOATBYTE = 4;
constexpr uint32_t baseD8 = 8;
constexpr uint32_t baseD16 = 16;
constexpr uint32_t baseD32 = 32;
constexpr uint32_t baseD64 = 64;
constexpr uint32_t baseD128 = 128;
constexpr uint32_t baseD256 = 256;
constexpr uint32_t baseD512 = 512;

template <typename T>
__simd_vf__ void MulReduceSumbase8VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                         const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                         const uint32_t baseD)
{
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vreg1;
    MicroAPI::RegTensor<T> vregSum;
    MicroAPI::RegTensor<T> vregSum0;
    MicroAPI::RegTensor<T> vregSum1;
    MicroAPI::RegTensor<T> vregSum2;
    MicroAPI::RegTensor<T> vregSum3;
    MicroAPI::RegTensor<T> vregSum4;
    MicroAPI::RegTensor<T> vregSum5;
    MicroAPI::RegTensor<T> vregSum6;
    MicroAPI::RegTensor<T> vregSum7;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg mask0 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL8>();
    MicroAPI::MaskReg mask1;
    MicroAPI::MaskReg mask2;
    MicroAPI::MaskReg mask3;
    MicroAPI::MaskReg mask4;
    MicroAPI::MaskReg mask5;
    MicroAPI::MaskReg mask6;
    MicroAPI::MaskReg mask7;
    MicroAPI::ShiftLeft(mask1, mask0, 8, mask);
    MicroAPI::ShiftLeft(mask2, mask0, 16, mask);
    MicroAPI::ShiftLeft(mask3, mask0, 24, mask);
    MicroAPI::ShiftLeft(mask4, mask0, 32, mask);
    MicroAPI::ShiftLeft(mask5, mask0, 40, mask);
    MicroAPI::ShiftLeft(mask6, mask0, 48, mask);
    MicroAPI::ShiftLeft(mask7, mask0, 56, mask);
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(vregSum, 0, mask);
        for (uint32_t rLoop = 0; rLoop < CeilDivT(coff * cmpRatio, 8); rLoop++) {
            MicroAPI::LoadAlign(vreg0, kvAddr + rLoop * baseD64);
            MicroAPI::LoadAlign(vreg1, scoreAddr + rLoop * baseD64);
            MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
            MicroAPI::Add(vregSum, vregSum, vreg0, mask);
        }
        MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum0, vregSum, mask0);
        if (coff * cmpRatio > 2) {
            MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum1, vregSum, mask1);
            MicroAPI::Add(vregSum0, vregSum0, vregSum1, mask0);
        }
        if (coff * cmpRatio > 4) {
            MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum2, vregSum, mask2);
            MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum3, vregSum, mask3);
            MicroAPI::Add(vregSum2, vregSum2, vregSum3, mask0);
            MicroAPI::Add(vregSum0, vregSum0, vregSum2, mask0);
        }
        if (coff * cmpRatio > 8) {
            MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum4, vregSum, mask4);
            MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum5, vregSum, mask5);
            MicroAPI::Add(vregSum4, vregSum4, vregSum5, mask0);
            MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum6, vregSum, mask6);
            MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum7, vregSum, mask7);
            MicroAPI::Add(vregSum6, vregSum6, vregSum7, mask0);
            MicroAPI::Add(vregSum4, vregSum4, vregSum6, mask0);
            MicroAPI::Add(vregSum0, vregSum0, vregSum4, mask0);
        }
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD, vregSum0, mask0);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase16VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                          const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                          const uint32_t baseD)
{
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vreg1;
    MicroAPI::RegTensor<T> vregSum;
    MicroAPI::RegTensor<T> vregSum0;
    MicroAPI::RegTensor<T> vregSum1;
    MicroAPI::RegTensor<T> vregSum2;
    MicroAPI::RegTensor<T> vregSum3;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg mask0 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL16>();
    MicroAPI::MaskReg mask1;
    MicroAPI::MaskReg mask2;
    MicroAPI::MaskReg mask3;
    MicroAPI::ShiftLeft(mask1, mask0, 16, mask);
    MicroAPI::ShiftLeft(mask2, mask0, 32, mask);
    MicroAPI::ShiftLeft(mask3, mask0, 48, mask);
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(vregSum, 0, mask);
        for (uint32_t rLoop = 0; rLoop < CeilDivT(coff * cmpRatio, 4); rLoop++) {
            MicroAPI::LoadAlign(vreg0, kvAddr + rLoop * baseD64);
            MicroAPI::LoadAlign(vreg1, scoreAddr + rLoop * baseD64);
            MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
            MicroAPI::Add(vregSum, vregSum, vreg0, mask);
        }
        MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum0, vregSum, mask0);
        if (coff * cmpRatio > 2) {
            MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum1, vregSum, mask1);
            MicroAPI::Add(vregSum0, vregSum0, vregSum1, mask0);
        }
        if (coff * cmpRatio > 4) {
            MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum2, vregSum, mask2);
            MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum3, vregSum, mask3);
            MicroAPI::Add(vregSum2, vregSum2, vregSum3, mask0);
            MicroAPI::Add(vregSum0, vregSum0, vregSum2, mask0);
        }
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD, vregSum0, mask0);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase32VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                          const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                          const uint32_t baseD)
{
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vreg1;
    MicroAPI::RegTensor<T> vregSum;
    MicroAPI::RegTensor<T> vregSum0;
    MicroAPI::RegTensor<T> vregSum1;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskLHalf = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL32>();
    MicroAPI::MaskReg maskRHalf;
    MicroAPI::Not(maskRHalf, maskLHalf, mask);
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(vregSum, 0, mask);
        for (uint32_t rLoop = 0; rLoop < CeilDivT(coff * cmpRatio, 2); rLoop++) {
            MicroAPI::LoadAlign(vreg0, kvAddr + rLoop * baseD64);
            MicroAPI::LoadAlign(vreg1, scoreAddr + rLoop * baseD64);
            MicroAPI::Mul(vreg0, vreg0, vreg1, mask);
            MicroAPI::Add(vregSum, vregSum, vreg0, mask);
        }
        MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum0, vregSum, maskLHalf);
        if (coff * cmpRatio > 2) {
            MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::NO_STORE_REG>(vregSum1, vregSum, maskRHalf);
            MicroAPI::Add(vregSum0, vregSum0, vregSum1, maskLHalf);
        }
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD, vregSum0, maskLHalf);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase64VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                          const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                          const uint32_t baseD)
{
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vreg1;
    MicroAPI::RegTensor<T> vregMul;
    MicroAPI::RegTensor<T> vregSum;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    uint32_t offset = 0;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(vregSum, 0, mask);
        for (uint32_t rLoop = 0; rLoop < coff * cmpRatio; rLoop++) {
            MicroAPI::LoadAlign(vreg0, kvAddr + offset);
            MicroAPI::LoadAlign(vreg1, scoreAddr + offset);
            MicroAPI::Mul(vregMul, vreg0, vreg1, mask);
            MicroAPI::Add(vregSum, vregSum, vregMul, mask);
            offset += baseD;
        }
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD, vregSum, mask);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase128VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                           const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                           const uint32_t baseD)
{
    MicroAPI::RegTensor<T> vreg00;
    MicroAPI::RegTensor<T> vreg01;
    MicroAPI::RegTensor<T> vreg10;
    MicroAPI::RegTensor<T> vreg11;
    MicroAPI::RegTensor<T> vregMul0;
    MicroAPI::RegTensor<T> vregMul1;
    MicroAPI::RegTensor<T> vregSum0;
    MicroAPI::RegTensor<T> vregSum1;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    uint32_t offset = 0;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(vregSum0, 0, mask);
        MicroAPI::Duplicate(vregSum1, 0, mask);
        for (uint32_t rLoop = 0; rLoop < coff * cmpRatio; rLoop++) {
            MicroAPI::LoadAlign(vreg00, kvAddr + offset);
            MicroAPI::LoadAlign(vreg01, kvAddr + offset + baseD64);
            MicroAPI::LoadAlign(vreg10, scoreAddr + offset);
            MicroAPI::LoadAlign(vreg11, scoreAddr + offset + baseD64);
            MicroAPI::Mul(vregMul0, vreg00, vreg10, mask);
            MicroAPI::Mul(vregMul1, vreg01, vreg11, mask);
            MicroAPI::Add(vregSum0, vregSum0, vregMul0, mask);
            MicroAPI::Add(vregSum1, vregSum1, vregMul1, mask);
            offset += 2 * baseD;
        }
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD, vregSum0, mask);
        MicroAPI::StoreAlign(outputAddr + baseD64 + scLoop * baseD, vregSum1, mask);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase256VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                           const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                           const uint32_t baseD)
{
    MicroAPI::RegTensor<T> vreg00;
    MicroAPI::RegTensor<T> vreg01;
    MicroAPI::RegTensor<T> vreg02;
    MicroAPI::RegTensor<T> vreg03;
    MicroAPI::RegTensor<T> vreg10;
    MicroAPI::RegTensor<T> vreg11;
    MicroAPI::RegTensor<T> vreg12;
    MicroAPI::RegTensor<T> vreg13;
    MicroAPI::RegTensor<T> vregMul0;
    MicroAPI::RegTensor<T> vregMul1;
    MicroAPI::RegTensor<T> vregMul2;
    MicroAPI::RegTensor<T> vregMul3;
    MicroAPI::RegTensor<T> vregSum0;
    MicroAPI::RegTensor<T> vregSum1;
    MicroAPI::RegTensor<T> vregSum2;
    MicroAPI::RegTensor<T> vregSum3;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    uint32_t offset = 0;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(vregSum0, 0, mask);
        MicroAPI::Duplicate(vregSum1, 0, mask);
        MicroAPI::Duplicate(vregSum2, 0, mask);
        MicroAPI::Duplicate(vregSum3, 0, mask);
        for (uint32_t rLoop = 0; rLoop < coff * cmpRatio; rLoop++) {
            MicroAPI::LoadAlign(vreg00, kvAddr + offset);
            MicroAPI::LoadAlign(vreg01, kvAddr + offset + baseD64);
            MicroAPI::LoadAlign(vreg02, kvAddr + offset + 2 * baseD64);
            MicroAPI::LoadAlign(vreg03, kvAddr + offset + 3 * baseD64);
            MicroAPI::LoadAlign(vreg10, scoreAddr + offset);
            MicroAPI::LoadAlign(vreg11, scoreAddr + offset + baseD64);
            MicroAPI::LoadAlign(vreg12, scoreAddr + offset + 2 * baseD64);
            MicroAPI::LoadAlign(vreg13, scoreAddr + offset + 3 * baseD64);
            MicroAPI::Mul(vregMul0, vreg00, vreg10, mask);
            MicroAPI::Mul(vregMul1, vreg01, vreg11, mask);
            MicroAPI::Mul(vregMul2, vreg02, vreg12, mask);
            MicroAPI::Mul(vregMul3, vreg03, vreg13, mask);
            MicroAPI::Add(vregSum0, vregSum0, vregMul0, mask);
            MicroAPI::Add(vregSum1, vregSum1, vregMul1, mask);
            MicroAPI::Add(vregSum2, vregSum2, vregMul2, mask);
            MicroAPI::Add(vregSum3, vregSum4, vregMul3, mask);
            offset += 4 * baseD;
        }
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD, vregSum0, mask);
        MicroAPI::StoreAlign(outputAddr + baseD64 + scLoop * baseD, vregSum1, mask);
        MicroAPI::StoreAlign(outputAddr + 2 * baseD64 + scLoop * baseD, vregSum2, mask);
        MicroAPI::StoreAlign(outputAddr + 3 * baseD64 + scLoop * baseD, vregSum3, mask);
    }
}

template <typename T>
__simd_vf__ void MulReduceSumbase512VFImpl(__ubuf__ T *kvAddr, __ubuf__ T *scoreAddr, __ubuf__ T *outputAddr,
                                           const uint32_t coff, const uint32_t cmpRatio, const uint32_t scLoopCnt,
                                           const uint32_t baseD)
{
    MicroAPI::RegTensor<T> vreg00;
    MicroAPI::RegTensor<T> vreg01;
    MicroAPI::RegTensor<T> vreg02;
    MicroAPI::RegTensor<T> vreg03;
    MicroAPI::RegTensor<T> vreg04;
    MicroAPI::RegTensor<T> vreg05;
    MicroAPI::RegTensor<T> vreg06;
    MicroAPI::RegTensor<T> vreg07;
    MicroAPI::RegTensor<T> vreg10;
    MicroAPI::RegTensor<T> vreg11;
    MicroAPI::RegTensor<T> vreg12;
    MicroAPI::RegTensor<T> vreg13;
    MicroAPI::RegTensor<T> vreg14;
    MicroAPI::RegTensor<T> vreg15;
    MicroAPI::RegTensor<T> vreg16;
    MicroAPI::RegTensor<T> vreg17;
    MicroAPI::RegTensor<T> vregMul0;
    MicroAPI::RegTensor<T> vregMul1;
    MicroAPI::RegTensor<T> vregMul2;
    MicroAPI::RegTensor<T> vregMul3;
    MicroAPI::RegTensor<T> vregMul4;
    MicroAPI::RegTensor<T> vregMul5;
    MicroAPI::RegTensor<T> vregMul6;
    MicroAPI::RegTensor<T> vregMul7;
    MicroAPI::RegTensor<T> vregSum0;
    MicroAPI::RegTensor<T> vregSum1;
    MicroAPI::RegTensor<T> vregSum2;
    MicroAPI::RegTensor<T> vregSum3;
    MicroAPI::RegTensor<T> vregSum4;
    MicroAPI::RegTensor<T> vregSum5;
    MicroAPI::RegTensor<T> vregSum6;
    MicroAPI::RegTensor<T> vregSum7;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    uint32_t offset = 0;
    for (uint32_t scLoop = 0; scLoop < scLoopCnt; scLoop++) {
        MicroAPI::Duplicate(vregSum0, 0, mask);
        MicroAPI::Duplicate(vregSum1, 0, mask);
        MicroAPI::Duplicate(vregSum2, 0, mask);
        MicroAPI::Duplicate(vregSum3, 0, mask);
        MicroAPI::Duplicate(vregSum4, 0, mask);
        MicroAPI::Duplicate(vregSum5, 0, mask);
        MicroAPI::Duplicate(vregSum6, 0, mask);
        MicroAPI::Duplicate(vregSum7, 0, mask);
        for (uint32_t rLoop = 0; rLoop < coff * cmpRatio; rLoop++) {
            MicroAPI::LoadAlign(vreg00, kvAddr + offset);
            MicroAPI::LoadAlign(vreg01, kvAddr + offset + baseD64);
            MicroAPI::LoadAlign(vreg02, kvAddr + offset + 2 * baseD64);
            MicroAPI::LoadAlign(vreg03, kvAddr + offset + 3 * baseD64);
            MicroAPI::LoadAlign(vreg04, kvAddr + offset + 4 * baseD64);
            MicroAPI::LoadAlign(vreg05, kvAddr + offset + 5 * baseD64);
            MicroAPI::LoadAlign(vreg06, kvAddr + offset + 6 * baseD64);
            MicroAPI::LoadAlign(vreg07, kvAddr + offset + 7 * baseD64);
            MicroAPI::LoadAlign(vreg10, scoreAddr + offset);
            MicroAPI::LoadAlign(vreg11, scoreAddr + offset + baseD64);
            MicroAPI::LoadAlign(vreg12, scoreAddr + offset + 2 * baseD64);
            MicroAPI::LoadAlign(vreg13, scoreAddr + offset + 3 * baseD64);
            MicroAPI::LoadAlign(vreg14, scoreAddr + offset + 4 * baseD64);
            MicroAPI::LoadAlign(vreg15, scoreAddr + offset + 5 * baseD64);
            MicroAPI::LoadAlign(vreg16, scoreAddr + offset + 6 * baseD64);
            MicroAPI::LoadAlign(vreg17, scoreAddr + offset + 7 * baseD64);
            MicroAPI::Mul(vregMul0, vreg00, vreg10, mask);
            MicroAPI::Mul(vregMul1, vreg01, vreg11, mask);
            MicroAPI::Mul(vregMul2, vreg02, vreg12, mask);
            MicroAPI::Mul(vregMul3, vreg03, vreg13, mask);
            MicroAPI::Mul(vregMul4, vreg04, vreg14, mask);
            MicroAPI::Mul(vregMul5, vreg05, vreg15, mask);
            MicroAPI::Mul(vregMul6, vreg06, vreg16, mask);
            MicroAPI::Mul(vregMul7, vreg07, vreg17, mask);
            MicroAPI::Add(vregSum0, vregSum0, vregMul0, mask);
            MicroAPI::Add(vregSum1, vregSum1, vregMul1, mask);
            MicroAPI::Add(vregSum2, vregSum2, vregMul2, mask);
            MicroAPI::Add(vregSum3, vregSum4, vregMul3, mask);
            MicroAPI::Add(vregSum4, vregSum4, vregMul4, mask);
            MicroAPI::Add(vregSum5, vregSum5, vregMul5, mask);
            MicroAPI::Add(vregSum6, vregSum6, vregMul6, mask);
            MicroAPI::Add(vregSum7, vregSum7, vregMul7, mask);
            offset += 4 * baseD;
        }
        MicroAPI::StoreAlign(outputAddr + scLoop * baseD, vregSum0, mask);
        MicroAPI::StoreAlign(outputAddr + baseD64 + scLoop * baseD, vregSum1, mask);
        MicroAPI::StoreAlign(outputAddr + 2 * baseD64 + scLoop * baseD, vregSum2, mask);
        MicroAPI::StoreAlign(outputAddr + 3 * baseD64 + scLoop * baseD, vregSum3, mask);
        MicroAPI::StoreAlign(outputAddr + 4 * baseD64 + scLoop * baseD, vregSum4, mask);
        MicroAPI::StoreAlign(outputAddr + 5 * baseD64 + scLoop * baseD, vregSum5, mask);
        MicroAPI::StoreAlign(outputAddr + 6 * baseD64 + scLoop * baseD, vregSum6, mask);
        MicroAPI::StoreAlign(outputAddr + 7 * baseD64 + scLoop * baseD, vregSum7, mask);
    }
}

/**
 * @brief MulReduceSumbaseVF 包含mul和reducesum
 * @param outputLocal 输出tensor []
 * @param coff 
 * @param cmpRatio 压缩块大小
 * @param baseD  核内d轴切分大小
 * @param scLoopCnt  sc数,
 */

//当前仅支持coff * cmpRatio为2的幂的情况
template <typename T>
__aicore__ inline void MulReduceSumbaseVF(LocalTensor<T> &kvLocal, LocalTensor<T> &scoreLocal,
                                          LocalTensor<T> &outputLocal, const uint32_t coff, const uint32_t cmpRatio,
                                          const uint32_t baseD, const uint32_t scLoopCnt)
{

    __ubuf__ T *kvAddr = (__ubuf__ T *)kvLocal.GetPhyAddr();
    __ubuf__ T *scoreAddr = (__ubuf__ T *)scoreLocal.GetPhyAddr();
    __ubuf__ T *outputAddr = (__ubuf__ T *)outputLocal.GetPhyAddr() + outIdx;
    if (baseD == baseD8) {
        MulReduceSumbase8VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == baseD16) {
        MulReduceSumbase16VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == baseD32) {
        MulReduceSumbase32VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == baseD64) {
        MulReduceSumbase64VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == baseD128) {
        MulReduceSumbase128VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == baseD256) {
        MulReduceSumbase256VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    } else if (baseD == baseD512) {
        MulReduceSumbase512VFImpl(kvAddr, scoreAddr, outputAddr, coff, cmpRatio, scLoopCnt, baseD);
    }
}


#endif