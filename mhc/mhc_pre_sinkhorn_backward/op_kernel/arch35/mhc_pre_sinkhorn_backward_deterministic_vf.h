/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file mhc_pre_sinkhorn_backward_deterministic_vf.h
 * \brief MhcPreSinkhornBackward Deterministic VF Functions (RegBase)
 */

#ifndef MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_VF_H
#define MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_VF_H

#include "kernel_operator.h"

using namespace AscendC;
using namespace MicroAPI;

constexpr uint32_t VL_FP32 = AscendC::VECTOR_REG_WIDTH / sizeof(float);

template<typename T>
__aicore__ inline void ComputeForwardZVf(
    LocalTensor<T> zPreLocal, LocalTensor<T> zPostLocal, LocalTensor<T> zResLocal,
    LocalTensor<T> normOutLocal, LocalTensor<T> alphaLocal, LocalTensor<T> biasLocal,
    int64_t n, uint32_t totalElements)
{
    __local_mem__ T* zPrePtr = (__local_mem__ T*) zPreLocal.GetPhyAddr();
    __local_mem__ T* zPostPtr = (__local_mem__ T*) zPostLocal.GetPhyAddr();
    __local_mem__ T* zResPtr = (__local_mem__ T*) zResLocal.GetPhyAddr();
    __local_mem__ T* normOutPtr = (__local_mem__ T*) normOutLocal.GetPhyAddr();
    __local_mem__ T* alphaPtr = (__local_mem__ T*) alphaLocal.GetPhyAddr();
    __local_mem__ T* biasPtr = (__local_mem__ T*) biasLocal.GetPhyAddr();
    
    __VEC_SCOPE__ {
        RegTensor<T> normPreReg, normPostReg, normResReg;
        RegTensor<T> alphaPreReg, alphaPostReg, alphaResReg;
        RegTensor<T> biasPreReg, biasPostReg, biasResReg;
        RegTensor<T> zPreReg, zPostReg, zResReg;
        
        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();
        
        uint32_t loopCount = (totalElements + VL_FP32 - 1) / VL_FP32;
        
        for (uint16_t i = 0; i < loopCount; i++) {
            uint32_t calcSize = (i == loopCount - 1) ? 
                (totalElements - i * VL_FP32) : VL_FP32;
            
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(normPreReg, normOutPtr + i * VL_FP32);
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(normPostReg, normOutPtr + n + i * VL_FP32);
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(normResReg, normOutPtr + 2 * n + i * VL_FP32);
            
            MicroAPI::Muls<T>(zPreReg, normPreReg, alphaPtr[0], mask);
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(biasPreReg, biasPtr + i * VL_FP32);
            MicroAPI::Add<T>(zPreReg, zPreReg, biasPreReg, mask);
            
            MicroAPI::Muls<T>(zPostReg, normPostReg, alphaPtr[1], mask);
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(biasPostReg, biasPtr + n + i * VL_FP32);
            MicroAPI::Add<T>(zPostReg, zPostReg, biasPostReg, mask);
            
            MicroAPI::Muls<T>(zResReg, normResReg, alphaPtr[2], mask);
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(biasResReg, biasPtr + 2 * n + i * VL_FP32);
            MicroAPI::Add<T>(zResReg, zResReg, biasResReg, mask);
            
            MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(zPrePtr + i * VL_FP32, zPreReg, mask);
            MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(zPostPtr + i * VL_FP32, zPostReg, mask);
            MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(zResPtr + i * VL_FP32, zResReg, mask);
        }
    }
}

template<typename T>
__aicore__ inline void ComputeSinkhornGradVf(
    LocalTensor<T> skGradLocal, LocalTensor<T> gradHResLocal,
    LocalTensor<T> sumOutLocal, LocalTensor<T> normOutLocal,
    int64_t skIterCount, int64_t n, uint32_t totalElements)
{
    __local_mem__ T* skGradPtr = (__local_mem__ T*) skGradLocal.GetPhyAddr();
    __local_mem__ T* gradHResPtr = (__local_mem__ T*) gradHResLocal.GetPhyAddr();
    __local_mem__ T* sumOutPtr = (__local_mem__ T*) sumOutLocal.GetPhyAddr();
    __local_mem__ T* normOutPtr = (__local_mem__ T*) normOutLocal.GetPhyAddr();
    
    __VEC_SCOPE__ {
        RegTensor<T> skGradReg, gradHResReg;
        RegTensor<T> colSumReg, colNormReg;
        RegTensor<T> rowSumReg, rowNormReg;
        RegTensor<T> tempReg, temp2Reg;
        
        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();
        
        uint32_t loopCount = (totalElements + VL_FP32 - 1) / VL_FP32;
        
        MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(skGradReg, gradHResPtr);
        
        for (uint16_t i = 0; i < static_cast<uint16_t>(skIterCount); i++) {
            int64_t iter = skIterCount - 1 - static_cast<int64_t>(i);
            for (uint16_t j = 0; j < loopCount; j++) {
                MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(colSumReg, sumOutPtr + (2 * iter + 1) * n);
                MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(colNormReg, normOutPtr + (2 * iter + 1) * n * n);
                
                MicroAPI::Div<T>(tempReg, skGradReg, colSumReg, mask);
                MicroAPI::Mul<T>(temp2Reg, skGradReg, colNormReg, mask);
                MicroAPI::Div<T>(temp2Reg, temp2Reg, colSumReg, mask);
                MicroAPI::Sub<T>(skGradReg, tempReg, temp2Reg, mask);
                
                MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(rowSumReg, sumOutPtr + 2 * iter * n);
                MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(rowNormReg, normOutPtr + 2 * iter * n * n);
                
                MicroAPI::Div<T>(tempReg, skGradReg, rowSumReg, mask);
                MicroAPI::Mul<T>(temp2Reg, skGradReg, rowNormReg, mask);
                MicroAPI::Div<T>(temp2Reg, temp2Reg, rowSumReg, mask);
                MicroAPI::Sub<T>(skGradReg, tempReg, temp2Reg, mask);
            }
        }
        
        MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(skGradPtr, skGradReg, mask);
    }
}

template<typename T>
__aicore__ inline void ComputeLinearGradVf(
    LocalTensor<T> gradNormOutLocal, LocalTensor<T> gradZPreLocal, 
    LocalTensor<T> gradZPostLocal, LocalTensor<T> gradZResLocal,
    LocalTensor<T> alphaLocal, int64_t n, uint32_t totalElements)
{
    __local_mem__ T* gradNormOutPtr = (__local_mem__ T*) gradNormOutLocal.GetPhyAddr();
    __local_mem__ T* gradZPrePtr = (__local_mem__ T*) gradZPreLocal.GetPhyAddr();
    __local_mem__ T* gradZPostPtr = (__local_mem__ T*) gradZPostLocal.GetPhyAddr();
    __local_mem__ T* gradZResPtr = (__local_mem__ T*) gradZResLocal.GetPhyAddr();
    __local_mem__ T* alphaPtr = (__local_mem__ T*) alphaLocal.GetPhyAddr();
    
    __VEC_SCOPE__ {
        RegTensor<T> gradZPreReg, gradZPostReg, gradZResReg;
        RegTensor<T> gradNormPreReg, gradNormPostReg, gradNormResReg;
        
        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();
        
        uint32_t loopCount = (totalElements + VL_FP32 - 1) / VL_FP32;
        
        for (uint16_t i = 0; i < loopCount; i++) {
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(gradZPreReg, gradZPrePtr + i * VL_FP32);
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(gradZPostReg, gradZPostPtr + i * VL_FP32);
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(gradZResReg, gradZResPtr + i * VL_FP32);
            
            MicroAPI::Muls<T>(gradNormPreReg, gradZPreReg, alphaPtr[0], mask);
            MicroAPI::Muls<T>(gradNormPostReg, gradZPostReg, alphaPtr[1], mask);
            MicroAPI::Muls<T>(gradNormResReg, gradZResReg, alphaPtr[2], mask);
            
            MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(gradNormOutPtr + i * VL_FP32, gradNormPreReg, mask);
            MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(gradNormOutPtr + n + i * VL_FP32, gradNormPostReg, mask);
            MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(gradNormOutPtr + 2 * n + i * VL_FP32, gradNormResReg, mask);
        }
    }
}

template<typename T>
__aicore__ inline void ComputeRMSNormGradVf(
    LocalTensor<T> gradHcLocal, LocalTensor<T> gradNormOutLocal,
    LocalTensor<T> invRmsLocal, uint32_t totalElements)
{
    __local_mem__ T* gradHcPtr = (__local_mem__ T*) gradHcLocal.GetPhyAddr();
    __local_mem__ T* gradNormOutPtr = (__local_mem__ T*) gradNormOutLocal.GetPhyAddr();
    __local_mem__ T* invRmsPtr = (__local_mem__ T*) invRmsLocal.GetPhyAddr();
    
    __VEC_SCOPE__ {
        RegTensor<T> gradNormOutReg, invRmsReg, gradHcReg;
        
        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();
        
        uint32_t loopCount = (totalElements + VL_FP32 - 1) / VL_FP32;
        
        for (uint16_t i = 0; i < loopCount; i++) {
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(gradNormOutReg, gradNormOutPtr + i * VL_FP32);
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(invRmsReg, invRmsPtr);
            
            MicroAPI::Mul<T>(gradHcReg, gradNormOutReg, invRmsReg, mask);
            
            MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(gradHcPtr + i * VL_FP32, gradHcReg, mask);
        }
    }
}

template<typename T>
__aicore__ inline void AccumulateAlphaBiasVf(
    LocalTensor<T> gradAlphaLocal, LocalTensor<T> gradBiasLocal,
    LocalTensor<T> gradZPreLocal, LocalTensor<T> gradZPostLocal, LocalTensor<T> gradZResLocal,
    int64_t n, uint32_t tileSize)
{
    __local_mem__ T* gradAlphaPtr = (__local_mem__ T*) gradAlphaLocal.GetPhyAddr();
    __local_mem__ T* gradBiasPtr = (__local_mem__ T*) gradBiasLocal.GetPhyAddr();
    __local_mem__ T* gradZPrePtr = (__local_mem__ T*) gradZPreLocal.GetPhyAddr();
    __local_mem__ T* gradZPostPtr = (__local_mem__ T*) gradZPostLocal.GetPhyAddr();
    __local_mem__ T* gradZResPtr = (__local_mem__ T*) gradZResLocal.GetPhyAddr();
    
    __VEC_SCOPE__ {
        RegTensor<T> gradZPreReg, gradZPostReg, gradZResReg;
        RegTensor<T> gradAlphaReg, gradBiasPreReg, gradBiasPostReg, gradBiasResReg;
        RegTensor<T> tmpReg;
        
        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();
        
        uint32_t preElements = tileSize * n;
        uint32_t resElements = tileSize * n * n;
        
        MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(gradZPreReg, gradZPrePtr);
        MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(gradZPostReg, gradZPostPtr);
        
        MicroAPI::ReduceSum<T>(gradAlphaReg, gradZPreReg, mask);
        MicroAPI::ReduceSum<T>(tmpReg, gradZPostReg, mask);
        
        MicroAPI::Add<T>(gradAlphaReg, gradAlphaReg, tmpReg, mask);
        
        MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(gradZResReg, gradZResPtr);
        MicroAPI::ReduceSum<T>(tmpReg, gradZResReg, mask);
        MicroAPI::Add<T>(gradAlphaReg, gradAlphaReg, tmpReg, mask);
        
        MicroAPI::ReduceSum<T>(gradBiasPreReg, gradZPreReg, mask);
        MicroAPI::ReduceSum<T>(gradBiasPostReg, gradZPostReg, mask);
        MicroAPI::ReduceSum<T>(gradBiasResReg, gradZResReg, mask);
        
        MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(gradAlphaPtr, gradAlphaReg, mask);
        MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(gradBiasPtr, gradBiasPreReg, mask);
        MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(gradBiasPtr + n, gradBiasPostReg, mask);
        MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(gradBiasPtr + 2 * n, gradBiasResReg, mask);
    }
}

template<typename T>
__aicore__ inline void WorkspaceAccumulateVf(
    LocalTensor<T> wsGradAlpha, LocalTensor<T> wsGradBias,
    LocalTensor<T> gradAlphaLocal, LocalTensor<T> gradBiasLocal,
    int64_t n, uint32_t totalAlpha, uint32_t totalBias)
{
    __local_mem__ T* wsGradAlphaPtr = (__local_mem__ T*) wsGradAlpha.GetPhyAddr();
    __local_mem__ T* wsGradBiasPtr = (__local_mem__ T*) wsGradBias.GetPhyAddr();
    __local_mem__ T* gradAlphaPtr = (__local_mem__ T*) gradAlphaLocal.GetPhyAddr();
    __local_mem__ T* gradBiasPtr = (__local_mem__ T*) gradBiasLocal.GetPhyAddr();
    
    __VEC_SCOPE__ {
        RegTensor<T> wsGradAlphaReg, gradAlphaReg;
        RegTensor<T> wsGradBiasReg, gradBiasReg;
        
        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();
        
        MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(wsGradAlphaReg, wsGradAlphaPtr);
        MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(gradAlphaReg, gradAlphaPtr);
        MicroAPI::Add<T>(wsGradAlphaReg, wsGradAlphaReg, gradAlphaReg, mask);
        MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(wsGradAlphaPtr, wsGradAlphaReg, mask);
        
        MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(wsGradBiasReg, wsGradBiasPtr);
        MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(gradBiasReg, gradBiasPtr);
        MicroAPI::Add<T>(wsGradBiasReg, wsGradBiasReg, gradBiasReg, mask);
        MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(wsGradBiasPtr, wsGradBiasReg, mask);
    }
}

template<typename T>
__aicore__ inline void ReduceWorkspaceVf(
    LocalTensor<T> gradAlphaGlobal, LocalTensor<T> gradBiasGlobal,
    LocalTensor<T> workspaceGlobal, int64_t wsGradAlphaOffset, int64_t wsGradBiasOffset,
    int64_t wsPerCoreSize, int64_t aivNum, int64_t n)
{
    __local_mem__ T* gradAlphaPtr = (__local_mem__ T*) gradAlphaGlobal.GetPhyAddr();
    __local_mem__ T* gradBiasPtr = (__local_mem__ T*) gradBiasGlobal.GetPhyAddr();
    __local_mem__ T* workspacePtr = (__local_mem__ T*) workspaceGlobal.GetPhyAddr();
    
    __VEC_SCOPE__ {
        RegTensor<T> gradAlphaReg, gradBiasReg;
        RegTensor<T> wsGradAlphaReg, wsGradBiasReg;
        RegTensor<T> tmpAlphaReg, tmpBiasReg;
        
        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();
        
        MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(gradAlphaReg, workspacePtr + wsGradAlphaOffset);
        MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(gradBiasReg, workspacePtr + wsGradBiasOffset);
        
        for (uint16_t c = 1; c < static_cast<uint16_t>(aivNum); c++) {
            int64_t wsOffset = c * wsPerCoreSize;
            
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(wsGradAlphaReg, workspacePtr + wsOffset + wsGradAlphaOffset);
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(wsGradBiasReg, workspacePtr + wsOffset + wsGradBiasOffset);
            
MicroAPI::Add<T>(gradAlphaReg, gradAlphaReg, wsGradAlphaReg, mask);
            MicroAPI::Add<T>(gradBiasReg, gradBiasReg, wsGradBiasReg, mask);
        }
        
        MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(gradAlphaPtr, gradAlphaReg, mask);
        MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(gradBiasPtr, gradBiasReg, mask);
    }
}
 
template<typename T>
__aicore__ inline void SigmoidGradArithmeticVf(
    LocalTensor<T> gradZPreLocal, LocalTensor<T> gradZPostLocal,
    LocalTensor<T> sigmaPreLocal, LocalTensor<T> sigmaPostLocal,
    LocalTensor<T> gradHPreLocal, LocalTensor<T> gradHPostLocal,
    uint32_t totalElements, T hcEps)
{
    __local_mem__ T* gradZPrePtr = (__local_mem__ T*) gradZPreLocal.GetPhyAddr();
    __local_mem__ T* gradZPostPtr = (__local_mem__ T*) gradZPostLocal.GetPhyAddr();
    __local_mem__ T* sigmaPrePtr = (__local_mem__ T*) sigmaPreLocal.GetPhyAddr();
    __local_mem__ T* sigmaPostPtr = (__local_mem__ T*) sigmaPostLocal.GetPhyAddr();
    __local_mem__ T* gradHPrePtr = (__local_mem__ T*) gradHPreLocal.GetPhyAddr();
    __local_mem__ T* gradHPostPtr = (__local_mem__ T*) gradHPostLocal.GetPhyAddr();
    
    __VEC_SCOPE__ {
        RegTensor<T> sigmaPreReg, sigmaPostReg;
        RegTensor<T> gradHPreReg, gradHPostReg;
        RegTensor<T> oneMinusSigmaPreReg, oneMinusSigmaPostReg;
        RegTensor<T> gradZPreReg, gradZPostTmpReg, gradZPostReg;
        RegTensor<T> onesReg;
        
        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();
        
        uint32_t loopCount = (totalElements + VL_FP32 - 1) / VL_FP32;
        
        for (uint16_t i = 0; i < loopCount; i++) {
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(sigmaPreReg, sigmaPrePtr + i * VL_FP32);
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(sigmaPostReg, sigmaPostPtr + i * VL_FP32);
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(gradHPreReg, gradHPrePtr + i * VL_FP32);
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(gradHPostReg, gradHPostPtr + i * VL_FP32);
            
            // HC_EPS修正: sigma_pre = sigmoid(z_pre) - hcEps
            MicroAPI::Adds<T>(sigmaPreReg, sigmaPreReg, static_cast<T>(-hcEps), mask);
            
            MicroAPI::Muls<T>(oneMinusSigmaPreReg, sigmaPreReg, static_cast<T>(-1.0), mask);
            MicroAPI::Adds<T>(oneMinusSigmaPreReg, oneMinusSigmaPreReg, static_cast<T>(1.0), mask);
            MicroAPI::Mul<T>(gradZPreReg, gradHPreReg, oneMinusSigmaPreReg, mask);
            MicroAPI::Mul<T>(gradZPreReg, gradZPreReg, sigmaPreReg, mask);
            
            MicroAPI::Muls<T>(oneMinusSigmaPostReg, sigmaPostReg, static_cast<T>(-1.0), mask);
            MicroAPI::Adds<T>(oneMinusSigmaPostReg, oneMinusSigmaPostReg, static_cast<T>(1.0), mask);
            MicroAPI::Mul<T>(gradZPostTmpReg, gradHPostReg, oneMinusSigmaPostReg, mask);
            MicroAPI::Mul<T>(gradZPostTmpReg, gradZPostTmpReg, sigmaPostReg, mask);
            MicroAPI::Muls<T>(gradZPostReg, gradZPostTmpReg, static_cast<T>(2.0), mask);
            
            MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(gradZPrePtr + i * VL_FP32, gradZPreReg, mask);
            MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(gradZPostPtr + i * VL_FP32, gradZPostReg, mask);
        }
    }
}

template<typename T>
__aicore__ inline void RMSNormInvGradVf(
    LocalTensor<T> gradXFromRmsChunk, LocalTensor<T> xFp32Chunk,
    LocalTensor<T> invRmsCubed, LocalTensor<T> gradInvRmsLocal,
    float ncScalar, uint32_t cur_c, int64_t n)
{
    __local_mem__ T* gradXFromRmsPtr = (__local_mem__ T*) gradXFromRmsChunk.GetPhyAddr();
    __local_mem__ T* xFp32Ptr = (__local_mem__ T*) xFp32Chunk.GetPhyAddr();
    __local_mem__ T* invRmsCubedPtr = (__local_mem__ T*) invRmsCubed.GetPhyAddr();
    __local_mem__ T* gradInvRmsPtr = (__local_mem__ T*) gradInvRmsLocal.GetPhyAddr();
    
    __VEC_SCOPE__ {
        RegTensor<T> xFp32Reg, gradXFromRmsReg;
        RegTensor<T> coeffReg;
        
        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();
        
        uint32_t loopCount = (cur_c + VL_FP32 - 1) / VL_FP32;
        
        for (uint16_t i = 0; i < loopCount; i++) {
            uint32_t calcSize = (i == loopCount - 1) ? 
                (cur_c - i * VL_FP32) : VL_FP32;
            
            T coeff = static_cast<T>(invRmsCubedPtr[0] * gradInvRmsPtr[0] * static_cast<float>(-1.0) / ncScalar);
            
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(xFp32Reg, xFp32Ptr + i * VL_FP32);
            MicroAPI::Muls<T>(gradXFromRmsReg, xFp32Reg, coeff, mask);
            MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(gradXFromRmsPtr + i * VL_FP32, gradXFromRmsReg, mask);
        }
    }
}

// ========== VF ReduceSum (任意长度，循环累加) ==========
template<typename T>
__aicore__ inline void ReduceSumVf(
    LocalTensor<T> dstLocal, LocalTensor<T> srcLocal, uint32_t totalElements)
{
    __local_mem__ T* dstPtr = (__local_mem__ T*) dstLocal.GetPhyAddr();
    __local_mem__ T* srcPtr = (__local_mem__ T*) srcLocal.GetPhyAddr();
    
    __VEC_SCOPE__ {
        RegTensor<T> srcReg, sumReg, tmpReg;
        
        MaskReg maskAll = MicroAPI::CreateMask<T, MaskPattern::ALL>();
        
        uint32_t loopCount = (totalElements + VL_FP32 - 1) / VL_FP32;
        
        MicroAPI::Duplicate<T>(sumReg, static_cast<T>(0.0), maskAll);
        
        for (uint16_t i = 0; i < loopCount; i++) {
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(srcReg, srcPtr + i * VL_FP32);
            MicroAPI::ReduceSum<T>(tmpReg, srcReg, maskAll);
            MicroAPI::Add<T>(sumReg, sumReg, tmpReg, maskAll);
        }
        
        MicroAPI::DataCopy<T, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr, sumReg, maskAll);
    }
}

// ========== VF Mul+ReduceSum组合 (任意长度) ==========
template<typename T>
__aicore__ inline void MulReduceSumVf(
    LocalTensor<T> dstLocal, LocalTensor<T> srcALocal, LocalTensor<T> srcBLocal,
    uint32_t totalElements)
{
    __local_mem__ T* dstPtr = (__local_mem__ T*) dstLocal.GetPhyAddr();
    __local_mem__ T* srcAPtr = (__local_mem__ T*) srcALocal.GetPhyAddr();
    __local_mem__ T* srcBPtr = (__local_mem__ T*) srcBLocal.GetPhyAddr();
    
    __VEC_SCOPE__ {
        RegTensor<T> srcAReg, srcBReg, mulReg, sumReg, tmpReg;
        
        MaskReg maskAll = MicroAPI::CreateMask<T, MaskPattern::ALL>();
        
        uint32_t loopCount = (totalElements + VL_FP32 - 1) / VL_FP32;
        
        MicroAPI::Duplicate<T>(sumReg, static_cast<T>(0.0), maskAll);
        
        for (uint16_t i = 0; i < loopCount; i++) {
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(srcAReg, srcAPtr + i * VL_FP32);
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(srcBReg, srcBPtr + i * VL_FP32);
            MicroAPI::Mul<T>(mulReg, srcAReg, srcBReg, maskAll);
            MicroAPI::ReduceSum<T>(tmpReg, mulReg, maskAll);
            MicroAPI::Add<T>(sumReg, sumReg, tmpReg, maskAll);
        }
        
        MicroAPI::DataCopy<T, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr, sumReg, maskAll);
    }
}

// ========== VF Cast (BF16/FP16 → FP32) ==========
template<typename T, typename U>
__aicore__ inline void CastVf(
    LocalTensor<T> dstLocal, LocalTensor<U> srcLocal, uint32_t totalElements)
{
    __local_mem__ T* dstPtr = (__local_mem__ T*) dstLocal.GetPhyAddr();
    __local_mem__ U* srcPtr = (__local_mem__ U*) srcLocal.GetPhyAddr();
    
    __VEC_SCOPE__ {
        RegTensor<U> srcReg;
        RegTensor<T> dstReg;
        
        static constexpr MicroAPI::CastTrait castTrait =
            {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
            MicroAPI::MaskMergeMode::MERGING, RoundMode::CAST_NONE};
        
        MaskReg mask = MicroAPI::CreateMask<U, MaskPattern::ALL>();
        
        uint32_t loopCount = (totalElements + VL_FP32 - 1) / VL_FP32;
        
        for (uint16_t i = 0; i < loopCount; i++) {
            uint32_t calcSize = (i == loopCount - 1) ? 
                (totalElements - i * VL_FP32) : VL_FP32;
            
            MicroAPI::DataCopy<U, LoadDist::DIST_BLK>(srcReg, srcPtr + i * VL_FP32);
            MicroAPI::Cast<T, U, castTrait>(dstReg, srcReg, mask);
            MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(dstPtr + i * VL_FP32, dstReg, mask);
        }
    }
}

// ========== VF Muls (Multiply by scalar) ==========
template<typename T>
__aicore__ inline void MulsVf(
    LocalTensor<T> dstLocal, LocalTensor<T> srcLocal, T scalar, uint32_t totalElements)
{
    __local_mem__ T* dstPtr = (__local_mem__ T*) dstLocal.GetPhyAddr();
    __local_mem__ T* srcPtr = (__local_mem__ T*) srcLocal.GetPhyAddr();
    
    __VEC_SCOPE__ {
        RegTensor<T> srcReg, dstReg;
        
        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();
        
        uint32_t loopCount = (totalElements + VL_FP32 - 1) / VL_FP32;
        
        for (uint16_t i = 0; i < loopCount; i++) {
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(srcReg, srcPtr + i * VL_FP32);
            MicroAPI::Muls<T>(dstReg, srcReg, scalar, mask);
            MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(dstPtr + i * VL_FP32, dstReg, mask);
        }
    }
}

// ========== VF Adds (Add scalar) ==========
template<typename T>
__aicore__ inline void AddsVf(
    LocalTensor<T> dstLocal, LocalTensor<T> srcLocal, T scalar, uint32_t totalElements)
{
    __local_mem__ T* dstPtr = (__local_mem__ T*) dstLocal.GetPhyAddr();
    __local_mem__ T* srcPtr = (__local_mem__ T*) srcLocal.GetPhyAddr();
    
    __VEC_SCOPE__ {
        RegTensor<T> srcReg, dstReg;
        
        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();
        
        uint32_t loopCount = (totalElements + VL_FP32 - 1) / VL_FP32;
        
        for (uint16_t i = 0; i < loopCount; i++) {
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(srcReg, srcPtr + i * VL_FP32);
            MicroAPI::Adds<T>(dstReg, srcReg, scalar, mask);
            MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(dstPtr + i * VL_FP32, dstReg, mask);
        }
    }
}

// ========== VF Mul (Element-wise multiply two vectors) ==========
template<typename T>
__aicore__ inline void MulVf(
    LocalTensor<T> dstLocal, LocalTensor<T> src1Local, LocalTensor<T> src2Local,
    uint32_t totalElements)
{
    __local_mem__ T* dstPtr = (__local_mem__ T*) dstLocal.GetPhyAddr();
    __local_mem__ T* src1Ptr = (__local_mem__ T*) src1Local.GetPhyAddr();
    __local_mem__ T* src2Ptr = (__local_mem__ T*) src2Local.GetPhyAddr();
    
    __VEC_SCOPE__ {
        RegTensor<T> src1Reg, src2Reg, dstReg;
        
        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();
        
        uint32_t loopCount = (totalElements + VL_FP32 - 1) / VL_FP32;
        
        for (uint16_t i = 0; i < loopCount; i++) {
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(src1Reg, src1Ptr + i * VL_FP32);
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(src2Reg, src2Ptr + i * VL_FP32);
            MicroAPI::Mul<T>(dstReg, src1Reg, src2Reg, mask);
            MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(dstPtr + i * VL_FP32, dstReg, mask);
        }
    }
}

// ========== VF InputAggregation (Step1大VF: gradHin Cast + Muls + MulReduceSum + gradHPre累加) ==========
// 【调整说明】: x的Cast外置到VF调用前，VF输入为已Cast的xFp32
// 【核心原则】: 按CopyIn→计算→CopyOut组织代码，动态Mask处理尾块
template<typename T, typename U>
template<typename T, typename U>
__aicore__ inline void InputAggregationVf_Optimized(
    LocalTensor<T> gradXLocal,
    LocalTensor<T> xFp32Local,
    LocalTensor<T> gradHPreLocal,
    uint32_t j,
    LocalTensor<U> gradHinInputLocal,
    LocalTensor<U> xInputLocal,
    T hPreScalar,
    uint32_t cur_c)
{
    __local_mem__ T* gradXPtr = (__local_mem__ T*) gradXLocal.GetPhyAddr();
    __local_mem__ T* xFp32Ptr = (__local_mem__ T*) xFp32Local.GetPhyAddr();
    __local_mem__ T* gradHPrePtr = (__local_mem__ T*) gradHPreLocal.GetPhyAddr();
    __local_mem__ U* gradHinInputPtr = (__local_mem__ U*) gradHinInputLocal.GetPhyAddr();
    __local_mem__ U* xInputPtr = (__local_mem__ U*) xInputLocal.GetPhyAddr();
    
    static constexpr MicroAPI::CastTrait castTraitZeroing =
        {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
         MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_NONE};
    
    __VEC_SCOPE__ {
        RegTensor<U> gradHinInputReg, xInputReg;
        RegTensor<T> gradHinFp32Reg, xFp32Reg, gradXReg;
        RegTensor<T> mulReg, sumReg, tmpReg, existingReg;
        RegTensor<T> gradHPreRegSumReg;
        
        MaskReg maskAll = MicroAPI::CreateMask<T, MaskPattern::ALL>();
        MaskReg maskU = MicroAPI::CreateMask<U, MaskPattern::ALL>();
        MicroAPI::MaskReg OneMaskReg = MicroAPI::CreateMask<T, MicroAPI::VL1>();
        MicroAPI::UnalignRegForStore gradHPreSumUreg;
        
        uint32_t vfLen = VL_FP32;
        uint32_t loopCount = (cur_c + vfLen - 1) / vfLen;
        uint32_t maskLen = static_cast<uint32_t>(cur_c);
        
        MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(existingReg, gradHPrePtr + j);
        MicroAPI::Duplicate<T>(sumReg, static_cast<T>(0.0), maskAll);
        
        for (uint16_t i = 0; i < loopCount; i++) {
            MicroAPI::MaskReg valueMaskReg = MicroAPI::UpdateMask<T>(maskLen);
            uint32_t calcSize = (i == loopCount - 1) ? 
                (cur_c - i * vfLen) : vfLen;
            MicroAPI::MaskReg maskCalc = MicroAPI::UpdateMask<T>(calcSize);
            
            MicroAPI::AddrReg gradHinAddrOfst = MicroAPI::CreateAddrReg<U>(i, vfLen);
            MicroAPI::DataCopy<U, MicroAPI::LoadDist::DIST_UNPACK_B16>(gradHinInputReg, gradHinInputPtr, gradHinAddrOfst);
            MicroAPI::Cast<T, U, castTraitZeroing>(gradHinFp32Reg, gradHinInputReg, valueMaskReg);
            
            MicroAPI::Muls<T>(gradXReg, gradHinFp32Reg, hPreScalar, maskCalc);
            
            MicroAPI::AddrReg xAddrOfst = MicroAPI::CreateAddrReg<U>(i, vfLen);
            MicroAPI::DataCopy<U, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInputReg, xInputPtr, xAddrOfst);
            MicroAPI::Cast<T, U, castTraitZeroing>(xFp32Reg, xInputReg, valueMaskReg);
            
            MicroAPI::Mul<T>(mulReg, gradHinFp32Reg, xFp32Reg, maskCalc);
            MicroAPI::Reduce<MicroAPI::ReduceType::SUM, T, T>(tmpReg, mulReg, maskCalc);
            MicroAPI::Add<T>(sumReg, sumReg, tmpReg, maskAll);
            
            MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(gradXPtr + i * vfLen, gradXReg, maskCalc);
            MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(xFp32Ptr + i * vfLen, xFp32Reg, maskCalc);
        }
        
        MicroAPI::Adds(gradHPreRegSumReg, sumReg, existingReg, OneMaskReg);
        auto gradHPreStart = gradHPrePtr + j;
        MicroAPI::StoreUnAlign(gradHPreStart, gradHPreRegSumReg, gradHPreSumUreg, 1);
        MicroAPI::StoreUnAlignPost(gradHPreStart, gradHPreSumUreg, 0);
    }
}

// ========== VF ComputeForwardZFull (Step2大VF: Mul(normOut)+Muls×3+Add×3) ==========
template<typename T>
__aicore__ inline void ComputeForwardZFullVf(
    LocalTensor<T> zPreLocal, LocalTensor<T> zPostLocal, LocalTensor<T> zResLocal,
    LocalTensor<T> normOutForwardLocal,  // 输出+中间: normOut=hcBeforeNorm*invRms
    LocalTensor<T> hcBeforeNormLocal, LocalTensor<T> invRmsLocal,
    LocalTensor<T> alphaLocal, LocalTensor<T> biasLocal,
    int64_t n, uint32_t totalElements)
{
    __local_mem__ T* zPrePtr = (__local_mem__ T*) zPreLocal.GetPhyAddr();
    __local_mem__ T* zPostPtr = (__local_mem__ T*) zPostLocal.GetPhyAddr();
    __local_mem__ T* zResPtr = (__local_mem__ T*) zResLocal.GetPhyAddr();
    __local_mem__ T* normOutPtr = (__local_mem__ T*) normOutForwardLocal.GetPhyAddr();
    __local_mem__ T* hcBeforeNormPtr = (__local_mem__ T*) hcBeforeNormLocal.GetPhyAddr();
    __local_mem__ T* invRmsPtr = (__local_mem__ T*) invRmsLocal.GetPhyAddr();
    __local_mem__ T* alphaPtr = (__local_mem__ T*) alphaLocal.GetPhyAddr();
    __local_mem__ T* biasPtr = (__local_mem__ T*) biasLocal.GetPhyAddr();
    
    __VEC_SCOPE__ {
        RegTensor<T> hcBeforeNormReg, invRmsReg, normOutReg;
        RegTensor<T> normPreReg, normPostReg, normResReg;
        RegTensor<T> alphaPreReg, alphaPostReg, alphaResReg;
        RegTensor<T> biasPreReg, biasPostReg, biasResReg;
        RegTensor<T> zPreReg, zPostReg, zResReg;
        
        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();
        
        uint32_t loopCount = (totalElements + VL_FP32 - 1) / VL_FP32;
        
        for (uint16_t i = 0; i < loopCount; i++) {
            // 1. Mul: normOut = hcBeforeNorm * invRms
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(hcBeforeNormReg, hcBeforeNormPtr + i * VL_FP32);
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(invRmsReg, invRmsPtr);
            MicroAPI::Mul<T>(normOutReg, hcBeforeNormReg, invRmsReg, mask);
            
            // Store normOut到UB（供Step3/45使用）
            MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(normOutPtr + i * VL_FP32, normOutReg, mask);
            
            // 2. 拆分三路normOut
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(normPreReg, normOutPtr + i * VL_FP32);
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(normPostReg, normOutPtr + n + i * VL_FP32);
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(normResReg, normOutPtr + 2 * n + i * VL_FP32);
            
            // 3. Muls×3: z = normOut_part * alpha + bias
            MicroAPI::Muls<T>(zPreReg, normPreReg, alphaPtr[0], mask);
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(biasPreReg, biasPtr + i * VL_FP32);
            MicroAPI::Add<T>(zPreReg, zPreReg, biasPreReg, mask);
            
            MicroAPI::Muls<T>(zPostReg, normPostReg, alphaPtr[1], mask);
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(biasPostReg, biasPtr + n + i * VL_FP32);
            MicroAPI::Add<T>(zPostReg, zPostReg, biasPostReg, mask);
            
            MicroAPI::Muls<T>(zResReg, normResReg, alphaPtr[2], mask);
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(biasResReg, biasPtr + 2 * n + i * VL_FP32);
            MicroAPI::Add<T>(zResReg, zResReg, biasResReg, mask);
            
            // 4. Store zPre/zPost/zRes
            MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(zPrePtr + i * VL_FP32, zPreReg, mask);
            MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(zPostPtr + i * VL_FP32, zPostReg, mask);
            MicroAPI::DataCopy<T, StoreDist::DIST_NORM>(zResPtr + i * VL_FP32, zResReg, mask);
        }
    }
}

// ========== VF AlphaBiasGradPrePost (Step45大VF: 4路MulReduceSum+ReduceSum合一) ==========
template<typename T>
__aicore__ inline void AlphaBiasGradPrePostVf(
    LocalTensor<T> gradAlphaLocal, LocalTensor<T> gradBiasLocal,
    LocalTensor<T> normOutForwardLocal,
    LocalTensor<T> gradZPreLocal, LocalTensor<T> gradZPostLocal,
    int64_t n, uint32_t tileSize)
{
    __local_mem__ T* gradAlphaPtr = (__local_mem__ T*) gradAlphaLocal.GetPhyAddr();
    __local_mem__ T* gradBiasPtr = (__local_mem__ T*) gradBiasLocal.GetPhyAddr();
    __local_mem__ T* normOutPtr = (__local_mem__ T*) normOutForwardLocal.GetPhyAddr();
    __local_mem__ T* gradZPrePtr = (__local_mem__ T*) gradZPreLocal.GetPhyAddr();
    __local_mem__ T* gradZPostPtr = (__local_mem__ T*) gradZPostLocal.GetPhyAddr();
    
    __VEC_SCOPE__ {
        RegTensor<T> normPreReg, normPostReg;
        RegTensor<T> gradZPreReg, gradZPostReg;
        RegTensor<T> mulReg, sumAlphaPreReg, sumAlphaPostReg;
        RegTensor<T> sumBiasPreReg, sumBiasPostReg, tmpReg;
        
        MaskReg maskAll = MicroAPI::CreateMask<T, MaskPattern::ALL>();
        
        uint32_t preElements = tileSize * n;
        uint32_t loopCount = (preElements + VL_FP32 - 1) / VL_FP32;
        
        // 初始化归约累加器为0
        MicroAPI::Duplicate<T>(sumAlphaPreReg, static_cast<T>(0.0), maskAll);
        MicroAPI::Duplicate<T>(sumAlphaPostReg, static_cast<T>(0.0), maskAll);
        MicroAPI::Duplicate<T>(sumBiasPreReg, static_cast<T>(0.0), maskAll);
        MicroAPI::Duplicate<T>(sumBiasPostReg, static_cast<T>(0.0), maskAll);
        
        for (uint16_t i = 0; i < loopCount; i++) {
            // Load inputs from UB
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(normPreReg, normOutPtr + i * VL_FP32);
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(normPostReg, normOutPtr + n + i * VL_FP32);
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(gradZPreReg, gradZPrePtr + i * VL_FP32);
            MicroAPI::DataCopy<T, LoadDist::DIST_BLK>(gradZPostReg, gradZPostPtr + i * VL_FP32);
            
            // grad_alpha[0] += ReduceSum(normOut[:N] * gradZPre)
            MicroAPI::Mul<T>(mulReg, normPreReg, gradZPreReg, maskAll);
            MicroAPI::ReduceSum<T>(tmpReg, mulReg, maskAll);
            MicroAPI::Add<T>(sumAlphaPreReg, sumAlphaPreReg, tmpReg, maskAll);
            
            // grad_alpha[1] += ReduceSum(normOut[N:2N] * gradZPost)
            MicroAPI::Mul<T>(mulReg, normPostReg, gradZPostReg, maskAll);
            MicroAPI::ReduceSum<T>(tmpReg, mulReg, maskAll);
            MicroAPI::Add<T>(sumAlphaPostReg, sumAlphaPostReg, tmpReg, maskAll);
            
            // grad_bias[:N] += ReduceSum(gradZPre)
            MicroAPI::ReduceSum<T>(tmpReg, gradZPreReg, maskAll);
            MicroAPI::Add<T>(sumBiasPreReg, sumBiasPreReg, tmpReg, maskAll);
            
            // grad_bias[N:2N] += ReduceSum(gradZPost)
            MicroAPI::ReduceSum<T>(tmpReg, gradZPostReg, maskAll);
            MicroAPI::Add<T>(sumBiasPostReg, sumBiasPostReg, tmpReg, maskAll);
        }
        
        // Store results to UB (scalar values via StoreUnAlign)
        MicroAPI::UnalignRegForStore gradAlphaSumUreg, gradBiasSumUreg;
        MicroAPI::MaskReg OneMaskReg = MicroAPI::CreateMask<T, MicroAPI::VL1>();
        
        MicroAPI::StoreUnAlign(gradAlphaPtr, sumAlphaPreReg, gradAlphaSumUreg, 1);
        MicroAPI::StoreUnAlignPost(gradAlphaPtr, gradAlphaSumUreg, 0);
        
        MicroAPI::StoreUnAlign(gradAlphaPtr + 1, sumAlphaPostReg, gradAlphaSumUreg, 1);
        MicroAPI::StoreUnAlignPost(gradAlphaPtr + 1, gradAlphaSumUreg, 0);
        
        MicroAPI::StoreUnAlign(gradBiasPtr, sumBiasPreReg, gradBiasSumUreg, 1);
        MicroAPI::StoreUnAlignPost(gradBiasPtr, gradBiasSumUreg, 0);
        
        MicroAPI::StoreUnAlign(gradBiasPtr + n, sumBiasPostReg, gradBiasSumUreg, 1);
        MicroAPI::StoreUnAlignPost(gradBiasPtr + n, gradBiasSumUreg, 0);
    }
}

#endif // MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_VF_H