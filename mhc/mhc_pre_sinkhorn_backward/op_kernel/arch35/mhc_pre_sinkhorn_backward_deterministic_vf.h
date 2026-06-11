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
            MicroAPI::Add<T>(zPreReg, zPreReg, biasPtr, mask);
            
            MicroAPI::Muls<T>(zPostReg, normPostReg, alphaPtr[1], mask);
            MicroAPI::Add<T>(zPostReg, zPostReg, biasPtr + n, mask);
            
            MicroAPI::Muls<T>(zResReg, normResReg, alphaPtr[2], mask);
            MicroAPI::Add<T>(zResReg, zResReg, biasPtr + 2 * n, mask);
            
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

#endif // MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_VF_H