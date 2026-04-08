#include "kernel_operator.h"

using namespace AscendC;
using namespace MicroAPI;

namespace {
constexpr int32_t FP32_BYTE_SIZE = 4;
constexpr int32_t UB_BLOCK_BYTE_SIZE = 32;
constexpr int32_t VEC_LENGTH = 256;
constexpr int32_t BLOCKNUM_PER_VL = VEC_LENGTH / UB_BLOCK_BYTE_SIZE;
constexpr int32_t FP32_PER_VL = VEC_LENGTH / FP32_BYTE_SIZE;
} // namespace

template<typename T>
__aicore__ inline void ComputeGradPreAndGradX3Vf(LocalTensor<T> gradX3Local, LocalTensor<T> gradPreLocal, LocalTensor<T> xLocal,
    LocalTensor<T> gradYLocal, LocalTensor<T> preLocal, int32_t n, uint16_t repeatTimes)
{
    __local_mem__ T* xPtr = (__local_mem__ T*) xLocal.GetPhyAddr();
    __local_mem__ T* yGradPtr = (__local_mem__ T*) gradYLocal.GetPhyAddr();
    __local_mem__ T* prePtr = (__local_mem__ T*) preLocal.GetPhyAddr();

    __local_mem__ T* gradPrePtr = (__local_mem__ T*) gradPreLocal.GetPhyAddr();
    __local_mem__ T* gradX3Ptr = (__local_mem__ T*) gradX3Local.GetPhyAddr();

    __VEC_SCOPE__ {
        RegTensor<T> xReg0, xReg1, xReg2, xReg3;
        RegTensor<T> gradYReg;

        RegTensor<T> preReg0, preReg1, preReg2, preReg3;
        RegTensor<T> tmpReg;

        RegTensor<T> gradxReg0, gradxReg1, gradxReg2, gradxReg3;
        RegTensor<T> gradpreReg0, gradpreReg1, gradpreReg2, gradpreReg3;

        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();

        MicroAPI::Duplicate(gradpreReg0, 0);
        MicroAPI::Duplicate(gradpreReg1, 0);
        MicroAPI::Duplicate(gradpreReg2, 0);
        MicroAPI::Duplicate(gradpreReg3, 0);

        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(preReg0, prePtr);
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(preReg1, prePtr + 1);
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(preReg2, prePtr + 2);
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(preReg3, prePtr + 3);

        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::DataCopy(xReg0, xPtr + i * FP32_PER_VL);
            MicroAPI::DataCopy(xReg1, xPtr + (i + 1 * repeatTimes) * FP32_PER_VL);
            MicroAPI::DataCopy(xReg2, xPtr + (i + 2 * repeatTimes) * FP32_PER_VL);
            MicroAPI::DataCopy(xReg3, xPtr + (i + 3 * repeatTimes) * FP32_PER_VL);
            MicroAPI::DataCopy(gradYReg, yGradPtr + i * FP32_PER_VL);

            MicroAPI::Mul(xReg0, xReg0, gradYReg, mask);
            MicroAPI::Mul(xReg1, xReg1, gradYReg, mask);
            MicroAPI::Mul(xReg2, xReg2, gradYReg, mask);
            MicroAPI::Mul(xReg3, xReg3, gradYReg, mask);

            MicroAPI::Add(gradpreReg0, gradpreReg0, xReg0, mask);
            MicroAPI::Add(gradpreReg1, gradpreReg1, xReg1, mask);
            MicroAPI::Add(gradpreReg2, gradpreReg2, xReg2, mask);
            MicroAPI::Add(gradpreReg3, gradpreReg3, xReg3, mask);

            MicroAPI::Mul(gradxReg0, preReg0, gradYReg, mask);
            MicroAPI::Mul(gradxReg1, preReg1, gradYReg, mask);
            MicroAPI::Mul(gradxReg2, preReg2, gradYReg, mask);
            MicroAPI::Mul(gradxReg3, preReg3, gradYReg, mask);

            MicroAPI::DataCopy(gradX3Ptr + i * FP32_PER_VL, gradxReg0, mask);
            MicroAPI::DataCopy(gradX3Ptr + (i + 1 * repeatTimes) * FP32_PER_VL, gradxReg1, mask);
            MicroAPI::DataCopy(gradX3Ptr + (i + 2 * repeatTimes) * FP32_PER_VL, gradxReg2, mask);
            MicroAPI::DataCopy(gradX3Ptr + (i + 3 * repeatTimes) * FP32_PER_VL, gradxReg3, mask);
        }

        MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(gradpreReg0, gradpreReg0, mask);
        MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(gradpreReg1, gradpreReg1, mask);
        MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(gradpreReg2, gradpreReg2, mask);
        MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(gradpreReg3, gradpreReg3, mask);

        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(gradPrePtr, gradpreReg0, mask);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(gradPrePtr + 1, gradpreReg1, mask);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(gradPrePtr + 2, gradpreReg2, mask);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(gradPrePtr + 3, gradpreReg3, mask);
    }
}

template<typename T>
__aicore__ inline void ComputeGradSigmoidVf(LocalTensor<T> gradSigmoidLocal1, LocalTensor<T> gradSigmoidLocal2, LocalTensor<T> fusedHPre2AndHPost2Local, LocalTensor<T> rsqrtLocal,
    LocalTensor<T> hcScaleLocal, LocalTensor<T> hcBaseLocal, uint16_t repeatTimes, uint32_t totalElements)
{
    __local_mem__ T* gradSigmoidPtr1 = (__local_mem__ T*) gradSigmoidLocal1.GetPhyAddr();
    __local_mem__ T* gradSigmoidPtr2 = (__local_mem__ T*) gradSigmoidLocal2.GetPhyAddr();

    __local_mem__ T* rsqrtPtr = (__local_mem__ T*) rsqrtLocal.GetPhyAddr();
    __local_mem__ T* fusedHPre2AndHPost2Ptr = (__local_mem__ T*) fusedHPre2AndHPost2Local.GetPhyAddr();
    __local_mem__ T* hcScalePtr = (__local_mem__ T*) hcScaleLocal.GetPhyAddr();
    __local_mem__ T* hcBasePtr = (__local_mem__ T*) hcBaseLocal.GetPhyAddr();
    
    __VEC_SCOPE__ {
        
        RegTensor<T> rsqrtBrcbReg;
        RegTensor<T> fusedHPre2AndHPost2Reg;
        RegTensor<T> hcScaleReg;
        RegTensor<T> hcBaseReg;
        RegTensor<T> gradSigmoidReg1;
        RegTensor<T> gradSigmoidReg2;
        RegTensor<T> mulMaskReg;
        RegTensor<T> tmpReg;
        RegTensor<T> onesReg;
        RegTensor<T> h2ValNormedReg;

        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();

        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BLK>(hcScaleReg, hcScalePtr);
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BLK>(hcBaseReg, hcBasePtr);

        // 初始化 mulMaskReg 为 [1,1,1,1, 2,2,2,2, 1,1,1,1, 2,2,2,2, ...]
        MicroAPI::Arange(mulMaskReg, 0.f);
        MicroAPI::Muls(mulMaskReg, mulMaskReg, 0.25f, mask);
        MicroAPI::Truncate<T, AscendC::RoundMode::CAST_FLOOR, AscendC::MicroAPI::MaskMergeMode::ZEROING>(mulMaskReg, mulMaskReg, mask);

        MicroAPI::Muls(tmpReg, mulMaskReg, 0.5f, mask);
        MicroAPI::Truncate<T, AscendC::RoundMode::CAST_FLOOR, AscendC::MicroAPI::MaskMergeMode::ZEROING>(tmpReg, tmpReg, mask);
        MicroAPI::Muls(tmpReg, tmpReg, 2.0f, mask);

        MicroAPI::Sub(mulMaskReg, mulMaskReg, tmpReg, mask);
        MicroAPI::Adds(mulMaskReg, mulMaskReg, 1.0f, mask);

        // 初始化 onesReg 为 [1,1,1,1, ...]
        MicroAPI::Duplicate(onesReg, 1.0f, mask);

        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::MaskReg mask1 = MicroAPI::UpdateMask<T>(totalElements);
            
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_E2B_B32>(rsqrtBrcbReg, rsqrtPtr + i * BLOCKNUM_PER_VL);
            MicroAPI::DataCopy(fusedHPre2AndHPost2Reg, fusedHPre2AndHPost2Ptr + i * FP32_PER_VL);

            MicroAPI::Mul(h2ValNormedReg, fusedHPre2AndHPost2Reg, rsqrtBrcbReg, mask);
            MicroAPI::Mul(gradSigmoidReg1, h2ValNormedReg, hcScaleReg, mask);
            MicroAPI::Add(gradSigmoidReg1, gradSigmoidReg1, hcBaseReg, mask);

            MicroAPI::Neg(gradSigmoidReg1, gradSigmoidReg1, mask);
            MicroAPI::Exp(gradSigmoidReg1, gradSigmoidReg1, mask);
            MicroAPI::Adds(gradSigmoidReg1, gradSigmoidReg1, 1.0f, mask);
            MicroAPI::Div(gradSigmoidReg1, onesReg, gradSigmoidReg1, mask);
            MicroAPI::Mul(tmpReg, gradSigmoidReg1, gradSigmoidReg1, mask);
            MicroAPI::Sub(gradSigmoidReg1, gradSigmoidReg1, tmpReg, mask);
            MicroAPI::Mul(gradSigmoidReg1, gradSigmoidReg1, mulMaskReg, mask);
            
            MicroAPI::Mul(gradSigmoidReg2, gradSigmoidReg1, h2ValNormedReg, mask);

            MicroAPI::DataCopy(gradSigmoidPtr1 + i * FP32_PER_VL, gradSigmoidReg1, mask1);
            MicroAPI::DataCopy(gradSigmoidPtr2 + i * FP32_PER_VL, gradSigmoidReg2, mask1);

            // MicroAPI::DataCopy(gradSigmoidPtr2 + i * FP32_PER_VL, mulMaskReg, mask);
        }
    }
}

template<typename T>
__aicore__ inline void ComputeGradPreVf(LocalTensor<T> gradPreLocal, LocalTensor<T> xLocal,
    LocalTensor<T> gradYLocal, int32_t n, uint16_t repeatTimes)
{
    __local_mem__ T* xPtr = (__local_mem__ T*) xLocal.GetPhyAddr();
    __local_mem__ T* yGradPtr = (__local_mem__ T*) gradYLocal.GetPhyAddr();
    __local_mem__ T* gradPrePtr = (__local_mem__ T*) gradPreLocal.GetPhyAddr();

    __VEC_SCOPE__ {
        RegTensor<T> xReg0, xReg1, xReg2, xReg3;
        RegTensor<T> gradYReg;
        RegTensor<T> tmpReg;

        RegTensor<T> gradpreReg0, gradpreReg1, gradpreReg2, gradpreReg3;

        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();

        MicroAPI::Duplicate(gradpreReg0, 0);
        MicroAPI::Duplicate(gradpreReg1, 0);
        MicroAPI::Duplicate(gradpreReg2, 0);
        MicroAPI::Duplicate(gradpreReg3, 0);

        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::DataCopy(xReg0, xPtr + i * FP32_PER_VL);
            MicroAPI::DataCopy(xReg1, xPtr + (i + 1 * repeatTimes) * FP32_PER_VL);
            MicroAPI::DataCopy(xReg2, xPtr + (i + 2 * repeatTimes) * FP32_PER_VL);
            MicroAPI::DataCopy(xReg3, xPtr + (i + 3 * repeatTimes) * FP32_PER_VL);
            MicroAPI::DataCopy(gradYReg, yGradPtr + i * FP32_PER_VL);

            MicroAPI::Mul(xReg0, xReg0, gradYReg, mask);
            MicroAPI::Mul(xReg1, xReg1, gradYReg, mask);
            MicroAPI::Mul(xReg2, xReg2, gradYReg, mask);
            MicroAPI::Mul(xReg3, xReg3, gradYReg, mask);

            MicroAPI::Add(gradpreReg0, gradpreReg0, xReg0, mask);
            MicroAPI::Add(gradpreReg1, gradpreReg1, xReg1, mask);
            MicroAPI::Add(gradpreReg2, gradpreReg2, xReg2, mask);
            MicroAPI::Add(gradpreReg3, gradpreReg3, xReg3, mask);
        }

        MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(gradpreReg0, gradpreReg0, mask);
        MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(gradpreReg1, gradpreReg1, mask);
        MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(gradpreReg2, gradpreReg2, mask);
        MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(gradpreReg3, gradpreReg3, mask);

        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(gradPrePtr, gradpreReg0, mask);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(gradPrePtr + 1, gradpreReg1, mask);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(gradPrePtr + 2, gradpreReg2, mask);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(gradPrePtr + 3, gradpreReg3, mask);
    }
}


template<typename T>
__aicore__ inline void ComputeGradXVf(LocalTensor<T> gradXLocal, LocalTensor<T> xLocal, LocalTensor<T> gradYLocal,
    LocalTensor<T> preLocal, const T gradRsqrtVal, const T gradRMSNormVal, const int32_t c, uint16_t repeatTimes)
{
    __local_mem__ T* gradXPtr = (__local_mem__ T*) gradXLocal.GetPhyAddr();
    __local_mem__ T* xPtr = (__local_mem__ T*) xLocal.GetPhyAddr();
    __local_mem__ T* gradYPtr = (__local_mem__ T*) gradYLocal.GetPhyAddr();
    __local_mem__ T* prePtr = (__local_mem__ T*) preLocal.GetPhyAddr();

    __VEC_SCOPE__ {
        RegTensor<T> xReg;
        RegTensor<T> gradXReg;
        RegTensor<T> preReg;
        RegTensor<T> gradYReg;
        RegTensor<T> tmpReg;

        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();

        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(preReg, prePtr);

        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::DataCopy(xReg, xPtr + i * FP32_PER_VL);
            MicroAPI::DataCopy(gradYReg, gradYPtr + i * FP32_PER_VL);

            MicroAPI::Muls(gradXReg, xReg, gradRsqrtVal * gradRMSNormVal, mask);
            MicroAPI::Mul(tmpReg, gradYReg, preReg, mask);
            MicroAPI::Add(gradXReg, tmpReg, gradXReg, mask);

            MicroAPI::DataCopy(gradXPtr + i * FP32_PER_VL, gradXReg, mask);
        }
    }
}