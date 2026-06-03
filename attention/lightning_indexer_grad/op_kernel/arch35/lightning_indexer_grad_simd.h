#include "kernel_operator.h"

using namespace AscendC;
using namespace MicroAPI;

namespace {
constexpr int32_t INT32_BYTE_SIZE = 4;
constexpr int32_t UB_BLOCK_BYTE_SIZE = 32;
constexpr int32_t VEC_LENGTH = 256;
} // namespace

template<typename T>
__aicore__ inline void ComputegradWeightsSimd(
    LocalTensor<float> gradWeightsLocal, LocalTensor<float> selectProbLocal,
    LocalTensor<float> gradValueLocal, const uint16_t repeatTimesRow,
    const uint16_t repeatTimesCol, const bool initGradWeights)
{
    __local_mem__ float* gradWeightsPtr =
        (__local_mem__ float*) gradWeightsLocal.GetPhyAddr();
    __local_mem__ float* selectProbPtr =
        (__local_mem__ float*) selectProbLocal.GetPhyAddr();
    __local_mem__ float* gradValuePtr =
        (__local_mem__ float*) gradValueLocal.GetPhyAddr();

    __VEC_SCOPE__ {
        RegTensor<float> selectProbReg;
        RegTensor<float> gradWeightsFloatReg;
        RegTensor<float> tmpFloatReg;
        RegTensor<float> gradValueReg;
        MicroAPI::MaskReg maskFloat =
            MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg copyOutMask =
            MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL8>();

        for (uint16_t r = 0; r < repeatTimesRow; r++) {
            if (!initGradWeights) {
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
                MicroAPI::DataCopy(gradWeightsFloatReg, gradWeightsPtr + r * 8);
            } else {
                MicroAPI::Duplicate(gradWeightsFloatReg, 0.0f);
            }
            
            for (uint16_t c = 0; c < repeatTimesCol; c++) {
                MicroAPI::DataCopy<float, DataCopyMode::DATA_BLOCK_COPY>(
                    selectProbReg, selectProbPtr + r * 64 * repeatTimesCol + c * 8,
                    (uint32_t) repeatTimesCol, maskFloat);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BLK>
                    (gradValueReg, gradValuePtr + c * 8);
                MicroAPI::Relu(selectProbReg, selectProbReg, maskFloat);
                MicroAPI::Mul(selectProbReg, gradValueReg, selectProbReg, maskFloat);
                MicroAPI::ReduceDataBlock<MicroAPI::ReduceType::SUM>(tmpFloatReg, selectProbReg, maskFloat);
                MicroAPI::Add(gradWeightsFloatReg, gradWeightsFloatReg, tmpFloatReg, copyOutMask);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
            MicroAPI::DataCopy(gradWeightsPtr + r * 8, gradWeightsFloatReg, copyOutMask);
        }
    }
}


template<typename T>
__aicore__ inline void ComputeGradReluSimd(
    LocalTensor<float> gradSelectProbLocal, LocalTensor<float> weightsLocal,
    LocalTensor<float> selectProbLocal, LocalTensor<float> gradValueLocal,
    const uint16_t repeatTimesRow, const uint16_t repeatTimesCol)
{
    __local_mem__ float* gradSelectProbPtr =
        (__local_mem__ float*) gradSelectProbLocal.GetPhyAddr();
    __local_mem__ float* weightsPtr =
        (__local_mem__ float*) weightsLocal.GetPhyAddr();
    __local_mem__ float* selectProbPtr =
        (__local_mem__ float*) selectProbLocal.GetPhyAddr();
    __local_mem__ float* gradValuePtr =
        (__local_mem__ float*) gradValueLocal.GetPhyAddr();
    
    __VEC_SCOPE__ {
        RegTensor<float> gradValueReg;
        RegTensor<float> weightsReg;
        RegTensor<float> tmpReg;
        RegTensor<float> gradReluOutReg;
        RegTensor<float> selectProbReg;
        RegTensor<float> zerosReg;
        RegTensor<float> gradSelectProbReg;

        MicroAPI::MaskReg maskReg = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        MicroAPI::MaskReg cmpMaskReg;

        MicroAPI::Duplicate(zerosReg, 0, maskReg);
        for (uint16_t r = 0; r < repeatTimesRow; r++) {
            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_E2B_B32>(weightsReg, weightsPtr + r * 8);
            for (uint16_t c = 0; c < repeatTimesCol; c++) {
                MicroAPI::LoadAlign<float, DataCopyMode::DATA_BLOCK_COPY>(
                    selectProbReg, selectProbPtr + r * 64 * repeatTimesCol + c * 8,
                    (uint32_t) repeatTimesCol, maskReg);
                MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BLK>(gradValueReg, gradValuePtr + c * 8);
                MicroAPI::Mul(gradReluOutReg, gradValueReg, weightsReg, maskReg);
                MicroAPI::Compares<float, CMPMODE::GT>(cmpMaskReg, selectProbReg, 0.f, maskReg);
                MicroAPI::Select(gradSelectProbReg, gradReluOutReg, zerosReg, cmpMaskReg);
                MicroAPI::StoreAlign<float, DataCopyMode::DATA_BLOCK_COPY>(
                    gradSelectProbPtr + r * 64 * repeatTimesCol + c * 8,
                    gradSelectProbReg, (uint32_t) repeatTimesCol, maskReg);
            }
        }
    }
}

template<typename T>
__aicore__ inline void ND2NZSimd(
    LocalTensor<T> dstLocal, LocalTensor<float> srcLocal,
    const uint16_t repeatTimesRow, const uint32_t elemCountPerRow)
{
    int32_t elemCountPerDataBlock = UB_BLOCK_BYTE_SIZE / sizeof(T);
    int32_t dataBlockPerRow = elemCountPerRow / (UB_BLOCK_BYTE_SIZE / sizeof(T));
    uint32_t elemCountPerRowForMask = elemCountPerRow;

    __local_mem__ float* srcPtr =
        (__local_mem__ float*) srcLocal.GetPhyAddr();
    __local_mem__ T* dstPtr =
        (__local_mem__ T*) dstLocal.GetPhyAddr();

    static constexpr MicroAPI::CastTrait castTrait = 
        {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    __VEC_SCOPE__ {
        RegTensor<float> tmpFP32Reg;
        RegTensor<T> tmp1Reg, tmp2Reg;
        MicroAPI::MaskReg maskRegB16 = MicroAPI::UpdateMask<T>(elemCountPerRowForMask);
        MicroAPI::MaskReg maskReg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

        for (uint16_t r = 0; r < repeatTimesRow; r++) {
            MicroAPI::DataCopy(tmpFP32Reg, srcPtr + r * elemCountPerRow);

            MicroAPI::Cast<T, float, castTrait>(tmp1Reg, tmpFP32Reg, maskReg);
            MicroAPI::DeInterleave(tmp1Reg, tmp2Reg, tmp1Reg, tmp2Reg);

            MicroAPI::DataCopy<T, DataCopyMode::DATA_BLOCK_COPY>(
                dstPtr + r * elemCountPerDataBlock,
                tmp1Reg, (uint32_t) repeatTimesRow, maskRegB16);
        }
    }
}