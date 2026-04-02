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
 * \file grouped_matmul_mxfp8fp4_block_mmad_resplit.h
 * \brief
 */
#ifndef GROUPED_MATMUL_MXFP8FP4_BLOCK_MMAD_RESPLIT_H
#define GROUPED_MATMUL_MXFP8FP4_BLOCK_MMAD_RESPLIT_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "include/experimental/tensor_api/tensor.h"
#include "../prologue/tool.h"
#include "../tile/tile_mmad_mx.h"
#include "../tile/copy_scale_gm_to_l1.h"
#include "../tile/copy_scale_l1_to_l0a.h"
#include "../tile/copy_scale_l1_to_l0b.h"

using AscendC::Dn2NzParams;
using AscendC::GetBlockIdx;
using AscendC::HardEvent;
using AscendC::PipeBarrier;
using AscendC::SetFlag;
using AscendC::TPosition;
using AscendC::WaitFlag;
using AscendC::BLOCK_CUBE;
using AscendC::TEventID;
using WeightQuantBatchMatmulV2::Arch35::GetKBUnit;
using WeightQuantBatchMatmulV2::Arch35::DOUBLE_BUFFER_NUM;
using WeightQuantBatchMatmulV2::Arch35::L1_SIZE;
using WeightQuantBatchMatmulV2::Arch35::L1_SIZE_BYTE;
using WeightQuantBatchMatmulV2::Arch35::BIAS_L1_SIZE;
using WeightQuantBatchMatmulV2::Arch35::SYNC_AIV_AIC_FLAG;
using WeightQuantBatchMatmulV2::Arch35::SYNC_AIC_AIV_FLAG;
using WeightQuantBatchMatmulV2::Arch35::SYNC_MODE4;
using WeightQuantBatchMatmulV2::Arch35::FLAG_ID_MAX;
using WeightQuantBatchMatmulV2::Arch35::CeilAlign;

namespace Block {

static constexpr int32_t SINGLE_BUFFER_NUM = 1;

struct BlockMmadOffsetParam {
    uint64_t mL1Size;
    uint64_t kaL1Size;
    uint64_t kbL1Size;
    uint64_t nL1Size;
    uint64_t kSize;
};

#define WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM                                                                                \
    template <class L1TileShape, class L0TileShape, class ATypeTuple_, class LayoutATuple_, class BTypeTuple_, class LayoutBTuple_,            \
              class CType_, class LayoutC_, class BiasType_>

#define WQBMM_CUBE_COMPUTE_CLASS                                                                                         \
    BlockMmad<GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit, L1TileShape, L0TileShape, ATypeTuple_, LayoutATuple_, BTypeTuple_,        \
              LayoutBTuple_, CType_, LayoutC_, BiasType_, void>

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
class WQBMM_CUBE_COMPUTE_CLASS {
public:
    using DispatchPolicy = GROUPED_MATMUL::KernelMixDynamicKL1NTailResplit;
    using ScaleBType = typename AscendC::Std::tuple_element<1, BTypeTuple_>::type;
    using ScaleAType = typename AscendC::Std::tuple_element<1, ATypeTuple_>::type;
    using BiasType = BiasType_;
    using AType = typename AscendC::Std::tuple_element<0, ATypeTuple_>::type;
    using CType = CType_;
    using LayoutA = typename AscendC::Std::tuple_element<0, LayoutATuple_>::type;
    using LayoutC = LayoutC_;
    using LayoutScaleA = typename AscendC::Std::tuple_element<1, LayoutATuple_>::type;
    using LayoutScaleB = typename AscendC::Std::tuple_element<1, LayoutBTuple_>::type;;
    // TODO assert x
    // TODO assert weight custom NZ

    struct Params {
        __gm__ AType *ptrA;
        __gm__ ScaleAType *ptrScaleA;
        __gm__ ScaleBType *ptrScaleB;
        __gm__ CType *ptrC;
        uint64_t kSize;
        uint8_t hasBias;
    };

    __aicore__ inline BlockMmad() = delete;
    __aicore__ inline BlockMmad(const Params& params);
    template <typename TensorA, typename TensorC, typename TensorScaleA, typename TensorScaleB>
    __aicore__ inline void operator()(const TensorA &tensorA, const TensorC &tensorC,
                                      const TensorScaleA &tensorScaleA, const TensorScaleB &tensorScaleB);
    __aicore__ inline void PrefetchA(uint64_t aPrefetchSize, uint64_t xSizeLimit);
    __aicore__ inline ~BlockMmad();

protected:
    __aicore__ inline void WaitAivToAic();
    __aicore__ inline void SetAicToAiv();

private:
    __aicore__ inline void CalcDynamicKBlock(uint64_t mL1Size, uint64_t nL1Size, uint64_t &kaL1Size,
                                             uint64_t &kbL1Size) const;
    template <typename TensorB>
    __aicore__ inline void LaunchMatmul(const TensorB &tensorBL1, int64_t kbOffset, uint64_t kbL1RealSize,
                                        uint64_t cvLoopIdx, const BlockMmadOffsetParam &param);
    __aicore__ inline void WaitMTE1ToMTE2(uint64_t kaGmOffset, const BlockMmadOffsetParam &offsetParam);
    __aicore__ inline void SetMTE1ToMTE2(uint64_t kaGmOffset, const BlockMmadOffsetParam &offsetParam);
    __aicore__ inline void WaitScaleMTE1ToMTE2(uint64_t kbGmOffset);
    __aicore__ inline void SetScaleMTE1ToMTE2(uint64_t kbGmOffset, const BlockMmadOffsetParam &offsetParam);
    template <typename TensorA>
    __aicore__ inline void CopyAAndBiasGmToL1(const TensorA &tensorA, const BlockMmadOffsetParam &param,
                                              int64_t kaGmOffset, uint64_t cvLoopIdx);
    template <typename TensorScaleA, typename TensorScaleB>
    __aicore__ inline void CopyMxScaleGmToL1(const TensorScaleA &tensorScaleA, const TensorScaleB &tensorScaleB,
                                             const BlockMmadOffsetParam &param, uint64_t kbL1Offset);
    template <typename TensorC>
    __aicore__ inline void GetTensorC(const TensorC &tensorC, const BlockMmadOffsetParam &param);
    __aicore__ inline void ClearAFullLoadFlag();
    using MakeLayoutBias = typename AscendC::Te::NDLayoutFormat<BiasType>;

    using MakeLayoutAL1 = AscendC::Te::NzLayoutFormat<AType>;
    using MakeLayoutScaleAL1 = typename AscendC::Te::ZzLayoutFormat<fp8_e8m0_t>;
    using MakeLayoutScaleBL1 = typename AscendC::Te::NnLayoutFormat<fp8_e8m0_t>;

    using MakeLayoutAL0 = AscendC::Te::NzLayoutFormat<AType>;
    using MakeLayoutBL0 = AscendC::Te::ZnLayoutFormat<AType>;
    using MakeLayoutScaleAL0 = typename AscendC::Te::ZzLayoutFormat<fp8_e8m0_t>;
    using MakeLayoutScaleBL0 = typename AscendC::Te::NnLayoutFormat<fp8_e8m0_t>;
    using MakeLayoutBt = typename AscendC::Te::NDLayoutFormat<BiasType>;
    static constexpr uint64_t L0_BUF_NUM = 2;
    static constexpr uint64_t MXFP_DIVISOR_SIZE = 64;

    __aicore__ inline void PrefetchA(uint64_t aPrefetchSize, uint64_t aGmSize, uint64_t aL1Offset);
    __aicore__ inline void InitSync();
    __aicore__ inline uint64_t CheckMaxSpace(const BlockMmadOffsetParam &param);
    template <typename TensorA>
    __aicore__ inline void CopyAGmToL1SingleBuffer(const BlockMmadOffsetParam &param, int64_t kaGmOffset,
                                                   int64_t kbL1RealSize, int64_t biasRealN,
                                                   const TensorA &tensorA);

    uint64_t cvLoopIdx_ = 0;
    uint64_t weightL1DbOffset_;

    int8_t aL1DbNum_;
    bool isBias_;
    static constexpr uint32_t KB_UNIT = GetKBUnit<AType>();
    static constexpr uint64_t MX_SCALE_L1_SIZE = 32 * GetKBUnit<AType>() * sizeof(AType);
    static constexpr uint64_t MX_SCALE_K_L1_SIZE = 4096;
    static constexpr uint64_t BIAS_TABLE_OFFSET_B32 = 2 * 256;
    static constexpr uint64_t MX_GROUP_SIZE = 32;
    static constexpr uint64_t MX_A8W4_L1_K_CONFIG_256 = 256;
    static constexpr uint64_t MX_A8W4_L1_K_CONFIG_512 = 512;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD = 128;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_256 = 256;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_240 = 240;

    uint64_t aL1Count_;
    uint64_t aL1MaxHalfCount_;
    uint64_t mxScaleBufIdx_ = 0;
    uint64_t aL1BufIdx_ = 0;

    AscendC::TEventID cubeEventIdsMxScaleMte1ToMte2_[DOUBLE_BUFFER_NUM];
    AscendC::TEventID cubeEventIdsMte1ToMte2_[DOUBLE_BUFFER_NUM];
    AscendC::TEventID cubeEventIdMte2ToMte1_;
    AscendC::TEventID cubeEventIdMte1ToMte2_;

    uint64_t aL1Offset_;
    uint64_t biasL1Offset_;
    uint64_t mxScaleAL1Offset_;
    uint64_t mxScaleBL1Offset_;

    uint64_t aL1DbOffset_;
    uint64_t biasL1DbOffset_;
    uint64_t mxScaleAL1DbOffset_;
    uint64_t mxScaleBL1DbOffset_;

    uint64_t l0LoopIdx_ = 0;
    uint64_t mxA8W4L1KDynamicConfigMThreshold_;

    __gm__ uint8_t *aPrefetchAddr_ = nullptr;

    decltype(AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr(reinterpret_cast<__cbuf__ AType*>(0)), MakeLayoutAL1{}(16UL, 16UL))) tensorAL1_;
    decltype(AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr(reinterpret_cast<__cbuf__ fp8_e8m0_t*>(0)), MakeLayoutScaleAL1{}(16UL, 16UL))) tensorScaleAL1_;
    decltype(AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr(reinterpret_cast<__cbuf__ fp8_e8m0_t*>(0)), MakeLayoutScaleBL1{}(16UL, 16UL))) tensorScaleBL1_;
};

} // namespace Block

namespace Block {

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline uint64_t WQBMM_CUBE_COMPUTE_CLASS::CheckMaxSpace(const BlockMmadOffsetParam &param)
{
    uint64_t maxSpace = aL1MaxHalfCount_ * param.kbL1Size * CeilAlign(param.mL1Size, static_cast<uint64_t>(BLOCK_CUBE));
    if (param.kbL1Size > 0 && param.kSize % param.kbL1Size == 0 && maxSpace <= aL1DbOffset_) {
        return maxSpace;
    }
    return 0;
}

static constexpr uint64_t L0_BUF_OFFSET_B8 = 32 * 1024;
static constexpr uint16_t UNIT_FLAG_ENABLE = 2;
static constexpr uint16_t UNIT_FLAG_ENABLE_AUTO_CLOSE = 3;
static constexpr TEventID eventIdMToMte1_ = 3;
static constexpr TEventID eventIdMte1ToM_ = 3;

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
template <typename TensorB>
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::LaunchMatmul(const TensorB &tensorBL1, int64_t kbOffset,
                                                              uint64_t kbL1RealSize, uint64_t cvLoopIdx,
                                                              const BlockMmadOffsetParam &param)
{
    SetFlag<HardEvent::MTE2_MTE1>(cubeEventIdMte2ToMte1_);
    WaitFlag<HardEvent::MTE2_MTE1>(cubeEventIdMte2ToMte1_);
    uint64_t aL1Offset = 0;
    if (aL1DbNum_ == SINGLE_BUFFER_NUM) {
        uint64_t maxSpace = CheckMaxSpace(param);
        if (maxSpace > 0) {
            aL1Offset = (aL1BufIdx_ & 1) * aL1DbOffset_ +
                        CeilAlign(param.mL1Size, static_cast<uint64_t>(BLOCK_CUBE)) *
                            (static_cast<uint64_t>(kbOffset) / (param.kbL1Size * 2) * param.kbL1Size);
        } else {
            aL1Offset = CeilAlign(param.mL1Size, static_cast<uint64_t>(BLOCK_CUBE)) * kbOffset;
        }
    } else {
        aL1Offset = (aL1BufIdx_ & 1) * aL1DbOffset_;
    }
    bool isLastGmK = kbOffset + kbL1RealSize >= param.kSize;
    bool isFirstGmK = kbOffset == 0;
    uint64_t l0KSize = (param.mL1Size <= 128 && param.nL1Size <= 128) ? 256 : 128;
    for (uint64_t l1KOffset = 0; l1KOffset < kbL1RealSize; l1KOffset += l0KSize) {
        bool isLastL1K = l1KOffset + l0KSize >= kbL1RealSize;
        uint64_t realL0k = isLastL1K ? kbL1RealSize - l1KOffset : l0KSize;
        uint64_t loopId = l0LoopIdx_ % L0_BUF_NUM;
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1_ + loopId);

        auto CopyL12L0 = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0{});
        auto layoutAL0 = MakeLayoutAL0{}(param.mL1Size, realL0k);
        auto tensorAL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeL0AmemPtr<AType>(loopId * L0_BUF_OFFSET_B8), layoutAL0);
        auto tensorBlockAL1 = tensorAL1_(
            AscendC::Te::MakeCoord(0, (l1KOffset + kbOffset) % param.kaL1Size),
            AscendC::Te::MakeShape(param.mL1Size, realL0k));
        AscendC::Te::Copy(CopyL12L0, tensorAL0, tensorBlockAL1);

        auto layoutScaleAL0 = MakeLayoutScaleAL0{}(
            param.mL1Size, Cgmct::Gemm::CeilDiv(realL0k, MXFP_DIVISOR_SIZE) * 2);
        auto tensorScaleAL0 =
            AscendC::Te::MakeTensor(AscendC::Te::MakeL0AmemPtr<fp8_e8m0_t>(loopId * L0_BUF_OFFSET_B8), layoutScaleAL0);
        auto CopyL12L0MxScaleA3510 = AscendC::Te::MakeCopy(Cgmct::Gemm::Tile::CopyL12L0MxScaleA3510{});
        AscendC::Te::Copy(
            CopyL12L0MxScaleA3510, tensorScaleAL0, tensorScaleAL1_,
            AscendC::Te::MakeCoord(0, ((l1KOffset + kbOffset) % MX_SCALE_K_L1_SIZE) / MX_GROUP_SIZE));

        auto layoutBL0 = MakeLayoutBL0{}(realL0k, param.nL1Size);
        auto tensorBL0 = AscendC::Te::MakeTensor(AscendC::Te::MakeL0BmemPtr<AType>(loopId * L0_BUF_OFFSET_B8), layoutBL0);
        auto tensorBlockBL1 = tensorBL1(
            AscendC::Te::MakeCoord(l1KOffset, 0), AscendC::Te::MakeShape(realL0k, param.nL1Size));
        AscendC::Te::Copy(CopyL12L0, tensorBL0, tensorBlockBL1);

        auto layoutScaleBL0 = MakeLayoutScaleBL0{}(
            Cgmct::Gemm::CeilDiv(realL0k, MXFP_DIVISOR_SIZE) * 2, param.nL1Size);
        auto tensorScaleBL0 =
            AscendC::Te::MakeTensor(AscendC::Te::MakeL0BmemPtr<fp8_e8m0_t>(loopId * L0_BUF_OFFSET_B8), layoutScaleBL0);
        auto CopyL12L0MxScaleB3510 = AscendC::Te::MakeCopy(Cgmct::Gemm::Tile::CopyL12L0MxScaleB3510{});
        AscendC::Te::Copy(
            CopyL12L0MxScaleB3510, tensorScaleBL0, tensorScaleBL1_,
            AscendC::Te::MakeCoord(((l1KOffset + kbOffset) % MX_SCALE_K_L1_SIZE) / MX_GROUP_SIZE, 0));

        if (isBias_ && isFirstGmK && l1KOffset == 0) {
            auto CopyL12BT = AscendC::Te::MakeCopy(AscendC::Te::CopyL12BT{});
            auto layoutBt = MakeLayoutBt{}(1L, Cgmct::Gemm::Align(param.nL1Size, AscendC::BLOCK_CUBE));
            auto tensorBt = AscendC::Te::MakeTensor(
                AscendC::Te::MakeBiasmemPtr<float>(loopId * BIAS_TABLE_OFFSET_B32 * sizeof(float)), layoutBt);
            auto layoutBiasL1 = MakeLayoutBias{}(1L, Cgmct::Gemm::Align(param.nL1Size, AscendC::BLOCK_CUBE));
            auto tensorBiasL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeL1memPtr<BiasType>(biasL1Offset_ + (cvLoopIdx & 1) * biasL1DbOffset_ * sizeof(BiasType)), layoutBiasL1);
            AscendC::Te::Copy(CopyL12BT, tensorBt, tensorBiasL1);
        }

        bool isFirstK = isFirstGmK && l1KOffset == 0;
        SetFlag<HardEvent::MTE1_M>(eventIdMte1ToM_);
        WaitFlag<HardEvent::MTE1_M>(eventIdMte1ToM_);
        uint8_t mmadUnitFlag = (isLastGmK && isLastL1K) ? UNIT_FLAG_ENABLE_AUTO_CLOSE : UNIT_FLAG_ENABLE;

        auto layoutL0C = AscendC::Te::MakeL0CLayout(param.mL1Size, param.nL1Size);
        auto tensorL0C = AscendC::Te::MakeTensor(AscendC::Te::MakeL0CmemPtr<float>(0), layoutL0C);
        if (isBias_ && isFirstK) {
            auto layoutBt =
                MakeLayoutBt{}(1L, Cgmct::Gemm::Align(param.nL1Size, AscendC::BLOCK_CUBE));
            auto tensorBt = AscendC::Te::MakeTensor(
                AscendC::Te::MakeBiasmemPtr<float>(loopId * BIAS_TABLE_OFFSET_B32 * sizeof(float)), layoutBt);
            AscendC::Te::Mad(
                        AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<Cgmct::Gemm::Tile::MmadMxWithBias>>{}.with(
                            static_cast<uint16_t>(param.mL1Size),
                            static_cast<uint16_t>(realL0k),
                            static_cast<uint16_t>(param.nL1Size), mmadUnitFlag, true, false),
                        tensorL0C, tensorAL0, tensorBL0, tensorBt);
        } else {
            AscendC::Te::Mad(
                        AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<Cgmct::Gemm::Tile::MmadMx>>{}.with(
                            static_cast<uint16_t>(param.mL1Size),
                            static_cast<uint16_t>(realL0k),
                            static_cast<uint16_t>(param.nL1Size), mmadUnitFlag, false, isFirstK),
                        tensorL0C, tensorAL0, tensorBL0);
        }

        SetFlag<HardEvent::M_MTE1>(eventIdMToMte1_ + loopId);
        l0LoopIdx_++;
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::WaitMTE1ToMTE2(uint64_t kaGmOffset,
                                                                const BlockMmadOffsetParam &offsetParam)
{
    if (aL1DbNum_ > SINGLE_BUFFER_NUM && kaGmOffset % offsetParam.kaL1Size == 0) {
        WaitFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMte1ToMte2_[aL1BufIdx_ & 1]);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::WaitScaleMTE1ToMTE2(uint64_t kbGmOffset)
{
    if (kbGmOffset % MX_SCALE_K_L1_SIZE == 0) {
        WaitFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMxScaleMte1ToMte2_[mxScaleBufIdx_ & 1]);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::SetScaleMTE1ToMTE2(uint64_t kbGmOffset,
                                                                    const BlockMmadOffsetParam &offsetParam)
{
    if ((kbGmOffset + offsetParam.kbL1Size) % MX_SCALE_K_L1_SIZE == 0 ||
        kbGmOffset + offsetParam.kbL1Size >= offsetParam.kSize) {
        SetFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMxScaleMte1ToMte2_[mxScaleBufIdx_ & 1]);
        mxScaleBufIdx_++;
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::SetMTE1ToMTE2(uint64_t kaGmOffset,
                                                               const BlockMmadOffsetParam &offsetParam)
{
    if (aL1DbNum_ > SINGLE_BUFFER_NUM && ((kaGmOffset + offsetParam.kbL1Size) % offsetParam.kaL1Size == 0 ||
                                          kaGmOffset + offsetParam.kbL1Size >= offsetParam.kSize)) {
        SetFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMte1ToMte2_[aL1BufIdx_ & 1]);
        aL1BufIdx_++;
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
template <typename TensorA>
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::CopyAGmToL1SingleBuffer(const BlockMmadOffsetParam &param,
                                                                         int64_t kaGmOffset, int64_t kbL1RealSize,
                                                                         int64_t biasRealN, const TensorA &tensorA)
{
    AscendC::Nd2NzParams nd2nzParams;
    uint64_t maxSpace = CheckMaxSpace(param);
    if (maxSpace > 0) {
        // TODO 不支持
    } else {
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        auto layoutAL1 = MakeLayoutAL1{}(param.mL1Size, param.kSize);
        tensorAL1_ = AscendC::Te::MakeTensor(AscendC::Te::MakeL1memPtr<AType>(0), layoutAL1);
        auto gmBlockA =
            tensorA(AscendC::Te::MakeCoord(0, kaGmOffset), AscendC::Te::MakeShape(param.mL1Size, param.kSize));
        AscendC::Te::Copy(copyGM2L1, tensorAL1_, gmBlockA);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
template <typename TensorA>
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::CopyAAndBiasGmToL1(const TensorA &tensorA,
                                                                    const BlockMmadOffsetParam &param,
                                                                    int64_t kaGmOffset, uint64_t cvLoopIdx)
{
    if (kaGmOffset % param.kaL1Size != 0) {
        return;
    }
    int64_t kaL1RealSize = (kaGmOffset + param.kaL1Size) >= param.kSize ? param.kSize - kaGmOffset : param.kaL1Size;
    auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
    if (aL1DbNum_ > SINGLE_BUFFER_NUM) {
        auto layoutAL1 = MakeLayoutAL1{}(param.mL1Size, kaL1RealSize);
        auto gmBlockA =
            tensorA(AscendC::Te::MakeCoord(0, kaGmOffset), AscendC::Te::MakeShape(param.mL1Size, kaL1RealSize));
        tensorAL1_ = AscendC::Te::MakeTensor(
            AscendC::Te::MakeL1memPtr<AType>(aL1Offset_ + (aL1BufIdx_ & 1) * aL1DbOffset_), layoutAL1);
        AscendC::Te::Copy(copyGM2L1, tensorAL1_, gmBlockA);
    } else if (aL1DbNum_ == SINGLE_BUFFER_NUM && kaGmOffset == 0) {
        CopyAGmToL1SingleBuffer(param, kaGmOffset, kaL1RealSize, param.nL1Size, tensorA);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
template <typename TensorScaleA, typename TensorScaleB>
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::CopyMxScaleGmToL1(const TensorScaleA &tensorScaleA,
                                                                   const TensorScaleB &tensorScaleB,
                                                                   const BlockMmadOffsetParam &param,
                                                                   uint64_t kbL1Offset)
{
    if (kbL1Offset % MX_SCALE_K_L1_SIZE != 0) {
        return;
    }
    uint64_t scaleKGmSize = param.kSize / MX_GROUP_SIZE;
    uint64_t scaleKL1StandardLen = MX_SCALE_K_L1_SIZE / MX_GROUP_SIZE;
    uint64_t scaleKL1RealSize = (kbL1Offset + MX_SCALE_K_L1_SIZE) > param.kSize ?
                                    (param.kSize - kbL1Offset) / MX_GROUP_SIZE :
                                    scaleKL1StandardLen;
    auto CopyScaleGM2L1 = AscendC::Te::MakeCopy(Cgmct::Gemm::Tile::CopyScaleGM2L1{});
    auto layoutScaleAL1 = MakeLayoutScaleAL1{}(param.mL1Size, scaleKL1RealSize);
    tensorScaleAL1_ = AscendC::Te::MakeTensor(
        AscendC::Te::MakeL1memPtr<fp8_e8m0_t>(mxScaleAL1Offset_ + (mxScaleBufIdx_ & 1) * mxScaleAL1DbOffset_), layoutScaleAL1);
    auto gmBlockScaleA = tensorScaleA(AscendC::Te::MakeCoord(0, kbL1Offset / MX_GROUP_SIZE),
                                      AscendC::Te::MakeShape(param.mL1Size, scaleKL1RealSize));
    AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleAL1_, gmBlockScaleA);

    auto layoutScaleBL1 = MakeLayoutScaleBL1{}(scaleKL1RealSize, param.nL1Size);
    tensorScaleBL1_ = AscendC::Te::MakeTensor(
        AscendC::Te::MakeL1memPtr<fp8_e8m0_t>(mxScaleBL1Offset_ + (mxScaleBufIdx_ & 1) * mxScaleBL1DbOffset_), layoutScaleBL1);
    auto gmBlockScaleB = tensorScaleB(AscendC::Te::MakeCoord(kbL1Offset / MX_GROUP_SIZE, 0),
                                      AscendC::Te::MakeShape(scaleKL1RealSize, param.nL1Size));
    AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleBL1_, gmBlockScaleB);
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::ClearAFullLoadFlag()
{
    if (aL1DbNum_ == SINGLE_BUFFER_NUM) {
        SetFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMte1ToMte2_[0]);
        WaitFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMte1ToMte2_[0]);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::InitSync()
{
    for (uint64_t i = 0; i < DOUBLE_BUFFER_NUM; i++) {
        cubeEventIdsMte1ToMte2_[i] = i;
        cubeEventIdsMxScaleMte1ToMte2_[i] = DOUBLE_BUFFER_NUM + i;
        SetFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMxScaleMte1ToMte2_[i]);
        if (aL1DbNum_ > SINGLE_BUFFER_NUM) {
            SetFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMte1ToMte2_[i]);
        }
    }
    cubeEventIdMte1ToMte2_ = 2 * DOUBLE_BUFFER_NUM;
    cubeEventIdMte2ToMte1_ = 0;
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::PrefetchA(uint64_t aPrefetchSize, uint64_t xSizeLimit,
                                                           uint64_t perloadBufferOffset)
{
    uint64_t xOffset = GetBlockIdx() * aPrefetchSize;
    if (aPrefetchSize == 0 || xOffset >= xSizeLimit) {
        return;
    }
#ifdef __XBL_PRINT__
    AscendC::printf("[ERROR] need prefetchA 3");
#endif
    // TODO 支持
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::PrefetchA(uint64_t aPrefetchSize, uint64_t xSizeLimit)
{
    uint64_t xOffset = GetBlockIdx() * aPrefetchSize;
    if (aPrefetchSize == 0 || xOffset >= xSizeLimit) {
        return;
    }
    SetFlag<HardEvent::MTE1_MTE2>(cubeEventIdMte1ToMte2_);
    WaitFlag<HardEvent::MTE1_MTE2>(cubeEventIdMte1ToMte2_);

    uint64_t blockLen = xOffset + aPrefetchSize > xSizeLimit ? xSizeLimit - xOffset : aPrefetchSize;
#ifdef __XBL_PRINT__
    AscendC::printf("[INFO] prefetchA 2 base_addr %p offset %d size %d", aPrefetchAddr_, xOffset, blockLen);
#endif
    auto aPrefetchGm = AscendC::Te::MakeTensor(
        AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ uint8_t*>(aPrefetchAddr_ + xOffset)),
        AscendC::Te::MakeNDLayout<uint8_t>(1UL, blockLen));
    auto aPrefetchL1 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeL1memPtr(reinterpret_cast<__cbuf__ uint8_t*>(aL1Offset_)),
        AscendC::Te::MakeNDLayout<uint8_t>(1UL, blockLen));
    auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
    AscendC::Te::Copy(copyGM2L1, aPrefetchL1, aPrefetchGm);
    PipeBarrier<PIPE_MTE2>();
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
template <typename TensorC>
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::GetTensorC(const TensorC &tensorC,
                                                            const BlockMmadOffsetParam &param)
{
    constexpr uint64_t FP32_64_AS_UINT64 = 0x42800000;
    auto layoutL0C = AscendC::Te::MakeL0CLayout(param.mL1Size, param.nL1Size);
    auto tensorL0C = AscendC::Te::MakeTensor(AscendC::Te::MakeL0CmemPtr<float>(0), layoutL0C);
    auto CopyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
    AscendC::Te::Copy(CopyL0C2GM, tensorC, tensorL0C, FP32_64_AS_UINT64, AscendC::Te::FixpipeParams{/*unitflag*/3});
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline WQBMM_CUBE_COMPUTE_CLASS::BlockMmad(const Params& params)
{
    isBias_ = params.hasBias;
    mxA8W4L1KDynamicConfigMThreshold_ = isBias_ ? MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_240 :
                                                  MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_256;
    biasL1DbOffset_ = 0;
    static constexpr uint64_t MXA8W4_WEIGHT_SIZE = 256 * 256;
    static constexpr uint64_t MX_BIAS_L1_SIZE = BIAS_L1_SIZE * GetKBUnit<BiasType>() * sizeof(BiasType);
    weightL1DbOffset_ = L1_SIZE * GetKBUnit<AType>() - MXA8W4_WEIGHT_SIZE;
    uint64_t l1RemainSize = L1_SIZE_BYTE - MXA8W4_WEIGHT_SIZE * DOUBLE_BUFFER_NUM;
    uint64_t l1StartSize = MXA8W4_WEIGHT_SIZE;
    uint64_t biasL1Offset = l1StartSize;
    if (isBias_) {
        biasL1DbOffset_ = (l1RemainSize - MX_BIAS_L1_SIZE) / sizeof(BiasType);
        l1RemainSize -= DOUBLE_BUFFER_NUM * MX_BIAS_L1_SIZE;
        l1StartSize += MX_BIAS_L1_SIZE;
    }

    biasL1Offset_ = biasL1Offset;
    aL1Count_ = 0;
    aL1MaxHalfCount_ = 0;
    aL1DbNum_ = DOUBLE_BUFFER_NUM;

    mxScaleAL1Offset_ = l1StartSize;
    mxScaleAL1DbOffset_ = l1RemainSize - MX_SCALE_L1_SIZE;
    l1RemainSize -= DOUBLE_BUFFER_NUM * MX_SCALE_L1_SIZE;
    l1StartSize += MX_SCALE_L1_SIZE;

    mxScaleBL1Offset_ = l1StartSize;
    mxScaleBL1DbOffset_ = l1RemainSize - MX_SCALE_L1_SIZE;
    l1RemainSize -= DOUBLE_BUFFER_NUM * MX_SCALE_L1_SIZE;
    l1StartSize += MX_SCALE_L1_SIZE;

    aL1Offset_ = l1StartSize;
    aL1DbOffset_ = l1RemainSize >> 1;

    PrefetchA(0, 0, aL1Offset_);
    for (uint64_t i = 0; i < L0_BUF_NUM; i++) {
        SetFlag<HardEvent::M_MTE1>(eventIdMToMte1_ + i);
    }
    InitSync();
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::CalcDynamicKBlock(
    uint64_t mL1Size, uint64_t nL1Size, uint64_t &kaL1Size, uint64_t &kbL1Size) const
{
    kbL1Size = (mL1Size <= mxA8W4L1KDynamicConfigMThreshold_ &&
                nL1Size <= MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD) ?
                   MX_A8W4_L1_K_CONFIG_512 :
                   MX_A8W4_L1_K_CONFIG_256;
    if (mL1Size < nL1Size) {
        uint64_t aL1Size = isBias_ ? 124 * GetKBUnit<AType>() : 128 * GetKBUnit<AType>();
        uint64_t mL1Align = AscendC::CeilAlign(mL1Size, static_cast<uint64_t>(BLOCK_CUBE));
        kaL1Size = aL1Size / (mL1Align * kbL1Size) * kbL1Size;
    } else {
        kaL1Size = kbL1Size;
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
template <typename TensorA, typename TensorC, typename TensorScaleA, typename TensorScaleB>
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::operator()(const TensorA &tensorA, const TensorC &tensorC,
                                                            const TensorScaleA &tensorScaleA,
                                                            const TensorScaleB &tensorScaleB)
{
    BlockMmadOffsetParam blockParam = {};
    blockParam.mL1Size = GetEleFromLayout<decltype(tensorC.Layout()), AttrInfo::SHAPE, AttrInfo::ROW, 1>(tensorC.Layout());
    blockParam.kSize = GetEleFromLayout<decltype(tensorA.Layout()), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(tensorA.Layout());
    blockParam.nL1Size =
        GetEleFromLayout<decltype(tensorC.Layout()), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(tensorC.Layout());
    CalcDynamicKBlock(blockParam.mL1Size, blockParam.nL1Size, blockParam.kaL1Size, blockParam.kbL1Size);
    for (uint64_t kbGmOffset = 0; kbGmOffset < blockParam.kSize; kbGmOffset += blockParam.kbL1Size, cvLoopIdx_++) {
        uint64_t kbL1RealSize = (kbGmOffset + blockParam.kbL1Size) >= blockParam.kSize ?
                                    blockParam.kSize - kbGmOffset :
                                    blockParam.kbL1Size;
        WaitScaleMTE1ToMTE2(kbGmOffset);
        CopyMxScaleGmToL1(tensorScaleA, tensorScaleB, blockParam, kbGmOffset);
        WaitMTE1ToMTE2(kbGmOffset, blockParam);
        CopyAAndBiasGmToL1(tensorA, blockParam, kbGmOffset, cvLoopIdx_);
        WaitAivToAic();
        auto tensorWeightL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeL1memPtr<AType>((cvLoopIdx_ & 1) * weightL1DbOffset_),
            AscendC::Te::ZnLayoutFormat<AType>{}(kbL1RealSize, blockParam.nL1Size));
        LaunchMatmul(tensorWeightL1, kbGmOffset, kbL1RealSize, cvLoopIdx_, blockParam);
        SetMTE1ToMTE2(kbGmOffset, blockParam);
        SetScaleMTE1ToMTE2(kbGmOffset, blockParam);
        SetAicToAiv();
    }
    GetTensorC(tensorC, blockParam);
    ClearAFullLoadFlag();
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline WQBMM_CUBE_COMPUTE_CLASS::~BlockMmad()
{
    for (uint64_t i = 0; i < L0_BUF_NUM; i++) {
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1_ + i);
    }

    for (uint64_t i = 0; i < DOUBLE_BUFFER_NUM; i++) {
        WaitFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMxScaleMte1ToMte2_[i]);
        if (aL1DbNum_ > SINGLE_BUFFER_NUM) {
            WaitFlag<HardEvent::MTE1_MTE2>(cubeEventIdsMte1ToMte2_[i]);
        }
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::WaitAivToAic()
{
    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::SetAicToAiv()
{
    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
}

#undef WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
#undef WQBMM_CUBE_COMPUTE_CLASS
} // namespace Block

#endif  // GROUPED_MATMUL_MXFP8FP4_BLOCK_MMAD_RESPLIT_H
