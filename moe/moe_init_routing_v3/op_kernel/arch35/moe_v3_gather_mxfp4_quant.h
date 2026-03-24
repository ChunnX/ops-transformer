/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v3_gather_mxfp4_quant.h
 * \brief
 */
#ifndef MOE_V3_GATHER_MXFP4_QUANT_H_REGBASE
#define MOE_V3_GATHER_MXFP4_QUANT_H_REGBASE

#include "moe_v3_common.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

template <typename T, typename U>
__simd_vf__ inline void vfMxfp4ComputeMaxExp(__ubuf__ T *xAddr, __ubuf__ uint16_t *maxExpOutAddr, uint32_t xElemNum,
                                        uint16_t vfLoopNum, uint32_t vlForT, uint32_t numVRegBlocks)
{
    using namespace AscendC::MicroAPI;

    // 用于把float16转为bfloat16，不涉及宽度变化
    static constexpr CastTrait traitFP16ToBF16 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                  RoundMode::CAST_TRUNC};
    static constexpr CastTrait traitFP32ToBF16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                  RoundMode::CAST_TRUNC};

    // 0存奇数位元素，1存偶数位元素
    RegTensor<T> x0, x1;
    RegTensor<bfloat16_t> x0BF16, x1BF16;
    RegTensor<uint16_t> exp0, exp1, exp0FP16, exp1FP16, maxExp;
    // 存储FP16/BF16的指数位为1的mask
    RegTensor<uint16_t> emaskFP16, emaskBF16;
    Duplicate(emaskFP16, FP16_EMASK_AND_INF_VAL);
    Duplicate(emaskBF16, BF16_EMASK_AND_INF_VAL);
    // 2字节Reg的MaskALL
    MaskReg maskAllB16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg mask0, mask1, mask0FP16NanInf, mask1FP16NanInf;
    // 非对齐搬出至UB用
    UnalignReg uReg;

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        mask0 = UpdateMask<T>(xElemNum);
        mask1 = UpdateMask<T>(xElemNum);
        MaskDeInterleave<T>(mask0, mask1, mask0, mask1);

        // 双搬入，奇数位放x0、偶数位放x1
        if constexpr (IsSameType<T, float>::value) {
            DataCopy<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B32>(x0, x1, xAddr, vlForT * 2);
        } else {
            DataCopy<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(x0, x1, xAddr, vlForT * 2);
        }
        if constexpr (IsSameType<T, float>::value) {
            // fp32先转bf16
            Cast<bfloat16_t, T, traitFP32ToBF16>(x0BF16, x0, mask0);
            Cast<bfloat16_t, T, traitFP32ToBF16>(x1BF16, x1, mask1);
            // 用BF16_EMASK_AND_INF_VAL提取BF16的指数位
            And(exp0, (RegTensor<uint16_t> &)x0BF16, emaskBF16, mask0);
            And(exp1, (RegTensor<uint16_t> &)x1BF16, emaskBF16, mask1);
        } else if constexpr (IsSameType<T, half>::value) {
            // 单独提取fp16中inf/nan的元素
            And(exp0FP16, (RegTensor<uint16_t> &)x0, emaskFP16, mask0);
            And(exp1FP16, (RegTensor<uint16_t> &)x1, emaskFP16, mask1);
            Compare<uint16_t, CMPMODE::EQ>(mask0FP16NanInf, exp0FP16, emaskFP16, mask0);
            Compare<uint16_t, CMPMODE::EQ>(mask1FP16NanInf, exp1FP16, emaskFP16, mask1);

            // fp16要先转成bf16
            Cast<bfloat16_t, T, traitFP16ToBF16>(x0BF16, x0, mask0);
            Cast<bfloat16_t, T, traitFP16ToBF16>(x1BF16, x1, mask1);
            // 用BF16_EMASK_AND_INF_VAL提取BF16的指数位
            And(exp0, (RegTensor<uint16_t> &)x0BF16, emaskBF16, mask0);
            And(exp1, (RegTensor<uint16_t> &)x1BF16, emaskBF16, mask1);

            //exp[expFP16==nan/inf]=nan/inf
            Select(exp0, emaskBF16, exp0, mask0FP16NanInf);
            Select(exp1, emaskBF16, exp1, mask1FP16NanInf);
        } else {
            // 用BF16_EMASK_AND_INF_VAL提取BF16的指数位
            And(exp0, (RegTensor<uint16_t> &)x0, emaskBF16, mask0);
            And(exp1, (RegTensor<uint16_t> &)x1, emaskBF16, mask1);
        }
        // 先计算奇数位、偶数位之间的最大值，即每2个元素的最大值，放在maxExp
        // 由于mask0和mask1相同或是比mask1多一个元素，使用mask0保证不漏元素
        Max(maxExp, exp0, exp1, mask0);
        // 再每32B（对于T来说就是每16个元素）取一个最大值，此时maskExp里放的是每2个元素的最大值，且末尾数据为0，因此用MaskALL
        ReduceMaxWithDataBlock(maxExp, maxExp, maskAllB16);
        // 非对齐搬出，每次写出numVRegBlocks个元素（ReduceMaxWithDataBlock后放在maxExp头部）
        DataCopyUnAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(maxExpOutAddr, maxExp, uReg, numVRegBlocks);
    }
    // 非对齐搬出收尾，尾部多出的位置写0
    DataCopyUnAlignPost(maxExpOutAddr, uReg, 0);
}

template <typename T, typename U>
__simd_vf__ inline void vfMxfp4ComputeScale(__ubuf__ uint16_t *maxExpInAddr, __ubuf__ uint16_t *mxScaleOutAddr,
                                       __ubuf__ uint16_t *invScaleOutAddr, uint32_t scaleElemNum, uint32_t validScaleElemNum, uint16_t vfLoopNum,
                                       uint32_t vlForT, uint16_t expLowerBoundValue)
{
    using namespace AscendC::MicroAPI;

    RegTensor<uint16_t> maxExp, sharedExp, mxScale, invScale;
    // infBF16存放bf16指数的inf值
    RegTensor<uint16_t> infBF16;
    Duplicate(infBF16, BF16_EMASK_AND_INF_VAL);
    // zeroB16存放B16长度的0值
    RegTensor<uint16_t> zeroB16;
    Duplicate(zeroB16, 0);
    // expLowerBound存放对于maxExp来说过小的值
    RegTensor<uint16_t> expLowerBound;
    Duplicate(expLowerBound, expLowerBoundValue);
    // nanForE8M0以16位的低8位存放float8_e8m0的nan值
    RegTensor<uint16_t> nanForE8M0;
    Duplicate(nanForE8M0, FP8_E8M0_NAN_VAL);
    // invSub用于计算invSCale=BF16_EXP_INVSUB-sharedExp，得到的invScale==1/realScale，可用于量化x
    RegTensor<uint16_t> invSub;
    Duplicate(invSub, BF16_EXP_INVSUB);
    // nanBF16存放bf16的nan值
    RegTensor<uint16_t> nanBF16;
    Duplicate(nanBF16, BF16_NAN_VAL);
    // specialMinE8M0存放float8_e8m0的特殊极小值
    RegTensor<uint16_t> specialMinE8M0;
    Duplicate(specialMinE8M0, FP8_E8M0_SPECIAL_MIN);

    // 循环用mask
    MaskReg maskLoop, maskValid;
    // 存储compare后的结果用的mask
    MaskReg maskInfBF16, maskZero, maskLowExp, maskSpecialMin;

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        maskLoop = UpdateMask<uint16_t>(scaleElemNum);
        maskValid = UpdateMask<uint16_t>(validScaleElemNum);
        // 拷入vfComputeMaxExp算好的maxExp
        DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(maxExp, maxExpInAddr, vlForT);

        // 1.计算并拷出mxScale（float8_e8m0）

        // maskLowExp提取maxExp过小的位置
        Compare<uint16_t, CMPMODE::LT>(maskLowExp, maxExp, expLowerBound, maskValid);
        // maxExp[<expLowerBound]=expLowerBound
        Select<uint16_t>(maxExp, expLowerBound, maxExp, maskLowExp);

        // sharedExp=maxExp-expLowerBound
        Sub(sharedExp, maxExp, expLowerBound, maskValid);
        // mxScale=sharedExp>>BF16_EXP_SHR_BITS) 即把sharedExp存储的指数位右移到低8位，以便存放在float8_e8m0中
        ShiftRights(mxScale, sharedExp, BF16_EXP_SHR_BITS, maskValid);

        // mxScale[maxExp==infBF16]=nanE8M0
        Compare<uint16_t, CMPMODE::EQ>(maskInfBF16, maxExp, infBF16, maskValid);
        Select<uint16_t>(mxScale, nanForE8M0, mxScale, maskInfBF16);
        // mxScale[maxExp==0]=0
        Compare<uint16_t, CMPMODE::EQ>(maskZero, maxExp, zeroB16, maskValid);
        Select<uint16_t>(mxScale, zeroB16, mxScale, maskZero);

        // 拷出算好的mxScale，即为算子输出expandedScale的值。
        // 对于每个uint16的mxScale，都只拷出其低8位，也就是以uint8的长度拷出，实际上存储的就是float8_e8m0类型的二进制值，因此POST_MODE_UPDATE的数目为vlForT/2。
        DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B16>(mxScaleOutAddr, mxScale, vlForT / 2,
                                                                                    maskLoop);

        // 2.计算并拷出invScale（bfloat16）

        // invScale=invSub-sharedExp
        Sub<uint16_t>(invScale, invSub, sharedExp, maskValid);
        // invScale[maxExp==InfBF16]=nanBF16
        Select<uint16_t>(invScale, nanBF16, invScale, maskInfBF16);
        // invScale[maxExp==0]=0
        Select<uint16_t>(invScale, zeroB16, invScale, maskZero);
        // invScale[(invScale=invSub-sharedExp)==0]=FP8_E8M0_SPECIAL_MIN
        Compare<uint16_t, CMPMODE::EQ>(maskSpecialMin, invSub, sharedExp, maskValid);
        Select<uint16_t>(invScale, specialMinE8M0, invScale, maskSpecialMin);

        // 拷出算好的invScale，用于在后续量化xQuant=invScale*x，以uint16的长度拷出，实际存储的为bfloat16类型的二进制值
        DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(invScaleOutAddr, invScale, vlForT, maskLoop);
    }
}

template <typename T, typename U>
__simd_vf__ inline void vfMxfp4ComputeData(__ubuf__ T *xAddr, __ubuf__ uint16_t *invScaleInAddr,
                                      __ubuf__ int8_t *xQuantOutAddr, uint32_t xElemNum, uint16_t vfLoopNum,
                                      uint32_t vlForT, uint32_t numVRegBlocks)
{
    using namespace AscendC::MicroAPI;

    // fp32转bf16，涉及宽度变化，精度截断
    static constexpr CastTrait traitFP32ToBF16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                  RoundMode::CAST_TRUNC};
    // fp16转bf16，不涉及宽度变化，精度截断
    static constexpr CastTrait traitFP16ToBF16 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                  RoundMode::CAST_TRUNC};
    // bf16转b4
    static constexpr CastTrait traitB16ToB4 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                      RoundMode::CAST_RINT};

    RegTensor<uint16_t> invScale;
    RegTensor<float> invScaleFP32;
    RegTensor<T> x0, x1;
    RegTensor<bfloat16_t> x0BF16, x1BF16;
    RegTensor<U> xQuant0, xQuant1;

    MaskReg maskXQuant;
    MaskReg maskAllB16 = CreateMask<uint16_t, MaskPattern::ALL>();

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        maskXQuant = UpdateMask<float>(xElemNum);

        // 拷入待量化的x：x0存放偶数位的值，x1存放奇数位的值，双搬共拷入vlForT*2个元素
        if constexpr (IsSameType<T, float>::value) {
            DataCopy<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B32>(x0, x1, xAddr, vlForT * 2);
        } else {
            DataCopy<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(x0, x1, xAddr, vlForT * 2);
        }
        // 拷入invScale，类型为uint16（计算时翻译成bfloat16），其中每个元素广播至32B，这里就是每个元素重复16次，共256B/32B=8个元素，也就是numVRegBlocks的值
        // 应该是加载(256/32=8)B的元素
        DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(invScale, invScaleInAddr, numVRegBlocks);
        if constexpr (IsSameType<T, float>::value) {
            // 对于T为fp32，先转换为bf16，并计算x*invScale，将结果转换为fp4
            // 1.将x转换为bf16
            Cast<bfloat16_t, T, traitFP32ToBF16>(x0BF16, x0, maskAllB16);
            Cast<bfloat16_t, T, traitFP32ToBF16>(x1BF16, x1, maskAllB16);

            // 2.计算x*invSacle
            Mul(x0BF16, x0BF16, (RegTensor<bfloat16_t> &)invScale, maskAllB16);
            Mul(x1BF16, x1BF16, (RegTensor<bfloat16_t> &)invScale, maskAllB16);
            Interleave(x0BF16, x1BF16, x0BF16, x1BF16);

            // 3. 将bf16结果转换为fp4
            Cast<U, bfloat16_t, traitB16ToB4>(xQuant0, x0BF16, maskAllB16);
            Cast<U, bfloat16_t, traitB16ToB4>(xQuant1, x1BF16, maskAllB16);
        } else if constexpr (IsSameType<T, half>::value) {
            // 对于T为fp16，先把x和invScale转成bf16，再计算x*invScale后转为float4对应类型

            // 1. 将x转换为bf16
            Cast<bfloat16_t, T, traitFP16ToBF16>(x0BF16, x0, maskAllB16);
            Cast<bfloat16_t, T, traitFP16ToBF16>(x1BF16, x1, maskAllB16);

            // 2.计算x*invScale
            Mul(x0BF16, x0BF16, (RegTensor<bfloat16_t> &)invScale, maskAllB16);
            Mul(x1BF16, x1BF16, (RegTensor<bfloat16_t> &)invScale, maskAllB16);
            Interleave(x0BF16, x1BF16, x0BF16, x1BF16);

            // 3.将bf16的结果转换为fp4
            Cast<U, bfloat16_t, traitB16ToB4>(xQuant0, x0BF16, maskAllB16);
            Cast<U, bfloat16_t, traitB16ToB4>(xQuant1, x1BF16, maskAllB16);

        } else {
          // 对于T为bf16，直接计算x*invScale，将结果转换为fp4
          // 1.乘以invScale
          Mul(x0, x0, (RegTensor<T> &)invScale, maskAllB16);
          Mul(x1, x1, (RegTensor<T> &)invScale, maskAllB16);
          Interleave(x0BF16, x1BF16, x0BF16, x1BF16);

          // 3.将bf16的结果转换为fp4
          Cast<U, bfloat16_t, traitB16ToB4>(xQuant0, x0BF16, maskAllB16);
          Cast<U, bfloat16_t, traitB16ToB4>(xQuant1, x1BF16, maskAllB16);

        }

        // 拷出xQuant到xOutAddr。OUT_ELE_NUM_ONE_BLK=64，即一个VL下有64个float32->fp8的元素，一次拷出64个元素
        DataCopy<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
            xQuantOutAddr, (RegTensor<int8_t> &)xQuant0, OUT_ELE_NUM_ONE_BLK, maskXQuant);
        DataCopy<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
            xQuantOutAddr, (RegTensor<int8_t> &)xQuant1, OUT_ELE_NUM_ONE_BLK, maskXQuant);
    }
}

template <typename T, typename U>
class MoeGatherOutMxfp4Quant {
public:
    __aicore__ inline MoeGatherOutMxfp4Quant(){};
    __aicore__ inline void Init(GM_ADDR xAddr, GM_ADDR unused_ScaleAddr, GM_ADDR expandedRowIdxAddr,
                                GM_ADDR expandedXAddr, GM_ADDR expandedScaleAddr, GM_ADDR sortedExpertIdxAddr,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitKernelTiling(GM_ADDR sortedExpertIdxAddr, const MoeInitRoutingV3Arch35TilingData *tilingData);
    __aicore__ inline void CopyInExpandedExpertIdx(int64_t progress);
    __aicore__ inline void CopyExpandedXandMXQuant(int64_t progress);
    __aicore__ inline void CopyIn(int64_t srcIdx, int64_t colIdx, int64_t loopCols);
    __aicore__ inline void Compute(uint32_t xElemNum, uint32_t scaleElemNum, uint32_t validScaleElemNum);
    __aicore__ inline void CopyOut(int64_t dstIdx, int64_t colIdx, int64_t loopCols, int64_t loopScaleCols);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECIN, 1> sortedRowIdxInQueue_;
    TQue<QuePosition::VECOUT, 1> xQuantOutQueue_;
    TQue<QuePosition::VECOUT, 1> mxScaleOutQueue_;
    TBuf<QuePosition::VECCALC> maxExpBuffer_;
    TBuf<QuePosition::VECCALC> invScaleBuffer_;

    GlobalTensor<T> xInGm_;
    GlobalTensor<uint8_t> expandedXOutGm_;
    GlobalTensor<uint8_t> expandedScaleOutGm_;
    GlobalTensor<int32_t> sortedRowIdxGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    const MoeV3Arch35GatherOutComputeTilingData *gatherOutTilingData_;

    int64_t needCoreNum_;
    int64_t blockIdx_;
    int64_t cols_;
    int64_t validScaleCols_; // 一个token实际有意义的scale有多少个元素（列）
    int64_t scaleCols_; // 一个token的scale有多少个元素（列），即CeilAlign(CeliDiv(h,32),2)，在actualScaleCols_为奇数时，scaleCols_=validScaleCols_+1
    int64_t n_;
    int64_t k_;
    int64_t perCoreRow_;
    int64_t currentLoopRows_;
    int64_t coreRows_;
    int64_t perLoopRows_;
    int64_t lastLoopRows_;
    int64_t rowLoops_;
    int64_t perLoopCols_;
    int64_t lastLoopCols_;
    int64_t colLoops_;
    int64_t perLoopScaleCols_;
    int64_t lastLoopValidScaleCols_;
    int64_t lastLoopScaleCols_;
    int64_t indicesOffset_;
    int64_t rowIdxType_ = 0;

    // 会赋值为计算maxExp时，maxExp对应不同的目标fp8类型U的下限
    uint16_t lowerBoundOfB16MaxExp_ = 0;

    // 一个RegTensor的长度Bytes
    const uint32_t vRegSize_ = Ops::Base::GetVRegSize();
    // ub的一个块大小，通常是32B
    const uint32_t ubBlockSize_ = Ops::Base::GetUbBlockSize();
    // 一个RegTensor能塞下多少T(B16)类型的元素
    const uint32_t vlForB16_ = vRegSize_ / sizeof(T);
    // 一个RegTensor有几个UbBlock，即ReduceWithBlock后的元素个数（256B/32B=8）
    // 这里如果ubBlockSize_获取的值为0，应该让kernel挂掉
    const uint32_t numUbBlocksPerVReg_ = vRegSize_ / ubBlockSize_; // 8个元素
    // 每个ubBlock对应多少T类型的元素
    const uint32_t numElemPerUbBlock_ = ubBlockSize_ / sizeof(T);
};

template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp4Quant<T, U>::Init(GM_ADDR xAddr, GM_ADDR unused_ScaleAddr,
                                                          GM_ADDR sortedExpertIdxAddr, GM_ADDR expandedRowIdxAddr,
                                                          GM_ADDR expandedXAddr, GM_ADDR expandedScaleAddr,
                                                          const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
#endif

    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    InitKernelTiling(sortedExpertIdxAddr, tilingData);

    xInGm_.SetGlobalBuffer((__gm__ T *)xAddr);
    expandedXOutGm_.SetGlobalBuffer((__gm__ uint8_t *)expandedXAddr);
    if (rowIdxType_ == SCATTER) {
        sortedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdxAddr + blockIdx_ * perCoreRow_,
                                        Align(perCoreRow_, sizeof(int32_t)));
    } else {
        sortedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)sortedExpertIdxAddr + Align(n_ * k_, sizeof(int32_t)) +
                                            blockIdx_ * perCoreRow_,
                                        Align(perCoreRow_, sizeof(int32_t)));
    }
    expandedScaleOutGm_.SetGlobalBuffer((__gm__ uint8_t *)expandedScaleAddr);

    // perrows * 2 * 2 * 4 expandRowIdx + sortedExpertId
    pipe_->InitBuffer(sortedRowIdxInQueue_, 1, AlignBytes(perLoopRows_, sizeof(int32_t)));
    pipe_->InitBuffer(xInQueue_, 1, AlignBytes(perLoopCols_, sizeof(T)));
    pipe_->InitBuffer(xQuantOutQueue_, 1, AlignBytes(perLoopCols_ / 4, sizeof(int8_t)) * 4);
    pipe_->InitBuffer(mxScaleOutQueue_, 1, AlignBytes(perLoopScaleCols_, sizeof(int8_t)));
    pipe_->InitBuffer(maxExpBuffer_, AlignBytes(perLoopScaleCols_, sizeof(T)));
    pipe_->InitBuffer(invScaleBuffer_, AlignBytes(perLoopScaleCols_, sizeof(T)));

    if constexpr (IsSameType<U, fp4x2_e2m1_t>::value) {
        lowerBoundOfB16MaxExp_ = LOWER_BOUND_OF_MAX_EXP_FOR_E2M1;
    } else if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
        lowerBoundOfB16MaxExp_ = LOWER_BOUND_OF_MAX_EXP_FOR_E4M3;
    } else {
        lowerBoundOfB16MaxExp_ = LOWER_BOUND_OF_MAX_EXP_FOR_E5M2;
    }
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp4Quant<T, U>::InitKernelTiling(GM_ADDR sortedExpertIdxAddr, const MoeInitRoutingV3Arch35TilingData *tilingData)
{
    gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
    cols_ = tilingData->cols;
    validScaleCols_ = Ops::Base::CeilDiv<int64_t>(cols_, MX_BLOCK_SIZE);
    scaleCols_ = Ops::Base::CeilAlign<int64_t>(validScaleCols_, 2); // CeilDiv(h, 32)后再向上到2的倍数（偶数）
    n_ = tilingData->n;
    k_ = tilingData->k;
    rowIdxType_ = tilingData->rowIdxType;

    // core split
    int64_t actualExpertNum_ = tilingData->actualExpertNum;
    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)sortedExpertIdxAddr + Align(n_ * k_, sizeof(int32_t)) * 2 +
                                            Align(actualExpertNum_, sizeof(int32_t)),
                                        actualExpertNum_);
    int64_t expertTotalCount_ = expertTotalCountGm_.GetValue(0);
    perCoreRow_ = Ceil(expertTotalCount_, tilingData->coreNum);
    needCoreNum_ = Ceil(expertTotalCount_, perCoreRow_);
    int64_t lastCoreIndicesElements = expertTotalCount_ - (needCoreNum_ - 1) * perCoreRow_;

    // inner core split
    int64_t originPerLoopElements;
    if (blockIdx_ == needCoreNum_ - 1) {
        coreRows_ = lastCoreIndicesElements;
        originPerLoopElements = gatherOutTilingData_->lastCorePerLoopIndicesElements;
    } else {
        coreRows_ = perCoreRow_;
        originPerLoopElements = gatherOutTilingData_->perCorePerLoopIndicesElements;
    }
    perLoopRows_ = Min(coreRows_, originPerLoopElements);
    rowLoops_ = Ceil(coreRows_, perLoopRows_);
    lastLoopRows_ = coreRows_ - (rowLoops_ - 1) * perLoopRows_;

    // cols split
    perLoopCols_ = gatherOutTilingData_->perLoopCols;
    lastLoopCols_ = gatherOutTilingData_->lastLoopCols;
    colLoops_ = gatherOutTilingData_->colsLoops;
    perLoopScaleCols_ = perLoopCols_ / MX_BLOCK_SIZE; // perLoopCols_在tiling侧计算，已经对齐到32的整数倍了
    lastLoopValidScaleCols_ = validScaleCols_ - (colLoops_ - 1) * perLoopScaleCols_;
    lastLoopScaleCols_ = scaleCols_ - (colLoops_ - 1) * perLoopScaleCols_;
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp4Quant<T, U>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        currentLoopRows_ = perLoopRows_;
        for (int64_t loop = 0; loop < rowLoops_ - 1; loop++) {
            CopyInExpandedExpertIdx(loop);
            CopyExpandedXandMXQuant(loop);
        }

        currentLoopRows_ = lastLoopRows_;
        CopyInExpandedExpertIdx(rowLoops_ - 1);
        CopyExpandedXandMXQuant(rowLoops_ - 1);
    }
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp4Quant<T, U>::CopyInExpandedExpertIdx(int64_t progress)
{
    indicesOffset_ = progress * perLoopRows_;
    LocalTensor<int32_t> indicesLocal = sortedRowIdxInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(currentLoopRows_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, sortedRowIdxGm_[indicesOffset_], dataCopyParams, dataCopyPadParams);
    sortedRowIdxInQueue_.EnQue<int32_t>(indicesLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp4Quant<T, U>::CopyExpandedXandMXQuant(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = sortedRowIdxInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    for (int64_t index = 0; index < currentLoopRows_; index++) {
        // 对加载进ub的indicesLocal按行循环
        int32_t srcIdx = indicesLocal.GetValue(index);
        int64_t dstIdx = perCoreRow_ * blockIdx_ + perLoopRows_ * progress + index;
        for (int64_t j = 0; j < colLoops_; j++) {
            // 每行切分成cols，按cols读入-计算量化-拷出
            int64_t loopCols = (j == colLoops_ - 1) ? lastLoopCols_ : perLoopCols_;
            uint32_t loopScaleCols = (j == colLoops_ - 1) ? lastLoopScaleCols_ : perLoopScaleCols_;
            uint32_t loopValidScaleCols = (j == colLoops_ - 1) ? lastLoopValidScaleCols_ : perLoopScaleCols_;
            CopyIn(srcIdx / k_, j, loopCols);
            Compute(loopCols, loopScaleCols, loopValidScaleCols);
            CopyOut(dstIdx, j, loopCols, loopScaleCols);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp4Quant<T, U>::CopyIn(int64_t srcIdx, int64_t colIdx, int64_t loopCols)
{
    LocalTensor<T> inLocal = xInQueue_.AllocTensor<T>();
    DataCopyExtParams copyInParam = {1, static_cast<uint32_t>(loopCols * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    int64_t loopColsTail = loopCols % MX_BLOCK_SIZE;
    if (loopColsTail != 0) {
        padParams.isPad = true;
        if (loopColsTail > numElemPerUbBlock_) {
            padParams.rightPadding = numElemPerUbBlock_ - (loopColsTail - numElemPerUbBlock_);
        } else {
            padParams.rightPadding = numElemPerUbBlock_;
        }
    }
    DataCopyPad(inLocal, xInGm_[srcIdx * cols_ + colIdx * perLoopCols_], copyInParam, padParams);
    xInQueue_.EnQue(inLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp4Quant<T, U>::Compute(uint32_t xElemNum, uint32_t scaleElemNum, uint32_t validScaleElemNum)
{
    // deque input
    LocalTensor<T> xLocal = xInQueue_.DeQue<T>();
    auto xLocalAddr = reinterpret_cast<__ubuf__ T *>(xLocal.GetPhyAddr());
    // alloc outputs
    LocalTensor<int8_t> xQuantLocal = xQuantOutQueue_.AllocTensor<int8_t>();
    auto xQuantLocalAddr = reinterpret_cast<__ubuf__ int8_t *>(xQuantLocal.GetPhyAddr());
    LocalTensor<uint16_t> mxScaleLocal = mxScaleOutQueue_.AllocTensor<uint16_t>();
    auto mxScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(mxScaleLocal.GetPhyAddr());
    // get tmp buffers
    LocalTensor<uint16_t> maxExpLocal = maxExpBuffer_.Get<uint16_t>();
    auto maxExpLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(maxExpLocal.GetPhyAddr());
    LocalTensor<uint16_t> invScaleLocal = invScaleBuffer_.Get<uint16_t>();
    auto invScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(invScaleLocal.GetPhyAddr());

    // 按每次双搬vlForHalfNumber_*2来计算loopNum
    uint16_t vfLoopNumForX = (xElemNum + vlForB16_ * 2 - 1) / (vlForB16_ * 2);
    uint16_t vfLoopNumForScale = (scaleElemNum + vlForB16_ - 1) / vlForB16_;

    VF_CALL<vfMxfp4ComputeMaxExp<T, U>>(xLocalAddr, maxExpLocalAddr, xElemNum, vfLoopNumForX, vlForB16_,
                                   numUbBlocksPerVReg_);
    VF_CALL<vfMxfp4ComputeScale<T, U>>(maxExpLocalAddr, mxScaleLocalAddr, invScaleLocalAddr, scaleElemNum, validScaleElemNum, vfLoopNumForScale,
                                  vlForB16_, lowerBoundOfB16MaxExp_);
    VF_CALL<vfMxfp4ComputeData<T, U>>(xLocalAddr, invScaleLocalAddr, xQuantLocalAddr, xElemNum, vfLoopNumForX, vlForB16_,
                                 numUbBlocksPerVReg_);

    // free input
    xInQueue_.FreeTensor(xLocal);
    // enque outputs
    xQuantOutQueue_.EnQue(xQuantLocal);
    mxScaleOutQueue_.EnQue(mxScaleLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp4Quant<T, U>::CopyOut(int64_t dstIdx, int64_t colIdx, int64_t loopCols,
                                                             int64_t loopScaleCols)
{
    LocalTensor<uint8_t> mxScaleLocal = mxScaleOutQueue_.DeQue<uint8_t>();
    LocalTensor<uint8_t> outLocal = xQuantOutQueue_.DeQue<uint8_t>();

    DataCopyExtParams copyOutParams = {1, static_cast<uint32_t>(loopCols * sizeof(uint8_t) / 2), 0, 0, 0};
    DataCopyPad<uint8_t>(expandedXOutGm_[(dstIdx * cols_ + colIdx * perLoopCols_) / 2], outLocal, copyOutParams);

    DataCopyExtParams copyScaleParams = {1, static_cast<uint32_t>(loopScaleCols * sizeof(uint8_t)), 0, 0, 0};
    DataCopyPad<uint8_t>(expandedScaleOutGm_[dstIdx * scaleCols_ + colIdx * perLoopScaleCols_], mxScaleLocal,
                         copyScaleParams);

    xQuantOutQueue_.FreeTensor(outLocal);
    mxScaleOutQueue_.FreeTensor(mxScaleLocal);
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_GATHER_MXFP4_QUANT_H_REGBASE