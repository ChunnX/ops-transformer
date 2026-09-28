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
 * \file weight_quant_grouped_matmul_cube_compute_tools.h
 * \brief
 */
#pragma once

#include "weight_quant_tool_asw.h"
using WeightQuantTools::B32_IN_B16_NUM;
using WeightQuantTools::B8_IN_B16_NUM;
using WeightQuantTools::MX_GROUP_SIZE_64;
using WeightQuantTools::MX_SCALE_DIVISOR_SIZE;

namespace WeightQuantGroupedMatmul::Arch35 {

static constexpr uint64_t C0_SIZE = 32UL;            // 单位Byte
static constexpr uint64_t BLOCK_SIZE = 32UL;         // 单位Byte
static constexpr uint64_t L0AB_OPERATE_UNIT = 512UL; // L0上搬运单位512Byte
static constexpr int32_t UNIT_FLAG_UPDATE = 3;       // 检查并更新标记位
static constexpr int32_t UNIT_FLAG_CHECK_ONLY = 2;   // 只检查标记位

struct L0CopyAndCalcParams {
    uint64_t mL0Size;      // M方向当前在L0上实际计算的大小
    uint64_t mL1Size;      // M方向当前在L1上实际载入的大小
    uint64_t kL0Size;      // K方向当前在L0上实际计算的大小
    uint64_t kL1Size;      // K方向当前在L1上实际载入的大小
    uint64_t nL0Size;      // N方向当前在L0上实际计算的大小
    uint64_t nL1Size;      // N方向当前在L1上实际载入的大小
    uint64_t aScaleL1GroupNum; // aScale的K方向在L1上实际载入的分组数
    uint64_t bScaleL1GroupNum; // bScale的K方向在L1上实际载入的分组数
    bool isFirstKLoop;     // 是否是第一次K方向的循环，用于控制L0C的初始值
    bool isLastKLoop;      // 是否是最后一次K方向的循环，用于设置unitFlag
};

struct FixL0CToDstParams {
    uint64_t mL0Size;         // M方向当前在L0上实际计算的大小
    uint64_t nL0Size;         // N方向当前在L0上实际计算的大小
    uint64_t outNSize;        // 输出N方向的总大小,用于dst_stride
    uint64_t splitNSize = 0;  // fixp输出多个矩阵时,N方向切分的大小
    uint64_t dstNdStride = 0; // fixp输出多个矩阵时,目的相邻DN矩阵起始地址间的偏移
};

enum QuantPreMode {
    NO_QUANT = 0,
    VEC_QUANT = 1,
    SCALAR_QUANT = 2
};

/**
 * @brief 将A矩阵数据从L1加载到AL0
 *
 * @param aL0Tensor L0A tensor
 * @param aL1Tensor L1A Tensor， 要求根据M,K方向的循环次数，取正确的L1起始地址后传入
 * @param l0CopyAndCalcParams 搬运和计算参数
 */
template <typename DstType, typename SrcType>
__aicore__ inline void LoadAL1ToL0(const LocalTensor<DstType> &aL0Tensor, const LocalTensor<SrcType> &aL1Tensor,
                                   const L0CopyAndCalcParams &l0CopyAndCalcParams)
{
    uint64_t mL0AlignSize =
        Ops::Base::CeilAlign(l0CopyAndCalcParams.mL0Size, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
    uint64_t mL1AlignSize =
        Ops::Base::CeilAlign(l0CopyAndCalcParams.mL1Size, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
    AscendC::LoadData2DParamsV2 aL0Load2dParams;
    aL0Load2dParams.mStartPosition = 0;
    aL0Load2dParams.kStartPosition = 0;
    aL0Load2dParams.mStep = mL0AlignSize / static_cast<uint64_t>(AscendC::BLOCK_CUBE);
    aL0Load2dParams.kStep = Ops::Base::CeilDiv(l0CopyAndCalcParams.kL0Size * sizeof(SrcType), C0_SIZE);
    aL0Load2dParams.srcStride = Ops::Base::CeilDiv(mL1AlignSize * C0_SIZE, L0AB_OPERATE_UNIT);
    aL0Load2dParams.dstStride = Ops::Base::CeilDiv(mL0AlignSize * C0_SIZE, L0AB_OPERATE_UNIT);
    AscendC::LoadData(aL0Tensor, aL1Tensor, aL0Load2dParams);
}

/**
 * @brief 将A矩阵和scaleA数据从L1加载到AL0
 *
 * @param aL0Tensor L0A tensor
 * @param aL1Tensor L1A Tensor 要求根据M,K方向的循环次数，取正确的L1起始地址后传入
 * @param scaleL1Tensor L1scale Tensor 要求根据M,K方向的循环次数，取正确的L1起始地址后传入
 * @param l0CopyAndCalcParams 搬运和计算参数
 */
template <typename DstType, typename SrcType, typename XScaleType>
__aicore__ inline void LoadAAndScaleL1ToL0(const LocalTensor<DstType> &aL0Tensor, const LocalTensor<SrcType> &aL1Tensor,
                                           const LocalTensor<XScaleType> &scaleL1Tensor,
                                           const L0CopyAndCalcParams &l0CopyAndCalcParams)
{
    uint64_t mL0AlignSize =
        Ops::Base::CeilAlign(l0CopyAndCalcParams.mL0Size, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
    uint64_t mL1AlignSize =
        Ops::Base::CeilAlign(l0CopyAndCalcParams.mL1Size, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
    AscendC::LoadData2DParamsV2 aL0Load2dParams;
    aL0Load2dParams.mStartPosition = 0;
    aL0Load2dParams.kStartPosition = 0;
    aL0Load2dParams.mStep = mL0AlignSize / static_cast<uint64_t>(AscendC::BLOCK_CUBE);
    aL0Load2dParams.kStep = Ops::Base::CeilDiv(l0CopyAndCalcParams.kL0Size * sizeof(SrcType), C0_SIZE);
    aL0Load2dParams.srcStride = Ops::Base::CeilDiv(mL1AlignSize * C0_SIZE, L0AB_OPERATE_UNIT);
    aL0Load2dParams.dstStride = Ops::Base::CeilDiv(mL0AlignSize * C0_SIZE, L0AB_OPERATE_UNIT);

    AscendC::LoadData2DMxParams aL0Load2dMx;
    aL0Load2dMx.xStartPosition = 0;
    aL0Load2dMx.yStartPosition = 0;
    aL0Load2dMx.xStep = mL0AlignSize / static_cast<uint64_t>(AscendC::BLOCK_CUBE);
    aL0Load2dMx.yStep = Ops::Base::CeilDiv(l0CopyAndCalcParams.kL0Size,
                                           64UL); // 64: 表示K方向跨过了几个32B分形,计算过程为k/32/2

    aL0Load2dMx.srcStride = Ops::Base::CeilDiv(l0CopyAndCalcParams.aScaleL1GroupNum, 2UL);
    aL0Load2dMx.dstStride = Ops::Base::CeilDiv(l0CopyAndCalcParams.kL0Size, 64UL);
    AscendC::LoadData(aL0Tensor, aL1Tensor, scaleL1Tensor, aL0Load2dParams, aL0Load2dMx);
}

/**
 * @brief 将B矩阵数据从L1加载到BL0
 *
 * @param bL0Tensor L0B tensor
 * @param bL1Tensor L1B Tensor 要求根据M,K方向的循环次数，取正确的L1起始地址后传入
 * @param l0CopyAndCalcParams 搬运和计算参数
 */
template <typename DstType, typename SrcType, bool bTrans = true>
__aicore__ inline void LoadBL1ToL0(const LocalTensor<DstType> &bL0Tensor, const LocalTensor<SrcType> &bL1Tensor,
                                   const L0CopyAndCalcParams &l0CopyAndCalcParams)
{
    uint64_t nL0AlignSize =
        Ops::Base::CeilAlign(l0CopyAndCalcParams.nL0Size, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
    uint64_t nL1AlignSize =
        Ops::Base::CeilAlign(l0CopyAndCalcParams.nL1Size, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
    uint64_t kL1AlignSize =
        Ops::Base::CeilAlign(l0CopyAndCalcParams.kL1Size, static_cast<uint64_t>(AscendC::BLOCK_CUBE));

    AscendC::LoadData2DParamsV2 bL0Load2dParams;
    bL0Load2dParams.mStartPosition = 0;
    bL0Load2dParams.kStartPosition = 0;
    if constexpr (bTrans) {
        bL0Load2dParams.mStep = nL0AlignSize / static_cast<uint64_t>(AscendC::BLOCK_CUBE);
        if constexpr (AscendC::IsSupportB4<SrcType>()) {
            bL0Load2dParams.kStep =
                Ops::Base::CeilDiv(l0CopyAndCalcParams.kL0Size, 64UL); // 64: 单位32B, B4类型要多除一个2
        } else {
            bL0Load2dParams.kStep = Ops::Base::CeilDiv(l0CopyAndCalcParams.kL0Size * sizeof(SrcType), C0_SIZE);
        }
        bL0Load2dParams.srcStride = Ops::Base::CeilDiv(nL1AlignSize * C0_SIZE, L0AB_OPERATE_UNIT);
        bL0Load2dParams.dstStride = Ops::Base::CeilDiv(nL0AlignSize * C0_SIZE, L0AB_OPERATE_UNIT);
    } else {
        bL0Load2dParams.mStep =
            Ops::Base::CeilDiv(l0CopyAndCalcParams.kL0Size, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        if constexpr (AscendC::IsSupportB4<SrcType>()) {
            bL0Load2dParams.kStep = Ops::Base::CeilDiv(nL0AlignSize, 64UL); // 64: 单位32B, B4类型要多除一个2
        } else {
            bL0Load2dParams.kStep = Ops::Base::CeilDiv(nL0AlignSize * sizeof(SrcType), C0_SIZE);
        }
        bL0Load2dParams.srcStride = Ops::Base::CeilDiv(kL1AlignSize * C0_SIZE, L0AB_OPERATE_UNIT);
        bL0Load2dParams.dstStride = Ops::Base::CeilDiv(nL0AlignSize * C0_SIZE, L0AB_OPERATE_UNIT);
        bL0Load2dParams.ifTranspose = true;
    }
    AscendC::LoadData(bL0Tensor, bL1Tensor, bL0Load2dParams);
}


/**
 * @brief 将B矩阵数据从L1加载到BL0
 *
 * @param bL0Tensor L0B tensor
 * @param bL1Tensor L1B Tensor 要求根据M,K方向的循环次数，取正确的L1起始地址后传入
 * @param scaleL1Tensor L1scale Tensor 要求根据M,K方向的循环次数，取正确的L1起始地址后传入
 * @param l0CopyAndCalcParams 搬运和计算参数
 */
template <typename DstType, typename SrcType, typename WScaleType>
__aicore__ inline void LoadBAndScaleL1ToL0(const LocalTensor<DstType> &bL0Tensor, const LocalTensor<SrcType> &bL1Tensor,
                                           const LocalTensor<WScaleType> &scaleL1Tensor,
                                           const L0CopyAndCalcParams &l0CopyAndCalcParams)
{
    uint64_t nL0AlignSize =
        Ops::Base::CeilAlign(l0CopyAndCalcParams.nL0Size, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
    uint64_t nL1AlignSize =
        Ops::Base::CeilAlign(l0CopyAndCalcParams.nL1Size, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
    AscendC::LoadData2DParamsV2 bL0Load2dParams;
    bL0Load2dParams.mStartPosition = 0;
    bL0Load2dParams.kStartPosition = 0;
    bL0Load2dParams.mStep = nL0AlignSize / static_cast<uint64_t>(AscendC::BLOCK_CUBE);
    if constexpr (AscendC::IsSupportB4<SrcType>()) {
        bL0Load2dParams.kStep = Ops::Base::CeilDiv(l0CopyAndCalcParams.kL0Size, 64UL); // 64: 单位32B, B4类型要多除一个2
    } else {
        bL0Load2dParams.kStep = Ops::Base::CeilDiv(l0CopyAndCalcParams.kL0Size, C0_SIZE);
    }
    bL0Load2dParams.srcStride = Ops::Base::CeilDiv(nL1AlignSize * C0_SIZE, L0AB_OPERATE_UNIT);
    bL0Load2dParams.dstStride = Ops::Base::CeilDiv(nL0AlignSize * C0_SIZE, L0AB_OPERATE_UNIT);

    AscendC::LoadData2DMxParams bL0Load2dMx;
    bL0Load2dMx.xStartPosition = 0;
    bL0Load2dMx.xStep = nL0AlignSize / static_cast<uint64_t>(AscendC::BLOCK_CUBE);
    bL0Load2dMx.yStartPosition = 0;
    // yStep的含义是源矩阵Y轴(横向)搬运的32B的个数，不是Y轴本身的长度，而是包含了分形
    // mxfp4的scale是e8m0，ystep计算方式为:
    // kL0/mx_gsize / 2 * (2 * 16) * sizeof(e8m0) / 32[B] = kL0/64，其中mx_gsize固定为32，
    // 除以2是表示2个e8m0构成一个分形， 2 * 16 * sizeof(e8m0)表示一个分形的大小，32B是单位
    // hif4的scale是b32类型，到L0Mx会拆分成（e6m2 + e1_8 + e1_16），凑成2个16 * 2的分形，ystep的计算方式为：
    // ceil(kL0/hif4_gsize) * 2 * (2 * 16) * sizeof(b8) / 32[B] = ceil(kL0/64) * 2，其中hif4_gsize固定为64
    // srcStride和dstStride计算过程类似
    if constexpr (AscendC::SupportType<DstType, hifloat4x2_t>()) {
        bL0Load2dMx.yStep = Ops::Base::CeilDiv(l0CopyAndCalcParams.kL0Size, MX_GROUP_SIZE_64) * B32_IN_B16_NUM;
        bL0Load2dMx.srcStride = l0CopyAndCalcParams.bScaleL1GroupNum * B32_IN_B16_NUM;
        bL0Load2dMx.dstStride = Ops::Base::CeilDiv(l0CopyAndCalcParams.kL0Size, MX_GROUP_SIZE_64) * B32_IN_B16_NUM;
    } else {
        bL0Load2dMx.yStep = Ops::Base::CeilDiv(l0CopyAndCalcParams.kL0Size, MX_SCALE_DIVISOR_SIZE);
        bL0Load2dMx.srcStride = Ops::Base::CeilDiv(l0CopyAndCalcParams.bScaleL1GroupNum, B8_IN_B16_NUM);
        bL0Load2dMx.dstStride = Ops::Base::CeilDiv(l0CopyAndCalcParams.kL0Size, MX_SCALE_DIVISOR_SIZE);
    }
    AscendC::LoadData(bL0Tensor, bL1Tensor, scaleL1Tensor, bL0Load2dParams, bL0Load2dMx);
}

/**
 * @brief 将Bias数据从L1加载到bias table
 *
 * @param biasBtTensor biasBt tensor
 * @param biasL1Tensor BiasL1 Tensor 要求根据循环次数，取正确的L1起始地址后传入
 * @param nSize 搬运量
 */
template <typename DstType, typename SrcType>
__aicore__ inline void CopyBiasL1ToBT(const LocalTensor<DstType> &biasBtTensor,
                                      const LocalTensor<SrcType> &biasL1Tensor, uint64_t nSize)
{
    uint16_t lenBurst = Ops::Base::CeilDiv(nSize * sizeof(SrcType), BLOCK_SIZE);
    if constexpr (IsSameType<float, SrcType>::value) {
        lenBurst = Ops::Base::CeilAlign(lenBurst, static_cast<uint16_t>(2));
    }
    AscendC::DataCopy(biasBtTensor, biasL1Tensor, {1, lenBurst, 0, 0});
}

/**
 * @brief 不带Bias的Mmad计算
 *
 * @param cL0Tensor L0C tensor
 * @param aL0Tensor L0A tensor
 * @param bL0Tensor L0B tensor
 * @param l0CopyAndCalcParams 搬运和计算参数
 */
template <typename DstType, typename SrcAType, typename SrcBType>
__aicore__ inline void MmadCompute(const LocalTensor<DstType> &cL0Tensor, const LocalTensor<SrcAType> &aL0Tensor,
                                   const LocalTensor<SrcBType> &bL0Tensor,
                                   const L0CopyAndCalcParams &l0CopyAndCalcParams)
{
    AscendC::MmadParams mmadParams;
    mmadParams.m = l0CopyAndCalcParams.mL0Size;
    mmadParams.n = l0CopyAndCalcParams.nL0Size;
    mmadParams.k = l0CopyAndCalcParams.kL0Size;
    mmadParams.cmatrixInitVal = l0CopyAndCalcParams.isFirstKLoop;
    mmadParams.disableGemv = true; // 关闭gemv, 避免特殊处理baseM=1的场景
    // 只在最后一次K循环设为3，自动更新标记位
    mmadParams.unitFlag = l0CopyAndCalcParams.isLastKLoop ? UNIT_FLAG_UPDATE : UNIT_FLAG_CHECK_ONLY;
    AscendC::Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
}

/**
 * @brief 带Bias的Mmad计算，应只在带bias的场景下，第一次k方向循环调用，其他情况调用不带bias的接口
 *
 * @param cL0Tensor L0C tensor
 * @param aL0Tensor L0A tensor
 * @param bL0Tensor L0B tensor
 * @param biasBtTensor bias tensor
 * @param l0CopyAndCalcParams 搬运和计算参数
 */
template <typename DstType, typename SrcAType, typename SrcBType, typename BiasType>
__aicore__ inline void MmadCompute(const LocalTensor<DstType> &cL0Tensor, const LocalTensor<SrcAType> &aL0Tensor,
                                   const LocalTensor<SrcBType> &bL0Tensor, const LocalTensor<BiasType> &biasBtTensor,
                                   const L0CopyAndCalcParams &l0CopyAndCalcParams)
{
    AscendC::MmadParams mmadParams;
    mmadParams.m = l0CopyAndCalcParams.mL0Size;
    mmadParams.n = l0CopyAndCalcParams.nL0Size;
    mmadParams.k = l0CopyAndCalcParams.kL0Size;
    mmadParams.cmatrixInitVal = false; // 带bias的接口，只要init置为false, 不需配置cmatrixSource，会自动选择根据bias选择
    mmadParams.disableGemv = true;
    // 只在最后一次K循环设为3，自动更新标记位
    mmadParams.unitFlag = l0CopyAndCalcParams.isLastKLoop ? UNIT_FLAG_UPDATE : UNIT_FLAG_CHECK_ONLY;
    AscendC::Mmad(cL0Tensor, aL0Tensor, bL0Tensor, biasBtTensor, mmadParams);
}

/**
 * @brief 设置NO_QUANT preQuant模式
 *
 * @param fixParams fixpipe接口参数
 */
template <typename DstType, typename SrcType>
__aicore__ inline void SetNoQuantPre(AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> &fixParams)
{
    if constexpr (IsSameType<DstType, SrcType>::value) {
        fixParams.quantPre = QuantMode_t::NoQuant;
    } else if constexpr (IsSameType<DstType, bfloat16_t>::value && IsSameType<SrcType, float>::value) {
        fixParams.quantPre = QuantMode_t::F322BF16;
    } else if constexpr (IsSameType<DstType, half>::value && IsSameType<SrcType, float>::value) {
        fixParams.quantPre = QuantMode_t::F322F16;
    }
}

/**
 * @brief 设置VEC_QUANT preQuant模式
 *
 * @param fixParams fixpipe接口参数
 */
template <typename DstType, typename SrcType>
__aicore__ inline void SetVecQuantPre(AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> &fixParams)
{
    if constexpr (IsSameType<DstType, SrcType>::value && IsSameType<SrcType, float>::value) {
        fixParams.quantPre = QuantMode_t::VQF322F32_PRE;
    } else if constexpr (IsSameType<DstType, bfloat16_t>::value && IsSameType<SrcType, float>::value) {
        fixParams.quantPre = QuantMode_t::VQF322BF16_PRE;
    } else if constexpr (IsSameType<DstType, half>::value && IsSameType<SrcType, float>::value) {
        fixParams.quantPre = QuantMode_t::VQF322F16_PRE;
    } else if constexpr (IsSameType<DstType, half>::value && IsSameType<SrcType, int32_t>::value) {
        fixParams.quantPre = QuantMode_t::VDEQF16;
    } else if constexpr (IsSameType<DstType, bfloat16_t>::value && IsSameType<SrcType, int32_t>::value) {
        fixParams.quantPre = QuantMode_t::VQS322BF16_PRE;
    }
}

/**
 * @brief 设置SCALAR_QUANT preQuant模式
 *
 * @param fixParams fixpipe接口参数
 */
template <typename DstType, typename SrcType>
__aicore__ inline void SetScalarQuantPre(AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> &fixParams)
{
    if constexpr (IsSameType<DstType, SrcType>::value && IsSameType<SrcType, float>::value) {
        fixParams.quantPre = QuantMode_t::QF322F32_PRE;
    } else if constexpr (IsSameType<DstType, bfloat16_t>::value && IsSameType<SrcType, float>::value) {
        fixParams.quantPre = QuantMode_t::QF322BF16_PRE;
    } else if constexpr (IsSameType<DstType, half>::value && IsSameType<SrcType, float>::value) {
        fixParams.quantPre = QuantMode_t::QF322F16_PRE;
    } else if constexpr (IsSameType<DstType, half>::value && IsSameType<SrcType, int32_t>::value) {
        fixParams.quantPre = QuantMode_t::DEQF16;
    } else if constexpr (IsSameType<DstType, bfloat16_t>::value && IsSameType<SrcType, int32_t>::value) {
        fixParams.quantPre = QuantMode_t::QS322BF16_PRE;
    }
}

/**
 * @brief 设置preQuant模式
 *
 * @param fixParams fixpipe接口参数
 */
template <typename DstType, typename SrcType, QuantPreMode QuantMode>
__aicore__ inline void SetQuantPre(AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> &fixParams)
{
    if constexpr (QuantMode == QuantPreMode::NO_QUANT) {
        SetNoQuantPre<DstType, SrcType>(fixParams);
    } else if constexpr (QuantMode == QuantPreMode::VEC_QUANT) {
        SetVecQuantPre<DstType, SrcType>(fixParams);
    } else {
        SetScalarQuantPre<DstType, SrcType>(fixParams);
    }
}

/**
 * @brief 设置Fixpipe API的参数
 *
 * @param fixParams Fixpipe API的参数
 * @param fixL0CToDstParams 搬运参数
 * @tparam DstType 目的数据类型
 * @tparam SrcType L0C数据类型
 * @tparam OutSplitN 是否要在N方向切分，搬运成多个ND矩阵. 如果为true, 需要fixL0CToDstParams中指定splitNSize和dstNdStride
 */
template <typename DstType, typename SrcType, bool OutSplitN = false>
__aicore__ inline void FillFixParams(AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> &fixParams,
                                     const FixL0CToDstParams &fixL0CToDstParams)
{
    uint64_t mL0AlignSize = Ops::Base::CeilAlign(fixL0CToDstParams.mL0Size, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
    if constexpr (OutSplitN) {
        fixParams.nSize = fixL0CToDstParams.splitNSize;
        fixParams.params.ndNum = Ops::Base::CeilDiv(fixL0CToDstParams.nL0Size, fixL0CToDstParams.splitNSize);
        fixParams.params.srcNdStride =
            mL0AlignSize * fixL0CToDstParams.splitNSize /
            (static_cast<uint64_t>(AscendC::BLOCK_CUBE)); // 不同NZ矩阵起始地址之间的间隔,单位C0_SIZE[16]
        fixParams.params.dstNdStride = fixL0CToDstParams.dstNdStride;
    } else {
        fixParams.nSize = fixL0CToDstParams.nL0Size;
    }
    fixParams.mSize = fixL0CToDstParams.mL0Size;
    fixParams.srcStride = mL0AlignSize;
    fixParams.dstStride = fixL0CToDstParams.outNSize;
    fixParams.unitFlag = UNIT_FLAG_UPDATE;
}

/**
 * @brief 将L0C的数据输出到GM，不做量化
 *
 * @param outTensor 输出 GM Tensor
 * @param cL0Tensor L0C Tensor
 * @param fixL0CToDstParams 搬运参数
 */
template <typename DstType, typename SrcType>
__aicore__ inline void FixL0CToDst(const GlobalTensor<DstType> &outTensor, const LocalTensor<SrcType> &cL0Tensor,
                                   const FixL0CToDstParams &fixL0CToDstParams)
{
    AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixParams;
    FillFixParams<DstType, SrcType, false>(fixParams, fixL0CToDstParams);
    SetQuantPre<DstType, SrcType, QuantPreMode::NO_QUANT>(fixParams);
    AscendC::Fixpipe<DstType, SrcType, AscendC::CFG_ROW_MAJOR>(outTensor, cL0Tensor, fixParams);
}

/**
 * @brief 将L0C的数据输出到GM，做量化
 *
 * @param outTensor 输出 GM Tensor
 * @param cL0Tensor L0C Tensor
 * @param quantPreTensor fixpipe量化参数，仅支持perchannel
 * @param fixL0CToDstParams 搬运参数
 */
template <typename DstType, typename SrcType>
__aicore__ inline void FixL0CToDst(const GlobalTensor<DstType> &outTensor, const LocalTensor<SrcType> &cL0Tensor,
                                   const LocalTensor<uint64_t> &quantPreTensor,
                                   const FixL0CToDstParams &fixL0CToDstParams)
{
    AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixParams;
    FillFixParams<DstType, SrcType, false>(fixParams, fixL0CToDstParams);
    SetQuantPre<DstType, SrcType, QuantPreMode::VEC_QUANT>(fixParams);
    AscendC::Fixpipe<DstType, SrcType, AscendC::CFG_ROW_MAJOR>(outTensor, cL0Tensor, quantPreTensor, fixParams);
}

/**
 * @brief 将L0C的数据输出到UB，不做量化
 *
 * @param outTensor 输出 UB Tensor
 * @param cL0Tensor L0C Tensor
 * @param fixL0CToDstParams 搬运参数
 * @tparam DstType UB数据类型
 * @tparam SrcType L0C数据类型
 * @tparam OutSplitN 是否要在N方向切分，搬运成多个ND矩阵
 */

template <typename DstType, typename SrcType, bool OutSplitN = false>
__aicore__ inline void FixL0CToDst(const LocalTensor<DstType> &outTensor, const LocalTensor<SrcType> &cL0Tensor,
                                   const FixL0CToDstParams &fixL0CToDstParams)
{
    AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixParams;
    FillFixParams<DstType, SrcType, OutSplitN>(fixParams, fixL0CToDstParams);
    SetQuantPre<DstType, SrcType, QuantPreMode::NO_QUANT>(fixParams);
    AscendC::Fixpipe<DstType, SrcType, AscendC::Impl::CFG_ROW_MAJOR_UB>(outTensor, cL0Tensor, fixParams);
}

/**
 * @brief 将L0C的数据输出到UB，做量化
 *
 * @param outTensor 输出 UB Tensor
 * @param cL0Tensor L0C Tensor
 * @param quantPreTensor fixpipe量化参数，仅支持perchannel
 * @param fixL0CToDstParams 搬运参数
 * @tparam DstType UB数据类型
 * @tparam SrcType L0C数据类型
 * @tparam OutSplitN 是否要在N方向切分，搬运成多个ND矩阵
 */
template <typename DstType, typename SrcType, bool OutSplitN = false>
__aicore__ inline void FixL0CToDst(const LocalTensor<DstType> &outTensor, const LocalTensor<SrcType> &cL0Tensor,
                                   const LocalTensor<uint64_t> &quantPreTensor,
                                   const FixL0CToDstParams &fixL0CToDstParams)
{
    AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixParams;
    FillFixParams<DstType, SrcType, OutSplitN>(fixParams, fixL0CToDstParams);
    SetQuantPre<DstType, SrcType, QuantPreMode::VEC_QUANT>(fixParams);
    AscendC::Fixpipe<DstType, SrcType, AscendC::Impl::CFG_ROW_MAJOR_UB>(outTensor, cL0Tensor, quantPreTensor,
                                                                        fixParams);
}

} // namespace WeightQuantGroupedMatmul::Arch35