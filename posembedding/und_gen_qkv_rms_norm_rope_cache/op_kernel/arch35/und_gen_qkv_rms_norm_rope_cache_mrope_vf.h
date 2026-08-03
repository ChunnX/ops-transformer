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
 * \file und_gen_qkv_rms_norm_rope_cache_mrope_vf.h
 * \brief UndGenQkvRmsNormRopeCache 的 Reg 矢量（VF）计算段：RMSNorm + MRoPE 融合
 *
 * 设计要点（三轴 cos_sin 不落 UB，用 Gather 在寄存器里合并）：
 *   1) axisLut 做成 gather 索引 gatherIndex[D/2]，每核 Init 期建一次：
 *        gatherIndex[lane] = axis(lane) * D + lane
 *      其中 axis(lane) 的规则与竞品 _mrope / golden.py:mrope_axis_map 完全一致
 *      （见 BuildMropeGatherIndex，注意 mrope_section[0] 不参与判断）。
 *   2) CopyIn 已经把每 token 的三轴 cos_sin 原样搬成 UB 上连续的 [3, D] 窗口
 *      （行序 T/H/W），VF 里只需两条 Gather：
 *        cos = Gather(rawToken,            gatherIndex)   // rawToken[axis*D + lane]
 *        sin = Gather(rawToken + D/2,      gatherIndex)   // rawToken[axis*D + D/2 + lane]
 *      同一份索引复用于 cos 和 sin，靠基址 +D/2 区分——因为 cos_sin_cache 每行的
 *      布局就是 [cos(0:D/2) | sin(0:D/2)]。
 *   3) 合并结果只存在 vreg 里，不占 UB，也不需要 cosSinExpand / ropeTmp 这类中间 buffer。
 *
 * 约束：本文件假定 D/2 恰好等于一个 float VL（950 上 VL=256B → 64 个 float，即 D=128），
 *       所有 RegTensor 操作都按「半宽 = 一个满 VL」写死，没有分段循环也没有尾块 mask。
 *       该不变量由 host 侧的 static_assert 绑定，见 op_host/..._tiling.h 的 VL_FP32_LANES；
 *       放宽 headDim 前必须先在这里补分段循环与尾块 mask。
 */

#ifndef UND_GEN_QKV_RMS_NORM_ROPE_CACHE_MROPE_VF_H_
#define UND_GEN_QKV_RMS_NORM_ROPE_CACHE_MROPE_VF_H_

#include "kernel_operator.h"
// 复用的 kv_rms_norm_rope_cache/arch35/platform.h 没有 include guard，
// 本 header 与 ..._regbase.h 互不依赖顺序，所以两边共用下面这个哨兵避免重复定义
#ifndef UND_GEN_QKV_RMS_NORM_ROPE_CACHE_PLATFORM_INCLUDED
#define UND_GEN_QKV_RMS_NORM_ROPE_CACHE_PLATFORM_INCLUDED
#ifdef __CCE_UT_TEST__
#include "../../../kv_rms_norm_rope_cache/op_kernel/arch35/platform.h"
#include "../../../../third_party/opbase/include/op_common/op_kernel/platform_util.h"
#else
#include "op_kernel/platform_util.h"
#endif
#endif

namespace UndGenQkvRmsNormRopeCache {
namespace MicroAPI = AscendC::MicroAPI;

constexpr static MicroAPI::CastTrait CAST_B16_TO_B32 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

constexpr static MicroAPI::CastTrait CAST_FP32_TO_B16 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

/**
 * @brief 单个 head 的 "RMSNorm -> MRoPE 旋转 -> Cast bf16 写回"
 *
 * Q 和 K 的差别只在 gamma 与输出位置，计算本身完全一样，抽出来给两个 head
 * 循环共用。cos/sin 与 gamma 由调用方在 token 层加载好后按引用传进来。
 */
__aicore__ inline void RmsNormRopeOneHead(__local_mem__ bfloat16_t* inHead, __local_mem__ bfloat16_t* outHead,
                                          MicroAPI::RegTensor<float>& gammaLow, MicroAPI::RegTensor<float>& gammaHigh,
                                          MicroAPI::RegTensor<float>& cosValue, MicroAPI::RegTensor<float>& sinValue,
                                          MicroAPI::MaskReg pFull, uint32_t halfDim, float epsilon, float reciprocal)
{
    MicroAPI::RegTensor<bfloat16_t> xLowB16;
    MicroAPI::RegTensor<bfloat16_t> xHighB16;
    MicroAPI::RegTensor<float> xLow;
    MicroAPI::RegTensor<float> xHigh;
    MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(xLowB16, inHead);
    MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(xHighB16, inHead + halfDim);
    MicroAPI::Cast<float, bfloat16_t, CAST_B16_TO_B32>(xLow, xLowB16, pFull);
    MicroAPI::Cast<float, bfloat16_t, CAST_B16_TO_B32>(xHigh, xHighB16, pFull);

    // ---- RMSNorm：x / sqrt(mean(x^2) + eps) * gamma，中间量全在 vreg ----
    MicroAPI::RegTensor<float> squareLow;
    MicroAPI::RegTensor<float> squareHigh;
    MicroAPI::RegTensor<float> rms;
    MicroAPI::Mul(squareLow, xLow, xLow, pFull);
    MicroAPI::Mul(squareHigh, xHigh, xHigh, pFull);
    MicroAPI::Add(squareLow, squareLow, squareHigh, pFull);
    // 两个半宽相加后再归约，一次 Reduce 覆盖整个 D
    MicroAPI::ReduceSum(squareLow, squareLow, pFull);
    MicroAPI::Muls(squareLow, squareLow, reciprocal, pFull);
    MicroAPI::Adds(squareLow, squareLow, epsilon, pFull);
    MicroAPI::Sqrt(rms, squareLow, pFull);
    MicroAPI::Duplicate(rms, rms, pFull); // 广播 lane0
    MicroAPI::Div(xLow, xLow, rms, pFull);
    MicroAPI::Div(xHigh, xHigh, rms, pFull);
    MicroAPI::Mul(xLow, xLow, gammaLow, pFull);
    MicroAPI::Mul(xHigh, xHigh, gammaHigh, pFull);

    // ---- MRoPE 旋转（half-split，cos/sin 已由 Gather 合并）----
    //   out_lo = x_lo * cos - x_hi * sin
    //   out_hi = x_hi * cos + x_lo * sin
    MicroAPI::RegTensor<float> outLow;
    MicroAPI::RegTensor<float> outHigh;
    MicroAPI::RegTensor<float> cross;
    MicroAPI::Mul(outLow, xLow, cosValue, pFull);
    MicroAPI::Mul(cross, xHigh, sinValue, pFull);
    MicroAPI::Sub(outLow, outLow, cross, pFull);
    MicroAPI::Mul(outHigh, xHigh, cosValue, pFull);
    MicroAPI::Mul(cross, xLow, sinValue, pFull);
    MicroAPI::Add(outHigh, outHigh, cross, pFull);

    MicroAPI::RegTensor<bfloat16_t> outLowB16;
    MicroAPI::RegTensor<bfloat16_t> outHighB16;
    MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_B16>(outLowB16, outLow, pFull);
    MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_B16>(outHighB16, outHigh, pFull);
    MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(outHead, outLowB16, pFull);
    MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(outHead + halfDim, outHighB16, pFull);
}

/**
 * @brief 对一个 tile 的 Q/K 做 "RMSNorm -> MRoPE -> Cast bf16" 融合计算，整 tile 一次 VF
 *
 * Q 和 K 放在同一个 VF 里：它们在 UB 上本来就是同一行 token 内相邻的 head，
 * 拆成两次调用会让每 token 的 cos/sin Gather 做两遍。合到一起后 token 循环只走一趟，
 * cos/sin 只 Gather 一次，Q/K 两段 head 循环共用。
 *
 * gamma 逐 token 加载：cat_indices 可以把 und/gen 任意交错（竞品用例就是
 * [0,5,1,6,2,7,3,4] 这种），同一 tile 内相邻 token 的 gamma 可能不同，所以调用方
 * 要先按 token 铺好 [tokenSize, 2, D] 的 gamma tile（每 token 先 Q gamma 后 K gamma）。
 * 这样无论怎么交错，一个 tile 都只发一次 VF，不会退化成逐 token 调用。
 *
 * @param qkvBf16     tile 内首 token 的行首（bf16，[token][N][D]，Q 头在前、K 头紧随其后）
 * @param gammaTile   按 token 铺好的 gamma，float [tokenSize, 2, D]，32B 对齐
 * @param rawCosSin   首 token 的三轴原始 cos_sin 起址（float，[token][3][D]），32B 对齐
 * @param gatherIndex axisLut 展开成的元素索引，uint32 [D/2]，32B 对齐
 * @param qOutBf16    输出 q 段起址（bf16，[token][Hq][D]）
 * @param kOutBf16    输出 k 段起址（bf16，[token][Hk][D]）
 * @param tokenSize   tile 内 token 数
 * @param qHeadSize   Hq
 * @param kHeadSize   Hk
 * @param inTokenStride     输入 token 步长（元素）= N*D
 * @param gammaTokenStride  gamma tile token 步长（元素）= 2*D
 * @param cosSinTokenStride cos_sin token 步长（元素）= 3*D
 * @param qOutTokenStride   q 段 token 步长（元素）= Hq*D
 * @param kOutTokenStride   k 段 token 步长（元素）= Hk*D
 * @param headStride  head 步长（元素）= D，输入与输出相同
 * @param halfDim     D/2，必须等于 VL_FP32
 * @param epsilon     RMSNorm eps
 * @param reciprocal  1/D
 */
__aicore__ inline void QkRmsNormMropeTileVF(__local_mem__ bfloat16_t* qkvBf16, __local_mem__ float* gammaTile,
                                            __local_mem__ float* rawCosSin, __local_mem__ uint32_t* gatherIndex,
                                            __local_mem__ bfloat16_t* qOutBf16, __local_mem__ bfloat16_t* kOutBf16,
                                            uint16_t tokenSize, uint16_t qHeadSize, uint16_t kHeadSize,
                                            uint32_t inTokenStride, uint32_t gammaTokenStride,
                                            uint32_t cosSinTokenStride, uint32_t qOutTokenStride,
                                            uint32_t kOutTokenStride, uint32_t headStride, uint32_t halfDim,
                                            float epsilon, float reciprocal)
{
    if (tokenSize == 0) {
        return;
    }

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<uint32_t> mropeIndex;

        // gather 索引与 token 无关，整个 tile 只加载一次
        MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_NORM>(mropeIndex, gatherIndex);

        for (uint16_t t = 0; t < tokenSize; ++t) {
            MicroAPI::RegTensor<float> cosValue;
            MicroAPI::RegTensor<float> sinValue;
            MicroAPI::RegTensor<float> qGammaLow;
            MicroAPI::RegTensor<float> qGammaHigh;
            MicroAPI::RegTensor<float> kGammaLow;
            MicroAPI::RegTensor<float> kGammaHigh;

            // 三轴合并：一条 Gather 从 [3, D] 窗口里按 axisLut 挑出 D/2 个 lane。
            // 本 token 的 Q、K 所有 head 共用这一份 cos/sin
            __local_mem__ float* rawToken = rawCosSin + t * cosSinTokenStride;
            MicroAPI::Gather(cosValue, rawToken, mropeIndex, pFull);
            MicroAPI::Gather(sinValue, rawToken + halfDim, mropeIndex, pFull);

            // gamma 按 token 取：und / gen 两份权重在 tile 里已经按 token 展开好
            __local_mem__ float* gammaRow = gammaTile + t * gammaTokenStride;
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(qGammaLow, gammaRow);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(qGammaHigh, gammaRow + halfDim);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(kGammaLow, gammaRow + headStride);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(kGammaHigh, gammaRow + headStride + halfDim);

            __local_mem__ bfloat16_t* qInRow = qkvBf16 + t * inTokenStride;
            __local_mem__ bfloat16_t* qOutRow = qOutBf16 + t * qOutTokenStride;
            for (uint16_t h = 0; h < qHeadSize; ++h) {
                RmsNormRopeOneHead(qInRow + h * headStride, qOutRow + h * headStride, qGammaLow, qGammaHigh, cosValue,
                                   sinValue, pFull, halfDim, epsilon, reciprocal);
            }

            // K 头在同一行 token 内紧跟 Q 头之后
            __local_mem__ bfloat16_t* kInRow = qInRow + qHeadSize * headStride;
            __local_mem__ bfloat16_t* kOutRow = kOutBf16 + t * kOutTokenStride;
            for (uint16_t h = 0; h < kHeadSize; ++h) {
                RmsNormRopeOneHead(kInRow + h * headStride, kOutRow + h * headStride, kGammaLow, kGammaHigh, cosValue,
                                   sinValue, pFull, halfDim, epsilon, reciprocal);
            }
        }
    }
}

} // namespace UndGenQkvRmsNormRopeCache

#endif // UND_GEN_QKV_RMS_NORM_ROPE_CACHE_MROPE_VF_H_
