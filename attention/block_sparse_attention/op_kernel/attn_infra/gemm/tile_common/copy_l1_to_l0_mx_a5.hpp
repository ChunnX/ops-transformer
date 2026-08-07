/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_TILE_COPY_L1_TO_L0_MX_A5_HPP
#define GEMM_TILE_COPY_L1_TO_L0_MX_A5_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_arch.hpp"
#include "../../../tla/tensor.hpp"

namespace NpuArch::Gemm::Tile {

// =============================================================================
// MXFP4 专用 L1 -> L0 搬运 policy（带 E8M0 scale 的 5 参 mx LoadData）
// -----------------------------------------------------------------------------
// 现有 copy_l1_to_l0a_a5.hpp / copy_l1_to_l0b_a5.hpp 的 TileCopyTla 特化只用
// 3 参 AscendC::LoadData(dst, src, loadDataParams)，不带 scale、不带 LoadData2DMxParams，
// 无法承载 mxfp4 的反量化。本文件提供独立的 mx functor，供 block 直接实例化，
// 不侵入既有 TileCopyTla dispatch（避免影响其它 kernel）。
//
// 忠实移植策略：LoadData2DParamsV2 / LoadData2DMxParams 的每个字段均逐行对照
// block_mmad_pv_arch35_mxfp4.hpp 原裸写法 LoadVToL0 / LoadPToL0，逻辑维改由形参传入，
// 不从 tla stride 凭空重推（mx LoadData 的 16/64 块单位与 tla C0 常量非一一对应，
// 重推风险高）。原行号在每条赋值处以 [移植 LoadVToL0:xxx] 标注。
// TODO(后续): 待 MakeMxScaleLayout 的 stride 语义稳定后，把维数改由 tla 张量 shape 推导。
// =============================================================================

/// mx L1->L0A 转置搬运：L1 V(nZ, 自然 [S2,D]) + L1 V-scale -> L0A(zN, Vᵀ=[D,S2])。
/// 对应原 LoadVToL0（block_mmad_pv_arch35_mxfp4.hpp:159-182）。
struct CopyL1ToL0AMxA5 {
    __aicore__ inline CopyL1ToL0AMxA5() {}

    // dAct  : D=embedReal（V 的 embed 维，源 k；转置后 L0A m）
    // s2Act : S2 对齐 64（V 的 seq 维，源 m；转置后 L0A k）
    template <class TensorDst, class TensorSrc, class TensorScale>
    __aicore__ inline void operator()(
        TensorDst const &dst, TensorSrc const &src, TensorScale const &scale,
        uint32_t dAct, uint32_t s2Act)
    {
        AscendC::LoadData2DParamsV2 loadData2DParamsA;
        loadData2DParamsA.mStartPosition = 0;                 // [移植 :162]
        loadData2DParamsA.kStartPosition = 0;                 // [移植 :163]
        loadData2DParamsA.mStep = s2Act / 16;                 // 源 m = S2（16-块）[移植 :164]
        loadData2DParamsA.kStep = dAct / 64;                 // 源 k = D（64-块）[移植 :165]
        loadData2DParamsA.srcStride = s2Act / 16;            // [移植 :166]
        loadData2DParamsA.dstStride = (dAct + 15) / 16 + 1;  // L0A dst stride，+1 给 D+16 seed 行间隔 [移植 :167]
        loadData2DParamsA.ifTranspose = true;                 // 转置：源 (S2,D) -> L0A (D,S2) [移植 :168]

        AscendC::LoadData2DMxParams load2DMxParamsA;
        load2DMxParamsA.xStartPosition = 0;                   // [移植 :171]
        load2DMxParamsA.yStartPosition = 0;                   // [移植 :172]
        load2DMxParamsA.xStep = (dAct + 15) / 16;             // 转置后 L0A m = D [移植 :173]
        load2DMxParamsA.yStep = (s2Act + 63) / 64;            // 转置后 L0A k = S2（V 量化轴）[移植 :174]
        load2DMxParamsA.srcStride = load2DMxParamsA.yStep;   // [移植 :175]
        load2DMxParamsA.dstStride = load2DMxParamsA.yStep;    // [移植 :176]

        // [改造] 原 ReinterpretCast<fp4x2_e2m1_t / fp8_e8m0_t> 散点删除：
        // 调用点传入的 LocalTensor 已是 fp4x2_e2m1_t / fp8_e8m0_t 元素类型，直接传给 LoadData。
        // [TODO] fp4 粒度下 tla stride 与 mx LoadData 的 16/64 块单位非一一对应，
        //        暂不经 tla 张量重推 stride，维数由形参 dAct/s2Act 显式传入（忠实移植）。
                AscendC::LoadData(dst, src, scale, loadData2DParamsA, load2DMxParamsA);
    }
};

/// mx L1->L0B 转置搬运：L1 P(zN, 自然 [M,S2]) + L1 P-scale -> L0B(nZ, Pᵀ=[S2,M])。
/// 对应原 LoadPToL0（block_mmad_pv_arch35_mxfp4.hpp:184-207）。
struct CopyL1ToL0BMxA5 {
    __aicore__ inline CopyL1ToL0BMxA5() {}

    // s1Act : M=s1Align128（P 的 M 维，源 k；转置后 L0B n）
    // s2Act : S2 对齐 64（P 的 seq 维，源 m；转置后 L0B k）
    // dAct  : D=embedReal（P 量化 x 维，D=M=128 时 = M/16）
    // s2Base: 整 tile 的 S2Base（256），用于推导 scale srcStride（残留硬编码 5 的来源）
    template <class TensorDst, class TensorSrc, class TensorScale>
    __aicore__ inline void operator()(
        TensorDst const &dst, TensorSrc const &src, TensorScale const &scale, uint32_t s2Align64, uint32_t dAct, uint32_t s2Base, uint32_t actScaleSrcStride, uint32_t s1Align64)
    {
        AscendC::LoadData2DParamsV2 loadData2DParamsB;
        loadData2DParamsB.mStartPosition = 0;                       // [移植 :187]
        loadData2DParamsB.kStartPosition = 0;                       // [移植 :188]
        loadData2DParamsB.mStep = (s2Align64 + 15) / 16;                // 源 m = S2（16-块）[移植 :189]
        loadData2DParamsB.kStep = (s1Align64 + 63) / 64;                       // 源 k = M（64-块）[移植 :190] 残留:尾块需 align128
        loadData2DParamsB.srcStride = loadData2DParamsB.mStep;      // [移植 :191]
        loadData2DParamsB.dstStride = s1Align64 / 16;                   // L0B dst stride [移植 :192] 残留:尾块需 align128
        loadData2DParamsB.ifTranspose = true;                       // 转置 -> P 在 L0B 直通 (k=S2,n=M) [移植 :193]

        AscendC::LoadData2DMxParams load2DMxParamsB;
        load2DMxParamsB.xStartPosition = 0;                         // [移植 :196]
        load2DMxParamsB.yStartPosition = 0;                         // [移植 :197]
        load2DMxParamsB.xStep = (s1Align64 + 15) / 16;                          // D=M=128 时 = M/16 [移植 :198]
        load2DMxParamsB.yStep = (s2Align64 + 63) / 64;                  // P 量化轴 S2（64-块）[移植 :199]
        // [残留硬编码] 原 srcStride=5 = S2Base/64+1 = 256/64+1，仅 S2=256 正确。
        // [改造] 改由 s2Base 派生，适配真实 S2Base。
        load2DMxParamsB.srcStride = actScaleSrcStride == 0 ? s2Base / 64 + 1 : actScaleSrcStride;                // [移植+去硬编码 :200]
        load2DMxParamsB.dstStride = load2DMxParamsB.yStep;          // [移植 :201]
       
        AscendC::LoadData(dst, src, scale, loadData2DParamsB, load2DMxParamsB);
    }
};

/// mx L1->L0A 非转置搬运（QK）：L1 K(nZ, [S,D]) + K-scale(沿 D) -> L0A。
/// 忠实移植 block_mmad_qk_arch35_mxfp4.hpp::LoadKToL0A。
struct CopyL1ToL0AMxQKA5 {
    __aicore__ inline CopyL1ToL0AMxQKA5() {}

    // nSubRowStart : = nSub * L0_TILE_N，本子块在整 tile L1 中的行起点
    // nCur         : 本子块实际 K 行数（≤L0_TILE_N）
    // alignK       : CeilAlign(D, 64)
    // s2Align16Full: 整 tile 行 16 对齐（L1 数据 srcStride，与 zN 画布/QFA actSingleLoopS2SizeAlign 一致）
    // scaleK       : Ceil(D/64)*2，沿 D
    template <class TensorDst, class TensorSrc, class TensorScale>
    __aicore__ inline void operator()(
        TensorDst const &dst, TensorSrc const &src, TensorScale const &scale,
        uint32_t nSubRowStart, uint32_t nCur, uint32_t alignK,
        uint32_t s2Align16Full, uint32_t scaleK)
    {
        uint32_t nCurAlign16 = (nCur + 15) / 16 * 16;

        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.sid = 0;
        loadDataParams.mStartPosition = nSubRowStart / 16;          // [移植 LoadKToL0A]
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = nCurAlign16 / 16;
        loadDataParams.kStep = (alignK + 63) / 64;
        loadDataParams.srcStride = s2Align16Full / 16;
        loadDataParams.dstStride = nCurAlign16 / 16;
        loadDataParams.ifTranspose = false;

        AscendC::LoadData2DMxParams loadMxDataParams;
        loadMxDataParams.xStartPosition = nSubRowStart / 16;
        loadMxDataParams.yStartPosition = 0;
        loadMxDataParams.xStep = nCurAlign16 / 16;
        loadMxDataParams.yStep = scaleK / 2;                        // = Ceil(D/64)
        loadMxDataParams.srcStride = scaleK / 2;
        loadMxDataParams.dstStride = scaleK / 2;

        AscendC::LoadData(dst, src, scale, loadDataParams, loadMxDataParams);
    }
};

/// mx L1->L0B 非转置搬运（QK）：L1 Q(nZ, [M,D]) + Q-scale(沿 D) -> L0B。
/// 忠实移植 block_mmad_qk_arch35_mxfp4.hpp::LoadQToL0B。
struct CopyL1ToL0BMxQKA5 {
    __aicore__ inline CopyL1ToL0BMxQKA5() {}

    // mAlignL1      : [QFA 对齐] 恒 128（actMSizeAlign128 语义），L1 行分形数来源（mStep=8）
    // alignK        : CeilAlign(D, 64)
    // mAlignL0      : L0B dst 行对齐（= mAlignL1 = 128）
    // scaleMAlignL1 : [QFA 对齐] 恒 128，scale xStep=8
    // scaleK        : Ceil(D/64)*2，沿 D
    template <class TensorDst, class TensorSrc, class TensorScale>
    __aicore__ inline void operator()(
        TensorDst const &dst, TensorSrc const &src, TensorScale const &scale,
        uint32_t mAlignL1, uint32_t alignK, uint32_t mAlignL0,
        uint32_t scaleMAlignL1, uint32_t scaleK)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.sid = 0;
        loadDataParams.mStartPosition = 0;                          // [移植 LoadQToL0B]
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = (mAlignL1 + 15) / 16;
        loadDataParams.kStep = (alignK + 63) / 64;
        loadDataParams.srcStride = (mAlignL1 + 15) / 16;
        loadDataParams.dstStride = (mAlignL0 + 15) / 16;
        loadDataParams.ifTranspose = false;

        AscendC::LoadData2DMxParams loadMxDataParams;
        loadMxDataParams.xStartPosition = 0;
        loadMxDataParams.yStartPosition = 0;
        loadMxDataParams.xStep = (scaleMAlignL1 + 15) / 16;
        loadMxDataParams.yStep = scaleK / 2;
        loadMxDataParams.srcStride = scaleK / 2;
        loadMxDataParams.dstStride = scaleK / 2;

        AscendC::LoadData(dst, src, scale, loadDataParams, loadMxDataParams);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace NpuArch::Gemm::Tile

#endif // GEMM_TILE_COPY_L1_TO_L0_MX_A5_HPP
