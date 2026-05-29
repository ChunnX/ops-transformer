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
 * \file block_mmad_mx_fp8fp4.h
 * \brief MMAD block implementation for weight-quant grouped matmul split-M pipeline.
 */
#pragma once

#include "block_mmad.h"
#include "kernel_basic_intf.h"
#include "include/tensor_api/tensor.h"
#include "../utils/tensor_utils.h"
#include "../policy/dispatch_policy.h"
#include "../utils/grouped_matmul_constant.h"
#include "../tile/tile_mmad_mx.h"
#include "../tile/copy_scale_l1_to_l0a.h"
#include "../tile/copy_scale_l1_to_l0b.h"

using AscendC::BLOCK_CUBE;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::HardEvent;
using AscendC::SetFlag;
using AscendC::TEventID;
using AscendC::WaitFlag;
using Blaze::Gemm::GroupedMatmulRecipe::DOUBLE_BUFFER;
using Blaze::Gemm::GroupedMatmulRecipe::FINAL_ACCUMULATION;
using Blaze::Gemm::GroupedMatmulRecipe::MX_DIVISOR_SIZE;
using Blaze::Gemm::GroupedMatmulRecipe::MX_GROUP_SIZE;
using Blaze::Gemm::GroupedMatmulRecipe::NON_FINAL_ACCUMULATION;
using Blaze::Gemm::GroupedMatmulRecipe::SYNC_MODE4;

namespace Blaze {
namespace Gemm {
namespace Block {
// Macro aliases keep the specialization declaration compact for this dispatch-policy binding.
#define WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM                                                                             \
    template <                                                                                                        \
        class ATypeTuple_, class LayoutATuple_, class BTypeTuple_, class LayoutBTuple_, class CType_, class LayoutC_, \
        class BiasType_, class LayoutBias_>

#define WQBMM_CUBE_COMPUTE_CLASS                                                                                   \
    BlockMmad<                                                                                                     \
        MatmulMxFp8Fp4DynamicKL1TailResplit, ATypeTuple_, LayoutATuple_, BTypeTuple_, LayoutBTuple_, CType_, LayoutC_, \
        BiasType_, LayoutBias_>

/*!
 * \brief AIC tile-compute unit for one tile inside one group in the weight-quant grouped matmul pipeline.
 *
 * Design reason:
 * - This class handles MMAD compute for a single-group tile only.
 * - AIC does not support direct FP4E2M1 -> FP8E4M3 conversion in the MMAD path.
 * - Therefore B must be preprocessed by prologue first, and this class consumes the converted B tiles.
 *
 * Distinctive behaviors:
 * 1) It synchronizes with prologue through cross-core flags, and the sync semantics must match prologue exactly.
 * 2) It uses dynamic kL1 splitting (kaL1/kbL1) and this policy is aligned with prologue's K-window organization.
 * 3) It uses dynamic kL0 splitting inside each kL1 tile to balance compute and memory movement.
 * 4) A/scaleA/scaleB/B use different transfer granularities by design:
 *    - scaleA/scaleB are moved with MX_SCALE_K_L1_SIZE = 4096 K-window,
 *    - typical kaL1/kbL1 are 256/512,
 *    - this larger 4096 window is used to organize scale transfer at 128B cacheline-friendly granularity,
 *      improving effective bandwidth and reuse.
 *
 * Key constraints:
 * 1) A must satisfy ND format.
 * 2) B must be prologue-converted ZN format and use the same compute type as AType.
 * 3) Scale layout must satisfy MX_DIVISOR_SIZE = 64.
 *
 * When to use:
 * - Use this block on the mxfp8fp4 path when dynamic kL1/kL0 blocking is needed to increase per-tile data workload
 *   and overall throughput.
 */
WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
class WQBMM_CUBE_COMPUTE_CLASS {
public:
    using DispatchPolicy = MatmulMxFp8Fp4DynamicKL1TailResplit;
    using ProblemShape = AscendC::Te::Shape<uint64_t, uint64_t, uint64_t, uint64_t>;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using AType = typename AscendC::Std::tuple_element<0, ATypeTuple_>::type;
    using ScaleBType = typename AscendC::Std::tuple_element<1, BTypeTuple_>::type;
    using ScaleAType = typename AscendC::Std::tuple_element<1, ATypeTuple_>::type;
    using CType = CType_;

    using LayoutA = typename AscendC::Std::tuple_element<0, LayoutATuple_>::type;
    using LayoutScaleA = typename AscendC::Std::tuple_element<1, LayoutATuple_>::type;
    using LayoutB = typename AscendC::Std::tuple_element<0, LayoutBTuple_>::type;
    using LayoutScaleB = typename AscendC::Std::tuple_element<1, LayoutBTuple_>::type;
    using LayoutC = LayoutC_;

    static_assert(AscendC::Te::IsSatisfiedPtnFormatV<decltype(AscendC::Te::MakeTensor(
                      AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>((__gm__ AType*)0), LayoutA{}(0UL, 0UL))),
                      AscendC::Te::NDExtLayoutPtn>);

    constexpr static int32_t C0_SIZE = AscendC::AuxGetC0Size<AType>();
    constexpr static int32_t SCALE_C0 = 2;
    constexpr static int32_t L0C_C0 = 16;

    // Keep the common BlockMmad Init form while preserving the Tensor-based operator() path.
    struct L1Params {
        uint64_t kL1;
        uint64_t scaleKL1;
        uint64_t l1BufNum;
    };

    __aicore__ inline void Init(
        const ProblemShape& problemShape, const BlockShape& l0TileShape, const L1Params& l1Params, bool isBias,
        bool dbL0C);

    __aicore__ inline BlockMmad();
    template <typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC>
    __aicore__ inline void operator()(
        const TensorA& tensorA, const TensorScaleA& tensorScaleA, const TensorScaleB& tensorScaleB,
        const TensorC& tensorC);
    __aicore__ inline ~BlockMmad();

private:
    struct BlockMmadOffsetParam {
        uint64_t mL1Size;
        uint64_t kaL1Size;
        uint64_t kbL1Size;
        uint64_t nL1Size;
        uint64_t kSize;
        uint64_t scaleKL1Size;
        uint64_t l0KSize;
    };

    __aicore__ inline void WaitAivToAic();
    __aicore__ inline void SetAicToAiv();

    __aicore__ inline void CalcDynamicKBlock(
        uint64_t mL1Size, uint64_t nL1Size, uint64_t& kaL1Size, uint64_t& kbL1Size) const;
    __aicore__ inline void ProcessTileL1(int64_t kbOffset, uint64_t kbL1RealSize, const BlockMmadOffsetParam& param);
    __aicore__ inline void WaitAMTE1ToMTE2();
    __aicore__ inline void SetMTE1ToMTE2();
    __aicore__ inline void WaitScaleMTE1ToMTE2();
    __aicore__ inline void SetScaleMTE1ToMTE2();
    template <typename TensorA>
    __aicore__ inline void CopyAGmToL1(const TensorA& tensorA, const BlockMmadOffsetParam& param, int64_t kaGmOffset);
    template <typename TensorScaleA, typename TensorScaleB>
    __aicore__ inline void CopyMxScaleGmToL1(
        const TensorScaleA& tensorScaleA, const TensorScaleB& tensorScaleB, const BlockMmadOffsetParam& param,
        uint64_t kbL1Offset);
    template <typename TensorC>
    __aicore__ inline void CopyCL0c2GmOrUb(const TensorC& tensorC, const BlockMmadOffsetParam& param);

    using MakeLayoutAL1 = AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutScaleAL1 = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>;
    using MakeLayoutScaleBL1 = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>;

    using MakeLayoutAL0 = AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutBL0 = AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutScaleAL0 = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>;
    using MakeLayoutScaleBL0 = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>;

    static constexpr uint64_t L1_M = 256;
    static constexpr uint64_t L1_N = 256;

    static constexpr uint64_t L1_K_CONFIG_512 = 512;
    static constexpr uint64_t L1_K_CONFIG_256 = 256;
    static constexpr uint64_t MX_SCALE_K_L1_SIZE = 4096;
    static constexpr uint64_t L1_K_DYNAMIC_CONFIG_N_THRESHOLD = L1_N >> 1;

    // Init state. Comments record which old implicit/runtime values they replace.
    uint64_t m_{1};
    uint64_t n_{1};
    // Previously `kSize`, derived from tensorA.Layout() in operator().
    uint64_t k_{1};
    // Previously the nominal `kbL1Size` chosen for non-tail tiles.
    uint64_t kL1_{1};
    // Previously the fixed constant `MX_SCALE_K_L1_SIZE`.
    uint64_t scaleKL1_{MX_SCALE_K_L1_SIZE};
    uint64_t l1BufNum_{DOUBLE_BUFFER};
    // Nominal tile M/N from Init. Actual current-tile M/N still come from tensorC.Layout() for tail compatibility.
    uint64_t baseM_{L1_M};
    uint64_t baseN_{L1_N};
    // Previously `l0KSize`, derived as `(mL1Size <= 128 && nL1Size <= 128) ? 256 : 128`.
    uint64_t baseK_{128};
    uint64_t kaL1Size_{L1_K_CONFIG_256};
    uint64_t kbL1Size_{L1_K_CONFIG_256};
    bool isBias_{false};
    bool enableL0cPingPong_{false};
    bool hasInitConfig_{false};

    uint64_t aL1BufIdx_ = 0;
    uint64_t bL1BufIdx_ = 0;
    uint64_t scaleL1BufIdx_ = 0;
    uint64_t l0BufIdx_ = 0;

    using TensorAL1 = kernel_utils::TensorType<AscendC::Hardware::L1, AType, MakeLayoutAL1>;
    using TensorScaleAL1 = kernel_utils::TensorType<AscendC::Hardware::L1, ScaleAType, MakeLayoutScaleAL1>;
    using TensorScaleBL1 = kernel_utils::TensorType<AscendC::Hardware::L1, ScaleBType, MakeLayoutScaleBL1>;

    TensorAL1 tensorAL1_;
    TensorScaleAL1 tensorScaleAL1_;
    TensorScaleBL1 tensorScaleBL1_;

    // --- sync ---
    // 2 buffer
    static constexpr TEventID eventIdsMte1ToMte2_ = 0;
    // 2 buffer
    static constexpr TEventID eventIdsMxScaleMte1ToMte2_ = 2;
    static constexpr TEventID eventIdMte1ToMte2_ = 4;
    static constexpr TEventID eventIdMte2ToMte1_ = 0;
    static constexpr TEventID eventIdMToMte1_ = 0;
    static constexpr TEventID eventIdMte1ToM_ = 0;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 0;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = 1;

    /**
     * L1 512KB Memory Map
     * * Segment 1 [0KB - 256KB]:
     * [0k]    [64k]      [96k]    [128k]                        [256k]
     * |--B B0---|--scA0---|--scB0---|----------- A (Part 1) -------|
     *     (64KB)     (32KB)    (32KB)            (128KB)
     * * Segment 2 [256KB - 512KB]:
     * [256k]                    [384k]        [448k]   [480k]   [512k]
     * |-------- A (Part 2) -------|---- B B1 ----|--scA1--|--scB1--|
     *          (128KB)               (64KB)        (32KB)    (32KB)
     * * Note: A is a contiguous 256KB block spanning the middle of the buffer.
     */
    static constexpr uint64_t SCALE_AL1_OFFSET = 64 * 1024;
    static constexpr uint64_t SCALE_BL1_OFFSET = 96 * 1024;
    static constexpr uint64_t A_L1_OFFSET = 128 * 1024;
    static constexpr uint64_t L1_BUF_OFFSET = 384 * 1024;
    static constexpr uint64_t A_L1_BUF_OFFSET = 128 * 1024;
    static constexpr uint64_t A_L1_SINGLE_BUF_SIZE = 128 * 1024;

    /**
     * L0 64KB Memory Map (double-buffered B tiles)
     * [0k]      [32KB]    [64KB]
     * |--- B0 ---|--- B1 ---|
     *    (32KB)     (32KB)
     */
    static constexpr uint64_t L0_BUF_OFFSET = 32 * 1024;
};

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::ProcessTileL1(
    int64_t kbOffset, uint64_t kbL1RealSize, const BlockMmadOffsetParam& param)
{
    auto tensorBL1 = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(bL1BufIdx_ * L1_BUF_OFFSET),
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>{}(kbL1RealSize, param.nL1Size));
    SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1_);
    WaitFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1_);
    // Decide the MMAD accumulation mode for this K-tile.
    bool isLastGmK = kbOffset + kbL1RealSize >= param.kSize;
    bool isFirstGmK = kbOffset == 0;
    uint64_t l0KSize = param.l0KSize;
    for (uint64_t l1KOffset = 0; l1KOffset < kbL1RealSize; l1KOffset += l0KSize) {
        bool isLastL1K = l1KOffset + l0KSize >= kbL1RealSize;
        uint64_t realL0k = isLastL1K ? kbL1RealSize - l1KOffset : l0KSize;
        uint64_t realL0ScaleK = CeilDiv(realL0k, MX_DIVISOR_SIZE) * 2;
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1_ + l0BufIdx_);

        auto CopyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
        auto layoutAL0 = MakeLayoutAL0{}(param.mL1Size, realL0k);
        auto tensorAL0 =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(l0BufIdx_ * L0_BUF_OFFSET), layoutAL0);
        auto tensorBlockAL1 = tensorAL1_.Slice(AscendC::Te::MakeCoord(0, (l1KOffset + kbOffset) % param.kaL1Size),
            AscendC::Te::MakeShape(param.mL1Size, realL0k));
        AscendC::Te::Copy(CopyL12L0A, tensorAL0, tensorBlockAL1);

        auto layoutScaleAL0 = MakeLayoutScaleAL0{}(param.mL1Size, realL0ScaleK);
        auto tensorScaleAL0 =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, fp8_e8m0_t>(l0BufIdx_ * L0_BUF_OFFSET), layoutScaleAL0);
        auto CopyL12L0MxA8W4ScaleA3510 = AscendC::Te::MakeCopy(Tile::CopyL12L0MxA8W4ScaleA3510{});
        AscendC::Te::Copy(
            CopyL12L0MxA8W4ScaleA3510, tensorScaleAL0, tensorScaleAL1_,
            AscendC::Te::MakeCoord(0, ((l1KOffset + kbOffset) % param.scaleKL1Size)));

        auto CopyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
        auto layoutBL0 = MakeLayoutBL0{}(realL0k, param.nL1Size);
        auto tensorBL0 =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, AType>(l0BufIdx_ * L0_BUF_OFFSET), layoutBL0);
        auto tensorBlockBL1 =
            tensorBL1.Slice(AscendC::Te::MakeCoord(l1KOffset, 0), AscendC::Te::MakeShape(realL0k, param.nL1Size));
        AscendC::Te::Copy(CopyL12L0B, tensorBL0, tensorBlockBL1);

        auto layoutScaleBL0 = MakeLayoutScaleBL0{}(realL0ScaleK, param.nL1Size);
        auto tensorScaleBL0 =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, fp8_e8m0_t>(l0BufIdx_ * L0_BUF_OFFSET), layoutScaleBL0);
        auto CopyL12L0MxA8W4ScaleB3510 = AscendC::Te::MakeCopy(Tile::CopyL12L0MxA8W4ScaleB3510{});
        AscendC::Te::Copy(
            CopyL12L0MxA8W4ScaleB3510, tensorScaleBL0, tensorScaleBL1_,
            AscendC::Te::MakeCoord(((l1KOffset + kbOffset) % param.scaleKL1Size), 0));

        SetFlag<HardEvent::MTE1_M>(eventIdMte1ToM_);
        WaitFlag<HardEvent::MTE1_M>(eventIdMte1ToM_);

        auto layoutL0C = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<L0C_C0>>(param.mL1Size, param.nL1Size);
        auto tensorL0C = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(0), layoutL0C);
        AscendC::Te::MmadParams params;
        params.m = static_cast<uint16_t>(param.mL1Size);
        params.k = static_cast<uint16_t>(realL0k);
        params.n = static_cast<uint16_t>(param.nL1Size);
        params.unitFlag = (isLastGmK && isLastL1K) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
        params.cmatrixInitVal = isFirstGmK && l1KOffset == 0;
        AscendC::Te::Mmad(
            AscendC::Te::MmadAtom<
                AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, AscendC::Te::MmadTraitMX>>{}
                .with(params), tensorL0C, tensorAL0, tensorBL0);

        SetFlag<HardEvent::M_MTE1>(eventIdMToMte1_ + l0BufIdx_);
        l0BufIdx_ ^= 1;
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::WaitAMTE1ToMTE2()
{
    WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2_ + aL1BufIdx_);
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::WaitScaleMTE1ToMTE2()
{
    WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMxScaleMte1ToMte2_ + scaleL1BufIdx_);
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::SetScaleMTE1ToMTE2()
{
    SetFlag<HardEvent::MTE1_MTE2>(eventIdsMxScaleMte1ToMte2_ + scaleL1BufIdx_);
    scaleL1BufIdx_ ^= 1;
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::SetMTE1ToMTE2()
{
    SetFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2_ + aL1BufIdx_);
    aL1BufIdx_ ^= 1;
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
template <typename TensorA>
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::CopyAGmToL1(
    const TensorA& tensorA, const BlockMmadOffsetParam& param, int64_t kaGmOffset)
{
    int64_t kaL1RealSize = (kaGmOffset + param.kaL1Size) >= param.kSize ? param.kSize - kaGmOffset : param.kaL1Size;
    auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
    auto layoutAL1 = MakeLayoutAL1{}(param.mL1Size, kaL1RealSize);
    auto gmBlockA = tensorA.Slice(AscendC::Te::MakeCoord(0, kaGmOffset), AscendC::Te::MakeShape(param.mL1Size, kaL1RealSize));
    tensorAL1_ = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(A_L1_OFFSET + aL1BufIdx_ * A_L1_BUF_OFFSET), layoutAL1);
    AscendC::Te::Copy(copyGM2L1, tensorAL1_, gmBlockA);
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
template <typename TensorScaleA, typename TensorScaleB>
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::CopyMxScaleGmToL1(
    const TensorScaleA& tensorScaleA, const TensorScaleB& tensorScaleB, const BlockMmadOffsetParam& param,
    uint64_t kbL1Offset)
{
    uint64_t scaleKGmSize = param.kSize / MX_GROUP_SIZE;
    uint64_t scaleKL1StandardLen = param.scaleKL1Size / MX_GROUP_SIZE;
    uint64_t scaleKL1RealSize = (kbL1Offset + param.scaleKL1Size) > param.kSize ?
                                    (param.kSize - kbL1Offset) / MX_GROUP_SIZE :
                                    scaleKL1StandardLen;
    auto CopyScaleGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
    auto layoutScaleAL1 = MakeLayoutScaleAL1{}(param.mL1Size, scaleKL1RealSize);
    tensorScaleAL1_ = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(SCALE_AL1_OFFSET + scaleL1BufIdx_ * L1_BUF_OFFSET), layoutScaleAL1);
    auto gmBlockScaleA = tensorScaleA.Slice(AscendC::Te::MakeCoord(0, kbL1Offset / MX_GROUP_SIZE), AscendC::Te::MakeShape(param.mL1Size, scaleKL1RealSize));
    AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleAL1_, gmBlockScaleA);

    auto layoutScaleBL1 = MakeLayoutScaleBL1{}(scaleKL1RealSize, param.nL1Size);
    tensorScaleBL1_ = AscendC::Te::MakeTensor(
        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(SCALE_BL1_OFFSET + scaleL1BufIdx_ * L1_BUF_OFFSET), layoutScaleBL1);
    auto gmBlockScaleB = tensorScaleB.Slice(AscendC::Te::MakeCoord(kbL1Offset / MX_GROUP_SIZE, 0), AscendC::Te::MakeShape(scaleKL1RealSize, param.nL1Size));
    AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleBL1_, gmBlockScaleB);
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
template <typename TensorC>
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::CopyCL0c2GmOrUb(const TensorC& tensorC, const BlockMmadOffsetParam& param)
{
    constexpr uint64_t FP32_64_AS_UINT64 = 0x42800000;
    auto layoutL0C = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<L0C_C0>>(param.mL1Size, param.nL1Size);
    auto tensorL0C = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(0), layoutL0C);
    if constexpr (Std::is_same_v<Te::GetMemLocation<TensorC>, Te::Location::UB>) {
        // C L0C->UB
        auto CopyL0C2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2UB{});
        AscendC::Te::Copy(CopyL0C2UB, tensorC, tensorL0C, FP32_64_AS_UINT64, AscendC::Te::FixpipeParams{/*unitflag*/ 3});
    } else {
        // C L0C->GM
        auto CopyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        AscendC::Te::Copy(CopyL0C2GM, tensorC, tensorL0C, FP32_64_AS_UINT64, AscendC::Te::FixpipeParams{/*unitflag*/ 3});
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline WQBMM_CUBE_COMPUTE_CLASS::BlockMmad()
{
    for (uint64_t i = 0; i < DOUBLE_BUFFER; i++) {
        SetAicToAiv();
        SetFlag<HardEvent::M_MTE1>(eventIdMToMte1_ + i);
        SetFlag<HardEvent::MTE1_MTE2>(eventIdsMxScaleMte1ToMte2_ + i);
        SetFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2_ + i);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline WQBMM_CUBE_COMPUTE_CLASS::~BlockMmad()
{
    for (uint64_t i = 0; i < DOUBLE_BUFFER; i++) {
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMte1_ + i);
        WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMxScaleMte1ToMte2_ + i);
        WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2_ + i);
    }
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::Init(
    const ProblemShape& problemShape, const BlockShape& l0TileShape, const L1Params& l1Params, bool isBias,
    bool dbL0C)
{
    m_ = AscendC::Te::Get<IDX_M_IDX>(problemShape);
    n_ = AscendC::Te::Get<IDX_N_IDX>(problemShape);
    // k_ replaces the old runtime `kSize`, which was previously derived from tensorA.Layout().
    k_ = AscendC::Te::Get<IDX_K_IDX>(problemShape);

    // kL1_ replaces the old nominal kbL1Size input for non-tail tiles.
    kL1_ = l1Params.kL1;
    // scaleKL1_ replaces the old fixed MX_SCALE_K_L1_SIZE value.
    scaleKL1_ = l1Params.scaleKL1;
    l1BufNum_ = l1Params.l1BufNum;

    // baseM_/baseN_ are the caller-provided nominal tile sizes. operator() still reads the current tile M/N from
    // tensorC.Layout() because the Tensor-only interface must preserve tail-tile behavior.
    baseM_ = AscendC::Te::Get<IDX_M_IDX>(l0TileShape);
    baseN_ = AscendC::Te::Get<IDX_N_IDX>(l0TileShape);
    // baseK_ replaces the old internal l0KSize heuristic once the caller explicitly provides l0TileShape.k.
    baseK_ = AscendC::Te::Get<IDX_K_IDX>(l0TileShape);

    isBias_ = isBias;
    enableL0cPingPong_ = dbL0C;
    CalcDynamicKBlock(baseM_, baseN_, kaL1Size_, kbL1Size_);

    aL1BufIdx_ = 0;
    bL1BufIdx_ = 0;
    scaleL1BufIdx_ = 0;
    l0BufIdx_ = 0;
    hasInitConfig_ = true;
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::CalcDynamicKBlock(
    uint64_t mL1Size, uint64_t nL1Size, uint64_t& kaL1Size, uint64_t& kbL1Size) const
{
    kbL1Size = nL1Size <= L1_K_DYNAMIC_CONFIG_N_THRESHOLD ? L1_K_CONFIG_512 : L1_K_CONFIG_256;
    if (mL1Size < nL1Size) {
        uint64_t mL1Align = CeilAlign(mL1Size, static_cast<uint64_t>(BLOCK_CUBE));
        kaL1Size = (A_L1_SINGLE_BUF_SIZE / sizeof(AType)) / (mL1Align * kbL1Size) * kbL1Size;
    } else {
        kaL1Size = kbL1Size;
    }
}

/*
 * kaL1Size % kbL1Size == 0
 * scaleL1Size % kbL1Size == 0
 */
WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
template <typename TensorA, typename TensorScaleA, typename TensorScaleB, typename TensorC>
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::operator()(
    const TensorA& tensorA, const TensorScaleA& tensorScaleA, const TensorScaleB& tensorScaleB, const TensorC& tensorC)
{
    BlockMmadOffsetParam blockParam = {};
    // These still come from tensorC.Layout() because operator() needs the real current tile shape, including tails.
    blockParam.mL1Size =
        AscendC::Te::GetElement<AscendC::Te::AttrInfo::Shape, AscendC::Te::AttrInfo::Row, 1>(tensorC.Layout());
    blockParam.nL1Size =
        AscendC::Te::GetElement<AscendC::Te::AttrInfo::Shape, AscendC::Te::AttrInfo::Column, 1>(tensorC.Layout());
    if (hasInitConfig_) {
        // kSize originally came from tensorA.Layout(); after Init it comes from problemShape.K.
        blockParam.kSize = k_;
        // scaleKL1Size originally used the fixed MX_SCALE_K_L1_SIZE constant.
        blockParam.scaleKL1Size = scaleKL1_;
        // l0KSize originally used the internal `(mL1Size <= 128 && nL1Size <= 128) ? 256 : 128` rule.
        blockParam.l0KSize = baseK_;
        if (blockParam.mL1Size == baseM_ && blockParam.nL1Size == baseN_) {
            blockParam.kaL1Size = kaL1Size_;
            blockParam.kbL1Size = kbL1Size_;
        } else {
            // Tail M/N tiles keep the original dynamic ka/kb derivation from the actual Tensor shape.
            CalcDynamicKBlock(blockParam.mL1Size, blockParam.nL1Size, blockParam.kaL1Size, blockParam.kbL1Size);
        }
    } else {
        blockParam.kSize =
            AscendC::Te::GetElement<AscendC::Te::AttrInfo::Shape, AscendC::Te::AttrInfo::Column, 1>(tensorA.Layout());
        blockParam.scaleKL1Size = MX_SCALE_K_L1_SIZE;
        blockParam.l0KSize = (blockParam.mL1Size <= 128 && blockParam.nL1Size <= 128) ? 256 : 128;
        CalcDynamicKBlock(blockParam.mL1Size, blockParam.nL1Size, blockParam.kaL1Size, blockParam.kbL1Size);
    }

    for (uint64_t kbGmOffset = 0; kbGmOffset < blockParam.kSize; kbGmOffset += blockParam.kbL1Size, bL1BufIdx_ ^= 1) {
        uint64_t kbL1RealSize = (kbGmOffset + blockParam.kbL1Size) >= blockParam.kSize ? blockParam.kSize - kbGmOffset :
                                                                                         blockParam.kbL1Size;
        if (kbGmOffset % blockParam.scaleKL1Size == 0) {
            WaitScaleMTE1ToMTE2();
            CopyMxScaleGmToL1(tensorScaleA, tensorScaleB, blockParam, kbGmOffset);
        }

        if (kbGmOffset % blockParam.kaL1Size == 0) {
            WaitAMTE1ToMTE2();
            CopyAGmToL1(tensorA, blockParam, kbGmOffset);
        }

        WaitAivToAic();
        ProcessTileL1(kbGmOffset, kbL1RealSize, blockParam);
        uint64_t nextKbGmOffset = kbGmOffset + blockParam.kbL1Size;
        if (nextKbGmOffset % blockParam.kaL1Size == 0 || nextKbGmOffset >= blockParam.kSize) {
            SetMTE1ToMTE2();
        }
        if (nextKbGmOffset % blockParam.scaleKL1Size == 0 || nextKbGmOffset >= blockParam.kSize) {
            SetScaleMTE1ToMTE2();
        }
        SetAicToAiv();
    }
    CopyCL0c2GmOrUb(tensorC, blockParam);
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::WaitAivToAic()
{
    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
}

WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void WQBMM_CUBE_COMPUTE_CLASS::SetAicToAiv()
{
    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
}

#undef WQBMM_CUBE_COMPUTE_TEMPLATE_PARAM
#undef WQBMM_CUBE_COMPUTE_CLASS
} // namespace Block
} // namespace Gemm
} // namespace Blaze