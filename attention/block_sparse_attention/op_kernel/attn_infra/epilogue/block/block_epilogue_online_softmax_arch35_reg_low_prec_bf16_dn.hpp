/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Stage 2（DN 列归约）：BF16 存 UB + 计算中升精度 FP32 的 Online Softmax。
// 输入 ubS 物理形状是单 AIV 的紧凑 [S2=256, S1=64] BF16（S1 连续，每行 64 个 bf16）。
// max 沿 S2 在 BF16 packed 域 in-lane 累积、末尾一次升 FP32；sum/exp 逐寄存器升 FP32，
// 出口把 P 降回 BF16 并按 §2.4 的 [128,128] 伪宽行 NZ 落 UB（上半⊕下半同号行）。
#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_BF16_DN_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_BF16_DN_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/tile_copy.hpp"
#include "../../../attn_infra/gemm_coord.hpp"
#include "../../../attn_infra/matrix_coord.hpp"
#include "../../../tla/tensor.hpp"
#include "../../../tla/layout.hpp"

namespace NpuArch::Epilogue::Block {

template <
    class OutputType_,
    class LayoutS_>
class BlockEpilogue<
    EpilogueOnlineSoftmaxBsaDn,
    OutputType_,
    Gemm::GemmType<bfloat16_t, LayoutS_>>
{
public:
    using DispatchPolicy = EpilogueOnlineSoftmaxBsaDn;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = bfloat16_t;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = LayoutS_;

    static constexpr uint32_t BLOCK_SIZE_IN_BYTE = 32;
    static constexpr uint32_t REPEAT_SIZE_IN_BYTE = 256;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 32768;
    static constexpr uint32_t VECTOR_SIZE = 128;
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 16384;
    static constexpr uint32_t DM_UB_GLOBAL_ELEM_NUM = 64;
    static constexpr uint32_t ELE_NUM_PER_C0 = 16;
    static constexpr uint32_t C0_NUM_PER_FRACTAL = 16;

    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;
    static constexpr uint32_t HALF_REP_SIZE = 128;
    static constexpr uint32_t FLOAT_REP_SIZE = 64;
    static constexpr uint32_t BLOCK_REP_SIZE = 8;
    static constexpr uint32_t REPEAT_STRIDE = 1;
    // 单 AIV：S1 列数（每行 bf16 个数），2 行恰好填满一个 128-lane 寄存器
    static constexpr uint32_t SM_ROW_MAX_ELEM_NUM = 64;
    // S2 列归约维度（= kvBaseTile），多算少搬：脏行填 MIN 后仍跑满 256
    static constexpr uint32_t SM_COL_MAX_ELEM_NUM = 256;
    static constexpr uint32_t SM_VREG_SIZE = 256 / sizeof(ElementInput);

    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue_)
    {
        // UB 地址布局与 ND BF16 路径完全一致（共用同一份偏移），便于 ls/lp 双缓冲与 RescaleO 共享状态向量区
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t LP_UB_TENSOR_OFFSET = 2 * UB_UINT8_BLOCK_SIZE;

        constexpr uint32_t LM_UB_TENSOR_OFFSET = 7 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = LM_UB_TENSOR_OFFSET + 64 * sizeof(float);
        constexpr uint32_t DM_UB_TENSOR_OFFSET = GM_UB_TENSOR_OFFSET + 64 * sizeof(float);
        constexpr uint32_t LL_UB_TENSOR_OFFSET = DM_UB_TENSOR_OFFSET + 3 * 64 * sizeof(float);
        constexpr uint32_t GL_UB_TENSOR_OFFSET = LL_UB_TENSOR_OFFSET + 64 * sizeof(float);

        subBlockIdx_ = AscendC::GetSubBlockIdx();
        scaleValueF = scaleValue_;
        MIN_VALUE = AscendC::ToBfloat16(-3.389531390315715675e+38);

        lsUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LS_UB_TENSOR_OFFSET);
        lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        lmUbFloatTensor = resource.ubBuf.template GetBufferByByte<float>(LM_UB_TENSOR_OFFSET);
        llUbFloatTensor = resource.ubBuf.template GetBufferByByte<float>(LL_UB_TENSOR_OFFSET);
    }

    __aicore__ inline
    ~BlockEpilogue()
    {
    }

    // Stage 3 解折叠：把 UB 的 [128,128] 伪宽行 NZ（左半=S2 上半、右半=S2 下半）拆回 L1 的 P'[S2,S1] 自然 NZ。
    // 两次 DataCopy 的几何（S2=256，单位均为 32B DataBlock=16 个 bf16，对齐 PFA prompt_flash_attention_mm2.h L126-151）：
    //   ① 左半 = UB CG0..CG_{cg-1}（S2 上半 [0,128)）→ 每个 L1 CG' 的上 8 个分形行
    //   ② 右半 = UB CG_{cg}..（S2 下半 [128,256)）→ 同 CG' 的下 8 个分形行（dst 下移 LOWER_DST）
    // 只搬位置、块内不动、不转置；转置交给 Cube 的 isTransposeA load。dirty S2 行已在 Stage 2 填 0，可统一走全块参数。
    __aicore__ inline
    void UnfoldPUbToPL1(AscendC::LocalTensor<ElementOutput> l1PBase,
        AscendC::LocalTensor<ElementOutput> ubPBase, uint32_t mRound, uint32_t mCopyOffset, uint32_t blockStride)
    {
        const uint32_t cgNum = mRound / C0_NUM_PER_FRACTAL;                 // 本 AIV 的列组（CG）数 = S1/16
        const uint32_t halfDb = SM_COL_MAX_ELEM_NUM >> 1;                   // S2/2，上/下半各 8 个分形行的 DataBlock 数
        // 右半（S2 下半）在 UB 中的起始：cgNum 个 CG，每个 CG 占 blockStride*16 个 bf16（含 1 行 padding）
        const uint32_t rightSrcOffset = cgNum * blockStride * ELE_NUM_PER_C0;
        // L1 每个 CG' 内下半分形行起点：8 个分形行 × (16×16) = SM_COL_MAX_ELEM_NUM*8 个 bf16
        const uint32_t lowerDstOffset = SM_COL_MAX_ELEM_NUM * 8;
        // 两个 AIV 各占 P'[S2,S1] 的一半列：AIV1 起点在 AIV0 的 mCopyOffset 列之后
        const uint32_t aivBaseOffset = subBlockIdx_ * (2u * mCopyOffset) * halfDb;

        AscendC::DataCopyParams params;
        params.blockCount = static_cast<uint16_t>(cgNum);
        params.blockLen = static_cast<uint16_t>(halfDb);                   // 每 CG 搬上(或下)半 = 8 分形行
        params.srcStride = 1;                                              // 跳过 UB CG 的 1 行 padding（blockStride=halfDb+1）
        params.dstStride = static_cast<uint16_t>(halfDb);                  // 跳到下一个 L1 CG'（CG'=2*halfDb 个 DataBlock）

        // ① 左半（S2 上半）→ L1 CG' 上 8 分形行
        AscendC::DataCopy(l1PBase[aivBaseOffset], ubPBase, params);
        // ② 右半（S2 下半）→ L1 CG' 下 8 分形行
        AscendC::DataCopy(l1PBase[aivBaseOffset + lowerDstOffset], ubPBase[rightSrcOffset], params);
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        if constexpr (MODE == 4U) {
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        if constexpr (MODE == 4U) {
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <class TensorP>
    __aicore__ inline
    void operator()(TensorP &l1PTensorTla, GemmCoord actualBlockShape,
        uint32_t isFirstKvSTile, uint32_t ubSBufId, uint32_t l1PBufId,
        Arch::CrossCoreFlag mm1ToSmFlag, Arch::CrossCoreFlag smToMm2Flag)
    {
        // DN 下两个 AIV 沿 S1 对半切核：单 AIV 处理 m 个 S1 列（典型 64）、全部 S2 列
        uint32_t mCopyOffset = RoundUp(actualBlockShape.m(), 8) / 2;
        uint32_t m = actualBlockShape.m() < mCopyOffset ? actualBlockShape.m() : mCopyOffset;
        m = subBlockIdx_ == 0 ? m : actualBlockShape.m() - m;
        if (m == 0) {
            WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            return;
        }
        uint32_t n = actualBlockShape.n();             // 当前 tile 有效 S2 行数（尾块可 < 256）
        uint16_t mRound = RoundUp(m, C0_NUM_PER_FRACTAL);
        // [128,128] 伪宽行 NZ：M 维 128 行 + 1 行 padding，blockStride = ubN>>1 | 0x1 = 129
        uint32_t blockStride = (SM_COL_MAX_ELEM_NUM >> 1) | 0x1u;

        __ubuf__ ElementOutput *pAddr = (__ubuf__ ElementOutput*) lpUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM].GetPhyAddr();
        __ubuf__ ElementInput *sAddr = (__ubuf__ ElementInput*) lsUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM].GetPhyAddr();
        __ubuf__ float *globalMaxAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *globalSumAddr = (__ubuf__ float*) glUbTensor.GetPhyAddr();
        __ubuf__ float *curMaxAddr = (__ubuf__ float*) lmUbFloatTensor.GetPhyAddr();
        __ubuf__ float *curSumAddr = (__ubuf__ float*) llUbFloatTensor.GetPhyAddr();
        __ubuf__ float *expMaxAddr = (__ubuf__ float *)dmUbTensor[l1PBufId * DM_UB_GLOBAL_ELEM_NUM].GetPhyAddr();

        // 首 tile 直接把列 max / 列 sum 写到 global 缓冲；后续 tile 写临时缓冲再做 flash 更新
        __ubuf__ float *maxDstAddr = isFirstKvSTile ? globalMaxAddr : curMaxAddr;
        __ubuf__ float *sumDstAddr = isFirstKvSTile ? globalSumAddr : curSumAddr;

        // wait QK Fixpipe finish
        WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);

        // —— Max pass：BF16 packed 域沿 S2 in-lane 累积，末尾升 FP32 并折叠 scale ——
        ComputeColMaxDn(sAddr, maxDstAddr, n, static_cast<uint16_t>(m), scaleValueF);
        // flash：newMax = max(curMax, oldGlobalMax)，expMax = exp(oldGlobalMax - newMax)
        if (!isFirstKvSTile) {
            UpdateColMaxExpMax(curMaxAddr, globalMaxAddr, expMaxAddr);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);

        // —— Sum/Exp/P pass：逐寄存器升 FP32，列向 in-lane 累加；P 降 BF16 落 [128,128] NZ ——
        ComputeExpSubSumDn(pAddr, sAddr, globalMaxAddr, sumDstAddr,
            static_cast<uint16_t>(m), scaleValueF, blockStride);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);

        // Stage 3：UB→L1 两次 DataCopy 解折叠——把 §2.4 的 [128,128] 伪宽行 NZ 还原成 P'[S2,S1] 自然 NZ
        // （右半=S2 下半 → 落到左半正下方）；不做转置，转置由 Cube 的 isTransposeA load 完成
        WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
        UnfoldPUbToPL1(l1PTensorTla.data(), lpUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM],
            static_cast<uint32_t>(mRound), mCopyOffset, blockStride);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);

        // flash：newSum = expMax * oldGlobalSum + curSum
        if (!isFirstKvSTile) {
            UpdateColSum(curSumAddr, globalSumAddr, expMaxAddr);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

private:
    float scaleValueF;
    AscendC::LocalTensor<ElementInput> lsUbTensor;
    AscendC::LocalTensor<ElementOutput> lpUbTensor;
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> dmUbTensor;
    AscendC::LocalTensor<float> lmUbFloatTensor;
    AscendC::LocalTensor<float> llUbFloatTensor;
    uint32_t subBlockIdx_;
    ElementInput MIN_VALUE;

    // 沿 S2 求每个 S1 列的 max（in-lane）：BF16 packed 域累积 → 末尾升 FP32 合并 → 折叠 scale
    __simd_vf__ inline void ComputeColMaxDn(
        __ubuf__ ElementInput *srcUb, __ubuf__ float *maxUb,
        uint32_t n, uint16_t m, float dScaleF)
    {
        using namespace AscendC::MicroAPI;
        // widening bf16->fp32：偶 lane / 奇 lane 各取 64 个升精度（配 Interleave 复原顺序）
        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        constexpr static CastTrait castTraitOne = {
            RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

        RegTensor<ElementInput> minVreg;
        RegTensor<ElementInput> srcVreg;
        RegTensor<ElementInput> accVreg;
        RegTensor<float> maxEvenF;
        RegTensor<float> maxOddF;
        RegTensor<float> maxF;
        MaskReg pregFull = CreateMask<ElementInput, MaskPattern::ALL>();       // 128 bf16
        MaskReg pregFloatFull = CreateMask<float, MaskPattern::ALL>();         // 64 fp32
        MaskReg pregRow = UpdateMask<ElementInput>(m);                         // 单行 m 个 bf16

        // 多算少搬：脏 S2 行 [n, 256) 填 MIN，保证不污染列 max / 列 sum
        Duplicate(minVreg, MIN_VALUE);
        for (uint16_t r = static_cast<uint16_t>(n); r < SM_COL_MAX_ELEM_NUM; ++r) {
            StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(srcUb + r * m, minVreg, pregRow);
        }
        mem_bar(VST_VLD);

        // 每次 LoadAlign 读 128 bf16 = 相邻两个 S2 行（lane0..63=行2k, lane64..127=行2k+1）
        Duplicate(accVreg, MIN_VALUE);
        for (uint16_t k = 0; k < SM_COL_MAX_ELEM_NUM / 2; ++k) {
            LoadAlign(srcVreg, srcUb + static_cast<uint32_t>(k) * 2u * m);
            Max(accVreg, accVreg, srcVreg, pregFull);
        }
        // accVreg[0:64]=偶数 S2 行 max，accVreg[64:128]=奇数 S2 行 max；升精度并复原顺序后取大
        Cast<float, ElementInput, castTraitZero>(maxEvenF, accVreg, pregFull);
        Cast<float, ElementInput, castTraitOne>(maxOddF, accVreg, pregFull);
        Interleave(maxEvenF, maxOddF, maxEvenF, maxOddF);
        Max(maxF, maxEvenF, maxOddF, pregFloatFull);
        Muls(maxF, maxF, dScaleF, pregFloatFull);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb, maxF, pregFloatFull);
    }

    // flash 更新列 max 与 expMax：newMax=max(cur,old)，expMax=exp(old-newMax)
    __simd_vf__ inline void UpdateColMaxExpMax(
        __ubuf__ float *curMaxUb, __ubuf__ float *globalMaxUb, __ubuf__ float *expMaxUb)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> curVreg;
        RegTensor<float> oldVreg;
        RegTensor<float> newVreg;
        RegTensor<float> expMaxVreg;
        MaskReg pregFloatFull = CreateMask<float, MaskPattern::ALL>();
        LoadAlign(curVreg, curMaxUb);
        LoadAlign(oldVreg, globalMaxUb);
        Max(newVreg, curVreg, oldVreg, pregFloatFull);
        FusedExpSub(expMaxVreg, oldVreg, newVreg, pregFloatFull);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(globalMaxUb, newVreg, pregFloatFull);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(expMaxUb, expMaxVreg, pregFloatFull);
    }

    // flash 更新列 sum：newSum = expMax * oldSum + curSum
    __simd_vf__ inline void UpdateColSum(
        __ubuf__ float *curSumUb, __ubuf__ float *globalSumUb, __ubuf__ float *expMaxUb)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> curVreg;
        RegTensor<float> oldVreg;
        RegTensor<float> expMaxVreg;
        MaskReg pregFloatFull = CreateMask<float, MaskPattern::ALL>();
        LoadAlign(curVreg, curSumUb);
        LoadAlign(oldVreg, globalSumUb);
        LoadAlign(expMaxVreg, expMaxUb);
        Mul(oldVreg, oldVreg, expMaxVreg, pregFloatFull);
        Add(oldVreg, oldVreg, curVreg, pregFloatFull);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(globalSumUb, oldVreg, pregFloatFull);
    }

    // 沿 S2 求 exp(scale*(s - max)) 并列向 in-lane 累加 sum；P 降 BF16 按 [128,128] NZ 落 UB
    __simd_vf__ inline void ComputeExpSubSumDn(
        __ubuf__ ElementOutput *expUb, __ubuf__ ElementInput *srcUb,
        __ubuf__ float *maxUb, __ubuf__ float *sumUb,
        uint16_t m, float dScaleF, uint32_t blockStride)
    {
        using namespace AscendC::MicroAPI;
        // 路径 A：DIST_UNPACK_B16 装载（保序，每行 64 个 bf16 解压到 64 个 fp32 槽）+ widening Cast
        constexpr static CastTrait castTraitB16ToF32 = {
            RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        // narrowing fp32->bf16：写偶数半区，两行各一次后由 DeInterleave 拼成连续 pack
        constexpr static CastTrait castTraitZeroDown = {
            RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_ROUND};

        // 上半（象限 0，S2 行 [0,128)）行 j ⊕ 下半（象限 1，S2 行 [128,256)）同号行 j+128 → 一个 pack
        constexpr uint16_t HALF = SM_COL_MAX_ELEM_NUM / 2;     // 128
        // x_exp_1 路落奇数 NZ 行；与 x_exp 路按 blockStride 跳空互填吃满 8 个 DataBlock
        __ubuf__ ElementOutput *expUb1 = expUb + SM_COL_MAX_ELEM_NUM * 4;

        RegTensor<ElementInput> uReg0;
        RegTensor<ElementInput> uReg1;
        RegTensor<ElementInput> lReg0;
        RegTensor<ElementInput> lReg1;
        RegTensor<float> uF0;
        RegTensor<float> uF1;
        RegTensor<float> lF0;
        RegTensor<float> lF1;
        RegTensor<float> eU0;
        RegTensor<float> eU1;
        RegTensor<float> eL0;
        RegTensor<float> eL1;
        RegTensor<float> maxVreg;
        RegTensor<float> sumAcc;
        RegTensor<ElementOutput> evU0;
        RegTensor<ElementOutput> evU1;
        RegTensor<ElementOutput> evL0;
        RegTensor<ElementOutput> evL1;
        RegTensor<ElementOutput> pack0;
        RegTensor<ElementOutput> pack0a;
        RegTensor<ElementOutput> pack1;
        RegTensor<ElementOutput> pack1a;

        MaskReg pregFloatFull = CreateMask<float, MaskPattern::ALL>();         // 64 fp32
        uint32_t storeLen = HALF_REP_SIZE;
        MaskReg pregStore = UpdateMask<uint16_t>(storeLen);                    // 128（一个 pack 宽）

        // 列 max 对所有 S2 行广播相同（lane=S1，64 个）
        LoadAlign(maxVreg, maxUb);
        Duplicate(sumAcc, 0.0f);
        for (uint16_t i = 0; i < HALF / 2; ++i) {
            // 入口：上/下半各两个相邻行，DIST_UNPACK_B16 + Cast 升出 4 个保序 FP32 行
            LoadAlign<ElementInput, LoadDist::DIST_UNPACK_B16>(uReg0, srcUb + static_cast<uint32_t>(2u * i) * m);
            LoadAlign<ElementInput, LoadDist::DIST_UNPACK_B16>(uReg1, srcUb + static_cast<uint32_t>(2u * i + 1u) * m);
            LoadAlign<ElementInput, LoadDist::DIST_UNPACK_B16>(lReg0, srcUb + static_cast<uint32_t>(2u * i + HALF) * m);
            LoadAlign<ElementInput, LoadDist::DIST_UNPACK_B16>(lReg1, srcUb + static_cast<uint32_t>(2u * i + HALF + 1u) * m);
            Cast<float, ElementInput, castTraitB16ToF32>(uF0, uReg0, pregFloatFull);
            Cast<float, ElementInput, castTraitB16ToF32>(uF1, uReg1, pregFloatFull);
            Cast<float, ElementInput, castTraitB16ToF32>(lF0, lReg0, pregFloatFull);
            Cast<float, ElementInput, castTraitB16ToF32>(lF1, lReg1, pregFloatFull);

            Muls(uF0, uF0, dScaleF, pregFloatFull);
            Muls(uF1, uF1, dScaleF, pregFloatFull);
            Muls(lF0, lF0, dScaleF, pregFloatFull);
            Muls(lF1, lF1, dScaleF, pregFloatFull);

            FusedExpSub(eU0, uF0, maxVreg, pregFloatFull);
            FusedExpSub(eU1, uF1, maxVreg, pregFloatFull);
            FusedExpSub(eL0, lF0, maxVreg, pregFloatFull);
            FusedExpSub(eL1, lF1, maxVreg, pregFloatFull);

            // 列向 in-lane 累加（4 行）；DN 下 sum 已是 per-S1-column，无需 cross-lane ReduceSum
            Add(sumAcc, sumAcc, eU0, pregFloatFull);
            Add(sumAcc, sumAcc, eU1, pregFloatFull);
            Add(sumAcc, sumAcc, eL0, pregFloatFull);
            Add(sumAcc, sumAcc, eL1, pregFloatFull);

            // 出口：上半行 ⊕ 下半同号行 → 连续 pack → 跳写 NZ（一轮 2 个 pack）
            Cast<ElementOutput, float, castTraitZeroDown>(evU0, eU0, pregFloatFull);
            Cast<ElementOutput, float, castTraitZeroDown>(evL0, eL0, pregFloatFull);
            DeInterleave(pack0, pack0a, evU0, evL0);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                (__ubuf__ ElementOutput *&)expUb, pack0, blockStride, REPEAT_STRIDE, pregStore);

            Cast<ElementOutput, float, castTraitZeroDown>(evU1, eU1, pregFloatFull);
            Cast<ElementOutput, float, castTraitZeroDown>(evL1, eL1, pregFloatFull);
            DeInterleave(pack1, pack1a, evU1, evL1);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                (__ubuf__ ElementOutput *&)expUb1, pack1, blockStride, REPEAT_STRIDE, pregStore);
        }
        StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb, sumAcc, pregFloatFull);
    }
};
}

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_BF16_DN_HPP
