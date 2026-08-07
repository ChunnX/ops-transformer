/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_S_TRANS_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_S_TRANS_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/epilogue/bsa_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/bsa_epilogue_tile_copy.hpp"
#include "../../../attn_infra/epilogue/block/block_epilogue_arch35_utils.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/bsa_matrix_coord.hpp"
#include "../../../tla/tensor.hpp"
#include "../../../tla/layout.hpp"

namespace NpuArch::Epilogue::Block {

template <
    class OutputType_,
    class LayoutS_,
    bool rescaleOnCube_>
class BlockEpilogue<
    // S is transposed, has shape [kvSTile, qSTile], softmax is done columnwise
    // P would be rearranged from [kvSTile, qSTile] to zN,
    // which can be viewed as from [qSTile, kvSTile] to nZ
    EpilogueOnlineSoftmaxBsa<true, rescaleOnCube_>,
    OutputType_,
    Gemm::GemmType<half, LayoutS_>>
{
public:
    using DispatchPolicy = EpilogueOnlineSoftmaxBsa<true>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = half;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = LayoutS_;

    static constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
    static constexpr uint32_t ELE_NUM_PER_C0 = 16;
    static constexpr uint32_t ELE_NUM_PER_C0_FP8 = 32;
    static constexpr uint32_t ELE_NUM_PER_DATABLOCK = 32 / sizeof(ElementOutput);

    static constexpr uint32_t UB_S_P_BUF_STAGES = 2;
    static constexpr uint32_t UB_DM_BUF_MAX_STAGES = 3;

    static constexpr half MIN_VALUE = -65504.0f;

    static constexpr bool FULL_QUANT_FP8 = AscendC::IsSameType<ElementOutput, fp8_e4m3fn_t>::value;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue_,
                                    UBufTileHelper &uBufTileHelper)
    {
        subBlockIdx_ = AscendC::GetSubBlockIdx();
        scaleValue = static_cast<ElementInput>(scaleValue_);
        uint32_t pExtraElemNum = Max(uBufTileHelper.qBaseTilePerSubCore, uBufTileHelper.kvBaseTilePerSubCore);
        for (uint32_t i = 0; i < UB_S_P_BUF_STAGES; i++) {
            lsUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementInput>(
                uBufTileHelper.sStartOffset +
                uBufTileHelper.qBaseTilePerSubCore * uBufTileHelper.kvBaseTilePerSubCore * sizeof(ElementInput) * i);
            lpUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOutput>(
                uBufTileHelper.pStartOffset +
                (uBufTileHelper.qBaseTilePerSubCore * uBufTileHelper.kvBaseTilePerSubCore + pExtraElemNum) *
                sizeof(ElementOutput) * i);
        }
        if constexpr (rescaleOnCube_) {
            // rescaleOnCube on cube：dm 向量仍驻留 UB（先算 dm 再生成 diag）；
            // diag 块中转 [qS/2, qS/2] fp32 3 份轮转（与 L1 diag 3 份、dmUb 3 份同节奏，索引 l1PBufId）
            for (uint32_t i = 0; i < UB_DM_BUF_MAX_STAGES; i++) {
                diagUbTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                    uBufTileHelper.diagStartOffset +
                    uBufTileHelper.qBaseTilePerSubCore * uBufTileHelper.qBaseTilePerSubCore * sizeof(float) * i);
            }
        }
        for (uint32_t i = 0; i < UB_DM_BUF_MAX_STAGES; i++) {
            dmUbTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                uBufTileHelper.dmStartOffset +
                uBufTileHelper.qBaseTilePerSubCore * sizeof(float) * i);
        }
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.gmStartOffset);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.glStartOffset);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(uBufTileHelper.lmStartOffset);
        llUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(uBufTileHelper.llStartOffset);
    }

    __aicore__ inline
    ~BlockEpilogue()
    {
    }

    template <class TensorP>
    __aicore__ inline
    void operator()(TensorP &l1PTensorTla, GemmCoord actualBlockShape,
        uint32_t isFirstKvSTile, uint32_t ubSBufId, uint32_t l1PBufId,
         Arch::CrossCoreFlag mm1ToSmFlag, Arch::CrossCoreFlag smToMm2Flag,
         AscendC::LocalTensor<float> l1DiagTensor = AscendC::LocalTensor<float>(),
         float vScale = 1.0f)
    {
        uint32_t nCopyOffset =
            FULL_QUANT_FP8 ? RoundUp(actualBlockShape.m(), 64) / 2 : RoundUp(actualBlockShape.m(), 32) / 2;
        uint32_t n = actualBlockShape.m() < nCopyOffset ? actualBlockShape.m() : nCopyOffset;
        n = subBlockIdx_ == 0 ? n : actualBlockShape.m() - n;
        if (n == 0) {
            WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            if constexpr (rescaleOnCube_) {
                // diag flag 双向转发：AIV1 分到 0 行（rowNum <= nCopyOffset 的尾块）时主路径的
                // wait/write/set 被跳过，这里仍需 wait（消费 CUBE 的 consumed，保持 mode-4 交替）
                // 并 set（CUBE 的 RE 会双 AIV wait ready，否则 rowNum<=64 时 CUBE 等不到 AIV1 的
                // ready 而死锁）；无数据可写，仅转发 flag
                if (!isFirstKvSTile) {
                    Arch::CrossCoreFlag diagFlag(6 + l1PBufId);
                    WaitCrossCoreSync<4, PIPE_MTE3>(diagFlag);
                    SetCrossCoreSync<4, PIPE_MTE3>(diagFlag);
                }
            }
            return;
        }
        uint32_t m = actualBlockShape.n();
        uint16_t mRound = RoundUp(m, C0_NUM_PER_FRACTAL);
        uint16_t nRound = FULL_QUANT_FP8 ? RoundUp(n, ELE_NUM_PER_C0_FP8) : RoundUp(n, ELE_NUM_PER_C0);
        uint32_t blockStride = mRound / 2 + 1;
        constexpr int16_t vlSize = static_cast<int16_t>(AscendC::GetVecLen() / sizeof(ElementInput));
        uint32_t tailN = (n - 1) % vlSize + 1;
        uint16_t s2BaseSize = 64;

        uint32_t mAlignedTile = mRound / 2;
        uint32_t mFirstTile = m % mAlignedTile;
        uint32_t mAligned16TileNum = m / mAlignedTile;
        if (mAligned16TileNum == 2) {
            mFirstTile = mAlignedTile;
        }
        __ubuf__ ElementInput *sUbAddr = (__ubuf__ ElementInput *)lsUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ ElementOutput *pUbAddr = (__ubuf__ ElementOutput *)lpUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ float *gmUbAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *gmUbStartAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *glUbAddr = (__ubuf__ float *)glUbTensor.GetPhyAddr();
        __ubuf__ half *lmUbAddr = (__ubuf__ half *)lmUbTensor.GetPhyAddr();
        __ubuf__ half *lmUbStartAddr = (__ubuf__ half *)lmUbTensor.GetPhyAddr();
        __ubuf__ half *llUbAddr = (__ubuf__ half *)llUbTensor.GetPhyAddr();
        __ubuf__ float *dmUbAddr = (__ubuf__ float *)dmUbTensor[l1PBufId].GetPhyAddr();
        // wait QK Fixpipe finish
        WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);

        // AscendC::printf("tkd lsUbTensor[ubSBufId]\n");
        // AscendC::DumpTensor(lsUbTensor[ubSBufId], 1, 128);

        ComputeLocalMax(sUbAddr, lmUbAddr, m, s2BaseSize, tailN, scaleValue);

        // AscendC::printf("tkd lmUbTensor\n");
        // AscendC::DumpTensor(lmUbTensor, 1, 32);

        if (!isFirstKvSTile) {
            UpdateHistoricalMax(lmUbAddr, gmUbAddr, tailN);
            // AscendC::printf("tkd historical max\n");
            // AscendC::DumpTensor(lmUbTensor, 1, 32);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);

        // AscendC::printf("tkd FULL_QUANT_FP8: %u\n", static_cast<uint32_t>(FULL_QUANT_FP8));
        if constexpr (FULL_QUANT_FP8) {
            if (mAligned16TileNum == 0) {
                ComputePAndLocalSumFP8<MAlignedTileNum::Zero>(sUbAddr, pUbAddr, lmUbAddr, llUbAddr, mRound, m,
                                                              mFirstTile, s2BaseSize, tailN, blockStride, scaleValue);
            } else if (mAligned16TileNum == 1) {
                ComputePAndLocalSumFP8<MAlignedTileNum::One>(sUbAddr, pUbAddr, lmUbAddr, llUbAddr, mRound, m,
                                                             mFirstTile, s2BaseSize, tailN, blockStride, scaleValue);
            } else {
                ComputePAndLocalSumFP8<MAlignedTileNum::Two>(sUbAddr, pUbAddr, lmUbAddr, llUbAddr, mRound, m,
                                                             mFirstTile, s2BaseSize, tailN, blockStride, scaleValue);
            }
        } else {
            if (mAligned16TileNum == 0) {
                ComputePAndLocalSum<MAlignedTileNum::Zero>(sUbAddr, pUbAddr, lmUbAddr, llUbAddr, mRound, m, mFirstTile,
                                                           s2BaseSize, tailN, blockStride, scaleValue);
            } else if (mAligned16TileNum == 1) {
                ComputePAndLocalSum<MAlignedTileNum::One>(sUbAddr, pUbAddr, lmUbAddr, llUbAddr, mRound, m, mFirstTile,
                                                          s2BaseSize, tailN, blockStride, scaleValue);
            } else {
                ComputePAndLocalSum<MAlignedTileNum::Two>(sUbAddr, pUbAddr, lmUbAddr, llUbAddr, mRound, m, mFirstTile,
                                                          s2BaseSize, tailN, blockStride, scaleValue);
            }
        }

        // AscendC::printf("tkd lpUbTensor[ubSBufId]\n");
        // AscendC::DumpTensor(lpUbTensor[ubSBufId], 1, 128);
        // AscendC::printf("tkd llUbTensor\n");
        // AscendC::DumpTensor(llUbTensor, 1, 128);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);

        auto ubPLayoutTla = tla::MakeLayout<ElementOutput, LayoutOutput>(mRound, nRound);
        auto ubPTensorTla = tla::MakeTensor(lpUbTensor[ubSBufId], ubPLayoutTla, Arch::PositionUB{});
        auto ubPTensorTlaTile = GetTile(ubPTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto l1PTensorTlaTile =
            GetTile(l1PTensorTla, tla::MakeCoord(subBlockIdx_ * nCopyOffset, 0), tla::MakeShape(m, n));
        WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);

        AscendC::DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = nRound / ELE_NUM_PER_DATABLOCK;
        dataCopyParams.blockLen = mRound;
        dataCopyParams.srcStride = 1;
        dataCopyParams.dstStride = 0;
        DataCopy(l1PTensorTla.data()[subBlockIdx_ * mRound * nCopyOffset], lpUbTensor[ubSBufId], dataCopyParams);

        // // AscendC::printf("tkd blockCount: %u\n", dataCopyParams.blockCount);
        // // AscendC::printf("tkd blockLen: %u\n", dataCopyParams.blockLen);
        // // AscendC::printf("tkd srcStride: %u\n", dataCopyParams.srcStride);
        // // AscendC::printf("tkd dstStride: %u\n", dataCopyParams.dstStride);
        // // AscendC::printf("tkd mRound: %u\n", mRound);
        // // AscendC::printf("tkd nCopyOffset: %u\n", nCopyOffset);
        // // AscendC::printf("tkd blockStride: %u\n", blockStride);
        // // AscendC::printf("tkd DataCopy1 dst offset: %u\n", subBlockIdx_ * mRound * nCopyOffset);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        // crossCoreSync after PIPE_MTE1 move
        SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);

        if (isFirstKvSTile) {
            SetGlobalMaxAndGlobalSum(lmUbAddr, gmUbAddr, llUbAddr, glUbAddr, tailN);
            // AscendC::printf("tkd SetGlobalMaxAndGlobalSum\n");
            // AscendC::printf("tkd gmUbTensor\n");
            // AscendC::DumpTensor(gmUbTensor, 1, 128);
            // AscendC::printf("tkd glUbTensor\n");
            // AscendC::DumpTensor(glUbTensor, 1, 128);
        } else {
            // 先算 dm 存 dmUb（3 份轮转），再据此生成 diag 块
            UpdateDeltaMaxGlobalMaxAndGlobalSum(lmUbAddr, gmUbAddr, llUbAddr, glUbAddr, dmUbAddr, tailN);
            if constexpr (rescaleOnCube_) {
                Arch::CrossCoreFlag diagFlag(6 + l1PBufId);
                // 等上一轮同 stage 的 MTE3 读完成再覆写中转缓冲（P 对称组：与末尾
                // SetFlag<MTE3_V>(EVENT_ID5 + l1PBufId) 成对，kernel InitSyncFlags 已预置初始 token）
                // P 模式：生成前 wait（与 P 的 WaitFlag<MTE3_V>(ubSBufId+2) 在 tile 开始一致）
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5 + l1PBufId);
                // 生成 diag 块 [qS/2, qS/2] fp32（zN 布局，CUBE RE 按 16×16 对角块读取）：
                // 参考 P 的 mAligned16TileNum 分派保证双发射（÷vScale 已并入 dmUb 或寄存器，见 SetDiagonalDeltaMax）
                uint32_t diagDmRound = RoundUp(n, C0_NUM_PER_FRACTAL);
                uint32_t diagDmAlignedTile = diagDmRound / 2;
                uint32_t diagDmFirstTile = n % diagDmAlignedTile;
                uint32_t diagDmTileNum = n / diagDmAlignedTile;
                if (diagDmTileNum == 2) {
                    diagDmFirstTile = diagDmAlignedTile;
                }
                __ubuf__ float *diagUbAddr = (__ubuf__ float *)diagUbTensor[l1PBufId].GetPhyAddr();
                if (diagDmTileNum == 0) {
                    SetDiagonalDeltaMax<MAlignedTileNum::Zero>(diagUbAddr, dmUbAddr, diagDmRound, diagDmFirstTile,
                                                               vScale);
                } else if (diagDmTileNum == 1) {
                    SetDiagonalDeltaMax<MAlignedTileNum::One>(diagUbAddr, dmUbAddr, diagDmRound, diagDmFirstTile,
                                                              vScale);
                } else {
                    SetDiagonalDeltaMax<MAlignedTileNum::Two>(diagUbAddr, dmUbAddr, diagDmRound, diagDmFirstTile,
                                                              vScale);
                }
                // VF 写完成 → MTE3 可读（P 对称组，tile 内自足无需预置；V_MTE3 通道 2/3/4 为 diag 专用）
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + l1PBufId);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + l1PBufId);
                // TLA 封装（与 P 的 P→L1 搬移同风格：表达布局意图，DataCopy 用裸地址 + DataCopyParams）
                auto ubDiagDmLayoutTla = tla::MakeLayout<float, LayoutOutput>(diagDmRound, diagDmRound);
                auto ubDiagDmTensorTla = tla::MakeTensor(diagUbTensor[l1PBufId], ubDiagDmLayoutTla, Arch::PositionUB{});
                auto ubDiagDmTensorTlaTile = GetTile(ubDiagDmTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(n, n));
                auto l1DiagDmLayoutTla = tla::MakeLayout<float, NpuArch::layout::zN>(64, 64);
                auto l1DiagDmTensorTla = tla::MakeTensor(l1DiagTensor, l1DiagDmLayoutTla, Arch::PositionL1{});
                auto l1DiagDmTensorTlaTile =
                    GetTile(l1DiagDmTensorTla, tla::MakeCoord(subBlockIdx_ * 64, 0), tla::MakeShape(n, n));
                // diag flag 双向 token：等 CUBE 消费完同 stage 的上一轮 diag（consumed）
                // P 模式：DataCopy 前 wait（与 P 的 WaitCrossCoreSync(smToMm2Flag) 在 DataCopy 前一致），
                // 保证 mmadPV 转置里的 RE 用完同 stage L1 diag 才覆写；VF 计算不被 CUBE 消费进度阻塞
                WaitCrossCoreSync<4, PIPE_MTE3>(diagFlag);
                // MTE3：diag 中转 -> L1（3 份轮转，与 P 的 P→L1 搬移同款参数；双子核 L1 分块，
                // 本子核块在 stage 内偏移 subBlockIdx × 4096 元素，避免互相覆盖）
                AscendC::DataCopyParams dataCopyParams;
                dataCopyParams.blockCount = diagDmRound / 8;   // 8: element per datablock when float
                dataCopyParams.blockLen = diagDmRound;
                dataCopyParams.srcStride = 1;
                dataCopyParams.dstStride = 0;
                DataCopy(l1DiagTensor[subBlockIdx_ * 64 * 64], diagUbTensor[l1PBufId], dataCopyParams);
                // MTE3 读完成 → 下一轮覆写前 wait（P 对称组）
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5 + l1PBufId);
                // diag 写完成（L1 就绪）：与 P 的 smToMm2 ready 方向对称，CUBE RE 读 diag 前 wait 同一 flag
                SetCrossCoreSync<4, PIPE_MTE3>(diagFlag);
            }
            // AscendC::printf("tkd UpdateDeltaMaxGlobalMaxAndGlobalSum\n");
            // AscendC::printf("tkd gmUbTensor\n");
            // AscendC::DumpTensor(gmUbTensor, 1, 128);
            // AscendC::printf("tkd glUbTensor\n");
            // AscendC::DumpTensor(glUbTensor, 1, 128);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

private:
    ElementInput scaleValue;
    AscendC::LocalTensor<ElementInput> lsUbTensor[UB_S_P_BUF_STAGES];
    AscendC::LocalTensor<ElementOutput> lpUbTensor[UB_S_P_BUF_STAGES];
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> dmUbTensor[UB_DM_BUF_MAX_STAGES];
    // rescaleOnCube on cube：diag 块中转 [qS/2, qS/2] fp32 3 份轮转（生成后 MTE3 搬 L1，索引 l1PBufId）
    AscendC::LocalTensor<float> diagUbTensor[UB_DM_BUF_MAX_STAGES];
    AscendC::LocalTensor<ElementInput> lmUbTensor;
    AscendC::LocalTensor<ElementInput> llUbTensor;
    uint32_t subBlockIdx_;

    enum class MAlignedTileNum {
        Zero = 0,
        One = 1,
        Two = 2,
    };

    __simd_vf__ inline void ComputeLocalMax(__ubuf__ half *sUb, __ubuf__ half *lmUb, uint16_t m, uint16_t s2BaseSize,
                                            uint32_t tailN, half scaleValue)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<half> sVreg0;
        RegTensor<half> sVreg1;
        RegTensor<half> sVreg2;
        RegTensor<half> sVreg3;
        RegTensor<half> sVreg4;

        RegTensor<half> maxVreg0;
        RegTensor<half> maxVreg1;
        RegTensor<half> maxVreg2;
        RegTensor<half> maxVreg3;

        MaskReg maskReg16TailN = UpdateMask<half>(tailN);

        __ubuf__ half *sUb0 = sUb;
        __ubuf__ half *sUb1 = sUb0 + s2BaseSize;
        __ubuf__ half *sUb2 = sUb0 + s2BaseSize * 2;
        __ubuf__ half *sUb3 = sUb0 + s2BaseSize * 3;
        __ubuf__ half *sUb4 = sUb0 + s2BaseSize * (m / 4) * 4;

        Duplicate(maxVreg0, MIN_VALUE);
        Duplicate(maxVreg1, MIN_VALUE);
        Duplicate(maxVreg2, MIN_VALUE);
        Duplicate(maxVreg3, MIN_VALUE);

        for (uint16_t i = 0; i < m / 4; i++) {
            LoadAlign(sVreg0, sUb0 + i * s2BaseSize * 4);
            LoadAlign(sVreg1, sUb1 + i * s2BaseSize * 4);
            LoadAlign(sVreg2, sUb2 + i * s2BaseSize * 4);
            LoadAlign(sVreg3, sUb3 + i * s2BaseSize * 4);
            Max(maxVreg0, maxVreg0, sVreg0, maskReg16TailN);
            Max(maxVreg1, maxVreg1, sVreg1, maskReg16TailN);
            Max(maxVreg2, maxVreg2, sVreg2, maskReg16TailN);
            Max(maxVreg3, maxVreg3, sVreg3, maskReg16TailN);
        }

        Max(maxVreg0, maxVreg0, maxVreg2, maskReg16TailN);
        Max(maxVreg1, maxVreg1, maxVreg3, maskReg16TailN);
        Max(maxVreg0, maxVreg0, maxVreg1, maskReg16TailN);
        for (uint16_t i = 0; i < m % 4; i++) {
            LoadAlign(sVreg4, sUb4 + i * s2BaseSize);
            Max(maxVreg0, maxVreg0, sVreg4, maskReg16TailN);
        }
    
        Muls(maxVreg0, maxVreg0, scaleValue, maskReg16TailN);
        StoreAlign<half, StoreDist::DIST_NORM_B16>(lmUb, maxVreg0, maskReg16TailN);
    }

    __simd_vf__ inline void UpdateHistoricalMax(__ubuf__ half *lmUb, __ubuf__ float *gmUb, uint32_t tailN)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<half> lmVreg;
        RegTensor<float> gmVreg;
        RegTensor<half> gmVregHalf;
        RegTensor<half> gmVregHalf0;
        RegTensor<half> gmVregHalf1;
        RegTensor<half> hmVreg;

        MaskReg maskReg16TailN = UpdateMask<half>(tailN);
        MaskReg maskReg32All = CreateMask<float, MaskPattern::ALL>();

        constexpr static CastTrait castTraitZeroROUND = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        LoadAlign(lmVreg, lmUb);
        LoadAlign(gmVreg, gmUb);
        Cast<half, float, castTraitZeroROUND>(gmVregHalf, gmVreg, maskReg32All);
        DeInterleave(gmVregHalf0, gmVregHalf1, gmVregHalf, gmVregHalf);
        Max(hmVreg, lmVreg, gmVregHalf0, maskReg16TailN);
        StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(lmUb, hmVreg, maskReg16TailN);
    }

    template <MAlignedTileNum mTileNum>
    __simd_vf__ inline void ComputePAndLocalSum(__ubuf__ half *sUb, __ubuf__ ElementOutput *pUb, __ubuf__ half *lmUb,
                                                __ubuf__ half *llUb, uint16_t mRound, uint16_t m, uint32_t mFirstTile,
                                                uint16_t s2BaseSize, uint32_t tailN, uint32_t blockStride,
                                                half scaleValue)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<half> sVreg0;
        RegTensor<half> sVreg1;

        RegTensor<half> pVreg0;
        RegTensor<half> pVreg1;

        RegTensor<half> hmVreg;

        RegTensor<half> sumVreg0;
        RegTensor<half> sumVreg1;

        Duplicate(sumVreg0, 0);
        Duplicate(sumVreg1, 0);

        MaskReg maskReg16TailN = UpdateMask<half>(tailN);
        MaskReg maskReg16VL64 = CreateMask<half, MaskPattern::VL64>();

        __ubuf__ half *pUb0 = pUb;
        __ubuf__ half *pUb1 = pUb + (mRound * 8); // *pUb1 = pUb0 + (mRound / 2 * ELE_NUM_PER_DATABLOCK)

        LoadAlign(hmVreg, lmUb);
        for (uint16_t i = 0; i < mFirstTile; i++) {
            LoadAlign(sVreg0, sUb + i * s2BaseSize);
            LoadAlign(sVreg1, sUb + mRound * s2BaseSize / 2 + i * s2BaseSize);

            Muls(sVreg0, sVreg0, scaleValue, maskReg16TailN);
            Muls(sVreg1, sVreg1, scaleValue, maskReg16TailN);

            Sub(sVreg0, sVreg0, hmVreg, maskReg16TailN);
            Sub(sVreg1, sVreg1, hmVreg, maskReg16TailN);

            Exp(pVreg0, sVreg0, maskReg16TailN);
            Exp(pVreg1, sVreg1, maskReg16TailN);

            StoreAlign<half, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                pUb0, pVreg0, mRound + 1, 1, maskReg16VL64);
            StoreAlign<half, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                pUb1, pVreg1, mRound + 1, 1, maskReg16VL64);

            if constexpr (mTileNum >= MAlignedTileNum::Zero) {
                Add(sumVreg0, sumVreg0, pVreg0, maskReg16TailN);
            }
            if constexpr (mTileNum >= MAlignedTileNum::One) {
                Add(sumVreg1, sumVreg1, pVreg1, maskReg16TailN);
            }
        }
        if constexpr (mTileNum < MAlignedTileNum::Two) {
            for (uint16_t i = mFirstTile; i < mRound / 2; i++) {
                LoadAlign(sVreg0, sUb + i * s2BaseSize);

                Muls(sVreg0, sVreg0, scaleValue, maskReg16TailN);

                Sub(sVreg0, sVreg0, hmVreg, maskReg16TailN);

                Exp(pVreg0, sVreg0, maskReg16TailN);

                StoreAlign<half, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    pUb0, pVreg0, mRound + 1, 1, maskReg16VL64);

                if constexpr (mTileNum > MAlignedTileNum::Zero) {
                    Add(sumVreg0, sumVreg0, pVreg0, maskReg16TailN);
                }
            }
        }
        Add(sumVreg0, sumVreg0, sumVreg1, maskReg16TailN);

        StoreAlign<half, StoreDist::DIST_NORM_B16>(llUb, sumVreg0, maskReg16TailN);
    }


    template <MAlignedTileNum mTileNum>
    __simd_vf__ inline void ComputePAndLocalSumFP8(__ubuf__ half *sUb, __ubuf__ ElementOutput *pUb, __ubuf__ half *lmUb,
                                                __ubuf__ half *llUb, uint16_t mRound, uint16_t m, uint32_t mFirstTile,
                                                uint16_t s2BaseSize, uint32_t tailN, uint32_t blockStride,
                                                half scaleValue)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<half> sVreg0;
        RegTensor<half> sVreg1;

        RegTensor<half> pVreg0;
        RegTensor<half> pVreg1;

        RegTensor<half> pVregScaleMaxValue0;
        RegTensor<half> pVregScaleMaxValue1;

        RegTensor<half> hmVreg;

        RegTensor<float> pVregFloat0;
        RegTensor<float> pVregFloat1;
        RegTensor<float> pVregFloat2;
        RegTensor<float> pVregFloat3;
        
        RegTensor<fp8_e4m3fn_t> pVregFP80;
        RegTensor<fp8_e4m3fn_t> pVregFP81;
        RegTensor<fp8_e4m3fn_t> pVregFP82;
        RegTensor<fp8_e4m3fn_t> pVregFP83;

        RegTensor<fp8_e4m3fn_t> deInterleaveVreg0;
        RegTensor<fp8_e4m3fn_t> deInterleaveVreg1;
        RegTensor<fp8_e4m3fn_t> deInterleaveVreg2;
        RegTensor<fp8_e4m3fn_t> deInterleaveVreg3;

        RegTensor<half> sumVreg0;
        RegTensor<half> sumVreg1;

        Duplicate(sumVreg0, 0);
        Duplicate(sumVreg1, 0);

        MaskReg maskReg8VL64 = CreateMask<uint8_t, MaskPattern::VL64>();
        MaskReg maskReg8All = CreateMask<uint8_t, MaskPattern::ALL>();
        MaskReg maskReg16TailN = UpdateMask<half>(tailN);
        MaskReg maskReg16VL64 = CreateMask<half, MaskPattern::VL64>();
        MaskReg maskReg16ALL = CreateMask<half, MaskPattern::ALL>();
        MaskReg maskReg32ALL = CreateMask<float, MaskPattern::ALL>();

        constexpr static CastTrait castTraitZeroUNKNOWN = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        constexpr static CastTrait castTraitOneUNKNOWN = {
            RegLayout::ONE,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        constexpr static CastTrait castTraitZeroRINT = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT,
        };

        constexpr static CastTrait castTraitTwoRINT = {
            RegLayout::TWO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT,
        };

        __ubuf__ fp8_e4m3fn_t *pUb0 = pUb;
        __ubuf__ fp8_e4m3fn_t *pUb1 = pUb + (mRound * 16); // *pUb1 = pUb0 + (mRound / 2 * ELE_NUM_PER_DATABLOCK)

        constexpr half maxValueFP8 = static_cast<half>(448.0f);

        LoadAlign(hmVreg, lmUb);
        for (uint16_t i = 0; i < mFirstTile; i++) {
            LoadAlign(sVreg0, sUb + i * s2BaseSize);
            LoadAlign(sVreg1, sUb + mRound * s2BaseSize / 2 + i * s2BaseSize);

            Muls(sVreg0, sVreg0, scaleValue, maskReg16TailN);
            Muls(sVreg1, sVreg1, scaleValue, maskReg16TailN);

            Sub(sVreg0, sVreg0, hmVreg, maskReg16TailN);
            Sub(sVreg1, sVreg1, hmVreg, maskReg16TailN);

            Exp(pVreg0, sVreg0, maskReg16TailN);
            Exp(pVreg1, sVreg1, maskReg16TailN);

            Muls(pVregScaleMaxValue0, pVreg0, maxValueFP8, maskReg16TailN);
            Muls(pVregScaleMaxValue1, pVreg1, maxValueFP8, maskReg16TailN);

            Cast<float, half, castTraitZeroUNKNOWN>(pVregFloat0, pVregScaleMaxValue0, maskReg16ALL);
            Cast<float, half, castTraitOneUNKNOWN>(pVregFloat1, pVregScaleMaxValue0, maskReg16ALL);
            Cast<float, half, castTraitZeroUNKNOWN>(pVregFloat2, pVregScaleMaxValue1, maskReg16ALL);
            Cast<float, half, castTraitOneUNKNOWN>(pVregFloat3, pVregScaleMaxValue1, maskReg16ALL);

            Cast<fp8_e4m3fn_t, float, castTraitZeroRINT>(pVregFP80, pVregFloat0, maskReg32ALL);
            Cast<fp8_e4m3fn_t, float, castTraitTwoRINT>(pVregFP81, pVregFloat1, maskReg32ALL);
            Cast<fp8_e4m3fn_t, float, castTraitZeroRINT>(pVregFP82, pVregFloat2, maskReg32ALL);
            Cast<fp8_e4m3fn_t, float, castTraitTwoRINT>(pVregFP83, pVregFloat3, maskReg32ALL);

            Or((RegTensor<uint8_t> &)pVregFP80, (RegTensor<uint8_t> &)pVregFP80, (RegTensor<uint8_t> &)pVregFP81, maskReg8All);
            Or((RegTensor<uint8_t> &)pVregFP82, (RegTensor<uint8_t> &)pVregFP82, (RegTensor<uint8_t> &)pVregFP83, maskReg8All);

            DeInterleave(deInterleaveVreg0, deInterleaveVreg1, pVregFP80, pVregFP80);
            DeInterleave(deInterleaveVreg2, deInterleaveVreg3, pVregFP82, pVregFP82);

            StoreAlign<fp8_e4m3fn_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                pUb0, deInterleaveVreg0, mRound + 1, 1, maskReg8VL64);
            StoreAlign<fp8_e4m3fn_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                pUb1, deInterleaveVreg2, mRound + 1, 1, maskReg8VL64);

            if constexpr (mTileNum >= MAlignedTileNum::Zero) {
                Add(sumVreg0, sumVreg0, pVreg0, maskReg16TailN);
            }
            if constexpr (mTileNum >= MAlignedTileNum::One) {
                Add(sumVreg1, sumVreg1, pVreg1, maskReg16TailN);
            }
        }
        if constexpr (mTileNum < MAlignedTileNum::Two) {
            for (uint16_t i = mFirstTile; i < mRound / 2; i++) {
                LoadAlign(sVreg0, sUb + i * s2BaseSize);

                Muls(sVreg0, sVreg0, scaleValue, maskReg16TailN);

                Sub(sVreg0, sVreg0, hmVreg, maskReg16TailN);

                Exp(pVreg0, sVreg0, maskReg16TailN);

                Muls(pVregScaleMaxValue0, pVreg0, maxValueFP8, maskReg16TailN);

                Cast<float, half, castTraitZeroUNKNOWN>(pVregFloat0, pVregScaleMaxValue0, maskReg16ALL);
                Cast<float, half, castTraitOneUNKNOWN>(pVregFloat1, pVregScaleMaxValue0, maskReg16ALL);

                Cast<fp8_e4m3fn_t, float, castTraitZeroRINT>(pVregFP80, pVregFloat0, maskReg32ALL);
                Cast<fp8_e4m3fn_t, float, castTraitTwoRINT>(pVregFP81, pVregFloat1, maskReg32ALL);

                Or((RegTensor<uint8_t> &)pVregFP80, (RegTensor<uint8_t> &)pVregFP80, (RegTensor<uint8_t> &)pVregFP81,
                   maskReg8All);

                DeInterleave(deInterleaveVreg0, deInterleaveVreg1, pVregFP80, pVregFP80);

                StoreAlign<fp8_e4m3fn_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    pUb0, deInterleaveVreg0, mRound + 1, 1, maskReg8VL64);

                if constexpr (mTileNum > MAlignedTileNum::Zero) {
                    Add(sumVreg0, sumVreg0, pVreg0, maskReg16TailN);
                }
            }
        }
        Add(sumVreg0, sumVreg0, sumVreg1, maskReg16TailN);

        StoreAlign<half, StoreDist::DIST_NORM_B16>(llUb, sumVreg0, maskReg16TailN);
    }
    
    __simd_vf__ inline void SetGlobalMaxAndGlobalSum(__ubuf__ half *lmUb, __ubuf__ float *gmUb, __ubuf__ half *llUb,
                                      __ubuf__ float *glUb, uint32_t tailN)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<half> lmVreg;
        RegTensor<half> lmVreg0;
        RegTensor<half> lmVreg1;
        RegTensor<float> lmVregFloat;

        RegTensor<half> llVreg;
        RegTensor<half> llVreg0;
        RegTensor<half> llVreg1;
        RegTensor<float> llVregFloat;

        MaskReg maskReg16All = CreateMask<half, MaskPattern::ALL>();
        MaskReg maskReg32TailN = UpdateMask<float>(tailN);

        constexpr static CastTrait castTraitZeroUNKNOWN = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        LoadAlign(lmVreg, lmUb);
        Interleave(lmVreg0, lmVreg1, lmVreg, lmVreg);
        Cast<float, half, castTraitZeroUNKNOWN>(lmVregFloat, lmVreg0, maskReg16All);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(gmUb, lmVregFloat, maskReg32TailN);

        LoadAlign(llVreg, llUb);
        Interleave(llVreg0, llVreg1, llVreg, llVreg);
        Cast<float, half, castTraitZeroUNKNOWN>(llVregFloat, llVreg0, maskReg16All);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(glUb, llVregFloat, maskReg32TailN);
    }

    __simd_vf__ inline void UpdateDeltaMaxGlobalMaxAndGlobalSum(__ubuf__ half *lmUb, __ubuf__ float *gmUb, __ubuf__ half *llUb,
                                              __ubuf__ float *glUb, __ubuf__ float *dmUb, uint32_t tailN)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<half> hmVreg;
        RegTensor<half> hmVreg0;
        RegTensor<half> hmVreg1;
        RegTensor<float> hmVregFloat;
        RegTensor<float> gmVreg;
        RegTensor<float> dmVreg;

        RegTensor<half> llVreg;
        RegTensor<half> llVreg0;
        RegTensor<half> llVreg1;
        RegTensor<float> llVregFloat;
        RegTensor<float> glVreg;

        MaskReg maskReg16All = CreateMask<half, MaskPattern::ALL>();
        MaskReg maskReg32TailN = UpdateMask<float>(tailN);

        constexpr static CastTrait castTraitZeroUNKNOWN = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        LoadAlign(hmVreg, lmUb);
        LoadAlign(gmVreg, gmUb);
        Interleave(hmVreg0, hmVreg1, hmVreg, hmVreg);
        Cast<float, half, castTraitZeroUNKNOWN>(hmVregFloat, hmVreg0, maskReg16All);
        FusedExpSub(dmVreg, gmVreg, hmVregFloat, maskReg32TailN);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(dmUb, dmVreg, maskReg32TailN);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(gmUb, hmVregFloat, maskReg32TailN);

        LoadAlign(llVreg, llUb);
        LoadAlign(glVreg, glUb);
        Interleave(llVreg0, llVreg1, llVreg, llVreg);
        Cast<float, half, castTraitZeroUNKNOWN>(llVregFloat, llVreg0, maskReg16All);
        Mul(glVreg, glVreg, dmVreg, maskReg32TailN);
        Add(glVreg, glVreg, llVregFloat, maskReg32TailN);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(glUb, glVreg, maskReg32TailN);
    }

    // rescaleOnCube on cube：由 dm（dmUb 3 份轮转的当前份）生成 diag 块 [qS/2, qS/2] fp32（zN 布局）。
    // 参考 P 的 mAligned16TileNum 分派保证双发射：行 i 与行 i+diagDmRound/2 双发写
    // diagDmUb0（前 diagDmRound/2 行区）/ diagDmUb1（后 diagDmRound/2 行区，偏移 4×diagDmRound 元素）
    // （P 的 pUb0/pUb1 双指针模式，各自 POST_MODE_UPDATE 前进）
    // 注：÷vScale（diag(dm/vScale) 语义）在 LoadAlign 后对 dmVreg 整体 Div 一次（VL64 全宽），
    // 与 CUBE 侧 FixPipe 的 deqScalar=vScale 对消（设计文档 §2.3 记账）
    template <MAlignedTileNum diagDmTileNum>
    __simd_vf__ inline void SetDiagonalDeltaMax(__ubuf__ float *diagonalDmUb, __ubuf__ float *dmUb,
                                                uint32_t diagDmRound, uint32_t diagDmFirstTile,
                                                float vDequantScaleValue)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<float> dmVreg;
        RegTensor<float> diagDmVreg0;
        RegTensor<float> diagDmVreg1;
        RegTensor<float> allZeroVreg;
        RegTensor<float> vScaleVreg;
        Duplicate(allZeroVreg, 0);
        MaskReg maskReg32VL64 = CreateMask<float, MaskPattern::VL64>();

        LoadAlign(dmVreg, dmUb);
        // 标量 vScale 先 Duplicate 成寄存器再真除（Div 全张量形式，无标量重载；VL64 全宽）
        Duplicate(vScaleVreg, vDequantScaleValue);
        Div(dmVreg, dmVreg, vScaleVreg, maskReg32VL64);

        __ubuf__ float *diagDmUb0 = diagonalDmUb;
        __ubuf__ float *diagDmUb1 = diagonalDmUb + 4 * diagDmRound; // diagDmRound/2 * 8(element per datablock)

        for (uint16_t i = 0; i < diagDmFirstTile; i++) {
            uint32_t first0 = i + 1;
            uint32_t first1 = i + 1 + diagDmRound / 2;
            uint32_t second0 = i;
            uint32_t second1 = i + diagDmRound / 2;
            MaskReg maskRegPlusOne0 = UpdateMask<float>(first0);
            MaskReg maskRegPlusOne1 = UpdateMask<float>(first1);
            MaskReg maskReg0 = UpdateMask<float>(second0);
            MaskReg maskReg1 = UpdateMask<float>(second1);
            Select(diagDmVreg0, dmVreg, allZeroVreg, maskRegPlusOne0);
            Select(diagDmVreg1, dmVreg, allZeroVreg, maskRegPlusOne1);
            Select(diagDmVreg0, allZeroVreg, diagDmVreg0, maskReg0);
            Select(diagDmVreg1, allZeroVreg, diagDmVreg1, maskReg1);
            StoreAlign<float, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                diagDmUb0, diagDmVreg0, diagDmRound + 1, 1, maskReg32VL64);
            StoreAlign<float, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                diagDmUb1, diagDmVreg1, diagDmRound + 1, 1, maskReg32VL64);
        }

        if constexpr (diagDmTileNum < MAlignedTileNum::Two) {
            for (uint16_t i = diagDmFirstTile; i < diagDmRound / 2; i++) {
                uint32_t first = i + 1;
                uint32_t second = i;
                MaskReg maskRegPlusOne = UpdateMask<float>(first);
                MaskReg maskReg = UpdateMask<float>(second);
                Select(diagDmVreg0, dmVreg, allZeroVreg, maskRegPlusOne);
                Select(diagDmVreg0, allZeroVreg, diagDmVreg0, maskReg);
                StoreAlign<float, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    diagDmUb0, diagDmVreg0, diagDmRound + 1, 1, maskReg32VL64);
            }
        }
    }
};
}

#endif
