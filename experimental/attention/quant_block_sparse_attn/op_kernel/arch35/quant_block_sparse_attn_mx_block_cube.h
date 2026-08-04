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
 * \file quant_block_sparse_attn_mx_block_cube.h
 * \brief QuantBlockSparseAttn 的 MXFP8 全量化 cube 路径。
 */
#ifndef QUANT_BLOCK_SPARSE_ATTN_MX_BLOCK_CUBE_H_
#define QUANT_BLOCK_SPARSE_ATTN_MX_BLOCK_CUBE_H_

#include "kernel_operator.h"
#include "quant_block_sparse_attn_common_arch35.h"
#include "quant_block_sparse_attn_mx_static.h"
#include "quant_block_sparse_attn_mx_static_matmul.h"
#include "common/CopyInL1.h"
#include "common/FixpipeOut.h"
#include "common/util_regbase_mx.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace fa_base_matmul;
using namespace regbasemx;

namespace BaseApi {
template <typename inputType, typename mmType, typename outputType, BSALayout layout, BSALayout kvLayout,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType,
          DTemplateType dVTemplateType, bool isPa, bool useDn = true>
class QuantBlockSparseAttnMxBlockCube {
public:
    using INPUT_T = inputType;
    using SCALE_T = fp8_e8m0_t;
    using MM_T = mmType;

    static constexpr bool USE_DN = useDn;
    static constexpr bool IS_PA = isPa;
    static constexpr BSALayout LAYOUT = layout;
    static constexpr BSALayout KV_LAYOUT = kvLayout;
    static constexpr uint32_t M_BASE = static_cast<uint32_t>(s1TemplateType);
    static constexpr uint32_t S2_BASE = static_cast<uint32_t>(s2TemplateType);
    static constexpr uint32_t S2_SPLIT = 256U;
    static constexpr uint32_t D_BASE = static_cast<uint32_t>(dTemplateType);
    static constexpr uint32_t DV_BASE = static_cast<uint32_t>(dVTemplateType);
    static_assert(LAYOUT == BSALayout::TND && KV_LAYOUT == BSALayout::PA_BNSD && IS_PA,
                  "MX cube currently only supports TND query and PA_BNSD KV");
    static_assert(M_BASE == 128U && S2_BASE == 512U && D_BASE == 128U && DV_BASE == 128U,
                  "MX cube currently only supports S1=128, S2=512, D=128 and DV=128");
    // 每 32 个 fp8 数据元素对应一个 e8m0 scale。
    static constexpr uint32_t MX_SCALE_GROUP = 32U;
    // V scale 为 [blockNum,N,blockSize/64,DV,2]。
    static constexpr uint32_t MX_TOKEN_GROUP = 64U;
    static constexpr uint32_t MX_SCALE_LAST_DIM = 2U;
    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, true};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *qScale, __gm__ uint8_t *kScale, __gm__ uint8_t *vScale)
    {
        if ASCEND_IS_AIC {
            queryGm_.SetGlobalBuffer((__gm__ INPUT_T *)query);
            keyGm_.SetGlobalBuffer((__gm__ INPUT_T *)key);
            valueGm_.SetGlobalBuffer((__gm__ INPUT_T *)value);
            blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);
            qScaleGm_.SetGlobalBuffer((__gm__ SCALE_T *)qScale);
            kScaleGm_.SetGlobalBuffer((__gm__ SCALE_T *)kScale);
            vScaleGm_.SetGlobalBuffer((__gm__ SCALE_T *)vScale);
        }
    }

    __aicore__ inline void InitBuffers()
    {
        if ASCEND_IS_AIC {
            static_assert(USE_DN, "MXFullQuantMode cube currently implements DN copy/load path; "
                                  "use_dn is kept for future non-DN extension");
            // data 与 scale 连续存放，LoadData 在 L1->L0 时完成随路反量化。
            constexpr uint32_t qSize = M_BASE * D_BASE * sizeof(INPUT_T);
            constexpr uint32_t qScaleSize = M_BASE * D_BASE / MX_SCALE_GROUP * sizeof(SCALE_T);
            constexpr uint32_t kvSize = S2_SPLIT * D_BASE * sizeof(INPUT_T);
            constexpr uint32_t kvScaleSize = S2_SPLIT * D_BASE / MX_SCALE_GROUP * sizeof(SCALE_T);
            constexpr uint32_t valueScaleSize =
                (S2_SPLIT / MX_TOKEN_GROUP) * DV_BASE * MX_SCALE_LAST_DIM * sizeof(SCALE_T);

            // Q 双 buffer；K 四 buffer；V 双 buffer。
            constexpr uint32_t qSlotBytes = qSize + qScaleSize;
            constexpr uint32_t kSlotBytes = kvSize + kvScaleSize;
            constexpr uint32_t vSlotBytes = kvSize + valueScaleSize;
            constexpr uint32_t qBytes = 2U * qSlotBytes;
            constexpr uint32_t kBytes = 4U * kSlotBytes;
            constexpr uint32_t vBytes = 2U * vSlotBytes;
            constexpr uint32_t qAddr = MxStatic::CUBE_PRIVATE_L1_BASE;
            constexpr uint32_t kAddr = qAddr + qBytes;
            constexpr uint32_t vAddr = kAddr + kBytes;
            constexpr uint32_t l1End = vAddr + vBytes;
            static_assert(l1End <= MxStatic::L1_CAPACITY, "MX cube L1 layout exceeds capacity");

            l1QTensor_ = LocalTensor<uint8_t>(TPosition::A1, qAddr, qBytes);
            l1KTensor_ = LocalTensor<uint8_t>(TPosition::A1, kAddr, kBytes);
            l1VTensor_ = LocalTensor<uint8_t>(TPosition::A1, vAddr, vBytes);

            // L0A/L0B 双 buffer；L0C 双 buffer。C1 最大输出为 [256,128] fp32，需要 128KB。
            for (uint32_t i = 0U; i < 2U; ++i) {
                l0aTensor_[i] = LocalTensor<uint8_t>(TPosition::A2, i * MxStatic::L0_SLOT_BYTES,
                                                     MxStatic::L0_SLOT_BYTES);
                l0bTensor_[i] = LocalTensor<uint8_t>(TPosition::B2, i * MxStatic::L0_SLOT_BYTES,
                                                     MxStatic::L0_SLOT_BYTES);
                l0cTensor_[i] = LocalTensor<uint8_t>(TPosition::CO1, i * MxStatic::L0C_SLOT_BYTES,
                                                     MxStatic::L0C_SLOT_BYTES);
            }

            SetFlag<HardEvent::MTE1_MTE2>(MxStatic::L1_Q_EVENT_ID0);
            SetFlag<HardEvent::MTE1_MTE2>(MxStatic::L1_Q_EVENT_ID1);
            SetFlag<HardEvent::MTE1_MTE2>(MxStatic::L1_K_EVENT_ID0);
            SetFlag<HardEvent::MTE1_MTE2>(MxStatic::L1_K_EVENT_ID1);
            SetFlag<HardEvent::MTE1_MTE2>(MxStatic::L1_K_EVENT_ID2);
            SetFlag<HardEvent::MTE1_MTE2>(MxStatic::L1_K_EVENT_ID3);
            SetFlag<HardEvent::MTE1_MTE2>(MxStatic::L1_V_EVENT_ID0);
            SetFlag<HardEvent::MTE1_MTE2>(MxStatic::L1_V_EVENT_ID1);
            SetFlag<HardEvent::M_MTE1>(MxStatic::L0A_EVENT_ID0);
            SetFlag<HardEvent::M_MTE1>(MxStatic::L0A_EVENT_ID1);
            SetFlag<HardEvent::M_MTE1>(MxStatic::L0B_EVENT_ID0);
            SetFlag<HardEvent::M_MTE1>(MxStatic::L0B_EVENT_ID1);
            SetFlag<HardEvent::FIX_M>(MxStatic::L0C_EVENT_ID0);
            SetFlag<HardEvent::FIX_M>(MxStatic::L0C_EVENT_ID1);
        }
    }

    __aicore__ inline void FinalizeBuffers()
    {
        if ASCEND_IS_AIC {
            WaitFlag<HardEvent::MTE1_MTE2>(MxStatic::L1_Q_EVENT_ID0);
            WaitFlag<HardEvent::MTE1_MTE2>(MxStatic::L1_Q_EVENT_ID1);
            WaitFlag<HardEvent::MTE1_MTE2>(MxStatic::L1_K_EVENT_ID0);
            WaitFlag<HardEvent::MTE1_MTE2>(MxStatic::L1_K_EVENT_ID1);
            WaitFlag<HardEvent::MTE1_MTE2>(MxStatic::L1_K_EVENT_ID2);
            WaitFlag<HardEvent::MTE1_MTE2>(MxStatic::L1_K_EVENT_ID3);
            WaitFlag<HardEvent::MTE1_MTE2>(MxStatic::L1_V_EVENT_ID0);
            WaitFlag<HardEvent::MTE1_MTE2>(MxStatic::L1_V_EVENT_ID1);
            WaitFlag<HardEvent::M_MTE1>(MxStatic::L0A_EVENT_ID0);
            WaitFlag<HardEvent::M_MTE1>(MxStatic::L0A_EVENT_ID1);
            WaitFlag<HardEvent::M_MTE1>(MxStatic::L0B_EVENT_ID0);
            WaitFlag<HardEvent::M_MTE1>(MxStatic::L0B_EVENT_ID1);
            WaitFlag<HardEvent::FIX_M>(MxStatic::L0C_EVENT_ID0);
            WaitFlag<HardEvent::FIX_M>(MxStatic::L0C_EVENT_ID1);
        }
    }

    __aicore__ inline void IterateBmm1(LocalTensor<uint8_t> &outputBuf, uint32_t outputReadyEventId,
                                       uint32_t outputFreeEventId,
                                       MxRunInfo &runInfo, const MxConstInfo &constInfo, uint32_t subLoop)
    {
        // C1: K[256,D] x Q^T[D,128]；K 由多个 sparse block 拼接。
        const uint32_t s2CalcSize = GetSubLoopS2Size(runInfo, subLoop);

        constexpr uint32_t qSlotBytes = M_BASE * D_BASE * sizeof(INPUT_T) +
                                        M_BASE * D_BASE / MX_SCALE_GROUP * sizeof(SCALE_T);
        uint32_t qSlot = currentQSlot_;
        TEventID qEventId = qSlot == 0U ? MxStatic::L1_Q_EVENT_ID0 : MxStatic::L1_Q_EVENT_ID1;
        LocalTensor<uint8_t> qBuf;
        const uint64_t qScaleOffset = static_cast<uint64_t>(AlignUp32(runInfo.actMSize)) * constInfo.dSize;
        if (unlikely(runInfo.isFirstS2Loop && subLoop == 0U)) {
            // Q 用 ND2NZ，QScale 用 DN2NZ；后续 S2 loop 复用。
            qSlot = qCursor_;
            qCursor_ ^= 1U;
            currentQSlot_ = qSlot;
            qEventId = qSlot == 0U ? MxStatic::L1_Q_EVENT_ID0 : MxStatic::L1_Q_EVENT_ID1;
            qBuf = l1QTensor_[qSlot * qSlotBytes];
            WaitFlag<HardEvent::MTE1_MTE2>(qEventId);
            LocalTensor<INPUT_T> qTensor = qBuf.ReinterpretCast<INPUT_T>();
            CopyToL1Nd2Nz<INPUT_T>(qTensor, queryGm_[runInfo.queryOffset], runInfo.actMSize, constInfo.dSize,
                                   constInfo.n2GD);

            LocalTensor<SCALE_T> qScaleTensor = qBuf.ReinterpretCast<SCALE_T>()[qScaleOffset];
            MxCopyScaleToL1Dn2Nz<SCALE_T>(qScaleTensor, qScaleGm_[runInfo.queryScaleOffset],
                                          constInfo.queryScaleDSize * constInfo.scaleLastDim, runInfo.actMSize,
                                          constInfo.qScaleN1D);
            SetFlag<HardEvent::MTE2_MTE1>(qEventId);
        } else {
            qBuf = l1QTensor_[qSlot * qSlotBytes];
            SetFlag<HardEvent::MTE2_MTE1>(qEventId);
        }

        constexpr uint32_t kSlotBytes = S2_SPLIT * D_BASE * sizeof(INPUT_T) +
                                        S2_SPLIT * D_BASE / MX_SCALE_GROUP * sizeof(SCALE_T);
        const uint32_t kSlot = kCursor_;
        kCursor_ = (kCursor_ + 1U) & 3U;
        const TEventID kEventId = static_cast<TEventID>(MxStatic::L1_K_EVENT_ID0 + kSlot);
        LocalTensor<uint8_t> kBuf = l1KTensor_[kSlot * kSlotBytes];
        WaitFlag<HardEvent::MTE1_MTE2>(kEventId);
        LocalTensor<INPUT_T> kTensor = kBuf.ReinterpretCast<INPUT_T>();
        const uint64_t kScaleOffset = static_cast<uint64_t>(AlignUp32(s2CalcSize)) * constInfo.dSize;
        LocalTensor<SCALE_T> kScaleTensor = kBuf.ReinterpretCast<SCALE_T>()[kScaleOffset];
        // K/KScale 共用 sparse overlap 计算，分别发起 PA data 与 scale 搬运。
        CopyKeyAndScaleToL1(kTensor, kScaleTensor, runInfo, constInfo, s2CalcSize, subLoop);
        SetFlag<HardEvent::MTE2_MTE1>(kEventId);

        WaitFlag<HardEvent::MTE2_MTE1>(kEventId);
        WaitFlag<HardEvent::MTE2_MTE1>(qEventId);

        const uint32_t l0cSlot = l0cCursor_;
        l0cCursor_ ^= 1U;
        const TEventID l0cEventId = l0cSlot == 0U ? MxStatic::L0C_EVENT_ID0 : MxStatic::L0C_EVENT_ID1;
        LocalTensor<MM_T> mm1ResL0C = l0cTensor_[l0cSlot].ReinterpretCast<MM_T>();
        WaitFlag<HardEvent::FIX_M>(l0cEventId);
        // MxMatmulFull 同时消费 fp8 data 与 e8m0 scale。
        MMParam param = MxMakeMMParam(s2CalcSize, runInfo.actMSize, constInfo.dSize, false, true);
        MxStatic::MxMatmulFullStatic<INPUT_T, INPUT_T, MM_T, SCALE_T, SCALE_T, mx_fp8_e4m3_t,
                                     mx_fp8_e4m3_t>(kTensor, qBuf.ReinterpretCast<INPUT_T>(), l0aTensor_, l0bTensor_,
                                                    l0aCursor_, l0bCursor_, mm1ResL0C, param, kScaleTensor,
                                                    qBuf.ReinterpretCast<SCALE_T>()[qScaleOffset]);

        if (unlikely(runInfo.isLastS2Loop && IsLastSubLoop(runInfo, subLoop))) {
            SetFlag<HardEvent::MTE1_MTE2>(qEventId);
        }
        SetFlag<HardEvent::MTE1_MTE2>(kEventId);
        SetFlag<HardEvent::M_FIX>(l0cEventId);
        WaitFlag<HardEvent::M_FIX>(l0cEventId);

        MxStatic::AicWaitUbFree(outputFreeEventId);
        FixpipeMm1(outputBuf.ReinterpretCast<MM_T>(), mm1ResL0C, runInfo, s2CalcSize);
        SetFlag<HardEvent::FIX_M>(l0cEventId);
        MxStatic::AicSetUbReady(outputReadyEventId);
    }

    __aicore__ inline void IterateBmm2(LocalTensor<uint8_t> &outputBuf, uint32_t outputReadyEventId,
                                       uint32_t outputFreeEventId,
                                       LocalTensor<uint8_t> &pBuf, uint32_t pEventId, MxRunInfo &runInfo,
                                       const MxConstInfo &constInfo)
    {
        // C2: P[128,512] x V[512,DV]，P/V 均携带 e8m0 scale。
        MxStatic::AicWaitL1PReady(pEventId);
        constexpr uint64_t pScaleOffset = static_cast<uint64_t>(M_BASE) * S2_BASE;
        if (runInfo.actSingleLoopS2Size > S2_SPLIT) {
            // tail PScale 填 e8m0(1.0)。
            constexpr uint32_t int16Bytes = 2U;
            LocalTensor<int16_t> pScalePadTensor =
                pBuf.ReinterpretCast<int16_t>()[pScaleOffset / int16Bytes +
                                                M_BASE * S2_SPLIT / MX_SCALE_GROUP / int16Bytes];
            InitConstValue(pScalePadTensor, {1, M_BASE * S2_SPLIT / MX_SCALE_GROUP * sizeof(SCALE_T) / 32U, 0, 0x7f7f});
        }
        LocalTensor<SCALE_T> pScaleTensor = pBuf.ReinterpretCast<SCALE_T>()[pScaleOffset];

        const uint32_t l0cSlot = l0cCursor_;
        l0cCursor_ ^= 1U;
        const TEventID l0cEventId = l0cSlot == 0U ? MxStatic::L0C_EVENT_ID0 : MxStatic::L0C_EVENT_ID1;
        LocalTensor<MM_T> mm2ResL0C = l0cTensor_[l0cSlot].ReinterpretCast<MM_T>();
        WaitFlag<HardEvent::FIX_M>(l0cEventId);
        constexpr uint32_t baseK = S2_SPLIT;
        constexpr uint64_t l1BaseKOffset = static_cast<uint64_t>(baseK) * M_BASE;
        constexpr uint64_t l1ScaleOffset = l1BaseKOffset / MX_SCALE_GROUP;
        const uint32_t kLoops = (runInfo.actSingleLoopS2Size + baseK - 1U) / baseK;
        for (uint32_t kIdx = 0U; kIdx < kLoops; ++kIdx) {
            const uint32_t realK = (kIdx == kLoops - 1U) ? runInfo.actSingleLoopS2Size - kIdx * baseK : baseK;
            constexpr uint32_t vSlotBytes = S2_SPLIT * D_BASE * sizeof(INPUT_T) +
                                            (S2_SPLIT / MX_TOKEN_GROUP) * DV_BASE * MX_SCALE_LAST_DIM *
                                                sizeof(SCALE_T);
            const uint32_t vSlot = vCursor_;
            vCursor_ ^= 1U;
            const TEventID vEventId = vSlot == 0U ? MxStatic::L1_V_EVENT_ID0 : MxStatic::L1_V_EVENT_ID1;
            LocalTensor<uint8_t> vBuf = l1VTensor_[vSlot * vSlotBytes];
            WaitFlag<HardEvent::MTE1_MTE2>(vEventId);
            LocalTensor<INPUT_T> vTensor = vBuf.ReinterpretCast<INPUT_T>();
            // V 使用 ND2NZ，tail 补零到 64-token 对齐。
            InitValuePadding(vTensor, realK, constInfo.dSizeV);
            CopyValueToL1(vTensor, runInfo, constInfo, realK, kIdx);

            const uint64_t vScaleOffset = static_cast<uint64_t>(AlignUp64(realK)) * constInfo.dSizeV;
            LocalTensor<SCALE_T> vScaleTensor = vBuf.ReinterpretCast<SCALE_T>()[vScaleOffset];
            // VScale 按 64-token group 寻址。
            CopyValueScaleToL1(vScaleTensor, runInfo, constInfo, realK, kIdx);
            SetFlag<HardEvent::MTE2_MTE1>(vEventId);
            WaitFlag<HardEvent::MTE2_MTE1>(vEventId);

            // Restrict the M dimension for an S1 tail.  With raw static L1 buffers, rows outside actMSize are not
            // producer-owned data and must not be loaded into the MX scale path.  Full 128-row tasks keep exactly
            // the previous parameters.
            MMParam param = MxMakeMMParam(M_BASE, constInfo.dSizeV, AlignUp64(realK), true, false, kIdx == 0U,
                                          kIdx == 0U, 0U, runInfo.actMSize);
            // P 转置，V 保持 KN。
            MxStatic::MxMatmulFullStatic<INPUT_T, INPUT_T, MM_T, SCALE_T, SCALE_T, mx_fp8_e4m3_t,
                                         mx_fp8_e4m3_t>(pBuf.ReinterpretCast<INPUT_T>()[kIdx * l1BaseKOffset],
                                                        vTensor, l0aTensor_, l0bTensor_, l0aCursor_, l0bCursor_,
                                                        mm2ResL0C, param, pScaleTensor[kIdx * l1ScaleOffset],
                                                        vScaleTensor);
            SetFlag<HardEvent::MTE1_MTE2>(vEventId);
        }

        SetFlag<HardEvent::M_FIX>(l0cEventId);
        WaitFlag<HardEvent::M_FIX>(l0cEventId);
        MxStatic::AicWaitUbFree(outputFreeEventId);
        FixpipeMm2(outputBuf.ReinterpretCast<MM_T>(), mm2ResL0C, runInfo, constInfo);
        SetFlag<HardEvent::FIX_M>(l0cEventId);
        MxStatic::AicSetUbReady(outputReadyEventId);
    }

private:
    __aicore__ inline uint32_t AlignUp32(uint32_t value) const { return (value + 31U) >> 5 << 5; }

    __aicore__ inline uint32_t AlignUp64(uint32_t value) const { return (value + 63U) >> 6 << 6; }

    __aicore__ inline uint32_t GetSubLoopS2Size(const MxRunInfo &runInfo, uint32_t subLoop) const
    {
        if (runInfo.actSingleLoopS2Size <= S2_SPLIT) {
            return subLoop == 0U ? runInfo.actSingleLoopS2Size : 0U;
        }
        return subLoop == 0U ? S2_SPLIT : runInfo.actSingleLoopS2Size - S2_SPLIT;
    }

    __aicore__ inline bool IsLastSubLoop(const MxRunInfo &runInfo, uint32_t subLoop) const
    {
        return (runInfo.actSingleLoopS2Size <= S2_SPLIT) || (subLoop != 0U);
    }

    __aicore__ inline uint32_t GetBlockOverlapStart(uint32_t tileStart, uint32_t blockStart) const
    {
        return tileStart > blockStart ? tileStart : blockStart;
    }

    __aicore__ inline uint32_t GetBlockOverlapEnd(uint32_t tileEnd, uint32_t blockEnd) const
    {
        return tileEnd < blockEnd ? tileEnd : blockEnd;
    }

    __aicore__ inline void CopyKeyAndScaleToL1(LocalTensor<INPUT_T> &keyDstTensor,
                                               LocalTensor<SCALE_T> &keyScaleDstTensor, const MxRunInfo &runInfo,
                                               const MxConstInfo &constInfo, uint32_t s2CalcSize, uint32_t subLoop)
    {
        // PA BNBD K 与 KScale 使用相同的 sparse 顺序和 token overlap。
        const uint32_t tileStart = subLoop * S2_SPLIT;
        const uint32_t tileEnd = tileStart + s2CalcSize;
        const uint32_t copyRowNumAlign = AlignUp32(s2CalcSize);
        constexpr uint32_t blockElementCnt = 32U / sizeof(INPUT_T);

        Position startPos;
        startPos.bIdx = runInfo.bIdx;
        startPos.n2Idx = runInfo.n2Idx;
        startPos.dIdx = 0U;

        PAShape keyShape;
        keyShape.blockSize = constInfo.paBlockSize;
        keyShape.headNum = constInfo.n2Size;
        keyShape.headDim = constInfo.dSize;
        keyShape.actHeadDim = constInfo.dSize;
        keyShape.maxblockNumPerBatch = constInfo.maxBlockNumPerBatch;
        keyShape.copyRowNumAlign = copyRowNumAlign;
        keyShape.pageStride = constInfo.paBlockStride;
        keyShape.rowStride = constInfo.dSize;

        // KScale GM 为 [physicalBlock,kvHeads,blockSize,D/64,2]，单行使用 DN2NZ。
        PAShape keyScaleShape;
        keyScaleShape.blockSize = constInfo.paBlockSize;
        keyScaleShape.headNum = constInfo.n2Size;
        keyScaleShape.headDim = constInfo.keyScaleDSize * constInfo.scaleLastDim;
        keyScaleShape.actHeadDim = keyScaleShape.headDim;
        keyScaleShape.maxblockNumPerBatch = constInfo.maxBlockNumPerBatch;
        keyScaleShape.copyRowNumAlign = copyRowNumAlign;
        keyScaleShape.pageStride = constInfo.n2Size * constInfo.kScaleN2D;
        keyScaleShape.rowStride = keyScaleShape.headDim;

        for (uint32_t i = 0U; i < runInfo.sparseBlockCount; ++i) {
            const uint32_t blockRealSize = runInfo.sparseBlockRealSize[i];
            const uint32_t blockStart = runInfo.sparseBlockTileOffset[i];
            const uint32_t blockEnd = blockStart + blockRealSize;
            if (blockRealSize == 0U || blockEnd <= tileStart || blockStart >= tileEnd) {
                continue;
            }

            const uint32_t overlapStart = GetBlockOverlapStart(tileStart, blockStart);
            const uint32_t overlapEnd = GetBlockOverlapEnd(tileEnd, blockEnd);
            const uint32_t copyStartInBlock = overlapStart - blockStart;
            const uint32_t copyRowNum = overlapEnd - overlapStart;

            startPos.s2Offset = runInfo.sparseBlockTokenOffset[i] + copyStartInBlock;
            keyShape.copyRowNum = copyRowNum;
            LocalTensor<INPUT_T> keyBlockDstTensor = keyDstTensor[(overlapStart - tileStart) * blockElementCnt];
            GmCopyInToL1PA<INPUT_T>(keyBlockDstTensor, keyGm_, blockTableGm_, KVLAYOUT::BNBD, keyShape, startPos);

            keyScaleShape.copyRowNum = copyRowNum;
            LocalTensor<SCALE_T> keyScaleBlockDstTensor =
                keyScaleDstTensor[(overlapStart - tileStart) * keyScaleShape.actHeadDim];
            MxGmFlatScaleCopyInToL1PAForDN<SCALE_T>(keyScaleBlockDstTensor, kScaleGm_, blockTableGm_, keyScaleShape,
                                                    startPos);
        }
    }

    __aicore__ inline void CopyValueToL1(LocalTensor<INPUT_T> &dstTensor, const MxRunInfo &runInfo,
                                         const MxConstInfo &constInfo, uint32_t realK, uint32_t kIdx)
    {
        // V 使用与 K 相同的 PA 拼接顺序。
        const uint32_t tileStart = kIdx * S2_SPLIT;
        const uint32_t tileEnd = tileStart + realK;
        constexpr uint32_t blockElementCnt = 32U / sizeof(INPUT_T);

        Position startPos;
        startPos.bIdx = runInfo.bIdx;
        startPos.n2Idx = runInfo.n2Idx;
        startPos.dIdx = 0U;

        PAShape valueShape;
        valueShape.blockSize = constInfo.paBlockSize;
        valueShape.headNum = constInfo.n2Size;
        valueShape.headDim = constInfo.dSizeV;
        valueShape.actHeadDim = constInfo.dSizeV;
        valueShape.maxblockNumPerBatch = constInfo.maxBlockNumPerBatch;
        valueShape.copyRowNumAlign = AlignUp64(realK);
        valueShape.pageStride = constInfo.paBlockStride;
        valueShape.rowStride = constInfo.dSizeV;

        for (uint32_t i = 0U; i < runInfo.sparseBlockCount; ++i) {
            const uint32_t blockRealSize = runInfo.sparseBlockRealSize[i];
            const uint32_t blockStart = runInfo.sparseBlockTileOffset[i];
            const uint32_t blockEnd = blockStart + blockRealSize;
            if (blockRealSize == 0U || blockEnd <= tileStart || blockStart >= tileEnd) {
                continue;
            }

            const uint32_t overlapStart = GetBlockOverlapStart(tileStart, blockStart);
            const uint32_t overlapEnd = GetBlockOverlapEnd(tileEnd, blockEnd);
            const uint32_t copyStartInBlock = overlapStart - blockStart;
            const uint32_t copyRowNum = overlapEnd - overlapStart;

            startPos.s2Offset = runInfo.sparseBlockTokenOffset[i] + copyStartInBlock;
            valueShape.copyRowNum = copyRowNum;
            LocalTensor<INPUT_T> valueDstTensor = dstTensor[(overlapStart - tileStart) * blockElementCnt];
            GmCopyInToL1PA<INPUT_T>(valueDstTensor, valueGm_, blockTableGm_, KVLAYOUT::BNBD, valueShape, startPos);
        }
    }

    __aicore__ inline void CopyValueScaleToL1(LocalTensor<SCALE_T> &dstTensor, const MxRunInfo &runInfo,
                                              const MxConstInfo &constInfo, uint32_t realK, uint32_t kIdx)
    {
        // VScale GM 为 [physicalBlock, kvHeads, blockSize/64, DV, 2]，offset 单位为 64-token group。
        const uint32_t tileStart = kIdx * S2_SPLIT;
        const uint32_t tileEnd = tileStart + realK;
        const uint32_t totalGroupCount = AlignUp64(realK) / MX_TOKEN_GROUP;
        const uint32_t scaleBlockSize = constInfo.paBlockSize / MX_TOKEN_GROUP;
        constexpr uint32_t blockElementCnt = 32U / sizeof(SCALE_T);
        uint32_t copiedGroupCount = 0U;

        Position startPos;
        startPos.bIdx = runInfo.bIdx;
        startPos.n2Idx = runInfo.n2Idx;
        startPos.dIdx = 0U;

        PAShape valueScaleShape;
        valueScaleShape.blockSize = scaleBlockSize;
        valueScaleShape.headNum = constInfo.n2Size;
        valueScaleShape.headDim = constInfo.valueScaleDSize;
        valueScaleShape.actHeadDim = constInfo.valueScaleDSize;
        valueScaleShape.maxblockNumPerBatch = constInfo.maxBlockNumPerBatch;
        valueScaleShape.copyRowNumAlign = totalGroupCount;
        valueScaleShape.pageStride = constInfo.n2Size * constInfo.valueScaleN2D;
        valueScaleShape.rowStride = valueScaleShape.headDim;

        for (uint32_t i = 0U; i < runInfo.sparseBlockCount; ++i) {
            const uint32_t blockRealSize = runInfo.sparseBlockRealSize[i];
            const uint32_t blockStart = runInfo.sparseBlockTileOffset[i];
            const uint32_t blockEnd = blockStart + blockRealSize;
            if (blockRealSize == 0U || blockEnd <= tileStart || blockStart >= tileEnd) {
                continue;
            }

            const uint32_t overlapStart = GetBlockOverlapStart(tileStart, blockStart);
            const uint32_t overlapEnd = GetBlockOverlapEnd(tileEnd, blockEnd);
            const uint32_t copyStartInBlock = overlapStart - blockStart;
            const uint32_t copyRowNum = overlapEnd - overlapStart;
            const uint32_t groupStartInBlock = copyStartInBlock / MX_TOKEN_GROUP;
            const uint32_t groupEndInBlock = (copyStartInBlock + copyRowNum + MX_TOKEN_GROUP - 1U) / MX_TOKEN_GROUP;
            const uint32_t copyGroupNum = groupEndInBlock - groupStartInBlock;

            startPos.s2Offset = runInfo.sparseBlockTokenOffset[i] / MX_TOKEN_GROUP + groupStartInBlock;
            valueScaleShape.copyRowNum = copyGroupNum;
            LocalTensor<SCALE_T> valueScaleDstTensor = dstTensor[copiedGroupCount * blockElementCnt];
            MxGmFlatScaleCopyInToL1PAForND<SCALE_T>(valueScaleDstTensor, vScaleGm_, blockTableGm_, valueScaleShape,
                                                    startPos);
            copiedGroupCount += copyGroupNum;
        }
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<MM_T> &dstTensor, const LocalTensor<MM_T> &l0C,
                                      const MxRunInfo &runInfo, uint32_t s2RealSize)
    {
        // C1 写为 VF 使用的 row-major UB layout。
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = AlignUp32(runInfo.actMSize);
        fixpipeParams.mSize = (s2RealSize + 7U) >> 3 << 3;
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15U) >> 4) << 4U;
        fixpipeParams.dstStride = 64U;
        fixpipeParams.dualDstCtl = 2;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<MM_T, MM_T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void FixpipeMm2(const LocalTensor<MM_T> &dstTensor, const LocalTensor<MM_T> &l0C,
                                      const MxRunInfo &, const MxConstInfo &constInfo)
    {
        // C2 以 fp32 写入 UB，供 V2 update。
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (constInfo.dSizeV + 7U) >> 3 << 3;
        fixpipeParams.mSize = M_BASE;
        fixpipeParams.srcStride = ((M_BASE + 15U) >> 4) << 4U;
        fixpipeParams.dstStride = (DV_BASE + 15U) >> 4 << 4;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<MM_T, MM_T, BMM2_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void InitValuePadding(const LocalTensor<INPUT_T> &valueL1, uint32_t realK, uint32_t curN)
    {
        // V tail 补零到 64-token 对齐。
        const uint32_t curPadK = AlignUp64(realK);
        if (likely(curPadK == realK)) {
            return;
        }
        InitConstValueParams<half> initConstValueParams;
        initConstValueParams.repeatTimes = AlignUp64(curN) >> 5;
        initConstValueParams.blockNum = curPadK - realK;
        initConstValueParams.dstGap = realK;
        initConstValueParams.initValue = 0;
        InitConstValue(valueL1.template ReinterpretCast<half>()[realK << 4U], initConstValueParams);
    }

    GlobalTensor<INPUT_T> queryGm_;
    GlobalTensor<INPUT_T> keyGm_;
    GlobalTensor<INPUT_T> valueGm_;
    GlobalTensor<SCALE_T> qScaleGm_;
    GlobalTensor<SCALE_T> kScaleGm_;
    GlobalTensor<SCALE_T> vScaleGm_;
    GlobalTensor<int32_t> blockTableGm_;

    LocalTensor<uint8_t> l1QTensor_;
    LocalTensor<uint8_t> l1KTensor_;
    LocalTensor<uint8_t> l1VTensor_;
    LocalTensor<uint8_t> l0aTensor_[2];
    LocalTensor<uint8_t> l0bTensor_[2];
    LocalTensor<uint8_t> l0cTensor_[2];
    uint32_t qCursor_ = 0U;
    uint32_t currentQSlot_ = 0U;
    uint32_t kCursor_ = 0U;
    uint32_t vCursor_ = 0U;
    uint32_t l0aCursor_ = 0U;
    uint32_t l0bCursor_ = 0U;
    uint32_t l0cCursor_ = 0U;
};
} // namespace BaseApi

#endif // QUANT_BLOCK_SPARSE_ATTN_MX_BLOCK_CUBE_H_
