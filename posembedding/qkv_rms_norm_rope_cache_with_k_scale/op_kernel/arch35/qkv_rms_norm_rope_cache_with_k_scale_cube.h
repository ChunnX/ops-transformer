/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CUBE_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CUBE_H_

#include "qkv_rms_norm_rope_cache_with_k_scale_common.h"

namespace QkvRmsNormRopeCacheWithKScale {

constexpr FixpipeConfig QKV_RRC_FIXPIPE_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};
constexpr uint8_t QKV_RRC_FIXPIPE_DUAL_DST_SPLIT_M = 1;

class QkvRmsNormRopeCacheWithKScaleCube {
public:
    __aicore__ inline void Init(const GlobalTensors &tensors, uint64_t qHeadNum, uint64_t kvHeadNum)
    {
        qHeadNum_ = qHeadNum;
        kvHeadNum_ = kvHeadNum;
        l0cBufferUseId_ = 0U;
        BindLocalTensors();
        BindGlobalTensors(tensors);
    }

    __aicore__ inline void InitIntraCoreEvents()
    {
        for (uint32_t bufferId = 0U; bufferId < AIC_L0_BUFFER_NUM; ++bufferId) {
            SetMmadToMte1L0Reusable(bufferId);
        }
    }

    __aicore__ inline void EndIntraCoreEvents()
    {
        for (uint32_t bufferId = 0U; bufferId < AIC_L0_BUFFER_NUM; ++bufferId) {
            WaitMmadToMte1L0Reusable(bufferId);
        }
    }

    __aicore__ inline void PrepareRotationBeforeLoop()
    {
        CopyRotationGmToL1Nz(qRotationL1Nz_, qRotationGm_);
        CopyRotationGmToL1Nz(kRotationL1Nz_, kRotationGm_);
        SetMte2ToMte1RotationReady();
        WaitMte2ToMte1RotationReady();
        LoadRotationToL0(l0b_, qRotationL1Nz_);
        LoadRotationToL0(l0b_[AIC_L0B_ONE_BUFFER_ELEMENTS], kRotationL1Nz_);
        SetMte1ToMmadRotationReady();
        WaitMte1ToMmadRotationReady();
    }

    __aicore__ inline void ComputeTile(const TileParam &tile, const LocalTensor<bfloat16_t> &aRotL1Nz,
                                       uint32_t outputBufferId)
    {
        const uint32_t l0cBufferId =
            static_cast<uint32_t>((l0cBufferUseId_++) & (QKV_RRC_DOUBLE_BUFFER_NUM - 1U));
        LocalTensor<float> qkAfterCubeUb = QkAfterCubeUb(outputBufferId);
        const uint64_t cubeTokenSize = MakeCubeTokenSize(tile);
        const uint64_t qRows = cubeTokenSize * qHeadNum_;
        const uint64_t kRows = cubeTokenSize * kvHeadNum_;
        const uint64_t qL0cOffset = 0U;
        const uint64_t kL0cOffset = AlignUp(qRows, QKV_RRC_NZ_C0) * QKV_RRC_HEAD_DIM_D128;
        ComputePreprocessedQkToL0C(aRotL1Nz, qRows, kRows, l0cBufferId, qL0cOffset, kL0cOffset);
        FixpipePreprocessedSegmentToAivUb(tile, qRows, true, l0cBufferId, qL0cOffset, qkAfterCubeUb);
        FixpipePreprocessedSegmentToAivUb(tile, kRows, false, l0cBufferId, kL0cOffset, qkAfterCubeUb);
    }

private:
    static constexpr uint32_t INTRA_EVENT_ID_MAX = 5U;
    static constexpr uint32_t AIC_L0_BUFFER_NUM = QKV_RRC_DOUBLE_BUFFER_NUM;
    static constexpr uint32_t AIC_L0C_BUFFER_NUM = QKV_RRC_DOUBLE_BUFFER_NUM;
    static constexpr uint32_t AIC_M_L0_TILE = 128U;
    static constexpr uint32_t AIC_L0A_BUFFER_ELEMENTS = 64U * QKV_RRC_KIB / sizeof(bfloat16_t);
    static constexpr uint32_t AIC_L0B_BUFFER_ELEMENTS = 64U * QKV_RRC_KIB / sizeof(bfloat16_t);
    static constexpr uint32_t AIC_L0C_BUFFER_ELEMENTS = 256U * QKV_RRC_KIB / sizeof(float);
    static constexpr uint32_t AIC_L0A_ONE_BUFFER_ELEMENTS = AIC_L0A_BUFFER_ELEMENTS / AIC_L0_BUFFER_NUM;
    static constexpr uint32_t AIC_L0A_ONE_BUFFER_ROWS = AIC_L0A_ONE_BUFFER_ELEMENTS / QKV_RRC_HEAD_DIM_D128;
    static constexpr uint32_t AIC_L0A_SPLIT_D_TILE = 64U;
    static constexpr uint32_t AIC_L0B_ONE_BUFFER_ELEMENTS = AIC_L0B_BUFFER_ELEMENTS / AIC_L0_BUFFER_NUM;
    static constexpr uint32_t AIC_L0C_ONE_BUFFER_ELEMENTS = AIC_L0C_BUFFER_ELEMENTS / AIC_L0C_BUFFER_NUM;
    static constexpr uint8_t AIC_UNIT_FLAG_CHECK_ONLY = 0b10;
    static constexpr uint8_t AIC_UNIT_FLAG_UPDATE = 0b11;
    static constexpr uint32_t EVT_MTE2_TO_MTE1_ROTATION_READY = 5U;
    static constexpr uint32_t EVT_MTE1_TO_MMAD_ROTATION_READY = 5U;
    static constexpr uint32_t EVT_MTE1_TO_MMAD_L0_BASE = 3U;
    static constexpr uint32_t EVT_MMAD_TO_MTE1_L0_BASE = 3U;

    static_assert(AIC_L0B_ONE_BUFFER_ELEMENTS == QKV_RRC_ROTATION_ONE_L1_ELEMENTS,
                  "QkvRmsNormRopeCacheWithKScale each resident L0B rotation segment must hold one D128 matrix.");
    static_assert(AIC_L0A_ONE_BUFFER_ELEMENTS >= AIC_M_L0_TILE * QKV_RRC_HEAD_DIM_D128,
                  "QkvRmsNormRopeCacheWithKScale each L0A pingpong buffer must hold one D128 A tile.");
    static_assert(AIC_L0A_ONE_BUFFER_ROWS == AIC_M_L0_TILE,
                  "QkvRmsNormRopeCacheWithKScale L0A row capacity must match the D128 A tile size.");
    static_assert(AIC_L0A_SPLIT_D_TILE % QKV_RRC_NZ_C0 == 0U,
                  "QkvRmsNormRopeCacheWithKScale L0A split D tile must be aligned to C0.");
    static_assert(EVT_MTE2_TO_MTE1_ROTATION_READY <= INTRA_EVENT_ID_MAX,
                  "QkvRmsNormRopeCacheWithKScale AIC rotation MTE2->MTE1 event exceeds event id range.");
    static_assert(EVT_MTE1_TO_MMAD_ROTATION_READY <= INTRA_EVENT_ID_MAX,
                  "QkvRmsNormRopeCacheWithKScale AIC rotation MTE1->MMAD event exceeds event id range.");
    static_assert(EVT_MTE1_TO_MMAD_L0_BASE + AIC_L0_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
                  "QkvRmsNormRopeCacheWithKScale AIC MTE1->MMAD L0 events exceed event id range.");
    static_assert(EVT_MMAD_TO_MTE1_L0_BASE + AIC_L0_BUFFER_NUM - 1U <= INTRA_EVENT_ID_MAX,
                  "QkvRmsNormRopeCacheWithKScale AIC MMAD->MTE1 L0 events exceed event id range.");

    __aicore__ inline void SetMte2ToMte1RotationReady()
    {
        SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(EVT_MTE2_TO_MTE1_ROTATION_READY));
    }

    __aicore__ inline void WaitMte2ToMte1RotationReady()
    {
        WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(EVT_MTE2_TO_MTE1_ROTATION_READY));
    }

    __aicore__ inline void SetMte1ToMmadRotationReady()
    {
        SetFlag<HardEvent::MTE1_M>(static_cast<event_t>(EVT_MTE1_TO_MMAD_ROTATION_READY));
    }

    __aicore__ inline void WaitMte1ToMmadRotationReady()
    {
        WaitFlag<HardEvent::MTE1_M>(static_cast<event_t>(EVT_MTE1_TO_MMAD_ROTATION_READY));
    }

    __aicore__ inline void WaitMmadToMte1L0Reusable(uint32_t bufferId)
    {
        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(EVT_MMAD_TO_MTE1_L0_BASE + bufferId));
    }

    __aicore__ inline void SetMmadToMte1L0Reusable(uint32_t bufferId)
    {
        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(EVT_MMAD_TO_MTE1_L0_BASE + bufferId));
    }

    __aicore__ inline void SetMte1ToMmadL0Ready(uint32_t bufferId)
    {
        SetFlag<HardEvent::MTE1_M>(static_cast<event_t>(EVT_MTE1_TO_MMAD_L0_BASE + bufferId));
    }

    __aicore__ inline void WaitMte1ToMmadL0Ready(uint32_t bufferId)
    {
        WaitFlag<HardEvent::MTE1_M>(static_cast<event_t>(EVT_MTE1_TO_MMAD_L0_BASE + bufferId));
    }

    __aicore__ inline void CopyRotationGmToL1Nz(const LocalTensor<bfloat16_t> &rotationL1Nz,
                                                const GlobalTensor<bfloat16_t> &rotationGm)
    {
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1U;
        nd2nzParams.nValue = QKV_RRC_HEAD_DIM_D128;
        nd2nzParams.dValue = QKV_RRC_HEAD_DIM_D128;
        nd2nzParams.srcNdMatrixStride = 0U;
        nd2nzParams.srcDValue = QKV_RRC_HEAD_DIM_D128;
        nd2nzParams.dstNzC0Stride = QKV_RRC_HEAD_DIM_D128;
        nd2nzParams.dstNzNStride = 1U;
        nd2nzParams.dstNzMatrixStride = 0U;
        DataCopy(rotationL1Nz, rotationGm, nd2nzParams);
    }

    __aicore__ inline void LoadRotationToL0(const LocalTensor<bfloat16_t> &rotationL0b,
                                            const LocalTensor<bfloat16_t> &rotationL1Nz)
    {
        LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0U;
        loadDataParams.kStartPosition = 0U;
        loadDataParams.mStep = static_cast<uint16_t>(CeilDiv(QKV_RRC_HEAD_DIM_D128, QKV_RRC_NZ_C0));
        loadDataParams.kStep = static_cast<uint16_t>(QKV_RRC_HEAD_DIM_D128 /
            (QKV_RRC_BLOCK_BYTES / sizeof(bfloat16_t)));
        loadDataParams.srcStride = static_cast<int32_t>(CeilDiv(QKV_RRC_HEAD_DIM_D128, QKV_RRC_NZ_C0));
        loadDataParams.dstStride = static_cast<uint16_t>(CeilDiv(QKV_RRC_HEAD_DIM_D128, QKV_RRC_NZ_C0));
        loadDataParams.ifTranspose = true;
        LoadData(rotationL0b, rotationL1Nz, loadDataParams);
    }

    __aicore__ inline void ComputePreprocessedQkToL0C(const LocalTensor<bfloat16_t> &aRotL1Nz, uint64_t qRows,
                                                      uint64_t kRows, uint32_t bufferId, uint64_t qL0cOffset,
                                                      uint64_t kL0cOffset)
    {
        if (qRows == 0U && kRows == 0U) {
            return;
        }

        const LocalTensor<bfloat16_t> kSegmentL1Nz = aRotL1Nz[NzMatrixElements(qRows)];
        uint32_t l0aDbIter = 0U;
        ComputeSegmentToL0C(qRows, aRotL1Nz, l0b_, bufferId, qL0cOffset, l0aDbIter);
        ComputeSegmentToL0C(kRows, kSegmentL1Nz, l0b_[AIC_L0B_ONE_BUFFER_ELEMENTS], bufferId, kL0cOffset, l0aDbIter);
    }

    __aicore__ inline void ComputeSegmentToL0C(uint64_t segmentRows, const LocalTensor<bfloat16_t> &segmentL1Nz,
                                               const LocalTensor<bfloat16_t> &rotationL0b, uint32_t l0cBufferId,
                                               uint64_t l0cOffset, uint32_t &l0aDbIter)
    {
        if (segmentRows == 0U) {
            return;
        }

        const uint64_t dTile = CalcSegmentL0aDTile(segmentRows);
        for (uint64_t dStart = 0U; dStart < QKV_RRC_HEAD_DIM_D128; dStart += dTile) {
            const uint64_t dSize = MinU64(dTile, QKV_RRC_HEAD_DIM_D128 - dStart);
            const bool isFirstD = (dStart == 0U);
            const bool isLastD = (dStart + dSize >= QKV_RRC_HEAD_DIM_D128);
            const uint32_t l0aBufferId = l0aDbIter & (AIC_L0_BUFFER_NUM - 1U);

            WaitMmadToMte1L0Reusable(l0aBufferId);
            LoadAToL0(segmentRows, segmentRows, dStart, dSize, l0aBufferId, 0U, segmentL1Nz);
            SetMte1ToMmadL0Ready(l0aBufferId);
            WaitMte1ToMmadL0Ready(l0aBufferId);
            ComputeMmad(segmentRows, dSize, isFirstD, isLastD, l0aBufferId, l0cBufferId,
                        RotationKChunk(rotationL0b, dStart), l0cOffset, 0U);
            SetMmadToMte1L0Reusable(l0aBufferId);
            ++l0aDbIter;
        }
    }

    __aicore__ inline uint64_t CalcSegmentL0aDTile(uint64_t segmentRows) const
    {
        const uint64_t fullDRequiredElements = NzMatrixElements(segmentRows);
        return fullDRequiredElements <= AIC_L0A_ONE_BUFFER_ELEMENTS ? QKV_RRC_HEAD_DIM_D128 : AIC_L0A_SPLIT_D_TILE;
    }

    __aicore__ inline void FixpipePreprocessedSegmentToAivUb(const TileParam &tile, uint64_t segmentRows,
                                                             bool isQSegment, uint32_t bufferId, uint64_t l0cBaseOffset,
                                                             LocalTensor<float> &qkAfterCubeUb)
    {
        if (segmentRows == 0U) {
            return;
        }

        const uint64_t qRowsPerAiv = MakeCubeHalfTokenSize(tile) * qHeadNum_;
        const uint64_t dstBaseRow = isQSegment ? 0U : qRowsPerAiv;
        FixpipeSegmentToAivUb(segmentRows, dstBaseRow, bufferId, l0cBaseOffset, qkAfterCubeUb);
    }

    __aicore__ inline void LoadAToL0(uint64_t segmentRows, uint64_t mSize, uint64_t dStart, uint64_t dSize,
                                     uint32_t bufferId, uint64_t l0aOffset,
                                     const LocalTensor<bfloat16_t> &segmentL1Nz)
    {
        if (mSize == 0U) {
            return;
        }

        const uint64_t rowSizeAligned = NzRowAligned(segmentRows);
        const uint64_t mSizeAligned = AlignUp(mSize, QKV_RRC_NZ_C0);
        LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0U;
        loadDataParams.kStartPosition = static_cast<uint16_t>(dStart / QKV_RRC_NZ_C0);
        loadDataParams.mStep = static_cast<uint16_t>(CeilDiv(mSize, QKV_RRC_NZ_C0));
        loadDataParams.kStep =
            static_cast<uint16_t>(dSize / (QKV_RRC_BLOCK_BYTES / sizeof(bfloat16_t)));
        loadDataParams.srcStride = static_cast<int32_t>(CeilDiv(rowSizeAligned, QKV_RRC_NZ_C0));
        loadDataParams.dstStride = static_cast<uint16_t>(CeilDiv(mSizeAligned, QKV_RRC_NZ_C0));
        loadDataParams.ifTranspose = false;
        LoadData(l0a_[bufferId * AIC_L0A_ONE_BUFFER_ELEMENTS + l0aOffset], segmentL1Nz, loadDataParams);
    }

    __aicore__ inline void ComputeMmad(uint64_t mSize, uint64_t dSize, bool initC, bool updateUnitFlag,
                                       uint32_t l0aBufferId, uint32_t l0cBufferId,
                                       const LocalTensor<bfloat16_t> &rotationL0b, uint64_t l0cOffset,
                                       uint64_t l0aOffset)
    {
        if (mSize == 0U) {
            return;
        }

        MmadParams params;
        params.m = static_cast<uint16_t>(AlignUp(mSize, QKV_RRC_NZ_C0));
        params.n = static_cast<uint16_t>(QKV_RRC_HEAD_DIM_D128);
        params.k = static_cast<uint16_t>(dSize);
        params.unitFlag = updateUnitFlag ? AIC_UNIT_FLAG_UPDATE : AIC_UNIT_FLAG_CHECK_ONLY;
        params.cmatrixInitVal = initC;
        params.cmatrixSource = false;
        Mmad(l0c_[l0cBufferId * AIC_L0C_ONE_BUFFER_ELEMENTS + l0cOffset],
             l0a_[l0aBufferId * AIC_L0A_ONE_BUFFER_ELEMENTS + l0aOffset], rotationL0b, params);
    }

    __aicore__ inline LocalTensor<bfloat16_t> RotationKChunk(const LocalTensor<bfloat16_t> &rotationL0b,
                                                            uint64_t dStart)
    {
        return rotationL0b[dStart * QKV_RRC_HEAD_DIM_D128];
    }

    __aicore__ inline void FixpipeSegmentToAivUb(uint64_t segmentRows, uint64_t dstBaseRow, uint32_t bufferId,
                                                 uint64_t l0cBaseOffset, LocalTensor<float> &qkAfterCubeUb)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = static_cast<uint16_t>(QKV_RRC_HEAD_DIM_D128);
        fixpipeParams.mSize = static_cast<uint16_t>(AlignUp(segmentRows, QKV_RRC_MIX_AIV_PER_AIC));
        fixpipeParams.srcStride = static_cast<uint16_t>(AlignUp(fixpipeParams.mSize, QKV_RRC_NZ_C0));
        fixpipeParams.dstStride = QKV_RRC_HEAD_DIM_D128;
        fixpipeParams.dualDstCtl = QKV_RRC_FIXPIPE_DUAL_DST_SPLIT_M;
        fixpipeParams.subBlockId = false;
        fixpipeParams.unitFlag = 0b11;
        fixpipeParams.params.ndNum = 1U;
        fixpipeParams.params.srcNdStride = 0U;
        fixpipeParams.params.dstNdStride = 0U;

        Fixpipe<float, float, QKV_RRC_FIXPIPE_ROW_MAJOR_UB>(
            qkAfterCubeUb[dstBaseRow * QKV_RRC_HEAD_DIM_D128],
            l0c_[bufferId * AIC_L0C_ONE_BUFFER_ELEMENTS + l0cBaseOffset], fixpipeParams);
    }

    __aicore__ inline void BindLocalTensors()
    {
        qRotationL1Nz_ =
            LocalTensor<bfloat16_t>(TPosition::TSCM, QKV_RRC_Q_ROTATION_L1_OFFSET, QKV_RRC_ROTATION_ONE_L1_ELEMENTS);
        kRotationL1Nz_ =
            LocalTensor<bfloat16_t>(TPosition::TSCM, QKV_RRC_K_ROTATION_L1_OFFSET, QKV_RRC_ROTATION_ONE_L1_ELEMENTS);
        l0a_ = LocalTensor<bfloat16_t>(TPosition::A2, 0, AIC_L0A_BUFFER_ELEMENTS);
        l0b_ = LocalTensor<bfloat16_t>(TPosition::B2, 0, AIC_L0B_BUFFER_ELEMENTS);
        l0c_ = LocalTensor<float>(TPosition::CO1, 0, AIC_L0C_BUFFER_ELEMENTS);
        outputDbPoolUb_ =
            LocalTensor<float>(TPosition::LCM, QKV_RRC_OUTPUT_DB_POOL_OFFSET, QKV_RRC_OUTPUT_DB_POOL_FLOAT_ELEMENTS);
    }

    __aicore__ inline LocalTensor<float> QkAfterCubeUb(uint32_t bufferId)
    {
        return outputDbPoolUb_[bufferId * QKV_RRC_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS];
    }

    __aicore__ inline void BindGlobalTensors(const GlobalTensors &tensors)
    {
        qRotationGm_.SetGlobalBuffer((__gm__ bfloat16_t *)tensors.qRotation);
        kRotationGm_.SetGlobalBuffer((__gm__ bfloat16_t *)tensors.kRotation);
    }

    uint64_t l0cBufferUseId_;
    uint64_t qHeadNum_;
    uint64_t kvHeadNum_;
    GlobalTensor<bfloat16_t> qRotationGm_;
    GlobalTensor<bfloat16_t> kRotationGm_;
    LocalTensor<bfloat16_t> qRotationL1Nz_;
    LocalTensor<bfloat16_t> kRotationL1Nz_;
    LocalTensor<bfloat16_t> l0a_;
    LocalTensor<bfloat16_t> l0b_;
    LocalTensor<float> l0c_;
    LocalTensor<float> outputDbPoolUb_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CUBE_H_
