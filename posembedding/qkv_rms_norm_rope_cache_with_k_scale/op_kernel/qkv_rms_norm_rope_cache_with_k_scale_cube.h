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

class D128FastCube {
public:
    __aicore__ inline void Init(uint32_t cubeIndex,
        const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData, const GlobalTensors &tensors)
    {
        cubeIndex_ = cubeIndex;
        tilingData_ = tilingData;
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
        LoadRotationToL0(qRotationL0b(), qRotationL1Nz_);
        LoadRotationToL0(kRotationL0b(), kRotationL1Nz_);
        SetMte1ToMmadRotationReady();
        WaitMte1ToMmadRotationReady();
    }

    __aicore__ inline void ComputeTile(const TileParam &tile)
    {
        const uint32_t dataBufferId = MakeDataBufferId(tile.bufferId);
        LocalTensor<bfloat16_t> aRotL1Nz = aRotL1Pool_[dataBufferId * QKV_RRC_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS];
        LocalTensor<float> qkAfterCubeUb = QkAfterCubeUb(dataBufferId);
        const uint32_t qRows = tile.tokenSize * tilingData_->numQHeads;
        const uint32_t kRows = tile.tokenSize * tilingData_->numKHeads;
        const uint32_t qL0cOffset = 0U;
        const uint32_t kL0cOffset = AlignUp(qRows, QKV_RRC_NZ_C0) * QKV_RRC_HEAD_DIM_D128;
        ComputePreprocessedQkToL0C(aRotL1Nz, qRows, kRows, dataBufferId, qL0cOffset, kL0cOffset);
        FixpipePreprocessedSegmentToAivUb(tile, qRows, true, dataBufferId, qL0cOffset, qkAfterCubeUb);
        FixpipePreprocessedSegmentToAivUb(tile, kRows, false, dataBufferId, kL0cOffset, qkAfterCubeUb);
        (void)workspaceGm_;
        (void)cubeIndex_;
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
    static constexpr uint32_t AIC_L0B_ONE_BUFFER_ELEMENTS = AIC_L0B_BUFFER_ELEMENTS / AIC_L0_BUFFER_NUM;
    static constexpr uint32_t AIC_L0C_ONE_BUFFER_ELEMENTS = AIC_L0C_BUFFER_ELEMENTS / AIC_L0C_BUFFER_NUM;
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
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(EVT_MTE2_TO_MTE1_ROTATION_READY));
#endif
    }

    __aicore__ inline void WaitMte2ToMte1RotationReady()
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(EVT_MTE2_TO_MTE1_ROTATION_READY));
#endif
    }

    __aicore__ inline void SetMte1ToMmadRotationReady()
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::MTE1_M>(static_cast<event_t>(EVT_MTE1_TO_MMAD_ROTATION_READY));
#endif
    }

    __aicore__ inline void WaitMte1ToMmadRotationReady()
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::MTE1_M>(static_cast<event_t>(EVT_MTE1_TO_MMAD_ROTATION_READY));
#endif
    }

    __aicore__ inline void WaitMmadToMte1L0Reusable(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(EVT_MMAD_TO_MTE1_L0_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void SetMmadToMte1L0Reusable(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(EVT_MMAD_TO_MTE1_L0_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void SetMte1ToMmadL0Ready(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        SetFlag<HardEvent::MTE1_M>(static_cast<event_t>(EVT_MTE1_TO_MMAD_L0_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void WaitMte1ToMmadL0Ready(uint32_t bufferId)
    {
#ifndef __CCE_KT_TEST__
        WaitFlag<HardEvent::MTE1_M>(static_cast<event_t>(EVT_MTE1_TO_MMAD_L0_BASE + bufferId));
#else
        (void)bufferId;
#endif
    }

    __aicore__ inline void CopyRotationGmToL1Nz(
        const LocalTensor<bfloat16_t> &rotationL1Nz, const GlobalTensor<bfloat16_t> &rotationGm)
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

    __aicore__ inline void LoadRotationToL0(
        const LocalTensor<bfloat16_t> &rotationL0b, const LocalTensor<bfloat16_t> &rotationL1Nz)
    {
        LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0U;
        loadDataParams.kStartPosition = 0U;
        loadDataParams.mStep = CeilDiv(QKV_RRC_HEAD_DIM_D128, QKV_RRC_NZ_C0);
        loadDataParams.kStep = QKV_RRC_HEAD_DIM_D128 / (QKV_RRC_BLOCK_BYTES / sizeof(bfloat16_t));
        loadDataParams.srcStride = CeilDiv(QKV_RRC_HEAD_DIM_D128, QKV_RRC_NZ_C0);
        loadDataParams.dstStride = CeilDiv(QKV_RRC_HEAD_DIM_D128, QKV_RRC_NZ_C0);
        loadDataParams.ifTranspose = true;
        LoadData(rotationL0b, rotationL1Nz, loadDataParams);
    }

    struct L0aQkChunk {
        uint32_t qRows;
        uint32_t kRows;
    };

    __aicore__ inline void ComputePreprocessedQkToL0C(const LocalTensor<bfloat16_t> &aRotL1Nz,
        uint32_t qRows, uint32_t kRows, uint32_t bufferId, uint32_t qL0cOffset, uint32_t kL0cOffset)
    {
        if (qRows == 0U && kRows == 0U) {
            return;
        }

        const LocalTensor<bfloat16_t> kSegmentL1Nz = aRotL1Nz[NzMatrixElements(qRows)];
        uint32_t qStartRow = 0U;
        uint32_t kStartRow = 0U;
        while (qStartRow < qRows || kStartRow < kRows) {
            const L0aQkChunk chunk = CalcL0aQkChunk(qRows - qStartRow, kRows - kStartRow);
            const uint32_t qL0aOffset = 0U;
            const uint32_t kL0aOffset = NzMatrixElements(chunk.qRows);

            WaitMmadToMte1L0Reusable(bufferId);
            LoadAToL0(qRows, qStartRow, chunk.qRows, bufferId, qL0aOffset, aRotL1Nz);
            LoadAToL0(kRows, kStartRow, chunk.kRows, bufferId, kL0aOffset, kSegmentL1Nz);
            SetMte1ToMmadL0Ready(bufferId);
            WaitMte1ToMmadL0Ready(bufferId);
            ComputeMmad(chunk.qRows, bufferId, bufferId, qRotationL0b(),
                qL0cOffset + qStartRow * QKV_RRC_HEAD_DIM_D128, qL0aOffset);
            ComputeMmad(chunk.kRows, bufferId, bufferId, kRotationL0b(),
                kL0cOffset + kStartRow * QKV_RRC_HEAD_DIM_D128, kL0aOffset);
            SetMmadToMte1L0Reusable(bufferId);
            qStartRow += chunk.qRows;
            kStartRow += chunk.kRows;
        }
    }

    __aicore__ inline L0aQkChunk CalcL0aQkChunk(uint32_t qRowsRemaining, uint32_t kRowsRemaining) const
    {
        constexpr uint32_t blockRows = QKV_RRC_NZ_C0;
        constexpr uint32_t capacityBlocks = AIC_L0A_ONE_BUFFER_ROWS / blockRows;
        const uint32_t qBlocksNeeded = CeilDiv(qRowsRemaining, blockRows);
        const uint32_t kBlocksNeeded = CeilDiv(kRowsRemaining, blockRows);
        if (qBlocksNeeded == 0U) {
            return {0U, MinU32(kRowsRemaining, capacityBlocks * blockRows)};
        }
        if (kBlocksNeeded == 0U) {
            return {MinU32(qRowsRemaining, capacityBlocks * blockRows), 0U};
        }
        if (qBlocksNeeded + kBlocksNeeded <= capacityBlocks) {
            return {qRowsRemaining, kRowsRemaining};
        }

        uint32_t qBlocks = (qBlocksNeeded * capacityBlocks) / (qBlocksNeeded + kBlocksNeeded);
        qBlocks = MaxU32(1U, MinU32(qBlocks, qBlocksNeeded));
        uint32_t kBlocks = capacityBlocks - qBlocks;
        if (kBlocks == 0U) {
            kBlocks = 1U;
            --qBlocks;
        }
        if (kBlocks > kBlocksNeeded) {
            const uint32_t extraBlocks = kBlocks - kBlocksNeeded;
            kBlocks = kBlocksNeeded;
            qBlocks = MinU32(qBlocksNeeded, qBlocks + extraBlocks);
        }
        return {MinU32(qRowsRemaining, qBlocks * blockRows), MinU32(kRowsRemaining, kBlocks * blockRows)};
    }

    __aicore__ inline void FixpipePreprocessedSegmentToAivUb(const TileParam &tile, uint32_t segmentRows,
        bool isQSegment, uint32_t bufferId, uint32_t l0cBaseOffset, LocalTensor<float> &qkAfterCubeUb)
    {
        if (segmentRows == 0U) {
            return;
        }

        for (uint32_t mStart = 0U; mStart < segmentRows; mStart += AIC_M_L0_TILE) {
            const uint32_t mSize = MinU32(AIC_M_L0_TILE, segmentRows - mStart);
            FixpipeSegmentToAivUb(tile, mStart, mSize, segmentRows, isQSegment, bufferId,
                l0cBaseOffset + mStart * QKV_RRC_HEAD_DIM_D128, qkAfterCubeUb);
        }
    }

    __aicore__ inline void LoadAToL0(uint32_t segmentRows, uint32_t mStart, uint32_t mSize, uint32_t bufferId,
        uint32_t l0aOffset, const LocalTensor<bfloat16_t> &segmentL1Nz)
    {
        if (mSize == 0U) {
            return;
        }

        const uint32_t rowSizeAligned = NzRowAligned(segmentRows);
        const uint32_t mSizeAligned = AlignUp(mSize, QKV_RRC_NZ_C0);
        LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0U;
        loadDataParams.kStartPosition = 0U;
        loadDataParams.mStep = CeilDiv(mSize, QKV_RRC_NZ_C0);
        loadDataParams.kStep = QKV_RRC_HEAD_DIM_D128 / (QKV_RRC_BLOCK_BYTES / sizeof(bfloat16_t));
        loadDataParams.srcStride = CeilDiv(rowSizeAligned, QKV_RRC_NZ_C0);
        loadDataParams.dstStride = CeilDiv(mSizeAligned, QKV_RRC_NZ_C0);
        loadDataParams.ifTranspose = false;
        LoadData(l0a_[bufferId * AIC_L0A_ONE_BUFFER_ELEMENTS + l0aOffset],
            segmentL1Nz[mStart * QKV_RRC_NZ_C0], loadDataParams);
    }

    __aicore__ inline void ComputeMmad(uint32_t mSize, uint32_t l0aBufferId, uint32_t l0cBufferId,
        const LocalTensor<bfloat16_t> &rotationL0b, uint32_t l0cOffset, uint32_t l0aOffset)
    {
        if (mSize == 0U) {
            return;
        }

        MmadParams params;
        params.m = AlignUp(mSize, QKV_RRC_NZ_C0);
        params.n = QKV_RRC_HEAD_DIM_D128;
        params.k = QKV_RRC_HEAD_DIM_D128;
        params.unitFlag = 0b11;
        params.cmatrixInitVal = true;
        params.cmatrixSource = false;
        Mmad(l0c_[l0cBufferId * AIC_L0C_ONE_BUFFER_ELEMENTS + l0cOffset],
            l0a_[l0aBufferId * AIC_L0A_ONE_BUFFER_ELEMENTS + l0aOffset], rotationL0b, params);
    }

    __aicore__ inline uint32_t GetAiv0TokenSize(const TileParam &tile) const
    {
        return CeilDiv(tile.tokenSize, QKV_RRC_MIX_AIV_PER_AIC);
    }

    __aicore__ inline void FixpipeSegmentToAivUb(const TileParam &tile, uint32_t mStart, uint32_t mSize,
        uint32_t segmentRows, bool isQSegment, uint32_t bufferId, uint32_t l0cBaseOffset,
        LocalTensor<float> &qkAfterCubeUb)
    {
        const uint32_t aiv0TokenSize = GetAiv0TokenSize(tile);
        const uint32_t aiv1TokenSize = tile.tokenSize - aiv0TokenSize;
        const uint32_t qAiv0Rows = aiv0TokenSize * tilingData_->numQHeads;
        const uint32_t qAiv1Rows = aiv1TokenSize * tilingData_->numQHeads;
        const uint32_t segmentAiv0Rows =
            aiv0TokenSize * (isQSegment ? tilingData_->numQHeads : tilingData_->numKHeads);
        const uint32_t segmentAiv1Rows =
            aiv1TokenSize * (isQSegment ? tilingData_->numQHeads : tilingData_->numKHeads);
        const uint32_t aiv0DstBase = isQSegment ? 0U : qAiv0Rows;
        const uint32_t aiv1DstBase = isQSegment ? 0U : qAiv1Rows;
        FixpipeRowRangeToAivUb(mStart, mSize, 0U, segmentAiv0Rows, aiv0DstBase,
            0U, bufferId, l0cBaseOffset, qkAfterCubeUb);
        FixpipeRowRangeToAivUb(mStart, mSize, segmentAiv0Rows, segmentRows,
            aiv1DstBase, 1U, bufferId, l0cBaseOffset, qkAfterCubeUb);
        (void)segmentAiv1Rows;
    }

    __aicore__ inline void FixpipeRowRangeToAivUb(uint32_t chunkRowStart, uint32_t chunkRowSize,
        uint32_t dstRowStart, uint32_t dstRowEnd, uint32_t segmentDstBase, uint32_t subBlockId, uint32_t bufferId,
        uint32_t l0cBaseOffset, LocalTensor<float> &qkAfterCubeUb)
    {
        const uint32_t rowBegin = MaxU32(chunkRowStart, dstRowStart);
        const uint32_t rowEnd = MinU32(chunkRowStart + chunkRowSize, dstRowEnd);
        if (rowBegin >= rowEnd) {
            return;
        }

        const uint32_t srcLocalRow = rowBegin - chunkRowStart;
        const uint32_t dstLocalRow = segmentDstBase + rowBegin - dstRowStart;
        FixpipeRowsToUb(srcLocalRow, dstLocalRow, rowEnd - rowBegin, chunkRowSize, subBlockId, bufferId,
            l0cBaseOffset, qkAfterCubeUb);
    }

    __aicore__ inline void FixpipeRowsToUb(uint32_t srcLocalRow, uint32_t dstLocalRow, uint32_t rowCount,
        uint32_t sourceMSize, uint32_t subBlockId, uint32_t bufferId, uint32_t l0cBaseOffset,
        LocalTensor<float> &qkAfterCubeUb)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = QKV_RRC_HEAD_DIM_D128;
        fixpipeParams.mSize = AlignUp(rowCount, 2U);
        fixpipeParams.srcStride = AlignUp(sourceMSize, QKV_RRC_NZ_C0);
        fixpipeParams.dstStride = QKV_RRC_HEAD_DIM_D128;
        fixpipeParams.dualDstCtl = 0U;
        fixpipeParams.subBlockId = static_cast<bool>(subBlockId);
        fixpipeParams.unitFlag = 0b11;
        fixpipeParams.params.ndNum = 1U;
        fixpipeParams.params.srcNdStride = 0U;
        fixpipeParams.params.dstNdStride = 0U;

        Fixpipe<float, float, QKV_RRC_FIXPIPE_ROW_MAJOR_UB>(
            qkAfterCubeUb[dstLocalRow * QKV_RRC_HEAD_DIM_D128],
            l0c_[bufferId * AIC_L0C_ONE_BUFFER_ELEMENTS + l0cBaseOffset + srcLocalRow * QKV_RRC_NZ_C0],
            fixpipeParams);
    }

    __aicore__ inline void BindLocalTensors()
    {
        qRotationL1Nz_ = LocalTensor<bfloat16_t>(
            TPosition::TSCM, QKV_RRC_Q_ROTATION_L1_OFFSET, QKV_RRC_ROTATION_ONE_L1_ELEMENTS);
        kRotationL1Nz_ = LocalTensor<bfloat16_t>(
            TPosition::TSCM, QKV_RRC_K_ROTATION_L1_OFFSET, QKV_RRC_ROTATION_ONE_L1_ELEMENTS);
        aRotL1Pool_ =
            LocalTensor<bfloat16_t>(TPosition::TSCM, QKV_RRC_A_ROT_L1_POOL_OFFSET, QKV_RRC_A_ROT_L1_POOL_ELEMENTS);
        l0a_ = LocalTensor<bfloat16_t>(TPosition::A2, 0, AIC_L0A_BUFFER_ELEMENTS);
        l0b_ = LocalTensor<bfloat16_t>(TPosition::B2, 0, AIC_L0B_BUFFER_ELEMENTS);
        l0c_ = LocalTensor<float>(TPosition::CO1, 0, AIC_L0C_BUFFER_ELEMENTS);
        outputDbPoolUb_ =
            LocalTensor<float>(TPosition::LCM, QKV_RRC_OUTPUT_DB_POOL_OFFSET, QKV_RRC_OUTPUT_DB_POOL_FLOAT_ELEMENTS);
    }

    __aicore__ inline LocalTensor<bfloat16_t> qRotationL0b()
    {
        return l0b_;
    }

    __aicore__ inline LocalTensor<bfloat16_t> kRotationL0b()
    {
        return l0b_[AIC_L0B_ONE_BUFFER_ELEMENTS];
    }

    __aicore__ inline LocalTensor<float> QkAfterCubeUb(uint32_t bufferId)
    {
        return outputDbPoolUb_[bufferId * QKV_RRC_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS];
    }

    __aicore__ inline void BindGlobalTensors(const GlobalTensors &tensors)
    {
        qRotationGm_.SetGlobalBuffer((__gm__ bfloat16_t *)tensors.qRotation);
        kRotationGm_.SetGlobalBuffer((__gm__ bfloat16_t *)tensors.kRotation);
        workspaceGm_.SetGlobalBuffer((__gm__ uint8_t *)tensors.workspace);
    }

    uint32_t cubeIndex_;
    const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData_;
    GlobalTensor<bfloat16_t> qRotationGm_;
    GlobalTensor<bfloat16_t> kRotationGm_;
    GlobalTensor<uint8_t> workspaceGm_;
    LocalTensor<bfloat16_t> qRotationL1Nz_;
    LocalTensor<bfloat16_t> kRotationL1Nz_;
    LocalTensor<bfloat16_t> aRotL1Pool_;
    LocalTensor<bfloat16_t> l0a_;
    LocalTensor<bfloat16_t> l0b_;
    LocalTensor<float> l0c_;
    LocalTensor<float> outputDbPoolUb_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_CUBE_H_
