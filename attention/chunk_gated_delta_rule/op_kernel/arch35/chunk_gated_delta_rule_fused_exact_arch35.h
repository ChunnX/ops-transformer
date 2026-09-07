/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_GATED_DELTA_RULE_FUSED_EXACT_ARCH35_H
#define CHUNK_GATED_DELTA_RULE_FUSED_EXACT_ARCH35_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include "kernel_operator.h"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/tile/ascend950/copy_l0c_to_ub.hpp"
#include "chunk_gated_delta_rule_matmul_basic.h"
#include "chunk_gated_delta_rule_utils.h"
#include "vf/chunk_gated_delta_rule_stage1_vf.h"
#include "vf/chunk_gated_delta_rule_stage2_vf.h"

namespace ChunkGatedDeltaRule {

using namespace AscendC;

constexpr FixpipeConfig CGDR_CFG_NZ_UB = {CO2Layout::NZ, true};

template <typename stateType>
struct FusedStage123ExactParams {
    GlobalTensor<bfloat16_t> query;
    GlobalTensor<bfloat16_t> key;
    GlobalTensor<bfloat16_t> value;
    GlobalTensor<bfloat16_t> beta;
    GlobalTensor<float> g;
    GlobalTensor<stateType> initialState;
    GlobalTensor<bfloat16_t> out;
    GlobalTensor<stateType> finalState;
    GlobalTensor<int32_t> actualSeqLengths;
    float scale;
    uint32_t nk;
    uint32_t nv;
    uint32_t dk;
    uint32_t dv;
    uint32_t aiCoreNum;
    uint64_t batchCount;
    uint64_t stateStride0;
    uint64_t stateStride1;
};

/*
 * Exact arch35 engine. One AIC and its two AIVs own one batch/value-head
 * work item for every padded chunk, so state and every cross-stage intermediate
 * stay in UB/L1.
 */
template <typename stateType, bool hasGamma>
class FusedStage123Exact {
public:
    __aicore__ inline void Init(const FusedStage123ExactParams<stateType> &params)
    {
        params_ = params;
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            ProcessAic();
        }
        if ASCEND_IS_AIV {
            ProcessAiv();
        }
        // Drain all exact-path pipelines and counted cross-core protocols
        // before returning to the wrapper.
        SyncAll<false>();
    }

private:
    using ArchTag = Catlass::Arch::Ascend950;
    static constexpr bool STATE_FP32 = std::is_same_v<stateType, float>;
    // The exact path manages every dependency explicitly, so expose the
    // physical address spaces directly and avoid implicit allocator state or
    // lifecycle side effects.
    template <TPosition position, uint32_t capacity>
    struct PhysicalBufferView {
        template <class Element>
        __aicore__ inline LocalTensor<Element> GetBufferByByte(uint32_t offset) const
        {
            return LocalTensor<Element>(position, offset, (capacity - offset) / sizeof(Element));
        }
    };

    struct BufferViews {
        PhysicalBufferView<TPosition::A1, ArchTag::L1_SIZE> l1Buf;
        PhysicalBufferView<TPosition::A2, ArchTag::L0A_SIZE> l0ABuf;
        PhysicalBufferView<TPosition::B2, ArchTag::L0B_SIZE> l0BBuf;
        PhysicalBufferView<TPosition::CO1, ArchTag::L0C_SIZE> l0CBuf;
        PhysicalBufferView<TPosition::VECCALC, ArchTag::UB_SIZE> ubBuf;
    };
    static constexpr uint32_t CHUNK = 64;
    static constexpr uint32_t HALF_CHUNK = 32;
    // The first complete-spec correctness path retains the validated maximum
    // physical feature tile. Logical Dk/Dv only bound GM traffic; every
    // physical suffix is explicitly zeroed before it reaches MMAD.
    static constexpr uint32_t DIM = 128;
    static constexpr uint32_t HALF_DIM = 64;
    static constexpr uint32_t STATE_SLICE_ELEMS = HALF_DIM * DIM;
    static constexpr uint32_t SCORE_SLICE_ELEMS = HALF_CHUNK * CHUNK;
    static constexpr uint32_t INVERSE_SLICE_ELEMS = HALF_CHUNK * CHUNK;

    static constexpr uint64_t PREFIX_DIRECT0_READY_FLAG = 0;
    static constexpr uint64_t PREFIX_DIRECT0_FREE_FLAG = 1;
    static constexpr uint64_t RECURRENCE_DIRECT0_READY_FLAG = 2;
    static constexpr uint64_t RECURRENCE_DIRECT0_FREE_FLAG = 3;
    static constexpr uint64_t PREFIX0_CRITICAL_READY_FLAG = 4;
    static constexpr uint64_t PREFIX1_CRITICAL_READY_FLAG = 5;
    static constexpr uint64_t PREFIX0_READY_FLAG = 6;
    static constexpr uint64_t PREFIX1_READY_FLAG = 7;
    static constexpr uint64_t STATE_READY_FLAG = 8;
    static constexpr uint64_t STATE_FREE_FLAG = 9;
    static constexpr uint64_t PREFIX_DIRECT1_READY_FLAG = 10;
    static constexpr uint64_t SLOT0_FREE_FLAG = 11;
    static constexpr uint64_t SLOT1_FREE_FLAG = 12;
    static constexpr uint64_t PREFIX_DIRECT1_FREE_FLAG = 13;

    static constexpr uint32_t L1_SLOT_BYTES = 160 * 1024;
    static constexpr uint32_t L1_Q_OFFSET = 0;
    static constexpr uint32_t L1_K_OFFSET = 16 * 1024;
    static constexpr uint32_t L1_ATTN_OFFSET = 32 * 1024;
    static constexpr uint32_t L1_LOWER_TMP_OFFSET = 40 * 1024;
    static constexpr uint32_t L1_RAW_KCD_OFFSET = 48 * 1024;
    static constexpr uint32_t L1_VBETA_OFFSET = 64 * 1024;
    static constexpr uint32_t L1_KCD_OFFSET = 80 * 1024;
    static constexpr uint32_t L1_VNEW_OFFSET = 112 * 1024;
    static constexpr uint32_t L1_KG_OFFSET = 128 * 1024;
    static constexpr uint32_t L1_MASKED_QKT_OFFSET = 144 * 1024;
    static constexpr uint32_t L1_STATE_OFFSET = 2 * L1_SLOT_BYTES;
    static constexpr uint32_t L0C_VNEW_OFFSET = 64 * 1024;
    static constexpr uint32_t L0C_OUTPUT_OFFSET = 96 * 1024;

    static constexpr uint32_t UB_DIRECT0_OFFSET = 0;
    static constexpr uint32_t UB_STATE_OFFSET = STATE_FP32 ? 0 : 16 * 1024;
    static constexpr uint32_t UB_ATTN_OFFSET = 40 * 1024;
    static constexpr uint32_t UB_INVERSE_OFFSET = 48 * 1024;
    static constexpr uint32_t UB_IDENTITY_OFFSET = 56 * 1024;
    static constexpr uint32_t UB_MASKED_QKT_PUB_OFFSET = 64 * 1024;
    static constexpr uint32_t UB_Q_OFFSET = 92 * 1024;
    static constexpr uint32_t UB_K_OFFSET = 100 * 1024;
    static constexpr uint32_t UB_V_OFFSET = 108 * 1024;
    static constexpr uint32_t UB_STATE_PUB_OFFSET = 116 * 1024;
    static constexpr uint32_t UB_RAW_G_OFFSET = 132 * 1024;
    static constexpr uint32_t UB_RAW_BETA_OFFSET = 134 * 1024;
    static constexpr uint32_t UB_BETA_BF16_OFFSET = 135 * 1024;
    static constexpr uint32_t UB_GCUM0_OFFSET = 136 * 1024;
    static constexpr uint32_t UB_GCUM1_OFFSET = 137 * 1024;
    static constexpr uint32_t UB_GEXP_OFFSET = 138 * 1024;
    static constexpr uint32_t UB_BETA_FP32_OFFSET = 139 * 1024;
    static constexpr uint32_t UB_GATHER_OFFSET = 140 * 1024;
    static constexpr uint32_t UB_PREFIX_DIRECT0_OFFSET = 160 * 1024;
    static constexpr uint32_t UB_INVERSE_PUB0_OFFSET = 168 * 1024;
    static constexpr uint32_t UB_INVERSE_PUB1_OFFSET = 172 * 1024;
    static constexpr uint32_t UB_STAGE1_PUB0_OFFSET = 176 * 1024;
    static constexpr uint32_t UB_STAGE1_PUB1_OFFSET = 208 * 1024;
    static constexpr uint32_t UB_PREFIX_DIRECT1_OFFSET = 240 * 1024;
    static constexpr uint32_t UB_STAGE1_MATRIX_BYTES = HALF_CHUNK * DIM * sizeof(bfloat16_t);
    static constexpr uint32_t UB_PREFIX_DIRECT_BYTES = 2 * SCORE_SLICE_ELEMS * sizeof(bfloat16_t);
    static constexpr uint32_t INPUT_BATCH_ROWS = HALF_CHUNK / 2;

    // The streaming branch is exclusive with the cube branch and therefore
    // reuses the same physical UB.  FP32 state is its sole canonical state;
    // every following region is a derived, short-lived transport view.
    static constexpr uint32_t UB_STREAM_STATE_FP32_OFFSET = 0;
    static constexpr uint32_t UB_STREAM_STATE_BF16_OFFSET = 32 * 1024;
    static constexpr uint32_t UB_STREAM_Q_OFFSET = 48 * 1024;
    static constexpr uint32_t UB_STREAM_K_OFFSET = UB_STREAM_Q_OFFSET + DIM * sizeof(bfloat16_t);
    static constexpr uint32_t UB_STREAM_V_BF16_OFFSET = UB_STREAM_K_OFFSET + DIM * sizeof(bfloat16_t);
    static constexpr uint32_t UB_STREAM_V_FP32_OFFSET = UB_STREAM_V_BF16_OFFSET + HALF_DIM * sizeof(bfloat16_t);
    static constexpr uint32_t UB_STREAM_BETA_BF16_OFFSET = UB_STREAM_V_FP32_OFFSET + HALF_DIM * sizeof(float);
    static constexpr uint32_t UB_STREAM_BETA_FP32_OFFSET = UB_STREAM_BETA_BF16_OFFSET + BLOCK_SIZE;
    static constexpr uint32_t UB_STREAM_G_OFFSET = UB_STREAM_BETA_FP32_OFFSET + BLOCK_SIZE;
    static constexpr uint32_t UB_STREAM_OUT_FP32_OFFSET = UB_STREAM_G_OFFSET + BLOCK_SIZE;
    static constexpr uint32_t UB_STREAM_OUT_BF16_OFFSET = UB_STREAM_OUT_FP32_OFFSET + HALF_DIM * sizeof(float);
    static constexpr uint32_t UB_STREAM_END = UB_STREAM_OUT_BF16_OFFSET + HALF_DIM * sizeof(bfloat16_t);

    // Conservative static selector.  One unit of vector recurrence work is
    // charged 256x one padded cube MAC proxy until matched device data can
    // replace this fail-closed calibration.
    static constexpr uint64_t STREAM_ARITHMETIC_FACTOR = 7;
    static constexpr uint64_t STREAM_VECTOR_COST = 256;
    static constexpr uint64_t CUBE_MAC_PER_CHUNK = 5832704;

    static_assert(L1_STATE_OFFSET + DIM * DIM * sizeof(bfloat16_t) <= 512 * 1024,
                  "arch35 exact L1 layout exceeds hardware capacity");
    static_assert(!STATE_FP32 || UB_STATE_OFFSET + STATE_SLICE_ELEMS * sizeof(float) <= UB_ATTN_OFFSET,
                  "arch35 exact FP32 state overlaps attention scratch");
    static_assert(UB_V_OFFSET + HALF_CHUNK * DIM * sizeof(bfloat16_t) <= UB_STATE_PUB_OFFSET,
                  "arch35 exact value input overlaps state publication");
    static_assert(UB_STATE_PUB_OFFSET + STATE_SLICE_ELEMS * sizeof(bfloat16_t) <= UB_RAW_G_OFFSET,
                  "arch35 exact state publication overlaps input scratch");
    static_assert(L1_VBETA_OFFSET == L1_RAW_KCD_OFFSET + CHUNK * DIM * sizeof(bfloat16_t),
                  "rawKcd and vBeta must stay contiguous for their merged publication");
    static_assert(UB_PREFIX_DIRECT0_OFFSET + UB_PREFIX_DIRECT_BYTES <= UB_INVERSE_PUB0_OFFSET,
                  "prefixDirect slot 0 overlaps inverse publication");
    static_assert(UB_STAGE1_PUB1_OFFSET + 4 * UB_STAGE1_MATRIX_BYTES <= UB_PREFIX_DIRECT1_OFFSET,
                  "stage1 publication overlaps prefixDirect slot 1");
    static_assert(!STATE_FP32 || UB_STAGE1_PUB0_OFFSET + STATE_SLICE_ELEMS * sizeof(float) <= UB_STAGE1_PUB1_OFFSET,
                  "arch35 exact FP32 recurrence direct slot 0 overlaps slot 1");
    static_assert(!STATE_FP32 || UB_STAGE1_PUB1_OFFSET + STATE_SLICE_ELEMS * sizeof(float) <= UB_PREFIX_DIRECT1_OFFSET,
                  "arch35 exact FP32 recurrence direct slot 1 overlaps prefix direct");
    static_assert(UB_PREFIX_DIRECT1_OFFSET + UB_PREFIX_DIRECT_BYTES <= 248 * 1024,
                  "arch35 UB layout exceeds hardware capacity");
    static_assert(UB_STREAM_STATE_FP32_OFFSET + STATE_SLICE_ELEMS * sizeof(float) == UB_STREAM_STATE_BF16_OFFSET,
                  "arch35 streaming FP32 state layout must be contiguous");
    static_assert(UB_STREAM_STATE_BF16_OFFSET + STATE_SLICE_ELEMS * sizeof(bfloat16_t) <= UB_STREAM_Q_OFFSET,
                  "arch35 streaming state staging overlaps token input");
    static_assert(UB_STREAM_END <= 248 * 1024, "arch35 streaming UB layout exceeds hardware capacity");
    static_assert(PREFIX_DIRECT0_READY_FLAG == 0 && PREFIX_DIRECT0_FREE_FLAG == 1 &&
                      RECURRENCE_DIRECT0_READY_FLAG == 2 && RECURRENCE_DIRECT0_FREE_FLAG == 3 &&
                      PREFIX0_CRITICAL_READY_FLAG == 4 && PREFIX1_CRITICAL_READY_FLAG == 5 && PREFIX0_READY_FLAG == 6 &&
                      PREFIX1_READY_FLAG == 7 && STATE_READY_FLAG == 8 && STATE_FREE_FLAG == 9 &&
                      PREFIX_DIRECT1_READY_FLAG == 10 && SLOT0_FREE_FLAG == 11 && SLOT1_FREE_FLAG == 12 &&
                      PREFIX_DIRECT1_FREE_FLAG == 13,
                  "arch35 exact protocol flags must be unique mode-2 IDs");
    static_assert(2 * HALF_DIM == DIM && 2 * HALF_CHUNK == CHUNK,
                  "arch35 exact AIV split must cover each physical tile exactly");

    static constexpr uint32_t L0AB_STAGES = 2;
    static constexpr uint32_t L0A_STAGE_BYTES = ArchTag::L0A_SIZE / L0AB_STAGES;
    static constexpr uint32_t L0B_STAGE_BYTES = ArchTag::L0B_SIZE / L0AB_STAGES;

    // The exact engine is the only mutex owner before it returns. Fixed IDs
    // keep get_buf/rls_buf operands immediate in the hot loop and avoid
    // depending on allocator state. AIC and AIV execute on different core
    // roles, so each role can use the same private range. SyncAll above drains
    // them before the exact path returns to its wrapper.
    static constexpr uint32_t MUTEX_COUNT = 8;
    static constexpr uint32_t AIC_L0AB_MUTEX_BASE = 0;
    static constexpr uint32_t AIC_L0C_TMP_MUTEX = 2;
    static constexpr uint32_t AIC_L0C_VNEW_MUTEX = 3;
    static constexpr uint32_t AIC_L0C_OUTPUT_MUTEX = 4;
    static constexpr uint32_t AIC_L1_QK_MUTEX_BASE = 5;
    static constexpr uint32_t AIC_L1_FIX_MUTEX = 7;
    static constexpr uint32_t AIV_INPUT_HEAD_MUTEX = 0;
    static constexpr uint32_t AIV_INPUT_TAIL_MUTEX = 1;
    static constexpr uint32_t AIV_STATE_MUTEX = 2;
    static constexpr uint32_t AIV_STAGE1_PUB_MUTEX_BASE = 3;
    static constexpr uint32_t AIV_PREFIX_PUB_MUTEX_BASE = 5;
    static constexpr uint32_t AIV_SCALAR_MUTEX = 7;
    static_assert(AIC_L1_FIX_MUTEX < MUTEX_COUNT && AIV_SCALAR_MUTEX < MUTEX_COUNT,
                  "arch35 exact mutex layout exceeds its allocation");

    static_assert(CHUNK * 2 * DIM * sizeof(bfloat16_t) <= L0A_STAGE_BYTES,
                  "arch35 exact L0A tile exceeds one ping-pong stage");
    static_assert(DIM * DIM * sizeof(bfloat16_t) <= L0B_STAGE_BYTES,
                  "arch35 exact L0B tile exceeds one ping-pong stage");

    __aicore__ inline uint32_t NextL0AbStage()
    {
        const uint32_t stage = l0abStage_;
        l0abStage_ ^= 1;
        return stage;
    }

    __aicore__ inline MutexID MutexId(uint32_t index) const
    {
        return static_cast<MutexID>(index);
    }

    __aicore__ inline uint32_t SetWorkContext(uint64_t workId, uint64_t &scannedBatch, uint64_t &seqStart)
    {
        const uint64_t batchId = workId / params_.nv;
        while (scannedBatch < batchId) {
            seqStart += static_cast<uint32_t>(params_.actualSeqLengths.GetValue(scannedBatch));
            ++scannedBatch;
        }
        currentBatchId_ = batchId;
        currentSeqStart_ = seqStart;
        currentLength_ = static_cast<uint32_t>(params_.actualSeqLengths.GetValue(batchId));
        return static_cast<uint32_t>(workId - batchId * params_.nv);
    }

    __aicore__ inline uint32_t ChunkNum() const
    {
        return (currentLength_ + CHUNK - 1) / CHUNK;
    }

    __aicore__ inline uint32_t ChunkValidRows(uint32_t chunk) const
    {
        const uint32_t chunkBegin = chunk * CHUNK;
        const uint32_t remaining = currentLength_ - chunkBegin;
        return remaining < CHUNK ? remaining : CHUNK;
    }

    __aicore__ inline uint32_t SubBlockValidRows(uint32_t chunk, uint32_t subBlockIdx) const
    {
        const uint32_t validRows = ChunkValidRows(chunk);
        const uint32_t rowBegin = subBlockIdx * HALF_CHUNK;
        if (validRows <= rowBegin) {
            return 0;
        }
        const uint32_t remaining = validRows - rowBegin;
        return remaining < HALF_CHUNK ? remaining : HALF_CHUNK;
    }

    __aicore__ inline uint32_t StateValidRows(uint32_t subBlockIdx) const
    {
        const uint32_t rowBegin = subBlockIdx * HALF_DIM;
        if (params_.dv <= rowBegin) {
            return 0;
        }
        const uint32_t remaining = params_.dv - rowBegin;
        return remaining < HALF_DIM ? remaining : HALF_DIM;
    }

    __aicore__ inline bool UseStreamingExact() const
    {
        if (currentLength_ == 0) {
            return true;
        }
        const uint64_t chunkCount = (static_cast<uint64_t>(currentLength_) + CHUNK - 1) / CHUNK;
        const uint64_t streamScore = static_cast<uint64_t>(currentLength_) * params_.dk * params_.dv *
                                     STREAM_ARITHMETIC_FACTOR * STREAM_VECTOR_COST;
        const uint64_t cubeScore = chunkCount * 2 * CUBE_MAC_PER_CHUNK;
        return streamScore < cubeScore;
    }

    __aicore__ inline uint32_t PrefixDirectOffset(uint32_t slot)
    {
        return slot == 0 ? UB_PREFIX_DIRECT0_OFFSET : UB_PREFIX_DIRECT1_OFFSET;
    }

    __aicore__ inline uint32_t Stage1PublicationOffset(uint32_t slot)
    {
        return slot == 0 ? UB_STAGE1_PUB0_OFFSET : UB_STAGE1_PUB1_OFFSET;
    }

    __aicore__ inline uint32_t RecurrenceDirectOffset(uint32_t slot)
    {
        return STATE_FP32 ? Stage1PublicationOffset(slot) : UB_DIRECT0_OFFSET;
    }

    __aicore__ inline uint64_t PrefixDirectReadyFlag(uint32_t slot)
    {
        return slot == 0 ? PREFIX_DIRECT0_READY_FLAG : PREFIX_DIRECT1_READY_FLAG;
    }

    __aicore__ inline uint64_t PrefixDirectFreeFlag(uint32_t slot)
    {
        return slot == 0 ? PREFIX_DIRECT0_FREE_FLAG : PREFIX_DIRECT1_FREE_FLAG;
    }

    __aicore__ inline void CopyGmToL1(GlobalTensor<bfloat16_t> gm, LocalTensor<bfloat16_t> l1, uint32_t rows,
                                      uint32_t cols, uint32_t srcRowStride, uint32_t dstRows)
    {
        Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = rows;
        params.dValue = cols;
        params.srcDValue = srcRowStride;
        params.dstNzC0Stride = MmAlign(dstRows, MM_BLOCK_CUBE);
        params.dstNzNStride = 1;
        params.srcNdMatrixStride = 0;
        params.dstNzMatrixStride = 0;
        DataCopy(l1, gm, params);
    }

    template <bool ifTranspose>
    __aicore__ inline void LoadL1ToL0(LocalTensor<bfloat16_t> l0, LocalTensor<bfloat16_t> l1, uint32_t rows,
                                      uint32_t cols, uint32_t fullRows, uint32_t rowStart = 0, uint32_t colStart = 0)
    {
        LoadData2DParamsV2 params;
        params.mStartPosition = rowStart / MM_BLOCK_CUBE;
        params.kStartPosition = colStart / MM_NZ_FRACTAL;
        params.mStep = MmAlign(rows, MM_BLOCK_CUBE) / MM_BLOCK_CUBE;
        params.kStep = MmAlign(cols, MM_NZ_FRACTAL) / MM_NZ_FRACTAL;
        params.srcStride = MmAlign(fullRows, MM_BLOCK_CUBE) / MM_BLOCK_CUBE;
        params.dstStride = ifTranspose ? params.kStep : params.mStep;
        params.ifTranspose = ifTranspose;
        LoadData(l0, l1, params);
    }

    __aicore__ inline void LoadL1RowsToL0A(LocalTensor<bfloat16_t> l0a, LocalTensor<bfloat16_t> l1, uint32_t rows,
                                           uint32_t cols, uint32_t srcFullRows, uint32_t srcRowStart,
                                           uint32_t dstFullRows, uint32_t dstRowStart)
    {
        LoadData2DParamsV2 params;
        params.mStartPosition = srcRowStart / MM_BLOCK_CUBE;
        params.kStartPosition = 0;
        params.mStep = MmAlign(rows, MM_BLOCK_CUBE) / MM_BLOCK_CUBE;
        params.kStep = MmAlign(cols, MM_NZ_FRACTAL) / MM_NZ_FRACTAL;
        params.srcStride = MmAlign(srcFullRows, MM_BLOCK_CUBE) / MM_BLOCK_CUBE;
        params.dstStride = MmAlign(dstFullRows, MM_BLOCK_CUBE) / MM_BLOCK_CUBE;
        params.ifTranspose = false;
        LoadData(l0a[dstRowStart * MM_NZ_FRACTAL], l1, params);
    }

    __aicore__ inline void Mmad(LocalTensor<float> l0c, LocalTensor<bfloat16_t> l0a, LocalTensor<bfloat16_t> l0b,
                                uint32_t m, uint32_t n, uint32_t k, bool init = true)
    {
        MmadParams params;
        params.m = m < MM_BLOCK_CUBE ? MM_BLOCK_CUBE : m;
        params.n = n;
        params.k = k;
        params.cmatrixInitVal = init;
        params.cmatrixSource = false;
        AscendC::Mmad(l0c, l0a, l0b, params);
    }

    template <bool splitN, uint32_t directSlot>
    __aicore__ inline void PublishDirect(LocalTensor<float> l0c, uint32_t m, uint32_t n, bool waitDirectFree,
                                         uint32_t prefixSlot = 0)
    {
        static_assert(directSlot == 0 || directSlot == 2, "invalid direct slot");
        const uint64_t readyFlag = directSlot == 0 ? RECURRENCE_DIRECT0_READY_FLAG : PrefixDirectReadyFlag(prefixSlot);
        const uint64_t freeFlag = directSlot == 0 ? RECURRENCE_DIRECT0_FREE_FLAG : PrefixDirectFreeFlag(prefixSlot);

        Mutex::Unlock<PIPE_M>(MutexId(AIC_L0C_TMP_MUTEX));
        Mutex::Lock<PIPE_FIX>(MutexId(AIC_L0C_TMP_MUTEX));
        if (waitDirectFree) {
            CrossCoreWaitFlag(freeFlag);
        }
        if constexpr (directSlot == 0) {
            static_assert(!splitN, "state update direct output splits along M");
            if constexpr (STATE_FP32) {
                auto direct = resource_.ubBuf.template GetBufferByByte<float>(RecurrenceDirectOffset(prefixSlot));
                FixpipeParamsC310<CO2Layout::ROW_MAJOR> params;
                params.nSize = n;
                params.mSize = m / 2;
                params.srcStride = MmAlign(m, MM_BLOCK_CUBE);
                params.dstStride = n;
                params.quantPre = QuantMode_t::NoQuant;
                params.dualDstCtl = 0;
                params.params.ndNum = 1;
                params.params.srcNdStride = 0;
                params.params.dstNdStride = 0;
                params.subBlockId = 0;
                Fixpipe<float, float, CFG_ROW_MAJOR_UB>(direct, l0c, params);
                params.subBlockId = 1;
                Fixpipe<float, float, CFG_ROW_MAJOR_UB>(direct, l0c[m / 2 * MM_NZ_FRACTAL], params);
            } else {
                auto direct = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(UB_DIRECT0_OFFSET);
                FixpipeParamsC310<CO2Layout::NZ> params;
                params.nSize = n;
                params.mSize = m / 2;
                params.srcStride = MmAlign(m, MM_BLOCK_CUBE);
                params.dstStride = MmAlign(m / 2, MM_BLOCK_CUBE) * MM_NZ_FRACTAL;
                params.quantPre = QuantMode_t::F322BF16;
                params.dualDstCtl = 0;
                params.subBlockId = 0;
                Fixpipe<bfloat16_t, float, CGDR_CFG_NZ_UB>(direct, l0c, params);
                params.subBlockId = 1;
                Fixpipe<bfloat16_t, float, CGDR_CFG_NZ_UB>(direct, l0c[m / 2 * MM_NZ_FRACTAL], params);
            }
        } else {
            auto direct = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(PrefixDirectOffset(prefixSlot));
            FixpipeParamsC310<CO2Layout::ROW_MAJOR> params;
            params.srcStride = MmAlign(m, MM_BLOCK_CUBE);
            params.quantPre = QuantMode_t::F322BF16;
            params.dualDstCtl = 0;
            params.params.ndNum = 1;
            params.params.srcNdStride = 0;
            params.params.dstNdStride = 0;
            if constexpr (splitN) {
                params.nSize = n / 2;
                params.mSize = m;
                params.dstStride = n / 2;
                params.subBlockId = 0;
                Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR_UB>(direct, l0c, params);
                params.subBlockId = 1;
                Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR_UB>(direct, l0c[(n / 2) * MmAlign(m, MM_BLOCK_CUBE)], params);
            } else {
                params.nSize = n;
                params.mSize = m / 2;
                params.dstStride = n;
                params.subBlockId = 0;
                Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR_UB>(direct, l0c, params);
                params.subBlockId = 1;
                Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR_UB>(direct, l0c[m / 2 * MM_NZ_FRACTAL], params);
            }
        }
        CrossCoreSetFlag<0x2, PIPE_FIX>(readyFlag);
        Mutex::Unlock<PIPE_FIX>(MutexId(AIC_L0C_TMP_MUTEX));
    }

    __aicore__ inline uint32_t GetZnOffset(uint32_t row, uint32_t col, uint32_t fullRows)
    {
        return (row % MM_BLOCK_CUBE) * MM_NZ_FRACTAL + (row / MM_BLOCK_CUBE) * MM_BLOCK_CUBE * MM_NZ_FRACTAL +
               (col % MM_NZ_FRACTAL) + (col / MM_NZ_FRACTAL) * MmAlign(fullRows, MM_BLOCK_CUBE) * MM_NZ_FRACTAL;
    }

    __aicore__ inline void PublishL1(LocalTensor<bfloat16_t> dst, LocalTensor<float> l0c, uint32_t m, uint32_t n,
                                     uint32_t fullRows, MutexID l0cMutex, uint32_t rowStart = 0, uint32_t colStart = 0)
    {
        FixpipeParamsC310<CO2Layout::NZ> params;
        params.nSize = n;
        params.mSize = m;
        params.srcStride = MmAlign(m, MM_BLOCK_CUBE);
        params.dstStride = MmAlign(fullRows, MM_BLOCK_CUBE) * MM_NZ_FRACTAL;
        params.quantPre = QuantMode_t::F322BF16;

        Mutex::Lock<PIPE_FIX>(MutexId(AIC_L1_FIX_MUTEX));
        Mutex::Unlock<PIPE_M>(l0cMutex);
        Mutex::Lock<PIPE_FIX>(l0cMutex);
        Fixpipe<bfloat16_t, float, CFG_NZ>(dst[GetZnOffset(rowStart, colStart, fullRows)], l0c, params);
        Mutex::Unlock<PIPE_FIX>(l0cMutex);
        Mutex::Unlock<PIPE_FIX>(MutexId(AIC_L1_FIX_MUTEX));
    }

    __aicore__ inline void PublishOutput(uint32_t nvId, uint32_t chunk, uint32_t validRows, LocalTensor<float> l0c)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> params;
        params.nSize = params_.dv;
        params.mSize = validRows;
        params.srcStride = MmAlign(CHUNK, MM_BLOCK_CUBE);
        params.dstStride = params_.nv * params_.dv;
        params.quantPre = QuantMode_t::F322BF16;
        params.dualDstCtl = 0;
        params.params.ndNum = 1;
        params.params.srcNdStride = 0;
        params.params.dstNdStride = 0;

        Mutex::Unlock<PIPE_M>(MutexId(AIC_L0C_OUTPUT_MUTEX));
        Mutex::Lock<PIPE_FIX>(MutexId(AIC_L0C_OUTPUT_MUTEX));
        const uint64_t outputOffset =
            (currentSeqStart_ + static_cast<uint64_t>(chunk) * CHUNK) * params_.nv * params_.dv +
            static_cast<uint64_t>(nvId) * params_.dv;
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(params_.out[outputOffset], l0c, params);
        Mutex::Unlock<PIPE_FIX>(MutexId(AIC_L0C_OUTPUT_MUTEX));
    }

    template <bool transA, bool loadBTranspose, bool splitNDirect = false, uint32_t directSlot = 0>
    __aicore__ inline void RunMm(LocalTensor<bfloat16_t> l1a, LocalTensor<bfloat16_t> l1b, uint32_t aRows,
                                 uint32_t aCols, uint32_t bRows, uint32_t bCols, uint32_t m, uint32_t n, uint32_t k,
                                 bool waitDirectFree, uint32_t directBufferSlot, uint32_t aFullRows = 0,
                                 uint32_t aRowStart = 0, uint32_t aColStart = 0, uint32_t bFullRows = 0,
                                 uint32_t bRowStart = 0, uint32_t bColStart = 0)
    {
        const uint32_t stage = NextL0AbStage();
        const MutexID l0abMutex = MutexId(AIC_L0AB_MUTEX_BASE + stage);
        auto l0a = resource_.l0ABuf.template GetBufferByByte<bfloat16_t>(stage * L0A_STAGE_BYTES);
        auto l0b = resource_.l0BBuf.template GetBufferByByte<bfloat16_t>(stage * L0B_STAGE_BYTES);
        auto l0c = resource_.l0CBuf.template GetBufferByByte<float>(0);
        const uint32_t actualAFullRows = aFullRows == 0 ? aRows : aFullRows;
        const uint32_t actualBFullRows = bFullRows == 0 ? bRows : bFullRows;

        Mutex::Lock<PIPE_MTE1>(l0abMutex);
        Mutex::Lock<PIPE_MTE1>(MutexId(AIC_L1_FIX_MUTEX));
        LoadL1ToL0<transA>(l0a, l1a, aRows, aCols, actualAFullRows, aRowStart, aColStart);
        LoadL1ToL0<loadBTranspose>(l0b, l1b, bRows, bCols, actualBFullRows, bRowStart, bColStart);
        Mutex::Unlock<PIPE_MTE1>(MutexId(AIC_L1_FIX_MUTEX));
        Mutex::Unlock<PIPE_MTE1>(l0abMutex);
        Mutex::Lock<PIPE_M>(l0abMutex);
        Mutex::Lock<PIPE_M>(MutexId(AIC_L0C_TMP_MUTEX));
        Mmad(l0c, l0a, l0b, m, n, k);
        Mutex::Unlock<PIPE_M>(l0abMutex);
        PublishDirect<splitNDirect, directSlot>(l0c, m, n, waitDirectFree, directBufferSlot);
    }

    template <bool transA, bool loadBTranspose, bool consumeFixOutput = false>
    __aicore__ inline void RunMmToL1(LocalTensor<bfloat16_t> dst, LocalTensor<bfloat16_t> l1a,
                                     LocalTensor<bfloat16_t> l1b, uint32_t aRows, uint32_t aCols, uint32_t bRows,
                                     uint32_t bCols, uint32_t m, uint32_t n, uint32_t k, uint32_t dstFullRows,
                                     uint32_t dstRowStart = 0, uint32_t dstColStart = 0, uint32_t aFullRows = 0,
                                     uint32_t aRowStart = 0, uint32_t aColStart = 0, uint32_t bFullRows = 0,
                                     uint32_t bRowStart = 0, uint32_t bColStart = 0)
    {
        const uint32_t stage = NextL0AbStage();
        const MutexID l0abMutex = MutexId(AIC_L0AB_MUTEX_BASE + stage);
        auto l0a = resource_.l0ABuf.template GetBufferByByte<bfloat16_t>(stage * L0A_STAGE_BYTES);
        auto l0b = resource_.l0BBuf.template GetBufferByByte<bfloat16_t>(stage * L0B_STAGE_BYTES);
        auto l0c = resource_.l0CBuf.template GetBufferByByte<float>(0);
        const uint32_t actualAFullRows = aFullRows == 0 ? aRows : aFullRows;
        const uint32_t actualBFullRows = bFullRows == 0 ? bRows : bFullRows;

        Mutex::Lock<PIPE_MTE1>(l0abMutex);
        if constexpr (consumeFixOutput) {
            Mutex::Lock<PIPE_MTE1>(MutexId(AIC_L1_FIX_MUTEX));
        }
        LoadL1ToL0<transA>(l0a, l1a, aRows, aCols, actualAFullRows, aRowStart, aColStart);
        LoadL1ToL0<loadBTranspose>(l0b, l1b, bRows, bCols, actualBFullRows, bRowStart, bColStart);
        if constexpr (consumeFixOutput) {
            Mutex::Unlock<PIPE_MTE1>(MutexId(AIC_L1_FIX_MUTEX));
        }
        Mutex::Unlock<PIPE_MTE1>(l0abMutex);
        Mutex::Lock<PIPE_M>(l0abMutex);
        Mutex::Lock<PIPE_M>(MutexId(AIC_L0C_TMP_MUTEX));
        Mmad(l0c, l0a, l0b, m, n, k);
        Mutex::Unlock<PIPE_M>(l0abMutex);
        PublishL1(dst, l0c, m, n, dstFullRows, MutexId(AIC_L0C_TMP_MUTEX), dstRowStart, dstColStart);
    }

    template <bool transA, bool loadBTranspose, bool waitFix, bool init>
    __aicore__ inline void RunMmToL0C(LocalTensor<float> l0c, LocalTensor<bfloat16_t> l1a, LocalTensor<bfloat16_t> l1b,
                                      uint32_t aRows, uint32_t aCols, uint32_t bRows, uint32_t bCols, uint32_t m,
                                      uint32_t n, uint32_t k, MutexID l0cMutex, uint32_t aFullRows = 0,
                                      uint32_t aRowStart = 0, uint32_t aColStart = 0, uint32_t bFullRows = 0,
                                      uint32_t bRowStart = 0, uint32_t bColStart = 0)
    {
        const uint32_t stage = NextL0AbStage();
        const MutexID l0abMutex = MutexId(AIC_L0AB_MUTEX_BASE + stage);
        auto l0a = resource_.l0ABuf.template GetBufferByByte<bfloat16_t>(stage * L0A_STAGE_BYTES);
        auto l0b = resource_.l0BBuf.template GetBufferByByte<bfloat16_t>(stage * L0B_STAGE_BYTES);
        const uint32_t actualAFullRows = aFullRows == 0 ? aRows : aFullRows;
        const uint32_t actualBFullRows = bFullRows == 0 ? bRows : bFullRows;

        Mutex::Lock<PIPE_MTE1>(l0abMutex);
        LoadL1ToL0<transA>(l0a, l1a, aRows, aCols, actualAFullRows, aRowStart, aColStart);
        LoadL1ToL0<loadBTranspose>(l0b, l1b, bRows, bCols, actualBFullRows, bRowStart, bColStart);
        Mutex::Unlock<PIPE_MTE1>(l0abMutex);
        Mutex::Lock<PIPE_M>(l0abMutex);
        if constexpr (waitFix) {
            Mutex::Lock<PIPE_M>(l0cMutex);
        }
        Mmad(l0c, l0a, l0b, m, n, k, init);
        Mutex::Unlock<PIPE_M>(l0abMutex);
    }

    __aicore__ inline void RunStatePair(LocalTensor<bfloat16_t> l1Vnew, LocalTensor<float> l0cVnew,
                                        LocalTensor<float> l0cOutput, LocalTensor<bfloat16_t> l1Kcd,
                                        LocalTensor<bfloat16_t> l1State)
    {
        const uint32_t stateStage = NextL0AbStage();
        const uint32_t outputStage = NextL0AbStage();
        const MutexID stateMutex = MutexId(AIC_L0AB_MUTEX_BASE + stateStage);
        const MutexID outputMutex = MutexId(AIC_L0AB_MUTEX_BASE + outputStage);
        auto l0aState = resource_.l0ABuf.template GetBufferByByte<bfloat16_t>(stateStage * L0A_STAGE_BYTES);
        auto l0aOutput = resource_.l0ABuf.template GetBufferByByte<bfloat16_t>(outputStage * L0A_STAGE_BYTES);
        auto l0bState = resource_.l0BBuf.template GetBufferByByte<bfloat16_t>(stateStage * L0B_STAGE_BYTES);

        Mutex::Lock<PIPE_MTE1>(stateMutex);
        Mutex::Lock<PIPE_MTE1>(MutexId(AIC_L1_FIX_MUTEX));
        LoadL1ToL0<false>(l0aState, l1Kcd, CHUNK, DIM, DIM, 0, 0);
        LoadL1ToL0<false>(l0bState, l1State, DIM, DIM, DIM);
        Mutex::Unlock<PIPE_MTE1>(MutexId(AIC_L1_FIX_MUTEX));
        CrossCoreSetFlag<0x2, PIPE_MTE1>(STATE_FREE_FLAG);
        Mutex::Unlock<PIPE_MTE1>(stateMutex);
        Mutex::Lock<PIPE_M>(stateMutex);
        Mmad(l0cVnew, l0aState, l0bState, CHUNK, DIM, DIM, false);

        PublishL1(l1Vnew, l0cVnew, CHUNK, DIM, CHUNK, MutexId(AIC_L0C_VNEW_MUTEX));

        Mutex::Lock<PIPE_MTE1>(outputMutex);
        LoadL1ToL0<false>(l0aOutput, l1Kcd, CHUNK, DIM, DIM, CHUNK, 0);
        Mutex::Unlock<PIPE_MTE1>(outputMutex);
        Mutex::Lock<PIPE_M>(outputMutex);
        Mutex::Lock<PIPE_M>(MutexId(AIC_L0C_OUTPUT_MUTEX));
        Mmad(l0cOutput, l0aOutput, l0bState, CHUNK, DIM, DIM, true);
        Mutex::Unlock<PIPE_M>(stateMutex);
        Mutex::Unlock<PIPE_M>(outputMutex);
    }

    __aicore__ inline void RunQkPrefixMm(LocalTensor<bfloat16_t> l1Q, LocalTensor<bfloat16_t> l1K, uint32_t prefixSlot,
                                         bool waitDirectFree)
    {
        const uint32_t stage = NextL0AbStage();
        const MutexID l0abMutex = MutexId(AIC_L0AB_MUTEX_BASE + stage);
        const MutexID qkMutex = MutexId(AIC_L1_QK_MUTEX_BASE + prefixSlot);
        auto l0a = resource_.l0ABuf.template GetBufferByByte<bfloat16_t>(stage * L0A_STAGE_BYTES);
        auto l0b = resource_.l0BBuf.template GetBufferByByte<bfloat16_t>(stage * L0B_STAGE_BYTES);
        auto l0c = resource_.l0CBuf.template GetBufferByByte<float>(0);

        Mutex::Lock<PIPE_MTE1>(l0abMutex);
        Mutex::Lock<PIPE_MTE1>(qkMutex);
        LoadL1RowsToL0A(l0a, l1K, HALF_CHUNK, DIM, CHUNK, 0, CHUNK * 2, 0);
        LoadL1RowsToL0A(l0a, l1Q, HALF_CHUNK, DIM, CHUNK, 0, CHUNK * 2, HALF_CHUNK);
        LoadL1RowsToL0A(l0a, l1K, HALF_CHUNK, DIM, CHUNK, HALF_CHUNK, CHUNK * 2, CHUNK);
        LoadL1RowsToL0A(l0a, l1Q, HALF_CHUNK, DIM, CHUNK, HALF_CHUNK, CHUNK * 2, CHUNK + HALF_CHUNK);
        LoadL1ToL0<false>(l0b, l1K, CHUNK, DIM, CHUNK);
        Mutex::Unlock<PIPE_MTE1>(qkMutex);
        Mutex::Unlock<PIPE_MTE1>(l0abMutex);
        Mutex::Lock<PIPE_M>(l0abMutex);
        Mutex::Lock<PIPE_M>(MutexId(AIC_L0C_TMP_MUTEX));
        Mmad(l0c, l0a, l0b, CHUNK * 2, CHUNK, DIM);
        Mutex::Unlock<PIPE_M>(l0abMutex);
        PublishDirect<false, 2>(l0c, CHUNK * 2, CHUNK, waitDirectFree, prefixSlot);
    }

    __aicore__ inline uint64_t PrefixReadyFlag(uint32_t slot)
    {
        return slot == 0 ? PREFIX0_READY_FLAG : PREFIX1_READY_FLAG;
    }

    __aicore__ inline uint64_t PrefixCriticalReadyFlag(uint32_t slot)
    {
        return slot == 0 ? PREFIX0_CRITICAL_READY_FLAG : PREFIX1_CRITICAL_READY_FLAG;
    }

    __aicore__ inline uint64_t SlotFreeFlag(uint32_t slot)
    {
        return slot == 0 ? SLOT0_FREE_FLAG : SLOT1_FREE_FLAG;
    }

    __aicore__ inline void PrefetchPrefixQkAic(uint32_t kvHead, uint32_t chunk)
    {
        const uint32_t slot = chunk & 1;
        const uint32_t slotBase = slot * L1_SLOT_BYTES;
        const uint32_t validRows = ChunkValidRows(chunk);
        const uint64_t tokenStart = currentSeqStart_ + static_cast<uint64_t>(chunk) * CHUNK;
        const uint64_t qkOffset = tokenStart * params_.nk * params_.dk + kvHead * params_.dk;
        auto l1Q = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(slotBase + L1_Q_OFFSET);
        auto l1K = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(slotBase + L1_K_OFFSET);

        const MutexID qkMutex = MutexId(AIC_L1_QK_MUTEX_BASE + slot);
        Mutex::Lock<PIPE_MTE2>(qkMutex);
        if (validRows < CHUNK || params_.dk < DIM) {
            constexpr uint16_t L1_QK_BLOCKS = CHUNK * DIM * sizeof(bfloat16_t) / BLOCK_SIZE;
            InitConstValueParams<bfloat16_t> zeroParams{1, L1_QK_BLOCKS, 0, static_cast<bfloat16_t>(0)};
            Fill(l1Q, zeroParams);
            Fill(l1K, zeroParams);
        }
        CopyGmToL1(params_.query[qkOffset], l1Q, validRows, params_.dk, params_.nk * params_.dk, CHUNK);
        CopyGmToL1(params_.key[qkOffset], l1K, validRows, params_.dk, params_.nk * params_.dk, CHUNK);
        Mutex::Unlock<PIPE_MTE2>(qkMutex);
    }

    __aicore__ inline void RunPrefetchedPrefixQkAic(uint32_t chunk)
    {
        const uint32_t slot = chunk & 1;
        const uint32_t slotBase = slot * L1_SLOT_BYTES;
        auto l1Q = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(slotBase + L1_Q_OFFSET);
        auto l1K = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(slotBase + L1_K_OFFSET);

        RunQkPrefixMm(l1Q, l1K, slot, chunk >= 2);
    }

    __aicore__ inline void ProcessRecurrenceAic(uint32_t nvId, uint32_t kvHead, uint32_t chunk)
    {
        const uint32_t slot = chunk & 1;
        const uint32_t slotBase = slot * L1_SLOT_BYTES;
        const uint32_t chunkNum = ChunkNum();
        const uint32_t validRows = ChunkValidRows(chunk);
        auto l1Attn = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(slotBase + L1_ATTN_OFFSET);
        auto l1Lower = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(slotBase + L1_LOWER_TMP_OFFSET);
        auto l1RawKcd = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(slotBase + L1_RAW_KCD_OFFSET);
        auto l1VBeta = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(slotBase + L1_VBETA_OFFSET);
        auto l1Kcd = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(slotBase + L1_KCD_OFFSET);
        auto l1State = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(L1_STATE_OFFSET);
        auto l1Vnew = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(slotBase + L1_VNEW_OFFSET);
        auto l1Kg = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(slotBase + L1_KG_OFFSET);
        auto l1MaskedQkt = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(slotBase + L1_MASKED_QKT_OFFSET);

        if (chunk + 1 < chunkNum) {
            PrefetchPrefixQkAic(kvHead, chunk + 1);
            RunPrefetchedPrefixQkAic(chunk + 1);
        }
        CrossCoreWaitFlag(PrefixCriticalReadyFlag(slot));
        RunMmToL1<false, true>(l1Lower, l1Attn, l1Attn, HALF_CHUNK, HALF_CHUNK, HALF_CHUNK, HALF_CHUNK, HALF_CHUNK,
                               HALF_CHUNK, HALF_CHUNK, HALF_CHUNK, 0, 0, CHUNK, HALF_CHUNK, 0, CHUNK, 0, 0);

        RunMmToL1<false, true, true>(l1Attn, l1Attn, l1Lower, HALF_CHUNK, HALF_CHUNK, HALF_CHUNK, HALF_CHUNK,
                                     HALF_CHUNK, HALF_CHUNK, HALF_CHUNK, CHUNK, HALF_CHUNK, 0, CHUNK, HALF_CHUNK,
                                     HALF_CHUNK, HALF_CHUNK, 0, 0);

        RunMmToL1<false, true, true>(l1Kcd, l1Attn, l1RawKcd, CHUNK, CHUNK, CHUNK, DIM, CHUNK, DIM, CHUNK, DIM);

        auto l0cVnew = resource_.l0CBuf.template GetBufferByByte<float>(L0C_VNEW_OFFSET);
        auto l0cOutput = resource_.l0CBuf.template GetBufferByByte<float>(L0C_OUTPUT_OFFSET);
        RunMmToL0C<false, true, true, true>(l0cVnew, l1Attn, l1VBeta, CHUNK, CHUNK, CHUNK, DIM, CHUNK, DIM, CHUNK,
                                            MutexId(AIC_L0C_VNEW_MUTEX));
        CrossCoreWaitFlag(STATE_READY_FLAG);
        RunStatePair(l1Vnew, l0cVnew, l0cOutput, l1Kcd, l1State);
        RunMm<true, true, false, 0>(l1Vnew, l1Kg, CHUNK, DIM, CHUNK, DIM, DIM, DIM, CHUNK, chunk != 0, slot);
        CrossCoreWaitFlag(PrefixReadyFlag(slot));
        RunMmToL0C<false, true, false, false>(l0cOutput, l1MaskedQkt, l1Vnew, CHUNK, CHUNK, CHUNK, DIM, CHUNK, DIM,
                                              CHUNK, MutexId(AIC_L0C_OUTPUT_MUTEX));
        if (chunk + 2 < chunkNum) {
            CrossCoreSetFlag<0x2, PIPE_MTE1>(SlotFreeFlag(slot));
        }
        PublishOutput(nvId, chunk, validRows, l0cOutput);
    }

    __aicore__ inline void ProcessAicHead(uint32_t nvId)
    {
        if (UseStreamingExact()) {
            return;
        }
        if (currentLength_ == 0) {
            return;
        }
        const uint32_t kvHead = nvId / (params_.nv / params_.nk);
        const uint32_t chunkNum = ChunkNum();
        l0abStage_ = 0;
        CrossCoreSetFlag<0x2, PIPE_MTE2>(SLOT0_FREE_FLAG);
        if (chunkNum > 1) {
            CrossCoreSetFlag<0x2, PIPE_MTE2>(SLOT1_FREE_FLAG);
        }
        PrefetchPrefixQkAic(kvHead, 0);
        RunPrefetchedPrefixQkAic(0);
        for (uint32_t chunk = 0; chunk < chunkNum; ++chunk) {
            ProcessRecurrenceAic(nvId, kvHead, chunk);
        }
        CrossCoreWaitFlag(RECURRENCE_DIRECT0_FREE_FLAG);
        CrossCoreWaitFlag(PREFIX_DIRECT0_FREE_FLAG);
        if (chunkNum > 1) {
            CrossCoreWaitFlag(PREFIX_DIRECT1_FREE_FLAG);
        }
    }

    __aicore__ inline void ProcessAic()
    {
        SetLoadDataPaddingValue<bfloat16_t>(static_cast<bfloat16_t>(0));
        const uint64_t totalWork = params_.batchCount * params_.nv;
        uint64_t scannedBatch = 0;
        uint64_t seqStart = 0;
        for (uint64_t workId = static_cast<uint32_t>(GetBlockIdx()); workId < totalWork; workId += params_.aiCoreNum) {
            const uint32_t nvId = SetWorkContext(workId, scannedBatch, seqStart);
            ProcessAicHead(nvId);
        }
    }

    template <uint32_t rows, uint32_t cols, uint32_t fullRows, uint32_t matrixCount = 1>
    __aicore__ inline void CopyCompactZnToL1(LocalTensor<bfloat16_t> src, LocalTensor<bfloat16_t> dst,
                                             uint32_t rowBegin)
    {
        static_assert(cols % MM_NZ_FRACTAL == 0, "compact zN columns must be C0 aligned");
        static_assert(rows <= fullRows, "compact zN source cannot exceed destination rows");
        DataCopyParams params{static_cast<uint16_t>(matrixCount * cols / MM_NZ_FRACTAL), static_cast<uint16_t>(rows), 0,
                              static_cast<uint16_t>(fullRows - rows)};
        DataCopy(dst[GetZnOffset(rowBegin, 0, fullRows)], src, params);
    }

    __aicore__ inline void CopyCompactStateToGm(GlobalTensor<bfloat16_t> dst, LocalTensor<bfloat16_t> src,
                                                uint32_t validRows)
    {
        if (validRows == 0) {
            return;
        }
        for (uint32_t colBegin = 0; colBegin < params_.dk; colBegin += MM_NZ_FRACTAL) {
            const uint32_t validCols = params_.dk - colBegin < MM_NZ_FRACTAL ? params_.dk - colBegin : MM_NZ_FRACTAL;
            const uint32_t srcOffset = (colBegin / MM_NZ_FRACTAL) * HALF_DIM * MM_NZ_FRACTAL;
            DataCopyExtParams params{static_cast<uint16_t>(validRows),
                                     static_cast<uint32_t>(validCols * sizeof(bfloat16_t)), 0,
                                     static_cast<int64_t>((params_.dk - validCols) * sizeof(bfloat16_t)), 0};
            DataCopyPad(dst[colBegin], src[srcOffset], params);
        }
    }

    template <typename copyType>
    __aicore__ inline void CopyRowMajorStateToGm(GlobalTensor<copyType> dst, LocalTensor<copyType> src,
                                                 uint32_t validRows)
    {
        if (validRows == 0) {
            return;
        }
        constexpr uint32_t elementsPerBlock = BLOCK_SIZE / sizeof(copyType);
        const uint32_t alignedCols = MmAlign(params_.dk, elementsPerBlock);
        DataCopyExtParams params{static_cast<uint16_t>(validRows), static_cast<uint32_t>(params_.dk * sizeof(copyType)),
                                 static_cast<uint32_t>((DIM - alignedCols) / elementsPerBlock), 0, 0};
        DataCopyPad(dst, src, params);
    }

    __aicore__ inline void CopyStridedBf16(LocalTensor<bfloat16_t> dst, GlobalTensor<bfloat16_t> src, uint32_t rows,
                                           uint32_t cols, uint32_t srcRowStride)
    {
        DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(bfloat16_t)),
                                 static_cast<uint32_t>((srcRowStride - cols) * sizeof(bfloat16_t)),
                                 static_cast<int64_t>((DIM - MmAlign(cols, MM_NZ_FRACTAL)) / MM_NZ_FRACTAL), 0};
        const uint32_t alignedCols = MmAlign(cols, MM_NZ_FRACTAL);
        DataCopyPadExtParams<bfloat16_t> pad{true, 0, static_cast<uint8_t>(alignedCols - cols),
                                             static_cast<bfloat16_t>(0)};
        DataCopyPad(dst, src, params, pad);
    }

    __aicore__ inline void CopyStridedFp32(LocalTensor<float> dst, GlobalTensor<float> src, uint32_t rows,
                                           uint32_t cols, uint32_t srcRowStride)
    {
        constexpr uint32_t elementsPerBlock = BLOCK_SIZE / sizeof(float);
        const uint32_t alignedCols = MmAlign(cols, elementsPerBlock);
        DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(float)),
                                 static_cast<uint32_t>((srcRowStride - cols) * sizeof(float)),
                                 static_cast<int64_t>((DIM - alignedCols) / elementsPerBlock), 0};
        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(alignedCols - cols), 0.0f};
        DataCopyPad(dst, src, params, pad);
    }

    __aicore__ inline void CopyBf16Vector(LocalTensor<bfloat16_t> dst, GlobalTensor<bfloat16_t> src, uint32_t count)
    {
        const uint32_t alignedCount = MmAlign(count, MM_NZ_FRACTAL);
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> pad{true, 0, static_cast<uint8_t>(alignedCount - count),
                                             static_cast<bfloat16_t>(0)};
        DataCopyPad(dst, src, params, pad);
    }

    __aicore__ inline void CopyFp32Scalar(LocalTensor<float> dst, GlobalTensor<float> src)
    {
        constexpr uint32_t elementsPerBlock = BLOCK_SIZE / sizeof(float);
        DataCopyExtParams params{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{true, 0, elementsPerBlock - 1, 0.0f};
        DataCopyPad(dst, src, params, pad);
    }

    __aicore__ inline void CopyStreamingOutputToGm(GlobalTensor<bfloat16_t> dst, LocalTensor<bfloat16_t> src,
                                                   uint32_t count)
    {
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(dst, src, params);
    }

    __aicore__ inline void ProcessStreamingAivHead(uint32_t nvId, uint32_t subBlockIdx)
    {
        const uint32_t stateRowBegin = subBlockIdx * HALF_DIM;
        const uint32_t stateValidRows = StateValidRows(subBlockIdx);
        if (stateValidRows == 0) {
            return;
        }
        if (currentLength_ == 0) {
            CopyEmptyStateAiv(nvId, subBlockIdx);
            return;
        }

        const uint32_t kvHead = nvId / (params_.nv / params_.nk);
        auto stateFp32 = resource_.ubBuf.template GetBufferByByte<float>(UB_STREAM_STATE_FP32_OFFSET);
        auto stateBf16 = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(UB_STREAM_STATE_BF16_OFFSET);
        auto query = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(UB_STREAM_Q_OFFSET);
        auto key = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(UB_STREAM_K_OFFSET);
        auto valueBf16 = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(UB_STREAM_V_BF16_OFFSET);
        auto valueFp32 = resource_.ubBuf.template GetBufferByByte<float>(UB_STREAM_V_FP32_OFFSET);
        auto betaBf16 = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(UB_STREAM_BETA_BF16_OFFSET);
        auto betaFp32 = resource_.ubBuf.template GetBufferByByte<float>(UB_STREAM_BETA_FP32_OFFSET);
        auto g = resource_.ubBuf.template GetBufferByByte<float>(UB_STREAM_G_OFFSET);
        auto outFp32 = resource_.ubBuf.template GetBufferByByte<float>(UB_STREAM_OUT_FP32_OFFSET);
        auto outBf16 = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(UB_STREAM_OUT_BF16_OFFSET);
        const MutexID stateMutex = MutexId(AIV_STATE_MUTEX);
        const MutexID tokenMutex = MutexId(AIV_INPUT_HEAD_MUTEX);

        Mutex::Lock<PIPE_V>(stateMutex);
        Duplicate(stateFp32, 0.0f, STATE_SLICE_ELEMS);
        PipeBarrier<PIPE_V>();
        Mutex::Unlock<PIPE_V>(stateMutex);

        const uint64_t initialStateOffset = currentBatchId_ * params_.stateStride0 +
                                            static_cast<uint64_t>(nvId) * params_.stateStride1 +
                                            static_cast<uint64_t>(stateRowBegin) * params_.dk;
        Mutex::Lock<PIPE_MTE2>(stateMutex);
        if constexpr (STATE_FP32) {
            CopyStridedFp32(stateFp32, params_.initialState[initialStateOffset], stateValidRows, params_.dk,
                            params_.dk);
        } else {
            CopyStridedBf16(stateBf16, params_.initialState[initialStateOffset], stateValidRows, params_.dk,
                            params_.dk);
        }
        Mutex::Unlock<PIPE_MTE2>(stateMutex);
        Mutex::Lock<PIPE_V>(stateMutex);
        if constexpr (!STATE_FP32) {
            Cast(stateFp32, stateBf16, RoundMode::CAST_NONE, stateValidRows * DIM);
            PipeBarrier<PIPE_V>();
        }

        for (uint32_t token = 0; token < currentLength_; ++token) {
            const uint64_t globalToken = currentSeqStart_ + token;
            const uint64_t qkOffset =
                globalToken * params_.nk * params_.dk + static_cast<uint64_t>(kvHead) * params_.dk;
            const uint64_t valueOffset =
                globalToken * params_.nv * params_.dv + static_cast<uint64_t>(nvId) * params_.dv + stateRowBegin;
            const uint64_t scalarOffset = globalToken * params_.nv + nvId;

            Mutex::Lock<PIPE_MTE2>(tokenMutex);
            CopyBf16Vector(query, params_.query[qkOffset], params_.dk);
            CopyBf16Vector(key, params_.key[qkOffset], params_.dk);
            CopyBf16Vector(valueBf16, params_.value[valueOffset], stateValidRows);
            CopyBf16Vector(betaBf16, params_.beta[scalarOffset], 1);
            if constexpr (hasGamma) {
                CopyFp32Scalar(g, params_.g[scalarOffset]);
            }
            Mutex::Unlock<PIPE_MTE2>(tokenMutex);

            Mutex::Lock<PIPE_V>(tokenMutex);
            Cast(valueFp32, valueBf16, RoundMode::CAST_NONE, stateValidRows);
            Cast(betaFp32, betaBf16, RoundMode::CAST_NONE, 1);
            PipeBarrier<PIPE_V>();
            auto stateAddr = reinterpret_cast<__ubuf__ float *>(stateFp32.GetPhyAddr());
            auto queryAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(query.GetPhyAddr());
            auto keyAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(key.GetPhyAddr());
            auto valueAddr = reinterpret_cast<__ubuf__ float *>(valueFp32.GetPhyAddr());
            auto betaAddr = reinterpret_cast<__ubuf__ float *>(betaFp32.GetPhyAddr());
            auto gAddr = reinterpret_cast<__ubuf__ float *>(g.GetPhyAddr());
            auto outAddr = reinterpret_cast<__ubuf__ float *>(outFp32.GetPhyAddr());
            AscendC::VF_CALL<StreamingExactTokenVF<hasGamma>>(
                stateAddr, queryAddr, keyAddr, valueAddr, betaAddr, gAddr, outAddr, params_.scale,
                static_cast<uint16_t>(stateValidRows), static_cast<uint16_t>(params_.dk), static_cast<uint16_t>(DIM));
            PipeBarrier<PIPE_V>();
            Cast(outBf16, outFp32, RoundMode::CAST_RINT, stateValidRows);
            PipeBarrier<PIPE_V>();
            Mutex::Unlock<PIPE_V>(tokenMutex);

            Mutex::Lock<PIPE_MTE3>(tokenMutex);
            const uint64_t outOffset =
                globalToken * params_.nv * params_.dv + static_cast<uint64_t>(nvId) * params_.dv + stateRowBegin;
            CopyStreamingOutputToGm(params_.out[outOffset], outBf16, stateValidRows);
            Mutex::Unlock<PIPE_MTE3>(tokenMutex);
        }

        if constexpr (!STATE_FP32) {
            Cast(stateBf16, stateFp32, RoundMode::CAST_RINT, stateValidRows * DIM);
            PipeBarrier<PIPE_V>();
        }
        Mutex::Unlock<PIPE_V>(stateMutex);
        Mutex::Lock<PIPE_MTE3>(stateMutex);
        const uint64_t finalStateOffset = (currentBatchId_ * params_.nv + nvId) * params_.dv * params_.dk +
                                          static_cast<uint64_t>(stateRowBegin) * params_.dk;
        if constexpr (STATE_FP32) {
            CopyRowMajorStateToGm(params_.finalState[finalStateOffset], stateFp32, stateValidRows);
        } else {
            CopyRowMajorStateToGm(params_.finalState[finalStateOffset], stateBf16, stateValidRows);
        }
        Mutex::Unlock<PIPE_MTE3>(stateMutex);
    }

    __aicore__ inline void CopyEmptyStateAiv(uint32_t nvId, uint32_t subBlockIdx)
    {
        const uint32_t stateRowBegin = subBlockIdx * HALF_DIM;
        const uint32_t stateValidRows = StateValidRows(subBlockIdx);
        if (stateValidRows == 0) {
            return;
        }
        const uint64_t initialStateOffset = currentBatchId_ * params_.stateStride0 +
                                            static_cast<uint64_t>(nvId) * params_.stateStride1 +
                                            static_cast<uint64_t>(stateRowBegin) * params_.dk;
        const uint64_t finalStateOffset = (currentBatchId_ * params_.nv + nvId) * params_.dv * params_.dk +
                                          static_cast<uint64_t>(stateRowBegin) * params_.dk;
        const MutexID stateMutex = MutexId(AIV_STATE_MUTEX);
        auto state = resource_.ubBuf.template GetBufferByByte<stateType>(UB_STATE_OFFSET);

        Mutex::Lock<PIPE_MTE2>(stateMutex);
        if constexpr (STATE_FP32) {
            CopyStridedFp32(state, params_.initialState[initialStateOffset], stateValidRows, params_.dk, params_.dk);
        } else {
            CopyStridedBf16(state, params_.initialState[initialStateOffset], stateValidRows, params_.dk, params_.dk);
        }
        Mutex::Unlock<PIPE_MTE2>(stateMutex);
        Mutex::Lock<PIPE_MTE3>(stateMutex);
        CopyRowMajorStateToGm(params_.finalState[finalStateOffset], state, stateValidRows);
        Mutex::Unlock<PIPE_MTE3>(stateMutex);
    }

    __aicore__ inline void IssueInputHeadMte2(uint32_t nvId, uint32_t kvHead, uint32_t chunk, uint32_t subBlockIdx,
                                              LocalTensor<bfloat16_t> q, LocalTensor<bfloat16_t> k,
                                              LocalTensor<bfloat16_t> v)
    {
        const uint32_t tokenBegin = subBlockIdx * HALF_CHUNK;
        const uint32_t validRows = ChunkValidRows(chunk);
        const uint32_t localValidRows = SubBlockValidRows(chunk, subBlockIdx);
        const uint32_t headValidRows = localValidRows < INPUT_BATCH_ROWS ? localValidRows : INPUT_BATCH_ROWS;
        const uint64_t tokenStart = currentSeqStart_ + static_cast<uint64_t>(chunk) * CHUNK;
        const uint64_t qkOffset = (tokenStart + tokenBegin) * params_.nk * params_.dk + kvHead * params_.dk;
        const uint64_t vOffset =
            (tokenStart + tokenBegin) * params_.nv * params_.dv + static_cast<uint64_t>(nvId) * params_.dv;
        auto rawG = resource_.ubBuf.template GetBufferByByte<float>(UB_RAW_G_OFFSET);
        auto rawBeta = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(UB_RAW_BETA_OFFSET);
        const MutexID inputHeadMutex = MutexId(AIV_INPUT_HEAD_MUTEX);
        Mutex::Lock<PIPE_MTE2>(inputHeadMutex);
        if (headValidRows > 0) {
            CopyStridedBf16(q, params_.query[qkOffset], headValidRows, params_.dk, params_.nk * params_.dk);
            CopyStridedBf16(k, params_.key[qkOffset], headValidRows, params_.dk, params_.nk * params_.dk);
            CopyStridedBf16(v, params_.value[vOffset], headValidRows, params_.dv, params_.nv * params_.dv);
        }

        if constexpr (hasGamma) {
            DataCopyExtParams gParams{static_cast<uint16_t>(validRows), static_cast<uint32_t>(sizeof(float)),
                                      static_cast<uint32_t>((params_.nv - 1) * sizeof(float)), 0, 0};
            DataCopyPadExtParams<float> gPad{true, 0, 7, 0.0f};
            DataCopyPad(rawG, params_.g[tokenStart * params_.nv + nvId], gParams, gPad);
        }
        if (localValidRows > 0) {
            DataCopyExtParams betaParams{static_cast<uint16_t>(localValidRows),
                                         static_cast<uint32_t>(sizeof(bfloat16_t)),
                                         static_cast<uint32_t>((params_.nv - 1) * sizeof(bfloat16_t)), 0, 0};
            DataCopyPadExtParams<bfloat16_t> betaPad{true, 0, 15, static_cast<bfloat16_t>(0)};
            DataCopyPad(rawBeta, params_.beta[(tokenStart + tokenBegin) * params_.nv + nvId], betaParams, betaPad);
        }
        Mutex::Unlock<PIPE_MTE2>(inputHeadMutex);
    }

    template <bool headPrefetched>
    __aicore__ inline void PrepareChunkInputs(uint32_t nvId, uint32_t kvHead, uint32_t chunk, uint32_t subBlockIdx,
                                              LocalTensor<bfloat16_t> q, LocalTensor<bfloat16_t> k,
                                              LocalTensor<bfloat16_t> v, LocalTensor<float> gCum,
                                              LocalTensor<float> gExp, LocalTensor<float> beta)
    {
        const uint32_t tokenBegin = subBlockIdx * HALF_CHUNK;
        const uint32_t validRows = ChunkValidRows(chunk);
        const uint32_t localValidRows = SubBlockValidRows(chunk, subBlockIdx);
        const uint32_t tailValidRows = localValidRows > INPUT_BATCH_ROWS ? localValidRows - INPUT_BATCH_ROWS : 0;
        const uint64_t tokenStart = currentSeqStart_ + static_cast<uint64_t>(chunk) * CHUNK;
        const uint64_t qkOffset = (tokenStart + tokenBegin) * params_.nk * params_.dk + kvHead * params_.dk;
        const uint64_t vOffset =
            (tokenStart + tokenBegin) * params_.nv * params_.dv + static_cast<uint64_t>(nvId) * params_.dv;
        auto rawG = resource_.ubBuf.template GetBufferByByte<float>(UB_RAW_G_OFFSET);
        auto rawBeta = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(UB_RAW_BETA_OFFSET);
        auto betaBf16 = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(UB_BETA_BF16_OFFSET);
        auto gather = resource_.ubBuf.template GetBufferByByte<uint32_t>(UB_GATHER_OFFSET);

        if constexpr (!headPrefetched) {
            IssueInputHeadMte2(nvId, kvHead, chunk, subBlockIdx, q, k, v);
        }

        const MutexID inputHeadMutex = MutexId(AIV_INPUT_HEAD_MUTEX);
        const MutexID inputTailMutex = MutexId(AIV_INPUT_TAIL_MUTEX);
        Mutex::Lock<PIPE_MTE2>(inputTailMutex);
        if (tailValidRows > 0) {
            CopyStridedBf16(q[INPUT_BATCH_ROWS * DIM],
                            params_.query[qkOffset + static_cast<uint64_t>(INPUT_BATCH_ROWS) * params_.nk * params_.dk],
                            tailValidRows, params_.dk, params_.nk * params_.dk);
            CopyStridedBf16(k[INPUT_BATCH_ROWS * DIM],
                            params_.key[qkOffset + static_cast<uint64_t>(INPUT_BATCH_ROWS) * params_.nk * params_.dk],
                            tailValidRows, params_.dk, params_.nk * params_.dk);
            CopyStridedBf16(v[INPUT_BATCH_ROWS * DIM],
                            params_.value[vOffset + static_cast<uint64_t>(INPUT_BATCH_ROWS) * params_.nv * params_.dv],
                            tailValidRows, params_.dv, params_.nv * params_.dv);
        }
        Mutex::Unlock<PIPE_MTE2>(inputTailMutex);

        // The caller consumes the first input half immediately after the
        // shared g/beta preprocessing, so retain V ownership across return.
        Mutex::Lock<PIPE_V>(inputHeadMutex);
        if constexpr (hasGamma) {
            if (validRows < CHUNK) {
                Duplicate(gCum, 0.0f, CHUNK);
            }
        } else {
            Duplicate(gCum, 0.0f, CHUNK);
            Duplicate(gExp, 1.0f, CHUNK);
        }
        if (localValidRows < HALF_CHUNK) {
            Duplicate(betaBf16, static_cast<bfloat16_t>(0), HALF_CHUNK);
        }
        if (validRows < CHUNK) {
            PipeBarrier<PIPE_V>();
        }
        if constexpr (hasGamma) {
            Gather(gCum, rawG, gather, static_cast<uint32_t>(0), validRows);
        }
        if (localValidRows > 0) {
            Gather(betaBf16, rawBeta, gather, static_cast<uint32_t>(0), localValidRows);
        }
        PipeBarrier<PIPE_V>();
        Cast(beta, betaBf16, RoundMode::CAST_NONE, HALF_CHUNK);
        PipeBarrier<PIPE_V>();
        if constexpr (hasGamma) {
            auto gAddr = reinterpret_cast<__ubuf__ float *>(gCum.GetPhyAddr());
            auto gExpAddr = reinterpret_cast<__ubuf__ float *>(gExp.GetPhyAddr());
            AscendC::VF_CALL<CumSumExpExactVF>(gAddr, gAddr, gExpAddr, static_cast<uint16_t>(CHUNK));
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildInverseSlice(uint32_t subBlockIdx, LocalTensor<float> attn, LocalTensor<float> inverse,
                                             LocalTensor<float> identity, LocalTensor<bfloat16_t> derived)
    {
        InverseAIVPanel8VF(attn, inverse, identity, subBlockIdx * HALF_CHUNK, CHUNK);
        PipeBarrier<PIPE_V>();
        auto inverseAddr = reinterpret_cast<__ubuf__ float *>(inverse.GetPhyAddr());
        auto derivedAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(derived.GetPhyAddr());
        AscendC::VF_CALL<CastFp32ToCompactZnBf16VF>(inverseAddr, derivedAddr, static_cast<uint16_t>(HALF_CHUNK),
                                                    static_cast<uint16_t>(CHUNK));
        PipeBarrier<PIPE_V>();
    }

    template <uint32_t directSlot>
    __aicore__ inline void WaitDirect()
    {
        static_assert(directSlot == 0, "only recurrence direct uses a fixed slot");
        CrossCoreWaitFlag(RECURRENCE_DIRECT0_READY_FLAG);
    }

    template <uint32_t directSlot>
    __aicore__ inline void ReleaseDirect()
    {
        static_assert(directSlot == 0, "only recurrence direct uses a fixed slot");
        CrossCoreSetFlag<0x2, PIPE_V>(RECURRENCE_DIRECT0_FREE_FLAG);
    }

    __aicore__ inline void WaitPrefixDirect(uint32_t slot)
    {
        CrossCoreWaitFlag(PrefixDirectReadyFlag(slot));
    }

    __aicore__ inline void ReleasePrefixDirect(uint32_t slot)
    {
        CrossCoreSetFlag<0x2, PIPE_V>(PrefixDirectFreeFlag(slot));
    }

    template <bool headPrefetched>
    __aicore__ inline void PreparePrefixAiv(uint32_t nvId, uint32_t kvHead, uint32_t chunk, uint32_t subBlockIdx,
                                            LocalTensor<float> attn, LocalTensor<float> inverse,
                                            LocalTensor<float> identity, LocalTensor<bfloat16_t> maskedQktPubBase,
                                            LocalTensor<bfloat16_t> q, LocalTensor<bfloat16_t> k,
                                            LocalTensor<bfloat16_t> v, LocalTensor<float> gExp, LocalTensor<float> beta)
    {
        const uint32_t slot = chunk & 1;
        const uint32_t slotBase = slot * L1_SLOT_BYTES;
        const uint32_t tokenBegin = subBlockIdx * HALF_CHUNK;
        const uint32_t localValidRows = SubBlockValidRows(chunk, subBlockIdx);
        const uint32_t headValidRows = localValidRows < INPUT_BATCH_ROWS ? localValidRows : INPUT_BATCH_ROWS;
        const uint32_t tailValidRows = localValidRows > INPUT_BATCH_ROWS ? localValidRows - INPUT_BATCH_ROWS : 0;
        const MutexID inputHeadMutex = MutexId(AIV_INPUT_HEAD_MUTEX);
        const MutexID inputTailMutex = MutexId(AIV_INPUT_TAIL_MUTEX);
        const MutexID stage1PubMutex = MutexId(AIV_STAGE1_PUB_MUTEX_BASE + slot);
        const MutexID prefixPubMutex = MutexId(AIV_PREFIX_PUB_MUTEX_BASE + slot);
        auto prefixDirect = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(PrefixDirectOffset(slot));
        auto gCum = resource_.ubBuf.template GetBufferByByte<float>(slot == 0 ? UB_GCUM0_OFFSET : UB_GCUM1_OFFSET);
        auto l1RawKcd = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(slotBase + L1_RAW_KCD_OFFSET);
        auto l1VBeta = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(slotBase + L1_VBETA_OFFSET);
        auto l1Kcd = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(slotBase + L1_KCD_OFFSET);
        auto l1Kg = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(slotBase + L1_KG_OFFSET);
        auto l1Attn = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(slotBase + L1_ATTN_OFFSET);
        auto l1MaskedQkt = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(slotBase + L1_MASKED_QKT_OFFSET);
        auto stage1Pub = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(slot == 0 ? UB_STAGE1_PUB0_OFFSET :
                                                                                          UB_STAGE1_PUB1_OFFSET);
        auto rawKcdPub = stage1Pub;
        auto vBetaPub = stage1Pub[UB_STAGE1_MATRIX_BYTES / sizeof(bfloat16_t)];
        auto qPrimePub = stage1Pub[2 * UB_STAGE1_MATRIX_BYTES / sizeof(bfloat16_t)];
        auto kgPub = stage1Pub[3 * UB_STAGE1_MATRIX_BYTES / sizeof(bfloat16_t)];
        auto inversePub = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(slot == 0 ? UB_INVERSE_PUB0_OFFSET :
                                                                                           UB_INVERSE_PUB1_OFFSET);
        auto maskedQktPub = maskedQktPubBase[slot * SCORE_SLICE_ELEMS];

        CrossCoreWaitFlag(SlotFreeFlag(slot));
        PrepareChunkInputs<headPrefetched>(nvId, kvHead, chunk, subBlockIdx, q, k, v, gCum, gExp, beta);

        auto qAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(q.GetPhyAddr());
        auto kAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(k.GetPhyAddr());
        auto vAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(v.GetPhyAddr());
        auto rawKcdAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(rawKcdPub.GetPhyAddr());
        auto vBetaPubAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(vBetaPub.GetPhyAddr());
        auto qPrimePubAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(qPrimePub.GetPhyAddr());
        auto kgPubAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(kgPub.GetPhyAddr());
        auto betaAddr = reinterpret_cast<__ubuf__ float *>(beta.GetPhyAddr());
        auto gCumAddr = reinterpret_cast<__ubuf__ float *>(gCum.GetPhyAddr());
        auto gExpAddr = reinterpret_cast<__ubuf__ float *>(gExp.GetPhyAddr());
        Mutex::Lock<PIPE_V>(stage1PubMutex);
        if (localValidRows < HALF_CHUNK || params_.dk < DIM || params_.dv < DIM) {
            Duplicate(stage1Pub, static_cast<bfloat16_t>(0), 4 * UB_STAGE1_MATRIX_BYTES / sizeof(bfloat16_t));
            PipeBarrier<PIPE_V>();
        }
        if (params_.dk == DIM && params_.dv == DIM) {
            AscendC::VF_CALL<PrepareStage1FullDimSliceVF>(
                qAddr, kAddr, vAddr, rawKcdAddr, vBetaPubAddr, qPrimePubAddr, kgPubAddr, betaAddr, gCumAddr, gExpAddr,
                params_.scale, static_cast<uint16_t>(tokenBegin), static_cast<uint16_t>(headValidRows),
                static_cast<uint16_t>(DIM), 0, static_cast<uint16_t>(HALF_CHUNK));
        } else {
            AscendC::VF_CALL<PrepareStage1SliceVF>(
                qAddr, kAddr, vAddr, rawKcdAddr, vBetaPubAddr, qPrimePubAddr, kgPubAddr, betaAddr, gCumAddr, gExpAddr,
                params_.scale, static_cast<uint16_t>(tokenBegin), static_cast<uint16_t>(headValidRows),
                static_cast<uint16_t>(params_.dk), static_cast<uint16_t>(params_.dv), static_cast<uint16_t>(DIM),
                static_cast<uint16_t>(DIM), 0, static_cast<uint16_t>(HALF_CHUNK));
        }
        Mutex::Unlock<PIPE_V>(inputHeadMutex);
        // The first 16 rows are dead after this VF call. Reuse their existing
        // Mutex lifecycle to load the next chunk while PIPE_V consumes the
        // disjoint tail rows. The next invocation acquires the same mutex on
        // PIPE_V, so no event token is carried across chunks.
        if (chunk + 1 < ChunkNum()) {
            IssueInputHeadMte2(nvId, kvHead, chunk + 1, subBlockIdx, q, k, v);
        }
        Mutex::Lock<PIPE_V>(inputTailMutex);
        if (params_.dk == DIM && params_.dv == DIM) {
            AscendC::VF_CALL<PrepareStage1FullDimSliceVF>(
                qAddr + INPUT_BATCH_ROWS * DIM, kAddr + INPUT_BATCH_ROWS * DIM, vAddr + INPUT_BATCH_ROWS * DIM,
                rawKcdAddr, vBetaPubAddr, qPrimePubAddr, kgPubAddr, betaAddr + INPUT_BATCH_ROWS, gCumAddr, gExpAddr,
                params_.scale, static_cast<uint16_t>(tokenBegin + INPUT_BATCH_ROWS),
                static_cast<uint16_t>(tailValidRows), static_cast<uint16_t>(DIM),
                static_cast<uint16_t>(INPUT_BATCH_ROWS), static_cast<uint16_t>(HALF_CHUNK));
        } else {
            AscendC::VF_CALL<PrepareStage1SliceVF>(
                qAddr + INPUT_BATCH_ROWS * DIM, kAddr + INPUT_BATCH_ROWS * DIM, vAddr + INPUT_BATCH_ROWS * DIM,
                rawKcdAddr, vBetaPubAddr, qPrimePubAddr, kgPubAddr, betaAddr + INPUT_BATCH_ROWS, gCumAddr, gExpAddr,
                params_.scale, static_cast<uint16_t>(tokenBegin + INPUT_BATCH_ROWS),
                static_cast<uint16_t>(tailValidRows), static_cast<uint16_t>(params_.dk),
                static_cast<uint16_t>(params_.dv), static_cast<uint16_t>(DIM), static_cast<uint16_t>(DIM),
                static_cast<uint16_t>(INPUT_BATCH_ROWS), static_cast<uint16_t>(HALF_CHUNK));
        }
        PipeBarrier<PIPE_V>();
        Mutex::Unlock<PIPE_V>(inputTailMutex);
        Mutex::Unlock<PIPE_V>(stage1PubMutex);
        // These Stage1 results do not depend on the prefix MM. Queue their UB->L1 copies now so MTE3 can
        // overlap the KKT/QKT wait and the following inverse construction. PrefixReadyFlag remains
        // ordered after the dependent inverse/qkt copies below on the same MTE3 pipeline.
        Mutex::Lock<PIPE_MTE3>(stage1PubMutex);
        CopyCompactZnToL1<HALF_CHUNK, DIM, CHUNK, 2>(rawKcdPub, l1RawKcd, tokenBegin);
        CopyCompactZnToL1<HALF_CHUNK, DIM, DIM>(qPrimePub, l1Kcd, CHUNK + tokenBegin);
        CopyCompactZnToL1<HALF_CHUNK, DIM, CHUNK>(kgPub, l1Kg, tokenBegin);
        Mutex::Unlock<PIPE_MTE3>(stage1PubMutex);

        WaitPrefixDirect(slot);
        auto attnAddr = reinterpret_cast<__ubuf__ float *>(attn.GetPhyAddr());
        auto inverseAddr = reinterpret_cast<__ubuf__ float *>(inverse.GetPhyAddr());
        auto maskedQktPubAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(maskedQktPub.GetPhyAddr());
        auto kktDirectAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(prefixDirect.GetPhyAddr());
        auto qktDirectAddr = kktDirectAddr + SCORE_SLICE_ELEMS;
        Mutex::Lock<PIPE_V>(prefixPubMutex);
        AscendC::VF_CALL<BuildAttnMaskedQktExactSliceVF>(
            attnAddr, inverseAddr, maskedQktPubAddr, kktDirectAddr, qktDirectAddr, gCumAddr, betaAddr, params_.scale,
            static_cast<uint16_t>(tokenBegin), static_cast<uint16_t>(HALF_CHUNK), static_cast<uint16_t>(CHUNK));
        PipeBarrier<PIPE_V>();
        BuildInverseSlice(subBlockIdx, attn, inverse, identity, inversePub);
        ReleasePrefixDirect(slot);
        Mutex::Unlock<PIPE_V>(prefixPubMutex);
        Mutex::Lock<PIPE_MTE3>(prefixPubMutex);
        CopyCompactZnToL1<HALF_CHUNK, CHUNK, CHUNK>(inversePub, l1Attn, tokenBegin);
        CrossCoreSetFlag<0x2, PIPE_MTE3>(PrefixCriticalReadyFlag(slot));
        CopyCompactZnToL1<HALF_CHUNK, CHUNK, CHUNK>(maskedQktPub, l1MaskedQkt, tokenBegin);
        CrossCoreSetFlag<0x2, PIPE_MTE3>(PrefixReadyFlag(slot));
        Mutex::Unlock<PIPE_MTE3>(prefixPubMutex);
    }

    __aicore__ inline void PublishInitialStateAiv(uint32_t subBlockIdx, LocalTensor<bfloat16_t> rowMajorState,
                                                  LocalTensor<bfloat16_t> compactState)
    {
        const uint32_t stateRowBegin = subBlockIdx * HALF_DIM;
        const uint32_t validRows = StateValidRows(subBlockIdx);
        auto l1State = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(L1_STATE_OFFSET);
        const MutexID stateMutex = MutexId(AIV_STATE_MUTEX);
        Mutex::Lock<PIPE_V>(stateMutex);
        if (validRows < HALF_DIM || params_.dk < DIM) {
            Duplicate(compactState, static_cast<bfloat16_t>(0), STATE_SLICE_ELEMS);
            PipeBarrier<PIPE_V>();
        }
        auto rowMajorAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(rowMajorState.GetPhyAddr());
        auto compactAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(compactState.GetPhyAddr());
        AscendC::VF_CALL<RowMajorToCompactZnBf16VF>(rowMajorAddr, compactAddr, static_cast<uint16_t>(validRows),
                                                    static_cast<uint16_t>(params_.dk), static_cast<uint16_t>(DIM),
                                                    static_cast<uint16_t>(HALF_DIM));
        PipeBarrier<PIPE_V>();
        Mutex::Unlock<PIPE_V>(stateMutex);
        Mutex::Lock<PIPE_MTE3>(stateMutex);
        CopyCompactZnToL1<HALF_DIM, DIM, DIM>(compactState, l1State, stateRowBegin);
        CrossCoreSetFlag<0x2, PIPE_MTE3>(STATE_READY_FLAG);
        Mutex::Unlock<PIPE_MTE3>(stateMutex);
    }

    __aicore__ inline void PublishCompactStateAiv(uint32_t subBlockIdx, LocalTensor<bfloat16_t> state)
    {
        const uint32_t stateRowBegin = subBlockIdx * HALF_DIM;
        auto l1State = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(L1_STATE_OFFSET);
        const MutexID stateMutex = MutexId(AIV_STATE_MUTEX);
        Mutex::Unlock<PIPE_V>(stateMutex);
        Mutex::Lock<PIPE_MTE3>(stateMutex);
        CopyCompactZnToL1<HALF_DIM, DIM, DIM>(state, l1State, stateRowBegin);
        CrossCoreSetFlag<0x2, PIPE_MTE3>(STATE_READY_FLAG);
        Mutex::Unlock<PIPE_MTE3>(stateMutex);
    }

    template <bool stateMutexHeld>
    __aicore__ inline void PublishFp32StateAiv(uint32_t subBlockIdx, LocalTensor<float> state)
    {
        const uint32_t stateRowBegin = subBlockIdx * HALF_DIM;
        const uint32_t validRows = StateValidRows(subBlockIdx);
        auto statePub = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(UB_STATE_PUB_OFFSET);
        auto l1State = resource_.l1Buf.template GetBufferByByte<bfloat16_t>(L1_STATE_OFFSET);
        const MutexID stateMutex = MutexId(AIV_STATE_MUTEX);
        if constexpr (!stateMutexHeld) {
            Mutex::Lock<PIPE_V>(stateMutex);
        }
        if (validRows < HALF_DIM || params_.dk < DIM) {
            Duplicate(statePub, static_cast<bfloat16_t>(0), STATE_SLICE_ELEMS);
            PipeBarrier<PIPE_V>();
        }
        auto stateAddr = reinterpret_cast<__ubuf__ float *>(state.GetPhyAddr());
        auto statePubAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(statePub.GetPhyAddr());
        AscendC::VF_CALL<RowMajorFp32ToCompactZnBf16VF>(stateAddr, statePubAddr, static_cast<uint16_t>(validRows),
                                                        static_cast<uint16_t>(params_.dk), static_cast<uint16_t>(DIM),
                                                        static_cast<uint16_t>(HALF_DIM));
        PipeBarrier<PIPE_V>();
        Mutex::Unlock<PIPE_V>(stateMutex);
        Mutex::Lock<PIPE_MTE3>(stateMutex);
        CopyCompactZnToL1<HALF_DIM, DIM, DIM>(statePub, l1State, stateRowBegin);
        CrossCoreSetFlag<0x2, PIPE_MTE3>(STATE_READY_FLAG);
        Mutex::Unlock<PIPE_MTE3>(stateMutex);
    }

    __aicore__ inline void ProcessRecurrenceAiv(uint32_t nvId, uint32_t kvHead, uint32_t chunk, uint32_t subBlockIdx,
                                                LocalTensor<stateType> state, LocalTensor<float> attn,
                                                LocalTensor<float> inverse, LocalTensor<float> identity,
                                                LocalTensor<bfloat16_t> maskedQktPubBase, LocalTensor<bfloat16_t> q,
                                                LocalTensor<bfloat16_t> k, LocalTensor<bfloat16_t> v,
                                                LocalTensor<float> gExp, LocalTensor<float> beta)
    {
        const uint32_t slot = chunk & 1;
        auto gCum = resource_.ubBuf.template GetBufferByByte<float>(slot == 0 ? UB_GCUM0_OFFSET : UB_GCUM1_OFFSET);

        if (chunk + 1 < ChunkNum()) {
            PreparePrefixAiv<true>(nvId, kvHead, chunk + 1, subBlockIdx, attn, inverse, identity, maskedQktPubBase, q,
                                   k, v, gExp, beta);
        }

        WaitDirect<0>();
        CrossCoreWaitFlag(STATE_FREE_FLAG);
        Mutex::Lock<PIPE_V>(MutexId(AIV_STATE_MUTEX));
        auto gCumAddr = reinterpret_cast<__ubuf__ float *>(gCum.GetPhyAddr());
        auto direct = resource_.ubBuf.template GetBufferByByte<stateType>(RecurrenceDirectOffset(slot));
        if constexpr (STATE_FP32) {
            auto stateAddr = reinterpret_cast<__ubuf__ float *>(state.GetPhyAddr());
            auto directAddr = reinterpret_cast<__ubuf__ float *>(direct.GetPhyAddr());
            AscendC::VF_CALL<UpdateFp32StateByExpVF>(stateAddr, directAddr, gCumAddr + CHUNK - 1, STATE_SLICE_ELEMS);
        } else {
            auto stateAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(state.GetPhyAddr());
            auto directAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(direct.GetPhyAddr());
            AscendC::VF_CALL<UpdateBf16StateByExpVF>(stateAddr, directAddr, gCumAddr + CHUNK - 1, STATE_SLICE_ELEMS);
        }
        PipeBarrier<PIPE_V>();
        ReleaseDirect<0>();
        if (chunk + 1 < ChunkNum()) {
            if constexpr (STATE_FP32) {
                PublishFp32StateAiv<true>(subBlockIdx, state);
            } else {
                PublishCompactStateAiv(subBlockIdx, state);
            }
        }
    }

    __aicore__ inline void ProcessAivHead(uint32_t nvId, uint32_t subBlockIdx)
    {
        if (UseStreamingExact()) {
            ProcessStreamingAivHead(nvId, subBlockIdx);
            return;
        }
        const uint32_t kvHead = nvId / (params_.nv / params_.nk);
        const uint32_t chunkNum = ChunkNum();
        if (chunkNum == 0) {
            CopyEmptyStateAiv(nvId, subBlockIdx);
            return;
        }
        const uint32_t stateRowBegin = subBlockIdx * HALF_DIM;
        const uint32_t stateValidRows = StateValidRows(subBlockIdx);
        auto state = resource_.ubBuf.template GetBufferByByte<stateType>(UB_STATE_OFFSET);
        auto attn = resource_.ubBuf.template GetBufferByByte<float>(UB_ATTN_OFFSET);
        auto inverse = resource_.ubBuf.template GetBufferByByte<float>(UB_INVERSE_OFFSET);
        auto identity = resource_.ubBuf.template GetBufferByByte<float>(UB_IDENTITY_OFFSET);
        auto maskedQktPubBase = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(UB_MASKED_QKT_PUB_OFFSET);
        auto q = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(UB_Q_OFFSET);
        auto k = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(UB_K_OFFSET);
        auto v = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(UB_V_OFFSET);
        auto gExp = resource_.ubBuf.template GetBufferByByte<float>(UB_GEXP_OFFSET);
        auto beta = resource_.ubBuf.template GetBufferByByte<float>(UB_BETA_FP32_OFFSET);
        auto gather = resource_.ubBuf.template GetBufferByByte<uint32_t>(UB_GATHER_OFFSET);

        const MutexID scalarMutex = MutexId(AIV_SCALAR_MUTEX);
        const MutexID stateMutex = MutexId(AIV_STATE_MUTEX);
        Mutex::Lock<PIPE_V>(scalarMutex);
        Duplicate(identity, 0.0f, INVERSE_SLICE_ELEMS);
        Mutex::Unlock<PIPE_V>(scalarMutex);
        Mutex::Lock<PIPE_S>(scalarMutex);
        for (uint32_t i = 0; i < HALF_CHUNK; ++i) {
            identity.SetValue(i * CHUNK + i, 1.0f);
        }
        for (uint32_t i = 0; i < CHUNK; ++i) {
            gather.SetValue(i, i * BLOCK_SIZE);
        }
        Mutex::Unlock<PIPE_S>(scalarMutex);
        // No scalar writer exists after initialization. Retain V ownership for
        // all identity/gather consumers and release it at the end of this head.
        Mutex::Lock<PIPE_V>(scalarMutex);

        const uint64_t initialStateOffset = currentBatchId_ * params_.stateStride0 +
                                            static_cast<uint64_t>(nvId) * params_.stateStride1 +
                                            static_cast<uint64_t>(stateRowBegin) * params_.dk;
        if constexpr (STATE_FP32) {
            Mutex::Lock<PIPE_V>(stateMutex);
            Duplicate(state, 0.0f, STATE_SLICE_ELEMS);
            Mutex::Unlock<PIPE_V>(stateMutex);
            Mutex::Lock<PIPE_MTE2>(stateMutex);
            if (stateValidRows > 0) {
                CopyStridedFp32(state, params_.initialState[initialStateOffset], stateValidRows, params_.dk,
                                params_.dk);
            }
            Mutex::Unlock<PIPE_MTE2>(stateMutex);
        } else {
            auto rowMajorState = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(UB_DIRECT0_OFFSET);
            Mutex::Lock<PIPE_MTE2>(stateMutex);
            if (stateValidRows > 0) {
                CopyStridedBf16(rowMajorState, params_.initialState[initialStateOffset], stateValidRows, params_.dk,
                                params_.dk);
            }
            Mutex::Unlock<PIPE_MTE2>(stateMutex);
        }

        PreparePrefixAiv<false>(nvId, kvHead, 0, subBlockIdx, attn, inverse, identity, maskedQktPubBase, q, k, v, gExp,
                                beta);
        if constexpr (STATE_FP32) {
            PublishFp32StateAiv<false>(subBlockIdx, state);
        } else {
            auto rowMajorState = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(UB_DIRECT0_OFFSET);
            PublishInitialStateAiv(subBlockIdx, rowMajorState, state);
        }
        for (uint32_t chunk = 0; chunk < chunkNum; ++chunk) {
            ProcessRecurrenceAiv(nvId, kvHead, chunk, subBlockIdx, state, attn, inverse, identity, maskedQktPubBase, q,
                                 k, v, gExp, beta);
        }

        Mutex::Unlock<PIPE_V>(stateMutex);
        Mutex::Lock<PIPE_MTE3>(stateMutex);
        if (stateValidRows > 0) {
            const uint64_t finalStateOffset = (currentBatchId_ * params_.nv + nvId) * params_.dv * params_.dk +
                                              static_cast<uint64_t>(stateRowBegin) * params_.dk;
            if constexpr (STATE_FP32) {
                CopyRowMajorStateToGm(params_.finalState[finalStateOffset], state, stateValidRows);
            } else {
                CopyCompactStateToGm(params_.finalState[finalStateOffset], state, stateValidRows);
            }
        }
        Mutex::Unlock<PIPE_MTE3>(stateMutex);
        Mutex::Unlock<PIPE_V>(scalarMutex);
    }

    __aicore__ inline void ProcessAiv()
    {
        const uint32_t subBlockIdx = static_cast<uint32_t>(GetSubBlockIdx());
        const uint32_t mixedGroupIdx = static_cast<uint32_t>(GetBlockIdx()) / static_cast<uint32_t>(GetSubBlockNum());
        const uint64_t totalWork = params_.batchCount * params_.nv;
        uint64_t scannedBatch = 0;
        uint64_t seqStart = 0;
        for (uint64_t workId = mixedGroupIdx; workId < totalWork; workId += params_.aiCoreNum) {
            const uint32_t nvId = SetWorkContext(workId, scannedBatch, seqStart);
            ProcessAivHead(nvId, subBlockIdx);
        }
    }

private:
    FusedStage123ExactParams<stateType> params_;
    BufferViews resource_;
    uint32_t l0abStage_;
    uint64_t currentBatchId_;
    uint64_t currentSeqStart_;
    uint32_t currentLength_;
};

} // namespace ChunkGatedDeltaRule

#endif // CHUNK_GATED_DELTA_RULE_FUSED_EXACT_ARCH35_H
