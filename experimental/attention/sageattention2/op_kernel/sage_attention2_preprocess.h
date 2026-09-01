/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software and is licensed under the CANN Open Software License Agreement
 * Version 2.0. THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND.
 */

#ifndef SAGE_ATTENTION2_PREPROCESS_H_
#define SAGE_ATTENTION2_PREPROCESS_H_

#ifndef SAGE_ATTENTION2_K_QUANT_PROBE
#define SAGE_ATTENTION2_K_QUANT_PROBE 0
#endif

#include "sage_attention2_common.h"

namespace SageAttention2Kernel {

template <typename InputT, bool LAYOUT_NHD, bool PER_THREAD, bool SMOOTH_K, bool SMOOTH_V>
class Preprocess {
public:
    static constexpr uint32_t K_QUANT_PROBE = SAGE_ATTENTION2_K_QUANT_PROBE;
    static_assert(K_QUANT_PROBE <= 3U, "unsupported SageAttention2 K quantization probe");
    static_assert(K_QUANT_PROBE == 0U || SAGE_ATTENTION2_DEBUG_STAGE == 22,
        "SageAttention2 K quantization probes are valid only with debug stage 22");

    template <typename ResourceT>
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR userWorkspace,
        const SageAttention2TilingData *tiling, ResourceT &resource)
    {
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(query));
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(key));
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(value));
        qTyped_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(query));
        kTyped_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(key));
        vTyped_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(value));
        q8_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(userWorkspace + tiling->workspace.q8Offset));
        k8_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(userWorkspace + tiling->workspace.k8Offset));
        v8_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(userWorkspace + tiling->workspace.v8Offset));
        qScale_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(userWorkspace + tiling->workspace.qScaleOffset));
        kScale_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(userWorkspace + tiling->workspace.kScaleOffset));
        vScale_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(userWorkspace + tiling->workspace.vScaleOffset));
        if constexpr (SMOOTH_K) {
            kMean_.SetGlobalBuffer(
                reinterpret_cast<__gm__ uint16_t *>(userWorkspace + tiling->workspace.kMeanOffset));
            kMeanTyped_.SetGlobalBuffer(
                reinterpret_cast<__gm__ InputT *>(userWorkspace + tiling->workspace.kMeanOffset));
        }
        if constexpr (SMOOTH_V) {
            vMean_.SetGlobalBuffer(
                reinterpret_cast<__gm__ float *>(userWorkspace + tiling->workspace.vMeanOffset));
        }
        batch_ = tiling->shape.batch;
        qHeads_ = tiling->shape.qHeads;
        kvHeads_ = tiling->shape.kvHeads;
        qSeq_ = tiling->shape.qSeq;
        kvSeq_ = tiling->shape.kvSeq;
        dim_ = tiling->shape.headDim;
        dimPad_ = tiling->shape.headDimPad;
        kvSeqPad_ = tiling->shape.kvSeqPad;
        qBlocks_ = tiling->shape.qBlocks;
        qGroups_ = tiling->cores.qScaleGroupsPerBlock;
        kGroups_ = tiling->cores.kScaleGroupsPerBlock;
        vecCoreNum_ = tiling->cores.vecCoreNum;
        epsilon_ = tiling->attrs.quantEpsilon;
        vScaleMax_ = tiling->attrs.vScaleMax;
        layout_ = LAYOUT_NHD ? Layout::NHD : Layout::HND;
        gran_ = PER_THREAD ? QuantGran::PER_THREAD : QuantGran::PER_WARP;
        // Preprocess and attention execute in disjoint phases, so preprocessing can use a larger view of the
        // shared UB. V encoding saves its sign mask in the consumed input16 area and aliases vCandidate with
        // the consumed inputFloat area, allowing a complete 128-row batch within 248 KiB.
        input16Ub_ = resource.ubBuf.template GetBufferByByte<InputT>(0U);
        inputFloatUb_ = resource.ubBuf.template GetBufferByByte<float>(32768U);
        vCandidateUb_ = resource.ubBuf.template GetBufferByByte<float>(32768U);
        quantAbsUb_ = resource.ubBuf.template GetBufferByByte<float>(98304U);
        quantWorkUb_ = resource.ubBuf.template GetBufferByByte<float>(163840U);
        quantInt8Ub_ = resource.ubBuf.template GetBufferByByte<int8_t>(229376U);
        statsSumUb_ = resource.ubBuf.template GetBufferByByte<float>(245760U);
        statsMaxUb_ = resource.ubBuf.template GetBufferByByte<float>(246272U);
        mean16Ub_ = resource.ubBuf.template GetBufferByByte<InputT>(246784U);
        meanFloatUb_ = resource.ubBuf.template GetBufferByByte<float>(247040U);
        channelScaleUb_ = resource.ubBuf.template GetBufferByByte<float>(247552U);
        fp8PackIndexUb_ = resource.ubBuf.template GetBufferByByte<uint8_t>(248064U);
    }

    __aicore__ inline void ComputeStats()
    {
        if ASCEND_IS_AIC {
            return;
        }
        if constexpr (!LAYOUT_NHD && !SMOOTH_V) {
            ComputeStatsHnd();
            if (OverlapQQuantWithStats()) {
                const uint32_t vecId = AscendC::GetBlockIdx();
                const uint32_t statsCores = kvHeads_;
                if (vecId >= statsCores) {
                    QuantizeQHndPerThreadContiguous(vecId - statsCores, vecCoreNum_ - statsCores);
                }
            }
            return;
        }
        const uint32_t vecId = AscendC::GetBlockIdx();
        const uint64_t channelCount = static_cast<uint64_t>(batch_) * kvHeads_ * dimPad_;
        // Scalar GM stores are cached. Give each AIV whole 64-byte cache-line groups so the DCCI at the
        // phase boundary cannot write back stale peer-core copies over another core's results.
        constexpr uint32_t CHANNELS_PER_CACHE_GROUP = 32U;
        for (uint64_t taskBase = static_cast<uint64_t>(vecId) * CHANNELS_PER_CACHE_GROUP;
             taskBase < channelCount;
             taskBase += static_cast<uint64_t>(vecCoreNum_) * CHANNELS_PER_CACHE_GROUP) {
          const uint64_t taskEnd = Min(channelCount, taskBase + CHANNELS_PER_CACHE_GROUP);
          for (uint64_t task = taskBase; task < taskEnd; ++task) {
            const uint32_t d = task % dimPad_;
            const uint64_t headLinear = task / dimPad_;
            const uint32_t h = headLinear % kvHeads_;
            const uint32_t b = headLinear / kvHeads_;
            float kSum = 0.0F;
            float vSum = 0.0F;
            if (d < dim_) {
                for (uint32_t s = 0U; s < kvSeq_; ++s) {
                    if constexpr (SMOOTH_K) {
                        kSum += DecodeInput16<InputT>(
                            k_.GetValue(SourceOffset<InputT>(layout_, b, h, s, d, kvHeads_, kvSeq_, dim_)));
                    }
                    if constexpr (SMOOTH_V) {
                        vSum += DecodeInput16<InputT>(
                            v_.GetValue(SourceOffset<InputT>(layout_, b, h, s, d, kvHeads_, kvSeq_, dim_)));
                    }
                }
            }
            const float kMeanValue = d < dim_ ? kSum / static_cast<float>(kvSeq_) : 0.0F;
            const float vMeanValue = d < dim_ ? vSum / static_cast<float>(kvSeq_) : 0.0F;
            if constexpr (SMOOTH_K) {
                kMean_.SetValue(task, EncodeInput16<InputT>(kMeanValue));
            }
            if constexpr (SMOOTH_V) {
                vMean_.SetValue(task, vMeanValue);
            }
            float maxAbs = 0.0F;
            if (d < dim_) {
                for (uint32_t s = 0U; s < kvSeq_; ++s) {
                    float value = DecodeInput16<InputT>(
                        v_.GetValue(SourceOffset<InputT>(layout_, b, h, s, d, kvHeads_, kvSeq_, dim_)));
                    if constexpr (SMOOTH_V) {
                        value -= vMeanValue;
                    }
                    maxAbs = Max(maxAbs, Abs(value));
                }
            }
            vScale_.SetValue(task, Max(maxAbs / vScaleMax_, epsilon_));
          }
        }
    }

    __aicore__ inline void Quantize()
    {
        if ASCEND_IS_AIC {
            return;
        }
        if constexpr (!LAYOUT_NHD) {
            if (dim_ != dimPad_) {
                // HND MTE2 copies only D columns into each Dpad-strided row. Q/K reuse this 32 KiB UB view,
                // so one initialization keeps every padding lane zero for all following bulk Cast/amax calls.
                constexpr uint32_t INPUT16_ELEMENTS = 32768U / sizeof(InputT);
                AscendC::Duplicate(input16Ub_, static_cast<InputT>(0.0F), INPUT16_ELEMENTS);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            }
        }
        if (!OverlapQQuantWithStats()) {
            QuantizeQ();
        }
#if SAGE_ATTENTION2_DEBUG_STAGE == 21
        return;
#endif
        QuantizeK();
#if SAGE_ATTENTION2_DEBUG_STAGE == 22
        return;
#endif
        QuantizeV();
    }

private:
    __aicore__ inline bool OverlapQQuantWithStats() const
    {
        if constexpr (LAYOUT_NHD || !PER_THREAD || SMOOTH_V) {
            return false;
        }
        return batch_ == 1U && kvHeads_ <= 8U && dim_ == 128U && dimPad_ == 128U && qGroups_ == 32U &&
            (qSeq_ % Q_QUANT_BLOCK) == 0U && kvHeads_ < vecCoreNum_;
    }

    template <bool SUM>
    __simd_vf__ static inline void ReduceHndColumnsVf(
        __ubuf__ float *src, __ubuf__ float *accum, uint16_t rows)
    {
        using namespace AscendC::MicroAPI;
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> accumEven;
        RegTensor<float> accumOdd;
        RegTensor<float> valueEven;
        RegTensor<float> valueOdd;
        RegTensor<float> contiguous0;
        RegTensor<float> contiguous1;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(accumEven, accumOdd, accum);
        for (uint16_t row = 0U; row < rows; ++row) {
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
                valueEven, valueOdd, src + static_cast<uint32_t>(row) * 128U);
            if constexpr (SUM) {
                Add(accumEven, accumEven, valueEven, full);
                Add(accumOdd, accumOdd, valueOdd, full);
            } else {
                Abs(valueEven, valueEven, full);
                Abs(valueOdd, valueOdd, full);
                Max(accumEven, accumEven, valueEven, full);
                Max(accumOdd, accumOdd, valueOdd, full);
            }
        }
        Interleave(contiguous0, contiguous1, accumEven, accumOdd);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(accum, contiguous0, full);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(accum + 64U, contiguous1, full);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    __aicore__ inline void ComputeStatsHndBatched()
    {
        constexpr uint32_t ROWS_PER_BATCH = 128U;
        constexpr uint32_t ELEMENTS_PER_BATCH = ROWS_PER_BATCH * 128U;
        const uint32_t vecId = AscendC::GetBlockIdx();
        const uint64_t headCount = static_cast<uint64_t>(batch_) * kvHeads_;
        for (uint64_t task = vecId; task < headCount; task += vecCoreNum_) {
            const uint32_t h = task % kvHeads_;
            const uint32_t b = task / kvHeads_;
            AscendC::Duplicate(statsSumUb_, 0.0F, dimPad_);
            AscendC::Duplicate(statsMaxUb_, 0.0F, dimPad_);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t rowBegin = 0U; rowBegin < kvSeq_; rowBegin += ROWS_PER_BATCH) {
                if constexpr (SMOOTH_K) {
                    AscendC::DataCopy(input16Ub_,
                        kTyped_[SourceOffset<InputT>(
                            Layout::HND, b, h, rowBegin, 0U, kvHeads_, kvSeq_, dim_)], ELEMENTS_PER_BATCH);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::Cast(inputFloatUb_, input16Ub_, AscendC::RoundMode::CAST_NONE, ELEMENTS_PER_BATCH);
                    AscendC::PipeBarrier<PIPE_V>();
                    ReduceHndColumnsVf<true>(
                        reinterpret_cast<__ubuf__ float *>(inputFloatUb_.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(statsSumUb_.GetPhyAddr()), ROWS_PER_BATCH);
                }
                AscendC::DataCopy(input16Ub_,
                    vTyped_[SourceOffset<InputT>(
                        Layout::HND, b, h, rowBegin, 0U, kvHeads_, kvSeq_, dim_)], ELEMENTS_PER_BATCH);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::Cast(inputFloatUb_, input16Ub_, AscendC::RoundMode::CAST_NONE, ELEMENTS_PER_BATCH);
                AscendC::PipeBarrier<PIPE_V>();
                ReduceHndColumnsVf<false>(
                    reinterpret_cast<__ubuf__ float *>(inputFloatUb_.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(statsMaxUb_.GetPhyAddr()), ROWS_PER_BATCH);
            }
            if constexpr (SMOOTH_K) {
                AscendC::Muls(statsSumUb_, statsSumUb_, 1.0F / static_cast<float>(kvSeq_), dim_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(input16Ub_, statsSumUb_, AscendC::RoundMode::CAST_RINT, dimPad_);
            }
            AscendC::Muls(statsMaxUb_, statsMaxUb_, 1.0F / vScaleMax_, dim_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Maxs(statsMaxUb_, statsMaxUb_, epsilon_, dimPad_);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            const uint64_t channelOffset = task * dimPad_;
            if constexpr (SMOOTH_K) {
                AscendC::DataCopy(kMeanTyped_[channelOffset], input16Ub_, dimPad_);
            }
            AscendC::DataCopy(vScale_[channelOffset], statsMaxUb_, dimPad_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

    __aicore__ inline void ComputeStatsHnd()
    {
        if (dim_ == 128U && dimPad_ == 128U && (kvSeq_ % 128U) == 0U) {
            ComputeStatsHndBatched();
            return;
        }
        const uint32_t vecId = AscendC::GetBlockIdx();
        const uint64_t headCount = static_cast<uint64_t>(batch_) * kvHeads_;
        for (uint64_t task = vecId; task < headCount; task += vecCoreNum_) {
            const uint32_t h = task % kvHeads_;
            const uint32_t b = task / kvHeads_;
            AscendC::Duplicate(statsSumUb_, 0.0F, dimPad_);
            AscendC::Duplicate(statsMaxUb_, 0.0F, dimPad_);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t s = 0U; s < kvSeq_; ++s) {
                if constexpr (SMOOTH_K) {
                    AscendC::DataCopy(input16Ub_,
                        kTyped_[SourceOffset<InputT>(Layout::HND, b, h, s, 0U, kvHeads_, kvSeq_, dim_)], dim_);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::Cast(inputFloatUb_, input16Ub_, AscendC::RoundMode::CAST_NONE, dim_);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(statsSumUb_, statsSumUb_, inputFloatUb_, dim_);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::DataCopy(input16Ub_,
                    vTyped_[SourceOffset<InputT>(Layout::HND, b, h, s, 0U, kvHeads_, kvSeq_, dim_)], dim_);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::Cast(inputFloatUb_, input16Ub_, AscendC::RoundMode::CAST_NONE, dim_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Abs(inputFloatUb_, inputFloatUb_, dim_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Max(statsMaxUb_, statsMaxUb_, inputFloatUb_, dim_);
                AscendC::PipeBarrier<PIPE_V>();
            }
            if constexpr (SMOOTH_K) {
                AscendC::Muls(statsSumUb_, statsSumUb_, 1.0F / static_cast<float>(kvSeq_), dim_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(input16Ub_, statsSumUb_, AscendC::RoundMode::CAST_RINT, dimPad_);
            }
            AscendC::Muls(statsMaxUb_, statsMaxUb_, 1.0F / vScaleMax_, dim_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Maxs(statsMaxUb_, statsMaxUb_, epsilon_, dimPad_);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            const uint64_t channelOffset = task * dimPad_;
            if constexpr (SMOOTH_K) {
                AscendC::DataCopy(kMeanTyped_[channelOffset], input16Ub_, dimPad_);
            }
            AscendC::DataCopy(vScale_[channelOffset], statsMaxUb_, dimPad_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

    __aicore__ inline bool QRowInGroup(uint32_t rowInBlock, uint32_t group) const
    {
        return QScaleGroup(rowInBlock, gran_) == group;
    }

    __aicore__ inline bool KRowInGroup(uint32_t rowInBlock, uint32_t group) const
    {
        return KScaleGroup(rowInBlock, gran_) == group;
    }

    __aicore__ inline void QuantizeQ()
    {
        if constexpr (!LAYOUT_NHD) {
            QuantizeQHnd();
            return;
        }
        const uint32_t vecId = AscendC::GetBlockIdx();
        const uint64_t taskCount = static_cast<uint64_t>(batch_) * qHeads_ * qBlocks_ * qGroups_;
        constexpr uint32_t SCALE_TASKS_PER_CACHE_LINE = 16U;
        for (uint64_t taskBase = static_cast<uint64_t>(vecId) * SCALE_TASKS_PER_CACHE_LINE;
             taskBase < taskCount;
             taskBase += static_cast<uint64_t>(vecCoreNum_) * SCALE_TASKS_PER_CACHE_LINE) {
          const uint64_t taskEnd = Min(taskCount, taskBase + SCALE_TASKS_PER_CACHE_LINE);
          for (uint64_t task = taskBase; task < taskEnd; ++task) {
            uint64_t linear = task;
            const uint32_t group = linear % qGroups_;
            linear /= qGroups_;
            const uint32_t block = linear % qBlocks_;
            linear /= qBlocks_;
            const uint32_t h = linear % qHeads_;
            const uint32_t b = linear / qHeads_;
            const uint32_t rowBegin = block * Q_QUANT_BLOCK;
            const uint32_t rowEnd = Min(qSeq_, rowBegin + Q_QUANT_BLOCK);
            float maxAbs = 0.0F;
            for (uint32_t s = rowBegin; s < rowEnd; ++s) {
                if (!QRowInGroup(s - rowBegin, group)) {
                    continue;
                }
                for (uint32_t d = 0U; d < dim_; ++d) {
                    const float value = DecodeInput16<InputT>(
                        q_.GetValue(SourceOffset<InputT>(layout_, b, h, s, d, qHeads_, qSeq_, dim_)));
                    maxAbs = Max(maxAbs, Abs(value));
                }
            }
            const float scale = maxAbs / INT8_MAX_VALUE + epsilon_;
            qScale_.SetValue(task, scale);
            for (uint32_t s = rowBegin; s < rowEnd; ++s) {
                if (!QRowInGroup(s - rowBegin, group)) {
                    continue;
                }
                for (uint32_t d = 0U; d < dimPad_; ++d) {
                    float value = 0.0F;
                    if (d < dim_) {
                        value = DecodeInput16<InputT>(
                            q_.GetValue(SourceOffset<InputT>(layout_, b, h, s, d, qHeads_, qSeq_, dim_)));
                    }
                    q8_.SetValue(BsNdOffset(b, h, s, d, qHeads_, qSeq_, dimPad_), RoundSaturateInt8(value / scale));
                }
            }
          }
        }
    }

    __aicore__ inline void QuantizeK()
    {
        if constexpr (!LAYOUT_NHD) {
            QuantizeKHnd();
            return;
        }
        const uint32_t vecId = AscendC::GetBlockIdx();
        const uint32_t kBlocks = (kvSeq_ + K_QUANT_BLOCK - 1U) / K_QUANT_BLOCK;
        const uint64_t taskCount = static_cast<uint64_t>(batch_) * kvHeads_ * kBlocks * kGroups_;
        constexpr uint32_t SCALE_TASKS_PER_CACHE_LINE = 16U;
        for (uint64_t taskBase = static_cast<uint64_t>(vecId) * SCALE_TASKS_PER_CACHE_LINE;
             taskBase < taskCount;
             taskBase += static_cast<uint64_t>(vecCoreNum_) * SCALE_TASKS_PER_CACHE_LINE) {
          const uint64_t taskEnd = Min(taskCount, taskBase + SCALE_TASKS_PER_CACHE_LINE);
          for (uint64_t task = taskBase; task < taskEnd; ++task) {
            uint64_t linear = task;
            const uint32_t group = linear % kGroups_;
            linear /= kGroups_;
            const uint32_t block = linear % kBlocks;
            linear /= kBlocks;
            const uint32_t h = linear % kvHeads_;
            const uint32_t b = linear / kvHeads_;
            const uint32_t rowBegin = block * K_QUANT_BLOCK;
            const uint32_t rowEnd = Min(kvSeq_, rowBegin + K_QUANT_BLOCK);
            const uint64_t meanBase = (static_cast<uint64_t>(b) * kvHeads_ + h) * dimPad_;
            float maxAbs = 0.0F;
            for (uint32_t s = rowBegin; s < rowEnd; ++s) {
                if (!KRowInGroup(s - rowBegin, group)) {
                    continue;
                }
                for (uint32_t d = 0U; d < dim_; ++d) {
                    float value = DecodeInput16<InputT>(
                        k_.GetValue(SourceOffset<InputT>(layout_, b, h, s, d, kvHeads_, kvSeq_, dim_)));
                    if constexpr (SMOOTH_K) {
                        value -= DecodeInput16<InputT>(kMean_.GetValue(meanBase + d));
                    }
                    maxAbs = Max(maxAbs, Abs(value));
                }
            }
            const float scale = maxAbs / INT8_MAX_VALUE + epsilon_;
            kScale_.SetValue(task, scale);
            for (uint32_t s = rowBegin; s < rowEnd; ++s) {
                if (!KRowInGroup(s - rowBegin, group)) {
                    continue;
                }
                for (uint32_t d = 0U; d < dimPad_; ++d) {
                    float value = 0.0F;
                    if (d < dim_) {
                        value = DecodeInput16<InputT>(
                            k_.GetValue(SourceOffset<InputT>(layout_, b, h, s, d, kvHeads_, kvSeq_, dim_)));
                        if constexpr (SMOOTH_K) {
                            value -= DecodeInput16<InputT>(kMean_.GetValue(meanBase + d));
                        }
                    }
                    k8_.SetValue(BsNdOffset(b, h, s, d, kvHeads_, kvSeq_, dimPad_), RoundSaturateInt8(value / scale));
                }
            }
          }
        }
    }

    __aicore__ inline void QuantizeQHnd()
    {
        if constexpr (PER_THREAD) {
            QuantizeQHndPerThreadBatched();
            return;
        }
        const uint32_t vecId = AscendC::GetBlockIdx();
        const uint64_t taskCount = static_cast<uint64_t>(batch_) * qHeads_ * qBlocks_ * qGroups_;
        constexpr uint32_t SCALE_TASKS_PER_CACHE_LINE = 16U;
        for (uint64_t taskBase = static_cast<uint64_t>(vecId) * SCALE_TASKS_PER_CACHE_LINE;
             taskBase < taskCount;
             taskBase += static_cast<uint64_t>(vecCoreNum_) * SCALE_TASKS_PER_CACHE_LINE) {
          const uint64_t taskEnd = Min(taskCount, taskBase + SCALE_TASKS_PER_CACHE_LINE);
          for (uint64_t task = taskBase; task < taskEnd; ++task) {
            uint64_t linear = task;
            const uint32_t group = linear % qGroups_;
            linear /= qGroups_;
            const uint32_t block = linear % qBlocks_;
            linear /= qBlocks_;
            const uint32_t h = linear % qHeads_;
            const uint32_t b = linear / qHeads_;
            const uint32_t rowBegin = block * Q_QUANT_BLOCK;
            const uint32_t rowEnd = Min(qSeq_, rowBegin + Q_QUANT_BLOCK);
            uint32_t rows = 0U;
            for (uint32_t s = rowBegin; s < rowEnd; ++s) {
                if (QRowInGroup(s - rowBegin, group)) {
                    AscendC::DataCopy(input16Ub_[static_cast<uint64_t>(rows) * dimPad_],
                        qTyped_[SourceOffset<InputT>(Layout::HND, b, h, s, 0U, qHeads_, qSeq_, dim_)], dim_);
                    ++rows;
                }
            }
            const uint32_t elements = rows * dimPad_;
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Cast(inputFloatUb_, input16Ub_, AscendC::RoundMode::CAST_NONE, elements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Abs(quantAbsUb_, inputFloatUb_, elements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceMax(statsMaxUb_, quantAbsUb_, quantWorkUb_, elements);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            const float scale = statsMaxUb_.GetValue(0U) / INT8_MAX_VALUE + epsilon_;
            qScale_.SetValue(task, scale);
            AscendC::AscendQuant(quantInt8Ub_, inputFloatUb_, 1.0F / scale, 0.0F, elements);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            rows = 0U;
            for (uint32_t s = rowBegin; s < rowEnd; ++s) {
                if (QRowInGroup(s - rowBegin, group)) {
                    AscendC::DataCopy(q8_[BsNdOffset(b, h, s, 0U, qHeads_, qSeq_, dimPad_)],
                        quantInt8Ub_[static_cast<uint64_t>(rows) * dimPad_], dimPad_);
                    ++rows;
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
          }
        }
    }

    __simd_vf__ static inline void QuantizeInt8GroupsVf(__ubuf__ float *src, __ubuf__ int8_t *dst,
        __ubuf__ float *scaleOut, uint16_t groups, float epsilon)
    {
        using namespace AscendC::MicroAPI;
        // Atlas A3 does not provide a directly packable float -> int8 conversion. Follow the
        // native attention path: float -> half in ZERO/ONE layouts, merge, half -> int8, then
        // retain the low byte of every half lane.
        constexpr static CastTrait CAST_ZERO = {
            RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        constexpr static CastTrait CAST_ONE = {
            RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

        MaskReg fp32Full = CreateMask<float, MaskPattern::ALL>();
        MaskReg halfFull = CreateMask<half, MaskPattern::ALL>();
        uint32_t packedLength = 128U;
        MaskReg int8Packed = UpdateMask<int8_t>(packedLength);
        RegTensor<uint32_t> laneZero;
        Duplicate(laneZero, 0U);

        RegTensor<float> value0;
        RegTensor<float> value1;
        RegTensor<float> absValue;
        RegTensor<float> maxValue;
        RegTensor<float> reducedMax;
        RegTensor<float> scale;
        RegTensor<half> halfEven;
        RegTensor<half> halfOdd;
        RegTensor<half> halfMerged;
        RegTensor<int8_t> int8Merged;
        RegTensor<int8_t> int8PackedReg;
        UnalignRegForStore scaleStore;

        for (uint16_t group = 0U; group < groups; ++group) {
            __ubuf__ float *groupSrc = src + static_cast<uint32_t>(group) * 512U;
            LoadAlign(value0, groupSrc);
            Abs(maxValue, value0, fp32Full);
            for (uint16_t part = 1U; part < 8U; ++part) {
                LoadAlign(value0, groupSrc + static_cast<uint32_t>(part) * 64U);
                Abs(absValue, value0, fp32Full);
                Max(maxValue, maxValue, absValue, fp32Full);
            }
            Reduce<ReduceType::MAX, float, float, MaskMergeMode::ZEROING>(
                reducedMax, maxValue, fp32Full);
            Muls(reducedMax, reducedMax, 1.0F / INT8_MAX_VALUE, fp32Full);
            Adds(reducedMax, reducedMax, epsilon, fp32Full);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(scaleOut, reducedMax, scaleStore, 1U);
            Gather(scale, reducedMax, laneZero);

            for (uint16_t block = 0U; block < 4U; ++block) {
                __ubuf__ float *blockSrc = groupSrc + static_cast<uint32_t>(block) * 128U;
                LoadAlign<float, LoadDist::DIST_DINTLV_B32>(value0, value1, blockSrc);
                Div(value0, value0, scale, fp32Full);
                Div(value1, value1, scale, fp32Full);
                Cast<half, float, CAST_ZERO>(halfEven, value0, fp32Full);
                Cast<half, float, CAST_ONE>(halfOdd, value1, fp32Full);
                Or(reinterpret_cast<RegTensor<uint16_t> &>(halfMerged),
                    reinterpret_cast<RegTensor<uint16_t> &>(halfEven),
                    reinterpret_cast<RegTensor<uint16_t> &>(halfOdd), halfFull);
                Cast<int8_t, half, CAST_ZERO>(int8Merged, halfMerged, halfFull);
                Pack<uint8_t, uint16_t, HighLowPart::LOWEST>(
                    reinterpret_cast<RegTensor<uint8_t> &>(int8PackedReg),
                    reinterpret_cast<RegTensor<uint16_t> &>(int8Merged));
                StoreAlign(dst + static_cast<uint32_t>(group) * 512U + static_cast<uint32_t>(block) * 128U,
                    int8PackedReg, int8Packed);
            }
        }
        StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(scaleOut, scaleStore, 0U);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    __simd_vf__ static inline void QuantizeInt8QWarpVf(
        __ubuf__ float *src, __ubuf__ int8_t *dst, __ubuf__ float *scaleOut, float epsilon)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait CAST_ZERO = {
            RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        constexpr static CastTrait CAST_ONE = {
            RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

        MaskReg fp32Full = CreateMask<float, MaskPattern::ALL>();
        MaskReg halfFull = CreateMask<half, MaskPattern::ALL>();
        uint32_t packedLength = 128U;
        MaskReg int8Packed = UpdateMask<int8_t>(packedLength);
        RegTensor<uint32_t> laneZero;
        Duplicate(laneZero, 0U);

        RegTensor<float> valueEven;
        RegTensor<float> valueOdd;
        RegTensor<float> absEven;
        RegTensor<float> absOdd;
        RegTensor<float> maxValue;
        RegTensor<float> reducedMax;
        RegTensor<float> scale;
        RegTensor<half> halfEven;
        RegTensor<half> halfOdd;
        RegTensor<half> halfMerged;
        RegTensor<int8_t> int8Merged;
        RegTensor<int8_t> int8PackedReg;
        UnalignRegForStore scaleStore;

        // PER_THREAD has eight scale groups in each 32-row warp. A group owns rows
        // lane, lane+8, lane+16 and lane+24 while the source remains in HND row order.
        for (uint16_t group = 0U; group < 8U; ++group) {
            const uint16_t rowBase = group;
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
                valueEven, valueOdd, src + static_cast<uint32_t>(rowBase) * 128U);
            Abs(absEven, valueEven, fp32Full);
            Abs(absOdd, valueOdd, fp32Full);
            Max(maxValue, absEven, absOdd, fp32Full);
            for (uint16_t rowInGroup = 1U; rowInGroup < 4U; ++rowInGroup) {
                const uint16_t row = static_cast<uint16_t>(rowBase + rowInGroup * 8U);
                LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
                    valueEven, valueOdd, src + static_cast<uint32_t>(row) * 128U);
                Abs(absEven, valueEven, fp32Full);
                Abs(absOdd, valueOdd, fp32Full);
                Max(maxValue, maxValue, absEven, fp32Full);
                Max(maxValue, maxValue, absOdd, fp32Full);
            }
            Reduce<ReduceType::MAX, float, float, MaskMergeMode::ZEROING>(
                reducedMax, maxValue, fp32Full);
            Muls(reducedMax, reducedMax, 1.0F / INT8_MAX_VALUE, fp32Full);
            Adds(reducedMax, reducedMax, epsilon, fp32Full);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(scaleOut, reducedMax, scaleStore, 1U);
            Gather(scale, reducedMax, laneZero);
            for (uint16_t rowInGroup = 0U; rowInGroup < 4U; ++rowInGroup) {
                const uint16_t row = static_cast<uint16_t>(rowBase + rowInGroup * 8U);
                const uint32_t rowOffset = static_cast<uint32_t>(row) * 128U;
                LoadAlign<float, LoadDist::DIST_DINTLV_B32>(valueEven, valueOdd, src + rowOffset);
                Div(valueEven, valueEven, scale, fp32Full);
                Div(valueOdd, valueOdd, scale, fp32Full);
                Cast<half, float, CAST_ZERO>(halfEven, valueEven, fp32Full);
                Cast<half, float, CAST_ONE>(halfOdd, valueOdd, fp32Full);
                Or(reinterpret_cast<RegTensor<uint16_t> &>(halfMerged),
                    reinterpret_cast<RegTensor<uint16_t> &>(halfEven),
                    reinterpret_cast<RegTensor<uint16_t> &>(halfOdd), halfFull);
                Cast<int8_t, half, CAST_ZERO>(int8Merged, halfMerged, halfFull);
                Pack<uint8_t, uint16_t, HighLowPart::LOWEST>(
                    reinterpret_cast<RegTensor<uint8_t> &>(int8PackedReg),
                    reinterpret_cast<RegTensor<uint16_t> &>(int8Merged));
                StoreAlign(dst + rowOffset, int8PackedReg, int8Packed);
            }
        }
        StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(scaleOut, scaleStore, 0U);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    __aicore__ inline void QuantizeQHndPerThreadContiguous(
        uint32_t workerId = UINT32_MAX, uint32_t workerCount = 0U)
    {
        constexpr uint32_t ROWS = Q_QUANT_BLOCK;
        constexpr uint32_t ELEMENTS = ROWS * 128U;
        const uint32_t vecId = workerId == UINT32_MAX ? AscendC::GetBlockIdx() : workerId;
        const uint32_t workers = workerCount == 0U ? vecCoreNum_ : workerCount;
        const uint64_t taskCount = static_cast<uint64_t>(batch_) * qHeads_ * qBlocks_;
        for (uint64_t task = vecId; task < taskCount; task += workers) {
            uint64_t linear = task;
            const uint32_t block = linear % qBlocks_;
            linear /= qBlocks_;
            const uint32_t h = linear % qHeads_;
            const uint32_t b = linear / qHeads_;
            const uint32_t rowBegin = block * ROWS;
            AscendC::DataCopy(input16Ub_,
                qTyped_[SourceOffset<InputT>(Layout::HND, b, h, rowBegin, 0U, qHeads_, qSeq_, dim_)], ELEMENTS);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Cast(inputFloatUb_, input16Ub_, AscendC::RoundMode::CAST_NONE, ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            auto inputFloatAddr = reinterpret_cast<__ubuf__ float *>(inputFloatUb_.GetPhyAddr());
            auto quantInt8Addr = reinterpret_cast<__ubuf__ int8_t *>(quantInt8Ub_.GetPhyAddr());
            auto scaleAddr = reinterpret_cast<__ubuf__ float *>(statsSumUb_.GetPhyAddr());
            for (uint32_t warp = 0U; warp < 4U; ++warp) {
                QuantizeInt8QWarpVf(inputFloatAddr + warp * 4096U,
                    quantInt8Addr + warp * 4096U, scaleAddr + warp * 8U, epsilon_);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(qScale_[task * 32U], statsSumUb_, 32U);
            const uint64_t q8Offset = BsNdOffset(b, h, rowBegin, 0U, qHeads_, qSeq_, dimPad_);
            const uint16_t dstGap = static_cast<uint16_t>((qHeads_ - 1U) * 4U);
            AscendC::DataCopy(q8_[q8Offset], quantInt8Ub_,
                {static_cast<uint16_t>(ROWS), 4U, 0U, dstGap});
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

    __aicore__ inline void QuantizeInt8Groups(uint32_t groups)
    {
        auto srcAddr = reinterpret_cast<__ubuf__ float *>(inputFloatUb_.GetPhyAddr());
        auto dstAddr = reinterpret_cast<__ubuf__ int8_t *>(quantInt8Ub_.GetPhyAddr());
        auto scaleAddr = reinterpret_cast<__ubuf__ float *>(statsSumUb_.GetPhyAddr());
        QuantizeInt8GroupsVf(srcAddr, dstAddr, scaleAddr, static_cast<uint16_t>(groups), epsilon_);
    }

    __aicore__ inline void QuantizeQHndPerThreadBatched()
    {
        if (dim_ == 128U && dimPad_ == 128U && qGroups_ == 32U && (qSeq_ % Q_QUANT_BLOCK) == 0U) {
            QuantizeQHndPerThreadContiguous();
            return;
        }
        constexpr uint32_t GROUPS_PER_BATCH = 16U;
        constexpr uint32_t ROWS_PER_GROUP = Q_QUANT_BLOCK / 32U;
        const uint32_t vecId = AscendC::GetBlockIdx();
        const uint64_t taskCount = static_cast<uint64_t>(batch_) * qHeads_ * qBlocks_ * qGroups_;
        constexpr uint32_t SCALE_TASKS_PER_CACHE_LINE = 16U;
        for (uint64_t taskBase = static_cast<uint64_t>(vecId) * SCALE_TASKS_PER_CACHE_LINE;
             taskBase < taskCount;
             taskBase += static_cast<uint64_t>(vecCoreNum_) * SCALE_TASKS_PER_CACHE_LINE) {
            const uint64_t taskEnd = Min(taskCount, taskBase + SCALE_TASKS_PER_CACHE_LINE);
            for (uint64_t batchBase = taskBase; batchBase < taskEnd; batchBase += GROUPS_PER_BATCH) {
                const uint32_t batchTasks = static_cast<uint32_t>(Min(
                    static_cast<uint64_t>(GROUPS_PER_BATCH), taskEnd - batchBase));
                uint32_t rowCounts[GROUPS_PER_BATCH] = {0U, 0U, 0U, 0U};
                for (uint32_t taskInBatch = 0U; taskInBatch < batchTasks; ++taskInBatch) {
                    const uint64_t task = batchBase + taskInBatch;
                    uint64_t linear = task;
                    const uint32_t group = linear % qGroups_;
                    linear /= qGroups_;
                    const uint32_t block = linear % qBlocks_;
                    linear /= qBlocks_;
                    const uint32_t h = linear % qHeads_;
                    const uint32_t b = linear / qHeads_;
                    const uint32_t rowBegin = block * Q_QUANT_BLOCK;
                    const uint32_t rowEnd = Min(qSeq_, rowBegin + Q_QUANT_BLOCK);
                    const uint32_t slotOffset = taskInBatch * ROWS_PER_GROUP * dimPad_;
                    uint32_t rows = 0U;
                    for (uint32_t s = rowBegin; s < rowEnd; ++s) {
                        if (QRowInGroup(s - rowBegin, group)) {
                            AscendC::DataCopy(input16Ub_[slotOffset + static_cast<uint64_t>(rows) * dimPad_],
                                qTyped_[SourceOffset<InputT>(
                                    Layout::HND, b, h, s, 0U, qHeads_, qSeq_, dim_)], dim_);
                            ++rows;
                        }
                    }
                    rowCounts[taskInBatch] = rows;
                }
                const uint32_t batchElements = batchTasks * ROWS_PER_GROUP * dimPad_;
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::Cast(inputFloatUb_, input16Ub_, AscendC::RoundMode::CAST_NONE, batchElements);
                AscendC::PipeBarrier<PIPE_V>();
                bool packedFast = dimPad_ == 128U && batchTasks == GROUPS_PER_BATCH;
                for (uint32_t taskInBatch = 0U; taskInBatch < batchTasks; ++taskInBatch) {
                    packedFast = packedFast && rowCounts[taskInBatch] == ROWS_PER_GROUP;
                }
                if (packedFast) {
                    QuantizeInt8Groups(batchTasks);
                    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                    for (uint32_t taskInBatch = 0U; taskInBatch < batchTasks; ++taskInBatch) {
                        qScale_.SetValue(batchBase + taskInBatch, statsSumUb_.GetValue(taskInBatch));
                    }
                } else {
                    AscendC::Abs(quantAbsUb_, inputFloatUb_, batchElements);
                    AscendC::PipeBarrier<PIPE_V>();
                    for (uint32_t taskInBatch = 0U; taskInBatch < batchTasks; ++taskInBatch) {
                        const uint32_t slotOffset = taskInBatch * ROWS_PER_GROUP * dimPad_;
                        const uint32_t elements = rowCounts[taskInBatch] * dimPad_;
                        if (elements == 0U) {
                            statsSumUb_.SetValue(taskInBatch * 8U, 0.0F);
                            continue;
                        }
                        AscendC::ReduceMax(
                            statsSumUb_[taskInBatch * 8U], quantAbsUb_[slotOffset], quantWorkUb_, elements);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                    for (uint32_t taskInBatch = 0U; taskInBatch < batchTasks; ++taskInBatch) {
                        const uint64_t task = batchBase + taskInBatch;
                        const uint32_t slotOffset = taskInBatch * ROWS_PER_GROUP * dimPad_;
                        const uint32_t elements = rowCounts[taskInBatch] * dimPad_;
                        const float scale = statsSumUb_.GetValue(taskInBatch * 8U) / INT8_MAX_VALUE + epsilon_;
                        qScale_.SetValue(task, scale);
                        if (elements != 0U) {
                            AscendC::AscendQuant(quantInt8Ub_[slotOffset], inputFloatUb_[slotOffset],
                                1.0F / scale, 0.0F, elements);
                        }
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                for (uint32_t taskInBatch = 0U; taskInBatch < batchTasks; ++taskInBatch) {
                    uint64_t linear = batchBase + taskInBatch;
                    const uint32_t group = linear % qGroups_;
                    linear /= qGroups_;
                    const uint32_t block = linear % qBlocks_;
                    linear /= qBlocks_;
                    const uint32_t h = linear % qHeads_;
                    const uint32_t b = linear / qHeads_;
                    const uint32_t rowBegin = block * Q_QUANT_BLOCK;
                    const uint32_t rowEnd = Min(qSeq_, rowBegin + Q_QUANT_BLOCK);
                    const uint32_t slotOffset = taskInBatch * ROWS_PER_GROUP * dimPad_;
                    uint32_t rows = 0U;
                    for (uint32_t s = rowBegin; s < rowEnd; ++s) {
                        if (QRowInGroup(s - rowBegin, group)) {
                            AscendC::DataCopy(q8_[BsNdOffset(b, h, s, 0U, qHeads_, qSeq_, dimPad_)],
                                quantInt8Ub_[slotOffset + static_cast<uint64_t>(rows) * dimPad_], dimPad_);
                            ++rows;
                        }
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
    }

    __aicore__ inline void QuantizeKHnd()
    {
        if constexpr (PER_THREAD) {
            QuantizeKHndPerThreadBatched();
            return;
        }
        const uint32_t vecId = AscendC::GetBlockIdx();
        const uint32_t kBlocks = (kvSeq_ + K_QUANT_BLOCK - 1U) / K_QUANT_BLOCK;
        const uint64_t taskCount = static_cast<uint64_t>(batch_) * kvHeads_ * kBlocks * kGroups_;
        constexpr uint32_t SCALE_TASKS_PER_CACHE_LINE = 16U;
        for (uint64_t taskBase = static_cast<uint64_t>(vecId) * SCALE_TASKS_PER_CACHE_LINE;
             taskBase < taskCount;
             taskBase += static_cast<uint64_t>(vecCoreNum_) * SCALE_TASKS_PER_CACHE_LINE) {
          const uint64_t taskEnd = Min(taskCount, taskBase + SCALE_TASKS_PER_CACHE_LINE);
          for (uint64_t task = taskBase; task < taskEnd; ++task) {
            uint64_t linear = task;
            const uint32_t group = linear % kGroups_;
            linear /= kGroups_;
            const uint32_t block = linear % kBlocks;
            linear /= kBlocks;
            const uint32_t h = linear % kvHeads_;
            const uint32_t b = linear / kvHeads_;
            const uint32_t rowBegin = block * K_QUANT_BLOCK;
            const uint32_t rowEnd = Min(kvSeq_, rowBegin + K_QUANT_BLOCK);
            const uint64_t meanBase = (static_cast<uint64_t>(b) * kvHeads_ + h) * dimPad_;
            if constexpr (SMOOTH_K) {
                AscendC::DataCopy(mean16Ub_, kMeanTyped_[meanBase], dimPad_);
            }
            uint32_t rows = 0U;
            for (uint32_t s = rowBegin; s < rowEnd; ++s) {
                if (KRowInGroup(s - rowBegin, group)) {
                    AscendC::DataCopy(input16Ub_[static_cast<uint64_t>(rows) * dimPad_],
                        kTyped_[SourceOffset<InputT>(Layout::HND, b, h, s, 0U, kvHeads_, kvSeq_, dim_)], dim_);
                    ++rows;
                }
            }
            const uint32_t elements = rows * dimPad_;
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Cast(inputFloatUb_, input16Ub_, AscendC::RoundMode::CAST_NONE, elements);
            if constexpr (SMOOTH_K) {
                AscendC::Cast(meanFloatUb_, mean16Ub_, AscendC::RoundMode::CAST_NONE, dimPad_);
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t r = 0U; r < rows; ++r) {
                    AscendC::Sub(inputFloatUb_[static_cast<uint64_t>(r) * dimPad_],
                        inputFloatUb_[static_cast<uint64_t>(r) * dimPad_], meanFloatUb_, dimPad_);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Abs(quantAbsUb_, inputFloatUb_, elements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceMax(statsMaxUb_, quantAbsUb_, quantWorkUb_, elements);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            const float scale = statsMaxUb_.GetValue(0U) / INT8_MAX_VALUE + epsilon_;
            kScale_.SetValue(task, scale);
            AscendC::AscendQuant(quantInt8Ub_, inputFloatUb_, 1.0F / scale, 0.0F, elements);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            rows = 0U;
            for (uint32_t s = rowBegin; s < rowEnd; ++s) {
                if (KRowInGroup(s - rowBegin, group)) {
                    AscendC::DataCopy(k8_[BsNdOffset(b, h, s, 0U, kvHeads_, kvSeq_, dimPad_)],
                        quantInt8Ub_[static_cast<uint64_t>(rows) * dimPad_], dimPad_);
                    ++rows;
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
          }
        }
    }

    template <bool DO_QUANT>
    __simd_vf__ static inline void QuantizeInt8KBlockVf(__ubuf__ float *src, __ubuf__ float *mean,
        __ubuf__ int8_t *dst, __ubuf__ float *scaleOut, float epsilon)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait CAST_ZERO = {
            RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        constexpr static CastTrait CAST_ONE = {
            RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        MaskReg fp32Full = CreateMask<float, MaskPattern::ALL>();
        MaskReg halfFull = CreateMask<half, MaskPattern::ALL>();
        uint32_t packedLength = 128U;
        MaskReg int8Packed = UpdateMask<int8_t>(packedLength);
        RegTensor<uint32_t> laneZero;
        Duplicate(laneZero, 0U);

        RegTensor<float> meanEven;
        RegTensor<float> meanOdd;
        RegTensor<float> valueEven;
        RegTensor<float> valueOdd;
        RegTensor<float> absEven;
        RegTensor<float> absOdd;
        RegTensor<float> maxValue;
        RegTensor<float> reducedMax;
        RegTensor<float> scale;
        RegTensor<half> halfEven;
        RegTensor<half> halfOdd;
        RegTensor<half> halfMerged;
        RegTensor<int8_t> int8Merged;
        RegTensor<int8_t> int8PackedReg;
        UnalignRegForStore scaleStore;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(meanEven, meanOdd, mean);

        for (uint16_t group = 0U; group < 4U; ++group) {
            uint16_t row = static_cast<uint16_t>(group * 2U);
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(valueEven, valueOdd,
                src + static_cast<uint32_t>(row) * 128U);
            Sub(valueEven, valueEven, meanEven, fp32Full);
            Sub(valueOdd, valueOdd, meanOdd, fp32Full);
            Abs(absEven, valueEven, fp32Full);
            Abs(absOdd, valueOdd, fp32Full);
            Max(maxValue, absEven, absOdd, fp32Full);
            for (uint16_t rowInGroup = 1U; rowInGroup < 16U; ++rowInGroup) {
                row = static_cast<uint16_t>((rowInGroup >> 1U) * 8U + group * 2U + (rowInGroup & 1U));
                LoadAlign<float, LoadDist::DIST_DINTLV_B32>(valueEven, valueOdd,
                    src + static_cast<uint32_t>(row) * 128U);
                Sub(valueEven, valueEven, meanEven, fp32Full);
                Sub(valueOdd, valueOdd, meanOdd, fp32Full);
                Abs(absEven, valueEven, fp32Full);
                Abs(absOdd, valueOdd, fp32Full);
                Max(maxValue, maxValue, absEven, fp32Full);
                Max(maxValue, maxValue, absOdd, fp32Full);
            }
            Reduce<ReduceType::MAX, float, float, MaskMergeMode::ZEROING>(
                reducedMax, maxValue, fp32Full);
            Muls(reducedMax, reducedMax, 1.0F / INT8_MAX_VALUE, fp32Full);
            Adds(reducedMax, reducedMax, epsilon, fp32Full);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(scaleOut, reducedMax, scaleStore, 1U);
            if constexpr (DO_QUANT) {
                Gather(scale, reducedMax, laneZero);
                for (uint16_t rowInGroup = 0U; rowInGroup < 16U; ++rowInGroup) {
                    row = static_cast<uint16_t>((rowInGroup >> 1U) * 8U + group * 2U + (rowInGroup & 1U));
                    const uint32_t rowOffset = static_cast<uint32_t>(row) * 128U;
                    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(valueEven, valueOdd, src + rowOffset);
                    Sub(valueEven, valueEven, meanEven, fp32Full);
                    Sub(valueOdd, valueOdd, meanOdd, fp32Full);
                    Div(valueEven, valueEven, scale, fp32Full);
                    Div(valueOdd, valueOdd, scale, fp32Full);
                    Cast<half, float, CAST_ZERO>(halfEven, valueEven, fp32Full);
                    Cast<half, float, CAST_ONE>(halfOdd, valueOdd, fp32Full);
                    Or(reinterpret_cast<RegTensor<uint16_t> &>(halfMerged),
                        reinterpret_cast<RegTensor<uint16_t> &>(halfEven),
                        reinterpret_cast<RegTensor<uint16_t> &>(halfOdd), halfFull);
                    Cast<int8_t, half, CAST_ZERO>(int8Merged, halfMerged, halfFull);
                    Pack<uint8_t, uint16_t, HighLowPart::LOWEST>(
                        reinterpret_cast<RegTensor<uint8_t> &>(int8PackedReg),
                        reinterpret_cast<RegTensor<uint16_t> &>(int8Merged));
                    StoreAlign(dst + rowOffset, int8PackedReg, int8Packed);
                }
            }
        }
        StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(scaleOut, scaleStore, 0U);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    __aicore__ inline void QuantizeKHndPerThreadContiguous()
    {
        constexpr uint32_t ROWS = K_QUANT_BLOCK;
        constexpr uint32_t ELEMENTS = ROWS * 128U;
        const uint32_t vecId = AscendC::GetBlockIdx();
        const uint32_t kBlocks = kvSeq_ / K_QUANT_BLOCK;
        const uint64_t taskCount = static_cast<uint64_t>(batch_) * kvHeads_ * kBlocks;
        constexpr uint32_t TASKS_PER_CACHE_LINE = 4U;
        for (uint64_t taskBase = static_cast<uint64_t>(vecId) * TASKS_PER_CACHE_LINE;
             taskBase < taskCount;
             taskBase += static_cast<uint64_t>(vecCoreNum_) * TASKS_PER_CACHE_LINE) {
          const uint64_t taskEnd = Min(taskCount, taskBase + TASKS_PER_CACHE_LINE);
          for (uint64_t task = taskBase; task < taskEnd; ++task) {
            uint64_t linear = task;
            const uint32_t block = linear % kBlocks;
            linear /= kBlocks;
            const uint32_t h = linear % kvHeads_;
            const uint32_t b = linear / kvHeads_;
            const uint32_t rowBegin = block * ROWS;
            const uint64_t meanBase = (static_cast<uint64_t>(b) * kvHeads_ + h) * dimPad_;
            AscendC::DataCopy(input16Ub_,
                kTyped_[SourceOffset<InputT>(Layout::HND, b, h, rowBegin, 0U, kvHeads_, kvSeq_, dim_)], ELEMENTS);
            AscendC::DataCopy(mean16Ub_, kMeanTyped_[meanBase], dimPad_);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Cast(inputFloatUb_, input16Ub_, AscendC::RoundMode::CAST_NONE, ELEMENTS);
            AscendC::Cast(meanFloatUb_, mean16Ub_, AscendC::RoundMode::CAST_NONE, dimPad_);
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (K_QUANT_PROBE == 1U) {
                continue;
            }
            QuantizeInt8KBlockVf<K_QUANT_PROBE != 2U>(
                reinterpret_cast<__ubuf__ float *>(inputFloatUb_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(meanFloatUb_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ int8_t *>(quantInt8Ub_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(statsSumUb_.GetPhyAddr()), epsilon_);
            if constexpr (K_QUANT_PROBE != 0U) {
                continue;
            }
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            for (uint32_t group = 0U; group < 4U; ++group) {
                kScale_.SetValue(task * 4U + group, statsSumUb_.GetValue(group));
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            const uint64_t k8Offset = BsNdOffset(b, h, rowBegin, 0U, kvHeads_, kvSeq_, dimPad_);
            const uint16_t dstGap = static_cast<uint16_t>((kvHeads_ - 1U) * 4U);
            AscendC::DataCopy(k8_[k8Offset], quantInt8Ub_,
                {static_cast<uint16_t>(ROWS), 4U, 0U, dstGap});
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
          }
        }
    }

    __aicore__ inline void QuantizeKHndPerThreadPairBatched()
    {
        constexpr uint32_t BLOCKS_PER_TASK = 2U;
        constexpr uint32_t ROWS_PER_TASK = BLOCKS_PER_TASK * K_QUANT_BLOCK;
        constexpr uint32_t ELEMENTS_PER_BLOCK = K_QUANT_BLOCK * 128U;
        constexpr uint32_t ELEMENTS_PER_TASK = ROWS_PER_TASK * 128U;
        static_assert(ELEMENTS_PER_TASK * sizeof(InputT) <= 32768U,
            "K128 input batch exceeds the preprocessing input16 buffer");
        static_assert(ELEMENTS_PER_TASK * sizeof(float) <= 65536U,
            "K128 input batch exceeds the preprocessing float buffer");
        static_assert(ELEMENTS_PER_TASK * sizeof(int8_t) <= 16384U,
            "K128 input batch exceeds the preprocessing int8 buffer");

        const uint32_t vecId = AscendC::GetBlockIdx();
        const uint32_t kBlocks = kvSeq_ / K_QUANT_BLOCK;
        const uint32_t blockPairs = kBlocks / BLOCKS_PER_TASK;
        const uint64_t taskCount = static_cast<uint64_t>(batch_) * kvHeads_ * blockPairs;
        // One pair publishes eight floats. Assign two consecutive pairs to one AIV so each 64-byte
        // kScale cache line has a single writer, matching the original four-K64 ownership rule.
        constexpr uint32_t TASKS_PER_CACHE_LINE = 2U;
        for (uint64_t taskBase = static_cast<uint64_t>(vecId) * TASKS_PER_CACHE_LINE;
             taskBase < taskCount;
             taskBase += static_cast<uint64_t>(vecCoreNum_) * TASKS_PER_CACHE_LINE) {
            const uint64_t taskEnd = Min(taskCount, taskBase + TASKS_PER_CACHE_LINE);
            for (uint64_t task = taskBase; task < taskEnd; ++task) {
                uint64_t linear = task;
                const uint32_t pair = linear % blockPairs;
                linear /= blockPairs;
                const uint32_t h = linear % kvHeads_;
                const uint32_t b = linear / kvHeads_;
                const uint32_t firstBlock = pair * BLOCKS_PER_TASK;
                const uint32_t rowBegin = firstBlock * K_QUANT_BLOCK;
                const uint64_t meanBase = (static_cast<uint64_t>(b) * kvHeads_ + h) * dimPad_;

                AscendC::DataCopy(input16Ub_,
                    kTyped_[SourceOffset<InputT>(
                        Layout::HND, b, h, rowBegin, 0U, kvHeads_, kvSeq_, dim_)], ELEMENTS_PER_TASK);
                AscendC::DataCopy(mean16Ub_, kMeanTyped_[meanBase], dimPad_);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::Cast(inputFloatUb_, input16Ub_, AscendC::RoundMode::CAST_NONE, ELEMENTS_PER_TASK);
                AscendC::Cast(meanFloatUb_, mean16Ub_, AscendC::RoundMode::CAST_NONE, dimPad_);
                AscendC::PipeBarrier<PIPE_V>();

                auto inputAddr = reinterpret_cast<__ubuf__ float *>(inputFloatUb_.GetPhyAddr());
                auto meanAddr = reinterpret_cast<__ubuf__ float *>(meanFloatUb_.GetPhyAddr());
                auto outputAddr = reinterpret_cast<__ubuf__ int8_t *>(quantInt8Ub_.GetPhyAddr());
                auto scaleAddr = reinterpret_cast<__ubuf__ float *>(statsSumUb_.GetPhyAddr());
                QuantizeInt8KBlockVf<true>(inputAddr, meanAddr, outputAddr, scaleAddr, epsilon_);
                QuantizeInt8KBlockVf<true>(inputAddr + ELEMENTS_PER_BLOCK, meanAddr,
                    outputAddr + ELEMENTS_PER_BLOCK, scaleAddr + 4U, epsilon_);

                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                const uint64_t firstScale =
                    ((static_cast<uint64_t>(b) * kvHeads_ + h) * kBlocks + firstBlock) * 4U;
                for (uint32_t group = 0U; group < 8U; ++group) {
                    kScale_.SetValue(firstScale + group, statsSumUb_.GetValue(group));
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                const uint64_t k8Offset = BsNdOffset(b, h, rowBegin, 0U, kvHeads_, kvSeq_, dimPad_);
                const uint16_t dstGap = static_cast<uint16_t>((kvHeads_ - 1U) * 4U);
                AscendC::DataCopy(k8_[k8Offset], quantInt8Ub_,
                    {static_cast<uint16_t>(ROWS_PER_TASK), 4U, 0U, dstGap});
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
    }

    __aicore__ inline void QuantizeKHndPerThreadBatched()
    {
        if constexpr (SMOOTH_K) {
            if constexpr (K_QUANT_PROBE == 0U) {
                if (dim_ == 128U && dimPad_ == 128U && kGroups_ == 4U &&
                    (kvSeq_ % (2U * K_QUANT_BLOCK)) == 0U) {
                    QuantizeKHndPerThreadPairBatched();
                    return;
                }
            }
            if (dim_ == 128U && dimPad_ == 128U && kGroups_ == 4U &&
                (kvSeq_ % K_QUANT_BLOCK) == 0U) {
                QuantizeKHndPerThreadContiguous();
                return;
            }
        }
        constexpr uint32_t GROUPS_PER_BATCH = 4U;
        constexpr uint32_t ROWS_PER_GROUP = K_QUANT_BLOCK / GROUPS_PER_BATCH;
        const uint32_t vecId = AscendC::GetBlockIdx();
        const uint32_t kBlocks = (kvSeq_ + K_QUANT_BLOCK - 1U) / K_QUANT_BLOCK;
        const uint64_t taskCount = static_cast<uint64_t>(batch_) * kvHeads_ * kBlocks * kGroups_;
        constexpr uint32_t SCALE_TASKS_PER_CACHE_LINE = 16U;
        for (uint64_t taskBase = static_cast<uint64_t>(vecId) * SCALE_TASKS_PER_CACHE_LINE;
             taskBase < taskCount;
             taskBase += static_cast<uint64_t>(vecCoreNum_) * SCALE_TASKS_PER_CACHE_LINE) {
            const uint64_t taskEnd = Min(taskCount, taskBase + SCALE_TASKS_PER_CACHE_LINE);
            for (uint64_t batchBase = taskBase; batchBase < taskEnd; batchBase += GROUPS_PER_BATCH) {
                const uint32_t batchTasks = static_cast<uint32_t>(Min(
                    static_cast<uint64_t>(GROUPS_PER_BATCH), taskEnd - batchBase));
                uint32_t rowCounts[GROUPS_PER_BATCH] = {0U, 0U, 0U, 0U};
                uint64_t firstLinear = batchBase / kGroups_;
                firstLinear /= kBlocks;
                const uint32_t firstHead = firstLinear % kvHeads_;
                const uint32_t firstBatch = firstLinear / kvHeads_;
                const uint64_t meanBase =
                    (static_cast<uint64_t>(firstBatch) * kvHeads_ + firstHead) * dimPad_;
                if constexpr (SMOOTH_K) {
                    AscendC::DataCopy(mean16Ub_, kMeanTyped_[meanBase], dimPad_);
                }
                for (uint32_t taskInBatch = 0U; taskInBatch < batchTasks; ++taskInBatch) {
                    uint64_t linear = batchBase + taskInBatch;
                    const uint32_t group = linear % kGroups_;
                    linear /= kGroups_;
                    const uint32_t block = linear % kBlocks;
                    linear /= kBlocks;
                    const uint32_t h = linear % kvHeads_;
                    const uint32_t b = linear / kvHeads_;
                    const uint32_t rowBegin = block * K_QUANT_BLOCK;
                    const uint32_t rowEnd = Min(kvSeq_, rowBegin + K_QUANT_BLOCK);
                    const uint32_t slotOffset = taskInBatch * ROWS_PER_GROUP * dimPad_;
                    uint32_t rows = 0U;
                    for (uint32_t s = rowBegin; s < rowEnd; ++s) {
                        if (KRowInGroup(s - rowBegin, group)) {
                            AscendC::DataCopy(input16Ub_[slotOffset + static_cast<uint64_t>(rows) * dimPad_],
                                kTyped_[SourceOffset<InputT>(
                                    Layout::HND, b, h, s, 0U, kvHeads_, kvSeq_, dim_)], dim_);
                            ++rows;
                        }
                    }
                    rowCounts[taskInBatch] = rows;
                }
                const uint32_t batchElements = batchTasks * ROWS_PER_GROUP * dimPad_;
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::Cast(inputFloatUb_, input16Ub_, AscendC::RoundMode::CAST_NONE, batchElements);
                if constexpr (SMOOTH_K) {
                    AscendC::Cast(meanFloatUb_, mean16Ub_, AscendC::RoundMode::CAST_NONE, dimPad_);
                    AscendC::PipeBarrier<PIPE_V>();
                    for (uint32_t taskInBatch = 0U; taskInBatch < batchTasks; ++taskInBatch) {
                        const uint32_t slotOffset = taskInBatch * ROWS_PER_GROUP * dimPad_;
                        for (uint32_t r = 0U; r < rowCounts[taskInBatch]; ++r) {
                            AscendC::Sub(inputFloatUb_[slotOffset + static_cast<uint64_t>(r) * dimPad_],
                                inputFloatUb_[slotOffset + static_cast<uint64_t>(r) * dimPad_],
                                meanFloatUb_, dimPad_);
                        }
                    }
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Abs(quantAbsUb_, inputFloatUb_, batchElements);
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t taskInBatch = 0U; taskInBatch < batchTasks; ++taskInBatch) {
                    const uint32_t slotOffset = taskInBatch * ROWS_PER_GROUP * dimPad_;
                    const uint32_t elements = rowCounts[taskInBatch] * dimPad_;
                    if (elements == 0U) {
                        statsSumUb_.SetValue(taskInBatch * 8U, 0.0F);
                        continue;
                    }
                    AscendC::ReduceMax(
                        statsSumUb_[taskInBatch * 8U], quantAbsUb_[slotOffset], quantWorkUb_, elements);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                for (uint32_t taskInBatch = 0U; taskInBatch < batchTasks; ++taskInBatch) {
                    const uint64_t task = batchBase + taskInBatch;
                    const uint32_t slotOffset = taskInBatch * ROWS_PER_GROUP * dimPad_;
                    const uint32_t elements = rowCounts[taskInBatch] * dimPad_;
                    const float scale = statsSumUb_.GetValue(taskInBatch * 8U) / INT8_MAX_VALUE + epsilon_;
                    kScale_.SetValue(task, scale);
                    if (elements != 0U) {
                        AscendC::AscendQuant(quantInt8Ub_[slotOffset], inputFloatUb_[slotOffset],
                            1.0F / scale, 0.0F, elements);
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                for (uint32_t taskInBatch = 0U; taskInBatch < batchTasks; ++taskInBatch) {
                    uint64_t linear = batchBase + taskInBatch;
                    const uint32_t group = linear % kGroups_;
                    linear /= kGroups_;
                    const uint32_t block = linear % kBlocks;
                    linear /= kBlocks;
                    const uint32_t h = linear % kvHeads_;
                    const uint32_t b = linear / kvHeads_;
                    const uint32_t rowBegin = block * K_QUANT_BLOCK;
                    const uint32_t rowEnd = Min(kvSeq_, rowBegin + K_QUANT_BLOCK);
                    const uint32_t slotOffset = taskInBatch * ROWS_PER_GROUP * dimPad_;
                    uint32_t rows = 0U;
                    for (uint32_t s = rowBegin; s < rowEnd; ++s) {
                        if (KRowInGroup(s - rowBegin, group)) {
                            AscendC::DataCopy(k8_[BsNdOffset(b, h, s, 0U, kvHeads_, kvSeq_, dimPad_)],
                                quantInt8Ub_[slotOffset + static_cast<uint64_t>(rows) * dimPad_], dimPad_);
                            ++rows;
                        }
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
    }

    __aicore__ inline void QuantizeV()
    {
        if constexpr (!LAYOUT_NHD && !SMOOTH_V) {
            QuantizeVHnd();
            return;
        }
        const uint32_t vecId = AscendC::GetBlockIdx();
        const uint64_t elementCount = static_cast<uint64_t>(batch_) * kvSeqPad_ * kvHeads_ * dimPad_;
        constexpr uint32_t FP8_ELEMENTS_PER_CACHE_LINE = 64U;
        for (uint64_t indexBase = static_cast<uint64_t>(vecId) * FP8_ELEMENTS_PER_CACHE_LINE;
             indexBase < elementCount;
             indexBase += static_cast<uint64_t>(vecCoreNum_) * FP8_ELEMENTS_PER_CACHE_LINE) {
          const uint64_t indexEnd = Min(elementCount, indexBase + FP8_ELEMENTS_PER_CACHE_LINE);
          for (uint64_t index = indexBase; index < indexEnd; ++index) {
            uint64_t linear = index;
            const uint32_t d = linear % dimPad_;
            linear /= dimPad_;
            const uint32_t h = linear % kvHeads_;
            linear /= kvHeads_;
            const uint32_t s = linear % kvSeqPad_;
            const uint32_t b = linear / kvSeqPad_;
            const uint64_t channelOffset = (static_cast<uint64_t>(b) * kvHeads_ + h) * dimPad_ + d;
            float value = 0.0F;
            if (s < kvSeq_ && d < dim_) {
                value = DecodeInput16<InputT>(
                    v_.GetValue(SourceOffset<InputT>(layout_, b, h, s, d, kvHeads_, kvSeq_, dim_)));
                if constexpr (SMOOTH_V) {
                    value -= vMean_.GetValue(channelOffset);
                }
                value /= vScale_.GetValue(channelOffset);
            }
            v8_.SetValue(index, EncodeFp8E4M3Fn(value));
          }
        }
    }

    __simd_vf__ static inline void EncodeFp8PackedVf(__ubuf__ float *src, __ubuf__ float *channelScale,
        __ubuf__ uint8_t *dst, __ubuf__ uint8_t *packIndex, uint16_t rows)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait CAST_FP8_EVEN = {
            RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        constexpr static CastTrait CAST_FP8_ODD = {
            RegLayout::TWO, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

        RegTensor<float> value0;
        RegTensor<float> value1;
        RegTensor<float> scale0;
        RegTensor<float> scale1;
        RegTensor<fp8_e4m3fn_t> fp8Even;
        RegTensor<fp8_e4m3fn_t> fp8Odd;
        RegTensor<fp8_e4m3fn_t> fp8Merged;
        RegTensor<fp8_e4m3fn_t> fp8Packed;
        RegTensor<uint8_t> index;
        MaskReg fp32Full = CreateMask<float, MaskPattern::ALL>();
        MaskReg fp8Full = CreateMask<fp8_e4m3fn_t, MaskPattern::ALL>();
        uint32_t fp8RowLength = 128U;
        MaskReg fp8Row = UpdateMask<fp8_e4m3fn_t>(fp8RowLength);

        LoadAlign(index, packIndex);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(scale0, scale1, channelScale);
        for (uint16_t r = 0U; r < rows; ++r) {
            const uint32_t rowOffset = static_cast<uint32_t>(r) * 128U;
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(value0, value1, src + rowOffset);
            Div(value0, value0, scale0, fp32Full);
            Div(value1, value1, scale1, fp32Full);
            Cast<fp8_e4m3fn_t, float, CAST_FP8_EVEN>(fp8Even, value0, fp32Full);
            Cast<fp8_e4m3fn_t, float, CAST_FP8_ODD>(fp8Odd, value1, fp32Full);
            Or(reinterpret_cast<RegTensor<uint8_t> &>(fp8Merged),
                reinterpret_cast<RegTensor<uint8_t> &>(fp8Even),
                reinterpret_cast<RegTensor<uint8_t> &>(fp8Odd), fp8Full);
            Gather(fp8Packed, fp8Merged, index);
            StoreAlign(reinterpret_cast<__ubuf__ fp8_e4m3fn_t *>(dst) + rowOffset, fp8Packed, fp8Row);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    __aicore__ inline void EncodeFp8Packed(const AscendC::LocalTensor<float> &src,
        const AscendC::LocalTensor<float> &channelScale, const AscendC::LocalTensor<int8_t> &dst, uint32_t rows)
    {
        auto srcAddr = reinterpret_cast<__ubuf__ float *>(src.GetPhyAddr());
        auto scaleAddr = reinterpret_cast<__ubuf__ float *>(channelScale.GetPhyAddr());
        auto dstAddr = reinterpret_cast<__ubuf__ uint8_t *>(dst.GetPhyAddr());
        auto indexAddr = reinterpret_cast<__ubuf__ uint8_t *>(fp8PackIndexUb_.GetPhyAddr());
        EncodeFp8PackedVf(srcAddr, scaleAddr, dstAddr, indexAddr, static_cast<uint16_t>(rows));
    }

    __aicore__ inline void QuantizeVHnd()
    {
        // The packed native-FP8 path below is specialized for two interleaved 64-lane registers (D=128).
        // Keep the scalar encoder for D=64 and non-padded tails until a separately packed one-register path exists.
        if (dim_ != dimPad_ || dimPad_ != 128U) {
            QuantizeVHndScalar();
            return;
        }
        constexpr uint32_t V_ROW_BATCH = 128U;
        constexpr uint32_t V_ROWS_PER_TASK = 128U;
        const uint32_t vecId = AscendC::GetBlockIdx();
        const uint64_t headCount = static_cast<uint64_t>(batch_) * kvHeads_;
        const uint32_t rowTasks = (kvSeq_ + V_ROWS_PER_TASK - 1U) / V_ROWS_PER_TASK;
        const uint64_t taskCount = headCount * rowTasks;
        for (uint32_t i = 0U; i < 128U; ++i) {
            fp8PackIndexUb_.SetValue(i, static_cast<uint8_t>(i << 1U));
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        for (uint64_t task = vecId; task < taskCount; task += vecCoreNum_) {
            const uint32_t rowTask = task % rowTasks;
            const uint64_t headTask = task / rowTasks;
            const uint32_t h = headTask % kvHeads_;
            const uint32_t b = headTask / kvHeads_;
            const uint32_t rowBegin = rowTask * V_ROWS_PER_TASK;
            const uint32_t rowEnd = Min(kvSeq_, rowBegin + V_ROWS_PER_TASK);
            AscendC::DataCopy(channelScaleUb_, vScale_[headTask * dimPad_], dimPad_);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            for (uint32_t rowBase = rowBegin; rowBase < rowEnd; rowBase += V_ROW_BATCH) {
                const uint32_t rows = Min(V_ROW_BATCH, rowEnd - rowBase);
                const uint32_t elements = rows * dimPad_;
                AscendC::DataCopy(input16Ub_,
                    vTyped_[SourceOffset<InputT>(Layout::HND, b, h, rowBase, 0U, kvHeads_, kvSeq_, dim_)],
                    elements);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::Cast(inputFloatUb_, input16Ub_, AscendC::RoundMode::CAST_NONE, elements);
                AscendC::PipeBarrier<PIPE_V>();
                EncodeFp8Packed(inputFloatUb_, channelScaleUb_, quantInt8Ub_, rows);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                if (rows == V_ROW_BATCH) {
                    const uint64_t v8Offset =
                        BsNdOffset(b, h, rowBase, 0U, kvHeads_, kvSeqPad_, dimPad_);
                    const uint16_t dstGap = static_cast<uint16_t>((kvHeads_ - 1U) * 4U);
                    AscendC::DataCopy(v8_[v8Offset], quantInt8Ub_.template ReinterpretCast<uint8_t>(),
                        {static_cast<uint16_t>(V_ROW_BATCH), 4U, 0U, dstGap});
                } else {
                    for (uint32_t r = 0U; r < rows; ++r) {
                        AscendC::DataCopy(v8_[BsNdOffset(
                            b, h, rowBase + r, 0U, kvHeads_, kvSeqPad_, dimPad_)],
                            quantInt8Ub_[static_cast<uint64_t>(r) * dimPad_].template ReinterpretCast<uint8_t>(),
                            dimPad_);
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
    }

    __aicore__ inline void QuantizeVHndScalar()
    {
        const uint32_t vecId = AscendC::GetBlockIdx();
        const uint64_t headCount = static_cast<uint64_t>(batch_) * kvHeads_;
        for (uint64_t task = vecId; task < headCount; task += vecCoreNum_) {
            const uint32_t h = task % kvHeads_;
            const uint32_t b = task / kvHeads_;
            AscendC::DataCopy(channelScaleUb_, vScale_[task * dimPad_], dimPad_);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            for (uint32_t s = 0U; s < kvSeq_; ++s) {
                AscendC::DataCopy(input16Ub_,
                    vTyped_[SourceOffset<InputT>(Layout::HND, b, h, s, 0U, kvHeads_, kvSeq_, dim_)], dim_);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::Cast(inputFloatUb_, input16Ub_, AscendC::RoundMode::CAST_NONE, dim_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Div(quantAbsUb_, inputFloatUb_, channelScaleUb_, dim_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Abs(inputFloatUb_, quantAbsUb_, dim_);
                AscendC::PipeBarrier<PIPE_V>();
                // A per-channel maximum can divide to a value slightly above 448 because the scale was
                // rounded to FP32. Clamp before the manual E4M3FN encoder; code 0x7f is NaN, not a larger
                // finite value. D128 uses the native saturating cast and does not need this scalar guard.
                AscendC::Mins(inputFloatUb_, inputFloatUb_, FP8_MAX_VALUE, dim_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(quantWorkUb_, inputFloatUb_, 512.0F, dim_);
                AscendC::PipeBarrier<PIPE_V>();
#define SAGE_FP8_SEGMENT(threshold, multiplier, addend) \
                AscendC::Muls(meanFloatUb_, inputFloatUb_, multiplier, dim_); \
                AscendC::PipeBarrier<PIPE_V>(); \
                AscendC::Adds(meanFloatUb_, meanFloatUb_, addend, dim_); \
                AscendC::PipeBarrier<PIPE_V>(); \
                AscendC::CompareScalar(quantInt8Ub_.template ReinterpretCast<uint8_t>(), inputFloatUb_, threshold, \
                    AscendC::CMPMODE::GE, dim_); \
                AscendC::PipeBarrier<PIPE_V>(); \
                AscendC::Select(quantWorkUb_, quantInt8Ub_.template ReinterpretCast<uint8_t>(), meanFloatUb_, \
                    quantWorkUb_, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, dim_); \
                AscendC::PipeBarrier<PIPE_V>()
                SAGE_FP8_SEGMENT(0.03125F, 256.0F, 8.0F);
                SAGE_FP8_SEGMENT(0.0625F, 128.0F, 16.0F);
                SAGE_FP8_SEGMENT(0.125F, 64.0F, 24.0F);
                SAGE_FP8_SEGMENT(0.25F, 32.0F, 32.0F);
                SAGE_FP8_SEGMENT(0.5F, 16.0F, 40.0F);
                SAGE_FP8_SEGMENT(1.0F, 8.0F, 48.0F);
                SAGE_FP8_SEGMENT(2.0F, 4.0F, 56.0F);
                SAGE_FP8_SEGMENT(4.0F, 2.0F, 64.0F);
                SAGE_FP8_SEGMENT(8.0F, 1.0F, 72.0F);
                SAGE_FP8_SEGMENT(16.0F, 0.5F, 80.0F);
                SAGE_FP8_SEGMENT(32.0F, 0.25F, 88.0F);
                SAGE_FP8_SEGMENT(64.0F, 0.125F, 96.0F);
                SAGE_FP8_SEGMENT(128.0F, 0.0625F, 104.0F);
                SAGE_FP8_SEGMENT(256.0F, 0.03125F, 112.0F);
#undef SAGE_FP8_SEGMENT
                AscendC::Adds(meanFloatUb_, quantWorkUb_, -128.0F, dim_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::CompareScalar(quantInt8Ub_.template ReinterpretCast<uint8_t>(), quantAbsUb_, 0.0F,
                    AscendC::CMPMODE::LT, dim_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Select(quantWorkUb_, quantInt8Ub_.template ReinterpretCast<uint8_t>(), meanFloatUb_,
                    quantWorkUb_, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, dim_);
                AscendC::PipeBarrier<PIPE_V>();
                if (dim_ < dimPad_) {
                    AscendC::Duplicate(quantInt8Ub_, static_cast<int8_t>(0), dimPad_);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::AscendQuant(quantInt8Ub_, quantWorkUb_, 1.0F, 0.0F, dim_);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::DataCopy(v8_[BsNdOffset(b, h, s, 0U, kvHeads_, kvSeqPad_, dimPad_)],
                    quantInt8Ub_.template ReinterpretCast<uint8_t>(), dimPad_);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
    }

    AscendC::GlobalTensor<uint16_t> q_;
    AscendC::GlobalTensor<uint16_t> k_;
    AscendC::GlobalTensor<uint16_t> v_;
    AscendC::GlobalTensor<InputT> kTyped_;
    AscendC::GlobalTensor<InputT> vTyped_;
    AscendC::GlobalTensor<InputT> qTyped_;
    AscendC::GlobalTensor<int8_t> q8_;
    AscendC::GlobalTensor<int8_t> k8_;
    AscendC::GlobalTensor<uint8_t> v8_;
    AscendC::GlobalTensor<float> qScale_;
    AscendC::GlobalTensor<float> kScale_;
    AscendC::GlobalTensor<float> vScale_;
    AscendC::GlobalTensor<uint16_t> kMean_;
    AscendC::GlobalTensor<InputT> kMeanTyped_;
    AscendC::GlobalTensor<float> vMean_;
    AscendC::LocalTensor<InputT> input16Ub_;
    AscendC::LocalTensor<float> inputFloatUb_;
    AscendC::LocalTensor<float> statsSumUb_;
    AscendC::LocalTensor<float> statsMaxUb_;
    AscendC::LocalTensor<float> quantAbsUb_;
    AscendC::LocalTensor<float> quantWorkUb_;
    AscendC::LocalTensor<int8_t> quantInt8Ub_;
    AscendC::LocalTensor<InputT> mean16Ub_;
    AscendC::LocalTensor<float> meanFloatUb_;
    AscendC::LocalTensor<float> vCandidateUb_;
    AscendC::LocalTensor<float> channelScaleUb_;
    AscendC::LocalTensor<uint8_t> fp8PackIndexUb_;
    uint32_t batch_ = 0U;
    uint32_t qHeads_ = 0U;
    uint32_t kvHeads_ = 0U;
    uint32_t qSeq_ = 0U;
    uint32_t kvSeq_ = 0U;
    uint32_t dim_ = 0U;
    uint32_t dimPad_ = 0U;
    uint32_t kvSeqPad_ = 0U;
    uint32_t qBlocks_ = 0U;
    uint32_t qGroups_ = 0U;
    uint32_t kGroups_ = 0U;
    uint32_t vecCoreNum_ = 0U;
    float epsilon_ = 0.0F;
    float vScaleMax_ = 0.0F;
    Layout layout_ = Layout::HND;
    QuantGran gran_ = QuantGran::PER_THREAD;
};

} // namespace SageAttention2Kernel
#endif // SAGE_ATTENTION2_PREPROCESS_H_
