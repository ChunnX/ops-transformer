/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Block-level RescaleO epilogue for prefill Phase2 (Arch35 / Atlas A5).
 *
 * Combines topK partial results: (O_tmp[gh,D], LSE[gh])
 * into final output O[groupSize, D] via LSE logsumexp weighting (design doc §7.2).
 */
#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_PREFILL_ARCH35_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_PREFILL_ARCH35_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "kernel_operator.h"

namespace NpuArch::Epilogue::Block {

template <class InDtype>
class BlockEpilogue<
    EpilogueRescaleOPrefillArch35,
    InDtype>
{
public:
    using DispatchPolicy = EpilogueRescaleOPrefillArch35;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementO = InDtype;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr float NEG_INF_LSE = -3.4028235e38f;

    __aicore__ inline
    BlockEpilogue() : D_(0), groupSize_(0) {}

    __aicore__ inline
    void Init(AscendC::LocalTensor<float> &ubAccBuf,
              AscendC::LocalTensor<float> &ubTmpBuf,
              AscendC::LocalTensor<float> &ubMetaBuf,
              uint32_t D,
              uint32_t groupSize)
    {
        accBuf_ = ubAccBuf;
        tmpBuf_ = ubTmpBuf;
        metaBuf_ = ubMetaBuf;
        D_ = D;
        groupSize_ = groupSize;
    }

    __aicore__ inline
    void operator()(AscendC::GlobalTensor<float> &gmWorkspaceO,
                    AscendC::GlobalTensor<float> &gmWorkspaceF,
                    AscendC::GlobalTensor<InDtype> &gmO,
                    uint32_t qToken,
                    uint32_t kvHeadIdx,
                    uint32_t qHeadStart,
                    uint32_t groupSize,
                    uint32_t D,
                    uint32_t topK,
                    uint32_t kvHeads,
                    uint32_t numHeads,
                    uint64_t partialSize,
                    uint64_t oBytesAligned,
                    uint64_t wsBase)
    {
        (void)kvHeadIdx;
        int32_t dLen = static_cast<int32_t>(D);

        for (uint32_t gh = 0; gh < groupSize; gh++) {
            float maxLse = NEG_INF_LSE;
            for (uint32_t k = 0; k < topK; k++) {
                uint64_t pBytes = wsBase + static_cast<uint64_t>(k) * partialSize;
                uint64_t fOff = (pBytes + oBytesAligned) / sizeof(float);
                LoadPartialMeta(gmWorkspaceF, fOff);
                float lse = metaBuf_.GetValue(gh);
                if (lse <= NEG_INF_LSE * 0.5f) {
                    continue;
                }
                if (lse > maxLse) {
                    maxLse = lse;
                }
            }
            if (maxLse <= NEG_INF_LSE * 0.5f) {
                continue;
            }

            float sumExp = 0.0f;
            for (uint32_t k = 0; k < topK; k++) {
                uint64_t pBytes = wsBase + static_cast<uint64_t>(k) * partialSize;
                uint64_t fOff = (pBytes + oBytesAligned) / sizeof(float);
                LoadPartialMeta(gmWorkspaceF, fOff);
                float lse = metaBuf_.GetValue(gh);
                if (lse <= NEG_INF_LSE * 0.5f) {
                    continue;
                }
                sumExp += ScalarExp(lse - maxLse);
            }
            if (sumExp <= 0.0f) {
                continue;
            }
            tmpBuf_.SetValue(D_, sumExp);
            AscendC::Ln(tmpBuf_[D_], tmpBuf_[D_], 1);
            AscendC::PipeBarrier<PIPE_V>();
            float lseGlobal = maxLse + tmpBuf_.GetValue(D_);

            bool isFirst = true;
            uint32_t accOff = gh * D;

            for (uint32_t k = 0; k < topK; k++) {
                uint64_t pBytes = wsBase + static_cast<uint64_t>(k) * partialSize;
                uint64_t oOff = pBytes / sizeof(float);
                uint64_t fOff = (pBytes + oBytesAligned) / sizeof(float);
                LoadPartialMeta(gmWorkspaceF, fOff);
                float lse = metaBuf_.GetValue(gh);
                if (lse <= NEG_INF_LSE * 0.5f) {
                    continue;
                }
                float weight = ScalarExp(lse - lseGlobal);

                LoadPartialO(gmWorkspaceO, oOff + static_cast<uint64_t>(gh) * D, D);

                AscendC::Muls(tmpBuf_, tmpBuf_, weight, dLen);
                AscendC::PipeBarrier<PIPE_V>();

                if (isFirst) {
                    AscendC::Adds(accBuf_[accOff], tmpBuf_, 0.0f, dLen);
                    AscendC::PipeBarrier<PIPE_V>();
                    isFirst = false;
                } else {
                    AscendC::Add(accBuf_[accOff], accBuf_[accOff], tmpBuf_, dLen);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }

            if (!isFirst) {
                StoreOutputO(gmO, qToken, qHeadStart + gh, numHeads, D, accOff);
            }
        }
    }

private:
    AscendC::LocalTensor<float> accBuf_;
    AscendC::LocalTensor<float> tmpBuf_;
    AscendC::LocalTensor<float> metaBuf_;
    uint32_t D_;
    uint32_t groupSize_;

    __aicore__ inline
    void LoadPartialMeta(AscendC::GlobalTensor<float> &gmWorkspaceF, uint64_t fOff)
    {
        AscendC::DataCopy(metaBuf_, gmWorkspaceF[fOff], groupSize_);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline float ScalarExp(float x)
    {
        uint32_t expOff = D_;
        tmpBuf_.SetValue(expOff, x);
        AscendC::Exp(tmpBuf_[expOff], tmpBuf_[expOff], FLOAT_VECTOR_SIZE);
        AscendC::PipeBarrier<PIPE_V>();
        return tmpBuf_.GetValue(expOff);
    }

    __aicore__ inline
    void LoadPartialO(AscendC::GlobalTensor<float> &gmSrc,
                      uint64_t srcOff, uint32_t D)
    {
        AscendC::DataCopy(tmpBuf_, gmSrc[srcOff], D);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
    }

    __aicore__ inline
    void StoreOutputO(AscendC::GlobalTensor<InDtype> &gmO,
                      uint32_t qToken, uint32_t headIdx,
                      uint32_t numHeads, uint32_t D, uint32_t accOff)
    {
        uint64_t outBase = static_cast<uint64_t>(qToken) * numHeads * D
            + static_cast<uint64_t>(headIdx) * D;
        AscendC::LocalTensor<InDtype> ubOutO = tmpBuf_.template ReinterpretCast<InDtype>();
        AscendC::Cast(
            ubOutO, accBuf_[accOff],
            AscendC::RoundMode::CAST_RINT,
            static_cast<int32_t>(D));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
        AscendC::DataCopy(gmO[outBase], ubOutO, D);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
    }
};

}  // namespace NpuArch::Epilogue::Block

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_PREFILL_ARCH35_HPP
