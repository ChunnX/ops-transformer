/**
 * Ascend950 (arch35) entry include for MoeFusedTopk.
 *
 * Keep the architecture selection and arch35-specific implementations at the
 * entry point. The common algorithm is declared in the parent header.
 */
#ifndef ASCENDC_MOE_FUSED_TOPK_ARCH35_H_
#define ASCENDC_MOE_FUSED_TOPK_ARCH35_H_

#define MOE_FUSED_TOPK_ARCH35 1
#include "moe_fused_topk_arch35_tiling_def.h"
#include "../moe_fused_topk.h"
#include "adv_api/reduce/sum.h"

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopk<inputT, calT, enableExpertMapping>::InitBuffer(TPipe *inputPipe)
{
    pipe_ = inputPipe;
    pipe_->InitBuffer(xInBuf_, sizeof(float) * groupElesAlignSortCount_ * groupNum_);
    pipe_->InitBuffer(addNumInBuf_, sizeof(float) * secondAlignBlockCountFp16_);
    pipe_->InitBuffer(yOutBuf_, sizeof(float) * topkPad_);
    pipe_->InitBuffer(indicesOutBuf_, sizeof(int32_t) * topKSortRepeatTimes_ * SORT_UNIT);
    pipe_->InitBuffer(sigmoidBuf_, sizeof(float) * secondAlignBlockCountFp32_);
    pipe_->InitBuffer(sigmoidAddBuf_, sizeof(float) * secondAlignBlockCountFp32_);
    // The temporary buffer is shared by TopK, Sigmoid, BroadCast, Sort and Sum on arch35.
    pipe_->InitBuffer(tempBuf_, topkMaxValue_);
    pipe_->InitBuffer(sortedBuf_, sizeof(int64_t) * groupNum_ * groupElesAlignSortCount_);
    pipe_->InitBuffer(topkValueBuf_, sizeof(float) * sortRepeatTimes_ * SORT_UNIT);
    pipe_->InitBuffer(assistBuf_, sizeof(uint32_t) * wholeSortNum_);
    if constexpr (enableExpertMapping) {
        pipe_->InitBuffer(mappingNumBuf_, sizeof(int32_t) * expertNum_);
    }
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopk<inputT, calT, enableExpertMapping>::GroupReduceSumInternelImpl()
{
    LocalTensor<float> xLocal = xInBuf_.Get<float>();
    LocalTensor<float> sortedTensor = sortedBuf_.Get<float>();
    LocalTensor<float> topkGroupValue = topkValueBuf_.Get<float>();
    LocalTensor<uint8_t> tempTensor = tempBuf_.Get<uint8_t>();
    LocalTensor<int32_t> indicesLocal = sortedTensor[wholeSortNum_].template ReinterpretCast<int32_t>();
    LocalTensor<bool> finishLocal;
    AscendC::TopKInfo topkInfo = {static_cast<int32_t>(groupNum_),
                                  CeilAlign(static_cast<int32_t>(groupEles_), SORT_UNIT),
                                  static_cast<int32_t>(groupEles_)};
    AscendC::TopK<calT, false, false, false, TopKMode::TOPK_NORMAL>(
        sortedTensor, indicesLocal, xLocal, indicesLocal, finishLocal, tempTensor, n_, topkTilingData_, topkInfo);
    AscendC::PipeBarrier<PIPE_V>();
    Duplicate<float>(topkGroupValue, floatNegativeInf_, CeilAlign(groupNum_, SORT_UNIT));
    AscendC::PipeBarrier<PIPE_V>();
    // 3510's basic ReduceSum cannot safely write every unaligned group scalar.
    AscendC::Sum<calT>(topkGroupValue, sortedTensor, tempTensor, {groupNum_, topnPad_, n_});
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopk<inputT, calT, enableExpertMapping>::GroupTopkImpl()
{
    // Sort the group scores and retain the hardware ordering for equal and
    // non-finite scores. The ATK reference uses the same AscendC TopK/Sort
    // tie behavior for these non-finite anomaly inputs (notably -inf).
    LocalTensor<uint32_t> assistLocal = assistBuf_.Get<uint32_t>();
    LocalTensor<float> sortedTensor = sortedBuf_.Get<float>();
    LocalTensor<float> topkGroupValue = topkValueBuf_.Get<float>();
    LocalTensor<float> sigmoidAddTensor = sigmoidAddBuf_.Get<float>();
    LocalTensor<float> tempTensor = tempBuf_.Get<float>();
    LocalTensor<uint8_t> selectedGroupMask = assistBuf_.Get<uint8_t>();

    Sort<float, true>(sortedTensor, topkGroupValue, assistLocal, tempTensor, CeilDiv(groupNum_, SORT_UNIT));
    AscendC::PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> dstOffset = sortedTensor.template ReinterpretCast<int32_t>();
    Duplicate(topkGroupValue, float(0), groupNum_);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    for (size_t i = 0; i < groupTopk_; i++) {
        int32_t selectedGroupIndex = dstOffset.GetValue(SORTED_COEF * i + 1);
        topkGroupValue.SetValue(selectedGroupIndex, float(1));
    }
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    uint32_t dstShape[BROADCAST_DIM] = {(uint32_t)groupNum_, (uint32_t)groupEles_};
    uint32_t srcShape[BROADCAST_DIM] = {(uint32_t)groupNum_, 1};
    LocalTensor<uint8_t> sharedTmpBuffer = tempBuf_.Get<uint8_t>();
    AscendC::PipeBarrier<PIPE_V>();
    BroadCast<float, BROADCAST_DIM, BROADCAST_AXIS>(sortedTensor, topkGroupValue, dstShape, srcShape, sharedTmpBuffer);
    AscendC::PipeBarrier<PIPE_V>();
    // Use IEEE -inf for the aligned tail. Valid -inf scores are clamped to
    // the finite sentinel below, so padded slots can never enter expert TopK.
    uint32_t negativeInfinityBits = 0xFF800000;
    float floatNegativeInfinity = *((float *)&negativeInfinityBits);
    Duplicate(topkGroupValue, floatNegativeInfinity, wholeSortNum_);
    // The reference semantics use +0 for a valid but deselected expert;
    // only the aligned tail must remain -inf.
    Duplicate(topkGroupValue, float(0), secondDimSize_);
    AscendC::PipeBarrier<PIPE_V>();
    CompareScalar(selectedGroupMask, sortedTensor, float(0), CMPMODE::GT, secondDimSize_);
    // Clamp only the candidate scores. The destination remains -inf for
    // deselected experts and for the aligned tail, while a selected -inf is
    // represented by the finite sentinel and therefore stays ahead of tail
    // slots during Sort.
    Maxs(sigmoidAddTensor, sigmoidAddTensor, floatNegativeInf_, secondDimSize_);
    Select(topkGroupValue, selectedGroupMask, sigmoidAddTensor, topkGroupValue, SELMODE::VSEL_TENSOR_TENSOR_MODE,
           secondDimSize_);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopk<inputT, calT, enableExpertMapping>::GatherSigmoidImpl()
{
    LocalTensor<int32_t> indicesLocal = indicesOutBuf_.Get<int32_t>();
    LocalTensor<float> sigmoidTensor = sigmoidBuf_.Get<float>();
    LocalTensor<float> sortedTensor = sortedBuf_.Get<float>();
    LocalTensor<float> groupTopkValue = topkValueBuf_.Get<float>();
    LocalTensor<float> yLocal = yOutBuf_.Get<float>();
    LocalTensor<uint32_t> assistLocal = assistBuf_.Get<uint32_t>();
    LocalTensor<int32_t> sortIndexInit = assistBuf_.Get<int32_t>();
    LocalTensor<float> tempTensor = tempBuf_.Get<float>();

    // Sort/Extract stay entirely on vector hardware; no O(topK * expertNum)
    // scalar tie scan is allowed on the performance path.
    ArithProgression(sortIndexInit, 0, 1, wholeSortNum_);
    AscendC::PipeBarrier<PIPE_V>();
    Sort<float, true>(sortedTensor, groupTopkValue, assistLocal, tempTensor, sortRepeatTimes_);
    AscendC::PipeBarrier<PIPE_V>();
    Extract(tempTensor, indicesLocal.template ReinterpretCast<uint32_t>(), sortedTensor, topKSortRepeatTimes_);
    AscendC::PipeBarrier<PIPE_V>();
    Muls(sortedTensor.template ReinterpretCast<int32_t>(), indicesLocal, FLOAT_BYTES, k_);
    AscendC::PipeBarrier<PIPE_V>();
    Gather(yLocal, sigmoidTensor, sortedTensor.template ReinterpretCast<uint32_t>(), 0, k_);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopk<inputT, calT, enableExpertMapping>::NormImpl()
{
    LocalTensor<float> yLocal = yOutBuf_.Get<float>();
    LocalTensor<float> tempTensor = tempBuf_.Get<float>();
    LocalTensor<uint8_t> sharedTmpBuffer = tempBuf_.Get<uint8_t>();
    AscendC::Sum<calT>(tempTensor, yLocal, sharedTmpBuffer, {1, topkPad_, k_});
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    float reduceSumValue = 1 / tempTensor.GetValue(0);
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    Muls(yLocal, yLocal, reduceSumValue, k_);
    AscendC::PipeBarrier<PIPE_V>();
    Muls(yLocal, yLocal, scale_, k_);
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void MoeFusedTopk<inputT, calT, enableExpertMapping>::CopyInMappingNum()
{
    LocalTensor<int32_t> mappingNumLocal = mappingNumBuf_.Get<int32_t>();
    DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(expertNum_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(mappingNumLocal, mappingNumGm_, copyParams, padParams);
}

#endif // ASCENDC_MOE_FUSED_TOPK_ARCH35_H_
