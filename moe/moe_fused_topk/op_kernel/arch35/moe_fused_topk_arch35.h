/**
 * Ascend950 (arch35) entry include for MoeFusedTopk.
 *
 * Keep the architecture selection and arch35-specific implementations at the
 * entry point. The common algorithm is declared in the parent header.
 */
#ifndef ASCENDC_MOE_FUSED_TOPK_ARCH35_H_
#define ASCENDC_MOE_FUSED_TOPK_ARCH35_H_

#define MOE_FUSED_TOPK_ARCH35 1
#include "../moe_fused_topk.h"
#include "adv_api/reduce/sum.h"

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
