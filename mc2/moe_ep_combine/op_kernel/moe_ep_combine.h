#ifndef MOE_EP_COMBINE_H
#define MOE_EP_COMBINE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "moe_ep_combine_tiling_key.h"

#if __has_include("../common/op_kernel/moe_distribute_base.h")
#include "../common/op_kernel/moe_distribute_base.h"
#include "../common/op_kernel/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

#include "moe_ep_combine_tiling_data.h"

namespace MoeEpCombineImpl {

using namespace AscendC;

static constexpr uint32_t MAX_TOPK = 8;

template <uint32_t HAS_TOPK_WEIGHTS>
class KernelMoeEpCombine {
public:
    __aicore__ inline KernelMoeEpCombine() {};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR dstBufferSlotIdx, GM_ADDR recvSrcMetadata,
        GM_ADDR numRecvPerRank, GM_ADDR topkWeights,
        GM_ADDR bias0, GM_ADDR bias1,
        GM_ADDR combinedX, GM_ADDR combinedTopkWeightsOptional,
        GM_ADDR workspace, GM_ADDR tilingGM,
        TPipe* pipe, const MoeEpCombineInfo* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SendPhaseExpertToToken();
    __aicore__ inline void RecvPhaseStashing();
    __aicore__ inline void RecvPhaseReduce();

    TPipe* tpipe_{nullptr};
    const MoeEpCombineInfo* td_{nullptr};
    __gm__ Mc2Kernel::HcclOpParam* winContext_{nullptr};

    GlobalTensor<bfloat16_t> xGm_;
    GlobalTensor<int32_t> dstSlotIdxGm_;
    GlobalTensor<int32_t> recvSrcMetadataGm_;
    GlobalTensor<int32_t> numRecvPerRankGm_;
    GlobalTensor<float> topkWeightsGm_;
    GlobalTensor<bfloat16_t> combinedXGm_;
    GlobalTensor<float> combinedTopkWeightsGm_;

    LocalTensor<bfloat16_t> ubX_[2];
    LocalTensor<float> ubAccFp32_;
    LocalTensor<float> ubTmpFp32_;
    LocalTensor<float> ubWeighted_;

    TBuf<QuePosition::VECIN> ubXBuf_[2];
    TBuf<QuePosition::VECIN> ubAccFp32Buf_;
    TBuf<QuePosition::VECIN> ubTmpFp32Buf_;
    TBuf<QuePosition::VECIN> ubWeightedBuf_;

    GM_ADDR workspace_{nullptr};
    uint32_t aivId_{0};
};

}  // namespace MoeEpCombineImpl

#endif  // MOE_EP_COMBINE_H