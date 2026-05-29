#ifndef MOE_EP_DISPATCH_EPILOGUE_H
#define MOE_EP_DISPATCH_EPILOGUE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "moe_ep_dispatch_epilogue_tiling_key.h"

#if __has_include("../common/op_kernel/moe_distribute_base.h")
#include "../common/op_kernel/moe_distribute_base.h"
#include "../common/op_kernel/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

#include "moe_ep_dispatch_epilogue_tiling_data.h"

namespace MoeEpDispatchEpilogueImpl {

using namespace AscendC;

static constexpr uint32_t WIN_ADDR_ALIGN = 512;

class KernelDispatchEpilogue {
public:
    __aicore__ inline KernelDispatchEpilogue() {};
    __aicore__ inline void Init(
        GM_ADDR dstBufferSlotIdx, GM_ADDR recvSrcMetadataInternal,
        GM_ADDR numRecvPerRankHp, GM_ADDR numRecvPerExpertHp,
        GM_ADDR recvX, GM_ADDR recvTopkIdx, GM_ADDR recvTopkWeights,
        GM_ADDR recvSrcMetadata, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
        GM_ADDR workspace, GM_ADDR tilingGM,
        TPipe* pipe, const MoeEpDispatchEpilogueInfo* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputePrefixSum();
    __aicore__ inline void CopyFromWindow();
    __aicore__ inline void CopyHostPinnedToDevice();

    TPipe* tpipe_{nullptr};
    const MoeEpDispatchEpilogueInfo* td_{nullptr};
    __gm__ Mc2Kernel::HcclOpParam* winContext_{nullptr};

    GlobalTensor<int32_t> dstSlotIdxGm_;
    GlobalTensor<int32_t> recvSrcMetaInternalGm_;
    GlobalTensor<int32_t> numRecvPerRankHpGm_;
    GlobalTensor<int64_t> numRecvPerExpertHpGm_;
    GlobalTensor<bfloat16_t> recvXGm_;
    GlobalTensor<int32_t> recvTopkIdxGm_;
    GlobalTensor<float> recvTopkWeightsGm_;
    GlobalTensor<int32_t> recvSrcMetadataGm_;
    GlobalTensor<int32_t> numRecvPerRankGm_;
    GlobalTensor<int64_t> numRecvPerExpertGm_;

    LocalTensor<bfloat16_t> ubToken_[2];
    LocalTensor<int32_t> ubMeta_;
    LocalTensor<int32_t> ubPrefixSum_;

    TBuf<QuePosition::VECIN> ubTokenBuf_[2];
    TBuf<QuePosition::VECIN> ubMetaBuf_;
    TBuf<QuePosition::VECIN> ubPrefixSumBuf_;

    GM_ADDR workspace_{nullptr};
    uint32_t aivId_{0};
};

}  // namespace MoeEpDispatchEpilogueImpl

#endif  // MOE_EP_DISPATCH_EPILOGUE_H