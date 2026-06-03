#ifndef MOE_EP_DISPATCH_EPILOGUE_H
#define MOE_EP_DISPATCH_EPILOGUE_H

#include "kernel_operator.h"
#include "kernel_operator_data_copy_intf.h"

#include "moe_ep_dispatch_epilogue_tiling_key.h"

#if __has_include("../common/op_kernel/moe_distribute_base.h")
#include "../common/op_kernel/moe_distribute_base.h"
#include "../common/op_kernel/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

#include "../../common/op_kernel/mc2_moe_context.h"
#include "moe_ep_dispatch_epilogue_tiling.h"

#ifndef ALIGN_UP
#define ALIGN_UP(x, align) (((x) + (align)-1) & ~((align)-1))
#endif

namespace MoeEpDispatchEpilogueImpl {

using namespace AscendC;
using Mc2Aclnn::Mc2MoeContext;
using Mc2Aclnn::HCCL_MAX_RANK_SIZE;

inline GM_ADDR winRankAddr_[HCCL_MAX_RANK_SIZE];

__aicore__ inline GM_ADDR GetBaseWinAddrByRankId(uint32_t rankId)
{
    return (GM_ADDR)(winRankAddr_[rankId] + Mc2Kernel::A5_MTE_STATE_WIN_SIZE);
}

static constexpr uint32_t WIN_ADDR_ALIGN = 512;

class KernelDispatchEpilogue {
public:
    __aicore__ inline KernelDispatchEpilogue() {};

    __aicore__ inline void Init(
        GM_ADDR context, GM_ADDR dstBufferSlotIdx, GM_ADDR recvSrcMetadataInternal,
        GM_ADDR numRecvPerRankHp, GM_ADDR numRecvPerExpertHp,
        GM_ADDR recvX, GM_ADDR recvTopkIdx, GM_ADDR recvTopkWeights,
        GM_ADDR recvSrcMetadata, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
        GM_ADDR workspace, GM_ADDR tilingGM,
        TPipe* pipe, const MoeEpDispatchEpilogueInfo* tilingData)
    {
        tpipe_ = pipe;
        td_ = tilingData;
        aivId_ = GetBlockIdx();
        workspace_ = workspace;

        mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext*>(context);
        rankId_ = mc2Context_->epRankId;
        for (uint32_t i = 0; i < td_->cfg.epWorldSize; ++i) {
            winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer_[i];
        }

        dstSlotIdxGm_.SetGlobalBuffer((__gm__ int32_t*)dstBufferSlotIdx,
                                        td_->cfg.numTokens * td_->cfg.topK);
        recvSrcMetaInternalGm_.SetGlobalBuffer((__gm__ int32_t*)recvSrcMetadataInternal,
                                                 td_->cfg.epWorldSize * td_->cfg.numMaxTokensPerRank * 3);
        numRecvPerRankHpGm_.SetGlobalBuffer((__gm__ int32_t*)numRecvPerRankHp,
                                               td_->cfg.epWorldSize);
        numRecvPerExpertHpGm_.SetGlobalBuffer((__gm__ int64_t*)numRecvPerExpertHp,
                                                td_->numLocalExperts);
        recvXGm_.SetGlobalBuffer((__gm__ bfloat16_t*)recvX,
                                   td_->aAlloc * td_->cfg.hidden);
        recvTopkIdxGm_.SetGlobalBuffer((__gm__ int32_t*)recvTopkIdx,
                                         td_->aAlloc * td_->cfg.topK);
        recvTopkWeightsGm_.SetGlobalBuffer((__gm__ float*)recvTopkWeights,
                                             td_->aAlloc * td_->cfg.topK);
        recvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t*)recvSrcMetadata,
                                             td_->aAlloc * 3);
        numRecvPerRankGm_.SetGlobalBuffer((__gm__ int32_t*)numRecvPerRank,
                                            td_->cfg.epWorldSize);
        numRecvPerExpertGm_.SetGlobalBuffer((__gm__ int64_t*)numRecvPerExpert,
                                              td_->numLocalExperts);

        uint32_t ubTokenBytes  = ALIGN_UP(td_->cfg.hidden * sizeof(bfloat16_t), 32);
        uint32_t ubMetaBytes   = 256;
        uint32_t ubPfxBytes    = (td_->cfg.epWorldSize + 1) * sizeof(int32_t);

        tpipe_->InitBuffer(ubTokenBuf_[0], ubTokenBytes);
        tpipe_->InitBuffer(ubTokenBuf_[1], ubTokenBytes);
        tpipe_->InitBuffer(ubMetaBuf_,     ubMetaBytes);
        tpipe_->InitBuffer(ubPrefixSumBuf_, ubPfxBytes);

        ubToken_[0]  = ubTokenBuf_[0].Get<bfloat16_t>();
        ubToken_[1]  = ubTokenBuf_[1].Get<bfloat16_t>();
        ubMeta_      = ubMetaBuf_.Get<int32_t>();
        ubPrefixSum_ = ubPrefixSumBuf_.Get<int32_t>();
    }

    __aicore__ inline void Process()
    {
        return;
    }

private:
    __aicore__ inline void ComputePrefixSum() {}
    __aicore__ inline void CopyFromWindow() {}
    __aicore__ inline void CopyHostPinnedToDevice() {}

    TPipe* tpipe_{nullptr};
    const MoeEpDispatchEpilogueInfo* td_{nullptr};
    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    uint32_t rankId_{0};

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