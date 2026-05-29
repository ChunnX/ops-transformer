#ifndef MOE_EP_DISPATCH_H
#define MOE_EP_DISPATCH_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "moe_ep_dispatch_tiling_key.h"

#if __has_include("../common/op_kernel/moe_distribute_base.h")
#include "../common/op_kernel/moe_distribute_base.h"
#include "../common/op_kernel/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

#include "moe_ep_dispatch_tiling_data.h"

namespace MoeEpDispatchImpl {

using namespace AscendC;

static constexpr uint32_t MAX_TOPK = 8;
static constexpr uint32_t WIN_ADDR_ALIGN = 512;

template <uint32_t DO_CPU_SYNC, uint32_t IS_CACHED, uint32_t NETWORK_MODE>
class KernelMoeEpDispatch {
public:
    __aicore__ inline KernelMoeEpDispatch() {};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights,
        GM_ADDR cachedNumPerExpert, GM_ADDR cachedSlotIdx, GM_ADDR cachedSrcMeta,
        GM_ADDR numRecvPerRankHp, GM_ADDR numRecvPerExpertHp,
        GM_ADDR dstBufferSlotIdx, GM_ADDR recvSrcMetadataInternal,
        GM_ADDR workspace, GM_ADDR tilingGM,
        TPipe* pipe, const MoeEpDispatchInfo* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ResetCounters();
    __aicore__ inline void SendPhaseFirstTime();
    __aicore__ inline void SendPhaseCached();
    __aicore__ inline void DedupAndSendDirect(uint32_t t);
    __aicore__ inline void RecvPhase();
    __aicore__ inline void WriteActualAToHostPinned();
    __aicore__ inline uint32_t ComputePerAivTokenStart();
    __aicore__ inline uint32_t ComputePerAivTokenEnd();

    TPipe* tpipe_{nullptr};
    const MoeEpDispatchInfo* td_{nullptr};
    __gm__ Mc2Kernel::HcclOpParam* winContext_{nullptr};

    GlobalTensor<bfloat16_t> xGm_;
    GlobalTensor<int32_t> topkIdxGm_;
    GlobalTensor<float> topkWeightsGm_;
    GlobalTensor<int32_t> dstSlotIdxGm_;
    GlobalTensor<int32_t> recvSrcMetaInternalGm_;
    GlobalTensor<int32_t> numRecvPerRankHpGm_;
    GlobalTensor<int64_t> numRecvPerExpertHpGm_;
    GlobalTensor<int32_t> cachedSlotIdxGm_;

    LocalTensor<bfloat16_t> ubXIn_;
    LocalTensor<bfloat16_t> ubSlot_[2];
    LocalTensor<int32_t> ubMeta_;
    LocalTensor<float> ubAccFp32_;

    TBuf<QuePosition::VECIN> ubXBuf_;
    TBuf<QuePosition::VECIN> ubSlotBuf_[2];
    TBuf<QuePosition::VECIN> ubMetaBuf_;

    GM_ADDR workspace_{nullptr};
    uint32_t aivId_{0};
    uint32_t actualA_{0};
    uint32_t blockIdx_{0};
};

}  // namespace MoeEpDispatchImpl

#endif  // MOE_EP_DISPATCH_H