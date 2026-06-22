#ifndef ENGRAM_FETCH_WAIT_ARCH35_H
#define ENGRAM_FETCH_WAIT_ARCH35_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "../engram_fetch_wait_tiling_data.h"
#include "../engram_fetch_wait.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/hcomm/hcomm.h"

namespace EngramFetchWaitImpl {

using namespace AscendC;
using namespace Mc2Aclnn;

class EngramFetchWaitArch35 {
public:
    __aicore__ inline EngramFetchWaitArch35() = default;

    __aicore__ inline void Init(GM_ADDR bufferHandle, GM_ADDR workspaceGM, TPipe *pipe,
                                const EngramFetchWaitTilingData *tilingData);

    __aicore__ inline void Process();

private:
    TPipe *tpipe_{nullptr};
    __gm__ EngramBufferHandle *bhPtr_{nullptr};

    uint32_t aivId_{0};
    uint32_t rankId_{0};
    uint32_t numRanks_{0};

    TBuf<> hcommBuf_;
    LocalTensor<uint8_t> hcommTensor_;

    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
};

__aicore__ inline void EngramFetchWaitArch35::Init(GM_ADDR bufferHandle, GM_ADDR workspaceGM, TPipe *pipe,
                                                   const EngramFetchWaitTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();

    bhPtr_ = (__gm__ EngramBufferHandle *)bufferHandle;
    rankId_ = bhPtr_->rankId;
    numRanks_ = bhPtr_->rankSize;

    tpipe_->InitBuffer(hcommBuf_, EngramFetchWaitImpl::HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, EngramFetchWaitImpl::HCOMM_INIT_SIZE);
}

__aicore__ inline void EngramFetchWaitArch35::Process()
{
    if ASCEND_IS_AIV {
        if (aivId_ >= numRanks_)
            return;
        if (aivId_ == rankId_)
            return;
        uint32_t channelIdx = aivId_;
        hcomm_.Drain(bhPtr_->hcommHandle_[channelIdx]);
    }
}

} // namespace EngramFetchWaitImpl

#endif
