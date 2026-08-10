#ifndef HCOMM_KERNEL_H
#define HCOMM_KERNEL_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#endif
#include "kernel_operator.h"
#include "urma_p2p_bench_tiling.h"
#include "aiv_urma_channel_decode.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/hcomm/hcomm.h"

namespace UrmaP2pBench {

constexpr uint32_t HCOMM_INIT_BUF_SIZE = 512U;

class HcommBenchOp {
public:
    __aicore__ inline void Init(GM_ADDR channelPtrTable, GM_ADDR stateGm,
                                GM_ADDR timingGm, AscendC::TPipe* pipe,
                                const UrmaP2pBenchTiling* tiling)
    {
        pipe_ = pipe;
        stateGm_ = stateGm;
        timingGm_ = timingGm;
        tiling_ = tiling;
        aivId_ = AscendC::GetBlockIdx();
        channelPtrTable_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(channelPtrTable));
        pipe_->InitBuffer(decodeScratch_, CommWithUrmaImpl::DECODE_SCRATCH_SIZE);
        pipe_->InitBuffer(hcommBuf_, HCOMM_INIT_BUF_SIZE);
    }

    __aicore__ inline void Process()
    {
        if (aivId_ != 0) return;

        const uint32_t channelSlot = tiling_->channelSlot;
        const uint32_t numSqe      = tiling_->numSqe;
        const uint64_t sqeBytes    = tiling_->sqeBytes;
        const uint64_t bufBytes    = tiling_->bufBytes;

        uint64_t channelEntityVA = channelPtrTable_.GetValue(channelSlot);
        if (channelEntityVA == 0) return;

        DecodedChannel ch;
        AscendC::LocalTensor<uint8_t> scratch = decodeScratch_.Get<uint8_t>();
        DecodePeerChannel(channelEntityVA, scratch, ch);

        uint32_t n = numSqe;
        if (n < 1U) n = 1U;

        uint64_t numSlots = (sqeBytes > 0U) ? (bufBytes / sqeBytes) : 1U;
        if (numSlots < 1U) numSlots = 1U;

        AscendC::LocalTensor<uint8_t> hcommBuf = hcommBuf_.Get<uint8_t>();
        AscendC::Hcomm<AscendC::COMM_PROTOCOL_UBC_CTP> hcomm;
        hcomm.Init(hcommBuf, HCOMM_INIT_BUF_SIZE);

        const uint32_t S = static_cast<uint32_t>(sqeBytes);

        int64_t timing[TIMING_SLOTS] = {0};
        timing[0] = AscendC::GetSystemCycle();

        for (uint32_t i = 0; i < n; ++i) {
            uint64_t off = (static_cast<uint64_t>(i) % numSlots) * sqeBytes;
            hcomm.WriteNbi(channelEntityVA,
                           ch.remoteBufAddr + off,
                           ch.localBufAddr + off,
                           S);
        }

        timing[1] = AscendC::GetSystemCycle();

        hcomm.Drain(channelEntityVA);

        timing[2] = AscendC::GetSystemCycle();
        timing[3] = AscendC::GetSystemCycle();

        WriteTiming(timing);
    }

private:
    __aicore__ inline void WriteTiming(int64_t* timing)
    {
        AscendC::GlobalTensor<int64_t> timingGm;
        timingGm.SetGlobalBuffer((__gm__ int64_t*)timingGm_);
        AscendC::LocalTensor<int64_t> timingLocal;
        timingLocal.SetLocalBuffer((__local__ int64_t*)timing, TIMING_SLOTS * sizeof(int64_t));
        AscendC::DataCopy(timingGm, timingLocal, TIMING_SLOTS);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }

    AscendC::TPipe* pipe_{nullptr};
    GM_ADDR stateGm_{nullptr};
    GM_ADDR timingGm_{nullptr};
    const UrmaP2pBenchTiling* tiling_{nullptr};
    uint32_t aivId_{0};
    AscendC::GlobalTensor<uint64_t> channelPtrTable_;
    AscendC::TBuf<> decodeScratch_;
    AscendC::TBuf<> hcommBuf_;
};

}  // namespace UrmaP2pBench

#endif  // HCOMM_KERNEL_H
