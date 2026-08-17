#ifndef HCOMM_KERNEL_H
#define HCOMM_KERNEL_H

// ============================================================================
// Hcomm Benchmark Kernel (WriteNbi + Commit + Drain 高层API)
//
// 与 mc2_native 对齐的两种 benchMode:
//   benchMode=0 (batch_write):
//     loopCount × WriteNbi<commit=false> → Commit → Drain
//     攒一批 WQE (不敲 DoorBell), 最后统一 Commit + Drain
//   benchMode=1 (write_drain_loop):
//     循环 loopCount 次 { WriteNbi<commit=true> → Drain }
//     每次 Write 后立即 Drain 等完成
//
// 与 mc2_native 的区别:
//   - 不手动组装 SQE/敲 DoorBell/poll CQ, Hcomm 内部封装全部底层操作
//   - commit=false 时 WQE 只写 ring 不敲 DoorBell, Commit() 统一敲
// ============================================================================

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#endif
#include "kernel_operator.h"
#include "../common/urma_p2p_bench_tiling.h"
#include "../sqe_write/aiv_urma_sqe.h"
#include "../common/aiv_urma_channel_decode.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/hcomm/hcomm.h"

namespace UrmaP2pBench {

using namespace CommWithUrmaImpl;

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
        pipe_->InitBuffer(decodeScratch_, DECODE_SCRATCH_SIZE);
        pipe_->InitBuffer(hcommBuf_, HCOMM_INIT_BUF_SIZE);
        pipe_->InitBuffer(timingBuf_, TIMING_SLOTS * sizeof(int64_t));
    }

    __aicore__ inline void Process()
    {
        if (aivId_ != 0) return;

        const uint32_t channelSlot = tiling_->channelSlot;
        const uint64_t sqeBytes    = tiling_->sqeBytes;
        const uint64_t bufBytes    = tiling_->bufBytes;
        const uint32_t loopCount   = (tiling_->loopCount > 0U) ? tiling_->loopCount : 1U;
        const uint32_t benchMode   = tiling_->benchMode;

        uint64_t channelEntityVA = channelPtrTable_.GetValue(channelSlot);
        if (channelEntityVA == 0) return;
        AscendC::ChannelHandle channelHandle = channelEntityVA;

        DecodedChannel ch;
        AscendC::LocalTensor<uint8_t> scratch = decodeScratch_.Get<uint8_t>();
        DecodePeerChannel(channelEntityVA, scratch, ch);

        uint64_t numSlots = (sqeBytes > 0U) ? (bufBytes / sqeBytes) : 1U;
        if (numSlots < 1U) numSlots = 1U;

        AscendC::LocalTensor<uint8_t> hcommBuf = hcommBuf_.Get<uint8_t>();
        AscendC::Hcomm<AscendC::COMM_PROTOCOL_UBC_CTP> hcomm;
        hcomm.Init(hcommBuf, HCOMM_INIT_BUF_SIZE);

        const uint32_t sqeDataBytes = static_cast<uint32_t>(sqeBytes);
        AscendC::LocalTensor<int64_t> timingLocal = timingBuf_.Get<int64_t>();

        if (benchMode == BENCH_MODE_BATCH_WRITE) {
            // ========== benchMode=0: loopCount × WriteNbi<commit=false> → Commit → Drain ==========
            int64_t t0 = AscendC::GetSystemCycle();
            for (uint32_t i = 0; i < loopCount; ++i) {
                uint64_t off = (static_cast<uint64_t>(i) % numSlots) * sqeBytes;
                hcomm.WriteNbi<false>(channelHandle,
                                      (GM_ADDR)(ch.remoteBufAddr + off),
                                      (GM_ADDR)(ch.localBufAddr + off),
                                      sqeDataBytes);
            }
            int64_t t1 = AscendC::GetSystemCycle();
            hcomm.Commit(channelHandle);
            int64_t t2 = AscendC::GetSystemCycle();
            hcomm.Drain(channelHandle);
            int64_t t3 = AscendC::GetSystemCycle();

            timingLocal.SetValue(0, t1 - t0);  // assemble: 全部 WriteNbi
            timingLocal.SetValue(1, t2 - t1);  // commit: Commit (敲一次 DoorBell)
            timingLocal.SetValue(2, t3 - t2);  // drain: Drain (poll CQ 等完成)
            timingLocal.SetValue(3, t3 - t0);  // total
            timingLocal.SetValue(4, t1 - t0);  // last assemble
            timingLocal.SetValue(5, t2 - t1);  // last commit
            timingLocal.SetValue(6, t3 - t2);  // last drain
            timingLocal.SetValue(7, t3 - t0);  // last total
        } else {
            // ========== benchMode=1: 循环 loopCount 次 { WriteNbi<commit=true> → Drain } ==========
            int64_t sumWrite = 0, sumDrain = 0, sumTotal = 0;
            int64_t lastWrite = 0, lastDrain = 0, lastTotal = 0;

            for (uint32_t i = 0; i < loopCount; ++i) {
                uint64_t off = (static_cast<uint64_t>(i) % numSlots) * sqeBytes;

                int64_t t0 = AscendC::GetSystemCycle();
                hcomm.WriteNbi<true>(channelHandle,
                                     (GM_ADDR)(ch.remoteBufAddr + off),
                                     (GM_ADDR)(ch.localBufAddr + off),
                                     sqeDataBytes);
                int64_t t1 = AscendC::GetSystemCycle();
                hcomm.Drain(channelHandle);
                int64_t t2 = AscendC::GetSystemCycle();

                lastWrite  = t1 - t0;
                lastDrain  = t2 - t1;
                lastTotal  = t2 - t0;
                sumWrite  += lastWrite;
                sumDrain  += lastDrain;
                sumTotal  += lastTotal;
            }

            timingLocal.SetValue(0, sumWrite);
            timingLocal.SetValue(1, sumWrite);
            timingLocal.SetValue(2, sumDrain);
            timingLocal.SetValue(3, sumTotal);
            timingLocal.SetValue(4, lastWrite);
            timingLocal.SetValue(5, lastWrite);
            timingLocal.SetValue(6, lastDrain);
            timingLocal.SetValue(7, lastTotal);
        }

        AscendC::GlobalTensor<int64_t> timingGm;
        timingGm.SetGlobalBuffer((__gm__ int64_t*)timingGm_);
        AscendC::DataCopy(timingGm, timingLocal, TIMING_SLOTS);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }

private:
    AscendC::TPipe* pipe_{nullptr};
    GM_ADDR stateGm_{nullptr};
    GM_ADDR timingGm_{nullptr};
    const UrmaP2pBenchTiling* tiling_{nullptr};
    uint32_t aivId_{0};
    AscendC::GlobalTensor<uint64_t> channelPtrTable_;
    AscendC::TBuf<> decodeScratch_;
    AscendC::TBuf<> hcommBuf_;
    AscendC::TBuf<> timingBuf_;
};

}  // namespace UrmaP2pBench

#endif  // HCOMM_KERNEL_H
