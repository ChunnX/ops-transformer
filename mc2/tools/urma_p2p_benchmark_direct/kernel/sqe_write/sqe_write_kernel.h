#ifndef SQE_WRITE_KERNEL_H
#define SQE_WRITE_KERNEL_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#endif
#include "kernel_operator.h"
#include "../common/urma_p2p_bench_tiling.h"
#include "aiv_urma_sqe.h"
#include "../common/aiv_urma_channel_decode.h"

namespace UrmaP2pBench {

using namespace CommWithUrmaImpl;

class SqeWriteBenchOp {
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
        pipe_->InitBuffer(sqeBuf_, WRITE_SQE_SIZE);
        pipe_->InitBuffer(cqeBuf_, WRITE_CQE_SIZE);
        pipe_->InitBuffer(timingBuf_, TIMING_SLOTS * sizeof(int64_t));
    }

    __aicore__ inline void Process()
    {
        if (aivId_ != 0) return;

        const uint32_t channelSlot = tiling_->channelSlot;
        const uint32_t stateStride = tiling_->stateStride;
        const uint32_t numSqe      = tiling_->numSqe;
        const uint64_t sqeBytes    = tiling_->sqeBytes;
        const uint64_t bufBytes    = tiling_->bufBytes;
        const uint32_t dbInterval  = (tiling_->dbInterval >= 1U) ? tiling_->dbInterval : 1U;

        uint64_t channelEntityVA = channelPtrTable_.GetValue(channelSlot);
        if (channelEntityVA == 0) return;

        DecodedChannel ch;
        AscendC::LocalTensor<uint8_t> scratch = decodeScratch_.Get<uint8_t>();
        DecodePeerChannel(channelEntityVA, scratch, ch);

        if (ch.cqVa == 0 && tiling_->cqOwnerSlot != channelSlot) {
            uint64_t ownerVA = channelPtrTable_.GetValue(tiling_->cqOwnerSlot);
            if (ownerVA != 0) {
                DecodedChannel owner;
                DecodePeerChannel(ownerVA, scratch, owner);
                ch.cqVa    = owner.cqVa;
                ch.cqDepth = owner.cqDepth;
                ch.cqeSize = owner.cqeSize;
            }
        }

        uint32_t maxN = (ch.sqDepth >= 3U) ? (ch.sqDepth - 2U) : 1U;
        uint32_t n = numSqe;
        if (n > maxN) n = maxN;
        if (n < 1U)   n = 1U;

        uint64_t numSlots = (sqeBytes > 0U) ? (bufBytes / sqeBytes) : 1U;
        if (numSlots < 1U) numSlots = 1U;

        AscendC::GlobalTensor<uint32_t> stateGm;
        stateGm.SetGlobalBuffer((__gm__ uint32_t*)stateGm_);
        uint32_t sqPiLinear = stateGm.GetValue(channelSlot);
        uint32_t sqPi = sqPiLinear % ch.sqDepth;

        uint32_t ownerCqCiLinear = stateGm.GetValue(stateStride + tiling_->cqOwnerSlot);
        if (ownerCqCiLinear == 0) {
            InvalidateCqRing(ch.cqVa, ch.cqeSize, ch.cqDepth,
                             scratch, DECODE_SCRATCH_SIZE / WRITE_CQE_SIZE);
        }

        AscendC::LocalTensor<uint8_t> sqe = sqeBuf_.Get<uint8_t>();
        GenerateCommWriteSQE(sqe);
        FillCommWriteSQEFromChannel(sqe, ch.tpID,
                                    ch.remoteEID0, ch.remoteEID1,
                                    ch.remoteEID2, ch.remoteEID3,
                                    ch.remoteTokenId, ch.remoteTokenValue);

        const uint32_t S = static_cast<uint32_t>(sqeBytes);
        AscendC::LocalTensor<int64_t> timingLocal = timingBuf_.Get<int64_t>();

        timingLocal.SetValue(0, AscendC::GetSystemCycle());

        for (uint32_t i = 0; i < n; ++i) {
            uint64_t off = (static_cast<uint64_t>(i) % numSlots) * sqeBytes;
            SetCommWriteSQE(sqe, ch.localBufAddr + off, ch.remoteBufAddr + off, S, ch.localTokenId, 0);
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            PutSQEToRing(ch.sqVa, ch.wqeSize, ch.sqDepth, sqe, sqPi, sqPiLinear);
            SyncFunc<AscendC::HardEvent::MTE3_S>();
            if ((i + 1U) % dbInterval == 0U) {
                RingJFSDoorbell(ch.sqDbVa, sqPiLinear);
            }
        }

        timingLocal.SetValue(1, AscendC::GetSystemCycle());

        SetCommWriteSQE(sqe, ch.localBufAddr, ch.remoteBufAddr,
                        COMPLETION_SQE_BYTES, ch.localTokenId, 1);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        PutSQEToRing(ch.sqVa, ch.wqeSize, ch.sqDepth, sqe, sqPi, sqPiLinear);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        RingJFSDoorbell(ch.sqDbVa, sqPiLinear);

        timingLocal.SetValue(2, AscendC::GetSystemCycle());

        AscendC::LocalTensor<uint8_t> cqe = cqeBuf_.Get<uint8_t>();
        uint32_t cqCi = ownerCqCiLinear % ch.cqDepth;
        while (CqeSlotEmpty(ch.cqVa, ch.cqeSize, cqCi, cqe)) {}
        ++ownerCqCiLinear;

        timingLocal.SetValue(3, AscendC::GetSystemCycle());

        stateGm.SetValue(channelSlot, sqPiLinear);
        stateGm.SetValue(stateStride + tiling_->cqOwnerSlot, ownerCqCiLinear);

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
    AscendC::TBuf<> sqeBuf_;
    AscendC::TBuf<> cqeBuf_;
    AscendC::TBuf<> timingBuf_;
};

}  // namespace UrmaP2pBench

#endif  // SQE_WRITE_KERNEL_H
