#ifndef SQE_WRITE_KERNEL_H
#define SQE_WRITE_KERNEL_H

// ============================================================================
// SQE Write Benchmark Kernel (64B Write SQE, opcode=0x3)
//
// 流程: decode channel → 生成SQE模板 → [warmup+loop循环] {
//   重置CQ ring → 组装n个数据SQE写入ring+doorbell → 发1个completion SQE →
//   poll CQ等待完成 → 记录timing(仅正式轮)
// }
// ============================================================================

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
        // UB缓冲区: decode scratch(640B) + SQE(64B) + CQE(64B) + timing(64B)
        pipe_->InitBuffer(decodeScratch_, DECODE_SCRATCH_SIZE);
        pipe_->InitBuffer(sqeBuf_, WRITE_SQE_SIZE);
        pipe_->InitBuffer(cqeBuf_, WRITE_CQE_SIZE);
        pipe_->InitBuffer(timingBuf_, TIMING_SLOTS * sizeof(int64_t));
    }

    __aicore__ inline void Process()
    {
        // 仅0核执行，单核驱动单通道P2P发送
        if (aivId_ != 0) return;

        const uint32_t channelSlot = tiling_->channelSlot;
        const uint32_t stateStride = tiling_->stateStride;
        const uint32_t numSqe      = tiling_->numSqe;
        const uint64_t sqeBytes    = tiling_->sqeBytes;
        const uint64_t bufBytes    = tiling_->bufBytes;
        const uint32_t dbInterval  = (tiling_->dbInterval >= 1U) ? tiling_->dbInterval : 1U;
        const uint32_t warmupCount = tiling_->warmupCount;
        const uint32_t loopCount   = (tiling_->loopCount > 0U) ? tiling_->loopCount : 1U;

        // 从channelPtrTable读取ChannelEntity的device VA
        uint64_t channelEntityVA = channelPtrTable_.GetValue(channelSlot);
        if (channelEntityVA == 0) return;

        // 解码ChannelEntity: 提取SQ/CQ/buffer/token等通信字段
        DecodedChannel ch;
        AscendC::LocalTensor<uint8_t> scratch = decodeScratch_.Get<uint8_t>();
        DecodePeerChannel(channelEntityVA, scratch, ch);

        // 共享CQ场景: 当前channel没有CQ时, 从CQ owner channel借用
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

        // SQE数量上限: 不超过SQ ring深度(留2个slot余量)
        uint32_t maxN = (ch.sqDepth >= 3U) ? (ch.sqDepth - 2U) : 1U;
        uint32_t n = numSqe;
        if (n > maxN) n = maxN;
        if (n < 1U)   n = 1U;

        // workspace内可放下的SQE槽位数(用于地址回绕)
        uint64_t numSlots = (sqeBytes > 0U) ? (bufBytes / sqeBytes) : 1U;
        if (numSlots < 1U) numSlots = 1U;

        AscendC::GlobalTensor<uint32_t> stateGm;
        stateGm.SetGlobalBuffer((__gm__ uint32_t*)stateGm_);

        // 生成SQE模板: 填固定字段(opcode=0x3, sge_num=1, owner, token_en等)
        // 同一peer的所有SQE共用此模板, 循环内只填localAddr/remoteAddr/length
        AscendC::LocalTensor<uint8_t> sqe = sqeBuf_.Get<uint8_t>();
        GenerateCommWriteSQE(sqe);
        FillCommWriteSQEFromChannel(sqe, ch.tpID,
                                    ch.remoteEID0, ch.remoteEID1,
                                    ch.remoteEID2, ch.remoteEID3,
                                    ch.remoteTokenId, ch.remoteTokenValue);

        const uint32_t S = static_cast<uint32_t>(sqeBytes);
        AscendC::LocalTensor<int64_t> timingLocal = timingBuf_.Get<int64_t>();
        AscendC::LocalTensor<uint8_t> cqe = cqeBuf_.Get<uint8_t>();

        // ========== 正式执行循环 (kernel内部, 无Host同步开销) ==========
        // 无预热, 每轮都记录timing, 最后一轮写GM供Host读取
        for (uint32_t iter = 0; iter < loopCount; ++iter) {
            uint32_t sqPiLinear = 0;
            uint32_t sqPi = 0;
            uint32_t ownerCqCiLinear = 0;

            // 仅首轮初始化CQ ring: 后续轮次CqeSlotEmpty消费完会自动写回0xff
            if (iter == 0) {
                InvalidateCqRing(ch.cqVa, ch.cqeSize, ch.cqDepth,
                                 scratch, DECODE_SCRATCH_SIZE / WRITE_CQE_SIZE);
            }

            timingLocal.SetValue(0, AscendC::GetSystemCycle());  // timing[0]: 起始

            // ---- 组装并发送n个数据SQE (cqe=0, 不产生完成事件) ----
            for (uint32_t i = 0; i < n; ++i) {
                uint64_t off = (static_cast<uint64_t>(i) % numSlots) * sqeBytes;
                SetCommWriteSQE(sqe, ch.localBufAddr + off, ch.remoteBufAddr + off, S, ch.localTokenId, 0);
                SyncFunc<AscendC::HardEvent::S_MTE3>();       // 等Scalar写完SQE字段再DataCopy
                PutSQEToRing(ch.sqVa, ch.wqeSize, ch.sqDepth, sqe, sqPi, sqPiLinear);  // DataCopy SQE→SQ ring
                SyncFunc<AscendC::HardEvent::MTE3_S>();       // 等DataCopy完成再继续
                // 每 dbInterval 个SQE敲一次DoorBell通知硬件
                if ((i + 1U) % dbInterval == 0U) {
                    RingJFSDoorbell(ch.sqDbVa, sqPiLinear);
                }
            }

            timingLocal.SetValue(1, AscendC::GetSystemCycle());  // timing[1]: n个数据SQE组装+发送完毕
            // SQ有序: 此SQE的CQE落地 = 之前所有数据SQE已drain
            SetCommWriteSQE(sqe, ch.localBufAddr, ch.remoteBufAddr,
                            COMPLETION_SQE_BYTES, ch.localTokenId, 1);
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            PutSQEToRing(ch.sqVa, ch.wqeSize, ch.sqDepth, sqe, sqPi, sqPiLinear);
            SyncFunc<AscendC::HardEvent::MTE3_S>();
            RingJFSDoorbell(ch.sqDbVa, sqPiLinear);

            timingLocal.SetValue(2, AscendC::GetSystemCycle());  // timing[2]: completion SQE+doorbell完毕

            // ---- Poll CQ: spin等待completion事件落地 ----
            uint32_t cqCi = ownerCqCiLinear % ch.cqDepth;
            while (CqeSlotEmpty(ch.cqVa, ch.cqeSize, cqCi, cqe)) {}
            ++ownerCqCiLinear;

            timingLocal.SetValue(3, AscendC::GetSystemCycle());  // timing[3]: CQ poll完成

            // 最后一轮: 把timing从UB拷到GM, 供Host读取
            if (iter == loopCount - 1U) {
                AscendC::GlobalTensor<int64_t> timingGm;
                timingGm.SetGlobalBuffer((__gm__ int64_t*)timingGm_);
                AscendC::DataCopy(timingGm, timingLocal, TIMING_SLOTS);
                SyncFunc<AscendC::HardEvent::MTE3_S>();
            }
        }
    }

private:
    AscendC::TPipe* pipe_{nullptr};
    GM_ADDR stateGm_{nullptr};
    GM_ADDR timingGm_{nullptr};
    const UrmaP2pBenchTiling* tiling_{nullptr};
    uint32_t aivId_{0};
    AscendC::GlobalTensor<uint64_t> channelPtrTable_;
    AscendC::TBuf<> decodeScratch_;   // ChannelEntity解码scratch (640B)
    AscendC::TBuf<> sqeBuf_;          // SQE组装buffer (64B)
    AscendC::TBuf<> cqeBuf_;          // CQE读取buffer (64B)
    AscendC::TBuf<> timingBuf_;       // timing打点buffer (8×int64)
};

}  // namespace UrmaP2pBench

#endif  // SQE_WRITE_KERNEL_H
