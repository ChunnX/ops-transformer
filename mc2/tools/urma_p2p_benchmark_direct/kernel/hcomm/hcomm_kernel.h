#ifndef HCOMM_KERNEL_H
#define HCOMM_KERNEL_H

// ============================================================================
// Hcomm Benchmark Kernel (WriteNbi + Drain 高层API)
//
// 与SQE模式的区别:
//   - 不手动组装SQE/敲DoorBell/poll CQ, Hcomm内部封装全部底层操作
//   - WriteNbi: 非阻塞写入(内部自动组装SQE+写ring+敲doorbell)
//   - Drain: 等待完成(内部自动poll CQ)
//   - 通信流程更简洁, 但无法拆分assemble/issue/wait细节
//
// 流程: decode channel → Hcomm.Init → [loop循环] {
//   n×WriteNbi → Drain → 记录timing(仅正式轮)
// }
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
        // UB缓冲区: decode scratch(640B) + Hcomm init(512B) + timing(64B)
        pipe_->InitBuffer(decodeScratch_, DECODE_SCRATCH_SIZE);
        pipe_->InitBuffer(hcommBuf_, HCOMM_INIT_BUF_SIZE);
        pipe_->InitBuffer(timingBuf_, TIMING_SLOTS * sizeof(int64_t));
    }

    __aicore__ inline void Process()
    {
        // 仅0核执行，单核驱动单通道P2P发送
        if (aivId_ != 0) return;

        const uint32_t channelSlot = tiling_->channelSlot;
        const uint32_t numSqe      = tiling_->numSqe;
        const uint64_t sqeBytes    = tiling_->sqeBytes;
        const uint64_t bufBytes    = tiling_->bufBytes;
        const uint32_t loopCount   = (tiling_->loopCount > 0U) ? tiling_->loopCount : 1U;

        // 从channelPtrTable读取ChannelEntity的device VA
        uint64_t channelEntityVA = channelPtrTable_.GetValue(channelSlot);
        if (channelEntityVA == 0) return;
        AscendC::ChannelHandle channelHandle = channelEntityVA;

        // 解码ChannelEntity: 提取localBufAddr/remoteBufAddr(数据搬运源/目的地址)
        DecodedChannel ch;
        AscendC::LocalTensor<uint8_t> scratch = decodeScratch_.Get<uint8_t>();
        DecodePeerChannel(channelEntityVA, scratch, ch);

        uint32_t n = numSqe;
        if (n < 1U) n = 1U;

        // workspace内可放下的SQE槽位数(用于地址回绕)
        uint64_t numSlots = (sqeBytes > 0U) ? (bufBytes / sqeBytes) : 1U;
        if (numSlots < 1U) numSlots = 1U;

        // 初始化Hcomm实例(UBC_CTP协议)
        AscendC::LocalTensor<uint8_t> hcommBuf = hcommBuf_.Get<uint8_t>();
        AscendC::Hcomm<AscendC::COMM_PROTOCOL_UBC_CTP> hcomm;
        hcomm.Init(hcommBuf, HCOMM_INIT_BUF_SIZE);

        const uint32_t S = static_cast<uint32_t>(sqeBytes);
        AscendC::LocalTensor<int64_t> timingLocal = timingBuf_.Get<int64_t>();

        // ========== 正式执行循环 (kernel内部, 无Host同步开销) ==========
        // 累加所有轮的timing, 最后一轮写GM供Host读取, Host侧除以loopCount得平均值
        // Hcomm模式无需手动初始化CQ: Drain内部自动管理SQ/CQ状态
        int64_t sumAssemble = 0;
        int64_t sumIssue = 0;
        int64_t sumWait = 0;
        int64_t sumTotal = 0;

        for (uint32_t iter = 0; iter < loopCount; ++iter) {
            int64_t t0 = AscendC::GetSystemCycle();  // 起始

            // ---- n个WriteNbi: 非阻塞写入(内部自动SQE组装+ring写入+doorbell) ----
            for (uint32_t i = 0; i < n; ++i) {
                uint64_t off = (static_cast<uint64_t>(i) % numSlots) * sqeBytes;
                hcomm.WriteNbi(channelHandle,
                               (GM_ADDR)(ch.remoteBufAddr + off),  // dst: 远端workspace
                               (GM_ADDR)(ch.localBufAddr + off),   // src: 本地workspace
                               S);
            }

            int64_t t1 = AscendC::GetSystemCycle();  // n×WriteNbi完毕

            // ---- Drain: 等待所有WriteNbi完成(内部poll CQ) ----
            hcomm.Drain(channelHandle);

            int64_t t2 = AscendC::GetSystemCycle();  // Drain完毕
            int64_t t3 = t2;  // Hcomm: wait≈0, Drain内部已等待

            // 累加本轮timing
            sumAssemble += (t1 - t0);
            sumIssue    += (t2 - t0);
            sumWait     += (t3 - t2);
            sumTotal    += (t3 - t0);

            // 最后一轮: 把累计timing和最后一轮timing都写入GM
            if (iter == loopCount - 1U) {
                timingLocal.SetValue(0, sumAssemble);             // 累计assemble
                timingLocal.SetValue(1, sumIssue);                // 累计issue
                timingLocal.SetValue(2, sumWait);                 // 累计wait
                timingLocal.SetValue(3, sumTotal);                // 累计total
                timingLocal.SetValue(4, t1 - t0);                 // 最后一轮assemble
                timingLocal.SetValue(5, t2 - t0);                 // 最后一轮issue
                timingLocal.SetValue(6, t3 - t2);                 // 最后一轮wait
                timingLocal.SetValue(7, t3 - t0);                 // 最后一轮total
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
    AscendC::TBuf<> hcommBuf_;        // Hcomm初始化buffer (512B)
    AscendC::TBuf<> timingBuf_;       // timing打点buffer (8×int64)
};

}  // namespace UrmaP2pBench

#endif  // HCOMM_KERNEL_H
