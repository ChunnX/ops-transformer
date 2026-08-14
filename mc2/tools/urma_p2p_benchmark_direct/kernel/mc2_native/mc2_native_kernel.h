#ifndef MC2_NATIVE_KERNEL_H
#define MC2_NATIVE_KERNEL_H

// ============================================================================
// mc2_native kernel — 使用 moe_distribute_base_native.h 的完整接口实现 P2P 通信
//
// 与 mc2 版 P2pUrmaSqeComm 的逻辑完全一致:
//   InitRankComm: GetURMASqInfoTensor → GetIsFirstInComm → GetPICI →
//                 InvalidateCqeStatus → PollCommCQUpdateSQCI →
//                 GenerateCommWriteWithNotifySQE → UpdateCommWriteWithNotifySQE
//   Write:        SetCommWriteWithNotifySQE + PutCommNotifySQE (不敲DoorBell)
//   Drain:        SendJFSDoorBell + PollCommCQUpdateSQCI + UpdatePICI
//
// Host 侧构造 HcclCombinOpParam 等效数据(sqs/cqs/windowsIn/windowsOut/PICI)传入 kernel。
// ============================================================================

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../common/urma_p2p_bench_tiling.h"
#include "moe_distribute_base_native.h"

namespace UrmaP2pBench {

using AscendC::SyncFunc;

// mc2 版的状态区常量(与 p2p_urma_comm.h 一致)
constexpr uint64_t P2P_STATE_SIZE = 350UL * 1024UL;
constexpr uint64_t P2P_WIN_STATE_OFFSET = 350UL * 1024UL;
constexpr uint32_t P2P_UB_ALIGN = 32U;
constexpr uint32_t P2P_CQ_DEPTH_256 = 256U;
constexpr uint32_t P2P_SQ_FLUSH_THRESHOLD = 200U;
constexpr uint32_t P2P_STATE_OFFSET = 32U;
constexpr uint64_t P2P_NOTIFY_DATA_FLAG = 0x3F800000UL;

class Mc2NativeBenchOp {
public:
    __aicore__ inline void Init(GM_ADDR hcclContextGM, GM_ADDR workspace,
                                GM_ADDR timingGm, AscendC::TPipe* pipe,
                                const UrmaP2pBenchTiling* tiling)
    {
        pipe_ = pipe;
        tiling_ = tiling;
        timingGm_ = timingGm;
        aivId_ = AscendC::GetBlockIdx();

        // 通信上下文: Host 侧构造的 HcclCombinOpParam (在 GM 中)
        hcclContext_ = (__gm__ HcclCombinOpParam *)hcclContextGM;
        epRankId_ = hcclContext_->rankId;
        epWorldSize_ = hcclContext_->rankDim;
        localUrmaWorkspace_ = workspace;

        // 分配 SQE 级 UB 缓冲区 (与 P2pUrmaSqeComm::InitCommBuffer 一致)
        uint32_t sqInfoSize = AscendC::Ceil(sizeof(HcclAiRMAWQ), P2P_UB_ALIGN) * P2P_UB_ALIGN;
        pipe_->InitBuffer(sqInfoBuf_, sqInfoSize);
        sqInfoTensor_ = sqInfoBuf_.Get<uint8_t>();

        uint32_t cqInfoSize = AscendC::Ceil(sizeof(HcclAiRMACQ), P2P_UB_ALIGN) * P2P_UB_ALIGN;
        pipe_->InitBuffer(cqInfoBuf_, cqInfoSize);
        cqInfoTensor_ = cqInfoBuf_.Get<uint8_t>();

        pipe_->InitBuffer(sqeBuf_, WRITE_WITH_NOTIFY_SQE_SIZE);
        sqeTensor_ = sqeBuf_.Get<uint8_t>();

        pipe_->InitBuffer(cqeBuf_, P2P_UB_ALIGN * P2P_CQ_DEPTH_256);
        cqeTensor_ = cqeBuf_.Get<uint8_t>();

        pipe_->InitBuffer(jfsDoorBellBuf_, P2P_UB_ALIGN);
        jfsDoorBellTensor_ = jfsDoorBellBuf_.Get<uint8_t>();

        pipe_->InitBuffer(jfcDoorBellBuf_, P2P_UB_ALIGN);
        jfcDoorBellTensor_ = jfcDoorBellBuf_.Get<uint8_t>();

        pipe_->InitBuffer(timingBuf_, TIMING_SLOTS * sizeof(int64_t));
    }

    // ========================================================================
    // InitRankComm — 与 P2pUrmaSqeComm::InitRankComm 逻辑完全一致
    // ========================================================================
    __aicore__ inline void InitRankComm(int32_t dstRankId)
    {
        // 步骤1: 从 HcclCombinOpParam 中获取 SQ 信息到 UB
        GetURMASqInfoTensor(sqInfoTensor_, (GM_ADDR)hcclContext_, dstRankId);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        // 步骤2: 从 HcclCombinOpParam 中获取 CQ 信息到 UB
        GetURMACqInfoTensor(cqInfoTensor_, (GM_ADDR)hcclContext_, dstRankId);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        // 步骤3: 检查是否为首次与该 rank 通信
        bool isFirstComm = false;
        GetIsFirstInComm((GM_ADDR)hcclContext_, epRankId_, dstRankId, isFirstComm);
        isFirstComm_ = !isFirstComm;

        // 步骤4: 获取 PICI (Producer-Consumer Index)
        GetPICI((GM_ADDR)hcclContext_, epRankId_, dstRankId,
                outSqPi_, outSqCi_, outCqPi_, outCqCi_, outSqPiLinear_, outCqCiLinear_);

        // 步骤5: 首次通信时初始化 CQ ring (全填 0xff)
        if (isFirstComm_) {
            InvalidateCqeStatus(cqInfoTensor_, cqeTensor_);
            SyncFunc<AscendC::HardEvent::MTE3_S>();
            UpdateIsFirstInComm((GM_ADDR)hcclContext_, epRankId_, dstRankId, true);
        }

        // 步骤6: 轮询 CQ 清理残余完成事件, 更新 sqCi
        PollCommCQUpdateSQCI(sqInfoTensor_, cqInfoTensor_, cqeTensor_, jfcDoorBellTensor_,
                             outSqCi_, outCqCi_, outCqCiLinear_);

        // 步骤7: 生成 WriteWithNotify SQE 模板
        GenerateCommWriteWithNotifySQE(sqeTensor_, 1);

        // 步骤8: 更新 SQE 模板中的通信元数据 (tp_id, jetty_id, rmt_eid, token 等)
        UpdateCommWriteWithNotifySQE(sqeTensor_, sqInfoTensor_);

        curDstRankId_ = dstRankId;
        pendingSqCnt_ = 0;
    }

    // ========================================================================
    // Write — 提交一个 WriteWithNotify SQE (不敲 DoorBell, 不 poll CQ)
    // ========================================================================
    __aicore__ inline void Write(int32_t dstRankId, uint64_t localAddr,
                                  uint64_t remoteAddr, uint32_t length)
    {
        // SQ 溢出保护
        if (pendingSqCnt_ >= P2P_SQ_FLUSH_THRESHOLD && dstRankId == curDstRankId_) {
            FlushPendingSqes();
        }

        // notifyAddr: 远端窗口状态区 + 本 rank 的状态偏移
        uint64_t notifyAddr = reinterpret_cast<uint64_t>(GetRemoteWinStateAddr(dstRankId, notifyDataState_))
                              + P2P_STATE_OFFSET * (uint64_t)epRankId_;
        uint64_t notifyData = P2P_NOTIFY_DATA_FLAG;

        SetCommWriteWithNotifySQE(sqeTensor_, localAddr, remoteAddr, length, notifyAddr, notifyData, 0);
        PutCommNotifySQE(sqInfoTensor_, cqInfoTensor_, sqeTensor_, cqeTensor_, jfcDoorBellTensor_,
                         1, outSqPi_, outSqPiLinear_, outSqCi_, outCqCi_, outCqCiLinear_);
        pendingSqCnt_++;
    }

    // ========================================================================
    // Drain — 敲 DoorBell + 轮询 CQ + 更新 PICI
    // ========================================================================
    __aicore__ inline void Drain(int32_t dstRankId)
    {
        // 步骤1: 敲 JFS DoorBell
        SendJFSDoorBell(jfsDoorBellTensor_, sqInfoTensor_, outSqPiLinear_);

        // 步骤2: 轮询 CQ 等待完成, 更新 sqCi
        PollCommCQUpdateSQCI(sqInfoTensor_, cqInfoTensor_, cqeTensor_, jfcDoorBellTensor_,
                             outSqCi_, outCqCi_, outCqCiLinear_);

        // 步骤3: 更新 PICI, 保存到 HcclCombinOpParam 的 windowsOut 中
        UpdatePICI((GM_ADDR)hcclContext_, epRankId_, dstRankId,
                   outSqPi_, outSqCi_, outCqPi_, outCqCi_, outSqPiLinear_, outCqCiLinear_);
        pendingSqCnt_ = 0;
    }

    // ========================================================================
    // Process — kernel 入口
    //
    // benchMode=0 (BATCH_WRITE): loopCount 次 Write → 一次 Drain
    //   - 测批量发送带宽: 攒一批 SQE 再统一 flush
    //   - timing: assemble=全部Write时间, wait=Drain时间, total=Write+Drain
    //
    // benchMode=1 (WRITE_DRAIN_LOOP): 循环 loopCount 次 { 一次Write → 一次Drain }
    //   - 测单次往返延迟: 每次 Write 后立即 Drain 等完成
    //   - timing 累加所有轮, Host 侧除以 loopCount 得平均一次 Write+Drain 时间
    // ========================================================================
    __aicore__ inline void Process()
    {
        if (aivId_ != 0) return;

        const uint32_t peer      = tiling_->peer;
        const uint64_t sqeBytes  = tiling_->sqeBytes;
        const uint64_t bufBytes  = tiling_->bufBytes;
        const uint32_t loopCount = (tiling_->loopCount > 0U) ? tiling_->loopCount : 1U;
        const uint32_t benchMode = tiling_->benchMode;

        uint64_t totalWinSizeEp = bufBytes;
        uint64_t winDataSizeOffset = 0;
        notifyDataState_ = 0;

        InitRankComm(peer);

        uint64_t numSlots = (sqeBytes > 0U) ? (bufBytes / sqeBytes) : 1U;
        if (numSlots < 1U) numSlots = 1U;
        const uint32_t sqeDataBytes = static_cast<uint32_t>(sqeBytes);

        AscendC::LocalTensor<int64_t> timingLocal = timingBuf_.Get<int64_t>();

        // 预计算地址: 放打点外, 不计入 timing
        uint64_t localBase = reinterpret_cast<uint64_t>(GetLocalWinAddr(peer, totalWinSizeEp, winDataSizeOffset));
        uint64_t remoteBase = reinterpret_cast<uint64_t>(GetRemoteWinAddr(peer, winDataSizeOffset));

        if (benchMode == BENCH_MODE_BATCH_WRITE) {
            // ========== 场景0: loopCount 次 Write → 一次 Drain ==========
            int64_t t0 = AscendC::GetSystemCycle();
            for (uint32_t i = 0; i < loopCount; ++i) {
                uint64_t off = (static_cast<uint64_t>(i) % numSlots) * sqeBytes;
                Write(peer, localBase + off, remoteBase + off, sqeDataBytes);
            }
            int64_t t1 = AscendC::GetSystemCycle();
            Drain(peer);
            int64_t t2 = AscendC::GetSystemCycle();

            timingLocal.SetValue(0, t1 - t0);  // assemble: 全部 Write
            timingLocal.SetValue(1, t1 - t0);  // issue: 同 assemble
            timingLocal.SetValue(2, t2 - t1);  // wait: Drain
            timingLocal.SetValue(3, t2 - t0);  // total
            timingLocal.SetValue(4, t1 - t0);
            timingLocal.SetValue(5, t1 - t0);
            timingLocal.SetValue(6, t2 - t1);
            timingLocal.SetValue(7, t2 - t0);
        } else {
            // ========== 场景1: 循环 loopCount 次 { Write → Drain } ==========
            int64_t sumWrite = 0, sumDrain = 0, sumTotal = 0;
            int64_t lastWrite = 0, lastDrain = 0, lastTotal = 0;

            for (uint32_t i = 0; i < loopCount; ++i) {
                uint64_t off = (static_cast<uint64_t>(i) % numSlots) * sqeBytes;

                int64_t t0 = AscendC::GetSystemCycle();
                Write(peer, localBase + off, remoteBase + off, sqeDataBytes);
                int64_t t1 = AscendC::GetSystemCycle();
                Drain(peer);
                int64_t t2 = AscendC::GetSystemCycle();

                lastWrite  = t1 - t0;
                lastDrain  = t2 - t1;
                lastTotal  = t2 - t0;
                sumWrite  += lastWrite;
                sumDrain  += lastDrain;
                sumTotal  += lastTotal;
            }

            // 累计值 [0-3], 最后一轮值 [4-7]
            timingLocal.SetValue(0, sumWrite);    // 累计 Write 时间
            timingLocal.SetValue(1, sumWrite);    // 累计 issue (同 Write)
            timingLocal.SetValue(2, sumDrain);    // 累计 Drain 时间
            timingLocal.SetValue(3, sumTotal);    // 累计 total
            timingLocal.SetValue(4, lastWrite);   // 最后一轮 Write
            timingLocal.SetValue(5, lastWrite);   // 最后一轮 issue
            timingLocal.SetValue(6, lastDrain);   // 最后一轮 Drain
            timingLocal.SetValue(7, lastTotal);   // 最后一轮 total
        }

        AscendC::GlobalTensor<int64_t> timingGm;
        timingGm.SetGlobalBuffer((__gm__ int64_t*)timingGm_);
        AscendC::DataCopy(timingGm, timingLocal, TIMING_SLOTS);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }

private:
    // ========================================================================
    // 地址计算 (与 p2p_urma_comm.h 一致)
    // ========================================================================
    __aicore__ inline GM_ADDR GetRemoteWinAddr(int32_t rankId, uint64_t winDataSizeOffset)
    {
        return (GM_ADDR)(hcclContext_->windowsIn[rankId]) + P2P_STATE_SIZE + winDataSizeOffset;
    }

    __aicore__ inline GM_ADDR GetRemoteWinStateAddr(int32_t rankId, uint32_t dataState)
    {
        return (GM_ADDR)(hcclContext_->windowsIn[rankId]) + dataState * P2P_WIN_STATE_OFFSET;
    }

    __aicore__ inline GM_ADDR GetLocalWinAddr(int32_t rankId, uint64_t totalWinSizeEp, uint64_t winDataSizeOffset)
    {
        return (GM_ADDR)localUrmaWorkspace_ + totalWinSizeEp * rankId + P2P_STATE_SIZE + winDataSizeOffset;
    }

    __aicore__ inline GM_ADDR GetLocalWinStateAddr(int32_t rankId, uint64_t totalWinSizeEp, uint32_t dataState)
    {
        return (GM_ADDR)localUrmaWorkspace_ + totalWinSizeEp * rankId + dataState * P2P_WIN_STATE_OFFSET;
    }

    __aicore__ inline void FlushPendingSqes()
    {
        if (pendingSqCnt_ == 0) return;
        SendJFSDoorBell(jfsDoorBellTensor_, sqInfoTensor_, outSqPiLinear_);
        PollCommCQUpdateSQCI(sqInfoTensor_, cqInfoTensor_, cqeTensor_, jfcDoorBellTensor_,
                             outSqCi_, outCqCi_, outCqCiLinear_);
        UpdatePICI((GM_ADDR)hcclContext_, epRankId_, curDstRankId_,
                   outSqPi_, outSqCi_, outCqPi_, outCqCi_, outSqPiLinear_, outCqCiLinear_);
        pendingSqCnt_ = 0;
    }

    // 通信上下文
    __gm__ HcclCombinOpParam *hcclContext_{nullptr};
    AscendC::TPipe *pipe_{nullptr};
    const UrmaP2pBenchTiling* tiling_{nullptr};
    GM_ADDR timingGm_{nullptr};
    GM_ADDR localUrmaWorkspace_{nullptr};
    uint32_t aivId_{0};
    int32_t epRankId_{0};
    int32_t epWorldSize_{0};
    uint32_t notifyDataState_{0};

    // SQE 级 UB 缓冲区
    AscendC::LocalTensor<uint8_t> sqInfoTensor_;
    AscendC::LocalTensor<uint8_t> cqInfoTensor_;
    AscendC::LocalTensor<uint8_t> sqeTensor_;
    AscendC::LocalTensor<uint8_t> cqeTensor_;
    AscendC::LocalTensor<uint8_t> jfsDoorBellTensor_;
    AscendC::LocalTensor<uint8_t> jfcDoorBellTensor_;
    AscendC::TBuf<> sqInfoBuf_;
    AscendC::TBuf<> cqInfoBuf_;
    AscendC::TBuf<> sqeBuf_;
    AscendC::TBuf<> cqeBuf_;
    AscendC::TBuf<> jfsDoorBellBuf_;
    AscendC::TBuf<> jfcDoorBellBuf_;
    AscendC::TBuf<> timingBuf_;

    // PICI 状态
    uint32_t outSqPi_{0};
    uint32_t outSqCi_{0};
    uint32_t outCqPi_{0};
    uint32_t outCqCi_{0};
    uint32_t outSqPiLinear_{0};
    uint32_t outCqCiLinear_{0};
    bool isFirstComm_{true};
    int32_t curDstRankId_{-1};
    uint32_t pendingSqCnt_{0};
};

}  // namespace UrmaP2pBench

#endif  // MC2_NATIVE_KERNEL_H
