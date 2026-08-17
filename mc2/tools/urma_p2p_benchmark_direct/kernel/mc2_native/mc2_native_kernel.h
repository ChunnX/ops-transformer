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
//
// ============================================================================
// 整体数据流 (以 rank0 → rank1 单向 P2P Write 为例):
//
//   Host (test_urma_p2p_bench.cpp):
//     build_hccl_context() 构造 HcclCombinOpParam → 拷到 device GM
//
//   Kernel (本文件):
//     Init:
//       hcclContext_ = (HcclCombinOpParam*)hcclContextGM  // 指向 GM 中的上下文
//       分配 UB 缓冲区 (sqInfo / cqInfo / sqe / cqe / doorBell / timing)
//
//     InitRankComm(peer):
//       从 GM 读 SQ/CQ 信息到 UB → 读 PICI 恢复状态 → 初始化 CQ ring →
//       生成 SQE 模板 + 填通信元数据
//
//     Write(peer, localAddr, remoteAddr, length):
//       填 SQE 数据字段 → PutCommNotifySQE (拷 SQE 到 SQ ring, 不敲 DoorBell)
//       (攒一批 SQE 后统一 flush, 减少 DoorBell 开销)
//
//     Drain(peer):
//       敲 JFS DoorBell (通知硬件取 WQE) → 轮询 CQ 等完成 → 更新 PICI (持久化状态)
//
//     Process():
//       benchMode=0: loopCount 次 Write → 一次 Drain (测批量带宽)
//       benchMode=1: 循环 loopCount 次 { Write → Drain } (测单次延迟)
//       打点 timing → DataCopy 到 GM → Host 读取打印
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

// ============================================================================
// mc2 版的状态区常量 (与 p2p_urma_comm.h 一致)
//
// workspace 内存布局:
//   [0 .. P2P_STATE_SIZE-1]          状态区 (350KB, 存 notify flag 等)
//   [P2P_STATE_SIZE ..]              数据区 (URMA 写入的数据落在这里)
//
// windowsIn[peer] 地址计算:
//   GetRemoteWinAddr = windowsIn[peer] + P2P_STATE_SIZE + offset
//   (对端 workspace 的状态区之后才是数据区)
//
// notify 地址计算:
//   GetRemoteWinStateAddr = windowsIn[peer] + dataState * P2P_WIN_STATE_OFFSET
//   (notify 写到对端的状态区, 用 dataState 区分不同通知类型)
// ============================================================================
constexpr uint64_t P2P_STATE_SIZE = 350UL * 1024UL;          // workspace 状态区大小 (350KB)
constexpr uint64_t P2P_WIN_STATE_OFFSET = 350UL * 1024UL;    // 状态窗口偏移 (350KB, 同 P2P_STATE_SIZE)
constexpr uint32_t P2P_UB_ALIGN = 32U;                       // UB 对齐 (32B)
constexpr uint32_t P2P_CQ_DEPTH_256 = 256U;                  // CQ 深度上限 (256 个 CQE)
constexpr uint32_t P2P_SQ_FLUSH_THRESHOLD = 200U;            // SQ 攒满 200 个 WQE 时自动 flush
constexpr uint32_t P2P_STATE_OFFSET = 32U;                   // 状态区内每个 rank 的偏移 (32B)
constexpr uint64_t P2P_NOTIFY_DATA_FLAG = 0x3F800000UL;      // notify 数据标志 (0x3F800000 = float 1.0 的位模式)

// ============================================================================
// Mc2NativeBenchOp — mc2_native 基准测试 kernel 类
//
// 生命周期:
//   Init()       — 绑定 GM 地址, 分配 UB 缓冲区
//   Process()    — kernel 入口, 调用 InitRankComm → Write/Drain → 打点
//
// 关键成员:
//   hcclContext_   — GM 上的 HcclCombinOpParam (Host 侧构造)
//   sqInfoTensor_  — UB 上的 SQ 信息 (从 GM 拷来)
//   cqInfoTensor_  — UB 上的 CQ 信息
//   sqeTensor_     — UB 上的 SQE 模板 (组装后拷到 SQ ring)
//   cqeTensor_     — UB 上的 CQE 缓冲区 (从 CQ ring 拷来读状态)
//   outSqPi_ 等    — PICI 索引 (跨 launch 持久化的 SQ/CQ 读写位置)
// ============================================================================
class Mc2NativeBenchOp {
public:
    // ========================================================================
    // Init — 绑定 GM 地址, 分配 UB 缓冲区
    //
    // 参数:
    //   hcclContextGM - HcclCombinOpParam 在 GM 中的地址 (Host 侧 build_hccl_context 构造)
    //   workspace     - 本端 URMA workspace 地址 (数据源, URMA 从这里读数据写到对端)
    //   timingGm      - timing 结果输出 GM 地址 (8 个 int64_t)
    //   pipe          - TPipe 指针 (管理 UB 分配)
    //   tiling        - tiling 参数 (peer/numSqe/sqeBytes/loopCount/benchMode 等)
    //
    // UB 缓冲区分配 (与 P2pUrmaSqeComm::InitCommBuffer 一致):
    //   sqInfoBuf_    — sizeof(HcclAiRMAWQ) 向上对齐到 32B
    //   cqInfoBuf_    — sizeof(HcclAiRMACQ) 向上对齐到 32B
    //   sqeBuf_       — WRITE_WITH_NOTIFY_SQE_SIZE (96B)
    //   cqeBuf_       — 32B * 256 = 8KB (CQE 轮询缓冲区)
    //   jfsDoorBellBuf_ — 32B (JFS DoorBell, 当前未实际使用)
    //   jfcDoorBellBuf_ — 32B (JFC DoorBell, 当前未实际使用)
    //   timingBuf_    — 8 * 8B = 64B (timing 打点)
    // ========================================================================
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
    // InitRankComm — 初始化与指定 rank 的通信上下文
    //
    // 与 P2pUrmaSqeComm::InitRankComm 逻辑完全一致, 共 8 步:
    //
    // 步骤1: GetURMASqInfoTensor — 从 HcclCombinOpParam.sqs[dstRankId] 读 SQ 信息到 UB
    //        SyncFunc<MTE2_S> 等 DataCopy (GM→UB) 完成
    //
    // 步骤2: GetURMACqInfoTensor — 从 HcclCombinOpParam.cqs[dstRankId] 读 CQ 信息到 UB
    //        SyncFunc<MTE2_S> 等 DataCopy 完成
    //
    // 步骤3: GetIsFirstInComm — 读 PICI 窗口的 isFirstFlag
    //        isFirstComm_=!isFirstComm (取反: flag=0 → 首次, flag=1 → 非首次)
    //
    // 步骤4: GetPICI — 读 windowsOut[myRank]+1MB+512B*dstRankId 的 sqPi/sqCi/cqCi
    //        恢复上次 launch 结束时的 SQ/CQ 环形索引
    //
    // 步骤5: InvalidateCqeStatus — 仅首次通信时, CQ ring 全填 0xff
    //        (CQE 初始 status 必须是 0xff, CheckCqeStatus 才能正确判断未完成)
    //        然后 UpdateIsFirstInComm 标记已初始化
    //
    // 步骤6: PollCommCQUpdateSQCI — 清理残余 CQE + 敲 JFC DoorBell
    //        (上次 launch 可能遗留未消费的 CQE, 清理后 sqCi 准确)
    //
    // 步骤7: GenerateCommWriteWithNotifySQE — 生成 SQE 模板 (opcode=0x5, 96B)
    //        初始化固定字段: odr/owner/jetty_type/token_en/opcode/sge_num
    //
    // 步骤8: UpdateCommWriteWithNotifySQE — 填 channel 元数据
    //        从 sqInfoTensor 读 tp_id/eid/token, 填入 SQE 模板
    //        此模板后续只需 SetCommWriteWithNotifySQE 填数据地址即可复用
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
        // GetIsFirstInComm 返回 true=已初始化(非首次), false=首次
        // isFirstComm_ 取反: true=首次需要初始化, false=不需要
        bool isFirstComm = false;
        GetIsFirstInComm((GM_ADDR)hcclContext_, epRankId_, dstRankId, isFirstComm);
        isFirstComm_ = !isFirstComm;

        // 步骤4: 获取 PICI (Producer-Consumer Index)
        // 从 windowsOut[myRank]+1MB+512B*dstRankId 读取上次 launch 保存的 SQ/CQ 索引
        GetPICI((GM_ADDR)hcclContext_, epRankId_, dstRankId,
                outSqPi_, outSqCi_, outCqPi_, outCqCi_, outSqPiLinear_, outCqCiLinear_);

        // 步骤5: 首次通信时初始化 CQ ring (全填 0xff)
        // CQE status=0xff 表示"无效/未完成", CheckCqeStatus 遇到 0xff 会停止轮询
        if (isFirstComm_) {
            InvalidateCqeStatus(cqInfoTensor_, cqeTensor_);
            SyncFunc<AscendC::HardEvent::MTE3_S>();
            UpdateIsFirstInComm((GM_ADDR)hcclContext_, epRankId_, dstRankId, true);
        }

        // 步骤6: 轮询 CQ 清理残余完成事件, 更新 sqCi
        // 上次 launch 可能遗留未消费的 CQE, 清理后 sqCi 准确反映硬件处理进度
        PollCommCQUpdateSQCI(sqInfoTensor_, cqInfoTensor_, cqeTensor_, jfcDoorBellTensor_,
                             outSqCi_, outCqCi_, outCqCiLinear_);

        // 步骤7: 生成 WriteWithNotify SQE 模板 (opcode=0x5, 96B)
        // 初始化固定字段: odr=0b001, owner=1, rmt_jetty_type=0b01, token_en=1, opcode=0x5, sge_num=1
        GenerateCommWriteWithNotifySQE(sqeTensor_, 1);

        // 步骤8: 更新 SQE 模板中的通信元数据 (tp_id, jetty_id, rmt_eid, token 等)
        // 从 sqInfoTensor (HcclAiRMAWQ) 读取, 填入 SQE 模板
        // 此后 SQE 模板的元数据部分固定, 只需 SetCommWriteWithNotifySQE 填数据地址即可
        UpdateCommWriteWithNotifySQE(sqeTensor_, sqInfoTensor_);

        curDstRankId_ = dstRankId;
        pendingSqCnt_ = 0;
    }

    // ========================================================================
    // Write — 提交一个 WriteWithNotify SQE (不敲 DoorBell, 不 poll CQ)
    //
    // 参数:
    //   dstRankId  - 目标 rank ID
    //   localAddr  - 本端源数据地址 (HBM, URMA 从这里读数据)
    //   remoteAddr - 远端目的地址 (对端 HBM, 数据写到这里)
    //   length     - 传输数据长度 (字节)
    //
    // 流程:
    //   1. SQ 溢出保护: 如果 pendingSqCnt_ >= 200, 先 FlushPendingSqes
    //      (敲 DoorBell + poll CQ 腾空间, 防止 SQ ring 写满)
    //   2. 计算 notifyAddr 和 notifyData:
    //      notifyAddr = 对端 workspace 状态区 + 本 rank 的偏移
    //      notifyData = 0x3F800000 (float 1.0 的位模式, 作为完成标志)
    //      (远端写完数据后, 往 notifyAddr 写 notifyData, 接收方可轮询此地址感知数据到达)
    //   3. SetCommWriteWithNotifySQE: 填 dataAddr/rmtAddr/length/notifyAddr/notifyData (cqe=0, 不生成 CQE)
    //   4. PutCommNotifySQE: 检查 SQ 空间 → DataCopy SQE 到 SQ ring → 更新 outSqPi
    //   5. pendingSqCnt_++
    //
    // 注意: 此函数不敲 JFS DoorBell, WQE 写到 SQ ring 但硬件不知道
    //       需要调用方在适当时机调用 Drain (或 FlushPendingSqes) 敲 DoorBell
    //       (批量写模式: 攒一批 SQE 后统一敲一次 DoorBell, 减少 DoorBell 开销)
    // ========================================================================
    __aicore__ inline void Write(int32_t dstRankId, uint64_t localAddr,
                                  uint64_t remoteAddr, uint32_t length)
    {
        // SQ 溢出保护: 攒满 200 个 WQE 时自动 flush
        if (pendingSqCnt_ >= P2P_SQ_FLUSH_THRESHOLD && dstRankId == curDstRankId_) {
            FlushPendingSqes();
        }

        // notifyAddr: 远端窗口状态区 + 本 rank 的状态偏移
        // 远端写完数据后, 往 notifyAddr 写 notifyData, 作为完成通知
        // GetRemoteWinStateAddr(dstRankId, 0) = windowsIn[dstRankId] + 0 * P2P_WIN_STATE_OFFSET
        //   = windowsIn[dstRankId] (状态区起始)
        // + P2P_STATE_OFFSET * epRankId_ = + 32 * myRank (每个 rank 占 32B 状态区)
        uint64_t notifyAddr = reinterpret_cast<uint64_t>(GetRemoteWinStateAddr(dstRankId, notifyDataState_))
                              + P2P_STATE_OFFSET * (uint64_t)epRankId_;
        uint64_t notifyData = P2P_NOTIFY_DATA_FLAG;

        // 填充 SQE 数据字段 (cqe=0: 不生成 CQE, 因为后面会统一 Drain)
        SetCommWriteWithNotifySQE(sqeTensor_, localAddr, remoteAddr, length, notifyAddr, notifyData, 0);
        // 拷 SQE 到 SQ ring (不敲 DoorBell), 更新 outSqPi
        PutCommNotifySQE(sqInfoTensor_, cqInfoTensor_, sqeTensor_, cqeTensor_, jfcDoorBellTensor_,
                         1, outSqPi_, outSqPiLinear_, outSqCi_, outCqCi_, outCqCiLinear_);
        pendingSqCnt_++;
    }

    // ========================================================================
    // Drain — 敲 DoorBell + 轮询 CQ 等完成 + 更新 PICI
    //
    // 参数:
    //   dstRankId - 目标 rank ID
    //
    // 流程:
    //   1. SendJFSDoorBell: 敲 JFS DoorBell, 通知硬件 "SQ 有新 WQE 了"
    //      传入 outSqPiLinear_ (线性计数, 不取模, 硬件用此判断新 WQE 数量)
    //   2. PollCommCQUpdateSQCI: 轮询 CQ 等待所有 WQE 完成
    //      循环读 CQE 直到 status=0xff (未完成), 消费已完成的 CQE, 更新 sqCi/cqCi
    //      敲 JFC DoorBell 通知硬件 CQE 已消费
    //   3. UpdatePICI: 将最新的 SQ/CQ 索引写回 PICI 窗口
    //      供下次 launch 的 GetPICI 读取, 实现跨 launch 状态续传
    //   4. pendingSqCnt_ = 0
    //
    // 注意: Drain 会阻塞直到所有已提交的 WQE 完成 (CQE 全部消费)
    // ========================================================================
    __aicore__ inline void Drain(int32_t dstRankId)
    {
        // 步骤1: 敲 JFS DoorBell, 通知硬件 SQ 有新 WQE
        // outSqPiLinear_ 是 SQ 线性计数 (不取模), 硬件用此判断新增的 WQE 数量
        SendJFSDoorBell(jfsDoorBellTensor_, sqInfoTensor_, outSqPiLinear_);

        // 步骤2: 轮询 CQ 等待完成, 更新 sqCi
        // 循环读 CQE: status=0 (完成) → 消费 + 递增 cqCi; status=0xff (未完成) → break
        // 敲 JFC DoorBell 通知硬件 cqCi 更新
        PollCommCQUpdateSQCI(sqInfoTensor_, cqInfoTensor_, cqeTensor_, jfcDoorBellTensor_,
                             outSqCi_, outCqCi_, outCqCiLinear_);

        // 步骤3: 更新 PICI, 保存到 HcclCombinOpParam 的 windowsOut 中
        // 下次 launch 的 GetPICI 会读出这些值, 实现 SQ/CQ 状态续传
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
    //   - 适合大块数据传输, 看 URMA 峰值带宽
    //
    // benchMode=1 (WRITE_DRAIN_LOOP): 循环 loopCount 次 { 一次Write → 一次Drain }
    //   - 测单次往返延迟: 每次 Write 后立即 Drain 等完成
    //   - timing 累加所有轮, Host 侧除以 loopCount 得平均一次 Write+Drain 时间
    //   - 测 URMA 单次操作延迟 (含 DoorBell + CQ 轮询开销)
    //
    // Timing 打点 (8 个 int64_t):
    //   [0]=assemble累计 [1]=issue累计 [2]=wait累计 [3]=total累计
    //   [4]=最后一轮assemble [5]=最后一轮issue [6]=最后一轮wait [7]=最后一轮total
    //
    // 地址预计算 (在打点外, 不计入 timing):
    //   localBase  = workspace + totalWinSizeEp * peer + P2P_STATE_SIZE
    //              = 本端 workspace 数据区起始 (源数据)
    //   remoteBase = windowsIn[peer] + P2P_STATE_SIZE
    //              = 对端 workspace 数据区起始 (目的地址)
    //   每次 Write 的偏移: off = (i % numSlots) * sqeBytes
    //   (在 workspace 内循环使用不同偏移, 避免每次写同一地址)
    // ========================================================================
    __aicore__ inline void Process()
    {
        // 单核执行 (blockDim=1, 仅 aivId==0 执行)
        if (aivId_ != 0) return;

        const uint32_t peer      = tiling_->peer;
        const uint64_t sqeBytes  = tiling_->sqeBytes;
        const uint64_t bufBytes  = tiling_->bufBytes;
        const uint32_t loopCount = (tiling_->loopCount > 0U) ? tiling_->loopCount : 1U;
        const uint32_t benchMode = tiling_->benchMode;

        // totalWinSizeEp: 每个 rank 的窗口大小 = bufBytes (整个 workspace)
        // winDataSizeOffset: 数据区在窗口内的偏移 = 0 (状态区已由地址计算处理)
        uint64_t totalWinSizeEp = bufBytes;
        uint64_t winDataSizeOffset = 0;
        notifyDataState_ = 0;

        // 初始化与 peer 的通信上下文 (8 步, 见 InitRankComm 注释)
        InitRankComm(peer);

        // numSlots: workspace 可容纳多少个 sqeBytes 大小的数据块
        // 用于循环写入不同偏移, 避免每次写同一地址
        uint64_t numSlots = (sqeBytes > 0U) ? (bufBytes / sqeBytes) : 1U;
        if (numSlots < 1U) numSlots = 1U;
        const uint32_t sqeDataBytes = static_cast<uint32_t>(sqeBytes);

        AscendC::LocalTensor<int64_t> timingLocal = timingBuf_.Get<int64_t>();

        // 预计算地址: 放打点外, 不计入 timing
        // localBase: 本端源数据地址 = workspace + peer * winSize + 状态区(350KB) + offset
        uint64_t localBase = reinterpret_cast<uint64_t>(GetLocalWinAddr(peer, totalWinSizeEp, winDataSizeOffset));
        // remoteBase: 对端目的地址 = windowsIn[peer] + 状态区(350KB) + offset
        uint64_t remoteBase = reinterpret_cast<uint64_t>(GetRemoteWinAddr(peer, winDataSizeOffset));

        if (benchMode == BENCH_MODE_BATCH_WRITE) {
            // ========== 场景0: loopCount 次 Write → 一次 Drain ==========
            // 攒一批 SQE (不敲 DoorBell), 最后统一 Drain
            // 测 URMA 批量写带宽 (DoorBell 开销分摊到所有 WQE)
            int64_t t0 = AscendC::GetSystemCycle();
            for (uint32_t i = 0; i < loopCount; ++i) {
                // 循环使用 workspace 内不同偏移: off = (i % numSlots) * sqeBytes
                uint64_t off = (static_cast<uint64_t>(i) % numSlots) * sqeBytes;
                Write(peer, localBase + off, remoteBase + off, sqeDataBytes);
            }
            int64_t t1 = AscendC::GetSystemCycle();
            Drain(peer);
            int64_t t2 = AscendC::GetSystemCycle();

            // batch 模式: 累计值和最后一轮值相同 (只有一轮)
            timingLocal.SetValue(0, t1 - t0);  // assemble: 全部 Write
            timingLocal.SetValue(1, t1 - t0);  // issue: 同 assemble (mc2_native 无 trailing SQE)
            timingLocal.SetValue(2, t2 - t1);  // wait: Drain
            timingLocal.SetValue(3, t2 - t0);  // total
            timingLocal.SetValue(4, t1 - t0);  // 最后一轮 (同累计, 只有一轮)
            timingLocal.SetValue(5, t1 - t0);
            timingLocal.SetValue(6, t2 - t1);
            timingLocal.SetValue(7, t2 - t0);
        } else {
            // ========== 场景1: 循环 loopCount 次 { Write → Drain } ==========
            // 每次 Write 后立即 Drain 等完成
            // 测 URMA 单次往返延迟 (含 DoorBell + CQ 轮询开销)
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
            // Host 侧除以 loopCount 得平均单次时间
            timingLocal.SetValue(0, sumWrite);    // 累计 Write 时间
            timingLocal.SetValue(1, sumWrite);    // 累计 issue (同 Write)
            timingLocal.SetValue(2, sumDrain);    // 累计 Drain 时间
            timingLocal.SetValue(3, sumTotal);    // 累计 total
            timingLocal.SetValue(4, lastWrite);   // 最后一轮 Write
            timingLocal.SetValue(5, lastWrite);   // 最后一轮 issue
            timingLocal.SetValue(6, lastDrain);   // 最后一轮 Drain
            timingLocal.SetValue(7, lastTotal);   // 最后一轮 total
        }

        // DataCopy timing 到 GM, Host 侧 aclrtMemcpy D2H 读取
        AscendC::GlobalTensor<int64_t> timingGm;
        timingGm.SetGlobalBuffer((__gm__ int64_t*)timingGm_);
        AscendC::DataCopy(timingGm, timingLocal, TIMING_SLOTS);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }

private:
    // ========================================================================
    // 地址计算 (与 p2p_urma_comm.h 一致)
    //
    // workspace 内存布局 (每个 rank):
    //   [0 .. P2P_STATE_SIZE-1]                    状态区 (350KB)
    //   [P2P_STATE_SIZE .. totalWinSize-1]         数据区
    //
    // GetRemoteWinAddr: 对端数据区地址 = windowsIn[rankId] + 350KB + offset
    // GetRemoteWinStateAddr: 对端状态区地址 = windowsIn[rankId] + dataState * 350KB
    // GetLocalWinAddr: 本端数据区地址 = workspace + rankId * winSize + 350KB + offset
    // GetLocalWinStateAddr: 本端状态区地址 = workspace + rankId * winSize + dataState * 350KB
    // ========================================================================

    // 对端数据区地址 (URMA 写入目的地址)
    __aicore__ inline GM_ADDR GetRemoteWinAddr(int32_t rankId, uint64_t winDataSizeOffset)
    {
        return (GM_ADDR)(hcclContext_->windowsIn[rankId]) + P2P_STATE_SIZE + winDataSizeOffset;
    }

    // 对端状态区地址 (notify 写入目的地址)
    __aicore__ inline GM_ADDR GetRemoteWinStateAddr(int32_t rankId, uint32_t dataState)
    {
        return (GM_ADDR)(hcclContext_->windowsIn[rankId]) + dataState * P2P_WIN_STATE_OFFSET;
    }

    // 本端数据区地址 (URMA 源数据地址)
    __aicore__ inline GM_ADDR GetLocalWinAddr(int32_t rankId, uint64_t totalWinSizeEp, uint64_t winDataSizeOffset)
    {
        return (GM_ADDR)localUrmaWorkspace_ + totalWinSizeEp * rankId + P2P_STATE_SIZE + winDataSizeOffset;
    }

    // 本端状态区地址
    __aicore__ inline GM_ADDR GetLocalWinStateAddr(int32_t rankId, uint64_t totalWinSizeEp, uint32_t dataState)
    {
        return (GM_ADDR)localUrmaWorkspace_ + totalWinSizeEp * rankId + dataState * P2P_WIN_STATE_OFFSET;
    }

    // ========================================================================
    // FlushPendingSqes — 攒满 SQ 时自动 flush (敲 DoorBell + poll CQ + 更新 PICI)
    //
    // 与 Drain 的区别: 不重置 pendingSqCnt_ 为 0 由调用方处理
    // (实际上这里也设为 0, Drain 内部也调了这个)
    // ========================================================================
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

    // ========================================================================
    // 成员变量
    // ========================================================================

    // 通信上下文
    __gm__ HcclCombinOpParam *hcclContext_{nullptr}; // GM 上的 HcclCombinOpParam (Host 侧构造)
    AscendC::TPipe *pipe_{nullptr};                   // TPipe (管理 UB 分配)
    const UrmaP2pBenchTiling* tiling_{nullptr};       // tiling 参数
    GM_ADDR timingGm_{nullptr};                       // timing 输出 GM 地址
    GM_ADDR localUrmaWorkspace_{nullptr};             // 本端 workspace 地址
    uint32_t aivId_{0};                               // AIV ID (仅 aivId==0 执行)
    int32_t epRankId_{0};                             // 本 rank ID
    int32_t epWorldSize_{0};                          // world size
    uint32_t notifyDataState_{0};                     // notify 状态类型 (0=默认)

    // SQE 级 UB 缓冲区
    AscendC::LocalTensor<uint8_t> sqInfoTensor_;      // SQ 信息 (HcclAiRMAWQ)
    AscendC::LocalTensor<uint8_t> cqInfoTensor_;      // CQ 信息 (HcclAiRMACQ)
    AscendC::LocalTensor<uint8_t> sqeTensor_;         // SQE 模板 (96B)
    AscendC::LocalTensor<uint8_t> cqeTensor_;         // CQE 缓冲区 (8KB, 256*32B)
    AscendC::LocalTensor<uint8_t> jfsDoorBellTensor_; // JFS DoorBell 缓冲区 (保留)
    AscendC::LocalTensor<uint8_t> jfcDoorBellTensor_; // JFC DoorBell 缓冲区 (保留)
    AscendC::TBuf<> sqInfoBuf_;
    AscendC::TBuf<> cqInfoBuf_;
    AscendC::TBuf<> sqeBuf_;
    AscendC::TBuf<> cqeBuf_;
    AscendC::TBuf<> jfsDoorBellBuf_;
    AscendC::TBuf<> jfcDoorBellBuf_;
    AscendC::TBuf<> timingBuf_;                       // timing 打点 (8*8B=64B)

    // PICI 状态 (跨 launch 持久化的 SQ/CQ 环形索引)
    uint32_t outSqPi_{0};        // SQ 生产者索引 (kernel 写到哪了, 环形)
    uint32_t outSqCi_{0};        // SQ 消费者索引 (硬件处理到哪了, 环形)
    uint32_t outCqPi_{0};        // CQ 生产者索引 (硬件写到哪了, 环形)
    uint32_t outCqCi_{0};        // CQ 消费者索引 (kernel 读到哪了, 环形)
    uint32_t outSqPiLinear_{0};  // SQ 线性计数 (不取模, DoorBell 用)
    uint32_t outCqCiLinear_{0};  // CQ 线性计数 (不取模, DoorBell 用)
    bool isFirstComm_{true};     // 是否首次通信 (需 InvalidateCqeStatus)
    int32_t curDstRankId_{-1};   // 当前目标 rank
    uint32_t pendingSqCnt_{0};   // 已提交但未 flush 的 SQE 数
};

}  // namespace UrmaP2pBench

#endif  // MC2_NATIVE_KERNEL_H
