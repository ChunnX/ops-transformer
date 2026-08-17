/**
 * moe_distribute_base_native.h — moe_distribute_base.h 的无依赖版本
 *
 * 保留全部接口(GenerateCommWriteWithNotifySQE / UpdateCommWriteWithNotifySQE /
 * SetCommWriteWithNotifySQE / PutCommNotifySQE / SendJFSDoorBell / PollCommCQUpdateSQCI /
 * GetPICI / UpdatePICI / GetIsFirstInComm / UpdateIsFirstInComm / InvalidateCqeStatus等),
 * 仅替换 mc2_kernel_utils.h 依赖(SyncFunc内联实现)。
 *
 * HcclCombinOpParam / HcclAiRMAWQ / HcclAiRMACQ 结构体定义与原版完全一致,
 * 直调版在 Host 侧构造等效数据后传入 kernel。
 *
 * ============================================================================
 * URMA 通信协议概述
 * ============================================================================
 *
 * URMA (Unified Remote Memory Access) 是 Ascend NPU 上的 P2P 远程内存访问机制,
 * 允许一个 rank 直接写入另一个 rank 的 HBM, 无需 CPU 介入。
 *
 * 通信模型 (生产者-消费者):
 *
 *   Sender (本端 kernel):                Receiver (对端硬件):
 *   ┌─────────────────────────┐          ┌─────────────────────────┐
 *   │ SQ (Submission Queue)   │ ──WQE──▶ │ URMA Engine             │
 *   │   kernel 写 WQE 到 ring │          │   解析 WQE, 执行远程写   │
 *   │                         │          │   写数据到对端 HBM       │
 *   │ CQ (Completion Queue)   │ ◀─CQE── │   写 CQE 到本端 CQ ring  │
 *   │   kernel 轮询 CQE       │          │                         │
 *   └─────────────────────────┘          └─────────────────────────┘
 *
 * WQE (Work Queue Entry): 描述一次远程写操作 (源地址、目的地址、长度、通知地址等)
 * CQE (Completion Queue Entry): 描述一次写操作的完成状态
 *
 * DoorBell 机制:
 *   - JFS DoorBell: 敲它通知硬件 "SQ 有新 WQE 了, 快来取" (更新 sqPi)
 *   - JFC DoorBell: 敲它通知硬件 "CQ 的 CQE 被消费了" (更新 cqCi)
 *
 * PICI (Producer-Consumer Index):
 *   跨 kernel launch 持久化的环形队列索引, 保存在 windowsOut 的专用窗口中。
 *   kernel 结束时 UpdatePICI 写回, 下次开始时 GetPICI 读出, 实现 SQ/CQ 状态续传。
 *
 * WriteWithNotify SQE (opcode=0x5, 96B):
 *   比 Write SQE (opcode=0x3, 64B) 多了 notify_addr + notify_data 字段,
 *   远端写完数据后, 额外往 notify_addr 写一个 notify_data, 作为完成通知。
 *   每个 WriteWithNotify SQE 占 2 个 WQEBB (96B 向上对齐到 128B = 2×64B)。
 */

#ifndef MOE_DISTRIBUTE_BASE_NATIVE_H
#define MOE_DISTRIBUTE_BASE_NATIVE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hccl/hccl.h"
#include "moe_distribute_base_types.h"

// ============================================================================
// SyncFunc: 内联替代 mc2_kernel_utils.h 中的同名函数
// 从 TPipe 获取 eventID, SetFlag + WaitFlag 实现流水线同步
// 定义在 AscendC 命名空间内, 与原版 mc2_kernel_utils.h 一致
//
// 用途: 在 Vector/Scalar/MTE2/MTE3 等不同流水线之间插入同步点,
//       确保 DataCopy 完成后再读取数据, 或 Scalar 写完后再 DataCopy。
// ============================================================================
namespace AscendC {
template<HardEvent event>
__aicore__ inline void SyncFunc()
{
    TEventID eventID = GetTPipePtr()->FetchEventID(event);
    SetFlag<event>(eventID);
    WaitFlag<event>(eventID);
}
} // namespace AscendC

// ============================================================================
// 常量定义 (与 moe_distribute_base.h 完全一致)
// ============================================================================
// MAX_RANK_NUM / MAX_OP_NUM / WIN_PICI_OFFSET / PICI_WIN_SIZE 已在 moe_distribute_base_types.h 中定义

// SQE (Submission Queue Entry) 大小
constexpr uint32_t WRITE_SQE_SIZE = 64U;             // Write SQE: 64B = 1 个 WQEBB
constexpr uint32_t WRITE_WITH_NOTIFY_SQE_SIZE = 96U; // WriteWithNotify SQE: 96B (占 2 个 WQEBB, 对齐到 128B)

// 窗口偏移常量 (用于地址计算, 与 mc2 算子版 p2p_urma_comm.h 一致)
constexpr uint64_t WIN_STATE_OFFSET = 350U * 1024U; // 状态区偏移 (350KB)
constexpr uint64_t STATE_WIN_OFFSET = 950U * 1024U; // 状态窗口偏移 (950KB)

// CQE 相关
constexpr uint32_t NORMAL_CQE_SIZE = 64U;  // 单个 CQE 大小 (64B)
constexpr uint32_t CQ_DEPTH_256 = 256U;    // UB 一次最多搬运 256 个 CQE (256*32B = 8KB)
constexpr uint32_t UB_ALIGN = 32U;         // UB 按 32 字节对齐

// ============================================================================
// PICI 窗口字段偏移 (每个 peer 占 PICI_WIN_SIZE=512B)
//
// PICI 窗口布局 (uint32_t 数组):
//   [0] sqPi        — SQ 生产者索引 (kernel 写到哪了, 即下一个可写的 WQE 位置)
//   [1] sqCi        — SQ 消费者索引 (硬件处理到哪了, 即下一个待处理的 WQE 位置)
//   [2] cqPi        — CQ 生产者索引 (硬件写到哪了, 即下一个待消费的 CQE 位置)
//   [3] cqCi        — CQ 消费者索引 (kernel 读到哪了, 即下一个待读取的 CQE 位置)
//   [4] sqPiLinear  — SQ 线性计数 (不取模, 用于 DoorBell, 单调递增)
//   [5] cqCiLinear  — CQ 线性计数 (不取模, 用于 DoorBell, 单调递增)
//   [6] isFirstFlag — 是否已初始化过通信 (首次需 InvalidateCqeStatus)
// ============================================================================
constexpr uint32_t WIN_SQPI_OFFSET = 0U;
constexpr uint32_t WIN_SQCI_OFFSET = 1U;
constexpr uint32_t WIN_CQPI_OFFSET = 2U;
constexpr uint32_t WIN_CQCI_OFFSET = 3U;
constexpr uint32_t WIN_SQPILINEAR_OFFSET = 4U;
constexpr uint32_t WIN_CQCILINEAR_OFFSET = 5U;
constexpr uint32_t WIN_FIRST_TIME_CREATE_WIN_FLAG_OFFSET = 6U;
constexpr uint32_t UINT8_BITS_OFFSET = 8U; // 8 bits per byte (用于 CQE entry_idx 高低字节合并)

// ============================================================================
// WQ (Work Queue) 结构体字段偏移 — HcclAiRMAWQ 的字段在内存中的位置
//
// HcclAiRMAWQ 以 uint32_t 数组方式访问时的下标:
//
// 32bit 偏移:
//   [0]  jettyId         — Jetty ID
//   [4]  wqeSize         — 单个 WQEBB 大小 (64B)
//   [5]  sqDepth         — SQ 深度 (256)
//   [12] tp_id           — Traffic Profile ID (24bit, URMA 路由用)
//   [13] rmtEid[0..3]    — 远端 EID 的第 0~3 个 uint32 (共 128bit = 16B)
//   [14] rmtEid[4..7]
//   [15] rmtEid[8..11]
//   [16] rmtEid[12..15]
//   [17] rmtObjId        — 远端 Token ID (填入 SQE 的 rmt_token_id / notify_token_id)
//   [18] rmtTokenValue   — 远端 Token Value (填入 SQE 的 rmt_token_value / notify_token_value)
//   [19] localTokenId    — 本端 Token ID
//
// 64bit 偏移 (ReinterpretCast<uint64_t> 后):
//   [1]  sqVA            — SQ ring 起始地址
//   [5]  dbAddr          — JFS DoorBell 地址
// ============================================================================
constexpr uint32_t WQ_JETTYID_OFFSET = 0U;
constexpr uint32_t WQ_WQESIZE_OFFSET = 4U;
constexpr uint32_t WQ_SQDEPTH_OFFSET = 5U;
constexpr uint32_t WQ_TP_ID_OFFSET = 12U;
constexpr uint32_t WQ_RMTEID_0_3_OFFSET = 13U;
constexpr uint32_t WQ_RMTEID_4_7_OFFSET = 14U;
constexpr uint32_t WQ_RMTEID_8_11_OFFSET = 15U;
constexpr uint32_t WQ_RMTEID_12_15_OFFSET = 16U;
constexpr uint32_t WQ_RMTOBJID_OFFSET = 17U;
constexpr uint32_t WQ_RMTTOKENVALUE_OFFSET = 18U;
constexpr uint32_t WQ_LOCALTOKENID_OFFSET = 19U;
// WQ 64bit offset
constexpr uint32_t WQ_SQVA_OFFSET = 1U;
constexpr uint32_t WQ_DBADDR_OFFSET = 5U;

// ============================================================================
// CQ (Completion Queue) 结构体字段偏移 — HcclAiRMACQ 的字段在内存中的位置
//
// 32bit 偏移:
//   [0]  jfcId    — JFC ID (填入 JFC DoorBell 值的高 32bit)
//   [4]  cqeSize  — 单个 CQE 大小 (64B)
//   [5]  cqDepth  — CQ 深度 (256)
//
// 64bit 偏移:
//   [1]  cqVA     — CQ ring 起始地址
//   [5]  dbAddr   — JFC DoorBell 地址
// ============================================================================
constexpr uint32_t CQ_JFCID_OFFSET = 0U;
constexpr uint32_t CQ_CQESIZE_OFFSET = 4U;
constexpr uint32_t CQ_CQDEPTH_OFFSET = 5U;
// CQ 64bit offset
constexpr uint32_t CQ_CQVA_OFFSET = 1U;
constexpr uint32_t CQ_DBADDR_OFFSET = 5U;

// ============================================================================
// URMA 协议 SQE (Submission Queue Entry) 字段偏移
//
// SQE 是写入 SQ ring 的 WQE, 描述一次远程写操作。
// 两种 SQE:
//   Write SQE (opcode=0x3):       64B, 仅数据搬运
//   WriteWithNotify SQE (opcode=0x5): 96B, 数据搬运 + 远端通知
//
// SQE 内存布局 (以 uint8_t / uint32_t / uint64_t 三种视角访问):
//
// === 8bit 偏移 ===
//   [2]  udf_flg:1 inline_en:1 cqe:1 se:1 fence:1 odr:3
//        - cqe: 是否生成 CQE (1=生成, 0=不生成)
//        - odr: Ordered flag (0b001=无序, 0b010=有序)
//   [3]  owner:1 rmt_jetty_type:2 token_en:1 nf:1 rsv:3
//        - owner: 环形队列 owner bit (硬件用此判断 WQE 是否有效)
//        - rmt_jetty_type: 远端 Jetty 类型 (0b01=UB)
//        - token_en: 是否启用 token 鉴权
//   [4]  target_hint    — 目标提示
//   [5]  opcode         — 操作码 (0x3=Write, 0x5=WriteWithNotify)
//   [11] sge_num        — SGE (Scatter Gather Entry) 个数
//
// === 32bit 偏移 (ReinterpretCast<uint32_t> 后) ===
//   [2]  sge_num:8 tp_id:24     — SGE 个数 + Traffic Profile ID
//   [3]  rmt_jetty_or_seg_id    — 远端 Jetty/Segment ID
//   [4]  rmt_eid[31:0]          — 远端 EID 的低 32bit
//   [5]  rmt_eid[63:32]
//   [6]  rmt_eid[95:64]
//   [7]  rmt_eid[127:96]
//   [8]  rmt_token_value        — 远端 Token Value (鉴权用)
//   [9]  udf                    — User Defined Field
//   [12] length (Write SQE)     — 传输数据长度 (字节)
//   [13] rsv:12 token_id:20     — Token ID (Write SQE)
//
//   WriteWithNotify SQE 额外字段:
//   [20] length                 — 传输数据长度 (字节)
//   [21] rsv:12 token_id:20     — Token ID
//   [12] rsv:12 notify_token_id:20     — 通知 Token ID
//   [13] notify_token_value           — 通知 Token Value
//
// === 64bit 偏移 (ReinterpretCast<uint64_t> 后) ===
//   [5]  rmt_addr               — 远端目的地址 (数据写入位置)
//   [7]  data_addr (Write SQE)  — 本端源数据地址
//
//   WriteWithNotify SQE 额外字段:
//   [7]  notify_addr            — 通知地址 (远端写完数据后, 往此地址写 notify_data)
//   [8]  notify_data            — 通知数据 (写入 notify_addr 的值)
//   [11] data_addr              — 本端源数据地址
// ============================================================================

// URMA protocol 8bit offset
constexpr uint32_t SQE_COMMON_UINT8_OFFSET_2 = 2U; // udf_flg:1 inline_en:1 cqe:1 se:1 fence:1 odr:3
constexpr uint32_t SQE_COMMON_UINT8_OFFSET_3 = 3U; // owner:1 rmt_jetty_type:2 token_en:1 nf:1 rsv:3
constexpr uint32_t SQE_COMMON_TARGET_HINT_OFFSET = 4U;
constexpr uint32_t SQE_COMMON_OPCODE_OFFSET = 5U;
constexpr uint32_t SQE_COMMON_SGE_NUM_OFFSET = 11U;
constexpr uint32_t CQE_STATUS_OFFSET = 3U;       // CQE 完成状态 (0=成功, 0xff=无效/未完成)
constexpr uint32_t CQE_SUBSTATUS_OFFSET = 2U;   // CQE 子状态
constexpr uint32_t CQE_ENTRY_IDX_HIGH_OFFSET = 5U; // CQE 关联的 WQE entry_idx 高字节
constexpr uint32_t CQE_ENTRY_IDX_LOW_OFFSET = 4U;  // CQE 关联的 WQE entry_idx 低字节

// URMA protocol 32bit offset
constexpr uint32_t SQE_COMMON_UINT32_OFFSET_2 = 2U; // sge_num:8 tp_id:24
constexpr uint32_t SQE_COMMON_RMT_JETTY_OR_SEG_ID_OFFSET = 3U;
constexpr uint32_t SQE_COMMON_RMT_EID_31_0_OFFSET = 4U;
constexpr uint32_t SQE_COMMON_RMT_EID_63_32_OFFSET = 5U;
constexpr uint32_t SQE_COMMON_RMT_EID_95_64_OFFSET = 6U;
constexpr uint32_t SQE_COMMON_RMT_EID_127_96_OFFSET = 7U;
constexpr uint32_t SQE_COMMON_RMT_TOKEN_VALUE_OFFSET = 8U;
constexpr uint32_t SQE_UDF_OFFSET = 9U;
constexpr uint32_t SQE_WITH_NOTIFY_UDF_OFFSET = SQE_UDF_OFFSET;
constexpr uint32_t SQE_LENGTH_OFFSET = 12U;                    // Write SQE 的 length 字段
constexpr uint32_t SQE_WITH_NOTIFY_LENGTH_OFFSET = 20U;        // WriteWithNotify SQE 的 length 字段
constexpr uint32_t SQE_TOKEN_ID_OFFSET = 13U;                    // rsv:12 token_id:20 (Write SQE)
constexpr uint32_t SQE_WITH_NOTIFY_TOKEN_ID_OFFSET = 21U;        // rsv:12 token_id:20 (WriteWithNotify SQE)
constexpr uint32_t SQE_WITH_NOTIFY_NOTIFY_TOKEN_ID_OFFSET = 12U; // rsv:12 notify_token_id:20
constexpr uint32_t SQE_WITH_NOTIFY_NOTIFY_TOKEN_VALUE_OFFSET = 13U;

// URMA protocol 64bit offset
constexpr uint32_t SQE_RMT_ADDR_OFFSET = 5U;                        // 远端目的地址
constexpr uint32_t SQE_WITH_NOTIFY_RMT_ADDR_OFFSET = SQE_RMT_ADDR_OFFSET;
constexpr uint32_t SQE_DATA_ADDR_OFFSET = 7U;                       // Write SQE 的本端源数据地址
constexpr uint32_t SQE_WITH_NOTIFY_DATA_ADDR_OFFSET = 11U;          // WriteWithNotify SQE 的本端源数据地址
constexpr uint32_t SQE_WITH_NOTIFY_NOTIFY_ADDR_OFFSET = 7U;         // 通知地址 (远端)
constexpr uint32_t SQE_WITH_NOTIFY_NOTIFY_DATA_OFFSET = 8U;         // 通知数据

// 结构体定义 (HcclAiRMAWQ / HcclAiRMACQ / HcclCombinOpParam)
// 已移到 moe_distribute_base_types.h, Host 侧和 Kernel 侧共用

// ============================================================================
// URMA通信接口 (与 moe_distribute_base.h 完全一致, 逐字复制)
// ============================================================================

// ---------------------------------------------------------------------------
// GetURMASqInfoGMAddr — 获取指定 dstRank 的 SQ 信息在 GM 中的地址
//
// 参数:
//   hcclContext - HcclCombinOpParam 的 GM 地址
//   dstRankId   - 目标 rank ID
//
// 返回: (__gm__ HcclAiRMAWQ *) 指向 sqs[dstRankId] 的 GM 地址
//
// 用途: kernel 从 GM 读取 SQ 信息到 UB (通过 GetURMASqInfoTensor)
// ---------------------------------------------------------------------------
__aicore__ inline GM_ADDR GetURMASqInfoGMAddr(GM_ADDR hcclContext, uint32_t dstRankId)
{
    ascendc_assert((hcclContext != nullptr && dstRankId < MAX_RANK_NUM),
                   "hcclContext is nullptr or dstRankId >= MAX_RANK_NUM");
    return (GM_ADDR)((__gm__ HcclAiRMAWQ *)(((__gm__ HcclCombinOpParam *)hcclContext)->sqs) + dstRankId);
}

// ---------------------------------------------------------------------------
// GetURMACqInfoGMAddr — 获取指定 dstRank 的 CQ 信息在 GM 中的地址
//
// 返回: (__gm__ HcclAiRMACQ *) 指向 cqs[dstRankId] 的 GM 地址
// ---------------------------------------------------------------------------
__aicore__ inline GM_ADDR GetURMACqInfoGMAddr(GM_ADDR hcclContext, uint32_t dstRankId)
{
    ascendc_assert((hcclContext != nullptr && dstRankId < MAX_RANK_NUM),
                   "hcclContext is nullptr or dstRankId >= MAX_RANK_NUM");
    return (GM_ADDR)((__gm__ HcclAiRMACQ *)(((__gm__ HcclCombinOpParam *)hcclContext)->cqs) + dstRankId);
}

// ---------------------------------------------------------------------------
// GetURMASqInfoTensor — 从 GM 读取 SQ 信息到 UB (DataCopy)
//
// 参数:
//   sqInfoTensor - UB 上的目标 LocalTensor (uint8_t, 大小 >= sizeof(HcclAiRMAWQ))
//   hcclContext  - HcclCombinOpParam 的 GM 地址
//   dstRankId    - 目标 rank ID
//
// 流程: GetURMASqInfoGMAddr → SetGlobalBuffer → DataCopyPad (GM → UB)
// 后续需要 SyncFunc<MTE2_S> 等待 DataCopy 完成后再访问 sqInfoTensor
// ---------------------------------------------------------------------------
__aicore__ inline void GetURMASqInfoTensor(const AscendC::LocalTensor<uint8_t> &sqInfoTensor, GM_ADDR hcclContext,
                                           uint32_t dstRankId)
{
    GM_ADDR sqInfoGMAddr = GetURMASqInfoGMAddr(hcclContext, dstRankId);
    ascendc_assert((sqInfoGMAddr != nullptr), "GetURMASqInfoGMAddr failed");
    AscendC::GlobalTensor<uint8_t> sqInfoGlobalTensor;
    sqInfoGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)(sqInfoGMAddr));
    AscendC::DataCopyExtParams sqInfoParams = {1U, static_cast<uint32_t>(sizeof(HcclAiRMAWQ)), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<uint8_t> sqInfoPadParams{false, 0U, 0U, 0U};
    AscendC::DataCopyPad(sqInfoTensor, sqInfoGlobalTensor, sqInfoParams, sqInfoPadParams);
}

// ---------------------------------------------------------------------------
// GetURMACqInfoTensor — 从 GM 读取 CQ 信息到 UB (DataCopy)
// 同 GetURMASqInfoTensor, 但读取 HcclAiRMACQ 结构体
// ---------------------------------------------------------------------------
__aicore__ inline void GetURMACqInfoTensor(const AscendC::LocalTensor<uint8_t> &cqInfoTensor, GM_ADDR hcclContext,
                                           uint32_t dstRankId)
{
    GM_ADDR cqInfoGMAddr = GetURMACqInfoGMAddr(hcclContext, dstRankId);
    ascendc_assert((cqInfoGMAddr != nullptr), "GetURMACqInfoGMAddr failed");
    AscendC::GlobalTensor<uint8_t> cqInfoGlobalTensor;
    cqInfoGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)(cqInfoGMAddr));
    AscendC::DataCopyExtParams cqInfoParams = {1U, static_cast<uint32_t>(sizeof(HcclAiRMACQ)), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<uint8_t> cqInfoPadParams{false, 0U, 0U, 0U};
    AscendC::DataCopyPad(cqInfoTensor, cqInfoGlobalTensor, cqInfoParams, cqInfoPadParams);
}

// ---------------------------------------------------------------------------
// GenerateCommWriteSQE — 生成 Write SQE 模板 (opcode=0x3, 64B)
//
// 参数:
//   sqeTensor - UB 上的 SQE LocalTensor (uint8_t, 大小 >= WRITE_SQE_SIZE=64B)
//   sqeCount  - SQE 个数 (当前未使用, 保留参数)
//
// 流程:
//   1. Duplicate 初始化全 0 (64B)
//   2. SyncFunc<V_S> 等 Vector 完成
//   3. 填充固定字段:
//      - offset[2] = 0b001     (odr=0b001, 无序)
//      - offset[3] = 0b10110000 (owner=1, rmt_jetty_type=0b01, token_en=1, nf=0)
//      - offset[5] = 0x3       (opcode=0x3, Write 操作)
//      - offset[11] = 1        (sge_num=1, 单个 SGE)
//
// 模板生成后, 还需 UpdateCommWriteSQE 填入 channel 元数据 (tp_id/eid/token),
// 再用 SetCommWriteSQE 填入数据地址和长度, 最后 PutCommSQE 拷到 SQ ring。
// ---------------------------------------------------------------------------
__aicore__ inline void GenerateCommWriteSQE(const AscendC::LocalTensor<uint8_t> &sqeTensor, uint32_t sqeCount = 1)
{
    AscendC::Duplicate<uint8_t>(sqeTensor, 0, WRITE_SQE_SIZE); // 初始化为全0
    AscendC::SyncFunc<AscendC::HardEvent::V_S>();
    sqeTensor(SQE_COMMON_UINT8_OFFSET_2) = 0b001;      // odr=0b001
    sqeTensor(SQE_COMMON_UINT8_OFFSET_3) = 0b10110000; // owner=1, rmt_jetty_type=0b01, token_en=1, nf=0
    sqeTensor(SQE_COMMON_OPCODE_OFFSET) = 0x3;         // opcode=0x3
    sqeTensor(SQE_COMMON_SGE_NUM_OFFSET) = 1;          // sge_num=1
}

// ---------------------------------------------------------------------------
// GenerateCommWriteWithNotifySQE — 生成 WriteWithNotify SQE 模板 (opcode=0x5, 96B)
//
// 与 GenerateCommWriteSQE 的区别:
//   - opcode = 0x5 (WriteWithNotify, 带远端通知)
//   - SQE 大小 = 96B (比 Write SQE 多 notify_addr + notify_data 字段)
//   - 每个 SQE 占 2 个 WQEBB (96B 对齐到 128B)
// ---------------------------------------------------------------------------
__aicore__ inline void GenerateCommWriteWithNotifySQE(const AscendC::LocalTensor<uint8_t> &sqeTensor,
                                                       uint32_t sqeCount = 1)
{
    AscendC::Duplicate<uint8_t>(sqeTensor, 0, WRITE_WITH_NOTIFY_SQE_SIZE); // 初始化为全0
    AscendC::SyncFunc<AscendC::HardEvent::V_S>();
    sqeTensor(SQE_COMMON_UINT8_OFFSET_2) = 0b001;      // odr=0b001
    sqeTensor(SQE_COMMON_UINT8_OFFSET_3) = 0b10110000; // owner=1, rmt_jetty_type=0b01, token_en=1, nf=0
    sqeTensor(SQE_COMMON_OPCODE_OFFSET) = 0x5;         // opcode=0x5，使用write with notify SQE
    sqeTensor(SQE_COMMON_SGE_NUM_OFFSET) = 1;          // sge_num=1
}

// ---------------------------------------------------------------------------
// UpdateCommonSQE — 用 SQ 信息更新 SQE 模板的公共通信元数据
//
// 从 sqInfoTensor (HcclAiRMAWQ) 读取字段, 填入 sqeTensor (SQE 模板):
//   - tp_id (24bit):        从 WQ_TP_ID_OFFSET 读取, 填入 SQE_COMMON_UINT32_OFFSET_2 的低 24bit
//                           (保留高 8bit 的 sge_num 不变)
//   - rmt_jetty_or_seg_id:  从 WQ_RMTOBJID_OFFSET 读取
//   - rmt_eid (128bit):     从 WQ_RMTEID_0_3..12_15_OFFSET 读取 4 个 uint32
//   - rmt_token_value:      从 WQ_RMTTOKENVALUE_OFFSET 读取
//
// 参数:
//   sqeTensor    - UB 上的 SQE 模板 (已由 Generate* 生成)
//   sqInfoTensor - UB 上的 SQ 信息 (已由 GetURMASqInfoTensor 读取)
// ---------------------------------------------------------------------------
__aicore__ inline void UpdateCommonSQE(const AscendC::LocalTensor<uint8_t> &sqeTensor,
                                       const AscendC::LocalTensor<uint8_t> &sqInfoTensor)
{
    // 更新tp_id(24b), jetty_id(20b), rmt_eid(128b), rmt_token_value(32b)

    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint32_t> templateSqeU32 = sqeTensor.ReinterpretCast<uint32_t>();
    templateSqeU32(SQE_COMMON_UINT32_OFFSET_2) &= 0xff000000;                                // 保留sge_num
    templateSqeU32(SQE_COMMON_UINT32_OFFSET_2) |= (sqInfoU32(WQ_TP_ID_OFFSET) & 0x00ffffff); // tp_id
    templateSqeU32(SQE_COMMON_RMT_JETTY_OR_SEG_ID_OFFSET) = sqInfoU32(WQ_RMTOBJID_OFFSET);   // rmt_jetty_or_seg_id
    templateSqeU32(SQE_COMMON_RMT_EID_31_0_OFFSET) = sqInfoU32(WQ_RMTEID_0_3_OFFSET);        // rmt_eid
    templateSqeU32(SQE_COMMON_RMT_EID_63_32_OFFSET) = sqInfoU32(WQ_RMTEID_4_7_OFFSET);
    templateSqeU32(SQE_COMMON_RMT_EID_95_64_OFFSET) = sqInfoU32(WQ_RMTEID_8_11_OFFSET);
    templateSqeU32(SQE_COMMON_RMT_EID_127_96_OFFSET) = sqInfoU32(WQ_RMTEID_12_15_OFFSET);
    templateSqeU32(SQE_COMMON_RMT_TOKEN_VALUE_OFFSET) = sqInfoU32(WQ_RMTTOKENVALUE_OFFSET); // rmt_token_value
}

// ---------------------------------------------------------------------------
// UpdateCommWriteSQE — 更新 Write SQE 的 token_id (在 UpdateCommonSQE 基础上)
//
// 额外填充: rmt_token_id (20bit) 从 WQ_RMTOBJID_OFFSET 读取
// ---------------------------------------------------------------------------
__aicore__ inline void UpdateCommWriteSQE(const AscendC::LocalTensor<uint8_t> &sqeTensor,
                                           const AscendC::LocalTensor<uint8_t> &sqInfoTensor)
{
    UpdateCommonSQE(sqeTensor, sqInfoTensor);

    // 更新rmt_token_id(20b)
    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint32_t> templateSqeU32 = sqeTensor.ReinterpretCast<uint32_t>();
    templateSqeU32(SQE_TOKEN_ID_OFFSET) = sqInfoU32(WQ_RMTOBJID_OFFSET); // rmt_token_id
}

// ---------------------------------------------------------------------------
// UpdateCommWriteWithNotifySQE — 更新 WriteWithNotify SQE 的 token 字段
//
// 在 UpdateCommonSQE 基础上额外填充:
//   - rmt_token_id (20bit):       从 WQ_RMTOBJID_OFFSET 读取 → SQE_WITH_NOTIFY_TOKEN_ID_OFFSET
//   - notify_token_value (32bit): 从 WQ_RMTTOKENVALUE_OFFSET 读取 → SQE_WITH_NOTIFY_NOTIFY_TOKEN_VALUE_OFFSET
//   - notify_token_id (20bit):    从 WQ_RMTOBJID_OFFSET 读取 → SQE_WITH_NOTIFY_NOTIFY_TOKEN_ID_OFFSET
//
// WriteWithNotify SQE 有两套 token:
//   - rmt_token: 数据写的鉴权 token
//   - notify_token: 通知写的鉴权 token
// ---------------------------------------------------------------------------
__aicore__ inline void UpdateCommWriteWithNotifySQE(const AscendC::LocalTensor<uint8_t> &sqeTensor,
                                                     const AscendC::LocalTensor<uint8_t> &sqInfoTensor)
{
    UpdateCommonSQE(sqeTensor, sqInfoTensor);

    // 更新rmt_token_id(20b), notify_token_value(32b), notify_token_id(20b)
    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint32_t> templateSqeU32 = sqeTensor.ReinterpretCast<uint32_t>();
    templateSqeU32(SQE_WITH_NOTIFY_TOKEN_ID_OFFSET) = sqInfoU32(WQ_RMTOBJID_OFFSET); // rmt_token_id
    templateSqeU32(SQE_WITH_NOTIFY_NOTIFY_TOKEN_VALUE_OFFSET) =
        sqInfoU32(WQ_RMTTOKENVALUE_OFFSET);                                                 // notify_token_value
    templateSqeU32(SQE_WITH_NOTIFY_NOTIFY_TOKEN_ID_OFFSET) = sqInfoU32(WQ_RMTOBJID_OFFSET); // notify_token_id
}

// ---------------------------------------------------------------------------
// SetCommWriteSQE — 设置 Write SQE 的数据传输参数
//
// 参数:
//   sqeTensor - UB 上的 SQE 模板 (已由 UpdateCommWriteSQE 填好元数据)
//   dataAddr  - 本端源数据地址 (HBM, 数据从这里读)
//   rmtAddr   - 远端目的地址 (对端 HBM, 数据写到这里)
//   length    - 传输数据长度 (字节)
//   cqe       - 是否生成 CQE (1=生成完成事件, 0=不生成)
//
// 填充字段:
//   - rmt_addr (64bit): SQE_RMT_ADDR_OFFSET
//   - data_addr (64bit): SQE_DATA_ADDR_OFFSET
//   - length (32bit): SQE_LENGTH_OFFSET
//   - cqe/odr 标志位: SQE_COMMON_UINT8_OFFSET_2
//     cqe=1: odr=0b010 (有序, 生成 CQE)
//     cqe=0: odr=0b001 (无序, 不生成 CQE)
// ---------------------------------------------------------------------------
__aicore__ inline void SetCommWriteSQE(const AscendC::LocalTensor<uint8_t> &sqeTensor, uint64_t dataAddr,
                                       uint64_t rmtAddr, uint32_t length, uint8_t cqe)
{
    // 设置cqe(1 bit), odr(3 bits), data_addr(64 bits), rmt_addr(64 bits), length(32 bits)

    AscendC::LocalTensor<uint32_t> sqeLocalU32 = sqeTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> sqeLocalU64 = sqeTensor.ReinterpretCast<uint64_t>();

    sqeLocalU64(SQE_RMT_ADDR_OFFSET) = rmtAddr;
    sqeLocalU64(SQE_DATA_ADDR_OFFSET) = dataAddr;
    sqeLocalU32(SQE_LENGTH_OFFSET) = length;
    if (cqe > 0U) {
        // cqe=1, odr=0b010
        sqeTensor(SQE_COMMON_UINT8_OFFSET_2) |= 0b00100010;
        sqeTensor(SQE_COMMON_UINT8_OFFSET_2) &= 0b11111010;
    } else {
        // cqe=0, odr=0b001
        sqeTensor(SQE_COMMON_UINT8_OFFSET_2) &= 0b11011001;
        sqeTensor(SQE_COMMON_UINT8_OFFSET_2) |= 0b00000001;
    }
}

// ---------------------------------------------------------------------------
// SetCommWriteWithNotifySQE — 设置 WriteWithNotify SQE 的数据传输参数
//
// 参数:
//   sqeTensor  - UB 上的 SQE 模板
//   dataAddr   - 本端源数据地址 (HBM)
//   rmtAddr    - 远端目的地址 (对端 HBM, 数据写到这里)
//   length     - 传输数据长度 (字节)
//   notifyAddr - 通知地址 (远端, 数据写完后往此地址写 notifyData)
//   notifyData - 通知数据 (写入 notifyAddr 的值, 用作完成标志)
//   cqe        - 是否生成 CQE
//
// 填充字段:
//   - rmt_addr:      SQE_WITH_NOTIFY_RMT_ADDR_OFFSET
//   - data_addr:     SQE_WITH_NOTIFY_DATA_ADDR_OFFSET
//   - length:        SQE_WITH_NOTIFY_LENGTH_OFFSET
//   - notify_addr:   SQE_WITH_NOTIFY_NOTIFY_ADDR_OFFSET
//   - notify_data:   SQE_WITH_NOTIFY_NOTIFY_DATA_OFFSET
//   - cqe/odr 标志位: 同 SetCommWriteSQE
// ---------------------------------------------------------------------------
__aicore__ inline void SetCommWriteWithNotifySQE(const AscendC::LocalTensor<uint8_t> &sqeTensor, uint64_t dataAddr,
                                                  uint64_t rmtAddr, uint32_t length, uint64_t notifyAddr,
                                                  uint64_t notifyData, uint8_t cqe)
{
    // 设置cqe(1 bit), odr(3 bits), data_addr(64 bits), rmt_addr(64 bits), length(32 bits), notify_addr(64 bits),
    // notify_data(64 bits)

    AscendC::LocalTensor<uint32_t> sqeLocalU32 = sqeTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> sqeLocalU64 = sqeTensor.ReinterpretCast<uint64_t>();

    sqeLocalU64(SQE_WITH_NOTIFY_RMT_ADDR_OFFSET) = rmtAddr;
    sqeLocalU64(SQE_WITH_NOTIFY_DATA_ADDR_OFFSET) = dataAddr;
    sqeLocalU32(SQE_WITH_NOTIFY_LENGTH_OFFSET) = length;
    sqeLocalU64(SQE_WITH_NOTIFY_NOTIFY_ADDR_OFFSET) = notifyAddr; // notify_addr
    sqeLocalU64(SQE_WITH_NOTIFY_NOTIFY_DATA_OFFSET) = notifyData; // notify_data

    if (cqe > 0U) {
        // cqe=1, odr=0b010
        sqeTensor(SQE_COMMON_UINT8_OFFSET_2) |= 0b00100010;
        sqeTensor(SQE_COMMON_UINT8_OFFSET_2) &= 0b11111010;
    } else {
        // cqe=0, odr=0b001
        sqeTensor(SQE_COMMON_UINT8_OFFSET_2) &= 0b11011001;
        sqeTensor(SQE_COMMON_UINT8_OFFSET_2) |= 0b00000001;
    }
}

// ---------------------------------------------------------------------------
// SendJFSDoorBell — 敲 JFS DoorBell, 通知硬件 SQ 有新 WQE
//
// 参数:
//   jfsDoorBellTensor - UB 上的 DoorBell 缓冲区 (当前未实际使用, 保留参数)
//   sqInfoTensor      - UB 上的 SQ 信息 (从中读取 dbAddr = DoorBell 地址)
//   sqPi              - 当前的 SQ 生产者线性索引 (告诉硬件 "WQE 写到了这里")
//
// 操作: st_dev(sqPi, dbAddr, 0) — Scalar 直接写 device 内存 (DoorBell 寄存器)
// 硬件收到 DoorBell 后, 从 SQ ring 取出新 WQE 执行
// ---------------------------------------------------------------------------
__aicore__ inline void SendJFSDoorBell(const AscendC::LocalTensor<uint8_t> &jfsDoorBellTensor,
                                       const AscendC::LocalTensor<uint8_t> &sqInfoTensor, uint32_t sqPi)
{
    // 敲JFS DoorBell，更新硬件的sqPi
    AscendC::LocalTensor<uint64_t> sqInfoU64 = sqInfoTensor.ReinterpretCast<uint64_t>();
    st_dev(sqPi, (__gm__ uint32_t *)(sqInfoU64(WQ_DBADDR_OFFSET)), 0); // Scalar操作
}

// ---------------------------------------------------------------------------
// SendJFCDoorBell — 敲 JFC DoorBell, 通知硬件 CQE 已被消费
//
// 参数:
//   jfcDoorBellTensor - UB 上的 DoorBell 缓冲区 (当前未实际使用, 保留参数)
//   cqInfoTensor      - UB 上的 CQ 信息 (从中读取 jfcId 和 dbAddr)
//   cqCi              - 当前的 CQ 消费者线性索引 (告诉硬件 "CQE 读到了这里")
//
// DoorBell 值 = (jfcId << 32) | cqCi  (高 32bit = JFC ID, 低 32bit = cqCi)
// 操作: st_dev(jfcDbValue, dbAddr, 0) — Scalar 写 device 内存
// 硬件收到后, 释放对应位置的 CQE 空间, 可以复用
// ---------------------------------------------------------------------------
__aicore__ inline void SendJFCDoorBell(const AscendC::LocalTensor<uint8_t> &jfcDoorBellTensor,
                                       const AscendC::LocalTensor<uint8_t> &cqInfoTensor, uint32_t cqCi)
{
    // 敲JFC DoorBell，更新硬件的cqCi
    AscendC::LocalTensor<uint32_t> cqInfoU32 = cqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> cqInfoU64 = cqInfoTensor.ReinterpretCast<uint64_t>();

    uint64_t jfcDbValue = (static_cast<uint64_t>(cqInfoU32(CQ_JFCID_OFFSET)) << 32) + static_cast<uint64_t>(cqCi);
    st_dev(jfcDbValue, (__gm__ uint64_t *)(cqInfoU64(CQ_DBADDR_OFFSET)), 0); // Scalar操作
}

// ---------------------------------------------------------------------------
// CheckCqeStatus — 检查单个 CQE 的完成状态
//
// 参数:
//   cqInfoTensor      - UB 上的 CQ 信息
//   cqeTensor         - UB 上的 CQE 缓冲区 (读取 8B: status + substatus + entry_idx)
//   outCqCi           - [in/out] 当前 CQ 消费者索引 (环形, 取模 cqDepth)
//   outCqCiLinear     - [in/out] CQ 消费者线性索引 (不取模, 用于 DoorBell)
//   newestCompletedPi - [out] 该 CQE 关联的 WQE 的 sqPi (用于更新 sqCi)
//
// 返回:
//   true  — CQE status==0xff, 表示尚未完成 (无效 CQE), 调用方应停止轮询
//   false — CQE status==0, 表示正常完成, 已处理 (设置 status=0xff 写回, 防止重复处理)
//
// 异常: status!=0 且 !=0xff, ascendc_assert 报错退出
//
// 流程:
//   1. 从 CQ ring 读取 8B CQE (cqGlobalTensor[cqeSize * outCqCi])
//   2. SyncFunc<MTE2_S> 等 DataCopy 完成
//   3. 检查 status:
//      - 0xff: 未完成, 返回 true (调用方 break)
//      - 0:    正常, 从 entry_idx 解析 sqPi → newestCompletedPi
//              设置 status=0xff 写回 CQ ring (标记已处理)
//              SyncFunc<S_MTE3> + SyncFunc<MTE3_MTE2> 确保写回完成
//      - 其他: 异常, ascendc_assert
// ---------------------------------------------------------------------------
__aicore__ inline bool CheckCqeStatus(const AscendC::LocalTensor<uint8_t> &cqInfoTensor,
                                      const AscendC::LocalTensor<uint8_t> &cqeTensor, uint32_t &outCqCi,
                                      uint32_t &outCqCiLinear, uint32_t &newestCompletedPi)
{
    AscendC::LocalTensor<uint32_t> cqInfoU32 = cqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> cqInfoU64 = cqInfoTensor.ReinterpretCast<uint64_t>();

    uint32_t cqeSize = cqInfoU32(CQ_CQESIZE_OFFSET);
    AscendC::GlobalTensor<uint8_t> cqGlobalTensor;
    cqGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)(cqInfoU64(CQ_CQVA_OFFSET)));
    AscendC::DataCopyExtParams cqeParams = {1U, 8U, 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<uint8_t> cqePadParams{false, 0U, 0U, 0U};

    AscendC::DataCopyPad(cqeTensor, cqGlobalTensor[cqeSize * outCqCi], cqeParams, cqePadParams);
    AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>();
    if (cqeTensor(CQE_STATUS_OFFSET) == 0xff) {
        return true;
    } else if (cqeTensor(CQE_STATUS_OFFSET) != 0) {
        // 退出kernel并报错
        ascendc_assert(false, "CQE status is abnormal! status is %d, substatus is %d.\n", cqeTensor(CQE_STATUS_OFFSET),
                       cqeTensor(CQE_SUBSTATUS_OFFSET));
    }

    // status==0，处理当前CQE，从entry_idx获取对应WQE的sqPi
    newestCompletedPi = (static_cast<uint32_t>(cqeTensor(CQE_ENTRY_IDX_HIGH_OFFSET)) << UINT8_BITS_OFFSET) +
                        static_cast<uint32_t>(cqeTensor(CQE_ENTRY_IDX_LOW_OFFSET));
    // 把CQE的status设置为无效值，并写回CQ
    cqeTensor(CQE_STATUS_OFFSET) = 0xff;
    AscendC::SyncFunc<AscendC::HardEvent::S_MTE3>();
    AscendC::DataCopyPad(cqGlobalTensor[cqeSize * outCqCi], cqeTensor, cqeParams);
    AscendC::SyncFunc<AscendC::HardEvent::MTE3_MTE2>(); // 防止cqeTensor被下一轮加载覆盖

    return false;
}

// ---------------------------------------------------------------------------
// PollCommCQUpdateSQCI — 轮询 CQ, 消费已完成 CQE, 更新 sqCi 和 cqCi
//
// 参数:
//   sqInfoTensor       - UB 上的 SQ 信息
//   cqInfoTensor       - UB 上的 CQ 信息
//   cqeTensor          - UB 上的 CQE 缓冲区
//   jfcDoorBellTensor  - UB 上的 DoorBell 缓冲区
//   outSqCi            - [in/out] SQ 消费者索引 (环形, 更新为 newestCompletedPi+1)
//   outCqCi            - [in/out] CQ 消费者索引 (环形, 每消费一个 CQE 递增)
//   outCqCiLinear      - [in/out] CQ 消费者线性索引 (不取模, 用于 DoorBell)
//
// 流程:
//   1. 循环检查 CQE (最多 cqDepth-1 次):
//      - CheckCqeStatus 返回 true (未完成) → break
//      - CheckCqeStatus 返回 false (已完成) → outCqCi++, outCqCiLinear++, pollTimes++
//   2. 如果 pollTimes > 0, 更新 outSqCi = (newestCompletedPi + 1) % sqDepth
//      (sqCi 跳到最后一个完成的 WQE 之后, 释放 SQ 空间)
//   3. SyncFunc<MTE3_S> 等 CQE status 写回完成
//   4. SendJFCDoorBell 敲 JFC DoorBell, 通知硬件 cqCi 更新
//
// 用途:
//   - PutCommNotifySQE 中 SQ 空间不足时调用, 腾出 SQ 空间
//   - Drain 中调用, 等待所有 WQE 完成
//   - InitRankComm 中调用, 清理残余 CQE
// ---------------------------------------------------------------------------
__aicore__ inline void PollCommCQUpdateSQCI(const AscendC::LocalTensor<uint8_t> &sqInfoTensor,
                                            const AscendC::LocalTensor<uint8_t> &cqInfoTensor,
                                            const AscendC::LocalTensor<uint8_t> &cqeTensor,
                                            const AscendC::LocalTensor<uint8_t> &jfcDoorBellTensor, uint32_t &outSqCi,
                                            uint32_t &outCqCi, uint32_t &outCqCiLinear)
{
    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint32_t> cqInfoU32 = cqInfoTensor.ReinterpretCast<uint32_t>();

    uint32_t sqDepth = sqInfoU32(WQ_SQDEPTH_OFFSET);
    uint32_t cqDepth = cqInfoU32(CQ_CQDEPTH_OFFSET);

    uint32_t pollTimes = 0;
    uint32_t newestCompletedPi = 0;
    while (pollTimes < cqDepth - 1) {
        if (CheckCqeStatus(cqInfoTensor, cqeTensor, outCqCi, outCqCiLinear, newestCompletedPi)) {
            break;
        }

        // 递增本地的outCqCi
        outCqCi = (outCqCi + 1) % cqDepth;
        ++outCqCiLinear;
        ++pollTimes;
    }

    // 更新本地的outSqCi
    if (pollTimes > 0) {
        outSqCi = (newestCompletedPi + 1) % sqDepth;
    }
    AscendC::SyncFunc<AscendC::HardEvent::MTE3_S>(); // 等cqe status写回无效值
    // 通过敲JFC DoorBell，更新硬件的cqCi
    SendJFCDoorBell(jfcDoorBellTensor, cqInfoTensor, outCqCiLinear);
}

// ---------------------------------------------------------------------------
// InvalidateCqeStatus — 初始化 CQ ring, 全部 CQE 的 status 设为 0xff (无效)
//
// 参数:
//   cqeInfoTensor - UB 上的 CQ 信息 (读取 cqVA, cqDepth)
//   cqeTensor     - UB 上的 CQE 缓冲区 (用作 Duplicate 源, 然后拷到 GM)
//
// 流程:
//   1. 从 cqInfo 读取 cqVA (CQ ring 地址) 和 cqDepth (CQE 个数)
//   2. cqeNum = min(cqDepth, 256) — UB 一次最多搬运 256 个 CQE
//   3. Duplicate cqeTensor 为全 0xff (UB_ALIGN * cqeNum = 32B * 256 = 8KB)
//   4. SyncFunc<V_MTE3> 等 Vector 完成
//   5. 如果 cqDepth > 256, 分多批 DataCopyPad 到 GM 的 CQ ring
//      (每批 256 个 CQE, 最后一批可能不足 256)
//
// 用途: InitRankComm 中首次通信时调用, 初始化 CQ ring
//       (CQE 初始 status 必须是 0xff=无效, 这样 CheckCqeStatus 才能正确判断)
// ---------------------------------------------------------------------------
__aicore__ inline void InvalidateCqeStatus(const AscendC::LocalTensor<uint8_t> &cqeInfoTensor,
                                           const AscendC::LocalTensor<uint8_t> &cqeTensor)
{
    // 为cqeBuf_申请256*32B空间，初始化HGM上的CQ空间时，如果cqDepth>256，则循环多次DataCopy
    AscendC::LocalTensor<uint64_t> cqInfoU64 = cqeInfoTensor.ReinterpretCast<uint64_t>();
    AscendC::LocalTensor<uint32_t> cqInfoU32 = cqeInfoTensor.ReinterpretCast<uint32_t>();

    AscendC::GlobalTensor<uint8_t> cqGlobalTensor;
    cqGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)(cqInfoU64(CQ_CQVA_OFFSET)));
    uint32_t cqDepth = cqInfoU32(CQ_CQDEPTH_OFFSET);
    uint32_t cqeNum = (cqDepth > CQ_DEPTH_256) ? CQ_DEPTH_256 : cqDepth;

    AscendC::Duplicate<uint8_t>(cqeTensor, 0xff, UB_ALIGN * cqeNum); // 初始化为全1
    AscendC::SyncFunc<AscendC::HardEvent::V_MTE3>();

    AscendC::DataCopyExtParams dataCopyParams = {static_cast<uint16_t>(cqeNum), UB_ALIGN, 0U,
                                                 NORMAL_CQE_SIZE - UB_ALIGN, 0U};
    uint32_t loopTimes = AscendC::Ceil(cqDepth, CQ_DEPTH_256);
    if (loopTimes > 1) {
        for (uint32_t i = 0; i < loopTimes - 1; ++i) {
            AscendC::DataCopyPad(cqGlobalTensor[NORMAL_CQE_SIZE * CQ_DEPTH_256 * i], cqeTensor, dataCopyParams);
        }
        uint32_t tailBlockCqeNum = cqDepth - CQ_DEPTH_256 * (loopTimes - 1);
        if (tailBlockCqeNum != CQ_DEPTH_256) {
            dataCopyParams = {static_cast<uint16_t>(tailBlockCqeNum), UB_ALIGN, 0U, NORMAL_CQE_SIZE - UB_ALIGN, 0U};
        }
    }
    AscendC::DataCopyPad(cqGlobalTensor[NORMAL_CQE_SIZE * CQ_DEPTH_256 * (loopTimes - 1)], cqeTensor, dataCopyParams);
}

// ---------------------------------------------------------------------------
// GetAvailableSpace — 计算 SQ 可用空间 (还能写多少个 WQE)
//
// 参数:
//   sqInfoTensor - UB 上的 SQ 信息 (读取 sqDepth)
//   outSqPi      - [in] SQ 生产者索引 (kernel 写到哪了)
//   outSqCi      - [in] SQ 消费者索引 (硬件处理到哪了)
//
// 返回: 可用 WQE 个数
//
// 算法 (SQ 不会用满, 预留 1 个 WQE 区分满/空):
//   outSqPi == outSqCi:           空 → 可用 sqDepth-1
//   (outSqPi+1) % sqDepth == sqCi: 满 → 可用 0
//   outSqPi > outSqCi:            pi 在前 → 可用 sqDepth - (pi - ci) - 1
//   outSqPi < outSqCi:            pi 在后 (回绕) → 可用 ci - pi - 1
// ---------------------------------------------------------------------------
__aicore__ inline uint32_t GetAvailableSpace(const AscendC::LocalTensor<uint8_t> &sqInfoTensor, uint32_t &outSqPi,
                                             uint32_t &outSqCi)
{
    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();

    // 根据本地的outSqPi outSqCi 计算SQ可用空间大小（SQ不会用满，会预留一个WQE）
    uint32_t sqDepth = sqInfoU32(WQ_SQDEPTH_OFFSET);

    if (outSqPi == outSqCi) {
        return (sqDepth - 1);
    } else if ((outSqPi + 1) % sqDepth == outSqCi) {
        return 0;
    } else if (outSqPi > outSqCi) {
        return (sqDepth - (outSqPi - outSqCi) - 1);
    }
    return (outSqCi - outSqPi - 1);
}

// ---------------------------------------------------------------------------
// PutCommSQE — 将 Write SQE 拷到 SQ ring, 更新 outSqPi (不敲 DoorBell)
//
// 参数:
//   sqInfoTensor      - UB 上的 SQ 信息
//   cqInfoTensor      - UB 上的 CQ 信息 (SQ 空间不足时用于 poll CQ)
//   sqeTensor         - UB 上的 SQE 数据 (已组装好的 WQE)
//   cqeTensor         - UB 上的 CQE 缓冲区 (poll CQ 时用)
//   jfcDoorBellTensor - UB 上的 DoorBell 缓冲区
//   sqeCount          - 要写入的 SQE 个数
//   outSqPi           - [in/out] SQ 生产者索引 (环形)
//   outSqPiLinear     - [in/out] SQ 生产者线性索引 (不取模)
//   outSqCi           - [in/out] SQ 消费者索引 (poll CQ 时更新)
//   outCqCi           - [in/out] CQ 消费者索引
//   outCqCiLinear     - [in/out] CQ 消费者线性索引
//
// 流程:
//   1. 检查 sqeCount < sqDepth
//   2. GetAvailableSpace 计算可用空间
//   3. while (可用空间 < sqeCount): PollCommCQUpdateSQCI 腾空间
//   4. DataCopy SQE 到 SQ ring:
//      - 如果 SQ 尾部空间足够 (sqDepth - outSqPi >= sqeCount): 一次拷贝
//      - 否则: 分两段拷贝 (尾部 + 头部, 环形回绕)
//   5. 更新 outSqPiLinear += sqeCount, outSqPi = outSqPiLinear % sqDepth
//
// 注意: 此函数不敲 JFS DoorBell, 调用方需在适当时机调用 SendJFSDoorBell
//       (批量写模式下攒一批 SQE 后统一敲一次 DoorBell)
// ---------------------------------------------------------------------------
__aicore__ inline void
PutCommSQE(const AscendC::LocalTensor<uint8_t> &sqInfoTensor, const AscendC::LocalTensor<uint8_t> &cqInfoTensor,
           const AscendC::LocalTensor<uint8_t> &sqeTensor, const AscendC::LocalTensor<uint8_t> &cqeTensor,
           const AscendC::LocalTensor<uint8_t> &jfcDoorBellTensor, uint32_t sqeCount, uint32_t &outSqPi,
           uint32_t &outSqPiLinear, uint32_t &outSqCi, uint32_t &outCqCi, uint32_t &outCqCiLinear)
{
    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> sqInfoU64 = sqInfoTensor.ReinterpretCast<uint64_t>();
    uint32_t sqeSize = sqInfoU32(WQ_WQESIZE_OFFSET); // 单个WQEBB占用大小（64B），由HCCL传递
    uint32_t sqDepth = sqInfoU32(WQ_SQDEPTH_OFFSET);
    ascendc_assert(sqeCount < sqDepth, "too many SQE! SQE num[%d] should less than sqDepth[%d].", sqeCount, sqDepth);

    // 计算SQ可用空间大小
    uint32_t availableSpace = GetAvailableSpace(sqInfoTensor, outSqPi, outSqCi);
    while (availableSpace < sqeCount) {
        // 可用空间不足时，轮询CQ，更新本地的cqCi_ sqCi_和硬件的cqCi
        PollCommCQUpdateSQCI(sqInfoTensor, cqInfoTensor, cqeTensor, jfcDoorBellTensor, outSqCi, outCqCi, outCqCiLinear);
        availableSpace = GetAvailableSpace(sqInfoTensor, outSqPi, outSqCi);
    }

    // 可用空间足够时，把WQE拷贝到HBM上的SQ中
    AscendC::GlobalTensor<uint8_t> sqGlobalTensor;
    sqGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)(sqInfoU64(WQ_SQVA_OFFSET)));

    if (likely(sqDepth - outSqPi >= sqeCount)) {
        AscendC::DataCopy(sqGlobalTensor[sqeSize * outSqPi], sqeTensor, WRITE_SQE_SIZE * sqeCount);
    } else {
        uint32_t firstPartSize = sqDepth - outSqPi;
        uint32_t secondPartSize = sqeCount - firstPartSize;

        AscendC::DataCopy(sqGlobalTensor[sqeSize * outSqPi], sqeTensor, WRITE_SQE_SIZE * firstPartSize);
        AscendC::DataCopy(sqGlobalTensor, sqeTensor[WRITE_SQE_SIZE * firstPartSize], WRITE_SQE_SIZE * secondPartSize);
    }

    // 更新本地的outSqPi
    outSqPiLinear += sqeCount;
    outSqPi = outSqPiLinear % sqDepth;
}

// ---------------------------------------------------------------------------
// PutCommNotifySQE — 将 WriteWithNotify SQE 拷到 SQ ring, 更新 outSqPi
//
// 与 PutCommSQE 的区别:
//   - 每个 WriteWithNotify SQE 占 2 个 WQEBB (96B 对齐到 128B = 2×64B)
//   - sqeCount 是 SQE 个数, 实际占用的 WQEBB 数 = sqeCount * 2
//   - DataCopy 用 DataCopyParams (blockCount=sqeCount, blockLen=96B>>5=3)
//   - 环形回绕时, 如果剩余奇数个 WQEBB, 需要特殊处理 (一个 WQE 被切分到尾部和头部)
//
// 环形回绕的 3 种情况:
//   1. 尾部空间够 (sqDepth - outSqPi >= sqeCount*2): 一次拷贝
//   2. 尾部剩余偶数个 WQEBB: 分两段拷贝
//   3. 尾部剩余奇数个 WQEBB: 其中一个 WQE 被切分:
//      - 先拷 sqeSize (64B) 到 sqDepth-1 位置
//      - 再拷剩余 (WRITE_WITH_NOTIFY_SQE_SIZE - sqeSize = 32B) 到 SQ 头部
//      - 然后拷剩余的完整 SQE 从 sqeSize 偏移开始
// ---------------------------------------------------------------------------
__aicore__ inline void
PutCommNotifySQE(const AscendC::LocalTensor<uint8_t> &sqInfoTensor, const AscendC::LocalTensor<uint8_t> &cqInfoTensor,
                 const AscendC::LocalTensor<uint8_t> &sqeTensor, const AscendC::LocalTensor<uint8_t> &cqeTensor,
                 const AscendC::LocalTensor<uint8_t> &jfcDoorBellTensor, uint32_t sqeCount, uint32_t &outSqPi,
                 uint32_t &outSqPiLinear, uint32_t &outSqCi, uint32_t &outCqCi, uint32_t &outCqCiLinear)
{
    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> sqInfoU64 = sqInfoTensor.ReinterpretCast<uint64_t>();
    uint32_t sqeSize = sqInfoU32(WQ_WQESIZE_OFFSET); // 单个WQEBB占用大小（64B），由HCCL传递
    uint32_t sqDepth = sqInfoU32(WQ_SQDEPTH_OFFSET);
    ascendc_assert((sqeCount << 1) < sqDepth, "too many SQE! SQE num[%d] should less than sqDepth[%d].",
                   (sqeCount << 1), sqDepth);

    // 计算SQ可用空间大小
    uint32_t availableSpace = GetAvailableSpace(sqInfoTensor, outSqPi, outSqCi);
    while (availableSpace < (sqeCount << 1)) {
        // 可用空间不足时，轮询CQ，更新本地的cqCi_ sqCi_和硬件的cqCi
        PollCommCQUpdateSQCI(sqInfoTensor, cqInfoTensor, cqeTensor, jfcDoorBellTensor, outSqCi, outCqCi, outCqCiLinear);
        availableSpace = GetAvailableSpace(sqInfoTensor, outSqPi, outSqCi);
    }

    // 可用空间足够时，把WQE拷贝到HBM上的SQ中
    AscendC::GlobalTensor<uint8_t> sqGlobalTensor;
    sqGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)(sqInfoU64(WQ_SQVA_OFFSET)));

    if (likely((sqDepth - outSqPi) >= (sqeCount << 1))) {
        AscendC::DataCopyParams intriParams{static_cast<uint16_t>(sqeCount), WRITE_WITH_NOTIFY_SQE_SIZE >> 5, 0, 1};
        AscendC::DataCopy(sqGlobalTensor[sqeSize * outSqPi], sqeTensor, intriParams);
    } else {
        uint32_t firstPartSize = (sqDepth - outSqPi) >> 1;
        uint32_t secondPartSize = sqeCount - firstPartSize;
        AscendC::DataCopyParams firstPartParams{static_cast<uint16_t>(firstPartSize), WRITE_WITH_NOTIFY_SQE_SIZE >> 5,
                                                0, 1};
        AscendC::DataCopy(sqGlobalTensor[sqeSize * outSqPi], sqeTensor, firstPartParams);

        if (((sqDepth - outSqPi) & 1) == 0) {
            AscendC::DataCopyParams secondPartParams{static_cast<uint16_t>(secondPartSize),
                                                     WRITE_WITH_NOTIFY_SQE_SIZE >> 5, 0, 1};
            AscendC::DataCopy(sqGlobalTensor, sqeTensor[WRITE_WITH_NOTIFY_SQE_SIZE * firstPartSize], secondPartParams);
        } else {
            // 如果剩余奇数个WQEBB，其中一个WQE会被切块分在尾和头
            AscendC::DataCopy(sqGlobalTensor[sqeSize * (sqDepth - 1)],
                              sqeTensor[WRITE_WITH_NOTIFY_SQE_SIZE * firstPartSize], sqeSize);
            AscendC::DataCopy(sqGlobalTensor, sqeTensor[WRITE_WITH_NOTIFY_SQE_SIZE * firstPartSize + sqeSize],
                              WRITE_WITH_NOTIFY_SQE_SIZE - sqeSize);

            AscendC::DataCopyParams secondPartParams{static_cast<uint16_t>(secondPartSize - 1),
                                                     WRITE_WITH_NOTIFY_SQE_SIZE >> 5, 0, 1};
            AscendC::DataCopy(sqGlobalTensor[sqeSize], sqeTensor[WRITE_WITH_NOTIFY_SQE_SIZE * (firstPartSize + 1)],
                              secondPartParams);
        }
    }

    // 更新本地的outSqPi
    outSqPiLinear += (sqeCount << 1);
    outSqPi = outSqPiLinear % sqDepth;
}

// ---------------------------------------------------------------------------
// GetPICI — 从 PICI 窗口读取 SQ/CQ 环形索引 (跨 launch 持久化)
//
// 参数:
//   hcclContext     - HcclCombinOpParam 的 GM 地址
//   curRankId       - 本 rank ID
//   dstRankId       - 目标 rank ID
//   outSqPi         - [out] SQ 生产者索引
//   outSqCi         - [out] SQ 消费者索引
//   outCqPi         - [out] CQ 生产者索引
//   outCqCi         - [out] CQ 消费者索引
//   outSqPiLinear   - [out] SQ 线性计数 (DoorBell 用)
//   outCqCiLinear   - [out] CQ 线性计数 (DoorBell 用)
//
// 地址计算:
//   piCiSpaceGM = windowsOut[curRankId] + WIN_PICI_OFFSET(1MB) + PICI_WIN_SIZE(512B) * dstRankId
//
// 流程:
//   1. DataCacheCleanAndInvalid 刷新 cache line (确保读到最新数据)
//   2. 读取 6 个 uint32: sqPi, sqCi, cqPi, cqCi, sqPiLinear, cqCiLinear
//
// 用途: InitRankComm 开头调用, 恢复上次 launch 结束时的 SQ/CQ 状态
// ---------------------------------------------------------------------------
__aicore__ inline void GetPICI(GM_ADDR hcclContext, uint32_t curRankId, uint32_t dstRankId, uint32_t &outSqPi,
                               uint32_t &outSqCi, uint32_t &outCqPi, uint32_t &outCqCi, uint32_t &outSqPiLinear,
                               uint32_t &outCqCiLinear)
{
    ascendc_assert((hcclContext != nullptr && curRankId < MAX_RANK_NUM && dstRankId < MAX_RANK_NUM),
                   "hcclContext is nullptr or curRankId >= MAX_RANK_NUM or dstRankId >= MAX_RANK_NUM");

    GM_ADDR piCiSpaceGM = (GM_ADDR)(((__gm__ HcclCombinOpParam *)hcclContext)->windowsOut[curRankId] + WIN_PICI_OFFSET +
                                    PICI_WIN_SIZE * dstRankId);
    AscendC::GlobalTensor<uint32_t> piCiGlobalTensor;
    piCiGlobalTensor.SetGlobalBuffer((__gm__ uint32_t *)piCiSpaceGM);
    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        piCiGlobalTensor);

    outSqPi = piCiGlobalTensor(WIN_SQPI_OFFSET);
    outSqCi = piCiGlobalTensor(WIN_SQCI_OFFSET);
    outCqPi = piCiGlobalTensor(WIN_CQPI_OFFSET);
    outCqCi = piCiGlobalTensor(WIN_CQCI_OFFSET);
    outSqPiLinear = piCiGlobalTensor(WIN_SQPILINEAR_OFFSET);
    outCqCiLinear = piCiGlobalTensor(WIN_CQCILINEAR_OFFSET);
}

// ---------------------------------------------------------------------------
// GetIsFirstInComm — 读取是否首次与该 rank 通信
//
// 地址: 同 GetPICI, 读取 WIN_FIRST_TIME_CREATE_WIN_FLAG_OFFSET (offset=6)
// 返回: outIsFirst=true 表示已初始化过 (非首次), false 表示首次
//
// 用途: InitRankComm 中判断是否需要 InvalidateCqeStatus 初始化 CQ ring
// ---------------------------------------------------------------------------
__aicore__ inline void GetIsFirstInComm(GM_ADDR hcclContext, uint32_t curRankId, uint32_t dstRankId, bool &outIsFirst)
{
    ascendc_assert((hcclContext != nullptr && curRankId < MAX_RANK_NUM && dstRankId < MAX_RANK_NUM),
                   "hcclContext is nullptr or curRankId >= MAX_RANK_NUM or dstRankId >= MAX_RANK_NUM");

    GM_ADDR piCiSpaceGM = (GM_ADDR)(((__gm__ HcclCombinOpParam *)hcclContext)->windowsOut[curRankId] + WIN_PICI_OFFSET +
                                    PICI_WIN_SIZE * dstRankId);
    AscendC::GlobalTensor<uint32_t> piCiGlobalTensor;
    piCiGlobalTensor.SetGlobalBuffer((__gm__ uint32_t *)piCiSpaceGM);
    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        piCiGlobalTensor);

    outIsFirst = (piCiGlobalTensor(WIN_FIRST_TIME_CREATE_WIN_FLAG_OFFSET) > 0);
}

// ---------------------------------------------------------------------------
// UpdatePICI — 将 SQ/CQ 环形索引写回 PICI 窗口 (持久化供下次 launch 用)
//
// 参数: 同 GetPICI, 但方向相反 (kernel → PICI 窗口)
//
// 流程:
//   1. 写 6 个 uint32 到 PICI 窗口
//   2. DataCacheCleanAndInvalid 刷新 cache line (确保写回 GM)
//
// 用途: Drain 结束时调用, 保存当前 SQ/CQ 状态
// ---------------------------------------------------------------------------
__aicore__ inline void UpdatePICI(GM_ADDR hcclContext, uint32_t curRankId, uint32_t dstRankId, uint32_t sqPi,
                                  uint32_t sqCi, uint32_t cqPi, uint32_t cqCi, uint32_t sqPiLinear, uint32_t cqCiLinear)
{
    ascendc_assert((hcclContext != nullptr && curRankId < MAX_RANK_NUM && dstRankId < MAX_RANK_NUM),
                   "hcclContext is nullptr or curRankId >= MAX_RANK_NUM or dstRankId >= MAX_RANK_NUM");

    GM_ADDR piCiSpaceGM = (GM_ADDR)(((__gm__ HcclCombinOpParam *)hcclContext)->windowsOut[curRankId] + WIN_PICI_OFFSET +
                                    PICI_WIN_SIZE * dstRankId);
    AscendC::GlobalTensor<uint32_t> piCiGlobalTensor;
    piCiGlobalTensor.SetGlobalBuffer((__gm__ uint32_t *)piCiSpaceGM);

    piCiGlobalTensor(WIN_SQPI_OFFSET) = sqPi;
    piCiGlobalTensor(WIN_SQCI_OFFSET) = sqCi;
    piCiGlobalTensor(WIN_CQPI_OFFSET) = cqPi;
    piCiGlobalTensor(WIN_CQCI_OFFSET) = cqCi;
    piCiGlobalTensor(WIN_SQPILINEAR_OFFSET) = sqPiLinear;
    piCiGlobalTensor(WIN_CQCILINEAR_OFFSET) = cqCiLinear;
    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        piCiGlobalTensor);
}

// ---------------------------------------------------------------------------
// UpdateIsFirstInComm — 更新首次通信标志
//
// isFirst=true: 设置为 1 (已初始化)
// isFirst=false: 设置为 0 (首次)
//
// 用途: InitRankComm 中 InvalidateCqeStatus 后调用, 标记已初始化
// ---------------------------------------------------------------------------
__aicore__ inline void UpdateIsFirstInComm(GM_ADDR hcclContext, uint32_t curRankId, uint32_t dstRankId, bool isFirst)
{
    ascendc_assert((hcclContext != nullptr && curRankId < MAX_RANK_NUM && dstRankId < MAX_RANK_NUM),
                   "hcclContext is nullptr or curRankId >= MAX_RANK_NUM or dstRankId >= MAX_RANK_NUM");

    GM_ADDR piCiSpaceGM = (GM_ADDR)(((__gm__ HcclCombinOpParam *)hcclContext)->windowsOut[curRankId] + WIN_PICI_OFFSET +
                                    PICI_WIN_SIZE * dstRankId);
    AscendC::GlobalTensor<uint32_t> piCiGlobalTensor;
    piCiGlobalTensor.SetGlobalBuffer((__gm__ uint32_t *)piCiSpaceGM);

    piCiGlobalTensor(WIN_FIRST_TIME_CREATE_WIN_FLAG_OFFSET) = isFirst ? 1 : 0;
    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        piCiGlobalTensor);
}

#endif // MOE_DISTRIBUTE_BASE_NATIVE_H
