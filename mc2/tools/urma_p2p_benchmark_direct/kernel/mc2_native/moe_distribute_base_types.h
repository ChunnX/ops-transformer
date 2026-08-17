#ifndef MOE_DISTRIBUTE_BASE_TYPES_H
#define MOE_DISTRIBUTE_BASE_TYPES_H

// ============================================================================
// moe_distribute_base_types.h — HcclCombinOpParam 结构体 + ChannelEntity 偏移常量
//
// 纯 C++ 结构体和常量, 不含 kernel API, Host 侧和 Kernel 侧共用。
// 从 moe_distribute_base_native.h 和 aiv_urma_channel_decode.h 抽出,
// 避免 host 编译器引入 kernel_operator.h。
//
// Host 侧 (test_urma_p2p_bench.cpp) 的 build_hccl_context() 用此头文件:
//   1. 读取 device 上的 ChannelEntity / SqContext / CqContext / RegedBuffer
//   2. 用 CH_OFF_* / SQ_OFF_* / CQ_OFF_* / RB_OFF_* 偏移解码字段
//   3. 填入 HcclAiRMAWQ / HcclAiRMACQ / HcclCombinOpParam
//   4. 拷到 device 供 kernel 使用
//
// Kernel 侧 (mc2_native_kernel.h) 通过 HcclCombinOpParam 访问 SQ/CQ/PICI。
// ============================================================================

#include <cstdint>
#include <cstring>

// ============================================================================
// 通信规模上限
// ============================================================================

constexpr uint32_t MAX_RANK_NUM = 64U; // 最大卡数 (sqs[MAX_RANK_NUM] / cqs[MAX_RANK_NUM])
constexpr uint32_t MAX_OP_NUM = 8U;    // MC2 最大通信算子数 (opType / algorithmType)

// ============================================================================
// PICI 窗口布局 (Producer-Consumer Index)
//
// PICI 窗口位于 windowsOut[myRank] + WIN_PICI_OFFSET 处,
// 每个 peer 占 PICI_WIN_SIZE 字节, 用于跨 kernel launch 持久化 SQ/CQ 环形索引。
//
// 内存布局:
//   windowsOut[myRank] + WIN_PICI_OFFSET + PICI_WIN_SIZE * dstRankId
//                         │                   │
//                         │ 1MB 偏移           │ 每个 peer 512B
//                         ▼                   ▼
//   [winOutDev][...1MB...][PICI for peer0][PICI for peer1]...
// ============================================================================

constexpr uint64_t WIN_PICI_OFFSET = 1024U * 1024U; // PICI 区域在 windowsOut 中的偏移 (1MB)
constexpr uint64_t PICI_WIN_SIZE = 512UL;            // 每个 peer 的 PICI 窗口大小 (512B)

// ============================================================================
// ChannelEntity (256B) 字段偏移
//
// ChannelEntity 是 HCCL HcclChannelAcquire 返回的 device 结构体,
// 包含 local/remote buffer 指针、SQ/CQ context 指针。
// Host 侧 build_hccl_context() 通过 aclrtMemcpy D2H 读取并解码。
//
// 内存布局 (256B):
//   [0..31]   保留
//   [32..35]  localBufferNum  (本端 RegedBuffer 个数)
//   [36..39]  remoteBufferNum (对端 RegedBuffer 个数)
//   [40..63]  保留
//   [64..71]  localBufferPtr  (本端 RegedBuffer 数组起始地址, 可能是相对偏移)
//   [72..79]  remoteBufferPtr (对端 RegedBuffer 数组起始地址)
//   [80..87]  sqContextPtr    (SqContext 128B 结构体地址)
//   [88..95]  cqContextPtr    (CqContext 128B 结构体地址)
//   [96..255] 保留
// ============================================================================

constexpr uint32_t CH_OFF_LOCAL_BUFFER_NUM  = 32U; // 本端 RegedBuffer 个数
constexpr uint32_t CH_OFF_REMOTE_BUFFER_NUM = 36U; // 对端 RegedBuffer 个数
constexpr uint32_t CH_OFF_LOCAL_BUFFER_PTR  = 64U; // 本端 RegedBuffer 数组指针
constexpr uint32_t CH_OFF_REMOTE_BUFFER_PTR = 72U; // 对端 RegedBuffer 数组指针
constexpr uint32_t CH_OFF_SQ_CONTEXT_PTR    = 80U; // SqContext 指针
constexpr uint32_t CH_OFF_CQ_CONTEXT_PTR    = 88U; // CqContext 指针

// ============================================================================
// SqContext (128B) 字段偏移 — SQ (Submission Queue) 上下文
//
// SQ 是生产者环形队列, kernel 往里写 WQE (Work Queue Entry),
// 硬件 URMA 引擎从里面取 WQE 执行远程写操作。
//
// 内存布局 (128B):
//   [0..7]    保留
//   [8..15]   sqVA       (SQ ring 在 device 上的起始地址, WQE 数组)
//   [16..31]  保留
//   [32..39]  dbAddr     (JFS DoorBell 地址, 敲它通知硬件 sqPi 更新)
//   [40..43]  保留
//   [44..47]  wqeSize    (单个 WQEBB 大小, 通常 64B)
//   [48..51]  sqDepth    (SQ ring 深度, WQEBB 个数, 通常 256)
//   [52..55]  tp_id      (Traffic Profile ID, URMA 路由用)
//   [56..71]  remoteEID  (远端 Endpoint ID, 128bit = 16B)
//   [72..127] 保留
// ============================================================================

constexpr uint32_t SQ_OFF_SQVA       = 8U;  // SQ ring 起始地址 (64bit)
constexpr uint32_t SQ_OFF_DBVA       = 32U; // JFS DoorBell 地址 (64bit)
constexpr uint32_t SQ_OFF_WQESIZE    = 44U; // 单个 WQEBB 大小 (32bit, 通常 64B)
constexpr uint32_t SQ_OFF_SQDEPTH    = 48U; // SQ 深度 (32bit, 通常 256)
constexpr uint32_t SQ_OFF_TPID       = 52U; // Traffic Profile ID (32bit)
constexpr uint32_t SQ_OFF_REMOTE_EID = 56U; // 远端 EID (128bit = 16B)

// ============================================================================
// CqContext (128B) 字段偏移 — CQ (Completion Queue) 上下文
//
// CQ 是消费者环形队列, 硬件 URMA 引擎处理完 WQE 后往里写 CQE (Completion Queue Entry),
// kernel 轮询 CQE 确认写操作完成。
//
// 内存布局 (128B):
//   [0..7]    保留
//   [8..15]   cqVA       (CQ ring 在 device 上的起始地址, CQE 数组)
//   [16..31]  保留
//   [32..39]  dbAddr     (JFC DoorBell 地址, 敲它通知硬件 cqCi 更新)
//   [40..43]  保留
//   [44..47]  cqeSize    (单个 CQE 大小, 通常 64B)
//   [48..51]  cqDepth    (CQ ring 深度, CQE 个数, 通常 256)
//   [52..127] 保留
// ============================================================================

constexpr uint32_t CQ_OFF_CQVA    = 8U;  // CQ ring 起始地址 (64bit)
constexpr uint32_t CQ_OFF_DBVA    = 32U; // JFC DoorBell 地址 (64bit)
constexpr uint32_t CQ_OFF_CQESIZE = 44U; // 单个 CQE 大小 (32bit, 通常 64B)
constexpr uint32_t CQ_OFF_CQDEPTH = 48U; // CQ 深度 (32bit, 通常 256)

// ============================================================================
// RegedBufferEntity (64B) 字段偏移 — 注册的缓冲区实体
//
// HCCL HcclCommMemReg 注册的内存块元数据,
// 包含 buffer 地址、大小、token 信息 (用于 URMA 鉴权)。
//
// 内存布局 (64B):
//   [0..7]    保留
//   [8..15]   addr         (buffer 起始地址, 即 workspace 地址)
//   [16..23]  size         (buffer 大小)
//   [24..27]  保留
//   [28..31]  ubTokenId    (UB Token ID, 填入 SQE 的 rmt_token_id)
//   [32..35]  ubTokenValue (UB Token Value, 填入 SQE 的 rmt_token_value)
//   [36..63]  保留
// ============================================================================

constexpr uint32_t RB_OFF_ADDR         = 8U;  // buffer 起始地址 (64bit)
constexpr uint32_t RB_OFF_SIZE         = 16U; // buffer 大小 (64bit)
constexpr uint32_t RB_OFF_UB_TOKEN_ID  = 28U; // UB Token ID (32bit)
constexpr uint32_t RB_OFF_UB_TOKEN_VAL = 32U; // UB Token Value (32bit)

// ============================================================================
// 结构体大小 (用于 aclrtMemcpy 读取)
// ============================================================================

constexpr uint32_t CHANNEL_ENTITY_SIZE = 256U; // ChannelEntity 大小
constexpr uint32_t SQ_CONTEXT_SIZE     = 128U; // SqContext 大小
constexpr uint32_t CQ_CONTEXT_SIZE     = 128U; // CqContext 大小
constexpr uint32_t REGED_BUFFER_SIZE   = 64U;  // RegedBufferEntity 大小

// ============================================================================
// HcclAiRMAWQ — SQ (Submission Queue) 信息
//
// 从 SqContext 解码后填入, kernel 侧用于:
//   - sqVA: 定位 SQ ring 地址, DataCopy WQE 到此处
//   - dbAddr: JFS DoorBell 地址, SendJFSDoorBell() 敲它通知硬件
//   - wqeSize/sqDepth: 计算 ring 环形回绕和可用空间
//   - tp_id/rmtEid/rmtObjId/rmtTokenValue: 填入 WQE 的通信路由元数据
// ============================================================================

struct HcclAiRMAWQ {
    uint32_t jettyId;       // Jetty ID (本端通信端点标识)
    uint64_t sqVA;          // SQ ring 在 HBM 上的起始地址 (WQE 数组)
    uint32_t wqeSize;       // 单个 WQEBB 占用内存大小 (通常 64B)
    uint32_t sqDepth;       // SQ ring 深度 (可用 WQEBB 个数, 通常 256)
    uint64_t headAddr;      // AIV 无依赖 (硬件维护)
    uint64_t tailAddr;      // AIV 无依赖 (硬件维护)
    uint64_t dbAddr;        // JFS DoorBell 地址 (敲它更新硬件 sqPi)
    uint32_t tp_id;         // Traffic Profile ID (URMA 路由用, 24bit)
    uint8_t rmtEid[16];     // 远端 Endpoint ID (128bit, URMA 寻址用)
    uint32_t rmtObjId;      // 远端 Token ID (填入 SQE 的 rmt_token_id)
    uint32_t rmtTokenValue; // 远端 Token Value (填入 SQE 的 rmt_token_value)
    uint32_t localTokenId;  // 本端 Token ID
};

// ============================================================================
// HcclAiRMACQ — CQ (Completion Queue) 信息
//
// 从 CqContext 解码后填入, kernel 侧用于:
//   - cqVA: 定位 CQ ring 地址, DataCopy 读取 CQE
//   - dbAddr: JFC DoorBell 地址, SendJFCDoorBell() 敲它通知硬件 cqCi 更新
//   - cqeSize/cqDepth: 计算 ring 环形回绕和轮询范围
// ============================================================================

struct HcclAiRMACQ {
    uint32_t jfcId;    // JFC ID (填入 JFC DoorBell 值的高 32bit)
    uint64_t cqVA;     // CQ ring 在 HBM 上的起始地址 (CQE 数组)
    uint32_t cqeSize;  // 单个 CQE 占用内存大小 (通常 64B)
    uint32_t cqDepth;  // CQ ring 深度 (CQE 个数, 通常 256)
    uint64_t headAddr; // 硬件维护
    uint64_t tailAddr; // 硬件维护
    uint64_t dbAddr;   // JFC DoorBell 地址 (敲它更新硬件 cqCi)
};

// ============================================================================
// HcclCombinOpParam — 完整的通信上下文 (Host 侧构造, 拷到 device)
//
// 这是 kernel 侧访问的顶层结构体, 包含:
//   - rank 信息 (rankId / rankDim)
//   - workspace (本端 URMA 目标缓冲区)
//   - windowsIn[peer]: 各 peer 的 workspace 地址 (URMA 写入目的地址)
//   - windowsOut[myRank]: 本 rank 的 PICI 窗口地址 (跨 launch 持久化 SQ/CQ 索引)
//   - sqs[peer]: 每个 peer 的 SQ 信息 (HcclAiRMAWQ)
//   - cqs[peer]: 每个 peer 的 CQ 信息 (HcclAiRMACQ)
//
// Host 侧 build_hccl_context() 从 ChannelEntity 解码后填充此结构体,
// 然后 aclrtMemcpy H2D 拷到 device, kernel 通过 __gm__ HcclCombinOpParam* 访问。
// ============================================================================

struct HcclCombinOpParam {
    uint64_t workSpace;                // client 和 server 之间通信的地址 (workspace 起始)
    uint64_t workSpaceSize;            // workspace 大小
    uint32_t rankId;                   // 当前卡 rankId
    uint32_t rankDim;                  // 总卡数 (world size)
    uint64_t winSize;                  // ccu 不使用
    uint64_t windowsIn[MAX_RANK_NUM];  // 各 peer 的 workspace 地址 (URMA 写入目的地址)
    uint64_t windowsOut[MAX_RANK_NUM]; // 本 rank 的窗口地址 (PICI 持久化区)

    // for ccu (本直调版不使用)
    uint64_t xnAddr;  // Xn 寄存器起始地址
    uint64_t ckeAddr; // CKE 寄存器起始地址
    uint64_t msAddr;  // MS 地址, 预留
    uint64_t msSize;  // 可写的 MS 个数, 预留

    uint32_t opType[MAX_OP_NUM];       // 操作类型
    uint8_t algorithmType[MAX_OP_NUM]; // 算法类型

    HcclAiRMAWQ sqs[MAX_RANK_NUM]; // 每个 peer 的 SQ 信息
    HcclAiRMACQ cqs[MAX_RANK_NUM]; // 每个 peer 的 CQ 信息
};

#endif // MOE_DISTRIBUTE_BASE_TYPES_H
