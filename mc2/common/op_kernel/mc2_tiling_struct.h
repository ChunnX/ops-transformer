/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mc2_tiling_struct.h
 * \brief
 */
#ifndef MC2_TILING_STRUCT_H
#define MC2_TILING_STRUCT_H

#include "kernel_tiling/kernel_tiling.h"

#ifndef MC2_DFX_ENABLE
#define MC2_DFX_ENABLE 1
#endif

namespace Mc2Tiling {
constexpr uint8_t COMM_ALG_DEFAULT = 0;
constexpr uint8_t COMM_ALG_FULL_MESH = 1;
constexpr uint8_t COMM_ALG_DOUBLE_RING = 2;
constexpr uint8_t COMM_ALG_SWITCH_WING = 3;
constexpr uint32_t DOUBLE_RING_FACTOR = 2;
constexpr int64_t KVALUE_MIN = 256;
constexpr int64_t KVALUE_MAX = 65535;

struct MC2ServerCfg {        // server 通用参数
    uint32_t verseion;       // tiling结构体版本号
    uint8_t debugMode;       // 调测模式
    uint8_t sendArgIndex;    // 发送数据参数索引，对应算子原型的参数顺序
    uint8_t recvArgIndex;    // 接收数据参数索引，对应算子原型的参数顺序
    uint8_t commOutArgIndex; // 通信输出参数索引，对应算子原型的参数顺序
    uint8_t reserved[8];     // 保留字段
}; // 16 bytes

struct MC2HcommCfg {
    uint8_t skipLocalRankCopy; // 跳过本卡拷贝，在通信结果只需要给MC2内部计算使用或者本卡拷贝由aicore完成时，
                               // 可以跳过本卡数据send-recv搬运
    uint8_t skipBufferWindowCopy; // 跳过hbm到window间搬运 0 不跳过， 1 跳过snd-window， 2 跳过 window-rcv
    uint8_t stepSize;             // 通信步长，粗粒度融合时填0
                                  // 细粒度融合时连续计算stepsize块数据再commit或wait通信
    char reserved[13];            // 保留字段,前面是3字节，为字节对齐，预留13字节
    char groupName[128];          // groupName， 128是通信域名称限制
    char algConfig[128];          // 算法配置， 128是算法名称限制
    uint32_t opType;              // tiling结构体版本号
    uint32_t reduceType;          // reduce类型
    uint32_t srcDataType;         // 输入数据类型
    uint32_t dstDataType;         // 输出数据类型
}; // 280 bytes

struct Mc2Msg {               // 同aicpu_hccl_def KFCTilingData
    uint32_t preparePosition; // 任务准备位置：0表示在host完成所有通信任务的准备，1表示在kernel侧完成
    uint64_t sendOff;      // 发送数据地址偏移，count * dataTypeSize
    uint64_t recvOff;      // 接收数据地址偏移, count * dataTypeSize
    uint64_t tailSendOff;  // 尾块发送数据地址偏移，count * dataTypeSize
    uint64_t tailRecvOff;  // 尾块接收数据地址偏移, count * dataTypeSize
    uint64_t sendCnt;      // 整块发送数据个数
    uint64_t recvCnt;      // 整块接收数据个数
    uint64_t tailSendCnt;  // 尾块发送数据个数
    uint64_t tailRecvCnt;  // 尾块接收数据个数
    uint64_t totalCnt;     // 总数据个数
    uint32_t turnNum;      // 总轮次
    uint32_t tailNum;      // 总轮次
    uint32_t stride;       // 跳写间隔
    uint32_t workspaceOff; // 使用workspace作为recvbuf时的workspace偏移
    uint32_t notifyOff;    // device notify write/read value偏移

    uint16_t notifyBeginCnt; // notift write value的使用个数
    uint16_t notifyEndCnt;   // notift write value的使用个数
    uint8_t useBufferType;   // recvBuf类型
    uint8_t funID;           // function ID
    uint8_t dataType;        // hccl 数据类型
    uint8_t groupNum;        // groupNum

    uint8_t reuseMode;  // 不复用填turnNum，内存优化选择复用的内存块个数
    uint8_t commType;   // 通信类型
    uint8_t reduceOp;   // reduce op type
    uint8_t commOrder;  // 通信顺序，0表示通信在前，1表示通信在后
    uint8_t waitPolicy; // 等待任务启动的阻塞策略，2、首轮等待，1、每轮等待。
                        // KFC根据此标记在主流任务前面加wait，AIC需要按策略发对应record才能触发执行
    uint8_t rspPolicy; // 任务执行结束时的响应策略， 2、最后通知一次，
                       // 1、每轮通知一次。KFC根据此标记在主流任务后面加record
    uint8_t exitPolicy; // 退出策略，0，一次通信任务下发完成直接退出；1. 通信任务执行完成退出；2.
                        // 等待AIC通知退出(可以多次执行任务)。
    uint8_t commAlg;         // 用于指定具体通信算法。
                             // 0：default, 1：fullmesh, 2：doublering, 3：switching
    uint8_t taskType;        // 从参数获取通信任务，直接下发。AIC自己发Record激活
    uint8_t debugMode;       // 调测模式
                             // 1:单独执行CUBE
                             // 2:单独执行Vector
                             // 4:单独执行AICPU KFC算子
                             // 8:KFC等待通信结束
                             // 16:KFC统计各阶段耗时
    uint8_t stepSize;        // 用于指定通算频率步长
    uint8_t sendArgIndex;    // 发送数据参数索引，对应算子原型的参数顺序
    uint8_t recvArgIndex;    // 接收数据参数索引，对应算子原型的参数顺序
    uint8_t commOutArgIndex; // 通信输出参数索引，对应算子原型的参数顺序
    uint8_t hasCommOut;      // 是否有通信输出
    uint8_t reserve;
    uint32_t reserve2;
};

struct RCSTiling {
    uint32_t rankDim;
    uint32_t rankID;
    uint32_t commtype;
    uint32_t subtype;
    uint32_t tileCnt;
    uint32_t tailM;
    uint32_t tailCnt;
    uint32_t biasLen;
    uint32_t isAdd;
    uint32_t rankM;
    uint32_t rankN;
    uint32_t rankK;
    uint32_t gatherIndex;
    uint32_t isTransposeA;
    uint32_t isTransposeB;
    uint32_t storageGather;
    uint64_t nd2NzWorkLen;
    uint64_t cToFloatLen;
    uint64_t gatherLen;
    uint32_t workspaceAddr4;
    uint32_t aicCoreNum;
    uint32_t needUbBuffer;
    uint32_t addX3UbCnt;
    uint32_t commWorkSpaceSize;
    uint32_t isInputCommQuantScale;
    uint32_t dataType;
    uint32_t commInt8WorkSpace;
    uint32_t dynamicQuantTempBuffSize;
};

struct TileL2Tiling {
    uint32_t mL2TileCnt;
    uint32_t nL2TileCnt;
    uint32_t mTileBlocks;
    uint32_t nTileBlocks;
    uint32_t mTailBlocks;
    uint32_t nTailBlocks;
    uint32_t rankTileNum;
    uint32_t calcOrder;
    uint32_t enableL2Tile;
};

struct TileInfo {
    uint32_t mTailCnt;
    uint32_t nTailCnt;
    uint32_t kTailCnt;
    uint32_t isHf32;
};

struct MC2MatmulV3TilingData {
    TCubeTiling matmulTiling;
    uint32_t mTailCnt;
    uint32_t nTailCnt;
    uint32_t kTailCnt;
    uint32_t mBaseTailSplitCnt;
    uint32_t nBaseTailSplitCnt;
    uint32_t mTailMain;
    uint32_t nTailMain;
    uint32_t isHf32;
    uint32_t aswWindowLen;
};

// ==================== Workspace 段布局信息 ====================
 	 
constexpr uint32_t MAX_WORKSPACE_SEGMENTS = 8U;
 
// Workspace 段类型
enum WorkspaceSegType : uint8_t {
    WS_SEG_LIB_API       = 0,   // 系统库预留空间
    WS_SEG_ND2NZ         = 1,   // ND2NZ 格式转换空间
    WS_SEG_GATHER        = 2,   // AllGather 通信结果空间
    WS_SEG_GATHER_SCALE1 = 3,   // AllGather scale1 空间 (FP8 perblock)
    WS_SEG_GATHER_SCALE  = 4,   // AllGather scale 空间
    WS_SEG_COMM_OUT      = 5,   // AlltoAll 通信结果空间
    WS_SEG_PERMUTE_OUT   = 6,   // AlltoAll 重排空间
    WS_SEG_BIAS          = 7,   // bias 空间
    WS_SEG_MM_RESULT     = 8,   // matmul 结果暂存空间
    WS_SEG_RECV_BUF      = 9,   // ReduceScatter 接收缓冲 (a2a 路径)
    WS_SEG_COMM_INT8     = 10,  // int8 通信 workspace
    WS_SEG_DYNAMIC_QUANT = 11,  // 动态量化临时空间
    WS_SEG_RUNTIME_INFO  = 12,  // 运行时信息段 (64KB)
    WS_SEG_MM_WORKSPACE  = 13,  // matmul 计算临时 workspace
    WS_SEG_RESERVED      = 255,
};
 
// 单个段描述 (24 bytes)
struct WorkspaceSegInfo {
    uint64_t offset;      // 相对 workspace 基地址的偏移 (字节)
    uint64_t size;        // 该段大小 (字节)
    uint8_t  type;        // WorkspaceSegType
    uint8_t  reserved[7]; // 对齐
};
 
// Workspace 布局信息 (208 bytes)
struct WorkspaceLayoutInfo {
    uint64_t totalSize;                                // workspace 总大小
    uint32_t segCount;                                 // 实际段数
    uint32_t reserved;                                 // 对齐
    WorkspaceSegInfo segments[MAX_WORKSPACE_SEGMENTS]; // 段数组 (按 offset 升序)
};
 
// ==================== Peermem 段布局信息 ====================
 
constexpr uint32_t MAX_PEERMEM_SEGMENTS = 8U;
 
// Peermem 段类型
enum PeermemSegType : uint8_t {
    PEERMEM_SEG_DATA     = 0,   // A data 区
    PEERMEM_SEG_SCALE    = 1,   // A scale 区
    PEERMEM_SEG_BIAS     = 2,   // bias 区 (预留)
    PEERMEM_SEG_FLAG     = 3,   // flag/counter 区 (预留)
    PEERMEM_SEG_RESERVED = 255,
};
 
// 单个段描述 (24 bytes)
struct PeermemSegInfo {
    uint64_t offset;      // 相对 peermem 窗口基地址的偏移 (字节)
    uint64_t size;        // 该段大小 (字节)
    uint8_t  type;        // PeermemSegType
    uint8_t  reserved[7]; // 对齐
};
 
// Peermem 布局信息 (208 bytes)
struct PeermemLayoutInfo {
    uint64_t totalSize;                                // peermem 总大小 (== peermemDataSize)
    uint32_t segCount;                                 // 实际段数
    uint32_t reserved;                                 // 对齐
    PeermemSegInfo segments[MAX_PEERMEM_SEGMENTS];     // 段数组 (按 offset 升序)
};
 
// ==================== Runtime Info ====================
 
constexpr uint32_t RUNTIME_INFO_CORE_MAX = 128U;       // 最大核数
constexpr uint32_t RUNTIME_INFO_PER_CORE_SIZE = 512U;  // 每核 512 字节
constexpr uint32_t RUNTIME_INFO_TOTAL_SIZE =
    RUNTIME_INFO_CORE_MAX * RUNTIME_INFO_PER_CORE_SIZE; // 64KB
 
// 通信阶段枚举：仅 HCCL 五态状态机
enum RuntimePhase : uint8_t {
    RT_PHASE_COMM_INIT      = 0,   // HCCL Init: 通信初始化
    RT_PHASE_COMM_PREPARE   = 1,   // HCCL Prepare: 通信准备 (AllGather/AlltoAll prepare)
    RT_PHASE_COMM_COMMIT    = 2,   // HCCL Commit: 通信提交 (下发描述符并激活)
    RT_PHASE_COMM_WAIT      = 3,   // HCCL Wait: 等待通信完成 (hccl.Wait)
    RT_PHASE_COMM_FINALIZE  = 4,   // HCCL Finalize: 结束清理
    RT_PHASE_RESERVED       = 255,
};
 
// 每个 core 的运行时信息 (512 字节)
struct alignas(512) RuntimeInfoPerCore {
    uint32_t magicNum;        // 4B  魔数 0x5A5A5A5A
    uint16_t coreId;          // 2B  core ID (C核: 0~aicCoreNum-1, V核: aicCoreNum~total-1)
    uint8_t  execTurn;        // 1B  执行轮次: scale=1轮, 每 tile=1轮, 连续递增 (0=pipeline未启动)
    uint8_t  execPosition;    // 1B  position tag: 0=COMM_BEFORE, 1=SCALE_COMM_BEFORE,
                            //     2=COMP_CUBE_MATMUL_BEFORE, 3=COMP_VEC_PERMUTE_BEFORE
    uint8_t  commPhase;       // 1B  当前通信阶段 (RuntimePhase, 仅 HCCL 五态)
    uint8_t  commCommitCount; // 1B  HCCL Commit 次数 (B层 communicator 递增)
    uint8_t  commWaitCount;   // 1B  HCCL Wait 次数 (B层 communicator 递增)
    uint8_t  reserved[501];   // 501B 预留
}; // 512 bytes
} // namespace Mc2Tiling
#endif // MC2_TILING_STRUCT_H
