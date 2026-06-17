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
 * \file matmul_reduce_scatter_aiv_mode_util.h
 * \brief MatmulReduceScatterV2算子的AIV模式工具类和通信基础设施
 *
 * 本文件定义了Matmul+ReduceScatter融合算子的核心通信基础设施，包括：
 *   - 常量定义：分块大小、对齐粒度、flag偏移等
 *   - BaseBlock模板：通用的内存对齐计算工具（32B/256B/512B对齐）
 *   - CommBase类：通信基类，封装peermem管理、跨rank同步、flag机制、DMA搬运等
 *   - CommColumnSplitter类：小M场景的列分块调度器，按列方向切分通信任务
 *   - GetSwizzledBlockIdx函数：Swizzle分块调度，优化L1 Cache命中率
 */
#ifndef MATMUL_REDUCE_SCATTER_AIV_MODE_UTIL
#define MATMUL_REDUCE_SCATTER_AIV_MODE_UTIL
#include "matmul_reduce_scatter_v2_aiv_mode_tiling.h"

using namespace AscendC;
using namespace matmulReduceScatterV2_aivmode_tiling;

namespace matmulReduceScatterV2_util {
// Padding参数传递宏：将所有padding参数打包传递给PaddingRunner::Run
#define PADDING_ARGS_CALL()                                                                                            \
    transA, transB, alignedA, alignedB, matrixAM, matrixAK, matrixBK, matrixBN, matrixAMAlign, matrixAKAlign,          \
        matrixBKAlign, matrixBNAlign, gmA, gmB, gmAAlign, gmBAlign

// Dequant参数传递宏：将所有反量化参数打包传递给DequantRunner::RunMatmulReduceScatter
#define DEQUANT_ARGS_CALL()                                                                                            \
    rowNum, colNum, perChannelScale, perTokenScale, workspace, reinterpret_cast<GM_ADDR>(peerMem),                     \
        reinterpret_cast<GM_ADDR>(output), reinterpret_cast<GM_ADDR>(biasptr), tileM0, tileN0, pValue, swizzlDirect,   \
        swizzlCount, coreIdx, coreNum, rankIdx, rankSize, calIdx, resource, needPerChannel, needPerToken

// Dequant函数参数声明宏：定义RunMatmulReduceScatter的完整参数列表
#define DEQUANT_ARGS_FUN()                                                                                             \
    uint32_t rowNum, uint32_t colNum, __gm__ float32_t *perChannelScale, __gm__ float32_t *perTokenScale,              \
        __gm__ int32_t *workspace, GM_ADDR peerMem, GM_ADDR output, GM_ADDR biasptr, uint32_t tileM0,                  \
        uint32_t tileN0, uint32_t pValue, uint32_t swizzlDirect, uint32_t swizzlCount, uint32_t coreIdx,               \
        uint32_t coreNum, uint32_t rankIdx, uint32_t rankSize, uint32_t calIdx,                                        \
        Arch::Resource<Arch::AtlasA2> resource, bool needPerChannel = false, bool needPerToken = false

constexpr int32_t MAX_BLOCK_COUNT = 2;          // 大M场景的pingpong缓冲区数量（双缓冲）
constexpr int32_t MAX_BLOCK_COUNT_SM = 4;       // 小M场景的pingpong缓冲区数量（4缓冲）
constexpr int32_t FLAG_ZERO_IDX = 0;            // IPC flag索引0：用于CrossRankSyncV1
constexpr int32_t FLAG_ONE_IDX = 1;             // IPC flag索引1：用于CrossRankSyncV2
constexpr int32_t FLAG_VALUE = 1;               // Flag增量值
constexpr int32_t FLAG_OFFSET = 180 * 1024 * 1024 / sizeof(int32_t); // Flag在peermem中的偏移（180MB之后）
constexpr int32_t USED_UB_SIZE = 160 * 1024;    // 实际使用的UB大小（160KB）
constexpr int32_t AIC_WAIT_AIV_FINISH_ALIGN_FLAG_ID = 12; // AIC等待AIV完成对齐的flag ID
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024; // 系统预留workspace大小：16MB
constexpr uint32_t BLOCK_SIZE_16 = 16;          // 16字节对齐粒度
constexpr uint32_t BASE_BLOCK_SIZE_32 = 32;     // 32字节基础块大小
constexpr uint32_t BASE_BLOCK_SIZE_256 = 256;   // 256字节基础块大小
constexpr uint32_t BASE_BLOCK_SIZE_512 = 512;   // 512字节基础块大小
constexpr uint32_t TILE_SHAPE_64 = 64;          // 64形状分块
constexpr uint32_t TILE_SHAPE_128 = 128;        // 128形状分块
constexpr uint32_t TILE_SHAPE_256 = 256;        // 256形状分块
constexpr uint32_t TILE_SHAPE_512 = 512;        // 512形状分块
constexpr uint32_t UB_BUFFER_NUM = 2;           // UB双缓冲数量
constexpr uint32_t AIV_CROSS_CORE_SYNC_MODE = 0;  // AIV跨核同步模式
constexpr uint32_t AIC_CROSS_CORE_SYNC_MODE = 2;  // AIC跨核同步模式
constexpr uint32_t RAND_BASE = 3;               // Flag值的随机基数，防止与初始值0混淆

// BaseBlock: 通用内存对齐计算模板，提供Count/IsAligned/AlignUp/AlignDown等工具方法
// T: 元素类型, SIZE: 对齐字节数（必须为2的幂）
template <typename T, size_t SIZE>
struct BaseBlock {
    static_assert((SIZE & (SIZE - 1)) == 0, "Invalid block size");
    static constexpr size_t size = SIZE / sizeof(T);  // 以元素为单位的对齐粒度

    // Count: 计算长度len需要对齐到size的块数（向上取整）
    static inline __aicore__ size_t Count(size_t len)
    {
        return (len + size - 1) / size;
    }

    // IsAligned: 判断长度len是否已对齐到size
    static inline __aicore__ bool IsAligned(size_t len)
    {
        return len % size == 0;
    }

    // AlignUp: 将长度len向上对齐到size的整数倍
    static inline __aicore__ size_t AlignUp(size_t len)
    {
        return (len + size - 1) & ~(size - 1);
    }

    // AlignDown: 将长度len向下对齐到size的整数倍
    static inline __aicore__ size_t AlignDown(size_t len)
    {
        return len & ~(size - 1);
    }
};

// Block32B: 32字节对齐计算工具
template <typename T>
using Block32B = BaseBlock<T, BASE_BLOCK_SIZE_32>;

// Block256B: 256字节对齐计算工具
template <typename T>
using Block256B = BaseBlock<T, BASE_BLOCK_SIZE_256>;

// Block512B: 512字节对齐计算工具，用于A/B矩阵的DMA对齐
template <typename T>
using Block512B = BaseBlock<T, BASE_BLOCK_SIZE_512>;

// CeilDev: 向上取整除法，计算ceil(num/div)
__aicore__ inline int32_t CeilDev(int32_t num, int32_t div)
{
    if (div == 0) {
        return 0;
    }
    return (num + div - 1) / div;
}

// GetSwizzledBlockIdx: 将线性loop索引通过Swizzle映射转换为2D的M/N分块坐标
// 支持两种Swizzle方向：Zn（按行分组，偶数组正向、奇数组反向）和Nz（按列分组）
// Swizzle调度可以改善L1 Cache命中率，减少cache冲突
__aicore__ inline void GetSwizzledBlockIdx(int32_t loop_idx, int32_t m_loop, int32_t n_loop, int32_t swizzl_direction,
                                   int32_t swizzl_count, int64_t &m_idx, int64_t &n_idx)
{
    uint32_t in_batch_idx = loop_idx % (m_loop * n_loop);
    if (swizzl_direction == 0) { // Zn
        uint32_t tile_block_loop = (m_loop + swizzl_count - 1) / swizzl_count;
        uint32_t tile_block_idx = in_batch_idx / (swizzl_count * n_loop);
        uint32_t in_tile_block_idx = in_batch_idx % (swizzl_count * n_loop);

        uint32_t n_row = swizzl_count;
        if (tile_block_idx == tile_block_loop - 1) {
            n_row = m_loop - swizzl_count * tile_block_idx;
        }
        m_idx = tile_block_idx * swizzl_count + in_tile_block_idx % n_row;
        n_idx = in_tile_block_idx / n_row;
        if (tile_block_idx % UB_BUFFER_NUM != 0) {
            n_idx = n_loop - n_idx - 1;
        }
    } else if (swizzl_direction == 1) { // Nz
        uint32_t tile_block_loop = (n_loop + swizzl_count - 1) / swizzl_count;
        uint32_t tile_block_idx = in_batch_idx / (swizzl_count * m_loop);
        uint32_t in_tile_block_idx = in_batch_idx % (swizzl_count * m_loop);

        uint32_t n_col = swizzl_count;
        if (tile_block_idx == tile_block_loop - 1) {
            n_col = n_loop - swizzl_count * tile_block_idx;
        }
        m_idx = in_tile_block_idx / n_col;
        n_idx = tile_block_idx * swizzl_count + in_tile_block_idx % n_col;
        if (tile_block_idx % UB_BUFFER_NUM != 0) {
            m_idx = m_loop - m_idx - 1;
        }
    }
}

// CommBase: 通信基类，封装MatmulReduceScatterV2算子的通信基础设施
// 提供peermem管理、跨rank同步、flag机制、DMA搬运等通用功能
// 所有AIV模式实现（大M/小M）均继承此类
class CommBase {
public:
    __aicore__ explicit CommBase(){};

    // SetArgs: 解析tiling数据并初始化通信参数
    // 包括：获取block_id/core_idx、解析HCCL上下文获取rank/peermem地址、计算分块循环次数、初始化UB缓冲区
    __aicore__ inline void SetArgs(MatmulReduceScatterV2AivModeTilingData tilingData)
    {
        block_id = AscendC::GetBlockIdx();
        core_num = AscendC::GetBlockNum();
        aiv_idx = GetSubBlockIdx();
        core_idx = block_id / GetTaskRation();

        SetTiling(tilingData);
        if (is910C) {
            __gm__ HcclOpResParam *winContext_{nullptr};
            auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
            winContext_ = (__gm__ HcclOpResParam *)contextGM0;
            rank = winContext_->localUsrRankId;
            rank_size = winContext_->rankSize;
            for (int i = 0; i < rank_size; i++) {
                buff[i] =
                    (GM_ADDR)((i == rank) ?
                                  winContext_->localWindowsIn :
                                  ((HcclRankRelationResV2 *)(winContext_->remoteRes[i].nextDevicePtr))->windowsIn);
            }
        } else {
            __gm__ HcclA2CombineOpParam *winContext_{nullptr};
            auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
            winContext_ = (__gm__ HcclA2CombineOpParam *)contextGM0;
            rank = winContext_->rankId;
            rank_size = winContext_->rankNum;
            for (int i = 0; i < rank_size; i++) {
                buff[i] = (GM_ADDR)winContext_->windowsIn[i];
            }
        }
        k_loop = CeilDev(k, k0);
        n_loop = CeilDev(n, n0);
        int32_t tail_m = (m / rank_size) % m0;
        m_loop = m / rank_size / m0;
        if (tail_m) {
            m_loop += 1;
        }
        m_loop *= rank_size;
        nAlign16 = n % BLOCK_SIZE_16 == 0;
        loop_num_per_comm = p_value * core_num;
        other_rank = (core_idx < rank_size) ? core_idx : -1;
        gm_c_pingpong_size = m0 * n0 * loop_num_per_comm;
        if ASCEND_IS_AIV {
            TPipe pipe;
            pipe.InitBuffer(uBuf_, USED_UB_SIZE);
        }
    }

    // SetTiling: 从tiling数据结构中提取分块参数和矩阵信息
    __aicore__ inline void SetTiling(MatmulReduceScatterV2AivModeTilingData &tilingData)
    {
        m0 = tilingData.cocTiling.m0;
        k0 = tilingData.cocTiling.k0;
        n0 = tilingData.cocTiling.n0;
        swizzl_count = tilingData.cocTiling.swizzlCount;
        swizzl_direct = tilingData.cocTiling.swizzlDirect;
        p_value = tilingData.cocTiling.pValue;
        max_ub_single_dma_size = tilingData.cocTiling.ubMoveNum;
        comm_npu_split = tilingData.cocTiling.commNpuSplit;
        comm_data_split = tilingData.cocTiling.commDataSplit;
        len_per_loop = tilingData.cocTiling.lenPerLoop;

        m = tilingData.matmulReduceScatterV2AivModeInfo.M;
        k = tilingData.matmulReduceScatterV2AivModeInfo.K;
        n = tilingData.matmulReduceScatterV2AivModeInfo.N;
        aAlignSize = tilingData.matmulReduceScatterV2AivModeInfo.aAlignSize;
        bAlignSize = tilingData.matmulReduceScatterV2AivModeInfo.bAlignSize;
        dequantSize = tilingData.matmulReduceScatterV2AivModeInfo.dequantSize;
        hasAAlign = tilingData.matmulReduceScatterV2AivModeInfo.hasAAlign;
        hasBAlign = tilingData.matmulReduceScatterV2AivModeInfo.hasBAlign;
        dequant_type = tilingData.matmulReduceScatterV2AivModeInfo.dequant_type;
        is910C = tilingData.matmulReduceScatterV2AivModeInfo.is910C;
        isX2ScaleTypeInt64 = tilingData.matmulReduceScatterV2AivModeInfo.isX2ScaleTypeInt64;
    }

    // AlignJudge: 判断A/B矩阵是否需要对齐padding
    // 根据转置方向和矩阵维度判断：非转置A看K是否对齐，非转置B看N是否对齐
    __aicore__ inline void AlignJudge(bool trans_a, bool trans_b, int32_t m, int32_t k, int32_t n, int32_t m_align,
                                      int32_t k_align, int32_t n_align, int32_t &aligned_a, int32_t &aligned_b)
    {
        if (!trans_a) {
            aligned_a = k != k_align;
        } else {
            aligned_a = (m != m_align && m != 1);
        }

        if (!trans_b) {
            aligned_b = (n != n_align);
        } else {
            aligned_b = (k != k_align);
        }
    }

    // ResetIpcFlags: 重置本rank peermem中的IPC通信flag为0（大M场景使用）
    // num_flags: 需要重置的flag数量
    __aicore__ inline void ResetIpcFlags(int32_t num_flags)
    {
        for (int32_t idx = 0; idx < num_flags; ++idx) {
            if (core_idx == 0 && aiv_idx == 0) {
                SetBuffFlag((__gm__ int32_t *)buff[rank] + FLAG_OFFSET + idx, 0);
            }
        }
    }

    // ResetIpcFlags重载: 重置本rank peermem中的IPC通信flag为指定值（小M场景使用）
    // num_flags: 需要重置的flag数量, val: 重置的目标值
    __aicore__ inline void ResetIpcFlags(int32_t num_flags, int32_t val)
    {
        if (core_idx == 0 && aiv_idx == 1) {
            for (int32_t idx = 0; idx < num_flags; ++idx) {
                SetBuffFlag((__gm__ int32_t *)buff[rank] + FLAG_OFFSET + idx, val);
            }
        }
    };

    // SetBuffFlag: 设置peermem中指定地址的flag值（覆盖写入）
    // 通过UB中转写入，添加RAND_BASE偏移防止与初始值0混淆
    __aicore__ inline void SetBuffFlag(__gm__ int32_t *buff, int32_t flag)
    {
        SetFlag<HardEvent::MTE3_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID2);
        LocalTensor<int32_t> ubTensor = uBuf_.AllocTensor<int32_t>();
        ubTensor(0) = flag + RAND_BASE;
        SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
        CopyUbufToGmAlignB16(buff, ubTensor, 1, sizeof(int32_t), 0, 0);
    }

    // SetBuffFlagByAdd: 通过原子加操作设置peermem中的flag值（累加写入）
    // 用于多rank同时设置同一flag的场景，如CrossRankSync
    __aicore__ inline void SetBuffFlagByAdd(__gm__ int32_t *buff, int32_t flag)
    {
        PipeBarrier<PIPE_ALL>();
        LocalTensor<int32_t> ubTensor = uBuf_.AllocTensor<int32_t>();
        ubTensor(0) = flag;
        PipeBarrier<PIPE_ALL>();
        SetAtomicAdd<int32_t>();
        PipeBarrier<PIPE_ALL>();
        CopyUbufToGmAlignB16(buff, ubTensor, 1, sizeof(int32_t), 0, 0);
        PipeBarrier<PIPE_ALL>();
        SetAtomicNone();
        PipeBarrier<PIPE_ALL>();
    }

    // CheckBuffFlag: 轮询等待peermem中指定地址的flag值等于目标值（精确匹配）
    // 用于等待远端rank设置flag，实现跨rank同步
    __aicore__ inline void CheckBuffFlag(__gm__ int32_t *buff, int32_t flag)
    {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        LocalTensor<int32_t> ubTensor = uBuf_.AllocTensor<int32_t>();
        while (true) {
            CopyGmToUbufAlignB16(ubTensor, buff, 1, sizeof(int32_t), 0, 0);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3); // Scalar等MTE2
            if (ubTensor(0) == flag + RAND_BASE) {
                break;
            }
        }
        uBuf_.FreeTensor<int32_t>(ubTensor);
    }

    // CheckBuffFlagV2: 轮询等待peermem中指定地址的flag值大于目标值（大于匹配）
    // 用于小M场景中等待多个rank累加flag后超过阈值
    __aicore__ inline void CheckBuffFlagV2(__gm__ int32_t *buff, int32_t flag)
    {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        LocalTensor<int32_t> ubTensor = uBuf_.AllocTensor<int32_t>();

        while (true) {
            CopyGmToUbufAlignB16(ubTensor, buff, 1, sizeof(int32_t), 0, 0);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID3); // Scalar等MTE2
            if (ubTensor(0) > flag + RAND_BASE) {
                uint32_t cut_flag = ubTensor(0);
                break;
            }
        }
        uBuf_.FreeTensor<int32_t>(ubTensor);
    }

    // CopyGmToUbufAlignB16: GM→UB数据拷贝（支持非32B对齐，使用DataCopyPad补齐）
    template <typename T>
    __aicore__ inline void CopyGmToUbufAlignB16(LocalTensor<T> ubTensor, __gm__ T *src, uint16_t nBurst,
                                                uint32_t lenBurst, uint16_t srcStride, uint16_t dstStride)
    {
        DataCopyExtParams dataCopyParams(nBurst,    // blockCount
                                         lenBurst,  // blockLen
                                         srcStride, // srcStride
                                         dstStride, // dstStride
                                         0);
        GlobalTensor<T> gmTensor;
        gmTensor.SetGlobalBuffer(src);
        DataCopyPadExtParams<T> padParams;
        DataCopyPad(ubTensor, gmTensor, dataCopyParams, padParams);
    }

    // CopyUbufToGmAlignB16: UB→GM数据拷贝（支持非32B对齐，使用DataCopyPad）
    template <typename T>
    __aicore__ inline void CopyUbufToGmAlignB16(__gm__ T *dst, LocalTensor<T> ubTensor, uint16_t nBurst,
                                                uint32_t lenBurst, uint16_t srcStride, uint16_t dstStride)
    {
        DataCopyExtParams dataCopyParams(nBurst,    // blockCount
                                         lenBurst,  // blockLen
                                         srcStride, // srcStride
                                         dstStride, // dstStride
                                         0);
        GlobalTensor<T> gmTensor;
        gmTensor.SetGlobalBuffer(dst);
        DataCopyPad(gmTensor, ubTensor, dataCopyParams);
    }

    // CopyGmToUbuf: GM→UB数据拷贝（要求32B对齐，使用DataCopy）
    template <typename T>
    __aicore__ inline void CopyGmToUbuf(LocalTensor<T> ubTensor, __gm__ T *src, uint16_t nBurst, uint32_t lenBurst,
                                        uint16_t srcStride, uint16_t dstStride)
    {
        DataCopyParams dataCopyParams(nBurst,    // blockCount
                                      lenBurst,  // blockLen
                                      srcStride, // srcStride
                                      dstStride  // dstStride
        );
        GlobalTensor<T> gmTensor;
        gmTensor.SetGlobalBuffer(src);
        DataCopy(ubTensor, gmTensor, dataCopyParams);
    }

    // CopyUbufToGm: UB→GM数据拷贝（要求32B对齐，使用DataCopy）
    template <typename T>
    __aicore__ inline void CopyUbufToGm(__gm__ T *dst, LocalTensor<T> ubTensor, uint16_t nBurst, uint16_t lenBurst,
                                        uint16_t srcStride, uint16_t dstStride)
    {
        DataCopyParams dataCopyParams(nBurst,    // blockCount
                                      lenBurst,  // blockLen
                                      srcStride, // srcStride
                                      dstStride  // dstStride
        );
        GlobalTensor<T> gmTensor;
        gmTensor.SetGlobalBuffer(dst);
        DataCopy(gmTensor, ubTensor, dataCopyParams);
    }

    // CopyUbufToGmUnknown: UB→GM数据拷贝（自动选择对齐/非对齐方式）
    // 根据nAlign16标志选择使用DataCopy（32B对齐）或DataCopyPad（非对齐）
    template <typename T>
    __aicore__ inline void CopyUbufToGmUnknown(bool nAlign16, __gm__ T *dst, LocalTensor<T> ubTensor, uint16_t nBurst,
                                               uint32_t lenBurst, uint16_t srcStride, uint16_t dstStride)
    {
        if (nAlign16) {
            CopyUbufToGm(dst, ubTensor, nBurst, lenBurst / 32, srcStride, dstStride / 32);
        } else {
            CopyUbufToGmAlignB16(dst, ubTensor, nBurst, lenBurst, srcStride, dstStride);
        }
    }
    // FFTSCrossCoreSync: 跨核同步原语，设置指定pipe和模式的flag
    template <pipe_t pipe, uint64_t mode>
    inline __aicore__ void FFTSCrossCoreSync(uint64_t flag_id)
    {
        AscendC::CrossCoreSetFlag<mode, pipe>(flag_id);
    }

    // SetAndWaitAivSync: AIV内部同步，设置flag并等待所有AIV核心到达
    // pipe_depth: 流水线深度偏移，默认为2，避免与AIC同步flag冲突
    __aicore__ inline void SetAndWaitAivSync(uint64_t flag_idx, int32_t pipe_depth = 2)
    {
        FFTSCrossCoreSync<PIPE_MTE3, AIV_CROSS_CORE_SYNC_MODE>(flag_idx + pipe_depth);
        WaitEvent(flag_idx + pipe_depth);
    }

    // SetAicSync: 通知AIC核心可以开始下一轮计算（AIV→AIC同步）
    __aicore__ inline void SetAicSync(uint64_t flag_idx)
    {
        FFTSCrossCoreSync<PIPE_MTE3, AIC_CROSS_CORE_SYNC_MODE>(flag_idx);
    }

    // CrossRankSyncV1: 跨rank同步第一阶段——通知其他rank本rank数据已准备好
    // 本rank设置自己的flag，其他rank轮询检查各自的flag
    __aicore__ inline void CrossRankSyncV1(int32_t flag_idx, int32_t flag_data)
    {
        if (core_idx == rank && aiv_idx == 0) {
            SetBuffFlagByAdd((__gm__ int32_t *)buff[rank] + FLAG_OFFSET + flag_idx, FLAG_VALUE);
        } else if (core_idx < rank_size && aiv_idx == 0) {
            CheckBuffFlag((__gm__ int32_t *)buff[core_idx] + FLAG_OFFSET + flag_idx, FLAG_VALUE * flag_data);
        }
    }

    // CrossRankSyncV2: 跨rank同步第二阶段——确认所有rank都已完成读取
    // 每个rank设置其他rank的flag，本rank等待所有rank设置自己的flag
    __aicore__ inline void CrossRankSyncV2(int32_t flag_idx, int32_t flag_data)
    {
        if (aiv_idx == 0 && core_idx < rank_size) {
            SetBuffFlagByAdd((__gm__ int32_t *)buff[core_idx] + FLAG_OFFSET + flag_idx, FLAG_VALUE);
        }
        if (aiv_idx == 0 && core_idx == rank) {
            CheckBuffFlag((__gm__ int32_t *)buff[rank] + FLAG_OFFSET + flag_idx, FLAG_VALUE * rank_size * flag_data);
        }
    }

public:
    int32_t block_id;              // 当前block的全局ID
    int32_t core_num;              // 总block数
    int32_t aiv_idx;               // 当前AIV子块索引（0或1，用于AIC+AIV混合核中的AIV标识）
    int32_t core_idx;              // 当前逻辑核索引（block_id / GetTaskRation()）
    int32_t rank;                  // 当前卡在本通信域中的rank ID
    int32_t rank_size;             // 通信域大小（总卡数）
    GM_ADDR buff[8];               // 各rank的peermem地址数组（buff[i]为第i个rank的通信窗口）
    TBuf<AscendC::TPosition::VECCALC> uBuf_; // UB缓冲区管理器

    int32_t m0;                    // M维度L1分块大小
    int32_t k0;                    // K维度L1分块大小
    int32_t n0;                    // N维度L1分块大小
    int32_t swizzl_count;          // Swizzle块数量
    int32_t swizzl_direct;         // Swizzle方向（0=Zn, 1=Nz）
    int32_t p_value;               // 每次通信前AIC计算的块数
    int32_t max_ub_single_dma_size; // UB单次DMA最大搬运大小
    int32_t comm_npu_split;        // 通信NPU切分策略
    int32_t comm_data_split;       // 通信数据切分策略
    int32_t len_per_loop;          // 每次循环处理的数据长度

    int32_t m;                     // 矩阵M维度（行数/token数）
    int32_t k;                     // 矩阵K维度（内积维度）
    int32_t n;                     // 矩阵N维度（输出维度）

    int32_t m_loop;                // M维度分块循环次数
    int32_t k_loop;                // K维度分块循环次数
    int32_t n_loop;                // N维度分块循环次数

    int32_t loop_num_per_comm;     // 每次通信之间的计算循环数 = p_value * core_num
    int32_t other_rank;            // 当前核负责通信的对端rank（-1表示不负责通信）
    int32_t max_ub_ping_pong_size; // UB双缓冲中单个ping/pong的大小
    GM_ADDR aGM_;                  // A矩阵GM地址
    GM_ADDR bGM_;                  // B矩阵GM地址
    GM_ADDR cGM_;                  // C矩阵（输出）GM地址
    GM_ADDR biasGM_;               // Bias GM地址
    GM_ADDR perChannelScaleGM_;    // Per-channel反量化scale GM地址
    GM_ADDR perTokenScaleGM_;      // Per-token反量化scale GM地址
    uint64_t aAlignSize;           // A矩阵对齐padding大小
    uint64_t bAlignSize;           // B矩阵对齐padding大小
    uint64_t dequantSize;          // 反量化中间结果大小
    bool hasAAlign;                // A矩阵是否需要padding
    bool hasBAlign;                // B矩阵是否需要padding
    bool nAlign16;                 // N维度是否16对齐
    int32_t gm_c_pingpong_size;    // GM上C矩阵pingpong缓冲区大小
    DequantType dequant_type;      // 反量化类型
    bool is910C;                   // 是否为910C芯片
    bool isX2ScaleTypeInt64;       // x2Scale是否为int64类型
};

// CommColumnSplitter: 小M场景的列分块调度器
// 将通信任务按列方向（N维度）切分，每次通信处理不同的列块范围
// 适用于小M大N的矩阵形状，减少单次通信的数据量
class CommColumnSplitter {
private:
    int32_t m_batchSize;       // 一次任务的批量计算块数（原loopNumPerComm）
    int32_t m_unitBlocksM;     // 单位任务块的行数（原mLoops）
    int32_t m_unitBlocksN;     // 单位任务块的列数（原nLoops）
    int32_t m_totalUnitBlocks; // 单位任务块的总数（m_unitBlocksM * m_unitBlocksN）
    int32_t m_n_max;           // N维度的最大列块数

public:
    // 构造函数：初始化列分块调度器的参数
    __aicore__ explicit CommColumnSplitter(int32_t batchSize, int32_t unitBlocksM, int32_t unitBlocksN,
                                                   int32_t n_max)
        : m_batchSize(batchSize), m_unitBlocksM(unitBlocksM), m_unitBlocksN(unitBlocksN),
          m_totalUnitBlocks(unitBlocksM * unitBlocksN), m_n_max(n_max)
    {
    }

    /**
     * 计算指定任务索引在n维度上的起始位置
     * @param taskIndex 任务索引（从0开始）
     * @return n维度上的起始位置
     */
    __aicore__ inline int32_t GetCulumnStartPos(int32_t taskIndex) const
    {
        // 计算覆盖的单位任务块数量
        int32_t coveredUnitBlocks = (taskIndex * m_batchSize) / m_totalUnitBlocks;

        // 计算并返回n维度上的起始位置
        return coveredUnitBlocks * m_unitBlocksN;
    }

    // GetCulumnEndPos: 计算指定任务索引在n维度上的结束位置（不含）
    __aicore__ inline int32_t GetCulumnEndPos(int32_t taskIndex) const
    {
        // 计算覆盖的单位任务块数量
        int32_t coveredUnitBlocks = ((taskIndex + 1) * m_batchSize) / m_totalUnitBlocks;
        int32_t ret = coveredUnitBlocks * m_unitBlocksN;
        // 计算并返回n维度上的起始位置
        return ret < m_n_max ? ret : m_n_max;
    }

    // getBatchSize: 获取一次任务的批量计算块数
    __aicore__ inline int32_t getBatchSize() const
    {
        return m_batchSize;
    }

    // getUnitBlocksM: 获取M维度的单位任务块数
    __aicore__ inline int32_t getUnitBlocksM() const
    {
        return m_unitBlocksM;
    }

    // getUnitBlocksN: 获取N维度的单位任务块数
    __aicore__ inline int32_t getUnitBlocksN() const
    {
        return m_unitBlocksN;
    }
};

} // namespace matmulReduceScatterV2_util
#endif