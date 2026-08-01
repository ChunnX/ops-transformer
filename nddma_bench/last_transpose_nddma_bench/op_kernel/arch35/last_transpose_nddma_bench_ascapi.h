/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file last_transpose_nddma_bench_ascapi.h
 * \brief LastTranspose [0,1,3,2] NDDMA benchmark kernel —— NDDMA 指令层实验变体。
 *
 *   隔离实验（不改 R-054 原头 last_transpose_nddma_bench.h、不动主线共用头）：
 *   把核心的"UB 内 [dim2,dim3]->[dim3,dim2] NDDMA 转置读"从高层模板
 *   DataCopy<T,2,dmaConfig> 降到 bisheng 指令层 intrinsic。
 *
 *   ★ 为何是指令层 intrinsic 而非 asc_* 公开 c_api：
 *     asc SDK 的 c_api（asc_ndim_copy_gm2ub 等）与本算子依赖的 basic_api
 *     （kernel_operator.h / DataCopy）fp8 类型系统互斥、无法共存于同一编译单元
 *     （c_api 把 fp8_e5m2_t 定义为 struct，basic_api 定义为 using = float8_*，
 *      且 dav-3510 后端拒绝 c_api 的 fp8 标量传值）。故降到 asc_* 的下一层——
 *     二者最终都发射的同一批 bisheng builtin（set_*_nddma + nddma_out_to_ub_b*），
 *     这些 builtin 在 basic_api 上下文可直接调用、无头冲突。
 *
 *   与高层 DataCopyWithNDDMAImpl（basic_api/dav_c310/kernel_operator_data_copy_impl.h）
 *   逐指令对齐：同 (src<<20)|(dst&0xfffff) stride 打包、同 padCount=0、
 *   同 set_pcie_rd_ctrl(0)、同 set_pad_val_nddma(0)、
 *   同 nddma_out_to_ub_b*(..., padding_mode=!isNearestValueMode=true, cacheMode)
 *   → 输出 bitwise 一致。写回 DataCopyPad 与全部 SetFlag/WaitFlag 同步结构保持不变，
 *   使 A/B 精确隔离"NDDMA 读路径 API 层次"这一个变量。
 *
 *   选择方式：launch .asc 编译时定义 LAST_TRANSPOSE_USE_ASC_API 即启用本变体。
 */
#ifndef LAST_TRANSPOSE_NDDMA_BENCH_ASCAPI_ARCH35_H
#define LAST_TRANSPOSE_NDDMA_BENCH_ASCAPI_ARCH35_H

#include "kernel_operator.h"

namespace LastTransposeNddmaBenchAscApiNs {
using namespace AscendC;

// ping-pong 双缓冲开关（编译期）：true=奇偶轮换半区，false=单缓冲同址复用
constexpr bool USE_PINGPONG = true;

template <typename T, uint32_t ubCapElems>
class KernelLastTransposeNDDMA {
public:
    __aicore__ inline KernelLastTransposeNDDMA() {}

    __aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* y,
                                uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3,
                                uint32_t isMte3)
    {
        dim0_ = dim0;
        dim1_ = dim1;
        dim2_ = dim2;
        dim3_ = dim3;
        isMte3_ = isMte3;
        blockElems_ = dim2 * dim3;  // 一个转置块的元素数（转置前后一致）

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    }

    __aicore__ inline void Process()
    {
        // 按 dim0 在核间切分：本核负责 [i0Start, i0Start + i0Count)
        uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
        uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
        uint32_t per = dim0_ / blockNum;
        uint32_t rem = dim0_ % blockNum;
        uint32_t i0Start = blockIdx * per + (blockIdx < rem ? blockIdx : rem);
        uint32_t i0Count = per + (blockIdx < rem ? 1u : 0u);
        if (i0Count == 0) {
            return;
        }

        // DB 双缓冲：一整块 UB 切成两半 ping-pong，i1 奇偶轮换落到不同地址。
        // 约束：blockElems_ <= halfCap（每半区容量）
        static constexpr uint32_t halfCap = ubCapElems / 2;
        LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
        LocalTensor<T> ubBig = ubAllocator.Alloc<T, ubCapElems>();

        // 把 (i0,i1) 两层循环拍平成线性块索引 g：blockOff = g * blockElems_
        uint32_t gStart = i0Start * dim1_;
        uint32_t gCount = i0Count * dim1_;
        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(blockElems_ * sizeof(T)), 0, 0, 0};

        // isMte3_ 只控制"是否发 DataCopyPad 写回"，其余（DCI + 全部同步 + 流水结构）
        // 两种模式完全一致，从而 isMte3_=false 精确隔离出 MTE3 搬运这一个变量。
        if constexpr (USE_PINGPONG) {
            // 双缓冲软件流水：奇偶 buffer + 奇偶 event（0/2/4→ID0，1/3/5→ID1）。
            // prologue：先发第 0 块的读（指令层 NDDMA 转置）
            NddmaReadTranspose(ubBig[0], static_cast<uint64_t>(gStart) * blockElems_);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);

            for (uint32_t k = 0; k < gCount; ++k) {
                uint32_t p = k & 1u;
                // 提前发射下一块的读（另一半区），与本块的写重叠
                if (k + 1 < gCount) {
                    uint32_t q = (k + 1) & 1u;
                    if (k + 1 >= 2) {
                        // 复用 q 半区前，等它上一轮写回完成（WAR）
                        WaitFlag<HardEvent::MTE3_MTE2>(q == 0 ? EVENT_ID0 : EVENT_ID1);
                    }
                    NddmaReadTranspose(ubBig[q * halfCap],
                        static_cast<uint64_t>(gStart + k + 1) * blockElems_);
                    SetFlag<HardEvent::MTE2_MTE3>(q == 0 ? EVENT_ID0 : EVENT_ID1);
                }
                // 等本块读完成 -> 写回（isMte3_=false 时跳过 DataCopyPad，其余同步不变）
                WaitFlag<HardEvent::MTE2_MTE3>(p == 0 ? EVENT_ID0 : EVENT_ID1);
                if (isMte3_) {
                    DataCopyPad(yGm_[static_cast<uint64_t>(gStart + k) * blockElems_],
                        ubBig[p * halfCap], copyParams);
                }
                SetFlag<HardEvent::MTE3_MTE2>(p == 0 ? EVENT_ID0 : EVENT_ID1);
            }
            // epilogue：排空尾部未被等待的写回 flag（两个 event 各剩 1 个）
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            if (gCount >= 2) {
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            }
        } else {
            // 单缓冲串行：read -> 同步 -> (write) -> 同步（无重叠）
            for (uint32_t k = 0; k < gCount; ++k) {
                uint64_t off = static_cast<uint64_t>(gStart + k) * blockElems_;
                NddmaReadTranspose(ubBig[0], off);
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                if (isMte3_) {
                    DataCopyPad(yGm_[off], ubBig[0], copyParams);
                }
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }
    }

private:
    // 指令层等价于原 `NdDmaDci(); DataCopy<T,2,dmaConfig>(ubDst, xGm_[gmOff], params)`。
    // 逐条复刻 DataCopyWithNDDMAImpl<T,2,dmaConfig> 对 loopInfo
    //   {loopSrcStride={1,dim3}, loopDstStride={dim2,1}, loopSize={dim3,dim2}, pad=0}
    // 的下降（basic_api/dav_c310/kernel_operator_data_copy_impl.h:1501-1567）：
    //   loop0(内)：src stride=1、dst stride=dim2、size=dim3
    //   loop1(外)：src stride=dim3、dst stride=1、size=dim2
    //   pad：常数填充、0 pad（默认 NdDmaConfig / constValue=0）
    __aicore__ inline void NddmaReadTranspose(const LocalTensor<T>& ubDst, uint64_t gmOff)
    {
        NdDmaDci();
        uint8_t cacheMode = ExtractCacheMode(xGm_);
        // LoopStride 打包：[59:20] srcStride, [19:0] dstStride
        constexpr uint8_t srcStrideOff = 20;
        uint64_t loop0Stride = (static_cast<uint64_t>(1) << srcStrideOff)
                               | (static_cast<uint64_t>(dim2_) & 0xfffff);
        uint64_t loop1Stride = (static_cast<uint64_t>(dim3_) << srcStrideOff)
                               | (static_cast<uint64_t>(1) & 0xfffff);
        set_pad_cnt_nddma(0);
        set_loop0_stride_nddma(loop0Stride);
        set_loop1_stride_nddma(loop1Stride);
        set_loop2_stride_nddma(0);
        set_loop3_stride_nddma(0);
        set_loop4_stride_nddma(0);
        set_pcie_rd_ctrl(0);

        __ubuf__ T* dst = reinterpret_cast<__ubuf__ T*>(ubDst.GetPhyAddr());
        __gm__ T* src = const_cast<__gm__ T*>(xGm_[gmOff].GetPhyAddr());
        constexpr uint8_t sid = 0;
        // padding_mode = !NdDmaConfig::isNearestValueMode = !false = true（常数 0 填充）
        constexpr bool paddingMode = true;
        if constexpr (sizeof(T) == 1) {
            set_pad_val_nddma(static_cast<uint8_t>(0));
            nddma_out_to_ub_b8(dst, src, sid, dim3_, dim2_, 1u, 1u, 1u, 0, 0, paddingMode, cacheMode);
        } else if constexpr (sizeof(T) == 2) {
            set_pad_val_nddma(static_cast<uint16_t>(0));
            nddma_out_to_ub_b16(dst, src, sid, dim3_, dim2_, 1u, 1u, 1u, 0, 0, paddingMode, cacheMode);
        } else if constexpr (sizeof(T) == 4) {
            set_pad_val_nddma(static_cast<uint32_t>(0));
            nddma_out_to_ub_b32(dst, src, sid, dim3_, dim2_, 1u, 1u, 1u, 0, 0, paddingMode, cacheMode);
        }
    }

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    uint32_t dim0_;
    uint32_t dim1_;
    uint32_t dim2_;
    uint32_t dim3_;
    uint32_t blockElems_;
    uint32_t isMte3_;
};

// UB 容量上界（编译期）：与原头逐字一致，保证 A/B 分区规划相同。
constexpr uint32_t UB_CAP_ELEMS_B16 = 256 * 1024 / sizeof(uint16_t);  // 131072 (2×128KB)
constexpr uint32_t UB_CAP_ELEMS_B32 = 128 * 1024 / sizeof(uint32_t);  // 32768
constexpr uint32_t UB_CAP_ELEMS_B8 = 128 * 1024 / sizeof(uint8_t);    // 131072 (2×64KB)

} // namespace LastTransposeNddmaBenchAscApiNs

#endif // LAST_TRANSPOSE_NDDMA_BENCH_ASCAPI_ARCH35_H
