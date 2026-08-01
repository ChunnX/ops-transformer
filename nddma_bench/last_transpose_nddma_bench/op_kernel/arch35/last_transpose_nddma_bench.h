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
 * \file last_transpose_nddma_bench.h
 * \brief LastTranspose [0,1,3,2] NDDMA benchmark kernel（仅 Ascend 950PR/950DT / arch35）。
 *        计算逻辑仅由一条 NDDMA(DataCopy) 完成 UB 内 [dim2,dim3]->[dim3,dim2] 转置。
 *
 * shape (dim0, dim1, dim2, dim3), permute [0,1,3,2] -> 输出 (dim0, dim1, dim3, dim2)
 *   - dim0        : 分核维，按 dim0 在核间切分
 *   - dim1        : 核内 for 循环维（每次循环一条 NDDMA）
 *   - dim2, dim3  : 一次 UB 内 NDDMA 转置的二维块（需小于 UB 容量）
 *
 * 注：由 Kernel 直调工程 last_transpose_nddma_bench.asc 迁移而来（R-054 only-move-not-rewrite）。
 *     计算逻辑逐字保留，仅将原 attr 结构读取改为 TilingData 字段（在入口 _apt.cpp 中完成）。
 */
#ifndef LAST_TRANSPOSE_NDDMA_BENCH_ARCH35_H
#define LAST_TRANSPOSE_NDDMA_BENCH_ARCH35_H

#include "kernel_operator.h"

namespace LastTransposeNddmaBenchNs {
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

        // DB 双缓冲（临时测试）：一整块 UB 切成两半 ping-pong，
        // i1 奇偶轮换落到不同地址，打破连续 NDDMA 的同址依赖以便重叠。
        // 约束：blockElems_ <= halfCap（每半区容量）
        static constexpr uint32_t halfCap = ubCapElems / 2;
        LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
        LocalTensor<T> ubBig = ubAllocator.Alloc<T, ubCapElems>();

        // NDDMA loopInfo：src[dim2,dim3] -> dst[dim3,dim2]，靠交换 stride 完成转置
        //   inner 维(索引0)：沿 src 列(stride=1)读 dim3 个，写 dst 行(stride=dim2)
        //   outer 维(索引1)：沿 src 行(stride=dim3)读 dim2 个，写 dst 列(stride=1)
        NdDmaLoopInfo<2> loopInfo{{1, dim3_}, {dim2_, 1}, {dim3_, dim2_}, {0, 0}, {0, 0}};
        NdDmaParams<T, 2> params{loopInfo, 0};
        static constexpr NdDmaConfig dmaConfig;  // 默认：常数填充、无 padding

        // 把 (i0,i1) 两层循环拍平成线性块索引 g：blockOff = g * blockElems_
        uint32_t gStart = i0Start * dim1_;
        uint32_t gCount = i0Count * dim1_;
        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(blockElems_ * sizeof(T)), 0, 0, 0};

        // isMte3_ 只控制"是否发 DataCopyPad 写回"，其余（DCI + 全部同步 + 流水结构）
        // 两种模式完全一致，从而 isMte3_=false 精确隔离出 MTE3 搬运这一个变量。
        if constexpr (USE_PINGPONG) {
            // 双缓冲软件流水：奇偶 buffer + 奇偶 event（0/2/4→ID0，1/3/5→ID1）。
            // read 提前一拍发射，落到另一半区，与本块的 write 在不同 pipe 上重叠。
            // prologue：先发第 0 块的读
            NdDmaDci();
            DataCopy<T, 2, dmaConfig>(
                ubBig[0], xGm_[static_cast<uint64_t>(gStart) * blockElems_], params);
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
                    NdDmaDci();
                    DataCopy<T, 2, dmaConfig>(ubBig[q * halfCap],
                        xGm_[static_cast<uint64_t>(gStart + k + 1) * blockElems_], params);
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
                NdDmaDci();
                DataCopy<T, 2, dmaConfig>(ubBig[0], xGm_[off], params);
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
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    uint32_t dim0_;
    uint32_t dim1_;
    uint32_t dim2_;
    uint32_t dim3_;
    uint32_t blockElems_;
    uint32_t isMte3_;
};

// UB 容量上界（编译期）：静态 Tensor 分配需常量。
// 满 UB 自适应：以架构全 UB 字节数（编译期常量）派生每 dtype 的元素容量，
//   三 dtype 统一占满全 UB，不再写死 256KB/128KB 定值。
//   FULL_UB_BYTES = AscendC::GetUBSizeInBytes()：constexpr，arch35(__NPU_ARCH__==3510)
//   返回芯片全 UB 字节数（Ascend 950PR/DT = 248KB），可用于静态 Alloc 的模板实参。
//   DB 双缓冲：cap 容纳 2 个半区，halfCap = ubCapElems/2 = 全 UB/2（随芯片自适应）。
constexpr uint32_t FULL_UB_BYTES = AscendC::GetUBSizeInBytes();
constexpr uint32_t UB_CAP_ELEMS_B16 = FULL_UB_BYTES / sizeof(uint16_t);  // 全 UB/2B, halfCap=全UB/4B
constexpr uint32_t UB_CAP_ELEMS_B32 = FULL_UB_BYTES / sizeof(uint32_t);  // 全 UB/4B, halfCap=全UB/8B
constexpr uint32_t UB_CAP_ELEMS_B8 = FULL_UB_BYTES / sizeof(uint8_t);    // 全 UB/1B, halfCap=全UB/2B

} // namespace LastTransposeNddmaBenchNs

#endif // LAST_TRANSPOSE_NDDMA_BENCH_ARCH35_H
