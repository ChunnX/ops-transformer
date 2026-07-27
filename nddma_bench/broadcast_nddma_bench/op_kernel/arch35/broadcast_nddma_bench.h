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
 * \file broadcast_nddma_bench.h
 * \brief Broadcast NDDMA benchmark kernel（仅 Ascend 950PR/950DT / arch35）——3D NDDMA 方案（DESIGN §9，权威）。
 *        计算逻辑由**一条 3D NdDmaLoopInfo<3> DataCopy** 完成沿广播轴复制搬运（含双/中轴 srcStride=0）。
 *        dim1 是唯一核内 for 循环维；dim2/dim3/dim4 全部是同一条 NDDMA 的 loopInfo 轴 →
 *        单条 tile=dim2*dim3*dim4。§8 方案 A（dim2 核内 for）已作废。
 *
 * shape 输入按 scene 的广播轴长度=1 的 5D 形态 -> 输出恒 (dim0, dim1, dim2, dim3, dim4)
 *   - dim0            : 分核维，按 dim0 在核间切分（dim0=物理 vector core 数 56，单波 per=1/rem=0）
 *   - dim1            : 唯一核内 for 循环维（与 dim0 拍平成线性块索引 g；每拍发一条 3D NDDMA）
 *   - dim2            : NDDMA 最外轴 loopInfo[2]（单条 NDDMA 内遍历，不是 for）
 *   - dim3            : NDDMA 中轴   loopInfo[1]（单条 NDDMA 内遍历）
 *   - dim4            : NDDMA 最内连续轴 loopInfo[0]（单条 NDDMA 内遍历；inner 连续块须 8B 对齐）
 *
 * 四场景（DESIGN §9.0/§9.2；NdDmaLoopInfo<3> 索引 0=最内 dim4，1=中 dim3，2=最外 dim2；单位=元素）：
 *   AB  src(d2,d3,1)  dim4 广播        srcStride[0,1,d3]   置零 1 个（inner）      SrcOff=g*(d2*d3)
 *   BA  src(d2,1,d4)  dim3 广播        srcStride[1,0,d4]   置零 1 个（中轴 dim3）  SrcOff=g*(d2*d4)
 *   BAB src(1,d3,1)   dim2+dim4 广播   srcStride[0,1,0]    置零 2 个（inner+outer）SrcOff=g*d3
 *   ABA src(d2,1,d4)  dim3 广播        srcStride[1,0,d4]   置零 1 个（中轴 dim3）  SrcOff=g*(d2*d4)
 *   dstStride 恒连续 [1, dim4, dim3*dim4]；置零轴由 srcStride=0 实现广播复制读。
 *
 * ★ BA ≡ ABA 同构（DESIGN §9.1，设计而非 bug）：3D 单条方案下 BA 与 ABA 的 loopInfo 与 SrcOff 完全相同。
 *   switch 中 SCENE_BA 单列、ABA 落 default 且分支体逐字节相同是**有意为之**，code review 勿判 bug。
 *
 * 注：由 Kernel 直调工程 broadcast_nddma_bench.asc 迁移而来（R-054 only-move-not-rewrite）。
 *     计算逻辑逐字保留，仅将原 attr 结构读取改为 TilingData 字段（在入口 _apt.cpp 中完成）。
 */
#ifndef BROADCAST_NDDMA_BENCH_ARCH35_H
#define BROADCAST_NDDMA_BENCH_ARCH35_H

#include "kernel_operator.h"

namespace BroadcastNddmaBenchNs {
using namespace AscendC;

// 5D 场景常量（DESIGN §9.0）
static constexpr uint32_t SCENE_AB = 0;   // dim4 广播（inner 置零）；srcStride[0,1,d3]
static constexpr uint32_t SCENE_BA = 1;   // dim3 广播（中轴置零）；srcStride[1,0,d4]
static constexpr uint32_t SCENE_BAB = 2;  // dim2+dim4 广播（inner+outer 双置零）；srcStride[0,1,0]
static constexpr uint32_t SCENE_ABA = 3;  // dim3 广播（中轴置零）；srcStride[1,0,d4]（≡ BA，§9.1）

// ping-pong 双缓冲开关（编译期）：true=奇偶轮换半区，false=单缓冲同址复用
constexpr bool USE_PINGPONG = true;

template <typename T, uint32_t ubCapElems>
class KernelBroadcastNDDMA {
public:
    __aicore__ inline KernelBroadcastNDDMA() {}

    __aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* y,
                                uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3,
                                uint32_t dim4, uint32_t isMte3, uint32_t scene)
    {
        dim0_ = dim0;
        dim1_ = dim1;
        dim2_ = dim2;
        dim3_ = dim3;
        dim4_ = dim4;
        isMte3_ = isMte3;
        scene_ = scene;
        // 3D 方案（DESIGN §9.1~§9.4）：单条 3D NDDMA 目的块 = dim2*dim3*dim4（dim2 回归 NDDMA 最外轴）。
        // Init 三断言（DESIGN §9.4）：正常路径永不触发，纯回归网只拦 host 配错。
        // (1) loopSize>1 红线：dim2/dim3/dim4 均 >=2。
        ascendc_assert(dim2 >= 2 && dim3 >= 2 && dim4 >= 2,
            "3D invariant: need all of dim2/dim3/dim4 >= 2 (no degenerate loopSize) "
            "(scene=%u d2=%u d3=%u d4=%u).\n", scene, dim2, dim3, dim4);
        // (2) inner 连续块 8B 对齐。
        ascendc_assert((dim4 * sizeof(T)) % 8 == 0,
            "3D inner-run must be 8B-aligned: dim4=%u * elem=%u not %%8==0.\n",
            dim4, (uint32_t)sizeof(T));
        // (3) UB tile 容量：单条 3D 目的块 dim2*dim3*dim4 须 ≤ DB halfCap（最严）。
        static constexpr uint32_t halfCap = ubCapElems / 2;
        ascendc_assert((uint64_t)dim2 * dim3 * dim4 <= halfCap,
            "3D tile d2*d3*d4=%llu > halfCap %u (scene=%u d2=%u d3=%u d4=%u).\n",
            (uint64_t)dim2 * dim3 * dim4, halfCap, scene, dim2, dim3, dim4);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    }

    // flat 线性序号 j ∈ [0, gCount) → 全局块号 g（DESIGN §9.2/§9.3）。
    __aicore__ inline uint32_t Gid(uint32_t j) const
    {
        return (j / unitCount_) * numBlocks_ + (unitStart_ + j % unitCount_);
    }

    // 读偏移（元素）：按 scene 直算 g 对应的源块基址（DESIGN §9.2）。
    __aicore__ inline uint64_t SrcOff(uint32_t j) const
    {
        uint32_t g = Gid(j);
        if (scene_ == SCENE_AB)  return (uint64_t)g * (dim2_ * dim3_);   // src(d2,d3,1) 整块 d2*d3
        if (scene_ == SCENE_BA)  return (uint64_t)g * (dim2_ * dim4_);   // src(d2,1,d4) 整块 d2*d4
        if (scene_ == SCENE_BAB) return (uint64_t)g * dim3_;             // src(1,d3,1)  整块 d3
        return (uint64_t)g * (dim2_ * dim4_);                            // ABA src(d2,1,d4) 整块 d2*d4（≡BA，§9.1）
    }

    // 写偏移（元素）：输出连续，块基址 = g * dim2*dim3*dim4（DESIGN §9.2）。
    __aicore__ inline uint64_t DstOff(uint32_t j) const
    {
        return (uint64_t)Gid(j) * (dim2_ * dim3_ * dim4_);
    }

    // 构造单条 3D loopInfo（DESIGN §9.2），并回填目的块元素数 = dim2*dim3*dim4。
    //   索引 0=最内 dim4，1=中 dim3，2=最外 dim2；dstStride 恒连续 [1, dim4, dim3*dim4]。
    //   ★ SCENE_BA 与 default(ABA) 分支体逐字节相同=BA≡ABA（§9.1，有意为之）。
    __aicore__ inline NdDmaLoopInfo<3> MakeLoopInfo(uint32_t /*j*/, uint32_t& curDstElems) const
    {
        curDstElems = dim2_ * dim3_ * dim4_;                            // 单条 3D 目的块
        const uint32_t dsIn = 1, dsMid = dim4_, dsOut = dim3_ * dim4_;  // dst 连续步长
        uint64_t ssIn, ssMid, ssOut;
        switch (scene_) {
            case SCENE_AB:  ssIn = 0; ssMid = 1; ssOut = dim3_; break;  // dim4 广播（inner 置零）
            case SCENE_BA:  ssIn = 1; ssMid = 0; ssOut = dim4_; break;  // dim3 广播（中轴置零）
            case SCENE_BAB: ssIn = 0; ssMid = 1; ssOut = 0;     break;  // dim2+dim4 广播（inner+outer 双置零）
            default:        ssIn = 1; ssMid = 0; ssOut = dim4_; break;  // ABA：dim3 广播（中轴置零，≡BA §9.1）
        }
        return NdDmaLoopInfo<3>{
            {ssIn, ssMid, ssOut}, {dsIn, dsMid, dsOut},
            {dim4_, dim3_, dim2_}, {0, 0, 0}, {0, 0, 0}};
    }

    __aicore__ inline void Process()
    {
        uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
        uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());

        // 分核单元：沿 dim0 切分（numBlocks_=dim0，每单元 dim1 条 3D NDDMA），DESIGN §9.3。
        numBlocks_ = dim0_;

        uint32_t per = numBlocks_ / blockNum;
        uint32_t rem = numBlocks_ % blockNum;
        uint32_t unitStart = blockIdx * per + (blockIdx < rem ? blockIdx : rem);
        uint32_t unitCount = per + (blockIdx < rem ? 1u : 0u);
        if (unitCount == 0) {
            return;
        }
        // 存本核分核参数供 Gid（const SrcOff/DstOff）解 g。
        unitStart_ = unitStart;
        unitCount_ = unitCount;

        static constexpr uint32_t halfCap = ubCapElems / 2;
        LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
        LocalTensor<T> ubBig = ubAllocator.Alloc<T, ubCapElems>();
        static constexpr NdDmaConfig dmaConfig;  // 默认：常数填充、无 padding

        // 3D 方案（DESIGN §9.3）：dim1 是唯一核内 for → gCount = unitCount*dim1。
        uint32_t gCount = unitCount * dim1_;

        // isMte3_ 只控制"是否发 DataCopyPad 写回"，其余（DCI + 全部同步 + 流水结构）两模式完全一致。
        if constexpr (USE_PINGPONG) {
            uint32_t dstElems0;
            NdDmaLoopInfo<3> li0 = MakeLoopInfo(0, dstElems0);
            NdDmaParams<T, 3> params0{li0, 0};
            NdDmaDci();
            DataCopy<T, 3, dmaConfig>(ubBig[0], xGm_[SrcOff(0)], params0);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);

            for (uint32_t k = 0; k < gCount; ++k) {
                uint32_t p = k & 1u;
                if (k + 1 < gCount) {
                    uint32_t q = (k + 1) & 1u;
                    if (k + 1 >= 2) {
                        WaitFlag<HardEvent::MTE3_MTE2>(q == 0 ? EVENT_ID0 : EVENT_ID1);
                    }
                    uint32_t dstElemsN;
                    NdDmaLoopInfo<3> liN = MakeLoopInfo(k + 1, dstElemsN);
                    NdDmaParams<T, 3> paramsN{liN, 0};
                    NdDmaDci();
                    DataCopy<T, 3, dmaConfig>(ubBig[q * halfCap], xGm_[SrcOff(k + 1)], paramsN);
                    SetFlag<HardEvent::MTE2_MTE3>(q == 0 ? EVENT_ID0 : EVENT_ID1);
                }
                WaitFlag<HardEvent::MTE2_MTE3>(p == 0 ? EVENT_ID0 : EVENT_ID1);
                if (isMte3_) {
                    uint32_t dstElems;
                    (void)MakeLoopInfo(k, dstElems);  // 取本块目的元素数（=d2*d3*d4）
                    DataCopyExtParams copyParams{
                        1, static_cast<uint32_t>(dstElems * sizeof(T)), 0, 0, 0};
                    DataCopyPad(yGm_[DstOff(k)], ubBig[p * halfCap], copyParams);
                }
                SetFlag<HardEvent::MTE3_MTE2>(p == 0 ? EVENT_ID0 : EVENT_ID1);
            }
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            if (gCount >= 2) {
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            }
        } else {
            for (uint32_t k = 0; k < gCount; ++k) {
                uint32_t dstElems;
                NdDmaLoopInfo<3> li = MakeLoopInfo(k, dstElems);
                NdDmaParams<T, 3> params{li, 0};
                NdDmaDci();
                DataCopy<T, 3, dmaConfig>(ubBig[0], xGm_[SrcOff(k)], params);
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                if (isMte3_) {
                    DataCopyExtParams copyParams{
                        1, static_cast<uint32_t>(dstElems * sizeof(T)), 0, 0, 0};
                    DataCopyPad(yGm_[DstOff(k)], ubBig[0], copyParams);
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
    uint32_t dim2_;      // NDDMA 最外轴 loopInfo[2]
    uint32_t dim3_;      // NDDMA 中轴   loopInfo[1]
    uint32_t dim4_;      // NDDMA 最内连续轴 loopInfo[0]
    uint32_t scene_;     // 5D 场景枚举
    uint32_t isMte3_;
    uint32_t numBlocks_; // 恒 = dim0_
    uint32_t unitStart_; // 本核分核起始单元（供 Gid 解 g）
    uint32_t unitCount_; // 本核分核单元数
};

// UB 容量上界（编译期）：静态 Tensor 分配需常量。逐 dtype 沿用参考算子已上板验证的定值。
//   DB 双缓冲：cap 容纳 2 个半区，halfCap = ubCapElems/2。
constexpr uint32_t UB_CAP_ELEMS_B16 = 256 * 1024 / sizeof(uint16_t);  // 131072 (2×128KB), halfCap=65536
constexpr uint32_t UB_CAP_ELEMS_B32 = 128 * 1024 / sizeof(uint32_t);  // 32768  (2×64KB),  halfCap=16384
// B8：UB 128KB，每元素 1B，elem 容量 131072（与 b16 数值相同），halfCap=65536。
constexpr uint32_t UB_CAP_ELEMS_B8 = 128 * 1024 / sizeof(uint8_t);    // 131072 (2×64KB),  halfCap=65536

} // namespace BroadcastNddmaBenchNs

#endif // BROADCAST_NDDMA_BENCH_ARCH35_H
