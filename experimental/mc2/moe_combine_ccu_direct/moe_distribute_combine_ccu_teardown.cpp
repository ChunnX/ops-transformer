/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_ccu_teardown.cpp
 * \brief combine_teardown AIV ascendc kernel（与 dispatch_teardown 对称，v20）。
 *
 * Algorithm 两段（纯 vector，无通信）：
 *   1. WaitForAllSrcRanks：轮询本卡 commCmdInfo 的 ready 区 [0,R) 等全为 1
 *      （combine_setup CCU 段 9 WaitAllPeersArriveNotify 收齐反向数据后，段 10 LocalCopyNb 置 ready=1）。
 *   2. K-stride 取 commWindow[i*K]（token i 的 j=0 副本）→ xOut[i]。
 *      identity 模式下 commWindow[i*K] 即 x[i]，故 xOut==x 闭环成立。
 *      生产版需对每 i 的 K 个副本加权求和（Cast + Muls expertScales + Add）。
 *
 * 简化范围（demo）：block 数 = 1，dtype = bfloat16_t；K-stride 取 j=0 副本（非加权求和）。
 *
 * 等齐机制（v20 收口，与 dispatch_teardown 完全相同）：
 *   - commCmdInfo 简化为只剩 ready 区 [0,R)（删 tokenCount/scratch），driver 初始化为 0。
 *   - combine_setup 段 10 用 1 次 LocalCopyNb 把 ready[0..R) 置 1（收齐后）。
 *   - teardown 轮询 ready[r] 全 == 1 → 数据齐。K-stride 算法不读 tokenCount。
 */

#include "kernel_operator.h"
#include "moe_distribute_combine_ccu_teardown.h"

using namespace AscendC;

// UB chunk 大小（与 dispatch_teardown 对齐）
constexpr uint32_t kCombineChunkElems = 256;

/**
 * v11 combine teardown AIV 算子核心实现。
 * 模板化 YType 便于未来支持 INT8/FP8 量化场景。
 */
template<typename YType>
class MoeDistributeCombineCcuTeardownAiv {
public:
    __aicore__ inline void Init(
        GM_ADDR commWindow, GM_ADDR commCmdInfo,
        GM_ADDR xOut,
        TPipe* pipe,
        const MoeDistributeCombineCcuTeardownTilingData* tiling)
    {
        epWorldSize_      = tiling->epWorldSize;
        bs_               = tiling->bs;
        topK_             = tiling->topK;
        hiddenSize_       = tiling->hiddenSize;
        commQuantMode_    = tiling->commQuantMode;     // v17 Phase 2 ★
        tokenWithMetaLen_ = tiling->tokenWithMetaLen;

        // v16 ★ 复用 dispatch 同款 commWindow（hccl 开辟），不再用 yCombine
        commWindowGM_.SetGlobalBuffer((__gm__ YType*)commWindow);
        commCmdInfoGM_.SetGlobalBuffer((__gm__ int32_t*)commCmdInfo);
        xOutGM_.SetGlobalBuffer((__gm__ YType*)xOut);
        pipe_ = pipe;

        // UB chunk buffer（GM→UB→GM 中转）
        pipe_->InitBuffer(ubBuf_, kCombineChunkElems * sizeof(YType));
    }

    /**
     * 收口 ★ 等齐：轮询本卡 commCmdInfo 的 ready 区 [0, R) 等全为 1（与 dispatch_teardown 完全相同）。
     *
     * 等齐信号 = "专用 ready 区全 1"。combine_setup CCU 段 9 WaitAllPeersArriveNotify(R-1) 确认本卡
     *   收齐反向数据后，段 10 用 1 次 LocalCopyNb 把 oneBuf 整块写到 ready 区置 ready[0..R)=1。
     *
     * 布局收口：commCmdInfo 简化为只剩 ready 区，从偏移 0 起（删了 tokenCount/scratch）。
     *   ready 区 int32 下标 = r（readyBase = 0）。combine_teardown 用 K-stride 算法（从 commWindow[i*K]
     *   取副本），本就不读 tokenCount，故此简化对 combine teardown 无其它影响。
     *   ★ setup 段 10 写的 readyAddr 偏移（0）== 此处 readyBase（0），务必一致。
     */
    __aicore__ inline void WaitForAllSrcRanks()
    {
        const uint32_t readyBase = 0U;          // 收口 ★ ready 区从偏移 0 起（与 setup 段 10 一致）
        bool allReady = false;
        while (!allReady) {
            allReady = true;
            for (uint32_t r = 0; r < epWorldSize_; ++r) {
                int32_t val = commCmdInfoGM_.GetValue(readyBase + r);
                if (val != 1) {
                    allReady = false;
                    break;
                }
            }
        }
    }

    /**
     * Process: 等齐 + K-stride 取 commWindow → xOut（v16 ★ 极简化）。
     *
     * v16 算法（用户描述）：
     *   combine_setup 已经把所有 token 副本按 slot = srcSlot*K + dstK 写到本卡 commWindow
     *   layout：commWindow[i*K + 0..K-1] 都是本卡 token i 的 K 个副本
     *
     *   构造 xOut：以 K 为 stride 取 commWindow[i*K] (j=0 副本)，连续放到 xOut[i]
     *   - identity 模式下 commWindow[i*K] 就是 x[i] → xOut[i] = x[i] ✓
     *   - 生产版需对每 i 做 K 个副本的加权求和（Cast + Muls + Add）
     *
     * 不再需要：routingTable / expertIds / peerRecvCount（v16 commWindow layout 自身已隐含定位信息）
     */
    __aicore__ inline void Process()
    {
        if (GetBlockIdx() != 0) {
            return;
        }

        // ====== 阶段 1：等齐所有对端反向写完 ======
        WaitForAllSrcRanks();

        LocalTensor<YType> ubLocal = ubBuf_.Get<YType>();
        // v16 ★ commWindow 中每槽 tokenWithMetaLen 字节（含 metadata），按 BF16 元素数算 stride
        const uint64_t tokenWithMetaElems = tokenWithMetaLen_ / sizeof(YType);

        // ====== 阶段 2：以 K 为 stride 取 commWindow[i*K] → xOut[i] ======
        //
        // v17 Phase 2 ★ 按 commQuantMode 切 copyLen + dstStride：
        //   commQuantMode=0 (BF16):
        //     - commWindow[i*K] data 段 = hiddenSize 个 BF16 元素 = 2*H 字节
        //     - xOut[i] = hiddenSize 个 BF16 元素 = 2*H 字节
        //     - copyElemsPerToken = hiddenSize 元素
        //   commQuantMode=1 (MXFP8 字节透传):
        //     - commWindow[i*K] data 段 = (H + H/32) 字节 (FP8 data + E8M0 scale)
        //     - xOut[i] = (H + H/32) 字节字节流（保留 MXFP8 格式，不反量化）
        //     - copyElemsPerToken = (H + H/32) / sizeof(BF16) 个 BF16 元素
        //
        //   ★★★ MXFP8 H 约束（host driver runtime check）：H 必须是 64 的倍数 ★★★
        //     原因：copyElemsPerToken = (H + H/32) / 2 要求 (H + H/32) 是 2 的倍数
        //          = H * 33/32 是 2 的倍数 ↔ H/32 是 2 的倍数 ↔ H 是 64 的倍数
        //     生产 H ∈ {256, 4096, 7168, 8192} 都满足 ✓
        //     违反约束的边缘 case (H=32/96/160 等) host driver 已 abort，device 端不必再 check
        //
        //   两路径都不引入类型转换——纯字节透传，xOut layout 随 quantMode 切
        //   生产版加权求和（Phase 3+）：BF16 模式可加 Muls expertScales；MXFP8 模式需先 FP8→FP32 cast
        const uint64_t dataBytesPerToken = (commQuantMode_ == 0)
            ? static_cast<uint64_t>(hiddenSize_) * sizeof(YType)              // BF16: 2*H 字节
            : static_cast<uint64_t>(hiddenSize_) + hiddenSize_ / 32U;          // MXFP8: H + H/32 字节
        const uint64_t copyElemsPerToken = dataBytesPerToken / sizeof(YType); // 按 BF16 视图换元素数

        for (uint32_t i = 0; i < bs_; ++i) {
            // commWindow[i*K] 起始元素偏移（j=0 副本的位置，按 BF16 视图）
            const uint64_t srcOffset = static_cast<uint64_t>(i) * topK_ * tokenWithMetaElems;
            // xOut[i] 起始元素偏移（按 BF16 视图）
            const uint64_t dstOffset = static_cast<uint64_t>(i) * copyElemsPerToken;

            for (uint64_t off = 0; off < copyElemsPerToken; off += kCombineChunkElems) {
                uint32_t copyLen = static_cast<uint32_t>(
                    (copyElemsPerToken - off >= kCombineChunkElems) ? kCombineChunkElems : (copyElemsPerToken - off));
                DataCopy(ubLocal, commWindowGM_[srcOffset + off], copyLen);
                PipeBarrier<PIPE_ALL>();
                DataCopy(xOutGM_[dstOffset + off], ubLocal, copyLen);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

private:
    GlobalTensor<YType> commWindowGM_;       // v16 ★ 复用 dispatch 同款 commWindow
    GlobalTensor<int32_t> commCmdInfoGM_;
    GlobalTensor<YType> xOutGM_;
    TBuf<TPosition::VECCALC> ubBuf_;
    TPipe* pipe_;

    uint32_t epWorldSize_;
    uint32_t bs_;
    uint32_t topK_;
    uint32_t hiddenSize_;
    uint32_t commQuantMode_;       // v17 Phase 2 ★ 0=BF16, 1=MXFP8（影响 copyLen）
    uint64_t tokenWithMetaLen_;    // v15 ★
};

/**
 * v11 combine teardown kernel 入口。
 *   入参顺序对齐库上 op_def（Inputs: expandX, quantExpandX, expertIds, expandIdx,
 *   expertScales, commCmdInfo, xActiveMask, sharedExpertX; Output: xOut）。
 *   + v11 工程内部参数：yCombineAddr（接收 buffer）/ routingTableAddr（路由表）。
 *
 * host driver 通过 `<<<blockDim, l2_ctrl, stream>>>(...)` 直接 launch。
 */
extern "C" __global__ __aicore__ void moe_distribute_combine_ccu_teardown(
    GM_ADDR commCmdInfo, GM_ADDR commWindow, GM_ADDR xOut, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeCombineCcuTeardownTilingData);
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineCcuTeardownTilingData, tilingData, tilingGM);

    MoeDistributeCombineCcuTeardownAiv<bfloat16_t> op;
    op.Init(commWindow, commCmdInfo, xOut, &pipe, &tilingData);
    op.Process();
}
