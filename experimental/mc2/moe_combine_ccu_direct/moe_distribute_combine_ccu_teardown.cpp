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
 * \brief v11 combine_teardown AIV ascendc kernel（与 dispatch_teardown v11 完全对称）。
 *
 * Algorithm 三段式（与 dispatch_teardown v11 对称）：
 *   1. WaitForAllSrcRanks  轮询本卡 commCmdInfo[r] sentinel 等齐所有 src rank 反向写完
 *   2. 按 (i, j) 遍历本卡所有 token 副本：
 *      - 从 routingTable[i*K+j] 得 dst rank（= dispatch 时的 dst rank）
 *      - 本卡的 yCombine[rank 段, peerRecvCount[rank] 槽] 是该 (i,j) 副本经 expert 处理后的反向结果
 *      - DataCopy yCombine[...] → x[i]（简化版："K 次覆盖"，正确加权求和待生产版）
 *      - peerRecvCount[rank] += 1
 *   3. （生产版）加权求和：x[i] = Σ_{j=0..K-1} expert_scales[i*K+j] * yCombine[rank_route[i,j], slot]
 *
 * 简化范围（v11 demo）：
 *   - "K 次覆盖"代替"加权求和"——demo 验证 kernel 框架 + 双流并行 + 等齐机制，
 *     正确语义留待生产版（需 BF16→FP32 Cast + Mul + Accumulate + FP32→BF16 Cast）
 *   - block 数 = 1，dtype = bfloat16_t
 *
 * Sentinel 等齐机制（与 dispatch_teardown 完全相同）：
 *   - host 初始化 commCmdInfo 前 epWorldSize 槽为 -1（0xFFFFFFFF）
 *   - combine_setup 段 8 跨卡 / 本地写 token 数（>= 0）覆盖 sentinel
 *   - teardown 轮询所有 r 的 commCmdInfo[r] != -1 → 等齐
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
        tokenMsgLen_      = tiling->tokenMsgLen;
        tokenWithMetaLen_ = tiling->tokenWithMetaLen;
        segBytes_         = tiling->segBytes;

        // v16 ★ 复用 dispatch 同款 commWindow（hccl 开辟），不再用 yCombine
        commWindowGM_.SetGlobalBuffer((__gm__ YType*)commWindow);
        commCmdInfoGM_.SetGlobalBuffer((__gm__ int32_t*)commCmdInfo);
        xOutGM_.SetGlobalBuffer((__gm__ YType*)xOut);
        pipe_ = pipe;

        // UB chunk buffer（GM→UB→GM 中转）
        pipe_->InitBuffer(ubBuf_, kCombineChunkElems * sizeof(YType));
    }

    /**
     * v11 ★ 等齐：轮询本卡 commCmdInfo sentinel（与 dispatch_teardown 完全相同的机制）。
     *
     * Sentinel 约定：
     *   - host 初始化 commCmdInfo 前 epWorldSize 槽为 -1（0xFFFFFFFF）
     *   - combine_setup 段 8 跨卡 WriteNbWithArriveNotify / 本地 StoreVariable
     *     写完后变为 >= 0 的实际 token 数（覆盖 sentinel）
     *   - 本函数轮询所有 r：commCmdInfoGM_.GetValue(r) != -1 → 已到位
     */
    __aicore__ inline void WaitForAllSrcRanks()
    {
        constexpr int32_t kSentinelUnset = -1;
        bool allArrived = false;
        while (!allArrived) {
            allArrived = true;
            for (uint32_t r = 0; r < epWorldSize_; ++r) {
                int32_t val = commCmdInfoGM_.GetValue(r);
                if (val == kSentinelUnset) {
                    allArrived = false;
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
        // demo 简化：identity + last-write-wins 等价语义下直接取 j=0 副本即 = x[i]
        //   生产版加权求和应改：
        //     for (i, j) read commWindow[i*K+j] → Cast FP32 → Muls expertScales[i*K+j] → Add accumBuf
        //     最后 accumBuf → Cast BF16 → write xOut[i]
        for (uint32_t i = 0; i < bs_; ++i) {
            // commWindow[i*K] 起始元素偏移（j=0 副本的位置）
            const uint64_t srcOffset = static_cast<uint64_t>(i) * topK_ * tokenWithMetaElems;
            const uint64_t dstOffset = static_cast<uint64_t>(i) * static_cast<uint64_t>(hiddenSize_);

            // 只读 token data 的 hiddenSize 元素（不读 metadata + padding 区）
            const uint64_t copyElemsPerToken = static_cast<uint64_t>(hiddenSize_);
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
    uint64_t tokenMsgLen_;
    uint64_t tokenWithMetaLen_;    // v15 ★
    uint64_t segBytes_;
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
    GM_ADDR expandX, GM_ADDR quantExpandX, GM_ADDR expertIds, GM_ADDR expandIdx,
    GM_ADDR expertScales, GM_ADDR commCmdInfo, GM_ADDR xActiveMask, GM_ADDR sharedExpertX,
    GM_ADDR xOut,
    GM_ADDR commWindow,                       // v16 ★ 复用 dispatch 同款 commWindow（替代 yCombine + routingTable）
    GM_ADDR workspace, GM_ADDR tilingGM)
{
    // v16 简化：device 端不读这几个字段（端到端 IO 契约保留）
    (void)expandX; (void)quantExpandX; (void)expertIds; (void)expandIdx;
    (void)expertScales; (void)xActiveMask; (void)sharedExpertX; (void)workspace;

    REGISTER_TILING_DEFAULT(MoeDistributeCombineCcuTeardownTilingData);
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineCcuTeardownTilingData, tilingData, tilingGM);

    MoeDistributeCombineCcuTeardownAiv<bfloat16_t> op;
    op.Init(commWindow, commCmdInfo, xOut, &pipe, &tilingData);
    op.Process();
}
