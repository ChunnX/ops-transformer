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
 * \file moe_distribute_dispatch_ccu_teardown.cpp
 * \brief v10 ★ 架构纠正：teardown 从 hcomm CCU 路径改为 AIV ascendc kernel（"AIV 直驱"）。
 *
 * v9→v10 关键差别：
 *   v9：teardown 继承 hcomm::CcuKernel，用 CCU DSL（CcuRep::Variable / LocalCopyNb /
 *       WaitAllPeersArriveNotify 等），跑在 CCU 引擎；通过 HcclCcuKernelRegister/Launch。
 *   v10：teardown 是标准 ascendc kernel（`__global__ __aicore__` 函数），跑在 AIV；
 *        用 GlobalTensor / LocalTensor / DataCopy 等标准 ascendc API；通过 `<<<>>>` 直调。
 *
 * 等齐机制：
 *   v9 teardown 内部用 WaitAllPeersArriveNotify 等 0.5RTT 通信完成。
 *   v10 等齐工作上移到 setup CCU 末尾（segment 9），setup CCU return 时所有对端数据
 *   已就位；stream 顺序保证 v10 teardown launch 时 y / commCmdInfo 内容稳定。
 *
 * Algorithm 三段式（纯 vector 计算，无任何通信）：
 *   1. 读 commCmdInfo[r] (r in [0, epWorldSize)) 得每个 src rank 发本卡的 token 数
 *   2. 按 src rank 整段 DataCopy y[r 段] → expand_x[r 段]（每段 segBytes 字节）
 *   3. 累计 token 数，写 assist_info[r] 前缀和
 *
 * 简化范围：
 *   - dtype 写死 bfloat16_t（生产可按 tilingData 字段切模板）
 *   - block 数 = 1（简化；后续可按 epWorldSize 拆 block 加速）
 *   - dynamicScales / expertTokenNums 不写（host driver sync 后归零）
 */

#include "kernel_operator.h"
#include "moe_distribute_dispatch_ccu_teardown.h"

using namespace AscendC;

// UB chunk 大小：单次 GM→UB / UB→GM 搬运的元素数（BF16/FP16），平衡 UB 占用与流水
constexpr uint32_t kChunkElems = 256;

/**
 * v10 teardown AIV 算子核心实现。
 * 模板化 YType 以便未来支持非 BF16 dtype（INT8/FP8 量化场景）。
 */
template<typename YType>
class MoeDistributeDispatchCcuTeardownAiv {
public:
    /**
     * Init: 解析 tiling、绑定 GM buffer、分配 UB chunk。
     *   只用到 y / commCmdInfo / expandX / assistInfo 四个 GM；x / expertIds /
     *   dynamicScales / expertTokenNums 端到端 IO 契约保留但 device 端 v10 不读写。
     */
    __aicore__ inline void Init(
        GM_ADDR commWindow, GM_ADDR commCmdInfo,
        GM_ADDR expandX, GM_ADDR assistInfoForCombine,
        TPipe* pipe,
        const MoeDistributeDispatchCcuTeardownTilingData* tiling)
    {
        epWorldSize_      = tiling->epWorldSize;
        bs_               = tiling->bs;
        topK_             = tiling->topK;
        tokenMsgLen_      = tiling->tokenMsgLen;
        tokenWithMetaLen_ = tiling->tokenWithMetaLen;   // = tokenMsgLen + 32

        // v13 ★ commWindowGM_ 绑定到对端接收 window（v13 工程内部 scratch，库上 op_def 无此字段）
        commWindowGM_.SetGlobalBuffer((__gm__ YType*)commWindow);
        // v16+ ★ commWindow 的 INT32 视图：用于 Process 读 token 末尾 metadata 三元组
        //   规避用 commWindowGM_.GetPhyAddr() （非 ascendc 标准 API），改在 Init 阶段
        //   用同一 GM_ADDR 创建独立 int32 GlobalTensor，Process 直接 GetValue 即可
        commWindowMetaGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(commWindow));
        expandXGM_.SetGlobalBuffer((__gm__ YType*)expandX);
        commCmdInfoGM_.SetGlobalBuffer((__gm__ int32_t*)commCmdInfo);
        assistInfoGM_.SetGlobalBuffer((__gm__ int32_t*)assistInfoForCombine);
        pipe_ = pipe;

        // UB chunk buffer：用作 GM→UB→GM 搬运的中转
        // ascendc 不直接支持 GM→GM DataCopy，必须经 UB 中转
        pipe_->InitBuffer(ubBuf_, kChunkElems * sizeof(YType));
    }

    /**
     * v11 ★ 等齐阶段：轮询本卡 commCmdInfo[r] 等 sentinel 被覆盖。
     *
     * 双流并行架构：setup CCU 立即返回（异步），teardown AIV 内部等齐。
     *
     * Sentinel 机制：
     *   - host 初始化 commCmdInfo 前 epWorldSize 槽为 0xFFFFFFFF (= -1 as int32)
     *   - setup 段 8 跨卡 WriteNbWithArriveNotify 写 commCmdInfo[本卡 epRankId] = token 数
     *     （>= 0，覆盖 sentinel）
     *   - setup 段 8 末尾本地 StoreVariable 写本卡 commCmdInfo[self] = token 数
     *   - teardown AIV 轮询本卡 commCmdInfo[r]（r in [0, epWorldSize)）：
     *       值 == -1 → src rank r 还未发到位，继续轮询
     *       值 >= 0  → src rank r 已到位（含发 0 token 的合法场景）
     *
     * 为何不依赖 0.5RTT totalCntXn 等齐：AIV 不能直接读 CCU 硬件寄存器；
     * GM 共享内存（commCmdInfo）是 AIV/CCU 都能访问的唯一同步介质。
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
            // 轮询间隔：GetValue 本身有 GM 访问延迟，无需显式 sleep。
            // 生产版本可加 PipeBarrier + 微等待以减少 cache 压力。
        }
    }

    /**
     * Process: 等齐 + 紧密拷贝 + 写三元组（v15 ★ 重写）。
     *
     * v15 关键算法变更：
     *   1. 不再按固定 segBytes 整段拷贝（含 padding），改为按 commCmdInfo[r] 实际数量紧密拷贝
     *   2. expandX 输出紧密排列：rank 0 的 commCmdInfo[0] 个 token 在前，rank 1 的接着，依此类推
     *   3. 每个 token 仍含 tokenWithMetaLen 字节（data + 32B metadata）
     *   4. 同时从每个 token 末尾的 metadata 提取三元组 (srcRank, srcSlot, dstK) 写到 assistInfoForCombine
     *      assistInfoForCombine layout: [(r0t0_meta), (r0t1_meta), ..., (r1t0_meta), ...]
     *      连续 sum_r(commCmdInfo[r]) * 3 个 INT32，与 expandX 紧密区每 token 一一对应
     */
    __aicore__ inline void Process()
    {
        if (GetBlockIdx() != 0) {
            return;
        }

        // v11 ★ 等齐：等所有 src rank 把 commCmdInfo 写过来
        WaitForAllSrcRanks();

        LocalTensor<YType> ubLocal = ubBuf_.Get<YType>();
        // v15 ★ token 在 commWindow 段中的 stride = tokenWithMetaLen / sizeof(YType) 元素
        const uint64_t tokenWithMetaElems = tokenWithMetaLen_ / sizeof(YType);
        // segment 在 commWindow 中的 stride（按元素数）
        const uint64_t segElemsCapacity = static_cast<uint64_t>(bs_) * topK_ * tokenWithMetaElems;
        // metadata 在每 token 内的起始元素偏移（按 YType 元素数；3 INT32 + padding 占 32B = 16 BF16 元素）
        const uint64_t metaOffsetElems = tokenMsgLen_ / sizeof(YType);

        // expandX 紧密拷贝目的偏移（按元素数累加）
        uint64_t dstTokenIdx = 0;   // expandX 中的紧密 token 索引（0..N-1）

        for (uint32_t r = 0; r < epWorldSize_; ++r) {
            // ====== 阶段 1：读 commCmdInfo[r] 得本段有效 token 数 ======
            int32_t srcTokens = commCmdInfoGM_.GetValue(r);

            // ====== 阶段 2：紧密拷贝 commWindow[r 段, slot 0..srcTokens-1] → expandX[紧密区 dstTokenIdx..+srcTokens] ======
            //   每个 token tokenWithMetaLen 字节，含 data + metadata（一起拷过去）
            //   段内 slot k 的 commWindow 起址 = r * segElemsCapacity + k * tokenWithMetaElems
            //   紧密区 token dstTokenIdx 的 expandX 起址 = dstTokenIdx * tokenWithMetaElems
            for (int32_t k = 0; k < srcTokens; ++k) {
                const uint64_t srcOff = static_cast<uint64_t>(r) * segElemsCapacity
                                       + static_cast<uint64_t>(k) * tokenWithMetaElems;
                const uint64_t dstOff = (dstTokenIdx + static_cast<uint64_t>(k)) * tokenWithMetaElems;

                // chunked GM→UB→GM 整 token + metadata
                for (uint64_t off = 0; off < tokenWithMetaElems; off += kChunkElems) {
                    uint32_t copyLen = static_cast<uint32_t>(
                        (tokenWithMetaElems - off >= kChunkElems) ? kChunkElems : (tokenWithMetaElems - off));
                    DataCopy(ubLocal, commWindowGM_[srcOff + off], copyLen);
                    PipeBarrier<PIPE_ALL>();
                    DataCopy(expandXGM_[dstOff + off], ubLocal, copyLen);
                    PipeBarrier<PIPE_ALL>();
                }

                // ====== 阶段 3：从该 token 末尾 metadata 提取三元组写 assistInfoForCombine ======
                //   v16+ ★ 用 Init 阶段创建的 commWindowMetaGM_（int32 视图）直接读，
                //          替代 GetPhyAddr() 临时 reinterpret（GetPhyAddr 非 ascendc 标准 API）
                //   metadata 三元组 (srcRank, srcSlot, dstK) 在 commWindow 该 token 偏移 = srcOff + metaOffsetElems
                //   按 int32 视图换算下标：metaSrcIntIdx = (srcOff + metaOffsetElems) * sizeof(YType) / sizeof(int32_t)
                const uint64_t metaSrcByteOff = (srcOff + metaOffsetElems) * sizeof(YType);
                const uint64_t metaSrcIntIdx  = metaSrcByteOff / sizeof(int32_t);
                int32_t srcRank = commWindowMetaGM_.GetValue(metaSrcIntIdx + 0);
                int32_t srcSlot = commWindowMetaGM_.GetValue(metaSrcIntIdx + 1);
                int32_t dstK    = commWindowMetaGM_.GetValue(metaSrcIntIdx + 2);

                const uint64_t assistIdx = (dstTokenIdx + static_cast<uint64_t>(k)) * 3;
                assistInfoGM_.SetValue(assistIdx + 0, srcRank);
                assistInfoGM_.SetValue(assistIdx + 1, srcSlot);
                assistInfoGM_.SetValue(assistIdx + 2, dstK);
            }

            dstTokenIdx += static_cast<uint64_t>(srcTokens);
        }

        // v15 ★ expandX 紧密区后面是 host 端预初始化的 sentinel (0xFF → metadata.srcRank = -1)
        //   combine_setup 段 5/7 host unroll bs*K 次时，padding token 的 srcRank=-1 不命中任何 r ∈ [0, rankSize)，
        //   自动被 CCU_IF 过滤掉，无需额外清理逻辑。
    }

private:
    GlobalTensor<YType> commWindowGM_;
    GlobalTensor<int32_t> commWindowMetaGM_;   // v16+ ★ commWindow 的 INT32 视图（读 metadata 用）
    GlobalTensor<YType> expandXGM_;
    GlobalTensor<int32_t> commCmdInfoGM_;
    GlobalTensor<int32_t> assistInfoGM_;
    TBuf<TPosition::VECCALC> ubBuf_;
    TPipe* pipe_;

    uint32_t epWorldSize_;
    uint32_t bs_;
    uint32_t topK_;
    uint64_t tokenMsgLen_;
    uint64_t tokenWithMetaLen_;     // v15 ★
};

/**
 * v10 ascendc kernel 入口。
 *   入参顺序对齐 op_def Input/Output（与库上 dispatch_teardown.cpp 入口一致）。
 *   host driver 通过 `<<<blockDim, l2_ctrl, stream>>>(...)` 直接 launch。
 */
extern "C" __global__ __aicore__ void moe_distribute_dispatch_ccu_teardown(
    GM_ADDR commCmdInfo, GM_ADDR commWindow, GM_ADDR expandX,
    GM_ADDR assistInfoForCombine, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeDispatchCcuTeardownTilingData);
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchCcuTeardownTilingData, tilingData, tilingGM);

    // dtype 写死 bfloat16_t；生产可按 tilingData 字段切模板（如 INT8 / FP8E5M2 等）。
    MoeDistributeDispatchCcuTeardownAiv<bfloat16_t> op;
    op.Init(commWindow, commCmdInfo, expandX, assistInfoForCombine, &pipe, &tilingData);
    op.Process();
}
