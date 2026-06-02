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
 *       WaitallpeersarriveNotify 等），跑在 CCU 引擎；通过 HcclCcuKernelRegister/Launch。
 *   v10：teardown 是标准 ascendc kernel（`__global__ __aicore__` 函数），跑在 AIV；
 *        用 GlobalTensor / LocalTensor / DataCopy 等标准 ascendc API；通过 `<<<>>>` 直调。
 *
 * 等齐机制：
 *   v9 teardown 内部用 WaitallpeersarriveNotify 等 0.5RTT 通信完成。
 *   v10 等齐工作上移到 setup CCU 末尾（segment 9），setup CCU return 时所有对端数据
 *   已就位；stream 顺序保证 v10 teardown launch 时 y / commCmdInfo 内容稳定。
 *
 * Algorithm（v20.6 ★，纯 vector 计算，无任何通信）：
 *   1. 轮询 commCmdInfo ready 区 [0, R) 全 1 等齐（ready 区在偏移 0）
 *   2.【按发送方分段自数·Req3】最外层按来源卡 r 循环，遍历段 r 的 bs*topK 个 slot，按 metadata.srcRank==r
 *      计有效 token（空 slot 0xFF→-1 跳过）；命中即把【干净 data(tokenMsgLen)】紧密 DataCopy 到 expandX
 *      （剥离 metadata·Req1），expandX/assistInfo 天然按 ownerRank 分段连续。
 *   3. 同 slot metadata 写 assistInfo[t]=dstByteOff（单字段 int64，combine 反向回送用；owner 由分段隐含）。
 *   4. 段 r 末写 commCmdInfo.sendBytes[r]=cnt[r]*tokenMsgLen（combine wishCnt）+ negCnt[r]=-cnt[r]
 *      （combine 按 owner 分段·每段 CCU_WHILE 用 negCnt[r] 自增到 0 跑 cnt[r] 次，规避 CCU 无减/乘）。
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
        // v20.6 ★ commWindow 的 INT64 视图：metadata 按 int64 排布（srcRank@字段0 / dstByteOff@字段3），
        //   直接 GetValue 读全 64 位（规避 int32 低半读 dstByteOff 截断），与 CCU LoadVariable 8 字节同口径。
        commWindowMetaI64GM_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(commWindow));
        expandXGM_.SetGlobalBuffer((__gm__ YType*)expandX);
        commCmdInfoGM_.SetGlobalBuffer((__gm__ int32_t*)commCmdInfo);
        // v20.7 ★ commCmdInfo 的 INT64 视图：写 sendBytes[0..R)（段6 wishCnt）+ negCnt[0..R)（=-cnt[r]，
        //   combine 按 owner 分段倒计数）。combine_setup CCU 用 LoadVariable 读（必 int64）。ready 区仍走
        //   int32 视图 commCmdInfoGM_（与 setup LocalCopyNb 写的 int32 一致）。
        commCmdInfoI64GM_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(commCmdInfo));
        // v20.7 ★ assistInfo 降为【单字段 dstByteOff】(1×int64/token)：combine 按 owner 分段消费，owner=外层
        //   循环常量已知，无需存 ownerRank——只需回送目的卡 commWindow 内字节偏移 dstByteOff。combine LoadVariable 读。
        assistInfoI64GM_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(assistInfoForCombine));
        pipe_ = pipe;

        // UB chunk buffer：用作 GM→UB→GM 搬运的中转
        // ascendc 不直接支持 GM→GM DataCopy，必须经 UB 中转
        pipe_->InitBuffer(ubBuf_, kChunkElems * sizeof(YType));
    }

    /**
     * 收口 ★ 等齐阶段：轮询本卡 commCmdInfo 的 ready 区 [0, R) 等全为 1。
     *
     * 双流并行架构：setup CCU 立即返回（异步），teardown AIV 内部等齐。
     *
     * 等齐信号 = "专用 ready 区全 1"。setup CCU 段 9 内部 WaitallpeersarriveNotify(R-1) 确认本卡作为
     *   接收方已收齐 *所有* 对端数据，段 10 用【1 次 LocalCopyNb】把 oneBuf（host 填 R 个 int32=1）
     *   整块写到 ready 区，置 ready[0..R) 全 1。所以 ready 区全 1 ⟺ 本卡数据彻底落地。
     *
     * 布局收口：commCmdInfo 简化为只剩 ready 区，从偏移 0 起（删了 tokenCount/scratch 区）。
     *   ready 区 int32 下标 = r（readyBase = 0）。tokenCount 不再由对端跨卡写——本卡每段的有效 token
     *   数改由本 teardown 在 Process 里【自数】（遍历 commWindow 段 r 的 slot 按 metadata.srcRank 判定）。
     *   ★ setup 段 10 写的 readyAddr 偏移（0）== 此处 readyBase（0），务必一致。
     *
     * 为何不依赖 0.5RTT totalCntXn 等齐：AIV 不能直接读 CCU 硬件寄存器；
     * GM 共享内存（commCmdInfo ready 区）是 AIV/CCU 都能访问的唯一同步介质。
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
            // 轮询间隔：GetValue 本身有 GM 访问延迟，无需显式 sleep。
        }
    }

    /**
     * Process（v20.7）: 等齐 + 按发送方分段自数 + 紧密拷【干净 data】 + 写 assistInfo/commCmdInfo。
     *
     * 关键算法：
     *   1. 等齐后遍历 commWindow 段 r（来自卡 r）全部 slot，按 metadata.srcRank==r 自数有效 token 数
     *      （空 slot 0xFF→srcRank=-1 跳过）。
     *   2. expandX 输出紧密排列且按 ownerRank 分段连续：卡0 的 token 在前、卡1 接着…（combine 按段消费的前提）。
     *      每个 token 只拷【干净 data】(tokenMsgLen，剥离 metadata，Req1)。
     *   3. assistInfoForCombine 写【单字段 dstByteOff】(int64)：combine 按 owner 分段、owner 已知，只需回送偏移。
     *      layout：连续 dstTokenIdx 个 INT64，与 expandX 紧密区每 token 一一对应。
     *   4. 段 r 末写 commCmdInfo.sendBytes[r]=cnt[r]*tokenMsgLen（combine 段6 wishCnt）+ negCnt[r]=-cnt[r]
     *      （combine 段7 该段 CCU_WHILE 倒计数初值）。
     */
    __aicore__ inline void Process()
    {
        if (GetBlockIdx() != 0) {
            return;
        }

        // v11 ★ 等齐：等所有 src rank 把 commCmdInfo 写过来
        WaitForAllSrcRanks();

        LocalTensor<YType> ubLocal = ubBuf_.Get<YType>();
        // commWindow 槽 stride（含 metadata，按字节）：每槽 = tokenMsgLen data + 32B metadata
        const uint64_t tokenWithMetaBytes = tokenWithMetaLen_;
        // v20.6 ★ expandX 干净 stride（按 YType 元素数）：只搬 data 段，不含 metadata
        const uint64_t tokenMsgElems = tokenMsgLen_ / sizeof(YType);
        // 段 r 在 commWindow 中的 stride（按字节）
        const uint64_t segByteCapacity = static_cast<uint64_t>(bs_) * topK_ * tokenWithMetaBytes;
        const uint32_t segSlotCapacity = bs_ * topK_;   // 段 r 的 slot 容量

        // v20.7 ★ commCmdInfo 布局（与 combine_setup CCU / driver 三处严格一致）：
        //   ready[0..R) int32（byte [0,4R)）│ 8 字节对齐 │ sendBytes[0..R) int64 │ negCnt[0..R) int64
        //   cinfoI64Base = 8 对齐后的 int64 基下标；sendBytes[r] @ +r；negCnt[r] @ +R+r。
        const uint64_t cinfoByteOff = ((static_cast<uint64_t>(epWorldSize_) * sizeof(int32_t) + 7U) / 8U) * 8U;
        const uint64_t cinfoI64Base = cinfoByteOff / sizeof(int64_t);

        // expandX 紧密区推进下标（= 全卡有效 token 总数累加器）
        uint64_t dstTokenIdx = 0;

        // v20.6 ★【按发送方分段自数】（Req3）：最外层按"来源卡 r"循环，遍历段 r 的全部 bs*topK 个 slot，
        //   读 metadata.srcRank（int64 全宽，空 slot 0xFF→-1）。仅 srcRank==r 的才是 r 真实发来的有效 token。
        //   命中即：① 干净 data（tokenMsgLen）紧密拷到 expandX（剥离 metadata，Req1）；
        //          ② 写 assistInfo[dstTokenIdx]=dstByteOff（单字段，combine 路由用，int64）；
        //          ③ dstTokenIdx++。段 r 结束记 cnt[r]，写 sendBytes[r]=cnt[r]*tokenMsgLen + negCnt[r]=-cnt[r]。
        //   ★ 关键不变式：因按 r 顺序紧密堆放，expandX/assistInfo 天然【按 ownerRank 严格分段连续】
        //     （来自卡0 的 cnt[0] 个、卡1 的 cnt[1] 个…依次紧密）。这正是 combine 按 owner 分段消费的前提。
        //   0xFF sentinel 由 driver 预初始化（区分 rank0 真 token 与空槽）。
        for (uint32_t r = 0; r < epWorldSize_; ++r) {
            const uint64_t cntBefore = dstTokenIdx;
            for (uint32_t k = 0; k < segSlotCapacity; ++k) {
                const uint64_t tokenStartByte = static_cast<uint64_t>(r) * segByteCapacity
                                              + static_cast<uint64_t>(k) * tokenWithMetaBytes;
                // metadata 起始 int64 下标（metadata 紧跟 data 段：byte = tokenStart + tokenMsgLen，8 对齐）
                const uint64_t metaI64Idx = (tokenStartByte + tokenMsgLen_) / sizeof(int64_t);
                const int64_t srcRank = commWindowMetaI64GM_.GetValue(metaI64Idx + 0);   // 字段0
                if (srcRank != static_cast<int64_t>(r)) {
                    continue;   // 空 slot（srcRank=-1）或非本段数据，跳过，不占紧密区
                }
                const int64_t dstByteOff = commWindowMetaI64GM_.GetValue(metaI64Idx + 3); // 字段3（=24/8）

                // ① 紧密拷贝【干净 data】：commWindow[段r,槽k] data 段(tokenMsgLen) → expandX[紧密区]
                const uint64_t srcElem = tokenStartByte / sizeof(YType);   // 槽起始（YType 元素）
                const uint64_t dstElem = dstTokenIdx * tokenMsgElems;      // expandX 干净 stride
                for (uint64_t off = 0; off < tokenMsgElems; off += kChunkElems) {
                    uint32_t copyLen = static_cast<uint32_t>(
                        (tokenMsgElems - off >= kChunkElems) ? kChunkElems : (tokenMsgElems - off));
                    DataCopy(ubLocal, commWindowGM_[srcElem + off], copyLen);
                    PipeBarrier<PIPE_ALL>();
                    DataCopy(expandXGM_[dstElem + off], ubLocal, copyLen);
                    PipeBarrier<PIPE_ALL>();
                }

                // ② 写 assistInfo（单字段 dstByteOff）：combine 按 owner 分段消费、owner 已知，只需回送偏移
                assistInfoI64GM_.SetValue(dstTokenIdx, dstByteOff);

                ++dstTokenIdx;   // ③ 自数推进
            }
            // v20.7 ★ 段 r 结束：cnt[r] = 本段有效 token 数。写两份供 combine：
            //   sendBytes[r] = cnt[r]*tokenMsgLen（段6 RecordExpectCount wishCnt，combine 干净字节）；
            //   negCnt[r]    = (uint64_t)(0 - cnt[r])（段7 该段 CCU_WHILE 倒计数初值，自增到 0 跑 cnt[r] 次）。
            //   倒计数在 AIV 侧用普通 C++ 算（CCU Variable 无减/乘）；cnt[r]==0 → negCnt[r]=0 → 该段 0 次，自然正确。
            const uint64_t cntR = dstTokenIdx - cntBefore;
            commCmdInfoI64GM_.SetValue(cinfoI64Base + r,
                                       static_cast<int64_t>(cntR * tokenMsgLen_));
            commCmdInfoI64GM_.SetValue(cinfoI64Base + epWorldSize_ + r,
                                       static_cast<int64_t>(static_cast<uint64_t>(0) - cntR));
        }
        // v20.7 ★ 不再写全局 negTotal：combine 改按 owner 分段（每段用 negCnt[r] 倒计数），无需全局 total。
    }

private:
    GlobalTensor<YType> commWindowGM_;
    GlobalTensor<int64_t> commWindowMetaI64GM_; // v20.6 ★ commWindow 的 INT64 视图（读 metadata 全宽）
    GlobalTensor<YType> expandXGM_;
    GlobalTensor<int32_t> commCmdInfoGM_;       // ready 区 int32 轮询
    GlobalTensor<int64_t> commCmdInfoI64GM_;    // v20.7 ★ 写 sendBytes[0..R) + negCnt[0..R)（combine 读，int64）
    GlobalTensor<int64_t> assistInfoI64GM_;     // v20.7 ★ 单字段 dstByteOff（1×int64/token，owner 由分段隐含）
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
