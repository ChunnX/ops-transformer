/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.hiascend.com/software/license
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file weight_l2_prefetch.hpp
 * \brief A3(910_93) 权重 L2 预取：AIV 用 fire-and-forget 的 GM->UB DataCopyPad 把专家权重
 *        字节拉进 L2，供 AIC 的 copyGmToL1B(GM->L1, MTE1) 读命中；UB 数据从不消费。
 *        机制移植自 arch35（common/mega_moe_utils.h 的 WeightL2Prefetch*），按 A3 的
 *        192KB UB 与空闲窗口位置重规划。
 */

#ifndef MEGA_MOE_WEIGHT_L2_PREFETCH_HPP
#define MEGA_MOE_WEIGHT_L2_PREFETCH_HPP

#include "kernel_operator.h"

#include "utils/get_tensor_addr.hpp"
#include "template_linear_algebra_v2/arch/mega_moe_arch.hpp"

/*
 * scratch 位于 UB 顶部洞 [179552, 195936)：
 *   - 下界 179552 = W4A8 swiglu BUFFER_SIZE(146784, TILE_LENGTH=3072 x UB_STAGES=2,
 *     block_epilogue_w4a8post_pertoken_swiglu.hpp:51-54) + MIN_SHARED_TMP_BYTES(32768,
 *     gated_activation.hpp:35)——该 epilogue 的 static_assert 即按此合约检查；
 *   - 上界让出 rankMask 常驻区 [UB_SIZE - AlignUp(EP*4,32), UB_SIZE)（kernel_a3.hpp:406-412，
 *     EP<=128 -> <=512B），最小间隙 160B。
 * 各阶段功能区上界（修改任一布局须同步校验本偏移）：dispatch <=75K；swiglu <=179552；
 * combine <=150528；unpermute ~= 10*K+7.5KB（<=89.5K@K=8192）。routing 时序在预取窗口之前。
 */
constexpr uint32_t WEIGHT_L2_PREFETCH_UB_OFFSET = 179552U;
constexpr uint32_t WEIGHT_L2_PREFETCH_CHUNK_BYTES = 16U * 1024U;
/*
 * MTE2_V 方向事件 ID 分时占用：dispatch 期 W4A8 用 6/7、int8/bf16 仅用 0 -> ID5 空闲；
 * swiglu/combine 期功能侧用 0-3 -> ID6 空闲（W4A8 dispatch 的 6/7 在每次调用尾已排空）。
 * 功能侧全部为 Set->Wait 背靠背配对，会话以 Drain 闭环后 ID 即释放。
 */
constexpr uint32_t WEIGHT_L2_PREFETCH_EVENT_ID_WINDOW0 = 5U;
constexpr uint32_t WEIGHT_L2_PREFETCH_EVENT_ID_WINDOW1 = 6U;
// 窗口 1 会话的全局预算（防过量预热挤掉 L2 内有复用价值的数据）；游标内部再夹紧到权重总量。
constexpr uint32_t WEIGHT_L2_PREFETCH_MAX_BUDGET_BYTES = 16U * 1024U * 1024U;
// 窗口 0 每空闲 AIV 的 chunk 配额（总量 = 空闲核数 x 256KB，如 EP=8/aivNum=40 时 8MB）。
constexpr uint32_t WEIGHT_L2_PREFETCH_WINDOW0_CHUNKS_PER_IDLE_AIV = 16U;
// 窗口 1 每次会话（每次等 SYNCFLAGC2V 前）的 burst 上限：128KB ~= 1-2us，钳制 Drain 过冲。
constexpr uint32_t WEIGHT_L2_PREFETCH_SESSION_CHUNKS = 8U;

static_assert(WEIGHT_L2_PREFETCH_UB_OFFSET % 32U == 0U, "prefetch scratch must be 32B aligned");
static_assert(WEIGHT_L2_PREFETCH_UB_OFFSET + WEIGHT_L2_PREFETCH_CHUNK_BYTES + 512U <= Catlass::Arch::AtlasA2::UB_SIZE,
              "prefetch scratch must leave room for the top rankMask region");

// 预取游标：区域序为先全部 W1 再全部 W2（与 kernel 消费序一致），区域内按 worker 交错认领 chunk。
struct WeightL2PrefetchCursor {
    __gm__ uint8_t *chunkAddr = nullptr; // 本 worker 在当前区域内的下一 chunk 地址
    __gm__ uint8_t *regionEnd = nullptr; // 当前区域结束地址（exclusive）
    uint32_t regionIndex = 0U;           // [0,E)=W1 各专家，[E,2E)=W2 各专家
    uint32_t regionCount = 0U;           // 2*expertCount；regionIndex>=regionCount 表示耗尽
    uint32_t workerIdx = 0U;
    uint32_t workerCount = 1U;
    uint64_t budgetRemaining = 0U; // 本 worker 的字节预算（全局预算/workerCount）
    GM_ADDR weight1List = nullptr;
    GM_ADDR weight2List = nullptr;
    bool isPerExpertWeightTensor = false; // listLen!=1：每专家独立 tensor；否则单堆叠 tensor 步进
    uint32_t expertCount = 0U;
    uint64_t w1BytesPerExpert = 0U; // 调用方按 dtype 算好字节直接传入（规避 0.5 字节/element 的整除）
    uint64_t w2BytesPerExpert = 0U;
    AscendC::LocalTensor<uint8_t> scratch; // [WEIGHT_L2_PREFETCH_UB_OFFSET, +CHUNK) 的单缓冲
    bool active = false;
};

// 窗口 0 确定性覆盖总量（全部空闲核一致计算，供窗口 1 startOffset 接续；恒为 CHUNK 的整数倍）。
__aicore__ inline uint64_t WeightL2PrefetchWindow0TotalBytes(uint32_t idleWorkerCount)
{
    return static_cast<uint64_t>(WEIGHT_L2_PREFETCH_WINDOW0_CHUNKS_PER_IDLE_AIV) * WEIGHT_L2_PREFETCH_CHUNK_BYTES *
           idleWorkerCount;
}

// 把 regionIndex 映射到 (tensor list, expertIdx, 区域字节数) 并定位本 worker 的起始 chunk。
__aicore__ inline void WeightL2PrefetchSetupRegion(WeightL2PrefetchCursor &cursor, uint64_t skipInRegion)
{
    if (cursor.regionIndex >= cursor.regionCount) {
        return;
    }
    bool isW1 = cursor.regionIndex < cursor.expertCount;
    uint32_t expertIdx = isW1 ? cursor.regionIndex : cursor.regionIndex - cursor.expertCount;
    uint64_t regionBytes = isW1 ? cursor.w1BytesPerExpert : cursor.w2BytesPerExpert;
    GM_ADDR list = isW1 ? cursor.weight1List : cursor.weight2List;
    __gm__ uint8_t *regionBase = GetTensorAddr<uint8_t>(cursor.isPerExpertWeightTensor ? expertIdx : 0U, list);
    if (!cursor.isPerExpertWeightTensor) {
        // listLen==1 堆叠模式：与 kernel 的 gmGroupOffsetB 元素步进同源（元素数 x dtype 字节）。
        regionBase += static_cast<uint64_t>(expertIdx) * regionBytes;
    }
    cursor.regionEnd = regionBase + regionBytes;
    // 区域内 chunk 交错认领：worker c 从偏移 (skip + c*CHUNK) mod regionBytes 起步；回绕重叠由
    // L2 去重，无害。
    uint64_t startOffset =
        regionBytes == 0U ?
            0U :
            (skipInRegion + static_cast<uint64_t>(cursor.workerIdx) * WEIGHT_L2_PREFETCH_CHUNK_BYTES) % regionBytes;
    cursor.chunkAddr = regionBase + startOffset;
}

// 本 worker 在当前区域的 chunk 走完后推进到下一区域（全部 W1 之后是全部 W2）。
__aicore__ inline void WeightL2PrefetchAdvanceRegion(WeightL2PrefetchCursor &cursor)
{
    ++cursor.regionIndex;
    WeightL2PrefetchSetupRegion(cursor, 0U);
}

/*
 * 初始化预取游标。budgetBytes 为全部 worker 共享的全局预算，内部按 workerCount 均摊（不足一个
 * chunk 时保底 1 个）。startOffsetBytes 为起始字节偏移（窗口 1 传窗口 0 已覆盖总量以接续）：
 * 跨过的整区域直接跳过，落点区域的剩余部分作为初始 skip 参与交错。
 */
__aicore__ inline void WeightL2PrefetchInit(WeightL2PrefetchCursor &cursor, GM_ADDR weight1List, GM_ADDR weight2List,
                                            bool isPerExpertWeightTensor, uint64_t w1BytesPerExpert,
                                            uint64_t w2BytesPerExpert, uint32_t expertCount, uint32_t workerIdx,
                                            uint32_t workerCount, uint64_t budgetBytes, uint64_t startOffsetBytes,
                                            AscendC::LocalTensor<uint8_t> scratchTensor)
{
    cursor.weight1List = weight1List;
    cursor.weight2List = weight2List;
    cursor.isPerExpertWeightTensor = isPerExpertWeightTensor;
    cursor.expertCount = expertCount;
    cursor.regionCount = 2U * expertCount;
    cursor.workerIdx = workerIdx;
    cursor.workerCount = workerCount > 0U ? workerCount : 1U;
    cursor.w1BytesPerExpert = w1BytesPerExpert;
    cursor.w2BytesPerExpert = w2BytesPerExpert;
    cursor.scratch = scratchTensor;
    cursor.scratch.SetSize(WEIGHT_L2_PREFETCH_CHUNK_BYTES);
    cursor.active = (workerCount > 0U && expertCount > 0U);
    cursor.budgetRemaining = budgetBytes / cursor.workerCount;
    if (budgetBytes != 0U && cursor.budgetRemaining == 0U) {
        cursor.budgetRemaining = WEIGHT_L2_PREFETCH_CHUNK_BYTES;
    }
    cursor.regionIndex = 0U;
    uint64_t skipInRegion = startOffsetBytes;
    while (cursor.regionIndex < cursor.regionCount && skipInRegion > 0U) {
        bool isW1 = cursor.regionIndex < cursor.expertCount;
        uint64_t regionBytes = isW1 ? cursor.w1BytesPerExpert : cursor.w2BytesPerExpert;
        if (skipInRegion >= regionBytes) {
            skipInRegion -= regionBytes;
            ++cursor.regionIndex;
        } else {
            break;
        }
    }
    WeightL2PrefetchSetupRegion(cursor, skipInRegion);
}

/*
 * 发射一个 chunk 的 GM->UB fire-and-forget 拷贝（不等待完成）；返回 false 表示游标耗尽（预算
 * 用完或全部区域走完）。单缓冲 scratch 即可：同核 MTE2 队列按程序序执行保证 UB 写-写按序，
 * 且 scratch 数据从不被消费。
 */
__aicore__ inline bool WeightL2PrefetchTryIssue(WeightL2PrefetchCursor &cursor)
{
    while (cursor.active && cursor.regionIndex < cursor.regionCount && cursor.budgetRemaining != 0U) {
        if (cursor.chunkAddr >= cursor.regionEnd) {
            WeightL2PrefetchAdvanceRegion(cursor);
            continue;
        }
        uint64_t bytesToRegionEnd = static_cast<uint64_t>(cursor.regionEnd - cursor.chunkAddr);
        // 拷贝量同时被区域尾与剩余预算钳住，防止预算均摊后不足一个 chunk 时 uint64 回绕。
        uint64_t copyLimit = bytesToRegionEnd < cursor.budgetRemaining ? bytesToRegionEnd : cursor.budgetRemaining;
        uint32_t copyBytes = copyLimit < WEIGHT_L2_PREFETCH_CHUNK_BYTES ? static_cast<uint32_t>(copyLimit) :
                                                                          WEIGHT_L2_PREFETCH_CHUNK_BYTES;
        AscendC::GlobalTensor<uint8_t> weightGm;
        weightGm.SetGlobalBuffer(cursor.chunkAddr);
        AscendC::DataCopyPad(cursor.scratch, weightGm, {1, static_cast<uint16_t>(copyBytes), 0, 0}, {});
        cursor.budgetRemaining -= copyBytes;
        // 下一 chunk 跳过其它 worker 认领的 chunk（步长 = workerCount 个 chunk）。
        cursor.chunkAddr += static_cast<uint64_t>(cursor.workerCount) * WEIGHT_L2_PREFETCH_CHUNK_BYTES;
        return true;
    }
    return false;
}

// 有界 burst：会话/每组循环的前置发射（TryIssue 耗尽即提前返回）。
__aicore__ inline void WeightL2PrefetchBurst(WeightL2PrefetchCursor &cursor, uint32_t maxChunks)
{
    for (uint32_t issued = 0U; issued < maxChunks; ++issued) {
        if (!WeightL2PrefetchTryIssue(cursor)) {
            break;
        }
    }
}

// 排空在飞的预取拷贝：SetFlag 进入 MTE2 队列、按序在全部已发射拷贝完成后置位事件。每个预取
// 会话必须以 Drain 收尾（含上一窗口的残留块），防止与后续功能流水访问交错。
__aicore__ inline void WeightL2PrefetchDrain(uint32_t eventId)
{
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
}

#endif // MEGA_MOE_WEIGHT_L2_PREFETCH_HPP
