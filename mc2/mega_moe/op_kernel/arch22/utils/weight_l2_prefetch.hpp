/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_l2_prefetch.hpp
 * \brief A3(910_93) 权重 L2 预取：AIV 用 fire-and-forget 的 GM->UB DataCopyPad 把专家权重
 *        字节拉进 L2，供 AIC 的 copyGmToL1B(GM->L1, MTE1) 读命中；UB 数据从不消费。
 *        四条预取路径由独立 constexpr 窗口开关控制（false 时对应构造点不构造、不发射，
 *        编译期裁剪零运行时开销；扫参/消融直接编辑本文件常量）。当前落地策略（W0 固定区
 *        实验）：仅窗口 0 开启——全体 AIV 交错认领 W1 头部固定区（前 W0_EXPERTS 个专家、
 *        W0_BUDGET_BYTES 预算双钳，游标固定不随 GMM1 消费前沿推进），发射点在 AllGather
 *        计数 push 与自旋两循环之间一次性发完——计数 push 已入队不被压队，随后的跨卡自旋
 *        （等最慢 rank 的计数标记）为最长 RTT 空闲窗口，预取的 HBM 读整段被掩盖；在飞块
 *        由函数内 PipeBarrier<PIPE_ALL>/SyncAll<true> 全流水屏障收口；W1 AIC 冷启动、
 *        W2 GMM2 段界、窗口 1（AIV W2 会话）三条路径休眠保留。
 *        机制对标 arch35 的同类 L2 权重预取设计，按 A3 的 192KB UB 与空闲窗口重规划。
 *        本仓适配（相对实现仓）：arch 常量改用三方 catlass；UB 顶部新增 rankMask+2xrouteCache
 *        常驻区，AIV scratch 洞上界改为运行时自适应（chunkBytes 按洞收缩，见 kernel 侧
 *        ComputeAivPrefetchHoleBytes 与 Init 的下限判定）。
 */

#ifndef MEGA_MOE_WEIGHT_L2_PREFETCH_HPP
#define MEGA_MOE_WEIGHT_L2_PREFETCH_HPP

#include "kernel_operator.h"

#include "get_tensor_addr.hpp"
#include "catlass/arch/arch.hpp"

// ---------------------------------------------------------------------------
// 窗口开关（constexpr 编译期裁剪；false 时对应路径不构造、不发射，零运行时开销）
// ---------------------------------------------------------------------------
// 窗口 0：AIV W1 固定区——发射点在 AllGather 计数 push 与自旋两循环之间（函数级公共路径，
//         EP<coreNum 时部分核零 push 迭代），一次性发完全部配额：计数 push 已入队不被压队，
//         随后的跨卡自旋为最长 RTT 空闲窗口，预取的 HBM 读整段被掩盖；全体 AIV 交错认领，
//         在飞块由函数内全流水屏障（PipeBarrier<PIPE_ALL>/SyncAll<true>）收口。当前实验主路径。
constexpr bool WEIGHT_L2_PREFETCH_ENABLE_WINDOW0 = true;
// 窗口 1：AIV W2 会话（遗留休眠）——swiglu 会话等 SYNCFLAGC2V 前 Retarget+burst 发射，会话
//         内 Drain 收口。实测双机负收益，已被 GMM2 段界路径取代。
constexpr bool WEIGHT_L2_PREFETCH_ENABLE_WINDOW1 = false;
// W1 AIC 冷启动（休眠）：GMM1 首个轮就绪 wait 前 GM->L1 哑拷贝预热前 N 个专家的 W1。
constexpr bool WEIGHT_L2_PREFETCH_ENABLE_W1_AIC_COLDSTART = false;
// W2 GMM2 段界预取（休眠）：GMM2 组循环每段 V2C wait 前 GM->L1 哑拷贝预热即将消费段的 W2。
constexpr bool WEIGHT_L2_PREFETCH_ENABLE_W2_GMM2_SECTION = false;

// 防呆：窗口 1 与 GMM2 段界都对 W2 预取，同时开启会对同一批 W2 重复搬运。
static_assert(!(WEIGHT_L2_PREFETCH_ENABLE_WINDOW1 && WEIGHT_L2_PREFETCH_ENABLE_W2_GMM2_SECTION),
              "window-1 session prefetch and GMM2-section W2 prefetch are mutually exclusive: "
              "duplicate W2 traffic");
/*
 * GMM L2 hint 联动（kernel 侧 if constexpr，非宏）：被预取的权重必须配 NORMAL——DISABLE 读写
 * 双向 bypass（实测）会使预取完全失效；未被预取的一侧保持基线条件 DISABLE（currentM<=单
 * M-tile 即权重单次读时 bypass）。GMM1 随窗口 0/W1 冷启动，GMM2 随窗口 1/W2 段界——由
 * kernel 侧 kGmm1/kGmm2WeightPrefetched 常量承接，hint 一致性由 if constexpr 结构性保证。
 */

// ---------------------------------------------------------------------------
// 窗口 0 剂量（固定区双钳：专家区与预算先到先停）
// ---------------------------------------------------------------------------
// 固定区覆盖的头部专家数：全体 AIV 从专家 0 起交错认领，区间走完即停——游标固定不随 GMM1
// 消费前沿推进（固定剂量便于归因，超出的专家保持冷 miss）。
constexpr uint32_t WEIGHT_L2_PREFETCH_W0_EXPERTS = 2U;
// 固定区全局字节预算：防过量预热挤掉 L2 内有复用价值的数据。发射点在 AllGather 计数 push
// 与自旋两循环之间（一次性发完），排空窗口 = 自旋等最慢 rank 的跨卡 RTT + 后续阶段全程。
constexpr uint32_t WEIGHT_L2_PREFETCH_W0_BUDGET_BYTES = 16U * 1024U * 1024U;

// ---------------------------------------------------------------------------
// 休眠路径剂量（对应开关关闭时不起作用；重启用时即为机制默认值）
// ---------------------------------------------------------------------------
// AIC W1 冷启动：专家数上限（覆盖 GMM1 最先消费的前 N 个专家，其后专家回到冷 miss）与每专家
// 每核 chunk 上限（AicRange 内按 capChunks x coreNum x CHUNK 与专家字节双钳）。发射点在首个
// 轮就绪 wait 之前，排空窗口 = prologue/AllGather/cumsum 等待 + 组 0 pull 全程，与 pull 的
// MTE2 零争抢。实测（发射点在 wait 之后时）全覆盖 128 负收益：排空尾部顶住 GMM1 首个真实
// B-tile 载入 + 与 pull 争带宽；前移后窗口扩大，可上调（0 等效关闭）。
constexpr uint32_t WEIGHT_L2_PREFETCH_W0_AIC_COLDSTART_EXPERTS = 4U;
constexpr uint32_t WEIGHT_L2_PREFETCH_W0_AIC_COLDSTART_CHUNKS = 64U;
// GMM2 段界 W2 预取每专家每 AIC 核 chunk 上限：wait 返回即消费（零距离，无驱逐窗口），剂量
// 上限钳制对 AIV 本段 swiglu gmC 读的 outstanding 干扰；首段无等待窗口不预取（保持冷）。
constexpr uint32_t WEIGHT_L2_PREFETCH_W2_GMM2_CHUNKS = 8U;
// 窗口 1 每会话（每次等 SYNCFLAGC2V 前）的 burst 上限（128KB ~= 1-2us，钳制 Drain 过冲）与
// 全局预算（防过量预热挤掉 L2 内有复用价值的数据；游标内部再夹紧到权重总量）。
constexpr uint32_t WEIGHT_L2_PREFETCH_SESSION_CHUNKS = 8U;
constexpr uint32_t WEIGHT_L2_PREFETCH_MAX_BUDGET_BYTES = 16U * 1024U * 1024U;

// ---------------------------------------------------------------------------
// 公共常量
// ---------------------------------------------------------------------------
/*
 * AIV scratch 位于 UB 顶部洞（下界固定、上界自适应）：
 *   - 下界 179552 = W4A8 swiglu BUFFER_SIZE(146784, TILE_LENGTH=3072 x UB_STAGES=2,
 *     block_epilogue_w4a8post_pertoken_swiglu.hpp) + MIN_SHARED_TMP_BYTES(32768,
 *     gated_activation.hpp)——该 epilogue 的 static_assert 即按此合约检查；
 *   - 上界让出本仓 UB 顶部常驻区 [rankMask 保留区][preSumBeforeRank 缓存][tokenPerExpert 缓存]
 *     （kernel_a3.hpp initBuffer：UB_SIZE - AlignUp(EP*4,32) - 2*AlignUp(EP*epr*4,32)），洞宽
 *     由 kernel 侧运行时计算并作为 chunkBytes 传入 Init（典型 EP=8/epr=16 时洞=15.5KB）。
 * 各阶段功能区上界（修改任一布局须同步校验本偏移）：dispatch <=75K；swiglu <=179552；
 * combine <=150528；unpermute ~= 10*K+7.5KB（<=89.5K@K=8192）。routing 时序在预取窗口之前。
 */
constexpr uint32_t WEIGHT_L2_PREFETCH_UB_OFFSET = 179552U;
constexpr uint32_t WEIGHT_L2_PREFETCH_CHUNK_BYTES = 16U * 1024U;
// AIV 自适应洞的最小 chunk 字节数：洞不足时该游标 inactive（路径 no-op），绝不越界写顶部常驻区。
constexpr uint32_t WEIGHT_L2_PREFETCH_AIV_MIN_CHUNK_BYTES = 2048U;
/*
 * 预取 Drain 的 MTE2_V 事件 ID：窗口 0 在轮就绪信号前、窗口 1 在每个会话内分时复用
 * （Set->Wait 背靠背配对，Drain 闭环后 ID 即释放）——swiglu/combine 期功能侧仅占 0-3，ID6
 * 空闲（dispatch 期 W4A8 的 6/7 在进入 swiglu 前已排空）。
 */
constexpr uint32_t WEIGHT_L2_PREFETCH_MTE2_V_EVENT_ID = 6U;

static_assert(WEIGHT_L2_PREFETCH_UB_OFFSET % 32U == 0U, "prefetch scratch must be 32B aligned");
/*
 * 下式仅静态兜底下界与 rankMask 最小预留；实际防重叠由运行时自适应洞宽保证（Init 的
 * chunkBytes 下限判定 + kernel 侧洞宽计算镜像 initBuffer 布局算术）——本仓 UB 顶部常驻区
 * 大于 512B（rankMask + 2xrouteCache，EP*epr>=32 即超过固定 16KB chunk 的剩余间隙），固定
 * chunk 写满 [UB_OFFSET, +16KB) 会下探踩踏 routeCache，故 AIV 路径必须走自适应 chunkBytes。
 */
static_assert(WEIGHT_L2_PREFETCH_UB_OFFSET + WEIGHT_L2_PREFETCH_CHUNK_BYTES + 512U <= Catlass::Arch::AtlasA2::UB_SIZE,
              "prefetch scratch must leave room for the top rankMask region");

// 预取游标：单一权重矩阵（W1 或 W2）的各专家区域。调用方按流水线消费进度 Retarget 目标专家
// 区间，区间内按 worker 交错认领 chunk；区间走完即停，不自动溢出到后续专家（零死字节）。
struct WeightL2PrefetchCursor {
    __gm__ uint8_t *chunkAddr = nullptr; // 本 worker 在当前区域内的下一 chunk 地址
    __gm__ uint8_t *regionEnd = nullptr; // 当前区域结束地址（exclusive）
    uint32_t regionIndex = 0U;           // 当前目标专家（每次 Retarget 重置）
    uint32_t regionLimit = 0U;           // 本次 Retarget 的专家区间上界（exclusive）
    uint32_t regionCount = 0U;           // expertCount；regionIndex>=regionCount 表示耗尽
    uint32_t workerIdx = 0U;
    uint32_t workerCount = 1U;
    uint64_t budgetRemaining = 0U; // 本 worker 的字节预算（全局预算/workerCount）
    uint64_t issuedBytes = 0U;     // 遥测：本游标累计已发射字节（相位计时/覆盖率核查用）
    GM_ADDR weightList = nullptr;
    bool isPerExpertWeightTensor = false; // listLen!=1：每专家独立 tensor；否则单堆叠 tensor 步进
    uint32_t expertCount = 0U;
    uint64_t bytesPerExpert = 0U; // 调用方按 dtype 算好字节直接传入（规避 0.5 字节/element 的整除）
    AscendC::LocalTensor<uint8_t> scratch;                // [WEIGHT_L2_PREFETCH_UB_OFFSET, +chunkBytes) 的单缓冲
    uint32_t chunkBytes = WEIGHT_L2_PREFETCH_CHUNK_BYTES; // AIC 路径固定 16KB；AIV 按 UB 洞自适应
    bool active = false;
};

// 把 regionIndex 映射到 (tensor list, expertIdx, 区域字节数) 并定位本 worker 的起始 chunk。
__aicore__ inline void WeightL2PrefetchSetupRegion(WeightL2PrefetchCursor &cursor, uint64_t skipInRegion)
{
    if (cursor.regionIndex >= cursor.regionCount) {
        return;
    }
    uint32_t expertIdx = cursor.regionIndex;
    uint64_t regionBytes = cursor.bytesPerExpert;
    __gm__ uint8_t *regionBase =
        GetTensorAddr<uint8_t>(cursor.isPerExpertWeightTensor ? expertIdx : 0U, cursor.weightList);
    if (!cursor.isPerExpertWeightTensor) {
        // listLen==1 堆叠模式：与 kernel 的 gmGroupOffsetB 元素步进同源（元素数 x dtype 字节）。
        regionBase += static_cast<uint64_t>(expertIdx) * regionBytes;
    }
    cursor.regionEnd = regionBase + regionBytes;
    // 区域内 chunk 交错认领：worker c 从偏移 (skip + c*CHUNK) mod regionBytes 起步；回绕重叠由
    // L2 去重，无害。
    uint64_t startOffset =
        regionBytes == 0U ? 0U :
                            (skipInRegion + static_cast<uint64_t>(cursor.workerIdx) * cursor.chunkBytes) % regionBytes;
    cursor.chunkAddr = regionBase + startOffset;
}

// 本 worker 在当前专家的 chunk 走完后推进到下一专家。
__aicore__ inline void WeightL2PrefetchAdvanceRegion(WeightL2PrefetchCursor &cursor)
{
    ++cursor.regionIndex;
    WeightL2PrefetchSetupRegion(cursor, 0U);
}

/*
 * 把游标重定位到专家区间 [firstExpert, lastExpert)：区间从首专家头部起交错认领。每个发射
 * 窗口（窗口 0 的每个 dispatch 组 / 窗口 1 的每个会话）调用一次，使预取目标与流水线消费
 * 位置（groupIdx / swiglu 分段）同步，绝不指向已消费区域。
 */
__aicore__ inline void WeightL2PrefetchRetarget(WeightL2PrefetchCursor &cursor, uint32_t firstExpert,
                                                uint32_t lastExpert)
{
    if (!cursor.active) {
        return;
    }
    if (firstExpert > cursor.regionCount) {
        firstExpert = cursor.regionCount;
    }
    if (lastExpert > cursor.regionCount) {
        lastExpert = cursor.regionCount;
    }
    if (lastExpert < firstExpert) {
        lastExpert = firstExpert;
    }
    cursor.regionIndex = firstExpert;
    cursor.regionLimit = lastExpert;
    WeightL2PrefetchSetupRegion(cursor, 0U);
}

/*
 * 初始化预取游标（单一权重矩阵）。budgetBytes 为全部 worker 共享的全局预算，内部按
 * workerCount 均摊（不足一个 chunk 时保底 1 个）。目标专家区间由后续 Retarget 指定。
 * chunkBytes 为本游标的 chunk 尺寸（交错步长与单次拷贝上限；AIV 路径传运行时洞宽，不足
 * AIV_MIN_CHUNK_BYTES 时游标 inactive，路径整体 no-op）。
 */
__aicore__ inline void WeightL2PrefetchInit(WeightL2PrefetchCursor &cursor, GM_ADDR weightList,
                                            bool isPerExpertWeightTensor, uint64_t bytesPerExpert, uint32_t expertCount,
                                            uint32_t workerIdx, uint32_t workerCount, uint64_t budgetBytes,
                                            AscendC::LocalTensor<uint8_t> scratchTensor,
                                            uint32_t chunkBytes = WEIGHT_L2_PREFETCH_CHUNK_BYTES)
{
    cursor.weightList = weightList;
    cursor.isPerExpertWeightTensor = isPerExpertWeightTensor;
    cursor.expertCount = expertCount;
    cursor.regionCount = expertCount;
    cursor.workerIdx = workerIdx;
    cursor.workerCount = workerCount > 0U ? workerCount : 1U;
    cursor.bytesPerExpert = bytesPerExpert;
    cursor.chunkBytes = chunkBytes & ~31U; // 32B 对齐向下取整（DataCopyPad 块粒度）
    cursor.scratch = scratchTensor;
    cursor.scratch.SetSize(cursor.chunkBytes);
    cursor.active =
        (workerCount > 0U && expertCount > 0U && cursor.chunkBytes >= WEIGHT_L2_PREFETCH_AIV_MIN_CHUNK_BYTES);
    cursor.budgetRemaining = budgetBytes / cursor.workerCount;
    if (budgetBytes != 0U && cursor.budgetRemaining == 0U) {
        cursor.budgetRemaining = cursor.chunkBytes;
    }
    cursor.regionIndex = 0U;
    cursor.regionLimit = 0U;
    WeightL2PrefetchSetupRegion(cursor, 0U);
}

/*
 * 发射一个 chunk 的 GM->UB fire-and-forget 拷贝（不等待完成）；返回 false 表示游标耗尽（预算
 * 用完或目标区间走完）。单缓冲 scratch 即可：同核 MTE2 队列按程序序执行保证 UB 写-写按序，
 * 且 scratch 数据从不被消费。
 */
__aicore__ inline bool WeightL2PrefetchTryIssue(WeightL2PrefetchCursor &cursor)
{
    while (cursor.active && cursor.regionIndex < cursor.regionLimit && cursor.budgetRemaining != 0U) {
        if (cursor.chunkAddr >= cursor.regionEnd) {
            WeightL2PrefetchAdvanceRegion(cursor);
            continue;
        }
        uint64_t bytesToRegionEnd = static_cast<uint64_t>(cursor.regionEnd - cursor.chunkAddr);
        // 拷贝量同时被区域尾与剩余预算钳住，防止预算均摊后不足一个 chunk 时 uint64 回绕。
        uint64_t copyLimit = bytesToRegionEnd < cursor.budgetRemaining ? bytesToRegionEnd : cursor.budgetRemaining;
        uint32_t copyBytes = copyLimit < cursor.chunkBytes ? static_cast<uint32_t>(copyLimit) : cursor.chunkBytes;
        AscendC::GlobalTensor<uint8_t> weightGm;
        weightGm.SetGlobalBuffer(cursor.chunkAddr);
        AscendC::DataCopyPad(cursor.scratch, weightGm, {1, static_cast<uint16_t>(copyBytes), 0, 0}, {});
        cursor.budgetRemaining -= copyBytes;
        cursor.issuedBytes += copyBytes;
        // 下一 chunk 跳过其它 worker 认领的 chunk（步长 = workerCount 个 chunk）。
        cursor.chunkAddr += static_cast<uint64_t>(cursor.workerCount) * cursor.chunkBytes;
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

// 全量发射：循环到 TryIssue 耗尽（预算用完或目标区间走完）——窗口 0 的主发射（AllGather
// push 与自旋两循环之间一次性发完）；窗口 1 休眠路径的预算交付同用本函数语义。
__aicore__ inline void WeightL2PrefetchBurstAll(WeightL2PrefetchCursor &cursor)
{
    while (WeightL2PrefetchTryIssue(cursor)) {
    }
}

// 排空在飞的预取拷贝：SetFlag 进入 MTE2 队列、按序在全部已发射拷贝完成后置位事件。仅窗口 1
// 会话尾调用收口（休眠路径）；窗口 0 的收口由发射函数（AllGather）内的全流水屏障
// PipeBarrier<PIPE_ALL>/SyncAll<true> 天然完成，无需独立 Drain。
__aicore__ inline void WeightL2PrefetchDrain(uint32_t eventId)
{
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
}

/*
 * AIC 侧区间权重哑预取：GM->L1 环写，数据路过 L2 不消费。[firstExpert, lastExpert) 逐专家，
 * 每专家限流 = min(单专家全覆盖所需, capChunks x coreNum x CHUNK)，(coreIdx, coreNum) 静态
 * mod 交错认领（零通信零重叠，与各核发射时序无关）。W1 冷启动与 GMM2 段界 W2 预取共用。
 * 安全性：scratch 由调用方传入，取流水线真实 L1 占用之上（BlockMmad::GetL1EndOffset 对齐
 * 取整）——与 stage/S/F 零地址交集，哑拷贝（GM->L1，MTE2）与 LoadData（L1->L0，MTE1）跨
 * 队列亦无碰撞，冷启动与流水线运行期发射同样安全，无需排空/同步；GM 读范围被专家字节与配额
 * 双钳（尾块按剩余截断，32B 对齐由 host 的 K/N 512 对齐保证）。
 */
__aicore__ inline void WeightL2PrefetchAicRange(GM_ADDR weightList, bool isPerExpertWeightTensor,
                                                uint64_t bytesPerExpert, uint32_t firstExpert, uint32_t lastExpert,
                                                uint32_t capChunks, uint32_t coreIdx, uint32_t coreNum,
                                                AscendC::LocalTensor<uint8_t> l1Scratch)
{
    if (capChunks == 0U || coreNum == 0U || bytesPerExpert == 0U || firstExpert >= lastExpert) {
        return;
    }
    uint64_t limit = static_cast<uint64_t>(capChunks) * coreNum * WEIGHT_L2_PREFETCH_CHUNK_BYTES;
    limit = limit < bytesPerExpert ? limit : bytesPerExpert;
    l1Scratch.SetSize(WEIGHT_L2_PREFETCH_CHUNK_BYTES);
    AscendC::GlobalTensor<uint8_t> weightGm;
    for (uint32_t expertIdx = firstExpert; expertIdx < lastExpert; ++expertIdx) {
        // per-expert 模式取列表项；stacked 模式基址即堆叠起点，按专家字节步进（与 gmGroupOffsetB 同源）。
        __gm__ uint8_t *regionBase = GetTensorAddr<uint8_t>(isPerExpertWeightTensor ? expertIdx : 0U, weightList);
        if (!isPerExpertWeightTensor) {
            regionBase += static_cast<uint64_t>(expertIdx) * bytesPerExpert;
        }
        // (coreIdx, coreNum) 交错认领 16KB chunk，环写同一块保留区。
        for (uint64_t offset = static_cast<uint64_t>(coreIdx) * WEIGHT_L2_PREFETCH_CHUNK_BYTES; offset < limit;
             offset += static_cast<uint64_t>(coreNum) * WEIGHT_L2_PREFETCH_CHUNK_BYTES) {
            uint64_t remain = limit - offset;
            uint32_t chunkBytes = remain < WEIGHT_L2_PREFETCH_CHUNK_BYTES ? static_cast<uint32_t>(remain) :
                                                                            WEIGHT_L2_PREFETCH_CHUNK_BYTES;
            weightGm.SetGlobalBuffer(regionBase + offset);
            AscendC::DataCopy(l1Scratch, weightGm, chunkBytes);
        }
    }
}

/*
 * W1 冷启动（AIC 侧 GM->L1 哑拷贝路径，休眠）：在 GMM1 入口、首个轮就绪 wait 之前由调用方
 * 发射——哑拷贝在 prologue/AllGather/cumsum 等待 + 组 0 pull 全程排空（GM->L1 与 pull 的
 * GM->GM 分属不同核的 MTE 引擎，靠错峰而非引擎隔离）。覆盖前 AIC_COLDSTART_EXPERTS 个专家
 * （GMM1 消费前沿头部）；W1 地址不依赖 cumsum（权重静态）；空专家此时不可判，接受死预取。
 * 当前 W1 预取主路径为窗口 0（AIV 固定区），本路径独立开关休眠保留。
 */
__aicore__ inline void WeightL2PrefetchAicColdStartW1(GM_ADDR weightList, bool isPerExpertWeightTensor,
                                                      uint64_t bytesPerExpert, uint32_t expertPerRank, uint32_t coreIdx,
                                                      uint32_t coreNum, AscendC::LocalTensor<uint8_t> l1Scratch)
{
    if (!WEIGHT_L2_PREFETCH_ENABLE_W1_AIC_COLDSTART) {
        return;
    }
    uint32_t expertCount = expertPerRank < WEIGHT_L2_PREFETCH_W0_AIC_COLDSTART_EXPERTS ?
                               expertPerRank :
                               WEIGHT_L2_PREFETCH_W0_AIC_COLDSTART_EXPERTS;
    WeightL2PrefetchAicRange(weightList, isPerExpertWeightTensor, bytesPerExpert, 0U, expertCount,
                             WEIGHT_L2_PREFETCH_W0_AIC_COLDSTART_CHUNKS, coreIdx, coreNum, l1Scratch);
}

#endif // MEGA_MOE_WEIGHT_L2_PREFETCH_HPP
