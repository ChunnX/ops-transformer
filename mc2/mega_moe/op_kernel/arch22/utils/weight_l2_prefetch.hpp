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
 * \file weight_l2_prefetch.hpp
 * \brief A3(910_93) 权重 L2 预取：AIV 用 fire-and-forget 的 GM->UB DataCopyPad 把专家权重
 *        字节拉进 L2，供 AIC 的 copyGmToL1B(GM->L1, MTE1) 读命中；UB 数据从不消费。
 *        双窗口按消费进度锁定目标：窗口 0（dispatch 组循环）逐组预热当前专家的 W1（GMM1
 *        消费），窗口 1（等 GMM1 的 SYNCFLAGC2V）按 swiglu 分段预热即将进入 GMM2 的专家
 *        的 W2——预取目标永远在消费前沿之前，任何 shape 下零死字节。机制对标 arch35 的
 *        同类 L2 权重预取设计（该实现未随本仓库存档），按 A3 的 192KB UB 与空闲窗口重规划。
 */

#ifndef MEGA_MOE_WEIGHT_L2_PREFETCH_HPP
#define MEGA_MOE_WEIGHT_L2_PREFETCH_HPP

#include "kernel_operator.h"

#include "get_tensor_addr.hpp"
#include "../template_linear_algebra_v2/arch/mega_moe_arch.hpp"

/*
 * 预取策略开关与参数（默认值=当前落地策略；均不进 tiling，编译期 -D 覆盖做消融/扫参）：
 *   MEGA_MOE_WEIGHT_PREFETCH_MODE            0=OFF / 1=仅窗口0(W1) / 2=仅窗口1(W2) / 3=双窗口
 *   MEGA_MOE_PREFETCH_W0_CHUNKS_PER_IDLE_AIV 窗口 0 每空闲核每组 chunk 安全上限（实际配额按
 *                                           shape 自适应=单专家全覆盖所需，见 CoverChunks）
 *   MEGA_MOE_PREFETCH_W0_BUSY_CHUNKS_PER_AIV 无空闲核（EP>=coreNum）时窗口 0 忙核兜底配额
 *                                           （组尾 SyncAll 前发射，0=关闭兜底）
 *   MEGA_MOE_PREFETCH_W0_AIC_COLDSTART_CHUNKS 核全忙时 AIC 冷启动配额：GMM1 等组 0 flag 的
 *                                           空闲窗口用 MTE1 哑拷贝预热专家 0 的 W1（0=关闭）
 *   MEGA_MOE_PREFETCH_SESSION_CHUNKS         窗口 1 每会话 burst 上限（钳制 Drain 过冲）
 *   MEGA_MOE_PREFETCH_MAX_BUDGET_MB          窗口 1 全局预算（MB）
 *   MEGA_MOE_PREFETCH_GMM1_HINT_MODE         GMM1(W1) L2 hint：-1=auto(默认，随窗口0使能)
 *                                           0=NORMAL / 1=条件 DISABLE
 *   MEGA_MOE_PREFETCH_GMM2_HINT_MODE         GMM2(W2) L2 hint：-1=auto(默认，随窗口1使能)
 *                                           0=NORMAL / 1=条件 DISABLE
 * hint 语义：NORMAL 只服务被预取的权重——auto 下未被预取的一侧保持基线条件 DISABLE
 * （currentM<=单 M-tile 即权重单次读时 bypass，与 d0086d1c5 对齐），故单个 MODE 宏即得
 * 语义干净的单因子实验组：MODE=0=基线等价；MODE=1=W1 预取+GMM2 基线；MODE=3=现状。
 */
/*
 * 实验期默认值=1（X1 首发：仅窗口 0/W1 预取，GMM2 不预取且基线条件 DISABLE）；
 * 正式合入前须改回 3（=双窗口现状），或维持 3 并在构建侧 -D 注入实验配置。
 */
#ifndef MEGA_MOE_WEIGHT_PREFETCH_MODE
#define MEGA_MOE_WEIGHT_PREFETCH_MODE 1
#endif
#ifndef MEGA_MOE_PREFETCH_W0_CHUNKS_PER_IDLE_AIV
#define MEGA_MOE_PREFETCH_W0_CHUNKS_PER_IDLE_AIV 128
#endif
#ifndef MEGA_MOE_PREFETCH_W0_BUSY_CHUNKS_PER_AIV
#define MEGA_MOE_PREFETCH_W0_BUSY_CHUNKS_PER_AIV 128
#endif
#ifndef MEGA_MOE_PREFETCH_W0_AIC_COLDSTART_CHUNKS
#define MEGA_MOE_PREFETCH_W0_AIC_COLDSTART_CHUNKS 128
#endif
#ifndef MEGA_MOE_PREFETCH_SESSION_CHUNKS
#define MEGA_MOE_PREFETCH_SESSION_CHUNKS 8
#endif
#ifndef MEGA_MOE_PREFETCH_MAX_BUDGET_MB
#define MEGA_MOE_PREFETCH_MAX_BUDGET_MB 16
#endif
#ifndef MEGA_MOE_PREFETCH_GMM1_HINT_MODE
#define MEGA_MOE_PREFETCH_GMM1_HINT_MODE (-1)
#endif
#ifndef MEGA_MOE_PREFETCH_GMM2_HINT_MODE
#define MEGA_MOE_PREFETCH_GMM2_HINT_MODE (-1)
#endif

static_assert(MEGA_MOE_WEIGHT_PREFETCH_MODE >= 0 && MEGA_MOE_WEIGHT_PREFETCH_MODE <= 3,
              "MEGA_MOE_WEIGHT_PREFETCH_MODE must be 0(OFF)/1(W0_ONLY)/2(W1_ONLY)/3(W0W1)");
static_assert(MEGA_MOE_PREFETCH_GMM1_HINT_MODE >= -1 && MEGA_MOE_PREFETCH_GMM1_HINT_MODE <= 1,
              "MEGA_MOE_PREFETCH_GMM1_HINT_MODE must be -1(auto)/0(NORMAL)/1(conditional DISABLE)");
static_assert(MEGA_MOE_PREFETCH_GMM2_HINT_MODE >= -1 && MEGA_MOE_PREFETCH_GMM2_HINT_MODE <= 1,
              "MEGA_MOE_PREFETCH_GMM2_HINT_MODE must be -1(auto)/0(NORMAL)/1(conditional DISABLE)");
static_assert(MEGA_MOE_PREFETCH_W0_BUSY_CHUNKS_PER_AIV >= 0 && MEGA_MOE_PREFETCH_W0_BUSY_CHUNKS_PER_AIV <= 128,
              "MEGA_MOE_PREFETCH_W0_BUSY_CHUNKS_PER_AIV must be 0(disabled)..128 chunks per busy AIV per group");
static_assert(MEGA_MOE_PREFETCH_W0_AIC_COLDSTART_CHUNKS >= 0 && MEGA_MOE_PREFETCH_W0_AIC_COLDSTART_CHUNKS <= 128,
              "MEGA_MOE_PREFETCH_W0_AIC_COLDSTART_CHUNKS must be 0(disabled)..128 chunks per AIC");
static_assert(MEGA_MOE_PREFETCH_W0_CHUNKS_PER_IDLE_AIV >= 0 && MEGA_MOE_PREFETCH_W0_CHUNKS_PER_IDLE_AIV <= 256,
              "MEGA_MOE_PREFETCH_W0_CHUNKS_PER_IDLE_AIV must be 0(disabled)..256 chunks per idle AIV");

/*
 * GMM L2 hint 最终判定（供 kernel 侧 #if 使用，须为预处理宏）：
 *   显式值优先；auto=随窗口使能——GMM1 随窗口 0、GMM2 随窗口 1。
 *   =1 表示恢复 d0086d1c5 的条件 DISABLE 分支；=0 表示全 NORMAL（预取生效前提）。
 */
#if MEGA_MOE_PREFETCH_GMM1_HINT_MODE == 0
#define MEGA_MOE_PREFETCH_GMM1_HINT_DISABLE 0
#elif MEGA_MOE_PREFETCH_GMM1_HINT_MODE == 1
#define MEGA_MOE_PREFETCH_GMM1_HINT_DISABLE 1
#elif MEGA_MOE_WEIGHT_PREFETCH_MODE == 1 || MEGA_MOE_WEIGHT_PREFETCH_MODE == 3
#define MEGA_MOE_PREFETCH_GMM1_HINT_DISABLE 0
#else
#define MEGA_MOE_PREFETCH_GMM1_HINT_DISABLE 1
#endif

#if MEGA_MOE_PREFETCH_GMM2_HINT_MODE == 0
#define MEGA_MOE_PREFETCH_GMM2_HINT_DISABLE 0
#elif MEGA_MOE_PREFETCH_GMM2_HINT_MODE == 1
#define MEGA_MOE_PREFETCH_GMM2_HINT_DISABLE 1
#elif MEGA_MOE_WEIGHT_PREFETCH_MODE == 2 || MEGA_MOE_WEIGHT_PREFETCH_MODE == 3
#define MEGA_MOE_PREFETCH_GMM2_HINT_DISABLE 0
#else
#define MEGA_MOE_PREFETCH_GMM2_HINT_DISABLE 1
#endif

// 配置一致性守卫：被预取的权重必须配 NORMAL——DISABLE 读写双向 bypass（实测）会使预取完全失效。
#if MEGA_MOE_WEIGHT_PREFETCH_MODE == 1 || MEGA_MOE_WEIGHT_PREFETCH_MODE == 3
static_assert(MEGA_MOE_PREFETCH_GMM1_HINT_DISABLE == 0,
              "W1 prefetch (window 0) requires GMM1 L2 hint NORMAL; CACHE_MODE_DISABLE voids the prefetch");
#endif
#if MEGA_MOE_WEIGHT_PREFETCH_MODE == 2 || MEGA_MOE_WEIGHT_PREFETCH_MODE == 3
static_assert(MEGA_MOE_PREFETCH_GMM2_HINT_DISABLE == 0,
              "W2 prefetch (window 1) requires GMM2 L2 hint NORMAL; CACHE_MODE_DISABLE voids the prefetch");
#endif

// 窗口使能开关（编译期常量，调用侧据此跳过 Init/Drain；游标未构造时 active=false 发射自动 no-op）。
constexpr bool WEIGHT_L2_PREFETCH_W0_ENABLED =
    (MEGA_MOE_WEIGHT_PREFETCH_MODE == 1) || (MEGA_MOE_WEIGHT_PREFETCH_MODE == 3);
constexpr bool WEIGHT_L2_PREFETCH_W1_ENABLED =
    (MEGA_MOE_WEIGHT_PREFETCH_MODE == 2) || (MEGA_MOE_WEIGHT_PREFETCH_MODE == 3);
constexpr bool WEIGHT_L2_PREFETCH_ANY_ENABLED = WEIGHT_L2_PREFETCH_W0_ENABLED || WEIGHT_L2_PREFETCH_W1_ENABLED;

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
 * 预取 Drain 的 MTE2_V 事件 ID：窗口 0 不使用事件（无独立 Drain，在飞块由窗口 1 首会话统一
 * 收口）；窗口 1 会话用 ID6——swiglu/combine 期功能侧仅占 0-3，ID6 空闲（dispatch 期 W4A8 的
 * 6/7 在进入 swiglu 前已排空）。功能侧全部为 Set->Wait 背靠背配对，Drain 闭环后 ID 即释放。
 */
constexpr uint32_t WEIGHT_L2_PREFETCH_EVENT_ID_WINDOW1 = 6U;
// 窗口 1（W2）会话的全局预算（防过量预热挤掉 L2 内有复用价值的数据）；游标内部再夹紧到权重总量。
constexpr uint32_t WEIGHT_L2_PREFETCH_MAX_BUDGET_BYTES = MEGA_MOE_PREFETCH_MAX_BUDGET_MB * 1024U * 1024U;
// 窗口 0 每空闲 AIV 每个专家组的 chunk 安全上限：实际配额 = 单专家全覆盖所需（CoverChunks 按
// HiddenSize/N/dtype 自适应），如 7168x4096 int4 专家 14MiB 在 EP=16/aivNum=48（32 空闲核）时
// 自适应 38 chunk 即全覆盖。按组锁定即时交付，L2 驻留只需跨一个 dispatch 组；A3 跨卡 RTT 较长、
// dispatch 通信窗口宽，空闲核 MTE2 余量充足；上限仅在极端 shape/空闲核极少时钳制。
constexpr uint32_t WEIGHT_L2_PREFETCH_WINDOW0_CHUNKS_PER_IDLE_AIV = MEGA_MOE_PREFETCH_W0_CHUNKS_PER_IDLE_AIV;
// 无空闲核（EP>=coreNum）时窗口 0 的忙核兜底上限：全部 AIV 在组尾 SyncAll 前发射、等待期排空。
// 实际配额 = 单专家全覆盖所需（CoverChunks 自适应，全核分摊后每核量小：14MiB/48 核 ~= 304KB/核，
// 排空大部分被 SyncAll 等待吸收）；实测 pull 变慢明显时可用 -D 调低上限回保守剂量（如 8）。
constexpr uint32_t WEIGHT_L2_PREFETCH_WINDOW0_BUSY_CHUNKS_PER_AIV = MEGA_MOE_PREFETCH_W0_BUSY_CHUNKS_PER_AIV;
// 核全忙（EP>=aivNum）时窗口 0 的 AIC 冷启动上限：GMM1 入口等组 0 flag 的空闲窗口，各 AIC 用
// 完全空闲的 MTE1 对专家 0 的 W1 做 GM->L1 环写哑拷贝预热 L2——不占 AIV 的 MTE2，零 dispatch
// 关键路径成本（EP<aivNum 时空闲 AIV 组顶 burst 已覆盖专家 0，AIC 侧直接返回，无双份）。实际
// 配额 = 单专家全覆盖所需（CoverChunks 自适应），上限仅在极端 shape 时钳制。
constexpr uint32_t WEIGHT_L2_PREFETCH_W0_AIC_COLDSTART_CHUNKS = MEGA_MOE_PREFETCH_W0_AIC_COLDSTART_CHUNKS;
// 窗口 1 每次会话（每次等 SYNCFLAGC2V 前）的 burst 上限：128KB ~= 1-2us，钳制 Drain 过冲。
constexpr uint32_t WEIGHT_L2_PREFETCH_SESSION_CHUNKS = MEGA_MOE_PREFETCH_SESSION_CHUNKS;

static_assert(WEIGHT_L2_PREFETCH_UB_OFFSET % 32U == 0U, "prefetch scratch must be 32B aligned");
// 【已知缺口，暂缓】下式预留 512B 仅覆盖 EP<=128；与 scratch 顶无重叠的精确条件是
// AlignUp(EP*4,32) <= UB_SIZE - (UB_OFFSET + CHUNK) = 672B，即 EP <= 168。EP>=169 时 rankMask
// 常驻区下探侵入 scratch [179552,195936)，窗口 0/1（含忙核兜底）的预取写会踩 mask，combine 期
// CrossRankSync*WithMask 全核读到脏数据 -> 功能错误。当前无运行时守卫（现有部署 EP<=128 不
// 触发）；后续解决：kernel Init 检测到重叠时置预取游标 active=false，或 host tiling 校验 EP 上限。
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
    AscendC::LocalTensor<uint8_t> scratch; // [WEIGHT_L2_PREFETCH_UB_OFFSET, +CHUNK) 的单缓冲
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
        regionBytes == 0U ?
            0U :
            (skipInRegion + static_cast<uint64_t>(cursor.workerIdx) * WEIGHT_L2_PREFETCH_CHUNK_BYTES) % regionBytes;
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
 */
__aicore__ inline void WeightL2PrefetchInit(WeightL2PrefetchCursor &cursor, GM_ADDR weightList,
                                            bool isPerExpertWeightTensor, uint64_t bytesPerExpert, uint32_t expertCount,
                                            uint32_t workerIdx, uint32_t workerCount, uint64_t budgetBytes,
                                            AscendC::LocalTensor<uint8_t> scratchTensor)
{
    cursor.weightList = weightList;
    cursor.isPerExpertWeightTensor = isPerExpertWeightTensor;
    cursor.expertCount = expertCount;
    cursor.regionCount = expertCount;
    cursor.workerIdx = workerIdx;
    cursor.workerCount = workerCount > 0U ? workerCount : 1U;
    cursor.bytesPerExpert = bytesPerExpert;
    cursor.scratch = scratchTensor;
    cursor.scratch.SetSize(WEIGHT_L2_PREFETCH_CHUNK_BYTES);
    cursor.active = (workerCount > 0U && expertCount > 0U);
    cursor.budgetRemaining = budgetBytes / cursor.workerCount;
    if (budgetBytes != 0U && cursor.budgetRemaining == 0U) {
        cursor.budgetRemaining = WEIGHT_L2_PREFETCH_CHUNK_BYTES;
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
        uint32_t copyBytes = copyLimit < WEIGHT_L2_PREFETCH_CHUNK_BYTES ? static_cast<uint32_t>(copyLimit) :
                                                                          WEIGHT_L2_PREFETCH_CHUNK_BYTES;
        AscendC::GlobalTensor<uint8_t> weightGm;
        weightGm.SetGlobalBuffer(cursor.chunkAddr);
        AscendC::DataCopyPad(cursor.scratch, weightGm, {1, static_cast<uint16_t>(copyBytes), 0, 0}, {});
        cursor.budgetRemaining -= copyBytes;
        cursor.issuedBytes += copyBytes;
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

/*
 * 覆盖单个专家区域所需的每 worker chunk 数 = CeilDiv(bytesPerExpert, workerCount x CHUNK)，
 * 被 cap 钳制。cap 语义 = 安全上限而非目标配额：实际配额由 HiddenSize/N/权重 dtype 算出的
 * 单专家字节数自适应得出——空闲核/AIC 冷启动模式给宽裕上限即得单专家全覆盖；忙核模式保持
 * 小上限，以限制对下一组功能拷贝的 MTE2 程序序推迟（小专家时自适应值自然低于上限）。
 */
__aicore__ inline uint32_t WeightL2PrefetchCoverChunks(uint64_t bytesPerExpert, uint32_t workerCount, uint32_t cap)
{
    uint32_t workers = workerCount > 0U ? workerCount : 1U;
    uint64_t stride = static_cast<uint64_t>(workers) * WEIGHT_L2_PREFETCH_CHUNK_BYTES;
    uint64_t need = (bytesPerExpert + stride - 1U) / stride;
    return static_cast<uint32_t>(need > cap ? cap : need);
}

// 排空在飞的预取拷贝：SetFlag 进入 MTE2 队列、按序在全部已发射拷贝完成后置位事件。每个预取
// 会话必须以 Drain 收尾（含上一窗口的残留块），防止与后续功能流水访问交错。
__aicore__ inline void WeightL2PrefetchDrain(uint32_t eventId)
{
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
}

#endif // MEGA_MOE_WEIGHT_L2_PREFETCH_HPP
