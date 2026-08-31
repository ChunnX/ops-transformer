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
 *   MEGA_MOE_WEIGHT_PREFETCH_MODE            总开关：1=W1 预取开（当前架构默认）；2/3 另会打开
 *                                           遗留窗口 1（AIV 侧 W2 会话预取，实测双机负收益，
 *                                           已被 GMM2 段界预取取代，勿与 W2_GMM2 同时开启）
 *   MEGA_MOE_PREFETCH_W0_AIC_COLDSTART_EXPERTS AIC 冷启动预取的专家数上限（0=关闭 W1 预取）
 *   MEGA_MOE_PREFETCH_W0_AIC_COLDSTART_CHUNKS 每专家每 AIC 核的 chunk 上限（实际配额按 shape
 *                                           自适应=单专家全覆盖所需，见 CoverChunks）
 *   MEGA_MOE_PREFETCH_W2_GMM2_CHUNKS       GMM2 段界 W2 预取：每专家每 AIC 核 chunk 上限，
 *                                           每段 V2C wait 前发射（0=关闭；试点 8，实测后定）。
 *                                           独立于 MODE——单因子消融直接用本宏与 EXPERTS
 *   MEGA_MOE_PREFETCH_W0_AIV_ENABLE          AIV 侧 dispatch 期 W1 预取开关（默认 0=关闭——
 *                                           与 pull 争共享 MTE outstanding，双机实测严重劣化；
 *                                           机制保留待后续策略复用）
 *   MEGA_MOE_PREFETCH_W0_CHUNKS_PER_IDLE_AIV AIV 开启时：每空闲核每组 chunk 上限（配额自适应）
 *   MEGA_MOE_PREFETCH_W0_BUSY_CHUNKS_PER_AIV AIV 开启时：全忙场景每核组尾 chunk 上限（同上）
 *   MEGA_MOE_PREFETCH_SESSION_CHUNKS         窗口 1 每会话 burst 上限（钳制 Drain 过冲）
 *   MEGA_MOE_PREFETCH_MAX_BUDGET_MB          窗口 1 全局预算（MB）
 *   MEGA_MOE_PREFETCH_GMM1_HINT_MODE         GMM1(W1) L2 hint：-1=auto(默认，随W1预取使能)
 *                                           0=NORMAL / 1=条件 DISABLE
 *   MEGA_MOE_PREFETCH_GMM2_HINT_MODE         GMM2(W2) L2 hint：-1=auto(默认，随任一 W2 预取
 *                                           路径使能) 0=NORMAL / 1=条件 DISABLE
 * hint 语义：NORMAL 只服务被预取的权重——auto 下未被预取的一侧保持基线条件 DISABLE
 * （currentM<=单 M-tile 即权重单次读时 bypass，与 d0086d1c5 对齐）。单因子消融用独立宏：
 * EXPERTS=0/W2_GMM2=0；MODE=0 关 W1（W2_GMM2 仍独立生效）。
 */
#ifndef MEGA_MOE_WEIGHT_PREFETCH_MODE
#define MEGA_MOE_WEIGHT_PREFETCH_MODE 1
#endif
#ifndef MEGA_MOE_PREFETCH_W0_AIC_COLDSTART_EXPERTS
#define MEGA_MOE_PREFETCH_W0_AIC_COLDSTART_EXPERTS 4
#endif
#ifndef MEGA_MOE_PREFETCH_W0_AIC_COLDSTART_CHUNKS
#define MEGA_MOE_PREFETCH_W0_AIC_COLDSTART_CHUNKS 64
#endif
#ifndef MEGA_MOE_PREFETCH_W2_GMM2_CHUNKS
#define MEGA_MOE_PREFETCH_W2_GMM2_CHUNKS 8
#endif
#ifndef MEGA_MOE_PREFETCH_W0_AIV_ENABLE
#define MEGA_MOE_PREFETCH_W0_AIV_ENABLE 0
#endif
#ifndef MEGA_MOE_PREFETCH_W0_CHUNKS_PER_IDLE_AIV
#define MEGA_MOE_PREFETCH_W0_CHUNKS_PER_IDLE_AIV 128
#endif
#ifndef MEGA_MOE_PREFETCH_W0_BUSY_CHUNKS_PER_AIV
#define MEGA_MOE_PREFETCH_W0_BUSY_CHUNKS_PER_AIV 128
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
static_assert(MEGA_MOE_PREFETCH_W0_AIC_COLDSTART_CHUNKS >= 0 && MEGA_MOE_PREFETCH_W0_AIC_COLDSTART_CHUNKS <= 128,
              "MEGA_MOE_PREFETCH_W0_AIC_COLDSTART_CHUNKS must be 0(disabled)..128 chunks per AIC");
static_assert(MEGA_MOE_PREFETCH_W0_AIC_COLDSTART_EXPERTS >= 0 && MEGA_MOE_PREFETCH_W0_AIC_COLDSTART_EXPERTS <= 16,
              "MEGA_MOE_PREFETCH_W0_AIC_COLDSTART_EXPERTS must be 0(disabled)..16 experts");
static_assert(MEGA_MOE_PREFETCH_W2_GMM2_CHUNKS >= 0 && MEGA_MOE_PREFETCH_W2_GMM2_CHUNKS <= 128,
              "MEGA_MOE_PREFETCH_W2_GMM2_CHUNKS must be 0(disabled)..128 chunks per AIC");
static_assert(MEGA_MOE_PREFETCH_W0_AIV_ENABLE == 0 || MEGA_MOE_PREFETCH_W0_AIV_ENABLE == 1,
              "MEGA_MOE_PREFETCH_W0_AIV_ENABLE must be 0(disabled) or 1(enabled)");
static_assert(MEGA_MOE_PREFETCH_W0_BUSY_CHUNKS_PER_AIV >= 0 && MEGA_MOE_PREFETCH_W0_BUSY_CHUNKS_PER_AIV <= 128,
              "MEGA_MOE_PREFETCH_W0_BUSY_CHUNKS_PER_AIV must be 0(disabled)..128 chunks per busy AIV per group");
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
#elif MEGA_MOE_WEIGHT_PREFETCH_MODE == 2 || MEGA_MOE_WEIGHT_PREFETCH_MODE == 3 || MEGA_MOE_PREFETCH_W2_GMM2_CHUNKS > 0
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
#if MEGA_MOE_PREFETCH_W2_GMM2_CHUNKS > 0
static_assert(MEGA_MOE_PREFETCH_GMM2_HINT_DISABLE == 0,
              "W2 GMM2 section prefetch requires GMM2 L2 hint NORMAL; CACHE_MODE_DISABLE voids the prefetch");
#endif
// 防呆：遗留窗口 1（MODE 2/3）与 GMM2 段界 W2 预取互斥——同时开启会对同一批 W2 重复预取。
#if (MEGA_MOE_WEIGHT_PREFETCH_MODE == 2 || MEGA_MOE_WEIGHT_PREFETCH_MODE == 3) && MEGA_MOE_PREFETCH_W2_GMM2_CHUNKS > 0
static_assert(false, "window-1 (MODE 2/3) and GMM2-section W2 prefetch (W2_GMM2_CHUNKS>0) are mutually exclusive: "
                     "duplicate W2 traffic; use W2_GMM2 alone");
#endif

// 窗口使能开关（编译期常量，调用侧据此跳过 Init/Drain；游标未构造时 active=false 发射自动 no-op）。
constexpr bool WEIGHT_L2_PREFETCH_W0_ENABLED =
    (MEGA_MOE_WEIGHT_PREFETCH_MODE == 1) || (MEGA_MOE_WEIGHT_PREFETCH_MODE == 3);
constexpr bool WEIGHT_L2_PREFETCH_W1_ENABLED =
    (MEGA_MOE_WEIGHT_PREFETCH_MODE == 2) || (MEGA_MOE_WEIGHT_PREFETCH_MODE == 3);
// AIC 侧 W1 冷启动总开关（MODE 含 W0 且专家数/配额非零）。
constexpr bool WEIGHT_L2_PREFETCH_W0_AIC_ENABLED = WEIGHT_L2_PREFETCH_W0_ENABLED &&
                                                   MEGA_MOE_PREFETCH_W0_AIC_COLDSTART_EXPERTS > 0 &&
                                                   MEGA_MOE_PREFETCH_W0_AIC_COLDSTART_CHUNKS > 0;
// GMM2 段界 W2 预取开关：GMM2 组循环每段 V2C wait 前由各 AIC 用 MTE1 哑拷贝预取即将消费段的
// W2——排空与 AIV 本段 swiglu 的 gmC 读并行（本地大块读，outstanding 敏感性待实测，剂量上限
// 即为此设）；wait 返回即消费（零距离，无驱逐窗口）。首段无等待窗口不预取（保持冷）；晚到核
// 窗口≈0，其小 burst 尾部由剂量上限兜底。
constexpr bool WEIGHT_L2_PREFETCH_W2_GMM2_ENABLED = MEGA_MOE_PREFETCH_W2_GMM2_CHUNKS > 0;
// AIV MTE2 预取路径总开关（遗留窗口 1 会话 + AIV dispatch 期 W1）：仅这些路径会在 AIV 的 MTE2
// 留下在飞块、需要会话 Drain 收口；全关（当前默认）时 Drain 整体跳过，会话恢复纯等待语义
// （AIC 侧 W1/W2 预取不经 AIV MTE2，无需此收口）。
constexpr bool WEIGHT_L2_PREFETCH_AIV_MTE2_ANY =
    WEIGHT_L2_PREFETCH_W1_ENABLED || (WEIGHT_L2_PREFETCH_W0_ENABLED && WEIGHT_L2_PREFETCH_W0_AIV_ENABLED);

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
 * AIC 预取专用 L1 保留区 [0, CHUNK)：任一 AIC 预取路径使能时，GMM1/GMM2 的 BlockMmad 构造以
 * l1BufAddrStart 整体上移让出，哑拷贝环写专用——GMM1 冷启动与 GMM2 段界预取共用。不预留则
 * GMM2 段界预取会踩流水线在用的活 L1 stage（冷启动无此问题仅因发射时流水线尚未载入任何
 * tile）。L1 总量 512KB、流水线 stages 仅 ~百 KB 量级，16KB 余量充足；全部预取关闭时保留 0，
 * 流水线布局与基线逐位一致。
 */
constexpr uint32_t WEIGHT_L2_PREFETCH_L1_RESERVED_BYTES =
    (WEIGHT_L2_PREFETCH_W0_AIC_ENABLED || WEIGHT_L2_PREFETCH_W2_GMM2_ENABLED) ? WEIGHT_L2_PREFETCH_CHUNK_BYTES : 0U;
/*
 * 预取 Drain 的 MTE2_V 事件 ID：窗口 0 不使用事件（无独立 Drain，在飞块由窗口 1 首会话统一
 * 收口）；窗口 1 会话用 ID6——swiglu/combine 期功能侧仅占 0-3，ID6 空闲（dispatch 期 W4A8 的
 * 6/7 在进入 swiglu 前已排空）。功能侧全部为 Set->Wait 背靠背配对，Drain 闭环后 ID 即释放。
 */
constexpr uint32_t WEIGHT_L2_PREFETCH_EVENT_ID_WINDOW1 = 6U;
// 窗口 1（W2）会话的全局预算（防过量预热挤掉 L2 内有复用价值的数据）；游标内部再夹紧到权重总量。
constexpr uint32_t WEIGHT_L2_PREFETCH_MAX_BUDGET_BYTES = MEGA_MOE_PREFETCH_MAX_BUDGET_MB * 1024U * 1024U;
// AIC 冷启动预取的专家数上限：覆盖 GMM1 最先消费的前 N 个专家（消费前沿头部），其后专家回到
// 冷 miss（实测 AIV 在 pull 期预取与远端读争带宽、双机场景劣化，W1 预取全部收敛到 AIC 侧）。
constexpr uint32_t WEIGHT_L2_PREFETCH_W0_AIC_COLDSTART_EXPERTS = MEGA_MOE_PREFETCH_W0_AIC_COLDSTART_EXPERTS;
// AIC 冷启动每专家每核的 chunk 上限：实际配额 = min(单专家全覆盖所需, 本上限)（CoverChunks 按
// HiddenSize/N/dtype 自适应）。发射点在 GMM1 首个 flag wait 之前——排空窗口 = routing+cumsum+
// AllGather 等待 + 组 0 pull 全程，与 pull 的 MTE2 零争抢（AIC MTE1 队列此时完全空闲）。实测
// （发射点在 wait 之后时）全覆盖 128 负收益：排空尾部顶住 GMM1 首个真实 B-tile 载入 + 与 pull
// 争带宽；前移后窗口扩大，可扫参上调。扫参 -D 调 0..128，0 等效 EXPERTS=0 关闭。
constexpr uint32_t WEIGHT_L2_PREFETCH_W0_AIC_COLDSTART_CHUNKS = MEGA_MOE_PREFETCH_W0_AIC_COLDSTART_CHUNKS;
// AIV 侧 dispatch 期 W1 预取开关，默认关闭（实测与 pull 争带宽、双机场景劣化，W1 预取收敛到
// AIC 冷启动）。机制保留（空闲核组顶/忙核组尾发射点与游标不删），供后续策略复用——如在该
// 阶段预取 AIC 未覆盖的后续专家（EXPERTS 之后）。开启用 -DMEGA_MOE_PREFETCH_W0_AIV_ENABLE=1。
constexpr bool WEIGHT_L2_PREFETCH_W0_AIV_ENABLED = (MEGA_MOE_PREFETCH_W0_AIV_ENABLE == 1);
// AIV 路径（开启时）每空闲核每组 chunk 上限：实际配额 = 单专家全覆盖所需（CoverChunks 按
// HiddenSize/N/dtype 自适应）。注意 EP 大/空闲核少时全覆盖剂量大，与 pull 争带宽易劣化。
constexpr uint32_t WEIGHT_L2_PREFETCH_WINDOW0_CHUNKS_PER_IDLE_AIV = MEGA_MOE_PREFETCH_W0_CHUNKS_PER_IDLE_AIV;
// AIV 路径（开启时）全忙场景每核组尾 chunk 上限：实际配额 = 单专家全覆盖所需（CoverChunks
// 自适应）；上限用于限制对下一组功能拷贝的 MTE2 程序序推迟。
constexpr uint32_t WEIGHT_L2_PREFETCH_WINDOW0_BUSY_CHUNKS_PER_AIV = MEGA_MOE_PREFETCH_W0_BUSY_CHUNKS_PER_AIV;
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
