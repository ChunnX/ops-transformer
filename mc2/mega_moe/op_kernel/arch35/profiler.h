/**
 * \file profiler.h
 * \brief Per-operator profiler header for MegaMoeLayered (URMA/MIX_AIC_1_2) — 子包路径.
 *
 * Include chain: profiler.h → profiler_core.h → profiler_config.h
 * 事件 ID 与 mega_moe_layered.h Process() 流水阶段一一对应。
 * 初始只保留粗粒度阶段对（KERNEL + 5 个主阶段），确认瓶颈后再按需细化。
 */

#pragma once

#include "profiler_core.h"

// ── Event ID enum ──
// 命名约定：EVT_<STAGE>_BEGIN / EVT_<STAGE>_END，配对关系见 op_host/scripts/profiler/pair_map.json。
// 0xFFFF 保留为 ring overflow sentinel，不可用。
enum ProfEventId : uint32_t {
    EVT_KERNEL_BEGIN = 0x0001,
    EVT_KERNEL_END   = 0x00FF,

    // 本卡数据处理：SendAndQuantBuffInit + SendMaskCal + ResetFlagList + ResetDispatchState + SharedExpertCopyInput
    EVT_PREPARE_BEGIN = 0x0002,
    EVT_PREPARE_END   = 0x0003,

    // 共享专家 GMM1 + SwiGLU（前移执行，仅 sharedExpertNum_ > 0）
    EVT_SHARED_GMM1_BEGIN = 0x0004,
    EVT_SHARED_GMM1_END   = 0x0005,

    // 路由专家主循环：dispatch prefetch(DispatchTokenToRmtServer/SendCntCal/MetaInfoCalAndDispatch)
    // + GMM1 consumer(GroupMatmulWithSwigluQuant) + ExpertTokenNumCopyOut
    EVT_DISPATCH_GMM1_BEGIN = 0x0006,
    EVT_DISPATCH_GMM1_END   = 0x0007,

    // GMM2 + Combine：GroupMatmulWithCombine + 共享专家 GMM2
    EVT_GMM2_COMBINE_BEGIN = 0x0008,
    EVT_GMM2_COMBINE_END   = 0x0009,

    // Unpermute
    EVT_UNPERMUTE_BEGIN = 0x000A,
    EVT_UNPERMUTE_END   = 0x000B,
};
