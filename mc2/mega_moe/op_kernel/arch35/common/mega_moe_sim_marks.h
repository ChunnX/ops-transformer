/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// MegaMoE simulator progress marks. Stable IDs; append new IDs, never reuse them.
// The including kernel header provides set_cond. No buffers, barriers or printf.
#ifndef MEGA_MOE_SIM_MARKS_H
#define MEGA_MOE_SIM_MARKS_H
#include <stdint.h>

#ifndef MEGAMOE_ENABLE_MARKS
#define MEGAMOE_ENABLE_MARKS 1
#endif

namespace MegaMoeImpl {
enum class SimMark : uint32_t {
    KERNEL = 0x001,             // kernel 入口至退出，所有核
    INPUT_PREPARE = 0x004,      // AIV 输入准备与下标/count 发送
    SHARED_GMM1 = 0x005,        // 共享专家 GMM1/激活
    MOE_STAGES = 0x006,         // MoE Dispatch/GMM1/激活/GMM2/Combine
    EXPORT_COUNTS = 0x007,      // 输出专家 token 数并清零 count
    SHARED_GMM2 = 0x008,        // 共享专家 GMM2
    RESET_WORKSPACE = 0x101,    // 输入阶段清零同步区
    PACK_INPUT = 0x102,         // 预量化输入打包指令下发
    COUNT_PREPARE = 0x104,      // A8W4 count 表准备
    GMM1_SLICE = 0x107,         // MoE GMM1/激活的专家 slice 调用范围
    DISPATCH_FIRST = 0x105,     // AIV1 首 wave 的 Dispatch 指令下发
    DISPATCH_SLICE = 0x106,     // AIV1 后续专家 slice 的 Dispatch 指令下发
    GMM2_WAVE = 0x108,          // A8W4 wave：GMM2/权重转换/Combine
    WAIT_INPUT_READY = 0x201,   // AIC 输入就绪及 MTE2 到 Scalar 依赖
    WAIT_REMOTE_COUNT = 0x208,  // AIV1 等待远端 count 整表轮次匹配
    WAIT_DISPATCH_DATA = 0x209, // GMM1 等待 Dispatch 输入 ready
    SEND_INDEX = 0x301,         // 下标发送指令下发
    SEND_COUNT = 0x302,         // count 发送指令下发
    MOE_ENTRY_SYNC = 0x303,     // MoE 入口依赖指令下发，优化版独立同步入口
};
} // namespace MegaMoeImpl

// BEGIN = 0x01000000 | ID; END = 0x02000000 | ID.
// These are scalar progress marks, not completion fences for asynchronous pipes.
#if MEGAMOE_ENABLE_MARKS
#define MEGA_MOE_MARK_BEGIN(name) \
    do { \
        set_cond(0x01000000U | static_cast<uint32_t>(::MegaMoeImpl::SimMark::name)); \
    } while (0)
#define MEGA_MOE_MARK_END(name) \
    do { \
        set_cond(0x02000000U | static_cast<uint32_t>(::MegaMoeImpl::SimMark::name)); \
    } while (0)
#else
#define MEGA_MOE_MARK_BEGIN(name) ((void)0)
#define MEGA_MOE_MARK_END(name) ((void)0)
#endif
#endif
