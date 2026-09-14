/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_HCOMM_DFX_DUMP_H
#define MC2_HCOMM_DFX_DUMP_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <vector>
#include "acl/acl_rt.h"
#include "mc2_log.h"
#include "adv_api/hcomm/dfx/hcomm_channel_dump.h"

namespace Mc2Hcomm {

namespace HcommDfxDumpDetail {

// HcommChannelInfoDump 内部会启动 queue-read kernel 并同步流，只能在 AIC 异常处理完成后的
// host 安全点（stream 同步失败返回之后）调用；在异常回调上下文中直接调用会与设备侧异常处理
// 互相等待导致死锁，因此延迟到后台线程执行，进程退出时兜底补一次
// 注：触发异常的 Python 子进程在 AIC error 后约百毫秒量级即退出，延迟必须足够小，
// 否则 detach 的后台线程会在 sleep 期间被进程退出杀死，导致 dump 丢失
constexpr uint32_t HCOMM_DFX_DUMP_DELAY_MS = 20U;

// 全局状态收敛到函数局部 static（C++11 起线程安全初始化），保证多个算子 TU 共享同一份
// once 语义，避免各 TU 私有副本导致的重复 dump
struct HcommDfxState {
    std::mutex mutex;
    std::atomic<bool> dumpStarted{false};
    std::vector<uint64_t> channelHandles;
    int32_t deviceId = -1;
    std::string opName;
    bool atExitRegistered = false;
};

inline HcommDfxState &GetHcommDfxState()
{
    static HcommDfxState state;
    return state;
}

inline int32_t DumpChannelInfoOnce(const std::vector<uint64_t> &channelHandles, const std::string &opName)
{
    if (GetHcommDfxState().dumpStarted.exchange(true)) {
        return 0;
    }
    int32_t dumpRet = 0;
    for (uint64_t channelHandle : channelHandles) {
        dumpRet = AscendC::HcommChannelInfoDump(channelHandle);
        OP_LOGE(opName.c_str(), "HcommChannelInfoDump ret=%d (deferred at host safe point, channelHandle=0x%lx)",
                dumpRet, channelHandle);
    }
    return dumpRet;
}

inline void DumpAtExit()
{
    std::vector<uint64_t> handles;
    int32_t deviceId = -1;
    std::string opName;
    {
        std::lock_guard<std::mutex> lock(GetHcommDfxState().mutex);
        handles = GetHcommDfxState().channelHandles;
        deviceId = GetHcommDfxState().deviceId;
        opName = GetHcommDfxState().opName;
        GetHcommDfxState().channelHandles.clear();
        GetHcommDfxState().deviceId = -1;
    }
    if (handles.empty()) {
        return;
    }
    if (deviceId >= 0) {
        (void)aclrtSetDevice(deviceId);
    }
    (void)DumpChannelInfoOnce(handles, opName);
}

inline void DumpWorker(std::vector<uint64_t> channelHandles, int32_t deviceId, const std::string &opName)
{
    if (deviceId >= 0) {
        (void)aclrtSetDevice(deviceId);
    }
    // 等待当前 AIC 异常处理流程结束、设备调度恢复，避免与异常处理线程竞争运行时锁
    std::this_thread::sleep_for(std::chrono::milliseconds(HCOMM_DFX_DUMP_DELAY_MS));
    (void)DumpChannelInfoOnce(channelHandles, opName);
}
} // namespace HcommDfxDumpDetail

// 在 host 安全点异步触发 hcomm channel dump。
inline void TriggerDeferredDump(const std::vector<uint64_t> &channelHandles, int32_t deviceId, const char *opName)
{
    if (channelHandles.empty()) {
        return;
    }
    OP_LOGE(opName, "Hcomm dump trigger deferred to host safe point, channelHandleNum=%zu, firstChannelHandle=0x%lx",
            channelHandles.size(), channelHandles.front());
    std::string opStr = (opName == nullptr) ? std::string() : std::string(opName);
    {
        std::lock_guard<std::mutex> lock(HcommDfxDumpDetail::GetHcommDfxState().mutex);
        HcommDfxDumpDetail::GetHcommDfxState().channelHandles = channelHandles;
        HcommDfxDumpDetail::GetHcommDfxState().deviceId = deviceId;
        HcommDfxDumpDetail::GetHcommDfxState().opName = opStr;
        if (!HcommDfxDumpDetail::GetHcommDfxState().atExitRegistered) {
            HcommDfxDumpDetail::GetHcommDfxState().atExitRegistered = true;
            (void)std::atexit(HcommDfxDumpDetail::DumpAtExit);
        }
    }
    try {
        std::thread(HcommDfxDumpDetail::DumpWorker, channelHandles, deviceId, opStr).detach();
    } catch (const std::system_error &e) {
        OP_LOGE(opName, "spawn deferred hcomm dump thread failed(%s), dump at process exit", e.what());
    }
}
} // namespace Mc2Hcomm

#endif // MC2_HCOMM_DFX_DUMP_H
