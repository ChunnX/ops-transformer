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
 * \file allto_all_matmul_v2_tiling.cpp
 * \brief Tiling implementation for AlltoAllMatmulV2
 */

#include "arch35/allto_all_matmul_v2_tiling_base.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <register/op_def_registry.h>
#include <register/op_impl_registry.h>
#include "graph/types.h"
#include "tiling/platform/platform_ascendc.h"
#include "mc2_exception_dump.h"
#include "apace/kernel/fusions/all_to_all_quant_matmul/all_to_all_matmul_tiling_data.h"
 
using namespace ge;

namespace MC2Tiling {

using Ops::Transformer::OpTiling::TilingRegistryArch;

static ge::graphStatus AlltoAllMatmulV2TilingFunc(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "[platform] platformInfo is nullptr");
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
    if (npuArch != NpuArch::DAV_3510) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "[platform] AlltoAllMatmulV2 only supports DAV_3510, current arch is not supported");
        return ge::GRAPH_FAILED;
    }
    return TilingRegistryArch::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingParseForAlltoAllMatmulV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct AlltoAllMatmulV2CompileInfo {};

IMPL_OP_OPTILING(AlltoAllMatmulV2)
    .Tiling(AlltoAllMatmulV2TilingFunc)
    .TilingParse<AlltoAllMatmulV2CompileInfo>(TilingParseForAlltoAllMatmulV2);

#if MC2_DFX_ENABLE
#if MC2_HCOMM_DFX_SHIM
extern "C" int32_t Mc2HcommChannelInfoDumpShim(uint64_t channel);

namespace {
// HcommChannelInfoDump 内部会启动 queue-read kernel 并同步流，只能在 AIC 异常处理完成后的
// host 安全点（stream 同步失败返回之后）调用；在异常回调上下文中直接调用会与设备侧异常处理
// 互相等待导致死锁，因此延迟到后台线程执行，进程退出时兜底补一次
// 注：触发异常的 Python 子进程在 AIC error 后约百毫秒量级即退出，延迟必须足够小，
// 否则 detach 的后台线程会在 sleep 期间被进程退出杀死，导致 dump 丢失
constexpr uint32_t HCOMM_DFX_DUMP_DELAY_MS = 20U;

std::mutex g_hcommDfxMutex;
std::atomic<bool> g_hcommDfxDumpStarted{false};
uint64_t g_hcommDfxChannelHandle = 0U;
aclrtContext g_hcommDfxContext = nullptr;
bool g_hcommDfxAtExitRegistered = false;

// 调试用：直接 write 系统调用绕过 slog/stdio 缓冲，验证退出阶段代码路径是否真正执行
#define HCOMM_DFX_TRACE(tag)                                                                        \
    do {                                                                                            \
        const char msg[] = "[HCOMM_DFX_DBG] " tag "\n";                                             \
        (void)write(STDERR_FILENO, msg, sizeof(msg) - 1U);                                          \
    } while (0)

// host 直读补充：AIC error 后 queue-read kernel 无法执行（device error 507015），shim 内
// WQE payload 不可达；此处在 host 安全点用 aclrtMemcpy 直接补读 channel 元数据
// （ChannelEntity/SqContext/head/tail 均在 0x1200... HBM 空间，host 可读；WQE 位于
// 0x3ffd... AIV JFS 队列空间，host 不可读且 device error 后 kernel 无法执行，勿尝试直读）
void Mc2HcommHostReadSqExperiment(uint64_t channelHandle)
{
    auto d2h = [](void *dst, uint64_t src, uint32_t size) -> bool {
        return aclrtMemcpy(dst, size, reinterpret_cast<void *>(src), size, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
    };

    uint8_t entity[256] = {0};
    if (!d2h(entity, channelHandle, sizeof(entity))) {
        OP_LOGE("AlltoAllMatmulV2", "[HostRead] read ChannelEntity failed, handle=0x%lx", channelHandle);
        return;
    }
    const uint64_t sqCtxAddr = *reinterpret_cast<uint64_t *>(entity + 80);
    const uint32_t sqNum = *reinterpret_cast<uint32_t *>(entity + 40);
    const uint32_t sqHead = *reinterpret_cast<uint32_t *>(entity + 96);
    const uint32_t sqTail = *reinterpret_cast<uint32_t *>(entity + 100);
    OP_LOGE("AlltoAllMatmulV2",
            "[HostRead] ChannelEntity: sqCtxAddr=0x%lx sqNum=%u sqHead=%u sqTail=%u",
            sqCtxAddr, sqNum, sqHead, sqTail);

    if (sqCtxAddr == 0U) {
        return;
    }
    uint8_t sqCtx[128] = {0};
    if (!d2h(sqCtx, sqCtxAddr, sizeof(sqCtx))) {
        OP_LOGE("AlltoAllMatmulV2", "[HostRead] read SqContext failed, addr=0x%lx", sqCtxAddr);
        return;
    }
    const uint32_t type = *reinterpret_cast<uint32_t *>(sqCtx + 0);
    const uint64_t sqVa = *reinterpret_cast<uint64_t *>(sqCtx + 8);
    const uint64_t headAddr = *reinterpret_cast<uint64_t *>(sqCtx + 16);
    const uint64_t tailAddr = *reinterpret_cast<uint64_t *>(sqCtx + 24);
    const uint32_t wqeSize = *reinterpret_cast<uint32_t *>(sqCtx + 44);
    const uint32_t sqDepth = *reinterpret_cast<uint32_t *>(sqCtx + 48);
    OP_LOGE("AlltoAllMatmulV2",
            "[HostRead] SqContext: type=%u sqVa=0x%lx wqeSize=%u depth=%u (WQE payload unreachable after AIC error)",
            type, sqVa, wqeSize, sqDepth);

    uint32_t head = 0U;
    uint32_t tail = 0U;
    if (headAddr != 0U && d2h(&head, headAddr, sizeof(head)) && tailAddr != 0U && d2h(&tail, tailAddr, sizeof(tail))) {
        OP_LOGE("AlltoAllMatmulV2", "[HostRead] sq cursor head=%u tail=%u", head, tail);
    } else {
        OP_LOGE("AlltoAllMatmulV2", "[HostRead] read sq cursor failed, headAddr=0x%lx tailAddr=0x%lx",
                headAddr, tailAddr);
    }
}

int32_t Mc2HcommDumpChannelInfoOnce(uint64_t channelHandle)
{
    HCOMM_DFX_TRACE("enter Mc2HcommDumpChannelInfoOnce");
    OP_LOGE("AlltoAllMatmulV2", "start Mc2HcommDumpChannelInfoOnce");
    if (g_hcommDfxDumpStarted.exchange(true)) {
        return 0;
    }
    OP_LOGE("AlltoAllMatmulV2", "before Mc2HcommChannelInfoDumpShim");
    Mc2HcommHostReadSqExperiment(channelHandle);
    int32_t dumpRet = Mc2HcommChannelInfoDumpShim(channelHandle);
    OP_LOGE("AlltoAllMatmulV2", "HcommChannelInfoDump ret=%d (deferred at host safe point, channelHandle=0x%lx)",
            dumpRet, channelHandle);
    return dumpRet;
}

void Mc2HcommDumpAtExit()
{
    HCOMM_DFX_TRACE("enter Mc2HcommDumpAtExit");
    uint64_t handle = 0U;
    aclrtContext ctx = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_hcommDfxMutex);
        handle = g_hcommDfxChannelHandle;
        ctx = g_hcommDfxContext;
        g_hcommDfxChannelHandle = 0U;
        g_hcommDfxContext = nullptr;
    }
    if (handle == 0U) {
        return;
    }
    if (ctx != nullptr) {
        (void)aclrtSetCurrentContext(ctx);
    }
    (void)Mc2HcommDumpChannelInfoOnce(handle);
}

void Mc2HcommDumpWorker(uint64_t channelHandle, aclrtContext capturedCtx, int32_t deviceId)
{
    if (capturedCtx != nullptr) {
        (void)aclrtSetCurrentContext(capturedCtx);
    } else {
        (void)aclrtSetDevice(deviceId);
    }
    OP_LOGE("AlltoAllMatmulV2", "Mc2HcommDumpWorker Inner");
    // 等待当前 AIC 异常处理流程结束、设备调度恢复，避免与异常处理线程竞争运行时锁
    std::this_thread::sleep_for(std::chrono::milliseconds(HCOMM_DFX_DUMP_DELAY_MS));
    HCOMM_DFX_TRACE("worker after sleep");
    OP_LOGE("AlltoAllMatmulV2", "Mc2HcommDumpWorker 950");
    (void)Mc2HcommDumpChannelInfoOnce(channelHandle);
}

void Mc2HcommTriggerDeferredDump(uint64_t channelHandle, int32_t deviceId)
{
    OP_LOGE("AlltoAllMatmulV2", "Hcomm dump trigger deferred to host safe point, channelHandle=0x%lx",
            channelHandle);
    aclrtContext currentCtx = nullptr;
    (void)aclrtGetCurrentContext(&currentCtx);
    {
        std::lock_guard<std::mutex> lock(g_hcommDfxMutex);
        g_hcommDfxChannelHandle = channelHandle;
        g_hcommDfxContext = currentCtx;
        if (!g_hcommDfxAtExitRegistered) {
            g_hcommDfxAtExitRegistered = true;
            (void)std::atexit(Mc2HcommDumpAtExit);
        }
    }
    try {
        OP_LOGE("AlltoAllMatmulV2", "before Mc2HcommDumpWorker");
        std::thread(Mc2HcommDumpWorker, channelHandle, currentCtx, deviceId).detach();
    } catch (const std::system_error &e) {
        OP_LOGE("AlltoAllMatmulV2", "spawn deferred hcomm dump thread failed(%s), dump at process exit", e.what());
    }
}
} // namespace
#endif

inline void AlltoAllMatmulV2ExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    const char *socName = aclrtGetSocName();
    if (std::strstr(socName, "Ascend950") == nullptr) {
        return;
    }
    uint64_t channelHandle = 0U;
    Mc2Exception::Mc2DumpTilingData(args, "AlltoAllMatmulV2",
        9U,   // tilingGmArgIdx
        0U,   // argsOffset
        8U,   // workspaceGmArgIdx
        0U,   // quantModeOffset
        offsetof(allToAllMatmulTilingData, workspaceLayout),  // wsLayoutOffsetNonQuant
        offsetof(allToAllMatmulTilingData, workspaceLayout),  // wsLayoutOffsetQuant
        offsetof(allToAllMatmulTilingData, workspaceLayout)); // wsLayoutOffsetApace
    Mc2Exception::Mc2DumpUrmaContext(args, "AlltoAllMatmulV2",
        0U,   // contextArgIdx
        0U,   // argsOffset
        9U,   // tilingGmArgIdx
        offsetof(allToAllMatmulTilingData, peermemDataSize),
        offsetof(allToAllMatmulTilingData, peermemLayout), &channelHandle);
    if (channelHandle != 0U) {
#if MC2_HCOMM_DFX_SHIM
        OP_LOGE("AlltoAllMatmulV2", "Mc2HcommTriggerDeferredDump before");
        Mc2HcommTriggerDeferredDump(channelHandle, static_cast<int32_t>(aclrtGetDeviceIdFromExceptionInfo(args)));
#else
        OP_LOGE("AlltoAllMatmulV2", "hcomm dfx shim unavailable, skip dump");
#endif
    } else {
        OP_LOGE("AlltoAllMatmulV2", "No valid channel handle, skip hcomm dump.");
    }
}
 
__attribute__((constructor)) void RegisterAlltoAllMatmulV2ExceptionFunc()
{
    IMPL_OP(AlltoAllMatmulV2)
        .ExceptionDumpParseFunc(AlltoAllMatmulV2ExceptionImplWrapper);
}
#endif // MC2_DFX_ENABLE
} // namespace MC2Tiling
