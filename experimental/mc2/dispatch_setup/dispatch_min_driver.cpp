/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License").
 *
 *  dispatch_min_driver.cpp —— 两算子集成驱动（单 die、两条流、写/等状态位串行同步）。
 *
 *  流程（最小功能，单 die、不跨卡）：
 *    1. acl 初始化 + 设备 + context + 两条 stream（streamSetup / streamTeardown）；
 *    2. 单 rank HCCL 通信域（CCU 拉起需要 HcclComm）；
 *    3. 分配共享状态区 statusGm（32B 对齐、memset 0）、oneBuf（host 填 slot0=1）、doneGm；
 *    4. CCU dispatch_setup 在 streamSetup 上拉起（HCOMM 路径）→ LocalCopyNb 把 1 写进 statusGm；
 *    5. AIV dispatch_teardown 经 aclnn 在 streamTeardown 上拉起 → 忙等 statusGm[0]==1 → 写 doneGm[0]=1；
 *    6. 同步两条流 → 拷 doneGm 回 host 校验 == 1（证明 teardown 等到了状态位并往下走）。
 *
 *  关键点：两条流无显式 event 依赖，teardown 的忙等轮询【本身】就是跨流同步——它一直
 *  自旋读 statusGm，直到 CCU 的写可见。这正是“根据写状态位等状态位的方式来同步”。
 *
 *  本机为 a5、仅作参考，不在此编译/上板；本文件聚焦 a6 可编译可跑通的结构。
 *  已坐实（从 a5 CANN 包查实，相对 include 串 a5/a6 一致；根目录 CMake 自动定位）：
 *    [T6✓] HCOMM 拉起 API 头：hccl/hccl_res.h（ThreadAcquireWithStream）、hcomm/ccu/hccl_ccu_res.h
 *          （CcuKernelRegister/Finish/Launch、CcuKernelHandle）、hcomm/ccu/ccu_assist_pub.h
 *          （hcomm::CcuRep::GetTokenInfo）、hcomm/ccu/ccu_kernel.h（CcuKernel/KernelCreator）、
 *          hcomm_res_defs.h（CommEngine COMM_ENGINE_CCU=5）、hcomm_primitives.h（ThreadHandle）。
 *  a6-TODO（真 a6 闭源、a5 看不到，需手抄校准）：
 *    [T1] LocalCopyNb a6 6 参原型 + CreateBlockCcuBuf（见 dispatch_setup_ccu）；
 *    [T2] WaitEvent a6 1 参形式；
 *    [T3] 状态字跨流可见性 / cache（见 op_kernel 注释）；
 *    [T7] 单 rank HCCL comm：rank_table.json（设备 ip/id 按环境填，模板见同目录 rank_table.json.tmpl）；
 *    [T8] SoC 串 ascend950pr_9599 是否 a6 真值（见 run.sh）；
 *    [T9] teardown 算子包 opapi 库名（CANN 惯例 libcust_opapi.so，装包后核实；CMake 已 find_library 兜底）。
 */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <memory>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnn/aclnn_base.h"

#include "dispatch_setup_ccu.h"            // 本工程：CCU kernel（含 ccu_kernel.h → CcuKernel/CcuRep/KernelCreator/CcuKernelHandle）
#include "aclnn_dispatch_teardown.h"       // teardown aclnn 算子对外接口

// HCOMM CCU 拉起 API：头文件已从 CANN 包坐实（相对 include 串 a5/a6 一致；include 根目录由 CMake
//   find_path 自动定位，见 CMakeLists）。原 [T6] 已解。
#include "hcomm_res_defs.h"                // CommEngine / COMM_ENGINE_CCU(=5)
#include "hcomm_primitives.h"              // ThreadHandle (= uint64_t)
#include "hccl/hccl_res.h"                 // HcclThreadAcquireWithStream
#include "hcomm/ccu/hccl_ccu_res.h"        // HcclCcuKernelRegister / RegisterFinish / Launch；CcuKernelHandle
#include "hcomm/ccu/ccu_assist_pub.h"      // hcomm::CcuRep::GetTokenInfo(va, size)

// ACLNN_SUCCESS 在本 CANN 版本公共 aclnn 头未导出（同 ACLNN_API），外部调用方自定义兜底（恒 0）。
#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS 0
#endif

namespace {

constexpr uint32_t STATUS_SLOTS = 8;                                  // 32B / 4 = 8 个 int32（对齐，避免 AIV 越界）
constexpr size_t   STATUS_BYTES = STATUS_SLOTS * sizeof(int32_t);     // 32B

#define ACLCHECK(expr)                                                                    \
    do {                                                                                  \
        aclError __r = (expr);                                                            \
        if (__r != ACL_SUCCESS) { printf("[ERR] %s ret=%d @%d\n", #expr, __r, __LINE__); return -1; } \
    } while (0)

#define HCCLCHECK(expr)                                                                   \
    do {                                                                                  \
        HcclResult __r = (expr);                                                          \
        if (__r != HCCL_SUCCESS) { printf("[ERR] %s ret=%d @%d\n", #expr, __r, __LINE__); return -1; } \
    } while (0)

// 用一个 device GM 指针 + shape 造 int32 的 aclTensor（contiguous, ND）。
aclTensor *MakeInt32Tensor(void *devAddr, int64_t numElem)
{
    std::vector<int64_t> shape = {numElem};
    std::vector<int64_t> strides = {1};
    return aclCreateTensor(shape.data(), shape.size(), ACL_INT32, strides.data(), 0,
                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), devAddr);
}

// ---- 段 A：CCU dispatch_setup 经 HCOMM 在 streamSetup 上拉起（裁剪自 ccu_kernel_test 的 ccu_kernel_launch）----
int LaunchDispatchSetupCcu(HcclComm comm, aclrtStream streamSetup,
                           void *statusGm, void *oneBuf)
{
#ifdef DISPATCH_CCU_A6
    uint32_t rid = 0, rsz = 0;
    HCCLCHECK(HcclGetRankId(comm, &rid));
    HCCLCHECK(HcclGetRankSize(comm, &rsz));

    // 单 die：channel_num = rsz - 1 = 0，纯本地 LocalCopyNb 不取跨卡 channel（[T7]）。
    std::vector<ChannelHandle> channels;  // 空

    ThreadHandle mainThread = 0;
    HCCLCHECK(HcclThreadAcquireWithStream(comm, CommEngine::COMM_ENGINE_CCU, streamSetup, 1, &mainThread));

    mc2::CcuKernelArgDispatchSetup kernelArg(rsz, rid);
    kernelArg.channels = channels;

    hcomm::KernelCreator creator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<mc2::CcuKernelDispatchSetup>(arg);
    };
    void *creatorPtr = reinterpret_cast<void *>(&creator);
    void *argPtr = static_cast<void *>(&kernelArg);

    CcuKernelHandle handle = 0;
    HCCLCHECK(HcclCcuKernelRegister(comm, &handle, creatorPtr, argPtr));
    HCCLCHECK(HcclCcuKernelRegisterFinish(comm));

    // 每块独立 buffer 各算各的 token（host 端）。
    uint64_t statusTok = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(statusGm), STATUS_BYTES);
    uint64_t oneTok    = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(oneBuf), STATUS_BYTES);

    mc2::CcuTaskArgDispatchSetup taskArg(
        reinterpret_cast<uint64_t>(statusGm), statusTok,
        reinterpret_cast<uint64_t>(oneBuf), oneTok);

    HCCLCHECK(HcclCcuKernelLaunch(comm, mainThread, handle, static_cast<void *>(&taskArg)));
    printf("[driver] dispatch_setup(CCU) launched on streamSetup\n");
    return 0;
#else
    // a5：HCOMM CCU 拉起符号（HcclThreadAcquireWithStream / HcclCcuKernel* / CcuKernel 构造）
    //   未导出于 a5 libhcomm（nm 实测为空），属 a6-only。a5 上桩掉以便驱动其余部分编译+链接通过；
    //   a6 移植：cmake -DDISPATCH_CCU_A6=ON（同步启用 dispatch_setup_ccu.cpp 段10 真实写）。
    (void)comm; (void)streamSetup; (void)statusGm; (void)oneBuf;
    printf("[driver] (a5) dispatch_setup(CCU) launch skipped — a6-only hcomm 符号；a6 上 -DDISPATCH_CCU_A6\n");
    return 0;
#endif
}

// ---- 段 B：AIV dispatch_teardown 经 aclnn 在 streamTeardown 上拉起 ----
int LaunchDispatchTeardownAclnn(aclrtStream streamTeardown, void *statusGm, void *doneGm)
{
    aclTensor *statusT = MakeInt32Tensor(statusGm, STATUS_SLOTS);
    aclTensor *doneT   = MakeInt32Tensor(doneGm, STATUS_SLOTS);

    uint64_t wsSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus s = aclnnDispatchTeardownGetWorkspaceSize(statusT, doneT, &wsSize, &executor);
    if (s != ACLNN_SUCCESS) { printf("[ERR] GetWorkspaceSize ret=%d\n", s); return -1; }

    void *ws = nullptr;
    if (wsSize > 0) {
        ACLCHECK(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    s = aclnnDispatchTeardown(ws, wsSize, executor, streamTeardown);
    if (s != ACLNN_SUCCESS) { printf("[ERR] aclnnDispatchTeardown ret=%d\n", s); return -1; }
    printf("[driver] dispatch_teardown(AIV/aclnn) launched on streamTeardown\n");

    // aclTensor / workspace 在流同步后释放（此处简化：交由进程退出回收；如需精确，sync 后 aclDestroyTensor + aclrtFree(ws)）。
    (void)statusT; (void)doneT; (void)ws;
    return 0;
}

}  // namespace

int main(int argc, char **argv)
{
    // 极简参数：rank_id 与 rankTableFile（[T7] 单 die 的 rank table 内容需按 a6 环境给）。
    int32_t rankId = 0;
    const char *rankTableFile = (argc > 1) ? argv[1] : "./rank_table.json";

    ACLCHECK(aclInit(nullptr));
    ACLCHECK(aclrtSetDevice(rankId));
    aclrtContext context = nullptr;
    ACLCHECK(aclrtCreateContext(&context, rankId));

    aclrtStream streamSetup = nullptr, streamTeardown = nullptr;
    ACLCHECK(aclrtCreateStream(&streamSetup));
    ACLCHECK(aclrtCreateStream(&streamTeardown));

    // ---- 单 rank HCCL 通信域（CCU 拉起需要 HcclComm）[T7] ----
    // [T10] a6 使能 CCU 可能还需：(a) config.hcclOpExpansionMode 设为 CCU 模式
    //   （HcclOpExpansionMode: CCU_MS=4 / CCU_SCHED=5）;(b) HcclCommQueryCcuIns 校验 CCU 实例数（别 hardcode）。
    //   低层 HcclThreadAcquireWithStream(COMM_ENGINE_CCU) 路径是否依赖 opExpansionMode 待 a6 核实——
    //   参考 ccu_kernel_launch 未显式设、可能不需要;a5/a6 仅靠运行期 SOC_VERSION 串区分（DevType 都是 DEV_TYPE_950）。
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    HcclComm comm = nullptr;
    HCCLCHECK(HcclCommInitClusterInfoConfig(rankTableFile, rankId, &config, &comm));

    // ---- 共享 GM：status（CCU 写 / AIV 读）、oneBuf（CCU 写源）、done（AIV 写 / host 校验）----
    void *statusGm = nullptr, *oneBuf = nullptr, *doneGm = nullptr;
    ACLCHECK(aclrtMalloc(&statusGm, STATUS_BYTES, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&oneBuf, STATUS_BYTES, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMalloc(&doneGm, STATUS_BYTES, ACL_MEM_MALLOC_HUGE_FIRST));
    ACLCHECK(aclrtMemset(statusGm, STATUS_BYTES, 0, STATUS_BYTES));  // 状态区清 0（未就绪=0）
    ACLCHECK(aclrtMemset(doneGm, STATUS_BYTES, 0, STATUS_BYTES));

    // oneBuf slot0 = 1（必须 memcpy，不能 memset 1：memset 是逐字节会写成 0x01010101）。
    {
        std::vector<int32_t> ones(STATUS_SLOTS, 0);
        ones[0] = mc2::DISPATCH_READY_VALUE;  // 仅 slot0 = 1，其余 0（与 teardown 只判 slot0 对齐）
        ACLCHECK(aclrtMemcpy(oneBuf, STATUS_BYTES, ones.data(), STATUS_BYTES, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // ---- 串行同步：先后异步拉起两算子到两条流，靠 teardown 忙等轮询完成跨流同步 ----
    if (LaunchDispatchSetupCcu(comm, streamSetup, statusGm, oneBuf) != 0) return -1;
    if (LaunchDispatchTeardownAclnn(streamTeardown, statusGm, doneGm) != 0) return -1;

    ACLCHECK(aclrtSynchronizeStream(streamSetup));
    ACLCHECK(aclrtSynchronizeStream(streamTeardown));

    // ---- 校验：done[0] 应为 1（teardown 等到状态位并往下走）----
    std::vector<int32_t> doneHost(STATUS_SLOTS, 0);
    ACLCHECK(aclrtMemcpy(doneHost.data(), STATUS_BYTES, doneGm, STATUS_BYTES, ACL_MEMCPY_DEVICE_TO_HOST));
    if (doneHost[0] == mc2::DISPATCH_READY_VALUE) {
        printf("[driver] PASS: teardown observed status==1 (done=%d)\n", doneHost[0]);
    } else {
        printf("[driver] FAIL: done=%d (expected 1)\n", doneHost[0]);
    }

    // ---- 清理 ----
    aclrtFree(statusGm);
    aclrtFree(oneBuf);
    aclrtFree(doneGm);
    HcclCommDestroy(comm);
    aclrtDestroyStream(streamSetup);
    aclrtDestroyStream(streamTeardown);
    aclrtDestroyContext(context);
    aclrtResetDevice(rankId);
    aclFinalize();
    return 0;
}
