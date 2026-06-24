/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License").
 *
 *  dispatch_min_driver.cpp —— 两算子集成驱动（2 die / 2 rank、每 rank 两条流、写/等状态位串行同步）。
 *
 *  流程（最小功能，2 die、MPI 起 2 进程、每 rank 各自本地写+轮询、不跨卡传数据）：
 *    1. MPI_Init → 每进程一个 rank（rank_id = MPI rank，dev_id = rank）；
 *    2. acl 初始化 + 设备 + context + 两条 stream（streamSetup / streamTeardown）；
 *    3. 【CCU 容器关键】2-rank HCCL 通信域：root 取 RootInfo → MPI_Bcast 广播 → 各 rank
 *       HcclCommInitRootInfoConfig(rank_size=2, hcclOpExpansionMode=CCU)。忠实镜像已在 a6 跑通的
 *       hcomm-test 基线 init_hcclComm（hccl_test_common.cc:861-941）——CCU 容器(ccuContainer)由库内部
 *       在【CCU 扩展模式 + rank_size>=2】下创建。单 rank 不建容器，故必须 2 die（mpirun -n 2）。
 *    4. 分配本 rank 的共享状态区 statusGm（32B 对齐、memset 0）、oneBuf（host 填 slot0=1）、doneGm；
 *    5. CCU dispatch_setup 在 streamSetup 上拉起（HCOMM 路径）→ 经 a6 已验证的 0.5RTT 跨卡写
 *       WriteNbWithArriveNotify 把 1 写进【对端 rank】statusGm（本卡 statusGm 由对端写）；
 *    6. AIV dispatch_teardown 经 aclnn 在 streamTeardown 上拉起 → 忙等 statusGm[0]==1 → 写 doneGm[0]=1；
 *    7. 同步两条流 → 拷 doneGm 回 host 校验 == 1（证明本 rank teardown 等到状态位并往下走）。
 *
 *  关键点：两条流无显式 event 依赖，teardown 的忙等轮询【本身】就是跨流同步——它一直自旋读
 *    statusGm，直到 CCU 的写可见。这正是"根据写状态位等状态位的方式来同步"。两个 rank 完全对称、
 *    各自独立 PASS；2-die 仅为满足 CCU 容器创建，最小 case 不做真跨卡数据交换。
 *
 *  本机为 a5、仅作参考，不在此编译/上板；本文件聚焦 a6 可编译可跑通的结构。
 *  已坐实（从 a5 CANN 包查实，相对 include 串 a5/a6 一致；根目录 CMake 自动定位）：
 *    [T6✓] HCOMM 拉起 API 头：hccl/hccl_res.h（ThreadAcquireWithStream）、hcomm/ccu/hccl_ccu_res.h
 *          （CcuKernelRegister/Finish/Launch、CcuKernelHandle）、hcomm/ccu/ccu_assist_pub.h
 *          （hcomm::CcuRep::GetTokenInfo）、hcomm/ccu/ccu_kernel.h（CcuKernel/KernelCreator）、
 *          hcomm_res_defs.h（CommEngine COMM_ENGINE_CCU=5）、hcomm_primitives.h（ThreadHandle）、
 *          hccl/hccl_rank_graph.h（HcclRankGraphGetLinks/CommLink/HcclChannelDesc/HcclChannelAcquire）。
 *  a6-TODO（a5 看不到、需在 a6 上实测确认）：
 *    [T1] LocalCopyNb a6 6 参原型 + CreateBlockCcuBuf（见 dispatch_setup_ccu）；
 *    [T2] WaitEvent a6 1 参形式；
 *    [T3] 状态字跨流可见性 / cache（见 op_kernel 注释）；
 *    [T5] CCU 扩展模式值：默认 DISPATCH_CCU_EXPANSION_MODE=6(ccu_sched)，按 a6 基线 -a 实际值校（可 =5）；
 *    [T7] 2-die 拓扑：dev_id=rank（假定本节点 die0/die1=dev0/dev1）；若 2 die 不是 dev0/1 改 devId 映射；
 *         MPI 起 2 进程（mpirun -n 2），无需 rank_table（RootInfo + MPI 广播自给 bootstrap）；
 *    [T8] SoC 串 ascend950pr_9599 是否 a6 真值（见 run.sh）；
 *    [T9] teardown 算子包 opapi 库名（CANN 惯例 libcust_opapi.so，装包后核实；CMake 已 find_library 兜底）；
 *    [T10] 最小本地 kernel 是否真需要建 channel：本版按基线忠实建 rsz-1 个 UBC_CTP channel（保 CCU comm
 *          处于基线验证态）；若 a6 实测纯本地 LocalCopyNb 的 Register/Launch 不依赖 channel，可去掉建 channel
 *          一段（kernel Algorithm 本就不在 channel 上发 op）。
 */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <memory>

#include <mpi.h>                            // 2-die：rootInfo 跨 rank 广播（镜像 hcomm-test init_hcclComm）

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
#ifdef DISPATCH_CCU_A6
#include "hccl/hccl_rank_graph.h"          // [a6] HcclRankGraphGetLinks / CommLink / HcclChannelDesc / HcclChannelAcquire
#endif

// ACLNN_SUCCESS 在本 CANN 版本公共 aclnn 头未导出（同 ACLNN_API），外部调用方自定义兜底（恒 0）。
#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS 0
#endif

// CCU 扩展模式值 —— 修 a6 "ccuContainer is nullptr" 的关键开关。
//   ★ 用 a6 基线「数组约定」ccu_ms=5 / ccu_sched=6（hcomm-test/src/common/hccl_test_common.cc:36 + :867），
//     【不是】a5 pkg 头枚举 HcclOpExpansionMode 的 CCU_MS=4 / CCU_SCHED=5 —— a5/a6 枚举漂移，按 a6 基线取值！
//   默认 6(ccu_sched)；用你 dispatch/combine 基线 -a 实际用的那个，可 -DDISPATCH_CCU_EXPANSION_MODE=5 覆盖。
#ifndef DISPATCH_CCU_EXPANSION_MODE
#define DISPATCH_CCU_EXPANSION_MODE 6
#endif

namespace {

constexpr uint32_t STATUS_SLOTS = 8;                                  // 32B / 4 = 8 个 int32（对齐，避免 AIV 越界）
constexpr size_t   STATUS_BYTES = STATUS_SLOTS * sizeof(int32_t);     // 32B

#define ACLCHECK(expr)                                                                    \
    do {                                                                                  \
        aclError __r = (expr);                                                            \
        if (__r != ACL_SUCCESS) { printf("[ERR] %s ret=%d @%d\n", #expr, __r, __LINE__); MPI_Abort(MPI_COMM_WORLD, -1); return -1; } \
    } while (0)

#define HCCLCHECK(expr)                                                                   \
    do {                                                                                  \
        HcclResult __r = (expr);                                                          \
        if (__r != HCCL_SUCCESS) { printf("[ERR] %s ret=%d @%d\n", #expr, __r, __LINE__); MPI_Abort(MPI_COMM_WORLD, -1); return -1; } \
    } while (0)

// 用一个 device GM 指针 + shape 造 int32 的 aclTensor（contiguous, ND）。
aclTensor *MakeInt32Tensor(void *devAddr, int64_t numElem)
{
    std::vector<int64_t> shape = {numElem};
    std::vector<int64_t> strides = {1};
    return aclCreateTensor(shape.data(), shape.size(), ACL_INT32, strides.data(), 0,
                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), devAddr);
}

#ifdef DISPATCH_CCU_A6
// A6 范式 channel 获取：本卡 srcRank → 对端 dstRank，从 RankGraph link 填 endpoint（UBC_CTP + notifyNum=2）。
//   裁剪自基线 hcomm_moe_dispatch_setup_test.cc:84-124，与其 channel 序铁律一致。
//   [T10] 最小本地 kernel 不在 channel 上发 op，建 channel 仅为把 CCU comm 拉到基线验证态。
static HcclResult HcclGetOneChannel(HcclComm comm, CommEngine engine,
    uint32_t srcRank, uint32_t dstRank, ChannelHandle *channel)
{
    uint32_t netLayer = 0, listSize = 0;
    CommLink *linkList = nullptr;
    HcclResult ret = HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &linkList, &listSize);
    if (ret != HCCL_SUCCESS) { printf("[ERR] HcclRankGraphGetLinks rank %u->%u ret=%d\n", srcRank, dstRank, ret); return ret; }

    HcclChannelDesc desc;
    ret = HcclChannelDescInit(&desc, 1);
    if (ret != HCCL_SUCCESS) { printf("[ERR] HcclChannelDescInit ret=%d\n", ret); return ret; }
    const CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
    const char *tmpArg[] = { "moe_dispatch_05rtt" };       // 与基线 resGroupTag 一致（CCU 资源组）
    char **resGroupTag = const_cast<char **>(tmpArg);

    bool found = false;
    for (uint32_t idx = 0; idx < listSize; ++idx) {
        CommLink link = linkList[idx];
        if (link.linkAttr.linkProtocol == protocol) {
            desc.remoteRank              = dstRank;
            desc.notifyNum               = 2;
            desc.channelProtocol         = link.linkAttr.linkProtocol;
            desc.localEndpoint.protocol  = link.srcEndpointDesc.protocol;
            desc.localEndpoint.commAddr  = link.srcEndpointDesc.commAddr;
            desc.localEndpoint.loc       = link.srcEndpointDesc.loc;
            desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
            desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
            desc.remoteEndpoint.loc      = link.dstEndpointDesc.loc;
            desc.ccuUbAttr.resGroupTag    = resGroupTag;
            desc.ccuUbAttr.resGroupTagNum = 1;
            found = true;
            break;
        }
    }
    if (!found) { printf("[ERR] no UBC_CTP link between rank %u and %u\n", srcRank, dstRank); return HCCL_E_NOT_FOUND; }
    return HcclChannelAcquire(comm, engine, &desc, 1, channel);
}
#endif  // DISPATCH_CCU_A6

// ---- 段 A：CCU dispatch_setup 经 HCOMM 在 streamSetup 上拉起（裁剪自基线 ccu_kernel_launch）----
//   2-die：建 rsz-1 个 channel → WithStream 取 thread → 注册 → 拉起。
//   kernel 在 channel 上跑 0.5RTT 跨卡写：PreSync 换对端 status 地址/token → WriteNbWithArriveNotify
//   把 oneBuf(1) 写进对端 status → WaitallpeersarriveNotify 等本卡被对端写好。channel 是真消费、非占位。
int LaunchDispatchSetupCcu(HcclComm comm, aclrtStream streamSetup,
                           void *statusGm, void *oneBuf)
{
#ifdef DISPATCH_CCU_A6
    uint32_t rid = 0, rsz = 0;
    HCCLCHECK(HcclGetRankId(comm, &rid));
    HCCLCHECK(HcclGetRankSize(comm, &rsz));

    // 2-die：channel_num = rsz - 1（R=2 → 1 个对端 channel）。channels[k] → dstRank=(k<rid)?k:k+1。
    //   [T10✓] kernel Algorithm 现在【真正消费 channel】跑 0.5RTT 跨卡写（PreSync + WriteNbWithArriveNotify）。
    //   channel 序与 kernel 内 channelIdx=(r<self)?r:r-1 严格一致（resGroupTag 绑 CCU 资源组，与 kernel TAG 同字符串）。
    const uint32_t channelNum = rsz - 1U;
    std::vector<ChannelHandle> channels(channelNum);
    for (uint32_t k = 0; k < channelNum; ++k) {
        uint32_t dstRank = (k < rid) ? k : k + 1;
        HCCLCHECK(HcclGetOneChannel(comm, CommEngine::COMM_ENGINE_CCU, rid, dstRank, &channels[k]));
    }

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
    printf("[driver] dispatch_setup(CCU) launched on streamSetup (rid=%u rsz=%u channels=%u)\n", rid, rsz, channelNum);
    return 0;
#else
    // a5：HCOMM CCU 拉起符号（HcclThreadAcquireWithStream / HcclCcuKernel* / CcuKernel 构造 / channel 建链）
    //   未导出于 a5 libhcomm（nm 实测为空），属 a6-only。a5 上桩掉以便驱动其余部分编译+链接通过；
    //   a6 移植：cmake -DDISPATCH_CCU_A6=ON（同步启用 dispatch_setup_ccu.cpp 段10 真实写 + channel 建链）。
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
    // ---- MPI：2 die / 2 进程，每进程一个 rank（镜像 hcomm-test）----
    MPI_Init(&argc, &argv);
    int mpiRank = 0, mpiSize = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    setvbuf(stdout, nullptr, _IONBF, 0);   // ★ stdout 无缓冲：上板 fault/卡死时每句打印立刻落屏、不被缓冲吃掉
                                           //   （host printf 本就正确，非 kernel 用法；崩溃前的诊断必须看得到）
    const uint32_t rankId   = static_cast<uint32_t>(mpiRank);
    const uint32_t rankSize = static_cast<uint32_t>(mpiSize);     // 期望 2（mpirun -n 2）
    const int32_t  devId    = static_cast<int32_t>(rankId);       // [T7] 单节点：die = rank（dev0/dev1）；若拓扑不同改此映射

    // ★ 构建路径自证（审核 #2 修订）：本工程 CMake 把 dispatch_setup_ccu.cpp 的编入与 DISPATCH_CCU_A6 宏【绑定】
    //   （a6 才编入 .cpp 且统一传宏；a5 不编 .cpp、不传宏）——kernel/driver 宏天然一致，无需跨 TU 比对（旧版那种比对
    //   会引用 a5 上根本没编译的 kernel 函数 → a5 链接 undefined，已纠）。此处仅 #ifdef 打印路径，作 a6 启动确认。
#ifdef DISPATCH_CCU_A6
    printf("[rank %u] build: DISPATCH_CCU_A6=ON —— CCU 跨卡写路径已编入\n", rankId);
#else
    printf("[rank %u] build: DISPATCH_CCU_A6=OFF —— a5 桩，CCU 跨卡写不执行（仅编译门禁）\n", rankId);
#endif

    if (rankSize < 2U) {
        printf("[rank %u][FATAL] rank_size=%u (<2)：CCU 容器需 rank_size>=2 才会创建（硬前置，非警告）。请 mpirun -n 2。\n",
               rankId, rankSize);
        MPI_Abort(MPI_COMM_WORLD, -1); return -1;
    }

    ACLCHECK(aclInit(nullptr));
    ACLCHECK(aclrtSetDevice(devId));
    // ★ 用 aclrtSetDevice 隐式建的【默认 context】，不显式 aclrtCreateContext——否则 stream 建在显式
    //   context、而 CCU/HCCL 运行时走默认 context → rtCCULaunch 报 "stream not in current context"(107003)。
    //   对齐基线 device_init（hccl_test_common.cc:841 仅 aclrtSetDevice、全程无 CreateContext/SetCurrentContext）。

    aclrtStream streamSetup = nullptr, streamTeardown = nullptr;
    ACLCHECK(aclrtCreateStream(&streamSetup));
    ACLCHECK(aclrtCreateStream(&streamTeardown));
    ACLCHECK(aclrtSetStreamFailureMode(streamSetup, 1));          // 遇错即停（对齐基线 hccl_test_common.cc:857，免掩盖 CCU 失败）
    ACLCHECK(aclrtSetStreamFailureMode(streamTeardown, 1));

    // ---- 2-rank HCCL 通信域：CCU 拉起需要【CCU 扩展模式 + rank_size>=2】的 HcclComm ----
    // ★★ 修 a6 "[HcclCcuKernelRegister] ptr [ccuContainer] is nullptr"：根因 = comm 没在 CCU 扩展模式下、
    //    或 rank_size<2 下初始化，ccuContainer 不会被创建 → Register 取到空指针。对齐已跑通的 hcomm-test 基线
    //    (hccl_test_common.cc:861-941)：root HcclGetRootInfo → MPI_Bcast 广播 rootInfo → 各 rank
    //    HcclCommInitRootInfoConfig(rank_size, &rootInfo, rank_id, &config{hcclOpExpansionMode=CCU}, &comm)。
    HcclRootInfo rootInfo{};                                       // 零初始化（非 root 在 Bcast 前不读，防脏字节泄漏）
    if (rankId == 0U) {
        HCCLCHECK(HcclGetRootInfo(&rootInfo));                     // 仅 root 取，广播给其余 rank
    }
    if (MPI_Bcast(&rootInfo, HCCL_ROOT_INFO_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
        printf("[rank %u][ERR] MPI_Bcast(rootInfo) failed\n", rankId); MPI_Abort(MPI_COMM_WORLD, -1); return -1;
    }

    HcclCommConfig config{};                                      // 零初始化（对齐基线 ={0}，免栈垃圾混入建 comm 的 config——它正是建 ccuContainer 的那个 config）
    HcclCommConfigInit(&config);
    config.hcclWorldRankID = rankId;
#ifdef DISPATCH_CCU_A6
    config.hcclOpExpansionMode = DISPATCH_CCU_EXPANSION_MODE;      // ★ 关键：CCU 扩展模式 → 创建 ccuContainer（a5 骨架不设）
#endif
    HcclComm comm = nullptr;
    HCCLCHECK(HcclCommInitRootInfoConfig(rankSize, &rootInfo, rankId, &config, &comm));

    // ---- 本 rank 的共享 GM：status（CCU 写 / AIV 读）、oneBuf（CCU 写源）、done（AIV 写 / host 校验）----
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

    // ---- 【诊断版】隔离 CCU 写端：先拉 CCU → 单独同步 CCU 流 → host 读回 statusGm 看 CCU 到底写没写对 → 再拉 AIV ----
    if (LaunchDispatchSetupCcu(comm, streamSetup, statusGm, oneBuf) != 0) { MPI_Abort(MPI_COMM_WORLD, -1); return -1; }
    printf("[rank %u] syncing streamSetup(CCU)... 若 CCU 非法访存，将 fault/卡在此处\n", rankId);
    ACLCHECK(aclrtSynchronizeStream(streamSetup));   // CCU 若非法访存 → 这里 fault/报错，而非往下走
    {
        std::vector<int32_t> st(STATUS_SLOTS, -1);
        ACLCHECK(aclrtMemcpy(st.data(), STATUS_BYTES, statusGm, STATUS_BYTES, ACL_MEMCPY_DEVICE_TO_HOST));
        printf("[rank %u][DIAG] CCU 后 statusGm[0..3]=%d %d %d %d（期望 slot0=%d）→ %s\n",
               rankId, st[0], st[1], st[2], st[3], mc2::DISPATCH_READY_VALUE,
               st[0] == mc2::DISPATCH_READY_VALUE
                   ? "CCU 写 1 成功 → 问题在 AIV 没观测到(cache/跨流可见性)"
                   : "CCU 没写进 1 → 问题在 CCU 写端(地址/token/LocalCopyNb)");
    }
    // CCU 已完成且 statusGm 已被 host 读过（建立可见性），再拉 AIV——若 CCU 写对，AIV 此时应立即读到 1
    if (LaunchDispatchTeardownAclnn(streamTeardown, statusGm, doneGm) != 0) { MPI_Abort(MPI_COMM_WORLD, -1); return -1; }
    ACLCHECK(aclrtSynchronizeStream(streamTeardown));

    // ---- 校验：done[0] 应为 1（本 rank teardown 等到状态位并往下走）----
    std::vector<int32_t> doneHost(STATUS_SLOTS, 0);
    ACLCHECK(aclrtMemcpy(doneHost.data(), STATUS_BYTES, doneGm, STATUS_BYTES, ACL_MEMCPY_DEVICE_TO_HOST));
    if (doneHost[0] == mc2::DISPATCH_READY_VALUE) {
        printf("[rank %u] PASS: teardown observed status==1 (done=%d)\n", rankId, doneHost[0]);
    } else {
        printf("[rank %u] FAIL: done=%d (expected 1)\n", rankId, doneHost[0]);
    }

    // ---- 清理 ----
    aclrtFree(statusGm);
    aclrtFree(oneBuf);
    aclrtFree(doneGm);
    HcclCommDestroy(comm);
    aclrtDestroyStream(streamSetup);
    aclrtDestroyStream(streamTeardown);
    aclrtResetDevice(devId);
    aclFinalize();
    MPI_Finalize();
    return 0;
}
