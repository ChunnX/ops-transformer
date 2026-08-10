// ============================================================================
// URMA P2P Benchmark Host - 单进程多线程模式
//
// 流程: aclInit → HcclGetRootInfo → 每设备一个线程 →
//   aclrtSetDevice → HcclCommInitRootInfoConfig →
//   aclrtMalloc workspace → HcclCommMemReg →
//   HcclRankGraphGetLinks(枚举UBC链路) → HcclChannelAcquire(跨rank握手) →
//   HcommChannelGetPtrByHandle(handle→device VA) →
//   <<<>>> kernel launch(loop在kernel内部) → 读timing
// ============================================================================

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <sstream>
#include <thread>
#include <atomic>
#include <chrono>
#include <dlfcn.h>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl/hcomm_primitives.h"
#include "hccl/hcomm_res_defs.h"
#include "hccl/hccl_types.h"
#include "../kernel/sqe_write/sqe_write_launch.h"
#include "../kernel/sqe_write_with_notify/sqe_write_with_notify_launch.h"
#include "../kernel/hcomm/hcomm_launch.h"
#include "../kernel/common/urma_p2p_bench_tiling.h"

constexpr double NS_PER_CYCLE = 1.0;
constexpr int64_t BLOCK_NUM = 1;
constexpr uint64_t DEFAULT_BUF_BYTES = 100 * 1024 * 1024ULL;
constexpr uint32_t DEFAULT_NUM_SQE = 16;
constexpr uint64_t DEFAULT_SQE_BYTES = 65536;
constexpr uint32_t DEFAULT_DB_INTERVAL = 1;
constexpr uint32_t TIMING_SLOTS_SIZE = 8;

// UBC_CTP=4, UBC_TP=5: AIV URMA通信协议(同机NPU间的UB fabric链路)
constexpr int COMM_PROTOCOL_UBC_CTP_VALUE = 4;
constexpr int COMM_PROTOCOL_UBC_TP_VALUE = 5;

// HcommChannelGetPtrByHandle: 旧版hcomm ABI需要把ChannelHandle翻译成device VA
// 新版CANN(9.1.0+)的HcclChannelAcquire直接返回device VA, 此符号不存在, 用dlsym兼容
typedef uint64_t ChannelPtr;
typedef int32_t HcommResult;
constexpr HcommResult HCOMM_SUCCESS_VALUE = 0;
extern "C" HcommResult HcommChannelGetPtrByHandle(const ChannelHandle *channelList,
                                                   uint32_t listNum, ChannelPtr *channelPtr);

struct BenchArgs {
    uint32_t numSqe = DEFAULT_NUM_SQE;
    uint64_t sqeBytes = DEFAULT_SQE_BYTES;
    uint64_t bufBytes = DEFAULT_BUF_BYTES;
    uint32_t dbInterval = DEFAULT_DB_INTERVAL;
    uint32_t loopCount = 100;
    uint32_t sqeMode = 1;
    uint32_t devCount = 2;
    std::vector<uint32_t> devIds;
    uint32_t peerOverride = 0;
    bool hasPeer = false;
};

static BenchArgs g_args;

struct BenchCtx {
    uint32_t rank{0};
    uint32_t worldSize{0};
    HcclComm hcclComm{nullptr};
    void *workspaceDev{nullptr};
    HcclMemHandle workspaceMemHandle{nullptr};
    void *channelPtrTableDev{nullptr};
    void *stateGmDev{nullptr};
    std::vector<ChannelHandle> channels;
    std::vector<ChannelPtr> channelPtrsHost;
    uint32_t numLayers{1};
};

// 枚举UBC_CTP/UBC_TP链路, 为每个peer构建一个HcclChannelDesc(附带workspace的memHandle)
static HcclResult build_channel_descs(HcclComm comm, uint32_t self, uint32_t world,
                                       HcclMemHandle *myMemHandle,
                                       std::vector<HcclChannelDesc> &outDescs,
                                       std::vector<size_t> &outSlots,
                                       uint32_t &outNumLayers)
{
    outDescs.clear();
    outSlots.clear();
    outNumLayers = 0;

    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    HcclResult r = HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum);
    if (r != HCCL_SUCCESS) return r;
    outNumLayers = netLayerNum;

    for (uint32_t layer = 0; layer < netLayerNum; ++layer) {
        for (uint32_t peer = 0; peer < world; ++peer) {
            if (peer == self) continue;
            CommLink *linkList = nullptr;
            uint32_t listSize = 0;
            r = HcclRankGraphGetLinks(comm, netLayers[layer], self, peer, &linkList, &listSize);
            if (r != HCCL_SUCCESS) continue;

            for (uint32_t i = 0; i < listSize; ++i) {
                int p = static_cast<int>(linkList[i].linkAttr.linkProtocol);
                if (p != COMM_PROTOCOL_UBC_CTP_VALUE && p != COMM_PROTOCOL_UBC_TP_VALUE)
                    continue;

                HcclChannelDesc desc;
                HcclChannelDescInit(&desc, 1);
                desc.remoteRank              = peer;
                desc.channelProtocol         = linkList[i].linkAttr.linkProtocol;
                desc.localEndpoint.protocol  = linkList[i].srcEndpointDesc.protocol;
                desc.localEndpoint.commAddr  = linkList[i].srcEndpointDesc.commAddr;
                desc.localEndpoint.loc       = linkList[i].srcEndpointDesc.loc;
                desc.remoteEndpoint.protocol = linkList[i].dstEndpointDesc.protocol;
                desc.remoteEndpoint.commAddr = linkList[i].dstEndpointDesc.commAddr;
                desc.remoteEndpoint.loc      = linkList[i].dstEndpointDesc.loc;
                desc.notifyNum               = 3;
                desc.memHandles              = myMemHandle;
                desc.memHandleNum            = 1;
                outDescs.push_back(desc);
                outSlots.push_back((static_cast<size_t>(layer) * 1 + 0) * world + peer);
                break;
            }
        }
    }
    return outDescs.empty() ? HCCL_E_NOT_FOUND : HCCL_SUCCESS;
}

// setup_ctx: 完整的通信上下文建立流程(每步都有[1/6]~[6/6]日志)
// 1. aclrtMalloc workspace(URMA目标缓冲区)
// 2. HcclCommMemReg(注册workspace给HCCL, 使对端URMA可写)
// 3. build_channel_descs(枚举UBC链路, 构建ChannelDesc)
// 4. HcclChannelAcquire(跨rank握手, 获取ChannelHandle)
// 5. resolve channel ptrs(handle→device VA, 兼容新旧ABI)
// 6. alloc table+state GM(channelPtrTable和state持久化到device)
static BenchCtx *setup_ctx(HcclComm comm, uint32_t rank, uint32_t world, uint64_t bufBytes)
{
    static thread_local BenchCtx ctx;
    ctx.rank = rank;
    ctx.worldSize = world;

    const std::string tag = "urma_p2p_bench_" + std::to_string(rank);
    fprintf(stderr, "[INFO] rank %u: [1/6] aclrtMalloc workspace (%lu bytes)\n", rank, bufBytes);
    aclError ar = aclrtMalloc(&ctx.workspaceDev, bufBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ar != ACL_SUCCESS) { fprintf(stderr, "[ERROR] rank %u: aclrtMalloc workspace failed: %d\n", rank, ar); return nullptr; }
    aclrtMemset(ctx.workspaceDev, bufBytes, 0, bufBytes);
    fprintf(stderr, "[INFO] rank %u: [1/6] workspace=%p\n", rank, ctx.workspaceDev);

    fprintf(stderr, "[INFO] rank %u: [2/6] HcclCommMemReg\n", rank);
    CommMem mem;
    mem.type = COMM_MEM_TYPE_DEVICE;
    mem.addr = ctx.workspaceDev;
    mem.size = bufBytes;
    HcclResult r = HcclCommMemReg(comm, tag.c_str(), &mem, &ctx.workspaceMemHandle);
    if (r != HCCL_SUCCESS) { fprintf(stderr, "[ERROR] rank %u: HcclCommMemReg failed: %d\n", rank, r); return nullptr; }
    fprintf(stderr, "[INFO] rank %u: [2/6] memHandle=%p\n", rank, ctx.workspaceMemHandle);

    fprintf(stderr, "[INFO] rank %u: [3/6] build_channel_descs\n", rank);
    std::vector<HcclChannelDesc> descs;
    std::vector<size_t> descSlots;
    r = build_channel_descs(comm, rank, world, &ctx.workspaceMemHandle, descs, descSlots, ctx.numLayers);
    if (r != HCCL_SUCCESS) { fprintf(stderr, "[ERROR] rank %u: build_channel_descs failed: %d\n", rank, r); return nullptr; }
    fprintf(stderr, "[INFO] rank %u: [3/6] descs=%zu layers=%u\n", rank, descs.size(), ctx.numLayers);

    fprintf(stderr, "[INFO] rank %u: [4/6] HcclChannelAcquire (waiting for peer...)\n", rank);
    ctx.channels.resize(descs.size(), 0);
    r = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AIV,
                           descs.data(), static_cast<uint32_t>(descs.size()),
                           ctx.channels.data());
    if (r != HCCL_SUCCESS) { fprintf(stderr, "[ERROR] rank %u: HcclChannelAcquire failed: %d\n", rank, r); return nullptr; }
    fprintf(stderr, "[INFO] rank %u: [4/6] HcclChannelAcquire done\n", rank);

    const size_t tableSlots = static_cast<size_t>(ctx.numLayers) * 1 * world;
    ctx.channelPtrsHost.resize(tableSlots, 0);

    fprintf(stderr, "[INFO] rank %u: [5/6] resolve channel ptrs\n", rank);
    using GetPtrByHandleFn = HcommResult (*)(const ChannelHandle *, uint32_t, ChannelPtr *);
    auto getPtrByHandle = reinterpret_cast<GetPtrByHandleFn>(dlsym(RTLD_DEFAULT, "HcommChannelGetPtrByHandle"));
    if (getPtrByHandle != nullptr) {
        fprintf(stderr, "[INFO] rank %u: [5/6] using HcommChannelGetPtrByHandle (old ABI)\n", rank);
        std::vector<ChannelPtr> ptrs(ctx.channels.size(), 0);
        HcommResult hr = getPtrByHandle(ctx.channels.data(),
                                        static_cast<uint32_t>(ctx.channels.size()), ptrs.data());
        if (hr != HCOMM_SUCCESS_VALUE) { fprintf(stderr, "[ERROR] rank %u: HcommChannelGetPtrByHandle failed: %d\n", rank, hr); return nullptr; }
        for (size_t i = 0; i < descs.size(); ++i) ctx.channelPtrsHost[descSlots[i]] = ptrs[i];
    } else {
        fprintf(stderr, "[INFO] rank %u: [5/6] using direct handle (new ABI)\n", rank);
        for (size_t i = 0; i < descs.size(); ++i)
            ctx.channelPtrsHost[descSlots[i]] = static_cast<ChannelPtr>(ctx.channels[i]);
    }
    for (size_t i = 0; i < ctx.channelPtrsHost.size(); ++i) {
        fprintf(stderr, "[INFO] rank %u: [5/6] slot[%zu]=0x%llx\n", rank, i,
                (unsigned long long)ctx.channelPtrsHost[i]);
    }

    fprintf(stderr, "[INFO] rank %u: [6/6] alloc table+state GM\n", rank);
    size_t tableBytes = tableSlots * sizeof(ChannelPtr);
    ar = aclrtMalloc(&ctx.channelPtrTableDev, tableBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ar != ACL_SUCCESS) { fprintf(stderr, "[ERROR] rank %u: aclrtMalloc table failed: %d\n", rank, ar); return nullptr; }
    aclrtMemcpy(ctx.channelPtrTableDev, tableBytes, ctx.channelPtrsHost.data(), tableBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    size_t stateBytes = tableSlots * 2 * sizeof(uint32_t);
    ar = aclrtMalloc(&ctx.stateGmDev, stateBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ar != ACL_SUCCESS) { fprintf(stderr, "[ERROR] rank %u: aclrtMalloc state failed: %d\n", rank, ar); return nullptr; }
    aclrtMemset(ctx.stateGmDev, stateBytes, 0, stateBytes);

    fprintf(stderr, "[INFO] rank %u: [6/6] tableDev=%p stateDev=%p\n",
            rank, ctx.channelPtrTableDev, ctx.stateGmDev);
    return &ctx;
}

struct ThreadContext {
    HcclRootInfo *rootInfo;
    uint32_t device;
    uint32_t devCount;
    BenchArgs *args;
};

static std::atomic<uint32_t> g_barrierCount{0};

static void barrier(uint32_t total)
{
    g_barrierCount.fetch_add(1, std::memory_order_acq_rel);
    while (g_barrierCount.load(std::memory_order_acquire) < total) {
        std::this_thread::yield();
    }
}

// Sample: 每个设备一个线程的worker函数
// rank 0 = sender(跑kernel), rank 1 = receiver(sleep等待)
// URMA Write是单向的, 接收方不需要做任何事
int Sample(void *arg)
{
    ThreadContext *tctx = (ThreadContext *)arg;
    uint32_t rank = tctx->device;
    uint32_t devCount = tctx->devCount;
    BenchArgs *args = tctx->args;

    uint32_t physicalDev = (rank < args->devIds.size()) ? args->devIds[rank] : rank;
    uint32_t peerRank;
    if (args->hasPeer) {
        peerRank = args->peerOverride;
    } else {
        peerRank = (rank == 0) ? 1 : 0;
        if (devCount > 2) peerRank = (rank + 1) % devCount;
    }

    aclrtSetDevice(static_cast<int32_t>(physicalDev));
    aclrtContext context;
    aclrtCreateContext(&context, physicalDev);

    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclDeterministic = 1;
    config.hcclBufferSize = 200;
    snprintf(config.hcclCommName, COMM_NAME_MAX_LENGTH - 1, "urma_p2p_bench");

    HcclComm hcclComm;
    HcclResult hr = HcclCommInitRootInfoConfig(devCount, tctx->rootInfo, rank, &config, &hcclComm);
    if (hr != HCCL_SUCCESS) { fprintf(stderr, "[ERROR] rank %u (dev %u): HcclCommInitRootInfo failed: %d\n", rank, physicalDev, hr); return hr; }
    fprintf(stderr, "[INFO] rank %u (dev %u): HCCL comm init success, peer=rank %u\n", rank, physicalDev, peerRank);

    fprintf(stderr, "[INFO] rank %u: setting up context...\n", rank);
    BenchCtx *ctx = setup_ctx(hcclComm, rank, devCount, args->bufBytes);
    if (ctx == nullptr) return -1;
    fprintf(stderr, "[INFO] rank %u: context setup done\n", rank);

    uint32_t slot = peerRank;
    if (slot >= ctx->channelPtrsHost.size() || ctx->channelPtrsHost[slot] == 0) {
        fprintf(stderr, "[ERROR] rank %u (dev %u): no URMA channel to peer rank %u\n", rank, physicalDev, peerRank);
        return -1;
    }
    fprintf(stderr, "[INFO] rank %u: channel OK, slot=%u\n", rank, slot);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    void *timingGm = nullptr;
    aclrtMalloc(&timingGm, TIMING_SLOTS_SIZE * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);

    uint32_t stateStride = (uint32_t)(ctx->numLayers * 1 * devCount);

    UrmaP2pBenchTiling tilingHost{};
    tilingHost.peer         = peerRank;
    tilingHost.numSqe       = args->numSqe;
    tilingHost.sqeBytes     = args->sqeBytes;
    tilingHost.bufBytes     = args->bufBytes;
    tilingHost.dbInterval   = args->dbInterval;
    tilingHost.channelSlot  = slot;
    tilingHost.cqOwnerSlot  = slot;
    tilingHost.stateStride  = stateStride;

    tilingHost.loopCount      = args->loopCount;

    barrier(devCount);
    fprintf(stderr, "[INFO] rank %u: barrier passed, launching kernel (loop=%u)\n", rank, args->loopCount);

    // rank 0 sends, rank 1 waits (P2P one-way benchmark)
    bool isSender = (rank == 0);

    if (isSender) {
        aclrtSynchronizeStream(stream);
        switch (args->sqeMode) {
            case 0:
                sqe_write_bench_launch((uint32_t)BLOCK_NUM, stream,
                    (uint64_t)ctx->channelPtrTableDev, (uint64_t)ctx->stateGmDev,
                    (uint64_t)timingGm, &tilingHost);
                break;
            case 1:
                sqe_write_with_notify_bench_launch((uint32_t)BLOCK_NUM, stream,
                    (uint64_t)ctx->channelPtrTableDev, (uint64_t)ctx->stateGmDev,
                    (uint64_t)timingGm, &tilingHost);
                break;
            case 2:
                hcomm_bench_launch((uint32_t)BLOCK_NUM, stream,
                    (uint64_t)ctx->channelPtrTableDev, (uint64_t)ctx->stateGmDev,
                    (uint64_t)timingGm, &tilingHost);
                break;
            default:
                fprintf(stderr, "[ERROR] invalid sqe_mode %u\n", args->sqeMode);
                return -1;
        }
        aclrtSynchronizeStream(stream);
        fprintf(stderr, "[INFO] rank %u: kernel done\n", rank);

        int64_t timingHost[TIMING_SLOTS_SIZE] = {0};
        aclrtMemcpy(timingHost, sizeof(timingHost), timingGm, sizeof(timingHost), ACL_MEMCPY_DEVICE_TO_HOST);

        // [0-3]=累计值, 除以loopCount得平均值; [4-7]=最后一轮单轮值
        double loopCount = (double)args->loopCount;
        double avgAssembleNs = (double)timingHost[0] * NS_PER_CYCLE / loopCount;
        double avgIssueNs    = (double)timingHost[1] * NS_PER_CYCLE / loopCount;
        double avgWaitNs     = (double)timingHost[2] * NS_PER_CYCLE / loopCount;
        double avgTotalNs    = (double)timingHost[3] * NS_PER_CYCLE / loopCount;

        double lastAssembleNs = (double)timingHost[4] * NS_PER_CYCLE;
        double lastIssueNs    = (double)timingHost[5] * NS_PER_CYCLE;
        double lastWaitNs     = (double)timingHost[6] * NS_PER_CYCLE;
        double lastTotalNs    = (double)timingHost[7] * NS_PER_CYCLE;

        double totalData = (double)args->numSqe * args->sqeBytes;
        double avgBwGBs = totalData / (avgTotalNs);
        double lastBwGBs = totalData / (lastTotalNs);

        printf("[rank %u (dev %u)] avg  assemble=%.2fus issue=%.2fus wait=%.2fus total=%.2fus bw=%.2fGB/s (over %u loops)\n",
               rank, physicalDev,
               avgAssembleNs / 1000.0, avgIssueNs / 1000.0, avgWaitNs / 1000.0,
               avgTotalNs / 1000.0, avgBwGBs, args->loopCount);
        printf("[rank %u (dev %u)] last assemble=%.2fus issue=%.2fus wait=%.2fus total=%.2fus bw=%.2fGB/s (last loop)\n",
               rank, physicalDev,
               lastAssembleNs / 1000.0, lastIssueNs / 1000.0, lastWaitNs / 1000.0,
               lastTotalNs / 1000.0, lastBwGBs);
    } else {
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }

    aclrtFree(timingGm);
    HcclCommDestroy(hcclComm);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(physicalDev);
    return 0;
}

// main: 单进程多线程入口
// 1. 解析命令行参数
// 2. aclInit + HcclGetRootInfo(主线程, 需要先SetDevice绑定context)
// 3. 每设备一个std::thread跑Sample
// 4. join所有线程后释放资源
int main(int argc, char *argv[])
{
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto find_val = [&](const std::string &name) -> std::string {
            if (arg.find(name + "=") == 0) return arg.substr(name.size() + 1);
            if (arg == name && i + 1 < argc) { return argv[++i]; }
            return "";
        };
        std::string val;
        if ((val = find_val("--num_sqe")).size()) g_args.numSqe = (uint32_t)std::stoul(val);
        else if ((val = find_val("--sqe_bytes")).size()) g_args.sqeBytes = std::stoull(val);
        else if ((val = find_val("--buf_bytes")).size()) g_args.bufBytes = std::stoull(val);
        else if ((val = find_val("--db_interval")).size()) g_args.dbInterval = (uint32_t)std::stoul(val);
        else if ((val = find_val("--loop_count")).size()) g_args.loopCount = (uint32_t)std::stoul(val);
        else if ((val = find_val("--sqe_mode")).size()) g_args.sqeMode = (uint32_t)std::stoul(val);
        else if ((val = find_val("--dev_count")).size()) g_args.devCount = (uint32_t)std::stoul(val);
        else if ((val = find_val("--peer")).size()) { g_args.peerOverride = (uint32_t)std::stoul(val); g_args.hasPeer = true; }
        else if ((val = find_val("--dev_ids")).size()) {
            g_args.devIds.clear();
            std::stringstream ss(val);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                g_args.devIds.push_back((uint32_t)std::stoul(tok));
            }
            g_args.devCount = (uint32_t)g_args.devIds.size();
        }
    }

    if (g_args.devIds.empty()) {
        for (uint32_t i = 0; i < g_args.devCount; i++) g_args.devIds.push_back(i);
    }
    g_args.devCount = (uint32_t)g_args.devIds.size();

    fprintf(stderr, "[INFO] ===== URMA P2P Bench =====\n");
    fprintf(stderr, "[INFO] dev_ids=");
    for (auto d : g_args.devIds) fprintf(stderr, "%u ", d);
    fprintf(stderr, "dev_count=%u num_sqe=%u sqe_bytes=%lu buf_bytes=%lu "
            "db_interval=%u loop=%u sqe_mode=%u(%s)",
            g_args.devCount, g_args.numSqe, g_args.sqeBytes, g_args.bufBytes,
            g_args.dbInterval, g_args.loopCount, g_args.sqeMode,
            g_args.sqeMode == 1 ? "WriteWithNotify(mc2)" :
            g_args.sqeMode == 2 ? "Hcomm(WriteNbi+Drain)" : "Write(samples)");
    if (g_args.hasPeer) fprintf(stderr, " peer=rank %u", g_args.peerOverride);
    fprintf(stderr, "\n");

    aclInit(nullptr);

    // HcclGetRootInfo需要当前线程绑定了device context
    aclrtSetDevice(0);
    void *rootInfoBuf = nullptr;
    aclrtMallocHost(&rootInfoBuf, sizeof(HcclRootInfo));
    HcclRootInfo *rootInfo = (HcclRootInfo *)rootInfoBuf;
    HcclGetRootInfo(rootInfo);
    aclrtResetDevice(0);  // 释放主线程的device, 子线程会各自SetDevice

    std::vector<std::thread> threads(g_args.devCount);
    std::vector<ThreadContext> args(g_args.devCount);
    for (uint32_t i = 0; i < g_args.devCount; i++) {
        args[i].rootInfo = rootInfo;
        args[i].device = i;
        args[i].devCount = g_args.devCount;
        args[i].args = &g_args;
        threads[i] = std::thread(Sample, (void *)&args[i]);
    }
    for (uint32_t i = 0; i < g_args.devCount; i++) {
        threads[i].join();
    }

    aclrtFreeHost(rootInfoBuf);
    aclFinalize();
    return 0;
}
