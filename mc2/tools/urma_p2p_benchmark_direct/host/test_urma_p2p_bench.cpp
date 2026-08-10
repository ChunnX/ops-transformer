#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <getopt.h>
#include <dlfcn.h>
#include <vector>
#include <string>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl/hcomm_primitives.h"
#include "hccl/hcomm_res_defs.h"
#include "hccl/hccl_types.h"
#include "urma_p2p_bench.h"

constexpr uint32_t SYS_CYCLE_FREQ_MHZ = 1000U;
constexpr double NS_PER_CYCLE = 1.0;

constexpr int64_t BLOCK_NUM = 1;
constexpr uint64_t DEFAULT_BUF_BYTES = 100 * 1024 * 1024ULL;
constexpr uint32_t DEFAULT_NUM_SQE = 16;
constexpr uint64_t DEFAULT_SQE_BYTES = 65536;
constexpr uint32_t DEFAULT_DB_INTERVAL = 1;
constexpr uint32_t TIMING_SLOTS_SIZE = 8;

constexpr int COMM_PROTOCOL_UBC_CTP_VALUE = 4;
constexpr int COMM_PROTOCOL_UBC_TP_VALUE = 5;

typedef uint64_t ChannelPtr;
typedef int32_t HcommResult;
constexpr HcommResult HCOMM_SUCCESS_VALUE = 0;
extern "C" HcommResult HcommChannelGetPtrByHandle(const ChannelHandle *channelList,
                                                   uint32_t listNum, ChannelPtr *channelPtr);

int32_t g_rankId = 0;
int32_t g_ranksize = 2;
uint32_t g_numSqe = DEFAULT_NUM_SQE;
uint64_t g_sqeBytes = DEFAULT_SQE_BYTES;
uint64_t g_bufBytes = DEFAULT_BUF_BYTES;
uint32_t g_dbInterval = DEFAULT_DB_INTERVAL;
uint32_t g_loopCount = 100;
uint32_t g_warmup = 10;
uint32_t g_sqeMode = 0;  // 0=Write(samples), 1=WriteWithNotify(mc2)

static const struct option kLongOptions[] = {
    {"rank_id", 1, 0, 'a'}, {"ranksize", 1, 0, 'd'},
    {"num_sqe", 1, 0, 'n'}, {"sqe_bytes", 1, 0, 's'},
    {"buf_bytes", 1, 0, 'b'}, {"db_interval", 1, 0, 'k'},
    {"loop_count", 1, 0, 'l'}, {"warmup", 1, 0, 'w'},
    {"sqe_mode", 1, 0, 'm'},
    {0, 0, 0, 0}};

void GetOption(int argc, char **argv)
{
    int optIdx = 0, c = 1;
    while (c != -1) {
        c = getopt_long(argc, argv, "a:d:n:s:b:k:l:w:m:", kLongOptions, &optIdx);
        switch (c) {
            case 'a': g_rankId = atoi(optarg); break;
            case 'd': g_ranksize = atoi(optarg); break;
            case 'n': g_numSqe = (uint32_t)atoi(optarg); break;
            case 's': g_sqeBytes = (uint64_t)atoll(optarg); break;
            case 'b': g_bufBytes = (uint64_t)atoll(optarg); break;
            case 'k': g_dbInterval = (uint32_t)atoi(optarg); break;
            case 'l': g_loopCount = (uint32_t)atoi(optarg); break;
            case 'w': g_warmup = (uint32_t)atoi(optarg); break;
            case 'm': g_sqeMode = (uint32_t)atoi(optarg); break;
            default: break;
        }
    }
}

struct BenchCtx {
    uint32_t rank{0};
    uint32_t worldSize{0};
    void *workspaceDev{nullptr};
    HcclMemHandle workspaceMemHandle{nullptr};
    void *channelPtrTableDev{nullptr};
    void *stateGmDev{nullptr};
    std::vector<ChannelHandle> channels;
    std::vector<ChannelPtr> channelPtrsHost;
    uint32_t numLayers{1};
};

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

static BenchCtx *setup_ctx(HcclComm comm, uint32_t rank, uint32_t world, uint64_t bufBytes)
{
    static BenchCtx ctx;
    ctx.rank = rank;
    ctx.worldSize = world;

    const std::string tag = "urma_p2p_bench";
    aclError ar = aclrtMalloc(&ctx.workspaceDev, bufBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ar != ACL_SUCCESS) { fprintf(stderr, "[ERROR] aclrtMalloc workspace failed: %d\n", ar); return nullptr; }
    aclrtMemset(ctx.workspaceDev, bufBytes, 0, bufBytes);

    CommMem mem;
    mem.type = COMM_MEM_TYPE_DEVICE;
    mem.addr = ctx.workspaceDev;
    mem.size = bufBytes;
    HcclResult r = HcclCommMemReg(comm, tag.c_str(), &mem, &ctx.workspaceMemHandle);
    if (r != HCCL_SUCCESS) { fprintf(stderr, "[ERROR] HcclCommMemReg failed: %d\n", r); return nullptr; }

    std::vector<HcclChannelDesc> descs;
    std::vector<size_t> descSlots;
    r = build_channel_descs(comm, rank, world, &ctx.workspaceMemHandle, descs, descSlots, ctx.numLayers);
    if (r != HCCL_SUCCESS) { fprintf(stderr, "[ERROR] build_channel_descs failed: %d\n", r); return nullptr; }

    ctx.channels.resize(descs.size(), 0);
    r = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AIV,
                           descs.data(), static_cast<uint32_t>(descs.size()),
                           ctx.channels.data());
    if (r != HCCL_SUCCESS) { fprintf(stderr, "[ERROR] HcclChannelAcquire failed: %d\n", r); return nullptr; }

    const size_t tableSlots = static_cast<size_t>(ctx.numLayers) * 1 * world;
    ctx.channelPtrsHost.resize(tableSlots, 0);

    using GetPtrByHandleFn = HcommResult (*)(const ChannelHandle *, uint32_t, ChannelPtr *);
    auto getPtrByHandle = reinterpret_cast<GetPtrByHandleFn>(dlsym(RTLD_DEFAULT, "HcommChannelGetPtrByHandle"));
    if (getPtrByHandle != nullptr) {
        std::vector<ChannelPtr> ptrs(ctx.channels.size(), 0);
        HcommResult hr = getPtrByHandle(ctx.channels.data(),
                                        static_cast<uint32_t>(ctx.channels.size()), ptrs.data());
        if (hr != HCOMM_SUCCESS_VALUE) { fprintf(stderr, "[ERROR] HcommChannelGetPtrByHandle failed: %d\n", hr); return nullptr; }
        for (size_t i = 0; i < descs.size(); ++i) ctx.channelPtrsHost[descSlots[i]] = ptrs[i];
    } else {
        for (size_t i = 0; i < descs.size(); ++i)
            ctx.channelPtrsHost[descSlots[i]] = static_cast<ChannelPtr>(ctx.channels[i]);
    }

    size_t tableBytes = tableSlots * sizeof(ChannelPtr);
    ar = aclrtMalloc(&ctx.channelPtrTableDev, tableBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ar != ACL_SUCCESS) { fprintf(stderr, "[ERROR] aclrtMalloc table failed: %d\n", ar); return nullptr; }
    aclrtMemcpy(ctx.channelPtrTableDev, tableBytes, ctx.channelPtrsHost.data(), tableBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    size_t stateBytes = tableSlots * 2 * sizeof(uint32_t);
    ar = aclrtMalloc(&ctx.stateGmDev, stateBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ar != ACL_SUCCESS) { fprintf(stderr, "[ERROR] aclrtMalloc state failed: %d\n", ar); return nullptr; }
    aclrtMemset(ctx.stateGmDev, stateBytes, 0, stateBytes);

    fprintf(stderr, "[INFO] rank=%u world=%u numLayers=%u tableSlots=%zu descs=%zu\n",
            rank, world, ctx.numLayers, tableSlots, descs.size());
    for (size_t i = 0; i < descSlots.size(); ++i) {
        fprintf(stderr, "[INFO] slot=%zu peer=%u channelPtr=0x%llx\n",
                descSlots[i], descs[i].remoteRank,
                (unsigned long long)ctx.channelPtrsHost[descSlots[i]]);
    }
    return &ctx;
}

int main(int argc, char *argv[])
{
    GetOption(argc, argv);

    uint32_t peer = (g_rankId == 0) ? 1 : 0;
    if (g_ranksize > 2) peer = (g_rankId + 1) % g_ranksize;

    fprintf(stderr, "[INFO] ===== URMA P2P Bench =====\n");
    fprintf(stderr, "[INFO] rank=%d ranksize=%d peer=%u num_sqe=%u sqe_bytes=%lu "
            "buf_bytes=%lu db_interval=%u loop=%u warmup=%u sqe_mode=%u(%s)\n",
            g_rankId, g_ranksize, peer, g_numSqe, g_sqeBytes,
            g_bufBytes, g_dbInterval, g_loopCount, g_warmup, g_sqeMode,
            g_sqeMode == 1 ? "WriteWithNotify(mc2)" :
            g_sqeMode == 2 ? "Hcomm(WriteNbi+Drain)" : "Write(samples)");

    int ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) { fprintf(stderr, "[ERROR] aclInit: %d\n", ret); return ret; }
    ret = aclrtSetDevice(g_rankId);
    if (ret != ACL_SUCCESS) { fprintf(stderr, "[ERROR] aclrtSetDevice: %d\n", ret); return ret; }

    aclrtContext context;
    aclrtCreateContext(&context, g_rankId);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclDeterministic = 1;
    config.hcclBufferSize = 200;
    strcpy_s(config.hcclCommName, COMM_NAME_MAX_LENGTH - 1, "urma_p2p_bench");

    const char *rankTableFile = getenv("RANK_TABLE_FILE");
    if (rankTableFile == nullptr) { fprintf(stderr, "[ERROR] RANK_TABLE_FILE not set\n"); return -1; }

    HcclComm comms;
    ret = HcclCommInitClusterInfoConfig(rankTableFile, g_rankId, &config, &comms);
    if (ret != HCCL_SUCCESS) { fprintf(stderr, "[ERROR] HcclCommInitClusterInfoConfig: %d\n", ret); return ret; }
    fprintf(stderr, "[INFO] HCCL comm init success\n");
    aclrtSetCurrentContext(context);

    BenchCtx *ctx = setup_ctx(comms, (uint32_t)g_rankId, (uint32_t)g_ranksize, g_bufBytes);
    if (ctx == nullptr) return -1;

    uint32_t slot = peer;
    if (slot >= ctx->channelPtrsHost.size() || ctx->channelPtrsHost[slot] == 0) {
        fprintf(stderr, "[ERROR] no URMA channel to peer %u\n", peer);
        return -1;
    }

    void *timingGm = nullptr;
    aclrtMalloc(&timingGm, TIMING_SLOTS_SIZE * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);

    uint32_t stateStride = (uint32_t)(ctx->numLayers * 1 * g_ranksize);

    UrmaP2pBenchTiling tilingHost{};
    tilingHost.rankId       = (uint32_t)g_rankId;
    tilingHost.peer         = peer;
    tilingHost.numSqe       = g_numSqe;
    tilingHost.sqeBytes     = g_sqeBytes;
    tilingHost.bufBytes     = g_bufBytes;
    tilingHost.dbInterval   = g_dbInterval;
    tilingHost.channelSlot  = slot;
    tilingHost.cqOwnerSlot  = slot;
    tilingHost.stateStride  = stateStride;

    for (uint32_t iter = 0; iter < g_warmup + g_loopCount; ++iter) {
        aclrtSynchronizeStream(stream);
        switch (g_sqeMode) {
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
                fprintf(stderr, "[ERROR] invalid sqe_mode %u\n", g_sqeMode);
                return -1;
        }
        aclrtSynchronizeStream(stream);

        if (iter < g_warmup) continue;

        int64_t timingHost[TIMING_SLOTS_SIZE] = {0};
        aclrtMemcpy(timingHost, sizeof(timingHost), timingGm, sizeof(timingHost), ACL_MEMCPY_DEVICE_TO_HOST);

        int64_t assembleCycles = timingHost[1] - timingHost[0];
        int64_t issueCycles    = timingHost[2] - timingHost[0];
        int64_t waitCycles     = timingHost[3] - timingHost[2];
        int64_t totalCycles    = timingHost[3] - timingHost[0];

        double totalNs   = (double)totalCycles * NS_PER_CYCLE;
        double totalUs   = totalNs / 1000.0;
        double totalData = (double)g_numSqe * g_sqeBytes;
        double bwGBs     = totalData / (totalNs);

        printf("[%u] assemble=%.2fus issue=%.2fus wait=%.2fus total=%.2fus bw=%.2fGB/s\n",
               iter,
               (double)assembleCycles * NS_PER_CYCLE / 1000.0,
               (double)issueCycles * NS_PER_CYCLE / 1000.0,
               (double)waitCycles * NS_PER_CYCLE / 1000.0,
               totalUs, bwGBs);
    }

    aclrtFree(timingGm);
    HcclCommDestroy(comms);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(g_rankId);
    aclFinalize();
    return 0;
}
