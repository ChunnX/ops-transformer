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
#include "../kernel/mc2_native/mc2_native_launch.h"
#include "../kernel/common/urma_p2p_bench_tiling.h"
#include "../kernel/mc2_native/moe_distribute_base_native.h"
#include "../kernel/common/aiv_urma_channel_decode.h"

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
    uint32_t benchMode = 0;  // 0=多次Write一次Drain, 1=一次Write一次Drain循环
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
    // mc2_native: HcclCombinOpParam 在 device 上的 GM 地址
    void *hcclContextDev{nullptr};
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

    fprintf(stderr, "[INFO] rank %u: build_channel_descs: netLayerNum=%u\n", self, netLayerNum);
    for (uint32_t li = 0; li < netLayerNum; ++li) {
        fprintf(stderr, "[INFO] rank %u:   layer[%u]=%u\n", self, li, netLayers[li]);
    }

    for (uint32_t layer = 0; layer < netLayerNum; ++layer) {
        for (uint32_t peer = 0; peer < world; ++peer) {
            if (peer == self) continue;
            CommLink *linkList = nullptr;
            uint32_t listSize = 0;
            r = HcclRankGraphGetLinks(comm, netLayers[layer], self, peer, &linkList, &listSize);
            if (r != HCCL_SUCCESS) {
                fprintf(stderr, "[INFO] rank %u:   layer=%u peer=%u: HcclRankGraphGetLinks failed=%d\n", self, layer, peer, r);
                continue;
            }

            fprintf(stderr, "[INFO] rank %u:   layer=%u peer=%u: total_links=%u\n", self, layer, peer, listSize);
            for (uint32_t li2 = 0; li2 < listSize; ++li2) {
                int proto = static_cast<int>(linkList[li2].linkAttr.linkProtocol);
                fprintf(stderr, "[INFO] rank %u:     link[%u]: protocol=%d(%s) srcAddr=0x%x dstAddr=0x%x\n",
                        self, li2, proto,
                        (proto == 4) ? "UBC_CTP" : (proto == 5) ? "UBC_TP" : "other",
                        linkList[li2].srcEndpointDesc.commAddr,
                        linkList[li2].dstEndpointDesc.commAddr);
            }

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
                fprintf(stderr, "[INFO] rank %u:   → created channel desc for layer=%u peer=%u (slot=%zu, protocol=%s)\n",
                        self, layer, peer, (static_cast<size_t>(layer) * 1 + 0) * world + peer,
                        (p == 4) ? "UBC_CTP" : "UBC_TP");
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

    // [1/6] 分配 workspace: URMA 通信的目标缓冲区
    // aclrtMalloc 申请 device 显存, 对端 URMA Write 的数据会写入这块内存
    const std::string tag = "urma_p2p_bench_" + std::to_string(rank);
    fprintf(stderr, "[INFO] rank %u: [1/6] aclrtMalloc workspace (%lu bytes)\n", rank, bufBytes);
    aclError ar = aclrtMalloc(&ctx.workspaceDev, bufBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ar != ACL_SUCCESS) { fprintf(stderr, "[ERROR] rank %u: aclrtMalloc workspace failed: %d\n", rank, ar); return nullptr; }
    aclrtMemset(ctx.workspaceDev, bufBytes, 0, bufBytes);
    fprintf(stderr, "[INFO] rank %u: [1/6] workspace=%p\n", rank, ctx.workspaceDev);

    // [2/6] 注册 workspace 给 HCCL: 使对端能通过 URMA 访问这块内存
    // aclrtMalloc 只是普通显存, 对端看不到; HcclCommMemReg 注册后生成 memHandle,
    // 后续 HcclChannelAcquire 时把这个 handle 附到 channel desc 上,
    // HCCL 跨 rank 交换后, 对端的 ChannelEntity.remoteBufferAddr 就指向了这块内存
    fprintf(stderr, "[INFO] rank %u: [2/6] HcclCommMemReg\n", rank);
    CommMem mem;
    mem.type = COMM_MEM_TYPE_DEVICE;
    mem.addr = ctx.workspaceDev;
    mem.size = bufBytes;
    HcclResult r = HcclCommMemReg(comm, tag.c_str(), &mem, &ctx.workspaceMemHandle);
    if (r != HCCL_SUCCESS) { fprintf(stderr, "[ERROR] rank %u: HcclCommMemReg failed: %d\n", rank, r); return nullptr; }
    fprintf(stderr, "[INFO] rank %u: [2/6] memHandle=%p\n", rank, ctx.workspaceMemHandle);

    // [3/6] 枚举 UBC_CTP/UBC_TP 链路, 为每个 peer 构建一个 HcclChannelDesc
    // 每个 desc 附带 workspace 的 memHandle, 使对端 URMA 写入能落到本端 workspace
    fprintf(stderr, "[INFO] rank %u: [3/6] build_channel_descs\n", rank);
    std::vector<HcclChannelDesc> descs;
    std::vector<size_t> descSlots;
    r = build_channel_descs(comm, rank, world, &ctx.workspaceMemHandle, descs, descSlots, ctx.numLayers);
    if (r != HCCL_SUCCESS) { fprintf(stderr, "[ERROR] rank %u: build_channel_descs failed: %d\n", rank, r); return nullptr; }
    fprintf(stderr, "[INFO] rank %u: [3/6] descs=%zu layers=%u\n", rank, descs.size(), ctx.numLayers);

    // [4/6] HcclChannelAcquire: 跨 rank 握手, 获取 ChannelHandle
    // HCCL 内部完成跨 rank 的 buffer/endpoint 交换, 返回的 handle 指向 device 上的 ChannelEntity
    // 两个 rank 必须同时到达此步骤才能握手成功
    fprintf(stderr, "[INFO] rank %u: [4/6] HcclChannelAcquire (waiting for peer...)\n", rank);
    ctx.channels.resize(descs.size(), 0);
    r = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AIV,
                           descs.data(), static_cast<uint32_t>(descs.size()),
                           ctx.channels.data());
    if (r != HCCL_SUCCESS) { fprintf(stderr, "[ERROR] rank %u: HcclChannelAcquire failed: %d\n", rank, r); return nullptr; }
    fprintf(stderr, "[INFO] rank %u: [4/6] HcclChannelAcquire done\n", rank);

    // 构建 channelPtrTable: 按 [layer][channel][peer] 扁平布局的 ChannelEntity VA 数组
    // 下标 = (layer * numChannels + channel) * worldSize + peer
    // 当前 numChannels=1, 所以简化为 layer * worldSize + peer
    // 2P 单 layer 场景: tableSlots=2, slot[0]=peer0(自己,通常0), slot[1]=peer1(对端)
    // 后续拷到 device GM, kernel 通过 channelPtrTable_.GetValue(slot) 读取 ChannelEntity 地址
    const size_t tableSlots = static_cast<size_t>(ctx.numLayers) * 1 * world;
    ctx.channelPtrsHost.resize(tableSlots, 0);

    // [5/6] 将 ChannelHandle 翻译为 device VA (ChannelEntity 的实际地址)
    // 旧 ABI: handle 是 opaque, 需要 HcommChannelGetPtrByHandle 翻译 (dlsym 动态查找)
    // 新 ABI (CANN 9.1.0+): handle 本身就是 device VA, 直接用
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

    // [6/6] 将 channelPtrTable 和 state 持久化到 device GM
    // channelPtrTableDev: ChannelEntity VA 数组, kernel 通过下标读取通信上下文
    // stateGmDev: SQ/CQ 环形索引的持久化状态 (sqPiLinear/cqCiLinear, 仅 -m 0/1/2 用)
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

// ============================================================================
// build_hccl_context: 为 mc2_native 模式构造 HcclCombinOpParam
//
// 从 device 读取 ChannelEntity (通过 aclrtMemcpy), 解码出 SQ/CQ/buffer/token,
// 填入 HcclAiRMAWQ/HcclAiRMACQ 结构体, 设置 windowsIn/windowsOut + PICI 窗口,
// 拷到 device 供 kernel 使用。
//
// 字段映射 (DecodedChannel → HcclAiRMAWQ):
//   sqVA       ← sqVa          dbAddr     ← sqDbVa
//   wqeSize    ← wqeSize       sqDepth    ← sqDepth
//   tp_id      ← tpID          rmtEid     ← remoteEID[16]
//   rmtObjId   ← remoteTokenId rmtTokenValue ← remoteTokenValue
//   localTokenId ← localTokenId
// ============================================================================
static void *build_hccl_context(BenchCtx *ctx, uint32_t rank, uint32_t world, uint64_t bufBytes)
{
    using namespace CommWithUrmaImpl;

    // 读取 ChannelEntity + SqContext + CqContext + RegedBuffer 到 host
    constexpr uint32_t READ_SIZE = CHANNEL_ENTITY_SIZE + SQ_CONTEXT_SIZE + CQ_CONTEXT_SIZE + 2 * REGED_BUFFER_SIZE;
    std::vector<uint8_t> buf(READ_SIZE, 0);

    HcclCombinOpParam param;
    memset(&param, 0, sizeof(param));
    param.rankId = rank;
    param.rankDim = world;
    param.workSpace = (uint64_t)ctx->workspaceDev;
    param.workSpaceSize = bufBytes;

    // windowsOut[myRank]: 本端 workspace 地址 (PICI 窗口在 workspace + 1MB 偏移处)
    // 需要额外分配 PICI 窗口区域: WIN_PICI_OFFSET=1MB, PICI_WIN_SIZE=512B * worldSize
    // 简化: windowsOut[myRank] = workspace 地址, PICI 区域在 workspace + 1MB
    // 但 workspace 的前 350KB 是状态区, 1MB 后才是数据区, PICI 在 1MB 偏移
    // 为避免冲突, 单独分配 PICI GM
    uint64_t piciBytes = PICI_WIN_SIZE * world;
    void *piciDev = nullptr;
    aclrtMalloc(&piciDev, piciBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(piciDev, piciBytes, 0, piciBytes);

    // windowsOut[myRank] 指向 PICI 区域: GetPICI 读 windowsOut[myRank] + WIN_PICI_OFFSET + PICI_WIN_SIZE * dstRankId
    // 所以 windowsOut[myRank] = piciDev - WIN_PICI_OFFSET (这样 +1MB 后指向 piciDev)
    // 但 piciDev 可能不够 1MB 对齐, 直接让 windowsOut = piciDev, WIN_PICI_OFFSET=0
    // 更简单: 重新定义 windowsOut[myRank] = piciDev, 然后在 kernel 侧调整
    // 但 moe_distribute_base_native.h 的 GetPICI 用 windowsOut + WIN_PICI_OFFSET(1MB) + PICI_WIN_SIZE * dst
    // 所以 piciDev 需要预留 1MB 前缀空间
    // 最简单: 分配 1MB + piciBytes, windowsOut = ptr, PICI 从 ptr+1MB 开始
    uint64_t winOutBytes = WIN_PICI_OFFSET + piciBytes;
    void *winOutDev = nullptr;
    aclrtMalloc(&winOutDev, winOutBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(winOutDev, winOutBytes, 0, winOutBytes);
    param.windowsOut[rank] = (uint64_t)winOutDev;

    for (uint32_t peer = 0; peer < world; ++peer) {
        if (peer == rank) continue;

        uint32_t slot = peer;
        if (slot >= ctx->channelPtrsHost.size() || ctx->channelPtrsHost[slot] == 0) {
            fprintf(stderr, "[ERROR] rank %u: no channel for peer %u\n", rank, peer);
            return nullptr;
        }

        // 从 device 读取 ChannelEntity blob
        uint64_t channelEntityVA = ctx->channelPtrsHost[slot];
        aclError ar = aclrtMemcpy(buf.data(), READ_SIZE, (void*)channelEntityVA, READ_SIZE, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ar != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] rank %u: aclrtMemcpy ChannelEntity failed: %d\n", rank, ar);
            return nullptr;
        }

        // 解析 ChannelEntity (256B) → 提取容器指针
        uint32_t *chU32 = reinterpret_cast<uint32_t*>(buf.data());
        uint64_t *chU64 = reinterpret_cast<uint64_t*>(buf.data());
        uint32_t localBufNum  = chU32[CH_OFF_LOCAL_BUFFER_NUM  / sizeof(uint32_t)];
        uint32_t remoteBufNum = chU32[CH_OFF_REMOTE_BUFFER_NUM / sizeof(uint32_t)];
        uint32_t lcIdx = (localBufNum  > 0U) ? (localBufNum  - 1U) : 0U;
        uint32_t rmIdx = (remoteBufNum > 0U) ? (remoteBufNum - 1U) : 0U;

        auto resolvePtr = [](uint64_t base, uint64_t stored) -> uint64_t {
            return (stored < base) ? (base + stored) : stored;
        };
        uint64_t lcBufVA = resolvePtr(channelEntityVA, chU64[CH_OFF_LOCAL_BUFFER_PTR  / sizeof(uint64_t)])
                           + (uint64_t)lcIdx * REGED_BUFFER_SIZE;
        uint64_t rmBufVA = resolvePtr(channelEntityVA, chU64[CH_OFF_REMOTE_BUFFER_PTR / sizeof(uint64_t)])
                           + (uint64_t)rmIdx * REGED_BUFFER_SIZE;
        uint64_t sqCtxVA = resolvePtr(channelEntityVA, chU64[CH_OFF_SQ_CONTEXT_PTR    / sizeof(uint64_t)]);
        uint64_t cqCtxVA = resolvePtr(channelEntityVA, chU64[CH_OFF_CQ_CONTEXT_PTR    / sizeof(uint64_t)]);

        // 读取 SqContext (128B)
        std::vector<uint8_t> sqBuf(SQ_CONTEXT_SIZE, 0);
        ar = aclrtMemcpy(sqBuf.data(), SQ_CONTEXT_SIZE, (void*)sqCtxVA, SQ_CONTEXT_SIZE, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ar != ACL_SUCCESS) { fprintf(stderr, "[ERROR] rank %u: aclrtMemcpy SqContext failed: %d\n", rank, ar); return nullptr; }
        uint64_t *sqU64 = reinterpret_cast<uint64_t*>(sqBuf.data());
        uint32_t *sqU32 = reinterpret_cast<uint32_t*>(sqBuf.data());

        // 读取 CqContext (128B)
        std::vector<uint8_t> cqBuf(CQ_CONTEXT_SIZE, 0);
        ar = aclrtMemcpy(cqBuf.data(), CQ_CONTEXT_SIZE, (void*)cqCtxVA, CQ_CONTEXT_SIZE, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ar != ACL_SUCCESS) { fprintf(stderr, "[ERROR] rank %u: aclrtMemcpy CqContext failed: %d\n", rank, ar); return nullptr; }
        uint64_t *cqU64 = reinterpret_cast<uint64_t*>(cqBuf.data());
        uint32_t *cqU32 = reinterpret_cast<uint32_t*>(cqBuf.data());

        // 读取 remote RegedBuffer (64B)
        std::vector<uint8_t> rmBuf(REGED_BUFFER_SIZE, 0);
        ar = aclrtMemcpy(rmBuf.data(), REGED_BUFFER_SIZE, (void*)rmBufVA, REGED_BUFFER_SIZE, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ar != ACL_SUCCESS) { fprintf(stderr, "[ERROR] rank %u: aclrtMemcpy RegedBuffer(remote) failed: %d\n", rank, ar); return nullptr; }
        uint64_t *rmU64 = reinterpret_cast<uint64_t*>(rmBuf.data());
        uint32_t *rmU32 = reinterpret_cast<uint32_t*>(rmBuf.data());

        // 读取 local RegedBuffer (64B)
        std::vector<uint8_t> lcBuf(REGED_BUFFER_SIZE, 0);
        ar = aclrtMemcpy(lcBuf.data(), REGED_BUFFER_SIZE, (void*)lcBufVA, REGED_BUFFER_SIZE, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ar != ACL_SUCCESS) { fprintf(stderr, "[ERROR] rank %u: aclrtMemcpy RegedBuffer(local) failed: %d\n", rank, ar); return nullptr; }
        uint32_t *lcU32 = reinterpret_cast<uint32_t*>(lcBuf.data());

        // 填充 HcclAiRMAWQ (SQ 信息)
        HcclAiRMAWQ &wq = param.sqs[peer];
        wq.sqVA       = sqU64[SQ_OFF_SQVA       / sizeof(uint64_t)];
        wq.dbAddr     = sqU64[SQ_OFF_DBVA       / sizeof(uint64_t)];
        wq.wqeSize    = sqU32[SQ_OFF_WQESIZE    / sizeof(uint32_t)];
        wq.sqDepth    = sqU32[SQ_OFF_SQDEPTH    / sizeof(uint32_t)];
        wq.tp_id      = sqU32[SQ_OFF_TPID       / sizeof(uint32_t)];
        memcpy(wq.rmtEid, &sqU32[SQ_OFF_REMOTE_EID / sizeof(uint32_t)], 16);
        wq.rmtObjId   = rmU32[RB_OFF_UB_TOKEN_ID  / sizeof(uint32_t)];   // remoteTokenId
        wq.rmtTokenValue = rmU32[RB_OFF_UB_TOKEN_VAL / sizeof(uint32_t)];
        wq.localTokenId = lcU32[RB_OFF_UB_TOKEN_ID  / sizeof(uint32_t)];

        // 填充 HcclAiRMACQ (CQ 信息)
        HcclAiRMACQ &cq = param.cqs[peer];
        cq.cqVA       = cqU64[CQ_OFF_CQVA    / sizeof(uint64_t)];
        cq.cqeSize    = cqU32[CQ_OFF_CQESIZE / sizeof(uint32_t)];
        cq.cqDepth    = cqU32[CQ_OFF_CQDEPTH / sizeof(uint32_t)];
        cq.dbAddr     = cqU64[CQ_OFF_DBVA    / sizeof(uint64_t)];

        // windowsIn[peer] = 对端的 workspace 地址 (remote buffer 的 addr)
        // RegedBuffer.addr = 对端注册的 buffer 起始地址 = 对端 workspace
        param.windowsIn[peer] = rmU64[RB_OFF_ADDR / sizeof(uint64_t)];

        fprintf(stderr, "[INFO] rank %u: built HcclCombinOpParam for peer %u: sqVA=0x%lx dbAddr=0x%lx "
                "sqDepth=%u wqeSize=%u tp_id=%u cqVA=0x%lx cqDepth=%u cqeSize=%u "
                "rmtObjId=%u rmtTokenValue=%u localTokenId=%u windowsIn=0x%lx\n",
                rank, peer,
                (unsigned long)wq.sqVA, (unsigned long)wq.dbAddr, wq.sqDepth, wq.wqeSize, wq.tp_id,
                (unsigned long)cq.cqVA, cq.cqDepth, cq.cqeSize,
                wq.rmtObjId, wq.rmtTokenValue, wq.localTokenId,
                (unsigned long)param.windowsIn[peer]);
    }

    // 拷贝 HcclCombinOpParam 到 device
    void *contextDev = nullptr;
    aclrtMalloc(&contextDev, sizeof(HcclCombinOpParam), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(contextDev, sizeof(HcclCombinOpParam), &param, sizeof(HcclCombinOpParam), ACL_MEMCPY_HOST_TO_DEVICE);

    ctx->hcclContextDev = contextDev;
    fprintf(stderr, "[INFO] rank %u: HcclCombinOpParam built, contextDev=%p (size=%zu)\n",
            rank, contextDev, sizeof(HcclCombinOpParam));
    return contextDev;
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
    tilingHost.benchMode      = args->benchMode;

    barrier(devCount);
    fprintf(stderr, "[INFO] rank %u: barrier passed, launching kernel (loop=%u)\n", rank, args->loopCount);

    // mc2_native 模式需要先构造 HcclCombinOpParam
    if (args->sqeMode == 3) {
        fprintf(stderr, "[INFO] rank %u: building HcclCombinOpParam for mc2_native...\n", rank);
        build_hccl_context(ctx, rank, devCount, args->bufBytes);
    }

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
            case 3:
                mc2_native_bench_launch((uint32_t)BLOCK_NUM, stream,
                    (uint64_t)ctx->hcclContextDev, (uint64_t)ctx->workspaceDev,
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

        // [0-3]=累计值, [4-7]=最后一轮单轮值
        // benchMode=1 (write_drain_loop): 累计值除以loopCount得平均一次Write+Drain时间
        // benchMode=0 (batch_write): [0-3]是总量, 不除以loopCount
        double divisor = (args->benchMode == 1) ? (double)args->loopCount : 1.0;
        double avgAssembleNs = (double)timingHost[0] * NS_PER_CYCLE / divisor;
        double avgIssueNs    = (double)timingHost[1] * NS_PER_CYCLE / divisor;
        double avgWaitNs     = (double)timingHost[2] * NS_PER_CYCLE / divisor;
        double avgTotalNs    = (double)timingHost[3] * NS_PER_CYCLE / divisor;

        double lastAssembleNs = (double)timingHost[4] * NS_PER_CYCLE;
        double lastIssueNs    = (double)timingHost[5] * NS_PER_CYCLE;
        double lastWaitNs     = (double)timingHost[6] * NS_PER_CYCLE;
        double lastTotalNs    = (double)timingHost[7] * NS_PER_CYCLE;

        // 带宽计算: benchMode=0 总数据量=loopCount*sqeBytes; benchMode=1 单次数据量=sqeBytes
        double totalData = (args->benchMode == 0)
                           ? (double)args->loopCount * args->sqeBytes
                           : (double)args->sqeBytes;
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
        else if ((val = find_val("--bench_mode")).size()) g_args.benchMode = (uint32_t)std::stoul(val);
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
            "db_interval=%u loop=%u sqe_mode=%u(%s) bench_mode=%u(%s)",
            g_args.devCount, g_args.numSqe, g_args.sqeBytes, g_args.bufBytes,
            g_args.dbInterval, g_args.loopCount, g_args.sqeMode,
            g_args.sqeMode == 1 ? "WriteWithNotify(mc2)" :
            g_args.sqeMode == 2 ? "Hcomm(WriteNbi+Drain)" :
            g_args.sqeMode == 3 ? "mc2_native(moe_distribute_base)" : "Write(samples)",
            g_args.benchMode, g_args.benchMode == 0 ? "batch_write" : "write_drain_loop");
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
