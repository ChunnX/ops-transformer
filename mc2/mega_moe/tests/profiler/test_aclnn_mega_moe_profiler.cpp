// ⚠️ mega_moe 子包路径 profiler 测试程序 — 两阶段（标定 + 打点）
//
// 编译：
//   标定阶段:  g++ -DENABLE_PROFILING -DPROF_ADDR_CALIB test_aclnn_mega_moe_profiler.cpp ... -o prof
//   打点阶段:  g++ -DENABLE_PROFILING test_aclnn_mega_moe_profiler.cpp ... -o prof   （先填 WS_ADDR_DELTA）
//   链接库/头文件路径见子包路径 README（mc2/common op_api、vendor set_env.bash 等）。
//
// 参考：ops-kernel-profiler references/subpackage/subpackage-test-skeleton.cpp
//      + all_gather_add examples/test_aclnn_all_gather_add.cpp（MC2 context 通过框架 Mc2Context 构造）

#include <thread>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <cstdint>
#include "hccl/hccl.h"
#include "aclnn/aclnn_base.h"
#include "../../ut/op_api/aclnn_mega_moe.h"

#ifdef ENABLE_PROFILING
#include "profiler_config.h"
#endif

#include "../../../common/op_api/mc2_context.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            LOG_PRINT("[ERROR] %s:%d CHECK failed\n", __FILE__, __LINE__);                                             \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

// ═══════════════════════════════════════════════════════════════════════
// [MODIFY] 运行时配置
// ═══════════════════════════════════════════════════════════════════════
constexpr int DEV_NUM = 4;       // [MODIFY] rank 数（与 epWorldSize 一致）
constexpr int64_t BS = 128;      // [MODIFY] token 数（x.dim0）
constexpr int64_t H = 4096;      // [MODIFY] hidden 维（x.dim1）
constexpr int64_t TOPK = 8;      // [MODIFY] topK
constexpr int64_t MOE_EXPERT_NUM = 16;   // [MODIFY] 全局专家数 = epWorldSize * expertPerRank
constexpr int64_t EP_WORLD_SIZE = DEV_NUM;
constexpr int64_t EXPERT_PER_RANK = MOE_EXPERT_NUM / EP_WORLD_SIZE; // 4
constexpr int64_t CCL_BUFFER_SIZE = 2097152; // [MODIFY] 需 >= 通信所需最小 buffer（README 有计算）
constexpr int64_t INTERMEDIATE = H; // weight1 的 N（2*intermediate 或 intermediate 视实现）

const char *GROUP_EP = "mega_moe_prof_group"; // [MODIFY] EP 通信域 group 名
// ⚠️ 运行前需保证 GROUP_EP 对应的 HCCL EP 通信域已创建（否则 GetMc2ContextTensor 内部
//    HcclCommGetHandleWithName 失败）。参考框架 op_graph 的 group 创建方式（HcclCreateGroup /
//    HcclCommInitAll + 子组），或复用已有 EP group 名。

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

struct Args {
    int rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
};

// 生成 expert-per-rank 个 weight1/weight2 TensorList（与 UT test_aclnn_mega_moe 形状一致）。
// weight1: [EXPERT_PER_RANK × (H × 2*INTERMEDIATE)]  fp8; weight2: [EXPERT_PER_RANK × (INTERMEDIATE × H)] fp8
// 仅用于打点（不校验结果），用全 0 数据即可让 kernel 正确跑完流水。
struct WeightLists {
    std::vector<aclTensor *> weight1;
    std::vector<aclTensor *> weight2;
    std::vector<void *> weight1Addr;
    std::vector<void *> weight2Addr;
    aclTensorList *w1List = nullptr;
    aclTensorList *w2List = nullptr;
};

int CreateWeightLists(WeightLists &wl, aclDataType dtype)
{
    std::vector<int64_t> w1Shape = {H, 2 * INTERMEDIATE};
    std::vector<int64_t> w2Shape = {INTERMEDIATE, H};
    for (int e = 0; e < EXPERT_PER_RANK; ++e) {
        std::vector<int8_t> w1Host(GetShapeSize(w1Shape), 0);
        std::vector<int8_t> w2Host(GetShapeSize(w2Shape), 0);
        aclTensor *t1 = nullptr, *t2 = nullptr;
        void *d1 = nullptr, *d2 = nullptr;
        CHECK_RET(CreateAclTensor(w1Host, w1Shape, &d1, dtype, &t1) == 0, return -1);
        CHECK_RET(CreateAclTensor(w2Host, w2Shape, &d2, dtype, &t2) == 0, return -1);
        wl.weight1.push_back(t1);
        wl.weight2.push_back(t2);
        wl.weight1Addr.push_back(d1);
        wl.weight2Addr.push_back(d2);
    }
    wl.w1List = aclCreateTensorList(wl.weight1.data(), wl.weight1.size());
    wl.w2List = aclCreateTensorList(wl.weight2.data(), wl.weight2.size());
    CHECK_RET(wl.w1List != nullptr && wl.w2List != nullptr, return -1);
    return 0;
}

int LaunchOneRank(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret = %d\n", ret); return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s\n", args.rankId, hcomName);

    // ── MC2 context tensor：复用框架 Mc2Context 构造（HcclEngineCtxCreate + URMA hcomm handle）──
    uint64_t hcclBuffSize = 0;
    aclTensor *contextTensor = nullptr;
    ret = Mc2Aclnn::Mc2Context::GetMc2ContextTensor(GROUP_EP, "MegaMoe", hcclBuffSize, contextTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] GetMc2ContextTensor failed. ret = %d\n", ret); return ret);
    LOG_PRINT("[INFO] rank=%d hcclBuffSize=%llu\n", args.rankId, (unsigned long long)hcclBuffSize);

    // ── 输入/输出 tensor ──
    std::vector<int64_t> xShape = {BS, H};
    std::vector<int64_t> topkShape = {BS, TOPK};
    std::vector<int64_t> yShape = {BS, H};
    std::vector<int64_t> tokenNumShape = {EXPERT_PER_RANK};

    std::vector<int16_t> xHost(GetShapeSize(xShape), 0);          // BF16
    std::vector<int32_t> topkHost(GetShapeSize(topkShape), 0);    // 全 0 → 所有 token 路由到 expert 0
    std::vector<float> topkWHost(GetShapeSize(topkShape), 0);
    std::vector<int16_t> yHost(GetShapeSize(yShape), 0);
    std::vector<int32_t> tokenNumHost(GetShapeSize(tokenNumShape), 0);

    void *xDev = nullptr, *topkDev = nullptr, *topkWDev = nullptr, *yDev = nullptr, *tokenNumDev = nullptr;
    aclTensor *x = nullptr, *topk = nullptr, *topkW = nullptr, *y = nullptr, *tokenNum = nullptr;
    CHECK_RET(CreateAclTensor(xHost, xShape, &xDev, aclDataType::ACL_BF16, &x) == 0, return -1);
    CHECK_RET(CreateAclTensor(topkHost, topkShape, &topkDev, aclDataType::ACL_INT32, &topk) == 0, return -1);
    CHECK_RET(CreateAclTensor(topkWHost, topkShape, &topkWDev, aclDataType::ACL_FLOAT, &topkW) == 0, return -1);
    CHECK_RET(CreateAclTensor(yHost, yShape, &yDev, aclDataType::ACL_BF16, &y) == 0, return -1);
    CHECK_RET(CreateAclTensor(tokenNumHost, tokenNumShape, &tokenNumDev, aclDataType::ACL_INT32, &tokenNum) == 0,
              return -1);

    WeightLists wl;
    CHECK_RET(CreateWeightLists(wl, aclDataType::ACL_FLOAT8_E4M3FN) == 0, return -1);

    // ── Phase 1: GetWorkspaceSize ──
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    // topoType=1 (URMA) + dispatchQuantMode=4 (MXFP) → 命中 mega_moe_layered.h 路径
    ret = aclnnMegaMoeGetWorkspaceSize(
        contextTensor, x, topk, topkW, wl.w1List, wl.w2List, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, MOE_EXPERT_NUM, EP_WORLD_SIZE, CCL_BUFFER_SIZE, 0, 4, 4, 0, "", 0,
        "swiglu", std::numeric_limits<float>::max(), 1, 2, 0, y, tokenNum, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMegaMoeGetWorkspaceSize failed. ret = %d\n", ret); return ret);
    LOG_PRINT("[INFO] rank=%d workspaceSize=%llu\n", args.rankId, (unsigned long long)workspaceSize);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d\n", ret); return ret);
    }

    // ── Phase 2: 首次执行（同步）──
    ret = aclnnMegaMoe(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMegaMoe failed. ret = %d\n", ret); return ret);
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 30000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret);
              return ret);
    LOG_PRINT("[INFO] device_%d aclnnMegaMoe execute successfully.\n", args.rankId);

#ifdef ENABLE_PROFILING
    constexpr int WARMUP_ROUNDS = 5;
    constexpr int PROFILING_ROUNDS = 5;
    // [MODIFY] PROF_N_CORES: 由 profiler_config.h 提供（Ascend 950 MIX_1_2 = 84）
    constexpr uint32_t PROF_N_CORES = ::PROF_N_CORES;
    constexpr size_t PROF_BUF_TOTAL = (size_t)PROF_N_CORES * 64 + (size_t)PROF_N_CORES * 4096 * 16;

    // Warmup + profiling rounds（禁止 aclrtMemset 清 profBuf 区域）
    for (int i = 0; i < WARMUP_ROUNDS; i++) {
        ret = aclnnMegaMoe(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] warmup aclnnMegaMoe failed. ret = %d\n", ret); return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 30000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] warmup sync failed. ret = %d\n", ret); return ret);
    }
    for (int i = 0; i < PROFILING_ROUNDS; i++) {
        ret = aclnnMegaMoe(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] profiling aclnnMegaMoe failed. ret = %d\n", ret); return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 30000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] profiling sync failed. ret = %d\n", ret); return ret);
    }

#ifdef PROF_ADDR_CALIB
    // ══════════════════════════════════════════════════════════════════
    // 标定阶段：kernel（-DPROF_ADDR_CALIB 编译）把 workspace 基址写到 yOut[0..8]，
    // host 读出后计算 delta = kernelWsBase - hostWsBase。
    // ══════════════════════════════════════════════════════════════════
    {
        uint64_t kernelWsBase = 0;
        ret = aclrtMemcpy(&kernelWsBase, 8, yDev, 8, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] bridge D2H failed. ret = %d\n", ret); return ret);
        uint64_t hostWsBase = reinterpret_cast<uint64_t>(workspaceAddr);
        uint64_t delta = kernelWsBase - hostWsBase;
        LOG_PRINT("[CALIB] rank %u: kernelWsBase=0x%016llX hostWsBase=0x%016llX delta=%llu (0x%llX)\n",
                  args.rankId, (unsigned long long)kernelWsBase, (unsigned long long)hostWsBase,
                  (unsigned long long)delta, (unsigned long long)delta);
        LOG_PRINT("[CALIB] Fill WS_ADDR_DELTA = %lluULL into profiling build, recompile without PROF_ADDR_CALIB\n",
                  (unsigned long long)delta);
    }
#else
    // ══════════════════════════════════════════════════════════════════
    // 打点阶段：从 workspace 尾部读 profBuf（用标定得到的 delta）
    // ══════════════════════════════════════════════════════════════════
    {
        // [MODIFY] 标定阶段输出的 delta 值填入此处（每次换环境/算子类型需重新标定）
        constexpr uint64_t WS_ADDR_DELTA = 0ULL; // 16MB = 16777216ULL for Ascend 950 MC2

        uint64_t profBufOff = workspaceSize - PROF_BUF_TOTAL;
        uint8_t *profDevPtr = reinterpret_cast<uint8_t *>(workspaceAddr) + WS_ADDR_DELTA + profBufOff;

        std::vector<uint8_t> hostBuf(PROF_BUF_TOTAL);
        ret = aclrtMemcpy(hostBuf.data(), PROF_BUF_TOTAL, profDevPtr, PROF_BUF_TOTAL, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] profBuf D2H failed. ret = %d\n", ret); return ret);

        char fname[256];
        snprintf(fname, sizeof(fname), "prof_rank%u.bin", args.rankId);
        std::ofstream ofs(fname, std::ios::binary);
        // layout_type: 3 = MIX_1_2（mega_moe KERNEL_TYPE_MIX_AIC_1_2）
        uint32_t header[4] = {0x50524F46u, (3u << 16) | 1u, PROF_N_CORES, 16u};
        ofs.write(reinterpret_cast<const char *>(header), sizeof(header));
        ofs.write(reinterpret_cast<const char *>(hostBuf.data()), PROF_BUF_TOTAL);
        ofs.close();

        uint32_t firstWord = 0;
        memcpy(&firstWord, hostBuf.data(), 4);
        LOG_PRINT("[PROFILER] rank %u dumped to %s (first word: 0x%08X)\n", args.rankId, fname, firstWord);
    }
#endif // PROF_ADDR_CALIB
#endif // ENABLE_PROFILING

    // ── Cleanup ──
    aclDestroyTensor(contextTensor);
    aclDestroyTensor(x);
    aclDestroyTensor(topk);
    aclDestroyTensor(topkW);
    aclDestroyTensor(y);
    aclDestroyTensor(tokenNum);
    aclDestroyTensorList(wl.w1List);
    aclDestroyTensorList(wl.w2List);
    for (auto t : wl.weight1) aclDestroyTensor(t);
    for (auto t : wl.weight2) aclDestroyTensor(t);
    for (auto d : wl.weight1Addr) aclrtFree(d);
    for (auto d : wl.weight2Addr) aclrtFree(d);
    aclrtFree(xDev);
    aclrtFree(topkDev);
    aclrtFree(topkWDev);
    aclrtFree(yDev);
    aclrtFree(tokenNumDev);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    HcclCommDestroy(args.hcclComm);
    aclrtDestroyStream(args.stream);
    aclrtResetDevice(args.rankId);
    aclrtDestroyContext(args.context);
    return 0;
}

int main(int argc, char *argv[])
{
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);

    aclrtStream stream[DEV_NUM];
    aclrtContext context[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d\n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d\n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
    }

    int32_t devices[DEV_NUM];
    for (int i = 0; i < DEV_NUM; i++) {
        devices[i] = i;
    }
    HcclComm comms[DEV_NUM];
    ret = HcclCommInitAll(DEV_NUM, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d\n", ret); return ret);

    Args args[DEV_NUM];
    std::vector<std::thread> threads;
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].stream = stream[rankId];
        args[rankId].context = context[rankId];
        threads.emplace_back(std::thread([&]() { LaunchOneRank(std::ref(args[rankId])); }));
    }
    for (auto &t : threads) {
        t.join();
    }

    aclFinalize();
    return 0;
}
