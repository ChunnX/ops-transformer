/**
 * Standalone host program for quant_reduce_scatter with device-side profiler.
 *
 * Initializes HCCL, builds tiling data, allocates profiler buffer,
 * launches the kernel with warmup protocol, and dumps profiler data.
 *
 * Usage:
 *   ./standalone_profiler --bs 1024 --hidden_size 2048 --rankNum 2
 *
 * Build with ENABLE_PROFILING=1 to activate profiler instrumentation.
 */

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <memory>
#include <fstream>
#include <getopt.h>
#include <unistd.h>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <condition_variable>
#include "hccl/hccl.h"
#include "acl/acl.h"
#include "securec.h"
#include "tiling/hccl/hccl_tiling.h"
#include "quant_reduce_scatter_tiling_data.h"
#ifdef ENABLE_PROFILING
#include "profiler_host_helper.h"
#endif

extern "C" HcclResult HcclCommInitAll(uint32_t commNum, int32_t *devIds, HcclComm *comm);
extern "C" HcclResult HcclCommDestroy(HcclComm comm);
extern "C" HcclResult HcclGetCommName(HcclComm comm, char *commName);
extern "C" HcclResult HcomGetCommHandleByGroup(const char *groupName, void **commHandle);
extern "C" void quant_reduce_scatter_demo(uint32_t blockDim, void* stream,
    uint8_t* x, uint8_t* scales, uint8_t* output,
    uint8_t* workspaceGM, uint8_t* tilingGM);
extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void *stream,
    void *mc2Tiling, void **commContext);

using namespace std;

static constexpr uint64_t AIV_CORE_NUM = 28UL;
constexpr int64_t MAX_RANK_NUM = 8;
constexpr uint32_t GROUP_SIZE = 128;
constexpr int STREAM_TIMEOUT_MS = 30000;

int gBs = 1024;
int gHiddenSize = 2048;
int gRankNum = 2;
bool gPerfMode = false;
constexpr int PERF_LOOP_COUNT = 10;

int64_t gHcclBufferSize = []() {
    const char* envStr = getenv("HCCL_BUFFSIZE");
    int64_t sizeMB = (envStr != nullptr) ? atoll(envStr) : 200;
    return sizeMB * 1024 * 1024;
}();

static std::mutex barrierMtx;
static std::condition_variable barrierCv;
static int barrierCount = 0;

static std::mutex setupMtx;
static std::condition_variable setupCv;
static int setupCount = 0;

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message "\n", ##__VA_ARGS__); \
    } while (0)

#define ACL_CHECK(status)                                                                    \
    do {                                                                                     \
        aclError error = status;                                                             \
        if (error != ACL_ERROR_NONE) {                                                       \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl;  \
        }                                                                                    \
    } while (0)

int CreateTilingDataAndContext(const char* hcomName, aclrtStream stream,
    int bs, int hiddenSize, int rankNum,
    void **deviceTilingAddr, void **deviceContextAddr)
{
    QuantReduceScatterTilingData *tilingData = new QuantReduceScatterTilingData();
    if (tilingData == nullptr) {
        LOG_PRINT("[ERROR] tilingData is nullptr");
        return -1;
    }
    *tilingData = {};

    uint64_t scaleHiddenSize = hiddenSize / GROUP_SIZE;

    tilingData->quantReduceScatterTilingInfo.bs = bs;
    tilingData->quantReduceScatterTilingInfo.hiddenSize = hiddenSize;
    tilingData->quantReduceScatterTilingInfo.scaleHiddenSize = scaleHiddenSize;
    tilingData->quantReduceScatterTilingInfo.aivNum = AIV_CORE_NUM;
    tilingData->quantReduceScatterTilingInfo.totalWinSize = gHcclBufferSize;
    tilingData->quantReduceScatterTilingInfo.xPerBlock = 0;
    tilingData->quantReduceScatterTilingInfo.alignBlock = GROUP_SIZE;
    tilingData->quantReduceScatterTilingInfo.profBufAddr = 0;

    LOG_PRINT("[INFO] Tiling: bs=%d, hiddenSize=%d, scaleHiddenSize=%lu, rankNum=%d, aivNum=%lu, winSize=%ld",
        bs, hiddenSize, scaleHiddenSize, rankNum, AIV_CORE_NUM, gHcclBufferSize);

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(hcomName, 6, "AlltoAll=level0:fullmesh;level1:pairwise");
    mc2CcTilingConfig.SetCommEngine(3);

    int ret = mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] GetTiling mc2InitTiling failed. ret = %d", ret); delete tilingData; return ret);
    ret = mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] GetTiling mc2CcTiling failed. ret = %d", ret); delete tilingData; return ret);

    int tilingSize = sizeof(QuantReduceScatterTilingData);
    ret = aclrtMalloc(deviceTilingAddr, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc TilingData failed. ret: %d", ret); delete tilingData; return ret);
    ret = aclrtMemcpy(*deviceTilingAddr, tilingSize, tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy TilingData failed. ret: %d", ret); delete tilingData; return ret);

    HcclComm commHandle;
    ret = HcomGetCommHandleByGroup(hcomName, &commHandle);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcomGetCommHandleByGroup failed. ret = %d", ret); delete tilingData; return ret);

    void *mc2Context = nullptr;
    ret = HcclAllocComResourceByTiling(commHandle, stream, tilingData, &mc2Context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclAllocComResourceByTiling failed. ret = %d", ret); delete tilingData; return ret);
    if (mc2Context == nullptr) {
        LOG_PRINT("[ERROR] mc2Context is nullptr");
        delete tilingData;
        return -1;
    }

    LOG_PRINT("[INFO] MC2 context allocated (mc2Context=%p)", mc2Context);

    *deviceContextAddr = mc2Context;

    delete tilingData;
    return ACL_SUCCESS;
}

struct Args {
    uint32_t rankId;
    uint32_t rankDim;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
    int bs;
    int hiddenSize;
};

int LaunchOneThread(Args &args) {
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret = %d", ret); return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d", ret); return ret);
    LOG_PRINT("[INFO] rank=%d hcomName=%s", args.rankId, hcomName);

    int bs = args.bs;
    int hiddenSize = args.hiddenSize;
    int rankNum = args.rankDim;
    uint64_t scaleHiddenSize = hiddenSize / GROUP_SIZE;

    void *tilingAddr = nullptr;
    void *mc2ContextAddr = nullptr;

    ret = CreateTilingDataAndContext(hcomName, args.stream, bs, hiddenSize, rankNum, &tilingAddr, &mc2ContextAddr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateTilingDataAndContext failed. ret = %d", ret); return ret);

    size_t sizeX = (size_t)bs * hiddenSize * sizeof(int8_t);
    size_t sizeScales = (size_t)bs * scaleHiddenSize * sizeof(float);
    size_t sizeOutput = (size_t)(bs / rankNum) * hiddenSize * sizeof(uint16_t);

    LOG_PRINT("[INFO] rank=%d Buffer sizes: x=%zu bytes, scales=%zu bytes, output=%zu bytes",
        args.rankId, sizeX, sizeScales, sizeOutput);

    void *deviceX = nullptr, *deviceScales = nullptr, *deviceOutput = nullptr;

    ret = aclrtMalloc(&deviceX, sizeX, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc x failed. ret: %d", ret); return ret);
    ret = aclrtMalloc(&deviceScales, sizeScales, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc scales failed. ret: %d", ret); return ret);
    ret = aclrtMalloc(&deviceOutput, sizeOutput, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc output failed. ret: %d", ret); return ret);

    {
        std::vector<int8_t> hostX(bs * hiddenSize);
        unsigned int seed = 42 + args.rankId;
        for (int i = 0; i < bs * hiddenSize; i++) {
            seed = seed * 1103515245u + 12345u;
            hostX[i] = static_cast<int8_t>((seed >> 16) & 0xFF);
        }
        ret = aclrtMemcpy(deviceX, sizeX, hostX.data(), sizeX, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy x failed. ret: %d", ret); return ret);
    }
    {
        std::vector<float> hostScales(bs * scaleHiddenSize);
        for (size_t i = 0; i < hostScales.size(); i++) {
            hostScales[i] = 0.01f * (float)(i % 128 + 1);
        }
        ret = aclrtMemcpy(deviceScales, sizeScales, hostScales.data(), sizeScales, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy scales failed. ret: %d", ret); return ret);
    }

#ifdef ENABLE_PROFILING
    ProfilerBuffer profBuf;
    ret = ProfilerAlloc(profBuf);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] ProfilerAlloc failed"); return ret);

    {
        QuantReduceScatterTilingData hostTilingData;
        ret = aclrtMemcpy(&hostTilingData, sizeof(hostTilingData), tilingAddr,
                          sizeof(hostTilingData), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy tiling D2H failed. ret: %d", ret); return ret);
        hostTilingData.quantReduceScatterTilingInfo.profBufAddr = (uint64_t)profBuf.devicePtr;
        ret = aclrtMemcpy(tilingAddr, sizeof(hostTilingData), &hostTilingData,
                          sizeof(hostTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy tiling H2D failed. ret: %d", ret); return ret);
    }
#endif

    LOG_PRINT("[INFO] rank=%d Ready to launch kernel: bs=%d, hiddenSize=%d, rankDim=%d, blockDim=%lu",
        args.rankId, bs, hiddenSize, args.rankDim, AIV_CORE_NUM);

    {
        std::unique_lock<std::mutex> lock(setupMtx);
        setupCount++;
        if (setupCount >= static_cast<int>(args.rankDim)) {
            setupCv.notify_all();
        } else {
            setupCv.wait(lock, [&]() { return setupCount >= static_cast<int>(args.rankDim); });
        }
    }

    int loopCount = gPerfMode ? PERF_LOOP_COUNT : 1;
#ifdef ENABLE_PROFILING
    ProfilerWarmup(profBuf, loopCount, [&]() {
        quant_reduce_scatter_demo(AIV_CORE_NUM, args.stream,
            (uint8_t*)deviceX, (uint8_t*)deviceScales, (uint8_t*)deviceOutput,
            (uint8_t*)mc2ContextAddr, (uint8_t*)tilingAddr);
        aclrtSynchronizeStreamWithTimeout(args.stream, STREAM_TIMEOUT_MS);
    });
#else
    for (int i = 0; i < loopCount; i++) {
        quant_reduce_scatter_demo(AIV_CORE_NUM, args.stream,
            (uint8_t*)deviceX, (uint8_t*)deviceScales, (uint8_t*)deviceOutput,
            (uint8_t*)mc2ContextAddr, (uint8_t*)tilingAddr);
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, STREAM_TIMEOUT_MS);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d", ret); return ret);
    }
#endif

    LOG_PRINT("[INFO] rank=%d kernel done", args.rankId);

#ifdef ENABLE_PROFILING
    ProfilerDump(profBuf, args.rankId);
#endif

    std::vector<uint8_t> hostOutput(sizeOutput, 0);
    ret = aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy output failed. ret: %d", ret); return ret);

    {
        char fname[256];
        snprintf(fname, sizeof(fname), "output_rank%u.bin", args.rankId);
        std::ofstream ofs(fname, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(hostOutput.data()), sizeOutput);
        LOG_PRINT("[INFO] rank=%d output dumped to %s (%zu bytes)", args.rankId, fname, sizeOutput);
    }

    aclrtFree(deviceX);
    aclrtFree(deviceScales);
    aclrtFree(deviceOutput);
    aclrtFree(mc2ContextAddr);
    aclrtFree(tilingAddr);
#ifdef ENABLE_PROFILING
    ProfilerFree(profBuf);
#endif

    {
        std::unique_lock<std::mutex> lock(barrierMtx);
        barrierCount++;
        if (barrierCount >= static_cast<int>(args.rankDim)) {
            barrierCv.notify_all();
        } else {
            barrierCv.wait(lock, [&]() { return barrierCount >= static_cast<int>(args.rankDim); });
        }
    }

    HcclCommDestroy(args.hcclComm);

    LOG_PRINT("[INFO] rank=%d resources freed", args.rankId);
    return ACL_SUCCESS;
}

void printUsage(const std::string& programName)
{
    std::cerr << "Usage: " << programName << " --bs BS --hidden_size H --rankNum R" << std::endl;
    std::cerr << "Args: " << std::endl;
    std::cerr << "  --bs:          batch size (default: 1024)" << std::endl;
    std::cerr << "  --hidden_size: hidden dimension, must be 128-aligned, range [1024,8192] (default: 2048)" << std::endl;
    std::cerr << "  --rankNum:     number of ranks, range [2,8] (default: 2)" << std::endl;
    std::cerr << "  --perf_mode:   enable performance mode (10 iterations)" << std::endl;
    std::cerr << "Example: " << programName << " --bs 1024 --hidden_size 2048 --rankNum 2" << std::endl;
}

int main(int argc, char* argv[])
{
    static const struct option longOptions[] = {
        {"bs",          1, 0, 'b'},
        {"hidden_size", 1, 0, 'h'},
        {"rankNum",     1, 0, 'r'},
        {"perf_mode",   0, 0, 'p'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "b:h:r:p", longOptions, nullptr)) != -1) {
        switch (opt) {
            case 'b': gBs = std::stoi(optarg); break;
            case 'h': gHiddenSize = std::stoi(optarg); break;
            case 'r': gRankNum = std::stoi(optarg); break;
            case 'p': gPerfMode = true; break;
        }
    }

    if (gBs <= 0 || gHiddenSize <= 0 || gRankNum <= 0) {
        printUsage(argv[0]);
        return 1;
    }
    if (gRankNum > MAX_RANK_NUM) {
        std::cerr << "ERROR: rankNum=" << gRankNum << " exceeds MAX_RANK_NUM=" << MAX_RANK_NUM << std::endl;
        return 1;
    }
    if (gRankNum < 2) {
        std::cerr << "ERROR: rankNum must be >= 2" << std::endl;
        return 1;
    }
    if (gHiddenSize % GROUP_SIZE != 0) {
        std::cerr << "ERROR: hidden_size=" << gHiddenSize << " must be " << GROUP_SIZE << "-aligned" << std::endl;
        return 1;
    }
    if (gHiddenSize < 1024 || gHiddenSize > 8192) {
        std::cerr << "ERROR: hidden_size=" << gHiddenSize << " must be in [1024, 8192]" << std::endl;
        return 1;
    }
    if (gBs % gRankNum != 0) {
        std::cerr << "ERROR: bs=" << gBs << " must be divisible by rankNum=" << gRankNum << std::endl;
        return 1;
    }

    LOG_PRINT("[INFO] Config: bs=%d, hidden_size=%d, rankNum=%d, hcclBufSize=%ld bytes",
        gBs, gHiddenSize, gRankNum, gHcclBufferSize);

    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d", ret); return ret);

    uint32_t devNum = static_cast<uint32_t>(gRankNum);
    std::vector<aclrtStream> stream(devNum);
    std::vector<aclrtContext> context(devNum);
    for (uint32_t rankId = 0; rankId < devNum; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d", ret); return ret);
    }

    std::vector<int32_t> devices(devNum);
    for (uint32_t i = 0; i < devNum; i++) {
        devices[i] = static_cast<int32_t>(i);
    }

    std::vector<HcclComm> comms(devNum);
    ret = HcclCommInitAll(devNum, devices.data(), comms.data());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d", ret); return ret);

    std::vector<Args> args(devNum);
    std::vector<std::unique_ptr<std::thread>> threads(devNum);
    std::vector<int> returnCodes(devNum, 0);

    for (uint32_t rankId = 0; rankId < devNum; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].rankDim = devNum;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].context = context[rankId];
        args[rankId].stream = stream[rankId];
        args[rankId].bs = gBs;
        args[rankId].hiddenSize = gHiddenSize;
        threads[rankId].reset(new(std::nothrow) std::thread([&args, &returnCodes, rankId]() {
            returnCodes[rankId] = LaunchOneThread(args[rankId]);
        }));
        if (!threads[rankId]) {
            std::cerr << "[ERROR] Failed to create thread for rank " << rankId << std::endl;
            returnCodes[rankId] = -1;
        }
    }

    for (uint32_t rankId = 0; rankId < devNum; rankId++) {
        if (threads[rankId]) {
            threads[rankId]->join();
        }
    }

    for (uint32_t rankId = 0; rankId < devNum; rankId++) {
        aclrtDestroyStream(stream[rankId]);
        aclrtResetDevice(rankId);
        aclrtDestroyContext(context[rankId]);
    }
    aclFinalize();

    int finalRet = 0;
    for (uint32_t rankId = 0; rankId < devNum; rankId++) {
        if (returnCodes[rankId] != 0) {
            finalRet = 1;
        }
    }

    if (finalRet == 0) {
        printf("========== All ranks kernel execution completed ==========\n");
    } else {
        printf("========== Some ranks FAILED ==========\n");
    }

    return finalRet;
}
