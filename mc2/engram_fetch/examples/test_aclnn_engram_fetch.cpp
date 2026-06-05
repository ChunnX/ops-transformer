#include <thread>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath> // 修复: 添加缺失的 <cmath> 头文件以使用 fabs
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "../op_api/aclnnEngramFetch.h"
#include "aclnn/opdev/fp16_t.h"

// 修复: 删除续行符 \ 后面的不可见空格，否则会导致宏定义语法错误
#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

// 修复: 删除续行符 \ 后面的不可见空格
#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

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

int CreateAclTensorFromFile(void *hostData, size_t dataSize, const std::vector<int64_t> &shape, void **deviceAddr,
                            aclDataType dataType, aclTensor **tensor)
{
    auto ret = aclrtMalloc(deviceAddr, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, dataSize, hostData, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
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
    uint32_t rankId;
    HcclComm hcclEpComm;
    aclrtStream stream;
    aclrtContext context;
};

const uint32_t EP_WORLD_SIZE = 2;
const uint32_t DEV_NUM = EP_WORLD_SIZE;
const int64_t NUM_ENTRIES_PER_RANK = 4;
const int64_t HIDDEN_SIZE = 8;
const int64_t NUM_TOKENS = 4;

int launchOneThreadEngramFetch(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);

    char hcomEpName[128] = {0};
    ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank=%d, hcomEpName=%s\n", args.rankId, hcomEpName);

    // 分配 pinBuffer: 每个 rank 存储 NUM_ENTRIES_PER_RANK 个 hidden state
    size_t pinBufferSize = NUM_ENTRIES_PER_RANK * HIDDEN_SIZE * sizeof(op::fp16_t);
    void *pinBufferAddr = nullptr;
    ret = aclrtMalloc(&pinBufferAddr, pinBufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc pinBuffer failed. ret: %d\n", ret); return ret);

    // 填充 pinBuffer 数据: entry[i] 的每个 hidden 元素 = i + 1.0
    std::vector<op::fp16_t> pinBufferHostData(NUM_ENTRIES_PER_RANK * HIDDEN_SIZE);
    for (int64_t entry = 0; entry < NUM_ENTRIES_PER_RANK; entry++) {
        for (int64_t h = 0; h < HIDDEN_SIZE; h++) {
            pinBufferHostData[entry * HIDDEN_SIZE + h] = op::fp16_t(static_cast<float>(entry + 1));
        }
    }
    ret = aclrtMemcpy(pinBufferAddr, pinBufferSize, pinBufferHostData.data(), pinBufferSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy pinBuffer failed. ret: %d\n", ret); return ret);

    // 构造 EngramBufferHandle 结构体
    // host 端只能填 rankId, rankSize, commBuffer; hcommHandle_ 由框架/HcclContext 处理
    struct HostEngramBufferHandle {
        uint32_t rankId;
        uint32_t rankSize;
        uint64_t hcommHandle_[8]; // HCCL_MAX_RANK_SIZE typically 8 or 64, use 8 for 2-rank test
        uint64_t commBuffer[8];
    };

    HostEngramBufferHandle bh;
    memset(&bh, 0, sizeof(bh));
    bh.rankId = args.rankId;
    bh.rankSize = EP_WORLD_SIZE;
    // commBuffer[self] = 本 rank pinBuffer 地址
    bh.commBuffer[args.rankId] = reinterpret_cast<uint64_t>(pinBufferAddr);
    // 其他 rank 的 commBuffer 暂填 0 (多卡时由通信域获取)
    for (uint32_t i = 0; i < EP_WORLD_SIZE; i++) {
        if (i != args.rankId) {
            bh.commBuffer[i] = 0; // TODO: get remote buffer addr from HCCL
        }
    }
    // hcommHandle_ 暂填 0 (需要从 HcclContext 获取 ChannelEntity)
    for (uint32_t i = 0; i < EP_WORLD_SIZE - 1; i++) {
        bh.hcommHandle_[i] = 0; // TODO: get ChannelEntity from HcclGetContext
    }

    // 创建 bufferhandle tensor (INT32 ND, 用来传 EngramBufferHandle 结构体)
    size_t bhSize = sizeof(bh);
    int64_t bhElemCount = (bhSize + sizeof(int32_t) - 1) / sizeof(int32_t); // 向上取整到 int32 元素
    void *bhDeviceAddr = nullptr;
    std::vector<int32_t> bhHostData(bhElemCount, 0);
    memcpy(bhHostData.data(), &bh, bhSize);
    aclTensor *bufferhandleTensor = nullptr;
    // 修复: 显式指定模板参数为 int32_t，因为 {bhElemCount} 作为初始化列表无法推导模板类型 T
    ret =
        CreateAclTensor<int32_t>(bhHostData, {bhElemCount}, &bhDeviceAddr, aclDataType::ACL_INT32, &bufferhandleTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateAclTensor bufferhandle failed.\n"); return ret);

    // 创建 indices tensor: 全部指向 self-rank 的 entry (单卡测试)
    std::vector<int32_t> indicesHostData;
    for (int64_t i = 0; i < NUM_TOKENS; i++) {
        indicesHostData.push_back(static_cast<int32_t>(i % NUM_ENTRIES_PER_RANK + args.rankId * NUM_ENTRIES_PER_RANK));
    }
    void *indicesDeviceAddr = nullptr;
    aclTensor *indicesTensor = nullptr;
    ret = CreateAclTensor(indicesHostData, {NUM_TOKENS}, &indicesDeviceAddr, aclDataType::ACL_INT32, &indicesTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateAclTensor indices failed.\n"); return ret);

    // 创建 fetched 输出 tensor (BF16 ND)
    std::vector<op::fp16_t> fetchedHostData(NUM_TOKENS * HIDDEN_SIZE, op::fp16_t(0.0f));
    void *fetchedDeviceAddr = nullptr;
    aclTensor *fetchedTensor = nullptr;
    ret = CreateAclTensor(fetchedHostData, {NUM_TOKENS, HIDDEN_SIZE}, &fetchedDeviceAddr, aclDataType::ACL_BF16,
                          &fetchedTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CreateAclTensor fetched failed.\n"); return ret);

    // 调用 aclnnEngramFetchGetWorkspaceSize
    int64_t numEntriesPerRank = NUM_ENTRIES_PER_RANK;
    int64_t hiddenSize = HIDDEN_SIZE;
    int64_t outputDtype = 1; // BF16

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    LOG_PRINT("[INFO] rank=%d, calling aclnnEngramFetchGetWorkspaceSize...\n", args.rankId);
    ret = aclnnEngramFetchGetWorkspaceSize(bufferhandleTensor, indicesTensor, numEntriesPerRank, hiddenSize, hcomEpName,
                                           outputDtype, fetchedTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnEngramFetchGetWorkspaceSize failed. ret=%d\n", ret);
              return ret);
    LOG_PRINT("[INFO] rank=%d, workspaceSize=%lu\n", args.rankId, workspaceSize);

    // 分配 workspace
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret: %d\n", ret); return ret);
    }

    // 调用 aclnnEngramFetch
    LOG_PRINT("[INFO] rank=%d, calling aclnnEngramFetch...\n", args.rankId);
    ret = aclnnEngramFetch(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnEngramFetch failed. ret=%d\n", ret); return ret);

    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStream failed. ret=%d\n", ret); return ret);

    LOG_PRINT("[INFO] rank=%d, aclnnEngramFetch execute success.\n", args.rankId);

    // 读回 fetched 输出
    ret = aclrtMemcpy(fetchedHostData.data(), fetchedHostData.size() * sizeof(op::fp16_t), fetchedDeviceAddr,
                      fetchedHostData.size() * sizeof(op::fp16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy fetched back failed. ret=%d\n", ret); return ret);

    // 验证: self-rank token 的 fetched 输出应等于 pinBuffer 对应 entry 的数据
    bool allCorrect = true;
    for (int64_t i = 0; i < NUM_TOKENS; i++) {
        int32_t globalIdx = indicesHostData[i];
        int32_t entryIdx = globalIdx - static_cast<int32_t>(args.rankId * NUM_ENTRIES_PER_RANK);
        float expected = static_cast<float>(entryIdx + 1);
        for (int64_t h = 0; h < HIDDEN_SIZE; h++) {
            float actual = static_cast<float>(fetchedHostData[i * HIDDEN_SIZE + h]);
            if (fabs(actual - expected) > 0.01) {
                LOG_PRINT("[ERROR] rank=%d, token=%ld, h=%ld, expected=%.2f, actual=%.2f\n", args.rankId, i, h,
                          expected, actual);
                allCorrect = false;
            }
        }
    }
    if (allCorrect) {
        LOG_PRINT("[INFO] rank=%d, all self-rank tokens verified OK!\n", args.rankId);
    }

    // 打印部分输出
    for (int64_t i = 0; i < NUM_TOKENS; i++) {
        LOG_PRINT("[INFO] rank=%d, token[%ld]: ", args.rankId, i);
        for (int64_t h = 0; h < HIDDEN_SIZE; h++) {
            LOG_PRINT("%.2f ", static_cast<float>(fetchedHostData[i * HIDDEN_SIZE + h]));
        }
        LOG_PRINT("\n");
    }

    // 释放资源
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclDestroyTensor(bufferhandleTensor);
    aclDestroyTensor(indicesTensor);
    aclDestroyTensor(fetchedTensor);
    aclrtFree(bhDeviceAddr);
    aclrtFree(indicesDeviceAddr);
    aclrtFree(fetchedDeviceAddr);
    aclrtFree(pinBufferAddr);

    return 0;
}

int run_example()
{
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret: %d\n", ret); return ret);

    aclrtStream stream[DEV_NUM];
    aclrtContext context[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret: %d\n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret: %d\n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret: %d\n", ret); return ret);
    }

    // 初始化 EP 通信域
    int32_t devices[DEV_NUM];
    for (uint32_t i = 0; i < DEV_NUM; i++) {
        devices[i] = i;
    }
    HcclComm comms[DEV_NUM];
    ret = HcclCommInitAll(DEV_NUM, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret: %d\n", ret); return ret);

    Args args[DEV_NUM];
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclEpComm = comms[rankId];
        args[rankId].stream = stream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new (std::nothrow) std::thread(&launchOneThreadEngramFetch, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }

    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        HcclCommDestroy(comms[rankId]);
        aclrtDestroyStream(stream[rankId]);
        aclrtDestroyContext(context[rankId]);
        aclrtResetDevice(rankId);
    }

    aclFinalize();
    LOG_PRINT("[INFO] aclFinalize success\n");
    return 0;
}

int main(int argc, char *argv[])
{
    return run_example();
}
