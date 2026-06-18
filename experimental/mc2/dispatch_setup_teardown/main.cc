/*!
 * \file main.cc
 * \brief 集成 launcher —— setup(CCU 写法) 与 teardown(AIV 写法) 在两条 stream 上靠状态位握手。
 *
 *   每个 device(rank) 起两条 stream:
 *     - streamSetup   : dispatch_setup —— CCU 算子(ccu:: 写法, HcclDispatchSetup 封装)
 *                       把状态值写进状态区(recvBuf)。
 *     - streamTeardown: dispatch_teardown —— AIV 算子(<<<>>> 写法)忙等轮询状态区,
 *                       见到非 0 就往下走、写 done 标记。
 *   先在 streamTeardown 发 teardown(忙等), 再在 streamSetup 发 setup(CCU 写状态)。
 *   若 teardown 越过同步点(done=7), 即证明 CCU setup ↔ AIV teardown 靠状态位握手成立。
 */
#include <iostream>
#include <vector>
#include <thread>
#include <cstring>

#include <acl/acl_rt.h>
#include <hccl/hccl_types.h>
#include <dispatch_setup.h>   // CCU setup 算子接口(ccu:: 实现)
#include "teardown_aiv.h"            // AIV teardown 算子(<<<>>> 实现)

#define ACLCHECK(ret)                                                                          \
    do {                                                                                       \
        if ((ret) != ACL_SUCCESS) {                                                            \
            printf("[acl err] %s:%d ret=%d\n", __FILE__, __LINE__, (ret));                     \
            return (ret);                                                                      \
        }                                                                                      \
    } while (0)
#define HCCLCHECK(ret)                                                                         \
    do {                                                                                       \
        if ((ret) != HCCL_SUCCESS) {                                                           \
            printf("[hccl err] %s:%d ret=%d\n", __FILE__, __LINE__, (ret));                    \
            return (ret);                                                                      \
        }                                                                                      \
    } while (0)

struct ThreadContext {
    HcclRootInfo *rootInfo;
    uint32_t device;
    uint32_t devCount;
};

int Sample(void *arg)
{
    ThreadContext *ctx = (ThreadContext *)arg;
    uint32_t device = ctx->device;
    ACLCHECK(aclrtSetDevice(static_cast<int32_t>(device)));

    HcclComm hcclComm;
    HCCLCHECK(HcclCommInitRootInfo(ctx->devCount, ctx->rootInfo, device, &hcclComm));

    // 两条 stream: 一条给 CCU setup, 一条给 AIV teardown
    aclrtStream streamSetup = nullptr, streamTeardown = nullptr;
    ACLCHECK(aclrtCreateStream(&streamSetup));
    ACLCHECK(aclrtCreateStream(&streamTeardown));

    // 输入(状态值 = 1 标志位) + 状态区(recvBuf, 被 CCU setup 写) + done 区(被 AIV teardown 写)
    uint64_t sendCount = 1U;
    uint64_t recvCount = ctx->devCount;
    size_t sendSize = sendCount * sizeof(int32_t);
    size_t recvSize = recvCount * sizeof(int32_t);
    if (recvSize < 32) recvSize = 32;  // 给 AIV DataCopy 32B 对齐留够
    void *sendBuf = nullptr, *statusBuf = nullptr, *doneBuf = nullptr;
    ACLCHECK(aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&statusBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&doneBuf, 32, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMemset(statusBuf, recvSize, 0, recvSize));  // 状态区初值 0
    ACLCHECK(aclrtMemset(doneBuf, 32, 0, 32));

    // 输入初始化为 1(标志位值), host->device
    void *hostBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&hostBuf, sendSize));
    static_cast<int32_t *>(hostBuf)[0] = 1;  // 标志位 = 1
    ACLCHECK(aclrtMemcpy(sendBuf, sendSize, hostBuf, sendSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtFreeHost(hostBuf));

    printf("[rank %u] launching teardown(AIV) FIRST on streamTeardown, then setup(CCU) on streamSetup\n", device);

    // 先发 teardown(AIV, 忙等轮询 statusBuf) —— 证明同步靠状态位
    dispatch_teardown_demo(1, streamTeardown, (uint8_t *)statusBuf, (uint8_t *)doneBuf);
    // 再发 setup(CCU, 把状态值写进 statusBuf=recvBuf) —— ccu:: 写法
    HCCLCHECK(HcclDispatchSetup(sendBuf, statusBuf, sendCount, HCCL_DATA_TYPE_INT32, hcclComm, streamSetup));

    // 同步两条 stream
    ACLCHECK(aclrtSynchronizeStream(streamSetup));
    ACLCHECK(aclrtSynchronizeStream(streamTeardown));

    // 读回 done, 验证 teardown 越过了同步点
    int32_t doneHost = 0;
    ACLCHECK(aclrtMemcpy(&doneHost, sizeof(int32_t), doneBuf, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));
    printf("[rank %u] done=%d -> %s\n", device, doneHost,
           (doneHost == 7) ? "SUCCESS: CCU setup <-> AIV teardown handshake via status bit works"
                             : "teardown did not pass sync point");

    HCCLCHECK(HcclCommDestroy(hcclComm));
    if (sendBuf) ACLCHECK(aclrtFree(sendBuf));
    if (statusBuf) ACLCHECK(aclrtFree(statusBuf));
    if (doneBuf) ACLCHECK(aclrtFree(doneBuf));
    ACLCHECK(aclrtDestroyStream(streamSetup));
    ACLCHECK(aclrtDestroyStream(streamTeardown));
    ACLCHECK(aclrtResetDevice(device));
    return 0;
}

int main()
{
    ACLCHECK(aclInit(NULL));
    uint32_t devCount;
    ACLCHECK(aclrtGetDeviceCount(&devCount));
    devCount = 1;  // 单卡两 stream 握手演示: rankSize=1 走本地拷贝, 不需卡间 channel
    printf("Using %u NPU device(s) (ranks)\n", devCount);

    ACLCHECK(aclrtSetDevice(0));
    void *rootInfoBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&rootInfoBuf, sizeof(HcclRootInfo)));
    HcclRootInfo *rootInfo = (HcclRootInfo *)rootInfoBuf;
    HCCLCHECK(HcclGetRootInfo(rootInfo));

    std::vector<std::thread> threads(devCount);
    std::vector<ThreadContext> args(devCount);
    for (uint32_t i = 0; i < devCount; i++) {
        args[i] = {rootInfo, i, devCount};
        threads[i] = std::thread(Sample, (void *)&args[i]);
    }
    for (auto &t : threads) t.join();

    ACLCHECK(aclrtFreeHost(rootInfoBuf));
    ACLCHECK(aclFinalize());
    return 0;
}
