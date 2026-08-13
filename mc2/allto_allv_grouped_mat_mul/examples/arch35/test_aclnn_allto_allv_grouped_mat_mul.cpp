/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_allto_allv_grouped_mat_mul_v2.h"
#include "shmem.h"

namespace {
constexpr uint64_t SHMEM_HEAP_BYTES = 1004ULL * 1024ULL * 1024ULL;
constexpr int64_t LOCAL_EXPERTS = 2;
constexpr int64_t K = 32;
constexpr int64_t N = 64;

#define CHECK_ACL(expression)                                                                                          \
    do {                                                                                                               \
        const int checkRet = (expression);                                                                             \
        if (checkRet != ACL_SUCCESS) {                                                                                  \
            std::cerr << #expression << " failed, ret=" << checkRet << std::endl;                                      \
            return checkRet;                                                                                           \
        }                                                                                                              \
    } while (0)

int SetShmemAttr(int32_t rank, int32_t rankSize, const char *ipPort, aclshmemx_init_attr_t &attr,
                 aclshmemx_uniqueid_t &uniqueId)
{
    std::memset(&attr, 0, sizeof(attr));
    std::memset(&uniqueId, 0, sizeof(uniqueId));
    const size_t ipLength = std::min(std::strlen(ipPort), sizeof(attr.ip_port) - 1);
    std::copy_n(ipPort, ipLength, attr.ip_port);
    attr.ip_port[ipLength] = '\0';
    attr.my_pe = rank;
    attr.n_pes = rankSize;
    attr.local_mem_size = SHMEM_HEAP_BYTES;
    const int attrVersion = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attr.option_attr = {attrVersion, ACLSHMEM_DATA_OP_UDMA, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    uniqueId.my_pe = rank;
    uniqueId.n_pes = rankSize;
    attr.comm_args = reinterpret_cast<void *>(&uniqueId);
    return ACLSHMEM_SUCCESS;
}

int64_t Numel(const std::vector<int64_t> &shape)
{
    int64_t result = 1;
    for (const int64_t dim : shape) {
        result *= dim;
    }
    return result;
}

aclTensor *CreateTensor(void *deviceAddress, const std::vector<int64_t> &shape, aclDataType dtype)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t index = static_cast<int64_t>(shape.size()) - 2; index >= 0; --index) {
        strides[index] = strides[index + 1] * shape[index + 1];
    }
    return aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                           shape.size(), deviceAddress);
}
} // namespace

// Start one process per rank:
//   test_aclnn_allto_allv_grouped_mat_mul <rank_size> <rank_id> <ip:port> <device_id>
int main(int argc, char **argv)
{
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " rank_size rank_id ip:port device_id" << std::endl;
        return 1;
    }
    const int32_t rankSize = std::atoi(argv[1]);
    const int32_t rank = std::atoi(argv[2]);
    const int32_t deviceId = std::atoi(argv[4]);
    if (rankSize < 1 || rankSize > 8 || rank < 0 || rank >= rankSize) {
        std::cerr << "This sample requires rank_size in [1,8]." << std::endl;
        return 1;
    }
    const int64_t tokensPerRoute = 2;
    const int64_t m = rankSize * LOCAL_EXPERTS * tokensPerRoute;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    aclshmemx_init_attr_t shmemAttr{};
    aclshmemx_uniqueid_t uniqueId{};
    CHECK_ACL(SetShmemAttr(rank, rankSize, argv[3], shmemAttr, uniqueId));
    CHECK_ACL(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &shmemAttr));

    const std::vector<int64_t> xShape{m, K};
    const std::vector<int64_t> weightShape{LOCAL_EXPERTS, K, N};
    const std::vector<int64_t> yShape{m, N};
    const uint64_t xBytes = Numel(xShape) * sizeof(uint16_t);
    const uint64_t weightBytes = Numel(weightShape) * sizeof(uint16_t);
    const uint64_t yBytes = Numel(yShape) * sizeof(uint16_t);

    void *xDevice = aclshmem_malloc(xBytes);
    if (xDevice == nullptr) {
        std::cerr << "aclshmem_malloc(gmmX) failed" << std::endl;
        return 1;
    }
    void *weightDevice = nullptr;
    void *yDevice = nullptr;
    CHECK_ACL(aclrtMalloc(&weightDevice, weightBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&yDevice, yBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    const std::vector<uint16_t> xHost(Numel(xShape), 0);
    const std::vector<uint16_t> weightHost(Numel(weightShape), 0);
    CHECK_ACL(aclrtMemcpy(xDevice, xBytes, xHost.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(weightDevice, weightBytes, weightHost.data(), weightBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    aclTensor *x = CreateTensor(xDevice, xShape, ACL_BF16);
    aclTensor *weight = CreateTensor(weightDevice, weightShape, ACL_BF16);
    aclTensor *y = CreateTensor(yDevice, yShape, ACL_BF16);
    if (x == nullptr || weight == nullptr || y == nullptr) {
        std::cerr << "aclCreateTensor failed" << std::endl;
        return 1;
    }

    const std::vector<int64_t> sendCounts(rankSize * LOCAL_EXPERTS, tokensPerRoute);
    const std::vector<int64_t> recvCounts(rankSize * LOCAL_EXPERTS, tokensPerRoute);
    aclIntArray *sendArray = aclCreateIntArray(sendCounts.data(), sendCounts.size());
    aclIntArray *recvArray = aclCreateIntArray(recvCounts.data(), recvCounts.size());
    if (sendArray == nullptr || recvArray == nullptr) {
        std::cerr << "aclCreateIntArray failed" << std::endl;
        return 1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *shmemContext = reinterpret_cast<void *>(shmemx_get_ffts_config());
    CHECK_ACL(aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize(x, weight, shmemContext, rankSize, sendArray, recvArray,
                                                            y, &workspaceSize, &executor));
    void *workspace = aclshmem_calloc(1, workspaceSize);
    if (workspace == nullptr) {
        std::cerr << "aclshmem_calloc(workspaceSize=" << workspaceSize << ") failed" << std::endl;
        return 1;
    }
    CHECK_ACL(aclnnAlltoAllvGroupedMatMulV2(workspace, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    std::cout << "rank " << rank << " completed AlltoAllvGroupedMatMulV2 through SHMEM/UDMA" << std::endl;

    aclDestroyIntArray(sendArray);
    aclDestroyIntArray(recvArray);
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(y);
    shmem_free(workspace);
    shmem_free(xDevice);
    aclrtFree(weightDevice);
    aclrtFree(yDevice);
    CHECK_ACL(shmem_finalize());
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}
