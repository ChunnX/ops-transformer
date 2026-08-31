/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_ffn_to_attention_v2.cpp
 * \brief ACLNN example for FFNToAttentionV2 on Ascend 950PR/Ascend 950DT.
 */

#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_ffn_to_attention_v2.h"
#include "hccl/hccl.h"
#if __has_include("common/op_api/mc2_context.h")
#include "common/op_api/mc2_context.h"
#else
#include "../../../common/op_api/mc2_context.h"
#endif

namespace {
constexpr uint32_t WORLD_SIZE = 8U;
constexpr uint32_t ATTENTION_WORKER_NUM = 4U;
constexpr int64_t MICRO_BATCH_NUM = 1;
constexpr int64_t TOKEN_NUM = 8;
constexpr int64_t HIDDEN_SIZE = 7168;
constexpr int64_t EXPERT_NUM_PER_TOKEN = 8;
constexpr uint64_t TOKEN_INFO_ALIGN = 512UL;
constexpr uint64_t CCL_BUFFER_ALIGN = 2UL * 1024UL * 1024UL;
constexpr uint16_t BF16_ONE = 0x3F80U;

#define LOG_PRINT(message, ...) std::printf(message, ##__VA_ARGS__)

bool CheckedMultiply(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (lhs != 0U && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool CheckedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool AlignUp(uint64_t value, uint64_t alignment, uint64_t &result)
{
    if (alignment == 0U || value > std::numeric_limits<uint64_t>::max() - (alignment - 1U)) {
        return false;
    }
    result = (value + alignment - 1U) / alignment * alignment;
    return true;
}

bool CalculateRequiredCclBufferSize(uint64_t &requiredSize)
{
    uint64_t slotNum = 0U;
    uint64_t tokenInfoSize = 0U;
    uint64_t tokenDataSize = 0U;
    uint64_t tokenInfoAlignedSize = 0U;
    uint64_t totalSize = 0U;
    return CheckedMultiply(static_cast<uint64_t>(MICRO_BATCH_NUM), static_cast<uint64_t>(TOKEN_NUM), slotNum) &&
           CheckedMultiply(slotNum, static_cast<uint64_t>(EXPERT_NUM_PER_TOKEN), slotNum) &&
           CheckedMultiply(slotNum, sizeof(int32_t), tokenInfoSize) &&
           CheckedMultiply(slotNum, static_cast<uint64_t>(HIDDEN_SIZE), tokenDataSize) &&
           CheckedMultiply(tokenDataSize, sizeof(uint16_t), tokenDataSize) &&
           AlignUp(tokenInfoSize, TOKEN_INFO_ALIGN, tokenInfoAlignedSize) &&
           CheckedAdd(tokenInfoAlignedSize, tokenDataSize, totalSize) &&
           AlignUp(totalSize, CCL_BUFFER_ALIGN, requiredSize);
}

bool GetElementCount(const std::vector<int64_t> &shape, uint64_t &elementCount)
{
    elementCount = 1U;
    for (int64_t dim : shape) {
        if (dim < 0 || !CheckedMultiply(elementCount, static_cast<uint64_t>(dim), elementCount)) {
            return false;
        }
    }
    return true;
}

class TensorResource {
public:
    TensorResource() = default;
    TensorResource(const TensorResource &) = delete;
    TensorResource &operator=(const TensorResource &) = delete;

    ~TensorResource()
    {
        if (tensor_ != nullptr) {
            aclDestroyTensor(tensor_);
        }
        if (deviceAddr_ != nullptr) {
            aclrtFree(deviceAddr_);
        }
    }

    template <typename T>
    aclError Create(const std::vector<T> &hostData, const std::vector<int64_t> &shape, aclDataType dataType)
    {
        uint64_t elementCount = 0U;
        if (!GetElementCount(shape, elementCount) || elementCount != hostData.size() ||
            elementCount > std::numeric_limits<size_t>::max() / sizeof(T)) {
            LOG_PRINT("[ERROR] invalid tensor shape or tensor byte size overflow.\n");
            return ACL_ERROR_INVALID_PARAM;
        }

        const size_t byteSize = static_cast<size_t>(elementCount) * sizeof(T);
        aclError ret = aclrtMalloc(&deviceAddr_, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret);
            return ret;
        }
        ret = aclrtMemcpy(deviceAddr_, byteSize, hostData.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[ERROR] aclrtMemcpy failed. ret = %d\n", ret);
            return ret;
        }

        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            if (shape[static_cast<size_t>(i + 1)] != 0 &&
                strides[static_cast<size_t>(i + 1)] >
                    std::numeric_limits<int64_t>::max() / shape[static_cast<size_t>(i + 1)]) {
                LOG_PRINT("[ERROR] tensor stride overflow.\n");
                return ACL_ERROR_INVALID_PARAM;
            }
            strides[static_cast<size_t>(i)] =
                shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
        }
        tensor_ = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                  shape.size(), deviceAddr_);
        if (tensor_ == nullptr) {
            LOG_PRINT("[ERROR] aclCreateTensor returned nullptr.\n");
            return ACL_ERROR_INVALID_PARAM;
        }
        return ACL_SUCCESS;
    }

    aclTensor *Get() const
    {
        return tensor_;
    }

private:
    void *deviceAddr_ = nullptr;
    aclTensor *tensor_ = nullptr;
};

class DeviceMemory {
public:
    DeviceMemory() = default;
    DeviceMemory(const DeviceMemory &) = delete;
    DeviceMemory &operator=(const DeviceMemory &) = delete;
    ~DeviceMemory()
    {
        if (addr_ != nullptr) {
            aclrtFree(addr_);
        }
    }

    aclError Allocate(uint64_t size)
    {
        if (size == 0U) {
            return ACL_SUCCESS;
        }
        return aclrtMalloc(&addr_, size, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    void *Get() const
    {
        return addr_;
    }

private:
    void *addr_ = nullptr;
};

class HostBarrier {
public:
    explicit HostBarrier(uint32_t participantNum) : participantNum_(participantNum) {}

    bool ArriveAndWait(bool success)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        success_ = success_ && success;
        ++arrivedNum_;
        if (arrivedNum_ == participantNum_) {
            condition_.notify_all();
        } else {
            condition_.wait(lock, [this]() { return arrivedNum_ == participantNum_; });
        }
        return success_;
    }

private:
    const uint32_t participantNum_;
    uint32_t arrivedNum_ = 0U;
    bool success_ = true;
    std::mutex mutex_;
    std::condition_variable condition_;
};

struct RankArgs {
    uint32_t rankId = 0U;
    HcclComm comm = nullptr;
    aclrtStream stream = nullptr;
    aclrtContext context = nullptr;
    int result = ACL_SUCCESS;
};

int RunSender(RankArgs &args, const char *groupName, aclTensor *mc2Context, uint64_t hcclBufferSize)
{
    uint64_t requiredCclBufferSize = 0U;
    if (!CalculateRequiredCclBufferSize(requiredCclBufferSize) || hcclBufferSize < requiredCclBufferSize ||
        hcclBufferSize > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        LOG_PRINT("[ERROR] rank %u has invalid HCCL buffer size %llu, required at least %llu bytes.\n", args.rankId,
                  static_cast<unsigned long long>(hcclBufferSize),
                  static_cast<unsigned long long>(requiredCclBufferSize));
        return ACL_ERROR_INVALID_PARAM;
    }

    const std::vector<int64_t> xShape = {TOKEN_NUM, HIDDEN_SIZE};
    const std::vector<int64_t> metadataShape = {TOKEN_NUM};
    const std::vector<int64_t> scalarShape = {1};
    const std::vector<int64_t> rankTableShape = {ATTENTION_WORKER_NUM};
    std::vector<uint16_t> xHostData(static_cast<size_t>(TOKEN_NUM * HIDDEN_SIZE), BF16_ONE);
    std::vector<int32_t> sessionIdsHostData(static_cast<size_t>(TOKEN_NUM), 0);
    std::vector<int32_t> microBatchIdsHostData(static_cast<size_t>(TOKEN_NUM), 0);
    std::vector<int32_t> tokenIdsHostData(static_cast<size_t>(TOKEN_NUM), 0);
    std::vector<int32_t> expertOffsetsHostData(static_cast<size_t>(TOKEN_NUM), 0);
    std::vector<int64_t> actualTokenNumHostData = {TOKEN_NUM};
    std::vector<int32_t> rankTableHostData(static_cast<size_t>(ATTENTION_WORKER_NUM), 0);
    for (int32_t i = 0; i < TOKEN_NUM; ++i) {
        sessionIdsHostData[static_cast<size_t>(i)] = i % static_cast<int32_t>(ATTENTION_WORKER_NUM);
        tokenIdsHostData[static_cast<size_t>(i)] = i;
        expertOffsetsHostData[static_cast<size_t>(i)] = i % static_cast<int32_t>(EXPERT_NUM_PER_TOKEN);
    }
    for (uint32_t i = 0U; i < ATTENTION_WORKER_NUM; ++i) {
        rankTableHostData[i] = static_cast<int32_t>(i);
    }

    TensorResource x;
    TensorResource sessionIds;
    TensorResource microBatchIds;
    TensorResource tokenIds;
    TensorResource expertOffsets;
    TensorResource actualTokenNum;
    TensorResource attnRankTable;
    aclError ret = x.Create(xHostData, xShape, ACL_BF16);
    if (ret == ACL_SUCCESS) {
        ret = sessionIds.Create(sessionIdsHostData, metadataShape, ACL_INT32);
    }
    if (ret == ACL_SUCCESS) {
        ret = microBatchIds.Create(microBatchIdsHostData, metadataShape, ACL_INT32);
    }
    if (ret == ACL_SUCCESS) {
        ret = tokenIds.Create(tokenIdsHostData, metadataShape, ACL_INT32);
    }
    if (ret == ACL_SUCCESS) {
        ret = expertOffsets.Create(expertOffsetsHostData, metadataShape, ACL_INT32);
    }
    if (ret == ACL_SUCCESS) {
        ret = actualTokenNum.Create(actualTokenNumHostData, scalarShape, ACL_INT64);
    }
    if (ret == ACL_SUCCESS) {
        ret = attnRankTable.Create(rankTableHostData, rankTableShape, ACL_INT32);
    }
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    const int64_t tokenInfoShapeData[] = {MICRO_BATCH_NUM, TOKEN_NUM, EXPERT_NUM_PER_TOKEN};
    const int64_t tokenDataShapeData[] = {MICRO_BATCH_NUM, TOKEN_NUM, EXPERT_NUM_PER_TOKEN, HIDDEN_SIZE};
    using IntArrayPtr = std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)>;
    IntArrayPtr tokenInfoTableShape(aclCreateIntArray(tokenInfoShapeData, 3U), aclDestroyIntArray);
    IntArrayPtr tokenDataShape(aclCreateIntArray(tokenDataShapeData, 4U), aclDestroyIntArray);
    if (tokenInfoTableShape == nullptr || tokenDataShape == nullptr) {
        LOG_PRINT("[ERROR] aclCreateIntArray failed.\n");
        return ACL_ERROR_INVALID_PARAM;
    }

    uint64_t workspaceSize = 0U;
    aclOpExecutor *executor = nullptr;
    ret = aclnnFFNToAttentionV2GetWorkspaceSize(
        mc2Context, x.Get(), sessionIds.Get(), microBatchIds.Get(), tokenIds.Get(), expertOffsets.Get(),
        actualTokenNum.Get(), attnRankTable.Get(), groupName, WORLD_SIZE, tokenInfoTableShape.get(),
        tokenDataShape.get(), static_cast<int64_t>(hcclBufferSize), &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] rank %u aclnnFFNToAttentionV2GetWorkspaceSize failed. ret = %d\n", args.rankId, ret);
        return ret;
    }

    DeviceMemory workspace;
    ret = workspace.Allocate(workspaceSize);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] rank %u workspace allocation failed. ret = %d\n", args.rankId, ret);
        return ret;
    }
    ret = aclnnFFNToAttentionV2(workspace.Get(), workspaceSize, executor, args.stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] rank %u aclnnFFNToAttentionV2 failed. ret = %d\n", args.rankId, ret);
        return ret;
    }
    ret = aclrtSynchronizeStream(args.stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] rank %u aclrtSynchronizeStream failed. ret = %d\n", args.rankId, ret);
        return ret;
    }
    LOG_PRINT("[INFO] FFN rank %u sent %lld tokens successfully.\n", args.rankId, TOKEN_NUM);
    return ACL_SUCCESS;
}

int RunRank(RankArgs &args, HostBarrier &contextReadyBarrier, HostBarrier &executionDoneBarrier)
{
    int ret = aclrtSetCurrentContext(args.context);
    char groupName[128] = {0};
    uint64_t hcclBufferSize = 0U;
    aclTensor *mc2Context = nullptr;
    if (ret == ACL_SUCCESS) {
        ret = HcclGetCommName(args.comm, groupName);
    }
    if (ret == ACL_SUCCESS) {
        ret = Mc2Aclnn::Mc2Context::GetMc2ContextTensor(groupName, "ffn_to_attention_v2", hcclBufferSize,
                                                        mc2Context);
    }
    using TensorPtr = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;
    TensorPtr mc2ContextGuard(mc2Context, aclDestroyTensor);
    const bool contextReady = contextReadyBarrier.ArriveAndWait(ret == ACL_SUCCESS && mc2Context != nullptr);
    if (!contextReady) {
        LOG_PRINT("[ERROR] rank %u failed to initialize the MC2 context. ret = %d\n", args.rankId, ret);
        return ret == ACL_SUCCESS ? ACL_ERROR_INVALID_PARAM : ret;
    }

    int runRet = ACL_SUCCESS;
    if (args.rankId >= ATTENTION_WORKER_NUM) {
        runRet = RunSender(args, groupName, mc2Context, hcclBufferSize);
    } else {
        LOG_PRINT("[INFO] rank %u is an Attention worker and keeps its communication window alive.\n", args.rankId);
    }
    const bool allRanksSucceeded = executionDoneBarrier.ArriveAndWait(runRet == ACL_SUCCESS);
    return allRanksSucceeded ? runRet : ACL_ERROR_FAILURE;
}

void CleanupRuntime(std::vector<aclrtContext> &contexts, std::vector<aclrtStream> &streams,
                    std::vector<HcclComm> &comms, bool commInitialized)
{
    for (uint32_t rankId = 0U; rankId < contexts.size(); ++rankId) {
        if (contexts[rankId] != nullptr) {
            aclrtSetCurrentContext(contexts[rankId]);
        }
        if (commInitialized && comms[rankId] != nullptr) {
            HcclCommDestroy(comms[rankId]);
        }
        if (streams[rankId] != nullptr) {
            aclrtDestroyStream(streams[rankId]);
        }
        if (contexts[rankId] != nullptr) {
            aclrtDestroyContext(contexts[rankId]);
        }
        aclrtResetDevice(static_cast<int32_t>(rankId));
    }
    aclFinalize();
}
} // namespace

int main()
{
    int ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret);
        return ret;
    }

    std::vector<aclrtContext> contexts(WORLD_SIZE, nullptr);
    std::vector<aclrtStream> streams(WORLD_SIZE, nullptr);
    std::vector<HcclComm> comms(WORLD_SIZE, nullptr);
    for (uint32_t rankId = 0U; rankId < WORLD_SIZE; ++rankId) {
        ret = aclrtSetDevice(static_cast<int32_t>(rankId));
        if (ret == ACL_SUCCESS) {
            ret = aclrtCreateContext(&contexts[rankId], static_cast<int32_t>(rankId));
        }
        if (ret == ACL_SUCCESS) {
            ret = aclrtCreateStream(&streams[rankId]);
        }
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[ERROR] initialize rank %u failed. ret = %d\n", rankId, ret);
            CleanupRuntime(contexts, streams, comms, false);
            return ret;
        }
    }

    int32_t devices[WORLD_SIZE] = {0};
    for (uint32_t rankId = 0U; rankId < WORLD_SIZE; ++rankId) {
        devices[rankId] = static_cast<int32_t>(rankId);
    }
    ret = HcclCommInitAll(WORLD_SIZE, devices, comms.data());
    if (ret != HCCL_SUCCESS) {
        LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d\n", ret);
        CleanupRuntime(contexts, streams, comms, false);
        return ret;
    }

    HostBarrier contextReadyBarrier(WORLD_SIZE);
    HostBarrier executionDoneBarrier(WORLD_SIZE);
    std::vector<RankArgs> args(WORLD_SIZE);
    std::vector<std::thread> threads;
    threads.reserve(WORLD_SIZE);
    for (uint32_t rankId = 0U; rankId < WORLD_SIZE; ++rankId) {
        args[rankId] = {rankId, comms[rankId], streams[rankId], contexts[rankId], ACL_SUCCESS};
        threads.emplace_back([&, rankId]() {
            args[rankId].result = RunRank(args[rankId], contextReadyBarrier, executionDoneBarrier);
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }

    int finalRet = ACL_SUCCESS;
    for (const RankArgs &rankArgs : args) {
        if (rankArgs.result != ACL_SUCCESS) {
            finalRet = rankArgs.result;
            break;
        }
    }
    CleanupRuntime(contexts, streams, comms, true);
    if (finalRet == ACL_SUCCESS) {
        LOG_PRINT("[INFO] FFNToAttentionV2 example completed successfully.\n");
    }
    return finalRet;
}
