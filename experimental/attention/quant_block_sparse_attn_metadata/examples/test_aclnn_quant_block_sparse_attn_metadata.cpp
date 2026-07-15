/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_block_sparse_attn_metadata.h"
#include "../op_kernel_aicpu/quant_block_sparse_attn_metadata.h"

namespace {

#define CHECK_RET(cond) ((cond) ? true : false)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        std::printf(message, ##__VA_ARGS__);                                                                           \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return 0;
}

void DumpMeta(void *data)
{
    auto sectionNum = static_cast<uint32_t *>(data)[optiling::QBSA_HEAD_SECTION_NUM_INDEX];
    optiling::detail::qbsaMetaData qbsaMetadata(data, sectionNum);

    printf("sectionNum: %u\n", qbsaMetadata.GetHeadMetadata(optiling::QBSA_HEAD_SECTION_NUM_INDEX));
    for (uint32_t sectionIdx = 0; sectionIdx < sectionNum; ++sectionIdx) {
        printf("sectionIdx: %u\n", sectionIdx);
        for (uint32_t aicIdx = 0; aicIdx < optiling::AIC_CORE_NUM; ++aicIdx) {
            uint32_t coreEnable =
                qbsaMetadata.GetQbsaMetadata(sectionIdx, aicIdx, optiling::QBSA_CORE_ENABLE_INDEX);
            uint32_t bn1Start =
                qbsaMetadata.GetQbsaMetadata(sectionIdx, aicIdx, optiling::QBSA_BN1_START_INDEX);
            uint32_t mStart =
                qbsaMetadata.GetQbsaMetadata(sectionIdx, aicIdx, optiling::QBSA_M_START_INDEX);
            uint32_t s2Start =
                qbsaMetadata.GetQbsaMetadata(sectionIdx, aicIdx, optiling::QBSA_S2_START_INDEX);
            uint32_t bn1End =
                qbsaMetadata.GetQbsaMetadata(sectionIdx, aicIdx, optiling::QBSA_BN1_END_INDEX);
            uint32_t mEnd =
                qbsaMetadata.GetQbsaMetadata(sectionIdx, aicIdx, optiling::QBSA_M_END_INDEX);
            uint32_t s2End =
                qbsaMetadata.GetQbsaMetadata(sectionIdx, aicIdx, optiling::QBSA_S2_END_INDEX);
            uint32_t firstFdDataWorkspaceIdx =
                qbsaMetadata.GetQbsaMetadata(sectionIdx, aicIdx, optiling::QBSA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX);

            if (coreEnable == 0U && bn1Start == 0U && mStart == 0U && s2Start == 0U &&
                bn1End == 0U && mEnd == 0U && s2End == 0U && firstFdDataWorkspaceIdx == 0U) {
                continue;
            }

            printf("  aicIdx: %u\n", aicIdx);
            printf("    coreEnable: %u\n", coreEnable);
            printf("    bn1 start: %u\n", bn1Start);
            printf("    m start: %u\n", mStart);
            printf("    s2 start: %u\n", s2Start);
            printf("    bn1 end: %u\n", bn1End);
            printf("    m end: %u\n", mEnd);
            printf("    s2 end: %u\n", s2End);
            printf("    first fd data ws idx: %u\n", firstFdDataWorkspaceIdx);
        }
    }
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void DestroyTensor(aclTensor *tensor, void *deviceAddr)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
    if (deviceAddr != nullptr) {
        aclrtFree(deviceAddr);
    }
}

} // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    int64_t batchSize = 2;
    int64_t numHeadsQ = 4;
    int64_t numHeadsKv = 2;
    int64_t headDim = 128;
    int64_t sparseBlockSizeQ = 128;
    int64_t sparseBlockSizeK = 128;
    int64_t quantMode = 1;
    int64_t maskMode = 3;
    int64_t maxSeqlenQ = 512;
    int64_t maxSeqlenKv = 1024;
    char layoutQ[] = "TND";
    char layoutKv[] = "PA_BNSD";
    char layoutSparseIndices[] = "B_N_Qb_Kb";

    int64_t qbMax = (maxSeqlenQ + sparseBlockSizeQ - 1) / sparseBlockSizeQ;
    std::vector<int64_t> sparseSeqLenShape = {batchSize, numHeadsQ, qbMax};
    std::vector<int32_t> sparseSeqLenHostData(GetShapeSize(sparseSeqLenShape), 0);
    for (int64_t bIdx = 0; bIdx < batchSize; ++bIdx) {
        for (int64_t n1Idx = 0; n1Idx < numHeadsQ; ++n1Idx) {
            for (int64_t qBlockIdx = 0; qBlockIdx < qbMax; ++qBlockIdx) {
                int64_t offset = (bIdx * numHeadsQ + n1Idx) * qbMax + qBlockIdx;
                sparseSeqLenHostData[offset] = static_cast<int32_t>((bIdx + 1) * (qBlockIdx + 1));
            }
        }
    }

    std::vector<int64_t> sequsedQShape = {batchSize};
    std::vector<int64_t> sequsedKvShape = {batchSize};
    std::vector<int32_t> sequsedQHostData = {static_cast<int32_t>(maxSeqlenQ), static_cast<int32_t>(maxSeqlenQ)};
    std::vector<int32_t> sequsedKvHostData = {static_cast<int32_t>(maxSeqlenKv), static_cast<int32_t>(maxSeqlenKv)};

    int64_t metadataSize = optiling::QBSA_HEAD_METADATA_SIZE +
        batchSize * numHeadsQ * optiling::AIC_CORE_NUM * optiling::QBSA_METADATA_SIZE +
        optiling::AIV_CORE_NUM * optiling::FD_METADATA_SIZE;
    int64_t alignedMetadataSize = ((metadataSize + 4095) / 4096) * 4096;
    std::vector<int64_t> metadataShape = {alignedMetadataSize};
    std::vector<uint32_t> metadataHostData(GetShapeSize(metadataShape), 0U);

    void *sparseSeqLenDeviceAddr = nullptr;
    void *sequsedQDeviceAddr = nullptr;
    void *sequsedKvDeviceAddr = nullptr;
    void *metadataDeviceAddr = nullptr;
    aclTensor *sparseSeqLenTensor = nullptr;
    aclTensor *sequsedQTensor = nullptr;
    aclTensor *sequsedKvTensor = nullptr;
    aclTensor *metadataTensor = nullptr;

    ret = CreateAclTensor(sparseSeqLenHostData, sparseSeqLenShape, &sparseSeqLenDeviceAddr,
        aclDataType::ACL_INT32, &sparseSeqLenTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = CreateAclTensor(sequsedQHostData, sequsedQShape, &sequsedQDeviceAddr, aclDataType::ACL_INT32,
        &sequsedQTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = CreateAclTensor(sequsedKvHostData, sequsedKvShape, &sequsedKvDeviceAddr, aclDataType::ACL_INT32,
        &sequsedKvTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32,
        &metadataTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    aclOpExecutor *executor = nullptr;
    uint64_t workspaceSize = 0U;
    void *workspaceAddr = nullptr;

    LOG_PRINT("start aclnnQuantBlockSparseAttnMetadata\n");
    ret = aclnnQuantBlockSparseAttnMetadataGetWorkspaceSize(sparseSeqLenTensor, nullptr, nullptr, sequsedQTensor,
        sequsedKvTensor, batchSize, numHeadsQ, numHeadsKv, headDim, sparseBlockSizeQ, sparseBlockSizeK, quantMode,
        maskMode, maxSeqlenQ, maxSeqlenKv, layoutQ, layoutKv, layoutSparseIndices, metadataTensor, &workspaceSize,
        &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnQuantBlockSparseAttnMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
        return -1;
    }

    if (workspaceSize > 0U) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("workspace aclrtMalloc failed. ERROR: %d\n", ret);
            return -1;
        }
    }

    ret = aclnnQuantBlockSparseAttnMetadata(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnQuantBlockSparseAttnMetadata failed. ERROR: %d\n", ret);
        return -1;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return -1;
    }

    ret = aclrtMemcpy(metadataHostData.data(), metadataHostData.size() * sizeof(metadataHostData[0]),
        metadataDeviceAddr, metadataHostData.size() * sizeof(metadataHostData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("metadata aclrtMemcpy failed. ERROR: %d\n", ret);
        return -1;
    }

    DumpMeta(metadataHostData.data());

    DestroyTensor(metadataTensor, metadataDeviceAddr);
    DestroyTensor(sequsedKvTensor, sequsedKvDeviceAddr);
    DestroyTensor(sequsedQTensor, sequsedQDeviceAddr);
    DestroyTensor(sparseSeqLenTensor, sparseSeqLenDeviceAddr);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
