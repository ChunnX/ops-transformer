#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_bsa_select_block_mask.h"

using namespace std;

#define CHECK_RET(cond, return_expr) \
    do {                               \
        if (!(cond)) {                   \
            return_expr;                   \
        }                                \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                              \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    if (shape.empty()) {
        LOG_PRINT("CreateAclTensor: ERROR - shape is empty\n");
        return -1;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            LOG_PRINT("CreateAclTensor: ERROR - shape[%zu]=%ld is invalid\n", i, shape[i]);
            return -1;
        }
    }
    
    auto size = GetShapeSize(shape) * sizeof(T);

    if (hostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
        LOG_PRINT("CreateAclTensor: ERROR - hostData size mismatch: %zu vs %ld\n", 
                  hostData.size(), GetShapeSize(shape));
        return -1;
    }
    
    *deviceAddr = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() > 1) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    *tensor = nullptr;
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n"); return -1);
    return 0;
}

void FreeResource(aclTensor *query, aclTensor *key, aclTensor *maskOut,
                  void *queryDeviceAddr, void *keyDeviceAddr, void *maskOutDeviceAddr,
                  void *workspaceAddr, int32_t deviceId, aclrtStream *stream)
{
    if (query) aclDestroyTensor(query);
    if (key) aclDestroyTensor(key);
    if (maskOut) aclDestroyTensor(maskOut);

    if (queryDeviceAddr) aclrtFree(queryDeviceAddr);
    if (keyDeviceAddr) aclrtFree(keyDeviceAddr);
    if (maskOutDeviceAddr) aclrtFree(maskOutDeviceAddr);
    if (workspaceAddr) aclrtFree(workspaceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main() {
    LOG_PRINT("=== BSASelectBlockMask ACLNN Test ===\n");
    
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int32_t batch = 1;
    int32_t numHeads = 8;
    int32_t qSeqlen = 4096;
    int32_t kvSeqlen = 4096;
    int32_t headDim = 128;
    int32_t blockShapeX = 128;
    int32_t blockShapeY = 128;
    float sparsity = 0.5f;
    
    int32_t xBlocks = (qSeqlen + blockShapeX - 1) / blockShapeX;
    int32_t yBlocks = (kvSeqlen + blockShapeY - 1) / blockShapeY;
    
    LOG_PRINT("Test parameters:\n");
    LOG_PRINT("  batch=%d, numHeads=%d, qSeqlen=%d, kvSeqlen=%d, headDim=%d\n", 
              batch, numHeads, qSeqlen, kvSeqlen, headDim);
    LOG_PRINT("  blockShape=[%d, %d], xBlocks=%d, yBlocks=%d, sparsity=%.2f\n",
              blockShapeX, blockShapeY, xBlocks, yBlocks, sparsity);

    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *maskOutTensor = nullptr;
    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *maskOutDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;

    std::vector<int64_t> queryShape = {batch, numHeads, qSeqlen, headDim};
    std::vector<op::fp16_t> queryHostData(batch * numHeads * qSeqlen * headDim, 1.0f);
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create query tensor\n");
              FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                           maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return ret);

    std::vector<int64_t> keyShape = {batch, numHeads, kvSeqlen, headDim};
    std::vector<op::fp16_t> keyHostData(batch * numHeads * kvSeqlen * headDim, 1.0f);
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create key tensor\n");
              FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                           maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return ret);

    std::vector<int64_t> maskOutShape = {batch, numHeads, xBlocks, yBlocks};
    int64_t maskOutSize = batch * numHeads * xBlocks * yBlocks;
    std::vector<int8_t> maskOutHostData(maskOutSize, 0);
    ret = CreateAclTensor(maskOutHostData, maskOutShape, &maskOutDeviceAddr, aclDataType::ACL_INT8, &maskOutTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create maskOut tensor\n");
              FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                           maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return ret);

    std::vector<int64_t> blockShapeData = {blockShapeX, blockShapeY};
    aclIntArray *blockShape = aclCreateIntArray(blockShapeData.data(), blockShapeData.size());
    CHECK_RET(blockShape != nullptr, LOG_PRINT("Failed to create blockShape array\n");
              FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                           maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return -1);

    char qLayoutBuffer[16] = "BNSD";
    char kvLayoutBuffer[16] = "BNSD";
    int64_t numKeyValueHeads = numHeads;
    double scaleValue = 1.0 / std::sqrt(static_cast<double>(headDim));

    LOG_PRINT("Calling aclnnBSASelectBlockMaskGetWorkspaceSize...\n");
    
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    
    ret = aclnnBSASelectBlockMaskGetWorkspaceSize(
        queryTensor,
        keyTensor,
        blockShape,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        qLayoutBuffer,
        kvLayoutBuffer,
        numKeyValueHeads,
        scaleValue,
        static_cast<double>(sparsity),
        maskOutTensor,
        &workspaceSize,
        &executor);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBSASelectBlockMaskGetWorkspaceSize failed. ERROR: %d\n", ret);
              aclDestroyIntArray(blockShape);
              FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                           maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return ret);
    CHECK_RET(executor != nullptr, LOG_PRINT("executor is null after GetWorkspaceSize\n");
              aclDestroyIntArray(blockShape);
              FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                           maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return -1);

    LOG_PRINT("workspaceSize = %lu bytes (%.2f MB)\n", workspaceSize, workspaceSize / 1024.0 / 1024.0);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  aclDestroyIntArray(blockShape);
                  FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                               maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return ret);
    }

    LOG_PRINT("Calling aclnnBSASelectBlockMask...\n");
    
    ret = aclnnBSASelectBlockMask(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBSASelectBlockMask failed. ERROR: %d\n", ret);
              aclDestroyIntArray(blockShape);
              FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                           maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return ret);

    LOG_PRINT("Synchronizing stream...\n");
    
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              aclDestroyIntArray(blockShape);
              FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                           maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return ret);

    LOG_PRINT("Copying result back to host...\n");
    
    std::vector<int8_t> resultData(maskOutSize, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(int8_t), maskOutDeviceAddr,
                      resultData.size() * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              aclDestroyIntArray(blockShape);
              FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                           maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return ret);

    int64_t selectedCount = 0;
    for (int64_t i = 0; i < maskOutSize; i++) {
        if (resultData[i] != 0) {
            selectedCount++;
        }
    }
    
    double actualSparsity = 1.0 - static_cast<double>(selectedCount) / static_cast<double>(maskOutSize);
    
    LOG_PRINT("\n=== Results ===\n");
    LOG_PRINT("maskOut shape: [%d, %d, %d, %d]\n", batch, numHeads, xBlocks, yBlocks);
    LOG_PRINT("Total elements: %ld\n", maskOutSize);
    LOG_PRINT("Selected (non-zero): %ld\n", selectedCount);
    LOG_PRINT("Actual sparsity: %.4f (expected: %.4f)\n", actualSparsity, sparsity);
    
    uint64_t printNum = 20;
    LOG_PRINT("First %lu elements:\n", printNum);
    for (uint64_t i = 0; i < printNum && i < resultData.size(); i++) {
        LOG_PRINT("  index %lu: %d\n", i, static_cast<int>(resultData[i]));
    }

    aclDestroyIntArray(blockShape);
    FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                 maskOutDeviceAddr, workspaceAddr, deviceId, &stream);
    
    LOG_PRINT("\n=== Test completed successfully! ===\n");
    return 0;
}
