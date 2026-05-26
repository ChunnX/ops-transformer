/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_block_sparse_attention.cpp
 * \brief BlockSparseAttention 算子测试用例
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <sys/stat.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_block_sparse_attention.h"

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

uint16_t FloatToBf16(float val) {
    uint32_t bits;
    memcpy(&bits, &val, sizeof(float));
    return static_cast<uint16_t>(bits >> 16);
}

float Bf16ToFloat(uint16_t val) {
    uint32_t bits = static_cast<uint32_t>(val) << 16;
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // 固定写法，AscendCL初始化
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
    // 检查shape是否有效
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

    // 检查hostData大小是否匹配
    if (hostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
        LOG_PRINT("CreateAclTensor: ERROR - hostData size mismatch: %zu vs %ld\n", 
                  hostData.size(), GetShapeSize(shape));
        return -1;
    }
    
    // 调用aclrtMalloc申请device侧内存
    *deviceAddr = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
              return ret);
    
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); 
              return ret);
    
    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() > 1) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = nullptr;
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n"); 
            return -1);
    return 0;
}

void FreeResource(aclTensor *query, aclTensor *key, aclTensor *value, aclTensor *blockSparseMask,
                  aclTensor *attentionOut, aclIntArray *actualSeqLengths, aclIntArray *actualSeqLengthsKv,
                  aclIntArray *blockShape, void *queryDeviceAddr, void *keyDeviceAddr, void *valueDeviceAddr,
                  void *blockSparseMaskDeviceAddr, void *attentionOutDeviceAddr, void *actualSeqLengthsDeviceAddr,
                  void *actualSeqLengthsKvDeviceAddr, void *workspaceAddr, int32_t deviceId, aclrtStream *stream)
{
    // 释放资源
    if (query) {
        aclDestroyTensor(query);
    }
    if (key) {
        aclDestroyTensor(key);
    }
    if (value != nullptr) {
        aclDestroyTensor(value);
    }
    if (blockSparseMask) {
        aclDestroyTensor(blockSparseMask);
    }
    if (attentionOut) {
        aclDestroyTensor(attentionOut);
    }
    if (actualSeqLengths) {
        aclDestroyIntArray(actualSeqLengths);
    }
    if (actualSeqLengthsKv) {
        aclDestroyIntArray(actualSeqLengthsKv);
    }
    if (blockShape) {
        aclDestroyIntArray(blockShape);
    }

    if (queryDeviceAddr) {
        aclrtFree(queryDeviceAddr);
    }
    if (keyDeviceAddr) {
        aclrtFree(keyDeviceAddr);
    }
    if (valueDeviceAddr) {
        aclrtFree(valueDeviceAddr);
    }
    if (blockSparseMaskDeviceAddr) {
        aclrtFree(blockSparseMaskDeviceAddr);
    }
    if (attentionOutDeviceAddr) {
        aclrtFree(attentionOutDeviceAddr);
    }
    if (actualSeqLengthsDeviceAddr) {
        aclrtFree(actualSeqLengthsDeviceAddr);
    }
    if (actualSeqLengthsKvDeviceAddr) {
        aclrtFree(actualSeqLengthsKvDeviceAddr);
    }
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

enum SparsityMode {
    SPARSITY_NONE = 0,    // 0% 稀疏度：所有 blocks 全部激活（全稠密）
    SPARSITY_HIGH = 1,    // ~99.8% 稀疏度：仅对角线+次对角线
    SPARSITY_MEDIUM = 2,  // ~50% 稀疏度：下三角 causal mask
    SPARSITY_LOW = 3      // ~25% 稀疏度：75% blocks 激活
};

enum InputFillMode {
    INPUT_FILL_ALL_ONE = 0,       // Q/K/V 全部为 1.0
    INPUT_FILL_RANDOM_BINARY = 1  // Q/K/V 每个元素随机取 0 或 1（固定 seed 可复现）
};

uint32_t MixBits(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

void FillQkvHostData(std::vector<uint16_t>& data, InputFillMode mode, uint32_t randomSeed, uint32_t streamId) {
    if (mode == INPUT_FILL_ALL_ONE) {
        const uint16_t one = FloatToBf16(1.0f);
        std::fill(data.begin(), data.end(), one);
        return;
    }

    const uint32_t base = randomSeed + streamId * 1000003u;
    for (size_t i = 0; i < data.size(); ++i) {
        const uint32_t bit = MixBits(base + static_cast<uint32_t>(i)) & 1u;
        data[i] = FloatToBf16(static_cast<float>(bit));
    }
}

void FillBlockSparseMask(std::vector<int8_t>& mask, int32_t batch, int32_t numHeads,
                         int32_t qBlockNum, int32_t kvBlockNum, SparsityMode mode) {
    std::fill(mask.begin(), mask.end(), 0);
    for (int32_t b = 0; b < batch; b++) {
        for (int32_t h = 0; h < numHeads; h++) {
            int64_t baseOffset = ((int64_t)b * numHeads + h) * qBlockNum * kvBlockNum;
            switch (mode) {
            case SPARSITY_NONE:
                for (int32_t i = 0; i < qBlockNum; i++) {
                    int64_t rowOffset = baseOffset + (int64_t)i * kvBlockNum;
                    for (int32_t j = 0; j < kvBlockNum; j++) {
                        mask[rowOffset + j] = 1;
                    }
                }
                break;
            case SPARSITY_HIGH:
                for (int32_t i = 0; i < qBlockNum; i++) {
                    int64_t rowOffset = baseOffset + (int64_t)i * kvBlockNum;
                    mask[rowOffset + i] = 1;
                    if (i > 0) {
                        mask[rowOffset + i - 1] = 1;
                    }
                }
                break;
            case SPARSITY_MEDIUM:
                for (int32_t i = 0; i < qBlockNum; i++) {
                    int64_t rowOffset = baseOffset + (int64_t)i * kvBlockNum;
                    for (int32_t j = 0; j <= i && j < kvBlockNum; j++) {
                        mask[rowOffset + j] = 1;
                    }
                }
                break;
            case SPARSITY_LOW:
                for (int32_t i = 0; i < qBlockNum; i++) {
                    int64_t rowOffset = baseOffset + (int64_t)i * kvBlockNum;
                    for (int32_t j = 0; j < kvBlockNum; j++) {
                        if (j <= i || (j % 4 != 3)) {
                            mask[rowOffset + j] = 1;
                        }
                    }
                }
                break;
            }
        }
    }
}

bool PathExists(const std::string& path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0;
}

std::string ResolveDefaultOutputDir() {
    if (const char* envDir = std::getenv("BSA_VERIFY_OUTPUT_DIR")) {
        return std::string(envDir);
    }
    // build.sh --run_example 在 ops-transformer/build/ 下执行可执行文件
    if (PathExists("../attention/block_sparse_attention")) {
        return "../attention/block_sparse_attention/py_verify/bsa_output";
    }
    // 在 examples/ 目录下手动执行
    if (PathExists("../py_verify")) {
        return "../py_verify/bsa_output";
    }
    return "./bsa_output";
}

bool EnsureDirExists(const std::string& dirPath) {
    struct stat st {};
    if (stat(dirPath.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }

    const size_t slashPos = dirPath.find_last_of('/');
    if (slashPos != std::string::npos && slashPos > 0) {
        const std::string parent = dirPath.substr(0, slashPos);
        if (!EnsureDirExists(parent)) {
            return false;
        }
    }
    return mkdir(dirPath.c_str(), 0755) == 0;
}

bool WriteBinaryFile(const std::string& filePath, const void* data, size_t byteSize) {
    std::ofstream out(filePath, std::ios::binary);
    if (!out.is_open()) {
        LOG_PRINT("Failed to open file for writing: %s\n", filePath.c_str());
        return false;
    }
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(byteSize));
    if (!out.good()) {
        LOG_PRINT("Failed to write file: %s\n", filePath.c_str());
        return false;
    }
    return true;
}

std::vector<float> Bf16HostToFloat32(const std::vector<uint16_t>& data) {
    std::vector<float> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result[i] = Bf16ToFloat(data[i]);
    }
    return result;
}

bool WriteMetadataJson(const std::string& filePath, int32_t batch, int32_t qSeqlen, int32_t kvSeqlen,
                       int32_t numHeads, int32_t numKvHeads, int32_t headDim, int32_t blockShapeX,
                       int32_t blockShapeY, int32_t qBlockNum, int32_t kvBlockNum, SparsityMode sparsityMode,
                       float scaleValue, int64_t innerPrecise, int64_t maskType, int64_t blockSize,
                       int64_t preTokens, int64_t nextTokens,
                       const std::vector<int64_t>& queryShape, const std::vector<int64_t>& keyShape,
                       const std::vector<int64_t>& valueShape, const std::vector<int64_t>& blockSparseMaskShape,
                       const std::vector<int64_t>& attentionOutShape) {
    std::ofstream out(filePath);
    if (!out.is_open()) {
        LOG_PRINT("Failed to open metadata file for writing: %s\n", filePath.c_str());
        return false;
    }

    out << "{\n";
    out << "  \"batch\": " << batch << ",\n";
    out << "  \"q_seqlen\": " << qSeqlen << ",\n";
    out << "  \"kv_seqlen\": " << kvSeqlen << ",\n";
    out << "  \"num_heads\": " << numHeads << ",\n";
    out << "  \"num_kv_heads\": " << numKvHeads << ",\n";
    out << "  \"head_dim\": " << headDim << ",\n";
    out << "  \"block_shape_x\": " << blockShapeX << ",\n";
    out << "  \"block_shape_y\": " << blockShapeY << ",\n";
    out << "  \"q_block_num\": " << qBlockNum << ",\n";
    out << "  \"kv_block_num\": " << kvBlockNum << ",\n";
    out << "  \"sparsity_mode\": " << static_cast<int>(sparsityMode) << ",\n";
    out << "  \"q_layout\": \"TND\",\n";
    out << "  \"kv_layout\": \"TND\",\n";
    out << "  \"scale_value\": " << scaleValue << ",\n";
    out << "  \"inner_precise\": " << innerPrecise << ",\n";
    out << "  \"mask_type\": " << maskType << ",\n";
    out << "  \"block_size\": " << blockSize << ",\n";
    out << "  \"pre_tokens\": " << preTokens << ",\n";
    out << "  \"next_tokens\": " << nextTokens << ",\n";
    out << "  \"input_dtype\": \"float32\",\n";
    out << "  \"query_file\": \"query.bin\",\n";
    out << "  \"query_shape\": [";
    for (size_t i = 0; i < queryShape.size(); ++i) {
        out << queryShape[i];
        if (i + 1 < queryShape.size()) {
            out << ", ";
        }
    }
    out << "],\n";
    out << "  \"key_file\": \"key.bin\",\n";
    out << "  \"key_shape\": [";
    for (size_t i = 0; i < keyShape.size(); ++i) {
        out << keyShape[i];
        if (i + 1 < keyShape.size()) {
            out << ", ";
        }
    }
    out << "],\n";
    out << "  \"value_file\": \"value.bin\",\n";
    out << "  \"value_shape\": [";
    for (size_t i = 0; i < valueShape.size(); ++i) {
        out << valueShape[i];
        if (i + 1 < valueShape.size()) {
            out << ", ";
        }
    }
    out << "],\n";
    out << "  \"block_sparse_mask_file\": \"block_sparse_mask.bin\",\n";
    out << "  \"block_sparse_mask_dtype\": \"int8\",\n";
    out << "  \"block_sparse_mask_shape\": [";
    for (size_t i = 0; i < blockSparseMaskShape.size(); ++i) {
        out << blockSparseMaskShape[i];
        if (i + 1 < blockSparseMaskShape.size()) {
            out << ", ";
        }
    }
    out << "],\n";
    out << "  \"output_dtype\": \"float32\",\n";
    out << "  \"output_file\": \"attention_out.bin\",\n";
    out << "  \"output_shape\": [";
    for (size_t i = 0; i < attentionOutShape.size(); ++i) {
        out << attentionOutShape[i];
        if (i + 1 < attentionOutShape.size()) {
            out << ", ";
        }
    }
    out << "]\n";
    out << "}\n";
    return out.good();
}

bool SaveVerifyArtifacts(const std::string& outputDir,
                         const std::vector<uint16_t>& queryHostData,
                         const std::vector<uint16_t>& keyHostData,
                         const std::vector<uint16_t>& valueHostData,
                         const std::vector<int8_t>& blockSparseMaskHostData,
                         const std::vector<float>& resultData, int32_t batch, int32_t qSeqlen, int32_t kvSeqlen,
                         int32_t numHeads, int32_t numKvHeads, int32_t headDim, int32_t blockShapeX,
                         int32_t blockShapeY, int32_t qBlockNum, int32_t kvBlockNum, SparsityMode sparsityMode,
                         float scaleValue, int64_t innerPrecise, int64_t maskType, int64_t blockSize,
                         int64_t preTokens, int64_t nextTokens, const std::vector<int64_t>& queryShape,
                         const std::vector<int64_t>& keyShape, const std::vector<int64_t>& valueShape,
                         const std::vector<int64_t>& blockSparseMaskShape,
                         const std::vector<int64_t>& attentionOutShape) {
    if (!EnsureDirExists(outputDir)) {
        LOG_PRINT("Failed to create output directory: %s\n", outputDir.c_str());
        return false;
    }

    const std::vector<float> queryFloat = Bf16HostToFloat32(queryHostData);
    const std::vector<float> keyFloat = Bf16HostToFloat32(keyHostData);
    const std::vector<float> valueFloat = Bf16HostToFloat32(valueHostData);

    const std::string queryPath = outputDir + "/query.bin";
    const std::string keyPath = outputDir + "/key.bin";
    const std::string valuePath = outputDir + "/value.bin";
    const std::string maskPath = outputDir + "/block_sparse_mask.bin";
    const std::string outputPath = outputDir + "/attention_out.bin";
    const std::string metaPath = outputDir + "/metadata.json";

    if (!WriteBinaryFile(queryPath, queryFloat.data(), queryFloat.size() * sizeof(float))) {
        return false;
    }
    if (!WriteBinaryFile(keyPath, keyFloat.data(), keyFloat.size() * sizeof(float))) {
        return false;
    }
    if (!WriteBinaryFile(valuePath, valueFloat.data(), valueFloat.size() * sizeof(float))) {
        return false;
    }
    if (!WriteBinaryFile(maskPath, blockSparseMaskHostData.data(),
                         blockSparseMaskHostData.size() * sizeof(int8_t))) {
        return false;
    }
    if (!WriteBinaryFile(outputPath, resultData.data(), resultData.size() * sizeof(float))) {
        return false;
    }
    if (!WriteMetadataJson(metaPath, batch, qSeqlen, kvSeqlen, numHeads, numKvHeads, headDim, blockShapeX,
                           blockShapeY, qBlockNum, kvBlockNum, sparsityMode, scaleValue, innerPrecise, maskType,
                           blockSize, preTokens, nextTokens, queryShape, keyShape, valueShape, blockSparseMaskShape,
                           attentionOutShape)) {
        return false;
    }

    LOG_PRINT("Saved query to %s (%zu bytes)\n", queryPath.c_str(), queryFloat.size() * sizeof(float));
    LOG_PRINT("Saved key to %s (%zu bytes)\n", keyPath.c_str(), keyFloat.size() * sizeof(float));
    LOG_PRINT("Saved value to %s (%zu bytes)\n", valuePath.c_str(), valueFloat.size() * sizeof(float));
    LOG_PRINT("Saved block_sparse_mask to %s (%zu bytes)\n", maskPath.c_str(),
              blockSparseMaskHostData.size() * sizeof(int8_t));
    LOG_PRINT("Saved attentionOut to %s (%zu bytes)\n", outputPath.c_str(), resultData.size() * sizeof(float));
    LOG_PRINT("Saved metadata to %s\n", metaPath.c_str());
    return true;
}

int main(int argc, char* argv[]) {
    // 1. （固定写法）device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // ===================== 稀疏度模式开关 =====================
    // SPARSITY_NONE   (0): 0% 稀疏度，所有 blocks 全部激活（全稠密，等价于标准 attention）
    // SPARSITY_HIGH   (1): ~99.8% 稀疏度，仅对角线+次对角线激活
    // SPARSITY_MEDIUM (2): ~50% 稀疏度，下三角 causal mask
    // SPARSITY_LOW    (3): ~25% 稀疏度，75% blocks 激活
    SparsityMode sparsityMode = SPARSITY_NONE;
    // =========================================================

    // ===================== QKV 输入填充模式 =====================
    // INPUT_FILL_ALL_ONE        (0): Q/K/V 全部为 1.0
    // INPUT_FILL_RANDOM_BINARY  (1): Q/K/V 每个元素随机取 0 或 1（固定 seed，CPU 参考可复现）
    InputFillMode inputFillMode = INPUT_FILL_RANDOM_BINARY;
    constexpr uint32_t kInputRandomSeed = 42;
    // =========================================================

    // 与 Python CPU 参考脚本对比验证时使用较小规模；大规模压测请改为 false
    constexpr bool kCpuVerifyMode = true;

    // 结果落盘目录：命令行 > 环境变量 BSA_VERIFY_OUTPUT_DIR > 按 cwd 自动推断
    const std::string outputDir = (argc > 1) ? argv[1] : ResolveDefaultOutputDir();
    LOG_PRINT("Verify output directory: %s\n", outputDir.c_str());

    // 2. 设置参数
    int32_t batch = 1;
    int32_t qSeqlen = kCpuVerifyMode ? 1024 : 1048576;
    int32_t kvSeqlen = kCpuVerifyMode ? 1024 : 1048576;
    int32_t numHeads = 1;
    int32_t numKvHeads = 1;
    int32_t headDim = 128;
    int32_t blockShapeX = 128;
    int32_t blockShapeY = 128;
    
    // 计算TND格式维度
    int64_t totalQTokens = batch * qSeqlen;
    int64_t totalKvTokens = batch * kvSeqlen;
    int32_t qBlockNum = (qSeqlen + blockShapeX - 1) / blockShapeX;  // Q块的X维度数量
    int32_t kvBlockNum = (kvSeqlen + blockShapeY - 1) / blockShapeY;  // KV块的Y维度数量
    // totalQBlocks = qBlockNum * numHeads (每个Q块对应一个head)
    int32_t totalQBlocks = qBlockNum * numHeads;
    int32_t maxKvBlockNum = kvBlockNum;
    
    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *valueTensor = nullptr;
    aclTensor *blockSparseMaskTensor = nullptr;
    aclTensor *attentionOutTensor = nullptr;
    aclIntArray *actualSeqLengths = nullptr;
    aclIntArray *actualSeqLengthsKv = nullptr;
    aclIntArray *blockShape = nullptr;

    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *blockSparseMaskDeviceAddr = nullptr;
    void *attentionOutDeviceAddr = nullptr;
    void *actualSeqLengthsDeviceAddr = nullptr;
    void *actualSeqLengthsKvDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;

    // 3. 创建Query tensor (TND format: [totalQTokens, numHeads, headDim])
    std::vector<int64_t> queryShape = {totalQTokens, numHeads, headDim};
    std::vector<uint16_t> queryHostData(totalQTokens * numHeads * headDim);
    FillQkvHostData(queryHostData, inputFillMode, kInputRandomSeed, 0);
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_BF16, &queryTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create query tensor\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 4. 创建Key/Value tensor (TND format: [totalKvTokens, numKvHeads, headDim])
    std::vector<int64_t> kvShape = {totalKvTokens, numKvHeads, headDim};
    std::vector<uint16_t> keyHostData(totalKvTokens * numKvHeads * headDim);
    std::vector<uint16_t> valueHostData(totalKvTokens * numKvHeads * headDim);
    FillQkvHostData(keyHostData, inputFillMode, kInputRandomSeed, 1);
    FillQkvHostData(valueHostData, inputFillMode, kInputRandomSeed, 2);
    LOG_PRINT("Input fill mode: %d, random seed: %u, dtype: bf16\n", static_cast<int>(inputFillMode), kInputRandomSeed);
    ret = CreateAclTensor(keyHostData, kvShape, &keyDeviceAddr, aclDataType::ACL_BF16, &keyTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create key tensor\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);
    ret = CreateAclTensor(valueHostData, kvShape, &valueDeviceAddr, aclDataType::ACL_BF16, &valueTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create value tensor\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 5. 创建blockSparseMask tensor ([batch, numHeads, qBlockNum, kvBlockNum])
    int64_t blockSparseMaskSize = static_cast<int64_t>(batch) * numHeads * qBlockNum * kvBlockNum;
    std::vector<int8_t> blockSparseMaskHostData(blockSparseMaskSize, 0);
    FillBlockSparseMask(blockSparseMaskHostData, batch, numHeads, qBlockNum, kvBlockNum, sparsityMode);
    
    int64_t activeBlocks = 0;
    for (auto v : blockSparseMaskHostData) { if (v) activeBlocks++; }
    double density = static_cast<double>(activeBlocks) / blockSparseMaskSize * 100.0;
    LOG_PRINT("Sparsity mode: %d, mask size: %ld, active blocks: %ld, density: %.2f%%, sparsity: %.2f%%\n",
              static_cast<int>(sparsityMode), blockSparseMaskSize, activeBlocks, density, 100.0 - density);
    std::vector<int64_t> blockSparseMaskShape = {batch, numHeads, qBlockNum, kvBlockNum};
    ret = CreateAclTensor(blockSparseMaskHostData, blockSparseMaskShape, &blockSparseMaskDeviceAddr, aclDataType::ACL_INT8, &blockSparseMaskTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create block sparse mask tensor\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 6. 创建输出tensor
    std::vector<int64_t> attentionOutShape = {totalQTokens, numHeads, headDim};
    int64_t attentionOutElementCount = totalQTokens * numHeads * headDim;
    std::vector<uint16_t> attentionOutHostData(attentionOutElementCount, 0);
    ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_BF16, &attentionOutTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create attentionOut tensor\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 7. 创建blockShape数组
    std::vector<int64_t> blockShapeData = {blockShapeX, blockShapeY};
    blockShape = aclCreateIntArray(blockShapeData.data(), blockShapeData.size());
    CHECK_RET(blockShape != nullptr, LOG_PRINT("Failed to create blockShape array\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return -1);

    // 8. 创建actualSeqLengths和actualSeqLengthsKv (必需参数)
    std::vector<int64_t> actualSeqLengthsHost(batch, static_cast<int64_t>(qSeqlen));
    std::vector<int64_t> actualSeqLengthsKvHost(batch, static_cast<int64_t>(kvSeqlen));
    
    size_t seqLengthsSize = batch * sizeof(int64_t);

    ret = aclrtMalloc(&actualSeqLengthsDeviceAddr, seqLengthsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to allocate actualSeqLengths memory\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);
    ret = aclrtMalloc(&actualSeqLengthsKvDeviceAddr, seqLengthsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to allocate actualSeqLengthsKv memory\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    ret = aclrtMemcpy(actualSeqLengthsDeviceAddr, seqLengthsSize, actualSeqLengthsHost.data(), seqLengthsSize,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to copy actualSeqLengths to device\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);
    ret = aclrtMemcpy(actualSeqLengthsKvDeviceAddr, seqLengthsSize, actualSeqLengthsKvHost.data(), seqLengthsSize,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to copy actualSeqLengthsKv to device\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // aclCreateIntArray 期望的是 host 侧的数据指针，而不是 device 侧的数据
    actualSeqLengths = aclCreateIntArray(actualSeqLengthsHost.data(), batch);
    actualSeqLengthsKv = aclCreateIntArray(actualSeqLengthsKvHost.data(), batch);
    CHECK_RET(actualSeqLengths != nullptr && actualSeqLengthsKv != nullptr,
              LOG_PRINT("Failed to create actualSeqLengths arrays\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return -1);

    // 9. 准备字符串参数（确保缓冲区大小足够，包含null terminator）
    const char* qLayoutStr = "TND";
    const char* kvLayoutStr = "TND";
    char qLayoutBuffer[16] = {0};
    char kvLayoutBuffer[16] = {0};
    strncpy(qLayoutBuffer, qLayoutStr, sizeof(qLayoutBuffer) - 1);
    strncpy(kvLayoutBuffer, kvLayoutStr, sizeof(kvLayoutBuffer) - 1);
    
    // 10. 计算scaleValue
    float scaleValue = 1.0f / std::sqrt(static_cast<float>(headDim));
    const int64_t innerPrecise = 4;  // ascend950 must use 4=LOW_HIGH_MIXED; 910b use 0 or 1
    const int64_t maskType = 0;
    const int64_t blockSize = 0;
    const int64_t preTokens = 2147483647;
    const int64_t nextTokens = 2147483647;
    const float inputFillValue = 1.0f;
    
    // 11. 调用第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    
    ret = aclnnBlockSparseAttentionGetWorkspaceSize(
        queryTensor,           // query
        keyTensor,             // key
        valueTensor,           // value
        blockSparseMaskTensor, // blockSparseMask
        nullptr,               // attenMaskOptional
        blockShape,            // blockShape
        actualSeqLengths,      // actualSeqLengthsOptional
        actualSeqLengthsKv,    // actualSeqLengthsKvOptional
        nullptr,               // blockTableOptional
        qLayoutBuffer,         // qInputLayout
        kvLayoutBuffer,        // kvInputLayout
        numKvHeads,            // numKeyValueHeads
        maskType,              // maskType
        scaleValue,            // scaleValue
        innerPrecise,          // innerPrecise
        blockSize,             // blockSize
        preTokens,             // preTokens
        nextTokens,            // nextTokens
        0,                     // softmaxLseFlag
        attentionOutTensor,          // attentionOut
        nullptr,               // softmaxLseOptional
        &workspaceSize,        // workspaceSize (out)
        &executor);            // executor (out)

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockSparseAttentionGetWorkspaceSize failed. ERROR: %d\n", ret);
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);
    CHECK_RET(executor != nullptr, LOG_PRINT("executor is null after GetWorkspaceSize\n");
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return -1);

    // 12. 分配workspace
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); 
                      FreeResource(
                      queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor, actualSeqLengths,
                      actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr, valueDeviceAddr,
                      blockSparseMaskDeviceAddr, attentionOutDeviceAddr, actualSeqLengthsDeviceAddr,
                      actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
                  return ret);
    }

    // 12. 调用第二段接口
    ret = aclnnBlockSparseAttention(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockSparseAttention failed. ERROR: %d\n", ret);
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 13. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 14. 获取输出的值，将device侧内存上的结果拷贝至host侧
    int64_t attentionOutSize = GetShapeSize(attentionOutShape);
    std::vector<uint16_t> resultData(attentionOutSize, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(uint16_t), attentionOutDeviceAddr,
                      attentionOutSize * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor,
                           actualSeqLengths, actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr,
                           valueDeviceAddr, blockSparseMaskDeviceAddr, attentionOutDeviceAddr,
                           actualSeqLengthsDeviceAddr, actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 15. 将完整结果转为 float32 落盘，并写入 metadata 供 Python CPU 参考脚本复现与对比
    std::vector<float> resultFloat(resultData.size());
    for (size_t i = 0; i < resultData.size(); ++i) {
        resultFloat[i] = Bf16ToFloat(resultData[i]);
    }
    if (!SaveVerifyArtifacts(outputDir, queryHostData, keyHostData, valueHostData, blockSparseMaskHostData,
                             resultFloat, batch, qSeqlen, kvSeqlen, numHeads, numKvHeads, headDim, blockShapeX,
                             blockShapeY, qBlockNum, kvBlockNum, sparsityMode, scaleValue, innerPrecise, maskType,
                             blockSize, preTokens, nextTokens, queryShape, kvShape, kvShape, blockSparseMaskShape,
                             attentionOutShape)) {
        FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor, actualSeqLengths,
                     actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr, valueDeviceAddr,
                     blockSparseMaskDeviceAddr, attentionOutDeviceAddr, actualSeqLengthsDeviceAddr,
                     actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
        return -1;
    }

    uint64_t printNum = 10;
    LOG_PRINT("attentionOut preview (first %lu elements):\n", printNum);
    for (uint64_t i = 0; i < printNum && i < resultData.size(); i++) {
        LOG_PRINT("  index %lu: %f\n", i, Bf16ToFloat(resultData[i]));
    }

    // 16. 释放资源
    FreeResource(queryTensor, keyTensor, valueTensor, blockSparseMaskTensor, attentionOutTensor, actualSeqLengths,
                 actualSeqLengthsKv, blockShape, queryDeviceAddr, keyDeviceAddr, valueDeviceAddr,
                 blockSparseMaskDeviceAddr, attentionOutDeviceAddr, actualSeqLengthsDeviceAddr,
                 actualSeqLengthsKvDeviceAddr, workspaceAddr, deviceId, &stream);
    LOG_PRINT("Test completed successfully!\n");
    return 0;
}