/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Standalone Ascend950 ACLNN regression.  Build it against an installed custom
// package; the command and expected coverage are recorded in the 950 report.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_moe_fused_topk.h"

namespace {

enum class InputType { kFp32, kFp16, kBf16 };

struct TestCase {
    const char *name;
    InputType inputType;
    int64_t tokenNum;
    int64_t expertNum;
    uint32_t groupNum;
    uint32_t groupTopk;
    uint32_t topN;
    uint32_t topK;
    bool isNorm;
    float scale;
    bool enableExpertMapping;
};

struct Bfloat16Storage {
    uint16_t bits;
};

struct DeviceTensor {
    void *deviceAddr = nullptr;
    aclTensor *tensor = nullptr;
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;

    ~DeviceTensor()
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
        if (deviceAddr != nullptr) {
            aclrtFree(deviceAddr);
        }
    }
};

template <typename T>
bool CreateTensor(DeviceTensor *result, const std::vector<T> &hostData, const std::vector<int64_t> &shape,
                  aclDataType dataType)
{
    result->shape = shape;
    result->stride.assign(shape.size(), 1);
    for (int64_t dim = static_cast<int64_t>(shape.size()) - 2; dim >= 0; --dim) {
        result->stride[dim] = result->stride[dim + 1] * shape[dim + 1];
    }
    const size_t bytes = hostData.size() * sizeof(T);
    if (aclrtMalloc(&result->deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return false;
    }
    if (aclrtMemcpy(result->deviceAddr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        return false;
    }
    result->tensor = aclCreateTensor(result->shape.data(), result->shape.size(), dataType, result->stride.data(), 0,
                                     ACL_FORMAT_ND, result->shape.data(), result->shape.size(), result->deviceAddr);
    return result->tensor != nullptr;
}

uint16_t FloatToBfloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    // Round to nearest even before discarding the low 16 bits.
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>(bits >> 16U);
}

float Bfloat16ToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

template <typename T>
T FromFloat(float value);

template <>
float FromFloat<float>(float value)
{
    return value;
}

template <>
aclFloat16 FromFloat<aclFloat16>(float value)
{
    return aclFloatToFloat16(value);
}

template <>
Bfloat16Storage FromFloat<Bfloat16Storage>(float value)
{
    return {FloatToBfloat16(value)};
}

template <typename T>
float ToFloat(T value);

template <>
float ToFloat<float>(float value)
{
    return value;
}

template <>
float ToFloat<aclFloat16>(aclFloat16 value)
{
    return aclFloat16ToFloat(value);
}

template <>
float ToFloat<Bfloat16Storage>(Bfloat16Storage value)
{
    return Bfloat16ToFloat(value.bits);
}

std::vector<int32_t> StableDescendingIndices(const std::vector<float> &values)
{
    std::vector<int32_t> indices(values.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = static_cast<int32_t>(i);
    }
    std::stable_sort(indices.begin(), indices.end(), [&values](int32_t lhs, int32_t rhs) {
        return values[lhs] > values[rhs];
    });
    return indices;
}

void CalculateGolden(const TestCase &testCase, const std::vector<float> &x, const std::vector<float> &addNum,
                     const std::vector<int32_t> &mappingNum, const std::vector<int32_t> &mappingTable,
                     std::vector<float> *y, std::vector<int32_t> *indices)
{
    const int64_t groupElements = testCase.expertNum / testCase.groupNum;
    y->clear();
    indices->clear();
    y->reserve(testCase.tokenNum * testCase.topK);
    indices->reserve(testCase.tokenNum * testCase.topK);
    for (int64_t token = 0; token < testCase.tokenNum; ++token) {
        std::vector<float> sigmoid(testCase.expertNum);
        std::vector<float> score(testCase.expertNum);
        for (int64_t expert = 0; expert < testCase.expertNum; ++expert) {
            const float input = x[token * testCase.expertNum + expert];
            sigmoid[expert] = 1.0F / (1.0F + std::exp(-input));
            score[expert] = sigmoid[expert] + addNum[expert];
        }

        std::vector<float> groupScore(testCase.groupNum);
        for (uint32_t group = 0; group < testCase.groupNum; ++group) {
            const int64_t begin = static_cast<int64_t>(group) * groupElements;
            std::vector<float> groupValues(score.begin() + begin, score.begin() + begin + groupElements);
            const auto localOrder = StableDescendingIndices(groupValues);
            for (uint32_t rank = 0; rank < testCase.topN; ++rank) {
                groupScore[group] += groupValues[localOrder[rank]];
            }
        }

        // This intentionally mirrors the established kernel/Python-Golden behavior:
        // unselected groups are zeroed before the final global stable TopK.
        const auto groupOrder = StableDescendingIndices(groupScore);
        std::vector<bool> selectedGroup(testCase.groupNum, false);
        for (uint32_t rank = 0; rank < testCase.groupTopk; ++rank) {
            selectedGroup[groupOrder[rank]] = true;
        }
        for (uint32_t group = 0; group < testCase.groupNum; ++group) {
            if (selectedGroup[group]) {
                continue;
            }
            const int64_t begin = static_cast<int64_t>(group) * groupElements;
            std::fill(score.begin() + begin, score.begin() + begin + groupElements, 0.0F);
        }

        const auto expertOrder = StableDescendingIndices(score);
        std::vector<float> rowY;
        rowY.reserve(testCase.topK);
        for (uint32_t rank = 0; rank < testCase.topK; ++rank) {
            const int32_t expert = expertOrder[rank];
            rowY.push_back(sigmoid[expert]);
            if (testCase.enableExpertMapping) {
                const int32_t offset = static_cast<int32_t>(token % mappingNum[expert]);
                indices->push_back(mappingTable[expert * 3 + offset]);
            } else {
                indices->push_back(expert);
            }
        }
        if (testCase.isNorm) {
            float sum = 0.0F;
            for (float value : rowY) {
                sum += value;
            }
            for (float &value : rowY) {
                value = value / sum * testCase.scale;
            }
        }
        y->insert(y->end(), rowY.begin(), rowY.end());
    }
}

template <typename T>
bool RunCase(const TestCase &testCase, aclDataType inputType, float tolerance, aclrtStream stream)
{
    std::vector<T> xStorage;
    std::vector<T> addStorage;
    std::vector<float> xForGolden;
    std::vector<float> addForGolden;
    xStorage.reserve(testCase.tokenNum * testCase.expertNum);
    addStorage.reserve(testCase.expertNum);
    xForGolden.reserve(testCase.tokenNum * testCase.expertNum);
    addForGolden.reserve(testCase.expertNum);
    for (int64_t token = 0; token < testCase.tokenNum; ++token) {
        for (int64_t expert = 0; expert < testCase.expertNum; ++expert) {
            const float value = static_cast<float>(((token * 37 + expert * 17 + 11) % 19) - 9) / 4.0F;
            const T stored = FromFloat<T>(value);
            xStorage.push_back(stored);
            xForGolden.push_back(ToFloat(stored));
        }
    }
    for (int64_t expert = 0; expert < testCase.expertNum; ++expert) {
        const float value = static_cast<float>(((expert * 7 + 3) % 13) - 6) / 8.0F;
        const T stored = FromFloat<T>(value);
        addStorage.push_back(stored);
        addForGolden.push_back(ToFloat(stored));
    }

    std::vector<int32_t> mappingNum(testCase.expertNum);
    std::vector<int32_t> mappingTable(testCase.expertNum * 3);
    for (int64_t expert = 0; expert < testCase.expertNum; ++expert) {
        mappingNum[expert] = static_cast<int32_t>(expert % 3 + 1);
        for (int32_t offset = 0; offset < 3; ++offset) {
            mappingTable[expert * 3 + offset] = static_cast<int32_t>(1000 + expert * 3 + offset);
        }
    }

    std::vector<float> expectedY;
    std::vector<int32_t> expectedIndices;
    CalculateGolden(testCase, xForGolden, addForGolden, mappingNum, mappingTable, &expectedY, &expectedIndices);

    DeviceTensor x;
    DeviceTensor addNum;
    DeviceTensor mappingNumTensor;
    DeviceTensor mappingTableTensor;
    DeviceTensor y;
    DeviceTensor indices;
    if (!CreateTensor(&x, xStorage, {testCase.tokenNum, testCase.expertNum}, inputType) ||
        !CreateTensor(&addNum, addStorage, {testCase.expertNum}, inputType) ||
        !CreateTensor(&y, std::vector<float>(expectedY.size(), 0.0F), {testCase.tokenNum, testCase.topK}, ACL_FLOAT) ||
        !CreateTensor(&indices, std::vector<int32_t>(expectedIndices.size(), 0), {testCase.tokenNum, testCase.topK},
                      ACL_INT32)) {
        std::printf("%s: failed to create tensors\n", testCase.name);
        return false;
    }
    if (testCase.enableExpertMapping &&
        (!CreateTensor(&mappingNumTensor, mappingNum, {testCase.expertNum}, ACL_INT32) ||
         !CreateTensor(&mappingTableTensor, mappingTable, {testCase.expertNum, 3}, ACL_INT32))) {
        std::printf("%s: failed to create mapping tensors\n", testCase.name);
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    const aclnnStatus getWorkspaceStatus = aclnnMoeFusedTopkGetWorkspaceSize(
        x.tensor, addNum.tensor, testCase.enableExpertMapping ? mappingNumTensor.tensor : nullptr,
        testCase.enableExpertMapping ? mappingTableTensor.tensor : nullptr, testCase.groupNum, testCase.groupTopk,
        testCase.topN, testCase.topK, 0, testCase.isNorm, testCase.scale, testCase.enableExpertMapping, y.tensor,
        indices.tensor, &workspaceSize, &executor);
    if (getWorkspaceStatus != ACL_SUCCESS) {
        std::printf("%s: GetWorkspaceSize failed: %d\n", testCase.name, getWorkspaceStatus);
        return false;
    }

    void *workspace = nullptr;
    if (workspaceSize > 0 && aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::printf("%s: workspace allocation failed\n", testCase.name);
        return false;
    }
    const aclnnStatus runStatus = aclnnMoeFusedTopk(workspace, workspaceSize, executor, stream);
    const aclError syncStatus = aclrtSynchronizeStream(stream);
    if (runStatus != ACL_SUCCESS || syncStatus != ACL_SUCCESS) {
        std::printf("%s: device execution failed: %d\n", testCase.name, runStatus);
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        return false;
    }
    // CommonOpExecutorRun takes the phase-1 executor through the CANN cache
    // path.  Do not destroy it explicitly; this matches the ACLNN sample
    // lifetime model.  The workspace must remain alive until the stream ends.
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    std::vector<float> actualY(expectedY.size());
    std::vector<int32_t> actualIndices(expectedIndices.size());
    const size_t yBytes = actualY.size() * sizeof(float);
    const size_t indicesBytes = actualIndices.size() * sizeof(int32_t);
    if (aclrtMemcpy(actualY.data(), yBytes, y.deviceAddr, yBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(actualIndices.data(), indicesBytes, indices.deviceAddr, indicesBytes, ACL_MEMCPY_DEVICE_TO_HOST) !=
            ACL_SUCCESS) {
        std::printf("%s: output copy failed\n", testCase.name);
        return false;
    }

    float maxAbsError = 0.0F;
    for (size_t i = 0; i < actualY.size(); ++i) {
        maxAbsError = std::max(maxAbsError, std::fabs(actualY[i] - expectedY[i]));
        if (actualIndices[i] != expectedIndices[i] || std::fabs(actualY[i] - expectedY[i]) > tolerance) {
            std::printf("%s: mismatch at %zu, y=%f expected=%f, index=%d expected=%d\n", testCase.name, i,
                        actualY[i], expectedY[i], actualIndices[i], expectedIndices[i]);
            return false;
        }
    }
    std::printf("PASS %-36s max_abs_error=%g\n", testCase.name, maxAbsError);
    return true;
}

bool RunTestCase(const TestCase &testCase, aclrtStream stream)
{
    switch (testCase.inputType) {
        case InputType::kFp32:
            return RunCase<float>(testCase, ACL_FLOAT, 2e-4F, stream);
        case InputType::kFp16:
            return RunCase<aclFloat16>(testCase, ACL_FLOAT16, 2e-3F, stream);
        case InputType::kBf16:
            return RunCase<Bfloat16Storage>(testCase, ACL_BF16, 3e-3F, stream);
    }
    return false;
}

}  // namespace

int main()
{
    const std::vector<TestCase> testCases = {
        {"fp32_no_mapping_multi_group", InputType::kFp32, 7, 12, 3, 2, 2, 4, false, 1.0F, false},
        {"fp32_mapping_normalized", InputType::kFp32, 5, 8, 4, 2, 1, 3, true, 1.3F, true},
        {"fp16_mapping_non16_expert", InputType::kFp16, 41, 136, 8, 4, 2, 8, true, 0.75F, true},
        {"fp16_no_mapping_single_group", InputType::kFp16, 2, 32, 1, 1, 32, 5, false, 1.0F, false},
        {"bf16_no_mapping_normalized", InputType::kBf16, 11, 12, 3, 1, 2, 3, true, 0.9F, false},
        {"bf16_mapping_multi_group", InputType::kBf16, 3, 15, 5, 2, 1, 4, false, 1.0F, true},
    };
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS) {
        std::printf("failed to initialize Ascend device 0\n");
        return 1;
    }
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        std::printf("failed to create ACL stream\n");
        aclrtResetDevice(0);
        aclFinalize();
        return 1;
    }
    bool passed = true;
    for (const auto &testCase : testCases) {
        passed = RunTestCase(testCase, stream) && passed;
    }
    const aclError destroyStreamStatus = aclrtDestroyStream(stream);
    const aclError resetStatus = aclrtResetDevice(0);
    const aclError finalizeStatus = aclFinalize();
    if (destroyStreamStatus != ACL_SUCCESS || resetStatus != ACL_SUCCESS || finalizeStatus != ACL_SUCCESS) {
        std::printf("failed to finalize ACL runtime\n");
        return 1;
    }
    return passed ? 0 : 1;
}
