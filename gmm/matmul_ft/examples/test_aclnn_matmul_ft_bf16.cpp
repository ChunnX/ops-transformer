/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <iostream>

#include "acl/acl.h"
#include "aclnnop/aclnn_matmul_ft.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...) \
    do {                        \
        std::printf(message, ##__VA_ARGS__); \
    } while (0)

using Bf16 = uint16_t;

Bf16 FloatToBf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<Bf16>(bits >> 16U);
}

float Bf16ToFloat(Bf16 value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

struct TestArgs {
    int64_t m = 4096;
    int64_t n = 4096;
    int64_t k = 4096;
    int32_t deviceId = 0;
    double eMax = 0.001;
    int64_t reduceCores = 8;
    int32_t verbose = 1;

    void Print() const
    {
        LOG_PRINT("Args: m=%ld n=%ld k=%ld eMax=%f reduceCores=%ld "
                  "verbose=%d deviceId=%d\n",
                  m, n, k, eMax, reduceCores, verbose, deviceId);
    }

    int Parse(int argc, const char** argv)
    {
        const std::string helper =
            "test_aclnn_matmul_ft_bf16 m n k eMax reduceCores verbose [deviceId]";
        enum {
            M_IDX = 1,
            N_IDX,
            K_IDX,
            EMAX_IDX,
            RED_CORES_IDX,
            VERBOSE_IDX,
            DEVICE_ID_IDX,
            ARGS_MAX
        };
        if (argc < DEVICE_ID_IDX || argc > ARGS_MAX) {
            std::cerr << helper << std::endl;
            return -1;
        }
        m = std::atol(argv[M_IDX]);
        n = std::atol(argv[N_IDX]);
        k = std::atol(argv[K_IDX]);
        eMax = std::stod(argv[EMAX_IDX]);
        reduceCores = std::atol(argv[RED_CORES_IDX]);
        verbose = std::atoi(argv[VERBOSE_IDX]);
        if (argc == ARGS_MAX) {
            deviceId = std::atoi(argv[DEVICE_ID_IDX]);
        }
        return 0;
    }
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * static_cast<int64_t>(sizeof(T));
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                              strides.data(), 0, ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

template <typename T>
int CopyDeviceToHost(std::vector<T>& hostData, const void* deviceAddr, const char* name)
{
    const size_t bytes = hostData.size() * sizeof(T);
    const auto ret = aclrtMemcpy(hostData.data(), bytes, deviceAddr, bytes,
                                 ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("copy %s from device to host failed. ERROR: %d\n", name, ret);
              return ret);
    return ACL_SUCCESS;
}


int InitAcl(int32_t deviceId, aclrtContext* context, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

void PrintChecksum(const std::vector<int64_t>& shape, void* deviceAddr, aclDataType dataType,
                   const char* name)
{
    auto elementCount = GetShapeSize(shape);
    if (dataType == ACL_FLOAT) {
        std::vector<float> hostData(elementCount, 0.0F);
        auto ret = aclrtMemcpy(hostData.data(), hostData.size() * sizeof(float),
                               deviceAddr, hostData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy %s from device to host failed. ERROR: %d\n", name, ret); return);

        double sum = 0.0;
        for (auto v : hostData) {
            sum += static_cast<double>(v);
        }
        LOG_PRINT("%s (FP32) first 8:", name);
        int64_t limit = elementCount < 8 ? elementCount : 8;
        for (int64_t i = 0; i < limit; ++i) {
            LOG_PRINT(" %.6f", hostData[i]);
        }
        LOG_PRINT("  checksum: %.6f\n", sum);

    } else if (dataType == ACL_BF16) {
        std::vector<Bf16> hostData(elementCount, 0);
        auto ret = aclrtMemcpy(hostData.data(), hostData.size() * sizeof(Bf16),
                               deviceAddr, hostData.size() * sizeof(Bf16), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy %s from device to host failed. ERROR: %d\n", name, ret); return);

        double sum = 0.0;
        for (auto v : hostData) {
            sum += static_cast<double>(Bf16ToFloat(v));
        }
        LOG_PRINT("%s (BF16) first 8:", name);
        int64_t limit = elementCount < 8 ? elementCount : 8;
        for (int64_t i = 0; i < limit; ++i) {
            LOG_PRINT(" %.6f", Bf16ToFloat(hostData[i]));
        }
        LOG_PRINT("  checksum: %.6f\n", sum);

    } else if (dataType == ACL_UINT8) {
        std::vector<uint8_t> hostData(elementCount, 0);
        auto ret = aclrtMemcpy(hostData.data(), hostData.size() * sizeof(uint8_t),
                               deviceAddr, hostData.size() * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy %s from device to host failed. ERROR: %d\n", name, ret); return);

        uint64_t sum = 0;
        for (auto v : hostData) {
            sum += static_cast<uint64_t>(v);
        }
        LOG_PRINT("%s (U8) first 8:", name);
        int64_t limit = elementCount < 8 ? elementCount : 8;
        for (int64_t i = 0; i < limit; ++i) {
            LOG_PRINT(" %u", static_cast<uint32_t>(hostData[i]));
        }
        LOG_PRINT("  checksum: %lu\n", sum);
    }
}

int main(int argc, const char** argv)
{
    TestArgs args;
    if (args.Parse(argc, argv) != 0) {
        return -1;
    }
    args.Print();

    const int64_t M = args.m;
    const int64_t N = args.n;
    const int64_t K = args.k;
    const int64_t splitN = (N + 256 - 1) / 256;       // L1_TILE_N = 256
    const int64_t bStatBlock = (splitN + 8 - 1) / 8;  // FLOAT_ELEMENTS_PER_BLOCK = 8
    const int64_t bStatLen = bStatBlock * 8 + 8;       // aligned length for B stats
    const int64_t rowSplitLen = M * splitN;            // length of per-row split outputs
    const int64_t beLen = K * splitN;                  // length of BE / BMaxSlice etc.

    // ── Input shapes ──
    const std::vector<int64_t> aShape = {M, K};
    const std::vector<int64_t> bShape = {K, N};
    const std::vector<int64_t> xShape = {N};
    const std::vector<int64_t> xvShape = {N};
    const std::vector<int64_t> vxForAeShape = {K};

    // ── Output shapes (from infershape) ──
    const std::vector<int64_t> zRowColShape = {rowSplitLen};
    const std::vector<int64_t> compRowShape = {((M + 7) / 8) * splitN};
    const std::vector<int64_t> compColShape = {(N + 7) / 8};
    const std::vector<int64_t> cShape = {M, N};
    const std::vector<int64_t> bStatShape = {bStatLen};
    const std::vector<int64_t> beShape = {beLen};
    const std::vector<int64_t> aRedShape = {M};

    // ── Init ACL ──
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = InitAcl(args.deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // ── Fill host data ──
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

    std::vector<Bf16> hostA(GetShapeSize(aShape), 0);
    std::vector<Bf16> hostB(GetShapeSize(bShape), 0);
    std::vector<float> hostX(GetShapeSize(xShape), 1.0F);
    std::vector<Bf16> hostXv(GetShapeSize(xvShape), 0);
    std::vector<Bf16> hostVxForAe(GetShapeSize(vxForAeShape), 0);

    for (auto& v : hostA) {
        v = FloatToBf16(dist(rng));
    }
    for (auto& v : hostB) {
        v = FloatToBf16(dist(rng));
    }
    for (auto& v : hostXv) {
        v = FloatToBf16(1.0F);
    }
    for (auto& v : hostVxForAe) {
        v = FloatToBf16(1.0F / static_cast<float>(K));
    }

    // Output data (zero-initialised)
    std::vector<float> hostZRow(GetShapeSize(zRowColShape), 0.0F);
    std::vector<float> hostZCol(GetShapeSize(zRowColShape), 0.0F);
    std::vector<float> hostDRow(GetShapeSize(zRowColShape), 0.0F);
    std::vector<float> hostDCol(GetShapeSize(zRowColShape), 0.0F);
    std::vector<uint8_t> hostCompRow(GetShapeSize(compRowShape), 0);
    std::vector<uint8_t> hostCompCol(GetShapeSize(compColShape), 0);
    std::vector<float> hostC(GetShapeSize(cShape), 0.0F);
    std::vector<float> hostThreshold(GetShapeSize(zRowColShape), 0.0F);
    std::vector<float> hostBMeanAbs(GetShapeSize(bStatShape), 0.0F);
    std::vector<float> hostBMeanSquare(GetShapeSize(bStatShape), 0.0F);
    std::vector<float> hostBVar(GetShapeSize(bStatShape), 0.0F);
    std::vector<Bf16> hostBe(GetShapeSize(beShape), 0);
    std::vector<float> hostBeForAiv(GetShapeSize(beShape), 0.0F);
    std::vector<float> hostBMaxSlice(GetShapeSize(beShape), 0.0F);
    std::vector<float> hostBMinSlice(GetShapeSize(beShape), 0.0F);
    std::vector<float> hostAMax(GetShapeSize(aRedShape), 0.0F);
    std::vector<float> hostAMean(GetShapeSize(aRedShape), 0.0F);
    std::vector<float> hostAMin(GetShapeSize(aRedShape), 0.0F);

    // ── Create device tensors ──
    void *devA = nullptr, *devB = nullptr, *devX = nullptr, *devXv = nullptr, *devVxForAe = nullptr;
    void *devZRow = nullptr, *devZCol = nullptr, *devDRow = nullptr, *devDCol = nullptr;
    void *devCompRow = nullptr, *devCompCol = nullptr, *devGemmAddend = nullptr, *devC = nullptr;
    void *devThreshold = nullptr;
    void *devBMeanAbs = nullptr, *devBMeanSquare = nullptr, *devBVar = nullptr;
    void *devBe = nullptr, *devBeForAiv = nullptr;
    void *devBMaxSlice = nullptr, *devBMinSlice = nullptr;
    void *devAMax = nullptr, *devAMean = nullptr, *devAMin = nullptr;

    aclTensor *tA = nullptr, *tB = nullptr, *tX = nullptr, *tXv = nullptr, *tVxForAe = nullptr;
    aclTensor *tZRow = nullptr, *tZCol = nullptr, *tDRow = nullptr, *tDCol = nullptr;
    aclTensor *tCompRow = nullptr, *tCompCol = nullptr, *tGemmAddend = nullptr, *tC = nullptr;
    aclTensor *tThreshold = nullptr;
    aclTensor *tBMeanAbs = nullptr, *tBMeanSquare = nullptr, *tBVar = nullptr;
    aclTensor *tBe = nullptr, *tBeForAiv = nullptr;
    aclTensor *tBMaxSlice = nullptr, *tBMinSlice = nullptr;
    aclTensor *tAMax = nullptr, *tAMean = nullptr, *tAMin = nullptr;

    CHECK_RET(CreateAclTensor(hostA, aShape, &devA, ACL_BF16, &tA) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostB, bShape, &devB, ACL_BF16, &tB) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostX, xShape, &devX, ACL_FLOAT, &tX) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostXv, xvShape, &devXv, ACL_BF16, &tXv) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostVxForAe, vxForAeShape, &devVxForAe, ACL_BF16, &tVxForAe) == ACL_SUCCESS, return ret);

    CHECK_RET(CreateAclTensor(hostZRow, zRowColShape, &devZRow, ACL_FLOAT, &tZRow) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostZCol, zRowColShape, &devZCol, ACL_FLOAT, &tZCol) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostDRow, zRowColShape, &devDRow, ACL_FLOAT, &tDRow) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostDCol, zRowColShape, &devDCol, ACL_FLOAT, &tDCol) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostCompRow, compRowShape, &devCompRow, ACL_UINT8, &tCompRow) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostCompCol, compColShape, &devCompCol, ACL_UINT8, &tCompCol) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostC, cShape, &devGemmAddend, ACL_FLOAT, &tGemmAddend) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostC, cShape, &devC, ACL_FLOAT, &tC) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostThreshold, zRowColShape, &devThreshold, ACL_FLOAT, &tThreshold) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostBMeanAbs, bStatShape, &devBMeanAbs, ACL_FLOAT, &tBMeanAbs) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostBMeanSquare, bStatShape, &devBMeanSquare, ACL_FLOAT, &tBMeanSquare) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostBVar, bStatShape, &devBVar, ACL_FLOAT, &tBVar) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostBe, beShape, &devBe, ACL_BF16, &tBe) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostBeForAiv, beShape, &devBeForAiv, ACL_FLOAT, &tBeForAiv) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostBMaxSlice, beShape, &devBMaxSlice, ACL_FLOAT, &tBMaxSlice) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostBMinSlice, beShape, &devBMinSlice, ACL_FLOAT, &tBMinSlice) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostAMax, aRedShape, &devAMax, ACL_FLOAT, &tAMax) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostAMean, aRedShape, &devAMean, ACL_FLOAT, &tAMean) == ACL_SUCCESS, return ret);
    CHECK_RET(CreateAclTensor(hostAMin, aRedShape, &devAMin, ACL_FLOAT, &tAMin) == ACL_SUCCESS, return ret);

    
    // ── Launch MatmulFt ──
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnMatmulFtGetWorkspaceSize(
        tA, tB, tX, tXv, tVxForAe,
        args.eMax,
        args.reduceCores,
        tZRow, tZCol, tDRow, tDCol,
        tCompRow, tCompCol, tC,
        tThreshold,
        tBMeanAbs, tBMeanSquare, tBVar,
        tBe, tBeForAiv,
        tBMaxSlice, tBMinSlice,
        tAMax, tAMean, tAMin,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMatmulFtGetWorkspaceSize failed. ERROR: %d.\n[ERROR msg]%s\n",
                        ret, aclGetRecentErrMsg());
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnMatmulFt(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMatmulFt failed. ERROR: %d.\n[ERROR msg]%s\n",
                        ret, aclGetRecentErrMsg());
              return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("MatmulFt run finished successfully.\n\n");
    // ── Print results ──
    if (args.verbose) {
        PrintChecksum(cShape, devC, ACL_FLOAT, "c");
        PrintChecksum(zRowColShape, devZRow, ACL_FLOAT, "zRow");
        PrintChecksum(zRowColShape, devZCol, ACL_FLOAT, "zCol");
        PrintChecksum(zRowColShape, devDRow, ACL_FLOAT, "dRow");
        PrintChecksum(zRowColShape, devDCol, ACL_FLOAT, "dCol");
        PrintChecksum(compRowShape, devCompRow, ACL_UINT8, "compRow");
        PrintChecksum(compColShape, devCompCol, ACL_UINT8, "compCol");
        PrintChecksum(zRowColShape, devThreshold, ACL_FLOAT, "threshold");
        PrintChecksum(bStatShape, devBMeanAbs, ACL_FLOAT, "bMeanAbs");
        PrintChecksum(bStatShape, devBMeanSquare, ACL_FLOAT, "bMeanSquare");
        PrintChecksum(bStatShape, devBVar, ACL_FLOAT, "bVar");
        PrintChecksum(beShape, devBe, ACL_BF16, "be");
        PrintChecksum(beShape, devBeForAiv, ACL_FLOAT, "beForAiv");
        PrintChecksum(beShape, devBMaxSlice, ACL_FLOAT, "bMaxSlice");
        PrintChecksum(beShape, devBMinSlice, ACL_FLOAT, "bMinSlice");
        PrintChecksum(aRedShape, devAMax, ACL_FLOAT, "aMax");
        PrintChecksum(aRedShape, devAMean, ACL_FLOAT, "aMean");
        PrintChecksum(aRedShape, devAMin, ACL_FLOAT, "aMin");
    }

    ret = aclDestroyAclOpExecutor(executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclDestroyAclOpExecutor failed. ERROR: %d\n", ret);
              return ret);
    executor = nullptr;

    // ── Cleanup ──
    auto Destroy = [](aclTensor*& t) { if (t) { aclDestroyTensor(t); t = nullptr; } };
    auto Free = [](void*& p) { if (p) { aclrtFree(p); p = nullptr; } };

    Destroy(tA); Destroy(tB); Destroy(tX); Destroy(tXv); Destroy(tVxForAe);
    Destroy(tZRow); Destroy(tZCol); Destroy(tDRow); Destroy(tDCol);
    Destroy(tCompRow); Destroy(tCompCol); Destroy(tGemmAddend); Destroy(tC);
    Destroy(tThreshold);
    Destroy(tBMeanAbs); Destroy(tBMeanSquare); Destroy(tBVar);
    Destroy(tBe); Destroy(tBeForAiv);
    Destroy(tBMaxSlice); Destroy(tBMinSlice);
    Destroy(tAMax); Destroy(tAMean); Destroy(tAMin);

    Free(devA); Free(devB); Free(devX); Free(devXv); Free(devVxForAe);
    Free(devZRow); Free(devZCol); Free(devDRow); Free(devDCol);
    Free(devCompRow); Free(devCompCol); Free(devGemmAddend); Free(devC);
    Free(devThreshold);
    Free(devBMeanAbs); Free(devBMeanSquare); Free(devBVar);
    Free(devBe); Free(devBeForAiv);
    Free(devBMaxSlice); Free(devBMinSlice);
    Free(devAMax); Free(devAMean); Free(devAMin);
    Free(workspaceAddr);

    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(args.deviceId);
    aclFinalize();

    return 0;
}
