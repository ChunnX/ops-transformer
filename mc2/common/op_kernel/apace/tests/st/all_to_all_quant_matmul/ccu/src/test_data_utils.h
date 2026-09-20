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
 * \file test_data_utils.h
 * \brief 测试辅助件——与调用算子无直接关系的函数（数据生成 / 期望值 / 校验 / usage）
 *
 * 实际场景中数据来自文件或上游，本文件的行级溯源生成器仅为 ST 自包含测试向量：
 *   数据语义（K-并行 GEMM）：x1=[M,K] FP8 E4M3FN（token 轴均分 R 块做 A2A），
 *   workspace=[recvData|recvScale]，x2=[R*K,N] 完整权重不通信，
 *   Y=[M/R,N] BF16 = Σ_r recvData[r块,:] @ x2[r*K段,:]（L0C 跨源累加，self 项直读 x1）。
 *   行级溯源数据（E4M3FN 2 的幂闭式，全部精确可表示）：位级断言零舍入噪声，
 *   可分类源 rank 混淆 / 目标路由混淆 / 行轮错位 / 字节损坏。
 *
 * 依赖：ACLCHECK 宏与 acl 运行时声明来自 apace core 头链，须在
 * all_to_all_ccu_kernel_impl.asc（锚点）之后 include 本文件。
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>

// FP8 E4M3FN / E8M0 常量编码（2 的幂，精确可表示；E8M0 127 = 1.0，dequant 恒等）
inline constexpr uint8_t FP8_E4M3FN_POW2[4] = {0x38, 0x40, 0x48, 0x50};
inline constexpr uint8_t FP8_E8M0_SCALE_1_0 = 127;

inline uint8_t ENC4(uint64_t v)
{
    return FP8_E4M3FN_POW2[v % 4];
}
inline uint8_t ENC2(uint64_t v)
{
    return (v % 2 == 0) ? FP8_E4M3FN_POW2[0] : FP8_E4M3FN_POW2[1];
}

// 期望 Y 值的 BF16 位型：Y[m] = K × Σ_{r=0}^{R-1} 2^((2i+r+m)%4 + r%2)
// （各加数均为 2 的幂，和 ≤ 256，×K ≤ 2^16——float 与 BF16 均精确表示）
inline uint16_t ExpectYBf16Bits(uint32_t myRank, uint64_t m, uint64_t rankSize, uint64_t K)
{
    float sum = 0.0f;
    for (uint64_t r = 0; r < rankSize; r++) {
        const uint64_t e = (2 * myRank + r + m) % 4 + (r % 2);
        sum += static_cast<float>(1ULL << e);
    }
    const float expect = static_cast<float>(K) * sum;
    const uint32_t bits = *reinterpret_cast<const uint32_t *>(&expect);
    return static_cast<uint16_t>(bits >> 16);
}

inline void PrintUsage(const char *prog)
{
    std::cout << "Usage:\n"
              << "  " << prog << "                          # precision, defaults\n"
              << "  " << prog << " precision [M] [tileM] [K] [N]\n"
              << "  " << prog << " perf [M] [tileM] [K] [N] [loop]\n"
              << "Defaults: M=512, tileM=128, K=256, N=256, loop=10"
              << std::endl;
}

// 测试向量生成器（A + scaleA）：rank s 的 x1 第 j 块第 m 行 = ENC4((2j+s+m)%4)
// （j=目标路由，s=源 rank，m=块内行号；scaleA 全 1.0）
inline int GenerateInputData(uint32_t myRank, uint64_t rankSize, uint64_t blockRows,
                             uint64_t K, uint64_t scaleKLen,
                             void *x1, uint64_t sendSize,
                             void *x1Scale, uint64_t sendScaleSize)
{
    void *hostBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&hostBuf, sendSize));
    uint8_t *tmpHost = static_cast<uint8_t *>(hostBuf);
    for (uint64_t j = 0; j < rankSize; j++) {
        for (uint64_t m = 0; m < blockRows; m++) {
            const uint8_t fillBits = ENC4(2 * j + myRank + m);
            uint8_t *row = tmpHost + (j * blockRows + m) * K;
            for (uint64_t k = 0; k < K; k++) {
                row[k] = fillBits;
            }
        }
    }
    aclError aclRet = aclrtMemcpy(x1, sendSize, hostBuf, sendSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        (void)aclrtFreeHost(hostBuf);
        return static_cast<int>(aclRet);
    }
    ACLCHECK(aclrtFreeHost(hostBuf));

    void *hostScaleBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&hostScaleBuf, sendScaleSize));
    memset(static_cast<uint8_t *>(hostScaleBuf), FP8_E8M0_SCALE_1_0, sendScaleSize);
    aclRet = aclrtMemcpy(x1Scale, sendScaleSize, hostScaleBuf, sendScaleSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        (void)aclrtFreeHost(hostScaleBuf);
        return static_cast<int>(aclRet);
    }
    ACLCHECK(aclrtFreeHost(hostScaleBuf));
    return 0;
}

// 权重向量生成器（B + scaleB，不通信）：B=[R*K,N] DN 列主序（元素 (k,n) 位于 k+n*R*K），
// 第 r 段全值 ENC2(r)；scaleB 全 127（值全同，任意布局读同值）
inline int GenerateWeightData(uint64_t rankSize, uint64_t K, uint64_t N, uint64_t scaleKLen,
                              void *x2, uint64_t weightSize,
                              void *x2Scale, uint64_t weightScaleSize)
{
    void *hostWeight = nullptr;
    ACLCHECK(aclrtMallocHost(&hostWeight, weightSize));
    uint8_t *tmpWeight = static_cast<uint8_t *>(hostWeight);
    for (uint64_t n = 0; n < N; n++) {
        for (uint64_t r = 0; r < rankSize; r++) {
            memset(tmpWeight + n * rankSize * K + r * K, ENC2(r), K);
        }
    }
    aclError aclRet = aclrtMemcpy(x2, weightSize, hostWeight, weightSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        (void)aclrtFreeHost(hostWeight);
        return static_cast<int>(aclRet);
    }
    ACLCHECK(aclrtFreeHost(hostWeight));

    void *hostScale = nullptr;
    ACLCHECK(aclrtMallocHost(&hostScale, weightScaleSize));
    memset(static_cast<uint8_t *>(hostScale), FP8_E8M0_SCALE_1_0, weightScaleSize);
    aclRet = aclrtMemcpy(x2Scale, weightScaleSize, hostScale, weightScaleSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        (void)aclrtFreeHost(hostScale);
        return static_cast<int>(aclRet);
    }
    ACLCHECK(aclrtFreeHost(hostScale));
    return 0;
}

// precision 模式校验：读回 output（BF16）→ 逐行逐元素位型断言 → 首错定位 → 落地 bin。
// 返回 0 = PASS；非 0 = FAIL（供 RunAllToAllMatmul 透传为进程退出码）
inline int VerifyAndDumpOutput(uint32_t myRank, uint64_t rankSize, uint64_t blockRows,
                               uint64_t N, uint64_t K,
                               void *output, uint64_t outputSize)
{
    void *resultBuff = nullptr;
    ACLCHECK(aclrtMallocHost(&resultBuff, outputSize));
    aclError aclRet = aclrtMemcpy(resultBuff, outputSize, output, outputSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        (void)aclrtFreeHost(resultBuff);
        return static_cast<int>(aclRet);
    }
    const uint16_t *outHost = static_cast<const uint16_t *>(resultBuff);

    const uint64_t totalElems = blockRows * N;
    uint64_t errCount = 0;
    bool hasFirstErr = false;
    uint64_t firstErrRow = 0;
    uint64_t firstErrCol = 0;
    uint16_t firstErrExpect = 0;
    uint16_t firstErrActual = 0;
    for (uint64_t m = 0; m < blockRows; m++) {
        const uint16_t expectBits = ExpectYBf16Bits(myRank, m, rankSize, K);
        const uint16_t *row = outHost + m * N;
        for (uint64_t n = 0; n < N; n++) {
            if (row[n] != expectBits) {
                errCount++;
                if (!hasFirstErr) {
                    hasFirstErr = true;
                    firstErrRow = m;
                    firstErrCol = n;
                    firstErrExpect = expectBits;
                    firstErrActual = row[n];
                }
            }
        }
    }

    if (errCount == 0) {
        std::cout << "rankId: " << myRank << ", A2A+MatMul PASS: all " << totalElems
                  << " output elems verified (K=" << K << ")" << std::endl;
    } else {
        std::cout << "rankId: " << myRank << ", A2A+MatMul FAIL: " << errCount
                  << " / " << totalElems << " elems mismatch" << std::endl;
        std::cout << "  first error at Y row " << firstErrRow << ", col " << firstErrCol
                  << ": expect 0x" << std::hex << firstErrExpect
                  << ", actual 0x" << firstErrActual << std::dec << std::endl;
    }

    std::string outputDir = "output/" + std::to_string(myRank);
    if (system(("mkdir -p " + outputDir).c_str()) != 0) {
        std::cout << "rankId: " << myRank << ", warn: mkdir " << outputDir << " failed" << std::endl;
    }
    std::string outFile = outputDir + "/npu_out.bin";
    std::ofstream ofs(outFile, std::ios::binary);
    ofs.write(static_cast<const char *>(resultBuff), outputSize);
    ofs.close();
    std::cout << "rankId: " << myRank << ", output written to " << outFile
              << " (" << outputSize << " bytes)" << std::endl;

    ACLCHECK(aclrtFreeHost(resultBuff));
    return (errCount == 0) ? 0 : 1;
}
