/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// segment 模式隔离验证：模拟 hcomm rank-major 接收布局，直驱 QuantGroupedMatmulKernel（segW=2）
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include "kernel_basic_intf.h"
#include "acl/acl.h"
#include "apace_st_utils.h"
#include "apace/kernel/fusions/allto_allv_quant_grouped_matmul/allto_allv_quant_grouped_matmul_tiling_data.h"
#include "apace_seg_gmm_kernel_launcher.h"

// E4M3/E8M0 host 反量化
static float E8m0ToFloat(uint8_t v) { return std::ldexp(1.0f, static_cast<int>(v) - 127); }
static float E4m3ToFloat(uint8_t v)
{
    int sign = (v & 0x80) ? -1 : 1;
    int e = (v >> 3) & 0xF;
    int m = v & 0x7;
    if (e == 0) {
        return sign * std::ldexp(static_cast<float>(m) / 8.0f, -6);
    }
    return sign * std::ldexp(1.0f + static_cast<float>(m) / 8.0f, e - 7);
}

int main(int argc, char *argv[])
{
    constexpr uint32_t W = 2;
    constexpr uint32_t E = 4;
    constexpr uint32_t K = 2048;
    constexpr uint32_t N = 2048;
    constexpr uint32_t scaleKLen = K / 64 * 2;
    // 与 E2E 相同的非均匀计数（rank0 视角）
    uint32_t segCnt[2 * 4] = {135, 130, 121, 134, 131, 122, 135, 130}; // gc[s*8+e] (myRank=0)
    uint32_t segCntSrc1[2 * 4] = {131, 122, 135, 130, 121, 135, 128, 122}; // gc[1*8+...] (dst 视角无关)
    // 接收行布局（rank-major）：src0 全部 4 个专家块 + src1 全部
    uint32_t srcStart[2] = {0, 0};
    srcStart[1] = segCnt[0] + segCnt[1] + segCnt[2] + segCnt[3];
    // segAOff(e,s) = srcStart[s] + sum_{e'<e} segCntSrcS(s,e')
    // 注意：src s 发给 me 的块内按 e 排列：计数 = segCnt[?]... 对 rank0: recvCnt[s][e] = gc[s*8+e]
    uint32_t recv[2][4] = {{135, 130, 121, 134}, {131, 122, 135, 130}};

    uint32_t segAOff[E * W];
    uint32_t segCOff[E * W];
    uint32_t segList[E * W + 1];
    uint32_t expertStart[E] = {0, 0, 0, 0};
    {
        uint32_t running = 0;
        for (uint32_t e = 0; e < E; ++e) {
            expertStart[e] = running;
            for (uint32_t s = 0; s < W; ++s) {
                running += recv[s][e];
            }
        }
    }
    {
        uint32_t cum = 0;
        for (uint32_t e = 0; e < E; ++e) {
            uint32_t srcPrefix = 0;
            for (uint32_t s = 0; s < W; ++s) {
                uint32_t segIdx = e * W + s;
                uint32_t ePrefix = 0;
                for (uint32_t e2 = 0; e2 < e; ++e2) {
                    ePrefix += recv[s][e2];
                }
                segAOff[segIdx] = srcStart[s] + ePrefix;
                segCOff[segIdx] = expertStart[e] + srcPrefix;
                srcPrefix += recv[s][e];
                cum += recv[s][e];
                segList[segIdx] = cum; // 包含式累计，与 kernel GetGroupRows 语义一致
            }
        }
    }
    const uint32_t totalRecv = segList[E * W - 1];

    printf("segList=[%u,%u,%u,%u,%u,%u,%u,%u] segAOff=[%u,%u,%u,%u,%u,%u,%u,%u] segCOff=[%u,%u,%u,%u,%u,%u,%u,%u] "
           "totalRecv=%u\n",
           segList[0], segList[1], segList[2], segList[3], segList[4], segList[5], segList[6], segList[7],
           segAOff[0], segAOff[1], segAOff[2], segAOff[3], segAOff[4], segAOff[5], segAOff[6], segAOff[7],
           segCOff[0], segCOff[1], segCOff[2], segCOff[3], segCOff[4], segCOff[5], segCOff[6], segCOff[7], totalRecv);

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));

    // 构造 rank-major 接收数据：src s 的行 = rank s 的 gmm_x（复刻 E2E 数据函数）
    const uint32_t srcRows[2] = {segCnt[0] + segCnt[1] + segCnt[2] + segCnt[3],
                                 segCntSrc1[0] + segCntSrc1[1] + segCntSrc1[2] + segCntSrc1[3]};
    const uint32_t totalARows = srcRows[0] + srcRows[1];
    std::vector<uint8_t> aRankMajor(totalARows * K);
    std::vector<uint8_t> aScaleRankMajor(totalARows * scaleKLen);
    for (uint32_t s = 0; s < W; ++s) {
        for (uint32_t i = 0; i < srcRows[s] * K; ++i) {
            aRankMajor[srcStart[s] * K + i] = static_cast<uint8_t>(0x30 + ((s * 131 + i) % 8));
        }
        for (uint32_t i = 0; i < srcRows[s] * scaleKLen; ++i) {
            aScaleRankMajor[srcStart[s] * scaleKLen + i] = static_cast<uint8_t>(127 + ((s * 17 + i) % 4));
        }
    }
    std::vector<uint8_t> aHost = std::move(aRankMajor);
    std::vector<uint8_t> aScaleHost = std::move(aScaleRankMajor);
    // 权重（本 rank 4 专家）
    std::vector<uint8_t> bHost(E * K * N);
    std::vector<uint8_t> bScaleHost(E * (K / 64) * N * 2);
    for (uint32_t i = 0; i < E * K * N; ++i) {
        bHost[i] = static_cast<uint8_t>(0x30 + (i % 8));
    }
    for (uint32_t i = 0; i < E * (K / 64) * N * 2; ++i) {
        bScaleHost[i] = static_cast<uint8_t>(127 + (i % 4));
    }

    // tiling（与 E2E 一致：baseM=144 baseN=224 baseK=128 kAL1=512）
    Apace::hcommA2avQuantGmmTilingData td = {};
    td.rankSize = W;
    td.expertNum = E;
    td.k = K;
    td.n = N;
    td.transB = 0;
    td.scaleKLen = scaleKLen;
    td.segNum = E * W;
    td.baseM = 144;
    td.baseN = 224;
    td.baseK = 128;
    td.kAL1 = 512;
    td.kBL1 = 512;
    td.scaleKAL1 = 512;
    memcpy(td.segList, segList, sizeof(segList));
    memcpy(td.segAOff, segAOff, sizeof(segAOff));
    memcpy(td.segCOff, segCOff, sizeof(segCOff));

    GM_ADDR aDev = nullptr;
    GM_ADDR aScaleDev = nullptr;
    GM_ADDR bDev = nullptr;
    GM_ADDR bScaleDev = nullptr;
    GM_ADDR cDev = nullptr;
    GM_ADDR tdDev = nullptr;
    const size_t cBytes = static_cast<size_t>(totalRecv) * N * 2;
    ACL_CHECK(aclrtMalloc((void **)&aDev, aHost.size(), ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void **)&aScaleDev, aScaleHost.size(), ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void **)&bDev, bHost.size(), ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void **)&bScaleDev, bScaleHost.size(), ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void **)&cDev, cBytes, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void **)&tdDev, sizeof(td), ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMemset(cDev, cBytes, 0, cBytes));
    ACL_CHECK(aclrtMemcpy(aDev, aHost.size(), aHost.data(), aHost.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(aScaleDev, aScaleHost.size(), aScaleHost.data(), aScaleHost.size(),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(bDev, bHost.size(), bHost.data(), bHost.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(bScaleDev, bScaleHost.size(), bScaleHost.data(), bScaleHost.size(),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(tdDev, sizeof(td), &td, sizeof(td), ACL_MEMCPY_HOST_TO_DEVICE));

    ApaceSegGmmKernel_E4M3<<<32, nullptr, stream>>>(
        reinterpret_cast<__gm__ Apace::hcommA2avQuantGmmTilingData *>(tdDev), aDev, aScaleDev, bDev, bScaleDev, cDev);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<uint16_t> cHost(totalRecv * N, 0);
    ACL_CHECK(aclrtMemcpy(cHost.data(), cBytes, cDev, cBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    printf("step5\n"); fflush(stdout);


    // golden：expert-major，逐段
    double maxRel = 0;
    long long bad = 0;
    long long sample = 0;
    for (uint32_t e = 0; e < E; ++e) {
        for (uint32_t s = 0; s < W; ++s) {
            const uint32_t segIdx = e * W + s;
            const uint32_t rows = recv[s][e];
            // 源行基（rank s 的 gmm_x 中发给 me 的 e 块起点）= segAOff[segIdx]（同布局）
            const uint32_t aRowBase = segAOff[segIdx];
            const uint32_t cRowBase = segCOff[segIdx];
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t nn = 0; nn < N; nn += 128) {
                    float acc = 0.0f;
                    for (uint32_t k = 0; k < K; ++k) {
                        const uint32_t aIdx = (aRowBase + r) * K + k;
                        const float a = E4m3ToFloat(aHost[aIdx]) *
                                        E8m0ToFloat(aScaleHost[(aIdx / K) * scaleKLen + (k / 64) * 2 + (k % 64) / 32]);
                        const float b =
                            E4m3ToFloat(bHost[(e * K + k) * N + nn]) *
                            E8m0ToFloat(bScaleHost[((e * (K / 64) + k / 64) * N + nn) * 2 + (k % 64) / 32]);
                        acc += a * b;
                    }
                    const uint32_t bits = static_cast<uint32_t>(cHost[(cRowBase + r) * N + nn]) << 16;
                    float got = 0.0f;
                    memcpy(&got, &bits, sizeof(got));
                    const double rel = std::fabs(got - acc) / std::max<double>(std::fabs(acc), 1.0);
                    if (rel > maxRel) maxRel = rel;
                    if (rel > 0.03) bad++;
                    sample++;
                }
            }
        }
    }
    printf("ST segment-mode: sample=%lld bad=%lld maxRel=%f => %s\n", sample, bad, maxRel, bad == 0 ? "PASS" : "FAIL");

    aclrtFree(aDev);
    aclrtFree(aScaleDev);
    aclrtFree(bDev);
    aclrtFree(bScaleDev);
    aclrtFree(cDev);
    aclrtFree(tdDev);
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return bad == 0 ? 0 : 1;
}
