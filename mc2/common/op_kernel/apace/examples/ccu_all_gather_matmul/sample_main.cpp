/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the root directory of the software repository for the full text of the License.
 */

/*!
 * \file sample_main.cpp
 * \brief Host 侧入口：第四阶段 MXFP8 AllGather + MXFP8 Matmul 通算并行
 *
 * 第四阶段 Case：M=512（每 rank），tileM=256，commTurn=2，K=256，N=256
 *   A/B = fp8_e4m3fn_t（1 byte），scaleA/scaleB = fp8_e8m0_t（1 byte），C = bfloat16_t（2 bytes）
 *   K=256: ceil(256/64)=4, 4 是偶数 ✓
 *
 *   - sendBuf = 512*256*1 = 128KB（FP8 data，每 rank）
 *   - recvBuf = 2*512*256*1 = 256KB（AllGather 后 FP8 data）
 *   - sendScaleBuf = 512*4*2*1 = 4KB（E8M0 scaleA，每 rank）
 *   - recvScaleBuf = 2*512*4*2*1 = 8KB（AllGather 后 scaleA）
 *   - weightBuf = 256*256*1 = 64KB（FP8 B）
 *   - weightScaleBuf = 4*2*256*1 = 2KB（E8M0 scaleB）
 *   - outputBuf = 2*512*256*2 = 512KB（BF16 C）
 *
 * 数据设计（简化验证）：
 *   A: FP8 E4M3FN 前 tileM=256 行全 1.0(0x3C)，后 256 行全 2.0(0x40)
 *   scaleA: E8M0 全 127（2^0=1.0）
 *   B: FP8 E4M3FN 全 1.0(0x3C)
 *   scaleB: E8M0 全 127（2^0=1.0）
 *   → C[i][j] = sum(dequant(A[i][0..K-1])) * dequant(B) = K * 1.0 = 256.0 (tile0) / K * 2.0 = 512.0 (tile1)
 *
 * Sample 流程：init HCCL → 创建双流 → 分配 send/recv/scale/weight/output → 填充 FP8+E8M0 数据
 * → 构造 MatmulTilingData → AllGatherMatmulCustom → sync → 读回 outputBuf → 落地 bin
 */

struct ThreadContext {
    HcclRootInfo *rootInfo;
    uint32_t device;
    uint32_t devCount;
};

// FP8 E4M3FN 编码（bias=7）：
//   1.0 → exp=7(=0b0111), mant=0(=0b000) → 0_0111_000 = 0x38
//   2.0 → exp=8(=0b1000), mant=0(=0b000) → 0_1000_000 = 0x40
// 注意：0x3C = 0_0111_100 = 1.75（非 1.0），修正为 0x38 = 1.0
static constexpr uint8_t FP8_E4M3FN_1_0 = 0x38;  // 1.0
static constexpr uint8_t FP8_E4M3FN_2_0 = 0x40;  // 2.0
static constexpr uint8_t FP8_E8M0_SCALE_1_0 = 127;  // 2^(127-127) = 1.0

int Sample(void *arg)
{
    ThreadContext *ctx = (ThreadContext *)arg;
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    void *sendScaleBuf = nullptr;
    void *recvScaleBuf = nullptr;
    void *weightBuf = nullptr;
    void *weightScaleBuf = nullptr;
    void *outputBuf = nullptr;
    uint32_t device = ctx->device;
    if (ctx->devCount > CCU_MAX_RANK_SIZE) {
        std::cout << "devCount " << ctx->devCount << " exceeds CCU_MAX_RANK_SIZE "
                  << CCU_MAX_RANK_SIZE << std::endl;
        return HCCL_E_PARA;
    }
    // 第四阶段 Case：M=512, tileM=256, commTurn=2, K=256, N=256
    constexpr uint64_t M = 512;
    constexpr uint64_t tileM = 256;
    constexpr uint64_t K = 256;
    constexpr uint64_t N = 256;
    constexpr uint64_t MXFP_DIVISOR = 64;
    constexpr uint64_t MXFP_MULTI = 2;

    constexpr uint64_t sendCount = M * K;                        // FP8 data 元素数
    constexpr uint64_t scaleKLen = (K + MXFP_DIVISOR - 1) / MXFP_DIVISOR * MXFP_MULTI;  // = 4*2 = 8
    constexpr uint64_t scaleCount = M * scaleKLen;               // scaleA 元素数 = 512*8 = 4096
    const uint64_t rankSize = ctx->devCount;
    const uint64_t recvCount = sendCount * rankSize;             // AllGather 后 A 总元素数
    const uint64_t recvScaleCount = scaleCount * rankSize;       // AllGather 后 scaleA 总元素数
    const uint64_t weightCount = K * N;                          // B 矩阵元素数
    const uint64_t weightScaleCount = scaleKLen * N;             // scaleB 元素数 = 8*256 = 2048
    const uint64_t outputCount = rankSize * M * N;               // C 输出元素数（BF16）

    // 字节计算
    size_t sendSize = sendCount * 1;          // FP8: 1 byte per element
    size_t recvSize = recvCount * 1;
    size_t sendScaleSize = scaleCount * 1;    // E8M0: 1 byte per element
    size_t recvScaleSize = recvScaleCount * 1;
    size_t weightSize = weightCount * 1;
    size_t weightScaleSize = weightScaleCount * 1;
    size_t outputSize = outputCount * 2;      // BF16: 2 bytes per element

    ACLCHECK(aclrtSetDevice(static_cast<int32_t>(device)));

    HcclComm hcclComm;
    HCCLCHECK(HcclCommInitRootInfo(ctx->devCount, ctx->rootInfo, device, &hcclComm));

    aclrtStream streamAivAic;
    ACLCHECK(aclrtCreateStream(&streamAivAic));
    aclrtStream streamCcu;
    ACLCHECK(aclrtCreateStream(&streamCcu));

    // 申请 Device 内存
    ACLCHECK(aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&sendScaleBuf, sendScaleSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&recvScaleBuf, recvScaleSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&weightBuf, weightSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&weightScaleBuf, weightScaleSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&outputBuf, outputSize, ACL_MEM_MALLOC_HUGE_ONLY));

    // ---- 填充 A（FP8 E4M3FN）+ scaleA（E8M0）----
    void *hostBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&hostBuf, sendSize));
    uint8_t *tmpHostBuff = static_cast<uint8_t *>(hostBuf);
    for (uint64_t row = 0; row < M; ++row) {
        uint8_t fillBits = (row < tileM) ? FP8_E4M3FN_1_0 : FP8_E4M3FN_2_0;
        for (uint64_t col = 0; col < K; ++col) {
            tmpHostBuff[row * K + col] = fillBits;
        }
    }
    ACLCHECK(aclrtMemcpy(sendBuf, sendSize, hostBuf, sendSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtFreeHost(hostBuf));

    // ---- 填充 scaleA（E8M0 全 127 = 2^0 = 1.0）----
    void *hostScaleBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&hostScaleBuf, sendScaleSize));
    uint8_t *tmpHostScale = static_cast<uint8_t *>(hostScaleBuf);
    memset(tmpHostScale, FP8_E8M0_SCALE_1_0, sendScaleSize);
    ACLCHECK(aclrtMemcpy(sendScaleBuf, sendScaleSize, hostScaleBuf, sendScaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtFreeHost(hostScaleBuf));

    // ---- 填充 B（FP8 E4M3FN 全 1.0）----
    void *hostWeight = nullptr;
    ACLCHECK(aclrtMallocHost(&hostWeight, weightSize));
    uint8_t *tmpHostWeight = static_cast<uint8_t *>(hostWeight);
    memset(tmpHostWeight, FP8_E4M3FN_1_0, weightSize);
    ACLCHECK(aclrtMemcpy(weightBuf, weightSize, hostWeight, weightSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtFreeHost(hostWeight));

    // ---- 填充 scaleB（E8M0 全 127 = 2^0 = 1.0）----
    void *hostWeightScale = nullptr;
    ACLCHECK(aclrtMallocHost(&hostWeightScale, weightScaleSize));
    uint8_t *tmpHostWeightScale = static_cast<uint8_t *>(hostWeightScale);
    memset(tmpHostWeightScale, FP8_E8M0_SCALE_1_0, weightScaleSize);
    ACLCHECK(aclrtMemcpy(weightScaleBuf, weightScaleSize, hostWeightScale, weightScaleSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtFreeHost(hostWeightScale));

    // ---- 构造 MatmulTilingData（MXFP8）----
    // m = rankSize * tileM（单 round AllGather 后总行数），n = N，k = K
    MatmulTilingData tilingData{};
    tilingData.m = static_cast<uint32_t>(rankSize * tileM);  // 2*256 = 512
    tilingData.n = static_cast<uint32_t>(N);                  // 256
    tilingData.k = static_cast<uint32_t>(K);                  // 256
    tilingData.baseM = 128;   // MXFP8 baseM（较小以适应 L1）
    tilingData.baseN = 256;
    tilingData.baseK = 64;    // MXFP8 需对齐 MXFP_DIVISOR_SIZE=64
    tilingData.stepK = 1;     // kL1 = baseK * stepK = 64
    tilingData.nBufferNum = 2; // L1 double buffer
    tilingData.dbL0c = 1;      // L0C single buffer
    tilingData.kL1 = tilingData.baseK * tilingData.stepK;    // 64
    tilingData.scaleKL1 = tilingData.kL1;   // scaleKL1 = kL1（对齐 64）
    tilingData.mL1 = tilingData.baseM;
    tilingData.nL1 = tilingData.baseN;
    tilingData.mTailTile = 1;
    tilingData.nTailTile = 1;
    tilingData.mBaseTailSplitCnt = 1;
    tilingData.nBaseTailSplitCnt = 1;
    tilingData.mTailMain = 0;
    tilingData.nTailMain = 0;
    tilingData.isHf32 = 0;
    // usedCoreNum：纯 AIC 核数
    {
        auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        if (platform != nullptr) {
            uint32_t aicNum = platform->GetCoreNumAic();
            if (aicNum > 0) {
                tilingData.usedCoreNum = aicNum;
            } else {
                tilingData.usedCoreNum = 8;
            }
        } else {
            tilingData.usedCoreNum = 8;
        }
    }
    std::cout << "rankId: " << device << ", phase4 MXFP8 matmul M=" << tilingData.m << " N=" << tilingData.n
              << " K=" << tilingData.k << ", tileM=" << tileM << ", commTurn=" << (M / tileM)
              << ", usedCoreNum=" << tilingData.usedCoreNum << std::endl;

    // ---- 执行第四阶段通算并行 ----
    AllGatherMatmulContext mmCtx;
    HCCLCHECK(AllGatherMatmulCustom(sendBuf, recvBuf, sendScaleBuf, recvScaleBuf,
        weightBuf, weightScaleBuf, outputBuf, sendCount, scaleCount,
        HCCL_DATA_TYPE_INT8, hcclComm, streamAivAic, streamCcu, tilingData,
        static_cast<uint32_t>(tileM), static_cast<uint32_t>(K), mmCtx));

    ACLCHECK(aclrtSynchronizeStream(streamAivAic));
    ACLCHECK(aclrtSynchronizeStream(streamCcu));

    // 释放 CCU 资源
    if (mmCtx.d_inputOffsetsAddr) {
        ACLCHECK(aclrtFree(reinterpret_cast<void *>(mmCtx.d_inputOffsetsAddr)));
    }
    if (mmCtx.d_outputOffsetsAddr) {
        ACLCHECK(aclrtFree(reinterpret_cast<void *>(mmCtx.d_outputOffsetsAddr)));
    }
    if (mmCtx.d_scaleInputOffsetsAddr) {
        ACLCHECK(aclrtFree(reinterpret_cast<void *>(mmCtx.d_scaleInputOffsetsAddr)));
    }
    if (mmCtx.d_scaleOutputOffsetsAddr) {
        ACLCHECK(aclrtFree(reinterpret_cast<void *>(mmCtx.d_scaleOutputOffsetsAddr)));
    }
    if (mmCtx.insHandle) {
        HCCLCHECK(HcommCcuInsDestroy(mmCtx.insHandle));
    }

    // 读回 outputBuf（BF16）
    std::this_thread::sleep_for(std::chrono::seconds(ctx->device));
    void *resultBuff;
    ACLCHECK(aclrtMallocHost(&resultBuff, outputSize));
    ACLCHECK(aclrtMemcpy(resultBuff, outputSize, outputBuf, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    // 落地 output/rank{id}/npu_out.bin（BF16，Host 侧用 uint16_t 承载）
    std::string outputDir = "output/" + std::to_string(device);
    system(("mkdir -p " + outputDir).c_str());
    std::string outFile = outputDir + "/npu_out.bin";
    std::ofstream ofs(outFile, std::ios::binary);
    ofs.write(static_cast<const char *>(resultBuff), outputSize);
    ofs.close();
    std::cout << "rankId: " << device << ", output written to " << outFile
              << " (" << outputSize << " bytes)" << std::endl;

    ACLCHECK(aclrtFreeHost(resultBuff));

    // 释放资源
    HCCLCHECK(HcclCommDestroy(hcclComm));
    if (sendBuf) { ACLCHECK(aclrtFree(sendBuf)); }
    if (recvBuf) { ACLCHECK(aclrtFree(recvBuf)); }
    if (sendScaleBuf) { ACLCHECK(aclrtFree(sendScaleBuf)); }
    if (recvScaleBuf) { ACLCHECK(aclrtFree(recvScaleBuf)); }
    if (weightBuf) { ACLCHECK(aclrtFree(weightBuf)); }
    if (weightScaleBuf) { ACLCHECK(aclrtFree(weightScaleBuf)); }
    if (outputBuf) { ACLCHECK(aclrtFree(outputBuf)); }
    ACLCHECK(aclrtDestroyStream(streamCcu));
    ACLCHECK(aclrtDestroyStream(streamAivAic));
    ACLCHECK(aclrtResetDevice(device));
    return 0;
}

int main()
{
    ACLCHECK(aclInit(NULL));
    uint32_t devCount;
    ACLCHECK(aclrtGetDeviceCount(&devCount));
    std::cout << "Found " << devCount << " NPU device(s) available" << std::endl;

    const char *schedMode = getenv("HCCL_OP_EXPANSION_MODE");
    if (schedMode == nullptr || std::string(schedMode) != "CCU_SCHED") {
        std::cout << "WARNING: HCCL_OP_EXPANSION_MODE=" << (schedMode ? schedMode : "(unset)")
                  << " (expected CCU_SCHED), CCU operations may fail!" << std::endl;
    } else {
        std::cout << "HCCL_OP_EXPANSION_MODE=" << schedMode << " OK" << std::endl;
    }

    int32_t rootRank = 0;
    ACLCHECK(aclrtSetDevice(rootRank));
    void *rootInfoBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&rootInfoBuf, sizeof(HcclRootInfo)));
    HcclRootInfo *rootInfo = (HcclRootInfo *)rootInfoBuf;
    HCCLCHECK(HcclGetRootInfo(rootInfo));

    std::vector<std::thread> threads(devCount);
    std::vector<ThreadContext> args(devCount);
    for (uint32_t i = 0; i < devCount; i++) {
        args[i].rootInfo = rootInfo;
        args[i].device = i;
        args[i].devCount = devCount;
        threads[i] = std::thread(Sample, (void *)&args[i]);
    }
    for (uint32_t i = 0; i < devCount; i++) {
        threads[i].join();
    }

    ACLCHECK(aclrtFreeHost(rootInfoBuf));
    ACLCHECK(aclFinalize());
    return 0;
}
