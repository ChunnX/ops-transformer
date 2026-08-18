/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

/*!
 * \file sample_main.cpp
 * \brief Host 侧入口：ThreadContext + Sample() + main()（第三阶段：通算并行流水）
 *
 * 第三阶段 Case：M=512（每 rank），tileM=256，commTurn=2，K=256，N=256，FP16 A/B，BF16 C
 *   - sendBuf = 512*256*2 = 256KB（每 rank，分 2 tile 各 128KB）
 *   - recvBuf = 2*512*256*2 = 512KB（AllGather 后 2 rank × 512 行）
 *   - outputBuf = 2*512*256*2 = 512KB（BF16，2 round matmul 各 512 行）
 *   - 数据填充：rank0/rank1 前 256 行 1.0(0x3C00) 后 256 行 2.0(0x4000)
 *   - tilingData.m = rankSize * tileM = 2*256 = 512（每 round matmul 的 M）
 *
 * Sample 流程：init HCCL → 创建双流 → 分配 send/recv/weight/output → 填充数据
 * → 构造 MatmulTilingData → AllGatherMatmulCustom（1 次 CCU launch 内循环 commTurn + 1 次 MIX launch）→ sync
 * → 读回 outputBuf → 落地 bin
 *
 * 本文件被 ccu_all_gather_matmul_kernel.asc 末尾 #include，不需要独立编译。
 * 依赖 .asc 中已定义的宏（ACLCHECK/HCCLCHECK）、CCU 通信层（ops_comm_ag）、
 * 融合 kernel（all_gather_matmul_kernel）、Host 编排（AllGatherMatmulCustom）。
 */

struct ThreadContext {
    HcclRootInfo *rootInfo;
    uint32_t device;
    uint32_t devCount;
};

int Sample(void *arg)
{
    ThreadContext *ctx = (ThreadContext *)arg;
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    void *weightBuf = nullptr;   // matmul 的 B 矩阵（K×N, FP16）
    void *outputBuf = nullptr;   // matmul 的 C 输出（rankSize*M×N, BF16）
    uint32_t device = ctx->device;
    if (ctx->devCount > CCU_MAX_RANK_SIZE) {
        std::cout << "devCount " << ctx->devCount << " exceeds CCU_MAX_RANK_SIZE "
                  << CCU_MAX_RANK_SIZE << std::endl;
        return HCCL_E_PARA;
    }
    // 第三阶段 Case：M=512（每 rank），tileM=256，commTurn=M/tileM=2，K=256，N=256
    // 修改 M 或 tileM 即可调整 commTurn（需 ≤ MAX_COMMTURN=16）
    constexpr uint64_t M = 512;
    constexpr uint64_t tileM = 256;  // 每 round AllGather 的行数
    constexpr uint64_t K = 256;
    constexpr uint64_t N = 256;
    constexpr uint64_t sendCount = M * K; // 每个 rank 输入 M*K 个 FP16 元素 = 512*256
    const uint64_t rankSize = ctx->devCount;
    const uint64_t recvCount = sendCount * rankSize;            // AllGather 后 A 总元素数 = 2*512*256
    const uint64_t weightCount = K * N;                         // B 矩阵元素数 = 256*256
    const uint64_t outputCount = rankSize * M * N;              // C 输出元素数（BF16）= 2*512*256
    // FP16: 2 bytes per element；BF16: 2 bytes per element
    // Host 侧不用 half/bfloat16_t 类型，避免软浮点符号依赖
    size_t sendSize = sendCount * 2;        // 512*256*2 = 256KB
    size_t recvSize = recvCount * 2;        // 2*512*256*2 = 512KB
    size_t weightSize = weightCount * 2;    // 256*256*2 = 128KB
    size_t outputSize = outputCount * 2;    // 2*512*256*2 = 512KB
    // 设置当前线程操作的设备
    ACLCHECK(aclrtSetDevice(static_cast<int32_t>(device)));

    // 初始化集合通信域
    HcclComm hcclComm;
    HCCLCHECK(HcclCommInitRootInfo(ctx->devCount, ctx->rootInfo, device, &hcclComm));

    // 创建任务流（streamAivAic：AIC 直轮询 + matmul；streamCcu：1 次 CCU launch 内循环 commTurn 次）
    aclrtStream streamAivAic;
    ACLCHECK(aclrtCreateStream(&streamAivAic));

    aclrtStream streamCcu;
    ACLCHECK(aclrtCreateStream(&streamCcu));

    // 申请集合通信操作的 Device 内存
    ACLCHECK(aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&weightBuf, weightSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&outputBuf, outputSize, ACL_MEM_MALLOC_HUGE_ONLY));

    // 申请 Host 内存用于存放输入数据，并将内容初始化
    // Host 侧用 uint16_t 承载 FP16 数据（与 apace ST 一致，不依赖 half 软模拟库）
    // 第三阶段数据填充：每 rank 前 tileM=256 行 1.0(0x3C00) 后 256 行 2.0(0x4000)
    //   rank0/rank1 均如此（两 rank 数据相同，便于校验 AllGather 后 tile0=256.0 tile1=512.0）
    void *hostBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&hostBuf, sendSize));
    uint16_t *tmpHostBuff = static_cast<uint16_t *>(hostBuf);
    for (uint64_t row = 0; row < M; ++row) {
        // 前 tileM 行 1.0，后 tileM 行 2.0
        uint16_t fillBits = (row < tileM) ? 0x3C00 : 0x4000;  // FP16 1.0 / 2.0
        for (uint64_t col = 0; col < K; ++col) {
            tmpHostBuff[row * K + col] = fillBits;
        }
    }

    // 将 Host 侧输入数据拷贝到 Device 侧
    ACLCHECK(aclrtMemcpy(sendBuf, sendSize, hostBuf, sendSize, ACL_MEMCPY_HOST_TO_DEVICE));
    // 释放 Host 侧内存
    ACLCHECK(aclrtFreeHost(hostBuf));

    // ---- 填充权重 B（全 1.0=0x3C00，FP16）----
    // B 全 1.0 时，C = A @ B，C[i][j] = sum(A[i][0..K-1]) = K * A[i][0]
    // tile0 行（A=1.0）：C = 256.0；tile1 行（A=2.0）：C = 512.0
    void *hostWeight = nullptr;
    ACLCHECK(aclrtMallocHost(&hostWeight, weightSize));
    uint16_t *tmpHostWeight = static_cast<uint16_t *>(hostWeight);
    for (uint16_t *p = tmpHostWeight; p < tmpHostWeight + weightCount; ++p) {
        *p = 0x3C00;  // FP16 1.0
    }
    ACLCHECK(aclrtMemcpy(weightBuf, weightSize, hostWeight, weightSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtFreeHost(hostWeight));

    // ---- 构造 MatmulTilingData（每 round 一组，M = rankSize * tileM = 2*256 = 512）----
    // m = rankSize * tileM（单 round AllGather 后总行数），n = N，k = K
    // baseM=256, baseN=256, baseK=64（mat_mul_basic 默认），mL1=baseM, nL1=baseN, kL1=baseK*2
    MatmulTilingData tilingData{};
    tilingData.m = static_cast<uint32_t>(rankSize * tileM);  // 2*256 = 512
    tilingData.n = static_cast<uint32_t>(N);                  // 256
    tilingData.k = static_cast<uint32_t>(K);                  // 256
    tilingData.baseM = 256;
    tilingData.baseN = 256;
    tilingData.baseK = 64;
    tilingData.mL1 = 256;
    tilingData.nL1 = 256;
    tilingData.kL1 = 128;  // = baseK * 2
    tilingData.mTailCnt = 1;
    tilingData.nTailCnt = 1;
    tilingData.isHf32 = 0;  // FP16 输入默认 FP32 累加
    // usedCoreNum：MIX 1:1 下 blockNum 是 AIV+AIC 对数。
    // 极简 case 用 PlatformAscendCManager 查询 AIC 核数，失败回退固定值 8。
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
    std::cout << "rankId: " << device << ", phase3 matmul M=" << tilingData.m << " N=" << tilingData.n
              << " K=" << tilingData.k << ", tileM=" << tileM << ", commTurn=" << (M / tileM)
              << ", usedCoreNum=" << tilingData.usedCoreNum << std::endl;

    // 执行第三阶段通算并行：commTurn round CCU AllGather（streamCcu 串行）+ AIC 直轮询 + commTurn 次 matmul
    // matmul 的 A = recvBuf（AllGather 结果，零拷贝），B = weightBuf，C = outputBuf
    // tile0 matmul 与 CCU round1 通信并行 overlap
    AllGatherMatmulContext mmCtx;
    HCCLCHECK(AllGatherMatmulCustom(sendBuf, recvBuf, weightBuf, outputBuf, sendCount,
        HCCL_DATA_TYPE_FP16, hcclComm, streamAivAic, streamCcu, tilingData,
        static_cast<uint32_t>(tileM), static_cast<uint32_t>(K), mmCtx));

    // 阻塞等待任务流中的集合通信 + matmul 任务执行完成
    ACLCHECK(aclrtSynchronizeStream(streamAivAic));
    ACLCHECK(aclrtSynchronizeStream(streamCcu));

    // 释放 CCU 资源（sync 后 kernel 已完成，可安全释放）
    if (mmCtx.d_inputOffsetsAddr) {
        ACLCHECK(aclrtFree(reinterpret_cast<void *>(mmCtx.d_inputOffsetsAddr)));
    }
    if (mmCtx.d_outputOffsetsAddr) {
        ACLCHECK(aclrtFree(reinterpret_cast<void *>(mmCtx.d_outputOffsetsAddr)));
    }
    if (mmCtx.insHandle) {
        HCCLCHECK(HcommCcuInsDestroy(mmCtx.insHandle));
    }

    // 将 Device 侧结果（outputBuf，BF16）拷贝到 Host，落地 bin 文件（用 Python 校验精度）
    std::this_thread::sleep_for(std::chrono::seconds(ctx->device));
    void *resultBuff;
    ACLCHECK(aclrtMallocHost(&resultBuff, outputSize));
    ACLCHECK(aclrtMemcpy(resultBuff, outputSize, outputBuf, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));

    // 落地 output/rank{id}/npu_out.bin（BF16，Host 侧用 uint16_t 承载）
    // 布局：[tile0 输出（rankSize*tileM=512 行 × N=256，全 256.0）| tile1 输出（512 行 × 256，全 512.0）]
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
    HCCLCHECK(HcclCommDestroy(hcclComm));  // 销毁通信域
    if (sendBuf) {
        ACLCHECK(aclrtFree(sendBuf));      // 释放 Device 侧内存
    }
    if (recvBuf) {
        ACLCHECK(aclrtFree(recvBuf));      // 释放 Device 侧内存
    }
    if (weightBuf) {
        ACLCHECK(aclrtFree(weightBuf));    // 释放 weightBuf
    }
    if (outputBuf) {
        ACLCHECK(aclrtFree(outputBuf));    // 释放 outputBuf
    }
    ACLCHECK(aclrtDestroyStream(streamCcu));     // 销毁任务流
    ACLCHECK(aclrtDestroyStream(streamAivAic));  // 销毁任务流
    ACLCHECK(aclrtResetDevice(device));    // 重置设备
    return 0;
}

int main()
{
    // 设备资源初始化
    ACLCHECK(aclInit(NULL));
    // 查询设备数量
    uint32_t devCount;
    ACLCHECK(aclrtGetDeviceCount(&devCount));
    std::cout << "Found " << devCount << " NPU device(s) available" << std::endl;

    // 环境检查：CCU_SCHED 模式必须设置
    const char *schedMode = getenv("HCCL_OP_EXPANSION_MODE");
    if (schedMode == nullptr || std::string(schedMode) != "CCU_SCHED") {
        std::cout << "WARNING: HCCL_OP_EXPANSION_MODE=" << (schedMode ? schedMode : "(unset)")
                  << " (expected CCU_SCHED), CCU operations may fail!" << std::endl;
    } else {
        std::cout << "HCCL_OP_EXPANSION_MODE=" << schedMode << " OK" << std::endl;
    }

    int32_t rootRank = 0;
    ACLCHECK(aclrtSetDevice(rootRank));
    // 生成 Root 节点信息，各线程使用同一份 RootInfo
    void *rootInfoBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&rootInfoBuf, sizeof(HcclRootInfo)));
    HcclRootInfo *rootInfo = (HcclRootInfo *)rootInfoBuf;
    HCCLCHECK(HcclGetRootInfo(rootInfo));

    // 启动线程执行集合通信操作
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

    // 释放资源
    ACLCHECK(aclrtFreeHost(rootInfoBuf));  // 释放 Host 内存
    ACLCHECK(aclFinalize());               // 设备去初始化
    return 0;
}
