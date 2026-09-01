/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sample_main.cpp
 * \brief Host 侧入口：阶段1 CCU AllToAll 纯通信验证（FP16）
 *
 * 用例参数：
 *   ./ccu_alltoall                          # precision 模式，默认参数
 *   ./ccu_alltoall precision [elemsPerBlock] [commTurn]
 *   ./ccu_alltoall perf [elemsPerBlock] [commTurn] [loop]
 *   默认: elemsPerBlock=65536（128KB FP16 块），commTurn=1，loop=10
 *
 * ─── 溯源编码数据设计（每个 slice、每个元素都不同；任何错位/漏发/块混淆
 *      可从位型直接定位，方便精度分析）───
 *   每个 uint16 元素的位型 = 溯源编码（纯字节搬运验证，位型不解释为 FP16 数值）：
 *     bit[15:12] = srcRank   数据产生 rank（填充侧 = 本 rank 的 device 号）
 *     bit[11:8]  = dstBlock  块号（该块发往的目标 rank）
 *     bit[7:0]   = tPos      块内元素位置摘要 = (t ^ (t>>8)) & 0xFF
 *                  （非线性混合：块内错位 Δ 及 256 整数倍类偏移均可检出——
 *                    裸 t&0xFF 对 Δ≡0 (mod 256) 的 round 边界类错位是盲区）
 *   发送侧：rank i 的 sendBuf 第 j 块元素 t = Encode(src=i, dst=j, t)
 *   期望侧：rank i 的 recvBuf 第 r 块元素 t = Encode(src=r, dst=i, t)
 *     —— AllToAll 语义：recv_i[r] = send_r[i]，即收到 rank r 发来的"它要给我的第 r 块"
 *
 *   错位诊断：actual 位型直接读出 srcRank/dstBlock（数据实际来自谁/本想去谁），
 *   与期望对比即可分类错误类型：src 混淆 / 块混淆 / 块内偏移 / 数据损坏
 *   （示例 R=2, i=0：recv 块0 期望 0x0900；若实际 0x1900 → srcRank=1 是自己的数据
 *     → 本地/远端搞反；若实际 0x0A00 → dstBlock=1 是发给 rank1 的块 → src 块错位）
 *
 * 退出码：precision 模式所有 rank 校验 PASS → 0；任一 rank FAIL 或参数错误 → 非 0
 * （run.sh 的 set -e 依赖此退出码形成"编译→运行→校验"闭环）
 */

// 运行模式
enum class RunMode {
    PRECISION,  // 精度验证：单次执行 + 读回校验 + 写 output bin
    PERF        // 性能测试：循环 loop 次，不写 output bin
};

// 命令行/线程参数（默认值的单一出处：main 的解析从这里取初值）
struct ThreadContext {
    HcclRootInfo *rootInfo;
    uint32_t device;
    uint32_t devCount;
    RunMode mode{RunMode::PRECISION};
    uint64_t elemsPerBlock{65536};  // 每块 FP16 元素数（默认 128KB 块）
    uint32_t commTurn{1};           // 通信轮数
    uint32_t loopCount{10};         // perf 模式循环次数
    int ret{0};                     // Sample 执行结果（main 聚合为进程退出码）
};

// ============================================================================
// A2aElemPattern — AllToAll 溯源编码（uint16 位型，见文件头"溯源编码数据设计"）
//   bit[15:12]=srcRank, bit[11:8]=dstBlock, bit[7:0]=tPos=(t ^ (t>>8)) & 0xFF
// 纯字节搬运验证专用：CCU Write/GroupCopy 均为字节级拷贝，dtype 仅用于
// SIZE_TABLE 取字节宽度，位型不解释为 FP16 数值（阶段3 另行设计数值型数据）
// ============================================================================
static inline uint16_t A2aElemPattern(uint32_t srcRank, uint32_t dstBlock, uint64_t t)
{
    return static_cast<uint16_t>(((srcRank & 0xF) << 12)
        | ((dstBlock & 0xF) << 8)
        | ((t ^ (t >> 8)) & 0xFF));
}

// 用法提示（参数解析失败/非法 mode 时打印）
static void PrintUsage(const char *prog)
{
    std::cout << "Usage:\n"
              << "  " << prog << "                          # precision, defaults\n"
              << "  " << prog << " precision [elemsPerBlock] [commTurn]\n"
              << "  " << prog << " perf [elemsPerBlock] [commTurn] [loop]\n"
              << "Defaults: elemsPerBlock=65536 (128KB FP16 block), commTurn=1, loop=10"
              << std::endl;
}

// ----------------------------------------------------------------------------
// FillSendBuffer — 填充 sendBuf（溯源编码：第 j 块元素 t = Encode(src=myRank, dst=j, t)）
// 每个 slice（块）、每个元素的数据都不同——块内错位/元素级偏移均可检出
// ----------------------------------------------------------------------------
static int FillSendBuffer(uint32_t myRank, uint64_t rankSize, uint64_t elemsPerBlock,
                          void *sendBuf, uint64_t sendSize)
{
    void *hostBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&hostBuf, sendSize));
    uint16_t *tmpHost = static_cast<uint16_t *>(hostBuf);
    for (uint64_t j = 0; j < rankSize; j++) {
        uint16_t *blockStart = tmpHost + j * elemsPerBlock;
        for (uint64_t t = 0; t < elemsPerBlock; t++) {
            blockStart[t] = A2aElemPattern(myRank, static_cast<uint32_t>(j), t);
        }
    }
    ACLCHECK(aclrtMemcpy(sendBuf, sendSize, hostBuf, sendSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtFreeHost(hostBuf));
    return 0;
}

// ----------------------------------------------------------------------------
// RunA2aOnce — 执行一轮 AllToAll（Prepare → 双流 launch → sync → 释放 CCU 资源）
// ----------------------------------------------------------------------------
static HcclResult RunA2aOnce(HcclComm hcclComm, aclrtStream streamAicore, aclrtStream streamCcu,
                             void *sendBuf, void *recvBuf, uint64_t sendCount, uint32_t commTurn)
{
    AllToAllContext a2aCtx;
    HCCLCHECK(AllToAllCustom(sendBuf, recvBuf, sendCount, HCCL_DATA_TYPE_FP16,
        hcclComm, streamAicore, streamCcu, commTurn, a2aCtx));

    // 同步双流，等待通信完成（本函数返回 HcclResult，aclxx 失败须转 HCCL_E_INTERNAL）
    aclError aclRet = aclrtSynchronizeStream(streamAicore);
    if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }
    aclRet = aclrtSynchronizeStream(streamCcu);
    if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }

    // 释放 CCU 资源（与 PrepareAllToAll 的分配配对）
    return ReleaseA2aContext(a2aCtx);
}

// ----------------------------------------------------------------------------
// VerifyAndDumpRecv — precision 模式校验：读回 recvBuf → 逐块逐元素断言 → 首错定位 → 落地 bin
// 返回：0 = PASS；非 0 = FAIL（不匹配元素数或 ACL 错误码，供 Sample 透传为进程退出码）
// ----------------------------------------------------------------------------
static int VerifyAndDumpRecv(uint32_t myRank, uint64_t rankSize, uint64_t elemsPerBlock,
                             void *recvBuf, uint64_t sendSize, uint64_t sendCount)
{
    void *resultBuff = nullptr;
    ACLCHECK(aclrtMallocHost(&resultBuff, sendSize));
    ACLCHECK(aclrtMemcpy(resultBuff, sendSize, recvBuf, sendSize, ACL_MEMCPY_DEVICE_TO_HOST));
    const uint16_t *recvHost = static_cast<const uint16_t *>(resultBuff);

    // 逐块逐元素断言：recvBuf 第 r 块元素 t 期望 = Encode(src=r, dst=myRank, t)
    // （AllToAll 语义 recv_i[r] = send_r[i]）
    uint64_t errCount = 0;
    bool hasFirstErr = false;
    uint64_t firstErrBlock = 0;
    uint64_t firstErrElem = 0;
    uint16_t firstErrExpect = 0;
    uint16_t firstErrActual = 0;
    for (uint64_t r = 0; r < rankSize; r++) {
        const uint16_t *blockStart = recvHost + r * elemsPerBlock;
        for (uint64_t t = 0; t < elemsPerBlock; t++) {
            const uint16_t expectBits = A2aElemPattern(static_cast<uint32_t>(r), myRank, t);
            if (blockStart[t] != expectBits) {
                errCount++;
                if (!hasFirstErr) {
                    hasFirstErr = true;
                    firstErrBlock = r;
                    firstErrElem = t;
                    firstErrExpect = expectBits;
                    firstErrActual = blockStart[t];
                }
            }
        }
    }

    if (errCount == 0) {
        std::cout << "rankId: " << myRank << ", AllToAll PASS: all " << rankSize
                  << " blocks (" << sendCount << " elems) verified" << std::endl;
    } else {
        // 首错定位：块号/元素偏移/期望 vs 实际位型 + 位型直接反解 srcRank/dstBlock
        // （actual 的 srcRank/dstBlock 即错误数据的真实来源与本想去向，
        //   与期望对比可分类：src 混淆 / 块混淆 / 块内偏移 / 数据损坏）
        std::cout << "rankId: " << myRank << ", AllToAll FAIL: " << errCount
                  << " / " << sendCount << " elems mismatch" << std::endl;
        std::cout << "  first error at recv block " << firstErrBlock
                  << ", elem offset " << firstErrElem
                  << ": expect 0x" << std::hex << firstErrExpect
                  << " (srcRank=" << (firstErrExpect >> 12)
                  << ", dstBlock=" << ((firstErrExpect >> 8) & 0xF) << ")"
                  << ", actual 0x" << firstErrActual
                  << " (srcRank=" << (firstErrActual >> 12)
                  << ", dstBlock=" << ((firstErrActual >> 8) & 0xF) << ")"
                  << std::dec << std::endl;
    }

    // 落地 output/rank{id}/npu_out.bin（uint16 位型，供 python 复核）
    std::string outputDir = "output/" + std::to_string(myRank);
    if (system(("mkdir -p " + outputDir).c_str()) != 0) {
        std::cout << "rankId: " << myRank << ", warn: mkdir " << outputDir << " failed" << std::endl;
    }
    std::string outFile = outputDir + "/npu_out.bin";
    std::ofstream ofs(outFile, std::ios::binary);
    ofs.write(static_cast<const char *>(resultBuff), sendSize);
    ofs.close();
    std::cout << "rankId: " << myRank << ", output written to " << outFile
              << " (" << sendSize << " bytes)" << std::endl;

    ACLCHECK(aclrtFreeHost(resultBuff));
    return (errCount == 0) ? 0 : 1;  // FAIL → 非 0 退出码
}

// ============================================================================
// Sample — 单 rank 的执行入口（每个 rank 一个线程调用本函数）
//
// 执行流程：
//   1. 设置 device + 初始化 HCCL 通信组 + 创建双流（streamAicore 桥 kernel / streamCcu 通信）
//   2. 分配 device 内存（sendBuf/recvBuf 等量——AllToAll 无膨胀）
//   3. FillSendBuffer：填充溯源编码数据
//   4. 循环执行 RunA2aOnce（precision 1 次；perf loop 次）
//   5. precision 模式：VerifyAndDumpRecv 校验 + 落地 bin（FAIL → ctx->ret 非 0）
//   6. 释放资源
// ============================================================================
int Sample(void *arg)
{
    ThreadContext *ctx = (ThreadContext *)arg;
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    const uint32_t device = ctx->device;        // 设备管理用（SetDevice 等）
    const uint32_t myRank = ctx->device;        // 通信语义用（数据编码/校验）——与 device 同值
    if (ctx->devCount > CCU_MAX_RANK_SIZE) {
        std::cout << "devCount " << ctx->devCount << " exceeds CCU_MAX_RANK_SIZE "
                  << CCU_MAX_RANK_SIZE << std::endl;
        ctx->ret = HCCL_E_PARA;
        return ctx->ret;
    }

    const uint64_t rankSize = ctx->devCount;           // R = rank 数
    const uint64_t elemsPerBlock = ctx->elemsPerBlock; // 每块 FP16 元素数
    const uint32_t commTurn = ctx->commTurn;

    // 元素数与字节计算（FP16 = 2 bytes）
    const uint64_t sendCount = elemsPerBlock * rankSize;  // 每 rank sendBuf 元素数
    const uint64_t sendSize = sendCount * 2;              // 每 rank sendBuf 字节（recvBuf 等量）

    // ---- 初始化 HCCL + 创建双流 ----
    ACLCHECK(aclrtSetDevice(static_cast<int32_t>(device)));

    HcclComm hcclComm;
    HCCLCHECK(HcclCommInitRootInfo(ctx->devCount, ctx->rootInfo, device, &hcclComm));

    aclrtStream streamAicore;
    ACLCHECK(aclrtCreateStream(&streamAicore));
    aclrtStream streamCcu;
    ACLCHECK(aclrtCreateStream(&streamCcu));

    // ---- 分配 device 内存（sendBuf/recvBuf 等量）----
    ACLCHECK(aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&recvBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY));

    // ---- 填充 sendBuf（溯源编码）----
    ctx->ret = FillSendBuffer(myRank, rankSize, elemsPerBlock, sendBuf, sendSize);
    if (ctx->ret != 0) {
        return ctx->ret;
    }

    std::cout << "rankId: " << myRank << ", mode: "
              << (ctx->mode == RunMode::PERF ? "perf" : "precision")
              << ", FP16 AllToAll: rankSize=" << rankSize
              << ", elemsPerBlock=" << elemsPerBlock
              << " (" << (elemsPerBlock * 2 / 1024) << "KB/block)"
              << ", commTurn=" << commTurn << ", sendBuf=" << (sendSize / 1024) << "KB";
    if (ctx->mode == RunMode::PERF) {
        std::cout << ", loop=" << ctx->loopCount;
    }
    std::cout << std::endl;

    // ---- 执行 AllToAll（precision 1 次；perf 循环 loop 次，每轮独立分配/释放 CCU 资源）----
    uint32_t totalRuns = (ctx->mode == RunMode::PERF) ? ctx->loopCount : 1;
    for (uint32_t iter = 0; iter < totalRuns; iter++) {
        HcclResult runRet = RunA2aOnce(hcclComm, streamAicore, streamCcu,
                                       sendBuf, recvBuf, sendCount, commTurn);
        if (runRet != HCCL_SUCCESS) {
            ctx->ret = static_cast<int>(runRet);
            break;
        }
        if (ctx->mode == RunMode::PERF) {
            std::cout << "rankId: " << myRank << ", perf iter " << (iter + 1) << "/" << totalRuns
                      << " done" << std::endl;
        }
    }

    // ---- precision 模式：校验 + 落地 bin（错峰 sleep 防各 rank stdout 交错）----
    if (ctx->mode == RunMode::PRECISION && ctx->ret == 0) {
        std::this_thread::sleep_for(std::chrono::seconds(ctx->device));
        const int verifyRet = VerifyAndDumpRecv(myRank, rankSize, elemsPerBlock,
                                                recvBuf, sendSize, sendCount);
        if (verifyRet != 0) {
            ctx->ret = verifyRet;  // 精度 FAIL → 非 0 退出码（main 聚合）
        }
    }

    // ---- 释放资源 ----
    HCCLCHECK(HcclCommDestroy(hcclComm));
    if (sendBuf) { ACLCHECK(aclrtFree(sendBuf)); }
    if (recvBuf) { ACLCHECK(aclrtFree(recvBuf)); }
    ACLCHECK(aclrtDestroyStream(streamCcu));
    ACLCHECK(aclrtDestroyStream(streamAicore));
    ACLCHECK(aclrtResetDevice(device));
    return ctx->ret;
}

// ============================================================================
// main — 程序入口（Host 侧）
//
// 执行流程：
//   1. 解析/校验命令行参数（非法参数 → usage + 非 0 退出）
//   2. aclInit + 查询 device 数量（决定 rankSize）+ 检查 CCU_SCHED 模式
//   3. rootRank(0) 获取 HcclRootInfo，为每个 device 创建线程执行 Sample
//   4. join 后聚合各线程结果：任一 rank 失败 → 进程退出码非 0
// ============================================================================
int main(int argc, char *argv[])
{
    // ---- 解析命令行参数（默认值单一出处：ThreadContext 成员默认值）----
    ThreadContext defaults;
    RunMode mode = RunMode::PRECISION;
    uint64_t elemsPerBlock = defaults.elemsPerBlock;
    uint32_t commTurn = defaults.commTurn;
    uint32_t loopCount = defaults.loopCount;

    if (argc >= 2) {
        const std::string modeStr = argv[1];
        if (modeStr == "perf") {
            mode = RunMode::PERF;
        } else if (modeStr == "precision") {
            mode = RunMode::PRECISION;
        } else {
            std::cout << "ERROR: unknown mode '" << modeStr << "'" << std::endl;
            PrintUsage(argv[0]);
            return 1;
        }
        try {
            if (argc >= 3) elemsPerBlock = std::stoull(argv[2]);
            if (argc >= 4) commTurn = static_cast<uint32_t>(std::stoul(argv[3]));
            if (argc >= 5 && mode == RunMode::PERF) loopCount = static_cast<uint32_t>(std::stoul(argv[4]));
        } catch (const std::exception &e) {
            std::cout << "ERROR: invalid numeric argument (" << e.what() << ")" << std::endl;
            PrintUsage(argv[0]);
            return 1;
        }
    }

    // ---- 校验参数（main 侧元素粒度早失败；字节粒度深层校验见 ValidateA2AParams）----
    if (commTurn == 0 || commTurn > MAX_COMMTURN) {
        std::cout << "ERROR: commTurn=" << commTurn << " must be in [1, " << MAX_COMMTURN << "]" << std::endl;
        return 1;
    }
    if (elemsPerBlock == 0 || elemsPerBlock % commTurn != 0) {
        std::cout << "ERROR: elemsPerBlock=" << elemsPerBlock
                  << " must be positive and divisible by commTurn=" << commTurn << std::endl;
        return 1;
    }

    std::cout << "Run mode: " << (mode == RunMode::PERF ? "perf" : "precision")
              << ", elemsPerBlock=" << elemsPerBlock
              << " (" << (elemsPerBlock * 2 / 1024) << "KB/block), commTurn=" << commTurn;
    if (mode == RunMode::PERF) {
        std::cout << ", loop=" << loopCount;
    }
    std::cout << std::endl;

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
        args[i].mode = mode;
        args[i].elemsPerBlock = elemsPerBlock;
        args[i].commTurn = commTurn;
        args[i].loopCount = loopCount;
        threads[i] = std::thread(Sample, (void *)&args[i]);
    }
    for (uint32_t i = 0; i < devCount; i++) {
        threads[i].join();
    }

    ACLCHECK(aclrtFreeHost(rootInfoBuf));
    ACLCHECK(aclFinalize());

    // ---- 聚合各 rank 结果：任一失败 → 进程退出码非 0（run.sh set -e 依赖）----
    int finalRet = 0;
    for (uint32_t i = 0; i < devCount; i++) {
        if (args[i].ret != 0) {
            std::cout << "rankId: " << i << " failed with ret=" << args[i].ret << std::endl;
            finalRet = (finalRet == 0) ? 1 : finalRet;
        }
    }
    if (finalRet == 0) {
        std::cout << "=== ALL RANKS PASS ===" << std::endl;
    } else {
        std::cout << "=== SOME RANKS FAILED ===" << std::endl;
    }
    return finalRet;
}
