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
 * \file test_local_ms_rw.cpp
 * \brief 本地读写 MS 测试 Host 侧代码，通过 <<<>>> 直调 Kernel
 */

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <getopt.h>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hcom.h"
#include "securec.h"
#include "./local_ms_rw.h"

using namespace std;

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

// ========== hcomm 暴露的接口声明 ==========
// 返回值类型
typedef int32_t CcuResult;
constexpr CcuResult CCU_SUCCESS = 0;

typedef enum {
    RES_ADD_TYPE_CCU = 4, /**< CCUA MS address. resId 0..3 maps to CCUA0..CCUA3. */
    RES_ADD_TYPE_RESERVED = 255
} CcuResAddrType;

/**
 * @brief 将CCU设备资源映射为当前进程可访问的VA。
 * @param[in]  resType 资源类型。当前支持RES_ADD_TYPE_CCU。
 * @param[in]  resId   资源ID。RES_ADD_TYPE_CCU场景下，0..3分别表示CCUA0..CCUA3的MS地址。
 * @param[out] va      输出映射后的VA，不可为nullptr。
 * @return CcuResult。CCU_SUCCESS表示成功；CCU_E_PARA表示resType/resId非法；
 *         CCU_E_PTR表示va为nullptr；其余为底层runtime错误。
 */
extern "C" CcuResult HcommCcuResAddrGet(uint32_t resType, uint32_t resId, uint64_t *va);
extern "C" CcuResult CcuGetTlvHandle(uint32_t devLogicId, void **outTlvHandle);
extern "C" CcuResult CcuPutTlvHandle(void *tlvHandle);

// ========== 系统周期频率 ==========
constexpr uint32_t SYS_CYCLE_FREQ_MHZ = 1000U;
constexpr double NS_PER_CYCLE = 1.0; // 1000MHz → 1 cycle = 1 ns

// ========== 默认参数 ==========
constexpr int64_t BLOCK_NUM = 1;
constexpr int64_t HCCL_BUFFER_SIZE = 200;
constexpr uint64_t MIN_DATA_SIZE = 32;                  // 32B，DataCopy 最小对齐
constexpr uint64_t MAX_DATA_SIZE_SINGLE_BUF = 240 * 1024;  // 240KB，单 buffer 上限（WRITE/READ）
constexpr uint64_t MAX_DATA_SIZE_DOUBLE_BUF = 120 * 1024;  // 120KB，双 buffer 上限（WTR/READ_WRITE）
constexpr uint32_t RESULT_WORDS = 8;
int32_t g_rankId = 0;
int32_t g_ranksize = 1;        // 卡数（本地测试默认单卡）
uint32_t g_mode0 = 3;          // 0=写, 1=读, 2=读写混合, 3=先写后读
uint32_t g_enableWarmup = 0;   // 预热次数，0=不预热
uint32_t g_msId = 0;           // 0..3，选择测试哪块 MS
uint32_t g_loopCount = 100;    // 正式打点内循环次数
uint64_t g_dataSize = 4096;    // 默认 4KB

static const struct option kLongOptions[] = {{"rank_id", 1, 0, 'a'},
                                             {"ranksize", 1, 0, 'd'},
                                             {"mode0", 1, 0, 'm'},
                                             {"warmup", 1, 0, 'w'},
                                             {"ms_id", 1, 0, 'u'},
                                             {"loop_count", 1, 0, 'l'},
                                             {"data_size", 1, 0, 's'},
                                             {0, 0, 0, 0}};

void GetOption(int argc, char **argv)
{
    int optionIndex = 0, c = 1;
    while (c != -1) {
        c = getopt_long(argc, argv, "a:d:m:w:u:l:s:", kLongOptions, &optionIndex);
        switch (c) {
            case 'a':
                g_rankId = atoi(optarg);
                break;
            case 'd':
                g_ranksize = atoi(optarg);
                break;
            case 'm': {
                std::string m(optarg);
                if (m == "write" || m == "WRITE") { g_mode0 = 0; }
                else if (m == "read" || m == "READ") { g_mode0 = 1; }
                else if (m == "read_write" || m == "READ_WRITE" || m == "rw" || m == "RW") { g_mode0 = 2; }
                else if (m == "write_then_read" || m == "WRITE_THEN_READ" || m == "wtr" || m == "WTR") { g_mode0 = 3; }
                else if (m == "dual_core_rw" || m == "DUAL_CORE_RW" || m == "dcrw" || m == "DCRW") { g_mode0 = 4; }
                else { g_mode0 = static_cast<uint32_t>(atoi(optarg)); }
                break;
            }
            case 'w':
                g_enableWarmup = (uint32_t)atoi(optarg);
                break;
            case 'u':
                g_msId = (uint32_t)atoi(optarg);
                break;
            case 'l':
                g_loopCount = (uint32_t)atoi(optarg);
                break;
            case 's':
                g_dataSize = (uint64_t)atoll(optarg);
                break;
            default:
                break;
        }
    }
}

const char *GetMode0Str(uint32_t mode0)
{
    switch (mode0) {
        case 0: return "WRITE";
        case 1: return "READ";
        case 2: return "READ_WRITE";
        case 3: return "WRITE_THEN_READ";
        case 4: return "DUAL_CORE_RW";
        default: return "UNKNOWN";
    }
}

int main(int argc, char *argv[])
{
    GetOption(argc, argv);

    LOG_PRINT("[INFO] ===== 本地读写 MS 测试 =====\n");
    LOG_PRINT("[INFO] rank_id=%d, ranksize=%d, mode0=%s, warmup=%u, ms_id=%u, loop_count=%u, "
              "data_size=%lu bytes (%.2f KB)\n",
              g_rankId, g_ranksize, GetMode0Str(g_mode0), g_enableWarmup, g_msId,
              g_loopCount, g_dataSize, (double)g_dataSize / 1024.0);

    CHECK_RET(g_mode0 <= 4, LOG_PRINT("[ERROR] mode0 invalid: %u\n", g_mode0); return -1);
    CHECK_RET(g_enableWarmup <= 1000, LOG_PRINT("[ERROR] warmup invalid: %u (0-1000)\n", g_enableWarmup); return -1);
    CHECK_RET(g_msId < 4, LOG_PRINT("[ERROR] ms_id invalid: %u, valid range is 0..3\n", g_msId); return -1);
    CHECK_RET(g_loopCount > 0, LOG_PRINT("[ERROR] loop_count must be greater than 0.\n"); return -1);

    // ========== dataSize 校验 ==========
    CHECK_RET(g_dataSize >= MIN_DATA_SIZE && (g_dataSize % MIN_DATA_SIZE) == 0,
              LOG_PRINT("[ERROR] dataSize=%lu 不满足 32B 对齐要求\n", g_dataSize); return -1);
    if (g_mode0 == 0 || g_mode0 == 1) {
        if (g_dataSize > MAX_DATA_SIZE_SINGLE_BUF) {
            LOG_PRINT("[ERROR] dataSize=%lu 超过单 buffer UB 上限 %lu bytes\n",
                      g_dataSize, MAX_DATA_SIZE_SINGLE_BUF);
            return -1;
        }
    } else {
        if (g_dataSize > MAX_DATA_SIZE_DOUBLE_BUF) {
            LOG_PRINT("[ERROR] dataSize=%lu 超过双 buffer UB 上限 %lu bytes\n",
                      g_dataSize, MAX_DATA_SIZE_DOUBLE_BUF);
            return -1;
        }
    }

    // ========== acl 初始化 ==========
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret: %d\n", ret); return ret);
    // deviceid == rankid
    ret = aclrtSetDevice(g_rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret: %d\n", ret); return ret);

    aclrtContext context;
    ret = aclrtCreateContext(&context, g_rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret: %d\n", ret); return ret);

    aclrtStream stream;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret: %d\n", ret); return ret);

    // ========== 初始化通信域 ==========
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclDeterministic = 1;
    config.hcclBufferSize = HCCL_BUFFER_SIZE;
    ret = strcpy_s(config.hcclCommName, COMM_NAME_MAX_LENGTH - 1, "hccl_comm_test");
    CHECK_RET(ret == EOK, LOG_PRINT("[ERROR] hcclCommName strcpy failed.\n"); return -1);

    const char *rankTableFile = getenv("RANK_TABLE_FILE");
    CHECK_RET(rankTableFile != nullptr, LOG_PRINT("[ERROR] get RANK_TABLE_FILE env failed.\n"); return -1);

    HcclComm comms;
    ret = HcclCommInitClusterInfoConfig(rankTableFile, g_rankId, &config, &comms);
    CHECK_RET(ret == HCCL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitClusterInfoConfig failed. ret: %d\n", ret);
              return ret);
    LOG_PRINT("[INFO] 通信域初始化成功。\n");
    ret = aclrtSetCurrentContext(context);

    // ========== 初始化 CCU Handle ==========
    LOG_PRINT("[INFO] 获取 CCU TLV Handle...\n");
    void *ccuHandle = nullptr;
    CcuResult ccuRet = CcuGetTlvHandle((uint32_t)g_rankId, &ccuHandle);
    CHECK_RET(ccuRet == CCU_SUCCESS, LOG_PRINT("[ERROR] CcuGetTlvHandle failed. ret: %d\n", ccuRet); return -1);
    CHECK_RET(ccuHandle != nullptr, LOG_PRINT("[ERROR] ccuHandle is nullptr.\n"); return -1);
    LOG_PRINT("[INFO] ccuHanle=%p\n", ccuHandle);
    // ========== 获取 MS 地址 ==========
    uint64_t msAddrs[4];
    for (uint32_t i = 0; i < 4; i++) {
        ccuRet = HcommCcuResAddrGet(RES_ADD_TYPE_CCU, i, &msAddrs[i]);
        CHECK_RET(ccuRet == CCU_SUCCESS,
                  LOG_PRINT("[ERROR] HcommCcuResAddrGet failed. resId: %u, ret: %d\n", i, ccuRet); return -1);
        LOG_PRINT("[INFO] MSAddr%d (CCUA%d): va=0x%016lx\n", i, i, msAddrs[i]);
    }

    // ========== 分配计时 GM ==========
    void *timeGm = nullptr;
    ret = aclrtMalloc(&timeGm, RESULT_WORDS * sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc timeGm failed. ret: %d\n", ret); return ret);

    // ========== <<<>>> 调用 Kernel ==========
    uint32_t blockDim = (g_mode0 == 4) ? 2 : (uint32_t)BLOCK_NUM;
    LOG_PRINT("[INFO] 启动 Kernel (blockDim=%u)...\n", blockDim);
    local_ms_rw_demo(g_mode0, g_enableWarmup, g_msId, g_loopCount, g_dataSize, blockDim, stream,
                     (uint8_t *)msAddrs[0], (uint8_t *)msAddrs[1], (uint8_t *)msAddrs[2], (uint8_t *)msAddrs[3],
                     (uint8_t *)timeGm);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStream failed. ret: %d\n", ret); return ret);
    LOG_PRINT("[INFO] Kernel 执行完成。\n");

    // ========== 读取计时结果 ==========
    uint64_t timeHost[RESULT_WORDS] = {0};
    ret = aclrtMemcpy(timeHost, sizeof(timeHost), timeGm, sizeof(timeHost), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy timeGm failed. ret: %d\n", ret); return ret);

    uint64_t startCycle = timeHost[0];
    uint64_t endCycle = timeHost[1];
    uint64_t verifyResult = timeHost[2];
    uint64_t warmupCycles = timeHost[3];
    uint64_t startCycle2 = timeHost[4];
    uint64_t endCycle2 = timeHost[5];
    uint64_t warmupCycles2 = timeHost[6];
    uint64_t elapsedCycles = endCycle - startCycle;
    double elapsedNs = (double)elapsedCycles * NS_PER_CYCLE;
    double avgNs = elapsedNs / static_cast<double>(g_loopCount);
    double elapsedUs = elapsedNs / 1000.0;
    double dataKB = (double)g_dataSize / 1024.0;
    uint64_t bytesPerLoop = g_dataSize;
    if (g_mode0 == 2 || g_mode0 == 3) {
        bytesPerLoop *= 2;
    }
    double bandwidthGBs = (double)bytesPerLoop * (double)g_loopCount / (elapsedNs);

    LOG_PRINT("\n");
    LOG_PRINT("===== MS 测试结果 =====\n");
    LOG_PRINT("  模式:         %s\n", GetMode0Str(g_mode0));
    LOG_PRINT("  warmup:       %u\n", g_enableWarmup);
    LOG_PRINT("  MS ID:        %u\n", g_msId);
    LOG_PRINT("  loop_count:   %u\n", g_loopCount);
    LOG_PRINT("  数据量:       %.2f KB (%lu bytes)\n", dataKB, g_dataSize);
    if (g_mode0 == 4) {
        double elapsedCycles2 = (double)(endCycle2 - startCycle2);
        double elapsedNs2 = elapsedCycles2 * NS_PER_CYCLE;
        double avgNs2 = elapsedNs2 / static_cast<double>(g_loopCount);
        double bwWrite = (double)g_dataSize * (double)g_loopCount / elapsedNs;
        double bwRead = (double)g_dataSize * (double)g_loopCount / elapsedNs2;
        LOG_PRINT("  --- 写核 (core 0) ---\n");
        LOG_PRINT("  warmupCycle:  %lu\n", warmupCycles);
        LOG_PRINT("  startCycle:   %lu\n", startCycle);
        LOG_PRINT("  endCycle:     %lu\n", endCycle);
        LOG_PRINT("  耗时:         %.2f ns\n", elapsedNs);
        LOG_PRINT("  平均:         %.4f ns/loop\n", avgNs);
        LOG_PRINT("  带宽:         %.4f GB/s\n", bwWrite);
        LOG_PRINT("  --- 读核 (core 1) ---\n");
        LOG_PRINT("  warmupCycle:  %lu\n", warmupCycles2);
        LOG_PRINT("  startCycle2:  %lu\n", startCycle2);
        LOG_PRINT("  endCycle2:    %lu\n", endCycle2);
        LOG_PRINT("  耗时:         %.2f ns\n", elapsedNs2);
        LOG_PRINT("  平均:         %.4f ns/loop\n", avgNs2);
        LOG_PRINT("  带宽:         %.4f GB/s\n", bwRead);
    } else {
        LOG_PRINT("  warmupCycle:  %lu\n", warmupCycles);
        LOG_PRINT("  startCycle:   %lu\n", startCycle);
        LOG_PRINT("  endCycle:     %lu\n", endCycle);
        LOG_PRINT("  耗时:         %.2f ns (%.2f us)\n", elapsedNs, elapsedUs);
        LOG_PRINT("  平均:         %.4f ns/loop\n", avgNs);
        LOG_PRINT("  带宽:         %.4f GB/s\n", bandwidthGBs);
        if (g_mode0 == 3) {
            LOG_PRINT("  数据校验:     %s\n", verifyResult == 0 ? "通过" : "失败");
        }
    }
    LOG_PRINT("=======================\n");

    // ========== 释放资源 ==========
    aclrtFree(timeGm);
    if (ccuHandle != nullptr) {
        CcuPutTlvHandle(ccuHandle);
    }
    HcclCommDestroy(comms);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(g_rankId);
    aclFinalize();

    return 0;
}
