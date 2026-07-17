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

#include <iostream>
#include <vector>
#include <string>
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

// MS 内存信息结构体
struct CcuMemInfo {
    uint64_t memType{0};
    uint64_t memVa{0};
    uint32_t memSize{0};
    uint32_t resv{0};
};

// MS_BLOCK 的 bitmap
constexpr uint64_t CCU_MEMTYPE_MS_BLOCK0 = 1ULL << 32;
constexpr uint64_t CCU_MEMTYPE_MS_BLOCK1 = 1ULL << 33;
constexpr uint64_t CCU_MEMTYPE_MS_BLOCK2 = 1ULL << 34;
constexpr uint64_t CCU_MEMTYPE_MS_BLOCK3 = 1ULL << 35;

// hcomm 暴露的两个 C 接口
extern "C" CcuResult CcuGetTlvHandle(uint32_t devLogicId, void **outTlvHandle);
extern "C" CcuResult CcuGetMemInfo(void *tlvHandle, uint32_t udieIdx, uint64_t memTypeBitmap,
                                    void *memInfoList, uint32_t count);

// ========== 系统周期频率 ==========
constexpr uint32_t SYS_CYCLE_FREQ_MHZ = 1000U;
constexpr double NS_PER_CYCLE = 1.0; // 1000MHz → 1 cycle = 1 ns

// ========== 默认参数 ==========
constexpr int64_t BLOCK_NUM = 1;
constexpr int64_t HCCL_BUFFER_SIZE = 200;
constexpr uint64_t MAX_DATA_SIZE = 120 * 1024; // 120KB，WriteThenRead 模式单次搬运上限
int32_t g_rankId = 0;
int32_t g_ranksize = 1;        // 卡数（本地测试默认单卡）
uint32_t g_mode0 = 0;          // 0=写, 1=读, 2=读写混合, 3=先写后读
uint32_t g_mode1 = 1;          // 0=不交织, 1=交织
uint64_t g_dataSize = 4096;    // 默认 4KB

static const struct option kLongOptions[] = {{"rank_id", 1, 0, 'a'},
                                             {"ranksize", 1, 0, 'd'},
                                             {"mode0", 1, 0, 'm'},
                                             {"mode1", 1, 0, 'i'},
                                             {"data_size", 1, 0, 's'},
                                             {0, 0, 0, 0}};

void GetOption(int argc, char **argv)
{
    int optionIndex = 0, c = 1;
    while (c != -1) {
        c = getopt_long(argc, argv, "a:d:m:i:s:", kLongOptions, &optionIndex);
        switch (c) {
            case 'a':
                g_rankId = atoi(optarg);
                break;
            case 'd':
                g_ranksize = atoi(optarg);
                break;
            case 'm':
                g_mode0 = (uint32_t)atoi(optarg);
                break;
            case 'i':
                g_mode1 = (uint32_t)atoi(optarg);
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
        default: return "UNKNOWN";
    }
}

const char *GetMode1Str(uint32_t mode1)
{
    return mode1 == 0 ? "SEQUENTIAL" : "INTERLEAVED";
}

int main(int argc, char *argv[])
{
    GetOption(argc, argv);

    LOG_PRINT("[INFO] ===== 本地读写 MS 测试 =====\n");
    LOG_PRINT("[INFO] rank_id=%d, ranksize=%d, mode0=%s, mode1=%s, data_size=%lu bytes (%.2f KB)\n", g_rankId,
              g_ranksize, GetMode0Str(g_mode0), GetMode1Str(g_mode1), g_dataSize, (double)g_dataSize / 1024.0);

    // ========== dataSize 上限校验 ==========
    if (g_mode0 == 3 && g_dataSize > MAX_DATA_SIZE) {
        LOG_PRINT("[ERROR] WriteThenRead 模式下 dataSize=%lu 超过单次搬运上限 %lu bytes，请减小 data_size\n",
                  g_dataSize, MAX_DATA_SIZE);
        return -1;
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

    // ========== 获取 MS 地址 ==========
    LOG_PRINT("[INFO] 获取 TLV Handle...\n");
    void *tlvHandle = nullptr;
    CcuResult ccuRet = CcuGetTlvHandle((uint32_t)g_rankId, &tlvHandle);
    CHECK_RET(ccuRet == CCU_SUCCESS, LOG_PRINT("[ERROR] CcuGetTlvHandle failed. ret: %d\n", ccuRet); return -1);
    CHECK_RET(tlvHandle != nullptr, LOG_PRINT("[ERROR] tlvHandle is nullptr.\n"); return -1);

    uint64_t msBitmap = CCU_MEMTYPE_MS_BLOCK0 | CCU_MEMTYPE_MS_BLOCK1 | CCU_MEMTYPE_MS_BLOCK2 | CCU_MEMTYPE_MS_BLOCK3;
    CcuMemInfo memInfoList[4];
    LOG_PRINT("[INFO] 查询 MS 地址...\n");
    ccuRet = CcuGetMemInfo(tlvHandle, 0, msBitmap, memInfoList, 4);
    CHECK_RET(ccuRet == CCU_SUCCESS, LOG_PRINT("[ERROR] CcuGetMemInfo failed. ret: %d\n", ccuRet); return -1);

    uint64_t msAddrs[4];
    for (uint32_t i = 0; i < 4; i++) {
        msAddrs[i] = memInfoList[i].memVa;
        LOG_PRINT("[INFO] MSAddr%d (CCUA%d): va=0x%016lx, size=%u bytes\n", i, i, msAddrs[i], memInfoList[i].memSize);
    }

    // ========== 分配计时 GM ==========
    void *timeGm = nullptr;
    ret = aclrtMalloc(&timeGm, 24, ACL_MEM_MALLOC_HUGE_FIRST); // 3 x uint64_t
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc timeGm failed. ret: %d\n", ret); return ret);

    // ========== <<<>>> 调用 Kernel ==========
    LOG_PRINT("[INFO] 启动 Kernel (blockDim=%ld)...\n", BLOCK_NUM);
    local_ms_rw_demo(g_mode0, g_mode1, g_dataSize, (uint32_t)BLOCK_NUM, stream,
                     (uint8_t *)msAddrs[0], (uint8_t *)msAddrs[1], (uint8_t *)msAddrs[2], (uint8_t *)msAddrs[3],
                     (uint8_t *)timeGm);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStream failed. ret: %d\n", ret); return ret);
    LOG_PRINT("[INFO] Kernel 执行完成。\n");

    // ========== 读取计时结果 ==========
    uint64_t timeHost[3];
    ret = aclrtMemcpy(timeHost, 24, timeGm, 24, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy timeGm failed. ret: %d\n", ret); return ret);

    uint64_t startCycle = timeHost[0];
    uint64_t endCycle = timeHost[1];
    uint64_t verifyResult = timeHost[2];
    uint64_t elapsedCycles = endCycle - startCycle;
    double elapsedNs = (double)elapsedCycles * NS_PER_CYCLE;
    double elapsedUs = elapsedNs / 1000.0;
    double dataKB = (double)g_dataSize / 1024.0;
    double bandwidthGBs = (double)g_dataSize / (elapsedNs); // bytes / ns = GB/s

    LOG_PRINT("\n");
    LOG_PRINT("===== 性能结果 =====\n");
    LOG_PRINT("  模式:       %s\n", GetMode0Str(g_mode0));
    LOG_PRINT("  交织:       %s\n", GetMode1Str(g_mode1));
    LOG_PRINT("  数据量:     %.2f KB (%lu bytes)\n", dataKB, g_dataSize);
    LOG_PRINT("  startCycle: %lu\n", startCycle);
    LOG_PRINT("  endCycle:   %lu\n", endCycle);
    LOG_PRINT("  耗时:       %.2f ns (%.2f us)\n", elapsedNs, elapsedUs);
    LOG_PRINT("  带宽:       %.4f GB/s\n", bandwidthGBs);
    if (g_mode0 == 3) {
        if (verifyResult == 0) {
            LOG_PRINT("  数据校验:   通过\n");
        } else {
            LOG_PRINT("  数据校验:   失败\n");
        }
    }
    LOG_PRINT("====================\n");

    // ========== 释放资源 ==========
    aclrtFree(timeGm);
    HcclCommDestroy(comms);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(g_rankId);
    aclFinalize();

    return 0;
}