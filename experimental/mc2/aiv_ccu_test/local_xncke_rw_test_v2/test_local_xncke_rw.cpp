/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_local_xncke_rw.cpp
 * \brief 本地读写 CCU Variable(XN)/Event(CKE) 测试 Host 侧代码
 *        使用 hcomm 的 HcommCcuInsCreateDefault / HcommCcuVariableAlloc / HcommCcuEventAlloc /
 *        HcommCcuVariableAddrGet / HcommCcuEventAddrGet / HcommCcuInsDestroy 接口获取资源地址，
 *        然后通过 <<<>>> 直调 AIV kernel 做读写验证。
 */

#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <iostream>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hcom.h"
#include "ccu/ccu_res.h"
#include "securec.h"
#include "./local_xncke_rw.h"

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

#define CHECK_CCU(ccuRet, msg, ...)                                                                                    \
    do {                                                                                                               \
        if ((ccuRet) != CCU_SUCCESS) {                                                                                 \
            LOG_PRINT("[ERROR] " msg " ret: %d\n", ##__VA_ARGS__, static_cast<int32_t>(ccuRet));                       \
            return -1;                                                                                                 \
        }                                                                                                              \
    } while (0)

enum TestResKind : uint32_t {
    TEST_RES_XN = 0,
    TEST_RES_CKE = 1,
    TEST_RES_BOTH = 2,
};

enum TestMode : uint32_t {
    TEST_MODE_WRITE = 0,
    TEST_MODE_READ = 1,
    TEST_MODE_WRITE_THEN_READ = 2,
    TEST_MODE_SET_VALUE = 3,
};

constexpr uint32_t RESULT_WORDS = 6;
constexpr uint32_t BLOCK_NUM = 1;
constexpr int64_t HCCL_BUFFER_SIZE = 200;
constexpr double NS_PER_CYCLE = 1.0;

int32_t g_rankId = 0;
int32_t g_ranksize = 1;
uint32_t g_mode = TEST_MODE_WRITE_THEN_READ;
uint32_t g_resKind = TEST_RES_BOTH;
uint32_t g_enableWarmup = 0;
uint32_t g_dieId = 0;
uint32_t g_xnNum = 1;
uint32_t g_xnIndex = 0;
uint32_t g_ckeNum = 1;
uint32_t g_ckeIndex = 0;
uint32_t g_loopCount = 100;
uint64_t g_pattern = 0x5A5A000000000001ULL;

static const struct option kLongOptions[] = {{"rank_id", 1, 0, 'a'},
                                             {"ranksize", 1, 0, 'd'},
                                             {"mode", 1, 0, 'm'},
                                             {"res_kind", 1, 0, 'k'},
                                             {"warmup", 1, 0, 'w'},
                                             {"die_id", 1, 0, 'i'},
                                             {"xn_num", 1, 0, 'x'},
                                             {"xn_index", 1, 0, 'X'},
                                             {"cke_num", 1, 0, 'e'},
                                             {"cke_index", 1, 0, 'E'},
                                             {"loop_count", 1, 0, 'l'},
                                             {"pattern", 1, 0, 'p'},
                                             {0, 0, 0, 0}};

void GetOption(int argc, char **argv)
{
    int optionIndex = 0;
    int c = 1;
    while (c != -1) {
        c = getopt_long(argc, argv, "a:d:m:k:w:i:x:X:e:E:l:p:", kLongOptions, &optionIndex);
        switch (c) {
            case 'a':
                g_rankId = atoi(optarg);
                break;
            case 'd':
                g_ranksize = atoi(optarg);
                break;
            case 'm':
                g_mode = static_cast<uint32_t>(atoi(optarg));
                break;
            case 'k':
                g_resKind = static_cast<uint32_t>(atoi(optarg));
                break;
            case 'w':
                g_enableWarmup = static_cast<uint32_t>(atoi(optarg));
                break;
            case 'i':
                g_dieId = static_cast<uint32_t>(atoi(optarg));
                break;
            case 'x':
                g_xnNum = static_cast<uint32_t>(atoi(optarg));
                break;
            case 'X':
                g_xnIndex = static_cast<uint32_t>(atoi(optarg));
                break;
            case 'e':
                g_ckeNum = static_cast<uint32_t>(atoi(optarg));
                break;
            case 'E':
                g_ckeIndex = static_cast<uint32_t>(atoi(optarg));
                break;
            case 'l':
                g_loopCount = static_cast<uint32_t>(atoi(optarg));
                break;
            case 'p':
                g_pattern = strtoull(optarg, nullptr, 0);
                break;
            default:
                break;
        }
    }
}

const char *GetModeStr(uint32_t mode)
{
    switch (mode) {
        case TEST_MODE_WRITE:
            return "WRITE";
        case TEST_MODE_READ:
            return "READ";
        case TEST_MODE_WRITE_THEN_READ:
            return "WRITE_THEN_READ";
        case TEST_MODE_SET_VALUE:
            return "SET_VALUE";
        default:
            return "UNKNOWN";
    }
}

const char *GetResKindStr(uint32_t resKind)
{
    switch (resKind) {
        case TEST_RES_XN:
            return "XN";
        case TEST_RES_CKE:
            return "CKE";
        case TEST_RES_BOTH:
            return "BOTH";
        default:
            return "UNKNOWN";
    }
}

bool NeedAccessXn()
{
    return g_resKind == TEST_RES_XN || g_resKind == TEST_RES_BOTH;
}

bool NeedAccessCke()
{
    return g_resKind == TEST_RES_CKE || g_resKind == TEST_RES_BOTH;
}

int main(int argc, char *argv[])
{
    GetOption(argc, argv);

    LOG_PRINT("[INFO] ===== 本地读写 XN(64bit)/CKE(16bit) 测试 =====\n");
    LOG_PRINT("[INFO] rank_id=%d, ranksize=%d, mode=%s, res_kind=%s, warmup=%u, die_id=%u, "
              "xn_num=%u, xn_index=%u, cke_num=%u, cke_index=%u, loop_count=%u, pattern=0x%016lx\n",
              g_rankId, g_ranksize, GetModeStr(g_mode), GetResKindStr(g_resKind), g_enableWarmup, g_dieId,
              g_xnNum, g_xnIndex, g_ckeNum, g_ckeIndex, g_loopCount, g_pattern);

    CHECK_RET(g_mode <= TEST_MODE_SET_VALUE, LOG_PRINT("[ERROR] mode invalid: %u\n", g_mode); return -1);
    CHECK_RET(g_resKind <= TEST_RES_BOTH, LOG_PRINT("[ERROR] res_kind invalid: %u\n", g_resKind); return -1);
    CHECK_RET(g_enableWarmup <= 1, LOG_PRINT("[ERROR] warmup invalid: %u\n", g_enableWarmup); return -1);
    CHECK_RET(g_loopCount > 0, LOG_PRINT("[ERROR] loop_count must be greater than 0.\n"); return -1);
    CHECK_RET(g_dieId < 2, LOG_PRINT("[ERROR] die_id must be 0 or 1.\n"); return -1);

    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret: %d\n", ret); return ret);

    ret = aclrtSetDevice(g_rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret: %d\n", ret); return ret);

    aclrtContext context;
    ret = aclrtCreateContext(&context, g_rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret: %d\n", ret); return ret);

    aclrtStream stream;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret: %d\n", ret); return ret);

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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);

    LOG_PRINT("[INFO] 创建 CCU 实例 (HcommCcuInsCreateDefault)...\n");
    CcuInsHandle ccuInsHandle = 0;
    CcuResult ccuRet = HcommCcuInsCreateDefault(nullptr, 0, &ccuInsHandle);
    CHECK_CCU(ccuRet, "HcommCcuInsCreateDefault failed.");
    CHECK_RET(ccuInsHandle != 0, LOG_PRINT("[ERROR] ccuInsHandle is 0.\n"); return -1);
    LOG_PRINT("[INFO] ccuInsHandle=0x%016lx\n", static_cast<uint64_t>(ccuInsHandle));

    uint64_t xnAddr = 0;
    uint64_t ckeAddr = 0;
    CcuVariableHandle varHandle = 0;
    CcuEventHandle eventHandle = 0;

    if (NeedAccessXn()) {
        LOG_PRINT("[INFO] 申请 Variable(XN) 资源: dieId=%u, num=%u\n", g_dieId, g_xnNum);
        ccuRet = HcommCcuVariableAlloc(ccuInsHandle, static_cast<uint8_t>(g_dieId), g_xnNum, &varHandle);
        CHECK_CCU(ccuRet, "HcommCcuVariableAlloc failed.");
        CHECK_RET(varHandle != 0, LOG_PRINT("[ERROR] varHandle is 0.\n"); return -1);
        LOG_PRINT("[INFO] varHandle=0x%016lx\n", static_cast<uint64_t>(varHandle));

        ccuRet = HcommCcuVariableAddrGet(varHandle, g_xnIndex, &xnAddr);
        CHECK_CCU(ccuRet, "HcommCcuVariableAddrGet failed. index=%u", g_xnIndex);
        CHECK_RET(xnAddr != 0, LOG_PRINT("[ERROR] xnAddr is 0.\n"); return -1);
        LOG_PRINT("[INFO] XNAddr:  varHandle=0x%016lx, index=%u, va=0x%016lx\n",
                  static_cast<uint64_t>(varHandle), g_xnIndex, xnAddr);
    }

    if (NeedAccessCke()) {
        LOG_PRINT("[INFO] 申请 Event(CKE) 资源: dieId=%u, num=%u\n", g_dieId, g_ckeNum);
        ccuRet = HcommCcuEventAlloc(ccuInsHandle, static_cast<uint8_t>(g_dieId), g_ckeNum, &eventHandle);
        CHECK_CCU(ccuRet, "HcommCcuEventAlloc failed.");
        CHECK_RET(eventHandle != 0, LOG_PRINT("[ERROR] eventHandle is 0.\n"); return -1);
        LOG_PRINT("[INFO] eventHandle=0x%016lx\n", static_cast<uint64_t>(eventHandle));

        ccuRet = HcommCcuEventAddrGet(eventHandle, g_ckeIndex, &ckeAddr);
        CHECK_CCU(ccuRet, "HcommCcuEventAddrGet failed. index=%u", g_ckeIndex);
        CHECK_RET(ckeAddr != 0, LOG_PRINT("[ERROR] ckeAddr is 0.\n"); return -1);
        LOG_PRINT("[INFO] CKEAddr: eventHandle=0x%016lx, index=%u, va=0x%016lx\n",
                  static_cast<uint64_t>(eventHandle), g_ckeIndex, ckeAddr);
    }

    void *timeGm = nullptr;
    ret = aclrtMalloc(&timeGm, RESULT_WORDS * sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc timeGm failed. ret: %d\n", ret); return ret);

    LOG_PRINT("[INFO] 启动 Kernel (blockDim=%u)...\n", BLOCK_NUM);
    local_xncke_rw_demo(g_mode, g_resKind, g_enableWarmup, g_loopCount, g_pattern, BLOCK_NUM, stream,
                        (uint8_t *)xnAddr, (uint8_t *)ckeAddr, (uint8_t *)timeGm);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStream failed. ret: %d\n", ret); return ret);
    LOG_PRINT("[INFO] Kernel 执行完成。\n");

    uint64_t timeHost[RESULT_WORDS] = {0};
    ret = aclrtMemcpy(timeHost, sizeof(timeHost), timeGm, sizeof(timeHost), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy timeGm failed. ret: %d\n", ret); return ret);

    uint64_t startCycle = timeHost[0];
    uint64_t endCycle = timeHost[1];
    uint64_t verifyResult = timeHost[2];
    uint64_t xnReadBack = timeHost[3];
    uint64_t ckeReadBack = timeHost[4];
    uint64_t warmupCycles = timeHost[5];
    uint64_t elapsedCycles = endCycle - startCycle;
    double elapsedNs = static_cast<double>(elapsedCycles) * NS_PER_CYCLE;
    double avgNs = elapsedNs / static_cast<double>(g_loopCount);

    LOG_PRINT("\n");
    LOG_PRINT("===== XN/CKE 测试结果 =====\n");
    LOG_PRINT("  资源:         %s\n", GetResKindStr(g_resKind));
    LOG_PRINT("  模式:         %s\n", GetModeStr(g_mode));
    LOG_PRINT("  预热:         %s\n", g_enableWarmup == 0 ? "关闭" : "开启");
    LOG_PRINT("  loop_count:   %u\n", g_loopCount);
    LOG_PRINT("  warmupCycle:  %lu\n", warmupCycles);
    LOG_PRINT("  startCycle:   %lu\n", startCycle);
    LOG_PRINT("  endCycle:     %lu\n", endCycle);
    LOG_PRINT("  耗时:         %.2f ns\n", elapsedNs);
    LOG_PRINT("  平均:         %.4f ns/loop\n", avgNs);
    if (g_mode != TEST_MODE_SET_VALUE && NeedAccessXn()) {
        LOG_PRINT("  XN readback:  0x%016lx (expected: 0x%016lx)\n", xnReadBack, g_pattern);
    }
    if (g_mode != TEST_MODE_SET_VALUE && NeedAccessCke()) {
        LOG_PRINT("  CKE readback: 0x%04x     (expected: 0x%04x)\n",
                  static_cast<uint32_t>(ckeReadBack & 0xFFFFU), static_cast<uint32_t>(g_pattern & 0xFFFFU));
    }
    if (g_mode == TEST_MODE_WRITE_THEN_READ) {
        LOG_PRINT("  数据校验:     %s\n", verifyResult == 0 ? "通过" : "失败");
    }
    LOG_PRINT("===========================\n");

    aclrtFree(timeGm);
    HcommCcuInsDestroy(ccuInsHandle);
    HcclCommDestroy(comms);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(g_rankId);
    aclFinalize();

    return 0;
}
