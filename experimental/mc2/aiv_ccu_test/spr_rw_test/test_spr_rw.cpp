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
 * \file test_spr_rw.cpp
 * \brief SPR 读写测试 Host 侧代码
 *
 * 实现 background.md 2.2 的 SPR 映射例子：
 *   1. halResMap 映射所有 AICORE 的 SPR vaddr (含 AIV)
 *   2. 打印获取的 SPR 地址
 *   3. 将 vaddr 表拷贝到 GM
 *   4. <<<>>> 调用 kernel 执行先写后读，验证 SPR 访问能力
 *   5. 拉回结果打印
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

// SPR 映射相关头文件 (background.md 2.2)
#include "driver/ascend_hal_base.h"
#include "driver/hal_pkg/dpa_pkg.h"
#include "driver/hal_pkg/common_pkg.h"

#include "./spr_rw.h"

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

// ========== halResMap 相关 ==========
// background.md 2.2 示例遍历 36 个 res_id
constexpr uint32_t DEFAULT_RES_ID_COUNT = 36;
// background.md 2.3: DATA_MAIN_BASE 寄存器偏移
constexpr uint64_t SPR_DATA_MAIN_BASE_OFFSET = 0x00D0;

// ========== 默认参数 ==========
constexpr int64_t BLOCK_NUM = 1;
constexpr int64_t HCCL_BUFFER_SIZE = 200;
constexpr uint32_t RESULT_WORDS = 5;
int32_t g_rankId = 0;
int32_t g_ranksize = 1;
// 默认目标核: res_id=18 (background.md 2.3: DIE0 AIV0, coreid: 18)
uint32_t g_targetCoreId = 18;
// 映射的核总数
uint32_t g_resIdCount = DEFAULT_RES_ID_COUNT;

static const struct option kLongOptions[] = {{"rank_id", 1, 0, 'a'},
                                             {"ranksize", 1, 0, 'd'},
                                             {"target_core_id", 1, 0, 't'},
                                             {"res_id_count", 1, 0, 'n'},
                                             {0, 0, 0, 0}};

void GetOption(int argc, char **argv)
{
    int optionIndex = 0, c = 1;
    while (c != -1) {
        c = getopt_long(argc, argv, "a:d:t:n:", kLongOptions, &optionIndex);
        switch (c) {
            case 'a':
                g_rankId = atoi(optarg);
                break;
            case 'd':
                g_ranksize = atoi(optarg);
                break;
            case 't':
                g_targetCoreId = static_cast<uint32_t>(atoi(optarg));
                break;
            case 'n':
                g_resIdCount = static_cast<uint32_t>(atoi(optarg));
                break;
            default:
                break;
        }
    }
}

// ========== halResMap 映射所有核 SPR vaddr ==========
// 对齐 background.md 2.2 的 main() 中的映射逻辑
static int MapAllSprVaddrs(int32_t deviceId, std::vector<uint64_t> &sprVaddrs, uint32_t count)
{
    LOG_PRINT("[INFO] MapAllSprVaddrs begin: deviceId=%d, count=%u\n", deviceId, count);
    sprVaddrs.assign(count, 0);
    struct res_map_info res_info = {};
    res_info.target_proc_type = PROCESS_CP1;
    res_info.res_type = RES_AICORE;
    res_info.flag = 0;
    LOG_PRINT("[INFO] res_map_info: target_proc_type=PROCESS_CP1(%d), res_type=RES_AICORE(%d), flag=0\n",
              (int)PROCESS_CP1, (int)RES_AICORE);

    unsigned int len = 0;
    int ret = 0;
    uint32_t localOk = 0;
    uint32_t localFail = 0;
    for (uint32_t i = 0; i < count; i++) {
        res_info.res_id = i;
        unsigned long va = 0;
        len = 0;
        ret = halResMap(static_cast<unsigned int>(deviceId), &res_info, &va, &len);
        if (ret != 0) {
            // background.md 2.2: 部分核映射可能失败，跳过即可
            LOG_PRINT("[WARN] halResMap failed for res_id=%u, ret=%d, va=0x%lx, len=%u (skip)\n",
                      i, ret, va, len);
            localFail++;
            continue;
        }
        sprVaddrs[i] = static_cast<uint64_t>(va);
        LOG_PRINT("[INFO] halResMap ok: res_id=%2u -> va=0x%016lx, len=%u\n", i, sprVaddrs[i], len);
        localOk++;
    }
    LOG_PRINT("[INFO] MapAllSprVaddrs end: ok=%u, fail=%u\n", localOk, localFail);
    return 0;
}

int main(int argc, char *argv[])
{
    GetOption(argc, argv);

    LOG_PRINT("[INFO] ===== SPR 读写测试 =====\n");
    LOG_PRINT("[INFO] rank_id=%d, ranksize=%d, target_core_id=%u, res_id_count=%u\n",
              g_rankId, g_ranksize, g_targetCoreId, g_resIdCount);

    CHECK_RET(g_targetCoreId < g_resIdCount,
              LOG_PRINT("[ERROR] target_core_id %u >= res_id_count %u\n", g_targetCoreId, g_resIdCount);
              return -1);

    // ========== acl 初始化 ==========
    LOG_PRINT("[INFO] [step] aclInit ...\n");
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret: %d\n", ret); return ret);
    LOG_PRINT("[INFO] [step] aclInit ok\n");

    LOG_PRINT("[INFO] [step] aclrtSetDevice(%d) ...\n", g_rankId);
    ret = aclrtSetDevice(g_rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret: %d\n", ret); return ret);
    LOG_PRINT("[INFO] [step] aclrtSetDevice ok\n");

    aclrtContext context;
    LOG_PRINT("[INFO] [step] aclrtCreateContext ...\n");
    ret = aclrtCreateContext(&context, g_rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret: %d\n", ret); return ret);
    LOG_PRINT("[INFO] [step] aclrtCreateContext ok, context=%p\n", (void *)context);

    aclrtStream stream;
    LOG_PRINT("[INFO] [step] aclrtCreateStream ...\n");
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret: %d\n", ret); return ret);
    LOG_PRINT("[INFO] [step] aclrtCreateStream ok, stream=%p\n", (void *)stream);

    // ========== 初始化通信域 ==========
    LOG_PRINT("[INFO] [step] HcclCommConfigInit ...\n");
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclDeterministic = 1;
    config.hcclBufferSize = HCCL_BUFFER_SIZE;
    ret = strcpy_s(config.hcclCommName, COMM_NAME_MAX_LENGTH - 1, "hccl_comm_spr_test");
    CHECK_RET(ret == EOK, LOG_PRINT("[ERROR] hcclCommName strcpy failed.\n"); return -1);
    LOG_PRINT("[INFO] [step] HcclCommConfig filled: deterministic=%u, bufferSize=%u, commName=%s\n",
              config.hcclDeterministic, config.hcclBufferSize, config.hcclCommName);

    const char *rankTableFile = getenv("RANK_TABLE_FILE");
    CHECK_RET(rankTableFile != nullptr, LOG_PRINT("[ERROR] get RANK_TABLE_FILE env failed.\n"); return -1);
    LOG_PRINT("[INFO] [step] RANK_TABLE_FILE=%s\n", rankTableFile);

    HcclComm comms;
    LOG_PRINT("[INFO] [step] HcclCommInitClusterInfoConfig(rank=%d) BEGIN ...\n", g_rankId);
    ret = HcclCommInitClusterInfoConfig(rankTableFile, g_rankId, &config, &comms);
    LOG_PRINT("[INFO] [step] HcclCommInitClusterInfoConfig END, ret=%d, comms=%p\n", ret, (void *)comms);
    CHECK_RET(ret == HCCL_SUCCESS,
              LOG_PRINT("[ERROR] HcclCommInitClusterInfoConfig failed. ret: %d\n", ret); return ret);
    LOG_PRINT("[INFO] 通信域初始化成功。\n");
    ret = aclrtSetCurrentContext(context);
    LOG_PRINT("[INFO] [step] aclrtSetCurrentContext ret=%d\n", ret);

    // ========== halResMap 映射所有核 SPR vaddr ==========
    LOG_PRINT("\n[INFO] ===== halResMap 映射 SPR vaddr =====\n");
    std::vector<uint64_t> sprVaddrs;
    ret = MapAllSprVaddrs(g_rankId, sprVaddrs, g_resIdCount);
    CHECK_RET(ret == 0, LOG_PRINT("[ERROR] MapAllSprVaddrs failed.\n"); return ret);

    // 汇总打印所有 res_id 的映射结果
    LOG_PRINT("\n[INFO] ===== SPR vaddr 映射汇总 (共 %u 个) =====\n", g_resIdCount);
    uint32_t okCnt = 0;
    uint32_t failCnt = 0;
    for (uint32_t i = 0; i < g_resIdCount; i++) {
        if (sprVaddrs[i] != 0) {
            LOG_PRINT("  res_id=%2u  va=0x%016lx  +0x00D0=0x%016lx  [OK]\n",
                      i, sprVaddrs[i], sprVaddrs[i] + SPR_DATA_MAIN_BASE_OFFSET);
            okCnt++;
        } else {
            LOG_PRINT("  res_id=%2u  va=0x0000000000000000  [FAIL]\n", i);
            failCnt++;
        }
    }
    LOG_PRINT("[INFO] 映射成功 %u 个, 失败 %u 个\n", okCnt, failCnt);

    // 打印目标核的 SPR 地址 (含 +0x00D0 偏移)
    uint64_t targetSprVaddr = 0;
    if (sprVaddrs[g_targetCoreId] == 0) {
        LOG_PRINT("[ERROR] target core %u spr vaddr map failed\n", g_targetCoreId);
        return -1;
    }
    targetSprVaddr = sprVaddrs[g_targetCoreId] + SPR_DATA_MAIN_BASE_OFFSET;
    LOG_PRINT("\n[INFO] 目标核 res_id=%u:\n", g_targetCoreId);
    LOG_PRINT("[INFO]   SPR base vaddr  = 0x%016lx\n", sprVaddrs[g_targetCoreId]);
    LOG_PRINT("[INFO]   SPR +0x00D0     = 0x%016lx (DATA_MAIN_BASE 访问地址)\n", targetSprVaddr);

    // ========== 将 vaddr 表拷贝到 GM (对齐 background.md 2.2) ==========
    void *sprVaddrTableGm = nullptr;
    uint64_t tableSize = static_cast<uint64_t>(g_resIdCount) * sizeof(uint64_t);
    LOG_PRINT("[INFO] [step] aclrtMalloc sprVaddrTableGm, size=%lu bytes ...\n", tableSize);
    ret = aclrtMalloc(&sprVaddrTableGm, tableSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc sprVaddrTableGm failed. ret: %d\n", ret);
              return ret);
    LOG_PRINT("[INFO] [step] aclrtMalloc ok, sprVaddrTableGm=%p\n", sprVaddrTableGm);

    LOG_PRINT("[INFO] [step] aclrtMemcpy H2D sprVaddrTableGm, %lu bytes ...\n", tableSize);
    ret = aclrtMemcpy(sprVaddrTableGm, tableSize, sprVaddrs.data(), tableSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy sprVaddrTableGm failed. ret: %d\n", ret);
              return ret);
    LOG_PRINT("[INFO] [step] aclrtMemcpy H2D ok\n");

    // ========== 分配结果 GM ==========
    void *resultGm = nullptr;
    uint64_t resultSize = RESULT_WORDS * sizeof(uint64_t);
    LOG_PRINT("[INFO] [step] aclrtMalloc resultGm, size=%lu bytes ...\n", resultSize);
    ret = aclrtMalloc(&resultGm, resultSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc resultGm failed. ret: %d\n", ret); return ret);
    LOG_PRINT("[INFO] [step] aclrtMalloc ok, resultGm=%p\n", resultGm);

    // ========== <<<>>> 调用 Kernel ==========
    LOG_PRINT("\n[INFO] ===== 启动 Kernel (blockDim=%ld, targetCoreId=%u) =====\n", BLOCK_NUM,
              g_targetCoreId);
    LOG_PRINT("[INFO]   targetSprVaddr   = 0x%016lx\n", targetSprVaddr);
    LOG_PRINT("[INFO]   sprVaddrTableGm  = %p\n", sprVaddrTableGm);
    LOG_PRINT("[INFO]   resultGm         = %p\n", resultGm);
    LOG_PRINT("[INFO]   stream           = %p\n", (void *)stream);
    spr_rw_demo(targetSprVaddr, static_cast<uint32_t>(BLOCK_NUM), stream,
                reinterpret_cast<uint8_t *>(sprVaddrTableGm), reinterpret_cast<uint8_t *>(resultGm),
                g_targetCoreId);
    LOG_PRINT("[INFO] [step] spr_rw_demo launched, waiting sync ...\n");

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStream failed. ret: %d\n", ret);
              return ret);
    LOG_PRINT("[INFO] Kernel 执行完成。\n");

    // ========== 读取结果 ==========
    uint64_t resultHost[RESULT_WORDS] = {0};
    LOG_PRINT("[INFO] [step] aclrtMemcpy D2H resultGm, %lu bytes ...\n", sizeof(resultHost));
    ret = aclrtMemcpy(resultHost, sizeof(resultHost), resultGm, sizeof(resultHost),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy resultGm failed. ret: %d\n", ret);
              return ret);
    LOG_PRINT("[INFO] [step] aclrtMemcpy D2H ok\n");

    LOG_PRINT("\n===== SPR 访问结果 =====\n");
    LOG_PRINT("  目标核 res_id:      %u\n", g_targetCoreId);
    LOG_PRINT("  sprVaddr (回显):    0x%016lx\n", resultHost[0]);
    LOG_PRINT("  写前读值:           0x%016lx\n", resultHost[1]);
    LOG_PRINT("  写后读值:           0x%016lx\n", resultHost[2]);
    LOG_PRINT("  verifyResult:       %lu (0=通过/有变化, 1=失败, 2=skip)\n", resultHost[3]);
    LOG_PRINT("  aivCoreId:          %lu\n", resultHost[4]);
    LOG_PRINT("========================\n");

    if (resultHost[3] == 0) {
        LOG_PRINT("[INFO] SPR 访问验证: 通过 (写后读值与写前不同，SPR 寄存器可写可读)\n");
    } else if (resultHost[3] == 2) {
        LOG_PRINT("[INFO] SPR 访问验证: SKIP (他核访问场景，需目标核配合读)\n");
    } else {
        LOG_PRINT("[WARN] SPR 访问验证: 失败或无变化 (写后读值 == 写前读值)\n");
    }

    // ========== 释放资源 ==========
    LOG_PRINT("[INFO] [step] release resources ...\n");
    LOG_PRINT("[INFO] [step] aclrtFree(resultGm=%p) ...\n", resultGm);
    aclrtFree(resultGm);
    LOG_PRINT("[INFO] [step] aclrtFree(sprVaddrTableGm=%p) ...\n", sprVaddrTableGm);
    aclrtFree(sprVaddrTableGm);
    LOG_PRINT("[INFO] [step] HcclCommDestroy(comms=%p) ...\n", (void *)comms);
    HcclCommDestroy(comms);
    LOG_PRINT("[INFO] [step] aclrtDestroyStream(stream=%p) ...\n", (void *)stream);
    aclrtDestroyStream(stream);
    LOG_PRINT("[INFO] [step] aclrtDestroyContext(context=%p) ...\n", (void *)context);
    aclrtDestroyContext(context);
    LOG_PRINT("[INFO] [step] aclrtResetDevice(%d) ...\n", g_rankId);
    aclrtResetDevice(g_rankId);
    LOG_PRINT("[INFO] [step] aclFinalize ...\n");
    aclFinalize();
    LOG_PRINT("[INFO] [step] all done, exit 0\n");

    return 0;
}
