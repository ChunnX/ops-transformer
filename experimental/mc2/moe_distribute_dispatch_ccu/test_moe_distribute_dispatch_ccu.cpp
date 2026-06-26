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
 * \file test_moe_distribute_dispatch_ccu.cpp
 * \brief dispatch-ccu <<<>>> 直调测试驱动（host）。
 *        P0 骨架：ACL / HCCL 初始化 + 调用 MoeDistributeDispatchCcu；
 *        输入数据填充与 golden 精度对比为 P0 步骤 6 待补（见 TODO）。
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "moe_distribute_dispatch_ccu_host.h"

#define CHECK_RET(cond, expr) do { if (!(cond)) { expr; } } while (0)
#define LOG_PRINT(fmt, ...)   do { printf(fmt, ##__VA_ARGS__); } while (0)

// 外部输入默认值
static int g_rankId     = 0;
static int g_rankSize   = 8;       // EP 域卡数
static int g_bs         = 16;      // 本卡 token 数
static int g_hidden     = 4096;    // 隐藏维
static int g_topk       = 8;       // 每 token 路由 expert 数
static int g_expertNum  = 64;      // expert 总数
static int64_t g_hcclBufferSize = 200;   // HCCL Buffer 大小（MB，用户可自定义）
static const int g_streamTimeout = 10000;

static const struct option kLongOptions[] = {
    {"rank_id",    1, 0, 'a'},
    {"rank_size",  1, 0, 'b'},
    {"bs",         1, 0, 'c'},
    {"hidden",     1, 0, 'd'},
    {"topk",       1, 0, 'e'},
    {"expert_num", 1, 0, 'f'},
    {0, 0, 0, 0}
};

static void GetOption(int argc, char** argv)
{
    int idx = 0;
    int c = 1;
    while (c != -1) {
        c = getopt_long(argc, argv, "a:b:c:d:e:f:", kLongOptions, &idx);
        switch (c) {
            case 'a': g_rankId = atoi(optarg); break;
            case 'b': g_rankSize = atoi(optarg); break;
            case 'c': g_bs = atoi(optarg); break;
            case 'd': g_hidden = atoi(optarg); break;
            case 'e': g_topk = atoi(optarg); break;
            case 'f': g_expertNum = atoi(optarg); break;
            default: break;
        }
    }
}

int main(int argc, char** argv)
{
    GetOption(argc, argv);

    // ── ACL 初始化 ──
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret=%d\n", ret); return ret);
    ret = aclrtSetDevice(g_rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret=%d\n", ret); return ret);
    aclrtContext context = nullptr;
    ret = aclrtCreateContext(&context, g_rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret=%d\n", ret); return ret);
    aclrtStream stream1 = nullptr;  // AICore 主算路 stream
    ret = aclrtCreateStream(&stream1);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret=%d\n", ret); return ret);

    // ── HCCL 通信域初始化（rank table 由环境变量 RANK_TABLE_FILE 指定）──
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclBufferSize = g_hcclBufferSize;  // 用户自定义 HCCL Buffer 大小
    (void)strcpy_s(config.hcclCommName, COMM_NAME_MAX_LENGTH - 1, "dispatch_ccu_comm");
    const char* rankTableFile = getenv("RANK_TABLE_FILE");
    CHECK_RET(rankTableFile != nullptr, LOG_PRINT("[ERROR] env RANK_TABLE_FILE not set\n"); return -1);
    HcclComm comm = nullptr;
    ret = HcclCommInitClusterInfoConfig(rankTableFile, g_rankId, &config, &comm);
    CHECK_RET(ret == HCCL_SUCCESS,
              LOG_PRINT("[ERROR] HcclCommInitClusterInfoConfig failed. ret=%d\n", ret); return ret);

    // ── 分配 device 输入 / 输出 ──
    // x: [bs, hidden] bf16；expertIds: [bs, topk] int32；expandXOut: 保守上界 [rankSize*bs*topk, hidden] bf16
    const size_t xBytes        = static_cast<size_t>(g_bs) * g_hidden * sizeof(uint16_t);
    const size_t expertIdBytes = static_cast<size_t>(g_bs) * g_topk * sizeof(int32_t);
    const size_t expandXBytes  =
        static_cast<size_t>(g_rankSize) * g_bs * g_topk * g_hidden * sizeof(uint16_t);
    void* xDev = nullptr;
    void* expertIdsDev = nullptr;
    void* expandXDev = nullptr;
    ret = aclrtMalloc(&xDev, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc x failed. ret=%d\n", ret); return ret);
    ret = aclrtMalloc(&expertIdsDev, expertIdBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc expertIds failed. ret=%d\n", ret); return ret);
    ret = aclrtMalloc(&expandXDev, expandXBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc expandX failed. ret=%d\n", ret); return ret);

    // TODO(P0 步骤 6)：从 golden 目录读入 x / expertIds（参考 scripts/data_generate.py）

    // ── 调用 dispatch-ccu 直调入口 ──
    DispatchCcuParam param;
    param.worldSize = static_cast<uint32_t>(g_rankSize);
    param.myRank    = static_cast<uint32_t>(g_rankId);
    param.blockDim  = 8U;
    param.topk      = static_cast<uint32_t>(g_topk);
    param.expertNum = static_cast<uint32_t>(g_expertNum);
    param.tokenNum  = static_cast<uint64_t>(g_bs);
    param.hidden    = static_cast<uint64_t>(g_hidden);

    ret = MoeDistributeDispatchCcu(comm, xDev, expertIdsDev, expandXDev, param, stream1);
    CHECK_RET(ret == HCCL_SUCCESS,
              LOG_PRINT("[ERROR] MoeDistributeDispatchCcu failed. ret=%d\n", ret); return ret);

    ret = aclrtSynchronizeStreamWithTimeout(stream1, g_streamTimeout);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] SynchronizeStream failed. ret=%d\n", ret); return ret);
    LOG_PRINT("[INFO] rank %d: MoeDistributeDispatchCcu executed.\n", g_rankId);

    // TODO(P0 步骤 6)：拷回 expandXDev → 与 golden 精度对比（fp16 atol/rtol=1e-3，bf16=5e-3）

    // ── 资源回收 ──
    aclrtFree(xDev);
    aclrtFree(expertIdsDev);
    aclrtFree(expandXDev);
    HcclCommDestroy(comm);
    aclrtDestroyStream(stream1);
    aclrtDestroyContext(context);
    aclrtResetDevice(g_rankId);
    aclFinalize();
    return 0;
}
