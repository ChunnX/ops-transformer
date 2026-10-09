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
 * \file main.cpp
 * \brief quantMatmulReduceScatter ST 主程序
 *
 * 流程: fork 多 rank → 每进程绑定 NPU → HCCL 建链 → Tiling → Kernel 执行 → 精度比对
 *
 * 与 all_to_all_quant_matmul 的关键区别：
 *   - K 不切分（每卡持有完整 K），m 按 rankSize 切分
 *   - 通信数据为 C (bfloat16)，非 A (fp8)
 *   - 输出 per rank: [m/rankSize, n]
 */

#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <limits.h>
#include <unistd.h>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <sys/wait.h>

#include "kernel_basic_intf.h"
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_rank_graph.h"
#include <cstdlib>
#include "apace_st_utils.h"
#include "apace/utils/comm_channel_builder.h"
#include "../../utils/root_info_exchanger.h"
#include "apace/tiling/quant_matmul_tiling_swat.h"
#include "apace_qmrs_kernel_launcher.h"
#include "apace/core/aiv_comm/collective_comm_context.h"

#define HCCL_CHECK(status) \
    do { \
        HcclResult ret = (status); \
        if (ret != HCCL_SUCCESS) { \
            std::cerr << __FILE__ << ":" << __LINE__ << " HcclResult:" << ret << std::endl; \
        } \
    } while (0)

inline uint64_t CeilDiv(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

// reduceSum dtype 扩展编译覆盖: fp16/fp32 暂无运行用例, 借未被调用的设备函数锁定编译兼容
// (实例化 ReduceSumForAlltoAll<half, float> / <float, float> 的 ExecuteReduceSum 全路径)
__aicore__ inline void ReduceSumDtypeCompileCheck()
{
    AiVReduceSumImpl::ReduceSumForAlltoAll<half, float> reduceSumHalf;
    AiVReduceSumImpl::ReduceSumForAlltoAll<float, float> reduceSumFloat;
    reduceSumHalf.Init({});
    reduceSumFloat.Init({});
    reduceSumHalf.ExecuteReduceSum(0);
    reduceSumFloat.ExecuteReduceSum(0);
}

// headMSize 命令行缺省值 (显式 <=0 可切换公式化 tileM, 见 FormulaHeadMSize)
constexpr int QMRS_DEFAULT_HEAD_M_SIZE = 512;

void printUsage(const std::string &programName)
{
    std::cerr << "Usage: " << programName << " m k n rankNum [mode] [headMSize] [localScopeLast]" << std::endl;
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m: row of matrix A (must be divisible by rankNum)" << std::endl;
    std::cerr << "  k: col of matrix A (full K, NOT split across ranks)" << std::endl;
    std::cerr << "  n: col of matrix B" << std::endl;
    std::cerr << "  rankNum: number of ranks" << std::endl;
    std::cerr << "  mode: optional, 'precision' (default) | 'perf'" << std::endl;
    std::cerr << "  headMSize: optional, absent = " << QMRS_DEFAULT_HEAD_M_SIZE
              << " (default), <=0 = formula tileM (experimental), positive = override" << std::endl;
    std::cerr << "  localScopeLast: optional, 0 = LOCAL scope first (default) | 1 = LOCAL scope last" << std::endl;
    std::cerr << "Example: " << programName << " 2048 3584 4096 4" << std::endl;
}

// ---- ST host 侧具名常量 (与 kernel 侧同口径) ----
// rankNum 范围: 下界 2 (单 rank 无通信语义); 上界与 kernel 侧 cFragAddrs_[MAX_RANK_SIZE]
// 及 tiling rank_size 上限一致 (同 all_gather_qbmm_mx_kernel.h)
constexpr int QMRS_MIN_RANK_NUM = 2;
constexpr int QMRS_MAX_RANK_NUM = 16;
// MX 量化组大小与 scale 字节数 (与 impl.h MX_GROUP_SIZE/SCALE_C0 同口径)
constexpr int QMRS_MX_GROUP_SIZE = 64;
constexpr int QMRS_MX_SCALE_BYTES = 2;
// Blaze matmul 最小行粒度 (headMSize/tailMSize 16 对齐)
constexpr int QMRS_MIN_ROW_ALIGN = 16;
// barrier 槽位字节数 (与 apace barrier_ubmem.h BARRIER_FLAG_SIZE 同口径)
constexpr uint64_t QMRS_BARRIER_FLAG_BYTES = 32UL;

void parseArguments(int argc, char *argv[], int &m, int &k, int &n, int &rankNum, std::string &mode, int &headMSize,
                    int &localScopeLast)
{
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        printUsage(argv[0]);
        exit(1);
    }
    if (argc < 5) {
        throw std::invalid_argument("ERROR: Lacks Arguments");
    }
    // stoi 超范围抛 std::out_of_range (与 invalid_argument 无继承关系), 两者均须捕获,
    // 否则异常逃逸导致 terminate 而非优雅报错
    try {
        m = std::stoi(argv[1]);
        k = std::stoi(argv[2]);
        n = std::stoi(argv[3]);
        rankNum = std::stoi(argv[4]);
    } catch (const std::invalid_argument &) {
        throw std::invalid_argument("ERROR: m k n rankNum must be Integer");
    } catch (const std::out_of_range &) {
        throw std::invalid_argument("ERROR: m k n rankNum exceeds int range");
    }

    mode = (argc >= 6) ? std::string(argv[5]) : std::string("precision");
    if (mode != "precision" && mode != "perf") {
        throw std::invalid_argument("ERROR: mode must be 'precision' or 'perf'");
    }

    // headMSize: 缺省 512; 显式 <=0 → 公式化 tileM (runQuantMatmulReduceScatter 内 FormulaHeadMSize)
    try {
        headMSize = (argc >= 7) ? std::stoi(argv[6]) : QMRS_DEFAULT_HEAD_M_SIZE;
        localScopeLast = (argc >= 8) ? std::stoi(argv[7]) : 0;
    } catch (const std::invalid_argument &) {
        throw std::invalid_argument("ERROR: headMSize/localScopeLast must be Integer");
    } catch (const std::out_of_range &) {
        throw std::invalid_argument("ERROR: headMSize/localScopeLast exceeds int range");
    }
    if (localScopeLast != 0 && localScopeLast != 1) {
        throw std::invalid_argument("ERROR: localScopeLast must be 0 or 1");
    }

    if (m <= 0 || k <= 0 || n <= 0 || rankNum <= 0) {
        throw std::invalid_argument("ERROR: m k n rankNum must be positive");
    }

    // m 必须被 rankNum 整除（M 轴按 rankSize 切分用于 AlltoAll）
    if (m % rankNum != 0) {
        throw std::invalid_argument("ERROR: m must be divisible by rankNum");
    }

    // rankNum 范围校验 (常量见文件头 QMRS_MIN/MAX_RANK_NUM)
    if (rankNum < QMRS_MIN_RANK_NUM || rankNum > QMRS_MAX_RANK_NUM) {
        throw std::invalid_argument("ERROR: rankNum must be in [2, 16] (kernel MAX_RANK_SIZE bound)");
    }

    // Scale 对齐约束：CeilDiv(k, QMRS_MX_GROUP_SIZE) 必须为偶数 (E8M0 每组 2 字节成对)
    if (CeilDiv(k, QMRS_MX_GROUP_SIZE) % QMRS_MX_SCALE_BYTES != 0) {
        throw std::invalid_argument("ERROR: CeilDiv(k, QMRS_MX_GROUP_SIZE) must be an even number");
    }
}

// 公式化 tileM (实验入口, headMSize 显式 <=0 触发): 每 tile 的 base-tile 网格
// (tileM/256)×(N/256) ≥ blockNum, 保证逐 tile 全核占用。
// 实测结论 (2026-09-28 方向批, 11 case): N ≥ blockNum*256 时 tileM=256 小幅正向/中性;
// N < blockNum*256 时公式抬 tileM 在 N=4096/5120 r4 族显著回退 (row10 +34%/row8 +15.7%,
// 占用率准则与该族 PUT/compute 重叠粒度最优相反), 故不作缺省, cases.csv 实测表为唯一生效配置。
// 约束链: tileM ≤ chunkM (钳制); T = CeilDiv(chunkM, tileM) ≤ QMRS_MAX_COMM_TILES (超限倍增)。
// 256 为固定粒度常数 (非求解器 baseM/baseN, 后者随 shape 变化); blockNum = mmTilingData.usedCoreNum
// (tiling 引擎输出的 AIC 核数, 平台自适应)。
uint32_t FormulaHeadMSize(uint32_t n, uint32_t chunkM, uint32_t blockNum)
{
    const uint32_t nTiles = (n / 256U) > 1U ? (n / 256U) : 1U; // N 方向 256 宽 slab 数 (下限 1 防 N<256 除零)
    const uint32_t blocks = blockNum > 1U ? blockNum : 1U;
    uint32_t tileM = CeilDiv(blocks, nTiles) * 256U;
    if (tileM > chunkM) {
        tileM = chunkM;
    }
    while (CeilDiv(chunkM, tileM) > QMRS_MAX_COMM_TILES) {
        tileM *= 2U;
        if (tileM > chunkM) {
            tileM = chunkM;
        }
    }
    return tileM;
}

int runQuantMatmulReduceScatter(int rankNum, int rankId, int m, int k, int n, const std::string &mode, int headMSizeArg,
                                int localScopeLastArg)
{
    const char *ipport = "tcp://127.0.0.1:8998";
    INFO_LOG("rankNum=%d, rankId=%d, ipport=%s, mode=%s, headMSize=%d (<=0: formula), localScopeLast=%d", rankNum,
             rankId, ipport, mode.c_str(), headMSizeArg, localScopeLastArg);

    // 与 all_to_all_quant_matmul 不同：K 不切分，每卡持有完整 K
    // m 按 rankSize 切分用于 AlltoAll 通信
    uint32_t chunkM = static_cast<uint32_t>(m) / static_cast<uint32_t>(rankNum);

    QuantMatmulReduceScatterTilingData tilingData;

    // MatMul tiling: 使用完整 m, n, k（无 TP 切分）
    QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN, mm::BiasDataType::DT_FLOAT>
        tilingEngine;
    tilingEngine.GetTilingData(m, n, k, false, true, tilingData.mmTilingData);

    // scope 执行顺序开关: 0 = LOCAL 先 (默认), 1 = LOCAL 后置
    // (单 flag 语义下二者时序等价 — flag = 双 scope 全部完成; 策略开关, 供实验与调优)
    tilingData.localScopeLast = static_cast<uint8_t>(localScopeLastArg);

    // 通信 tiling: PUT 模式多 tile
    // 切分轴 = M, 非切分轴 = N
    // chunkM = m / rankSize, 按 headMSize 切分为 headTileCnt 个头块 + 可选 1 个尾块
    // PUT 模式: AIC 写 workspace, AIV 从 workspace 推送到其他 rank win 区
    // win 区布局 = C_i 连续 [m,n] = rank-major chunks (与 workspace 一致)
    // headMSize 解析: <=0 (含缺省) → 公式化 tileM; 显式正数 → 逐 case 覆盖 (钳制到 chunkM)
    uint32_t headMSize;
    if (headMSizeArg <= 0) {
        headMSize = FormulaHeadMSize(static_cast<uint32_t>(n), chunkM, tilingData.mmTilingData.usedCoreNum);
        if (rankId == 0) {
            INFO_LOG("FormulaHeadMSize: n=%u chunkM=%u blockNum=%u -> tileM=%u", static_cast<uint32_t>(n), chunkM,
                     tilingData.mmTilingData.usedCoreNum, headMSize);
        }
    } else {
        headMSize = static_cast<uint32_t>(headMSizeArg);
        if (headMSize > chunkM) {
            headMSize = chunkM;
        }
    }
    headMSize = (headMSize / QMRS_MIN_ROW_ALIGN) * QMRS_MIN_ROW_ALIGN;
    if (headMSize == 0) {
        headMSize = QMRS_MIN_ROW_ALIGN;
    }

    uint32_t headTileCnt = chunkM / headMSize;
    uint32_t tailMSize = chunkM % headMSize;
    // tailMSize 必须 16 对齐 (Blaze matmul 最小行粒度)
    if (tailMSize > 0 && tailMSize % QMRS_MIN_ROW_ALIGN != 0) {
        throw std::invalid_argument("ERROR: tailMSize=" + std::to_string(tailMSize) +
                                    " not 16-aligned. "
                                    "chunkM=" +
                                    std::to_string(chunkM) + " headMSize=" + std::to_string(headMSize) +
                                    ". Use a headMSize that divides chunkM into 16-aligned segments.");
    }
    uint32_t tailTileCnt = (tailMSize > 0) ? 1 : 0;
    uint32_t tileCnt = headTileCnt + tailTileCnt;

    // 通信 tiles 上限校验: remote tile 通知 flagid 直接使用 tid (kernel 侧同源常量, 经 include 链引入).
    // dav_3510 flagid 为 4bit: id 14 保留给 SyncAll<true>, id 15 保留未用, tiles 可用 {0..13} ⇒ T ≤ 14.
    if (tileCnt > QMRS_MAX_COMM_TILES) {
        throw std::invalid_argument("ERROR: tileCnt=" + std::to_string(tileCnt) +
                                    " exceeds QMRS_MAX_COMM_TILES=" + std::to_string(QMRS_MAX_COMM_TILES) +
                                    " (tid-as-flagid safety bound). Reduce M or increase headMSize.");
    }

    // win 容量防御校验: data win = [0, winReserved) barrier 保留区(对齐 kernel 侧 win 数据区起点公式:
    // localFlag R*32 + teamSyncCounter R*32, 对齐 128) + [winReserved, winReserved+R*chunkBytes) PUT 数据区,
    // 需 <= hcclBufSize. tiling 阶段 HCCL 建链未完成、实际 hcclBufSize 不可得, 以实测上限 400MB 为界.
    constexpr uint64_t QMRS_WIN_ALIGN = 128;                           // 与 impl.h WIN_DATA_ALIGN 同口径
    constexpr uint64_t QMRS_HCCL_BUF_SIZE_LIMIT = 400UL * 1024 * 1024; // hcclBufSize 实测上限
    uint64_t winReservedBytes = (2UL * static_cast<uint64_t>(rankNum) * QMRS_BARRIER_FLAG_BYTES + QMRS_WIN_ALIGN - 1) /
                                QMRS_WIN_ALIGN * QMRS_WIN_ALIGN;
    uint64_t chunkBytes = static_cast<uint64_t>(chunkM) * static_cast<uint64_t>(n) * sizeof(uint16_t);
    uint64_t winRequiredBytes = winReservedBytes + static_cast<uint64_t>(rankNum) * chunkBytes;
    if (winRequiredBytes > QMRS_HCCL_BUF_SIZE_LIMIT) {
        throw std::invalid_argument(
            "ERROR: win required=" + std::to_string(winRequiredBytes) + " bytes (" + std::to_string(winReservedBytes) +
            " reserved + rankNum=" + std::to_string(rankNum) + " x chunkBytes=" + std::to_string(chunkBytes) +
            ") exceeds hcclBufSize limit=" + std::to_string(QMRS_HCCL_BUF_SIZE_LIMIT) +
            " bytes. Reduce m/n or rankNum.");
    }

    tilingData.commTilingData.splitAxisTileCnt = static_cast<uint64_t>(headTileCnt);
    tilingData.commTilingData.splitAxisTileSize = static_cast<uint64_t>(headMSize);
    tilingData.commTilingData.splitAxisTailSize = static_cast<uint64_t>(tailMSize);
    tilingData.commTilingData.splitAxisTailCnt = static_cast<uint64_t>(tailTileCnt);
    tilingData.commTilingData.nonSplitAxisSize = static_cast<uint64_t>(n);

    // scope 级 tiling: 各 scope fragment 数不同 (REMOTE: rankNum-1 个, LOCAL: 1 个),
    // 调度器 tail 参数 (mTailMain/mBaseTailSplitCnt 等) 按设计 m 编码绝对位置, m 不匹配时
    // tile 形状非法 → L0A/L1 越界。每个 scope 的 tiling m 必须等于该 scope 实际逻辑 M:
    //   REMOTE head 片段 = (R-1)*headMSize, REMOTE tail 片段 = (R-1)*tailMSize,
    //   LOCAL head 片段 = headMSize, LOCAL tail 片段 = tailMSize (T=1 时 headMSize=chunkM)
    {
        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN,
                              mm::BiasDataType::DT_FLOAT>
            headTilingEngine;
        headTilingEngine.EnableBaseMHalving(true); // 块数<2×核数时 baseM 减半, 提升负载均衡(同 allto_all_matmul_v2)
        uint32_t remoteCnt = static_cast<uint32_t>(rankNum) - 1U;
        headTilingEngine.GetTilingData(remoteCnt * headMSize, n, k, false, true, tilingData.remoteHeadMmTilingData);
        headTilingEngine.GetTilingData(headMSize, n, k, false, true, tilingData.localHeadMmTilingData);
        if (tailTileCnt > 0 && tailMSize > 0) {
            QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN,
                                  mm::BiasDataType::DT_FLOAT>
                tailTilingEngine;
            uint32_t remoteTailLogicalM = (static_cast<uint32_t>(rankNum) - 1U) * tailMSize;
            tailTilingEngine.GetTilingData(remoteTailLogicalM, n, k, false, true, tilingData.remoteTailMmTilingData);
            tailTilingEngine.GetTilingData(tailMSize, n, k, false, true, tilingData.localTailMmTilingData);
        } else {
            tilingData.remoteTailMmTilingData = {};
            tilingData.localTailMmTilingData = {};
        }
    }

    if (rankId == 0) {
        INFO_LOG("chunkM=%u, headMSize=%u, headTileCnt=%u, tailMSize=%u, tailTileCnt=%u, tileCnt=%u (PUT multi-tile), "
                 "rankSize=%d",
                 chunkM, headMSize, headTileCnt, tailMSize, tailTileCnt, tileCnt, rankNum);

        auto printTiling = [](const char *name, const QuantMatmulTilingData &t) {
            INFO_LOG("[Tiling] %s: m=%u n=%u k=%u baseM=%u baseN=%u baseK=%u stepK=%u usedCoreNum=%u dbL0c=%u "
                     "nBufferNum=%u scaleKL1=%u mTailTile=%u nTailTile=%u mBaseTailSplitCnt=%u nBaseTailSplitCnt=%u "
                     "mTailMain=%u nTailMain=%u",
                     name, t.m, t.n, t.k, t.baseM, t.baseN, t.baseK, t.stepK, t.usedCoreNum, t.dbL0c, t.nBufferNum,
                     t.scaleKL1, t.mTailTile, t.nTailTile, t.mBaseTailSplitCnt, t.nBaseTailSplitCnt, t.mTailMain,
                     t.nTailMain);
        };
        printTiling("mmTilingData(usedCoreNum only)", tilingData.mmTilingData);
        printTiling("remoteHeadMmTilingData", tilingData.remoteHeadMmTilingData);
        printTiling("localHeadMmTilingData", tilingData.localHeadMmTilingData);
        if (tailTileCnt > 0 && tailMSize > 0) {
            printTiling("remoteTailMmTilingData", tilingData.remoteTailMmTilingData);
            printTiling("localTailMmTilingData", tilingData.localTailMmTilingData);
        }
    }

    // blockDim/rsCoreNum 下溢防护: rsCoreNum = usedCoreNum - rankSize (kernel 侧 uint32 减法),
    // usedCoreNum <= rankSize 时回绕为超大值 → 通信/计算角色模型破坏; == 时 rsCoreNum=0 →
    // 归约核缺失, 输出永不写入。host 侧强校验 (kernel 侧 InitAivRunContext 同源注释)。
    if (tilingData.mmTilingData.usedCoreNum <= static_cast<uint32_t>(rankNum)) {
        throw std::invalid_argument(
            "ERROR: mmTilingData.usedCoreNum=" + std::to_string(tilingData.mmTilingData.usedCoreNum) +
            " must be greater than rankNum=" + std::to_string(rankNum) + " (rsCoreNum underflow guard)");
    }

    INFO_LOG("[Rank %d] Step 1: aclInit", rankId);
    ACL_CHECK(aclInit(nullptr));
    int32_t deviceId = rankId;
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    INFO_LOG("[Rank %d] Step 2: device set, stream created", rankId);

    RootInfoExchanger exchanger(static_cast<uint32_t>(rankId), static_cast<uint32_t>(rankNum), ipport);
    HcclRootInfo rootInfo;
    INFO_LOG("[Rank %d] Step 3: exchanging rootInfo", rankId);
    if (!exchanger.Exchange(rootInfo)) {
        ERROR_LOG("rank %d exchange rootInfo failed", rankId);
        return -1;
    }
    INFO_LOG("[Rank %d] Step 4: rootInfo exchanged", rankId);

    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclWorldRankID = static_cast<uint32_t>(rankId);
    HcclComm comm = nullptr;
    INFO_LOG("[Rank %d] Step 5: HcclCommInitRootInfoConfig", rankId);
    HCCL_CHECK(HcclCommInitRootInfoConfig(static_cast<uint32_t>(rankNum), &rootInfo, static_cast<uint32_t>(rankId),
                                          &config, &comm));
    if (comm == nullptr) {
        ERROR_LOG("rank %d HcclCommInitRootInfoConfig failed", rankId);
        return -1;
    }
    INFO_LOG("[Rank %d] Step 6: HCCL comm initialized", rankId);

    CommChannelBuilder builder(comm);
    CommContext hostCtx = {};
    const char *ctxTag = "quant_matmul_reduce_scatter";

    INFO_LOG("[Rank %d] Step 7: CreateDeviceContext", rankId);
    CommContext *devContext = reinterpret_cast<CommContext *>(
        builder.CreateDeviceContext(&hostCtx, sizeof(CommContext), ctxTag, &hostCtx.udmaCtx, &hostCtx.ubmemCtx));
    if (devContext == nullptr) {
        ERROR_LOG("rank %d CreateDeviceContext failed", rankId);
        return -1;
    }
    INFO_LOG("[Rank %d] Step 8: device context created", rankId);

    exchanger.Barrier();
    exchanger.Close();
    INFO_LOG("[Rank %d] Step 9: exchanger barrier + close", rankId);

    // 输入数据：A [m, k] fp8, ScaleA [m, ceil(k/64)*2] fp8_e8m0
    // B [n, k] fp8 (TransB=true, 列主序), ScaleB [n, ceil(k/64)*2] fp8_e8m0
    // size_t 先行提升防 int32×int32 有符号溢出 (m*k/n*k 可超 INT_MAX → UB/回绕)
    std::vector<uint8_t> hostA(static_cast<size_t>(m) * k, 0);
    std::vector<uint8_t> hostB(static_cast<size_t>(n) * k, 0); // TransB=true: B 存储为 [n, k] C-order
    std::vector<uint8_t> hostScaleA(static_cast<size_t>(m) * CeilDiv(k, QMRS_MX_GROUP_SIZE) * QMRS_MX_SCALE_BYTES, 0);
    std::vector<uint8_t> hostScaleB(static_cast<size_t>(n) * CeilDiv(k, QMRS_MX_GROUP_SIZE) * QMRS_MX_SCALE_BYTES, 0);
    // 输出：[chunkM, n] bfloat16 per rank
    std::vector<uint16_t> hostOutput(static_cast<size_t>(chunkM) * n, 0);

    auto sizeA = static_cast<size_t>(1) * hostA.size() * sizeof(uint8_t);
    auto sizeB = static_cast<size_t>(1) * hostB.size() * sizeof(uint8_t);
    auto sizeScaleA = static_cast<size_t>(1) * hostScaleA.size() * sizeof(uint8_t);
    auto sizeScaleB = static_cast<size_t>(1) * hostScaleB.size() * sizeof(uint8_t);
    auto sizeOutput = static_cast<size_t>(1) * hostOutput.size() * sizeof(uint16_t);

    char exePath[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    std::string baseDir = ".";
    if (len > 0) {
        exePath[len] = '\0';
        baseDir = exePath;
        size_t lastSlash = baseDir.find_last_of('/');
        if (lastSlash != std::string::npos) {
            baseDir.resize(lastSlash);
        }
    }
    std::string inputDir = baseDir + "/input/" + std::to_string(rankId);
    std::string outputDir = baseDir + "/output/" + std::to_string(rankId);
    // 输入加载失败须显式失败: 静默用全零数据继续会输出全零结果, 精度比对失败且无根因线索
    if (!ReadFile(inputDir + "/input_a.bin", hostA.data(), sizeA)) {
        ERROR_LOG("ReadFile failed: %s", (inputDir + "/input_a.bin").c_str());
        return -1;
    }
    if (!ReadFile(inputDir + "/input_b.bin", hostB.data(), sizeB)) {
        ERROR_LOG("ReadFile failed: %s", (inputDir + "/input_b.bin").c_str());
        return -1;
    }
    if (!ReadFile(inputDir + "/input_scaleA.bin", hostScaleA.data(), sizeScaleA)) {
        ERROR_LOG("ReadFile failed: %s", (inputDir + "/input_scaleA.bin").c_str());
        return -1;
    }
    if (!ReadFile(inputDir + "/input_scaleB.bin", hostScaleB.data(), sizeScaleB)) {
        ERROR_LOG("ReadFile failed: %s", (inputDir + "/input_scaleB.bin").c_str());
        return -1;
    }

    GM_ADDR deviceA = nullptr;
    GM_ADDR deviceB = nullptr;
    GM_ADDR deviceScaleA = nullptr;
    GM_ADDR deviceScaleB = nullptr;
    GM_ADDR deviceOutput = nullptr;
    GM_ADDR deviceWorkspace = nullptr;

    // 显存申请失败必须退出: ACL_CHECK 仅打印不返回, nullptr 传入 memcpy/kernel 会触发
    // 设备侧非法访问; 失败路径释放已申请资源
    auto mallocOrCleanup = [&](GM_ADDR *ptr, size_t size, const char *name) -> bool {
        if (aclrtMalloc((void **)ptr, size, ACL_MEM_MALLOC_HUGE_ONLY) != ACL_SUCCESS) {
            std::cerr << "aclrtMalloc failed: " << name << " (" << size << " bytes)" << std::endl;
            if (deviceA != nullptr)
                aclrtFree(deviceA);
            if (deviceB != nullptr)
                aclrtFree(deviceB);
            if (deviceScaleA != nullptr)
                aclrtFree(deviceScaleA);
            if (deviceScaleB != nullptr)
                aclrtFree(deviceScaleB);
            if (deviceOutput != nullptr)
                aclrtFree(deviceOutput);
            if (deviceWorkspace != nullptr)
                aclrtFree(deviceWorkspace);
            return false;
        }
        return true;
    };
    if (!mallocOrCleanup(&deviceA, sizeA, "deviceA") || !mallocOrCleanup(&deviceB, sizeB, "deviceB") ||
        !mallocOrCleanup(&deviceScaleA, sizeScaleA, "deviceScaleA") ||
        !mallocOrCleanup(&deviceScaleB, sizeScaleB, "deviceScaleB")) {
        return -1;
    }
    // cGm: 最终输出 [chunkM, n] bfloat16
    size_t sizeC = static_cast<size_t>(chunkM) * n * sizeof(uint16_t);
    if (!mallocOrCleanup(&deviceOutput, sizeC, "deviceOutput")) {
        return -1;
    }
    // workspace: AIC 写入 C_i 的缓冲区, 大小 = m * n * sizeof(bfloat16)
    // PUT 模式下 localAddr = workspace, AIV 从此处推送到其他 rank win 区
    size_t sizeWorkspace = static_cast<size_t>(m) * n * sizeof(uint16_t);
    if (!mallocOrCleanup(&deviceWorkspace, sizeWorkspace, "deviceWorkspace")) {
        return -1;
    }
    INFO_LOG("[Rank %d] workspace size=%zu bytes (m=%d, n=%d)", rankId, sizeWorkspace, m, n);

    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceScaleA, sizeScaleA, hostScaleA.data(), sizeScaleA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceScaleB, sizeScaleB, hostScaleB.data(), sizeScaleB, ACL_MEMCPY_HOST_TO_DEVICE));
    INFO_LOG("[Rank %d] Step 10: data loaded to device", rankId);

    INFO_LOG("[Rank %d] Step 11: launching kernel (usedCoreNum=%u, PUT mode)", rankId,
             tilingData.mmTilingData.usedCoreNum);
    QuantMatmulReduceScatterKernelE4M3E4M3<<<tilingData.mmTilingData.usedCoreNum, nullptr, stream>>>(
        devContext, deviceA, deviceScaleA, deviceB, deviceScaleB, deviceWorkspace, deviceOutput, tilingData);
    INFO_LOG("[Rank %d] Step 12: kernel launched, syncing...", rankId);

    ACL_CHECK(aclrtSynchronizeStream(stream));
    INFO_LOG("[Rank %d] Step 13: stream synced", rankId);

    // 只拷回 chunkM * n 的输出（reduceSum 后的 per-rank 结果），precision/perf 模式共用
    ACL_CHECK(aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST));

    if (!WriteFile(outputDir + "/npu_out.bin", hostOutput.data(), sizeOutput)) {
        ERROR_LOG("WriteFile failed: %s", (outputDir + "/npu_out.bin").c_str());
        return -1;
    }

    constexpr int32_t BENCHMARK_ITERATIONS = 10;
    int32_t iterations = (mode == "perf") ? BENCHMARK_ITERATIONS : 1;
    if (iterations > 1) {
        for (int32_t i = 1; i < iterations; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            QuantMatmulReduceScatterKernelE4M3E4M3<<<tilingData.mmTilingData.usedCoreNum, nullptr, stream>>>(
                devContext, deviceA, deviceScaleA, deviceB, deviceScaleB, deviceWorkspace, deviceOutput, tilingData);
            ACL_CHECK(aclrtSynchronizeStream(stream));
            auto t1 = std::chrono::steady_clock::now();
            double elapsedUs = std::chrono::duration<double, std::micro>(t1 - t0).count();
            INFO_LOG("[Rank %d] iteration %d/%d: %.3f us", rankId, i + 1, iterations, elapsedUs);
        }
    }

    aclrtFree(deviceA);
    aclrtFree(deviceScaleA);
    aclrtFree(deviceB);
    aclrtFree(deviceScaleB);
    aclrtFree(deviceOutput);
    aclrtFree(deviceWorkspace);

    HcclCommDestroy(comm);

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return 0;
}

int main(int argc, char *argv[])
{
    int m, k, n, rankNum, headMSize;
    int localScopeLast = 0;
    std::string mode;
    try {
        parseArguments(argc, argv, m, k, n, rankNum, mode, headMSize, localScopeLast);
    } catch (const std::invalid_argument &e) {
        std::cerr << e.what() << std::endl;
        printUsage(argv[0]);
        return -1;
    } catch (const std::out_of_range &e) {
        std::cerr << "ERROR: argument exceeds int range: " << e.what() << std::endl;
        printUsage(argv[0]);
        return -1;
    }

    INFO_LOG("Master (PID=%d) will fork %d processes (mode=%s, headMSize=%d, localScopeLast=%d)", getpid(), rankNum,
             mode.c_str(), headMSize, localScopeLast);

    std::vector<pid_t> pids(rankNum);
    for (int rankId = 0; rankId < rankNum; ++rankId) {
        pid_t pid = fork();

        if (pid < 0) {
            ERROR_LOG("Fork failed for rank %d", rankId);
            exit(-1);
        } else if (pid == 0) {
            int ret = runQuantMatmulReduceScatter(rankNum, rankId, m, k, n, mode, headMSize, localScopeLast);
            exit(ret);
        } else {
            pids[rankId] = pid;
            INFO_LOG("Forked Rank %d -> PID %d", rankId, pid);
        }
    }

    int status = 0;
    bool all_success = true;
    for (int rankId = 0; rankId < rankNum; ++rankId) {
        pid_t pid = pids[rankId];
        pid_t waitRet = waitpid(pid, &status, 0);
        if (waitRet == -1) {
            all_success = false;
            ERROR_LOG("waitpid failed for PID %d", pid);
            continue;
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            all_success = false;
            ERROR_LOG("Worker PID %d failed", pid);
        }
    }

    std::cout << "All workers finished. Status: " << (all_success ? "SUCCESS" : "FAILURE") << std::endl;
    return all_success ? 0 : -1;
}
