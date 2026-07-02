/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License").
 *
 *  chain_e2e_driver.cpp —— a6 端到端【4 算子链】集成驱动（真数据面，2 die / 2 rank MPI）。
 *  ============================================================================================
 *  链序（连跑，非单跑）：
 *    ① dispatch_setup  (CCU 直调) —— 本卡 token(mxfp8 payload+scale) 按 host 预算 routeTable 跨卡写到
 *                                     对端 expert-major 数据区；置 ready。
 *    ② dispatch_teardown (AIV aclnn) —— 读本卡数据区 + wishCnt → 去空洞紧密化 expandX + sendCounts
 *                                       + expandIdx + expertTokenNums。
 *    ③ combine_setup   (CCU 直调) —— 本卡 expandX 按 host 预算 assistTable 送回各源卡 combine 数据区。
 *    ④ combine_teardown (AIV aclnn) —— 读 combine 数据区 + expert_scales + expandIdx → 加权和 → x(bf16)。
 *
 *  与旧 dispatch_min_driver.cpp（status 位握手版，KernelArg 2 参/TaskArg 4 参）的区别：本驱动接【当前真
 *  数据面 kernel】—— CcuKernelArgDispatchSetup(6 参)/CcuTaskArgDispatchSetup(14 参=7 addr+7 token)，
 *  combine 同构。机制母体（MPI/comm 拉起/HcclGetOneChannel/清理/宏）忠实复用旧 driver（已在 a6 跑通结构）。
 *
 *  host 预算簿记（技术总监拍板，CCU 无动态标量 GM 写）：routeTable / wishCnt / assistTable / expandIdx 全部
 *  由本驱动 host 端用 setup_host_precompute.h 算好、传给算子。地址布局与已测 teardown 契约字节级一致。
 *
 *  ★ 可验证性边界（G5 三层，如实）：
 *    · a5：只能编【CCU 关的 shell】(CHAIN_CCU_A6=OFF)——CCU 本体符号 a6-only、a5 libhcomm 未导出。
 *    · a6：CHAIN_CCU_A6=ON 编入两个 CCU kernel + 真跨卡写；mpirun -n 2 跑全链。CCU 数据面正确性 a6 坐实。
 *    · golden：host 侧按 mxfp8 dequant + 加权和重建 x，与算子输出比（bf16 容差）。
 *
 *  a6-TODO（a5 看不到、a6 实测确认）：见旧 driver [T1/T2/T3/T5/T7/T8/T9]；本文件额外：
 *    [C1] fp8/e8m0 的 aclDataType 枚举名（ACL_FLOAT8_E4M3FN / ACL_FLOAT8_E8M0）以 a6 acl/acl_base.h 为准；
 *    [C2] combine_setup 与 dispatch_setup 若共用 commCmdInfo，需确认两算子 ready 区不冲突（本版各分配一份）。
 */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>

#include <mpi.h>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnn/aclnn_base.h"

#include "setup_host_precompute.h"         // 纯 C++：mc2::BuildDispatchRoute/WishCnt/ExpandIdx/CombineAssist（a5 可编）
#include "aclnn_dispatch_teardown.h"       // aclnnDispatchTeardown 两段式（aclnn C API，a5 可编）
#include "aclnn_combine_teardown.h"        // aclnnCombineTeardown 两段式

// ★ CCU/hcomm 头 + 两个 CCU kernel 头【全部门控 a6】：a5 缺 ccu_kernel_arg.h 等（见 memory a5_ccu_env_facts），
//   且这些类型只在 #ifdef CHAIN_CCU_A6 的 launch 代码里用；a5 shell 不 include → 只依赖 CANN 公共头，编译稳。
#ifdef CHAIN_CCU_A6
#include "dispatch_setup_ccu.h"            // mc2::CcuKernel/Arg/TaskArg DispatchSetup（含 ccu_kernel.h）
#include "combine_setup_ccu.h"             // mc2::CcuKernel/Arg/TaskArg CombineSetup
#include "hcomm_res_defs.h"                // CommEngine / COMM_ENGINE_CCU
#include "hcomm_primitives.h"              // ThreadHandle
#include "hccl/hccl_res.h"                 // HcclThreadAcquireWithStream
#include "hcomm/ccu/hccl_ccu_res.h"        // HcclCcuKernelRegister / RegisterFinish / Launch；CcuKernelHandle
#include "hcomm/ccu/ccu_assist_pub.h"      // hcomm::CcuRep::GetTokenInfo
#include "hccl/hccl_rank_graph.h"          // HcclRankGraphGetLinks / CommLink / HcclChannelDesc / HcclChannelAcquire
#endif

#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS 0
#endif

// CCU 扩展模式值（修 a6 "ccuContainer is nullptr"）；a6 基线数组约定 ccu_sched=6，可 -DCHAIN_CCU_EXPANSION_MODE=5。
#ifndef CHAIN_CCU_EXPANSION_MODE
#define CHAIN_CCU_EXPANSION_MODE 6
#endif

// fp8/e8m0 的 acl dtype（[C1] a6 以 acl_base.h 实名为准；若枚举名不同，改这两行即可）。
#ifndef CHAIN_ACL_FP8_E4M3
#define CHAIN_ACL_FP8_E4M3 ACL_FLOAT8_E4M3FN
#endif
#ifndef CHAIN_ACL_E8M0
#define CHAIN_ACL_E8M0 ACL_FLOAT8_E8M0
#endif

namespace {

// ============================== 宏（复用旧 driver）==============================
#define ACLCHECK(expr)                                                                    \
    do {                                                                                  \
        aclError __r = (expr);                                                            \
        if (__r != ACL_SUCCESS) { printf("[ERR] %s ret=%d @%d\n", #expr, __r, __LINE__); MPI_Abort(MPI_COMM_WORLD, -1); return -1; } \
    } while (0)

#define HCCLCHECK(expr)                                                                   \
    do {                                                                                  \
        HcclResult __r = (expr);                                                          \
        if (__r != HCCL_SUCCESS) { printf("[ERR] %s ret=%d @%d\n", #expr, __r, __LINE__); MPI_Abort(MPI_COMM_WORLD, -1); return -1; } \
    } while (0)

#define ACLNNCHECK(expr)                                                                  \
    do {                                                                                  \
        aclnnStatus __s = (expr);                                                         \
        if (__s != ACLNN_SUCCESS) { printf("[ERR] %s ret=%d @%d\n", #expr, __s, __LINE__); MPI_Abort(MPI_COMM_WORLD, -1); return -1; } \
    } while (0)

// ============================== 链配置（真实场景默认值）==============================
struct ChainCfg {
    uint32_t bs   = 8;       // 每卡 token 数
    uint32_t h    = 7168;    // hidden
    uint32_t topk = 8;       // 每 token 选专家数
    uint32_t E    = 4;       // 每卡本地专家数 (expert_per_rank)
    uint32_t R    = 2;       // rank_num（= mpiSize，2 die）
    uint32_t seed = 1234;
    // 派生
    uint32_t GE()      const { return R * E; }                          // 全局专家数
    uint64_t A()       const { return (uint64_t)R * bs * std::min(topk, E); }  // expandX 行上限
    uint64_t dataRows()const { return (uint64_t)GE() * bs; }            // 数据区/combine区行数 = R*E*bs
    uint64_t hB()      const { return h; }                              // fp8 payload：1 字节/元素
    uint64_t sB()      const { return h / 32U; }                        // e8m0 scale：h/32
};

// ============================== helpers ==============================
// 通用 aclTensor 构造（contiguous, ND）：给 GM 地址 + shape + dtype。
aclTensor *MakeTensor(void *devAddr, const std::vector<int64_t> &shape, aclDataType dtype)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; --i) strides[i] = strides[i + 1] * shape[i + 1];
    return aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), devAddr);
}

// fp8_e4m3fn(1 字节) → float（sign1/exp4/mant3, bias7；e=0 subnormal；e=15&m=7 是 NaN，按值域用不触发）。
// [[maybe_unused]]：golden 在 #ifdef CHAIN_CCU_A6 下才用；a5 shell 编不进 → 加标记免 -Wunused-function。
[[maybe_unused]] float Fp8E4m3ToFloat(uint8_t b)
{
    uint32_t s = (b >> 7) & 0x1, e = (b >> 3) & 0xF, m = b & 0x7;
    float val;
    if (e == 0) { val = std::ldexp((float)m / 8.0f, 1 - 7); }        // subnormal: 2^(1-bias) * m/8
    else        { val = std::ldexp(1.0f + (float)m / 8.0f, (int)e - 7); }  // normal
    return s ? -val : val;
}
// e8m0(1 字节) → 缩放因子 2^(E-127)。
[[maybe_unused]] float E8m0ToScale(uint8_t E) { return std::ldexp(1.0f, (int)E - 127); }

#ifdef CHAIN_CCU_A6
// A6 channel 获取（复用旧 driver HcclGetOneChannel，resGroupTag 按算子 TAG 传入）。
static HcclResult HcclGetOneChannel(HcclComm comm, CommEngine engine,
    uint32_t srcRank, uint32_t dstRank, const char *resTag, ChannelHandle *channel)
{
    uint32_t netLayer = 0, listSize = 0;
    CommLink *linkList = nullptr;
    HcclResult ret = HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &linkList, &listSize);
    if (ret != HCCL_SUCCESS) { printf("[ERR] HcclRankGraphGetLinks %u->%u ret=%d\n", srcRank, dstRank, ret); return ret; }

    HcclChannelDesc desc;
    ret = HcclChannelDescInit(&desc, 1);
    if (ret != HCCL_SUCCESS) { printf("[ERR] HcclChannelDescInit ret=%d\n", ret); return ret; }
    const CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
    const char *tmpArg[] = { resTag };
    char **resGroupTag = const_cast<char **>(tmpArg);

    for (uint32_t idx = 0; idx < listSize; ++idx) {
        CommLink link = linkList[idx];
        if (link.linkAttr.linkProtocol == protocol) {
            desc.remoteRank              = dstRank;
            desc.notifyNum               = 2;
            desc.channelProtocol         = link.linkAttr.linkProtocol;
            desc.localEndpoint.protocol  = link.srcEndpointDesc.protocol;
            desc.localEndpoint.commAddr  = link.srcEndpointDesc.commAddr;
            desc.localEndpoint.loc       = link.srcEndpointDesc.loc;
            desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
            desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
            desc.remoteEndpoint.loc      = link.dstEndpointDesc.loc;
            desc.ccuUbAttr.resGroupTag    = resGroupTag;
            desc.ccuUbAttr.resGroupTagNum = 1;
            return HcclChannelAcquire(comm, engine, &desc, 1, channel);
        }
    }
    printf("[ERR] no UBC_CTP link %u->%u\n", srcRank, dstRank); return HCCL_E_NOT_FOUND;
}
#endif  // CHAIN_CCU_A6

// ============================== GM 区（本 rank 全部设备指针 + token）==============================
struct ChainGm {
    // ① dispatch_setup 输入
    void *xData = nullptr, *xScale = nullptr, *routeTable = nullptr, *oneBuf = nullptr;
    // ①→② 本卡【接收】数据区（对端 CCU 写入 + dispatch_teardown 读）
    void *dataRegion = nullptr, *scaleRegion = nullptr, *commCmdInfoD = nullptr;
    // ② dispatch_teardown：wishCnt(host 预算输入) + 输出 / ③ combine_setup 输入
    void *wishCnt = nullptr;
    void *expandX = nullptr, *expandScale = nullptr, *sendCounts = nullptr, *expandIdx = nullptr, *expertTokenNums = nullptr;
    // ③ combine_setup：assistTable + 本卡【接收】combine 数据区（对端写 + combine_teardown 读）
    void *assistTable = nullptr, *commCmdInfoC = nullptr, *combineData = nullptr, *combineScale = nullptr;
    // ④ combine_teardown 输入 + 输出
    void *expertScales = nullptr, *expertIds = nullptr, *xOut = nullptr;
    // token（CCU 需要 (va,token) 对）
    uint64_t xDataTok = 0, xScaleTok = 0, routeTableTok = 0, oneBufTok = 0;
    uint64_t dataRegionTok = 0, scaleRegionTok = 0, commCmdInfoDTok = 0;
    uint64_t expandXTok = 0, expandScaleTok = 0, assistTableTok = 0, commCmdInfoCTok = 0;
    uint64_t combineDataTok = 0, combineScaleTok = 0;
    // host 侧留存（golden 用）
    std::vector<uint8_t> hXData;      // [bs*h] fp8
    std::vector<uint8_t> hXScale;     // [bs*(h/32)] e8m0
    std::vector<int32_t> hExpertIds;  // 本卡 [bs*topk]
    std::vector<float>   hExpertScales;// 本卡 [bs*topk]
};

// 全卡 distinct-topk expert_ids（各卡确定性、同 seed → 各 rank 可本地重建全卡表算 wishCnt）。
void GenAllExpertIds(const ChainCfg &c, std::vector<int32_t> &allIds /*[R*bs*topk]*/)
{
    allIds.assign((size_t)c.R * c.bs * c.topk, 0);
    const uint32_t GE = c.GE();
    // 简单确定性 LCG（不依赖 <random> 的实现差异，跨端一致）。
    uint64_t st = c.seed * 6364136223846793005ULL + 1442695040888963407ULL;
    auto next = [&](uint32_t n) { st = st * 6364136223846793005ULL + 1442695040888963407ULL; return (uint32_t)((st >> 33) % n); };
    for (uint32_t C = 0; C < c.R; ++C)
        for (uint32_t i = 0; i < c.bs; ++i) {
            std::vector<uint32_t> pool(GE); for (uint32_t g = 0; g < GE; ++g) pool[g] = g;
            for (uint32_t j = 0; j < c.topk; ++j) {                       // 部分 Fisher-Yates → 前 topk 互异
                uint32_t r = j + next(GE - j);
                std::swap(pool[j], pool[r]);
                allIds[((size_t)C * c.bs + i) * c.topk + j] = (int32_t)pool[j];
            }
        }
}

// 分配一块 device GM + 可选清零，返回指针（失败 MPI_Abort）。
#define MALLOC_GM(pp, bytes) do { ACLCHECK(aclrtMalloc((pp), (bytes), ACL_MEM_MALLOC_HUGE_FIRST)); } while (0)

// 分配所有 GM、造 host 输入、预算 LUT、灌 device。返回 0 成功。
int PrepChain(const ChainCfg &c, uint32_t rankId, ChainGm &g)
{
    const uint64_t hB = c.hB(), sB = c.sB();
    const uint64_t dataRows = c.dataRows(), A = c.A();
    const uint64_t xDataBytes = (uint64_t)c.bs * hB, xScaleBytes = (uint64_t)c.bs * sB;
    const uint64_t dataRegionBytes = dataRows * hB, scaleRegionBytes = dataRows * sB;
    const uint64_t expandXBytes = A * hB, expandScaleBytes = A * sB;
    const uint64_t combineDataBytes = dataRows * hB, combineScaleBytes = dataRows * sB;
    const uint64_t routeBytes = (uint64_t)c.bs * c.topk * mc2::ROUTE_FIELDS * sizeof(int64_t);
    const uint64_t assistBytes = A * mc2::ROUTE_FIELDS * sizeof(int64_t);
    const uint64_t wishBytes = (uint64_t)c.GE() * sizeof(int32_t);       // sendCounts 同形
    const uint64_t idxBytes = (uint64_t)c.bs * c.topk * sizeof(int32_t);
    const uint64_t etnBytes = (uint64_t)c.E * sizeof(int32_t);
    const uint64_t scalesBytes = (uint64_t)c.bs * c.topk * sizeof(float);
    const uint64_t xOutBytes = (uint64_t)c.bs * hB * 2;                  // bf16 = 2 字节/元素
    const uint64_t cmdBytes = (uint64_t)c.R * 64;                        // commCmdInfo：每 peer 64B（对齐旧版）
    const uint64_t oneBytes = (uint64_t)c.R * sizeof(int32_t);

    // ---- device 分配 ----
    MALLOC_GM(&g.xData, xDataBytes);        MALLOC_GM(&g.xScale, xScaleBytes);
    MALLOC_GM(&g.routeTable, routeBytes);   MALLOC_GM(&g.oneBuf, oneBytes);
    MALLOC_GM(&g.dataRegion, dataRegionBytes);  MALLOC_GM(&g.scaleRegion, scaleRegionBytes);
    MALLOC_GM(&g.commCmdInfoD, cmdBytes);
    MALLOC_GM(&g.expandX, expandXBytes);    MALLOC_GM(&g.expandScale, expandScaleBytes);
    MALLOC_GM(&g.sendCounts, wishBytes);    MALLOC_GM(&g.expandIdx, idxBytes);   MALLOC_GM(&g.expertTokenNums, etnBytes);
    MALLOC_GM(&g.assistTable, assistBytes); MALLOC_GM(&g.commCmdInfoC, cmdBytes);
    MALLOC_GM(&g.combineData, combineDataBytes); MALLOC_GM(&g.combineScale, combineScaleBytes);
    MALLOC_GM(&g.expertScales, scalesBytes); MALLOC_GM(&g.expertIds, idxBytes); MALLOC_GM(&g.xOut, xOutBytes);
    MALLOC_GM(&g.wishCnt, wishBytes);           // dispatch_teardown 输入；host 预算

    // 接收区/ready 区/输出区先清 0（未收到=0）
    ACLCHECK(aclrtMemset(g.dataRegion, dataRegionBytes, 0, dataRegionBytes));
    ACLCHECK(aclrtMemset(g.scaleRegion, scaleRegionBytes, 0, scaleRegionBytes));
    ACLCHECK(aclrtMemset(g.combineData, combineDataBytes, 0, combineDataBytes));
    ACLCHECK(aclrtMemset(g.combineScale, combineScaleBytes, 0, combineScaleBytes));
    ACLCHECK(aclrtMemset(g.commCmdInfoD, cmdBytes, 0, cmdBytes));
    ACLCHECK(aclrtMemset(g.commCmdInfoC, cmdBytes, 0, cmdBytes));

    // ---- host 造数据 ----
    std::vector<int32_t> allIds; GenAllExpertIds(c, allIds);
    const int32_t *myIds = allIds.data() + (size_t)rankId * c.bs * c.topk;
    g.hExpertIds.assign(myIds, myIds + (size_t)c.bs * c.topk);

    // x（mxfp8）★ golden 判别力（审核 golden 项修）：payload 逐 (i,k) 取【E4M3 精确可表达且随 i,k 变化】的字节，
    //   scale 逐 token 变（e8m0∈{126,127,128}→{0.5,1,2}）。这样 token 置换(expandIdx/pos off-by-one)+元素乱序+
    //   scale 路径错 都会让 golden 失配、被 bad>0 抓到（旧版 v=(rank+1)+i*0.03125 会被 fp8 坍缩成相邻同字节→假 PASS）。
    //   payload 字节 = (expBase<<3)|((i+k)&7)：sign0、exp=7+rankId(rank 可辨)、尾数随 i+k（0..7 精确、非 NaN）。
    g.hXData.assign((size_t)c.bs * hB, 0);
    g.hXScale.assign((size_t)c.bs * sB, 0);
    const uint8_t expBase = (uint8_t)(7u + rankId);           // rank0 exp7(~1.x) / rank1 exp8(~2.x)，跨 rank 可辨
    for (uint32_t i = 0; i < c.bs; ++i) {
        for (uint64_t k = 0; k < hB; ++k)
            g.hXData[(size_t)i * hB + k] = (uint8_t)((expBase << 3) | (uint8_t)((i + k) & 0x7u));
        const uint8_t se = (uint8_t)(126u + (i % 3u));        // e8m0 逐 token 轮换 0.5/1/2
        for (uint64_t gg = 0; gg < sB; ++gg) g.hXScale[(size_t)i * sB + gg] = se;
    }
    // expert_scales：one-hot（第 0 个专家权重 1、其余 0）→ combine_teardown 输出 = 单副本 dequant，golden 精确可算。
    g.hExpertScales.assign((size_t)c.bs * c.topk, 0.0f);
    for (uint32_t i = 0; i < c.bs; ++i) g.hExpertScales[(size_t)i * c.topk + 0] = 1.0f;

    // ---- host 预算 4 张 LUT ----
    std::vector<int64_t> route((size_t)c.bs * c.topk * mc2::ROUTE_FIELDS);
    mc2::BuildDispatchRoute(myIds, c.bs, c.topk, c.E, c.R, rankId, hB, route.data());
    std::vector<int32_t> wish(c.GE());
    mc2::BuildDispatchWishCnt(rankId, allIds.data(), c.bs, c.topk, c.E, c.R, wish.data());  // 本卡作 owner 的 wishCnt
    std::vector<int64_t> assist((size_t)A * mc2::ROUTE_FIELDS);
    mc2::BuildCombineAssist(rankId, wish.data(), c.bs, c.topk, c.E, c.R, hB, A, assist.data());

    // ---- 灌 device ----
    ACLCHECK(aclrtMemcpy(g.xData, xDataBytes, g.hXData.data(), xDataBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemcpy(g.xScale, xScaleBytes, g.hXScale.data(), xScaleBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemcpy(g.routeTable, routeBytes, route.data(), routeBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemcpy(g.wishCnt, wishBytes, wish.data(), wishBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemcpy(g.assistTable, assistBytes, assist.data(), assistBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemcpy(g.expertIds, idxBytes, g.hExpertIds.data(), idxBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemcpy(g.expertScales, scalesBytes, g.hExpertScales.data(), scalesBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    { std::vector<int32_t> ones(c.R, 1);
      ACLCHECK(aclrtMemcpy(g.oneBuf, oneBytes, ones.data(), oneBytes, ACL_MEMCPY_HOST_TO_DEVICE)); }
    return 0;
}

// ============================== ① dispatch_setup（CCU 直调，14 参真数据面）==============================
int LaunchDispatchSetupCcu(HcclComm comm, aclrtStream stream, const ChainCfg &c, ChainGm &g)
{
#ifdef CHAIN_CCU_A6
    uint32_t rid = 0, rsz = 0;
    HCCLCHECK(HcclGetRankId(comm, &rid));
    HCCLCHECK(HcclGetRankSize(comm, &rsz));
    const uint32_t channelNum = rsz - 1U;
    std::vector<ChannelHandle> channels(channelNum);
    for (uint32_t k = 0; k < channelNum; ++k) {
        uint32_t dstRank = (k < rid) ? k : k + 1;
        HCCLCHECK(HcclGetOneChannel(comm, CommEngine::COMM_ENGINE_CCU, rid, dstRank, "moe_dispatch_05rtt", &channels[k]));
    }
    ThreadHandle mainThread = 0;
    HCCLCHECK(HcclThreadAcquireWithStream(comm, CommEngine::COMM_ENGINE_CCU, stream, 1, &mainThread));

    mc2::CcuKernelArgDispatchSetup kernelArg(rsz, rid, c.E, c.bs, c.topk, c.h);
    kernelArg.channels = channels;
    hcomm::KernelCreator creator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<mc2::CcuKernelDispatchSetup>(arg);
    };
    CcuKernelHandle handle = 0;
    HCCLCHECK(HcclCcuKernelRegister(comm, &handle, reinterpret_cast<void *>(&creator), static_cast<void *>(&kernelArg)));
    HCCLCHECK(HcclCcuKernelRegisterFinish(comm));

    const uint64_t hB = c.hB(), sB = c.sB(), dataRows = c.dataRows();
    g.xDataTok       = hcomm::CcuRep::GetTokenInfo((uint64_t)g.xData,       (uint64_t)c.bs * hB);
    g.xScaleTok      = hcomm::CcuRep::GetTokenInfo((uint64_t)g.xScale,      (uint64_t)c.bs * sB);
    g.routeTableTok  = hcomm::CcuRep::GetTokenInfo((uint64_t)g.routeTable,  (uint64_t)c.bs * c.topk * mc2::ROUTE_FIELDS * sizeof(int64_t));
    g.dataRegionTok  = hcomm::CcuRep::GetTokenInfo((uint64_t)g.dataRegion,  dataRows * hB);
    g.scaleRegionTok = hcomm::CcuRep::GetTokenInfo((uint64_t)g.scaleRegion, dataRows * sB);
    g.commCmdInfoDTok= hcomm::CcuRep::GetTokenInfo((uint64_t)g.commCmdInfoD,(uint64_t)c.R * 64);
    g.oneBufTok      = hcomm::CcuRep::GetTokenInfo((uint64_t)g.oneBuf,      (uint64_t)c.R * sizeof(int32_t));

    mc2::CcuTaskArgDispatchSetup taskArg(
        (uint64_t)g.xData, (uint64_t)g.xScale, (uint64_t)g.routeTable,
        (uint64_t)g.dataRegion, (uint64_t)g.scaleRegion, (uint64_t)g.commCmdInfoD, (uint64_t)g.oneBuf,
        g.xDataTok, g.xScaleTok, g.routeTableTok,
        g.dataRegionTok, g.scaleRegionTok, g.commCmdInfoDTok, g.oneBufTok);
    HCCLCHECK(HcclCcuKernelLaunch(comm, mainThread, handle, static_cast<void *>(&taskArg)));
    printf("[driver] ① dispatch_setup(CCU) launched (rid=%u rsz=%u channels=%u)\n", rid, rsz, channelNum);
    return 0;
#else
    (void)comm; (void)stream; (void)c; (void)g;
    printf("[driver] (a5) ① dispatch_setup(CCU) skipped — a6-only（-DCHAIN_CCU_A6=ON）\n");
    return 0;
#endif
}

// ============================== ② dispatch_teardown（AIV aclnn，11 张量）==============================
int LaunchDispatchTeardownAclnn(aclrtStream stream, const ChainCfg &c, ChainGm &g)
{
    const int64_t dataRows = (int64_t)c.dataRows(), A = (int64_t)c.A();
    const int64_t h = c.h, s = c.h / 32, GE = c.GE(), bs = c.bs, topk = c.topk, E = c.E;
    aclTensor *tokenData = MakeTensor(g.dataRegion, {dataRows, h}, CHAIN_ACL_FP8_E4M3);
    aclTensor *tokenScale= MakeTensor(g.scaleRegion,{dataRows, s}, CHAIN_ACL_E8M0);
    aclTensor *wishCntT  = MakeTensor(g.wishCnt,    {GE},          ACL_INT32);
    aclTensor *expertIdsT= MakeTensor(g.expertIds,  {bs, topk},    ACL_INT32);
    aclTensor *expandXT  = MakeTensor(g.expandX,    {A, h},        CHAIN_ACL_FP8_E4M3);
    aclTensor *expandScT = MakeTensor(g.expandScale,{A, s},        CHAIN_ACL_E8M0);
    aclTensor *sendCntT  = MakeTensor(g.sendCounts, {GE},          ACL_INT32);
    aclTensor *expandIdxT= MakeTensor(g.expandIdx,  {bs, topk},    ACL_INT32);
    aclTensor *etnT      = MakeTensor(g.expertTokenNums, {E},      ACL_INT32);

    uint64_t wsSize = 0; aclOpExecutor *exec = nullptr;
    ACLNNCHECK(aclnnDispatchTeardownGetWorkspaceSize(tokenData, tokenScale, wishCntT, expertIdsT,
        (int64_t)c.R, (int64_t)c.E, expandXT, expandScT, sendCntT, expandIdxT, etnT, &wsSize, &exec));
    void *ws = nullptr; if (wsSize > 0) MALLOC_GM(&ws, wsSize);
    ACLNNCHECK(aclnnDispatchTeardown(ws, wsSize, exec, stream));
    printf("[driver] ② dispatch_teardown(AIV/aclnn) launched (A=%lld)\n", (long long)A);
    (void)tokenData; (void)tokenScale; (void)wishCntT; (void)expertIdsT; (void)expandXT;
    (void)expandScT; (void)sendCntT; (void)expandIdxT; (void)etnT; (void)ws;   // sync 后进程退出回收
    return 0;
}

// ============================== ③ combine_setup（CCU 直调，14 参）==============================
int LaunchCombineSetupCcu(HcclComm comm, aclrtStream stream, const ChainCfg &c, ChainGm &g)
{
#ifdef CHAIN_CCU_A6
    uint32_t rid = 0, rsz = 0;
    HCCLCHECK(HcclGetRankId(comm, &rid));
    HCCLCHECK(HcclGetRankSize(comm, &rsz));
    const uint32_t channelNum = rsz - 1U;
    std::vector<ChannelHandle> channels(channelNum);
    for (uint32_t k = 0; k < channelNum; ++k) {
        uint32_t dstRank = (k < rid) ? k : k + 1;
        HCCLCHECK(HcclGetOneChannel(comm, CommEngine::COMM_ENGINE_CCU, rid, dstRank, "moe_combine_05rtt", &channels[k]));
    }
    ThreadHandle mainThread = 0;
    HCCLCHECK(HcclThreadAcquireWithStream(comm, CommEngine::COMM_ENGINE_CCU, stream, 1, &mainThread));

    mc2::CcuKernelArgCombineSetup kernelArg(rsz, rid, c.E, c.bs, c.topk, c.h);
    kernelArg.channels = channels;
    hcomm::KernelCreator creator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<mc2::CcuKernelCombineSetup>(arg);
    };
    CcuKernelHandle handle = 0;
    HCCLCHECK(HcclCcuKernelRegister(comm, &handle, reinterpret_cast<void *>(&creator), static_cast<void *>(&kernelArg)));
    HCCLCHECK(HcclCcuKernelRegisterFinish(comm));

    const uint64_t hB = c.hB(), sB = c.sB(), dataRows = c.dataRows(), A = c.A();
    g.expandXTok    = hcomm::CcuRep::GetTokenInfo((uint64_t)g.expandX,     A * hB);
    g.expandScaleTok= hcomm::CcuRep::GetTokenInfo((uint64_t)g.expandScale, A * sB);
    g.assistTableTok= hcomm::CcuRep::GetTokenInfo((uint64_t)g.assistTable, A * mc2::ROUTE_FIELDS * sizeof(int64_t));
    g.combineDataTok= hcomm::CcuRep::GetTokenInfo((uint64_t)g.combineData, dataRows * hB);
    g.combineScaleTok=hcomm::CcuRep::GetTokenInfo((uint64_t)g.combineScale,dataRows * sB);
    g.commCmdInfoCTok=hcomm::CcuRep::GetTokenInfo((uint64_t)g.commCmdInfoC,(uint64_t)c.R * 64);
    g.oneBufTok     = hcomm::CcuRep::GetTokenInfo((uint64_t)g.oneBuf,      (uint64_t)c.R * sizeof(int32_t));

    mc2::CcuTaskArgCombineSetup taskArg(
        (uint64_t)g.expandX, (uint64_t)g.expandScale, (uint64_t)g.assistTable,
        (uint64_t)g.combineData, (uint64_t)g.combineScale, (uint64_t)g.commCmdInfoC, (uint64_t)g.oneBuf,
        g.expandXTok, g.expandScaleTok, g.assistTableTok,
        g.combineDataTok, g.combineScaleTok, g.commCmdInfoCTok, g.oneBufTok);
    HCCLCHECK(HcclCcuKernelLaunch(comm, mainThread, handle, static_cast<void *>(&taskArg)));
    printf("[driver] ③ combine_setup(CCU) launched (rid=%u channels=%u)\n", rid, channelNum);
    return 0;
#else
    (void)comm; (void)stream; (void)c; (void)g;
    printf("[driver] (a5) ③ combine_setup(CCU) skipped — a6-only（-DCHAIN_CCU_A6=ON）\n");
    return 0;
#endif
}

// ============================== ④ combine_teardown（AIV aclnn）==============================
int LaunchCombineTeardownAclnn(aclrtStream stream, const ChainCfg &c, ChainGm &g)
{
    const int64_t dataRows = (int64_t)c.dataRows();
    const int64_t h = c.h, s = c.h / 32, bs = c.bs, topk = c.topk;
    aclTensor *combineDataT = MakeTensor(g.combineData, {dataRows, h}, CHAIN_ACL_FP8_E4M3);
    aclTensor *combineScT   = MakeTensor(g.combineScale,{dataRows, s}, CHAIN_ACL_E8M0);
    aclTensor *scalesT      = MakeTensor(g.expertScales,{bs, topk},    ACL_FLOAT);
    aclTensor *expandIdxT   = MakeTensor(g.expandIdx,   {bs, topk},    ACL_INT32);  // ← dispatch_teardown 的输出
    aclTensor *expertIdsT   = MakeTensor(g.expertIds,   {bs, topk},    ACL_INT32);
    aclTensor *xT           = MakeTensor(g.xOut,        {bs, h},       ACL_BF16);

    uint64_t wsSize = 0; aclOpExecutor *exec = nullptr;
    ACLNNCHECK(aclnnCombineTeardownGetWorkspaceSize(combineDataT, combineScT, scalesT, expandIdxT, expertIdsT,
        (int64_t)c.R, (int64_t)c.E, xT, &wsSize, &exec));
    void *ws = nullptr; if (wsSize > 0) MALLOC_GM(&ws, wsSize);
    ACLNNCHECK(aclnnCombineTeardown(ws, wsSize, exec, stream));
    printf("[driver] ④ combine_teardown(AIV/aclnn) launched\n");
    (void)combineDataT; (void)combineScT; (void)scalesT; (void)expandIdxT; (void)expertIdsT; (void)xT; (void)ws;
    return 0;
}

// bf16(uint16 高 16 位) → float。
[[maybe_unused]] float Bf16ToFloat(uint16_t b) { uint32_t bits = (uint32_t)b << 16; float f; std::memcpy(&f, &bits, 4); return f; }

}  // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int mpiRank = 0, mpiSize = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    setvbuf(stdout, nullptr, _IONBF, 0);

    ChainCfg c;
    c.R = (uint32_t)mpiSize;                          // rank_num = MPI 进程数（2 die）
    if (argc >= 5) { c.bs = atoi(argv[1]); c.topk = atoi(argv[2]); c.h = atoi(argv[3]); c.E = atoi(argv[4]); }
    if (argc >= 6) c.seed = atoi(argv[5]);
    const uint32_t rankId = (uint32_t)mpiRank;
    const int32_t  devId  = (int32_t)rankId;          // [T7] die=rank

#ifdef CHAIN_CCU_A6
    printf("[rank %u] build: CHAIN_CCU_A6=ON —— CCU 跨卡写已编入；bs=%u topk=%u h=%u E=%u R=%u\n",
           rankId, c.bs, c.topk, c.h, c.E, c.R);
#else
    printf("[rank %u] build: CHAIN_CCU_A6=OFF —— a5 shell（CCU 段桩掉，仅编译门禁）；bs=%u topk=%u h=%u E=%u R=%u\n",
           rankId, c.bs, c.topk, c.h, c.E, c.R);
#endif

    if (c.R < 2U) {
        printf("[rank %u][FATAL] rank_size=%u (<2)：CCU 容器需 rank>=2。请 mpirun -n 2。\n", rankId, c.R);
        MPI_Abort(MPI_COMM_WORLD, -1); return -1;
    }
    if (c.h % 32U != 0U) {
        printf("[rank %u][FATAL] h=%u 非 32 整除（mxfp8 每 32 元素一个 scale）。\n", rankId, c.h);
        MPI_Abort(MPI_COMM_WORLD, -1); return -1;
    }
    if (c.topk > c.GE()) {
        printf("[rank %u][FATAL] topk=%u > GE=%u（distinct 契约要求 topk<=GE）。\n", rankId, c.topk, c.GE());
        MPI_Abort(MPI_COMM_WORLD, -1); return -1;
    }

    ACLCHECK(aclInit(nullptr));
    ACLCHECK(aclrtSetDevice(devId));                  // 隐式默认 context（不 CreateContext，否则 107003）
    aclrtStream stream = nullptr;
    ACLCHECK(aclrtCreateStream(&stream));
    ACLCHECK(aclrtSetStreamFailureMode(stream, 1));

    // 2-rank CCU 扩展模式 HcclComm（创建 ccuContainer）
    HcclRootInfo rootInfo{};
    if (rankId == 0U) HCCLCHECK(HcclGetRootInfo(&rootInfo));
    if (MPI_Bcast(&rootInfo, HCCL_ROOT_INFO_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
        printf("[rank %u][ERR] MPI_Bcast(rootInfo) failed\n", rankId); MPI_Abort(MPI_COMM_WORLD, -1); return -1;
    }
    HcclCommConfig config{};
    HcclCommConfigInit(&config);
    config.hcclWorldRankID = rankId;
#ifdef CHAIN_CCU_A6
    config.hcclOpExpansionMode = CHAIN_CCU_EXPANSION_MODE;
#endif
    HcclComm comm = nullptr;
    HCCLCHECK(HcclCommInitRootInfoConfig(c.R, &rootInfo, rankId, &config, &comm));

    // ---- 分配 GM + host 预算 + 灌输入 ----
    ChainGm g;
    if (PrepChain(c, rankId, g) != 0) { MPI_Abort(MPI_COMM_WORLD, -1); return -1; }
    printf("[rank %u] PrepChain done（GM 分配 + host 预算 routeTable/wishCnt/assistTable + 灌输入）\n", rankId);

    // ---- 串 4 算子（每段后同步 + 跨 rank barrier 保 CCU 两卡对齐）----
    if (LaunchDispatchSetupCcu(comm, stream, c, g) != 0)   { MPI_Abort(MPI_COMM_WORLD, -1); return -1; }
    ACLCHECK(aclrtSynchronizeStream(stream)); MPI_Barrier(MPI_COMM_WORLD);
    if (LaunchDispatchTeardownAclnn(stream, c, g) != 0)    { MPI_Abort(MPI_COMM_WORLD, -1); return -1; }
    ACLCHECK(aclrtSynchronizeStream(stream)); MPI_Barrier(MPI_COMM_WORLD);
    if (LaunchCombineSetupCcu(comm, stream, c, g) != 0)    { MPI_Abort(MPI_COMM_WORLD, -1); return -1; }
    ACLCHECK(aclrtSynchronizeStream(stream)); MPI_Barrier(MPI_COMM_WORLD);
    if (LaunchCombineTeardownAclnn(stream, c, g) != 0)     { MPI_Abort(MPI_COMM_WORLD, -1); return -1; }
    ACLCHECK(aclrtSynchronizeStream(stream));

    // ---- golden：one-hot scale → x_out[i] = dequant(本卡 token i 的副本) = Fp8E4m3ToFloat(payload_i)*scale_i ----
    std::vector<uint16_t> xOutHost((size_t)c.bs * c.h, 0);
    ACLCHECK(aclrtMemcpy(xOutHost.data(), (uint64_t)c.bs * c.h * 2, g.xOut, (uint64_t)c.bs * c.h * 2, ACL_MEMCPY_DEVICE_TO_HOST));
#ifdef CHAIN_CCU_A6
    // ★ 逐 (i,k) 精确重建（审核 golden 项修）：x_out[i][k] = one-hot 加权 · dequant(payload[i][k], scale[i][k/32])
    //   = Fp8E4m3ToFloat(hXData[i*h+k]) * E8m0ToScale(hXScale[i*(h/32)+k/32])。payload/scale 都随 (i,k)/token 变，
    //   故 token 置换 / 元素乱序 / scale 读错行 都会失配（不再是"整 token 同值"的弱 golden）。
    double maxAbs = 0.0, maxRel = 0.0; uint32_t bad = 0;
    const uint64_t sB = c.h / 32U;
    for (uint32_t i = 0; i < c.bs; ++i) {
        for (uint32_t k = 0; k < c.h; ++k) {
            float pay = Fp8E4m3ToFloat(g.hXData[(size_t)i * c.h + k]);
            float scl = E8m0ToScale(g.hXScale[(size_t)i * sB + k / 32U]);
            float golden = pay * scl;
            float got = Bf16ToFloat(xOutHost[(size_t)i * c.h + k]);
            double a = std::fabs(got - golden);
            double r = golden != 0.0f ? a / std::fabs(golden) : a;
            maxAbs = std::max(maxAbs, a); maxRel = std::max(maxRel, r);
            if (a > 5e-3 && r > 5e-3) ++bad;            // bf16 容差 5e-3/5e-3（ops-precision-standard）
        }
    }
    printf("[rank %u] %s: 链端到端 golden  maxAbs=%.4g maxRel=%.4g bad=%u/%llu (bf16 容差 5e-3)\n",
           rankId, bad == 0 ? "PASS" : "FAIL", maxAbs, maxRel, bad, (unsigned long long)c.bs * c.h);
#else
    printf("[rank %u] (a5 shell) 4 段驱动结构编译/链接通过；CCU 段桩掉未执行 → golden 不适用（a6 上 -DCHAIN_CCU_A6=ON 跑真链）\n", rankId);
    (void)xOutHost;
#endif

    // ---- 清理 ----
    for (void *p : { g.xData, g.xScale, g.routeTable, g.oneBuf, g.dataRegion, g.scaleRegion, g.commCmdInfoD,
                     g.wishCnt, g.expandX, g.expandScale, g.sendCounts, g.expandIdx, g.expertTokenNums,
                     g.assistTable, g.commCmdInfoC, g.combineData, g.combineScale, g.expertScales, g.expertIds, g.xOut })
        if (p) aclrtFree(p);
    HcclCommDestroy(comm);
    aclrtDestroyStream(stream);
    aclrtResetDevice(devId);
    aclFinalize();
    MPI_Finalize();
    return 0;
}
