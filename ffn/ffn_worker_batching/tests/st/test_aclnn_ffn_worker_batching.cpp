/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * FfnWorkerBatching ST 测试 (C++ 原生，新增 Ascend950/A5 支持，范围对齐 A2)
 * ---------------------------------------------------------------------------
 * 算子类别：非计算类（Sort 排序 + IndexGather 索引 gather + Scan 分组计数），
 *          无浮点算术，输出为整数索引或原样字节搬运。
 * 精度口径：二进制一致（Bitwise Match, np.array_equal, rtol=atol=0）。
 *          取自 spec.yaml numerical_tolerance.per_dtype（全 dtype bitwise_equal）。
 *
 * 本文件包含（迭代二：多 shape + NORM/RECV 双路径 L0+L1）：
 *   1. byte-accurate CPU 参考模型（NORM need_schedule=0 / RECV need_schedule=1，token_dtype 0/1/2）：
 *        sort(生成 gather 索引) -> 多核 gather(原样字节搬运) -> 单核扫描 group_listing
 *      - NORM 语义与 op_kernel/ffn_worker_batching.cpp(FfnWorkerBatchingProcess) 一致：
 *        session/micro_batch 经 sessionIdsBuf/microBatchIdsBuf 查表。
 *      - RECV 语义与 FfnWorkerBatchingRecvProcess + ffn_wb_gather_out_all.h(isScanFlag=true) 一致：
 *        仅扫描单个 curMicroBatchID 的 token_info，session=aI 恒等、micro=curMicroBatchID 常量。
 *   2. CPU Golden 自测：手工可核对小用例 + 不变式(invariant)机器断言 + RECV 专属不变式。
 *   3. Mock 模式（-DUSE_MOCK_ACLNN）：CPU golden 自测 + L0/L1 CSV + 内置多 shape 用例驱动，无需 NPU。
 *   4. Real 模式：构造 1024B schedule_context + device token 缓冲（NORM: session/micro/expert buf；
 *      RECV: token_info buf @384 + pollIdx @416），调用 aclnn 两段式接口，D2H 后与参考模型做
 *      二进制一致比对（真实 NPU 精度验收，需安装 A5 算子包）。
 *
 * schedule_context(1024B) 字节布局来源 op_kernel/ffn_wb_get_schedule_context.h：
 *   [0]=session_num(A,u32) [4]=micro_batch_num(M,u32) [8]=micro_batch_size(BS,u32)
 *   [12]=selected_expert_num(K=topK+1,u32) [20]=attn_to_ffn_token_size(HS bytes,u32)
 *   非扫描路径: [400]=token_data_buf(u64) [528]=session_ids_buf(u64)
 *              [544]=micro_batch_ids_buf(u64) [560]=expert_ids_buf(u64) [576]=out_num(u32)
 */

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>

#ifndef USE_MOCK_ACLNN
#include "acl/acl.h"
#include "aclnn_ffn_worker_batching.h"
#endif

#define LOG_PRINT(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

// ---------------------------------------------------------------------------
// 常量（与 op_kernel/op_host 对齐）
// ---------------------------------------------------------------------------
static constexpr int64_t EXPERT_MASK_SENTINEL = 1000000; // sort_base.h expertStart_，>= 视为无效 token
static constexpr int64_t BLOCK_BYTES = 32;               // ffn_wb_common.h：int8 动态量化尾部 scale 块
static constexpr int64_t CTX_LEN = 1024;                 // schedule_context 字节长度

// ---------------------------------------------------------------------------
// 用例配置
// ---------------------------------------------------------------------------
struct FwbCase {
    std::string name;
    int64_t A = 0;  // max_out_shape[0]，attention session 数
    int64_t BS = 0; // max_out_shape[1]，micro batch size
    int64_t K = 0;  // max_out_shape[2]，topK+1
    int64_t H = 0;  // max_out_shape[3]，hidden size（元素个数）
    int64_t expertNum = 0;
    int64_t tokenDtype = 0; // 0=fp16 / 1=bf16 / 2=int8(dynamic quant)
    int64_t needSchedule = 0;
    int64_t layerNum = 0;
    int64_t M = 2;               // micro_batch_num（<=64，不影响输出 shape，测试自选）
    int64_t outNum = -1;         // 有效 attention session 数（默认=A）；仅 NORM 路径使用
    int64_t curMicroBatchID = 0; // RECV(need_schedule=1) 扫描的 micro batch（须 < M）
    int64_t maskStride = 0;      // >0 时 p%maskStride==maskStride-1 的 token 被 mask；0=全有效
    bool allSameExpert = false; // true 时全部有效 token 路由到同一专家（all_same_index / same_bin 极端访存）
    unsigned seed = 12345;

    int64_t GetOutNum() const
    {
        return outNum < 0 ? A : outNum;
    }
    int64_t ElemSize() const
    {
        return tokenDtype == 2 ? 1 : 2;
    } // fp16/bf16=2B, int8=1B
    int64_t PayloadBytes() const
    {
        return H * ElemSize();
    }
    int64_t HS() const
    {
        return tokenDtype == 2 ? (H + BLOCK_BYTES) : (H * ElemSize());
    }
    int64_t Y() const
    {
        return A * BS * K;
    }
    int64_t TokenSlots() const
    {
        return A * M * BS * K;
    } // token_data 逻辑 slot 数
    // 参考模型待排序 token 数：
    //   NORM(need_schedule=0)：expert_ids 索引空间 = [outNum, BS, K]（session 经查表映射）
    //   RECV(need_schedule=1)：仅扫描单个 micro batch，索引空间 = [A, BS, K]（session=aI 恒等，micro=curMicroBatchID）
    int64_t NumTokens() const
    {
        return needSchedule == 0 ? GetOutNum() * BS * K : A * BS * K;
    }
    // token_data 逻辑源 slot（[A,M,BS,K] 行优先，stride 单位=HS 字节）
    // NORM：session/micro 由查表得到；RECV：session=aI 恒等、micro=curMicroBatchID 常量
    int64_t SrcSlot(int64_t sess, int64_t mb, int64_t bsI, int64_t kI) const
    {
        return (((sess * M + mb) * BS + bsI) * K + kI);
    }
};

// ---------------------------------------------------------------------------
// 参考模型：确定性合成输入
// ---------------------------------------------------------------------------
// 每个 token_data 字节位置的确定性填充值：纯 mixing hash（与 gather 源位置强绑定）
static inline uint8_t ByteVal(uint64_t globalBytePos)
{
    uint64_t x = globalBytePos * 1103515245ULL + 12345ULL;
    return static_cast<uint8_t>((x >> 16) & 0xFF);
}

struct FwbInputs {
    std::vector<int32_t> expertIds;        // 长度 outNum*BS*K
    std::vector<int32_t> sessionIdsBuf;    // 长度 A，compact aIndices -> 真实 session
    std::vector<int32_t> microBatchIdsBuf; // 长度 A，compact aIndices -> micro batch
    std::vector<uint8_t> tokenData;        // 长度 TokenSlots*HS 字节
};

static FwbInputs GenerateInputs(const FwbCase &c)
{
    FwbInputs in;
    const int64_t nTok = c.NumTokens(); // NORM: outNum*BS*K；RECV: A*BS*K（单 micro batch）

    in.expertIds.resize(nTok);
    for (int64_t p = 0; p < nTok; ++p) {
        bool masked = (c.maskStride > 0) && (p % c.maskStride == c.maskStride - 1);
        // 有效 token：默认 round-robin 分布到 [0, expertNum) 保证 group_list 非平凡；
        //           allSameExpert=true 时全部落同一专家（all_same_index / same_bin 极端访存）
        int32_t eid = c.allSameExpert ? 0 : static_cast<int32_t>(p % c.expertNum);
        in.expertIds[p] = masked ? static_cast<int32_t>(EXPERT_MASK_SENTINEL) : eid;
    }

    in.sessionIdsBuf.resize(c.A);
    in.microBatchIdsBuf.resize(c.A);
    for (int64_t a = 0; a < c.A; ++a) {
        in.sessionIdsBuf[a] = static_cast<int32_t>(a); // identity（合法映射到 token_data 的 A 维）
        in.microBatchIdsBuf[a] = static_cast<int32_t>(a % c.M);
    }

    const int64_t bytes = c.TokenSlots() * c.HS();
    in.tokenData.resize(bytes);
    for (int64_t i = 0; i < bytes; ++i) {
        in.tokenData[i] = ByteVal(static_cast<uint64_t>(i));
    }
    return in;
}

// ---------------------------------------------------------------------------
// id 解码：从待排序 token 序号 p 还原 (session, micro_batch, token_id, expert_offset)
//   NORM(need_schedule=0)：aI=p/(BS*K)，session=sessionIdsBuf[aI]、micro=microBatchIdsBuf[aI]
//                          （对齐 ffn_wb_gather_out_all.h isScanFlag=false 查表分支）
//   RECV(need_schedule=1)：aI=p/(BS*K)，session=aI（恒等）、micro=curMicroBatchID（常量）
//                          （对齐 ffn_wb_gather_out_all.h isScanFlag=true 分支）
// ---------------------------------------------------------------------------
static inline void DecodeToken(const FwbCase &c, const FwbInputs &in, int64_t p, int32_t &sess, int32_t &mb,
                               int32_t &tokenId, int32_t &expOff)
{
    const int64_t bsk = c.BS * c.K;
    const int64_t aI = p / bsk;
    const int64_t rem = p - aI * bsk;
    const int64_t bsI = rem / c.K;
    const int64_t kI = rem - bsI * c.K;
    if (c.needSchedule == 0) {
        sess = in.sessionIdsBuf[aI];
        mb = in.microBatchIdsBuf[aI];
    } else {
        sess = static_cast<int32_t>(aI);
        mb = static_cast<int32_t>(c.curMicroBatchID);
    }
    tokenId = static_cast<int32_t>(bsI);
    expOff = static_cast<int32_t>(kI);
}

// ---------------------------------------------------------------------------
// 参考模型：golden 输出（NORM 路径，need_schedule=0）
// ---------------------------------------------------------------------------
struct FwbGolden {
    std::vector<uint8_t> y;             // Y*PayloadBytes 字节（仅 [0,validCnt) 有效）
    std::vector<int64_t> groupList;     // expertNum*2
    std::vector<int32_t> sessionIds;    // Y（仅 [0,validCnt) 有效）
    std::vector<int32_t> microBatchIds; // Y
    std::vector<int32_t> tokenIds;      // Y
    std::vector<int32_t> expertOffsets; // Y
    std::vector<float> dynamicScale;    // Y（仅 token_dtype=2 有效）
    int64_t actualTokenNum = 0;         // = validCnt

    std::vector<int64_t> gatherIdx;       // 排序后 gather 索引（长度 validCnt）
    std::vector<int32_t> sortedExpertIds; // 排序后专家 id（长度 validCnt）
    int64_t groupListRows = 0;            // 已填充的 group_list 行数（含末尾 [0,0]）
};

static FwbGolden ComputeGolden(const FwbCase &c, const FwbInputs &in)
{
    FwbGolden g;
    const int64_t payload = c.PayloadBytes();
    const int64_t HS = c.HS();

    // --- step 1: sort（稳定升序 by expert_id，无效 token(>=SENTINEL) 剔除） ---
    std::vector<int64_t> validPos;
    validPos.reserve(in.expertIds.size());
    for (int64_t p = 0; p < static_cast<int64_t>(in.expertIds.size()); ++p) {
        if (in.expertIds[p] < EXPERT_MASK_SENTINEL)
            validPos.push_back(p);
    }
    std::stable_sort(validPos.begin(), validPos.end(),
                     [&](int64_t a, int64_t b) { return in.expertIds[a] < in.expertIds[b]; });

    const int64_t validCnt = static_cast<int64_t>(validPos.size());
    g.actualTokenNum = validCnt;
    g.gatherIdx = validPos;
    g.sortedExpertIds.resize(validCnt);
    for (int64_t i = 0; i < validCnt; ++i)
        g.sortedExpertIds[i] = in.expertIds[validPos[i]];

    // --- 输出缓冲（全 Y 尺寸，尾部保持 0；比对仅覆盖 [0,validCnt)） ---
    g.y.assign(c.Y() * payload, 0);
    g.sessionIds.assign(c.Y(), 0);
    g.microBatchIds.assign(c.Y(), 0);
    g.tokenIds.assign(c.Y(), 0);
    g.expertOffsets.assign(c.Y(), 0);
    g.dynamicScale.assign(c.Y(), 0.0f);

    // --- step 2: gather（原样字节搬运 + 查表 id/scale；NORM/RECV 统一走 DecodeToken） ---
    for (int64_t o = 0; o < validCnt; ++o) {
        const int64_t p = validPos[o];
        int32_t sess, mb, tokenId, expOff;
        DecodeToken(c, in, p, sess, mb, tokenId, expOff);

        g.sessionIds[o] = sess;
        g.microBatchIds[o] = mb;
        g.tokenIds[o] = tokenId;
        g.expertOffsets[o] = expOff;

        // token_data 逻辑源 slot：[A,M,BS,K] 行优先（stride 单位=HS 字节）
        const int64_t srcByte = c.SrcSlot(sess, mb, tokenId, expOff) * HS;
        std::memcpy(&g.y[o * payload], &in.tokenData[srcByte], payload);
        if (c.tokenDtype == 2) {
            float sc;
            std::memcpy(&sc, &in.tokenData[srcByte + c.H], sizeof(float)); // 尾部 32B 块首 4B
            g.dynamicScale[o] = sc;
        }
    }

    // --- step 3: group_listing（升序打包 (expert_id, count) + 末尾 [0,0]） ---
    g.groupList.assign(c.expertNum * 2, 0);
    int64_t row = 0;
    int64_t i = 0;
    while (i < validCnt && row < c.expertNum) {
        int32_t eid = g.sortedExpertIds[i];
        int64_t cnt = 0;
        while (i < validCnt && g.sortedExpertIds[i] == eid) {
            ++cnt;
            ++i;
        }
        g.groupList[row * 2] = eid;
        g.groupList[row * 2 + 1] = cnt;
        ++row;
    }
    if (row < c.expertNum) { // 末尾补 [0,0]（kernel 语义）
        g.groupList[row * 2] = 0;
        g.groupList[row * 2 + 1] = 0;
        ++row;
    }
    g.groupListRows = row;
    return g;
}

// ---------------------------------------------------------------------------
// 比对工具（二进制一致）
// ---------------------------------------------------------------------------
template <typename T>
static bool BitwiseEqual(const T *golden, const T *actual, size_t n, const char *tag)
{
    for (size_t i = 0; i < n; ++i) {
        if (golden[i] != actual[i]) {
            LOG_PRINT("  [FAIL] %s 不匹配 [%zu]: golden=%lld actual=%lld", tag, i, static_cast<long long>(golden[i]),
                      static_cast<long long>(actual[i]));
            return false;
        }
    }
    LOG_PRINT("  [PASS] %s 二进制一致 (%zu 元素)", tag, n);
    return true;
}

static bool ByteEqual(const uint8_t *golden, const uint8_t *actual, size_t n, const char *tag)
{
    for (size_t i = 0; i < n; ++i) {
        if (golden[i] != actual[i]) {
            LOG_PRINT("  [FAIL] %s 字节不匹配 [%zu]: golden=%u actual=%u", tag, i, golden[i], actual[i]);
            return false;
        }
    }
    LOG_PRINT("  [PASS] %s 二进制一致 (%zu 字节)", tag, n);
    return true;
}

// ---------------------------------------------------------------------------
// 桶内集合相等比对（BucketSetEqual）
// ---------------------------------------------------------------------------
// 语义依据：生产多核 sort(KernelSortMaskMultiCore = A2 逐字节)对同专家 token 走
// "每核 Sort + MrgSort 归并"，等专家键的 token 桶内顺序按归并序，不保证全局行号稳定序。
// 而 golden 用 std::stable_sort（行号序）。两者桶内是【同一集合】仅顺序不同——单核路径
// (Sort 全局稳定)恰好匹配 golden，多核归并序不匹配。逐位置比会误报。
// 正确口径（对齐 A2 实际契约 + spec：专家分桶正确 + 桶内集合正确 + 计数正确，桶内序不约束）：
//   按 sortedExpertIds 划分专家桶（npu/golden 专家序与逐专家计数已由 kernel 保证一致，
//   否则先在此暴露），桶内把每 token 的 (sid,mbid,tid,eoff[,scale]) + y payload 打包成
//   一条记录，规范排序后逐条比。任一路（含 y）在某桶内集合不一致即 FAIL。
// 注：sortedExpertIds 的 npu 侧不直接输出，用 golden 的专家分桶边界（专家序两边一致，
//     由 group_list 已比对保证）切分 npu 输出的相同 [0,vc) 区间。
static bool BucketSetEqual(const FwbGolden &g, const int32_t *nSid, const int32_t *nMbid, const int32_t *nTid,
                           const int32_t *nEoff, const uint8_t *nY, const float *nScale, int64_t payload, bool hasScale)
{
    const int64_t vc = g.actualTokenNum;
    // 每条 token 记录：绑定 5 路 id/scale + y payload，作为不可分整体
    struct Rec {
        int32_t sid, mbid, tid, eoff;
        float scale;
        const uint8_t *y; // 指向该 token 的 payload 字节
        int64_t payload;
        bool hasScale;
        // scale 按 bit 位（字节）比较，而非浮点 ==：dynamic_quant 随机输入常使 scale 字节凑成
        // NaN，此时 golden/npu 位模式相同（同一份输入）但 float== 恒 false 会误报。二进制一致
        // 契约下正确口径就是 bit-exact 比较。
        static uint32_t ScaleBits(float f)
        {
            uint32_t u;
            std::memcpy(&u, &f, sizeof(u));
            return u;
        }
        // 规范序：先按 (tid,eoff,sid,mbid) 排（这四者由原始位置 p 唯一决定，是稳定的桶内键），
        // 再按 y 字节兜底，保证同集合两侧规范序一致。
        bool operator<(const Rec &o) const
        {
            if (tid != o.tid)
                return tid < o.tid;
            if (eoff != o.eoff)
                return eoff < o.eoff;
            if (sid != o.sid)
                return sid < o.sid;
            if (mbid != o.mbid)
                return mbid < o.mbid;
            int c = std::memcmp(y, o.y, (size_t)payload);
            if (c != 0)
                return c < 0;
            if (hasScale && ScaleBits(scale) != ScaleBits(o.scale))
                return ScaleBits(scale) < ScaleBits(o.scale);
            return false;
        }
        bool operator==(const Rec &o) const
        {
            if (sid != o.sid || mbid != o.mbid || tid != o.tid || eoff != o.eoff)
                return false;
            if (std::memcmp(y, o.y, (size_t)payload) != 0)
                return false;
            if (hasScale && ScaleBits(scale) != ScaleBits(o.scale))
                return false;
            return true;
        }
    };
    // 按 golden 专家桶边界切分（sortedExpertIds 升序分段）
    int64_t i = 0;
    while (i < vc) {
        int32_t eid = g.sortedExpertIds[i];
        int64_t j = i;
        while (j < vc && g.sortedExpertIds[j] == eid)
            ++j; // [i,j) 为专家 eid 的桶
        std::vector<Rec> gb, nb;
        gb.reserve(j - i);
        nb.reserve(j - i);
        for (int64_t k = i; k < j; ++k) {
            gb.push_back({g.sessionIds[k], g.microBatchIds[k], g.tokenIds[k], g.expertOffsets[k],
                          hasScale ? g.dynamicScale[k] : 0.0f, &g.y[k * payload], payload, hasScale});
            nb.push_back({nSid[k], nMbid[k], nTid[k], nEoff[k], hasScale ? nScale[k] : 0.0f, &nY[k * payload], payload,
                          hasScale});
        }
        std::sort(gb.begin(), gb.end());
        std::sort(nb.begin(), nb.end());
        for (int64_t k = 0; k < (int64_t)gb.size(); ++k) {
            if (!(gb[k] == nb[k])) {
                LOG_PRINT("  [FAIL] 专家桶 eid=%d 内 token 集合不一致 (桶大小=%lld，第 %lld 条规范序记录不符)", eid,
                          (long long)gb.size(), (long long)k);
#ifdef FWB_DIAG
                // 临时诊断：打印不符条两侧 id/scale + y 前 16 字节，定位是 y 错还是 scale/id 错
                const Rec &G = gb[k];
                const Rec &N = nb[k];
                LOG_PRINT("    [DIAG] golden: sid=%d mbid=%d tid=%d eoff=%d scale=%.6f", G.sid, G.mbid, G.tid, G.eoff,
                          G.scale);
                LOG_PRINT("    [DIAG] npu   : sid=%d mbid=%d tid=%d eoff=%d scale=%.6f", N.sid, N.mbid, N.tid, N.eoff,
                          N.scale);
                bool idEq = (G.sid == N.sid && G.mbid == N.mbid && G.tid == N.tid && G.eoff == N.eoff);
                bool yEq = (std::memcmp(G.y, N.y, (size_t)payload) == 0);
                bool scEq = (!hasScale) || (G.scale == N.scale);
                LOG_PRINT("    [DIAG] idEq=%d yEq=%d scaleEq=%d payload=%lld", idEq, yEq, scEq, (long long)payload);
                {
                    char gb_hex[64] = {0}, nb_hex[64] = {0};
                    int nb2 = (int)std::min<int64_t>(payload, 16);
                    for (int t = 0; t < nb2; ++t) {
                        sprintf(gb_hex + t * 3, "%02x ", G.y[t]);
                        sprintf(nb_hex + t * 3, "%02x ", N.y[t]);
                    }
                    LOG_PRINT("    [DIAG] golden.y[0:%d]= %s", nb2, gb_hex);
                    LOG_PRINT("    [DIAG] npu.y[0:%d]   = %s", nb2, nb_hex);
                }
#endif
                return false;
            }
        }
        i = j;
    }
    LOG_PRINT("  [PASS] 桶内集合一致 (%lld token, 按专家分桶规范序比对 y+ids%s)", (long long)vc,
              hasScale ? "+scale" : "");
    return true;
}

// ---------------------------------------------------------------------------
// CPU Golden 自测：不变式（invariant）机器断言
// ---------------------------------------------------------------------------
// 校验参考模型自身语义正确性（与 spec.invariants / boundary 对齐）：
//   - gather 索引是有效 token 位置的排列；排序后专家 id 升序
//   - group_list 计数守恒 (sum == validCnt)、专家 id 严格升序
//   - id 输出与 gather 解码一致
//   - y 字节确实取自正确源 slot（byte-accurate）
//   - determinism：重复计算二进制一致
static bool CheckInvariants(const FwbCase &c, const FwbInputs &in, const FwbGolden &g)
{
    const int64_t payload = c.PayloadBytes();
    const int64_t HS = c.HS();
    bool ok = true;

    // shape 契约
    if (static_cast<int64_t>(g.groupList.size()) != c.expertNum * 2) {
        LOG_PRINT("  [FAIL] group_list shape != [expert_num,2]");
        ok = false;
    }
    if (static_cast<int64_t>(g.sessionIds.size()) != c.Y()) {
        LOG_PRINT("  [FAIL] session_ids shape != [Y]");
        ok = false;
    }

    // gatherIdx 是有效位置排列，排序升序
    {
        std::vector<int64_t> sorted = g.gatherIdx;
        std::sort(sorted.begin(), sorted.end());
        bool dup = false;
        for (size_t i = 1; i < sorted.size(); ++i)
            if (sorted[i] == sorted[i - 1])
                dup = true;
        if (dup) {
            LOG_PRINT("  [FAIL] gather 索引存在重复");
            ok = false;
        }
        for (int64_t p : g.gatherIdx)
            if (in.expertIds[p] >= EXPERT_MASK_SENTINEL) {
                LOG_PRINT("  [FAIL] gather 命中无效 token");
                ok = false;
                break;
            }
        for (size_t i = 1; i < g.sortedExpertIds.size(); ++i)
            if (g.sortedExpertIds[i] < g.sortedExpertIds[i - 1]) {
                LOG_PRINT("  [FAIL] 排序非升序");
                ok = false;
                break;
            }
    }

    // group_list 计数守恒 + 专家 id 升序
    {
        int64_t sum = 0;
        int64_t lastEid = -1;
        // 有效行 = groupListRows - (末尾是否补了[0,0])；这里直接统计 count>0 的行
        for (int64_t r = 0; r < c.expertNum; ++r) {
            int64_t eid = g.groupList[r * 2];
            int64_t cnt = g.groupList[r * 2 + 1];
            if (cnt == 0)
                break; // 到达 [0,0] 终止或未使用行
            sum += cnt;
            if (eid <= lastEid) {
                LOG_PRINT("  [FAIL] group_list 专家 id 非严格升序");
                ok = false;
            }
            lastEid = eid;
        }
        if (sum != g.actualTokenNum) {
            LOG_PRINT("  [FAIL] group_list 计数和 %lld != actual_token_num %lld", (long long)sum,
                      (long long)g.actualTokenNum);
            ok = false;
        }
    }

    // id 解码一致性 + y byte-accurate 源校验（NORM/RECV 统一走 DecodeToken）
    for (int64_t o = 0; o < g.actualTokenNum; ++o) {
        const int64_t p = g.gatherIdx[o];
        int32_t sess, mb, tokenId, expOff;
        DecodeToken(c, in, p, sess, mb, tokenId, expOff);
        if (g.sessionIds[o] != sess || g.microBatchIds[o] != mb || g.tokenIds[o] != tokenId ||
            g.expertOffsets[o] != expOff) {
            LOG_PRINT("  [FAIL] out[%lld] id 解码不一致", (long long)o);
            ok = false;
            break;
        }
        const int64_t srcByte = c.SrcSlot(sess, mb, tokenId, expOff) * HS;
        for (int64_t j = 0; j < payload; ++j) {
            if (g.y[o * payload + j] != ByteVal(static_cast<uint64_t>(srcByte + j))) {
                LOG_PRINT("  [FAIL] out[%lld] y 未取自正确源 slot", (long long)o);
                ok = false;
                break;
            }
        }
        if (!ok)
            break;
        if (c.tokenDtype == 2) {
            // dynamic_scale 为原样字节 gather，按 bitwise 比对（禁用 float !=，避免 NaN 位模式误判）
            const int64_t sb = srcByte + c.H;
            uint8_t tmp[4];
            for (int k = 0; k < 4; ++k)
                tmp[k] = ByteVal(static_cast<uint64_t>(sb + k));
            if (std::memcmp(&g.dynamicScale[o], tmp, 4) != 0) {
                LOG_PRINT("  [FAIL] out[%lld] dynamic_scale 源不一致", (long long)o);
                ok = false;
                break;
            }
        }
    }

    // determinism：重复计算二进制一致（dynamic_scale 用字节比对，避免 NaN 位模式误判）
    {
        FwbGolden g2 = ComputeGolden(c, in);
        bool scaleSame = (g2.dynamicScale.size() == g.dynamicScale.size()) &&
                         (g.dynamicScale.empty() || std::memcmp(g2.dynamicScale.data(), g.dynamicScale.data(),
                                                                g.dynamicScale.size() * sizeof(float)) == 0);
        if (g2.y != g.y || g2.groupList != g.groupList || g2.sessionIds != g.sessionIds ||
            g2.microBatchIds != g.microBatchIds || g2.tokenIds != g.tokenIds || g2.expertOffsets != g.expertOffsets ||
            !scaleSame || g2.actualTokenNum != g.actualTokenNum) {
            LOG_PRINT("  [FAIL] determinism：重复计算结果不一致");
            ok = false;
        }
    }
    return ok;
}

// ---------------------------------------------------------------------------
// 手工可核对小用例（golden 正确性硬证据）
// ---------------------------------------------------------------------------
static bool TestHandVerifiedCase()
{
    LOG_PRINT("\n[自测-1] 手工可核对小用例 (A=2,M=1,BS=2,K=2,H=4,expert_num=4,fp16,need_schedule=0)");
    FwbCase c;
    c.name = "hand_verified_fp16";
    c.A = 2;
    c.M = 1;
    c.BS = 2;
    c.K = 2;
    c.H = 4;
    c.expertNum = 4;
    c.tokenDtype = 0;
    c.needSchedule = 0;
    c.outNum = 2;
    c.maskStride = 0;

    FwbInputs in = GenerateInputs(c);
    // 覆写 expertIds 为手工设计值（p: 0..7），SENTINEL 表示被 mask
    // p:  0    1    2      3    4    5    6      7
    in.expertIds = {2, 0, (int32_t)EXPERT_MASK_SENTINEL, 1, 0, 3, 1, (int32_t)EXPERT_MASK_SENTINEL};
    in.sessionIdsBuf = {0, 1};
    in.microBatchIdsBuf = {0, 0};

    FwbGolden g = ComputeGolden(c, in);
    bool ok = true;

    // 期望：有效 p = {0,1,3,4,5,6}；稳定升序 by expert_id →
    //   e0: p1,p4 ; e1: p3,p6 ; e2: p0 ; e3: p5
    std::vector<int64_t> expGather = {1, 4, 3, 6, 0, 5};
    std::vector<int32_t> expSorted = {0, 0, 1, 1, 2, 3};
    std::vector<int32_t> expSession = {0, 1, 0, 1, 0, 1}; // aI: 0,1,0,1,0,1 → sessBuf identity
    std::vector<int32_t> expMicro = {0, 0, 0, 0, 0, 0};
    std::vector<int32_t> expTokenId = {0, 0, 1, 1, 0, 0};     // bsI
    std::vector<int32_t> expExpOff = {1, 0, 1, 0, 0, 1};      // kI
    std::vector<int64_t> expGroup = {0, 2, 1, 2, 2, 1, 3, 1}; // 4 专家全用，无 [0,0] 余量

    // id 输出为全 Y 长度（尾部补 0），比对仅覆盖有效区 [0, actual_token_num)
    const int64_t vc = g.actualTokenNum;
    std::vector<int32_t> sid(g.sessionIds.begin(), g.sessionIds.begin() + vc);
    std::vector<int32_t> mic(g.microBatchIds.begin(), g.microBatchIds.begin() + vc);
    std::vector<int32_t> tid(g.tokenIds.begin(), g.tokenIds.begin() + vc);
    std::vector<int32_t> eof(g.expertOffsets.begin(), g.expertOffsets.begin() + vc);

    if (g.actualTokenNum != 6) {
        LOG_PRINT("  [FAIL] actual_token_num=%lld != 6", (long long)g.actualTokenNum);
        ok = false;
    }
    if (g.gatherIdx != expGather) {
        LOG_PRINT("  [FAIL] gatherIdx 不符");
        ok = false;
    }
    if (g.sortedExpertIds != expSorted) {
        LOG_PRINT("  [FAIL] sortedExpertIds 不符");
        ok = false;
    }
    if (sid != expSession) {
        LOG_PRINT("  [FAIL] session_ids 不符");
        ok = false;
    }
    if (mic != expMicro) {
        LOG_PRINT("  [FAIL] micro_batch_ids 不符");
        ok = false;
    }
    if (tid != expTokenId) {
        LOG_PRINT("  [FAIL] token_ids 不符");
        ok = false;
    }
    if (eof != expExpOff) {
        LOG_PRINT("  [FAIL] expert_offsets 不符");
        ok = false;
    }
    if (g.groupList != expGroup) {
        LOG_PRINT("  [FAIL] group_list 不符，实际:");
        for (size_t i = 0; i < g.groupList.size(); i += 2)
            LOG_PRINT("        [%lld,%lld]", (long long)g.groupList[i], (long long)g.groupList[i + 1]);
        ok = false;
    }
    if (!CheckInvariants(c, in, g))
        ok = false;
    LOG_PRINT("  结果: %s", ok ? "PASS" : "FAIL");
    return ok;
}

static bool TestKZeroCase()
{
    LOG_PRINT("\n[自测-2] k=0 边界：全部 token 被 mask（actual_token_num=0，group_list 首行 [0,0]）");
    FwbCase c;
    c.name = "boundary_k_zero";
    c.A = 2;
    c.M = 1;
    c.BS = 2;
    c.K = 2;
    c.H = 4;
    c.expertNum = 4;
    c.tokenDtype = 0;
    c.needSchedule = 0;
    c.outNum = 2;
    c.maskStride = 1; // 全 mask
    FwbInputs in = GenerateInputs(c);
    FwbGolden g = ComputeGolden(c, in);
    bool ok = true;
    if (g.actualTokenNum != 0) {
        LOG_PRINT("  [FAIL] actual_token_num != 0");
        ok = false;
    }
    if (g.groupList[0] != 0 || g.groupList[1] != 0) {
        LOG_PRINT("  [FAIL] group_list 首行非 [0,0]");
        ok = false;
    }
    LOG_PRINT("  结果: %s", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// L0 CSV 解析（读 tests/st/testcases/aclnnFfnWorkerBatching_l0_test_cases.csv）
// ---------------------------------------------------------------------------
static bool ParseIntAfter(const std::string &s, const std::string &key, int64_t &out)
{
    size_t k = s.find(key);
    if (k == std::string::npos)
        return false;
    size_t colon = s.find(':', k);
    if (colon == std::string::npos)
        return false;
    size_t i = colon + 1;
    while (i < s.size() && (s[i] == ' ' || s[i] == '"'))
        ++i;
    bool neg = false;
    if (i < s.size() && s[i] == '-') {
        neg = true;
        ++i;
    }
    if (i >= s.size() || !isdigit((unsigned char)s[i]))
        return false;
    int64_t v = 0;
    while (i < s.size() && isdigit((unsigned char)s[i])) {
        v = v * 10 + (s[i] - '0');
        ++i;
    }
    out = neg ? -v : v;
    return true;
}

static bool ParseMaxOutShape(const std::string &s, int64_t &A, int64_t &BS, int64_t &K, int64_t &H)
{
    size_t k = s.find("max_out_shape");
    if (k == std::string::npos)
        return false;
    size_t lb = s.find('[', k);
    size_t rb = s.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos)
        return false;
    std::string body = s.substr(lb + 1, rb - lb - 1);
    std::vector<int64_t> vals;
    std::stringstream ss(body);
    std::string tok;
    while (std::getline(ss, tok, ','))
        vals.push_back(std::stoll(tok));
    if (vals.size() != 4)
        return false;
    A = vals[0];
    BS = vals[1];
    K = vals[2];
    H = vals[3];
    return true;
}

// 通用 CSV 解析（L0 / L1 同格式）。NORM/RECV 双路径均由 need_schedule 列驱动。
// mem_cap_slots：token_data 逻辑 slot 上限（A*M*BS*K）；超过则将 M 降到 1 以约束 Mock 内存/耗时，
//               对输出语义与二进制一致无影响（session/micro 映射仍自洽，仅 micro 多样性收敛）。
static std::vector<FwbCase> ParseCsvCases(const std::string &path, int64_t memCapSlots = (64LL << 20))
{
    std::vector<FwbCase> cases;
    std::ifstream f(path);
    if (!f.is_open()) {
        LOG_PRINT("  [WARN] 未能打开 CSV: %s（跳过 CSV 驱动用例）", path.c_str());
        return cases;
    }
    std::string line;
    bool header = true;
    while (std::getline(f, line)) {
        if (header) {
            header = false;
            continue;
        }
        if (line.empty())
            continue;
        FwbCase c;
        c.name = line.substr(0, line.find(','));
        if (!ParseMaxOutShape(line, c.A, c.BS, c.K, c.H))
            continue;
        ParseIntAfter(line, "expert_num", c.expertNum);
        ParseIntAfter(line, "token_dtype", c.tokenDtype);
        ParseIntAfter(line, "need_schedule", c.needSchedule);
        ParseIntAfter(line, "layer_num", c.layerNum);
        // 名字含 k_zero → 全 mask；含 k_eq_n / full_valid / same_bin / all_same → 全有效；其余稀疏。
        if (c.name.find("k_zero") != std::string::npos)
            c.maskStride = 1; // 全 mask
        else if (c.name.find("k_eq_n") != std::string::npos || c.name.find("full_valid") != std::string::npos ||
                 c.name.find("same_bin") != std::string::npos || c.name.find("all_same") != std::string::npos)
            c.maskStride = 0; // 全有效
        else
            c.maskStride = 5; // 约 20% mask
        // all_same_index / same_bin：全部有效 token 落同一专家（极端访存 all_same_index）
        if (c.name.find("all_same") != std::string::npos || c.name.find("same_bin") != std::string::npos)
            c.allSameExpert = true;
        c.M = 2;
        // 大 shape 下降 M 以约束 token_data 物化内存（不改变 shape 契约与二进制一致口径）
        if (c.A * c.M * c.BS * c.K > memCapSlots)
            c.M = 1;
        c.outNum = c.A;
        c.curMicroBatchID = 0; // RECV 扫描 micro batch 0（须 < M）
        cases.push_back(c);
    }
    return cases;
}

// ---------------------------------------------------------------------------
// Mock 模式：单用例 = 计算 golden + 不变式断言
// ---------------------------------------------------------------------------
static const char *DtypeName(int64_t td)
{
    return td == 0 ? "fp16" : (td == 1 ? "bf16" : "int8");
}

// RECV(need_schedule=1) 路径专属不变式：
//   - session_ids[o] == aI（gather 恒等，无 sessionIdsBuf 查表）
//   - micro_batch_ids[o] == curMicroBatchID（扫描单一 micro batch，恒定）
// 对齐 op_kernel/ffn_wb_gather_out_all.h isScanFlag=true 分支语义。
static bool CheckRecvInvariants(const FwbCase &c, const FwbGolden &g)
{
    bool ok = true;
    const int64_t bsk = c.BS * c.K;
    for (int64_t o = 0; o < g.actualTokenNum; ++o) {
        const int64_t p = g.gatherIdx[o];
        const int64_t aI = p / bsk;
        if (g.sessionIds[o] != static_cast<int32_t>(aI)) {
            LOG_PRINT("  [FAIL] RECV out[%lld] session_ids=%d != aI=%lld (非恒等映射)", (long long)o, g.sessionIds[o],
                      (long long)aI);
            ok = false;
            break;
        }
        if (g.microBatchIds[o] != static_cast<int32_t>(c.curMicroBatchID)) {
            LOG_PRINT("  [FAIL] RECV out[%lld] micro_batch_ids=%d != curMicroBatchID=%lld", (long long)o,
                      g.microBatchIds[o], (long long)c.curMicroBatchID);
            ok = false;
            break;
        }
    }
    if (ok)
        LOG_PRINT("  [PASS] RECV 语义：session=aI 恒等 / micro=curMicroBatchID=%lld", (long long)c.curMicroBatchID);
    return ok;
}

// 精度驱动仅对「合法属性」用例造 golden 对拍;非法属性(越界)属负路径,归 test_negative(测错误码)覆盖。
// 此处防御性跳过,避免 host golden 造数(p % expertNum、assign(expertNum*2) 等)在非法输入上除 0/越界崩溃。
static bool GoldenComputable(const FwbCase &c)
{
    return c.expertNum > 0 && c.expertNum <= 8192 && c.A > 0 && c.A <= 1024 && c.BS > 0 && c.K > 0 &&
           c.K <= 64 && c.H > 0 && c.tokenDtype >= 0 && c.tokenDtype <= 2 && c.needSchedule >= 0 &&
           c.needSchedule <= 1;
}

static bool RunCaseMock(const FwbCase &c)
{
    if (!GoldenComputable(c)) {
        LOG_PRINT("\n[Mock] %s  [SKIP] 非法属性,精度驱动不造 golden(负路径归 test_negative)", c.name.c_str());
        return true;
    }
    LOG_PRINT(
        "\n[Mock] %s  (A=%lld BS=%lld K=%lld H=%lld expert_num=%lld token_dtype=%s need_schedule=%lld[%s] Y=%lld)",
        c.name.c_str(), (long long)c.A, (long long)c.BS, (long long)c.K, (long long)c.H, (long long)c.expertNum,
        DtypeName(c.tokenDtype), (long long)c.needSchedule, c.needSchedule == 0 ? "NORM/100" : "RECV/101",
        (long long)c.Y());
    FwbInputs in = GenerateInputs(c);
    FwbGolden g = ComputeGolden(c, in);
    LOG_PRINT("  待排序 token 数=%lld  actual_token_num=%lld / Y=%lld  group_list 有效行=%lld",
              (long long)c.NumTokens(), (long long)g.actualTokenNum, (long long)c.Y(), (long long)g.groupListRows);
    bool ok = CheckInvariants(c, in, g);
    if (c.needSchedule == 1)
        ok &= CheckRecvInvariants(c, g);
    LOG_PRINT("  结果: %s", ok ? "PASS" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Real 模式：构造 schedule_context + 调用 aclnn 两段式接口 + 二进制一致比对
// ---------------------------------------------------------------------------
#ifndef USE_MOCK_ACLNN

static void WriteU32(std::vector<int8_t> &buf, int64_t off, uint32_t v)
{
    std::memcpy(&buf[off], &v, 4);
}
static void WriteU64(std::vector<int8_t> &buf, int64_t off, uint64_t v)
{
    std::memcpy(&buf[off], &v, 8);
}

static aclDataType YAclDtype(int64_t td)
{
    if (td == 0)
        return ACL_FLOAT16;
    if (td == 1)
        return ACL_BF16;
    return ACL_INT8;
}

static aclTensor *CreateTensorRaw(void *devAddr, const std::vector<int64_t> &shape, aclDataType dt)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; --i)
        strides[i] = shape[i + 1] * strides[i + 1];
    return aclCreateTensor(shape.data(), shape.size(), dt, strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                           devAddr);
}

static bool RunCaseReal(const FwbCase &c, aclrtStream stream)
{
    LOG_PRINT("\n[Real] %s  (token_dtype=%s need_schedule=%lld[%s])", c.name.c_str(), DtypeName(c.tokenDtype),
              (long long)c.needSchedule, c.needSchedule == 0 ? "NORM/100" : "RECV/101");

    if (!GoldenComputable(c)) {
        LOG_PRINT("  [SKIP] 非法属性,精度驱动不造 golden(负路径归 test_negative)");
        return true;
    }

    FwbInputs in = GenerateInputs(c);
    FwbGolden g = ComputeGolden(c, in);

    const int64_t Y = c.Y();
    const int64_t payload = c.PayloadBytes();
    const bool isRecv = (c.needSchedule == 1);
    bool ok = true;

    // --- device 输入缓冲 ---
    void *dTokenData = nullptr;
    void *dSession = nullptr;
    void *dMicro = nullptr;
    void *dExpert = nullptr;
    void *dTokenInfo = nullptr;
    void *dCtx = nullptr;
    aclrtMalloc(&dTokenData, in.tokenData.size(), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dTokenData, in.tokenData.size(), in.tokenData.data(), in.tokenData.size(), ACL_MEMCPY_HOST_TO_DEVICE);

    if (!isRecv) {
        // NORM 路径：sessionIds/microBatchIds/expertIds 三个独立 buf（非扫描）
        aclrtMalloc(&dSession, in.sessionIdsBuf.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(dSession, in.sessionIdsBuf.size() * 4, in.sessionIdsBuf.data(), in.sessionIdsBuf.size() * 4,
                    ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMalloc(&dMicro, in.microBatchIdsBuf.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(dMicro, in.microBatchIdsBuf.size() * 4, in.microBatchIdsBuf.data(), in.microBatchIdsBuf.size() * 4,
                    ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMalloc(&dExpert, in.expertIds.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(dExpert, in.expertIds.size() * 4, in.expertIds.data(), in.expertIds.size() * 4,
                    ACL_MEMCPY_HOST_TO_DEVICE);
    } else {
        // RECV 路径：token_info buffer [A, M, F]，F = 1(flag) + 1(layer_id) + BS*K(expert_ids)。
        //   对齐 op_kernel/ffn_wb_scan_token_info.h + ffn_wb_scan_sort_one_core.h：
        //     flag  = token_info[(a*M + m)*F + 0]（curMicroBatchID 行须为 1 表示 ready）
        //     lid   = token_info[(a*M + m)*F + 1]
        //     eids  = token_info[(a*M + curMicroBatchID)*F + 2 + j] = in.expertIds[a*BS*K + j]
        const int64_t F = 2 + c.BS * c.K;
        std::vector<int32_t> tokenInfo((size_t)c.A * c.M * F, 0);
        const int64_t bsk = c.BS * c.K;
        for (int64_t a = 0; a < c.A; ++a) {
            for (int64_t m = 0; m < c.M; ++m) {
                int64_t base = (a * c.M + m) * F;
                tokenInfo[base + 0] = 1;                   // flag=ready（扫描循环需 sum==A）
                tokenInfo[base + 1] = (int32_t)c.layerNum; // layer_id
            }
            int64_t eBase = (a * c.M + c.curMicroBatchID) * F + 2;
            for (int64_t j = 0; j < bsk; ++j)
                tokenInfo[eBase + j] = in.expertIds[a * bsk + j];
        }
        aclrtMalloc(&dTokenInfo, tokenInfo.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(dTokenInfo, tokenInfo.size() * 4, tokenInfo.data(), tokenInfo.size() * 4,
                    ACL_MEMCPY_HOST_TO_DEVICE);
    }

    // --- 1024B schedule_context（device 绝对地址写入 offset） ---
    std::vector<int8_t> ctx(CTX_LEN, 0);
    WriteU32(ctx, 0, (uint32_t)c.A);
    WriteU32(ctx, 4, (uint32_t)c.M);
    WriteU32(ctx, 8, (uint32_t)c.BS);
    WriteU32(ctx, 12, (uint32_t)c.K);
    WriteU32(ctx, 20, (uint32_t)c.HS());
    WriteU64(ctx, 400, (uint64_t)dTokenData);
    if (!isRecv) {
        WriteU64(ctx, 528, (uint64_t)dSession);
        WriteU64(ctx, 544, (uint64_t)dMicro);
        WriteU64(ctx, 560, (uint64_t)dExpert);
        WriteU32(ctx, 576, (uint32_t)c.GetOutNum());
    } else {
        WriteU64(ctx, 384, (uint64_t)dTokenInfo);        // token_info_buf（扫描路径）
        WriteU64(ctx, 416, (uint64_t)c.curMicroBatchID); // pollIdx = curMicroBatchID（uint64）
    }
    aclrtMalloc(&dCtx, CTX_LEN, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dCtx, CTX_LEN, ctx.data(), CTX_LEN, ACL_MEMCPY_HOST_TO_DEVICE);
    aclTensor *tCtx = CreateTensorRaw(dCtx, {CTX_LEN}, ACL_INT8);

    // --- device 输出缓冲 ---
    void *dY = nullptr;
    void *dGroup = nullptr;
    void *dSid = nullptr;
    void *dMbid = nullptr;
    void *dTid = nullptr;
    void *dEoff = nullptr;
    void *dScale = nullptr;
    void *dAtn = nullptr;
    size_t yBytes = (size_t)Y * payload;
    aclrtMalloc(&dY, yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dGroup, (size_t)c.expertNum * 2 * 8, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dSid, (size_t)Y * 4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dMbid, (size_t)Y * 4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dTid, (size_t)Y * 4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dEoff, (size_t)Y * 4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dScale, (size_t)Y * 4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dAtn, 8, ACL_MEM_MALLOC_HUGE_FIRST);
    // 输出缓冲 malloc 后清零：防长 batch 连跑跨用例残值污染（尤其 actual_token_num
    // 读到上个用例的脏值表现为 INT_MIN+残值负数）。kernel 只写有效区，尾部/未写区靠此清零。
    aclrtMemset(dY, yBytes, 0, yBytes);
    aclrtMemset(dGroup, (size_t)c.expertNum * 2 * 8, 0, (size_t)c.expertNum * 2 * 8);
    aclrtMemset(dSid, (size_t)Y * 4, 0, (size_t)Y * 4);
    aclrtMemset(dMbid, (size_t)Y * 4, 0, (size_t)Y * 4);
    aclrtMemset(dTid, (size_t)Y * 4, 0, (size_t)Y * 4);
    aclrtMemset(dEoff, (size_t)Y * 4, 0, (size_t)Y * 4);
    aclrtMemset(dScale, (size_t)Y * 4, 0, (size_t)Y * 4);
    aclrtMemset(dAtn, 8, 0, 8);

    aclTensor *tY = CreateTensorRaw(dY, {Y, c.H}, YAclDtype(c.tokenDtype));
    aclTensor *tGroup = CreateTensorRaw(dGroup, {c.expertNum, 2}, ACL_INT64);
    aclTensor *tSid = CreateTensorRaw(dSid, {Y}, ACL_INT32);
    aclTensor *tMbid = CreateTensorRaw(dMbid, {Y}, ACL_INT32);
    aclTensor *tTid = CreateTensorRaw(dTid, {Y}, ACL_INT32);
    aclTensor *tEoff = CreateTensorRaw(dEoff, {Y}, ACL_INT32);
    aclTensor *tScale = CreateTensorRaw(dScale, {Y}, ACL_FLOAT);
    aclTensor *tAtn = CreateTensorRaw(dAtn, {1}, ACL_INT64);

    std::vector<int64_t> maxOutShape = {c.A, c.BS, c.K, c.H};
    aclIntArray *mos = aclCreateIntArray(maxOutShape.data(), maxOutShape.size());

    // --- 两段式调用 ---
    uint64_t wsSize = 0;
    aclOpExecutor *exec = nullptr;
    aclnnStatus ret =
        aclnnFfnWorkerBatchingGetWorkspaceSize(tCtx, c.expertNum, mos, c.tokenDtype, c.needSchedule, c.layerNum, tY,
                                               tGroup, tSid, tMbid, tTid, tEoff, tScale, tAtn, &wsSize, &exec);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] GetWorkspaceSize ret=%d", (int)ret);
        ok = false;
    }

    void *ws = nullptr;
    if (ok && wsSize > 0) {
        aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        // workspace 必须清零：长 batch 连跑时 ACL_MEM_MALLOC_HUGE_FIRST 会复用已释放的脏 huge page，
        // kernel 多核 sort 的 scratch / 跨核 SetAtomicAdd 计数累加区若读到上个用例残值 → actual_token_num
        // 变少/减半（单跑拿到系统新清零页则正确，故仅长跑复现）。逐用例清零根治该跨用例污染。
        aclrtMemset(ws, wsSize, 0, wsSize);
    }
    if (ok) {
        ret = aclnnFfnWorkerBatching(ws, wsSize, exec, stream);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] aclnnFfnWorkerBatching ret=%d", (int)ret);
            ok = false;
        }
    }
    if (ok && aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        LOG_PRINT("  [FAIL] SyncStream");
        ok = false;
    }

    // --- D2H + 二进制一致比对（仅覆盖 [0,validCnt) 有效区 + 全量 actual_token_num/group_list 前缀） ---
    if (ok) {
        const int64_t vc = g.actualTokenNum;
        std::vector<uint8_t> hY(yBytes);
        std::vector<int64_t> hGroup(c.expertNum * 2);
        std::vector<int32_t> hSid(Y), hMbid(Y), hTid(Y), hEoff(Y);
        std::vector<float> hScale(Y);
        int64_t hAtn = 0;
        aclrtMemcpy(hY.data(), yBytes, dY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(hGroup.data(), hGroup.size() * 8, dGroup, hGroup.size() * 8, ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(hSid.data(), Y * 4, dSid, Y * 4, ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(hMbid.data(), Y * 4, dMbid, Y * 4, ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(hTid.data(), Y * 4, dTid, Y * 4, ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(hEoff.data(), Y * 4, dEoff, Y * 4, ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(hScale.data(), Y * 4, dScale, Y * 4, ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(&hAtn, 8, dAtn, 8, ACL_MEMCPY_DEVICE_TO_HOST);

        if (hAtn != g.actualTokenNum) {
            LOG_PRINT("  [FAIL] actual_token_num golden=%lld npu=%lld", (long long)g.actualTokenNum, (long long)hAtn);
            ok = false;
        }
        // actual_token_num（计数）与 group_list（升序 (eid,count) 聚合）与桶内顺序无关，严格逐位置比。
        ok &= BitwiseEqual(g.groupList.data(), hGroup.data(), (size_t)(g.groupListRows * 2), "group_list");
        // y / session/micro/token/expert(/scale)：生产多核 sort(=A2) 桶内走 MrgSort 归并序、非全局
        // 行号序，逐位置比会因桶内换位误报（详见 BucketSetEqual 注释）。改按专家桶【集合相等】比对：
        // 验证每专家桶内 token 集合(含 y payload 与各 id/scale)完全一致，不约束桶内顺序 —— 对齐 A2
        // 实际契约与 spec（专家分桶+计数精确+桶内集合正确）。单核路径桶内序天然=golden，同样通过。
        ok &= BucketSetEqual(g, hSid.data(), hMbid.data(), hTid.data(), hEoff.data(), hY.data(), hScale.data(), payload,
                             /*hasScale*/ c.tokenDtype == 2);
    }

    // --- 清理 ---
    if (ws)
        aclrtFree(ws);
    aclDestroyIntArray(mos);
    aclDestroyTensor(tCtx);
    aclDestroyTensor(tY);
    aclDestroyTensor(tGroup);
    aclDestroyTensor(tSid);
    aclDestroyTensor(tMbid);
    aclDestroyTensor(tTid);
    aclDestroyTensor(tEoff);
    aclDestroyTensor(tScale);
    aclDestroyTensor(tAtn);
    aclrtFree(dTokenData);
    aclrtFree(dCtx);
    if (dSession)
        aclrtFree(dSession);
    if (dMicro)
        aclrtFree(dMicro);
    if (dExpert)
        aclrtFree(dExpert);
    if (dTokenInfo)
        aclrtFree(dTokenInfo);
    aclrtFree(dY);
    aclrtFree(dGroup);
    aclrtFree(dSid);
    aclrtFree(dMbid);
    aclrtFree(dTid);
    aclrtFree(dEoff);
    aclrtFree(dScale);
    aclrtFree(dAtn);

    LOG_PRINT("  结果: %s", ok ? "PASS" : "FAIL");
    return ok;
}
#endif // USE_MOCK_ACLNN

// ---------------------------------------------------------------------------
// 迭代一 L0 标准用例（基础 shape + 单 dtype fp16，need_schedule=0）
// ---------------------------------------------------------------------------
static std::vector<FwbCase> BuiltinL0Cases()
{
    std::vector<FwbCase> v;
    // L0 标准用例：基础 shape + fp16（迭代一必需，手工规模，快速自测）
    {
        FwbCase c;
        c.name = "L0_std_fp16_basic";
        c.A = 8;
        c.M = 2;
        c.BS = 4;
        c.K = 4;
        c.H = 16;
        c.expertNum = 8;
        c.tokenDtype = 0;
        c.needSchedule = 0;
        c.outNum = 8;
        c.maskStride = 5;
        v.push_back(c);
    }
    // k=N：无 mask，全部 token 有效（boundary k_eq_n 缩比）
    {
        FwbCase c;
        c.name = "L0_std_fp16_full_valid";
        c.A = 8;
        c.M = 2;
        c.BS = 4;
        c.K = 4;
        c.H = 16;
        c.expertNum = 8;
        c.tokenDtype = 0;
        c.needSchedule = 0;
        c.outNum = 8;
        c.maskStride = 0;
        v.push_back(c);
    }
    // same_bin：全部有效 token 路由同一专家（expert_num 较大）
    {
        FwbCase c;
        c.name = "L0_std_fp16_same_bin";
        c.A = 4;
        c.M = 1;
        c.BS = 4;
        c.K = 4;
        c.H = 8;
        c.expertNum = 16;
        c.tokenDtype = 0;
        c.needSchedule = 0;
        c.outNum = 4;
        c.maskStride = 0;
        c.allSameExpert = true;
        v.push_back(c);
    }
    return v;
}

// ---------------------------------------------------------------------------
// 迭代二 L1 内置用例（多 shape：小/中/上界；NORM(100)+RECV(101) 双路径；fp16/bf16/int8 三 dtype）
// ---------------------------------------------------------------------------
//   命名前缀 L1B_ 便于与 CSV L1 区分；每条显式设定路径与 dtype，作为可追溯的确定性机器证据。
//   token_data 物化以 slot 计（A*M*BS*K），全部控制在数十 MB 内，Mock 快速自测。
static std::vector<FwbCase> BuiltinL1Cases()
{
    std::vector<FwbCase> v;
    auto mk = [&](const char *name, int64_t A, int64_t M, int64_t BS, int64_t K, int64_t H, int64_t ep, int64_t td,
                  int64_t ns, int64_t mask, bool allSame = false, int64_t curMb = 0) {
        FwbCase c;
        c.name = name;
        c.A = A;
        c.M = M;
        c.BS = BS;
        c.K = K;
        c.H = H;
        c.expertNum = ep;
        c.tokenDtype = td;
        c.needSchedule = ns;
        c.outNum = A;
        c.maskStride = mask;
        c.allSameExpert = allSame;
        c.curMicroBatchID = curMb;
        v.push_back(c);
    };

    // --- 小 shape：三 dtype × NORM/RECV（快速覆盖 profile×path 6 组合） ---
    mk("L1B_small_fp16_NORM", 4, 2, 4, 4, 16, 8, /*fp16*/ 0, /*NORM*/ 0, 5);
    mk("L1B_small_fp16_RECV", 4, 2, 4, 4, 16, 8, 0, /*RECV*/ 1, 5);
    mk("L1B_small_bf16_NORM", 4, 2, 4, 4, 16, 8, /*bf16*/ 1, 0, 5);
    mk("L1B_small_bf16_RECV", 4, 2, 4, 4, 16, 8, 1, 1, 5);
    mk("L1B_small_int8_NORM", 4, 2, 4, 4, 16, 8, /*int8*/ 2, 0, 5); // dynamic_scale 有效
    mk("L1B_small_int8_RECV", 4, 2, 4, 4, 16, 8, 2, 1, 5);

    // --- 中 shape：稀疏/全有效各一，NORM+RECV ---
    mk("L1B_mid_fp16_NORM_sparse", 32, 4, 8, 8, 64, 128, 0, 0, 5);
    mk("L1B_mid_fp16_RECV_sparse", 32, 4, 8, 8, 64, 128, 0, 1, 5, false, /*curMb*/ 2);
    mk("L1B_mid_bf16_NORM_full", 24, 2, 8, 6, 32, 64, 1, 0, 0);
    mk("L1B_mid_int8_RECV_full", 24, 2, 8, 6, 32, 64, 2, 1, 0, false, /*curMb*/ 1);

    // --- 上界 shape：A/topK+1/expert_num 取上界（缩 M 控内存），NORM+RECV ---
    // A 上界=1024（M=1 控内存：1024*1*2*8*HS）
    mk("L1B_upper_A1024_NORM", 1024, 1, 2, 8, 16, 256, 0, 0, 7);
    mk("L1B_upper_A1024_RECV", 1024, 1, 2, 8, 16, 256, 0, 1, 7);
    // topK+1 上界=64
    mk("L1B_upper_topk64_NORM", 8, 2, 4, 64, 32, 256, 0, 0, 5);
    mk("L1B_upper_topk64_RECV", 8, 2, 4, 64, 32, 256, 0, 1, 5);
    // expert_num 上界=8192 + same_bin（全落同一专家，极端访存）
    mk("L1B_upper_expert8192_same_bin_NORM", 16, 1, 4, 8, 32, 8192, 0, 0, 0, /*allSame*/ true);
    mk("L1B_upper_expert8192_same_bin_RECV", 16, 1, 4, 8, 32, 8192, 0, 1, 0, /*allSame*/ true);

    // --- k=0 边界（全 mask）NORM+RECV：actual_token_num=0 ---
    mk("L1B_k_zero_NORM", 8, 2, 4, 8, 16, 32, 0, 0, /*全mask*/ 1);
    mk("L1B_k_zero_RECV", 8, 2, 4, 8, 16, 32, 0, 1, 1);

    return v;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    LOG_PRINT("========================================");
    LOG_PRINT("FfnWorkerBatching 算子 ST 测试 (A5/arch35，范围对齐 A2)");
    LOG_PRINT("精度口径: 二进制一致 (np.array_equal, rtol=atol=0)");
#ifdef USE_MOCK_ACLNN
    LOG_PRINT("模式: Mock (CPU golden 自测，无需 NPU)");
#else
    LOG_PRINT("模式: Real (NPU 执行 + 二进制一致比对)");
#endif
    LOG_PRINT("========================================");

    int passed = 0, failed = 0;
    auto tally = [&](bool ok) {
        if (ok)
            ++passed;
        else
            ++failed;
    };

    // === CPU Golden 自测（Mock/Real 均先执行，作为 golden 正确性硬证据） ===
    LOG_PRINT("\n############ CPU Golden 自测 ############");
    tally(TestHandVerifiedCase());
    tally(TestKZeroCase());

    // --- 命令行开关 ---
    //   argv[1] 非 "--" 开头视为 L0 CSV 路径覆盖
    //   --l1 / --no-l1 控制是否运行 L1（默认运行）
    //   --l1-csv <path> 指定 L1 CSV
    std::string l0Override, l1Override;
    bool runL1 = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--no-l1")
            runL1 = false;
        else if (a == "--l1")
            runL1 = true;
        else if (a == "--l1-csv" && i + 1 < argc)
            l1Override = argv[++i];
        else if (a.rfind("--", 0) != 0 && l0Override.empty())
            l0Override = a;
    }

    // === CSV 加载（L0 + L1，同格式；ParseCsvCases 自动按 need_schedule 分路径） ===
    auto loadCsv = [&](const std::string &override, const std::vector<std::string> &defaults, const char *tag) {
        std::vector<std::string> cands;
        if (!override.empty())
            cands.push_back(override);
        for (const auto &d : defaults)
            cands.push_back(d);
        std::vector<FwbCase> cs;
        for (const auto &p : cands) {
            cs = ParseCsvCases(p);
            if (!cs.empty()) {
                LOG_PRINT("已加载 %s CSV: %s (%zu 条)", tag, p.c_str(), cs.size());
                break;
            }
        }
        return cs;
    };

    std::vector<FwbCase> l0Cases = loadCsv(l0Override,
                                           {"../testcases/aclnnFfnWorkerBatching_l0_test_cases.csv",
                                            "testcases/aclnnFfnWorkerBatching_l0_test_cases.csv",
                                            "/home/zl/ops-transformer/ffn/ffn_worker_batching/tests/st/testcases/"
                                            "aclnnFfnWorkerBatching_l0_test_cases.csv"},
                                           "L0");

    std::vector<FwbCase> l1Cases;
    if (runL1)
        l1Cases = loadCsv(l1Override,
                          {"../testcases/aclnnFfnWorkerBatching_l1_test_cases.csv",
                           "testcases/aclnnFfnWorkerBatching_l1_test_cases.csv",
                           "/home/zl/ops-transformer/ffn/ffn_worker_batching/tests/st/testcases/"
                           "aclnnFfnWorkerBatching_l1_test_cases.csv"},
                          "L1");

#ifdef USE_MOCK_ACLNN
    // === L0 用例（Mock：CPU golden 自测 + 不变式断言） ===
    LOG_PRINT("\n############ L0 用例执行 (Mock) ############");
    LOG_PRINT("\n---- 内置 L0 标准用例（基础 shape + fp16，NORM）----");
    for (const auto &c : BuiltinL0Cases())
        tally(RunCaseMock(c));
    LOG_PRINT("\n---- CSV 驱动 L0 用例（NORM+RECV 双路径）----");
    for (const auto &c : l0Cases)
        tally(RunCaseMock(c));

    // === L1 用例（Mock：多 shape + NORM/RECV 双路径 + 三 dtype） ===
    LOG_PRINT("\n############ L1 用例执行 (Mock) ############");
    LOG_PRINT("\n---- 内置 L1 用例（多 shape：小/中/上界；NORM(100)+RECV(101)；fp16/bf16/int8）----");
    for (const auto &c : BuiltinL1Cases())
        tally(RunCaseMock(c));
    LOG_PRINT("\n---- CSV 驱动 L1 用例（%zu 条，pairwise + boundary + profile + extreme）----", l1Cases.size());
    for (const auto &c : l1Cases)
        tally(RunCaseMock(c));
#else
    // Real 模式：初始化 NPU，执行内置 + CSV 用例的真实调用与二进制比对
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    if (aclInit(nullptr) != ACL_SUCCESS) {
        LOG_PRINT("[FATAL] aclInit 失败");
        return 1;
    }
    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        LOG_PRINT("[FATAL] aclrtSetDevice 失败");
        aclFinalize();
        return 1;
    }
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        LOG_PRINT("[FATAL] aclrtCreateStream 失败");
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    LOG_PRINT("\n---- [Real] 内置 L0 标准用例 ----");
    for (const auto &c : BuiltinL0Cases())
        tally(RunCaseReal(c, stream));
    LOG_PRINT("\n---- [Real] CSV 驱动 L0 用例 ----");
    for (const auto &c : l0Cases)
        tally(RunCaseReal(c, stream));
    LOG_PRINT("\n---- [Real] 内置 L1 用例 ----");
    for (const auto &c : BuiltinL1Cases())
        tally(RunCaseReal(c, stream));
    LOG_PRINT("\n---- [Real] CSV 驱动 L1 用例 ----");
    for (const auto &c : l1Cases)
        tally(RunCaseReal(c, stream));

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
#endif

    LOG_PRINT("\n========================================");
    LOG_PRINT("测试报告");
    LOG_PRINT("总计: %d  通过: %d  失败: %d", passed + failed, passed, failed);
    LOG_PRINT("========================================");
    return failed == 0 ? 0 : 1;
}
