/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.  Licensed under Apache-2.0.
 *
 *  setup_addressing_check.cpp —— setup/teardown 地址 math 的端到端 host 侧模型自检（a5 纯 C++，无 CCU）。
 *  用 setup_host_precompute.h 的 LUT，模拟整条数据流并验证每个 token marker 完整往返：
 *    dispatch(routeTable 放置到对端 expert-major 数据区) → dispatch_teardown(去空洞紧密化成 expandX)
 *    → combine(assistTable 送回源卡 combine 数据区) → combine_teardown(expert_ids+expand_idx gather) → 重建。
 *  全绿 = setup 的 host 预算地址布局与已测 teardown 契约端到端自洽（含字节偏移/expert-major块序/块内位）。
 *
 *  编译运行（a5）：g++ -std=c++17 -I<common/inc> setup_addressing_check.cpp -o /tmp/addr_chk && /tmp/addr_chk
 */
#include "setup_host_precompute.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <cassert>

using namespace mc2;

// 唯一 marker：token 来自卡 C、token 序 i、发往全局专家 ge。非 0。
static inline int64_t Marker(uint32_t C, uint32_t i, uint32_t ge, uint32_t bs, uint32_t GE) {
    return static_cast<int64_t>(((static_cast<uint64_t>(C) * bs + i) * GE + ge) + 1);
}

static bool RunScenario(uint32_t bs, uint32_t topk, uint32_t h, uint32_t E, uint32_t R, uint32_t seed) {
    const uint32_t GE = R * E;
    const uint64_t hBytes = h, scaleBytes = h / 32U;
    const uint64_t A = static_cast<uint64_t>(R) * bs * (topk < E ? topk : E);
    const uint64_t dataRows = static_cast<uint64_t>(GE) * bs;         // 数据区/combine区行数 = R*E*bs
    std::mt19937 rng(seed);

    // ---- 1. 造全卡 expert_ids（distinct topk in [0,GE)）----
    std::vector<int32_t> ids(static_cast<size_t>(R) * bs * topk);
    for (uint32_t C = 0; C < R; ++C)
        for (uint32_t i = 0; i < bs; ++i) {
            std::vector<uint32_t> pool(GE); for (uint32_t g = 0; g < GE; ++g) pool[g] = g;
            for (uint32_t j = 0; j < topk; ++j) {
                std::uniform_int_distribution<uint32_t> d(j, GE - 1);
                std::swap(pool[j], pool[d(rng)]);
                ids[(static_cast<size_t>(C) * bs + i) * topk + j] = static_cast<int32_t>(pool[j]);
            }
        }
    const int32_t *allIds = ids.data();
    auto cardIds = [&](uint32_t C) { return allIds + static_cast<size_t>(C) * bs * topk; };

    // ---- 2. dispatch：每卡 routeTable，放置 marker 到对端数据区（按字节偏移/h 折回行）----
    std::vector<std::vector<int64_t>> dataRegion(R, std::vector<int64_t>(dataRows, 0));  // 每卡 [R*E*bs]
    std::vector<std::vector<int8_t>>  written(R, std::vector<int8_t>(dataRows, 0));       // 防重叠
    for (uint32_t C = 0; C < R; ++C) {
        std::vector<int64_t> route(static_cast<size_t>(bs) * topk * ROUTE_FIELDS);
        BuildDispatchRoute(cardIds(C), bs, topk, E, R, C, h, route.data());
        for (uint32_t i = 0; i < bs; ++i)
            for (uint32_t j = 0; j < topk; ++j) {
                const int64_t *e = route.data() + static_cast<size_t>(i * topk + j) * ROUTE_FIELDS;
                uint32_t dstRank = static_cast<uint32_t>(e[0]);
                uint64_t row = static_cast<uint64_t>(e[1]) / hBytes;      // dataDstByteOff/h
                if (static_cast<uint64_t>(e[2]) / scaleBytes != row) { printf("  [FAIL] data/scale 行不一致 C%u i%u j%u\n", C, i, j); return false; }
                if (row >= dataRows || written[dstRank][row]) { printf("  [FAIL] dispatch 越界/重叠 dst%u row%llu\n", dstRank, (unsigned long long)row); return false; }
                uint32_t ge = static_cast<uint32_t>(cardIds(C)[i * topk + j]);
                dataRegion[dstRank][row] = Marker(C, i, ge, bs, GE);
                written[dstRank][row] = 1;
            }
    }

    // ---- 3. 每卡 wishCnt + 校验数据区块填充与 wishCnt 一致（expert-major）----
    std::vector<std::vector<int32_t>> wish(R, std::vector<int32_t>(GE));
    for (uint32_t C = 0; C < R; ++C) {
        BuildDispatchWishCnt(C, allIds, bs, topk, E, R, wish[C].data());
        for (uint32_t blk = 0; blk < GE; ++blk) {
            uint32_t filled = 0;
            for (uint32_t p = 0; p < bs; ++p) if (written[C][static_cast<uint64_t>(blk) * bs + p]) ++filled;
            if (filled != static_cast<uint32_t>(wish[C][blk])) { printf("  [FAIL] wishCnt 不符 C%u blk%u filled%u wish%d\n", C, blk, filled, wish[C][blk]); return false; }
        }
    }

    // ---- 4. dispatch_teardown：去空洞紧密化 → expandX（expert-major le 外 src 内 块内 pos<wishCnt）----
    std::vector<std::vector<int64_t>> expandX(R);
    for (uint32_t O = 0; O < R; ++O) {
        for (uint32_t le = 0; le < E; ++le)
            for (uint32_t src = 0; src < R; ++src) {
                uint32_t blk = le * R + src;
                for (int32_t p = 0; p < wish[O][blk]; ++p)
                    expandX[O].push_back(dataRegion[O][static_cast<uint64_t>(blk) * bs + p]);
            }
        if (expandX[O].size() > A) { printf("  [FAIL] expandX 超 A\n"); return false; }
    }

    // ---- 5. combine：每 owner assistTable，送回源卡 combine 数据区 ----
    std::vector<std::vector<int64_t>> combineData(R, std::vector<int64_t>(dataRows, 0));
    std::vector<std::vector<int8_t>>  cwritten(R, std::vector<int8_t>(dataRows, 0));
    for (uint32_t O = 0; O < R; ++O) {
        std::vector<int64_t> assist(static_cast<size_t>(A) * ROUTE_FIELDS);
        BuildCombineAssist(O, wish[O].data(), bs, topk, E, R, h, A, assist.data());
        for (uint64_t r = 0; r < expandX[O].size(); ++r) {   // 只发有效行
            const int64_t *e = assist.data() + static_cast<size_t>(r) * ROUTE_FIELDS;
            if (e[0] < 0) { printf("  [FAIL] combine 有效行落到 padding\n"); return false; }
            uint32_t srcCard = static_cast<uint32_t>(e[0]);
            uint64_t row = static_cast<uint64_t>(e[1]) / hBytes;
            if (row >= dataRows || cwritten[srcCard][row]) { printf("  [FAIL] combine 越界/重叠 src%u row%llu\n", srcCard, (unsigned long long)row); return false; }
            combineData[srcCard][row] = expandX[O][r];
            cwritten[srcCard][row] = 1;
        }
        // padding 行 [有效行, A) 必须全是哨兵——否则上板 CCU 会把脏行误当有效行、写入 combine 数据区
        for (uint64_t r = expandX[O].size(); r < A; ++r) {
            const int64_t *e = assist.data() + static_cast<size_t>(r) * ROUTE_FIELDS;
            if (e[0] != OWNER_PAD_SENTINEL) {
                printf("  [FAIL] combine padding 非哨兵 O%u r%llu e0=%lld\n", O, (unsigned long long)r, (long long)e[0]); return false;
            }
        }
    }

    // ---- 6. combine_teardown：每卡 expert_ids+expand_idx gather，校验重建 == 原始 marker ----
    for (uint32_t C = 0; C < R; ++C) {
        std::vector<int32_t> eidx(static_cast<size_t>(bs) * topk);
        BuildExpandIdx(cardIds(C), bs, topk, E, R, eidx.data());
        for (uint32_t i = 0; i < bs; ++i)
            for (uint32_t j = 0; j < topk; ++j) {
                uint32_t ge = static_cast<uint32_t>(cardIds(C)[i * topk + j]);
                uint32_t pos = static_cast<uint32_t>(eidx[i * topk + j]);
                uint64_t row = static_cast<uint64_t>(ge) * bs + pos;
                int64_t got = combineData[C][row];
                int64_t want = Marker(C, i, ge, bs, GE);
                if (got != want) { printf("  [FAIL] 重建不符 C%u i%u j%u ge%u pos%u got%lld want%lld\n",
                                          C, i, j, ge, pos, (long long)got, (long long)want); return false; }
            }
    }
    return true;
}

int main() {
    struct { uint32_t bs, topk, h, E, R, seed; const char *tag; } cases[] = {
        {8, 8, 7168, 4, 2, 1234, "real degen topk=8"},
        {8, 3, 7168, 4, 2, 7,    "sparse topk=3"},
        {8, 2, 7168, 4, 2, 99,   "sparse topk=2"},
        {4, 2, 64,   2, 2, 5,    "small"},
        {16,6, 2048, 8, 2, 2025, "E8 GE16"},
        {8, 5, 7168, 4, 2, 42,   "topk=5"},
        {8, 4, 7168, 4, 1, 11,   "R=1 self-only（退化，dstRank恒0）"},
        {8, 8, 2048, 4, 4, 33,   "R=4（多卡非对称块交织）"},
        {8, 8, 2048, 4, 8, 77,   "R=8（大rank，combine trace R²情形）"},
    };
    int pass = 0, fail = 0;
    for (auto &c : cases) {
        bool ok = RunScenario(c.bs, c.topk, c.h, c.E, c.R, c.seed);
        printf("[%s] bs=%u topk=%u h=%u E=%u R=%u  %s\n", ok ? "PASS" : "FAIL",
               c.bs, c.topk, c.h, c.E, c.R, c.tag);
        ok ? ++pass : ++fail;
    }
    printf("==== 地址 math 端到端自检: PASS=%d FAIL=%d ====\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
