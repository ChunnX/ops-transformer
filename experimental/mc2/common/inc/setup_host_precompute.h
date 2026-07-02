/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 *
 *  setup_host_precompute.h —— dispatch/combine setup CCU 的 host 预算逻辑（纯 C++，a5 可编可测）。
 *  技术总监拍板 host 预算簿记（CCU 无动态标量 GM 写）：这些 LUT 由 driver host 端算好、传给 CCU kernel。
 *  产出的 expert-major 地址布局与已测 dispatch/combine teardown 的输入契约严格一致。
 *
 *  8 字节铁律：CCU LoadVariable 读的表一律 int64 排布（routeTable/assistTable/wishCnt 用 int64/int32 见注）。
 */
#ifndef SETUP_HOST_PRECOMPUTE_H
#define SETUP_HOST_PRECOMPUTE_H

#include <cassert>
#include <cstdint>
#include <vector>

namespace mc2 {

// routeTable 每行 3 个 int64：{dstRank, dataDstByteOff, scaleDstByteOff}
constexpr uint32_t ROUTE_FIELDS = 3;
constexpr int64_t  OWNER_PAD_SENTINEL = -1;   // combine assist：padding 行 ownerRank 哨兵，锁定 -1(0xFFFF..FF)。CCU 侧 CCU_IF(ownerVar==r) 对 r∈[0,R) 全 64 位 EQ 恒不等→跳过；self-check 用 e[0]<0 判 padding。勿改成 R（会与 <0 检查冲突）。

// ---- dispatch：本卡 myRank 的 routeTable（LoadVariable 读，int64）----
//   token i 第 j 复制体 → 全局专家 ge=expert_ids[i][j]；目的卡 dstRank=ge/E；对端本地专家 le=ge%E；
//   对端数据区 expert-major 块 block=le*R+myRank；块内位 pos=该 ge 在本卡出现次数（=expand_idx）；
//   目的行 row=block*bs+pos；dataDstByteOff=row*h；scaleDstByteOff=row*(h/32)。
inline void BuildDispatchRoute(const int32_t *expertIds, uint32_t bs, uint32_t topk,
                               uint32_t E, uint32_t R, uint32_t myRank, uint64_t h,
                               int64_t *routeTable /*[bs*topk*3]*/) {
    const uint32_t GE = R * E;
    const uint64_t hBytes = h, scaleBytes = h / 32U;
    std::vector<uint32_t> occ(GE, 0U);
    for (uint32_t i = 0; i < bs; ++i) {
        for (uint32_t j = 0; j < topk; ++j) {
            const uint32_t ge = static_cast<uint32_t>(expertIds[i * topk + j]);
            const uint32_t dstRank = ge / E;
            const uint32_t le = ge % E;
            const uint32_t block = le * R + myRank;          // expert-major，src=myRank
            const uint32_t pos = occ[ge]++;                  // = expand_idx（本卡该 ge 出现次数）
            assert(pos < bs && "occ[ge] 超块容量 bs：违反 distinct 契约（同 token topk 专家须互异），row 会踩相邻块");
            const uint64_t row = static_cast<uint64_t>(block) * bs + pos;
            int64_t *e = routeTable + static_cast<size_t>(i * topk + j) * ROUTE_FIELDS;
            e[0] = static_cast<int64_t>(dstRank);
            e[1] = static_cast<int64_t>(row * hBytes);
            e[2] = static_cast<int64_t>(row * scaleBytes);
        }
    }
}

// ---- dispatch：接收卡 cardC 的 wishCnt[R*E]（expert-major idx=le*R+src）----
//   wishCnt[le*R+src] = 源卡 src 发往全局专家(cardC*E+le) 的 token 数（= src 的 expert_ids 里选该 ge 的个数）。
//   allExpertIds 排布 [R][bs][topk]（driver 有全卡 expert_ids）。
inline void BuildDispatchWishCnt(uint32_t cardC, const int32_t *allExpertIds,
                                 uint32_t bs, uint32_t topk, uint32_t E, uint32_t R,
                                 int32_t *wishCnt /*[R*E]*/) {
    for (uint32_t le = 0; le < E; ++le) {
        const uint32_t ge = cardC * E + le;
        for (uint32_t src = 0; src < R; ++src) {
            const int32_t *ids = allExpertIds + static_cast<size_t>(src) * bs * topk;
            uint32_t cnt = 0;
            for (uint32_t i = 0; i < bs; ++i)
                for (uint32_t j = 0; j < topk; ++j)
                    if (static_cast<uint32_t>(ids[i * topk + j]) == ge) ++cnt;
            assert(cnt <= bs && "wishCnt 超块容量 bs：违反 distinct 契约，teardown 会按块长越界读");
            wishCnt[le * R + src] = static_cast<int32_t>(cnt);
        }
    }
}

// ---- dispatch：本卡 expand_idx[bs*topk]（= dispatch_teardown AIV 也会算的出现次数；driver 预算冗余一份备用）----
inline void BuildExpandIdx(const int32_t *expertIds, uint32_t bs, uint32_t topk, uint32_t E, uint32_t R,
                           int32_t *expandIdx /*[bs*topk]*/) {
    const uint32_t GE = R * E;
    std::vector<uint32_t> occ(GE, 0U);
    for (uint32_t i = 0; i < bs; ++i)
        for (uint32_t j = 0; j < topk; ++j) {
            const uint32_t ge = static_cast<uint32_t>(expertIds[i * topk + j]);
            expandIdx[i * topk + j] = static_cast<int32_t>(occ[ge]++);
        }
}

// ---- combine：owner 卡 O 的 assistTable[A*3]（int64：{ownerRank(送回卡), dataDstByteOff, scaleDstByteOff}）----
//   O 的 expandX[A,h] 是 dispatch_teardown 的输出（expert-major：外 le∈[0,E)、内 src∈[0,R)、块内 pos）。
//   O 上 expandX 行 r 属于块(le,src) 的第 pos 个 → 全局专家 ge=O*E+le；送回源卡 src 的 combine 数据区块 ge、位 pos。
//     dataDstByteOff=(ge*bs+pos)*h；scaleDstByteOff=(ge*bs+pos)*(h/32)。
//   有效行数 = Σ wishCntO[le*R+src]；[有效行, A) 为 padding → ownerRank=sentinel（CCU 侧跳过）。
//   wishCntO = O 卡的 wishCnt（= BuildDispatchWishCnt(O, ...)），决定 expandX 各块实长与顺序。
inline void BuildCombineAssist(uint32_t ownerCard, const int32_t *wishCntO /*[R*E]*/,
                               uint32_t bs, uint32_t topk, uint32_t E, uint32_t R, uint64_t h, uint64_t A,
                               int64_t *assistTable /*[A*3]*/, int64_t sentinel = OWNER_PAD_SENTINEL) {
    const uint64_t hBytes = h, scaleBytes = h / 32U;
    uint64_t r = 0;   // expandX 行游标（expert-major：le 外、src 内、块内 pos）
    for (uint32_t le = 0; le < E; ++le) {
        const uint32_t ge = ownerCard * E + le;
        for (uint32_t src = 0; src < R; ++src) {
            const int32_t cnt = wishCntO[le * R + src];
            for (int32_t pos = 0; pos < cnt; ++pos) {
                assert(r < A && "有效行 ΣwishCnt 超上界 A=R*bs*min(topk,E)：违反 distinct 契约，assistTable 会写越界");
                const uint64_t dstRow = static_cast<uint64_t>(ge) * bs + static_cast<uint64_t>(pos);
                int64_t *e = assistTable + static_cast<size_t>(r) * ROUTE_FIELDS;
                e[0] = static_cast<int64_t>(src);              // 送回源卡
                e[1] = static_cast<int64_t>(dstRow * hBytes);
                e[2] = static_cast<int64_t>(dstRow * scaleBytes);
                ++r;
            }
        }
    }
    for (; r < A; ++r) {   // padding 行
        int64_t *e = assistTable + static_cast<size_t>(r) * ROUTE_FIELDS;
        e[0] = sentinel; e[1] = 0; e[2] = 0;
    }
}

}  // namespace mc2

#endif  // SETUP_HOST_PRECOMPUTE_H
