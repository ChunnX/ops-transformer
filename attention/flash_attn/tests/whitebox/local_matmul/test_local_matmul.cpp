// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "flash_attn_matmul.h"
#include <array>
#include <iostream>
#include <stdexcept>
using namespace AscendC;
namespace FM = FlashAttnMatmul;
void Check(bool value) { if (!value) { throw std::runtime_error("instruction contract mismatch"); } }
uint32_t Fractals(uint32_t x) { return (x + 15) / 16; }

// Verify the instruction stream against the NZ layout and the PIPE_MTE1 -> PIPE_M
// ownership contract, including the shared forward barrier of split matmuls.
template<class T, bool AT, bool BT, bool SingleB>
void Run(char split, FM::MMParam p, uint32_t &aIdx, uint32_t &bIdx)
{
    const uint32_t initialA = aIdx, initialB = bIdx;
    const LocalTensor<T> a{TPosition::A1, 0, 131072}, b{TPosition::B1, 0, 131072};
    const LocalTensor<float> c{TPosition::CO1, 0, 32768};
    events.clear();
    if (split == 'K') { FM::MatmulK<128, AT, BT, SingleB>(a, b, aIdx, bIdx, c, p); }
    else if (split == 'N') { FM::MatmulN<128, AT, BT, SingleB>(a, b, aIdx, bIdx, c, p); }
    else { FM::MatmulFull<AT, BT, SingleB>(a, b, aIdx, bIdx, c, p); }
    const uint32_t parts = split == 'K' ? (p.singleK + 127) / 128 :
                           split == 'N' ? (p.singleN + 127) / 128 : 1;
    const uint32_t m = p.realM ? p.realM : p.singleM;
    uint32_t ai = initialA, bi = initialB, mmCount = 0, aLoads = 0, bLoads = 0;
    std::array<bool, 10> owned{};
    uint32_t aMutex = 0, bMutex = 0;
    uint64_t aAddress = 0, bAddress = 0;
    std::vector<std::array<uint32_t, 3>> sync;
    for (const auto &e : events) {
        if (e.kind == 'L' || e.kind == 'U') {
            Check(e.id >= 6 && e.id <= 9);
            sync.push_back({static_cast<uint32_t>(e.kind), e.pipe, e.id});
            if (e.kind == 'L' && e.pipe == PIPE_MTE1) { Check(!owned[e.id]); owned[e.id] = true; }
            if (e.kind == 'U' && e.pipe == PIPE_M) { Check(owned[e.id]); owned[e.id] = false; }
            continue;
        }
        const uint32_t tileK = split == 'K' ? std::min(128U, p.singleK - mmCount * 128) : p.singleK;
        const uint32_t tileN = split == 'N' ? std::min(128U, p.singleN - mmCount * 128) : p.singleN;
        if (e.kind == 'D') {
            const auto &l = e.load;
            Check(l.mStartPosition == 0 && l.kStartPosition == 0);
            if (e.position == TPosition::A2) {
                ai ^= 1; aMutex = 6 + ai; aAddress = ai * 32768;
                Check(owned[aMutex] && e.dst == aAddress);
                const uint64_t offset = split == 'K' ? mmCount * 128ULL * (AT ? 16 : Fractals(p.singleM) * 16) : 0;
                Check(e.src == offset * sizeof(T));
                Check(l.ifTranspose == AT);
                Check(l.mStep == (p.realM ? Fractals(m) : Fractals(AT ? tileK : p.singleM)));
                Check(l.kStep == Fractals(AT ? p.singleM : tileK));
                Check(l.srcStride == (AT ? Fractals(p.singleK) : Fractals(p.singleM)));
                Check(l.dstStride == (AT ? Fractals(p.singleM) : Fractals(m)));
                ++aLoads;
            } else {
                Check(e.position == TPosition::B2);
                if (!SingleB) { bi ^= 1; }
                bMutex = 8 + (SingleB ? 0 : bi); bAddress = (SingleB ? 0 : bi) * 32768;
                Check(owned[bMutex] && e.dst == bAddress);
                const uint64_t offset = split == 'K' ? mmCount * 128ULL * (BT ? Fractals(p.singleN) * 16 : 16) :
                                        split == 'N' ? mmCount * 128ULL * (BT ? 16 : Fractals(p.singleK) * 16) : 0;
                Check(e.src == offset * sizeof(T));
                Check(l.ifTranspose == !BT);
                Check(l.mStep == Fractals(BT ? tileN : tileK));
                Check(l.kStep == Fractals(BT ? tileK : tileN));
                Check(l.srcStride == Fractals(BT ? p.singleN : p.singleK));
                Check(l.dstStride == Fractals(tileN));
                ++bLoads;
            }
        } else {
            Check(e.kind == 'M' && owned[aMutex] && owned[bMutex]);
            Check(e.src == aAddress && e.b == bAddress);
            Check(e.dst == (split == 'N' ? mmCount * 128ULL * Fractals(m) * 16 * sizeof(float) : 0));
            Check(e.mm.m == (m == 1 ? 16 : m) && e.mm.n == tileN && e.mm.k == tileK);
            Check(e.mm.cmatrixInitVal == (split != 'K' || mmCount == 0));
            Check(!e.mm.cmatrixSource && e.mm.unitFlag == 0);
            ++mmCount;
        }
    }
    Check(mmCount == parts && bLoads == parts && aLoads == (split == 'N' ? 1 : parts));
    Check(aIdx == ai && bIdx == bi);
    for (bool held : owned) { Check(!held); }
    // Exact barrier order: split paths use B's forward barrier for both loads;
    // N splitting keeps A owned throughout all submatmuls.
    std::vector<std::array<uint32_t, 3>> expected;
    auto add = [&](char op, pipe_t pipe, uint32_t id) { expected.push_back({static_cast<uint32_t>(op), pipe, id}); };
    ai = initialA; bi = initialB;
    if (split == 'N') { ai ^= 1; add('L', PIPE_MTE1, 6 + ai); }
    for (uint32_t i = 0; i < parts; ++i) {
        if (split != 'N') { ai ^= 1; add('L', PIPE_MTE1, 6 + ai); }
        if (split == 'F') { add('U', PIPE_MTE1, 6 + ai); }
        if (!SingleB) { bi ^= 1; }
        const uint32_t bid = 8 + (SingleB ? 0 : bi);
        add('L', PIPE_MTE1, bid); add('U', PIPE_MTE1, bid);
        if (split == 'F') { add('L', PIPE_M, 6 + ai); }
        add('L', PIPE_M, bid);
        if (split != 'N') { add('U', PIPE_M, 6 + ai); }
        add('U', PIPE_M, bid);
    }
    if (split == 'N') { add('U', PIPE_M, 6 + ai); }
    Check(sync == expected);
}

template<class T, bool SingleB> uint32_t Suite()
{
    uint32_t cases = 0;
    // Both initial parities, then persistent indices across alternating BMMs.
    for (uint32_t startA : {0U, 1U}) for (uint32_t startB : {0U, 1U}) {
        uint32_t a = startA, b = startB;
        for (uint32_t m : {1U, 2U, 15U, 16U, 17U, 31U, 32U, 63U, 64U, 127U, 128U}) {
            for (uint32_t s : {1U, 15U, 16U, 17U, 127U, 128U, 129U, 255U, 256U}) {
                for (uint32_t d : {64U, 72U, 128U, 192U, 256U}) {
                    FM::MMParam qk{m, s, d, 0, Fractals(d) * 16, 0};
                    // ND: capacity-compatible choices for each split direction.
                    if (m <= 32 && s > 128) { Run<T, false, true, SingleB>('N', qk, a, b); ++cases; }
                    else if (s <= 128) { Run<T, false, true, SingleB>(d > 128 ? 'K' : 'F', qk, a, b); ++cases; }
                    // DN BMM1 (S2 <=128), and its transposed P BMM2.
                    if (s <= 128 && !SingleB) {
                        FM::MMParam kq{s, m, d, 0, Fractals(d) * 16, 0};
                        Run<T, false, true, SingleB>(d > 128 ? 'K' : 'F', kq, a, b); ++cases;
                        FM::MMParam pv{128, std::min(d, 128U), s, 0, 0, Fractals(std::min(d, 128U)) * 16};
                        Run<T, true, false, SingleB>('F', pv, a, b); ++cases;
                    }
                    FM::MMParam pv{128, std::min(d, 128U), s, m, 0, Fractals(std::min(d, 128U)) * 16};
                    Run<T, false, false, SingleB>(s > 128 ? 'K' : 'F', pv, a, b); ++cases;
                }
            }
        }
    }
    return cases;
}
int main()
{
    const auto count = Suite<half, false>() + Suite<half, true>() + Suite<bfloat16_t, false>() + Suite<bfloat16_t, true>();
    std::cout << "PASS: " << count << " matmul instruction contracts (FP16/BF16, persistent L0 indices)\n";
}
