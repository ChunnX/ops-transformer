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
 * \file moe_tutel_combine_gates_16b.h
 * \brief MoeTutelCombineGates bfloat16 / float16 kernel class.
 */

#ifndef MOE_TUTEL_COMBINE_GATES_16B_H_
#define MOE_TUTEL_COMBINE_GATES_16B_H_

#include "kernel_operator.h"
#include "moe_tutel_combine_gates_tiling_struct.h"

using namespace AscendC;

// ---------------------------------------------------------------------------
// INNER_OFFSET — number of tasks in one inner transfer block.
//
// DESIGN-BRANCH-0.md §2 formula (2-8): innerOffset = 16, a compile-time
// constant shared with the host side (HostTiling.md §4.3). It is a transfer
// granularity, not a shape quantity, so it is NOT carried in TilingData. It is also
// NOT the size of the block-level routes on its own since R4-7: the two of them a
// vector load reads additionally carry a one-vector-register floor, so the group is
// 736 B rather than the plain 16 * 20 = 320 B (see Init()).
// ---------------------------------------------------------------------------
constexpr int32_t INNER_OFFSET = 16;

// ---------------------------------------------------------------------------
// Reduction-scale helpers (S6b) — binary32 exponent arithmetic on the bit pattern.
//
// WHY THIS EXISTS (root cause of the case00049 mismatch, fixed here): the fp32
// intermediate accumulator has the SAME 8-bit exponent field as bfloat16, so
// "Cast to fp32 then reduce" (A9 / fp32_intermediate_accumulation) buys mantissa
// accuracy but NO extra dynamic range. With |x| at the bfloat16 ceiling
// (+-3.38953139e38, i.e. within 0.4% of fp32 max 3.40282347e38) a pairwise fold of
// as few as two same-sign products overflows the fp32 accumulator even though the
// exact sum is still bfloat16 representable; the overflow cancels to +/-Inf or NaN
// and the kernel then disagrees with the float64 oracle, which never overflows.
//
// Scaling every product by one power of two before the reduction keeps every
// partial sum inside a range where it cannot overflow, and scaling the result back
// afterwards is exact (a power of two), so the accumulation itself is still the
// mandated fp32 Mul/ReduceSum — only its exponent window is widened. No UB route is
// added (Abs/Muls reuse xCastBuf/yCastBuf, which S6 leaves dead; ReduceMax reuses
// workBuf/tmpBuf), so P stays 12 and the 20H + 736 budget is unchanged (the 736 B
// small group is the R4-7 one-register floor, see Init()).
// ---------------------------------------------------------------------------
union MoeTutelCombineGatesFloatBits {
    float f;
    uint32_t u;
};

// Pow2Of — 2^exponent for exponent in [-126, 126] (normal binary32 range only, which
// is exactly what the callers' clamp to that window guarantees).
__aicore__ inline float Pow2Of(int32_t exponent)
{
    MoeTutelCombineGatesFloatBits bits;
    bits.u = static_cast<uint32_t>(exponent + 127) << 23;
    return bits.f;
}

// ReductionShiftOf — shift k such that 2^-k is the reduction scale for a largest
// product magnitude of maxAbs. For finite maxAbs > 0 the exponent is read straight
// off the bit pattern, so maxAbs * 2^-k lands in [0.5, 2): the scaled partial sums
// stay below 2 * H, and H <= 12608 (the branch's UB-derived ceiling) keeps them far
// from the fp32 range. k is clamped to [-126, 126] because 2^-k must stay normal (a
// subnormal scale would round the products away) and 2^k must stay representable for
// the scale-back. Zero / inf / NaN fall out of the same formula with no extra test:
// an all-zero pattern reads as exponent -127 (k = -126, and the scaled products stay
// exactly 0), while the inf/NaN pattern reads as exponent 128 (clamped to k = 126, so
// inf/NaN propagate through Mul/ReduceSum and the scale-back unchanged).
__aicore__ inline int32_t ReductionShiftOf(float maxAbs)
{
    MoeTutelCombineGatesFloatBits bits;
    bits.f = maxAbs;
    int32_t shift = static_cast<int32_t>((bits.u >> 23) & 0xFFu) - 127 + 1;
    if (shift > 126) {
        shift = 126;
    } else if (shift < -126) {
        shift = -126;
    }
    return shift;
}

// ===========================================================================
// MoeTutelCombineGatesMulVF — chain step S6 (element-wise multiply) on the
// Vector main chain.
//
// Registers only: the two fp32 operands are loaded from UB, multiplied in the
// vector register file and stored back, so no intermediate value occupies a UB
// route (this is why the UB route count stays at the designed P = 12).
//
// count          — number of valid elements (= hidden_, never the padded length;
//                  Kernel.md §9.3 "row pad single conclusion")
// oneRepeatSize  — elements per vector repeat = GetVecLen() / sizeof(float) (= 64;
//                  the VF below is instantiated with T = float, see the
//                  asc_vf_call<MoeTutelCombineGatesMulVF<float>> in
//                  MulFp32OnMainChain)
// ---------------------------------------------------------------------------
template <typename T>
__simd_vf__ inline void MoeTutelCombineGatesMulVF(__ubuf__ T *dstAddr, __ubuf__ T *src0Addr, __ubuf__ T *src1Addr,
                                                  uint32_t count, uint32_t oneRepeatSize)
{
    AscendC::Reg::RegTensor<T> src0Reg;
    AscendC::Reg::RegTensor<T> src1Reg;
    AscendC::Reg::RegTensor<T> dstReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;

    const uint16_t repeatTimes = static_cast<uint16_t>((count + oneRepeatSize - 1) / oneRepeatSize);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        // Re-derived every iteration so the mask is exact on the tail repeat.
        uint32_t remain = count - static_cast<uint32_t>(i) * oneRepeatSize;
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(remain);
        AscendC::Reg::LoadAlign(src0Reg, src0Addr, aReg);
        AscendC::Reg::LoadAlign(src1Reg, src1Addr, aReg);
        AscendC::Reg::Mul(dstReg, src0Reg, src1Reg, mask);
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg, mask);
    }
}

// ===========================================================================
// MoeTutelCombineGatesCompensatedReduceExactVF — fp32 chain step S7 (compensated
// reduction over H) on the Vector main chain; replaces the single `ReduceSum`
// call of the single-fold design (accuracy fix; the fp32 branch takes the same step).
//
// WHY (root cause, measured — not a guess): a single fp32 pairwise fold (what
// `ReduceSum` does) carries a rounding error of order u*Sum_h|p_h| (u = 2^-24).
// For every task whose exact inner product suffers cancellation (kappa =
// Sum|p|/|T| up to 3.8e7 in the black-box fp16 cases) that error is many times
// the fp16 output quantum, so the emitted fp16 value is NOT the correctly
// rounded one. Device evidence from a failing case: the worst element had |g| = 5.96e-6, exact
// sum
// 5.95301e-6, and the kernel returned 4.768e-6 — 20 fp16 quanta away from the
// correctly rounded value, while the float64 golden sat exactly on it (0 ulp).
// The pure relative metric `mare` only makes such deviations *visible* on
// near-zero elements; it does not create them, so this is an implementable
// accuracy defect of the accumulation, and it is fixed here.
//
// WHY THIS IS EXACT FOR THE 16-BIT BRANCH (no TwoProduct channel is needed,
// unlike the fp32 branch):
//   * An fp16 product x*y has at most 11 + 11 = 22 significant bits, a bfloat16
//     product at most 8 + 8 = 16, and an fp32 significand holds 24 => every
//     product is representable in fp32 EXACTLY (subnormal inputs only reduce the
//     significand, so the bound still holds). The S6b power-of-two scaling is
//     exact as well (it only moves the exponent and the products never leave the
//     normal fp32 range: |p| >= 2^-49 * maxAbs by the input dtype's own range).
//   * Carrying the two-sum rounding error in a second accumulator therefore
//     accumulates the EXACT sum of the exact products, and the final
//     `hi + ReduceSum(lo)` is its correctly rounded fp32 value (error <= 1 ulp).
//   * S8 then scales back by the exact power of two 2^k, so the value handed to
//     the narrowing Cast (S10 / CAST_ROUND) is the correctly rounded fp32 value
//     of the exact inner product => the fp16/bf16 output equals the float64
//     golden's own rounded value except for the (measure-zero, ~1e-5) double
//     rounding corner.
//
// ALGORITHM (all in vector registers — no UB staging, no scalar loop):
//   Phase 1  per-lane compensated accumulation over chunks of `oneRepeatSize`
//            elements (lane j takes elements j, j+V, j+2V, ...). Each step is a
//            full Knuth TwoSum:
//                s = hi + x; bb = s - hi; t1 = s - bb; err = (hi - t1) + (x - bb)
//                lo = lo + err ; hi = s
//            TwoSum (not the 3-op Fast2Sum / 4-op Kahan form) is deliberate: it
//            is exact for *any* operand magnitudes, and the failing elements are
//            precisely the near-cancelling ones where the |hi| >= |x|
//            precondition of the cheaper forms does not hold.
//   Phase 2  compensated binary halving tree over the lanes
//            (Reg::DeInterleave pairs lane 2i with lane 2i+1); every level's
//            error terms are merged into `lo`, and different levels occupy
//            disjoint lane sets, so one final lane sum collects each of them
//            exactly once.
//   Final    result = (lane sum of hi) + (lane sum of lo).
//
// SEMANTICS PRESERVED (branch hard constraints):
//   * H == 0 short circuits in Process(): count >= 1 here.
//   * NaN / +/-Inf on a legal task propagate verbatim: `hi` follows IEEE-754
//     addition exactly, and the compensation channel — the only place where an
//     infinite operand can create a spurious Inf - Inf = NaN — is NaN-filtered
//     lane-wise before the horizontal sum; +/-Inf is therefore emitted, never
//     saturated or clamped (Kernel.md §9.2 / §9.5).
//   * Lane assignment does not change the value of the sum (each product enters
//     exactly once), so the partial last chunk may reuse an ALIGNED in-route
//     window instead of over-reading UB (B7: lanes past `count` must never
//     contribute; the window mask sets them to exact 0 / keeps the old hi).
//
// count          — number of valid products (= hidden_, never the padded length)
// oneRepeatSize  — elements per vector register = GetVecLen() / sizeof(float) (= 64;
//                  NOT sizeof(T): this VF is instantiated with T = float)
// bufElems       — element capacity of the source route, used to pick the tail
//                  window so that no read leaves the route window
// ===========================================================================
template <typename T>
__simd_vf__ inline void MoeTutelCombineGatesCompensatedReduceExactVF(__ubuf__ T *srcAddr, __ubuf__ T *dstAddr,
                                                                     uint32_t count, uint32_t oneRepeatSize,
                                                                     uint32_t bufElems)
{
    AscendC::Reg::RegTensor<T> xReg;
    AscendC::Reg::RegTensor<T> hiReg;
    AscendC::Reg::RegTensor<T> loReg;
    AscendC::Reg::RegTensor<T> sReg;
    AscendC::Reg::RegTensor<T> bReg;
    AscendC::Reg::RegTensor<T> tReg;
    AscendC::Reg::RegTensor<T> eReg;
    AscendC::Reg::RegTensor<T> aReg;
    AscendC::Reg::RegTensor<T> zReg;
    AscendC::Reg::MaskReg mAll = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg mNot, mUp, mWin, mFin;

    const uint32_t fullChunks = count / oneRepeatSize;
    const uint32_t rest = count - fullChunks * oneRepeatSize;
    // Lanes >= min(V, count) hold exact zeros, so folding them is exact and the
    // tree may stop as soon as only lane 0 is left.
    uint32_t width = (count < oneRepeatSize) ? count : oneRepeatSize;

    AscendC::Reg::Duplicate(hiReg, static_cast<T>(0));
    AscendC::Reg::Duplicate(loReg, static_cast<T>(0));
    AscendC::Reg::Duplicate(zReg, static_cast<T>(0));

    // ---- Phase 1a : full chunks (every lane carries a product) ----
    for (uint32_t c = 0; c < fullChunks; ++c) {
        AscendC::Reg::LoadAlign(xReg, srcAddr + c * oneRepeatSize);
        AscendC::Reg::Add(sReg, hiReg, xReg, mAll); // s  = hi + x
        AscendC::Reg::Sub(bReg, sReg, hiReg, mAll); // bb = s - hi
        AscendC::Reg::Sub(tReg, sReg, bReg, mAll);  // t1 = s - bb
        AscendC::Reg::Sub(eReg, hiReg, tReg, mAll); // t2 = hi - t1
        AscendC::Reg::Sub(tReg, xReg, bReg, mAll);  // t3 = x - bb
        AscendC::Reg::Add(eReg, eReg, tReg, mAll);  // err = t2 + t3
        AscendC::Reg::Add(loReg, loReg, eReg, mAll);
        hiReg = sReg;
    }

    // ---- Phase 1b : partial tail chunk ----
    // The last `rest` products are reached through the aligned window
    // [base, base + V): base is the smaller of the natural offset and the largest
    // remaining window start, so base <= count - rest always holds (the window
    // starts at or before the first new product) and base + V >= count holds
    // whenever the route can hold one whole window.
    //
    // R4-2 (underflow guard): `bufElems` is the downlinked TilingData field
    // ubFp32Elems, whose lower bound is authoritative on the HOST side, while V
    // (oneRepeatSize) is the kernel's own GetVecLen()/sizeof(float). The two agree
    // today (both 64) but come from different sources, so `bufElems - V` may not be
    // written as a bare subtraction: for bufElems < V it would wrap in uint32_t to
    // ~4.29e9 elements and `LoadAlign` would read a wild UB address.
    // 0 is the semantically safe lower bound, not a liveness patch:
    //   * the mask below is exactly [laneFrom, laneTo) with
    //     laneTo = laneFrom + rest = count - base, so with base = 0 every SELECTED
    //     lane is <= count <= bufElems, i.e. still inside the route — no stale UB
    //     value can enter the sum and no product is counted twice
    //     (fullChunks == 0 whenever bufElems < V, because count <= bufElems);
    //   * only the address-level load of lanes >= bufElems leaves the route, and
    //     that read is bounded by one vector register, so it is never a wild
    //     address. (The SDK main-chain calls of S5/S6/S6b used to be in the same
    //     bounded, store-masked over-read class; since the R4-6 fix their window is
    //     a subset of their H-long route, so among the H-LONG routes this guard path
    //     is now the only crossing. The block-level small buffers used to be the one
    //     separate class of this kind — the S10 narrowing Cast read one whole fp32
    //     register from the 64 B sumBuf_ and the UB -> UB DataCopy one whole T
    //     register from the 32 B zTmpBuf_; R4-7 floored both routes at one whole
    //     vector register, so that class is empty now too — see the S10 note in
    //     CopyOut and the byte accounts above.)
    // Init() additionally re-floors ubFp32Elems_ at one whole vector register, so
    // this path is a backstop for a corrupt/mismatched downlink, not a
    // configuration the host can produce: the host aligns ubFp32Elems to
    // vectorSize/4, hence ubFp32Elems >= 64 && (ubFp32Elems % 64) == 0 always holds.
    if (rest > 0) {
        uint32_t base = fullChunks * oneRepeatSize;
        if (base + oneRepeatSize > bufElems) {
            base = (bufElems > oneRepeatSize) ? (bufElems - oneRepeatSize) : 0u;
        }
        uint32_t laneFrom = fullChunks * oneRepeatSize - base; // lane of the first new product
        uint32_t laneTo = laneFrom + rest;                     // one past the last new product
        AscendC::Reg::LoadAlign(xReg, srcAddr + base);
        AscendC::Reg::Add(sReg, hiReg, xReg, mAll);
        AscendC::Reg::Sub(bReg, sReg, hiReg, mAll);
        AscendC::Reg::Sub(tReg, sReg, bReg, mAll);
        AscendC::Reg::Sub(eReg, hiReg, tReg, mAll);
        AscendC::Reg::Sub(tReg, xReg, bReg, mAll);
        AscendC::Reg::Add(eReg, eReg, tReg, mAll);
        // Window mask [laneFrom, laneTo): lanes below laneFrom were already
        // accumulated by the full chunks and lanes from laneTo lie past the row,
        // so neither may take part — no product is added twice and no stale UB
        // value ever enters the sum.
        mUp = AscendC::Reg::UpdateMask<T>(laneFrom);
        AscendC::Reg::Not(mNot, mUp, mAll);
        mUp = AscendC::Reg::UpdateMask<T>(laneTo);
        AscendC::Reg::And(mWin, mNot, mUp, mAll);
        AscendC::Reg::Select(hiReg, sReg, hiReg, mWin);
        AscendC::Reg::Select(eReg, eReg, zReg, mWin); // out-of-window lanes -> exact 0
        AscendC::Reg::Add(loReg, loReg, eReg, mAll);
    }

    // ---- Phase 2 : compensated binary halving tree over the lanes ----
    while (width > 1) {
        AscendC::Reg::DeInterleave(aReg, bReg, hiReg, zReg); // a = even lanes, b = odd lanes
        AscendC::Reg::Add(sReg, aReg, bReg, mAll);           // s  = a + b
        AscendC::Reg::Sub(tReg, sReg, aReg, mAll);           // bb = s - a
        AscendC::Reg::Sub(xReg, sReg, tReg, mAll);           // t1 = s - bb
        AscendC::Reg::Sub(eReg, aReg, xReg, mAll);           // t2 = a - t1
        AscendC::Reg::Sub(xReg, bReg, tReg, mAll);           // t3 = b - bb
        AscendC::Reg::Add(eReg, eReg, xReg, mAll);           // err = t2 + t3
        AscendC::Reg::Add(loReg, loReg, eReg, mAll);
        hiReg = sReg;
        width = (width + 1u) >> 1u;
    }

    // ---- Non-finite guard of the compensation channel ----
    // An error-free transformation is only *defined* on finite operands: if a
    // product is +/-Inf, `s = hi + x` is Inf and the difference terms become
    // Inf - Inf = NaN. That NaN can only ever live in the compensation channel,
    // because `hi` holds exactly what IEEE-754 addition produces (Inf stays Inf,
    // NaN stays NaN). Dropping the NaN lanes of `lo` before the horizontal sum
    // therefore leaves `hi` — and with it the emitted value — untouched:
    // +/-Inf is propagated, never saturated or clamped, and a NaN product still
    // makes the whole result NaN through `hi`.
    AscendC::Reg::Compare<T, AscendC::CMPMODE::EQ>(mFin, loReg, loReg, mAll);
    AscendC::Reg::Select(xReg, loReg, zReg, mFin);

    // ---- Final : lane sum of the accumulator + lane sum of all compensation ----
    AscendC::Reg::ReduceSum<T, T>(sReg, xReg, mAll);
    AscendC::Reg::Add(hiReg, hiReg, sReg, mAll);
    uint32_t oneLane = 1;
    mWin = AscendC::Reg::UpdateMask<T>(oneLane);
    AscendC::Reg::StoreAlign(dstAddr, hiReg, mWin);
}

// ===========================================================================
// class MoeTutelCombineGatesB16<T>
//
// T = bfloat16_t (tilingKey 0, Branch-0, default) or half (tilingKey 2, Branch-2).
// The input rows, the two 16-slot staging blocks and the output block use `T`;
// everything between S5 and S10 is fp32.
// ===========================================================================
template <typename T>
class MoeTutelCombineGatesB16 {
public:
    __aicore__ inline MoeTutelCombineGatesB16() {}

    // -----------------------------------------------------------------------
    // Init — GM binding, TilingData unpack, core range, TBuf allocation and the
    // DataCopyPad parameter pre-computation (DESIGN-BRANCH-0.md §5.1).
    //
    // The TPipe is created by the entry point and passed in by pointer
    // (Kernel.md §1/§4); no workspace is bound (A12, DESIGN-BRANCH-0.md §8).
    // -----------------------------------------------------------------------
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y_grad, GM_ADDR indices, GM_ADDR locations, GM_ADDR gates_grad,
                                const MoeTutelCombineGatesTilingData *tilingData, AscendC::TPipe *pipe)
    {
        td_ = tilingData;
        pipe_ = pipe;

        // ---- TilingData unpack (TilingData.md §1, 7 fields) ----
        samples_ = td_->samples;
        hidden_ = td_->hidden;
        capacity_ = td_->capacity;
        batchSize_ = td_->batchSize;
        taskNumPerBlock_ = td_->taskNumPerBlock;

        // ---- UB route sizes come FROM TilingData and are used verbatim ----
        // P-08 fix (TilingData.md §3.2, HostTiling.md §4.3): the host computes each
        // H-long route's aligned element count once, charges exactly those bytes in
        // CalcUbNeedBytes() and hands the counts down; the kernel no longer derives
        // any size from hidden_ (no align unit, no vector-register floor, no mirror
        // formula). ubRowElems_ rows hold the input dtype T, ubFp32Elems_ rows hold
        // fp32; both are element CAPACITIES, never counts — every Cast / Mul / Abs /
        // ReduceMax / Muls / reduction count stays the valid hidden_ (Kernel.md §9.3).
        ubRowElems_ = td_->ubRowElems;
        ubFp32Elems_ = td_->ubFp32Elems;

        // ---- R4-4 backstop, hardened by R5-1/R5-2: the two downlinked fields are
        //      re-established on the HOST'S OWN invariants, not on a flat ">= hidden" ----
        // REGISTER_TILINGDATA_SIZE(...) is sizeof(tiling_struct) and
        // GET_TILING_DATA_WITH_STRUCT copies exactly that many bytes: the tiling
        // buffer carries neither a length nor a version field. A kernel .o running
        // over an OLD host .so therefore reads 8 stale bytes here and the two values
        // go STRAIGHT into TPipe::InitBuffer below (huge allocation / hang, or an
        // under-sized route). This backstop therefore re-establishes, locally, the
        // three invariants the H-long chain really depends on — the ones R4-6/R4-7
        // defined on the host side in CalcUbTiers() / CalcSmallBufBytes():
        //   (i)   both fields are MULTIPLES of their own W, where
        //         W = max(blockSize, vectorSize) / sizeof(dtype) is the per-repeat
        //         register window of the consumers (16b: 128 T elements, 64 fp32
        //         elements) — R4-6's "window is a subset of the route" reduces to
        //         exactly this divisibility plus (ii), so a field that is >= hidden_
        //         but NOT a multiple of W is a violation, not a valid configuration
        //   (ii)  ubRowElems_  >= CeilAlign(max(hidden_, 1), W_T)     the T rows hold H
        //         ubFp32Elems_ >= CeilAlign(max(hidden_, 1), W_fp32)  values AND the
        //         ceil(hidden_/W)*W window every consumer touches fits (R4-6; this also
        //         subsumes R4-2's max(..., V) tail-window floor, since
        //         CeilAlign(h, V) >= V for every h >= 1)
        //   (iii) the BYTE SUM of the P routes fits TOTAL_UB_SIZE — the per-route cap
        //         alone is NOT enough (R5-2): two fields can each be under
        //         TOTAL_UB_SIZE/sizeof(dtype) and the PAIR still overshoot the pool,
        //         and release builds carry no TPipe::InitBuffer total assertion
        //         (kernel_tpipe_impl.h:395-400 is ASCENDC_CPU_DEBUG-only, the release
        //         path at :387-393 only advances curPoolAddr), so such a pair would let
        //         the last route silently cross the top of the UB pool
        // A violating field falls back to the value the host itself would have
        // downlinked: CeilAlign(max(hidden_, 1), W), saturated at the one-route cap. A
        // valid configuration is never touched and only a stale one is repaired.
        //
        // SCOPE — what this backstop does NOT buy: it repairs the two downlinked
        // fields, and it is only meaningful while hidden_ itself is in contract. If the
        // same stale host downlinked a hidden_ beyond the host's own UB bound
        // (CalcHiddenUpperBound(): 12608 here, 15808 on the fp32 path) then NO
        // allocation of these P routes is legal, and the backstop takes the FLOORS —
        // the smallest allocation that still holds hidden_ values. It deliberately does
        // NOT shrink a route below its floor: a route smaller than hidden_ would
        // silently truncate the H-long chain, turning a tiling error into a WRONG
        // RESULT, which is worse than the over-pool request. That residual (an
        // over-pool request in the already-impossible stale-hidden_ state) is
        // accepted as a documented residual.
        // No warning is emitted: this operator's kernel has no print / dump buffer
        // (A12 keeps the workspace at 0), so the violation is registered in
        // from the device.
        //
        // W and the small-route byte constants are compile-time and hold the SAME
        // numbers the host computes. The two device getters are exact platform-value
        // twins of the two host getters (no shared header exists, so equality is by
        // value, not by symbol):
        //   AscendC::GetVecLen()                == Ops::Base::GetVRegSize(context)    = 256 B
        //   AscendC::GetDataBlockSizeInBytes()  == Ops::Base::GetUbBlockSize(context) = 32 B
        // both live in asc/include/basic_api/kernel_operator_sys_var_intf.h (reading
        // VECTOR_REG_WIDTH / ONE_BLK_SIZE from asc/impl/basic_api/utils/
        // kernel_utils_constants.h), while the host pair is op_common/op_host/util/
        // platform_util.h. The equality is re-checked on the device — see LOG entry R5.
        constexpr uint32_t kUbBlockSize = AscendC::GetDataBlockSizeInBytes(); // 32 B
        constexpr uint32_t kRegBytes = AscendC::GetVecLen();                  // 256 B = V
        constexpr uint32_t kAlignBytes = (kUbBlockSize > kRegBytes) ? kUbBlockSize : kRegBytes;
        constexpr uint32_t kRowAlign = kAlignBytes / sizeof(T);               // W_T   = 128
        constexpr uint32_t kFp32Align = kAlignBytes / sizeof(float);          // W_f32 = 64
        constexpr uint32_t kRowCap = AscendC::TOTAL_UB_SIZE / sizeof(T);      // 126976
        constexpr uint32_t kFp32Cap = AscendC::TOTAL_UB_SIZE / sizeof(float); // 63488
        constexpr uint32_t kSumBytes = (INNER_OFFSET * sizeof(float) > kRegBytes) ?
                                           static_cast<uint32_t>(INNER_OFFSET * sizeof(float)) :
                                           kRegBytes; // 256 B
        constexpr uint32_t kZTmpBytes = (INNER_OFFSET * sizeof(T) > kRegBytes) ?
                                            static_cast<uint32_t>(INNER_OFFSET * sizeof(T)) :
                                            kRegBytes; // 256 B
        // The 6 block-level routes' byte sum, spelled with the SAME expressions the six
        // InitBuffer calls below use => 3 x 64 + 256 + 32 + 256 = 736 B, byte-identical
        // to the host's CalcSmallBufBytes(): on this platform blockSize == 32, so every
        // CeilAlign(INNER_OFFSET * sizeof(dtype), blockSize) the host applies is a no-op.
        constexpr uint32_t kSmallBytes =
            3u * (INNER_OFFSET * sizeof(int32_t)) + kSumBytes + (INNER_OFFSET * sizeof(T)) + kZTmpBytes; // 736 B

        const uint32_t hiddenElems = (hidden_ > 0) ? static_cast<uint32_t>(hidden_) : 1u;
        // The addition cannot overflow (hidden_ <= INT32_MAX) and the ceiling is
        // saturated at the one-route cap: a floor above the cap is not a legal route
        // either way (the host rejects such an H long before the kernel runs), and
        // saturating keeps the int32_t store and the InitBuffer byte count well defined.
        const uint32_t rowRaw = ((hiddenElems + kRowAlign - 1u) / kRowAlign) * kRowAlign;
        const uint32_t fp32Raw = ((hiddenElems + kFp32Align - 1u) / kFp32Align) * kFp32Align;
        const uint32_t rowFloor = (rowRaw > kRowCap) ? kRowCap : rowRaw;      // == host ubRowElems
        const uint32_t fp32Floor = (fp32Raw > kFp32Cap) ? kFp32Cap : fp32Raw; // == host ubFp32Elems

        // (i) + (ii), per tier — the "% W != 0" case is what the old flat "< hidden" test
        // missed: e.g. H = 100 with a stale ubFp32Elems_ = 104 (old host's
        // CeilAlign(H, 8)) passed the old guard while the consumer's window is
        // ceil(100/64)*64 = 128, i.e. 24 elements (96 B) past the route.
        if (ubRowElems_ <= 0 || (static_cast<uint32_t>(ubRowElems_) % kRowAlign) != 0u ||
            static_cast<uint32_t>(ubRowElems_) < rowFloor || static_cast<uint32_t>(ubRowElems_) > kRowCap) {
            ubRowElems_ = static_cast<int32_t>(rowFloor);
        }
        if (ubFp32Elems_ <= 0 || (static_cast<uint32_t>(ubFp32Elems_) % kFp32Align) != 0u ||
            static_cast<uint32_t>(ubFp32Elems_) < fp32Floor || static_cast<uint32_t>(ubFp32Elems_) > kFp32Cap) {
            ubFp32Elems_ = static_cast<int32_t>(fp32Floor);
        }
        // (iii) the TOTAL, in 64-bit so a stale-inflated pair cannot wrap: the sum of
        // the P routes of this branch exactly as the InitBuffer calls below request it
        // (2 T rows x 2 B + 4 fp32 rows x 4 B + the block-level group). Out of budget
        // => both tiers go to their (W-aligned) floors at once, which is the smallest
        // possible request that still satisfies (i) and (ii).
        {
            const uint64_t needBytes = 2ull * static_cast<uint64_t>(ubRowElems_) * sizeof(T) +
                                       4ull * static_cast<uint64_t>(ubFp32Elems_) * sizeof(float) +
                                       static_cast<uint64_t>(kSmallBytes);
            if (needBytes > static_cast<uint64_t>(AscendC::TOTAL_UB_SIZE)) {
                ubRowElems_ = static_cast<int32_t>(rowFloor);
                ubFp32Elems_ = static_cast<int32_t>(fp32Floor);
            }
        }

        // ---- quantities derived inside the kernel (never carried in TilingData) ----
        // taskNum = S * B as an int64 intermediate (B6: never int32 * int32).
        taskNum_ = static_cast<int64_t>(samples_) * static_cast<int64_t>(batchSize_);
        blockIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx());
        // Formulas (2-6)/(2-7); int64 holds startOffset_ + taskNumPerBlock_ which can
        // exceed INT32_MAX (L8), and the tail core is clamped to taskNum_.
        startOffset_ = blockIdx_ * static_cast<int64_t>(taskNumPerBlock_);
        endOffset_ = startOffset_ + static_cast<int64_t>(taskNumPerBlock_);
        if (endOffset_ > taskNum_) {
            endOffset_ = taskNum_;
        }

        // xCastBuf_ carries the S6b-scaled products into the compensated reduction
        // (S7) and is also its source route; S7's partial last chunk reads one whole
        // V-wide aligned window (V = GetVecLen() / sizeof(float) = 64 fp32 elements)
        // and that window must stay inside the route. The host-side ubFp32Elems tier
        // guarantees both facts for a host that downlinks its own CalcUbTiers() value
        // (CeilAlign'ed to exactly one vector register, so a multiple of V and >= V), and
        // the R5-hardened backstop above re-establishes the SAME two facts locally — a
        // multiple of W and >= CeilAlign(hidden_, W) — so the guarantee no longer rests
        // on "the host behaved": after Init() the pair is W-aligned, >= its floor and
        // within the UB pool for ANY downlinked values. (Condition: hidden_ itself in
        // contract, see the SCOPE note in the backstop.) Nothing further is clamped
        // here: the kernel allocates what the backstop settled on and passes the same
        // value to the VF as its window bound. Every UB window of this class — this VF's
        // and the SDK main-chain calls of S5/S6/S6b (one whole register per repeat, mask
        // on the store only) — is then contained in its route; for the T-row half of that
        // statement see the evidence grade recorded next to the S5/S6b chain (R5-11).

        // ---- GM binding (sizes from TilingData.md field consumption table) ----
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x),
                             static_cast<uint64_t>(samples_) * static_cast<uint64_t>(hidden_));
        yGradGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(y_grad),
            static_cast<uint64_t>(batchSize_) * static_cast<uint64_t>(capacity_) * static_cast<uint64_t>(hidden_));
        idxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indices),
                               static_cast<uint64_t>(samples_) * static_cast<uint64_t>(batchSize_));
        locGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(locations),
                               static_cast<uint64_t>(samples_) * static_cast<uint64_t>(batchSize_));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gates_grad),
                               static_cast<uint64_t>(samples_) * static_cast<uint64_t>(batchSize_));

        // ---- TBuf allocation: the 12 routes of DESIGN-BRANCH-0.md §4.3, depth 1 ----
        // All TBuf, so InitBuffer must use the two-argument form: the three-argument
        // form is static_assert'ed to TQue only. "Depth 1" means one block per route.
        // Sizes come from TilingData (ubRowElems_ / ubFp32Elems_), never from a
        // kernel-side derivation; × sizeof(dtype) recovers the exact byte count the
        // host charged in CalcUbNeedBytes().
        // Block-level small buffers: 6 routes, 736 B total (constant: INNER_OFFSET
        // slots of int32 / fp32 / T, independent of hidden_).
        // R4-7: the two routes the VECTOR engine READS (sumBuf_ by the S10 narrowing
        // Cast, zTmpBuf_ by the UB -> UB DataCopy) are floored at one whole vector
        // register, because on dav_3510 that load form carries no mask
        // (Reg::LoadAlign(vreg, addr)) and therefore always touches GetVecLen() bytes:
        // 64 B / 32 B routes would be over-read by 192 B / 224 B. The host applies the
        // same rule in CalcSmallBufBytes(blockSize, vectorSize, isFp32Path) and charges
        // exactly these bytes, so the budget and these InitBuffer calls stay one
        // arithmetic. The other four routes are only DMA'd or scalarly read, so they
        // keep the plain block alignment.
        // kRegBytes / kSumBytes / kZTmpBytes / kSmallBytes are declared with the
        // R4-4/R5 backstop above, so this allocation and the UB budget check above are
        // literally one arithmetic rather than two that happen to agree.
        pipe_->InitBuffer(idxBuf_, INNER_OFFSET * sizeof(int32_t)); // 64 B
        pipe_->InitBuffer(locBuf_, INNER_OFFSET * sizeof(int32_t)); // 64 B
        pipe_->InitBuffer(sumBuf_, kSumBytes);                    // 256 B fp32 accumulator block (whole-register read)
        pipe_->InitBuffer(tmpBuf_, INNER_OFFSET * sizeof(float)); // 64 B ReduceSum scalar out
        pipe_->InitBuffer(outBuf_, INNER_OFFSET * sizeof(T));     // 32 B write-back carrier
        pipe_->InitBuffer(zTmpBuf_, kZTmpBytes);                  // 256 B narrowing Cast staging (whole-register read)
        // H-long routes: 6 routes, 20H bytes total (2H + 2H + 4H + 4H + 4H + 4H).
        // The two row routes hold T (=2 B here); both tiers are CeilAlign'ed on the
        // host side to one whole vector register (128 T elements / 64 fp32 elements),
        // which makes every register window of the H-long chain a subset of its route
        // (R4-6; the block-level small buffers are sized by INNER_OFFSET, not by hidden,
        // and the two of them a vector op reads carry the R4-7 register floor).
        pipe_->InitBuffer(xBuf_, static_cast<uint32_t>(ubRowElems_) * sizeof(T)); // 2H low-precision x row
        pipe_->InitBuffer(yBuf_, static_cast<uint32_t>(ubRowElems_) * sizeof(T)); // 2H low-precision y_grad row
        pipe_->InitBuffer(xCastBuf_, static_cast<uint32_t>(ubFp32Elems_) * sizeof(float)); // 4H (S6b scale / S7 src)
        pipe_->InitBuffer(yCastBuf_, static_cast<uint32_t>(ubFp32Elems_) * sizeof(float)); // 4H fp32 widen target
        pipe_->InitBuffer(mulBuf_, static_cast<uint32_t>(ubFp32Elems_) * sizeof(float));   // 4H fp32 product
        // workBuf_ is S6b's ReduceMax sharedTmpBuffer ARGUMENT, not a route this
        // kernel reads: on dav_3510 that parameter is not consumed at all
        // (ReduceMaxImpl -> ReduceNoIndexTemplate<...>(dst, src, workLocal, ...), whose
        // body never references workLocal — dav_3510/kernel_operator_vec_reduce_impl.h
        // :871-898), so on this device the route is a DEAD PATH. The API requires the
        // argument, and the accounting direction is the safe one (4H charged, nothing
        // read), so the route stays and keeps taking its size from TilingData — the
        // budget and this InitBuffer call remain one arithmetic (P-08).
        pipe_->InitBuffer(workBuf_, static_cast<uint32_t>(ubFp32Elems_) *
                                        sizeof(float)); // 4H ReduceMax sharedTmpBuffer (unused on dav_3510)

        // ---- LocalTensor handles (TBuf form: no AllocTensor, take once here) ----
        idxLocal_ = idxBuf_.Get<int32_t>();
        locLocal_ = locBuf_.Get<int32_t>();
        sumLocal_ = sumBuf_.Get<float>();
        tmpLocal_ = tmpBuf_.Get<float>();
        outLocal_ = outBuf_.Get<T>();
        zTmpLocal_ = zTmpBuf_.Get<T>();
        xLocal_ = xBuf_.Get<T>();
        yLocal_ = yBuf_.Get<T>();
        xCastLocal_ = xCastBuf_.Get<float>();
        yCastLocal_ = yCastBuf_.Get<float>();
        mulLocal_ = mulBuf_.Get<float>();
        workLocal_ = workBuf_.Get<float>();

        // ---- DataCopyPad parameters (byte precision, B7) ----
        // rsv has no default in the 5-argument constructor, so it is set explicitly.
        xCopyParams_.blockCount = 1;
        xCopyParams_.blockLen = static_cast<uint32_t>(hidden_) * static_cast<uint32_t>(sizeof(T));
        xCopyParams_.srcStride = 0;
        xCopyParams_.dstStride = 0;
        xCopyParams_.rsv = 0;
        yCopyParams_ = xCopyParams_; // the y_grad slot row has the same shape

        // blockLen of the three staged copies is rewritten per block from innerTask
        // (§5.2 / §5.4): a tail block must never transfer INNER_OFFSET elements.
        idxCopyParams_.blockCount = 1;
        idxCopyParams_.blockLen = 0;
        idxCopyParams_.srcStride = 0;
        idxCopyParams_.dstStride = 0;
        idxCopyParams_.rsv = 0;
        locCopyParams_ = idxCopyParams_;
        outCopyParams_ = idxCopyParams_;

        // No SetFlag/WaitFlag event pair is used anywhere in this branch: every
        // crossing is covered by the PipeBarrier calls of §5.5, so there is no
        // eventID to allocate (and none may be hard-coded).
    }

    // -----------------------------------------------------------------------
    // Process — block loop plus the explicit H == 0 short circuit (§3 / §5.5).
    //
    // Every core owns the flat task range [startOffset_, endOffset_); an empty
    // tensor (taskNum_ == 0) leaves the loop with zero iterations and writes
    // nothing (returns_empty, Kernel.md §9.7).
    // -----------------------------------------------------------------------
    __aicore__ inline void Process()
    {
        if (startOffset_ >= taskNum_) {
            return; // tail core with no task at all (formula (2-7))
        }

        // ---- explicit early short circuit for H == 0 (Kernel.md §9.9) ----
        // H = 0 is a legal degenerate input: the empty inner product is 0, so the
        // whole block is written back as 0. ReduceSum must NOT be called (its
        // arch35 implementation carries ASCENDC_CHECK_VALUE_RANGE(count, 1, ...),
        // so count = 0 is out of range) and no row DataCopyPad may be issued
        // (blockLen has to be in [1, 2097151], so a 0-byte transfer is illegal).
        // The test is `<= 0` and not `== 0` as a last line of defence: the host
        // rejects unresolved negative dims (HostTiling §3.3 item 6, A1-7), and a
        // negative hidden would otherwise reach the counts through uint32_t.
        if (hidden_ <= 0) {
            for (int64_t taskOffset = startOffset_; taskOffset < endOffset_; taskOffset += INNER_OFFSET) {
                const int32_t innerTask = InnerTaskOf(taskOffset);
                AscendC::Duplicate<float>(sumLocal_, 0.0f, INNER_OFFSET);
                CopyOut(taskOffset, innerTask); // narrowing Cast + write back 0 only
            }
            return;
        }

        for (int64_t taskOffset = startOffset_; taskOffset < endOffset_; taskOffset += INNER_OFFSET) {
            const int32_t innerTask = InnerTaskOf(taskOffset);
            CopyInIdxLoc(taskOffset, innerTask); // block header (S1)
            AscendC::PipeBarrier<PIPE_ALL>();    // #1  MTE2 -> S (scalar decode)
            ComputeBlock(taskOffset, innerTask); // S2..S8 (#7/#2/#3/#4/#9 inside)
            CopyOut(taskOffset, innerTask);      // S9/S10  (#6+#10 / #5 inside)
        }
    }

private:
    // -----------------------------------------------------------------------
    // InnerTaskOf — valid task count of the inner block starting at taskOffset.
    //
    // Formula (2-8): min(INNER_OFFSET, taskNum - taskOffset). Its only two uses
    // are the inner loop bound and the CopyOut blockLen (B3): writing back
    // INNER_OFFSET elements on a tail block would run past S*B.
    // -----------------------------------------------------------------------
    __aicore__ inline int32_t InnerTaskOf(int64_t taskOffset) const
    {
        const int64_t remaining = taskNum_ - taskOffset;
        return (remaining < static_cast<int64_t>(INNER_OFFSET)) ? static_cast<int32_t>(remaining) : INNER_OFFSET;
    }

    // -----------------------------------------------------------------------
    // CopyInIdxLoc — chain step S1: the block's indices / locations (§5.2).
    //
    // blockLen = innerTask * sizeof(int32_t), NOT INNER_OFFSET * sizeof(int32_t):
    // a constant 16 would read past S*B on every tail block
    // (paradigms/common/datacopypad-rules.md:141). The UB side still holds
    // INNER_OFFSET slots, of which only the first innerTask are valid.
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyInIdxLoc(int64_t taskOffset, int32_t innerTask)
    {
        idxCopyParams_.blockLen = static_cast<uint32_t>(innerTask) * static_cast<uint32_t>(sizeof(int32_t));
        locCopyParams_.blockLen = idxCopyParams_.blockLen;
        AscendC::DataCopyPad(idxLocal_, idxGm_[static_cast<uint64_t>(taskOffset)], idxCopyParams_,
                             AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        AscendC::DataCopyPad(locLocal_, locGm_[static_cast<uint64_t>(taskOffset)], locCopyParams_,
                             AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
    }

    // -----------------------------------------------------------------------
    // CopyInRowPair — chain steps S3/S4: the x token row and the y_grad slot row.
    //
    // Called only for a task that passed the `ok` mask, so an illegal routing
    // never issues a DMA. token = t mod S (A7), slot = e * C + p computed in
    // int64 (B6). The UB tail beyond ubRowElems_ stays stale (isPad = false, i.e.
    // no padding value is written); it is never consumed (B7, Kernel.md §9.3) — since the
    // R4-6 fix the host aligns ubRowElems_ to one whole vector register of T elements,
    // so the register windows of the S5/S6/S6b consumers stay inside the route and no
    // lane past hidden_ is consumed at all (see the note before S5, which states the
    // evidence grade of the T-row half of that claim: an SDK per-repeat contract, not a
    // measured ISA identity — R5-11).
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyInRowPair(int64_t taskIndex, int32_t e, int32_t p)
    {
        const int64_t token = taskIndex % static_cast<int64_t>(samples_);
        const int64_t slot = static_cast<int64_t>(e) * static_cast<int64_t>(capacity_) + static_cast<int64_t>(p);
        AscendC::DataCopyPad(xLocal_, xGm_[static_cast<uint64_t>(token * static_cast<int64_t>(hidden_))], xCopyParams_,
                             AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
        AscendC::DataCopyPad(yLocal_, yGradGm_[static_cast<uint64_t>(slot * static_cast<int64_t>(hidden_))],
                             yCopyParams_, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
    }

    // -----------------------------------------------------------------------
    // ComputeBlock — the per-block compute chain, steps S2..S8 (§5.3).
    //
    // Held routes: sumBuf / idxBuf / locBuf (block level) + xBuf + yBuf +
    // xCastBuf + yCastBuf + mulBuf + workBuf + tmpBuf (task level) = 10; the
    // remaining 2 of P = 12 (outBuf, zTmpBuf) are the CopyOut carriers, so the
    // peak residency over the whole block is exactly P = 12 (DESIGN-BRANCH-0 §5.3).
    // -----------------------------------------------------------------------
    __aicore__ inline void ComputeBlock(int64_t taskOffset, int32_t innerTask)
    {
        // ---- S9 zero mask (A8/B2): every block starts with all 16 slots at 0, so
        // an illegal task is 0 without any per-task write-back ----
        AscendC::Duplicate<float>(sumLocal_, 0.0f, INNER_OFFSET);

        for (int32_t i = 0; i < innerTask; ++i) {
            // #7 V -> MTE2 (WAR, across tasks and across blocks): the next task's
            // DMA may overwrite xBuf_/yBuf_ only after this core finished reading
            // them. Single buffering, so this crossing must not be dropped.
            AscendC::PipeBarrier<PIPE_ALL>();

            // ---- S2 scalar address decode + the four-condition `ok` mask ----
            const int32_t e = idxLocal_.GetValue(static_cast<uint32_t>(i)); // indices[t]
            const int32_t p = locLocal_.GetValue(static_cast<uint32_t>(i)); // locations[t]
            // ok = (e >= 0) && (e < B) && (p >= 0) && (p < C) — all four terms from
            // spec.yaml math_semantics.formula; an illegal task keeps its 0 slot.
            const bool ok = (e >= 0) && (e < batchSize_) && (p >= 0) && (p < capacity_);
            if (!ok) {
                continue;
            }

            // #2 S -> MTE2: token/slot are scalar results and must be ordered before
            // the two task-level DataCopyPad calls below. Required for ordering, not
            // an optional optimisation.
            AscendC::PipeBarrier<PIPE_ALL>();
            CopyInRowPair(taskOffset + static_cast<int64_t>(i), e, p); // S3/S4
            // #3 MTE2 -> V: a second, independent barrier after the two DMA calls and
            // before their consumers. #2 and #3 sit on opposite sides of the same
            // pair of DataCopyPad calls, so they can never be merged into one.
            AscendC::PipeBarrier<PIPE_ALL>();

            // ---- S5 pre-elewise widening Cast (Kernel.md §9.1) ----
            // count is the valid hidden_, never ubRowElems_/ubFp32Elems_ (Kernel.md §9.3).
            //
            // R4-6 fix (09-15) — WHY THE TAIL IS NOT READ ANY MORE. On dav_3510 the
            // vector instructions of the S5/S6/S6b group keep their mask on the STORE
            // only and load a whole register per repeat:
            //   * widening Cast  GenLoadL2 -> Reg::LoadAlign<T, DIST_UNPACK_B16>, no
            //     mask (dav_3510/kernel_operator_vec_vconv_impl.h:216-231, called at
            //     :275 with oneRepSize = GetVecLen()/sizeof(float) = 64 source elements)
            //   * S6 Mul VF and S6b Abs / Muls / ReduceMax: Reg::LoadAlign without a
            //     mask (this file's MoeTutelCombineGatesMulVF and the SDK entries)
            // The consumed source range is therefore ceil(hidden_/W)*W, W = one vector
            // register counted in the elements of the route being read (W = 64 for the
            // fp32 rows, which are loaded as fp32; W = 64 for the T rows too, because the
            // widening cast's repeat is counted in fp32 lanes).
            //
            // ★ EVIDENCE GRADE OF THE T-ROW HALF OF THAT CLAIM (R5-11 — stated at the
            // strength actually available). For the fp32 rows W = 64 is CONFIRMED from
            // the SDK level-2 template (repeatStride = GetVecLen()/sizeof(T), LoadAlign
            // unmasked, mask only on StoreAlign). For the T rows the single consumer is
            // the widening Cast below, whose Reg::LoadAlign<T, DIST_UNPACK_B16> has no
            // semantic documentation anywhere in the CANN 9.2.0 tree, and whose ISA
            // footprint could not be measured here: the driver refuses -emit-llvm /
            // -print-after, and the shipped .o disassembly shows only
            // `VLD V0,[S10],A1,#14`, whose mode-#14 read width is not decidable from the
            // mnemonic. What IS established is the SDK-level contract this whole chain is
            // written against — one repeat consumes W = GetVecLen()/sizeof(DST) = 64
            // source elements (dav_3510/kernel_operator_vec_vconv_impl.h:251-296) — and
            // the H bound is 12608 under BOTH readings of #14 (re-computed on the
            // CeilAlign(h, 64) grid as well), so there is no capacity risk either way. The
            // wording is therefore an SDK-contract statement, NOT an ISA-level "not one
            // byte outside the route is ever read" identity. A sentinel experiment that
            // would upgrade it to "measured" has not been run.
            // Before this fix the host aligned the two tiers only to the UB block
            // (16 / 8 elements), so that window could reach past the route by up to
            // 48 elements (96 B) for the T rows and 56 elements (224 B) for the fp32
            // rows (worst case hidden_ = 64k+1). The host now aligns BOTH tiers to W
            // (128 T elements / 64 fp32 elements, CalcUbTiers() in
            // op_host/moe_tutel_combine_gates_tiling_arch35.cpp; the R4-4/R5
            // backstop in Init() re-establishes the same invariant locally, so this holds
            // for a stale host too), so
            //     ceil(hidden_/W)*W <= ubRowElems_  and  <= ubFp32Elems_
            // holds for every legal hidden_ under that per-repeat contract: the consumed
            // window is a SUBSET of the route, no lane past hidden_ is consumed by the
            // chain, and the "UB tail is never read" statement of Kernel.md §9.3 holds
            // again. NOTE the same alignment is what contains the widening Cast's
            // STORE-side tail: in the fused VF the tail repeat's predicate comes from the
            // entry `count` (the loop body does not decrement it), so the stored tail
            // window can likewise reach 64 fp32 lanes — harmless today because
            // ubFp32Elems_ is aligned to exactly that grid, but it means the fp32 tier's
            // alignment must never be relaxed back to blockSize.
            // The price is H capacity — the binding bound falls by 8 elements on this path
            // and by 40 on the fp32 path with R4-6, and R4-7's larger small-route group
            // takes the 16b bound down a FURTHER 64: 12672 -> 12608 (the fp32 bound is
            // unchanged at 15808, its +192 B increase being absorbed by the 704 B slack
            // that point had) — accepted deliberately; the exact old/new contract values,
            // the per-route table and the measured byte accounts are recorded in
            // the host UB budget note
            // (op_host/moe_tutel_combine_gates_tiling_arch35.cpp, CalcUbTiers /
            //  CalcUbNeedBytes).
            AscendC::Cast(xCastLocal_, xLocal_, RoundMode::CAST_NONE, static_cast<uint32_t>(hidden_));
            AscendC::Cast(yCastLocal_, yLocal_, RoundMode::CAST_NONE, static_cast<uint32_t>(hidden_));

            // ---- S6 element-wise multiply in fp32 (VF, register resident) ----
            MulFp32OnMainChain(mulLocal_, xCastLocal_, yCastLocal_, static_cast<uint32_t>(hidden_));

            // ---- S6b reduction-scale normalization (numerics hardening, see the
            // Pow2Of / ReductionShiftOf header note) ----
            // maxAbs = max_h |x_h * y_h|. xCastBuf/yCastBuf are dead after S6, so both
            // are reused here and no UB route is added.
            AscendC::PipeBarrier<PIPE_V>(); // V -> V: S6 -> Abs
            AscendC::Abs(yCastLocal_, mulLocal_, static_cast<int32_t>(hidden_));
            AscendC::PipeBarrier<PIPE_V>(); // V -> V: Abs -> ReduceMax
            AscendC::ReduceMax<float>(tmpLocal_, yCastLocal_, workLocal_, static_cast<int32_t>(hidden_));
            AscendC::PipeBarrier<PIPE_ALL>(); // V -> S: the max is consumed in the scalar unit
            const int32_t reductionShift = ReductionShiftOf(tmpLocal_.GetValue(0u));
            AscendC::PipeBarrier<PIPE_ALL>(); // S -> V: the scale is a scalar operand of Muls
            AscendC::Muls(xCastLocal_, mulLocal_, Pow2Of(-reductionShift), static_cast<int32_t>(hidden_));

            // #4 V -> V: the reduction reads the scaled products written by the Muls above
            // (RAW within the V pipe, same requirement as the original Mul -> ReduceSum).
            AscendC::PipeBarrier<PIPE_V>();
            // ---- S7 compensated reduction along H (Task 30 accuracy fix) ----
            // Replaces DESIGN-BRANCH-0/2.md §5.3's single `ReduceSum`: an fp32
            // pairwise fold cannot reach the correctly rounded value on the
            // cancellation-heavy tasks of the low-precision paths (measured
            // 20 fp16 quanta of deviation; see the VF comment above). src =
            // the S6b-scaled products in xCastBuf_, dst = tmpBuf_[0] (cacheCount
            // = 1: R is never split, so the tree root is the only slot), count =
            // hidden_ >= 1 (H = 0 took the short circuit in Process()). No UB
            // scratch is used, so the old `workBuf_` scratch role of §9.4 (now
            // serving ReduceMax in S6b) is untouched.
            CompensatedReduceOnMainChain(xCastLocal_, tmpLocal_, static_cast<uint32_t>(hidden_));

            // #9 V -> S: tmpBuf_[0] is written by the vector engine and read by the
            // scalar engine; that visibility has no implicit ordering.
            AscendC::PipeBarrier<PIPE_ALL>();
            // ---- S8 masked slot write (scale back by the exact power of two: exact
            // unless the sum itself leaves the fp32 range, which is then a genuine
            // overflow and must stay +/-Inf, not a clamped finite value) ----
            sumLocal_.SetValue(static_cast<uint32_t>(i), tmpLocal_.GetValue(0u) * Pow2Of(reductionShift));
        }
    }

    // -----------------------------------------------------------------------
    // MulFp32OnMainChain — S6 dispatch: the kernel-side VF wrapper of
    // MoeTutelCombineGatesMulVF. Kept as its own helper so the repeat geometry
    // (GetVecLen-based) is computed in ordinary kernel code, next to the
    // PipeBarrier that orders it against the surrounding vector work.
    // -----------------------------------------------------------------------
    __aicore__ inline void MulFp32OnMainChain(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                              const LocalTensor<float> &src1, uint32_t count)
    {
        constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(float);
        __ubuf__ float *dstAddr = reinterpret_cast<__ubuf__ float *>(dst.GetPhyAddr());
        __ubuf__ float *src0Addr = reinterpret_cast<__ubuf__ float *>(src0.GetPhyAddr());
        __ubuf__ float *src1Addr = reinterpret_cast<__ubuf__ float *>(src1.GetPhyAddr());
        asc_vf_call<MoeTutelCombineGatesMulVF<float>>(dstAddr, src0Addr, src1Addr, count, oneRepeatSize);
    }

    // -----------------------------------------------------------------------
    // CompensatedReduceOnMainChain — S7 dispatch: the kernel-side VF wrapper of
    // MoeTutelCombineGatesCompensatedReduceExactVF. Kept as its own helper so the
    // register geometry (GetVecLen-based) is computed in ordinary kernel code,
    // next to the PipeBarrier that orders it against the surrounding vector work.
    // -----------------------------------------------------------------------
    __aicore__ inline void CompensatedReduceOnMainChain(const LocalTensor<float> &src, const LocalTensor<float> &dst,
                                                        uint32_t count)
    {
        constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(float);
        __ubuf__ float *srcAddr = reinterpret_cast<__ubuf__ float *>(src.GetPhyAddr());
        __ubuf__ float *dstAddr = reinterpret_cast<__ubuf__ float *>(dst.GetPhyAddr());
        asc_vf_call<MoeTutelCombineGatesCompensatedReduceExactVF<float>>(srcAddr, dstAddr, count, oneRepeatSize,
                                                                         static_cast<uint32_t>(ubFp32Elems_));
    }

    // -----------------------------------------------------------------------
    // CopyOut — narrowing Cast plus the single dense write-back path (§5.4).
    //
    // Unconditional: every block is written back, including a block whose tasks
    // are all illegal (B2 — the slots then hold the S9 zeros, and the reference
    // implementation's `if (moveCounts > 0)` behaviour is deliberately not kept).
    // blockLen = innerTask * sizeof(T) (B3: never past S*B).
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyOut(int64_t taskOffset, int32_t innerTask)
    {
        // Entry barrier covering two crossings at once:
        //   #6  MTE3 -> V : this block's Cast may overwrite zTmpBuf_/outBuf_ only
        //                   after the previous block's DataCopyPad finished reading them
        //   #10 S -> V    : sumBuf_ is written slot by slot by the scalar engine and
        //                   read as a whole by the Cast below
        AscendC::PipeBarrier<PIPE_ALL>();

        // ---- S10 narrowing Cast: fp32 accumulator -> T ----
        // CAST_ROUND (round to nearest) is mandated by spec.yaml / A9 and by the
        // reference implementation; CAST_RINT must not be mixed in (Interface.md
        // "special value semantics"). The conversion is element-wise with no
        // cross-lane dependency, so converting all INNER_OFFSET slots is safe
        // (Kernel.md §9.6 mechanism 3).
        //
        // FIXED (R4-7; was registered as an open item by R4-6 and is now closed): the
        // SDK's narrowing cast uses oneRepSize = GetVecLen()/sizeof(SRC) = 64 and loads
        // its source with a whole-register, UNMASKED Reg::LoadAlign
        // (CastIntrinsicsImplVF + GenLoadL2, dav_3510/kernel_operator_vec_vconv_impl.h
        // :251-296 / :216-231 / :275), i.e. it touches 256 B = 64 fp32 elements from the
        // base of sumBuf_. With the old 64 B route that single read left the route by
        // 192 B, and the UB -> UB DataCopy below (CopyImpl level 2 in
        // dav_3510/kernel_operator_data_copy_impl.h:1043-1063, also
        // Reg::LoadAlign(vreg, src) with no mask) left the old 32 B zTmpBuf_ by 224 B.
        // Both routes are now floored at one whole vector register in Init()
        // (kSumBytes / kZTmpBytes = 256 B) and the host charges the same bytes in
        // CalcSmallBufBytes(), so window = 256 B <= capacity = 256 B holds for both:
        // neither read leaves its own route any more. The old "the window lands in the
        // pool because >=1664 B of routes follow it" argument is therefore retired, not
        // relied upon. Numerically nothing changed: the store of both ops is masked to
        // the 16 valid lanes, so the extra lanes were and remain dead. The UB-offset
        // level check is still not available on this machine (same limitation as
        // the TilingData ABI note in op_host/moe_tutel_combine_gates_tiling_arch35.cpp); the containment is now an
        // arithmetic identity instead of an argument about the allocation order. See the byte accounts in the host UB
        // budget note.
        AscendC::Cast(zTmpLocal_, sumLocal_, RoundMode::CAST_ROUND, static_cast<uint32_t>(INNER_OFFSET));

        // ---- UB -> UB reshaping of the [16] result block ----
        // DataCopy with the element-count overload; a UB -> UB DataCopyPad has no
        // three-argument form in the SDK (the LocalTensor/LocalTensor overloads all
        // need Nd2NzParams), so this is the only legal spelling.
        AscendC::DataCopy(outLocal_, zTmpLocal_, static_cast<uint32_t>(INNER_OFFSET));

        // #5 V -> MTE3: zTmpBuf_/outBuf_ are written by the vector engine and read
        // by the MTE3 copy below. The barrier must sit between the write and the
        // read; placing it before CopyOut() would be ineffective.
        AscendC::PipeBarrier<PIPE_ALL>();

        outCopyParams_.blockLen = static_cast<uint32_t>(innerTask) * static_cast<uint32_t>(sizeof(T));
        AscendC::DataCopyPad(outGm_[static_cast<uint64_t>(taskOffset)], outLocal_, outCopyParams_);
    }

private:
    // ---- TilingData and the quantities derived from it ----
    const MoeTutelCombineGatesTilingData *td_ = nullptr;
    AscendC::TPipe *pipe_ = nullptr;
    int32_t samples_ = 0;         // S = x.shape[0]
    int32_t hidden_ = 0;          // H = reduction axis length
    int32_t capacity_ = 0;        // C = y_grad.shape[1]
    int32_t batchSize_ = 0;       // B = y_grad.shape[0]
    int32_t taskNumPerBlock_ = 0; // tasks per core (already folded into the key)
    int64_t taskNum_ = 0;         // S * B (int64 intermediate, B6)
    int64_t blockIdx_ = 0;
    int64_t startOffset_ = 0; // formula (2-6)
    int64_t endOffset_ = 0;   // formula (2-7)
    int32_t ubRowElems_ = 0;  // elements per input-dtype H-long route (from TilingData)
    int32_t ubFp32Elems_ = 0; // elements per fp32 H-long route (from TilingData, >= 1 vector register)

    // ---- GM views ----
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGradGm_;
    GlobalTensor<T> outGm_;
    GlobalTensor<int32_t> idxGm_;
    GlobalTensor<int32_t> locGm_;

    // ---- the 12 UB routes, all TBuf (no TQue anywhere) ----
    TBuf<TPosition::VECCALC> idxBuf_;
    TBuf<TPosition::VECCALC> locBuf_;
    TBuf<TPosition::VECCALC> sumBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> outBuf_;
    TBuf<TPosition::VECCALC> zTmpBuf_;
    TBuf<TPosition::VECCALC> xBuf_;
    TBuf<TPosition::VECCALC> yBuf_;
    TBuf<TPosition::VECCALC> xCastBuf_;
    TBuf<TPosition::VECCALC> yCastBuf_;
    TBuf<TPosition::VECCALC> mulBuf_;
    TBuf<TPosition::VECCALC> workBuf_;

    // ---- handles taken once in Init (TBuf has no Alloc/Free) ----
    LocalTensor<int32_t> idxLocal_;
    LocalTensor<int32_t> locLocal_;
    LocalTensor<float> sumLocal_;
    LocalTensor<float> tmpLocal_;
    LocalTensor<T> outLocal_;
    LocalTensor<T> zTmpLocal_;
    LocalTensor<T> xLocal_;
    LocalTensor<T> yLocal_;
    LocalTensor<float> xCastLocal_;
    LocalTensor<float> yCastLocal_;
    LocalTensor<float> mulLocal_;
    LocalTensor<float> workLocal_;

    // ---- DataCopyPad parameters (blockLen is in BYTES) ----
    DataCopyExtParams xCopyParams_;
    DataCopyExtParams yCopyParams_;
    DataCopyExtParams idxCopyParams_;
    DataCopyExtParams locCopyParams_;
    DataCopyExtParams outCopyParams_;
};

#endif // MOE_TUTEL_COMBINE_GATES_16B_H_
