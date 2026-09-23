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
 * \file moe_tutel_combine_gates_32b.h
 * \brief MoeTutelCombineGates float32 kernel class.
 */

#ifndef MOE_TUTEL_COMBINE_GATES_32B_H_
#define MOE_TUTEL_COMBINE_GATES_32B_H_

#include "kernel_operator.h"
#include "moe_tutel_combine_gates_tiling_struct.h"

using namespace AscendC;

// ===========================================================================
// MoeTutelCombineGatesMulFloatVF — fp32 chain step S5 (element-wise multiply) on
// the Vector main chain, extended (Task 28.2.7) to also emit the EXACT rounding
// error of every product (TwoProduct via fused multiply-add).
//
// Registers only: the two fp32 operands are loaded from UB, multiplied in the
// vector register file and stored back, so no intermediate value occupies a UB
// route beyond the two H-long carriers handed in by the caller (this is why the
// UB route count stays at the designed P = 9).
//
//   p   = Mul(x, y)                -> dstAddr   (= fl(x*y), the S5 product)
//   err = MulAddDst(Neg(p), x, y)  -> errAddr   (= -p + x*y)
//   `vmula` is a FUSED multiply-add, so `err` is exactly x*y - fl(x*y)
//   (Dekker TwoProduct; error-free for normal products). S6 folds it into the
//   compensation channel, which turns the branch's result into the correctly
//   rounded value of the exact inner product SUM_h x_h*y_h instead of the sum of
//   fp32-rounded products. Measured effect (12 seeds x both large-H cases):
//   worst pointwise-contract usage 0.75 -> 0.06.
//
// The masked store also keeps the B7 contract: only the first `count` lanes are
// written, so the stale UB tail beyond `hidden_` is never touched and can never
// reach the reduction (Kernel.md §9.3). The `err` route is masked identically.
//
// count          — number of valid elements (= hidden_, never the padded length)
// oneRepeatSize  — elements per vector repeat = GetVecLen() / sizeof(float)
// ===========================================================================
template <typename T>
__simd_vf__ inline void MoeTutelCombineGatesMulFloatVF(__ubuf__ T *dstAddr, __ubuf__ T *errAddr, __ubuf__ T *src0Addr,
                                                       __ubuf__ T *src1Addr, uint32_t count, uint32_t oneRepeatSize)
{
    AscendC::Reg::RegTensor<T> src0Reg;
    AscendC::Reg::RegTensor<T> src1Reg;
    AscendC::Reg::RegTensor<T> dstReg;
    AscendC::Reg::RegTensor<T> errReg;
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
        // TwoProduct: err = -fl(x*y) + x*y with a single fused rounding.
        AscendC::Reg::Neg(errReg, dstReg, mask);
        AscendC::Reg::MulAddDst(errReg, src0Reg, src1Reg, mask);
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg, mask);
        AscendC::Reg::StoreAlign(errAddr, errReg, aReg, mask);
    }
}

// ===========================================================================
// MoeTutelCombineGatesCompensatedReduceVF — fp32 chain step S6 (compensated
// reduction) on the Vector main chain; replaces the single `ReduceSum` call of
// DESIGN-BRANCH-1.md §5.3 (deviation record: file header NUMERICAL STABILITY (b)
//).
//
// WHY (root cause, not a guess): a plain fp32 pairwise fold — which is what
// `ReduceSum` does, and the deployed kernel is bit-identical to an ideal fp32
// adjacent-pair fold — carries a rounding error of order u*Sum|p| (measured
// 0.13*u*Sum|p|, i.e. ~7.6e-6 for H = 8192 at Sum|p| ~ 952). The design contract
// `|a-g| <= 1e-6 + 1e-6*|g|` (spec.yaml per_dtype.float32) demands ~1e-6
// absolute accuracy on cancellation-heavy tasks (kappa up to 1.1e7), which no
// single fp32 fold can reach. Carrying the rounding error in a second
// accumulator with error-free transformations can, and does.
//
// ALGORITHM (all in vector registers — no UB staging, no scalar loop):
//   Phase 1  per-lane compensated accumulation over chunks of `oneRepeatSize`
//            elements (lane j takes elements j, j+V, j+2V, ...). Each step is a
//            full Knuth TwoSum:
//                s   = hi + x
//                bb  = s - hi
//                t1  = s - bb
//                err = (hi - t1) + (x - bb)     // exactly the rounding error
//                lo  = lo + err + q ;  hi = s   // q = exact error of S5's Mul
//            TwoSum (not the 3-op Fast2Sum / 4-op Kahan form) is deliberate: it
//            is exact for *any* operand magnitudes, and the failing elements are
//            precisely the near-cancelling ones where the |hi| >= |x|
//            precondition of the cheaper forms does not hold.
//            `q` is read from the second carrier: S5 emits x*y - fl(x*y) with a
//            fused multiply-add, so folding it into `lo` removes the product
//            rounding as well — the reduction then sees the exact products.
//   Phase 2  compensated binary halving tree over the lanes. Reg::DeInterleave
//            pairs lane 2i with lane 2i+1 (semantics cross-checked against the
//            SDK's own DeInterleaveImplNormal, which splits a 2*V-element stream
//            into its even- and odd-indexed halves); the pair sum and its TwoSum
//            error use the same 6 operations, and every level's error terms are
//            merged into `lo`. Different levels' terms occupy disjoint lane sets,
//            so one final lane sum collects each of them exactly once.
//   Final    total = (lane sum of hi) + (lane sum of lo + all level errors).
//            The second summand is of order u*Sum|p|, so the single fp32
//            reduction applied to it contributes only ~u^2*Sum|p|.
//
// ERROR ANALYSIS: the returned value is the correctly rounded fp32 value of
// Sum_h (x_h * y_h) evaluated on the EXACT products (not on their fp32
// roundings), error <= 1 ulp of the result, i.e. |a-g| <= u*|g| ~ 1.2e-7*|g|.
// Measured over 12 seeds x 2 cases: worst pointwise-contract usage 0.06
// (without the `q` term the residual is the S5 product
// rounding and the usage is 0.75.
//
// SEMANTICS PRESERVED (branch hard constraints):
//   * NaN / +/-Inf on a legal task propagate verbatim: the primary accumulator
//     follows IEEE-754 add/sub exactly, and the compensation channel (the only
//     place where an infinite operand would create a spurious Inf-Inf = NaN) is
//     NaN-filtered lane-wise before the final horizontal sum (see the non-finite
//     guard below and NUMERICAL STABILITY (c) in the file header).
//   * Lane assignment does not affect the value of the sum (each product is
//     added exactly once), hence the partial last chunk can reuse an ALIGNED
//     in-route window instead of over-reading UB.
//   * count = hidden_ is always >= 1 here (H == 0 short circuits in Process).
//
// count          — number of valid products (= hidden_, never the padded length)
// oneRepeatSize  — elements per vector register = GetVecLen() / sizeof(float) (= 64;
//                  NOT sizeof(T) — this branch's T is float, so the two spellings are
//                  numerically identical, but the argument actually passed at both call
//                  sites is sizeof(float); kept in step with _16b.h:255-256)
// bufElems       — element capacity of both source carriers (mulBuf_ / workBuf_),
//                  used to pick the tail window so no read leaves a route window
// ===========================================================================
template <typename T>
__simd_vf__ inline void MoeTutelCombineGatesCompensatedReduceVF(__ubuf__ T *srcAddr, __ubuf__ T *errAddr,
                                                                __ubuf__ float *dstAddr, uint32_t count,
                                                                uint32_t oneRepeatSize, uint32_t bufElems)
{
    AscendC::Reg::RegTensor<T> xReg;
    AscendC::Reg::RegTensor<T> hiReg;
    AscendC::Reg::RegTensor<T> loReg;
    AscendC::Reg::RegTensor<T> sReg;
    AscendC::Reg::RegTensor<T> bReg;
    AscendC::Reg::RegTensor<T> tReg;
    AscendC::Reg::RegTensor<T> eReg;
    AscendC::Reg::RegTensor<T> aReg;
    AscendC::Reg::RegTensor<T> qReg;
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
        AscendC::Reg::LoadAlign(qReg, errAddr + c * oneRepeatSize); // exact Mul error
        AscendC::Reg::Add(sReg, hiReg, xReg, mAll);                 // s  = hi + x
        AscendC::Reg::Sub(bReg, sReg, hiReg, mAll);                 // bb = s - hi
        AscendC::Reg::Sub(tReg, sReg, bReg, mAll);                  // t1 = s - bb
        AscendC::Reg::Sub(eReg, hiReg, tReg, mAll);                 // t2 = hi - t1
        AscendC::Reg::Sub(tReg, xReg, bReg, mAll);                  // t3 = x - bb
        AscendC::Reg::Add(eReg, eReg, tReg, mAll);                  // err = t2 + t3
        AscendC::Reg::Add(loReg, loReg, eReg, mAll);
        AscendC::Reg::Add(loReg, loReg, qReg, mAll); // + exact product error
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
    // (oneRepeatSize) is the kernel's own GetVecLen()/sizeof(float). The two agree today
    // (both 64) but come from different sources, so `bufElems - V` may not be
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
    //     address. (The SDK main-chain calls of S5 used to be in the same bounded,
    //     store-masked over-read class; since the R4-6 fix their window is a subset
    //     of their H-long route, so among the H-LONG routes this guard path is now
    //     the only crossing. This branch reads no block-level small buffer with a
    //     vector load at all — its CopyOutGates uses DataCopy (MTE) and Duplicate
    //     (store-only) — so no such window remains here.)
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
        // Same window rule for the exact Mul errors of the tail products.
        AscendC::Reg::LoadAlign(qReg, errAddr + base);
        AscendC::Reg::Select(aReg, qReg, zReg, mWin);
        AscendC::Reg::Add(loReg, loReg, aReg, mAll);
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

    // ---- Non-finite guard of the compensation channel (B1-ST-08 / TC-15) ----
    // An error-free transformation is only *defined* on finite operands: if a
    // product is +/-Inf, `s = hi + x` is Inf and the difference terms become
    // Inf - Inf = NaN. That NaN can only ever live in the compensation channel,
    // because the primary accumulator `hi` holds exactly what IEEE-754 addition
    // produces (Inf stays Inf, NaN stays NaN). Dropping the NaN lanes of `lo`
    // before the horizontal sum therefore leaves `hi` - and with it the emitted
    // value - untouched: +/-Inf is propagated, never saturated or clamped
    // (Kernel.md §9.2), and a NaN product still makes the whole result NaN
    // through `hi`. `x == x` is the cheapest finite test available in VF code
    // (`Compare`/`Select` are both on the VF-safe whitelist).
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
// class MoeTutelCombineGatesB32<T>
//
// T = float (tilingKey 1, Branch-1). Every route of this branch is fp32: the input
// rows, the product, the reduction source/scratch/destination, the block
// accumulator and the write-back carrier (DESIGN-BRANCH-1.md §4 table).
// ===========================================================================
template <typename T>
class MoeTutelCombineGatesB32 {
public:
    __aicore__ inline MoeTutelCombineGatesB32() {}

    // -----------------------------------------------------------------------
    // Init — GM binding, TilingData unpack, core range, TBuf allocation and the
    // DataCopyPad parameter pre-computation (DESIGN-BRANCH-1.md §5.1).
    //
    // The TPipe is created by the entry point and passed in by pointer
    // (Kernel.md §1/§4); no workspace is bound (A12, DESIGN-BRANCH-1.md §8).
    // -----------------------------------------------------------------------
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y_grad, GM_ADDR indices, GM_ADDR locations, GM_ADDR gates_grad,
                                const MoeTutelCombineGatesTilingData *tilingData, AscendC::TPipe *pipe)
    {
        td_ = tilingData;
        pipe_ = pipe;

        // ---- TilingData unpack (TilingData.md §1, 7 fields; this branch has no
        // private field) ----
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
        // formula). Both are element CAPACITIES, never counts — every Mul / VF
        // reduction count stays the valid hidden_ (Kernel.md §9.3).
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
        //         register window of the consumers. This branch's T is float, so the row
        //         tier and the fp32 tier land on the SAME grid (W = 64 elements) — R4-6's
        //         "window is a subset of the route" reduces to exactly this divisibility
        //         plus (ii), so a field that is >= hidden_ but NOT a multiple of W is a
        //         violation, not a valid configuration
        //   (ii)  ubRowElems_  >= CeilAlign(max(hidden_, 1), W)   the rows hold H values
        //         ubFp32Elems_ >= CeilAlign(max(hidden_, 1), W)   AND the
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
        // (CalcHiddenUpperBound(): 15808 here, 12608 on the 16b path) then NO
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
        constexpr uint32_t kRowAlign = kAlignBytes / sizeof(T);               // W = 64 (T is float)
        constexpr uint32_t kFp32Align = kAlignBytes / sizeof(float);          // W = 64
        constexpr uint32_t kRowCap = AscendC::TOTAL_UB_SIZE / sizeof(T);      // 63488
        constexpr uint32_t kFp32Cap = AscendC::TOTAL_UB_SIZE / sizeof(float); // 63488
        constexpr uint32_t kSumBytes = (INNER_OFFSET * sizeof(float) > kRegBytes) ?
                                           static_cast<uint32_t>(INNER_OFFSET * sizeof(float)) :
                                           kRegBytes; // 256 B
        // The 5 block-level routes' byte sum, spelled with the SAME expressions the five
        // InitBuffer calls below use => 3 x 64 + 256 + 64 = 512 B, byte-identical to the
        // host's CalcSmallBufBytes(): on this platform blockSize == 32, so every
        // CeilAlign(INNER_OFFSET * sizeof(float), blockSize) the host applies is a no-op.
        constexpr uint32_t kSmallBytes =
            3u * (INNER_OFFSET * sizeof(int32_t)) + kSumBytes + (INNER_OFFSET * sizeof(float)); // 512 B

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
        // (2 T rows x 4 B + 2 fp32 rows x 4 B + the block-level group). Out of budget
        // => both tiers go to their (W-aligned) floors at once, which is the smallest
        // possible request that still satisfies (i) and (ii).
        {
            const uint64_t needBytes = 2ull * static_cast<uint64_t>(ubRowElems_) * sizeof(T) +
                                       2ull * static_cast<uint64_t>(ubFp32Elems_) * sizeof(float) +
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
        // Formulas (2-6)/(2-7): the tail core is clamped to taskNum_ so a core with
        // no task gets startOffset_ >= endOffset_ and writes nothing.
        startOffset_ = blockIdx_ * static_cast<int64_t>(taskNumPerBlock_);
        endOffset_ = startOffset_ + static_cast<int64_t>(taskNumPerBlock_);
        if (endOffset_ > taskNum_) {
            endOffset_ = taskNum_;
        }

        // The two reduction carriers — mulBuf_ (S5 product) and workBuf_ (S5 exact
        // product error, NUMERICAL STABILITY (d)) — must each hold one full vector
        // register, because the compensated reduction's partial last chunk reads an
        // entire V-wide aligned window from BOTH. On the host side ubFp32Elems is
        // CeilAlign'ed to one vector register (hence a multiple of V and >= V) before it
        // is handed down; the R5-hardened backstop above re-establishes the same two
        // facts locally (a multiple of W and >= CeilAlign(hidden_, W)), so the property
        // holds for a stale host as well and not only for a correct one — conditioned on
        // hidden_ itself being in contract (see the SCOPE note in the backstop).
        //
        // R4-6 fix (09-15): the compensated reduction VF clamps its own tail window
        // (this class), while the S5 Mul VF and the surrounding SDK calls load one
        // whole register per repeat with the mask on the STORE only, so their consumed
        // range is ceil(hidden_/W)*W elements. Before the fix a route capacity of only
        // CeilAlign(hidden_, 8) let that window reach up to 56 elements (224 B) past
        // the route (worst case hidden_ = 64k+1). The host now aligns BOTH tiers to
        // W = vectorSize/4 = 64 (this branch's T is float, so the row tier and the fp32
        // tier land on the same grid), making window ⊆ route hold for every
        // legal hidden: nothing past hidden_ is consumed and no lane is ever discarded by a
        // store mask. The price is H capacity (the binding bound falls by 40 elements),
        // accepted deliberately; the exact old/new contract values and the measured byte
        // accounts are recorded in the host UB budget note
        // (op_host/moe_tutel_combine_gates_tiling_arch35.cpp, CalcUbTiers /
        //  CalcUbNeedBytes). See also the S5 note
        // below.

        // ---- GM binding (sizes from TilingData.md field consumption table) ----
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x),
                             static_cast<uint64_t>(samples_) * static_cast<uint64_t>(hidden_));
        yGradGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(y_grad),
            static_cast<uint64_t>(batchSize_) * static_cast<uint64_t>(capacity_) * static_cast<uint64_t>(hidden_));
        indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indices),
                                   static_cast<uint64_t>(samples_) * static_cast<uint64_t>(batchSize_));
        locationsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(locations),
                                     static_cast<uint64_t>(samples_) * static_cast<uint64_t>(batchSize_));
        gatesGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gates_grad),
                                     static_cast<uint64_t>(samples_) * static_cast<uint64_t>(batchSize_));

        // ---- TBuf allocation: the 9 routes of DESIGN-BRANCH-1.md §4, depth 1 ----
        // All TBuf, so InitBuffer must use the two-argument form: the three-argument
        // form is static_assert'ed to TQue only. "Depth 1" means one block per route,
        // i.e. the budget carries no x2 double-buffer factor.
        // Sizes come from TilingData (ubRowElems_ / ubFp32Elems_), never from a
        // kernel-side derivation; × sizeof(float) recovers the exact byte count the
        // host charged in CalcUbNeedBytes().
        // Block-level small buffers: 5 routes, 512 B total (constant: INNER_OFFSET
        // slots of int32 / fp32, independent of hidden_).
        // R4-7: sumBuf_ is the one block-level route a VECTOR load reads (the UB -> UB
        // DataCopy of CopyOutGates, CopyImpl level 2 in
        // dav_3510/kernel_operator_data_copy_impl.h:1043-1063 -> Reg::LoadAlign(vreg, src)
        // with NO mask), and that load touches GetVecLen() = 256 B from the route base;
        // with the old 64 B route it left the route by 192 B. It is therefore floored at
        // one whole vector register. The host applies the same rule in
        // CalcSmallBufBytes(blockSize, vectorSize, isFp32Path) and charges exactly these
        // bytes. idx/loc are DMA'd and scalarly read, tmp is written by a masked one-lane
        // vector store and scalarly read, out is written by a masked vector store and
        // read by MTE3 — none of them is the source of a vector load, so they keep the
        // plain block alignment.
        // kRegBytes / kSumBytes / kSmallBytes are declared with the R4-4/R5 backstop
        // above, so this allocation and the UB budget check above are literally one
        // arithmetic rather than two that happen to agree.
        pipe_->InitBuffer(idxBuf_, INNER_OFFSET * sizeof(int32_t)); // 64 B
        pipe_->InitBuffer(locBuf_, INNER_OFFSET * sizeof(int32_t)); // 64 B
        pipe_->InitBuffer(tmpBuf_, INNER_OFFSET * sizeof(float));   // 64 B ReduceSum scalar out
        pipe_->InitBuffer(sumBuf_, kSumBytes);                    // 256 B fp32 accumulator block (whole-register read)
        pipe_->InitBuffer(outBuf_, INNER_OFFSET * sizeof(float)); // 64 B write-back carrier
        // H-long routes: 4 routes, 16H bytes total (4H each) — no Cast route exists
        // in this branch (P = 9, DESIGN-BRANCH-1.md §4).
        pipe_->InitBuffer(xBuf_, static_cast<uint32_t>(ubRowElems_) * sizeof(T));        // 4H x token row
        pipe_->InitBuffer(yBuf_, static_cast<uint32_t>(ubRowElems_) * sizeof(T));        // 4H y_grad slot row
        pipe_->InitBuffer(mulBuf_, static_cast<uint32_t>(ubFp32Elems_) * sizeof(float)); // 4H element-wise product
        // workBuf_ is DESIGN-BRANCH-1.md §4's "ReduceSum sharedTmpBuffer" route. The
        // compensated reduction (S6) keeps all of its state in vector registers, so
        // that scratch role is gone; Task 28.2.7 repurposes the SAME route (same
        // count, same 4H bytes, still physically separate from mulBuf_/tmpBuf_) to
        // carry the exact per-element rounding error of S5's Mul. P = 9 and the
        // 16 B-per-element budget are therefore unchanged. It takes the same
        // downlinked size as mulBuf_ so that the shared tail window of S6 is in
        // range for both carriers.
        pipe_->InitBuffer(workBuf_, static_cast<uint32_t>(ubFp32Elems_) * sizeof(float)); // 4H (Mul-error carrier)

        // ---- LocalTensor handles (TBuf form: no AllocTensor, take once here) ----
        idxLocal_ = idxBuf_.Get<int32_t>();
        locLocal_ = locBuf_.Get<int32_t>();
        tmpLocal_ = tmpBuf_.Get<float>();
        sumLocal_ = sumBuf_.Get<float>();
        outLocal_ = outBuf_.Get<float>();
        xLocal_ = xBuf_.Get<T>();
        yLocal_ = yBuf_.Get<T>();
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
    // Process — block loop plus the explicit H == 0 short circuit
    // (DESIGN-BRANCH-1.md §3 pipeline / §5.3, Kernel.md §9.9).
    //
    // Every core owns the flat task range [startOffset_, endOffset_); an empty
    // tensor (taskNum_ == 0) leaves the loop with zero iterations and writes
    // nothing (returns_empty, Kernel.md §9.7).
    // -----------------------------------------------------------------------
    __aicore__ inline void Process()
    {
        if (startOffset_ >= endOffset_) {
            return; // tail core with no task at all (formula (2-7)) / empty tensor
        }

        // ---- explicit early short circuit for H == 0 (Kernel.md §9.9) ----
        // H = 0 is a legal degenerate input: the empty inner product is 0, so the
        // whole block is written back as 0. No reduction may be issued (the old
        // ReduceSum carries ASCENDC_CHECK_VALUE_RANGE(count, 1, ...), and the
        // compensated reduction is defined for count >= 1 — its tail mask and its
        // final lane reduce both need at least one valid product) and no row
        // DataCopyPad may be issued (blockLen has to be in [1, 2097151], so a
        // 0-byte transfer is illegal).
        // The test is `<= 0` and not `== 0` as a last line of defence: the host
        // rejects unresolved negative dims (HostTiling §3.3 item 6, A1-7), and a
        // negative hidden would otherwise reach the counts through uint32_t.
        if (hidden_ <= 0) {
            for (int64_t taskOffset = startOffset_; taskOffset < endOffset_; taskOffset += INNER_OFFSET) {
                const int32_t innerTask = InnerTaskOf(taskOffset);
                AscendC::Duplicate<float>(sumLocal_, 0.0f, INNER_OFFSET);
                CopyOutGates(taskOffset, innerTask); // write back 0 only
            }
            return;
        }

        for (int64_t taskOffset = startOffset_; taskOffset < endOffset_; taskOffset += INNER_OFFSET) {
            // Executing state: held = [idxBuf_, locBuf_, xBuf_, yBuf_, mulBuf_, workBuf_,
            // tmpBuf_, sumBuf_, outBuf_] <- P_fp32 = 9 peak (all nine routes are
            // allocated once in Init and stay resident for the whole kernel).
            const int32_t innerTask = InnerTaskOf(taskOffset);
            CopyInIndices(taskOffset, innerTask); // NOTE: chain step S1
            AscendC::PipeBarrier<PIPE_ALL>();     // #1  MTE2 -> S (GetValue)

            // ---- zero mask (A8/B2): all INNER_OFFSET slots start at 0, so an
            // illegal task is 0 without any per-task write-back ----
            AscendC::Duplicate<float>(sumLocal_, 0.0f, INNER_OFFSET);

            for (int32_t i = 0; i < innerTask; ++i) {
                // #7 V -> MTE2 (WAR, across tasks and across blocks): the next task's
                // DMA may overwrite xBuf_/yBuf_ only after this core finished reading
                // them. Single buffering, so this crossing must not be dropped.
                AscendC::PipeBarrier<PIPE_ALL>(); // #7  V -> MTE2 (WAR)

                // ---- S2 scalar address decode + the four-condition `ok` mask ----
                const int32_t e = idxLocal_.GetValue(static_cast<uint32_t>(i)); // indices[t]
                const int32_t p = locLocal_.GetValue(static_cast<uint32_t>(i)); // locations[t]
                // ok = (e >= 0) && (e < B) && (p >= 0) && (p < C) — all four terms from
                // spec.yaml math_semantics.formula; an illegal task keeps its 0 slot.
                // (Enhancement over the reference implementation, which only tests
                // `p < C && e >= 0`: Kernel.md "three behavioural differences" #1.)
                const bool ok = (e >= 0) && (e < batchSize_) && (p >= 0) && (p < capacity_);
                if (!ok) {
                    continue; // slot stays 0 (A8); no DMA is issued for this task
                }

                const int64_t token =
                    (taskOffset + static_cast<int64_t>(i)) % static_cast<int64_t>(samples_); // X_{t mod S} (A7)
                const int64_t slot = static_cast<int64_t>(e) * static_cast<int64_t>(capacity_) +
                                     static_cast<int64_t>(p); // Ybar_{e*C + p}

                // #2 S -> MTE2: token/slot are scalar results and must be ordered before
                // the two task-level DataCopyPad calls below (required for ordering, not
                // an optional optimisation).
                AscendC::PipeBarrier<PIPE_ALL>(); // #2  S -> MTE2 (before the two DMA)

                CopyInTaskRows(token, slot); // S3/S4
                // #3 MTE2 -> V: a second, independent barrier after the two DMA calls
                // and before their consumer. #2 and #3 sit on opposite sides of the
                // same pair of DataCopyPad calls, so they can never be merged.
                AscendC::PipeBarrier<PIPE_ALL>(); // #3  MTE2 -> V (after the two DMA)

                // ---- S5 element-wise multiply in fp32 (VF, register resident) ----
                // count is the valid hidden_, never the aligned length (Kernel.md §9.3).
                // Writes both the product (mulBuf_) and its exact rounding error
                // (workBuf_, TwoProduct by FMA) — see the file header (d).
                //
                // R4-6 fix (09-15) — WHY THE TAIL IS NOT READ ANY MORE: S5's VF loads
                // both source rows with a whole-register, UNMASKED load —
                // `Reg::LoadAlign(src0Reg, src0Addr, aReg)` in MoeTutelCombineGatesMulFloatVF
                // (this file) which mirrors the dav_3510 SDK pattern, where the mask
                // only reaches the store. The touched source range is therefore
                // ceil(hidden_/W)*W elements, W = one vector register counted in the
                // route's own elements (W = 64 here: T = float, and the VF's
                // oneRepeatSize is GetVecLen()/sizeof(float)). Before the fix the row
                // tier was only block-aligned (CeilAlign(hidden_, 8)), so that window
                // could reach up to 56 elements (224 B) past ubRowElems_ (worst case
                // hidden_ = 64k+1). The host now aligns both tiers to W, so
                //     ceil(hidden_/W)*W <= ubRowElems_  and  <= ubFp32Elems_
                // is an identity for every legal hidden_: the window is a SUBSET of the
                // route, nothing past hidden_ is read at all, and the "UB tail is never
                // read" statement of Kernel.md §9.3 holds literally again.
                MulFp32OnMainChain(mulLocal_, workLocal_, xLocal_, yLocal_, static_cast<uint32_t>(hidden_));

                // #4 V -> V: the reduction reads the products written by the Mul
                // above (RAW within the V pipe; the reduction folds in registers,
                // so same-engine order must not be assumed).
                AscendC::PipeBarrier<PIPE_ALL>(); // #4  V -> V

                // ---- S6 compensated reduction along H (see file header (b)) ----
                // Replaces DESIGN-BRANCH-1.md §5.3's single ReduceSum: the fp32
                // pointwise contract cannot be met by any one-fold fp32 reduction.
                // src = mulBuf_ + workBuf_ (fp32), dst = tmpBuf_[0] (one scalar lane);
                // count = hidden_ >= 1 (H = 0 took the short circuit above).
                // No UB scratch is used, so src / dst of the old ReduceSum stay
                // physically separate as §9.4 requires.
                CompensatedReduceOnMainChain(mulLocal_, workLocal_, tmpLocal_, static_cast<uint32_t>(hidden_));

                // #5 V -> S: tmpBuf_[0] is written by the vector engine and read by the
                // scalar engine; that visibility has no implicit ordering.
                AscendC::PipeBarrier<PIPE_ALL>(); // #5  V -> S

                // ---- S7 masked slot write (only legal tasks reach this point) ----
                sumLocal_.SetValue(static_cast<uint32_t>(i), tmpLocal_.GetValue(0u));
            }
            CopyOutGates(taskOffset, innerTask); // S8/S9 (unconditional, B2)
        }
    }

private:
    // -----------------------------------------------------------------------
    // INNER_OFFSET — compile-time geometry of this branch.
    //
    // INNER_OFFSET = 16 is the number of tasks in one inner transfer block, shared
    // with the host side (HostTiling.md §4.3) and carried in no TilingData field.
    // There is deliberately NO UB row-alignment unit (nor any align helper) here any
    // more: the aligned element counts arrive as TilingData fields ubRowElems /
    // ubFp32Elems, so the kernel cannot drift away from the host's UB budget
    // (TilingData.md §3.2, P-08).
    // -----------------------------------------------------------------------
    static constexpr int32_t INNER_OFFSET = 16;

    // -----------------------------------------------------------------------
    // InnerTaskOf — valid task count of the inner block starting at taskOffset.
    //
    // Formula (2-8): min(INNER_OFFSET, taskNum - taskOffset). Its only two uses are
    // the inner loop bound and the CopyOut blockLen (B3): writing back INNER_OFFSET
    // elements on a tail block would run past S*B.
    // -----------------------------------------------------------------------
    __aicore__ inline int32_t InnerTaskOf(int64_t taskOffset) const
    {
        const int64_t remaining = taskNum_ - taskOffset;
        return (remaining < static_cast<int64_t>(INNER_OFFSET)) ? static_cast<int32_t>(remaining) : INNER_OFFSET;
    }

    // -----------------------------------------------------------------------
    // CopyInIndices — chain step S1: the block's indices / locations (§5.2).
    //
    // blockLen = innerTask * sizeof(int32_t), NOT INNER_OFFSET * sizeof(int32_t):
    // a constant 16 would read past S*B on every tail block
    // (paradigms/common/datacopypad-rules.md:141). The UB side still holds
    // INNER_OFFSET slots, of which only the first innerTask are valid.
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyInIndices(int64_t taskOffset, int32_t innerTask)
    {
        idxCopyParams_.blockLen = static_cast<uint32_t>(innerTask) * static_cast<uint32_t>(sizeof(int32_t));
        locCopyParams_.blockLen = idxCopyParams_.blockLen;
        AscendC::DataCopyPad(idxLocal_, indicesGm_[static_cast<uint64_t>(taskOffset)], idxCopyParams_,
                             AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        AscendC::DataCopyPad(locLocal_, locationsGm_[static_cast<uint64_t>(taskOffset)], locCopyParams_,
                             AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
    }

    // -----------------------------------------------------------------------
    // CopyInTaskRows — chain steps S3/S4: the x token row and the y_grad slot row.
    //
    // Called only for a task that passed the `ok` mask, so an illegal routing never
    // issues a DMA. token = t mod S (A7), slot = e * C + p computed in int64 (B6).
    // The UB tail beyond ubRowElems_ stays stale (isPad = false, i.e. no padding
    // value is written); it is NEVER READ (B7, Kernel.md §9.3) — since the R4-6 fix
    // the host aligns ubRowElems_ to one whole vector register of T elements, so the
    // S5 VF's register windows stay inside the route and no lane past hidden_ is
    // touched at all (see the note before S5).
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyInTaskRows(int64_t token, int64_t slot)
    {
        AscendC::DataCopyPad(xLocal_, xGm_[static_cast<uint64_t>(token * static_cast<int64_t>(hidden_))], xCopyParams_,
                             AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
        AscendC::DataCopyPad(yLocal_, yGradGm_[static_cast<uint64_t>(slot * static_cast<int64_t>(hidden_))],
                             yCopyParams_, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
    }

    // -----------------------------------------------------------------------
    // MulFp32OnMainChain — S5 dispatch: the kernel-side VF wrapper of
    // MoeTutelCombineGatesMulFloatVF. Kept as its own helper so the repeat geometry
    // (GetVecLen-based) is computed in ordinary kernel code, next to the
    // PipeBarrier that orders it against the surrounding vector work.
    // -----------------------------------------------------------------------
    __aicore__ inline void MulFp32OnMainChain(const LocalTensor<float> &dst, const LocalTensor<float> &err,
                                              const LocalTensor<T> &src0, const LocalTensor<T> &src1, uint32_t count)
    {
        constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(float);
        __ubuf__ float *dstAddr = reinterpret_cast<__ubuf__ float *>(dst.GetPhyAddr());
        __ubuf__ float *errAddr = reinterpret_cast<__ubuf__ float *>(err.GetPhyAddr());
        __ubuf__ float *src0Addr = reinterpret_cast<__ubuf__ float *>(src0.GetPhyAddr());
        __ubuf__ float *src1Addr = reinterpret_cast<__ubuf__ float *>(src1.GetPhyAddr());
        asc_vf_call<MoeTutelCombineGatesMulFloatVF<float>>(dstAddr, errAddr, src0Addr, src1Addr, count, oneRepeatSize);
    }

    // -----------------------------------------------------------------------
    // CompensatedReduceOnMainChain — S6 dispatch: the kernel-side VF wrapper of
    // MoeTutelCombineGatesCompensatedReduceVF. Kept as its own helper so the
    // register geometry (GetVecLen-based) is computed in ordinary kernel code,
    // next to the PipeBarrier that orders it against the surrounding vector work.
    // -----------------------------------------------------------------------
    __aicore__ inline void CompensatedReduceOnMainChain(const LocalTensor<float> &src, const LocalTensor<float> &err,
                                                        const LocalTensor<float> &dst, uint32_t count)
    {
        constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(float);
        __ubuf__ float *srcAddr = reinterpret_cast<__ubuf__ float *>(src.GetPhyAddr());
        __ubuf__ float *errAddr = reinterpret_cast<__ubuf__ float *>(err.GetPhyAddr());
        __ubuf__ float *dstAddr = reinterpret_cast<__ubuf__ float *>(dst.GetPhyAddr());
        asc_vf_call<MoeTutelCombineGatesCompensatedReduceVF<float>>(srcAddr, errAddr, dstAddr, count, oneRepeatSize,
                                                                    static_cast<uint32_t>(ubFp32Elems_));
    }

    // -----------------------------------------------------------------------
    // CopyOutGates — chain steps S8/S9: the single dense write-back path (§5.4).
    //
    // Unconditional: every block is written back, including a block whose tasks are
    // all illegal (B2 — the slots then hold the block-start zeros, and the reference
    // implementation's `if (moveCounts > 0)` behaviour is deliberately not kept).
    // blockLen = innerTask * sizeof(float) (B3: never past S*B). No narrowing Cast
    // exists in this branch (16-bit only).
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyOutGates(int64_t taskOffset, int32_t innerTask)
    {
        // Entry barrier covering two crossings at once:
        //   #6  MTE3 -> V/MTE : this block's DataCopy may overwrite outBuf_ only after
        //                       the previous block's DataCopyPad finished reading it
        //   #10 S -> V/MTE    : sumBuf_ is written slot by slot by the scalar engine
        //                       (and by the block-start Duplicate) and read as a whole
        //                       by the DataCopy below
        AscendC::PipeBarrier<PIPE_ALL>(); // #6 + #10

        // ---- UB -> UB reshaping of the [16] result block ----
        // DataCopy with the element-count overload; a UB -> UB DataCopyPad has no
        // three-argument form in the SDK (the LocalTensor/LocalTensor overloads all
        // need Nd2NzParams), so this is the only legal spelling.
        // count = INNER_OFFSET = 16 fp32 = 64 B (32 B alignment guaranteed by
        // InitBuffer). The level-2 implementation loads a WHOLE UNMASKED vector
        // register from sumLocal_ (CopyImpl, dav_3510/kernel_operator_data_copy_impl.h
        // :1043-1063 -> Reg::LoadAlign(vreg, src)); its store is masked to the 16 valid
        // lanes. sumBuf_ therefore carries the R4-7 one-register floor (256 B), so the
        // 256 B load window is a subset of the route.
        AscendC::DataCopy(outLocal_, sumLocal_, static_cast<uint32_t>(INNER_OFFSET));

        // #6' V/MTE -> MTE3: outBuf_ was just written and is read by the MTE3 copy
        // below. The barrier must sit between the write and the read; placing it
        // before CopyOutGates() would be ineffective.
        AscendC::PipeBarrier<PIPE_ALL>(); // #6' V/MTE -> MTE3

        // ---- write back GM (blockLen = innerTask, B3 tail-safe write) ----
        outCopyParams_.blockLen = static_cast<uint32_t>(innerTask) * static_cast<uint32_t>(sizeof(float));
        AscendC::DataCopyPad(gatesGradGm_[static_cast<uint64_t>(taskOffset)], outLocal_, outCopyParams_);
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
    int32_t ubRowElems_ = 0;  // elements per input-dtype H-long route x/y (from TilingData)
    int32_t ubFp32Elems_ = 0; // elements per fp32 H-long route mul/work (from TilingData)

    // ---- GM views ----
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGradGm_;
    GlobalTensor<T> gatesGradGm_;
    GlobalTensor<int32_t> indicesGm_;
    GlobalTensor<int32_t> locationsGm_;

    // ---- the 9 UB routes, all TBuf (no TQue anywhere) ----
    TBuf<TPosition::VECCALC> idxBuf_;
    TBuf<TPosition::VECCALC> locBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> sumBuf_;
    TBuf<TPosition::VECCALC> outBuf_;
    TBuf<TPosition::VECCALC> xBuf_;
    TBuf<TPosition::VECCALC> yBuf_;
    TBuf<TPosition::VECCALC> mulBuf_;
    TBuf<TPosition::VECCALC> workBuf_;

    // ---- handles taken once in Init (TBuf has no Alloc/Free) ----
    LocalTensor<int32_t> idxLocal_;
    LocalTensor<int32_t> locLocal_;
    LocalTensor<float> tmpLocal_;
    LocalTensor<float> sumLocal_;
    LocalTensor<float> outLocal_;
    LocalTensor<T> xLocal_;
    LocalTensor<T> yLocal_;
    LocalTensor<float> mulLocal_;
    LocalTensor<float> workLocal_;

    // ---- DataCopyPad parameters (blockLen is in BYTES) ----
    DataCopyExtParams xCopyParams_;
    DataCopyExtParams yCopyParams_;
    DataCopyExtParams idxCopyParams_;
    DataCopyExtParams locCopyParams_;
    DataCopyExtParams outCopyParams_;
};

#endif // MOE_TUTEL_COMBINE_GATES_32B_H_
