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
 * \file moe_tutel_combine_gates_tiling_arch35.cpp
 * \brief MoeTutelCombineGates arch35 (Ascend 950) tiling implementation.
 */

#include "register/op_def_registry.h"             // IMPL_OP_OPTILING
#include "op_common/log/log.h"                    // OP_CHECK_NULL_WITH_CONTEXT, OP_LOGE, OP_LOGI
#include "op_common/op_host/util/math_util.h"     // Ops::Base::CeilDiv / CeilAlign
#include "op_common/op_host/util/platform_util.h" // Ops::Base::GetUbBlockSize / GetCacheLineSize / GetVRegSize
#include "tiling/platform/platform_ascendc.h"     // platform_ascendc::PlatformAscendC (TilingParse)
#include "../op_kernel/arch35/moe_tutel_combine_gates_tiling_struct.h" // MoeTutelCombineGatesTilingData
#include "../op_kernel/arch35/moe_tutel_combine_gates_struct.h" // MOE_TUTEL_COMBINE_GATES_*_MODE, GET_TPL_TILING_KEY
#include "moe_tutel_combine_gates_tiling_arch35.h"              // MoeTutelCombineGatesCompileInfo

#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// 结构化错误上报宏的选择依据
//
// 本文件的输入侧校验一律用 CANN 9.2.0 的
// OP_LOGE_FOR_INVALID_{DTYPE,DTYPES,FORMAT,SHAPEDIM,SHAPES,SHAPESIZE,VALUE}_WITH_REASON
// 宏（`$ASCEND_HOME_PATH/x86_64-linux/include/op_common/log/log.h:325/350/401/478/554/630/655/784`），
// 它们经 OP_LOGE_LIBOPAPI_REPORT + REPORT_PREDEFINED_ERR_MSG 做预定义码（EZ0009~EZ0035）
// 结构化上报。与上一轮的差别（当时的顶替写法与其理由已作废）：
//   * CANN 9.0.0-beta.2 里这一族宏不存在（当时全树 grep "OP_LOGE_FOR" 零命中），只能用
//     OP_LOGE_WITH_INVALID_INPUT_*（EZ0001~EZ0007）顶替，而**输出侧**（gates_grad）校验
//     因那些宏的文案写死为「第 N 个输入 ...」只能保留自由文本 OP_LOGE；
//   * 9.2.0 起这一族宏存在，且参数名是**字符串**（复数宏收 paramNames），输出侧与多参数
//     联合判定都能结构化上报 ⇒ 输出侧不再有「宏不适用」的例外。
// 逐处映射：dtype（含配对与输出侧）→ DTYPE/DTYPES_WITH_REASON；format → FORMAT_WITH_REASON；
// rank → SHAPEDIM_WITH_REASON；shape 关系（含输出侧）→ SHAPES_WITH_REASON；元素数 →
// SHAPESIZE_WITH_REASON；维度取值域 / 取值域 / 派生上界 → VALUE_WITH_REASON。
// **仍保留裸 OP_LOGE 的只有 GetPlatformInfo() 的 5 处**：那是平台/硬件查询失败（不是输入
// 参数非法），R6 的 OP_LOGE_FOR_INVALID_* 语义为「某输入/参数取值非法」，硬套会把平台项
// 报成参数错误；它们已带 REPORT_INNER_ERR_MSG(EZ9999)，逐处注释见该函数。
// ---------------------------------------------------------------------------

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;

// ---------------------------------------------------------------------------
// R4-4: TilingData ABI guard (the kernel side carries the same assert —
// op_kernel/moe_tutel_combine_gates_apt.cpp).
//
// This file and the kernel share ONE definition of MoeTutelCombineGatesTilingData
// (op_kernel/arch35/moe_tutel_combine_gates_tiling_struct.h) and the tiling buffer
// travels as that raw struct: REGISTER_TILINGDATA_SIZE(...) is sizeof(struct) and
// GET_TILING_DATA_WITH_STRUCT memcpy's exactly that many bytes, with no length and no
// version field. The failure mode is asymmetric: new host + old kernel is safe (the old
// kernel ignores the extra bytes), but a NEW KERNEL over an OLD host .so reads 8 bytes of
// stale memory into ubRowElems / ubFp32Elems and hands them to TPipe::InitBuffer as
// allocation sizes. Asserting the size on both sides makes a one-sided field change a
// compile error instead of a runtime hang; the deployment rule (host .so and kernel .o
// must be rebuilt and replaced TOGETHER) is registered in
// the TilingData ABI note in op_host/moe_tutel_combine_gates_tiling_arch35.cpp.
// ---------------------------------------------------------------------------
static_assert(sizeof(MoeTutelCombineGatesTilingData) == 7 * sizeof(int32_t),
              "MoeTutelCombineGates TilingData ABI changed: host .so and kernel .o must be rebuilt and "
              "replaced together (see the TilingData ABI note in op_host/moe_tutel_combine_gates_tiling_arch35.cpp)");

// ---------------------------------------------------------------------------
// Compile-time constants (HostTiling §4.3 / §5.2 / §5.1; all shared with the
// kernel side or justified by the design documents — none is a magic number)
// ---------------------------------------------------------------------------

// HostTiling §5.2: per-core task count alignment granularity (reference
// implementation _tiling.cpp:24). It is also the source of the INT32_MAX - 256
// upper bound on taskNum.
constexpr int64_t OFFSET = 256;

// HostTiling §4.3: number of tasks in one inner transfer block (reference
// implementation _16b.h:146 / _32b.h:135). Host uses it for the UB budget; the
// kernel uses the same value for the inner loop, so both sides must agree.
constexpr int64_t INNER_OFFSET = 16;

// HostTiling §5.1 (last bullet): the operator never calls ReserveLocalMemory(),
// so the reserved UB amount is 0 and ubAvail == ubSize.
constexpr int64_t UB_RESERVED = 0;

// HostTiling §3.3 item 3 (rank contract; no rank templating).
constexpr size_t RANK_X = 2;
constexpr size_t RANK_YGRAD = 3;
constexpr size_t RANK_IDX = 2;
constexpr size_t RANK_OUT = 2;
constexpr size_t NUM_INPUTS = 4;
constexpr size_t NUM_OUTPUTS = 1;

// y_grad's reduction-axis dim index (reference implementation _tiling.cpp:25).
constexpr size_t HIDDEN_DIM = 2;

// ---------------------------------------------------------------------------
// IsSupportedFloatDtype — HostTiling §3.3 item 1: x / y_grad / gates_grad must be
// in {bf16, fp16, fp32}; indices / locations must be int32.
// ---------------------------------------------------------------------------
static inline bool IsSupportedFloatDtype(ge::DataType dtype)
{
    return dtype == ge::DT_BF16 || dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT;
}

// ---------------------------------------------------------------------------
// CalcSmallBufBytes(blockSize, vectorSize, isFp32Path) — HostTiling §4.3
//
// Bytes of the small (innerOffset-sized) buffer group, each route aligned to
// blockSize ON ITS OWN and then summed (never one alignment of the total, which
// would underestimate). blockSize comes from the platform interface: the literal
// 32 appearing in HostTiling §4.2 is only the blockSize == 32 substitution.
//
// R4-7 (decided 09-15, same "remove the over-read, accept the contract change"
// policy as R4-6 — see the LOG entry R4-7): the SAME one-whole-register rule that
// R4-6 applied to the H-long tiers applies to the small routes that a vector
// instruction READS. On dav_3510 the aligned load form has NO mask
// (Reg::LoadAlign(vreg, addr) — see the two measured call sites in the 16b/32b
// kernel headers), so the window such a consumer touches is exactly one vector
// register, vectorSize bytes, measured from the route base. Reading a 64 B route
// with a 256 B window leaves the route by 192 B; the 16b path's 32 B zTmp route by
// 224 B. A route therefore has to be at least one whole vector register wide:
//
//   regBytes = max(blockSize, vectorSize)   // 256 B on Ascend950PR (dav_3510)
//
// The rule is applied to the routes the vector engine reads and to those only:
//   * sum  (both paths) — source of the 16b S10 narrowing Cast
//                         (CastIntrinsicsImplVF -> GenLoadL2 -> Reg::LoadAlign, no
//                         mask) and of the fp32 path's UB -> UB DataCopy
//                         (CopyImpl level 2 -> Reg::LoadAlign(vreg, src), no mask).
//   * zTmp (16b only)   — source of the UB -> UB DataCopy that stages the narrowed
//                         block for the MTE3 write-back (same CopyImpl level 2).
// idx / loc are written by MTE2 and read by the SCALAR engine (GetValue); tmp is
// written by a masked one-lane vector store (ReduceMax / compensated reduce) and
// read by the scalar engine; out is written by a masked vector store and read by
// MTE3. None of the five is ever the source of a vector load, so they keep the
// plain blockSize alignment (charging them a register would buy nothing).
//
//   fp32 : idx(16x4) + loc(16x4) + tmp(16x4)                    -> 3 x small32
//          + sum(regBytes) + out(16x4)                          -> 512 B
//   16b  : idx(16x4) + loc(16x4) + tmp(16x4)                    -> 3 x small32
//          + sum(regBytes) + out(16x2) + zTmp(regBytes)         -> 736 B
// ---------------------------------------------------------------------------
static inline int64_t CalcSmallBufBytes(int64_t blockSize, int64_t vectorSize, bool isFp32Path)
{
    const int64_t small32 = CeilAlign(INNER_OFFSET * 4, blockSize); // int32 / fp32 small buffers
    const int64_t small16 = CeilAlign(INNER_OFFSET * 2, blockSize); // 16b small buffers
    // One whole vector register: the floor for every route a vector load reads.
    const int64_t regBytes = (blockSize > vectorSize) ? blockSize : vectorSize;
    const int64_t sumBytes = CeilAlign(INNER_OFFSET * 4, regBytes);  // 256 B: read by a whole-register load
    const int64_t zTmpBytes = CeilAlign(INNER_OFFSET * 2, regBytes); // 256 B: read by a whole-register load
    return isFp32Path ? (3 * small32 + sumBytes + small32)           // idx/loc/tmp + sum + out
                        :
                        (3 * small32 + sumBytes + small16 + zTmpBytes); // idx/loc/tmp + sum + out(2B) + zTmp
}

// ---------------------------------------------------------------------------
// MoeTutelCombineGatesUbTiers / CalcUbTiers — the SINGLE source of every UB route
// size (HostTiling §4.3, TilingData.md §3.2; P-08 fix).
//
// The H-long routes fall into exactly two tiers, and both are computed HERE once:
//   rowElems  — elements of ONE input-dtype (T) H-long route: xBuf_ / yBuf_.
//               = CeilAlign(max(hidden, 1), max(blockSize, vectorSize) / sizeof(T))
//   fp32Elems — elements of ONE fp32 H-long route: 16b path xCastBuf_ / yCastBuf_ /
//               mulBuf_ / workBuf_ (4 routes), fp32 path mulBuf_ / workBuf_ (2).
//               = CeilAlign(max(hidden, 1), max(blockSize, vectorSize) / 4)
//
// WHY THE ALIGNMENT GRANULARITY IS max(blockSize, vectorSize)/sizeof(dtype) AND NOT
// JUST blockSize/sizeof(dtype) (R4-6 fix, decided 09-15; "option A — remove the
// over-read, accept the contract change"):
//
// Every consumer of an H-long route is a vector instruction that loads ONE WHOLE
// REGISTER PER REPEAT and keeps its mask on the STORE only, so the range it actually
// TOUCHES is ceil(hidden / W) * W elements, where W is the per-repeat element count of
// that consumer. The host must therefore guarantee
//     ceil(hidden / W) * W <= routeElems,
// which holds for EVERY legal hidden iff routeElems is a multiple of W and
// routeElems >= hidden — i.e. iff routeElems is hidden CeilAlign'ed to W.
// W is measured from the SDK/this-repo sources, not assumed (per consumer):
//   * 16b path S5 widening Cast  — asc/impl/basic_api/dav_3510/
//     kernel_operator_vec_vconv_impl.h:251-296 CastIntrinsicsImplVF: for
//     sizeof(SRC) < sizeof(DST) it sets oneRepSize = GetVecLen() / sizeof(DST) and
//     loads `src + i * oneRepSize` with no mask (GenLoadL2 :216-231 -> Reg::LoadAlign
//     <T, DIST_UNPACK_B16>, 64 source elements per repeat = one repeat in fp32 lanes).
//   * fp32 path S5 Mul VF (op_kernel/arch35/moe_tutel_combine_gates_32b.h,
//     MoeTutelCombineGatesMulFloatVF) and this repo's MulVF / compensated-reduce VF —
//     Reg::LoadAlign of a full fp32 register per repeat = GetVecLen()/sizeof(float).
//   * 16b path S6b Abs / Muls / ReduceMax — SDK entries of the same class (full
//     register load, store-masked; fp32 operands => the same 64-element window).
//       => W == vectorSize / 4 == GetVecLen() / sizeof(float) == 64 ELEMENTS OF THE
//          ROUTE, for every one of them (the 16b rows are read by the WIDENING cast,
//          whose repeat is counted in fp32 lanes, so their window is 64 T-elements —
//          not 128; the fp32 routes are loaded as fp32, also 64).
//
// The two grids therefore coincide, and one quantity expresses both:
//   rowElems  = CeilAlign(hidden, vectorSize / sizeof(T))   (16b: CeilAlign(h, 128))
//   fp32Elems = CeilAlign(hidden, vectorSize / 4)           (both branches: h -> 64)
// CeilAlign to vectorSize/sizeof(T) is the STRICTLY SAFER of the two equivalent
// 16b spellings: it also covers a hypothetical 128-element T-row window, and it costs
// nothing (the fp32 tier carries 4x the weight of the row tier, so 12608 is the bound
// under either spelling — re-checked for R4-7, whose larger small group shifts both
// tiers by the same constant and so cannot change that conclusion). blockSize/sizeof(dtype)
// is a divisor of both
// grids (128 % 16 == 0, 64 % 8 == 0, 64 % 64 == 0), so the InitBuffer/DMA block-size
// requirement is implied and the max() is written explicitly only to keep the
// dependence visible if a platform ever reports vectorSize < blockSize.
//
// THE ALIGNMENT SUBSUMES THE OLD ONE-REGISTER FLOOR: the compensated reduction's
// partial tail chunk loads one whole V-wide aligned window per carrier (S7 in the 16b
// path, S6 in the fp32 path), which used to be an explicit `max(..., vectorSize / 4)`
// floor; CeilAlign(h, vectorSize / 4) >= vectorSize / 4 for every h >= 1, so the floor
// is now automatic and the R4-2 underflow guard in the kernel keeps its meaning
// unchanged (bufElems >= V for every host-produced configuration).
//
// CAPACITY COST (the accepted price of option A, measured on the dev machine): the
// binding H bound fell by 8 elements on the 16b path and by 40 on the fp32 path.
// The old and new contract values, the per-route byte table and the device byte
// accounts are recorded in HostTiling.md §4.5.
//
// R4-7 then applied the same one-register rule to the small routes a vector load
// reads (CalcSmallBufBytes above), which raises the group from 320 B to 736 B (16b)
// / 512 B (fp32) and moves the 16b bound 12672 -> 12608; the fp32 bound is unchanged
// at 15808. "Raises to 512 B" is the new GROUP SIZE, not the increment: the fp32
// increment is +192 B (320 -> 512), and it is absorbed by the 704 B slack that point
// had (at H = 15808 the pre-R4-7 need was 253248 B against ubAvail = 253952 B, i.e.
// 704 B spare; R4-7 spends 192 B of it and leaves 512 B). See the LOG entry R4-7.
//
// needBytes is the sum of the P route byte sizes of the branch, each route aligned
// on its own (never one alignment of the total, which would underestimate):
//   fp32 (P=9) : 2 x rowElems*4 (x, y) + 2 x fp32Elems*4 (mul, work) + small
//   16b (P=12) : 2 x rowElems*2 (x, y) + 4 x fp32Elems*4 (xCast, yCast, mul, work)
//                + small
// Because the two tier values computed here are ALSO the ones written into
// TilingData (ubRowElems / ubFp32Elems) and the kernel allocates exactly
// tierElems * sizeof(dtype) per route, "the host thinks it fits" and "the kernel
// allocates it" are the same arithmetic — a mirror formula no longer exists.
// ---------------------------------------------------------------------------
struct MoeTutelCombineGatesUbTiers {
    int64_t rowElems = 0;      // elements of one input-dtype H-long route
    int64_t fp32Elems = 0;     // elements of one fp32 H-long route
    int64_t smallBufBytes = 0; // block-sized route group (idx/loc/sum/tmp/out[/zTmp]); R4-7:
                               // the two routes a vector load reads are >= one register
    int64_t needBytes = 0;     // == the byte sum of all P routes == kernel request
};

static inline MoeTutelCombineGatesUbTiers CalcUbTiers(int64_t hidden, bool isFp32Path, int64_t blockSize,
                                                      int64_t vectorSize)
{
    MoeTutelCombineGatesUbTiers tiers;
    const int64_t elemBytesT = isFp32Path ? 4 : 2; // sizeof(T) of the branch
    // H = 0 is a legal degenerate input: a zero-length InitBuffer would leave the
    // H-long routes with a degenerate 0 B window, so the element count is floored at
    // 1 before aligning (H = 0 -> one block). Process() never touches these routes on
    // that path, and CheckUbCapacity() charges the very same bytes.
    const int64_t hiddenForAlign = (hidden > 0) ? hidden : 1;
    // R4-6 fix: align to the LARGER of the UB allocation block and one whole vector
    // register (in elements of the tier's own dtype), so that the register window of
    // every consumer stays inside the route: ceil(hidden/W)*W <= tierElems.
    const int64_t alignBytes = (blockSize > vectorSize) ? blockSize : vectorSize;
    tiers.rowElems = CeilAlign(hiddenForAlign, alignBytes / elemBytesT);
    // One vector register, fp32 elements (== the kernel's oneRepeatSize, see above);
    // CeilAlign already implies the >= one-register floor the tail windows need.
    tiers.fp32Elems = CeilAlign(hiddenForAlign, alignBytes / 4);
    tiers.smallBufBytes = CalcSmallBufBytes(blockSize, vectorSize, isFp32Path);
    const int64_t rowBytes = tiers.rowElems * elemBytesT;
    const int64_t fp32Bytes = tiers.fp32Elems * 4;
    tiers.needBytes = isFp32Path ? (2 * rowBytes + 2 * fp32Bytes + tiers.smallBufBytes) :
                                   (2 * rowBytes + 4 * fp32Bytes + tiers.smallBufBytes);
    return tiers;
}

// ---------------------------------------------------------------------------
// CalcUbNeedBytes(hidden, isFp32Path, blockSize, vectorSize) — HostTiling §4.3
//
// UB budget in bytes = Σ(route element count x element bytes) over the P routes of
// the branch, taken from CalcUbTiers() above (the same call that fills the
// TilingData fields). Kept as a named function because the UB gate and its log line
// are phrased in terms of it.
// ---------------------------------------------------------------------------
static inline int64_t CalcUbNeedBytes(int64_t hidden, bool isFp32Path, int64_t blockSize, int64_t vectorSize)
{
    return CalcUbTiers(hidden, isFp32Path, blockSize, vectorSize).needBytes;
}

// ---------------------------------------------------------------------------
// CalcHiddenUpperBound(ubSize, isFp32Path, blockSize, ubReserved, vectorSize) —
// HostTiling §4.3
//
// Largest H whose UB budget still fits: max{ H | CalcUbNeedBytes(H) <= ubAvail },
// ubAvail = ubSize - ubReserved. hidden above this bound is rejected explicitly by
// the caller (HostTiling §4.4: no H splitting, no silent downgrade). The values
// 15808 / 12608 in HostTiling §4.5 are only the ubSize = 248 KB substitution and
// must never be written into the code.
//
// CalcUbNeedBytes() is non-decreasing in H (every route byte count is), so the
// bound is found by bisection; ubAvail is a safe loose ceiling for H because the
// cheapest branch spends more than one byte per H element.
// ---------------------------------------------------------------------------
static int64_t CalcHiddenUpperBound(int64_t ubSize, bool isFp32Path, int64_t blockSize, int64_t ubReserved,
                                    int64_t vectorSize)
{
    const int64_t ubAvail = ubSize - ubReserved;
    int64_t lo = 0;
    int64_t hi = ubAvail; // ubAvail is a safe (loose) ceiling for H
    while (lo < hi) {
        const int64_t mid = lo + (hi - lo + 1) / 2;
        if (CalcUbNeedBytes(mid, isFp32Path, blockSize, vectorSize) <= ubAvail) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

// ---------------------------------------------------------------------------
// class MoeTutelCombineGatesTiling
//
// Orchestrates the public tiling pipeline in the order fixed by HostTiling §9
// (the order may not be changed):
//   1. GetPlatformInfo()     platform parameters (each one checked non-zero)
//   2. ParseInputShapes()    x[S,H] / y_grad[B,C,H] / indices / locations + dtypes
//   3. ValidateInputs()      dtype -> format -> rank -> attr(N/A) -> shape -> numeric
//   4. MultiCoreSplit()      taskNum(int64) -> INT32 bound -> CeilAlign(CeilDiv, 256)
//   5. the single dtype fork (UB budget path + DTYPE_MODE)
//   6. UB capacity check
//   7. empty tensor short circuit (taskNum == 0)
//   8. fill TilingData / SetBlockDim / SetTilingKey
//
// NOTE (order revision, independent-review A1-3/A1-4): steps 5 and 6 run BEFORE
// the step 7 early exit. The dtype fork must precede the short circuit so that an
// empty input (S == 0 or B == 0) publishes the tilingKey of ITS OWN dtype instead
// of the BF16 default, and CheckUbCapacity() must precede it so that an empty task
// axis cannot bypass the only UB gate (hidden above the platform bound is rejected
// on that path too). HostTiling §9 fixes the same relative order of the validation
// stages but lists the dtype fork after the short circuit; this file follows the
// revised order above.
// ---------------------------------------------------------------------------
class MoeTutelCombineGatesTiling {
public:
    explicit MoeTutelCombineGatesTiling(gert::TilingContext *context)
        : tilingContext(context)
    {}

    ge::graphStatus RunTiling();

private:
    // --- pipeline stages ---
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus ParseInputShapes();
    ge::graphStatus ValidateInputs();
    ge::graphStatus MultiCoreSplit();

    // --- validation sub-steps (HostTiling §3.3, order is part of the contract) ---
    ge::graphStatus CheckDtypeAndCombination();
    ge::graphStatus CheckFormat();
    ge::graphStatus CheckExactRanks();
    ge::graphStatus CheckShapeContract();
    ge::graphStatus CheckNumericRange();
    ge::graphStatus CheckIndexValueRange();
    ge::graphStatus CheckOneIndexTensor(size_t inputIndex, int64_t upperBound, bool rejectNegative);
    ge::graphStatus CheckUbCapacity();

    // --- output side ---
    bool HandleEmptyTensor(MoeTutelCombineGatesTilingData *td);
    void FillAndLogTilingData(MoeTutelCombineGatesTilingData *td);
    ge::graphStatus SetWorkspaceSize();

    // --- platform parameters (all live from the platform interfaces, never hardcoded) ---
    gert::TilingContext *tilingContext = nullptr;
    uint32_t coreNum = 0;      // AIV (vector) core count — GetCoreNumAiv()
    int64_t ubSize = 0;        // UB bytes per core      [bytes]
    int64_t blockSize = 0;     // UB allocation block    [bytes]
    int64_t cacheLineSize = 0; // cache line             [bytes]
    int64_t vectorSize = 0;    // vector register width  [bytes]

    // --- inputs (shapes in elements, dtypes) ---
    gert::Shape xShape;
    gert::Shape yShape;
    gert::Shape idxShape;
    gert::Shape locShape;
    ge::DataType xDtype = ge::DT_UNDEFINED;

    // --- derived public quantities (elements) ---
    int64_t samples = 0;         // S = x.shape[0]
    int64_t batchSize = 0;       // B = y_grad.shape[0]
    int64_t hidden = 0;          // H = x.shape[1] = y_grad.shape[2]
    int64_t capacity = 0;        // C = y_grad.shape[1]
    int64_t taskNum = 0;         // S * B (int64 intermediate, B6)
    int32_t taskNumPerBlock = 0; // CeilAlign(CeilDiv(taskNum, aivCoreNum), 256)

    // --- dtype fork results ---
    bool isFp32Path = false;
    uint64_t dtypeMode = MOE_TUTEL_COMBINE_GATES_BF16_MODE;
};

// ---------------------------------------------------------------------------
// GetPlatformInfo() — HostTiling §5.1
//
// Five platform parameters, all through the platform interfaces (writing any of
// them as a literal — 32 / 256 / 248 KB — is forbidden by HostTiling §7.5 #4);
// each one is checked non-zero, the error being reported with the queried
// interface name so the source stays traceable.
//
// 这 5 处**保留裸 OP_LOGE**（唯一例外，理由见文件头）：失败原因是平台/硬件
// 接口返回 0（环境或驱动问题），不是某个输入参数取值非法；R6 要求的
// OP_LOGE_FOR_INVALID_* 语义是「参数 <paramName> 取值非法」，套上去会把平台项误报成
// 参数错误、并给出错误的 param_name 键值。裸 OP_LOGE 已带 REPORT_INNER_ERR_MSG(EZ9999)。
// ---------------------------------------------------------------------------
ge::graphStatus MoeTutelCombineGatesTiling::GetPlatformInfo()
{
    auto platformInfo = tilingContext->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    // 1) AIV (vector) core count — B5: a pure Vector operator must split by AIV
    //    cores; the reference's GetCoreNum() (cube cores) must not be reused.
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0,
                OP_LOGE(tilingContext->GetNodeName(), // R6 不适用：平台接口返回 0，非输入校验（见本函数头注释）
                        "shape_mismatch: platform.coreNumAiv is 0 (GetCoreNumAiv() returned 0); "
                        "a non-zero AIV core count is required for multicore splitting"),
                return ge::GRAPH_FAILED);

    // 2) UB capacity in bytes — the only source of the UB budget (HostTiling §4)
    uint64_t ubSizeU64 = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeU64);
    ubSize = static_cast<int64_t>(ubSizeU64);
    OP_CHECK_IF(ubSize == 0,
                OP_LOGE(tilingContext->GetNodeName(), // R6 不适用：平台接口返回 0，非输入校验（见本函数头注释）
                        "shape_mismatch: platform.ubSize is 0 (GetCoreMemSize(UB) returned 0); "
                        "a non-zero UB size is required for the UB capacity check"),
                return ge::GRAPH_FAILED);

    // 3) UB allocation block size — passed as a parameter to every UB budget
    //    formula; writing 32 here is forbidden (HostTiling §4.3 / §7.5 #9)
    blockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(tilingContext));
    OP_CHECK_IF(blockSize == 0,
                OP_LOGE(tilingContext->GetNodeName(), // R6 不适用：平台接口返回 0，非输入校验（见本函数头注释）
                        "shape_mismatch: platform.blockSize is 0 (GetUbBlockSize() returned 0); "
                        "a non-zero UB block size is required for the UB budget"),
                return ge::GRAPH_FAILED);

    // 4) cache line size — kernel-side alignment rationale, logged for traceability
    cacheLineSize = static_cast<int64_t>(Ops::Base::GetCacheLineSize(tilingContext));
    OP_CHECK_IF(cacheLineSize == 0,
                OP_LOGE(tilingContext->GetNodeName(), // R6 不适用：平台接口返回 0，非输入校验（见本函数头注释）
                        "shape_mismatch: platform.cacheLineSize is 0 (GetCacheLineSize() returned 0); "
                        "a non-zero cache line size is required"),
                return ge::GRAPH_FAILED);

    // 5) vector register width — kernel-side mask / repeat alignment rationale
    vectorSize = static_cast<int64_t>(Ops::Base::GetVRegSize(tilingContext));
    OP_CHECK_IF(vectorSize == 0,
                OP_LOGE(tilingContext->GetNodeName(), // R6 不适用：平台接口返回 0，非输入校验（见本函数头注释）
                        "shape_mismatch: platform.vectorSize is 0 (GetVRegSize() returned 0); "
                        "a non-zero vector register width is required"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// ParseInputShapes() — read the four input shapes and dtypes
//
// HostTiling §2.1 declares that this operator has no value dependency and that the
// host should therefore not read indices / locations. This function itself reads
// only shape / dtype / format; the DESIGNED EXCEPTION is the 7th validation step
// CheckIndexValueRange(), called from ValidateInputs() below, which reads the
// content of indices / locations **only when the framework supplies them as host
// tensors** (kOnHost / kFollowing) and logs + passes on the device path — see its
// comment block for the measured reason (no stride/offset/version information
// reaches tiling on the aclnn path, so a linear read cannot be proved correct).
// The HostTiling §2.1 declaration is therefore NOT true of this implementation as
// a whole.
// ---------------------------------------------------------------------------
ge::graphStatus MoeTutelCombineGatesTiling::ParseInputShapes()
{
    const auto *xShapePtr = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, xShapePtr);
    const auto *yShapePtr = tilingContext->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, yShapePtr);
    const auto *idxShapePtr = tilingContext->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, idxShapePtr);
    const auto *locShapePtr = tilingContext->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, locShapePtr);
    xShape = xShapePtr->GetStorageShape();
    yShape = yShapePtr->GetStorageShape();
    idxShape = idxShapePtr->GetStorageShape();
    locShape = locShapePtr->GetStorageShape();

    const auto *xTensor = tilingContext->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, xTensor);
    xDtype = xTensor->GetDataType();

    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// CheckDtypeAndCombination() — HostTiling §3.3 item 1 (first in the order)
//
// x / y_grad / gates_grad in {bf16, fp16, fp32} with x.dtype == y_grad.dtype, and
// indices / locations == int32 and mutually equal. Failure -> GRAPH_FAILED with
// the dtype_not_supported semantics (no tilingKey is set on this path).
// ---------------------------------------------------------------------------
ge::graphStatus MoeTutelCombineGatesTiling::CheckDtypeAndCombination()
{
    const auto *yTensor = tilingContext->GetInputTensor(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, yTensor);
    const auto *idxTensor = tilingContext->GetInputTensor(2);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, idxTensor);
    const auto *locTensor = tilingContext->GetInputTensor(3);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, locTensor);

    const ge::DataType yDtype = yTensor->GetDataType();
    const ge::DataType idxDtype = idxTensor->GetDataType();
    const ge::DataType locDtype = locTensor->GetDataType();

    // 结构化上报用的 dtype 文本（宏的文案固定，键值由这里给全；变量必须先具名，
    // 宏内部会把实参展开两次并在 std::vector<const char*> 里继续使用）
    const std::string xDtypeStr = std::to_string(static_cast<int32_t>(xDtype));
    const std::string sameAsXStr = "same as x (dtype " + xDtypeStr + ")";
    const std::string yDtypeStr = std::to_string(static_cast<int32_t>(yDtype));
    const std::string idxDtypeStr = std::to_string(static_cast<int32_t>(idxDtype));
    const std::string sameAsIdxStr = "same as indices (dtype " + idxDtypeStr + ")";
    const std::string locDtypeStr = std::to_string(static_cast<int32_t>(locDtype));
    const std::string floatDtypeReasonStr = "dtype must be one of DT_BF16/DT_FLOAT16/DT_FLOAT";

    // 单参数 dtype 违规 → ..._DTYPE_WITH_REASON（EZ0020）；配对/输出侧多参数 →
    // ..._DTYPES_WITH_REASON（EZ0021，收 paramNames 参数名列表）。
    OP_CHECK_IF(!IsSupportedFloatDtype(xDtype),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(tilingContext->GetNodeName(), "x", xDtypeStr.c_str(),
                                                      floatDtypeReasonStr.c_str()),
                return ge::GRAPH_FAILED);
    const std::string xYGotStr = "x=" + xDtypeStr + ", y_grad=" + yDtypeStr;
    OP_CHECK_IF(yDtype != xDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext->GetNodeName(), "x, y_grad", xYGotStr.c_str(),
                                                       sameAsXStr.c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(idxDtype != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(tilingContext->GetNodeName(), "indices", idxDtypeStr.c_str(),
                                                      "dtype must be DT_INT32"),
                return ge::GRAPH_FAILED);
    const std::string idxLocGotStr = "indices=" + idxDtypeStr + ", locations=" + locDtypeStr;
    OP_CHECK_IF(locDtype != idxDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext->GetNodeName(), "indices, locations",
                                                       idxLocGotStr.c_str(), sameAsIdxStr.c_str()),
                return ge::GRAPH_FAILED);

    // gates_grad shares x's dtype (spec.yaml outputs[0].dtype_rule = x.dtype).
    // The output desc is absent on some framework paths, in which case the dtype
    // contract is already covered by InferShape.
    const auto *outDesc = tilingContext->GetOutputDesc(0);
    if (outDesc != nullptr) {
        const std::string outDtypeStr = std::to_string(static_cast<int32_t>(outDesc->GetDataType()));
        const std::string xOutGotStr = "x=" + xDtypeStr + ", gates_grad=" + outDtypeStr;
        OP_CHECK_IF(outDesc->GetDataType() != xDtype,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext->GetNodeName(), "x, gates_grad",
                                                           xOutGotStr.c_str(), sameAsXStr.c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// CheckFormat() — HostTiling §3.3 item 2
//
// The runtime format of the 4 inputs and the output must be ND (the operator
// declares FORMAT_ND / UnknownShapeFormat(ND) only). Failure -> GRAPH_FAILED with
// shape_mismatch semantics. Paths whose context carries no CompileTimeTensorDesc
// are covered by the dtype / rank / shape checks instead.
// ---------------------------------------------------------------------------
ge::graphStatus MoeTutelCombineGatesTiling::CheckFormat()
{
    for (size_t i = 0; i < NUM_INPUTS; ++i) {
        const auto *desc = tilingContext->GetInputDesc(i);
        if (desc == nullptr) {
            continue;
        }
        const std::string paramName = "input " + std::to_string(i);
        const std::string formatStr = std::to_string(static_cast<int32_t>(desc->GetStorageFormat()));
        // 单参数 format 违规 → ..._FORMAT_WITH_REASON（EZ0035）。
        OP_CHECK_IF(desc->GetStorageFormat() != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(tilingContext->GetNodeName(), paramName.c_str(),
                                                           formatStr.c_str(), "format must be FORMAT_ND"),
                    return ge::GRAPH_FAILED);
    }
    for (size_t i = 0; i < NUM_OUTPUTS; ++i) {
        const auto *desc = tilingContext->GetOutputDesc(i);
        if (desc == nullptr) {
            continue;
        }
        const std::string paramName = "output " + std::to_string(i);
        const std::string formatStr = std::to_string(static_cast<int32_t>(desc->GetStorageFormat()));
        // 单参数 format 违规 → ..._FORMAT_WITH_REASON（EZ0035）。
        OP_CHECK_IF(desc->GetStorageFormat() != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(tilingContext->GetNodeName(), paramName.c_str(),
                                                           formatStr.c_str(), "format must be FORMAT_ND"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// CheckExactRanks() — HostTiling §3.3 item 3
//
// Ranks are an exact contract (no rank templating): x = 2, y_grad = 3,
// indices = 2, locations = 2, gates_grad = 2.
// ---------------------------------------------------------------------------
ge::graphStatus MoeTutelCombineGatesTiling::CheckExactRanks()
{
    const std::string xRankStr = "rank=" + std::to_string(xShape.GetDimNum());
    const std::string yRankStr = "rank=" + std::to_string(yShape.GetDimNum());
    const std::string idxRankStr = "rank=" + std::to_string(idxShape.GetDimNum());
    const std::string locRankStr = "rank=" + std::to_string(locShape.GetDimNum());

    // 单参数「维数（rank）」违规 → ..._SHAPEDIM_WITH_REASON（EZ0012）；输出侧同样可用
    // （9.2.0 的宏收参数名字符串，不再限定输入序号）。reason 里给出该参数的 rank 契约。
    OP_CHECK_IF(xShape.GetDimNum() != RANK_X,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(tilingContext->GetNodeName(), "x", xRankStr.c_str(),
                                                         "rank must be 2 (x is [S, H])"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(yShape.GetDimNum() != RANK_YGRAD,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(tilingContext->GetNodeName(), "y_grad", yRankStr.c_str(),
                                                         "rank must be 3 (y_grad is [B, C, H])"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(idxShape.GetDimNum() != RANK_IDX,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(tilingContext->GetNodeName(), "indices", idxRankStr.c_str(),
                                                         "rank must be 2 (indices is [B, S])"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(locShape.GetDimNum() != RANK_IDX,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(tilingContext->GetNodeName(), "locations", locRankStr.c_str(),
                                                         "rank must be 2 (locations is [B, S])"),
                return ge::GRAPH_FAILED);

    const auto *outShape = tilingContext->GetOutputShape(0);
    if (outShape != nullptr) {
        const std::string outRankStr = "rank=" + std::to_string(outShape->GetStorageShape().GetDimNum());
        OP_CHECK_IF(
            outShape->GetStorageShape().GetDimNum() != RANK_OUT,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(tilingContext->GetNodeName(), "gates_grad", outRankStr.c_str(),
                                                     "rank must be 2 (gates_grad is [B, S])"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// CheckShapeContract() — HostTiling §3.3 item 5 + InferShapeDtype.md §2
//
// broadcast.kind = none, so no broadcast alignment: the dimensions are strictly
// equal. The 8 checks are: 4 exact ranks (CheckExactRanks) + H match + B match +
// S match + indices/locations element count == S*B.
// ---------------------------------------------------------------------------
ge::graphStatus MoeTutelCombineGatesTiling::CheckShapeContract()
{
    samples = xShape.GetDim(0);                        // S = x.shape[0]
    hidden = xShape.GetDim(1);                         // H = x.shape[1]
    batchSize = yShape.GetDim(0);                      // B = y_grad.shape[0]
    capacity = yShape.GetDim(1);                       // C = y_grad.shape[1]
    const int64_t yHidden = yShape.GetDim(HIDDEN_DIM); // H = y_grad.shape[2]

    // 这一组都是「两个参数联合判定」⇒ 复数宏 ..._SHAPES_WITH_REASON（EZ0010，收 paramNames）。
    // 后三处即上一轮遗留的 U-1（indices/locations dim0、dim1、元素数三处裸 OP_LOGE）。
    const std::string hGotStr = "x.shape[1]=" + std::to_string(hidden);
    const std::string hWantStr = "y_grad.shape[2]=" + std::to_string(yHidden);
    const std::string hGotPairStr = hGotStr + ", " + hWantStr;
    OP_CHECK_IF(hidden != yHidden,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "x, y_grad", hGotPairStr.c_str(),
                                                       "x.shape[1] (H) must equal y_grad.shape[2] (H)"),
                return ge::GRAPH_FAILED);
    const std::string dim0GotStr = "indices.shape[0]=" + std::to_string(idxShape.GetDim(0)) +
                                   ", locations.shape[0]=" + std::to_string(locShape.GetDim(0));
    const std::string dim0ReasonStr =
        "indices.shape[0] and locations.shape[0] must equal B = " + std::to_string(batchSize) + " (y_grad.shape[0])";
    OP_CHECK_IF(idxShape.GetDim(0) != batchSize || locShape.GetDim(0) != batchSize,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "indices, locations, y_grad",
                                                       dim0GotStr.c_str(), dim0ReasonStr.c_str()),
                return ge::GRAPH_FAILED);
    const std::string dim1GotStr = "indices.shape[1]=" + std::to_string(idxShape.GetDim(1)) +
                                   ", locations.shape[1]=" + std::to_string(locShape.GetDim(1));
    const std::string dim1ReasonStr =
        "indices.shape[1] and locations.shape[1] must equal S = " + std::to_string(samples) + " (x.shape[0])";
    OP_CHECK_IF(idxShape.GetDim(1) != samples || locShape.GetDim(1) != samples,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "indices, locations, x",
                                                       dim1GotStr.c_str(), dim1ReasonStr.c_str()),
                return ge::GRAPH_FAILED);

    const int64_t elemNum = samples * batchSize; // S * B (int64)
    const std::string elemGotStr =
        "indices=" + std::to_string(idxShape.GetShapeSize()) + ", locations=" + std::to_string(locShape.GetShapeSize());
    const std::string elemReasonStr = "both must hold exactly S*B = " + std::to_string(elemNum) +
                                      " elements (they are consumed as flat S*B index arrays)";
    OP_CHECK_IF(idxShape.GetShapeSize() != elemNum || locShape.GetShapeSize() != elemNum,
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(tilingContext->GetNodeName(), "indices, locations",
                                                           elemGotStr.c_str(), elemReasonStr.c_str()),
                return ge::GRAPH_FAILED);

    // ---- 输出 shape 一致性：gates_grad.shape == (indices.shape[0], indices.shape[1]) ----
    // 依据 InferShapeDtype.md §1「gates_grad.shape = (indices.shape[0], indices.shape[1])」
    // 与 Interface.md「Outputs」的 shape 约定：调用方预置的输出张量必须与推导结果一致
    // （InferShapeDtype.md §3 对 dtype 的同型条款即「预置/声明与推导结果不一致 → 拒绝」）。
    // 未知维（< 0，A13 动态 shape）不参与判定，避免误拒合法动态 shape 输入；rank 由
    // CheckExactRanks() 单独把关。输出侧用 ..._SHAPES_WITH_REASON 上报（9.2.0 的宏收
    // 参数名字符串，gates_grad 可作为具名参数，见文件头选择依据）。
    const auto *outShapePtr = tilingContext->GetOutputShape(0);
    if (outShapePtr != nullptr) {
        const gert::Shape &outShape = outShapePtr->GetStorageShape();
        if (outShape.GetDimNum() == RANK_OUT) {
            for (size_t d = 0; d < RANK_OUT; ++d) {
                const int64_t got = outShape.GetDim(d);
                const int64_t want = idxShape.GetDim(d);
                const std::string dimGotPairStr = "gates_grad.shape[" + std::to_string(d) + "]=" + std::to_string(got) +
                                                  ", indices.shape[" + std::to_string(d) + "]=" + std::to_string(want);
                OP_CHECK_IF(got >= 0 && want >= 0 && got != want,
                            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                                tilingContext->GetNodeName(), "gates_grad, indices", dimGotPairStr.c_str(),
                                "the declared gates_grad shape must equal indices.shape, dim by dim"),
                            return ge::GRAPH_FAILED);
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// CheckNumericRange() — HostTiling §3.3 item 6 (int32 representability)
//
// The 5 TilingData fields are int32_t, and capacity is used by the kernel as the
// address multiplier slot = e*C + p, so a silent truncation would compute wrong
// addresses. Every shape symbol is therefore checked dimension by dimension, not
// only taskNum (which MultiCoreSplit checks with its own INT32_MAX - OFFSET bound).
//
// LOWER bound (independent-review A1-7): the operator declares dynamic shape /
// dynamic rank support (OpAICoreConfig's DynamicShapeSupportFlag /
// DynamicRankSupportFlag in def.cpp), and both InferShape (the IsKnownDim /
// DimsConflict helpers in infershape.cpp) and this file's output-shape check
// (CheckShapeContract) accept negative (unresolved) dims. A negative dim that
// reached the kernel would be cast to uint32_t and hand ~4.29e9 to the count
// arguments of Abs / Cast / Mul / ReduceSum and to the blockLen of DataCopyPad
// (legal domain [1, 2097151]) — so a dimension that is still unresolved is
// rejected here instead.
// ---------------------------------------------------------------------------
ge::graphStatus MoeTutelCombineGatesTiling::CheckNumericRange()
{
    constexpr int64_t INT32_MAX_VALUE = static_cast<int64_t>(INT32_MAX);
    // 这 8 处是维度**取值域**违规（下界 / int32 上界），不是 rank 违规也不是元素数
    // 违规，故用 ..._VALUE_WITH_REASON（EZ0026）而不是 SHAPEDIM / SHAPESIZE。
    const std::string samplesNegStr = "samples=" + std::to_string(samples);
    OP_CHECK_IF(samples < 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "samples", samplesNegStr.c_str(),
                                                      "a resolved non-negative dimension is required "
                                                      "(unresolved dynamic dims are rejected)"),
                return ge::GRAPH_FAILED);
    const std::string batchSizeNegStr = "batchSize=" + std::to_string(batchSize);
    OP_CHECK_IF(
        batchSize < 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "batchSize", batchSizeNegStr.c_str(),
                                              "a resolved non-negative dimension is required "
                                              "(unresolved dynamic dims are rejected)"),
        return ge::GRAPH_FAILED);
    const std::string hiddenNegStr = "hidden=" + std::to_string(hidden);
    OP_CHECK_IF(hidden < 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "hidden", hiddenNegStr.c_str(),
                                                      "a resolved non-negative dimension is required "
                                                      "(unresolved dynamic dims are rejected)"),
                return ge::GRAPH_FAILED);
    const std::string capacityNegStr = "capacity=" + std::to_string(capacity);
    OP_CHECK_IF(capacity < 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "capacity", capacityNegStr.c_str(),
                                                      "a resolved non-negative dimension is required "
                                                      "(unresolved dynamic dims are rejected)"),
                return ge::GRAPH_FAILED);
    const std::string samplesOverStr = "samples=" + std::to_string(samples);
    OP_CHECK_IF(samples > INT32_MAX_VALUE,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "samples", samplesOverStr.c_str(),
                                                      "exceeds INT32_MAX (the TilingData fields are int32)"),
                return ge::GRAPH_FAILED);
    const std::string batchOverStr = "batchSize=" + std::to_string(batchSize);
    OP_CHECK_IF(batchSize > INT32_MAX_VALUE,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "batchSize", batchOverStr.c_str(),
                                                      "exceeds INT32_MAX (the TilingData fields are int32)"),
                return ge::GRAPH_FAILED);
    const std::string hiddenOverStr = "hidden=" + std::to_string(hidden);
    OP_CHECK_IF(hidden > INT32_MAX_VALUE,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "hidden", hiddenOverStr.c_str(),
                                                      "exceeds INT32_MAX (the TilingData fields are int32)"),
                return ge::GRAPH_FAILED);
    const std::string capacityOverStr = "capacity=" + std::to_string(capacity);
    OP_CHECK_IF(capacity > INT32_MAX_VALUE,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "capacity", capacityOverStr.c_str(),
                                                      "exceeds INT32_MAX; it multiplies the y_grad slot address "
                                                      "(slot = e*C + p), so a truncation would compute a wrong "
                                                      "address"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// CheckIndexValueRange() — 第 7 类校验：indices / locations 取值域
//
// 设计依据（唯一裁定，逐字同源）：
//   * Interface.md「越界取值域的两层口径」/ HostTiling.md §2.1 / InferShapeDtype.md §6：
//     接口 / 校验层**只拒绝真正会越界访存 y_grad 的两类取值** —— `indices >= B`
//     （slot = e*C + p >= B*C，越过张量末尾）与 `locations < 0`（slot < 0，下溢）；
//     **`indices < 0` 与 `locations >= C` 不由接口层拒绝**（属 A8 的 ok 掩码置 0 语义域，
//     由 kernel 负责输出精确 0），本函数因此**不判**这两类取值，也不得把它们改成拒绝。
//   * REQUIREMENTS §7.5 B1 与 §5.9.5：越界必须前置拦截，不得静默越界访存。
//
// 为什么落在 Tiling 层（本工程内「值可读层」的唯一落点，如实登记）：
//   接口设计把该拒绝指派给「aclnn 单算子模式的 GetWorkspaceSize 参数校验阶段」，但本仓库的
//   ACLNN 实现是 op_build 的**自动生成桩**（aclnn_*.cpp 由 op_build 生成、本仓库不覆盖），
//   没有可插入校验的钩子；tiling 回调是 aclnn 通路里本算子自有代码中唯一可返回失败的一段。
//
// ★ 读值口径：设备张量**不读取、也不作为拒绝依据**，只保留 host 侧张量的直读拒绝。
//   理由是本层拿不到判定「线性缓冲 == 逻辑元素序列」所需的任何信息：
//     a) aclnn 通路喂给 tiling 的描述符一律是 `gert::kTensorV1`（`GetVersion()==0`，即
//        「不携带非连续描述信息」）：`InputIsView(2/3)==false`、`GetInputStride(2/3)==nullptr`、
//        `GetInputOffset(2/3)==-1`，对**连续与非连续输入给出完全相同的结果**；
//     b) `GetInputShape(i)` 的 cur/storage/origin 三份 shape 全部等于**逻辑（视图）维**，
//        真实的 storage 维被丢弃；
//     c) `GetTensorData().GetSize()` 也恒为逻辑字节数；
//     d) 数据地址仍是**原始 storage 基址**（连续化发生在桩的 `aclnnContiguous` 里、在本次
//        tiling 之后），故按线性缓冲读 `S*B` 个元素读到的是**视图跳过的 storage 元素**
//        ⇒ 一个完全合法的非连续视图会被误拒，而误拒合法输入比漏检更严重。
//   于是这里取「宁可放行、不误拒」。放行是安全的：kernel 两侧分支都用四条件
//   `ok = (e>=0) && (e<B) && (p>=0) && (p<C)` 掩码，非法 task `continue` 且**不发起任何 DMA**，
//   槽位保持精确 0 ⇒ 既不越界访存，结果也仍是约定的 0 语义。
//   代价（如实登记）：设备通路上 `indices >= B` / `locations < 0` 不再以失败码报出，
//   而与 `indices < 0` / `locations >= C` 同口径由 kernel 置 0。
//
// 仍保留的拒绝路径（限定语：**host 放置且取值指针非空**才直读拒收）：
//   host 侧张量（kOnHost / kFollowing，例如常量折叠等**由框架把取值物化到 host** 的通路）
//   且 `GetData<int32_t>() != nullptr` → 直读，读到的就是逻辑元素序列本身，判据可信，
//   照旧 OP_LOGE_FOR_INVALID_VALUE_WITH_REASON + `GRAPH_FAILED`；
//   无张量描述 / host 放置但取值指针为空 → 记 INFO 放行，由 kernel 掩码兜底。
//   （该直读分支的 MoeTutelCombineGates 侧可达性由 `tests/ut/op_host/` 的
//    `MoeTutelCombineGatesTiling` 用例覆盖：tiling 用例默认构造的 tensor 即
//    host 放置但无数据指针，走放行分支。）
// ---------------------------------------------------------------------------
ge::graphStatus MoeTutelCombineGatesTiling::CheckIndexValueRange()
{
    // indices：只拒 e >= B（e < 0 不拒）
    if (CheckOneIndexTensor(2, batchSize, false) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // locations：只拒 p < 0（p >= C 不拒）
    if (CheckOneIndexTensor(3, capacity, true) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// rejectNegative == false : 违规判据 = (value >= upperBound)      —— indices
// rejectNegative == true  : 违规判据 = (value < 0)                —— locations
ge::graphStatus MoeTutelCombineGatesTiling::CheckOneIndexTensor(size_t inputIndex, int64_t upperBound,
                                                                bool rejectNegative)
{
    const char *const name = rejectNegative ? "locations" : "indices";
    const auto *tensor = tilingContext->GetInputTensor(inputIndex);
    if (tensor == nullptr) {
        return ge::GRAPH_SUCCESS; // 无张量描述：本层不可判（DynamicShape 场景由 kernel 掩码兜底）
    }
    const int64_t elems = tensor->GetShapeSize();
    if (elems <= 0) {
        return ge::GRAPH_SUCCESS; // 空张量无取值可违例（returns_empty 由 HandleEmptyTensor 承担）
    }

    const gert::TensorPlacement placement = tensor->GetPlacement();
    if (placement != gert::kOnHost && placement != gert::kFollowing) {
        // 设备张量：本层无法判断「按线性缓冲读到的 S*B 个 int32」是否就是逻辑元素序列
        // （见上方 ★ 读值口径实测），故不读、不拒，交 kernel 的四条件 ok 掩码兜底。
        OP_LOGI(tilingContext->GetNodeName(),
                "[MoeTutelCombineGates] value-range check skipped for %s: device tensor (placement=%d) carries "
                "no non-contiguous description (kTensorV1) and no stride/offset, so a linear read cannot be "
                "proved to be the logical element sequence; rejecting on it would false-reject legal "
                "non-contiguous views. Kernel-side four-condition ok mask remains the last line of defence.",
                name, static_cast<int>(placement));
        return ge::GRAPH_SUCCESS;
    }

    // host 侧张量：直读（读到的就是逻辑元素序列本身，判据可信）
    const int32_t *hostData = tensor->GetData<int32_t>();
    if (hostData == nullptr) {
        OP_LOGI(tilingContext->GetNodeName(),
                "[MoeTutelCombineGates] value-range check skipped for %s: host tensor carries no data "
                "pointer (kernel-side ok mask remains the last line of defence)",
                name);
        return ge::GRAPH_SUCCESS;
    }
    for (int64_t j = 0; j < elems; ++j) {
        const int64_t v = static_cast<int64_t>(hostData[j]);
        const bool bad = rejectNegative ? (v < 0) : (v >= upperBound);
        if (bad) {
            // 取值域违规（张量元素取值，非 shape/dtype）→ ..._VALUE_WITH_REASON（EZ0026）。
            const std::string badValueStr = std::string(name) + "[" + std::to_string(j) + "]=" + std::to_string(v);
            const std::string badReasonStr =
                std::string("rejected by the value-range check (") +
                (rejectNegative ? "locations < 0" : "indices >= B") + "; y_grad B=" + std::to_string(batchSize) +
                " C=" + std::to_string(capacity) + "). Note: " + (rejectNegative ? "locations >= C" : "indices < 0") +
                " is NOT rejected here (A8 zero-fill semantics is the kernel's job)";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), name, badValueStr.c_str(),
                                                  badReasonStr.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// ValidateInputs() — HostTiling §3.3: the check order is part of the contract
// (dtype -> format -> rank -> attr(N/A) -> shape -> numeric range), and the first
// failure returns GRAPH_FAILED without setting a tilingKey.
//
// The attr step is N/A (spec.yaml attributes: []) and therefore has no call here;
// the 6th class (UB capacity) needs the UB budget and runs after the dtype fork.
// ---------------------------------------------------------------------------
ge::graphStatus MoeTutelCombineGatesTiling::ValidateInputs()
{
    if (CheckDtypeAndCombination() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckFormat() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckExactRanks() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // attr value range: N/A — the operator declares no attribute.
    if (CheckShapeContract() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNumericRange() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // 第 7 类：取值域校验（Task 30 补齐，判决口径见 CheckIndexValueRange 注释）
    if (CheckIndexValueRange() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// MultiCoreSplit() — HostTiling §5.2
//
//   taskNum          = (int64_t)samples * batchSize          (B6: int64 intermediate)
//   taskNumPerBlock  = CeilAlign(CeilDiv(taskNum, aivCoreNum), OFFSET = 256)
//
// The single split axis is the flat task axis; H is never split (HostTiling §4.1),
// so there is no cross-core partial reduction and no workspace (A12). The load
// imbalance bound is taskNumPerBlock - 1 (HostTiling §5.2), reached when the last
// used core gets a single task.
// ---------------------------------------------------------------------------
ge::graphStatus MoeTutelCombineGatesTiling::MultiCoreSplit()
{
    taskNum = samples * batchSize; // B6: int64 intermediate, never int32 * int32
    const std::string taskNumStr = "taskNum=" + std::to_string(taskNum);
    const std::string taskNumReasonStr = "taskNum = samples*batchSize must fit the int32 TilingData fields "
                                         "(bound INT32_MAX - " +
                                         std::to_string(OFFSET) + ")";
    OP_CHECK_IF(taskNum > static_cast<int64_t>(INT32_MAX) - OFFSET,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "samples*batchSize",
                                                      taskNumStr.c_str(), taskNumReasonStr.c_str()),
                return ge::GRAPH_FAILED);

    const int64_t perBlock = CeilDiv(taskNum, static_cast<int64_t>(coreNum)); // ceil-average
    const int64_t aligned = CeilAlign(perBlock, OFFSET);                      // 256-aligned
    taskNumPerBlock = static_cast<int32_t>(aligned);                          // no overflow: bounded above
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// HandleEmptyTensor(td) — HostTiling §3.2
//
// taskNum == 0 (S == 0 or B == 0): the host skips the multicore split, the 5
// TilingData fields still carry the real shapes, taskNumPerBlock becomes 0, and
// the launch uses SetBlockDim(1) — SetBlockDim(0) is illegal and forbidden.
// Returns true when the case was short-circuited.
//
// The dtype fork and CheckUbCapacity() have ALREADY run when this returns true
// (RunTiling steps 5/6), so the key published on this path follows x.dtype and an
// over-large hidden has been rejected before the early exit (A1-3 / A1-4).
// ---------------------------------------------------------------------------
bool MoeTutelCombineGatesTiling::HandleEmptyTensor(MoeTutelCombineGatesTilingData *td)
{
    (void)td; // the TilingData fields are written once by FillAndLogTilingData below
    if (taskNum != 0) {
        return false;
    }
    taskNumPerBlock = 0;            // every core iterates zero times
    tilingContext->SetBlockDim(1U); // SetBlockDim(0) would be an illegal launch
    return true;
}

// ---------------------------------------------------------------------------
// FillAndLogTilingData(td) — HostTiling §7.2 (7/7 fields + full INFO logging)
//
// Field units: elements. The dtype does NOT enter TilingData — it is dispatched
// statically through the tilingKey (TilingData.md §2).
// ---------------------------------------------------------------------------
void MoeTutelCombineGatesTiling::FillAndLogTilingData(MoeTutelCombineGatesTilingData *td)
{
    // The two UB-size fields come from the SAME CalcUbTiers() call that the UB gate
    // used, so host budget and kernel allocation are one arithmetic (P-08 fix).
    const MoeTutelCombineGatesUbTiers tiers = CalcUbTiers(hidden, isFp32Path, blockSize, vectorSize);

    td->batchSize = static_cast<int32_t>(batchSize);         // B = y_grad.shape[0]
    td->taskNumPerBlock = taskNumPerBlock;                   // CeilAlign(CeilDiv(taskNum, aivCoreNum), 256)
    td->samples = static_cast<int32_t>(samples);             // S = x.shape[0]
    td->hidden = static_cast<int32_t>(hidden);               // H = y_grad.shape[2] = x.shape[1]
    td->capacity = static_cast<int32_t>(capacity);           // C = y_grad.shape[1]
    td->ubRowElems = static_cast<int32_t>(tiers.rowElems);   // elements per input-dtype H-long route
    td->ubFp32Elems = static_cast<int32_t>(tiers.fp32Elems); // elements per fp32 H-long route

    OP_LOGI(tilingContext->GetNodeName(),
            "[MoeTutelCombineGates] TilingData: batchSize=%d, taskNumPerBlock=%d, samples=%d, hidden=%d, "
            "capacity=%d, ubRowElems=%d, ubFp32Elems=%d",
            td->batchSize, td->taskNumPerBlock, td->samples, td->hidden, td->capacity, td->ubRowElems, td->ubFp32Elems);
    OP_LOGI(tilingContext->GetNodeName(),
            "[MoeTutelCombineGates] x=[%ld,%ld] y_grad=[%ld,%ld,%ld] indices/locations elems=%ld dtype=%d, "
            "taskNum=%ld, coreNum=%u, ubSize=%ld, blockSize=%ld, cacheLineSize=%ld, vectorSize=%ld",
            xShape.GetDim(0), xShape.GetDim(1), yShape.GetDim(0), yShape.GetDim(1), yShape.GetDim(2),
            idxShape.GetShapeSize(), static_cast<int32_t>(xDtype), taskNum, coreNum, ubSize, blockSize, cacheLineSize,
            vectorSize);
    OP_LOGI(tilingContext->GetNodeName(),
            "[MoeTutelCombineGates] path=%s P=%d perHiddenBytes=%ld ubNeedBytes=%ld ubSize=%ld hiddenMax=%ld",
            isFp32Path ? "fp32" : "16b", isFp32Path ? 9 : 12, isFp32Path ? 16L : 20L, tiers.needBytes, ubSize,
            CalcHiddenUpperBound(ubSize, isFp32Path, blockSize, UB_RESERVED, vectorSize));
}

// ---------------------------------------------------------------------------
// CheckUbCapacity() — HostTiling §4.2/§4.4 (the 6th validation class, which needs
// the UB budget and therefore runs after the dtype fork)
//
// hidden above the H upper bound is rejected explicitly: no H splitting and no
// silent downgrade. The budget is CalcUbTiers()'s byte sum over the P routes, i.e.
// exactly the sum of the sizes the same call writes into TilingData (ubRowElems /
// ubFp32Elems) — since P-08 the host no longer keeps a second, independently
// maintained sizing formula in the kernel, so "the host thinks it fits" and "the
// kernel allocates it" cannot diverge.
// ---------------------------------------------------------------------------
ge::graphStatus MoeTutelCombineGatesTiling::CheckUbCapacity()
{
    const int64_t ubAvail = ubSize - UB_RESERVED;
    const int64_t hiddenUpperBound = CalcHiddenUpperBound(ubSize, isFp32Path, blockSize, UB_RESERVED, vectorSize);
    const int64_t ubNeedBytes = CalcUbNeedBytes(hidden, isFp32Path, blockSize, vectorSize);
    const std::string ubHiddenStr = "hidden=" + std::to_string(hidden);
    const std::string ubReasonStr =
        std::string("exceeds the UB capacity bound ") + std::to_string(hiddenUpperBound) + " for the " +
        (isFp32Path ? "fp32" : "16b") + " path (ubNeedBytes=" + std::to_string(ubNeedBytes) +
        " > ubAvail=" + std::to_string(ubAvail) + "); H is not split and the bound is not silently relaxed";
    OP_CHECK_IF(hidden > hiddenUpperBound,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "hidden", ubHiddenStr.c_str(),
                                                      ubReasonStr.c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// SetWorkspaceSize() — A12: the operator needs no workspace (no cross-core
// partial reduction, no buffer relay), so the single workspace slot is 0.
// ---------------------------------------------------------------------------
ge::graphStatus MoeTutelCombineGatesTiling::SetWorkspaceSize()
{
    size_t *workspaces = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, workspaces);
    workspaces[0] = 0;
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// RunTiling() — the pipeline of HostTiling §9, with steps D/E (dtype fork + UB
// capacity) ahead of the empty-tensor short circuit F (see the class comment)
// ---------------------------------------------------------------------------
ge::graphStatus MoeTutelCombineGatesTiling::RunTiling()
{
    // ===== A. platform and inputs (independent of dtype, done first) =====
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParseInputShapes() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // ===== B. validation (order fixed, first failure returns) =====
    if (ValidateInputs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // ===== C. multicore split (independent of dtype) =====
    if (MultiCoreSplit() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto *td = tilingContext->GetTilingData<MoeTutelCombineGatesTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, td);

    // ===== D. the single dtype fork (UB budget path + DTYPE_MODE) =====
    // BranchRoute.md §2: the default is BF16_MODE and only DT_FLOAT / DT_FLOAT16
    // override it, in that order (reference implementation _tiling.cpp:79-84).
    // It runs BEFORE the empty-tensor short circuit (A1-3): the no-data path
    // publishes the key of its own dtype, not the bf16 default.
    isFp32Path = (xDtype == ge::DT_FLOAT);
    if (xDtype == ge::DT_FLOAT) {
        dtypeMode = MOE_TUTEL_COMBINE_GATES_FP32_MODE;
    } else if (xDtype == ge::DT_FLOAT16) {
        dtypeMode = MOE_TUTEL_COMBINE_GATES_FP16_MODE;
    } else {
        dtypeMode = MOE_TUTEL_COMBINE_GATES_BF16_MODE;
    }

    // ===== E. UB capacity check (the 6th validation class) =====
    // Before the empty-tensor short circuit (A1-4): taskNum == 0 must not bypass
    // the only gate that rejects hidden above the platform UB bound (HostTiling §4.4).
    if (CheckUbCapacity() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // ===== F. empty tensor short circuit (the only early exit) =====
    if (HandleEmptyTensor(td)) {
        // taskNumPerBlock is 0 here; the other four fields still carry the real
        // shapes (they may be 0). The dtype fork above already ran, so the key is
        // the one of this input's dtype (0 bf16 / 1 fp32 / 2 fp16) even though
        // there is no data path (HostTiling §3.2 / §9, revised order).
        FillAndLogTilingData(td);
        OP_LOGI(tilingContext->GetNodeName(),
                "[MoeTutelCombineGates] empty tensor (taskNum=0): batchSize=%d, samples=%d, hidden=%d, "
                "capacity=%d, taskNumPerBlock=0, blockDim=1, tilingKey=%lu",
                td->batchSize, td->samples, td->hidden, td->capacity,
                static_cast<unsigned long>(GET_TPL_TILING_KEY(dtypeMode)));
        tilingContext->SetTilingKey(GET_TPL_TILING_KEY(dtypeMode));
        return SetWorkspaceSize();
    }

    // ===== G. write the data and launch =====
    FillAndLogTilingData(td);
    // HostTiling §7.3: non-empty -> the AIV core count; the empty case above took
    // SetBlockDim(1). SetBlockDim(0) is impossible on both paths, and
    // min(usedCores, coreNum) is explicitly not used (§5.2).
    tilingContext->SetBlockDim(coreNum);
    tilingContext->SetTilingKey(GET_TPL_TILING_KEY(dtypeMode));
    return SetWorkspaceSize();
}

// ---------------------------------------------------------------------------
// TilingFuncMoeTutelCombineGates(context) — runtime tiling callback
//
// Called by the CANN runtime during graph compilation. It computes the public
// tiling quantities, writes the seven TilingData fields and sets blockDim /
// tilingKey / the (zero) workspace size.
//
// VALUE DEPENDENCY (independent-review P-05/A1-2, stating it as it is): no
// .TilingInputsDataDependency() is registered below, yet RunTiling() ->
// ValidateInputs() -> CheckIndexValueRange() DOES read the content of indices and
// locations whenever the framework hands them over as host tensors (kOnHost /
// kFollowing); on the device path (the aclnn thread) it logs and
// passes because tiling receives no stride / offset / kTensorV2 information there.
// The framework therefore owes nothing about data readiness at this point: the
// check is best-effort, and whenever a value cannot be read (device tensor, no data
// pointer, no tensor description) it logs at INFO and passes, leaving the kernel's
// `ok` mask as the last line of defence. The HostTiling §2.1 statement "the host
// never reads indices / locations" describes the reference implementation, not this
// one.
// ---------------------------------------------------------------------------
ge::graphStatus TilingFuncMoeTutelCombineGates(gert::TilingContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    MoeTutelCombineGatesTiling tiling(context);
    return tiling.RunTiling();
}

// ---------------------------------------------------------------------------
// TilingPrepareForMoeTutelCombineGates(context) — compile-time preparation
//
// Fills MoeTutelCombineGatesCompileInfo (coreNum = AIV core count, ubSize = UB
// bytes per core). The tiling function itself always queries the platform live
// (HostTiling §5.1), so this callback stays a carrier only.
// ---------------------------------------------------------------------------
ge::graphStatus TilingPrepareForMoeTutelCombineGates(gert::TilingParseContext *context)
{
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    auto compileInfo = context->GetCompiledInfo<MoeTutelCombineGatesCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto ap = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ap.GetCoreNumAiv();
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// IMPL_OP_OPTILING(MoeTutelCombineGates) — register the tiling callbacks:
//   .Tiling(TilingFuncMoeTutelCombineGates)                     runtime tiling
//   .TilingParse<MoeTutelCombineGatesCompileInfo>(...)          compile-time info
// (no .TilingInputsDataDependency is registered: the tiling callback reads
//  indices / locations in CheckIndexValueRange() only if the framework supplies them
//  as host tensors, and that read is deliberately best-effort — a device tensor or an
//  unavailable read logs at INFO and passes, so no readiness guarantee is required
//  from the framework. Do NOT read this absence as "the host does not touch the
//  inputs"; see TilingFuncMoeTutelCombineGates above.)
// ---------------------------------------------------------------------------
IMPL_OP_OPTILING(MoeTutelCombineGates)
    .Tiling(TilingFuncMoeTutelCombineGates)
    .TilingParse<MoeTutelCombineGatesCompileInfo>(TilingPrepareForMoeTutelCombineGates);

} // namespace optiling
