/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cast.cpp
 * \brief Cast kernel for Atlas 300V Pro / ascend310p (dav_m200).
 *
 * Supported software paths are kept in this file because dav_m200 lacks native
 * int64 and bf16 vector conversion primitives for the task-required pairs.
 */

#include "kernel_operator.h"

// dtype enum fallbacks (opc normally injects these defines)
#ifndef DT_FLOAT
#define DT_FLOAT 0
#endif
#ifndef DT_FLOAT16
#define DT_FLOAT16 1
#endif
#ifndef DT_INT8
#define DT_INT8 2
#endif
#ifndef DT_INT32
#define DT_INT32 3
#endif
#ifndef DT_UINT8
#define DT_UINT8 4
#endif
#ifndef DT_INT16
#define DT_INT16 6
#endif
#ifndef DT_INT64
#define DT_INT64 9
#endif
#ifndef DT_BOOL
#define DT_BOOL 12
#endif
#ifndef DT_BF16
#define DT_BF16 27
#endif

#ifndef ORIG_DTYPE_X
#define ORIG_DTYPE_X DT_FLOAT16
#endif
#ifndef ORIG_DTYPE_Y
#define ORIG_DTYPE_Y DT_FLOAT
#endif

#pragma pack(push, 8)
struct CastTilingData {
    int64_t blockNum;
    int64_t ubFormer;
    int64_t blockFormer;
    int64_t ubLoopOfFormerBlock;
    int64_t ubLoopOfTailBlock;
    int64_t ubTailOfFormerBlock;
    int64_t ubTailOfTailBlock;
};
#pragma pack(pop)

namespace cast_310p {
using AscendC::RoundMode;

constexpr int CAST_X = ORIG_DTYPE_X;
constexpr int CAST_Y = ORIG_DTYPE_Y;
constexpr int BUFFER_NUM = 2;
constexpr int64_t DATA_BLOCK_BYTES = 32;

// Storage type: bf16 input is uint16_t; int64 is two uint32_t lanes per element.
template <int D> struct DtMap;
template <> struct DtMap<DT_FLOAT> { using T = float; };
template <> struct DtMap<DT_FLOAT16> { using T = half; };
template <> struct DtMap<DT_INT32> { using T = int32_t; };
template <> struct DtMap<DT_INT16> { using T = int16_t; };
template <> struct DtMap<DT_INT8> { using T = int8_t; };
template <> struct DtMap<DT_UINT8> { using T = uint8_t; };
template <> struct DtMap<DT_BOOL> { using T = int8_t; };
template <> struct DtMap<DT_BF16> { using T = uint16_t; };
template <> struct DtMap<DT_INT64> { using T = uint32_t; };

__aicore__ constexpr int LanesOf(int d) { return d == DT_INT64 ? 2 : 1; }

__aicore__ constexpr bool IsDirectPair(int x, int y)
{
    return (x == DT_FLOAT16 && (y == DT_FLOAT || y == DT_INT32 || y == DT_INT8 || y == DT_UINT8 || y == DT_INT16)) ||
           (x == DT_FLOAT && (y == DT_FLOAT16 || y == DT_INT32)) ||
           (x == DT_INT32 && (y == DT_FLOAT || y == DT_FLOAT16 || y == DT_INT16)) ||
           (x == DT_INT16 && y == DT_FLOAT) ||
           ((x == DT_INT8 || x == DT_BOOL || x == DT_UINT8 || x == DT_INT16) && y == DT_FLOAT16);
}

// round mode aligned with TBE: float->int RINT, f16->i8/u8 ROUND (no RINT on m200), others NONE
__aicore__ constexpr RoundMode GetRound(int x, int y)
{
    if ((x == DT_FLOAT16 || x == DT_FLOAT) && (y == DT_INT32 || y == DT_INT16)) {
        return RoundMode::CAST_RINT;
    }
    if (x == DT_FLOAT16 && (y == DT_INT8 || y == DT_UINT8)) {
        return RoundMode::CAST_ROUND;
    }
    return RoundMode::CAST_NONE;
}

template <int X, int Y>
class CastKernel {
public:
    using TX = typename DtMap<X>::T;
    using TY = typename DtMap<Y>::T;
    static constexpr bool HAS64 = (X == DT_INT64 || Y == DT_INT64);
    static_assert(Y != DT_BF16, "BF16 output is not supported by this task");
    static constexpr bool HASBF = (X == DT_BF16);
    static constexpr bool DST_BOOL = (Y == DT_BOOL) && !HASBF;
    static constexpr bool SCALAR_I16_F32 = (X == DT_INT16 && Y == DT_FLOAT);
    static constexpr bool DIRECT = !DST_BOOL && !HAS64 && !HASBF && !SCALAR_I16_F32 && IsDirectPair(X, Y);
    static constexpr bool TWO_CAST = !DST_BOOL && !DIRECT && !HAS64 && !HASBF;
    static constexpr bool NEED_MID = TWO_CAST || DST_BOOL || HAS64 || HASBF;
    static constexpr bool NEED_DEQ = (X == DT_INT32 && (NEED_MID || Y == DT_FLOAT16));
    static constexpr bool NEED_SCALAR_MTE2_FENCE = HAS64 || HASBF || SCALAR_I16_F32;
    static constexpr int STAGE_BYTES = (HAS64 || HASBF) ? 4 : 2;

public:
    __aicore__ inline CastKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const CastTilingData& t)
    {
        int64_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= t.blockNum) {
            ubLoop_ = 0;
            return;
        }
        bool isTailBlock = (blockIdx == t.blockNum - 1);
        ubFormer_ = t.ubFormer;
        ubLoop_ = isTailBlock ? t.ubLoopOfTailBlock : t.ubLoopOfFormerBlock;
        ubTail_ = isTailBlock ? t.ubTailOfTailBlock : t.ubTailOfFormerBlock;
        int64_t blockOffset = blockIdx * t.blockFormer;

        xGm_.SetGlobalBuffer((__gm__ TX*)x + blockOffset * LanesOf(X));
        yGm_.SetGlobalBuffer((__gm__ TY*)y + blockOffset * LanesOf(Y));

        pipe_.InitBuffer(inQue_, BUFFER_NUM, ubFormer_ * sizeof(TX) * LanesOf(X));
        pipe_.InitBuffer(outQue_, BUFFER_NUM, ubFormer_ * sizeof(TY) * LanesOf(Y));
        if constexpr (NEED_MID) {
            pipe_.InitBuffer(midBuf_, ubFormer_ * STAGE_BYTES);
        }
        if constexpr (HAS64 || HASBF) {
            pipe_.InitBuffer(mid2Buf_, ubFormer_ * STAGE_BYTES);
        }
        if constexpr (NEED_DEQ) {
            AscendC::SetDeqScale(static_cast<half>(1.0));
        }
    }

    __aicore__ inline void Process()
    {
        for (int64_t i = 0; i < ubLoop_; ++i) {
            int64_t count = (i == ubLoop_ - 1) ? ubTail_ : ubFormer_;
            CopyIn(i, count);
            Compute(count);
            CopyOut(i, count);
        }
    }

private:
    __aicore__ inline int64_t CeilAlign(int64_t n, int64_t align)
    {
        return (n + align - 1) / align * align;
    }

    template <typename T>
    __aicore__ inline int64_t AlignDataBlockLanes(int64_t lanes)
    {
        return CeilAlign(lanes, DATA_BLOCK_BYTES / sizeof(T));
    }

    template <AscendC::HardEvent EVENT>
    __aicore__ inline void SyncEvent()
    {
        event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(EVENT));
        AscendC::SetFlag<EVENT>(evt);
        AscendC::WaitFlag<EVENT>(evt);
    }

    __aicore__ inline void CopyIn(int64_t loop, int64_t count)
    {
        AscendC::LocalTensor<TX> inLocal = inQue_.AllocTensor<TX>();
        int64_t lanes = count * LanesOf(X);
        AscendC::DataCopy(inLocal, xGm_[loop * ubFormer_ * LanesOf(X)], AlignDataBlockLanes<TX>(lanes));
        inQue_.EnQue(inLocal);
    }

    // i64 (u32 pairs) -> i32: scalar low-lane extraction avoids GatherMask UB OOB on large tiles.
    __aicore__ inline void Narrow64(const AscendC::LocalTensor<uint32_t>& src,
                                    const AscendC::LocalTensor<int32_t>& dst, int64_t count)
    {
        for (int64_t i = 0; i < count; ++i) {
            dst.SetValue(i, static_cast<int32_t>(src.GetValue(2 * i)));
        }
    }

    __aicore__ inline void SyncScalarToVector()
    {
        SyncEvent<AscendC::HardEvent::S_V>();
    }


    // i32 -> i64 (u32 pairs): scalar even-lane write + sign extension.
    __aicore__ inline void Widen64(const AscendC::LocalTensor<int32_t>& src,
                                   const AscendC::LocalTensor<uint32_t>& dst, int64_t count)
    {
        for (int64_t i = 0; i < count; ++i) {
            int32_t v = src.GetValue(i);
            dst.SetValue(2 * i, static_cast<uint32_t>(v));
            dst.SetValue(2 * i + 1, v < 0 ? 0xFFFFFFFFu : 0u);
        }
    }

    // bf16 (u16) -> f32: bit splice into high16, lossless.
    __aicore__ inline void Bf16ToF32(const AscendC::LocalTensor<uint16_t>& src,
                                     const AscendC::LocalTensor<float>& dst, int64_t count)
    {
        AscendC::LocalTensor<uint32_t> d32 = dst.ReinterpretCast<uint32_t>();
        for (int64_t i = 0; i < count; ++i) {
            d32.SetValue(i, static_cast<uint32_t>(src.GetValue(i)) << 16);
        }
    }

    __aicore__ inline void Bf16ToBool(const AscendC::LocalTensor<uint16_t>& src,
                                      const AscendC::LocalTensor<int8_t>& dst, int64_t count)
    {
        for (int64_t i = 0; i < count; ++i) {
            dst.SetValue(i, src.GetValue(i) == 0 ? static_cast<int8_t>(0) : static_cast<int8_t>(1));
        }
    }

    __aicore__ inline void Compute(int64_t count)
    {
        AscendC::LocalTensor<TX> inLocal = inQue_.DeQue<TX>();
        AscendC::LocalTensor<TY> outLocal = outQue_.AllocTensor<TY>();
        uint32_t calc = static_cast<uint32_t>(CeilAlign(count, 32));
        if constexpr (NEED_SCALAR_MTE2_FENCE) {
            SyncEvent<AscendC::HardEvent::MTE2_S>();
        }
        if constexpr (SCALAR_I16_F32) {
            for (int64_t i = 0; i < count; ++i) {
                outLocal.SetValue(i, static_cast<float>(inLocal.GetValue(i)));
            }
        } else if constexpr (DIRECT) {
            AscendC::Cast(outLocal, inLocal, GetRound(X, Y), calc);
        } else if constexpr (DST_BOOL) {
            AscendC::LocalTensor<half> mid = midBuf_.Get<half>();
            if constexpr (X == DT_FLOAT16) {
                AscendC::Abs(mid, inLocal, calc);
            } else {
                AscendC::Cast(mid, inLocal, RoundMode::CAST_NONE, calc);
                AscendC::Abs(mid, mid, calc);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins(mid, mid, static_cast<half>(1.0), calc);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(outLocal, mid, RoundMode::CAST_CEIL, calc);
        } else if constexpr (TWO_CAST) {
            AscendC::LocalTensor<half> mid = midBuf_.Get<half>();
            AscendC::Cast(mid, inLocal, RoundMode::CAST_NONE, calc);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(outLocal, mid, GetRound(DT_FLOAT16, Y), calc);
        } else if constexpr (Y == DT_INT64) {  // includes bf16 -> i64
            ComputeTo64(inLocal, outLocal, count, calc);
        } else if constexpr (X == DT_INT64) {
            ComputeFrom64(inLocal, outLocal, count, calc);
        } else {  // X == DT_BF16
            ComputeFromBf16(inLocal, outLocal, count, calc);
        }
        if constexpr (NEED_SCALAR_MTE2_FENCE) {
            SyncEvent<AscendC::HardEvent::S_MTE3>();
        }
        inQue_.FreeTensor(inLocal);
        outQue_.EnQue(outLocal);
    }

    __aicore__ inline void ComputeFrom64(const AscendC::LocalTensor<TX>& inLocal,
                                         const AscendC::LocalTensor<TY>& outLocal, int64_t count, uint32_t calc)
    {
        AscendC::LocalTensor<uint32_t> src = inLocal.template ReinterpretCast<uint32_t>();
        if constexpr (Y == DT_INT32) {
            Narrow64(src, outLocal.template ReinterpretCast<int32_t>(), count);
        } else if constexpr (Y == DT_FLOAT) {
            AscendC::LocalTensor<int32_t> mid = midBuf_.Get<int32_t>();
            Narrow64(src, mid, count);
            SyncScalarToVector();
            AscendC::Cast(outLocal, mid, RoundMode::CAST_NONE, calc);  // i32->f32
        } else {  // Y == DT_FLOAT16: avoid i32->f16 deq (loses sign on overflow); go i32->f32->f16
            AscendC::LocalTensor<int32_t> mi = midBuf_.Get<int32_t>();
            Narrow64(src, mi, count);
            SyncScalarToVector();
            AscendC::LocalTensor<float> mf = mid2Buf_.Get<float>();
            AscendC::Cast(mf, mi, RoundMode::CAST_NONE, calc);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(outLocal, mf, RoundMode::CAST_NONE, calc);
        }
    }

    __aicore__ inline void ComputeTo64(const AscendC::LocalTensor<TX>& inLocal,
                                       const AscendC::LocalTensor<TY>& outLocal, int64_t count, uint32_t calc)
    {
        AscendC::LocalTensor<uint32_t> dst = outLocal.template ReinterpretCast<uint32_t>();
        if constexpr (X == DT_INT32) {
            Widen64(inLocal.template ReinterpretCast<int32_t>(), dst, count);
        } else if constexpr (X == DT_FLOAT) {
            AscendC::LocalTensor<int32_t> mid = midBuf_.Get<int32_t>();
            AscendC::Cast(mid, inLocal, RoundMode::CAST_RINT, calc);
            AscendC::PipeBarrier<PIPE_V>();
            SyncEvent<AscendC::HardEvent::V_S>();
            Widen64(mid, dst, count);
        } else {  // DT_BF16 -> i64: bits->f32->i32->widen
            AscendC::LocalTensor<float> midF = midBuf_.Get<float>();
            AscendC::LocalTensor<int32_t> midI = mid2Buf_.Get<int32_t>();
            Bf16ToF32(inLocal.template ReinterpretCast<uint16_t>(), midF, count);
            SyncEvent<AscendC::HardEvent::S_V>();
            AscendC::Cast(midI, midF, RoundMode::CAST_RINT, calc);
            SyncEvent<AscendC::HardEvent::V_S>();
            Widen64(midI, dst, count);
        }
    }

    __aicore__ inline void ComputeFromBf16(const AscendC::LocalTensor<TX>& inLocal,
                                           const AscendC::LocalTensor<TY>& outLocal, int64_t count, uint32_t calc)
    {
        AscendC::LocalTensor<uint16_t> src = inLocal.template ReinterpretCast<uint16_t>();
        if constexpr (Y == DT_FLOAT) {
            Bf16ToF32(src, outLocal.template ReinterpretCast<float>(), count);
            return;
        }
        if constexpr (Y == DT_BOOL) {
            Bf16ToBool(src, outLocal.template ReinterpretCast<int8_t>(), count);
            return;
        }
        AscendC::LocalTensor<float> midF = midBuf_.Get<float>();
        Bf16ToF32(src, midF, count);
        SyncEvent<AscendC::HardEvent::S_V>();
        if constexpr (Y == DT_FLOAT16) {
            AscendC::Cast(outLocal, midF, RoundMode::CAST_NONE, calc);
        } else if constexpr (Y == DT_INT32) {
            AscendC::Cast(outLocal, midF, RoundMode::CAST_RINT, calc);
        } else if constexpr (Y == DT_INT16) {
            AscendC::LocalTensor<int32_t> midI = mid2Buf_.Get<int32_t>();
            AscendC::Cast(midI, midF, RoundMode::CAST_RINT, calc);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(outLocal, midI, RoundMode::CAST_NONE, calc);  // vcbd s32->s16
        } else if constexpr (Y == DT_INT8 || Y == DT_UINT8) {
            AscendC::LocalTensor<half> midH = mid2Buf_.Get<half>();
            AscendC::Cast(midH, midF, RoundMode::CAST_NONE, calc);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(outLocal, midH, RoundMode::CAST_ROUND, calc);
        }
    }

    __aicore__ inline void CopyOut(int64_t loop, int64_t count)
    {
        AscendC::LocalTensor<TY> outLocal = outQue_.DeQue<TY>();
        int64_t lanes = count * LanesOf(Y);
        AscendC::DataCopy(yGm_[loop * ubFormer_ * LanesOf(Y)], outLocal, AlignDataBlockLanes<TY>(lanes));
        outQue_.FreeTensor(outLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> midBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mid2Buf_;
    AscendC::GlobalTensor<TX> xGm_;
    AscendC::GlobalTensor<TY> yGm_;
    int64_t ubFormer_ = 0;
    int64_t ubLoop_ = 0;
    int64_t ubTail_ = 0;
};
}  // namespace cast_310p

// CAST_UT_NO_ENTRY lets the kernel UT #include this TU for direct CastKernel<X,Y> instantiation
// without re-emitting the cast entry symbol (which the separately-compiled KernelFile already provides).
#ifndef CAST_UT_NO_ENTRY
extern "C" __global__ __aicore__ void cast(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
#ifndef __CCE_KT_TEST__
    REGISTER_TILING_DEFAULT(CastTilingData);
#endif
    GET_TILING_DATA(tilingData, tiling);
    cast_310p::CastKernel<cast_310p::CAST_X, cast_310p::CAST_Y> op;
    op.Init(x, y, tilingData);
    op.Process();
}
#endif  // CAST_UT_NO_ENTRY
