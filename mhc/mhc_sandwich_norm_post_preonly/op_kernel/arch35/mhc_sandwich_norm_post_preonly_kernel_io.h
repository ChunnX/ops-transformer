/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// A5单AIV与双sibling AIV路径共用的输入输出辅助函数。

__aicore__ inline void QuantizeRoundTrip(LocalTensor<float> buf, uint32_t count)
{
    LocalTensor<T> tmp = bufHInPrimeBf16_.AllocTensor<T>();
    CastToIO(tmp, buf, count);
    PipeBarrier<PIPE_V>();
    Cast(buf, tmp, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    bufHInPrimeBf16_.FreeTensor(tmp);
}

__aicore__ inline void CastToIO(LocalTensor<T> dst, LocalTensor<float> src, uint32_t count)
{
    if constexpr (IsSameT<T, half>::value) {
        Cast(dst, src, RoundMode::CAST_NONE, count);
    } else {
        Cast(dst, src, RoundMode::CAST_RINT, count);
    }
}

__aicore__ inline void FenceMte3ToMte2()
{
    SetFlag<HardEvent::MTE3_MTE2>(evtMte3toMte2_);
    WaitFlag<HardEvent::MTE3_MTE2>(evtMte3toMte2_);
}

__aicore__ inline void FenceMte3ToV()
{
    SetFlag<HardEvent::MTE3_V>(evtMte3toV_);
    WaitFlag<HardEvent::MTE3_V>(evtMte3toV_);
}

__aicore__ inline void FenceVToMte2()
{
    SetFlag<HardEvent::V_MTE2>(evtVtoMte2_);
    WaitFlag<HardEvent::V_MTE2>(evtVtoMte2_);
}

__aicore__ inline void LoadLowPrecisionTile(
    GlobalTensor<T> &src, uint64_t gmOffset, LocalTensor<float> dst, uint32_t count)
{
    LocalTensor<T> tmp = bufHOutBf16_.AllocTensor<T>();
    DataCopy(tmp, src[gmOffset], count);
    bufHOutBf16_.EnQue(tmp);
    tmp = bufHOutBf16_.DeQue<T>();
    Cast(dst, tmp, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    bufHOutBf16_.FreeTensor(tmp);
}

__aicore__ inline void WriteLowPrecisionTile(
    GlobalTensor<T> &dst, uint64_t gmOffset, LocalTensor<float> src, uint32_t count)
{
    LocalTensor<T> tmp = bufHInPrimeBf16_.AllocTensor<T>();
    CastToIO(tmp, src, count);
    bufHInPrimeBf16_.EnQue(tmp);
    tmp = bufHInPrimeBf16_.DeQue<T>();
    DataCopy(dst[gmOffset], tmp, count);
    bufHInPrimeBf16_.FreeTensor(tmp);
}

__aicore__ inline void WriteHInPrimeTile(LocalTensor<float> src, uint64_t gmOffset, uint32_t count)
{
    WriteLowPrecisionTile(hInPrimeGm_, gmOffset, src, count);
    if constexpr (RETURN_H_IN_F32) {
        SetFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        WaitFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        DataCopy(hInF32Gm_[gmOffset], src, count);
        FenceMte3ToV();
    }
}

__aicore__ inline void WriteX2OutTile(LocalTensor<float> src, uint64_t gmOffset, uint32_t count)
{
    WriteLowPrecisionTile(x2OutGm_, gmOffset, src, count);
}

__aicore__ inline uint32_t GetTileD(uint32_t tile) const
{
    if (lastDimTile_ != 0 && tile == dimLoop_ - 1) {
        return lastDimTile_;
    }
    return dimTile_;
}
