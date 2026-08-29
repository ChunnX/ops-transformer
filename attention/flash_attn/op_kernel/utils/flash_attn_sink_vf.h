/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attn_sink_vf.h
 * \brief sink 注入展开链的 regbase VF 移植(替代 membase 的 Brcb + SinkExpand + WholeReduceMax + DataCopy)。
 *
 * 数据流与旧链完全一致(布局/对齐语义不变):
 *   阶段A(展开): expandUb[r*8 .. r*8+8) = castUb[idx(r)] —— 等价旧 Brcb(1->8) + SinkExpand 逐行拷贝;
 *                每行 8 float = 32B, StoreAlign 目的地址天然对齐; BRC_B32 单值广播 load 仅需 4B 对齐。
 *   阶段B(归约): maxUb[8g .. 8g+8) = ReduceDataBlock<MAX>(expandUb[64g .. 64g+64))
 *                —— 等价旧 WholeReduceMax(ORDER_ONLY_VALUE) 逐 8 元素归约 + 末尾 DataCopy。
 *   归约尾部越界行为与旧链一致(Align(m,8) 向上取整, 尾部为残留值, 消费方只读 [0,m))。
 * 行->源索引 idx(r) 与旧 SinkExpandGS1/SinkExpandS1G 逐分支等价:
 *   GS1(BNSD): idx(r) = (gs1Start%actS1Size + r) / actS1Size, 计数器推进免除法;
 *   S1G case1(单 s1 行): idx(r) = r;
 *   S1G case2.1(两行, gStart>gEnd): 头段 idx 从 Align(gEnd+1,16) 起, 尾段 idx 从 0 起(区带 gap 布局);
 *   S1G case2.2/3(整 gSize 区): idx(r) = (gStart + r) % gSize, 回绕计数器免取模。
 * API 依据: asc-devkit SIMD-API(LoadAlign DIST_BRC_B32 4B 对齐 / StoreAlign DIST_NORM_B32 32B 对齐 /
 * ReduceDataBlock vcgmax / LocalMemBar VEC_STORE->VEC_LOAD)。
 * 设计文档: my-skills/docs/25_flash_attn_sinks_vf_perf_plan.md。
 */
#ifndef FLASH_ATTN_SINK_VF_H_
#define FLASH_ATTN_SINK_VF_H_

#include "kernel_operator.h"

namespace AscendC {
namespace FaSinkVfApi {
using namespace MicroAPI;

constexpr static uint32_t SINK_BRCB_NUM = 8U; // 每值 8 lane(32B), 与旧 Brcb 布局一致

// b16 -> b32 宽转换: roundMode 不生效(asc-devkit Cast.md 表 7); RegLayout::ZERO 与
// DIST_BRC_B16 的排布配套(值位于每个 32bit 组的低 16bit)。
constexpr static MicroAPI::CastTrait CAST_TRAIT_SINK_B16_TO_B32 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

// 单行发射: expandUb[row*8 .. row*8+8) = inUb[idx](单值广播 + reg 内 16->32 cast, mask 8)
// 源直接读 INPUT_T 区(免张量 Cast 与其后的 PipeBarrier), 寄存器依赖自动保序
template <typename T, typename INPUT_T>
__simd_callee__ inline void SinkEmitRow(__ubuf__ T *expandUb, __ubuf__ INPUT_T *inUb, uint32_t row, uint32_t idx,
                                        MaskReg &pregRow)
{
    RegTensor<INPUT_T> srcReg;
    RegTensor<T> valReg;
    MaskReg pregCast = CreateMask<INPUT_T, MicroAPI::MaskPattern::ALL>();
    LoadAlign<INPUT_T, MicroAPI::LoadDist::DIST_BRC_B16>(srcReg, inUb + idx);
    MicroAPI::Cast<T, INPUT_T, CAST_TRAIT_SINK_B16_TO_B32>(valReg, srcReg, pregCast);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(expandUb + row * SINK_BRCB_NUM, valReg, pregRow);
}

// GS1(BNSD): idx(r) = (s0 + r) / actS1Size —— 计数器推进, 免除法
template <typename T, typename INPUT_T>
__simd_vf__ inline void SinkExpandGs1VF(__ubuf__ T *expandUb, __ubuf__ INPUT_T *inUb, uint32_t m, uint32_t s0,
                                        uint32_t actS1Size)
{
    MaskReg pregRow = CreateMask<T, MicroAPI::MaskPattern::VL8>();
    uint32_t rowsLeft = (s0 == 0) ? actS1Size : (actS1Size - s0);
    uint32_t idx = 0;
    for (uint16_t r = 0; r < static_cast<uint16_t>(m); ++r) {
        if (rowsLeft == 0) {
            idx++;
            rowsLeft = actS1Size;
        }
        SinkEmitRow<T, INPUT_T>(expandUb, inUb, r, idx, pregRow);
        rowsLeft--;
    }
}

// S1G 连续值段: rows [rowStart, rowStart+rowCount), idx = idxStart + (r - rowStart)
template <typename T, typename INPUT_T>
__simd_vf__ inline void SinkExpandRunVF(__ubuf__ T *expandUb, __ubuf__ INPUT_T *inUb, uint32_t rowStart,
                                        uint32_t rowCount, uint32_t idxStart)
{
    MaskReg pregRow = CreateMask<T, MicroAPI::MaskPattern::VL8>();
    uint32_t idx = idxStart;
    for (uint16_t r = 0; r < static_cast<uint16_t>(rowCount); ++r) {
        SinkEmitRow<T, INPUT_T>(expandUb, inUb, rowStart + r, idx, pregRow);
        idx++;
    }
}

// S1G 旋转(整 gSize 区): idx(r) = (gStart + r) % gSize —— 回绕计数器, 免取模
template <typename T, typename INPUT_T>
__simd_vf__ inline void SinkExpandRotVF(__ubuf__ T *expandUb, __ubuf__ INPUT_T *inUb, uint32_t m, uint32_t gStart,
                                        uint32_t gSize)
{
    MaskReg pregRow = CreateMask<T, MicroAPI::MaskPattern::VL8>();
    uint32_t idx = gStart;
    for (uint16_t r = 0; r < static_cast<uint16_t>(m); ++r) {
        SinkEmitRow<T, INPUT_T>(expandUb, inUb, r, idx, pregRow);
        idx++;
        if (idx == gSize) {
            idx = 0;
        }
    }
}

// 阶段B: 每 8 行一组, ReduceDataBlock<MAX> 得 8 个行 max, 直接写 maxUb(组间 32B 对齐)
template <typename T>
__simd_vf__ inline void SinkReduceMaxVF(__ubuf__ T *maxUb, __ubuf__ T *expandUb, uint32_t m)
{
    // 等待阶段A(或前置 VF)的 VEC_STORE 完成, 保证 expandUb 可读
    LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    RegTensor<T> srcReg;
    MaskReg pregAll = CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MaskReg pregBlk = CreateMask<T, MicroAPI::MaskPattern::VL8>();
    uint32_t groups = (m + SINK_BRCB_NUM - 1) / SINK_BRCB_NUM;
    for (uint16_t g = 0; g < static_cast<uint16_t>(groups); ++g) {
        LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(srcReg, expandUb + g * 64);
        MicroAPI::ReduceDataBlock<MicroAPI::ReduceType::MAX>(srcReg, srcReg, pregAll);
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(maxUb + g * SINK_BRCB_NUM, srcReg, pregBlk);
    }
}

/*!
 * \brief sink 展开 dispatcher(V 锁内调用)。源直接读 INPUT_T 值区(reg 内 16->32 cast, 免张量 Cast)。
 * \param maxUb        [out] softmaxMax 槽位(64 float)
 * \param sinkTmpUb    [in]  INPUT_T 值区(CopyIn 目标, 布局不变)
 * \param sinkExpandUb [tmp] 展开区(容量 >= actVecMSize*8 float, 旧链同位置)
 */
template <typename T, typename INPUT_T, bool IS_GS1_LAYOUT>
__aicore__ inline void SinkExpandMaxVf(const LocalTensor<T> &maxUb, const LocalTensor<INPUT_T> &sinkTmpUb,
                                       const LocalTensor<T> &sinkExpandUb, uint32_t gs1Start, uint32_t actVecMSize,
                                       uint32_t actS1Size, uint32_t gSize)
{
    if (actVecMSize == 0) {
        return;
    }
    __ubuf__ T *maxUbPtr = (__ubuf__ T *)maxUb.GetPhyAddr();
    __ubuf__ INPUT_T *inUb = (__ubuf__ INPUT_T *)sinkTmpUb.GetPhyAddr();
    __ubuf__ T *expandUb = (__ubuf__ T *)sinkExpandUb.GetPhyAddr();

    if constexpr (IS_GS1_LAYOUT) {
        uint32_t s0 = gs1Start % actS1Size;
        SinkExpandGs1VF<T, INPUT_T>(expandUb, inUb, actVecMSize, s0, actS1Size);
    } else {
        uint32_t gStart = gs1Start % gSize;
        uint32_t s1Start = gs1Start / gSize;
        uint32_t gs1End = gs1Start + actVecMSize - 1;
        uint32_t gEnd = gs1End % gSize;
        uint32_t s1End = gs1End / gSize;
        if (s1Start == s1End) {
            // 情况1: 单 s1 行, 区[0..gCount) 连续, idx(r) = r
            SinkExpandRunVF<T, INPUT_T>(expandUb, inUb, 0, actVecMSize, 0);
        } else if (s1Start + 1 == s1End && gStart > gEnd) {
            // 情况2.1: 头段在区[Align(gEnd+1,16) ..), 尾段在区[0 .. gEnd](区带 gap 布局)
            uint32_t headOff = (gEnd + 1 + 15U) & ~15U;
            uint32_t headLen = gSize - gStart;
            SinkExpandRunVF<T, INPUT_T>(expandUb, inUb, 0, headLen, headOff);
            SinkExpandRunVF<T, INPUT_T>(expandUb, inUb, headLen, gEnd + 1, 0);
        } else {
            // 情况2.2/3: 整 gSize 区, idx(r) = (gStart + r) % gSize
            SinkExpandRotVF<T, INPUT_T>(expandUb, inUb, actVecMSize, gStart, gSize);
        }
    }
    SinkReduceMaxVF<T>(maxUbPtr, expandUb, actVecMSize);
}
} // namespace FaSinkVfApi
} // namespace AscendC

#endif // FLASH_ATTN_SINK_VF_H_
