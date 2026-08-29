/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file vf_mask_invalid_rows_mxfp4.h
 * \brief 把 chunk 内超出行置 MIN_VALUE(-65504), 供各 softmax VF 预计算前调用。
 *        由 aicore 在调用 softmax VF 之前执行, 保证预计算(读首组 32 行)读到的是 mask 后的数据。
 *        掩码行区间与原各 softmax VF 内部语义一致:
 *          行 [effClamped, min((validGroups+1)*32, rows)) 写 MIN_VALUE, 其中
 *            effClamped = min(validRows, rows)   -- 有效行数截断到块行数
 *            validGroups = ceil(effClamped/32)   -- 有效组数向上取整
 *          覆盖: 1) 最后一个有效组内超出行 [effClamped%32, 32) (组内计算读到 MIN_VALUE 后 exp 下溢为 0);
 *               2) 组 validGroups 整组 (组循环最后一轮 next 预取读到, 若为 MM1 真实数据会污染
 *                  group_gmax/global_max, 跨 chunk online rescale 的 alpha)。
 *          rows 为当前块/大块的行数(KVS 上界): kvs32=32, kvs256 每大块=128, multi chunk0/chunk1 依 actKvsTile 截断,
 *          保证不写到当前块 KVS 范围之外。
 *        validRows==0(整块无有效行) 时循环不执行:
 *         避免 -inf 污染预计算 -> scale 255/NaN, 与各 softmax VF 内部 mask 语义一致。
 */

#ifndef VF_MASK_INVALID_ROWS_MXFP4_H
#define VF_MASK_INVALID_ROWS_MXFP4_H
#include "vf_common_def.h"

namespace NpuArch::Epilogue::Block::Mxfp4VF {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;

// ============================================================================
// 公共 mask: 把 chunk 内超出行置 MIN_VALUE(-65504), 保证对 VF 预计算可见。
//   s:             S buffer 首地址(物理行宽 QsBase=128 half = 256B)
//   chunkRowBase:  当前 chunk 起始全局行 (chunk0=0, chunk1=128)
//   validRows:     当前 chunk 有效行数(effY)
//   rows:          当前 chunk 总行数(KVS 上界, 128 或 32 或 actKvsTile 截断值)
// QS64=true 时仅写每行前 64 half(VL64), 否则写整行 128 half(ALL)。
// ============================================================================
template <typename T, bool QS64 = false>
__simd_vf__ inline void mask_invalid_rows_to_min_value_vf(__ubuf__ T *s, uint16_t chunkRowBase, uint16_t validRows,
                                                          uint16_t rows)
{
    constexpr uint16_t QsBase = 128;
    constexpr uint16_t ROWS_PER_GROUP = 32;

    const uint16_t effClamped = validRows < rows ? validRows : rows;
    const uint16_t validGroups = (effClamped + ROWS_PER_GROUP - 1) / ROWS_PER_GROUP;
    const uint16_t writeEnd = ((validGroups + 1) * ROWS_PER_GROUP) < rows
                                  ? ((validGroups + 1) * ROWS_PER_GROUP)
                                  : rows;

    MaskReg mask = CreateMask<uint16_t, (QS64 ? MaskPattern::VL64 : MaskPattern::ALL)>();
    RegTensor<T> min_val_reg;
    Duplicate(min_val_reg, MIN_VALUE, mask);

    for (uint16_t r = effClamped; validGroups > 0 && r < writeEnd; ++r) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(s + ((chunkRowBase + r) * QsBase) * 2, min_val_reg, mask);
    }
}

template <typename T, bool QS64 = false>
__aicore__ inline void MaskInvalidRowsToMinValueCallVF(const LocalTensor<T> &srcTensor, uint16_t chunkRowBase,
                                                       uint16_t validRows, uint16_t rows)
{
    __ubuf__ T *s_ub = (__ubuf__ T *)srcTensor.GetPhyAddr();

    mask_invalid_rows_to_min_value_vf<T, QS64>(s_ub, chunkRowBase, validRows, rows);
}

} // namespace NpuArch::Epilogue::Block::Mxfp4VF
#endif // VF_MASK_INVALID_ROWS_MXFP4_H
