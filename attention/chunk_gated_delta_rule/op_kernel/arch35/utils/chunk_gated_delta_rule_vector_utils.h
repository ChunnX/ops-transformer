/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "chunk_gated_delta_rule_utils.h"
#include "basic_api/kernel_tensor.h"

namespace ChunkGatedDeltaRule {

using namespace AscendC;

// 通用 Vector 搬运/格式转换：只依赖 tensor、shape 和 stride。
// 调用方管理 barrier、Lock/Unlock 和跨核握手；本文件不读取 KernelContext 或固定缓冲偏移。

// MTE2：GM [rows,cols] -> UB [rows,dstRowStride]；行跨度均以元素传入，内部换算为字节/块。
// DMA 仅补齐最后一个 32B 块，完整 tile 的尾部仍由各计算路径显式清零或用 mask 排除。
template <uint32_t dstRowStride, typename T>
__aicore__ inline void CopyRowsGmToUb(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t rows, uint32_t cols,
                                      uint32_t srcRowStride)
{
    constexpr uint32_t elementsPerBlock = BLOCK_SIZE / sizeof(T);
    const uint32_t alignedCols = CeilAlign(cols, elementsPerBlock);
    DataCopyExtParams copy{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(T)),
                           static_cast<uint32_t>((srcRowStride - cols) * sizeof(T)),
                           static_cast<int64_t>((dstRowStride - alignedCols) / elementsPerBlock), 0};
    DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(alignedCols - cols), static_cast<T>(0)};
    DataCopyPad(dst, src, copy, pad);
}

// MTE3：UB [validRows,srcRowStride] -> GM 行跨度 dstRowStride；仅写回有效 [validRows,validCols]。
template <uint32_t srcRowStride, typename T>
__aicore__ inline void CopyRowsUbToGm(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t validRows, uint32_t validCols,
                                      uint32_t dstRowStride)
{
    if (validRows == 0) {
        return;
    }
    constexpr uint32_t elementsPerBlock = BLOCK_SIZE / sizeof(T);
    const uint32_t alignedCols = CeilAlign(validCols, elementsPerBlock);
    DataCopyExtParams params{static_cast<uint16_t>(validRows), static_cast<uint32_t>(validCols * sizeof(T)),
                             static_cast<uint32_t>((srcRowStride - alignedCols) / elementsPerBlock),
                             static_cast<int64_t>((dstRowStride - validCols) * sizeof(T)), 0};
    DataCopyPad(dst, src, params);
}

template <uint32_t rows, uint32_t cols, uint32_t fullRows, uint32_t matrixCount = 1>
__aicore__ inline void CopyCompactNzUbToL1(LocalTensor<bfloat16_t> src, LocalTensor<bfloat16_t> dst, uint32_t rowBegin)
{
    // MTE3：UB [matrixCount,rows,cols] -> L1[:,rowBegin:rowBegin+rows,:]。
    // NZ 布局为 [matrixCount,cols/16,rows,16]，每行恰好一个 32B 块；L1 行跨度为 fullRows。
    // 优化方案：相邻矩阵合并为一次 DMA；静态模板参数集中推导块数/长度/间隔。
    static_assert(rows % MM_BLOCK_CUBE == 0 && fullRows % MM_BLOCK_CUBE == 0, "NZ rows must be fractal aligned");
    static_assert(cols % MM_NZ_FRACTAL == 0 && rows <= fullRows, "the aligned UB tile must fit into the L1 matrix");
    DataCopyParams params{static_cast<uint16_t>(matrixCount * cols / MM_NZ_FRACTAL), static_cast<uint16_t>(rows), 0,
                          static_cast<uint16_t>(fullRows - rows)};
    DataCopy(dst[GetZnOffset(rowBegin, 0, fullRows)], src, params);
}

// MTE2：以 srcStride 元素为间隔读取 count 个标量，每个标量在 UB 独占一个 32B 块。
// GM [count] strided -> UB [count,BLOCK_SIZE/sizeof(T)]，其余元素补零，便于 VF 广播。
template <typename T>
__aicore__ inline void CopyScalarsGmToUb(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t count, uint32_t srcStride)
{
    constexpr uint32_t elementsPerBlock = BLOCK_SIZE / sizeof(T);
    DataCopyExtParams copy{static_cast<uint16_t>(count), sizeof(T), static_cast<int64_t>((srcStride - 1) * sizeof(T)),
                           0, 0};
    DataCopyPadExtParams<T> pad{true, 0, elementsPerBlock - 1, static_cast<T>(0)};
    DataCopyPad(dst, src, copy, pad);
}

// VF 使用 UB 指针；LocalTensor 仍负责描述物理地址空间。
template <typename T>
__aicore__ inline __ubuf__ T* UbAddress(LocalTensor<T> tensor)
{
    return reinterpret_cast<__ubuf__ T*>(tensor.GetPhyAddr());
}

// V：FP32 [rowCount,colCount] -> BF16 NZ [colCount/16,rowCount,16]，CAST_RINT 舍入。
// 每行放入一个 FP32 寄存器，要求 colCount<=VECTOR_REG_WIDTH/sizeof(float)，行首满足 LoadAlign 对齐。
// Pack 后按16元素块跨步写出，供 UB->L1 直接搬运；较宽矩阵使用下方带 repeats 的版本。
__simd_vf__ inline void CastFp32ToCompactZnBf16VF(__ubuf__ float* srcAddr, __ubuf__ bfloat16_t* dstZnAddr,
                                                  uint16_t rowCount, uint16_t colCount)
{
    using namespace Reg;
    static constexpr CastTrait castFp32ToBf16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                 RoundMode::CAST_RINT};
    RegTensor<float> srcReg;
    RegTensor<bfloat16_t> bf16Reg;
    RegTensor<uint16_t> packedBf16Reg;
    uint32_t validCount = colCount;
    uint32_t packedCount = colCount;
    MaskReg validMask = UpdateMask<float>(validCount);
    MaskReg packedMask = UpdateMask<uint16_t>(packedCount);

    for (uint16_t row = 0; row < rowCount; ++row) {
        LoadAlign(srcReg, srcAddr + static_cast<uint32_t>(row) * colCount);
        Cast<bfloat16_t, float, castFp32ToBf16>(bf16Reg, srcReg, validMask);
        Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg, reinterpret_cast<RegTensor<uint32_t>&>(bf16Reg));
        StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
            reinterpret_cast<__ubuf__ uint16_t*>(dstZnAddr) + static_cast<uint32_t>(row) * 16, packedBf16Reg, rowCount,
            packedMask);
    }
}

// V：FP32 src:[rowCount,srcRowStride] -> BF16 NZ dst:[ceil(colCount/16),dstFullRows,16]。
// 只转换有效 [rowCount,colCount]，CAST_RINT 舍入；补零与流水同步由调用方完成。
__simd_vf__ inline void RowMajorFp32ToCompactZnBf16VF(__ubuf__ float* srcAddr, __ubuf__ bfloat16_t* dstZnAddr,
                                                      uint16_t rowCount, uint16_t colCount, uint16_t srcRowStride,
                                                      uint16_t dstFullRows)
{
    using namespace Reg;
    constexpr uint32_t elementsPerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint32_t elementsPerBlock = 32 / sizeof(bfloat16_t);
    static constexpr CastTrait castFp32ToBf16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                 RoundMode::CAST_RINT};
    RegTensor<float> srcReg;
    RegTensor<bfloat16_t> bf16Reg;
    RegTensor<uint16_t> packedBf16Reg;

    const uint16_t repeats = static_cast<uint16_t>((colCount + elementsPerRepeat - 1) / elementsPerRepeat);
    for (uint16_t row = 0; row < rowCount; ++row) {
        for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
            const uint32_t featureBegin = static_cast<uint32_t>(repeat) * elementsPerRepeat;
            const uint32_t remaining = colCount - featureBegin;
            uint32_t validCount = remaining < elementsPerRepeat ? remaining : elementsPerRepeat;
            uint32_t packedCount = validCount;
            MaskReg validMask = UpdateMask<float>(validCount);
            MaskReg packedMask = UpdateMask<uint16_t>(packedCount);
            const uint32_t srcOffset = static_cast<uint32_t>(row) * srcRowStride + featureBegin;
            const uint32_t dstOffset = static_cast<uint32_t>(repeat) * (elementsPerRepeat / elementsPerBlock) *
                                           dstFullRows * elementsPerBlock +
                                       static_cast<uint32_t>(row) * elementsPerBlock;
            LoadAlign(srcReg, srcAddr + srcOffset);
            Cast<bfloat16_t, float, castFp32ToBf16>(bf16Reg, srcReg, validMask);
            Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg,
                                                          reinterpret_cast<RegTensor<uint32_t>&>(bf16Reg));
            StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
                reinterpret_cast<__ubuf__ uint16_t*>(dstZnAddr) + dstOffset, packedBf16Reg, dstFullRows, packedMask);
        }
    }
}

// V：BF16 行主序 [rowCount,srcRowStride] -> NZ [ceil(colCount/16),dstFullRows,16]；只重排，不改精度。
__simd_vf__ inline void RowMajorToCompactZnBf16VF(__ubuf__ bfloat16_t* srcAddr, __ubuf__ bfloat16_t* dstZnAddr,
                                                  uint16_t rowCount, uint16_t colCount, uint16_t srcRowStride,
                                                  uint16_t dstFullRows)
{
    using namespace Reg;
    constexpr uint32_t elementsPerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint32_t elementsPerBlock = 32 / sizeof(bfloat16_t);
    RegTensor<bfloat16_t> srcBf16Reg;
    RegTensor<uint16_t> packedBf16Reg;

    const uint16_t repeats = static_cast<uint16_t>((colCount + elementsPerRepeat - 1) / elementsPerRepeat);
    for (uint16_t row = 0; row < rowCount; ++row) {
        for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
            const uint32_t srcOffset =
                static_cast<uint32_t>(row) * srcRowStride + static_cast<uint32_t>(repeat) * elementsPerRepeat;
            const uint32_t dstOffset = static_cast<uint32_t>(repeat) * (elementsPerRepeat / elementsPerBlock) *
                                           dstFullRows * elementsPerBlock +
                                       static_cast<uint32_t>(row) * elementsPerBlock;
            const uint32_t remaining = colCount - static_cast<uint32_t>(repeat) * elementsPerRepeat;
            uint32_t packedCount = remaining < elementsPerRepeat ? remaining : elementsPerRepeat;
            MaskReg packedMask = UpdateMask<uint16_t>(packedCount);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(srcBf16Reg, srcAddr + srcOffset);
            Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg,
                                                          reinterpret_cast<RegTensor<uint32_t>&>(srcBf16Reg));
            StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
                reinterpret_cast<__ubuf__ uint16_t*>(dstZnAddr) + dstOffset, packedBf16Reg, dstFullRows, packedMask);
        }
    }
}

} // namespace ChunkGatedDeltaRule
