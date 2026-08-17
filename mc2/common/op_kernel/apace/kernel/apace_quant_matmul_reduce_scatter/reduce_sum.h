/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_sum.h
 * \brief 独立的 reduceSum 实现 — float32 累加保证精度
 *
 * 修复历史:
 * - MF-1: 使用 TOTAL_UB_SIZE 替代硬编码 192KB (DAV_3510 实际 248KB)
 * - MF-2: BF16→float32 Cast 累加，最终 Cast 回 BF16 输出 (参考 vec_comp.h DequantReduceSum)
 *   关键: float Add/Cast/Duplicate 的 repeatTime 为 uint8_t (max 255), 每次处理 256/sizeof(float)=64 元素
 *   单次调用上限 255*64=16320 元素, 超过需分批调用
 * - RN-2: 修正参数名与实际语义一致
 * - RN-2-OPT: 添加 RunTile 方法支持多 tile 分块处理（Round 2 性能优化）
 *
 * reduceSum 流程 (float32 累加, 参考 vec_comp.h DequantReduceSum):
 *   1. Duplicate 初始化 accumF32 为 0
 *   2. 逐 rank (含 self) 读数据 (BF16) → Cast 到 float32 → Add 到 accumF32
 *   3. Cast accumF32 (float32) → BF16 → 写入 output GM
 */

#pragma once

#include "kernel_operator.h"

namespace Apace {

using namespace AscendC;

template<typename CType>
class ReduceSumImpl {
public:
    __aicore__ inline ReduceSumImpl() {}
    __aicore__ inline ~ReduceSumImpl() {}

    /*!
     * \brief 执行 reduceSum (float32 累加) — 单 tile（整个 chunkM 一次性处理）
     * \param selfDataGM     本 rank 的 self 数据基址 (AIC 写入 workspace 的 C_i)
     * \param receivedDataGM received 数据基址 (PUT 推送到 win 区的其他 rank 数据)
     * \param outputGM       输出 GM 基址 (最终求和结果)
     * \param rankId         本 rank ID
     * \param rankSize       总 rank 数
     * \param chunkM         每个 rank 的 M 分片大小 (m / rankSize)
     * \param nSize          N 维度大小
     */
    __aicore__ inline void Run(
        GM_ADDR selfDataGM,
        GM_ADDR receivedDataGM,
        GM_ADDR outputGM,
        uint32_t rankId,
        uint32_t rankSize,
        uint32_t chunkM,
        uint32_t nSize)
    {
        RunImpl(selfDataGM, receivedDataGM, outputGM,
                rankId, rankSize, chunkM, nSize, chunkM, 0);
    }

    /*!
     * \brief 执行 reduceSum (float32 累加) — 多 tile 分块处理
     * \param selfDataGM      本 rank 的 self 数据基址 (AIC 写入 workspace 的 C_i)
     * \param receivedDataGM  received 数据基址 (PUT 推送到 win 区的其他 rank 数据)
     * \param outputGM        输出 GM 基址 (最终求和结果)
     * \param rankId          本 rank ID
     * \param rankSize        总 rank 数
     * \param totalChunkM     每个 rank 的完整 M 分片大小 (m / rankSize，地址偏移用)
     * \param nSize           N 维度大小
     * \param tileM           当前 tile 的 M 行数
     * \param tileMOffset     当前 tile 在 chunk 内的 M 偏移（行数）
     */
    __aicore__ inline void RunTile(
        GM_ADDR selfDataGM,
        GM_ADDR receivedDataGM,
        GM_ADDR outputGM,
        uint32_t rankId,
        uint32_t rankSize,
        uint32_t totalChunkM,
        uint32_t nSize,
        uint32_t tileM,
        uint32_t tileMOffset)
    {
        RunImpl(selfDataGM, receivedDataGM, outputGM,
                rankId, rankSize, totalChunkM, nSize, tileM, tileMOffset);
    }

private:
    /*!
     * \brief reduceSum 核心实现（支持单 tile 和多 tile）
     * \param totalChunkM  地址偏移用的完整 chunkM
     * \param tileM        实际处理的 M 行数
     * \param tileMOffset  tile 在 chunk 内的 M 偏移
     */
    __aicore__ inline void RunImpl(
        GM_ADDR selfDataGM,
        GM_ADDR receivedDataGM,
        GM_ADDR outputGM,
        uint32_t rankId,
        uint32_t rankSize,
        uint32_t totalChunkM,
        uint32_t nSize,
        uint32_t tileM,
        uint32_t tileMOffset)
    {
        using FloatType = float;
        constexpr uint32_t FLOAT_BYTES = sizeof(FloatType);  // 4
        uint32_t elemBytes = sizeof(CType);                  // 2 for bfloat16_t
        uint32_t rowBytes = nSize * elemBytes;               // bytes per row (BF16)
        // 地址偏移用 totalChunkM（跨 rank 步进），处理大小用 tileM
        uint64_t chunkBytes = static_cast<uint64_t>(totalChunkM) * nSize * elemBytes;
        uint32_t chunkM = tileM;  // 处理逻辑用 tileM 作为 chunkM

        // MF-1: 使用框架提供的 TOTAL_UB_SIZE 常量，不硬编码
        constexpr uint32_t COMM_RESERVED = 1024;
        uint32_t availUb = TOTAL_UB_SIZE - COMM_RESERVED;

        // 3 个缓冲区: accumF32 (float) + tempCType (CType) + tempF32 (float)
        uint32_t perRowBytes = nSize * (FLOAT_BYTES + elemBytes + FLOAT_BYTES);
        uint32_t subM = availUb / perRowBytes;

        // 对齐: 确保 DataCopy (32B) 和 Cast (32B) 对齐
        uint32_t castAlignElems = 32 / elemBytes;  // 16 for BF16
        uint32_t alignRows = 1;
        while (alignRows < castAlignElems && (alignRows * nSize) % castAlignElems != 0) {
            alignRows++;
        }
        subM = (subM / alignRows) * alignRows;
        if (subM == 0) { subM = alignRows; }
        if (subM > chunkM) { subM = chunkM; }

        // MF-2 关键: float Vector API repeatTime 为 uint8_t (max 255)
        // 每次处理 256/FLOAT_BYTES = 64 个 float 元素, 单次上限 255*64 = 16320
        // 超过会导致 repeatTime 溢出, 部分元素不被处理 → 数据错误
        constexpr uint32_t FLOAT_ELEMS_PER_REPEAT = 256 / FLOAT_BYTES;  // 64
        constexpr uint32_t MAX_FLOAT_ELEMS_PER_CALL = 255 * FLOAT_ELEMS_PER_REPEAT;  // 16320
        uint32_t maxSubMForFloat = MAX_FLOAT_ELEMS_PER_CALL / nSize;
        if (subM > maxSubMForFloat) { subM = maxSubMForFloat; }
        // 重新对齐
        subM = (subM / alignRows) * alignRows;
        if (subM == 0) { subM = alignRows; }
        if (subM > chunkM) { subM = chunkM; }

        // UB 分配: 使用 LocalTensor 覆盖可用区域
        LocalTensor<CType> ubCType{TPosition::VECIN, 0, TOTAL_UB_SIZE / elemBytes};
        LocalTensor<FloatType> ubFloat{TPosition::VECIN, 0, TOTAL_UB_SIZE / FLOAT_BYTES};

        // 缓冲区字节偏移
        uint32_t accumOffsetBytes = COMM_RESERVED;
        uint32_t tempCTypeOffsetBytes = accumOffsetBytes + subM * nSize * FLOAT_BYTES;
        uint32_t tempF32OffsetBytes = tempCTypeOffsetBytes + subM * nSize * elemBytes;

        // 元素偏移 (各自类型)
        uint32_t accumFloatOff = accumOffsetBytes / FLOAT_BYTES;
        uint32_t tempCTypeOff = tempCTypeOffsetBytes / elemBytes;
        uint32_t tempF32FloatOff = tempF32OffsetBytes / FLOAT_BYTES;

        // 分批处理的 chunk 大小 (确保 float Add/Cast/Duplicate 不超 255 repeats)
        uint32_t floatChunkSize = MAX_FLOAT_ELEMS_PER_CALL;  // 16320
        // 确保按 nSize 对齐 (整行处理)
        if (floatChunkSize > nSize) {
            floatChunkSize = (floatChunkSize / nSize) * nSize;
        }
        if (floatChunkSize == 0) { floatChunkSize = nSize; }

        uint32_t processedRows = 0;

        // 逐 sub-tile 处理
        for (uint32_t mOffset = 0; mOffset < chunkM; mOffset += subM) {
            uint32_t curSubM = (mOffset + subM <= chunkM) ? subM : (chunkM - mOffset);
            uint32_t curCount = curSubM * nSize;

            auto accumF32 = ubFloat[accumFloatOff];
            auto tempCType = ubCType[tempCTypeOff];
            auto tempF32 = ubFloat[tempF32FloatOff];

            // 1. 初始化 accumF32 为 0 (分批 Duplicate, 避免 repeatTime 溢出)
            for (uint32_t off = 0; off < curCount; off += floatChunkSize) {
                uint32_t chunkCount = (off + floatChunkSize <= curCount) ? floatChunkSize : (curCount - off);
                Duplicate<FloatType>(accumF32[off], (FloatType)0.0, chunkCount);
            }
            PipeBarrier<PIPE_V>();

            // 2. 逐 rank (含 self) 读数据 → Cast → Add 到 accumF32
            for (uint32_t r = 0; r < rankSize; ++r) {
                GM_ADDR dataGM = (r == rankId) ? selfDataGM : receivedDataGM;
                // 地址偏移 = rank 步进 + tile 偏移 + sub-tile 内偏移
                uint64_t readOffset = static_cast<uint64_t>(r) * chunkBytes
                                    + static_cast<uint64_t>(tileMOffset + mOffset) * rowBytes;

                GlobalTensor<CType> dataGm;
                dataGm.SetGlobalBuffer(reinterpret_cast<__gm__ CType*>(dataGM + readOffset));

                // V→MTE2 sync: 确保 V (Duplicate/Add) 完成后再 MTE2 DataCopy
                // 对于 r=0: V 包含 WaitFlag<MTE3_V>, V_MTE2 间接保证 MTE3→MTE2 (WAR on tempCType)
                // 对于 r>0: V 包含 Cast/Add, V_MTE2 保证 V 读 tempCType → MTE2 写 tempCType (WAR)
                SetFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(0);
                // DataCopy (BF16, 不受 float repeat 限制)
                DataCopy(tempCType, dataGm, curCount);
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);

                // Cast BF16 → float32 + Add (分批处理, 避免 repeatTime 溢出)
                for (uint32_t off = 0; off < curCount; off += floatChunkSize) {
                    uint32_t chunkCount = (off + floatChunkSize <= curCount) ? floatChunkSize : (curCount - off);
                    // Cast BF16 → float32 (lossless)
                    Cast(tempF32[off], tempCType[off], RoundMode::CAST_NONE, chunkCount);
                    PipeBarrier<PIPE_V>();
                    // Add in float32 (高精度累加)
                    Add(accumF32[off], accumF32[off], tempF32[off], static_cast<int32_t>(chunkCount));
                    PipeBarrier<PIPE_V>();
                }
            }

            // 3. Cast accumF32 (float32) → tempCType (BF16) for output (分批)
            for (uint32_t off = 0; off < curCount; off += floatChunkSize) {
                uint32_t chunkCount = (off + floatChunkSize <= curCount) ? floatChunkSize : (curCount - off);
                Cast(tempCType[off], accumF32[off], RoundMode::CAST_RINT, chunkCount);
            }
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);

            // 4. 写 tempCType (BF16) → output GM (偏移包含 tileMOffset)
            uint64_t writeOffset = static_cast<uint64_t>(tileMOffset + processedRows) * rowBytes;
            GlobalTensor<CType> outGm;
            outGm.SetGlobalBuffer(reinterpret_cast<__gm__ CType*>(outputGM + writeOffset));
            DataCopy(outGm, tempCType, curCount);
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);

            processedRows += curSubM;
        }
    }
};

} // namespace Apace
