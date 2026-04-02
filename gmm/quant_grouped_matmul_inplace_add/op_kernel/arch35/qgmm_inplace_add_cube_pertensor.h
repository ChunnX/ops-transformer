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
 * \file qgmm_inplace_add_cube_pertensor.h
 * \brief pertensor-pertensor 量化模式的 QuantGroupedMatmulInplaceAdd 纯Cube模板（AIC_ONLY）
 *
 * 量化规格：
 *   - x1 量化参数 scale1: float32, shape (g,) 或 (g,1)，每个分组一个标量
 *   - x2 量化参数 scale2: float32, shape (g,) 或 (g,1)，每个分组一个标量
 *
 * 计算逻辑（参考 gqmm_cube_on_the_fly.h 的 pertensor-pertensor 分支）：
 *   1. 对第 g 组，读取 scale2[g] 和 scale1[g]（均为 float32 标量）
 *   2. 两者相乘得到 deqScale（float32），取高19位（& 0xFFFFE000），cast 成 uint64
 *   3. 通过 mm.SetQuantScalar(scaleScalar) 传入 cube fixpipe 随路反量化
 *   4. 矩阵乘结果通过 SetAtomicAdd 加到 y 上（inplace add 语义）
 *
 * 多核调度策略：
 *   - 参照 gqmm_cube_on_the_fly.h 的 ASWT (Adaptive Sliding Window Tiling) 思路
 *   - 每个 core 处理自己负责的 M/N tile，通过 blockIdx + usedCoreNum 按行切分
 *   - 在同一 group 内，blockIdx < usedCoreNum 的 core 参与该 group 的计算
 */

#ifndef QGMM_INPLACE_ADD_CUBE_PERTENSOR_H
#define QGMM_INPLACE_ADD_CUBE_PERTENSOR_H

#include "qgmm_inplace_add_utils.h"
#include "quant_grouped_matmul_inplace_add_tiling_data.h"

using namespace AscendC;
using namespace matmul;

// fixpipe 只能取 float32 dequant scale 的高19位，低13位清零
constexpr uint32_t QGMM_PERTENSOR_DEQ_SCALE_HIGH19_MASK = 0xFFFFE000U;

/*!
 * \brief pertensor-pertensor 量化模式下的 QuantGroupedMatmulInplaceAdd
 *
 * \tparam layoutA  x1 的布局类型 (Cgmct::Gemm::layout::ColumnMajor，对应 transA=true)
 * \tparam layoutB  x2 的布局类型 (Cgmct::Gemm::layout::RowMajor，对应 transB=false)
 */
template <typename layoutA, typename layoutB>
__aicore__ inline void QGmmInplaceAddPertensorAswt(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale2, GM_ADDR groupList,
                                                   GM_ADDR scale1, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_MEMBER(QuantGroupedMatmulInplaceAdd::QGmmInplaceAddTilingDataParams, quantGmmInplaceAddParams,
                           gmmBaseParams_, tiling);
    GET_TILING_DATA_MEMBER(QuantGroupedMatmulInplaceAdd::QGmmInplaceAddTilingDataParams, mmTilingData, mmTilingData_,
                           tiling);

    if ASCEND_IS_AIV {
        return;
    }

    // 数据类型由编译期宏 DTYPE_X1/DTYPE_X2/DTYPE_Y 提供
    using AType = DTYPE_X1;
    using BType = DTYPE_X2;
    using CType = DTYPE_Y;
    // transA/transB：此处调用约定固定为 transA=true（x1 列主序）、transB=false（x2 行主序）
    // 对应算子的 TN 转置模式，与调用处 QGmmInplaceAddPertensorAswt 的模板参数一致
    static constexpr bool transA = true;
    static constexpr bool transB = false;

    // 定义 matmul 类型（非 MX 模式，使用普通 MatmulType 不带 Scale）
    using AMatmulType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType, transA>;
    using BMatmulType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType, transB>;
    using CMatmulType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>;
    using BiasMatmulType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    // MatmulImpl 不带 MX Scale policy，走标准 cube fixpipe 路径
    using MmType = matmul::MatmulImpl<AMatmulType, BMatmulType, CMatmulType, BiasMatmulType>;

    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t usedCoreNum = static_cast<uint32_t>(mmTilingData_.usedCoreNum);
    uint32_t groupNum = gmmBaseParams_.groupNum;
    uint8_t groupListType = gmmBaseParams_.groupListType;

    // 初始化 matmul
    MmType mm;
    mm.SetSubBlockIdx(0);
    mm.Init(const_cast<TCubeTiling *>(&mmTilingData_), GetTPipePtr());

    // 设置全局 tensor
    GlobalTensor<AType> xGlobal;
    GlobalTensor<BType> wGlobal;
    GlobalTensor<CType> yGlobal;
    GlobalTensor<int64_t> groupListGlobal;

    if (groupList != nullptr) {
        groupListGlobal.SetGlobalBuffer((__gm__ int64_t *)groupList);
    }

    // 全局 M, N 大小（所有 group 共享相同的 M, N）
    int64_t mTotal = static_cast<int64_t>(mmTilingData_.M);
    int64_t nTotal = static_cast<int64_t>(mmTilingData_.N);
    int64_t baseM = static_cast<int64_t>(mmTilingData_.baseM);
    int64_t baseN = static_cast<int64_t>(mmTilingData_.baseN);

    // 计算 M/N 方向上的 tile 数量
    int64_t mCnt = (mTotal + baseM - 1) / baseM;
    int64_t nCnt = (nTotal + baseN - 1) / baseN;
    int64_t totalTile = mCnt * nCnt;

    // 各分组的步长（按元素数量）
    int64_t cGroupStride = mTotal * nTotal;

    // 累计 K 偏移（用于 x1, x2 的地址计算）
    int64_t aBaseOffset = 0; // x1: (k, m)，前 kAccum * mTotal 个元素
    int64_t bBaseOffset = 0; // x2: (k, n)，前 kAccum * nTotal 个元素
    int64_t cBaseOffset = 0; // y:  第 g 组，偏移 g * m * n
    int32_t preOffset = 0;

    // inplace add 语义：矩阵乘结果原子加到 y 上
    AscendC::SetAtomicAdd<float>();

    for (uint32_t groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
        // 从 groupList 获取本组的 K
        int32_t kSize = 0;
        if (groupList != nullptr) {
            if (groupListType == 0) {
                // cumsum 模式：groupList[g] = 前 g+1 组的 K 累计值
                int32_t curOffset = static_cast<int32_t>(groupListGlobal.GetValue(groupIdx));
                kSize = curOffset - preOffset;
                preOffset = curOffset;
            } else {
                // count 模式：groupList[g] = 第 g 组的 K 大小
                kSize = static_cast<int32_t>(groupListGlobal.GetValue(groupIdx));
            }
        } else {
            kSize = static_cast<int32_t>(mmTilingData_.Ka);
        }

        if (kSize <= 0) {
            // K=0 时 y += 0，直接跳过
            aBaseOffset += static_cast<int64_t>(kSize) * mTotal;
            bBaseOffset += static_cast<int64_t>(kSize) * nTotal;
            cBaseOffset += cGroupStride;
            continue;
        }

        // 读取本组 pertensor scale（float32 标量）
        // scale2 shape: (g,) 或 (g,1) → 取第 groupIdx 个 float
        float scaleBValue = *((__gm__ float *)scale2 + groupIdx);
        // scale1 shape: (g,) 或 (g,1) → 取第 groupIdx 个 float
        float scaleAValue = *((__gm__ float *)scale1 + groupIdx);

        // 两个 pertensor scale 相乘，取高19位（低13位清零），cast 成 uint64
        // 这是 fixpipe 硬件随路反量化的要求
        float deqScaleF32 = scaleBValue * scaleAValue;
        uint32_t uint32Scale = *(reinterpret_cast<uint32_t *>(&deqScaleF32));
        uint64_t scaleScalar = static_cast<uint64_t>(uint32Scale & QGMM_PERTENSOR_DEQ_SCALE_HIGH19_MASK);

        // 多核切分：每个 core 处理若干 M/N tile
        // tile_idx 按 [mIdx, nIdx] 二维索引展平为一维，core blockIdx 处理 tile_idx % usedCoreNum == blockIdx 的 tile
        for (int64_t tileIdx = static_cast<int64_t>(blockIdx); tileIdx < totalTile;
             tileIdx += static_cast<int64_t>(usedCoreNum)) {
            int64_t mIdx = tileIdx / nCnt;
            int64_t nIdx = tileIdx % nCnt;

            // 计算本 tile 的实际 M, N 大小（处理尾部不足 baseM/baseN 的情况）
            int64_t singleM = (mIdx == mCnt - 1) ? (mTotal - mIdx * baseM) : baseM;
            int64_t singleN = (nIdx == nCnt - 1) ? (nTotal - nIdx * baseN) : baseN;
            if (singleM <= 0 || singleN <= 0) {
                continue;
            }

            // 计算本 tile 在各 tensor 中的元素偏移
            // x1 layout: (k, m)，transA=true，按列主序存储
            //   tile 在 x1 中的偏移 = aBaseOffset + mIdx * baseM
            int64_t offsetA = aBaseOffset + mIdx * baseM;
            // x2 layout: (k, n)，transB=false，按行主序存储
            //   tile 在 x2 中的偏移 = bBaseOffset + nIdx * baseN
            int64_t offsetB = bBaseOffset + nIdx * baseN;
            // y layout: (g, m, n)，按行主序，tile 在 y 中的偏移
            //   = cBaseOffset + mIdx * baseM * nTotal + nIdx * baseN
            int64_t offsetC = cBaseOffset + mIdx * baseM * nTotal + nIdx * baseN;

            xGlobal.SetGlobalBuffer((__gm__ AType *)x1 + offsetA);
            wGlobal.SetGlobalBuffer((__gm__ BType *)x2 + offsetB);
            yGlobal.SetGlobalBuffer((__gm__ CType *)y + offsetC);
            yGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

            mm.SetSingleShape(static_cast<int32_t>(singleM), static_cast<int32_t>(singleN), kSize);
            mm.SetQuantScalar(scaleScalar);
            mm.SetTensorA(xGlobal[0], transA);
            mm.SetTensorB(wGlobal[0], transB);
            mm.Iterate();
            mm.GetTensorC(yGlobal[0]);
        }

        // 更新下一组的 tensor 偏移
        aBaseOffset += static_cast<int64_t>(kSize) * mTotal;
        bBaseOffset += static_cast<int64_t>(kSize) * nTotal;
        cBaseOffset += cGroupStride;
    }

    AscendC::SetAtomicNone();
}

#endif // QGMM_INPLACE_ADD_CUBE_PERTENSOR_H
