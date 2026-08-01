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
 * \file und_gen_qkv_rms_norm_rope_cache_regbase.h
 * \brief UndGenQkvRmsNormRopeCache regbase(DAV_3510 / arch35) 模板骨架
 *
 * 计算流程（每个 out_t 独立）：
 *   1) src_t = hasCatIndices ? cat_indices[out_t] : out_t；isUnd = (src_t < undLen)
 *   2) 从 und_qkv / gen_qkv 取 [N, D] 行 -> Cast bf16->float32 -> 按 N 维拆 Q/K/V
 *   3) Q/K 做 RMSNorm（und/gen 两套权重按 isUnd 选择）
 *   4) 读 positions[0..2, out_t]，gather 三轴 cos_sin 并按 mrope_section 合并成一份，
 *      再对 Q/K 做标准 RoPE；V 不参与 norm/rope
 *   5) Q Cast float32->bf16 写 q[out_t]；K/V Cast float32->bf16 后按
 *      slot = slot_mapping[out_t] 写入 k_cache/v_cache 的 (slot / Bs, slot % Bs) 行
 *
 * 注意：本文件当前仅为骨架，计算实现待补充（见各 TODO）。
 */

#ifndef UND_GEN_QKV_RMS_NORM_ROPE_CACHE_REGBASE_H_
#define UND_GEN_QKV_RMS_NORM_ROPE_CACHE_REGBASE_H_

#include "kernel_operator.h"
#ifdef __CCE_UT_TEST__
#include "../../../kv_rms_norm_rope_cache/op_kernel/arch35/platform.h"
#include "../../../../third_party/opbase/include/op_common/op_kernel/platform_util.h"
#else
#include "op_kernel/platform_util.h"
#endif

namespace UndGenQkvRmsNormRopeCache {
using namespace AscendC;

constexpr static int64_t MROPE_AXIS_NUM = 3;
constexpr static int64_t DIGIT_TWO = 2;

/**
 * @brief UndGenQkvRmsNormRopeCache regbase 模板
 * @tparam T_QKV   und_qkv/gen_qkv 与 q 输出的数据类型（bfloat16_t）
 * @tparam T_CACHE k_cache/v_cache 的数据类型（bfloat16_t）
 */
template <typename T_QKV, typename T_CACHE>
class UndGenQkvRmsNormRopeCacheRegbase {
public:
    __aicore__ inline UndGenQkvRmsNormRopeCacheRegbase(TPipe* pipe, const UndGenQkvRmsNormRopeCacheTilingData* tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(GM_ADDR undQkv, GM_ADDR undWeightsQ, GM_ADDR undWeightsK, GM_ADDR cosSinCache,
                                GM_ADDR kCache, GM_ADDR vCache, GM_ADDR slotMapping, GM_ADDR positions, GM_ADDR genQkv,
                                GM_ADDR genWeightsQ, GM_ADDR genWeightsK, GM_ADDR catIndices, GM_ADDR q)
    {
        // shape / 切分参数
        totalTokens_ = tiling_->totalTokens;
        undLen_ = tiling_->undLen;
        numHead_ = tiling_->numHead;
        numHeadQ_ = tiling_->numHeadQ;
        numHeadK_ = tiling_->numHeadK;
        numHeadV_ = tiling_->numHeadV;
        headDim_ = tiling_->headDim;
        blockSize_ = tiling_->blockSize;
        hasGen_ = tiling_->hasGen != 0;
        hasCatIndices_ = tiling_->hasCatIndices != 0;
        epsilon_ = tiling_->epsilon;
        reciprocal_ = tiling_->reciprocal;

        // GlobalTensor 绑定
        undQkvGm_.SetGlobalBuffer((__gm__ T_QKV*)undQkv);
        undWeightsQGm_.SetGlobalBuffer((__gm__ T_QKV*)undWeightsQ);
        undWeightsKGm_.SetGlobalBuffer((__gm__ T_QKV*)undWeightsK);
        cosSinCacheGm_.SetGlobalBuffer((__gm__ float*)cosSinCache);
        kCacheGm_.SetGlobalBuffer((__gm__ T_CACHE*)kCache);
        vCacheGm_.SetGlobalBuffer((__gm__ T_CACHE*)vCache);
        slotMappingGm_.SetGlobalBuffer((__gm__ int64_t*)slotMapping);
        positionsGm_.SetGlobalBuffer((__gm__ int64_t*)positions);
        if (hasGen_) {
            genQkvGm_.SetGlobalBuffer((__gm__ T_QKV*)genQkv);
            genWeightsQGm_.SetGlobalBuffer((__gm__ T_QKV*)genWeightsQ);
            genWeightsKGm_.SetGlobalBuffer((__gm__ T_QKV*)genWeightsK);
        }
        if (hasCatIndices_) {
            catIndicesGm_.SetGlobalBuffer((__gm__ int64_t*)catIndices);
        }
        qGm_.SetGlobalBuffer((__gm__ T_QKV*)q);

        // TODO(kernel): 计算本核负责的 token 区间 [coreStart_, coreEnd_)，
        //               申请各 UB buffer（权重常驻、cos_sin 三轴缓冲、Q/K/V float 缓冲等），
        //               预加载 4 个 RMSNorm 权重并 Cast 为 float32。
    }

    __aicore__ inline void Process()
    {
        // TODO(kernel): 主循环 for (out_t = coreStart_; out_t < coreEnd_; ++out_t)：
        //               CopyIn -> RmsNorm(Q/K) -> MergeCosSin -> Rope(Q/K) -> CopyOutQ -> WriteKvCache
    }

private:
    // TODO(kernel): 以下子流程待实现
    __aicore__ inline void CopyIn(int64_t outT) {}
    __aicore__ inline void RmsNorm(int64_t outT) {}
    __aicore__ inline void MergeCosSin(int64_t outT) {}
    __aicore__ inline void Rope(int64_t outT) {}
    __aicore__ inline void CopyOutQ(int64_t outT) {}
    __aicore__ inline void WriteKvCache(int64_t outT) {}

private:
    TPipe* pipe_ = nullptr;
    const UndGenQkvRmsNormRopeCacheTilingData* tiling_ = nullptr;

    GlobalTensor<T_QKV> undQkvGm_;
    GlobalTensor<T_QKV> genQkvGm_;
    GlobalTensor<T_QKV> undWeightsQGm_;
    GlobalTensor<T_QKV> undWeightsKGm_;
    GlobalTensor<T_QKV> genWeightsQGm_;
    GlobalTensor<T_QKV> genWeightsKGm_;
    GlobalTensor<float> cosSinCacheGm_;
    GlobalTensor<T_CACHE> kCacheGm_;
    GlobalTensor<T_CACHE> vCacheGm_;
    GlobalTensor<int64_t> slotMappingGm_;
    GlobalTensor<int64_t> positionsGm_;
    GlobalTensor<int64_t> catIndicesGm_;
    GlobalTensor<T_QKV> qGm_;

    int64_t totalTokens_ = 0;
    int64_t undLen_ = 0;
    int64_t numHead_ = 0;
    int64_t numHeadQ_ = 0;
    int64_t numHeadK_ = 0;
    int64_t numHeadV_ = 0;
    int64_t headDim_ = 0;
    int64_t blockSize_ = 0;
    int64_t coreStart_ = 0;
    int64_t coreEnd_ = 0;
    bool hasGen_ = false;
    bool hasCatIndices_ = false;
    float epsilon_ = 0.0f;
    float reciprocal_ = 0.0f;
};
} // namespace UndGenQkvRmsNormRopeCache

#endif // UND_GEN_QKV_RMS_NORM_ROPE_CACHE_REGBASE_H_
