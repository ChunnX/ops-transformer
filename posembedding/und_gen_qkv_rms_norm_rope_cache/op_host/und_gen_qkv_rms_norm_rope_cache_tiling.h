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
 * \file und_gen_qkv_rms_norm_rope_cache_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_UND_GEN_QKV_RMS_NORM_ROPE_CACHE_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_UND_GEN_QKV_RMS_NORM_ROPE_CACHE_H_

#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(UndGenQkvRmsNormRopeCacheTilingData)
// ---- shape 信息 ----
TILING_DATA_FIELD_DEF(int64_t, totalTokens);   // T = und_len + gen_len，与 slot_mapping/positions 一致
TILING_DATA_FIELD_DEF(int64_t, undLen);        // und_qkv 的 token 数
TILING_DATA_FIELD_DEF(int64_t, genLen);        // gen_qkv 的 token 数，无 gen 时为 0
TILING_DATA_FIELD_DEF(int64_t, numHead);       // N = Hq + Hk + Hv
TILING_DATA_FIELD_DEF(int64_t, numHeadQ);      // Hq
TILING_DATA_FIELD_DEF(int64_t, numHeadK);      // Hk
TILING_DATA_FIELD_DEF(int64_t, numHeadV);      // Hv
TILING_DATA_FIELD_DEF(int64_t, headDim);       // D
TILING_DATA_FIELD_DEF(int64_t, maxPos);        // cos_sin_cache 的行数
TILING_DATA_FIELD_DEF(int64_t, blockNum);      // KV Cache 页数 Bn
TILING_DATA_FIELD_DEF(int64_t, blockSize);     // KV Cache 页内行数 Bs
// ---- 可选输入分支标志 ----
TILING_DATA_FIELD_DEF(int64_t, hasGen);        // gen_qkv/gen_weights 是否存在
TILING_DATA_FIELD_DEF(int64_t, hasCatIndices); // cat_indices 是否存在，否则 src_t = out_t
// ---- MRoPE 三轴分段（axisLut 由 kernel 侧按此规则展开）----
TILING_DATA_FIELD_DEF(int64_t, mropeSectionT);
TILING_DATA_FIELD_DEF(int64_t, mropeSectionH);
TILING_DATA_FIELD_DEF(int64_t, mropeSectionW);
// ---- 标量参数 ----
TILING_DATA_FIELD_DEF(float, epsilon);         // RMSNorm eps
TILING_DATA_FIELD_DEF(float, reciprocal);      // 1 / D
// ---- 切分结果 ----
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, formerCoreNum);  // 前 formerCoreNum 个核多处理 1 个 token
TILING_DATA_FIELD_DEF(int64_t, blockFactor);    // 前 formerCoreNum 个核的 token 数
TILING_DATA_FIELD_DEF(int64_t, tailBlockFactor);// 其余核的 token 数
TILING_DATA_FIELD_DEF(int64_t, ubFactor);       // 单次 UB 内处理的 token 数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UndGenQkvRmsNormRopeCache, UndGenQkvRmsNormRopeCacheTilingData)

struct UndGenQkvRmsNormRopeCacheCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

// 输入索引
constexpr int64_t UND_QKV_INDEX = 0;
constexpr int64_t UND_WEIGHTS_Q_INDEX = 1;
constexpr int64_t UND_WEIGHTS_K_INDEX = 2;
constexpr int64_t COS_SIN_CACHE_INDEX = 3;
constexpr int64_t K_CACHE_INDEX = 4;
constexpr int64_t V_CACHE_INDEX = 5;
constexpr int64_t SLOT_MAPPING_INDEX = 6;
constexpr int64_t POSITIONS_INDEX = 7;
constexpr int64_t GEN_QKV_INDEX = 8;
constexpr int64_t GEN_WEIGHTS_Q_INDEX = 9;
constexpr int64_t GEN_WEIGHTS_K_INDEX = 10;
constexpr int64_t CAT_INDICES_INDEX = 11;

// 输出索引
constexpr int64_t Q_OUT_INDEX = 0;
constexpr int64_t K_CACHE_OUT_INDEX = 1;
constexpr int64_t V_CACHE_OUT_INDEX = 2;

// 属性索引
constexpr int64_t NUM_HEADS_Q_ATTR_IDX = 0;
constexpr int64_t NUM_HEADS_K_ATTR_IDX = 1;
constexpr int64_t NUM_HEADS_V_ATTR_IDX = 2;
constexpr int64_t NORM_EPS_ATTR_IDX = 3;
constexpr int64_t MROPE_SECTION_ATTR_IDX = 4;

constexpr int64_t DIM_ZERO = 0;
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DIM_THREE = 3;
constexpr int64_t DIM_NUM_ONE = 1;
constexpr int64_t DIM_NUM_TWO = 2;
constexpr int64_t DIM_NUM_THREE = 3;
constexpr int64_t DIM_NUM_FOUR = 4;

constexpr int64_t SUPPORTED_HEAD_DIM = 128;
constexpr int64_t SUPPORTED_BLOCK_SIZE = 128;
constexpr int64_t MAX_TOTAL_TOKENS = 64 * 1024;
// 本期支持的 (Hq, Hk, Hv) 组合，需要放宽时改这里即可
constexpr int64_t SUPPORTED_HEAD_COMBO_NUM = 2;
constexpr int64_t SUPPORTED_HEAD_COMBOS[SUPPORTED_HEAD_COMBO_NUM][3] = {{8, 1, 1}, {16, 2, 2}};
constexpr int64_t MROPE_AXIS_NUM = 3;
constexpr int64_t BF16_BYTES = 2;
constexpr int64_t FLOAT32_BYTES = 4;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t BLOCK_ALIGN_BYTES = 32;
constexpr int64_t WEIGHT_NUM = 4;           // und_q / und_k / gen_q / gen_k
constexpr int64_t WEIGHT_VARIANT_NUM = 2;   // 按头展开的权重要 und / gen 各存一份
constexpr int64_t REDUCE_TMP_BYTES = 4096;  // ReduceSum 的 sharedTmpBuffer 下限

/* UB 划分（kernel 侧必须按同一套公式分配，改这里要同步改 kernel）
 *
 * 记 N = Hq+Hk+Hv、H2 = Hq+Hk（V 不参与 RMSNorm/RoPE）、D = headDim、u = ubFactor。
 * CopyIn 按 tile 批量搬 u 个 token，计算与写出逐 token，因此计算区不随 u 变化。
 *
 * 计算区（TBuf，与 u 无关）:
 *   wBf16Buf         WEIGHT_NUM * D * 2            4 个权重原始 bf16
 *   wExpandBuf       2 * H2 * D * 4                und/gen 各一份按头展开的 float 权重，Init 建好
 *   cosSinExpandBuf  H2 * D * 4                    合并后的 cos_sin 按头展开，每头 [cos(0:D/2)|sin(0:D/2)]
 *   ropeTmpBuf       H2 * (D/2) * 4                RoPE 半宽中间量
 *   qkvFp32Buf       N * D * 4                     单 token 的 Q/K/V float32
 *   reduceTmpBuf     REDUCE_TMP_BYTES              ReduceSum sharedTmpBuffer
 *   rmsInvBuf        align32(H2 * 4)               每头 rsqrt(mean+eps)
 *
 * 队列（随 u，均开 DoubleBuffer）:
 *   qkvInQue    VECIN   2 * u * N * D * 2
 *   cosSinInQue VECIN   2 * u * MROPE_AXIS_NUM * D * 4
 *   outQue      VECOUT  2 * u * N * D * 2          UB 内按 [u*q | u*k | u*v] 分段，便于 q 一笔连续写出
 */

class UndGenQkvRmsNormRopeCacheTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit UndGenQkvRmsNormRopeCacheTilingBase(gert::TilingContext* tilingContext) : TilingBaseClass(tilingContext)
    {}
    ~UndGenQkvRmsNormRopeCacheTilingBase() override
    {}
    uint64_t tilingKey_{0};
    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t totalTokens_ = 0;
    int64_t undLen_ = 0;
    int64_t genLen_ = 0;
    int64_t numHead_ = 0;
    int64_t numHeadQ_ = 0;
    int64_t numHeadK_ = 0;
    int64_t numHeadV_ = 0;
    int64_t headDim_ = 0;
    int64_t maxPos_ = 0;
    int64_t blockNum_ = 0;
    int64_t blockSize_ = 0;
    bool hasGen_ = false;
    bool hasCatIndices_ = false;
    int64_t mropeSection_[MROPE_AXIS_NUM] = {0, 0, 0};
    float epsilon_ = 0.0f;
    float reciprocal_ = 0.0f;

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override
    {
        return false;
    }
    ge::graphStatus DoOpTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t GetTilingKey() const override;

    ge::graphStatus CheckDtypeValid();
    ge::graphStatus CheckOutputShapeValid();
    ge::graphStatus CheckSupportRange();
    ge::graphStatus CheckUndGenQkvValid();
    ge::graphStatus CheckWeightsValid();
    ge::graphStatus CheckCosSinCacheValid();
    ge::graphStatus CheckKvCacheValid();
    ge::graphStatus CheckSlotMappingValid();
    ge::graphStatus CheckPositionsValid();
    ge::graphStatus CheckCatIndicesValid();
    ge::graphStatus CheckAttrsValid();
};

class UndGenQkvRmsNormRopeCacheRegbaseTiling : virtual public UndGenQkvRmsNormRopeCacheTilingBase {
public:
    explicit UndGenQkvRmsNormRopeCacheRegbaseTiling(gert::TilingContext* tilingContext)
        : UndGenQkvRmsNormRopeCacheTilingBase(tilingContext)
    {}
    ~UndGenQkvRmsNormRopeCacheRegbaseTiling()
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

protected:
    ge::graphStatus CalBlockTiling();
    ge::graphStatus CalUbTiling();
    void PrintTilingData();

private:
    UndGenQkvRmsNormRopeCacheTilingData tilingData_;
};
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_UND_GEN_QKV_RMS_NORM_ROPE_CACHE_H_
