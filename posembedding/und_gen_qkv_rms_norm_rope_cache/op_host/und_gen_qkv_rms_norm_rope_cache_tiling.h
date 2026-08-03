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
// DAV_3510 的矢量寄存器是 256B，一个 float VL 就是 64 个元素。kernel 的 VF 段
// （op_kernel/arch35/..._mrope_vf.h）把「headDim/2 恰好等于一个满 VL」写死了，
// 没有分段循环也没有尾块 mask。下面的 static_assert 把这个不变量钉在编译期：
// 放宽 SUPPORTED_HEAD_DIM 会直接编译失败，提醒先去补 VF 的分段与尾块处理。
constexpr int64_t VL_FP32_LANES = 64;
static_assert(SUPPORTED_HEAD_DIM / 2 == VL_FP32_LANES,
              "VF kernel assumes headDim/2 == one float VL; add tail-mask handling in "
              "op_kernel/arch35/und_gen_qkv_rms_norm_rope_cache_mrope_vf.h before relaxing SUPPORTED_HEAD_DIM");
constexpr int64_t SUPPORTED_BLOCK_SIZE = 128;
// NOTE: T (= und_len + gen_len) 不设上限。多核切分是按 T 取余数分摊、UB 切分只切 token 数，
//       两者都与 T 的绝对大小无关；host/kernel 侧所有 GM 偏移与 tiling 字段一律 int64_t，
//       kernel 里的 uint32_t 只用于 UB 内的 gather 索引与 tile 内 stride（被 UB 容量兜住）。
//       T 的真实约束是 KV Cache 容量，由 CheckKvCacheValid 的 blockNum * blockSize >= T 负责。
// 本期支持的 (Hq, Hk, Hv) 组合，需要放宽时改这里即可
constexpr int64_t SUPPORTED_HEAD_COMBO_NUM = 2;
constexpr int64_t SUPPORTED_HEAD_COMBOS[SUPPORTED_HEAD_COMBO_NUM][3] = {{8, 1, 1}, {16, 2, 2}};
constexpr int64_t MROPE_AXIS_NUM = 3;
constexpr int64_t BF16_BYTES = 2;
constexpr int64_t FLOAT32_BYTES = 4;
constexpr int64_t UINT32_BYTES = 4;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t BLOCK_ALIGN_BYTES = 32;
constexpr int64_t WEIGHT_NUM = 4;           // und_q / und_k / gen_q / gen_k
constexpr int64_t GAMMA_TILE_NUM = 2;       // 按 token 铺平的 gamma tile：Q 一份、K 一份

/* UB 划分（kernel 侧必须按同一套公式分配，改这里要同步改 kernel）
 *
 * 记 N = Hq+Hk+Hv、H2 = Hq+Hk（V 不参与 RMSNorm/RoPE）、D = headDim、u = ubFactor。
 * CopyIn 按 tile 批量搬 u 个 token，一个 tile 内 Q/K 各发一次 VF。
 *
 * RMSNorm + MRoPE 融合在一个 VF 段里完成（op_kernel/arch35/..._mrope_vf.h）：
 * bf16->fp32 的 x、Gather 合并后的 cos/sin、RoPE 交叉项都只存在 vreg 里，
 * 所以计算区不再有 cosSinExpand / ropeTmp / qkvFp32 / reduceTmp / rmsInv 这些中间 buffer。
 *
 * gammaTileBuf 是随 u 的：cat_indices 可以把 und/gen 任意交错，同一 tile 内相邻 token
 * 的 gamma 可能不同，VF 里按 token 索引 gamma，所以要先把每 token 该用的那份铺平。
 * 它写完就在同一次 Compute 里消费掉，不需要 DoubleBuffer。
 *
 * 计算区（TBuf，与 u 无关）:
 *   wBf16Buf         WEIGHT_NUM * D * 2            4 个 gamma 原始 bf16
 *   wFp32Buf         WEIGHT_NUM * D * 4            4 个 gamma 的 float 版本，作为 gamma tile 的源
 *   gatherIdxBuf     align32((D/2) * 4)            axisLut 展开的 uint32 元素索引，每核建一次
 *
 * 随 u 的部分:
 *   qkvInQue     VECIN   2 * u * N * D * 2
 *   cosSinInQue  VECIN   2 * u * MROPE_AXIS_NUM * D * 4
 *   outQue       VECOUT  2 * u * N * D * 2         UB 内按 [u*q | u*k | u*v] 分段，便于 q 一笔连续写出
 *   gammaTileBuf VECCALC GAMMA_TILE_NUM * u * D * 4  单 buffer，按 token 交织
 *                        [t0 q gamma | t0 k gamma | t1 q gamma | t1 k gamma | ...]，
 *                        与 VF 里 gammaRow = base + t*2D、k gamma 在 +D 的取法对应
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
