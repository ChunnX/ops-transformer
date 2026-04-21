/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * [PSEUDOCODE] MOE Dispatch Tiling 重构公共层 - C 风格工具函数
 * 配套文档：dispatch_analysis_opus_3/architecture_v0.md §4.2 §4.3 §5.2
 * 配套伪代码：dispatch_analysis_opus_3/pseudocode_v0.md §1.2
 *
 * 原则：无状态、inline、参数 <= 5、单函数 <= 50 行
 */

#ifndef MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_UTILS_H
#define MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_UTILS_H

#include <cstdint>
#include "defs.h"
// TODO: include framework headers (TilingContext, graphStatus, DataType, TensorDesc, etc.)

namespace moe_dispatch_utils {

using namespace moe_dispatch_defs;

// ========== 数据结构 ==========

// Tensor 校验规则（单个 tensor 的 dim+dtype+format 校验描述）
// 见 architecture_v0.md §4.2
enum class TensorType { INPUT, OUTPUT };

struct TensorCheckRule {
    uint32_t    index;                          // tensor 在 context 中的索引（从 Layout 获取）
    TensorType  type;                           // INPUT / OUTPUT
    uint32_t    expectedDim;                    // 期望维度（0 = 不检查）
    DataType    validDtypes[MAX_VALID_DTYPES];  // 合法 dtype 集合（空 = 不检查）
    uint32_t    dtypeCount;                     // validDtypes 有效个数
    const char* name;                           // 日志用名称
};

// 属性范围校验规则（单个属性的范围校验描述）
// 见 architecture_v0.md §4.3
struct AttrCheckRule {
    uint32_t    attrIndex;  // 属性在 context 中的索引（从 Layout 获取）
    const char* attrName;   // 日志用名称
    uint32_t    minVal;     // 下界（含）
    uint32_t    maxVal;     // 上界（含）
    bool        required;   // 是否必选（必选时 null 即报错）
};

// shared_expert 计算结果
// 见 architecture_v0.md §4.6
struct SharedExpertResult {
    uint32_t localExpertNum;       // 本 rank 负责的 expert 数量
    bool     isSharedRank;         // 是否为 shared expert rank
    uint32_t moeExpertPerRankNum;  // 每 rank expert 数量
};

// elastic 计算结果（仅 v2 使用）
// 见 architecture_v0.md §4.6
struct ElasticResult {
    bool     hasElastic;
    uint32_t adjustedA;
    uint32_t adjustedExpertTokenDim0;
    uint32_t adjustedEpRecvDim0;
};

// 平台信息
struct PlatformInfo {
    uint32_t aivNum;
    uint32_t ubSize;
    uint32_t blockDim;
};

// ========== 批量校验函数 ==========

// 批量校验必选 tensor（dim + dtype + format），按 tensor 拆分
// 见 architecture_v0.md §4.2
inline graphStatus CheckTensorRules(TilingContext* ctx,
                                    const TensorCheckRule* rules,
                                    uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        // TODO: 调用 CheckOneTensorRule(ctx, rules[i])
    }
    return ge::GRAPH_SUCCESS;
}

// 单个 tensor 校验（条件性 tensor 统一入口）
inline graphStatus CheckOneTensorRule(TilingContext* ctx, const TensorCheckRule& rule) {
    // TODO: 1. 获取 tensor desc（INPUT 或 OUTPUT 按 rule.type）
    // TODO: 2. rule.expectedDim > 0 时调用 CheckDimNum
    // TODO: 3. rule.dtypeCount > 0 时调用 CheckDtypeIn
    // TODO: 4. 调用 CheckNotFractalNZ 校验 format
    // 失败时 OPS_LOG_E 输出 rule.name 相关错误
    return ge::GRAPH_SUCCESS;
}

// 批量属性范围校验
// 见 architecture_v0.md §4.3
inline graphStatus CheckAttrRules(TilingContext* ctx,
                                  const AttrCheckRule* rules,
                                  uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        // TODO: 从 ctx 获取 attr 值
        // TODO: rules[i].required 时 null 即 OPS_LOG_E 并返回 GRAPH_FAILED
        // TODO: 范围校验：val < minVal || val > maxVal 时 OPS_LOG_E 并返回 GRAPH_FAILED
    }
    return ge::GRAPH_SUCCESS;
}

// ========== 原子校验工具（供内部和特性 checker 使用）==========

inline graphStatus CheckDimNum(const ge::Shape& shape, uint32_t expected) {
    // TODO: shape.GetDimNum() != expected 时 OPS_LOG_E 并返回 GRAPH_FAILED
    return ge::GRAPH_SUCCESS;
}

inline graphStatus CheckDtypeIn(DataType dtype, const DataType* validSet, uint32_t count) {
    // TODO: 遍历 validSet，找不到时 OPS_LOG_E 并返回 GRAPH_FAILED
    return ge::GRAPH_SUCCESS;
}

inline graphStatus CheckNotFractalNZ(Format format) {
    // TODO: format == FORMAT_FRACTAL_NZ 时 OPS_LOG_E 并返回 GRAPH_FAILED
    return ge::GRAPH_SUCCESS;
}

inline graphStatus CheckRangeU32(uint32_t val, uint32_t minVal, uint32_t maxVal,
                                 const char* name) {
    // TODO: val < minVal || val > maxVal 时 OPS_LOG_E 并返回 GRAPH_FAILED
    return ge::GRAPH_SUCCESS;
}

// ========== 公共中间值计算 ==========

// shared_expert 相关中间值计算
// 见 architecture_v0.md §4.6, §5.2
inline SharedExpertResult CalcLocalExpertNum(uint32_t epRankId,
                                             uint32_t sharedRankNum,
                                             uint32_t expertNum,
                                             uint32_t epWorldSize) {
    SharedExpertResult result{};
    // TODO: 计算 localExpertNum, isSharedRank, moeExpertPerRankNum
    // 逻辑来自原 v2/setup/teardown 中各自重复的计算
    return result;
}

// shared_expert 关系完整性校验（整除、范围）
inline graphStatus CheckSharedExpertRelation(uint32_t sharedExpertNum,
                                             uint32_t sharedExpertRankNum) {
    // TODO: sharedExpertNum > 0 时校验 sharedExpertNum % sharedExpertRankNum == 0
    // 以及 sharedExpertNum <= MAX_SHARED_EXPERT_NUM 等
    return ge::GRAPH_SUCCESS;
}

// ========== 通用数学工具 ==========

inline uint32_t CeilDiv(uint32_t a, uint32_t b) {
    // TODO: assert b != 0
    return (a + b - 1) / b;
}

inline uint32_t AlignUp(uint32_t val, uint32_t align) {
    return CeilDiv(val, align) * align;
}

// ========== 平台信息获取 ==========

inline PlatformInfo GetPlatformInfo(TilingContext* ctx) {
    PlatformInfo info{};
    // TODO: 从 ctx 获取 aivNum、ubSize、blockDim
    return info;
}

}  // namespace moe_dispatch_utils

#endif  // MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_UTILS_H
