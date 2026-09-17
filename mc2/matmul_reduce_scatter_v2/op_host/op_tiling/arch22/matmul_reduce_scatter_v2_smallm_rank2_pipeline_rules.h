/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_REDUCE_SCATTER_V2_SMALLM_RANK2_PIPELINE_RULES_H
#define MATMUL_REDUCE_SCATTER_V2_SMALLM_RANK2_PIPELINE_RULES_H

#include "matmul_reduce_scatter_v2_smallm_rank2_layout_rules.h"

namespace Tiling_Small_M::Tiling_Rank2_A2 {

// pValue优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, mn_div_k）
const DecisionNode pvalueRule[] = {
    // | 层级0 (索引0-0) |
    Threshold(FeatureType::M_DIV_N, 12.000000f), // 索引0
    // || 层级1 (索引1-2) ||
    Threshold(FeatureType::M_VALUE, 6144.0f), // 索引1
    Threshold(FeatureType::K_VALUE, 6144.0f), // 索引2
    // ||| 层级2 (索引3-6) |||
    Threshold(FeatureType::MN_DIV_K, 1536.000000f),  // 索引3
    Threshold(FeatureType::MN_DIV_K, 98304.000000f), // 索引4
    Threshold(FeatureType::M_MUL_N, 1572864.0f),     // 索引5
    Threshold(FeatureType::MN_DIV_K, 384.000000f),   // 索引6
    // |||| 层级3 (索引7-14) ||||
    Threshold(FeatureType::M_VALUE, 3072.0f),    // 索引7
    Threshold(FeatureType::M_MUL_N, 6291456.0f), // 索引8
    Threshold(FeatureType::K_VALUE, 6144.0f),    // 索引9
    Threshold(FeatureType::N_VALUE, 6144.0f),    // 索引10
    Ret(2),                                      // 索引11
    Threshold(FeatureType::N_VALUE, 384.0f),     // 索引12
    Ret(4),                                      // 索引13
    Ret(7),                                      // 索引14
    // ||||| 层级4 (索引15-30) |||||
    Threshold(FeatureType::M_MUL_N, 6291456.0f),    // 索引15
    Threshold(FeatureType::K_VALUE, 3072.0f),       // 索引16
    Threshold(FeatureType::MN_DIV_K, 6144.000000f), // 索引17
    Threshold(FeatureType::M_VALUE, 3072.0f),       // 索引18
    Threshold(FeatureType::M_DIV_N, 1.500000f),     // 索引19
    Ret(10),                                        // 索引20
    Ret(20),                                        // 索引21
    Ret(20),                                        // 索引22
    Placeholder(),                                  // 索引23: placeholder
    Placeholder(),                                  // 索引24: 占位符
    Ret(4),                                         // 索引25
    Ret(4),                                         // 索引26
    Placeholder(),                                  // 索引27: 空节点
    Placeholder(),                                  // 索引28: 未定义
    Placeholder(),                                  // 索引29: placeholder
    Placeholder(),                                  // 索引30: 占位
    // ====================== 层级5 (索引31-62) ======================
    Ret(1),                                          // 索引31
    Ret(2),                                          // 索引32
    Ret(2),                                          // 索引33
    Ret(2),                                          // 索引34
    Threshold(FeatureType::M_MUL_N, 786432.0f),      // 索引35
    Threshold(FeatureType::M_MUL_N, 3145728.0f),     // 索引36
    Threshold(FeatureType::MN_DIV_K, 24576.000000f), // 索引37
    Threshold(FeatureType::MN_DIV_K, 49152.000000f), // 索引38
    Ret(10),                                         // 索引39
    Threshold(FeatureType::MN_DIV_K, 24576.000000f), // 索引40
    Placeholder(),                                   // 索引41: placeholder
    Placeholder(),                                   // 索引42: 占位符
    Placeholder(),                                   // 索引43: 空节点
    Placeholder(),                                   // 索引44: 未定义
    Placeholder(),                                   // 索引45: placeholder
    Placeholder(),                                   // 索引46: 占位符
    Placeholder(),                                   // 索引47: 空节点
    Placeholder(),                                   // 索引48: 未定义
    Placeholder(),                                   // 索引49: placeholder
    Placeholder(),                                   // 索引50: 占位符
    Placeholder(),                                   // 索引51: 空节点
    Placeholder(),                                   // 索引52: 未定义
    Placeholder(),                                   // 索引53: placeholder
    Placeholder(),                                   // 索引54: 占位符
    Placeholder(),                                   // 索引55: 空节点
    Placeholder(),                                   // 索引56: 未定义
    Placeholder(),                                   // 索引57: placeholder
    Placeholder(),                                   // 索引58: 占位符
    Placeholder(),                                   // 索引59: 空节点
    Placeholder(),                                   // 索引60: 未定义
    Placeholder(),                                   // 索引61: placeholder
    Placeholder(),                                   // 索引62: 占位符
    // <层级6> (索引63-126)
    Placeholder(), // 索引63: placeholder
    Placeholder(), // 索引64: 占位符
    Placeholder(), // 索引65: 空节点
    Placeholder(), // 索引66: 未定义
    Placeholder(), // 索引67: placeholder
    Placeholder(), // 索引68: 占位符
    Placeholder(), // 索引69: 空节点
    Placeholder(), // 索引70: 未定义
    Ret(1),        // 索引71
    Ret(2),        // 索引72
    Ret(4),        // 索引73
    Ret(4),        // 索引74
    Ret(3),        // 索引75
    Ret(8),        // 索引76
    Ret(5),        // 索引77
    Ret(10),       // 索引78
    Placeholder(), // 索引79: placeholder
    Placeholder(), // 索引80: 占位符
    Ret(4),        // 索引81
    Ret(8),        // 索引82
    Placeholder(), // 索引83: 空节点
    Placeholder(), // 索引84: 未定义
    Placeholder(), // 索引85: placeholder
    Placeholder(), // 索引86: 占位符
    Placeholder(), // 索引87: 空节点
    Placeholder(), // 索引88: 未定义
    Placeholder(), // 索引89: placeholder
    Placeholder(), // 索引90: 占位符
    Placeholder(), // 索引91: 空节点
    Placeholder(), // 索引92: 未定义
    Placeholder(), // 索引93: placeholder
    Placeholder(), // 索引94: 占位符
    Placeholder(), // 索引95: 空节点
    Placeholder(), // 索引96: 未定义
    Placeholder(), // 索引97: placeholder
    Placeholder(), // 索引98: 占位符
    Placeholder(), // 索引99: 空节点
    Placeholder(), // 索引100: 未定义
    Placeholder(), // 索引101: placeholder
    Placeholder(), // 索引102: 占位符
    Placeholder(), // 索引103: 空节点
    Placeholder(), // 索引104: 未定义
    Placeholder(), // 索引105: placeholder
    Placeholder(), // 索引106: 占位符
    Placeholder(), // 索引107: 空节点
    Placeholder(), // 索引108: 未定义
    Placeholder(), // 索引109: placeholder
    Placeholder(), // 索引110: 占位符
    Placeholder(), // 索引111: 空节点
    Placeholder(), // 索引112: 未定义
    Placeholder(), // 索引113: placeholder
    Placeholder(), // 索引114: 占位符
    Placeholder(), // 索引115: 空节点
    Placeholder(), // 索引116: 未定义
    Placeholder(), // 索引117: placeholder
    Placeholder(), // 索引118: 占位符
    Placeholder(), // 索引119: 空节点
    Placeholder(), // 索引120: 未定义
    Placeholder(), // 索引121: placeholder
    Placeholder(), // 索引122: 占位符
    Placeholder(), // 索引123: 空节点
    Placeholder(), // 索引124: 未定义
    Placeholder(), // 索引125: placeholder
    Placeholder(), // 索引126: 占位符
};

// ubMoveNum优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode ubmovenumRule[] = {
    // (层级0) [索引0-0]
    Threshold(FeatureType::M_MUL_N, 98304.0f), // 索引0
    // (层级1) [索引1-2]
    Threshold(FeatureType::MN_DIV_K, 48.000000f), // 索引1
    Threshold(FeatureType::K_VALUE, 3072.0f),     // 索引2
    // (层级2) [索引3-6]
    Ret(16),                                      // 索引3
    Threshold(FeatureType::MN_DIV_K, 96.000000f), // 索引4
    Threshold(FeatureType::M_MUL_N, 12582912.0f), // 索引5
    Threshold(FeatureType::MN_DIV_K, 96.000000f), // 索引6
    // (层级3) [索引7-14]
    Placeholder(),                                 // 索引7: placeholder
    Placeholder(),                                 // 索引8: 占位符
    Threshold(FeatureType::M_VALUE, 192.0f),       // 索引9
    Ret(16),                                       // 索引10
    Threshold(FeatureType::MN_DIV_K, 192.000000f), // 索引11
    Threshold(FeatureType::WORLD_MUL_K, 768.0f),   // 索引12
    Threshold(FeatureType::M_VALUE, 192.0f),       // 索引13
    Threshold(FeatureType::M_VALUE, 6144.0f),      // 索引14
    // 【层级4】(索引15-30)
    Placeholder(),                                 // 索引15: 空节点
    Placeholder(),                                 // 索引16: 未定义
    Placeholder(),                                 // 索引17: placeholder
    Placeholder(),                                 // 索引18: 占位符
    Ret(8),                                        // 索引19
    Ret(16),                                       // 索引20
    Placeholder(),                                 // 索引21: 空节点
    Placeholder(),                                 // 索引22: 未定义
    Threshold(FeatureType::M_DIV_N, 0.187500f),    // 索引23
    Threshold(FeatureType::M_MUL_N, 196608.0f),    // 索引24
    Threshold(FeatureType::M_DIV_N, 1.500000f),    // 索引25
    Threshold(FeatureType::N_VALUE, 6144.0f),      // 索引26
    Threshold(FeatureType::M_MUL_N, 393216.0f),    // 索引27
    Threshold(FeatureType::M_MUL_N, 196608.0f),    // 索引28
    Threshold(FeatureType::WORLD_MUL_K, 12288.0f), // 索引29
    Threshold(FeatureType::MN_DIV_K, 768.000000f), // 索引30
    // 【层级5】(索引31-62)
    Placeholder(),                                // 索引31: placeholder
    Placeholder(),                                // 索引32: 占位符
    Placeholder(),                                // 索引33: 空节点
    Placeholder(),                                // 索引34: 未定义
    Placeholder(),                                // 索引35: placeholder
    Placeholder(),                                // 索引36: 占位符
    Placeholder(),                                // 索引37: 空节点
    Placeholder(),                                // 索引38: 未定义
    Placeholder(),                                // 索引39: placeholder
    Placeholder(),                                // 索引40: 占位符
    Placeholder(),                                // 索引41: 空节点
    Placeholder(),                                // 索引42: 未定义
    Placeholder(),                                // 索引43: placeholder
    Placeholder(),                                // 索引44: 占位符
    Placeholder(),                                // 索引45: 空节点
    Placeholder(),                                // 索引46: 未定义
    Threshold(FeatureType::MN_DIV_K, 96.000000f), // 索引47
    Ret(8),                                       // 索引48
    Threshold(FeatureType::M_VALUE, 384.0f),      // 索引49
    Threshold(FeatureType::M_DIV_N, 0.023438f),   // 索引50
    Threshold(FeatureType::M_MUL_N, 25165824.0f), // 索引51
    Ret(4),                                       // 索引52
    Threshold(FeatureType::M_MUL_N, 25165824.0f), // 索引53
    Threshold(FeatureType::WORLD_MUL_K, 1536.0f), // 索引54
    Threshold(FeatureType::MN_DIV_K, 24.000000f), // 索引55
    Ret(16),                                      // 索引56
    Ret(8),                                       // 索引57
    Threshold(FeatureType::MN_DIV_K, 48.000000f), // 索引58
    Threshold(FeatureType::M_DIV_N, 0.093750f),   // 索引59
    Threshold(FeatureType::M_VALUE, 1536.0f),     // 索引60
    Threshold(FeatureType::K_VALUE, 6144.0f),     // 索引61
    Threshold(FeatureType::N_VALUE, 6144.0f),     // 索引62
    // 【层级6】(索引63-126)
    Placeholder(), // 索引63: placeholder
    Placeholder(), // 索引64: 占位符
    Placeholder(), // 索引65: 空节点
    Placeholder(), // 索引66: 未定义
    Placeholder(), // 索引67: placeholder
    Placeholder(), // 索引68: 占位符
    Placeholder(), // 索引69: 空节点
    Placeholder(), // 索引70: 未定义
    Placeholder(), // 索引71: placeholder
    Placeholder(), // 索引72: 占位符
    Placeholder(), // 索引73: 空节点
    Placeholder(), // 索引74: 未定义
    Placeholder(), // 索引75: placeholder
    Placeholder(), // 索引76: 占位符
    Placeholder(), // 索引77: 空节点
    Placeholder(), // 索引78: 未定义
    Placeholder(), // 索引79: placeholder
    Placeholder(), // 索引80: 占位符
    Placeholder(), // 索引81: 空节点
    Placeholder(), // 索引82: 未定义
    Placeholder(), // 索引83: placeholder
    Placeholder(), // 索引84: 占位符
    Placeholder(), // 索引85: 空节点
    Placeholder(), // 索引86: 未定义
    Placeholder(), // 索引87: placeholder
    Placeholder(), // 索引88: 占位符
    Placeholder(), // 索引89: 空节点
    Placeholder(), // 索引90: 未定义
    Placeholder(), // 索引91: placeholder
    Placeholder(), // 索引92: 占位符
    Placeholder(), // 索引93: 空节点
    Placeholder(), // 索引94: 未定义
    Ret(4),        // 索引95
    Ret(8),        // 索引96
    Placeholder(), // 索引97: placeholder
    Placeholder(), // 索引98: 占位符
    Ret(8),        // 索引99
    Ret(4),        // 索引100
    Ret(4),        // 索引101
    Ret(16),       // 索引102
    Ret(8),        // 索引103
    Ret(4),        // 索引104
    Placeholder(), // 索引105: 空节点
    Placeholder(), // 索引106: 未定义
    Ret(8),        // 索引107
    Ret(8),        // 索引108
    Ret(8),        // 索引109
    Ret(8),        // 索引110
    Ret(8),        // 索引111
    Ret(4),        // 索引112
    Placeholder(), // 索引113: placeholder
    Placeholder(), // 索引114: 占位符
    Placeholder(), // 索引115: 空节点
    Placeholder(), // 索引116: 未定义
    Ret(16),       // 索引117
    Ret(8),        // 索引118
    Ret(4),        // 索引119
    Ret(8),        // 索引120
    Ret(4),        // 索引121
    Ret(4),        // 索引122
    Ret(16),       // 索引123
    Ret(4),        // 索引124
    Ret(8),        // 索引125
    Ret(4),        // 索引126
};

inline int GetOptimalM0(int m, int k, int n)
{
    return TraverseDecisionTree(m0Rule, m, k, n, RANKSIZE_TWO, DEFAULT_M0);
}

inline int GetOptimalSwizzlCount(int m, int k, int n)
{
    return TraverseDecisionTree(swizzlcountRule, m, k, n, RANKSIZE_TWO, DEFAULT_SWIZZLCOUNT);
}

inline int GetOptimalSwizzlDirect(int m, int k, int n)
{
    return TraverseDecisionTree(swizzldirectRule, m, k, n, RANKSIZE_TWO, DEFAULT_SWIZZLDIRECT);
}

inline int GetOptimalPValue(int m, int k, int n)
{
    return TraverseDecisionTree(pvalueRule, m, k, n, RANKSIZE_TWO, DEFAULT_PVALUE);
}

inline int GetOptimalUbmovenum(int m, int k, int n)
{
    return TraverseDecisionTree(ubmovenumRule, m, k, n, RANKSIZE_TWO, DEFAULT_UBMOVENUM);
}

} // namespace Tiling_Small_M::Tiling_Rank2_A2

#endif
