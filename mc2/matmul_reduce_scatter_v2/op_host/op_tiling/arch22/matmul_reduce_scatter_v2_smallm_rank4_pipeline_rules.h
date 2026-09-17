/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_REDUCE_SCATTER_V2_SMALLM_RANK4_PIPELINE_RULES_H
#define MATMUL_REDUCE_SCATTER_V2_SMALLM_RANK4_PIPELINE_RULES_H

#include "matmul_reduce_scatter_v2_smallm_rank4_layout_rules.h"

namespace Tiling_Small_M::Tiling_Rank4_A2 {

// pValue优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode pvalueRule[] = {
    // ====================== 层级0 (索引0-0) ======================
    Threshold(FeatureType::M_VALUE, 3072.0f), // 索引0
    // ====================== 层级1 (索引1-2) ======================
    Threshold(FeatureType::M_VALUE, 768.0f),  // 索引1
    Threshold(FeatureType::M_VALUE, 6144.0f), // 索引2
    // ====================== 层级2 (索引3-6) ======================
    Threshold(FeatureType::MN_DIV_K, 3072.000000f),  // 索引3
    Threshold(FeatureType::MN_DIV_K, 12288.000000f), // 索引4
    Threshold(FeatureType::K_VALUE, 768.0f),         // 索引5
    Threshold(FeatureType::M_DIV_N, 12.000000f),     // 索引6
    // ====================== 层级3 (索引7-14) ======================
    Threshold(FeatureType::MN_DIV_K, 768.000000f),   // 索引7
    Threshold(FeatureType::M_DIV_N, 0.093750f),      // 索引8
    Threshold(FeatureType::M_MUL_N, 1572864.0f),     // 索引9
    Threshold(FeatureType::M_VALUE, 1536.0f),        // 索引10
    Threshold(FeatureType::N_VALUE, 1536.0f),        // 索引11
    Threshold(FeatureType::M_MUL_N, 3145728.0f),     // 索引12
    Threshold(FeatureType::MN_DIV_K, 98304.000000f), // 索引13
    Threshold(FeatureType::MN_DIV_K, 12288.000000f), // 索引14
    // ====================== 层级4 (索引15-30) ======================
    Threshold(FeatureType::WORLD_MUL_K, 1536.0f),    // 索引15
    Threshold(FeatureType::M_DIV_N, 0.187500f),      // 索引16
    Threshold(FeatureType::M_DIV_N, 0.023438f),      // 索引17
    Ret(2),                                          // 索引18
    Threshold(FeatureType::MN_DIV_K, 768.000000f),   // 索引19
    Threshold(FeatureType::M_VALUE, 1536.0f),        // 索引20
    Ret(2),                                          // 索引21
    Threshold(FeatureType::M_MUL_N, 6291456.0f),     // 索引22
    Threshold(FeatureType::MN_DIV_K, 12288.000000f), // 索引23
    Threshold(FeatureType::N_VALUE, 3072.0f),        // 索引24
    Ret(2),                                          // 索引25
    Threshold(FeatureType::M_DIV_N, 0.750000f),      // 索引26
    Threshold(FeatureType::WORLD_MUL_K, 1536.0f),    // 索引27
    Ret(20),                                         // 索引28
    Threshold(FeatureType::K_VALUE, 384.0f),         // 索引29
    Ret(7),                                          // 索引30
    // ====================== 层级5 (索引31-62) ======================
    Ret(1),                                         // 索引31
    Ret(1),                                         // 索引32
    Ret(1),                                         // 索引33
    Threshold(FeatureType::K_VALUE, 768.0f),        // 索引34
    Ret(2),                                         // 索引35
    Threshold(FeatureType::M_MUL_N, 1572864.0f),    // 索引36
    Placeholder(),                                  // 索引37: 占位
    Placeholder(),                                  // 索引38: 占位
    Ret(1),                                         // 索引39
    Threshold(FeatureType::M_MUL_N, 786432.0f),     // 索引40
    Threshold(FeatureType::MN_DIV_K, 6144.000000f), // 索引41
    Threshold(FeatureType::WORLD_MUL_K, 3072.0f),   // 索引42
    Placeholder(),                                  // 索引43: 占位
    Placeholder(),                                  // 索引44: 占位
    Ret(4),                                         // 索引45
    Ret(5),                                         // 索引46
    Threshold(FeatureType::M_MUL_N, 1572864.0f),    // 索引47
    Ret(7),                                         // 索引48
    Ret(8),                                         // 索引49
    Ret(10),                                        // 索引50
    Placeholder(),                                  // 索引51: 占位
    Placeholder(),                                  // 索引52: 占位
    Threshold(FeatureType::MN_DIV_K, 6144.000000f), // 索引53
    Threshold(FeatureType::K_VALUE, 6144.0f),       // 索引54
    Ret(8),                                         // 索引55
    Threshold(FeatureType::M_DIV_N, 1.500000f),     // 索引56
    Placeholder(),                                  // 索引57: 占位
    Placeholder(),                                  // 索引58: 占位
    Ret(4),                                         // 索引59
    Ret(4),                                         // 索引60
    Placeholder(),                                  // 索引61: 占位
    Placeholder(),                                  // 索引62: 占位
    // ====================== 层级6 (索引63-126) ======================
    Placeholder(), // 索引63: 占位
    Placeholder(), // 索引64: 占位
    Placeholder(), // 索引65: 占位
    Placeholder(), // 索引66: 占位
    Placeholder(), // 索引67: 占位
    Placeholder(), // 索引68: 占位
    Ret(1),        // 索引69
    Ret(2),        // 索引70
    Placeholder(), // 索引71: 占位
    Placeholder(), // 索引72: 占位
    Ret(1),        // 索引73
    Ret(1),        // 索引74
    Placeholder(), // 索引75: 占位
    Placeholder(), // 索引76: 占位
    Placeholder(), // 索引77: 占位
    Placeholder(), // 索引78: 占位
    Placeholder(), // 索引79: 占位
    Placeholder(), // 索引80: 占位
    Ret(1),        // 索引81
    Ret(2),        // 索引82
    Ret(2),        // 索引83
    Ret(3),        // 索引84
    Ret(4),        // 索引85
    Ret(3),        // 索引86
    Placeholder(), // 索引87: 占位
    Placeholder(), // 索引88: 占位
    Placeholder(), // 索引89: 占位
    Placeholder(), // 索引90: 占位
    Placeholder(), // 索引91: 占位
    Placeholder(), // 索引92: 占位
    Placeholder(), // 索引93: 占位
    Placeholder(), // 索引94: 占位
    Ret(2),        // 索引95
    Ret(4),        // 索引96
    Placeholder(), // 索引97: 占位
    Placeholder(), // 索引98: 占位
    Placeholder(), // 索引99: 占位
    Placeholder(), // 索引100: 占位
    Placeholder(), // 索引101: 占位
    Placeholder(), // 索引102: 占位
    Placeholder(), // 索引103: 占位
    Placeholder(), // 索引104: 占位
    Placeholder(), // 索引105: 占位
    Placeholder(), // 索引106: 占位
    Ret(8),        // 索引107
    Ret(5),        // 索引108
    Ret(2),        // 索引109
    Ret(5),        // 索引110
    Placeholder(), // 索引111: 占位
    Placeholder(), // 索引112: 占位
    Ret(20),       // 索引113
    Ret(10),       // 索引114
    Placeholder(), // 索引115: 占位
    Placeholder(), // 索引116: 占位
    Placeholder(), // 索引117: 占位
    Placeholder(), // 索引118: 占位
    Placeholder(), // 索引119: 占位
    Placeholder(), // 索引120: 占位
    Placeholder(), // 索引121: 占位
    Placeholder(), // 索引122: 占位
    Placeholder(), // 索引123: 占位
    Placeholder(), // 索引124: 占位
    Placeholder(), // 索引125: 占位
    Placeholder(), // 索引126: 占位
};

// ubMoveNum优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode ubmovenumRule[] = {
    // ====================== 层级0 (索引0-0) ======================
    Threshold(FeatureType::M_VALUE, 384.0f), // 索引0
    // ====================== 层级1 (索引1-2) ======================
    Threshold(FeatureType::N_VALUE, 768.0f),    // 索引1
    Threshold(FeatureType::M_MUL_N, 786432.0f), // 索引2
    // ====================== 层级2 (索引3-6) ======================
    Threshold(FeatureType::MN_DIV_K, 192.000000f), // 索引3
    Threshold(FeatureType::M_DIV_N, 0.046875f),    // 索引4
    Threshold(FeatureType::WORLD_MUL_K, 1536.0f),  // 索引5
    Threshold(FeatureType::N_VALUE, 3072.0f),      // 索引6
    // ====================== 层级3 (索引7-14) ======================
    Threshold(FeatureType::K_VALUE, 384.0f),       // 索引7
    Threshold(FeatureType::MN_DIV_K, 384.000000f), // 索引8
    Threshold(FeatureType::MN_DIV_K, 768.000000f), // 索引9
    Threshold(FeatureType::K_VALUE, 1536.0f),      // 索引10
    Threshold(FeatureType::MN_DIV_K, 768.000000f), // 索引11
    Threshold(FeatureType::M_DIV_N, 1.500000f),    // 索引12
    Threshold(FeatureType::M_MUL_N, 1572864.0f),   // 索引13
    Threshold(FeatureType::WORLD_MUL_K, 3072.0f),  // 索引14
    // ====================== 层级4 (索引15-30) ======================
    Ret(8),                                         // 索引15
    Threshold(FeatureType::M_DIV_N, 0.750000f),     // 索引16
    Ret(8),                                         // 索引17
    Ret(16),                                        // 索引18
    Threshold(FeatureType::MN_DIV_K, 96.000000f),   // 索引19
    Threshold(FeatureType::MN_DIV_K, 6144.000000f), // 索引20
    Threshold(FeatureType::N_VALUE, 1536.0f),       // 索引21
    Threshold(FeatureType::M_VALUE, 192.0f),        // 索引22
    Ret(16),                                        // 索引23
    Ret(8),                                         // 索引24
    Threshold(FeatureType::WORLD_MUL_K, 6144.0f),   // 索引25
    Ret(8),                                         // 索引26
    Threshold(FeatureType::K_VALUE, 384.0f),        // 索引27
    Threshold(FeatureType::WORLD_MUL_K, 24576.0f),  // 索引28
    Threshold(FeatureType::M_MUL_N, 6291456.0f),    // 索引29
    Threshold(FeatureType::MN_DIV_K, 768.000000f),  // 索引30
    // ====================== 层级5 (索引31-62) ======================
    Placeholder(),                                 // 索引31: 占位
    Placeholder(),                                 // 索引32: 占位
    Threshold(FeatureType::MN_DIV_K, 12.000000f),  // 索引33
    Threshold(FeatureType::MN_DIV_K, 12.000000f),  // 索引34
    Placeholder(),                                 // 索引35: 占位
    Placeholder(),                                 // 索引36: 占位
    Placeholder(),                                 // 索引37: 占位
    Placeholder(),                                 // 索引38: 占位
    Ret(8),                                        // 索引39
    Threshold(FeatureType::MN_DIV_K, 384.000000f), // 索引40
    Threshold(FeatureType::M_DIV_N, 0.023438f),    // 索引41
    Ret(16),                                       // 索引42
    Ret(8),                                        // 索引43
    Threshold(FeatureType::M_MUL_N, 786432.0f),    // 索引44
    Ret(8),                                        // 索引45
    Ret(8),                                        // 索引46
    Placeholder(),                                 // 索引47: 占位
    Placeholder(),                                 // 索引48: 占位
    Placeholder(),                                 // 索引49: 占位
    Placeholder(),                                 // 索引50: 占位
    Threshold(FeatureType::M_DIV_N, 0.750000f),    // 索引51
    Ret(8),                                        // 索引52
    Placeholder(),                                 // 索引53: 占位
    Placeholder(),                                 // 索引54: 占位
    Threshold(FeatureType::M_VALUE, 1536.0f),      // 索引55
    Threshold(FeatureType::WORLD_MUL_K, 12288.0f), // 索引56
    Threshold(FeatureType::M_DIV_N, 1.500000f),    // 索引57
    Threshold(FeatureType::N_VALUE, 1536.0f),      // 索引58
    Threshold(FeatureType::K_VALUE, 384.0f),       // 索引59
    Threshold(FeatureType::M_DIV_N, 1.500000f),    // 索引60
    Threshold(FeatureType::N_VALUE, 6144.0f),      // 索引61
    Threshold(FeatureType::M_MUL_N, 6291456.0f),   // 索引62
    // ====================== 层级6 (索引63-126) ======================
    Placeholder(), // 索引63: 占位
    Placeholder(), // 索引64: 占位
    Placeholder(), // 索引65: 占位
    Placeholder(), // 索引66: 占位
    Ret(16),       // 索引67
    Ret(16),       // 索引68
    Ret(8),        // 索引69
    Ret(16),       // 索引70
    Placeholder(), // 索引71: 占位
    Placeholder(), // 索引72: 占位
    Placeholder(), // 索引73: 占位
    Placeholder(), // 索引74: 占位
    Placeholder(), // 索引75: 占位
    Placeholder(), // 索引76: 占位
    Placeholder(), // 索引77: 占位
    Placeholder(), // 索引78: 占位
    Placeholder(), // 索引79: 占位
    Placeholder(), // 索引80: 占位
    Ret(16),       // 索引81
    Ret(16),       // 索引82
    Ret(4),        // 索引83
    Ret(8),        // 索引84
    Placeholder(), // 索引85: 占位
    Placeholder(), // 索引86: 占位
    Placeholder(), // 索引87: 占位
    Placeholder(), // 索引88: 占位
    Ret(4),        // 索引89
    Ret(8),        // 索引90
    Placeholder(), // 索引91: 占位
    Placeholder(), // 索引92: 占位
    Placeholder(), // 索引93: 占位
    Placeholder(), // 索引94: 占位
    Placeholder(), // 索引95: 占位
    Placeholder(), // 索引96: 占位
    Placeholder(), // 索引97: 占位
    Placeholder(), // 索引98: 占位
    Placeholder(), // 索引99: 占位
    Placeholder(), // 索引100: 占位
    Placeholder(), // 索引101: 占位
    Placeholder(), // 索引102: 占位
    Ret(8),        // 索引103
    Ret(4),        // 索引104
    Placeholder(), // 索引105: 占位
    Placeholder(), // 索引106: 占位
    Placeholder(), // 索引107: 占位
    Placeholder(), // 索引108: 占位
    Placeholder(), // 索引109: 占位
    Placeholder(), // 索引110: 占位
    Ret(4),        // 索引111
    Ret(4),        // 索引112
    Ret(8),        // 索引113
    Ret(4),        // 索引114
    Ret(8),        // 索引115
    Ret(4),        // 索引116
    Ret(8),        // 索引117
    Ret(4),        // 索引118
    Ret(8),        // 索引119
    Ret(4),        // 索引120
    Ret(4),        // 索引121
    Ret(4),        // 索引122
    Ret(4),        // 索引123
    Ret(8),        // 索引124
    Ret(8),        // 索引125
    Ret(8),        // 索引126
};

inline int GetOptimalM0(int m, int k, int n)
{
    return TraverseDecisionTree(m0Rule, m, k, n, RANKSIZE_FOUR, DEFAULT_M0);
}

inline int GetOptimalSwizzlCount(int m, int k, int n)
{
    return TraverseDecisionTree(swizzlcountRule, m, k, n, RANKSIZE_FOUR, DEFAULT_SWIZZLCOUNT);
}

inline int GetOptimalSwizzlDirect(int m, int k, int n)
{
    return TraverseDecisionTree(swizzldirectRule, m, k, n, RANKSIZE_FOUR, DEFAULT_SWIZZLDIRECT);
}

inline int GetOptimalPValue(int m, int k, int n)
{
    return TraverseDecisionTree(pvalueRule, m, k, n, RANKSIZE_FOUR, DEFAULT_PVALUE);
}

inline int GetOptimalUbmovenum(int m, int k, int n)
{
    return TraverseDecisionTree(ubmovenumRule, m, k, n, RANKSIZE_FOUR, DEFAULT_UBMOVENUM);
}

} // namespace Tiling_Small_M::Tiling_Rank4_A2

#endif
