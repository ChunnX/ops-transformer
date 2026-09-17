/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_REDUCE_SCATTER_V2_SMALLM_RANK8_PIPELINE_RULES_H
#define MATMUL_REDUCE_SCATTER_V2_SMALLM_RANK8_PIPELINE_RULES_H

#include "matmul_reduce_scatter_v2_smallm_rank8_layout_rules.h"

namespace Tiling_Small_M::Tiling_Rank8_A2 {

// pValue优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode pvalueRule[] = {
    // ====================== 层级0 (索引0-0) ======================
    Threshold(FeatureType::M_MUL_N, 6291456.0f), // 索引0
    // ====================== 层级1 (索引1-2) ======================
    Threshold(FeatureType::M_MUL_N, 1572864.0f), // 索引1
    Threshold(FeatureType::M_VALUE, 6144.0f),    // 索引2
    // ====================== 层级2 (索引3-6) ======================
    Threshold(FeatureType::M_MUL_N, 786432.0f),   // 索引3
    Threshold(FeatureType::K_VALUE, 6144.0f),     // 索引4
    Threshold(FeatureType::M_VALUE, 1536.0f),     // 索引5
    Threshold(FeatureType::M_MUL_N, 12582912.0f), // 索引6
    // ====================== 层级3 (索引7-14) ======================
    Ret(1),                                        // 索引7
    Threshold(FeatureType::N_VALUE, 6144.0f),      // 索引8
    Threshold(FeatureType::M_VALUE, 768.0f),       // 索引9
    Threshold(FeatureType::M_DIV_N, 0.187500f),    // 索引10
    Ret(2),                                        // 索引11
    Threshold(FeatureType::M_MUL_N, 12582912.0f),  // 索引12
    Threshold(FeatureType::WORLD_MUL_K, 24576.0f), // 索引13
    Threshold(FeatureType::WORLD_MUL_K, 49152.0f), // 索引14
    // ====================== 层级4 (索引15-30) ======================
    Placeholder(),                                  // 索引15: 占位
    Placeholder(),                                  // 索引16: 占位
    Threshold(FeatureType::M_VALUE, 768.0f),        // 索引17
    Threshold(FeatureType::WORLD_MUL_K, 3072.0f),   // 索引18
    Threshold(FeatureType::MN_DIV_K, 768.000000f),  // 索引19
    Threshold(FeatureType::M_VALUE, 1536.0f),       // 索引20
    Threshold(FeatureType::M_DIV_N, 0.093750f),     // 索引21
    Threshold(FeatureType::M_DIV_N, 3.000000f),     // 索引22
    Placeholder(),                                  // 索引23: 占位
    Placeholder(),                                  // 索引24: 占位
    Threshold(FeatureType::MN_DIV_K, 6144.000000f), // 索引25
    Threshold(FeatureType::M_DIV_N, 0.375000f),     // 索引26
    Ret(8),                                         // 索引27
    Ret(10),                                        // 索引28
    Threshold(FeatureType::M_DIV_N, 3.000000f),     // 索引29
    Ret(20),                                        // 索引30
    // ====================== 层级5 (索引31-62) ======================
    Placeholder(),                                   // 索引31: 占位
    Placeholder(),                                   // 索引32: 占位
    Placeholder(),                                   // 索引33: 占位
    Placeholder(),                                   // 索引34: 占位
    Threshold(FeatureType::MN_DIV_K, 192.000000f),   // 索引35
    Threshold(FeatureType::K_VALUE, 1536.0f),        // 索引36
    Ret(2),                                          // 索引37
    Ret(1),                                          // 索引38
    Ret(2),                                          // 索引39
    Threshold(FeatureType::K_VALUE, 3072.0f),        // 索引40
    Threshold(FeatureType::MN_DIV_K, 1536.000000f),  // 索引41
    Threshold(FeatureType::MN_DIV_K, 3072.000000f),  // 索引42
    Ret(1),                                          // 索引43
    Ret(4),                                          // 索引44
    Threshold(FeatureType::MN_DIV_K, 384.000000f),   // 索引45
    Threshold(FeatureType::M_DIV_N, 6.000000f),      // 索引46
    Placeholder(),                                   // 索引47: 占位
    Placeholder(),                                   // 索引48: 占位
    Placeholder(),                                   // 索引49: 占位
    Placeholder(),                                   // 索引50: 占位
    Threshold(FeatureType::MN_DIV_K, 1536.000000f),  // 索引51
    Threshold(FeatureType::N_VALUE, 3072.0f),        // 索引52
    Ret(5),                                          // 索引53
    Threshold(FeatureType::K_VALUE, 1536.0f),        // 索引54
    Placeholder(),                                   // 索引55: 占位
    Placeholder(),                                   // 索引56: 占位
    Placeholder(),                                   // 索引57: 占位
    Placeholder(),                                   // 索引58: 占位
    Threshold(FeatureType::MN_DIV_K, 98304.000000f), // 索引59
    Threshold(FeatureType::WORLD_MUL_K, 12288.0f),   // 索引60
    Placeholder(),                                   // 索引61: 占位
    Placeholder(),                                   // 索引62: 占位
    // ====================== 层级6 (索引63-126) ======================
    Placeholder(), // 索引63: 占位
    Placeholder(), // 索引64: 占位
    Placeholder(), // 索引65: 占位
    Placeholder(), // 索引66: 占位
    Placeholder(), // 索引67: 占位
    Placeholder(), // 索引68: 占位
    Placeholder(), // 索引69: 占位
    Placeholder(), // 索引70: 占位
    Ret(1),        // 索引71
    Ret(2),        // 索引72
    Ret(2),        // 索引73
    Ret(1),        // 索引74
    Placeholder(), // 索引75: 占位
    Placeholder(), // 索引76: 占位
    Placeholder(), // 索引77: 占位
    Placeholder(), // 索引78: 占位
    Placeholder(), // 索引79: 占位
    Placeholder(), // 索引80: 占位
    Ret(2),        // 索引81
    Ret(1),        // 索引82
    Ret(1),        // 索引83
    Ret(4),        // 索引84
    Ret(2),        // 索引85
    Ret(7),        // 索引86
    Placeholder(), // 索引87: 占位
    Placeholder(), // 索引88: 占位
    Placeholder(), // 索引89: 占位
    Placeholder(), // 索引90: 占位
    Ret(3),        // 索引91
    Ret(3),        // 索引92
    Ret(5),        // 索引93
    Ret(7),        // 索引94
    Placeholder(), // 索引95: 占位
    Placeholder(), // 索引96: 占位
    Placeholder(), // 索引97: 占位
    Placeholder(), // 索引98: 占位
    Placeholder(), // 索引99: 占位
    Placeholder(), // 索引100: 占位
    Placeholder(), // 索引101: 占位
    Placeholder(), // 索引102: 占位
    Ret(5),        // 索引103
    Ret(5),        // 索引104
    Ret(5),        // 索引105
    Ret(8),        // 索引106
    Placeholder(), // 索引107: 占位
    Placeholder(), // 索引108: 占位
    Ret(10),       // 索引109
    Ret(5),        // 索引110
    Placeholder(), // 索引111: 占位
    Placeholder(), // 索引112: 占位
    Placeholder(), // 索引113: 占位
    Placeholder(), // 索引114: 占位
    Placeholder(), // 索引115: 占位
    Placeholder(), // 索引116: 占位
    Placeholder(), // 索引117: 占位
    Placeholder(), // 索引118: 占位
    Ret(20),       // 索引119
    Ret(20),       // 索引120
    Ret(8),        // 索引121
    Ret(10),       // 索引122
    Placeholder(), // 索引123: 占位
    Placeholder(), // 索引124: 占位
    Placeholder(), // 索引125: 占位
    Placeholder(), // 索引126: 占位
};

// ubMoveNum优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode ubmovenumRule[] = {
    // ====================== 层级0 (索引0-0) ======================
    Threshold(FeatureType::M_VALUE, 768.0f), // 索引0
    // ====================== 层级1 (索引1-2) ======================
    Threshold(FeatureType::M_MUL_N, 3145728.0f), // 索引1
    Threshold(FeatureType::M_DIV_N, 3.000000f),  // 索引2
    // ====================== 层级2 (索引3-6) ======================
    Threshold(FeatureType::M_MUL_N, 786432.0f),      // 索引3
    Threshold(FeatureType::MN_DIV_K, 12288.000000f), // 索引4
    Threshold(FeatureType::M_MUL_N, 6291456.0f),     // 索引5
    Threshold(FeatureType::MN_DIV_K, 384.000000f),   // 索引6
    // ====================== 层级3 (索引7-14) ======================
    Threshold(FeatureType::MN_DIV_K, 768.000000f),  // 索引7
    Threshold(FeatureType::M_MUL_N, 1572864.0f),    // 索引8
    Ret(16),                                        // 索引9
    Ret(4),                                         // 索引10
    Threshold(FeatureType::MN_DIV_K, 3072.000000f), // 索引11
    Threshold(FeatureType::WORLD_MUL_K, 49152.0f),  // 索引12
    Threshold(FeatureType::K_VALUE, 1536.0f),       // 索引13
    Threshold(FeatureType::M_MUL_N, 393216.0f),     // 索引14
    // ====================== 层级4 (索引15-30) ======================
    Threshold(FeatureType::M_MUL_N, 98304.0f),      // 索引15
    Threshold(FeatureType::N_VALUE, 3072.0f),       // 索引16
    Threshold(FeatureType::M_VALUE, 192.0f),        // 索引17
    Threshold(FeatureType::MN_DIV_K, 3072.000000f), // 索引18
    Placeholder(),                                  // 索引19: 占位
    Placeholder(),                                  // 索引20: 占位
    Placeholder(),                                  // 索引21: 占位
    Placeholder(),                                  // 索引22: 占位
    Threshold(FeatureType::M_VALUE, 1536.0f),       // 索引23
    Threshold(FeatureType::M_DIV_N, 0.750000f),     // 索引24
    Threshold(FeatureType::M_VALUE, 3072.0f),       // 索引25
    Threshold(FeatureType::M_DIV_N, 1.500000f),     // 索引26
    Ret(16),                                        // 索引27
    Threshold(FeatureType::K_VALUE, 6144.0f),       // 索引28
    Ret(8),                                         // 索引29
    Ret(8),                                         // 索引30
    // ====================== 层级5 (索引31-62) ======================
    Threshold(FeatureType::M_VALUE, 192.0f),         // 索引31
    Threshold(FeatureType::N_VALUE, 1536.0f),        // 索引32
    Threshold(FeatureType::N_VALUE, 768.0f),         // 索引33
    Ret(16),                                         // 索引34
    Threshold(FeatureType::MN_DIV_K, 384.000000f),   // 索引35
    Threshold(FeatureType::WORLD_MUL_K, 12288.0f),   // 索引36
    Threshold(FeatureType::MN_DIV_K, 384.000000f),   // 索引37
    Threshold(FeatureType::MN_DIV_K, 6144.000000f),  // 索引38
    Placeholder(),                                   // 索引39: 占位
    Placeholder(),                                   // 索引40: 占位
    Placeholder(),                                   // 索引41: 占位
    Placeholder(),                                   // 索引42: 占位
    Placeholder(),                                   // 索引43: 占位
    Placeholder(),                                   // 索引44: 占位
    Placeholder(),                                   // 索引45: 占位
    Placeholder(),                                   // 索引46: 占位
    Threshold(FeatureType::MN_DIV_K, 192.000000f),   // 索引47
    Threshold(FeatureType::MN_DIV_K, 768.000000f),   // 索引48
    Threshold(FeatureType::M_DIV_N, 0.375000f),      // 索引49
    Threshold(FeatureType::MN_DIV_K, 12288.000000f), // 索引50
    Threshold(FeatureType::N_VALUE, 6144.0f),        // 索引51
    Ret(8),                                          // 索引52
    Threshold(FeatureType::M_VALUE, 3072.0f),        // 索引53
    Ret(8),                                          // 索引54
    Placeholder(),                                   // 索引55: 占位
    Placeholder(),                                   // 索引56: 占位
    Ret(8),                                          // 索引57
    Threshold(FeatureType::M_VALUE, 6144.0f),        // 索引58
    Placeholder(),                                   // 索引59: 占位
    Placeholder(),                                   // 索引60: 占位
    Placeholder(),                                   // 索引61: 占位
    Placeholder(),                                   // 索引62: 占位
    // ====================== 层级6 (索引63-126) ======================
    Ret(4),        // 索引63
    Ret(8),        // 索引64
    Ret(16),       // 索引65
    Ret(4),        // 索引66
    Ret(8),        // 索引67
    Ret(4),        // 索引68
    Placeholder(), // 索引69: 占位
    Placeholder(), // 索引70: 占位
    Ret(4),        // 索引71
    Ret(4),        // 索引72
    Ret(4),        // 索引73
    Ret(8),        // 索引74
    Ret(4),        // 索引75
    Ret(4),        // 索引76
    Ret(16),       // 索引77
    Ret(4),        // 索引78
    Placeholder(), // 索引79: 占位
    Placeholder(), // 索引80: 占位
    Placeholder(), // 索引81: 占位
    Placeholder(), // 索引82: 占位
    Placeholder(), // 索引83: 占位
    Placeholder(), // 索引84: 占位
    Placeholder(), // 索引85: 占位
    Placeholder(), // 索引86: 占位
    Placeholder(), // 索引87: 占位
    Placeholder(), // 索引88: 占位
    Placeholder(), // 索引89: 占位
    Placeholder(), // 索引90: 占位
    Placeholder(), // 索引91: 占位
    Placeholder(), // 索引92: 占位
    Placeholder(), // 索引93: 占位
    Placeholder(), // 索引94: 占位
    Ret(4),        // 索引95
    Ret(8),        // 索引96
    Ret(4),        // 索引97
    Ret(8),        // 索引98
    Ret(4),        // 索引99
    Ret(4),        // 索引100
    Ret(8),        // 索引101
    Ret(4),        // 索引102
    Ret(4),        // 索引103
    Ret(8),        // 索引104
    Placeholder(), // 索引105: 占位
    Placeholder(), // 索引106: 占位
    Ret(8),        // 索引107
    Ret(4),        // 索引108
    Placeholder(), // 索引109: 占位
    Placeholder(), // 索引110: 占位
    Placeholder(), // 索引111: 占位
    Placeholder(), // 索引112: 占位
    Placeholder(), // 索引113: 占位
    Placeholder(), // 索引114: 占位
    Placeholder(), // 索引115: 占位
    Placeholder(), // 索引116: 占位
    Ret(8),        // 索引117
    Ret(4),        // 索引118
    Placeholder(), // 索引119: 占位
    Placeholder(), // 索引120: 占位
    Placeholder(), // 索引121: 占位
    Placeholder(), // 索引122: 占位
    Placeholder(), // 索引123: 占位
    Placeholder(), // 索引124: 占位
    Placeholder(), // 索引125: 占位
    Placeholder(), // 索引126: 占位
};

inline int GetOptimalM0(int m, int k, int n)
{
    return TraverseDecisionTree(m0Rule, m, k, n, RANKSIZE_EIGHT, DEFAULT_M0);
}

inline int GetOptimalSwizzlCount(int m, int k, int n)
{
    return TraverseDecisionTree(swizzlcountRule, m, k, n, RANKSIZE_EIGHT, DEFAULT_SWIZZLCOUNT);
}

inline int GetOptimalSwizzlDirect(int m, int k, int n)
{
    return TraverseDecisionTree(swizzldirectRule, m, k, n, RANKSIZE_EIGHT, DEFAULT_SWIZZLDIRECT);
}

inline int GetOptimalPValue(int m, int k, int n)
{
    return TraverseDecisionTree(pvalueRule, m, k, n, RANKSIZE_EIGHT, DEFAULT_PVALUE);
}

inline int GetOptimalUbmovenum(int m, int k, int n)
{
    return TraverseDecisionTree(ubmovenumRule, m, k, n, RANKSIZE_EIGHT, DEFAULT_UBMOVENUM);
}

} // namespace Tiling_Small_M::Tiling_Rank8_A2

#endif
