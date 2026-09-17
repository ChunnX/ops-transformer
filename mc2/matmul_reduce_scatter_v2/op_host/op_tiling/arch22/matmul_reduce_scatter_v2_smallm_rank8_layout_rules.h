/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_REDUCE_SCATTER_V2_SMALLM_RANK8_LAYOUT_RULES_H
#define MATMUL_REDUCE_SCATTER_V2_SMALLM_RANK8_LAYOUT_RULES_H

#include "matmul_reduce_scatter_v2_smallm_decision_tree.h"

namespace Tiling_Small_M::Tiling_Rank8_A2 {

// m0优化参数的决策树规则（使用特征：）
const DecisionNode m0Rule[] = {
    // ====================== 层级0 (索引0-0) ======================
    Ret(128), // 索引0
};

// swizzlCount优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode swizzlcountRule[] = {
    // ====================== 层级0 (索引0-0) ======================
    Threshold(FeatureType::M_VALUE, 1536.0f), // 索引0
    // ====================== 层级1 (索引1-2) ======================
    Threshold(FeatureType::M_VALUE, 768.0f),  // 索引1
    Threshold(FeatureType::M_VALUE, 3072.0f), // 索引2
    // ====================== 层级2 (索引3-6) ======================
    Threshold(FeatureType::M_VALUE, 384.0f),        // 索引3
    Threshold(FeatureType::MN_DIV_K, 1536.000000f), // 索引4
    Threshold(FeatureType::N_VALUE, 3072.0f),       // 索引5
    Threshold(FeatureType::M_VALUE, 6144.0f),       // 索引6
    // ====================== 层级3 (索引7-14) ======================
    Threshold(FeatureType::M_VALUE, 192.0f),        // 索引7
    Threshold(FeatureType::MN_DIV_K, 1536.000000f), // 索引8
    Threshold(FeatureType::WORLD_MUL_K, 6144.0f),   // 索引9
    Threshold(FeatureType::M_MUL_N, 3145728.0f),    // 索引10
    Threshold(FeatureType::K_VALUE, 6144.0f),       // 索引11
    Threshold(FeatureType::K_VALUE, 1536.0f),       // 索引12
    Threshold(FeatureType::M_MUL_N, 6291456.0f),    // 索引13
    Threshold(FeatureType::WORLD_MUL_K, 49152.0f),  // 索引14
    // ====================== 层级4 (索引15-30) ======================
    Threshold(FeatureType::N_VALUE, 384.0f),         // 索引15
    Threshold(FeatureType::WORLD_MUL_K, 12288.0f),   // 索引16
    Threshold(FeatureType::MN_DIV_K, 192.000000f),   // 索引17
    Threshold(FeatureType::M_DIV_N, 0.093750f),      // 索引18
    Threshold(FeatureType::MN_DIV_K, 768.000000f),   // 索引19
    Threshold(FeatureType::M_DIV_N, 0.187500f),      // 索引20
    Threshold(FeatureType::M_MUL_N, 786432.0f),      // 索引21
    Threshold(FeatureType::MN_DIV_K, 24576.000000f), // 索引22
    Threshold(FeatureType::WORLD_MUL_K, 12288.0f),   // 索引23
    Threshold(FeatureType::M_DIV_N, 3.000000f),      // 索引24
    Threshold(FeatureType::N_VALUE, 6144.0f),        // 索引25
    Threshold(FeatureType::MN_DIV_K, 1536.000000f),  // 索引26
    Threshold(FeatureType::MN_DIV_K, 192.000000f),   // 索引27
    Threshold(FeatureType::M_DIV_N, 1.500000f),      // 索引28
    Threshold(FeatureType::M_DIV_N, 1.500000f),      // 索引29
    Threshold(FeatureType::M_MUL_N, 12582912.0f),    // 索引30
    // ====================== 层级5 (索引31-62) ======================
    Threshold(FeatureType::K_VALUE, 768.0f),         // 索引31
    Threshold(FeatureType::MN_DIV_K, 384.000000f),   // 索引32
    Threshold(FeatureType::MN_DIV_K, 6144.000000f),  // 索引33
    Threshold(FeatureType::MN_DIV_K, 384.000000f),   // 索引34
    Threshold(FeatureType::MN_DIV_K, 96.000000f),    // 索引35
    Threshold(FeatureType::MN_DIV_K, 384.000000f),   // 索引36
    Ret(4),                                          // 索引37
    Threshold(FeatureType::MN_DIV_K, 3072.000000f),  // 索引38
    Ret(6),                                          // 索引39
    Ret(3),                                          // 索引40
    Ret(3),                                          // 索引41
    Threshold(FeatureType::K_VALUE, 6144.0f),        // 索引42
    Ret(6),                                          // 索引43
    Threshold(FeatureType::WORLD_MUL_K, 3072.0f),    // 索引44
    Ret(3),                                          // 索引45
    Ret(8),                                          // 索引46
    Threshold(FeatureType::N_VALUE, 1536.0f),        // 索引47
    Threshold(FeatureType::M_DIV_N, 6.000000f),      // 索引48
    Ret(3),                                          // 索引49
    Ret(16),                                         // 索引50
    Ret(3),                                          // 索引51
    Threshold(FeatureType::MN_DIV_K, 24576.000000f), // 索引52
    Ret(3),                                          // 索引53
    Ret(6),                                          // 索引54
    Ret(3),                                          // 索引55
    Threshold(FeatureType::K_VALUE, 6144.0f),        // 索引56
    Threshold(FeatureType::WORLD_MUL_K, 12288.0f),   // 索引57
    Ret(3),                                          // 索引58
    Threshold(FeatureType::MN_DIV_K, 98304.000000f), // 索引59
    Threshold(FeatureType::MN_DIV_K, 98304.000000f), // 索引60
    Threshold(FeatureType::N_VALUE, 384.0f),         // 索引61
    Ret(6),                                          // 索引62
    // ====================== 层级6 (索引63-126) ======================
    Ret(6),        // 索引63
    Ret(3),        // 索引64
    Ret(1),        // 索引65
    Ret(1),        // 索引66
    Ret(2),        // 索引67
    Ret(3),        // 索引68
    Ret(2),        // 索引69
    Ret(3),        // 索引70
    Ret(3),        // 索引71
    Ret(4),        // 索引72
    Ret(6),        // 索引73
    Ret(6),        // 索引74
    Placeholder(), // 索引75: 占位
    Placeholder(), // 索引76: 占位
    Ret(4),        // 索引77
    Ret(6),        // 索引78
    Placeholder(), // 索引79: 占位
    Placeholder(), // 索引80: 占位
    Placeholder(), // 索引81: 占位
    Placeholder(), // 索引82: 占位
    Placeholder(), // 索引83: 占位
    Placeholder(), // 索引84: 占位
    Ret(8),        // 索引85
    Ret(8),        // 索引86
    Placeholder(), // 索引87: 占位
    Placeholder(), // 索引88: 占位
    Ret(8),        // 索引89
    Ret(8),        // 索引90
    Placeholder(), // 索引91: 占位
    Placeholder(), // 索引92: 占位
    Placeholder(), // 索引93: 占位
    Placeholder(), // 索引94: 占位
    Ret(16),       // 索引95
    Ret(6),        // 索引96
    Ret(16),       // 索引97
    Ret(16),       // 索引98
    Placeholder(), // 索引99: 占位
    Placeholder(), // 索引100: 占位
    Placeholder(), // 索引101: 占位
    Placeholder(), // 索引102: 占位
    Placeholder(), // 索引103: 占位
    Placeholder(), // 索引104: 占位
    Ret(6),        // 索引105
    Ret(6),        // 索引106
    Placeholder(), // 索引107: 占位
    Placeholder(), // 索引108: 占位
    Placeholder(), // 索引109: 占位
    Placeholder(), // 索引110: 占位
    Placeholder(), // 索引111: 占位
    Placeholder(), // 索引112: 占位
    Ret(32),       // 索引113
    Ret(32),       // 索引114
    Ret(6),        // 索引115
    Ret(3),        // 索引116
    Placeholder(), // 索引117: 占位
    Placeholder(), // 索引118: 占位
    Ret(3),        // 索引119
    Ret(6),        // 索引120
    Ret(64),       // 索引121
    Ret(6),        // 索引122
    Ret(64),       // 索引123
    Ret(3),        // 索引124
    Placeholder(), // 索引125: 占位
    Placeholder(), // 索引126: 占位
};

// swizzlDirect优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode swizzldirectRule[] = {
    // ====================== 层级0 (索引0-0) ======================
    Threshold(FeatureType::M_MUL_N, 3145728.0f), // 索引0
    // ====================== 层级1 (索引1-2) ======================
    Threshold(FeatureType::N_VALUE, 384.0f),    // 索引1
    Threshold(FeatureType::M_DIV_N, 0.093750f), // 索引2
    // ====================== 层级2 (索引3-6) ======================
    Threshold(FeatureType::MN_DIV_K, 96.000000f),  // 索引3
    Threshold(FeatureType::K_VALUE, 3072.0f),      // 索引4
    Threshold(FeatureType::K_VALUE, 6144.0f),      // 索引5
    Threshold(FeatureType::WORLD_MUL_K, 49152.0f), // 索引6
    // ====================== 层级3 (索引7-14) ======================
    Threshold(FeatureType::MN_DIV_K, 48.000000f),   // 索引7
    Threshold(FeatureType::WORLD_MUL_K, 6144.0f),   // 索引8
    Threshold(FeatureType::N_VALUE, 768.0f),        // 索引9
    Threshold(FeatureType::MN_DIV_K, 96.000000f),   // 索引10
    Threshold(FeatureType::K_VALUE, 384.0f),        // 索引11
    Ret(1),                                         // 索引12
    Threshold(FeatureType::MN_DIV_K, 1536.000000f), // 索引13
    Threshold(FeatureType::M_MUL_N, 6291456.0f),    // 索引14
    // ====================== 层级4 (索引15-30) ======================
    Threshold(FeatureType::M_MUL_N, 196608.0f),     // 索引15
    Ret(1),                                         // 索引16
    Threshold(FeatureType::M_DIV_N, 24.000000f),    // 索引17
    Threshold(FeatureType::MN_DIV_K, 1536.000000f), // 索引18
    Threshold(FeatureType::K_VALUE, 768.0f),        // 索引19
    Threshold(FeatureType::M_MUL_N, 786432.0f),     // 索引20
    Threshold(FeatureType::M_VALUE, 384.0f),        // 索引21
    Threshold(FeatureType::M_DIV_N, 0.023438f),     // 索引22
    Ret(0),                                         // 索引23
    Ret(0),                                         // 索引24
    Placeholder(),                                  // 索引25: 占位
    Placeholder(),                                  // 索引26: 占位
    Ret(0),                                         // 索引27
    Threshold(FeatureType::M_DIV_N, 3.000000f),     // 索引28
    Ret(1),                                         // 索引29
    Ret(1),                                         // 索引30
    // ====================== 层级5 (索引31-62) ======================
    Threshold(FeatureType::MN_DIV_K, 24.000000f),   // 索引31
    Ret(0),                                         // 索引32
    Placeholder(),                                  // 索引33: 占位
    Placeholder(),                                  // 索引34: 占位
    Threshold(FeatureType::MN_DIV_K, 3072.000000f), // 索引35
    Ret(1),                                         // 索引36
    Threshold(FeatureType::K_VALUE, 1536.0f),       // 索引37
    Ret(1),                                         // 索引38
    Threshold(FeatureType::M_VALUE, 1536.0f),       // 索引39
    Threshold(FeatureType::MN_DIV_K, 48.000000f),   // 索引40
    Threshold(FeatureType::M_DIV_N, 0.187500f),     // 索引41
    Threshold(FeatureType::WORLD_MUL_K, 6144.0f),   // 索引42
    Threshold(FeatureType::N_VALUE, 3072.0f),       // 索引43
    Threshold(FeatureType::MN_DIV_K, 48.000000f),   // 索引44
    Ret(1),                                         // 索引45
    Threshold(FeatureType::M_VALUE, 768.0f),        // 索引46
    Placeholder(),                                  // 索引47: 占位
    Placeholder(),                                  // 索引48: 占位
    Placeholder(),                                  // 索引49: 占位
    Placeholder(),                                  // 索引50: 占位
    Placeholder(),                                  // 索引51: 占位
    Placeholder(),                                  // 索引52: 占位
    Placeholder(),                                  // 索引53: 占位
    Placeholder(),                                  // 索引54: 占位
    Placeholder(),                                  // 索引55: 占位
    Placeholder(),                                  // 索引56: 占位
    Threshold(FeatureType::M_DIV_N, 0.750000f),     // 索引57
    Threshold(FeatureType::WORLD_MUL_K, 24576.0f),  // 索引58
    Placeholder(),                                  // 索引59: 占位
    Placeholder(),                                  // 索引60: 占位
    Placeholder(),                                  // 索引61: 占位
    Placeholder(),                                  // 索引62: 占位
    // ====================== 层级6 (索引63-126) ======================
    Ret(1),        // 索引63
    Ret(1),        // 索引64
    Placeholder(), // 索引65: 占位
    Placeholder(), // 索引66: 占位
    Placeholder(), // 索引67: 占位
    Placeholder(), // 索引68: 占位
    Placeholder(), // 索引69: 占位
    Placeholder(), // 索引70: 占位
    Ret(1),        // 索引71
    Ret(0),        // 索引72
    Placeholder(), // 索引73: 占位
    Placeholder(), // 索引74: 占位
    Ret(0),        // 索引75
    Ret(0),        // 索引76
    Placeholder(), // 索引77: 占位
    Placeholder(), // 索引78: 占位
    Ret(1),        // 索引79
    Ret(0),        // 索引80
    Ret(1),        // 索引81
    Ret(0),        // 索引82
    Ret(0),        // 索引83
    Ret(0),        // 索引84
    Ret(0),        // 索引85
    Ret(1),        // 索引86
    Ret(0),        // 索引87
    Ret(1),        // 索引88
    Ret(0),        // 索引89
    Ret(1),        // 索引90
    Placeholder(), // 索引91: 占位
    Placeholder(), // 索引92: 占位
    Ret(0),        // 索引93
    Ret(0),        // 索引94
    Placeholder(), // 索引95: 占位
    Placeholder(), // 索引96: 占位
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
    Placeholder(), // 索引107: 占位
    Placeholder(), // 索引108: 占位
    Placeholder(), // 索引109: 占位
    Placeholder(), // 索引110: 占位
    Placeholder(), // 索引111: 占位
    Placeholder(), // 索引112: 占位
    Placeholder(), // 索引113: 占位
    Placeholder(), // 索引114: 占位
    Ret(1),        // 索引115
    Ret(1),        // 索引116
    Ret(0),        // 索引117
    Ret(1),        // 索引118
    Placeholder(), // 索引119: 占位
    Placeholder(), // 索引120: 占位
    Placeholder(), // 索引121: 占位
    Placeholder(), // 索引122: 占位
    Placeholder(), // 索引123: 占位
    Placeholder(), // 索引124: 占位
    Placeholder(), // 索引125: 占位
    Placeholder(), // 索引126: 占位
};

} // namespace Tiling_Small_M::Tiling_Rank8_A2

#endif
