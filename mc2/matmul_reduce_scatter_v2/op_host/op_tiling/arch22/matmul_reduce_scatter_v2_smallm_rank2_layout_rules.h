/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_REDUCE_SCATTER_V2_SMALLM_RANK2_LAYOUT_RULES_H
#define MATMUL_REDUCE_SCATTER_V2_SMALLM_RANK2_LAYOUT_RULES_H

#include "matmul_reduce_scatter_v2_smallm_decision_tree.h"

namespace Tiling_Small_M::Tiling_Rank2_A2 {

const DecisionNode m0Rule[] = {
    // --- 层级0 (索引0-0) ---
    Ret(128), // 索引0
};

// swizzlCount优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode swizzlcountRule[] = {
    // === 层级0 (索引0-0) ===
    Threshold(FeatureType::M_VALUE, 1536.0f), // 索引0
    // ----- 层级1 (索引1-2) -----
    Threshold(FeatureType::M_VALUE, 768.0f),  // 索引1
    Threshold(FeatureType::M_VALUE, 3072.0f), // 索引2
    // ~~~~~ 层级2 (索引3-6) ~~~~~
    Threshold(FeatureType::M_VALUE, 384.0f),         // 索引3
    Threshold(FeatureType::M_DIV_N, 1.500000f),      // 索引4
    Threshold(FeatureType::MN_DIV_K, 24576.000000f), // 索引5
    Threshold(FeatureType::M_VALUE, 6144.0f),        // 索引6
    // >>>>> 层级3 (索引7-14) >>>>>
    Threshold(FeatureType::M_VALUE, 192.0f),         // 索引7
    Threshold(FeatureType::MN_DIV_K, 48.000000f),    // 索引8
    Threshold(FeatureType::WORLD_MUL_K, 3072.0f),    // 索引9
    Threshold(FeatureType::MN_DIV_K, 768.000000f),   // 索引10
    Threshold(FeatureType::M_MUL_N, 12582912.0f),    // 索引11
    Threshold(FeatureType::MN_DIV_K, 49152.000000f), // 索引12
    Threshold(FeatureType::N_VALUE, 384.0f),         // 索引13
    Threshold(FeatureType::K_VALUE, 6144.0f),        // 索引14
    // :::::: 层级4 (索引15-30) ::::::
    Threshold(FeatureType::MN_DIV_K, 384.000000f),   // 索引15
    Threshold(FeatureType::M_DIV_N, 0.093750f),      // 索引16
    Threshold(FeatureType::MN_DIV_K, 24.000000f),    // 索引17
    Threshold(FeatureType::M_DIV_N, 0.375000f),      // 索引18
    Threshold(FeatureType::K_VALUE, 384.0f),         // 索引19
    Threshold(FeatureType::M_MUL_N, 6291456.0f),     // 索引20
    Threshold(FeatureType::MN_DIV_K, 384.000000f),   // 索引21
    Threshold(FeatureType::MN_DIV_K, 1536.000000f),  // 索引22
    Threshold(FeatureType::MN_DIV_K, 6144.000000f),  // 索引23
    Threshold(FeatureType::MN_DIV_K, 3072.000000f),  // 索引24
    Ret(6),                                          // 索引25
    Ret(6),                                          // 索引26
    Threshold(FeatureType::MN_DIV_K, 192.000000f),   // 索引27
    Threshold(FeatureType::M_DIV_N, 0.750000f),      // 索引28
    Threshold(FeatureType::MN_DIV_K, 49152.000000f), // 索引29
    Threshold(FeatureType::M_DIV_N, 24.000000f),     // 索引30
    // <<<<< 层级5 (索引31-62) <<<<<
    Threshold(FeatureType::MN_DIV_K, 48.000000f),    // 索引31
    Threshold(FeatureType::K_VALUE, 384.0f),         // 索引32
    Threshold(FeatureType::WORLD_MUL_K, 3072.0f),    // 索引33
    Threshold(FeatureType::WORLD_MUL_K, 6144.0f),    // 索引34
    Ret(3),                                          // 索引35
    Ret(6),                                          // 索引36
    Threshold(FeatureType::K_VALUE, 768.0f),         // 索引37
    Threshold(FeatureType::MN_DIV_K, 1536.000000f),  // 索引38
    Ret(8),                                          // 索引39
    Threshold(FeatureType::MN_DIV_K, 12288.000000f), // 索引40
    Ret(8),                                          // 索引41
    Threshold(FeatureType::MN_DIV_K, 1536.000000f),  // 索引42
    Threshold(FeatureType::WORLD_MUL_K, 3072.0f),    // 索引43
    Ret(6),                                          // 索引44
    Ret(8),                                          // 索引45
    Ret(3),                                          // 索引46
    Threshold(FeatureType::M_DIV_N, 3.000000f),      // 索引47
    Threshold(FeatureType::K_VALUE, 384.0f),         // 索引48
    Ret(3),                                          // 索引49
    Threshold(FeatureType::K_VALUE, 3072.0f),        // 索引50
    Placeholder(),                                   // 索引51: placeholder
    Placeholder(),                                   // 索引52: 占位符
    Placeholder(),                                   // 索引53: 空节点
    Placeholder(),                                   // 索引54: 未定义
    Ret(3),                                          // 索引55
    Threshold(FeatureType::MN_DIV_K, 768.000000f),   // 索引56
    Threshold(FeatureType::WORLD_MUL_K, 1536.0f),    // 索引57
    Threshold(FeatureType::MN_DIV_K, 49152.000000f), // 索引58
    Threshold(FeatureType::M_DIV_N, 1.500000f),      // 索引59
    Threshold(FeatureType::MN_DIV_K, 98304.000000f), // 索引60
    Threshold(FeatureType::M_DIV_N, 1.500000f),      // 索引61
    Ret(64),                                         // 索引62
    // ++++++ 层级6 (索引63-126) ++++++
    Ret(3),        // 索引63
    Ret(1),        // 索引64
    Ret(3),        // 索引65
    Ret(6),        // 索引66
    Ret(2),        // 索引67
    Ret(2),        // 索引68
    Ret(6),        // 索引69
    Ret(2),        // 索引70
    Placeholder(), // 索引71: placeholder
    Placeholder(), // 索引72: 占位符
    Placeholder(), // 索引73: 空节点
    Placeholder(), // 索引74: 未定义
    Ret(4),        // 索引75
    Ret(4),        // 索引76
    Ret(4),        // 索引77
    Ret(6),        // 索引78
    Placeholder(), // 索引79: placeholder
    Placeholder(), // 索引80: 占位符
    Ret(8),        // 索引81
    Ret(6),        // 索引82
    Placeholder(), // 索引83: 空节点
    Placeholder(), // 索引84: 未定义
    Ret(3),        // 索引85
    Ret(8),        // 索引86
    Ret(8),        // 索引87
    Ret(3),        // 索引88
    Placeholder(), // 索引89: 占位
    Placeholder(), // 索引90: 占位
    Placeholder(), // 索引91: 占位
    Placeholder(), // 索引92: 占位
    Placeholder(), // 索引93: 占位
    Placeholder(), // 索引94: 占位
    Ret(16),       // 索引95
    Ret(16),       // 索引96
    Ret(16),       // 索引97
    Ret(3),        // 索引98
    Placeholder(), // 索引99: placeholder
    Placeholder(), // 索引100: 占位符
    Ret(3),        // 索引101
    Ret(16),       // 索引102
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
    Ret(6),        // 索引113
    Ret(6),        // 索引114
    Ret(6),        // 索引115
    Ret(3),        // 索引116
    Ret(32),       // 索引117
    Ret(6),        // 索引118
    Ret(3),        // 索引119
    Ret(64),       // 索引120
    Ret(3),        // 索引121
    Ret(6),        // 索引122
    Ret(3),        // 索引123
    Ret(3),        // 索引124
    Placeholder(), // 索引125: placeholder
    Placeholder(), // 索引126: 占位符
};

// swizzlDirect优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode swizzldirectRule[] = {
    // -*- 层级0 (索引0-0) -*-
    Threshold(FeatureType::M_MUL_N, 25165824.0f), // 索引0
    // ====== 层级1 (索引1-2) ======
    Threshold(FeatureType::M_MUL_N, 786432.0f), // 索引1
    Threshold(FeatureType::N_VALUE, 6144.0f),   // 索引2
    // #### 层级2 (索引3-6) ####
    Threshold(FeatureType::M_DIV_N, 6.000000f), // 索引3
    Threshold(FeatureType::N_VALUE, 384.0f),    // 索引4
    Threshold(FeatureType::K_VALUE, 6144.0f),   // 索引5
    Ret(1),                                     // 索引6
    // *** 层级3 (索引7-14) ***
    Threshold(FeatureType::MN_DIV_K, 48.000000f), // 索引7
    Threshold(FeatureType::K_VALUE, 1536.0f),     // 索引8
    Threshold(FeatureType::M_VALUE, 6144.0f),     // 索引9
    Threshold(FeatureType::N_VALUE, 6144.0f),     // 索引10
    Threshold(FeatureType::WORLD_MUL_K, 1536.0f), // 索引11
    Ret(1),                                       // 索引12
    Placeholder(),                                // 索引13: placeholder
    Placeholder(),                                // 索引14: 占位符
    // >>> 层级4 (索引15-30) >>>
    Threshold(FeatureType::M_VALUE, 192.0f),      // 索引15
    Threshold(FeatureType::K_VALUE, 3072.0f),     // 索引16
    Ret(0),                                       // 索引17
    Threshold(FeatureType::K_VALUE, 3072.0f),     // 索引18
    Ret(1),                                       // 索引19
    Threshold(FeatureType::WORLD_MUL_K, 3072.0f), // 索引20
    Threshold(FeatureType::K_VALUE, 6144.0f),     // 索引21
    Threshold(FeatureType::K_VALUE, 6144.0f),     // 索引22
    Ret(1),                                       // 索引23
    Ret(0),                                       // 索引24
    Placeholder(),                                // 索引25: placeholder
    Placeholder(),                                // 索引26: 占位符
    Placeholder(),                                // 索引27: 空节点
    Placeholder(),                                // 索引28: 未定义
    Placeholder(),                                // 索引29: placeholder
    Placeholder(),                                // 索引30: 占位符
    // -*- -*- -*- 层级5 (索引31-62) -*- -*- -*-
    Ret(1),                                       // 索引31
    Threshold(FeatureType::M_VALUE, 384.0f),      // 索引32
    Threshold(FeatureType::M_DIV_N, 1.500000f),   // 索引33
    Threshold(FeatureType::M_DIV_N, 0.093750f),   // 索引34
    Placeholder(),                                // 索引35: 空节点
    Placeholder(),                                // 索引36: 未定义
    Ret(1),                                       // 索引37
    Ret(0),                                       // 索引38
    Placeholder(),                                // 索引39: placeholder
    Placeholder(),                                // 索引40: 占位符
    Ret(1),                                       // 索引41
    Threshold(FeatureType::WORLD_MUL_K, 6144.0f), // 索引42
    Threshold(FeatureType::M_DIV_N, 0.750000f),   // 索引43
    Threshold(FeatureType::M_VALUE, 6144.0f),     // 索引44
    Threshold(FeatureType::M_MUL_N, 1572864.0f),  // 索引45
    Ret(1),                                       // 索引46
    Placeholder(),                                // 索引47: 空节点
    Placeholder(),                                // 索引48: 未定义
    Placeholder(),                                // 索引49: placeholder
    Placeholder(),                                // 索引50: 占位符
    Placeholder(),                                // 索引51: 空节点
    Placeholder(),                                // 索引52: 未定义
    Placeholder(),                                // 索引53: placeholder
    Placeholder(),                                // 索引54: 占位符
    Placeholder(),                                // 索引55: 空节点
    Placeholder(),                                // 索引56: 未定义
    Placeholder(),                                // 索引57: placeholder
    Placeholder(),                                // 索引58: 占位符
    Placeholder(),                                // 索引59: 空节点
    Placeholder(),                                // 索引60: 未定义
    Placeholder(),                                // 索引61: placeholder
    Placeholder(),                                // 索引62: 占位符
    // [层级6] (索引63-126)
    Placeholder(), // 索引63: 占位
    Placeholder(), // 索引64: 占位
    Ret(1),        // 索引65
    Ret(1),        // 索引66
    Ret(1),        // 索引67
    Ret(1),        // 索引68
    Ret(0),        // 索引69
    Ret(1),        // 索引70
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
    Ret(0),        // 索引85
    Ret(1),        // 索引86
    Ret(0),        // 索引87
    Ret(0),        // 索引88
    Ret(0),        // 索引89
    Ret(1),        // 索引90
    Ret(1),        // 索引91
    Ret(0),        // 索引92
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

} // namespace Tiling_Small_M::Tiling_Rank2_A2

#endif
