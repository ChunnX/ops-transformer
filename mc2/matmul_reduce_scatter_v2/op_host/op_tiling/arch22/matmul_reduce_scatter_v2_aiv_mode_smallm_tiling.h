/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_reduce_scatter_v2_aiv_mode_smallm_tiling.h
 * \brief
 */

#ifndef DECISION_RULES_H
#define DECISION_RULES_H

#include <cstdint>

namespace Tiling_Small_M {
enum class FeatureType {
    M_VALUE = 0,       // m的数值大小
    K_VALUE = 1,       // k的数值大小
    N_VALUE = 2,       // n的数值大小
    M_DIV_N = 3,       // m/n的比值
    MN_DIV_K = 4,      // (m*n)/k的比值
    WORLD_MUL_K = 5,   // world_size*k的乘积
    M_MUL_N = 6,       // m*n的乘积
    RETURN_VALUE = -1, // 叶子节点返回值标记
    PLACEHOLDER = -2   // 空节点占位符
};

constexpr int FEATURE_COUNT = 7;
constexpr int MAX_TREE_DEPTH = 7;
constexpr int32_t RANKSIZE_TWO = 2;
constexpr int32_t RANKSIZE_FOUR = 4;
constexpr int32_t RANKSIZE_EIGHT = 8;

constexpr int32_t DEFAULT_M0 = 128;
constexpr int32_t DEFAULT_SWIZZLCOUNT = 1;
constexpr int32_t DEFAULT_SWIZZLDIRECT = 0;
constexpr int32_t DEFAULT_PVALUE = 2;
constexpr int32_t DEFAULT_UBMOVENUM = 4;
constexpr int32_t DEFAULT_COMMNPUSPLIT = 1;
constexpr int32_t DEFAULT_COMMDATASPLIT = 16;


union NodeValue {
    float threshold;  // 内部节点的阈值
    int return_value; // 叶子节点的返回值
};

struct DecisionNode {
    FeatureType feature; // 特征类型
    NodeValue value;     // 阈值或返回值（通过union区分）
};

inline void PrecomputeFeatures(float features[FEATURE_COUNT], int m, int k, int n, int world_size)
{
    features[static_cast<int>(FeatureType::M_VALUE)] = static_cast<float>(m);
    features[static_cast<int>(FeatureType::K_VALUE)] = static_cast<float>(k);
    features[static_cast<int>(FeatureType::N_VALUE)] = static_cast<float>(n);
    features[static_cast<int>(FeatureType::M_DIV_N)] = (n != 0) ? static_cast<float>(m) / n : 0.0f;
    features[static_cast<int>(FeatureType::MN_DIV_K)] = (k != 0) ? static_cast<float>(m * n) / k : 0.0f;
    features[static_cast<int>(FeatureType::WORLD_MUL_K)] = static_cast<float>(world_size * k);
    features[static_cast<int>(FeatureType::M_MUL_N)] = static_cast<float>(m * n);
}

template <size_t N>
inline int TraverseDecisionTree(const DecisionNode (&decision_rules)[N], int m, int k, int n, int world_size,
                                int default_value)
{
    // 预计算所有特征值
    float features[FEATURE_COUNT];
    PrecomputeFeatures(features, m, k, n, world_size);

    int node_idx = 0; // 从根节点开始
    constexpr int SUB_TREE_SCALE = 2;
    constexpr int LEFT_NODE_OFFSET = 1;
    constexpr int RIGHT_NODE_OFFSET = 2;

    for (int i = 0; i < MAX_TREE_DEPTH; i++) {
        const DecisionNode &node = decision_rules[node_idx];

        // 走到空占位符，理论不应发生，返回默认值
        if (node.feature == FeatureType::PLACEHOLDER) {
            return default_value;
        }

        // 如果是叶子节点，返回值
        if (node.feature == FeatureType::RETURN_VALUE) {
            return node.value.return_value;
        }

        float feature_value = features[static_cast<int>(node.feature)];

        if (feature_value <= node.value.threshold) {
            node_idx = SUB_TREE_SCALE * node_idx + LEFT_NODE_OFFSET; // 左子节点
        } else {
            node_idx = SUB_TREE_SCALE * node_idx + RIGHT_NODE_OFFSET; // 右子节点
        }
    }
    return default_value;
}


namespace Tiling_Rank2_A2 {
const DecisionNode m0Rule[] = {
    // --- 层级0 (索引0-0) ---
    {FeatureType::RETURN_VALUE, {.return_value = 128}}, // 索引0
};

// [R2]  swizzlCount优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode swizzlcountRule[] = {
    // === 层级0 (索引0-0) ===
    {FeatureType::M_VALUE, {.threshold = 1536.0f}}, // 索引0
    // ----- 层级1 (索引1-2) -----
    {FeatureType::M_VALUE, {.threshold = 768.0f}},  // 索引1
    {FeatureType::M_VALUE, {.threshold = 3072.0f}}, // 索引2
    // ~~~~~ 层级2 (索引3-6) ~~~~~
    {FeatureType::M_VALUE, {.threshold = 384.0f}},         // 索引3
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},      // 索引4
    {FeatureType::MN_DIV_K, {.threshold = 24576.000000f}}, // 索引5
    {FeatureType::M_VALUE, {.threshold = 6144.0f}},        // 索引6
    // >>>>> 层级3 (索引7-14) >>>>>
    {FeatureType::M_VALUE, {.threshold = 192.0f}},         // 索引7
    {FeatureType::MN_DIV_K, {.threshold = 48.000000f}},    // 索引8
    {FeatureType::WORLD_MUL_K, {.threshold = 3072.0f}},    // 索引9
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},   // 索引10
    {FeatureType::M_MUL_N, {.threshold = 12582912.0f}},    // 索引11
    {FeatureType::MN_DIV_K, {.threshold = 49152.000000f}}, // 索引12
    {FeatureType::N_VALUE, {.threshold = 384.0f}},         // 索引13
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},        // 索引14
    // :::::: 层级4 (索引15-30) ::::::
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引15
    {FeatureType::M_DIV_N, {.threshold = 0.093750f}},      // 索引16
    {FeatureType::MN_DIV_K, {.threshold = 24.000000f}},    // 索引17
    {FeatureType::M_DIV_N, {.threshold = 0.375000f}},      // 索引18
    {FeatureType::K_VALUE, {.threshold = 384.0f}},         // 索引19
    {FeatureType::M_MUL_N, {.threshold = 6291456.0f}},     // 索引20
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引21
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}},  // 索引22
    {FeatureType::MN_DIV_K, {.threshold = 6144.000000f}},  // 索引23
    {FeatureType::MN_DIV_K, {.threshold = 3072.000000f}},  // 索引24
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引25
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引26
    {FeatureType::MN_DIV_K, {.threshold = 192.000000f}},   // 索引27
    {FeatureType::M_DIV_N, {.threshold = 0.750000f}},      // 索引28
    {FeatureType::MN_DIV_K, {.threshold = 49152.000000f}}, // 索引29
    {FeatureType::M_DIV_N, {.threshold = 24.000000f}},     // 索引30
    // <<<<< 层级5 (索引31-62) <<<<<
    {FeatureType::MN_DIV_K, {.threshold = 48.000000f}},    // 索引31
    {FeatureType::K_VALUE, {.threshold = 384.0f}},         // 索引32
    {FeatureType::WORLD_MUL_K, {.threshold = 3072.0f}},    // 索引33
    {FeatureType::WORLD_MUL_K, {.threshold = 6144.0f}},    // 索引34
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引35
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引36
    {FeatureType::K_VALUE, {.threshold = 768.0f}},         // 索引37
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}},  // 索引38
    {FeatureType::RETURN_VALUE, {.return_value = 8}},      // 索引39
    {FeatureType::MN_DIV_K, {.threshold = 12288.000000f}}, // 索引40
    {FeatureType::RETURN_VALUE, {.return_value = 8}},      // 索引41
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}},  // 索引42
    {FeatureType::WORLD_MUL_K, {.threshold = 3072.0f}},    // 索引43
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引44
    {FeatureType::RETURN_VALUE, {.return_value = 8}},      // 索引45
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引46
    {FeatureType::M_DIV_N, {.threshold = 3.000000f}},      // 索引47
    {FeatureType::K_VALUE, {.threshold = 384.0f}},         // 索引48
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引49
    {FeatureType::K_VALUE, {.threshold = 3072.0f}},        // 索引50
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx51: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx52: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx53: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx54: skip
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引55
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},   // 索引56
    {FeatureType::WORLD_MUL_K, {.threshold = 1536.0f}},    // 索引57
    {FeatureType::MN_DIV_K, {.threshold = 49152.000000f}}, // 索引58
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},      // 索引59
    {FeatureType::MN_DIV_K, {.threshold = 98304.000000f}}, // 索引60
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},      // 索引61
    {FeatureType::RETURN_VALUE, {.return_value = 64}},     // 索引62
    // ++++++ 层级6 (索引63-126) ++++++
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引63
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引64
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引65
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引66
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引67
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引68
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引69
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引70
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx71: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx72: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx73: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx74: skip
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引75
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引76
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引77
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引78
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx79: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx80: void
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引81
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引82
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx83: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx84: skip
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引85
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引86
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引87
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引88
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx89: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx90: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx91: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx92: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx93: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx94: void
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引95
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引96
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引97
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引98
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx99: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx100: skip
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引101
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引102
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx103: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx104: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx105: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx106: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx107: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx108: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx109: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx110: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx111: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx112: void
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引113
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引114
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引115
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引116
    {FeatureType::RETURN_VALUE, {.return_value = 32}}, // 索引117
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引118
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引119
    {FeatureType::RETURN_VALUE, {.return_value = 64}}, // 索引120
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引121
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引122
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引123
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引124
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx125: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx126: skip
};

// [R2]  swizzlDirect优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode swizzldirectRule[] = {
    // -*- 层级0 (索引0-0) -*-
    {FeatureType::M_MUL_N, {.threshold = 25165824.0f}}, // 索引0
    // ====== 层级1 (索引1-2) ======
    {FeatureType::M_MUL_N, {.threshold = 786432.0f}}, // 索引1
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},   // 索引2
    // #### 层级2 (索引3-6) ####
    {FeatureType::M_DIV_N, {.threshold = 6.000000f}}, // 索引3
    {FeatureType::N_VALUE, {.threshold = 384.0f}},    // 索引4
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},   // 索引5
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引6
    // *** 层级3 (索引7-14) ***
    {FeatureType::MN_DIV_K, {.threshold = 48.000000f}}, // 索引7
    {FeatureType::K_VALUE, {.threshold = 1536.0f}},     // 索引8
    {FeatureType::M_VALUE, {.threshold = 6144.0f}},     // 索引9
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},     // 索引10
    {FeatureType::WORLD_MUL_K, {.threshold = 1536.0f}}, // 索引11
    {FeatureType::RETURN_VALUE, {.return_value = 1}},   // 索引12
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx13: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx14: void
    // >>> 层级4 (索引15-30) >>>
    {FeatureType::M_VALUE, {.threshold = 192.0f}},      // 索引15
    {FeatureType::K_VALUE, {.threshold = 3072.0f}},     // 索引16
    {FeatureType::RETURN_VALUE, {.return_value = 0}},   // 索引17
    {FeatureType::K_VALUE, {.threshold = 3072.0f}},     // 索引18
    {FeatureType::RETURN_VALUE, {.return_value = 1}},   // 索引19
    {FeatureType::WORLD_MUL_K, {.threshold = 3072.0f}}, // 索引20
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},     // 索引21
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},     // 索引22
    {FeatureType::RETURN_VALUE, {.return_value = 1}},   // 索引23
    {FeatureType::RETURN_VALUE, {.return_value = 0}},   // 索引24
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx25: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx26: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx27: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx28: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx29: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx30: skip
    // -*- -*- -*- 层级5 (索引31-62) -*- -*- -*-
    {FeatureType::RETURN_VALUE, {.return_value = 1}},   // 索引31
    {FeatureType::M_VALUE, {.threshold = 384.0f}},      // 索引32
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},   // 索引33
    {FeatureType::M_DIV_N, {.threshold = 0.093750f}},   // 索引34
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx35: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx36: void
    {FeatureType::RETURN_VALUE, {.return_value = 1}},   // 索引37
    {FeatureType::RETURN_VALUE, {.return_value = 0}},   // 索引38
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx39: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx40: skip
    {FeatureType::RETURN_VALUE, {.return_value = 1}},   // 索引41
    {FeatureType::WORLD_MUL_K, {.threshold = 6144.0f}}, // 索引42
    {FeatureType::M_DIV_N, {.threshold = 0.750000f}},   // 索引43
    {FeatureType::M_VALUE, {.threshold = 6144.0f}},     // 索引44
    {FeatureType::M_MUL_N, {.threshold = 1572864.0f}},  // 索引45
    {FeatureType::RETURN_VALUE, {.return_value = 1}},   // 索引46
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx47: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx48: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx49: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx50: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx51: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx52: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx53: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx54: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx55: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx56: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx57: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx58: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx59: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx60: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx61: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx62: skip
    // [层级6] (索引63-126)
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx63: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx64: void
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引65
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引66
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引67
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引68
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引69
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引70
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx71: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx72: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx73: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx74: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx75: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx76: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx77: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx78: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx79: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx80: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx81: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx82: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx83: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx84: skip
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引85
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引86
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引87
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引88
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引89
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引90
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引91
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引92
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx93: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx94: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx95: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx96: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx97: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx98: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx99: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx100: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx101: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx102: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx103: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx104: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx105: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx106: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx107: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx108: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx109: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx110: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx111: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx112: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx113: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx114: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx115: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx116: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx117: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx118: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx119: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx120: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx121: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx122: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx123: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx124: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx125: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx126: void
};

// [R2]  pValue优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, mn_div_k）
const DecisionNode pvalueRule[] = {
    // | 层级0 (索引0-0) |
    {FeatureType::M_DIV_N, {.threshold = 12.000000f}}, // 索引0
    // || 层级1 (索引1-2) ||
    {FeatureType::M_VALUE, {.threshold = 6144.0f}}, // 索引1
    {FeatureType::K_VALUE, {.threshold = 6144.0f}}, // 索引2
    // ||| 层级2 (索引3-6) |||
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}},  // 索引3
    {FeatureType::MN_DIV_K, {.threshold = 98304.000000f}}, // 索引4
    {FeatureType::M_MUL_N, {.threshold = 1572864.0f}},     // 索引5
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引6
    // |||| 层级3 (索引7-14) ||||
    {FeatureType::M_VALUE, {.threshold = 3072.0f}},    // 索引7
    {FeatureType::M_MUL_N, {.threshold = 6291456.0f}}, // 索引8
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},    // 索引9
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},    // 索引10
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引11
    {FeatureType::N_VALUE, {.threshold = 384.0f}},     // 索引12
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引13
    {FeatureType::RETURN_VALUE, {.return_value = 7}},  // 索引14
    // ||||| 层级4 (索引15-30) |||||
    {FeatureType::M_MUL_N, {.threshold = 6291456.0f}},    // 索引15
    {FeatureType::K_VALUE, {.threshold = 3072.0f}},       // 索引16
    {FeatureType::MN_DIV_K, {.threshold = 6144.000000f}}, // 索引17
    {FeatureType::M_VALUE, {.threshold = 3072.0f}},       // 索引18
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},     // 索引19
    {FeatureType::RETURN_VALUE, {.return_value = 10}},    // 索引20
    {FeatureType::RETURN_VALUE, {.return_value = 20}},    // 索引21
    {FeatureType::RETURN_VALUE, {.return_value = 20}},    // 索引22
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx23: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx24: skip
    {FeatureType::RETURN_VALUE, {.return_value = 4}},     // 索引25
    {FeatureType::RETURN_VALUE, {.return_value = 4}},     // 索引26
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx27: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx28: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx29: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx30: skip
    // ====================== 层级5 (索引31-62) ======================
    {FeatureType::RETURN_VALUE, {.return_value = 1}},      // 索引31
    {FeatureType::RETURN_VALUE, {.return_value = 2}},      // 索引32
    {FeatureType::RETURN_VALUE, {.return_value = 2}},      // 索引33
    {FeatureType::RETURN_VALUE, {.return_value = 2}},      // 索引34
    {FeatureType::M_MUL_N, {.threshold = 786432.0f}},      // 索引35
    {FeatureType::M_MUL_N, {.threshold = 3145728.0f}},     // 索引36
    {FeatureType::MN_DIV_K, {.threshold = 24576.000000f}}, // 索引37
    {FeatureType::MN_DIV_K, {.threshold = 49152.000000f}}, // 索引38
    {FeatureType::RETURN_VALUE, {.return_value = 10}},     // 索引39
    {FeatureType::MN_DIV_K, {.threshold = 24576.000000f}}, // 索引40
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx41: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx42: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx43: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx44: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx45: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx46: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx47: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx48: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx49: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx50: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx51: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx52: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx53: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx54: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx55: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx56: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx57: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx58: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx59: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx60: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx61: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx62: void
    // <层级6> (索引63-126)
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx63: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx64: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx65: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx66: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx67: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx68: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx69: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx70: void
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引71
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引72
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引73
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引74
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引75
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引76
    {FeatureType::RETURN_VALUE, {.return_value = 5}},  // 索引77
    {FeatureType::RETURN_VALUE, {.return_value = 10}}, // 索引78
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx79: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx80: skip
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引81
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引82
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx83: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx84: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx85: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx86: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx87: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx88: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx89: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx90: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx91: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx92: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx93: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx94: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx95: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx96: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx97: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx98: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx99: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx100: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx101: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx102: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx103: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx104: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx105: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx106: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx107: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx108: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx109: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx110: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx111: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx112: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx113: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx114: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx115: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx116: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx117: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx118: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx119: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx120: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx121: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx122: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx123: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx124: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx125: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx126: skip
};

// [R2]  ubMoveNum优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode ubmovenumRule[] = {
    // (层级0) [索引0-0]
    {FeatureType::M_MUL_N, {.threshold = 98304.0f}}, // 索引0
    // (层级1) [索引1-2]
    {FeatureType::MN_DIV_K, {.threshold = 48.000000f}}, // 索引1
    {FeatureType::K_VALUE, {.threshold = 3072.0f}},     // 索引2
    // (层级2) [索引3-6]
    {FeatureType::RETURN_VALUE, {.return_value = 16}},  // 索引3
    {FeatureType::MN_DIV_K, {.threshold = 96.000000f}}, // 索引4
    {FeatureType::M_MUL_N, {.threshold = 12582912.0f}}, // 索引5
    {FeatureType::MN_DIV_K, {.threshold = 96.000000f}}, // 索引6
    // (层级3) [索引7-14]
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx7: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx8: void
    {FeatureType::M_VALUE, {.threshold = 192.0f}},       // 索引9
    {FeatureType::RETURN_VALUE, {.return_value = 16}},   // 索引10
    {FeatureType::MN_DIV_K, {.threshold = 192.000000f}}, // 索引11
    {FeatureType::WORLD_MUL_K, {.threshold = 768.0f}},   // 索引12
    {FeatureType::M_VALUE, {.threshold = 192.0f}},       // 索引13
    {FeatureType::M_VALUE, {.threshold = 6144.0f}},      // 索引14
    // 【层级4】(索引15-30)
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx15: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx16: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx17: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx18: void
    {FeatureType::RETURN_VALUE, {.return_value = 8}},    // 索引19
    {FeatureType::RETURN_VALUE, {.return_value = 16}},   // 索引20
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx21: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx22: skip
    {FeatureType::M_DIV_N, {.threshold = 0.187500f}},    // 索引23
    {FeatureType::M_MUL_N, {.threshold = 196608.0f}},    // 索引24
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},    // 索引25
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},      // 索引26
    {FeatureType::M_MUL_N, {.threshold = 393216.0f}},    // 索引27
    {FeatureType::M_MUL_N, {.threshold = 196608.0f}},    // 索引28
    {FeatureType::WORLD_MUL_K, {.threshold = 12288.0f}}, // 索引29
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}}, // 索引30
    // 【层级5】(索引31-62)
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx31: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx32: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx33: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx34: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx35: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx36: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx37: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx38: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx39: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx40: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx41: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx42: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx43: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx44: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx45: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx46: skip
    {FeatureType::MN_DIV_K, {.threshold = 96.000000f}}, // 索引47
    {FeatureType::RETURN_VALUE, {.return_value = 8}},   // 索引48
    {FeatureType::M_VALUE, {.threshold = 384.0f}},      // 索引49
    {FeatureType::M_DIV_N, {.threshold = 0.023438f}},   // 索引50
    {FeatureType::M_MUL_N, {.threshold = 25165824.0f}}, // 索引51
    {FeatureType::RETURN_VALUE, {.return_value = 4}},   // 索引52
    {FeatureType::M_MUL_N, {.threshold = 25165824.0f}}, // 索引53
    {FeatureType::WORLD_MUL_K, {.threshold = 1536.0f}}, // 索引54
    {FeatureType::MN_DIV_K, {.threshold = 24.000000f}}, // 索引55
    {FeatureType::RETURN_VALUE, {.return_value = 16}},  // 索引56
    {FeatureType::RETURN_VALUE, {.return_value = 8}},   // 索引57
    {FeatureType::MN_DIV_K, {.threshold = 48.000000f}}, // 索引58
    {FeatureType::M_DIV_N, {.threshold = 0.093750f}},   // 索引59
    {FeatureType::M_VALUE, {.threshold = 1536.0f}},     // 索引60
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},     // 索引61
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},     // 索引62
    // 【层级6】(索引63-126)
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx63: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx64: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx65: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx66: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx67: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx68: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx69: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx70: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx71: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx72: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx73: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx74: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx75: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx76: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx77: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx78: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx79: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx80: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx81: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx82: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx83: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx84: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx85: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx86: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx87: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx88: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx89: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx90: skip
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx91: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx92: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx93: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx94: skip
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引95
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引96
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx97: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx98: void
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引99
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引100
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引101
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引102
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引103
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引104
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx105: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx106: skip
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引107
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引108
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引109
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引110
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引111
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引112
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx113: nil
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx114: void
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx115: blank
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx116: skip
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引117
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引118
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引119
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引120
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引121
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引122
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引123
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引124
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引125
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引126
};

inline int GetOptimalM0(int dimM, int dimK, int dimN)
{
    return TraverseDecisionTree(m0Rule, dimM, dimK, dimN, RANKSIZE_TWO, DEFAULT_M0);
}

inline int GetOptimalSwizzlCount(int dimM, int dimK, int dimN)
{
    return TraverseDecisionTree(swizzlcountRule, dimM, dimK, dimN, RANKSIZE_TWO, DEFAULT_SWIZZLCOUNT);
}

inline int GetOptimalSwizzlDirect(int dimM, int dimK, int dimN)
{
    return TraverseDecisionTree(swizzldirectRule, dimM, dimK, dimN, RANKSIZE_TWO, DEFAULT_SWIZZLDIRECT);
}

inline int GetOptimalPValue(int dimM, int dimK, int dimN)
{
    return TraverseDecisionTree(pvalueRule, dimM, dimK, dimN, RANKSIZE_TWO, DEFAULT_PVALUE);
}

inline int GetOptimalUbmovenum(int dimM, int dimK, int dimN)
{
    return TraverseDecisionTree(ubmovenumRule, dimM, dimK, dimN, RANKSIZE_TWO, DEFAULT_UBMOVENUM);
}

} // namespace Tiling_Rank2_A2


namespace Tiling_Rank4_A2 {
// [R4]  m0优化参数的决策树规则（使用特征：）
const DecisionNode m0Rule[] = {
    // >>>> 层级0 (索引0-0) >>>>
    {FeatureType::RETURN_VALUE, {.return_value = 128}}, // 索引0
};

// [R4]  swizzlCount优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode swizzlcountRule[] = {
    // >> 层级0 (索引0-0) >>>
    {FeatureType::M_VALUE, {.threshold = 3072.0f}}, // 索引0
    // > 层级1 (索引1-2) >
    {FeatureType::M_VALUE, {.threshold = 768.0f}},  // 索引1
    {FeatureType::M_VALUE, {.threshold = 6144.0f}}, // 索引2
    // << 层级2 (索引3-6) <<
    {FeatureType::M_VALUE, {.threshold = 384.0f}},         // 索引3
    {FeatureType::M_VALUE, {.threshold = 1536.0f}},        // 索引4
    {FeatureType::MN_DIV_K, {.threshold = 12288.000000f}}, // 索引5
    {FeatureType::N_VALUE, {.threshold = 3072.0f}},        // 索引6
    // <<< 层级3 (索引7-14) <<<
    {FeatureType::M_VALUE, {.threshold = 192.0f}},         // 索引7
    {FeatureType::WORLD_MUL_K, {.threshold = 3072.0f}},    // 索引8
    {FeatureType::M_MUL_N, {.threshold = 393216.0f}},      // 索引9
    {FeatureType::MN_DIV_K, {.threshold = 3072.000000f}},  // 索引10
    {FeatureType::N_VALUE, {.threshold = 384.0f}},         // 索引11
    {FeatureType::MN_DIV_K, {.threshold = 24576.000000f}}, // 索引12
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},        // 索引13
    {FeatureType::MN_DIV_K, {.threshold = 49152.000000f}}, // 索引14
    // <<<< 层级4 (索引15-30) <<<<
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},   // 索引15
    {FeatureType::M_DIV_N, {.threshold = 0.093750f}},      // 索引16
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引17
    {FeatureType::N_VALUE, {.threshold = 3072.0f}},        // 索引18
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},   // 索引19
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}},  // 索引20
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引21
    {FeatureType::MN_DIV_K, {.threshold = 12288.000000f}}, // 索引22
    {FeatureType::K_VALUE, {.threshold = 1536.0f}},        // 索引23
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},        // 索引24
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引25
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},      // 索引26
    {FeatureType::M_DIV_N, {.threshold = 24.000000f}},     // 索引27
    {FeatureType::N_VALUE, {.threshold = 768.0f}},         // 索引28
    {FeatureType::K_VALUE, {.threshold = 3072.0f}},        // 索引29
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},      // 索引30
    // ====================== 层级5 (索引31-62) ======================
    {FeatureType::M_DIV_N, {.threshold = 0.375000f}},      // 索引31
    {FeatureType::K_VALUE, {.threshold = 384.0f}},         // 索引32
    {FeatureType::WORLD_MUL_K, {.threshold = 24576.0f}},   // 索引33
    {FeatureType::M_MUL_N, {.threshold = 196608.0f}},      // 索引34
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引35
    {FeatureType::N_VALUE, {.threshold = 1536.0f}},        // 索引36
    {FeatureType::M_DIV_N, {.threshold = 0.375000f}},      // 索引37
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},   // 索引38
    {FeatureType::K_VALUE, {.threshold = 3072.0f}},        // 索引39
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引40
    {FeatureType::M_DIV_N, {.threshold = 0.187500f}},      // 索引41
    {FeatureType::MN_DIV_K, {.threshold = 3072.000000f}},  // 索引42
    {FeatureType::N_VALUE, {.threshold = 768.0f}},         // 索引43
    {FeatureType::M_DIV_N, {.threshold = 0.375000f}},      // 索引44
    {FeatureType::M_MUL_N, {.threshold = 1572864.0f}},     // 索引45
    {FeatureType::N_VALUE, {.threshold = 3072.0f}},        // 索引46
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引47
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引48
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},        // 索引49
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引50
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx51: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx52: none
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引53
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引54
    {FeatureType::M_MUL_N, {.threshold = 6291456.0f}},     // 索引55
    {FeatureType::K_VALUE, {.threshold = 1536.0f}},        // 索引56
    {FeatureType::RETURN_VALUE, {.return_value = 64}},     // 索引57
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引58
    {FeatureType::MN_DIV_K, {.threshold = 24576.000000f}}, // 索引59
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},      // 索引60
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引61
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引62
    // <<<<< 层级6 (索引63-126) <<<<<
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引63
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引64
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引65
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引66
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引67
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引68
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引69
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引70
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx71: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx72: gap
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引73
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引74
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引75
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引76
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引77
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引78
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引79
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引80
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx81: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx82: none
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引83
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引84
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引85
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引86
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引87
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引88
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引89
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引90
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引91
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引92
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引93
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引94
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx95: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx96: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx97: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx98: none
    {FeatureType::RETURN_VALUE, {.return_value = 32}}, // 索引99
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引100
    {FeatureType::RETURN_VALUE, {.return_value = 32}}, // 索引101
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引102
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx103: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx104: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx105: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx106: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx107: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx108: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx109: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx110: none
    {FeatureType::RETURN_VALUE, {.return_value = 64}}, // 索引111
    {FeatureType::RETURN_VALUE, {.return_value = 64}}, // 索引112
    {FeatureType::RETURN_VALUE, {.return_value = 64}}, // 索引113
    {FeatureType::RETURN_VALUE, {.return_value = 64}}, // 索引114
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx115: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx116: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx117: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx118: none
    {FeatureType::RETURN_VALUE, {.return_value = 64}}, // 索引119
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引120
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引121
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引122
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx123: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx124: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx125: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx126: none
};

// [R4]  swizzlDirect优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode swizzldirectRule[] = {
    // ## 层级0 (索引0-0) ##
    {FeatureType::M_MUL_N, {.threshold = 25165824.0f}}, // 索引0
    // ## 层级1 (索引1-2) ##
    {FeatureType::M_MUL_N, {.threshold = 786432.0f}}, // 索引1
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},   // 索引2
    // ## 层级2 (索引3-6) ##
    {FeatureType::N_VALUE, {.threshold = 768.0f}},         // 索引3
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},        // 索引4
    {FeatureType::MN_DIV_K, {.threshold = 12288.000000f}}, // 索引5
    {FeatureType::RETURN_VALUE, {.return_value = 1}},      // 索引6
    // ## 层级3 (索引7-14) ##
    {FeatureType::M_MUL_N, {.threshold = 49152.0f}},    // 索引7
    {FeatureType::N_VALUE, {.threshold = 1536.0f}},     // 索引8
    {FeatureType::WORLD_MUL_K, {.threshold = 6144.0f}}, // 索引9
    {FeatureType::M_VALUE, {.threshold = 384.0f}},      // 索引10
    {FeatureType::RETURN_VALUE, {.return_value = 1}},   // 索引11
    {FeatureType::WORLD_MUL_K, {.threshold = 6144.0f}}, // 索引12
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx13: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},    // idx14: gap
    // ## 层级4 (索引15-30) ##
    {FeatureType::RETURN_VALUE, {.return_value = 1}},    // 索引15
    {FeatureType::M_MUL_N, {.threshold = 98304.0f}},     // 索引16
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}}, // 索引17
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}}, // 索引18
    {FeatureType::M_DIV_N, {.threshold = 0.375000f}},    // 索引19
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},      // 索引20
    {FeatureType::WORLD_MUL_K, {.threshold = 6144.0f}},  // 索引21
    {FeatureType::K_VALUE, {.threshold = 768.0f}},       // 索引22
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx23: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx24: none
    {FeatureType::RETURN_VALUE, {.return_value = 1}},    // 索引25
    {FeatureType::RETURN_VALUE, {.return_value = 0}},    // 索引26
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx27: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx28: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx29: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx30: none
    // ## 层级5 (索引31-62) ##
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx31: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx32: gap
    {FeatureType::MN_DIV_K, {.threshold = 48.000000f}},  // 索引33
    {FeatureType::K_VALUE, {.threshold = 768.0f}},       // 索引34
    {FeatureType::WORLD_MUL_K, {.threshold = 1536.0f}},  // 索引35
    {FeatureType::WORLD_MUL_K, {.threshold = 1536.0f}},  // 索引36
    {FeatureType::N_VALUE, {.threshold = 3072.0f}},      // 索引37
    {FeatureType::RETURN_VALUE, {.return_value = 1}},    // 索引38
    {FeatureType::M_MUL_N, {.threshold = 1572864.0f}},   // 索引39
    {FeatureType::M_VALUE, {.threshold = 6144.0f}},      // 索引40
    {FeatureType::WORLD_MUL_K, {.threshold = 12288.0f}}, // 索引41
    {FeatureType::M_DIV_N, {.threshold = 0.750000f}},    // 索引42
    {FeatureType::M_MUL_N, {.threshold = 1572864.0f}},   // 索引43
    {FeatureType::WORLD_MUL_K, {.threshold = 24576.0f}}, // 索引44
    {FeatureType::M_MUL_N, {.threshold = 12582912.0f}},  // 索引45
    {FeatureType::RETURN_VALUE, {.return_value = 1}},    // 索引46
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx47: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx48: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx49: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx50: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx51: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx52: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx53: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx54: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx55: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx56: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx57: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx58: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx59: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx60: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx61: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx62: gap
    // ====================== 层级6 (索引63-126) ======================
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx63: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx64: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx65: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx66: gap
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引67
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引68
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引69
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引70
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引71
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引72
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引73
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引74
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引75
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引76
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx77: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx78: none
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引79
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引80
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引81
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引82
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引83
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引84
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引85
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引86
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引87
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引88
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引89
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引90
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引91
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引92
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx93: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx94: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx95: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx96: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx97: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx98: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx99: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx100: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx101: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx102: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx103: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx104: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx105: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx106: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx107: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx108: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx109: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx110: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx111: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx112: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx113: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx114: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx115: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx116: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx117: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx118: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx119: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx120: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx121: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx122: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx123: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx124: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx125: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx126: gap
};

// [R4]  pValue优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode pvalueRule[] = {
    // ====================== 层级0 (索引0-0) ======================
    {FeatureType::M_VALUE, {.threshold = 3072.0f}}, // 索引0
    // ====================== 层级1 (索引1-2) ======================
    {FeatureType::M_VALUE, {.threshold = 768.0f}},  // 索引1
    {FeatureType::M_VALUE, {.threshold = 6144.0f}}, // 索引2
    // ====================== 层级2 (索引3-6) ======================
    {FeatureType::MN_DIV_K, {.threshold = 3072.000000f}},  // 索引3
    {FeatureType::MN_DIV_K, {.threshold = 12288.000000f}}, // 索引4
    {FeatureType::K_VALUE, {.threshold = 768.0f}},         // 索引5
    {FeatureType::M_DIV_N, {.threshold = 12.000000f}},     // 索引6
    // ====================== 层级3 (索引7-14) ======================
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},   // 索引7
    {FeatureType::M_DIV_N, {.threshold = 0.093750f}},      // 索引8
    {FeatureType::M_MUL_N, {.threshold = 1572864.0f}},     // 索引9
    {FeatureType::M_VALUE, {.threshold = 1536.0f}},        // 索引10
    {FeatureType::N_VALUE, {.threshold = 1536.0f}},        // 索引11
    {FeatureType::M_MUL_N, {.threshold = 3145728.0f}},     // 索引12
    {FeatureType::MN_DIV_K, {.threshold = 98304.000000f}}, // 索引13
    {FeatureType::MN_DIV_K, {.threshold = 12288.000000f}}, // 索引14
    // ====================== 层级4 (索引15-30) ======================
    {FeatureType::WORLD_MUL_K, {.threshold = 1536.0f}},    // 索引15
    {FeatureType::M_DIV_N, {.threshold = 0.187500f}},      // 索引16
    {FeatureType::M_DIV_N, {.threshold = 0.023438f}},      // 索引17
    {FeatureType::RETURN_VALUE, {.return_value = 2}},      // 索引18
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},   // 索引19
    {FeatureType::M_VALUE, {.threshold = 1536.0f}},        // 索引20
    {FeatureType::RETURN_VALUE, {.return_value = 2}},      // 索引21
    {FeatureType::M_MUL_N, {.threshold = 6291456.0f}},     // 索引22
    {FeatureType::MN_DIV_K, {.threshold = 12288.000000f}}, // 索引23
    {FeatureType::N_VALUE, {.threshold = 3072.0f}},        // 索引24
    {FeatureType::RETURN_VALUE, {.return_value = 2}},      // 索引25
    {FeatureType::M_DIV_N, {.threshold = 0.750000f}},      // 索引26
    {FeatureType::WORLD_MUL_K, {.threshold = 1536.0f}},    // 索引27
    {FeatureType::RETURN_VALUE, {.return_value = 20}},     // 索引28
    {FeatureType::K_VALUE, {.threshold = 384.0f}},         // 索引29
    {FeatureType::RETURN_VALUE, {.return_value = 7}},      // 索引30
    // ====================== 层级5 (索引31-62) ======================
    {FeatureType::RETURN_VALUE, {.return_value = 1}},     // 索引31
    {FeatureType::RETURN_VALUE, {.return_value = 1}},     // 索引32
    {FeatureType::RETURN_VALUE, {.return_value = 1}},     // 索引33
    {FeatureType::K_VALUE, {.threshold = 768.0f}},        // 索引34
    {FeatureType::RETURN_VALUE, {.return_value = 2}},     // 索引35
    {FeatureType::M_MUL_N, {.threshold = 1572864.0f}},    // 索引36
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx37: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx38: none
    {FeatureType::RETURN_VALUE, {.return_value = 1}},     // 索引39
    {FeatureType::M_MUL_N, {.threshold = 786432.0f}},     // 索引40
    {FeatureType::MN_DIV_K, {.threshold = 6144.000000f}}, // 索引41
    {FeatureType::WORLD_MUL_K, {.threshold = 3072.0f}},   // 索引42
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx43: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx44: gap
    {FeatureType::RETURN_VALUE, {.return_value = 4}},     // 索引45
    {FeatureType::RETURN_VALUE, {.return_value = 5}},     // 索引46
    {FeatureType::M_MUL_N, {.threshold = 1572864.0f}},    // 索引47
    {FeatureType::RETURN_VALUE, {.return_value = 7}},     // 索引48
    {FeatureType::RETURN_VALUE, {.return_value = 8}},     // 索引49
    {FeatureType::RETURN_VALUE, {.return_value = 10}},    // 索引50
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx51: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx52: none
    {FeatureType::MN_DIV_K, {.threshold = 6144.000000f}}, // 索引53
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},       // 索引54
    {FeatureType::RETURN_VALUE, {.return_value = 8}},     // 索引55
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},     // 索引56
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx57: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx58: gap
    {FeatureType::RETURN_VALUE, {.return_value = 4}},     // 索引59
    {FeatureType::RETURN_VALUE, {.return_value = 4}},     // 索引60
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx61: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx62: none
    // ====================== 层级6 (索引63-126) ======================
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx63: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx64: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx65: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx66: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx67: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx68: gap
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引69
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引70
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx71: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx72: none
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引73
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引74
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx75: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx76: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx77: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx78: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx79: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx80: gap
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引81
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引82
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引83
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引84
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引85
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引86
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx87: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx88: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx89: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx90: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx91: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx92: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx93: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx94: gap
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引95
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引96
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx97: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx98: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx99: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx100: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx101: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx102: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx103: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx104: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx105: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx106: none
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引107
    {FeatureType::RETURN_VALUE, {.return_value = 5}},  // 索引108
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引109
    {FeatureType::RETURN_VALUE, {.return_value = 5}},  // 索引110
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx111: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx112: gap
    {FeatureType::RETURN_VALUE, {.return_value = 20}}, // 索引113
    {FeatureType::RETURN_VALUE, {.return_value = 10}}, // 索引114
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx115: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx116: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx117: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx118: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx119: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx120: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx121: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx122: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx123: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx124: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx125: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx126: gap
};

// [R4]  ubMoveNum优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode ubmovenumRule[] = {
    // ====================== 层级0 (索引0-0) ======================
    {FeatureType::M_VALUE, {.threshold = 384.0f}}, // 索引0
    // ====================== 层级1 (索引1-2) ======================
    {FeatureType::N_VALUE, {.threshold = 768.0f}},    // 索引1
    {FeatureType::M_MUL_N, {.threshold = 786432.0f}}, // 索引2
    // ====================== 层级2 (索引3-6) ======================
    {FeatureType::MN_DIV_K, {.threshold = 192.000000f}}, // 索引3
    {FeatureType::M_DIV_N, {.threshold = 0.046875f}},    // 索引4
    {FeatureType::WORLD_MUL_K, {.threshold = 1536.0f}},  // 索引5
    {FeatureType::N_VALUE, {.threshold = 3072.0f}},      // 索引6
    // ====================== 层级3 (索引7-14) ======================
    {FeatureType::K_VALUE, {.threshold = 384.0f}},       // 索引7
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}}, // 索引8
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}}, // 索引9
    {FeatureType::K_VALUE, {.threshold = 1536.0f}},      // 索引10
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}}, // 索引11
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},    // 索引12
    {FeatureType::M_MUL_N, {.threshold = 1572864.0f}},   // 索引13
    {FeatureType::WORLD_MUL_K, {.threshold = 3072.0f}},  // 索引14
    // ====================== 层级4 (索引15-30) ======================
    {FeatureType::RETURN_VALUE, {.return_value = 8}},     // 索引15
    {FeatureType::M_DIV_N, {.threshold = 0.750000f}},     // 索引16
    {FeatureType::RETURN_VALUE, {.return_value = 8}},     // 索引17
    {FeatureType::RETURN_VALUE, {.return_value = 16}},    // 索引18
    {FeatureType::MN_DIV_K, {.threshold = 96.000000f}},   // 索引19
    {FeatureType::MN_DIV_K, {.threshold = 6144.000000f}}, // 索引20
    {FeatureType::N_VALUE, {.threshold = 1536.0f}},       // 索引21
    {FeatureType::M_VALUE, {.threshold = 192.0f}},        // 索引22
    {FeatureType::RETURN_VALUE, {.return_value = 16}},    // 索引23
    {FeatureType::RETURN_VALUE, {.return_value = 8}},     // 索引24
    {FeatureType::WORLD_MUL_K, {.threshold = 6144.0f}},   // 索引25
    {FeatureType::RETURN_VALUE, {.return_value = 8}},     // 索引26
    {FeatureType::K_VALUE, {.threshold = 384.0f}},        // 索引27
    {FeatureType::WORLD_MUL_K, {.threshold = 24576.0f}},  // 索引28
    {FeatureType::M_MUL_N, {.threshold = 6291456.0f}},    // 索引29
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},  // 索引30
    // ====================== 层级5 (索引31-62) ======================
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx31: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx32: none
    {FeatureType::MN_DIV_K, {.threshold = 12.000000f}},  // 索引33
    {FeatureType::MN_DIV_K, {.threshold = 12.000000f}},  // 索引34
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx35: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx36: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx37: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx38: none
    {FeatureType::RETURN_VALUE, {.return_value = 8}},    // 索引39
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}}, // 索引40
    {FeatureType::M_DIV_N, {.threshold = 0.023438f}},    // 索引41
    {FeatureType::RETURN_VALUE, {.return_value = 16}},   // 索引42
    {FeatureType::RETURN_VALUE, {.return_value = 8}},    // 索引43
    {FeatureType::M_MUL_N, {.threshold = 786432.0f}},    // 索引44
    {FeatureType::RETURN_VALUE, {.return_value = 8}},    // 索引45
    {FeatureType::RETURN_VALUE, {.return_value = 8}},    // 索引46
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx47: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx48: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx49: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx50: none
    {FeatureType::M_DIV_N, {.threshold = 0.750000f}},    // 索引51
    {FeatureType::RETURN_VALUE, {.return_value = 8}},    // 索引52
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx53: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},     // idx54: gap
    {FeatureType::M_VALUE, {.threshold = 1536.0f}},      // 索引55
    {FeatureType::WORLD_MUL_K, {.threshold = 12288.0f}}, // 索引56
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},    // 索引57
    {FeatureType::N_VALUE, {.threshold = 1536.0f}},      // 索引58
    {FeatureType::K_VALUE, {.threshold = 384.0f}},       // 索引59
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},    // 索引60
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},      // 索引61
    {FeatureType::M_MUL_N, {.threshold = 6291456.0f}},   // 索引62
    // ====================== 层级6 (索引63-126) ======================
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx63: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx64: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx65: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx66: gap
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引67
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引68
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引69
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引70
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx71: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx72: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx73: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx74: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx75: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx76: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx77: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx78: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx79: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx80: none
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引81
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引82
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引83
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引84
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx85: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx86: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx87: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx88: none
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引89
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引90
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx91: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx92: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx93: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx94: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx95: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx96: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx97: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx98: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx99: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx100: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx101: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx102: none
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引103
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引104
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx105: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx106: gap
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx107: empty
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx108: none
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx109: unused
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx110: gap
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引111
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引112
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引113
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引114
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引115
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引116
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引117
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引118
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引119
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引120
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引121
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引122
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引123
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引124
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引125
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引126
};

inline int GetOptimalM0(int mVal, int kVal, int nVal)
{
    return TraverseDecisionTree(m0Rule, mVal, kVal, nVal, RANKSIZE_FOUR, DEFAULT_M0);
}

inline int GetOptimalSwizzlCount(int mVal, int kVal, int nVal)
{
    return TraverseDecisionTree(swizzlcountRule, mVal, kVal, nVal, RANKSIZE_FOUR, DEFAULT_SWIZZLCOUNT);
}

inline int GetOptimalSwizzlDirect(int mVal, int kVal, int nVal)
{
    return TraverseDecisionTree(swizzldirectRule, mVal, kVal, nVal, RANKSIZE_FOUR, DEFAULT_SWIZZLDIRECT);
}

inline int GetOptimalPValue(int mVal, int kVal, int nVal)
{
    return TraverseDecisionTree(pvalueRule, mVal, kVal, nVal, RANKSIZE_FOUR, DEFAULT_PVALUE);
}

inline int GetOptimalUbmovenum(int mVal, int kVal, int nVal)
{
    return TraverseDecisionTree(ubmovenumRule, mVal, kVal, nVal, RANKSIZE_FOUR, DEFAULT_UBMOVENUM);
}

} // namespace Tiling_Rank4_A2

namespace Tiling_Rank8_A2 {
// [R8]  m0优化参数的决策树规则（使用特征：）
const DecisionNode m0Rule[] = {
    // ====================== 层级0 (索引0-0) ======================
    {FeatureType::RETURN_VALUE, {.return_value = 128}}, // 索引0
};

// [R8]  swizzlCount优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode swizzlcountRule[] = {
    // ====================== 层级0 (索引0-0) ======================
    {FeatureType::M_VALUE, {.threshold = 1536.0f}}, // 索引0
    // ====================== 层级1 (索引1-2) ======================
    {FeatureType::M_VALUE, {.threshold = 768.0f}},  // 索引1
    {FeatureType::M_VALUE, {.threshold = 3072.0f}}, // 索引2
    // ====================== 层级2 (索引3-6) ======================
    {FeatureType::M_VALUE, {.threshold = 384.0f}},        // 索引3
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}}, // 索引4
    {FeatureType::N_VALUE, {.threshold = 3072.0f}},       // 索引5
    {FeatureType::M_VALUE, {.threshold = 6144.0f}},       // 索引6
    // ====================== 层级3 (索引7-14) ======================
    {FeatureType::M_VALUE, {.threshold = 192.0f}},        // 索引7
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}}, // 索引8
    {FeatureType::WORLD_MUL_K, {.threshold = 6144.0f}},   // 索引9
    {FeatureType::M_MUL_N, {.threshold = 3145728.0f}},    // 索引10
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},       // 索引11
    {FeatureType::K_VALUE, {.threshold = 1536.0f}},       // 索引12
    {FeatureType::M_MUL_N, {.threshold = 6291456.0f}},    // 索引13
    {FeatureType::WORLD_MUL_K, {.threshold = 49152.0f}},  // 索引14
    // ====================== 层级4 (索引15-30) ======================
    {FeatureType::N_VALUE, {.threshold = 384.0f}},         // 索引15
    {FeatureType::WORLD_MUL_K, {.threshold = 12288.0f}},   // 索引16
    {FeatureType::MN_DIV_K, {.threshold = 192.000000f}},   // 索引17
    {FeatureType::M_DIV_N, {.threshold = 0.093750f}},      // 索引18
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},   // 索引19
    {FeatureType::M_DIV_N, {.threshold = 0.187500f}},      // 索引20
    {FeatureType::M_MUL_N, {.threshold = 786432.0f}},      // 索引21
    {FeatureType::MN_DIV_K, {.threshold = 24576.000000f}}, // 索引22
    {FeatureType::WORLD_MUL_K, {.threshold = 12288.0f}},   // 索引23
    {FeatureType::M_DIV_N, {.threshold = 3.000000f}},      // 索引24
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},        // 索引25
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}},  // 索引26
    {FeatureType::MN_DIV_K, {.threshold = 192.000000f}},   // 索引27
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},      // 索引28
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},      // 索引29
    {FeatureType::M_MUL_N, {.threshold = 12582912.0f}},    // 索引30
    // ====================== 层级5 (索引31-62) ======================
    {FeatureType::K_VALUE, {.threshold = 768.0f}},         // 索引31
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引32
    {FeatureType::MN_DIV_K, {.threshold = 6144.000000f}},  // 索引33
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引34
    {FeatureType::MN_DIV_K, {.threshold = 96.000000f}},    // 索引35
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引36
    {FeatureType::RETURN_VALUE, {.return_value = 4}},      // 索引37
    {FeatureType::MN_DIV_K, {.threshold = 3072.000000f}},  // 索引38
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引39
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引40
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引41
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},        // 索引42
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引43
    {FeatureType::WORLD_MUL_K, {.threshold = 3072.0f}},    // 索引44
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引45
    {FeatureType::RETURN_VALUE, {.return_value = 8}},      // 索引46
    {FeatureType::N_VALUE, {.threshold = 1536.0f}},        // 索引47
    {FeatureType::M_DIV_N, {.threshold = 6.000000f}},      // 索引48
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引49
    {FeatureType::RETURN_VALUE, {.return_value = 16}},     // 索引50
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引51
    {FeatureType::MN_DIV_K, {.threshold = 24576.000000f}}, // 索引52
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引53
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引54
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引55
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},        // 索引56
    {FeatureType::WORLD_MUL_K, {.threshold = 12288.0f}},   // 索引57
    {FeatureType::RETURN_VALUE, {.return_value = 3}},      // 索引58
    {FeatureType::MN_DIV_K, {.threshold = 98304.000000f}}, // 索引59
    {FeatureType::MN_DIV_K, {.threshold = 98304.000000f}}, // 索引60
    {FeatureType::N_VALUE, {.threshold = 384.0f}},         // 索引61
    {FeatureType::RETURN_VALUE, {.return_value = 6}},      // 索引62
    // ====================== 层级6 (索引63-126) ======================
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引63
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引64
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引65
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引66
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引67
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引68
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引69
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引70
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引71
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引72
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引73
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引74
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx75: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx76: idle
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引77
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引78
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx79: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx80: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx81: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx82: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx83: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx84: spare
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引85
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引86
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx87: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx88: idle
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引89
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引90
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx91: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx92: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx93: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx94: idle
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引95
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引96
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引97
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引98
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx99: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx100: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx101: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx102: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx103: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx104: spare
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引105
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引106
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx107: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx108: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx109: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx110: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx111: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx112: idle
    {FeatureType::RETURN_VALUE, {.return_value = 32}}, // 索引113
    {FeatureType::RETURN_VALUE, {.return_value = 32}}, // 索引114
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引115
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引116
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx117: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx118: spare
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引119
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引120
    {FeatureType::RETURN_VALUE, {.return_value = 64}}, // 索引121
    {FeatureType::RETURN_VALUE, {.return_value = 6}},  // 索引122
    {FeatureType::RETURN_VALUE, {.return_value = 64}}, // 索引123
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引124
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx125: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx126: idle
};

// [R8]  swizzlDirect优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode swizzldirectRule[] = {
    // ====================== 层级0 (索引0-0) ======================
    {FeatureType::M_MUL_N, {.threshold = 3145728.0f}}, // 索引0
    // ====================== 层级1 (索引1-2) ======================
    {FeatureType::N_VALUE, {.threshold = 384.0f}},    // 索引1
    {FeatureType::M_DIV_N, {.threshold = 0.093750f}}, // 索引2
    // ====================== 层级2 (索引3-6) ======================
    {FeatureType::MN_DIV_K, {.threshold = 96.000000f}},  // 索引3
    {FeatureType::K_VALUE, {.threshold = 3072.0f}},      // 索引4
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},      // 索引5
    {FeatureType::WORLD_MUL_K, {.threshold = 49152.0f}}, // 索引6
    // ====================== 层级3 (索引7-14) ======================
    {FeatureType::MN_DIV_K, {.threshold = 48.000000f}},   // 索引7
    {FeatureType::WORLD_MUL_K, {.threshold = 6144.0f}},   // 索引8
    {FeatureType::N_VALUE, {.threshold = 768.0f}},        // 索引9
    {FeatureType::MN_DIV_K, {.threshold = 96.000000f}},   // 索引10
    {FeatureType::K_VALUE, {.threshold = 384.0f}},        // 索引11
    {FeatureType::RETURN_VALUE, {.return_value = 1}},     // 索引12
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}}, // 索引13
    {FeatureType::M_MUL_N, {.threshold = 6291456.0f}},    // 索引14
    // ====================== 层级4 (索引15-30) ======================
    {FeatureType::M_MUL_N, {.threshold = 196608.0f}},     // 索引15
    {FeatureType::RETURN_VALUE, {.return_value = 1}},     // 索引16
    {FeatureType::M_DIV_N, {.threshold = 24.000000f}},    // 索引17
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}}, // 索引18
    {FeatureType::K_VALUE, {.threshold = 768.0f}},        // 索引19
    {FeatureType::M_MUL_N, {.threshold = 786432.0f}},     // 索引20
    {FeatureType::M_VALUE, {.threshold = 384.0f}},        // 索引21
    {FeatureType::M_DIV_N, {.threshold = 0.023438f}},     // 索引22
    {FeatureType::RETURN_VALUE, {.return_value = 0}},     // 索引23
    {FeatureType::RETURN_VALUE, {.return_value = 0}},     // 索引24
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx25: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx26: spare
    {FeatureType::RETURN_VALUE, {.return_value = 0}},     // 索引27
    {FeatureType::M_DIV_N, {.threshold = 3.000000f}},     // 索引28
    {FeatureType::RETURN_VALUE, {.return_value = 1}},     // 索引29
    {FeatureType::RETURN_VALUE, {.return_value = 1}},     // 索引30
    // ====================== 层级5 (索引31-62) ======================
    {FeatureType::MN_DIV_K, {.threshold = 24.000000f}},   // 索引31
    {FeatureType::RETURN_VALUE, {.return_value = 0}},     // 索引32
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx33: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx34: idle
    {FeatureType::MN_DIV_K, {.threshold = 3072.000000f}}, // 索引35
    {FeatureType::RETURN_VALUE, {.return_value = 1}},     // 索引36
    {FeatureType::K_VALUE, {.threshold = 1536.0f}},       // 索引37
    {FeatureType::RETURN_VALUE, {.return_value = 1}},     // 索引38
    {FeatureType::M_VALUE, {.threshold = 1536.0f}},       // 索引39
    {FeatureType::MN_DIV_K, {.threshold = 48.000000f}},   // 索引40
    {FeatureType::M_DIV_N, {.threshold = 0.187500f}},     // 索引41
    {FeatureType::WORLD_MUL_K, {.threshold = 6144.0f}},   // 索引42
    {FeatureType::N_VALUE, {.threshold = 3072.0f}},       // 索引43
    {FeatureType::MN_DIV_K, {.threshold = 48.000000f}},   // 索引44
    {FeatureType::RETURN_VALUE, {.return_value = 1}},     // 索引45
    {FeatureType::M_VALUE, {.threshold = 768.0f}},        // 索引46
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx47: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx48: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx49: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx50: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx51: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx52: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx53: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx54: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx55: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx56: spare
    {FeatureType::M_DIV_N, {.threshold = 0.750000f}},     // 索引57
    {FeatureType::WORLD_MUL_K, {.threshold = 24576.0f}},  // 索引58
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx59: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx60: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx61: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx62: spare
    // ====================== 层级6 (索引63-126) ======================
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引63
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引64
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx65: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx66: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx67: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx68: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx69: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx70: idle
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引71
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引72
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx73: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx74: spare
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引75
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引76
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx77: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx78: idle
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引79
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引80
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引81
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引82
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引83
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引84
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引85
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引86
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引87
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引88
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引89
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引90
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx91: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx92: spare
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引93
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引94
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx95: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx96: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx97: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx98: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx99: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx100: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx101: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx102: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx103: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx104: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx105: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx106: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx107: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx108: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx109: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx110: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx111: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx112: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx113: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx114: spare
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引115
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引116
    {FeatureType::RETURN_VALUE, {.return_value = 0}}, // 索引117
    {FeatureType::RETURN_VALUE, {.return_value = 1}}, // 索引118
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx119: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx120: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx121: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx122: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx123: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx124: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx125: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},  // idx126: spare
};

// [R8]  pValue优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode pvalueRule[] = {
    // ====================== 层级0 (索引0-0) ======================
    {FeatureType::M_MUL_N, {.threshold = 6291456.0f}}, // 索引0
    // ====================== 层级1 (索引1-2) ======================
    {FeatureType::M_MUL_N, {.threshold = 1572864.0f}}, // 索引1
    {FeatureType::M_VALUE, {.threshold = 6144.0f}},    // 索引2
    // ====================== 层级2 (索引3-6) ======================
    {FeatureType::M_MUL_N, {.threshold = 786432.0f}},   // 索引3
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},     // 索引4
    {FeatureType::M_VALUE, {.threshold = 1536.0f}},     // 索引5
    {FeatureType::M_MUL_N, {.threshold = 12582912.0f}}, // 索引6
    // ====================== 层级3 (索引7-14) ======================
    {FeatureType::RETURN_VALUE, {.return_value = 1}},    // 索引7
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},      // 索引8
    {FeatureType::M_VALUE, {.threshold = 768.0f}},       // 索引9
    {FeatureType::M_DIV_N, {.threshold = 0.187500f}},    // 索引10
    {FeatureType::RETURN_VALUE, {.return_value = 2}},    // 索引11
    {FeatureType::M_MUL_N, {.threshold = 12582912.0f}},  // 索引12
    {FeatureType::WORLD_MUL_K, {.threshold = 24576.0f}}, // 索引13
    {FeatureType::WORLD_MUL_K, {.threshold = 49152.0f}}, // 索引14
    // ====================== 层级4 (索引15-30) ======================
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx15: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx16: idle
    {FeatureType::M_VALUE, {.threshold = 768.0f}},        // 索引17
    {FeatureType::WORLD_MUL_K, {.threshold = 3072.0f}},   // 索引18
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},  // 索引19
    {FeatureType::M_VALUE, {.threshold = 1536.0f}},       // 索引20
    {FeatureType::M_DIV_N, {.threshold = 0.093750f}},     // 索引21
    {FeatureType::M_DIV_N, {.threshold = 3.000000f}},     // 索引22
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx23: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx24: spare
    {FeatureType::MN_DIV_K, {.threshold = 6144.000000f}}, // 索引25
    {FeatureType::M_DIV_N, {.threshold = 0.375000f}},     // 索引26
    {FeatureType::RETURN_VALUE, {.return_value = 8}},     // 索引27
    {FeatureType::RETURN_VALUE, {.return_value = 10}},    // 索引28
    {FeatureType::M_DIV_N, {.threshold = 3.000000f}},     // 索引29
    {FeatureType::RETURN_VALUE, {.return_value = 20}},    // 索引30
    // ====================== 层级5 (索引31-62) ======================
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx31: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx32: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx33: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx34: spare
    {FeatureType::MN_DIV_K, {.threshold = 192.000000f}},   // 索引35
    {FeatureType::K_VALUE, {.threshold = 1536.0f}},        // 索引36
    {FeatureType::RETURN_VALUE, {.return_value = 2}},      // 索引37
    {FeatureType::RETURN_VALUE, {.return_value = 1}},      // 索引38
    {FeatureType::RETURN_VALUE, {.return_value = 2}},      // 索引39
    {FeatureType::K_VALUE, {.threshold = 3072.0f}},        // 索引40
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}},  // 索引41
    {FeatureType::MN_DIV_K, {.threshold = 3072.000000f}},  // 索引42
    {FeatureType::RETURN_VALUE, {.return_value = 1}},      // 索引43
    {FeatureType::RETURN_VALUE, {.return_value = 4}},      // 索引44
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引45
    {FeatureType::M_DIV_N, {.threshold = 6.000000f}},      // 索引46
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx47: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx48: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx49: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx50: spare
    {FeatureType::MN_DIV_K, {.threshold = 1536.000000f}},  // 索引51
    {FeatureType::N_VALUE, {.threshold = 3072.0f}},        // 索引52
    {FeatureType::RETURN_VALUE, {.return_value = 5}},      // 索引53
    {FeatureType::K_VALUE, {.threshold = 1536.0f}},        // 索引54
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx55: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx56: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx57: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx58: spare
    {FeatureType::MN_DIV_K, {.threshold = 98304.000000f}}, // 索引59
    {FeatureType::WORLD_MUL_K, {.threshold = 12288.0f}},   // 索引60
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx61: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx62: idle
    // ====================== 层级6 (索引63-126) ======================
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx63: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx64: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx65: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx66: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx67: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx68: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx69: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx70: idle
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引71
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引72
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引73
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引74
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx75: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx76: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx77: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx78: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx79: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx80: spare
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引81
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引82
    {FeatureType::RETURN_VALUE, {.return_value = 1}},  // 索引83
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引84
    {FeatureType::RETURN_VALUE, {.return_value = 2}},  // 索引85
    {FeatureType::RETURN_VALUE, {.return_value = 7}},  // 索引86
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx87: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx88: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx89: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx90: spare
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引91
    {FeatureType::RETURN_VALUE, {.return_value = 3}},  // 索引92
    {FeatureType::RETURN_VALUE, {.return_value = 5}},  // 索引93
    {FeatureType::RETURN_VALUE, {.return_value = 7}},  // 索引94
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx95: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx96: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx97: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx98: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx99: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx100: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx101: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx102: spare
    {FeatureType::RETURN_VALUE, {.return_value = 5}},  // 索引103
    {FeatureType::RETURN_VALUE, {.return_value = 5}},  // 索引104
    {FeatureType::RETURN_VALUE, {.return_value = 5}},  // 索引105
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引106
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx107: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx108: idle
    {FeatureType::RETURN_VALUE, {.return_value = 10}}, // 索引109
    {FeatureType::RETURN_VALUE, {.return_value = 5}},  // 索引110
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx111: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx112: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx113: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx114: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx115: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx116: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx117: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx118: idle
    {FeatureType::RETURN_VALUE, {.return_value = 20}}, // 索引119
    {FeatureType::RETURN_VALUE, {.return_value = 20}}, // 索引120
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引121
    {FeatureType::RETURN_VALUE, {.return_value = 10}}, // 索引122
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx123: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx124: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx125: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx126: idle
};

// [R8]  ubMoveNum优化参数的决策树规则（使用特征：m, k, n, m_div_n, m_mul_n, world_mul_k, mn_div_k）
const DecisionNode ubmovenumRule[] = {
    // ====================== 层级0 (索引0-0) ======================
    {FeatureType::M_VALUE, {.threshold = 768.0f}}, // 索引0
    // ====================== 层级1 (索引1-2) ======================
    {FeatureType::M_MUL_N, {.threshold = 3145728.0f}}, // 索引1
    {FeatureType::M_DIV_N, {.threshold = 3.000000f}},  // 索引2
    // ====================== 层级2 (索引3-6) ======================
    {FeatureType::M_MUL_N, {.threshold = 786432.0f}},      // 索引3
    {FeatureType::MN_DIV_K, {.threshold = 12288.000000f}}, // 索引4
    {FeatureType::M_MUL_N, {.threshold = 6291456.0f}},     // 索引5
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引6
    // ====================== 层级3 (索引7-14) ======================
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},  // 索引7
    {FeatureType::M_MUL_N, {.threshold = 1572864.0f}},    // 索引8
    {FeatureType::RETURN_VALUE, {.return_value = 16}},    // 索引9
    {FeatureType::RETURN_VALUE, {.return_value = 4}},     // 索引10
    {FeatureType::MN_DIV_K, {.threshold = 3072.000000f}}, // 索引11
    {FeatureType::WORLD_MUL_K, {.threshold = 49152.0f}},  // 索引12
    {FeatureType::K_VALUE, {.threshold = 1536.0f}},       // 索引13
    {FeatureType::M_MUL_N, {.threshold = 393216.0f}},     // 索引14
    // ====================== 层级4 (索引15-30) ======================
    {FeatureType::M_MUL_N, {.threshold = 98304.0f}},      // 索引15
    {FeatureType::N_VALUE, {.threshold = 3072.0f}},       // 索引16
    {FeatureType::M_VALUE, {.threshold = 192.0f}},        // 索引17
    {FeatureType::MN_DIV_K, {.threshold = 3072.000000f}}, // 索引18
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx19: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx20: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx21: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},      // idx22: idle
    {FeatureType::M_VALUE, {.threshold = 1536.0f}},       // 索引23
    {FeatureType::M_DIV_N, {.threshold = 0.750000f}},     // 索引24
    {FeatureType::M_VALUE, {.threshold = 3072.0f}},       // 索引25
    {FeatureType::M_DIV_N, {.threshold = 1.500000f}},     // 索引26
    {FeatureType::RETURN_VALUE, {.return_value = 16}},    // 索引27
    {FeatureType::K_VALUE, {.threshold = 6144.0f}},       // 索引28
    {FeatureType::RETURN_VALUE, {.return_value = 8}},     // 索引29
    {FeatureType::RETURN_VALUE, {.return_value = 8}},     // 索引30
    // ====================== 层级5 (索引31-62) ======================
    {FeatureType::M_VALUE, {.threshold = 192.0f}},         // 索引31
    {FeatureType::N_VALUE, {.threshold = 1536.0f}},        // 索引32
    {FeatureType::N_VALUE, {.threshold = 768.0f}},         // 索引33
    {FeatureType::RETURN_VALUE, {.return_value = 16}},     // 索引34
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引35
    {FeatureType::WORLD_MUL_K, {.threshold = 12288.0f}},   // 索引36
    {FeatureType::MN_DIV_K, {.threshold = 384.000000f}},   // 索引37
    {FeatureType::MN_DIV_K, {.threshold = 6144.000000f}},  // 索引38
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx39: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx40: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx41: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx42: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx43: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx44: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx45: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx46: idle
    {FeatureType::MN_DIV_K, {.threshold = 192.000000f}},   // 索引47
    {FeatureType::MN_DIV_K, {.threshold = 768.000000f}},   // 索引48
    {FeatureType::M_DIV_N, {.threshold = 0.375000f}},      // 索引49
    {FeatureType::MN_DIV_K, {.threshold = 12288.000000f}}, // 索引50
    {FeatureType::N_VALUE, {.threshold = 6144.0f}},        // 索引51
    {FeatureType::RETURN_VALUE, {.return_value = 8}},      // 索引52
    {FeatureType::M_VALUE, {.threshold = 3072.0f}},        // 索引53
    {FeatureType::RETURN_VALUE, {.return_value = 8}},      // 索引54
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx55: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx56: spare
    {FeatureType::RETURN_VALUE, {.return_value = 8}},      // 索引57
    {FeatureType::M_VALUE, {.threshold = 6144.0f}},        // 索引58
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx59: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx60: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx61: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},       // idx62: spare
    // ====================== 层级6 (索引63-126) ======================
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引63
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引64
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引65
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引66
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引67
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引68
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx69: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx70: idle
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引71
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引72
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引73
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引74
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引75
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引76
    {FeatureType::RETURN_VALUE, {.return_value = 16}}, // 索引77
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引78
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx79: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx80: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx81: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx82: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx83: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx84: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx85: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx86: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx87: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx88: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx89: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx90: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx91: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx92: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx93: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx94: idle
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引95
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引96
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引97
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引98
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引99
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引100
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引101
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引102
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引103
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引104
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx105: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx106: spare
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引107
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引108
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx109: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx110: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx111: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx112: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx113: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx114: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx115: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx116: spare
    {FeatureType::RETURN_VALUE, {.return_value = 8}},  // 索引117
    {FeatureType::RETURN_VALUE, {.return_value = 4}},  // 索引118
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx119: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx120: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx121: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx122: spare
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx123: vacant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx124: idle
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx125: dormant
    {FeatureType::PLACEHOLDER, {.threshold = 0.0f}},   // idx126: spare
};

inline int GetOptimalM0(int mSize, int kSize, int nSize)
{
    return TraverseDecisionTree(m0Rule, mSize, kSize, nSize, RANKSIZE_EIGHT, DEFAULT_M0);
}

inline int GetOptimalSwizzlCount(int mSize, int kSize, int nSize)
{
    return TraverseDecisionTree(swizzlcountRule, mSize, kSize, nSize, RANKSIZE_EIGHT, DEFAULT_SWIZZLCOUNT);
}

inline int GetOptimalSwizzlDirect(int mSize, int kSize, int nSize)
{
    return TraverseDecisionTree(swizzldirectRule, mSize, kSize, nSize, RANKSIZE_EIGHT, DEFAULT_SWIZZLDIRECT);
}

inline int GetOptimalPValue(int mSize, int kSize, int nSize)
{
    return TraverseDecisionTree(pvalueRule, mSize, kSize, nSize, RANKSIZE_EIGHT, DEFAULT_PVALUE);
}

inline int GetOptimalUbmovenum(int mSize, int kSize, int nSize)
{
    return TraverseDecisionTree(ubmovenumRule, mSize, kSize, nSize, RANKSIZE_EIGHT, DEFAULT_UBMOVENUM);
}

} // namespace Tiling_Rank8_A2

}; // namespace Tiling_Small_M

namespace optiling {
ge::graphStatus MatmulReduceScatterTilingV2AivModeFunc(gert::TilingContext *context);
}

#endif // DECISION_RULES_H
