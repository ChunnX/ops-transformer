/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file defs.h
 * \brief MOE Distribute Dispatch Tiling 公共常量和枚举定义
 */

#ifndef MOE_DISPATCH_DEFS_H
#define MOE_DISPATCH_DEFS_H

#include <cstdint>
#include <map>
#include <set>
#include "graph/utils/type_utils.h"
#include "mc2_common_log.h"

namespace moe_dispatch_defs {

// ========== Layout 常量 ==========
constexpr int32_t LAYOUT_INDEX_ABSENT = -1;

// ========== 量化模式枚举 ==========
enum class QuantMode : uint32_t {
    UNQUANT = 0,
    NON_QUANT = 0,  // 别名，向后兼容
    STATIC_QUANT = 1,
    PERTOKEN_DYNAMIC_QUANT = 2,
    PERGROUP_DYNAMIC_QUANT = 3,
    MX_QUANT = 4,
    BUTT
};

// ========== 实际量化模式（考虑 dtype）==========
enum class RealMode : uint32_t {
    NO_SCALES = 0,
    STATIC_SCALES = 1,
    PERTOKEN_SCALES = 2,
    PERGROUP_SCALES = 3,
    MX_SCALES = 4,
    HIF8_SCALES = 5,
    INVALID_MODE
};

// ========== 对齐常量 ==========
constexpr uint32_t ALIGN_32 = 32U;
constexpr uint32_t ALIGN_128 = 128U;
constexpr uint32_t ALIGN_256 = 256U;
constexpr uint32_t ALIGN_512 = 512U;
constexpr uint32_t EVEN_ALIGN = 2U;
constexpr uint32_t UB_ALIGN = 32U;

// ========== 通信常量 ==========
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;
constexpr uint32_t OP_TYPE_ALL_GATHER = 6U;
constexpr uint32_t OP_TYPE_BATCH_WRITE = 18U;
constexpr uint32_t URMA_COMM = 2U;
constexpr uint32_t SDMA_COMM = 0U;

// ========== 维度常量 ==========
constexpr uint32_t ONE_DIM = 1U;
constexpr uint32_t TWO_DIMS = 2U;
constexpr uint32_t DYNAMIC_SCALE_ONE_DIM_NUM = 1U;   // quantMode 2 (PerToken)
constexpr uint32_t DYNAMIC_SCALE_TWO_DIM_NUM = 2U;   // quantMode 3-4 (PerGroup, MX)

// ========== 公共范围基础值 ==========
constexpr int64_t MIN_H_DEFAULT = 1024;
constexpr int64_t MAX_H_DEFAULT = 8192;
constexpr int64_t MAX_H_LAYERED = 7168;
constexpr int64_t MAX_BS_DEFAULT = 512;
constexpr int64_t MAX_BS_LAYERED = 256;
constexpr int64_t FULLMESH_BS_MAX = 256;
constexpr int64_t MAX_K_DEFAULT = 16;
constexpr int64_t FULLMESH_K_MAX = 12;
constexpr int64_t MAX_SHARED_EXPERT_NUM = 4;
constexpr int64_t MIN_EP_WORLD_SIZE = 2;
constexpr int64_t MAX_EP_WORLD_SIZE = 768;            // A3/A5 上界（dispatch_v2）
constexpr int64_t MAX_EP_WORLD_SIZE_V2 = 768;
constexpr int64_t MAX_EP_WORLD_SIZE_LAYERED = 256;
constexpr int64_t MAX_EP_WORLD_SIZE_SETUP_TD = 384;
constexpr int64_t MAX_TP_WORLD_SIZE = 2;              // TP 世界大小上界
constexpr int64_t MAX_TP_WORLD_SIZE_LAYERED = 2;
constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
constexpr int64_t MAX_MOE_EXPERT_NUM = 512;
constexpr int64_t MAX_MOE_EXPERT_NUM_V2 = 1024;
constexpr int64_t MAX_MOE_EXPERT_NUM_LAYERED = 512;
constexpr int64_t MAX_LOCAL_EXPERT_NUM = 2048;

// ========== A2 架构特化常量（值与公共常量不同）==========
constexpr int64_t MAX_MOE_EXPERT_NUM_A2 = 512;
constexpr int64_t MAX_H_A2 = 7168;
constexpr int64_t MAX_H_A2_LAYERED = 10240;
constexpr int64_t MAX_BS_A2 = 256;
constexpr int64_t MAX_BS_A2_LAYERED = 512;
constexpr int64_t MAX_EP_WORLD_SIZE_A2 = 384;
constexpr int64_t MAX_EP_WORLD_SIZE_A2_LAYERED = 64;

// ========== 别名常量（兼容算子层命名）==========
constexpr int64_t H_MIN = MIN_H_DEFAULT;         // h 最小值
constexpr int64_t K_MAX = MAX_K_DEFAULT;         // k 最大值
constexpr uint32_t SIZE_OF_HALF = 2U;            // sizeof(float16)

// ========== 其他公共常量 ==========
constexpr uint64_t MX_BLOCK_SIZE = 32U;
constexpr uint64_t PERGROUP_BLOCK_SIZE = 128U;
constexpr uint32_t QUANT_HS_OFFSET = 4U;
constexpr uint32_t ELASTIC_INFO_OFFSET = 4U;
constexpr uint32_t RANK_LIST_NUM = 2U;
constexpr uint32_t ASSIST_INFO_NUM_PER_A = 128U;
constexpr uint32_t COMM_CMD_INFO_MULTIPLY = 16U;
constexpr uint64_t SYSTEM_NEED_WORKSPACE = 16UL * 1024UL * 1024UL;
constexpr uint32_t TP_WORLD_SIZE_TWO = 2U;
constexpr uint32_t TP_WORLD_SIZE_ONE = 1U;

// ========== 通信引擎常量 ==========
constexpr uint8_t AIV_ENGINE = 3U;
constexpr uint8_t A5_CCU_ENGINE = 5U;

// ========== 支持的 dtype 集合 ==========
const std::set<ge::DataType> NON_QUANT_DTYPE = {
    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_HIFLOAT8,
    ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
    ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2
};

const std::set<ge::DataType> QUANT_STATIC_INPUT_DTYPE = {
    ge::DT_INT8, ge::DT_HIFLOAT8
};

const std::set<ge::DataType> QUANT_PERTOKEN_INPUT_DTYPE = {
    ge::DT_INT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN
};

const std::set<ge::DataType> QUANT_PERGROUP_INPUT_DTYPE = {
    ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN
};

const std::set<ge::DataType> QUANT_MX_INPUT_DTYPE = {
    ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN,
    ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2
};

// ========== TensorType 枚举 ==========
enum class TensorType : uint32_t {
    INPUT = 0,
    OUTPUT = 1,
    OPTIONAL_INPUT = 2
};

// ========== QuantMode + dtype → RealMode 查表 ==========
inline RealMode CheckQuantModeMap(QuantMode quantMode, ge::DataType dtype)
{
    static const std::map<std::pair<QuantMode, ge::DataType>, RealMode> kQuantModeMap = {
        {{QuantMode::NON_QUANT, ge::DT_BF16}, RealMode::NO_SCALES},
        {{QuantMode::NON_QUANT, ge::DT_FLOAT16}, RealMode::NO_SCALES},
        {{QuantMode::NON_QUANT, ge::DT_FLOAT8_E4M3FN}, RealMode::NO_SCALES},
        {{QuantMode::NON_QUANT, ge::DT_FLOAT8_E5M2}, RealMode::NO_SCALES},
        {{QuantMode::NON_QUANT, ge::DT_HIFLOAT8}, RealMode::NO_SCALES},
        {{QuantMode::NON_QUANT, ge::DT_FLOAT4_E2M1}, RealMode::NO_SCALES},
        {{QuantMode::NON_QUANT, ge::DT_FLOAT4_E1M2}, RealMode::NO_SCALES},
        {{QuantMode::STATIC_QUANT, ge::DT_INT8}, RealMode::STATIC_SCALES},
        {{QuantMode::STATIC_QUANT, ge::DT_HIFLOAT8}, RealMode::HIF8_SCALES},
        {{QuantMode::PERTOKEN_DYNAMIC_QUANT, ge::DT_INT8}, RealMode::PERTOKEN_SCALES},
        {{QuantMode::PERTOKEN_DYNAMIC_QUANT, ge::DT_FLOAT8_E5M2}, RealMode::PERTOKEN_SCALES},
        {{QuantMode::PERTOKEN_DYNAMIC_QUANT, ge::DT_FLOAT8_E4M3FN}, RealMode::PERTOKEN_SCALES},
        {{QuantMode::PERGROUP_DYNAMIC_QUANT, ge::DT_FLOAT8_E5M2}, RealMode::PERGROUP_SCALES},
        {{QuantMode::PERGROUP_DYNAMIC_QUANT, ge::DT_FLOAT8_E4M3FN}, RealMode::PERGROUP_SCALES},
        {{QuantMode::MX_QUANT, ge::DT_FLOAT8_E5M2}, RealMode::MX_SCALES},
        {{QuantMode::MX_QUANT, ge::DT_FLOAT8_E4M3FN}, RealMode::MX_SCALES},
        {{QuantMode::MX_QUANT, ge::DT_FLOAT4_E2M1}, RealMode::MX_SCALES},
        {{QuantMode::MX_QUANT, ge::DT_FLOAT4_E1M2}, RealMode::MX_SCALES},
    };
    auto it = kQuantModeMap.find({quantMode, dtype});
    return (it != kQuantModeMap.end()) ? it->second : RealMode::INVALID_MODE;
}

} // namespace moe_dispatch_defs

#endif // MOE_DISPATCH_DEFS_H
