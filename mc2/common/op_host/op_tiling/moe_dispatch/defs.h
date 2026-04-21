/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * [PSEUDOCODE] MOE Dispatch Tiling 重构公共层 - 常量与枚举定义
 * 配套文档：dispatch_analysis_opus_3/architecture_v0.md §5.1
 * 配套伪代码：dispatch_analysis_opus_3/pseudocode_v0.md §1.1
 */

#ifndef MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_DEFS_H
#define MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_DEFS_H

#include <cstdint>

namespace moe_dispatch_defs {

// ========== 量化模式 ==========
constexpr uint32_t UNQUANT                = 0;
constexpr uint32_t STATIC_QUANT           = 1;
constexpr uint32_t PERTOKEN_DYNAMIC_QUANT = 2;
constexpr uint32_t PERGROUP_DYNAMIC_QUANT = 3;
constexpr uint32_t MX_QUANT              = 4;

// ========== 对齐常量 ==========
constexpr uint32_t ALIGN_32  = 32;
constexpr uint32_t ALIGN_128 = 128;
constexpr uint32_t ALIGN_256 = 256;
constexpr uint32_t ALIGN_512 = 512;
constexpr uint32_t EVEN_ALIGN = 2;

// ========== 通信常量 ==========
constexpr uint32_t OP_TYPE_ALL_TO_ALL  = 8;
constexpr uint32_t OP_TYPE_ALL_GATHER  = 6;
constexpr uint32_t OP_TYPE_BATCH_WRITE = 18;
constexpr uint32_t URMA_COMM           = 2;
constexpr uint32_t SDMA_COMM           = 0;

// ========== 维度常量 ==========
constexpr uint32_t ONE_DIM  = 1;
constexpr uint32_t TWO_DIMS = 2;

// ========== 公共范围基础值 ==========
// 各算子的特化上界写在各算子自己的 AttrCheckRule 中，不在此处
constexpr uint32_t MIN_H_DEFAULT         = 1024;
constexpr uint32_t MAX_H_DEFAULT         = 8192;
constexpr uint32_t MAX_BS_DEFAULT        = 512;
constexpr uint32_t MAX_K_DEFAULT         = 16;
constexpr uint32_t MAX_SHARED_EXPERT_NUM = 4;
constexpr uint32_t MIN_EP_WORLD_SIZE     = 2;
constexpr uint32_t MAX_GROUP_NAME_LENGTH = 128;

// ========== 其他公共常量 ==========
constexpr uint32_t ASSIST_INFO_FACTOR    = 128;
constexpr uint32_t COMM_CMD_INFO_FACTOR  = 16;
constexpr uint32_t MX_BLOCK_SIZE         = 32;
constexpr uint32_t PERGROUP_BLOCK_SIZE   = 128;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;  // 16MB

// ========== Layout 索引缺省值 ==========
// 可选 tensor/attr 索引字段缺失时使用此值
constexpr int32_t LAYOUT_INDEX_ABSENT = -1;

// ========== TensorCheckRule 辅助常量 ==========
constexpr uint32_t MAX_VALID_DTYPES = 4;

}  // namespace moe_dispatch_defs

#endif  // MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_DEFS_H
