#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_MAX_POOL3D_GRAD_CONSTANTS_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_MAX_POOL3D_GRAD_CONSTANTS_H

namespace optiling {

constexpr uint32_t X_INDEX = 0;
constexpr uint32_t GRAD_INDEX = 1;
constexpr uint32_t ARGMAX_INDEX = 2;
constexpr size_t KSIZE_ATTR_INDEX = 0U;
constexpr size_t STRIDES_ATTR_INDEX = 1U;
constexpr size_t PADS_ATTR_INDEX = 2U;
constexpr size_t DILATION_ATTR_INDEX = 3U;
constexpr size_t CEIL_MODE_ATTR_INDEX = 4U;

constexpr size_t NC_DIM_NUM = 2;
constexpr size_t NCDHW_DIM_NUM = 5;
constexpr uint32_t DTYPE_LEN_B8 = 1;
constexpr uint32_t DTYPE_LEN_B16 = 2;
constexpr uint32_t DTYPE_LEN_B32 = 4;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t NUM_PER_REP_B16 = 128;
constexpr uint32_t NUM_PER_REP_B32 = 64;
constexpr uint32_t SELECT_RESERVED_UB_SIZE = 8192;
constexpr uint64_t MAX_INT32 = 2147483647;

constexpr uint32_t TILING_OVERLAP = 100;
constexpr uint32_t TILING_UB_NO_CUT = 0;
constexpr uint32_t TILING_UB_CUT_NC = 10;
constexpr uint32_t TILING_UB_CUT_DO = 20;
constexpr uint32_t TILING_UB_CUT_HO = 30;
constexpr uint32_t TILING_UB_CUT_WO = 40;
constexpr uint32_t TILING_TYPE_SCATTER = 2;

} // namespace optiling

#endif