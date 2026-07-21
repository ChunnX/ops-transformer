#ifndef ALL_GATHER_MATMUL_TILING_DATA_H
#define ALL_GATHER_MATMUL_TILING_DATA_H

#ifndef __CCE_AICORE__
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#endif

#if __has_include("quant_matmul/op_host/op_tiling/quant_matmul_tiling_data.h")
#include "quant_matmul/op_host/op_tiling/quant_matmul_tiling_data.h"
#else
#include "../../3rd/quant_matmul/op_host/op_tiling/quant_matmul_tiling_data.h"
#endif


struct allGatherTilingData {
    uint32_t M = 0;
    uint32_t Ka = 0;
    uint32_t N = 0;
    uint64_t xSize = 0;
    uint64_t xNums = 0;
    uint64_t scaleSize = 0;
    uint32_t mteTileK = 0;
    uint32_t mteMSplitNum = 0;
    uint32_t mteKSplitNum = 0;
    uint32_t mteMSplitSize = 0;
    uint32_t mteKSplitSize = 0;
    uint32_t mteMSplitLargeBlockNum = 0;
    uint32_t singleCoreM = 0;
    uint32_t aivNum = 0;
    uint32_t aicNum = 0;
    uint32_t rankSize = 0;
    uint32_t sendCoreNumPerRank = 0;
    uint64_t ubSize = 0;
    bool isOptionalOutput = false;
};

struct allGatherMatmulTilingData {
    allGatherTilingData allGatherTiling;
    QuantMatmulTilingData qbmmTilingData;
};

#endif // ALL_GATHER_MATMUL_TILING_DATA_H
