#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_REPEAT_INTERLEAVE_GRAD_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_REPEAT_INTERLEAVE_GRAD_H
#include <cstdint>
#include "op_common/atvoss/reduce/reduce_tiling.h"

namespace optiling {

struct RepeatInterleaveGradCompileInfo {
    int64_t coreNum;
    int64_t ubSize;
    int64_t blockSize;
    bool isAscendC;
    uint32_t clSize;
    uint32_t vRegSize;
};
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_REPEAT_INTERLEAVE_GRAD_H