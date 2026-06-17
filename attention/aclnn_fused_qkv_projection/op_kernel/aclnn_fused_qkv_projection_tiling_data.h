#ifndef ACLNN_FUSED_QKV_PROJECTION_TILING_DATA_H
#define ACLNN_FUSED_QKV_PROJECTION_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

struct AclnnFusedQkvProjectionTilingData {
    TCubeTiling cubeTiling;
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t singleCoreM;
    int32_t singleCoreN;
    int32_t baseM;
    int32_t baseN;
    int32_t baseK;
    int32_t qDim;
    int32_t kDim;
    int32_t vDim;
    int64_t sysWsOffset;
    int64_t fusedOutOffset;
    bool hasBias;
    uint32_t blockDim;
    int32_t dtype; // ge::DT_FLOAT(0) or ge::DT_FLOAT16(1)
};

#endif
