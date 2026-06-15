#ifndef ACLNN_FUSED_QKV_PROJECTION_TILING_DATA_H
#define ACLNN_FUSED_QKV_PROJECTION_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

struct AclnnFusedQkvProjectionTilingData {
    TCubeTiling cubeTiling;

    int32_t M;                  // 总行数 (batch * seq_len)
    int32_t N;                  // 总列数 (q_dim + k_dim + v_dim)
    int32_t K;                  // 隐藏层维度
    int32_t singleCoreM;        // 单核处理的行数
    int32_t singleCoreN;        // 单核处理的列数
    int32_t baseM;              // 单次迭代处理的行数
    int32_t baseN;              // 单次迭代处理的列数
    int32_t baseK;              // 单次迭代处理的 K 维度
    int32_t qDim;
    int32_t kDim;
    int32_t vDim;
    
    int64_t sysWsOffset;       // byte
    int64_t fusedOutOffset;    // byte

    bool    hasBias;
    uint32_t blockDim;
};
#endif
