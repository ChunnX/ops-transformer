#ifndef ACLNN_FUSED_QKV_PROJECTION_H
#define ACLNN_FUSED_QKV_PROJECTION_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "aclnn_fused_qkv_projection_tiling_data.h"

using namespace AscendC;
using namespace matmul;

class KernelFusedQkvProjection {
public:
    __aicore__ inline KernelFusedQkvProjection() {}
    __aicore__ inline void Init(GM_ADDR hs, GM_ADDR wt, GM_ADDR bias,
                                GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                GM_ADDR ws, const AclnnFusedQkvProjectionTilingData* td);
    __aicore__ inline void Process();

private:
    using aType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using bType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using cType = MatmulType<TPosition::LCM, CubeFormat::ND, float>;
    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    TPipe pipe;
    Matmul<aType, bType, cType, biasType> mm; 

    GlobalTensor<float> hsGm_, wtGm_, biasGm_;
    GlobalTensor<float> qGm_, kGm_, vGm_;
    TBuf<> localWsBuf_;
    TBuf<> tileBuf_;

    const AclnnFusedQkvProjectionTilingData* td_;
    int32_t qDim_, kDim_, vDim_, fusedDim_;
    bool hasBias_;
    int32_t realCoreM_;
};

__aicore__ inline void KernelFusedQkvProjection::Init(
    GM_ADDR hs, GM_ADDR wt, GM_ADDR bias,
    GM_ADDR q, GM_ADDR k, GM_ADDR v,
    GM_ADDR ws, const AclnnFusedQkvProjectionTilingData* td)
{
    td_ = td;
    SetSysWorkspace(ws);
    qDim_ = td_->qDim; 
    kDim_ = td_->kDim; 
    vDim_ = td_->vDim;
    fusedDim_ = qDim_ + kDim_ + vDim_;
    hasBias_ = td_->hasBias;

    uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= td_->blockDim) return;

    int64_t blkOff = td_->singleCoreM * blockIdx;
    realCoreM_ = (blkOff >= td_->M) ? 0 : (td_->singleCoreM < (td_->M - blkOff) ? td_->singleCoreM : static_cast<int32_t>(td_->M - blkOff));
    
    hsGm_.SetGlobalBuffer((__gm__ float*)hs, td_->M * td_->K);
    wtGm_.SetGlobalBuffer((__gm__ float*)wt, td_->K * td_->N);
    if (hasBias_) {
        biasGm_.SetGlobalBuffer((__gm__ float*)bias, td_->N);
    }


    qGm_.SetGlobalBuffer((__gm__ float*)q + blkOff * qDim_, realCoreM_ * qDim_);
    kGm_.SetGlobalBuffer((__gm__ float*)k + blkOff * kDim_, realCoreM_ * kDim_);
    vGm_.SetGlobalBuffer((__gm__ float*)v + blkOff * vDim_, realCoreM_ * vDim_);


    pipe.InitBuffer(tileBuf_, td_->baseM * td_->cubeTiling.N * sizeof(float));

    
    pipe.InitBuffer(localWsBuf_, 100 * 1024);
}

__aicore__ inline void KernelFusedQkvProjection::Process()
{
    uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= td_->blockDim || realCoreM_ == 0) return;

    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &td_->cubeTiling);
    
    mm.SetOrgShape(td_->M, td_->N, td_->K);

    mm.SetLocalWorkspace(localWsBuf_.Get<uint8_t>());
    mm.SetTensorA(hsGm_);
    mm.SetTensorB(wtGm_);
    if (hasBias_) {
        mm.SetBias(biasGm_);
    }

    int32_t computedRows = 0;
    int32_t matmulStride = td_->cubeTiling.N; // 必须使用对齐后的物理宽度（Padding Width）

    while (computedRows < realCoreM_ && mm.template Iterate<true>()) {
        int32_t tileRows = td_->baseM;
        if (computedRows + tileRows > realCoreM_) tileRows = realCoreM_ - computedRows;

        LocalTensor<float> cOut = tileBuf_.Get<float>();
        mm.template GetTensorC<true>(cOut, false, true);
        
        for (int32_t r = 0; r < tileRows; r++) {
            int32_t gr = computedRows + r; 
            int32_t srcOff = r * matmulStride;

            // 尺寸完美的正常搬运 vs 畸形尺寸的标量安全保底
            if (qDim_ % 8 == 0 && kDim_ % 8 == 0 && vDim_ % 8 == 0) {
                DataCopy(qGm_[gr * qDim_], cOut[srcOff], qDim_);
                DataCopy(kGm_[gr * kDim_], cOut[srcOff + qDim_], kDim_);
                DataCopy(vGm_[gr * vDim_], cOut[srcOff + qDim_ + kDim_], vDim_);
            } else {
                for (int32_t i = 0; i < qDim_; i++) qGm_.SetValue(gr * qDim_ + i, cOut.GetValue(srcOff + i));
                for (int32_t i = 0; i < kDim_; i++) kGm_.SetValue(gr * kDim_ + i, cOut.GetValue(srcOff + qDim_ + i));
                for (int32_t i = 0; i < vDim_; i++) vGm_.SetValue(gr * vDim_ + i, cOut.GetValue(srcOff + qDim_ + kDim_ + i));
            }
        }
        
        AscendC::PipeBarrier<PIPE_ALL>();
        computedRows += tileRows;
    }
    mm.End();
}

#endif