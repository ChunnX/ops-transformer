#ifndef ACLNN_FUSED_QKV_PROJECTION_H
#define ACLNN_FUSED_QKV_PROJECTION_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "aclnn_fused_qkv_projection_tiling_data.h"

using namespace AscendC;
using namespace matmul;

template <typename T>
class KernelFusedQkvProjection {
public:
    __aicore__ inline KernelFusedQkvProjection() {}
    __aicore__ inline void Init(GM_ADDR hs, GM_ADDR wt, GM_ADDR bias,
                                GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                GM_ADDR ws, const AclnnFusedQkvProjectionTilingData* td);
    __aicore__ inline void Process();

private:
    using aType = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using bType = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using cType = MatmulType<TPosition::LCM, CubeFormat::ND, T>;
    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, T>;

    // ---- CopyOut ----
    __aicore__ inline void CopyQkvOutputs(const LocalTensor<T>& cOut, int32_t computedRows, int32_t tileRows, int32_t matmulStride) {
        for (int32_t r = 0; r < tileRows; r++) {
            int32_t gr = computedRows + r;
            int32_t srcOff = r * matmulStride;
            
            if constexpr (std::is_same_v<T, float>) {
                // FP32 按 8 元素 (32 Bytes) 对齐
                if (qDim_ % 8 == 0 && kDim_ % 8 == 0 && vDim_ % 8 == 0) {
                    DataCopy(qGm_[gr * qDim_], cOut[srcOff], qDim_);
                    DataCopy(kGm_[gr * kDim_], cOut[srcOff + qDim_], kDim_);
                    DataCopy(vGm_[gr * vDim_], cOut[srcOff + qDim_ + kDim_], vDim_);
                } else {
                    for (int32_t i = 0; i < qDim_; i++) qGm_.SetValue(gr * qDim_ + i, cOut.GetValue(srcOff + i));
                    for (int32_t i = 0; i < kDim_; i++) kGm_.SetValue(gr * kDim_ + i, cOut.GetValue(srcOff + qDim_ + i));
                    for (int32_t i = 0; i < vDim_; i++) vGm_.SetValue(gr * vDim_ + i, cOut.GetValue(srcOff + qDim_ + kDim_ + i));
                }
            } else {
                // FP16 按 32 元素 (64 Bytes) 严格对齐，否则硬件报错。若不对齐则降级为 SetValue
                if (qDim_ % 32 == 0 && kDim_ % 32 == 0 && vDim_ % 32 == 0 && 
                    srcOff % 32 == 0 && (srcOff + qDim_) % 32 == 0 && (srcOff + qDim_ + kDim_) % 32 == 0 &&
                    (gr * qDim_) % 32 == 0 && (gr * kDim_) % 32 == 0 && (gr * vDim_) % 32 == 0) {
                    DataCopy(qGm_[gr * qDim_], cOut[srcOff], qDim_);
                    DataCopy(kGm_[gr * kDim_], cOut[srcOff + qDim_], kDim_);
                    DataCopy(vGm_[gr * vDim_], cOut[srcOff + qDim_ + kDim_], vDim_);
                } else {
                    for (int32_t i = 0; i < qDim_; i++) qGm_.SetValue(gr * qDim_ + i, cOut.GetValue(srcOff + i));
                    for (int32_t i = 0; i < kDim_; i++) kGm_.SetValue(gr * kDim_ + i, cOut.GetValue(srcOff + qDim_ + i));
                    for (int32_t i = 0; i < vDim_; i++) vGm_.SetValue(gr * vDim_ + i, cOut.GetValue(srcOff + qDim_ + kDim_ + i));
                }
            }
        }
    }

    TPipe pipe;
    Matmul<aType, bType, cType, biasType> mm;
    GlobalTensor<T> hsGm_, wtGm_, biasGm_;
    GlobalTensor<T> qGm_, kGm_, vGm_;
    TBuf<> localWsBuf_;
    TBuf<> tileBuf_;
    const AclnnFusedQkvProjectionTilingData* td_;
    int32_t qDim_, kDim_, vDim_, fusedDim_;
    bool hasBias_;
    int32_t realCoreM_;
};

template <typename T>
__aicore__ inline void KernelFusedQkvProjection<T>::Init(
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

    hsGm_.SetGlobalBuffer((__gm__ T*)hs, td_->M * td_->K);
    wtGm_.SetGlobalBuffer((__gm__ T*)wt, td_->K * td_->N);
    if (hasBias_) {
        biasGm_.SetGlobalBuffer((__gm__ T*)bias, td_->N);
    }

    qGm_.SetGlobalBuffer((__gm__ T*)q + blkOff * qDim_, realCoreM_ * qDim_);
    kGm_.SetGlobalBuffer((__gm__ T*)k + blkOff * kDim_, realCoreM_ * kDim_);
    vGm_.SetGlobalBuffer((__gm__ T*)v + blkOff * vDim_, realCoreM_ * vDim_);

    pipe.InitBuffer(tileBuf_, td_->baseM * td_->cubeTiling.N * sizeof(T));
    pipe.InitBuffer(localWsBuf_, 100 * 1024);
}

template <typename T>
__aicore__ inline void KernelFusedQkvProjection<T>::Process()
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
    int32_t matmulStride = td_->cubeTiling.N;

    while (computedRows < realCoreM_ && mm.template Iterate<true>()) {
        int32_t tileRows = td_->baseM;
        if (computedRows + tileRows > realCoreM_) {
            tileRows = realCoreM_ - computedRows;
        }

        LocalTensor<T> cOut = tileBuf_.Get<T>();
        mm.template GetTensorC<true>(cOut, false, true);

        CopyQkvOutputs(cOut, computedRows, tileRows, matmulStride);
        AscendC::PipeBarrier<PIPE_ALL>();
        computedRows += tileRows;
    }
    mm.End();
}

#endif
