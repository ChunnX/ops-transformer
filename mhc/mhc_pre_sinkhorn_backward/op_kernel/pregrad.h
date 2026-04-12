#ifndef MHC_PRE_GRAD_H
#define MHC_PRE_GRAD_H
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "mhc_pre_grad_simd_vf.h"
#include "mhc_pre_grad_simt_vf.h"

using namespace AscendC;
using namespace MicroAPI;

namespace {
constexpr int32_t DOUBLE_BUF = 2;
constexpr int32_t BYTE_SIZE_PER_BLOCK = 32;
constexpr int32_t ELEMENTS_SIZE_PER_BLOCK = BYTE_SIZE_PER_BLOCK / sizeof(float);
constexpr int32_t FP32_PER_REPEAT = 64;

constexpr MatmulConfig MHC_PRE_GRAD_MM1_CFG = GetMDLConfig(false, false, 0, true, false, false, true);
constexpr MatmulConfig MHC_PRE_GRAD_MM2_CFG = GetMDLConfig(false, false, 0, true, false, false, true);
}

template<typename T>
class MhcPreGradOneStageKernel {
public:

    using A0Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using A1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;

    matmul::MatmulImpl<A0Type, BType, CType, CType, MHC_PRE_GRAD_MM1_CFG> mm1_;
    matmul::MatmulImpl<A1Type, BType, CType, CType, MHC_PRE_GRAD_MM2_CFG> mm2_;

    __aicore__ inline MhcPreGradOneStageKernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hc_fn, GM_ADDR pre, GM_ADDR grad_y, GM_ADDR grad_post, GM_ADDR grad_comb, GM_ADDR hc_scale,
        GM_ADDR hc_base, GM_ADDR h_hat2, GM_ADDR rsqrt, GM_ADDR sk_colSum, GM_ADDR sk_colNorm, GM_ADDR sk_rowSum, GM_ADDR sk_rowNorm,
        GM_ADDR grad_x, GM_ADDR grad_hc_fn, GM_ADDR grad_hc_scale, GM_ADDR grad_hc_base, GM_ADDR workspace, const MhcPreGradTilingData* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;
        blkIdx_ = GetBlockIdx();

        InitTiling(tilingData);
        InitGM(x, hc_fn, pre, grad_y, grad_post, grad_comb, hc_scale, hc_base, h_hat2, rsqrt, sk_colSum, sk_colNorm, sk_rowSum, sk_rowNorm, grad_x, grad_hc_fn,
            grad_hc_scale, grad_hc_base, workspace);
        InitBuffer();
    }
    __aicore__ inline void Process();

protected:
    int8_t ping1_ = 0;
    int64_t blkIdx_, aivNum_, aicNum_;
    TPipe* pipe_;
    int64_t batchSize_, seqLength_, totalTasks_, totalTasksAligned_;
    int64_t c_, n_, c0_, c1_, cTail_;
    int64_t tile_;
    int64_t skIterCount_;
    int64_t mm1K_, mm1M_, mm1N_;
    int64_t mm2K_, mm2M_, mm2N_;
    T hcScalePre_, hcScalePost_, hcScaleRes_;

    TBuf<TPosition::VECCALC> gradYBuf_, preBuf_, xBuf_, gradXBuf_, gradPreBuf_, fusedGradHPreAndGradHPostBuf_, gradHResBuf_;
    TBuf<TPosition::VECCALC> hcAlphaBuf_, hcBaseBuf_, rsqrtBuf_, gradRsqrtBuf_;
    TBuf<TPosition::VECCALC> fusedGradHPre2AndGradHPost2Buf_, fusedHPre2AndHPost2Buf_, hRes2Buf_;
    TBuf<TPosition::VECCALC> sigmoidGradBuf1_, sigmoidGradBuf2_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> gradRMSNormBuf_;
    TBuf<TPosition::VECOUT> gradH2Buf_;
    TBuf<TPosition::VECOUT> gradHcScaleBuf_, gradHcBaseBuf_;
    

    LocalTensor<T> gradYLocal_, xLocal_, gradXLocal_;
    LocalTensor<T> preLocal_, gradPreLocal_;
    LocalTensor<T> gradHPreLocal_, gradHPostLocal_, fusedGradHPreAndGradHPostLocal_, gradHResLocal_;
    LocalTensor<T> hcBaseLocal_, hcScaleLocal_;
    LocalTensor<T> gradHcBaseLocal_, gradHcScaleLocal_;
    
    LocalTensor<T> rsqrtLocal_, gradRsqrtLocal_;
    LocalTensor<T> fusedGradHPre2AndGradHPost2Local_, gradH2Local_;
    LocalTensor<T> fusedHPre2AndHPost2Local_, hRes2Local_;
    LocalTensor<T> tmpLocal_;
    LocalTensor<T> gradSigmoidLocal1_;
    LocalTensor<T> gradSigmoidLocal2_;
    LocalTensor<T> gradRMSNormLocal_;

    GlobalTensor<T> xGlobal_, gradYGlobal_;
    GlobalTensor<T> preGlobal_;
    GlobalTensor<T> gradXGlobal_;
    GlobalTensor<T> gradPreGlobal_, gradPostGlobal_;
    GlobalTensor<T> hcScaleGlobal_, hcBaseGlobal_;
    GlobalTensor<T> rsqrtGlobal_;
    GlobalTensor<T> gradHcScaleGlobal_, gradHcBaseGlobal_;
    GlobalTensor<T> h2Global_;
    GlobalTensor<T> skRowNormGlobal_, skRowSumGlobal_, skColNormGlobal_, skColSumGlobal_;
    GlobalTensor<T> gradHResGlobal_;
    GlobalTensor<T> gradH2Global_;
    GlobalTensor<T> gradWeightGlobal_;
    GlobalTensor<T> weightGlobal_;

private:
    __aicore__ inline void InitTiling(const MhcPreGradTilingData* tilingData)
    {
        batchSize_ = tilingData->batchSize;
        seqLength_ = tilingData->seqLength;
        aivNum_ = tilingData->aivNum;
        aicNum_ = tilingData->aivNum / 2;
        c_ = tilingData->c;
        n_ = tilingData->n;
        c0_ = tilingData->c0;
        c1_ = tilingData->c1;
        cTail_ = tilingData->cTail;
        skIterCount_ = tilingData->skIterCount;
        tile_ = tilingData->tileSize;
        totalTasks_ = batchSize_ * seqLength_;
        totalTasksAligned_ = AlignUp(totalTasks_, aivNum_ * tile_);
        
        if ASCEND_IS_AIC {
            mm1K_ = n_ * n_ + 2 * n_;
            mm1M_ = tile_ * 2;
            mm1N_ = n_ * c_;

            mm2K_ = batchSize_ * seqLength_;
            mm2M_ = n_ * n_ + 2 * n_;
            mm2N_ = n_ * c_;
        }
    }

    __aicore__ inline void InitGM(GM_ADDR x, GM_ADDR hc_fn, GM_ADDR pre, GM_ADDR grad_y, GM_ADDR grad_post, GM_ADDR grad_comb,
        GM_ADDR hc_scale, GM_ADDR hc_base, GM_ADDR h_hat2, GM_ADDR rsqrt, GM_ADDR sk_colSum, GM_ADDR sk_colNorm, GM_ADDR sk_rowSum,
        GM_ADDR sk_rowNorm, GM_ADDR grad_x, GM_ADDR grad_hc_fn, GM_ADDR grad_hc_scale, GM_ADDR grad_hc_base, GM_ADDR workspace)
    {
        // cube 和 vector 共用
        xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));               // fp32
        gradWeightGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_hc_fn));  // fp32
        weightGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(hc_fn));  // fp32
        gradXGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_x));   // fp32
        gradH2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace));   // fp32

        if ASCEND_IS_AIV {
            preGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(pre));        // fp32
            gradYGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_y));   // fp32
            
            gradPostGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_post));   // fp32

            hcScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(hc_scale));   // fp32
            hcBaseGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(hc_base));   // fp32

            rsqrtGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(rsqrt));   // fp32
            gradHcScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_hc_scale));
            gradHcBaseGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_hc_base));

            h2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(h_hat2));   // fp32

            skRowNormGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(sk_rowNorm));   // fp32
            skRowSumGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(sk_rowSum));   // fp32
            skColNormGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(sk_colNorm));   // fp32
            skColSumGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(sk_colSum));   // fp32

            gradHResGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_comb));   // fp32
        }
    }

    __aicore__ inline void InitBuffer()
    {
        
        if ASCEND_IS_AIV {
            // pipe_->Reset();
            pipe_->InitBuffer(xBuf_, DOUBLE_BUF * n_ * c_ * sizeof(T));
            pipe_->InitBuffer(gradXBuf_, DOUBLE_BUF * c_ * sizeof(T));
            pipe_->InitBuffer(gradYBuf_, DOUBLE_BUF * c_ * sizeof(T));
            pipe_->InitBuffer(preBuf_, DOUBLE_BUF * 8 * sizeof(T));
            pipe_->InitBuffer(rsqrtBuf_, tile_ * sizeof(T));
            pipe_->InitBuffer(gradH2Buf_, tile_ * (n_ * n_ + 2 * n_) * sizeof(T));

            pipe_->InitBuffer(sigmoidGradBuf1_, tile_ * 2 * n_ * sizeof(T));
            pipe_->InitBuffer(sigmoidGradBuf2_, tile_ * 2 * n_ * sizeof(T));

            pipe_->InitBuffer(hcBaseBuf_, (n_ * n_ + 2 * n_) * sizeof(T));
            pipe_->InitBuffer(hcAlphaBuf_, (n_ * n_ + 2 * n_) * sizeof(T));

            pipe_->InitBuffer(tmpBuf_, tile_ * (n_ * n_ + 2 * n_) * sizeof(T));
            pipe_->InitBuffer(gradRsqrtBuf_, tile_ * sizeof(T));
            pipe_->InitBuffer(gradHResBuf_, tile_ * n_ * n_ * sizeof(T));
            pipe_->InitBuffer(fusedGradHPreAndGradHPostBuf_, tile_ * 2 * n_ * sizeof(T));
            pipe_->InitBuffer(fusedHPre2AndHPost2Buf_, tile_ * 2 * n_ * sizeof(T));
            pipe_->InitBuffer(gradHcScaleBuf_, 8 * sizeof(T));
            pipe_->InitBuffer(gradHcBaseBuf_, 2 * n_ * sizeof(T));
            pipe_->InitBuffer(gradRMSNormBuf_, tile_* sizeof(T));

            xLocal_ = xBuf_.Get<T>();
            gradYLocal_ = gradYBuf_.Get<T>();

            fusedGradHPreAndGradHPostLocal_ = fusedGradHPreAndGradHPostBuf_.Get<T>();
            fusedHPre2AndHPost2Local_ = fusedHPre2AndHPost2Buf_.Get<T>();

            hcBaseLocal_ = hcBaseBuf_.Get<T>();
            hcScaleLocal_ = hcAlphaBuf_.Get<T>();

            gradH2Local_ = gradH2Buf_.Get<T>();
            tmpLocal_ = tmpBuf_.Get<T>();
            
            gradHResLocal_ = gradHResBuf_.Get<T>();
            
            rsqrtLocal_ = rsqrtBuf_.Get<T>();
            gradRsqrtLocal_ = gradRsqrtBuf_.Get<T>();

            gradSigmoidLocal1_ = sigmoidGradBuf1_.Get<T>();
            gradSigmoidLocal2_ = sigmoidGradBuf2_.Get<T>();

            gradHcBaseLocal_ = gradHcBaseBuf_.Get<T>();
            gradHcScaleLocal_ = gradHcScaleBuf_.Get<T>();

            gradRMSNormLocal_ = gradRMSNormBuf_.Get<T>();
            gradXLocal_ = gradXBuf_.Get<T>();

            preLocal_ = preBuf_.Get<T>();
        }
    }

    __aicore__ inline void GetHcScaleAndHcBase();
    __aicore__ inline void ComputeGradResHHat2(const int32_t taskOffset, const int32_t tileTaskCount);
    __aicore__ inline void ComputeGradSigmoid(const int32_t taskOffset, const int32_t tileTaskCount);
    __aicore__ inline T ComputeGradRsqrt(const int32_t i);
    __aicore__ inline void ComputeGradRMSNorm(const int32_t tileTaskCount);
    __aicore__ inline void ComputeGradXVector(const T gradRsqrtVal, const T gradRMSNormVal, const int32_t taskOffset);
    __aicore__ inline void ProcessMatmul1(const int32_t taskOffset, const int32_t mm1M);
    __aicore__ inline void ProcessMatmul2(const int32_t taskOffset, const int32_t mm2K);
};

template<typename T>
__aicore__ inline void MhcPreGradOneStageKernel<T>::GetHcScaleAndHcBase()
{
    // HcBase 常驻 UB
    hcScalePre_ = hcScaleGlobal_.GetValue(0);
    hcScalePost_ = hcScaleGlobal_.GetValue(1);
    hcScaleRes_ = hcScaleGlobal_.GetValue(2);
    DataCopyPad(hcBaseLocal_, hcBaseGlobal_, {static_cast<uint16_t>(1),
        static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    
    for (int32_t i = 0; i < n_; i++) {
        hcScaleLocal_.SetValue(i, hcScalePre_);
        hcScaleLocal_.SetValue(i + n_, hcScalePost_);
    }
}

template<typename T>
__aicore__ inline void MhcPreGradOneStageKernel<T>::ComputeGradResHHat2(const int32_t taskOffset, const int32_t tileTaskCount)
{
    // Compute sigmoidGrad
    DataCopyPad(hRes2Local_, h2Global_[taskOffset * (n_ * n_ + 2 * n_) + 2 * n_], {static_cast<uint16_t>(tileTaskCount),
        static_cast<uint32_t>(n_ * n_ * sizeof(T)), static_cast<int64_t>(2 * n_ * sizeof(T)), 0, 0}, {false, 0, 0, 0});
    DataCopyPad(gradHResLocal_, gradHResGlobal_[taskOffset * n_ * n_], {static_cast<uint16_t>(tileTaskCount),
        static_cast<uint32_t>(n_ * n_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(rsqrtLocal_, rsqrtGlobal_[taskOffset], {static_cast<uint16_t>(1),
        static_cast<uint32_t>(tileTaskCount * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});

    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    uint32_t thread_num = min(static_cast<uint32_t>(THREAD_NUM), static_cast<uint32_t>(tileTaskCount * n_ * n_));
    uint32_t xThreadCount = n_ * n_;
    uint32_t yThreadCount = thread_num / xThreadCount;

    Duplicate(gradRsqrtLocal_, 0.f, tileTaskCount);
    Simt::VF_CALL<SinkhornKnoppSimt<T>>(Simt::Dim3{xThreadCount, yThreadCount},
        (__ubuf__ T*) gradH2Local_.GetPhyAddr(),
        (__ubuf__ T*) gradRsqrtLocal_.GetPhyAddr(),
        (__gm__ T*) gradHcScaleGlobal_[2].GetPhyAddr(),
        (__gm__ T*) gradHcBaseGlobal_[n_ * 2].GetPhyAddr(),
        (__gm__ T*) skRowNormGlobal_.GetPhyAddr(),
        (__gm__ T*) skRowSumGlobal_.GetPhyAddr(),
        (__gm__ T*) skColNormGlobal_.GetPhyAddr(),
        (__gm__ T*) skColSumGlobal_.GetPhyAddr(),
        (__ubuf__ T*) gradHResLocal_.GetPhyAddr(),
        (__ubuf__ T*) hRes2Local_.GetPhyAddr(),
        (__ubuf__ T*) rsqrtLocal_.GetPhyAddr(),
        (__ubuf__ T*) hcBaseLocal_[n_ * 2].GetPhyAddr(),
        (__ubuf__ T*) tmpLocal_.GetPhyAddr(),
        hcScaleRes_, totalTasks_, taskOffset, tileTaskCount, skIterCount_, n_, 1e-6);
}

template<typename T>
__aicore__ inline void MhcPreGradOneStageKernel<T>::ComputeGradSigmoid(const int32_t taskOffset, const int32_t tileTaskCount)
{
    DataCopyPad(fusedHPre2AndHPost2Local_, h2Global_[taskOffset * (n_ * n_ + 2 * n_)], {static_cast<uint16_t>(tileTaskCount),
        static_cast<uint32_t>(2 * n_ * sizeof(T)), static_cast<int64_t>(n_ * n_ * sizeof(T)), 0, 0}, {false, 0, 0, 0});

    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    uint16_t repeatTimes = (tileTaskCount * n_ * 2 + FP32_PER_VL - 1) / FP32_PER_VL;
    ComputeGradSigmoidVf<T>(gradSigmoidLocal1_, gradSigmoidLocal2_, fusedHPre2AndHPost2Local_, rsqrtLocal_,
        hcScaleLocal_, hcBaseLocal_, repeatTimes, tileTaskCount * n_ * 2);
}

template<typename T>
__aicore__ inline void MhcPreGradOneStageKernel<T>::ComputeGradRMSNorm(const int32_t tileTaskCount)
{
    Mul(gradRMSNormLocal_, rsqrtLocal_, rsqrtLocal_, tileTaskCount);
    Mul(gradRMSNormLocal_, gradRMSNormLocal_, rsqrtLocal_, tileTaskCount);
    Muls(gradRMSNormLocal_, gradRMSNormLocal_, -1.f / static_cast<T>(n_ * c_), tileTaskCount);
}

template<typename T>
__aicore__ inline T MhcPreGradOneStageKernel<T>::ComputeGradRsqrt(const int32_t i)
{
    Mul(gradSigmoidLocal1_[i * 2 * n_], fusedGradHPreAndGradHPostLocal_[i * 2 * n_],
        gradSigmoidLocal1_[i * 2 * n_], 2 * n_);
    Mul(gradSigmoidLocal2_[i * 2 * n_], fusedGradHPreAndGradHPostLocal_[i * 2 * n_],
        gradSigmoidLocal2_[i * 2 * n_], 2 * n_);

    Mul(tmpLocal_[i * 2 * n_], gradSigmoidLocal1_[i * 2 * n_], hcScaleLocal_, 2 * n_);
    Muls(gradH2Local_[i * (2 * n_ + n_ * n_)], tmpLocal_[i * 2 * n_], rsqrtLocal_.GetValue(i), 2 * n_);
    Mul(tmpLocal_[i * 2 * n_], tmpLocal_[i * 2 * n_], fusedHPre2AndHPost2Local_[i * 2 * n_], 2 * n_);

    ReduceSum<T>(tmpLocal_[i * 2 * n_], tmpLocal_[i * 2 * n_], tmpLocal_[i * 2 * n_], 2 * n_);

    T gradRsqrtVal = tmpLocal_.GetValue(i * 2 * n_) + gradRsqrtLocal_.GetValue(i);                
    return gradRsqrtVal;
}

template<typename T>
__aicore__ inline void MhcPreGradOneStageKernel<T>::ComputeGradXVector(const T gradRsqrtVal, const T gradRMSNormVal, const int32_t taskOffset)
{
    int8_t ping = 0;

    SetFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::MTE3_V>(1);
    for (int32_t i = 0; i < n_; i++) {
        WaitFlag<HardEvent::MTE3_V>(ping);

        ComputeGradXVf(gradXLocal_[ping * c_], xLocal_[ping1_ * n_ * c_ + i * c_], gradYLocal_[ping1_ * c_],
            preLocal_[ping1_ * 8 + i], gradRsqrtVal, gradRMSNormVal, c_, c1_);

        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);

        DataCopyPad(gradXGlobal_[taskOffset * n_ * c_ + i * c_], gradXLocal_[ping * c_], {static_cast<uint16_t>(1),
            static_cast<uint32_t>(c_ * sizeof(T)), 0, 0, 0});
        
        SetFlag<HardEvent::MTE3_V>(ping);
        ping = 1 - ping;
    }
    WaitFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(1);
}

/**
    约束：
    c: c >= 64 && c % 64 == 0
    n: n == 4
*/
template<typename T>
__aicore__ inline void MhcPreGradOneStageKernel<T>::Process()
{
    if ASCEND_IS_AIV {
        int8_t ping = 0;
        GetHcScaleAndHcBase();

        for (int32_t taskOffset = blkIdx_ * tile_; taskOffset < totalTasksAligned_; taskOffset += aivNum_ * tile_) {
            int32_t tileTaskCount = min(tile_, totalTasks_ - taskOffset);

            if (tileTaskCount > 0) {
                ComputeGradResHHat2(taskOffset, tileTaskCount);
                ComputeGradSigmoid(taskOffset, tileTaskCount);
                ComputeGradRMSNorm(tileTaskCount);

                DataCopyPad(fusedGradHPreAndGradHPostLocal_, gradPostGlobal_[taskOffset * n_], {static_cast<uint16_t>(tileTaskCount),
                    static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {true, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0, 0});

                SetFlag<HardEvent::V_MTE2>(0);
                SetFlag<HardEvent::V_MTE2>(1);
                for (int32_t i = 0; i < tileTaskCount; i++) {
                    WaitFlag<HardEvent::V_MTE2>(ping1_);

                    DataCopyPad(preLocal_[ping1_ * 8], preGlobal_[(taskOffset + i) * n_], {static_cast<uint16_t>(1),
                        static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
                    DataCopyPad(xLocal_[ping1_ * n_ * c_], xGlobal_[(taskOffset + i) * n_ * c_], {static_cast<uint16_t>(n_),
                        static_cast<uint32_t>(c_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
                    DataCopyPad(gradYLocal_[ping1_ * c_], gradYGlobal_[(taskOffset + i) * c_], {static_cast<uint16_t>(1),
                        static_cast<uint32_t>(c_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
                    
                    SetFlag<HardEvent::MTE2_V>(0);
                    WaitFlag<HardEvent::MTE2_V>(0);

                    ComputeGradPreVf<T>(fusedGradHPreAndGradHPostLocal_[i * 2 * n_], xLocal_[ping1_ * n_ * c_],
                        gradYLocal_[ping1_ * c_], n_, c_ / FP32_PER_VL);
                    T gradRsqrtVal = ComputeGradRsqrt(i);
                    ComputeGradXVector(gradRsqrtVal, gradRMSNormLocal_.GetValue(i), taskOffset + i);

                    SetFlag<HardEvent::V_MTE2>(ping1_);
                    ping1_ = 1 - ping1_;
                }
                WaitFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(1);

                DataCopyPad(gradH2Global_[taskOffset * (n_ * n_ + 2 * n_)], gradH2Local_, {static_cast<uint16_t>(tileTaskCount),
                    static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0});
            }

            CrossCoreSetFlag<0x2, PIPE_MTE3>(ping);
            ping = (ping + 1) % 10;

            if (tileTaskCount > 0) {
                if (tileTaskCount < tile_) {
                    Duplicate(gradSigmoidLocal1_[tileTaskCount * 2 * n_], 0.f, (tile_ - tileTaskCount) * 2 * n_);
                    Duplicate(gradSigmoidLocal2_[tileTaskCount * 2 * n_], 0.f, (tile_ - tileTaskCount) * 2 * n_);
                }

                // compute gradHcScale
                int32_t repeatTimes = (tileTaskCount * n_ * 2 + FP32_PER_REPEAT - 1) / FP32_PER_REPEAT;
                uint64_t preMask[] = {0X0F0F0F0F0F0F0F0F};
                uint64_t postMask[] = {0XF0F0F0F0F0F0F0F0};
                ReduceSum<T>(gradHcScaleLocal_, gradSigmoidLocal2_, gradSigmoidLocal2_, preMask, repeatTimes, 8);
                ReduceSum<T>(gradHcScaleLocal_[1], gradSigmoidLocal2_, gradSigmoidLocal2_, postMask, repeatTimes, 8);

                uint32_t shape[] = { static_cast<uint32_t>(tileTaskCount), static_cast<uint32_t>(2 * n_) };
                ReduceSum<T, Pattern::Reduce::RA, true>(gradHcBaseLocal_, gradSigmoidLocal1_, tmpLocal_.template ReinterpretCast<uint8_t>(), shape, true);

                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);

                SetAtomicAdd<T>();
                DataCopyPad(gradHcScaleGlobal_, gradHcScaleLocal_, {static_cast<uint16_t>(1), static_cast<uint32_t>(2 * sizeof(T)), 0, 0, 0});
                DataCopyPad(gradHcBaseGlobal_, gradHcBaseLocal_, {static_cast<uint16_t>(1), static_cast<uint32_t>(2 * n_ * sizeof(T)), 0, 0, 0});
                SetAtomicNone();
            }
        }
    }

    if ASCEND_IS_AIC {
        int8_t ping = 0;
        for (int32_t taskOffset = blkIdx_ * 2 * tile_; taskOffset < totalTasksAligned_; taskOffset += aicNum_ * 2 * tile_) {
            int32_t tileTaskCount = min(2 * tile_, totalTasks_ - taskOffset);
            CrossCoreWaitFlag<0x2>(ping);
            ping = (ping + 1) % 10;

            if (tileTaskCount > 0) {
                ProcessMatmul1(taskOffset, tileTaskCount);
                ProcessMatmul2(taskOffset, tileTaskCount);
            }
        }
    }
}


// mm1K_ = n_ * n_ + 2 * n_;
// mm1M_ = tile_ * 2;
// mm1N_ = n_ * c_;

// mm2K_ = batchSize_ * seqLength_;
// mm2M_ = n_ * n_ + 2 * n_;
// mm2N_ = n_ * c_;
template<typename T>
__aicore__ inline void MhcPreGradOneStageKernel<T>::ProcessMatmul1(const int32_t taskOffset, const int32_t mm1M)
{
    if (mm1M <= 0)
        return;

    mm1_.SetTensorA(gradH2Global_[taskOffset * mm1K_]);
    mm1_.SetTensorB(weightGlobal_);
    mm1_.SetHF32(false, 1);
    mm1_.SetOrgShape(mm1M, mm1N_, mm1K_);
    mm1_.SetSingleShape(mm1M, mm1N_, mm1K_);
    mm1_.template IterateAll<false> (gradXGlobal_[taskOffset * (n_ * c_)], 1);
    mm1_.End();
}

template<typename T>
__aicore__ inline void MhcPreGradOneStageKernel<T>::ProcessMatmul2(const int32_t taskOffset, const int32_t mm2K)
{
    if (mm2K <= 0)
        return;

    mm2_.SetTensorA(gradH2Global_[taskOffset * mm2M_], true);
    mm2_.SetTensorB(xGlobal_[taskOffset * mm2N_]);
    mm2_.SetHF32(false, 1);
    mm2_.SetOrgShape(mm2M_, mm2N_, mm2K);
    mm2_.SetSingleShape(mm2M_, mm2N_, mm2K);
    mm2_.template IterateAll<false> (gradWeightGlobal_, 1);
    mm2_.End();
}

#endif // MHC_PRE_GRAD_H