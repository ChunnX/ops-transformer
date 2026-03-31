#ifndef MHC_PRE_GRAD_H
#define MHC_PRE_GRAD_H
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "mhc_pre_grad_simd_vf.h"
#include "mhc_pre_grad_simt_vf.h"

using namespace AscendC;
using namespace MicroAPI;

namespace {
constexpr int32_t BYTE_SIZE_PER_BLOCK = 32;
constexpr int32_t ELEMENTS_SIZE_PER_BLOCK = BYTE_SIZE_PER_BLOCK / sizeof(float);

constexpr MatmulConfig MHC_PRE_GRAD_MM1_CFG = GetMDLConfig(false, false, 0, false, false, false, true);
constexpr MatmulConfig MHC_PRE_GRAD_MM2_CFG = GetMDLConfig(false, false, 0, false, false, false, true);
}

template<typename T>
class MhcPreGradKernel {
public:

    using A0Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using A1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;

    matmul::MatmulImpl<A0Type, BType, CType, CType, MHC_PRE_GRAD_MM1_CFG> mm1_;
    matmul::MatmulImpl<A1Type, BType, CType, CType, MHC_PRE_GRAD_MM2_CFG> mm2_;

    __aicore__ inline MhcPreGradKernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hc_fn, GM_ADDR pre, GM_ADDR grad_y, GM_ADDR grad_post, GM_ADDR grad_comb, GM_ADDR hc_scale,
        GM_ADDR hc_base, GM_ADDR h_hat2, GM_ADDR rsqrt, GM_ADDR sk_colSum, GM_ADDR sk_colNorm, GM_ADDR sk_rowSum, GM_ADDR sk_rowNorm,
        GM_ADDR grad_x, GM_ADDR grad_hc_fn, GM_ADDR grad_hc_scale, GM_ADDR grad_hc_base, GM_ADDR workspace, const MhcPreGradTilingData* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;
        blkIdx_ = GetBlockIdx();

        InitTiling(tilingData);
        InitGM(x, hc_fn, pre, grad_y, grad_post, grad_comb, hc_scale, hc_base, h_hat2, rsqrt, sk_colSum, sk_colNorm, sk_rowSum, sk_rowNorm, grad_x, grad_hc_fn,
            grad_hc_scale, grad_hc_base, workspace);
        InitGradPreStageBuffer();
    }
    __aicore__ inline void Process();

protected:
    int64_t blkIdx_, aivNum_, aicNum_;
    TPipe* pipe_;
    int64_t batchSize_, seqLength_, totalTasks_, totalTasksAligned_;
    int64_t c_, n_, c0_, c1_;
    int64_t tileGradY_, tilePre_, tileHHat2_;
    int64_t skIterCount_;
    int64_t mm1K_, mm1M_, mm1N_;
    int64_t mm2K_, mm2M_, mm2N_;
    T hcScalePre_, hcScalePost_, hcScaleRes_;

    TBuf<TPosition::VECCALC> gradYBuf_, preBuf_, xBuf_, gradXBuf_, gradPreBuf_, fusedGradHPreAndGradHPostBuf_, gradHResBuf_;
    TBuf<TPosition::VECCALC> hcBaseBuf_, rsqrtBuf_, gradRsqrtBuf_;
    TBuf<TPosition::VECCALC> fusedGradHPre2AndGradHPost2Buf_, fusedHPre2AndHPost2Buf_, hRes2Buf_;
    TBuf<TPosition::VECOUT> gradH2Buf_;
    
    LocalTensor<T> gradYLocal_, xLocal_, gradXLocal_;
    LocalTensor<T> preLocal_, gradPreLocal_;
    LocalTensor<T> gradHPreLocal_, gradHPostLocal_, fusedGradHPreAndGradHPostLocal_, gradHResLocal_;
    LocalTensor<T> hcBaseLocal_;
    LocalTensor<T> rsqrtLocal_, gradRsqrtLocal_;
    LocalTensor<T> fusedGradHPre2AndGradHPost2Local_, gradH2Local_;
    LocalTensor<T> fusedHPre2AndHPost2Local_, hRes2Local_;

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
        tileGradY_ = tilingData->tileGradY;             // stage 1
        skIterCount_ = tilingData->skIterCount;
        tilePre_ = (c0_ * tileGradY_ - 2) / c_ + 2;     // stage 1
        tileHHat2_ = tilingData->tileHHat2;
        totalTasks_ = batchSize_ * seqLength_ * c1_;
        
        if ASCEND_IS_AIC {
            mm1K_ = n_ * n_ + 2 * n_;
            mm1M_ = tileHHat2_ * 2;
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
        gradPreGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace));   // fp32
        gradH2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace) + batchSize_ * seqLength_ * n_);   // fp32
        weightGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(hc_fn));  // fp32
        gradXGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_x));   // fp32

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

            // 初始化 workspace 上的 gradPreGlobal_ 为 0 
            int32_t totalInitTaskCount = batchSize_ * seqLength_ * n_;
            int32_t bigCoreCount = totalInitTaskCount % aivNum_;
            int32_t curCoreInitTaskCount = totalInitTaskCount / aivNum_;
            int32_t globalMemOffset;
            if (blkIdx_ < bigCoreCount) {
                globalMemOffset = (curCoreInitTaskCount + 1) * blkIdx_;
                curCoreInitTaskCount = curCoreInitTaskCount + 1;
            } else {
                globalMemOffset = (curCoreInitTaskCount + 1) * bigCoreCount +
                                    curCoreInitTaskCount * (blkIdx_ - bigCoreCount);
            }
            GlobalTensor<T> gradpreGlobal = gradPreGlobal_[globalMemOffset];
            InitGlobalMemory(gradpreGlobal, curCoreInitTaskCount, static_cast<T>(0));
        }
    }

    __aicore__ inline void InitGradPreStageBuffer()
    {
        if ASCEND_IS_AIV {
            pipe_->InitBuffer(xBuf_, tileGradY_ * n_ * c0_ * sizeof(T)); // grad_y: n_ * c0_
            pipe_->InitBuffer(gradXBuf_, tileGradY_ * n_ * c0_ * sizeof(T)); // grad_y: n_ * c0_

            // Stage 1
            pipe_->InitBuffer(gradYBuf_, tileGradY_ * c0_ * sizeof(T)); // grad_y: tileGradY_ * c0_

            int32_t preUbSize = AlignUp(tilePre_ * n_ * sizeof(T), 32);
            pipe_->InitBuffer(preBuf_, preUbSize); // pre: tilePre_ * n
            pipe_->InitBuffer(gradPreBuf_, preUbSize); // pre: tilePre_ * n

            xLocal_ = xBuf_.Get<T>();
            gradYLocal_ = gradYBuf_.Get<T>();
            preLocal_ = preBuf_.Get<T>();
            gradPreLocal_ = gradPreBuf_.Get<T>();
            gradXLocal_ = gradXBuf_.Get<T>();
        }
    }

    __aicore__ inline void InitHhat2StageBuffer()
    {
        totalTasks_ = batchSize_ * seqLength_;
        totalTasksAligned_ = AlignUp(totalTasks_, aivNum_ * tileHHat2_);
        if ASCEND_IS_AIV {
            pipe_->Reset();
            pipe_->InitBuffer(fusedGradHPreAndGradHPostBuf_, tileHHat2_ * n_ * n_ * sizeof(T));
            pipe_->InitBuffer(rsqrtBuf_, tileHHat2_ * sizeof(T));
            pipe_->InitBuffer(gradRsqrtBuf_, tileHHat2_ * sizeof(T));
            pipe_->InitBuffer(hcBaseBuf_, (n_ * n_ + 2 * n_) * sizeof(T));
            pipe_->InitBuffer(fusedHPre2AndHPost2Buf_, tileHHat2_ * 2 * n_ * sizeof(T));
            pipe_->InitBuffer(hRes2Buf_, tileHHat2_ * n_ * n_ * sizeof(T));
            pipe_->InitBuffer(gradH2Buf_, tileHHat2_ * (n_ * n_ + 2 * n_) * sizeof(T));
            pipe_->InitBuffer(gradHResBuf_, tileHHat2_ * n_ * n_ * sizeof(T));
            
            rsqrtLocal_ = rsqrtBuf_.Get<T>();
            gradRsqrtLocal_ = gradRsqrtBuf_.Get<T>();
            hcBaseLocal_ = hcBaseBuf_.Get<T>();

            fusedGradHPreAndGradHPostLocal_ = fusedGradHPreAndGradHPostBuf_.Get<T>();
            gradHPostLocal_ = fusedGradHPreAndGradHPostLocal_[tileHHat2_ * n_ * 2];
            gradHPreLocal_ = fusedGradHPreAndGradHPostLocal_;

            fusedHPre2AndHPost2Local_ = fusedHPre2AndHPost2Buf_.Get<T>();
            hRes2Local_ = hRes2Buf_.Get<T>();
            gradH2Local_ = gradH2Buf_.Get<T>();
            gradHResLocal_ = gradHResBuf_.Get<T>();
        }
    }

    __aicore__ inline void ComputeGradPreAndGradX3();
    __aicore__ inline void ComputeGradPreAndGradPosHHat2(const int32_t taskOffset, const int32_t tileTaskCount);
    __aicore__ inline void ComputeGradResHHat2(const int32_t taskOffset, const int32_t tileTaskCount);
    __aicore__ inline void ComputeGradX1(const int32_t taskOffset, const int32_t tileTaskCount);
    __aicore__ inline void GetHcScaleAndHcBase();
    __aicore__ inline void ProcessMatmul1(const int32_t taskOffset, const int32_t mm1M);
    __aicore__ inline void ProcessMatmul2(const int32_t taskOffset, const int32_t mm2K);
};


template<typename T>
__aicore__ inline void MhcPreGradKernel<T>::GetHcScaleAndHcBase()
{
    // HcBase 常驻 UB
    hcScalePre_ = hcScaleGlobal_.GetValue(0);
    hcScalePost_ = hcScaleGlobal_.GetValue(1);
    hcScaleRes_ = hcScaleGlobal_.GetValue(2);
    DataCopyPad(hcBaseLocal_, hcBaseGlobal_, {static_cast<uint16_t>(1),
        static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
}

template<typename T>
__aicore__ inline void MhcPreGradKernel<T>::ComputeGradPreAndGradX3()
{
    SetFlag<HardEvent::V_MTE2>(0);
    for (int32_t gradYOffset = blkIdx_ * tileGradY_; gradYOffset < totalTasks_; gradYOffset += aivNum_ * tileGradY_) {
        
        // Copy in
        int32_t curLoopGradYTaskCount = min(tileGradY_, totalTasks_ - gradYOffset);
        int32_t bsStartOffset = gradYOffset / c1_;
        int32_t bsEndOffset = (gradYOffset + curLoopGradYTaskCount - 1) / c1_;
        int32_t curLoopPreTaskCount = (bsEndOffset - bsStartOffset + 1);

        WaitFlag<HardEvent::V_MTE2>(0);

        DataCopy(gradYLocal_, gradYGlobal_[gradYOffset * c0_], curLoopGradYTaskCount * c0_);
        DataCopyPad(preLocal_, preGlobal_[bsStartOffset * n_], {static_cast<uint16_t>(1),
            static_cast<uint32_t>(curLoopPreTaskCount * n_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        
        int32_t gradYLocalTaskOffset = 0;
        int32_t preLocalTaskOffset = 0;

        for (int32_t bsIdx = bsStartOffset; bsIdx <= bsEndOffset; bsIdx+=1) {
            int32_t startGradYOffset = max(gradYOffset, static_cast<int32_t>(bsIdx * c1_));
            int32_t endGradYOffset = min(gradYOffset + curLoopGradYTaskCount, static_cast<int32_t>((bsIdx + 1) * c1_));
            int32_t xGlobalOffset = (bsIdx * n_ * c_) + (startGradYOffset % c1_) * c0_;
            int32_t processGradYTaskCount = endGradYOffset - startGradYOffset;
            
            DataCopyPad(xLocal_[gradYLocalTaskOffset * c0_ * n_], xGlobal_[xGlobalOffset], {static_cast<uint16_t>(n_),
                static_cast<uint32_t>(processGradYTaskCount * c0_ * sizeof(T)), static_cast<int64_t>((c_ - processGradYTaskCount * c0_) * sizeof(T)), 0, 0}, {false, 0, 0, 0});

            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);

            ComputeGradPreAndGradX3Vf<T>(gradXLocal_[gradYLocalTaskOffset * c0_ * n_], gradPreLocal_[preLocalTaskOffset * n_], xLocal_[gradYLocalTaskOffset * c0_ * n_],
                gradYLocal_[gradYLocalTaskOffset * c0_], preLocal_[preLocalTaskOffset * n_], n_, processGradYTaskCount);

            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            
            DataCopyPad(gradXGlobal_[xGlobalOffset], gradXLocal_[gradYLocalTaskOffset * c0_ * n_],
                {static_cast<uint16_t>(n_), static_cast<uint32_t>(processGradYTaskCount * c0_ * sizeof(T)), 0,
                 static_cast<int64_t>((c_ - processGradYTaskCount * c0_) * sizeof(T)), 0});

            gradYLocalTaskOffset += processGradYTaskCount;
            preLocalTaskOffset += 1;
        }
        SetFlag<HardEvent::V_MTE2>(0);

        SetAtomicAdd<T>();
        DataCopyPad(gradPreGlobal_[bsStartOffset * n_], gradPreLocal_,
                {static_cast<uint16_t>(1), static_cast<uint32_t>(curLoopPreTaskCount * n_ * sizeof(T)), 0, 0, 0});
        SetAtomicNone();
    }
    WaitFlag<HardEvent::V_MTE2>(0);
}

template<typename T>
__aicore__ inline void MhcPreGradKernel<T>::ComputeGradPreAndGradPosHHat2(const int32_t taskOffset, const int32_t tileTaskCount)
{
    WaitFlag<HardEvent::V_MTE2>(0);
    DataCopyPad(gradHPreLocal_, gradPreGlobal_[taskOffset * n_], {static_cast<uint16_t>(tileTaskCount),
        static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {true, 0, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0});
    DataCopyPad(gradHPostLocal_, gradPostGlobal_[taskOffset * n_], {static_cast<uint16_t>(tileTaskCount),
        static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {true, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0, 0});
    
    DataCopyPad(fusedHPre2AndHPost2Local_, h2Global_[taskOffset * (n_ * n_ + 2 * n_)], {static_cast<uint16_t>(tileTaskCount),
        static_cast<uint32_t>(2 * n_ * sizeof(T)), static_cast<int64_t>(n_ * n_ * sizeof(T)), 0, 0}, {false, 0, 0, 0});
    DataCopyPad(rsqrtLocal_, rsqrtGlobal_[taskOffset], {static_cast<uint16_t>(1),
        static_cast<uint32_t>(tileTaskCount * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    Add(fusedGradHPreAndGradHPostLocal_, gradHPreLocal_, gradHPostLocal_, tileTaskCount * ELEMENTS_SIZE_PER_BLOCK);

    uint32_t thread_num = min(static_cast<uint32_t>(THREAD_NUM), static_cast<uint32_t>(tileTaskCount * n_ * 2));
    uint32_t xThreadCount = 2 * n_;
    uint32_t yThreadCount = thread_num / xThreadCount;
    LocalTensor<T> tmpLocal = gradHPostLocal_;
    Simt::VF_CALL<ComputeGradFusedGradH2PreAndGradH2PostSimt<T>>(Simt::Dim3{xThreadCount, yThreadCount},
        (__ubuf__ T*) gradH2Local_.GetPhyAddr(),
        (__ubuf__ T*) gradRsqrtLocal_.GetPhyAddr(),
        (__gm__ T*) gradHcScaleGlobal_.GetPhyAddr(),
        (__gm__ T*) gradHcBaseGlobal_.GetPhyAddr(),
        (__ubuf__ T*) fusedGradHPreAndGradHPostLocal_.GetPhyAddr(),
        (__ubuf__ T*) fusedHPre2AndHPost2Local_.GetPhyAddr(),
        (__ubuf__ T*) rsqrtLocal_.GetPhyAddr(),
        (__ubuf__ T*) hcBaseLocal_.GetPhyAddr(),
        (__ubuf__ T*) tmpLocal.GetPhyAddr(),  // gradHPostLocal_ 用作临时空间
        hcScalePre_, hcScalePost_, tileTaskCount, n_);
}

template<typename T>
__aicore__ inline void MhcPreGradKernel<T>::ComputeGradResHHat2(const int32_t taskOffset, const int32_t tileTaskCount)
{
    DataCopyPad(hRes2Local_, h2Global_[taskOffset * (n_ * n_ + 2 * n_) + 2 * n_], {static_cast<uint16_t>(tileTaskCount),
        static_cast<uint32_t>(n_ * n_ * sizeof(T)), static_cast<int64_t>(2 * n_ * sizeof(T)), 0, 0}, {false, 0, 0, 0});
    DataCopyPad(gradHResLocal_, gradHResGlobal_[taskOffset * n_ * n_], {static_cast<uint16_t>(tileTaskCount),
        static_cast<uint32_t>(n_ * n_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});

    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    
    uint32_t thread_num = min(static_cast<uint32_t>(THREAD_NUM), static_cast<uint32_t>(tileTaskCount * n_ * n_));
    uint32_t xThreadCount = n_ * n_;
    uint32_t yThreadCount = thread_num / xThreadCount;
    
    LocalTensor<T> tmpLocal = fusedGradHPreAndGradHPostLocal_;
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
        (__ubuf__ T*) tmpLocal.GetPhyAddr(),  // fusedGradHPreAndGradHPostLocal_ 用作临时空间
        hcScaleRes_, totalTasks_, taskOffset, tileTaskCount, skIterCount_, n_, 1e-6);

    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);

    DataCopyPad(gradH2Global_[taskOffset * (n_ * n_ + 2 * n_)], gradH2Local_, {static_cast<uint16_t>(tileTaskCount),
        static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0});
}

template<typename T>
__aicore__ inline void MhcPreGradKernel<T>::ComputeGradX1(const int32_t taskOffset, const int32_t tileTaskCount)
{
    using VECTOR_DTYPE = float4;
    uint32_t thread_num = min(static_cast<uint32_t>(THREAD_NUM), static_cast<uint32_t>(n_ * tileTaskCount * c_ / 4));

    Simt::VF_CALL<ComputeGradX1Simt<T, VECTOR_DTYPE>>(Simt::Dim3{thread_num},
        (__gm__ T*)  gradXGlobal_[taskOffset * n_ * c_].GetPhyAddr(),
        (__gm__ VECTOR_DTYPE*)  xGlobal_[taskOffset * n_ * c_].GetPhyAddr(),
        (__ubuf__ T*) gradRsqrtLocal_.GetPhyAddr(),
        (__ubuf__ T*) rsqrtLocal_.GetPhyAddr(),pre
        tileTaskCount, n_, c_);
    SetFlag<HardEvent::V_MTE2>(0);
}


/**
    约束：
    c: c >= 64 && c % 64 == 0
    n: n == 4
*/
template<typename T>
__aicore__ inline void MhcPreGradKernel<T>::Process()
{
    if ASCEND_IS_AIV {
        // 同步所有vector核
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x8);
        CrossCoreWaitFlag(0x8);

        ComputeGradPreAndGradX3();

        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x8);
        CrossCoreWaitFlag(0x8);
    }
        
    InitHhat2StageBuffer();
    
    if ASCEND_IS_AIV {
        GetHcScaleAndHcBase();

        int8_t ping = 0;
        SetFlag<HardEvent::V_MTE2>(0);
        for (int32_t taskOffset = blkIdx_ * tileHHat2_; taskOffset < totalTasksAligned_; taskOffset += aivNum_ * tileHHat2_) {
            int32_t tileTaskCount = min(tileHHat2_, totalTasks_ - taskOffset);

            if (tileTaskCount > 0) {
                ComputeGradPreAndGradPosHHat2(taskOffset, tileTaskCount);
                ComputeGradResHHat2(taskOffset, tileTaskCount);
            }

            CrossCoreSetFlag<0x2, PIPE_MTE3>(ping);
            ping = (ping + 1) % 10;

            if (tileTaskCount > 0) {
                ComputeGradX1(taskOffset, tileTaskCount);
            }
        }
        WaitFlag<HardEvent::V_MTE2>(0);
    }

    if ASCEND_IS_AIC {
        int8_t ping = 0;
        for (int32_t taskOffset = blkIdx_ * 2 * tileHHat2_; taskOffset < totalTasksAligned_; taskOffset += aicNum_ * 2 * tileHHat2_) {
            int32_t tileTaskCount = min(2 * tileHHat2_, totalTasks_ - taskOffset);
            CrossCoreWaitFlag<0x2>(ping);

            if (tileTaskCount > 0) {
                ProcessMatmul1(taskOffset, tileTaskCount);
                ProcessMatmul2(taskOffset, tileTaskCount);
            }

            ping = (ping + 1) % 10;
        }
    }
}

template<typename T>
__aicore__ inline void MhcPreGradKernel<T>::ProcessMatmul1(const int32_t taskOffset, const int32_t mm1M)
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
__aicore__ inline void MhcPreGradKernel<T>::ProcessMatmul2(const int32_t taskOffset, const int32_t mm2K)
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