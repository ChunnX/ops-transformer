#ifndef MHC_PRE_GRAD_H
#define MHC_PRE_GRAD_H
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "mhc_pre_sinkhorn_backward_simd_vf.h"
#include "mhc_pre_sinkhorn_backward_simt_vf.h"
#include <cstdint>

using namespace AscendC;
using namespace MicroAPI;

namespace {
constexpr int32_t BYTE_SIZE_PER_BLOCK = 32;
constexpr int32_t ELEMENTS_SIZE_PER_BLOCK = BYTE_SIZE_PER_BLOCK / sizeof(float);
constexpr int32_t BYTE_SIZE_PER_REPEAT = 256;
constexpr int32_t ELEMENTS_SIZE_PER_REPEAT = 256 / sizeof(float);
constexpr int32_t BLOCK_PER_REPEAT = 8;
constexpr uint64_t MASK_PRE = 0b0000111100001111000011110000111100001111000011110000111100001111;
constexpr uint64_t MASK_POST = 0b1111000011110000111100001111000011110000111100001111000011110000;

constexpr MatmulConfig MHC_PRE_GRAD_MM1_CFG = GetMDLConfig(false, false, 0, false, false, false, true);
constexpr MatmulConfig MHC_PRE_GRAD_MM2_CFG = GetMDLConfig(false, false, 0, false, false, false, true);
}

template<typename TYPE_X, typename T>
class MhcPreGradKernel {
public:

    using A0Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using A1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;

    matmul::MatmulImpl<A0Type, BType, CType, CType, MHC_PRE_GRAD_MM1_CFG> mm1_;
    matmul::MatmulImpl<A1Type, BType, CType, CType, MHC_PRE_GRAD_MM2_CFG> mm2_;// grad 

    __aicore__ inline MhcPreGradKernel() = default;
    // op.Init(x, phi, h_pre, grad_hin, grad_h_post, grad_h_res, alpha, bias, hc_before_norm, inv_rms, sum_out, norm_out,
    //         grad_x, grad_phi, grad_alpha, grad_bias, usrWorkspace, &tilingData, &pipe);
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hc_fn, GM_ADDR pre, GM_ADDR grad_y, GM_ADDR grad_post, GM_ADDR grad_comb, GM_ADDR hc_scale,
        GM_ADDR hc_base, GM_ADDR h_hat2, GM_ADDR rsqrt, GM_ADDR sum_out, GM_ADDR norm_out,
        GM_ADDR grad_x, GM_ADDR grad_hc_fn, GM_ADDR grad_hc_scale, GM_ADDR grad_hc_base, GM_ADDR workspace, const MhcPreGradTilingData* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;
        blkIdx_ = GetBlockIdx();

        InitTiling(tilingData);
        InitGM(x, hc_fn, pre, grad_y, grad_post, grad_comb, hc_scale, hc_base, h_hat2, rsqrt, sum_out, norm_out, grad_x, grad_hc_fn,
            grad_hc_scale, grad_hc_base, workspace);
        InitGradPreStageBuffer();
    }
    __aicore__ inline void Process();

protected:
    int64_t blkIdx_, aivNum_, aicNum_;
    TPipe* pipe_;
    int64_t batchSize_, seqLength_, totalTasks_, totalTasksAligned_, BSNN, BSN;
    int64_t c_, n_, c0_, c1_, cTail_, c0RepeatTime_, cTailAlign_, cTailBlockStride_, c1Align_;
    int64_t tileGradY_, tilePre_, tileHHat2_;
    int64_t skIterCount_;
    int64_t mm1K_, mm1M_, mm1N_;
    int64_t mm2K_, mm2M_, mm2N_;
    int64_t tileRepeatTimes_;
    float eps_;
    T hcScalePre_, hcScalePost_, hcScaleRes_;

    TBuf<TPosition::VECCALC> gradYBuf_, preBuf_, preBrcbBuf_, xBuf_, gradXBuf_, gradPreBuf_, fusedGradHPreAndGradHPostBuf_, gradHResBuf_;
    TBuf<TPosition::VECCALC> gradYCastBuf_, gradXCastBuf_, xCastBuf_;

    TBuf<TPosition::VECCALC> hcBaseBuf_, rsqrtBuf_, gradRsqrtBuf_;
    TBuf<TPosition::VECCALC> fusedGradHPre2AndGradHPost2Buf_, fusedHPre2AndHPost2Buf_, hRes2Buf_;
    TBuf<TPosition::VECOUT> gradH2Buf_;
    LocalTensor<TYPE_X> gradYLocal_, gradXLocal_, xLocal_;

    LocalTensor<T> xCastLocal_, gradYCastLocal_, gradXCastLocal_;
    LocalTensor<T> preLocal_, preBrcbLocal_, gradPreLocal_;
    LocalTensor<T> gradHPreLocal_, gradHPostLocal_, fusedGradHPreAndGradHPostLocal_, gradHResLocal_;
    LocalTensor<T> hcBaseLocal_;
    LocalTensor<T> rsqrtLocal_, gradRsqrtLocal_;
    LocalTensor<T> fusedGradHPre2AndGradHPost2Local_, gradhatLocal_;
    LocalTensor<T> hat2Local, hRes2Local_ h2Local_;
    LocalTensor<T> onesLocal_;

    GlobalTensor<T> xGlobal_, gradYGlobal_;
    GlobalTensor<T> preGlobal_;
    GlobalTensor<T> gradXGlobal_;
    GlobalTensor<T> gradPreGlobal_, gradPostGlobal_;
    GlobalTensor<T> hcScaleGlobal_, hcBaseGlobal_;
    GlobalTensor<T> rsqrtGlobal_;
    GlobalTensor<T> gradHcScaleGlobal_, gradHcBaseGlobal_;
    GlobalTensor<T> h2Global_;
    GlobalTensor<T> skNormGlobal_, skSumGlobal_;
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
        BSNN = batchSize_ * seqLength_ * n_ * n_;
        BSN = batchSize_ * seqLength_ * n_;
        c1Align_ = AlignUp(c_, c0_);
        cTail_ = tilingData->cTail;
        cTailAlign_ = AlignUp(cTail_, ELEMENTS_SIZE_PER_BLOCK);
        cTailBlockStride_ = c0_ / ELEMENTS_SIZE_PER_BLOCK - cTailAlign_ / ELEMENTS_SIZE_PER_BLOCK
        tileGradY_ = tilingData->tileGradY;             // stage 1
        skIterCount_ = tilingData->skIterCount;
        eps_ = tilingData->eps;
        tilePre_ = (c0_ * tileGradY_ - 2) / c_ + 2;     // stage 1
        tileHHat2_ = tilingData->tileHHat2;
        totalTasks_ = batchSize_ * seqLength_ * c1_;
        c0RepeatTime = c0_ / ELEMENTS_SIZE_PER_REPEAT;
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
        GM_ADDR hc_scale, GM_ADDR hc_base, GM_ADDR h_hat2, GM_ADDR rsqrt, GM_ADDR sum_out, GM_ADDR norm_out,
        GM_ADDR grad_x, GM_ADDR grad_hc_fn, GM_ADDR grad_hc_scale, GM_ADDR grad_hc_base, GM_ADDR workspace)
    {
        // cube 和 vector 共用
        xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE_X*>(x));               // fp32
        gradWeightGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_hc_fn));  // fp32
        gradPreGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace));   // fp32
        gradH2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace) + batchSize_ * seqLength_ * n_);   // fp32
        weightGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(hc_fn));  // fp32
        gradXGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_x));   // fp32

        if ASCEND_IS_AIV {
            preGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(pre));        // fp32
            gradYGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE_X*>(grad_y));   // fp32
            
            gradPostGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_post));   // fp32

            hcScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(hc_scale));   // fp32
            hcBaseGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(hc_base));   // fp32

            rsqrtGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(rsqrt));   // fp32
            gradHcScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_hc_scale));
            gradHcBaseGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_hc_base));

            h2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(h_hat2));   // fp32

            skNormGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(norm_out));   // fp32
            skSumGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(sum_out));   // fp32

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
            pipe_->InitBuffer(xBuf_, tileGradY_ * n_ * c0_ * sizeof(TYPE_X)); // grad_y: n_ * c0_
            pipe_->InitBuffer(gradXBuf_, tileGradY_ * n_ * c0_ * sizeof(TYPE_X)); // grad_y: n_ * c0_

            // Stage 1
            pipe_->InitBuffer(gradYBuf_, tileGradY_ * c0_ * sizeof(TYPE_X)); // grad_y: tileGradY_ * c0_

            int32_t preUbSize = AlignUp(tilePre_ * n_ * sizeof(T), 32);
            pipe_->InitBuffer(preBuf_, preUbSize); // pre: tilePre_ * n
            pipe_->InitBuffer(preBrcbBuf_, tilePre_ * n_ * BYTE_SIZE_PER_BLOCK); // pre: tilePre_ * n
            pipe_->InitBuffer(xCastBuf_, tileGradY_ * n_ * c0_ * sizeof(float)); // grad_y: n_ * c0_
            pipe_->InitBuffer(gradYCastBuf_, tileGradY_ * c0_ * sizeof(float)); // grad_y: tileGradY_ * c0_
            pipe_->InitBuffer(gradXCastBuf_, tileGradY_ * c0_ * sizeof(float)); // grad_y: tileGradY_ * c0_

            pipe_->InitBuffer(gradPreBuf_, preUbSize); // pre: tilePre_ * n
            pipe_->InitBuffer(tmpBuf_, tilePre_ * n_ * BYTE_SIZE_PER_REPEAT); // pre: tilePre_ * n
            pipe_->InitBuffer(tmpPrePostBuf_, tilePre_ * AlignUp(n_ * n_ + 2 * n_, ELEMENTS_SIZE_PER_BLOCK)) * sizeof(float); // pre: tilePre_ * n
            pipe_->InitBuffer(onesBuf_, tilePre_ * n_ * 2 * sizeof(float)); // pre: tilePre_ * n

            xLocal_ = xBuf_.Get<TYPE_X>();
            gradYLocal_ = gradYBuf_.Get<TYPE_X>();
            gradXLocal_ = gradXBuf_.Get<TYPE_X>();
            tmpLocal_ = tmpBuf_.Get<T>();
            onesLocal_ = onesBuf_.Get<T>();
            Duplicate(onesLocal_, 1.f, tilePre_ * n_ * 2);
            Duplicate(dScaleLocal, 0.f, tilePre_ * (2 * n_ + n_ * n_));

            
            xCastLocal_ = xCastBuf_.Get<float>();
            gradYCastLocal_ = gradYCastBuf_.Get<float>();
            gradXCastLocal_ = gradXCastBuf_.Get<float>();
            preLocal_ = preBuf_.Get<T>();
            preBrcbLocal_ = preBrcbBuf_.Get<T>();
            gradPreLocal_ = gradPreBuf_.Get<T>();
            dPrePostTempLocal_ = tmpPrePostBuf_.Get<T>();
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
            pipe_->InitBuffer(h2Buf_, tileHHat2_ * (2 * n_ + n_ * n_ ) * sizeof(T));
            pipe_->InitBuffer(rsqrtBrcbBuf_, tileHHat2_ * (2 * n_ + n_ * n_ ) * sizeof(T));

            pipe_->InitBuffer(hRes2Buf_, tileHHat2_ * n_ * n_ * sizeof(T));
            pipe_->InitBuffer(gradH2Buf_, tileHHat2_ * (n_ * n_ + 2 * n_) * sizeof(T));
            pipe_->InitBuffer(gradHResBuf_, tileHHat2_ * n_ * n_ * sizeof(T));
            
            rsqrtLocal_ = rsqrtBuf_.Get<T>();
            gradRsqrtLocal_ = gradRsqrtBuf_.Get<T>();
            hcBaseLocal_ = hcBaseBuf_.Get<T>();

            fusedGradHPreAndGradHPostLocal_ = fusedGradHPreAndGradHPostBuf_.Get<T>();
            gradHPostLocal_ = fusedGradHPreAndGradHPostLocal_[tileHHat2_ * n_ * 2];
            gradHPreLocal_ = fusedGradHPreAndGradHPostLocal_;
            rsqrtBrcbLocal_ = rsqrtBrcbBuf_.Get<T>();
            hat2Local = fusedHPre2AndHPost2Buf_.Get<T>();
            hRes2Local_ = hRes2Buf_.Get<T>();
            gradhatLocal_ = gradH2Buf_.Get<T>();
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


template<typename TYPE_X, typename T>
__aicore__ inline void MhcPreGradKernel<T>::GetHcScaleAndHcBase()
{
    // HcBase 常驻 UB
    hcScalePre_ = hcScaleGlobal_.GetValue(0);
    hcScalePost_ = hcScaleGlobal_.GetValue(1);
    hcScaleRes_ = hcScaleGlobal_.GetValue(2);
    Scalelocal_.
    Duplicate(Scalelocal_[8], hcScaleRes_, 8);
    Duplicate(Scalelocal_, hcScalePost_, 8);
    PipeBarrier<PIPE_V>();

    Duplicate(Scalelocal_, hcScalePre_, 4);

    DataCopyPad(hcBaseLocal_, hcBaseGlobal_, {static_cast<uint16_t>(1),
        static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
}

template<typename TYPE_X, typename T>
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
template<typename TYPE_X, typename T>
__aicore__ inline void BroadCast4to8(const int32_t taskOffset, const int32_t tileTaskCount)

template<typename TYPE_X, typename T>
__aicore__ inline void MhcPreGradKernel<T>::SinkhornGrad(const int32_t taskOffset, const int32_t tileTaskCount)
{
    WaitFlag<HardEvent::V_MTE2>(0);
    // DataCopyPad(gradHPostLocal_, gradHResGlobal_[taskOffset * n_ * n_], {static_cast<uint16_t>(1),
    //     static_cast<uint32_t>(tileTaskCount  * n_ * n_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(gradHResLocal_, gradHResGlobal_[taskOffset * n_ * n_], {static_cast<uint16_t>(tileTaskCount  * n_),
        static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {true, 0, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0});
    int64_t iterRowNormOffset = (skIterCount_ - 1) * BSNN + taskOffset * n_ * n_;
    int64_t iterColNormOffset = skIterCount_ * BSNN + (skIterCount_ - 1) * BSNN + taskOffset * n_ * n_;

    int64_t iterRowSumOffset = (skIterCount_ - 1) * BSN + taskOffset * n_;
    int64_t iterColSumOffset = skIterCount_ * BSN + (skIterCount_ - 1) * BSN + taskOffset * n_;

    for (int32_t iter = skIterCount_ - 1; iter > 0; iter--) {
        // DataCopyPad(skColNormLocal_, skNormGlobal_[iterColNormOffset], {static_cast<uint16_t>(1),
        //     static_cast<uint32_t>(tileTaskCount  * n_ * n_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        // DataCopyPad(skColSumLocal_, skSumGlobal_[iterColSumOffset], {static_cast<uint16_t>(1),
        //     static_cast<uint32_t>(tileTaskCount  * n_ * n_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});   
        // DataCopyPad(skColNormLocal_, skNormGlobal_[iterColNormOffset], {static_cast<uint16_t>(tileTaskCount  * n_),
        //     static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {true, 0, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0});
        DataCopyPad(skRowNormLocal_, skNormGlobal_[iterRowNormOffset], {static_cast<uint16_t>(tileTaskCount  * n_),
            static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {true, 0, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0});
        DataCopyPad(skColSumLocal_, skSumGlobal_[iterColSumOffset], {static_cast<uint16_t>(tileTaskCount),
            static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {true, 0, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0});
        PipeBarrier<PIPE_V>();
            sw
        Adds(skColSumLocal_, skColSumLocal_, eps_,  tileTaskCount * ELEMENTS_SIZE_PER_BLOCK);
        PipeBarrier<PIPE_V>();

        for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
            Div(gradHResTempLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK], gradHResLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK],
                skColSumLocal_, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,{n_, n_, 1, n_ * 8, n_ * 8, 8});
        }
        PipeBarrier<PIPE_V>(); 
        gradHResTempLocal_
        gradHResTemp2Local_
        // xg * normed
        Mul(gradHResTemp2Local_, gradHResLocal_, skRowNormLocal_, tileTaskCount * n_ * ELEMENTS_SIZE_PER_BLOCK);
        PipeBarrier<PIPE_V>();
        // (sum + eps) ** 2
        Mul(skColSumLocal_, skColSumLocal_, skColSumLocal_, tileTaskCount * ELEMENTS_SIZE_PER_BLOCK);
        // bs * n *8 reduce bs * 1 *8 out : value other other other value other other other
        // sum(xg * normed )
        for (int32_t loopIdN = 1; loopIdN < n_; loopIdN += 1) {
            Add(gradHResTemp2Local_, gradHResTemp2Local_, 
                gradHResTemp2Local_[loopIdN * ELEMENTS_SIZE_PER_BLOCK], ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_, {n_, n_, n_, n_ * 8, n_ * 8, n_ * 8});
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_V>();
        // sum(xg * normed )/ (sum + eps) ** 2
        Div(gradHResTemp2Local_, gradHResTemp2Local_, skColSumLocal_,ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_, {n_, n_, 1, n_ * 8, n_ * 8, 8});
        PipeBarrier<PIPE_V>();
        // for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
        //     Div(gradHResTemp2Local_[loopIdN * ELEMENTS_SIZE_PER_BLOCK], gradHResTempLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK],
        //         skColSumLocal_, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_ * n_,{n_, n_, n_, n_ * 8, n_ * 8, n_ * 8});
        // }
       for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
            Sub(gradHResLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK], gradHResLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK],
                gradHResTemp2Local_, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,{n_, n_, n_, n_ * 8, n_ * 8, n_ * 8});
        }
        PipeBarrier<PIPE_V>();



        AscendC::BlockReduceSum<half>(skColNormLocal_, skColNormLocal_, tileRepeatTimes_, MASK_PRE, 1, 1, 8);

        

        DataCopyPad(skRowSumLocal_, skSumGlobal_[iterRowSumOffset], {static_cast<uint16_t>(1),
            static_cast<uint32_t>(tileTaskCount  * n_ * sizeof(T)), 0, 0, 0}, {true, 0, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0});
        // sw
        // bsn -> bsn8
        Adds(skRowSumLocal_, skRowSumLocal_, eps_,  tileTaskCount * n_);
        PipeBarrier<PIPE_V>();

        Brcb(gradHResTemp2Local_, skRowSumLocal_, tileTaskCount / 2, {1,8});
        PipeBarrier<PIPE_V>();
        Div(gradHResTempLocal_, gradHResLocal_, gradHResTemp2Local_, tileTaskCount * ELEMENTS_SIZE_PER_BLOCK);
        Mul(skRowNormLocal_, gradHResLocal_, skRowNormLocal_, tileTaskCount * n_ * ELEMENTS_SIZE_PER_BLOCK);
        PipeBarrier<PIPE_V>();
        Mul(skRowNormLocal_, skRowNormLocal_, gradHResTemp2Local_, tileTaskCount * n_ * ELEMENTS_SIZE_PER_BLOCK);
        PipeBarrier<PIPE_V>();
        AscendC::BlockReduceSum<half>(skRowNormLocal_, skRowNormLocal_, tileRepeatTimes_ * n_, MASK_PRE, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Brcb(gradHResTemp2Local_, skRowNormLocal_, tileTaskCount / 2, {1,8});
        PipeBarrier<PIPE_V>();
        Sub(gradHResLocal_, gradHResTempLocal_, gradHResTemp2Local_, tileTaskCount * n_ * ELEMENTS_SIZE_PER_BLOCK);
        iterRowNormOffset = iterRowNormOffset - BSNN;
        iterRowSumOffset = iterRowSumOffset - BSN; 
        iterColSumOffset = iterColSumOffset - BSN;
    }
    DataCopyPad(skRowNormLocal_, skNormGlobal_[iterRowNormOffset], {static_cast<uint16_t>(tileTaskCount  * n_),
        static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {true, 0, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0});
    DataCopyPad(skColSumLocal_, skSumGlobal_[iterColSumOffset], {static_cast<uint16_t>(tileTaskCount),
        static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {true, 0, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0});
    PipeBarrier<PIPE_V>();
        sw
    Adds(skColSumLocal_, skColSumLocal_, eps_,  tileTaskCount * ELEMENTS_SIZE_PER_BLOCK);
    PipeBarrier<PIPE_V>();

    for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
        Div(gradHResTempLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK], gradHResLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK],
            skColSumLocal_, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,{n_, n_, 1, n_ * 8, n_ * 8, 8});
    }
    PipeBarrier<PIPE_V>(); 
    gradHResTempLocal_
    gradHResTemp2Local_
    // xg * normed
    Mul(gradHResTemp2Local_, gradHResLocal_, skRowNormLocal_, tileTaskCount * n_ * ELEMENTS_SIZE_PER_BLOCK);
    PipeBarrier<PIPE_V>();
    // (sum + eps) ** 2
    Mul(skColSumLocal_, skColSumLocal_, skColSumLocal_, tileTaskCount * ELEMENTS_SIZE_PER_BLOCK);
    // bs * n *8 reduce bs * 1 *8 out : value other other other value other other other
    // sum(xg * normed )
    for (int32_t loopIdN = 1; loopIdN < n_; loopIdN += 1) {
        Add(gradHResTemp2Local_, gradHResTemp2Local_, 
            gradHResTemp2Local_[loopIdN * ELEMENTS_SIZE_PER_BLOCK], ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_, {n_, n_, n_, n_ * 8, n_ * 8, n_ * 8});
        PipeBarrier<PIPE_V>();
    }
    PipeBarrier<PIPE_V>();
    // sum(xg * normed )/ (sum + eps) ** 2
    Div(gradHResTemp2Local_, gradHResTemp2Local_, skColSumLocal_,ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_, {n_, n_, 1, n_ * 8, n_ * 8, 8});
    PipeBarrier<PIPE_V>();
    // for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
    //     Div(gradHResTemp2Local_[loopIdN * ELEMENTS_SIZE_PER_BLOCK], gradHResTempLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK],
    //         skColSumLocal_, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_ * n_,{n_, n_, n_, n_ * 8, n_ * 8, n_ * 8});
    // }
    for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
        Sub(gradHResLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK], gradHResLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK],
            gradHResTemp2Local_, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,{n_, n_, n_, n_ * 8, n_ * 8, n_ * 8});
    }
    PipeBarrier<PIPE_V>();
    Mul(gradHResTempLocal_, gradHResLocal_, skRowNormLocal_, tileTaskCount * n_ * ELEMENTS_SIZE_PER_BLOCK);
    PipeBarrier<PIPE_V>();
    AscendC::BlockReduceSum<half>(gradHResTempLocal_, gradHResTempLocal_, tileRepeatTimes_ * n_, MASK_PRE, 1, 1, 8);
    PipeBarrier<PIPE_V>();
    Brcb(gradHResTemp2Local_, gradHResTempLocal_, tileTaskCount / 2, {1,8});
    PipeBarrier<PIPE_V>();
    Sub(gradHResLocal_, gradHResLocal_, gradHResTemp2Local_, tileTaskCount * n_ * ELEMENTS_SIZE_PER_BLOCK);
    PipeBarrier<PIPE_V>();
    Mul(gradHResLocal_, gradHResLocal_, gradHResTempLocal_, tileTaskCount * n_ * ELEMENTS_SIZE_PER_BLOCK);
    PipeBarrier<PIPE_V>();
    //4v 4o 4v 40 -> 4v 4v
    Cast(gradHResTempLocal_.ReinterpretCast<AscendC::INT64>(), gradHResLocal_ template.ReinterpretCast<AscendC::INT32>(),tileTaskCount * n_ * ELEMENTS_SIZE_PER_BLOCK);
    PipeBarrier<PIPE_V>();

    Copy(gradHResLocal_, gradHResTempLocal_, ELEMENTS_SIZE_PER_REPEAT,n_ * tileRepeatTimes_ / 2, {1, 2, 8, 16});
    PipeBarrier<PIPE_V>();

    Cast(gradHResLocal_.ReinterpretCast<AscendC::INT32>(), gradHResLocal_ template.ReinterpretCast<AscendC::INT64>(),tileTaskCount * n_ * ELEMENTS_SIZE_PER_BLOCK);
    
    PipeBarrier<PIPE_V>();

}

template<typename TYPE_X, typename T>
__aicore__ inline void MhcPreGradKernel<T>::ComputeGradPreAndGradPosHHat2(const int32_t taskOffset, const int32_t tileTaskCount)
{
    WaitFlag<HardEvent::V_MTE2>(0);
    SinkhornGrad(taskOffset, tileTaskCount);
    for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
        Copy(dhatLocal[ELEMENTS_SIZE_PER_BLOCK + loopIdN * ELEMENTS_SIZE_PER_BLOCK], gradHResLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK],
            ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,{n_ + 1, n_, (n_ + 1) * 8, n_ * 8});
    }
    // DataCopyPad(gradHPreLocal_, gradPreGlobal_[taskOffset * n_], {static_cast<uint16_t>(tileTaskCount),
    //     static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {true, 0, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0});
    DataCopyPad(gradHPostLocal_, gradPostGlobal_[taskOffset * n_], {static_cast<uint16_t>(tileTaskCount),
        static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {true, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0, 0});
    
    DataCopyPad(hat2Local, h2Global_[taskOffset * (n_ * n_ + 2 * n_)], {static_cast<uint16_t>(tileTaskCount),
        static_cast<uint32_t>( (n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(rsqrtLocal_, rsqrtGlobal_[taskOffset], {static_cast<uint16_t>(1),
        static_cast<uint32_t>(tileTaskCount * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    PipeBarrier<PIPE_V>();
    // GRAD_PRE*1_POST*2
    Axpy(dPrePostTempLocal_, gradHPostLocal_,float(2), tileTaskCount * 2 * n);
    const uint32_t srcShape[2] = {tileTaskCount * n_, 1}; // 源数据shape
    const uint32_t dstShape[2] = {tileTaskCount * n_, n_ * n_ + 2 * n};// broadcast数据shape
    PipeBarrier<PIPE_V>();

    AscendC::Broadcast<float, 2, 1>(rsqrtbrcbLocal_, rsqrtLocal_, dstShape, srcShape);
    PipeBarrier<PIPE_V>();
    // GRAD_HAT_RSQRT
    Mul(hat2Scale, Scalelocal_,hat2Local ,ELEMENTS_SIZE_PER_REPEAT,tileRepeatTimes_,{3, 0, 1, 24, 0, 8});
    Mul(hat2Scale[8], Scalelocal_[8],hat2Local[8] ,ELEMENTS_SIZE_PER_REPEAT,tileRepeatTimes_,{3, 0, 1, 24, 0, 8});
    Mul(hat2Scale[16], Scalelocal_[16],hat2Local[16] ,ELEMENTS_SIZE_PER_REPEAT,tileRepeatTimes_,{3, 0, 1, 24, 0, 8});
    PipeBarrier<PIPE_V>();

    Add(hatBeforeNormLocal, hcBaseLocal_,hat2Scale ,ELEMENTS_SIZE_PER_REPEAT,tileRepeatTimes_,{3, 0, 1, 24, 0, 8});
    Add(hatBeforeNormLocal[8], hcBaseLocal_[8],hat2Scale[8] ,ELEMENTS_SIZE_PER_REPEAT,tileRepeatTimes_,{3, 0, 1, 24, 0, 8});
    Add(hatBeforeNormLocal[16], hcBaseLocal_[16],hat2Scale[16] ,ELEMENTS_SIZE_PER_REPEAT,tileRepeatTimes_,{3, 0, 1, 24, 0, 8});

    PipeBarrier<PIPE_V>();

    Mul(hatLocal, hatBeforeNormLocal,rsqrtbrcbLocal_ ,(n_ * n_ + 2 * n) * tileTaskCount);
    PipeBarrier<PIPE_V>();
    // 正向pre post sigmoid
    Muls(sigmoidTmp, hatLocal, -1, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_, {1,3,8,24});
    PipeBarrier<PIPE_V>();

    Exp(sigmoidTmp, sigmoidTmp, (2 * n) * tileTaskCount);
    PipeBarrier<PIPE_V>();

    adds(sigmoidTmp, sigmoidTmp, float(1), (2 * n) * tileTaskCount);
    PipeBarrier<PIPE_V>();
    div(sigmoidTmp, onesLocal_, sigmoidTmp, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,{1, 0, 1, 8, 0, 8});
    // 正向pre post sigmoidgrad x_sigmoid * (1 - x_sigmoid) * grad_output
    PipeBarrier<PIPE_V>();
    Mul(sigmoidTmp2, sigmoidTmp, sigmoidTmp, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,{1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    Sub(sigmoidTmp2, sigmoidTmp, sigmoidTmp2, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,{1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    
    Mul(dhatLocal, dPrePostTempLocal_, sigmoidTmp2, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,{3, 1, 1, 24, 8, 8});
    PipeBarrier<PIPE_V>();

    // 不同批次dbias累加


    PipeBarrier<PIPE_V>();
    add(dBias, dhatLocal, dBias, (n_ * n_ + 2 * n) * tileTaskCount);
    // dRsqrt = dhatLocal * beforenorm
    mul(dRsqrtLocal, dhatLocal, hatBeforeNormLocal, (n_ * n_ + 2 * n) * tileTaskCount);
    PipeBarrier<PIPE_V>();
    // rmsnormgrad (- (rsqrt ** 3)) * grad_rsqrt / float(NC)
    WholeReduceSum(dRsqrtLocal, dRsqrtLocal, (n_ * n_ + 2 * n), tileTaskCount, 1, 1, 3);// dst repeat stride  srcblkstride srcrepstride (n*2 +2n)/8
    PipeBarrier<PIPE_V>();
    mul(rsqrtTempLocal_, rsqrtLocal_, rsqrtLocal_, tileTaskCount);
    PipeBarrier<PIPE_V>();
    mul(rsqrtTempLocal_, rsqrtTempLocal_, rsqrtLocal_, tileTaskCount);
    PipeBarrier<PIPE_V>();
    mul(dRsqrtLocal, dRsqrtLocal, rsqrtTempLocal_, tileTaskCount);
    PipeBarrier<PIPE_V>();
    muls(dRsqrtLocal, dRsqrtLocal, -1 / (n_ * c_),  tileTaskCount);

    //dscale = dhatLocal * hat pre * rsqrt
    //dhat2 = dhatLocal  * rsqrt  * scale
    Mul(dhatBeforeNormLocal, rsqrtbrcbLocal_,dhatLocal ,(n_ * n_ + 2 * n) * tileTaskCount);
    PipeBarrier<PIPE_V>();
    // 不同批次dScale累加
    MulAddDst(dScaleLocal, hat2Local,dhatBeforeNormLocal ,(n_ * n_ + 2 * n) * tileTaskCount);

    Mul(dhat2Local, Scalelocal_,dhatBeforeNormLocal ,ELEMENTS_SIZE_PER_REPEAT,tileRepeatTimes_,{3, 0, 1, 24, 0, 8});
    Mul(dhat2Local[8], Scalelocal_[8],dhatBeforeNormLocal[8] ,ELEMENTS_SIZE_PER_REPEAT,tileRepeatTimes_,{3, 0, 1, 24, 0, 8});
    Mul(dhat2Local[16], Scalelocal_[16],dhatBeforeNormLocal[16] ,ELEMENTS_SIZE_PER_REPEAT,tileRepeatTimes_,{3, 0, 1, 24, 0, 8});
    PipeBarrier<PIPE_V>();
    // reduce dscaleres
    {

    // 调换顺序
    Add(dScaleLocal[8], dScaleLocal[8], dScaleLocal[16],ELEMENTS_SIZE_PER_REPEAT,tileRepeatTimes_,{3, 3, 3, 24, 24, 24});
    // 和tileTaskCount相关
    PipeBarrier<PIPE_V>();
    // tileHHat2_ 必须8 *2的幂次
    for (int32_t bsCount = tileHHat2_ / 2; bsCount > 4; bsCount = bsCount / 2) {
        Add(dScaleLocal, dScaleLocal, dScaleLocal[bsCount * (n_ * n_ + 2 * n)], bsCount * (n_ * n_ + 2 * n));
    }
    PipeBarrier<PIPE_V>();

    WholeReduceSum(dscale, dScaleLocal, MASK_PRE, 1, 1, 3, 8);// dst repeat stride  srcblkstride srcrepstride
    WholeReduceSum(dscale[1], dScaleLocal, MASK_POST, 1, 1, 3, 8);// dst repeat stride  srcblkstride srcrepstride
    WholeReduceSum(dscale[2], dScaleLocal[8], ELEMENTS_SIZE_PER_REPEAT, 1, 1, 3, 8);// dst repeat stride  srcblkstride srcrepstride
    }





    // 
    // uint32_t thread_num = min(static_cast<uint32_t>(THREAD_NUM), static_cast<uint32_t>(tileTaskCount * n_ * 2));
    // uint32_t xThreadCount = 2 * n_;
    // uint32_t yThreadCount = thread_num / xThreadCount;
    // LocalTensor<T> tmpLocal = gradHPostLocal_;
    // Simt::VF_CALL<ComputeGradFusedGradH2PreAndGradH2PostSimt<T>>(Simt::Dim3{xThreadCount, yThreadCount},
    //     (__ubuf__ T*) gradhatLocal_.GetPhyAddr(),
    //     (__ubuf__ T*) gradRsqrtLocal_.GetPhyAddr(),
    //     (__gm__ T*) gradHcScaleGlobal_.GetPhyAddr(),
    //     (__gm__ T*) gradHcBaseGlobal_.GetPhyAddr(),
    //     (__ubuf__ T*) fusedGradHPreAndGradHPostLocal_.GetPhyAddr(),
    //     (__ubuf__ T*) hat2Local.GetPhyAddr(),
    //     (__ubuf__ T*) rsqrtLocal_.GetPhyAddr(),
    //     (__ubuf__ T*) hcBaseLocal_.GetPhyAddr(),
    //     (__ubuf__ T*) tmpLocal.GetPhyAddr(),  // gradHPostLocal_ 用作临时空间
    //     hcScalePre_, hcScalePost_, tileTaskCount, n_);
}

template<typename TYPE_X, typename T>
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
        (__ubuf__ T*) gradhatLocal_.GetPhyAddr(),
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

    DataCopyPad(gradH2Global_[taskOffset * (n_ * n_ + 2 * n_)], gradhatLocal_, {static_cast<uint16_t>(tileTaskCount),
        static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0});
}

template<typename TYPE_X, typename T>
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

template<typename TYPE_X, typename T>
__aicore__ inline void MhcPreGradKernel<T>::ComputeGradPreAndGradX3()
{
    SetFlag<HardEvent::V_MTE2>(0);
    Duplicate(tmpLocal_, 0.f, tilePre_ * n_ * BYTE_SIZE_PER_REPEAT);

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

template<typename TYPE_X, typename T>
__aicore__ inline void MhcPreGradKernel<T>::ComputeGradPreAndGradX3BS(const int32_t taskOffset, const int32_t tileTaskCount)

template<typename TYPE_X, typename T>
__aicore__ inline void MhcPreGradKernel<T>::ComputeGradPreAndGradX3BS(const int32_t taskBSOffset, const int32_t tileTaskBSCount)
{
    WaitFlag<HardEvent::V_MTE2>(0);
    Duplicate(tmpLocal_, 0.f, tilePre_ * n_ * BYTE_SIZE_PER_REPEAT);

    DataCopyPad(preLocal_, preGlobal_[taskBSOffset * n_], {static_cast<uint16_t>(1),
        static_cast<uint32_t>(tileTaskBSCount * n_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    const uint32_t srcShape[2] = {tileTaskBSCount * n_, 1}; // 源数据shape
    const uint32_t dstShape[2] = {tileTaskBSCount * n_, ELEMENTS_SIZE_PER_BLOCK};// broadcast数据shape
    AscendC::Broadcast<float, 2, 1>(preBrcbLocal_, preLocal_, dstShape, srcShape)
    for (int32_t loopIdC = 0; loopIdC < c1Align_; loopIdC+=1) {
        int64_t copyLen = c0_;
        bool isPad = false;
        uint8_t padLen = 0;
        if (loopIdC == c1_) {
            isPad = true;
            copyLen = cTail_;
            padLen = static_cast<uint8_t>(cTailAlign_ - cTail_);
            Duplicate(gradYLocal_, 0.f, c0_ - cTailAlign_, tileTaskBSCount, 1, c0_ / ELEMENTS_SIZE_PER_BLOCK);
            Duplicate(xLocal_, 0.f, c0_ - cTailAlign_, tileTaskBSCount * n_, 1, c0_ / ELEMENTS_SIZE_PER_BLOCK);
        }
        DataCopyPad(gradYLocal_, gradYGlobal_[taskBSOffset * c_ + c0_ * loopIdC], {static_cast<uint16_t>(tileTaskBSCount),
            static_cast<uint32_t>(copyLen * sizeof(TYPE_X)), static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0, 0}, {isPad, 0, padLen, 0});
        DataCopyPad(xLocal_, xGlobal_[taskBSOffset * n_ * c_ + c0_ * loopIdC], {static_cast<uint16_t>(n_ * tileTaskBSCount),
            static_cast<uint32_t>(copyLen * sizeof(TYPE_X)), static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0, 0}, {isPad, 0, padLen, 0});
        sat_waitflag
        PipeBarrier<PIPE_V>();
        sat_waitflag

        Cast(xCastLocal_, xLocal_, RoundMode::CAST_NONE,c0_ * n_ * tileTaskBSCount);
        sat_waitflag

        DataCopyPad(xWorkspaceGlobal_[taskBSOffset * n_ * c_ + c0_ * loopIdC], xCastLocal_, {static_cast<uint16_t>(n_ * tileTaskBSCount),
            static_cast<uint32_t>(copyLen * sizeof(float)), static_cast<uint32_t>((c_ - copyLen) * sizeof(float)), 0, 0});
        Cast(gradYCastLocal_, gradYLocal_, RoundMode::CAST_NONE,c0_ * tileTaskBSCount);
        PipeBarrier<PIPE_V>();
        for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
            for (int32_t loopOffsetC0 = 0; loopOffsetC0 < c0_; loopOffsetC0+=ELEMENTS_SIZE_PER_REPEAT) {
                mul(xCastLocal_[loopIdN * c0_ + loopOffsetC0], xCastLocal_[loopIdN * c0_ + loopOffsetC0], gradYCastLocal_[loopOffsetC0],
                    ELEMENTS_SIZE_PER_REPEAT, tileTaskBSCount, { 1, 1, 1, n_ * c0_ / ELEMENTS_SIZE_PER_BLOCK, n_ * c0_ / ELEMENTS_SIZE_PER_BLOCK, c0_ / ELEMENTS_SIZE_PER_BLOCK });
            }
        }
        PipeBarrier<PIPE_V>();

        int64_t reduceLen = c0_;
        Add(xCastLocal_, xCastLocal_, xCastLocal_[128], REPEAT_LENTH, tileTaskBSCount * n_,
            {1, 1, 1, c0_ / ELEMENTS_SIZE_PER_BLOCK, c0_ / ELEMENTS_SIZE_PER_BLOCK, c0_ / ELEMENTS_SIZE_PER_BLOCK});
        Add(xCastLocal_[64], xCastLocal_[64], xCastLocal_[128 + 64], REPEAT_LENTH, tileTaskBSCount * n_,
            {1, 1, 1, c0_ / ELEMENTS_SIZE_PER_BLOCK, c0_ / ELEMENTS_SIZE_PER_BLOCK, c0_ / ELEMENTS_SIZE_PER_BLOCK});
        PipeBarrier<PIPE_V>();

        Add(xCastLocal_, xCastLocal_, xCastLocal_[64], REPEAT_LENTH, tileTaskBSCount * n_,
            {1, 1, 1, c0_ / ELEMENTS_SIZE_PER_BLOCK, c0_ / ELEMENTS_SIZE_PER_BLOCK, c0_ / ELEMENTS_SIZE_PER_BLOCK});
        PipeBarrier<PIPE_V>();
        Add(tmpLocal_, tmpLocal_, xCastLocal_, REPEAT_LENTH, tileTaskBSCount * n_,
            {1, 1, 1, 8, 8, c0_ / ELEMENTS_SIZE_PER_BLOCK});
        PipeBarrier<PIPE_V>();


    }
    PipeBarrier<PIPE_V>();
    Duplicate(dPrePostTempLocal_, 0.f, tilePre_ * n_ * 2);

    for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
        WholeReduceSum(dPrePostTempLocal_[loopIdN], tmpLocal_[loopIdN * c0_], REPEAT_LENTH, tileTaskBSCount, ELEMENTS_SIZE_PER_BLOCK, 1, n_ * 8);// dst repeat stride  srcblkstride srcrepstride
        // dst 间隔 8 2 *n
    }
    for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
        for (int32_t loopOffsetC0 = 0; loopOffsetC0 < c0_; loopOffsetC0+=ELEMENTS_SIZE_PER_REPEAT) {
            mul(xCastLocal_[loopIdN * c0_ + loopOffsetC0 ], gradYCastLocal_[loopOffsetC0 ], preBrcbLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK], ELEMENTS_SIZE_PER_REPEAT, tileTaskBSCount, { 1, 1, 1, c0_ / ELEMENTS_SIZE_PER_BLOCK, c0_ / ELEMENTS_SIZE_PER_BLOCK, 0 });
        }
    }
    // tailc


    PipeBarrier<PIPE_V>();
    DataCopyPad(gradHPreLocal_, gradPreGlobal_[taskOffset * n_], {static_cast<uint16_t>(tileTaskCount),
        static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {true, 0, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0});
    DataCopyPad(gradHPostLocal_, gradPostGlobal_[taskOffset * n_], {static_cast<uint16_t>(tileTaskCount),
        static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {true, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0, 0});
    
    DataCopyPad(hat2Local, h2Global_[taskOffset * (n_ * n_ + 2 * n_)], {static_cast<uint16_t>(tileTaskCount),
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
        (__ubuf__ T*) gradhatLocal_.GetPhyAddr(),
        (__ubuf__ T*) gradRsqrtLocal_.GetPhyAddr(),
        (__gm__ T*) gradHcScaleGlobal_.GetPhyAddr(),
        (__gm__ T*) gradHcBaseGlobal_.GetPhyAddr(),
        (__ubuf__ T*) fusedGradHPreAndGradHPostLocal_.GetPhyAddr(),
        (__ubuf__ T*) hat2Local.GetPhyAddr(),
        (__ubuf__ T*) rsqrtLocal_.GetPhyAddr(),
        (__ubuf__ T*) hcBaseLocal_.GetPhyAddr(),
        (__ubuf__ T*) tmpLocal.GetPhyAddr(),  // gradHPostLocal_ 用作临时空间
        hcScalePre_, hcScalePost_, tileTaskCount, n_);
}
/**
    约束：
    c: c >= 64 && c % 64 == 0
    n: n == 4
*/
template<typename TYPE_X, typename T>
__aicore__ inline void MhcPreGradKernel<T>::Process()
{
    if ASCEND_IS_AIV {
        // 同步所有vector核
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x8);
        CrossCoreWaitFlag(0x8);
        for (int32_t taskOffset = blkIdx_ * tileHHat2_; taskOffset < totalTasksAligned_; taskOffset += aivNum_ * tileHHat2_) {
            int32_t tileTaskCount = min(tileHHat2_, totalTasks_ - taskOffset);

            if (tileTaskCount > 0) {
                ComputeGradPreAndGradX3BS(taskOffset, tileTaskCount);
            }
        }
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
            tileRepeatTimes_ = CeilDiv(tileTaskCount * 2 * n_, ELEMENTS_SIZE_PER_REPEAT);
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

template<typename TYPE_X, typename T>
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

template<typename TYPE_X, typename T>
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