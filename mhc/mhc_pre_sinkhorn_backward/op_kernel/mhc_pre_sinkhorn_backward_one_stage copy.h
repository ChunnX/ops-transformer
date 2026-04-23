#ifndef MHC_PRE_GRAD_H
#define MHC_PRE_GRAD_H
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

#include <cstdint>

using namespace AscendC;
using namespace MicroAPI;

namespace {
constexpr int32_t BYTE_SIZE_PER_BLOCK = 32;
constexpr int32_t ELEMENTS_SIZE_PER_BLOCK = BYTE_SIZE_PER_BLOCK / sizeof(float);
constexpr int32_t BYTE_SIZE_PER_REPEAT = 256;
constexpr int32_t ELEMENTS_SIZE_PER_REPEAT = 256 / sizeof(float);
constexpr int32_t REPEAT_LENTH = ELEMENTS_SIZE_PER_REPEAT;
constexpr int32_t BLOCK_PER_REPEAT = 8;
constexpr uint64_t MASK_PRE[] = {0b0000111100001111000011110000111100001111000011110000111100001111};
constexpr uint64_t MASK_POST[] = {0b1111000011110000111100001111000011110000111100001111000011110000};
constexpr uint64_t MASK_POST_SCALE[] = {0b0000111100000000000000000000000000000000000000000000000000000000};
constexpr int32_t INNER_SPILT_NUM = 8;

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
        GM_ADDR grad_x, GM_ADDR grad_hc_fn, GM_ADDR grad_hc_scale, GM_ADDR grad_hc_base, GM_ADDR workspace, const MhcPreSinkhornBackwardTilingData* tilingData, TPipe* pipe)
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
    int64_t tileCoreBS_, tileCoreN_;
    int64_t skIterCount_;
    int64_t mm1K_, mm1M_, mm1N_;
    int64_t mm2K_, mm2M_, mm2N_;
    int64_t tileRepeatTimes_;
    float eps_;
    event_t eventIdVToMTE3XCast;
    event_t eventIdMTE3ToVXCast;
    event_t eventIdVToMTE3GradX;
    event_t eventIdMTE3ToVGradX;
    T hcScalePre_, hcScalePost_, hcScaleRes_;

    TQue<QuePosition::VECIN, 1> inputXInQueue;
    TQue<QuePosition::VECIN, 1> inputXInFloatQueue;
    TQue<QuePosition::VECIN, 1> inputGradQueue;

    TQue<QuePosition::VECIN, 2> SKInQueue;

    TQue<QuePosition::VECOUT, 1> OutQueue;

    TBuf<TPosition::VECCALC> fusedGradHPre2AndGradHPost2Buf_, gradHRes2Buf_,gradRsqrtBuf_,gradBiasBuf_,onesBuf_,ScaleBuf_,hcBaseBuf_,tempBuf_;

    // TBuf<TPosition::VECOUT> gradH2Buf_;
    LocalTensor<T> gradBiasLocal_, gradRsqrtLocal_, gradHRes2Local_, dPrePostTempLocal_;
    LocalTensor<TYPE_X> gradYLocal_, gradXLocal_, xLocal_;

    LocalTensor<T> xCastLocal_, gradYCastLocal_, gradXCastLocal_;
    LocalTensor<T> scaleLocal_, dScaleLocal_, dBiasLocal_;
    int32_t onceTask_;
    LocalTensor<T> preLocal_, gradPreLocal_;
    LocalTensor<T> gradHPreLocal_, gradHPostLocal_, fusedGradHPreAndGradHPostLocal_, gradHResLocal_;
    LocalTensor<T> hcBaseLocal_;
    LocalTensor<T> rsqrtLocal_;
    LocalTensor<T> preBrcbLocal_, dRsqrtBrcbLocal_, rsqrtbrcbLocal_, tmpLocal_, hat2Scale, dhatBeforeNormLocal, gradHResTempLocal_, gradHResTempLocal2_, dhatLocal_, rsqrtTempLocal_, gradXCubeLocal_;

    LocalTensor<T> hat2Local, hRes2Local_,hatLocal;
    LocalTensor<T> onesLocal_;

    GlobalTensor<TYPE_X> xGlobal_, gradYGlobal_;
    GlobalTensor<T> preGlobal_;
    GlobalTensor<TYPE_X> gradXGlobal_;
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
    GlobalTensor<T> gradHcBaseWSGlobal_;
    GlobalTensor<T> gradHcScaleWSGlobal_;
    GlobalTensor<T> xWorkspaceGlobal_;
    GlobalTensor<T> gradXCubeGlobal_;

private:
    __aicore__ inline void InitTiling(const MhcPreSinkhornBackwardTilingData* tilingData)
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
        c1Align_ = CeilDiv(c_, c0_);
        cTail_ = max((c_ - (c1Align_ - 1) * c0_), static_cast<int64_t>(0));
        // printf("cTail_ %d \n",cTail_);
        // printf("c1Align_ %d \n",c1Align_);
        // printf("c0_ %d \n",c0_);
        // printf("c_ %d \n",c_);

        cTailAlign_ = AlignUp(cTail_, ELEMENTS_SIZE_PER_BLOCK);
        cTailBlockStride_ = c0_ / ELEMENTS_SIZE_PER_BLOCK - cTailAlign_ / ELEMENTS_SIZE_PER_BLOCK;
        skIterCount_ = tilingData->skIterCount;
        eps_ = tilingData->eps;
        tileCoreBS_ = tilingData->tileSize;

        c0RepeatTime_ = c0_ / ELEMENTS_SIZE_PER_REPEAT;
        totalTasks_ = batchSize_ * seqLength_;
        totalTasksAligned_ = AlignUp(totalTasks_, aivNum_ * tileCoreBS_);
        if ASCEND_IS_AIC {
            mm1K_ = n_ * n_ + 2 * n_;
            mm1M_ = tileCoreBS_ * 2;
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
        int64_t workspaceOffset = 0;
        gradH2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace) + workspaceOffset);   // fp32
        workspaceOffset += batchSize_ * seqLength_ * (n_ * 2 + n_ * n_);
        xWorkspaceGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace)+workspaceOffset);               // fp32
        workspaceOffset += batchSize_ * seqLength_ * (n_ * c_);
        gradXCubeGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace)+ workspaceOffset);

        workspaceOffset += batchSize_ * seqLength_ * (n_ * c_);
        weightGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(hc_fn));  // fp32
        gradXGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE_X*>(grad_x));   // fp32

        if ASCEND_IS_AIV {
            preGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(pre));        // fp32
            gradYGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE_X*>(grad_y));   // fp32
            
            gradPostGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_post));   // fp32

            hcScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(hc_scale));   // fp32
            hcBaseGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(hc_base));   // fp32

            rsqrtGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(rsqrt));   // fp32
            gradHcScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_hc_scale));
            gradHcBaseGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_hc_base));
            gradHcScaleWSGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace)+ workspaceOffset);
            workspaceOffset += aivNum_  *  (n_ * 2 + n_ * n_);
            gradHcBaseWSGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace)+ workspaceOffset);
            workspaceOffset += aivNum_ * (n_ * 2 + n_ * n_);

            h2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(h_hat2));   // fp32

            skNormGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(norm_out));   // fp32
            skSumGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(sum_out));   // fp32

            gradHResGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad_comb));   // fp32
        }
    }

    __aicore__ inline void InitGradPreStageBuffer()
    {
        if ASCEND_IS_AIV {
            auto ubSize = 192 * 1024;
            pipe_->InitBuffer(fusedGradHPre2AndGradHPost2Buf_, tileCoreBS_ * n_ * 2 * sizeof(float));
            auto rr = fusedGradHPre2AndGradHPost2Buf_.Get<T>();
            // // DumpTensor(rr,__LINE__,256);

            pipe_->InitBuffer(gradHRes2Buf_, tileCoreBS_ * n_ * n_ * sizeof(float)); // useless 
            pipe_->InitBuffer(gradRsqrtBuf_, tileCoreBS_ * (n_ * n_ + 2 * n_) * sizeof(float) * 2);
            pipe_->InitBuffer(gradBiasBuf_,  tileCoreBS_ * (n_ * n_ + 2 * n_) * sizeof(float));
            
            pipe_->InitBuffer(onesBuf_, tileCoreBS_ * n_ * 2 * sizeof(float)); // pre: tileCoreBS_ * n
            pipe_->InitBuffer(hcBaseBuf_, (n_ * n_ + 2 * n_) * sizeof(float));
            pipe_->InitBuffer(ScaleBuf_, BYTE_SIZE_PER_BLOCK * 2); // pre: tileCoreBS_ * n

            pipe_->InitBuffer(inputXInQueue, 2, tileCoreBS_ * n_ * c0_ * sizeof(float) / 4);
            
            pipe_->InitBuffer(inputGradQueue, 1, tileCoreBS_ * n_ * ELEMENTS_SIZE_PER_BLOCK * sizeof(float));
            pipe_->InitBuffer(SKInQueue, 2, tileCoreBS_ * (n_ * ELEMENTS_SIZE_PER_BLOCK + ELEMENTS_SIZE_PER_BLOCK + n_)  * sizeof(float));
            pipe_->InitBuffer(OutQueue, 2, tileCoreBS_ * n_ * c0_ * sizeof(float) / 8);
            auto ubSizeRemain = ubSize - tileCoreBS_ * n_ * 2 * sizeof(float) - tileCoreBS_ * n_ * n_ * sizeof(float) -
                                tileCoreBS_ * (n_ * n_ + 2 * n_) * sizeof(float) * 4 - tileCoreBS_ * n_ * 2 * sizeof(float) -
                                ELEMENTS_SIZE_PER_BLOCK * 3 - 
                                tileCoreBS_ * n_ * c0_ * sizeof(float) / 4 * 2 -
                                tileCoreBS_ * n_ * ELEMENTS_SIZE_PER_BLOCK * sizeof(float) -
                                tileCoreBS_ * (n_ * ELEMENTS_SIZE_PER_BLOCK + ELEMENTS_SIZE_PER_BLOCK + n_)  * sizeof(float) * 2 -
                                tileCoreBS_ * n_ * c0_ * sizeof(float) / 8 * 2;

            pipe_->InitBuffer(tempBuf_, ubSizeRemain); // grad_y: n_ * c0_
            {
                auto rr1 = tempBuf_.Get<T>();
                // // DumpTensor(rr1,__LINE__,256); 
            }
            eventIdVToMTE3XCast = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
            eventIdMTE3ToVXCast = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
            eventIdVToMTE3GradX = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
            eventIdMTE3ToVGradX = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
            dPrePostTempLocal_  = fusedGradHPre2AndGradHPost2Buf_.Get<T>();
            hcBaseLocal_ = hcBaseBuf_.Get<T>();
            onesLocal_ = onesBuf_.Get<T>();
            scaleLocal_ = ScaleBuf_.Get<T>();
            gradRsqrtLocal_ = gradRsqrtBuf_.Get<T>();
            dBiasLocal_  = gradBiasBuf_.Get<T>();
            dScaleLocal_ = dBiasLocal_[tileCoreBS_ * (n_ * n_ + 2 * n_)];
            onceTask_ = tileCoreBS_ / INNER_SPILT_NUM;

            int32_t offset = 0;
            int32_t brcbAlign = CeilDiv(tileCoreBS_ * n_, ELEMENTS_SIZE_PER_BLOCK);
            preBrcbLocal_ = tempBuf_.GetWithOffset<T>(brcbAlign * ELEMENTS_SIZE_PER_REPEAT, offset);
            offset += brcbAlign * ELEMENTS_SIZE_PER_REPEAT * sizeof(float);
            brcbAlign = CeilDiv(tileCoreBS_, ELEMENTS_SIZE_PER_BLOCK);
            dRsqrtBrcbLocal_ = tempBuf_.GetWithOffset<T>(brcbAlign * ELEMENTS_SIZE_PER_REPEAT, offset);
            offset += brcbAlign * ELEMENTS_SIZE_PER_REPEAT * sizeof(float);
            gradYCastLocal_ = tempBuf_.GetWithOffset<T>(onceTask_ * c0_, offset);
            offset += onceTask_ * c0_ * sizeof(float);
            gradXCastLocal_ = tempBuf_.GetWithOffset<T>(onceTask_ * c0_, offset);
            offset += onceTask_ * c0_ * sizeof(float);
            xCastLocal_ = tempBuf_.GetWithOffset<T>(onceTask_ * c0_, offset);
            offset = 0;
            // ComputeGradPre
            tmpLocal_ = gradXCastLocal_;
            // SinkhornGrad
            gradHResTempLocal_ = tempBuf_.GetWithOffset<T>(tileCoreBS_ * n_ * 8 * 2, offset);// int64大小
            offset += tileCoreBS_ * n_ * 8 * 2 * sizeof(float);
            gradHResTempLocal2_ = tempBuf_.GetWithOffset<T>(tileCoreBS_ * n_ * 8, offset);
            offset = 0;

            
            ///
            dhatLocal_ = tempBuf_.GetWithOffset<T>(tileCoreBS_ * (n_ * n_ + 2 * n_), offset);// int64大小
            offset += tileCoreBS_ * (n_ * n_ + 2 * n_) * sizeof(float);
            brcbAlign = CeilDiv(tileCoreBS_, ELEMENTS_SIZE_PER_BLOCK);
            rsqrtbrcbLocal_ = tempBuf_.GetWithOffset<T>(brcbAlign * BYTE_SIZE_PER_BLOCK * (n_ * n_ + 2 * n_), offset);
            offset += brcbAlign * BYTE_SIZE_PER_BLOCK * (n_ * n_ + 2 * n_) * sizeof(float);
            hat2Scale = tempBuf_.GetWithOffset<T>(tileCoreBS_ * (n_ * n_ + 2 * n_), offset);// int64大小
            offset += tileCoreBS_ * (n_ * n_ + 2 * n_) * sizeof(float);
            dhatBeforeNormLocal = tempBuf_.GetWithOffset<T>(tileCoreBS_ * (2 * n_), offset);// int64大小
            offset += tileCoreBS_ * (n_ * n_ + 2 * n_) * sizeof(float);
            hatLocal = tempBuf_.GetWithOffset<T>(tileCoreBS_ * (n_ * n_ + 2 * n_), offset);// int64大小
            
            Duplicate(onesLocal_, 1.f, tileCoreBS_ * n_ * 2);
            Duplicate(dScaleLocal_, 0.f, tileCoreBS_ * (2 * n_ + n_ * n_));
            Duplicate(dBiasLocal_, 0.f, tileCoreBS_ * (2 * n_ + n_ * n_));
        }
    }

    __aicore__ inline void ComputeGradPre(const int32_t taskOffset, const int32_t tileTaskCount, const int32_t innerId);

    __aicore__ inline void ComputeGradHHat2(const int32_t taskOffset, const int32_t tileTaskCount);
    __aicore__ inline void SinkhornGrad(const int32_t taskOffset, const int32_t tileTaskCount);

    __aicore__ inline void ComputeGradX1(const int32_t taskOffset, const int32_t tileTaskCount, const int32_t innerId);
    __aicore__ inline void GetHcScaleAndHcBase();
    __aicore__ inline void ProcessMatmul1(const int32_t taskOffset, const int32_t mm1M);
    __aicore__ inline void ProcessMatmul2(const int32_t taskOffset, const int32_t mm2K);
    __aicore__ inline void ComputeScaleBias();
};


template<typename TYPE_X, typename T>
__aicore__ inline void MhcPreGradKernel<TYPE_X, T>::GetHcScaleAndHcBase()
{
    // HcBase 常驻 UB
    hcScalePre_ = hcScaleGlobal_.GetValue(0);
    hcScalePost_ = hcScaleGlobal_.GetValue(1);
    hcScaleRes_ = hcScaleGlobal_.GetValue(2);
    event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIDSToV);
    WaitFlag<HardEvent::S_V>(eventIDSToV);
    Duplicate(scaleLocal_[8], hcScaleRes_, 8);
    Duplicate(scaleLocal_, hcScalePost_, 8);
    PipeBarrier<PIPE_V>();

    Duplicate(scaleLocal_, hcScalePre_, 4);

    DataCopyPad(hcBaseLocal_, hcBaseGlobal_, {static_cast<uint16_t>(1),
        static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    
}

template<typename TYPE_X, typename T>
__aicore__ inline void BroadCast4to8(const int32_t taskOffset, const int32_t tileTaskCount)
{

}

template<typename TYPE_X, typename T>
__aicore__ inline void MhcPreGradKernel<TYPE_X, T>::SinkhornGrad(const int32_t taskOffset, const int32_t tileTaskCount)
{
    // DataCopyPad(gradHPostLocal_, gradHResGlobal_[taskOffset * n_ * n_], {static_cast<uint16_t>(1),
    //     static_cast<uint32_t>(tileTaskCount  * n_ * n_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    gradHResLocal_ = inputGradQueue.AllocTensor<T>();
    if (blkIdx_ == 0 && taskOffset ==0) {
        printf("taskOffset %d \n",taskOffset);
        printf("__LINE__ %d \n",__LINE__);

        // // DumpTensor(dhatLocal_,__LINE__,tileTaskCount * ((n_ * n_ + 2 * n_)) +32);
    }
    DataCopyPad(gradHResLocal_, gradHResGlobal_[taskOffset * n_ * n_], {static_cast<uint16_t>(tileTaskCount  * n_),
        static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {true, 0, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0});
    inputGradQueue.EnQue(gradHResLocal_);
    inputGradQueue.DeQue();
    if (blkIdx_ == 0 && taskOffset ==0) {
        printf("taskOffset %d \n",taskOffset);
        printf("__LINE__ %d \n",__LINE__);
        DumpTensor(gradHResLocal_,__LINE__,512);
        DumpTensor(gradHResGlobal_,__LINE__,512);

        // // DumpTensor(dhatLocal_,__LINE__,tileTaskCount * ((n_ * n_ + 2 * n_)) +32);
    }
    int64_t iterRowNormOffset = (skIterCount_ - 1) * 2 * BSNN + taskOffset * n_ * n_;
    int64_t iterColNormOffset = ((skIterCount_ - 1) * 2 + 1)* BSNN + taskOffset * n_ * n_;

    int64_t iterRowSumOffset = (skIterCount_ - 1) * 2 * BSN + taskOffset * n_;
    int64_t iterColSumOffset = ((skIterCount_ - 1) * 2 + 1) * BSN + taskOffset * n_;
    int32_t brcbAlign = CeilDiv(tileTaskCount * n_, ELEMENTS_SIZE_PER_BLOCK);
    for (int32_t iter = skIterCount_ - 1; iter > 0; iter--) {
        // DataCopyPad(skColNormLocal_, skNormGlobal_[iterColNormOffset], {static_cast<uint16_t>(1),
        //     static_cast<uint32_t>(tileTaskCount  * n_ * n_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        // DataCopyPad(skColSumLocal_, skSumGlobal_[iterColSumOffset], {static_cast<uint16_t>(1),
        //     static_cast<uint32_t>(tileTaskCount  * n_ * n_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});   
        // DataCopyPad(skColNormLocal_, skNormGlobal_[iterColNormOffset], {static_cast<uint16_t>(tileTaskCount  * n_),

        //     static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {true, 0, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0});
        auto skRowNormLocal_ = SKInQueue.AllocTensor<T>();
        auto skRowSumLocal_ = skRowNormLocal_[tileCoreBS_  * n_ * ELEMENTS_SIZE_PER_BLOCK];
        auto skColSumLocal_ = skRowNormLocal_[tileCoreBS_  * n_ * ELEMENTS_SIZE_PER_BLOCK + tileCoreBS_ * ELEMENTS_SIZE_PER_BLOCK];

        DataCopyPad(skRowNormLocal_, skNormGlobal_[iterRowNormOffset], {static_cast<uint16_t>(tileTaskCount  * n_),
            static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {true, 0, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0});
        DataCopyPad(skColSumLocal_, skSumGlobal_[iterColSumOffset], {static_cast<uint16_t>(tileTaskCount),
            static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {true, 0, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0});
        DataCopyPad(skRowSumLocal_, skSumGlobal_[iterRowSumOffset], {static_cast<uint16_t>(1),
            static_cast<uint32_t>(tileTaskCount  * n_ * sizeof(T)), 0, 0, 0}, {false, 0, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0});
        SKInQueue.EnQue(gradHResLocal_);
        SKInQueue.DeQue();
        PipeBarrier<PIPE_V>();
        // if (blkIdx_ == 0 && taskOffset ==0) {
        //     printf("iter %d \n",iter);
        //     printf("__LINE__ %d \n",__LINE__);

        //     // // DumpTensor(dhatLocal_,__LINE__,tileTaskCount * ((n_ * n_ + 2 * n_)) +32);
        // }
        Adds(skColSumLocal_, skColSumLocal_, eps_,  tileTaskCount * ELEMENTS_SIZE_PER_BLOCK);
        PipeBarrier<PIPE_V>();
        // if (blkIdx_ == 0 && taskOffset ==0) {
        //     printf("iter %d \n",iter);
        //     printf("__LINE__ %d \n",__LINE__);
        //     DumpTensor(gradHResLocal_,__LINE__,128);

        //     DumpTensor(skColSumLocal_,__LINE__,128);
        // }

        for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
            Div(gradHResTempLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK], gradHResLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK],
                skColSumLocal_, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,{static_cast<uint8_t>(n_), static_cast<uint8_t>(n_), 1, static_cast<uint8_t>(n_ * 8), static_cast<uint8_t>(n_ * 8), 8});
        }
        // if (blkIdx_ == 0 && taskOffset ==0) {
        //     printf("iter %d \n",iter);
        //     printf("__LINE__ %d \n",__LINE__);
        //     DumpTensor(gradHResTempLocal_,__LINE__,256);
        //     DumpTensor(skRowSumLocal_,__LINE__,256);

        //     // // DumpTensor(dhatLocal_,__LINE__,tileTaskCount * ((n_ * n_ + 2 * n_)) +32);
        // }
        PipeBarrier<PIPE_V>(); 
        // xg * normed
        Mul(gradHResTempLocal2_, gradHResLocal_, skRowNormLocal_, tileTaskCount * n_ * ELEMENTS_SIZE_PER_BLOCK);
        PipeBarrier<PIPE_V>();
        // (sum + eps) ** 2
        Mul(skColSumLocal_, skColSumLocal_, skColSumLocal_, tileTaskCount * ELEMENTS_SIZE_PER_BLOCK);
        // bs * n *8 reduce bs * 1 *8 out : value other other other value other other other
        // sum(xg * normed )
        // if (blkIdx_ == 0 && taskOffset ==0) {
        //     printf("iter %d \n",iter);
        //     printf("__LINE__ %d \n",__LINE__);
        //     DumpTensor(skColSumLocal_,__LINE__,256);

        //     // // DumpTensor(dhatLocal_,__LINE__,tileTaskCount * ((n_ * n_ + 2 * n_)) +32);
        // }
        for (int32_t loopIdN = 1; loopIdN < n_; loopIdN += 1) {
            Add(gradHResTempLocal2_, gradHResTempLocal2_, 
                gradHResTempLocal2_[loopIdN * ELEMENTS_SIZE_PER_BLOCK], ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_, {static_cast<uint8_t>(n_), static_cast<uint8_t>(n_), static_cast<uint8_t>(n_), static_cast<uint8_t>(n_ * 8), static_cast<uint8_t>(n_ * 8), static_cast<uint8_t>(n_ * 8)});
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_V>();
        // sum(xg * normed )/ (sum + eps) ** 2
        Div(gradHResTempLocal2_, gradHResTempLocal2_, skColSumLocal_,ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_, {static_cast<uint8_t>(n_), static_cast<uint8_t>(n_), 1, static_cast<uint8_t>(n_ * 8), static_cast<uint8_t>(n_ * 8), 8});
        PipeBarrier<PIPE_V>();
        // for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
        //     Div(gradHResTempLocal2_[loopIdN * ELEMENTS_SIZE_PER_BLOCK], gradHResTempLocal2_[loopIdN * ELEMENTS_SIZE_PER_BLOCK],
        //         skColSumLocal_, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,{static_cast<uint8_t>(n_), static_cast<uint8_t>(n_), 1, static_cast<uint8_t>(n_ * 8), static_cast<uint8_t>(n_ * 8), 8});
        // }
        // if (blkIdx_ == 0 && taskOffset ==0) {
        //     printf("iter %d \n",iter);
        //     printf("__LINE__ %d \n",__LINE__);

        //     DumpTensor(gradHResTempLocal2_,__LINE__,256);
        // }
        for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
            Sub(gradHResLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK], gradHResTempLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK],
                gradHResTempLocal2_, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,{static_cast<uint8_t>(n_), static_cast<uint8_t>(n_), static_cast<uint8_t>(n_), static_cast<uint8_t>(n_ * 8), static_cast<uint8_t>(n_ * 8), static_cast<uint8_t>(n_ * 8)});
        }
        PipeBarrier<PIPE_V>();
        // if (blkIdx_ == 0 && taskOffset ==0) {
        //     printf("iter %d \n",iter);
        //     printf("__LINE__ %d \n",__LINE__);

        //     DumpTensor(gradHResLocal_,__LINE__,512);
        // }


        // AscendC::BlockReduceSum<half>(skColNormLocal_, skColNormLocal_, tileRepeatTimes_, MASK_PRE, 1, 1, 8);

        


        // sw
        // bsn -> bsn8
        Adds(skRowSumLocal_, skRowSumLocal_, eps_,  tileTaskCount * n_);
        PipeBarrier<PIPE_V>();

        Brcb(gradHResTempLocal2_, skRowSumLocal_, brcbAlign, {1,8});
        // if (blkIdx_ == 0 && taskOffset ==0) {
        //     printf("iter %d \n",iter);
        //     printf("__Lrn__ %d \n",tileTaskCount * n_ * ELEMENTS_SIZE_PER_BLOCK);
        //     DumpTensor(gradHResTempLocal2_,__LINE__,512);
        //     DumpTensor(gradHResLocal_,__LINE__,512);

        //     // // DumpTensor(dhatLocal_,__LINE__,tileTaskCount * ((n_ * n_ + 2 * n_)) +32);
        // }

        PipeBarrier<PIPE_V>();
        Div(gradHResTempLocal_, gradHResLocal_, gradHResTempLocal2_, tileTaskCount * n_ * ELEMENTS_SIZE_PER_BLOCK);
        PipeBarrier<PIPE_V>();
        // if (blkIdx_ == 0 && taskOffset ==0) {
        //     printf("iter %d \n",iter);
        //     printf("__Lrn__ %d \n",tileTaskCount * n_ * ELEMENTS_SIZE_PER_BLOCK);
        //     DumpTensor(gradHResTempLocal_,__LINE__,512);

        //     // // DumpTensor(dhatLocal_,__LINE__,tileTaskCount * ((n_ * n_ + 2 * n_)) +32);
        // }
        Mul(skRowNormLocal_, skRowNormLocal_, gradHResTempLocal_, tileTaskCount * n_ * ELEMENTS_SIZE_PER_BLOCK);
        PipeBarrier<PIPE_V>();
        // if (blkIdx_ == 0 && taskOffset ==0) {
        //     printf("iter %d \n",iter);
        //     printf("__LINE__ %d \n",__LINE__);
        //     DumpTensor(skRowNormLocal_,__LINE__,512);

        //     // // DumpTensor(dhatLocal_,__LINE__,tileTaskCount * ((n_ * n_ + 2 * n_)) +32);
        // }

        AscendC::BlockReduceSum(skRowNormLocal_, skRowNormLocal_, static_cast<int32_t>(static_cast<int64_t>(tileRepeatTimes_) * n_), MASK_PRE, 1, 1, 8);
        // if (blkIdx_ == 0 && taskOffset ==0) {
        //     printf("iter %d \n",iter);
        //     printf("__LINE__ %d \n",__LINE__);
        //     DumpTensor(skRowNormLocal_,__LINE__,512);

        //     // // DumpTensor(dhatLocal_,__LINE__,tileTaskCount * ((n_ * n_ + 2 * n_)) +32);
        // }
        PipeBarrier<PIPE_V>();
        Brcb(gradHResTempLocal2_, skRowNormLocal_, brcbAlign, {1,8});
        SKInQueue.FreeTensor(skRowNormLocal_);

        PipeBarrier<PIPE_V>();
        Sub(gradHResLocal_, gradHResTempLocal_, gradHResTempLocal2_, tileTaskCount * n_ * ELEMENTS_SIZE_PER_BLOCK);
        if (blkIdx_ == 0 && taskOffset ==0) {
            printf("iter %d \n",iter);
            printf("__Lrn__ %d \n",tileTaskCount * n_ * ELEMENTS_SIZE_PER_BLOCK);
            DumpTensor(gradHResLocal_,__LINE__,256);

            // // DumpTensor(dhatLocal_,__LINE__,tileTaskCount * ((n_ * n_ + 2 * n_)) +32);
        }
        iterRowNormOffset = iterRowNormOffset - 2 * BSNN;
        iterRowSumOffset = iterRowSumOffset - 2 * BSN; 
        iterColSumOffset = iterColSumOffset - 2 * BSN;
    }
    auto skRowNormLocal_ = SKInQueue.AllocTensor<T>();
    auto skColSumLocal_ = skRowNormLocal_[tileTaskCount  * n_ * ELEMENTS_SIZE_PER_BLOCK];
    printf("__LINE__ %d \n",__LINE__);

    DataCopyPad(skRowNormLocal_, skNormGlobal_[iterRowNormOffset], {static_cast<uint16_t>(tileTaskCount  * n_),
        static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {true, 0, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0});
    DataCopyPad(skColSumLocal_, skSumGlobal_[iterColSumOffset], {static_cast<uint16_t>(tileTaskCount),
        static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {true, 0, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0});
    PipeBarrier<PIPE_V>();
    SKInQueue.EnQue(gradHResLocal_);
    SKInQueue.DeQue();
    printf("__LINE__ %d \n",__LINE__);

    Adds(skColSumLocal_, skColSumLocal_, eps_,  tileTaskCount * ELEMENTS_SIZE_PER_BLOCK);
    PipeBarrier<PIPE_V>();
    printf("__LINE__ %d \n",__LINE__);

    for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
        Div(gradHResTempLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK], gradHResLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK],
            skColSumLocal_, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,{static_cast<uint8_t>(n_), static_cast<uint8_t>(n_), 1, static_cast<uint8_t>(n_ * 8), static_cast<uint8_t>(n_ * 8), 8});
    }
    printf("__LINE__ %d \n",__LINE__);

    if (blkIdx_ == 0 && taskOffset ==0) {
        printf("__Lrn__ \n");
        DumpTensor(gradHResLocal_,__LINE__,256);

        // // DumpTensor(dhatLocal_,__LINE__,tileTaskCount * ((n_ * n_ + 2 * n_)) +32);
    }
    PipeBarrier<PIPE_V>(); 
    // xg * normed
        printf("__LINE__ %d \n",__LINE__);

    Mul(gradHResTempLocal2_, gradHResLocal_, skRowNormLocal_, tileTaskCount * n_ * ELEMENTS_SIZE_PER_BLOCK);
    PipeBarrier<PIPE_V>();
    // (sum + eps) ** 2
    Mul(skColSumLocal_, skColSumLocal_, skColSumLocal_, tileTaskCount * ELEMENTS_SIZE_PER_BLOCK);
    // bs * n *8 reduce bs * 1 *8 out : value other other other value other other other
    // sum(xg * normed )
    for (int32_t loopIdN = 1; loopIdN < n_; loopIdN += 1) {
        Add(gradHResTempLocal2_, gradHResTempLocal2_, 
            gradHResTempLocal2_[loopIdN * ELEMENTS_SIZE_PER_BLOCK], ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_, {static_cast<uint8_t>(n_), static_cast<uint8_t>(n_), static_cast<uint8_t>(n_), static_cast<uint8_t>(n_ * 8), static_cast<uint8_t>(n_ * 8), static_cast<uint8_t>(n_ * 8)});
        PipeBarrier<PIPE_V>();
    }
    printf("__LINE__ %d \n",__LINE__);

    PipeBarrier<PIPE_V>();
    // sum(xg * normed )/ (sum + eps) ** 2
    Div(gradHResTempLocal2_, gradHResTempLocal2_, skColSumLocal_,ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_, {static_cast<uint8_t>(n_), static_cast<uint8_t>(n_), 1, static_cast<uint8_t>(n_ * 8), static_cast<uint8_t>(n_ * 8), 8});
    PipeBarrier<PIPE_V>();
    // for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
    //     Div(gradHResTempLocal2_[loopIdN * ELEMENTS_SIZE_PER_BLOCK], gradHResTempLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK],
    //         skColSumLocal_, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_ * n_,{n_, n_, n_, n_ * 8, n_ * 8, n_ * 8});
    // }
    for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
        Sub(gradHResLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK], gradHResTempLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK],
            gradHResTempLocal2_, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,{static_cast<uint8_t>(n_), static_cast<uint8_t>(n_), static_cast<uint8_t>(n_), static_cast<uint8_t>(n_ * 8), static_cast<uint8_t>(n_ * 8), static_cast<uint8_t>(n_ * 8)});
    }


    // SOFTMAX
    PipeBarrier<PIPE_V>();
    Mul(gradHResTempLocal_, gradHResLocal_, skRowNormLocal_, tileTaskCount * n_ * ELEMENTS_SIZE_PER_BLOCK);
    SKInQueue.FreeTensor(skRowNormLocal_);

    PipeBarrier<PIPE_V>();
    AscendC::BlockReduceSum(gradHResTempLocal_, gradHResTempLocal_, static_cast<int32_t>(static_cast<int64_t>(tileRepeatTimes_) * n_), MASK_PRE, 1, 1, 8);
    PipeBarrier<PIPE_V>();
    Brcb(gradHResTempLocal2_, gradHResTempLocal_, brcbAlign, {1,8});
    PipeBarrier<PIPE_V>();
    Sub(gradHResLocal_, gradHResLocal_, gradHResTempLocal2_, tileTaskCount * n_ * ELEMENTS_SIZE_PER_BLOCK);
    // if (blkIdx_ == 0 && taskOffset ==0) {
    //     printf("__LINE__111 %d \n",__LINE__);

    //     // // DumpTensor(dhatLocal_,__LINE__,tileTaskCount * ((n_ * n_ + 2 * n_)) +32);
    // }

    PipeBarrier<PIPE_V>();
    Mul(gradHResLocal_, gradHResLocal_, skRowNormLocal_, tileTaskCount * n_ * ELEMENTS_SIZE_PER_BLOCK);
    PipeBarrier<PIPE_V>();
    if (blkIdx_ == 0 && taskOffset ==0) {
        printf("__LINE__111 %d \n",__LINE__);

        DumpTensor(gradHResLocal_,__LINE__,512);
    }
    //4v 4o 4v 40 -> 4v 4v
    Cast(gradHResTempLocal_.template ReinterpretCast<int64_t>(), gradHResLocal_.template ReinterpretCast<int32_t>(), RoundMode::CAST_NONE,tileTaskCount * n_ * ELEMENTS_SIZE_PER_BLOCK);
    PipeBarrier<PIPE_V>();

    Copy(gradHResLocal_, gradHResTempLocal_, ELEMENTS_SIZE_PER_REPEAT,n_ * tileRepeatTimes_, {1, 2, 8, 16});
    PipeBarrier<PIPE_V>();

    Cast(gradHResLocal_.template ReinterpretCast<int32_t>(), gradHResLocal_.template ReinterpretCast<int64_t>(), RoundMode::CAST_NONE, tileTaskCount * n_ * n_);
    if (blkIdx_ == 0 && taskOffset ==0) {
        printf("__LINE__111 %d \n",__LINE__);

        DumpTensor(gradHResLocal_,__LINE__,512);
    }
    PipeBarrier<PIPE_V>();

}

template<typename TYPE_X, typename T>
__aicore__ inline void MhcPreGradKernel<TYPE_X, T>::ComputeGradHHat2(const int32_t taskOffset, const int32_t tileTaskCount)
{
    if (blkIdx_ == 0 && taskOffset ==0) {
        DumpTensor(dhatLocal_,__LINE__,512);
    }

    SinkhornGrad(taskOffset, tileTaskCount);
    Duplicate(dhatLocal_, 0.f, tileTaskCount * 24);
    PipeBarrier<PIPE_V>();

    if (blkIdx_ == 0 && taskOffset ==0) {
        DumpTensor(dhatLocal_,__LINE__,512);
    }
    for (int32_t loopIdN = 0; loopIdN < 2; loopIdN+=1) {
        Copy(dhatLocal_[ELEMENTS_SIZE_PER_BLOCK + loopIdN * ELEMENTS_SIZE_PER_BLOCK], gradHResLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK],
            ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,{static_cast<uint16_t>(3), static_cast<uint16_t>(2), static_cast<uint16_t>((2 + 1) * 8), static_cast<uint16_t>(2 * 8)});
    }
    
    if (blkIdx_ == 0 && taskOffset ==0) {
        DumpTensor(dhatLocal_,__LINE__,512);
    }
    inputGradQueue.FreeTensor(gradHResLocal_);

    auto gradHPostLocal_ = inputXInQueue.AllocTensor<T>();
    auto hat2LocalTemp = gradHPostLocal_[tileCoreBS_ * ELEMENTS_SIZE_PER_BLOCK];
    auto rsqrtLocal_ =  gradHPostLocal_[tileCoreBS_ * ELEMENTS_SIZE_PER_BLOCK + tileCoreBS_ * (n_ * n_ + 2 * n_)];
    
    // DataCopyPad(gradHPreLocal_, gradPreGlobal_[taskOffset * n_], {static_cast<uint16_t>(tileTaskCount),
    //     static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {true, 0, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0});
    DataCopyPad(gradHPostLocal_, gradPostGlobal_[taskOffset * n_], {static_cast<uint16_t>(tileTaskCount),
        static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {true, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0, 0});
    
    DataCopyPad(hat2LocalTemp, h2Global_[taskOffset * (n_ * n_ + 2 * n_)], {static_cast<uint16_t>(tileTaskCount),
        static_cast<uint32_t>( (n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(rsqrtLocal_, rsqrtGlobal_[taskOffset], {static_cast<uint16_t>(1),
        static_cast<uint32_t>(tileTaskCount * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    inputXInQueue.EnQue(gradHPostLocal_);
    inputXInQueue.DeQue(); 

    PipeBarrier<PIPE_V>();
    if (taskOffset == 0) {
        DumpTensor(dPrePostTempLocal_,__LINE__,512);

    }
    // GRAD_PRE*1_POST*2
    Axpy(dPrePostTempLocal_, gradHPostLocal_,float(2), tileTaskCount * 2 * n_);
    if (taskOffset == 0) {
        DumpTensor(dPrePostTempLocal_,__LINE__,512);

    }
    const uint32_t srcShape[2] = {static_cast<uint32_t>(tileTaskCount * n_), 1}; // 源数据shape
    const uint32_t dstShape[2] = {static_cast<uint32_t>(tileTaskCount * n_), static_cast<uint32_t>(n_ * n_ + 2 * n_)};// broadcast数据shape
    PipeBarrier<PIPE_V>();

    AscendC::Broadcast<float, 2, 1>(rsqrtbrcbLocal_, rsqrtLocal_, dstShape, srcShape);


    PipeBarrier<PIPE_V>();
    if (taskOffset == 0) {
        DumpTensor(rsqrtbrcbLocal_,__LINE__,512);
        DumpTensor(hat2LocalTemp,__LINE__,512);
        DumpTensor(scaleLocal_,__LINE__,512);

    }
    // GRAD_HAT_RSQRT
    Mul(hat2Scale, scaleLocal_,hat2LocalTemp ,ELEMENTS_SIZE_PER_REPEAT,tileRepeatTimes_,{3, 0, 3, 24, 0, 24});
    Mul(hat2Scale[8], scaleLocal_[8],hat2LocalTemp[8] ,ELEMENTS_SIZE_PER_REPEAT,tileRepeatTimes_,{3, 0, 3, 24, 0, 24});
    Mul(hat2Scale[16], scaleLocal_[8],hat2LocalTemp[16] ,ELEMENTS_SIZE_PER_REPEAT,tileRepeatTimes_,{3, 0, 3, 24, 0, 24});
    PipeBarrier<PIPE_V>();
    if (taskOffset == 0) {
        DumpTensor(hat2Scale,__LINE__,512);
    }
    Add(hatLocal, hcBaseLocal_,hat2Scale ,ELEMENTS_SIZE_PER_REPEAT,tileRepeatTimes_,{3, 0, 3, 24, 0, 24});
    Add(hatLocal[8], hcBaseLocal_[8],hat2Scale[8] ,ELEMENTS_SIZE_PER_REPEAT,tileRepeatTimes_,{3, 0, 3, 24, 0, 24});
    Add(hatLocal[16], hcBaseLocal_[16],hat2Scale[16] ,ELEMENTS_SIZE_PER_REPEAT,tileRepeatTimes_,{3, 0, 3, 24, 0, 24});
    if (taskOffset == 0) {
        DumpTensor(hatLocal,__LINE__,512);
    }
    PipeBarrier<PIPE_V>();

    Mul(hatLocal, hatLocal,rsqrtbrcbLocal_ ,(n_ * n_ + 2 * n_) * tileTaskCount);
    PipeBarrier<PIPE_V>();
    if (taskOffset == 0) {
        DumpTensor(hatLocal,__LINE__,512);
    }
    auto hatLocalTemp = dhatBeforeNormLocal;

    // 正向pre post sigmoid
    Muls(hatLocalTemp, hatLocal, float(-1), ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_, {1,3,8,24});
    PipeBarrier<PIPE_V>();

    Exp(hatLocal, hatLocalTemp, (2 * n_) * tileTaskCount);
    PipeBarrier<PIPE_V>();

    Adds(hatLocal, hatLocal, float(1), (2 * n_) * tileTaskCount);
    PipeBarrier<PIPE_V>();
    Div(hatLocal, onesLocal_, hatLocal, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,{1, 0, 1, 8, 0, 8});
    if (taskOffset == 0) {
        DumpTensor(hatLocal,__LINE__,512);
    }
    // 正向pre post sigmoidgrad x_sigmoid * (1 - x_sigmoid) * grad_output
    PipeBarrier<PIPE_V>();
    Mul(hatLocalTemp, hatLocal, hatLocal, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,{1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    Sub(hatLocalTemp, hatLocal, hatLocalTemp, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,{1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    if (taskOffset == 0) {
        DumpTensor(hatLocalTemp,__LINE__,512);
        DumpTensor(dPrePostTempLocal_,__LINE__,512);

    }
    Mul(dhatLocal_, dPrePostTempLocal_, hatLocalTemp, ELEMENTS_SIZE_PER_REPEAT, tileRepeatTimes_,{3, 1, 1, 24, 8, 8});
    PipeBarrier<PIPE_V>();

    // 不同批次dbias累加
    if (taskOffset == 0) {
        DumpTensor(dhatLocal_,__LINE__,512);
    }
    PipeBarrier<PIPE_V>();
    Add(dBiasLocal_, dhatLocal_, dBiasLocal_, (n_ * n_ + 2 * n_) * tileTaskCount);
    // dRsqrt = dhatLocal_ * beforenorm
    Mul(gradRsqrtLocal_, dhatLocal_, hat2Scale, (n_ * n_ + 2 * n_) * tileTaskCount);
    if (taskOffset == 0) {
        DumpTensor(gradRsqrtLocal_,__LINE__,512);
    }
    PipeBarrier<PIPE_V>();
    // rmsnormgrad (- (rsqrt ** 3)) * grad_rsqrt / float(NC)
    WholeReduceSum(rsqrtTempLocal_, gradRsqrtLocal_, (n_ * n_ + 2 * n_), tileTaskCount, 1, 1, 3);// dst repeat stride  srcblkstride srcrepstride (n*2 +2n)/8
    PipeBarrier<PIPE_V>();
    if (taskOffset == 0) {
        DumpTensor(rsqrtTempLocal_,__LINE__,512);
        DumpTensor(rsqrtLocal_,__LINE__,512);
    }
    Mul(rsqrtTempLocal_, rsqrtLocal_, rsqrtLocal_, tileTaskCount);
    PipeBarrier<PIPE_V>();
    Mul(rsqrtTempLocal_, rsqrtTempLocal_, rsqrtLocal_, tileTaskCount);
    inputXInQueue.FreeTensor(gradHPostLocal_);

    PipeBarrier<PIPE_V>();
    // Mul(gradRsqrtLocal_, gradRsqrtLocal_, rsqrtTempLocal_, tileTaskCount);
    PipeBarrier<PIPE_V>();
    Muls(gradRsqrtLocal_, rsqrtTempLocal_, float(-1) / (n_ * c_),  tileTaskCount);
    if (taskOffset == 0) {
        DumpTensor(gradRsqrtLocal_,__LINE__,512);
    }
    //dscale = dhatLocal_ * hat pre * rsqrt
    //dhat2 = dhatLocal_  * rsqrt  * scale
    Mul(dhatBeforeNormLocal, rsqrtbrcbLocal_,dhatLocal_ ,(n_ * n_ + 2 * n_) * tileTaskCount);
    PipeBarrier<PIPE_V>();
    // 不同批次dScale累加
    MulAddDst(dScaleLocal_, hat2LocalTemp,dhatBeforeNormLocal ,(n_ * n_ + 2 * n_) * tileTaskCount);
    if (taskOffset < 64) {
        DumpTensor(dScaleLocal_,__LINE__,512);
    }
    auto dhat2Local = OutQueue.AllocTensor<float>();

    Mul(dhat2Local, scaleLocal_,dhatBeforeNormLocal ,ELEMENTS_SIZE_PER_REPEAT,tileRepeatTimes_,{3, 0, 3, 24, 0, 24});
    Mul(dhat2Local[8], scaleLocal_[8],dhatBeforeNormLocal[8] ,ELEMENTS_SIZE_PER_REPEAT,tileRepeatTimes_,{3, 0, 3, 24, 0, 24});
    Mul(dhat2Local[16], scaleLocal_[8],dhatBeforeNormLocal[16] ,ELEMENTS_SIZE_PER_REPEAT,tileRepeatTimes_,{3, 0, 3, 24, 0, 24});
    OutQueue.EnQue(dhat2Local);
    dhat2Local = OutQueue.DeQue<T>();
    DataCopyPad(gradH2Global_[taskOffset * (n_ * n_ + 2 * n_)], dhat2Local, {static_cast<uint16_t>(1),
        static_cast<uint32_t>(tileTaskCount * (n_ * n_ + 2 * n_) * sizeof(float)), 0, 0, 0});
    // if (blkIdx_ == 0) {
    //     DumpTensor(dhat2Local,__LINE__,512);
    // }
    OutQueue.FreeTensor(dhat2Local);
    // PipeBarrier<PIPE_V>();
    // reduce dscaleres



    // 
    // uint32_t thread_num = min(static_cast<uint32_t>(THREAD_NUM), static_cast<uint32_t>(tileTaskCount * n_ * 2));
    // uint32_t xThreadCount = 2 * n_;
    // uint32_t yThreadCount = thread_num / xThreadCount;
    // LocalTensor<T> tmpLocal_ = gradHPostLocal_;
    // Simt::VF_CALL<ComputeGradFusedGradH2PreAndGradH2PostSimt<T>>(Simt::Dim3{xThreadCount, yThreadCount},
    //     (__ubuf__ T*) gradhatLocal_.GetPhyAddr(),
    //     (__ubuf__ T*) gradRsqrtLocal_.GetPhyAddr(),
    //     (__gm__ T*) gradHcScaleGlobal_.GetPhyAddr(),
    //     (__gm__ T*) gradHcBaseGlobal_.GetPhyAddr(),
    //     (__ubuf__ T*) fusedGradHPreAndGradHPostLocal_.GetPhyAddr(),
    //     (__ubuf__ T*) hat2Local.GetPhyAddr(),
    //     (__ubuf__ T*) rsqrtLocal_.GetPhyAddr(),
    //     (__ubuf__ T*) hcBaseLocal_.GetPhyAddr(),
    //     (__ubuf__ T*) tmpLocal_.GetPhyAddr(),  // gradHPostLocal_ 用作临时空间
    //     hcScalePre_, hcScalePost_, tileTaskCount, n_);
}


template<typename TYPE_X, typename T>
__aicore__ inline void MhcPreGradKernel<TYPE_X, T>::ComputeGradX1(const int32_t taskOffset, const int32_t tileTaskCount, const int32_t innerId)
{


    PipeBarrier<PIPE_V>();



    // AscendC::Broadcast<float, 2, 1>(dRsqrtBrcbLocal_, gradRsqrtLocal_, dstRsqrtShape, srcRsqrtShape)
    PipeBarrier<PIPE_V>();

    for (int32_t loopIdC = 0; loopIdC < c1Align_; loopIdC+=1) {
        int64_t copyLen = c0_;
        bool isPad = false;
        uint8_t padLen = 0;
        int64_t ubAlignC = c0_;
        if (loopIdC == c1_) {
            isPad = true;
            copyLen = cTail_;
            ubAlignC = cTailAlign_;
            padLen = static_cast<uint8_t>(cTailAlign_ - cTail_);

        }
        xLocal_ = inputXInQueue.AllocTensor<TYPE_X>();
        gradXCubeLocal_ = xLocal_.template ReinterpretCast<float>()[onceTask_ * n_ * c0_ / 2];
        gradYLocal_ =  xLocal_[onceTask_ * n_ * c0_ * 3];
        
        DataCopyPad(gradYLocal_, gradYGlobal_[taskOffset * c_ + c0_ * loopIdC], {static_cast<uint16_t>(tileTaskCount),
            static_cast<uint32_t>(copyLen * sizeof(TYPE_X)), static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0, 0}, {isPad, 0, padLen, 0});
        DataCopyPad(xLocal_, xGlobal_[taskOffset * n_ * c_ + c0_ * loopIdC], {static_cast<uint16_t>(n_ * tileTaskCount),
            static_cast<uint32_t>(copyLen * sizeof(TYPE_X)), static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0, 0}, {isPad, 0, padLen, 0});
        DataCopyPad(gradXCubeLocal_, gradXCubeGlobal_[taskOffset * n_ * c_ + c0_ * loopIdC], {static_cast<uint16_t>(n_ * tileTaskCount),
            static_cast<uint32_t>(copyLen * sizeof(T)), static_cast<uint32_t>((c_ - copyLen) * sizeof(T)), 0, 0}, {isPad, 0, padLen, 0});

        inputXInQueue.EnQue(xLocal_);
        inputXInQueue.DeQue();
        PipeBarrier<PIPE_V>();
        Cast(gradYCastLocal_, gradYLocal_, RoundMode::CAST_NONE,ubAlignC * tileTaskCount);

        Cast(xCastLocal_, xLocal_, RoundMode::CAST_NONE,ubAlignC * n_ * tileTaskCount);

        PipeBarrier<PIPE_V>();
        uint8_t blkStride1 = static_cast<uint8_t>(ubAlignC / ELEMENTS_SIZE_PER_BLOCK);
        uint8_t blkStride2 = static_cast<uint8_t>(n_ * ubAlignC / ELEMENTS_SIZE_PER_BLOCK);
        for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
            for (int32_t loopOffsetC0 = 0; loopOffsetC0 < copyLen; loopOffsetC0+=ELEMENTS_SIZE_PER_REPEAT) {
                uint64_t mask = min(static_cast<uint64_t>(ELEMENTS_SIZE_PER_REPEAT), static_cast<uint64_t>(copyLen - loopOffsetC0));
                Mul(gradXCastLocal_[loopIdN * ubAlignC + loopOffsetC0 ], gradYCastLocal_[loopOffsetC0 ],
                    preBrcbLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK + innerId * n_ * ELEMENTS_SIZE_PER_BLOCK], mask, tileTaskCount,
                    { 1, 1, 1, blkStride2, blkStride1, static_cast<uint8_t>(n_) });
            }
        }
        PipeBarrier<PIPE_V>();
        for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
            for (int32_t loopOffsetC0 = 0; loopOffsetC0 < copyLen; loopOffsetC0+=ELEMENTS_SIZE_PER_REPEAT) {
                uint64_t mask = min(static_cast<uint64_t>(ELEMENTS_SIZE_PER_REPEAT), static_cast<uint64_t>(copyLen - loopOffsetC0));
                MulAddDst(gradXCastLocal_[loopIdN * ubAlignC + loopOffsetC0], xCastLocal_[loopIdN * ubAlignC + loopOffsetC0], dRsqrtBrcbLocal_[innerId * ELEMENTS_SIZE_PER_BLOCK],
                    mask, tileTaskCount, { 1, 1, 1, blkStride2, blkStride2, 1 });
            }
        }


        Add(gradXCastLocal_, gradXCubeLocal_, gradXCastLocal_, ubAlignC * n_ * tileTaskCount);
        PipeBarrier<PIPE_V>();
        inputXInQueue.FreeTensor(xLocal_);

        auto gradXLocalOut = OutQueue.AllocTensor<TYPE_X>();

        Cast(gradXLocalOut, gradXCastLocal_, RoundMode::CAST_RINT,ubAlignC * n_ * tileTaskCount);
        OutQueue.EnQue(gradXLocalOut);
        gradXLocalOut = OutQueue.DeQue<TYPE_X>();
    
        DataCopyPad(gradXGlobal_[taskOffset * n_ * c_ + c0_ * loopIdC], gradXLocalOut, {static_cast<uint16_t>(n_ * tileTaskCount),
            static_cast<uint32_t>(copyLen * sizeof(TYPE_X)), 0, static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0});
        OutQueue.FreeTensor(gradXLocalOut);

        PipeBarrier<PIPE_V>();
    }
}
template<typename TYPE_X, typename T>
__aicore__ inline void MhcPreGradKernel<TYPE_X, T>::ComputeGradPre(const int32_t taskOffset, const int32_t tileTaskCount, const int32_t innerId)
{
    Duplicate(tmpLocal_, 0.f, tileTaskCount * n_ * ELEMENTS_SIZE_PER_REPEAT);
    for (int32_t loopIdC = 0; loopIdC < c1Align_; loopIdC+=1) {
        int64_t copyLen = c0_;
        bool isPad = false;
        uint8_t padLen = 0;
        int64_t ubAlignC = c0_;
        if (loopIdC == c1_) {
            isPad = true;
            copyLen = cTail_;
            ubAlignC = cTailAlign_;
            padLen = static_cast<uint8_t>(cTailAlign_ - cTail_);
            // Duplicate(gradYLocal_, 0.f, c0_ - cTailAlign_, tileTaskCount, 1, c0_ / ELEMENTS_SIZE_PER_BLOCK);
            // Duplicate(xLocal_, 0.f, c0_ - cTailAlign_, tileTaskCount * n_, 1, c0_ / ELEMENTS_SIZE_PER_BLOCK);
        }
        auto xLocal_ = inputXInQueue.AllocTensor<TYPE_X>();

        auto gradYLocal_ =  xLocal_[tileTaskCount *n_* c0_];
        // // printf("loopIdC %d,taskOffset %d ,tileTaskCount %d ,innerId %d ",loopIdC,taskOffset,tileTaskCount,innerId);
        DataCopyPad(gradYLocal_, gradYGlobal_[taskOffset * c_ + c0_ * loopIdC], {static_cast<uint16_t>(tileTaskCount),
            static_cast<uint32_t>(copyLen * sizeof(TYPE_X)), static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0, 0}, {isPad, 0, padLen, 0});

        
        DataCopyPad(xLocal_, xGlobal_[taskOffset * n_ * c_ + c0_ * loopIdC], {static_cast<uint16_t>(n_ * tileTaskCount),
            static_cast<uint32_t>(copyLen * sizeof(TYPE_X)), static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0, 0}, {isPad, 0, padLen, 0});
        // // printf("loopIdC %d,taskOffset %d ,tileTaskCount %d ,innerId %d ",loopIdC,taskOffset,tileTaskCount,innerId);

        
        inputXInQueue.EnQue(xLocal_);
        inputXInQueue.DeQue();
        PipeBarrier<PIPE_V>();
        // // printf("__LINE__ %d \n",__LINE__);

        auto xCastOutLocal = OutQueue.AllocTensor<float>();
        // // DumpTensor(xCastOutLocal,__LINE__,32);
        // // printf("__LINE__ %d \n",__LINE__);
        // // printf("ubAlignC * n_ * tileTaskCount %d \n",ubAlignC * n_ * tileTaskCount);
        // // DumpTensor(xCastOutLocal[8],__LINE__,32);

        Cast(xCastOutLocal, xLocal_, RoundMode::CAST_NONE,ubAlignC * n_ * tileTaskCount);
        OutQueue.EnQue(xCastOutLocal);
        xCastOutLocal = OutQueue.DeQue<T>();
        // // printf("__LINE__33 %d \n",blkIdx_);

        DataCopyPad(xWorkspaceGlobal_[taskOffset * n_ * c_ + c0_ * loopIdC], xCastOutLocal,
            {static_cast<uint16_t>(n_ * tileTaskCount),
             static_cast<uint32_t>(copyLen * sizeof(float)),
             0, static_cast<uint32_t>((c_ - copyLen) * sizeof(float)), 0});
        // // printf("__LINE__ %d \n",__LINE__);

        Cast(gradYCastLocal_, gradYLocal_, RoundMode::CAST_NONE,ubAlignC * tileTaskCount);
        inputXInQueue.FreeTensor(xLocal_);
        // // printf("__LINE__ %d \n",__LINE__);
        // if (taskOffset == 0) {

        //     DumpTensor(xCastOutLocal,__LINE__,512);
        //     DumpTensor(gradYLocal_,__LINE__,512);
        //     DumpTensor(gradYGlobal_,__LINE__,512);

        //     DumpTensor(gradYCastLocal_,__LINE__,512);

        // }

        PipeBarrier<PIPE_V>();
        uint8_t blkStride = static_cast<uint8_t>(ubAlignC / ELEMENTS_SIZE_PER_BLOCK);

        uint8_t blkStride3 = static_cast<uint8_t>(n_ * ubAlignC / ELEMENTS_SIZE_PER_BLOCK);
        for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
            for (int32_t loopOffsetC0 = 0; loopOffsetC0 < copyLen; loopOffsetC0+=ELEMENTS_SIZE_PER_REPEAT) {
                uint64_t mask = min(static_cast<uint64_t>(ELEMENTS_SIZE_PER_REPEAT), static_cast<uint64_t>(copyLen - loopOffsetC0));
                Mul(xCastOutLocal[loopIdN * ubAlignC + loopOffsetC0], xCastOutLocal[loopIdN * ubAlignC + loopOffsetC0],
                    gradYCastLocal_[loopOffsetC0],
                    mask, tileTaskCount,
                    { 1, 1, 1, blkStride3, blkStride3, blkStride});
            }
        }
        PipeBarrier<PIPE_V>();
        // // printf("__LINE__ %d \n",__LINE__);
        // if (taskOffset == 0) {
        //     DumpTensor(xCastOutLocal,__LINE__,512);
        //     DumpTensor(xCastOutLocal[512],__LINE__,512);
        //     DumpTensor(xCastOutLocal[1024],__LINE__,512);
        //     DumpTensor(xCastOutLocal[1024+512],__LINE__,512);

        // }

        int64_t reduceLen = ubAlignC;
        if (loopIdC != c1_) {
            Add(xCastOutLocal[64], xCastOutLocal[64], xCastOutLocal[128 + 64], ELEMENTS_SIZE_PER_REPEAT, tileTaskCount * n_,
                {1, 1, 1, blkStride, blkStride, blkStride});
            Add(xCastOutLocal, xCastOutLocal, xCastOutLocal[128], ELEMENTS_SIZE_PER_REPEAT, tileTaskCount * n_,
                {1, 1, 1, blkStride, blkStride, blkStride});
            PipeBarrier<PIPE_V>();

            Add(xCastOutLocal, xCastOutLocal, xCastOutLocal[64], ELEMENTS_SIZE_PER_REPEAT, tileTaskCount * n_,
                {1, 1, 1, blkStride, blkStride, blkStride});
        } else {
            if (cTail_ - (128 + 64) > 0) {
                uint64_t mask = min(static_cast<uint64_t>(cTail_ - (128 + 64)), static_cast<uint64_t>(REPEAT_LENTH));

                Add(xCastOutLocal[64], xCastOutLocal[64], xCastOutLocal[128 + 64], mask, tileTaskCount * n_,
                    {1, 1, 1, blkStride, blkStride, blkStride});
            }

            if (cTail_ - (128) > 0) {
                uint64_t mask = min(static_cast<uint64_t>(cTail_ - (128)), static_cast<uint64_t>(REPEAT_LENTH));

                Add(xCastOutLocal, xCastOutLocal, xCastOutLocal[128], mask, tileTaskCount * n_,
                    {1, 1, 1, blkStride, blkStride, blkStride});
            }
            PipeBarrier<PIPE_V>();
            if (cTail_ - (64) > 0) {
                uint64_t mask = min(static_cast<uint64_t>(cTail_ - (64)), static_cast<uint64_t>(REPEAT_LENTH));

                Add(xCastOutLocal, xCastOutLocal, xCastOutLocal[64], mask, tileTaskCount * n_,
                    {1, 1, 1, blkStride, blkStride, blkStride});  
            }
        }
        PipeBarrier<PIPE_V>();

        uint64_t mask = min(static_cast<uint64_t>(cTail_) , static_cast<uint64_t>(REPEAT_LENTH));
        // // printf("__LINE__ %d \n",__LINE__);
        // if (taskOffset == 0) {
        //     DumpTensor(xCastOutLocal,__LINE__,32);
        //     DumpTensor(xCastOutLocal[512],__LINE__,32);
        //     DumpTensor(xCastOutLocal[1024],__LINE__,32);
        //     DumpTensor(xCastOutLocal[1024+512],__LINE__,32);

        //     printf("mask %d \n",mask);
        // }
        PipeBarrier<PIPE_V>();
        
        Add(tmpLocal_, tmpLocal_, xCastOutLocal, mask, tileTaskCount * n_,
            {1, 1, 1, 8, 8, blkStride});
        PipeBarrier<PIPE_V>();

        // if (taskOffset == 0) {
        //     DumpTensor(tmpLocal_,__LINE__,512);
        // }
        // if (taskOffset == 0) {
        //     DumpTensor(dPrePostTempLocal_,__LINE__,512);

        // }
        OutQueue.FreeTensor(xCastOutLocal);
    }
    PipeBarrier<PIPE_V>();
    PipeBarrier<PIPE_V>();
    // if (taskOffset == 0) {
    //     // printf("inner12Id %d \n",innerId);

    //     DumpTensor(tmpLocal_,__LINE__,512);
    // }
    for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
        WholeReduceSum(dPrePostTempLocal_[loopIdN + innerId * n_ * 2], tmpLocal_[loopIdN * REPEAT_LENTH], REPEAT_LENTH,
            tileTaskCount, n_ * 2, 1, n_ * 8);// dst repeat stride  srcblkstride srcrepstride
        // dst 间隔 8 2 *n
    }


    // if (blkIdx_ == 0) {
    //     // printf("inner34Id %d \n",innerId);

    //     DumpTensor(dPrePostTempLocal_,__LINE__,512);
    // }
}





template<typename TYPE_X, typename T>
__aicore__ inline void MhcPreGradKernel<TYPE_X, T>::ComputeScaleBias()
{

    // 调换顺序
    Add(dScaleLocal_[8], dScaleLocal_[8], dScaleLocal_[16],ELEMENTS_SIZE_PER_REPEAT,tileRepeatTimes_,{3, 3, 3, 24, 24, 24});
    // 和tileTaskCount相关
    PipeBarrier<PIPE_V>();
    // tileCoreBS_ 必须8 *2的幂次
    for (int32_t bsCount = tileCoreBS_ / 2; bsCount > 0; bsCount = bsCount / 2) {
        Add(dBiasLocal_, dBiasLocal_, dBiasLocal_[bsCount * (n_ * n_ + 2 * n_)], bsCount * (n_ * n_ + 2 * n_));
        Add(dScaleLocal_, dScaleLocal_, dScaleLocal_[bsCount * (n_ * n_ + 2 * n_)], bsCount * (n_ * n_ + 2 * n_));
        PipeBarrier<PIPE_V>();
    }
    if (blkIdx_ == 0) {
        // printf("inner34Id %d \n",innerId);
        DumpTensor(dScaleLocal_,__LINE__,512);

        DumpTensor(dBiasLocal_,__LINE__,512);
    }
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3XCast);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3XCast);

    SetAtomicAdd<T>();

    LocalTensor<T> dscaleOut = tempBuf_.GetWithOffset<T>(ELEMENTS_SIZE_PER_BLOCK, 0);

    DataCopyPad(gradHcBaseGlobal_, dBiasLocal_, {static_cast<uint16_t>(1),
        static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0});
    PipeBarrier<PIPE_V>();

    WholeReduceSum(dscaleOut, dScaleLocal_, 4, 1, 1, 3, 8);// dst repeat stride  srcblkstride srcrepstride
    WholeReduceSum(dscaleOut[1], dScaleLocal_, MASK_POST_SCALE, 1, 1, 3, 8);// dst repeat stride  srcblkstride srcrepstride
    WholeReduceSum(dscaleOut[2], dScaleLocal_[8], 16, 1, 1, 3, 8);// dst repeat stride  srcblkstride srcrepstride
    if (blkIdx_ == 0) {
        // printf("inner34Id %d \n",innerId);
        DumpTensor(dscaleOut,__LINE__,32);

    }
    
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3XCast);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3XCast);
    // DataCopyPad(gradHcScaleGlobal_, dscaleOut, {static_cast<uint16_t>(1),
    //     static_cast<uint32_t>((3) * sizeof(T)), 0, 0, 0});
    SetAtomicNone();

}
/**
    约束：
    c: c >= 64 && c % 64 == 0
    n: n == 4
*/

template<typename TYPE_X, typename T>
__aicore__ inline void MhcPreGradKernel<TYPE_X, T>::Process()
{

    if ASCEND_IS_AIV {
        GetHcScaleAndHcBase();

        int8_t ping = 0;
        if (blkIdx_ == aivNum_ - 1) {
            InitOutput<T>(gradHcBaseGlobal_, (n_ * n_ + 2 * n_), 0);
            InitOutput<T>(gradHcScaleGlobal_, 3, 0);
        }
        for (int64_t taskOffset = blkIdx_ * tileCoreBS_; taskOffset < n_ * c_; taskOffset += aivNum_ * tileCoreBS_) {
            int32_t tileTaskCount = min(static_cast<int32_t>(tileCoreBS_), static_cast<int32_t>(n_ * c_ - taskOffset));
            InitOutput<T>(gradWeightGlobal_[taskOffset * (n_ * n_ + 2 * n_)], (n_ * n_ + 2 * n_) * tileTaskCount, 0);

        }
        SyncAll<true>();
        // // DumpTensor(gradWeightGlobal_,__LINE__,32 * 8 +32);

        for (int32_t taskOffset = blkIdx_ * tileCoreBS_; taskOffset < totalTasksAligned_; taskOffset += aivNum_ * tileCoreBS_) {
            int32_t tileTaskCount = min(static_cast<int32_t>(tileCoreBS_), static_cast<int32_t>(totalTasks_ - taskOffset));
            tileRepeatTimes_ = CeilDiv(tileTaskCount * 2 * n_, ELEMENTS_SIZE_PER_REPEAT);
            if (tileTaskCount > 0) {
                int32_t innerId = 0;
                Duplicate(dPrePostTempLocal_, 0.f, tileCoreBS_ * n_ * 2);

                for (int32_t taskOffsetInner = 0; taskOffsetInner < tileTaskCount; taskOffsetInner += onceTask_) {
                    int32_t tileTaskCountInner = min(static_cast<int32_t>(onceTask_), static_cast<int32_t>(tileTaskCount - taskOffsetInner));

                    ComputeGradPre(taskOffset + taskOffsetInner, tileTaskCountInner, taskOffsetInner);
                    innerId++;
                    // printf("innerId222 %d \n",innerId);
                    if (blkIdx_ <  3) {

                    // DumpTensor(dPrePostTempLocal_,__LINE__,512);
                    // DumpTensor(dPrePostTempLocal_[512],__LINE__,512);
                    }
                }

                
                if (taskOffset == 0) {

                    DumpTensor(scaleLocal_,__LINE__,512);

                }
                if (blkIdx_ <  3) {


                // DumpTensor(dPrePostTempLocal_,__LINE__,512);
                // DumpTensor(dPrePostTempLocal_[512],__LINE__,512);
                }
                ComputeGradHHat2(taskOffset, tileTaskCount);
            }
            CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
            CrossCoreWaitFlag<0x2>(1);
            ping = (ping + 1) % 10;

            if (tileTaskCount > 0) {
                int32_t innerId = 0;
                for (int32_t taskOffsetInner = 0; taskOffsetInner < tileTaskCount; taskOffsetInner += onceTask_) {
                    int32_t tileTaskCountInner = min(static_cast<int32_t>(onceTask_), static_cast<int32_t>(tileTaskCount - taskOffsetInner));
                        int32_t brcbAlign = CeilDiv(tileTaskCount * n_, ELEMENTS_SIZE_PER_BLOCK);
                    int32_t offset = 0;
                    auto preLocal_ = inputGradQueue.AllocTensor<T>();
                    DataCopyPad(preLocal_, preGlobal_[taskOffset * n_], {static_cast<uint16_t>(1),
                        static_cast<uint32_t>(tileTaskCount * n_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
                    inputGradQueue.EnQue(preLocal_);
                    inputGradQueue.DeQue();

                    const uint32_t srcShape[2] = {static_cast<uint32_t>(tileTaskCount * n_), 1}; // 源数据shape
                    const uint32_t dstShape[2] = {static_cast<uint32_t>(tileTaskCount * n_), ELEMENTS_SIZE_PER_BLOCK};// broadcast数据shape
                    preBrcbLocal_ = tempBuf_.GetWithOffset<T>(brcbAlign * 8, offset);
                    Brcb(preBrcbLocal_, preLocal_, brcbAlign, {static_cast<uint8_t>(1), static_cast<uint8_t>(8)});
                    inputGradQueue.FreeTensor(preLocal_);


                    offset += brcbAlign * 8 * sizeof(float);
                    brcbAlign = CeilDiv(tileTaskCount, ELEMENTS_SIZE_PER_BLOCK);

                //    dRsqrtBrcbLocal_ = tempBuf_.GetWithOffset<T>(brcbAlign * 8, offset);
                    offset += brcbAlign * 8 * sizeof(float);

                    // AscendC::Broadcast<float, 2, 1>(preBrcbLocal_, preLocal_, dstShape, srcShape)
                    
                    const uint32_t srcRsqrtShape[2] = {static_cast<uint32_t>(tileTaskCount), 1}; // 源数据shape
                    const uint32_t dstRsqrtShape[2] = {static_cast<uint32_t>(tileTaskCount), ELEMENTS_SIZE_PER_BLOCK};// broadcast数据shape
                    Brcb(dRsqrtBrcbLocal_, gradRsqrtLocal_, brcbAlign, {static_cast<uint8_t>(1), static_cast<uint8_t>(8)});
                    ComputeGradX1(taskOffset + taskOffsetInner, tileTaskCountInner, taskOffsetInner);
                    innerId++;
                }
            }
        }
        ComputeScaleBias();


        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIdMTE3ToVXCast);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMTE3XCast);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIdMTE3ToVGradX);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMTE3GradX);
    }

    if ASCEND_IS_AIC {
        int8_t ping = 0;
        for (int32_t taskOffset = blkIdx_ * 2 * tileCoreBS_; taskOffset < totalTasksAligned_; taskOffset += aicNum_ * 2 * tileCoreBS_) {
            int32_t tileTaskCount = min(static_cast<int32_t>(2 * tileCoreBS_), static_cast<int32_t>(totalTasks_ - taskOffset));
            CrossCoreWaitFlag<0x2>(0);

            if (tileTaskCount > 0) {
                ProcessMatmul1(taskOffset, tileTaskCount);
                ProcessMatmul2(taskOffset, tileTaskCount);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(1);

            ping = (ping + 1) % 10;
        }
    }
}

template<typename TYPE_X, typename T>
__aicore__ inline void MhcPreGradKernel<TYPE_X, T>::ProcessMatmul1(const int32_t taskOffset, const int32_t mm1M)
{
    if (mm1M <= 0)
        return;

    mm1_.SetTensorA(gradH2Global_[taskOffset * mm1K_]);
    mm1_.SetTensorB(weightGlobal_);
    mm1_.SetHF32(false, 1);
    mm1_.SetOrgShape(mm1M, mm1N_, mm1K_);
    mm1_.SetSingleShape(mm1M, mm1N_, mm1K_);
    mm1_.template IterateAll<false> (gradXCubeGlobal_[taskOffset * (n_ * c_)], 1);
    mm1_.End();
}

template<typename TYPE_X, typename T>
__aicore__ inline void MhcPreGradKernel<TYPE_X, T>::ProcessMatmul2(const int32_t taskOffset, const int32_t mm2K)
{
    if (mm2K <= 0)
        return;

    mm2_.SetTensorA(gradH2Global_[taskOffset * mm2M_], true);
    mm2_.SetTensorB(xWorkspaceGlobal_[taskOffset * mm2N_]);
    mm2_.SetHF32(false, 1);
    mm2_.SetOrgShape(mm2M_, mm2N_, mm2K);
    mm2_.SetSingleShape(mm2M_, mm2N_, mm2K);
    mm2_.template IterateAll<false> (gradWeightGlobal_, 1);
    mm2_.End();
}

#endif // MHC_PRE_GRAD_H