/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */
/*!
 * \file mhc_pre_sinkhorn_backward_deterministic.h
 * \brief MhcPreSinkhornBackward Deterministic Kernel (arch35, RegBase VF)
 * 
 * Based on golden_v2.py reference implementation
 */

#ifndef MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_H
#define MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "mhc_pre_sinkhorn_backward_data_arch35.h"
#include "mhc_pre_sinkhorn_backward_deterministic_vf.h"

using namespace AscendC;
using namespace MicroAPI;

// Note: FP32_BYTE_SIZE, DOUBLE_BUF, FP32_PER_VL come from simd_vf.h/one_stage.h
// via apt.cpp include chain. deterministic.h must be included after one_stage.h.

namespace {
constexpr int32_t FLAG_ID_AIV_DONE = 0;
constexpr int32_t FLAG_ID_LLH_DONE = 1;
constexpr int32_t FLAG_ID_HBJ_DONE = 2;
constexpr int32_t FLAG_ID_LHP_DONE = 3;
constexpr int32_t FLAG_ID_ALL_DONE = 4;
constexpr float HC_EPS = 1e-6f;
constexpr uint8_t SYNC_MODE1 = 2;

constexpr MatmulConfig MM1_CFG = GetMDLConfig(false, false, 0, true, false, false, true);
constexpr MatmulConfig MM2_CFG = GetMDLConfig(false, false, 0, true, false, false, true);
}

template<typename T, typename U>
class MhcPreSinkhornBackwardDeterministic {
public:
    using A0Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using C1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, U>;
    using C2Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    
    matmul::MatmulImpl<A0Type, BType, C1Type, C1Type, MM1_CFG> mm1_;
    matmul::MatmulImpl<A0Type, BType, C2Type, C2Type, MM2_CFG> mm2_;
    
    __aicore__ inline MhcPreSinkhornBackwardDeterministic() = default;
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR phi, GM_ADDR h_pre, GM_ADDR gradHin, GM_ADDR gradHPost,
        GM_ADDR gradHRes, GM_ADDR alpha, GM_ADDR bias, GM_ADDR hcBeforeNorm, GM_ADDR invRms,
        GM_ADDR sumOut, GM_ADDR normOut, GM_ADDR gradX, GM_ADDR gradPhi,
        GM_ADDR gradAlpha, GM_ADDR gradBias, GM_ADDR workspace,
        const MhcPreSinkhornBackwardArch35TilingData* tilingData, TPipe* pipe);
    
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTiling(const MhcPreSinkhornBackwardArch35TilingData* tilingData);
    __aicore__ inline void InitGM(GM_ADDR x, GM_ADDR phi, GM_ADDR h_pre, GM_ADDR gradHin, GM_ADDR gradHPost,
        GM_ADDR gradHRes, GM_ADDR alpha, GM_ADDR bias, GM_ADDR hcBeforeNorm, GM_ADDR invRms,
        GM_ADDR sumOut, GM_ADDR normOut, GM_ADDR gradX, GM_ADDR gradPhi,
        GM_ADDR gradAlpha, GM_ADDR gradBias, GM_ADDR workspace);
    __aicore__ inline void InitBuffer();
    
    // AIV计算流（按golden_v2.py步骤）
    // 【责任人: llh】步骤1-5: 输入聚合、前向z计算、sigmoid梯度、alpha/bias梯度
    __aicore__ inline void Step1_InputAggregation();           // grad_h_pre = (grad_hin*x).sum(-1), grad_x第一路写入
    __aicore__ inline void Step2_ComputeForwardZ();           // z = norm_out*alpha+bias, split三路
    __aicore__ inline void Step3_SigmoidGrad();               // grad_z_pre/post, 系数2修正
    __aicore__ inline void Step45_AlphaBiasGradPrePost();     // grad_alpha/bias累加到workspace
    
    // 【责任人: hbj】步骤6-8: Sinkhorn反向、Exp梯度、res支路alpha/bias梯度
    __aicore__ inline void Step678_SinkhornExpGrad();         // sk_grad + grad_z_res + grad_alpha/bias累加
    
    // 【责任人: lhp】步骤9-15: Linear梯度拼接、RMSNorm梯度、Matmul梯度、grad_x汇总
    __aicore__ inline void Step911_LinearGradConcat();       // grad_norm_out = grad_z*alpha (三路拼接)
    __aicore__ inline void Step12_RMSNormGrad();             // grad_hc = grad_norm_out*inv_rms
    __aicore__ inline void Step14_RMSNormInvGrad();          // grad_x_from_rms (第二路累加写入)
    __aicore__ inline void Step15_MatmulGradX();             // grad_hc写入GM供AIC matmul
    __aicore__ inline void Step15_MatmulGradPhi();           // grad_phi累加到workspace
    
    // AIC计算流
    // 【责任人: lhp】步骤15: Cube硬件Matmul
    __aicore__ inline void AIC_MatmulGradX();                 // grad_x第三路AtomicAdd累加
    __aicore__ inline void AIC_MatmulGradPhi();               // grad_phi累加
    
    // 确定性最终输出
    // 【责任人: zyf】步骤16: workspace归约，grad_alpha/bias组装
    __aicore__ inline void ReduceWorkspaceToGM();

private:
    TPipe* pipe_;
    int64_t blkIdx_, aivNum_, aicNum_;
    int64_t batchSize_, seqLength_, totalTasks_, totalTasksAligned_;
    int64_t c_, n_;  // c=D(feature_dim), n=N(hidden_dim)
    int64_t tile_, tileAligned_;
    int64_t skIterCount_;
    
    T alphaPre_, alphaPost_, alphaRes_;
    
    int64_t wsGradAlphaOffset_, wsGradBiasOffset_, wsGradPhiOffset_, wsGradXOffset_;
    int64_t wsXFp32Offset_, wsGradXFromRmsOffset_, wsPerCoreSize_;
    int64_t c0_, cTail_;  // D切分参数（从tiling读取）
    int64_t D_chunk_count_;
    
    // Queue（流水线）
    TQue<QuePosition::VECIN, DOUBLE_BUF> hcBeforeNormQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUF> invRmsQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUF> gradHResQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUF> gradHcQueue_;
    
    // TBuf（单buffer）
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> alphaBuf_;
    TBuf<TPosition::VECCALC> biasBuf_;
    TBuf<TPosition::VECCALC> hPreBuf_;
    TBuf<TPosition::VECCALC> gradAlphaBuf_;
    TBuf<TPosition::VECCALC> gradBiasBuf_;
    
    TBuf<TPosition::VECIN> sumOutBuf_;
    TBuf<TPosition::VECIN> normOutBuf_;
    
    TBuf<TPosition::VECCALC> zPreBuf_, zPostBuf_, zResBuf_;
    TBuf<TPosition::VECCALC> gradZPreBuf_, gradZPostBuf_, gradZResBuf_;
    TBuf<TPosition::VECCALC> gradNormOutBuf_;
    
    TBuf<TPosition::VECCALC> gradHPreBuf_;
    TBuf<TPosition::VECCALC> normOutForwardBuf_;
    
    // GlobalTensor
    GlobalTensor<U> xGlobal_;
    GlobalTensor<T> phiGlobal_;
    GlobalTensor<T> hPreGlobal_;
    GlobalTensor<U> gradHinGlobal_;
    GlobalTensor<T> gradHPostGlobal_;
    GlobalTensor<T> gradHResGlobal_;
    GlobalTensor<T> alphaGlobal_, biasGlobal_;
    GlobalTensor<T> hcBeforeNormGlobal_;
    GlobalTensor<T> invRmsGlobal_;
    GlobalTensor<T> sumOutGlobal_, normOutGlobal_;
    GlobalTensor<U> gradXGlobal_;
    GlobalTensor<T> gradPhiGlobal_;
    GlobalTensor<T> gradAlphaGlobal_, gradBiasGlobal_;
    GlobalTensor<T> gradHcGlobal_;
    GlobalTensor<T> workspaceGlobal_;
};

// ========== Init实现==========
template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<T, U>::Init(
    GM_ADDR x, GM_ADDR phi, GM_ADDR h_pre, GM_ADDR gradHin, GM_ADDR gradHPost,
    GM_ADDR gradHRes, GM_ADDR alpha, GM_ADDR bias, GM_ADDR hcBeforeNorm, GM_ADDR invRms,
    GM_ADDR sumOut, GM_ADDR normOut, GM_ADDR gradX, GM_ADDR gradPhi,
    GM_ADDR gradAlpha, GM_ADDR gradBias, GM_ADDR workspace,
    const MhcPreSinkhornBackwardArch35TilingData* tilingData, TPipe* pipe) {
    
    pipe_ = pipe;
    blkIdx_ = GetBlockIdx();
    
    InitTiling(tilingData);
    InitGM(x, phi, h_pre, gradHin, gradHPost, gradHRes, alpha, bias, hcBeforeNorm, invRms,
        sumOut, normOut, gradX, gradPhi, gradAlpha, gradBias, workspace);
    InitBuffer();
}

template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<T, U>::InitTiling(
    const MhcPreSinkhornBackwardArch35TilingData* tilingData) {
    
    batchSize_ = tilingData->batchSize;
    seqLength_ = tilingData->seqLength;
    aivNum_ = tilingData->aivNum;
    aicNum_ = tilingData->aivNum / 2;
    c_ = tilingData->c;
    n_ = tilingData->n;
    skIterCount_ = tilingData->skIterCount;
    tile_ = tilingData->tileSize;
    tileAligned_ = AlignUp(tile_, 32 / sizeof(T));
    totalTasks_ = batchSize_ * seqLength_;
    totalTasksAligned_ = AlignUp(totalTasks_, aivNum_ * tile_);
    
    c0_ = tilingData->c0;
    cTail_ = tilingData->cTail;
    D_chunk_count_ = (c_ + c0_ - 1) / c0_;
    
    alphaPre_ = static_cast<T>(0.5);
    alphaPost_ = static_cast<T>(0.5);
    alphaRes_ = static_cast<T>(0.5);
    
    wsGradAlphaOffset_ = 0;
    wsGradBiasOffset_ = 3 * FP32_BYTE_SIZE;
    wsGradPhiOffset_ = wsGradBiasOffset_ + (n_ * n_ + 2 * n_) * FP32_BYTE_SIZE;
    wsGradXOffset_ = wsGradPhiOffset_ + (n_ * n_ + 2 * n_) * (n_ * c_) * FP32_BYTE_SIZE;
    wsXFp32Offset_ = wsGradXOffset_ + tile_ * n_ * c_ * FP32_BYTE_SIZE;
    wsGradXFromRmsOffset_ = wsXFp32Offset_ + tile_ * n_ * c_ * FP32_BYTE_SIZE;
    wsPerCoreSize_ = wsGradXFromRmsOffset_ + tile_ * n_ * c_ * FP32_BYTE_SIZE;
}

template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<T, U>::InitGM(
    GM_ADDR x, GM_ADDR phi, GM_ADDR h_pre, GM_ADDR gradHin, GM_ADDR gradHPost,
    GM_ADDR gradHRes, GM_ADDR alpha, GM_ADDR bias, GM_ADDR hcBeforeNorm, GM_ADDR invRms,
    GM_ADDR sumOut, GM_ADDR normOut, GM_ADDR gradX, GM_ADDR gradPhi,
    GM_ADDR gradAlpha, GM_ADDR gradBias, GM_ADDR workspace) {
    
    xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(x));
    phiGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(phi));
    hPreGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(h_pre));
    gradHinGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(gradHin));
    gradHPostGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradHPost));
    gradHResGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradHRes));
    alphaGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(alpha));
    biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));
    hcBeforeNormGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(hcBeforeNorm));
    invRmsGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(invRms));
    sumOutGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(sumOut));
    normOutGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(normOut));
    gradXGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(gradX));
    gradPhiGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradPhi));
    gradAlphaGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradAlpha));
    gradBiasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradBias));
    workspaceGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace));
    gradHcGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace) + batchSize_ * seqLength_ * n_ * c_);
}

template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<T, U>::InitBuffer() {
    int64_t tileSize = tile_ * (n_ * n_ + 2 * n_) * FP32_BYTE_SIZE;
    
    pipe_->InitBuffer(hcBeforeNormQueue_, DOUBLE_BUF, tileSize);
    pipe_->InitBuffer(invRmsQueue_, DOUBLE_BUF, tile_ * FP32_BYTE_SIZE);
    pipe_->InitBuffer(gradHResQueue_, DOUBLE_BUF, tile_ * n_ * n_ * FP32_BYTE_SIZE);
    pipe_->InitBuffer(gradHcQueue_, DOUBLE_BUF, tileSize);
    
    pipe_->InitBuffer(tmpBuf_, tile_ * n_ * c0_ * FP32_BYTE_SIZE);
    pipe_->InitBuffer(alphaBuf_, 3 * FP32_BYTE_SIZE);
    pipe_->InitBuffer(biasBuf_, (n_ * n_ + 2 * n_) * FP32_BYTE_SIZE);
    pipe_->InitBuffer(hPreBuf_, tile_ * n_ * FP32_BYTE_SIZE);
    pipe_->InitBuffer(gradAlphaBuf_, 3 * FP32_BYTE_SIZE);
    pipe_->InitBuffer(gradBiasBuf_, (n_ * n_ + 2 * n_) * FP32_BYTE_SIZE);
    
    pipe_->InitBuffer(sumOutBuf_, 2 * skIterCount_ * n_ * FP32_BYTE_SIZE);
    pipe_->InitBuffer(normOutBuf_, 2 * skIterCount_ * n_ * n_ * FP32_BYTE_SIZE);
    
    pipe_->InitBuffer(zPreBuf_, tile_ * n_ * FP32_BYTE_SIZE);
    pipe_->InitBuffer(zPostBuf_, tile_ * n_ * FP32_BYTE_SIZE);
    pipe_->InitBuffer(zResBuf_, tile_ * n_ * n_ * FP32_BYTE_SIZE);
    
    pipe_->InitBuffer(gradZPreBuf_, tile_ * n_ * FP32_BYTE_SIZE);
    pipe_->InitBuffer(gradZPostBuf_, tile_ * n_ * FP32_BYTE_SIZE);
    pipe_->InitBuffer(gradZResBuf_, tile_ * n_ * n_ * FP32_BYTE_SIZE);
    
    pipe_->InitBuffer(gradNormOutBuf_, tileSize);
    
    pipe_->InitBuffer(gradHPreBuf_, tile_ * n_ * FP32_BYTE_SIZE);
    pipe_->InitBuffer(normOutForwardBuf_, tileSize);
}

// ========== Process实现==========
template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<T, U>::Process() {
    
    if ASCEND_IS_AIV {
        for (int64_t taskIdx = blkIdx_; taskIdx < totalTasksAligned_; taskIdx += aivNum_) {
            // 【责任人: llh】步骤1-5
            Step1_InputAggregation();
            Step2_ComputeForwardZ();
            Step3_SigmoidGrad();
            Step45_AlphaBiasGradPrePost();
            CrossCoreSetFlag<2, PIPE_FIX>(FLAG_ID_LLH_DONE);
            
            // 【责任人: hbj】步骤6-8
            Step678_SinkhornExpGrad();
            CrossCoreSetFlag<2, PIPE_FIX>(FLAG_ID_HBJ_DONE);
            
            // 【责任人: lhp】步骤9-15
            Step911_LinearGradConcat();
            Step12_RMSNormGrad();
            Step14_RMSNormInvGrad();
            Step15_MatmulGradX();
            CrossCoreSetFlag<2, PIPE_FIX>(FLAG_ID_LHP_DONE);
        }
        CrossCoreWaitFlag(FLAG_ID_ALL_DONE);
        ReduceWorkspaceToGM();
    }
    
    if ASCEND_IS_AIC {
        for (int64_t taskIdx = blkIdx_; taskIdx < totalTasksAligned_; taskIdx += aicNum_) {
            CrossCoreWaitFlag(FLAG_ID_LHP_DONE);
            
            AIC_MatmulGradX();
            AIC_MatmulGradPhi();
            CrossCoreSetFlag<2, PIPE_FIX>(FLAG_ID_ALL_DONE);
        }
    }
}

// ========== Step 1: 输入聚合（golden_v2.py 117-118行）==========
// 【责任人: llh】
// 【计算过程】:
//   ★D切分循环 + 随路broadcast优化（Muls替代Dup+Mul）
//   1. grad_x_from_hin: 逐D_chunk, Muls随路乘h_pre_scalar，写入workspace
//   2. x_fp32: 逐D_chunk Cast后写入workspace（供Step14使用）
//   3. grad_h_pre: 逐D_chunk累加ReduceSum
template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<T, U>::Step1_InputAggregation() {
    int64_t gmOffset = blkIdx_ * tile_;
    int64_t wsBase = blkIdx_ * wsPerCoreSize_;
    
    // 加载h_pre到UB（[1,N], N维度，放UB安全）
    LocalTensor<T> hPreLocal = hPreBuf_.Get<T>();
    DataCopyPad(hPreLocal, hPreGlobal_[gmOffset * n_], {1, static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    AscendC::PipeBarrier<PIPE_ALL>();
    
    // ★D切分参数
    uint32_t c0_u = static_cast<uint32_t>(c0_);
    uint32_t cTail_u = static_cast<uint32_t>(cTail_);
    uint32_t c_u = static_cast<uint32_t>(c_);
    uint32_t n_u = static_cast<uint32_t>(n_);
    uint32_t D_chunk_cnt = static_cast<uint32_t>(D_chunk_count_);
    
    // ★grad_h_pre = (grad_hin * x).sum(-1) [golden 118行]
    // 逐D_chunk累加：先初始化为0，然后逐chunk累加
    LocalTensor<T> gradHPreLocal = gradHPreBuf_.Get<T>();
    Duplicate(gradHPreLocal, static_cast<T>(0), n_);
    
    LocalTensor<U> gradHinInput = tmpBuf_.Get<U>();
    LocalTensor<U> xInput = tmpBuf_.Get<U>();
    LocalTensor<T> gradHinFp32Chunk = tmpBuf_.Get<T>();
    LocalTensor<T> xFp32Chunk = tmpBuf_.Get<T>();
    LocalTensor<T> gradXChunk = tmpBuf_.Get<T>();
    LocalTensor<T> mulHinXChunk = tmpBuf_.Get<T>();
    
    // ★逐N × 逐D_chunk: grad_x_from_hin写入workspace + grad_h_pre累加 + x_fp32写入workspace
    for (uint16_t j = 0; j < n_u; j++) {
        T hPreScalar = hPreLocal.GetValue(j);
        
        for (uint16_t d_idx = 0; d_idx < D_chunk_cnt; d_idx++) {
            uint32_t cur_c = (d_idx == D_chunk_cnt - 1) ? cTail_u : c0_u;
            uint32_t d_offset = d_idx * c0_u;
            
            // 从GM加载grad_hin[1, d_offset:d_offset+cur_c]
            DataCopyPad(gradHinInput, gradHinGlobal_[gmOffset * c_u + d_offset], {1, static_cast<uint32_t>(cur_c * sizeof(U)), 0, 0, 0}, {false, 0, 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();
            Cast(gradHinFp32Chunk, gradHinInput, RoundMode::CAST_NONE, cur_c);
            
            // ★随路broadcast: Muls(grad_x_chunk, grad_hin_fp32_chunk, h_pre_scalar, cur_c)
            // 不做真dup，Muls直接乘标量
            Muls(gradXChunk, gradHinFp32Chunk, hPreScalar, cur_c);
            
            // ★写入workspace: wsGradXFromHin[j*c + d_offset]
            int64_t wsGradXOffset_j = wsBase + wsGradXOffset_ + j * c_ * FP32_BYTE_SIZE + d_offset * FP32_BYTE_SIZE;
            DataCopyPad(workspaceGlobal_[wsGradXOffset_j / sizeof(T)], gradXChunk, {1, static_cast<uint32_t>(cur_c * sizeof(T)), 0, 0, 0});
            
            // grad_h_pre累加: 加载x[1, j, d_offset:d_offset+cur_c]
            DataCopyPad(xInput, xGlobal_[gmOffset * n_u * c_u + j * c_u + d_offset], {1, static_cast<uint32_t>(cur_c * sizeof(U)), 0, 0, 0}, {false, 0, 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();
            Cast(xFp32Chunk, xInput, RoundMode::CAST_NONE, cur_c);
            
            // Mul(grad_hin * x) → ReduceSum累加到grad_h_pre[j]
            Mul(mulHinXChunk, gradHinFp32Chunk, xFp32Chunk, cur_c);
            LocalTensor<T> reduceTmp = tmpBuf_.Get<T>();
            ReduceSum(gradHPreLocal[j], mulHinXChunk, reduceTmp, cur_c);
            
            // ★写入workspace: wsXFp32[j*c + d_offset]（x_fp32供Step14使用）
            int64_t wsXFp32Offset_j = wsBase + wsXFp32Offset_ + j * c_ * FP32_BYTE_SIZE + d_offset * FP32_BYTE_SIZE;
            DataCopyPad(workspaceGlobal_[wsXFp32Offset_j / sizeof(T)], xFp32Chunk, {1, static_cast<uint32_t>(cur_c * sizeof(T)), 0, 0, 0});
        }
    }
}

// ========== Step 2: 计算前向z（golden_v2.py 123-130行）==========
// 【责任人: llh】
// 【计算过程】:
//   1. 加载hc_before_norm[B,S,24], inv_rms[B,S,1]从GM
//   2. norm_out_forward = hc_before_norm * inv_rms → [B,S,24]
//   3. 拆分三路:
//      - norm_out_part1[..., :N] = norm_out_forward[..., :N]
//      - norm_out_part2[..., N:2N] = norm_out_forward[..., N:2N]
//      - norm_out_part3[..., 2N:] = norm_out_forward[..., 2N:]
//   4. 计算z:
//      - z_pre = norm_out_part1 * alpha[0] + bias[:N] → [B,S,N]
//      - z_post = norm_out_part2 * alpha[1] + bias[N:2N] → [B,S,N]
//      - z_res = (norm_out_part3 * alpha[2] + bias[2N:]).reshape(B,S,N,N) → [B,S,N,N]
template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<T, U>::Step2_ComputeForwardZ() {
    int64_t gmOffset = blkIdx_ * tile_;
    
    LocalTensor<T> hcBeforeNormLocal = hcBeforeNormQueue_.AllocTensor<T>();
    LocalTensor<T> invRmsLocal = invRmsQueue_.AllocTensor<T>();
    
    DataCopyPad(hcBeforeNormLocal, hcBeforeNormGlobal_[gmOffset * (n_ * n_ + 2 * n_)], 
        {1, static_cast<uint32_t>(tile_ * (n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(invRmsLocal, invRmsGlobal_[gmOffset], {1, static_cast<uint32_t>(tile_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    AscendC::PipeBarrier<PIPE_ALL>();
    
    hcBeforeNormQueue_.EnQue(hcBeforeNormLocal);
    invRmsQueue_.EnQue(invRmsLocal);
    
    LocalTensor<T> alphaLocal = alphaBuf_.Get<T>();
    LocalTensor<T> biasLocal = biasBuf_.Get<T>();
    DataCopyPad(alphaLocal, alphaGlobal_, {1, static_cast<uint32_t>(3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(biasLocal, biasGlobal_, {1, static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    AscendC::PipeBarrier<PIPE_ALL>();
    
    hcBeforeNormLocal = hcBeforeNormQueue_.DeQue<T>();
    invRmsLocal = invRmsQueue_.DeQue<T>();
    
    LocalTensor<T> normOutForwardLocal = normOutForwardBuf_.Get<T>();
    Mul(normOutForwardLocal, hcBeforeNormLocal, invRmsLocal, tile_ * (n_ * n_ + 2 * n_));
    AscendC::PipeBarrier<PIPE_ALL>();
    
    LocalTensor<T> zPreLocal = zPreBuf_.Get<T>();
    LocalTensor<T> zPostLocal = zPostBuf_.Get<T>();
    LocalTensor<T> zResLocal = zResBuf_.Get<T>();
    
    // ★VF: z = norm_out * alpha + bias (三路)
    ComputeForwardZVf<T>(zPreLocal, zPostLocal, zResLocal, normOutForwardLocal, alphaLocal, biasLocal, n_, tile_ * (n_ * n_ + 2 * n_));
    AscendC::PipeBarrier<PIPE_ALL>();
    
    hcBeforeNormQueue_.FreeTensor(hcBeforeNormLocal);
    invRmsQueue_.FreeTensor(invRmsLocal);
}

// ========== Step 3: Sigmoid梯度（golden_v2.py 135-136行）==========
// 【责任人: llh】
// 【计算过程】:
//   1. 加载z_pre[B,S,N], z_post[B,S,N], grad_h_pre[B,S,N], grad_h_post[B,S,N]
//   2. grad_z_pre = SigmoidGrad(z_pre, grad_h_pre, is_pre=True, hc_eps)
//      公式: grad = grad_h_pre * sigma_pre * (1 - sigma_pre)
//      其中: sigma_pre = sigmoid(z_pre) - hc_eps (修正版)
//   3. grad_z_post = 2 * SigmoidGrad(z_post, grad_h_post, is_pre=False)
//      公式: grad = grad_h_post * sigma_post * (1 - sigma_post)
//      **关键**: 系数2修正（golden_v2.py 136行明确要求）
template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<T, U>::Step3_SigmoidGrad() {
    LocalTensor<T> zPreLocal = zPreBuf_.Get<T>();
    LocalTensor<T> zPostLocal = zPostBuf_.Get<T>();
    LocalTensor<T> gradZPreLocal = gradZPreBuf_.Get<T>();
    LocalTensor<T> gradZPostLocal = gradZPostBuf_.Get<T>();
    
    LocalTensor<T> gradHPostLocal = tmpBuf_.Get<T>();
    LocalTensor<T> gradHPreLocal = gradHPreBuf_.Get<T>();
    int64_t gmOffset = blkIdx_ * tile_;
    DataCopyPad(gradHPostLocal, gradHPostGlobal_[gmOffset * n_], {1, static_cast<uint32_t>(tile_ * n_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    AscendC::PipeBarrier<PIPE_ALL>();
    
    LocalTensor<T> sigmaPreLocal = tmpBuf_.Get<T>();
    Sigmoid(sigmaPreLocal, zPreLocal, tile_ * n_);
    Adds(sigmaPreLocal, sigmaPreLocal, static_cast<T>(-HC_EPS), tile_ * n_);
    
    LocalTensor<T> sigmaPostLocal = tmpBuf_.Get<T>();
    Sigmoid(sigmaPostLocal, zPostLocal, tile_ * n_);
    AscendC::PipeBarrier<PIPE_ALL>();
    
    SigmoidGradArithmeticVf<T>(gradZPreLocal, gradZPostLocal, sigmaPreLocal, sigmaPostLocal, gradHPreLocal, gradHPostLocal, tile_ * n_);
    AscendC::PipeBarrier<PIPE_ALL>();
}

// ========== Step 4-5: alpha/bias梯度（pre/post）（golden_v2.py 141-148行）==========
// 【责任人: llh】
// 【计算过程】:
//   1. grad_alpha_pre = (norm_out_forward_part1 * grad_z_pre).sum() → scalar
//   2. grad_alpha_post = (norm_out_forward_part2 * grad_z_post).sum() → scalar
//   3. grad_bias_part1 = grad_z_pre.sum((0,1)) → [N]
//   4. grad_bias_part2 = grad_z_post.sum((0,1)) → [N]
//   5. 累加到workspace（后续ReduceWorkspaceToGM汇总）
template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<T, U>::Step45_AlphaBiasGradPrePost() {
    LocalTensor<T> normOutForwardLocal = normOutForwardBuf_.Get<T>();
    LocalTensor<T> gradZPreLocal = gradZPreBuf_.Get<T>();
    LocalTensor<T> gradZPostLocal = gradZPostBuf_.Get<T>();
    
    LocalTensor<T> gradAlphaLocal = gradAlphaBuf_.Get<T>();
    LocalTensor<T> gradBiasLocal = gradBiasBuf_.Get<T>();
    
    // grad_alpha_pre = (norm_out_forward_part1 * grad_z_pre).sum()  [golden 141行]
    LocalTensor<T> tempAlphaPre = tmpBuf_.Get<T>();
    Mul(tempAlphaPre, normOutForwardLocal[0], gradZPreLocal, tile_ * n_);
    
    LocalTensor<T> reduceTmpAlphaPre = tmpBuf_.Get<T>();
    ReduceSum(gradAlphaLocal[0], tempAlphaPre, reduceTmpAlphaPre, tile_ * n_);
    
    // grad_alpha_post = (norm_out_forward_part2 * grad_z_post).sum()  [golden 142行]
    LocalTensor<T> tempAlphaPost = tmpBuf_.Get<T>();
    Mul(tempAlphaPost, normOutForwardLocal[n_], gradZPostLocal, tile_ * n_);
    
    LocalTensor<T> reduceTmpAlphaPost = tmpBuf_.Get<T>();
    ReduceSum(gradAlphaLocal[1], tempAlphaPost, reduceTmpAlphaPost, tile_ * n_);
    
    // grad_bias_part1 = grad_z_pre.sum((0,1))  [golden 147行]
    LocalTensor<T> reduceTmp1 = tmpBuf_.Get<T>();
    ReduceSum(gradBiasLocal[0], gradZPreLocal, reduceTmp1, tile_ * n_);
    
    // grad_bias_part2 = grad_z_post.sum((0,1))  [golden 148行]
    LocalTensor<T> reduceTmp2 = tmpBuf_.Get<T>();
    ReduceSum(gradBiasLocal[n_], gradZPostLocal, reduceTmp2, tile_ * n_);
    
    // workspace累加
    int64_t wsOffset = blkIdx_ * wsPerCoreSize_;
    GlobalTensor<T> wsGradAlpha = workspaceGlobal_[wsOffset + wsGradAlphaOffset_];
    GlobalTensor<T> wsGradBias = workspaceGlobal_[wsOffset + wsGradBiasOffset_];
    
    // ★workspace是GlobalTensor，需要用DataCopy+Add方式累加（先从GM读到UB，再写回）
    LocalTensor<T> wsGradAlphaLocal = tmpBuf_.Get<T>();
    LocalTensor<T> wsGradBiasLocal = tmpBuf_.Get<T>();
    DataCopyPad(wsGradAlphaLocal, wsGradAlpha, {1, static_cast<uint32_t>(3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(wsGradBiasLocal, wsGradBias, {1, static_cast<uint32_t>((2 * n_) * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    AscendC::PipeBarrier<PIPE_ALL>();
    Add(wsGradAlphaLocal, wsGradAlphaLocal, gradAlphaLocal, 3);
    Add(wsGradBiasLocal, wsGradBiasLocal, gradBiasLocal, 2 * n_);
    DataCopyPad(wsGradAlpha, wsGradAlphaLocal, {1, static_cast<uint32_t>(3 * sizeof(T)), 0, 0, 0});
    DataCopyPad(wsGradBias, wsGradBiasLocal, {1, static_cast<uint32_t>((2 * n_) * sizeof(T)), 0, 0, 0});
}

// ========== Step 6-8: Sinkhorn + Exp梯度（golden_v2.py 155-166行）==========
// 【责任人: hbj】
// 【计算过程】:
//   1. SinkhornGrad反向迭代iters次 → sk_grad[B,S,N,N]
//   2. ExpGrad(z_res, sk_grad) → grad_z_res[B,S,N,N]
//   3. grad_alpha_res累加到workspace[2]
//   4. grad_bias_part3累加到workspace[2N:]
template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<T, U>::Step678_SinkhornExpGrad() {
    int64_t gmOffset = blkIdx_ * tile_;
    
    LocalTensor<T> gradHResLocal = gradHResQueue_.AllocTensor<T>();
    DataCopyPad(gradHResLocal, gradHResGlobal_[gmOffset * n_ * n_], {1, static_cast<uint32_t>(tile_ * n_ * n_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    AscendC::PipeBarrier<PIPE_ALL>();
    gradHResQueue_.EnQue(gradHResLocal);
    
    LocalTensor<T> sumOutLocal = sumOutBuf_.Get<T>();
    LocalTensor<T> normOutLocal = normOutBuf_.Get<T>();
    DataCopyPad(sumOutLocal, sumOutGlobal_, {1, static_cast<uint32_t>(2 * skIterCount_ * n_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(normOutLocal, normOutGlobal_, {1, static_cast<uint32_t>(2 * skIterCount_ * n_ * n_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    AscendC::PipeBarrier<PIPE_ALL>();
    
    gradHResLocal = gradHResQueue_.DeQue<T>();
    
    LocalTensor<T> skGradLocal = tmpBuf_.Get<T>();
    Adds(skGradLocal, gradHResLocal, static_cast<T>(0), tile_ * n_ * n_);
    
    // Sinkhorn反向迭代 [golden 177-193行]
    // ★逐矩阵处理，每个n×n矩阵独立做行列归约
    for (int64_t i = skIterCount_ - 1; i >= 0; i--) {
        LocalTensor<T> colSum = sumOutLocal[2 * i + 1];  // [n] 每列sum
        LocalTensor<T> rowSum = sumOutLocal[2 * i];      // [n] 每行sum
        LocalTensor<T> xRowNormed = normOutLocal[2 * i];  // [tile*n*n] 归一化矩阵
        
        LocalTensor<T> colSumBroadcast = tmpBuf_.Get<T>();   // [n*n] broadcast后的colSum
        LocalTensor<T> rowSumBroadcast = tmpBuf_.Get<T>();   // [n*n] broadcast后的rowSum
        LocalTensor<T> term1 = tmpBuf_.Get<T>();
        LocalTensor<T> term2 = tmpBuf_.Get<T>();
        LocalTensor<T> reduceResult = tmpBuf_.Get<T>();
        LocalTensor<T> reduceBroadcast = tmpBuf_.Get<T>();
        LocalTensor<uint8_t> reduceTmpBuf8 = tmpBuf_.Get<T>().template ReinterpretCast<uint8_t>();
        
        // broadcast colSum[n] → colSumBroadcast[n*n]: 每行复制colSum
        Duplicate(colSumBroadcast, static_cast<T>(0), n_ * n_);
        for (int64_t row = 0; row < n_; row++) {
            Add(colSumBroadcast[row * n_], colSumBroadcast[row * n_], colSum, n_);
        }
        
        // broadcast rowSum[n] → rowSumBroadcast[n*n]: 每行填rowSum[row]
        for (int64_t row = 0; row < n_; row++) {
            Muls(rowSumBroadcast[row * n_], rowSumBroadcast[row * n_], rowSum.GetValue(row), n_);
        }
        
        for (int64_t t = 0; t < tile_; t++) {
            int64_t matrixOff = t * n_ * n_;
            
            // 列归一化梯度 [golden 188行]
            // grad_x_row_normed = x_grad / col_sum - (x_grad * x_row_normed / col_sum²).sum(dim=-2)
            Div(term1, skGradLocal[matrixOff], colSumBroadcast, n_ * n_);
            Mul(term2, skGradLocal[matrixOff], xRowNormed[matrixOff], n_ * n_);
            Div(term2, term2, colSumBroadcast, n_ * n_);
            // 列归约 dim=-2: Pattern::Reduce::RA (R=第一维)
            uint32_t srcShape[] = {static_cast<uint32_t>(n_), static_cast<uint32_t>(n_)};
            ReduceSum<T, Pattern::Reduce::RA>(reduceResult, term2, reduceTmpBuf8, srcShape, false);
            // broadcast reduceResult[1,n] → reduceBroadcast[n*n]: 每行复制reduceResult
            Duplicate(reduceBroadcast, static_cast<T>(0), n_ * n_);
            for (int64_t row = 0; row < n_; row++) {
                Add(reduceBroadcast[row * n_], reduceBroadcast[row * n_], reduceResult, n_);
            }
            Sub(skGradLocal[matrixOff], term1, reduceBroadcast, n_ * n_);
            
            // 行归一化梯度 [golden 192行]
            // x_grad = grad_x_row_normed / row_sum - (grad_x_row_normed * x_row_normed / row_sum).sum(dim=-1)
            Div(term1, skGradLocal[matrixOff], rowSumBroadcast, n_ * n_);
            Mul(term2, skGradLocal[matrixOff], xRowNormed[matrixOff], n_ * n_);
            Div(term2, term2, rowSumBroadcast, n_ * n_);
            // 行归约 dim=-1: Pattern::Reduce::AR (R=第二维)
            ReduceSum<T, Pattern::Reduce::AR>(reduceResult, term2, reduceTmpBuf8, srcShape, false);
            // broadcast reduceResult[n,1] → reduceBroadcast[n*n]: 每行填reduceResult[row]
            for (int64_t row = 0; row < n_; row++) {
                Muls(reduceBroadcast[row * n_], reduceBroadcast[row * n_], reduceResult.GetValue(row), n_);
            }
            Sub(skGradLocal[matrixOff], term1, reduceBroadcast, n_ * n_);
        }
    }
    
    // ExpGrad [golden 156行]
    LocalTensor<T> zResLocal = zResBuf_.Get<T>();
    LocalTensor<T> gradZResLocal = gradZResBuf_.Get<T>();
    LocalTensor<T> expZResLocal = tmpBuf_.Get<T>();
    // ExpGrad: grad_z_res = sk_grad * exp(z_res) [golden 156行]
    Exp(expZResLocal, zResLocal, tile_ * n_ * n_);
    Mul(gradZResLocal, expZResLocal, skGradLocal, tile_ * n_ * n_);
    
    // grad_alpha_res [golden 161行]
    LocalTensor<T> normOutForwardLocal = normOutForwardBuf_.Get<T>();
    LocalTensor<T> tempAlphaRes = tmpBuf_.Get<T>();
    Mul(tempAlphaRes, normOutForwardLocal[2 * n_], gradZResLocal, tile_ * n_ * n_);
    
    LocalTensor<T> gradAlphaLocal = gradAlphaBuf_.Get<T>();
    LocalTensor<T> reduceTmpAlpha = tmpBuf_.Get<T>();
    ReduceSum(gradAlphaLocal[2], tempAlphaRes, reduceTmpAlpha, tile_ * n_ * n_);
    
    // grad_bias_part3 [golden 166行]
    LocalTensor<T> gradBiasLocal = gradBiasBuf_.Get<T>();
    LocalTensor<T> reduceTmpBias = tmpBuf_.Get<T>();
    ReduceSum(gradBiasLocal[2 * n_], gradZResLocal, reduceTmpBias, tile_ * n_ * n_);
    
    // workspace累加
    int64_t wsOffset = blkIdx_ * wsPerCoreSize_;
    GlobalTensor<T> wsGradAlpha = workspaceGlobal_[wsOffset + wsGradAlphaOffset_];
    GlobalTensor<T> wsGradBias = workspaceGlobal_[wsOffset + wsGradBiasOffset_];
    
    LocalTensor<T> wsGradAlphaLocal = tmpBuf_.Get<T>();
    LocalTensor<T> wsGradBiasLocal = tmpBuf_.Get<T>();
    DataCopyPad(wsGradAlphaLocal, wsGradAlpha, {1, static_cast<uint32_t>(3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(wsGradBiasLocal, wsGradBias, {1, static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    AscendC::PipeBarrier<PIPE_ALL>();
    Add(wsGradAlphaLocal, wsGradAlphaLocal, gradAlphaLocal, 3);
    Add(wsGradBiasLocal, wsGradBiasLocal, gradBiasLocal, n_ * n_ + 2 * n_);
    DataCopyPad(wsGradAlpha, wsGradAlphaLocal, {1, static_cast<uint32_t>(3 * sizeof(T)), 0, 0, 0});
    DataCopyPad(wsGradBias, wsGradBiasLocal, {1, static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0});
    
    gradHResQueue_.FreeTensor(gradHResLocal);
}

// ========== Step 9-11: Linear梯度拼接（golden_v2.py 175-186行）==========
// 【责任人: lhp】
// 【计算过程】:
//   1. grad_norm_out_pre = grad_z_pre * alpha[0]
//   2. grad_norm_out_post = grad_z_post * alpha[1]
//   3. grad_norm_out_res = grad_z_res * alpha[2]
//   4. 三路拼接为grad_norm_out[B,S,24]
template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<T, U>::Step911_LinearGradConcat() {
    LocalTensor<T> gradZPreLocal = gradZPreBuf_.Get<T>();
    LocalTensor<T> gradZPostLocal = gradZPostBuf_.Get<T>();
    LocalTensor<T> gradZResLocal = gradZResBuf_.Get<T>();
    LocalTensor<T> alphaLocal = alphaBuf_.Get<T>();
    
    LocalTensor<T> gradNormOutLocal = gradNormOutBuf_.Get<T>();
    
    // ★VF: grad_norm_out = grad_z * alpha (三路)
    ComputeLinearGradVf<T>(gradNormOutLocal, gradZPreLocal, gradZPostLocal, gradZResLocal, alphaLocal, n_, tile_ * (n_ * n_ + 2 * n_));
    AscendC::PipeBarrier<PIPE_ALL>();
}

// ========== Step 12: RMSNorm梯度（golden_v2.py 191-192行）==========
// 【责任人: lhp】
// 【计算过程】: grad_hc = grad_norm_out*inv_rms, grad_inv_rms = (hc*grad_norm_out).sum(-1)
template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<T, U>::Step12_RMSNormGrad() {
    LocalTensor<T> gradNormOutLocal = gradNormOutBuf_.Get<T>();
    LocalTensor<T> invRmsLocal = tmpBuf_.Get<T>();
    
    int64_t gmOffset = blkIdx_ * tile_;
    DataCopyPad(invRmsLocal, invRmsGlobal_[gmOffset], {1, static_cast<uint32_t>(tile_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    AscendC::PipeBarrier<PIPE_ALL>();
    
    LocalTensor<T> gradHcLocal = gradHcQueue_.AllocTensor<T>();
    
    // ★VF: grad_hc = grad_norm_out * inv_rms
    ComputeRMSNormGradVf<T>(gradHcLocal, gradNormOutLocal, invRmsLocal, tile_ * (n_ * n_ + 2 * n_));
    AscendC::PipeBarrier<PIPE_ALL>();
    
    gradHcQueue_.EnQue(gradHcLocal);
}

// ========== Step 14: RMSNorm倒数梯度（golden_v2.py 192,207行）==========
// 【责任人: lhp】
// ★D切分循环：从workspace读取x_fp32，逐chunk计算grad_x_from_rms，写回workspace
template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<T, U>::Step14_RMSNormInvGrad() {
    int64_t gmOffset = blkIdx_ * tile_;
    int64_t wsBase = blkIdx_ * wsPerCoreSize_;
    
    // 加载inv_rms和hc_before_norm（N维度，UB安全）
    LocalTensor<T> gradNormOutLocal = gradNormOutBuf_.Get<T>();
    LocalTensor<T> hcBeforeNormLocal = tmpBuf_.Get<T>();
    LocalTensor<T> invRmsLocal = tmpBuf_.Get<T>();
    
    DataCopyPad(hcBeforeNormLocal, hcBeforeNormGlobal_[gmOffset * (n_ * n_ + 2 * n_)], 
        {1, static_cast<uint32_t>(tile_ * (n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(invRmsLocal, invRmsGlobal_[gmOffset], {1, static_cast<uint32_t>(tile_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    AscendC::PipeBarrier<PIPE_ALL>();
    
    // grad_inv_rms = (hc_before_norm * grad_norm_out).sum(-1) [golden 192行]
    LocalTensor<T> gradInvRmsLocal = tmpBuf_.Get<T>();
    LocalTensor<T> tempGradInvRms = tmpBuf_.Get<T>();
    LocalTensor<T> reduceTmpInvRms = tmpBuf_.Get<T>();
    Mul(tempGradInvRms, hcBeforeNormLocal, gradNormOutLocal, tile_ * (n_ * n_ + 2 * n_));
    ReduceSum(gradInvRmsLocal, tempGradInvRms, reduceTmpInvRms, tile_ * (n_ * n_ + 2 * n_));
    
    // ★D切分循环计算grad_x_from_rms，逐chunk写入workspace
    // 公式：grad_x_from_rms = - (inv_rms^3) * x * grad_inv_rms / NC
    uint32_t c0_u = static_cast<uint32_t>(c0_);
    uint32_t cTail_u = static_cast<uint32_t>(cTail_);
    uint32_t c_u = static_cast<uint32_t>(c_);
    uint32_t D_chunk_cnt = static_cast<uint32_t>(D_chunk_count_);
    float ncScalar = static_cast<float>(n_ * c_);
    
    LocalTensor<T> xFp32Chunk = tmpBuf_.Get<T>();
    LocalTensor<T> gradXFromRmsChunk = tmpBuf_.Get<T>();
    
    // inv_rms^3 预计算
    LocalTensor<T> invRmsCubed = tmpBuf_.Get<T>();
    Mul(invRmsCubed, invRmsLocal, invRmsLocal, tile_);
    Mul(invRmsCubed, invRmsCubed, invRmsLocal, tile_);
    
    // Use Mul+Muls to compute: -inv_rms^3 * x * grad_inv_rms / NC
    
    // ★逐D_chunk: 从workspace读取x_fp32，计算grad_x_from_rms，写回workspace
    for (uint16_t d_idx = 0; d_idx < D_chunk_cnt; d_idx++) {
        uint32_t cur_c = (d_idx == D_chunk_cnt - 1) ? cTail_u : c0_u;
        uint32_t d_offset = d_idx * c0_u;
        
        int64_t wsXFp32ReadOffset = wsBase + wsXFp32Offset_ + d_offset * FP32_BYTE_SIZE;
        DataCopyPad(xFp32Chunk, workspaceGlobal_[wsXFp32ReadOffset / sizeof(T)], {1, static_cast<uint32_t>(cur_c * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();
        
        // -inv_rms^3 * x * grad_inv_rms / NC  (broadcast invRmsCubed & gradInvRms)
        // 逐N做Muls: gradXFromRms[n][d_chunk] = Muls(xFp32[n][d_chunk], -ncScalar * invRmsCubed[n] * gradInvRms[n])
        uint32_t n_u_local = static_cast<uint32_t>(n_);
        for (uint16_t j = 0; j < n_u_local; j++) {
            T coeff = static_cast<T>(invRmsCubed.GetValue(j) * gradInvRmsLocal.GetValue(j) * static_cast<float>(-1.0) / ncScalar);
            Muls(gradXFromRmsChunk, xFp32Chunk, coeff, cur_c);
            
            int64_t wsGradXFromRmsWriteOffset_j = wsBase + wsGradXFromRmsOffset_ + j * c_ * FP32_BYTE_SIZE + d_offset * FP32_BYTE_SIZE;
            DataCopyPad(workspaceGlobal_[wsGradXFromRmsWriteOffset_j / sizeof(T)], gradXFromRmsChunk, {1, static_cast<uint32_t>(cur_c * sizeof(T)), 0, 0, 0});
        }
    }
}

// ========== Step 15: Matmul梯度 + grad_x汇总（golden_v2.py 221-230行）==========
// 【责任人: lhp】
// 【计算过程】: grad_x = grad_hc@phi, grad_phi = grad_hc.T@x_flat, grad_x第三路AtomicAdd累加
template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<T, U>::Step15_MatmulGradX() {
    int64_t gmOffset = blkIdx_ * tile_;
    
    LocalTensor<T> gradHcLocal = gradHcQueue_.DeQue<T>();
    
    // 写grad_hc到GM供AIC matmul
    DataCopyPad(gradHcGlobal_[gmOffset * (n_ * n_ + 2 * n_)], gradHcLocal, 
        {1, static_cast<uint32_t>(tile_ * (n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0});
    
    gradHcQueue_.FreeTensor(gradHcLocal);
}

template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<T, U>::AIC_MatmulGradX() {
    int64_t mmOffset = blkIdx_ * tile_;
    
    mm1_.SetTensorA(gradHcGlobal_[mmOffset * (n_ * n_ + 2 * n_)]);
    mm1_.SetTensorB(phiGlobal_);
    
    mm1_.template IterateAll<false>(gradXGlobal_[mmOffset * n_ * c_], 1);  
}

template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<T, U>::AIC_MatmulGradPhi() {
    int64_t mmOffset = blkIdx_ * tile_;

    // mm2: grad_phi = grad_hc.T @ x_flat  [golden 224行]
    mm2_.SetTensorA(gradHcGlobal_[mmOffset * (n_ * n_ + 2 * n_)]);
    int64_t wsXFp32Offset = blkIdx_ * wsPerCoreSize_ + wsXFp32Offset_;
    mm2_.SetTensorB(workspaceGlobal_[wsXFp32Offset]);

    int64_t wsOffset = blkIdx_ * wsPerCoreSize_ + wsGradPhiOffset_;
    mm2_.template IterateAll<false>(workspaceGlobal_[wsOffset], 1);
}

// ========== ReduceWorkspaceToGM（golden_v2.py 步骤16，242-243行）==========
// 【功能】多核workspace归约，组装grad_alpha/bias/phi/x
// ★core0做最终累加（syncall机制保证所有核完成后才累加）
template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<T, U>::ReduceWorkspaceToGM() {
    if (blkIdx_ == 0) {
        // grad_alpha累加
        LocalTensor<T> totalGradAlpha = gradAlphaBuf_.Get<T>();
        DataCopyPad(totalGradAlpha, workspaceGlobal_[wsGradAlphaOffset_], {1, static_cast<uint32_t>(3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();
        
        for (int64_t c = 1; c < aivNum_; c++) {
            int64_t wsOffset = c * wsPerCoreSize_ + wsGradAlphaOffset_;
            GlobalTensor<T> wsGradAlphaCore = workspaceGlobal_[wsOffset];
            LocalTensor<T> wsGradAlphaLocal = tmpBuf_.Get<T>();
            DataCopyPad(wsGradAlphaLocal, wsGradAlphaCore, {1, static_cast<uint32_t>(3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();
            Add(totalGradAlpha, totalGradAlpha, wsGradAlphaLocal, 3);
        }
        DataCopyPad(gradAlphaGlobal_, totalGradAlpha, {1, static_cast<uint32_t>(3 * sizeof(T)), 0, 0, 0});
        
        // grad_bias累加
        LocalTensor<T> totalGradBias = gradBiasBuf_.Get<T>();
        DataCopyPad(totalGradBias, workspaceGlobal_[wsGradBiasOffset_], {1, static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();
        
        for (int64_t c = 1; c < aivNum_; c++) {
            int64_t wsOffset = c * wsPerCoreSize_ + wsGradBiasOffset_;
            GlobalTensor<T> wsGradBiasCore = workspaceGlobal_[wsOffset];
            LocalTensor<T> wsGradBiasLocal = tmpBuf_.Get<T>();
            DataCopyPad(wsGradBiasLocal, wsGradBiasCore, {1, static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();
            Add(totalGradBias, totalGradBias, wsGradBiasLocal, n_ * n_ + 2 * n_);
        }
        DataCopyPad(gradBiasGlobal_, totalGradBias, {1, static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0});
        
        // grad_phi累加
        LocalTensor<T> totalGradPhi = tmpBuf_.Get<T>();
        DataCopyPad(totalGradPhi, workspaceGlobal_[wsGradPhiOffset_], {1, static_cast<uint32_t>((n_ * n_ + 2 * n_) * n_ * c_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();
        
        for (int64_t c = 1; c < aivNum_; c++) {
            int64_t wsOffset = c * wsPerCoreSize_ + wsGradPhiOffset_;
            GlobalTensor<T> wsGradPhiCore = workspaceGlobal_[wsOffset];
            LocalTensor<T> wsGradPhiLocal = tmpBuf_.Get<T>();
            DataCopyPad(wsGradPhiLocal, wsGradPhiCore, {1, static_cast<uint32_t>((n_ * n_ + 2 * n_) * n_ * c_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();
            Add(totalGradPhi, totalGradPhi, wsGradPhiLocal, (n_ * n_ + 2 * n_) * n_ * c_);
        }
        DataCopyPad(gradPhiGlobal_, totalGradPhi, {1, static_cast<uint32_t>((n_ * n_ + 2 * n_) * n_ * c_ * sizeof(T)), 0, 0, 0});
        
        // ★grad_x累加：三路汇总（from_hin + from_rms + from_matmul）
        // 逐D_chunk累加workspace中各核的grad_x
        uint32_t c0_u = static_cast<uint32_t>(c0_);
        uint32_t cTail_u = static_cast<uint32_t>(cTail_);
        uint32_t c_u = static_cast<uint32_t>(c_);
        uint32_t D_chunk_cnt = static_cast<uint32_t>(D_chunk_count_);
        int64_t n_c = n_ * c_;
        
        LocalTensor<U> gradXOutput = tmpBuf_.Get<U>();
        LocalTensor<T> gradXFromHinChunk = tmpBuf_.Get<T>();
        LocalTensor<T> gradXFromRmsChunk = tmpBuf_.Get<T>();
        LocalTensor<T> gradXFromMatmulChunk = tmpBuf_.Get<T>();
        LocalTensor<T> gradXTempChunk = tmpBuf_.Get<T>();
        LocalTensor<T> gradXChunk = tmpBuf_.Get<T>();
        
        for (int64_t taskIdx = 0; taskIdx < totalTasks_; taskIdx++) {
            int64_t gmOffset = taskIdx;
            
            for (uint16_t d_idx = 0; d_idx < D_chunk_cnt; d_idx++) {
                uint32_t cur_c = (d_idx == D_chunk_cnt - 1) ? cTail_u : c0_u;
                uint32_t d_offset = d_idx * c0_u;
                
                // 初始化累加结果（从core0的workspace读取）
                int64_t wsGradXOffset_core0 = wsGradXOffset_ + d_offset * FP32_BYTE_SIZE;
                DataCopyPad(gradXFromHinChunk, workspaceGlobal_[wsGradXOffset_core0 / sizeof(T)], {1, static_cast<uint32_t>(cur_c * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
                AscendC::PipeBarrier<PIPE_ALL>();
                
                int64_t wsGradXFromRmsOffset_core0 = wsGradXFromRmsOffset_ + d_offset * FP32_BYTE_SIZE;
                DataCopyPad(gradXFromRmsChunk, workspaceGlobal_[wsGradXFromRmsOffset_core0 / sizeof(T)], {1, static_cast<uint32_t>(cur_c * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
                AscendC::PipeBarrier<PIPE_ALL>();
                
                LocalTensor<U> gradXFromMatmulInput = tmpBuf_.Get<U>();
                DataCopyPad(gradXFromMatmulInput, gradXGlobal_[gmOffset * n_c + d_offset], {1, static_cast<uint32_t>(cur_c * sizeof(U)), 0, 0, 0}, {false, 0, 0, 0});
                AscendC::PipeBarrier<PIPE_ALL>();
                Cast(gradXFromMatmulChunk, gradXFromMatmulInput, RoundMode::CAST_NONE, cur_c);
                
                for (int64_t core = 1; core < aivNum_; core++) {
                    int64_t wsBaseCore = core * wsPerCoreSize_;
                    int64_t wsGradXOff_core = wsBaseCore + wsGradXOffset_ + d_offset * FP32_BYTE_SIZE;
                    GlobalTensor<T> wsGradXFromHinCore = workspaceGlobal_[wsGradXOff_core / sizeof(T)];
                    LocalTensor<T> wsGradXFromHinLocal = tmpBuf_.Get<T>();
                    DataCopyPad(wsGradXFromHinLocal, wsGradXFromHinCore, {1, static_cast<uint32_t>(cur_c * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
                    AscendC::PipeBarrier<PIPE_ALL>();
                    Add(gradXFromHinChunk, gradXFromHinChunk, wsGradXFromHinLocal, cur_c);
                    
                    int64_t wsGradXFromRmsOff_core = wsBaseCore + wsGradXFromRmsOffset_ + d_offset * FP32_BYTE_SIZE;
                    GlobalTensor<T> wsGradXFromRmsCore = workspaceGlobal_[wsGradXFromRmsOff_core / sizeof(T)];
                    LocalTensor<T> wsGradXFromRmsLocal = tmpBuf_.Get<T>();
                    DataCopyPad(wsGradXFromRmsLocal, wsGradXFromRmsCore, {1, static_cast<uint32_t>(cur_c * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
                    AscendC::PipeBarrier<PIPE_ALL>();
                    Add(gradXFromRmsChunk, gradXFromRmsChunk, wsGradXFromRmsLocal, cur_c);
                }
                
                Add(gradXTempChunk, gradXFromHinChunk, gradXFromRmsChunk, cur_c);
                Add(gradXChunk, gradXTempChunk, gradXFromMatmulChunk, cur_c);
                
                Cast(gradXOutput, gradXChunk, RoundMode::CAST_RINT, cur_c);
                DataCopyPad(gradXGlobal_[gmOffset * n_c + d_offset], gradXOutput, {1, static_cast<uint32_t>(cur_c * sizeof(U)), 0, 0, 0});
            }
        }
    }
}

#endif // MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_H