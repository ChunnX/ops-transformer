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
    
    // TBuf（单buffer — 纯UB内计算，不涉及流水同步）
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> alphaBuf_;
    TBuf<TPosition::VECCALC> biasBuf_;
    
    // Step1 计算块模型：独立buffer，不与tmpBuf_共享，支持并行搬入+独立VF计算
    TBuf<TPosition::VECCALC> gradHinInputBuf_;   // BF16/FP16 grad_hin输入chunk（独立buffer，不与xInput共享）
    TBuf<TPosition::VECCALC> xInputBuf_;         // BF16/FP16 x输入chunk（独立buffer）
    TBuf<TPosition::VECCALC> xFp32Buf_;          // FP32 x cast结果（独立buffer）
    // Step3: 独立buffer（SigmoidGrad需要4个输入同时共存）
    TBuf<TPosition::VECCALC> sigmaPreBuf_;         // Sigmoid(z_pre)结果
    TBuf<TPosition::VECCALC> sigmaPostBuf_;        // Sigmoid(z_post)结果
    TBuf<TPosition::VECCALC> gradHPostBuf_;        // grad_h_post MTE2输入
    
    // Queue（流水线）— GM→UB输入用TQue替代PipeBarrier
    TQue<QuePosition::VECIN, DOUBLE_BUF> hPreQueue_;
    TBuf<TPosition::VECCALC> gradAlphaBuf_;
    TBuf<TPosition::VECCALC> gradBiasBuf_;
    
    TQue<QuePosition::VECIN, DOUBLE_BUF> sumOutQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUF> normOutQueue_;
    
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
    
    // ===== 调试打印：第一个核打印全局信息 =====
    if (blkIdx_ == 0) {
        AscendC::printf("[Init] ========== Global Info (Core 0) ==========\n");
        AscendC::printf("[Init] BatchSize=%d, SeqLength=%d\n", tilingData->batchSize, tilingData->seqLength);
        AscendC::printf("[Init] c=D=%d, n=%d, skIterCount=%d\n", tilingData->c, tilingData->n, tilingData->skIterCount);
        AscendC::printf("[Init] tileSize=%d, aivNum=%d\n", tilingData->tileSize, tilingData->aivNum);
        AscendC::printf("[Init] c0=%d, cTail=%d, D_chunk_count=%d\n", tilingData->c0, tilingData->cTail, (tilingData->c + tilingData->c0 - 1) / tilingData->c0);
        AscendC::printf("[Init] totalTasks=%d (B*S), totalTasksAligned=%d\n", 
            tilingData->batchSize * tilingData->seqLength, 
            AlignUp(tilingData->batchSize * tilingData->seqLength, tilingData->aivNum * tilingData->tileSize));
        AscendC::printf("[Init] GetBlockIdx()=%d, current block index\n", blkIdx_);
    }
    
    InitTiling(tilingData);
    
    // ===== 调试打印：第一个核打印workspace配置 =====
    if (blkIdx_ == 0) {
        AscendC::printf("[Init] ========== Workspace Config ==========\n");
        AscendC::printf("[Init] wsGradAlphaOffset=%d (3*FP32=%d bytes)\n", wsGradAlphaOffset_, 3 * FP32_BYTE_SIZE);
        AscendC::printf("[Init] wsGradBiasOffset=%d, wsGradPhiOffset=%d\n", wsGradBiasOffset_, wsGradPhiOffset_);
        AscendC::printf("[Init] wsGradXOffset=%d, wsXFp32Offset=%d\n", wsGradXOffset_, wsXFp32Offset_);
        AscendC::printf("[Init] wsGradXFromRmsOffset=%d, wsPerCoreSize=%d\n", wsGradXFromRmsOffset_, wsPerCoreSize_);
        AscendC::printf("[Init] aivNum=%d, aicNum=%d\n", aivNum_, aicNum_);
    }
    
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
    
// Workspace偏移计算（256字节对齐）
    constexpr int64_t ALIGN_256 = 256;
    wsGradAlphaOffset_ = 0;
    wsGradBiasOffset_ = AlignUp(3 * FP32_BYTE_SIZE, ALIGN_256);
    wsGradPhiOffset_ = wsGradBiasOffset_ + AlignUp((n_ * n_ + 2 * n_) * FP32_BYTE_SIZE, ALIGN_256);
    wsGradXOffset_ = wsGradPhiOffset_ + AlignUp((n_ * n_ + 2 * n_) * (n_ * c_) * FP32_BYTE_SIZE, ALIGN_256);
    wsXFp32Offset_ = wsGradXOffset_ + AlignUp(tile_ * n_ * c_ * FP32_BYTE_SIZE, ALIGN_256);
    wsGradXFromRmsOffset_ = wsXFp32Offset_ + AlignUp(tile_ * n_ * c_ * FP32_BYTE_SIZE, ALIGN_256);
    wsPerCoreSize_ = wsGradXFromRmsOffset_ + AlignUp(tile_ * n_ * c_ * FP32_BYTE_SIZE, ALIGN_256);
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
    // 256字节对齐常量
    constexpr int64_t ALIGN_256 = 256;
    
    // 计算对齐后的buffer大小
    int64_t tileSize = AlignUp(tile_ * (n_ * n_ + 2 * n_) * FP32_BYTE_SIZE, ALIGN_256);
    int64_t tileSizeN = AlignUp(tile_ * n_ * FP32_BYTE_SIZE, ALIGN_256);
    int64_t tileSizeNN = AlignUp(tile_ * n_ * n_ * FP32_BYTE_SIZE, ALIGN_256);
    int64_t tileSizeNC0 = AlignUp(tile_ * n_ * c0_ * FP32_BYTE_SIZE, ALIGN_256);
    int64_t invRmsSize = AlignUp(tile_ * FP32_BYTE_SIZE, ALIGN_256);
    int64_t alphaSize = AlignUp(3 * FP32_BYTE_SIZE, ALIGN_256);
    int64_t biasSize = AlignUp((n_ * n_ + 2 * n_) * FP32_BYTE_SIZE, ALIGN_256);
    int64_t c0USize = AlignUp(c0_ * sizeof(U), ALIGN_256);
    int64_t c0Fp32Size = AlignUp(c0_ * FP32_BYTE_SIZE, ALIGN_256);
    int64_t sumOutSize = AlignUp(2 * skIterCount_ * n_ * FP32_BYTE_SIZE, ALIGN_256);
    int64_t normOutSize = AlignUp(2 * skIterCount_ * n_ * n_ * FP32_BYTE_SIZE, ALIGN_256);
    
    // Queue初始化（256字节对齐）
    pipe_->InitBuffer(hcBeforeNormQueue_, DOUBLE_BUF, tileSize);
    pipe_->InitBuffer(invRmsQueue_, DOUBLE_BUF, invRmsSize);
    pipe_->InitBuffer(gradHResQueue_, DOUBLE_BUF, tileSizeNN);
    pipe_->InitBuffer(gradHcQueue_, DOUBLE_BUF, tileSize);
    pipe_->InitBuffer(hPreQueue_, DOUBLE_BUF, tileSizeN);
    pipe_->InitBuffer(sumOutQueue_, DOUBLE_BUF, sumOutSize);
    pipe_->InitBuffer(normOutQueue_, DOUBLE_BUF, normOutSize);
    
    // TBuf初始化（256字节对齐）
    pipe_->InitBuffer(tmpBuf_, tileSizeNC0);
    pipe_->InitBuffer(alphaBuf_, alphaSize);
    pipe_->InitBuffer(biasBuf_, biasSize);
    
    // Step1: 独立输入/中间buffer（256字节对齐）
    pipe_->InitBuffer(gradHinInputBuf_, c0USize);
    pipe_->InitBuffer(xInputBuf_, c0USize);
    pipe_->InitBuffer(xFp32Buf_, c0Fp32Size);
    
    // Step3: 独立Sigmoid中间buffer（256字节对齐）
    pipe_->InitBuffer(sigmaPreBuf_, tileSizeN);
    pipe_->InitBuffer(sigmaPostBuf_, tileSizeN);
    pipe_->InitBuffer(gradHPostBuf_, tileSizeN);
    
    // Step45: alpha/bias梯度buffer（256字节对齐）
    pipe_->InitBuffer(gradAlphaBuf_, alphaSize);
    pipe_->InitBuffer(gradBiasBuf_, biasSize);
    
    // Step2: zPre/zPost/zRes buffer（256字节对齐）
    pipe_->InitBuffer(zPreBuf_, tileSizeN);
    pipe_->InitBuffer(zPostBuf_, tileSizeN);
    pipe_->InitBuffer(zResBuf_, tileSizeNN);
    
    // Step3: gradZ buffer（256字节对齐）
    pipe_->InitBuffer(gradZPreBuf_, tileSizeN);
    pipe_->InitBuffer(gradZPostBuf_, tileSizeN);
    pipe_->InitBuffer(gradZResBuf_, tileSizeNN);
    
    // Step9-12: gradNormOut buffer（256字节对齐）
    pipe_->InitBuffer(gradNormOutBuf_, tileSize);
    
    // Step1: gradHPre buffer（256字节对齐）
    pipe_->InitBuffer(gradHPreBuf_, tileSizeN);
    
    // Step2: normOutForward buffer（256字节对齐）
    pipe_->InitBuffer(normOutForwardBuf_, tileSize);
}

// ========== Process实现==========
template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<T, U>::Process() {
    
    if ASCEND_IS_AIV {
        for (int64_t taskIdx = blkIdx_; taskIdx < totalTasksAligned_; taskIdx += aivNum_) {
            // 【责任人: llh】步骤1-5
            // Step1_InputAggregation();
            Step2_ComputeForwardZ();
            // Step3_SigmoidGrad();
            // Step45_AlphaBiasGradPrePost();
            // CrossCoreSetFlag<2, PIPE_FIX>(FLAG_ID_LLH_DONE);
            
            // // 【责任人: hbj】步骤6-8
            // Step678_SinkhornExpGrad();
            // CrossCoreSetFlag<2, PIPE_FIX>(FLAG_ID_HBJ_DONE);
            
            // // 【责任人: lhp】步骤9-15
            // Step911_LinearGradConcat();
            // Step12_RMSNormGrad();
            // Step14_RMSNormInvGrad();
            // Step15_MatmulGradX();
            // CrossCoreSetFlag<2, PIPE_FIX>(FLAG_ID_LHP_DONE);
        }
    //     CrossCoreWaitFlag(FLAG_ID_ALL_DONE);
    //     ReduceWorkspaceToGM();
    // }
    
    // if ASCEND_IS_AIC {
    //     for (int64_t taskIdx = blkIdx_; taskIdx < totalTasksAligned_; taskIdx += aicNum_) {
    //         CrossCoreWaitFlag(FLAG_ID_LHP_DONE);
            
    //         AIC_MatmulGradX();
    //         AIC_MatmulGradPhi();
    //         CrossCoreSetFlag<2, PIPE_FIX>(FLAG_ID_ALL_DONE);
    //     }
    }
}

// ========== Step 1: 输入聚合（golden_v2.py 117-118行）==========
// 【责任人: llh】
// 【验证范围】当前验证步骤1-5，后续步骤见标记
// 【设计原则】:
//   - x的Cast前置到j循环之前（用户要求），一次性Cast并搬workspace
//   - h_pre一次搬入，随路broadcast使用（Muls替代Dup+Mul）
//   - grad_hin逐D_chunk搬入，VF内Cast
//   - grad_x写入workspace，x_fp32写入workspace供Step14使用
//   - grad_h_pre逐D_chunk累加ReduceSum
// 【调试信息】AscendC::printf在__aicore__内，不在__VEC_SCOPE__内打印前8个值
template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<T, U>::Step1_InputAggregation() {
    int64_t gmOffset = blkIdx_ * tile_;
    int64_t wsBase = blkIdx_ * wsPerCoreSize_;
    
    // ===== 准备数据: h_pre一次搬入（整个tile生命周期驻留）=====
    LocalTensor<T> hPreLocal = hPreQueue_.AllocTensor<T>();
    AscendC::DataCopyPad(hPreLocal, hPreGlobal_[gmOffset * n_], 
        {1, static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    hPreQueue_.EnQue(hPreLocal);
    hPreLocal = hPreQueue_.DeQue<T>();
    
    uint32_t c0_u = static_cast<uint32_t>(c0_);
    uint32_t cTail_u = static_cast<uint32_t>(cTail_);
    uint32_t c_u = static_cast<uint32_t>(c_);
    uint32_t n_u = static_cast<uint32_t>(n_);
    uint32_t D_chunk_cnt = static_cast<uint32_t>(D_chunk_count_);
    
    LocalTensor<T> gradHPreLocal = gradHPreBuf_.Get<T>();
    AscendC::Duplicate(gradHPreLocal, static_cast<T>(0), n_);
    
    // ===== 计算: 两层循环（d_idx × j），借鉴pr_6646_new优化 =====
    // 优化1: 删除Phase 1前置x Cast → 改为VF内部顺手用完
    // 优化2: 删除Phase 2读回xFp32 → x在VF内部Load→Cast→立即用→写workspace
    // 优化3: gradHin VL级加载（借鉴pr_6646_new的loopCnt切分）
    
    LocalTensor<U> gradHinInput = gradHinInputBuf_.Get<U>();
    LocalTensor<U> xInputChunk = xInputBuf_.Get<U>();
    LocalTensor<T> gradXChunk = tmpBuf_.Get<T>();
    
    for (uint16_t d_idx = 0; d_idx < D_chunk_cnt; d_idx++) {
        uint32_t cur_c = (d_idx == D_chunk_cnt - 1) ? cTail_u : c0_u;
        uint32_t d_offset = d_idx * c0_u;
        
        // j循环：逐个计算gradHPre[j]（借鉴pr_6646_new的n循环）
        for (uint16_t j = 0; j < n_u; j++) {
            T hPreScalar = hPreLocal.GetValue(j);
            
            AscendC::DataCopyPad(gradHinInput, gradHinGlobal_[gmOffset * c_u + d_offset],
                {1, static_cast<uint32_t>(cur_c * sizeof(U)), 0, 0, 0}, {false, 0, 0, 0});
            AscendC::DataCopyPad(xInputChunk, xGlobal_[gmOffset * n_u * c_u + j * c_u + d_offset],
                {1, static_cast<uint32_t>(cur_c * sizeof(U)), 0, 0, 0}, {false, 0, 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();
            
            LocalTensor<T> xFp32Chunk = xFp32Buf_.Get<T>();
            
            // ===== 调试打印：InputAggregationVf参数（第一个核第一次调用）=====
                AscendC::printf("[Step1-Debug] InputAggregationVf params:\n");
                AscendC::printf("  j=%d, cur_c=%d, c0_u=%d, cTail_u=%d\n", j, cur_c, c0_u, cTail_u);
                AscendC::printf("  hPreScalar=%f, n_u=%d, d_offset=%d\n", hPreScalar, n_u, d_offset);
                AscendC::printf("  gradHinInput addr=%p, xInputChunk addr=%p\n", 
                    gradHinInput.GetPhyAddr(), xInputChunk.GetPhyAddr());
                AscendC::printf("  gradXChunk addr=%p, xFp32Chunk addr=%p\n",
                    gradXChunk.GetPhyAddr(), xFp32Chunk.GetPhyAddr());
            
            InputAggregationVf_Optimized<T, U>(
                gradXChunk, xFp32Chunk, gradHPreLocal, j,
                gradHinInput, xInputChunk,
                hPreScalar, cur_c);
            
            AscendC::PipeBarrier<PIPE_ALL>();
            
            int64_t wsXFp32Offset_j = wsBase + wsXFp32Offset_ + j * c_ * FP32_BYTE_SIZE + d_offset * FP32_BYTE_SIZE;
            AscendC::DataCopyPad(workspaceGlobal_[wsXFp32Offset_j / sizeof(T)], xFp32Chunk,
                {1, static_cast<uint16_t>(cur_c * sizeof(T)), 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();
            
            int64_t wsGradXOffset_j = wsBase + wsGradXOffset_ + j * c_ * FP32_BYTE_SIZE + d_offset * FP32_BYTE_SIZE;
            AscendC::DataCopyPad(workspaceGlobal_[wsGradXOffset_j / sizeof(T)], gradXChunk,
                {1, static_cast<uint16_t>(cur_c * sizeof(T)), 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    
    // ===== 搬出: hPre释放 =====
    hPreQueue_.FreeTensor(hPreLocal);
    
    // ===== 调试打印（对照golden_v2.py行295-306）=====
    // AscendC::printf在__aicore__内，不在__VEC_SCOPE__内
    // 打印grad_h_pre前8个值（golden: grad_h_pre = (grad_hin*x).sum(-1)）
    AscendC::printf("[Step1] grad_h_pre (first 8 values):\n");
    for (int i = 0; i < 8 && i < n_; i++) {
        AscendC::printf("  grad_h_pre[%d] = %f\n", i, gradHPreLocal.GetValue(i));
    }
    
    // 打印h_pre前8个值（golden: h_pre input）
    AscendC::printf("[Step1] h_pre (first 8 values):\n");
    for (int i = 0; i < 8 && i < n_; i++) {
        AscendC::printf("  h_pre[%d] = %f\n", i, hPreLocal.GetValue(i));
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
    
    AscendC::DataCopyPad(hcBeforeNormLocal, hcBeforeNormGlobal_[gmOffset * (n_ * n_ + 2 * n_)], 
        {1, static_cast<uint32_t>(tile_ * (n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    AscendC::DataCopyPad(invRmsLocal, invRmsGlobal_[gmOffset], 
        {1, static_cast<uint32_t>(tile_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    
    LocalTensor<T> alphaLocal = alphaBuf_.Get<T>();
    LocalTensor<T> biasLocal = biasBuf_.Get<T>();
    AscendC::DataCopyPad(alphaLocal, alphaGlobal_, 
        {1, static_cast<uint32_t>(3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    AscendC::DataCopyPad(biasLocal, biasGlobal_, 
        {1, static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    
    AscendC::PipeBarrier<PIPE_ALL>();
    
    hcBeforeNormQueue_.EnQue(hcBeforeNormLocal);
    invRmsQueue_.EnQue(invRmsLocal);
    hcBeforeNormLocal = hcBeforeNormQueue_.DeQue<T>();
    invRmsLocal = invRmsQueue_.DeQue<T>();
    
    LocalTensor<T> normOutForwardLocal = normOutForwardBuf_.Get<T>();
    LocalTensor<T> zPreLocal = zPreBuf_.Get<T>();
    LocalTensor<T> zPostLocal = zPostBuf_.Get<T>();
    LocalTensor<T> zResLocal = zResBuf_.Get<T>();
    
    // ===== 调试打印：ComputeForwardZFullVf参数（第一个核）=====
    if (blkIdx_ == 0) {
        AscendC::printf("[Step2-Debug] ComputeForwardZFullVf params:\n");
        AscendC::printf("  n=%d, totalElements=%d (tile*(n*n+2n))\n", n_, tile_ * (n_ * n_ + 2 * n_));
        AscendC::printf("  hcBeforeNormLocal size=%d, invRmsLocal size=%d\n", 
            tile_ * (n_ * n_ + 2 * n_), tile_);
        AscendC::printf("  alphaLocal size=3, biasLocal size=%d\n", n_ * n_ + 2 * n_);
        AscendC::printf("  normOutForwardLocal size=%d\n", tile_ * (n_ * n_ + 2 * n_));
        AscendC::printf("  zPreLocal size=%d, zPostLocal size=%d, zResLocal size=%d\n",
            tile_ * n_, tile_ * n_, tile_ * n_ * n_);
        AscendC::printf("  Addresses:\n");
        AscendC::printf("    hcBeforeNormLocal=%p, invRmsLocal=%p\n", 
            hcBeforeNormLocal.GetPhyAddr(), invRmsLocal.GetPhyAddr());
        AscendC::printf("    alphaLocal=%p, biasLocal=%p\n",
            alphaLocal.GetPhyAddr(), biasLocal.GetPhyAddr());
        AscendC::printf("    normOutForwardLocal=%p\n", normOutForwardLocal.GetPhyAddr());
        AscendC::printf("    zPreLocal=%p, zPostLocal=%p, zResLocal=%p\n",
            zPreLocal.GetPhyAddr(), zPostLocal.GetPhyAddr(), zResLocal.GetPhyAddr());
        AscendC::printf("  VF loopCount = %d (totalElements/VL_FP32)\n",
            (tile_ * (n_ * n_ + 2 * n_) + 64 - 1) / 64);
        AscendC::printf("  WARNING: biasPtr offset in VF may cause overflow!\n");
        AscendC::printf("    biasPre offset range: [0, %d) but VF uses [0, %d)\n",
            n_, tile_ * (n_ * n_ + 2 * n_));
        AscendC::printf("    biasPost offset range: [%d, %d) but VF uses [%d, %d)\n",
            n_, 2 * n_, n_, n_ + tile_ * (n_ * n_ + 2 * n_));
    }
    
    ComputeForwardZFullVf<T>(zPreLocal, zPostLocal, zResLocal, normOutForwardLocal, hcBeforeNormLocal, invRmsLocal, alphaLocal, biasLocal, n_, tile_);
    
    hcBeforeNormQueue_.FreeTensor(hcBeforeNormLocal);
    invRmsQueue_.FreeTensor(invRmsLocal);
    
    // ===== 调试打印（对照golden_v2.py行317-331）=====
    // 打印norm_out_forward前8个值（golden: norm_out = hc_before_norm*inv_rms）
    AscendC::printf("[Step2] norm_out_forward (first 8 values):\n");
    for (int i = 0; i < 8 && i < tile_ * (n_ * n_ + 2 * n_); i++) {
        AscendC::printf("  norm_out[%d] = %f\n", i, normOutForwardLocal.GetValue(i));
    }
    
    // 打印z_pre前8个值（golden: z_pre = norm_out[:n]*alpha[0]+bias[:n]）
    AscendC::printf("[Step2] z_pre (first 8 values):\n");
    for (int i = 0; i < 8 && i < tile_ * n_; i++) {
        AscendC::printf("  z_pre[%d] = %f\n", i, zPreLocal.GetValue(i));
    }
    
    // 打印alpha和bias（golden: alpha[3], bias[n²+2n]）
    AscendC::printf("[Step2] alpha = [%f, %f, %f]\n", alphaLocal.GetValue(0), alphaLocal.GetValue(1), alphaLocal.GetValue(2));
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
    
    LocalTensor<T> gradHPostLocal = gradHPostBuf_.Get<T>();
    int64_t gmOffset = blkIdx_ * tile_;
    AscendC::DataCopyPad(gradHPostLocal, gradHPostGlobal_[gmOffset * n_], 
        {1, static_cast<uint32_t>(tile_ * n_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    AscendC::PipeBarrier<PIPE_ALL>();
    
    LocalTensor<T> sigmaPreLocal = sigmaPreBuf_.Get<T>();
    AscendC::Sigmoid(sigmaPreLocal, zPreLocal, tile_ * n_);
    
    LocalTensor<T> sigmaPostLocal = sigmaPostBuf_.Get<T>();
    AscendC::Sigmoid(sigmaPostLocal, zPostLocal, tile_ * n_);
    
    LocalTensor<T> gradHPreLocal = gradHPreBuf_.Get<T>();
    SigmoidGradArithmeticVf<T>(gradZPreLocal, gradZPostLocal, sigmaPreLocal, sigmaPostLocal, gradHPreLocal, gradHPostLocal, tile_ * n_, static_cast<T>(HC_EPS));
    
    // ===== 调试打印（对照golden_v2.py行347-357）=====
    // 打印grad_z_pre前8个值（golden: grad_z_pre = SigmoidGrad(z_pre, grad_h_pre)）
    AscendC::printf("[Step3] grad_z_pre (first 8 values):\n");
    for (int i = 0; i < 8 && i < tile_ * n_; i++) {
        AscendC::printf("  grad_z_pre[%d] = %f\n", i, gradZPreLocal.GetValue(i));
    }
    
    // 打印grad_z_post前8个值（golden: grad_z_post = 2*SigmoidGrad(z_post, grad_h_post)）
    AscendC::printf("[Step3] grad_z_post (first 8 values):\n");
    for (int i = 0; i < 8 && i < tile_ * n_; i++) {
        AscendC::printf("  grad_z_post[%d] = %f\n", i, gradZPostLocal.GetValue(i));
    }
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
    
    // ===== 计算阶段1：大VF（AlphaBiasGradPrePostVf）=====
    AlphaBiasGradPrePostVf<T>(gradAlphaLocal, gradBiasLocal, normOutForwardLocal, gradZPreLocal, gradZPostLocal, n_, tile_ * n_);
    
    // ===== 搬入阶段：workspace旧值并行读取 =====
    int64_t wsOffset = blkIdx_ * wsPerCoreSize_;
    GlobalTensor<T> wsGradAlpha = workspaceGlobal_[wsOffset + wsGradAlphaOffset_];
    GlobalTensor<T> wsGradBias = workspaceGlobal_[wsOffset + wsGradBiasOffset_];
    
    // tmpBuf偏移分离：wsGradAlphaLocal在偏移0处，wsGradBiasLocal在偏移3处，互不覆盖
    LocalTensor<T> wsGradAlphaLocal = tmpBuf_.Get<T>();
    LocalTensor<T> wsGradBiasLocal = tmpBuf_.Get<T>()[3];
    
    AscendC::DataCopyPad(wsGradAlphaLocal, wsGradAlpha,
        {1, static_cast<uint32_t>(3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    AscendC::DataCopyPad(wsGradBiasLocal, wsGradBias,
        {1, static_cast<uint32_t>((2 * n_) * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    AscendC::PipeBarrier<PIPE_ALL>();
    
    // ===== 计算阶段2：加法累加 =====
    AscendC::Add(wsGradAlphaLocal, wsGradAlphaLocal, gradAlphaLocal, 3);
    AscendC::Add(wsGradBiasLocal, wsGradBiasLocal, gradBiasLocal, 2 * n_);
    
    // ===== 搬出阶段：并行写回workspace =====
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(wsGradBias, wsGradBiasLocal,
        {1, static_cast<uint16_t>((2 * n_) * sizeof(T)), 0, 0});
    
    // ===== 调试打印（对照golden_v2.py行369-381）=====
    // 打印grad_alpha（scalar，golden: grad_alpha_pre/post scalar）
    AscendC::printf("[Step45] grad_alpha (scalar values):\n");
    AscendC::printf("  grad_alpha_pre = %f\n", gradAlphaLocal.GetValue(0));
    AscendC::printf("  grad_alpha_post = %f\n", gradAlphaLocal.GetValue(1));
    
    // 打印grad_bias前8个值（golden: grad_bias_part1/2）
    AscendC::printf("[Step45] grad_bias (first 8 values):\n");
    for (int i = 0; i < 8 && i < 2 * n_; i++) {
        AscendC::printf("  grad_bias[%d] = %f\n", i, gradBiasLocal.GetValue(i));
    }
}

// ========== 后续步骤入口标记（Step6-15，非验证范围）==========
// 【说明】以下步骤不在当前验证范围内（Step1-5验证）
// 【Step678】行541-662：Sinkhorn + Exp梯度（hbj负责）
// 【Step911】行672-682：Linear梯度拼接（lhp负责）
// 【Step12】行686-701：RMSNorm梯度（lhp负责）
// 【Step14】行707-764：RMSNorm倒数梯度（lhp负责）
// 【Step15】行770-780：Matmul梯度（lhp负责）
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
    AscendC::PipeBarrier<PIPE_MTE2>();
    gradHResQueue_.EnQue(gradHResLocal);
    
    LocalTensor<T> sumOutLocal = sumOutQueue_.AllocTensor<T>();
    LocalTensor<T> normOutLocal = normOutQueue_.AllocTensor<T>();
    DataCopyPad(sumOutLocal, sumOutGlobal_, {1, static_cast<uint32_t>(2 * skIterCount_ * n_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(normOutLocal, normOutGlobal_, {1, static_cast<uint32_t>(2 * skIterCount_ * n_ * n_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    sumOutQueue_.EnQue(sumOutLocal);
    normOutQueue_.EnQue(normOutLocal);
    
    gradHResLocal = gradHResQueue_.DeQue<T>();
    sumOutLocal = sumOutQueue_.DeQue<T>();
    normOutLocal = normOutQueue_.DeQue<T>();
    
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
    AscendC::PipeBarrier<PIPE_MTE2>();
    Add(wsGradAlphaLocal, wsGradAlphaLocal, gradAlphaLocal, 3);
    Add(wsGradBiasLocal, wsGradBiasLocal, gradBiasLocal, n_ * n_ + 2 * n_);
    DataCopyPad(wsGradAlpha, wsGradAlphaLocal, {1, static_cast<uint32_t>(3 * sizeof(T)), 0, 0, 0});
    DataCopyPad(wsGradBias, wsGradBiasLocal, {1, static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0});
    
    gradHResQueue_.FreeTensor(gradHResLocal);
    sumOutQueue_.FreeTensor(sumOutLocal);
    normOutQueue_.FreeTensor(normOutLocal);
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
    
    ComputeLinearGradVf<T>(gradNormOutLocal, gradZPreLocal, gradZPostLocal, gradZResLocal, alphaLocal, n_, tile_ * (n_ * n_ + 2 * n_));
    AscendC::PipeBarrier<PIPE_ALL>();
}

// ========== Step 12: RMSNorm梯度（golden_v2.py 191-192行）==========
template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<T, U>::Step12_RMSNormGrad() {
    LocalTensor<T> gradNormOutLocal = gradNormOutBuf_.Get<T>();
    LocalTensor<T> invRmsLocal = tmpBuf_.Get<T>();
    
    int64_t gmOffset = blkIdx_ * tile_;
    DataCopyPad(invRmsLocal, invRmsGlobal_[gmOffset], {1, static_cast<uint32_t>(tile_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    AscendC::PipeBarrier<PIPE_MTE2>();
    
    LocalTensor<T> gradHcLocal = gradHcQueue_.AllocTensor<T>();
    
    // ★VF: grad_hc = grad_norm_out * inv_rms
    ComputeRMSNormGradVf<T>(gradHcLocal, gradNormOutLocal, invRmsLocal, tile_ * (n_ * n_ + 2 * n_));
    AscendC::PipeBarrier<PIPE_V>();
    
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
    AscendC::PipeBarrier<PIPE_MTE2>();
    
    // ★VF: grad_inv_rms = (hc_before_norm * grad_norm_out).sum(-1) [golden 192行]
    LocalTensor<T> gradInvRmsLocal = tmpBuf_.Get<T>();
    MulReduceSumVf<T>(gradInvRmsLocal, hcBeforeNormLocal, gradNormOutLocal, tile_ * (n_ * n_ + 2 * n_));
    AscendC::PipeBarrier<PIPE_V>();
    
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
        AscendC::PipeBarrier<PIPE_MTE2>();
        
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
        AscendC::PipeBarrier<PIPE_MTE2>();
        
        for (int64_t c = 1; c < aivNum_; c++) {
            int64_t wsOffset = c * wsPerCoreSize_ + wsGradAlphaOffset_;
            GlobalTensor<T> wsGradAlphaCore = workspaceGlobal_[wsOffset];
            LocalTensor<T> wsGradAlphaLocal = tmpBuf_.Get<T>();
            DataCopyPad(wsGradAlphaLocal, wsGradAlphaCore, {1, static_cast<uint32_t>(3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
            AscendC::PipeBarrier<PIPE_MTE2>();
            Add(totalGradAlpha, totalGradAlpha, wsGradAlphaLocal, 3);
        }
        DataCopyPad(gradAlphaGlobal_, totalGradAlpha, {1, static_cast<uint32_t>(3 * sizeof(T)), 0, 0, 0});
        
        // grad_bias累加
        LocalTensor<T> totalGradBias = gradBiasBuf_.Get<T>();
        DataCopyPad(totalGradBias, workspaceGlobal_[wsGradBiasOffset_], {1, static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        AscendC::PipeBarrier<PIPE_MTE2>();
        
        for (int64_t c = 1; c < aivNum_; c++) {
            int64_t wsOffset = c * wsPerCoreSize_ + wsGradBiasOffset_;
            GlobalTensor<T> wsGradBiasCore = workspaceGlobal_[wsOffset];
            LocalTensor<T> wsGradBiasLocal = tmpBuf_.Get<T>();
            DataCopyPad(wsGradBiasLocal, wsGradBiasCore, {1, static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
            AscendC::PipeBarrier<PIPE_MTE2>();
            Add(totalGradBias, totalGradBias, wsGradBiasLocal, n_ * n_ + 2 * n_);
        }
        DataCopyPad(gradBiasGlobal_, totalGradBias, {1, static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0});
        
        // grad_phi累加
        LocalTensor<T> totalGradPhi = tmpBuf_.Get<T>();
        DataCopyPad(totalGradPhi, workspaceGlobal_[wsGradPhiOffset_], {1, static_cast<uint32_t>((n_ * n_ + 2 * n_) * n_ * c_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        AscendC::PipeBarrier<PIPE_MTE2>();
        
        for (int64_t c = 1; c < aivNum_; c++) {
            int64_t wsOffset = c * wsPerCoreSize_ + wsGradPhiOffset_;
            GlobalTensor<T> wsGradPhiCore = workspaceGlobal_[wsOffset];
            LocalTensor<T> wsGradPhiLocal = tmpBuf_.Get<T>();
            DataCopyPad(wsGradPhiLocal, wsGradPhiCore, {1, static_cast<uint32_t>((n_ * n_ + 2 * n_) * n_ * c_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
            AscendC::PipeBarrier<PIPE_MTE2>();
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
                AscendC::PipeBarrier<PIPE_MTE2>();
                
                int64_t wsGradXFromRmsOffset_core0 = wsGradXFromRmsOffset_ + d_offset * FP32_BYTE_SIZE;
                DataCopyPad(gradXFromRmsChunk, workspaceGlobal_[wsGradXFromRmsOffset_core0 / sizeof(T)], {1, static_cast<uint32_t>(cur_c * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
                AscendC::PipeBarrier<PIPE_MTE2>();
                
                LocalTensor<U> gradXFromMatmulInput = tmpBuf_.Get<U>();
                DataCopyPad(gradXFromMatmulInput, gradXGlobal_[gmOffset * n_c + d_offset], {1, static_cast<uint32_t>(cur_c * sizeof(U)), 0, 0, 0}, {false, 0, 0, 0});
                AscendC::PipeBarrier<PIPE_MTE2>();
                Cast(gradXFromMatmulChunk, gradXFromMatmulInput, RoundMode::CAST_NONE, cur_c);
                
                for (int64_t core = 1; core < aivNum_; core++) {
                    int64_t wsBaseCore = core * wsPerCoreSize_;
                    int64_t wsGradXOff_core = wsBaseCore + wsGradXOffset_ + d_offset * FP32_BYTE_SIZE;
                    GlobalTensor<T> wsGradXFromHinCore = workspaceGlobal_[wsGradXOff_core / sizeof(T)];
                    LocalTensor<T> wsGradXFromHinLocal = tmpBuf_.Get<T>();
                    DataCopyPad(wsGradXFromHinLocal, wsGradXFromHinCore, {1, static_cast<uint32_t>(cur_c * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    Add(gradXFromHinChunk, gradXFromHinChunk, wsGradXFromHinLocal, cur_c);
                    
                    int64_t wsGradXFromRmsOff_core = wsBaseCore + wsGradXFromRmsOffset_ + d_offset * FP32_BYTE_SIZE;
                    GlobalTensor<T> wsGradXFromRmsCore = workspaceGlobal_[wsGradXFromRmsOff_core / sizeof(T)];
                    LocalTensor<T> wsGradXFromRmsLocal = tmpBuf_.Get<T>();
                    DataCopyPad(wsGradXFromRmsLocal, wsGradXFromRmsCore, {1, static_cast<uint32_t>(cur_c * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
                    AscendC::PipeBarrier<PIPE_MTE2>();
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