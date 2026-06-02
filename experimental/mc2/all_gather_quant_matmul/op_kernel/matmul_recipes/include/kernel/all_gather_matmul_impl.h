/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

/*!
 * \file all_gather_matmul_impl.h
 * \brief AllGather + QuantizedMatmul 融合算子（1:2 CV 混核模式）
 */

#ifndef ALL_GATHER_MATMUL_IMPL_H
#define ALL_GATHER_MATMUL_IMPL_H

#include "basic_api/kernel_basic_intf.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/sum.h"
#include "adv_api/pad/broadcast.h"
#include "kernel_tiling/kernel_tiling.h"
#include "all_gather_mte.h"
#include "all_gather_mte_comm.h"
#include "block/block_mmad.h"
#include "block/block_scheduler_policy.h"
#include "../../common/op_kernel/moe_distribute_base.h"
#include "add_rms_norm_dynamic_quant_v2_helper.h"
#include "common.h"
#include "quantize_functions.h"
#include "reduce_common.h"
#include "kernel/quant_matmul_mx_kernel_swat.h"
#include "kernel_utils/common_utils.h"
#include "block/quant_matmul_mx_block_mmad_swat.h"

namespace allGatherMatmulImpl {

#define TemplateMC2TypeClass typename X1Type, typename X2Type, typename ScaleType, uint32_t SyncMode, bool isOptionalOutput, bool isSmoothScale
#define TemplateMC2TypeFunc X1Type, X2Type, ScaleType, SyncMode, isOptionalOutput, isSmoothScale

// reg 单指令饱和模式
#ifndef FLOAT_OVERFLOW_MODE_CTRL
#define FLOAT_OVERFLOW_MODE_CTRL 60
#endif

using namespace AscendC;
using namespace AllGatherImpl;

constexpr static uint64_t WIN_ALIGN = 512;
constexpr static uint64_t UB_ALIGN = 32;
constexpr static uint8_t BUFFER_NUM = 2;           // 多Buf

template<TemplateMC2TypeClass>
class allGatherMatmulImpl {
public:
    __aicore__ inline allGatherMatmulImpl() {};

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR residual, GM_ADDR y, GM_ADDR gamma,
                                GM_ADDR scaleInv1, GM_ADDR smoothScale, GM_ADDR biasGM,
                                GM_ADDR scaleInv2, GM_ADDR cGM, GM_ADDR z, GM_ADDR dynamicQuantOut,
                                GM_ADDR allGatherDataOut, GM_ADDR allGatherScalesOut,
                                GM_ADDR tilingGM);
    __aicore__ inline void Process();

    

    __gm__ Mc2Kernel::HcclOpParam* context_; // HCCL 通信上下文（含 win 区地址数组）

private:
    __aicore__ inline void Add2RmsNormDynamicQuantProcess();
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startId, uint32_t &endId,uint32_t &sendNum);
    __aicore__ inline void Add2RmsNormCompute(int32_t gmOffset, int32_t elementCount);
    __aicore__ inline void GammaWeightAndCopyOut(int32_t gmOffset);
    __aicore__ inline void MxFp8DynamicQuant(int32_t offset, LocalTensor<float> &yTempTensor,
                                             LocalTensor<float> &xTempTensor);
    __aicore__ inline void InitBaseParams(const __gm__ allGatherTilingData *tilingData);
    __aicore__ inline void AllGatherProcess();
    __aicore__ inline void MatmulProcess();

    TPipe *tpipe_{nullptr};
    AllGatherMte<int8_t, int8_t, int8_t, SyncMode, isOptionalOutput> allGatherMte_;
    GM_ADDR allGatherDataOutAddr_;
    GM_ADDR allGatherScalesOutAddr_;
    GM_ADDR inputXAddr_;
    GM_ADDR inputXScaleAddr_;
    GM_ADDR selfRankAddr_;
    GM_ADDR scaleAAddr_;
    GM_ADDR bGM_;
    GM_ADDR cGM_;
    GM_ADDR scaleInv2GM;
    GM_ADDR tilingGM_;
    __gm__ allGatherMatmulTilingData* tilingData;

    uint32_t aivNum_{0};
    uint32_t aivId_{0};
    uint32_t aicId_{0};
    uint32_t rankId_{0};
    uint32_t axisM_{0};
    uint32_t axisKa_{0};
    uint32_t axisN_{0};
    uint32_t rankSize_{0};
    uint32_t tileK_{0};
    uint32_t sendCoreNumPerRank_{0};
    uint32_t kMteCoreK_{0};
    uint32_t numAgSignalIter_{0};
    uint64_t kL1_{0};
    uint64_t axisKaAlignSize_{0};
    uint64_t axisKaAlignFloatSize_{0};
    uint64_t axisKaAlignInt8Size_{0};
    int64_t residualNormMode_{0};
    float eps_{0};
    float aveNum_{0};

    TBuf<> smoothScaleBuf_; // 搬入smoothScale
    TBuf<> gammaBuf_;
    TBuf<> x1TempBuf_;
    TBuf<> yTempBuf_;
    TBuf<> normOutTempBuf_;
    TBuf<> dynamicScaleBuf_;

    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> x1OutQueue_;
    TQue<QuePosition::VECOUT, 1> zOutQueue_;

    GlobalTensor<X1Type> x1GMTensor_;
    GlobalTensor<X1Type> residualGMTensor_;
    GlobalTensor<X1Type> yGMTensor_;
    GlobalTensor<float> gammaGMTensor_;
    GlobalTensor<float> smoothScaleGMTensor_;

    GlobalTensor<X1Type> zGMTensor_;
    GlobalTensor<X2Type> x1WinGMTensor_; // fp8 量化后的过程数据, win区上
    GlobalTensor<fp8_e8m0_t> scaleWinGMTensor_;
    GlobalTensor<X2Type> dynamicQuantOutGMTensor_;
    GlobalTensor<fp8_e8m0_t> allGatherScalesOutGMTensor_;


    LocalTensor<X1Type> x1Tensor_;
    LocalTensor<float> gammaTensor_;
    LocalTensor<float> smoothScaleTensor_; // to be fixed: no init !!!
    LocalTensor<fp8_e8m0_t> dynamicScaleLocalTensor_; // dynamic quant
    LocalTensor<float> xLocalTensorFp32_;
    LocalTensor<float> yLocalTensorFp32_;
    LocalTensor<X2Type> x1OutLocalTensor_; // x1 quant out type depends on X2 Type, try X1Type
    LocalTensor<X1Type> normOutLocalTensor_; 

    GM_ADDR winContext_{nullptr};
};

template<TemplateMC2TypeClass>
__aicore__ inline void allGatherMatmulImpl<TemplateMC2TypeFunc>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR residual, GM_ADDR y, GM_ADDR gamma,
    GM_ADDR scaleInv1, GM_ADDR smoothScale, GM_ADDR biasGM,
    GM_ADDR scaleInv2, GM_ADDR cGM, GM_ADDR z, GM_ADDR dynamicQuantOut,
    GM_ADDR allGatherDataOut, GM_ADDR allGatherScalesOut,
    GM_ADDR tilingGM)
{
    tilingData = (__gm__ allGatherMatmulTilingData*)tilingGM;
    if ASCEND_IS_AIV {
        TPipe pipe;
        tpipe_ =  &pipe;
    }
    winContext_ = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    context_ = (__gm__ Mc2Kernel::HcclOpParam*)winContext_;
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    
    InitBaseParams(&tilingData->allGatherTiling);
    aivNum_ = tilingData->allGatherTiling.aicNum;

    allGatherDataOutAddr_ = allGatherDataOut;
    allGatherScalesOutAddr_ = allGatherScalesOut;
    inputXAddr_ = aGM;
    inputXScaleAddr_ = scaleInv1;
    allGatherMte_.InitParams(&tilingData->allGatherTiling, winContext_);

    uint64_t xSize = axisM_ * axisKa_ * sizeof(int8_t);
    uint64_t winOffset = AscendC::CeilDiv(rankSize_ * xSize, WIN_ALIGN) * WIN_ALIGN;

    selfRankAddr_ = (GM_ADDR)allGatherMte_.GetSelfDataAddrGm();
    scaleAAddr_ = (GM_ADDR)allGatherMte_.GetSelfDataAddrGm() + winOffset;
    bGM_ = bGM;
    cGM_ = cGM;
    scaleInv2GM = scaleInv2;

    if ASCEND_IS_AIC {
        return;
    }
}

template<TemplateMC2TypeClass>
__aicore__ inline void allGatherMatmulImpl<TemplateMC2TypeFunc>::Add2RmsNormDynamicQuantProcess()
{
    // PRINTF("Enter Add2RmsNormDynamicQuantProcess \n");
    DataCopyEx(gammaTensor_, gammaGMTensor_, axisKa_);
    if constexpr (isSmoothScale) {
        DataCopyEx(smoothScaleTensor_, smoothScaleGMTensor_, axisKa_);
    }

    uint32_t startRowId = 0;
    uint32_t endRowId = 0;
    uint32_t rowNum = 0;
    SplitToCore(axisM_, aivNum_, startRowId, endRowId, rowNum);
    if (startRowId > axisM_) {
        return;
    }

    // dyn mx scale --> rowNum * K / 32 * sizeof(fp8_e8m0_t)
    uint32_t rowNumSize = Ceil(rowNum * sizeof(fp8_e8m0_t), UB_ALIGN) * UB_ALIGN;
    uint32_t mxScaleNumPerRow = Align2(Ceil32(axisKa_)); // 向偶数对齐，用于后续regbase成组计算
    tpipe_->InitBuffer(dynamicScaleBuf_, rowNumSize * mxScaleNumPerRow); 
    dynamicScaleLocalTensor_ = dynamicScaleBuf_.Get<fp8_e8m0_t>();
    // if (GetBlockIdx() == 2) {
    //     printf("check rowNumSize params: rowNumSize: %u, mxScaleNumPerRow: %u \n", rowNumSize, mxScaleNumPerRow);
    // }
    uint32_t gmOffset = startRowId * axisKa_;
    uint32_t elementCount = axisKaAlignSize_ / sizeof(X1Type);
    // if (GetBlockIdx() == 2) {
    //     printf("check gmOffset params: gmOffset: %d, elementCount: %d \n", gmOffset, elementCount);
    // }
    for (int32_t rowIdx = startRowId; rowIdx < endRowId; ++rowIdx) { // for 循环内一次计算一行数据+ Ka/32 个mxscale值
        Add2RmsNormCompute(gmOffset, elementCount);
        if (rowIdx == startRowId) {
            SyncFunc<AscendC::HardEvent::MTE2_V>(); // wait gammaTensor_ and smoothScaleTensor_
        }
        GammaWeightAndCopyOut(gmOffset); // normOut = xLocalTensorFp32_
        // Dynamic Quant
        x1OutLocalTensor_ = x1OutQueue_.AllocTensor<X2Type>();
        if constexpr (isSmoothScale) {
            Mul(yLocalTensorFp32_, xLocalTensorFp32_, smoothScaleTensor_, axisKa_); // y * smooth
            PipeBarrier<PIPE_V>();
            MxFp8DynamicQuant(rowIdx - startRowId, yLocalTensorFp32_, xLocalTensorFp32_); // 单核保存mxscale->ub
        } else {
            MxFp8DynamicQuant(rowIdx - startRowId, xLocalTensorFp32_, yLocalTensorFp32_);
        }
        // copy out 到本卡win区
        x1OutQueue_.EnQue<X2Type>(x1OutLocalTensor_);
        x1OutLocalTensor_ = x1OutQueue_.DeQue<X2Type>();
        DataCopyEx(x1WinGMTensor_[gmOffset], x1OutLocalTensor_, axisKa_);
        // copy out to dynamicQuantOut, too
        // if constexpr (isOptionalOutput) {
        if (true) { // 调试用，强制输出dynamic quant out校验动态量化后的结果 20260511
            DataCopyEx(dynamicQuantOutGMTensor_[gmOffset], x1OutLocalTensor_, axisKa_);
        }
        x1OutQueue_.FreeTensor(x1OutLocalTensor_);
        gmOffset += axisKa_;
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    // 一次性把当前核计算的所有mx scale 搬出到local Win
    DataCopyEx(scaleWinGMTensor_[startRowId * mxScaleNumPerRow], dynamicScaleLocalTensor_, rowNum * mxScaleNumPerRow);
    // // 调试用，把dynamic quant scale 输出到allgather scale里面（但其实还没做allgather,只有本卡上的数据），用于校验动态量化后的mx scale值
    DataCopyEx(allGatherScalesOutGMTensor_[startRowId * mxScaleNumPerRow], dynamicScaleLocalTensor_, rowNum * mxScaleNumPerRow);
}

template <TemplateMC2TypeClass>
__aicore__ inline void allGatherMatmulImpl<TemplateMC2TypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startId, uint32_t &endId, uint32_t &sendNum)
{
    // printf("Enter SplitToCore");
    sendNum = curSendCnt / curUseAivNum;               // 每个aiv需要处理的数
    uint32_t remainderNum = curSendCnt % curUseAivNum; // 余数
    startId = sendNum * aivId_;                        // 每个aiv发送时的起始id
    if (aivId_ < remainderNum) {                       // 前remainderNum个aiv需要多处理1个数据
        sendNum += 1;
        startId += aivId_;
    } else {
        startId += remainderNum;
    }
    endId = startId + sendNum;
}

template <TemplateMC2TypeClass>
__aicore__ inline void
allGatherMatmulImpl<TemplateMC2TypeFunc>::Add2RmsNormCompute(int32_t gmOffset, int32_t elementCount)
{
    // Copy In
    LocalTensor<X1Type> copyInLocalTensor = inQueue_.AllocTensor<X1Type>();
    DataCopyEx(copyInLocalTensor[0], x1GMTensor_[gmOffset], axisKa_);
    DataCopyEx(copyInLocalTensor[elementCount], residualGMTensor_[gmOffset], axisKa_);
    DataCopyEx(copyInLocalTensor[elementCount * 2], yGMTensor_[gmOffset], axisKa_);
    inQueue_.EnQue(copyInLocalTensor);
    copyInLocalTensor = inQueue_.DeQue<X1Type>();
    LocalTensor<X1Type> x1LocalTensor = copyInLocalTensor[0];
    LocalTensor<X1Type> residualLocalTensor = copyInLocalTensor[elementCount];
    LocalTensor<X1Type> yLocalTensor = copyInLocalTensor[elementCount * 2];

    // Add
    Cast(xLocalTensorFp32_, x1LocalTensor, RoundMode::CAST_NONE, axisKa_);
    Cast(yLocalTensorFp32_, residualLocalTensor, RoundMode::CAST_NONE, axisKa_);
    PipeBarrier<PIPE_V>();
    Add(xLocalTensorFp32_, xLocalTensorFp32_, yLocalTensorFp32_, axisKa_);
    PipeBarrier<PIPE_V>();
    Cast(yLocalTensorFp32_, yLocalTensor, RoundMode::CAST_NONE, axisKa_);
    PipeBarrier<PIPE_V>();
    Add(xLocalTensorFp32_, xLocalTensorFp32_, yLocalTensorFp32_, axisKa_);
    inQueue_.FreeTensor<X1Type>(copyInLocalTensor);
    PipeBarrier<PIPE_V>();

    // Copy out z
    if (residualNormMode_ == 0) {
        LocalTensor<X1Type> zOutLocalTensor = zOutQueue_.AllocTensor<X1Type>();
        if constexpr (is_same<X1Type, half>::value) {
            Cast(zOutLocalTensor, xLocalTensorFp32_, RoundMode::CAST_NONE, axisKa_);
        } else { // BF16
            Cast(zOutLocalTensor, xLocalTensorFp32_, RoundMode::CAST_RINT, axisKa_);
        }
        PipeBarrier<PIPE_V>();
        zOutQueue_.EnQue(zOutLocalTensor);
        zOutLocalTensor = zOutQueue_.DeQue<X1Type>();
        DataCopyEx(zGMTensor_[gmOffset], zOutLocalTensor, axisKa_);
        zOutQueue_.FreeTensor<X1Type>(zOutLocalTensor);
    }

    // RMS Norm
    PipeBarrier<PIPE_V>();
    Mul(yLocalTensorFp32_, xLocalTensorFp32_, xLocalTensorFp32_, axisKa_); // yLocalTensorFp32_ <- x ** 2
    PipeBarrier<PIPE_V>();

    // reduce#1 for mean
    float squareSumTemp = ReduceSumHalfInterval(yLocalTensorFp32_, axisKa_); // aveLocalTemp <-- E(x**2)
    float rstdLocalTemp = 1 / sqrt(squareSumTemp * aveNum_ + eps_);

    SyncFunc<AscendC::HardEvent::S_V>();
    Muls(xLocalTensorFp32_, xLocalTensorFp32_, rstdLocalTemp, axisKa_); // xLocalTensorFp32_ <- x * rstd
    PipeBarrier<PIPE_V>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void allGatherMatmulImpl<TemplateMC2TypeFunc>::GammaWeightAndCopyOut(int32_t gmOffset)
{
    // Gamma Weight
    Mul(xLocalTensorFp32_, xLocalTensorFp32_, gammaTensor_, axisKa_); // xLocalTensorFp32_ <- x * rstd * gamma
    PipeBarrier<PIPE_V>();
    // CopyOut addRmsNormOut
    // if (residualNormMode_ == 1) {
    //     LocalTensor<X1Type> zOutLocalTensor = zOutQueue_.AllocTensor<X1Type>();
    //     if constexpr (is_same<X1Type, half>::value) {
    //         Cast(zOutLocalTensor, xLocalTensorFp32_, RoundMode::CAST_NONE, axisKa_);
    //     } else { // BF16
    //         Cast(zOutLocalTensor, xLocalTensorFp32_, RoundMode::CAST_RINT, axisKa_);
    //     }
    //     PipeBarrier<PIPE_V>();
    //     zOutQueue_.EnQue(zOutLocalTensor);
    //     zOutLocalTensor = zOutQueue_.DeQue<X1Type>();
    //     DataCopyEx(zGMTensor_[gmOffset], zOutLocalTensor, axisKa_);
    //     zOutQueue_.FreeTensor<X1Type>(zOutLocalTensor);
    // }
}

template <TemplateMC2TypeClass>
__aicore__ inline void allGatherMatmulImpl<TemplateMC2TypeFunc>::MxFp8DynamicQuant(
    int32_t rowOffset, LocalTensor<float> &yTempTensor, LocalTensor<float> &xTempTensor)
{
    // printf("Enter MxFp8DynamicQuant \n");
    // yTempTensor --> norm out --> float --> bf16
    // xTempTensor --> for temp use only
    // x1OutLocalTensor_ --> mx dyn quant output --> fp8
    // dynamicScaleLocalTensor_ --> mx scale output --> fp8e8m0
    uint32_t mxScaleNum = Align2(Ceil32(axisKa_)); // 一行要生成多少个mxscale值, 对齐到2，用于后续成组计算mx scale
    uint32_t scaleOffset = rowOffset * mxScaleNum; // 计算mxscale 拷贝要offset 多少bit

    Cast(normOutLocalTensor_, yTempTensor, RoundMode::CAST_RINT, axisKa_); // 必要: cast norm out input from fp32 to 16
    PipeBarrier<PIPE_V>();

    __ubuf__ X1Type *srcAddr = (__ubuf__ X1Type *)normOutLocalTensor_.GetPhyAddr();

    __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)xTempTensor.GetPhyAddr();
    __ubuf__ uint16_t *halfScaleLocalAddr = (__ubuf__ uint16_t *)xTempTensor[Align32(mxScaleNum)].GetPhyAddr();
    __ubuf__ int8_t *outLocalAddr = (__ubuf__ int8_t *)x1OutLocalTensor_.GetPhyAddr(); // 数据类型不同，可以尝试改成8
    __ubuf__ uint16_t *mxScaleLocalAddr = (__ubuf__ uint16_t *)dynamicScaleLocalTensor_[scaleOffset].GetPhyAddr(); //?

    quant::ComputeMaxExp(srcAddr, maxExpAddr, axisKa_); // 计算最大Exp
    if constexpr (Std::IsSame<X2Type, fp8_e5m2_t>::value) {
        // 计算scales并填充
        quant::ComputeScale<fp8_e5m2_t>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum);
        quant::ComputeFp8Data<X1Type, fp8_e5m2_t, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleLocalAddr, outLocalAddr, axisKa_); // 计算量化后的expandx并填充
    } else if constexpr (Std::IsSame<X2Type, fp8_e4m3fn_t>::value) {
        // 计算scales并填充
        quant::ComputeScale<fp8_e4m3fn_t>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum);
        quant::ComputeFp8Data<X1Type, fp8_e4m3fn_t, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleLocalAddr, outLocalAddr, axisKa_); // 计算量化后的expandx并填充
    }
}

template<TemplateMC2TypeClass>
__aicore__ inline void allGatherMatmulImpl<TemplateMC2TypeFunc>::InitBaseParams(const __gm__ allGatherTilingData *tilingData)
{
    rankId_ = Mc2Kernel::GetRankId(context_);
    aivNum_ = tilingData->aivNum;
    aivId_ = GetBlockIdx();
    aicId_ = aivId_ / GetTaskRation();
    axisM_ = tilingData->M;
    axisKa_ = tilingData->Ka;
    axisN_ = tilingData->N;
    rankSize_ = tilingData->rankSize;
    sendCoreNumPerRank_ = tilingData->sendCoreNumPerRank;
    tileK_ = tilingData->mteTileK;
    kMteCoreK_ = tilingData->mteKSplitSize;
    numAgSignalIter_ = kMteCoreK_ / X_PER_BLOCK_NUM;
    
    if (kMteCoreK_ % X_PER_BLOCK_NUM != 0) {
        numAgSignalIter_++;
    }
}

template<TemplateMC2TypeClass>
__aicore__ inline void allGatherMatmulImpl<TemplateMC2TypeFunc>::MatmulProcess()
{
    using TypeA = X2Type;
    using TypeB = X2Type;
    using TypeC = bfloat16_t;
    constexpr int32_t SCALE_C0 = 2;
    constexpr int32_t C0_SIZE = AscendC::AuxGetC0Size<TypeA>();
    using LayoutA = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn>;
    using LayoutB = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>;
    using LayoutC = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn>;
    using LayoutScaleA = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<SCALE_C0>>;
    using LayoutScaleB = typename AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBNDLayoutPtn, AscendC::Std::Int<SCALE_C0>>;

    using BlockScheduler = QuantMatmulMxSwatScheduler<NO_FULL_LOAD_MODE>;
    using DispatchPolicy = QuantMatmulMxMultiBlockWithSwat<NO_FULL_LOAD_MODE, 2UL>;
    using ProblemShape = MatmulShape;
    using TypeATuple = AscendC::Std::tuple<TypeA, fp8_e8m0_t>;
    using TypeBTuple = AscendC::Std::tuple<TypeB, fp8_e8m0_t>;
    using LayoutATuple = AscendC::Std::tuple<LayoutA, LayoutScaleA>;
    using LayoutBTuple = AscendC::Std::tuple<LayoutB, LayoutScaleB>;
    using BlockMmad = Block::BlockMmad<DispatchPolicy, TypeATuple, LayoutATuple, TypeBTuple, LayoutBTuple, TypeC, LayoutC>;
    using KernelImpl = Kernel::QuantMatmulMxKernelSwat<ProblemShape, BlockMmad, BlockScheduler>;
    using Params = typename KernelImpl::Params;

    ProblemShape problemShape{tilingData->qbmmTilingData.m, tilingData->qbmmTilingData.n,
        tilingData->qbmmTilingData.k, 1UL};
    typename BlockMmad::Params mmadParams{selfRankAddr_, bGM_, cGM_, scaleAAddr_, scaleInv2GM};
    typename KernelImpl::L1Params l1Params{
        static_cast<uint64_t>(tilingData->qbmmTilingData.stepK) * tilingData->qbmmTilingData.baseK,
        tilingData->qbmmTilingData.scaleKL1};
    kL1_ = l1Params.kL1;
    typename KernelImpl::BlockSchedulerParams schedulerParams{
        tilingData->qbmmTilingData.baseM,
        tilingData->qbmmTilingData.baseN,
        tilingData->qbmmTilingData.mTailTile,
        tilingData->qbmmTilingData.nTailTile,
        tilingData->qbmmTilingData.mBaseTailSplitCnt,
        tilingData->qbmmTilingData.nBaseTailSplitCnt,
        tilingData->qbmmTilingData.mTailMain,
        tilingData->qbmmTilingData.nTailMain};
    typename KernelImpl::QBMMTiling qbmmParams{
        tilingData->qbmmTilingData.baseM,
        tilingData->qbmmTilingData.baseN,
        tilingData->qbmmTilingData.baseK,
        tilingData->qbmmTilingData.dbL0c};
    Params params{problemShape, mmadParams, l1Params, schedulerParams, qbmmParams};


    KernelImpl quantMatmulKernelImpl;
    quantMatmulKernelImpl(params);
}

template<TemplateMC2TypeClass>
__aicore__ inline void allGatherMatmulImpl<TemplateMC2TypeFunc>::AllGatherProcess()
{
    allGatherMte_.InitBuffer(tpipe_);
    allGatherMte_.CopyLocalData(inputXAddr_, inputXScaleAddr_);
    allGatherMte_.SetRemoteFlag();
    allGatherMte_.WaitRemoteFlag();
    allGatherMte_.ExecuteAllGather(allGatherDataOutAddr_, allGatherScalesOutAddr_);
}

template<TemplateMC2TypeClass>
__aicore__ inline void allGatherMatmulImpl<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        AllGatherProcess();
    }
    if ASCEND_IS_AIC {
        MatmulProcess();
    }
}

} // allGatherMatmulImpl

#endif  // ALL_GATHER_MATMUL_IMPL_H
