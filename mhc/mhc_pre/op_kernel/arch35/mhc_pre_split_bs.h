#ifndef MHC_PRE_SPLIT_BS_H
#define MHC_PRE_SPLIT_BS_H

#include "mhc_pre_bs_common.h"
#include "mhc_pre_cube_compute_arch35.h"

namespace MhcPre {

template <class T, class P, int8_t HAS_RESI>
class MhcPreKernelSplitBS : public MhcPreKernelBsCommon<T, P, HAS_RESI> {
public:
    using Parent = MhcPreKernelBsCommon<T, P, HAS_RESI>;

    __aicore__ inline MhcPreKernelSplitBS() = default;

    __aicore__ inline void Init(InitParams initParams)
    {
        this->BindGlobalTensors(initParams);
        this->InitFromTilingData(initParams.tilingData);
        this->InitMNConfig();
        this->InitHMixBuffer(initParams);
        this->InitPipeAndCoreIdx(initParams.tPipeIn);

        // Keep this as the first UB allocation so AIC and both AIVs share the same dual-dst address.
        uint32_t mmRowsPerAiv = (this->chunkTSize_ + this->kHalfSplitDivisor - 1U) / this->kHalfSplitDivisor;
        this->pipe_->InitBuffer(mmXBuf_, mmRowsPerAiv * this->tiling_->fusionAlign * sizeof(float));
        mmXLocal_ = mmXBuf_.Get<float>();

        InitUbBuffersLowApi();
        SyncAll<false>();

        if ASCEND_IS_AIV {
            this->coreIdx_ = GetBlockIdx() / this->kDoubleBufferCount;
            this->AIVPreLoad();
        }
        if ASCEND_IS_AIC {
            this->pipe_->InitBuffer(l1Buffer_, MHC_PRE_LOW_API_L1_ALLOC_SIZE);
            aL1_ = l1Buffer_.Get<float>();
            bL1_ = aL1_[MHC_PRE_LOW_API_L1_BUF_NUM * MHC_PRE_LOW_API_L1_BUF_OFFSET];
            mmService_.Init(this->implMode_);
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t tBlockNum = Ceil(this->totalLength_, this->chunkTSize_);

        for (uint64_t curblock = this->coreIdx_; curblock < tBlockNum; curblock += this->coreNum_) {
            this->InitBlockParams(curblock, tBlockNum);
            ConfigureDirectUbRowPartition();

            if ASCEND_IS_AIC {
                this->mnConfig_.curSingleCoreK = this->mnConfig_.singleCoreK;
            }

            for (uint32_t offsetNd = 0; offsetNd < this->matrixInfo_.nD; offsetNd += this->ND_LENGTH) {
                // Two X staging slots are free initially. From the third slice on,
                // AIV waits until AIC returns the slot through X_CONSUMED_FLAG.
                if ASCEND_IS_AIV {
                    uint32_t curNdLen = MhcPreUtils::Min(this->ND_LENGTH, this->matrixInfo_.nD - offsetNd);
                    if (this->vectorCount_ >= 2) {
                        AscendC::CrossCoreWaitFlag<MhcPreUtils::CROSS_CORE_SYNC_MODE, PIPE_MTE3>(
                            MhcPreUtils::X_CONSUMED_FLAG);
                    }
                    this->V0Prologue(curNdLen, offsetNd);
                    CrossCoreSetFlag<MhcPreUtils::CROSS_CORE_SYNC_MODE, PIPE_MTE3>(MhcPreUtils::X_READY_FLAG);
                    this->vectorCount_++;
                }

                if ASCEND_IS_AIC {
                    AscendC::CrossCoreWaitFlag<MhcPreUtils::CROSS_CORE_SYNC_MODE, PIPE_MTE2>(MhcPreUtils::X_READY_FLAG);
                    AscendC::CrossCoreWaitFlag<MhcPreUtils::CROSS_CORE_SYNC_MODE, PIPE_MTE2>(
                        MhcPreUtils::X_READY_FLAG + MhcPreUtils::SUBBLOCK_FLAG_OFFSET);
                    AICProcessLowApi(offsetNd);
                    CrossCoreSetFlag<MhcPreUtils::CROSS_CORE_SYNC_MODE, PIPE_MTE2>(MhcPreUtils::X_CONSUMED_FLAG);
                    CrossCoreSetFlag<MhcPreUtils::CROSS_CORE_SYNC_MODE, PIPE_MTE2>(
                        MhcPreUtils::X_CONSUMED_FLAG + MhcPreUtils::SUBBLOCK_FLAG_OFFSET);
                }
            }
            this->mmCount_++;
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<MhcPreUtils::CROSS_CORE_SYNC_MODE, PIPE_FIX>(MhcPreUtils::MM_READY_FLAG);
                CrossCoreSetFlag<MhcPreUtils::CROSS_CORE_SYNC_MODE, PIPE_FIX>(
                    MhcPreUtils::MM_READY_FLAG + MhcPreUtils::SUBBLOCK_FLAG_OFFSET);
            }

            if ASCEND_IS_AIV {
                this->V0PostProcess(curblock, tBlockNum);
                AIV1ProcessDirect();
                this->vec1Count_++;
            }
        }

        if ASCEND_IS_AIV {
            this->invRmsOutQueue_.FreeTensor(this->invRmsUb_);
            this->biasInQue_.FreeTensor(this->biasInUb_);
        } else {
            mmService_.End(this->implMode_);
        }
    }

private:
    __aicore__ inline void InitUbBuffersLowApi()
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }

        static constexpr uint32_t kXInQueueBufferBytes = 80 * 1024;
        static constexpr uint32_t kOutQueueBufferBytes = 20 * 1024;

        this->pipe_->InitBuffer(this->xInQueue_, this->kDoubleBufferCount, kXInQueueBufferBytes);
        this->pipe_->InitBuffer(this->outQueue_, this->kDoubleBufferCount, kOutQueueBufferBytes);

        uint32_t invRmsRows =
            (this->curSingleT_ + this->kHalfSplitDivisor - 1U) / this->kHalfSplitDivisor;
        uint32_t invRmsBytes = MhcPreUtils::Align(
            invRmsRows, this->kAlignmentBytes / sizeof(P)) * sizeof(P);
        this->pipe_->InitBuffer(this->invRmsOutQueue_, this->kSingleBufferCount, invRmsBytes);

        if (this->hasGamma_) {
            this->pipe_->InitBuffer(this->gammaInQueue_, this->kSingleBufferCount,
                this->ND_LENGTH * sizeof(P));
        }

        uint32_t parameterBufferBytes = MhcPreUtils::Align(
            this->mnConfig_.n * sizeof(P), this->kAlignmentBytes);
        this->pipe_->InitBuffer(this->biasInQue_, this->kSingleBufferCount, parameterBufferBytes);
        this->pipe_->InitBuffer(this->alphaBuf_, parameterBufferBytes);
        this->alphaInUb_ = this->alphaBuf_.template Get<P>();

        uint32_t hSegmentBytes = MhcPreUtils::Align(
            this->V1_BASE_T * this->N_ * sizeof(P), this->kAlignmentBytes);
        this->pipe_->InitBuffer(this->tmpBuff_, 2U * hSegmentBytes);
        this->hPreBuff_ = this->tmpBuff_.template GetWithOffset<P>(
            static_cast<uint32_t>(this->V1_BASE_T * this->N_), 0);
        this->hPostBuff_ = this->tmpBuff_.template GetWithOffset<P>(
            static_cast<uint32_t>(this->V1_BASE_T * this->N_), hSegmentBytes);
    }

    __aicore__ inline void ConfigureDirectUbRowPartition()
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }
        uint64_t firstRows = (this->curSingleT_ + 1U) / 2U;
        if (this->subBlockIdx_ == 0) {
            this->vectorOffset_.singleCoreM = firstRows;
            this->vectorOffset_.offsetMStart = 0;
            this->vectorOffset_.offsetMEnd = firstRows;
        } else {
            this->vectorOffset_.singleCoreM = this->curSingleT_ - firstRows;
            this->vectorOffset_.offsetMStart = firstRows;
            this->vectorOffset_.offsetMEnd = this->curSingleT_;
        }
    }

    __aicore__ inline void AICProcessLowApi(uint32_t offsetNd)
    {
        if (offsetNd + this->mnConfig_.singleCoreK > this->mnConfig_.k) {
            this->mnConfig_.curSingleCoreK = this->mnConfig_.k - offsetNd;
        }

        uint32_t currentK = this->mnConfig_.curSingleCoreK;
        uint64_t xOffset = this->chunkTSize_ * this->ND_LENGTH *
            (this->coreIdx_ + (this->cubeCount_ % this->PARALLEL_NUM) * this->coreNum_);
        uint8_t aL1BufferId = this->cubeCount_ & 1;
        uint8_t bL1BufferId = mmService_.GetBL1BufferId();

        LocalTensor<float> currentAL1 = aL1_[aL1BufferId * MHC_PRE_LOW_API_L1_BUF_OFFSET];
        LocalTensor<float> currentBL1 = bL1_[bL1BufferId * MHC_PRE_LOW_API_L1_BUF_OFFSET];
        mmService_.CopyInA1Nd2Nz(this->mnConfig_.curSingleCoreM, currentK,
            this->xFloatGm_[xOffset], currentAL1);
        mmService_.CopyInB1Nd2Nz(this->mnConfig_.k, currentK, this->mnConfig_.n,
            this->phiGm_[offsetNd], currentBL1);

        bool isFirstKL1 = (offsetNd == 0);
        bool isLastKL1 = (offsetNd + currentK >= this->matrixInfo_.nD);
        uint64_t mAlign = LowApiAlign(this->mnConfig_.curSingleCoreM, AscendC::BLOCK_CUBE);
        uint64_t nAlign = LowApiAlign(this->mnConfig_.n, AscendC::BLOCK_CUBE);
        uint64_t baseK = (256 / AscendC::Std::max(mAlign, nAlign)) * 32;
        mmService_.Process(this->mnConfig_.curSingleCoreM, this->mnConfig_.n,
            this->mnConfig_.curSingleCoreM, baseK, isFirstKL1, isLastKL1, currentAL1, currentBL1);

        if (isLastKL1) {
            mmService_.CopyOut(mmXLocal_);
        }
        this->cubeCount_++;
    }

    __aicore__ inline void AIV1ProcessDirect()
    {
        AscendC::CrossCoreWaitFlag<MhcPreUtils::CROSS_CORE_SYNC_MODE, PIPE_V>(MhcPreUtils::MM_READY_FLAG);
        if (this->vectorOffset_.singleCoreM == 0) {
            return;
        }

        uint64_t localOffset = 0;
        for (uint64_t offsetT = this->vectorOffset_.offsetMStart;
             offsetT < this->vectorOffset_.offsetMEnd; offsetT += this->V1_BASE_T) {
            uint64_t lenT = AscendC::Std::min(static_cast<uint64_t>(this->V1_BASE_T),
                this->vectorOffset_.offsetMEnd - offsetT);
            AIV1PrologueDirect(offsetT, lenT, localOffset);
            this->AIV1ProcessHPost(offsetT, lenT);
            this->AIV1ProcessHPre(offsetT, lenT);
            this->AIV1ProcessHInBatched(offsetT, lenT);
            localOffset += lenT;
        }
    }

    __aicore__ inline void CopyRawHMixToGm(uint64_t offsetT, uint64_t lenT)
    {
        LocalTensor<P> copySource = this->matmulRes_;
        LocalTensor<P> compactLocal;
        // Fixpipe uses fusionAlign as the UB row stride. Compact rows before
        // writing raw hMix when the logical fusion size is not 32-byte aligned.
        if (this->tiling_->fusionAlign != this->mnConfig_.n) {
            compactLocal = this->xInQueue_.template AllocTensor<P>();
            MhcPreVFCompactRows(compactLocal, this->matmulRes_, static_cast<uint16_t>(lenT),
                this->tiling_->fusionAlign, this->mnConfig_.n);
            copySource = compactLocal;
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(lenT * this->mnConfig_.n * sizeof(P));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(this->hMixGm_[(this->globalOffsetM_ + offsetT) * this->mnConfig_.n],
            copySource, copyParams);

        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        if (this->tiling_->fusionAlign != this->mnConfig_.n) {
            this->xInQueue_.FreeTensor(compactLocal);
        }
    }

    __aicore__ inline void AIV1PrologueDirect(uint64_t offsetT, uint64_t lenT, uint64_t localOffset)
    {
        uint64_t outputOffset = this->globalOffsetM_ + offsetT;
        LocalTensor<P> hResOutLocal;
        if constexpr (HAS_RESI == MHC_PRE_HAS_RESI) {
            hResOutLocal = this->outQueue_.template AllocTensor<P>();
        }

        this->matmulRes_ = mmXLocal_[localOffset * this->tiling_->fusionAlign];
        if (this->outFlag_) {
            CopyRawHMixToGm(offsetT, lenT);
        }

        if (this->N_ == this->kSupportedN4) {
            MhcPreVFPostSegments<P, 4U, HAS_RESI == MHC_PRE_HAS_RESI>(
                this->hPreBuff_, this->hPostBuff_, hResOutLocal, this->matmulRes_,
                this->invRmsUb_, this->alphaInUb_, this->biasInUb_, static_cast<uint16_t>(lenT),
                static_cast<uint32_t>(localOffset), this->tiling_->fusionAlign);
        } else if (this->N_ == this->kSupportedN6) {
            MhcPreVFPostSegments<P, 6U, HAS_RESI == MHC_PRE_HAS_RESI>(
                this->hPreBuff_, this->hPostBuff_, hResOutLocal, this->matmulRes_,
                this->invRmsUb_, this->alphaInUb_, this->biasInUb_, static_cast<uint16_t>(lenT),
                static_cast<uint32_t>(localOffset), this->tiling_->fusionAlign);
        } else {
            MhcPreVFPostSegments<P, 8U, HAS_RESI == MHC_PRE_HAS_RESI>(
                this->hPreBuff_, this->hPostBuff_, hResOutLocal, this->matmulRes_,
                this->invRmsUb_, this->alphaInUb_, this->biasInUb_, static_cast<uint16_t>(lenT),
                static_cast<uint32_t>(localOffset), this->tiling_->fusionAlign);
        }
        PipeBarrier<PIPE_V>();

        if constexpr (HAS_RESI == MHC_PRE_HAS_RESI) {
            this->outQueue_.EnQue(hResOutLocal);
            hResOutLocal = this->outQueue_.template DeQue<P>();
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(lenT * this->N_ * this->N_ * sizeof(P));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(this->hResGm_[outputOffset * this->N_ * this->N_], hResOutLocal, copyParams);
            this->outQueue_.FreeTensor(hResOutLocal);
        }
    }

    MhcPreCubeCompute mmService_;
    TBuf<TPosition::A1> l1Buffer_;
    TBuf<TPosition::VECCALC> mmXBuf_;
    LocalTensor<float> aL1_;
    LocalTensor<float> bL1_;
    LocalTensor<float> mmXLocal_;
};

} // namespace MhcPre

#endif // MHC_PRE_SPLIT_BS_H
