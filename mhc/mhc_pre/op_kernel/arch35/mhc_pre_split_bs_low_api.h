#ifndef MHC_PRE_SPLIT_BS_LOW_API_H
#define MHC_PRE_SPLIT_BS_LOW_API_H

#include "mhc_pre_split_bs.h"
#include "mhc_pre_cube_compute_arch35.h"

namespace MhcPre {

template <class T, class P, int8_t HAS_RESI>
class MhcPreKernelSplitBSLowApi : public MhcPreKernelSplitBS<T, P, HAS_RESI> {
public:
    using Parent = MhcPreKernelSplitBS<T, P, HAS_RESI>;

    __aicore__ inline explicit MhcPreKernelSplitBSLowApi(MT &matmul) : Parent(matmul) {}

    __aicore__ inline void Init(InitParams initParams)
    {
        Parent::Init(initParams);
        if ASCEND_IS_AIC {
            this->pipe_->InitBuffer(l1Buffer_, MHC_PRE_LOW_API_L1_ALLOC_SIZE);
            aL1_ = l1Buffer_.Get<float>();
            bL1_ = aL1_[MHC_PRE_LOW_API_L1_BUF_NUM * MHC_PRE_LOW_API_L1_BUF_OFFSET];
            mmService_.Init();
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t tBlockNum = Ceil(this->totalLength_, this->chunkTSize_);

        for (uint64_t curblock = this->coreIdx_; curblock < tBlockNum; curblock += this->coreNum_) {
            this->InitBlockParams(curblock, tBlockNum);

            uint64_t outOffset = 0;
            if ASCEND_IS_AIC {
                if (this->outFlag_) {
                    outOffset = this->globalOffsetM_ * this->mnConfig_.n;
                } else {
                    outOffset = this->mnConfig_.singleCoreM * this->mnConfig_.singleCoreN *
                        ((this->mmCount_ % 2) * this->coreNum_ + this->coreIdx_);
                }
                this->mnConfig_.curSingleCoreK = this->mnConfig_.singleCoreK;
            }

            for (uint32_t offsetNd = 0; offsetNd < this->matrixInfo_.nD; offsetNd += this->ND_LENGTH) {
                if ASCEND_IS_AIV {
                    uint32_t curNdLen = MhcPreUtils::Min(this->ND_LENGTH, this->matrixInfo_.nD - offsetNd);
                    if (this->vectorCount_ >= 2) {
                        AscendC::CrossCoreWaitFlag(SYNC_C2V);
                    }
                    this->V0Prologue(curNdLen, offsetNd);
                    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_V2C);
                    this->vectorCount_++;
                }

                if ASCEND_IS_AIC {
                    AscendC::CrossCoreWaitFlag(SYNC_V2C);
                    AICProcessLowApi(offsetNd, outOffset);
                    CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_C2V);
                }
            }
            this->mmCount_++;
            CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_C2V1);

            if ASCEND_IS_AIV {
                this->V0PostProcess(curblock, tBlockNum);
                this->AIV1Process(curblock, tBlockNum);
                this->vec1Count_++;
            }
        }

        if ASCEND_IS_AIV {
            this->invRmsOutQueue_.FreeTensor(this->invRmsUb_);
            this->biasInQue_.FreeTensor(this->biasInUb_);
        } else {
            mmService_.End();
        }
    }

private:
    __aicore__ inline void AICProcessLowApi(uint32_t offsetNd, uint32_t outOffset)
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
            mmService_.CopyOut(this->hMixGm_[outOffset], this->mnConfig_.n);
        }
        this->cubeCount_++;
    }

    MhcPreCubeCompute mmService_;
    TBuf<TPosition::A1> l1Buffer_;
    LocalTensor<float> aL1_;
    LocalTensor<float> bL1_;
};

} // namespace MhcPre

#endif // MHC_PRE_SPLIT_BS_LOW_API_H
