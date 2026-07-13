#ifndef MHC_PRE_SPLIT_BS_LOW_API_H
#define MHC_PRE_SPLIT_BS_LOW_API_H

#include "mhc_pre_kernel_base.h"
#include "mhc_pre_cube_compute_arch35.h"

namespace MhcPre {

template <typename T, typename P, int8_t HAS_RESI>
class MhcPreKernelSplitBSLowApi {
public:
    __aicore__ inline MhcPreKernelSplitBSLowApi() {}

    __aicore__ inline void Init(InitParams initParams)
    {
        pipe_ = initParams.tPipeIn;
        tiling_ = initParams.tilingData;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.x));
        phiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.phi));
        alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.alpha));
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.bias));
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.gamma));
        hinGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.hin));
        hPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_post));
        hResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_res));
        invRmsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.inv_rms));
        hMixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_mix));
        hPreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_pre));

        TBuf<TPosition::A1> l1Buffer;
        pipe_->InitBuffer(l1Buffer, MHC_PRE_LOW_API_L1_ALLOC_SIZE);
        xL1_ = l1Buffer.Get<float>();
        phiL1_ = l1Buffer.Get<float>()[MHC_PRE_LOW_API_L1_BUF_NUM * MHC_PRE_LOW_API_L1_BUF_OFFSET];

        pipe_->InitBufPool(tbufPool0_, tiling_->bufferPool0Size);
        uint32_t mmXSize = tiling_->mL1Size * tiling_->fusionAlign;
        tbufPool0_.InitBuffer(mmXBuf_, mmXSize * sizeof(float));
        mmXLocal_ = mmXBuf_.Get<float>();

        if ASCEND_IS_AIC {
            mmService_.Init();
            CrossCoreSetFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
            CrossCoreSetFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
            CrossCoreSetFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
            CrossCoreSetFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
        } else {
            uint32_t halfM = LowApiCeilDiv(tiling_->mL1Size, 2);
            uint32_t kUbTAlign = LowApiRoundUp<T>(tiling_->kUbSize);
            uint32_t kUbFAlign = LowApiRoundUp<float>(tiling_->kUbSize);
            uint32_t mUbAlign = LowApiAlign(tiling_->mUbSize, MHC_PRE_LOW_API_C0_SIZE);
            uint32_t nd2NzSize = LowApiAlign(tiling_->kUbSize, MHC_PRE_LOW_API_C0_SIZE) * mUbAlign;
            tbufPool0_.InitBuffer(xQue_, 1, tiling_->mUbSize * kUbTAlign * sizeof(T));
            tbufPool0_.InitBuffer(xCastBuf_, tiling_->mUbSize * kUbFAlign * sizeof(float));
            tbufPool0_.InitBuffer(xNd2NzBuf_, nd2NzSize * sizeof(float));
            tbufPool0_.InitBuffer(rmsNormBuf_, 2 * LowApiRoundUp<float>(halfM) * sizeof(float));
            tbufPool0_.InitBuffer(alphaBuf_, LowApiRoundUp<float>(3) * sizeof(float));
            tbufPool0_.InitBuffer(biasBuf_, LowApiRoundUp<float>(tiling_->fusionSize) * sizeof(float));
            tbufPool0_.InitBufPool(tbufPool1_, tiling_->bufferPool1Size);
            rmsNormLocal_ = rmsNormBuf_.Get<float>();
            alphaLocal_ = alphaBuf_.Get<float>();
            biasLocal_ = biasBuf_.Get<float>();
            xCastLocal_ = xCastBuf_.Get<float>();
            xNd2NzLocal_ = xNd2NzBuf_.Get<float>();
            LowApiCopyIn(alphaGm_, alphaLocal_, 1, 3);
            LowApiCopyIn(biasGm_, biasLocal_, 1, tiling_->fusionSize);
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t blockCount = LowApiCeilDiv(static_cast<uint32_t>(tiling_->totalLength), tiling_->mL1Size);
        uint32_t blockIdx = GetBlockIdx();
        uint32_t logicalBlockIdx = blockIdx;
        if ASCEND_IS_AIV {
            logicalBlockIdx = blockIdx / 2;
        }
        if (logicalBlockIdx >= blockCount) {
            return;
        }

        uint32_t rowStart = logicalBlockIdx * tiling_->mL1Size;
        uint32_t mReal = AscendC::Std::min(static_cast<uint64_t>(tiling_->mL1Size), tiling_->totalLength - rowStart);
        uint32_t kEnd = static_cast<uint32_t>(tiling_->nD);
        uint8_t aL1BufferId = 0;

        if ASCEND_IS_AIV {
            Duplicate(rmsNormLocal_, 0.0f, LowApiRoundUp<float>(LowApiCeilDiv(mReal, 2)));
        }

        for (uint32_t kOffset = 0; kOffset < kEnd; kOffset += tiling_->kL1Size) {
            uint32_t kL1Real = AscendC::Std::min(static_cast<uint64_t>(tiling_->kL1Size), static_cast<uint64_t>(kEnd - kOffset));
            if ASCEND_IS_AIC {
                bool isFirstKL1 = (kOffset == 0);
                bool isLastKL1 = (kOffset + tiling_->kL1Size >= kEnd);
                mmService_.CopyInB1Nd2Nz(kEnd, kL1Real, tiling_->fusionSize, phiGm_[kOffset],
                    phiL1_[mmService_.GetBL1BufferId() * MHC_PRE_LOW_API_L1_BUF_OFFSET]);
                CrossCoreWaitFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
                CrossCoreWaitFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
                uint64_t mAlign = LowApiAlign(mReal, AscendC::BLOCK_CUBE);
                uint64_t nAlign = LowApiAlign(static_cast<uint32_t>(tiling_->fusionSize), AscendC::BLOCK_CUBE);
                uint64_t baseK = (256 / AscendC::Std::max(mAlign, nAlign)) * 32;
                mmService_.Process(mReal, tiling_->fusionSize, mReal, baseK, isFirstKL1, isLastKL1,
                    xL1_[aL1BufferId * MHC_PRE_LOW_API_L1_BUF_OFFSET],
                    phiL1_[mmService_.GetBL1BufferId() * MHC_PRE_LOW_API_L1_BUF_OFFSET]);
                if (isLastKL1) {
                    mmService_.CopyOut(mmXLocal_);
                    CrossCoreSetFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_FIX>(SYNC_AIC_AIV_PRE_POST_FLAG);
                    CrossCoreSetFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_FIX>(SYNC_AIC_AIV_PRE_POST_FLAG + FLAG_ID_MAX);
                }
                CrossCoreSetFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
                CrossCoreSetFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
            } else {
                CrossCoreWaitFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
                CopyXChunkToL1(rowStart, mReal, kOffset, kL1Real, aL1BufferId, blockIdx & 1);
                CrossCoreSetFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
            }
            aL1BufferId ^= 1;
        }

        if ASCEND_IS_AIV {
            FinalizeAndPostProcess(rowStart, mReal, blockIdx & 1);
        } else {
            mmService_.End();
        }
    }

private:
    __aicore__ inline void CopyXChunkToL1(uint32_t rowStart, uint32_t mReal, uint32_t kOffset, uint32_t kL1Real,
        uint8_t aL1BufferId, uint32_t aivPart)
    {
        uint32_t firstRows = LowApiCeilDiv(mReal, 2);
        uint32_t rowOffset = (aivPart == 0) ? 0 : firstRows;
        uint32_t currentRows = (aivPart == 0) ? firstRows : (mReal - firstRows);
        uint32_t mL1Align = LowApiAlign(mReal, MHC_PRE_LOW_API_C0_SIZE);
        uint32_t bufferIdx = 0;
        for (uint32_t kInner = 0; kInner < kL1Real; kInner += tiling_->kUbSize) {
            uint32_t kReal = AscendC::Std::min(static_cast<uint64_t>(tiling_->kUbSize), static_cast<uint64_t>(kL1Real - kInner));
            for (uint32_t rowInner = 0; rowInner < currentRows; rowInner += tiling_->mUbSize) {
                uint32_t curRows = AscendC::Std::min(static_cast<uint64_t>(tiling_->mUbSize), static_cast<uint64_t>(currentRows - rowInner));
                xLocal_ = xQue_.template AllocTensor<T>();
                uint32_t globalRow = rowStart + rowOffset + rowInner;
                LowApiCopyIn(xGm_[globalRow * tiling_->nD + kOffset + kInner], xLocal_, curRows, kReal,
                    static_cast<uint32_t>(tiling_->nD) - kReal);
                xLocal_ = xQue_.template DeQue<T>();
                CastAndAccumulateRms(curRows, kReal, kOffset + kInner, rowInner);
                xQue_.template FreeTensor(xLocal_);

                MhcPreLowApiVFTransND2NZ(xNd2NzLocal_, xCastLocal_, curRows, kReal);
                DataCopyParams copyParams;
                copyParams.blockCount = LowApiCeilDiv(kReal, MHC_PRE_LOW_API_C0_SIZE);
                copyParams.blockLen = curRows * MHC_PRE_LOW_API_C0_SIZE * sizeof(float) / MHC_PRE_LOW_API_BLOCK_SIZE;
                copyParams.srcStride = LowApiAlign(curRows, MHC_PRE_LOW_API_C0_SIZE) - curRows;
                copyParams.dstStride = mL1Align - curRows;
                uint32_t dstRowOffset = (rowOffset + rowInner) * MHC_PRE_LOW_API_C0_SIZE;
                uint32_t dstKOffset = (kInner / MHC_PRE_LOW_API_C0_SIZE) * MHC_PRE_LOW_API_C0_SIZE * mL1Align;
                LowApiCopyToL1(xNd2NzLocal_,
                    xL1_[aL1BufferId * MHC_PRE_LOW_API_L1_BUF_OFFSET + dstRowOffset + dstKOffset], copyParams);
                bufferIdx++;
            }
        }
    }

    __aicore__ inline void CastAndAccumulateRms(uint32_t curRows, uint32_t kReal, uint32_t kGlobalOffset,
        uint32_t rowOffsetInPart)
    {
        uint32_t kTAlign = LowApiRoundUp<T>(tiling_->kUbSize);
        uint32_t kFAlign = LowApiRoundUp<float>(tiling_->kUbSize);
        MhcPreLowApiCastToFloat(xCastLocal_, xLocal_, curRows, kReal);
        for (uint32_t row = 0; row < curRows; ++row) {
            float sum = rmsNormLocal_.GetValue(rowOffsetInPart + row);
            for (uint32_t k = 0; k < kReal; ++k) {
                float value = xCastLocal_.GetValue(row * kFAlign + k);
                if (tiling_->hasGamma != 0) {
                    value *= gammaGm_.GetValue((kGlobalOffset + k) % tiling_->D);
                    xCastLocal_.SetValue(row * kFAlign + k, value);
                }
                sum += value * value * tiling_->scaleMean;
            }
            rmsNormLocal_.SetValue(rowOffsetInPart + row, sum);
        }
    }

    __aicore__ inline void FinalizeAndPostProcess(uint32_t rowStart, uint32_t mReal, uint32_t aivPart)
    {
        uint32_t firstRows = LowApiCeilDiv(mReal, 2);
        uint32_t rowOffset = (aivPart == 0) ? 0 : firstRows;
        uint32_t currentRows = (aivPart == 0) ? firstRows : (mReal - firstRows);
        if (currentRows == 0) {
            return;
        }
        Adds(rmsNormLocal_, rmsNormLocal_, tiling_->normEps, currentRows);
        Sqrt(rmsNormLocal_, rmsNormLocal_, currentRows);
        LocalTensor<float> oneLocal = rmsNormLocal_[LowApiRoundUp<float>(currentRows)];
        Duplicate(oneLocal, 1.0f, currentRows);
        Div(rmsNormLocal_, oneLocal, rmsNormLocal_, currentRows);
        LowApiCopyOut(rmsNormLocal_, invRmsGm_[rowStart + rowOffset], 1, currentRows);

        CrossCoreWaitFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_V>(SYNC_AIC_AIV_PRE_POST_FLAG + aivPart * FLAG_ID_MAX);
        tbufPool1_.Reset();
        tbufPool1_.InitBuffer(preBuf_, 2 * currentRows * tiling_->nAlign * sizeof(float));
        tbufPool1_.InitBuffer(postBuf_, 2 * currentRows * tiling_->nAlign * sizeof(float));
        if constexpr (HAS_RESI == MHC_PRE_HAS_RESI) {
            tbufPool1_.InitBuffer(resBuf_, 2 * currentRows * LowApiRoundUp<float>(MHC_PRE_LOW_API_N4 * MHC_PRE_LOW_API_N4) * sizeof(float));
        }
        tbufPool1_.InitBuffer(xQuePost_, 1, tiling_->rowInnerFactor * MHC_PRE_LOW_API_N4 * LowApiRoundUp<T>(tiling_->dFactor) * sizeof(T));
        tbufPool1_.InitBuffer(yQue_, 1, tiling_->rowInnerFactor * LowApiRoundUp<T>(tiling_->dFactor) * sizeof(T));
        preLocal_ = preBuf_.Get<float>();
        postLocal_ = postBuf_.Get<float>();
        if constexpr (HAS_RESI == MHC_PRE_HAS_RESI) {
            resLocal_ = resBuf_.Get<float>();
        }

        LocalTensor<float> mmRowsLocal = mmXLocal_[rowOffset * tiling_->fusionAlign];
        if (tiling_->outFlag != 0) {
            LowApiCopyOut(mmRowsLocal, hMixGm_[(rowStart + rowOffset) * tiling_->fusionSize], currentRows,
                tiling_->fusionSize, 0, tiling_->fusionAlign - tiling_->fusionSize);
        }
        MhcPreLowApiProcessPreN4(preLocal_, mmRowsLocal, rmsNormLocal_, alphaLocal_, biasLocal_, currentRows,
            tiling_->fusionAlign, tiling_->hcEps);
        if (tiling_->outFlag != 0) {
            LowApiCopyOut(preLocal_, hPreGm_[(rowStart + rowOffset) * MHC_PRE_LOW_API_N4], currentRows,
                MHC_PRE_LOW_API_N4, 0, tiling_->nAlign - MHC_PRE_LOW_API_N4);
        }
        MhcPreLowApiProcessSegmentN4(postLocal_, mmRowsLocal, rmsNormLocal_, alphaLocal_, biasLocal_, currentRows,
            tiling_->fusionAlign, MHC_PRE_LOW_API_POST_OFFSET_N4, MHC_PRE_LOW_API_N4, 1,
            MHC_PRE_LOW_API_POST_OFFSET_N4, true, true, 0.0f);
        LowApiCopyOut(postLocal_, hPostGm_[(rowStart + rowOffset) * MHC_PRE_LOW_API_N4], currentRows,
            MHC_PRE_LOW_API_N4, 0, tiling_->nAlign - MHC_PRE_LOW_API_N4);

        if constexpr (HAS_RESI == MHC_PRE_HAS_RESI) {
            MhcPreLowApiProcessSegmentN4(resLocal_, mmRowsLocal, rmsNormLocal_, alphaLocal_, biasLocal_, currentRows,
                tiling_->fusionAlign, MHC_PRE_LOW_API_RES_OFFSET_N4, MHC_PRE_LOW_API_N4 * MHC_PRE_LOW_API_N4, 2,
                MHC_PRE_LOW_API_RES_OFFSET_N4, false, false, 0.0f);
            LowApiCopyOut(resLocal_, hResGm_[(rowStart + rowOffset) * MHC_PRE_LOW_API_N4 * MHC_PRE_LOW_API_N4],
                currentRows, MHC_PRE_LOW_API_N4 * MHC_PRE_LOW_API_N4);
        }
        ProcessHin(rowStart + rowOffset, currentRows);
    }

    __aicore__ inline void ProcessHin(uint32_t globalRowStart, uint32_t rowCount)
    {
        for (uint32_t dOffset = 0; dOffset < tiling_->D; dOffset += tiling_->dFactor) {
            uint32_t dReal = AscendC::Std::min(static_cast<uint64_t>(tiling_->dFactor), tiling_->D - dOffset);
            for (uint32_t row = 0; row < rowCount; row += tiling_->rowInnerFactor) {
                uint32_t curRows = AscendC::Std::min(static_cast<uint64_t>(tiling_->rowInnerFactor), static_cast<uint64_t>(rowCount - row));
                LocalTensor<T> xPost = xQuePost_.template AllocTensor<T>();
                LowApiCopyIn(xGm_[(globalRowStart + row) * tiling_->nD + dOffset], xPost,
                    curRows * MHC_PRE_LOW_API_N4, dReal, static_cast<uint32_t>(tiling_->D) - dReal);
                xPost = xQuePost_.template DeQue<T>();
                LocalTensor<T> yPost = yQue_.template AllocTensor<T>();
                MhcPreLowApiProcessHinN4(yPost, xPost, preLocal_[row * tiling_->nAlign], curRows, dReal);
                yQue_.template EnQue<T>(yPost);
                yPost = yQue_.template DeQue<T>();
                LowApiCopyOut(yPost, hinGm_[(globalRowStart + row) * tiling_->D + dOffset], curRows, dReal,
                    static_cast<uint32_t>(tiling_->D) - dReal, LowApiRoundUp<T>(dReal) - dReal);
                yQue_.template FreeTensor(yPost);
                xQuePost_.template FreeTensor(xPost);
            }
        }
    }

private:
    TPipe *pipe_{nullptr};
    const MhcPreTilingData *tiling_{nullptr};
    GlobalTensor<T> xGm_;
    GlobalTensor<P> phiGm_;
    GlobalTensor<P> alphaGm_;
    GlobalTensor<P> biasGm_;
    GlobalTensor<P> gammaGm_;
    GlobalTensor<T> hinGm_;
    GlobalTensor<P> hPostGm_;
    GlobalTensor<P> hResGm_;
    GlobalTensor<P> invRmsGm_;
    GlobalTensor<P> hMixGm_;
    GlobalTensor<P> hPreGm_;

    MhcPreCubeCompute mmService_;
    LocalTensor<float> xL1_;
    LocalTensor<float> phiL1_;
    LocalTensor<float> mmXLocal_;
    LocalTensor<float> rmsNormLocal_;
    LocalTensor<float> alphaLocal_;
    LocalTensor<float> biasLocal_;
    LocalTensor<float> xCastLocal_;
    LocalTensor<float> xNd2NzLocal_;
    LocalTensor<float> preLocal_;
    LocalTensor<float> postLocal_;
    LocalTensor<float> resLocal_;
    LocalTensor<T> xLocal_;

    TQue<QuePosition::VECIN, 1> xQue_;
    TQue<QuePosition::VECIN, 1> xQuePost_;
    TQue<QuePosition::VECOUT, 1> yQue_;
    TBuf<QuePosition::VECCALC> mmXBuf_;
    TBuf<QuePosition::VECCALC> rmsNormBuf_;
    TBuf<QuePosition::VECCALC> alphaBuf_;
    TBuf<QuePosition::VECCALC> biasBuf_;
    TBuf<QuePosition::VECCALC> xCastBuf_;
    TBuf<QuePosition::VECCALC> xNd2NzBuf_;
    TBuf<QuePosition::VECCALC> preBuf_;
    TBuf<QuePosition::VECCALC> postBuf_;
    TBuf<QuePosition::VECCALC> resBuf_;
    TBufPool<QuePosition::VECCALC, 16> tbufPool0_;
    TBufPool<QuePosition::VECCALC, 16> tbufPool1_;

    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;
    static constexpr uint64_t SYNC_AIC_AIV_PRE_POST_FLAG = 10;
    static constexpr uint64_t FLAG_ID_MAX = 16;
};

} // namespace MhcPre

#endif // MHC_PRE_SPLIT_BS_LOW_API_H
