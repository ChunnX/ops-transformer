#ifndef MHC_PRE_M_K_SPLIT_CORE_ARCH35_H
#define MHC_PRE_M_K_SPLIT_CORE_ARCH35_H

#include "mhc_pre_cube_compute_arch35.h"
#include "mhc_pre_split_bs.h"

namespace MhcPre {

__aicore__ inline void MhcPreMkVFTransND2NZ(const LocalTensor<float>& dstLocal,
    const LocalTensor<float>& srcLocal, uint16_t rowCount, uint16_t columnCount)
{
    __local_mem__ float* dst = (__local_mem__ float*)dstLocal.GetPhyAddr();
    __local_mem__ float* src = (__local_mem__ float*)srcLocal.GetPhyAddr();
    uint16_t rowAlign = LowApiAlign(rowCount, MHC_PRE_LOW_API_C0_SIZE);
    uint16_t dataBlockElements = MHC_PRE_LOW_API_BLOCK_SIZE / sizeof(float);
    uint16_t columnAlign = LowApiRoundUp<float>(columnCount);
    uint16_t mainRowBlocks = rowCount / MHC_PRE_LOW_API_C0_SIZE;
    uint16_t tailRows = rowCount % MHC_PRE_LOW_API_C0_SIZE;
    uint16_t columnLoops = columnAlign / dataBlockElements;
    uint32_t srcBlockStride = columnAlign / dataBlockElements;

    if (tailRows == 0) {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> data;
            AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float>();
            for (uint16_t rowBlock = 0; rowBlock < mainRowBlocks; ++rowBlock) {
                for (uint16_t columnBlock = 0; columnBlock < columnLoops; ++columnBlock) {
                    AscendC::MicroAPI::DataCopy<float,
                        AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(data,
                        src + rowBlock * MHC_PRE_LOW_API_C0_SIZE * columnAlign +
                            columnBlock * dataBlockElements,
                        srcBlockStride, mask);
                    AscendC::MicroAPI::DataCopy(
                        dst + rowBlock * MHC_PRE_LOW_API_C0_SIZE * dataBlockElements +
                            columnBlock * rowAlign * dataBlockElements,
                        data, mask);
                }
            }
        }
        return;
    }

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> data;
        AscendC::MicroAPI::MaskReg fullMask = AscendC::MicroAPI::CreateMask<float>();
        uint32_t tailElements = tailRows * MHC_PRE_LOW_API_C0_SIZE;
        AscendC::MicroAPI::MaskReg tailMask =
            AscendC::MicroAPI::UpdateMask<float>(tailElements);
        for (uint16_t rowBlock = 0; rowBlock < mainRowBlocks; ++rowBlock) {
            for (uint16_t columnBlock = 0; columnBlock < columnLoops; ++columnBlock) {
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(data,
                    src + rowBlock * MHC_PRE_LOW_API_C0_SIZE * columnAlign +
                        columnBlock * dataBlockElements,
                    srcBlockStride, fullMask);
                AscendC::MicroAPI::DataCopy(
                    dst + rowBlock * MHC_PRE_LOW_API_C0_SIZE * dataBlockElements +
                        columnBlock * rowAlign * dataBlockElements,
                    data, fullMask);
            }
        }
        src += mainRowBlocks * MHC_PRE_LOW_API_C0_SIZE * columnAlign;
        dst += mainRowBlocks * MHC_PRE_LOW_API_C0_SIZE * dataBlockElements;
        for (uint16_t columnBlock = 0; columnBlock < columnLoops; ++columnBlock) {
            AscendC::MicroAPI::DataCopy<float,
                AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(data,
                src + columnBlock * dataBlockElements, srcBlockStride, tailMask);
            AscendC::MicroAPI::DataCopy(dst + columnBlock * rowAlign * dataBlockElements,
                data, tailMask);
        }
    }
}

__aicore__ inline void MhcPreMkVFReducePartials(const LocalTensor<float>& mmOutLocal,
    const LocalTensor<float>& invRmsOutLocal, const LocalTensor<float>& mmPartialLocal,
    const LocalTensor<float>& rmsPartialLocal, uint16_t groupK, uint16_t rowCount,
    uint16_t fusionSize, uint16_t rmsStride, float scaleMean, float normEps)
{
    __local_mem__ float* mmOut = (__local_mem__ float*)mmOutLocal.GetPhyAddr();
    __local_mem__ float* invRmsOut = (__local_mem__ float*)invRmsOutLocal.GetPhyAddr();
    __local_mem__ float* mmPartial = (__local_mem__ float*)mmPartialLocal.GetPhyAddr();
    __local_mem__ float* rmsPartial = (__local_mem__ float*)rmsPartialLocal.GetPhyAddr();
    uint16_t fourLoopCount = groupK / 4U;
    uint16_t tailLoopCount = groupK % 4U;
    uint32_t mmGroupStride = static_cast<uint32_t>(rowCount) * fusionSize;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> rms0;
        AscendC::MicroAPI::RegTensor<float> rms1;
        AscendC::MicroAPI::RegTensor<float> rms2;
        AscendC::MicroAPI::RegTensor<float> rms3;
        AscendC::MicroAPI::RegTensor<float> mm0;
        AscendC::MicroAPI::RegTensor<float> mm1;
        AscendC::MicroAPI::RegTensor<float> mm2;
        AscendC::MicroAPI::RegTensor<float> mm3;
        AscendC::MicroAPI::RegTensor<float> sumRms0;
        AscendC::MicroAPI::RegTensor<float> sumRms1;
        AscendC::MicroAPI::RegTensor<float> sumRms2;
        AscendC::MicroAPI::RegTensor<float> sumRms3;
        AscendC::MicroAPI::RegTensor<float> sumMm0;
        AscendC::MicroAPI::RegTensor<float> sumMm1;
        AscendC::MicroAPI::RegTensor<float> sumMm2;
        AscendC::MicroAPI::RegTensor<float> sumMm3;
        AscendC::MicroAPI::RegTensor<float> one;
        uint32_t rmsMaskSize = 1U;
        uint32_t mmMaskSize = fusionSize;
        AscendC::MicroAPI::MaskReg rmsMask = AscendC::MicroAPI::UpdateMask<float>(rmsMaskSize);
        AscendC::MicroAPI::MaskReg mmMask = AscendC::MicroAPI::UpdateMask<float>(mmMaskSize);
        AscendC::MicroAPI::Duplicate(one, 1.0f, rmsMask);

        for (uint16_t row = 0; row < rowCount; ++row) {
            AscendC::MicroAPI::Duplicate(sumRms0, 0.0f, rmsMask);
            AscendC::MicroAPI::Duplicate(sumRms1, 0.0f, rmsMask);
            AscendC::MicroAPI::Duplicate(sumRms2, 0.0f, rmsMask);
            AscendC::MicroAPI::Duplicate(sumRms3, 0.0f, rmsMask);
            AscendC::MicroAPI::Duplicate(sumMm0, 0.0f, mmMask);
            AscendC::MicroAPI::Duplicate(sumMm1, 0.0f, mmMask);
            AscendC::MicroAPI::Duplicate(sumMm2, 0.0f, mmMask);
            AscendC::MicroAPI::Duplicate(sumMm3, 0.0f, mmMask);

            for (uint16_t loop = 0; loop < fourLoopCount; ++loop) {
                uint16_t kBase = loop * 4U;
                MhcPreLowApiLoadBroadcast(rms0, rmsPartial, rmsMask,
                    static_cast<uint32_t>(kBase) * rmsStride + row);
                MhcPreLowApiLoadBroadcast(rms1, rmsPartial, rmsMask,
                    static_cast<uint32_t>(kBase + 1U) * rmsStride + row);
                MhcPreLowApiLoadBroadcast(rms2, rmsPartial, rmsMask,
                    static_cast<uint32_t>(kBase + 2U) * rmsStride + row);
                MhcPreLowApiLoadBroadcast(rms3, rmsPartial, rmsMask,
                    static_cast<uint32_t>(kBase + 3U) * rmsStride + row);
                AscendC::MicroAPI::Add(sumRms0, sumRms0, rms0, rmsMask);
                AscendC::MicroAPI::Add(sumRms1, sumRms1, rms1, rmsMask);
                AscendC::MicroAPI::Add(sumRms2, sumRms2, rms2, rmsMask);
                AscendC::MicroAPI::Add(sumRms3, sumRms3, rms3, rmsMask);

                AscendC::MicroAPI::Load<float>(mm0,
                    mmPartial + static_cast<uint32_t>(kBase) * mmGroupStride + row * fusionSize);
                AscendC::MicroAPI::Load<float>(mm1,
                    mmPartial + static_cast<uint32_t>(kBase + 1U) * mmGroupStride + row * fusionSize);
                AscendC::MicroAPI::Load<float>(mm2,
                    mmPartial + static_cast<uint32_t>(kBase + 2U) * mmGroupStride + row * fusionSize);
                AscendC::MicroAPI::Load<float>(mm3,
                    mmPartial + static_cast<uint32_t>(kBase + 3U) * mmGroupStride + row * fusionSize);
                AscendC::MicroAPI::Add(sumMm0, sumMm0, mm0, mmMask);
                AscendC::MicroAPI::Add(sumMm1, sumMm1, mm1, mmMask);
                AscendC::MicroAPI::Add(sumMm2, sumMm2, mm2, mmMask);
                AscendC::MicroAPI::Add(sumMm3, sumMm3, mm3, mmMask);
            }

            for (uint16_t tail = 0; tail < tailLoopCount; ++tail) {
                uint16_t kIndex = fourLoopCount * 4U + tail;
                MhcPreLowApiLoadBroadcast(rms0, rmsPartial, rmsMask,
                    static_cast<uint32_t>(kIndex) * rmsStride + row);
                AscendC::MicroAPI::Add(sumRms0, sumRms0, rms0, rmsMask);
                AscendC::MicroAPI::Load<float>(mm0,
                    mmPartial + static_cast<uint32_t>(kIndex) * mmGroupStride + row * fusionSize);
                AscendC::MicroAPI::Add(sumMm0, sumMm0, mm0, mmMask);
            }

            AscendC::MicroAPI::Add(sumRms0, sumRms0, sumRms3, rmsMask);
            AscendC::MicroAPI::Add(sumRms1, sumRms1, sumRms2, rmsMask);
            AscendC::MicroAPI::Add(sumRms0, sumRms0, sumRms1, rmsMask);
            AscendC::MicroAPI::Muls(sumRms0, sumRms0, scaleMean, rmsMask);
            AscendC::MicroAPI::Adds(sumRms0, sumRms0, normEps, rmsMask);
            AscendC::MicroAPI::Sqrt(sumRms0, sumRms0, rmsMask);
            AscendC::MicroAPI::Div(sumRms0, one, sumRms0, rmsMask);

            AscendC::MicroAPI::Add(sumMm0, sumMm0, sumMm3, mmMask);
            AscendC::MicroAPI::Add(sumMm1, sumMm1, sumMm2, mmMask);
            AscendC::MicroAPI::Add(sumMm0, sumMm0, sumMm1, mmMask);
            AscendC::MicroAPI::Store<float>(invRmsOut + row, sumRms0, 1U);
            AscendC::MicroAPI::Store<float>(mmOut + row * fusionSize, sumMm0, fusionSize);
        }
    }
}

template <class T, class P, int8_t HAS_RESI>
class MhcPreKernelMKPart1 : public MhcPreKernelSplitBS<T, P, HAS_RESI> {
public:
    using Parent = MhcPreKernelSplitBS<T, P, HAS_RESI>;

    __aicore__ inline explicit MhcPreKernelMKPart1(MT &matmul) : Parent(matmul) {}

    __aicore__ inline void Init(InitParams initParams)
    {
        this->BindGlobalTensors(initParams);
        this->InitFromTilingData(initParams.tilingData);
        this->InitMNConfig();
        this->pipe_ = initParams.tPipeIn;

        partialMmGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(
            initParams.workspace + this->tiling_->mkWorkspaceMmOffset));
        partialRmsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(
            initParams.workspace + this->tiling_->mkWorkspaceRmsOffset));

        this->pipe_->InitBuffer(l1Buffer_, MHC_PRE_LOW_API_L1_ALLOC_SIZE);
        aL1_ = l1Buffer_.Get<P>();
        bL1_ = aL1_[MHC_PRE_LOW_API_L1_BUF_NUM * MHC_PRE_LOW_API_L1_BUF_OFFSET];

        if ASCEND_IS_AIC {
            mmService_.Init();
            CrossCoreSetFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
            CrossCoreSetFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
            CrossCoreSetFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
            CrossCoreSetFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
        }
        if ASCEND_IS_AIV {
            uint32_t rowCapacity = (this->tiling_->mL1Size + 1U) / 2U;
            uint32_t kCapacity = LowApiRoundUp<T>(this->tiling_->kUbSize);
            this->pipe_->InitBuffer(this->xInQueue_, 1, rowCapacity * kCapacity * sizeof(T));
            this->pipe_->InitBuffer(this->outQueue_, 1,
                rowCapacity * LowApiRoundUp<P>(this->tiling_->kUbSize) * sizeof(P));
            this->pipe_->InitBuffer(this->invRmsOutQueue_, 1,
                LowApiRoundUp<P>(rowCapacity) * sizeof(P));
            if (this->hasGamma_) {
                this->pipe_->InitBuffer(this->gammaInQueue_, 1,
                    LowApiRoundUp<P>(this->tiling_->kUbSize) * sizeof(P));
            }
            this->pipe_->InitBuffer(nd2NzBuf_,
                LowApiAlign(rowCapacity, MHC_PRE_LOW_API_C0_SIZE) *
                LowApiRoundUp<P>(this->tiling_->kUbSize) * sizeof(P) * 2U);
            nd2NzLocal_ = nd2NzBuf_.Get<P>();
            this->invRmsUb_ = this->invRmsOutQueue_.template AllocTensor<P>();
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t aicIndex = GetBlockIdx();
        if ASCEND_IS_AIV {
            aicIndex /= 2U;
        }
        uint32_t usedCoreNum = this->tiling_->cubeBlockDimM * this->tiling_->cubeBlockDimK;
        if (aicIndex < usedCoreNum) {
            uint32_t mIndex = aicIndex / this->tiling_->cubeBlockDimK;
            uint32_t kIndex = aicIndex % this->tiling_->cubeBlockDimK;
            uint32_t mStart = mIndex * this->tiling_->mL1Size;
            uint32_t mReal = AscendC::Std::min(
                static_cast<uint64_t>(this->tiling_->mL1Size), this->totalLength_ - mStart);
            uint32_t kStart = kIndex * this->tiling_->multCoreSplitKSize;
            uint32_t kEnd = AscendC::Std::min(
                static_cast<uint64_t>(kStart + this->tiling_->multCoreSplitKSize),
                static_cast<uint64_t>(this->matrixInfo_.nD));
            if ASCEND_IS_AIC {
                ProcessAic(mStart, mReal, kIndex, kStart, kEnd);
            } else {
                ProcessAiv(mStart, mReal, kIndex, kStart, kEnd);
            }
        }
        SyncAll<false>();
        if ASCEND_IS_AIC {
            mmService_.End();
        }
    }

private:
    __aicore__ inline void ProcessAic(uint32_t mStart, uint32_t mReal, uint32_t kIndex,
        uint32_t kStart, uint32_t kEnd)
    {
        uint8_t aL1BufferId = 0;
        for (uint32_t kOffset = kStart; kOffset < kEnd; kOffset += this->tiling_->kL1Size) {
            uint32_t currentK = AscendC::Std::min(
                static_cast<uint64_t>(this->tiling_->kL1Size), static_cast<uint64_t>(kEnd - kOffset));
            uint8_t bL1BufferId = mmService_.GetBL1BufferId();
            LocalTensor<P> currentAL1 = aL1_[aL1BufferId * MHC_PRE_LOW_API_L1_BUF_OFFSET];
            LocalTensor<P> currentBL1 = bL1_[bL1BufferId * MHC_PRE_LOW_API_L1_BUF_OFFSET];
            mmService_.CopyInB1Nd2Nz(this->matrixInfo_.nD, currentK, this->matrixInfo_.fusionSize,
                this->phiGm_[kOffset], currentBL1);
            CrossCoreWaitFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
            CrossCoreWaitFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);

            bool isFirst = kOffset == kStart;
            bool isLast = kOffset + currentK >= kEnd;
            uint64_t mAlign = LowApiAlign(mReal, AscendC::BLOCK_CUBE);
            uint64_t nAlign = LowApiAlign(this->matrixInfo_.fusionSize, AscendC::BLOCK_CUBE);
            uint64_t baseK = (256U / AscendC::Std::max(mAlign, nAlign)) * 32U;
            mmService_.Process(mReal, this->matrixInfo_.fusionSize, mReal, baseK,
                isFirst, isLast, currentAL1, currentBL1);
            if (isLast) {
                uint64_t outputOffset = static_cast<uint64_t>(kIndex) * this->totalLength_ *
                    this->matrixInfo_.fusionSize + static_cast<uint64_t>(mStart) * this->matrixInfo_.fusionSize;
                mmService_.CopyOut(partialMmGm_[outputOffset], this->matrixInfo_.fusionSize);
            }
            CrossCoreSetFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
            CrossCoreSetFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
            aL1BufferId ^= 1U;
        }
    }

    __aicore__ inline void ProcessAiv(uint32_t mStart, uint32_t mReal, uint32_t kIndex,
        uint32_t kStart, uint32_t kEnd)
    {

        uint32_t firstRows = mReal / 2U;
        uint32_t rowStart = GetSubBlockIdx() == 0 ? 0U : firstRows;
        uint32_t rowCount = GetSubBlockIdx() == 0 ? firstRows : mReal - firstRows;
        uint8_t aL1BufferId = 0;
        this->globalOffsetM_ = mStart;
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        uint32_t nd2NzBufferElements = LowApiAlign(rowCount, MHC_PRE_LOW_API_C0_SIZE) *
            LowApiRoundUp<P>(this->tiling_->kUbSize);
        uint32_t bufferIndex = 0;
        for (uint32_t kOffset = kStart; kOffset < kEnd; kOffset += this->tiling_->kL1Size) {
            uint32_t currentK = AscendC::Std::min(
                static_cast<uint64_t>(this->tiling_->kL1Size), static_cast<uint64_t>(kEnd - kOffset));
            CrossCoreWaitFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);

            for (uint32_t kUbOffset = 0; kUbOffset < currentK; kUbOffset += this->tiling_->kUbSize) {
                uint32_t currentKUb = AscendC::Std::min(
                    static_cast<uint64_t>(this->tiling_->kUbSize),
                    static_cast<uint64_t>(currentK - kUbOffset));
                this->xLocal_ = this->xInQueue_.template AllocTensor<T>();
                this->DataCopyX(rowCount, currentKUb, rowStart, kOffset + kUbOffset);
                this->xLocal_ = this->xInQueue_.template DeQue<T>();
                LocalTensor<P> xFloatLocal = this->outQueue_.template AllocTensor<P>();
                bool initializeRms = kOffset == kStart && kUbOffset == 0;
                if (this->hasGamma_) {
                    this->gammaUb_ = this->gammaInQueue_.template AllocTensor<P>();
                    this->DataCopyGamma(currentKUb, kOffset + kUbOffset);
                    this->gammaUb_ = this->gammaInQueue_.template DeQue<P>();
                    if (initializeRms) {
                        this->template VFDoV0ProcessXIn<true, true>((__ubuf__ P *)xFloatLocal.GetPhyAddr(),
                            (__ubuf__ P *)this->invRmsUb_.GetPhyAddr(),
                            (__ubuf__ T *)this->xLocal_.GetPhyAddr(),
                            (__ubuf__ P *)this->gammaUb_.GetPhyAddr(), rowCount, currentKUb);
                    } else {
                        this->template VFDoV0ProcessXIn<true, false>((__ubuf__ P *)xFloatLocal.GetPhyAddr(),
                            (__ubuf__ P *)this->invRmsUb_.GetPhyAddr(),
                            (__ubuf__ T *)this->xLocal_.GetPhyAddr(),
                            (__ubuf__ P *)this->gammaUb_.GetPhyAddr(), rowCount, currentKUb);
                    }
                    this->gammaInQueue_.FreeTensor(this->gammaUb_);
                } else if (initializeRms) {
                    this->template VFDoV0ProcessXIn<false, true>((__ubuf__ P *)xFloatLocal.GetPhyAddr(),
                        (__ubuf__ P *)this->invRmsUb_.GetPhyAddr(),
                        (__ubuf__ T *)this->xLocal_.GetPhyAddr(), nullptr, rowCount, currentKUb);
                } else {
                    this->template VFDoV0ProcessXIn<false, false>((__ubuf__ P *)xFloatLocal.GetPhyAddr(),
                        (__ubuf__ P *)this->invRmsUb_.GetPhyAddr(),
                        (__ubuf__ T *)this->xLocal_.GetPhyAddr(), nullptr, rowCount, currentKUb);
                }
                this->xInQueue_.FreeTensor(this->xLocal_);

                event_t eventId = static_cast<event_t>(bufferIndex & 1U);
                WaitFlag<HardEvent::MTE3_V>(eventId);
                LocalTensor<P> currentNd2Nz = nd2NzLocal_[(bufferIndex & 1U) * nd2NzBufferElements];
                MhcPreMkVFTransND2NZ(currentNd2Nz, xFloatLocal, rowCount, currentKUb);
                this->outQueue_.FreeTensor(xFloatLocal);
                SetFlag<HardEvent::V_MTE3>(eventId);
                WaitFlag<HardEvent::V_MTE3>(eventId);
                DataCopyParams copyParams;
                copyParams.blockCount = LowApiCeilDiv(currentKUb, MHC_PRE_LOW_API_C0_SIZE);
                copyParams.blockLen =
                    rowCount * MHC_PRE_LOW_API_C0_SIZE * sizeof(P) / MHC_PRE_LOW_API_BLOCK_SIZE;
                copyParams.srcStride = LowApiAlign(rowCount, MHC_PRE_LOW_API_C0_SIZE) - rowCount;
                copyParams.dstStride = LowApiAlign(mReal, AscendC::BLOCK_CUBE) - rowCount;
                uint32_t dstRowOffset = GetSubBlockIdx() == 0 ? 0U :
                    firstRows * (MHC_PRE_LOW_API_BLOCK_SIZE / sizeof(P));
                LowApiCopyToL1(currentNd2Nz,
                    aL1_[aL1BufferId * MHC_PRE_LOW_API_L1_BUF_OFFSET +
                        kUbOffset * LowApiAlign(mReal, AscendC::BLOCK_CUBE) + dstRowOffset], copyParams);
                SetFlag<HardEvent::MTE3_V>(eventId);
                ++bufferIndex;
            }
            CrossCoreSetFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
            aL1BufferId ^= 1U;
        }
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = rowCount * sizeof(P);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        uint64_t rmsOffset = static_cast<uint64_t>(kIndex) * this->totalLength_ + mStart + rowStart;
        DataCopyPad(partialRmsGm_[rmsOffset], this->invRmsUb_, copyParams);
        this->invRmsOutQueue_.FreeTensor(this->invRmsUb_);
    }

private:
    GlobalTensor<P> partialMmGm_;
    GlobalTensor<P> partialRmsGm_;
    TBuf<TPosition::A1> l1Buffer_;
    TBuf<TPosition::VECCALC> nd2NzBuf_;
    LocalTensor<P> aL1_;
    LocalTensor<P> bL1_;
    LocalTensor<P> nd2NzLocal_;
    MhcPreCubeCompute mmService_;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;
    static constexpr uint64_t FLAG_ID_MAX = 16;
};

template <class T, class P, int8_t HAS_RESI>
class MhcPreKernelMKPart2 : public MhcPreKernelSplitBS<T, P, HAS_RESI> {
public:
    using Parent = MhcPreKernelSplitBS<T, P, HAS_RESI>;

    __aicore__ inline explicit MhcPreKernelMKPart2(MT &matmul) : Parent(matmul) {}

    __aicore__ inline void Init(InitParams initParams)
    {
        this->BindGlobalTensors(initParams);
        this->InitFromTilingData(initParams.tilingData);
        this->InitMNConfig();
        this->pipe_ = initParams.tPipeIn;
        partialMmGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(
            initParams.workspace + this->tiling_->mkWorkspaceMmOffset));
        partialRmsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(
            initParams.workspace + this->tiling_->mkWorkspaceRmsOffset));
        if (this->outFlag_) {
            this->hMixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_mix));
        } else {
            this->hMixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(
                initParams.workspace + this->tiling_->mkWorkspaceFinalOffset));
        }
        this->InitUbBuffers();

        if ASCEND_IS_AIV {
            uint32_t aicIndex = GetBlockIdx() / 2U;
            selected_ = aicIndex < this->tiling_->cubeBlockDimM * this->tiling_->cubeBlockDimK &&
                        aicIndex % this->tiling_->cubeBlockDimK == 0;
            if (selected_) {
                uint32_t mIndex = aicIndex / this->tiling_->cubeBlockDimK;
                this->coreIdx_ = mIndex;
                this->subBlockIdx_ = GetSubBlockIdx();
                this->globalOffsetM_ = static_cast<uint64_t>(mIndex) * this->tiling_->mL1Size;
                this->curSingleT_ = AscendC::Std::min(
                    static_cast<uint64_t>(this->tiling_->mL1Size), this->totalLength_ - this->globalOffsetM_);
                this->mnConfig_.curSingleCoreM = this->curSingleT_;
                this->VectorComputeOffset();
                this->AIVPreLoad();
            }
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }
        if (!selected_ || this->vectorOffset_.singleCoreM == 0) {
            return;
        }
        ReducePartials();
        PostProcess();
        this->invRmsOutQueue_.FreeTensor(this->invRmsUb_);
        this->biasInQue_.FreeTensor(this->biasInUb_);
    }

private:
    __aicore__ inline void ReducePartials()
    {
        for (uint32_t offsetT = this->vectorOffset_.offsetMStart;
             offsetT < this->vectorOffset_.offsetMEnd; offsetT += this->V1_BASE_T) {
            uint32_t lenT = AscendC::Std::min(
                static_cast<uint64_t>(this->V1_BASE_T), this->vectorOffset_.offsetMEnd - offsetT);
            uint32_t groupK = this->tiling_->cubeBlockDimK;
            uint32_t fusionSize = this->matrixInfo_.fusionSize;
            uint32_t rmsStride = LowApiRoundUp<P>(lenT);
            uint32_t mmPartialElements = groupK * lenT * fusionSize;
            LocalTensor<P> partialLocal = this->xInQueue_.template AllocTensor<P>();
            LocalTensor<P> rmsPartialLocal = partialLocal[mmPartialElements];

            DataCopyExtParams mmCopyParams;
            mmCopyParams.blockCount = groupK;
            mmCopyParams.blockLen = lenT * fusionSize * sizeof(P);
            mmCopyParams.srcStride = (this->totalLength_ - lenT) * fusionSize * sizeof(P);
            mmCopyParams.dstStride = 0;
            DataCopyPadExtParams<P> mmPadParams{true, 0, 0, 0};
            uint64_t mmSourceOffset = (this->globalOffsetM_ + offsetT) * fusionSize;
            DataCopyPad(partialLocal, partialMmGm_[mmSourceOffset], mmCopyParams, mmPadParams);

            DataCopyExtParams rmsCopyParams;
            rmsCopyParams.blockCount = groupK;
            rmsCopyParams.blockLen = lenT * sizeof(P);
            rmsCopyParams.srcStride = (this->totalLength_ - lenT) * sizeof(P);
            rmsCopyParams.dstStride = 0;
            DataCopyPadExtParams<P> rmsPadParams{true, 0,
                static_cast<uint8_t>(rmsStride - lenT), 0};
            uint64_t rmsSourceOffset = this->globalOffsetM_ + offsetT;
            DataCopyPad(rmsPartialLocal, partialRmsGm_[rmsSourceOffset], rmsCopyParams, rmsPadParams);

            this->xInQueue_.template EnQue<P>(partialLocal);
            partialLocal = this->xInQueue_.template DeQue<P>();
            rmsPartialLocal = partialLocal[mmPartialElements];
            LocalTensor<P> accumulator = this->outQueue_.template AllocTensor<P>();
            uint32_t invRmsOffset = offsetT - this->vectorOffset_.offsetMStart;
            MhcPreMkVFReducePartials(accumulator, this->invRmsUb_[invRmsOffset], partialLocal,
                rmsPartialLocal, static_cast<uint16_t>(groupK), static_cast<uint16_t>(lenT),
                static_cast<uint16_t>(fusionSize), static_cast<uint16_t>(rmsStride),
                this->scaleMean_, this->matrixInfo_.normEps);
            this->xInQueue_.FreeTensor(partialLocal);
            this->outQueue_.template EnQue<P>(accumulator);
            accumulator = this->outQueue_.template DeQue<P>();
            DataCopyExtParams outputParams;
            outputParams.blockCount = 1;
            outputParams.blockLen = lenT * fusionSize * sizeof(P);
            outputParams.srcStride = 0;
            outputParams.dstStride = 0;
            uint64_t outputOffset = (this->globalOffsetM_ + offsetT) * this->matrixInfo_.fusionSize;
            DataCopyPad(this->hMixGm_[outputOffset], accumulator, outputParams);
            this->outQueue_.FreeTensor(accumulator);
        }
        this->DataCopyOutInvRmsUb(this->vectorOffset_.singleCoreM,
            this->vectorOffset_.offsetMStart);
    }

    __aicore__ inline void PostProcess()
    {
        uint64_t localOffset = 0;
        for (uint64_t offsetT = this->vectorOffset_.offsetMStart;
             offsetT < this->vectorOffset_.offsetMEnd; offsetT += this->V1_BASE_T) {
            uint64_t lenT = AscendC::Std::min(
                static_cast<uint64_t>(this->V1_BASE_T), this->vectorOffset_.offsetMEnd - offsetT);
            this->AIV1Prologue(offsetT, lenT, localOffset);
            this->AIV1ProcessHPost(offsetT, lenT);
            this->AIV1ProcessHPre(offsetT, lenT);
            this->AIV1ProcessHIn(offsetT, lenT, this->v1ChunkDSize_);
            localOffset += this->V1_BASE_T;
        }
    }

private:
    GlobalTensor<P> partialMmGm_;
    GlobalTensor<P> partialRmsGm_;
    bool selected_ = false;
};

} // namespace MhcPre

#endif // MHC_PRE_M_K_SPLIT_CORE_ARCH35_H
