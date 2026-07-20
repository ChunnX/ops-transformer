#ifndef MHC_PRE_CUBE_COMPUTE_ARCH35_H
#define MHC_PRE_CUBE_COMPUTE_ARCH35_H

#include "mhc_pre_low_api_base_arch35.h"

namespace MhcPre {

constexpr static uint32_t FINAL_ACCUMULATION = 3;
constexpr static uint32_t NON_FINAL_ACCUMULATION = 2;
constexpr static bool MHC_PRE_LOW_API_SPLIT_M = true;
constexpr static uint64_t MHC_PRE_LOW_API_SPLIT_M_ALIGN = 2;
constexpr static uint64_t MHC_PRE_LOW_API_L1_ALLOC_SIZE = 512 * 1024;
constexpr static uint64_t MHC_PRE_LOW_API_L1_BUF_NUM = 2;
constexpr static uint64_t MHC_PRE_LOW_API_L1_BUF_OFFSET = 128 * 256;
constexpr static int MHC_PRE_LOW_API_SYNC_MODE4 = 4;

class MhcPreCubeCompute {
public:
    uint64_t m_{0};
    uint64_t n_{0};
    uint64_t k_{0};
    uint64_t baseM_{0};
    uint64_t baseN_{0};
    uint64_t baseK_{0};
    uint64_t kL1_{0};
    uint8_t bL1BufferID_{0};
    uint8_t l0PingPongID_{0};
    uint8_t cl0PingPongID_{0};

    AscendC::LocalTensor<float> aL0Ping_;
    AscendC::LocalTensor<float> aL0Pong_;
    AscendC::LocalTensor<float> bL0Ping_;
    AscendC::LocalTensor<float> bL0Pong_;
    AscendC::LocalTensor<float> cL0Ping_;
    AscendC::LocalTensor<float> cL0Pong_;

    __aicore__ inline MhcPreCubeCompute() {}

    __aicore__ inline uint8_t GetBL1BufferId()
    {
        return bL1BufferID_;
    }

    __aicore__ inline void CopyInA1Nd2Nz(uint64_t m, uint64_t currentK,
        const AscendC::GlobalTensor<float>& aGlobal, const AscendC::LocalTensor<float>& al1Local)
    {
        CopyInA1Nd2Nz(m, currentK, currentK, aGlobal, al1Local);
    }

    __aicore__ inline void CopyInA1Nd2Nz(uint64_t m, uint64_t currentK, uint64_t srcDValue,
        const AscendC::GlobalTensor<float>& aGlobal, const AscendC::LocalTensor<float>& al1Local)
    {
        AscendC::Nd2NzParams nd2nzParam;
        nd2nzParam.ndNum = 1;
        nd2nzParam.nValue = m;
        nd2nzParam.dValue = currentK;
        nd2nzParam.srcNdMatrixStride = 1;
        nd2nzParam.srcDValue = srcDValue;
        nd2nzParam.dstNzC0Stride = LowApiAlign(m, AscendC::BLOCK_CUBE);
        nd2nzParam.dstNzNStride = 1;
        nd2nzParam.dstNzMatrixStride = 1;
        AscendC::DataCopy(al1Local, aGlobal, nd2nzParam);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(2);
    }

    __aicore__ inline void Init(uint32_t implMode)
    {
        uint32_t aL0OneBuffer = 256 * 32;
        uint32_t bL0OneBuffer = 256 * 32;
        uint32_t cL0OneBuffer = 256 * 128;

        aL0Ping_ = AscendC::LocalTensor<float>(AscendC::TPosition::A2, 0, aL0OneBuffer);
        aL0Pong_ = AscendC::LocalTensor<float>(AscendC::TPosition::A2, aL0OneBuffer * sizeof(float), aL0OneBuffer);
        bL0Ping_ = AscendC::LocalTensor<float>(AscendC::TPosition::B2, 0, bL0OneBuffer);
        bL0Pong_ = AscendC::LocalTensor<float>(AscendC::TPosition::B2, bL0OneBuffer * sizeof(float), bL0OneBuffer);
        cL0Ping_ = AscendC::LocalTensor<float>(AscendC::TPosition::CO1, 0, cL0OneBuffer);
        cL0Pong_ = AscendC::LocalTensor<float>(AscendC::TPosition::CO1, cL0OneBuffer * sizeof(float), cL0OneBuffer);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(4);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);

        if ASCEND_IS_AIC {
            if (implMode == MHC_PRE_IMPL_MODE_HF32) {
                AscendC::SetHF32Mode(1);
                AscendC::SetHF32TransMode(1);
            }
        }
    }

    __aicore__ inline void CopyInB1Nd2Nz(uint64_t k, uint64_t currentK, uint64_t baseN,
        const AscendC::GlobalTensor<float>& bGlobal, const AscendC::LocalTensor<float>& bl1Local)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(bL1BufferID_);
        k_ = k;
        kL1_ = currentK;
        baseN_ = baseN;
        AscendC::Nd2NzParams nd2nzParam;
        nd2nzParam.ndNum = 1;
        nd2nzParam.nValue = baseN_;
        nd2nzParam.dValue = kL1_;
        nd2nzParam.srcNdMatrixStride = 1;
        nd2nzParam.srcDValue = k_;
        nd2nzParam.dstNzC0Stride = LowApiAlign(baseN_, AscendC::BLOCK_CUBE);
        nd2nzParam.dstNzNStride = 1;
        nd2nzParam.dstNzMatrixStride = 1;
        AscendC::DataCopy(bl1Local, bGlobal, nd2nzParam);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufferID_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufferID_);
    }

    __aicore__ inline void Process(uint64_t m, uint64_t n, uint64_t baseM, uint64_t baseK,
        bool isFirstKL1, bool isLastKL1, const AscendC::LocalTensor<float>& al1Local,
        const AscendC::LocalTensor<float>& bl1Local)
    {
        m_ = m;
        n_ = n;
        baseM_ = baseM;
        baseK_ = baseK;
        if (isFirstKL1) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(cl0PingPongID_);
        }
        uint64_t kL1Offset = 0;
        for (uint64_t kb = 0; kb < kL1_; kb += baseK_) {
            bool isLastKL0 = (kb + baseK_) >= kL1_;
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPongID_ + 3);
            CopyInA2(kb, kL1Offset, al1Local);
            CopyInB2(kb, kL1Offset, bl1Local);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPongID_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPongID_);
            MmadBase(kb, isFirstKL1, isLastKL1, isLastKL0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPongID_ + 3);
            l0PingPongID_ ^= 1;
            kL1Offset += baseK_;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(bL1BufferID_);
        bL1BufferID_ ^= 1;
    }

    __aicore__ inline void CopyInA2(uint64_t kOffset, uint64_t kAL1Offset,
        const AscendC::LocalTensor<float>& al1Local)
    {
        uint64_t mAL1 = LowApiAlign(baseM_, AscendC::BLOCK_CUBE);
        uint64_t offsetAL1 = LowApiAlign(kAL1Offset, MHC_PRE_LOW_API_C0_SIZE) * mAL1;
        AscendC::LoadData2DParamsV2 loadData2dParams;
        uint64_t currM = baseM_;
        uint64_t currK = AscendC::Std::min(baseK_, kL1_ - kOffset);
        loadData2dParams.mStartPosition = 0;
        loadData2dParams.kStartPosition = 0;
        loadData2dParams.mStep = LowApiCeilDiv(currM, AscendC::BLOCK_CUBE);
        loadData2dParams.kStep = LowApiCeilDiv(currK, MHC_PRE_LOW_API_C0_SIZE);
        loadData2dParams.srcStride = LowApiCeilDiv(currM, AscendC::BLOCK_CUBE);
        loadData2dParams.dstStride = loadData2dParams.mStep;
        loadData2dParams.ifTranspose = false;
        AscendC::LoadData(l0PingPongID_ == 0 ? aL0Ping_ : aL0Pong_, al1Local[offsetAL1], loadData2dParams);
    }

    __aicore__ inline void CopyInB2(uint64_t kOffset, uint64_t kBL1Offset,
        const AscendC::LocalTensor<float>& bl1Local)
    {
        uint64_t nBL1 = LowApiAlign(baseN_, AscendC::BLOCK_CUBE);
        uint64_t offsetBL1 = LowApiAlign(kBL1Offset, MHC_PRE_LOW_API_C0_SIZE) * nBL1;
        AscendC::LoadData2DParamsV2 loadData2dParams;
        uint64_t currN = baseN_;
        uint64_t currK = AscendC::Std::min(baseK_, kL1_ - kOffset);
        loadData2dParams.mStartPosition = 0;
        loadData2dParams.kStartPosition = 0;
        loadData2dParams.mStep = LowApiCeilDiv(currN, AscendC::BLOCK_CUBE);
        loadData2dParams.kStep = LowApiCeilDiv(currK, MHC_PRE_LOW_API_C0_SIZE);
        loadData2dParams.srcStride = LowApiCeilDiv(currN, AscendC::BLOCK_CUBE);
        loadData2dParams.dstStride = loadData2dParams.mStep;
        loadData2dParams.ifTranspose = false;
        AscendC::LoadData(l0PingPongID_ == 0 ? bL0Ping_ : bL0Pong_, bl1Local[offsetBL1], loadData2dParams);
    }

    __aicore__ inline void MmadBase(uint64_t kOffset, bool isFirstKL1, bool isLastKL1, bool isLastKL0)
    {
        uint32_t mmK = AscendC::Std::min(baseK_, kL1_ - kOffset);
        AscendC::MmadParams params;
        params.m = baseM_;
        params.n = baseN_;
        params.k = mmK;
        params.disableGemv = true;
        params.cmatrixInitVal = (isFirstKL1 && kOffset == 0);
        params.cmatrixSource = false;
        params.unitFlag = (isLastKL1 && isLastKL0) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
        AscendC::Mmad(cl0PingPongID_ == 0 ? cL0Ping_ : cL0Pong_,
            l0PingPongID_ == 0 ? aL0Ping_ : aL0Pong_,
            l0PingPongID_ == 0 ? bL0Ping_ : bL0Pong_, params);
        AscendC::PipeBarrier<PIPE_M>();
    }

    __aicore__ inline void CopyOut(const AscendC::LocalTensor<float>& dstLocal)
    {
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(cl0PingPongID_);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(cl0PingPongID_);
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams;
        uint64_t c0 = AscendC::AuxGetC0Size<float>();
        fixpipeParams.nSize = LowApiAlign(baseN_, c0);
        fixpipeParams.mSize = MHC_PRE_LOW_API_SPLIT_M ? LowApiAlign(baseM_, MHC_PRE_LOW_API_SPLIT_M_ALIGN) : baseM_;
        fixpipeParams.dstStride = fixpipeParams.nSize;
        fixpipeParams.srcStride = LowApiAlign(baseM_, AscendC::BLOCK_CUBE);
        fixpipeParams.quantPre = QuantMode_t::NoQuant;
        fixpipeParams.dualDstCtl = MHC_PRE_LOW_API_SPLIT_M ?
            static_cast<uint8_t>(AscendC::McgShfMode::DUAL_DST_SPLIT_M) : 0;
        fixpipeParams.unitFlag = FINAL_ACCUMULATION;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 1;
        fixpipeParams.params.dstNdStride = 1;
        AscendC::Fixpipe<float, float, AscendC::Impl::CFG_ROW_MAJOR_UB>(dstLocal,
            cl0PingPongID_ == 0 ? cL0Ping_ : cL0Pong_, fixpipeParams);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(cl0PingPongID_);
        cl0PingPongID_ ^= 1;
    }

    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<float>& dstGlobal, uint32_t dstStride)
    {
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(cl0PingPongID_);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(cl0PingPongID_);
        AscendC::DataCopyCO12DstParams copyParams;
        copyParams.nSize = baseN_;
        copyParams.mSize = baseM_;
        copyParams.dstStride = dstStride;
        copyParams.srcStride = LowApiAlign(baseM_, AscendC::BLOCK_CUBE);
        copyParams.quantPre = QuantMode_t::NoQuant;
        copyParams.nz2ndEn = true;
        copyParams.unitFlag = FINAL_ACCUMULATION;
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
        AscendC::DataCopy(dstGlobal, cl0PingPongID_ == 0 ? cL0Ping_ : cL0Pong_, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(cl0PingPongID_);
        cl0PingPongID_ ^= 1;
    }
    
    __aicore__ inline void End(uint32_t implMode)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(4);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
        if ASCEND_IS_AIC {
            if (implMode == MHC_PRE_IMPL_MODE_HF32) {
                AscendC::SetHF32Mode(0);
            }
        }
    }
};

} // namespace MhcPre

#endif // MHC_PRE_CUBE_COMPUTE_ARCH35_H
