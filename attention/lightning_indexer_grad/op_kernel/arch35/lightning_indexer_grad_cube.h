#ifndef LIGHTNING_INDEXER_GRAD_CUBE_H
#define LIGHTNING_INDEXER_GRAD_CUBE_H

#include "kernel_operator.h"
#include "lightning_indexer_grad_base.h"

namespace LIG {
using namespace AscendC;
constexpr int32_t C0_BYTE_SIZE = 32;
constexpr FixpipeConfig CFG_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};

template<typename T, typename U>
class LightningIndexerGradCube1Service {
public:
    __aicore__ inline LightningIndexerGradCube1Service() = default;

    __aicore__ inline void Init(const int32_t m, const int32_t n,
        const int32_t k, int32_t baseM, int32_t baseN, int32_t baseK)
    {
        m_ = m;
        n_ = n;
        k_ = k;
        baseM_ = baseM;
        baseN_ = baseN;
        baseK_ = baseK;
        // L0 空间的分配
        int32_t aL0OneBufferSize = 32 * 1024;
        int32_t bL0OneBufferSize = 32 * 1024;
        int32_t cL0OneBufferSize = 128 * 1024;

        int32_t aL0OneBufferElemCount = aL0OneBufferSize / sizeof(T);
        int32_t bL0OneBufferElemCount = bL0OneBufferSize / sizeof(T);
        int32_t cL0OneBufferElemCount = cL0OneBufferSize / sizeof(U);

        aL0Ping_ = LocalTensor<T>(TPosition::A2, 0, aL0OneBufferElemCount);
        aL0Pong_ = LocalTensor<T>(TPosition::A2, aL0OneBufferSize, aL0OneBufferElemCount);
        bL0Ping_ = LocalTensor<T>(TPosition::B2, 0, bL0OneBufferElemCount);
        bL0Pong_ = LocalTensor<T>(TPosition::B2, bL0OneBufferSize, bL0OneBufferElemCount);
        cL0Ping_ = LocalTensor<U>(TPosition::CO1, 0, cL0OneBufferElemCount);
        cL0Pong_ = LocalTensor<U>(TPosition::CO1, cL0OneBufferSize, cL0OneBufferElemCount);

        // hf32 compute
        SetHF32Mode(false);
        SetHF32TransMode(false);
    }

    __aicore__ inline void CopyInA1Nd2Nz(const LocalTensor<T> a1Local,
        const GlobalTensor<T> a1Global)
    {
        // copyIn m * k
        Nd2NzParams nd2nzParam;
        
        nd2nzParam.ndNum = 1;
        nd2nzParam.srcNdMatrixStride = 0;
        nd2nzParam.dstNzNStride = 1;
        nd2nzParam.dstNzMatrixStride = 0;
        nd2nzParam.nValue = m_;
        nd2nzParam.dValue = k_;
        nd2nzParam.srcDValue = k_;
        nd2nzParam.dstNzC0Stride = m_;

        DataCopyGM2L1(a1Local, a1Global, nd2nzParam);
        SetFlag<HardEvent::MTE2_MTE1>(0);
        
    }

    __aicore__ inline void CopyInA2(const LocalTensor<T> &aL1Local)
    {
        // 910B: NZ -> ZZ
        // 910D: NZ -> NZ
        LoadData2DParamsV2 loadData2dParams;
        loadData2dParams.mStartPosition = 0;
        loadData2dParams.kStartPosition = 0;
        loadData2dParams.mStep = CeilDiv(baseM_, BLOCK_CUBE);
        loadData2dParams.kStep = CeilDiv(baseK_ * sizeof(T), C0_BYTE_SIZE);
        loadData2dParams.srcStride = CeilDiv(baseM_, BLOCK_CUBE);
        loadData2dParams.dstStride = CeilDiv(baseM_, BLOCK_CUBE);
        loadData2dParams.ifTranspose = false;
        WaitFlag<HardEvent::MTE2_MTE1>(0);
        LoadData(aL0Ping_, aL1Local, loadData2dParams);
        SetFlag<HardEvent::MTE1_M>(0);
    }

    __aicore__ inline void CopyInB2(LocalTensor<T> &bl1Local)
    {
        // NZ -> ZN.Transpose = NZ
        LoadData2DParamsV2 loadData2dParams;
        loadData2dParams.mStartPosition = 0;
        loadData2dParams.kStartPosition = 0;
        loadData2dParams.mStep = CeilDiv(baseN_, BLOCK_CUBE);
        loadData2dParams.kStep = CeilDiv(baseK_ * sizeof(T), C0_BYTE_SIZE);
        loadData2dParams.srcStride = CeilDiv(baseN_, BLOCK_CUBE);
        loadData2dParams.dstStride = CeilDiv(baseN_, BLOCK_CUBE);
        loadData2dParams.ifTranspose = false;

        LoadData(bL0Ping_, bl1Local, loadData2dParams);
        SetFlag<HardEvent::MTE1_M>(1);
    }

    __aicore__ inline void Process()
    {
        MmadParams mmadParams;
        mmadParams.m = baseM_;
        mmadParams.n = baseN_;
        mmadParams.k = baseK_;

        mmadParams.disableGemv = true;
        mmadParams.cmatrixInitVal = true;
        mmadParams.cmatrixSource = false;

        WaitFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(1);
        Mmad(cL0Ping_, aL0Ping_, bL0Ping_, mmadParams);
        SetFlag<HardEvent::M_FIX>(0);
    }

    __aicore__ inline void CopyOut(LocalTensor<U> &ubLocal, const int32_t elementsCountPerRow)
    {
        FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.mSize = baseM_;
        // 按N维度拆分，N必须为32的倍数
        fixpipeParams.nSize = CeilAlign(baseN_, 32);
        fixpipeParams.srcStride = CeilAlign(baseM_, BLOCK_CUBE);
        fixpipeParams.dstStride = elementsCountPerRow;
        fixpipeParams.dualDstCtl = 0b10;

        WaitFlag<HardEvent::M_FIX>(0);
        Fixpipe<U, U, CFG_ROW_MAJOR_UB>(ubLocal, cL0Ping_, fixpipeParams);
    }

private:
    int32_t m_, n_, k_;
    int32_t baseM_, baseN_, baseK_;
    LocalTensor<T> aL0Ping_;
    LocalTensor<T> aL0Pong_;
    LocalTensor<T> bL0Ping_;
    LocalTensor<T> bL0Pong_;
    LocalTensor<U> cL0Ping_;
    LocalTensor<U> cL0Pong_;
    int32_t l0PingPongID_ = 0;
};

template<typename T, typename U>
class LightningIndexerGradCube2Service {
public:
    __aicore__ inline LightningIndexerGradCube2Service() = default;

    __aicore__ inline void Init(int32_t headNumQ, int32_t headDim, int32_t topK)
    {
        mm1BaseM_ = headNumQ;
        mm1BaseN_ = headDim;
        mm1BaseK_ = topK;

        mm2BaseM_ = topK;
        mm2BaseN_ = headDim;
        mm2BaseK_ = headNumQ;

        // L0 空间的分配
        int32_t aL0OneBufferSize = 32 * 1024;
        int32_t bL0OneBufferSize = 32 * 1024;
        int32_t cL0OneBufferSize = 128 * 1024;

        int32_t aL0OneBufferElemCount = aL0OneBufferSize / sizeof(T);
        int32_t bL0OneBufferElemCount = bL0OneBufferSize / sizeof(T);
        int32_t cL0OneBufferElemCount = cL0OneBufferSize / sizeof(U);

        aL0Ping_ = LocalTensor<T>(TPosition::A2, 0, aL0OneBufferElemCount);
        aL0Pong_ = LocalTensor<T>(TPosition::A2, aL0OneBufferSize, aL0OneBufferElemCount);
        bL0Ping_ = LocalTensor<T>(TPosition::B2, 0, bL0OneBufferElemCount);
        bL0Pong_ = LocalTensor<T>(TPosition::B2, bL0OneBufferSize, bL0OneBufferElemCount);
        cL0Ping_ = LocalTensor<U>(TPosition::CO1, 0, cL0OneBufferElemCount);
        cL0Pong_ = LocalTensor<U>(TPosition::CO1, cL0OneBufferSize, cL0OneBufferElemCount);

        // hf32 compute
        SetHF32Mode(false);
        SetHF32TransMode(false);
    }


    __aicore__ inline void CopyInA2(
        const LocalTensor<T> &aL1Local1)
    {
        // 910B: NZ -> ZZ
        // 910D: NZ -> NZ
        LoadData2DParamsV2 loadData2dParams1;
        loadData2dParams1.mStartPosition = 0;
        loadData2dParams1.kStartPosition = 0;
        loadData2dParams1.sid = 0;
        loadData2dParams1.mStep = CeilDiv(mm1BaseM_, BLOCK_CUBE);
        loadData2dParams1.kStep = CeilDiv(mm1BaseK_ * sizeof(T), C0_BYTE_SIZE);
        loadData2dParams1.srcStride = CeilDiv(mm1BaseM_, BLOCK_CUBE);
        loadData2dParams1.dstStride = CeilDiv(mm1BaseM_, BLOCK_CUBE);
        loadData2dParams1.ifTranspose = false;

        LoadData2DParamsV2 loadData2dParams2;
        loadData2dParams2.mStartPosition = 0;
        loadData2dParams2.kStartPosition = 0;
        loadData2dParams2.sid = 0;
        loadData2dParams2.mStep = CeilDivision(mm2BaseK_, BLOCK_CUBE);
        loadData2dParams2.kStep = CeilDivision(mm2BaseM_ * sizeof(T), C0_BYTE_SIZE);
        loadData2dParams2.srcStride = CeilDivision(mm2BaseK_, BLOCK_CUBE);
        loadData2dParams2.dstStride = CeilDivision(mm2BaseM_, BLOCK_CUBE);
        loadData2dParams2.ifTranspose = true;
        
        LoadData(aL0Ping_, aL1Local1, loadData2dParams1);
        LoadData(aL0Pong_, aL1Local1, loadData2dParams2);
        SetFlag<HardEvent::MTE1_M>(0);
    }

    __aicore__ inline void CopyInB2(
        LocalTensor<T> &bl1Local1, LocalTensor<T> &bl1Local2)
    {
        // NZ -> ZN
        LoadData2DParamsV2 loadData2dParams1;
        loadData2dParams1.mStartPosition = 0;
        loadData2dParams1.kStartPosition = 0;
        loadData2dParams1.sid = 0;
        loadData2dParams1.mStep = CeilDiv(mm1BaseK_, BLOCK_CUBE);
        loadData2dParams1.kStep = CeilDiv(mm1BaseN_ * sizeof(T), C0_BYTE_SIZE);
        loadData2dParams1.srcStride = CeilDiv(mm1BaseK_, BLOCK_CUBE);
        loadData2dParams1.dstStride = CeilDiv(mm1BaseN_, BLOCK_CUBE);
        loadData2dParams1.ifTranspose = true;
        
        // NZ -> ZN
        LoadData2DParamsV2 loadData2dParams2;
        loadData2dParams2.mStartPosition = 0;
        loadData2dParams2.kStartPosition = 0;
        loadData2dParams2.sid = 0;
        loadData2dParams2.mStep = CeilDiv(mm2BaseK_, BLOCK_CUBE);
        loadData2dParams2.kStep = CeilDiv(mm2BaseN_ * sizeof(T), C0_BYTE_SIZE);
        loadData2dParams2.srcStride = CeilDiv(mm2BaseK_, BLOCK_CUBE);
        loadData2dParams2.dstStride = CeilDiv(mm2BaseN_, BLOCK_CUBE);
        loadData2dParams2.ifTranspose = true;
        
        LoadData(bL0Pong_, bl1Local2, loadData2dParams2);
        LoadData(bL0Ping_, bl1Local1, loadData2dParams1);
        SetFlag<HardEvent::MTE1_M>(1);
    }

    __aicore__ inline void Process(const bool firstTileFlag)
    {
        MmadParams mmadParams1;
        mmadParams1.m = mm1BaseM_;
        mmadParams1.n = mm1BaseN_;
        mmadParams1.k = mm1BaseK_;
        mmadParams1.disableGemv = true;
        mmadParams1.cmatrixInitVal = firstTileFlag;
        mmadParams1.cmatrixSource = false;
        mmadParams1.isBias = false;

        MmadParams mmadParams2;
        mmadParams2.m = mm2BaseM_;
        mmadParams2.n = mm2BaseN_;
        mmadParams2.k = mm2BaseK_;
        mmadParams2.disableGemv = true;
        mmadParams2.cmatrixInitVal = true;
        mmadParams2.cmatrixSource = false;
        mmadParams2.isBias = false;

        WaitFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(1);
        Mmad(cL0Pong_, aL0Ping_, bL0Ping_, mmadParams1);
        Mmad(cL0Ping_, aL0Pong_, bL0Pong_, mmadParams2);
        SetFlag<HardEvent::M_FIX>(0);
    }

    __aicore__ inline void CopyOutMM1Reslut(GlobalTensor<T> outGlobal)
    {
        FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = mm1BaseN_;
        fixpipeParams.mSize = mm1BaseM_;
        fixpipeParams.srcStride = mm1BaseM_;
        fixpipeParams.dstStride = mm1BaseN_;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = mm1BaseM_ * mm1BaseN_;
        if (std::is_same<T, half>::value) {
            fixpipeParams.quantPre = QuantMode_t::F322F16;
        } else {
            fixpipeParams.quantPre = QuantMode_t::F322BF16;
        }
        
        PipeBarrier<PIPE_ALL>();
        Fixpipe(outGlobal, cL0Pong_, fixpipeParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyOutMM2Reslut(LocalTensor<float> ubLocal)
    {
        FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.mSize = CeilAlign(mm2BaseM_, 2);
        fixpipeParams.nSize = mm2BaseN_;
        fixpipeParams.srcStride = CeilAlign(mm2BaseM_, BLOCK_CUBE);
        fixpipeParams.dstStride = mm2BaseN_;
        fixpipeParams.dualDstCtl = 0b01;
        WaitFlag<HardEvent::M_FIX>(0);
        Fixpipe<float, float, CFG_ROW_MAJOR_UB>(ubLocal, cL0Ping_, fixpipeParams);
    }

private:
    int32_t mm1BaseM_, mm1BaseN_, mm1BaseK_;
    int32_t mm2BaseM_, mm2BaseN_, mm2BaseK_;
    LocalTensor<T> aL0Ping_;
    LocalTensor<T> aL0Pong_;
    LocalTensor<T> bL0Ping_;
    LocalTensor<T> bL0Pong_;
    LocalTensor<U> cL0Ping_;
    LocalTensor<U> cL0Pong_;
};

}
#endif