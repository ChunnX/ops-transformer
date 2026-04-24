/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dense_lightning_indexer_softmax_lse_cube_block.h
 * \brief Arch35 cube block implementation
 */

#ifndef DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_CUBE_BLOCK_H
#define DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_CUBE_BLOCK_H

#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "dense_lightning_indexer_softmax_lse_regbase_common.h"

namespace DenseLISoftmaxLseKernelRegbase {

TEMPLATES_DEF
class DenseLISoftmaxLseBlockCube {
public:
    TPipe *pipe;

    constexpr static uint32_t QUERY_BUFFER_SIZE = 128 * 512 * 2;
    constexpr static uint32_t KEY_BUFFER_SIZE = 512 * 128 * 3;
    constexpr static uint32_t L0C_SINGLE_BUFFER_SIZE = 64 * 1024;
    constexpr static uint32_t L0_SINGLE_BUFFER_SIZE = 32 * 1024;

    static constexpr uint32_t M_SPLIT_SIZE = 128;
    static constexpr uint32_t N_SPLIT_SIZE = 128;
    static constexpr uint32_t K_SPLIT_SIZE = 128;
    static constexpr uint32_t D_SIZE = 128;

    static constexpr uint8_t SYNC_MODE = 2;
    static constexpr uint32_t C0_SIZE = 16;

    GlobalTensor<INPUT_T> queryGm;
    GlobalTensor<INPUT_T> keyGm;
    GlobalTensor<T> mm1ResGm;

    BufferManager<BufferType::L1> *l1BufferManagerPtr;
    BuffersPolicySingleBuffer<BufferType::L1> queryBuf;
    BuffersPolicyDB<BufferType::L1> keyBuf;

    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BuffersPolicyDB<BufferType::L0A> l0aBuf;
    BuffersPolicyDB<BufferType::L0B> l0bBuf;

    BufferManager<BufferType::L0C> l0cBufferManager;
    BuffersPolicyDB<BufferType::L0C> l0cBuf;

    DenseLISoftmaxLseConstInfo constInfo{};

    __aicore__ inline DenseLISoftmaxLseBlockCube(){};
    __aicore__ inline void SetCubeBlockParams(TPipe *tPipe, BufferManager<BufferType::L1> *l1BuffMgr);
    __aicore__ inline void InitCubeBuffers();
    __aicore__ inline void InitGlobalBuffer(GM_ADDR query, GM_ADDR key, GlobalTensor<T> &mm1Res);
    __aicore__ inline void ComputeMm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &mm1ResBuf,
                                      DenseLISoftmaxLseRunInfo &runInfo,
                                      DenseLISoftmaxLseConstInfo &constInfo);

private:
    __aicore__ inline void CopyGmToL1(LocalTensor<INPUT_T> &l1Tensor, GlobalTensor<INPUT_T> &gmSrcTensor,
                                      uint32_t srcN, uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void CopyInQueryToL1(LocalTensor<INPUT_T> &l1Tensor, 
                                           const DenseLISoftmaxLseRunInfo &info,
                                           DenseLISoftmaxLseConstInfo &constInfo);
    __aicore__ inline void CopyInKeyToL1(LocalTensor<INPUT_T> &l1Tensor, 
                                         const DenseLISoftmaxLseRunInfo &info,
                                         DenseLISoftmaxLseConstInfo &constInfo,
                                         uint64_t s2GmOffset, uint32_t s2L1RealSize);
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseBlockCube<TEMPLATE_ARGS>::SetCubeBlockParams(
    TPipe *tPipe, BufferManager<BufferType::L1> *l1BuffMgr)
{
    this->pipe = tPipe;
    this->l1BufferManagerPtr = l1BuffMgr;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseBlockCube<TEMPLATE_ARGS>::InitCubeBuffers()
{
    queryBuf.Init(*l1BufferManagerPtr, QUERY_BUFFER_SIZE);
    keyBuf.Init(*l1BufferManagerPtr, KEY_BUFFER_SIZE);

    l0aBufferManager.Init(pipe, L0_MAX_SIZE);
    l0bBufferManager.Init(pipe, L0_MAX_SIZE);
    l0aBuf.Init(l0aBufferManager, L0_SINGLE_BUFFER_SIZE);
    l0bBuf.Init(l0bBufferManager, L0_SINGLE_BUFFER_SIZE);

    l0cBufferManager.Init(pipe, L0C_MAX_SIZE);
    l0cBuf.Init(l0cBufferManager, L0C_SINGLE_BUFFER_SIZE);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseBlockCube<TEMPLATE_ARGS>::InitGlobalBuffer(
    GM_ADDR query, GM_ADDR key, GlobalTensor<T> &mm1Res)
{
    queryGm.SetGlobalBuffer((__gm__ INPUT_T *)query);
    keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
    this->mm1ResGm = mm1Res;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseBlockCube<TEMPLATE_ARGS>::CopyGmToL1(
    LocalTensor<INPUT_T> &l1Tensor, GlobalTensor<INPUT_T> &gmSrcTensor,
    uint32_t srcN, uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN;
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcDValue = srcDstride;
    nd2nzPara.dstNzC0Stride = (srcN + 15) / 16 * 16;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseBlockCube<TEMPLATE_ARGS>::CopyInQueryToL1(
    LocalTensor<INPUT_T> &l1Tensor, const DenseLISoftmaxLseRunInfo &info,
    DenseLISoftmaxLseConstInfo &constInfo)
{
    auto srcGm = queryGm[info.queryTensorOffset];
    uint32_t mSize = info.actMBaseSize;
    CopyGmToL1(l1Tensor, srcGm, mSize, constInfo.dSize, constInfo.dSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseBlockCube<TEMPLATE_ARGS>::CopyInKeyToL1(
    LocalTensor<INPUT_T> &l1Tensor, const DenseLISoftmaxLseRunInfo &info,
    DenseLISoftmaxLseConstInfo &constInfo, uint64_t s2GmOffset, uint32_t s2L1RealSize)
{
    auto srcGm = keyGm[info.keyTensorOffset + s2GmOffset * constInfo.dSize];
    CopyGmToL1(l1Tensor, srcGm, s2L1RealSize, constInfo.dSize, constInfo.dSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseBlockCube<TEMPLATE_ARGS>::ComputeMm1(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &mm1ResBuf,
    DenseLISoftmaxLseRunInfo &runInfo, DenseLISoftmaxLseConstInfo &constInfo)
{
    uint32_t mSize = runInfo.actMBaseSize;
    uint32_t kSize = constInfo.dSize;
    uint32_t nSize = runInfo.actualSingleProcessSInnerSize;

    uint32_t mLoopTimes = DenseLISoftmaxLseCeilDiv<uint32_t>(mSize, M_SPLIT_SIZE);
    uint32_t nLoopTimes = DenseLISoftmaxLseCeilDiv<uint32_t>(nSize, N_SPLIT_SIZE);
    uint32_t mTailSize = mSize - (mLoopTimes - 1) * M_SPLIT_SIZE;
    uint32_t nTailSize = nSize - (nLoopTimes - 1) * N_SPLIT_SIZE;

    LocalTensor<T> outTensor = mm1ResBuf.template GetTensor<T>();

    Buffer<BufferType::L1> queryL1Buffer;
    LocalTensor<INPUT_T> queryL1Tensor;

    MMParam mmParam = {
        M_SPLIT_SIZE,
        N_SPLIT_SIZE,
        K_SPLIT_SIZE,
        false,
        true,
        true,
        true,
        UNITFLAG_DISABLE
    };

    queryL1Buffer = queryBuf.Get();
    queryL1Tensor = queryL1Buffer.GetTensor<INPUT_T>();
    CopyInQueryToL1(queryL1Tensor, runInfo, constInfo);
    queryL1Buffer.Set<HardEvent::MTE2_MTE1>();
    queryL1Buffer.Wait<HardEvent::MTE2_MTE1>();

    Buffer<BufferType::L0C> mm1L0CBuffer = l0cBuf.Get();
    LocalTensor<T> mm1L0CTensor = mm1L0CBuffer.GetTensor<T>();
    mm1L0CBuffer.Wait<HardEvent::FIX_M>();

    for (uint32_t nIdx = 0; nIdx < nLoopTimes; nIdx++) {
        uint32_t curNSize = (nIdx == nLoopTimes - 1) ? nTailSize : N_SPLIT_SIZE;
        uint64_t s2GmOffset = nIdx * N_SPLIT_SIZE;

        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> keyL1Buf = keyBuf.Get();
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_GATHER_TO_MM1_FLAG[nIdx % MODE_NUM_2]);
        LocalTensor<INPUT_T> keyL1Tensor = keyL1Buf.GetTensor<INPUT_T>();
        CopyInKeyToL1(keyL1Tensor, runInfo, constInfo, s2GmOffset, curNSize);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_GATHER_TO_MM1_FLAG[nIdx % MODE_NUM_2]);

        for (uint32_t mIdx = 0; mIdx < mLoopTimes; mIdx++) {
            uint32_t curMSize = (mIdx == mLoopTimes - 1) ? mTailSize : M_SPLIT_SIZE;
            mmParam.singleM = curMSize;
            mmParam.singleN = curNSize;

            uint64_t l0cOffset = mIdx * N_SPLIT_SIZE + nIdx * mSize * N_SPLIT_SIZE;
            MatmulBase<INPUT_T, INPUT_T, T, M_SPLIT_SIZE, N_SPLIT_SIZE, K_SPLIT_SIZE, 
                       ABLayout::MK, ABLayout::KN>(
                queryL1Tensor[mIdx * M_SPLIT_SIZE * AlignTo(kSize, C0_SIZE)],
                keyL1Tensor,
                l0aBuf, l0bBuf, mm1L0CTensor[mIdx * curNSize * AlignTo(curMSize, C0_SIZE)],
                mmParam);
        }
    }

    mm1L0CBuffer.Set<HardEvent::M_FIX>();
    mm1L0CBuffer.Wait<HardEvent::M_FIX>();

    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_MM1_TO_V1_FLAG[runInfo.loop % MODE_NUM_2]);
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipe2UbParams;
    fixpipe2UbParams.nSize = (nSize + 7) >> 3 << 3;
    fixpipe2UbParams.mSize = (mSize + 1) >> 1 << 1;
    fixpipe2UbParams.srcStride = AlignTo(fixpipe2UbParams.mSize, static_cast<uint16_t>(C0_SIZE));
    fixpipe2UbParams.dstStride = AlignTo(constInfo.s2BaseSize, static_cast<int32_t>(C0_SIZE));
    fixpipe2UbParams.dualDstCtl = 1;
    fixpipe2UbParams.params.ndNum = 1;
    fixpipe2UbParams.params.srcNdStride = 0;
    fixpipe2UbParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outTensor, mm1L0CTensor, fixpipe2UbParams);
    mm1L0CBuffer.Set<HardEvent::FIX_M>();
    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_MM1_TO_V1_FLAG[runInfo.loop % MODE_NUM_2]);

    queryL1Buffer.Set<HardEvent::MTE1_MTE2>();
}

TEMPLATES_DEF
class DenseLISoftmaxLseBlockCubeDummy {
public:
    __aicore__ inline DenseLISoftmaxLseBlockCubeDummy(){};
    __aicore__ inline void SetCubeBlockParams(TPipe *tPipe, BufferManager<BufferType::L1> *l1BuffMgr){};
    __aicore__ inline void InitCubeBuffers(){};
    __aicore__ inline void InitGlobalBuffer(GM_ADDR query, GM_ADDR key, GlobalTensor<T> &mm1Res){};
    __aicore__ inline void ComputeMm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &mm1ResBuf,
                                      DenseLISoftmaxLseRunInfo &runInfo,
                                      DenseLISoftmaxLseConstInfo &constInfo){};
};

template <typename T>
struct CubeBlockTraits;

#define GEN_TRAIT_TYPE(name, ...) using name##_TRAITS = name;
#define GEN_TRAIT_CONST(name, type, ...) static constexpr type name##Traits = name;

#define DEFINE_CUBE_BLOCK_TRAITS(CUBE_BLOCK_CLASS)                                                                     \
    TEMPLATES_DEF_NO_DEFAULT                                                                                           \
    struct CubeBlockTraits<CUBE_BLOCK_CLASS<TEMPLATE_ARGS>> {                                                          \
        CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TRAIT_TYPE)                                                                  \
        CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TRAIT_CONST)                                                                \
    };

DEFINE_CUBE_BLOCK_TRAITS(DenseLISoftmaxLseBlockCube);
DEFINE_CUBE_BLOCK_TRAITS(DenseLISoftmaxLseBlockCubeDummy);

#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS                                                                                                    \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE)                                                                       \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)

} // namespace DenseLISoftmaxLseKernelRegbase

#endif // DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_CUBE_BLOCK_H