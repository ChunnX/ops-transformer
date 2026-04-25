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
 * \brief Cube block for A5 (arch35) implementation
 */
#ifndef DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_CUBE_BLOCK_H
#define DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_CUBE_BLOCK_H

#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "dense_lightning_indexer_softmax_lse_regbase_common.h"

namespace DenseLISoftmaxLseRegbase {
using T = float;

TEMPLATES_DEF
class DenseLISoftmaxLseCubeBlock {
public:
    TPipe *pipe;
    constexpr static uint32_t QUERY_INDEX_BUFFER_SIZE = 64 * 128 * 2;
    constexpr static uint32_t KEY_INDEX_BUFFER_SIZE = 128 * 128 * 2;
    constexpr static uint32_t L0C_SINGLE_BUFFER_SIZE = 64 * 1024;
    constexpr static uint32_t L0_SINGLE_BUFFER_SIZE = 32 * 1024;

    static constexpr uint32_t M_SPLIT_SIZE = 128;
    static constexpr uint32_t N_SPLIT_SIZE = 128;
    static constexpr uint32_t K_SPLIT_SIZE = 128;

    static constexpr uint8_t SYNC_MODE = 2;
    static constexpr uint32_t C0_SIZE = 16;

    GlobalTensor<INPUT_T> queryIndexGm;
    GlobalTensor<INPUT_T> keyIndexGm;
    GlobalTensor<T> mm1ResGm;

    BufferManager<BufferType::L1> *l1BufferManagerPtr;
    BuffersPolicySingleBuffer<BufferType::L1> queryBuf;
    BuffersPolicySingleBuffer<BufferType::L1> keyBuf;

    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BuffersPolicyDB<BufferType::L0A> l0aBuf;
    BuffersPolicyDB<BufferType::L0B> l0bBuf;

    BufferManager<BufferType::L0C> l0cBufferManager;
    BuffersPolicy3buff<BufferType::L0C> commonL0CBuf;

    DenseLISoftmaxLseConstInfo constInfo{};

    __aicore__ inline DenseLISoftmaxLseCubeBlock(){};
    __aicore__ inline void SetCubeBlockParams(TPipe *tPipe, BufferManager<BufferType::L1> *l1BuffMgr);
    __aicore__ inline void InitCubeBuffers();
    __aicore__ inline void InitGlobalBuffer(GM_ADDR queryIndex, GM_ADDR keyIndex, GlobalTensor<T> &mm1ResGm);
    __aicore__ inline void ComputeMm1(
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &mm1ResBuf,
        DenseLISoftmaxLseRunInfo &runInfo,
        DenseLISoftmaxLseConstInfo &constInfo);

private:
    __aicore__ inline void CopyGmToL1(LocalTensor<INPUT_T> &l1Tensor,
                                      GlobalTensor<INPUT_T> &gmSrcTensor,
                                      uint32_t srcN, uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void CopyInQueryIndexToL1(LocalTensor<INPUT_T> &l1Tensor,
                                                const DenseLISoftmaxLseRunInfo &info,
                                                DenseLISoftmaxLseConstInfo &constInfo);
    __aicore__ inline void CopyInKeyIndexToL1(LocalTensor<INPUT_T> &l1Tensor,
                                               const DenseLISoftmaxLseRunInfo &info,
                                               DenseLISoftmaxLseConstInfo &constInfo);
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseCubeBlock<TEMPLATE_ARGS>::SetCubeBlockParams(
    TPipe *tPipe, BufferManager<BufferType::L1> *l1BuffMgr) {
    this->pipe = tPipe;
    this->l1BufferManagerPtr = l1BuffMgr;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseCubeBlock<TEMPLATE_ARGS>::InitCubeBuffers() {
    queryBuf.Init(*l1BufferManagerPtr, QUERY_INDEX_BUFFER_SIZE);
    keyBuf.Init(*l1BufferManagerPtr, KEY_INDEX_BUFFER_SIZE);

    l0aBufferManager.Init(pipe, L0_MAX_SIZE);
    l0bBufferManager.Init(pipe, L0_MAX_SIZE);
    l0aBuf.Init(l0aBufferManager, L0_SINGLE_BUFFER_SIZE);
    l0bBuf.Init(l0bBufferManager, L0_SINGLE_BUFFER_SIZE);

    l0cBufferManager.Init(pipe, L0C_MAX_SIZE);
    commonL0CBuf.Init(l0cBufferManager, L0C_SINGLE_BUFFER_SIZE);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseCubeBlock<TEMPLATE_ARGS>::InitGlobalBuffer(
    GM_ADDR queryIndex, GM_ADDR keyIndex, GlobalTensor<T> &mm1ResGm) {
    queryIndexGm.SetGlobalBuffer((__gm__ INPUT_T *)queryIndex);
    keyIndexGm.SetGlobalBuffer((__gm__ INPUT_T *)keyIndex);
    this->mm1ResGm = mm1ResGm;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseCubeBlock<TEMPLATE_ARGS>::CopyGmToL1(
    LocalTensor<INPUT_T> &l1Tensor, GlobalTensor<INPUT_T> &gmSrcTensor,
    uint32_t srcN, uint32_t srcD, uint32_t srcDstride) {
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
__aicore__ inline void DenseLISoftmaxLseCubeBlock<TEMPLATE_ARGS>::CopyInQueryIndexToL1(
    LocalTensor<INPUT_T> &l1Tensor, const DenseLISoftmaxLseRunInfo &info,
    DenseLISoftmaxLseConstInfo &constInfo) {
    auto srcGm = queryIndexGm[info.queryIndexTensorOffset];
    CopyGmToL1(l1Tensor, srcGm, constInfo.gSizeQueryIndex, constInfo.dSizeQueryIndex, constInfo.dSizeQueryIndex);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseCubeBlock<TEMPLATE_ARGS>::CopyInKeyIndexToL1(
    LocalTensor<INPUT_T> &l1Tensor, const DenseLISoftmaxLseRunInfo &info,
    DenseLISoftmaxLseConstInfo &constInfo) {
    auto srcGm = keyIndexGm[info.keyIndexTensorOffset];
    CopyGmToL1(l1Tensor, srcGm, constInfo.gSizeQueryIndex, constInfo.dSizeQueryIndex, constInfo.dSizeQueryIndex);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DenseLISoftmaxLseCubeBlock<TEMPLATE_ARGS>::ComputeMm1(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &mm1ResBuf,
    DenseLISoftmaxLseRunInfo &runInfo,
    DenseLISoftmaxLseConstInfo &constInfo) {
    LocalTensor<INPUT_T> queryL1Tensor = queryBuf.GetTensor<INPUT_T>();
    LocalTensor<INPUT_T> keyL1Tensor = keyBuf.GetTensor<INPUT_T>();
    LocalTensor<T> mm1L0CTensor = commonL0CBuf.GetTensor<T>();
    LocalTensor<T> mm1UbTensor = mm1ResBuf.GetTensor<T>();

    CopyInQueryIndexToL1(queryL1Tensor, runInfo, constInfo);
    CopyInKeyIndexToL1(keyL1Tensor, runInfo, constInfo);

    MMParam mmParam = {
        constInfo.gSizeQueryIndex,
        N_SPLIT_SIZE,
        constInfo.dSizeQueryIndex,
        false,
        true,
        true,
        true,
        UNITFLAG_DISABLE
    };

    MatmulBase<INPUT_T, INPUT_T, T, M_SPLIT_SIZE, N_SPLIT_SIZE, K_SPLIT_SIZE,
               ABLayout::MK, ABLayout::KN>(
        queryL1Tensor, keyL1Tensor, l0aBuf, l0bBuf, mm1L0CTensor, mmParam);

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = AlignTo(runInfo.s2RealSize, FP32_ALIGN_NUM);
    fixpipeParams.mSize = AlignTo(runInfo.nRealSize, FP32_ALIGN_NUM);
    fixpipeParams.reluMode = 1;

    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(mm1UbTensor, mm1L0CTensor, fixpipeParams);
}

TEMPLATES_DEF
class DenseLISoftmaxLseCubeBlockDummy {
public:
    __aicore__ inline DenseLISoftmaxLseCubeBlockDummy(){};
    __aicore__ inline void SetCubeBlockParams(TPipe *tPipe, BufferManager<BufferType::L1> *l1BuffMgr) {};
    __aicore__ inline void InitCubeBuffers() {};
    __aicore__ inline void InitGlobalBuffer(GM_ADDR queryIndex, GM_ADDR keyIndex, GlobalTensor<T> &mm1ResGm) {};
    __aicore__ inline void ComputeMm1(
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &mm1ResBuf,
        DenseLISoftmaxLseRunInfo &runInfo,
        DenseLISoftmaxLseConstInfo &constInfo) {};
};

}

#endif