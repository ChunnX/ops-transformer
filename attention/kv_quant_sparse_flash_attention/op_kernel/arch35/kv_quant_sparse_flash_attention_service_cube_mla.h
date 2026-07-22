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
 * \file kv_quant_sparse_flash_attention_service_cube_mla.h
 */
#ifndef KV_QUANT_SPARSE_FLASH_ATTENTION_SERVICE_CUBE_MLA_H
#define KV_QUANT_SPARSE_FLASH_ATTENTION_SERVICE_CUBE_MLA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "kv_quant_sparse_flash_attention_common_arch35.h"
#include "common/matmul_local.h"

#if __has_include("../../common/op_kernel/offset_calculator.h")
#include "../../common/op_kernel/offset_calculator.h"
#else
#include "../common/offset_calculator.h"
#endif

#if __has_include("../../common/op_kernel/matmul.h")
#include "../../common/op_kernel/matmul.h"
#else
#include "../common/matmul.h"
#endif
#if __has_include("../../common/op_kernel/CopyInL1.h")
#include "../../common/op_kernel/CopyInL1.h"
#else
#include "../common/CopyInL1.h"
#endif
#if __has_include("../../common/op_kernel/FixpipeOut.h")
#include "../../common/op_kernel/FixpipeOut.h"
#else
#include "../common/FixpipeOut.h"
#endif

using namespace AscendC;
using namespace AscendC::Impl::Detail;

using namespace fa_base_matmul;
using namespace regbaseutil;

namespace BaseApi {

template <QSFA_LAYOUT LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == QSFA_LAYOUT::BSND) {
        return GmFormat::BSNGD;
    } else {
        return GmFormat::TNGD;
    }
}

TEMPLATES_DEF
class QSFAMatmulService {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t dBaseSize = 576;
    static constexpr uint32_t dBaseMatmulSize = 128;
    static constexpr uint32_t l1QBufNum = 3;
    static constexpr uint32_t l1KBufNum = 6;
    static constexpr uint32_t rightBufNum = 3;
    static constexpr uint32_t rightBufSingleSize = s2BaseSize * dBaseSize;
    static constexpr uint32_t rightBufTotalSize = rightBufSingleSize * rightBufNum;

    __aicore__ inline QSFAMatmulService() {};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, __gm__ uint8_t *query);
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *cuSeqlensQ, const ConstInfo& constInfo);
    __aicore__ inline void IterateLoadQK(
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        RunInfo &runInfo, ConstInfo &constInfo, bool isFirstLoop);
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &output,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        bool notLastTwoLoop, RunInfo &runInfoNext,
        RunInfo &runInfo, ConstInfo &constInfo);

    __aicore__ inline void IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
        RunInfo &runInfo, ConstInfo &constInfo);

private:
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitGmTensor(__gm__ uint8_t *cuSeqlensQ, const ConstInfo& constInfo);

    __aicore__ inline void CopyQGmToL1(RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void IterateBmm1QSFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        bool notLastTwoLoop, RunInfo &runInfoNext,
        RunInfo &runInfo, ConstInfo &constInfo);

    // --------------------Bmm2--------------------------
    __aicore__ inline void IterateBmm2QSFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
        RunInfo &runInfo, ConstInfo &constInfo);
    TPipe *tPipe;
    /* =====================GM变量==================== */
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    FaGmTensor<Q_T, Q_FORMAT, int32_t> queryGm;

    /* =====================运行时变量==================== */
    uint32_t l0ABBufIdx = 0;
    uint32_t l0CBufIdx = 0;
    uint32_t l1QBufIdx = 0;
    uint32_t l1KLoadBufIdx = 0;
    uint32_t l1KMatmul1BufIdx = 0;
    uint32_t l1KMatmul2BufIdx = 0;
    uint32_t l0ABBufId = 9;
    uint32_t l0CBufId = 11;
    uint32_t l1QBufId = 0;
    uint32_t l1KBufId = 3;

    /* =====================LocalBuffer变量==================== */
    TBuf<TPosition::A1> l1QBuffers;
    TBuf<TPosition::B1> l1RightBuffers;
    TBuf<TPosition::A2> mmL0ABuffers;
    TBuf<TPosition::B2> mmL0BBuffers;
    TBuf<TPosition::CO1> mmL0CBuffers;
    LocalTensor<Q_T> l1RightTensor;
    LocalTensor<Q_T> l1QTensor;
    LocalTensor<Q_T> mmL0ATensor;
    LocalTensor<Q_T> mmL0BTensor;
    LocalTensor<T> mmL0CTensor;
};

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAMatmulService<TEMPLATE_ARGS>::InitCubeBlock(
    TPipe *pipe, __gm__ uint8_t *query)
{
    if ASCEND_IS_AIC {
        tPipe = pipe;
        this->queryGm.gmTensor.SetGlobalBuffer((__gm__ Q_T *)query);
        InitLocalBuffer();
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
QSFAMatmulService<TEMPLATE_ARGS>::InitCubeInput(__gm__ uint8_t *qsfaActualSeqLengthsQ, const ConstInfo& constInfo)
{
    if ASCEND_IS_AIC {
        InitGmTensor(qsfaActualSeqLengthsQ, constInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
QSFAMatmulService<TEMPLATE_ARGS>::InitLocalBuffer()
{
    tPipe->InitBuffer(l1RightBuffers, rightBufTotalSize * sizeof(Q_T));
    l1RightTensor = l1RightBuffers.Get<Q_T>();

    tPipe->InitBuffer(l1QBuffers, BUFFER_SIZE_96K);
    l1QTensor = l1QBuffers.Get<Q_T>();

    tPipe->InitBuffer(mmL0ABuffers, BUFFER_SIZE_32K);
    tPipe->InitBuffer(mmL0BBuffers, BUFFER_SIZE_64K);
    tPipe->InitBuffer(mmL0CBuffers, BUFFER_SIZE_256K);
    mmL0ATensor = mmL0ABuffers.Get<Q_T>();
    mmL0BTensor = mmL0BBuffers.Get<Q_T>();
    mmL0CTensor = mmL0CBuffers.Get<T>();

    SetFlag<HardEvent::M_MTE1>(l0ABBufId);
    SetFlag<HardEvent::M_MTE1>(l0ABBufId + 1);
    SetFlag<HardEvent::FIX_M>(l0CBufId);
    SetFlag<HardEvent::FIX_M>(l0CBufId + 1);
    SetFlag<HardEvent::MTE1_MTE2>(l1QBufId);
    SetFlag<HardEvent::MTE1_MTE2>(l1QBufId + 1);
    SetFlag<HardEvent::MTE1_MTE2>(l1QBufId + 2);
    SetFlag<HardEvent::MTE1_MTE2>(l1KBufId);
    SetFlag<HardEvent::MTE1_MTE2>(l1KBufId + 1);
    SetFlag<HardEvent::MTE1_MTE2>(l1KBufId + 2);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
QSFAMatmulService<TEMPLATE_ARGS>::InitGmTensor(__gm__ uint8_t *qsfaActualSeqLengthsQ, const ConstInfo& constInfo)
{
    if constexpr (LAYOUT_T == QSFA_LAYOUT::BSND) {
        this->queryGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize,
            constInfo.s1Size, constInfo.dSize);
    } else {  // QSFA_LAYOUT::TND
        GlobalTensor<int32_t> actualSeqQLen;
        actualSeqQLen.SetGlobalBuffer((__gm__ int32_t *)qsfaActualSeqLengthsQ);
        this->queryGm.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSize,
            actualSeqQLen, constInfo.actualSeqLenSize);
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAMatmulService<TEMPLATE_ARGS>::CopyQGmToL1(RunInfo &runInfo, ConstInfo &constInfo)
{
    uint64_t gmOffset = this->queryGm.offsetCalculator.GetOffset(runInfo.boIdx, runInfo.n2oIdx, runInfo.goIdx,
        runInfo.s1oIdx * runInfo.qSNumInOneBlock, 0);
    for (uint32_t i = 0; i < 2; i++) {
        uint32_t curL1QBufId = (l1QBufIdx + i) % 3;
        WaitFlag<HardEvent::MTE1_MTE2>(l1QBufId + curL1QBufId);
        uint64_t curGmOffset = gmOffset + i * (constInfo.dSize >> 1);
        CopyToL1Nd2Nz<Q_T>(l1QTensor[curL1QBufId * BUFFER_SIZE_16K],
            this->queryGm.gmTensor[curGmOffset], runInfo.mRealSize, constInfo.dSize >> 1,
            constInfo.mm1Ka);
        SetFlag<HardEvent::MTE2_MTE1>(l1QBufId + curL1QBufId);
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAMatmulService<TEMPLATE_ARGS>::IterateLoadQK(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    RunInfo &runInfo, ConstInfo &constInfo, bool isFirstLoop)
{
    if (unlikely(isFirstLoop)) {
        CopyQGmToL1(runInfo, constInfo);
    }

    WaitFlag<HardEvent::MTE1_MTE2>(l1KBufId + l1KLoadBufIdx);
    WaitFlag<HardEvent::MTE1_MTE2>(l1KBufId + l1KLoadBufIdx + 1);

    LocalTensor<Q_T> dst = l1RightTensor[runInfo.taskIdMod3 * rightBufSingleSize];
    v0ResGm.WaitCrossCore();
    if constexpr (IS_SPLIT_G) {
        CrossCoreSetFlag<QSFA_SYNC_MODE0, PIPE_MTE2>(15);
        CrossCoreWaitFlag<QSFA_SYNC_MODE0, PIPE_MTE2>(15);
    }
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    DataCopy(dst, v0ResGmTensor, Align16Func(runInfo.s2RealSize) * (constInfo.dSize >> 1));
    SetFlag<HardEvent::MTE2_MTE1>(l1KBufId + l1KLoadBufIdx);
    l1KLoadBufIdx = (l1KLoadBufIdx + 1) % l1KBufNum;

    DataCopy(dst[Align16Func(runInfo.s2RealSize) * (constInfo.dSize >> 1)],
        v0ResGmTensor[Align16Func(runInfo.s2RealSize) * (constInfo.dSize >> 1)],
        Align16Func(runInfo.s2RealSize) * (constInfo.dSize >> 1));
    SetFlag<HardEvent::MTE2_MTE1>(l1KBufId + l1KLoadBufIdx);
    l1KLoadBufIdx = (l1KLoadBufIdx + 1) % l1KBufNum;
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAMatmulService<TEMPLATE_ARGS>::IterateBmm1(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    bool notLastTwoLoop, RunInfo &runInfoNext,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    IterateBmm1QSFA(outputBuf, v0ResGm, notLastTwoLoop, runInfoNext, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAMatmulService<TEMPLATE_ARGS>::IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    IterateBmm2QSFA(outputBuf, inputLeftBuffers, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAMatmulService<TEMPLATE_ARGS>::IterateBmm1QSFA(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    bool notLastTwoLoop, RunInfo &runInfoNext,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    WaitFlag<HardEvent::MTE2_MTE1>(l1KBufId + l1KMatmul1BufIdx);
    l1KMatmul1BufIdx = (l1KMatmul1BufIdx + 1) % l1KBufNum;
    WaitFlag<HardEvent::FIX_M>(l0CBufId + l0CBufIdx);

    MMParam param = {static_cast<uint32_t>(runInfo.mRealSize),     // singleM
                     static_cast<uint32_t>(runInfo.s2RealSize),  // singleN
                     static_cast<uint32_t>(constInfo.dSize),   // singleK
                     0,    // isLeftTranspose
                     1     // isRightTranspose
                    };
    uint32_t curL1QBufId = l1QBufIdx;
    if (unlikely(runInfo.s2LoopCount == 0)) {
        WaitFlag<HardEvent::MTE2_MTE1>(l1QBufId + curL1QBufId);
    }

    LocalTensor<Q_T> curL1RightTensor = l1RightTensor[runInfo.taskIdMod3 * rightBufSingleSize];
    MatmulKFull<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        l1QTensor[curL1QBufId * BUFFER_SIZE_16K],
        curL1RightTensor,
        mmL0ATensor, mmL0BTensor, mmL0CTensor[BUFFER_SIZE_32K * l0CBufIdx],
        param, l0ABBufId, l0ABBufIdx);

    l1KMatmul1BufIdx = (l1KMatmul1BufIdx + 1) % l1KBufNum;
    WaitFlag<HardEvent::MTE2_MTE1>(l1KBufId + l1KMatmul1BufIdx);
    l1KMatmul1BufIdx = (l1KMatmul1BufIdx + 1) % l1KBufNum;

    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        SetFlag<HardEvent::MTE1_MTE2>(l1QBufId + l1QBufIdx);
        l1QBufIdx = (l1QBufIdx + 1) % l1QBufNum;
        if (notLastTwoLoop) {
            CopyQGmToL1(runInfoNext, constInfo);
        }
    }

    SetFlag<HardEvent::M_FIX>(l0CBufId + l0CBufIdx);
    WaitFlag<HardEvent::M_FIX>(l0CBufId + l0CBufIdx);

    outputBuf.WaitCrossCore();
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.mSize = Align2Func(runInfo.mRealSize);
    fixpipeParams.nSize = Align8Func(runInfo.s2RealSize);
    fixpipeParams.srcStride = Align16Func(fixpipeParams.mSize);
    fixpipeParams.dstStride = s2BaseSize;
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    fixpipeParams.params.ndNum = 1;

    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(),
        mmL0CTensor[BUFFER_SIZE_32K * l0CBufIdx], fixpipeParams);
    SetFlag<HardEvent::FIX_M>(l0CBufId + l0CBufIdx);
    l0CBufIdx ^= 1;
    outputBuf.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAMatmulService<TEMPLATE_ARGS>::IterateBmm2QSFA(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    WaitFlag<HardEvent::FIX_M>(l0CBufId + l0CBufIdx);

    MMParam qsfaParam = {static_cast<uint32_t>(runInfo.mRealSize),   // singleM 64
                         static_cast<uint32_t>(constInfo.dSizeNope), // singleN 512
                         static_cast<uint32_t>(runInfo.s2RealSize),  // singleK 128
                         0, 0};

    LocalTensor<Q_T> curL1RightTensor = l1RightTensor[runInfo.taskIdMod3 * rightBufSingleSize];
    MatmulN1<Q_T, Q_T, T, s1BaseSize, dBaseMatmulSize, s2BaseSize, ABLayout::MK, ABLayout::KN>(
        curL1RightTensor[s2BaseSize * constInfo.dSizeNope],
        curL1RightTensor,
        mmL0ATensor, mmL0BTensor, mmL0CTensor[BUFFER_SIZE_32K * l0CBufIdx],
        qsfaParam, l0ABBufId, l0ABBufIdx);

    SetFlag<HardEvent::M_FIX>(l0CBufId + l0CBufIdx);
    WaitFlag<HardEvent::M_FIX>(l0CBufId + l0CBufIdx);

    SetFlag<HardEvent::MTE1_MTE2>(l1KBufId + l1KMatmul2BufIdx);
    l1KMatmul2BufIdx = (l1KMatmul2BufIdx + 1) % l1KBufNum;
    SetFlag<HardEvent::MTE1_MTE2>(l1KBufId + l1KMatmul2BufIdx);
    l1KMatmul2BufIdx = (l1KMatmul2BufIdx + 1) % l1KBufNum;

    outputBuf.WaitCrossCore();

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.mSize = Align2Func(runInfo.mRealSize);
    fixpipeParams.nSize = Align8Func(constInfo.dSizeNope);
    fixpipeParams.srcStride = Align16Func(fixpipeParams.mSize);
    fixpipeParams.dstStride = Align16Func(constInfo.dSizeNope);
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    fixpipeParams.params.ndNum = 1;

    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(),
        mmL0CTensor[BUFFER_SIZE_32K * l0CBufIdx], fixpipeParams);
    SetFlag<HardEvent::FIX_M>(l0CBufId + l0CBufIdx);
    l0CBufIdx ^= 1;

    outputBuf.SetCrossCore();
}

TEMPLATES_DEF
class QSFAMatmulServiceDummy {
public:
    __aicore__ inline QSFAMatmulServiceDummy() {};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, __gm__ uint8_t *query) {}
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *cuSeqlensQ, const ConstInfo& constInfo) {}
    __aicore__ inline void IterateLoadQK(
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        RunInfo &runInfo, ConstInfo &constInfo, bool isFirstLoop) {}
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        bool notLastTwoLoop, RunInfo &runInfoNext,
        RunInfo &runInfo, ConstInfo &constInfo) {}

    __aicore__ inline void IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
        RunInfo &runInfo, ConstInfo &constInfo) {}
};

template <typename T>
struct CubeBlockTraits;  // 声明
/* 生成CubeBlockTraits */
#define GEN_TRAIT_TYPE(name, ...) using name##_TRAITS = name;
#define GEN_TRAIT_CONST(name, type, ...) static constexpr type name##Traits = name;

#define DEFINE_QSFA_CUBE_BLOCK_TRAITS(CUBE_BLOCK_CLASS) \
    TEMPLATES_DEF_NO_DEFAULT \
    struct CubeBlockTraits<CUBE_BLOCK_CLASS<TEMPLATE_ARGS>> { \
        QSFA_CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TRAIT_TYPE) \
        QSFA_CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TRAIT_CONST) \
    }

DEFINE_QSFA_CUBE_BLOCK_TRAITS(QSFAMatmulService);
DEFINE_QSFA_CUBE_BLOCK_TRAITS(QSFAMatmulServiceDummy);

// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS \
    QSFA_CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE) \
    QSFA_CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)
}
#endif // KV_QUANT_SPARSE_FLASH_ATTENTION_SERVICE_CUBE_MLA_H