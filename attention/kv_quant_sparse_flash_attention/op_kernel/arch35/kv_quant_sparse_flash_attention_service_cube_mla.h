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

    __aicore__ inline QSFAMatmulService() {};
    __aicore__ inline void InitLocalBuffer(TPipe *pipe);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *query, __gm__ uint8_t *cuSeqlensQ,
        const ConstInfo& constInfo);
    __aicore__ inline void IterateLoadQK(
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        RunInfo &runInfo, ConstInfo &constInfo, bool isFirstLoop);
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &output,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        bool notLastTwoLoop, RunInfo &runInfoNext,
        RunInfo &runInfo, ConstInfo &constInfo);

    __aicore__ inline void IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo,
        ConstInfo &constInfo);

private:
    __aicore__ inline void InitGmTensor(__gm__ uint8_t *cuSeqlensQ, const ConstInfo& constInfo);

    __aicore__ inline void CopyQGmToL1(RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void IterateBmm1QSFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        bool notLastTwoLoop, RunInfo &runInfoNext,
        RunInfo &runInfo, ConstInfo &constInfo);

    // --------------------Bmm2--------------------------
    __aicore__ inline void IterateBmm2QSFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo,
        ConstInfo &constInfo);
    /* =====================GM变量==================== */
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    FaGmTensor<Q_T, Q_FORMAT, int32_t> queryGm;

    /* =====================运行时变量==================== */
    uint32_t l0ABBufIdx = 0;
    uint32_t l0CBufIdx = 0;
    uint32_t l1KLoadBufIdx = 0;
    uint32_t l1KMatmul1BufIdx = 0;
    uint32_t l1KMatmul2BufIdx = 0;
    uint32_t l0ABBufId = 3;
    uint32_t l0CBufId = 0;
    uint32_t l1QBufId = 0;
    uint32_t l1KBufId = 1;

    /* =====================LocalBuffer变量==================== */
    TBuf<TPosition::A1> l1QBuffers;
    TBuf<TPosition::A2> mmL0ABuffers;
    TBuf<TPosition::B2> mmL0BBuffers;
    TBuf<TPosition::CO1> mmL0CBuffers;
    LocalTensor<Q_T> l1QTensor;
    LocalTensor<Q_T> mmL0ATensor;
    LocalTensor<Q_T> mmL0BTensor;
    LocalTensor<T> mmL0CTensor;
};

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAMatmulService<TEMPLATE_ARGS>::InitLocalBuffer(TPipe *pipe)
{
    if ASCEND_IS_AIC {
        pipe->InitBuffer(l1QBuffers, s1BaseSize * dBaseSize * sizeof(Q_T));
        l1QTensor = l1QBuffers.Get<Q_T>();

        pipe->InitBuffer(mmL0ABuffers, BUFFER_SIZE_64K);
        pipe->InitBuffer(mmL0BBuffers, BUFFER_SIZE_64K);
        pipe->InitBuffer(mmL0CBuffers, BUFFER_SIZE_256K);
        mmL0ATensor = mmL0ABuffers.Get<Q_T>();
        mmL0BTensor = mmL0BBuffers.Get<Q_T>();
        mmL0CTensor = mmL0CBuffers.Get<T>();

        SetFlag<HardEvent::M_MTE1>(l0ABBufId);
        SetFlag<HardEvent::M_MTE1>(l0ABBufId + 1);
        SetFlag<HardEvent::FIX_M>(l0CBufId);
        SetFlag<HardEvent::FIX_M>(l0CBufId + 1);
        SetFlag<HardEvent::MTE1_MTE2>(l1QBufId);
        SetFlag<HardEvent::MTE1_MTE2>(l1KBufId);
        SetFlag<HardEvent::MTE1_MTE2>(l1KBufId + 1);
        SetFlag<HardEvent::MTE1_MTE2>(l1KBufId + 2);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
QSFAMatmulService<TEMPLATE_ARGS>::InitGlobalBuffer(__gm__ uint8_t *query, __gm__ uint8_t *qsfaActualSeqLengthsQ,
    const ConstInfo& constInfo)
{
    if ASCEND_IS_AIC {
        this->queryGm.gmTensor.SetGlobalBuffer((__gm__ Q_T *)query);
        InitGmTensor(qsfaActualSeqLengthsQ, constInfo);
    }
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
    WaitFlag<HardEvent::MTE1_MTE2>(l1QBufId);
    CopyToL1Nd2Nz<Q_T>(l1QTensor, this->queryGm.gmTensor[gmOffset], runInfo.mRealSize, constInfo.dSize,
        constInfo.mm1Ka);
    SetFlag<HardEvent::MTE2_MTE1>(l1QBufId);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAMatmulService<TEMPLATE_ARGS>::IterateLoadQK(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    RunInfo &runInfo, ConstInfo &constInfo, bool isFirstLoop)
{
    if (unlikely(isFirstLoop)) {
        CopyQGmToL1(runInfo, constInfo);
    }

    WaitFlag<HardEvent::MTE1_MTE2>(l1KBufId + l1KLoadBufIdx);

    LocalTensor<Q_T> dst = inputRightBuf.GetTensor<Q_T>();
    v0ResGm.WaitCrossCore();
    if constexpr (IS_SPLIT_G) {
        CrossCoreSetFlag<QSFA_SYNC_MODE0, PIPE_MTE2>(15);
        CrossCoreWaitFlag<QSFA_SYNC_MODE0, PIPE_MTE2>(15);
    }
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    DataCopy(dst, v0ResGmTensor, Align16Func(runInfo.s2RealSize) * constInfo.dSize);

    SetFlag<HardEvent::MTE2_MTE1>(l1KBufId + l1KLoadBufIdx);
    l1KLoadBufIdx = (l1KLoadBufIdx + 1) % 3;
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAMatmulService<TEMPLATE_ARGS>::IterateBmm1(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    bool notLastTwoLoop, RunInfo &runInfoNext,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    IterateBmm1QSFA(outputBuf, inputRightBuf, v0ResGm, notLastTwoLoop, runInfoNext, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAMatmulService<TEMPLATE_ARGS>::IterateBmm2(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    IterateBmm2QSFA(outputBuf, inputLeftBuffers, inputRightBuf, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAMatmulService<TEMPLATE_ARGS>::IterateBmm1QSFA(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    bool notLastTwoLoop, RunInfo &runInfoNext,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    WaitFlag<HardEvent::MTE2_MTE1>(l1KBufId + l1KMatmul1BufIdx);
    l1KMatmul1BufIdx = (l1KMatmul1BufIdx + 1) % 3;
    WaitFlag<HardEvent::FIX_M>(l0CBufId + l0CBufIdx);

    MMParam param = {static_cast<uint32_t>(runInfo.mRealSize),     // singleM
                     static_cast<uint32_t>(runInfo.s2RealSize),  // singleN
                     static_cast<uint32_t>(constInfo.dSize),   // singleK
                     0,    // isLeftTranspose
                     1     // isRightTranspose
                    };
    if (unlikely(runInfo.s2LoopCount == 0)) {
        WaitFlag<HardEvent::MTE2_MTE1>(l1QBufId);
    }

    LocalTensor<Q_T> curL1RightTensor = inputRightBuf.GetTensor<Q_T>();
    MatmulKFull<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        l1QTensor,
        curL1RightTensor,
        mmL0ATensor, mmL0BTensor, mmL0CTensor[BUFFER_SIZE_32K * l0CBufIdx],
        param, l0ABBufId, l0ABBufIdx);

    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        SetFlag<HardEvent::MTE1_MTE2>(l1QBufId);
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
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    inputRightBuf.WaitCrossCore();
    WaitFlag<HardEvent::FIX_M>(l0CBufId + l0CBufIdx);

    MMParam qsfaParam = {static_cast<uint32_t>(runInfo.mRealSize),   // singleM 64
                         static_cast<uint32_t>(constInfo.dSizeNope), // singleN 512
                         static_cast<uint32_t>(runInfo.s2RealSize),  // singleK 128
                         0, 0};

    MatmulN1<Q_T, Q_T, T, s1BaseSize, dBaseMatmulSize, s2BaseSize, ABLayout::MK, ABLayout::KN>(
        inputRightBuf.GetTensor<Q_T>(s2BaseSize * constInfo.dSizeNope),
        inputRightBuf.GetTensor<Q_T>(),
        mmL0ATensor, mmL0BTensor, mmL0CTensor[BUFFER_SIZE_32K * l0CBufIdx],
        qsfaParam, l0ABBufId, l0ABBufIdx);

    inputRightBuf.SetCrossCore();
    SetFlag<HardEvent::M_FIX>(l0CBufId + l0CBufIdx);
    WaitFlag<HardEvent::M_FIX>(l0CBufId + l0CBufIdx);

    SetFlag<HardEvent::MTE1_MTE2>(l1KBufId + l1KMatmul2BufIdx);
    l1KMatmul2BufIdx = (l1KMatmul2BufIdx + 1) % 3;

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
    __aicore__ inline void InitLocalBuffer(TPipe *pipe) {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *query, __gm__ uint8_t *cuSeqlensQ,
        const ConstInfo& constInfo) {}
    __aicore__ inline void IterateLoadQK(
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        RunInfo &runInfo, ConstInfo &constInfo, bool isFirstLoop) {}
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        bool notLastTwoLoop, RunInfo &runInfoNext,
        RunInfo &runInfo, ConstInfo &constInfo) {}

    __aicore__ inline void IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo,
        ConstInfo &constInfo) {}
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