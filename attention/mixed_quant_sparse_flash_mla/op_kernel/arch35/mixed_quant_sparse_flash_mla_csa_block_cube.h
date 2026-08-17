/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file mixed_quant_sparse_flash_mla_csa_block_cube.h
 * \brief
 */
#ifndef MIXED_QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_CUBE_H
#define MIXED_QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_CUBE_H
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
#if __has_include("../../common/op_kernel/FixpipeOut.h")
#include "../../common/op_kernel/FixpipeOut.h"
#else
#include "../common/FixpipeOut.h"
#endif
#if __has_include("../../common/op_kernel/CopyInL1.h")
#include "../../common/op_kernel/CopyInL1.h"
#else
#include "../common/CopyInL1.h"
#endif

#include "kernel_operator_list_tensor_intf.h"
#include "util_regbase.h"
#include "mixed_quant_sparse_flash_mla_common_arch35.h"
#if __has_include("../../../sparse_flash_mla/op_kernel/arch35/common/matmul_static.h")
#include "../../../sparse_flash_mla/op_kernel/arch35/common/matmul_static.h"
#else
#include "../../sparse_flash_mla/arch35/common/matmul_static.h"
#endif

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
namespace BaseApi {
template <QSMLA_LAYOUT LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == QSMLA_LAYOUT::BSND) {
        return GmFormat::BSNGD;
    } else {
        return GmFormat::TNGD;
    }
}

TEMPLATES_DEF
class CSABlockCube {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t dBaseSize = 512;
    static constexpr uint32_t dBaseMatmulSize = 128;

    __aicore__ inline CSABlockCube(){};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, __gm__ uint8_t *query, uint32_t l1CubeBase);
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
                                         const ConstInfo &constInfo);
    __aicore__ inline void IterateLoadQK(StaticBuffer<Q_T> &inputRightBuf,
                                         Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                         RunInfo &runInfo, ConstInfo &constInfo, bool isFirstLoop);
    __aicore__ inline void IterateBmm1(StaticBuffer<T> &output,
                                       StaticBuffer<Q_T> &inputRightBuf,
                                       Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                       bool notLastTwoLoop, RunInfo &runInfoNext, RunInfo &runInfo, ConstInfo &constInfo);

    __aicore__ inline void IterateBmm2(StaticBuffer<T> &outputBuf,
                                       StaticBuffer<Q_T> &l1PBuffer,
                                       StaticBuffer<Q_T> &inputRightBuf, RunInfo &runInfo,
                                       ConstInfo &constInfo);

private:
    __aicore__ inline void InitLocalBuffer(uint32_t l1CubeBase);
    __aicore__ inline void InitGmTensor(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
                                        const ConstInfo &constInfo);
    __aicore__ inline void CopyQGmToL1(RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void IterateBmm1CSA(StaticBuffer<T> &outputBuf,
                                          StaticBuffer<Q_T> &inputRightBuf,
                                          Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                          bool notLastTwoLoop, RunInfo &runInfoNext, RunInfo &runInfo, ConstInfo &constInfo);

    // --------------------Bmm2--------------------------
    __aicore__ inline void IterateBmm2CSA(StaticBuffer<T> &outputBuf,
                                          StaticBuffer<Q_T> &l1PBuffer,
                                          StaticBuffer<Q_T> &inputRightBuf, RunInfo &runInfo,
                                          ConstInfo &constInfo);
    TPipe *tPipe;
    /* =====================GM变量==================== */
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    static constexpr bool Q_WITH_ZERO_HEAD = (LAYOUT_T == QSMLA_LAYOUT::TND);
    FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_WITH_ZERO_HEAD> queryGm;

    /* =====================运行时变量==================== */
    uint32_t l1QBufId = 0;
    uint32_t l1KLoadBufId = 0;
    uint32_t l1KMatmul1BufId = 0;
    uint32_t l1KMatmul2BufId = 0;

    TEventID l0aMte1ToMId[2];
    TEventID l0bMte1ToMId[2];
    TEventID l0bMToMte1Id[2];

    /* =====================LocalBuffer变量==================== */
    StaticBuffer<Q_T> l1QBufs[3];
    StaticBuffer<Q_T> l0ABufs[2];
    RingBuffer<Q_T>   l0A;
    StaticBuffer<Q_T> l0BBufs[2];
    RingBuffer<Q_T>   l0B;
    StaticBuffer<T>   l0CBufs[2];
    RingBuffer<T>     l0C;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::InitCubeBlock(
    TPipe *pipe, __gm__ uint8_t *query, uint32_t l1CubeBase)
{
    if ASCEND_IS_AIC {
        tPipe = pipe;
        this->queryGm.gmTensor.SetGlobalBuffer((__gm__ Q_T *)query);
        InitLocalBuffer(l1CubeBase);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::InitCubeInput(
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ, const ConstInfo &constInfo)
{
    if ASCEND_IS_AIC {
        InitGmTensor(cuSeqlensQ, sequsedQ, constInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::InitLocalBuffer(uint32_t l1CubeBase)
{
    uint32_t l1Addr = l1CubeBase;
    for (uint32_t i = 0; i < 3; i++) {
        l1QBufs[i] = {LocalTensor<Q_T>(TPosition::A1, l1Addr, L1Q_ELEM_PER_BUF), i};
        l1Addr += L1Q_ELEM_PER_BUF * sizeof(Q_T);
    }

    uint32_t l0aAddr = 0;
    l0ABufs[0] = {LocalTensor<Q_T>(TPosition::A2, l0aAddr, L0A_ELEM_PER_BUF), 0};
    l0aAddr += L0A_ELEM_PER_BUF * sizeof(Q_T);
    l0ABufs[1] = {LocalTensor<Q_T>(TPosition::A2, l0aAddr, L0A_ELEM_PER_BUF), 1};
    l0aAddr += L0A_ELEM_PER_BUF * sizeof(Q_T);

    uint32_t l0bAddr = 0;
    l0BBufs[0] = {LocalTensor<Q_T>(TPosition::B2, l0bAddr, L0B_ELEM_PER_BUF), 0};
    l0bAddr += L0B_ELEM_PER_BUF * sizeof(Q_T);
    l0BBufs[1] = {LocalTensor<Q_T>(TPosition::B2, l0bAddr, L0B_ELEM_PER_BUF), 1};
    l0bAddr += L0B_ELEM_PER_BUF * sizeof(Q_T);

    uint32_t l0cAddr = 0;
    l0CBufs[0] = {LocalTensor<T>(TPosition::CO1, l0cAddr, L0C_ELEM_PER_BUF), 0};
    l0cAddr += L0C_ELEM_PER_BUF * sizeof(T);
    l0CBufs[1] = {LocalTensor<T>(TPosition::CO1, l0cAddr, L0C_ELEM_PER_BUF), 1};
    l0cAddr += L0C_ELEM_PER_BUF * sizeof(T);

    l0A = RingBuffer<Q_T>(l0ABufs, 2);
    l0B = RingBuffer<Q_T>(l0BBufs, 2);
    l0C = RingBuffer<T>(l0CBufs, 2);

    SetFlag<HardEvent::FIX_M>(CUBE_L0C_FIXM(0));
    SetFlag<HardEvent::FIX_M>(CUBE_L0C_FIXM(1));
    SetFlag<HardEvent::MTE1_MTE2>(CUBE_L1Q_MTE1M(0));
    SetFlag<HardEvent::MTE1_MTE2>(CUBE_L1Q_MTE1M(1));
    SetFlag<HardEvent::MTE1_MTE2>(CUBE_L1Q_MTE1M(2));
    SetFlag<HardEvent::MTE1_MTE2>(CUBE_L1K_MTE1M(0));
    SetFlag<HardEvent::MTE1_MTE2>(CUBE_L1K_MTE1M(1));
    SetFlag<HardEvent::MTE1_MTE2>(CUBE_L1K_MTE1M(2));

    for (uint32_t i = 0; i < 2; i++) {
        l0aMte1ToMId[i] = GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>();
        SetFlag<HardEvent::M_MTE1>(l0aMte1ToMId[i]);
        l0bMte1ToMId[i] = GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>();
        SetFlag<HardEvent::M_MTE1>(l0bMte1ToMId[i]);
        l0bMToMte1Id[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>();
    }
}

/* 初始化GmTensor,设置shape信息并计算strides */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::InitGmTensor(
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ, const ConstInfo &constInfo)
{
    if constexpr (LAYOUT_T == QSMLA_LAYOUT::BSND) {
        this->queryGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize,
                                            constInfo.s1Size, constInfo.dSize);
    } else { // QSMLA_LAYOUT::TND
        uint32_t sequsedQSize = (sequsedQ == nullptr) ? 0 : constInfo.bSize;
        ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true> parser;
        parser.Init(cuSeqlensQ, constInfo.bSize + 1, sequsedQ, sequsedQSize);
        this->queryGm.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSize,
                                            parser);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::IterateBmm1(
    StaticBuffer<T> &outputBuf,
    StaticBuffer<Q_T> &inputRightBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    bool notLastTwoLoop, RunInfo &runInfoNext, RunInfo &runInfo, ConstInfo &constInfo)
{
    IterateBmm1CSA(outputBuf, inputRightBuf, v0ResGm, notLastTwoLoop, runInfoNext, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::IterateBmm2(
    StaticBuffer<T> &outputBuf,
    StaticBuffer<Q_T> &l1PBuffer,
    StaticBuffer<Q_T> &inputRightBuf, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    IterateBmm2CSA(outputBuf, l1PBuffer, inputRightBuf, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::CopyQGmToL1(RunInfo &runInfo, ConstInfo &constInfo)
{
    uint64_t gmOffset = this->queryGm.offsetCalculator.GetOffset(runInfo.boIdx, runInfo.n2oIdx, runInfo.goIdx,
                                                                 runInfo.s1oIdx * runInfo.qSNumInOneBlock, 0);
    for (uint32_t i = 0; i < 2; i++) {
        uint32_t curL1QBufId = (l1QBufId + i) % 3;
        WaitFlag<HardEvent::MTE1_MTE2>(CUBE_L1Q_MTE1M(curL1QBufId));
        uint64_t curGmOffset = gmOffset + i * (constInfo.dSize >> 1);
        CopyToL1Nd2Nz<Q_T>(l1QBufs[curL1QBufId].tensor,
                           this->queryGm.gmTensor[curGmOffset], runInfo.mRealSize, constInfo.dSize >> 1, constInfo.mm1Ka);
        SetFlag<HardEvent::MTE2_MTE1>(CUBE_L1Q_MTE2M(curL1QBufId));
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::IterateLoadQK(
    StaticBuffer<Q_T> &inputRightBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    RunInfo &runInfo, ConstInfo &constInfo, bool isFirstLoop)
{
    if (unlikely(isFirstLoop)) {
        CopyQGmToL1(runInfo, constInfo);
    }
    WaitFlag<HardEvent::MTE1_MTE2>(CUBE_L1K_MTE1M(l1KLoadBufId));
    LocalTensor<Q_T> dst = inputRightBuf.tensor;
    v0ResGm.WaitCrossCore();
    if constexpr (IS_SPLIT_G) {
        CrossCoreSetFlag<0, PIPE_MTE2>(15);
        CrossCoreWaitFlag<0, PIPE_MTE2>(15);
    }
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    DataCopy(dst, v0ResGmTensor, Align16Func(runInfo.s2RealSize) * constInfo.dSize);
    SetFlag<HardEvent::MTE2_MTE1>(CUBE_L1K_MTE2M(l1KLoadBufId));
    l1KLoadBufId = (l1KLoadBufId + 1) % 3;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::IterateBmm1CSA(
    StaticBuffer<T> &outputBuf,
    StaticBuffer<Q_T> &inputRightBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    bool notLastTwoLoop, RunInfo &runInfoNext, RunInfo &runInfo, ConstInfo &constInfo)
{
    WaitFlag<HardEvent::MTE2_MTE1>(CUBE_L1K_MTE2M(l1KMatmul1BufId));
    l1KMatmul1BufId = (l1KMatmul1BufId + 1) % 3;

    StaticBuffer<T> &cBuf = l0C.GetNext();
    WaitFlag<HardEvent::FIX_M>(CUBE_L0C_FIXM(cBuf.idx));

    MMParam param = {
        static_cast<uint32_t>(runInfo.mRealSize),    // singleM
        static_cast<uint32_t>(runInfo.s2RealSize),   // singleN
        static_cast<uint32_t>(constInfo.dSize >> 1), // singleK
        0,                                           // isLeftTranspose
        1                                            // isRightTranspose
    };
    uint32_t curL1QBufId = l1QBufId;
    if (unlikely(runInfo.s2LoopCount == 0)) {
        WaitFlag<HardEvent::MTE2_MTE1>(CUBE_L1Q_MTE2M(curL1QBufId));
    }
    MatmulKStatic<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize>(
        l1QBufs[curL1QBufId].tensor, inputRightBuf.tensor,
        l0A, l0B, cBuf.tensor,
        param, l0aMte1ToMId, l0bMte1ToMId, l0bMToMte1Id);

    curL1QBufId = (curL1QBufId + 1) % 3;
    if (unlikely(runInfo.s2LoopCount == 0)) {
        WaitFlag<HardEvent::MTE2_MTE1>(CUBE_L1Q_MTE2M(curL1QBufId));
    }
    param.singleK = constInfo.dSize - param.singleK;
    param.isOutKFisrt = false;
    MatmulKStatic<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize>(
        l1QBufs[curL1QBufId].tensor,
        inputRightBuf.tensor[(constInfo.dSize >> 1) * Align16Func(runInfo.s2RealSize)],
        l0A, l0B, cBuf.tensor,
        param, l0aMte1ToMId, l0bMte1ToMId, l0bMToMte1Id);
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        SetFlag<HardEvent::MTE1_MTE2>(CUBE_L1Q_MTE1M(l1QBufId));
        SetFlag<HardEvent::MTE1_MTE2>(CUBE_L1Q_MTE1M(curL1QBufId));
        l1QBufId = (l1QBufId + 2) % 3;
        if (notLastTwoLoop) {
            CopyQGmToL1(runInfoNext, constInfo);
        }
    }

    SetFlag<HardEvent::M_FIX>(CUBE_L0C_MFIX(cBuf.idx));
    WaitFlag<HardEvent::M_FIX>(CUBE_L0C_MFIX(cBuf.idx));

    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1(outputBuf.idx));
    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1_D(outputBuf.idx));
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB
    // L0C上的bmm1结果矩阵N方向的size大小; 同mmadParams.n; 为什么要8个元素对齐(32B对齐) // 128
    fixpipeParams.nSize = Align8Func(runInfo.s2RealSize);
    // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
    fixpipeParams.mSize = Align2Func(runInfo.mRealSize);
    // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T) // 源Nz矩阵中相邻大Z排布的起始地址偏移
    fixpipeParams.srcStride = Align16Func(fixpipeParams.mSize);
    fixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element。 // 128:根据比对dump文件得到, ND方案(S1*S2)时脏数据用mask剔除
    fixpipeParams.dualDstCtl = 1;         // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;

    // 将matmul结果从L0C搬运到UB
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.tensor, cBuf.tensor, fixpipeParams);
    SetFlag<HardEvent::FIX_M>(CUBE_L0C_FIXM(cBuf.idx));
    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1(outputBuf.idx));
    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1_D(outputBuf.idx));
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::IterateBmm2CSA(
    StaticBuffer<T> &outputBuf,
    StaticBuffer<Q_T> &l1PBuffer,
    StaticBuffer<Q_T> &inputRightBuf, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(CROSSCORE_L1P(l1PBuffer.idx));
    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(CROSSCORE_L1P_D(l1PBuffer.idx));

    StaticBuffer<T> &cBuf = l0C.GetNext();
    WaitFlag<HardEvent::FIX_M>(CUBE_L0C_FIXM(cBuf.idx));

    MMParam param = {
        static_cast<uint32_t>(runInfo.mRealSize),  // singleM 64
        static_cast<uint32_t>(constInfo.dSizeV),   // singleN 512
        static_cast<uint32_t>(runInfo.s2RealSize), // singleK 128
        0,                                         // isLeftTranspose
        0                                          // isRightTranspose
    };
    MatmulNStatic<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize>(
        l1PBuffer.tensor,
        inputRightBuf.tensor,
        l0A, l0B, cBuf.tensor,
        param, l0aMte1ToMId, l0bMte1ToMId, l0bMToMte1Id);

    SetFlag<HardEvent::M_FIX>(CUBE_L0C_MFIX(cBuf.idx));
    WaitFlag<HardEvent::M_FIX>(CUBE_L0C_MFIX(cBuf.idx));
    SetFlag<HardEvent::MTE1_MTE2>(CUBE_L1K_MTE1M(l1KMatmul2BufId));
    l1KMatmul2BufId = (l1KMatmul2BufId + 1) % 3;

    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2);
    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2_D);
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;      // L0C→UB;FixpipeParamsM300:L0C→UB
    fixpipeParams.nSize = Align8Func(constInfo.dSizeV);         // L0C上的bmm1结果矩阵N方向的size大小, 分档计算且vector2中通过mask筛选出实际有效值
    fixpipeParams.mSize = Align2Func(runInfo.mRealSize);        // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小; 同mmadParams.m
    fixpipeParams.srcStride = Align16Func(fixpipeParams.mSize); // L0C上bmm1结果相邻连续数据片段间隔（前面一个数据块的头与后面数据块的头的间隔）
    fixpipeParams.dstStride = Align16Func(constInfo.dSizeV);
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.tensor, cBuf.tensor, fixpipeParams);
    SetFlag<HardEvent::FIX_M>(CUBE_L0C_FIXM(cBuf.idx));

    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2);
    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2_D);
}

TEMPLATES_DEF
class CSABlockCubeDummy {
public:
    __aicore__ inline CSABlockCubeDummy(){};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, __gm__ uint8_t *query, uint32_t l1CubeBase) {}
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
                                         const ConstInfo &constInfo) {}
};

template <typename T>
struct CubeBlockTraits; // 声明

/* 生成CubeBlockTraits */
#define GEN_TRAIT_TYPE(name, ...) using name##_TRAITS = name;
#define GEN_TRAIT_CONST(name, type, ...) static constexpr type name##Traits = name;

#define DEFINE_CUBE_BLOCK_TRAITS(CUBE_BLOCK_CLASS) \
    TEMPLATES_DEF_NO_DEFAULT \
    struct CubeBlockTraits<CUBE_BLOCK_CLASS<TEMPLATE_ARGS>> { \
        CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TRAIT_TYPE) \
        CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TRAIT_CONST) \
    }

DEFINE_CUBE_BLOCK_TRAITS(CSABlockCube);
DEFINE_CUBE_BLOCK_TRAITS(CSABlockCubeDummy);

// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)
} // namespace BaseApi
#endif // MIXED_QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_CUBE_H
