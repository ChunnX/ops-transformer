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
 * \file kv_quant_sparse_attn_sharedkv_scfa_block_cube.h
 * \brief
 */
#ifndef KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_CUBE_H
#define KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_CUBE_H
#include "common/offset_calculator.h"
#include "common/matmul.h"
#include "common/FixpipeOut.h"
#include "common/CopyInL1.h"
#include "kernel_operator_list_tensor_intf.h"

#include "util_regbase.h"
#include "kv_quant_sparse_attn_sharedkv_common_arch35.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
namespace BaseApi {

template <SAS_LAYOUT LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == SAS_LAYOUT::BSND) {
        return GmFormat::BSNGD;
    } else {
        return GmFormat::TNGD;
    }
}

TEMPLATES_DEF
class SCFABlockCube {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t dBaseSize = 512;
    static constexpr uint32_t dBaseMatmulSize = 128;
    static constexpr uint32_t l1QBufNum = 3;
    static constexpr uint32_t l1KBufNum = 6;
    static constexpr uint32_t rightBufNum = 3;
    static constexpr uint32_t rightBufSingleSize = s2BaseSize * dBaseSize;
    static constexpr uint32_t rightBufTotalSize = rightBufSingleSize * rightBufNum;

    __aicore__ inline SCFABlockCube() {};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, __gm__ uint8_t *query, uint32_t l1BaseAddr);
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *cuSeqlensQ, const ConstInfo &constInfo);
    __aicore__ inline void IterateLoadQK(
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        RunInfo &runInfo, ConstInfo &constInfo, bool isFirstLoop);
    __aicore__ inline void IterateBmm1(StaticBuffer<T> &outputBuf,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        bool notLastTwoLoop, RunInfo &runInfoNext,
        RunInfo &runInfo, ConstInfo &constInfo);

    __aicore__ inline void IterateBmm2(StaticBuffer<T> &outputBuf,
        StaticBuffer<Q_T> &l1PBuffer,
        RunInfo &runInfo, ConstInfo &constInfo);

private:
    __aicore__ inline void InitLocalBuffer(uint32_t l1BaseAddr);
    __aicore__ inline void InitGmTensor(__gm__ uint8_t *cuSeqlensQ, const ConstInfo &constInfo);

    __aicore__ inline void CopyQGmToL1(RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void IterateBmm1SCFA(StaticBuffer<T> &outputBuf,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        bool notLastTwoLoop, RunInfo &runInfoNext,
        RunInfo &runInfo, ConstInfo &constInfo);

    // --------------------Bmm2--------------------------
    __aicore__ inline void IterateBmm2SCFA(StaticBuffer<T> &outputBuf,
        StaticBuffer<Q_T> &l1PBuffer,
        RunInfo &runInfo, ConstInfo &constInfo);
    TPipe *tPipe;
    /* =====================GM变量==================== */
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    FaGmTensor<Q_T, Q_FORMAT> queryGm;

    /* =====================运行时变量==================== */
    uint32_t l1QBufIdx = 0; // 3 buffer,0-2
    uint32_t l1KLoadBufIdx = 0; // 6 buffer,0-5
    uint32_t l1KMatmul1BufIdx = 0;
    uint32_t l1KMatmul2BufIdx = 0;

    /* =====================LocalBuffer变量==================== */
    StaticBuffer<Q_T> l1QBufs[l1QBufNum];
    StaticBuffer<Q_T> l1RightBufs[rightBufNum];
    StaticBuffer<Q_T> l0ABufs[2];
    RingBuffer<Q_T>   l0A;
    StaticBuffer<Q_T> l0BBufs[2];
    RingBuffer<Q_T>   l0B;
    StaticBuffer<T>   l0CBufs[2];
    RingBuffer<T>     l0C;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitCubeBlock(
    TPipe *pipe, __gm__ uint8_t *query, uint32_t l1BaseAddr)
{
    if ASCEND_IS_AIC {
        tPipe = pipe;
        this->queryGm.gmTensor.SetGlobalBuffer((__gm__ Q_T *)query);
        InitLocalBuffer(l1BaseAddr);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitCubeInput(__gm__ uint8_t *cuSeqlensQ, const ConstInfo &constInfo)
{
    if ASCEND_IS_AIC {
        InitGmTensor(cuSeqlensQ, constInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitLocalBuffer(uint32_t l1BaseAddr)
{
    uint32_t l1Addr = l1BaseAddr;

    l1QBufs[0] = {LocalTensor<Q_T>(TPosition::A1, l1Addr, L1Q_ELEM_PER_BUF), 0};
    l1Addr += L1Q_ELEM_PER_BUF * sizeof(Q_T);
    l1QBufs[1] = {LocalTensor<Q_T>(TPosition::A1, l1Addr, L1Q_ELEM_PER_BUF), 1};
    l1Addr += L1Q_ELEM_PER_BUF * sizeof(Q_T);
    l1QBufs[2] = {LocalTensor<Q_T>(TPosition::A1, l1Addr, L1Q_ELEM_PER_BUF), 2};
    l1Addr += L1Q_ELEM_PER_BUF * sizeof(Q_T);

    l1RightBufs[0] = {LocalTensor<Q_T>(TPosition::B1, l1Addr, L1_RIGHT_ELEM_PER_BLOCK), 0};
    l1Addr += L1_RIGHT_ELEM_PER_BLOCK * sizeof(Q_T);
    l1RightBufs[1] = {LocalTensor<Q_T>(TPosition::B1, l1Addr, L1_RIGHT_ELEM_PER_BLOCK), 1};
    l1Addr += L1_RIGHT_ELEM_PER_BLOCK * sizeof(Q_T);
    l1RightBufs[2] = {LocalTensor<Q_T>(TPosition::B1, l1Addr, L1_RIGHT_ELEM_PER_BLOCK), 2};
    l1Addr += L1_RIGHT_ELEM_PER_BLOCK * sizeof(Q_T);

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
}

/* 初始化GmTensor,设置shape信息并计算strides */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitGmTensor(__gm__ uint8_t *cuSeqlensQ, const ConstInfo &constInfo)
{
    if constexpr (LAYOUT_T == SAS_LAYOUT::BSND) {
        this->queryGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize,
            constInfo.s1Size, constInfo.dSize);
    } else {  // SAS_LAYOUT::TND
        GlobalTensor<int32_t> actualSeqQLen;
        actualSeqQLen.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
        this->queryGm.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSize,
            actualSeqQLen, constInfo.actualSeqLenSize);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm1(
    StaticBuffer<T> &outputBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    bool notLastTwoLoop, RunInfo &runInfoNext,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    IterateBmm1SCFA(outputBuf, v0ResGm, notLastTwoLoop, runInfoNext, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm2(StaticBuffer<T> &outputBuf,
    StaticBuffer<Q_T> &l1PBuffer,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    IterateBmm2SCFA(outputBuf, l1PBuffer, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::CopyQGmToL1(RunInfo &runInfo, ConstInfo &constInfo)
{
    uint64_t gmOffset = this->queryGm.offsetCalculator.GetOffset(runInfo.boIdx, runInfo.n2oIdx, runInfo.goIdx,
        runInfo.s1oIdx * runInfo.qSNumInOneBlock, 0);
    for (uint32_t i = 0; i < 2; i++) {
        uint32_t curL1QBufId = (l1QBufIdx + i) % l1QBufNum;
        Mutex::Lock<PIPE_MTE2>(INNERCORE_L1Q(curL1QBufId));
        uint64_t curGmOffset = gmOffset + i * (constInfo.dSize >> 1);
        CopyToL1Nd2Nz<Q_T>(l1QBufs[curL1QBufId].tensor,
            this->queryGm.gmTensor[curGmOffset], runInfo.mRealSize, constInfo.dSize >> 1,
            constInfo.mm1Ka);
        Mutex::Unlock<PIPE_MTE2>(INNERCORE_L1Q(curL1QBufId));
    }
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateLoadQK(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    RunInfo &runInfo, ConstInfo &constInfo, bool isFirstLoop)
{
    if (unlikely(isFirstLoop)) {
        CopyQGmToL1(runInfo, constInfo);
    }

    Mutex::Lock<PIPE_MTE2>(INNERCORE_L1KV(l1KLoadBufIdx));
    Mutex::Lock<PIPE_MTE2>(INNERCORE_L1KV(l1KLoadBufIdx + 1));
    LocalTensor<Q_T> dst = l1RightBufs[runInfo.taskIdMod3].tensor;
    v0ResGm.WaitCrossCore();
    if constexpr (IS_SPLIT_G) {
        CrossCoreSetFlag<0, PIPE_MTE2>(10);
        CrossCoreWaitFlag<0, PIPE_MTE2>(10);
    }
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    DataCopy(dst, v0ResGmTensor, Align16Func(runInfo.s2RealSize) * (constInfo.dSize >> 1));
    Mutex::Unlock<PIPE_MTE2>(INNERCORE_L1KV(l1KLoadBufIdx));

    l1KLoadBufIdx = (l1KLoadBufIdx + 1) % l1KBufNum;
    DataCopy(dst[Align16Func(runInfo.s2RealSize) * (constInfo.dSize >> 1)],
        v0ResGmTensor[Align16Func(runInfo.s2RealSize) * (constInfo.dSize >> 1)],
        Align16Func(runInfo.s2RealSize) * (constInfo.dSize >> 1));
    Mutex::Unlock<PIPE_MTE2>(INNERCORE_L1KV(l1KLoadBufIdx));
    l1KLoadBufIdx = (l1KLoadBufIdx + 1) % l1KBufNum;
}
 

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm1SCFA(
    StaticBuffer<T> &outputBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    bool notLastTwoLoop, RunInfo &runInfoNext,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    LocalTensor<Q_T> curL1RightTensor = l1RightBufs[runInfo.taskIdMod3].tensor;

    Mutex::Lock<PIPE_MTE1>(INNERCORE_L1KV(l1KMatmul1BufIdx));

    StaticBuffer<T> &cBuf = l0C.GetNext();
    Mutex::Lock<PIPE_M>(INNERCORE_L0C(cBuf.idx));

    MMParam param = {static_cast<uint32_t>(runInfo.mRealSize),
                     static_cast<uint32_t>(runInfo.s2RealSize),
                     static_cast<uint32_t>(constInfo.dSize >> 1),
                     0, 1};
    uint32_t curL1QBufId = l1QBufIdx;
    if (unlikely(runInfo.s2LoopCount == 0)) {
        Mutex::Lock<PIPE_MTE1>(INNERCORE_L1Q(curL1QBufId));
    }

    MatmulK1Static<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        l1QBufs[curL1QBufId].tensor, curL1RightTensor,
        l0A, l0B, cBuf.tensor, param);

    l1KMatmul1BufIdx = (l1KMatmul1BufIdx + 1) % l1KBufNum;
    Mutex::Lock<PIPE_MTE1>(INNERCORE_L1KV(l1KMatmul1BufIdx));
    l1KMatmul1BufIdx = (l1KMatmul1BufIdx + 1) % l1KBufNum;

    curL1QBufId = (curL1QBufId + 1) % l1QBufNum;
    if (unlikely(runInfo.s2LoopCount == 0)) {
        Mutex::Lock<PIPE_MTE1>(INNERCORE_L1Q(curL1QBufId));
    }
    param.singleK = constInfo.dSize - param.singleK;
    param.isOutKFisrt = false;

    MatmulK1Static<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        l1QBufs[curL1QBufId].tensor,
        curL1RightTensor[(constInfo.dSize >> 1) * Align16Func(runInfo.s2RealSize)],
        l0A, l0B, cBuf.tensor, param);

    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        Mutex::Unlock<PIPE_MTE1>(INNERCORE_L1Q(l1QBufIdx));
        Mutex::Unlock<PIPE_MTE1>(INNERCORE_L1Q(curL1QBufId));
        l1QBufIdx = (l1QBufIdx + 2) % l1QBufNum;
        if (notLastTwoLoop) {
            CopyQGmToL1(runInfoNext, constInfo);
        }
    }

    Mutex::Unlock<PIPE_M>(INNERCORE_L0C(cBuf.idx));
    Mutex::Lock<PIPE_FIX>(INNERCORE_L0C(cBuf.idx));

    CrossCoreWaitFlag<4, PIPE_FIX>(CROSSCORE_BMM1(outputBuf.idx));
    CrossCoreWaitFlag<4, PIPE_FIX>(CROSSCORE_BMM1_D(outputBuf.idx));
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB
    fixpipeParams.nSize = Align8Func(runInfo.s2RealSize); // L0C上的bmm1结果矩阵N方向的size大小; 同mmadParams.n; 为什么要8个元素对齐(32B对齐) // 128
    fixpipeParams.mSize = Align2Func(runInfo.mRealSize); // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
    fixpipeParams.srcStride = Align16Func(fixpipeParams.mSize); // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T) // 源Nz矩阵中相邻大Z排布的起始地址偏移
    fixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element。 // 128:根据比对dump文件得到, ND方案(S1*S2)时脏数据用mask剔除
    fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;

    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.tensor, cBuf.tensor, fixpipeParams); // 将matmul结果从L0C搬运到UB
    Mutex::Unlock<PIPE_FIX>(INNERCORE_L0C(cBuf.idx));
    CrossCoreSetFlag<4, PIPE_FIX>(CROSSCORE_BMM1(outputBuf.idx));
    CrossCoreSetFlag<4, PIPE_FIX>(CROSSCORE_BMM1_D(outputBuf.idx));
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm2SCFA(StaticBuffer<T> &outputBuf,
    StaticBuffer<Q_T> &l1PBuffer,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    LocalTensor<Q_T> curL1RightTensor = l1RightBufs[runInfo.taskIdMod3].tensor;

    CrossCoreWaitFlag<4, PIPE_MTE1>(CROSSCORE_L1P(l1PBuffer.idx));
    CrossCoreWaitFlag<4, PIPE_MTE1>(CROSSCORE_L1P_D(l1PBuffer.idx));

    StaticBuffer<T> &cBuf = l0C.GetNext();
    Mutex::Lock<PIPE_M>(INNERCORE_L0C(cBuf.idx));

    MMParam param = {static_cast<uint32_t>(s1BaseSize),
                     static_cast<uint32_t>(constInfo.dSizeV),
                     static_cast<uint32_t>(runInfo.s2RealSize),
                     0, 0};

    MatmulN1Static<Q_T, Q_T, T, s1BaseSize, dBaseMatmulSize, s2BaseSize, ABLayout::MK, ABLayout::KN>(
        l1PBuffer.tensor, curL1RightTensor,
        l0A, l0B, cBuf.tensor, param);

    Mutex::Unlock<PIPE_M>(INNERCORE_L0C(cBuf.idx));
    Mutex::Lock<PIPE_FIX>(INNERCORE_L0C(cBuf.idx));

    Mutex::Unlock<PIPE_MTE1>(INNERCORE_L1KV(l1KMatmul2BufIdx));
    l1KMatmul2BufIdx = (l1KMatmul2BufIdx + 1) % l1KBufNum;
    Mutex::Unlock<PIPE_MTE1>(INNERCORE_L1KV(l1KMatmul2BufIdx));
    l1KMatmul2BufIdx = (l1KMatmul2BufIdx + 1) % l1KBufNum;

    CrossCoreWaitFlag<4, PIPE_FIX>(CROSSCORE_BMM2);
    CrossCoreWaitFlag<4, PIPE_FIX>(CROSSCORE_BMM2_D);
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB;FixpipeParamsM300:L0C→UB
    fixpipeParams.nSize = Align8Func(constInfo.dSizeV);    // L0C上的bmm1结果矩阵N方向的size大小, 分档计算且vector2中通过mask筛选出实际有效值
    fixpipeParams.mSize = s1BaseSize;                      // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小; 同mmadParams.m
    fixpipeParams.srcStride = Align16Func(s1BaseSize);     // L0C上bmm1结果相邻连续数据片段间隔（前面一个数据块的头与后面数据块的头的间隔）
    fixpipeParams.dstStride = Align16Func(constInfo.dSizeV);
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.tensor, cBuf.tensor, fixpipeParams); // 将matmul结果从L0C搬运到UB
    Mutex::Unlock<PIPE_FIX>(INNERCORE_L0C(cBuf.idx));

    CrossCoreSetFlag<4, PIPE_FIX>(CROSSCORE_BMM2);
    CrossCoreSetFlag<4, PIPE_FIX>(CROSSCORE_BMM2_D);
}

TEMPLATES_DEF
class SCFABlockCubeDummy {
public:
    __aicore__ inline SCFABlockCubeDummy() {};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, __gm__ uint8_t *query, uint32_t l1BaseAddr) {}
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *cuSeqlensQ, const ConstInfo& constInfo) {}
};

template <typename T>
struct CubeBlockTraits;  // 声明

/* 生成CubeBlockTraits */
#define GEN_TRAIT_TYPE(name, ...) using name##_TRAITS = name;
#define GEN_TRAIT_CONST(name, type, ...) static constexpr type name##Traits = name;

#define DEFINE_CUBE_BLOCK_TRAITS(CUBE_BLOCK_CLASS) \
    TEMPLATES_DEF_NO_DEFAULT \
    struct CubeBlockTraits<CUBE_BLOCK_CLASS<TEMPLATE_ARGS>> { \
        CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TRAIT_TYPE) \
        CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TRAIT_CONST) \
    }

DEFINE_CUBE_BLOCK_TRAITS(SCFABlockCube);
DEFINE_CUBE_BLOCK_TRAITS(SCFABlockCubeDummy);

// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)
}
#endif // KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_CUBE_H
