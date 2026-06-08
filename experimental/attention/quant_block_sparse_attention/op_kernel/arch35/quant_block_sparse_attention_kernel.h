/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// BSA-COPY-OF: FIA fullquant GQA local migration.
// TODO: 待实现 - 当前已完成自包含迁移和命名/include 适配；sparseSeqLen/sparseIndices 流水、per-token qScale、vScale、空 sparse 输出和 mask sparse offset 仍需按 BSA 方案 2.4 实现。

/*!
 * ile quant_block_sparse_attention_kernel.h
 * \brief
 */

#ifndef QUANT_BLOCK_SPARSE_ATTENTION_KERNEL_H_
#define QUANT_BLOCK_SPARSE_ATTENTION_KERNEL_H_
#include "quant_block_sparse_attention_block_cube.h"
#include "quant_block_sparse_attention_block_vec.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "quant_block_sparse_attention_attenmask.h"

// 线上编包
#include "common/matmul.h"
#include "common/FixpipeOut.h"
#include "common/CopyInL1.h"

#include "quant_block_sparse_attention_common_arch35.h"
#include "quant_block_sparse_attention_kvcache.h"
#include "quant_block_sparse_attention_sparse.h"
#include "kernel_operator_list_tensor_intf.h"
#include "adv_api/utils/init_global_memory.h"

using matmul::MatmulType;
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
namespace BSAKernelBase {
/* ============确定bmm2ResBuffer的类型============= */
template <bool useDn, bool isFp8>
struct Bmm2ResBuffSel {
    using Type = std::conditional_t<(useDn && isFp8),
        BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
        BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>>;
};
}


template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
class QuantBlockSparseAttentionKernelBase {
public:
    ARGS_TRAITS;
    static constexpr ImplModeEnum implMode = kBsaImplMode;
    static constexpr PseTypeEnum pseMode = kBsaPseMode;
    static constexpr bool hasDrop = kBsaHasDrop;
    static constexpr bool hasRope = kBsaHasRope;
    static constexpr bool isInfer = kBsaIsInfer;
    static constexpr bool enableKVPrefix = kBsaEnableKVPrefix;
    __aicore__ inline QuantBlockSparseAttentionKernelBase() {};

    __aicore__ inline void InitBaseAPI(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                            __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask,
                            __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths,
                            __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *blockTable,
                            __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseSeqLen, __gm__ uint8_t *queryPaddingSize,
                            __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK,
                            __gm__ uint8_t *deqScaleV, __gm__ uint8_t *pScale,
                            __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t *actualSharedPrefixLen,
                            __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxMax,
                            __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                            __gm__ uint8_t *workspace, const QuantBlockSparseAttentionTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                            __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask,
                            __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
                            __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *pScale,
                            __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t *actualSharedPrefixLen,
                            __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize,
                            __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseSeqLen,
                            __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                            __gm__ uint8_t *softmaxOut, __gm__ uint8_t *workspace,
                            const QuantBlockSparseAttentionTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitMMResBuf();
    __aicore__ inline void InitActualKVPrefixLen(__gm__ uint8_t *actualSharedPrefixLen);
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void SetRunInfo(RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam, int64_t taskId, int64_t s2LoopCount,
                                      int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void ComputeBmm1Tail(RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam);
    __aicore__ inline void GetSeqQlenKvlenByBoidx(int64_t boIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvLen);

    __aicore__ inline ChildClass* GetDerived() {
        return static_cast<ChildClass*>(this);
    }
    TPipe *pipe;

    const QuantBlockSparseAttentionTilingData *__restrict tilingData;
    /* 编译期常量的基本块信息 */
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr bool isFp8 = CubeBlockType::isFp8;
    /* 是否使能dn的信息; 没有可选输入并且S2切分的时候使用dn，s2比较小的时候nd效果更好 */
    static constexpr bool useDn = CubeBlockType::useDn;
    /* HIFLOAT8场景 softmax计算使用Nz格式计算，vec1ResBuffer可以和bmm1ResBuffer进行复用*/
    static constexpr bool useNz = CubeBlockType::useNz;
    static constexpr TPosition bmm2OutPos = CubeBlockType::bmm2OutPos;
    static constexpr bool bmm2Write2Ub = CubeBlockType::bmm2Write2Ub;
    static constexpr bool splitD =  CubeBlockType::splitD;
    static constexpr uint64_t SYNC_MODE = 4;
    static constexpr uint64_t SYNC_C1_V1_FLAG[2] = {0, 1};
    static constexpr uint64_t SYNC_V1_C2_FLAG[3] = {2, 3, 4};
    static constexpr uint64_t SYNC_C2_V2_FLAG[2] = {5, 6};
    /* 核间通道 */
    BufferManager<BufferType::GM> gmBufferManager;
    BuffersPolicy3buff<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD> bmm2ResGmBuffers;

    BufferManager<BufferType::UB> ubBufferManager;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    using bmm2ResBufferType = typename BSAKernelBase::Bmm2ResBuffSel<useDn, isFp8>::Type;
    bmm2ResBufferType bmm2Buffers;

    // mm2左矩阵P
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers;
    CVSharedParams<isInfer, isPa> sharedParams;
    /* GM信息 */
    using keyGmType = typename std::conditional<isInfer, GlobalTensor<INPUT_T>, int32_t>::type;
    keyGmType keyGm; // kv不连续场景需要使用来获取shape
    __gm__ int64_t *actualSeqQlenAddr;
    __gm__ int64_t *actualSeqKvlenAddr;
    __gm__ int32_t *sparseIndicesAddr;
    __gm__ int32_t *sparseSeqLenAddr;
    /* 核Index信息 */
    int32_t aicIdx;

    /* 初始化后不变的信息 */
    ConstInfo<isInfer, hasRope> constInfo;
    AttenMaskInfo attenMaskInfo;
    /* 其他正向独有的一些信息 */
    // Unpack参数
    uint64_t s1OuterSizeAcc;
    uint64_t s1SizeAcc;
    uint64_t s2SizeAcc;
    uint64_t s1ScaleNumAcc;
    uint64_t s2ScaleNumAcc;

    // prefix
    int64_t actualKVPrefixLen = 0;

    /* 模板库Block */
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;
};

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernelBase<ChildClass, CubeBlockType, VecBlockType>::InitBaseAPI(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *blockTable, __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *sparseSeqLen, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
    __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *pScale,
    __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
    __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *learnableSink,
    __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const QuantBlockSparseAttentionTilingData *__restrict tiling, TPipe *tPipe)
{
    fa_base_matmul::idCounterNum = 0;
    constInfo.subBlockIdx = GetSubBlockIdx();
    if ASCEND_IS_AIC {
        this->aicIdx = GetBlockIdx();
    } else {
        constInfo.aivIdx = GetBlockIdx();
        this->aicIdx = constInfo.aivIdx >> 1;
        this->tilingData = tiling;
    }
    this->pipe = tPipe;
    vecBlock.InitVecBlock(tPipe, this->tilingData, this->sharedParams, this->aicIdx, constInfo.subBlockIdx,
        attenMaskInfo);
    vecBlock.CleanOutput(softmaxLse, attentionOut, constInfo);
    /* cube侧不依赖sharedParams的scalar前置 */
    InitMMResBuf();
    if ASCEND_IS_AIC {
        cubeBlock.InitCubeBlock(pipe, &l1BufferManager, query, key, value, blockTable, queryRope, keyRope);
        /* wait kfc message */
        CrossCoreWaitFlag<SYNC_MODE, PIPE_S>(15);
        auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0); // 从ssbuf的0地址开始拷贝
        auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);
        #pragma unroll
        for (int i = 0; i < sizeof(CVSharedParams<isInfer, isPa>) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
            *tempTiling = *tempTilingSSbuf;
        }
    }

    this->InitActualKVPrefixLen(actualSharedPrefixLen);
    this->ComputeConstexpr();
    this->InitGlobalBuffer(query, key, value, dropMask, paddingMask, attenMask, prefix, actualSeqLengths,
        actualSeqLengthsKv, deqScaleQ, deqScaleK, deqScaleV, pScale,
        keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope, blockTable, queryPaddingSize,
        kvPaddingSize, sparseIndices, sparseSeqLen, learnableSink, softmaxMax, softmaxSum, softmaxOut, workspace,
        tiling, tPipe); // gm设置
    this->InitLocalBuffer();
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernelBase<ChildClass, CubeBlockType, VecBlockType>::InitActualKVPrefixLen(__gm__ uint8_t *actualSharedPrefixLen)
{
    if constexpr (isInfer) {
        if constexpr (enableKVPrefix) {
            constInfo.kvPrefixSize = this->sharedParams.kvPrefixSize;
            if (!this->sharedParams.isActualSharedPrefixLenNull) {
                this->actualKVPrefixLen = ((__gm__ int64_t *)actualSharedPrefixLen)[0];
            } else {
                this->actualKVPrefixLen = this->sharedParams.kvPrefixSize;
            }
            constInfo.actualKVPrefixSize = this->actualKVPrefixLen;
        }
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernelBase<ChildClass, CubeBlockType, VecBlockType>::InitGlobalBuffer(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
    __gm__ uint8_t *pScale, __gm__ uint8_t *keySharedPrefix,
    __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t*actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
    __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseSeqLen, __gm__ uint8_t *learnableSink,
    __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *workspace,
    const QuantBlockSparseAttentionTilingData *__restrict tiling, TPipe *tPipe)
{
    // 初始化vector用到的global buffer
    if constexpr (isInfer) {
        keyGm.SetGlobalBuffer((__gm__ INPUT_T *)(key));
        if (constInfo.isQHasLeftPadding) {
            constInfo.queryRightPaddingSize = ((__gm__ int64_t *)queryPaddingSize)[0];
            if (constInfo.queryRightPaddingSize < 0) {
                constInfo.queryRightPaddingSize = 0;
            }
        }
        if (constInfo.isKVHasLeftPadding) {
            constInfo.kvRightPaddingSize = ((__gm__ int64_t *)kvPaddingSize)[0];
            if (constInfo.kvRightPaddingSize < 0) {
                constInfo.kvRightPaddingSize = 0;
            }
        }
    }
    if constexpr (hasAtten) {
        attenMaskInfo.prefixNAddr = prefix;
    }
    sparseIndicesAddr = (__gm__ int32_t *)sparseIndices;
    sparseSeqLenAddr = (__gm__ int32_t *)sparseSeqLen;
    if constexpr (q_layout == LayOutTypeEnum::LAYOUT_TND || q_layout == LayOutTypeEnum::LAYOUT_NTD) {
        actualSeqQlenAddr = (__gm__ int64_t *)actualSeqLengths;
        actualSeqKvlenAddr = (__gm__ int64_t *)actualSeqLengthsKv;
    } else {
        if constexpr (isInfer) {
            if (!constInfo.isActualLenDimsNull) {
                actualSeqQlenAddr = (__gm__ int64_t *)actualSeqLengths;
            }
            if (!constInfo.isActualLenDimsKVNull) {
                actualSeqKvlenAddr = (__gm__ int64_t *)actualSeqLengthsKv;
            }
        }
    }

    vecBlock.InitDropOut(dropMask, workspace);
    uint64_t singleCoreOffset = 0;
    if constexpr (!bmm2Write2Ub) {
        int64_t bmm2ResBlock = this->sharedParams.dSizeV;
        if constexpr (splitD) {
            bmm2ResBlock = (int64_t)dVTemplateType;
        }
        int64_t mm2ResultSize = (s1BaseSize) * bmm2ResBlock; // 使用Cube计算的总大小，Gm上的数据按照实际的dSize存储
        int64_t mm2Offset = CeilDiv(mm2ResultSize, 128) * 128 * sizeof(T);
        int64_t vec2ResultSize = (s1BaseSize) * constInfo.dBasicBlock;
        int64_t vec2Offset = CeilDiv(vec2ResultSize, 128) * 128 * sizeof(T);
        singleCoreOffset = mm2Offset;
        if constexpr (splitD) {
            singleCoreOffset = mm2Offset + vec2Offset;
        }
        int64_t totalOffset = this->aicIdx * 3 * singleCoreOffset; // 3为preload次数
        // SameB模式下V0和V1调用IterateAll的时候填写的地址相同
        gmBufferManager.Init(workspace + totalOffset);
        bmm2ResGmBuffers.Init(gmBufferManager, mm2Offset);
        workspace += (totalOffset + mm2Offset * 3);
    }
    vecBlock.InitGlobalBuffer(deqScaleQ, deqScaleK, deqScaleV, pScale,
        prefix, attenMask, blockTable, queryPaddingSize, kvPaddingSize, learnableSink, softmaxMax, softmaxSum,
        workspace, singleCoreOffset,
        this->aicIdx, constInfo);
    if constexpr (q_layout == LayOutTypeEnum::LAYOUT_TND && !isInfer) {
        if ASCEND_IS_AIV {
            int64_t actualS1Len;
            int64_t actualS2Len;
            for (int64_t i = 0; i < this->sharedParams.bSize; ++i) {
                this->GetSeqQlenKvlenByBoidx(i, actualS1Len, actualS2Len);
                if (actualS2Len <= 0 && actualS1Len != 0) {
                    int64_t accumSize = (i == 0) ? 0 : ((__gm__ int64_t *)this->actualSeqQlenAddr)[i - 1];
                    if (actualS1Len < 0 && accumSize > 0) {
                        actualS1Len = constInfo.s1Size - accumSize;
                        int64_t frontCoreNum = actualS1Len % (this->sharedParams.coreNum * 2);
                        int64_t splitFactor = frontCoreNum > 0 ? 1 : 0;
                        int64_t s1SizeInner = actualS1Len / this->sharedParams.coreNum / 2;
                        int64_t innerOffset1 = (s1SizeInner + splitFactor) * (constInfo.aivIdx >= frontCoreNum ?
                                            frontCoreNum : constInfo.aivIdx);
                        int64_t innerOffset2 = s1SizeInner * (constInfo.aivIdx >= frontCoreNum ?
                                            constInfo.aivIdx - frontCoreNum : 0);
                        accumSize = accumSize + innerOffset1 + innerOffset2;
                        actualS1Len = s1SizeInner + (constInfo.aivIdx >= frontCoreNum ? 0 : splitFactor);
                    }
                    GlobalTensor<OUTPUT_T> attentionOutEodGm = this->vecBlock.attentionOutGm[accumSize * this->constInfo.n2GDv];
                    GlobalTensor<float> softmaxMaxEodGm = this->vecBlock.softmaxMaxGm[accumSize * this->constInfo.n2G * 8];
                    GlobalTensor<float> softmaxSumEodGm = this->vecBlock.softmaxSumGm[accumSize * this->constInfo.n2G * 8];
                    AscendC::Fill<OUTPUT_T>(attentionOutEodGm, actualS1Len * this->constInfo.n2GDv, static_cast<OUTPUT_T>(0.0));
                    AscendC::Fill<float>(softmaxMaxEodGm, actualS1Len * this->constInfo.n2G * 8, static_cast<float>(0.0));
                    AscendC::Fill<float>(softmaxSumEodGm, actualS1Len * this->constInfo.n2G * 8, static_cast<float>(0.0));
                    SyncAll();
                }
            }
        }
    }
    cubeBlock.InitCubeInput(key, value, &sharedParams, &attenMaskInfo, actualSeqQlenAddr, actualSeqKvlenAddr, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen);
    cubeBlock.InitDequantParams(deqScaleQ, deqScaleK, deqScaleV);
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernelBase<ChildClass, CubeBlockType, VecBlockType>::InitMMResBuf()
{
    uint32_t mm1OutDtype = sizeof(T);
    if constexpr (useNz) {
        mm1OutDtype = sizeof(half);
    }
    uint32_t mm1ResultSize = s1BaseSize / CV_RATIO * s2BaseSize * mm1OutDtype;
    constexpr uint32_t mm2ResultSize = s1BaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
    constexpr uint32_t mm2LeftSize = s1BaseSize * s2BaseSize * sizeof(INPUT_T);
    l1BufferManager.Init(pipe, 524288); // 512 * 1024
    // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
    l1PBuffers.Init(l1BufferManager, mm2LeftSize);
    if constexpr (bmm2Write2Ub) {
        if constexpr (!(useDn && isFp8)) {
            ubBufferManager.Init(pipe, mm1ResultSize * 2 + mm2ResultSize * 2);
            bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
            if ASCEND_IS_AIV {
                bmm2Buffers.Get().SetCrossCore();
                bmm2Buffers.Get().SetCrossCore();
            }
        } else {
            ubBufferManager.Init(pipe, mm1ResultSize * 2 + mm2ResultSize);
            bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
            if ASCEND_IS_AIV {
                bmm2Buffers.Get().SetCrossCore();
            }
        }
    } else {
        ubBufferManager.Init(pipe, mm1ResultSize * 2);
    }
    bmm1Buffers.Init(ubBufferManager, mm1ResultSize);
    if ASCEND_IS_AIV {
        if constexpr (useNz) {
            bmm1Buffers.Get().SetCrossCore<true>();
            bmm1Buffers.Get().SetCrossCore<true>();
        } else {
            bmm1Buffers.Get().SetCrossCore();
            bmm1Buffers.Get().SetCrossCore();
        }
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernelBase<ChildClass, CubeBlockType, VecBlockType>::InitLocalBuffer()
{
    vecBlock.InitLocalBuffer(pipe, constInfo);
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernelBase<ChildClass, CubeBlockType, VecBlockType>::ComputeConstexpr()
{
    constInfo.s1BaseSize = s1BaseSize;
    constInfo.s2BaseSize = s2BaseSize;
    // 计算轴的乘积

    constInfo.bSize = sharedParams.bSize;
    constInfo.t1Size = sharedParams.t1Size;
    constInfo.t2Size = sharedParams.t2Size;
    constInfo.n2Size = sharedParams.n2Size;
    constInfo.s1Size = sharedParams.s1Size;
    constInfo.s2Size = sharedParams.s2Size;
    constInfo.dSize = sharedParams.dSize;
    constInfo.dSizeV = sharedParams.dSizeV;
    constInfo.combineDim = (constInfo.dSize + constInfo.dSizeV + sizeof(float) + 31U) / 32U * 32U;
    constInfo.dBasicBlock = Align64Func((uint16_t)constInfo.dSizeV);
    if constexpr (hasRope) {
        constInfo.dSizeRope = sharedParams.dSizeRope;
    } else {
        constInfo.dSizeRope = 0;
    }
    constInfo.gSize = sharedParams.gSize;
    constInfo.s1OuterSize = sharedParams.s1OuterSize;
    constInfo.s1S2 = constInfo.s1Size * constInfo.s2Size;
    constInfo.gS1 = constInfo.gSize * constInfo.s1Size;
    constInfo.n2G = constInfo.n2Size * constInfo.gSize;

    constInfo.s1Dv = constInfo.s1Size * constInfo.dSizeV;
    constInfo.s2Dv = constInfo.s2Size * constInfo.dSizeV;
    constInfo.n2Dv = constInfo.n2Size * constInfo.dSizeV;
    constInfo.gDv = constInfo.gSize * constInfo.dSizeV;
    constInfo.gS1Dv = constInfo.gSize * constInfo.s1Dv;
    constInfo.n2S2Dv = constInfo.n2Size * constInfo.s2Dv;
    constInfo.n2GDv = constInfo.n2Size * constInfo.gDv;
    constInfo.s2BaseN2Dv = s2BaseSize * constInfo.n2Dv;
    constInfo.n2GS1Dv = constInfo.n2Size * constInfo.gS1Dv;
    constInfo.layoutType = sharedParams.layoutType;

    if (sharedParams.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BNSD_NBSD) ||
        sharedParams.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BSND_NBSD) ||
        sharedParams.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BSH_NBSD)) {
        constInfo.t1Size = constInfo.bSize * constInfo.s1Size;
    }

    if constexpr (isInfer) {
        constInfo.s1D = constInfo.s1Size * constInfo.dSize;
        constInfo.gD = constInfo.gSize * constInfo.dSize;
        constInfo.n2GD = constInfo.n2Size * constInfo.gD;
        constInfo.gS1D = constInfo.gS1 * constInfo.dSize;
    }

    if constexpr (q_layout == LayOutTypeEnum::LAYOUT_TND) {
        // (BS)ND
        constInfo.s1BaseN2GDv = s1BaseSize * constInfo.n2GDv;

        constInfo.mm1Ka = constInfo.n2Size * constInfo.gSize * constInfo.dSize;
        constInfo.mm1Kb = constInfo.n2Size * constInfo.dSize;
        constInfo.mm2Kb = constInfo.n2Dv;
        if constexpr (isInfer) {
            if (sharedParams.isGqa) {
                constInfo.mm1Ka = constInfo.dSize;
            }
        }
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - 1) * constInfo.dSizeV * sizeof(OUTPUT_T);
            if constexpr (isInfer) {
                if (sharedParams.isPfaGS1Merge == 1) {
                    constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
                } else if (sharedParams.isGqa == 1) {
                    constInfo.attentionOutStride = 0;
                }
            }
        }
    } else if constexpr (q_layout == LayOutTypeEnum::LAYOUT_BSH) {
        // BSH/BSNGD
        constInfo.s1BaseN2GDv = s1BaseSize * constInfo.n2GDv;
        constInfo.mm1Ka = constInfo.n2Size * constInfo.gSize * constInfo.dSize;
        constInfo.mm1Kb = constInfo.n2Size * constInfo.dSize;
        constInfo.mm2Kb = constInfo.n2Dv;
        if constexpr (isInfer) {
            if (sharedParams.isGqa) {
                constInfo.mm1Ka = constInfo.dSize;
            }
        }
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride =
                (constInfo.n2G - 1) * constInfo.dSizeV * sizeof(OUTPUT_T);
            if constexpr (isInfer) {
                if (sharedParams.isPfaGS1Merge == 1) {
                    constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
                } else if (sharedParams.isGqa == 1) {
                    constInfo.attentionOutStride = 0;
                }
            }
        }
    } else if constexpr (q_layout == LayOutTypeEnum::LAYOUT_SBH) {
        // SBH/SBNGD
        constInfo.bN2GDv = sharedParams.bSize * constInfo.n2GDv;
        constInfo.s1BaseBN2GDv = s1BaseSize * constInfo.bN2GDv;
        constInfo.s2BaseBN2Dv = sharedParams.bSize * constInfo.s2BaseN2Dv;
        constInfo.mm1Ka = sharedParams.bSize * constInfo.n2Size * constInfo.gSize * constInfo.dSize;
        constInfo.mm1Kb = sharedParams.bSize * constInfo.n2Size * constInfo.dSize;
        constInfo.mm2Kb = sharedParams.bSize * constInfo.n2Dv;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride =
                (sharedParams.bSize * constInfo.n2Size * constInfo.gSize - 1) * constInfo.dSizeV * sizeof(OUTPUT_T);
        }
    } else if constexpr (q_layout == LayOutTypeEnum::LAYOUT_BNSD) {
        // bnsd
        constInfo.s1BaseDv = s1BaseSize * constInfo.dSizeV;
        constInfo.s2BaseDv = s2BaseSize * constInfo.dSizeV;
        constInfo.mm1Ka = constInfo.dSize;
        constInfo.mm1Kb = constInfo.dSize;
        constInfo.mm2Kb = constInfo.dSizeV;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = 0;
        }
    } else if constexpr (q_layout == LayOutTypeEnum::LAYOUT_NTD) {
        // NG(BS)D
        constInfo.s1BaseDv = s1BaseSize * constInfo.dSizeV;
        constInfo.s2BaseDv = s2BaseSize * constInfo.dSizeV;
        constInfo.mm1Ka = constInfo.dSize;
        constInfo.mm1Kb = constInfo.dSize;
        constInfo.mm2Kb = constInfo.dSizeV;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = 0;
        }
    }

    if ASCEND_IS_AIV {
        auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
        if constexpr (hasAtten) {
            attenMaskInfo.preTokens = sharedParams.preTokens;
            attenMaskInfo.nextTokens = sharedParams.nextTokens;
            attenMaskInfo.attenMaskShapeType = inputParamsRegbase.attenMaskShapeType;
            attenMaskInfo.attenMaskS1Size = inputParamsRegbase.attenMaskS1Size;
            attenMaskInfo.attenMaskS2Size = inputParamsRegbase.attenMaskS2Size;
            attenMaskInfo.bandIndex = inputParamsRegbase.bandIndex;
        }
        attenMaskInfo.compressMode = inputParamsRegbase.attenMaskCompressMode;
        constInfo.scaleValue = static_cast<float>(inputParamsRegbase.scaleValue);
    }

    GetDerived()->InitUniqueConstInfo();
}


template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetSeqQlenKvlenByBoidx(int64_t boIdx,
    int64_t &actualSeqQlen, int64_t &actualSeqKvlen)
{
    if (unlikely(boIdx == 0)) {
        actualSeqQlen = actualSeqQlenAddr[0];
        actualSeqKvlen = actualSeqKvlenAddr[0];
        return;
    }
    actualSeqQlen = actualSeqQlenAddr[boIdx] - actualSeqQlenAddr[boIdx - 1];
    actualSeqKvlen = actualSeqKvlenAddr[boIdx] - actualSeqKvlenAddr[boIdx - 1];
}


template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernelBase<ChildClass, CubeBlockType, VecBlockType>::SetRunInfo(
    RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    runInfo.s2StartIdx = runParam.s2LineStartIdx;
    runInfo.s2EndIdx = runParam.s2LineEndIdx;
    runInfo.s2LoopCount = s2LoopCount;
    if (runInfo.multiCoreInnerIdx != multiCoreInnerIdx) {
        runInfo.s1oIdx = runParam.s1oIdx;
        runInfo.boIdx = runParam.boIdx;
        runInfo.n2oIdx = runParam.n2oIdx;
        runInfo.goIdx = runParam.goIdx;
        runInfo.multiCoreInnerIdx = multiCoreInnerIdx;
        runInfo.multiCoreIdxMod2 = multiCoreInnerIdx & 1;
        runInfo.multiCoreIdxMod3 = multiCoreInnerIdx % 3;
    }
    if constexpr (q_layout == LayOutTypeEnum::LAYOUT_TND || q_layout == LayOutTypeEnum::LAYOUT_NTD) {
        runInfo.boIdx = runParam.boIdx;
        runInfo.s1ScaleNumAcc = s1ScaleNumAcc;
        runInfo.s2ScaleNumAcc = s2ScaleNumAcc;
        runInfo.s1SizeAcc = s1SizeAcc;
        runInfo.s2SizeAcc = s2SizeAcc;
    } else {
        runInfo.s2SizeAcc = runInfo.boIdx * constInfo.s2Size;
    }
    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = taskId & 1;
    runInfo.taskIdMod3 = taskId % 3;
    runInfo.s2LoopLimit = s2LoopLimit;

    runInfo.actualS1Size = runParam.actualS1Size;
    runInfo.actualS2Size = runParam.actualS2Size;
    runInfo.attentionOutOffset = runParam.attentionOutOffset;
    runInfo.queryOffset = runParam.tensorQOffset;
    runInfo.qRopeOffset = runParam.qRopeNBGOffset;
    if constexpr (isInfer) {
        runInfo.sOuterOffset = runParam.sOuterOffset;
    }
    this->ComputeBmm1Tail(runInfo, runParam);
    runInfo.sparseBase = runParam.sparseBase;
    runInfo.actSparseLen = runParam.actSparseLen;
    const int64_t sparseIdx = runParam.sparseBase + s2LoopCount;
    const int64_t sparseBlockIdx = static_cast<int64_t>(sparseIndicesAddr[sparseIdx]);
    const int64_t sparseS2TokenOffset = sparseBlockIdx * static_cast<int64_t>(s2BaseSize);
    runInfo.sparseS2TokenOffset = sparseS2TokenOffset;
    int64_t sparseS2RealSize = runInfo.actualS2Size - sparseS2TokenOffset;
    sparseS2RealSize = Min(static_cast<int64_t>(s2BaseSize), sparseS2RealSize);
    if (sparseS2RealSize < 0) {
        sparseS2RealSize = 0;
    }
    runInfo.s2RealSize = sparseS2RealSize;
    runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
    GetDerived()->InitUniqueRunInfo(runParam, runInfo);
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernelBase<ChildClass, CubeBlockType, VecBlockType>::ComputeBmm1Tail(
    RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam)
{
    // ------------------------S1 Base Related---------------------------
    runInfo.s1RealSize = runParam.s1RealSize;
    runInfo.s1RealSizeAlign32 = runParam.s1RealSizeAlign32;
    runInfo.halfS1RealSize = runParam.halfS1RealSize;
    runInfo.firstHalfS1RealSize = runParam.firstHalfS1RealSize;

    runInfo.vec2S1BaseSize = runInfo.halfS1RealSize;  // D>128 这里需要适配

    // ------------------------S2 Base Related----------------------------
    runInfo.s2RealSize = s2BaseSize;
    runInfo.s2AlignedSize = runInfo.s2RealSize;
    if constexpr (enableKVPrefix) {
        if ((runInfo.s2LoopCount + runInfo.s2StartIdx / s2BaseSize) < constInfo.prefixLoopCount) {
            if (runInfo.s2StartIdx + (runInfo.s2LoopCount + 1) * runInfo.s2RealSize > constInfo.actualKVPrefixSize) {
                runInfo.s2RealSize = constInfo.actualKVPrefixSize - runInfo.s2LoopCount * runInfo.s2RealSize - runInfo.s2StartIdx;
                runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
            }
        } else {
            if (runInfo.s2StartIdx + (runInfo.s2LoopCount - constInfo.prefixLoopCount + 1) * runInfo.s2RealSize > runInfo.s2EndIdx - constInfo.actualKVPrefixSize) {
                runInfo.s2RealSize = (runInfo.s2EndIdx - constInfo.actualKVPrefixSize) - (runInfo.s2LoopCount - constInfo.prefixLoopCount) * runInfo.s2RealSize - runInfo.s2StartIdx;
                runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
            }
        }
    } else {
        if (runInfo.s2StartIdx + (runInfo.s2LoopCount + 1) * runInfo.s2RealSize > runInfo.s2EndIdx) {
            runInfo.s2RealSize = runInfo.s2EndIdx - runInfo.s2LoopCount * runInfo.s2RealSize - runInfo.s2StartIdx;
            runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
        }
    }
}
}

namespace BaseApi {
template <typename CubeBlockType, typename VecBlockType>
class QuantBlockSparseAttentionKernel : public QuantBlockSparseAttentionKernelBase<QuantBlockSparseAttentionKernel<CubeBlockType, VecBlockType>, CubeBlockType, VecBlockType> {
public:
    ARGS_TRAITS;
    static constexpr ImplModeEnum implMode = kBsaImplMode;
    static constexpr PseTypeEnum pseMode = kBsaPseMode;
    static constexpr bool hasDrop = kBsaHasDrop;
    static constexpr bool hasRope = kBsaHasRope;
    static constexpr bool isInfer = kBsaIsInfer;
    static constexpr bool enableKVPrefix = kBsaEnableKVPrefix;
    using BaseClass = QuantBlockSparseAttentionKernelBase<QuantBlockSparseAttentionKernel<CubeBlockType, VecBlockType>, CubeBlockType, VecBlockType>;
    /* =====================UB变量==================== */
    __aicore__ inline void InitUniqueConstInfo();
    __aicore__ inline void InitUniqueRunInfo(const RunParamStr<isInfer> &runParam,
        RunInfo<isInfer> &runInfo);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessMainLoop();

private:
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndex, int64_t gS1Index,
                                                    RunParamStr<isInfer> &runParam);
    __aicore__ inline void ComputeSparseRunParam(RunParamStr<isInfer> &runParam);
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
QuantBlockSparseAttentionKernel<CubeBlockType, VecBlockType>::InitUniqueConstInfo()
{
    this->constInfo.isRowInvalid = this->sharedParams.isRowInvalid;
    this->constInfo.headNumRatio = this->sharedParams.headNumRatio;
    this->constInfo.isGqa = this->sharedParams.isGqa;
    this->constInfo.isPfaGS1Merge = this->sharedParams.isPfaGS1Merge;
    this->constInfo.isKvContinuous = this->sharedParams.isKvContinuous;
    this->constInfo.actualSeqLenSize = this->sharedParams.actualSeqLengthsSize;
    this->constInfo.actualSeqLenKVSize = this->sharedParams.actualSeqLengthsKVSize;
    this->constInfo.isActualLenDimsNull = static_cast<bool>(this->sharedParams.isActualSeqLengthsNull);
    this->constInfo.isActualLenDimsKVNull = static_cast<bool>(this->sharedParams.isActualSeqLengthsKVNull);
    this->constInfo.isQHasLeftPadding = static_cast<bool>(this->sharedParams.isQHasLeftPadding);
    this->constInfo.isKVHasLeftPadding = static_cast<bool>(this->sharedParams.isKVHasLeftPadding);
    // pageAttention
    if constexpr (isPa) {
        this->constInfo.blockTableDim2 = this->sharedParams.blockTableDim2;
        this->constInfo.blockSize = this->sharedParams.blockSize;
        this->constInfo.paLayoutType = this->sharedParams.paLayoutType;
        this->constInfo.paBlockNumSum = this->sharedParams.paBlockNumSum;
        this->constInfo.paBlockStride = this->sharedParams.paBlockStride;
    }

    this->constInfo.transposeLayout = this->sharedParams.transposeLayout;
    if (this->constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BNSD_BSND) ||
        this->constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::NTD_TND)) {
        this->constInfo.attentionOutStride =
            (this->constInfo.n2GDv - this->constInfo.dSizeV) * sizeof(OUTPUT_T);
    } else if (this->constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BSND_BNSD) ||
        this->constInfo.transposeLayout == static_cast<uint32_t>(TransposeLayoutEnum::BSH_BNSD)) {
        this->constInfo.attentionOutStride = 0;
    }

    // prefix
    if constexpr (enableKVPrefix) {
        this->constInfo.prefixLoopCount = (this->constInfo.actualKVPrefixSize + this->constInfo.s2BaseSize - 1) / this->constInfo.s2BaseSize;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
QuantBlockSparseAttentionKernel<CubeBlockType, VecBlockType>::InitUniqueRunInfo(
    const RunParamStr<isInfer> &runParam, RunInfo<isInfer> &runInfo)
{
    InitTaskParamByRun<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, runInfo);
    ComputeOffset<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, this->constInfo, runInfo.s2LoopCount + runInfo.s2StartIdx / this->constInfo.s2BaseSize, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernel<CubeBlockType, VecBlockType>::ProcessMainLoop()
{
    int32_t actualCoreNums = this->sharedParams.coreNum;
    if (this->aicIdx >= actualCoreNums) {
        return;
    }
    // 确定核内切分起点
    int64_t gS1StartIdx;
    uint32_t bnStartIdx;
    uint32_t bnEndIdx;
    int64_t s2LoopLimit;
    int64_t nextGs1Idx = this->sharedParams.multiCoreInnerLimit;
    bnStartIdx = this->sharedParams.bnStartIdx;
    gS1StartIdx = this->sharedParams.multiCoreInnerOffset;
    bnEndIdx = this->sharedParams.bnEndIdx;
    if (nextGs1Idx != 0) {
        bnEndIdx++;
    }
    int64_t taskId = 0;
    bool notLast = true;
    bool isLastBmm1 = false;
    LocalTensor<INPUT_T> scmTensor[2];
    RunInfo<isInfer> runInfo[4];
    RunParamStr<isInfer> runParam;

    // 注意这里不等于0是因为，推理的在SetRunInfo中第一次也需要赋值runInfo.s1oIdx，boIdx，n2oIdx，goIdx
    // 训练这些值在multiCoreInnerIdx = 0的时候都是0，两边不统一
    int64_t multiCoreInnerIdx = 1;
    for (uint32_t bnIdx = bnStartIdx; bnIdx < bnEndIdx; ++bnIdx) {
        bool lastBN = (bnIdx == bnEndIdx - 1);
        runParam.boIdx = bnIdx / this->constInfo.n2Size;
        runParam.n2oIdx = bnIdx % this->constInfo.n2Size;
        ComputeParamBatch<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, this->constInfo, this->attenMaskInfo,
            this->keyGm, this->actualSeqQlenAddr, this->actualSeqKvlenAddr);
        ComputeS1LoopInfo<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(runParam, this->constInfo, lastBN,
                                                                      nextGs1Idx);
        int64_t tempGS1End = lastBN ? (runParam.s1LoopTimes + 3) : runParam.s1LoopTimes;
        for (int64_t gS1Index = gS1StartIdx; gS1Index < tempGS1End; ++gS1Index) {
            bool notLastThreeLoop = true;
            bool notLastTwoLoop = true;
            if (lastBN) {
                int32_t extraGS1 = gS1Index - runParam.s1LoopTimes;
                switch (extraGS1) {
                    case -1:
                        isLastBmm1 = true;
                        break;
                    case 0:
                        notLastThreeLoop = false;
                        break;
                    case 1:
                        notLastThreeLoop = false;
                        notLastTwoLoop = false;
                        break;
                    case 2:
                        notLast = false;
                        notLastThreeLoop = false;
                        notLastTwoLoop = false;
                        break;
                    default:
                        break;
                }
            }
            if (notLastThreeLoop) {
                this->ComputeAxisIdxByBnAndGs1(bnIdx, gS1Index, runParam);
                bool s1NoNeedCalc = ComputeParamS1<CHILD_SPEC_TEMPLATE_ARGS, BaseClass::useDn, BaseClass::enableKVPrefix>(
                    runParam, this->constInfo, gS1Index, this->actualSeqQlenAddr);
                this->ComputeSparseRunParam(runParam);
                bool s2NoNeedCalc = runParam.actSparseLen <= 0 || runParam.actualS2Size <= 0;
                // s1和s2有任意一个不需要算, 则continue, 如果是当前核最后一次循环，则补充计算taskIdx+2的部分
                if (s1NoNeedCalc || s2NoNeedCalc) {
                    continue;
                }
                s2LoopLimit = runParam.actSparseLen - 1;
            } else {
                s2LoopLimit = 0;
            }

            for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
                if (notLastThreeLoop) {
                    RunInfo<isInfer> &runInfo1 = runInfo[taskId & 3];
                    this->SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);
                    if ASCEND_IS_AIC {
                        this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), runInfo1, this->constInfo);
                    }
                }
                if (taskId > 0 && notLastTwoLoop) {
                    if ASCEND_IS_AIV {
                        auto &runInfo3 = runInfo[(taskId + 3) & 3];
                        this->vecBlock.ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), runInfo3,
                            this->constInfo);
                    }
                }
                if (taskId > 1 && notLast) {
                    if ASCEND_IS_AIC {
                        RunInfo<isInfer> &runInfo2 = runInfo[(taskId + 2) & 3];
                        if constexpr (BaseClass::bmm2Write2Ub) {
                            this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers, runInfo2,
                                this->constInfo);
                        } else {
                            this->cubeBlock.IterateBmm2(this->bmm2ResGmBuffers.Get(), this->l1PBuffers, runInfo2,
                                this->constInfo);
                        }
                    }
                }
                if (taskId > 2) {
                    if ASCEND_IS_AIV {
                        RunInfo<isInfer> &runInfo3 = runInfo[(taskId + 1) & 3];
                        if constexpr (BaseClass::bmm2Write2Ub) {
                            this->vecBlock.ProcessVec2(this->bmm2Buffers.Get(), runInfo3, this->constInfo);
                        } else {
                            this->vecBlock.ProcessVec2(this->bmm2ResGmBuffers.Get(), runInfo3, this->constInfo);
                        }
                    }
                }
                ++taskId;
            }
            ++multiCoreInnerIdx;
        }
        gS1StartIdx = 0;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernel<CubeBlockType, VecBlockType>::Process()
{
    // SyncAll Cube和Vector都需要调用
    if (this->sharedParams.needInit) {
        SyncAll<false>();
    }
    ProcessMainLoop();
}

// =========================================== private functions ===========================================
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernel<CubeBlockType, VecBlockType>::ComputeSparseRunParam(
    RunParamStr<isInfer> &runParam)
{
    runParam.s2LineStartIdx = 0;
    runParam.s2LineEndIdx = runParam.actualS2Size;
    const int64_t sparseCount = this->sharedParams.s2SparseValidSize;
    const int64_t qbIdx = runParam.s1oIdx;
    const int64_t sparseSeqLenIdx =
        ((runParam.boIdx * this->constInfo.n2Size + runParam.n2oIdx) * this->constInfo.gSize + runParam.goIdx) *
            this->constInfo.s1OuterSize +
        qbIdx;
    runParam.sparseBase = sparseSeqLenIdx * sparseCount;
    runParam.actSparseLen = this->sparseSeqLenAddr[sparseSeqLenIdx];
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantBlockSparseAttentionKernel<CubeBlockType, VecBlockType>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t gS1Index, RunParamStr<isInfer> &runParam)
{
    constexpr uint64_t fp8QBlockSize = 128U; // 128 is SOuterSize
    constexpr uint64_t fp8KvBlockSize = 256U; // 256 is SInnerSize
    if constexpr (q_layout == LayOutTypeEnum::LAYOUT_NTD) {
        if (runParam.boIdx == 0 || runParam.boIdx == 1) {
            this->s1ScaleNumAcc = runParam.boIdx == 0 ? 0 : CeilDiv(this->actualSeqQlenAddr[0], fp8QBlockSize);
            this->s2ScaleNumAcc = runParam.boIdx == 0 ? 0 : CeilDiv(this->actualSeqKvlenAddr[0], fp8KvBlockSize);
            this->s1SizeAcc = runParam.boIdx == 0 ? 0 : this->actualSeqQlenAddr[0];
            this->s2SizeAcc = runParam.boIdx == 0 ? 0 : this->actualSeqKvlenAddr[0];
        } else {
            this->s1ScaleNumAcc = CeilDiv(this->actualSeqQlenAddr[0], fp8QBlockSize);
            this->s2ScaleNumAcc = CeilDiv(this->actualSeqKvlenAddr[0], fp8KvBlockSize);
            for (uint32_t boIdx = 1; boIdx < runParam.boIdx; boIdx++) {
                this->s1ScaleNumAcc += CeilDiv(this->actualSeqQlenAddr[boIdx] - this->actualSeqQlenAddr[boIdx - 1], fp8QBlockSize);
                this->s2ScaleNumAcc += CeilDiv(this->actualSeqKvlenAddr[boIdx] - this->actualSeqKvlenAddr[boIdx - 1], fp8KvBlockSize);
            }
            this->s1SizeAcc = this->actualSeqQlenAddr[runParam.boIdx - 1];
            this->s2SizeAcc = this->actualSeqKvlenAddr[runParam.boIdx - 1];
        }
    }
    // GS1合轴时，g轴信息包含在gS1中；GS1不合轴时，g轴信息包含在bn2g中；
    if (this->constInfo.isGqa) {
        runParam.goIdx = gS1Index / this->constInfo.s1OuterSize;
    } else {
        runParam.goIdx = bnIndex % this->constInfo.headNumRatio;
    }
    runParam.s1oIdx = gS1Index % this->constInfo.s1OuterSize;
}
}
#endif // QUANT_BLOCK_SPARSE_ATTENTION_KERNEL_H_
