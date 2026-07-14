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
 * \file fa_block_cube.h
 * \brief
 */
#ifndef FA_BLOCK_CUBE_H
#define FA_BLOCK_CUBE_H
#include "../memcopy/offset_calculator.h"
#include "../matmul/matmul.h"
#include "../vector/vector.h"
#include "../memcopy/memory_copy.h"

#include "../fa_kernel_public.h"
#include "kernel_operator_list_tensor_intf.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace fa_kernel::hardware;
using namespace fa_kernel::config;
using namespace fa_kernel::layout;
using namespace fa_kernel;
using namespace fa_base_matmul;

namespace BaseApi {

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128>
class FABlockCube {
public:
    /* ============确定L0C的类型============= */
    template <uint32_t mBaseSize, uint32_t s2BaseSize, uint32_t dVBaseSize>
    struct L0CBuffSel {
        using Type = std::conditional_t<(mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                                         mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4),
                                        BufferPolicy4buff<BufferType::L0C>, BufferPolicyDB<BufferType::L0C>>;
    };

    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr uint32_t l1BaseD = 128;
    static constexpr LayOutTypeEnum LAYOUT = layout;

    static constexpr FixpipeConfig COMMON_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, true};
    static constexpr GmFormat Q_FORMAT = GmFormat::BSNGD;
    static constexpr GmFormat KV_FORMAT = GmFormat::BSND;

    using ROPE_T = INPUT_T;
    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;
    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;

    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_ABUF_POLICY_T = BufferPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_ABUF_T = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;

    using L1KvType = BufferPolicyDB<BufferType::L1>;
    using L1QType = BufferPolicyDB<BufferType::L1>;
    using L0AType = BufferPolicyDB<BufferType::L0A>;
    using L0BType = BufferPolicyDB<BufferType::L0B>;
    using L0CType = BufferPolicy4buff<BufferType::L0C>;

    using ConstInfoX = ConstInfo_t<FaKernelType::NO_QUANT>;

    TPipe *tPipe = nullptr;

    /* =====================GM变量(with layout)==================== */
    FaGmTensor<Q_T, Q_FORMAT> queryGm;
    FaGmTensor<KV_T, KV_FORMAT> keyGm;
    FaGmTensor<KV_T, KV_FORMAT> valueGm;
    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    GlobalTensor<uint64_t> actualSeqLengthsGm;

    /* =====================LocalBuffer变量====================*/
    BufferManager<BufferType::L1> *l1BufferManagerPtr;
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BufferManager<BufferType::L0C> l0cBufferManager;

    L1QType l1QBuffers;
    L1KvType l1KBuffers;
    L1KvType l1VBuffers;

    L0AType mmL0ABuffers;
    L0BType mmL0BBuffers;
    L0CType mmL0CBuffers;

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    const ConstInfoX &constInfo;

    /*============================================================================== */
    __aicore__ inline FABlockCube(ConstInfoX &constInfo) : constInfo(constInfo){};

    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> *l1BuffMgr, __gm__ uint8_t *query,
                                         __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *actualSeqQlenAddr,
                                         __gm__ uint8_t *actualSeqKvlenAddr)
    {
        tPipe = pipe;
        l1BufferManagerPtr = l1BuffMgr;
        InitCubeInput(query, key, value, actualSeqQlenAddr, actualSeqKvlenAddr);
    }

    __aicore__ inline void InitBuffers()
    {
        constexpr uint32_t mm1LeftSize = mBaseSize * dBaseSize * sizeof(Q_T);
        constexpr uint32_t mm1RightSize = dBaseSize * s2BaseSize * sizeof(KV_T);
        constexpr uint32_t mm2RightSize = (uint32_t)dVTemplateType * s2BaseSize * sizeof(KV_T);
        l1QBuffers.Init((*l1BufferManagerPtr), mm1LeftSize);
        l1KBuffers.Init((*l1BufferManagerPtr), mm1RightSize);
        l1VBuffers.Init((*l1BufferManagerPtr), mm2RightSize);

        // L0A B C 当前写死，能否通过基础api获取
        l0aBufferManager.Init(tPipe, 65536);  // 64 * 1024
        l0bBufferManager.Init(tPipe, 65536);  // 64 * 1024
        l0cBufferManager.Init(tPipe, 262144); // 256 * 1024
        // L0A B C当前写死，要改成通过计算获取
        mmL0ABuffers.Init(l0aBufferManager, 32 * 1024);
        mmL0BBuffers.Init(l0bBufferManager, 32 * 1024);

        if constexpr (mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                      mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4) {
            mmL0CBuffers.Init(l0cBufferManager, (L0C_SIZE / NUM_4) * KB_TO_BYTES);
        } else {
            mmL0CBuffers.Init(l0cBufferManager, (L0C_SIZE / NUM_2) * KB_TO_BYTES);
        }
    }

    __aicore__ inline void InitCubeInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                         __gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *actualSeqKvlenAddr)
    {
        if (constInfo.actualSeqLenSize != 0) {
            actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqQlenAddr, constInfo.actualSeqLenSize);
        }
        if (constInfo.actualSeqLenKVSize != 0) {
            actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualSeqKvlenAddr, constInfo.actualSeqLenKVSize);
        }

        InitQBuffer(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size, constInfo.dSize,
                    actualSeqLengthsGmQ, constInfo.actualSeqLenSize, queryGm, query);

        keyPtr = key;
        valuePtr = value;
        InitKVBuffer(constInfo.bSize, constInfo.s2Size, actualSeqLengthsGm, constInfo.actualSeqLenKVSize,
                     constInfo.n2Size, constInfo.blockSize, constInfo.dSize, keyGm, key);
        InitKVBuffer(constInfo.bSize, constInfo.s2Size, actualSeqLengthsGm, constInfo.actualSeqLenKVSize,
                     constInfo.n2Size, constInfo.blockSize, constInfo.dSizeV, valueGm, value);
    }

    __aicore__ inline void InitQBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                       uint32_t headDim, GlobalTensor<uint64_t> actualSeqLengthsGmQ,
                                       uint32_t actualLenQDims, FaGmTensor<Q_T, Q_FORMAT> &qGmTensor,
                                       __gm__ uint8_t *gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T *)gm);
        if constexpr (Q_FORMAT == GmFormat::BSNGD) {
            InitOffset(qGmTensor.offsetInfo, batchSize, n2Size, gSize, qSeqSize, headDim);
        }
    }

    __aicore__ inline void InitKVBuffer(uint32_t batchSize, uint32_t kvSeqSize,
                                        GlobalTensor<uint64_t> actualSeqLengthsGmQ, uint32_t actualLenDims,
                                        uint32_t n2Size, uint32_t kvCacheBlockSize, uint32_t headDim,
                                        FaGmTensor<KV_T, KV_FORMAT> &kvGmTensor, __gm__ uint8_t *gm)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T *)gm);

        if constexpr (KV_FORMAT == GmFormat::BSND) {
            InitOffset(kvGmTensor.offsetInfo, batchSize, n2Size, kvSeqSize, headDim, 0);
        }
    }

    __aicore__ inline void AllocEventID()
    {
        // InitBuffers阶段已完成eventId申请和SetFlag，这里为空实现
    }

    __aicore__ inline void FreeEventID()
    {
        l1QBuffers.Uninit((*l1BufferManagerPtr));
        l1KBuffers.Uninit((*l1BufferManagerPtr));
        l1VBuffers.Uninit((*l1BufferManagerPtr));
        mmL0ABuffers.Uninit(l0aBufferManager);
        mmL0BBuffers.Uninit(l0bBufferManager);
        mmL0CBuffers.Uninit(l0cBufferManager);
    }

    // copy query with full s1g and split D
    __aicore__ inline void CopyQuerySlice(const LocalTensor<Q_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfoX &runInfo)
    {
        uint32_t nopeDealSize = dRealSize;
        uint32_t ropeDealSize = 0;

        uint32_t dstStride = (runInfo.actMSize + 15) >> 4 << 4;
        if (nopeDealSize > 0) {
            FaL1Tensor<Q_T> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

            GmCoord gmCoord{.bIdx = runInfo.bIdx,
                            .n2Idx = runInfo.n2Idx,
                            .gS1Idx = runInfo.gS1Idx,
                            .dIdx = dOffset,
                            .gS1DealSize = runInfo.actMSize,
                            .dDealSize = nopeDealSize};
            CopyQueryGmToL1<Q_T, Q_FORMAT>(l1Tensor, queryGm, gmCoord);
        }
    }

    __aicore__ inline void CopyQueryTile(const LocalTensor<Q_T> &dstTensor, RunInfoX &runInfo)
    {
        uint32_t dSize = constInfo.dSize;

        CopyQuerySlice(dstTensor, 0, dSize, runInfo);
    }

    // copy key with full s2 and split D
    __aicore__ inline void CopyKeySlice(const LocalTensor<KV_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                        RunInfoX &runInfo)
    {
        uint32_t dstStride = (runInfo.actSingleLoopS2Size + 15) >> 4 << 4;

        uint32_t nopeDealSize = dRealSize;
        uint32_t ropeDealSize = 0;

        if (nopeDealSize > 0) {
            FaL1Tensor<KV_T> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

            GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                              .n2Idx = runInfo.n2Idx,
                              .s2Idx = runInfo.s2Idx,
                              .dIdx = dOffset,
                              .s2DealSize = runInfo.actSingleLoopS2Size,
                              .dDealSize = nopeDealSize};
            CopyKvGmToL1<KV_T, KV_FORMAT>(l1Tensor, keyGm, gmCoord);
        }
    }

    // copy key with full s2 and split D
    __aicore__ inline void CopyValueSlice(const LocalTensor<KV_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfoX &runInfo)
    {
        FaL1Tensor<KV_T> l1Tensor{.tensor = dstTensor, .rowCount = Align(runInfo.actSingleLoopS2Size, 16U)};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = runInfo.s2Idx,
                          .dIdx = dOffset,
                          .s2DealSize = runInfo.actSingleLoopS2Size,
                          .dDealSize = dRealSize};
        CopyKvGmToL1<KV_T, KV_FORMAT>(l1Tensor, valueGm, gmCoord);
    }

    // 全量拷贝
    __aicore__ inline void CopyKeyTile(const LocalTensor<KV_T> &dstTensor, RunInfoX &runInfo)
    {
        uint32_t dSize = constInfo.dSize;
        CopyKeySlice(dstTensor, 0, dSize, runInfo);
    }

    __aicore__ inline void CopyValueTile(const LocalTensor<KV_T> &dstTensor, RunInfoX &runInfo)
    {
        CopyValueSlice(dstTensor, 0, constInfo.dSizeV, runInfo);
    }

    __aicore__ inline void IterateBmm1(MM1_DBUF_T &outputBuf, RunInfoX &runInfo)
    {
        IterateBmm1NdL0Split(outputBuf, runInfo);
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小, 使能NZ2ND, nSize*sizeof(T) 必须是32B的倍数
        fixpipeParams.nSize = (runInfo.actSingleLoopS2Size + 7) >> 3 << 3;
        // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
        fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
        // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T)
        // 源NZ矩阵中相邻Z排布的起始地址偏移
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        fixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element
        fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, COMMON_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }

    /* 针对S1Base=128, S2Base = 128, D > 128场景，L1全载，左矩阵驻留 + L0切D + L0Db*/
    __aicore__ inline void IterateBmm1NdL0Split(MM1_DBUF_T &outputBuf, RunInfoX &runInfo)
    {
        Buffer<BufferType::L1> mm1A;
        if (unlikely(runInfo.isFirstS2Loop)) {
            mm1A = l1QBuffers.Get();
            mm1A.Wait<HardEvent::MTE1_MTE2>();
            LocalTensor<Q_T> mm1ATensor = mm1A.GetTensor<Q_T>();
            CopyQueryTile(mm1ATensor, runInfo);
            mm1A.Set<HardEvent::MTE2_MTE1>();
        } else {
            mm1A = l1QBuffers.GetPre();
            mm1A.Set<HardEvent::MTE2_MTE1>();
        }

        Buffer<BufferType::L1> mm1B = l1KBuffers.Get();
        mm1B.Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<KV_T> mm1BTensor = mm1B.GetTensor<KV_T>();

        CopyKeyTile(mm1BTensor, runInfo);
        mm1B.Set<HardEvent::MTE2_MTE1>();
        mm1A.Wait<HardEvent::MTE2_MTE1>();
        mm1B.Wait<HardEvent::MTE2_MTE1>();

        Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
        mm1ResL0C.Wait<HardEvent::FIX_M>();

        MMParam param = MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)runInfo.actSingleLoopS2Size,
                                    (uint32_t)(constInfo.dSize), false, true);
        MatmulBase<Q_T, KV_T, T, 128, 128, dBaseSize, ABLayout::MK, ABLayout::KN>(
            mm1A.GetTensor<Q_T>(), mm1B.GetTensor<KV_T>(), mmL0ABuffers, mmL0BBuffers, mm1ResL0C.GetTensor<T>(), param);

        if (unlikely(runInfo.isLastS2Loop)) {
            mm1A.Set<HardEvent::MTE1_MTE2>();
        }
        mm1B.Set<HardEvent::MTE1_MTE2>();   // 释放L1B
        mm1ResL0C.Set<HardEvent::M_FIX>();  // 通知
        mm1ResL0C.Wait<HardEvent::M_FIX>(); // 等待L0C

        outputBuf.WaitCrossCore();

        FixpipeMm1(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), runInfo);

        mm1ResL0C.Set<HardEvent::FIX_M>();
        outputBuf.SetCrossCore();
    }

    __aicore__ inline void IterateBmm2(mm2ResPos &outputBuf, MM2_ABUF_POLICY_T &inputBuf, RunInfoX &runInfo)
    {
        IterateBmm2l0Split(outputBuf, inputBuf, runInfo);
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2PartialN(const DST_TENSOR_T &dstTensor, const LocalTensor<T> &l0C, uint32_t realN,
                                              RunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB;FixpipeParamsM300:L0C→UB
        fixpipeParams.nSize = (realN + 7) >> 3 << 3;
        fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        fixpipeParams.dstStride = ((uint32_t)dVTemplateType + 15) >> 4 << 4;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, COMMON_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm2l0Split(mm2ResPos &outputBuf, MM2_ABUF_POLICY_T &inputBuf, RunInfoX &runInfo)
    {
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> mm2A = inputBuf.Get();
        Buffer<BufferType::L1> mm2B = l1VBuffers.Get();
        mm2A.WaitCrossCore();
        mm2B.Wait<HardEvent::MTE1_MTE2>(); // 占用L1B
        LocalTensor<KV_T> mm2BTensor = mm2B.GetTensor<KV_T>();
        CopyValueTile(mm2BTensor, runInfo);
        mm2B.Set<HardEvent::MTE2_MTE1>(); // 通知

        Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
        mm2ResL0C.Wait<HardEvent::FIX_M>(); // 占用
        MMParam param = {
            (uint32_t)mBaseSize,                   // singleM 128
            (uint32_t)constInfo.dSizeV,            // singleN 128
            (uint32_t)runInfo.actSingleLoopS2Size, // singleK
            false,                                 // isLeftTranspose
            false                                  // isRightTranspose
        };
        param.realM = (uint32_t)runInfo.actMSize;
        mm2B.Wait<HardEvent::MTE2_MTE1>(); // 等待

        MatmulFull<Q_T, KV_T, T, 128, (uint32_t)dVTemplateType, 128, ABLayout::MK, ABLayout::KN>(
            mm2A.GetTensor<Q_T>(), mm2BTensor, mmL0ABuffers, mmL0BBuffers, mm2ResL0C.GetTensor<T>(), param);

        mm2B.Set<HardEvent::MTE1_MTE2>();   // 释放L1B
        mm2ResL0C.Set<HardEvent::M_FIX>();  // 通知
        mm2ResL0C.Wait<HardEvent::M_FIX>(); // 等待

        outputBuf.WaitCrossCore();

        FixpipeMm2PartialN(outputBuf.template GetTensor<T>(), mm2ResL0C.GetTensor<T>(), constInfo.dSizeV, runInfo);

        mm2ResL0C.Set<HardEvent::FIX_M>(); // 释放
        outputBuf.SetCrossCore();
    }

}; // FABlockCube

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128>
class FABlockCubeDummy {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr LayOutTypeEnum LAYOUT = layout;

    using ROPE_T = INPUT_T;
    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;
    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;

    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_ABUF_POLICY_T = BufferPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_ABUF_T = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;

    using ConstInfoX = ConstInfo_t<FaKernelType::NO_QUANT>;
    __aicore__ inline FABlockCubeDummy(ConstInfoX &constInfo){};
};

} // namespace BaseApi

#endif // FA_BLOCK_CUBE_H
