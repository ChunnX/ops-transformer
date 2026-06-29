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
 * \file quant_lightning_indexer_v2_service_cube.h
 * \brief use 5 buffer for matmul l1, better pipeline
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_SERVICE_CUBE_H
#define QUANT_LIGHTNING_INDEXER_V2_SERVICE_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_lightning_indexer_v2_common.h"

namespace QLIV2Kernel {
using namespace QLIV2Common;
template <typename QLIV2T>
class QLIV2Matmul {
public:
    using Q_T = typename QLIV2T::queryType;
    using K_T = typename QLIV2T::keyType;
    using QK_T = typename QLIV2T::queryKeyType;
    using SCALE_T = typename QLIV2T::scaleType;
    static constexpr bool IS_MXFP8 = QLIV2T::isMxFp8;

    __aicore__ inline QLIV2Matmul() {};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<K_T> &keyGm,
                                               const GlobalTensor<Q_T> &queryGm,
                                               const GlobalTensor<SCALE_T> &qScaleGm = {},
                                               const GlobalTensor<SCALE_T> &kScaleGm = {});
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const QLIV2Common::RunInfo &runInfo);

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true}; // isSetFMatrix isSetPadding;
    static constexpr uint64_t KEY_BUF_NUM = 3;
    static constexpr uint64_t QUERY_BUF_NUM = 2;
    static constexpr uint64_t L0_BUF_NUM = 2;

    static constexpr uint32_t KEY_MTE1_MTE2_EVENT = EVENT_ID2;
    static constexpr uint32_t QUERY_MTE1_MTE2_EVENT = EVENT_ID5;         // KEY_MTE1_MTE2_EVENT + KEY_BUF_NUM;
    static constexpr uint32_t M_MTE1_EVENT = EVENT_ID3;

    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;
    static constexpr uint32_t FIX_M_EVENT = EVENT_ID2;
    static constexpr uint32_t M_FIX_EVENT = EVENT_ID3;

    static constexpr uint64_t M_BASIC_BLOCK = 256;
    static constexpr uint64_t D_BASIC_BLOCK = 128;

    static constexpr uint64_t M_BASIC_BLOCK_L0 = 256;
    static constexpr uint64_t D_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 128;

    static constexpr uint64_t FP8_BLOCK_CUBE = 32;
    static constexpr FixpipeConfig QLIV2_CFG_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR,
        // ROW_MAJOR: 使能NZ2ND，输出ND格式; true: 用户指定目的地址是否是UB
        true};

    static constexpr uint64_t QUERY_BUFFER_OFFSET = M_BASIC_BLOCK * D_BASIC_BLOCK;
    // MXFP8: qScale 使用独立的 L1 buffer，每个 ping-pong 需要 M_BASIC_BLOCK * D_BASIC_BLOCK / 32 个元素
    static constexpr uint64_t QSCALE_L1_PINGPONG_SIZE = M_BASIC_BLOCK * D_BASIC_BLOCK / 32; // = 1024
    // MXFP8: kScale 使用独立的 L1 buffer，每个 buffer 需要 s2BasicBlock * D_BASIC_BLOCK / 32 个元素
    static constexpr uint64_t KSCALE_L1_SIZE = S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK / 32; // = 512
    static constexpr uint64_t L0AB_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0;
    static constexpr uint64_t L0C_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0;

protected:
    __aicore__ inline void Fixp(uint64_t s1gGmOffset, uint64_t s2GmOffset, uint64_t s1gL0RealSize,
                                uint64_t s2L0RealSize, const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void ComputeL0c(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                      const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void LoadKeyToL0b(uint64_t s2L0Offset, uint64_t s2L1RealSize, uint64_t s2L0RealSize,
                                        const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL0Offset, uint64_t s1gL1RealSize,
                                          uint64_t s1gL0RealSize, const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gL1Offset,
                                       const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void KeyNd2Nz(uint64_t s2L1RealSize, uint64_t s2GmOffset, const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset,
        const QLIV2Common::RunInfo &runInfo);
    // MXFP8: 加载 qScale/kScale 到 L1
    __aicore__ inline void LoadQScaleToL1(uint64_t s1gL1RealSize, uint64_t s1gGmOffset,
                                          const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void LoadKScaleToL1(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                          const QLIV2Common::RunInfo &runInfo);
    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<Q_T> queryGm_;
    GlobalTensor<SCALE_T> qScaleGm_;
    GlobalTensor<SCALE_T> kScaleGm_;

    TBuf<TPosition::A1> bufQL1_;
    LocalTensor<Q_T> queryL1_;
    TBuf<TPosition::A1> bufQScaleL1_;
    LocalTensor<SCALE_T> qScaleL1_;
    TBuf<TPosition::B1> bufKeyL1_;
    LocalTensor<K_T> keyL1_;
    TBuf<TPosition::B1> bufKScaleL1_;
    LocalTensor<SCALE_T> kScaleL1_;

    TBuf<TPosition::A2> bufQL0_;
    LocalTensor<Q_T> queryL0_;
    TBuf<TPosition::B2> bufKeyL0_;
    LocalTensor<K_T> keyL0_;

    TBuf<TPosition::CO1> bufL0C_;
    LocalTensor<float> cL0_;

    TBuf<TPosition::VECCALC> bufUB_;
    LocalTensor<QK_T> mm1ResUB_;

    uint64_t keyL1BufIdx_ = 0;
    uint64_t queryL1Mte2BufIdx_ = 0;
    uint64_t queryL1Mte1BufIdx_ = 0;
    uint64_t l0BufIdx_ = 0;

    bool isKeyCacheValid_ = false; // L1中是否有可复用的数据
    uint64_t keyGmStart_ = 0; // L1数据对应的GM S2偏移
    uint64_t keyLoadedSize_ = 0; // L1中实际加载的S2元素数量
    uint64_t s2BasicBlock_ = 128;
    uint64_t keyBufferOffset_ = 16384; // 128*128
    
    ConstInfo constInfo_;

private:
    static constexpr bool PAGE_ATTENTION = QLIV2T::pageAttention;
};

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::InitParams(const ConstInfo &constInfo)
{
    constInfo_ = constInfo;
    s2BasicBlock_ = 128;
    keyBufferOffset_ = s2BasicBlock_ * D_BASIC_BLOCK;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(bufUB_, 2 * CeilDiv(constInfo_.mBaseSize, 2) * constInfo_.s2BaseSize * sizeof(float));
        // 大小：2(开dB) * 2 * 64 * 128 * 4 = 128KB
    mm1ResUB_ = bufUB_.Get<QK_T>();
    pipe->InitBuffer(bufQL1_, QUERY_BUF_NUM * QUERY_BUFFER_OFFSET * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, KEY_BUF_NUM * s2BasicBlock_ * D_BASIC_BLOCK * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();
    // MXFP8: qScale 使用独立 L1 buffer
    if constexpr (IS_MXFP8) {
        pipe->InitBuffer(bufQScaleL1_, QUERY_BUF_NUM * QSCALE_L1_PINGPONG_SIZE * sizeof(SCALE_T));
        qScaleL1_ = bufQScaleL1_.Get<SCALE_T>();
        pipe->InitBuffer(bufKScaleL1_, KEY_BUF_NUM * KSCALE_L1_SIZE * sizeof(SCALE_T));
        kScaleL1_ = bufKScaleL1_.Get<SCALE_T>();
    }

    pipe->InitBuffer(bufQL0_, L0_BUF_NUM * M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 * sizeof(Q_T));
    queryL0_ = bufQL0_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL0_, L0_BUF_NUM * D_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(K_T));
    keyL0_ = bufKeyL0_.Get<K_T>();

    pipe->InitBuffer(bufL0C_, L0_BUF_NUM * M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(float));
    cL0_ = bufL0C_.Get<float>();
}

template <typename QLIV2T>
__aicore__ inline void
QLIV2Matmul<QLIV2T>::InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<K_T> &keyGm,
                                   const GlobalTensor<Q_T> &queryGm,
                                   const GlobalTensor<SCALE_T> &qScaleGm,
                                   const GlobalTensor<SCALE_T> &kScaleGm)
{
    blkTableGm_ = blkTableGm;
    keyGm_ = keyGm;
    queryGm_ = queryGm;
    if constexpr (IS_MXFP8) {
        qScaleGm_ = qScaleGm;
        kScaleGm_ = kScaleGm;
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::ComputeMm1(const QLIV2Common::RunInfo &runInfo)
{
    CrossCoreWaitFlag<QLIV2Common::ConstInfo::QLIV2_SYNC_MODE4, PIPE_FIX>(QLIV2Common::ConstInfo::CROSS_VC_EVENT
        + runInfo.loop % 2);
#if CVRATIO > 1
    CrossCoreWaitFlag<QLIV2Common::ConstInfo::QLIV2_SYNC_MODE4, PIPE_FIX>(QLIV2Common::ConstInfo::CROSS_VC_EVENT
        + runInfo.loop % 2 + QLIV2Common::ConstInfo::AIV0_AIV1_OFFSET);
#endif
    uint64_t s2GmBaseOffset = runInfo.s2Idx * constInfo_.s2BaseSize;
    uint64_t s1gProcessSize = runInfo.actMBaseSize;
    uint64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    if (s2BasicBlock_ == 128) {
        for (uint64_t s2GmOffset = 0; s2GmOffset < s2ProcessSize; s2GmOffset += s2BasicBlock_) {
            WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
            uint64_t s2L1RealSize =
                s2GmOffset + s2BasicBlock_ > s2ProcessSize ? s2ProcessSize - s2GmOffset : s2BasicBlock_;
            if (PAGE_ATTENTION) {
                KeyNd2NzForPA(s2L1RealSize, s2GmBaseOffset + s2GmOffset, runInfo);
                // MXFP8: PA 路径需要绝对偏移
                if constexpr (IS_MXFP8) {
                    LoadKScaleToL1(s2L1RealSize, s2GmBaseOffset + s2GmOffset, runInfo);
                }
            } else {
                KeyNd2Nz(s2L1RealSize, s2GmOffset, runInfo);
                // MXFP8: 非 PA 路径只传循环内相对偏移（tensorKeyScaleOffset 已含 s2GmBaseOffset）
                if constexpr (IS_MXFP8) {
                    LoadKScaleToL1(s2L1RealSize, s2GmOffset, runInfo);
                }
            }

            SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            // s1gProcessSize当前必定不会超过2倍的s1g basic block
            for (uint64_t s1gGmOffset = 0; s1gGmOffset < s1gProcessSize; s1gGmOffset += M_BASIC_BLOCK) {
                uint64_t s1gL1RealSize =
                    s1gGmOffset + M_BASIC_BLOCK > s1gProcessSize ? s1gProcessSize - s1gGmOffset : M_BASIC_BLOCK;
                uint64_t s1gL1SizeAlign2G = CeilAlign(s1gL1RealSize, 2 * constInfo_.gSize);
                if (runInfo.isFirstS2InnerLoop && s2GmOffset == 0) {
                    queryL1Mte2BufIdx_++;
                    queryL1Mte1BufIdx_ = queryL1Mte2BufIdx_;
                    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte2BufIdx_ % QUERY_BUF_NUM);
                    QueryNd2Nz(s1gL1RealSize, s1gGmOffset, runInfo);
                    // MXFP8: 加载 qScale 到 L1（与 Query 数据一起）
                    if constexpr (IS_MXFP8) {
                        LoadQScaleToL1(s1gL1RealSize, s1gGmOffset, runInfo);
                    }
                    SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                    WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                } else {
                    queryL1Mte1BufIdx_ =
                        queryL1Mte2BufIdx_ - (CeilDiv(s1gProcessSize, M_BASIC_BLOCK) - 1 - (s1gGmOffset > 0));
                }
                for (uint64_t s2L1Offset = 0; s2L1Offset < s2L1RealSize; s2L1Offset += S2_BASIC_BLOCK_L0) {
                    uint64_t s2L0RealSize =
                        s2L1Offset + S2_BASIC_BLOCK_L0 > s2L1RealSize ? s2L1RealSize - s2L1Offset : S2_BASIC_BLOCK_L0;
                    for (uint64_t s1gL1Offset = 0; s1gL1Offset < s1gL1SizeAlign2G; s1gL1Offset += M_BASIC_BLOCK_L0) {
                        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        uint64_t s1gL0RealSize =
                            s1gL1Offset + M_BASIC_BLOCK_L0 > s1gL1SizeAlign2G
                                ? s1gL1SizeAlign2G - s1gL1Offset : M_BASIC_BLOCK_L0;
                        LoadQueryToL0a(s1gGmOffset, s1gL1Offset, s1gL1SizeAlign2G, s1gL0RealSize, runInfo);
                        LoadKeyToL0b(s2L1Offset, s2L1RealSize, s2L0RealSize, runInfo);

                        SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
                        WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);

                        WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        ComputeL0c(s1gL0RealSize, s2L0RealSize, runInfo);

                        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);

                        Fixp(s1gGmOffset + s1gL1Offset, s2GmOffset + s2L1Offset, s1gL0RealSize, s2L0RealSize, runInfo);
                        SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        l0BufIdx_++;
                    }
                }
                if (s2GmOffset + s2BasicBlock_ >= s2ProcessSize && runInfo.isLastS2InnerLoop) {
                    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte1BufIdx_ % QUERY_BUF_NUM);
                }
            }
            SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
            keyL1BufIdx_++;
        }
        PipeBarrier<PIPE_FIX>();
    } else if (s2BasicBlock_ == 256) {
        // 第一个s2循环 keycache置为false
        if (runInfo.isFirstS2InnerLoop) {
            isKeyCacheValid_ = false;
        }
        for (uint64_t s2GmOffset = 0; s2GmOffset < s2ProcessSize; s2GmOffset += s2BasicBlock_) {
            // 缓存命中不需要进行key的搬运
            bool keyCacheHit = isKeyCacheValid_ && (s2GmBaseOffset >= keyGmStart_) &&
                                 (s2GmBaseOffset + s2ProcessSize <= keyGmStart_ + keyLoadedSize_);
            if (!keyCacheHit) {
                WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
                // 缓存未命中，需要从GM搬到L1 min（256, 剩余s2）
                uint64_t s2TotalRemainNum = runInfo.actS2Size - s2GmBaseOffset;
                uint64_t s2L1LoadSize = (s2TotalRemainNum < s2BasicBlock_) ? s2TotalRemainNum : s2BasicBlock_;
                if (PAGE_ATTENTION) {
                    KeyNd2NzForPA(s2L1LoadSize, s2GmBaseOffset + s2GmOffset, runInfo);
                }else {
                    KeyNd2Nz(s2L1LoadSize, s2GmOffset, runInfo);
                }

                SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);

                isKeyCacheValid_ = true;
                keyGmStart_ = s2GmBaseOffset;
                keyLoadedSize_ = s2L1LoadSize;
            }
            uint64_t l1S2Offset = s2GmBaseOffset - keyGmStart_;
            uint64_t l1TotalSize = keyLoadedSize_;
            // s1gProcessSize当前必定不会超过2倍的s1g basic block
            for (uint64_t s1gGmOffset = 0; s1gGmOffset < s1gProcessSize; s1gGmOffset += M_BASIC_BLOCK) {
                uint64_t s1gL1RealSize =
                    s1gGmOffset + M_BASIC_BLOCK > s1gProcessSize ?
                                           s1gProcessSize - s1gGmOffset :
                                           M_BASIC_BLOCK;
                uint64_t s1gL1SizeAlign2G = CeilAlign(s1gL1RealSize, 2 * constInfo_.gSize);
                if (runInfo.isFirstS2InnerLoop && s2GmOffset == 0) {
                    queryL1Mte2BufIdx_++;
                    queryL1Mte1BufIdx_ = queryL1Mte2BufIdx_;
                    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte2BufIdx_ % QUERY_BUF_NUM);
                    QueryNd2Nz(s1gL1SizeAlign2G, s1gGmOffset, runInfo);
                    SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                    WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                } else {
                    queryL1Mte1BufIdx_ =
                        queryL1Mte2BufIdx_ - (CeilDiv(s1gProcessSize, M_BASIC_BLOCK) - 1 - (s1gGmOffset > 0));
                }
                uint64_t s2Boundry = l1S2Offset + s2ProcessSize;
                for (uint64_t s2L1Offset = l1S2Offset; s2L1Offset < s2Boundry; s2L1Offset += S2_BASIC_BLOCK_L0) {
                    uint64_t s2L0RealSize =
                        s2L1Offset + S2_BASIC_BLOCK_L0 > l1S2Offset + s2ProcessSize ?
                                            l1S2Offset + s2ProcessSize - s2L1Offset :
                                            S2_BASIC_BLOCK_L0;
                    for (uint64_t s1gOffset = 0; s1gOffset < s1gL1SizeAlign2G; s1gOffset += M_BASIC_BLOCK_L0) {
                        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        uint64_t s1gL0RealSize =
                            s1gOffset + M_BASIC_BLOCK_L0 > s1gL1SizeAlign2G ?
                                                    s1gL1SizeAlign2G - s1gOffset :
                                                    M_BASIC_BLOCK_L0;
                        LoadQueryToL0a(s1gGmOffset, s1gOffset, s1gL1SizeAlign2G, s1gL0RealSize, runInfo);
                        LoadKeyToL0b(s2L1Offset, l1TotalSize, s2L0RealSize, runInfo);

                        SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
                        WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
                        
                        WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        ComputeL0c(s1gL0RealSize, s2L0RealSize, runInfo);

                        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        
                        Fixp(s1gGmOffset + s1gOffset, (s2L1Offset - l1S2Offset),
                                                s1gL0RealSize, s2L0RealSize, runInfo);
                        SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        l0BufIdx_++;
                    }
                }
                if (s2GmOffset + s2BasicBlock_ >= s2ProcessSize && runInfo.isLastS2InnerLoop) {
                    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte1BufIdx_ % QUERY_BUF_NUM);
                }
            }
            bool l1FullyUsed = (s2GmBaseOffset + s2ProcessSize >= keyGmStart_ + keyLoadedSize_);
            if (l1FullyUsed || runInfo.isLastS2InnerLoop) {
                SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
                keyL1BufIdx_++;
                isKeyCacheValid_ =false;
            }
        }
        PipeBarrier<PIPE_FIX>();
    }

    CrossCoreSetFlag<QLIV2Common::ConstInfo::QLIV2_SYNC_MODE4, PIPE_FIX>(QLIV2Common::ConstInfo::CROSS_CV_EVENT
        + runInfo.loop % 2);
#if CVRATIO > 1
    CrossCoreSetFlag<QLIV2Common::ConstInfo::QLIV2_SYNC_MODE4, PIPE_FIX>(QLIV2Common::ConstInfo::CROSS_CV_EVENT
        + runInfo.loop % 2 + QLIV2Common::ConstInfo::AIV0_AIV1_OFFSET);
#endif
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::KeyNd2Nz(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                    const QLIV2Common::RunInfo &runInfo)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s2L1RealSize; // 行数
    nd2nzPara.dValue = constInfo_.headDim;
    nd2nzPara.srcDValue = constInfo_.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * keyBufferOffset_],
             keyGm_[runInfo.tensorKeyOffset + s2GmOffset * constInfo_.headDim], nd2nzPara);
}

// blkNum, blkSize, N2, D
template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                    const QLIV2Common::RunInfo &runInfo)
{
    uint64_t s2L1Offset = 0;
    while (s2L1Offset < s2L1RealSize) {
        uint64_t s2BlkId = (s2L1Offset + s2GmOffset) / constInfo_.kCacheBlockSize;
        uint64_t s2BlkOffset = (s2L1Offset + s2GmOffset) % constInfo_.kCacheBlockSize;
        uint64_t keyGmOffset = blkTableGm_.GetValue(runInfo.bIdx * constInfo_.maxBlockNumPerBatch + s2BlkId)
            * constInfo_.keyStride0 +
                               s2BlkOffset * constInfo_.headDim;

        uint64_t s2Mte2Size = s2L1RealSize - s2L1Offset;
        s2Mte2Size = s2BlkOffset + s2Mte2Size >= constInfo_.kCacheBlockSize ? constInfo_.kCacheBlockSize - s2BlkOffset
                                                                            : s2Mte2Size;
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = s2Mte2Size; // 行数
        nd2nzPara.dValue = constInfo_.headDim;
        nd2nzPara.srcDValue = constInfo_.headDim;
        nd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * keyBufferOffset_ + s2L1Offset * FP8_BLOCK_CUBE],
                 keyGm_[keyGmOffset], nd2nzPara);

        s2L1Offset += s2Mte2Size;
    }
}

// MXFP8: 加载 qScale 到独立的 L1 buffer
// qScale shape: [S1*G, D/32]，每个 token 对应 D/32=4 个 fp8_e8m0 scale
template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::LoadQScaleToL1(uint64_t s1gL1RealSize, uint64_t s1gGmOffset,
                                                         const QLIV2Common::RunInfo &runInfo)
{
    // qScale 存储在独立的 bufQScaleL1_ 中
    uint64_t scaleOffsetInBuf = (queryL1Mte2BufIdx_ % QUERY_BUF_NUM) * QSCALE_L1_PINGPONG_SIZE;
    LocalTensor<bfloat16_t> scaleL1 = qScaleL1_[scaleOffsetInBuf].template ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> scaleGmCast;
    scaleGmCast.SetGlobalBuffer((__gm__ bfloat16_t *)(qScaleGm_.GetPhyAddr()));

    uint32_t scalePerToken = constInfo_.headDim / MXFP8_BLOCK_SIZE;  // D/32 = 4
    Dn2NzParams dn2Nzparam;
    dn2Nzparam.dnNum = 1;
    dn2Nzparam.nValue = scalePerToken / FP8_TWO;  // 2个fp8_e8m0打包成1个bf16
    dn2Nzparam.dValue = s1gL1RealSize;
    dn2Nzparam.srcDnMatrixStride = 0;
    dn2Nzparam.srcDValue = scalePerToken / FP8_TWO;
    dn2Nzparam.dstNzC0Stride = scalePerToken / FP8_TWO;
    dn2Nzparam.dstNzNStride = 1;
    dn2Nzparam.dstNzMatrixStride = scalePerToken / FP8_TWO;
    uint64_t gmOffset = runInfo.tensorQScaleOffset + s1gGmOffset * scalePerToken;
    DataCopy(scaleL1, scaleGmCast[gmOffset / FP8_TWO], dn2Nzparam);
}

// MXFP8: 加载 kScale 到独立的 L1 buffer
// kScale shape: [S2, D/32]，每个 S2 token 对应 D/32=4 个 fp8_e8m0 scale
template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::LoadKScaleToL1(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                         const QLIV2Common::RunInfo &runInfo)
{
    // kScale 存储在独立的 bufKScaleL1_ 中，使用 triple buffer
    uint64_t scaleOffsetInBuf = (keyL1BufIdx_ % KEY_BUF_NUM) * KSCALE_L1_SIZE;
    LocalTensor<bfloat16_t> scaleL1 = kScaleL1_[scaleOffsetInBuf].template ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> scaleGmCast;
    scaleGmCast.SetGlobalBuffer((__gm__ bfloat16_t *)(kScaleGm_.GetPhyAddr()));

    uint32_t scalePerToken = constInfo_.headDim / MXFP8_BLOCK_SIZE;  // D/32 = 4

    if constexpr (PAGE_ATTENTION) {
        // PA_BBND: kScale 按 block 存储，需要通过 blockTable 查找物理 block
        uint64_t s2L1Offset = 0;
        while (s2L1Offset < s2L1RealSize) {
            uint64_t s2BlkId = (s2L1Offset + s2GmOffset) / constInfo_.kCacheBlockSize;
            uint64_t s2BlkOffset = (s2L1Offset + s2GmOffset) % constInfo_.kCacheBlockSize;
            uint64_t physicalBlkId = blkTableGm_.GetValue(runInfo.bIdx * constInfo_.maxBlockNumPerBatch + s2BlkId);

            uint64_t s2Mte2Size = s2L1RealSize - s2L1Offset;
            s2Mte2Size = s2BlkOffset + s2Mte2Size >= constInfo_.kCacheBlockSize
                         ? constInfo_.kCacheBlockSize - s2BlkOffset : s2Mte2Size;

            // kScale GM offset: physicalBlkId * stride0 + s2BlkOffset * scalePerToken
            uint64_t gmOffset = physicalBlkId * constInfo_.keyDequantScaleStride0 + s2BlkOffset * scalePerToken;

            Dn2NzParams dn2Nzparam;
            dn2Nzparam.dnNum = 1;
            dn2Nzparam.nValue = scalePerToken / FP8_TWO;
            dn2Nzparam.dValue = s2Mte2Size;
            dn2Nzparam.srcDnMatrixStride = 0;
            dn2Nzparam.srcDValue = scalePerToken / FP8_TWO;
            dn2Nzparam.dstNzC0Stride = scalePerToken / FP8_TWO;
            dn2Nzparam.dstNzNStride = 1;
            dn2Nzparam.dstNzMatrixStride = scalePerToken / FP8_TWO;
            DataCopy(scaleL1[s2L1Offset * scalePerToken / FP8_TWO], scaleGmCast[gmOffset / FP8_TWO], dn2Nzparam);

            s2L1Offset += s2Mte2Size;
        }
    } else {
        Dn2NzParams dn2Nzparam;
        dn2Nzparam.dnNum = 1;
        dn2Nzparam.nValue = scalePerToken / FP8_TWO;
        dn2Nzparam.dValue = s2L1RealSize;
        dn2Nzparam.srcDnMatrixStride = 0;
        dn2Nzparam.srcDValue = scalePerToken / FP8_TWO;
        dn2Nzparam.dstNzC0Stride = scalePerToken / FP8_TWO;
        dn2Nzparam.dstNzNStride = 1;
        dn2Nzparam.dstNzMatrixStride = scalePerToken / FP8_TWO;
        uint64_t gmOffset = runInfo.tensorKeyScaleOffset + s2GmOffset * scalePerToken;
        DataCopy(scaleL1, scaleGmCast[gmOffset / FP8_TWO], dn2Nzparam);
    }
}

// batch, s1, n2, g, d
template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gGmOffset,
                                                 const QLIV2Common::RunInfo &runInfo)
{
    uint64_t dstNzC0Stride = CeilAlign(s1gL1RealSize, 2 * constInfo_.gSize);
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s1gL1RealSize; // 行数
    nd2nzPara.dValue = constInfo_.headDim;
    nd2nzPara.srcDValue = constInfo_.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(dstNzC0Stride, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(queryL1_[(queryL1Mte2BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET],
             queryGm_[runInfo.tensorQueryOffset + s1gGmOffset * constInfo_.headDim], nd2nzPara);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::LoadQueryToL0a(uint64_t s1gGmOffset, uint64_t s1gL1Offset,
    uint64_t s1gL1RealSize,
                                                     uint64_t s1gL0RealSize, const QLIV2Common::RunInfo &runInfo)
{
    // 公共部分：LoadData2DParamsV2 初始化
    LoadData2DParamsV2 loadData2DParams;
    loadData2DParams.mStartPosition = CeilDiv(s1gL1Offset, BLOCK_CUBE);
    loadData2DParams.kStartPosition = 0;
    loadData2DParams.mStep = CeilDiv(s1gL0RealSize, BLOCK_CUBE);
    loadData2DParams.kStep = CeilDiv(constInfo_.headDim, FP8_BLOCK_CUBE);
    loadData2DParams.srcStride = CeilDiv(s1gL1RealSize, BLOCK_CUBE);
    loadData2DParams.dstStride = CeilDiv(s1gL0RealSize, BLOCK_CUBE);
    loadData2DParams.ifTranspose = false;

    if constexpr (IS_MXFP8) {
        // MXFP8: 使用 LoadData Mx 变体，同时加载数据和 scale 到 L0
        LocalTensor<mx_fp8_e4m3_t> dstTensor =
            queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET].template ReinterpretCast<mx_fp8_e4m3_t>();
        // qScale 从独立的 bufQScaleL1_ 读取
        uint64_t scaleOffsetInBuf = (queryL1Mte1BufIdx_ % QUERY_BUF_NUM) * QSCALE_L1_PINGPONG_SIZE;
        LocalTensor<fp8_e8m0_t> scaleL1 = qScaleL1_[scaleOffsetInBuf].template ReinterpretCast<fp8_e8m0_t>();

        LoadData2DMxParams loadAScaleParam;
        loadAScaleParam.xStartPosition = 0;
        loadAScaleParam.yStartPosition = 0;
        loadAScaleParam.xStep = CeilDiv(s1gL0RealSize, BLOCK_CUBE);
        loadAScaleParam.yStep = CeilDiv(constInfo_.headDim, FP8_BLOCK_CUBE * FP8_TWO);
        // srcStride 基于 K 维度(headDim)，参考 MLA Prolog: CeilDiv(kL1Size, K_STEP_SIZE_32 * FP8_TWO)
        loadAScaleParam.srcStride = CeilDiv(constInfo_.headDim, FP8_BLOCK_CUBE * FP8_TWO);
        loadAScaleParam.dstStride = loadAScaleParam.yStep;
        LoadData(dstTensor,
                 queryL1_[(queryL1Mte1BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET],
                 scaleL1, loadData2DParams, loadAScaleParam);
    } else {
        LoadData(queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
                 queryL1_[(queryL1Mte1BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET], loadData2DParams);
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::LoadKeyToL0b(uint64_t s2L1Offset, uint64_t s2L1RealSize,
    uint64_t s2L0RealSize,
                                                   const QLIV2Common::RunInfo &runInfo)
{
    // 公共部分：LoadData2DParamsV2 初始化（除了 ifTranspose）
    LoadData2DParamsV2 loadData2DParams;
    loadData2DParams.mStartPosition = CeilDiv(s2L1Offset, BLOCK_CUBE);
    loadData2DParams.kStartPosition = 0;
    loadData2DParams.mStep = CeilDiv(s2L0RealSize, BLOCK_CUBE);
    loadData2DParams.kStep = CeilDiv(constInfo_.headDim, FP8_BLOCK_CUBE);
    loadData2DParams.srcStride = CeilDiv(s2L1RealSize, BLOCK_CUBE);
    loadData2DParams.dstStride = CeilDiv(s2L0RealSize, BLOCK_CUBE);
    loadData2DParams.ifTranspose = false;

    if constexpr (IS_MXFP8) {
        // MXFP8: 使用 LoadData Mx 变体，同时加载数据和 scale 到 L0
        LocalTensor<mx_fp8_e4m3_t> dstTensor =
            keyL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET].template ReinterpretCast<mx_fp8_e4m3_t>();
        // kScale 从独立的 bufKScaleL1_ 读取，使用 triple buffer
        uint64_t scaleOffsetInBuf = (keyL1BufIdx_ % KEY_BUF_NUM) * KSCALE_L1_SIZE;
        LocalTensor<fp8_e8m0_t> scaleL1 = kScaleL1_[scaleOffsetInBuf].template ReinterpretCast<fp8_e8m0_t>();

        LoadData2DMxParams loadDataMxParams;
        loadDataMxParams.xStartPosition = 0;
        loadDataMxParams.yStartPosition = 0;
        loadDataMxParams.xStep = CeilDiv(s2L0RealSize, BLOCK_CUBE);
        loadDataMxParams.yStep = CeilDiv(constInfo_.headDim, FP8_BLOCK_CUBE * FP8_TWO);
        // srcStride 基于 K 维度(headDim)，参考 MLA Prolog: CeilDiv(kL1Size, K_STEP_SIZE_32 * FP8_TWO)
        loadDataMxParams.srcStride = CeilDiv(constInfo_.headDim, FP8_BLOCK_CUBE * FP8_TWO);
        loadDataMxParams.dstStride = loadDataMxParams.yStep;
        LoadData(dstTensor,
                 keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * keyBufferOffset_],
                 scaleL1, loadData2DParams, loadDataMxParams);
    } else {
        LoadData(keyL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
                 keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * keyBufferOffset_], loadData2DParams);
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::ComputeL0c(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                                const QLIV2Common::RunInfo &runInfo)
{
    MmadParams mmadParams;
    mmadParams.m = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    mmadParams.n = s2L0RealSize;
    mmadParams.k = constInfo_.headDim;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    if constexpr (IS_MXFP8) {
        // MXFP8: mmad_mx 硬件指令要求 n 必须是 16 的倍数
        // 当 actS2Size 不是 16 的倍数时（如 red_2: actS2Size=8, red_3: actS2Size=1~4），
        // 尾部 S2 块的 s2L0RealSize 不满足对齐要求，需要向上对齐并清零 L0B 中的 extra columns
        uint64_t alignedN = CeilAlign(s2L0RealSize, (uint64_t)BLOCK_CUBE);
        if (alignedN > s2L0RealSize) {
            mmadParams.n = alignedN;
            // fp8 NZ block 内部布局: [BLOCK_CUBE, FP8_BLOCK_CUBE] = [16, 32], 512 bytes/block
            // 每列 (n维度) 在 block 内占 BLOCK_CUBE=16 bytes，列 b 起始偏移 = b * 16
            // K 方向有 headDim/BLOCK_CUBE 个 NZ block 顺序排列
            // extra columns 在每个 NZ block 内不连续，需逐 block 清零
            LocalTensor<uint8_t> keyL0Bytes =
                keyL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET].template ReinterpretCast<uint8_t>();
            uint64_t nzBlockSize = (uint64_t)BLOCK_CUBE * FP8_BLOCK_CUBE;  // 512 bytes
            uint64_t numKBlocks = constInfo_.headDim / BLOCK_CUBE;
            uint64_t extraInBlockOffset = s2L0RealSize * BLOCK_CUBE;  // block 内 extra columns 起始
            uint64_t extraBytesPerBlock = (FP8_BLOCK_CUBE - s2L0RealSize) * BLOCK_CUBE;
            for (uint64_t kBlk = 0; kBlk < numKBlocks; kBlk++) {
                Duplicate(keyL0Bytes[kBlk * nzBlockSize + extraInBlockOffset], (uint8_t)0, extraBytesPerBlock);
            }
        }
        // MXFP8: 使用 mx_fp8_e4m3_t 类型触发 mmad_mx 硬件指令
        LocalTensor<mx_fp8_e4m3_t> qL0Mx =
            queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET].template ReinterpretCast<mx_fp8_e4m3_t>();
        LocalTensor<mx_fp8_e4m3_t> kL0Mx =
            keyL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET].template ReinterpretCast<mx_fp8_e4m3_t>();
        Mmad(cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET], qL0Mx, kL0Mx, mmadParams);
    } else {
        Mmad(cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET], queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
             keyL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET], mmadParams);
    }
    // if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
        PipeBarrier<PIPE_M>();
    // }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::Fixp(uint64_t s1gGmOffset, uint64_t s2GmOffset, uint64_t s1gL0RealSize,
                                           uint64_t s2L0RealSize, const QLIV2Common::RunInfo &runInfo)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT + l0BufIdx_ % L0_BUF_NUM);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT + l0BufIdx_ % L0_BUF_NUM);

    if constexpr (std::is_same_v<QK_T, float>) {
        // s1gL0RealSize：2*gSize(128)对齐, 最大256
        // s2L0RealSize <= S2_BASIC_BLOCK_L0, 未约束
        uint32_t nSize = (s2L0RealSize + 7) >> 3 << 3; // 32B对齐
        uint32_t mSize = (s1gL0RealSize + 1) >> 1 << 1;
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // 固定参数
        fixpipeParams.mSize = mSize;
        fixpipeParams.srcStride = mSize; // 已16对齐
        fixpipeParams.dstStride = UB_BANK_DEPTH_STRIDE / sizeof(QK_T); // 落到同一个bank
        // 单目标模式（CVRATIO=1，不需要拆M）
        fixpipeParams.dualDstCtl = 0;
        fixpipeParams.reluEn = true;   // 启用随路ReLU

        // nSize已保证N方向32B对齐
        if (nSize <= (256 / sizeof(float))) {
            // N方向小于一个bank(256B), 只需搬一个ND块, 且不用补齐
            fixpipeParams.nSize = nSize;
            fixpipeParams.params.ndNum = 1;
            fixpipeParams.params.srcNdStride = 0;
            fixpipeParams.params.dstNdStride = 0;
        } else {
            // N方向在(256B, 512B]范围， 直接按512B搬, 注意此时不能开unitflag
            fixpipeParams.nSize = S2_BASIC_BLOCK_L0 / 2; // 分2个ND搬, S2_BASIC_BLOCK_L0不为128会有问题
            fixpipeParams.params.ndNum = 2;
            fixpipeParams.params.srcNdStride = ((fixpipeParams.mSize + 15) / 16) * fixpipeParams.nSize;
            // dstNdStride = 第一个ND块的总占用 = mSize × dstStride（含每行padding）
            fixpipeParams.params.dstNdStride = fixpipeParams.mSize * fixpipeParams.dstStride;
        }
        Fixpipe<QK_T, float, QLIV2_CFG_ROW_MAJOR_UB>(mm1ResUB_[(runInfo.loop % 2) * constInfo_.s2BaseSize / 2],
        // 未考虑s1gGmOffset和s2GmOffset，将matmul结果从L0C搬运到UB
            cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET],
            fixpipeParams);
    } else {
        // nSize * sizeof(QT) <= 256B, 小于一个UB bank大小(VL)
        uint32_t nSize = (s2L0RealSize + 7) >> 3 << 3; // 8个元素（32B)对齐
        // 有效数据不足16行，只需输出部分行即可; L0C上bmm1结果矩阵M方向size必须是偶数
        uint32_t mSize = (s1gL0RealSize + 1) >> 1 << 1;
        // L0C上matmul结果相邻连续数据片断间隔, 单位为16 * sizeof(T)
        uint32_t srcStride = ((mSize + 15) / 16) * 16;
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C->UB
        fixpipeParams.nSize = nSize; // N方向全部输出
        fixpipeParams.mSize = mSize / 2; // M方向每个AIV一半
        fixpipeParams.srcStride = srcStride;
        fixpipeParams.dstStride = UB_BANK_DEPTH_STRIDE / sizeof(QK_T); // 落到同一个bank
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        fixpipeParams.dualDstCtl = 0;
        fixpipeParams.quantPre = F322BF16;
        fixpipeParams.reluEn = true; // ReLU激活
        fixpipeParams.subBlockId = 0;
        Fixpipe<QK_T, float, QLIV2_CFG_ROW_MAJOR_UB>(mm1ResUB_[(runInfo.loop % 2) * (UB_BANK_STRIDE / sizeof(QK_T))],
        // 未考虑s1gGmOffset和s2GmOffset，将matmul结果从L0C搬运到UB
            cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET],
            fixpipeParams);

        fixpipeParams.subBlockId = 1;
        Fixpipe<QK_T, float, QLIV2_CFG_ROW_MAJOR_UB>(mm1ResUB_[(runInfo.loop % 2) * (UB_BANK_STRIDE / sizeof(QK_T))],
            // 未考虑s1gGmOffset和s2GmOffset
                                                    cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET + mSize / 2 * 16],
                                                        fixpipeParams); // 将matmul结果从L0C搬运到UB
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::AllocEventID()
{
    SetMMLayoutTransform(true);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);

    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);

    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + 0);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::FreeEventID()
{
    SetMMLayoutTransform(false);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);

    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);

    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 0);
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);
}
} // namespace QLIV2Kernel
#endif  // QUANT_LIGHTNING_INDEXER_V2_SERVICE_CUBE_H