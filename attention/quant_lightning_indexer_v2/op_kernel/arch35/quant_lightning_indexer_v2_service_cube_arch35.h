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
 * \file quant_lightning_indexer_v2_service_cube_arch35.h
 * \brief use 5 buffer for matmul l1, better pipeline
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_SERVICE_CUBE_H
#define QUANT_LIGHTNING_INDEXER_V2_SERVICE_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_lightning_indexer_v2_common_arch35.h"
#if __has_include("../../lightning_indexer_v2/arch35/common/li_v2_cube_service_common.h")
#include "../../lightning_indexer_v2/arch35/common/li_v2_cube_service_common.h"
#else
#include "../../../lightning_indexer_v2/op_kernel/arch35/common/li_v2_cube_service_common.h"
#endif

namespace QLIV2Kernel {
using namespace QLIV2Common;
using namespace liCubeCommon;
template <typename QLIV2T>
class QLIV2Matmul {
public:
    using Q_T = typename QLIV2T::queryType;
    using K_T = typename QLIV2T::keyType;
    using QK_T = typename QLIV2T::queryKeyType;
    using SCALE_T = typename QLIV2T::scaleType;
    // MXFP4的GM/L1按uint8_t存packed字节，Load2DMX源操作数语义使用原始E2M1类型。
    using MX_DATA_SRC_T = std::conditional_t<QLIV2T::isMxFp4, typename QLIV2T::rawQueryType, Q_T>;
    // Load2DMX要求：fp8_e4m3fn_t源操作数的dst使用mx_fp8_e4m3_t；其他类型dst与src一致。
    using MX_DATA_DST_T = std::conditional_t<QLIV2T::isMxFp8, mx_fp8_e4m3_t, MX_DATA_SRC_T>;
    using L0_Q_T = std::conditional_t<QLIV2T::isMx, MX_DATA_DST_T, Q_T>;
    using L0_K_T = std::conditional_t<QLIV2T::isMx, MX_DATA_DST_T, K_T>;
    using CL0_T = std::conditional_t<std::is_same_v<QK_T, int32_t>, int32_t, float>;

    __aicore__ inline QLIV2Matmul(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<K_T> &keyGm,
                                               const GlobalTensor<Q_T> &queryGm,
                                               const GlobalTensor<bfloat16_t> &keyScaleGmBf16 = {},
                                               const GlobalTensor<bfloat16_t> &queryScaleGmBf16 = {});
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const QLIV2Common::RunInfo &runInfo);

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true}; // isSetFMatrix isSetPadding;
    static constexpr uint64_t KEY_BUF_NUM = 3;
    static constexpr uint64_t QUERY_BUF_NUM = 2;
    static constexpr uint64_t L0_BUF_NUM = 2;

    static constexpr uint32_t KEY_MTE1_MTE2_EVENT = EVENT_ID2;
    static constexpr uint32_t QUERY_MTE1_MTE2_EVENT = EVENT_ID5; // KEY_MTE1_MTE2_EVENT + KEY_BUF_NUM;
    static constexpr uint32_t M_MTE1_EVENT = EVENT_ID3;

    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;
    static constexpr uint32_t FIX_M_EVENT = EVENT_ID2;
    static constexpr uint32_t M_FIX_EVENT = EVENT_ID3;

    static constexpr uint64_t M_BASIC_BLOCK = 256;
    static constexpr uint64_t M_BASIC_BLOCK_SMALL = 128;
    static constexpr uint64_t D_BASIC_BLOCK = 128;

    static constexpr uint64_t M_BASIC_BLOCK_L0 = 256;
    static constexpr uint64_t D_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 128;

    static constexpr uint64_t FP8_BLOCK_CUBE = 32;
    static constexpr uint64_t BLOCK_CUBE = 16;

    static constexpr uint64_t QUERY_BUFFER_OFFSET = M_BASIC_BLOCK * D_BASIC_BLOCK;
    static constexpr uint64_t L0AB_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0;
    static constexpr uint64_t L0C_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0;
    // Key L0 ping-pong is sized by S2, not by the (potentially larger) M block.
    static constexpr uint64_t KEY_L0_BUFFER_OFFSET = S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0;
    // MX qScale L1乒乓缓冲区步长：M基本块 * 固定D基本块 / 32
    static constexpr uint64_t QUERY_SCALE_BUFFER_OFFSET = M_BASIC_BLOCK * D_BASIC_BLOCK / MX_SCALE_GROUP_SIZE;
    // MXFP4打包路径每次32字节搬入/scale对应64个逻辑D元素
    static constexpr uint64_t MX_LOAD_SCALE_ALIGN = MX_SCALE_GROUP_SIZE * FP4_PACK_NUM;

protected:
    __aicore__ inline void Fixp(uint64_t s1gGmOffset, uint64_t s2GmOffset, uint64_t s1gL0RealSize,
                                uint64_t s2L0RealSize, uint64_t s1gL1SizeAlign2G, const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void ComputeL0c(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                      const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void LoadKeyToL0b(uint64_t s2L0Offset, uint64_t s2L1RealSize, uint64_t s2L0RealSize,
                                        const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL1RealSize, uint64_t s1gL0RealSize,
                                          const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gL1Offset,
                                      const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void KeyNd2Nz(uint64_t s2L1RealSize, uint64_t s2GmOffset, const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                         const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void LoadQScaleToL1(uint64_t s1gL1RealSize, uint64_t s1gGmOffset,
                                          const QLIV2Common::RunInfo &runInfo);
    __aicore__ inline void LoadKScaleToL1(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                          const QLIV2Common::RunInfo &runInfo);
    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<Q_T> queryGm_;
    GlobalTensor<bfloat16_t> mxKeyScaleGmBf16_;
    GlobalTensor<bfloat16_t> mxQueryScaleGmBf16_;

    TBuf<TPosition::A1> bufQL1_;
    LocalTensor<Q_T> queryL1_;
    TBuf<TPosition::B1> bufKeyL1_;
    LocalTensor<K_T> keyL1_;
    TBuf<TPosition::A1> bufQScaleL1_;
    LocalTensor<SCALE_T> queryScaleL1_;
    TBuf<TPosition::B1> bufKeyScaleL1_;
    LocalTensor<SCALE_T> keyScaleL1_;

    TBuf<TPosition::A2> bufQL0_;
    LocalTensor<L0_Q_T> queryL0_;
    TBuf<TPosition::B2> bufKeyL0_;
    LocalTensor<L0_K_T> keyL0_;

    TBuf<TPosition::CO1> bufL0C_;
    LocalTensor<CL0_T> cL0_;

    TBuf<TPosition::VECCALC> bufUB_;
    LocalTensor<QK_T> mm1ResUB_;

    uint64_t keyL1BufIdx_ = 0;
    uint64_t queryL1Mte2BufIdx_ = 0;
    uint64_t queryL1Mte1BufIdx_ = 0;
    uint64_t l0BufIdx_ = 0;

    bool isKeyCacheValid_ = false; // L1中是否有可复用的数据
    uint64_t keyGmStart_ = 0;      // L1数据对应的GM S2偏移
    uint64_t keyLoadedSize_ = 0;   // L1中实际加载的S2元素数量
    uint64_t s2BasicBlock_ = 128;
    uint64_t qkHeadDim_ = 128;  // Q/K单行GM搬入宽度；MXFP4为打包后的headDim/2，其他场景为headDim
    uint64_t scaleHeadDim_ = 4; // MX scale单行元素数，即headDim/32，MXFP8/MXFP4共用
    uint64_t keyBufferOffset_ = 16384;    // Key L1乒乓缓冲区步长，s2BasicBlock_ * D_BASIC_BLOCK
    uint64_t keyScaleBufferOffset_ = 512; // Key scale L1乒乓缓冲区步长，s2BasicBlock_ * D_BASIC_BLOCK / 32

    ConstInfo constInfo_;
    uint64_t queryBufferOffset_ = 0;
    uint64_t l0abBufferOffset_ = 0;
    uint64_t l0cBufferOffset_ = 0;
    uint64_t queryScaleBufferOffset = 0;

private:
    static constexpr bool PAGE_ATTENTION = QLIV2T::pageAttention;
    static constexpr bool IS_MX = QLIV2T::isMx;
    static constexpr bool IS_MXFP4 = QLIV2T::isMxFp4;
};

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::InitParams(const ConstInfo &constInfo)
{
    constInfo_ = constInfo;
    s2BasicBlock_ = (constInfo_.maxSeqlenQ <= 4 && constInfo_.maxSeqlenQ >= 0) ? 256 : 128;
    qkHeadDim_ = constInfo_.headDim;
    if constexpr (IS_MXFP4) {
        qkHeadDim_ = constInfo_.headDim / FP4_PACK_NUM;
    }
    scaleHeadDim_ = constInfo_.headDim / MX_SCALE_GROUP_SIZE;
    keyBufferOffset_ = s2BasicBlock_ * D_BASIC_BLOCK;
    keyScaleBufferOffset_ = s2BasicBlock_ * D_BASIC_BLOCK / MX_SCALE_GROUP_SIZE;
    queryBufferOffset_ = constInfo_.mBaseSizeMax * D_BASIC_BLOCK;
    l0abBufferOffset_ = constInfo_.mBaseSizeMax * D_BASIC_BLOCK_L0;
    l0cBufferOffset_ = constInfo_.mBaseSizeMax * S2_BASIC_BLOCK_L0;
    queryScaleBufferOffset = constInfo_.mBaseSizeMax * D_BASIC_BLOCK / MX_SCALE_GROUP_SIZE;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(bufUB_, 2 * CeilDiv(constInfo_.mBaseSizeMax, 2) * constInfo_.s2BaseSize * sizeof(QK_T));
    // 大小：2(开dB) * 2 * 64 * 128 * 4 = 128KB
    mm1ResUB_ = bufUB_.Get<QK_T>();
    pipe->InitBuffer(bufQL1_, QUERY_BUF_NUM * constInfo_.mBaseSizeMax * D_BASIC_BLOCK * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, KEY_BUF_NUM * keyBufferOffset_ * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();
    if constexpr (IS_MX) {
        pipe->InitBuffer(bufQScaleL1_, QUERY_BUF_NUM * queryScaleBufferOffset * sizeof(SCALE_T));
        queryScaleL1_ = bufQScaleL1_.Get<SCALE_T>();
        pipe->InitBuffer(bufKeyScaleL1_, KEY_BUF_NUM * keyScaleBufferOffset_ * sizeof(SCALE_T));
        keyScaleL1_ = bufKeyScaleL1_.Get<SCALE_T>();
    }

    pipe->InitBuffer(bufQL0_, L0_BUF_NUM * constInfo_.mBaseSizeMax * D_BASIC_BLOCK_L0 * sizeof(L0_Q_T));
    queryL0_ = bufQL0_.Get<L0_Q_T>();
    pipe->InitBuffer(bufKeyL0_, L0_BUF_NUM * D_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(L0_K_T));
    keyL0_ = bufKeyL0_.Get<L0_K_T>();

    pipe->InitBuffer(bufL0C_, L0_BUF_NUM * constInfo_.mBaseSizeMax * S2_BASIC_BLOCK_L0 * sizeof(float));
    cL0_ = bufL0C_.Get<CL0_T>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm,
                                                                const GlobalTensor<K_T> &keyGm,
                                                                const GlobalTensor<Q_T> &queryGm,
                                                                const GlobalTensor<bfloat16_t> &keyScaleGmBf16,
                                                                const GlobalTensor<bfloat16_t> &queryScaleGmBf16)
{
    blkTableGm_ = blkTableGm;
    keyGm_ = keyGm;
    queryGm_ = queryGm;
    if constexpr (IS_MX) {
        mxKeyScaleGmBf16_ = keyScaleGmBf16;
        mxQueryScaleGmBf16_ = queryScaleGmBf16;
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::ComputeMm1(const QLIV2Common::RunInfo &runInfo)
{
    CrossCoreWaitFlag<QLIV2Common::ConstInfo::QLIV2_SYNC_MODE4, PIPE_FIX>(QLIV2Common::ConstInfo::CROSS_VC_EVENT +
                                                                          runInfo.loop % 2);
    CrossCoreWaitFlag<QLIV2Common::ConstInfo::QLIV2_SYNC_MODE4, PIPE_FIX>(
        QLIV2Common::ConstInfo::CROSS_VC_EVENT + runInfo.loop % 2 + QLIV2Common::ConstInfo::AIV0_AIV1_OFFSET);
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
                if constexpr (IS_MX) {
                    // MX: PA路径需要绝对偏移
                    LoadKScaleToL1(s2L1RealSize, s2GmBaseOffset + s2GmOffset, runInfo);
                }
            } else {
                KeyNd2Nz(s2L1RealSize, s2GmOffset, runInfo);
                if constexpr (IS_MX) {
                    // MX: 非PA路径只传循环内相对偏移（tensorKeyScaleOffset 已含 s2GmBaseOffset）
                    LoadKScaleToL1(s2L1RealSize, s2GmOffset, runInfo);
                }
            }

            SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            // s1gProcessSize当前必定不会超过2倍的s1g basic block
            for (uint64_t s1gGmOffset = 0; s1gGmOffset < s1gProcessSize; s1gGmOffset += constInfo_.mBaseSizeMax) {
                uint64_t s1gL1RealSize = s1gGmOffset + constInfo_.mBaseSizeMax > s1gProcessSize ?
                                             s1gProcessSize - s1gGmOffset :
                                             constInfo_.mBaseSizeMax;
                uint64_t s1gL1SizeAlign2G = CeilAlign(s1gL1RealSize, 2 * constInfo_.gSize);
                uint64_t s1gL1SizeAlign = CeilAlign(s1gL1SizeAlign2G, BLOCK_CUBE);
                if (runInfo.isFirstS2InnerLoop && s2GmOffset == 0) {
                    queryL1Mte2BufIdx_++;
                    queryL1Mte1BufIdx_ = queryL1Mte2BufIdx_;
                    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte2BufIdx_ % QUERY_BUF_NUM);
                    QueryNd2Nz(s1gL1RealSize, s1gGmOffset, runInfo);
                    if constexpr (IS_MX) {
                        LoadQScaleToL1(s1gL1RealSize, s1gGmOffset, runInfo);
                    }
                    SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                    WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                } else {
                    queryL1Mte1BufIdx_ =
                        queryL1Mte2BufIdx_ - (CeilDiv(s1gProcessSize, constInfo_.mBaseSizeMax) - 1 - (s1gGmOffset > 0));
                }
                for (uint64_t s2L1Offset = 0; s2L1Offset < s2L1RealSize; s2L1Offset += S2_BASIC_BLOCK_L0) {
                    uint64_t s2L0RealSize =
                        s2L1Offset + S2_BASIC_BLOCK_L0 > s2L1RealSize ? s2L1RealSize - s2L1Offset : S2_BASIC_BLOCK_L0;
                    for (uint64_t s1gL1Offset = 0; s1gL1Offset < s1gL1SizeAlign;
                         s1gL1Offset += constInfo_.mBaseSizeMax) {
                        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        uint64_t s1gL0RealSize = s1gL1Offset + constInfo_.mBaseSizeMax > s1gL1SizeAlign ?
                                                     s1gL1SizeAlign - s1gL1Offset :
                                                     constInfo_.mBaseSizeMax;
                        LoadQueryToL0a(s1gL1Offset, s1gL1SizeAlign, s1gL0RealSize, runInfo);
                        LoadKeyToL0b(s2L1Offset, s2L1RealSize, s2L0RealSize, runInfo);

                        SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
                        WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);

                        WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        ComputeL0c(s1gL0RealSize, s2L0RealSize, runInfo);

                        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);

                        Fixp(s1gGmOffset + s1gL1Offset, s2GmOffset + s2L1Offset, s1gL0RealSize, s2L0RealSize,
                             s1gL1SizeAlign2G, runInfo);
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
                    if constexpr (IS_MX) {
                        LoadKScaleToL1(s2L1LoadSize, s2GmBaseOffset + s2GmOffset, runInfo);
                    }
                } else {
                    KeyNd2Nz(s2L1LoadSize, s2GmOffset, runInfo);
                    if constexpr (IS_MX) {
                        LoadKScaleToL1(s2L1LoadSize, s2GmOffset, runInfo);
                    }
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
            for (uint64_t s1gGmOffset = 0; s1gGmOffset < s1gProcessSize; s1gGmOffset += constInfo_.mBaseSizeMax) {
                uint64_t s1gL1RealSize = s1gGmOffset + constInfo_.mBaseSizeMax > s1gProcessSize ?
                                             s1gProcessSize - s1gGmOffset :
                                             constInfo_.mBaseSizeMax;
                uint64_t s1gL1SizeAlign2G = CeilAlign(s1gL1RealSize, 2 * constInfo_.gSize);
                uint64_t s1gL1SizeAlign = CeilAlign(s1gL1SizeAlign2G, BLOCK_CUBE);
                if (runInfo.isFirstS2InnerLoop && s2GmOffset == 0) {
                    queryL1Mte2BufIdx_++;
                    queryL1Mte1BufIdx_ = queryL1Mte2BufIdx_;
                    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte2BufIdx_ % QUERY_BUF_NUM);
                    QueryNd2Nz(s1gL1RealSize, s1gGmOffset, runInfo);
                    if constexpr (IS_MX) {
                        LoadQScaleToL1(s1gL1RealSize, s1gGmOffset, runInfo);
                    }
                    SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                    WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                } else {
                    queryL1Mte1BufIdx_ =
                        queryL1Mte2BufIdx_ - (CeilDiv(s1gProcessSize, constInfo_.mBaseSizeMax) - 1 - (s1gGmOffset > 0));
                }
                uint64_t s2Boundry = l1S2Offset + s2ProcessSize;
                for (uint64_t s2L1Offset = l1S2Offset; s2L1Offset < s2Boundry; s2L1Offset += S2_BASIC_BLOCK_L0) {
                    uint64_t s2L0RealSize = s2L1Offset + S2_BASIC_BLOCK_L0 > l1S2Offset + s2ProcessSize ?
                                                l1S2Offset + s2ProcessSize - s2L1Offset :
                                                S2_BASIC_BLOCK_L0;
                    for (uint64_t s1gOffset = 0; s1gOffset < s1gL1SizeAlign; s1gOffset += constInfo_.mBaseSizeMax) {
                        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        uint64_t s1gL0RealSize = s1gOffset + constInfo_.mBaseSizeMax > s1gL1SizeAlign ?
                                                     s1gL1SizeAlign - s1gOffset :
                                                     constInfo_.mBaseSizeMax;
                        LoadQueryToL0a(s1gOffset, s1gL1SizeAlign, s1gL0RealSize, runInfo);
                        LoadKeyToL0b(s2L1Offset, l1TotalSize, s2L0RealSize, runInfo);

                        SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
                        WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);

                        WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
                        ComputeL0c(s1gL0RealSize, s2L0RealSize, runInfo);

                        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);

                        Fixp(s1gGmOffset + s1gOffset, (s2L1Offset - l1S2Offset), s1gL0RealSize, s2L0RealSize,
                             s1gL1SizeAlign2G, runInfo);
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
                isKeyCacheValid_ = false;
            }
        }
    }

    CrossCoreSetFlag<QLIV2Common::ConstInfo::QLIV2_SYNC_MODE4, PIPE_FIX>(QLIV2Common::ConstInfo::CROSS_CV_EVENT +
                                                                         runInfo.loop % 2);
    CrossCoreSetFlag<QLIV2Common::ConstInfo::QLIV2_SYNC_MODE4, PIPE_FIX>(
        QLIV2Common::ConstInfo::CROSS_CV_EVENT + runInfo.loop % 2 + QLIV2Common::ConstInfo::AIV0_AIV1_OFFSET);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::KeyNd2Nz(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                     const QLIV2Common::RunInfo &runInfo)
{
    liCubeCommon::KeyNd2Nz(keyL1_, keyGm_, keyL1BufIdx_, KEY_BUF_NUM, keyBufferOffset_, qkHeadDim_, s2L1RealSize,
                           s2GmOffset, runInfo.tensorKeyOffset);
}

// blkNum, blkSize, N2, D
template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                          const QLIV2Common::RunInfo &runInfo)
{
    liCubeCommon::KeyNd2NzForPA(keyL1_, keyGm_, blkTableGm_, keyL1BufIdx_, KEY_BUF_NUM, keyBufferOffset_, qkHeadDim_,
                                FP8_BLOCK_CUBE, s2L1RealSize, s2GmOffset, constInfo_.kCacheBlockSize, runInfo.bIdx,
                                constInfo_.maxBlockNumPerBatch, constInfo_.keyStride0);
}

// batch, s1, n2, g, d
template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gGmOffset,
                                                       const QLIV2Common::RunInfo &runInfo)
{
    liCubeCommon::QueryNd2Nz(queryL1_, queryGm_, queryL1Mte2BufIdx_, QUERY_BUF_NUM, queryBufferOffset_, qkHeadDim_,
                             constInfo_.gSize, s1gL1RealSize, s1gGmOffset, runInfo.tensorQueryOffset);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::LoadQScaleToL1(uint64_t s1gL1RealSize, uint64_t s1gGmOffset,
                                                           const QLIV2Common::RunInfo &runInfo)
{
    uint64_t scaleOffsetInBuf = (queryL1Mte2BufIdx_ % QUERY_BUF_NUM) * queryScaleBufferOffset;
    LocalTensor<bfloat16_t> scaleL1 = queryScaleL1_[scaleOffsetInBuf].template ReinterpretCast<bfloat16_t>();
    uint32_t scalePerToken = scaleHeadDim_;
    Dn2NzParams dn2Nzparam;
    dn2Nzparam.dnNum = 1;
    dn2Nzparam.nValue = scalePerToken / FP8_TWO;
    dn2Nzparam.dValue = s1gL1RealSize;
    dn2Nzparam.srcDnMatrixStride = 0;
    dn2Nzparam.srcDValue = scalePerToken / FP8_TWO;
    dn2Nzparam.dstNzC0Stride = scalePerToken / FP8_TWO;
    dn2Nzparam.dstNzNStride = 1;
    dn2Nzparam.dstNzMatrixStride = 0;
    uint64_t gmOffset = runInfo.tensorQScaleOffset + s1gGmOffset * scalePerToken;
    DataCopy(scaleL1, mxQueryScaleGmBf16_[gmOffset / FP8_TWO], dn2Nzparam);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::LoadKScaleToL1(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                           const QLIV2Common::RunInfo &runInfo)
{
    uint64_t scaleOffsetInBuf = (keyL1BufIdx_ % KEY_BUF_NUM) * keyScaleBufferOffset_;
    LocalTensor<bfloat16_t> scaleL1 = keyScaleL1_[scaleOffsetInBuf].template ReinterpretCast<bfloat16_t>();
    uint32_t scalePerToken = scaleHeadDim_;

    if constexpr (PAGE_ATTENTION) {
        uint64_t s2L1Offset = 0;
        while (s2L1Offset < s2L1RealSize) {
            uint64_t s2BlkId = (s2L1Offset + s2GmOffset) / constInfo_.kCacheBlockSize;
            uint64_t s2BlkOffset = (s2L1Offset + s2GmOffset) % constInfo_.kCacheBlockSize;
            uint64_t physicalBlkId = blkTableGm_.GetValue(runInfo.bIdx * constInfo_.maxBlockNumPerBatch + s2BlkId);
            // kScale GM偏移：physicalBlkId * stride0 + s2BlkOffset * scalePerToken
            uint64_t gmOffset = physicalBlkId * constInfo_.keyDequantScaleStride0 + s2BlkOffset * scalePerToken;

            uint64_t s2Mte2Size = s2L1RealSize - s2L1Offset;
            s2Mte2Size = s2BlkOffset + s2Mte2Size >= constInfo_.kCacheBlockSize ?
                             constInfo_.kCacheBlockSize - s2BlkOffset :
                             s2Mte2Size;

            Dn2NzParams dn2Nzparam;
            dn2Nzparam.dnNum = 1;
            dn2Nzparam.nValue = scalePerToken / FP8_TWO;
            dn2Nzparam.dValue = s2Mte2Size;
            dn2Nzparam.srcDnMatrixStride = 0;
            dn2Nzparam.srcDValue = scalePerToken / FP8_TWO;
            dn2Nzparam.dstNzC0Stride = scalePerToken / FP8_TWO;
            dn2Nzparam.dstNzNStride = 1;
            dn2Nzparam.dstNzMatrixStride = 0;
            DataCopy(scaleL1[s2L1Offset * scalePerToken / FP8_TWO], mxKeyScaleGmBf16_[gmOffset / FP8_TWO], dn2Nzparam);

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
        dn2Nzparam.dstNzMatrixStride = 0;
        uint64_t gmOffset = runInfo.tensorKeyScaleOffset + s2GmOffset * scalePerToken;
        DataCopy(scaleL1, mxKeyScaleGmBf16_[gmOffset / FP8_TWO], dn2Nzparam);
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL1RealSize,
                                                           uint64_t s1gL0RealSize, const QLIV2Common::RunInfo &runInfo)
{
    if constexpr (IS_MX) {
        // MX: 使用LoadData Mx变体，同时加载数据和scale到L0
        LoadData2DParamsV2 loadData2DParamsV2;
        loadData2DParamsV2.mStartPosition = CeilDiv(s1gL1Offset, BLOCK_CUBE);
        loadData2DParamsV2.kStartPosition = 0;
        loadData2DParamsV2.mStep = CeilDiv(s1gL0RealSize, BLOCK_CUBE);
        loadData2DParamsV2.kStep =
            IS_MXFP4 ? constInfo_.headDim / MX_LOAD_SCALE_ALIGN : CeilDiv(constInfo_.headDim, FP8_BLOCK_CUBE);
        loadData2DParamsV2.srcStride = CeilDiv(s1gL1RealSize, BLOCK_CUBE);
        loadData2DParamsV2.dstStride = CeilDiv(s1gL0RealSize, BLOCK_CUBE);
        loadData2DParamsV2.ifTranspose = false;

        LoadData2DMxParams loadDataMxParams;
        loadDataMxParams.xStartPosition = CeilDiv(s1gL1Offset, BLOCK_CUBE);
        loadDataMxParams.yStartPosition = 0;
        loadDataMxParams.xStep = CeilDiv(s1gL0RealSize, BLOCK_CUBE);
        loadDataMxParams.yStep = scaleHeadDim_ / FP8_TWO;
        loadDataMxParams.srcStride = loadDataMxParams.yStep;
        loadDataMxParams.dstStride = loadDataMxParams.yStep;

        uint64_t queryDataOffsetInBuf = (queryL1Mte1BufIdx_ % QUERY_BUF_NUM) * queryBufferOffset_;
        uint64_t queryScaleOffsetInBuf = (queryL1Mte1BufIdx_ % QUERY_BUF_NUM) * queryScaleBufferOffset;
        LocalTensor<L0_Q_T> queryL0Tensor = queryL0_[(l0BufIdx_ % L0_BUF_NUM) * l0abBufferOffset_];
        LocalTensor<fp8_e8m0_t> queryScaleL1Tensor =
            queryScaleL1_[queryScaleOffsetInBuf].template ReinterpretCast<fp8_e8m0_t>();
        if constexpr (IS_MXFP4) {
            // MXFP4: Q data本身按E2M1参与计算，而queryL1_是uint8_t类型
            LocalTensor<MX_DATA_SRC_T> queryL1MxTensor =
                queryL1_[queryDataOffsetInBuf].template ReinterpretCast<MX_DATA_SRC_T>();
            LoadData(queryL0Tensor, queryL1MxTensor, queryScaleL1Tensor, loadData2DParamsV2, loadDataMxParams);
        } else {
            LocalTensor<MX_DATA_SRC_T> queryL1MxTensor = queryL1_[queryDataOffsetInBuf];
            LoadData(queryL0Tensor, queryL1MxTensor, queryScaleL1Tensor, loadData2DParamsV2, loadDataMxParams);
        }
    } else {
        liCubeCommon::LoadData2DPartToL0(queryL0_, queryL1_, l0BufIdx_, L0_BUF_NUM, l0abBufferOffset_,
                                         queryL1Mte1BufIdx_, QUERY_BUF_NUM, queryBufferOffset_, s1gL1Offset,
                                         s1gL1RealSize, s1gL0RealSize, CeilDiv(constInfo_.headDim, FP8_BLOCK_CUBE));
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::LoadKeyToL0b(uint64_t s2L1Offset, uint64_t s2L1RealSize,
                                                         uint64_t s2L0RealSize, const QLIV2Common::RunInfo &runInfo)
{
    if constexpr (IS_MX) {
        // MX: 使用LoadData Mx变体，同时加载数据和scale到L0
        LoadData2DParamsV2 loadData2DParamsV2;
        loadData2DParamsV2.mStartPosition = CeilDiv(s2L1Offset, BLOCK_CUBE);
        loadData2DParamsV2.kStartPosition = 0;
        loadData2DParamsV2.mStep = CeilDiv(s2L0RealSize, BLOCK_CUBE);
        loadData2DParamsV2.kStep =
            IS_MXFP4 ? constInfo_.headDim / MX_LOAD_SCALE_ALIGN : CeilDiv(constInfo_.headDim, FP8_BLOCK_CUBE);
        loadData2DParamsV2.srcStride = CeilDiv(s2L1RealSize, BLOCK_CUBE);
        loadData2DParamsV2.dstStride = CeilDiv(s2L0RealSize, BLOCK_CUBE);
        loadData2DParamsV2.ifTranspose = false;

        LoadData2DMxParams loadDataMxParams;
        loadDataMxParams.xStartPosition = CeilDiv(s2L1Offset, BLOCK_CUBE);
        loadDataMxParams.yStartPosition = 0;
        loadDataMxParams.xStep = CeilDiv(s2L0RealSize, BLOCK_CUBE);
        loadDataMxParams.yStep = scaleHeadDim_ / FP8_TWO;
        loadDataMxParams.srcStride = loadDataMxParams.yStep;
        loadDataMxParams.dstStride = loadDataMxParams.yStep;

        uint64_t keyDataOffsetInBuf = (keyL1BufIdx_ % KEY_BUF_NUM) * keyBufferOffset_;
        uint64_t keyScaleOffsetInBuf = (keyL1BufIdx_ % KEY_BUF_NUM) * keyScaleBufferOffset_;
        LocalTensor<L0_K_T> keyL0Tensor = keyL0_[(l0BufIdx_ % L0_BUF_NUM) * KEY_L0_BUFFER_OFFSET];
        LocalTensor<fp8_e8m0_t> keyScaleL1Tensor =
            keyScaleL1_[keyScaleOffsetInBuf].template ReinterpretCast<fp8_e8m0_t>();
        if constexpr (IS_MXFP4) {
            LocalTensor<MX_DATA_SRC_T> keyL1MxTensor =
                keyL1_[keyDataOffsetInBuf].template ReinterpretCast<MX_DATA_SRC_T>();
            LoadData(keyL0Tensor, keyL1MxTensor, keyScaleL1Tensor, loadData2DParamsV2, loadDataMxParams);
        } else {
            LocalTensor<MX_DATA_SRC_T> keyL1MxTensor = keyL1_[keyDataOffsetInBuf];
            LoadData(keyL0Tensor, keyL1MxTensor, keyScaleL1Tensor, loadData2DParamsV2, loadDataMxParams);
        }
    } else {
        liCubeCommon::LoadData2DPartToL0(keyL0_, keyL1_, l0BufIdx_, L0_BUF_NUM, KEY_L0_BUFFER_OFFSET, keyL1BufIdx_,
                                         KEY_BUF_NUM, keyBufferOffset_, s2L1Offset, s2L1RealSize, s2L0RealSize,
                                         CeilDiv(constInfo_.headDim, FP8_BLOCK_CUBE));
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::ComputeL0c(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                                       const QLIV2Common::RunInfo &runInfo)
{
    liCubeCommon::ComputeL0c(cL0_, queryL0_, keyL0_, l0BufIdx_, L0_BUF_NUM, l0cBufferOffset_, l0abBufferOffset_,
                             KEY_L0_BUFFER_OFFSET, s1gL0RealSize, s2L0RealSize, constInfo_.headDim);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::Fixp(uint64_t s1gGmOffset, uint64_t s2GmOffset, uint64_t s1gL0RealSize,
                                                 uint64_t s2L0RealSize, uint64_t s1gSizeAlign2G,
                                                 const QLIV2Common::RunInfo &runInfo)
{
    liCubeCommon::FixpImpl(mm1ResUB_, cL0_, l0BufIdx_, L0_BUF_NUM, l0cBufferOffset_, M_FIX_EVENT, s1gL0RealSize,
                           s1gSizeAlign2G, s2L0RealSize, constInfo_.mBaseSizeMax, constInfo_.s2BaseSize,
                           S2_BASIC_BLOCK_L0, runInfo.loop);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::AllocEventID()
{
    liCubeCommon::AllocCubeEventID();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::FreeEventID()
{
    liCubeCommon::FreeCubeEventID();
}
} // namespace QLIV2Kernel
#endif // QUANT_LIGHTNING_INDEXER_V2_SERVICE_CUBE_H
