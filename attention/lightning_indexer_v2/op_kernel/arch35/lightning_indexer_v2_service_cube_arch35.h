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
 * \file lightning_indexer_v2_service_cube_arch35.h
 * \brief use 5 buffer for matmul l1, better pipeline
 */
#ifndef LIGHTNING_INDEXER_V2_SERVICE_CUBE_ARCH35_H
#define LIGHTNING_INDEXER_V2_SERVICE_CUBE_ARCH35_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_v2_common_arch35.h"
#include "common/li_v2_cube_service_common.h"

namespace LIV2Kernel {
using namespace LIV2Common;
using namespace liCubeCommon;
template <typename LIT>
class LightningIndexerV2ServiceCube {
public:
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using QK_T = typename LIT::queryKeyType;
    using SCORE_T = typename LIT::scoreType;

    __aicore__ inline LightningIndexerV2ServiceCube(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<K_T> &keyGm,
                                               const GlobalTensor<Q_T> &queryGm);
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const LIV2Common::RunInfo &runInfo);

    static constexpr uint64_t KEY_BUF_NUM = 3;
    static constexpr uint64_t QUERY_BUF_NUM = 2;
    static constexpr uint64_t L0_BUF_NUM = 2;

    static constexpr uint32_t KEY_MTE1_MTE2_EVENT = EVENT_ID2;
    // KEY_MTE1_MTE2_EVENT + KEY_BUF_NUM;
    static constexpr uint32_t QUERY_MTE1_MTE2_EVENT = EVENT_ID5;
    static constexpr uint32_t M_MTE1_EVENT = EVENT_ID3;

    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;
    static constexpr uint32_t FIX_M_EVENT = EVENT_ID2;
    static constexpr uint32_t M_FIX_EVENT = EVENT_ID3;

    static constexpr uint64_t M_BASIC_BLOCK = 128;
    static constexpr uint64_t D_BASIC_BLOCK = 128;
    static constexpr uint64_t S2_BASIC_BLOCK = 128;
    static constexpr uint64_t D_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t BLOCK_CUBE = 16;
    static constexpr uint64_t FP16_BLOCK_CUBE = 16;

    static constexpr uint64_t KEY_BUFFER_OFFSET = S2_BASIC_BLOCK * D_BASIC_BLOCK;

protected:
    __aicore__ inline void Fixp(uint64_t s1gGmOffset, uint64_t s2GmOffset, uint64_t s1gL0RealSize,
                                uint64_t s2L0RealSize, uint64_t s1gSizeAlign2G, const LIV2Common::RunInfo &runInfo);
    __aicore__ inline void ComputeL0c(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                      const LIV2Common::RunInfo &runInfo);
    __aicore__ inline void LoadKeyToL0b(uint64_t s2L0Offset, uint64_t s2L1RealSize, uint64_t s2L0RealSize,
                                        const LIV2Common::RunInfo &runInfo);
    __aicore__ inline void LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL0Offset, uint64_t s1gL1RealSize,
                                          uint64_t s1gL0RealSize, const LIV2Common::RunInfo &runInfo);
    __aicore__ inline void QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gL1Offset, const LIV2Common::RunInfo &runInfo);
    __aicore__ inline void KeyNd2Nz(uint64_t s2L1RealSize, uint64_t s2GmOffset, const LIV2Common::RunInfo &runInfo);
    __aicore__ inline void KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                         const LIV2Common::RunInfo &runInfo);
    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<Q_T> queryGm_;

    TBuf<TPosition::A1> bufQL1_;
    LocalTensor<Q_T> queryL1_;
    TBuf<TPosition::B1> bufKeyL1_;
    LocalTensor<K_T> keyL1_;

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

    ConstInfo constInfo_;

    uint64_t queryBufferOffset_ = 0;
    uint64_t l0abBufferOffset_ = 0;
    uint64_t l0cBufferOffset_ = 0;

private:
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
};

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::InitParams(const ConstInfo &constInfo)
{
    constInfo_ = constInfo;
    queryBufferOffset_ = M_BASIC_BLOCK * D_BASIC_BLOCK;
    l0abBufferOffset_ = M_BASIC_BLOCK * D_BASIC_BLOCK_L0;
    l0cBufferOffset_ = M_BASIC_BLOCK * S2_BASIC_BLOCK_L0;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::InitBuffers(TPipe *pipe)
{
    // 大小：2(开dB) * 2 * 64 * 128 * 4 = 128KB
    pipe->InitBuffer(bufUB_, 2 * CeilDiv(M_BASIC_BLOCK, 2) * constInfo_.s2BaseSize * sizeof(QK_T));
    mm1ResUB_ = bufUB_.Get<QK_T>();
    pipe->InitBuffer(bufQL1_, QUERY_BUF_NUM * M_BASIC_BLOCK * D_BASIC_BLOCK * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, KEY_BUF_NUM * S2_BASIC_BLOCK * D_BASIC_BLOCK * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();

    pipe->InitBuffer(bufQL0_, L0_BUF_NUM * M_BASIC_BLOCK * D_BASIC_BLOCK_L0 * sizeof(Q_T));
    queryL0_ = bufQL0_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL0_, L0_BUF_NUM * D_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(K_T));
    keyL0_ = bufKeyL0_.Get<K_T>();

    pipe->InitBuffer(bufL0C_, L0_BUF_NUM * M_BASIC_BLOCK * S2_BASIC_BLOCK_L0 * sizeof(float));
    cL0_ = bufL0C_.Get<float>();
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm,
                                                                               const GlobalTensor<K_T> &keyGm,
                                                                               const GlobalTensor<Q_T> &queryGm)
{
    blkTableGm_ = blkTableGm;
    keyGm_ = keyGm;
    queryGm_ = queryGm;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::ComputeMm1(const LIV2Common::RunInfo &runInfo)
{
    CrossCoreWaitFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(LIV2Common::ConstInfo::CROSS_VC_EVENT +
                                                                      runInfo.loop % 2);
    CrossCoreWaitFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(
        LIV2Common::ConstInfo::CROSS_VC_EVENT + runInfo.loop % 2 + LIV2Common::ConstInfo::AIV0_AIV1_OFFSET);
    uint64_t s2GmBaseOffset = runInfo.s2Idx * constInfo_.s2BaseSize;
    uint64_t s1gProcessSize = runInfo.actMBaseSize;
    uint64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    for (uint64_t s2GmOffset = 0; s2GmOffset < s2ProcessSize; s2GmOffset += S2_BASIC_BLOCK) {
        WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        uint64_t s2L1RealSize =
            s2GmOffset + S2_BASIC_BLOCK > s2ProcessSize ? s2ProcessSize - s2GmOffset : S2_BASIC_BLOCK;
        if (PAGE_ATTENTION) {
            KeyNd2NzForPA(s2L1RealSize, s2GmBaseOffset + s2GmOffset, runInfo);
        } else {
            KeyNd2Nz(s2L1RealSize, s2GmOffset, runInfo);
        }

        SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        // s1gProcessSize当前必定不会超过2倍的s1g basic block
        for (uint64_t s1gGmOffset = 0; s1gGmOffset < s1gProcessSize; s1gGmOffset += M_BASIC_BLOCK) {
            uint64_t s1gL1RealSize =
                s1gGmOffset + M_BASIC_BLOCK > s1gProcessSize ? s1gProcessSize - s1gGmOffset : M_BASIC_BLOCK;
            uint64_t s1gL1SizeAlign2G = CeilAlign(s1gL1RealSize, 2 * constInfo_.gSize);
            uint64_t s1gL1SizeAlign = CeilAlign(s1gL1SizeAlign2G, BLOCK_CUBE);
            if (runInfo.isFirstS2InnerLoop && s2GmOffset == 0) {
                queryL1Mte2BufIdx_++;
                queryL1Mte1BufIdx_ = queryL1Mte2BufIdx_;
                WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte2BufIdx_ % QUERY_BUF_NUM);
                QueryNd2Nz(s1gL1RealSize, s1gGmOffset, runInfo);
                SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            } else {
                queryL1Mte1BufIdx_ =
                    queryL1Mte2BufIdx_ - (CeilDiv(s1gProcessSize, M_BASIC_BLOCK) - 1 - (s1gGmOffset > 0));
            }
            for (uint64_t s2L1Offset = 0; s2L1Offset < s2L1RealSize; s2L1Offset += S2_BASIC_BLOCK_L0) {
                uint64_t s2L0RealSize =
                    s2L1Offset + S2_BASIC_BLOCK_L0 > s2L1RealSize ? s2L1RealSize - s2L1Offset : S2_BASIC_BLOCK_L0;
                for (uint64_t s1gL1Offset = 0; s1gL1Offset < s1gL1SizeAlign; s1gL1Offset += M_BASIC_BLOCK) {
                    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
                    uint64_t s1gL0RealSize =
                        s1gL1Offset + M_BASIC_BLOCK > s1gL1SizeAlign ? s1gL1SizeAlign - s1gL1Offset : M_BASIC_BLOCK;
                    LoadQueryToL0a(s1gGmOffset, s1gL1Offset, s1gL1SizeAlign, s1gL0RealSize, runInfo);
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
            if (s2GmOffset + S2_BASIC_BLOCK >= s2ProcessSize && runInfo.isLastS2InnerLoop) {
                SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte1BufIdx_ % QUERY_BUF_NUM);
            }
        }
        SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        keyL1BufIdx_++;
    }
    CrossCoreSetFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(LIV2Common::ConstInfo::CROSS_CV_EVENT +
                                                                     runInfo.loop % 2);
    CrossCoreSetFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(
        LIV2Common::ConstInfo::CROSS_CV_EVENT + runInfo.loop % 2 + LIV2Common::ConstInfo::AIV0_AIV1_OFFSET);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::KeyNd2Nz(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                                    const LIV2Common::RunInfo &runInfo)
{
    liCubeCommon::KeyNd2Nz(keyL1_, keyGm_, keyL1BufIdx_, KEY_BUF_NUM, KEY_BUFFER_OFFSET, constInfo_.headDim,
                           s2L1RealSize, s2GmOffset, runInfo.tensorKeyOffset);
}

// blkNum, blkSize, N2, D
template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                                         const LIV2Common::RunInfo &runInfo)
{
    liCubeCommon::KeyNd2NzForPA(keyL1_, keyGm_, blkTableGm_, keyL1BufIdx_, KEY_BUF_NUM, KEY_BUFFER_OFFSET,
                                constInfo_.headDim, FP16_BLOCK_CUBE, s2L1RealSize, s2GmOffset,
                                constInfo_.kCacheBlockSize, runInfo.bIdx, constInfo_.maxBlockNumPerBatch,
                                constInfo_.keyStride0);
}

// batch, s1, n2, g, d
template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gGmOffset,
                                                                      const LIV2Common::RunInfo &runInfo)
{
    liCubeCommon::QueryNd2Nz(queryL1_, queryGm_, queryL1Mte2BufIdx_, QUERY_BUF_NUM, queryBufferOffset_,
                             constInfo_.headDim, constInfo_.gSize, s1gL1RealSize, s1gGmOffset,
                             runInfo.tensorQueryOffset);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::LoadQueryToL0a(uint64_t s1gGmOffset, uint64_t s1gL1Offset,
                                                                          uint64_t s1gL1RealSize,
                                                                          uint64_t s1gL0RealSize,
                                                                          const LIV2Common::RunInfo &runInfo)
{
    liCubeCommon::LoadData2DPartToL0(queryL0_, queryL1_, l0BufIdx_, L0_BUF_NUM, l0abBufferOffset_, queryL1Mte1BufIdx_,
                                     QUERY_BUF_NUM, queryBufferOffset_, s1gL1Offset, s1gL1RealSize, s1gL0RealSize,
                                     CeilDiv(constInfo_.headDim, FP16_BLOCK_CUBE));
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::LoadKeyToL0b(uint64_t s2L1Offset, uint64_t s2L1RealSize,
                                                                        uint64_t s2L0RealSize,
                                                                        const LIV2Common::RunInfo &runInfo)
{
    liCubeCommon::LoadData2DPartToL0(keyL0_, keyL1_, l0BufIdx_, L0_BUF_NUM, l0abBufferOffset_, keyL1BufIdx_,
                                     KEY_BUF_NUM, KEY_BUFFER_OFFSET, s2L1Offset, s2L1RealSize, s2L0RealSize,
                                     CeilDiv(constInfo_.headDim, FP16_BLOCK_CUBE));
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::ComputeL0c(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                                                      const LIV2Common::RunInfo &runInfo)
{
    liCubeCommon::ComputeL0c(cL0_, queryL0_, keyL0_, l0BufIdx_, L0_BUF_NUM, l0cBufferOffset_, l0abBufferOffset_,
                             l0abBufferOffset_, s1gL0RealSize, s2L0RealSize, constInfo_.headDim);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::Fixp(uint64_t s1gGmOffset, uint64_t s2GmOffset,
                                                                uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                                                uint64_t s1gSizeAlign2G,
                                                                const LIV2Common::RunInfo &runInfo)
{
    liCubeCommon::FixpImpl(mm1ResUB_, cL0_, l0BufIdx_, L0_BUF_NUM, l0cBufferOffset_, M_FIX_EVENT, s1gL0RealSize,
                           s1gSizeAlign2G, s2L0RealSize, M_BASIC_BLOCK, constInfo_.s2BaseSize, S2_BASIC_BLOCK_L0,
                           runInfo.loop);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::AllocEventID()
{
    liCubeCommon::AllocCubeEventID();
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCube<LIT>::FreeEventID()
{
    liCubeCommon::FreeCubeEventID();
}
} // namespace LIV2Kernel
#endif
