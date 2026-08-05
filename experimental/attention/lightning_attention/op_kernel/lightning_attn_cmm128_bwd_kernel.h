/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "decay/cmm_decay_generator.h"
#include "decay/decay_generator.h"
#include "common/lo_cmm128_tiling_data.h"
#include "op_kernel/utils.h"

#include <kernel_operator.h>
#include <lib/matmul_intf.h>

using AscendC::DataCopy;
using AscendC::GetBlockIdx;
using AscendC::GetBlockNum;
using AscendC::HardEvent;
using AscendC::IsSameType;
using AscendC::LoadData;
using AscendC::LoadData2DParams;
using AscendC::PipeBarrier;
using AscendC::SetFlag;
using AscendC::TQue;
using AscendC::WaitFlag;
using AscendC::Cast;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::RoundMode;
using AscendC::TPipe;
using AscendC::TPosition;


namespace LightningAttn::Kernel
{

template <typename inputT, typename outputT> class LaBackwardCMM128
{
public:
    static constexpr uint32_t CUBE_BLOCK = 16;
    static constexpr uint32_t CUBE_BLOCK_SIZE = 16 * 16;
    static constexpr uint64_t ALIGNER_L1 = 512l;

    __aicore__ inline LaBackwardCMM128() {}

    __aicore__ inline void Init(GM_ADDR dO,
                                GM_ADDR Q,
                                GM_ADDR K,
                                GM_ADDR V,
                                GM_ADDR slopeRate,
                                GM_ADDR kvState,
                                GM_ADDR gradQueryOut,
                                GM_ADDR gradKeyOut,
                                GM_ADDR gradValueOut,
                                GM_ADDR workspace,
                                Common::LoCmm128TilingData* tilingStruct,
                                AscendC::TPipe* pipe)
    {
        InitKernelParams(tilingStruct); 
        InitOffsets();
        AllocateGlobalBuffers(dO, Q, K, V, slopeRate, kvState, gradQueryOut, gradKeyOut, gradValueOut, workspace);        
        FillInDecayMatrices(slopeRate, pipe);
        InitQueueBuffers(pipe);
    }

    __aicore__ inline void Process()
    {
        if (is_independent)
        {
            if (squadron_Q)
                ProcessdQindependent();
            if (squadron_KV)
                ProcessdKdV();
        }
        else
            ProcessdQdKdV();
    }


private:
    __aicore__ inline uint32_t CeilCubeBlock(uint32_t len)
    {
        return (len + CUBE_BLOCK - 1) / CUBE_BLOCK;
    }

    __aicore__ inline void SetWaitFlagMte1ToFIX()
    {
        event_t eventIdMte1ToFIX = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_FIX));
        SetFlag<HardEvent::MTE1_FIX>(eventIdMte1ToFIX);
        WaitFlag<HardEvent::MTE1_FIX>(eventIdMte1ToFIX);
        PipeBarrier<PIPE_M>();
    }

    __aicore__ inline void SetWaitFlagMte2ToFIX()
    {
        event_t eventIdMte2ToFIX = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_FIX));
        SetFlag<HardEvent::MTE2_FIX>(eventIdMte2ToFIX);
        WaitFlag<HardEvent::MTE2_FIX>(eventIdMte2ToFIX);
        PipeBarrier<PIPE_M>();
    }
    __aicore__ inline void SetWaitFlagFIXToMte1()
    {
        event_t eventIdFIXToMte1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::FIX_MTE1));
        SetFlag<HardEvent::FIX_MTE1>(eventIdFIXToMte1);
        WaitFlag<HardEvent::FIX_MTE1>(eventIdFIXToMte1);
        PipeBarrier<PIPE_M>();
    }
    __aicore__ inline void SetWaitFlagFIXToMte2()
    {
        event_t eventIdFIXToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::FIX_MTE2));
        SetFlag<HardEvent::FIX_MTE2>(eventIdFIXToMte2);
        WaitFlag<HardEvent::FIX_MTE2>(eventIdFIXToMte2);
        PipeBarrier<PIPE_M>();
    }
    __aicore__ inline void SetWaitFlagMte1ToMte2()
    {
        event_t eventIdMte1ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE2));
        SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2);
        PipeBarrier<PIPE_M>();
    }

    __aicore__ inline void InitKernelParams(Common::LoCmm128TilingData* tilingStruct)
    {
        InitKernelDataParams(tilingStruct);
        InitKernelCoreParams();
    }

    __aicore__ inline void InitKernelDataParams(Common::LoCmm128TilingData* tilingStruct)
    {
        m = tilingStruct->seqLen;
        n = tilingStruct->dimSize;
        batch = tilingStruct->batchSize * tilingStruct->numHeads;
        head = tilingStruct->numHeads;
        chunk_m = tilingStruct->blockSize;
        chunk_n = n;
        num_chunks = (m + chunk_m - 1) / chunk_m;
        chunk_m_valid = m % chunk_m;
        is_independent = !(tilingStruct->isTraining);
        if (m % chunk_m == 0)
        {
            has_tail_chunk = false;
            chunk_m_valid = chunk_m;
        }
        else
        {
            has_tail_chunk = true;
            chunk_m_valid = m % chunk_m;
        }
    }

    __aicore__ inline void InitKernelCoreParams()
    {   
        block_id = GetBlockIdx();
        block_num = GetBlockNum();
        squadron_Q  = 0;
        squadron_KV = 0;
        squad_KV_size = 1;
        squad_Q_size  = 1;
        CalculateSquadParams();

        block_Q_size = batch / squad_Q_size;
        extra_Q_tasks = batch % squad_Q_size;
        block_KV_size = batch / squad_KV_size;
        extra_KV_tasks = batch % squad_KV_size;
        block_size = batch / block_num;
        extra_tasks = batch % block_num;    

        if ASCEND_IS_AIC
        {           
            if (block_id < extra_tasks)
                block_size++;
            if (squadron_Q && squad_Q_idx < extra_Q_tasks)            
                block_Q_size++;            
            if (squadron_KV && squad_KV_idx < extra_KV_tasks)
                block_KV_size++;
        }
        else
        {
            if (block_id / 2 < extra_tasks)
                block_size++;
            if (squadron_Q && squad_Q_idx / 2 < extra_Q_tasks)
                block_Q_size++;
            if (squadron_KV && squad_KV_idx / 2 < extra_KV_tasks)
                block_KV_size++;
        }
    }

    __aicore__ inline void CalculateSquadParams()
    {
        // Solve eq: batch + [batch/z] == block_num -> squad_KV_size = block_num - math.trunc((block_num+z)/(z+1))
        if (block_num > 1)
        {
            uint64_t squad_KV_size2 = block_num - (block_num + 2) / 3;
            squad_KV_size = block_num - (block_num + 1) / 2;
            if ((batch - 1) % squad_KV_size2 > (batch - 1) % squad_KV_size)
                squad_KV_size = squad_KV_size2;
            squad_Q_size = block_num - squad_KV_size;
            if ASCEND_IS_AIC
            {
                if (block_id < squad_KV_size)
                {
                    squadron_KV = 1;
                    squadron_Q = 0;
                    squad_KV_idx = block_id;
                    squad_Q_idx = 0;
                }
                else
                {
                    squadron_Q = 1;
                    squadron_KV = 0;
                    squad_Q_idx = block_id - squad_KV_size;
                    squad_KV_idx = 0;
                }
            }
            if ASCEND_IS_AIV
            {
                if (block_id / 2 < squad_KV_size)
                {
                    squadron_Q = 0;
                    squadron_KV = 1;
                    squad_Q_idx = 0;
                    squad_KV_idx = block_id;
                }
                else
                {
                    squadron_Q = 1;
                    squadron_KV = 0;
                    squad_Q_idx = block_id - 2 * squad_KV_size;
                    squad_KV_idx = 0;
                }
            }
        }
        else
        {
            squadron_KV = 1;
            squadron_Q = 1;
            squad_KV_size = block_num;
            squad_Q_size = block_num;
            squad_Q_idx = block_id;
            squad_KV_idx = block_id;
        }
    }
    
    __aicore__ inline void InitOffsets()
    {        
        InitQueueOffsets();
        InitGMOffsets();
    }

     __aicore__ inline void InitGMOffsets()
    {         
        offset_dO = m * n - chunk_n * chunk_m_valid;
        offset_K = m * n - chunk_n * chunk_m_valid;
        offset_Q = m * n - chunk_n * chunk_m_valid;
        offset_V = m * n - chunk_n * chunk_m_valid;
        offset_KVforward = chunk_n * chunk_n * (num_chunks - 1);
        offset_ResQ = m * n - chunk_n * chunk_m_valid;
        offset_ResK = m * n - chunk_n * chunk_m_valid;
        offset_ResV = m * n - chunk_n * chunk_m_valid;
        offset_Mq = 0;
        offset_Mkv = 0;
        offset_Mk = 0;
        offset_batch = 0;
        offset_batch_KVf = 0;
        offset_batch_M = 0;

        if ASCEND_IS_AIC
        {
            offset_batch = block_id * m * n;
            offset_batch_KVf = block_id * n * n * num_chunks;
            offset_batch_TMP = block_id * chunk_m * chunk_m;
            offset_batch_M = (block_id * chunk_m * chunk_m) % (head * chunk_m * chunk_m);
            offset_Mq = (block_id * chunk_m * chunk_m) % (head * chunk_m * chunk_m);
            offset_Mkv = (block_id * chunk_n * chunk_n) % (head * chunk_n * chunk_n);
            offset_Mk = (block_id * chunk_m * chunk_m) % (head * chunk_m * chunk_m);
        }
        else
        {
            offset_batch = (block_id / 2) * m * n;
            offset_batch_TMP = (block_id / 2) * chunk_m * chunk_m;
            offset_batch_M = ((block_id / 2) * chunk_m * chunk_m) % (head * chunk_m * chunk_m);
        }
    }

     __aicore__ inline void InitQueueOffsets()
    {  
         // offsets for inqueueDec
        queue_offset_Mq = 0;
        queue_offset_Mk = queue_offset_Mq + AlignItemsByBytes(chunk_m * chunk_m, sizeof(inputT), ALIGNER_L1);
        queue_offset_Mkv = queue_offset_Mk + AlignItemsByBytes(chunk_m * chunk_m, sizeof(inputT), ALIGNER_L1); 

        // offsets for inqueueA1
        queue_offset_dO = 0;
        queue_offset_Q = queue_offset_dO + AlignItemsByBytes(chunk_m * chunk_n, sizeof(inputT), ALIGNER_L1);
        queue_offset_K = queue_offset_Q + AlignItemsByBytes(chunk_m * chunk_n, sizeof(inputT), ALIGNER_L1);
        queue_offset_V = queue_offset_K + AlignItemsByBytes(chunk_m * chunk_n, sizeof(inputT), ALIGNER_L1);

        queue_offset_tmp2 = 0;
        queue_offset_tmp = 0;

        queue_offset_KV = 0;
        queue_offset_FP = 0;
    }

    __aicore__ inline void FillInDecayMatrices(GM_ADDR slopeRate, TPipe* pipe)
    {
        if ASCEND_IS_AIV
        {
            AscendC::GlobalTensor<float> slopeGm;
            slopeGm.SetGlobalBuffer((__gm__ float*)slopeRate);

            LightningAttn::Kernel::CmmDecayGenerator<float> cdgOp;
            cdgOp.Init(head, chunk_n, chunk_m, pipe);
            cdgOp.Generate(MqGM, MkGM, MkvGM, slopeGm);
            pipe->Reset();

            LightningAttn::Kernel::DiagDecayGenerator<float> ddgOp;
            ddgOp.Init(pipe, head, chunk_m);
            ddgOp.GenerateDiagDecay(MGM, slopeGm);
            pipe->Reset();
        }
    }

     __aicore__ inline void AllocateGlobalBuffers(GM_ADDR dO,
                                GM_ADDR Q,
                                GM_ADDR K,
                                GM_ADDR V,
                                GM_ADDR slopeRate,
                                GM_ADDR kvState,
                                GM_ADDR gradQueryOut,
                                GM_ADDR gradKeyOut,
                                GM_ADDR gradValueOut,
                                GM_ADDR workspace)
    {
        dOGM.SetGlobalBuffer((__gm__ inputT*)dO, batch * m * n);
        QGM.SetGlobalBuffer((__gm__ inputT*)Q, batch * m * n);
        KGM.SetGlobalBuffer((__gm__ inputT*)K, batch * m * n);
        VGM.SetGlobalBuffer((__gm__ inputT*)V, batch * m * n);
	    if (!is_independent)
            KVforwardGM.SetGlobalBuffer((__gm__ inputT*)kvState, batch * n * n * num_chunks);

        const int64_t blockMatrixSize = head * chunk_m * chunk_m;
        const int64_t headMatrixSize = head * chunk_n * chunk_n;
        workspace = SetTensorBufferFromWorkspace(MqGM, blockMatrixSize, workspace);
        workspace = SetTensorBufferFromWorkspace(MkGM, blockMatrixSize, workspace);
        workspace = SetTensorBufferFromWorkspace(MkvGM, headMatrixSize, workspace);
        workspace = SetTensorBufferFromWorkspace(MGM, blockMatrixSize, workspace);

        int64_t ovmGMSize = block_num * chunk_m * chunk_m * sizeof(inputT);
        ovmGMSize = CeilDiv(ovmGMSize, 512l) * 512l / sizeof(inputT);
        int64_t qkmGMSize = block_num * chunk_m * chunk_m * sizeof(inputT);
        qkmGMSize = CeilDiv(qkmGMSize, 512l) * 512l / sizeof(inputT);
        int64_t dovtGMSize = block_num * chunk_m * chunk_m * sizeof(float);
        dovtGMSize = CeilDiv(dovtGMSize, 512l) * 512l / sizeof(float);
        int64_t qktGMSize = block_num * chunk_m * chunk_m * sizeof(float);
        qktGMSize = CeilDiv(qktGMSize, 512l) * 512l / sizeof(float);
        OVMGM.SetGlobalBuffer((__gm__ inputT*)workspace, ovmGMSize / sizeof(inputT));
        workspace = (GM_ADDR)((__gm__ inputT*)workspace + ovmGMSize);
        QKMGM.SetGlobalBuffer((__gm__ inputT*)workspace, qkmGMSize / sizeof(inputT));
        workspace = (GM_ADDR)((__gm__ inputT*)workspace + qkmGMSize);
        dOVTGM.SetGlobalBuffer((__gm__ float*)workspace, dovtGMSize / sizeof(float));
        workspace = (GM_ADDR)((__gm__ float*)workspace + dovtGMSize);
        QKTGM.SetGlobalBuffer((__gm__ float*)workspace, qktGMSize / sizeof(float));
        workspace = (GM_ADDR)((__gm__ float*)workspace + qktGMSize);

        K_resultGM.SetGlobalBuffer((__gm__ outputT*)gradKeyOut, batch * m * n);
        V_resultGM.SetGlobalBuffer((__gm__ outputT*)gradValueOut, batch * m * n);
        Q_resultGM.SetGlobalBuffer((__gm__ outputT*)gradQueryOut, batch * m * n);
    }

    __aicore__ inline void InitQueueBuffers(TPipe* pipe)
    {
        AscendC::SyncAll<false>();
        if ASCEND_IS_AIC
        {            
            uint64_t chunk_max = (chunk_n > chunk_m) ? chunk_n : chunk_m;
            inQueueFPSizeItems = queue_offset_FP + AlignItemsByBytes(chunk_max * chunk_max, sizeof(inputT), ALIGNER_L1);
            inQueueKVSizeItems = queue_offset_KV + AlignItemsByBytes(chunk_n * chunk_n, sizeof(inputT), ALIGNER_L1);
            inQueueA1SizeItems = queue_offset_V + AlignItemsByBytes(chunk_max * chunk_max, sizeof(inputT), ALIGNER_L1);
            inQueueTMPSizeItems = queue_offset_tmp + AlignItemsByBytes(chunk_max * chunk_max, sizeof(inputT), ALIGNER_L1);
            inQueueTMP2SizeItems = queue_offset_tmp2 + AlignItemsByBytes(chunk_max * chunk_max, sizeof(inputT), ALIGNER_L1);
            inQueueDecSizeItems = queue_offset_Mkv + AlignItemsByBytes(chunk_n * chunk_n, sizeof(inputT), ALIGNER_L1);
            uint64_t QueueL0SizeItems = AlignItemsByBytes(chunk_max * chunk_max, sizeof(inputT), ALIGNER_L1);

            pipe->InitBuffer(inQueueDec, 1, inQueueDecSizeItems * sizeof(inputT));
            pipe->InitBuffer(inQueueA1, 1, inQueueA1SizeItems * sizeof(inputT));
            pipe->InitBuffer(inQueueFP, 1, inQueueFPSizeItems * sizeof(inputT));
            pipe->InitBuffer(inQueueKV, 1, 2 * inQueueKVSizeItems * sizeof(inputT));
            pipe->InitBuffer(inQueueTMP, 1, inQueueTMPSizeItems * sizeof(inputT));
            pipe->InitBuffer(inQueueTMP2, 1, inQueueTMP2SizeItems * sizeof(inputT));

            pipe->InitBuffer(inQueueA2, 2, QueueL0SizeItems * sizeof(inputT));
            pipe->InitBuffer(inQueueB2, 2, QueueL0SizeItems * sizeof(inputT));
            pipe->InitBuffer(outQueueCO1, 2, QueueL0SizeItems * sizeof(float));
        }
        if ASCEND_IS_AIV
        {
            pipe->InitBuffer(inQueueM, 1, (chunk_m * chunk_m) / 2 * sizeof(float));
            pipe->InitBuffer(inQueueX, 1, (chunk_m * chunk_m) / 2 * sizeof(float));
            pipe->InitBuffer(outQueueZ, 1, (chunk_m * chunk_m) / 2 * sizeof(float));
            pipe->InitBuffer(outQueueTMP, 1, (chunk_m * chunk_m) / 2 * sizeof(inputT));
        }
    }
    
    __aicore__ inline void ProcessdQindependent()
    {
        auto flag1 = 0x1;
        auto flag2 = 0x2;
        auto flag4 = 0x4;
        InitGMOffsetsForQindependent();
        for (int batch_idx = 0; batch_idx < block_Q_size; ++batch_idx)
        {
            ProcessCubeCalculationsForQindependent(flag1, flag2, flag4);
            ProcessVectorCalculationsForQindependent(flag1, flag2);
            RecalculateGMOffsetsForQindependent();
        } 
    }

    __aicore__ inline void InitGMOffsetsForQindependent()
    {
        if ASCEND_IS_AIC
        {
            offset_batch = squad_Q_idx * m * n;
            offset_batch_KVf = squad_Q_idx * n * n * num_chunks;
            offset_batch_TMP = block_id * chunk_m * chunk_m;
            offset_batch_M = (squad_Q_idx * chunk_m * chunk_m) % (head * chunk_m * chunk_m);
            offset_Mq = (squad_Q_idx * chunk_m * chunk_m) % (head * chunk_m * chunk_m);
            offset_Mkv = (squad_Q_idx * chunk_n * chunk_n) % (head * chunk_n * chunk_n);
            offset_Mk = (squad_Q_idx * chunk_m * chunk_m) % (head * chunk_m * chunk_m);
        }
        else
        {
            offset_batch = (squad_Q_idx / 2) * m * n;
            offset_batch_TMP = (block_id / 2) * chunk_m * chunk_m;
            offset_batch_M = ((squad_Q_idx / 2) * chunk_m * chunk_m) % (head * chunk_m * chunk_m);
        }

        offset_ResQ = 0;
        offset_dO = 0;
        offset_K = 0;
        offset_Q = 0;
        offset_V = 0;
        chunk_m_valid = chunk_m;
    }

    __aicore__ inline void ProcessCubeCalculationsForQindependent(uint16_t flag1, uint16_t flag2, uint16_t flag4)
    {
        if ASCEND_IS_AIC
        {
            LocalTensor<inputT> decLocal = inQueueDec.AllocTensor<inputT>();
            inQueueDec.EnQue(decLocal);

            CopyInMq();
            CopyInMkv();
            CopyInMk();

            for (uint16_t i = 0; i < num_chunks; ++i)
            {
                if (i == num_chunks - 1 && has_tail_chunk)
                    UpdateValidParams();
                else
                {
                    LocalTensor<inputT> a1Local = inQueueA1.AllocTensor<inputT>();
                    inQueueA1.EnQue<inputT>(a1Local);
                }
                CopyIndO(true);
                CopyInV(true);
                ComputedOVT();
                CopyOutdOVT();

                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(flag1);
                CopyInK(true);

                // dQ
                if (i > 0)
                {
                    if (i < num_chunks - 1)
                    {
                        ComputeKV(i, false);
                        CopyOutKV((i + 1) % 2, false);
                    }

                    ComputeInterQ(i);
                    AscendC::CrossCoreWaitFlag(flag2);
                    CopyInTMPintraOVM();
                    ComputeIntraQ(false, false);
                    CopyOutQ(true);
                }
                else
                {
                    if (num_chunks > 1)
                    {
                        ComputeKV(i, true);
                        CopyOutKV((i + 1) % 2, true);
                    }
                    AscendC::CrossCoreWaitFlag(flag2);
                    CopyInTMPintraOVM();
                    ComputeIntraQ(true, false);
                    CopyOutQ(true);
                }
                LocalTensor<inputT> a1Local = inQueueA1.DeQue<inputT>();
                inQueueA1.FreeTensor(a1Local);
                chunk_m_valid = chunk_m;
            }

            if (num_chunks > 1)
            {
                LocalTensor<inputT> kvLocal = inQueueKV.DeQue<inputT>();
                inQueueKV.FreeTensor(kvLocal);
            }
            decLocal = inQueueDec.DeQue<inputT>();
            inQueueDec.FreeTensor(decLocal);
        } // AIC
    }

    __aicore__ inline void ProcessVectorCalculationsForQindependent(uint16_t flag1, uint16_t flag2)
    {
        if ASCEND_IS_AIV
        {
            AdamarCopyInM();
            LocalTensor<float> mLocal = inQueueM.DeQue<float>();
            for (uint16_t i = 0; i < num_chunks; i++)
            {
                AscendC::CrossCoreWaitFlag(flag1);
                AdamarCopyIn(&dOVTGM);
                AdamarCompute(&mLocal);
                AdamarCopyOut(&OVMGM);
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(flag2);
            }
            inQueueM.FreeTensor(mLocal);

        } // AIV
    }

    __aicore__ inline void RecalculateGMOffsetsForQindependent()
    {
        chunk_m_valid = chunk_m;
        offset_batch += squad_Q_size * m * n;
        offset_ResQ = 0;
        offset_dO = 0;
        offset_K = 0;
        offset_Q = 0;
        offset_V = 0;
        offset_KVforward = chunk_n * chunk_n * (num_chunks - 1);
        offset_batch_KVf += squad_Q_size * n * n * num_chunks;
        offset_Mq += squad_Q_size * chunk_m * chunk_m;
        offset_Mkv += squad_Q_size * chunk_n * chunk_n;
        offset_Mk += squad_Q_size * chunk_m * chunk_m;
        offset_batch_M += squad_Q_size * chunk_m * chunk_m;
        offset_Mq %= head * chunk_m * chunk_m;
        offset_Mkv %= head * chunk_n * chunk_n;
        offset_Mk %= head * chunk_m * chunk_m;
        offset_batch_M %= head * chunk_m * chunk_m;
    }

    __aicore__ inline void ProcessdKdV()
    {
        auto flag1 = 0x1;
        auto flag2 = 0x2;
        auto flag4 = 0x4;
        InitGMOffsetsFordKdV();
        for (int batch_idx = 0; batch_idx < block_KV_size; ++batch_idx)
        {
            ProcessCubeCalculationsFordKdV(flag1, flag2, flag4);            
            ProcessVectorCalculationsFordKdV(flag1, flag2, flag4);
            RecalculateGMOffsetsFordKdV();
        } 
    }

    __aicore__ inline void InitGMOffsetsFordKdV()
    {
        if (m % chunk_m == 0)
            chunk_m_valid = chunk_m;
        else
            chunk_m_valid = m % chunk_m;
        offset_ResK = m * n - chunk_n * chunk_m_valid;
        offset_ResV = m * n - chunk_n * chunk_m_valid;
        offset_dO = m * n - chunk_n * chunk_m_valid;
        offset_K = m * n - chunk_n * chunk_m_valid;
        offset_Q = m * n - chunk_n * chunk_m_valid;
        offset_V = m * n - chunk_n * chunk_m_valid;

        if ASCEND_IS_AIC
        {
            offset_batch = squad_KV_idx * m * n;
            offset_batch_KVf = squad_KV_idx * n * n * num_chunks;
            offset_batch_TMP = block_id * chunk_m * chunk_m;
            offset_batch_M = (squad_KV_idx * chunk_m * chunk_m) % (head * chunk_m * chunk_m);
            offset_Mq = (squad_KV_idx * chunk_m * chunk_m) % (head * chunk_m * chunk_m);
            offset_Mkv = (squad_KV_idx * chunk_n * chunk_n) % (head * chunk_n * chunk_n);
            offset_Mk = (squad_KV_idx * chunk_m * chunk_m) % (head * chunk_m * chunk_m);
        }
        else
        {
            offset_batch = (squad_KV_idx / 2) * m * n;
            offset_batch_TMP = (block_id / 2) * chunk_m * chunk_m;
            offset_batch_M = ((squad_KV_idx / 2) * chunk_m * chunk_m) % (head * chunk_m * chunk_m);
        }
    }

    __aicore__ inline void ProcessCubeCalculationsFordKdV(uint16_t flag1, uint16_t flag2, uint16_t flag4)
    {
        if ASCEND_IS_AIC
        {
            LocalTensor<inputT> decLocal = inQueueDec.AllocTensor<inputT>();
            inQueueDec.EnQue(decLocal);
            AscendC::LocalTensor<inputT> kvLocal = inQueueKV.AllocTensor<inputT>();
            inQueueKV.EnQue<inputT>(kvLocal);
            CopyInMq();
            CopyInMkv();
            CopyInMk();

            for (uint16_t i = 0; i < num_chunks; ++i)
            {
                if (i == 0 && has_tail_chunk)
                    UpdateValidParams();
                else
                {
                    LocalTensor<inputT> a1Local = inQueueA1.AllocTensor<inputT>();
                    inQueueA1.EnQue<inputT>(a1Local);
                }
                CopyIndO(false);
                CopyInV(false);
                ComputedOVT();
                CopyOutdOVT();

                CopyInQ(false);
                CopyInK(false);
                ComputeQKT();
                CopyOutQKT();
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(flag1);

                // dKV
                if (i < num_chunks - 1)
                {
                    ComputedKV((i == 0), i);
                    CopyOutdKV((i + 1) % 2);
                }

                // dK
                if (i > 0)
                {
                    ComputeInterK(i);
                    AscendC::CrossCoreWaitFlag(flag2);
                    CopyInTMPintraOVM();
                    ComputeIntraK(false);
                    CopyOutK();
                }
                else
                {
                    AscendC::CrossCoreWaitFlag(flag2);
                    CopyInTMPintraOVM();
                    ComputeIntraK(true);
                    CopyOutK();
                }

                // dV
                if (i > 0)
                {
                    ComputeInterV(i);
                    AscendC::CrossCoreWaitFlag(flag4);
                    CopyInTMPintraQKM();
                    ComputeIntraV(false);
                    CopyOutV();
                }
                else
                {
                    AscendC::CrossCoreWaitFlag(flag4);
                    CopyInTMPintraQKM();
                    ComputeIntraV(true);
                    CopyOutV();
                }

                LocalTensor<inputT> a1Local = inQueueA1.DeQue<inputT>();
                inQueueA1.FreeTensor(a1Local);

                chunk_m_valid = chunk_m;
            }

            decLocal = inQueueDec.DeQue<inputT>();
            inQueueDec.FreeTensor(decLocal);

            kvLocal = inQueueKV.DeQue<inputT>();
            inQueueKV.FreeTensor(kvLocal);

        } // AIC
    }

    __aicore__ inline void ProcessVectorCalculationsFordKdV(uint16_t flag1, uint16_t flag2, uint16_t flag4)
    {
        if ASCEND_IS_AIV
        {
            AdamarCopyInM();
            LocalTensor<float> mLocal = inQueueM.DeQue<float>();

            for (uint16_t i = 0; i < num_chunks; i++)
            {
                AscendC::CrossCoreWaitFlag(flag1);
                AdamarCopyIn(&dOVTGM);
                AdamarCompute(&mLocal);
                AdamarCopyOut(&OVMGM);
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(flag2);
                AdamarCopyIn(&QKTGM);
                AdamarCompute(&mLocal);
                AdamarCopyOut(&QKMGM);
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(flag4);
            }
            inQueueM.FreeTensor(mLocal);
        } // AIV
    }

    __aicore__ inline void RecalculateGMOffsetsFordKdV()
    {
        if (m % chunk_m == 0)
            chunk_m_valid = chunk_m;
        else
            chunk_m_valid = m % chunk_m;

        offset_batch += squad_KV_size * m * n;
        offset_ResQ = m * n - chunk_n * chunk_m_valid;
        offset_ResK = m * n - chunk_n * chunk_m_valid;
        offset_ResV = m * n - chunk_n * chunk_m_valid;
        offset_dO = m * n - chunk_n * chunk_m_valid;
        offset_K = m * n - chunk_n * chunk_m_valid;
        offset_Q = m * n - chunk_n * chunk_m_valid;
        offset_V = m * n - chunk_n * chunk_m_valid;
        offset_KVforward = chunk_n * chunk_n * (num_chunks - 1);
        offset_batch_KVf += squad_KV_size * n * n * num_chunks;
        offset_Mq += squad_KV_size * chunk_m * chunk_m;
        offset_Mkv += squad_KV_size * chunk_n * chunk_n;
        offset_Mk += squad_KV_size * chunk_m * chunk_m;
        offset_batch_M += squad_KV_size * chunk_m * chunk_m;
        offset_Mq %= head * chunk_m * chunk_m;
        offset_Mkv %= head * chunk_n * chunk_n;
        offset_Mk %= head * chunk_m * chunk_m;
        offset_batch_M %= head * chunk_m * chunk_m;
    }

    __aicore__ inline void ProcessdQdKdV()
    {
        auto flag1 = 0x1;
        auto flag2 = 0x2;
        auto flag4 = 0x4;
        for (int batch_idx = 0; batch_idx < block_size; ++batch_idx)
        {
            ProcessCubeCalculationsFordQdKdV(flag1, flag2, flag4);
            ProcessVectorCalculationsFordQdKdV(flag1, flag2, flag4);
            RecalculateGMOffsetsFordQdKdV();            
        } 
    }

    __aicore__ inline void ProcessCubeCalculationsFordQdKdV(uint16_t flag1, uint16_t flag2, uint16_t flag4)
    {
        if ASCEND_IS_AIC
        {
            LocalTensor<inputT> decLocal = inQueueDec.AllocTensor<inputT>();
            inQueueDec.EnQue(decLocal);
            AscendC::LocalTensor<inputT> kvLocal = inQueueKV.AllocTensor<inputT>();
            inQueueKV.EnQue<inputT>(kvLocal);
            CopyInMq();
            CopyInMkv();
            CopyInMk();

            for (uint16_t i = 0; i < num_chunks; ++i)
            {
                if (i == 0 && has_tail_chunk)
                    UpdateValidParams();
                else
                {
                    LocalTensor<inputT> a1Local = inQueueA1.AllocTensor<inputT>();
                    inQueueA1.EnQue<inputT>(a1Local);
                }
                CopyIndO(false);
                CopyInV(false);
                ComputedOVT();
                CopyOutdOVT();
                CopyInQ(false);
                CopyInK(false);
                ComputeQKT();
                CopyOutQKT();
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(flag1);

                // dKV
                if (i < num_chunks - 1)
                {
                    ComputedKV((i == 0), i);
                    CopyOutdKV((i + 1) % 2);
                }

                // dQ
                if (i < num_chunks - 1)
                {
                    ComputeInterQ(i);
                    AscendC::CrossCoreWaitFlag(flag2);
                    CopyInTMPintraOVM();
                    ComputeIntraQ(false, true);
                    CopyOutQ(false);
                }
                else
                {
                    AscendC::CrossCoreWaitFlag(flag2);
                    CopyInTMPintraOVM();
                    ComputeIntraQ(true, true);
                    CopyOutQ(false);
                }

                // dK
                if (i > 0)
                {
                    ComputeInterK(i);
                    ComputeIntraK(false);
                    CopyOutK();
                }
                else
                {
                    ComputeIntraK(true);
                    CopyOutK();
                }

                // dV
                if (i > 0)
                {
                    ComputeInterV(i);
                    AscendC::CrossCoreWaitFlag(flag4);
                    CopyInTMPintraQKM();
                    ComputeIntraV(false);
                    CopyOutV();
                }
                else
                {
                    AscendC::CrossCoreWaitFlag(flag4);
                    CopyInTMPintraQKM();
                    ComputeIntraV(true);
                    CopyOutV();
                }

                LocalTensor<inputT> a1Local = inQueueA1.DeQue<inputT>();
                inQueueA1.FreeTensor(a1Local);
                chunk_m_valid = chunk_m;
            }

            decLocal = inQueueDec.DeQue<inputT>();
            inQueueDec.FreeTensor(decLocal);
            kvLocal = inQueueKV.DeQue<inputT>();
            inQueueKV.FreeTensor(kvLocal);
        } // AIC
    }

    __aicore__ inline void ProcessVectorCalculationsFordQdKdV(uint16_t flag1, uint16_t flag2, uint16_t flag4)
    {
        if ASCEND_IS_AIV
        {
            AdamarCopyInM();
            LocalTensor<float> mLocal = inQueueM.DeQue<float>();
            for (uint16_t i = 0; i < num_chunks; i++)
            {
                AscendC::CrossCoreWaitFlag(flag1);
                AdamarCopyIn(&dOVTGM);
                AdamarCompute(&mLocal);
                AdamarCopyOut(&OVMGM);
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(flag2);
                AdamarCopyIn(&QKTGM);
                AdamarCompute(&mLocal);
                AdamarCopyOut(&QKMGM);
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(flag4);
            }
            inQueueM.FreeTensor(mLocal);
        } // AIV
    }

    __aicore__ inline void RecalculateGMOffsetsFordQdKdV()
    {
        if (m % chunk_m == 0)
            chunk_m_valid = chunk_m;
        else
            chunk_m_valid = m % chunk_m;

        offset_batch += block_num * m * n;
        offset_ResQ = m * n - chunk_n * chunk_m_valid;
        offset_ResK = m * n - chunk_n * chunk_m_valid;
        offset_ResV = m * n - chunk_n * chunk_m_valid;
        offset_dO = m * n - chunk_n * chunk_m_valid;
        offset_K = m * n - chunk_n * chunk_m_valid;
        offset_Q = m * n - chunk_n * chunk_m_valid;
        offset_V = m * n - chunk_n * chunk_m_valid;
        offset_KVforward = chunk_n * chunk_n * (num_chunks - 1);
        offset_batch_KVf += block_num * n * n * num_chunks;
        offset_Mq += block_num * chunk_m * chunk_m;
        offset_Mkv += block_num * chunk_n * chunk_n;
        offset_Mk += block_num * chunk_m * chunk_m;
        offset_batch_M += block_num * chunk_m * chunk_m;
        offset_Mq %= head * chunk_m * chunk_m;
        offset_Mkv %= head * chunk_n * chunk_n;
        offset_Mk %= head * chunk_m * chunk_m;
        offset_batch_M %= head * chunk_m * chunk_m;
    }

    __aicore__ inline void UpdateValidParams()
    {
        AscendC::LocalTensor<inputT> dO1Local = inQueueA1.AllocTensor<inputT>();
        chunk_m_valid = m % chunk_m;
        // initialize with zeros dO + K + Q + V
        inputT zero = (inputT)0.0;
        uint16_t iter_block_size = static_cast<uint16_t>((chunk_m * chunk_n * sizeof(inputT)) / 32);
        AscendC::InitConstValue(dO1Local[queue_offset_dO], {1, iter_block_size, 0, zero});
        AscendC::InitConstValue(dO1Local[queue_offset_Q], {1, iter_block_size, 0, zero});
        AscendC::InitConstValue(dO1Local[queue_offset_K], {1, iter_block_size, 0, zero});
        AscendC::InitConstValue(dO1Local[queue_offset_V], {1, iter_block_size, 0, zero});
        inQueueA1.EnQue<inputT>(dO1Local);
    }

    __aicore__ inline void AdamarCopyInM()
    {
        AscendC::LocalTensor<float> mLocal = inQueueM.AllocTensor<float>();
        AscendC::DataCopy(mLocal, MGM[offset_batch_M + (block_id % 2) * (chunk_m * chunk_m / 2)], chunk_m * chunk_m / 2);
        inQueueM.EnQue<float>(mLocal);
    }

    __aicore__ inline void AdamarCopyIn(AscendC::GlobalTensor<float>* operandGM)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(
            xLocal, (*operandGM)[offset_batch_TMP + (block_id % 2) * (chunk_m * chunk_m / 2)], chunk_m * chunk_m / 2);

        inQueueX.EnQue<float>(xLocal);
    }

    __aicore__ inline void AdamarCompute(LocalTensor<float>* mLocal)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        AscendC::Mul(zLocal, xLocal, *mLocal, chunk_m * chunk_m / 2);

        outQueueZ.EnQue<float>(zLocal);
        zLocal = outQueueZ.DeQue<float>();

        AscendC::LocalTensor<inputT> tmpLocal = outQueueTMP.AllocTensor<inputT>();
        AscendC::Cast(tmpLocal, zLocal, AscendC::RoundMode::CAST_ROUND, chunk_m * chunk_m / 2);

        outQueueTMP.EnQue<inputT>(tmpLocal);
        inQueueX.FreeTensor(xLocal);
        outQueueZ.FreeTensor(zLocal);
    }

    __aicore__ inline void AdamarCopyOut(AscendC::GlobalTensor<inputT>* resultGM)
    {
        AscendC::LocalTensor<inputT> tmpLocal = outQueueTMP.DeQue<inputT>();

        AscendC::DataCopy(
            (*resultGM)[offset_batch_TMP + (block_id % 2) * (chunk_m * chunk_m / 2)], tmpLocal, chunk_m * chunk_m / 2);

        outQueueTMP.FreeTensor(tmpLocal);
    }

    __aicore__ inline void CopyInTMPintraOVM()
    {
        AscendC::LocalTensor<inputT> tmpLocal = inQueueTMP.AllocTensor<inputT>();

        AscendC::Nd2NzParams nd2nzA1Params;
        nd2nzA1Params.ndNum = 1;
        nd2nzA1Params.nValue = chunk_m;
        nd2nzA1Params.dValue = chunk_m;
        nd2nzA1Params.srcNdMatrixStride = 0;
        nd2nzA1Params.srcDValue = chunk_m;
        nd2nzA1Params.dstNzC0Stride = CeilCubeBlock(chunk_m) * CUBE_BLOCK;
        nd2nzA1Params.dstNzNStride = 1;
        nd2nzA1Params.dstNzMatrixStride = 0;

        AscendC::DataCopy(tmpLocal[queue_offset_tmp], OVMGM[offset_batch_TMP], nd2nzA1Params);
        inQueueTMP.EnQue<inputT>(tmpLocal);
    }

    __aicore__ inline void CopyInTMPintraQKM()
    {
        AscendC::LocalTensor<inputT> tmpLocal = inQueueTMP.AllocTensor<inputT>();

        AscendC::Nd2NzParams nd2nzA1Params;
        nd2nzA1Params.ndNum = 1;
        nd2nzA1Params.nValue = chunk_m;
        nd2nzA1Params.dValue = chunk_m;
        nd2nzA1Params.srcNdMatrixStride = 0;
        nd2nzA1Params.srcDValue = chunk_m;
        nd2nzA1Params.dstNzC0Stride = CeilCubeBlock(chunk_m) * CUBE_BLOCK;
        nd2nzA1Params.dstNzNStride = 1;
        nd2nzA1Params.dstNzMatrixStride = 0;

        AscendC::DataCopy(tmpLocal[queue_offset_tmp], QKMGM[offset_batch_TMP], nd2nzA1Params);
        inQueueTMP.EnQue<inputT>(tmpLocal);
    }

    __aicore__ inline void CopyInMq()
    {
        AscendC::LocalTensor<inputT> MqLocal = inQueueDec.DeQue<inputT>();

        AscendC::Nd2NzParams nd2nzA1Params;
        nd2nzA1Params.ndNum = 1;
        nd2nzA1Params.nValue = chunk_m;
        nd2nzA1Params.dValue = chunk_m;
        nd2nzA1Params.srcNdMatrixStride = 0;
        nd2nzA1Params.srcDValue = chunk_m;
        nd2nzA1Params.dstNzC0Stride = CeilCubeBlock(chunk_m) * CUBE_BLOCK;
        nd2nzA1Params.dstNzNStride = 1;
        nd2nzA1Params.dstNzMatrixStride = 0;

        AscendC::DataCopy(MqLocal[queue_offset_Mq], MqGM[offset_Mq], nd2nzA1Params);
        inQueueDec.EnQue<inputT>(MqLocal);
    }

    __aicore__ inline void CopyInMk()
    {
        LocalTensor<inputT> MkLocal = inQueueDec.DeQue<inputT>();
        AscendC::Nd2NzParams nd2nzB1Params;
        nd2nzB1Params.ndNum = 1;
        nd2nzB1Params.nValue = chunk_m;
        nd2nzB1Params.dValue = chunk_m;
        nd2nzB1Params.srcNdMatrixStride = 0;
        nd2nzB1Params.srcDValue = chunk_m;
        nd2nzB1Params.dstNzC0Stride = CeilCubeBlock(chunk_m) * CUBE_BLOCK;
        nd2nzB1Params.dstNzNStride = 1;
        nd2nzB1Params.dstNzMatrixStride = 0;

        AscendC::DataCopy(MkLocal[queue_offset_Mk], MkGM[offset_Mk], nd2nzB1Params);
        inQueueDec.EnQue<inputT>(MkLocal);
    }

    __aicore__ inline void CopyInMkv()
    {
        LocalTensor<inputT> MkvLocal = inQueueDec.DeQue<inputT>();

        AscendC::Nd2NzParams nd2nzA1Params;
        nd2nzA1Params.ndNum = 1;
        nd2nzA1Params.nValue = chunk_n;
        nd2nzA1Params.dValue = chunk_n;
        nd2nzA1Params.srcNdMatrixStride = 0;
        nd2nzA1Params.srcDValue = chunk_n;
        nd2nzA1Params.dstNzC0Stride = CeilCubeBlock(chunk_n) * CUBE_BLOCK;
        nd2nzA1Params.dstNzNStride = 1;
        nd2nzA1Params.dstNzMatrixStride = 0;

        AscendC::DataCopy(MkvLocal[queue_offset_Mkv], MkvGM[offset_Mkv], nd2nzA1Params);
        inQueueDec.EnQue<inputT>(MkvLocal);
    }

    __aicore__ inline void CopyIndO(bool ahead)
    {
        AscendC::LocalTensor<inputT> dO1Local = inQueueA1.DeQue<inputT>();

        AscendC::Nd2NzParams nd2nzB1Params;
        nd2nzB1Params.ndNum = 1;
        nd2nzB1Params.nValue = chunk_m_valid;
        nd2nzB1Params.dValue = chunk_n;
        nd2nzB1Params.srcNdMatrixStride = 0;
        nd2nzB1Params.srcDValue = chunk_n;
        nd2nzB1Params.dstNzC0Stride = CeilCubeBlock(chunk_m) * CUBE_BLOCK;
        nd2nzB1Params.dstNzNStride = 1;
        nd2nzB1Params.dstNzMatrixStride = 0;

        AscendC::DataCopy(dO1Local[queue_offset_dO], dOGM[offset_batch + offset_dO], nd2nzB1Params);
        inQueueA1.EnQue<inputT>(dO1Local);
        if (ahead)
            offset_dO += chunk_m * chunk_n;
        else
            offset_dO -= chunk_m * chunk_n;
    }

    __aicore__ inline void CopyInQ(bool ahead)
    {
        AscendC::LocalTensor<inputT> Q1Local = inQueueA1.DeQue<inputT>();

        AscendC::Nd2NzParams nd2nzA1Params;
        nd2nzA1Params.ndNum = 1;
        nd2nzA1Params.nValue = chunk_m_valid;
        nd2nzA1Params.dValue = chunk_n;
        nd2nzA1Params.srcNdMatrixStride = 0;
        nd2nzA1Params.srcDValue = chunk_n;
        nd2nzA1Params.dstNzC0Stride = CeilCubeBlock(chunk_m) * CUBE_BLOCK;
        nd2nzA1Params.dstNzNStride = 1;
        nd2nzA1Params.dstNzMatrixStride = 0;

        AscendC::DataCopy(Q1Local[queue_offset_Q], QGM[offset_batch + offset_Q], nd2nzA1Params);
        inQueueA1.EnQue<inputT>(Q1Local);
        if (ahead)
            offset_Q += chunk_m * chunk_n;
        else
            offset_Q -= chunk_m * chunk_n;
    }

    __aicore__ inline void CopyInK(bool ahead)
    {
        AscendC::LocalTensor<inputT> K1Local = inQueueA1.DeQue<inputT>();

        AscendC::Nd2NzParams nd2nzB1Params;
        nd2nzB1Params.ndNum = 1;
        nd2nzB1Params.nValue = chunk_m_valid;
        nd2nzB1Params.dValue = chunk_n;
        nd2nzB1Params.srcNdMatrixStride = 0;
        nd2nzB1Params.srcDValue = chunk_n;
        nd2nzB1Params.dstNzC0Stride = CeilCubeBlock(chunk_m) * CUBE_BLOCK;
        nd2nzB1Params.dstNzNStride = 1;
        nd2nzB1Params.dstNzMatrixStride = 0;

        AscendC::DataCopy(K1Local[queue_offset_K], KGM[offset_batch + offset_K], nd2nzB1Params);
        inQueueA1.EnQue<inputT>(K1Local);
        if (ahead)
            offset_K += chunk_m * chunk_n;
        else
            offset_K -= chunk_m * chunk_n;
    }

    __aicore__ inline void CopyInV(bool ahead)
    {
        AscendC::LocalTensor<inputT> V1Local = inQueueA1.DeQue<inputT>();

        AscendC::Nd2NzParams nd2nzB1Params;
        nd2nzB1Params.ndNum = 1;
        nd2nzB1Params.nValue = chunk_m_valid;
        nd2nzB1Params.dValue = chunk_n;
        nd2nzB1Params.srcNdMatrixStride = 0;
        nd2nzB1Params.srcDValue = chunk_n;
        nd2nzB1Params.dstNzC0Stride = CeilCubeBlock(chunk_m) * CUBE_BLOCK;
        nd2nzB1Params.dstNzNStride = 1;
        nd2nzB1Params.dstNzMatrixStride = 0;

        AscendC::DataCopy(V1Local[queue_offset_V], VGM[offset_batch + offset_V], nd2nzB1Params);
        inQueueA1.EnQue<inputT>(V1Local);
        if (ahead)
            offset_V += chunk_m * chunk_n;
        else
            offset_V -= chunk_m * chunk_n;
    }

    __aicore__ inline void CopyInKVforward()
    {
        AscendC::LocalTensor<inputT> KVforwardLocal = inQueueTMP2.AllocTensor<inputT>();

        AscendC::Nd2NzParams nd2nzB1Params;
        nd2nzB1Params.ndNum = 1;
        nd2nzB1Params.nValue = chunk_n;
        nd2nzB1Params.dValue = chunk_n;
        nd2nzB1Params.srcNdMatrixStride = 0;
        nd2nzB1Params.srcDValue = chunk_n;
        nd2nzB1Params.dstNzC0Stride = CeilCubeBlock(chunk_n) * CUBE_BLOCK;
        nd2nzB1Params.dstNzNStride = 1;
        nd2nzB1Params.dstNzMatrixStride = 0;

        AscendC::DataCopy(
            KVforwardLocal[queue_offset_tmp2], KVforwardGM[offset_batch_KVf + offset_KVforward], nd2nzB1Params);
        inQueueTMP2.EnQue<inputT>(KVforwardLocal);
        offset_KVforward -= chunk_n * chunk_n;
    }

    __aicore__ inline void SplitMq() 
    {
        AscendC::LocalTensor<inputT> a1Local = inQueueDec.DeQue<inputT>();
        AscendC::LocalTensor<inputT> a2Local = inQueueA2.AllocTensor<inputT>();

        // Nz -> Zz
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = (chunk_m / CUBE_BLOCK) * (chunk_m / CUBE_BLOCK);
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        AscendC::LoadData(a2Local, a1Local[queue_offset_Mq], loadDataParams);

        inQueueA2.EnQue<inputT>(a2Local);
        inQueueDec.EnQue<inputT>(a1Local);
    }

    __aicore__ inline void SplitMkv()
    {
        AscendC::LocalTensor<inputT> a1Local = inQueueDec.DeQue<inputT>();
        AscendC::LocalTensor<inputT> a2Local = inQueueA2.AllocTensor<inputT>();

        // Nz -> Zz
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = (chunk_n / CUBE_BLOCK) * (chunk_n / CUBE_BLOCK);
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        AscendC::LoadData(a2Local, a1Local[queue_offset_Mkv], loadDataParams);

        inQueueA2.EnQue<inputT>(a2Local);
        inQueueDec.EnQue<inputT>(a1Local);
    }

    __aicore__ inline void SplitMk()
    {
        AscendC::LocalTensor<inputT> b1Local = inQueueDec.DeQue<inputT>();
        AscendC::LocalTensor<inputT> b2Local = inQueueA2.AllocTensor<inputT>();

        // Nz -> Zn
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = (chunk_m / CUBE_BLOCK) * (chunk_m / CUBE_BLOCK);
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        AscendC::LoadData(b2Local, b1Local[queue_offset_Mk], loadDataParams);

        inQueueA2.EnQue<inputT>(b2Local);
        inQueueDec.EnQue<inputT>(b1Local);
    }

    __aicore__ inline void SplitMkB()
    {
        AscendC::LocalTensor<inputT> b1Local = inQueueDec.DeQue<inputT>();
        AscendC::LocalTensor<inputT> b2Local = inQueueB2.AllocTensor<inputT>();

        // Nz -> Zn
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = (chunk_m / CUBE_BLOCK) * (chunk_m / CUBE_BLOCK);
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        AscendC::LoadData(b2Local, b1Local[queue_offset_Mk], loadDataParams);

        inQueueB2.EnQue<inputT>(b2Local);
        inQueueDec.EnQue<inputT>(b1Local);
    }

    __aicore__ inline void SplitdOA()
    {
        AscendC::LocalTensor<inputT> dO1Local = inQueueA1.DeQue<inputT>();
        AscendC::LocalTensor<inputT> dO2Local = inQueueA2.AllocTensor<inputT>();

        uint32_t dstOffset = CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        // Nz -> Zz
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = chunk_n / CUBE_BLOCK;
        loadDataParams.srcStride = chunk_m / CUBE_BLOCK;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        for (int i = 0; i < chunk_m / CUBE_BLOCK; ++i)
            LoadData(dO2Local[(chunk_n / CUBE_BLOCK) * dstOffset * i],
                     dO1Local[srcOffset * i + queue_offset_dO],
                     loadDataParams);

        inQueueA2.EnQue<inputT>(dO2Local);
        inQueueA1.EnQue<inputT>(dO1Local);
    }

    __aicore__ inline void SplitdOB() 
    {
        AscendC::LocalTensor<inputT> dO1Local = inQueueA1.DeQue<inputT>();
        AscendC::LocalTensor<inputT> dO2Local = inQueueB2.AllocTensor<inputT>();

        uint32_t dstOffset = CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        // Nz -> Zn
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = chunk_n / CUBE_BLOCK;
        loadDataParams.srcStride = chunk_m / CUBE_BLOCK;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        for (int i = 0; i < chunk_m / CUBE_BLOCK; ++i)
            LoadData(dO2Local[i * dstOffset * (chunk_n / CUBE_BLOCK)],
                     dO1Local[i * srcOffset + queue_offset_dO],
                     loadDataParams);

        inQueueB2.EnQue<inputT>(dO2Local);
        inQueueA1.EnQue<inputT>(dO1Local);
    }

    __aicore__ inline void SplitQA()
    {
        AscendC::LocalTensor<inputT> q1Local = inQueueA1.DeQue<inputT>();
        AscendC::LocalTensor<inputT> q2Local = inQueueA2.AllocTensor<inputT>();

        uint32_t dstOffset = CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        // Nz -> Zz
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = chunk_n / CUBE_BLOCK;
        loadDataParams.srcStride = chunk_m / CUBE_BLOCK;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        for (int i = 0; i < chunk_m / CUBE_BLOCK; ++i)
            LoadData(
                q2Local[(chunk_n / CUBE_BLOCK) * dstOffset * i], q1Local[srcOffset * i + queue_offset_Q], loadDataParams);

        inQueueA2.EnQue<inputT>(q2Local);
        inQueueA1.EnQue<inputT>(q1Local);
    }

    __aicore__ inline void SplitQB()
    {
        AscendC::LocalTensor<inputT> b1Local = inQueueA1.DeQue<inputT>();
        AscendC::LocalTensor<inputT> b2Local = inQueueB2.AllocTensor<inputT>();

        uint32_t dstOffset = CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        // Nz -> Zn
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = (chunk_n / CUBE_BLOCK);
        loadDataParams.srcStride = (chunk_m / CUBE_BLOCK);
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        for (int i = 0; i < chunk_m / CUBE_BLOCK; ++i)
            LoadData(
                b2Local[i * dstOffset * (chunk_n / CUBE_BLOCK)], b1Local[i * srcOffset + queue_offset_Q], loadDataParams);

        inQueueA1.EnQue<inputT>(b1Local);
        inQueueB2.EnQue<inputT>(b2Local);
    }

    __aicore__ inline void SplitKTA()
    {
        AscendC::LocalTensor<inputT> KT1Local = inQueueA1.DeQue<inputT>();
        AscendC::LocalTensor<inputT> KT2Local = inQueueA2.AllocTensor<inputT>();

        uint32_t dstOffset = CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        // Nz -> Zz
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = (chunk_n / CUBE_BLOCK) * (chunk_m / CUBE_BLOCK);
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        AscendC::LoadData(KT2Local, KT1Local[queue_offset_K], loadDataParams);

        inQueueA2.EnQue<inputT>(KT2Local);
        inQueueA1.EnQue<inputT>(KT1Local);
    }

    __aicore__ inline void SplitKTB() 
    {
        AscendC::LocalTensor<inputT> KT1Local = inQueueA1.DeQue<inputT>();
        AscendC::LocalTensor<inputT> KT2Local = inQueueB2.AllocTensor<inputT>();

        uint32_t dstOffset = CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        // Nz -> Zn
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = (chunk_n / CUBE_BLOCK) * (chunk_m / CUBE_BLOCK);
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        AscendC::LoadData(KT2Local, KT1Local[queue_offset_K], loadDataParams);

        inQueueA1.EnQue<inputT>(KT1Local);
        inQueueB2.EnQue<inputT>(KT2Local);
    }

    __aicore__ inline void SplitKB()
    {
        AscendC::LocalTensor<inputT> k1Local = inQueueA1.DeQue<inputT>();
        AscendC::LocalTensor<inputT> k2Local = inQueueB2.AllocTensor<inputT>();

        uint32_t dstOffset = CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        // Nz -> Zn
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = chunk_n / CUBE_BLOCK;
        loadDataParams.srcStride = chunk_m / CUBE_BLOCK;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        for (int i = 0; i < chunk_m / CUBE_BLOCK; ++i)
            LoadData(
                k2Local[i * dstOffset * (chunk_n / CUBE_BLOCK)], k1Local[i * srcOffset + queue_offset_K], loadDataParams);

        inQueueA1.EnQue<inputT>(k1Local); // reuse
        inQueueB2.EnQue<inputT>(k2Local);
    }

    __aicore__ inline void SplitMqdO()
    {
        AscendC::LocalTensor<inputT> a1Local = inQueueFP.DeQue<inputT>();
        AscendC::LocalTensor<inputT> a2Local = inQueueA2.AllocTensor<inputT>();

        uint32_t dstOffset = CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;
        // Nz -> Zz
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = chunk_n / CUBE_BLOCK;
        loadDataParams.srcStride = chunk_m / CUBE_BLOCK;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;

        SetWaitFlagFIXToMte1();
        for (int i = 0; i < chunk_m / CUBE_BLOCK; ++i)
            LoadData(a2Local[(chunk_n / CUBE_BLOCK) * dstOffset * i],
                     a1Local[srcOffset * i + queue_offset_FP],
                     loadDataParams);

        inQueueA2.EnQue<inputT>(a2Local);
        inQueueFP.FreeTensor(a1Local);
    }

    __aicore__ inline void SplitKVT(bool reuse, uint64_t chunk_index)
    {
        AscendC::LocalTensor<inputT> kv1Local = inQueueKV.DeQue<inputT>();
        AscendC::LocalTensor<inputT> kv2Local = inQueueB2.AllocTensor<inputT>();

        uint32_t dstOffset = CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        // Nz -> Zn
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = (chunk_n / CUBE_BLOCK) * (chunk_n / CUBE_BLOCK);
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;

        uint64_t curKVOffset = queue_offset_KV + (chunk_index % 2) * chunk_n * chunk_n;
        SetWaitFlagFIXToMte1();
        AscendC::LoadData(kv2Local, kv1Local[curKVOffset], loadDataParams);

        if (reuse)
            inQueueKV.EnQue<inputT>(kv1Local);
        else
            inQueueKV.FreeTensor(kv1Local);
        inQueueB2.EnQue<inputT>(kv2Local);
    }

    __aicore__ inline void SplitKV(bool reuse, uint64_t chunk_index)
    {
        AscendC::LocalTensor<inputT> b1Local = inQueueKV.DeQue<inputT>();
        AscendC::LocalTensor<inputT> b2Local = inQueueB2.AllocTensor<inputT>();

        uint32_t dstOffset = CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        // Nz -> Zn
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = chunk_n / CUBE_BLOCK;
        loadDataParams.srcStride = chunk_n / CUBE_BLOCK;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;

        uint64_t curKVOffset = queue_offset_KV + (chunk_index % 2) * chunk_n * chunk_n;
        SetWaitFlagFIXToMte1();
        for (int i = 0; i < chunk_n / CUBE_BLOCK; ++i)
            LoadData(
                b2Local[i * dstOffset * (chunk_n / CUBE_BLOCK)], b1Local[curKVOffset + i * srcOffset], loadDataParams);

        if (reuse)
            inQueueKV.EnQue<inputT>(b1Local);
        else
            inQueueKV.FreeTensor(b1Local);
        inQueueB2.EnQue<inputT>(b2Local);
    }

    __aicore__ inline void SplitV()
    {
        AscendC::LocalTensor<inputT> b1Local = inQueueA1.DeQue<inputT>();
        AscendC::LocalTensor<inputT> b2Local = inQueueB2.AllocTensor<inputT>();

        uint32_t dstOffset = CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        // Nz -> Zn
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = chunk_n / CUBE_BLOCK;
        loadDataParams.srcStride = chunk_m / CUBE_BLOCK;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        for (int i = 0; i < chunk_m / CUBE_BLOCK; ++i)
            LoadData(
                b2Local[i * dstOffset * (chunk_n / CUBE_BLOCK)], b1Local[i * srcOffset + queue_offset_V], loadDataParams);

        inQueueA1.EnQue<inputT>(b1Local);
        inQueueB2.EnQue<inputT>(b2Local);
    }

    __aicore__ inline void SplitVTB() 
    {
        AscendC::LocalTensor<inputT> b1Local = inQueueA1.DeQue<inputT>();
        AscendC::LocalTensor<inputT> b2Local = inQueueB2.AllocTensor<inputT>();

        uint32_t dstOffset = CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        // Nz -> Nz
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = (chunk_n / CUBE_BLOCK) * (chunk_m / CUBE_BLOCK);
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        AscendC::LoadData(b2Local, b1Local[queue_offset_V], loadDataParams);

        inQueueA1.EnQue<inputT>(b1Local);
        inQueueB2.EnQue<inputT>(b2Local);
    }

    __aicore__ inline void SplitMqQT() 
    {
        AscendC::LocalTensor<inputT> a1Local = inQueueFP.DeQue<inputT>();
        AscendC::LocalTensor<inputT> a2Local = inQueueA2.AllocTensor<inputT>();

        uint32_t dstOffset = CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        // Nz -> Zz
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = (chunk_n / CUBE_BLOCK) * (chunk_m / CUBE_BLOCK);
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;

        SetWaitFlagFIXToMte1();
        AscendC::LoadData(a2Local, a1Local[queue_offset_FP], loadDataParams);

        inQueueA2.EnQue<inputT>(a2Local);
        inQueueFP.FreeTensor(a1Local);
    }

    __aicore__ inline void SplitTMPintraOVM(bool reuse) 
    {
        AscendC::LocalTensor<inputT> tmp1Local = inQueueTMP.DeQue<inputT>();
        AscendC::LocalTensor<inputT> tmp2Local = inQueueA2.AllocTensor<inputT>();

        uint32_t dstOffset = CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        // Nz -> Zz
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = chunk_m / CUBE_BLOCK;
        loadDataParams.srcStride = chunk_m / CUBE_BLOCK;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;

        for (int i = 0; i < chunk_m / CUBE_BLOCK; ++i)
            LoadData(tmp2Local[(chunk_m / CUBE_BLOCK) * dstOffset * i],
                     tmp1Local[srcOffset * i + queue_offset_tmp],
                     loadDataParams);

        inQueueA2.EnQue<inputT>(tmp2Local);

        // reuse only for process all
        if (reuse)
            inQueueTMP.EnQue<inputT>(tmp1Local);
        else
            inQueueTMP.FreeTensor(tmp1Local);
    }

    __aicore__ inline void SplitTMPintraOVMT()
    {
        AscendC::LocalTensor<inputT> tmp1Local = inQueueTMP.DeQue<inputT>();
        AscendC::LocalTensor<inputT> tmp2Local = inQueueA2.AllocTensor<inputT>();

        uint32_t dstOffset = CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        // Nz -> Zz
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = (chunk_m / CUBE_BLOCK) * (chunk_m / CUBE_BLOCK);
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;

        AscendC::LoadData(tmp2Local, tmp1Local[queue_offset_tmp], loadDataParams);
        inQueueA2.EnQue<inputT>(tmp2Local);
        inQueueTMP.FreeTensor(tmp1Local);
    }

    __aicore__ inline void SplitTMPintraQKMT()
    {
        AscendC::LocalTensor<inputT> tmp1Local = inQueueTMP.DeQue<inputT>();
        AscendC::LocalTensor<inputT> tmp2Local = inQueueA2.AllocTensor<inputT>();

        uint32_t dstOffset = CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        // Nz -> Zz
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = (chunk_m / CUBE_BLOCK) * (chunk_m / CUBE_BLOCK);
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;

        AscendC::LoadData(tmp2Local, tmp1Local[queue_offset_tmp], loadDataParams);
        inQueueA2.EnQue<inputT>(tmp2Local);
        inQueueTMP.FreeTensor(tmp1Local);
    }

    __aicore__ inline void SplitKVforwardT()
    {
        AscendC::LocalTensor<inputT> KVforward1Local = inQueueTMP2.DeQue<inputT>();
        AscendC::LocalTensor<inputT> KVforward2Local = inQueueB2.AllocTensor<inputT>();

        uint32_t dstOffset = CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        // Nz -> Zn
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = (chunk_n / CUBE_BLOCK) * (chunk_n / CUBE_BLOCK);
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;

        AscendC::LoadData(KVforward2Local, KVforward1Local[queue_offset_tmp2], loadDataParams);
        inQueueB2.EnQue<inputT>(KVforward2Local);
        inQueueTMP2.FreeTensor(KVforward1Local);
    }

    __aicore__ inline void SplitMkV()
    {
        AscendC::LocalTensor<inputT> a1Local = inQueueFP.DeQue<inputT>();
        AscendC::LocalTensor<inputT> a2Local = inQueueA2.AllocTensor<inputT>();

        uint32_t dstOffset = CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        // Nz -> Zz
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = chunk_n / CUBE_BLOCK;
        loadDataParams.srcStride = chunk_m / CUBE_BLOCK;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;

        SetWaitFlagFIXToMte1();
        for (int i = 0; i < chunk_m / CUBE_BLOCK; ++i)
            LoadData(a2Local[(chunk_n / CUBE_BLOCK) * dstOffset * i],
                     a1Local[srcOffset * i + queue_offset_FP],
                     loadDataParams);

        inQueueA2.EnQue<inputT>(a2Local);
        inQueueFP.FreeTensor(a1Local);
    }

    __aicore__ inline void SplitKA()
    {
        AscendC::LocalTensor<inputT> k1Local = inQueueA1.DeQue<inputT>();
        AscendC::LocalTensor<inputT> k2Local = inQueueA2.AllocTensor<inputT>();

        uint32_t dstOffset = CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        // Nz -> Zz
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = chunk_n / CUBE_BLOCK;
        loadDataParams.srcStride = chunk_m / CUBE_BLOCK;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;

        for (int i = 0; i < chunk_m / CUBE_BLOCK; ++i)
            LoadData(
                k2Local[(chunk_n / CUBE_BLOCK) * dstOffset * i], k1Local[srcOffset * i + queue_offset_K], loadDataParams);

        inQueueA2.EnQue<inputT>(k2Local);
        inQueueA1.EnQue<inputT>(k1Local);
    }

    __aicore__ inline void SplitQTB()
    {
        AscendC::LocalTensor<inputT> QT1Local = inQueueA1.DeQue<inputT>();
        AscendC::LocalTensor<inputT> QT2Local = inQueueB2.AllocTensor<inputT>();

        uint32_t dstOffset = CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        // Nz -> Zn
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = (chunk_n / CUBE_BLOCK) * (chunk_m / CUBE_BLOCK);
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        AscendC::LoadData(QT2Local, QT1Local[queue_offset_Q], loadDataParams);

        inQueueA1.EnQue<inputT>(QT1Local);
        inQueueB2.EnQue<inputT>(QT2Local);
    }

    __aicore__ inline void SplitKTMk()
    {
        LocalTensor<inputT> a1Local = inQueueFP.DeQue<inputT>();
        LocalTensor<inputT> a2Local = inQueueA2.AllocTensor<inputT>();

        uint32_t dstOffset = CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;

        // Nz -> Zz
        AscendC::LoadData2DParams nz2zzChunkNMparams;
        nz2zzChunkNMparams.repeatTimes = (chunk_m / CUBE_BLOCK);
        nz2zzChunkNMparams.srcStride = (chunk_n / CUBE_BLOCK);
        nz2zzChunkNMparams.dstGap = 0;
        nz2zzChunkNMparams.ifTranspose = false;

        SetWaitFlagFIXToMte1();

        for (int i = 0; i < chunk_n / CUBE_BLOCK; ++i)
            LoadData(a2Local[(chunk_m / CUBE_BLOCK) * dstOffset * i],
                     a1Local[srcOffset * i + queue_offset_FP],
                     nz2zzChunkNMparams);

        inQueueA2.EnQue<inputT>(a2Local);
        inQueueFP.FreeTensor(a1Local);
    }

    __aicore__ inline void ComputedOVT()
    {
        AscendC::LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();
        SplitdOA();
        SplitVTB();

        AscendC::LocalTensor<inputT> dOLocal = inQueueA2.DeQue<inputT>();
        AscendC::LocalTensor<inputT> vTLocal = inQueueB2.DeQue<inputT>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = chunk_m;
        mmadParams.n = chunk_m;
        mmadParams.k = chunk_n;
        mmadParams.cmatrixInitVal = true;

        AscendC::Mmad(c1Local, dOLocal, vTLocal, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);

        inQueueA2.FreeTensor(dOLocal);
        inQueueB2.FreeTensor(vTLocal);
    }

    __aicore__ inline void ComputeQKT()
    {
        AscendC::LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();
        SplitQA();
        SplitKTB();

        AscendC::LocalTensor<inputT> qLocal = inQueueA2.DeQue<inputT>();
        AscendC::LocalTensor<inputT> kTLocal = inQueueB2.DeQue<inputT>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = chunk_m;
        mmadParams.n = chunk_m;
        mmadParams.k = chunk_n;
        mmadParams.cmatrixInitVal = true;

        AscendC::Mmad(c1Local, qLocal, kTLocal, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);

        inQueueA2.FreeTensor(qLocal);
        inQueueB2.FreeTensor(kTLocal);
    }

    __aicore__ inline void ComputeInterQ(uint64_t chunk_index)
    {
        AscendC::LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();
        SplitMq();
        SplitdOB();
        AscendC::LocalTensor<inputT> mqLocal = inQueueA2.DeQue<inputT>();
        AscendC::LocalTensor<inputT> dOLocal = inQueueB2.DeQue<inputT>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = chunk_m;
        mmadParams.n = chunk_n;
        mmadParams.k = chunk_m;
        mmadParams.cmatrixInitVal = true;

        AscendC::Mmad(c1Local, mqLocal, dOLocal, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);

        inQueueA2.FreeTensor(mqLocal);
        inQueueB2.FreeTensor(dOLocal);

        CopyOutMqdO();

        if (!is_independent)
        {
            CopyInKVforward();
            SplitKVforwardT();
        }
        else
        {
            SplitKVT(true, chunk_index);
        }
        SplitMqdO();

        AscendC::LocalTensor<inputT> mqdOLocal = inQueueA2.DeQue<inputT>();
        AscendC::LocalTensor<inputT> kvLocal = inQueueB2.DeQue<inputT>();

        mmadParams.m = chunk_m;
        mmadParams.n = chunk_n;
        mmadParams.k = chunk_n;
        
        c1Local = outQueueCO1.AllocTensor<float>();
        
        AscendC::Mmad(c1Local, mqdOLocal, kvLocal, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);

        inQueueA2.FreeTensor(mqdOLocal);
        inQueueB2.FreeTensor(kvLocal);
    }

    __aicore__ inline void ComputeIntraQ(bool cmatrixInitVal, bool reuse)
    {
        AscendC::LocalTensor<float> c1Local;
        if (cmatrixInitVal)
            c1Local = outQueueCO1.AllocTensor<float>();
        else
            c1Local = outQueueCO1.DeQue<float>();
        SplitTMPintraOVM(reuse);
        SplitKB();

        AscendC::LocalTensor<inputT> tmpLocal = inQueueA2.DeQue<inputT>();
        AscendC::LocalTensor<inputT> kLocal = inQueueB2.DeQue<inputT>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = chunk_m;
        mmadParams.n = chunk_n;
        mmadParams.k = chunk_m;
        mmadParams.cmatrixInitVal = cmatrixInitVal;

        AscendC::Mmad(c1Local, tmpLocal, kLocal, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);

        inQueueA2.FreeTensor(tmpLocal);
        inQueueB2.FreeTensor(kLocal);
    }

    __aicore__ inline void ComputeInterK(uint64_t chunk_index)
    {
        AscendC::LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();
        SplitMk();
        SplitV();
        AscendC::LocalTensor<inputT> mkLocal = inQueueA2.DeQue<inputT>();
        AscendC::LocalTensor<inputT> vLocal = inQueueB2.DeQue<inputT>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = chunk_m;
        mmadParams.n = chunk_n;
        mmadParams.k = chunk_m;
        mmadParams.cmatrixInitVal = true;

        AscendC::Mmad(c1Local, mkLocal, vLocal, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);

        inQueueA2.FreeTensor(mkLocal);
        inQueueB2.FreeTensor(vLocal);

        CopyOutMkV();

        SplitKVT(true, chunk_index);
        SplitMkV();

        AscendC::LocalTensor<inputT> mkvLocal = inQueueA2.DeQue<inputT>();
        AscendC::LocalTensor<inputT> kvLocal = inQueueB2.DeQue<inputT>();

        mmadParams.m = chunk_m;
        mmadParams.n = chunk_n;
        mmadParams.k = chunk_n;

	    c1Local = outQueueCO1.AllocTensor<float>();

        AscendC::Mmad(c1Local, mkvLocal, kvLocal, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);

        inQueueA2.FreeTensor(mkvLocal);
        inQueueB2.FreeTensor(kvLocal);
    }

    __aicore__ inline void ComputeIntraK(bool cmatrixInitVal)
    {
        AscendC::LocalTensor<float> c1Local;
        if (cmatrixInitVal)
            c1Local = outQueueCO1.AllocTensor<float>();
        else
            c1Local = outQueueCO1.DeQue<float>();      
        SplitTMPintraOVMT();
        SplitQB();

        AscendC::LocalTensor<inputT> tmpLocal = inQueueA2.DeQue<inputT>();
        AscendC::LocalTensor<inputT> qLocal = inQueueB2.DeQue<inputT>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = chunk_m;
        mmadParams.n = chunk_n;
        mmadParams.k = chunk_m;
        mmadParams.cmatrixInitVal = cmatrixInitVal;

        AscendC::Mmad(c1Local, tmpLocal, qLocal, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);
        inQueueA2.FreeTensor(tmpLocal);
        inQueueB2.FreeTensor(qLocal);
    }

    __aicore__ inline void ComputeInterV(uint64_t chunk_index)
    {
        AscendC::LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();
        SplitMk();
        SplitKB();
        AscendC::LocalTensor<inputT> mkLocal = inQueueA2.DeQue<inputT>();
        AscendC::LocalTensor<inputT> kLocal = inQueueB2.DeQue<inputT>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = chunk_m;
        mmadParams.n = chunk_n;
        mmadParams.k = chunk_m;
        mmadParams.cmatrixInitVal = true;

        AscendC::Mmad(c1Local, mkLocal, kLocal, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);
        inQueueA2.FreeTensor(mkLocal);
        inQueueB2.FreeTensor(kLocal);

        CopyOutMkV();

        SplitKV(true, chunk_index);
        SplitMkV();

        AscendC::LocalTensor<inputT> mkkLocal = inQueueA2.DeQue<inputT>();
        AscendC::LocalTensor<inputT> kvLocal = inQueueB2.DeQue<inputT>();

        mmadParams.m = chunk_m;
        mmadParams.n = chunk_n;
        mmadParams.k = chunk_n;

	    c1Local = outQueueCO1.AllocTensor<float>();

        AscendC::Mmad(c1Local, mkkLocal, kvLocal, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);

        inQueueA2.FreeTensor(mkkLocal);
        inQueueB2.FreeTensor(kvLocal);
    }

    __aicore__ inline void ComputeIntraV(bool cmatrixInitVal)
    {
        AscendC::LocalTensor<float> c1Local;
        if (cmatrixInitVal)
        {
            c1Local = outQueueCO1.AllocTensor<float>();
        }
        else
        {
            c1Local = outQueueCO1.DeQue<float>();
        }
        SplitTMPintraQKMT();
        SplitdOB();

        AscendC::LocalTensor<inputT> tmpLocal = inQueueA2.DeQue<inputT>();
        AscendC::LocalTensor<inputT> dOLocal = inQueueB2.DeQue<inputT>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = chunk_m;
        mmadParams.n = chunk_n;
        mmadParams.k = chunk_m;
        mmadParams.cmatrixInitVal = cmatrixInitVal;

        AscendC::Mmad(c1Local, tmpLocal, dOLocal, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);
        inQueueA2.FreeTensor(tmpLocal);
        inQueueB2.FreeTensor(dOLocal);
    }

    __aicore__ inline void ComputedKV(bool first_chunk, uint64_t chunk_index)
    {
        SplitMq();
        SplitQB();
        AscendC::LocalTensor<inputT> mqLocal = inQueueA2.DeQue<inputT>();
        AscendC::LocalTensor<inputT> qLocal = inQueueB2.DeQue<inputT>();

        AscendC::LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = chunk_m;
        mmadParams.n = chunk_n;
        mmadParams.k = chunk_m;

        AscendC::Mmad(c1Local, mqLocal, qLocal, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);

        inQueueA2.FreeTensor(mqLocal);
        inQueueB2.FreeTensor(qLocal);

        CopyOutMqQ();

        SplitMqQT();
        SplitdOB();

        AscendC::LocalTensor<inputT> MqQTLocal = inQueueA2.DeQue<inputT>();
        AscendC::LocalTensor<inputT> dOLocal = inQueueB2.DeQue<inputT>();

        mmadParams.m = chunk_n;
        mmadParams.n = chunk_n;
        mmadParams.k = chunk_m;

	    c1Local = outQueueCO1.AllocTensor<float>();

        AscendC::Mmad(c1Local, MqQTLocal, dOLocal, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);

        inQueueA2.FreeTensor(MqQTLocal);
        inQueueB2.FreeTensor(dOLocal);

        if (first_chunk)
        {
            return;
        }

        SplitKV(true, chunk_index);
        SplitMkv();

        AscendC::LocalTensor<inputT> mkvLocal = inQueueA2.DeQue<inputT>();
        AscendC::LocalTensor<inputT> kvLocal = inQueueB2.DeQue<inputT>();

        mmadParams.m = chunk_n;
        mmadParams.n = chunk_n;
        mmadParams.k = chunk_n;
        mmadParams.cmatrixInitVal = false;

	    c1Local = outQueueCO1.DeQue<float>();

        AscendC::Mmad(c1Local, mkvLocal, kvLocal, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);
        inQueueA2.FreeTensor(mkvLocal);
        inQueueB2.FreeTensor(kvLocal);
    }

    __aicore__ inline void ComputeKV(uint64_t chunk_index, bool first_chunk)
    {
        SplitKTA();
        SplitMkB();
        LocalTensor<inputT> k2Local = inQueueA2.DeQue<inputT>();
        LocalTensor<inputT> mkLocal = inQueueB2.DeQue<inputT>();

        LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = chunk_n;
        mmadParams.n = chunk_m;
        mmadParams.k = chunk_m;

        AscendC::Mmad(c1Local, k2Local, mkLocal, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);

        inQueueA2.FreeTensor(k2Local);
        inQueueB2.FreeTensor(mkLocal);

        CopyOutMqQ();

        SplitV();
        SplitKTMk();

        LocalTensor<inputT> KTMkLocal = inQueueA2.DeQue<inputT>();
        LocalTensor<inputT> v2Local = inQueueB2.DeQue<inputT>();

        mmadParams.m = chunk_n;
        mmadParams.n = chunk_n;
        mmadParams.k = chunk_m;

	    c1Local = outQueueCO1.AllocTensor<float>();

        AscendC::Mmad(c1Local, KTMkLocal, v2Local, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);

        inQueueA2.FreeTensor(KTMkLocal);
        inQueueB2.FreeTensor(v2Local);

        if (first_chunk)
        {
            return;
        }

        SplitMkv();
        SplitKV(true, chunk_index);

        LocalTensor<inputT> mkvLocal = inQueueA2.DeQue<inputT>();
        LocalTensor<inputT> kvLocal = inQueueB2.DeQue<inputT>();

        mmadParams.m = chunk_n;
        mmadParams.n = chunk_n;
        mmadParams.k = chunk_n;
        mmadParams.cmatrixInitVal = false;

	    c1Local = outQueueCO1.DeQue<float>();

        AscendC::Mmad(c1Local, mkvLocal, kvLocal, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);
        inQueueA2.FreeTensor(mkvLocal);
        inQueueB2.FreeTensor(kvLocal);
    }

    __aicore__ inline void CopyOutKV(uint64_t chunk_index, bool first_chunk)
    {
        LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        LocalTensor<inputT> kvLocal;
        if (first_chunk)
            kvLocal = inQueueKV.AllocTensor<inputT>();
        else
            kvLocal = inQueueKV.DeQue<inputT>();

        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = chunk_n;
        fixpipeParams.mSize = chunk_n;
        fixpipeParams.srcStride = chunk_n;
        fixpipeParams.dstStride = chunk_n;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        fixpipeParams.isChannelSplit = false;
        if constexpr (IsSameType<inputT, half>::value)
            fixpipeParams.quantPre = QuantMode_t::F322F16;
        else
            fixpipeParams.quantPre = QuantMode_t::F322BF16;

        AscendC::Fixpipe(kvLocal[queue_offset_KV + (chunk_index % 2) * chunk_n * chunk_n], c1Local, fixpipeParams);

        outQueueCO1.FreeTensor(c1Local);
        inQueueKV.EnQue<inputT>(kvLocal);
    }

    __aicore__ inline void CopyOutMqQ()
    {
        AscendC::LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        AscendC::LocalTensor<inputT> a1Local = inQueueFP.AllocTensor<inputT>();

        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = chunk_m;
        fixpipeParams.mSize = chunk_n;
        fixpipeParams.srcStride = chunk_n;
        fixpipeParams.dstStride = chunk_n;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        fixpipeParams.isChannelSplit = false;
        if constexpr (IsSameType<inputT, half>::value)
            fixpipeParams.quantPre = QuantMode_t::F322F16;
        else
            fixpipeParams.quantPre = QuantMode_t::F322BF16;

        AscendC::Fixpipe(a1Local[queue_offset_FP], c1Local, fixpipeParams);

        outQueueCO1.FreeTensor(c1Local);
        inQueueFP.EnQue<inputT>(a1Local);
    }

    __aicore__ inline void CopyOutMqdO()
    {
        AscendC::LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        AscendC::LocalTensor<inputT> mqdOLocal = inQueueFP.AllocTensor<inputT>();

        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = chunk_n;
        fixpipeParams.mSize = chunk_m;
        fixpipeParams.srcStride = chunk_m;
        fixpipeParams.dstStride = chunk_m;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        fixpipeParams.isChannelSplit = false;
        if constexpr (IsSameType<inputT, half>::value)
            fixpipeParams.quantPre = QuantMode_t::F322F16;
        else
            fixpipeParams.quantPre = QuantMode_t::F322BF16;

        AscendC::Fixpipe(mqdOLocal[queue_offset_FP], c1Local, fixpipeParams);

        outQueueCO1.FreeTensor(c1Local);
        inQueueFP.EnQue<inputT>(mqdOLocal);
    }

    __aicore__ inline void CopyOutMkV()
    {
        AscendC::LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        AscendC::LocalTensor<inputT> mkvLocal = inQueueFP.AllocTensor<inputT>();

        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = chunk_n;
        fixpipeParams.mSize = chunk_m;
        fixpipeParams.srcStride = chunk_m;
        fixpipeParams.dstStride = chunk_m;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        fixpipeParams.isChannelSplit = false;
        if constexpr (IsSameType<inputT, half>::value)
            fixpipeParams.quantPre = QuantMode_t::F322F16;
        else
            fixpipeParams.quantPre = QuantMode_t::F322BF16;

        AscendC::Fixpipe(mkvLocal[queue_offset_FP],
                         c1Local,
                         fixpipeParams); // here need *TMP2, because in dK calculation in *TMP - stored OVM

        outQueueCO1.FreeTensor(c1Local);
        inQueueFP.EnQue<inputT>(mkvLocal);
    }

    __aicore__ inline void CopyOutQ(bool ahead)
    {
        AscendC::LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();

        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = chunk_n;
        fixpipeParams.mSize = chunk_m_valid;
        fixpipeParams.srcStride = chunk_m;
        fixpipeParams.dstStride = chunk_n;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        if constexpr (IsSameType<outputT, bfloat16_t>::value)
            fixpipeParams.quantPre = QuantMode_t::F322BF16;
        AscendC::Fixpipe(Q_resultGM[offset_batch + offset_ResQ], c1Local, fixpipeParams);
        outQueueCO1.FreeTensor(c1Local);
        if (ahead)
            offset_ResQ += chunk_m * chunk_n;
        else
            offset_ResQ -= chunk_m * chunk_n;
    }

    __aicore__ inline void CopyOutK()
    {
        AscendC::LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();

        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = chunk_n;
        fixpipeParams.mSize = chunk_m_valid;
        fixpipeParams.srcStride = chunk_m;
        fixpipeParams.dstStride = chunk_n;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        if constexpr (IsSameType<outputT, bfloat16_t>::value)
            fixpipeParams.quantPre = QuantMode_t::F322BF16;

        AscendC::Fixpipe(K_resultGM[offset_batch + offset_ResK], c1Local, fixpipeParams);

        outQueueCO1.FreeTensor(c1Local);

        offset_ResK -= chunk_m * chunk_n;
    }

    __aicore__ inline void CopyOutV()
    {
        AscendC::LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();

        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = chunk_n;
        fixpipeParams.mSize = chunk_m_valid;
        fixpipeParams.srcStride = chunk_m;
        fixpipeParams.dstStride = chunk_n;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        if constexpr (IsSameType<outputT, bfloat16_t>::value)
            fixpipeParams.quantPre = QuantMode_t::F322BF16;

        AscendC::Fixpipe(V_resultGM[offset_batch + offset_ResV], c1Local, fixpipeParams);
        outQueueCO1.FreeTensor(c1Local);

        offset_ResV -= chunk_m * chunk_n;
    }

    __aicore__ inline void CopyOutdKV(uint64_t chunk_index)
    {
        AscendC::LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        AscendC::LocalTensor<inputT> kvLocal = inQueueKV.DeQue<inputT>();

        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = chunk_n;
        fixpipeParams.mSize = chunk_n;
        fixpipeParams.srcStride = chunk_n;
        fixpipeParams.dstStride = chunk_n;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        fixpipeParams.isChannelSplit = false;
        if constexpr (IsSameType<inputT, half>::value)
            fixpipeParams.quantPre = QuantMode_t::F322F16;
        else
            fixpipeParams.quantPre = QuantMode_t::F322BF16;

        AscendC::Fixpipe(kvLocal[queue_offset_KV + (chunk_index % 2) * chunk_n * chunk_n], c1Local, fixpipeParams);

        outQueueCO1.FreeTensor(c1Local);
        inQueueKV.EnQue<inputT>(kvLocal);
    }

    __aicore__ inline void CopyOutdOVT()
    {
        AscendC::LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();

        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = chunk_m;
        fixpipeParams.mSize = chunk_m;
        fixpipeParams.srcStride = chunk_m;
        fixpipeParams.dstStride = chunk_m;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        AscendC::Fixpipe(dOVTGM[offset_batch_TMP], c1Local, fixpipeParams);

        outQueueCO1.FreeTensor(c1Local);
    }

    __aicore__ inline void CopyOutQKT()
    {
        AscendC::LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();

        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = chunk_m;
        fixpipeParams.mSize = chunk_m;
        fixpipeParams.srcStride = chunk_m;
        fixpipeParams.dstStride = chunk_m;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        AscendC::Fixpipe(QKTGM[offset_batch_TMP], c1Local, fixpipeParams);

        outQueueCO1.FreeTensor(c1Local);
    }

private:
    AscendC::TPipe* pipe;

    TQue<TPosition::A1, 1> inQueueA1;
    TQue<TPosition::A1, 1> inQueueFP;
    TQue<TPosition::A1, 1> inQueueKV;
    TQue<TPosition::A1, 1> inQueueTMP;
    TQue<TPosition::A1, 1> inQueueTMP2;
    TQue<TPosition::A1, 1> inQueueDec;
    TQue<TPosition::A2, 1> inQueueA2;
    TQue<TPosition::B2, 1> inQueueB2;
    TQue<TPosition::CO1, 1> outQueueCO1;

    TQue<TPosition::VECIN, 1> inQueueX, inQueueM;
    TQue<TPosition::VECOUT, 1> outQueueZ;
    TQue<TPosition::VECOUT, 1> outQueueTMP;

    GlobalTensor<inputT> dOGM;
    GlobalTensor<inputT> QGM;
    GlobalTensor<inputT> KGM;
    GlobalTensor<inputT> VGM;

    GlobalTensor<inputT> MqGM;
    GlobalTensor<inputT> MkvGM;
    GlobalTensor<inputT> MkGM;
    GlobalTensor<float> MGM;

    GlobalTensor<inputT> KVforwardGM;

    GlobalTensor<outputT> K_resultGM;
    GlobalTensor<outputT> V_resultGM;
    GlobalTensor<outputT> Q_resultGM;

    GlobalTensor<inputT> OVMGM;
    GlobalTensor<inputT> QKMGM;

    GlobalTensor<float> dOVTGM;
    GlobalTensor<float> QKTGM;

    uint64_t m, n, chunk_m, chunk_n, chunk_m_valid,  num_chunks, batch, head;

    uint64_t offset_dO, offset_Q, offset_K, offset_V;
    uint64_t offset_batch, offset_Mq, offset_Mkv, offset_Mk,  offset_batch_M, offset_batch_TMP, offsetTMPintra,
        offset_batch_KVf, offset_KVforward;

    uint64_t queue_offset_Mq, queue_offset_Mkv, queue_offset_Mk;
    uint64_t queue_offset_dO, queue_offset_Q, queue_offset_K, queue_offset_V, queue_offset_KV, queue_offset_FP,
        queue_offset_tmp, queue_offset_tmp2;

    uint64_t offset_ResQ, offset_ResK, offset_ResV;

    uint64_t block_id, block_num, block_size, extra_tasks;

    bool has_tail_chunk, is_independent;
    uint64_t inQueueA1SizeItems, inQueueDecSizeItems, inQueueKVSizeItems, inQueueFPSizeItems, inQueueTMPSizeItems,
        inQueueTMP2SizeItems;

    uint64_t squadron_Q, squadron_KV, squad_Q_size, squad_KV_size, squad_Q_idx, squad_KV_idx, block_Q_size, block_KV_size,
        extra_Q_tasks, extra_KV_tasks;
};

} // namespace LightningAttn::Kernel

