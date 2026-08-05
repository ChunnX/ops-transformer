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
template <typename inputT, typename outputT> class LaForwardCMM128
{
public:
    static constexpr uint32_t CUBE_BLOCK = 16;
    static constexpr uint32_t CUBE_BLOCK_SIZE = 16 * 16;
    static constexpr uint64_t ALIGNER_L1 = 512l;

    __aicore__ inline LaForwardCMM128() {}

    __aicore__ inline void Init(GM_ADDR Q,
                                GM_ADDR K,
                                GM_ADDR V,
                                GM_ADDR slopeRate,
                                GM_ADDR resultOut,
                                GM_ADDR kvStateOut,
                                GM_ADDR workspace,
                                Common::LoCmm128TilingData* tilingStruct,
                                TPipe* pipe)
    {
        InitKernelParams(tilingStruct);
        InitOffsets();
        AllocateGlobalBuffers(Q, K, V, resultOut, kvStateOut, workspace);
        FillInDecayMatrices(slopeRate, pipe);        
        InitQueueBuffers(pipe);
    }

    __aicore__ inline void Process()
    {
        auto flag1 = 0x1;
        auto flag2 = 0x2;
        for (int batch_idx = 0; batch_idx < block_size; ++batch_idx)
        {
            ProcessCubeCalculations(flag1, flag2);
            ProcessVectorCalculations(flag1, flag2);            
            RecalculateGMOffsets();
        } 
    }

private:
    __aicore__ inline uint32_t CeilCubeBlock(uint32_t len)
    {
        return (len + CUBE_BLOCK - 1) / CUBE_BLOCK;
    }

    __aicore__ inline void SetWaitFlagMte1ToM()
    {
        event_t eventIdMte1ToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_M));
        SetFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
        WaitFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
        PipeBarrier<PIPE_M>();
    }
    __aicore__ inline void SetWaitFlagMte1ToFIX()
    {
        event_t eventIdMte1ToFIX = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_FIX));
        SetFlag<HardEvent::MTE1_FIX>(eventIdMte1ToFIX);
        WaitFlag<HardEvent::MTE1_FIX>(eventIdMte1ToFIX);
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


    __aicore__ inline void InitKernelParams(Common::LoCmm128TilingData* tilingStruct)
    {
        blockIdx = GetBlockIdx();
        block_num = GetBlockNum();

        m = tilingStruct->seqLen;
        n = tilingStruct->dimSize;
        is_training = tilingStruct->isTraining;
        numHeads = tilingStruct->numHeads;
        batch = tilingStruct->batchSize * tilingStruct->numHeads;
        chunk_m = tilingStruct->blockSize;
        chunk_n = n;
        num_chunks = (m + chunk_m - 1) / chunk_m;
        if (m % chunk_m == 0)
            has_tail_chunk = false;
        else
            has_tail_chunk = true;
    }

    __aicore__ inline void InitOffsets()
    {        
        InitQueueOffsets();
        InitGMOffsets();
    }

    __aicore__ inline void InitGMOffsets()
    { 
        offset_K = 0;
        offset_Q = 0;
        offset_V = 0;
        offset_Res = 0;
        offset_Mq = 0;
        offset_Mkv = 0;
        offset_Mk = 0;
        offset_batch = 0;
        offset_DumpKV = chunk_n * chunk_n;
        offset_ResIntra = 0;
        offset_batch_TMP = 0;
        offset_batch_M = 0;
        offset_batch_KV = 0;
        
        block_size = batch / block_num;
        extra_tasks = batch % block_num;
        if ASCEND_IS_AIC
        {
            offset_batch_KV = blockIdx * num_chunks * n * n;
            offset_batch = blockIdx * m * n;
            offset_batch_TMP = blockIdx * chunk_m * chunk_m;
            offset_batch_M = (blockIdx * chunk_m * chunk_m) % (numHeads * chunk_m * chunk_m);
            offset_Mq = (blockIdx * chunk_m * chunk_m) % (numHeads * chunk_m * chunk_m);
            offset_Mkv = (blockIdx * chunk_n * chunk_n) % (numHeads * chunk_n * chunk_n);
            offset_Mk = (blockIdx * chunk_m * chunk_m) % (numHeads * chunk_m * chunk_m);
            if (blockIdx < extra_tasks)
                block_size++;
        }
        if ASCEND_IS_AIV
        {
            offset_batch = (blockIdx / 2) * m * n;
            offset_batch_TMP = (blockIdx / 2) * chunk_m * chunk_m;
            offset_batch_M = ((blockIdx / 2) * chunk_m * chunk_m) % (numHeads * chunk_m * chunk_m);
            if (blockIdx / 2 < extra_tasks)
                block_size++;
        }
    }

     __aicore__ inline void InitQueueOffsets()
    {  
        
        uint64_t chunk_max = chunk_m > chunk_n ? chunk_m : chunk_n;        
        dst_offset = CUBE_BLOCK_SIZE;
        src_offset = CUBE_BLOCK_SIZE;
        queue_offset_Q = 0;
        queue_offset_K = queue_offset_Q + AlignItemsByBytes(chunk_m * chunk_n, sizeof(inputT), ALIGNER_L1);
        queue_offset_V = queue_offset_K + AlignItemsByBytes(chunk_m * chunk_n, sizeof(inputT), ALIGNER_L1);
        queue_offset_TMP = queue_offset_V + AlignItemsByBytes(chunk_m * chunk_n, sizeof(inputT), ALIGNER_L1);
        queue_offset_Mq = 0;
        queue_offset_Mk = queue_offset_Mq + AlignItemsByBytes(chunk_m * chunk_m, sizeof(inputT), ALIGNER_L1);
        queue_offset_Mkv = queue_offset_Mk + AlignItemsByBytes(chunk_m * chunk_m, sizeof(inputT), ALIGNER_L1);
        queue_offset_FP = 0;
        queue_offset_KV = 0;
        inQueueA1SizeItems = queue_offset_TMP + AlignItemsByBytes(chunk_m * chunk_max, sizeof(inputT), ALIGNER_L1);
        inQueueDecSizeItems = queue_offset_Mkv + AlignItemsByBytes(chunk_n * chunk_n, sizeof(inputT), ALIGNER_L1);
        inQueueFPSizeItems = queue_offset_FP + AlignItemsByBytes(chunk_max * chunk_max, sizeof(inputT), ALIGNER_L1);
        inQueueKVSizeItems = queue_offset_KV + AlignItemsByBytes(2 * chunk_n * chunk_n, sizeof(inputT), ALIGNER_L1);
        QueueL0SizeItems = AlignItemsByBytes(chunk_max * chunk_max, sizeof(inputT), ALIGNER_L1);
    }

    __aicore__ inline void AllocateGlobalBuffers(GM_ADDR Q,
                                GM_ADDR K,
                                GM_ADDR V,
                                GM_ADDR resultOut,
                                GM_ADDR kvStateOut,
                                GM_ADDR workspace)
    {
        QGM.SetGlobalBuffer((__gm__ inputT*)Q, batch * m * n);
        KGM.SetGlobalBuffer((__gm__ inputT*)K, batch * m * n);
        VGM.SetGlobalBuffer((__gm__ inputT*)V, batch * m * n);   
        resultGM.SetGlobalBuffer((__gm__ outputT*)resultOut, batch * m * n);
	    if (is_training)
            DumpKVGM.SetGlobalBuffer((__gm__ inputT*)kvStateOut, batch * num_chunks * n * n);

        int64_t qkDecaySize = numHeads * chunk_m * chunk_m * sizeof(inputT);
        qkDecaySize = CeilDiv(qkDecaySize, 512l) * 512l / sizeof(inputT);
        int64_t kvDecaySize = numHeads * chunk_n * chunk_n * sizeof(inputT);
        kvDecaySize = CeilDiv(kvDecaySize, 512l) * 512l / sizeof(inputT);
        int64_t diagDecaySize = numHeads * chunk_m * chunk_m * sizeof(float);
        diagDecaySize = CeilDiv(diagDecaySize, 512l) * 512l / sizeof(float);
        int64_t qkGMSize = batch * chunk_m * chunk_m * sizeof(float);
        qkGMSize = CeilDiv(qkGMSize, 512l) * 512l / sizeof(float);
        int64_t qkmGMSize = batch * chunk_m * chunk_m * sizeof(inputT);
        qkmGMSize = CeilDiv(qkmGMSize, 512l) * 512l / sizeof(inputT);

        MqGM.SetGlobalBuffer((__gm__ inputT*)workspace, qkDecaySize);
        workspace = (GM_ADDR)((__gm__ inputT*)workspace + qkDecaySize);
        MkGM.SetGlobalBuffer((__gm__ inputT*)workspace, qkDecaySize);
        workspace = (GM_ADDR)((__gm__ inputT*)workspace + qkDecaySize);
        MkvGM.SetGlobalBuffer((__gm__ inputT*)workspace, kvDecaySize);
        workspace = (GM_ADDR)((__gm__ inputT*)workspace + kvDecaySize);
        MGM.SetGlobalBuffer((__gm__ float*)workspace, diagDecaySize);
        workspace = (GM_ADDR)((__gm__ float*)workspace + diagDecaySize);     

        QKGM.SetGlobalBuffer((__gm__ float*)workspace, qkGMSize);
        workspace = (GM_ADDR)((__gm__ float*)workspace + qkGMSize);
        QKMGM.SetGlobalBuffer((__gm__ inputT*)workspace, qkmGMSize);
        workspace = (GM_ADDR)((__gm__ inputT*)workspace + qkmGMSize);
    }

    __aicore__ inline void FillInDecayMatrices(GM_ADDR slopeRate,
                                TPipe* pipe)
    {
        if ASCEND_IS_AIV
        {
            GlobalTensor<float> slopeGM;
            slopeGM.SetGlobalBuffer((__gm__ float*)slopeRate);

            LightningAttn::Kernel::CmmDecayGenerator<float> cmgOp;
            cmgOp.Init(numHeads, chunk_n, chunk_m, pipe);
            cmgOp.Generate(MqGM, MkGM, MkvGM, slopeGM);
            pipe->Reset();

            LightningAttn::Kernel::DiagDecayGenerator<float> ddgOp;
            ddgOp.Init(pipe, numHeads, chunk_m);
            ddgOp.GenerateDiagDecay(MGM, slopeGM);
            pipe->Reset();
        }
    }

    __aicore__ inline void InitQueueBuffers(TPipe* pipe)
    {
        AscendC::SyncAll<false>();
        if ASCEND_IS_AIC
        {
            pipe->InitBuffer(inQueueA1, 1, inQueueA1SizeItems * sizeof(inputT));
            pipe->InitBuffer(inQueueDec, 1, inQueueDecSizeItems * sizeof(inputT));
            pipe->InitBuffer(inQueueFP, 1, inQueueFPSizeItems * sizeof(inputT));
            pipe->InitBuffer(inQueueKV, 1, inQueueKVSizeItems * sizeof(inputT));
            pipe->InitBuffer(inQueueA2, 2, QueueL0SizeItems * sizeof(inputT));
            pipe->InitBuffer(inQueueB2, 2, QueueL0SizeItems * sizeof(inputT));
            pipe->InitBuffer(outQueueCO1, 2, QueueL0SizeItems * sizeof(float));
            InitConvertingParams();
        }
        if ASCEND_IS_AIV
        {
            pipe->InitBuffer(inQueueM, 1, (chunk_m * chunk_m) / 2 * sizeof(float));
            pipe->InitBuffer(inQueueX, 1, (chunk_m * chunk_m) / 2 * sizeof(float));
            pipe->InitBuffer(outQueueZ, 1, (chunk_m * chunk_m) / 2 * sizeof(float));
            pipe->InitBuffer(outQueueTMP, 1, (chunk_m * chunk_m) / 2 * sizeof(inputT));
        }
    }

     __aicore__ inline void ProcessCubeCalculations(uint16_t flag1, uint16_t flag2)
    {
        if ASCEND_IS_AIC
        {
            LocalTensor<inputT> decLocal = inQueueDec.AllocTensor<inputT>();
            inQueueDec.EnQue(decLocal);

            CopyInMq();
            CopyInMkv();
            CopyInMk();

            uint64_t i = 0;
            while (i < num_chunks)
            {
                LocalTensor<inputT> a1Local = inQueueA1.AllocTensor<inputT>();
                inQueueA1.EnQue(a1Local);
                if (i == num_chunks - 1 && has_tail_chunk)
                    UpdateValidParams();

                CopyInQ(queue_offset_Q);
                CopyInK(queue_offset_K);
                ComputeQK(queue_offset_Q, queue_offset_K);
                CopyOutQK(&QKGM);
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(flag1);
                if (i > 0)
                    ProcessNotFirstChunkCubeCalculations(i, flag2);
                else
                    ProcessFirstChunkCubeCalculations(i, flag2);
                i++;
                a1Local = inQueueA1.DeQue<inputT>();
                inQueueA1.FreeTensor(a1Local);
            }
            ReturnValidParams();
            if (num_chunks > 1)
            {
                LocalTensor<inputT> kvLocal = inQueueKV.DeQue<inputT>();
                inQueueKV.FreeTensor(kvLocal);
            }
            decLocal = inQueueDec.DeQue<inputT>();
            inQueueDec.FreeTensor(decLocal);
        } // AIC
    }

    __aicore__ inline void ProcessFirstChunkCubeCalculations(uint64_t i, uint16_t flag2)
    {
        CopyInV(queue_offset_V);
        if (num_chunks > 1)
        {
            ComputeKV(queue_offset_K, queue_offset_V, i, true);
            CopyOutKV((i + 1) % 2, true);
        }
        AscendC::CrossCoreWaitFlag(flag2);
        CopyInTMPintra(&QKMGM);
        ComputeIntra(true, queue_offset_V);
        CopyOut();
    }

    __aicore__ inline void ProcessNotFirstChunkCubeCalculations(uint64_t i, uint16_t flag2)
    {
        CopyInV(queue_offset_V);
        if (i < num_chunks - 1)
        {
            ComputeKV(queue_offset_K, queue_offset_V, i, false);
            CopyOutKV((i + 1) % 2, false);
        }

        ComputeInter(queue_offset_Q, i);
        AscendC::CrossCoreWaitFlag(flag2);
        CopyInTMPintra(&QKMGM);
        ComputeIntra(false, queue_offset_V);
        CopyOut();
    }
    
    __aicore__ inline void ProcessVectorCalculations(uint16_t flag1, uint16_t flag2)
    {
        if ASCEND_IS_AIV
        {
            AdamarCopyInM();
            LocalTensor<outputT> mLocal = inQueueM.DeQue<outputT>();
            uint32_t i = 0;
            while (i < num_chunks)
            {
                AscendC::CrossCoreWaitFlag(flag1);
                AdamarCopyIn(&QKGM);
                AdamarCompute(&mLocal);
                AdamarCopyOut(&QKMGM);
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(flag2);
                i++;
            }
            inQueueM.FreeTensor(mLocal);
            offset_batch_M += block_num * chunk_m * chunk_m;
        } // AIV
    }
    
    __aicore__ inline void RecalculateGMOffsets()
    {        
        offset_batch += block_num * m * n;
        offset_ResIntra = 0;
        offset_K = 0;
        offset_Q = 0;
        offset_V = 0;
        offset_Res = 0;
        offset_Mq += block_num * chunk_m * chunk_m;
        offset_Mkv += block_num * chunk_n * chunk_n;
        offset_Mk += block_num * chunk_m * chunk_m;
        offset_DumpKV = chunk_n * chunk_n;
        offset_batch_KV += block_num * num_chunks * n * n;

        offset_Mq %= numHeads * chunk_m * chunk_m;
        offset_Mkv %= numHeads * chunk_n * chunk_n;
        offset_Mk %= numHeads * chunk_m * chunk_m;
        offset_batch_M %= numHeads * chunk_m * chunk_m;
    }

    __aicore__ inline void AdamarCopyInM()
    {
        LocalTensor<float> mLocal = inQueueM.AllocTensor<float>();
        DataCopy(mLocal, MGM[offset_batch_M + (blockIdx % 2) * (chunk_m * chunk_m / 2)], chunk_m * chunk_m / 2);
        inQueueM.EnQue(mLocal);
    }

    __aicore__ inline void AdamarCopyIn(GlobalTensor<float>* operandGM)
    {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(
            xLocal, (*operandGM)[offset_batch_TMP + (blockIdx % 2) * (chunk_m * chunk_m / 2)], chunk_m * chunk_m / 2);
        inQueueX.EnQue<float>(xLocal);
    }

    __aicore__ inline void AdamarCompute(LocalTensor<float>* mLocal)
    {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        AscendC::Mul(zLocal, xLocal, *mLocal, chunk_m * chunk_m / 2);

        outQueueZ.EnQue<float>(zLocal);
        zLocal = outQueueZ.DeQue<float>();

        LocalTensor<inputT> tmpLocal = outQueueTMP.AllocTensor<inputT>();
        Cast(tmpLocal, zLocal, RoundMode::CAST_ROUND, chunk_m * chunk_m / 2);

        outQueueTMP.EnQue<inputT>(tmpLocal);
        inQueueX.FreeTensor(xLocal);
        outQueueZ.FreeTensor(zLocal);
    }

    __aicore__ inline void AdamarCopyOut(GlobalTensor<inputT>* resultGM)
    {
        LocalTensor<inputT> tmpLocal = outQueueTMP.DeQue<inputT>();

        DataCopy(
            (*resultGM)[offset_batch_TMP + (blockIdx % 2) * (chunk_m * chunk_m / 2)], tmpLocal, chunk_m * chunk_m / 2);

        outQueueTMP.EnQue<inputT>(tmpLocal);
        tmpLocal = outQueueTMP.DeQue<inputT>();
        outQueueTMP.FreeTensor(tmpLocal);
    }

    __aicore__ inline void InitConvertingParams()
    {
        InitSquareMatrixNDNZZNParams();
        InitNonSquareMatrixNDNZZNParams();
        InitFixpipeParams();        
    }

    __aicore__ inline void InitSquareMatrixNDNZZNParams()
    {
        InitMMMatrixParams();
        InitNNMatrixParams();
    }

    __aicore__ inline void InitMMMatrixParams()
    {
        nd2nzChunkMMparams.ndNum = 1;
        nd2nzChunkMMparams.nValue = chunk_m;
        nd2nzChunkMMparams.dValue = chunk_m;
        nd2nzChunkMMparams.srcNdMatrixStride = 0;
        nd2nzChunkMMparams.srcDValue = chunk_m;
        nd2nzChunkMMparams.dstNzC0Stride = CeilCubeBlock(chunk_m) * CUBE_BLOCK;
        nd2nzChunkMMparams.dstNzNStride = 1;
        nd2nzChunkMMparams.dstNzMatrixStride = 0;

        nz2nzChunkMMparams.repeatTimes = (chunk_m / CUBE_BLOCK) * (chunk_m / CUBE_BLOCK);
        nz2nzChunkMMparams.srcStride = 1;
        nz2nzChunkMMparams.dstGap = 0;
        nz2nzChunkMMparams.ifTranspose = false;

        nz2zzChunkMMparams.repeatTimes = chunk_m / CUBE_BLOCK;
        nz2zzChunkMMparams.srcStride = (chunk_m / CUBE_BLOCK);
        nz2zzChunkMMparams.dstGap = 0;
        nz2zzChunkMMparams.ifTranspose = false;
    }

    __aicore__ inline void InitNNMatrixParams()
    {
        nd2nzChunkNNparams.ndNum = 1;
        nd2nzChunkNNparams.nValue = chunk_n;
        nd2nzChunkNNparams.dValue = chunk_n;
        nd2nzChunkNNparams.srcNdMatrixStride = 0;
        nd2nzChunkNNparams.srcDValue = chunk_n;
        nd2nzChunkNNparams.dstNzC0Stride = CeilCubeBlock(chunk_n) * CUBE_BLOCK;
        nd2nzChunkNNparams.dstNzNStride = 1;
        nd2nzChunkNNparams.dstNzMatrixStride = 0;

        nz2nzChunkNNparams.repeatTimes = CeilCubeBlock(chunk_n) * CeilCubeBlock(chunk_n);  
        nz2nzChunkNNparams.srcStride = 1;
        nz2nzChunkNNparams.dstGap = 0;
        nz2nzChunkNNparams.ifTranspose = false;

        nz2zzChunkNNparams.repeatTimes = CeilCubeBlock(chunk_n);
        nz2zzChunkNNparams.srcStride = CeilCubeBlock(chunk_n);
        nz2zzChunkNNparams.dstGap = 0;
        nz2zzChunkNNparams.ifTranspose = false;

        nz2znChunkNNparams.repeatTimes = CeilCubeBlock(chunk_n);
        nz2znChunkNNparams.srcStride = CeilCubeBlock(chunk_n);
        nz2znChunkNNparams.dstGap = 0;
        nz2znChunkNNparams.ifTranspose = true;
    }

    __aicore__ inline void InitNonSquareMatrixNDNZZNParams()
    {
        nd2nzChunkMNparams.ndNum = 1;
        nd2nzChunkMNparams.nValue = chunk_m;
        nd2nzChunkMNparams.dValue = chunk_n;
        nd2nzChunkMNparams.srcNdMatrixStride = 0;
        nd2nzChunkMNparams.srcDValue = chunk_n;
        nd2nzChunkMNparams.dstNzC0Stride = CeilCubeBlock(chunk_m) * CUBE_BLOCK;
        nd2nzChunkMNparams.dstNzNStride = 1;
        nd2nzChunkMNparams.dstNzMatrixStride = 0;

        nz2zzChunkMNparams.repeatTimes = CeilCubeBlock(chunk_n);
        nz2zzChunkMNparams.srcStride = CeilCubeBlock(chunk_m);
        nz2zzChunkMNparams.dstGap = 0;
        nz2zzChunkMNparams.ifTranspose = false;

        nz2zzChunkNMparams.repeatTimes = CeilCubeBlock(chunk_m);
        nz2zzChunkNMparams.srcStride = CeilCubeBlock(chunk_n);
        nz2zzChunkNMparams.dstGap = 0;
        nz2zzChunkNMparams.ifTranspose = false;

        nz2znChunkParams.repeatTimes = CeilCubeBlock(chunk_n);
        nz2znChunkParams.srcStride = (chunk_m / CUBE_BLOCK);
        nz2znChunkParams.dstGap = 0;
        nz2znChunkParams.ifTranspose = true;

        nz2zzChunkMNparamsTranspose.repeatTimes = CeilCubeBlock(chunk_n) * (chunk_m / CUBE_BLOCK);
        nz2zzChunkMNparamsTranspose.srcStride = 1;
        nz2zzChunkMNparamsTranspose.dstGap = 0;
        nz2zzChunkMNparamsTranspose.ifTranspose = true;

        nz2znChunkParamsMNtranspose.repeatTimes = CeilCubeBlock(chunk_n) * (chunk_m / CUBE_BLOCK);
        nz2znChunkParamsMNtranspose.srcStride = 1;
        nz2znChunkParamsMNtranspose.dstGap = 0;
        nz2znChunkParamsMNtranspose.ifTranspose = false;
    }

    __aicore__ inline void InitFixpipeParams()
    {
        InitSquareMatrixFixpipeParams();
        InitNonSquareMatrixFixpipeParams();       
    }

    __aicore__ inline void InitSquareMatrixFixpipeParams()
    {
        fixpipeParamsKV.nSize = chunk_n;
        fixpipeParamsKV.mSize = chunk_n;
        fixpipeParamsKV.srcStride = chunk_n;
        fixpipeParamsKV.dstStride = chunk_n;
        fixpipeParamsKV.ndNum = 1;
        fixpipeParamsKV.srcNdStride = 0;
        fixpipeParamsKV.dstNdStride = 0;
        fixpipeParamsKV.isChannelSplit = false;
        if constexpr (IsSameType<inputT, half>::value)
            fixpipeParamsKV.quantPre = QuantMode_t::F322F16;
        else
            fixpipeParamsKV.quantPre = QuantMode_t::F322BF16;

        fixpipeParamsQK.nSize = chunk_m;
        fixpipeParamsQK.mSize = chunk_m;
        fixpipeParamsQK.srcStride = chunk_m;
        fixpipeParamsQK.dstStride = chunk_m;
        fixpipeParamsQK.ndNum = 1;
        fixpipeParamsQK.srcNdStride = 0;
        fixpipeParamsQK.dstNdStride = 0;
    }

    __aicore__ inline void InitNonSquareMatrixFixpipeParams()
    {
        fixpipeParamsChunkL1.nSize = chunk_n;
        fixpipeParamsChunkL1.mSize = chunk_m;
        fixpipeParamsChunkL1.srcStride = chunk_m;
        fixpipeParamsChunkL1.dstStride = chunk_m;
        fixpipeParamsChunkL1.ndNum = 1;
        fixpipeParamsChunkL1.srcNdStride = 0;
        fixpipeParamsChunkL1.dstNdStride = 0;
        fixpipeParamsChunkL1.isChannelSplit = false;
        if constexpr (IsSameType<inputT, half>::value)
            fixpipeParamsChunkL1.quantPre = QuantMode_t::F322F16;
        else
            fixpipeParamsChunkL1.quantPre = QuantMode_t::F322BF16;

        fixpipeParamsChunkL1T.nSize = chunk_m;
        fixpipeParamsChunkL1T.mSize = chunk_n;
        fixpipeParamsChunkL1T.srcStride = chunk_n;
        fixpipeParamsChunkL1T.dstStride = chunk_n;
        fixpipeParamsChunkL1T.ndNum = 1;
        fixpipeParamsChunkL1T.srcNdStride = 0;
        fixpipeParamsChunkL1T.dstNdStride = 0;
        fixpipeParamsChunkL1T.isChannelSplit = false;
        if constexpr (IsSameType<inputT, half>::value)
            fixpipeParamsChunkL1T.quantPre = QuantMode_t::F322F16;
        else
            fixpipeParamsChunkL1T.quantPre = QuantMode_t::F322BF16;

        fixpipeParamsChunkGM.nSize = chunk_n;
        fixpipeParamsChunkGM.mSize = chunk_m;
        fixpipeParamsChunkGM.srcStride = chunk_m;
        fixpipeParamsChunkGM.dstStride = chunk_n;
        fixpipeParamsChunkGM.ndNum = 1;
        fixpipeParamsChunkGM.srcNdStride = 0;
        fixpipeParamsChunkGM.dstNdStride = 0;
        if constexpr (IsSameType<outputT, bfloat16_t>::value)
            fixpipeParamsChunkGM.quantPre = QuantMode_t::F322BF16;
    }

    __aicore__ inline void UpdateValidParams()
    {
        LocalTensor<inputT> qkvLocal = inQueueA1.DeQue<inputT>();
        inputT zero = (inputT)0.0;
        uint16_t iter_block_size = static_cast<uint16_t>((chunk_m * chunk_n * sizeof(inputT) + 31) / 32);

        AscendC::InitConstValue(qkvLocal[queue_offset_Q], {1, iter_block_size, 0, zero});
        AscendC::InitConstValue(qkvLocal[queue_offset_K], {1, iter_block_size, 0, zero});
        AscendC::InitConstValue(qkvLocal[queue_offset_V], {1, iter_block_size, 0, zero});
        inQueueA1.EnQue(qkvLocal);

        chunk_m_valid = m % chunk_m;
        nd2nzChunkMNparams.nValue = chunk_m_valid;
        fixpipeParamsChunkGM.mSize = chunk_m_valid;
    }

    __aicore__ inline void ReturnValidParams()
    {
        nd2nzChunkMNparams.nValue = chunk_m;
        fixpipeParamsChunkGM.mSize = chunk_m;
    }

    __aicore__ inline void CopyInTMPintra(GlobalTensor<inputT>* operandGM)
    {
        LocalTensor<inputT> tmpLocal = inQueueA1.DeQue<inputT>();
        DataCopy(tmpLocal[queue_offset_TMP], (*operandGM)[offset_batch_TMP], nd2nzChunkMMparams);
        inQueueA1.EnQue(tmpLocal);
    }

    __aicore__ inline void CopyInMq()
    {
        LocalTensor<inputT> MqLocal = inQueueDec.DeQue<inputT>();
        DataCopy(MqLocal[queue_offset_Mq], MqGM[offset_Mq], nd2nzChunkMMparams);
        inQueueDec.EnQue(MqLocal);
    }

    __aicore__ inline void CopyInMkv()
    {
        LocalTensor<inputT> MkvLocal = inQueueDec.DeQue<inputT>();
        DataCopy(MkvLocal[queue_offset_Mkv], MkvGM[offset_Mkv], nd2nzChunkNNparams);
        inQueueDec.EnQue<inputT>(MkvLocal);
    }

    __aicore__ inline void CopyInMk()
    {
        LocalTensor<inputT> MkLocal = inQueueDec.DeQue<inputT>();
        DataCopy(MkLocal[queue_offset_Mk], MkGM[offset_Mk], nd2nzChunkMMparams);
        inQueueDec.EnQue<inputT>(MkLocal);
    }

    __aicore__ inline void CopyInQ(uint64_t queue_offset_Q)
    {
        LocalTensor<inputT> Q1Local = inQueueA1.DeQue<inputT>();
        DataCopy(Q1Local[queue_offset_Q], QGM[offset_batch + offset_Q], nd2nzChunkMNparams);
        inQueueA1.EnQue<inputT>(Q1Local);
        offset_Q += chunk_m * chunk_n;
    }

    __aicore__ inline void CopyInK(uint64_t queue_offset_K)
    {
        LocalTensor<inputT> K1Local = inQueueA1.DeQue<inputT>();
        DataCopy(K1Local[queue_offset_K], KGM[offset_batch + offset_K], nd2nzChunkMNparams);
        inQueueA1.EnQue<inputT>(K1Local);
        offset_K += chunk_m * chunk_n;
    }

    __aicore__ inline void CopyInV(uint64_t queue_offset_V)
    {
        LocalTensor<inputT> V1Local = inQueueA1.DeQue<inputT>();
        DataCopy(V1Local[queue_offset_V], VGM[offset_batch + offset_V], nd2nzChunkMNparams);
        inQueueA1.EnQue<inputT>(V1Local);
        offset_V += chunk_m * chunk_n;
    }

    __aicore__ inline void SplitMq()
    {
        LocalTensor<inputT> a1Local = inQueueDec.DeQue<inputT>();
        LocalTensor<inputT> a2Local = inQueueA2.AllocTensor<inputT>();

        // Nz -> Zz
        LoadData(a2Local, a1Local[queue_offset_Mq], nz2nzChunkMMparams);

        inQueueA2.EnQue<inputT>(a2Local);
        inQueueDec.EnQue<inputT>(a1Local);
    }

    __aicore__ inline void SplitMkv()
    {
        LocalTensor<inputT> a1Local = inQueueDec.DeQue<inputT>();
        LocalTensor<inputT> a2Local = inQueueA2.AllocTensor<inputT>();

        // Nz -> Zz
        LoadData(a2Local, a1Local[queue_offset_Mkv], nz2nzChunkNNparams);

        inQueueA2.EnQue<inputT>(a2Local);
        inQueueDec.EnQue<inputT>(a1Local);
    }

    __aicore__ inline void SplitMk()
    {
        LocalTensor<inputT> b1Local = inQueueDec.DeQue<inputT>();
        LocalTensor<inputT> b2Local = inQueueB2.AllocTensor<inputT>();

        // Nz -> Zn
        LoadData(b2Local, b1Local[queue_offset_Mk], nz2nzChunkMMparams);

        inQueueB2.EnQue<inputT>(b2Local);
        inQueueDec.EnQue<inputT>(b1Local);
    }

    __aicore__ inline void SplitQA(uint64_t queue_offset_Q)
    {
        LocalTensor<inputT> q1Local = inQueueA1.DeQue<inputT>();
        LocalTensor<inputT> q2Local = inQueueA2.AllocTensor<inputT>();

        // Nz -> Zz
        for (int i = 0; i < chunk_m / CUBE_BLOCK; ++i)
            LoadData(q2Local[CeilCubeBlock(chunk_n) * dst_offset * i],
                     q1Local[src_offset * i + queue_offset_Q],
                     nz2zzChunkMNparams);

        inQueueA2.EnQue<inputT>(q2Local);
        inQueueA1.EnQue<inputT>(q1Local);
    }

    __aicore__ inline void SplitQB(uint64_t queue_offset_Q)
    {
        LocalTensor<inputT> b1Local = inQueueA1.DeQue<inputT>();
        LocalTensor<inputT> b2Local = inQueueB2.AllocTensor<inputT>();

        // Nz -> Zn
        for (int i = 0; i < chunk_m / CUBE_BLOCK; ++i)
            LoadData(b2Local[CeilCubeBlock(chunk_n) * dst_offset * i],
                     b1Local[src_offset * i + queue_offset_Q],
                     nz2znChunkParams);

        inQueueA1.EnQue<inputT>(b1Local);
        inQueueB2.EnQue<inputT>(b2Local);
    }

    __aicore__ inline void SplitKTA(uint64_t queue_offset_K)
    {
        LocalTensor<inputT> KT1Local = inQueueA1.DeQue<inputT>();
        LocalTensor<inputT> KT2Local = inQueueA2.AllocTensor<inputT>();

        // Nz -> Zz
        LoadData(KT2Local, KT1Local[queue_offset_K], nz2zzChunkMNparamsTranspose);

        inQueueA2.EnQue<inputT>(KT2Local);
        inQueueA1.EnQue<inputT>(KT1Local);
    }

    __aicore__ inline void SplitKTB(uint64_t queue_offset_K)
    {
        LocalTensor<inputT> KT1Local = inQueueA1.DeQue<inputT>();
        LocalTensor<inputT> KT2Local = inQueueB2.AllocTensor<inputT>();

        // Nz -> Zn
        LoadData(KT2Local, KT1Local[queue_offset_K], nz2znChunkParamsMNtranspose);

        inQueueA1.EnQue<inputT>(KT1Local);
        inQueueB2.EnQue<inputT>(KT2Local);
    }

    __aicore__ inline void SplitMqQ()
    {
        LocalTensor<inputT> a1Local = inQueueFP.DeQue<inputT>();
        LocalTensor<inputT> a2Local = inQueueA2.AllocTensor<inputT>();

        // Nz -> Zz
        SetWaitFlagFIXToMte1();
        for (int i = 0; i < chunk_m / CUBE_BLOCK; ++i)
            LoadData(a2Local[CeilCubeBlock(chunk_n) * dst_offset * i],
                     a1Local[src_offset * i + queue_offset_FP],
                     nz2zzChunkMNparams);

        inQueueA2.EnQue<inputT>(a2Local);
        inQueueFP.FreeTensor(a1Local);
    }

    __aicore__ inline void SplitKV(uint64_t chunk_index)
    {
        LocalTensor<inputT> b1Local = inQueueKV.DeQue<inputT>();
        LocalTensor<inputT> b2Local = inQueueB2.AllocTensor<inputT>();

        // Nz -> Zn
        SetWaitFlagFIXToMte1();
        uint64_t curKVOffset = queue_offset_KV + (chunk_index % 2) * chunk_n * chunk_n;
        for (int i = 0; i < CeilCubeBlock(chunk_n); ++i)
            LoadData(b2Local[i * dst_offset * CeilCubeBlock(chunk_n)],
                     b1Local[curKVOffset + i * src_offset],
                     nz2znChunkNNparams);

        inQueueKV.EnQue<inputT>(b1Local);
        inQueueB2.EnQue<inputT>(b2Local);
    }

    __aicore__ inline void SplitV(uint64_t queue_offset_V)
    {
        LocalTensor<inputT> b1Local = inQueueA1.DeQue<inputT>();
        LocalTensor<inputT> b2Local = inQueueB2.AllocTensor<inputT>();

        // Nz -> Zn
        for (int i = 0; i < chunk_m / CUBE_BLOCK; ++i)
            LoadData(b2Local[i * dst_offset * CeilCubeBlock(chunk_n)],
                     b1Local[i * src_offset + queue_offset_V],
                     nz2znChunkParams);

        inQueueA1.EnQue<inputT>(b1Local);
        inQueueB2.EnQue<inputT>(b2Local);
    }

    __aicore__ inline void SplitKTMk()
    {
        LocalTensor<inputT> a1Local = inQueueFP.DeQue<inputT>();
        LocalTensor<inputT> a2Local = inQueueA2.AllocTensor<inputT>();

        // Nz -> Zz
        SetWaitFlagFIXToMte1();
        for (int i = 0; i < CeilCubeBlock(chunk_n); ++i)
            LoadData(a2Local[(chunk_m / CUBE_BLOCK) * dst_offset * i],
                     a1Local[src_offset * i + queue_offset_FP],
                     nz2zzChunkNMparams);

        inQueueA2.EnQue<inputT>(a2Local);
        inQueueFP.FreeTensor(a1Local);
    }

    __aicore__ inline void SplitTMPintra()
    {
        LocalTensor<inputT> tmp1Local = inQueueA1.DeQue<inputT>();
        LocalTensor<inputT> tmp2Local = inQueueA2.AllocTensor<inputT>();

        // Nz -> Zz
        for (int i = 0; i < chunk_m / CUBE_BLOCK; ++i)
            LoadData(tmp2Local[(chunk_m / CUBE_BLOCK) * dst_offset * i],
                     tmp1Local[src_offset * i + queue_offset_TMP],
                     nz2zzChunkMMparams);

        inQueueA2.EnQue<inputT>(tmp2Local);
        inQueueA1.EnQue<inputT>(tmp1Local);
    }

    __aicore__ inline void ComputeQK(uint64_t queue_offset_Q, uint64_t queue_offset_K)
    {
        LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();
        SplitKTB(queue_offset_K);
        SplitQA(queue_offset_Q);

        LocalTensor<inputT> qLocal = inQueueA2.DeQue<inputT>();
        LocalTensor<inputT> kTLocal = inQueueB2.DeQue<inputT>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = chunk_m;
        mmadParams.n = chunk_m;
        mmadParams.k = chunk_n;
        AscendC::Mmad(c1Local, qLocal, kTLocal, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);
        inQueueA2.FreeTensor(qLocal);
        inQueueB2.FreeTensor(kTLocal);
    }

    __aicore__ inline void ComputeInter(uint64_t queue_offset_Q, uint64_t chunk_index)
    {
        LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();
        SplitMq();
        SplitQB(queue_offset_Q);
        LocalTensor<inputT> mqLocal = inQueueA2.DeQue<inputT>();
        LocalTensor<inputT> qLocal = inQueueB2.DeQue<inputT>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = chunk_m;
        mmadParams.n = chunk_n;
        mmadParams.k = chunk_m;
        AscendC::Mmad(c1Local, mqLocal, qLocal, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);
        inQueueA2.FreeTensor(mqLocal);
        inQueueB2.FreeTensor(qLocal);

        CopyOutMqQ();
        SplitKV(chunk_index);
        SplitMqQ();

        LocalTensor<inputT> mqqLocal = inQueueA2.DeQue<inputT>();
        LocalTensor<inputT> kvLocal = inQueueB2.DeQue<inputT>();

        mmadParams.m = chunk_m;
        mmadParams.n = chunk_n;
        mmadParams.k = chunk_n;
	    c1Local = outQueueCO1.AllocTensor<float>();
        AscendC::Mmad(c1Local, mqqLocal, kvLocal, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);
        inQueueA2.FreeTensor(mqqLocal);
        inQueueB2.FreeTensor(kvLocal);
    }

    __aicore__ inline void ComputeIntra(bool cmatrixInitVal, uint64_t queue_offset_V)
    {
        LocalTensor<float> c1Local;

        if (cmatrixInitVal)
            c1Local = outQueueCO1.AllocTensor<float>();
        else
            c1Local = outQueueCO1.DeQue<float>();

        SplitTMPintra();
        SplitV(queue_offset_V);

        LocalTensor<inputT> tmpLocal = inQueueA2.DeQue<inputT>();
        LocalTensor<inputT> vLocal = inQueueB2.DeQue<inputT>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = chunk_m;
        mmadParams.n = chunk_n;
        mmadParams.k = chunk_m;
        mmadParams.cmatrixInitVal = cmatrixInitVal;
        AscendC::Mmad(c1Local, tmpLocal, vLocal, mmadParams);

        outQueueCO1.EnQue<float>(c1Local);
        inQueueA2.FreeTensor(tmpLocal);
        inQueueB2.FreeTensor(vLocal);
    }

    __aicore__ inline void
    ComputeKV(uint64_t queue_offset_K, uint64_t queue_offset_V, uint64_t chunk_index, bool first_chunk)
    {
        SplitKTA(queue_offset_K);
        SplitMk();
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
        CopyOutKTMk();

        SplitV(queue_offset_V);
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
        SplitKV(chunk_index);

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

    __aicore__ inline void CopyOutKTMk()
    {
        LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        LocalTensor<inputT> a1Local = inQueueFP.AllocTensor<inputT>();

        AscendC::Fixpipe(a1Local[queue_offset_FP], c1Local, fixpipeParamsChunkL1T);

        outQueueCO1.FreeTensor(c1Local);
        inQueueFP.EnQue<inputT>(a1Local);
    }

    __aicore__ inline void CopyOutMqQ()
    {
        LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        LocalTensor<inputT> m1qLocal = inQueueFP.AllocTensor<inputT>();

        AscendC::Fixpipe(m1qLocal[queue_offset_FP], c1Local, fixpipeParamsChunkL1);

        outQueueCO1.FreeTensor(c1Local);
        inQueueFP.EnQue<inputT>(m1qLocal);
    }

    __aicore__ inline void CopyOut()
    {
        LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();

        AscendC::Fixpipe(resultGM[offset_batch + offset_Res], c1Local, fixpipeParamsChunkGM);

        outQueueCO1.FreeTensor(c1Local);

        offset_Res += chunk_m * chunk_n;
    }

    __aicore__ inline void CopyOutKV(uint64_t chunk_index, bool first_chunk)
    {
        LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        LocalTensor<inputT> kvLocal;
        if (first_chunk)
            kvLocal = inQueueKV.AllocTensor<inputT>();
        else
            kvLocal = inQueueKV.DeQue<inputT>();

        AscendC::Fixpipe(kvLocal[queue_offset_KV + (chunk_index % 2) * chunk_n * chunk_n], c1Local, fixpipeParamsKV);

        if (is_training)
        {
            AscendC::Fixpipe(DumpKVGM[offset_batch_KV + offset_DumpKV], c1Local, fixpipeParamsKV);
            offset_DumpKV += chunk_n * chunk_n;
        }

        outQueueCO1.FreeTensor(c1Local);
        inQueueKV.EnQue<inputT>(kvLocal);
    }

    __aicore__ inline void CopyOutQK(GlobalTensor<float>* operandGM)
    {
        LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();

        AscendC::Fixpipe((*operandGM)[offset_batch_TMP], c1Local, fixpipeParamsQK);

        outQueueCO1.FreeTensor(c1Local);
    }

    __aicore__ inline void FreeL1()
    {
        LocalTensor<inputT> qkvLocal = inQueueA1.DeQue<inputT>();
        inQueueA1.FreeTensor(qkvLocal);
        LocalTensor<inputT> decLocal = inQueueDec.DeQue<inputT>();
        inQueueDec.FreeTensor(decLocal);
    }

private:
    TPipe* pipe;
    TQue<TPosition::A1, 1> inQueueA1;
    TQue<TPosition::A1, 1> inQueueFP;
    TQue<TPosition::A1, 1> inQueueKV;
    TQue<TPosition::A1, 1> inQueueDec;
    TQue<TPosition::A2, 1> inQueueA2;
    TQue<TPosition::B2, 1> inQueueB2;
    TQue<TPosition::CO1, 1> outQueueCO1;
    TQue<TPosition::VECIN, 1> inQueueX, inQueueM;
    TQue<TPosition::VECOUT, 1> outQueueZ;
    TQue<TPosition::VECOUT, 1> outQueueTMP;

    GlobalTensor<inputT> QGM;
    GlobalTensor<inputT> KGM;
    GlobalTensor<inputT> VGM;
    GlobalTensor<inputT> MqGM;
    GlobalTensor<inputT> MkvGM;
    GlobalTensor<inputT> MkGM;
    GlobalTensor<float> MGM;
    GlobalTensor<outputT> resultGM;
    GlobalTensor<inputT> DumpKVGM;
    GlobalTensor<float> QKGM;
    GlobalTensor<inputT> QKMGM;

    uint64_t m, n, chunk_m, chunk_n, chunk_m_valid, num_chunks, batch;
    uint64_t numHeads;
    uint64_t offset_Q, offset_K, offset_V, offset_Res, offset_DumpKV, offset_batch, queue_offset_FP;
    uint64_t offset_Mq, offset_Mkv, offset_Mk, offset_ResIntra, offset_batch_M, offset_batch_TMP, offset_batch_KV;
    uint64_t blockIdx, block_num, block_size, extra_tasks;
    uint64_t is_training, has_tail_chunk;

    uint64_t inQueueFPSizeItems, inQueueA1SizeItems, inQueueDecSizeItems, QueueL0SizeItems, inQueueKVSizeItems;

    uint64_t queue_offset_Q, queue_offset_K, queue_offset_V, queue_offset_Mq, queue_offset_Mkv, queue_offset_Mk, queue_offset_KV,
        queue_offset_TMP;

    AscendC::Nd2NzParams nd2nzChunkMNparams, nd2nzChunkMMparams, nd2nzChunkNNparams;
    LoadData2DParams nz2nzChunkMMparams, nz2nzChunkNNparams, nz2zzChunkMMparams, nz2zzChunkNNparams, nz2zzChunkMNparams,
        nz2zzChunkNMparams, nz2znChunkParams;
    LoadData2DParams nz2zzChunkMNparamsTranspose, nz2znChunkParamsMNtranspose, nz2znChunkNNparams;

    AscendC::FixpipeParamsV220 fixpipeParamsChunkL1, fixpipeParamsChunkL1T, fixpipeParamsChunkGM, fixpipeParamsKV,
        fixpipeParamsQK;

    uint64_t dst_offset, src_offset;
};

} // namespace LightningAttn::Kernel