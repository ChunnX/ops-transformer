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
 * \file sparse_flash_attention_grad_kernel.h
 * \brief
 */

#ifndef SPARSE_FLASH_ATTENTION_GRAD_KERNEL_H
#define SPARSE_FLASH_ATTENTION_GRAD_KERNEL_H
 
#include "sparse_flash_attention_grad_common.h"
#include "sparse_flash_attention_grad_kernel_base.h"
#include "sparse_flash_attention_grad_tiling_data_regbase.h"
 
namespace SfagBaseApi {
 
template <typename CubeBlockType, typename VecBlockType>
 
class FlashAttentionScoreGradKernel
    : public FlashAttentionScoreGradKernelBase<FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>,
                                               CubeBlockType, VecBlockType> {
public:
    ARGS_TRAITS;
    using BaseClass = FlashAttentionScoreGradKernelBase<FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>,
                                                        CubeBlockType, VecBlockType>;
    __aicore__ inline void SetUniqueRunInfo(FagRunInfo &runInfo);
    __aicore__ inline void SetUniqueConstInfo(FagConstInfo &constInfo);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessDeter();
    __aicore__ inline void ProcessUnDeter();

private:
    __aicore__ inline void ProcessPreload(FagRunInfo &runInfo, int64_t taskId, bool &isFirstBlock);
    __aicore__ inline void ProcessNotFirst(FagRunInfo &runInfo);
    __aicore__ inline void CopyLse(FagRunInfo &runInfo);
};
 
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::SetUniqueRunInfo(FagRunInfo &runInfo)
{
}
 
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::SetUniqueConstInfo(FagConstInfo &constInfo)
{
}
 
 template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::ProcessPreload(FagRunInfo &runInfo, int64_t taskId, bool &isFirstBlock) {
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C3_TO_V0_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    }
    int32_t s2RealSize = this->vecBlock.GatherKV(this->selectedKWorkSpaceGm, this->constInfo, runInfo);
    
    if ASCEND_IS_AIV {
        this->vecBlock.InitCubeVecSharedParams(runInfo, s2RealSize);

        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V0_TO_C1_FLAG[runInfo.commonRunInfo.taskIdMod2]); // dqk must wait ds copy completely
    } else {
        this->cubeBlock.GetCubeVecSharedParams(runInfo);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_V0_TO_C1_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(16 + SYNC_V0_TO_C1_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        
        // pipe_barrier(PIPE_ALL);
        // uint32_t array[] = {static_cast<uint32_t>(128), static_cast<uint32_t>(512)};
        // AscendC::ShapeInfo shapeInfo(2, array);
        // AscendC::DumpTensor(this->selectedKWorkSpaceGm[runInfo.kSelectedWsAddr], 110, 128 * 512, shapeInfo);
    }
    LocalTensor<CALC_TYPE> mm1ResTensor = this->mm1ResBuf[runInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
    this->cubeBlock.IterateMmDyV(mm1ResTensor, this->selectedKWorkSpaceGm, this->constInfo, runInfo);
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C1_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    }

    LocalTensor<CALC_TYPE> mm2ResTensor = this->mm2ResBuf[runInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
    this->cubeBlock.IterateMmQK(mm2ResTensor, this->selectedKWorkSpaceGm, this->constInfo, runInfo);
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C2_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C2_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    }
    if (isFirstBlock) {
        this->vecBlock.CopyLse(this->constInfo, runInfo); // copy in lse double buffer
        if constexpr (IsSink) {
            this->vecBlock.ProcessSoftmaxSink(this->constInfo, runInfo);
        }
        isFirstBlock = false;
    }
    runInfo.taskStep = TASK_C1C2;
 }


template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::ProcessNotFirst(FagRunInfo &runInfo) {
    this->vecBlock.ProcessVec1(this->constInfo, runInfo);
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C1_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]); // dqk must wait ds copy completely
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C2_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]); // dqk must wait ds copy completely
    }

    LocalTensor<CALC_TYPE> mm1ResTensor = this->mm1ResBuf[runInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
    LocalTensor<CALC_TYPE> mm2ResTensor = this->mm2ResBuf[runInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();

    // if ASCEND_IS_AIV {
    //     pipe_barrier(PIPE_ALL);
    //     uint32_t array[] = {static_cast<uint32_t>(32), static_cast<uint32_t>(128)};
    //     AscendC::ShapeInfo shapeInfo(2, array);
    //     // dy @ v
    //     AscendC::DumpTensor(mm1ResTensor, 111, 32 * 128, shapeInfo);
    //     // q @ k
    //     AscendC::DumpTensor(mm2ResTensor, 112, 32 * 128, shapeInfo);
    // }

    this->vecBlock.ProcessVec2(mm2ResTensor, this->constInfo, runInfo); // v2: pse + attenMask + simpleSoftmax
    // if ASCEND_IS_AIV {
    //     pipe_barrier(PIPE_ALL);
    //     uint32_t array[] = {static_cast<uint32_t>(32), static_cast<uint32_t>(128)};
    //     AscendC::ShapeInfo shapeInfo(2, array);
    //     // p
    //     AscendC::DumpTensor(mm2ResTensor, 113, 32 * 128, shapeInfo);
    // }
    if constexpr (IsSink) {
        this->vecBlock.ProcessVecSink(mm1ResTensor, mm2ResTensor, this->constInfo, runInfo);
    }
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C4_TO_V3_FLAG);
    }
    Buffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = this->dSL1Buf.Get();
    Buffer<BufferType::L1, SyncType::NO_SYNC> pL1Buffer = this->pL1Buf.Get();
    this->vecBlock.ProcessVec3(dSL1Buffer, mm1ResTensor, mm2ResTensor, this->constInfo,
                    runInfo); // v3: dropout + cast + nd2nz
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V3_TO_C3_FLAG); // dqk must wait ds copy completely
    }

    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C5_TO_V4_FLAG);
    }
    this->vecBlock.ProcessVec4(pL1Buffer, mm2ResTensor, this->constInfo, runInfo); // v4: cast + nd2nz
    if ASCEND_IS_AIV {
        // pipe_barrier(PIPE_ALL);
        // InitOutput<INPUT_TYPE>(this->selectedKWorkSpaceGm, 16384, 0);
        // for (int i=0;i<128;i++) {
        //     int64_t offset0 = i * 128;
        //     this->selectedKWorkSpaceGm.SetValue(offset0 + i, static_cast<INPUT_TYPE>(1));
        // }
        // AscendC::DataCacheCleanAndInvalid<INPUT_TYPE, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(this->selectedKWorkSpaceGm);
        // pipe_barrier(PIPE_ALL);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V4_TO_C5_FLAG); // dv must wait ds copy completely
    }
    if ASCEND_IS_AIC {
        // wait ds in ub copy to l1
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V3_TO_C3_FLAG);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V3_TO_C3_FLAG);

        // pipe_barrier(PIPE_ALL);
        // uint32_t array[] = {static_cast<uint32_t>(2), static_cast<uint32_t>(512)};
        // AscendC::ShapeInfo shapeInfo(2, array);
        // // dq
        // AscendC::DumpTensor(this->dqWorkSpaceGm, 114, 2 * 512, shapeInfo);
        // pipe_barrier(PIPE_ALL);
    }
    // compute dq
    this->cubeBlock.template IterateMmDsK<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(
        this->dqWorkSpaceGm, this->selectedKWorkSpaceGm, this->dSL1Buf, this->constInfo, runInfo); // c3
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C3_TO_V0_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(16 + SYNC_C3_TO_V0_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        if constexpr (!IsDETER) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_V5_TO_C4_FLAG[runInfo.commonRunInfo.taskIdMod2]);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_V5_TO_C4_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        }
        
        // pipe_barrier(PIPE_ALL);
        // uint32_t array[] = {static_cast<uint32_t>(64), static_cast<uint32_t>(512)};
        // AscendC::ShapeInfo shapeInfo(2, array);
        // // dq
        // AscendC::DumpTensor(this->dqWorkSpaceGm, 115, 64 * 512, shapeInfo);
        // pipe_barrier(PIPE_ALL);
    }

    // compute dk
    this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(
        this->mm4ResWorkSpaceGm, this->dSL1Buf, this->constInfo, runInfo); // c4
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C4_TO_V3_FLAG);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C4_TO_V3_FLAG);

        // pipe_barrier(PIPE_ALL);
        // uint32_t array[] = {static_cast<uint32_t>(128), static_cast<uint32_t>(512)};
        // AscendC::ShapeInfo shapeInfo(2, array);
        // // dk
        // AscendC::DumpTensor(this->mm4ResWorkSpaceGm[runInfo.mm4ResWsAddr], 116, 128 * 512, shapeInfo);
        // pipe_barrier(PIPE_ALL);
    }

    if ASCEND_IS_AIC {
        // wait p in ub copy to l1
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V4_TO_C5_FLAG);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V4_TO_C5_FLAG);
    }
    // compute dv
    this->cubeBlock.template IterateMmPDy<CALC_TYPE, BaseClass::IS_DV_WRITE_UB>(
        this->mm5ResWorkSpaceGm, this->pL1Buf, this->constInfo, runInfo); // c5
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C5_TO_V4_FLAG);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C5_TO_V4_FLAG);
        if constexpr (!IsDETER) {
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C5_TO_V5_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C5_TO_V5_FLAG);
        }

        // pipe_barrier(PIPE_ALL);
        // uint32_t array[] = {static_cast<uint32_t>(128), static_cast<uint32_t>(512)};
        // AscendC::ShapeInfo shapeInfo(2, array);
        // // dv
        // AscendC::DumpTensor(this->mm5ResWorkSpaceGm[runInfo.mm5ResWsAddr], 117, 128 * 512, shapeInfo);
        // pipe_barrier(PIPE_ALL);
    }
    runInfo.taskStep = TASK_C3C4C5;
    if constexpr (!IsDETER) {
        if ASCEND_IS_AIV {
            // wait mm4 mm5 result
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C5_TO_V5_FLAG);
        }
        GlobalTensor<CALC_TYPE> outWorkspace = runInfo.isOriKV ? this->dOriKVWorkSpaceGm : this->dCmpKVWorkSpaceGm;
        bool isSparse = runInfo.isOriKV ? IsOriKVSparse : IsCmpKVSparse;
        
        if (isSparse) {
            this->vecBlock.template ScatterAdd<true>(this->mm4ResWorkSpaceGm, this->mm5ResWorkSpaceGm, outWorkspace,
                                mm1ResTensor, mm2ResTensor, this->constInfo, runInfo);
        } else {
            this->vecBlock.template ScatterAdd<false>(this->mm4ResWorkSpaceGm, this->mm5ResWorkSpaceGm, outWorkspace,
                                mm1ResTensor, mm2ResTensor, this->constInfo, runInfo);
        }
        
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(SYNC_V5_TO_C4_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        }
    }
}


template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::Process()
{
    if constexpr (IsDETER) {
        ProcessDeter();
    } else {
        ProcessUnDeter();
    }
}
 	 
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::ProcessDeter()
{
    this->AllocEventID();
    if constexpr (IsSink) {
        this->vecBlock.CopySink(this->constInfo);
    }
    int64_t taskId = 0;
    int64_t sTaskId = 0;
    int64_t deterTaskId = 0;
    FagRunInfo runInfos[2]; // for cv ping pong
    FagRunInfo deterRunInfos[2];
    for (int32_t i = 0; i < this->processBS1ByCore; i++) {
        bool isFirstBlock = true;
        this->usedT1Index = this->cBlockIdx + this->usedCoreNum * i;
        this->GetTndSeqLen(this->usedT1Index, this->bIndex);
        for (this->n2Index = 0; this->n2Index < this->tilingData->baseParams.n2; this->n2Index++) {
            this->GetActualSelCount(this->usedT1Index, this->n2Index, this->actualSelectedBlockCount);
            if constexpr(IsOriKVExist) {
                for (this->blkCntOffset = 0; this->blkCntOffset < this->actualOriSelectedBlockCount; this->blkCntOffset += this->constInfo.selectedCountOffset) {
                    if (taskId >= 0) {
                        this->template SetRunInfo<true>(runInfos[taskId & 1], taskId, sTaskId);
                        ProcessPreload(runInfos[taskId & 1], taskId, isFirstBlock);
                    } 
                    int64_t curTaskS2RealSize = runInfos[taskId & 1].commonRunInfo.s2RealSize;
                    int64_t lastTaskS2RealSize = runInfos[(taskId + 1) & 1].commonRunInfo.s2RealSize;
                    if (taskId > 0) {
                        ProcessNotFirst(runInfos[(taskId + 1) & 1]);
                    }
                    taskId++;
                    if (curTaskS2RealSize < this->constInfo.selectedCountOffset) {
                        break;
                    }
                }
                this->template SetDeterRunInfo<true>(deterRunInfos[deterTaskId & 1], sTaskId);
                if (deterTaskId > 0) {
                    if ASCEND_IS_AIC {
                        CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                        CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SCATTER_SYNC_FLAG);
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_SYNC_FLAG);
                    } else {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SCATTER_SYNC_FLAG);
                        this->vecBlock.ScatterAddDeter(this->mm4ResWorkSpaceGm, this->mm5ResWorkSpaceGm, this->dOriKVWorkSpaceGm,
                                        this->dCmpKVWorkSpaceGm, this->constInfo, deterRunInfos[(deterTaskId + 1) & 1]);
                    }
                }
                deterTaskId++;
            }
    
            if constexpr(IsCmpKVExist) {
                for (this->blkCntOffset = this->actualOriSelectedBlockCount; this->blkCntOffset < this->actualSelectedBlockCount; this->blkCntOffset += this->constInfo.selectedCountOffset) {
                    if (taskId >= 0) {
                        this->template SetRunInfo<false>(runInfos[taskId & 1], taskId, sTaskId);
                        ProcessPreload(runInfos[taskId & 1], taskId, isFirstBlock);
                    }
                    int64_t curTaskS2RealSize = runInfos[taskId & 1].commonRunInfo.s2RealSize;
                    int64_t lastTaskS2RealSize = runInfos[(taskId + 1) & 1].commonRunInfo.s2RealSize;
                    if (taskId > 0) {
                        ProcessNotFirst(runInfos[(taskId + 1) & 1]);
                    }
                    taskId++;
                    if (curTaskS2RealSize < this->constInfo.selectedCountOffset) {
                        break;
                    }
                }
                this->template SetDeterRunInfo<false>(deterRunInfos[deterTaskId & 1], sTaskId);
                if (deterTaskId > 0) {
                    if ASCEND_IS_AIC {
                        CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                        CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SCATTER_SYNC_FLAG);
                        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_SYNC_FLAG);
                    } else {
                        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SCATTER_SYNC_FLAG);
                        this->vecBlock.ScatterAddDeter(this->mm4ResWorkSpaceGm, this->mm5ResWorkSpaceGm, this->dOriKVWorkSpaceGm,
                                        this->dCmpKVWorkSpaceGm, this->constInfo, deterRunInfos[(deterTaskId + 1) & 1]);
                    }
                }
                deterTaskId++;
            }
        }
        // if (sTaskId == 0) {
        //     if ASCEND_IS_AIC {
        //         CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SCATTER_SYNC_FLAG);
        //         CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_SYNC_FLAG);
        //     } else {
        //         CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SCATTER_SYNC_FLAG);
        //     }
        // }
        sTaskId++;
    }

    if (runInfos[(taskId + 1) & 1].taskStep == TASK_C1C2) {
        ProcessNotFirst(runInfos[(taskId + 1) & 1]);
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
            CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SCATTER_SYNC_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_SYNC_FLAG);
        } else {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SCATTER_SYNC_FLAG);
            this->vecBlock.ScatterAddDeter(this->mm4ResWorkSpaceGm, this->mm5ResWorkSpaceGm, this->dOriKVWorkSpaceGm,
                            this->dCmpKVWorkSpaceGm, this->constInfo, deterRunInfos[(deterTaskId + 1) & 1]);
        }
        deterTaskId++;
    }

    // 分核不均匀时部分核会少一个C核的全核同步
    if constexpr (IsOriKVExist) {
        if (this->processBS1ByCore < this->formerCoreProcessS1Num) {
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SCATTER_SYNC_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_SYNC_FLAG);
            } else {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SCATTER_SYNC_FLAG);
                this->template SetDeterRunInfo<true>(deterRunInfos[(deterTaskId + 1) & 1], sTaskId);
                this->vecBlock.ScatterAddDeter(this->mm4ResWorkSpaceGm, this->mm5ResWorkSpaceGm, this->dOriKVWorkSpaceGm, 
                                this->dCmpKVWorkSpaceGm, this->constInfo, deterRunInfos[(deterTaskId + 1) & 1]);
            }
            deterTaskId++;
        }
    }
    if constexpr (IsCmpKVExist) {
        if (this->processBS1ByCore < this->formerCoreProcessS1Num) {
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SCATTER_SYNC_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_SYNC_FLAG);
            } else {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SCATTER_SYNC_FLAG);
                this->template SetDeterRunInfo<false>(deterRunInfos[(deterTaskId + 1) & 1], sTaskId);
                this->vecBlock.ScatterAddDeter(this->mm4ResWorkSpaceGm, this->mm5ResWorkSpaceGm, this->dOriKVWorkSpaceGm, 
                                this->dCmpKVWorkSpaceGm, this->constInfo, deterRunInfos[(deterTaskId + 1) & 1]);
            }
            deterTaskId++;
        }
    }
    this->FreeEventID();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::ProcessUnDeter()
{
    this->AllocEventID();
    if constexpr (IsSink) {
        this->vecBlock.CopySink(this->constInfo);
    }
    int64_t taskId = 0;
    int64_t sTaskId = 0;
    FagRunInfo runInfos[2]; // for cv ping pong
    for (int32_t i = 0; i < this->processBS1ByCore; i++) {
        bool isFirstBlock = true;
        this->usedT1Index = this->cBlockIdx + this->usedCoreNum * i;
        this->GetTndSeqLen(this->usedT1Index, this->bIndex);
        for (this->n2Index = 0; this->n2Index < this->tilingData->baseParams.n2; this->n2Index++) {
            this->GetActualSelCount(this->usedT1Index, this->n2Index, this->actualSelectedBlockCount);
            if constexpr(IsOriKVExist) {
                for (this->blkCntOffset = 0; this->blkCntOffset < this->actualOriSelectedBlockCount; this->blkCntOffset += this->constInfo.selectedCountOffset) {
                    if (taskId >= 0) {
                        this->template SetRunInfo<true>(runInfos[taskId & 1], taskId, sTaskId);
                        ProcessPreload(runInfos[taskId & 1], taskId, isFirstBlock);
                    } 
                    int64_t curTaskS2RealSize = runInfos[taskId & 1].commonRunInfo.s2RealSize;
                    int64_t lastTaskS2RealSize = runInfos[(taskId + 1) & 1].commonRunInfo.s2RealSize;
                    if (taskId > 0) {
                        ProcessNotFirst(runInfos[(taskId + 1) & 1]);
                    }
                    taskId++;
                    if (curTaskS2RealSize < this->constInfo.selectedCountOffset) {
                        break;
                    }
                }
            }
    
            if constexpr(IsCmpKVExist) {
                for (this->blkCntOffset = this->actualOriSelectedBlockCount; this->blkCntOffset < this->actualSelectedBlockCount; this->blkCntOffset += this->constInfo.selectedCountOffset) {
                    if (taskId >= 0) {
                        this->template SetRunInfo<false>(runInfos[taskId & 1], taskId, sTaskId);
                        ProcessPreload(runInfos[taskId & 1], taskId, isFirstBlock);
                    }
                    int64_t curTaskS2RealSize = runInfos[taskId & 1].commonRunInfo.s2RealSize;
                    int64_t lastTaskS2RealSize = runInfos[(taskId + 1) & 1].commonRunInfo.s2RealSize;
                    if (taskId > 0) {
                        ProcessNotFirst(runInfos[(taskId + 1) & 1]);
                    }
                    taskId++;
                    if (curTaskS2RealSize < this->constInfo.selectedCountOffset) {
                        break;
                    }
                }
            }
        }
        sTaskId++;
    }

    if (runInfos[(taskId + 1) & 1].taskStep == TASK_C1C2) {
        ProcessNotFirst(runInfos[(taskId + 1) & 1]);
    }
    this->FreeEventID();
}
} // namespace SfagBaseApi
 
 
#endif