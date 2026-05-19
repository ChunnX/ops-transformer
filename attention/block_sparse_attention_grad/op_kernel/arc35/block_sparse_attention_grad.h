/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "kernel_operator.h"
#include "common_header.h"
#include "addr_compute.h"
#include "cube_op.h"
#include "vec_op.h"
using namespace AscendC;

namespace BSA_ARC35 {

template <typename INPUT_TYPE, class TILING_CLASS, bool DETERMINISTIC_ENABLE>
struct BSA_TYPE {
    using input_type = INPUT_TYPE;
    using tiling_class = TILING_CLASS;
    static constexpr bool deterministic_enable = DETERMINISTIC_ENABLE;
};

template <typename BSA_TYPE>
class BlockSparseAttentionGradArch35 {
    using INPUT_TYPE = typename BSA_TYPE::input_type;
    using TILING_CLASS = typename BSA_TYPE::tiling_class;
    static constexpr bool DETERMINISTIC_ENABLE = BSA_TYPE::deterministic_enable;

private:
    RunTimeInfo runTimeInfo_;
    AddrComputeModule<BSA_TYPE> addr_;
    GlobalTensor<INPUT_TYPE> query_gm_, key_gm_, val_gm_, dout_gm_;
    GlobalTensor<INPUT_TYPE> dq_gm_, dk_gm_, dv_gm_;
    GlobalTensor<float> dq_workspace_, dk_workspace_, dv_workspace_;
    TBuf<TPosition::VECCALC> ub_buffer_;
    TBuf<TPosition::A1> l1_buffer_;
    LocalTensor<float> mm1_res_ub_tensor_ping_;
    LocalTensor<float> mm1_res_ub_tensor_pong_;
    LocalTensor<INPUT_TYPE> softmax_res_l1_tensor_ping_;
    LocalTensor<INPUT_TYPE> softmax_res_l1_tensor_pong_;
    uint32_t ub_offset_ = 0;
    uint32_t l1_offset_ = 0;
    static constexpr int32_t UB_SIZE = 247 * 1024;
    static constexpr int32_t L1_SIZE = 512 * 1024;

public:
    __aicore__ inline BlockSparseAttentionGradArch35(){};

    __aicore__ inline void Process(GM_ADDR dout, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR attention_out,
                                   GM_ADDR softmaxLse, GM_ADDR blockSparseMask, GM_ADDR blockShape,
                                   GM_ADDR attentionMask, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR dq,
                                   GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, const TILING_CLASS *tilingData,
                                   TPipe *tPipe)
    {
        uint32_t base_m = tilingData->baseM;
        uint32_t base_n = tilingData->baseN;
        uint32_t vec_base_m = base_m / 2;
        uint32_t vec_base_n = base_n;
        uint64_t dq_workspace_offset = tilingData->dqWorkspaceOffset;
        uint64_t dk_workspace_offset = tilingData->dkWorkspaceOffset;
        uint64_t dv_workspace_offset = tilingData->dvWorkspaceOffset;

        query_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)q);
        key_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)k);
        val_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)v);
        dout_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)dout);
        dq_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)dq);
        dk_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)dk);
        dv_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)dv);
        dq_workspace_.SetGlobalBuffer((__gm__ float *)(workspace + dq_workspace_offset));
        dk_workspace_.SetGlobalBuffer((__gm__ float *)(workspace + dk_workspace_offset));
        dv_workspace_.SetGlobalBuffer((__gm__ float *)(workspace + dv_workspace_offset));

        addr_.Init(tilingData, actualQseqlen, actualKvseqlen);
        tPipe->InitBuffer(ub_buffer_, UB_SIZE);
        tPipe->InitBuffer(l1_buffer_, L1_SIZE);
        // ub_buffer
        mm1_res_ub_tensor_ping_ = ub_buffer_.GetWithOffset<float>(vec_base_m * vec_base_n, ub_offset_);
        ub_offset_ += vec_base_m * vec_base_n * sizeof(float);
        mm1_res_ub_tensor_pong_ = ub_buffer_.GetWithOffset<float>(vec_base_m * vec_base_n, ub_offset_);
        ub_offset_ += vec_base_m * vec_base_n * sizeof(float);
        // l1_buffer
        softmax_res_l1_tensor_ping_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
        l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);
        softmax_res_l1_tensor_pong_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
        l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);

        if ASCEND_IS_AIC {
            printf("=====================cubeIdx=%d=====================\n", GetBlockIdx());
            CubeProcess(dout, q, k, v, attention_out, softmaxLse, blockSparseMask, blockShape, attentionMask,
                        actualQseqlen, actualKvseqlen, dq, dk, dv, workspace, tilingData, tPipe);
            printf("====================================================\n\n");
        }
        if ASCEND_IS_AIV {
            printf("=====================vectorIdx=%d=====================\n", GetBlockIdx());
            VectorProcess(dout, q, k, v, attention_out, softmaxLse, blockSparseMask, blockShape, attentionMask,
                          actualQseqlen, actualKvseqlen, dq, dk, dv, workspace, tilingData, tPipe);
            printf("====================================================\n\n");
        }
    }

    __aicore__ inline void CubeProcess(GM_ADDR dout, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR attention_out,
                                       GM_ADDR softmaxLse, GM_ADDR blockSparseMask, GM_ADDR blockShape,
                                       GM_ADDR attentionMask, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR dq,
                                       GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, const TILING_CLASS *tilingData,
                                       TPipe *tPipe)
    {
        uint32_t taskId = 0;
        CubeOp<BSA_TYPE> cubeOp;
        cubeOp.Init(tilingData, tPipe, l1_buffer_, l1_offset_);

        while (true) {
            addr_.GetRunTimeInfo(&runTimeInfo_);
            if (runTimeInfo_.loop_finish) {
                break;
            }
            cubeOp.SendMatmulQK(query_gm_[runTimeInfo_.queryGmOffset], key_gm_[runTimeInfo_.keyGmOffset],
                                mm1_res_ub_tensor_ping_, runTimeInfo_);
            // cubeOp.SendMatmulDyV(dout_gm_[runTimeInfo_.queryGmOffset], val_gm_[runTimeInfo_.keyGmOffset],
            //                      mm1_res_ub_tensor_ping_, runTimeInfo_);
            cubeOp.SendMatmulDv(softmax_res_l1_tensor_ping_, dout_gm_[runTimeInfo_.queryGmOffset],
                                dv_workspace_[runTimeInfo_.keyGmOffset], runTimeInfo_);
        }
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(FLAG_CUBE_POST);
    }

    __aicore__ inline void VectorProcess(GM_ADDR dout, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR attention_out,
                                         GM_ADDR softmaxLse, GM_ADDR blockSparseMask, GM_ADDR blockShape,
                                         GM_ADDR attentionMask, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
                                         GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                         const TILING_CLASS *tilingData, TPipe *tPipe)
    {
        uint32_t taskId = 0;
        VecOp<BSA_TYPE> vecOp;
        vecOp.Init(dout, q, k, v, attention_out, softmaxLse, blockSparseMask, blockShape, attentionMask, actualQseqlen,
                   actualKvseqlen, dq, dk, dv, workspace, tilingData, ub_buffer_, ub_offset_);

        while (true) {
            addr_.GetRunTimeInfo(&runTimeInfo_);
            if (runTimeInfo_.loop_finish) {
                break;
            }
            vecOp.SendVecSoftmax(softmax_res_l1_tensor_ping_, mm1_res_ub_tensor_ping_, runTimeInfo_);
            // vecOp.SendVecSoftmaxGrad(softmax_res_l1_tensor_ping_, mm1_res_ub_tensor_ping_, &runTimeInfo_);
        }

        PipeBarrier<PIPE_ALL>();
        CrossCoreWaitFlag(FLAG_CUBE_POST);
        SyncAll();
        vecOp.SendVecPost(dq_gm_, dk_gm_, dv_gm_, dq_workspace_, dk_workspace_, dv_workspace_, tilingData, ub_buffer_);
    }
};

} // namespace BSA_ARC35
