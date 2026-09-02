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
#include "common_header.h"
#include "addr_compute.h"
#include "cube_op.h"
#include "vec_op.h"
using namespace AscendC;

namespace BSA_ARC35 {

template <typename INPUT_TYPE, uint32_t INPUT_LAYOUT, class TILING_CLASS, bool DETERMINISTIC_ENABLE>
struct BSA_TYPE {
    using input_type = INPUT_TYPE;
    static constexpr uint32_t input_layout = INPUT_LAYOUT;
    using tiling_class = TILING_CLASS;
    static constexpr bool deterministic_enable = DETERMINISTIC_ENABLE;
};

template <typename BSA_TYPE>
class BlockSparseAttentionGradArch35 {
    using INPUT_TYPE = typename BSA_TYPE::input_type;
    static constexpr uint32_t INPUT_LAYOUT = BSA_TYPE::input_layout;
    using TILING_CLASS = typename BSA_TYPE::tiling_class;
    static constexpr bool DETERMINISTIC_ENABLE = BSA_TYPE::deterministic_enable;

private:
    RunTimeInfo runTimeInfo_[2];
    AddrComputeModule<BSA_TYPE> addr_;
    GlobalTensor<INPUT_TYPE> query_gm_, key_gm_, val_gm_, dout_gm_, attention_out_gm_;
    GlobalTensor<INPUT_TYPE> dq_gm_, dk_gm_, dv_gm_;
    GlobalTensor<float> dq_workspace_, dk_workspace_, dv_workspace_, sftg_workspace_;
    TBuf<TPosition::VECCALC> ub_buffer_;
    TBuf<TPosition::A1> l1_buffer_;
    LocalTensor<float> mm1_res_ub_tensor_ping_, mm1_res_ub_tensor_pong_;
    LocalTensor<float> mm2_res_ub_tensor_ping_, mm2_res_ub_tensor_pong_;
    LocalTensor<float> mm1_res_ub_tensor_, mm2_res_ub_tensor_;
    LocalTensor<INPUT_TYPE> p_l1_tensor_ping_, p_l1_tensor_pong_;
    LocalTensor<INPUT_TYPE> ds_l1_tensor_ping_, ds_l1_tensor_pong_;
    LocalTensor<INPUT_TYPE> p_l1_tensor_, ds_l1_tensor_;
    uint32_t taskId = 0;
    uint32_t ping_pong_idx = 0;
    uint32_t last_ping_pong_idx = 0;
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

        query_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)q);
        key_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)k);
        val_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)v);
        dout_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)dout);
        attention_out_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)attention_out);
        dq_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)dq);
        dk_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)dk);
        dv_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)dv);
        dq_workspace_.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->dqWorkspaceOffset));
        dk_workspace_.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->dkWorkspaceOffset));
        dv_workspace_.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->dvWorkspaceOffset));
        sftg_workspace_.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->sftgWorkspaceOffset));

        addr_.Init(tilingData, actualQseqlen, actualKvseqlen, blockSparseMask, attentionMask);
        tPipe->InitBuffer(ub_buffer_, UB_SIZE);
        tPipe->InitBuffer(l1_buffer_, L1_SIZE);
        // ub_buffer [base_m * base_n * sizeof(float) * 2]
        mm1_res_ub_tensor_ping_ = ub_buffer_.GetWithOffset<float>(vec_base_m * vec_base_n, ub_offset_);
        ub_offset_ += vec_base_m * vec_base_n * sizeof(float);
        mm1_res_ub_tensor_pong_ = ub_buffer_.GetWithOffset<float>(vec_base_m * vec_base_n, ub_offset_);
        ub_offset_ += vec_base_m * vec_base_n * sizeof(float);
        mm2_res_ub_tensor_ping_ = ub_buffer_.GetWithOffset<float>(vec_base_m * vec_base_n, ub_offset_);
        ub_offset_ += vec_base_m * vec_base_n * sizeof(float);
        mm2_res_ub_tensor_pong_ = ub_buffer_.GetWithOffset<float>(vec_base_m * vec_base_n, ub_offset_);
        ub_offset_ += vec_base_m * vec_base_n * sizeof(float);
        // l1_buffer [base_m * base_n * sizeof(INPUT_TYPE) * 4]
        p_l1_tensor_ping_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
        l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);
        p_l1_tensor_pong_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
        l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);
        ds_l1_tensor_ping_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
        l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);
        ds_l1_tensor_pong_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
        l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);

        if ASCEND_IS_AIC {
            // printf("=====================cubeIdx=%d=====================\n", GetBlockIdx());
            CubeProcess(dout, q, k, v, attention_out, softmaxLse, blockSparseMask, blockShape, attentionMask,
                        actualQseqlen, actualKvseqlen, dq, dk, dv, workspace, tilingData, tPipe);
            // printf("====================================================\n\n");
        }
        if ASCEND_IS_AIV {
            // printf("=====================vectorIdx=%d=====================\n", GetBlockIdx());
            VectorProcess(dout, q, k, v, attention_out, softmaxLse, blockSparseMask, blockShape, attentionMask,
                          actualQseqlen, actualKvseqlen, dq, dk, dv, workspace, tilingData, tPipe);
            // printf("====================================================\n\n");
        }
    }

    __aicore__ inline void CubeProcess(GM_ADDR dout, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR attention_out,
                                       GM_ADDR softmaxLse, GM_ADDR blockSparseMask, GM_ADDR blockShape,
                                       GM_ADDR attentionMask, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR dq,
                                       GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, const TILING_CLASS *tilingData,
                                       TPipe *tPipe)
    {
        CubeOp<BSA_TYPE> cubeOp;
        cubeOp.Init(tilingData, tPipe, l1_buffer_, l1_offset_);
        CrossCoreWaitFlag(FLAG_CUBE_POST);

        while (true) {
            ping_pong_idx = taskId % 2;
            last_ping_pong_idx = 1 - ping_pong_idx;
            mm1_res_ub_tensor_ = ping_pong_idx ? mm1_res_ub_tensor_ping_ : mm1_res_ub_tensor_pong_;
            mm2_res_ub_tensor_ = ping_pong_idx ? mm2_res_ub_tensor_ping_ : mm2_res_ub_tensor_pong_;
            p_l1_tensor_ = last_ping_pong_idx ? p_l1_tensor_ping_ : p_l1_tensor_pong_;
            ds_l1_tensor_ = last_ping_pong_idx ? ds_l1_tensor_ping_ : ds_l1_tensor_pong_;
            addr_.GetRunTimeInfo(runTimeInfo_[ping_pong_idx]);

            if (runTimeInfo_[ping_pong_idx].need_compute) {
                // mm12：AIC->AIV 的 UB 结果通知（mode4，参照生产 UB buffer 模式）：
                // AIC set id 与 id+16（分别对应两个 AIV 子核），AIV wait 自己的 id
                // fixpipe 覆写保护：本轮将覆写 mm1/mm2_res_ub[slot=taskId%2]，该槽上一次
                // 被 AIV 读取是 taskId-2 轮（SoftmaxGrad 读 mm1(p)/mm2）。必须在 FIX pipe
                // 上于 fixpipe 之前等待 AIV 的释放通知 FLAG_VR（SoftmaxGrad 完成后 V pipe
                // 置位）——参照 L0_minimal FA 的 BMM1_READY 双向握手。此前仅靠 mm345 块
                // 的 MTE1 等待间接覆盖，实测 fixpipe（FIX pipe）未与之形成有效顺序，
                // blockShape=64 时 T+2 轮覆写早于 AIV 本轮 SoftmaxGrad 读完成，
                // ds 部分行读到 T+2 新数据 → dq 部分行/ dk 全行精度错误。
                CrossCoreWaitFlag<4, PIPE_FIX>(FLAG_VR);
                CrossCoreWaitFlag<4, PIPE_FIX>(FLAG_VR + AIV_SUB_FLAG_OFFSET);
                cubeOp.SendMatmulQK(query_gm_, key_gm_, mm1_res_ub_tensor_, runTimeInfo_[ping_pong_idx], ping_pong_idx);
                CrossCoreSetFlag<4, PIPE_FIX>(FLAG_C1_V1);
                CrossCoreSetFlag<4, PIPE_FIX>(FLAG_C1_V1 + AIV_SUB_FLAG_OFFSET);
                cubeOp.SendMatmulDyV(dout_gm_, val_gm_, mm2_res_ub_tensor_, runTimeInfo_[ping_pong_idx], ping_pong_idx);
                CrossCoreSetFlag<4, PIPE_FIX>(FLAG_C2_V2);
                CrossCoreSetFlag<4, PIPE_FIX>(FLAG_C2_V2 + AIV_SUB_FLAG_OFFSET);
            }

            if (taskId > 0 && runTimeInfo_[last_ping_pong_idx].need_compute) {
                // mm345：p/ds 的 L1 数据由两个 AIV 子核各写一半，mode4 下必须
                // wait id 与 id+16 两个 flag（分别对应两个子核）才能启动 GEMM
                CrossCoreWaitFlag<4, PIPE_MTE1>(FLAG_V1_C3);
                CrossCoreWaitFlag<4, PIPE_MTE1>(FLAG_V1_C3 + AIV_SUB_FLAG_OFFSET);
                cubeOp.SendMatmulDv(p_l1_tensor_, dv_workspace_, runTimeInfo_[last_ping_pong_idx], last_ping_pong_idx);
                CrossCoreWaitFlag<4, PIPE_MTE1>(FLAG_V2_C45);
                CrossCoreWaitFlag<4, PIPE_MTE1>(FLAG_V2_C45 + AIV_SUB_FLAG_OFFSET);
                cubeOp.SendMatmulDq(ds_l1_tensor_, dq_workspace_, runTimeInfo_[last_ping_pong_idx], last_ping_pong_idx);
                cubeOp.SendMatmulDk(ds_l1_tensor_, dk_workspace_, runTimeInfo_[last_ping_pong_idx], last_ping_pong_idx);
                SET_FLAG(MTE1, MTE2, EVENT_ID0);
                WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
            }

            if (runTimeInfo_[ping_pong_idx].need_compute == false) {
                break;
            }
            taskId++;
        }
        cubeOp.Destroy();
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(FLAG_CUBE_POST);
    }

    __aicore__ inline void VectorProcess(GM_ADDR dout, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR attention_out,
                                         GM_ADDR softmaxLse, GM_ADDR blockSparseMask, GM_ADDR blockShape,
                                         GM_ADDR attentionMask, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
                                         GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                         const TILING_CLASS *tilingData, TPipe *tPipe)
    {
        VecOp<BSA_TYPE> vecOp;
        vecOp.Init(dout, q, k, v, attention_out, softmaxLse, blockSparseMask, blockShape, attentionMask, actualQseqlen,
                   actualKvseqlen, dq, dk, dv, workspace, tilingData, ub_buffer_, ub_offset_);

        vecOp.SendVecPre(dq_workspace_, dk_workspace_, dv_workspace_, dout_gm_, attention_out_gm_, sftg_workspace_,
                         tilingData, ub_buffer_);
        SET_FLAG(MTE3, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
        vecOp.SendVecSftgFront(dout_gm_, attention_out_gm_, sftg_workspace_, tilingData, ub_buffer_);
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(FLAG_CUBE_POST);

        vecOp.SetFlag();
        // mm1/mm2 UB 槽初始视为已释放（无在途读取），AIC 首两轮 fixpipe 的
        // FLAG_VR 等待由该初始置位满足（mode4 下两 AIV 子核各置位 id / id+16）
        CrossCoreSetFlag<4, PIPE_V>(FLAG_VR);
        // AIV->AIC 的 p/ds 完成通知（参照生产 FA Buffer policy 的 L1 CROSS_CORE_SYNC 模式）：
        // mode4 语义下两个 AIV 子核 set 同一 flag id（硬件按 sub-block 自动分账），
        // AIC 侧 wait id 与 id+16 才代表两个子核均完成。此前 mode2 下两子核 set
        // 同一 id，快子核置位 AIC 即通过，慢子核仍可能在读 mm1/mm2 UB，被下一轮
        // T+2 的 fixpipe（乒乓同槽）覆写 → 非确定性精度错误（blockShape=64 高发）
        while (true) {
            ping_pong_idx = taskId % 2;
            last_ping_pong_idx = 1 - ping_pong_idx;
            mm1_res_ub_tensor_ = ping_pong_idx ? mm1_res_ub_tensor_ping_ : mm1_res_ub_tensor_pong_;
            mm2_res_ub_tensor_ = ping_pong_idx ? mm2_res_ub_tensor_ping_ : mm2_res_ub_tensor_pong_;
            p_l1_tensor_ = ping_pong_idx ? p_l1_tensor_ping_ : p_l1_tensor_pong_;
            ds_l1_tensor_ = ping_pong_idx ? ds_l1_tensor_ping_ : ds_l1_tensor_pong_;
            addr_.GetRunTimeInfo(runTimeInfo_[ping_pong_idx]);

            if (runTimeInfo_[ping_pong_idx].need_compute) {
                vecOp.SendVecSftPreProcess(runTimeInfo_[ping_pong_idx], ping_pong_idx);
                // AIC->AIV 的 mm1/mm2 UB 结果等待。两个要点（blockShape=64 场景实测）：
                // 1) PIPE_S 标量队列等待：PIPE_V 向量队列等待存在返回后数据尚未对向量
                //    指令可见的窗口（mm1 很短时高发，表现为 p≈exp(s) 丢失 lse 减项）；
                //    标量队列阻塞可保证 wait 返回后向量指令才开始发射。
                // 2) CrossCoreDelay：FFTS flag 信号通路与 UB 写数据通路独立传播，
                //    flag 置位可能先于数据可见，需少量自旋覆盖该传播窗口。
                CrossCoreWaitFlag<4, PIPE_S>(FLAG_C1_V1);
                CrossCoreDelay();
                vecOp.SendVecSoftmax(p_l1_tensor_, mm1_res_ub_tensor_, runTimeInfo_[ping_pong_idx]);
                CrossCoreSetFlag<4, PIPE_MTE3>(FLAG_V1_C3);

                CrossCoreWaitFlag<4, PIPE_S>(FLAG_C2_V2);
                CrossCoreDelay();
                vecOp.SendVecSoftmaxGrad(ds_l1_tensor_, mm1_res_ub_tensor_, mm2_res_ub_tensor_,
                                         runTimeInfo_[ping_pong_idx]);
                CrossCoreSetFlag<4, PIPE_MTE3>(FLAG_V2_C45);
                // 本任务对 mm1/mm2_res_ub[slot=taskId%2] 的全部读取（Softmax 读 s、
                // SoftmaxGrad 读 p/mm2）已完成，在 V pipe 上向 AIC 发释放通知：
                // AIC 在 T+2 轮 fixpipe 覆写该槽前等待 FLAG_VR（含 +16 子核分账）
                CrossCoreSetFlag<4, PIPE_V>(FLAG_VR);
            }

            if (runTimeInfo_[ping_pong_idx].need_compute == false) {
                break;
            }
            taskId++;
        }
        vecOp.WaitFlag();
        PipeBarrier<PIPE_ALL>();
        CrossCoreWaitFlag(FLAG_CUBE_POST);
        SyncAll();
        vecOp.SendVecPost(dq_gm_, dk_gm_, dv_gm_, dq_workspace_, dk_workspace_, dv_workspace_, tilingData, ub_buffer_);
    }
};

} // namespace BSA_ARC35
