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
 * \file three_interpolate_backward.cpp
 * \brief
 */
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1u;
constexpr uint32_t BLOCK_BYTE_SIZE = 32u;
constexpr uint32_t C0 = 16;
constexpr uint32_t N0 = 16;

template <typename dataType, typename idxType>
class KernelThreeInterpolateBackward
{
public:
    __aicore__ inline KernelThreeInterpolateBackward() = default;
    __aicore__ inline void Init(
        GM_ADDR grad_x, GM_ADDR idx, GM_ADDR weight, GM_ADDR grad_y, GM_ADDR workspace,
        const ThreeInterpolateBackwardTilingData* __restrict tiling);

    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessMuiltCoreMode0();
    __aicore__ inline void ProcessMuiltCoreMode1();
    __aicore__ inline void InitMuiltCoreMode0(GM_ADDR grad_x, GM_ADDR idx, GM_ADDR weight, GM_ADDR grad_y);
    __aicore__ inline void InitMuiltCoreMode1(GM_ADDR grad_x, GM_ADDR idx, GM_ADDR weight, GM_ADDR grad_y);
    __aicore__ inline void ProcessEachBatch(uint32_t b_idx);
    __aicore__ inline void CopyIn(uint32_t b_idx, uint32_t c0_idx, uint32_t n0_idx);
    __aicore__ inline void Compute(uint32_t n0_idx);
    __aicore__ inline void CopyOut(uint32_t b_idx, uint32_t c0_idx, uint32_t n0_idx);
    __aicore__ inline void CleanOutputGm();

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> grad_x_input_queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> idx_input_queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> weight_input_queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> grad_y_output_queue;

    GlobalTensor<dataType> grad_x_gm;
    GlobalTensor<idxType> idx_gm;
    GlobalTensor<dataType> weight_gm;
    GlobalTensor<dataType> grad_y_gm;

    uint32_t core_loop_times{0};
    uint32_t core_proc_num{0};
    uint32_t core_each_loop_n_cnt{0};
    uint32_t core_last_loop_n_cnt{0};
    uint32_t data_per_b_ele_size{0};
    uint32_t data_per_c_move_ele_size{0};
    uint32_t idx_per_b_ele_size{0};
    uint32_t weight_per_b_ele_size{0};
    uint32_t output_per_b_ele_size{0};
    uint32_t output_per_c_move_ele_size{0};
    uint32_t core_proc_start_batch_idx{0};
    uint32_t core_proc_batch_cnt{0};

    const uint32_t compute_src_rep_stride_blk_size{N0 * C0 * static_cast<uint32_t>(sizeof(dataType)) / static_cast<uint32_t>(32)};
    const uint32_t compute_dst_rep_stride_blk_size{static_cast<uint32_t>(3) * compute_src_rep_stride_blk_size};
    const uint32_t copy_out_block_len{C0 * static_cast<uint32_t>(sizeof(dataType)) / static_cast<uint32_t>(32)};
    const uint32_t copy_out_src_stride_block_size = {
        static_cast<uint32_t>(3) * C0 * N0 * static_cast<uint32_t>(sizeof(dataType)) / static_cast<uint32_t>(32) - copy_out_block_len};
    uint32_t copy_in_each_loop_block_size{0};
    uint32_t copy_in_last_loop_block_size{0};
    uint32_t copy_out_dst_stride_block_size{0};
    const ThreeInterpolateBackwardTilingData* __restrict tiling_device{nullptr};
};

template <typename dataType, typename idxType>
__aicore__ inline void KernelThreeInterpolateBackward<dataType, idxType>::CleanOutputGm()
{
#ifndef __CCE_KT_TEST__
    if (GetBlockIdx() == 0) {
        auto clean_ele_size = tiling_device->bs * tiling_device->c1 * tiling_device->ms * C0;
        InitOutput<dataType>(this->grad_y_gm, clean_ele_size, 0);
    }
    SyncAll();
#endif
}

template <typename dataType, typename idxType>
__aicore__ inline void KernelThreeInterpolateBackward<dataType, idxType>::Init(
    GM_ADDR grad_x, GM_ADDR idx, GM_ADDR weight, GM_ADDR grad_y, GM_ADDR workspace,
    const ThreeInterpolateBackwardTilingData* __restrict tiling)
{
    this->tiling_device = tiling;
    if (this->tiling_device->mulit_core_mode == 0) {
        InitMuiltCoreMode0(grad_x, idx, weight, grad_y);
    } else {
        InitMuiltCoreMode1(grad_x, idx, weight, grad_y);
    }

    CleanOutputGm();
    this->pipe.InitBuffer(
        this->grad_x_input_queue, BUFFER_NUM, this->tiling_device->grad_x_move_block_size * BLOCK_BYTE_SIZE);
    this->pipe.InitBuffer(
        this->idx_input_queue, BUFFER_NUM, this->tiling_device->idx_move_block_size * BLOCK_BYTE_SIZE);
    this->pipe.InitBuffer(
        this->weight_input_queue, BUFFER_NUM, this->tiling_device->weight_move_block_size * BLOCK_BYTE_SIZE);
    this->pipe.InitBuffer(
        this->grad_y_output_queue, BUFFER_NUM, this->tiling_device->grad_y_move_block_size * BLOCK_BYTE_SIZE);

    this->data_per_b_ele_size = this->tiling_device->c1 * C0 * this->tiling_device->ns;
    this->data_per_c_move_ele_size = this->tiling_device->ns * C0 * this->tiling_device->c_move_num;

    this->idx_per_b_ele_size = this->tiling_device->ns * 3;
    this->weight_per_b_ele_size = this->tiling_device->ns * 3;

    this->output_per_b_ele_size = this->tiling_device->c1 * C0 * this->tiling_device->ms;
    this->output_per_c_move_ele_size = this->tiling_device->ms * C0 * this->tiling_device->c_move_num;
    this->copy_out_dst_stride_block_size =
        C0 * this->tiling_device->ms * sizeof(dataType) / 32 - this->copy_out_block_len;
    this->copy_in_each_loop_block_size = C0 * this->core_each_loop_n_cnt * sizeof(dataType) / 32;
    this->copy_in_last_loop_block_size = C0 * this->core_last_loop_n_cnt * sizeof(dataType) / 32;
}

template <typename dataType, typename idxType>
__aicore__ inline void KernelThreeInterpolateBackward<dataType, idxType>::Process()
{
    return tiling_device->mulit_core_mode == 0 ? ProcessMuiltCoreMode0() : ProcessMuiltCoreMode1();
}

template <typename dataType, typename idxType>
__aicore__ inline void KernelThreeInterpolateBackward<dataType, idxType>::ProcessMuiltCoreMode0()
{
    for (auto b_idx = 0u; b_idx < tiling_device->bs; b_idx++) {
        ProcessEachBatch(b_idx);
    }
}

template <typename dataType, typename idxType>
__aicore__ inline void KernelThreeInterpolateBackward<dataType, idxType>::ProcessMuiltCoreMode1()
{
    for (auto b_idx = 0; b_idx < this->core_proc_batch_cnt; b_idx++) {
        ProcessEachBatch(b_idx + this->core_proc_start_batch_idx);
    }
}

template <typename dataType, typename idxType>
__aicore__ inline void KernelThreeInterpolateBackward<dataType, idxType>::InitMuiltCoreMode0(
    GM_ADDR grad_x, GM_ADDR idx, GM_ADDR weight, GM_ADDR grad_y)
{
    uint32_t core_id = GetBlockIdx();
    bool is_last_core = (core_id == (tiling_device->used_core_num - 1));
    if (!is_last_core) {
        this->core_proc_num = tiling_device->each_core_proc_num;
        this->core_loop_times = tiling_device->each_core_loop_times;
        this->core_each_loop_n_cnt = tiling_device->each_core_each_loop_n_cnt;
        this->core_last_loop_n_cnt = tiling_device->each_core_last_loop_n_cnt;
    } else {
        this->core_proc_num = tiling_device->last_core_proc_num;
        this->core_loop_times = tiling_device->last_core_loop_times;
        this->core_each_loop_n_cnt = tiling_device->last_core_each_loop_n_cnt;
        this->core_last_loop_n_cnt = tiling_device->last_core_last_loop_n_cnt;
    }

    uint32_t core_offset = core_id * tiling_device->each_core_proc_num;
    this->grad_x_gm.SetGlobalBuffer((__gm__ dataType*)grad_x + core_offset * C0);
    this->idx_gm.SetGlobalBuffer((__gm__ idxType*)idx + core_offset * 3);
    this->weight_gm.SetGlobalBuffer((__gm__ dataType*)weight + core_offset * 3);
    this->grad_y_gm.SetGlobalBuffer((__gm__ dataType*)grad_y);
}

template <typename dataType, typename idxType>
__aicore__ inline void KernelThreeInterpolateBackward<dataType, idxType>::InitMuiltCoreMode1(
    GM_ADDR grad_x, GM_ADDR idx, GM_ADDR weight, GM_ADDR grad_y)
{
    this->core_proc_num = tiling_device->each_core_proc_num;
    this->core_loop_times = tiling_device->each_core_loop_times;
    this->core_each_loop_n_cnt = tiling_device->each_core_each_loop_n_cnt;
    this->core_last_loop_n_cnt = tiling_device->each_core_last_loop_n_cnt;

    uint32_t core_id = GetBlockIdx();
    if (core_id < tiling_device->core_proc_batch_padding_idx) {
        this->core_proc_batch_cnt = tiling_device->each_core_proc_batch_num + 1;
        this->core_proc_start_batch_idx = core_id * (tiling_device->each_core_proc_batch_num + 1);
    } else {
        this->core_proc_batch_cnt = tiling_device->each_core_proc_batch_num;
        this->core_proc_start_batch_idx =
            tiling_device->core_proc_batch_padding_idx * (tiling_device->each_core_proc_batch_num + 1) +
            (core_id - tiling_device->core_proc_batch_padding_idx) * tiling_device->each_core_proc_batch_num;
    }

    uint32_t core_offset = core_id * tiling_device->each_core_proc_num;
    this->grad_x_gm.SetGlobalBuffer((__gm__ dataType*)grad_x);
    this->idx_gm.SetGlobalBuffer((__gm__ idxType*)idx);
    this->weight_gm.SetGlobalBuffer((__gm__ dataType*)weight);
    this->grad_y_gm.SetGlobalBuffer((__gm__ dataType*)grad_y);
}

template <typename dataType, typename idxType>
__aicore__ inline void KernelThreeInterpolateBackward<dataType, idxType>::ProcessEachBatch(uint32_t b_idx)
{
    for (auto c0_idx = 0u; c0_idx < tiling_device->c_move_loop_times; c0_idx++) {
        for (auto n0_idx = 0u; n0_idx < core_loop_times; n0_idx++) {
            CopyIn(b_idx, c0_idx, n0_idx);
            Compute(n0_idx);
            CopyOut(b_idx, c0_idx, n0_idx);
        }
    }
}

template <typename dataType, typename idxType>
__aicore__ inline void KernelThreeInterpolateBackward<dataType, idxType>::CopyIn(
    uint32_t b_idx, uint32_t c0_idx, uint32_t n0_idx)
{
    LocalTensor<dataType> grad_x_local = grad_x_input_queue.AllocTensor<dataType>();
    LocalTensor<idxType> idx_local = idx_input_queue.AllocTensor<idxType>();
    LocalTensor<dataType> weight_local = weight_input_queue.AllocTensor<dataType>();

    auto gard_x_addr_offset =
        b_idx * this->data_per_b_ele_size + c0_idx * this->data_per_c_move_ele_size + n0_idx * N0 * C0;

    auto idx_addr_offset = b_idx * this->idx_per_b_ele_size + n0_idx * N0 * static_cast<uint32_t>(3);

    auto weight_addr_offset = b_idx * this->weight_per_b_ele_size + n0_idx * N0 * static_cast<uint32_t>(3);

    auto move_c_cnt = (c0_idx != tiling_device->c_move_loop_times - 1) ? tiling_device->c_move_num :
                                                                         tiling_device->c_last_loop_move_num;

    DataCopyParams data_copy_params;
    data_copy_params.blockCount = move_c_cnt;
    data_copy_params.blockLen =
        (n0_idx != this->core_loop_times - 1) ? this->copy_in_each_loop_block_size : this->copy_in_last_loop_block_size;
    data_copy_params.dstStride = C0 * N0 * sizeof(dataType) / 32 - data_copy_params.blockLen;
    DataCopy(grad_x_local, grad_x_gm[gard_x_addr_offset], data_copy_params);
    DataCopy(idx_local, idx_gm[idx_addr_offset], N0 * 3);
    DataCopy(weight_local, weight_gm[weight_addr_offset], N0 * 3);

    grad_x_input_queue.EnQue<dataType>(grad_x_local);
    idx_input_queue.EnQue<idxType>(idx_local);
    weight_input_queue.EnQue<dataType>(weight_local);
}

template <typename dataType, typename idxType>
__aicore__ inline void KernelThreeInterpolateBackward<dataType, idxType>::Compute(uint32_t n0_idx)
{
    LocalTensor<dataType> grad_x_local = grad_x_input_queue.DeQue<dataType>();
    LocalTensor<dataType> weight_local = weight_input_queue.DeQue<dataType>();
    LocalTensor<dataType> grad_y_local = grad_y_output_queue.AllocTensor<dataType>();
    SetFlag<HardEvent::MTE2_S>(EVENT_ID3);

    // 计算逻辑
    auto compute_n_cnt =
        (n0_idx != this->core_loop_times - 1) ? this->core_each_loop_n_cnt : this->core_last_loop_n_cnt;

    UnaryRepeatParams compute_repeat_info;
    compute_repeat_info.dstBlkStride = 1;
    compute_repeat_info.srcBlkStride = 1;
    compute_repeat_info.srcRepStride = this->compute_src_rep_stride_blk_size;
    compute_repeat_info.dstRepStride = this->compute_dst_rep_stride_blk_size;
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
    for (auto n_idx = 0u; n_idx < compute_n_cnt; n_idx++) {
        auto idx = static_cast<decltype(n_idx)>(3) * n_idx;
        auto weight0 = weight_local.GetValue(idx + 0);
        auto weight1 = weight_local.GetValue(idx + 1);
        auto weight2 = weight_local.GetValue(idx + 2);

        SetFlag<HardEvent::S_V>(EVENT_ID2);
        WaitFlag<HardEvent::S_V>(EVENT_ID2);

        Muls(
            grad_y_local[(idx + 0) * C0], grad_x_local[n_idx * C0], weight0, C0, tiling_device->c_move_num,
            compute_repeat_info);

        Muls(
            grad_y_local[(idx + 1) * C0], grad_x_local[n_idx * C0], weight1, C0, tiling_device->c_move_num,
            compute_repeat_info);

        Muls(
            grad_y_local[(idx + 2) * C0], grad_x_local[n_idx * C0], weight2, C0, tiling_device->c_move_num,
            compute_repeat_info);
    }

    grad_y_output_queue.EnQue<dataType>(grad_y_local);
    grad_x_input_queue.FreeTensor(grad_x_local);
    weight_input_queue.FreeTensor(weight_local);
}

template <typename dataType, typename idxType>
__aicore__ inline void KernelThreeInterpolateBackward<dataType, idxType>::CopyOut(
    uint32_t b_idx, uint32_t c0_idx, uint32_t n0_idx)
{
    LocalTensor<dataType> grad_y_local = grad_y_output_queue.DeQue<dataType>();
    LocalTensor<idxType> idx_local = idx_input_queue.DeQue<idxType>();
    SetFlag<HardEvent::MTE2_S>(EVENT_ID3);

    auto move_n_cnt = (n0_idx != this->core_loop_times - 1) ? this->core_each_loop_n_cnt : this->core_last_loop_n_cnt;

    auto move_c_cnt = (c0_idx != tiling_device->c_move_loop_times - 1) ? tiling_device->c_move_num :
                                                                         tiling_device->c_last_loop_move_num;

    auto grad_y_start_addr_offset = b_idx * this->output_per_b_ele_size + c0_idx * this->output_per_c_move_ele_size;

    DataCopyParams data_copy_params;
    data_copy_params.blockCount = move_c_cnt;
    data_copy_params.blockLen = this->copy_out_block_len;
    data_copy_params.srcStride = this->copy_out_src_stride_block_size;
    data_copy_params.dstStride = this->copy_out_dst_stride_block_size;
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
    SetAtomicAdd<dataType>();
    for (auto n_idx = 0u; n_idx < move_n_cnt; n_idx++) {
        auto idx = static_cast<decltype(n_idx)>(3) * n_idx;
        auto grad_y_addr_offset_0 = grad_y_start_addr_offset + idx_local.GetValue(idx + 0) * C0;
        auto grad_y_addr_offset_1 = grad_y_start_addr_offset + idx_local.GetValue(idx + 1) * C0;
        auto grad_y_addr_offset_2 = grad_y_start_addr_offset + idx_local.GetValue(idx + 2) * C0;

        SetFlag<HardEvent::S_V>(EVENT_ID2);
        WaitFlag<HardEvent::S_V>(EVENT_ID2);

        DataCopy(grad_y_gm[grad_y_addr_offset_0], grad_y_local[C0 * (idx + 0)], data_copy_params);
        PipeBarrier<PIPE_MTE3>();;
        DataCopy(grad_y_gm[grad_y_addr_offset_1], grad_y_local[C0 * (idx + 1)], data_copy_params);
        PipeBarrier<PIPE_MTE3>();;
        DataCopy(grad_y_gm[grad_y_addr_offset_2], grad_y_local[C0 * (idx + 2)], data_copy_params);
        PipeBarrier<PIPE_MTE3>();;
    }
    SetAtomicNone();

    grad_y_output_queue.FreeTensor(grad_y_local);
    idx_input_queue.FreeTensor(idx_local);
}

extern "C" __global__ __aicore__ void three_interpolate_backward(
    GM_ADDR grad_x, GM_ADDR idx, GM_ADDR weight, GM_ADDR grad_y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR user_ws = GetUserWorkspace(workspace);
    if (user_ws == nullptr) {
        return;
    }

    GET_TILING_DATA(tiling_data, tiling);
    const ThreeInterpolateBackwardTilingData* __restrict tiling_device = &tiling_data;

    if (TILING_KEY_IS(0)) { // float32 int32
        KernelThreeInterpolateBackward<float, int32_t> op;
        op.Init(grad_x, idx, weight, grad_y, user_ws, tiling_device);
        op.Process();
    } else if (TILING_KEY_IS(1)) { // float32 int64
        KernelThreeInterpolateBackward<float, int64_t> op;
        op.Init(grad_x, idx, weight, grad_y, user_ws, tiling_device);
        op.Process();
    } else if (TILING_KEY_IS(2)) { // float16 int32
        KernelThreeInterpolateBackward<half, int32_t> op;
        op.Init(grad_x, idx, weight, grad_y, user_ws, tiling_device);
        op.Process();
    } else if (TILING_KEY_IS(3)) { // float16 int64
        KernelThreeInterpolateBackward<half, int64_t> op;
        op.Init(grad_x, idx, weight, grad_y, user_ws, tiling_device);
        op.Process();
    }
}