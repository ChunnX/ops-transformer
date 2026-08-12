/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file ffn_wb_a5_scan.h
 * \brief arch35(A5) RECV 路径的握手等待:等本 micro batch 的全部 session 就绪,再推进轮询下标。
 *
 * token_info 按 [A][M][F] 排布,F = flag + layer_id + expert_ids[BS*K](见公共契约 FfnDataDesc)。
 * 本阶段只读每个 session 在当前 micro batch 上的 flag:跨 session 以 M*F 为步长取一个 int32,
 * 求和等于 A 即全部就绪。就绪后把 polling_index 推进到下一个 micro batch,通知调度侧。
 *
 * 轮询下标的回写走 UB + DataCopyPad,不用 GM 标量写 + 手工刷 cache:
 * 后者依赖"写入落在单条 cache line 内"才碰巧成立,不是可依赖的语义。
 */
#ifndef OP_KERNEL_ARCH35_FFN_WB_A5_SCAN_H
#define OP_KERNEL_ARCH35_FFN_WB_A5_SCAN_H
#include "ffn_wb_a5_context.h"

namespace FfnWbBatchingArch35 {
using namespace AscendC;

class FfnWbA5RecvWait {
public:
    __aicore__ inline FfnWbA5RecvWait(){};

    __aicore__ inline void Init(GM_ADDR schedule_context, GM_ADDR tokenInfoBuf, const ScheduleContextInfo *ctx,
                                TPipe *pipe)
    {
        ctx_ = ctx;
        pipe_ = pipe;
        // FfnDataDesc 每块的 int32 个数:flag + layer_id + expert_ids[BS*K],块数由契约结构给出。
        descWords_ = static_cast<int64_t>(sizeof(aicpu::FfnDataDesc)) / static_cast<int64_t>(sizeof(int32_t)) +
                     static_cast<int64_t>(ctx_->BS) * ctx_->K;
        tokenInfoGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tokenInfoBuf),
                                     static_cast<int64_t>(ctx_->A) * ctx_->M * descWords_);
        ctxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(schedule_context));

        const int64_t flagWords = static_cast<int64_t>(ctx_->A) * (ONE_BLK_SIZE / sizeof(int32_t));
        pipe_->InitBuffer(flagQue_, 1, flagWords * sizeof(int32_t));
        pipe_->InitBuffer(workBuf_, flagWords * sizeof(float));
        pipe_->InitBuffer(pollBuf_, ONE_BLK_SIZE);
    }

    // 由 0 号核忙等;其余核在调用方的 SyncAll 处等待。
    __aicore__ inline void Process()
    {
        const int64_t subNum = GetTaskRation() > 0 ? GetTaskRation() : 1;
        if (GetBlockIdx() * subNum + GetSubBlockIdx() != 0) {
            return;
        }
        const int64_t flagElems = static_cast<int64_t>(ctx_->A) * (ONE_BLK_SIZE / sizeof(int32_t));
        const int64_t offset = static_cast<int64_t>(ctx_->curMicroBatchID) * descWords_;

        while (true) {
            LocalTensor<int32_t> flagLocal = flagQue_.AllocTensor<int32_t>();
            // 每 session 取一个 flag:块数 A、块长 4B、块间跨度 (M*F-1) 个 int32
            DataCopyExtParams cp{static_cast<uint16_t>(ctx_->A), static_cast<uint32_t>(sizeof(int32_t)),
                                 static_cast<uint32_t>((static_cast<int64_t>(ctx_->M) * descWords_ - 1) *
                                                       sizeof(int32_t)),
                                 0, 0};
            DataCopyPadExtParams<int32_t> pad{true, 0,
                                              static_cast<uint8_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t)),
                                              0};
            DataCopyPad(flagLocal, tokenInfoGm_[offset], cp, pad);
            flagQue_.EnQue(flagLocal);

            LocalTensor<int32_t> flags = flagQue_.DeQue<int32_t>();
            LocalTensor<float> work = workBuf_.Get<float>();
            Cast(work, flags, RoundMode::CAST_ROUND, flagElems);
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(work, work, work, flagElems);
            PipeBarrier<PIPE_V>();
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            const float readyNum = work.GetValue(0);
            flagQue_.FreeTensor(flags);
            if (static_cast<uint32_t>(readyNum) >= ctx_->A) {
                break;
            }
        }

        // 推进轮询下标:偏移取自公共契约结构,回写经 UB 整段搬运。
        LocalTensor<uint64_t> pollLocal = pollBuf_.Get<uint64_t>();
        pollLocal.SetValue(0, (ctx_->curMicroBatchID + 1) % ctx_->M);
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyExtParams cpPoll{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(uint64_t)), 0, 0, 0};
        DataCopyPad(ctxGm_[FFN_WB_CTX_OFFSET(ffn.polling_index) / static_cast<int32_t>(sizeof(uint64_t))], pollLocal,
                    cpPoll);
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

private:
    const ScheduleContextInfo *ctx_ = nullptr;
    TPipe *pipe_ = nullptr;
    GlobalTensor<int32_t> tokenInfoGm_;
    GlobalTensor<uint64_t> ctxGm_;
    TQue<QuePosition::VECIN, 1> flagQue_;
    TBuf<TPosition::VECCALC> workBuf_;
    TBuf<TPosition::VECCALC> pollBuf_;
    int64_t descWords_ = 0;
};

}  // namespace FfnWbBatchingArch35
#endif  // OP_KERNEL_ARCH35_FFN_WB_A5_SCAN_H
