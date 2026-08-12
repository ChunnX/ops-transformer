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
 * \file ffn_wb_a5_extract.h
 * \brief arch35(A5) 归并收尾:把最终一路 proposal 对拆成「有序 expert_id」与「gather_idx」两个数组。
 *
 * 排序阶段用的是取负后的降序键,这里拆包后再取负还原成原始 expert_id。
 * 两个输出各自被下游独占使用:有序 id 供 group_list 计数,gather_idx 供 gather 搬运。
 * 分块由 host 下发(块长按 Sort32 粒度对齐,保证 Extract 的 repeat 边界不跨块)。
 */
#ifndef OP_KERNEL_ARCH35_FFN_WB_A5_EXTRACT_H
#define OP_KERNEL_ARCH35_FFN_WB_A5_EXTRACT_H
#include "ffn_wb_a5_context.h"

namespace FfnWbBatchingArch35 {
using namespace AscendC;

struct A5ExtractParam {
    int64_t totalValid = 0;      // 最终有序序列的元素数
    int64_t perLoopElements = 0; // 单次载入 UB 的元素数(host 保证为 Sort32 粒度的整数倍)
};

class FfnWbA5Extract {
public:
    __aicore__ inline FfnWbA5Extract(){};

    // pairWs   : 最终一路 proposal 对
    // idsWs    : 输出,有序 expert_id(int32)
    // idxWs    : 输出,gather_idx(int32,指向原始扁平序列的下标)
    __aicore__ inline void Init(GM_ADDR pairWs, GM_ADDR idsWs, GM_ADDR idxWs, const A5ExtractParam &param, TPipe *pipe)
    {
        param_ = param;
        pipe_ = pipe;
        const int64_t subNum = GetTaskRation() > 0 ? GetTaskRation() : 1;
        vecId_ = GetBlockIdx() * subNum + GetSubBlockIdx();
        vecNum_ = GetBlockNum() * subNum;

        pairGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pairWs), GetSortLen<float>(param_.totalValid));
        idsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(idsWs), param_.totalValid);
        idxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(idxWs), param_.totalValid);

        // Extract 按 ONE_REPEAT_SORT_NUM 成批写出:落盘量是**对齐后的长度**,不是有效长度。
        // 按有效长度分配会溢出到相邻缓冲(实测把索引缓冲的开头冲成 0),故三块都按对齐长度给。
        const int64_t alignLoop =
            Ceil(param_.perLoopElements, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        pipe_->InitBuffer(pairQue_, 1, GetSortLen<float>(alignLoop) * sizeof(float) + ONE_BLK_SIZE);
        pipe_->InitBuffer(idsBuf_, alignLoop * sizeof(int32_t) + ONE_BLK_SIZE);
        pipe_->InitBuffer(idxBuf_, alignLoop * sizeof(int32_t) + ONE_BLK_SIZE);
    }

    __aicore__ inline void Process()
    {
        const int64_t loops = Ceil(param_.totalValid, param_.perLoopElements);
        for (int64_t l = vecId_; l < loops; l += vecNum_) {
            const int64_t begin = l * param_.perLoopElements;
            const int64_t len = Min(param_.perLoopElements, param_.totalValid - begin);
            if (len <= 0) {
                continue;
            }
            ProcessOneChunk(begin, len);
        }
    }

private:
    __aicore__ inline void ProcessOneChunk(int64_t begin, int64_t len)
    {
        // Extract 按 Sort32 粒度成批工作,故按对齐后的长度拆包,超出有效长度的部分不写出。
        const int64_t alignLen = Ceil(len, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;

        LocalTensor<float> pairLocal = pairQue_.AllocTensor<float>();
        DataCopyExtParams cpIn{static_cast<uint16_t>(1),
                               static_cast<uint32_t>(GetSortLen<float>(len) * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(pairLocal, pairGm_[GetSortLen<float>(begin)], cpIn, pad);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

        LocalTensor<int32_t> idsLocal = idsBuf_.Get<int32_t>();
        LocalTensor<int32_t> idxLocal = idxBuf_.Get<int32_t>();
        LocalTensor<float> keysLocal = idsLocal.template ReinterpretCast<float>();
        LocalTensor<uint32_t> idxU32 = idxLocal.template ReinterpretCast<uint32_t>();

        Extract(keysLocal, idxU32, pairLocal, static_cast<int32_t>(alignLen / ONE_REPEAT_SORT_NUM));
        PipeBarrier<PIPE_V>();
        // 还原排序前取的负号,再转回 int32
        Muls(keysLocal, keysLocal, static_cast<float>(-1), len);
        PipeBarrier<PIPE_V>();
        Cast(idsLocal, keysLocal, RoundMode::CAST_ROUND, len);
        PipeBarrier<PIPE_V>();

        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopyExtParams cpOut{static_cast<uint16_t>(1), static_cast<uint32_t>(len * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(idsGm_[begin], idsLocal, cpOut);
        DataCopyPad(idxGm_[begin], idxLocal, cpOut);
        // 下一块会用 Extract(向量)重写这两块缓冲:必须等本块搬出读完。
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        pairQue_.FreeTensor(pairLocal);
    }

private:
    A5ExtractParam param_;
    TPipe *pipe_ = nullptr;

    GlobalTensor<float> pairGm_;
    GlobalTensor<int32_t> idsGm_;
    GlobalTensor<int32_t> idxGm_;

    TQue<QuePosition::VECIN, 1> pairQue_;
    TBuf<TPosition::VECCALC> idsBuf_;
    TBuf<TPosition::VECCALC> idxBuf_;

    int64_t vecId_ = 0;
    int64_t vecNum_ = 0;
};

}  // namespace FfnWbBatchingArch35
#endif  // OP_KERNEL_ARCH35_FFN_WB_A5_EXTRACT_H
