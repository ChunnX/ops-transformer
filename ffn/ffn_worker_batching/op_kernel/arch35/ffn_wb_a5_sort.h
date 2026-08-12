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
 * \file ffn_wb_a5_sort.h
 * \brief arch35(A5) 段内排序(VBS 段):把一段扁平 expert_id 压缩去无效项后排成有序 proposal 对。
 *
 * 与 A2 的关系:算法与硬件原语相同(Sort32 比较排序 + proposal 对表示,这两者在 A5 上仍是
 * 硬件加速路径),但本文件不引用任何 A2 头:切分参数全部来自 host tiling,预处理改用
 * regbase MicroAPI(__VEC_SCOPE__ 内的 RegTensor 计算),中间结果布局由本文件与 host 约定。
 *
 * 单段流程(与 golden 的 stable argsort 对齐):
 *   int32 expert_id --Cast--> fp32 --取负--> 降序键
 *     · 取负后用降序排序,等价于按 expert_id 升序;相等键保持输入顺序(硬件 Sort32 稳定)
 *   --CompareScalar(> -expertStart)--> 有效位 --GatherMask--> 压掉被 mask 的 token
 *     · 被 mask 的 token 在上游被置为大值(>= expertStart),压缩后不参与排序
 *     · 压缩后的个数即该段的 validCount,累加得到全局 validGatherIdxLength
 *   --尾部补 SORT_FILL_VALUE 至 ONE_REPEAT_SORT_NUM 对齐--> Concat --> Sort --> proposal 对
 */
#ifndef OP_KERNEL_ARCH35_FFN_WB_A5_SORT_H
#define OP_KERNEL_ARCH35_FFN_WB_A5_SORT_H
#include "ffn_wb_a5_context.h"

namespace FfnWbBatchingArch35 {
using namespace AscendC;

// 段内排序的切分参数,全部由 host tiling 计算后下发。
struct A5SortSegParam {
    int64_t segNum = 0;        // 序列被切成多少段
    int64_t perSegElements = 0;  // 每段元素数(末段可不足)
    int64_t totalElements = 0;   // 序列总元素数
    int64_t expertStart = 0;     // 有效 expert_id 的上界:>= 该值者视为被 mask
    int64_t sortLenPerSeg = 0;   // 每段 proposal 对区的 float 个数(host 按对齐后段长算)
};

class FfnWbA5SegSort {
public:
    __aicore__ inline FfnWbA5SegSort(){};

    // flatIdsWs   : 归一后的扁平 expert_id(int32)
    // pairWs      : 段内排序结果的 proposal 对区(float),按段 slice
    // cntWs       : 每段有效元素数(int32),每段占一个数据块槽位
    __aicore__ inline void Init(GM_ADDR flatIdsWs, GM_ADDR pairWs, GM_ADDR cntWs, const A5SortSegParam &param,
                                const ScheduleContextInfo *ctx, TPipe *pipe)
    {
        param_ = param;
        ctx_ = ctx;
        pipe_ = pipe;
        // 认领段按**真实向量核编号**:一个 block 下挂多个 AIV 子核时 GetBlockIdx() 相同,
        // 只用它认领会让同一段算两遍、另一段没人算。
        const int64_t subNum = GetTaskRation() > 0 ? GetTaskRation() : 1;
        vecId_ = GetBlockIdx() * subNum + GetSubBlockIdx();
        vecNum_ = GetBlockNum() * subNum;

        flatIdsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(flatIdsWs), param_.totalElements);
        pairGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pairWs), param_.segNum * param_.sortLenPerSeg);
        cntGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cntWs),
                               param_.segNum * (ONE_BLK_SIZE / static_cast<int64_t>(sizeof(int32_t))));

        // UB:输入段(int32 id + int32 原下标) / proposal 对区 / 排序临时区
        // 输入区按比较指令的 repeat 粒度对齐：CompareScalar 以 ONE_REPEAT_COMPARE_NUM 为单位处理，
        // 传入的元素数向上取整后可能越过段长，缓冲区需按同一粒度留足，避免读写越界。
        const int64_t alignSeg = Ceil(param_.perSegElements, ONE_REPEAT_COMPARE_NUM) * ONE_REPEAT_COMPARE_NUM;
        pipe_->InitBuffer(inQue_, 1, alignSeg * NUM_TWO * sizeof(int32_t) + ONE_BLK_SIZE);
        pipe_->InitBuffer(concatBuf_, param_.sortLenPerSeg * sizeof(float) + ONE_BLK_SIZE);
        pipe_->InitBuffer(tmpBuf_, param_.sortLenPerSeg * sizeof(float) + ONE_BLK_SIZE);
        pipe_->InitBuffer(sortedBuf_, param_.sortLenPerSeg * sizeof(float) + ONE_BLK_SIZE);
        pipe_->InitBuffer(maskBuf_, alignSeg * sizeof(uint32_t) + ONE_BLK_SIZE);
    }

    __aicore__ inline void Process()
    {
        for (int64_t seg = vecId_; seg < param_.segNum; seg += vecNum_) {
            ProcessOneSeg(seg);
        }
    }

private:
    // 取负 + 判定有效位。取负让降序排序等价于 expert_id 升序;判定阈值同样取负后比较,
    // 于是"有效"= 键 > -expertStart。这一段是纯规则的逐元素向量运算,用 MicroAPI 写。
    __aicore__ inline void PrepareKeys(const LocalTensor<float> &keys, const LocalTensor<uint32_t> &maskBits,
                                       int64_t count)
    {
        const uint16_t repeatTimes = static_cast<uint16_t>(Ceil(count, FLOAT_REG_ELEMENTS));
        uint32_t remain = static_cast<uint32_t>(count);
        __local_mem__ float *keyAddr = reinterpret_cast<__local_mem__ float *>(keys.GetPhyAddr());
        const float negOne = -1.0f;

        __VEC_SCOPE__
        {
            MicroAPI::MaskReg loopMask;
            MicroAPI::RegTensor<float> keyReg;
            for (uint16_t i = 0; i < repeatTimes; i++) {
                loopMask = MicroAPI::UpdateMask<float>(remain);
                MicroAPI::DataCopy(keyReg, keyAddr + i * FLOAT_REG_ELEMENTS);
                MicroAPI::Muls(keyReg, keyReg, negOne, loopMask);
                MicroAPI::DataCopy(keyAddr + i * FLOAT_REG_ELEMENTS, keyReg, loopMask);
            }
        }
        PipeBarrier<PIPE_V>();
        // 有效位:键 > -expertStart(即原 expert_id < expertStart)。比较按 64 元素粒度对齐。
        LocalTensor<uint8_t> maskU8 = maskBits.template ReinterpretCast<uint8_t>();
        CompareScalar(maskU8, keys, static_cast<float>(-param_.expertStart), CMPMODE::GT,
                      Ceil(count, ONE_REPEAT_COMPARE_NUM) * ONE_REPEAT_COMPARE_NUM);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessOneSeg(int64_t seg)
    {
        const int64_t begin = seg * param_.perSegElements;
        const int64_t len = (begin >= param_.totalElements)
                                ? 0
                                : Min(param_.perSegElements, param_.totalElements - begin);

        LocalTensor<int32_t> inLocal = inQue_.AllocTensor<int32_t>();
        int64_t validCnt = 0;
        if (len > 0) {
            DataCopyExtParams cp{static_cast<uint16_t>(1), static_cast<uint32_t>(len * sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
            DataCopyPad(inLocal, flatIdsGm_[begin], cp, pad);

            // 原下标:排序的 payload。段内以全局下标编号,归并后即 gather_idx。
            const int64_t alignSeg = Ceil(param_.perSegElements, ONE_REPEAT_COMPARE_NUM) * ONE_REPEAT_COMPARE_NUM;
            LocalTensor<int32_t> idxLocal = inLocal[alignSeg];
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            ArithProgression<int32_t>(idxLocal, static_cast<int32_t>(begin), 1, len);
            PipeBarrier<PIPE_V>();

            LocalTensor<float> keys = inLocal.template ReinterpretCast<float>();
            Cast(keys, inLocal, RoundMode::CAST_ROUND, len);
            PipeBarrier<PIPE_V>();

            LocalTensor<uint32_t> maskBits = maskBuf_.Get<uint32_t>();
            PrepareKeys(keys, maskBits, len);

            // 压掉被 mask 的 token:键与下标用同一套有效位压缩,压缩后个数即本段有效数。
            uint64_t rsvd = 0;
            GatherMaskParams gp;
            gp.repeatTimes = 1;
            gp.src0BlockStride = 1;
            gp.src0RepeatStride = BLOCKS_PER_REPEAT;
            gp.src1RepeatStride = 0;
            GatherMask(keys, keys, maskBits, true, static_cast<uint32_t>(len), gp, rsvd);
            PipeBarrier<PIPE_V>();
            LocalTensor<uint32_t> idxU32 = idxLocal.template ReinterpretCast<uint32_t>();
            GatherMask(idxU32, idxU32, maskBits, true, static_cast<uint32_t>(len), gp, rsvd);
            PipeBarrier<PIPE_V>();
            validCnt = static_cast<int64_t>(rsvd);

            if (validCnt > 0) {
                // 尾部补最小值到 Sort32 粒度:补位在降序中落到末尾,不影响有效元素次序。
                const int64_t alignCnt = Ceil(validCnt, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
                const int64_t tailNum = validCnt % ONE_REPEAT_SORT_NUM;
                if (tailNum > 0) {
                    // 补尾必须从**对齐的起点**发起:向量指令要求 UB 地址按数据块对齐,
                    // 直接从 keys[validCnt] 开始会触发 "VEC access UB not aligned" 硬件异常。
                    // 故起点退到本 repeat 的开头,用位掩码只写尾部那几个元素。
                    uint64_t maskBits = UINT64_MAX << tailNum;
                    maskBits &= (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
                    uint64_t dupMask[NUM_TWO] = {maskBits, 0};
                    Duplicate(keys[validCnt - tailNum], SORT_FILL_VALUE, dupMask, 1, DST_BLK_STRIDE, DST_REP_STRIDE);
                    PipeBarrier<PIPE_V>();
                }
                // Concat/Sort 的各 tensor 不允许地址重叠:concat 结果、临时区、排序结果各占一块。
                LocalTensor<float> concatLocal = concatBuf_.Get<float>();
                LocalTensor<float> tmpLocal = tmpBuf_.Get<float>();
                LocalTensor<float> sortedLocal = sortedBuf_.Get<float>();
                Concat(concatLocal, keys, tmpLocal, alignCnt / ONE_REPEAT_SORT_NUM);
                PipeBarrier<PIPE_V>();
                Sort<float, true>(sortedLocal, concatLocal, idxU32, tmpLocal, alignCnt / ONE_REPEAT_SORT_NUM);
                PipeBarrier<PIPE_V>();

                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                DataCopyExtParams cpOut{static_cast<uint16_t>(1),
                                        static_cast<uint32_t>(GetSortLen<float>(alignCnt) * sizeof(float)), 0, 0, 0};
                DataCopyPad(pairGm_[seg * param_.sortLenPerSeg], sortedLocal, cpOut);
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
            }
        }

        // 本段有效数写入计数区:跨核可见的交换一律经 UB + DataCopyPad,不用 GM 标量写。
        LocalTensor<int32_t> cntLocal = inLocal;
        cntLocal.SetValue(0, static_cast<int32_t>(validCnt));
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyExtParams cpCnt{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(cntGm_[seg * (ONE_BLK_SIZE / static_cast<int64_t>(sizeof(int32_t)))], cntLocal, cpCnt);
        // 下一段会用 MTE2 重新载入、用向量指令重写同一批缓冲:必须等本段的搬出真正读完。
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        inQue_.FreeTensor(inLocal);
    }

private:
    // 一个向量寄存器容纳的 fp32 个数,以及一次 repeat 覆盖的数据块数/比较元素数。
    static constexpr int64_t FLOAT_REG_ELEMENTS = 64;
    static constexpr int64_t BLOCKS_PER_REPEAT = 8;
    static constexpr int64_t ONE_REPEAT_COMPARE_NUM = 64;
    // Duplicate 的目的地块内步长与 repeat 间步长(按数据块计),取同仓 arch35 算子同款配置。
    static constexpr int64_t DST_BLK_STRIDE = 1;
    static constexpr int64_t DST_REP_STRIDE = 8;

    A5SortSegParam param_;
    const ScheduleContextInfo *ctx_ = nullptr;
    TPipe *pipe_ = nullptr;

    GlobalTensor<int32_t> flatIdsGm_;
    GlobalTensor<float> pairGm_;
    GlobalTensor<int32_t> cntGm_;

    TQue<QuePosition::VECIN, 1> inQue_;
    TBuf<TPosition::VECCALC> concatBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> sortedBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;

    int64_t vecId_ = 0;
    int64_t vecNum_ = 0;
};

}  // namespace FfnWbBatchingArch35
#endif  // OP_KERNEL_ARCH35_FFN_WB_A5_SORT_H
