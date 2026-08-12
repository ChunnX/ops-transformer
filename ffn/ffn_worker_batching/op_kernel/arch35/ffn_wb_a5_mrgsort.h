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
 * \file ffn_wb_a5_mrgsort.h
 * \brief arch35(A5) 段间归并(VMS 段):把段内已有序的若干路 proposal 对归并成一路。
 *
 * 归并是**真归并**:每轮用硬件 MrgSort 做 MRG_LIST_NUM 路合并,轮与轮之间在两块工作区之间乒乓,
 * 直到只剩一路。段的长度不等(各段被 mask 掉的 token 数不同),故每路的剩余长度需逐轮维护。
 *
 * 槽位布局:两块工作区都按"每段一个固定 slot(sortLenPerSeg 个 float)"划分。
 * 第 r 轮把第 g 组的 MRG_LIST_NUM 路(段号 g*4^(r+1) 起,步长 4^r)归并后写到对面工作区的
 * 段号 g*4^(r+1) 处 —— 一组占的 4 个 slot 恰好容得下这组的合并结果,因此不需要前缀和重排。
 *
 * 单轮流式:UB 放不下整路,故每路每次载入至多 oneLoopMaxElements 个元素做一次 MrgSort;
 * MrgSort 会给出每路实际消耗的个数(listSortedNums),据此推进各路偏移,直到全部耗尽。
 */
#ifndef OP_KERNEL_ARCH35_FFN_WB_A5_MRGSORT_H
#define OP_KERNEL_ARCH35_FFN_WB_A5_MRGSORT_H
#include "ffn_wb_a5_context.h"

namespace FfnWbBatchingArch35 {
using namespace AscendC;

// 归并的切分参数,由 host tiling 计算下发。
struct A5MergeParam {
    int64_t segNum = 0;             // 段数(= 第 0 轮的路数)
    int64_t sortLenPerSeg = 0;      // 每段 slot 的 float 个数
    int64_t oneLoopMaxElements = 0; // 单路单次载入 UB 的最大元素数
    int64_t rounds = 0;             // 归并轮数 = ceil(log4(segNum))
};

class FfnWbA5MrgSort {
public:
    __aicore__ inline FfnWbA5MrgSort(){};

    // wsA/wsB:两块等大的 proposal 对工作区(乒乓);cntWs:各段有效元素数
    __aicore__ inline void Init(GM_ADDR wsA, GM_ADDR wsB, GM_ADDR cntWs, const A5MergeParam &param, TPipe *pipe)
    {
        param_ = param;
        pipe_ = pipe;
        const int64_t subNum = GetTaskRation() > 0 ? GetTaskRation() : 1;
        vecId_ = GetBlockIdx() * subNum + GetSubBlockIdx();
        vecNum_ = GetBlockNum() * subNum;

        const int64_t wsFloats = param_.segNum * param_.sortLenPerSeg;
        wsGm_[0].SetGlobalBuffer(reinterpret_cast<__gm__ float *>(wsA), wsFloats);
        wsGm_[1].SetGlobalBuffer(reinterpret_cast<__gm__ float *>(wsB), wsFloats);
        cntGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cntWs),
                               param_.segNum * (ONE_BLK_SIZE / static_cast<int64_t>(sizeof(int32_t))));

        // UB:MRG_LIST_NUM 路输入 + 1 路输出,均按 proposal 对宽度分配
        const int64_t loopFloats = GetSortLen<float>(param_.oneLoopMaxElements);
        pipe_->InitBuffer(inQue_, 1, loopFloats * MRG_LIST_NUM * sizeof(float) + ONE_BLK_SIZE);
        pipe_->InitBuffer(outQue_, 1, loopFloats * MRG_LIST_NUM * sizeof(float) + ONE_BLK_SIZE);
        pipe_->InitBuffer(cntBuf_, Align(param_.segNum * (ONE_BLK_SIZE / sizeof(int32_t)), sizeof(int32_t)) *
                                           sizeof(int32_t) +
                                       ONE_BLK_SIZE);
    }

    // 跑完全部归并轮次;返回最终结果所在的工作区序号(0=wsA,1=wsB)。
    // 每轮内各组由不同向量核认领,轮与轮之间由调用方 SyncAll。
    __aicore__ inline int64_t ProcessRound(int64_t round, int64_t srcIdx)
    {
        LoadSegCounts();
        const int64_t groupStride = Pow4(round + 1);  // 本轮一组覆盖的段数
        const int64_t listStride = Pow4(round);       // 组内相邻两路相隔的段数
        const int64_t groupNum = Ceil(param_.segNum, groupStride);

        for (int64_t g = vecId_; g < groupNum; g += vecNum_) {
            MergeOneGroup(g, groupStride, listStride, srcIdx);
        }
        return 1 - srcIdx;
    }

    // 最终一路的元素总数 = 各段有效数之和
    __aicore__ inline int64_t TotalValid()
    {
        LoadSegCounts();
        int64_t total = 0;
        for (int64_t s = 0; s < param_.segNum; s++) {
            total += SegCount(s);
        }
        return total;
    }

private:
    __aicore__ inline int64_t Pow4(int64_t e)
    {
        int64_t v = 1;
        for (int64_t i = 0; i < e; i++) {
            v *= MRG_LIST_NUM;
        }
        return v;
    }

    __aicore__ inline void LoadSegCounts()
    {
        if (cntLoaded_) {
            return;
        }
        cntLocal_ = cntBuf_.Get<int32_t>();
        const int64_t words = param_.segNum * (ONE_BLK_SIZE / static_cast<int64_t>(sizeof(int32_t)));
        DataCopyExtParams cp{static_cast<uint16_t>(1), static_cast<uint32_t>(words * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(cntLocal_, cntGm_, cp, pad);
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        cntLoaded_ = true;
    }

    __aicore__ inline int64_t SegCount(int64_t seg)
    {
        if (seg >= param_.segNum) {
            return 0;
        }
        return static_cast<int64_t>(cntLocal_.GetValue(seg * (ONE_BLK_SIZE / sizeof(int32_t))));
    }

    // 第 r 轮中,以 seg 为首段、跨度 listStride 的那一路的元素数 = 其覆盖段的有效数之和
    __aicore__ inline int64_t ListLength(int64_t firstSeg, int64_t listStride)
    {
        int64_t sum = 0;
        for (int64_t s = firstSeg; s < firstSeg + listStride && s < param_.segNum; s++) {
            sum += SegCount(s);
        }
        return sum;
    }

    __aicore__ inline void MergeOneGroup(int64_t group, int64_t groupStride, int64_t listStride, int64_t srcIdx)
    {
        const int64_t baseSeg = group * groupStride;
        int64_t offsets[MRG_LIST_NUM];
        int64_t remains[MRG_LIST_NUM];
        int64_t listNum = 0;
        for (int64_t i = 0; i < MRG_LIST_NUM; i++) {
            const int64_t firstSeg = baseSeg + i * listStride;
            if (firstSeg >= param_.segNum) {
                break;
            }
            const int64_t len = ListLength(firstSeg, listStride);
            offsets[listNum] = firstSeg * param_.sortLenPerSeg;
            remains[listNum] = len;
            listNum++;
        }
        if (listNum == 0) {
            return;
        }

        LocalTensor<float> inLocal = inQue_.AllocTensor<float>();
        LocalTensor<float> outLocal = outQue_.AllocTensor<float>();
        const int64_t loopFloats = GetSortLen<float>(param_.oneLoopMaxElements);
        int64_t outOffset = baseSeg * param_.sortLenPerSeg;

        int64_t allRemain = 0;
        for (int64_t i = 0; i < listNum; i++) {
            allRemain += remains[i];
        }

        while (allRemain > 0) {
            // 载入各路当前块
            uint16_t counts[MRG_LIST_NUM] = {0, 0, 0, 0};
            LocalTensor<float> lists[MRG_LIST_NUM];
            int64_t liveNum = 0;
            int64_t liveMap[MRG_LIST_NUM];
            for (int64_t i = 0; i < listNum; i++) {
                const int64_t take = Min(param_.oneLoopMaxElements, remains[i]);
                if (take <= 0) {
                    continue;
                }
                DataCopyExtParams cp{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(GetSortLen<float>(take) * sizeof(float)), 0, 0, 0};
                DataCopyPadExtParams<float> pad{false, 0, 0, 0};
                DataCopyPad(inLocal[liveNum * loopFloats], wsGm_[srcIdx][offsets[i]], cp, pad);
                lists[liveNum] = inLocal[liveNum * loopFloats];
                counts[liveNum] = static_cast<uint16_t>(take);
                liveMap[liveNum] = i;
                liveNum++;
            }
            if (liveNum == 0) {
                break;
            }
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

            uint32_t sortedNums[MRG_LIST_NUM] = {0, 0, 0, 0};
            if (liveNum == 1) {
                // 只剩一路:直接搬,不必归并
                DataCopy(outLocal, lists[0], Align(GetSortLen<float>(counts[0]), sizeof(float)));
                sortedNums[0] = counts[0];
            } else {
                MrgSortSrcList srcList = MrgSortSrcList(lists[0], lists[liveNum > 1 ? 1 : 0],
                                                        lists[liveNum > NUM_TWO ? NUM_TWO : 0],
                                                        lists[liveNum > NUM_THREE ? NUM_THREE : 0]);
                const uint16_t validBit = static_cast<uint16_t>((1U << liveNum) - 1U);
                MrgSort<float, true>(outLocal, srcList, counts, sortedNums, validBit, 1);
            }
            PipeBarrier<PIPE_V>();

            // 推进各路偏移与剩余
            int64_t produced = 0;
            for (int64_t j = 0; j < liveNum; j++) {
                const int64_t consumed = static_cast<int64_t>(sortedNums[j]);
                const int64_t i = liveMap[j];
                offsets[i] += GetSortLen<float>(consumed);
                remains[i] -= consumed;
                allRemain -= consumed;
                produced += consumed;
            }
            if (produced <= 0) {
                break;  // 保护:不应发生,避免死循环
            }

            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopyExtParams cpOut{static_cast<uint16_t>(1),
                                    static_cast<uint32_t>(GetSortLen<float>(produced) * sizeof(float)), 0, 0, 0};
            DataCopyPad(wsGm_[1 - srcIdx][outOffset], outLocal, cpOut);
            // 下一轮要用 MTE2 载入输入、用 MrgSort(向量)重写输出缓冲,两者都要等本次搬出读完。
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            outOffset += GetSortLen<float>(produced);
        }

        inQue_.FreeTensor(inLocal);
        outQue_.FreeTensor(outLocal);
    }

private:
    A5MergeParam param_;
    TPipe *pipe_ = nullptr;

    GlobalTensor<float> wsGm_[NUM_TWO];
    GlobalTensor<int32_t> cntGm_;

    TQue<QuePosition::VECIN, 1> inQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TBuf<TPosition::VECCALC> cntBuf_;
    LocalTensor<int32_t> cntLocal_;
    bool cntLoaded_ = false;

    int64_t vecId_ = 0;
    int64_t vecNum_ = 0;
};

}  // namespace FfnWbBatchingArch35
#endif  // OP_KERNEL_ARCH35_FFN_WB_A5_MRGSORT_H
