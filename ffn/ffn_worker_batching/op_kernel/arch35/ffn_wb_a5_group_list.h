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
 * \file ffn_wb_a5_group_list.h
 * \brief arch35(A5) group_list:从有序 expert_id 统计每专家 token 数,压成 [expert_id, tokenNum] 行。
 *
 * 计数用 SIMT(asc_vf_call):序列有序,故"某专家的首元素"与"某专家的末元素"各只被一个线程命中,
 * 两趟分别记录首下标与算出计数,期间不需要任何原子操作(与 moe_v3_expert_tokens_count.h 同法)。
 * 跨核汇总用 SetAtomicAdd + DataCopyPad,把各核的局部计数加到 GM 计数区。
 *
 * 写出严格走整段搬运:行在 UB 拼好后按 host 给的块长 DataCopyPad 一次搬出,未使用的专家行
 * 用 Duplicate 填 0 随块带出。**不对 GM 做标量 SetValue**——历史缺陷正是"逐行 SetValue +
 * 末尾一次 SINGLE_CACHE_LINE 刷回"导致只有首条 cache line(4 行)落盘。
 */
#ifndef OP_KERNEL_ARCH35_FFN_WB_A5_GROUP_LIST_H
#define OP_KERNEL_ARCH35_FFN_WB_A5_GROUP_LIST_H
#include "ffn_wb_a5_context.h"
#include "simt_api/asc_simt.h"

namespace FfnWbBatchingArch35 {
using namespace AscendC;

// 单次 asc_vf_call 的线程数:dim3 各维乘积的上限即为此值(SIMT 编程模型约束)。
constexpr int64_t SIMT_THREAD_NUM = 2048;

struct A5GroupListParam {
    int64_t segNum = 0;         // 计数切片数
    int64_t perSegElements = 0; // 每片元素数
    int64_t rowsPerLoop = 0;    // 写出时每块拼多少行(每行 16B)
};

// 第一趟:记录每个专家在本片内的首元素下标。有序序列中"与前一个不同"者即为该专家首元素。
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void FfnWbExpertFirstIdxSimt(
    int32_t elemNum, int32_t expertNum, __gm__ int32_t *ids, __local_mem__ int32_t *slot)
{
    for (auto i = static_cast<int32_t>(threadIdx.x); i < elemNum; i += static_cast<int32_t>(blockDim.x)) {
        auto cur = ids[i];
        if (cur >= expertNum) {
            break;
        }
        auto prev = (i == 0) ? -1 : ids[i - 1];
        if (cur != prev) {
            slot[cur] = i;
        }
    }
}

// 第二趟:在每个专家的末元素处算出计数(末下标 + 1 - 首下标),就地覆盖首下标。
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void FfnWbExpertCountSimt(
    int32_t elemNum, int32_t expertNum, __gm__ int32_t *ids, __local_mem__ int32_t *slot)
{
    for (auto i = static_cast<int32_t>(threadIdx.x); i < elemNum; i += static_cast<int32_t>(blockDim.x)) {
        auto cur = ids[i];
        if (cur >= expertNum) {
            break;
        }
        if (i == elemNum - 1 || cur != ids[i + 1]) {
            slot[cur] = i + 1 - slot[cur];
        }
    }
}

class FfnWbA5GroupList {
public:
    __aicore__ inline FfnWbA5GroupList(){};

    // sortedIdsWs:有序 expert_id;cntWs:每专家计数汇总区(int32,expertNum 个);groupList:输出
    __aicore__ inline void Init(GM_ADDR sortedIdsWs, GM_ADDR cntWs, GM_ADDR groupList,
                                const A5GroupListParam &param, const ScheduleContextInfo *ctx, TPipe *pipe)
    {
        param_ = param;
        ctx_ = ctx;
        pipe_ = pipe;
        const int64_t subNum = GetTaskRation() > 0 ? GetTaskRation() : 1;
        vecId_ = GetBlockIdx() * subNum + GetSubBlockIdx();
        vecNum_ = GetBlockNum() * subNum;
        expertNum_ = ctx->expertNum;
        totalLen_ = ctx->validGatherIdxLength;

        idsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sortedIdsWs), totalLen_);
        cntGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cntWs), expertNum_);
        groupListGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(groupList), expertNum_ * NUM_TWO);

        pipe_->InitBuffer(cntOutQue_, 1, Align(expertNum_, sizeof(int32_t)) * sizeof(int32_t) + ONE_BLK_SIZE);
        pipe_->InitBuffer(cntInQue_, 1, Align(expertNum_, sizeof(int32_t)) * sizeof(int32_t) + ONE_BLK_SIZE);
        pipe_->InitBuffer(rowBuf_, param_.rowsPerLoop * NUM_TWO * sizeof(int64_t) + ONE_BLK_SIZE);
    }

    // 计数区清零:汇总用原子累加,必须先归零(由 0 号核做,调用方随后 SyncAll)。
    __aicore__ inline void ClearCount()
    {
        if (vecId_ != 0) {
            return;
        }
        InitGlobalMemory(cntGm_, expertNum_, static_cast<int32_t>(0));
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }

    // 各核对认领的切片计数,原子累加到 GM 计数区。
    __aicore__ inline void Count()
    {
        for (int64_t seg = vecId_; seg < param_.segNum; seg += vecNum_) {
            LocalTensor<int32_t> cntLocal = cntOutQue_.AllocTensor<int32_t>();
            const int64_t begin = seg * param_.perSegElements;
            const int64_t len = (begin >= totalLen_) ? 0 : Min(param_.perSegElements, totalLen_ - begin);
            if (len <= 0) {
                cntOutQue_.FreeTensor(cntLocal);
                continue;
            }
            Duplicate(cntLocal, 0, expertNum_);

            // GetPhyAddr 返回 const 指针，SIMT 入参需可写裸指针，故按参照实现用 C 风格转换。
            __gm__ int32_t *idsAddr = (__gm__ int32_t *)idsGm_[begin].GetPhyAddr();
            __local_mem__ int32_t *slotAddr = (__local_mem__ int32_t *)cntLocal.GetPhyAddr();
            asc_vf_call<FfnWbExpertFirstIdxSimt>(dim3{SIMT_THREAD_NUM, 1, 1}, static_cast<int32_t>(len),
                                                 static_cast<int32_t>(expertNum_), idsAddr, slotAddr);
            asc_vf_call<FfnWbExpertCountSimt>(dim3{SIMT_THREAD_NUM, 1, 1}, static_cast<int32_t>(len),
                                              static_cast<int32_t>(expertNum_), idsAddr, slotAddr);

            // SIMT 写完 UB 与后续 MTE3 搬出之间必须建立依赖:仅靠 S_MTE3 标志不覆盖 SIMT 的写,
            // 会把尚未落盘的缓冲搬出去(表现为计数是残留的位型垃圾)。经队列 EnQue/DeQue 建立依赖。
            cntOutQue_.EnQue<int32_t>(cntLocal);
            LocalTensor<int32_t> cntOut = cntOutQue_.DeQue<int32_t>();
            // 搬出前等的是**向量写**(Duplicate 与 SIMT 的写),不是标量写:
            // 用 S_MTE3 会把尚未落盘的缓冲搬走,各段累加后表现为上亿的垃圾计数。
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopyExtParams cp{static_cast<uint16_t>(1), static_cast<uint32_t>(expertNum_ * sizeof(int32_t)), 0, 0,
                                 0};
            SetAtomicAdd<int32_t>();
            DataCopyPad(cntGm_, cntOut, cp);
            SetAtomicNone();
            SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
            cntOutQue_.FreeTensor(cntOut);
        }
    }

    // 0 号核把计数压成 [expert_id, tokenNum] 行并整块搬出;未用到的专家为 [0,0]。
    __aicore__ inline void Emit()
    {
        if (vecId_ != 0) {
            return;
        }
        LocalTensor<int32_t> cntLocal = cntInQue_.AllocTensor<int32_t>();
        DataCopyExtParams cpIn{static_cast<uint16_t>(1), static_cast<uint32_t>(expertNum_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(cntLocal, cntGm_, cpIn, pad);
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

        LocalTensor<int64_t> rows = rowBuf_.Get<int64_t>();
        LocalTensor<int32_t> rows32 = rows.template ReinterpretCast<int32_t>();
        // 每块从整块清零开始:清零起点是缓冲区首地址(天然对齐),未填的行随块带出即为契约要求的 [0,0]。
        // 若改成"只清尾部",起点会落在 fill*16 字节处,奇数行时不满足向量指令的对齐要求。
        ClearRows(rows32);
        int64_t outIdx = 0;
        int64_t fill = 0;
        for (int64_t e = 0; e < expertNum_; e++) {
            const int64_t cnt = static_cast<int64_t>(cntLocal.GetValue(e));
            if (cnt <= 0) {
                continue;
            }
            rows.SetValue(fill * NUM_TWO, e);
            rows.SetValue(fill * NUM_TWO + 1, cnt);
            fill++;
            if (fill == param_.rowsPerLoop) {
                FlushRows(rows, outIdx, fill);
            }
        }
        // 收尾:未填的行已在块清零时置 0,直接整块搬出,直到写满 expertNum 行
        while (outIdx < expertNum_) {
            const int64_t n = Min(param_.rowsPerLoop, expertNum_ - outIdx);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyExtParams cpOut{static_cast<uint16_t>(1),
                                    static_cast<uint32_t>(n * NUM_TWO * sizeof(int64_t)), 0, 0, 0};
            DataCopyPad(groupListGm_[outIdx * NUM_TWO], rows, cpOut);
            SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
            outIdx += n;
            fill = 0;
            ClearRows(rows32);
        }
        cntInQue_.FreeTensor(cntLocal);
    }

private:
    __aicore__ inline void FlushRows(const LocalTensor<int64_t> &rows, int64_t &outIdx, int64_t &fill)
    {
        if (fill <= 0) {
            return;
        }
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyExtParams cp{static_cast<uint16_t>(1), static_cast<uint32_t>(fill * NUM_TWO * sizeof(int64_t)), 0, 0,
                             0};
        DataCopyPad(groupListGm_[outIdx * NUM_TWO], rows, cp);
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        outIdx += fill;
        fill = 0;
        // 复位缓冲:不复位的话,下一块未填满的行会带出上一块的残留值。
        ClearRows(rows.template ReinterpretCast<int32_t>());
    }

    // 整块清零:起点为缓冲区首地址,满足向量指令的对齐要求。
    __aicore__ inline void ClearRows(const LocalTensor<int32_t> &rows32)
    {
        // 清零是向量写:既要等前面的标量填充,也要等上一块的搬出真正读完。
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
        Duplicate<int32_t>(rows32, 0, static_cast<int32_t>(param_.rowsPerLoop * NUM_FOUR));
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    }

    A5GroupListParam param_;
    const ScheduleContextInfo *ctx_ = nullptr;
    TPipe *pipe_ = nullptr;

    GlobalTensor<int32_t> idsGm_;
    GlobalTensor<int32_t> cntGm_;
    GlobalTensor<int64_t> groupListGm_;

    TQue<QuePosition::VECOUT, 1> cntOutQue_;  // 计数结果需被 MTE3 搬出:位置必须是 VECOUT
    TQue<QuePosition::VECIN, 1> cntInQue_;    // 回读计数用:位置是 VECIN
    TBuf<TPosition::VECCALC> rowBuf_;

    int64_t vecId_ = 0;
    int64_t vecNum_ = 0;
    int64_t expertNum_ = 0;
    int64_t totalLen_ = 0;
};

}  // namespace FfnWbBatchingArch35
#endif  // OP_KERNEL_ARCH35_FFN_WB_A5_GROUP_LIST_H
