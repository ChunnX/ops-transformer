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
 * \file ffn_wb_a5_prepare.h
 * \brief phase0:等待就绪(RECV) + 把两条路径的 expert_id 归一到同一个扁平缓冲。
 */
#ifndef OP_KERNEL_ARCH35_FFN_WB_A5_PREPARE_H
#define OP_KERNEL_ARCH35_FFN_WB_A5_PREPARE_H
#include "ffn_wb_a5_context.h"
#include "kernel_operator.h"
namespace FfnWbBatchingArch35 {
using namespace AscendC;

// ===================== RECV:选择本次消费的 micro batch 和 session =====================
// 同步：只轮询入口 polling_index 指向的 micro batch，等 A 个 flag 全为 1。
// 异步：循环扫描 micro batch，遇到至少一个 flag=1 就停止，将该轮观察到的 flag
// 保存到 readyWs。后续 prepare/release 只认这份快照，不能重新查询生产者的实时 flag。
// readyWs 布局：[一个 32B 头块，低 8B 为 selectedMb][A 个 32B 块，每块低 4B 为 flag]。
// 只有 0 号向量核写快照和 polling_index；调用方负责发布前后的跨核 SyncAll。
class FfnWbA5RecvWait {
public:
    __aicore__ inline FfnWbA5RecvWait(){};

    __aicore__ inline void Init(GM_ADDR schedule_context, GM_ADDR tokenInfoBuf, const ScheduleContextInfo *ctx,
                                TPipe *pipe, GM_ADDR readyWs)
    {
        ctx_ = ctx;
        readyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(readyWs));
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
        // selectedMb 是本核的扫描游标。空轮只更新局部值，尚不回写共享 polling_index；
        // 找到数据后才发布选中项，并为下一次调用推进共享游标。
        uint64_t selectedMb = ctx_->curMicroBatchID;

        while (true) {
            LocalTensor<int32_t> flagLocal = flagQue_.AllocTensor<int32_t>();
            // 每 session 取一个 flag:块数 A、块长 4B、块间跨度 (M*F-1) 个 int32
            DataCopyExtParams cp{
                static_cast<uint16_t>(ctx_->A), static_cast<uint32_t>(sizeof(int32_t)),
                static_cast<uint32_t>((static_cast<int64_t>(ctx_->M) * descWords_ - 1) * sizeof(int32_t)), 0, 0};
            DataCopyPadExtParams<int32_t> pad{
                true, 0, static_cast<uint8_t>((ONE_BLK_SIZE - sizeof(int32_t)) / sizeof(int32_t)), 0};
            DataCopyPad(flagLocal, tokenInfoGm_[selectedMb * descWords_], cp, pad);
            flagQue_.EnQue(flagLocal);

            LocalTensor<int32_t> flags = flagQue_.DeQue<int32_t>();
            LocalTensor<float> work = workBuf_.Get<float>();
            Cast(work, flags, RoundMode::CAST_ROUND, flagElems);
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(work, work, work, flagElems);
            PipeBarrier<PIPE_V>();
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            const float readyNum = work.GetValue(0);
            // 协议要求有效 flag 只取 0/1；DataCopyPad 补出的 7 个 int32 都是 0，
            // 因而 padded 数组的 ReduceSum 等于本轮观察到的 ready session 数。
            // 此处判断描述符就绪，不判断有效 token 数；全 mask 描述符也需要被消费。
            const bool ready = ctx_->asyncRecv ? readyNum > 0 : static_cast<uint32_t>(readyNum) >= ctx_->A;
            if (ready && ctx_->asyncRecv) {
                // flags 仍保留本轮 MTE2 读取结果，Cast/ReduceSum 使用另一个 work 缓冲。
                // 将原始 padded flag 整段存入快照，不重新读取 tokenInfoGm_：否则判定后
                // 到达的 session 可能被不同核纳入不同的集合。快照不代表所有 flag 的
                // 原子瞬时读取，而是这次跨步 DMA 实际观察到的集合。
                // 搬出完成后才释放 flags，防止下一轮 AllocTensor 复用尚在读取的 UB。
                DataCopyExtParams snapshot{1, static_cast<uint32_t>(flagElems * sizeof(int32_t)), 0, 0, 0};
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                DataCopyPad(readyGm_[ONE_BLK_SIZE / sizeof(int32_t)], flags, snapshot);
                SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            }
            flagQue_.FreeTensor(flags);
            if (ready) {
                break;
            }
            if (ctx_->asyncRecv) {
                // 异步上游可能永远不给当前空 micro batch 置 flag，因此必须跳过空项。
                // 扫完 M 项仍无数据会回到起点继续等；发现任意 ready 即结束扫描。
                selectedMb = (selectedMb + 1) % ctx_->M;
            }
        }

        // 推进轮询下标:偏移取自公共契约结构,回写经 UB 整段搬运。
        LocalTensor<uint64_t> pollLocal = pollBuf_.Get<uint64_t>();
        if (ctx_->asyncRecv) {
            // 头块记录“本次消费哪一项”，共享 polling_index 则记录“下次从哪项开始”。
            // 两者含义不同：其它核必须从 readyWs 头块取本次索引，不能回读推进后的游标。
            pollLocal.SetValue(0, selectedMb);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyExtParams selected{1, static_cast<uint32_t>(sizeof(uint64_t)), 0, 0, 0};
            DataCopyPad(readyGm_, pollLocal.ReinterpretCast<int32_t>(), selected);
            SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        }
        pollLocal.SetValue(0, (selectedMb + 1) % ctx_->M);
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
    GlobalTensor<int32_t> readyGm_;
    TQue<QuePosition::VECIN, 1> flagQue_;
    TBuf<TPosition::VECCALC> workBuf_;
    TBuf<TPosition::VECCALC> pollBuf_;
    int64_t descWords_ = 0;
};

// ===================== expert_id 归一 + 握手回写 =====================
// 补位/失效标记:>= sort 的 expertStart_(1000000),排序后落到末尾并被 mask 判据剔除。
constexpr int32_t MASK_SENTINEL = 2147483647;

class FfnWbPrepareArch35 {
public:
    __aicore__ inline FfnWbPrepareArch35(){};

    // flatIdsWs:归一后的扁平 expert_id 缓冲(长度 totalLen)。
    // rowsPerLoop 由 host 按运行时 UB 容量反推(见 tiling 的 preparePerLoopRows),此处不设任何容量常数。
    __aicore__ inline void Init(GM_ADDR flatIdsWs, const ScheduleContextInfo *contextInfo, TPipe *pipe,
                                int64_t totalLen, int64_t rowsPerLoop, GM_ADDR readyWs = nullptr)
    {
        contextInfo_ = contextInfo;
        pipe_ = pipe;
        totalLen_ = totalLen;
        bsk_ = contextInfo_->BS * contextInfo_->K;
        F_ = NUM_TWO + bsk_;
        // 每 session 行在扁平缓冲中的跨度(RECV 含补位;NORM 无补位时即为 bsk)
        rowSpan_ = (contextInfo_->A > 0) ? (totalLen_ / contextInfo_->A) : bsk_;
        // 异步 ready session 可不连续，因此每轮处理一行，先查快照再决定读/释放该行。
        // 这里的 1 是选择粒度，不是核数或硬件容量；同步/NORM 仍使用 host 给出的批量行数。
        rowsPerLoop_ = contextInfo_->asyncRecv ? 1 : ((rowsPerLoop > 0) ? rowsPerLoop : 1);
        if (contextInfo_->asyncRecv) {
            readyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(readyWs));
            pipe_->InitBuffer(readyBuf_, ONE_BLK_SIZE);
        }
        if (rowsPerLoop_ > contextInfo_->A) {
            rowsPerLoop_ = contextInfo_->A;
        }
        // 按**运行时实际块数**把 session 行分段,各核只处理自己那段:
        // 各段写入 flatIds 的区间互不重叠,握手回写也按行分离,故无需跨核同步。
        const int64_t coreNum = GetBlockNum();
        const int64_t perCore = (contextInfo_->A + coreNum - 1) / coreNum;
        rowBegin_ = GetBlockIdx() * perCore;
        rowEnd_ = rowBegin_ + perCore;
        if (rowEnd_ > contextInfo_->A) {
            rowEnd_ = contextInfo_->A;
        }
        if (rowBegin_ > contextInfo_->A) {
            rowBegin_ = contextInfo_->A;
        }
        if (rowsPerLoop_ > (rowEnd_ - rowBegin_) && (rowEnd_ - rowBegin_) > 0) {
            rowsPerLoop_ = rowEnd_ - rowBegin_;
        }
        flatIdsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(flatIdsWs), totalLen_);
        pipe_->InitBuffer(que_, 1, Align(rowsPerLoop_ * rowSpan_ * sizeof(int32_t), BLOCK_BYTES));
        pipe_->InitBuffer(clrBuf_, Align(rowsPerLoop_ * BLOCK_BYTES, BLOCK_BYTES));
    }

    // NORM:expert_ids_buf 已是扁平布局,整段搬到 flatIds(单核即可,量级为 Y 个 int32)。
    __aicore__ inline void ProcessNorm(GM_ADDR expertIdsBuf)
    {
        if (rowEnd_ <= rowBegin_) {
            return;
        }
        GlobalTensor<int32_t> srcGm;
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(expertIdsBuf), totalLen_);
        const int64_t perLoopElems = rowsPerLoop_ * rowSpan_;
        const int64_t beginElem = rowBegin_ * rowSpan_;
        const int64_t endElem = (rowEnd_ * rowSpan_ > totalLen_) ? totalLen_ : rowEnd_ * rowSpan_;
        for (int64_t off = beginElem; off < endElem; off += perLoopElems) {
            const int64_t n = ((endElem - off) > perLoopElems) ? perLoopElems : (endElem - off);
            LocalTensor<int32_t> buf = que_.AllocTensor<int32_t>();
            DataCopyExtParams cp{static_cast<uint16_t>(1), static_cast<uint32_t>(n * sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
            DataCopyPad(buf, srcGm[off], cp, pad);
            SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
            DataCopyPad(flatIdsGm_[off], buf, cp);
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            que_.FreeTensor(buf);
        }
    }

    // RECV 分为读入和释放两个阶段，共用 session 切分及同一份 ready 快照：
    //   同步 + release=false：读取所有 ids，并按原逻辑立即清 ids/flag。
    //   异步 + release=false：只读选中行；未选中行只在 flatIds workspace 填无效值。
    //   异步 + release=true ：所有核 gather 完成后调用，只清选中行，不再次读取 ids。
    // 异步第一阶段不能清 flag：生产者若复用该槽，会覆盖其它核尚未 gather 的 token。
    // rowsPerLoop_ 控制本轮驻留的 id/清零区，异步固定一行以支持不连续的选择集合。
    __aicore__ inline void ProcessRecv(GM_ADDR tokenInfoBuf, bool release = false)
    {
        if (rowEnd_ <= rowBegin_) {
            return;
        }
        GlobalTensor<int32_t> tokenInfoGm;
        tokenInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tokenInfoBuf),
                                    contextInfo_->A * contextInfo_->M * F_);
        const int64_t base = contextInfo_->curMicroBatchID * F_;
        const int64_t rowStride = contextInfo_->M * F_; // 相邻 session 在 token_info 中的跨度
        LocalTensor<int32_t> clr = clrBuf_.Get<int32_t>();

        for (int64_t r0 = rowBegin_; r0 < rowEnd_; r0 += rowsPerLoop_) {
            const int64_t rows = ((rowEnd_ - r0) > rowsPerLoop_) ? rowsPerLoop_ : (rowEnd_ - r0);
            LocalTensor<int32_t> buf = que_.AllocTensor<int32_t>();

            // 未选中行即使含有看似合法的 expert_id，也属于未发布/迟到的数据，不能读取。
            // 仅在私有 workspace 中填 sentinel，保持原 session 下标与 gather 解码一致；
            // release 阶段则直接跳过，禁止清除未选中行的 ids、flag 或 layer_id。
            if (contextInfo_->asyncRecv && !IsSelected(r0)) {
                if (!release) {
                    Duplicate<int32_t>(buf, MASK_SENTINEL, rowSpan_);
                    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                    DataCopyExtParams masked{1, static_cast<uint32_t>(rowSpan_ * sizeof(int32_t)), 0, 0, 0};
                    DataCopyPad(flatIdsGm_[r0 * rowSpan_], buf, masked);
                    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
                }
                que_.FreeTensor(buf);
                continue;
            }

            if (!release) {
                // 取 ids:每块 bsk 个 int32,块间跳过 flag/layer 与其余 micro batch;
                // 右侧补位由 DataCopyPad 按 BsKPaddingCount 填 MASK_SENTINEL,使每行在 UB 中占 rowSpan_
                DataCopyExtParams inParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(bsk_ * sizeof(int32_t)),
                                           static_cast<uint32_t>((rowStride - bsk_) * sizeof(int32_t)), 0, 0};
                DataCopyPadExtParams<int32_t> inPad{true, 0, static_cast<uint8_t>(contextInfo_->BsKPaddingCount),
                                                    MASK_SENTINEL};
                DataCopyPad(buf, tokenInfoGm[base + r0 * rowStride + NUM_TWO], inParams, inPad);
                SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);

                DataCopyExtParams outParams{static_cast<uint16_t>(1),
                                            static_cast<uint32_t>(rows * rowSpan_ * sizeof(int32_t)), 0, 0, 0};
                DataCopyPad(flatIdsGm_[r0 * rowSpan_], buf, outParams);

                // 握手回写前必须等上面的搬出真正读完 buf:下面要就地把 buf 覆盖成回写内容。
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            }
            // flatIds 已搬出，排序/gather 可以使用；原描述符和 token 槽仍由本次消费持有。
            // 与后续 release 共用快照，不会因生产者在其它行置位而扩大释放范围。
            if (contextInfo_->asyncRecv && !release) {
                que_.FreeTensor(buf);
                continue;
            }
            // 回写值处处相同(ids 全 MASK_SENTINEL、flag 全 0),故按连续块读出即可,与行内跨度无关
            Duplicate<int32_t>(buf, MASK_SENTINEL, rows * rowSpan_);
            Duplicate<int32_t>(clr, 0, rows * (BLOCK_BYTES / static_cast<int64_t>(sizeof(int32_t))));
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopyExtParams clrIds{static_cast<uint16_t>(rows), static_cast<uint32_t>(bsk_ * sizeof(int32_t)), 0,
                                     static_cast<uint32_t>((rowStride - bsk_) * sizeof(int32_t)), 0};
            DataCopyExtParams clrFlag{static_cast<uint16_t>(rows), static_cast<uint32_t>(sizeof(int32_t)), 0,
                                      static_cast<uint32_t>((rowStride - 1) * sizeof(int32_t)), 0};
            DataCopyPad(tokenInfoGm[base + r0 * rowStride + NUM_TWO], buf, clrIds);
            if (contextInfo_->asyncRecv) {
                // 先完成 expert_ids 的失效回写，再清 flag。flag=0 是释放槽位的握手信号；
                // 不能让生产者看到空闲后写入的新 ids 又被本次迟到的清理覆盖。
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
            }
            DataCopyPad(tokenInfoGm[base + r0 * rowStride], clr, clrFlag);
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            que_.FreeTensor(buf);
        }
    }

private:
    // 从 workspace 快照读取，而不是从 token_info 读取实时 flag。
    // (row + 1) 跳过头块；每个 session 的 flag 独占一个块，与 Waiter 的 DMA padding 一致。
    // MTE2_S 确保 DMA 已完成，随后标量 GetValue(0) 才能用于分支判断。
    __aicore__ inline bool IsSelected(int64_t row)
    {
        LocalTensor<int32_t> flag = readyBuf_.Get<int32_t>();
        DataCopyExtParams cp{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(flag, readyGm_[(row + 1) * (ONE_BLK_SIZE / sizeof(int32_t))], cp, pad);
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        return flag.GetValue(0) == 1;
    }

    GlobalTensor<int32_t> readyGm_;
    TBuf<TPosition::VECIN> readyBuf_;
    const ScheduleContextInfo *contextInfo_ = nullptr;
    TPipe *pipe_ = nullptr;
    GlobalTensor<int32_t> flatIdsGm_;
    TQue<QuePosition::VECCALC, 1> que_;
    TBuf<QuePosition::VECCALC> clrBuf_;
    int64_t totalLen_ = 0;
    int64_t bsk_ = 0;
    int64_t rowSpan_ = 0;
    int64_t rowsPerLoop_ = 0;
    int64_t rowBegin_ = 0;
    int64_t rowEnd_ = 0;
    int64_t F_ = 0;
};
} // namespace FfnWbBatchingArch35
#endif // OP_KERNEL_ARCH35_FFN_WB_A5_PREPARE_H
