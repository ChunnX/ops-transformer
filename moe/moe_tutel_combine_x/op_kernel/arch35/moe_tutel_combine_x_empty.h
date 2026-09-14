/**
 * =============================================================================
 * op_kernel/arch35/moe_tutel_combine_x_empty.h
 * =============================================================================
 * Role: EMPTY branch (tilingKey = 1) kernel template for MoeTutelCombineX on
 *        arch35 (Ascend 950): `template <typename T> class
 *        MoeTutelCombineXEmptyKernel` — empty-tensor / all-zero-output
 *        short-circuit (any of B*S*EC*H == 0).
 *
 * Two sub-paths:
 *   A (safe no-op): outTotal = B*S*H == 0 (S=0 / H=0 / B=0) — the output is
 *      an empty tensor; the kernel returns early, zero IO.
 *   B (zero-fill): outTotal > 0 (only EC=0 degeneration: C=0 makes every
 *      entry invalid) — Duplicate(0) a constant zero template once, then
 *      DataCopyPad it out tileNum times (byte-precise tail tile).
 *
 * Sync: no SetFlag/WaitFlag pairs at all — zeroBuf is written once (V) then
 *   read-only (MTE3); the single crossing is serialized by one
 *   PipeBarrier<PIPE_ALL> right after the Duplicate. Single core
 *   (host SetBlockDim(1)), no cross-core sync.
 *
 * Note: TPipe is passed in by the kernel entry — unlike the normal-path
 *   class, this class does not own a pipe.
 *
 * Operator name variants:
 *   - PascalCase:   MoeTutelCombineX
 *   - snake_case:   moe_tutel_combine_x
 *   - UPPER_SNAKE:  MOE_TUTEL_COMBINE_X
 * =============================================================================
 */

#ifndef MOE_TUTEL_COMBINE_X_EMPTY_H_
#define MOE_TUTEL_COMBINE_X_EMPTY_H_

#include "kernel_operator.h"
#include "moe_tutel_combine_x_tiling_struct.h"

template <typename T>
class MoeTutelCombineXEmptyKernel {
public:
    __aicore__ inline MoeTutelCombineXEmptyKernel() {}

    // Init: 子路径分流 + GM 绑定 + TBuf 分配；四输入 GM 地址接收但不消费
    // （EC=0 => y_grad 0 元素、C=0 与 gates/indices/locations 取值无关，输入一律不绑定、不读取）
    __aicore__ inline void Init(GM_ADDR yGrad, GM_ADDR gates, GM_ADDR indices, GM_ADDR locations, GM_ADDR xGrad,
                                const MoetutelcombinexTilingData *td, AscendC::TPipe *pipe)
    {
        (void)yGrad;
        (void)gates;
        (void)indices;
        (void)locations;
        pipe_ = pipe;
        // 仅消费三字段：outTotal = B*S*H（int64 乘积，子路径分流判据）
        outTotal_ = td->batchSize * td->samples * td->hidden;
        if (outTotal_ == 0) {
            return; // 子路径 A（安全空转）：空 Tensor 的 GM 指针可能为空，
                    //   此处不绑定 GM、不分配 UB（Process 顶部同判早退）
        }
        // 子路径 B（零填充）：仅绑定输出 x_grad
        gmXgrad_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(xGrad));
        // TBuf 分配（单路 zeroBuf，kZeroTileElems * sizeof(T) <= 245760 B）
        pipe_->InitBuffer(zeroBuf_, static_cast<uint32_t>(kZeroTileElems * static_cast<int64_t>(sizeof(T))));
    }

    // Process — 子路径 A 早退 / 子路径 B 一次填零 + 串行写出循环
    // 对应数学退化推论：EC=0 => C=0 => 所有条目无效 => x_grad = 0；
    //   S=0 / H=0 / B=0 => 输出空 Tensor（安全空转）。
    __aicore__ inline void Process()
    {
        if (outTotal_ == 0) {
            return; // 子路径 A（安全空转）：零计算、零 IO，不触碰 GM / UB
        }
        // -- 子路径 B：零填充 --
        // Compute：常量零源一次生成（P=1）
        // 执行前: 持有=[]            // zeroBuf 已分配（Init）、未写
        // 执行中: 持有=[zeroBuf]     <- P=1 峰值（Duplicate 常量广播，整条流水仅此一次 V 操作）
        // 执行后: 持有=[zeroBuf]     // 内容恒零，此后只读——CopyOut 循环永不重写，无 WAR/WAW
        AscendC::LocalTensor<T> zero = zeroBuf_.Get<T>();
        AscendC::Duplicate(zero, static_cast<T>(0), static_cast<int32_t>(kZeroTileElems));
        AscendC::PipeBarrier<PIPE_ALL>(); // V->MTE3 唯一 crossing 的一次性串行化
        // CopyOut：tileNum 次整块写出，末块 tailLength 字节精度收尾
        const int64_t tileNum = (outTotal_ + kZeroTileElems - 1) / kZeroTileElems; // CeilDiv
        const int64_t tailLength = outTotal_ - (tileNum - 1) * kZeroTileElems;     // 尾块
        for (int64_t i = 0; i < tileNum; ++i) {
            const int64_t n = (i < tileNum - 1) ? kZeroTileElems : tailLength;
            CopyOutZeroTile(i * kZeroTileElems, n);
        }
    }

private:
    // CopyOutZeroTile — zeroBuf 常量零源整块写出（UB -> GM，DataCopyPad 字节精度）。
    __aicore__ inline void CopyOutZeroTile(int64_t gmOffset, int64_t count)
    {
        AscendC::DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(T))); // 有效字节，不要求 32B 对齐
        params.srcStride = 0;                                                             // UB 侧 gap=0
        params.dstStride = 0; // GM 侧 dense（输出扁平一维线性铺）
        AscendC::DataCopyPad(gmXgrad_[gmOffset], zeroBuf_.Get<T>(), params);
        // UB->GM 方向无 padParams；末块 count=tailLength 字节精度收尾
    }

    // UB 设计预算 245760 B（240 KiB）封顶；kZeroTileElems 按 dtype 实例化期常量
    //   fp32: 61440 / fp16*bf16: 122880
    static constexpr int64_t kUbBudgetBytes = 245760;
    static constexpr int64_t kZeroTileElems = kUbBudgetBytes / static_cast<int64_t>(sizeof(T));

    int64_t outTotal_ = 0; // B*S*H（三字段乘积）
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::GlobalTensor<T> gmXgrad_;                   // 输出 x_grad（唯一绑定的 GM 张量）
    AscendC::TBuf<AscendC::TPosition::VECCALC> zeroBuf_; // 常量零源
};

#endif // MOE_TUTEL_COMBINE_X_EMPTY_H_
