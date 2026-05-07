/**
 * \file gmmdsq_kernel.h
 * \brief Grouped MatMul with Dynamic Scale Quantization Kernel
 *
 * 设计原语: E:{matM:[matN:[C0]->matM_i:[V0]]}
 * 核启动模式: 1c2v
 * CV 序列: C0(matmul int8×int8→int32) → V0(Cast+dequant+SwiGLU+quantize)
 *
 * Stage 0 (AIC): C0, matM×matN 2D 多核切分
 * Stage 1 (AIV): V0, matM 多核切分 + matM_i 核内切分(vec_chunk=2)
 */

#ifndef __GMMDSQ_KERNEL_H_
#define __GMMDSQ_KERNEL_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"

namespace Gmmdsq {

using namespace AscendC;
using namespace matmul;

// ============================================================================
// [SECTION: TYPE_DEFS] 类型定义
// ============================================================================

// -------------------- Matmul 类型 --------------------
// [SECTION: MATMUL_TYPES]
// int8 × int8 → int32, B transposed (isTransposeB=true)
using aT = MatmulType<TPosition::GM, CubeFormat::ND, int8_t>;          // x [M, K]
using bT = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, true>;    // weight^T [N, K]
using cT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;         // output [M, N]
using MT = matmul::MatmulImpl<aT, bT, cT>;
// [END_SECTION: MATMUL_TYPES]

// -------------------- 自定义类型别名 --------------------
using XMAT_T = int8_t;
using WMAT_T = int8_t;
using SCALE_T = float;
using OUT_T = int8_t;

// -------------------- Init 参数结构体 --------------------
// [SECTION: INIT_PARAMS]
struct InitParams {
    GM_ADDR x;                  // [M, K] int8
    GM_ADDR weight;             // tensorList [E, K, N] int8
    GM_ADDR weightScale;        // tensorList [E, N] float
    GM_ADDR xScale;             // [M] float
    GM_ADDR groupList;          // [E] int64
    GM_ADDR quantOutput;        // [M, N/2] int8
    GM_ADDR quantScaleOutput;   // [M] float
    GM_ADDR workspace;
    TPipe *tPipe;
    GmmdsqTilingData *tilingData;
};
// [END_SECTION: INIT_PARAMS]

// -------------------- 多核切分结构体 --------------------
// [SECTION: S1_MULTI_CORE_TILING]
struct MatMTilingParams {
    uint64_t totalMatM;     // 总 M (=matM from tiling)
    uint32_t numMCores;     // M 轴核数
    uint32_t singleMatM;    // 每 M-block 行数
    uint32_t numMBlocks;    // M block 总数
};

struct MatNTilingParams {
    uint64_t totalMatN;     // 总 N (=matN from tiling)
    uint32_t numNCores;     // N 轴核数
    uint32_t singleMatN;    // 每 N-block 列数
    uint32_t numNBlocks;    // N block 总数
};

struct MatMKernalIter {
    uint32_t mIdxBegin;
    uint32_t mIdxEnd;
};
// [END_SECTION: S1_MULTI_CORE_TILING]

// -------------------- 编译时常量 --------------------
// [SECTION: COMPILE_CONSTANTS]
static constexpr uint64_t SYNC_MODE = 0x2;

// 同步标志
// Stage 0 C0 → Stage 1 V0: per M-block N-axis barrier
static constexpr uint64_t SYNC_S0_L1_C2V_0 = 0x1;

// V0 vec_chunk (matM_i intra-core)
static constexpr uint32_t DEFAULT_VEC_CHUNK_SIZE = 2;  // rows per chunk
// [END_SECTION: COMPILE_CONSTANTS]

// [END_SECTION: TYPE_DEFS]

// ============================================================================
// [SECTION: KERNEL_CLASS] Kernel 类定义
// ============================================================================

template <class T, class P = void>
class GmmdsqKernel {
public:
    // [SECTION: CONSTRUCTOR]
    __aicore__ inline GmmdsqKernel(MT &matmul) : mm_(matmul) {}
    // [END_SECTION: CONSTRUCTOR]

    __aicore__ inline void Init(const InitParams &params);
    __aicore__ inline void InitLocalBuffers();  // AIV 专用初始化
    __aicore__ inline void Process();

private:
    // ========================================================================
    // [SECTION: MEMBERS] 成员变量
    // ========================================================================

    // -------------------- GlobalTensor --------------------
    // [SECTION: GLOBAL_TENSORS]
    GlobalTensor<int8_t> xGm_;              // x [M, K]
    GlobalTensor<int8_t> weightGm_;         // weight tensorList current element [K, N]
    GlobalTensor<float> weightScaleGm_;     // weightScale tensorList current element [N]
    GlobalTensor<float> xScaleGm_;          // xScale [M]
    GlobalTensor<int64_t> groupListGm_;     // groupList [E]
    GlobalTensor<int8_t> quantOutputGm_;    // quantOutput [M, N/2]
    GlobalTensor<float> quantScaleOutputGm_; // quantScaleOutput [M]
    GlobalTensor<int32_t> workspaceGm_;     // workspace for C0→V0 int32 results
    // [END_SECTION: GLOBAL_TENSORS]

    // -------------------- Queue（Step 5 生成）--------------------
    // [SECTION: S6_QUEUES]
    // Step 5 根据 SSA 生命周期分析生成
    TQue<QuePosition::VECIN, 1> int32InQueue_;    // 占位符, Step 5 覆盖
    TQue<QuePosition::VECOUT, 1> int8OutQueue_;    // 占位符, Step 5 覆盖
    // [END_SECTION: S6_QUEUES]

    // -------------------- Buffer（Step 5 生成）--------------------
    // [SECTION: S6_BUFFERS]
    // Step 5 根据 SSA 生命周期分析生成
    TBuf<TPosition::VECCALC> fp32Buf_;       // 占位符, Step 5 覆盖
    TBuf<TPosition::VECCALC> weightScaleBuf_; // 占位符, Step 5 覆盖
    // [END_SECTION: S6_BUFFERS]

    // -------------------- LocalTensor（Step 5 生成）--------------------
    // [SECTION: S6_LOCAL_TENSORS]
    // Step 5 根据 SSA 映射生成
    // 占位符, Step 5 覆盖
    // [END_SECTION: S6_LOCAL_TENSORS]

    // -------------------- TilingData --------------------
    // [SECTION: TILING_DATA]
    const GmmdsqTilingData *tiling_;

    // From tiling
    uint32_t coreNum_;
    int64_t matM_;          // 总 M
    int64_t matK_;          // K 维度
    int64_t matN_;          // N 维度
    uint32_t groupNum_;     // E (分组数)

    // Step 1: 多核切分参数
    uint32_t numMCores_;    // M 轴核数
    uint32_t numNCores_;    // N 轴核数
    uint32_t singleMatM_;   // 每 M-block 行数
    uint32_t singleMatN_;   // 每 N-block 列数
    uint32_t numMBlocks_;   // M block 总数
    uint32_t numNBlocks_;   // N block 总数

    // 核内切分 (V0 vec_chunk)
    // DEFAULT_VEC_CHUNK_SIZE = 2 (constexpr, 不放 TilingData)

    // Step 3: Stride 参数
    // [SECTION: S4_STRIDE_DATA]
    uint64_t stride_x_M_;      // = matK (x [M,K] 行 stride)
    uint64_t stride_ws_M_;     // = matN (workspace [M,N] int32 行 stride)
    uint64_t stride_out_M_;    // = matN / 2 (quantOutput [M,N/2] int8 行 stride)
    // [END_SECTION: S4_STRIDE_DATA]
    // [END_SECTION: TILING_DATA]

    // -------------------- 运行时状态 --------------------
    // [SECTION: RUNTIME_VARS]
    TPipe *pipe_;
    uint32_t coreIdx_;
    uint32_t mIdx_;     // 当前核在 M 轴的索引
    uint32_t nIdx_;     // 当前核在 N 轴的索引 (AIC only)
    // [END_SECTION: RUNTIME_VARS]

    // -------------------- Matmul 实例 --------------------
    // [SECTION: MATMUL_INSTANCE]
    MT &mm_;
    // [END_SECTION: MATMUL_INSTANCE]

    // [END_SECTION: MEMBERS]

    // ========================================================================
    // [SECTION: SUBPROCESS_DECL] 子流程声明
    // ========================================================================
    // Stage 0: C0 (AIC, matM×matN 2D)
    __aicore__ inline void S0_C0Process(uint32_t groupIdx,
                                         uint64_t mOffset, uint64_t mSize,
                                         uint32_t nBlockIdx);
    // Stage 1: V0 (AIV, matM + matM_i intra-core)
    __aicore__ inline void S0_V0Process(uint32_t groupIdx,
                                         uint64_t mOffset, uint64_t mSize);
    // [END_SECTION: SUBPROCESS_DECL]

    // ========================================================================
    // [SECTION: HELPER_DECL] 辅助方法声明
    // ========================================================================
    __aicore__ inline void RunStage0();
    __aicore__ inline uint32_t FindGroupIdx(uint64_t row);
    // [END_SECTION: HELPER_DECL]
};

// ============================================================================
// [SECTION: S1_IMPL] Step 1 实现：多核切分
// ============================================================================

// -------------------- 辅助方法: 查找行所属 group --------------------
// [SECTION: HELPER_IMPL]
template <class T, class P>
__aicore__ inline uint32_t GmmdsqKernel<T, P>::FindGroupIdx(uint64_t row)
{
    // [SECTION: V0_P0_S4_OFFSET]
    // Step 4: 实现 groupList 二分查找或线性查找
    // 输入: row (全局行号)
    // 输出: groupIdx
    // [ASCEND_MODIFY] Step 4 填充具体实现
    return 0;
    // [END_SECTION]
}
// [END_SECTION: HELPER_IMPL]

// -------------------- RunStage0: 主循环框架 --------------------
// 设计原语: E:{matM:[matN:[C0]->matM_i:[V0]]}
//
// L0: E 实例轴 → group 循环
// L1: matM 多核切分 → M-block 循环
//   Stage 0: matN:[C0] (AIC, N-tile 并行)
//   → SYNC (N-axis barrier)
//   Stage 1: matM_i:[V0] (AIV, vec_chunk)
// [SECTION: S1_LOOP_BODY]
template <class T, class P>
__aicore__ inline void GmmdsqKernel<T, P>::RunStage0()
{
    // ====================================================================
    // L0: E 实例轴 — 遍历 group
    // ====================================================================
    for (uint32_t groupIdx = 0; groupIdx < groupNum_; ++groupIdx) {

        // [SECTION: GROUP_BOUNDARY]
        uint64_t groupStart = (groupIdx == 0) ? 0 :
            (uint64_t)(groupListGm_.GetValue(groupIdx - 1));
        uint64_t groupEnd = (uint64_t)(groupListGm_.GetValue(groupIdx));
        uint64_t currentM = groupEnd - groupStart;

        // 绑定当前 group 的 weight 和 weightScale (tensorList)
        // [ASCEND_MODIFY] tensorList 动态绑定由 tiling 层实现
        // weightGm_.SetGlobalBuffer(...)
        // weightScaleGm_.SetGlobalBuffer(...)
        // [END_SECTION: GROUP_BOUNDARY]

        // ====================================================================
        // L1: matM 多核切分 — 每个 M-block
        // ====================================================================
        // AIC: mIdx = coreIdx / numNCores_, 按 numMCores_ 步进
        // AIV: mIdx = coreIdx, 按 numMCores_ 步进 (1c2v 核独立分配)
        uint32_t assignedMStart = mIdx_;                    // 当前核的 M 起始 block
        uint32_t assignedMEnd = (currentM + singleMatM_ - 1) / singleMatM_;  // 当前 group 的 M-block 总数

        for (uint32_t mBlock = assignedMStart; mBlock < assignedMEnd; mBlock += numMCores_) {
            // [SECTION: MBLOCK_OFFSET]
            uint64_t mOffset = (uint64_t)mBlock * singleMatM_;
            uint64_t mSize = ((uint64_t)singleMatM_ < currentM - mOffset)
                             ? (uint64_t)singleMatM_ : (currentM - mOffset);
            // [END_SECTION: MBLOCK_OFFSET]

            // ================================================================
            // Stage 0: C0 — AIC only, matN 多核切分
            // ================================================================
            if ASCEND_IS_AIC {
                // [SECTION: N_RANGE_CALC]
                // Step 4: 计算当前 M-block 内的 N-block 范围
                //   nIdx_ 是全局核 N 轴索引, 需限制在当前 M_i 行内
                //   startIdx = nIdx_  (该核负责的首个 N-block)
                //   endIdx = numNBlocks_  (当前 M-block 的 N-block 总数)
                // [END_SECTION: N_RANGE_CALC]
                uint32_t nStartIdx = nIdx_;
                uint32_t nEndIdx = numNBlocks_;

                // L2: matN 多核切分 — 遍历当前核在 M-block 内负责的 N-tile
                for (uint32_t nBlock = nStartIdx; nBlock < nEndIdx; nBlock += numNCores_) {
                    // [SECTION: C0_NBLOCK_OFFSET]
                    uint64_t nOffset = (uint64_t)nBlock * singleMatN_;
                    uint64_t nSize = ((uint64_t)singleMatN_ < (uint64_t)matN_ - nOffset)
                                     ? (uint64_t)singleMatN_ : ((uint64_t)matN_ - nOffset);
                    // [END_SECTION: C0_NBLOCK_OFFSET]

                    S0_C0Process(groupIdx, mOffset, mSize, nBlock);
                }

                // N-axis barrier: 所有 N-tile 完成, 通知 AIV
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_S0_L1_C2V_0);
            }

            // ================================================================
            // Stage 1: V0 — AIV only, matM_i 核内切分
            // ================================================================
            if ASCEND_IS_AIV {
                CrossCoreWaitFlag(SYNC_S0_L1_C2V_0);
                SysSyncAll<true>();

                S0_V0Process(groupIdx, mOffset, mSize);
            }
        }
    }
}
// [END_SECTION: S1_LOOP_BODY]

// [END_SECTION: S1_IMPL]

// ============================================================================
// [SECTION: SUBPROCESS_IMPL] 子流程实现
// ============================================================================

// -------------------- S0_C0Process (AIC, Cube matmul) --------------------
// Stage 0: C0 — int8×int8→int32 matmul, 2D (matM×matN) 多核
//
// Cube API 处理 [singleM, singleN] tile
// isTransposeB=true, has_bias=false
template <class T, class P>
__aicore__ inline void GmmdsqKernel<T, P>::S0_C0Process(
    uint32_t groupIdx,
    uint64_t mOffset, uint64_t mSize,
    uint32_t nBlockIdx)
{
    // [SECTION: C0_P0_S4_OFFSET]
    uint64_t x_offset = (groupStart + mOffset) * stride_x_M_;
    uint64_t nOffset = (uint64_t)nBlockIdx * singleMatN_;
    uint64_t nSize = ((uint64_t)singleMatN_ < (uint64_t)matN_ - nOffset)
                     ? (uint64_t)singleMatN_ : ((uint64_t)matN_ - nOffset);
    uint64_t ws_offset = (groupStart + mOffset) * stride_ws_M_ + nOffset;
    // [END_SECTION: C0_P0_S4_OFFSET]

    // [SECTION: C0_P0_S5_INSTRUCTIONS]
    // Python: c_temp1 = matmul(x, weight^T)  # int8×int8→int32
    // 参数: inputs=[x], outputs=[C0_output→workspace]
    // SSA: xGm_[x_offset], weightGm_, workspaceGm_[ws_offset]
    // load x from GM (Cube 直接读 GM)
    mm_.SetTensorA(xGm_[x_offset]);
    mm_.SetTensorB(weightGm_);  // isTransposeB=true in type def
    mm_.SetOrgShape(mSize, nSize, matK_, matK_, matN_);  // orgKc=matN (N 多核切分)
    mm_.SetSingleShape(mSize, nSize, matK_);
    while (mm_.Iterate()) {
        mm_.GetTensorC(workspaceGm_[ws_offset], 0, true);  // 连续写, 供 V0 读取
    }
    mm_.End();
    // [END_SECTION: C0_P0_S5_INSTRUCTIONS]
}

// -------------------- S0_V0Process (AIV, Vector 后处理) --------------------
// Stage 1: V0 — Cast + dequant + SwiGLU + dynamic quantize
// matM_i 核内切分, vec_chunk = 2 rows
template <class T, class P>
__aicore__ inline void GmmdsqKernel<T, P>::S0_V0Process(
    uint32_t groupIdx,
    uint64_t mOffset, uint64_t mSize)
{
    // [SECTION: S3_TILING_LOOP]
    // matM_i 核内切分: vec_chunk = DEFAULT_VEC_CHUNK_SIZE = 2
    constexpr uint32_t VEC_CHUNK_SIZE = DEFAULT_VEC_CHUNK_SIZE;

    for (uint64_t chunkOffset = 0; chunkOffset < mSize; chunkOffset += VEC_CHUNK_SIZE) {
        uint64_t chunkSize = (VEC_CHUNK_SIZE < mSize - chunkOffset)
                             ? VEC_CHUNK_SIZE : (mSize - chunkOffset);

        // [SECTION: V0_P0_S4_OFFSET]
        uint64_t ws_read_offset = (groupStart + mOffset + chunkOffset) * stride_ws_M_;
        uint64_t xScale_offset = groupStart + mOffset + chunkOffset;
        uint64_t quantOut_offset = (groupStart + mOffset + chunkOffset) * stride_out_M_;
        uint64_t quantScaleOut_offset = groupStart + mOffset + chunkOffset;
        // [END_SECTION: V0_P0_S4_OFFSET]

        // [SECTION: V0_P0_S5_INSTRUCTIONS]
        uint64_t totalElem = chunkSize * matN_;       // chunk × N elements
        uint64_t halfElem = chunkSize * (matN_ / 2); // chunk × N/2 elements

        // #1 Python: c_temp1 = c_temp1.to(torch.float32)
        // load int32 from workspace → Cast to fp32
        // load %0_in, workspaceGm_[ws_read_offset], totalElem
        // Cast int32→fp32
        Cast(%1_alc, %0_in, RoundMode::CAST_NONE, totalElem);

        // #2 Python: c_temp2 = c_temp1 * perChannelScale
        // weightScale broadcast [N]→[chunk,N]
        // c_temp2 = c_temp1 * perChannelScale
        Mul(%2_alc, %1_alc, weightScale_alc, totalElem);

        // #3 Python: c_temp3 = c_temp2 * perTokenScale.reshape(m,1)
        // load xScale slice
        // load xScale_alc, xScaleGm_[xScale_offset], chunkSize
        // per-row Muls: broadcast perTokenScale to each row
        for (uint32_t r = 0; r < chunkSize; ++r) {
            float scaleVal = xScale_alc.GetValue(r);
            Muls(%2_alc[r * matN_], %2_alc[r * matN_], scaleVal, matN_);
        }

        // #4 Python: c_temp4, gate = c_temp3.chunk(2, dim=-1)
        // memory view: left = %2_alc[0..halfElem), gate = %2_alc[halfElem..totalElem)
        // left_view = %2_alc, offset=0, size=halfElem
        // gate_view = %2_alc, offset=halfElem, size=halfElem

        // #5 Python: c_temp5 = c_temp4 * sigmoid(c_temp4)  (SiLU)
        Sigmoid(%4_alc, left_view, halfElem);
        Mul(%5_alc, left_view, %4_alc, halfElem);

        // #6 Python: c_temp6 = c_temp5 * gate
        Mul(%6_alc, %5_alc, gate_view, halfElem);

        // #7 Python: abs_max = max(abs(c_temp6), dim=-1).values
        Abs(%7_alc, %6_alc, halfElem);
        ReduceMax(%8_alc, %7_alc, tmpBuf_, chunkSize, matN_ / 2);

        // #8 Python: quantScale = 127.0 / abs_max
        Reciprocal(%8_alc, %8_alc, chunkSize);
        Muls(%9_alc, %8_alc, 127.0f, chunkSize);

        // #9 Python: quantOutput = round(c_temp6 * quantScale).to(torch.int8)
        // per-row: Muls with quantScale, then Cast fp32→int8 with round
        for (uint32_t r = 0; r < chunkSize; ++r) {
            float qScale = %9_alc.GetValue(r);
            Muls(%6_alc[r * (matN_ / 2)], %6_alc[r * (matN_ / 2)], qScale, matN_ / 2);
        }
        Cast(%6_out, %6_alc, RoundMode::CAST_ROUND, halfElem);
        // store quantOutput to GM
        // store %6_out, quantOutputGm_[quantOut_offset], halfElem

        // #10 Python: quantScaleOutput = 1.0 / quantScale
        Reciprocal(%10_alc, %9_alc, chunkSize);
        // store quantScaleOutput to GM
        // store %10_alc, quantScaleOutputGm_[quantScaleOut_offset], chunkSize
        // [END_SECTION: V0_P0_S5_INSTRUCTIONS]
    }
    // [END_SECTION: S3_TILING_LOOP]
}

// [END_SECTION: SUBPROCESS_IMPL]

// ============================================================================
// [SECTION: INIT_IMPL] Init 实现
// ============================================================================

template <class T, class P>
__aicore__ inline void GmmdsqKernel<T, P>::Init(const InitParams &params)
{
    // 1. 绑定 GlobalTensor
    // [SECTION: INIT_GLOBAL]
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(params.x));
    // weight, weightScale: tensorList, 需在 group 循环中动态绑定
    xScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.xScale));
    groupListGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(params.groupList));
    quantOutputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(params.quantOutput));
    quantScaleOutputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.quantScaleOutput));
    workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.workspace));
    // [END_SECTION: INIT_GLOBAL]

    // 2. 解析 TilingData
    // [SECTION: INIT_TILING]
    tiling_ = params.tilingData;

    coreNum_ = tiling_->coreNum;
    matM_ = tiling_->matM;
    matK_ = tiling_->matK;
    matN_ = tiling_->matN;
    groupNum_ = tiling_->groupNum;

    // Step 1: 多核切分参数
    // numMCores_, numNCores_ 由 tiling 或推导确定
    // singleMatM_ = ceil(matM_ / numMCores_)
    // singleMatN_ = ceil(matN_ / numNCores_)
    // numMBlocks_ = ceil(matM_ / singleMatM_)
    // numNBlocks_ = ceil(matN_ / singleMatN_)
    // [ASCEND_MODIFY] 从 tiling 读取或推导

    // Step 3: Stride 计算
    stride_x_M_ = (uint64_t)matK_;
    stride_ws_M_ = (uint64_t)matN_;
    stride_out_M_ = (uint64_t)matN_ / 2;

    // 核 ID 映射
    coreIdx_ = GetBlockIdx();
    if ASCEND_IS_AIC {
        mIdx_ = coreIdx_ / numNCores_;
        nIdx_ = coreIdx_ % numNCores_;
    }
    if ASCEND_IS_AIV {
        mIdx_ = coreIdx_;
        nIdx_ = 0;  // V0 不切 N
    }
    // [END_SECTION: INIT_TILING]

    // 3. 初始化 Pipe 和 Queue/Buffer
    // [SECTION: INIT_PIPE]
    pipe_ = params.tPipe;

    if ASCEND_IS_AIV {
        InitLocalBuffers();
    }
    // [END_SECTION: INIT_PIPE]

    SyncAll();
}

// -------------------- InitLocalBuffers (AIV 专用) --------------------
// [SECTION: S6_IMPL] Step 5: UB 资源分配 + InitLocalBuffers 实现
template <class T, class P>
__aicore__ inline void GmmdsqKernel<T, P>::InitLocalBuffers()
{
    // [SECTION: S6_INIT_QUEUES]
    // Step 5: Queue 初始化（根据 SSA 生命周期分析生成）
    // 占位符, Step 5 覆盖
    // [END_SECTION: S6_INIT_QUEUES]

    // [SECTION: S6_INIT_BUFFERS]
    // Step 5: Buffer 初始化（根据 SSA 生命周期分析生成）
    // 占位符, Step 5 覆盖
    // [END_SECTION: S6_INIT_BUFFERS]

    // [SECTION: S6_INIT_LOCAL_TENSORS]
    // Step 5: LocalTensor 获取（根据 SSA 映射生成）
    // 占位符, Step 5 覆盖
    // [END_SECTION: S6_INIT_LOCAL_TENSORS]
}
// [END_SECTION: S6_IMPL]

// ============================================================================
// [SECTION: PROCESS_IMPL] Process 实现
// ============================================================================

template <class T, class P>
__aicore__ inline void GmmdsqKernel<T, P>::Process()
{
    RunStage0();
    SyncAll();
}

// [END_SECTION: PROCESS_IMPL]

} // namespace Gmmdsq

#endif // __GMMDSQ_KERNEL_H_
