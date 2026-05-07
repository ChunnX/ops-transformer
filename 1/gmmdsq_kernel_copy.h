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

    // Step 4: Stride 参数（按需生成）
    // [SECTION: S4_STRIDE_DATA]
    // stride 由 Step 4 根据实际 tensor patterns 生成
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
        // Step 4: 计算 group 起止行
        //   groupStart = (groupIdx == 0) ? 0 : groupList_[groupIdx - 1]
        //   groupEnd = groupList_[groupIdx]
        //   currentM = groupEnd - groupStart
        //
        // 绑定当前 group 的 weight 和 weightScale
        //   weightGm_.SetGlobalBuffer(tensorList_element(weight, groupIdx))
        //   weightScaleGm_.SetGlobalBuffer(tensorList_element(weightScale, groupIdx))
        // [END_SECTION: GROUP_BOUNDARY]

        // ====================================================================
        // L1: matM 多核切分 — 每个 M-block
        // ====================================================================
        // AIC: mIdx = coreIdx / numNCores_, 按 numMCores_ 步进
        // AIV: mIdx = coreIdx, 按 numMCores_ 步进 (1c2v 核独立分配)
        uint32_t assignedMStart = 0;  // [ASCEND_MODIFY] Step 4 计算
        uint32_t assignedMEnd = 0;    // [ASCEND_MODIFY] Step 4 计算

        for (uint32_t mBlock = assignedMStart; mBlock < assignedMEnd; ++mBlock) {
            // [SECTION: MBLOCK_OFFSET]
            // Step 4: M-block 偏移
            //   uint64_t mOffset = groupStart + mBlock * singleMatM_;
            //   uint64_t mSize = min(singleMatM_, currentM - mBlock * singleMatM_);
            // [END_SECTION: MBLOCK_OFFSET]

            // ================================================================
            // Stage 0: C0 — AIC only, matN 多核切分
            // ================================================================
            if ASCEND_IS_AIC {
                // L2: matN 多核切分 — 遍历当前核负责的 N-tile
                for (uint32_t nBlock = nIdx_; nBlock < numNBlocks_; nBlock += numNCores_) {
                    // [SECTION: C0_NBLOCK_OFFSET]
                    // Step 4: N-block 偏移
                    //   uint64_t nOffset = nBlock * singleMatN_;
                    //   uint64_t nSize = min(singleMatN_, matN_ - nOffset);
                    // [END_SECTION: C0_NBLOCK_OFFSET]

                    S0_C0Process(groupIdx, /*mOffset*/0, /*mSize*/0, nBlock);
                }

                // N-axis barrier: 所有 N-tile 完成, 通知 AIV
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_S0_L1_C2V_0);
            }

            // ================================================================
            // Stage 1: V0 — AIV only, matM_i 核内切分
            // ================================================================
            if ASCEND_IS_AIV {
                CrossCoreWaitFlag(SYNC_S0_L1_C2V_0);

                S0_V0Process(groupIdx, /*mOffset*/0, /*mSize*/0);
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
    // Step 4: 偏移计算
    //
    // x_slice offset:   mOffset * matK_               (行偏移 × K)
    // weight offset:    0 (tensorList 已绑定当前 group)
    // workspace offset: mOffset * matN_ + nBlockIdx * singleMatN_  (2D 布局)
    // [END_SECTION: C0_P0_S4_OFFSET]

    // [SECTION: C0_P0_S5_INSTRUCTIONS]
    // Step 5: 指令填充 (Cube matmul)
    //
    // --- 指令占位: Cube matmul int8×int8→int32 ---
    // 输入: x_slice [GM, mSize×K, int8] (producer: GM, offset from groupStart+mOffset)
    //       weight_i [GM, K×N, int8] (producer: tensorList[groupIdx])
    // 输出: C0_output [workspace GM, mSize×nSize, int32]
    // 类型: Matmul API
    //       mm_.SetTensorA(xGm_[x_offset])
    //       mm_.SetTensorB(weightGm_)  // isTransposeB=true in type def
    //       mm_.IterateAll(workspaceGm_[ws_offset])
    //       mm_.End()
    // [ASCEND_MODIFY] Step 5 填充具体 API 调用
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
        // Step 4: 偏移计算
        //
        // workspace read offset: (mOffset + chunkOffset) * matN_ (int32 结果)
        // xScale offset:         mOffset + chunkOffset             (per-token scale)
        // quantOutput offset:    (mOffset + chunkOffset) * (matN_ / 2)  (N/2 列)
        // quantScaleOutput offset: mOffset + chunkOffset
        // [END_SECTION: V0_P0_S4_OFFSET]

        // [SECTION: V0_P0_S5_INSTRUCTIONS]
        // Step 5: 指令填充 (Vector 后处理, vec_chunk=2 rows)
        //
        // --- 指令占位: Read matmul result from workspace ---
        // 输入: C0_output [workspace, chunk×N, int32]
        // 输出: int32_buf [UB]
        // 类型: DataCopy (load)
        //
        // --- 指令占位: Cast int32→fp32 ---
        // 输入: int32_buf [UB]
        // 输出: fp32_buf [UB]
        // 类型: Cast
        //
        // --- 指令占位: Load perChannelScale and broadcast ---
        // 输入: weightScale_i [GM, N, fp32] (tensorList[groupIdx])
        // 输出: weightScale_ub [UB]
        // 类型: DataCopy (load) — 仅首次或 group 切换时加载
        //
        // --- 指令占位: Dequant step1: mul perChannelScale ---
        // 输入: fp32_buf [UB], weightScale_ub [UB, broadcast N→chunk×N]
        // 输出: dequant1_buf [UB]
        // 类型: Mul
        //
        // --- 指令占位: Load perTokenScale slice ---
        // 输入: xScale_slice [GM, chunk, fp32]
        // 输出: xScale_ub [UB]
        // 类型: DataCopy (load)
        //
        // --- 指令占位: Dequant step2: mul perTokenScale ---
        // 输入: dequant1_buf [UB], xScale_ub [UB, broadcast chunk×1→chunk×N]
        // 输出: dequant2_buf [UB, chunk×N]
        // 类型: Mul
        //
        // --- 指令占位: SwiGLU: chunk N→N/2 split ---
        // 输入: dequant2_buf [UB, chunk×N]
        // 输出: left_half [UB, chunk×N/2], gate [UB, chunk×N/2]
        // 类型: memory view (no compute)
        //
        // --- 指令占位: SwiGLU: sigmoid(left) ---
        // 输入: left_half [UB]
        // 输出: sigmoid_buf [UB, chunk×N/2]
        // 类型: Sigmoid
        //
        // --- 指令占位: SwiGLU: SiLU = left * sigmoid(left) ---
        // 输入: left_half [UB], sigmoid_buf [UB]
        // 输出: silu_buf [UB, chunk×N/2]
        // 类型: Mul
        //
        // --- 指令占位: SwiGLU: result = SiLU * gate ---
        // 输入: silu_buf [UB], gate [UB]
        // 输出: swiglu_buf [UB, chunk×N/2]
        // 类型: Mul
        //
        // --- 指令占位: Dynamic quant: abs + max reduce ---
        // 输入: swiglu_buf [UB, chunk×N/2]
        // 输出: absmax_buf [UB, chunk]
        // 类型: Abs + ReduceMax (reduce along N/2)
        //
        // --- 指令占位: Dynamic quant: scale = 127.0 / absmax ---
        // 输入: absmax_buf [UB, chunk]
        // 输出: quantScale_ub [UB, chunk]
        // 类型: Muls (scalar: 127.0)
        //       注意: 这里实际是 127.0 / x, 需用 Reciprocal 或 Divs
        //
        // --- 指令占位: Dynamic quant: round(swiglu * scale) ---
        // 输入: swiglu_buf [UB, chunk×N/2], quantScale_ub [UB, broadcast]
        // 输出: rounded_buf [UB, chunk×N/2]
        // 类型: Mul + Round
        //
        // --- 指令占位: Cast fp32→int8 ---
        // 输入: rounded_buf [UB]
        // 输出: quantOutput_ub [UB, int8]
        // 类型: Cast
        //
        // --- 指令占位: Store quantOutput to GM ---
        // 输入: quantOutput_ub [UB, int8]
        // 输出: quantOutput [GM, offset=(mOffset+chunkOffset)*(N/2)]
        // 类型: DataCopy (store)
        //
        // --- 指令占位: Compute dequantScale = 1.0 / quantScale ---
        // 输入: quantScale_ub [UB, chunk]
        // 输出: dequantScale_ub [UB, chunk]
        // 类型: Reciprocal (1/x)
        //
        // --- 指令占位: Store quantScaleOutput to GM ---
        // 输入: dequantScale_ub [UB, fp32]
        // 输出: quantScaleOutput [GM, offset=mOffset+chunkOffset]
        // 类型: DataCopy (store)
        //
        // [ASCEND_MODIFY] Step 5 填充具体 SSA 指令和 AscendC API 调用
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
