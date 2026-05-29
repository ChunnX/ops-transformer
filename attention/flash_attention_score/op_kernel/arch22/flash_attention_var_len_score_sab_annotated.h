/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * ============================================================================
 * 【带注释版本】FlashAttentionVarLenScoreSameAB — 变长序列 Flash Attention 正向算子
 *
 * ─── 这个文件在做什么 ───
 * 在华为 Ascend NPU 上执行 Flash Attention 的前向计算，支持变长序列。
 * QKV 按 TND 布局存储（Token-major, No-padding, D-contiguous）。
 *
 * ─── 跟定长版本的核心区别 ───
 * 1. 每条 batch 的序列长度不同，通过 actualSeqLengths/actualSeqLengthsKv 前缀和数组编码
 * 2. 地址计算不再用固定 stride，而是逐 batch 累加 s1SizeAcc/s2SizeAcc
 * 3. 支持退化 case：某条 batch 的 KV 为空但 Q 不为空，需要预清零输出
 *
 * ─── 硬件架构 ───
 * NPU 上有两种计算核：
 *   AIC (AI Core): 负责矩阵乘 (Cube 单元)，跑 BMM1 和 BMM2
 *   AIV (AI Vector): 负责向量运算，跑 softmax/dropout/mask (ProcessVec1/ProcessVec2)
 * 每个核内 AIC 和 AIV 通过 CrossCoreSetFlag/CrossCoreWaitFlag 做事件同步
 *
 * ─── 计算流程 (S = Q×Kᵀ → P = softmax(S) → O = P×V) ───
 * 外循环 s1: 遍历 Q 的各个分块 (每次处理 cubeS1BaseSize=256 个 query token)
 * 内循环 s2: 对固定 Q 块，遍历所有 K/V 分块 (每次处理 256 个 KV token)
 * 三级流水: BMM1 → Vec1(softmax) → BMM2 → Vec2(写回) 四阶段并行
 *
 * ─── L1Carry 优化 ───
 * 当条件满足时 (dSize=192/128/88/80, cubeS1BaseSize≤256, 无 RoPE):
 *   Q 矩阵在 s2 内循环中常驻 L1 缓存，只拷一次
 *   K/V/P 用 L1 双缓冲 ping-pong，掩盖搬运延迟
 * 相比标准路径省掉 Q 的重复 GM→L1 搬运
 */

#ifndef FLASH_ATTENTION_VAR_LEN_SCORE_SAB_H
#define FLASH_ATTENTION_VAR_LEN_SCORE_SAB_H

#include "flash_attention_score_s1s2_bn2gs1_sab.h"

// ============================================================================
// 事件 ID 定义 — 用于 L1Carry 模式下 MTE1(搬运)↔MTE2(计算) 的同步
// EVENT_ID2: Q 张量的搬运/计算同步
// EVENT_ID4: KV 张量的搬运/计算同步 (双缓冲用+0/+1)
// EVENT_ID6: P 张量的搬运/计算同步 (双缓冲用+0/+1)
// ============================================================================
#define Q_EVENT0  EVENT_ID2
#define KV_EVENT0 EVENT_ID4
#define P_EVENT0  EVENT_ID6

template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T, typename T = INPUT_T,
          CubeFormat bmm1Format = CubeFormat::NZ, MmPolicyType mmPolicyType = MmPolicyType::NORMAL,
          ImplModeEnum implMode = ImplModeEnum::AA_HIGH_PRECISION, bool hasRope = false>
class FlashAttentionVarLenScoreSameAB
    : public FlashAttentionScoreS1s2Bn2gs1SameAB<implMode, layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                           bmm1Format, mmPolicyType, hasRope> {
public:
    __aicore__ inline FlashAttentionVarLenScoreSameAB(){};

    // ============================================================================
    // UnpackInit: 每个核执行一次的入口，完成所有初始化工作
    // 参数中 actualSeqLengths/actualSeqLengthsKv 是变长序列的核心输入
    // 它们是前缀和数组: actualSeqLengths[i] = sum_{j=0}^{i} qLen[j]
    // ============================================================================
    __aicore__ inline void
    UnpackInit(__gm__ uint8_t *query, __gm__ uint8_t *queryRope, __gm__ uint8_t *key, __gm__ uint8_t *keyRope,
               __gm__ uint8_t *value, __gm__ uint8_t *pse,
               __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask, __gm__ uint8_t *sink,
               __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *softmaxMax,
               __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut,
               __gm__ uint8_t *workspace, const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe);

    // ============================================================================
    // Process: 主计算流程 — AIC/AIV 三阶段流水线
    // ============================================================================
    __aicore__ inline void Process();

    // ============================================================================
    // 矩阵乘配置: CFG_CUSTOM_EXCEED — 允许 Cube 单元处理超出标准尺寸的矩阵块
    // UNSPLITK 模式用 CFG_DIS_UNIT_FLAG_EXCEED, NORMAL 模式用 CFG_EXCEED
    // ============================================================================
    static constexpr MatmulConfig CFG_CUSTOM_EXCEED = (mmPolicyType == MmPolicyType::UNSPLITK) ? CFG_DIS_UNIT_FLAG_EXCEED : CFG_EXCEED;

    // ============================================================================
    // L1Carry 专用矩阵乘实例 — Q/K/V 数据在 L1 buffer 中，Cube 直接读 L1
    // 普通路径用的是父类的 bmm1/bmm2 (数据在 GM 中)
    // ============================================================================
    // bmm1L1Carry: S=Q×Kᵀ, L1Carry 版本, 输出到 GM (ND 格式)
    using a1TypeL1Carry = MatmulType<TPosition::A1, CubeFormat::NZ, INPUT_T, false, LayoutMode::NONE, true>;
    using b1TypeL1Carry = MatmulType<TPosition::B1, CubeFormat::NZ, INPUT_T, true,  LayoutMode::NONE, true>;
    using bias1TypeL1Carry = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c1TypeL1Carry = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using mm1TypeL1Carry = matmul::MatmulImpl<a1TypeL1Carry, b1TypeL1Carry, c1TypeL1Carry, bias1TypeL1Carry, CFG_CUSTOM_EXCEED>;
    mm1TypeL1Carry bmm1L1Carry;

    // bmm1NzL1Carry: S=Q×Kᵀ, L1Carry 版本, 输出到 GM (NZ 格式)
    // 当 s2 尾块不对齐 64 时使用，NZ 格式可以标记无效数据列
    using c1NzTypeL1Carry = MatmulType<TPosition::GM, CubeFormat::NZ, T>;
    using mm1NzTypeL1Carry = matmul::MatmulImpl<a1TypeL1Carry, b1TypeL1Carry, c1NzTypeL1Carry, bias1TypeL1Carry, CFG_CUSTOM_EXCEED>;
    mm1NzTypeL1Carry bmm1NzL1Carry;

    // bmm2L1Carry: O=P×V, L1Carry 版本
    // BMM2 的 P 来自 Vec1 输出(stage1Res), V 来自 valueGm
    // Cube 内部用 L0C 做累加: k==0 覆盖, k>0 累加
    using a2TypeL1Carry = MatmulType<TPosition::A1, CubeFormat::NZ, INPUT_T, false, LayoutMode::NONE, true>;
    using b2TypeL1Carry = MatmulType<TPosition::B1, CubeFormat::NZ, INPUT_T, false, LayoutMode::NONE, true>;
    using bias2TypeL1Carry = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c2NzTypeL1Carry = MatmulType<TPosition::GM, CubeFormat::NZ, T>;
    using mm2TypeL1Carry = matmul::MatmulImpl<a2TypeL1Carry, b2TypeL1Carry, c2NzTypeL1Carry, bias2TypeL1Carry, CFG_MDL_EXCEED>;
    mm2TypeL1Carry bmm2L1Carry;

protected:
    // ============================================================================
    // 子类重写的核心方法
    // ============================================================================
    __aicore__ inline void ComputeConstexpr();              // 预计算所有维度乘积 (gD, n2D, s2BaseN2D 等)
    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx); // 从展平的工作项索引还原 (boIdx, n2oIdx, goIdx, s1oIdx)
    __aicore__ inline void ComputeBmm1Tail(SplitSameABExtraInfo &extraInfo); // 尾块处理: 算实际大小 + duplicateMask
    __aicore__ inline void SetExtraInfo(SplitSameABExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount,
                                        int64_t s2LoopLimit, int64_t multiCoreInnerIdx); // 每轮迭代装配 extraInfo
    __aicore__ inline void CalS1OuterSize(int64_t offset); // 变长核心: 逐 batch 累加定位 s1OuterSizeAcc
    __aicore__ inline void GetSeqQlenKvlenByBoidx(int64_t boIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvlen); // 从前缀和还原单条batch长度
    __aicore__ inline void GetS2LoopRange();                // 根据稀疏模式确定 s2 循环的 [start, end)

    // ============================================================================
    // L1Carry 版 BMM1: S = Q × Kᵀ
    // Q: L1 常驻 (只在 s2LoopCount==0 时拷一次)
    // K: L1 双缓冲，按 N=256 切片搬运
    // ============================================================================
    template <typename T2>
    __aicore__ inline void ComputeMm1(SplitSameABExtraInfo &extraInfo,
                                      matmul::MatmulImpl<a1TypeL1Carry, b1TypeL1Carry, T2, bias1TypeL1Carry, CFG_CUSTOM_EXCEED> &bmm1);

    // ============================================================================
    // L1Carry 版 BMM2: O = P × V
    // 双重循环: 外层 N(d2Size切128), 内层 K(s2RealSize切192)
    // P 和 V 都用 L1 双缓冲
    // ============================================================================
    __aicore__ inline void ComputeMm2(SplitSameABExtraInfo &extraInfo,
                                      matmul::MatmulImpl<a2TypeL1Carry, b2TypeL1Carry, c2NzTypeL1Carry, bias2TypeL1Carry, CFG_MDL_EXCEED> &bmm2);

    // ============================================================================
    // 数据搬运函数 — GM↔L1 的数据拷贝
    // ============================================================================
    __aicore__ inline void CopyInMm1AToL1(LocalTensor<INPUT_T> &l1Tensor, SplitSameABExtraInfo &extraInfo);   // Q: GM→L1
    __aicore__ inline void CopyInMm1BToL1(LocalTensor<INPUT_T> &bL1Tensor, const SplitSameABExtraInfo &extraInfo, // K: GM→L1 (N切片)
                                          uint32_t subNid, uint32_t subNSize, uint32_t nSplitSize);
    __aicore__ inline void CopyInMm2AToL1(LocalTensor<INPUT_T> &al1Tensor, SplitSameABExtraInfo &extraInfo,       // P: stage1Res→L1 (K切片)
                                          uint32_t subKid, uint32_t subKSize, uint32_t kSplitSize);
    __aicore__ inline void CopyInMm2BToL1(LocalTensor<INPUT_T> &bL1Tensor,                                        // V: GM→L1 (K×N切片)
                                          uint32_t subKid, uint32_t subKSize, uint32_t kSplitSize, uint32_t nLoopIdx, bool isLastN);
    __aicore__ inline void CopyGmToL1(LocalTensor<INPUT_T> &l1Tensor, GlobalTensor<INPUT_T> &gmSrcTensor, uint32_t srcN,  // 通用 GM→L1 (ND→NZ格式转换)
                                      uint32_t srcD, uint32_t srcDstride);

    // ============================================================================
    // 变长序列专用成员变量
    // ============================================================================
    GM_ADDR actualSeqQlenAddr;        // actualSeqLengths 的 GM 地址 (query 长度前缀和)
    GM_ADDR actualSeqKvlenAddr;       // actualSeqLengthsKv 的 GM 地址 (KV 长度前缀和)
    GM_ADDR sinkAddr;                 // sink token 偏移数组
    GM_ADDR prefixNAddr;              // prefix N 数组 (PREFIX 稀疏模式用)

    // ============================================================================
    // 地址累积量 — 变长场景的核心，用于从展平的 work item idx 定位到正确的 batch+offset
    // s1OuterSizeAcc: 前面所有 batch 的 s1 outer 块数累计 → 用于判断当前 work item 属于哪个 batch
    // s1SizeAcc:      前面所有 batch 的 query token 总数 → 用于计算 Q 的 GM 偏移
    // s2SizeAcc:      前面所有 batch 的 KV token 总数   → 用于计算 K/V 的 GM 偏移
    // attenB1SSOffset: 前面所有 batch 的 S1×S2 乘积累计  → 用于 attention mask 的偏移
    // ============================================================================
    uint64_t s1OuterSizeAcc;
    uint64_t s1SizeAcc;
    uint64_t s2SizeAcc;
    uint64_t attenB1SSOffset;
    uint64_t softmaxPingPongCnt = 0;

    // 前缀和数组的 GlobalTensor 包装
    GlobalTensor<int64_t> qListGm;    // actualSeqLengths: prefix_sum[i] = sum_{j≤i} qLen[j]
    GlobalTensor<int64_t> kvListGm;   // actualSeqLengthsKv: prefix_sum[i] = sum_{j≤i} kvLen[j]
    GlobalTensor<float> sinkGm;       // sink token 数据

    // ============================================================================
    // AIC↔AIV 同步事件组 — 三套事件实现流水线同步
    // C1=Compute1(BMM1), V1=Vec1(softmax), C2=Compute2(BMM2), V2=Vec2(writeback)
    // 三套 extraInfo 轮转，对应三组事件各 3 个
    // ============================================================================
    uint64_t SYNC_C1_V1_FLAG[3] = {4, 5, 6};  // AIC(BMM1完成) → AIV(开始Vec1)
    uint64_t SYNC_V1_C2_FLAG[3] = {7, 8, 9};  // AIV(Vec1完成) → AIC(开始BMM2)
    uint64_t SYNC_C2_V2_FLAG[3] = {1, 2, 3};  // AIC(BMM2完成) → AIV(开始Vec2)

    // ============================================================================
    // L1 Buffer 大小常量
    // Q:  192×256 elements  → 一行 192 个 d 元素，256 个 query token
    // KV: 256×192 elements  → 256 个 KV token × 192 个 d 元素 (刚好是 Q 的转置)
    // P:  256×192 elements  → softmax 输出，跟 KV 同形状
    // L0C: 256×192 elements → Cube 累加器，BMM2 的结果暂存
    // 注意: Q 和 KV 的尺寸互为转置 (192×256 vs 256×192)，对应 BMM1: [s1,d] × [d,s2] → [s1,s2]
    // ============================================================================
    uint32_t L1_Q_SIZE = (192 * 256 * sizeof(INPUT_T));
    uint32_t L1_KV_SIZE = (256 * 192 * sizeof(INPUT_T));
    uint32_t L1_P_SIZE = (256 * 192 * sizeof(INPUT_T));
    uint32_t L0C_SIZE = (256 * 192 * sizeof(T));

    // ============================================================================
    // L1/L0C Buffer 和对应的 LocalTensor
    // ============================================================================
    TBuf<TPosition::A1> queryBufL1;     // Q 的 L1 buffer (常驻，s2 循环只拷一次)
    LocalTensor<INPUT_T> qL1Tensor;
    TBuf<TPosition::B1> kvBufL1;        // K/V 的 L1 buffer (双缓冲: 2 × L1_KV_SIZE)
    LocalTensor<INPUT_T> kvL1Tensor;
    TBuf<TPosition::A1> pBufL1;         // P 的 L1 buffer (双缓冲: 2 × L1_P_SIZE)
    LocalTensor<INPUT_T> pL1Tensor;
    TBuf<TPosition::CO1> oBufL0C;       // BMM2 累加器的 L0C buffer
    LocalTensor<T> oL0CTensor;

    int64_t kCoreOffset;       // 当前 K 块的 GM 偏移地址
    int64_t vCoreOffset;       // 当前 V 块的 GM 偏移地址
    uint32_t kvL1BufIter = 0;  // KV L1 双缓冲的迭代计数 (每次+1, %2 取槽位)
    uint32_t pL1BufIter = 0;   // P  L1 双缓冲的迭代计数
    bool needL1Carry = false;  // 是否启用 L1Carry 优化
};

// ====================================================================================
// 以下是各成员函数的实现
// ====================================================================================

// ====================================================================================
// UnpackInit: 初始化入口
// 流程: 父类初始化 → L1Carry 检查 → ComputeConstexpr → 退化 case 清零 → buffer 分配
// ====================================================================================
template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                       bmm1Format, mmPolicyType, implMode, hasRope>::UnpackInit(
    __gm__ uint8_t *query, __gm__ uint8_t *queryRope, __gm__ uint8_t *key, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *prefix, __gm__ uint8_t *attenMask, __gm__ uint8_t *sink, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe)
{
    // 步骤1: 调用父类初始化 → 绑定所有 GM 指针、初始化标准 BMM 实例、解析 tiling 参数
    this->InitInput(query, queryRope, key, keyRope, value, pse, dropMask, paddingMask, prefix, attenMask, sink, softmaxMax, softmaxSum,
                    softmaxOut, attentionOut, workspace, tiling, tPipe);

    if ASCEND_IS_AIC {
        // 步骤2: L1Carry 条件判断
        // 条件: 无 RoPE, S1 块≤256, dSize 在 {192,128,88,80} 中
        // 理由: 这些 dSize 配合 256 的 s1 块大小正好能放入 L1 buffer
        if constexpr (hasRope == false) {
            if (this->cubeS1BaseSize <= 256 &&
               (this->dSize == 192 || this->dSize == 128 || this->dSize == 88 || this->dSize == 80)) {
                this->needL1Carry = true;
                // 初始化 L1Carry 专用的 BMM 实例
                this->bmm1L1Carry .SetSubBlockIdx(0);
                this->bmm1L1Carry .Init(&tiling->bmm1TilingData, tPipe);
                this->bmm1NzL1Carry .SetSubBlockIdx(0);
                this->bmm1NzL1Carry .Init(&tiling->bmm1TilingData, tPipe);
                this->bmm2L1Carry .SetSubBlockIdx(0);
                this->bmm2L1Carry .Init(&tiling->bmm2TilingData, tPipe);
            }
        }
    }

    // 步骤3: 预计算所有维度乘积 (gD, n2D, s2BaseN2D, mm1Ka1 等)
    this->ComputeConstexpr();

    // 步骤4: 包装变长序列前缀和数组 → GlobalTensor 方便后续用 GetValue() 访问
    qListGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengths);
    kvListGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengthsKv);
    sinkGm.SetGlobalBuffer((__gm__ float *)sink);

    // 保存 GM 地址供后续使用
    actualSeqQlenAddr = actualSeqLengths;
    actualSeqKvlenAddr = actualSeqLengthsKv;
    sinkAddr = sink;

    // 步骤5: 退化 case 处理 — 遍历所有 batch，找出 KV 为空但 Q 不为空的
    // 这些 batch 没有 KV 可以 attend，输出应当为零
    // 但因为流水线不会为它们产生任何 BMM 结果，需要手动清零
    int64_t actualS1Len;
    int64_t actualS2Len;
    for (int64_t i = 0; i < this->tilingData->inputParams.bSize; ++i) {
        // 从前缀和解析第 i 条 batch 的实际长度
        GetSeqQlenKvlenByBoidx(i, actualS1Len, actualS2Len);

        // 条件: KV 为空(s2Len≤0) 但 Q 不为空(s1Len≠0)
        if (actualS2Len <= 0 && actualS1Len != 0) {
            // 计算累加偏移: 前面所有 batch 的 token 总数
            // 前缀和数组: actualSeqQlenAddr[i-1] 就是前 i 条 batch 的总 token 数
            int64_t accumSize = (i == 0) ? 0 : ((__gm__ int64_t *)actualSeqQlenAddr)[i - 1];

            // 处理 s1Len 为负数的情况 (标记位): 用总 s1Size 减去前面所有 batch 得到当前 batch 的真实长度
            if (actualS1Len < 0 && accumSize > 0) {
                actualS1Len = this->s1Size - accumSize;

                // 多核分配计算: 把这条 batch 的 token 按核均分
                int64_t frontCoreNum = actualS1Len % this->tilingData->multiCoreParams.coreNum;
                int64_t splitFactor = frontCoreNum > 0 ? 1 : 0;
                int64_t s1SizeInner = (actualS1Len / this->tilingData->multiCoreParams.coreNum);

                // 前 frontCoreNum 个核多拿一个 token (余数分配)
                int64_t innerOffset1 = (s1SizeInner + splitFactor) * (this->vecBlockIdx >= frontCoreNum ?
                                       frontCoreNum : this->vecBlockIdx);
                int64_t innerOffset2 = s1SizeInner * (this->vecBlockIdx >= frontCoreNum ?
                                       this->vecBlockIdx - frontCoreNum : 0);
                accumSize = accumSize + innerOffset1 + innerOffset2;
                actualS1Len = s1SizeInner + (this->vecBlockIdx >= frontCoreNum ? 0 : splitFactor);
            }

            // 清零 attentionOut: 该 batch 对应的输出区域全部置零
            AscendC::InitOutput<INPUT_T>(this->attentionOutGm[accumSize * this->n2GD2],
                                         actualS1Len * this->n2GD2, static_cast<INPUT_T>(0.0));
            // 清零 softmaxMax: max 值置零 (后续 online softmax 会正确初始化)
            AscendC::InitOutput<float>(this->softmaxMaxGm[accumSize * this->n2G * 8],
                                       actualS1Len * this->n2G * 8, static_cast<float>(0.0));
            // 清零 softmaxSum: sum 值置零
            AscendC::InitOutput<float>(this->softmaxSumGm[accumSize * this->n2G * 8],
                                       actualS1Len * this->n2G * 8, static_cast<float>(0.0));
        }
    }

    // 步骤6: 分配 buffer — 父类的 stage1/stage2/softmax/common buffer
    this->InitBuffer();

    // 步骤7: L1Carry 模式下分配 L1/L0C buffer
    if (this->needL1Carry) {
        this->pipe->InitBuffer(this->queryBufL1, L1_Q_SIZE);
        qL1Tensor = this->queryBufL1.template Get<INPUT_T>();
        this->pipe->InitBuffer(this->kvBufL1, L1_KV_SIZE * 2);    // ×2 = 双缓冲
        kvL1Tensor = this->kvBufL1.template Get<INPUT_T>();
        this->pipe->InitBuffer(this->pBufL1, L1_P_SIZE * 2);      // ×2 = 双缓冲
        pL1Tensor = this->pBufL1.template Get<INPUT_T>();
        this->pipe->InitBuffer(this->oBufL0C, L0C_SIZE);
        oL0CTensor = this->oBufL0C.template Get<T>();
    }

    // 步骤8: 杂项初始化
    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    DropOutBitModeInit(apiTmpBuffer);  // Dropout 随机种子初始化

    // 预计算 Alibi 位置编码 (只有主核需要)
    if (this->vecBlockIdx < this->tilingData->multiCoreParams.coreNum) {
        LocalTensor<half> pseHelpBuffer = this->stage1PingBuf.template Get<half>();
        PseInnerAlibiCreate<hasPse>(this->pseAlibiGm, pseHelpBuffer, this->pseInfo);
    }

    prefixNAddr = prefix;

    // 取前缀和数组最后一个元素 = 所有 batch 的 KV 总长度
    // 用于 attention mask TT 模式的全量 s2 尺寸
    this->s2SizeSum = ((__gm__ uint64_t *)actualSeqLengthsKv)[this->tilingData->inputParams.bSize - 1];
    return;
}

// ====================================================================================
// ComputeConstexpr: 预计算所有维度乘积
// 把形如 N2*D, G*D, S2*N2*D 的常量乘积提前算好存为成员变量
// 避免运行时重复计算，同时让 compiler 可以做常量优化
// ====================================================================================
template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                       bmm1Format, mmPolicyType, implMode, hasRope>::ComputeConstexpr()
{
    // 基础维度乘积
    this->gD = this->tilingData->inputParams.gSize * this->dSize;     // Group × D
    this->n2D = this->tilingData->inputParams.n2Size * this->dSize;   // N2 × D
    this->n2G = this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize;  // N2 × G
    this->n2GD = this->tilingData->inputParams.n2Size * this->gD;     // N2 × G × D
    this->gD2 = this->tilingData->inputParams.gSize * this->d2Size;   // G × D2 (MLA: D2≠D)
    this->n2D2 = this->tilingData->inputParams.n2Size * this->d2Size; // N2 × D2
    this->n2GD2 = this->tilingData->inputParams.n2Size * this->gD2;   // N2 × G × D2

    // S2 基本块的维度乘积 → 一次内循环步处理的数据量
    this->s2BaseN2D = this->s2BaseSize * this->n2D;
    this->s2BaseNratioSize = this->s2BaseSize * this->tilingData->coreParams.nRatio;

    this->s1Size = this->tilingData->inputParams.s1Size;  // 总 query token 数 (所有 batch 的总和)

    // S1 基本块的维度乘积 → 一次外循环步处理的数据量
    this->s1BaseN2GD = this->cubeS1BaseSize * this->n2GD;
    this->s1BaseN2GD2 = this->cubeS1BaseSize * this->n2GD2;
    this->s2BaseNratioN2D = this->s2BaseN2D * this->tilingData->coreParams.nRatio;

    // RoPE 相关维度乘积
    if constexpr (hasRope == true) {
        this->gDRope = this->tilingData->inputParams.gSize * this->dRopeSize;
        this->n2DRope = this->tilingData->inputParams.n2Size * this->dRopeSize;
        this->n2GDRope = this->tilingData->inputParams.n2Size * this->gDRope;
        this->s2BaseN2DRope = this->s2BaseSize * this->n2DRope;
        this->s1BaseN2GDRope = this->cubeS1BaseSize * this->n2GDRope;
        this->s2BaseNratioN2DRope = this->s2BaseN2DRope * this->tilingData->coreParams.nRatio;

        // BMM1 在 hasRope 时 K 维度被拆成两段:
        // 段1: 标准部分 (d)  → mm1Ka1/mm1Kb1
        // 段2: RoPE 部分 (dRope) → mm1Ka2/mm1Kb2
        this->mm1Ka2 = this->n2GDRope;
        this->mm1Kb2 = this->n2DRope;
    }

    // BMM1 的矩阵维度
    // S = Q×Kᵀ, Q:[s1, G×D], K:[s2, N2×D]
    // mm1Ka1 = G×D (K 方向在 Q 侧, 即 Q 的列数 / K 的列数)
    // mm1Kb1 = N2×D (K 方向在 K 侧)
    this->mm1Ka1 = this->n2GD;
    this->mm1Kb1 = this->n2D;
    this->mm1Kb = this->mm1Kb1 + this->mm1Kb2;  // 标准部分 + RoPE 部分 = 总 K 维度
    this->mm2Kb = this->n2D2;                   // BMM2 中 V 的列数

    // PSE (Position Encoding) 相关参数
    if constexpr (hasPse == true) {
        this->pseInfo.gSize = this->tilingData->inputParams.gSize;
        this->pseInfo.pseShapeType = this->tilingData->inputParams.pseShapeType;
        this->pseInfo.n2G = this->n2G;
        this->pseInfo.pseBSize = this->tilingData->inputParams.pseBSize;
        this->pseInfo.s1BaseSize = this->cubeS1BaseSize;
        this->pseInfo.pseS1Size = this->tilingData->inputParams.pseS1Size;
        this->pseInfo.pseS2Size = this->tilingData->inputParams.pseS2Size;
        this->pseInfo.s2BaseNratioSize = this->s2BaseNratioSize;
        this->pseInfo.pseEncodeType = (uint32_t)this->tilingData->inputParams.pseEncodeType;
        this->pseInfo.pseType = this->tilingData->inputParams.pseType;
        this->pseInfo.pseAlibiBaseS1 = this->tilingData->coreParams.pseAlibiBaseS1;
        this->pseInfo.pseAlibiBaseS2 = this->tilingData->coreParams.pseAlibiBaseS2;
        this->pseInfo.qStartIdx = this->tilingData->inputParams.qStartIdx;
        this->pseInfo.kvStartIdx = this->tilingData->inputParams.kvStartIdx;
    }

    if constexpr (hasDrop == true) {
        this->dropMaskInfo.gSize = this->tilingData->inputParams.gSize;
        this->dropMaskInfo.n2G = this->n2G;
        this->dropMaskInfo.s1BaseSize = this->cubeS1BaseSize;
        this->dropMaskInfo.s2BaseNratioSize = this->s2BaseNratioSize;
    }
}

// ====================================================================================
// GetSeqQlenKvlenByBoidx: 从前缀和数组还原第 boIdx 条 batch 的实际长度
//
// 前缀和数组: prefix[i] = sum_{j=0}^{i} len[j]
// 还原:     len[i] = (i==0) ? prefix[0] : prefix[i] - prefix[i-1]
//
// 例如: actualSeqLengths = [100, 450, 530, 730]
// 含义: batch0=100, batch1=350, batch2=80, batch3=200
// ====================================================================================
template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void
FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, implMode, hasRope>::GetSeqQlenKvlenByBoidx(int64_t boIdx,
                                                                                  int64_t &actualSeqQlen,
                                                                                  int64_t &actualSeqKvlen)
{
    if (boIdx == 0) {
        // 第一条 batch: 长度 = 前缀和第一个元素
        actualSeqQlen = qListGm.GetValue(0);
        actualSeqKvlen = kvListGm.GetValue(0);
    } else {
        // 后续 batch: 长度 = 当前前缀和 - 前一个前缀和
        actualSeqQlen = qListGm.GetValue(boIdx) - qListGm.GetValue(boIdx - 1);
        actualSeqKvlen = kvListGm.GetValue(boIdx) - kvListGm.GetValue(boIdx - 1);
    }
    return;
}

// ====================================================================================
// CalS1OuterSize: 变长序列的地址定位 — 逐 batch 累加确定 s1OuterSizeAcc
//
// 背景: multiCoreInnerIdx 是一个展平的一维工作项索引，编码了 (boIdx, head, s1_block)
// 但变长场景下每个 boIdx 的 s1 块数不同，不能简单用除法还原
// 需要逐 batch 累加 s1 块数，直到找到 multiCoreInnerIdx 所在的 batch
//
// offset: 当前核的起始 multiCoreInnerIdx
// 结果: s1OuterSizeAcc / s1SizeAcc / s2SizeAcc / boIdx 被设置为本 batch 之前的累积量
// ====================================================================================
template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                       bmm1Format, mmPolicyType, implMode, hasRope>::CalS1OuterSize(int64_t offset)
{
    int64_t actualS1Outersize = 0;

    // 初始化: 所有累积量为零
    this->boIdx = 0;
    this->s1OuterSizeAcc = 0;   // S1 外层块数累计
    this->attenB1SSOffset = 0;  // attention mask S1×S2 偏移累计
    this->s1SizeAcc = 0;        // query token 数累计
    this->s2SizeAcc = 0;        // KV token 数累计

    int64_t actualS1Len;
    int64_t actualS2Len;
    // 逐 batch 累加
    for (int64_t i = 0; i < this->tilingData->inputParams.bSize; i++) {
        GetSeqQlenKvlenByBoidx(i, actualS1Len, actualS2Len);

        // 第 i 条 batch 有多少个 s1 outer 块: CeilDiv(s1Len, cubeS1BaseSize) × n2g
        actualS1Outersize += (CeilDiv(actualS1Len, this->cubeS1BaseSize) * this->n2G);

        if (offset >= actualS1Outersize) {
            // 还没到目标 batch: 继续累加, 更新累积量
            this->s1OuterSizeAcc = actualS1Outersize;
            this->s1SizeAcc += actualS1Len;
            this->s2SizeAcc += actualS2Len;
            this->attenB1SSOffset += actualS1Len * actualS2Len;
            this->boIdx++;
        } else {
            // 找到了! offset 落在第 i 条 batch 中
            break;
        }
    }
    return;
}

// ====================================================================================
// ComputeAxisIdx: 从展平的工作项 multiCoreInnerIdx 还原出四维坐标 (boIdx, n2oIdx, goIdx, s1oIdx)
//
// 与定长版本的区别: boIdx 之前已经由 CalS1OuterSize 确定
// 这里继续处理 n2oIdx (head 方向), goIdx (group 方向), s1oIdx (s1 分块方向)
//
// 同时更新当前 batch 的实际 s1Size/s2Size
// ====================================================================================
template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                       bmm1Format, mmPolicyType, implMode, hasRope>::ComputeAxisIdx(int64_t multiCoreInnerIdx)
{
    int64_t actualS1Len;
    int64_t actualS2Len;

    // 获取当前 batch 的实际长度
    GetSeqQlenKvlenByBoidx(this->boIdx, actualS1Len, actualS2Len);

    // 计算当前 batch 的 s1 outer 上限
    int64_t actualS1Outersize = this->s1OuterSizeAcc + (CeilDiv(actualS1Len, this->cubeS1BaseSize) * this->n2G);

    // 如果 multiCoreInnerIdx 超出一整个 batch，继续往后跳
    // (这种情况发生在多核分配时，一个核可能跨越多个 batch)
    while (multiCoreInnerIdx >= actualS1Outersize) {
        this->s1OuterSizeAcc = actualS1Outersize;
        this->s1SizeAcc += actualS1Len;
        this->s2SizeAcc += actualS2Len;
        this->attenB1SSOffset += actualS1Len * actualS2Len;
        this->boIdx++;
        GetSeqQlenKvlenByBoidx(this->boIdx, actualS1Len, actualS2Len);
        actualS1Outersize = this->s1OuterSizeAcc + (CeilDiv(actualS1Len, this->cubeS1BaseSize) * this->n2G);
    }

    // 在当前 batch 内计算 (n2oIdx, goIdx, s1oIdx) 三维坐标
    // s1 outer 的布局: [N2 × G × S1_blocks]
    // tmpS1Outersize = 当前 batch 的 s1 块数
    int64_t tmpS1Outersize = CeilDiv(actualS1Len, this->cubeS1BaseSize);

    // 减去前面 batch 的累积, 得到在当前 batch 内的偏移
    actualS1Outersize = multiCoreInnerIdx - this->s1OuterSizeAcc;

    // 三维坐标还原:
    // n2oIdx: 当前 s1 outer 偏移 / s1块数 / gSize → N2 方向索引
    this->n2oIdx = actualS1Outersize / tmpS1Outersize / this->tilingData->inputParams.gSize;
    // goIdx: 在 N2 方向的一个 slice 内, 除以 s1块数 对 gSize 取余 → group 方向索引
    this->goIdx = actualS1Outersize / tmpS1Outersize % this->tilingData->inputParams.gSize;
    // s1oIdx: 余数 → s1 块索引
    this->s1oIdx = actualS1Outersize % tmpS1Outersize;

    // 更新当前 batch 的实际 s1/s2 长度 (供后续地址计算使用)
    GetSeqQlenKvlenByBoidx(this->boIdx, this->s1Size, this->s2Size);
}

// ====================================================================================
// GetS2LoopRange: 根据稀疏模式确定当前 s1 块的 s2 循环范围 [s2StartIdx, s2EndIdx)
//
// 支持的稀疏模式:
//   CAUSAL:              下三角, query 只能看到当前及之前的 KV
//   RIGHT_DOWN_CAUSAL:   右下对齐的因果 (qLen>kvLen 的场景)
//   BAND:                对角线附近的固定宽度带
//   BAND_COMPRESS:       自适应带宽 (preTokens/nextTokens)
//   PREFIX:              前缀注意力, s2EndIdx = max(s1_end, prefixN[boIdx])
//   BAND_LEFT_UP_CAUSAL: 左上因果+band
//   RIGHT_DOWN_CAUSAL_BAND: 右下因果+band
//
// 所有模式的 start/end 最后都对齐到 8 的倍数 (适配 8 元素向量化)
// ====================================================================================
template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                       bmm1Format, mmPolicyType, implMode, hasRope>::GetS2LoopRange()
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    GetSeqQlenKvlenByBoidx(this->boIdx, actualS1Len, actualS2Len);

    // 根据稀疏模式计算 s2 的可见范围
    if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::CAUSAL)) {
        // 下三角因果: 第 s1oIdx 个块能看到的最远 KV 位置 = (s1oIdx+1) * cubeS1BaseSize
        this->s2StartIdx = 0;
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize, actualS2Len);
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::RIGHT_DOWN_CAUSAL)) {
        // 右下对齐因果: 考虑 s1 和 s2 长度不等的偏移
        this->s2StartIdx = 0;
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize + actualS2Len - actualS1Len, actualS2Len);
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND)) {
        // 带状: 以当前 s1 块为中心, 上下各扩展 validSize
        this->s2StartIdx = Max(
            this->s1oIdx * this->cubeS1BaseSize - this->tilingData->coreParams.s1SparseValidSize, 0);
        this->s2EndIdx =
            Min((this->s1oIdx + 1) * this->cubeS1BaseSize + this->tilingData->coreParams.s2SparseValidSize, actualS2Len);
        // 如果带内没有有效 KV token → 退化为全量计算
        if (this->s2EndIdx - this->s2StartIdx <= 0) {
            this->s2StartIdx = 0;
            this->s2EndIdx = actualS2Len;
        }
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND_COMPRESS)) {
        // 自适应带宽: preTokens 个前置 + nextTokens 个后置
        this->s2StartIdx = Max(this->s1oIdx * this->cubeS1BaseSize - actualS1Len +
                                   Max(actualS2Len - this->tilingData->inputParams.preTokens, 0),
                               0);
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize + actualS2Len -
                                 Max(actualS1Len - this->tilingData->inputParams.nextTokens, 0),
                             actualS2Len);
        if (this->s2EndIdx - this->s2StartIdx <= 0) {
            this->s2StartIdx = 0;
            this->s2EndIdx = actualS2Len;
        }
    } else if (this->tilingData->inputParams.sparseType ==
               static_cast<uint8_t>(SparseModeEnum::RIGHT_DOWN_CAUSAL_BAND)) {
        // 右下因果+band: 指定 bandIndex 的 batch 额外加 nextTokens
        if (this->boIdx == this->tilingData->inputParams.bandIndex) {
            this->s2StartIdx = 0;
            this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize + actualS2Len +
                                     this->tilingData->inputParams.nextTokens - actualS1Len,
                                 actualS2Len);
        } else {
            this->s2StartIdx = 0;
            this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize + actualS2Len - actualS1Len, actualS2Len);
        }
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND_LEFT_UP_CAUSAL)) {
        // 左上因果+band
        if (this->boIdx == this->tilingData->inputParams.bandIndex) {
            this->s2StartIdx = 0;
            this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize + actualS2Len -
                                     Max(actualS1Len - this->tilingData->inputParams.nextTokens, 0),
                                 actualS2Len);
        } else {
            this->s2StartIdx = 0;
            this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize, actualS2Len);
        }
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::PREFIX)) {
        // 前缀注意力: 每个 batch 有独立的 prefix 长度
        this->s2StartIdx = 0;
        // s2EndIdx = max(s1块的结束位置 - qLen + kvLen, prefixN[boIdx])
        // 保证至少能看到 prefixN 个 KV token
        this->s2EndIdx =
            Max((this->s1oIdx + 1) * this->cubeS1BaseSize - actualS1Len + actualS2Len,
                ((__gm__ int64_t *)prefixNAddr)[this->boIdx]);
        this->s2EndIdx = Min(this->s2EndIdx, this->s2Size);
        if (this->s2EndIdx - this->s2StartIdx <= 0) {
            this->s2StartIdx = 0;
            this->s2EndIdx = actualS2Len;
        }
    } else {
        // 全量计算 (无稀疏)
        this->s2StartIdx = 0;
        this->s2EndIdx = actualS2Len;
    }

    // 对齐到 8: start 向下, end 向上 → 保证 8 元素向量化能完整处理
    this->s2StartIdx = this->s2StartIdx / 8 * 8;
    this->s2EndIdx = Min(CeilDiv(this->s2EndIdx, 8) * 8, actualS2Len);
    return;
}

// ====================================================================================
// Process: 主计算循环 — AIC/AIV 三级流水线
//
// ─── 流水线结构 ───
//   外循环 (s1): for multiCoreInnerIdx → 每次处理一个 S1 块
//   内循环 (s2): for s2LoopCount → 每次处理一个 S2 块
//
// ─── 三级流水数据流 ───
//   S = Q×Kᵀ (BMM1, AIC) → P = softmax(S) (Vec1, AIV) → O = P×V (BMM2, AIC) → O 写回 (Vec2, AIV)
//
// ─── 流水线调度 ───
//   taskId=0: 只有 BMM1 执行 (预热)
//   taskId=1: BMM1(task1) ∥ Vec1(task0) (启动)
//   taskId≥2: BMM1(taskN) ∥ Vec1(taskN-1) ∥ BMM2(taskN-2) ∥ Vec2(taskN-3) (满流水)
//   末尾: 三段排空, 把剩余任务做完
//
// ─── 同步机制 ───
//   AIC 和 AIV 通过 CrossCoreSetFlag/CrossCoreWaitFlag 同步
//   SYNC_C1_V1: AIC(BMM1完成) 通知 AIV 开始 Vec1
//   SYNC_V1_C2: AIV(Vec1完成) 通知 AIC 开始 BMM2
//   SYNC_C2_V2: AIC(BMM2完成) 通知 AIV 开始 Vec2
// ====================================================================================
template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                       bmm1Format, mmPolicyType, implMode, hasRope>::Process()
{
    // ─── 步骤1: 确定该核负责的工作区间 ───
    // multiCoreInnerIdx 是所有 batch 的所有 S1 块 × N2 × G 展平后的一维索引
    // 各核按 splitFactorSize 均匀切分
    int64_t multiCoreInnerOffset = this->cubeBlockIdx * this->tilingData->multiCoreParams.splitFactorSize;
    int64_t multiCoreInnerLimit = multiCoreInnerOffset + this->tilingData->multiCoreParams.splitFactorSize;
    if (this->tilingData->multiCoreParams.totalSize < multiCoreInnerLimit) {
        multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;  // 最后一块边界裁切
    }

    this->GetS1LoopRange(multiCoreInnerOffset, multiCoreInnerLimit);  // 稀疏场景的 s1 范围调整

    // ─── 步骤2: 定位起始 batch ───
    // 变长核心: 通过逐 batch 累加 s1 外层块数，确定 multiCoreInnerOffset 落在哪个 batch
    this->CalS1OuterSize(multiCoreInnerOffset);

    // ─── 步骤3: 初始化流水线数据结构 ───
    SplitSameABExtraInfo extraInfo[3];  // 三缓冲: 每套记录一组 (S1块, S2块) 的完整计算元数据
    int64_t taskId = 0;                  // 单调递增的任务编号

    // 获取 MTE3→MTE2 事件 ID, 用于 AIV 内部搬运和计算的同步
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    // ─── 步骤4: L1Carry 事件预置 ───
    // 预先将 Q/KV/P 的 MTE1→MTE2 事件设为已完成状态
    // 这样第一轮 WaitFlag 不会阻塞 (还没有数据搬运)
    if ASCEND_IS_AIC {
        if (this->needL1Carry) {
            SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);
            SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0);
            SetFlag<HardEvent::MTE1_MTE2>(P_EVENT0);
            SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + 1);
            SetFlag<HardEvent::MTE1_MTE2>(P_EVENT0 + 1);
        }
    }

    // ==========================================================================
    // ─── 步骤5: 主循环 → 外循环 s1 + 内循环 s2 ───
    // ==========================================================================
    for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit;
         ++multiCoreInnerIdx) {
        this->softMaxCheckRes = SOFTMAX_CHECK_RES_DEFAULT_VALUE;

        // ─── 5a: 确定当前 S1 块参数 ───
        this->ComputeAxisIdx(multiCoreInnerIdx);   // 解出 (boIdx, n2oIdx, goIdx, s1oIdx)

        // ─── 5b: 确定 s2 循环范围 ───
        this->GetS2LoopRange();                    // 根据稀疏模式计算 [s2StartIdx, s2EndIdx)
        int64_t s2LoopLimit = CeilDiv(this->s2EndIdx - this->s2StartIdx, this->s2BaseNratioSize) - 1;

        // ======================================================================
        // ─── 5c: 内循环 → 遍历 s2 方向所有分块 ───
        // ======================================================================
        for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; s2LoopCount++) {
            if (s2LoopCount == 0) {
                this->softmaxPingPongCnt++;  // 软最大双缓冲切换计数
            }

            // ─── 5c-1: 等待 AIV 完成上一轮的 Vec1 ───
            // extraInfo[(taskId+2)%3] 是上上轮 BMM1 输出的那套数据
            // 现在要等它的 Vec1 完成 (不能覆盖还没读的数据)
            if (taskId > 0) {
                if ASCEND_IS_AIV {
                    CrossCoreWaitFlag(SYNC_C1_V1_FLAG[(taskId + 2) % 3]);
                }
            }

            // ─── 5c-2: 装配当前任务的 extraInfo ───
            this->SetExtraInfo(extraInfo[taskId % 3], taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);

            // ─── 5c-3: BMM1 — S = Q × Kᵀ (AIC) ───
            // needNz2Nd: s2 尾块不对齐 64 时用 NZ 格式输出 (标记无效数据列)
            // 否则用 ND 格式直接输出
            if (extraInfo[taskId % 3].needNz2Nd == 1) {
                if ASCEND_IS_AIC {
                    if (this->needL1Carry) {
                        this->ComputeMm1(extraInfo[taskId % 3], this->bmm1NzL1Carry);
                    } else {
                        this->IterateBmm1(extraInfo[taskId % 3], this->bmm1Nz);
                    }
                    // 通知 AIV: BMM1 完成, 可以开始 Vec1
                    CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C1_V1_FLAG[taskId % 3]);
                }
            } else {
                if ASCEND_IS_AIC {
                    if (this->needL1Carry) {
                        this->ComputeMm1(extraInfo[taskId % 3], this->bmm1L1Carry);
                    } else {
                        this->IterateBmm1(extraInfo[taskId % 3], this->bmm1);
                    }
                    CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C1_V1_FLAG[taskId % 3]);
                }
            }

            // ─── 5c-4: Vec1 — P = softmax(S) (AIV) ───
            // 处理上一轮的 BMM1 结果: extraInfo[(taskId+2)%3]
            if (taskId > 0) {
                if ASCEND_IS_AIV {
                    this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
                    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                    // 通知 AIC: Vec1 完成, P 已经写入 stage1Res, 可以开始 BMM2
                    CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V1_C2_FLAG[(taskId + 2) % 3]);
                }
            }

            // ─── 5c-5: 等待 AIC 完成 BMM2 ───
            if (taskId > 1) {
                if ASCEND_IS_AIV {
                    CrossCoreWaitFlag(SYNC_C2_V2_FLAG[(taskId + 1) % 3]);
                }
            }

            // ─── 5c-6: BMM2 — O = P × V (AIC) ───
            // 用 Vec1 输出的 P 乘 V, Cube 内部 L0C 累加
            if (taskId > 0) {
                if ASCEND_IS_AIV {
                    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                }

                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag(SYNC_V1_C2_FLAG[(taskId + 2) % 3]);
                    if (this->needL1Carry) {
                        this->ComputeMm2(extraInfo[(taskId + 2) % 3], this->bmm2L1Carry);
                    } else {
                        this->IterateBmm2(extraInfo[(taskId + 2) % 3], this->bmm2);
                    }
                    // 通知 AIV: BMM2 完成, O 已经写入 mm2Res, 可以开始 Vec2 写回
                    CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C2_V2_FLAG[(taskId + 2) % 3]);
                }
            }

            // ─── 5c-7: Vec2 — O 归一化写回 (AIV) ───
            if (taskId > 1) {
                if ASCEND_IS_AIV {
                    this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
                }
            }
            taskId++;
        }
    }

    // ==========================================================================
    // ─── 步骤6: 排空阶段 — 把流水线中剩余的任务全部完成 ───
    // ==========================================================================
    if (taskId >= 1) {
        this->softMaxCheckRes = SOFTMAX_CHECK_RES_DEFAULT_VALUE;

        // 排空段1: 等待最后 BMM1 → 完成 Vec1 → 启动 BMM2
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(SYNC_C1_V1_FLAG[(taskId + 2) % 3]);
            this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V1_C2_FLAG[(taskId + 2) % 3]);
        }
        if (taskId > 1) {
            if ASCEND_IS_AIV {
                CrossCoreWaitFlag(SYNC_C2_V2_FLAG[(taskId + 1) % 3]);
            }
        }
        if ASCEND_IS_AIV {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag(SYNC_V1_C2_FLAG[(taskId + 2) % 3]);
            if (this->needL1Carry) {
                this->ComputeMm2(extraInfo[(taskId + 2) % 3], this->bmm2L1Carry);
            } else {
                this->IterateBmm2(extraInfo[(taskId + 2) % 3], this->bmm2);
            }
            CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C2_V2_FLAG[(taskId + 2) % 3]);
        }
        // 排空段2: 完成最后一个 Vec2
        if (taskId > 1) {
            if ASCEND_IS_AIV {
                this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
            }
        }
    }
    taskId++;

    // 排空段3: 最终 Vec2
    if (taskId >= 2) {
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(SYNC_C2_V2_FLAG[(taskId + 1) % 3]);
            this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
        }
    }

    // ─── 步骤7: L1Carry 收尾 — 等待所有 L1 数据搬运完成 ───
    if ASCEND_IS_AIC {
        if (this->needL1Carry) {
            if (taskId >= 2) {
                // 多轮执行: 等待两个槽位的 KV 和 P 搬运完成
                WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);
                WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (this->kvL1BufIter % 2));
                WaitFlag<HardEvent::MTE1_MTE2>(P_EVENT0 + (this->pL1BufIter % 2));
                WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (this->kvL1BufIter % 2) ^ 1);  // 另一个槽位
                WaitFlag<HardEvent::MTE1_MTE2>(P_EVENT0 + (this->pL1BufIter % 2) ^ 1);
            } else {
                // 只有一轮: 等待初始槽位
                WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);
                WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0);
                WaitFlag<HardEvent::MTE1_MTE2>(P_EVENT0);
                WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + 1);
                WaitFlag<HardEvent::MTE1_MTE2>(P_EVENT0 + 1);
            }
        }
    }
};

// ====================================================================================
// SetExtraInfo: 每轮 s2 循环装配完整的切片元数据
//
// 把当前 (S1块, S2块) 的所有参数打包到 extraInfo 中,
// 传给 BMM1/Vec1/BMM2/Vec2 使用
// ====================================================================================
template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                       bmm1Format, mmPolicyType, implMode, hasRope>::SetExtraInfo(
    SplitSameABExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    // 基本坐标信息
    extraInfo.s2StartIdx = this->s2StartIdx;
    extraInfo.s2EndIdx = this->s2EndIdx;
    extraInfo.s2LoopCount = s2LoopCount;
    extraInfo.s1oIdx = this->s1oIdx;
    extraInfo.boIdx = this->boIdx;
    extraInfo.n2oIdx = this->n2oIdx;
    extraInfo.goIdx = this->goIdx;
    extraInfo.taskId = taskId;
    extraInfo.taskIdMod2 = taskId % 2;
    extraInfo.s2LoopLimit = s2LoopLimit;
    extraInfo.multiCoreInnerIdx = multiCoreInnerIdx;
    extraInfo.multiCoreInnerIdxMod2 = this->softmaxPingPongCnt % 2;

    // 变长累积量 — 用于计算 GM 地址偏移
    extraInfo.s1SizeAcc = s1SizeAcc;
    extraInfo.s2SizeAcc = s2SizeAcc;
    extraInfo.attenB1SSOffset = attenB1SSOffset;

    // Attention Mask 的 s2 维度大小
    extraInfo.attenMaskS2Size = extraInfo.s2Size;  // 默认用当前 batch 的 s2 长度
    if (this->tilingData->inputParams.attenMaskShapeType == attenMaskS1S2) {
        extraInfo.attenMaskS2Size = this->tilingData->inputParams.s2Size;  // 全局 s2
    } else if (this->tilingData->inputParams.attenMaskShapeType == attenMaskTT) {
        extraInfo.attenMaskS2Size = this->s2SizeSum;  // 所有 batch 的 KV 总长
    }

    // 压缩 Attention Mask 模式
    if (this->tilingData->inputParams.attenMaskCompressMode !=
        static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE)) {
        extraInfo.attenMaskS2Size = this->tilingData->inputParams.attenMaskS2Size;
    }

    // 获取当前 batch 的实际 s1/s2 长度
    GetSeqQlenKvlenByBoidx(extraInfo.boIdx, extraInfo.s1Size, extraInfo.s2Size);

    // 当前 S1 块的实际有效行数 (末尾 batch 可能不足 cubeS1BaseSize)
    extraInfo.cubeS1RealSize = Min(this->cubeS1BaseSize, extraInfo.s1Size - extraInfo.s1oIdx * this->cubeS1BaseSize);

    // 计算 BMM1 尾块参数 (s1 核内切分, s2 尾块, duplicateMask 等)
    this->ComputeBmm1Tail(extraInfo);
}

// ====================================================================================
// ComputeBmm1Tail: 计算尾块的实际大小和对齐信息
//
// 处理三种"不对齐"情况:
//   1. s1 方向: cubeS1BaseSize 块内有两层切分 (AIC sub-core + Vec1 向量化)
//   2. s2 方向: 当 s2 分块是最后一块时，实际大小不足 s2BaseNratioSize
//   3. Cube 格式: s2 尾块不对齐 64 时，需要 duplicateMask 和 NZ 格式
// ====================================================================================
template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                       bmm1Format, mmPolicyType, implMode, hasRope>::ComputeBmm1Tail(SplitSameABExtraInfo &extraInfo)
{
    // ─── s1 方向核内切分 ───
    // AIC 核内两个 sub-core (cubeSubIdx=0/1) 各处理一半的行
    extraInfo.s1RealSize = (extraInfo.cubeS1RealSize + 1) / 2;
    extraInfo.vecCoreOffset = this->cubeSubIdx * extraInfo.s1RealSize;
    if (this->cubeSubIdx == 1) {
        // 第二个 sub-core: 拿剩下的行 (可能比第一个少 1)
        extraInfo.s1RealSize = extraInfo.cubeS1RealSize - extraInfo.s1RealSize;
    }

    // ─── s2 方向尾块判断 ───
    extraInfo.s2RealSize = this->s2BaseNratioSize;        // 默认: 满块大小
    extraInfo.s2AlignedSize = extraInfo.s2RealSize;
    bool isS2Tail = false;
    // 如果当前块 + 下一块的起始位置 ≥ s2EndIdx → 这是 s2 方向的最后一块
    if (extraInfo.s2StartIdx + (extraInfo.s2LoopCount + 1) * extraInfo.s2RealSize >= extraInfo.s2EndIdx) {
        extraInfo.s2RealSize = extraInfo.s2EndIdx - extraInfo.s2LoopCount * extraInfo.s2RealSize - extraInfo.s2StartIdx;
        extraInfo.s2AlignedSize = Align(extraInfo.s2RealSize);  // 向上对齐 (用于数据搬运)
        isS2Tail = true;
    }

    extraInfo.s2RealSizeAlign64 = extraInfo.s2AlignedSize;
    extraInfo.s2RealSizeFloorAlign8 = extraInfo.s2AlignedSize;
    extraInfo.s2LastLoop = false;

    if (isS2Tail) {
        // ─── s2 尾块特殊处理 ───
        // s2RealSizeAlign64: 向上对齐到 64 (Cube 输出要求)
        extraInfo.s2RealSizeAlign64 = CeilDiv(extraInfo.s2RealSize, 64) * 64;
        // s2RealSizeFloorAlign8: 向下对齐到 8 (向量化要求)
        extraInfo.s2RealSizeFloorAlign8 = extraInfo.s2RealSize / 8 * 8;
        extraInfo.s2LastLoop = true;

        // ─── duplicateMask: 标记有效数据列 ───
        // Cube 单元输出 64 对齐，但实际有效数据可能少于 64
        // duplicateMask 的每个 bit 对应一个 64 对齐的列:
        //   bit=0 → 有效列 (有真实数据)
        //   bit=1 → 无效列 (填充数据, 需要 mask 掉)
        // 示例: s2RealSize=70, s2RealSizeAlign64=128, s2RealSizeFloorAlign8=64
        //   mask_hi = (1<<(128-64)) - 1 = 全1
        //   mask_lo = (1<<(70-64)) - 1   = 0b111111
        //   duplicateMask = mask_hi & (~mask_lo) → 无效列 bit 为 1
        extraInfo.duplicateMask =
            (((uint64_t)1 << (extraInfo.s2RealSizeAlign64 - extraInfo.s2RealSizeFloorAlign8)) - 1) &
            (~(((uint64_t)1 << (extraInfo.s2RealSize - extraInfo.s2RealSizeFloorAlign8)) - 1));
        if (extraInfo.s2RealSizeAlign64 - extraInfo.s2RealSizeFloorAlign8 == 64) {
            // 对齐差距正好 64 → 所有高位都无效
            extraInfo.duplicateMask =
                0xffffffffffffffff & (~(((uint64_t)1 << (extraInfo.s2RealSize - extraInfo.s2RealSizeFloorAlign8)) - 1));
        }
    }

    // ─── Vec1 向量化切块策略 ───
    // 当 s2 非常小 (<8) 时, 每次遍历 s1 base size 有访存问题 → 改为一次计算
    if (unlikely(extraInfo.s2RealSize < fp32BaseSize)) {
        extraInfo.vec1S1BaseSize = extraInfo.s1RealSize;  // s1 一次处理完
        extraInfo.realSplitN = 1;
    } else {
        // 向量化切块: 最大化利用 UB 容量
        // 1024 / s2RealSizeAlign64 表示 UB 能放多少行, ×8 是列方向向量化因子
        extraInfo.vec1S1BaseSize = Min(1024 / extraInfo.s2RealSizeAlign64 * 8, extraInfo.s1RealSize);
        extraInfo.realSplitN = CeilDiv(extraInfo.s1RealSize, extraInfo.vec1S1BaseSize);
    }

    // ─── Vec2 向量化切块 ───
    if (this->dSizeAlign16 > 64) {
        extraInfo.vec2S1BaseSize = 1024 / this->dSizeAlign16 * 8;
    } else {
        extraInfo.vec2S1BaseSize = Min(128, extraInfo.s1RealSize);
    }

    // ─── 是否需要 NZ→ND 格式转换 ───
    // s2 对齐 64 → BMM1 直接输出 ND 格式
    // s2 不对齐 64 → BMM1 输出 NZ 格式, Vec1 做 NZ→ND 转换
    extraInfo.needNz2Nd = (extraInfo.s2RealSize % 64 == 0) ? 0 : 1;
    return;
}

// ====================================================================================
// ComputeMm1: L1Carry 版 BMM1 — S = Q × Kᵀ
//
// 核心优化: Q 在 L1 常驻, s2 循环只拷一次 (s2LoopCount==0 时)
// K 沿 N(s2) 方向切 256, 每次拷一块到 L1 双缓冲
// Cube 单元从 L1 直读 → 省掉重复的 GM→Cube 搬运
//
// 数据流:
//   Q: GM → L1 (qL1Tensor, 常驻)
//   K: GM → L1 (kvL1Tensor, N 切片双缓冲)
//   S: Cube → GM (mm1Res, 供 Vec1 使用)
// ====================================================================================
template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
template <typename T2>
__aicore__ inline void
FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, implMode, hasRope>::ComputeMm1(SplitSameABExtraInfo &extraInfo,
                                    matmul::MatmulImpl<a1TypeL1Carry, b1TypeL1Carry, T2, bias1TypeL1Carry, CFG_CUSTOM_EXCEED> &bmm1) {

    // ─── 步骤1: 设置矩阵形状 ───
    // cubeS1RealSize: s1 方向实际行数 (M 维度)
    // mm1Kb:         总 K 维度 (mm1Kb1 + mm1Kb2, RoPE 时有两段)
    // mm1Ka1:        Q 的 K 维度 (= N2*G*D)
    // mm1Kb1:        K 的 K 维度 (= N2*D)
    // s2RealSize:    s2 方向列数 (N 维度)
    bmm1.SetOrgShape(extraInfo.cubeS1RealSize, this->mm1Kb, this->mm1Ka1, this->mm1Kb1, extraInfo.s2RealSize);

    LocalTensor<INPUT_T> qTensor = qL1Tensor;

    // ─── 步骤2: s2LoopCount==0 时拷贝 Q 到 L1 ───
    // 整个 s2 内循环中 Q 只有这一次 GM→L1 搬运
    if (extraInfo.s2LoopCount == 0) {
        WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);   // 等上一轮 Q 用完
        CopyInMm1AToL1(qTensor, extraInfo);          // Q: GM → L1 (含 ND→NZ 格式转换)
        SetFlag<HardEvent::MTE2_MTE1>(Q_EVENT0);     // 通知搬运完成
        WaitFlag<HardEvent::MTE2_MTE1>(Q_EVENT0);    // 等搬运完成
    }

    // ─── 步骤3: 计算 K 的 GM 偏移 ───
    // bOffset:   前面所有 batch 的 KV token 数 × n2D
    // s2Offset:  当前 s2 块的起始位置 × n2D
    // n2Offset:  head 方向偏移
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;

    bOffset = extraInfo.s2SizeAcc * this->n2D;
    s2Offset = extraInfo.s2StartIdx * this->n2D + extraInfo.s2LoopCount * this->s2BaseNratioN2D;
    n2Offset = extraInfo.n2oIdx * this->dSize;

    this->kCoreOffset = bOffset + n2Offset + s2Offset;

    // ─── 步骤4: N 方向循环 (s2 方向切片, 每次处理 256 列) ───
    const uint32_t nSplitSize = 256;  // 每次处理 256 个 KV token
    uint32_t nloops = (extraInfo.s2RealSize + nSplitSize - 1) / nSplitSize;
    uint32_t nTail = extraInfo.s2RealSize - (nloops - 1) * nSplitSize;
    uint32_t subNSizeAct = nSplitSize;

    for (uint32_t n = 0; n < nloops; n++) {
        if (n == nloops - 1) {
            subNSizeAct = nTail;  // 最后一块的实际大小
        }

        // ─── 4a: 等待 L1 槽位空闲 → 拷贝 K 的 N×D 切片 ───
        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (kvL1BufIter % 2));
        LocalTensor<INPUT_T> kTensor = kvL1Tensor[(kvL1BufIter % 2) * (L1_KV_SIZE / sizeof(INPUT_T))];
        CopyInMm1BToL1(kTensor, extraInfo, n, subNSizeAct, nSplitSize);  // K 切片: GM → L1

        SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + (kvL1BufIter % 2));   // 搬运完成
        WaitFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + (kvL1BufIter % 2));  // 等搬运完成

        // ─── 4b: 设置 Cube 矩阵乘参数 ───
        bmm1.SetTensorA(qTensor);          // A = Q
        bmm1.SetTensorB(kTensor, true);    // B = K ᵀ
        bmm1.SetTail(extraInfo.cubeS1RealSize, subNSizeAct);

        // ─── 4c: 发射 Cube 计算 ───
        if(extraInfo.needNz2Nd == 1){
            bmm1.template IterateAll<false>(this->mm1Res[extraInfo.taskIdMod2][extraInfo.cubeS1RealSize * n * nSplitSize]);
        }else{
            bmm1.template IterateAll<false>(this->mm1Res[extraInfo.taskIdMod2][n * nSplitSize]);
        }
        bmm1.End();

        // ─── 4d: 释放 K 的 L1 槽位, 双缓冲切换到下一个 ───
        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (kvL1BufIter % 2));
        kvL1BufIter++;
    }

    // ─── 步骤5: s2 内循环最后一轮, 释放 Q 的 L1 槽位 ───
    if (extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
        SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);
    }
}

// ====================================================================================
// ComputeMm2: L1Carry 版 BMM2 — O = P × V
//
// 双重循环:
//   外层 N: V 的列方向 (d2Size 切 128) → 对应输出 O 的列
//   内层 K: V 的行方向 (s2 切 192)     → 对应 s2 累加方向
//
// P 来自 Vec1 的 stage1Res (GM, NZ 格式)
// V 来自 valueGm (GM, ND 格式)
//
// Cube L0C 做累加: k==0 时覆盖, k>0 时累加
// 所有 K 块处理完 → GetTensorC 读出最终结果到 mm2Res
// ====================================================================================
template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                                       bmm1Format, mmPolicyType, implMode, hasRope>::ComputeMm2(SplitSameABExtraInfo &extraInfo,
                                                            matmul::MatmulImpl<a2TypeL1Carry, b2TypeL1Carry, c2NzTypeL1Carry, bias2TypeL1Carry, CFG_MDL_EXCEED> &bmm2)
{
    // ─── 步骤1: 计算 V 的 GM 偏移 ───
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;

    bOffset = extraInfo.s2SizeAcc * this->n2D2;
    s2Offset = extraInfo.s2StartIdx * this->n2D2 + extraInfo.s2LoopCount * this->s2BaseNratioSize * this->n2D2;
    n2Offset = extraInfo.n2oIdx * this->d2Size;

    this->vCoreOffset = bOffset + n2Offset + s2Offset;

    // ─── 步骤2: K 方向切分 (V 的 s2 方向, 每次 192) ───
    uint32_t kSplitSize = 192;
    uint32_t kloops = (extraInfo.s2RealSize + kSplitSize - 1) / kSplitSize;
    uint32_t kTail = extraInfo.s2RealSize - (kloops - 1) * kSplitSize;
    uint32_t subKSizeAct = kSplitSize;
    uint32_t subKSizeActAligned = CeilDiv(subKSizeAct, 16) * 16;

    // ─── 步骤3: N 方向切分 (V 的 d2 方向, 每次 128) ───
    uint32_t nSplitSize = 128;
    uint32_t nLoops = CeilDiv(this->d2Size, nSplitSize);
    uint32_t nTail = this->d2Size - (nLoops - 1) * nSplitSize;
    uint32_t subNSizeAct = nSplitSize;

    LocalTensor<T> oTensor = oL0CTensor;  // Cube L0C 累加器

    // 保存双缓冲迭代状态 (因为内层 K 循环也会推进迭代器, 外层 N 需要重置)
    uint32_t tmpTNDkvL1BufIter = kvL1BufIter;
    uint32_t tmpTNDpL1BufIter = pL1BufIter;

    // ─── 步骤4: 双重循环 ───
    for (auto n = 0; n < nLoops; ++n) {
        bool isLastN = n == nLoops - 1;
        if (isLastN) {
            subNSizeAct = nTail;
        }

        // 重置双缓冲迭代器 (每个 N 循环步独立)
        kvL1BufIter = tmpTNDkvL1BufIter;
        pL1BufIter = tmpTNDpL1BufIter;
        subKSizeAct = kSplitSize;
        subKSizeActAligned = CeilDiv(subKSizeAct, 16) * 16;

        // ─── 内层 K 循环 (s2 累加方向) ───
        for (uint32_t k = 0; k < kloops; k++) {
            if (k == kloops - 1) {
                subKSizeAct = kTail;
                subKSizeActAligned = CeilDiv(kTail, 16) * 16;
            }

            // 4a: 拷贝 P 的 K 切片到 L1 (stage1Res → L1, NZ 格式)
            WaitFlag<HardEvent::MTE1_MTE2>(P_EVENT0 + (pL1BufIter % 2));
            LocalTensor<INPUT_T> pTensor = pL1Tensor[(pL1BufIter % 2) * (L1_P_SIZE / sizeof(INPUT_T))];
            CopyInMm2AToL1(pTensor, extraInfo, k, subKSizeActAligned, kSplitSize);

            SetFlag<HardEvent::MTE2_MTE1>(P_EVENT0 + (pL1BufIter % 2));
            WaitFlag<HardEvent::MTE2_MTE1>(P_EVENT0 + (pL1BufIter % 2));

            // 4b: 拷贝 V 的 K×N 切片到 L1 (valueGm → L1, ND→NZ 格式)
            WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (kvL1BufIter % 2));
            LocalTensor vTensor = kvL1Tensor[(kvL1BufIter % 2) * (L1_KV_SIZE / sizeof(INPUT_T))];
            CopyInMm2BToL1(vTensor, k, subKSizeAct, kSplitSize, n, isLastN);

            SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + (kvL1BufIter % 2));
            WaitFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + (kvL1BufIter % 2));

            // 4c: Cube 矩阵乘
            // SetOrgShape: M=s1行数, N=d2列数, 满K=切分大小, 实际K=当前K块大小
            bmm2.SetOrgShape(extraInfo.cubeS1RealSize, this->d2Size, subKSizeActAligned, subKSizeActAligned, this->d2Size);
            bmm2.SetTail(extraInfo.cubeS1RealSize, subNSizeAct, subKSizeAct);

            bmm2.SetTensorA(pTensor);   // P: [s1, k]
            bmm2.SetTensorB(vTensor);   // V: [k, d2]

            // k==0: 覆盖 L0C,  k>0: 累加到 L0C
            bmm2.template Iterate(k!=0, oTensor);

            // 4d: 释放 L1 槽位
            SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (kvL1BufIter % 2));
            SetFlag<HardEvent::MTE1_MTE2>(P_EVENT0 + (pL1BufIter % 2));
            kvL1BufIter++;
            pL1BufIter++;
        }

        // ─── K 循环结束: 从 L0C 读出累加结果到 mm2Res ───
        bmm2.template GetTensorC(this->mm2Res[extraInfo.taskIdMod2][extraInfo.cubeS1RealSize * nSplitSize * n]);
    }
    bmm2.End();
}

// ====================================================================================
// CopyInMm1AToL1: Q 矩阵 GM → L1
// 变长地址计算: offset = s1SizeAcc * n2GD + s1Block * n2GD + head_offset + group_offset
// ====================================================================================
template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void
FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, implMode, hasRope>::CopyInMm1AToL1(LocalTensor<INPUT_T> &l1Tensor,
                                                                                    SplitSameABExtraInfo &extraInfo)
{
    int64_t bOffset = 0;
    int64_t s1Offset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;

    // bOffset:  前面所有 batch 的 query token 数 × n2GD
    bOffset = extraInfo.s1SizeAcc * this->n2GD;
    // s1Offset: 当前 S1 块的偏移
    s1Offset = extraInfo.s1oIdx * this->s1BaseN2GD;
    // n2Offset: head 方向偏移
    n2Offset = extraInfo.n2oIdx * this->gD;
    // gOffset:  group 方向偏移
    gOffset = extraInfo.goIdx * this->dSize;

    this->qCoreOffset = bOffset + n2Offset + gOffset + s1Offset;
    extraInfo.qCoreOffset = this->qCoreOffset;
    auto srcGm = this->queryGm[extraInfo.qCoreOffset];

    // ND→NZ 格式转换拷贝: [cubeS1RealSize, dSize] 以 n2GD 为 stride
    CopyGmToL1(l1Tensor, srcGm, extraInfo.cubeS1RealSize, this->dSize, this->n2GD);
}

// ====================================================================================
// CopyInMm1BToL1: K 矩阵 GM → L1 (N 切片)
// 当前 K 片对应 offset: kCoreOffset + subNid * nSplitSize * n2D → [subNSize, dSize]
// ====================================================================================
template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void
FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, implMode, hasRope>::CopyInMm1BToL1(LocalTensor<INPUT_T> &bL1Tensor,
                                                                                const SplitSameABExtraInfo &extraInfo, uint32_t subNid, uint32_t subNSize, uint32_t nSplitSize)
{
    auto srcGm = this->keyGm[this->kCoreOffset + subNid * nSplitSize * this->n2D];
    CopyGmToL1(bL1Tensor, srcGm, subNSize, this->dSize, this->n2D);
}

// ====================================================================================
// CopyInMm2AToL1: P 矩阵 stage1Res(GM) → L1 (K 切片)
// P 数据已在 stage1Res 中以 NZ 格式存储, 直接用 DataCopy
// ====================================================================================
template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void
FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, implMode, hasRope>::CopyInMm2AToL1(LocalTensor<INPUT_T> &al1Tensor,
                                                                                    SplitSameABExtraInfo &extraInfo, uint32_t subKid, uint32_t subKSize, uint32_t kSplitSize)
{
    int64_t cubeS1RealSizeAligned = CeilDiv(extraInfo.cubeS1RealSize, 16) * 16;
    auto srcGm = this->stage1Res[extraInfo.taskIdMod2][cubeS1RealSizeAligned * subKid * kSplitSize];
    DataCopy(al1Tensor, srcGm, cubeS1RealSizeAligned * subKSize);
}

// ====================================================================================
// CopyInMm2BToL1: V 矩阵 GM → L1 (K×N 切片)
// vCoreOffset + subKid * kSplitSize * n2D2 + nStartOffset → [subKSize, nSplitSize]
// ====================================================================================
template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void
FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, implMode, hasRope>::CopyInMm2BToL1(LocalTensor<INPUT_T> &bL1Tensor,
                                                                                uint32_t subKid, uint32_t subKSize, uint32_t kSplitSize,
                                                                                uint32_t nLoopIdx, bool isLastN)
{
    uint32_t nSplitSize = 128;
    uint32_t nStartOffset = nLoopIdx * nSplitSize;
    auto srcGm = this->valueGm[this->vCoreOffset + subKid * kSplitSize * this->n2D2 + nStartOffset];
    if (isLastN) {
        nSplitSize = this->d2Size - nLoopIdx * nSplitSize;  // 尾块实际大小
    }
    CopyGmToL1(bL1Tensor, srcGm, subKSize, nSplitSize, this->n2D2);
}

// ====================================================================================
// CopyGmToL1: 通用 GM→L1 拷贝 (含 ND→NZ 格式转换)
//
// 为什么需要 ND→NZ:
//   GM 中数据是 ND (行主序), 但 Cube 单元处理 NZ (16×16 fractal 格式)
//   搬运时直接做格式转换，省掉后续的专门转换步骤
//
// srcDstride: GM 中一行到下一行的字节跨度 (用于跳过 batch/head 维度)
// dstNzC0Stride: NZ 格式的 C0 步长 = CeilDiv(N, 16) * 16
// ====================================================================================
template <LayOutTypeEnum layOutType, bool hasPse, bool hasAtten, bool hasDrop, typename INPUT_T,
          typename T, CubeFormat bmm1Format, MmPolicyType mmPolicyType, ImplModeEnum implMode, bool hasRope>
__aicore__ inline void
FlashAttentionVarLenScoreSameAB<layOutType, hasPse, hasAtten, hasDrop, INPUT_T, T,
                                bmm1Format, mmPolicyType, implMode, hasRope>::CopyGmToL1(LocalTensor<INPUT_T> &l1Tensor, GlobalTensor<INPUT_T> &gmSrcTensor,
                                                     uint32_t srcN, uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN;                          // 行数
    nd2nzPara.dValue = srcD;                          // 列数
    nd2nzPara.srcDValue = srcDstride;                 // GM 中每行的实际跨度 (stride, 跳过 batch/head)
    nd2nzPara.dstNzC0Stride = CeilDiv(srcN, 16) * 16; // NZ 格式 C0 步长: 16 对齐的行数
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    AscendC::DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

#endif // FLASH_ATTENTION_VAR_LEN_SCORE_SAB_H
