#ifndef BSA_SELECT_BLOCK_MASK_SOFTMAX_H
#define BSA_SELECT_BLOCK_MASK_SOFTMAX_H


#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bsa_select_block_mask_common.h"
#include "bsa_select_block_mask_tiling_data.h"


template <typename BSAT>
class BSAVecSmService {
public:
    using T = float;                      //< 中间计算类型 (float, FP32)
    using IN_T = typename BSAT::inputT;   //< 输入数据类型 (FP16/BF16)
    using OUT_T = typename BSAT::outputT; //< 输出数据类型 (INT8)

    static constexpr BSALayout LAYOUT_Q = BSAT::layoutQ;   // BNSD or TND
    static constexpr BSALayout LAYOUT_KV = BSAT::layoutKV; // BNSD or TND

    __aicore__ inline BSAVecSmService(){};
    __aicore__ inline void InitParams(const BSAConstInfo &constInfo,
                                      const optiling::BSASelectBlockMaskTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers(TBuf<>* uBuf_);
    // __aicore__ inline void InitGM(GlobalTensor<float> &qCmpGm, GlobalTensor<float> &kCmpGm,
    //                                GlobalTensor<half> &softmaxOutGm, GlobalTensor<T> &ScoreFp32Gm,
    //                                GlobalTensor<uint8_t> &maskTmpGm,
    //                                GlobalTensor<IN_T> &queryGm, GlobalTensor<IN_T> &keyGm,
    //                                GlobalTensor<int64_t> &actualBlockLenQGm,
    //                                GlobalTensor<int64_t> &actualBlockLenKVGm,
    //                                GlobalTensor<OUT_T> &maskOutGm);
    __aicore__ inline void InitGM(
        GlobalTensor<T> &ScoreFp32Gm, GlobalTensor<half> &attnScoreFp16Gm, GlobalTensor<half> &softmaxOutGm);


    /**
     * @brief Online Softmax 第一遍 —— 增量更新 m(最大值) 和 l(归一化分母)
     *
     * Online Softmax 算法避免一次性加载全部注意力分数。
     * 对于每个 K Chunk 的 score 分块，增量更新:
     *   m_new = max(m, max(score_chunk))                          —— 行最大值
     *   l_new = l * exp(m - m_new) + Σexp(score_chunk - m_new)   —— 指数和
     *
     * 完成后将原始 score 分块写回 softmaxTmpGm，供第二遍使用。
     *
     * @param qChunkStart Q Chunk 起始 Block 索引
     * @param qChunkSize  Q Chunk 包含的 Block 数量
     * @param kChunkStart K Chunk 起始 Block 索引
     * @param kChunkSize  K Chunk 包含的 Block 数量
     * @param mFinal      [in/out] 行最大值 (跨 K Chunk 累积)
     * @param lFinal      [in/out] 行指数和 (跨 K Chunk 累积)
     */
    __aicore__ inline void OnlineSoftmaxFirstPassChunk(uint32_t qChunkStart, uint32_t qChunkSize, uint32_t kChunkStart,
                                                       uint32_t kChunkSize);

    /**
     * @brief Softmax 第二遍 —— 归一化 + FP32→FP16 类型转换
     *
     * 利用第一遍求得的 mFinal 和 lFinal，对所有 YBlocks 个 score 逐元素:
     *   attn_score = exp(score - mFinal) / lFinal
     *   Cast FP32 → FP16
     *
     * 最终 FP16 score 写入 softmaxOutGm[qBlockIdx * yBlocks :]。
     *
     */
    __aicore__ inline void SoftmaxSecondPassAndCast(
        uint32_t qChunkStart, uint32_t qChunkSize, uint32_t kChunkStart, uint32_t kChunkSize, uint32_t batchIdx, uint32_t headIdx);



private:
    BSAConstInfo constInfo;                                     //< 运行时常量缓存
    const optiling::BSASelectBlockMaskTilingData *__restrict tilingData; //< Host 端 Tiling 数据

    static constexpr int32_t BUFFER = 1;   // 开启double buffer
    uint32_t vecSubBlockIdx = 0; // vector核在aic的序号： 0 or 1
    static constexpr uint32_t CV_EXEC_RATIO = 1; // cube执行次数与vec执行次数的比例

    // ---- GM 全局张量引用 ----
    GlobalTensor<half> attnScoreFp16GmTensor;              //< 注意力分数 [xBlocks * yBlocks] (FP32→FP16)
    GlobalTensor<T> scoreFp32GmTensor;                 //< Softmax 临时缓冲 (FP32)
    GlobalTensor<half> softmaxOutGmTensor; 

    TBuf<> uBuf_; //< 统一 UB 缓冲区 (192KB)

    LocalTensor<uint8_t> reduceSharedTemp; // reduce相关 高阶接口的临时存储空间，暂定24k
    LocalTensor<T> tmpMax;              //< Softmax Max (FP32) * 8 (即FLOAT_DATA_BLOCK_NUM)
    LocalTensor<T> tmpSum;              //< Softmax Sum (FP32) * 8 (即FLOAT_DATA_BLOCK_NUM)
    LocalTensor<T> globalMax;           //< Softmax old Max (FP32) 在一轮k里会驻守 <= 64
    LocalTensor<T> globalSum;           //< Softmax old Sum (FP32) 在一轮k里会驻守 <= 64
    LocalTensor<T> globalMaxBroc;       //< Softmax Max (FP32) 在一轮k里会驻守 <= 64
    LocalTensor<T> globalSumBroc;       //< Softmax Max (FP32) 在一轮k里会驻守 <= 64
    LocalTensor<T> diffMax;

    LocalTensor<T> scoreUb;             // out shape [x, y] < [64, 128 * 5]
    LocalTensor<half> scoreFp16Ub;      // 上一轮矩阵乘结果 input [x, y] < [64, 128 * 5]

    // ---- 硬件同步事件 ----
    event_t eventIdMte2ToV;    // MTE2→Vector 同步 (GM→UB 数据搬运完成通知 Vector)
    event_t eventIdVToMte3;    // Vector→MTE3 同步 (Vector 计算完成通知 MTE3 写回)
    // event_t eventIdMte3ToV;    // MTE3→Vector 同步 (MTE3 写回完成通知 Vector)
    event_t eventIdVToMte2;    // Vector→MTE2 同步 Vector 计算完成通知 MTE2 写入)
    event_t eventIdMte3ToMte2; // MTE3→MTE2   同步 (MTE3 写回完成通知 MTE2 写入)

    event_t eventSft = (event_t)3;

};

/**
 * @brief 初始化 Vector 服务运行参数
 * @details 将 Host 端 Tiling 阶段计算得到的常量信息和张量配置加载到 Vector 服务实例中，
 *          供后续所有 Pooling / Softmax / MaskGen 操作使用。
 * @tparam BSAT 算子模板类型（包含数据类型和 Layout 信息）
 * @param constInfo BSA 常量信息（含 blockShape、dSize、yBlocks 等推导结果）
 * @param tilingData Host 端 Tiling 阶段生成的多核切分参数
 */
template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::InitParams(const BSAConstInfo &constInfo,
                                                          const optiling::BSASelectBlockMaskTilingData *__restrict tilingData)
{
    this->constInfo = constInfo;
    this->tilingData = tilingData;
    vecSubBlockIdx = constInfo.subBlockIdx; // 0 or 1
    // 同步事件赋值
//     eventIdMte2ToV;    // MTE2→Vector 同步 (GM→UB 数据搬运完成通知 Vector)
//     eventIdVToMte3;    // Vector→MTE3 同步 (Vector 计算完成通知 MTE3 写回)
//     // event_t eventIdMte3ToV;    // MTE3→Vector 同步 (MTE3 写回完成通知 Vector)
//     eventIdVToMte2;    // Vector→MTE2 同步 Vector 计算完成通知 MTE2 写入)
//     eventIdMte3ToMte2; // MTE3→MTE2   同步 (MTE3 写回完成通知 MTE2 写入)
}


template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::InitBuffers(TBuf<>* uBuf_)
{
    // ub 划分
    // todo 为了性能优化，可能多次cube计算，进行一次vector计算
    // 根据UB计算，最多可以5次cube，一次vector计算， 即score ub 占用64 * 128 * 5 * 4 = 160k
    // 剩余空间还需要给m l 以及临时辅助reduce计算的空间
    // 不考虑double buffer版
    // 1个cube计算128 * 128， 计算5次
    // 一个vector 计算64 * 128 * 5 
    // max 64 * 4 = 0.25k
    // sum 64 * 4 = 0.25k
    // dif (max - global_max) 64 * 4 = 0.25k
    // score 64 * 128 * 4 * 5 = 160 k // 1 个cube核计算量均分给俩个vectore核
    // global Max 64 * 4 = 0.25k // 长期驻守
    // global L 64 * 4 = 0.25kk // 长期驻守
    // max_broc 64 * 8 * 4 = 2k
    // sum_broc 64 * 8 * 4 = 2k
    // shareTempReduce 24k

    uint32_t ubOffset = 0;
    uint32_t chunkSize = Q_CHUNK_SIZE;

    // 先不考虑double buffer
    globalMax = uBuf_->GetWithOffset<T>(chunkSize, ubOffset);
    ubOffset += chunkSize * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    globalMaxBroc = uBuf_->GetWithOffset<T>(chunkSize * FLOAT_DATA_BLOCK_NUM, ubOffset);
    ubOffset += chunkSize * FLOAT_DATA_BLOCK_NUM * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    globalSum = uBuf_->GetWithOffset<T>(chunkSize, ubOffset);
    ubOffset += chunkSize * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    globalSumBroc = uBuf_->GetWithOffset<T>(chunkSize * FLOAT_DATA_BLOCK_NUM, ubOffset);
    ubOffset += chunkSize * FLOAT_DATA_BLOCK_NUM * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    tmpMax = uBuf_->GetWithOffset<T>(chunkSize, ubOffset);
    ubOffset += chunkSize * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    diffMax = uBuf_->GetWithOffset<T>(chunkSize, ubOffset);
    ubOffset += chunkSize * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    tmpSum = uBuf_->GetWithOffset<T>(chunkSize, ubOffset);
    ubOffset += chunkSize * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    uint32_t scoreElements = (Q_CHUNK_SIZE / 2)  * K_CHUNK_SIZE * CV_EXEC_RATIO;
    scoreUb = uBuf_->GetWithOffset<T>(scoreElements, ubOffset);
    scoreFp16Ub = uBuf_->GetWithOffset<half>(scoreElements, ubOffset); // 复用fp32空间

    ubOffset += scoreElements * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    uint32_t sharedReduceEles =  24 * 1024 / sizeof(uint8_t);
    reduceSharedTemp = uBuf_->GetWithOffset<uint8_t>(sharedReduceEles, ubOffset);
    ubOffset += scoreElements * sizeof(uint8_t);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));
}


template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::InitGM(
    GlobalTensor<T> &ScoreFp32Gm, GlobalTensor<half> &attnScoreFp16Gm, GlobalTensor<half> &softmaxOutGm)
{
    this->attnScoreFp16GmTensor = attnScoreFp16Gm;
    this->scoreFp32GmTensor = ScoreFp32Gm;
    this->softmaxOutGmTensor = softmaxOutGm;
}

__aicore__ inline
void SubBroadcast(
    LocalTensor<float> const &ubOut, LocalTensor<float> const &ubIn0, LocalTensor<float> const &ubIn1, uint64_t row, uint64_t col)
{
    // 默认 col 安装32字节对齐
    // 分核逻辑的关系，矩阵shape不会超过[128, 128 * 5], 若启动俩个vector，shape不会超过[64, 128 * 5]
    // 所以，一次sub， 最大可计算[255, 64], 迭代次数为行数, 一次迭代计算元素大小最多为64
    // 所以，对列按照64切分，循环sub计算
    // constexpr uint32_t maxRepeatTimes = 255;
    // constexpr uint32_t eleNumPerBlk = FLOAT_DATA_BLOCK_NUM;

    // // FIX: col is runtime parameter, blkNumPerColumn must be const not constexpr
    // uint32_t blkNumPerColumn = col / eleNumPerBlk;
    // AscendC::BinaryRepeatParams repeatParams;
    // repeatParams.dstBlkStride = 1;
    // repeatParams.src0BlkStride = 1;
    // repeatParams.src1BlkStride = 1;
    // repeatParams.dstRepStride = blkNumPerColumn;
    // repeatParams.src0RepStride = blkNumPerColumn;
    // repeatParams.src1RepStride = 0;

    // constexpr uint32_t rowNumPerCompute = maxRepeatTimes;
    // constexpr uint32_t colNumPerCompute = FLOAT_REPEAT_NUM;
    // for (uint32_t rowOffset = 0; rowOffset < row; rowOffset += rowNumPerCompute) {
    //     uint32_t residueM = row - rowOffset;
    //     uint32_t temprepeatTimes = (residueM > rowNumPerCompute) ? rowNumPerCompute : residueM;
    //     uint8_t repeatTimes = static_cast<uint8_t>(temprepeatTimes);
    //     for (uint32_t colOffset = 0; colOffset < col; colOffset += colNumPerCompute) {
    //         uint32_t residueN = col - colOffset;
    //         uint64_t mask = (residueN > colNumPerCompute) ? colNumPerCompute : residueN;
    //         AscendC::Sub(
    //             ubOut[rowOffset * col + colOffset],
    //             ubIn0[rowOffset * col + colOffset],
    //             ubIn1[colOffset],
    //             mask, repeatTimes, repeatParams
    //         );
    //     }
    // }

        uint32_t countEachRepeat = 256 / sizeof(float); // 每次迭代计算的元素 64
        uint32_t colLoop = (col + countEachRepeat - 1) / countEachRepeat;  // 上取整
        uint32_t remain = col % countEachRepeat;
        uint64_t mask = countEachRepeat; // 参与计算的元素个数
        uint8_t repeatTimes = row;
        AscendC::BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = col / FLOAT_DATA_BLOCK_NUM;
        repeatParams.src0RepStride = col / FLOAT_DATA_BLOCK_NUM;
        repeatParams.src1RepStride = 1;

        for (uint32_t i = 0; i < colLoop; i++) {
            if (i == colLoop - 1 && remain != 0) {
                mask = remain;
            }
            AscendC::Sub(
                ubOut[i * countEachRepeat], ubIn0[i * countEachRepeat], ubIn1[0], mask, repeatTimes, repeatParams);
            AscendC::PipeBarrier<PIPE_V>();
        }
}

__aicore__ inline
void DivBroadcast(
    LocalTensor<float> const &ubOut, LocalTensor<float> const &ubIn0, LocalTensor<float> const &ubIn1, uint64_t row, uint64_t col)
{
    // 默认 col 安装32字节对齐
    // 分核逻辑的关系，矩阵shape不会超过[128, 128 * 5], 若启动俩个vector，shape不会超过[64, 128 * 5]
    // 所以，一次div， 最大可计算[255, 64], 迭代次数为行数, 一次迭代计算元素大小最多为64
    // 所以，对列按照64切分，循环div计算
    // constexpr uint32_t maxRepeatTimes = 255;
    // constexpr uint32_t eleNumPerBlk = FLOAT_DATA_BLOCK_NUM;

    // // FIX: col is runtime parameter, blkNumPerColumn must be const not constexpr
    // const uint32_t blkNumPerColumn = col / eleNumPerBlk;
    // AscendC::BinaryRepeatParams repeatParams;
    // repeatParams.dstBlkStride = 1;
    // repeatParams.src0BlkStride = 1;
    // repeatParams.src1BlkStride = 1;
    // repeatParams.dstRepStride = blkNumPerColumn;
    // repeatParams.src0RepStride = blkNumPerColumn;
    // repeatParams.src1RepStride = 0;

    // constexpr uint32_t rowNumPerCompute = maxRepeatTimes;
    // constexpr uint32_t colNumPerCompute = FLOAT_REPEAT_NUM;
    // for (uint32_t rowOffset = 0; rowOffset < row; rowOffset += rowNumPerCompute) {
    //     uint32_t residueM = row - rowOffset;
    //     uint8_t temprepeatTimes = (residueM > rowNumPerCompute) ? rowNumPerCompute : residueM;
    //     uint8_t repeatTimes = static_cast<uint8_t>(temprepeatTimes);
    //     for (uint32_t colOffset = 0; colOffset < col; colOffset += colNumPerCompute) {
    //         uint32_t residueN = col - colOffset;
    //         uint64_t mask = (residueN > colNumPerCompute) ? colNumPerCompute : residueN;
    //         AscendC::Div(
    //             ubOut[rowOffset * col + colOffset],
    //             ubIn0[rowOffset * col + colOffset],
    //             ubIn1[colOffset],
    //             mask, repeatTimes, repeatParams
    //         );
    //     }
    // }

        uint32_t countEachRepeat = 256 / sizeof(float); // 每次迭代计算的元素 64
        uint32_t colLoop = (col + countEachRepeat - 1) / countEachRepeat;  // 上取整
        uint32_t remain = col % countEachRepeat;
        uint64_t mask = countEachRepeat; // 参与计算的元素个数
        uint8_t repeatTimes = row;
        AscendC::BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = col / FLOAT_DATA_BLOCK_NUM;
        repeatParams.src0RepStride = col / FLOAT_DATA_BLOCK_NUM;
        repeatParams.src1RepStride = 1;

        for (uint32_t i = 0; i < colLoop; i++) {
            if (i == colLoop - 1 && remain != 0) {
                mask = remain;
            }
            AscendC::Div(
                ubOut[i * countEachRepeat], ubIn0[i * countEachRepeat], ubIn1[0], mask, repeatTimes, repeatParams);
            AscendC::PipeBarrier<PIPE_V>();
        }
}

/**
 * @brief Online Softmax 第一遍——对一个 K-Chunk 的 Score 块完成局部 softmax 的 m/l 累积
 * @details 采用 Online Softmax 算法（两遍法），避免全尺寸 score 矩阵的物化。
 *          第一遍对 score 矩阵的一个子块 [qChunkStart:qChunkStart+qChunkSize, kChunkStart:kChunkStart+kChunkSize]
 *          逐行更新 m（逐行最大值）和 l（exp 和）：
 *          - 从 softmaxTmpGm（FP32）读入 score_partial 子块
 *          - m_new = max(m, row_max(score_partial))
 *          - l_new = l × exp(m - m_new) + row_sum(exp(score_partial - m_new))
 *          - 将原始 score_partial 写回 softmaxTmpGm（用于第二遍归一化）
 *          - m = m_new, l = l_new
 *          最终 mFinal 和 lFinal 通过引用返回，跨所有 K-Chunk 累积。
 * @tparam BSAT 算子模板类型
 * @param qChunkStart Q 维度 chunk 起始偏移
 * @param qChunkSize Q 维度 chunk 大小（block 数）
 * @param kChunkStart K 维度 chunk 起始偏移
 * @param kChunkSize K 维度 chunk 大小（block 数）
 * @param[out] mFinal 跨所有 K-Chunk 累积后的逐行最大值（FP32）
 * @param[out] lFinal 跨所有 K-Chunk 累积后的逐行 exp 和（FP32）
 */
template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::OnlineSoftmaxFirstPassChunk(
    uint32_t qChunkStart, uint32_t qChunkSize, uint32_t kChunkStart, uint32_t kChunkSize)
{
    // 输入: score[qChunkStart:qChunkStart+qChunkSize, kChunkStart:kChunkStart+kChunkSize]
    //       存储在 ScoreFp32Gm (FP32)
    // 对每个 qBlock 行维护 m(标量) 和 l(标量):
    //   搬入 score_partial (qChunkSize × kChunkSize 个 FP32)
    //   m_new = max(m, max(score_partial))
    //   l_new = l * exp(m - m_new) + Σexp(score_partial - m_new)
    //   将原始score_partial写回 ScoreFp32Gm
    //   m = m_new, l = l_new
    // 输出: mFinal, lFinal 通过引用返回（跨kChunk累积）
    
    // 更新max
    // M = ReduceMax(score_partial)
    // m_new = max(old_m, M) # 复用old_m
    // 获取vector 实际执行计算的start row 和 excute row
    // FIX: remove duplicate type specifier on actualQExcuteRow
    uint32_t actualQRowStart = 0, actualQExcuteRow = 0;
    if (vecSubBlockIdx == 0) {
        actualQExcuteRow = qChunkSize / 2 + qChunkSize % 2;
        actualQRowStart = qChunkStart;
    } else {
        actualQRowStart = qChunkStart + qChunkSize / 2 + qChunkSize % 2;
        actualQExcuteRow = qChunkSize / 2;
    }

    // qChunkSize 为 1 的场景
    if (actualQExcuteRow == 0) {
        return;
    }

    // printf("第一轮=======================\n");
    // 32字节对齐
    uint32_t kChunkSizeAlign = (kChunkSize + FLOAT_DATA_BLOCK_NUM - 1 ) / FLOAT_DATA_BLOCK_NUM * FLOAT_DATA_BLOCK_NUM;

    // SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    // WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);

    SetFlag<HardEvent::MTE3_MTE2>(eventSft);
    WaitFlag<HardEvent::MTE3_MTE2>(eventSft);

    SetFlag<HardEvent::V_MTE2>(eventSft);
    WaitFlag<HardEvent::V_MTE2>(eventSft);
    
    // score 矩阵workspace存储还需再想想
    DataCopyPad(scoreUb,
        scoreFp32GmTensor[actualQRowStart * constInfo.yBlocks + kChunkStart],
        {static_cast<uint16_t>(actualQExcuteRow),
            static_cast<uint32_t>(kChunkSize * sizeof(float)),
            static_cast<uint32_t>((constInfo.yBlocks - kChunkSize) * sizeof(float)),
            0,
            0},
        {true, 0, static_cast<uint8_t>(kChunkSizeAlign - kChunkSize), SOFTMAX_NEG_INF});

    // SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    // WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    SetFlag<HardEvent::MTE2_V>(eventSft);
    WaitFlag<HardEvent::MTE2_V>(eventSft);


    // printf("vecSubBlockIdx :%d\n", vecSubBlockIdx);
    // printf("actualQRowStart :%d, actualQExcuteRow:%d, kChunkSizeAlign:%d\n", actualQRowStart, actualQExcuteRow, kChunkSizeAlign);

    // AscendC::PipeBarrier<PIPE_ALL>();
    // AscendC::PRINTF("scoreUb\n");
    // AscendC::DumpTensor(scoreUb, 1, 8);
    // AscendC::PipeBarrier<PIPE_ALL>();

    Muls(scoreUb, scoreUb, (float)(constInfo.scaleValue), actualQExcuteRow * kChunkSizeAlign);
    AscendC::PipeBarrier<PIPE_V>();

    // AscendC::PipeBarrier<PIPE_ALL>();
    // AscendC::PRINTF("scoreUb mul\n");
    // AscendC::DumpTensor(scoreUb, 2, actualQExcuteRow * kChunkSizeAlign);
    // AscendC::PipeBarrier<PIPE_ALL>();

    // 更新new max
    // FIX: use const (not constexpr) for isSrcInnerPad since it depends on runtime kChunkSizeAlign
    uint32_t maxShape[] = { actualQExcuteRow, kChunkSize };
    bool isSrcInnerPad = (kChunkSize % FLOAT_DATA_BLOCK_NUM == 0);

    // uint32_t maxValue = 0;
    // uint32_t minValue = 0;
    // bool isReuseSource = false;
    // AscendC::GetReduceMaxMaxMinTmpSize(maxShape, ge::DataType::DT_FLOAT, AscendC::ReducePattern::AR, isSrcInnerPad, isReuseSource, maxValue, minValue);
    // printf("reduce max\n");
    // printf("maxValue: %d, minValue: %d\n", maxValue, minValue);
    // FIX: Remove GetReduceMaxMaxMinTmpSize - it's a HOST-side tiling API, not usable in __aicore__ code
    // The sharedTmpBuffer (reduceSharedTemp) is already allocated at 24KB in InitBuffers

    // FIX: Use correct CANN 9.1 high-level ReduceMax API: 5-param version with sharedTmpBuffer
    // ReduceMax<T, Pattern::Reduce::AR, isReuseSource>(dst, src, sharedTmpBuffer, srcShape, srcInnerPad)
    // AR pattern: reduce along first axis (row-wise max of 2D matrix)
    AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR, false>(tmpMax, scoreUb, reduceSharedTemp, maxShape, isSrcInnerPad);
    AscendC::PipeBarrier<PIPE_V>();

    // AscendC::PipeBarrier<PIPE_ALL>();
    // AscendC::PRINTF("tmpMax\n");
    // AscendC::DumpTensor(tmpMax, 3, actualQExcuteRow);
    // AscendC::PipeBarrier<PIPE_ALL>();

    // AscendC::PipeBarrier<PIPE_V>();

    // AscendC::PipeBarrier<PIPE_ALL>();
    // AscendC::PRINTF("globalMax\n");
    // AscendC::DumpTensor(globalMax, 4, actualQExcuteRow);
    // AscendC::PipeBarrier<PIPE_ALL>();

    // 更新global sum = global_sum * exp(global_max - newMax)  + exp(score - newMax).sum(dim=-1)

    // step 1: 求t3 = global_sum * exp(global_max - newMax)
    // 即：newMax = max(global_max, newMax)
    // t1 = global_max - newMax
    // t2 = torch.exp(t1)
    // t3 = global_sum * t2

    if (kChunkStart != 0) {
        AscendC::Max(tmpMax, tmpMax, globalMax, actualQExcuteRow);
    } else {
        // 第一次k循环不用max
        // todo 这个算vector??
        // FIX: AscendC::Copy is not a valid API, use DataCopy for local tensor element copy
        // AscendC::DataCopy(globalMax, tmpMax, actualQExcuteRow);
        // AscendC::DataCopy(globalMax, tmpMax, (actualQExcuteRow + 8 - 1) / 8 * 8);
        AscendC::Duplicate(globalMax, float(SOFTMAX_NEG_INF), actualQExcuteRow);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(globalSum, float(0.0f), actualQExcuteRow);
    }
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Sub(diffMax, globalMax, tmpMax, actualQExcuteRow);
    AscendC::PipeBarrier<PIPE_V>();

    // AscendC::PipeBarrier<PIPE_ALL>();
    // AscendC::PRINTF("globalMax - tmpMax -即 diff = m - m_new\n");
    // AscendC::DumpTensor(diffMax, 5, actualQExcuteRow);
    // AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::Exp(diffMax, diffMax, actualQExcuteRow);
    AscendC::PipeBarrier<PIPE_V>();

    // AscendC::PipeBarrier<PIPE_ALL>();
    // AscendC::PRINTF("exp(diff)\n");
    // AscendC::DumpTensor(diffMax, 6, actualQExcuteRow);
    // AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::Mul(globalSum, globalSum, diffMax, actualQExcuteRow);
    AscendC::PipeBarrier<PIPE_V>();

    // AscendC::PipeBarrier<PIPE_ALL>();
    // AscendC::PRINTF(" global_sum * t2\n");
    // AscendC::DumpTensor(globalSum, 3, actualQExcuteRow);
    // AscendC::PipeBarrier<PIPE_ALL>();

    // step 2: 求t3 = global_sum * exp(score - newMax)
    // 即 t4 = score - tmpMax.unsqueeze(1)
    // t5 = torch.exp(t4)
    // t6 = t5.sum(dim=1)
    // global_sum = t3 + t6
    uint8_t repeatimes = BSACeilDiv(actualQExcuteRow, BROC_BASE_NUM);
    Brcb(globalMaxBroc, tmpMax, repeatimes, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();

    // AscendC::PipeBarrier<PIPE_ALL>();
    // AscendC::PRINTF("globalMaxBroc - max\n");
    // AscendC::DumpTensor(globalMaxBroc, 3, actualQExcuteRow);
    // AscendC::PipeBarrier<PIPE_ALL>();

    // FIX: add missing semicolon
    SubBroadcast(scoreUb, scoreUb, globalMaxBroc, actualQExcuteRow, kChunkSizeAlign);
    AscendC::PipeBarrier<PIPE_V>();

    // AscendC::PipeBarrier<PIPE_ALL>();
    // AscendC::PRINTF("scoreUb - max\n");
    // AscendC::DumpTensor(scoreUb, 3, actualQExcuteRow * kChunkSizeAlign);
    // AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::Exp(scoreUb, scoreUb, actualQExcuteRow * kChunkSizeAlign);
    AscendC::PipeBarrier<PIPE_V>();

    // AscendC::PipeBarrier<PIPE_ALL>();
    // AscendC::PRINTF("exp(scoreUb - max)\n");
    // AscendC::DumpTensor(scoreUb, 3, actualQExcuteRow * kChunkSizeAlign);
    // AscendC::PipeBarrier<PIPE_ALL>();

    // FIX: use different variable name for ReduceMean shape to avoid confusion
    uint32_t sumShape[] = { actualQExcuteRow, kChunkSize};
    AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR, true>(tmpSum, scoreUb, reduceSharedTemp, sumShape, isSrcInnerPad);
    AscendC::PipeBarrier<PIPE_V>();

    // AscendC::PipeBarrier<PIPE_ALL>();
    // AscendC::PRINTF("tmpSum\n");
    // AscendC::DumpTensor(tmpSum, 3, actualQExcuteRow);
    // AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::Add(globalSum, tmpSum, globalSum, actualQExcuteRow);
    AscendC::PipeBarrier<PIPE_V>();

    // AscendC::PipeBarrier<PIPE_ALL>();
    // AscendC::PRINTF("globalSum\n");
    // AscendC::DumpTensor(globalSum, 3, actualQExcuteRow);
    // AscendC::PipeBarrier<PIPE_ALL>();

    // 最后一次k循环第二轮循环存储sumbroc
    if (kChunkStart + kChunkSize >= constInfo.yBlocks) {
        Brcb(globalSumBroc, globalSum, repeatimes, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
    }

    // 更新globalMax
    AscendC::Max(globalMax, tmpMax, globalMax, actualQExcuteRow);
    // AscendC::DataCopy(globalMax, tmpMax, (actualQExcuteRow + 8 - 1) / 8 * 8);
    // AscendC::PipeBarrier<PIPE_ALL>();
    // AscendC::PRINTF("globalMax\n");
    // AscendC::DumpTensor(globalMax, 3, actualQExcuteRow);
    // AscendC::PipeBarrier<PIPE_ALL>();

    // SetFlag<HardEvent::MTE3>(eventSft);
    // WaitFlag<HardEvent::MTE2_V>(eventSft);
}

/**
 * @brief Online Softmax 第二遍归一化 + 类型转换（FP32→FP16）
 * @details 在第一遍完成 m/l 累积后，对指定 qBlockIdx 行的所有 yBlocks 个 score 进行归一化：
 *          - 逐 chunk 从 ScoreFp32Gm 读取 FP32 score
 *          - attn_score = exp(score - mFinal) / lFinal（在线 softmax 归一化公式）
 *          - Cast FP32 → FP16
 *          - 写入 softmaxOutGm[qBlockIdx × yBlocks:]（FP16），供后续 Radix TopK 使用
 * @tparam BSAT 算子模板类型
 * @param qBlockIdx 当前处理的 Query Block 行索引
 * @param mFinal 跨所有 K-Chunk 累积后的逐行最大值
 * @param lFinal 跨所有 K-Chunk 累积后的逐行 exp 和
 */
template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::SoftmaxSecondPassAndCast(
    uint32_t qChunkStart, uint32_t qChunkSize, uint32_t kChunkStart, uint32_t kChunkSize, uint32_t batchIdx, uint32_t headIdx)
{
    // 对 qBlockIdx 行的所有 Yblocks 个 score:
    //   逐chunk读 ScoreFp32Gm 中的 FP32 score
    //   attn_score = exp(score - mFinal) / lFinal
    //   Cast FP32 → FP16
    //   写入 softmaxOutGm[qBlockIdx * yBlocks :] (FP16)
    // FIX: remove duplicate type specifier on actualQExcuteRow
    uint32_t actualQRowStart = 0, actualQExcuteRow = 0;
    if (vecSubBlockIdx == 0) {
        actualQExcuteRow = qChunkSize / 2 + qChunkSize % 2;
        actualQRowStart = qChunkStart;
    } else {
        actualQRowStart = qChunkStart + qChunkSize / 2 + qChunkSize % 2;
        actualQExcuteRow = qChunkSize / 2;
    }

    // qChunkSize 为 1 的场景
    if (actualQExcuteRow == 0) {
        return;
    }

    // printf("第二轮=======================\n");
    // 32字节对齐 (FP32)
    uint32_t kChunkSizeAlign32 = (kChunkSize + FLOAT_DATA_BLOCK_NUM - 1 ) / FLOAT_DATA_BLOCK_NUM * FLOAT_DATA_BLOCK_NUM;
    // 32字节对齐 (FP16)
    uint32_t kChunkSizeAlign16 = (kChunkSize + FP16_DATA_BLOCK_NUM - 1 ) / FP16_DATA_BLOCK_NUM * FP16_DATA_BLOCK_NUM;
    uint32_t kChunkSizeAlign = kChunkSizeAlign16;

    // SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    // WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);

    SetFlag<HardEvent::MTE3_MTE2>(eventSft);
    WaitFlag<HardEvent::MTE3_MTE2>(eventSft);

    SetFlag<HardEvent::V_MTE2>(eventSft);
    WaitFlag<HardEvent::V_MTE2>(eventSft);

    if (kChunkSizeAlign32 * sizeof(IN_T) % VEC_ALIGN_SIZE == 0) {
        // CAST TO FP16时候保持32字节对齐
        DataCopyPad(scoreUb,
            scoreFp32GmTensor[actualQRowStart * constInfo.yBlocks + kChunkStart],
            {static_cast<uint16_t>(actualQExcuteRow),
                static_cast<uint32_t>(kChunkSize * sizeof(float)),
                static_cast<uint32_t>((constInfo.yBlocks - kChunkSize) * sizeof(float)),
                0,
                0},
            {true, 0, static_cast<uint8_t>(kChunkSizeAlign32 - kChunkSize), SOFTMAX_NEG_INF});
    } else {
        // CAST TO FP16时候未保持32字节对齐，需要加一个1个fp32的block
        DataCopyPad(scoreUb,
            scoreFp32GmTensor[actualQRowStart * constInfo.yBlocks + kChunkStart],
            {static_cast<uint16_t>(actualQExcuteRow),
                static_cast<uint32_t>(kChunkSize * sizeof(float)),
                static_cast<uint32_t>((constInfo.yBlocks - kChunkSize) * sizeof(float)),
                1,
                0},
            {true, 0, static_cast<uint8_t>(kChunkSizeAlign32 - kChunkSize), SOFTMAX_NEG_INF});
    }

    // AscendC::PipeBarrier<PIPE_ALL>();
    // AscendC::PRINTF("globalMax\n");
    // AscendC::DumpTensor(globalMax, 4, actualQExcuteRow);
    // AscendC::PipeBarrier<PIPE_ALL>();

    // SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    // WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    SetFlag<HardEvent::MTE2_V>(eventSft);
    WaitFlag<HardEvent::MTE2_V>(eventSft);

    Muls(scoreUb, scoreUb, (float)(constInfo.scaleValue), actualQExcuteRow * kChunkSizeAlign);
    AscendC::PipeBarrier<PIPE_V>();

    SubBroadcast(scoreUb, scoreUb, globalMaxBroc, actualQExcuteRow, kChunkSizeAlign16);
    AscendC::PipeBarrier<PIPE_V>();

    // AscendC::PipeBarrier<PIPE_ALL>();
    // AscendC::PRINTF("scoreUb subBroc\n");
    // AscendC::DumpTensor(scoreUb, 6, actualQExcuteRow * kChunkSizeAlign);
    // AscendC::PipeBarrier<PIPE_ALL>();

    // AscendC::PipeBarrier<PIPE_ALL>();
    // AscendC::PRINTF("scoreUb globalMaxBroc\n");
    // AscendC::DumpTensor(globalMaxBroc, 6, actualQExcuteRow * kChunkSizeAlign);
    // AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::Exp(scoreUb, scoreUb, actualQExcuteRow * kChunkSizeAlign16);
    AscendC::PipeBarrier<PIPE_V>();

    // AscendC::PipeBarrier<PIPE_ALL>();
    // AscendC::PRINTF("scoreUb Exp\n");
    // AscendC::DumpTensor(scoreUb, 6, actualQExcuteRow * kChunkSizeAlign);
    // AscendC::PipeBarrier<PIPE_ALL>();

    DivBroadcast(scoreUb, scoreUb, globalSumBroc, actualQExcuteRow, kChunkSizeAlign16);
    AscendC::PipeBarrier<PIPE_V>();

    // AscendC::PipeBarrier<PIPE_ALL>();
    // AscendC::PRINTF("scoreUb DivBroadcast\n");
    // AscendC::DumpTensor(scoreUb, 6, actualQExcuteRow * kChunkSizeAlign);
    // AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::Cast(scoreFp16Ub, scoreUb, AscendC::RoundMode::CAST_ROUND, actualQExcuteRow * kChunkSizeAlign16);
    AscendC::PipeBarrier<PIPE_V>();

    // AscendC::PipeBarrier<PIPE_ALL>();
    // AscendC::PRINTF("scoreFp16Ub cast\n");
    // AscendC::DumpTensor(scoreFp16Ub, 6, actualQExcuteRow * kChunkSizeAlign);
    // AscendC::PipeBarrier<PIPE_ALL>();

    // SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    // WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    SetFlag<HardEvent::V_MTE3>(eventSft);
    WaitFlag<HardEvent::V_MTE3>(eventSft);

    DataCopyPad(attnScoreFp16GmTensor[actualQRowStart * constInfo.yBlocks + kChunkStart],
        scoreFp16Ub,
        {static_cast<uint16_t>(actualQExcuteRow),
            static_cast<uint32_t>(kChunkSize * sizeof(half)),
            0,
            static_cast<uint32_t>((constInfo.yBlocks - kChunkSize) * sizeof(half)),
            0});  // FIX: add missing semicolon after closing brace

    uint32_t yBlocks = constInfo.yBlocks;
    uint32_t xBlocks = constInfo.xBlocks;
    uint32_t numHeads = constInfo.numHeads;

    uint64_t perHead = static_cast<uint64_t>(xBlocks) * static_cast<uint64_t>(yBlocks);
    uint64_t perBatch = perHead * numHeads;

    uint32_t sftOutELeOffset = batchIdx * perBatch + headIdx * perHead;
    DataCopyPad(softmaxOutGmTensor[actualQRowStart * constInfo.yBlocks + kChunkStart + sftOutELeOffset],
        scoreFp16Ub,
        {static_cast<uint16_t>(actualQExcuteRow),
            static_cast<uint32_t>(kChunkSize * sizeof(half)),
            0,
            static_cast<uint32_t>((constInfo.yBlocks - kChunkSize) * sizeof(half)),
            0});  // FIX: add missing semicolon after closing brace
}

#endif // BSA_SELECT_BLOCK_MASK_SOFTMAX_H
