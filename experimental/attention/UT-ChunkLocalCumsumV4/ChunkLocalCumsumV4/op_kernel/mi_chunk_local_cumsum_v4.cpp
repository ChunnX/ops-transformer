#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"

using namespace AscendC;
__aicore__ __forceinline__ constexpr int AlignUp8(int a) { return (a + 7) & (~7); }

constexpr int MAX_CHUNK_SIZE = 64;
constexpr int MAX_HEAD_NUM = 32;
constexpr int FLOAT_ALIGN_BYTES = 8;

enum CumsumPattern {
    LargeHead = 0,
    MediumHead,
    SmallHead,
};

struct ChunkInfo {
        int32_t batch;
        int32_t pre_num;
        int32_t chunk_idx_in_batch;
        bool valid;  // 该chunk是否有效
    };

class ChunkLocalCumsum {
   public:
    __aicore__ inline ChunkLocalCumsum(){};
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR cu_seqlens, GM_ADDR output,
                                const MIChunkLocalCumsumV4TilingData &tilingData);

    template <int CHUNKS_PER_CORE>
    __aicore__ inline void Process(const MIChunkLocalCumsumV4TilingData &tilingData);

    template <int N>
    __aicore__ inline void ProcessChunkGroup(const ChunkInfo (&chunk_infos)[N]); 

    __aicore__ inline void InitTiling(const MIChunkLocalCumsumV4TilingData &tilingData);
    __aicore__ inline bool NeedHeadPadding() const;
    __aicore__ inline int32_t GetHeadStride() const;
    __aicore__ inline void CopyChunkToLocal(LocalTensor<float> dstLocal, int32_t dstOffset,
                                            int32_t srcOffset, int32_t rowNum);
    __aicore__ inline void CopyChunkToGlobal(int32_t dstOffset, LocalTensor<float> srcLocal,
                                             int32_t srcOffset, int32_t rowNum);

    GlobalTensor<float> inputGlobal;
    GlobalTensor<float> outputGlobal;
    GlobalTensor<int64_t> cuSeqlenGlobal;

    int32_t m_tokenNum, m_headNum;
    int32_t m_coreNum, m_chunkNum, m_chunkSize;
    uint8_t m_ratio;

   private:
    // UB
    TQue<QuePosition::VECIN, 1> inputQ;
    TQue<QuePosition::VECOUT, 1> outputQ;
};

__aicore__ inline void ChunkLocalCumsum::InitTiling(
    const MIChunkLocalCumsumV4TilingData &tilingData) {
    this->m_coreNum = tilingData.coreNum;
    this->m_tokenNum = tilingData.tokenNum;
    this->m_headNum = tilingData.headNum;
    this->m_chunkNum = tilingData.chunkNum;
    this->m_chunkSize = tilingData.chunkSize;
    this->m_ratio = m_headNum >= 32 ? MAX_CHUNK_SIZE / 2 : MAX_CHUNK_SIZE;
}

__aicore__ inline void ChunkLocalCumsum::Init(GM_ADDR input, GM_ADDR cu_seqlens, GM_ADDR output,
                                              const MIChunkLocalCumsumV4TilingData &tilingData) {
    InitTiling(tilingData);

    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input));
    cuSeqlenGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cu_seqlens));
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));

    // UB
    GetTPipePtr()->InitBuffer(inputQ, 2,
                              MAX_CHUNK_SIZE * MAX_HEAD_NUM * sizeof(float));  // 64*32*4B = 8KB
    GetTPipePtr()->InitBuffer(outputQ, 2,
                              MAX_CHUNK_SIZE * MAX_HEAD_NUM * sizeof(float));  // 64*32*4B = 8KB
}

__aicore__ inline bool ChunkLocalCumsum::NeedHeadPadding() const {
    return (m_headNum % FLOAT_ALIGN_BYTES) != 0;
}

__aicore__ inline int32_t ChunkLocalCumsum::GetHeadStride() const {
    return NeedHeadPadding() ? AlignUp8(m_headNum) : m_headNum;
}

__aicore__ inline void ChunkLocalCumsum::CopyChunkToLocal(LocalTensor<float> dstLocal,
                                                          int32_t dstOffset, int32_t srcOffset,
                                                          int32_t rowNum) {
    if (rowNum <= 0) {
        return;
    }

    if (!NeedHeadPadding()) {
        DataCopy(dstLocal[dstOffset], inputGlobal[srcOffset], rowNum * m_headNum);
        return;
    }

    const int32_t headStride = GetHeadStride();
    const int32_t rightPadding = headStride - m_headNum;
    AscendC::DataCopyPad(
        dstLocal[dstOffset], inputGlobal[srcOffset],
        AscendC::DataCopyExtParams(rowNum, m_headNum * sizeof(float), 0, 0, 0),
        AscendC::DataCopyPadExtParams<float>(true, 0, rightPadding, 0));
}

__aicore__ inline void ChunkLocalCumsum::CopyChunkToGlobal(int32_t dstOffset,
                                                           LocalTensor<float> srcLocal,
                                                           int32_t srcOffset, int32_t rowNum) {
    if (rowNum <= 0) {
        return;
    }

    if (!NeedHeadPadding()) {
        DataCopy(outputGlobal[dstOffset], srcLocal[srcOffset], rowNum * m_headNum);
        return;
    }

    const int32_t headStride = GetHeadStride();
    AscendC::DataCopyPad(outputGlobal[dstOffset], srcLocal[srcOffset],
                         AscendC::DataCopyExtParams(rowNum, m_headNum * sizeof(float), 0, 0, 0));
}

// 处理一组chunk的辅助函数
template <int N>
__aicore__ inline void ChunkLocalCumsum::ProcessChunkGroup(const ChunkInfo (&chunk_infos)[N]) {
    int32_t handle_lens[N];
    int32_t compute_lens[N];
    int32_t io_offsets[N];
    int32_t pre_sums[N];
    bool needs_computation[N];

    int32_t max_compute_len = 0;
    int32_t valid_count = 0;
    int32_t all_compute_num = 0;
    // 第一步：收集所有chunk信息
    int32_t loopCount = m_chunkSize > m_ratio ? m_chunkSize / 2 : m_chunkSize;
    for (int i = 0; i < N; i++) {
        if (!chunk_infos[i].valid) {
            needs_computation[i] = false;
            continue;
        }

        pre_sums[i] = cuSeqlenGlobal.GetValue(chunk_infos[i].batch);
        int32_t cur_seqlen = cuSeqlenGlobal.GetValue(chunk_infos[i].batch + 1) - pre_sums[i];

        io_offsets[i] =
            chunk_infos[i].chunk_idx_in_batch * m_chunkSize * m_headNum + pre_sums[i] * m_headNum;

        handle_lens[i] =
            min(m_chunkSize, cur_seqlen - chunk_infos[i].chunk_idx_in_batch * m_chunkSize);
        compute_lens[i] = (handle_lens[i] + 1) / 2 * 2;

        max_compute_len = max(max_compute_len, compute_lens[i]);
        needs_computation[i] = true;
        valid_count++;
        all_compute_num += handle_lens[i];
    }
    bool isCombin = (all_compute_num == N *m_chunkSize);
    const int32_t headStride = GetHeadStride();

    if (NeedHeadPadding()) {
        auto inputLocal = inputQ.AllocTensor<float>();

        for (int i = 0; i < N; i++) {
            if (!needs_computation[i]) continue;
            CopyChunkToLocal(inputLocal, i * m_chunkSize * headStride, io_offsets[i], handle_lens[i]);
        }

        inputQ.EnQue(inputLocal);
        auto outLocal = outputQ.AllocTensor<float>();
        inputLocal = inputQ.DeQue<float>();
        Adds(outLocal, inputLocal, (float)0, N * m_chunkSize * headStride);

        for (int i = 0; i < N; i++) {
            if (!needs_computation[i]) continue;

            const int32_t chunkBase = i * m_chunkSize * headStride;
            int32_t loop_count = compute_lens[i] > m_ratio ? compute_lens[i] / 2 : compute_lens[i];
            // Hillis-Steele 扫描：stride 需覆盖到最大小于 n 的 2 的幂（n=64 需要 stride=32）
            for (int stride = 1; stride <= loop_count; stride *= 2) {
                Add(outLocal[chunkBase + stride * headStride],
                    outLocal[chunkBase],
                    outLocal[chunkBase + stride * headStride],
                    (compute_lens[i] - stride) * headStride);
            }
        }
        inputQ.FreeTensor(inputLocal);

        outputQ.EnQue(outLocal);
        outLocal = outputQ.DeQue<float>();
        for (int i = 0; i < N; i++) {
            if (!needs_computation[i]) continue;
            CopyChunkToGlobal(io_offsets[i], outLocal, i * m_chunkSize * headStride, handle_lens[i]);
        }
        outputQ.FreeTensor(outLocal);
        return;
    }

    

    if (isCombin){
        auto inputLocal = inputQ.AllocTensor<float>();
        DataCopy(inputLocal, inputGlobal[io_offsets[0]], N*m_chunkSize*m_headNum);
        inputQ.EnQue(inputLocal);
        auto outLocal = outputQ.AllocTensor<float>();
        inputLocal = inputQ.DeQue<float>();
        Adds(outLocal,inputLocal,(float)0,N*m_chunkSize*m_headNum);
        inputQ.FreeTensor(inputLocal);

        for (int i = 0; i < N; ++i) {
            for (int stride = 1; stride < loopCount; stride *= 2) {
                Add(outLocal[i * m_chunkSize * m_headNum + stride * m_headNum],
                    outLocal[i * m_chunkSize * m_headNum],
                    outLocal[i * m_chunkSize * m_headNum + stride * m_headNum],
                    (m_chunkSize - stride) * m_headNum);
            }
        }
        outputQ.EnQue(outLocal);
        outLocal = outputQ.DeQue<float>();
        DataCopy(outputGlobal[io_offsets[0]], outLocal, N*m_chunkSize*m_headNum);
        outputQ.FreeTensor(outLocal);
    }else{


        auto inputLocal = inputQ.AllocTensor<float>();
        for (int i = 0; i < N; i++) {
            if (needs_computation[i]) {
                DataCopy(inputLocal[i*m_chunkSize*m_headNum], inputGlobal[io_offsets[i]], 
                        handle_lens[i] * m_headNum);
            }
        }
        inputQ.EnQue(inputLocal);
        auto outLocal = outputQ.AllocTensor<float>();
        inputLocal = inputQ.DeQue<float>();
        Adds(outLocal, inputLocal, (float)0,  N* m_chunkSize * m_headNum);
        inputQ.FreeTensor(inputLocal);
        for (int i = 0; i < N; i++) {
            if (!needs_computation[i]) continue;
        
            // 执行cumsum计算
            int32_t loop_count = compute_lens[i] > m_ratio ? compute_lens[i] / 2 : compute_lens[i];
            for (int stride = 1; stride <= loop_count; stride *= 2) {
                Add(outLocal[i*m_chunkSize*m_headNum+stride * m_headNum], outLocal[i*m_chunkSize*m_headNum], 
                    outLocal[i*m_chunkSize*m_headNum+stride * m_headNum], 
                    (compute_lens[i] - stride) * m_headNum);
            }
        
        }

        outputQ.EnQue(outLocal);
        outLocal = outputQ.DeQue<float>();
        for (int i = 0; i < N; i++) {
            if (!needs_computation[i]) continue;
            DataCopy(outputGlobal[io_offsets[i]], outLocal[i*m_chunkSize*m_headNum], 
                    handle_lens[i] * m_headNum);
        }

        outputQ.FreeTensor(outLocal);

    }

}

template <int CHUNKS_PER_CORE>
__aicore__ inline void ChunkLocalCumsum::Process(const MIChunkLocalCumsumV4TilingData &tilingData) {
    int32_t block_id = GetBlockIdx();
    int32_t cur_batch = 0;
    int32_t pre_total_chunk_num = 0;
    int32_t process_num = (m_chunkNum + CHUNKS_PER_CORE - 1) / CHUNKS_PER_CORE;

    int32_t cur_total_chunk_num = tilingData.curChunkNumDict[cur_batch];

    

    for (uint32_t process_group = block_id; process_group < process_num;
         process_group += m_coreNum) {
        ChunkInfo chunk_infos[CHUNKS_PER_CORE];
        int32_t valid_chunk_count = 0;

        for (int i = 0; i < CHUNKS_PER_CORE; i++) {
            int32_t global_chunk_idx = process_group * CHUNKS_PER_CORE + i;

            if (global_chunk_idx >= m_chunkNum) {
                chunk_infos[i].valid = false;
                continue;
            }

            int32_t batch = cur_batch;
            int32_t pre_num = pre_total_chunk_num;
            int32_t total_num = cur_total_chunk_num;

            while (global_chunk_idx >= total_num) {
                batch++;
                pre_num = total_num;
                total_num = tilingData.curChunkNumDict[batch];
            }

            chunk_infos[i].batch = batch;
            chunk_infos[i].pre_num = pre_num;
            chunk_infos[i].chunk_idx_in_batch = global_chunk_idx - pre_num;
            chunk_infos[i].valid = true;
            valid_chunk_count++;

            cur_batch = batch;
            pre_total_chunk_num = pre_num;
            cur_total_chunk_num = total_num;
        }

        if (valid_chunk_count > 0) {
            ProcessChunkGroup<CHUNKS_PER_CORE>(chunk_infos);
        }

        // while (process >= cur_total_chunk_num) {
        //     cur_batch++;
        //     pre_total_chunk_num = cur_total_chunk_num;
        //     cur_total_chunk_num = tilingData.curChunkNumDict[cur_batch];
        // }
        // int32_t pre_sum = cuSeqlenGlobal.GetValue(cur_batch);
        // int32_t cur_seqlen = cuSeqlenGlobal.GetValue(cur_batch + 1) - pre_sum;

        // int32_t process_idx = process - pre_total_chunk_num;
        // int32_t chunk_idx = process_idx;

        // int32_t io_offset =
        //     chunk_idx * CHUNKS_PER_CORE * m_chunkSize * m_headNum + pre_sum * m_headNum;
        // int32_t copyLen = CHUNKS_PER_CORE * m_chunkSize * m_headNum;
        // // int32_t handleLen = min(m_chunkSize, cur_seqlen - chunk_idx * m_chunkSize);
        // int32_t handleLen = m_chunkSize;
        // int32_t computeLen = (handleLen + 1) / 2 * 2;
        // int32_t loopCount = computeLen > m_ratio ? computeLen / 2 : computeLen;

        // auto inputLocal = inputQ.AllocTensor<float>();
        // DataCopy(inputLocal, inputGlobal[io_offset], copyLen);
        // inputQ.EnQue(inputLocal);
        // auto outLocal = outputQ.AllocTensor<float>();
        // inputLocal = inputQ.DeQue<float>();

        // for (int i = 0; i < CHUNKS_PER_CORE; ++i) {
        //     Adds(outLocal, inputLocal, (float)0, copyLen);
        // }
        // // Adds(outLocal,inputLocal,(float)0,computeLen*m_headNum);
        // inputQ.FreeTensor(inputLocal);
        // for (int i = 0; i < CHUNKS_PER_CORE; ++i) {
        //     for (int stride = 1; stride < loopCount; stride *= 2) {
        //         Add(outLocal[i * m_chunkSize * m_headNum + stride * m_headNum],
        //             outLocal[i * m_chunkSize * m_headNum],
        //             outLocal[i * m_chunkSize * m_headNum + stride * m_headNum],
        //             (m_chunkSize - stride) * m_headNum);
        //     }
        // }
        // outputQ.EnQue(outLocal);
        // outLocal = outputQ.DeQue<float>();
        // DataCopy(outputGlobal[io_offset], outLocal, copyLen);
        // outputQ.FreeTensor(outLocal);
    }
}

extern "C" __global__ __aicore__ void mi_chunk_local_cumsum_v4(GM_ADDR input, GM_ADDR cu_seqlens,
                                                               GM_ADDR output, GM_ADDR workspace,
                                                               GM_ADDR tiling) {
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    ChunkLocalCumsum kernel;
    kernel.Init(input, cu_seqlens, output, tilingData);

    if (TILING_KEY_IS(1)) {
        kernel.Process<1>(tilingData);
    } else if (TILING_KEY_IS(2)) {
        kernel.Process<2>(tilingData);
    } else if (TILING_KEY_IS(3)) {
        kernel.Process<2>(tilingData);
    }else if (TILING_KEY_IS(4)) {
        kernel.Process<2>(tilingData);
    }
}
