#ifndef LIGHTNING_INDEXER_GRAD_BASE_H
#define LIGHTNING_INDEXER_GRAD_BASE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"

namespace LIG {
using namespace AscendC;
constexpr int32_t THREAD_NUM = 2048;

template<typename T>
__aicore__ inline void DataCopyUB2L1(const LocalTensor<T> dstTensor,
    const LocalTensor<T> srcTensor, const Nd2NzParams nd2nzParam)
{
    DataCopy(dstTensor, srcTensor, nd2nzParam);
}

template<typename T>
__aicore__ inline void DataCopyGM2L1(const LocalTensor<T> dstTensor,
    const GlobalTensor<T> srcTensor, const Nd2NzParams nd2nzParam)
{
    DataCopy(dstTensor, srcTensor, nd2nzParam);
}

template<typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void GatherKeyNZSimt(
    __ubuf__ float4* selecteKeyLocal, __ubuf__ int32_t* sparseIndicesLocal,
    __gm__ float4* volatile keyGlobal, const int32_t tileTopK)
{
    int32_t tileTopKAligned = AlignUp(tileTopK, 32 / sizeof(T));
    int32_t NZOffset = (threadIdx.x / 2) * tileTopKAligned * 2 + threadIdx.x % 2;

    for (int32_t i = threadIdx.y; i < tileTopKAligned; i += blockDim.y) {
        int32_t selecteKeyOffset = NZOffset + i * 2;
        selecteKeyLocal[selecteKeyOffset] =
            i < tileTopK ? keyGlobal[sparseIndicesLocal[i] * blockDim.x + threadIdx.x] : 0;
    }
}

template<typename DTYPE, typename VECTOR_DTYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ScatterKeySimt(
    __gm__ VECTOR_DTYPE* gradKeyGlobal,
    __ubuf__ float2* gradKeyLocal,
    __ubuf__ int32_t* sparseIndicesLocal,
    const int32_t tileTopK)
{
    for (int32_t i = threadIdx.y; i < tileTopK; i += blockDim.y) {
        float2 outValFP32 = gradKeyLocal[i * blockDim.x + threadIdx.x];
        VECTOR_DTYPE outVal = {static_cast<DTYPE>(outValFP32.x),
            static_cast<DTYPE>(outValFP32.y)};
        asc_atomic_add(gradKeyGlobal + sparseIndicesLocal[i] * blockDim.x + threadIdx.x, outVal);
    }
}
}
#endif