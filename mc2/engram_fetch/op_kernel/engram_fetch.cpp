/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * EngramFetch kernel implementation (极简骨架版)
 */

#include "kernel_operator.h"
#include "engram_fetch_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_SIZE = 1024;

class EngramFetchKernel {
public:
    __aicore__ EngramFetchKernel()
    {
    }

    __aicore__ void Init(GM_ADDR handle, GM_ADDR indices, GM_ADDR fetched, GM_ADDR tiling)
    {
        handleGM = handle;
        indicesGM = indices;
        fetchedGM = fetched;
        tilingGM = tiling;

        pipe.InitBuffer(outputQueue, 1, BUFFER_SIZE);
    }

    __aicore__ void Process()
    {
        // 骨架实现：直接返回
        // 实际实现需要在后续迭代中添加 RDMA 逻辑
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECOUT, 1> outputQueue;

    GlobalTensor<half> gtFetched;
    GM_ADDR handleGM;
    GM_ADDR indicesGM;
    GM_ADDR fetchedGM;
    GM_ADDR tilingGM;
};

extern "C" __global__ __aicore__ void engram_fetch(GM_ADDR handle, GM_ADDR indices, GM_ADDR fetched, GM_ADDR workspace,
                                                   GM_ADDR tiling)
{
    EngramFetchKernel op;
    op.Init(handle, indices, fetched, tiling);
    op.Process();
}