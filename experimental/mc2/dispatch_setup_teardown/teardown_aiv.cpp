/*!
 * \file teardown_aiv.cpp
 * \brief dispatch_teardown —— AIV 写法(<<<>>> ascendc kernel)。
 *   只做一件事: 忙等轮询状态区, 见到非 0(>=1) 就往下走, 并写 done 标记。
 *   与 CCU 写法的 dispatch_setup 在两条 stream 上靠状态位握手。
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t DONE_SENTINEL = 7;

// AIV teardown kernel
extern "C" __global__ __aicore__ void dispatch_teardown_kernel(GM_ADDR statusGM, GM_ADDR doneGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (GetBlockIdx() != 0) {
        return;
    }
    TPipe pipe;
    TBuf<> buf;
    pipe.InitBuffer(buf, 32);
    LocalTensor<int32_t> local = buf.Get<int32_t>();

    GlobalTensor<int32_t> status;
    status.SetGlobalBuffer((__gm__ int32_t*)statusGM);

    int32_t flag = 0;
    while (flag == 0) {  // 忙等, 直到 CCU setup 把状态区标志位写成 1(非 0)
        PipeBarrier<PIPE_ALL>();
        DataCopy(local, status, 8);  // GM -> UB, 每轮重读状态区
        PipeBarrier<PIPE_ALL>();
        flag = local.GetValue(0);
    }
    AscendC::PRINTF("[dispatch_teardown] AIV stream: saw status set (!=0), proceeding.\n");

    Duplicate<int32_t>(local, DONE_SENTINEL, 8);
    GlobalTensor<int32_t> done;
    done.SetGlobalBuffer((__gm__ int32_t*)doneGM);
    PipeBarrier<PIPE_ALL>();
    DataCopy(done, local, 8);
    PipeBarrier<PIPE_ALL>();
}

// <<<>>> 调用 wrapper
void dispatch_teardown_demo(uint32_t blockDim, void* stream, uint8_t* statusGM, uint8_t* doneGM)
{
    dispatch_teardown_kernel<<<blockDim, nullptr, stream>>>(statusGM, doneGM);
}
