#include "kernel_operator.h"
#include "data_copy_example.h"

extern "C" __global__ __aicore__ void data_copy_example(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling){

    GET_TILING_DATA(tiling_data, tiling);
    if(TILING_KEY_IS(0)) {
        TPipe pipe;
        DataCopyExample<float> op;
        op.Init(x, y, z, workspace, &tiling_data, &pipe);
        op.InitBuffer();
        op.GetLocalTensor();
        op.Process();
        op.ReleaseEventID();
    }
}

