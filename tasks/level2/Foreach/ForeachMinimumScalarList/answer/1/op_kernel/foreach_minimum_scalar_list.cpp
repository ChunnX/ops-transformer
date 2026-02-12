/*!
 * \file foreach_maximum_scalar_list.cpp
 * \brief
 */

#include "kernel_operator.h"

// op kernel building at build_out directory, it's not fully aligned with source code structure
// current op_kernel folder is absent in build_out directory, so the relative path to common has just one layer
#include "foreach_one_scalar_list_binary.h"

using namespace AscendC;
using namespace Common::OpKernel;

template <typename T>
__aicore__ void MinsAdapter(
    const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue, const int32_t& uValue)
{
    Mins(dstLocal, srcLocal, scalarValue, static_cast<uint32_t>(uValue));
}

extern "C" __global__ __aicore__ void foreach_minimum_scalar_list(
    GM_ADDR inputs, GM_ADDR scalar, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    // foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachOneScalarListBinary<half, half, MinsAdapter<half>, 2, 1> op;
        op.Init(inputs, scalar, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachOneScalarListBinary<float, float, MinsAdapter<float>, 2, 1> op;
        op.Init(inputs, scalar, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        ForeachOneScalarListBinary<int, int, MinsAdapter<int>, 2, 1> op;
        op.Init(inputs, scalar, outputs, userWS, &tilingData);
        op.Process();
    }
#if __CCE_AICORE__ >= 220 && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    else if (TILING_KEY_IS(4)) {
        ForeachOneScalarListBinary<bfloat16_t, float, MinsAdapter<float>, 2, 1> op;
        op.Init(inputs, scalar, outputs, userWS, &tilingData);
        op.Process();
    }
#endif
}