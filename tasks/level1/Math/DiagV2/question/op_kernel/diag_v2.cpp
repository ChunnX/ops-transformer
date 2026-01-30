#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void hans_encode(
    GM_ADDR input, GM_ADDR pdf, GM_ADDR pdf_ref, GM_ADDR mantissa, GM_ADDR fixed, GM_ADDR var, GM_ADDR workspace,
    GM_ADDR tiling)
{
}