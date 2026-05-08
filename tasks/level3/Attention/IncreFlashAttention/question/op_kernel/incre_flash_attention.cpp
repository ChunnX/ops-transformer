#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void incre_flash_attention_FIAS(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR pseShift, GM_ADDR attenMask, GM_ADDR actualSeqLengthsQ,
    GM_ADDR actualSeqLengths, GM_ADDR deqScale1, GM_ADDR quantScale1, GM_ADDR deqScale2, GM_ADDR quantScale2,
    GM_ADDR quantOffset2, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR blocktable, GM_ADDR queryPaddingSize,
    GM_ADDR kvPaddingSize, GM_ADDR keyAntiquantScale, GM_ADDR keyAntiquantOffset, GM_ADDR valueAntiquantScale,
    GM_ADDR valueAntiquantOffset, GM_ADDR keySharedPrefix, GM_ADDR valueSharedPrefix, GM_ADDR actualSharedPrefixLen,
    GM_ADDR queryRope, GM_ADDR keyRope, GM_ADDR keyRopeAntiquantScale, GM_ADDR dequantScaleQuery, GM_ADDR attentionOut,
    GM_ADDR softmaxLse, GM_ADDR workspace, GM_ADDR tiling)
{
}

extern "C" __global__ __aicore__ void incre_flash_attention(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR pseShift, GM_ADDR attenMask, GM_ADDR actualSeqLengths,
    GM_ADDR deqScale1, GM_ADDR quantScale1, GM_ADDR deqScale2, GM_ADDR quantScale2, GM_ADDR quantOffset2,
    GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR blocktable, GM_ADDR kvPaddingSize, GM_ADDR attentionOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
}
