/**
 * 题目占位：保留入口符号；主体见 answer/0。
 */
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void
nsa_compress_attention(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *attenMask,
                         __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsCmpKv,
                         __gm__ uint8_t *actualSeqLengthsSelKv, __gm__ uint8_t *topkMask,
                         __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *attentionOut,
                         __gm__ uint8_t *topkIndicesOut, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    (void)query;
    (void)tiling;
}
