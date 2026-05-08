/**
 * 题目占位：保留入口符号；主体见 answer/0。
 */
#include "kernel_operator.h"

__global__ __aicore__ void
flash_attention_score(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                      __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask,
                      __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
                      __gm__ uint8_t *qStartIdx, __gm__ uint8_t *kvStartIdx, __gm__ uint8_t *deqScaleQ,
                      __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
                      __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *sink, __gm__ uint8_t *softmaxMax,
                      __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut,
                      __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    (void)query;
    (void)tiling;
}
