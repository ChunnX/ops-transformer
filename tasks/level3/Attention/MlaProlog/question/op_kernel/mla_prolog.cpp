/**
 * 题目占位：保留 mla_prolog 核函数签名，主体在 answer 中补全。
 */
#define MLA_PROLOG_VERSION 1
#include "kernel_operator.h"

template <uint8_t CacheMode, uint8_t Scenario, uint8_t QuantMode, bool EnableDequantOpt,
          bool EnableGroupComputeOpt, uint8_t EmptyTensorMode, uint8_t ActualSeqLenMode,
          uint8_t SplitMMode, uint8_t CvMode>
__global__ __aicore__ void mla_prolog(
    __gm__ uint8_t *tokenX, __gm__ uint8_t *weightDq, __gm__ uint8_t *weightUqQr, __gm__ uint8_t *weightUk,
    __gm__ uint8_t *weightDkvKr, __gm__ uint8_t *rmsnormGammaCq, __gm__ uint8_t *rmsnormGammaCkv,
    __gm__ uint8_t *ropeSin, __gm__ uint8_t *ropeCos, __gm__ uint8_t *cacheIndex, __gm__ uint8_t *kvCache,
    __gm__ uint8_t *krCache, __gm__ uint8_t *dequantScaleX, __gm__ uint8_t *dequantScaleWDq,
    __gm__ uint8_t *dequantScaleWUqQr, __gm__ uint8_t *dequantScaleWDkvKr, __gm__ uint8_t *quantScaleCkv,
    __gm__ uint8_t *quantScaleCkr, __gm__ uint8_t *smoothScalesCq, __gm__ uint8_t *queryOut,
    __gm__ uint8_t *queryRopeOut, __gm__ uint8_t *kvCacheOut, __gm__ uint8_t *krCacheOut, __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    (void)tokenX;
    (void)tiling;
}
