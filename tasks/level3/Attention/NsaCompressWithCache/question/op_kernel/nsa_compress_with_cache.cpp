/**
 * 题目占位：保留入口符号；主体见 answer/0。
 */
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void nsa_compress_with_cache(
    GM_ADDR input, GM_ADDR weight, GM_ADDR slotMapping, GM_ADDR output_cache, GM_ADDR act_seq_len, GM_ADDR block_table,
    GM_ADDR output_cache_ref, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)input;
    (void)tiling;
}
