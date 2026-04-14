#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void advance_step(
    GM_ADDR input_tokens, GM_ADDR sampled_token_ids, GM_ADDR input_positions, GM_ADDR seq_lens, GM_ADDR slot_mapping,
    GM_ADDR block_tables, GM_ADDR spec_token, GM_ADDR accepted_num, GM_ADDR workspace, GM_ADDR tiling)
{
}
