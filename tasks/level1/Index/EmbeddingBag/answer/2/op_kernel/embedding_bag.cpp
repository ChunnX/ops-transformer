#include "embedding_bag_fp16.h"

using namespace AscendC;
// kernel function
extern "C" __global__ __aicore__ void embedding_bag(
    GM_ADDR weight, GM_ADDR indices, GM_ADDR offsets, GM_ADDR per_sample_weights, GM_ADDR y, GM_ADDR offset2bag,
    GM_ADDR bag_size, GM_ADDR max_indices, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr) {
        return;
    }
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR gmTensor[8] = {weight, indices, offsets, per_sample_weights, y, offset2bag, bag_size, max_indices};
    if (TILING_KEY_IS(2)) {
        EmbeddingBagFP16<half, int> op(gmTensor, tilingData, pipe);
        op.Process();
    }
    if (TILING_KEY_IS(3)) {
        EmbeddingBagFP16<bfloat16_t, int> op(gmTensor, tilingData, pipe);
        op.Process();
    }
}