/**
 * Minimal aclnn call example for CausalConv1dDecode.
 *
 * This file documents the intended two-phase aclnn API call. It follows the
 * style of existing examples in ops-transformer, but leaves tensor allocation
 * helpers to the repository's shared example utilities.
 */

#include "aclnn_causal_conv1d_decode.h"

#include <cstdint>
#include <cstdio>

int main()
{
    const int64_t batch = 2;
    const int64_t seqLen = 4;
    const int64_t dim = 64;
    const int64_t stateLen = 3 + seqLen;
    (void)batch;
    (void)seqLen;
    (void)dim;
    (void)stateLen;

    // Expected tensor shapes:
    // x: [batch, seqLen, dim], BF16
    // weight: [4, dim], BF16
    // convStates: [numCacheSlots, stateLen, dim], BF16, in-place updated
    // cacheIndices: [batch], INT32
    // bias: [dim], BF16, optional
    // numAcceptedTokens: [batch], INT32, optional
    // y: [batch, seqLen, dim], BF16
    //
    // Two-phase call:
    // uint64_t workspaceSize = 0;
    // aclOpExecutor *executor = nullptr;
    // aclnnCausalConv1dDecodeGetWorkspaceSize(
    //     xTensor, weightTensor, convStatesTensor, cacheIndicesTensor,
    //     biasTensor, numAcceptedTokensTensor,
    //     0, -1, 0, yTensor, &workspaceSize, &executor);
    // aclnnCausalConv1dDecode(workspace, workspaceSize, executor, stream);

    std::printf("CausalConv1dDecode example shape: batch=%ld seqLen=%ld dim=%ld stateLen=%ld\n",
                batch, seqLen, dim, stateLen);
    return 0;
}

