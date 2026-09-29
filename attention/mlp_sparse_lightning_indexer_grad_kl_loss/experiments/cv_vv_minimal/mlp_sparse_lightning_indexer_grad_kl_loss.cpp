/* Diagnostic entry retaining the existing ABI, installed in a separate OPP only. */
#include "kernel_operator.h"
#include "cann_mix_wrapper_shim.h"
#include "cv_vv_minimal.h"

extern "C" __global__ __aicore__ void mlp_sparse_lightning_indexer_grad_kl_loss(
    GM_ADDR query, GM_ADDR key, GM_ADDR queryIndex, GM_ADDR keyIndex,
    GM_ADDR weight, GM_ADDR sparseIndices, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
    GM_ADDR queryRope, GM_ADDR keyRope, GM_ADDR mask,
    GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKey,
    GM_ADDR dQueryIndex, GM_ADDR dKeyIndex, GM_ADDR dWeight, GM_ADDR loss,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    SlIgCvVvMinimal::Process();
}
