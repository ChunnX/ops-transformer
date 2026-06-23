/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "kernel_operator.h"
#if (__CCE_AICORE__ == 310)
#include "arch35/sparse_attention_score_prefill_kernel_arch35.h"
#endif

#if (__CCE_AICORE__ == 310)

using namespace SparseAttnPrefill;

template <class InDtype, class SMDtype, class REDtype>
__global__ __aicore__ void SasaPrefillInferIntf(
    GM_ADDR q, GM_ADDR k, GM_ADDR v,
    GM_ADDR selectIdx, GM_ADDR blockTable,
    GM_ADDR kvToQSelectIdx, GM_ADDR kvToQSelectNum,
    GM_ADDR kvToQSlotIdx, GM_ADDR kvBlockValidSize,
    GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
    GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling)
{
    SasaPrefillKernelParams params{
        q, k, v, selectIdx, blockTable,
        kvToQSelectIdx, kvToQSelectNum, kvToQSlotIdx, kvBlockValidSize,
        actualQseqlen, actualKvseqlen,
        o, workspace, tiling};

    using KernelBlockMmadQK = PrefillBlockMmadQK<InDtype, InDtype, float>;
    using KernelEpilogueSoftmax = BlockSoftmaxPrefill<InDtype>;
    using KernelBlockMmadPV = PrefillBlockMmadPV<InDtype, InDtype, REDtype>;
    using KernelEpilogueRescaleO = BlockRescaleOPrefill<InDtype>;

    SasaPrefillKernelArch35<
        KernelBlockMmadQK,
        KernelEpilogueSoftmax,
        KernelBlockMmadPV,
        KernelEpilogueRescaleO> kernel;
    kernel(params);
}

#endif
