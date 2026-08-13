/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*! \file allto_allv_grouped_mat_mul_shmem.cpp
 *  \brief Ascend950 SHMEM/UDMA AllToAllV GroupedMatMul kernel entry.
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "allto_allv_grouped_mat_mul_shmem.h"

using namespace AscendC;
using namespace matmul;

using ShmemAType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using ShmemBType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
using ShmemCType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bfloat16_t>;
using ShmemBiasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
constexpr MatmulConfig SHMEM_MM_CFG_NO_PRELOAD = GetMDLConfig(false, false, 0, false, false, false, true);

extern "C" __global__ __aicore__ void allto_allv_grouped_mat_mul_shmem(
    GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR gmmyGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (gmmxGM == nullptr || gmmweightGM == nullptr || gmmyGM == nullptr || workspaceGM == nullptr ||
        tilingGM == nullptr) {
        return;
    }

    SetSysWorkspace(workspaceGM);
    GM_ADDR symmetricGM = GetUserWorkspace(workspaceGM);
    if (symmetricGM == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(AlltoAllvGmmShmem::AlltoAllvGroupedMatMulShmemTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    const uint32_t rank = static_cast<uint32_t>(shmem_my_pe());
    const uint32_t peerCount = static_cast<uint32_t>(shmem_n_pes());
    if (peerCount != tilingData.epWorldSize || rank >= peerCount) {
        return;
    }

    TPipe pipe;
    if ASCEND_IS_AIV {
        AlltoAllvGmmShmem::AivShmemCommunicator communicator;
        communicator.Init(gmmxGM, symmetricGM, &tilingData, &pipe, rank, peerCount);
        communicator.Process();
    }
    if ASCEND_IS_AIC {
        MatmulImpl<ShmemAType, ShmemBType, ShmemCType, ShmemBiasType, SHMEM_MM_CFG_NO_PRELOAD> mm;
        mm.SetSubBlockIdx(0);
        mm.Init(&tilingData.matmulTiling, &pipe);
        AlltoAllvGmmShmem::RunAicGroupedMatmul(mm, gmmweightGM, gmmyGM, symmetricGM, &tilingData, peerCount);
    }
}
