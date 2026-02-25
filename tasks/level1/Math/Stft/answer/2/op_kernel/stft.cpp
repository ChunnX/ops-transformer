/*!
 * \file stft.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "stft_plan_mul.h"
#include "stft_generalized.h"

using namespace STFTND;
using namespace AscendC;

constexpr MatmulConfig MM_CFG = {true, false, false, 0, 0, 0, false, false, false, false, 0,     0,
                                 0,    0,     0,     0, 0, 0, true,  false, false, false, false, false};

#define STFT_GENERALIZED_IMPL(INPUT_TYPE, BUFFER_NUM, MATMUL_CFG)                                           \
    do {                                                                                                    \
        GET_TILING_DATA_WITH_STRUCT(STFTGeneralizedTilingData, tiling_data_in, tiling);                     \
        STFTGeneralizedTilingData* __restrict tilingData = &tiling_data_in;                                 \
        if (window != nullptr) {                                                                            \
            TPipe planPipe;                                                                                 \
            StftPlanMul<float, 1> planOp;                                                                   \
            planOp.Init(plan, window, userWs, &(tilingData->planTilingData), &planPipe);                    \
            planOp.Process();                                                                               \
            planPipe.Destroy();                                                                             \
        }                                                                                                   \
        TPipe pipeOp;                                                                                       \
        TCubeTiling* __restrict mm0Tiling = &(tilingData->mm0TilingData);                                   \
        TCubeTiling* __restrict mm1Tiling = &(tilingData->mm1TilingData);                                   \
        TCubeTiling* __restrict mm2Tiling = &(tilingData->mm2TilingData);                                   \
        TCubeTiling* __restrict mm3Tiling = &(tilingData->mm3TilingData);                                   \
        STFTGeneralized<INPUT_TYPE, BUFFER_NUM, MATMUL_CFG> op;                                             \
        REGIST_MATMUL_OBJ(                                                                                  \
            &pipeOp, GetSysWorkSpacePtr(), op.mm0, mm0Tiling, op.mm1, mm1Tiling, op.mm2, mm2Tiling, op.mm3, \
            mm3Tiling);                                                                                     \
        op.Init(x, plan, window, y, userWs, tilingData, &pipeOp);                                           \
        op.Process();                                                                                       \
    } while (0)

// mix算子
extern "C" __global__ __aicore__ void stft(
    GM_ADDR x, GM_ADDR plan, GM_ADDR window, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }

    if (TILING_KEY_IS(1)) {
        STFT_GENERALIZED_IMPL(float, 1, MM_CFG);
    } else if (TILING_KEY_IS(3)) {
        STFT_GENERALIZED_IMPL(half, 1, MM_CFG);
    }
    return;
}