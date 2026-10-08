/*
 * compressor_grad op_host — tiling calculation + sequential kernel launches
 *
 * Kernel chain (all inside one operator), R5: transpose kernels eliminated —
 * the gemm consumes/produces natural layouts (dx B = (C,H) weights via
 * load3dv2; dw writes (C,H) direct via A=dn^T transposed L0A load):
 *   K0 cast_zero (AIV)  : x/wkv/wgate -> fp16; dn/d_ape zero-fill (fp16: dn dual HiLo planes)
 *   K1 scatter   (AIV)  : softmax backward + routing + d_ape (fp16: writes dn hi/lo planes)
 *   K2 gemm      (AIC)  : d_x; d_wkv/d_wgate direct (fp16: dual Mmad per dn k-tile)
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <torch/extension.h>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"

#include "aclrtlaunch_compressor_grad_cast.h"
#include "aclrtlaunch_compressor_grad_scatter.h"
#include "aclrtlaunch_compressor_grad_gemm.h"

namespace ascend_kernel {

std::vector<at::Tensor> compressor_grad(
    const at::Tensor &x,
    const at::Tensor &wkv,
    const at::Tensor &wgate,
    const at::Tensor &dCmpKv,
    const at::Tensor &softmaxScore,
    const at::Tensor &kv,
    int64_t cmpRatio,
    int64_t coff)
{
    TORCH_CHECK(x.dim() == 3, "x must be (B, S, H)");
    TORCH_CHECK(wkv.dim() == 2 && wgate.dim() == 2, "wkv/wgate must be 2D");
    TORCH_CHECK(dCmpKv.dim() == 3, "d_cmp_kv must be (B, nblk, D)");
    TORCH_CHECK(softmaxScore.dim() == 4 && kv.dim() == 4, "softmax_score/kv must be 4D");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "x must be fp16/bf16");
    TORCH_CHECK(wkv.scalar_type() == x.scalar_type() && wgate.scalar_type() == x.scalar_type(),
                "wkv/wgate dtype must match x");
    TORCH_CHECK(dCmpKv.scalar_type() == x.scalar_type(), "d_cmp_kv dtype must match x");
    TORCH_CHECK(softmaxScore.scalar_type() == at::kFloat && kv.scalar_type() == at::kFloat,
                "softmax_score/kv must be fp32");
    TORCH_CHECK(x.is_contiguous() && wkv.is_contiguous() && wgate.is_contiguous()
                    && dCmpKv.is_contiguous() && softmaxScore.is_contiguous() && kv.is_contiguous(),
                "all inputs must be contiguous");

    const int64_t B = x.sizes()[0];
    const int64_t S = x.sizes()[1];
    const int64_t H = x.sizes()[2];
    const int64_t cr = cmpRatio;
    const int64_t coffI = coff;
    TORCH_CHECK(coffI == 1 || coffI == 2, "coff must be 1 or 2");
    TORCH_CHECK(cr > 0, "cmp_ratio must be positive");
    // scatter repeat-form vector ops take uint8_t repeatTime: the N=coff*cr
    // sites are chunked to <=255 per call; the crSize_ sites require cr<=255
    TORCH_CHECK(cr <= 255, "cmp_ratio must be <= 255 (scatter vector repeat limit)");
    TORCH_CHECK(wkv.sizes()[0] % coffI == 0, "wkv rows must be divisible by coff");
    TORCH_CHECK(wgate.sizes()[0] == wkv.sizes()[0], "wgate rows must match wkv");
    TORCH_CHECK(wkv.sizes()[1] == H && wgate.sizes()[1] == H,
                "wkv/wgate must be (coff*D, H)");
    const int64_t D = wkv.sizes()[0] / coffI;
    TORCH_CHECK(D % 64 == 0 && D >= 64, "head dim must be multiple of 64");
    const int64_t C = coffI * D;
    const int64_t N = coffI * cr;
    const int64_t nblk = S / cr;          // complete blocks only
    const int64_t nblkIn = dCmpKv.sizes()[1];
    TORCH_CHECK(nblkIn >= nblk, "d_cmp_kv blocks must cover complete blocks");
    TORCH_CHECK(dCmpKv.sizes()[0] == B && dCmpKv.sizes()[2] == D,
                "d_cmp_kv must be (B, nblk, D)");
    TORCH_CHECK(softmaxScore.sizes()[0] == B && softmaxScore.sizes()[1] == nblkIn
                    && softmaxScore.sizes()[2] == N && softmaxScore.sizes()[3] == D,
                "softmax_score must be (B, nblk, coff*cr, D)");
    TORCH_CHECK(kv.sizes()[0] == B && kv.sizes()[1] == nblkIn
                    && kv.sizes()[2] == N && kv.sizes()[3] == D,
                "kv must be (B, nblk, coff*cr, D)");
    const int64_t T = B * S;
    const int64_t T_pad = (T + 127) / 128 * 128;
    TORCH_CHECK(H % 128 == 0, "H must be multiple of 128 (tile alignment)");
    TORCH_CHECK(C % 128 == 0, "C must be multiple of 128");
    TORCH_CHECK(nblk >= 1, "at least one complete block required");

    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t aivCoreNum = platform->GetCoreNumAiv();
    int32_t aicCoreNum = platform->GetCoreNumAic();
    if (aivCoreNum <= 0) {
        aivCoreNum = 48;
    }
    if (aicCoreNum <= 0) {
        aicCoreNum = 24;
    }

    // dtype flag: 0 = fp16, 1 = bf16
    int64_t dtypeFlag = (x.scalar_type() == at::kBFloat16) ? 1 : 0;

    // ---- outputs ----
    at::Tensor dX = at::empty({B, S, H}, x.options());
    at::Tensor dWkv = at::empty({C, H}, wkv.options());
    at::Tensor dWgate = at::empty({C, H}, wgate.options());
    at::Tensor dApe = at::empty({cr, C}, x.options().dtype(at::kFloat));

    // ---- workspace ----
    // fp16 path: dn workspaces carry HiLo dual planes (dn = hi + lo, plane
    // stride T_pad*C); bf16 stays single-plane.
    // R7 (P54): one allocation for all five workspaces + the gemm schedule
    // dummy; every sub-block is a multiple of 128x128 half elements (T_pad,
    // H, C all 128-aligned), so each offset keeps 32KB alignment. Saves five
    // at::empty calls (~30us) of fixed host overhead per op call.
    const int64_t dnPlanes = (dtypeFlag == 0) ? 2 : 1;
    const int64_t xElems = T_pad * H;
    const int64_t wElems = C * H;
    const int64_t dnElems = dnPlanes * T_pad * C;
    const int64_t offX = 0;
    const int64_t offW = offX + xElems;
    const int64_t offG = offW + wElems;
    const int64_t offDnKv = offG + wElems;
    const int64_t offDnSc = offDnKv + dnElems;
    const int64_t offDummy = offDnSc + dnElems;
    at::Tensor wsAll = at::empty({offDummy + 16}, x.options().dtype(at::kHalf));
    uint8_t *wsBase = static_cast<uint8_t *>(wsAll.data_ptr());
    uint8_t *xF16 = wsBase + offX * 2;
    uint8_t *wkvF16 = wsBase + offW * 2;
    uint8_t *wgateF16 = wsBase + offG * 2;
    uint8_t *dnKv = wsBase + offDnKv * 2;
    uint8_t *dnSc = wsBase + offDnSc * 2;
    uint8_t *wsDummy = wsBase + offDummy * 2;

    // debug stage gate: CG_DEBUG_STAGE=1|2|3 limits how many kernels run
    const char *stageEnv = std::getenv("CG_DEBUG_STAGE");
    const int stageLimit = stageEnv ? std::atoi(stageEnv) : 3;

    // ---- K0: cast + zero fill ----
    if (stageLimit >= 1) {
        int64_t totalRows = T_pad + 2 * C;
        int32_t castCores = std::min<int32_t>(aivCoreNum,
                                              static_cast<int32_t>(std::max<int64_t>(1, totalRows)));
        int32_t tLenL = static_cast<int32_t>(T);
        int32_t hL = static_cast<int32_t>(H);
        int32_t cL = static_cast<int32_t>(C);
        int32_t crL = static_cast<int32_t>(cr);
        int32_t tPadL = static_cast<int32_t>(T_pad);
        int32_t coreNumL = castCores;
        uint32_t blockDim = static_cast<uint32_t>(castCores);
        EXEC_KERNEL_CMD(compressor_grad_cast, blockDim,
                        x, wkv, wgate,
                        xF16, wkvF16, wgateF16, dnKv, dnSc, dApe,
                        tLenL, hL, cL, crL, tPadL, coreNumL, dtypeFlag);
    }

    // ---- K1: scatter ----
    if (stageLimit >= 2) {
        int64_t nbTotal = B * nblk;
        int32_t scatterCores = std::min<int32_t>(aivCoreNum, static_cast<int32_t>(nbTotal));
        int32_t blocksPerCore =
            static_cast<int32_t>((nbTotal + scatterCores - 1) / scatterCores);
        // SEG: UB budget (arch2201 TBuf ceiling = TOTAL_VEC_LOCAL_SIZE = 184KB)
        //   bytes(S) = (2*cr + 3*N + 5)*S*4 + (planes-1)*cr*S*4 + S*ioT
        //              + slots*planes*cr*S*2
        //   planes = 2 for fp16 HiLo else 1; slots = 2 for coff==2 else 1; S <= D
        int32_t segLen = 64;
        {
            int64_t planes = (dtypeFlag == 0) ? 2 : 1;
            int64_t slots = (coffI == 2) ? 2 : 1;
            auto segBytes = [&](int64_t s) {
                return (2 * cr + 3 * N + 5) * s * 4 + (planes - 1) * cr * s * 4 + s * 2
                    + slots * planes * cr * s * 2;
            };
            if (segBytes(64) > 184 * 1024LL || D < 64) {
                segLen = 32;
            }
            if (segBytes(segLen) > 184 * 1024LL) {
                segLen = 16;
            }
            // final UB guardrail: reject shapes that overflow even after the
            // 64->32->16 segLen downgrade (arch2201 TBuf ceiling)
            TORCH_CHECK(segBytes(16) <= 184 * 1024LL,
                        "scatter UB budget exceeded even at segLen=16; shape not supported");
        }
        int32_t bL = static_cast<int32_t>(B);
        int32_t sL = static_cast<int32_t>(S);
        int32_t dL = static_cast<int32_t>(D);
        int32_t crL = static_cast<int32_t>(cr);
        int32_t coffL = static_cast<int32_t>(coffI);
        int32_t nblkL = static_cast<int32_t>(nblk);
        int32_t nblkInL = static_cast<int32_t>(nblkIn);
        int32_t nRowsL = static_cast<int32_t>(N);
        int32_t cL = static_cast<int32_t>(C);
        int32_t segL = segLen;
        int32_t bpcL = blocksPerCore;
        int32_t tPadL = static_cast<int32_t>(T_pad);
        uint32_t blockDim = static_cast<uint32_t>(scatterCores);
        EXEC_KERNEL_CMD(compressor_grad_scatter, blockDim,
                        dCmpKv, softmaxScore, kv,
                        dnKv, dnSc, dApe,
                        bL, sL, dL, crL, coffL, nblkL, nblkInL, nRowsL, cL,
                        segL, bpcL, tPadL, dtypeFlag);
    }

    // ---- K2: gemm ----
    if (stageLimit >= 3) {
        int32_t gemmCores = aicCoreNum;
        int32_t gDx = std::max<int32_t>(1, gemmCores / 2);
        // R8: shape-adaptive dx/dw core split, recalibrated post-PX (S3).
        // PX flattened the dx branch into a cross-pair term stream, cutting
        // the per-k-tile dx cost ~33%; what remains per dx unit is a fixed
        // overhead (accumulator rotation, FixOut drain, stream prologue)
        // that does not scale with C. Model: dx unit = 4 per k-tile (C/128)
        // + 2 fixed per unit; fp16 HiLo scales x4/3 (dual A planes). dw
        // unchanged: dw tile ~0.62 when the x workspace stays L2-resident,
        // ~1.16 when it is HBM-streamed. Shift cores dx-ward when dx
        // dominates (>=1.15x, lowered from 1.25 as dx got cheaper), clamp to
        // [half, half+4]: balanced shapes (case5) keep the even split —
        // dw-ward moves measurably regress there. Calibrated by CG_GDX
        // sweeps of the PX kernel: case3 -> g13, case2 -> g14.
        {
            const int64_t nM = (T + 127) / 128;
            const int64_t nH = H / 128;          // H % 128 == 0 (checked above)
            const int64_t nCn = C / 128;         // C % 128 == 0
            const int64_t nPairRow = (nH + 1) / 2;
            const bool fp16 = (dtypeFlag == 0);
            const double dxCost = static_cast<double>(nM * nPairRow)
                * (4.0 * nCn + 2.0) * (fp16 ? 4.0 / 3.0 : 1.0);
            const double dwTiles =
                static_cast<double>(nCn * nH * (T_pad / 128)) * (fp16 ? 5 : 3);
            const bool xResident = (T_pad * H * 2) <= (int64_t)128 * 1024 * 1024;
            const double dwCost = dwTiles * (xResident ? 0.62 : 1.16);
            if (dxCost >= 1.15 * dwCost) {
                int64_t g = static_cast<int64_t>(
                    static_cast<double>(gemmCores) * dxCost / (dxCost + dwCost) + 0.5);
                g = std::min<int64_t>(std::max<int64_t>(g, gemmCores / 2), gemmCores / 2 + 4);
                g = std::min<int64_t>(g, gemmCores);
                g = std::min(g, std::max<int64_t>(1, nM * nPairRow));
                gDx = static_cast<int32_t>(g);
            }
        }
        // shm-tree sweep knob: CG_GDX overrides the dx/dw core split
        const char *gdxEnv = std::getenv("CG_GDX");
        if (gdxEnv != nullptr) {
            gDx = std::max(0, std::min(gemmCores, std::atoi(gdxEnv)));
        }
        // debug: CG_GEMM_MODE=dx -> only d_x branch; dw -> only d_w branch
        const char *gemmMode = std::getenv("CG_GEMM_MODE");
        if (gemmMode != nullptr) {
            if (gemmMode[0] == 'd' && gemmMode[1] == 'x') {
                gDx = gemmCores;
            } else if (gemmMode[0] == 'd' && gemmMode[1] == 'w') {
                gDx = 0;
            }
        }
        int32_t tLenL = static_cast<int32_t>(T);
        int32_t hL = static_cast<int32_t>(H);
        int32_t cL = static_cast<int32_t>(C);
        int32_t tPadL = static_cast<int32_t>(T_pad);
        int32_t gDxL = gDx;
        const char *gemmDbgEnv = std::getenv("CG_GEMM_DBG");
        int32_t gemmDbgL = gemmDbgEnv ? std::atoi(gemmDbgEnv) : 0;
        // MIX_AIC_1_2 schedule requires non-zero workspace;
        // blockDim counts AIC cores (AIV slots are auto-derived x2).
        uint32_t blockDimG = static_cast<uint32_t>(gemmCores);
        EXEC_KERNEL_CMD(compressor_grad_gemm, blockDimG,
                        xF16, wkvF16, wgateF16, dnKv, dnSc,
                        dX, dWkv, dWgate, wsDummy,
                        tLenL, hL, cL, tPadL, gDxL, gemmDbgL, dtypeFlag);
    }

    return {dX, dWkv, dWgate, dApe};
}

}  // namespace ascend_kernel
