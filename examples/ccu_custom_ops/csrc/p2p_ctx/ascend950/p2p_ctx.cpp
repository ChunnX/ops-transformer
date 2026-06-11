/** Intra-process multi-device P2P context builder (reuses AllGatherCcu put/get kernels). */
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "../../common/inc/kernel/mc2_moe_context.h"
#include <vector>
#include <cstring>

namespace ascend_ops {
namespace P2pCtx {

static Mc2MoeContext g_ctx;

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("p2pEnable(Tensor anchor, int world_size) -> int");
    m.def("buildCtxFromPtrs(Tensor anchor, int[] ptrs, int rank, int world_size) -> Tensor");
}

int64_t p2p_enable(const torch::Tensor &anchor, int64_t world_size)
{
    const c10::OptionalDeviceGuard guard(anchor.device());
    int32_t cur = (int32_t)anchor.device().index();
    for (int64_t p = 0; p < world_size; ++p) {
        if ((int32_t)p == cur) { continue; }
        int32_t can = 0;
        aclrtDeviceCanAccessPeer(&can, cur, (int32_t)p);
        if (can != 0) { aclrtDeviceEnablePeerAccess((int32_t)p, 0); }
    }
    return 0;
}

torch::Tensor build_ctx_from_ptrs(const torch::Tensor &anchor, std::vector<int64_t> ptrs, int64_t rank, int64_t world_size)
{
    const c10::OptionalDeviceGuard guard(anchor.device());
    std::memset(&g_ctx, 0, sizeof(g_ctx));
    g_ctx.epRankId = (uint64_t)rank;
    for (int64_t p = 0; p < world_size; ++p) { g_ctx.epHcclBuffer[p] = (uint64_t)ptrs[p]; }
    int64_t nInt = (int64_t)(sizeof(Mc2MoeContext) / sizeof(int32_t));
    at::Tensor hostT = at::from_blob((void *)&g_ctx, {nInt}, at::kInt);
    at::Tensor ctx = at::empty({nInt}, at::TensorOptions().dtype(at::kInt).device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    ctx.copy_(hostT);
    return ctx;
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("p2pEnable", p2p_enable);
    m.impl("buildCtxFromPtrs", build_ctx_from_ptrs);
}

} // namespace P2pCtx
} // namespace ascend_ops
