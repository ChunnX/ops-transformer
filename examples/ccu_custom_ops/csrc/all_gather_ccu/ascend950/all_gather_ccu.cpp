/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Custom CCU AllGather on Ascend950PR via ACL cross-process VMM IPC (FABRIC handles).
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "../../common/inc/kernel/mc2_moe_context.h"
#include <vector>
#include <cstring>

// The acl_rt.h on the build include path lacks the V2 FABRIC IPC block, but
// libascendcl.so exports these symbols. Forward-declare (extern "C": link by name).
namespace { struct AclFabric { uint8_t data[128]; }; }
static constexpr int32_t ACL_SHARE_FABRIC = 2;  // ACL_MEM_SHARE_HANDLE_TYPE_FABRIC
extern "C" {
aclError aclrtMemExportToShareableHandleV2(aclrtDrvMemHandle handle, uint64_t flags, int32_t shareType, void *shareableHandle);
aclError aclrtMemSetPidToShareableHandleV2(void *shareableHandle, int32_t shareType, int32_t *pid, size_t pidNum);
aclError aclrtMemImportFromShareableHandleV2(void *shareableHandle, int32_t shareType, uint64_t flags, aclrtDrvMemHandle *handle);
}

namespace ascend_ops {
namespace AllGatherCcu {

constexpr int64_t WIN_DATA_OFFSET = 0;
constexpr int64_t ALIGN_32 = 32;
constexpr size_t PHYS_GRANULARITY = 2 * 1024 * 1024;
constexpr int64_t FABRIC_BYTES = 128;

__aicore__ inline int64_t AlignUp(int64_t v, int64_t a) { return (v + a - 1) / a * a; }

struct IpcState {
    bool inited{false};
    size_t allocSize{0};
    void *localVa{nullptr};
    aclrtDrvMemHandle localHandle{nullptr};
    AclFabric fabric{};
    int32_t pid{0};
    std::vector<void *> peerVa;
    std::vector<aclrtDrvMemHandle> peerHandle;
};
static IpcState g_ipc;
static Mc2MoeContext g_ctxHost;

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("ipcAllocExport(Tensor anchor, int size_bytes, int device_id) -> (Tensor, int)");
    m.def("ipcSetpid(Tensor anchor, int[] peer_pids) -> int");
    m.def("ipcImportBuild(Tensor anchor, Tensor handles, int rank, int world_size, int device_id) -> Tensor");
    m.def("AllGatherCcuPut(Tensor x, Tensor mc2_context, int rank, int world_size) -> Tensor");
    m.def("AllGatherCcuGet(Tensor x, Tensor mc2_context, int rank, int world_size) -> Tensor");
}

std::tuple<torch::Tensor, int64_t> ipc_alloc_export(const torch::Tensor &anchor, int64_t size_bytes, int64_t device_id)
{
    const c10::OptionalDeviceGuard guard(anchor.device());
    size_t gran = PHYS_GRANULARITY;
    size_t allocSize = ((size_t)size_bytes + gran - 1) / gran * gran;

    aclrtPhysicalMemProp prop{};
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = ACL_HBM_MEM_HUGE;
    prop.location.id = (int32_t)device_id;
    prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    prop.reserve = 0;

    aclrtDrvMemHandle handle = nullptr;
    TORCH_CHECK(aclrtMallocPhysical(&handle, allocSize, &prop, 0) == ACL_SUCCESS, "aclrtMallocPhysical failed");
    void *va = nullptr;
    TORCH_CHECK(aclrtReserveMemAddress(&va, allocSize, 0, nullptr, 1) == ACL_SUCCESS, "aclrtReserveMemAddress failed");
    TORCH_CHECK(aclrtMapMem(va, allocSize, 0, handle, 0) == ACL_SUCCESS, "aclrtMapMem failed");
    TORCH_CHECK(aclrtMemset(va, allocSize, 0, allocSize) == ACL_SUCCESS, "aclrtMemset failed");

    std::memset(&g_ipc.fabric, 0, sizeof(g_ipc.fabric));
    aclError eret = aclrtMemExportToShareableHandleV2(handle, 0, ACL_SHARE_FABRIC, &g_ipc.fabric);
    TORCH_CHECK(eret == ACL_SUCCESS, "aclrtMemExportToShareableHandleV2 failed ret=", (int64_t)eret);
    int32_t pid = 0;
    TORCH_CHECK(aclrtDeviceGetBareTgid(&pid) == ACL_SUCCESS, "aclrtDeviceGetBareTgid failed");

    g_ipc.allocSize = allocSize;
    g_ipc.localVa = va;
    g_ipc.localHandle = handle;
    g_ipc.pid = pid;

    auto fabricT = at::empty({FABRIC_BYTES}, at::TensorOptions().dtype(at::kByte));
    std::memcpy(fabricT.data_ptr(), g_ipc.fabric.data, FABRIC_BYTES);
    return std::make_tuple(fabricT, (int64_t)pid);
}

int64_t ipc_setpid(const torch::Tensor &anchor, std::vector<int64_t> peer_pids)
{
    const c10::OptionalDeviceGuard guard(anchor.device());
    std::vector<int32_t> pids;
    pids.reserve(peer_pids.size());
    for (auto p : peer_pids) { pids.push_back((int32_t)p); }
    aclError ret = aclrtMemSetPidToShareableHandleV2(&g_ipc.fabric, ACL_SHARE_FABRIC, pids.data(), pids.size());
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMemSetPidToShareableHandleV2 failed ret=", (int64_t)ret);
    return 0;
}

torch::Tensor ipc_import_build(const torch::Tensor &anchor, const torch::Tensor &handles, int64_t rank, int64_t world_size, int64_t device_id)
{
    const c10::OptionalDeviceGuard guard(anchor.device());
    auto handlesCpu = handles.to(at::kCPU).contiguous();
    const uint8_t *hptr = (const uint8_t *)handlesCpu.data_ptr();

    std::memset(&g_ctxHost, 0, sizeof(g_ctxHost));
    g_ctxHost.epRankId = (uint64_t)rank;
    g_ctxHost.kfcContextAddr = 0;
    g_ipc.peerVa.assign(world_size, nullptr);
    g_ipc.peerHandle.assign(world_size, nullptr);

    for (int64_t p = 0; p < world_size; ++p) {
        if (p == rank) {
            g_ipc.peerVa[p] = g_ipc.localVa;
            g_ctxHost.epHcclBuffer[p] = (uint64_t)(uintptr_t)g_ipc.localVa;
            continue;
        }
        int32_t canAccess = 0;
        aclrtDeviceCanAccessPeer(&canAccess, (int32_t)device_id, (int32_t)p);
        if (canAccess != 0) { aclrtDeviceEnablePeerAccess((int32_t)p, 0); }
        AclFabric fab{};
        std::memcpy(fab.data, hptr + p * FABRIC_BYTES, FABRIC_BYTES);
        aclrtDrvMemHandle ph = nullptr;
        aclError impRet = aclrtMemImportFromShareableHandleV2(&fab, ACL_SHARE_FABRIC, 0, &ph);
        TORCH_CHECK(impRet == ACL_SUCCESS, "aclrtMemImportFromShareableHandleV2 failed for peer ", p, " canAccessPeer=", canAccess, " ret=", (int64_t)impRet);
        void *va = nullptr;
        TORCH_CHECK(aclrtReserveMemAddress(&va, g_ipc.allocSize, 0, nullptr, 1) == ACL_SUCCESS, "aclrtReserveMemAddress(peer) failed for peer ", p);
        TORCH_CHECK(aclrtMapMem(va, g_ipc.allocSize, 0, ph, 0) == ACL_SUCCESS, "aclrtMapMem(peer) failed for peer ", p);
        g_ipc.peerHandle[p] = ph;
        g_ipc.peerVa[p] = va;
        g_ctxHost.epHcclBuffer[p] = (uint64_t)(uintptr_t)va;
    }
    g_ipc.inited = true;

    int64_t nInt = (int64_t)(sizeof(Mc2MoeContext) / sizeof(int32_t));
    at::Tensor hostT = at::from_blob((void *)&g_ctxHost, {nInt}, at::kInt);
    at::Tensor ctx = at::empty({nInt}, at::TensorOptions().dtype(at::kInt).device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    ctx.copy_(hostT);
    return ctx;
}

template <typename T>
__global__ __aicore__ void ag_put_kernel(GM_ADDR x, GM_ADDR mc2Context, int64_t rank, int64_t worldSize, int64_t nElems)
{
    if (AscendC::GetBlockIdx() != 0) { return; }
    __gm__ Mc2MoeContext *ctx = (__gm__ Mc2MoeContext *)mc2Context;
    int64_t chunkBytes = nElems * sizeof(T);
    int64_t ubBytes = AlignUp(chunkBytes, ALIGN_32);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQ;
    pipe.InitBuffer(inQ, 1, ubBytes);
    AscendC::GlobalTensor<T> xGm;
    xGm.SetGlobalBuffer((__gm__ T *)x);
    AscendC::DataCopyExtParams cpIn;
    cpIn.blockCount = 1; cpIn.blockLen = (uint32_t)chunkBytes; cpIn.srcStride = 0; cpIn.dstStride = 0;
    AscendC::DataCopyPadExtParams<T> padIn{false, 0, 0, 0};
    AscendC::LocalTensor<T> xUb = inQ.AllocTensor<T>();
    AscendC::DataCopyPad(xUb, xGm, cpIn, padIn);
    inQ.EnQue(xUb);
    xUb = inQ.DeQue<T>();
    for (int64_t p = 0; p < worldSize; ++p) {
        GM_ADDR peerWin = (GM_ADDR)(ctx->epHcclBuffer[p]);
        GM_ADDR dst = peerWin + WIN_DATA_OFFSET + rank * chunkBytes;
        AscendC::GlobalTensor<T> dstGm;
        dstGm.SetGlobalBuffer((__gm__ T *)dst);
        AscendC::DataCopyExtParams cpOut;
        cpOut.blockCount = 1; cpOut.blockLen = (uint32_t)chunkBytes; cpOut.srcStride = 0; cpOut.dstStride = 0;
        AscendC::DataCopyPad(dstGm, xUb, cpOut);
    }
    inQ.FreeTensor(xUb);
}

template <typename T>
__global__ __aicore__ void ag_get_kernel(GM_ADDR mc2Context, GM_ADDR out, int64_t rank, int64_t worldSize, int64_t nElems)
{
    if (AscendC::GetBlockIdx() != 0) { return; }
    __gm__ Mc2MoeContext *ctx = (__gm__ Mc2MoeContext *)mc2Context;
    int64_t chunkBytes = nElems * sizeof(T);
    int64_t totalBytes = chunkBytes * worldSize;
    int64_t ubBytes = AlignUp(totalBytes, ALIGN_32);
    GM_ADDR myWin = (GM_ADDR)(ctx->epHcclBuffer[rank]);
    GM_ADDR src = myWin + WIN_DATA_OFFSET;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> q;
    pipe.InitBuffer(q, 1, ubBytes);
    AscendC::GlobalTensor<T> srcGm;
    srcGm.SetGlobalBuffer((__gm__ T *)src);
    AscendC::GlobalTensor<T> outGm;
    outGm.SetGlobalBuffer((__gm__ T *)out);
    AscendC::DataCopyExtParams cpIn;
    cpIn.blockCount = 1; cpIn.blockLen = (uint32_t)totalBytes; cpIn.srcStride = 0; cpIn.dstStride = 0;
    AscendC::DataCopyPadExtParams<T> padIn{false, 0, 0, 0};
    AscendC::LocalTensor<T> ub = q.AllocTensor<T>();
    AscendC::DataCopyPad(ub, srcGm, cpIn, padIn);
    q.EnQue(ub);
    ub = q.DeQue<T>();
    AscendC::DataCopyExtParams cpOut;
    cpOut.blockCount = 1; cpOut.blockLen = (uint32_t)totalBytes; cpOut.srcStride = 0; cpOut.dstStride = 0;
    AscendC::DataCopyPad(outGm, ub, cpOut);
    q.FreeTensor(ub);
}

torch::Tensor ag_put_npu(const torch::Tensor &x, const torch::Tensor &mc2_context, int64_t rank, int64_t world_size)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    int64_t n = x.numel();
    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto ctx_ptr = (GM_ADDR)mc2_context.data_ptr();
    auto acl_call = [=]() -> int {
        AT_DISPATCH_SWITCH(x.scalar_type(), "ag_put_npu",
            AT_DISPATCH_CASE(torch::kFloat32, [&] { ag_put_kernel<float><<<1, nullptr, stream>>>(x_ptr, ctx_ptr, rank, world_size, n); })
            AT_DISPATCH_CASE(torch::kFloat16, [&] { ag_put_kernel<half><<<1, nullptr, stream>>>(x_ptr, ctx_ptr, rank, world_size, n); })
            AT_DISPATCH_CASE(torch::kBFloat16, [&] { ag_put_kernel<bfloat16_t><<<1, nullptr, stream>>>(x_ptr, ctx_ptr, rank, world_size, n); })
        );
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("AllGatherCcuPut", acl_call);
    return x;
}

torch::Tensor ag_get_npu(const torch::Tensor &x, const torch::Tensor &mc2_context, int64_t rank, int64_t world_size)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    int64_t n = x.numel();
    auto out = torch::empty({world_size, n}, x.options());
    auto ctx_ptr = (GM_ADDR)mc2_context.data_ptr();
    auto out_ptr = (GM_ADDR)out.data_ptr();
    auto acl_call = [=]() -> int {
        AT_DISPATCH_SWITCH(x.scalar_type(), "ag_get_npu",
            AT_DISPATCH_CASE(torch::kFloat32, [&] { ag_get_kernel<float><<<1, nullptr, stream>>>(ctx_ptr, out_ptr, rank, world_size, n); })
            AT_DISPATCH_CASE(torch::kFloat16, [&] { ag_get_kernel<half><<<1, nullptr, stream>>>(ctx_ptr, out_ptr, rank, world_size, n); })
            AT_DISPATCH_CASE(torch::kBFloat16, [&] { ag_get_kernel<bfloat16_t><<<1, nullptr, stream>>>(ctx_ptr, out_ptr, rank, world_size, n); })
        );
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("AllGatherCcuGet", acl_call);
    return out;
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("ipcAllocExport", ipc_alloc_export);
    m.impl("ipcSetpid", ipc_setpid);
    m.impl("ipcImportBuild", ipc_import_build);
    m.impl("AllGatherCcuPut", ag_put_npu);
    m.impl("AllGatherCcuGet", ag_get_npu);
}

} // namespace AllGatherCcu
} // namespace ascend_ops
