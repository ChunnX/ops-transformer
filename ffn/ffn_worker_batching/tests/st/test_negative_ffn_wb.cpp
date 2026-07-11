// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// FfnWorkerBatching A5/arch35 —— 异常校验(负路径)错误码实测
// 目的：对每类非法输入实调 aclnnFfnWorkerBatchingGetWorkspaceSize，观测真实返回码，
//       核对 host 校验是否按 spec 四类错误(null_input/shape_mismatch/dtype_not_supported/
//       attribute_value_out_of_range)准确拦截。期望：除 valid 基线外全部返回非 ACL_SUCCESS。
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn_ffn_worker_batching.h"

static constexpr int64_t CTX_LEN = 1024;
static constexpr int64_t A = 2, BS = 2, K = 2, H = 128, M = 2; // 合法基线：Y=A*BS*K=8

static void WriteU32(std::vector<int8_t> &b, int64_t o, uint32_t v)
{
    std::memcpy(&b[o], &v, 4);
}
static void WriteU64(std::vector<int8_t> &b, int64_t o, uint64_t v)
{
    std::memcpy(&b[o], &v, 8);
}

static aclTensor *MakeTensor(void *dev, const std::vector<int64_t> &shape, aclDataType dt)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = (int64_t)shape.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * shape[i + 1];
    return aclCreateTensor(shape.data(), shape.size(), dt, strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                           dev);
}

// 负路径测试台：持有一份合法基线的设备缓冲，对每类非法输入调 GetWorkspaceSize 观测返回码。
// GetWorkspaceSize 仅做 host tiling，不解引用 GM 地址，故设备缓冲只需分配、无需真实内容。
struct NegHarness {
    void *dTok = nullptr, *dSes = nullptr, *dMic = nullptr, *dExp = nullptr, *dCtx = nullptr;
    void *dY, *dG, *dSid, *dMb, *dTid, *dEo, *dSc, *dAtn;
    aclTensor *ctxI8, *ctxF16, *ctxI32, *ctx2D;
    int pass = 0, total = 0;

    void Setup();
    bool Run(const std::string &tag, const std::string &cat, aclTensor *tctx, int64_t en, std::vector<int64_t> mos,
             int64_t td, int64_t ns, int64_t ln);
    void RunAll();
};

void NegHarness::Setup()
{
    aclrtMalloc(&dTok, 4096, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dSes, 256, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dMic, 256, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dExp, 256, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dCtx, CTX_LEN, ACL_MEM_MALLOC_HUGE_FIRST);

    std::vector<int8_t> ctx(CTX_LEN, 0);
    WriteU32(ctx, 0, (uint32_t)A);
    WriteU32(ctx, 4, (uint32_t)M);
    WriteU32(ctx, 8, (uint32_t)BS);
    WriteU32(ctx, 12, (uint32_t)K);
    WriteU32(ctx, 20, (uint32_t)(H * 2)); // HS = H*2 (fp16)
    WriteU64(ctx, 400, (uint64_t)dTok);
    WriteU64(ctx, 528, (uint64_t)dSes);
    WriteU64(ctx, 544, (uint64_t)dMic);
    WriteU64(ctx, 560, (uint64_t)dExp);
    WriteU32(ctx, 576, (uint32_t)A);
    aclrtMemcpy(dCtx, CTX_LEN, ctx.data(), CTX_LEN, ACL_MEMCPY_HOST_TO_DEVICE);

    // 充裕的合法输出缓冲(负用例在校验阶段即被拦，输出不会被真正写)
    const int64_t Y = 2048;
    aclrtMalloc(&dY, (size_t)Y * H * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dG, (size_t)256 * 2 * 8, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dSid, (size_t)Y * 4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dMb, (size_t)Y * 4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dTid, (size_t)Y * 4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dEo, (size_t)Y * 4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dSc, (size_t)Y * 4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dAtn, 8, ACL_MEM_MALLOC_HUGE_FIRST);

    ctxI8 = MakeTensor(dCtx, {CTX_LEN}, ACL_INT8);
    ctxF16 = MakeTensor(dCtx, {CTX_LEN}, ACL_FLOAT16);
    ctxI32 = MakeTensor(dCtx, {CTX_LEN}, ACL_INT32);
    ctx2D = MakeTensor(dCtx, {2, 512}, ACL_INT8);
}

// 一个用例：(ctx tensor, expertNum, maxOutShape, tokenDtype, needSchedule, layerNum)。cat!="VALID" 即期望被拒。
bool NegHarness::Run(const std::string &tag, const std::string &cat, aclTensor *tctx, int64_t en,
                     std::vector<int64_t> mos, int64_t td, int64_t ns, int64_t ln)
{
    aclDataType yd = td == 2 ? ACL_INT8 : (td == 1 ? ACL_BF16 : ACL_FLOAT16);
    const int64_t Y = 2048;
    aclTensor *y = MakeTensor(dY, {Y, H}, yd), *g = MakeTensor(dG, {en, 2}, ACL_INT64);
    aclTensor *sid = MakeTensor(dSid, {Y}, ACL_INT32), *mb = MakeTensor(dMb, {Y}, ACL_INT32);
    aclTensor *tid = MakeTensor(dTid, {Y}, ACL_INT32), *eo = MakeTensor(dEo, {Y}, ACL_INT32);
    aclTensor *sc = MakeTensor(dSc, {Y}, ACL_FLOAT), *atn = MakeTensor(dAtn, {1}, ACL_INT64);
    aclIntArray *mosArr = aclCreateIntArray(mos.data(), mos.size());
    uint64_t ws = 0;
    aclOpExecutor *exec = nullptr;
    aclnnStatus ret = aclnnFfnWorkerBatchingGetWorkspaceSize(tctx, en, mosArr, td, ns, ln, y, g, sid, mb, tid, eo, sc,
                                                             atn, &ws, &exec);
    bool got = (ret != ACL_SUCCESS);
    bool p = (got == (cat != "VALID"));
    printf("  %-38s [%-22s] ret=%-8d %-12s %s\n", tag.c_str(), cat.c_str(), (int)ret, got ? "(rejected)" : "(SUCCESS)",
           p ? "PASS" : "**FAIL**");
    aclDestroyIntArray(mosArr);
    return p;
}

void NegHarness::RunAll()
{
    printf("======== FfnWorkerBatching 异常校验错误码实测 (ascend950/arch35) ========\n");
    auto T = [&](bool p) {
        ++total;
        if (p)
            ++pass;
    };
    // 0) 合法控制组 —— 证明 harness 本身可用(合法必过)
    T(Run("VALID_baseline_norm_fp16", "VALID", ctxI8, 8, {A, BS, K, H}, 0, 0, 0));
    // 1) null_input
    T(Run("ERR_null_schedule_context", "null_input", nullptr, 8, {A, BS, K, H}, 0, 0, 0));
    // 2) dtype_not_supported
    T(Run("ERR_ctx_dtype_fp16", "dtype_not_supported", ctxF16, 8, {A, BS, K, H}, 0, 0, 0));
    T(Run("ERR_ctx_dtype_int32", "dtype_not_supported", ctxI32, 8, {A, BS, K, H}, 0, 0, 0));
    // 3) shape_mismatch
    T(Run("ERR_ctx_shape_2d", "shape_mismatch", ctx2D, 8, {A, BS, K, H}, 0, 0, 0));
    // 4) attribute_value_out_of_range
    T(Run("ERR_expert_num_zero", "attr_out_of_range", ctxI8, 0, {A, BS, K, H}, 0, 0, 0));
    T(Run("ERR_expert_num_8193", "attr_out_of_range", ctxI8, 8193, {A, BS, K, H}, 0, 0, 0));
    T(Run("ERR_max_out_shape_len3", "attr_out_of_range", ctxI8, 8, {A, BS, K}, 0, 0, 0));
    T(Run("ERR_A_1025", "attr_out_of_range", ctxI8, 8, {1025, BS, K, H}, 0, 0, 0));
    T(Run("ERR_topk_plus1_65", "attr_out_of_range", ctxI8, 8, {A, BS, 65, H}, 0, 0, 0));
    T(Run("ERR_bs_zero", "attr_out_of_range", ctxI8, 8, {A, 0, K, H}, 0, 0, 0));
    T(Run("ERR_hidden_zero", "attr_out_of_range", ctxI8, 8, {A, BS, K, 0}, 0, 0, 0));
    T(Run("ERR_token_dtype_3", "attr_out_of_range", ctxI8, 8, {A, BS, K, H}, 3, 0, 0));
    T(Run("ERR_need_schedule_2", "attr_out_of_range", ctxI8, 8, {A, BS, K, H}, 0, 2, 0));
    T(Run("ERR_layer_num_gt_expert", "attr_out_of_range", ctxI8, 8, {A, BS, K, H}, 0, 0, 9));
    printf("======== 总计: %d  符合预期: %d  不符: %d ========\n", total, pass, total - pass);
}

int main()
{
    if (aclInit(nullptr) != ACL_SUCCESS) {
        printf("[FATAL] aclInit\n");
        return 1;
    }
    if (aclrtSetDevice(0) != ACL_SUCCESS) {
        printf("[FATAL] setDevice\n");
        aclFinalize();
        return 1;
    }
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    NegHarness h;
    h.Setup();
    h.RunAll();

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return (h.pass == h.total) ? 0 : 1;
}
