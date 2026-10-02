// 模块说明：SageAttention2（INT8 QK^T + FP8 PV）Host 侧 Tiling：四段校验、pad 几何、数值派生、三级核数解析与 workspace 布局。
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "log/log.h"
#include "err/ops_err.h"
#include <register/op_impl_registry.h>
#include "securec.h"
#include "tiling/platform/platform_ascendc.h"

// op_kernel 契约头（并行开发提供，字段与派生规则按附录 C.1，禁止在本文件另建）：
// sa2_spec.h 提供 sa2::Sa2TilingData / sa2::DeriveNumericPolicy 与 Gran/PvAccum 枚举；
// sa2_template_key.h 声明模板维（GET_TPL_TILING_KEY 由 CANN codegen 依声明序生成，key 位序见该头注释）。
#include "../op_kernel/sa2_spec.h"
#include "../op_kernel/sa2_template_key.h"
#include "sa2_tiling.h"

// 三级核数解析的第三级走链接期 acl runtime（附录 B B-38：禁 dlsym 字符串解析）；
// 以 __has_include 作编译开关，头不可得时退化为两级解析并按失败处理。
#if defined(__has_include)
#if __has_include(<acl/acl.h>)
#define SA2_HOST_ACL_RUNTIME_QUERY 1
#include <acl/acl.h>
#endif
#endif

namespace optiling {
namespace {

// 输入/输出/attr 位序与 sa2_def.cpp 声明序严格一致（attr 序 = tiling 索引序，禁增删改序；
// aclnn/python 形参序为各自 API 签名——见 op_api/aclnn_sage_attention2.cpp 与
// python csrc EXEC_NPU_CMD，按名映射到本序，两套顺序不得混淆）
constexpr size_t IN_Q = 0;
constexpr size_t IN_K = 1;
constexpr size_t IN_V = 2;
constexpr size_t OUT_O = 0;
constexpr size_t OUT_LSE = 1;
constexpr size_t ATTR_SM_SCALE = 0;
constexpr size_t ATTR_SMOOTH_K = 1;
constexpr size_t ATTR_IS_CAUSAL = 2;
constexpr size_t ATTR_RETURN_LSE = 3;
constexpr size_t ATTR_QK_QUANT_GRAN = 4;
constexpr size_t ATTR_PV_ACCUM_DTYPE = 5;
constexpr size_t ATTR_SMOOTH_V = 6;

// TilingParse（编译期缓存结构体）：仅存 aic/aiv 核数，tiling 期作第二级核数来源
struct Sa2TilingParse {
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
};

// ── 第一段校验：shape（4 维、正维、<2^32-64、k/v 同形、b/d 相同、d≤128、Hq%Hk==0、b·Hk<2^32）
ge::graphStatus CheckShape(gert::TilingContext *context, Sa2Dims &dims)
{
    const auto qShape = context->GetInputShape(IN_Q);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);
    const auto kShape = context->GetInputShape(IN_K);
    OP_CHECK_NULL_WITH_CONTEXT(context, kShape);
    const auto vShape = context->GetInputShape(IN_V);
    OP_CHECK_NULL_WITH_CONTEXT(context, vShape);

    const gert::Shape &q = qShape->GetStorageShape();
    const gert::Shape &k = kShape->GetStorageShape();
    const gert::Shape &v = vShape->GetStorageShape();

    // ND 布局 [B,H,S,D] 固定 4 维
    OP_CHECK_IF(q.GetDimNum() != 4 || k.GetDimNum() != 4 || v.GetDimNum() != 4,
        OP_LOGE(context, "q/k/v must be 4-D, got %zu/%zu/%zu", q.GetDimNum(), k.GetDimNum(), v.GetDimNum()),
        return ge::GRAPH_FAILED);

    // 正维且 < 2^32-64
    const gert::Shape *shapes[3] = {&q, &k, &v};
    for (size_t inputIdx = 0; inputIdx < 3; ++inputIdx) {
        for (size_t dimIdx = 0; dimIdx < 4; ++dimIdx) {
            const int64_t dim = shapes[inputIdx]->GetDim(dimIdx);
            OP_CHECK_IF(dim <= 0,
                OP_LOGE(context, "dim %zu of input %zu must be positive, got %ld", dimIdx, inputIdx,
                    static_cast<long>(dim)),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(static_cast<uint64_t>(dim) >= DIM_UPPER,
                OP_LOGE(context, "dim %zu of input %zu must be < 2^32-64, got %ld", dimIdx, inputIdx,
                    static_cast<long>(dim)),
                return ge::GRAPH_FAILED);
        }
    }

    // k/v 同形（逐维比对，覆盖 b/hk/sk/d 四项）
    for (size_t dimIdx = 0; dimIdx < 4; ++dimIdx) {
        OP_CHECK_IF(k.GetDim(dimIdx) != v.GetDim(dimIdx),
            OP_LOGE(context, "k and v must share identical shape, dim %zu mismatch: %ld/%ld", dimIdx,
                static_cast<long>(k.GetDim(dimIdx)), static_cast<long>(v.GetDim(dimIdx))),
            return ge::GRAPH_FAILED);
    }
    // q 与 k：b 相同、d 相同（S/Hk 允许不同，支持 Sq≠Sk 与 GQA）
    OP_CHECK_IF(q.GetDim(0) != k.GetDim(0),
        OP_LOGE(context, "q and k batch must match: %ld/%ld", static_cast<long>(q.GetDim(0)),
            static_cast<long>(k.GetDim(0))),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(q.GetDim(3) != k.GetDim(3),
        OP_LOGE(context, "q and k head_dim must match: %ld/%ld", static_cast<long>(q.GetDim(3)),
            static_cast<long>(k.GetDim(3))),
        return ge::GRAPH_FAILED);

    dims.b = static_cast<uint32_t>(q.GetDim(0));
    dims.hq = static_cast<uint32_t>(q.GetDim(1));
    dims.hk = static_cast<uint32_t>(k.GetDim(1));
    dims.sq = static_cast<uint32_t>(q.GetDim(2));
    dims.sk = static_cast<uint32_t>(k.GetDim(2));
    dims.d = static_cast<uint32_t>(q.GetDim(3));

    // GQA：Hq % Hk == 0（任务书 §2.2 Note）
    OP_CHECK_IF(dims.hq % dims.hk != 0U,
        OP_LOGE(context, "num_qo_heads(%u) must be divisible by num_kv_heads(%u)", dims.hq, dims.hk),
        return ge::GRAPH_FAILED);
    // b·Hk < 2^32
    const uint64_t bh = static_cast<uint64_t>(dims.b) * dims.hk;
    OP_CHECK_IF(bh >= BH_UPPER,
        OP_LOGE(context, "b*Hk(%llu) must be < 2^32", static_cast<unsigned long long>(bh)),
        return ge::GRAPH_FAILED);
    // head_dim 上限（D>128 报错，任务书 §3）
    OP_CHECK_IF(dims.d > MAX_HEAD_DIM,
        OP_LOGE(context, "head_dim(%u) must be <= %u", dims.d, MAX_HEAD_DIM),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ── 第二段校验：dtype（三输入同 dtype 且仅 FP16/BF16；out 随 q；lse 固定 FP32）
ge::graphStatus CheckDtype(gert::TilingContext *context, ge::DataType &qDtype)
{
    const auto qDesc = context->GetInputDesc(IN_Q);
    OP_CHECK_NULL_WITH_CONTEXT(context, qDesc);
    const auto kDesc = context->GetInputDesc(IN_K);
    OP_CHECK_NULL_WITH_CONTEXT(context, kDesc);
    const auto vDesc = context->GetInputDesc(IN_V);
    OP_CHECK_NULL_WITH_CONTEXT(context, vDesc);
    const auto outDesc = context->GetOutputDesc(OUT_O);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDesc);
    const auto lseDesc = context->GetOutputDesc(OUT_LSE);
    OP_CHECK_NULL_WITH_CONTEXT(context, lseDesc);

    qDtype = qDesc->GetDataType();
    OP_CHECK_IF(qDtype != ge::DT_FLOAT16 && qDtype != ge::DT_BF16,
        OP_LOGE(context, "q dtype must be FP16 or BF16"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(kDesc->GetDataType() != qDtype || vDesc->GetDataType() != qDtype,
        OP_LOGE(context, "q/k/v dtypes must be identical"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(outDesc->GetDataType() != qDtype,
        OP_LOGE(context, "out dtype must equal q dtype"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(lseDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE(context, "lse dtype must be FP32"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ── 第三段校验：attr（gran/accum 枚举域、sm_scale 有限、causal 仅 Sq==Sk）
ge::graphStatus CheckAttrs(gert::TilingContext *context, const Sa2Dims &dims, Sa2AttrValues &attr)
{
    const auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float *smScale = attrs->GetAttrPointer<float>(ATTR_SM_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, smScale);
    const bool *smoothK = attrs->GetAttrPointer<bool>(ATTR_SMOOTH_K);
    OP_CHECK_NULL_WITH_CONTEXT(context, smoothK);
    const bool *isCausal = attrs->GetAttrPointer<bool>(ATTR_IS_CAUSAL);
    OP_CHECK_NULL_WITH_CONTEXT(context, isCausal);
    const bool *returnLse = attrs->GetAttrPointer<bool>(ATTR_RETURN_LSE);
    OP_CHECK_NULL_WITH_CONTEXT(context, returnLse);
    const int64_t *gran = attrs->GetAttrPointer<int64_t>(ATTR_QK_QUANT_GRAN);
    OP_CHECK_NULL_WITH_CONTEXT(context, gran);
    const int64_t *accum = attrs->GetAttrPointer<int64_t>(ATTR_PV_ACCUM_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, accum);
    const bool *smoothV = attrs->GetAttrPointer<bool>(ATTR_SMOOTH_V);
    OP_CHECK_NULL_WITH_CONTEXT(context, smoothV);

    // sm_scale 须有限（NaN/Inf 拒绝；数值合法性由内核 scale 拆分保障）
    OP_CHECK_IF(!std::isfinite(*smScale),
        OP_LOGE(context, "sm_scale must be finite"), return ge::GRAPH_FAILED);
    // gran 枚举域 {0,1}；accum 枚举域 {0,1,2}（int64 传枚举值）
    const bool granValid = *gran == static_cast<int64_t>(sa2::Gran::kPerWarp) ||
                           *gran == static_cast<int64_t>(sa2::Gran::kPerThread);
    OP_CHECK_IF(!granValid,
        OP_LOGE(context, "qk_quant_gran must be kPerWarp(0) or kPerThread(1), got %ld", static_cast<long>(*gran)),
        return ge::GRAPH_FAILED);
    const bool accumValid = *accum == static_cast<int64_t>(sa2::PvAccum::kFp32) ||
                            *accum == static_cast<int64_t>(sa2::PvAccum::kFp32Fp32) ||
                            *accum == static_cast<int64_t>(sa2::PvAccum::kFp32Fp16);
    OP_CHECK_IF(!accumValid,
        OP_LOGE(context, "pv_accum_dtype must be kFp32(0), kFp32Fp32(1) or kFp32Fp16(2), got %ld",
            static_cast<long>(*accum)),
        return ge::GRAPH_FAILED);
    // 因果 mask 仅 Sq==Sk 合法（任务书 §2.2）
    OP_CHECK_IF(*isCausal && dims.sq != dims.sk,
        OP_LOGE(context, "is_causal requires Sq(%u) == Sk(%u)", dims.sq, dims.sk),
        return ge::GRAPH_FAILED);

    attr.smScale = *smScale;
    attr.smoothK = *smoothK;
    attr.isCausal = *isCausal;
    attr.returnLse = *returnLse;
    attr.gran = static_cast<uint32_t>(*gran);
    attr.accum = static_cast<uint32_t>(*accum);
    // smooth_v 仅 fp32 档生效（DESIGN_SPEC §4.8；warning 行为由 python/aclnn 层承担，此处下发生效值）
    attr.smoothV = *smoothV && (*accum == static_cast<int64_t>(sa2::PvAccum::kFp32));
    return ge::GRAPH_SUCCESS;
}

// 第二级核数来源：compile info（TilingPrepare 阶段缓存的 platform 核数）
void ResolveCoresFromCompileInfo(gert::TilingContext *context, uint32_t &aicNum, uint32_t &aivNum)
{
    auto *compileInfo = context->GetCompiledInfo<Sa2TilingParse>();
    if (compileInfo == nullptr) {
        return;
    }
    if (aicNum == 0U) {
        aicNum = compileInfo->aicNum;
    }
    if (aivNum == 0U) {
        aivNum = compileInfo->aivNum;
    }
}

#if defined(SA2_HOST_ACL_RUNTIME_QUERY)
// 第三级核数来源：链接期 acl runtime 直查（aclrtGetDevice + aclrtGetDeviceInfo，无 dlsym 字符串解析）
void ResolveCoresFromAclRuntime(uint32_t &aicNum, uint32_t &aivNum)
{
    int32_t deviceId = -1;
    if (aclrtGetDevice(&deviceId) != ACL_SUCCESS || deviceId < 0) {
        return;
    }
    int64_t coreNum = 0;
    if (aicNum == 0U && aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_CUBE_CORE_NUM, &coreNum) == ACL_SUCCESS) {
        aicNum = coreNum > 0 ? static_cast<uint32_t>(coreNum) : 0U;
    }
    if (aivNum == 0U && aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &coreNum) == ACL_SUCCESS) {
        aivNum = coreNum > 0 ? static_cast<uint32_t>(coreNum) : 0U;
    }
}
#endif

// ── 第四段校验：资源（三级核数解析 + MIX 1:2 比例校验；失败即整单失败）
ge::graphStatus CheckResource(gert::TilingContext *context, uint32_t &aicNum, uint32_t &aivNum)
{
    // 一级：platform info（调用方已查得则跳过）；二级：compile info；三级：链接期 acl runtime
    if (aicNum == 0U || aivNum == 0U) {
        ResolveCoresFromCompileInfo(context, aicNum, aivNum);
    }
#if defined(SA2_HOST_ACL_RUNTIME_QUERY)
    if (aicNum == 0U || aivNum == 0U) {
        ResolveCoresFromAclRuntime(aicNum, aivNum);
    }
#endif
    OP_CHECK_IF(aicNum == 0U || aivNum == 0U,
        OP_LOGE(context, "failed to resolve aic/aiv core numbers"), return ge::GRAPH_FAILED);
    // MIX 1 AIC : 2 AIV：AIV 不足 2·AIC 时全栅格驻留与双 AIV 流模型不成立
    OP_CHECK_IF(aivNum < 2U * aicNum,
        OP_LOGE(context, "aiv core num(%u) must be >= 2 * aic core num(%u)", aivNum, aicNum),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// workspace 六分区（附录 C.1 顺序与公式，512B 对齐）+ 系统 workspace，全链 u64 溢出防护
ge::graphStatus CalcWorkspace(gert::TilingContext *context, sa2::Sa2TilingData &td, uint64_t sysWs,
    uint64_t &totalWs)
{
    Sa2Dims dims;
    dims.b = td.params.b;
    dims.hq = td.params.hq;
    dims.hk = td.params.hk;
    dims.sq = td.params.sq;
    dims.sk = td.params.sk;
    dims.d = td.params.d;

    Sa2WorkspaceLayout layout{};
    const bool ok = Sa2Optiling::CalcWorkspaceLayout(
        dims, static_cast<uint32_t>(td.params.dp), static_cast<uint32_t>(td.params.kp),
        static_cast<uint32_t>(td.params.segWidth), sysWs, layout);
    OP_CHECK_IF(!ok, OP_LOGE(context, "workspace calculation or alignment check failed"), return ge::GRAPH_FAILED);

    td.kMean = layout.kMeanOffset;
    td.vMean = layout.vMeanOffset;
    td.k8 = layout.k8Offset;
    td.kScale = layout.kScaleOffset;
    td.v8 = layout.v8Offset;
    td.vScale = layout.vScaleOffset;
    totalWs = layout.totalWs;
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus SageAttention2TilingFunc(gert::TilingContext *context)
{
    // 一级核数来源与系统 workspace 同源于 platform info；缺失时由 CheckResource 走二/三级
    auto platformInfo = context->GetPlatformInfo();
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint64_t sysWs = SYS_WS_FALLBACK;
    if (platformInfo != nullptr) {
        platform_ascendc::PlatformAscendC platform(platformInfo);
        aicNum = platform.GetCoreNumAic();
        aivNum = platform.GetCoreNumAiv();
        sysWs = platform.GetLibApiWorkSpaceSize();
        if (sysWs == 0U) {
            sysWs = SYS_WS_FALLBACK;
        }
    }

    // 四段校验：shape → dtype → attr → 资源；任一失败即返回失败码（错误日志在段内给出）
    Sa2Dims dims;
    OP_CHECK_IF(CheckShape(context, dims) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "shape segment validation failed"), return ge::GRAPH_FAILED);
    ge::DataType qDtype = ge::DT_UNDEFINED;
    OP_CHECK_IF(CheckDtype(context, qDtype) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "dtype segment validation failed"), return ge::GRAPH_FAILED);
    Sa2AttrValues attr;
    OP_CHECK_IF(CheckAttrs(context, dims, attr) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "attr segment validation failed"), return ge::GRAPH_FAILED);

    // 契约结构体（平铺 POD，随附录 C.1 定义于 sa2_spec.h），整块零初始化后填充
    static_assert(std::is_trivially_copyable<sa2::Sa2TilingData>::value,
        "Sa2TilingData must be trivially copyable for raw tiling serialization");
    sa2::Sa2TilingData td{};
    sa2::Sa2Params &p = td.params;

    // 原始维度
    p.b = dims.b;
    p.hq = dims.hq;
    p.hk = dims.hk;
    p.sq = dims.sq;
    p.sk = dims.sk;
    p.d = dims.d;

    // pad 几何与数值派生（契约入口）：dp=PaddedHeadDim(d)、qp 对齐 BR=64、kp 对齐 BC=128
    const auto policy = sa2::DeriveNumericPolicy(
        static_cast<sa2::Gran>(attr.gran),
        static_cast<sa2::PvAccum>(attr.accum),
        dims.sq, dims.sk, dims.d, attr.smScale);
    p.dp = policy.dp;
    p.qp = policy.qp;
    p.kp = policy.kp;
    p.qFactor = policy.qFactor;
    p.kFactor = policy.kFactor;
    p.pDomain = policy.pDomain;
    p.lnPDomain = policy.lnPDomain;
    p.segWidth = policy.segWidth;

    // M2.7.2 动态 UB 水位绝对硬防护（UB & Tiling Hard Guard）
    const Sa2TileConfig tileConfig = Sa2Optiling::DeriveTileConfig(dims, attr.gran, static_cast<uint32_t>(p.dp));
    OP_CHECK_IF(Sa2Optiling::VerifyUbSafetyHardGuard(context, dims, attr.gran, tileConfig) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "UB safety hard guard verification failed"), return ge::GRAPH_FAILED);

    const uint64_t d64 = dims.d;
    uint64_t strideB = 0;
    OP_CHECK_IF(!CheckedMul(dims.hq, dims.sq, strideB) || !CheckedMul(strideB, d64, strideB),
        OP_LOGE(context, "q stride overflows uint64"), return ge::GRAPH_FAILED);
    p.qStride[0] = static_cast<int64_t>(strideB);
    p.qStride[1] = static_cast<int64_t>(dims.sq) * d64;
    p.qStride[2] = d64;

    OP_CHECK_IF(!CheckedMul(dims.hk, dims.sk, strideB) || !CheckedMul(strideB, d64, strideB),
        OP_LOGE(context, "k stride overflows uint64"), return ge::GRAPH_FAILED);
    p.kStride[0] = static_cast<int64_t>(strideB);
    p.kStride[1] = static_cast<int64_t>(dims.sk) * d64;
    p.kStride[2] = d64;

    OP_CHECK_IF(!CheckedMul(dims.hk, dims.sk, strideB) || !CheckedMul(strideB, d64, strideB),
        OP_LOGE(context, "v stride overflows uint64"), return ge::GRAPH_FAILED);
    p.vStride[0] = static_cast<int64_t>(strideB);
    p.vStride[1] = static_cast<int64_t>(dims.sk) * d64;
    p.vStride[2] = d64;

    p.smScale = attr.smScale;
    p.causal = attr.isCausal;
    p.smoothK = attr.smoothK;
    p.smoothV = attr.smoothV;
    p.returnLse = attr.returnLse;
    p.gran = attr.gran;
    p.accum = attr.accum;

    // 资源段：二/三级核数解析 + aiv≥2·aic 校验
    OP_CHECK_IF(CheckResource(context, aicNum, aivNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "resource segment validation failed"), return ge::GRAPH_FAILED);
    p.cores = aicNum;

    // workspace：六分区 + 系统 workspace
    uint64_t totalWs = 0;
    OP_CHECK_IF(CalcWorkspace(context, td, sysWs, totalWs) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "workspace calculation failed"), return ge::GRAPH_FAILED);
    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = static_cast<size_t>(totalWs); // LP64 下 u64→size_t 无损

    // blockDim = AIC 数（每块 1 AIC + 2 AIV，GetBlockIdx 枚举见 DESIGN_SPEC §2）
    OP_CHECK_IF(context->SetBlockDim(aicNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "SetBlockDim(%u) failed", aicNum), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->SetScheduleMode(SCHEDULE_MODE_ALL_CORE) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "SetScheduleMode(%u) failed", SCHEDULE_MODE_ALL_CORE), return ge::GRAPH_FAILED);
    // 模板 tiling key：DTYPE(uint32 ge 值) | GRAN(per_thread) | CAUSAL（spec §8 位序由契约头负责）
    OP_CHECK_IF(context->SetTilingKey(GET_TPL_TILING_KEY(static_cast<uint32_t>(qDtype),
                    p.gran == static_cast<int64_t>(sa2::Gran::kPerThread), p.causal)) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "SetTilingKey failed"), return ge::GRAPH_FAILED);

    // 整块拷入原始 tiling buffer（内核侧按同一契约结构体解引用）
    auto rawTiling = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTiling);
    OP_CHECK_IF(rawTiling->GetCapacity() < sizeof(sa2::Sa2TilingData),
        OP_LOGE(context, "raw tiling capacity(%zu) < Sa2TilingData size(%zu)", rawTiling->GetCapacity(),
            sizeof(sa2::Sa2TilingData)),
        return ge::GRAPH_FAILED);
    const errno_t copyRet = memcpy_s(rawTiling->GetData(), rawTiling->GetCapacity(), &td, sizeof(sa2::Sa2TilingData));
    OP_CHECK_IF(copyRet != EOK,
        OP_LOGE(context, "copy Sa2TilingData into raw tiling buffer failed, ret=%d", copyRet),
        return ge::GRAPH_FAILED);
    rawTiling->SetDataSize(sizeof(sa2::Sa2TilingData));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForSageAttention2(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr,
        OP_LOGE("SageAttention2", "tiling parse context is null"), return ge::GRAPH_FAILED);
    auto compileInfo = context->GetCompiledInfo<Sa2TilingParse>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    // 缓存第一级核数，供 tiling 期 platform info 缺失时兜底
    compileInfo->aicNum = platform.GetCoreNumAic();
    compileInfo->aivNum = platform.GetCoreNumAiv();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SageAttention2)
    .Tiling(SageAttention2TilingFunc)
    .TilingParse<Sa2TilingParse>(TilingPrepareForSageAttention2);
} // namespace optiling
