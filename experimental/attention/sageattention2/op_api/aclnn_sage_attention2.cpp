// aclnn 两段式实现：SageAttention2（INT8 QK^T + FP8 PV）；NHD 布局经 CreateView 换位为 HND 后下发 AICORE
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

#include "aclnn_sage_attention2.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "op_common/log/log.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {

constexpr int64_t DIM_NUM_3 = 3;
constexpr int64_t DIM_NUM_4 = 4;
constexpr int64_t DIM_B = 0;
constexpr int64_t DIM_H = 1;
constexpr int64_t DIM_S = 2;
constexpr int64_t DIM_D = 3;
// [第一性原理推导：四维张量各维合法上界 2^32 - 64]
// 硬件依据：在昇腾 Atlas 950PR 硬件架构中，DaVinci 协处理器向量与张量 MTE2/MTE3 DMA 地址计算使用 32 位无符号整数寄存器寻址，
// 单次猝发访问与缓存行对齐基准为 64 字节（512-bit cacheline）。若张量单维长度达到或超过 2^32 - 64，
// 在添加硬件行对齐或地址尾部偏移（+64B 余量）时将触发 32 位无符号整数溢出回绕（wrap-around），
// 破坏内存保护边界并引发非法访问。因此严格限制单维上限必须小于 2^32 - 64。
constexpr int64_t MAX_DIM_VALUE = static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) - 64;
constexpr int64_t MAX_HEAD_DIM = 128;
// 与 op_host/sa2_def.cpp 的 attr 声明序一一对应的枚举取值
constexpr int64_t QUANT_GRAN_PER_WARP = 0;
constexpr int64_t QUANT_GRAN_PER_THREAD = 1;
constexpr int64_t ACCUM_FP32 = 0;
constexpr int64_t ACCUM_FP32_FP32 = 1;
constexpr int64_t ACCUM_FP32_FP16 = 2;
const char *const API_NAME = "aclnnSa2Attention2GetWorkspaceSize";

bool IsAscend950Platform()
{
    return GetCurrentPlatformInfo().GetSocLongVersion().rfind("Ascend950", 0) == 0;
}

// 布局字符串解析：仅接受 "HND" / "NHD"（其余返回 nullptr 由调用方报 INVALID）
const char *ParseTensorLayout(const char *tensorLayout)
{
    if (tensorLayout != nullptr && strcmp(tensorLayout, "HND") == 0) {
        return "HND";
    }
    if (tensorLayout != nullptr && strcmp(tensorLayout, "NHD") == 0) {
        return "NHD";
    }
    return nullptr;
}

// QK 量化粒度字符串解析：仅接受 "per_warp" / "per_thread"
bool ParseQkQuantGran(const char *qkQuantGran, int64_t &granCode)
{
    if (qkQuantGran != nullptr && strcmp(qkQuantGran, "per_warp") == 0) {
        granCode = QUANT_GRAN_PER_WARP;
        return true;
    }
    if (qkQuantGran != nullptr && strcmp(qkQuantGran, "per_thread") == 0) {
        granCode = QUANT_GRAN_PER_THREAD;
        return true;
    }
    return false;
}

// PV 两级累加精度字符串解析：仅接受 "fp32" / "fp32+fp32" / "fp32+fp16"
bool ParsePvAccumDtype(const char *pvAccumDtype, int64_t &accumCode)
{
    if (pvAccumDtype != nullptr && strcmp(pvAccumDtype, "fp32") == 0) {
        accumCode = ACCUM_FP32;
        return true;
    }
    if (pvAccumDtype != nullptr && strcmp(pvAccumDtype, "fp32+fp32") == 0) {
        accumCode = ACCUM_FP32_FP32;
        return true;
    }
    if (pvAccumDtype != nullptr && strcmp(pvAccumDtype, "fp32+fp16") == 0) {
        accumCode = ACCUM_FP32_FP16;
        return true;
    }
    return false;
}

aclnnStatus CheckNotNull(const aclTensor *q, const aclTensor *k, const aclTensor *v, aclTensor *out,
                         const char *tensorLayout, const char *qkQuantGran, const char *pvAccumDtype,
                         uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (q == nullptr || k == nullptr || v == nullptr || out == nullptr || tensorLayout == nullptr ||
        qkQuantGran == nullptr || pvAccumDtype == nullptr || workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s: required parameters must not be nullptr.", API_NAME);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckStringParams(const char *tensorLayout, const char *qkQuantGran, const char *pvAccumDtype,
                              int64_t &granCode, int64_t &accumCode)
{
    if (ParseTensorLayout(tensorLayout) == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            API_NAME, "tensorLayout", tensorLayout == nullptr ? "nullptr" : tensorLayout,
            "tensorLayout must be one of [HND, NHD]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!ParseQkQuantGran(qkQuantGran, granCode)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "qkQuantGran", qkQuantGran,
                                              "qkQuantGran must be one of [per_warp, per_thread]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!ParsePvAccumDtype(pvAccumDtype, accumCode)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "pvAccumDtype", pvAccumDtype,
                                              "pvAccumDtype must be one of [fp32, fp32+fp32, fp32+fp16]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckSmScale(double smScale)
{
    if (!std::isfinite(smScale) || std::fabs(smScale) > static_cast<double>(std::numeric_limits<float>::max())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "smScale", std::to_string(smScale).c_str(),
                                              "smScale must be finite and within the representable range of fp32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckTensorDtype(const aclTensor *q, const aclTensor *k, const aclTensor *v)
{
    const DataType dtype = q->GetDataType();
    if (dtype != DataType::DT_FLOAT16 && dtype != DataType::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(API_NAME, "q", op::ToString(dtype).GetString(), "DT_FLOAT16 or DT_BF16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (k->GetDataType() != dtype || v->GetDataType() != dtype) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "q.dtype, k.dtype, v.dtype",
                                              "mismatched input dtypes", "q/k/v must share the same fp16 or bf16 dtype");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckDimValue(const aclTensor *tensor, const char *tensorName)
{
    const auto &shape = tensor->GetViewShape();
    if (shape.GetDimNum() != DIM_NUM_4) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(API_NAME, tensorName, (std::to_string(shape.GetDimNum()) + "D").c_str(), "4D");
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim <= 0 || dim > MAX_DIM_VALUE) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, tensorName, std::to_string(dim).c_str(),
                                                  "each dim must be positive and less than 2^32 - 64");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckLastDimStride(const aclTensor *tensor, const char *tensorName)
{
    const auto &strides = tensor->GetViewStrides();
    if (strides.size() < DIM_NUM_4 || strides[DIM_D] != 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, tensorName,
                                              std::to_string(strides.size() < DIM_NUM_4 ? -1 : strides[DIM_D]).c_str(),
                                              "the stride of the last dim (head_dim) must be 1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckAllTensorDims(const aclTensor *q, const aclTensor *k, const aclTensor *v)
{
    for (const auto &item : {std::make_pair(q, "q"), std::make_pair(k, "k"), std::make_pair(v, "v")}) {
        const aclnnStatus ret = CheckDimValue(item.first, item.second);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckAllLastDimStrides(const aclTensor *q, const aclTensor *k, const aclTensor *v)
{
    for (const auto &item : {std::make_pair(q, "q"), std::make_pair(k, "k"), std::make_pair(v, "v")}) {
        const aclnnStatus ret = CheckLastDimStride(item.first, item.second);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
    }
    return ACLNN_SUCCESS;
}

// 依据布局抽取 (B, H, S, D) 语义维度：HND = [B, H, S, D]；NHD = [B, S, H, D]
void GetBhsd(const aclTensor *tensor, bool isNhd, int64_t &b, int64_t &h, int64_t &s, int64_t &d)
{
    const auto &shape = tensor->GetViewShape();
    b = shape.GetDim(DIM_B);
    h = isNhd ? shape.GetDim(DIM_S) : shape.GetDim(DIM_H);
    s = isNhd ? shape.GetDim(DIM_H) : shape.GetDim(DIM_S);
    d = shape.GetDim(DIM_D);
}

aclnnStatus CheckShapeRelations(const aclTensor *q, const aclTensor *k, const aclTensor *v, bool isNhd,
                                bool isCausal)
{
    int64_t qb = 0, qh = 0, sq = 0, qd = 0;
    int64_t kb = 0, kh = 0, sk = 0, kd = 0;
    int64_t vb = 0, vh = 0, sv = 0, vd = 0;
    GetBhsd(q, isNhd, qb, qh, sq, qd);
    GetBhsd(k, isNhd, kb, kh, sk, kd);
    GetBhsd(v, isNhd, vb, vh, sv, vd);

    // k/v 必须同 shape（b、h、s、d 全一致）
    if (kb != vb || kh != vh || sk != sv || kd != vd) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(API_NAME, "k.shape, v.shape",
                                               "key and value shapes differ", "k and v must share the same shape");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // b、d 在 q/k/v 间一致
    if (qb != kb || qd != kd) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(API_NAME, "q.shape, k.shape",
                                               "batch or head_dim mismatch between q and k/v",
                                               "q and k/v must share the same batch size and head_dim");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // head_dim 上界：D <= 128（pad 到 64/128 由下游处理）
    if (qd > MAX_HEAD_DIM) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "headDim", std::to_string(qd).c_str(),
                                              "head_dim must not be greater than 128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // GQA：Hq % Hk == 0
    if (qh % kh != 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(API_NAME, "numQoHeads, numKvHeads",
                                               "numQoHeads is not divisible by numKvHeads",
                                               "num_qo_heads must be divisible by num_kv_heads");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 因果 mask 仅支持 Sq == Sk
    if (isCausal && sq != sk) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(API_NAME, "qoLen, kvLen", "causal mask requires qoLen == kvLen",
                                               "is_causal is valid only when qo_len equals kv_len");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckOutTensor(const aclTensor *out, const aclTensor *q, bool isNhd)
{
    const auto &qShape = q->GetViewShape();
    const auto &outShape = out->GetViewShape();
    if (out->GetDataType() != q->GetDataType()) {
        OP_LOGE_FOR_INVALID_DTYPE(API_NAME, "out", op::ToString(out->GetDataType()).GetString(),
                                  op::ToString(q->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (outShape.GetDimNum() != DIM_NUM_4 || outShape != qShape) {
        OP_LOGE_FOR_INVALID_SHAPE(API_NAME, "out",
                                  op::ToString(outShape).GetString(), op::ToString(qShape).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckLseTensor(const aclTensor *lse, const aclTensor *q, bool isNhd)
{
    int64_t b = 0, h = 0, s = 0, d = 0;
    GetBhsd(q, isNhd, b, h, s, d);
    const auto &lseShape = lse->GetViewShape();
    if (lse->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(API_NAME, "lse", op::ToString(lse->GetDataType()).GetString(),
                                  op::ToString(DataType::DT_FLOAT).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto expectShape = op::Shape({b, h, s});
    if (lseShape.GetDimNum() != DIM_NUM_3 || lseShape != expectShape) {
        OP_LOGE_FOR_INVALID_SHAPE(API_NAME, "lse", op::ToString(lseShape).GetString(),
                                  op::ToString(expectShape).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// NHD -> HND：构造 1、2 维互换的视图（shape 与 strides 同步换位，零拷贝），由调用方再行 Contiguous
const aclTensor *TransposeNhdToHnd(const aclTensor *tensor, aclOpExecutor *executor)
{
    op::Shape hndShape = tensor->GetViewShape();
    op::Strides hndStrides = tensor->GetViewStrides();
    std::swap(hndShape[DIM_H], hndShape[DIM_S]);
    std::swap(hndStrides[DIM_H], hndStrides[DIM_S]);
    return executor->CreateView(
        tensor, hndShape, tensor->GetStorageShape(), hndStrides, tensor->GetViewOffset());
}

} // namespace

OP_TYPE_REGISTER(SageAttention2);

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnSa2Attention2GetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const char *tensorLayout, bool isCausal,
    const char *qkQuantGran, double smScale, const char *pvAccumDtype, bool smoothK, bool smoothV, bool returnLse,
    aclTensor *out, aclTensor *lse, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnSa2Attention2,
                   DFX_IN(q, k, v, tensorLayout, isCausal, qkQuantGran, smScale, pvAccumDtype, smoothK, smoothV,
                          returnLse),
                   DFX_OUT(out, lse));

    const aclnnStatus notNullRet = CheckNotNull(q, k, v, out, tensorLayout, qkQuantGran, pvAccumDtype, workspaceSize,
                                                executor);
    CHECK_RET(notNullRet == ACLNN_SUCCESS, notNullRet);

    if (!IsAscend950Platform()) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "SageAttention2 only supports the Ascend950 platform series.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }

    int64_t granCode = QUANT_GRAN_PER_WARP;
    int64_t accumCode = ACCUM_FP32;
    const aclnnStatus stringRet = CheckStringParams(tensorLayout, qkQuantGran, pvAccumDtype, granCode, accumCode);
    CHECK_RET(stringRet == ACLNN_SUCCESS, stringRet);
    const aclnnStatus smScaleRet = CheckSmScale(smScale);
    CHECK_RET(smScaleRet == ACLNN_SUCCESS, smScaleRet);

    const bool isNhd = strcmp(tensorLayout, "NHD") == 0;
    const aclnnStatus dtypeRet = CheckTensorDtype(q, k, v);
    CHECK_RET(dtypeRet == ACLNN_SUCCESS, dtypeRet);
    const aclnnStatus dimRet = CheckAllTensorDims(q, k, v);
    CHECK_RET(dimRet == ACLNN_SUCCESS, dimRet);
    const aclnnStatus strideRet = CheckAllLastDimStrides(q, k, v);
    CHECK_RET(strideRet == ACLNN_SUCCESS, strideRet);
    const aclnnStatus shapeRet = CheckShapeRelations(q, k, v, isNhd, isCausal);
    CHECK_RET(shapeRet == ACLNN_SUCCESS, shapeRet);
    const aclnnStatus outRet = CheckOutTensor(out, q, isNhd);
    CHECK_RET(outRet == ACLNN_SUCCESS, outRet);
    // return_lse=true 时必须提供 FP32 的 [B, Hq, Sq] 输出；为 false 时忽略外部 lse（内核不写）
    if (returnLse) {
        if (lse == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s: lse is required when returnLse is true.", API_NAME);
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        const aclnnStatus lseRet = CheckLseTensor(lse, q, isNhd);
        CHECK_RET(lseRet == ACLNN_SUCCESS, lseRet);
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto *executorImpl = uniqueExecutor.get();

    if (smoothV && accumCode != ACCUM_FP32) {
        OP_LOGW("SageAttention2: smooth_v only takes effect with pv_accum_dtype fp32, it will be ignored.");
        smoothV = false;
    }

    // 布局统一为 HND：NHD 先建 1/2 维互换视图，再 Contiguous 物化；HND 直接 Contiguous（连续输入零拷贝）
    const aclTensor *qFinal = q;
    const aclTensor *kFinal = k;
    const aclTensor *vFinal = v;
    if (isNhd) {
        qFinal = TransposeNhdToHnd(q, executorImpl);
        kFinal = TransposeNhdToHnd(k, executorImpl);
        vFinal = TransposeNhdToHnd(v, executorImpl);
        CHECK_RET(qFinal != nullptr && kFinal != nullptr && vFinal != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    qFinal = l0op::Contiguous(qFinal, executorImpl);
    kFinal = l0op::Contiguous(kFinal, executorImpl);
    vFinal = l0op::Contiguous(vFinal, executorImpl);
    CHECK_RET(qFinal != nullptr && kFinal != nullptr && vFinal != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // out 直写优化：非 NHD 且存储连续时内核直接写用户 out，免去 ViewCopy；否则内部分配后回拷
    aclTensor *outFinal = out;
    const bool outDirectWrite = !isNhd && IsContiguous(out);
    if (!outDirectWrite) {
        outFinal = executorImpl->AllocTensor(q->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
        CHECK_RET(outFinal != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // lse：return_lse=false 时占位（内核不写）；直写要求连续，非连续时内部分配后回拷
    aclTensor *lseFinal = nullptr;
    bool lseDirectWrite = false;
    if (returnLse) {
        lseDirectWrite = IsContiguous(lse);
        lseFinal = lseDirectWrite ? lse : executorImpl->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND,
                                                                   Format::FORMAT_ND);
        CHECK_RET(lseFinal != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        lseFinal = executorImpl->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
        CHECK_RET(lseFinal != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto inferRet = INFER_SHAPE(SageAttention2,
                                OP_INPUT(qFinal, kFinal, vFinal),
                                OP_OUTPUT(outFinal, lseFinal),
                                OP_ATTR(smScale, smoothK, isCausal, returnLse, granCode, accumCode, smoothV));
    if (inferRet != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "SageAttention2 InferShape failed.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // attr 顺序与 op_host/sa2_def.cpp 严格一致：
    // sm_scale, smooth_k, is_causal, return_lse, qk_quant_gran, pv_accum_dtype, smooth_v
    auto launchRet = ADD_TO_LAUNCHER_LIST_AICORE(SageAttention2,
                                                 OP_INPUT(qFinal, kFinal, vFinal),
                                                 OP_OUTPUT(outFinal, lseFinal),
                                                 OP_ATTR(smScale, smoothK, isCausal, returnLse, granCode, accumCode,
                                                         smoothV));
    if (launchRet != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "SageAttention2 ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (!outDirectWrite) {
        aclTensor *outTarget = out;
        if (isNhd) {
            outTarget = const_cast<aclTensor *>(TransposeNhdToHnd(out, executorImpl));
            CHECK_RET(outTarget != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        CHECK_RET(l0op::ViewCopy(outFinal, outTarget, executorImpl) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (returnLse && !lseDirectWrite) {
        CHECK_RET(l0op::ViewCopy(lseFinal, lse, executorImpl) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = executorImpl->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSa2Attention2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSa2Attention2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
