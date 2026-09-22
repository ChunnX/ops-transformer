/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_tutel_combine_gates_infershape.cpp
 * \brief MoeTutelCombineGates shape and dtype inference.
 */

#include "register/op_impl_registry.h"                // IMPL_OP_INFERSHAPE macro
#include "exe_graph/runtime/infer_shape_context.h"    // InferShapeContext, gert::Shape
#include "exe_graph/runtime/infer_datatype_context.h" // InferDataTypeContext
#include "op_common/log/log.h"                        // OP_CHECK_NULL_WITH_CONTEXT, OP_LOGE macros

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// 结构化错误上报宏的选择依据
//
// 本文件的所有**输入侧**校验一律用 CANN 9.2.0 的
// OP_LOGE_FOR_INVALID_{SHAPE,SHAPES,SHAPEDIM,SHAPESIZE,FORMAT,DTYPE}[S]_WITH_REASON
// 宏（`$ASCEND_HOME_PATH/x86_64-linux/include/op_common/log/log.h:325/350/401/478/554/630/655`），
// 它们经 OP_LOGE_LIBOPAPI_REPORT + REPORT_PREDEFINED_ERR_MSG 做预定义码（EZ0009~EZ0035）
// 结构化上报。与上一轮的差别（当时的顶替写法与其理由已作废）：
//   * CANN 9.0.0-beta.2 里确实没有这一族宏（当时全树 grep "OP_LOGE_FOR" 零命中），
//     只能用 OP_LOGE_WITH_INVALID_INPUT_*（EZ0001~EZ0007）顶替，且**输出侧**校验因为
//     那些宏的文案写死为「第 N 个输入 ...」而只能保留自由文本 OP_LOGE；
//   * 9.2.0 起这一族宏存在，且它们的第一个参数族是**参数名字符串**（不限于输入、
//     也不限于单个参数：复数宏收 `paramNames`），因此输出侧（gates_grad）与多参数
//     联合判定都能用结构化宏上报，不再需要自由文本兜底。
// 逐处映射见各校验点注释：rank → SHAPEDIM_WITH_REASON；shape 关系（含输出侧）→
// SHAPES_WITH_REASON；元素数 → SHAPESIZE_WITH_REASON；format → FORMAT_WITH_REASON；
// dtype（含配对与输出侧）→ DTYPE/DTYPES_WITH_REASON。
// ---------------------------------------------------------------------------

using namespace ge;

namespace ops {

namespace {

// ---------------------------------------------------------------------------
// Ports — proto.md §1: 4 inputs (x / y_grad / indices / locations) -> 1 output
// ---------------------------------------------------------------------------
constexpr size_t kInputX = 0;
constexpr size_t kInputYGrad = 1;
constexpr size_t kInputIndices = 2;
constexpr size_t kInputLocations = 3;
constexpr size_t kOutputGatesGrad = 0;
constexpr size_t kInputNum = 4;

// ---------------------------------------------------------------------------
// Rank contract — design/InferShapeDtype.md "rank 契约" (fixed ranks, not ranges)
// ---------------------------------------------------------------------------
constexpr size_t kRankX = 2;
constexpr size_t kRankYGrad = 3;
constexpr size_t kRankIndices = 2;
constexpr size_t kRankLocations = 2;
constexpr size_t kRankOutput = 2;

// gert::Shape uses dim_num == 0xFFFFFFFF for "unknown rank"; unknown dims are
// negative dim values. Neither is ever guessed (design §4).
constexpr size_t kUnknownRank = 0xFFFFFFFFU;

bool IsUnknownRank(const gert::Shape &shape)
{
    return shape.GetDimNum() == kUnknownRank;
}

bool IsKnownDim(const int64_t dim)
{
    return dim >= 0;
}

// Two dims disagree only when both are known and different: an unknown dim can
// never be judged at compile time (design §4 "未知维：原样透传，不猜测数值").
bool DimsConflict(const int64_t lhs, const int64_t rhs)
{
    return IsKnownDim(lhs) && IsKnownDim(rhs) && lhs != rhs;
}

bool AllDimsKnown(const gert::Shape &shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (!IsKnownDim(shape.GetDim(i))) {
            return false;
        }
    }
    return true;
}

// Compile-time format contract: the operator has no channel axis and every
// declared format is FORMAT_ND (op_host/*_def.cpp, proto.md §1), so a non-ND
// runtime format is rejected here. A missing desc is not judged.
ge::graphStatus CheckFormat(gert::InferShapeContext *context, const char *name, const gert::CompileTimeTensorDesc *desc)
{
    if (desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (desc->GetStorageFormat() != ge::FORMAT_ND) {
        const std::string formatStr = std::to_string(static_cast<int32_t>(desc->GetStorageFormat()));
        // 单参数 format 违规 → ..._FORMAT_WITH_REASON（EZ0035）。
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context->GetNodeName(), name, formatStr.c_str(),
                                               "the operator declares FORMAT_ND only (it has no channel axis)");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace

// ---------------------------------------------------------------------------
// InferShape4MoeTutelCombineGates(context) — shape inference + contract checks
//
// Parameters:
//   context — InferShapeContext providing access to input shapes/descs and
//             allowing the output shape to be set
//
// Returns:
//   ge::graphStatus — ge::GRAPH_SUCCESS on success (output shape written),
//                     ge::GRAPH_FAILED when the shape contract is violated
//
// MACRO: OP_CHECK_NULL_WITH_CONTEXT(context, ptr)
//   Checks that ptr is not null; if null, logs an error with context info
//   and returns GRAPH_FAILED (a missing REQUIRED input lands here).
// ---------------------------------------------------------------------------
static ge::graphStatus InferShape4MoeTutelCombineGates(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(kInputX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape *yGradShape = context->GetInputShape(kInputYGrad);
    OP_CHECK_NULL_WITH_CONTEXT(context, yGradShape);
    const gert::Shape *indicesShape = context->GetInputShape(kInputIndices);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);
    const gert::Shape *locationsShape = context->GetInputShape(kInputLocations);
    OP_CHECK_NULL_WITH_CONTEXT(context, locationsShape);

    // GetOutputShape(0): mutable reference to output gates_grad's shape descriptor
    gert::Shape *outputShape = context->GetOutputShape(kOutputGatesGrad);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    const gert::Shape *const inputs[kInputNum] = {xShape, yGradShape, indicesShape, locationsShape};
    const char *const inputNames[kInputNum] = {"x", "y_grad", "indices", "locations"};

    // 1) unknown rank: propagate as unknown rank, do not infer and do not fail
    for (size_t i = 0; i < kInputNum; ++i) {
        if (IsUnknownRank(*inputs[i])) {
            outputShape->SetDimNum(kUnknownRank);
            OP_LOGD(context->GetNodeName(), "input %s has unknown rank; gates_grad set to unknown rank", inputNames[i]);
            return ge::GRAPH_SUCCESS;
        }
    }

    // 2) rank contract: x == 2, y_grad == 3, indices == 2, locations == 2
    const size_t expectedRanks[kInputNum] = {kRankX, kRankYGrad, kRankIndices, kRankLocations};
    for (size_t i = 0; i < kInputNum; ++i) {
        if (inputs[i]->GetDimNum() != expectedRanks[i]) {
            const std::string gotRankStr = "rank=" + std::to_string(inputs[i]->GetDimNum());
            const std::string wantRankReasonStr =
                "input " + std::string(inputNames[i]) + " must be rank " + std::to_string(expectedRanks[i]);
            // 单参数「维数（rank）」违规 → ..._SHAPEDIM_WITH_REASON（EZ0012）。
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), inputNames[i], gotRankStr.c_str(),
                                                     wantRankReasonStr.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    // 3) format contract: 4 inputs + output must run as FORMAT_ND
    for (size_t i = 0; i < kInputNum; ++i) {
        if (CheckFormat(context, inputNames[i], context->GetInputDesc(i)) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    if (CheckFormat(context, "gates_grad", context->GetOutputDesc(kOutputGatesGrad)) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 4) shape binding contract (design §2 items 5-8); unknown dims are skipped
    const int64_t samples = xShape->GetDim(0);   // S
    const int64_t hidden = xShape->GetDim(1);    // H
    const int64_t batch = yGradShape->GetDim(0); // B
    const int64_t hiddenYGrad = yGradShape->GetDim(2);

    // 形状关系校验：都是「两个参数联合判定」，故一律用复数宏
    // OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON（EZ0010，收 paramNames 参数名列表），
    // 冲突的两个维度名与数值放进 incorrectShapes / reason 两个文本里。
    if (DimsConflict(hidden, hiddenYGrad)) {
        const std::string gotStr = "x.shape[1]=" + std::to_string(hidden);
        const std::string wantStr = "y_grad.shape[2]=" + std::to_string(hiddenYGrad);
        const std::string gotPairStr = gotStr + ", " + wantStr;
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, y_grad", gotPairStr.c_str(),
                                               "x.shape[1] (H) must equal y_grad.shape[2] (H)");
        return ge::GRAPH_FAILED;
    }
    if (DimsConflict(indicesShape->GetDim(0), batch)) {
        const std::string gotStr = "indices.shape[0]=" + std::to_string(indicesShape->GetDim(0));
        const std::string wantStr = "y_grad.shape[0]=" + std::to_string(batch);
        const std::string gotPairStr = gotStr + ", " + wantStr;
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "indices, y_grad", gotPairStr.c_str(),
                                               "indices.shape[0] (B) must equal y_grad.shape[0] (B)");
        return ge::GRAPH_FAILED;
    }
    if (DimsConflict(indicesShape->GetDim(1), samples)) {
        const std::string gotStr = "indices.shape[1]=" + std::to_string(indicesShape->GetDim(1));
        const std::string wantStr = "x.shape[0]=" + std::to_string(samples);
        const std::string gotPairStr = gotStr + ", " + wantStr;
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "indices, x", gotPairStr.c_str(),
                                               "indices.shape[1] (S) must equal x.shape[0] (S)");
        return ge::GRAPH_FAILED;
    }
    if (DimsConflict(locationsShape->GetDim(1), samples)) {
        const std::string gotStr = "locations.shape[1]=" + std::to_string(locationsShape->GetDim(1));
        const std::string wantStr = "x.shape[0]=" + std::to_string(samples);
        const std::string gotPairStr = gotStr + ", " + wantStr;
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "locations, x", gotPairStr.c_str(),
                                               "locations.shape[1] (S) must equal x.shape[0] (S)");
        return ge::GRAPH_FAILED;
    }
    // S > 0 makes "locations.shape[0] == B" equivalent to the element-count
    // contract below; with S == 0 the leading dim is unconstrained (design §2).
    if (samples != 0 && DimsConflict(locationsShape->GetDim(0), batch)) {
        const std::string gotStr = "locations.shape[0]=" + std::to_string(locationsShape->GetDim(0));
        const std::string wantStr = "y_grad.shape[0]=" + std::to_string(batch);
        const std::string gotPairStr = gotStr + ", " + wantStr;
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "locations, y_grad", gotPairStr.c_str(),
                                               "locations.shape[0] (B) must equal y_grad.shape[0] (B) when S != 0");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *const taskShapes[2] = {indicesShape, locationsShape};
    const char *const taskNames[2] = {"indices", "locations"};
    if (IsKnownDim(samples) && IsKnownDim(batch)) {
        const int64_t taskNum = samples * batch;
        for (size_t i = 0; i < 2; ++i) {
            const int64_t dim0 = taskShapes[i]->GetDim(0);
            const int64_t dim1 = taskShapes[i]->GetDim(1);
            if (IsKnownDim(dim0) && IsKnownDim(dim1) && dim0 * dim1 != taskNum) {
                // 单参数「元素数」违规 → ..._SHAPESIZE_WITH_REASON（EZ0015）。
                const std::string gotSizeStr = std::to_string(dim0 * dim1);
                const std::string sizeReasonStr = "must equal S*B = " + std::to_string(samples) + "*" +
                                                  std::to_string(batch) + " = " + std::to_string(taskNum) +
                                                  " elements (indices / locations are consumed as flat S*B arrays)";
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), taskNames[i], gotSizeStr.c_str(),
                                                          sizeReasonStr.c_str());
                return ge::GRAPH_FAILED;
            }
        }
    }

    // 5) declared output shape: a fully-known gates_grad shape must equal
    //    indices.shape; otherwise the caller's declaration contradicts the
    //    operator contract (rejected rather than silently overwritten).
    if (!IsUnknownRank(*outputShape) && outputShape->GetDimNum() > 0 && AllDimsKnown(*outputShape) &&
        AllDimsKnown(*indicesShape)) {
        bool same = (outputShape->GetDimNum() == indicesShape->GetDimNum());
        for (size_t i = 0; same && i < outputShape->GetDimNum(); ++i) {
            same = (outputShape->GetDim(i) == indicesShape->GetDim(i));
        }
        if (!same) {
            // 输出侧同样用复数 STRUCTURED 宏：9.2.0 的 ..._SHAPES_WITH_REASON 收的是
            // **参数名字符串**（EZ0010 的文案为 "Parameters %s of %s have incorrect shapes %s"），
            // 不再像 9.0.0-beta.2 的 OP_LOGE_WITH_INVALID_INPUT_SHAPE 那样把文案写死为
            // 「第 N 个输入」，因此 gates_grad 能作为具名参数上报（见文件头选择依据）。
            const std::string outShapesStr = "gates_grad rank " + std::to_string(outputShape->GetDimNum()) +
                                             " vs indices rank " + std::to_string(indicesShape->GetDimNum());
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "gates_grad, indices", outShapesStr.c_str(),
                                                   "the declared gates_grad shape must equal indices.shape");
            return ge::GRAPH_FAILED;
        }
    }

    // 6) derive: gates_grad.shape = indices.shape (proto.md §4)
    *outputShape = *indicesShape;

    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// InferDataType4MoeTutelCombineGates(context) — 输出 dtype 推导 + dtype 契约校验
//
// 依据 design/InferShapeDtype.md「InferDataType」1)~3)（与 spec.yaml
// dtype_policy / outputs[0].dtype_rule 逐行一致，不自行发挥）：
//   1) gates_grad.dtype = x.dtype（promotion = same_as_first_input）；indices /
//      locations 不参与输出 dtype 推导（只作地址索引，从不做算术转换）；
//   2) 合法组合只有 3 组（bf16 / fp16 / fp32）：x / y_grad 必须同属
//      {DT_BF16, DT_FLOAT16, DT_FLOAT} 且二者相等；indices / locations 恒为
//      DT_INT32（二者相等由「都等于 int32」蕴含）；
//   3) 预置/声明的输出 dtype 若本身已是受支持浮点类型却 ≠ x.dtype ⇒ 拒绝，
//      不静默覆盖。
//
// 图模式（GE）通路使用本注册（本算子的 aclnn 单算子调用方式已删除）：注册表按
// OpType 合并后 `OP_IMPL` 上同时带 [Tiling][TilingParse][InferShape][InferDataType]，
// 推理与 tiling 的判决口径因此天然一致。
//
// 返回：ge::GRAPH_SUCCESS（gates_grad dtype 已写入）/ ge::GRAPH_FAILED（dtype 组合非法）。
// ---------------------------------------------------------------------------
static bool IsSupportedFloatDtype(const ge::DataType dtype)
{
    return dtype == ge::DT_BF16 || dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT;
}

static ge::graphStatus InferDataType4MoeTutelCombineGates(gert::InferDataTypeContext *context)
{
    const ge::DataType xDtype = context->GetInputDataType(kInputX);
    const ge::DataType yGradDtype = context->GetInputDataType(kInputYGrad);
    const ge::DataType indicesDtype = context->GetInputDataType(kInputIndices);
    const ge::DataType locationsDtype = context->GetInputDataType(kInputLocations);

    // 结构化上报用的 dtype 文本（宏内部会把实参展开两次并在 std::vector<const char*>
    // 里继续使用，故先具名，与 tiling 侧同一写法）。
    const std::string xDtypeStr = std::to_string(static_cast<int32_t>(xDtype));
    const std::string yDtypeStr = std::to_string(static_cast<int32_t>(yGradDtype));
    const std::string idxDtypeStr = std::to_string(static_cast<int32_t>(indicesDtype));
    const std::string locDtypeStr = std::to_string(static_cast<int32_t>(locationsDtype));
    const std::string floatReasonStr = "must be one of DT_BF16/DT_FLOAT16/DT_FLOAT (the 3 supported combinations)";

    // 1) 两个浮点输入各自必须落在支持面内，且互相配对
    if (!IsSupportedFloatDtype(xDtype)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x", xDtypeStr.c_str(), floatReasonStr.c_str());
        return ge::GRAPH_FAILED;
    }
    if (!IsSupportedFloatDtype(yGradDtype)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "y_grad", yDtypeStr.c_str(),
                                              floatReasonStr.c_str());
        return ge::GRAPH_FAILED;
    }
    if (xDtype != yGradDtype) {
        const std::string gotStr = "x=" + xDtypeStr + ", y_grad=" + yDtypeStr;
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x, y_grad", gotStr.c_str(),
                                               "the two floating-point inputs must share one dtype "
                                               "(promotion = same_as_first_input)");
        return ge::GRAPH_FAILED;
    }

    // 2) 两个索引张量恒为 int32（二者一致由「都等于 int32」蕴含）
    if (indicesDtype != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "indices", idxDtypeStr.c_str(),
                                              "indices must be DT_INT32 (it is used as an address index only)");
        return ge::GRAPH_FAILED;
    }
    if (locationsDtype != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "locations", locDtypeStr.c_str(),
                                              "locations must be DT_INT32 (it is used as an address index only)");
        return ge::GRAPH_FAILED;
    }

    // 3) 已声明的输出 dtype 若受支持浮点却与推导结果不符 ⇒ 拒绝（不静默覆盖）
    const ge::DataType declaredDtype = context->GetOutputDataType(kOutputGatesGrad);
    if (IsSupportedFloatDtype(declaredDtype) && declaredDtype != xDtype) {
        const std::string outDtypeStr = std::to_string(static_cast<int32_t>(declaredDtype));
        const std::string gotStr = "x=" + xDtypeStr + ", gates_grad=" + outDtypeStr;
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x, gates_grad", gotStr.c_str(),
                                               "the declared gates_grad dtype must equal x.dtype "
                                               "(gates_grad.dtype follows x.dtype)");
        return ge::GRAPH_FAILED;
    }

    // 推导：gates_grad.dtype = x.dtype
    if (context->SetOutputDataType(kOutputGatesGrad, xDtype) != ge::GRAPH_SUCCESS) {
        // 保留自由文本 OP_LOGE：这不是输入校验失败，而是框架接口调用失败，R6 的
        // OP_LOGE_FOR_INVALID_*（语义为「某参数取值非法」）没有对应项，硬套会报错参数名。
        OP_LOGE(context->GetNodeName(), "dtype_not_supported: cannot set gates_grad dtype");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// IMPL_OP_INFERSHAPE(MoeTutelCombineGates) —— InferShape / InferDataType 双挂：
//   .InferShape(func)      — 形状推导 + shape 契约校验
//   .InferDataType(func)   — 输出 dtype 推导 + dtype 契约校验
// 只挂 InferShape 会让 GE 在本注册面上拿不到输出 dtype 推导；两个回调挂在**同一个**
// OpImplRegisterV2 上，注册表最终把它与 op_graph 的 IMPL_OP 注册合并成
// `register OP_IMPL : [Tiling][TilingParse][InferShape][InferDataType]` 一条。
IMPL_OP_INFERSHAPE(MoeTutelCombineGates)
    .InferShape(InferShape4MoeTutelCombineGates)
    .InferDataType(InferDataType4MoeTutelCombineGates);

} // namespace ops
