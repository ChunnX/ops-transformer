/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef GROUPED_ROTATE_QUANT_CONTEXT_H
#define GROUPED_ROTATE_QUANT_CONTEXT_H

#include "register/op_impl_registry.h"
#include "grouped_rotate_quant_model.h"

namespace optiling::grq {
// Keep this order identical to OP_ATTR in op_api/grouped_rotate_quant.cpp.
enum AttrIndex : size_t {
    SPLIT_ITEM, GROUP_TYPE, GROUP_LIST_TYPE, AXIS, ROUND_MODE, SCALE_ALG,
    DST_TYPE_MAX, TRANS, DST_TYPE, TUNING_CONFIG
};
enum InputIndex : size_t { X, ROTATION, ALPHA, GROUP_LIST };
enum OutputIndex : size_t { Y, SCALE };

inline bool GetOutputType(int64_t dtype, OutputType &type)
{
    if (dtype == ge::DT_FLOAT4_E2M1) {
        type = OutputType::FP4;
    } else if (dtype == ge::DT_FLOAT8_E4M3FN) {
        type = OutputType::FP8_E4M3;
    } else if (dtype == ge::DT_FLOAT8_E5M2) {
        type = OutputType::FP8_E5M2;
    } else {
        return false;
    }
    return true;
}

template <class Context>
bool ReadAttributes(Context *context, Attributes &a, ge::DataType &dstType, std::string &error)
{
    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return Fail(error, "missing attributes");
    }
    auto readInt = [attrs](size_t index, int64_t &value) {
        const auto *ptr = attrs->template GetAttrPointer<int64_t>(index);
        if (ptr == nullptr) {
            return false;
        }
        value = *ptr;
        return true;
    };
    int64_t dtype = 0;
    if (!readInt(SPLIT_ITEM, a.splitItem) || !readInt(GROUP_TYPE, a.groupType) ||
        !readInt(GROUP_LIST_TYPE, a.groupListType) || !readInt(AXIS, a.axis) ||
        !readInt(SCALE_ALG, a.scaleAlg) || !readInt(DST_TYPE, dtype) || !GetOutputType(dtype, a.outputType)) {
        return Fail(error, "missing integer attributes or unsupported dst_type");
    }
    const auto *round = attrs->template GetAttrPointer<char>(ROUND_MODE);
    const auto *maximum = attrs->template GetAttrPointer<float>(DST_TYPE_MAX);
    const auto *trans = attrs->template GetAttrPointer<bool>(TRANS);
    if (round == nullptr || maximum == nullptr || trans == nullptr) {
        return Fail(error, "missing round_mode/dst_type_max/trans");
    }
    a.roundMode = round;
    a.dstTypeMax = *maximum;
    a.trans = *trans;
    dstType = static_cast<ge::DataType>(dtype);
    const auto *tuning = attrs->template GetAttrPointer<gert::ContinuousVector>(TUNING_CONFIG);
    if (tuning != nullptr && tuning->GetSize() > 0) {
        const auto *values = reinterpret_cast<const int64_t *>(tuning->GetData());
        if (values == nullptr) {
            return Fail(error, "tuning_config has no backing data");
        }
        a.expectedM = values[0];
    }
    return CheckAttributes(a, error);
}

inline const gert::Shape &LogicalShape(const gert::Shape &shape) { return shape; }
inline const gert::Shape &LogicalShape(const gert::StorageShape &shape) { return shape.GetOriginShape(); }
template <class GertShape>
Shape ReadShape(const GertShape &source)
{
    const auto &shape = LogicalShape(source);
    Shape result;
    result.reserve(shape.GetDimNum());
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        result.push_back(shape.GetDim(i));
    }
    return result;
}

template <class Context>
bool ReadInputList(Context *context, size_t irIndex, std::vector<Shape> &shapes, ge::DataType dtype,
                   bool optional, std::string &error)
{
    if (optional && context->GetOptionalInputShape(irIndex) == nullptr) {
        return true;
    }
    const auto *info = context->GetIrInputInstanceInfo(irIndex);
    const size_t count = info == nullptr ? 0 : info->GetInstanceNum();
    if ((optional && count > 1) || (!optional && (count == 0 || count > GroupedRotateQuant::MAX_GROUP_NUM))) {
        return Fail(error, "invalid input instance count");
    }
    for (size_t i = 0; i < count; ++i) {
        const size_t index = info->GetInstanceStart() + i;
        const auto *shape = context->GetInputShape(index);
        const auto *desc = context->GetInputDesc(index);
        if (shape == nullptr || desc == nullptr || desc->GetDataType() != dtype ||
            desc->GetStorageFormat() != ge::FORMAT_ND || desc->GetOriginFormat() != ge::FORMAT_ND) {
            return Fail(error, "missing input metadata, inconsistent input dtype, or non-ND format");
        }
        shapes.push_back(ReadShape(*shape));
    }
    return true;
}

template <class Context>
bool ReadInputs(Context *context, InputParams &p, ge::DataType &dstType, std::string &error)
{
    if (!ReadAttributes(context, p.attrs, dstType, error)) {
        return false;
    }
    const auto *xInfo = context->GetIrInputInstanceInfo(X);
    if (xInfo == nullptr || xInfo->GetInstanceNum() == 0) {
        return Fail(error, "missing x list");
    }
    const auto *xDesc = context->GetInputDesc(xInfo->GetInstanceStart());
    if (xDesc == nullptr || (xDesc->GetDataType() != ge::DT_FLOAT16 && xDesc->GetDataType() != ge::DT_BF16)) {
        return Fail(error, "x must be FLOAT16 or BFLOAT16");
    }
    const auto dtype = xDesc->GetDataType();
    std::vector<Shape> alpha;
    std::vector<Shape> groupList;
    if (!ReadInputList(context, X, p.x, dtype, false, error) ||
        !ReadInputList(context, ROTATION, p.rotation, dtype, false, error) ||
        !ReadInputList(context, ALPHA, alpha, ge::DT_BF16, true, error) ||
        !ReadInputList(context, GROUP_LIST, groupList, ge::DT_INT64, true, error)) {
        return false;
    }
    p.hasAlpha = !alpha.empty();
    p.hasGroupList = !groupList.empty();
    if (p.hasAlpha) {
        p.alpha = alpha[0];
    }
    if (p.hasGroupList) {
        p.groupList = groupList[0];
    }
    const auto *yInfo = context->GetIrOutputInstanceInfo(Y);
    const auto *scaleInfo = context->GetIrOutputInstanceInfo(SCALE);
    if (yInfo == nullptr || scaleInfo == nullptr) {
        return Fail(error, "missing y or scale output list");
    }
    p.yNum = yInfo->GetInstanceNum();
    p.scaleNum = scaleInfo->GetInstanceNum();
    return true;
}
} // namespace optiling::grq
#endif
