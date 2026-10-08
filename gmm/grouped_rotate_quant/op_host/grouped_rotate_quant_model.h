/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef GROUPED_ROTATE_QUANT_MODEL_H
#define GROUPED_ROTATE_QUANT_MODEL_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include "../op_kernel/arch35/grouped_rotate_quant_tiling_data.h"

namespace optiling::grq {
using Shape = std::vector<int64_t>;
enum class OutputType { FP4, FP8_E4M3, FP8_E5M2 };
struct Attributes {
    int64_t splitItem = 0;
    int64_t groupType = -1;
    int64_t groupListType = 0;
    int64_t axis = -1;
    std::string roundMode = "rint";
    int64_t scaleAlg = 0;
    float dstTypeMax = 0;
    bool trans = false;
    OutputType outputType = OutputType::FP4;
    int64_t expectedM = 0;
};
struct InputParams {
    Attributes attrs;
    std::vector<Shape> x;
    std::vector<Shape> rotation;
    bool hasAlpha = false;
    Shape alpha;
    bool hasGroupList = false;
    Shape groupList;
    size_t yNum = 0;
    size_t scaleNum = 0;
    size_t groupNum = 0;
    int64_t totalM = 0;
    int64_t maxM = 0;
    int64_t totalElements = 0;
    bool runtimeM = false;
};

inline bool Fail(std::string &error, const char *reason)
{
    error = reason;
    return false;
}
inline int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return value / divisor + (value % divisor != 0);
}
inline bool Add(int64_t a, int64_t b, int64_t &result)
{
    if (a < 0 || b < 0) {
        result = -1;
        return true;
    }
    if (a > std::numeric_limits<int64_t>::max() - b) {
        return false;
    }
    result = a + b;
    return true;
}
inline bool Multiply(int64_t a, int64_t b, int64_t &result)
{
    if (a == 0 || b == 0) {
        result = 0;
        return true;
    }
    if (a < 0 || b < 0) {
        result = -1;
        return true;
    }
    if (a > std::numeric_limits<int64_t>::max() / b) {
        return false;
    }
    result = a * b;
    return true;
}
inline bool Compatible(int64_t a, int64_t b) { return a == b || a == -1 || b == -1; }
inline bool ValidShape(const Shape &shape, bool symbolic)
{
    return std::all_of(shape.begin(), shape.end(), [symbolic](int64_t d) { return d >= (symbolic ? -1 : 0); });
}
inline bool FlattenM(const Shape &shape, int64_t &m)
{
    m = 1;
    // Zero dimensions dominate unknown dimensions and avoid spurious intermediate overflow.
    if (std::find(shape.begin(), shape.end() - 1, 0) != shape.end() - 1) {
        m = 0;
        return true;
    }
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
        if (!Multiply(m, shape[i], m)) {
            return false;
        }
    }
    return true;
}
inline Shape ScaleShape(Shape y)
{
    y.back() = y.back() < 0 ? -1 : CeilDiv(y.back(), 64);
    y.push_back(2);
    return y;
}

inline bool CheckAttributes(const Attributes &a, std::string &error)
{
    if ((a.groupType != -1 && a.groupType != 0) || a.splitItem < 0 || a.splitItem > 3 ||
        (a.groupListType != 0 && a.groupListType != 1) || a.trans || a.expectedM < 0) {
        return Fail(error, "unsupported group_type/split_item/group_list_type/trans/tuning_config");
    }
    const bool fp4 = a.outputType == OutputType::FP4;
    if (a.roundMode != "rint" && !(fp4 && (a.roundMode == "round" || a.roundMode == "floor"))) {
        return Fail(error, "round_mode is incompatible with dst_type");
    }
    if (!(fp4 ? (a.scaleAlg == 0 || a.scaleAlg == 2) : (a.scaleAlg == 0 || a.scaleAlg == 1))) {
        return Fail(error, "scale_alg is incompatible with dst_type");
    }
    if (!std::isfinite(a.dstTypeMax) ||
        (a.scaleAlg == 2 ? (a.dstTypeMax < 6 || a.dstTypeMax > 12) : a.dstTypeMax != 0)) {
        return Fail(error, "dst_type_max must be [6,12] for scale_alg=2, otherwise 0");
    }
    return true;
}

inline bool CheckLists(InputParams &p, bool symbolic, std::string &error)
{
    if (p.x.empty() || p.x.size() > GroupedRotateQuant::MAX_GROUP_NUM || p.rotation.empty() ||
        p.rotation.size() > GroupedRotateQuant::MAX_GROUP_NUM) {
        return Fail(error, "x and rotation must contain 1..1024 tensors");
    }
    p.groupNum = p.x.size();
    if (p.attrs.groupType == -1) {
        if (p.hasGroupList || p.attrs.splitItem >= 2) {
            return Fail(error, "independent tensors require absent group_list and separate outputs");
        }
    } else if (p.hasGroupList) {
        if (p.groupList.size() != 1 || p.groupList[0] < 1 ||
            p.groupList[0] > GroupedRotateQuant::MAX_GROUP_NUM) {
            return Fail(error, "group_list must have statically known shape [E], E in [1,1024]");
        }
        p.groupNum = static_cast<size_t>(p.groupList[0]);
    } else if (p.x.size() == 1) {
        return Fail(error, "single M-grouped x requires group_list");
    }
    const auto outNum = p.attrs.splitItem < 2 ? p.groupNum : 1;
    if ((p.x.size() != 1 && p.x.size() != p.groupNum) ||
        (p.rotation.size() != 1 && p.rotation.size() != p.groupNum) ||
        p.yNum != outNum || p.scaleNum != outNum) {
        return Fail(error, "TensorList lengths do not match E and split_item");
    }
    if (p.hasAlpha && (p.alpha.size() != 1 || !ValidShape(p.alpha, symbolic) ||
                      !Compatible(p.alpha[0], 1))) {
        return Fail(error, "alpha must have shape [1]");
    }
    p.runtimeM = p.attrs.groupType == 0 && p.x.size() == 1 && p.hasGroupList;
    return true;
}

inline bool ValidateInputs(InputParams &p, bool symbolic, std::string &error)
{
    if (!CheckAttributes(p.attrs, error) || !CheckLists(p, symbolic, error)) {
        return false;
    }
    p.totalM = 0;
    p.maxM = 0;
    p.totalElements = 0;
    int64_t commonN = -1;
    for (const auto &x : p.x) {
        if (x.empty() || x.size() > 7 || !ValidShape(x, symbolic) ||
            (p.attrs.groupType == 0 && x.size() != 2) ||
            (p.attrs.axis != -1 && p.attrs.axis != static_cast<int64_t>(x.size()) - 1)) {
            return Fail(error, "x rank/dimensions/axis are invalid; M grouping requires rank 2");
        }
        if (p.attrs.groupType == 0 && !Compatible(commonN, x.back())) {
            return Fail(error, "M-grouped tensors must have the same N");
        }
        if (x.back() >= 0) {
            commonN = x.back();
        }
        int64_t m;
        int64_t elements;
        if (!FlattenM(x, m) || !Add(p.totalM, m, p.totalM) || !Multiply(m, x.back(), elements) ||
            !Add(p.totalElements, elements, p.totalElements)) {
            return Fail(error, "input shape arithmetic overflows int64");
        }
        p.maxM = (p.maxM < 0 || m < 0) ? -1 : std::max(p.maxM, m);
    }
    if (p.totalElements > std::numeric_limits<int64_t>::max() / 2 ||
        (p.maxM >= 0 && p.attrs.expectedM > p.maxM)) {
        return Fail(error, "input byte size overflows or expected M exceeds input capacity");
    }
    for (size_t i = 0; i < p.groupNum; ++i) {
        const auto &rot = p.rotation[p.rotation.size() == 1 ? 0 : i];
        const auto &x = p.x[p.x.size() == 1 ? 0 : i];
        if ((rot.size() != 2 && rot.size() != 3) || !ValidShape(rot, symbolic)) {
            return Fail(error, "rotation must have rank 2 or 3");
        }
        const int64_t r = rot.back();
        const int64_t n = x.back();
        if ((r != 32 && r != 64 && r != 128 && !(symbolic && r == -1)) ||
            !Compatible(rot[rot.size() - 2], r) || (r > 0 && n >= 0 && n % r != 0) ||
            (rot.size() == 3 && r > 0 && n >= 0 && !Compatible(rot[0], n / r))) {
            return Fail(error, "rotation must be [R,R] or [N/R,R,R], R=32/64/128");
        }
        int64_t rotElements = 1;
        for (auto d : rot) {
            if (!Multiply(rotElements, d, rotElements)) {
                return Fail(error, "rotation shape arithmetic overflows int64");
            }
        }
        if (rotElements > std::numeric_limits<int64_t>::max() / 2) {
            return Fail(error, "rotation byte size overflows int64");
        }
    }
    return true;
}

// Symbolic inference never inspects group_list device contents. For split_item=1
// with multiple groups, output rows remain -1 until concrete output metadata exists.
inline std::vector<Shape> InferY(const InputParams &p)
{
    if (p.attrs.groupType == -1 || p.x.size() == p.yNum) {
        return p.x;
    }
    if (p.yNum == 1) {
        return {{p.totalM, p.x[0].back()}};
    }
    return std::vector<Shape>(p.yNum, Shape{-1, p.x[0].back()});
}

inline bool ValidateOutputs(const InputParams &p, const std::vector<Shape> &y,
                            const std::vector<Shape> &scale, std::string &error)
{
    if (y.size() != p.yNum || scale.size() != p.scaleNum) {
        return Fail(error, "output list lengths are inconsistent");
    }
    const auto expected = InferY(p);
    int64_t outputM = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        if (!ValidShape(y[i], false) || y[i].size() != expected[i].size()) {
            return Fail(error, "tiling requires concrete output shapes of the inferred rank");
        }
        for (size_t d = 0; d < y[i].size(); ++d) {
            if (!Compatible(y[i][d], expected[i][d])) {
                return Fail(error, "y shape differs from inferred shape");
            }
        }
        if (scale[i] != ScaleShape(y[i])) {
            return Fail(error, "scale shape must be [...,ceil(N/64),2]");
        }
        if (p.attrs.groupType == 0 && !Add(outputM, y[i][0], outputM)) {
            return Fail(error, "sum of output rows overflows int64");
        }
    }
    if (p.attrs.groupType == 0 && outputM != p.totalM) {
        return Fail(error, "sum of output rows must equal total input rows");
    }
    return true;
}
} // namespace optiling::grq
#endif
