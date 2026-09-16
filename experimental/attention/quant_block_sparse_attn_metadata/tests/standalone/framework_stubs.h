// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
// See LICENSE in the root of the software repository for the full text of the License.
#pragma once

// Minimal framework containers for CPU-only tests. Scheduling, validation,
// metadata serialization and device address helpers come from production files.
#include <cstdint>
#include <map>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#define KERNEL_LOG_ERROR(...) ((void)0)
#define OP_LOGE(...) ((void)0)
#define REGISTER_CPU_KERNEL(...)
#define CHECK_RET(condition, value) do { if (!(condition)) { return value; } } while (false)

namespace aicpu {
enum DataType { DT_INT32, DT_INT64, DT_INT16, DT_UINT32, DT_UINT64, DT_UINT16, DT_FLOAT };
struct TensorShape {
    std::vector<int64_t> dims;
    int32_t GetDims() const { return static_cast<int32_t>(dims.size()); }
    size_t GetDimNum() const { return dims.size(); }
    int64_t GetDimSize(size_t i) const { return dims.at(i); }
    int64_t GetDim(size_t i) const { return dims.at(i); }
};
struct Tensor {
    TensorShape shape;
    void *data;
    DataType dtype = DT_INT32;
    Tensor(std::vector<int64_t> dims, void *ptr) : shape{dims}, data(ptr) {}
    TensorShape *GetTensorShape() const { return const_cast<TensorShape *>(&shape); }
    const TensorShape &GetViewShape() const { return shape; }
    void *GetData() const { return data; }
    DataType GetDataType() const { return dtype; }
    int64_t NumElements() const {
        return std::accumulate(shape.dims.begin(), shape.dims.end(), int64_t{1}, std::multiplies<int64_t>());
    }
};
struct AttrValue {
    int64_t number = 0;
    std::string text;
    AttrValue() = default;
    AttrValue(int64_t value) : number(value) {}
    AttrValue(const char *value) : text(value) {}
    int64_t GetInt() const { return number; }
    std::string GetString() const { return text; }
    bool GetBool() const { return number != 0; }
};
struct CpuKernelContext {
    std::vector<Tensor *> inputs;
    std::vector<Tensor *> outputs;
    std::map<std::string, AttrValue> attrs;
    Tensor *Input(uint32_t i) { return inputs.at(i); }
    Tensor *Output(uint32_t i) { return outputs.at(i); }
    AttrValue *GetAttr(const std::string &key) {
        auto it = attrs.find(key);
        return it == attrs.end() ? nullptr : &it->second;
    }
};
class CpuKernel {
public:
    virtual ~CpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext &) = 0;
};
} // namespace aicpu

using aclTensor = aicpu::Tensor;
using aclnnStatus = int;
constexpr int ACLNN_SUCCESS = 0;
constexpr int ACLNN_ERR_PARAM_INVALID = 1;
constexpr int ACLNN_ERR_RUNTIME_ERROR = 2;
constexpr auto ACL_INT32 = aicpu::DT_INT32;
