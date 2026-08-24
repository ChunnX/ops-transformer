#ifndef AICPU_STUB_CPU_TENSOR_SHAPE_H
#define AICPU_STUB_CPU_TENSOR_SHAPE_H

#include <vector>
#include <memory>
#include <cstdint>
#include "cpu_types.h"

namespace aicpu {

class TensorShape {
public:
    TensorShape() = default;
    explicit TensorShape(const std::vector<int64_t> &dims) : dims_(dims) {}
    ~TensorShape() = default;

    void SetFormat(Format format) { format_ = format; }
    Format GetFormat() const { return format_; }

    void SetDimSizes(const std::vector<int64_t> &dims) { dims_ = dims; }
    std::vector<int64_t> GetDimSizes() const { return dims_; }

    int64_t GetDimSize(int32_t index) const {
        if (index < 0 || index >= static_cast<int32_t>(dims_.size())) {
            return -1;
        }
        return dims_[index];
    }

    int32_t GetDims() const { return static_cast<int32_t>(dims_.size()); }

    int64_t NumElements() const {
        if (dims_.empty()) return 0;
        int64_t n = 1;
        for (auto d : dims_) { n *= d; }
        return n;
    }

    bool GetUnknownRank() const { return unknownRank_; }
    void SetUnknownRank(bool unknownRank) { unknownRank_ = unknownRank; }

private:
    std::vector<int64_t> dims_;
    Format format_ = FORMAT_ND;
    bool unknownRank_ = false;
};

} // namespace aicpu

#endif // AICPU_STUB_CPU_TENSOR_SHAPE_H
