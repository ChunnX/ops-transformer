#ifndef AICPU_STUB_CPU_TENSOR_H
#define AICPU_STUB_CPU_TENSOR_H

#include <memory>
#include <cstdint>
#include "cpu_tensor_shape.h"

namespace aicpu {

class Tensor {
public:
    Tensor() = default;
    ~Tensor() = default;

    bool SetTensorShape(const TensorShape *shape) {
        if (shape) {
            shape_ = std::make_shared<TensorShape>(*shape);
            return true;
        }
        return false;
    }

    std::shared_ptr<TensorShape> GetTensorShape() const {
        return shape_;
    }

    void SetDataType(DataType type) { dataType_ = type; }
    DataType GetDataType() const { return dataType_; }

    void SetData(void *addr) { data_ = addr; }
    void *GetData() const { return data_; }

    void SetDataSize(uint64_t size) { dataSize_ = size; }
    uint64_t GetDataSize() const { return dataSize_; }

    int64_t CalcDataSizeByShape() const {
        if (!shape_) return 0;
        return shape_->NumElements();
    }

    int64_t NumElements() const {
        if (!shape_) return 0;
        return shape_->NumElements();
    }

private:
    std::shared_ptr<TensorShape> shape_;
    DataType dataType_ = DT_UNDEFINED;
    void *data_ = nullptr;
    uint64_t dataSize_ = 0;
};

} // namespace aicpu

#endif // AICPU_STUB_CPU_TENSOR_H
