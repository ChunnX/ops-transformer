/**
 * @file tensor.h
 *
 * Copyright (C) 2024. Tsingmao Intelligence(Beijing) Technologies Co., Ltd. All rights reserved.
 *
 * This program ...
 */

#ifndef MI_TENSOR_H
#define MI_TENSOR_H
#include <string>
#include <cstdint>

class Tensor {
   public:
    Tensor(size_t capacity);
    ~Tensor();
    inline uint8_t *device() { return device_; }
    inline uint8_t *host() { return host_; }
    inline size_t size() { return size_; }
    inline bool SetSize(size_t size) {
        size_ = size;
        return true;
    };

    bool FromFile(const std::string &fileName = "NONE");
    bool ToFile(const std::string &fileName = "NONE");

    // void SetValue(int32_t value);
    // void SetValue(float value);
    // void SetValue(half value);

    // void SetVector(const std::vector<int64_t> &data);
    // void SetVector(const std::vector<float> &data);

   private:
    size_t capacity_, size_;
    uint8_t *host_, *device_;
};

#endif  // MI_TENSOR
