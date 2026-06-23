#ifndef AICPU_STUB_CPU_ATTR_VALUE_H
#define AICPU_STUB_CPU_ATTR_VALUE_H

#include <string>
#include <vector>
#include <cstdint>
#include "cpu_tensor.h"
#include "cpu_tensor_shape.h"

namespace aicpu {

class AttrValue {
public:
    AttrValue() = default;
    ~AttrValue() = default;

    // String
    std::string GetString() const { return strVal_; }
    void SetString(const std::string &s) { strVal_ = s; }

    // Int
    int64_t GetInt() const { return intVal_; }
    void SetInt(int64_t i) { intVal_ = i; }

    // Bool
    bool GetBool() const { return boolVal_; }
    void SetBool(bool b) { boolVal_ = b; }

    // Float
    float GetFloat() const { return floatVal_; }
    void SetFloat(float f) { floatVal_ = f; }

    // List Int
    std::vector<int64_t> GetListInt() const { return listInt_; }
    void SetListInt(const std::vector<int64_t> &v) { listInt_ = v; }

    // List String
    std::vector<std::string> GetListString() const { return listStr_; }
    void SetListString(const std::vector<std::string> &v) { listStr_ = v; }

private:
    int64_t intVal_ = 0;
    std::string strVal_;
    bool boolVal_ = false;
    float floatVal_ = 0.0f;
    std::vector<int64_t> listInt_;
    std::vector<std::string> listStr_;
};

} // namespace aicpu

#endif // AICPU_STUB_CPU_ATTR_VALUE_H
