#ifndef AICPU_STUB_CPU_CONTEXT_H
#define AICPU_STUB_CPU_CONTEXT_H

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <cstdint>

#include "cpu_types.h"
#include "cpu_tensor.h"
#include "cpu_attr_value.h"

namespace aicpu {

class CpuKernelContext {
public:
    explicit CpuKernelContext(DeviceType type = HOST) : deviceType_(type) {}
    ~CpuKernelContext() = default;

    // Input/Output management
    void AddInput(std::shared_ptr<Tensor> tensor) {
        inputs_.push_back(tensor);
    }

    void AddOutput(std::shared_ptr<Tensor> tensor) {
        outputs_.push_back(tensor);
    }

    Tensor *Input(uint32_t index) const {
        if (index < inputs_.size()) {
            return inputs_[index].get();
        }
        return nullptr;
    }

    Tensor *Output(uint32_t index) const {
        if (index < outputs_.size()) {
            return outputs_[index].get();
        }
        return nullptr;
    }

    uint32_t GetInputsSize() const { return static_cast<uint32_t>(inputs_.size()); }
    uint32_t GetOutputsSize() const { return static_cast<uint32_t>(outputs_.size()); }

    // Attribute management
    void SetAttr(const std::string &name, std::shared_ptr<AttrValue> attr) {
        attrs_[name] = attr;
    }

    AttrValue *GetAttr(std::string name) const {
        auto it = attrs_.find(name);
        if (it != attrs_.end()) {
            return it->second.get();
        }
        return nullptr;
    }

    std::string GetOpType() const { return opType_; }
    void SetOpType(const std::string &opType) { opType_ = opType; }

private:
    DeviceType deviceType_;
    std::string opType_;
    std::vector<std::shared_ptr<Tensor>> inputs_;
    std::vector<std::shared_ptr<Tensor>> outputs_;
    std::unordered_map<std::string, std::shared_ptr<AttrValue>> attrs_;
};

} // namespace aicpu

#endif // AICPU_STUB_CPU_CONTEXT_H
