#ifndef SAGE_ATTENTION2_DEF_H
#define SAGE_ATTENTION2_DEF_H

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <initializer_list>

#if defined(__has_include)
#if __has_include("graph/operator_reg.h")
#include "graph/operator_reg.h"
#define SAGE_HAS_GRAPH_OPERATOR_REG 1
#endif
#if __has_include("register/op_def_registry.h")
#include "register/op_def_registry.h"
#define SAGE_HAS_OP_DEF_REGISTRY 1
#endif
#if __has_include("register/op_impl_registry.h")
#include "register/op_impl_registry.h"
#define SAGE_HAS_OP_IMPL_REGISTRY 1
#endif
#endif

#ifndef SAGE_HAS_GRAPH_OPERATOR_REG

namespace ge {

enum DataType {
    DT_UNDEFINED = 0,
    DT_FLOAT = 1,
    DT_FLOAT16 = 2,
    DT_INT8 = 3,
    DT_INT32 = 4,
    DT_UINT8 = 5,
    DT_BOOL = 6,
    DT_DOUBLE = 7,
    DT_BF16 = 27
};

enum Format {
    FORMAT_ND = 0,
    FORMAT_NCHW = 1,
    FORMAT_NHWC = 2
};

enum graphStatus {
    GRAPH_SUCCESS = 0,
    GRAPH_FAILED = 1
};

class TensorType {
public:
    TensorType() = default;
    explicit TensorType(std::initializer_list<DataType> types) : types_(types) {}
    const std::vector<DataType>& GetTypes() const { return types_; }
private:
    std::vector<DataType> types_;
};

struct AttrValueMock {
    std::string type;
    std::string str_val;
    bool bool_val{false};
    float float_val{0.0f};
    int64_t int_val{0};
};

class OperatorRegMock {
public:
    explicit OperatorRegMock(const std::string &name) : op_name_(name) {}

    OperatorRegMock& AddInput(const std::string &name, const TensorType &type) {
        inputs_[name] = type;
        return *this;
    }

    OperatorRegMock& AddOptionalInput(const std::string &name, const TensorType &type) {
        optional_inputs_[name] = type;
        return *this;
    }

    OperatorRegMock& AddOutput(const std::string &name, const TensorType &type) {
        outputs_[name] = type;
        return *this;
    }

    OperatorRegMock& AddAttr(const std::string &name, const std::string &type, const std::string &val) {
        AttrValueMock attr;
        attr.type = type;
        attr.str_val = val;
        attrs_[name] = attr;
        return *this;
    }

    OperatorRegMock& AddAttr(const std::string &name, const std::string &type, const char *val) {
        return AddAttr(name, type, std::string(val ? val : ""));
    }

    OperatorRegMock& AddAttr(const std::string &name, const std::string &type, bool val) {
        AttrValueMock attr;
        attr.type = type;
        attr.bool_val = val;
        attrs_[name] = attr;
        return *this;
    }

    OperatorRegMock& AddAttr(const std::string &name, const std::string &type, float val) {
        AttrValueMock attr;
        attr.type = type;
        attr.float_val = val;
        attrs_[name] = attr;
        return *this;
    }

    OperatorRegMock& AddAttr(const std::string &name, const std::string &type, double val) {
        return AddAttr(name, type, static_cast<float>(val));
    }

    OperatorRegMock& AddAttr(const std::string &name, const std::string &type, int64_t val) {
        AttrValueMock attr;
        attr.type = type;
        attr.int_val = val;
        attrs_[name] = attr;
        return *this;
    }

    OperatorRegMock& EndFactoryReg() { return *this; }

    const std::string& GetOpName() const { return op_name_; }
    bool HasInput(const std::string &name) const { return inputs_.find(name) != inputs_.end() || optional_inputs_.find(name) != optional_inputs_.end(); }
    bool HasOptionalInput(const std::string &name) const { return optional_inputs_.find(name) != optional_inputs_.end(); }
    bool HasOutput(const std::string &name) const { return outputs_.find(name) != outputs_.end(); }
    bool HasAttr(const std::string &name) const { return attrs_.find(name) != attrs_.end(); }
    const AttrValueMock& GetAttr(const std::string &name) const { return attrs_.at(name); }

private:
    std::string op_name_;
    std::map<std::string, TensorType> inputs_;
    std::map<std::string, TensorType> optional_inputs_;
    std::map<std::string, TensorType> outputs_;
    std::map<std::string, AttrValueMock> attrs_;
};

#define REG_OP(OpType) \
    class OpType : public ::ge::OperatorRegMock { \
    public: \
        OpType() : ::ge::OperatorRegMock(#OpType) {} \
        static OpType& GetInstance() { \
            static OpType instance; \
            return instance; \
        } \
    }; \
    ::ge::OperatorRegMock& Get##OpType##Reg() { \
        return OpType::GetInstance(); \
    } \
    inline ::ge::OperatorRegMock& g_##OpType##_op_reg = OpType::GetInstance()

#define INPUT(name, type) AddInput(#name, type)
#define OPTIONAL_INPUT(name, type) AddOptionalInput(#name, type)
#define OUTPUT(name, type) AddOutput(#name, type)
#define ATTR(name, type, default_val) AddAttr(#name, #type, default_val)
#define OP_END_FACTORY_REG(OpType) EndFactoryReg()

OperatorRegMock& GetSageAttention2Reg();

} // namespace ge

#endif // SAGE_HAS_GRAPH_OPERATOR_REG

#ifndef SAGE_HAS_OP_DEF_REGISTRY

namespace ops {

enum OpParamType {
    REQUIRED,
    OPTIONAL
};

class OpTensorDef {
public:
    OpTensorDef& ParamType(OpParamType pt) { param_type_ = pt; return *this; }
    // 真实 CANN OpDef 使用 DataTypeList/FormatList 列表接口；本地桩同名对齐
    OpTensorDef& DataTypeList(const std::vector<ge::DataType> &dt) { data_types_ = dt; return *this; }
    OpTensorDef& FormatList(const std::vector<ge::Format> &fmt) { formats_ = fmt; return *this; }
    OpTensorDef& AutoContiguous() { auto_contiguous_ = true; return *this; }

    OpParamType GetParamType() const { return param_type_; }

private:
    OpParamType param_type_{REQUIRED};
    std::vector<ge::DataType> data_types_;
    std::vector<ge::Format> formats_;
    bool auto_contiguous_{false};
};

class OpAttrDef {
public:
    OpAttrDef& AttrType(OpParamType pt) { param_type_ = pt; return *this; }
    OpAttrDef& String(const std::string &val = "") { str_val_ = val; return *this; }
    OpAttrDef& Bool(bool val = false) { bool_val_ = val; return *this; }
    OpAttrDef& Float(float val = 0.0f) { float_val_ = val; return *this; }
    OpAttrDef& Int(int64_t val = 0) { int_val_ = val; return *this; }

    OpParamType GetParamType() const { return param_type_; }

private:
    OpParamType param_type_{REQUIRED};
    std::string str_val_;
    bool bool_val_{false};
    float float_val_{0.0f};
    int64_t int_val_{0};
};

class OpAICoreConfig {
public:
    OpAICoreConfig& DynamicCompileStaticFlag(bool val) { dynamic_compile_static_ = val; return *this; }
    OpAICoreConfig& DynamicFormatFlag(bool val) { dynamic_format_ = val; return *this; }
    OpAICoreConfig& DynamicRankSupportFlag(bool val) { dynamic_rank_ = val; return *this; }
    OpAICoreConfig& DynamicShapeSupportFlag(bool val) { dynamic_shape_ = val; return *this; }
    OpAICoreConfig& NeedCheckSupportFlag(bool val) { need_check_ = val; return *this; }
    OpAICoreConfig& PrecisionReduceFlag(bool val) { precision_reduce_ = val; return *this; }
    OpAICoreConfig& ExtendCfgInfo(const std::string &k, const std::string &v) { ext_info_[k] = v; return *this; }

    const std::string& GetExtInfo(const std::string &k) const {
        static const std::string empty;
        auto it = ext_info_.find(k);
        return (it != ext_info_.end()) ? it->second : empty;
    }
private:
    bool dynamic_compile_static_{true};
    bool dynamic_format_{false};
    bool dynamic_rank_{true};
    bool dynamic_shape_{true};
    bool need_check_{false};
    bool precision_reduce_{true};
    std::map<std::string, std::string> ext_info_;
};

class AICoreTargetConfig {
public:
    void AddConfig(const std::string &soc, const OpAICoreConfig &cfg) {
        soc_configs_[soc] = cfg;
    }
    bool HasSoc(const std::string &soc) const {
        return soc_configs_.find(soc) != soc_configs_.end();
    }
    const OpAICoreConfig& GetConfig(const std::string &soc) const {
        return soc_configs_.at(soc);
    }
private:
    std::map<std::string, OpAICoreConfig> soc_configs_;
};

class OpDef {
public:
    explicit OpDef(const char *name) : op_name_(name) {}
    virtual ~OpDef() = default;

    OpTensorDef& Input(const char *name) {
        inputs_[name] = OpTensorDef();
        return inputs_[name];
    }

    OpTensorDef& Output(const char *name) {
        outputs_[name] = OpTensorDef();
        return outputs_[name];
    }

    OpAttrDef& Attr(const char *name) {
        attrs_[name] = OpAttrDef();
        return attrs_[name];
    }

    AICoreTargetConfig& AICore() { return aicore_; }

    const std::string& GetName() const { return op_name_; }
    bool HasInput(const std::string &n) const { return inputs_.find(n) != inputs_.end(); }
    bool HasOutput(const std::string &n) const { return outputs_.find(n) != outputs_.end(); }
    bool HasAttr(const std::string &n) const { return attrs_.find(n) != attrs_.end(); }
    const OpTensorDef& GetInput(const std::string &n) const { return inputs_.at(n); }
    const OpTensorDef& GetOutput(const std::string &n) const { return outputs_.at(n); }
    const OpAttrDef& GetAttr(const std::string &n) const { return attrs_.at(n); }
    const AICoreTargetConfig& GetAICore() const { return aicore_; }

private:
    std::string op_name_;
    std::map<std::string, OpTensorDef> inputs_;
    std::map<std::string, OpTensorDef> outputs_;
    std::map<std::string, OpAttrDef> attrs_;
    AICoreTargetConfig aicore_;
};

class OpDefRegistry {
public:
    static OpDefRegistry& GetInstance() {
        static OpDefRegistry instance;
        return instance;
    }
    void Register(std::shared_ptr<OpDef> op) {
        ops_[op->GetName()] = op;
    }
    std::shared_ptr<OpDef> GetOp(const std::string &name) {
        auto it = ops_.find(name);
        return (it != ops_.end()) ? it->second : nullptr;
    }
private:
    std::map<std::string, std::shared_ptr<OpDef>> ops_;
};

#define OP_ADD(OpType) \
    static const bool g_##OpType##_op_add_reg = []() { \
        ::ops::OpDefRegistry::GetInstance().Register(std::make_shared<OpType>(#OpType)); \
        return true; \
    }()

} // namespace ops

#endif // SAGE_HAS_OP_DEF_REGISTRY

#ifndef SAGE_HAS_OP_IMPL_REGISTRY

namespace gert {

class Shape {
public:
    Shape() = default;
    // 隐式 initializer_list 构造是 `*lseShape = {b, hq, sq}` 等赋值语法所必需
    Shape(std::initializer_list<int64_t> dims) : dims_(dims) {}

    size_t GetDimNum() const { return dims_.size(); }
    int64_t GetDim(size_t idx) const {
        return (idx < dims_.size()) ? dims_[idx] : -1;
    }

    bool operator==(const Shape &other) const { return dims_ == other.dims_; }

private:
    std::vector<int64_t> dims_;
};

class RuntimeAttrs {
public:
    const std::string* GetString(size_t idx) const {
        return (idx < str_list_.size()) ? &str_list_[idx] : nullptr;
    }
    const int64_t* GetInt(size_t idx) const {
        return (idx < int_list_.size()) ? &int_list_[idx] : nullptr;
    }

    void AddString(const std::string &s) { str_list_.push_back(s); }
    void AddInt(int64_t i) { int_list_.push_back(i); }

private:
    std::vector<std::string> str_list_;
    std::vector<int64_t> int_list_;
};

class InferShapeContext {
public:
    explicit InferShapeContext(const char *node_name = "SageAttention2")
        : node_name_(node_name) {}

    const char* GetNodeName() const { return node_name_.c_str(); }

    void SetInputShape(size_t idx, const Shape *s) {
        if (idx >= input_shapes_.size()) input_shapes_.resize(idx + 1, nullptr);
        input_shapes_[idx] = s;
    }

    void SetOutputShape(size_t idx, Shape *s) {
        if (idx >= output_shapes_.size()) output_shapes_.resize(idx + 1, nullptr);
        output_shapes_[idx] = s;
    }

    void SetAttrs(const RuntimeAttrs *attrs) { attrs_ = attrs; }

    const Shape* GetInputShape(size_t idx) const {
        return (idx < input_shapes_.size()) ? input_shapes_[idx] : nullptr;
    }

    Shape* GetOutputShape(size_t idx) {
        return (idx < output_shapes_.size()) ? output_shapes_[idx] : nullptr;
    }

    const RuntimeAttrs* GetAttrs() const { return attrs_; }

private:
    std::string node_name_;
    std::vector<const Shape*> input_shapes_;
    std::vector<Shape*> output_shapes_;
    const RuntimeAttrs *attrs_{nullptr};
};

class InferDataTypeContext {
public:
    explicit InferDataTypeContext(const char *node_name = "SageAttention2")
        : node_name_(node_name) {}

    const char* GetNodeName() const { return node_name_.c_str(); }

    void SetInputDataType(size_t idx, ge::DataType dt) {
        if (idx >= input_dtypes_.size()) input_dtypes_.resize(idx + 1, ge::DT_UNDEFINED);
        input_dtypes_[idx] = dt;
    }

    ge::DataType GetInputDataType(size_t idx) const {
        return (idx < input_dtypes_.size()) ? input_dtypes_[idx] : ge::DT_UNDEFINED;
    }

    void SetOutputDataType(size_t idx, ge::DataType dt) {
        if (idx >= output_dtypes_.size()) output_dtypes_.resize(idx + 1, ge::DT_UNDEFINED);
        output_dtypes_[idx] = dt;
    }

    ge::DataType GetOutputDataType(size_t idx) const {
        return (idx < output_dtypes_.size()) ? output_dtypes_[idx] : ge::DT_UNDEFINED;
    }

    void SetAttrs(const RuntimeAttrs *attrs) { attrs_ = attrs; }
    const RuntimeAttrs* GetAttrs() const { return attrs_; }

private:
    std::string node_name_;
    std::vector<ge::DataType> input_dtypes_;
    std::vector<ge::DataType> output_dtypes_;
    const RuntimeAttrs *attrs_{nullptr};
};

} // namespace gert

namespace ops {

using InferShapeFunc = ge::graphStatus (*)(gert::InferShapeContext *context);
using InferDataTypeFunc = ge::graphStatus (*)(gert::InferDataTypeContext *context);

class OpImplRegistryMock {
public:
    static OpImplRegistryMock& GetInstance(const std::string &name) {
        static std::map<std::string, OpImplRegistryMock> instances;
        return instances[name];
    }

    OpImplRegistryMock& InferShape(InferShapeFunc func) {
        infer_shape_ = func;
        return *this;
    }

    OpImplRegistryMock& InferDataType(InferDataTypeFunc func) {
        infer_data_type_ = func;
        return *this;
    }

    InferShapeFunc GetInferShape() const { return infer_shape_; }
    InferDataTypeFunc GetInferDataType() const { return infer_data_type_; }

private:
    InferShapeFunc infer_shape_{nullptr};
    InferDataTypeFunc infer_data_type_{nullptr};
};

#define IMPL_OP_INFERSHAPE(OpType) \
    inline ::ops::OpImplRegistryMock& g_##OpType##_impl_reg = \
        ::ops::OpImplRegistryMock::GetInstance(#OpType)

} // namespace ops

#endif // SAGE_HAS_OP_IMPL_REGISTRY

namespace ops {

/**
 * SageAttention2 参数索引（唯一定义点，严格对齐 sage_attention2_def.cpp 的
 * INPUT/OPTIONAL_INPUT/OUTPUT/ATTR 声明顺序）；op_host 各 TU 一律经此引用，
 * 禁止再各自维护本地索引表。
 */
namespace sage_attn2_idx {
constexpr size_t IN_QUERY = 0;
constexpr size_t IN_KEY = 1;
constexpr size_t IN_VALUE = 2;
constexpr size_t IN_KEY_MEAN = 3;
constexpr size_t IN_LSE_CORRECTION = 4;

constexpr size_t OUT_OUTPUT = 0;
constexpr size_t OUT_LSE = 1;

constexpr size_t ATTR_TENSOR_LAYOUT = 0;
constexpr size_t ATTR_IS_CAUSAL = 1;
constexpr size_t ATTR_QK_QUANT_GRAN = 2;
constexpr size_t ATTR_SM_SCALE = 3;
constexpr size_t ATTR_PV_ACCUM_DTYPE = 4;
constexpr size_t ATTR_SMOOTH_K = 5;
constexpr size_t ATTR_SMOOTH_V = 6;
} // namespace sage_attn2_idx

/**
 * RuntimeAttrs 字符串属性读取封装（唯一定义点）：
 * 真实 CANN gert::RuntimeAttrs 为 GetStr(idx) -> const char*，
 * 本地桩为 GetString(idx) -> const std::string*。
 */
inline const char *ReadStringAttr(const gert::RuntimeAttrs *attrs, size_t idx)
{
#if defined(SAGE_HAS_OP_IMPL_REGISTRY)
    return attrs->GetStr(idx);
#else
    const std::string *str_val = attrs->GetString(idx);
    return (str_val != nullptr) ? str_val->c_str() : nullptr;
#endif
}

} // namespace ops

#endif // SAGE_ATTENTION2_DEF_H
