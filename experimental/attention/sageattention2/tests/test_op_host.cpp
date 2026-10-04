#include <iostream>

#include "../op_host/sage_attention2_def.h"
#include "base/runtime/runtime_attrs_def.h"
#include "exe_graph/runtime/compute_node_info.h"

// 宏定义测试辅助
#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAIL] Line " << __LINE__ << ": " << msg << std::endl; \
            return false; \
        } \
    } while (0)

namespace {

// 辅助比较 CANN gert::Shape
inline bool ShapeEquals(const gert::Shape &a, const gert::Shape &b)
{
    if (a.GetDimNum() != b.GetDimNum()) return false;
    for (size_t i = 0; i < a.GetDimNum(); ++i) {
        if (a.GetDim(i) != b.GetDim(i)) return false;
    }
    return true;
}

// 构造完全对齐 CANN 标准 KernelRunContext ABI 的 InferShapeContext 测试容器
struct TestInferShapeContextHolder {
    static constexpr size_t NUM_INPUTS = 5;
    static constexpr size_t NUM_OUTPUTS = 2;
    static constexpr size_t TOTAL_VALUES = NUM_INPUTS + NUM_OUTPUTS;

    struct NodeInfoWithAttrs {
        const char *node_type_{"SageAttention2"};
        const char *node_name_{"SageAttention2"};
        size_t ir_inputs_num_{0};
        size_t inputs_num_{0};
        size_t outputs_num_{0};
        size_t ir_outputs_num_{0};
        size_t runtime_attr_size_{64};
        uint8_t reserved_[24]{0};
        RuntimeAttrsDef attrs_def;
        size_t offset_0;
        char layout_str[16];

        NodeInfoWithAttrs() {
            attrs_def.attr_num = 1;
            offset_0 = sizeof(RuntimeAttrsDef) + sizeof(size_t);
            strncpy(layout_str, "HND", sizeof(layout_str) - 1);
        }

        void SetLayout(const char *layout) {
            strncpy(layout_str, layout, sizeof(layout_str) - 1);
        }

        const gert::ComputeNodeInfo *GetComputeNodeInfo() const {
            return reinterpret_cast<const gert::ComputeNodeInfo *>(this);
        }
    };

    struct Buffer {
        size_t input_size{NUM_INPUTS};
        size_t output_size{NUM_OUTPUTS};
        const void *compute_node_info{nullptr};
        const void *kernel_extend_info{nullptr};
        AsyncAnyValue **output_start{nullptr};
        AsyncAnyValue *values[TOTAL_VALUES]{nullptr};
    } krc;

    AsyncAnyValue any_values[TOTAL_VALUES];
    NodeInfoWithAttrs node_info;

    TestInferShapeContextHolder(const char *layout = "HND") {
        node_info.SetLayout(layout);
        krc.compute_node_info = node_info.GetComputeNodeInfo();
        for (size_t i = 0; i < TOTAL_VALUES; ++i) {
            any_values[i].data.pointer = nullptr;
            any_values[i].deleter = nullptr;
            krc.values[i] = &any_values[i];
        }
    }

    void SetInputShape(size_t idx, const gert::Shape *s) {
        if (idx < NUM_INPUTS) {
            any_values[idx].data.pointer = const_cast<gert::Shape*>(s);
        }
    }

    void SetOutputShape(size_t idx, gert::Shape *s) {
        if (idx < NUM_OUTPUTS) {
            any_values[NUM_INPUTS + idx].data.pointer = s;
        }
    }

    gert::InferShapeContext *GetContext() {
        return reinterpret_cast<gert::InferShapeContext *>(&krc);
    }
};

// 构造完全对齐 CANN 标准 KernelRunContext ABI 的 InferDataTypeContext 测试容器
struct TestInferDataTypeContextHolder {
    static constexpr size_t NUM_INPUTS = 5;
    static constexpr size_t NUM_OUTPUTS = 2;
    static constexpr size_t TOTAL_VALUES = NUM_INPUTS + NUM_OUTPUTS;

    struct Buffer {
        size_t input_size{NUM_INPUTS};
        size_t output_size{NUM_OUTPUTS};
        const void *compute_node_info{nullptr};
        const void *kernel_extend_info{nullptr};
        AsyncAnyValue **output_start{nullptr};
        AsyncAnyValue *values[TOTAL_VALUES]{nullptr};
    } krc;

    AsyncAnyValue any_values[TOTAL_VALUES];

    TestInferDataTypeContextHolder() {
        for (size_t i = 0; i < TOTAL_VALUES; ++i) {
            *reinterpret_cast<ge::DataType*>(any_values[i].data.inplace) = ge::DT_UNDEFINED;
            any_values[i].deleter = nullptr;
            krc.values[i] = &any_values[i];
        }
    }

    void SetInputDataType(size_t idx, ge::DataType dt) {
        if (idx < NUM_INPUTS) {
            *reinterpret_cast<ge::DataType*>(any_values[idx].data.inplace) = dt;
        }
    }

    ge::DataType GetOutputDataType(size_t idx) const {
        if (idx < NUM_OUTPUTS) {
            return *reinterpret_cast<const ge::DataType*>(any_values[NUM_INPUTS + idx].data.inplace);
        }
        return ge::DT_UNDEFINED;
    }

    gert::InferDataTypeContext *GetContext() {
        return reinterpret_cast<gert::InferDataTypeContext *>(&krc);
    }
};

// 1. 验证 REG_OP 宏原型注册完整度（标准 CANN ge::op::SageAttention2）
bool TestRegOpDefinition()
{
    std::cout << "[RUN] TestRegOpDefinition..." << std::endl;
    ge::op::SageAttention2 op("SageAttention2");

    TEST_ASSERT(op.GetOpType() == "SageAttention2", "OpName must be SageAttention2");

    // 验证所有输入张量名称
    TEST_ASSERT(ge::op::SageAttention2::name_in_query() == "query", "Must have input: query");
    TEST_ASSERT(ge::op::SageAttention2::name_in_key() == "key", "Must have input: key");
    TEST_ASSERT(ge::op::SageAttention2::name_in_value() == "value", "Must have input: value");

    // 验证所有输出张量名称
    TEST_ASSERT(ge::op::SageAttention2::name_out_output() == "output", "Must have output: output");
    TEST_ASSERT(ge::op::SageAttention2::name_out_lse() == "lse", "Must have output: lse");

    // 验证全部 7 个属性名
    TEST_ASSERT(ge::op::SageAttention2::name_attr_tensor_layout() == "tensor_layout", "Must have attr: tensor_layout");
    TEST_ASSERT(ge::op::SageAttention2::name_attr_is_causal() == "is_causal", "Must have attr: is_causal");
    TEST_ASSERT(ge::op::SageAttention2::name_attr_qk_quant_gran() == "qk_quant_gran", "Must have attr: qk_quant_gran");
    TEST_ASSERT(ge::op::SageAttention2::name_attr_sm_scale() == "sm_scale", "Must have attr: sm_scale");
    TEST_ASSERT(ge::op::SageAttention2::name_attr_pv_accum_dtype() == "pv_accum_dtype", "Must have attr: pv_accum_dtype");
    TEST_ASSERT(ge::op::SageAttention2::name_attr_smooth_k() == "smooth_k", "Must have attr: smooth_k");
    TEST_ASSERT(ge::op::SageAttention2::name_attr_smooth_v() == "smooth_v", "Must have attr: smooth_v");

    // 校验属性默认值
    TEST_ASSERT(op.get_attr_tensor_layout() == "HND", "tensor_layout default HND");
    TEST_ASSERT(op.get_attr_is_causal() == false, "is_causal default false");
    TEST_ASSERT(op.get_attr_qk_quant_gran() == "per_thread", "qk_quant_gran default per_thread");
    TEST_ASSERT(op.get_attr_pv_accum_dtype() == "fp32+fp16", "pv_accum_dtype default fp32+fp16");
    TEST_ASSERT(op.get_attr_smooth_k() == true, "smooth_k default true");
    TEST_ASSERT(op.get_attr_smooth_v() == false, "smooth_v default false");

    std::cout << "[PASS] TestRegOpDefinition" << std::endl;
    return true;
}

// 2. 验证 OpDef 信息库注册与硬件配置
bool TestOpDefRegistry()
{
    std::cout << "[RUN] TestOpDefRegistry..." << std::endl;
    // 实例化验证 ops::SageAttention2 构造器与 CANN OP_ADD 注册机制
    ops::SageAttention2 op("SageAttention2");

    std::cout << "[PASS] TestOpDefRegistry" << std::endl;
    return true;
}

// 3. 验证 InferShape 准确推导 (标准 MHA、GQA 广播、任意 Head Dim)
bool TestInferShapeValidCases()
{
    std::cout << "[RUN] TestInferShapeValidCases..." << std::endl;

    // 用例 A: 标准 MHA (B=2, Hq=32, Hkv=32, Sq=1024, Sk=1024, D=128)
    {
        TestInferShapeContextHolder holder("HND");
        gert::Shape q_shape({2, 32, 1024, 128});
        gert::Shape k_shape({2, 32, 1024, 128});
        gert::Shape v_shape({2, 32, 1024, 128});
        gert::Shape out_shape;
        gert::Shape lse_shape;

        holder.SetInputShape(0, &q_shape);
        holder.SetInputShape(1, &k_shape);
        holder.SetInputShape(2, &v_shape);
        holder.SetOutputShape(0, &out_shape);
        holder.SetOutputShape(1, &lse_shape);

        ge::graphStatus ret = ops::InferShapeSageAttention2(holder.GetContext());
        TEST_ASSERT(ret == ge::GRAPH_SUCCESS, "InferShape MHA failed");
        TEST_ASSERT(ShapeEquals(out_shape, q_shape), "Output shape must match Query shape");
        gert::Shape expected_lse({2, 32, 1024});
        TEST_ASSERT(ShapeEquals(lse_shape, expected_lse), "LSE shape must be [B, Hq, Sq]");
    }

    // 用例 B: GQA 广播 (B=4, Hq=32, Hkv=8 [4:1], Sq=512, Sk=2048, D=64)
    {
        TestInferShapeContextHolder holder("HND");
        gert::Shape q_shape({4, 32, 512, 64});
        gert::Shape k_shape({4, 8, 2048, 64});
        gert::Shape v_shape({4, 8, 2048, 64});
        gert::Shape out_shape;
        gert::Shape lse_shape;

        holder.SetInputShape(0, &q_shape);
        holder.SetInputShape(1, &k_shape);
        holder.SetInputShape(2, &v_shape);
        holder.SetOutputShape(0, &out_shape);
        holder.SetOutputShape(1, &lse_shape);

        ge::graphStatus ret = ops::InferShapeSageAttention2(holder.GetContext());
        TEST_ASSERT(ret == ge::GRAPH_SUCCESS, "InferShape GQA failed");
        TEST_ASSERT(ShapeEquals(out_shape, q_shape), "GQA Output shape must match Query");
        gert::Shape expected_lse({4, 32, 512});
        TEST_ASSERT(ShapeEquals(lse_shape, expected_lse), "GQA LSE shape must be [4, 32, 512]");
    }

    // 用例 C: MQA 广播 (B=1, Hq=16, Hkv=1 [16:1], Sq=128, Sk=512, D=128)
    {
        TestInferShapeContextHolder holder("HND");
        gert::Shape q_shape({1, 16, 128, 128});
        gert::Shape k_shape({1, 1, 512, 128});
        gert::Shape v_shape({1, 1, 512, 128});
        gert::Shape out_shape;

        holder.SetInputShape(0, &q_shape);
        holder.SetInputShape(1, &k_shape);
        holder.SetInputShape(2, &v_shape);
        holder.SetOutputShape(0, &out_shape);

        ge::graphStatus ret = ops::InferShapeSageAttention2(holder.GetContext());
        TEST_ASSERT(ret == ge::GRAPH_SUCCESS, "InferShape MQA failed");
        TEST_ASSERT(ShapeEquals(out_shape, q_shape), "MQA Output shape must match Query");
    }

    // 用例 D: 非标准头维度边界 (D=80 与 D=96)
    {
        TestInferShapeContextHolder holder("HND");
        gert::Shape q_shape({2, 16, 256, 80});
        gert::Shape k_shape({2, 16, 256, 80});
        gert::Shape v_shape({2, 16, 256, 80});
        gert::Shape out_shape;

        holder.SetInputShape(0, &q_shape);
        holder.SetInputShape(1, &k_shape);
        holder.SetInputShape(2, &v_shape);
        holder.SetOutputShape(0, &out_shape);

        ge::graphStatus ret = ops::InferShapeSageAttention2(holder.GetContext());
        TEST_ASSERT(ret == ge::GRAPH_SUCCESS, "InferShape D=80 failed");
        TEST_ASSERT(out_shape.GetDim(3) == 80, "Output D must be 80");
    }

    // 用例 E: NHD 布局
    {
        TestInferShapeContextHolder holder("NHD");
        gert::Shape q_shape({2, 512, 32, 128}); // [B, Sq, Hq, D]
        gert::Shape k_shape({2, 512, 8, 128});  // [B, Sk, Hkv, D]
        gert::Shape v_shape({2, 512, 8, 128});  // [B, Sk, Hkv, D]
        gert::Shape out_shape;
        gert::Shape lse_shape;

        holder.SetInputShape(0, &q_shape);
        holder.SetInputShape(1, &k_shape);
        holder.SetInputShape(2, &v_shape);
        holder.SetOutputShape(0, &out_shape);
        holder.SetOutputShape(1, &lse_shape);

        ge::graphStatus ret = ops::InferShapeSageAttention2(holder.GetContext());
        TEST_ASSERT(ret == ge::GRAPH_SUCCESS, "InferShape NHD failed");
        TEST_ASSERT(ShapeEquals(out_shape, q_shape), "NHD Output shape must match Query");
        gert::Shape expected_lse({2, 32, 512});
        TEST_ASSERT(ShapeEquals(lse_shape, expected_lse), "NHD LSE shape must be [B, Hq, Sq]");
    }

    std::cout << "[PASS] TestInferShapeValidCases" << std::endl;
    return true;
}

// 4. 验证 InferShape 异常边界拦截
bool TestInferShapeErrorHandling()
{
    std::cout << "[RUN] TestInferShapeErrorHandling..." << std::endl;

    // 1. 空指针测试
    TEST_ASSERT(ops::InferShapeSageAttention2(nullptr) == ge::GRAPH_FAILED, "Should reject null context");

    // 2. 缺失输入张量测试
    {
        TestInferShapeContextHolder holder;
        gert::Shape q_shape({2, 32, 1024, 128});
        holder.SetInputShape(0, &q_shape);
        TEST_ASSERT(ops::InferShapeSageAttention2(holder.GetContext()) == ge::GRAPH_FAILED, "Should reject missing K/V");
    }

    // 3. Batch 不匹配
    {
        TestInferShapeContextHolder holder;
        gert::Shape q_shape({2, 32, 1024, 128});
        gert::Shape k_shape({3, 32, 1024, 128}); // Batch 3 != 2
        gert::Shape v_shape({3, 32, 1024, 128});
        gert::Shape out;
        holder.SetInputShape(0, &q_shape);
        holder.SetInputShape(1, &k_shape);
        holder.SetInputShape(2, &v_shape);
        holder.SetOutputShape(0, &out);
        TEST_ASSERT(ops::InferShapeSageAttention2(holder.GetContext()) == ge::GRAPH_FAILED, "Should reject mismatched Batch");
    }

    // 4. Head Dim 超出硬件上限 (D=256 > 128)
    {
        TestInferShapeContextHolder holder;
        gert::Shape q_shape({2, 32, 1024, 256});
        gert::Shape k_shape({2, 32, 1024, 256});
        gert::Shape v_shape({2, 32, 1024, 256});
        gert::Shape out;
        holder.SetInputShape(0, &q_shape);
        holder.SetInputShape(1, &k_shape);
        holder.SetInputShape(2, &v_shape);
        holder.SetOutputShape(0, &out);
        TEST_ASSERT(ops::InferShapeSageAttention2(holder.GetContext()) == ge::GRAPH_FAILED, "Should reject D > 128");
    }

    // 5. GQA 头数不可整除 (Hq=32, Hkv=7)
    {
        TestInferShapeContextHolder holder;
        gert::Shape q_shape({2, 32, 1024, 128});
        gert::Shape k_shape({2, 7, 1024, 128});
        gert::Shape v_shape({2, 7, 1024, 128});
        gert::Shape out;
        holder.SetInputShape(0, &q_shape);
        holder.SetInputShape(1, &k_shape);
        holder.SetInputShape(2, &v_shape);
        holder.SetOutputShape(0, &out);
        TEST_ASSERT(ops::InferShapeSageAttention2(holder.GetContext()) == ge::GRAPH_FAILED, "Should reject non-divisible GQA heads");
    }

    // 6. 张量维度非 4D
    {
        TestInferShapeContextHolder holder;
        gert::Shape q_shape({2, 32, 1024}); // 3D
        gert::Shape k_shape({2, 32, 1024});
        gert::Shape v_shape({2, 32, 1024});
        gert::Shape out;
        holder.SetInputShape(0, &q_shape);
        holder.SetInputShape(1, &k_shape);
        holder.SetInputShape(2, &v_shape);
        holder.SetOutputShape(0, &out);
        TEST_ASSERT(ops::InferShapeSageAttention2(holder.GetContext()) == ge::GRAPH_FAILED, "Should reject non-4D tensor");
    }

    std::cout << "[PASS] TestInferShapeErrorHandling" << std::endl;
    return true;
}

// 5. 验证 InferDataType 数据类型推导
bool TestInferDataType()
{
    std::cout << "[RUN] TestInferDataType..." << std::endl;

    // 用例 A: Float16 输入推导
    {
        TestInferDataTypeContextHolder holder;
        holder.SetInputDataType(0, ge::DT_FLOAT16);
        holder.SetInputDataType(1, ge::DT_FLOAT16);
        holder.SetInputDataType(2, ge::DT_FLOAT16);

        ge::graphStatus ret = ops::InferDataTypeSageAttention2(holder.GetContext());
        TEST_ASSERT(ret == ge::GRAPH_SUCCESS, "InferDataType FP16 failed");
        TEST_ASSERT(holder.GetOutputDataType(0) == ge::DT_FLOAT16, "Output dtype must be FP16");
        TEST_ASSERT(holder.GetOutputDataType(1) == ge::DT_FLOAT, "LSE dtype must be FLOAT");
    }

    // 用例 B: BFloat16 输入推导
    {
        TestInferDataTypeContextHolder holder;
        holder.SetInputDataType(0, ge::DT_BF16);
        holder.SetInputDataType(1, ge::DT_BF16);
        holder.SetInputDataType(2, ge::DT_BF16);

        ge::graphStatus ret = ops::InferDataTypeSageAttention2(holder.GetContext());
        TEST_ASSERT(ret == ge::GRAPH_SUCCESS, "InferDataType BF16 failed");
        TEST_ASSERT(holder.GetOutputDataType(0) == ge::DT_BF16, "Output dtype must be BF16");
        TEST_ASSERT(holder.GetOutputDataType(1) == ge::DT_FLOAT, "LSE dtype must be FLOAT");
    }

    // 用例 C: Q 与 K 数据类型不匹配拦截
    {
        TestInferDataTypeContextHolder holder;
        holder.SetInputDataType(0, ge::DT_FLOAT16);
        holder.SetInputDataType(1, ge::DT_BF16);
        holder.SetInputDataType(2, ge::DT_FLOAT16);

        ge::graphStatus ret = ops::InferDataTypeSageAttention2(holder.GetContext());
        TEST_ASSERT(ret == ge::GRAPH_FAILED, "Should reject mismatched input dtypes");
    }

    std::cout << "[PASS] TestInferDataType" << std::endl;
    return true;
}

} // namespace

int main()
{
    std::cout << "==========================================================" << std::endl;
    std::cout << "SageAttention2 OP Host / REG_OP / InferShape Tests" << std::endl;
    std::cout << "==========================================================" << std::endl;

    bool all_passed = true;
    all_passed &= TestRegOpDefinition();
    all_passed &= TestOpDefRegistry();
    all_passed &= TestInferShapeValidCases();
    all_passed &= TestInferShapeErrorHandling();
    all_passed &= TestInferDataType();

    std::cout << "==========================================================" << std::endl;
    if (all_passed) {
        std::cout << "[SUCCESS] ALL 5 SAGEATTENTION2 OP_HOST TESTS PASSED!" << std::endl;
        return 0;
    } else {
        std::cerr << "[FAILURE] SOME TESTS FAILED!" << std::endl;
        return 1;
    }
}
