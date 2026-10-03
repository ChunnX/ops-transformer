#include <iostream>
#include <cassert>
#include <vector>
#include <string>

#include "../op_host/sage_attention2_def.h"

// 宏定义测试辅助
#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAIL] Line " << __LINE__ << ": " << msg << std::endl; \
            return false; \
        } \
    } while (0)

namespace {

// 1. 验证 REG_OP 宏原型注册完整度
bool TestRegOpDefinition()
{
    std::cout << "[RUN] TestRegOpDefinition..." << std::endl;
    auto &reg = ge::GetSageAttention2Reg();

    TEST_ASSERT(reg.GetOpName() == "SageAttention2", "OpName must be SageAttention2");

    // 验证所有输入张量
    TEST_ASSERT(reg.HasInput("query"), "Must have input: query");
    TEST_ASSERT(reg.HasInput("key"), "Must have input: key");
    TEST_ASSERT(reg.HasInput("value"), "Must have input: value");
    TEST_ASSERT(reg.HasInput("key_mean"), "Must have optional input: key_mean");
    TEST_ASSERT(reg.HasInput("lse_correction"), "Must have optional input: lse_correction");
    TEST_ASSERT(reg.HasOptionalInput("key_mean"), "key_mean must be optional input");
    TEST_ASSERT(reg.HasOptionalInput("lse_correction"), "lse_correction must be optional input");

    // 验证所有输出张量
    TEST_ASSERT(reg.HasOutput("output"), "Must have output: output");
    TEST_ASSERT(reg.HasOutput("lse"), "Must have output: lse");

    // 验证全部 7 个属性，包括 qk_quant_gran 与 pv_accum_dtype
    TEST_ASSERT(reg.HasAttr("tensor_layout"), "Must have attr: tensor_layout");
    TEST_ASSERT(reg.HasAttr("is_causal"), "Must have attr: is_causal");
    TEST_ASSERT(reg.HasAttr("qk_quant_gran"), "Must have attr: qk_quant_gran");
    TEST_ASSERT(reg.HasAttr("sm_scale"), "Must have attr: sm_scale");
    TEST_ASSERT(reg.HasAttr("pv_accum_dtype"), "Must have attr: pv_accum_dtype");
    TEST_ASSERT(reg.HasAttr("smooth_k"), "Must have attr: smooth_k");
    TEST_ASSERT(reg.HasAttr("smooth_v"), "Must have attr: smooth_v");

    // 校验属性默认值
    TEST_ASSERT(reg.GetAttr("tensor_layout").str_val == "HND", "tensor_layout default HND");
    TEST_ASSERT(reg.GetAttr("is_causal").bool_val == false, "is_causal default false");
    TEST_ASSERT(reg.GetAttr("qk_quant_gran").str_val == "per_thread", "qk_quant_gran default per_thread");
    TEST_ASSERT(reg.GetAttr("pv_accum_dtype").str_val == "fp32+fp16", "pv_accum_dtype default fp32+fp16");
    TEST_ASSERT(reg.GetAttr("smooth_k").bool_val == true, "smooth_k default true");
    TEST_ASSERT(reg.GetAttr("smooth_v").bool_val == false, "smooth_v default false");

    std::cout << "[PASS] TestRegOpDefinition" << std::endl;
    return true;
}

// 2. 验证 OpDef 信息库注册与硬件配置
bool TestOpDefRegistry()
{
    std::cout << "[RUN] TestOpDefRegistry..." << std::endl;
    auto op = ops::OpDefRegistry::GetInstance().GetOp("SageAttention2");
    TEST_ASSERT(op != nullptr, "SageAttention2 must be registered in OpDefRegistry");

    TEST_ASSERT(op->HasInput("query"), "OpDef query input");
    TEST_ASSERT(op->HasInput("key"), "OpDef key input");
    TEST_ASSERT(op->HasInput("value"), "OpDef value input");
    TEST_ASSERT(op->HasInput("key_mean"), "OpDef key_mean input");
    TEST_ASSERT(op->HasInput("lse_correction"), "OpDef lse_correction input");
    TEST_ASSERT(op->GetInput("key_mean").GetParamType() == ops::OPTIONAL, "key_mean must be OPTIONAL");

    TEST_ASSERT(op->HasOutput("output"), "OpDef output");
    TEST_ASSERT(op->HasOutput("lse"), "OpDef lse");

    TEST_ASSERT(op->HasAttr("qk_quant_gran"), "OpDef qk_quant_gran");
    TEST_ASSERT(op->HasAttr("pv_accum_dtype"), "OpDef pv_accum_dtype");

    // 验证 AI Core 硬件架构配置包含 ascend950
    TEST_ASSERT(op->GetAICore().HasSoc("ascend950"), "OpDef AICore must configure ascend950");
    auto cfg950 = op->GetAICore().GetConfig("ascend950");
    TEST_ASSERT(cfg950.GetExtInfo("opFile.value") == "sageattention2", "opFile.value must be sageattention2");
    TEST_ASSERT(cfg950.GetExtInfo("opInterface.value") == "sage_attention2", "opInterface.value must be sage_attention2");

    std::cout << "[PASS] TestOpDefRegistry" << std::endl;
    return true;
}

// 3. 验证 InferShape 准确推导 (标准 MHA、GQA 广播、任意 Head Dim)
bool TestInferShapeValidCases()
{
    std::cout << "[RUN] TestInferShapeValidCases..." << std::endl;
    auto infer_shape_func = ops::OpImplRegistryMock::GetInstance("SageAttention2").GetInferShape();
    TEST_ASSERT(infer_shape_func != nullptr, "InferShape function must be registered");

    // 用例 A: 标准 MHA (B=2, Hq=32, Hkv=32, Sq=1024, Sk=1024, D=128)
    {
        gert::InferShapeContext ctx("SageAttention2_MHA");
        gert::Shape q_shape({2, 32, 1024, 128});
        gert::Shape k_shape({2, 32, 1024, 128});
        gert::Shape v_shape({2, 32, 1024, 128});
        gert::Shape out_shape;
        gert::Shape lse_shape;

        ctx.SetInputShape(0, &q_shape);
        ctx.SetInputShape(1, &k_shape);
        ctx.SetInputShape(2, &v_shape);
        ctx.SetOutputShape(0, &out_shape);
        ctx.SetOutputShape(1, &lse_shape);

        ge::graphStatus ret = infer_shape_func(&ctx);
        TEST_ASSERT(ret == ge::GRAPH_SUCCESS, "InferShape MHA failed");
        TEST_ASSERT(out_shape == q_shape, "Output shape must match Query shape");
        TEST_ASSERT(lse_shape == gert::Shape({2, 32, 1024}), "LSE shape must be [B, Hq, Sq]");
    }

    // 用例 B: GQA 广播 (B=4, Hq=32, Hkv=8 [4:1], Sq=512, Sk=2048, D=64)
    {
        gert::InferShapeContext ctx("SageAttention2_GQA");
        gert::Shape q_shape({4, 32, 512, 64});
        gert::Shape k_shape({4, 8, 2048, 64});
        gert::Shape v_shape({4, 8, 2048, 64});
        gert::Shape out_shape;
        gert::Shape lse_shape;

        ctx.SetInputShape(0, &q_shape);
        ctx.SetInputShape(1, &k_shape);
        ctx.SetInputShape(2, &v_shape);
        ctx.SetOutputShape(0, &out_shape);
        ctx.SetOutputShape(1, &lse_shape);

        ge::graphStatus ret = infer_shape_func(&ctx);
        TEST_ASSERT(ret == ge::GRAPH_SUCCESS, "InferShape GQA failed");
        TEST_ASSERT(out_shape == q_shape, "GQA Output shape must match Query");
        TEST_ASSERT(lse_shape == gert::Shape({4, 32, 512}), "GQA LSE shape must be [4, 32, 512]");
    }

    // 用例 C: MQA 广播 (B=1, Hq=16, Hkv=1 [16:1], Sq=128, Sk=512, D=128)
    {
        gert::InferShapeContext ctx("SageAttention2_MQA");
        gert::Shape q_shape({1, 16, 128, 128});
        gert::Shape k_shape({1, 1, 512, 128});
        gert::Shape v_shape({1, 1, 512, 128});
        gert::Shape out_shape;

        ctx.SetInputShape(0, &q_shape);
        ctx.SetInputShape(1, &k_shape);
        ctx.SetInputShape(2, &v_shape);
        ctx.SetOutputShape(0, &out_shape);

        ge::graphStatus ret = infer_shape_func(&ctx);
        TEST_ASSERT(ret == ge::GRAPH_SUCCESS, "InferShape MQA failed");
        TEST_ASSERT(out_shape == q_shape, "MQA Output shape must match Query");
    }

    // 用例 D: 非标准头维度边界 (D=80 与 D=96)
    {
        gert::InferShapeContext ctx("SageAttention2_D80");
        gert::Shape q_shape({2, 16, 256, 80});
        gert::Shape k_shape({2, 16, 256, 80});
        gert::Shape v_shape({2, 16, 256, 80});
        gert::Shape out_shape;

        ctx.SetInputShape(0, &q_shape);
        ctx.SetInputShape(1, &k_shape);
        ctx.SetInputShape(2, &v_shape);
        ctx.SetOutputShape(0, &out_shape);

        ge::graphStatus ret = infer_shape_func(&ctx);
        TEST_ASSERT(ret == ge::GRAPH_SUCCESS, "InferShape D=80 failed");
        TEST_ASSERT(out_shape.GetDim(3) == 80, "Output D must be 80");
    }

    // 用例 E: NHD 布局
    {
        gert::InferShapeContext ctx("SageAttention2_NHD");
        gert::RuntimeAttrs attrs;
        attrs.AddString("NHD");
        ctx.SetAttrs(&attrs);

        gert::Shape q_shape({2, 512, 32, 128}); // [B, Sq, Hq, D]
        gert::Shape k_shape({2, 512, 8, 128});  // [B, Sk, Hkv, D]
        gert::Shape v_shape({2, 512, 8, 128});  // [B, Sk, Hkv, D]
        gert::Shape out_shape;
        gert::Shape lse_shape;

        ctx.SetInputShape(0, &q_shape);
        ctx.SetInputShape(1, &k_shape);
        ctx.SetInputShape(2, &v_shape);
        ctx.SetOutputShape(0, &out_shape);
        ctx.SetOutputShape(1, &lse_shape);

        ge::graphStatus ret = infer_shape_func(&ctx);
        TEST_ASSERT(ret == ge::GRAPH_SUCCESS, "InferShape NHD failed");
        TEST_ASSERT(out_shape == q_shape, "NHD Output shape must match Query");
        // LSE 布局恒为 [B, Hq, Sq]（与核内扁平写址
        // op_kernel FinalizeAndStoreOutput 的 ((b*Hq+h)*Sq+s) 一致，不随 NHD 翻转）
        TEST_ASSERT(lse_shape == gert::Shape({2, 32, 512}), "NHD LSE shape must be [B, Hq, Sq]");
    }

    std::cout << "[PASS] TestInferShapeValidCases" << std::endl;
    return true;
}

// 4. 验证 InferShape 异常边界拦截
bool TestInferShapeErrorHandling()
{
    std::cout << "[RUN] TestInferShapeErrorHandling..." << std::endl;
    auto infer_shape_func = ops::OpImplRegistryMock::GetInstance("SageAttention2").GetInferShape();

    // 1. 空指针测试
    TEST_ASSERT(infer_shape_func(nullptr) == ge::GRAPH_FAILED, "Should reject null context");

    // 2. 缺失输入张量测试
    {
        gert::InferShapeContext ctx;
        gert::Shape q_shape({2, 32, 1024, 128});
        ctx.SetInputShape(0, &q_shape);
        TEST_ASSERT(infer_shape_func(&ctx) == ge::GRAPH_FAILED, "Should reject missing K/V");
    }

    // 3. Batch 不匹配
    {
        gert::InferShapeContext ctx;
        gert::Shape q_shape({2, 32, 1024, 128});
        gert::Shape k_shape({3, 32, 1024, 128}); // Batch 3 != 2
        gert::Shape v_shape({3, 32, 1024, 128});
        gert::Shape out;
        ctx.SetInputShape(0, &q_shape);
        ctx.SetInputShape(1, &k_shape);
        ctx.SetInputShape(2, &v_shape);
        ctx.SetOutputShape(0, &out);
        TEST_ASSERT(infer_shape_func(&ctx) == ge::GRAPH_FAILED, "Should reject mismatched Batch");
    }

    // 4. Head Dim 超出硬件上限 (D=256 > 128)
    {
        gert::InferShapeContext ctx;
        gert::Shape q_shape({2, 32, 1024, 256});
        gert::Shape k_shape({2, 32, 1024, 256});
        gert::Shape v_shape({2, 32, 1024, 256});
        gert::Shape out;
        ctx.SetInputShape(0, &q_shape);
        ctx.SetInputShape(1, &k_shape);
        ctx.SetInputShape(2, &v_shape);
        ctx.SetOutputShape(0, &out);
        TEST_ASSERT(infer_shape_func(&ctx) == ge::GRAPH_FAILED, "Should reject D > 128");
    }

    // 5. GQA 头数不可整除 (Hq=32, Hkv=7)
    {
        gert::InferShapeContext ctx;
        gert::Shape q_shape({2, 32, 1024, 128});
        gert::Shape k_shape({2, 7, 1024, 128});
        gert::Shape v_shape({2, 7, 1024, 128});
        gert::Shape out;
        ctx.SetInputShape(0, &q_shape);
        ctx.SetInputShape(1, &k_shape);
        ctx.SetInputShape(2, &v_shape);
        ctx.SetOutputShape(0, &out);
        TEST_ASSERT(infer_shape_func(&ctx) == ge::GRAPH_FAILED, "Should reject non-divisible GQA heads");
    }

    // 6. 张量维度非 4D
    {
        gert::InferShapeContext ctx;
        gert::Shape q_shape({2, 32, 1024}); // 3D
        gert::Shape k_shape({2, 32, 1024});
        gert::Shape v_shape({2, 32, 1024});
        gert::Shape out;
        ctx.SetInputShape(0, &q_shape);
        ctx.SetInputShape(1, &k_shape);
        ctx.SetInputShape(2, &v_shape);
        ctx.SetOutputShape(0, &out);
        TEST_ASSERT(infer_shape_func(&ctx) == ge::GRAPH_FAILED, "Should reject non-4D tensor");
    }

    std::cout << "[PASS] TestInferShapeErrorHandling" << std::endl;
    return true;
}

// 5. 验证 InferDataType 数据类型推导
bool TestInferDataType()
{
    std::cout << "[RUN] TestInferDataType..." << std::endl;
    auto infer_data_type_func = ops::OpImplRegistryMock::GetInstance("SageAttention2").GetInferDataType();
    TEST_ASSERT(infer_data_type_func != nullptr, "InferDataType function must be registered");

    // 用例 A: Float16 输入推导
    {
        gert::InferDataTypeContext ctx("SageAttention2_FP16");
        ctx.SetInputDataType(0, ge::DT_FLOAT16);
        ctx.SetInputDataType(1, ge::DT_FLOAT16);
        ctx.SetInputDataType(2, ge::DT_FLOAT16);

        ge::graphStatus ret = infer_data_type_func(&ctx);
        TEST_ASSERT(ret == ge::GRAPH_SUCCESS, "InferDataType FP16 failed");
        TEST_ASSERT(ctx.GetOutputDataType(0) == ge::DT_FLOAT16, "Output dtype must be FP16");
        TEST_ASSERT(ctx.GetOutputDataType(1) == ge::DT_FLOAT, "LSE dtype must be FLOAT");
    }

    // 用例 B: BFloat16 输入推导
    {
        gert::InferDataTypeContext ctx("SageAttention2_BF16");
        ctx.SetInputDataType(0, ge::DT_BF16);
        ctx.SetInputDataType(1, ge::DT_BF16);
        ctx.SetInputDataType(2, ge::DT_BF16);

        ge::graphStatus ret = infer_data_type_func(&ctx);
        TEST_ASSERT(ret == ge::GRAPH_SUCCESS, "InferDataType BF16 failed");
        TEST_ASSERT(ctx.GetOutputDataType(0) == ge::DT_BF16, "Output dtype must be BF16");
        TEST_ASSERT(ctx.GetOutputDataType(1) == ge::DT_FLOAT, "LSE dtype must be FLOAT");
    }

    // 用例 C: Q 与 K 数据类型不匹配拦截
    {
        gert::InferDataTypeContext ctx("SageAttention2_Mismatch");
        ctx.SetInputDataType(0, ge::DT_FLOAT16);
        ctx.SetInputDataType(1, ge::DT_BF16);
        ctx.SetInputDataType(2, ge::DT_FLOAT16);

        ge::graphStatus ret = infer_data_type_func(&ctx);
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
