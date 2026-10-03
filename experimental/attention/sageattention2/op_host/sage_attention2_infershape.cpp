/*!
 * \file sage_attention2_infershape.cpp
 * \brief SageAttention2 算子 InferShape 与 InferDataType 准确推导实现
 */

#include <iostream>

#include "sage_attention2_def.h"

// 布局数值编码（LAYOUT_NHD/LAYOUT_HND）唯一定义位于 aclnn/sage_attn_proto.h
// （经 ascendc/tiling.h），本文件不再维护第二套 LayoutType 枚举
#include "../aclnn/sage_attn_proto.h"

namespace ops {

namespace {

static constexpr size_t EXPECTED_DIM_NUM = 4;
static constexpr int64_t MAX_HEAD_DIM = 128;

/**
 * 布局属性解析：字符串经 sage_attn::TensorLayoutCodeFromString 唯一映射，
 * 数值属性直接比对 proto 编码；默认 HND。
 */
inline sage_attn::TensorLayout DetectLayout(const gert::InferShapeContext *context)
{
    const auto *attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const char *layout_str = ReadStringAttr(attrs, sage_attn2_idx::ATTR_TENSOR_LAYOUT);
        if (layout_str != nullptr &&
            sage_attn::TensorLayoutCodeFromString(layout_str) == static_cast<int64_t>(sage_attn::LAYOUT_NHD)) {
            return sage_attn::LAYOUT_NHD;
        }
        const int64_t *layout_int = attrs->GetInt(sage_attn2_idx::ATTR_TENSOR_LAYOUT);
        if (layout_int != nullptr && *layout_int == static_cast<int64_t>(sage_attn::LAYOUT_NHD)) {
            return sage_attn::LAYOUT_NHD;
        }
    }
    return sage_attn::LAYOUT_HND;
}

} // namespace

/**
 * InferShapeSageAttention2:
 * 自动根据输入 Q/K/V 形状推导 Output 与 LSE 的维度及边界
 * 1. 严格校验 Q/K/V 为 4D 张量
 * 2. 校验 Head Dim 物理约束 (D <= 128，支持 32/64/80/96/128 等边界)
 * 3. 校验 Batch Size 严格匹配
 * 4. 严格校验 GQA 分组广播整除性 (Hq % Hkv == 0 且 Hq >= Hkv)
 * 5. 推导并设置 Output 与 LSE 目标形状
 */
static ge::graphStatus InferShapeSageAttention2(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *queryShape = context->GetInputShape(sage_attn2_idx::IN_QUERY);
    const gert::Shape *keyShape = context->GetInputShape(sage_attn2_idx::IN_KEY);
    const gert::Shape *valueShape = context->GetInputShape(sage_attn2_idx::IN_VALUE);

    if (queryShape == nullptr || keyShape == nullptr || valueShape == nullptr) {
        std::cerr << "[SageAttention2][InferShape] Query, Key or Value input shape is null!" << std::endl;
        return ge::GRAPH_FAILED;
    }

    if (queryShape->GetDimNum() != EXPECTED_DIM_NUM ||
        keyShape->GetDimNum() != EXPECTED_DIM_NUM ||
        valueShape->GetDimNum() != EXPECTED_DIM_NUM) {
        std::cerr << "[SageAttention2][InferShape] Q, K, V must all be 4-dimensional tensors!" << std::endl;
        return ge::GRAPH_FAILED;
    }

    sage_attn::TensorLayout layout = DetectLayout(context);

    int64_t b = 0, hq = 0, sq = 0, d = 0;
    int64_t kb = 0, hkv = 0, sk = 0, kd = 0;
    int64_t vb = 0, vhkv = 0, vsk = 0, vd = 0;

    if (layout == sage_attn::LAYOUT_HND) {
        // HND: [B, H, S, D]
        b = queryShape->GetDim(0);
        hq = queryShape->GetDim(1);
        sq = queryShape->GetDim(2);
        d = queryShape->GetDim(3);

        kb = keyShape->GetDim(0);
        hkv = keyShape->GetDim(1);
        sk = keyShape->GetDim(2);
        kd = keyShape->GetDim(3);

        vb = valueShape->GetDim(0);
        vhkv = valueShape->GetDim(1);
        vsk = valueShape->GetDim(2);
        vd = valueShape->GetDim(3);
    } else {
        // NHD: [B, S, H, D]
        b = queryShape->GetDim(0);
        sq = queryShape->GetDim(1);
        hq = queryShape->GetDim(2);
        d = queryShape->GetDim(3);

        kb = keyShape->GetDim(0);
        sk = keyShape->GetDim(1);
        hkv = keyShape->GetDim(2);
        kd = keyShape->GetDim(3);

        vb = valueShape->GetDim(0);
        vsk = valueShape->GetDim(1);
        vhkv = valueShape->GetDim(2);
        vd = valueShape->GetDim(3);
    }

    // 1. Batch 校验
    if (b <= 0 || kb != b || vb != b) {
        std::cerr << "[SageAttention2][InferShape] Batch dimensions mismatch! Q: "
                  << b << ", K: " << kb << ", V: " << vb << std::endl;
        return ge::GRAPH_FAILED;
    }

    // 2. Head Dim 校验 (必须匹配且 <= 128)
    if (d <= 0 || kd != d || vd != d) {
        std::cerr << "[SageAttention2][InferShape] Head dim mismatch! Q: "
                  << d << ", K: " << kd << ", V: " << vd << std::endl;
        return ge::GRAPH_FAILED;
    }
    if (d > MAX_HEAD_DIM) {
        std::cerr << "[SageAttention2][InferShape] Head dim exceeds hardware limit (128): " << d << std::endl;
        return ge::GRAPH_FAILED;
    }

    // 3. Sequence 长度校验
    if (sq <= 0 || sk <= 0 || vsk != sk) {
        std::cerr << "[SageAttention2][InferShape] Sequence length mismatch! K: "
                  << sk << ", V: " << vsk << std::endl;
        return ge::GRAPH_FAILED;
    }

    // 4. GQA 广播头数校验 (Hq 必须能被 Hkv 整除，且 Hkv == vHkv)
    if (hkv <= 0 || vhkv != hkv) {
        std::cerr << "[SageAttention2][InferShape] Key/Value heads mismatch! K: "
                  << hkv << ", V: " << vhkv << std::endl;
        return ge::GRAPH_FAILED;
    }
    if (hq < hkv || (hq % hkv != 0)) {
        std::cerr << "[SageAttention2][InferShape] Invalid GQA configuration! Hq ("
                  << hq << ") must be divisible by Hkv (" << hkv << ")!" << std::endl;
        return ge::GRAPH_FAILED;
    }

    // 5. 设置 Output 形状 (与 Query 形状完全一致)
    gert::Shape *outputShape = context->GetOutputShape(sage_attn2_idx::OUT_OUTPUT);
    if (outputShape != nullptr) {
        *outputShape = *queryShape;
    }

    // 6. 若存在 LSE 输出，推导并设置 LSE 形状
    gert::Shape *lseShape = context->GetOutputShape(sage_attn2_idx::OUT_LSE);
    if (lseShape != nullptr) {
        // LSE 布局恒为 [B, Hq, Sq]（对齐任务书 §2.2 返回值契约与核内寻址）
        *lseShape = {b, hq, sq};
    }

    // 7. 可选平滑与修正项形状校验
    const gert::Shape *keyMeanShape = context->GetInputShape(sage_attn2_idx::IN_KEY_MEAN);
    if (keyMeanShape != nullptr && keyMeanShape->GetDimNum() > 0) {
        // km 契约（DEFECT-L-1 三方对齐：内核 float* 读 d_pad、孪生 assign d_pad、
        // python 桥按 pad 后 K 求均值）：末维 = d_pad（d<=64→64，否则 128），
        // 位型仅 DT_FLOAT（内核 float* 直读，fp16 km 会按 fp32 误读）
        const int64_t d_pad_km = (d <= 64) ? 64 : 128;
        if (keyMeanShape->GetDimNum() == 3) {
            if (keyMeanShape->GetDim(0) != b || keyMeanShape->GetDim(1) != hkv ||
                keyMeanShape->GetDim(2) != d_pad_km) {
                std::cerr << "[SageAttention2][InferShape] key_mean 3D shape mismatch! Expected ["
                          << b << ", " << hkv << ", " << d_pad_km << "] (padded head dim)" << std::endl;
                return ge::GRAPH_FAILED;
            }
        }
        // km 位型约束由 def.cpp Input 注册（DT_FLOAT）在图编译期强制；
        // InferShapeContext 无 dtype 查询接口，此处只做形状校验。
    }

    const gert::Shape *lseCorrShape = context->GetInputShape(sage_attn2_idx::IN_LSE_CORRECTION);
    if (lseCorrShape != nullptr && lseCorrShape->GetDimNum() > 0) {
        if (lseCorrShape->GetDimNum() == 3) {
            // lse_correction 契约恒为 [B, Hq, Sq]（对齐接口规范与核内行语义，不随 tensor_layout 翻转）
            if (lseCorrShape->GetDim(0) != b || lseCorrShape->GetDim(1) != hq || lseCorrShape->GetDim(2) != sq) {
                std::cerr << "[SageAttention2][InferShape] lse_correction shape mismatch! Expected ["
                          << b << ", " << hq << ", " << sq << "]" << std::endl;
                return ge::GRAPH_FAILED;
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

/**
 * InferDataTypeSageAttention2:
 * 自动根据输入 Query 数据类型推导 Output 与 LSE 的数据类型
 * Output 类型严格与 Query 类型对齐 (DT_FLOAT16 或 DT_BF16)
 * LSE 类型严格为单精度浮点 (DT_FLOAT)
 */
static ge::graphStatus InferDataTypeSageAttention2(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto queryDtype = context->GetInputDataType(sage_attn2_idx::IN_QUERY);
    auto keyDtype = context->GetInputDataType(sage_attn2_idx::IN_KEY);
    auto valDtype = context->GetInputDataType(sage_attn2_idx::IN_VALUE);

    // 校验输入数据类型（任务书 §3：输入 dtype 仅 FP16/BF16）
    if (queryDtype != ge::DT_FLOAT16 && queryDtype != ge::DT_BF16) {
        std::cerr << "[SageAttention2][InferDataType] Unsupported Query dtype: " << queryDtype << std::endl;
        return ge::GRAPH_FAILED;
    }
    if (keyDtype != queryDtype || valDtype != queryDtype) {
        std::cerr << "[SageAttention2][InferDataType] Q, K, V data types must match! Q="
                  << queryDtype << ", K=" << keyDtype << ", V=" << valDtype << std::endl;
        return ge::GRAPH_FAILED;
    }

    // 设置 Output 数据类型同 Query
    context->SetOutputDataType(sage_attn2_idx::OUT_OUTPUT, queryDtype);

    // 设置 LSE 数据类型为 DT_FLOAT
    context->SetOutputDataType(sage_attn2_idx::OUT_LSE, ge::DT_FLOAT);

    return ge::GRAPH_SUCCESS;
}

// 注册 InferShape 与 InferDataType 函数至算子原型系统
IMPL_OP_INFERSHAPE(SageAttention2)
    .InferShape(InferShapeSageAttention2)
    .InferDataType(InferDataTypeSageAttention2);

} // namespace ops
