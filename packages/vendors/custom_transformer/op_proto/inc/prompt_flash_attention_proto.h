#ifndef PROMPT_FLASH_ATTENTION_PROTO_H_
#define PROMPT_FLASH_ATTENTION_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(PromptFlashAttention)
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(key, ge::TensorType::ALL())
    .INPUT(value, ge::TensorType::ALL())
    .OPTIONAL_INPUT(pse_shift, ge::TensorType::ALL())
    .OPTIONAL_INPUT(atten_mask, ge::TensorType::ALL())
    .OPTIONAL_INPUT(actual_seq_lengths, ge::TensorType::ALL())
    .OPTIONAL_INPUT(actual_seq_lengths_kv, ge::TensorType::ALL())
    .OPTIONAL_INPUT(deq_scale1, ge::TensorType::ALL())
    .OPTIONAL_INPUT(quant_scale1, ge::TensorType::ALL())
    .OPTIONAL_INPUT(deq_scale2, ge::TensorType::ALL())
    .OPTIONAL_INPUT(quant_scale2, ge::TensorType::ALL())
    .OPTIONAL_INPUT(quant_offset2, ge::TensorType::ALL())
    .OUTPUT(attention_out, ge::TensorType::ALL())
    .REQUIRED_ATTR(num_heads, Int)
    .ATTR(scale_value, Float, 1)
    .ATTR(pre_tokens, Int, 214748647)
    .ATTR(next_tokens, Int, 0)
    .ATTR(input_layout, String, "BSH")
    .ATTR(num_key_value_heads, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(inner_precise, Int, 1)
    .OP_END_FACTORY_REG(PromptFlashAttention);

}

#endif
