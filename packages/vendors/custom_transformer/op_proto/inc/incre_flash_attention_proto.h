#ifndef INCRE_FLASH_ATTENTION_PROTO_H_
#define INCRE_FLASH_ATTENTION_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(IncreFlashAttention)
    .INPUT(query, ge::TensorType::ALL())
    .DYNAMIC_INPUT(key, ge::TensorType::ALL())
    .DYNAMIC_INPUT(value, ge::TensorType::ALL())
    .OPTIONAL_INPUT(pse_shift, ge::TensorType::ALL())
    .OPTIONAL_INPUT(atten_mask, ge::TensorType::ALL())
    .OPTIONAL_INPUT(actual_seq_lengths, ge::TensorType::ALL())
    .OPTIONAL_INPUT(dequant_scale1, ge::TensorType::ALL())
    .OPTIONAL_INPUT(quant_scale1, ge::TensorType::ALL())
    .OPTIONAL_INPUT(dequant_scale2, ge::TensorType::ALL())
    .OPTIONAL_INPUT(quant_scale2, ge::TensorType::ALL())
    .OPTIONAL_INPUT(quant_offset2, ge::TensorType::ALL())
    .OPTIONAL_INPUT(antiquant_scale, ge::TensorType::ALL())
    .OPTIONAL_INPUT(antiquant_offset, ge::TensorType::ALL())
    .OPTIONAL_INPUT(block_table, ge::TensorType::ALL())
    .OPTIONAL_INPUT(kv_padding_size, ge::TensorType::ALL())
    .OUTPUT(attention_out, ge::TensorType::ALL())
    .REQUIRED_ATTR(num_heads, Int)
    .ATTR(scale_value, Float, 1)
    .ATTR(input_layout, String, "BSH")
    .ATTR(num_key_value_heads, Int, 0)
    .ATTR(block_size, Int, 0)
    .ATTR(inner_precise, Int, 1)
    .OP_END_FACTORY_REG(IncreFlashAttention);

}

#endif
