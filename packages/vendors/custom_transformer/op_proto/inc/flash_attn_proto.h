#ifndef FLASH_ATTN_PROTO_H_
#define FLASH_ATTN_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(FlashAttn)
    .INPUT(q, ge::TensorType::ALL())
    .INPUT(k, ge::TensorType::ALL())
    .INPUT(v, ge::TensorType::ALL())
    .OPTIONAL_INPUT(block_table, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cu_seqlens_q, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cu_seqlens_kv, ge::TensorType::ALL())
    .OPTIONAL_INPUT(seqused_q, ge::TensorType::ALL())
    .OPTIONAL_INPUT(seqused_kv, ge::TensorType::ALL())
    .OPTIONAL_INPUT(sinks, ge::TensorType::ALL())
    .OPTIONAL_INPUT(attn_mask, ge::TensorType::ALL())
    .OPTIONAL_INPUT(metadata, ge::TensorType::ALL())
    .OUTPUT(attn_out, ge::TensorType::ALL())
    .OUTPUT(softmax_lse, ge::TensorType::ALL())
    .ATTR(softmax_scale, Float, 0)
    .ATTR(mask_mode, Int, 0)
    .ATTR(win_left, Int, -1)
    .ATTR(win_right, Int, -1)
    .ATTR(max_seqlen_q, Int, -1)
    .ATTR(max_seqlen_kv, Int, -1)
    .ATTR(layout_q, String, "BSND")
    .ATTR(layout_kv, String, "BSND")
    .ATTR(layout_out, String, "BSND")
    .ATTR(return_softmax_lse, Int, 0)
    .ATTR(deterministic, Int, 0)
    .OP_END_FACTORY_REG(FlashAttn);

}

#endif
