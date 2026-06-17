import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torchair
import torch_npu
import logging
from typing import Tuple

# region 图模式相关类和函数
def compile_model(model, graph_type, is_full_core=True):
    ''' 图模式编译模型
    '''
    if graph_type == 0:
        logging.debug("graph_type=0，测试场景：单算子模式")
        return model

    from torchair.configs.compiler_config import CompilerConfig
    if is_full_core:
        compiler_config = None
    else:
        # 使用一半的核
        compiler_config = CompilerConfig()
        compiler_config.ge_config.aicore_num = "12|24"
        print(f"当前图模式仅使用: 12 aic核, 24 aiv核")
    npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)
    torch._dynamo.reset()
    if graph_type == 1:  
        # 传统入图模式，静态shape+在线编译场景
        compiled_model = torch.compile(model, backend=npu_backend, dynamic=False)
    elif graph_type == 2:  
        # ACLNN入图模式，动态shape+二进制
        compiled_model = torch.compile(model, backend=npu_backend, dynamic=True)

    return compiled_model

class ATTENTION_TO_FFN_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(
        self, 
        x, 
        session_id, 
        micro_batch_id, 
        layer_id, 
        expert_ids, 
        expert_rank_table, 
        group, 
        world_size, 
        ffn_token_info_table_shape,
        ffn_token_data_shape, 
        attn_token_info_table_shape, 
        moe_expert_num, 
        scales, 
        active_mask, 
        quant_mode, 
        sync_flag
    ):
        ffn_token_info_table_shape_list = [
            int(ffn_token_info_table_shape[0]),
            int(ffn_token_info_table_shape[1]),
            int(ffn_token_info_table_shape[2])
        ]
        
        ffn_token_data_shape_list = [
            int(ffn_token_data_shape[0]),
            int(ffn_token_data_shape[1]),
            int(ffn_token_data_shape[2]),
            int(ffn_token_data_shape[3]),
            int(ffn_token_data_shape[4]),
        ]

        attn_token_info_table_shape_list = [
            int(attn_token_info_table_shape[0]),
            int(attn_token_info_table_shape[1]),
            int(attn_token_info_table_shape[2])
        ]
        output = torch_npu.npu_attention_to_ffn(
            x=x,
            session_id=session_id,
            micro_batch_id=micro_batch_id,
            layer_id=layer_id,
            expert_ids=expert_ids,
            expert_rank_table=expert_rank_table,
            group=group,
            world_size=world_size,
            ffn_token_info_table_shape=ffn_token_info_table_shape_list,
            ffn_token_data_shape=ffn_token_data_shape_list,
            attn_token_info_table_shape=attn_token_info_table_shape_list,
            moe_expert_num=moe_expert_num,
            scales = scales,
            active_mask = active_mask,
            quant_mode = quant_mode,
            sync_flag = sync_flag,
        )
        return output