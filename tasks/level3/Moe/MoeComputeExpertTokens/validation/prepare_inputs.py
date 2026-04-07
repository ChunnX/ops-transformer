import torch
import random

def get_inputs(param, device=None):
    # 设置固定种子，确保每次生成相同的随机数
    torch.manual_seed(42)
    random.seed(42)
    """
    Generate input tensors for MoeComputeExpertTokens operator.
    
    输入:
        sorted_experts: 已排序的专家索引，形状为 [num_tokens]，数据类型 int32
        num_experts: 专家数量，标量整数
    
    输出:
        total_rows_before_expert: 每个专家之前的总 token 数，形状为 [num_experts]，数据类型 int32
    """
    shape = eval(param.get('input_shape', '[2048]'))
    dtype_str = param.get('dtype', 'int32')
    num_experts = int(param.get('num_experts', '8'))
    
    dtype = getattr(torch, dtype_str)
    
    # 生成排序后的专家索引（0 到 num_experts-1 之间的整数，已排序）
    sorted_experts = torch.randint(0, num_experts, shape, device=device, dtype=dtype)
    sorted_experts, _ = torch.sort(sorted_experts)
    
    return (sorted_experts, num_experts)


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for MoeComputeExpertTokens.
    """
    return []
