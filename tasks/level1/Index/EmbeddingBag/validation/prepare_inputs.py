import torch
from typing import Optional


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    # 解析参数
    num_embeddings = int(param.get('num_embeddings', 10))
    embedding_dim = int(param.get('embedding_dim', 4))
    num_indices = int(param.get('num_indices', 8))
    num_bags = int(param.get('num_bags', 2))
    dtype_str = param.get('dtype', 'float32')
    mode = param.get('mode', 'mean')
    include_last_offset = param.get('include_last_offset', 'False') == 'True'
    padding_idx = int(param.get('padding_idx', -1))
    has_per_sample_weights = param.get('has_per_sample_weights', 'False') == 'True'
    scale_grad_by_freq = param.get('scale_grad_by_freq', 'False') == 'True'
    sparse = param.get('sparse', 'False') == 'True'
    
    dtype = getattr(torch, dtype_str)
    
    # 生成 weight (num_embeddings, embedding_dim)
    weight = torch.randn(num_embeddings, embedding_dim, dtype=dtype, device=device)
    
    # 生成 indices - 确保在有效范围内
    indices = torch.randint(0, num_embeddings, (num_indices,), dtype=torch.int32, device=device)
    
    # 生成 offsets - 根据 num_bags 动态生成分段点
    # offsets 必须是递增序列，表示每个 bag 在 indices 中的起始位置
    if include_last_offset:
        # include_last_offset=True: offsets 长度为 num_bags + 1
        # 最后一个元素等于 num_indices
        num_segments = num_bags
        segment_size = max(1, num_indices // num_segments)
        offsets_list = [0]
        for i in range(1, num_segments):
            next_offset = min(offsets_list[-1] + segment_size, num_indices - (num_segments - i))
            offsets_list.append(next_offset)
        offsets_list.append(num_indices)  # 最后一个元素
        offsets = torch.tensor(offsets_list, dtype=torch.int32, device=device)
    else:
        # include_last_offset=False: offsets 长度为 num_bags
        segment_size = max(1, num_indices // num_bags)
        offsets_list = [0]
        for i in range(1, num_bags):
            next_offset = min(offsets_list[-1] + segment_size, num_indices - (num_bags - i))
            offsets_list.append(next_offset)
        offsets = torch.tensor(offsets_list, dtype=torch.int32, device=device)
    
    # 生成 per_sample_weights (可选)
    if has_per_sample_weights:
        per_sample_weights = torch.randn(num_indices, dtype=dtype, device=device)
    else:
        per_sample_weights = None
    
    return weight, indices, offsets, per_sample_weights, mode, scale_grad_by_freq, sparse, include_last_offset, padding_idx


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for EmbeddingBag.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed


def custom_check_precision(param, outputs, outputs_new):
    """
    Custom precision check for EmbeddingBag operator.
    """
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    
    # 根据数据类型设置容差
    if dtype == torch.float16:
        RTOL_GENERAL = 1 / 512
    elif dtype == torch.bfloat16:
        RTOL_GENERAL = 1 / 256 + 1 / 16384
    elif dtype == torch.float32:
        RTOL_GENERAL = 1 / 2048 + 1 / 16384
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    rtol = RTOL_GENERAL
    
    # outputs 和 outputs_new 都是 tuple: (y, offset2bag, bag_size, max_indices)
    all_abs_diff, all_rel_diff = [], []
    is_pass = 1
    
    # 只检查第一个输出 y（主要输出）
    out_y = outputs[0]
    out_new_y = outputs_new[0]
    
    abs_diff = torch.abs(out_y - out_new_y)
    rel_diff = abs_diff / (torch.abs(out_y) + 1e-7)
    
    # 检查容差
    tolerance = rtol * torch.maximum(torch.tensor(1.0, device=out_y.device), torch.abs(out_y))
    if torch.any(abs_diff > tolerance):
        is_pass = 0
    
    all_abs_diff.append(abs_diff.view(-1))
    all_rel_diff.append(rel_diff.view(-1))
    
    # 检查整数类型输出 (offset2bag, bag_size, max_indices)
    for idx in [1, 2, 3]:
        out = outputs[idx]
        out_new = outputs_new[idx]
        
        abs_diff_int = torch.abs(out.to(torch.float32) - out_new.to(torch.float32))
        if torch.any(abs_diff_int > 0):
            is_pass = 0
        
        all_abs_diff.append(abs_diff_int.view(-1))
        all_rel_diff.append(torch.zeros_like(abs_diff_int).view(-1))
    
    all_abs_diff = torch.cat(all_abs_diff)
    all_rel_diff = torch.cat(all_rel_diff)
    
    return is_pass, all_abs_diff, all_rel_diff