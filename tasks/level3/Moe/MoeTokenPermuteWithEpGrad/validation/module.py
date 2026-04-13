import torch

class Model(torch.nn.Module):
    def __init__(self, unpermuted_tokens_grad, sorted_indices, permuted_probs_output_grad, topk_num, range_vals, padded_mode):
        super(Model, self).__init__()
        # 存储参数，但不参与前向计算
        self.topk_num = topk_num
        self.range_vals = range_vals
        self.padded_mode = padded_mode
        
    def forward(self, unpermuted_tokens_grad, sorted_indices, permuted_probs_output_grad, topk_num, range_vals, padded_mode):
        # 根据ACLNN接口文档实现正确的计算逻辑
        hidden_size = unpermuted_tokens_grad.shape[1]
        num_tokens = sorted_indices.shape[0]
        
        # 初始化输出梯度张量
        token_grad_out = torch.zeros(num_tokens, hidden_size, dtype=unpermuted_tokens_grad.dtype, device=unpermuted_tokens_grad.device)
        
        # 将sorted_indices转换为LongTensor
        sorted_indices_long = sorted_indices.to(torch.int64)
        
        # 计算token_grad_out
        # 当range_vals[0] <= sorted_indices[i] < range_vals[1]时：
        # token_grad_out[i] = unpermuted_tokens_grad[sorted_indices[i] - range_vals[0]]
        # 否则：token_grad_out[i] = 0
        if range_vals is not None and len(range_vals) == 2:
            start = range_vals[0]
            end = range_vals[1]
            
            # 创建掩码，标记哪些索引在范围内
            mask = (sorted_indices_long >= start) & (sorted_indices_long < end)
            
            # 计算有效索引
            valid_indices = sorted_indices_long[mask] - start
            
            # 将unpermuted_tokens_grad中对应位置的值复制到token_grad_out中
            token_grad_out[mask] = unpermuted_tokens_grad[valid_indices]
        
        # reshape并沿topK维度求和
        token_grad_out = token_grad_out.reshape(-1, topk_num, hidden_size)
        token_grad_out = token_grad_out.sum(dim=1)
        
        # probs_grad_out的处理（如果需要）
        if permuted_probs_output_grad is not None:
            probs_grad_out = torch.zeros(num_tokens, topk_num, dtype=permuted_probs_output_grad.dtype, device=permuted_probs_output_grad.device)
            
            if range_vals is not None and len(range_vals) == 2:
                start = range_vals[0]
                end = range_vals[1]
                
                # 创建掩码，标记哪些索引在范围内
                mask = (sorted_indices_long >= start) & (sorted_indices_long < end)
                
                # 计算有效索引
                valid_indices = sorted_indices_long[mask] - start
                
                # 将permuted_probs_output_grad中对应位置的值复制到probs_grad_out中
                probs_grad_out[mask] = permuted_probs_output_grad.view(-1, 1).expand(-1, topk_num)[valid_indices]
            
            # 切片操作
            if range_vals is not None and len(range_vals) == 2:
                probs_grad_out = probs_grad_out[range_vals[0]:range_vals[1]]
            
            probs_grad_out = probs_grad_out.reshape(-1, topk_num)
        else:
            probs_grad_out = None
        
        return [
            token_grad_out,
            probs_grad_out
        ]

class ModelNew(torch.nn.Module):
    def __init__(self, unpermuted_tokens_grad, sorted_indices, permuted_probs_output_grad, topk_num, range_vals, padded_mode):
        super(ModelNew, self).__init__()
        # 存储参数，但不参与前向计算
        self.topk_num = topk_num
        self.range_vals = range_vals
        self.padded_mode = padded_mode
        
    def forward(self, unpermuted_tokens_grad, sorted_indices, permuted_probs_output_grad, topk_num, range_vals, padded_mode):
        # 调用绑定的C++算子
        from kernel_gen_ops import moe_token_permute_with_ep_grad
        return list(moe_token_permute_with_ep_grad(
            unpermuted_tokens_grad, 
            sorted_indices, 
            permuted_probs_output_grad, 
            topk_num, 
            range_vals, 
            padded_mode
        ))