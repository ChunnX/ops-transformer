import torch
import kernel_gen_ops

def permute(tokens, indices, num_out_tokens=None, padded_mode=False):
    if padded_mode:
        return tokens.index_select(dim=0, index=indices.view(-1)), indices
    
    if indices.dim() == 1:
        topk = 1
    else:
        topk = indices.size(1)
    flatten_indices = indices.view(-1)

    sorted_indices = torch.argsort(flatten_indices, stable=True)
    sorted_indices1 = torch.argsort(sorted_indices, stable=True)

    if num_out_tokens is not None and num_out_tokens != 0:
        sorted_indices = sorted_indices[:num_out_tokens]
    s_k = sorted_indices // topk
    permuted_tokens = tokens.index_select(0, s_k)
    return permuted_tokens, sorted_indices1

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, tokens, permuted_output_grad, indices, num_topk, padded_mode=False):
        tokens.requires_grad_(True)
        permuted_tokens, sorted_indices = permute(tokens, indices, num_topk, padded_mode)
        permuted_tokens.backward(permuted_output_grad)
        return tokens.grad

class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, tokens, permuted_output_grad, indices, num_topk, padded_mode=False):
        tmp, sorted_indices = permute(tokens.cpu(), indices.cpu(), num_topk, padded_mode)
        if indices.dim() == 1:
            topk = 1
        else:
            topk = indices.size(1)

        output = kernel_gen_ops.moe_token_permute_grad(
            permuted_output_grad.npu(), 
            sorted_indices.to(torch.int32).npu(), 
            topk, 
            padded_mode
        )
        return output