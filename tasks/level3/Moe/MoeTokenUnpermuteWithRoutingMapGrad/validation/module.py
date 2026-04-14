from typing import List, Optional
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """PyTorch native reference implementation (golden model)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, unpermuted_tokens_grad, out_index, permute_token_id,
                routing_map, permuted_tokens, probs, drop_and_pad, restore_shape):
        """
        Args:
            unpermuted_tokens_grad: [tokens_num, hidden_size] - gradient from forward output
            out_index: [total_length] - output position indices
            permute_token_id: [total_length] - token id for each position
            routing_map: optional [tokens_num, experts_num] bool/int8 mask
            permuted_tokens: optional [total_length, hidden_size] - forward permuted tokens
            probs: optional [tokens_num, experts_num] - expert probabilities
            drop_and_pad: bool - padded mode flag
            restore_shape: List[int] - shape info for padded mode
        Returns:
            permuted_tokens_grad: [out_length, hidden_size]
            probs_grad: optional [tokens_num, experts_num]
        """
        orig_dtype = unpermuted_tokens_grad.dtype
        tokens_num = unpermuted_tokens_grad.shape[0]
        hidden_size = unpermuted_tokens_grad.shape[1]
        total_length = out_index.shape[0]

        grad_f = unpermuted_tokens_grad.float()
        out_index_cpu = out_index.cpu().long()
        permute_token_id_cpu = permute_token_id.cpu().long()

        # Compute permuted_tokens_grad (always produced)
        if probs is None:
            # probs=None case: simple scatter
            permuted_tokens_grad = torch.zeros(total_length, hidden_size,
                                               dtype=torch.float32, device=unpermuted_tokens_grad.device)
            for i in range(total_length):
                tid = permute_token_id_cpu[i].item()
                oi = out_index_cpu[i].item()
                permuted_tokens_grad[oi] = grad_f[tid]
            return permuted_tokens_grad.to(orig_dtype), None

        # probs is not None
        num_experts = probs.shape[1]
        probs_cpu = probs.float().cpu()
        permuted_tokens_f = permuted_tokens.float()
        permuted_tokens_cpu = permuted_tokens_f.cpu()

        # Step 1: scatter unpermuted grad to permuted positions
        permuted_tokens_grad = torch.zeros(total_length, hidden_size,
                                           dtype=torch.float32, device=unpermuted_tokens_grad.device)
        for i in range(total_length):
            tid = permute_token_id_cpu[i].item()
            oi = out_index_cpu[i].item()
            permuted_tokens_grad[oi] = grad_f[tid]

        # Step 2: compute prob_grad
        if not drop_and_pad:
            # paddedMode=false
            topK = total_length // tokens_num
            # permutedProbsGrad = permutedTokensGrad * permutedTokens
            permuted_probs_grad = permuted_tokens_grad * permuted_tokens_f
            # probsGradExpertOrder = reduce sum over hidden_size, per position
            probs_grad_expert_order = permuted_probs_grad.sum(dim=-1)  # [total_length]

            # Build probs_grad using routing_map scatter
            probs_grad = torch.zeros(tokens_num, num_experts,
                                     dtype=torch.float32, device=unpermuted_tokens_grad.device)
            if routing_map is not None:
                routing_map_cpu = routing_map.cpu()
                # Scatter: for each position i, find the corresponding (token, expert) via routing_map
                idx = 0
                for t in range(tokens_num):
                    for e in range(num_experts):
                        if routing_map_cpu[t, e].item():
                            if idx < total_length:
                                probs_grad[t, e] = probs_grad_expert_order[idx].item()
                                idx += 1

            # Compute final permuted_tokens_grad_out = prob * permuted_tokens_grad
            permuted_probs = []
            if routing_map is not None:
                routing_map_cpu = routing_map.cpu()
                for t in range(tokens_num):
                    for e in range(num_experts):
                        if routing_map_cpu[t, e].item():
                            permuted_probs.append(probs_cpu[t, e].item())
            if len(permuted_probs) > 0:
                permuted_probs_tensor = torch.tensor(permuted_probs,
                                                     dtype=torch.float32,
                                                     device=unpermuted_tokens_grad.device)
                permuted_tokens_grad = permuted_probs_tensor.unsqueeze(-1) * permuted_tokens_grad
        else:
            # paddedMode=true
            capacity = total_length // num_experts
            # permutedProbsGrad = permutedTokensGrad * permutedTokens
            permuted_probs_grad = permuted_tokens_grad * permuted_tokens_f
            # probsGradExpertOrder = reduce sum over hidden_size
            probs_grad_expert_order = permuted_probs_grad.sum(dim=-1)  # [total_length]

            # Build probs_grad
            probs_grad = torch.zeros(tokens_num, num_experts,
                                     dtype=torch.float32, device=unpermuted_tokens_grad.device)
            for i in range(total_length):
                tid = permute_token_id_cpu[i].item()
                oi = out_index_cpu[i].item()
                expert_id = oi // capacity
                probs_grad[tid, expert_id] = probs_grad_expert_order[oi].item()

            # Compute final permuted_tokens_grad_out
            for i in range(total_length):
                tid = permute_token_id_cpu[i].item()
                oi = out_index_cpu[i].item()
                expert_id = oi // capacity
                prob_val = probs_cpu[tid, expert_id].item()
                permuted_tokens_grad[oi] = prob_val * permuted_tokens_grad[oi]

        return permuted_tokens_grad.to(orig_dtype), probs_grad.to(probs.dtype)


class ModelNew(nn.Module):
    """Custom kernel version."""
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, unpermuted_tokens_grad, out_index, permute_token_id,
                routing_map, permuted_tokens, probs, drop_and_pad, restore_shape):
        tokens_num = unpermuted_tokens_grad.shape[0]
        hidden_size = unpermuted_tokens_grad.shape[1]
        total_length = out_index.shape[0]

        if probs is not None:
            num_experts = probs.shape[1]
        else:
            num_experts = 0

        # Pre-allocate outputs
        out_length = total_length

        permuted_tokens_grad = torch.zeros(out_length, hidden_size,
                                           dtype=unpermuted_tokens_grad.dtype,
                                           device=unpermuted_tokens_grad.device)

        if probs is not None:
            probs_grad = torch.zeros(tokens_num, num_experts,
                                     dtype=probs.dtype,
                                     device=unpermuted_tokens_grad.device)
        else:
            probs_grad = torch.zeros(0, dtype=unpermuted_tokens_grad.dtype,
                                     device=unpermuted_tokens_grad.device)

        result = kernel_gen_ops.moe_token_unpermute_with_routing_map_grad(
            unpermuted_tokens_grad, out_index, permute_token_id,
            routing_map, permuted_tokens, probs,
            drop_and_pad, restore_shape,
            permuted_tokens_grad, probs_grad
        )
        return result
