# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import torch
import torch.nn.functional as F
import numpy as np
import warnings
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


@register("ascend_function_mhc_pre_sinkhorn")
class MethodTorchMhcPreSinkhornApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTorchMhcPreSinkhornApi, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        h_res = input_data.kwargs["h_res"]
        eps = input_data.kwargs["eps"]
        num_iters = input_data.kwargs["num_iters"]
        out_flag = input_data.kwargs["out_flag"]
        
        h_res_torch = h_res.to(torch.float32)
        
        # Check for special values (inf, nan)
        if torch.any(torch.isinf(h_res_torch)) or torch.any(torch.isnan(h_res_torch)):
            warnings.warn(
                "Input contains inf or nan values. Output will contain nan.",
                RuntimeWarning
            )
        
        # Initialize intermediate result storage
        if out_flag == 1:
            if h_res_torch.ndim == 3:
                batch_dims = h_res_torch.shape[0]
            else:
                batch_dims = h_res_torch.shape[0] * h_res_torch.shape[1]
            
            if h_res_torch.ndim == 3:
                n = h_res_torch.shape[1]
            else:
                n = h_res_torch.shape[2]
            
            norm_out = torch.zeros((2 * num_iters, n, n, batch_dims), 
                               dtype=h_res_torch.dtype, device=h_res_torch.device)
            sum_out = torch.zeros((2 * num_iters, n, batch_dims), 
                               dtype=h_res_torch.dtype, device=h_res_torch.device)
        else:
            norm_out = None
            sum_out = None
        
        # First iteration: Softmax + Column normalization
        # Softmax along rows (dim=-1)
        current = F.softmax(h_res_torch, dim=-1)
        
        if out_flag == 1:
            norm_out[0] = self._transpose_to_norm_out_format(current)
        
        # Column normalization
        current, col_sums = self._col_normalize(current, eps=eps)
        
        if out_flag == 1:
            sum_out[1] = self._transpose_to_sum_out_format(col_sums)
            norm_out[1] = self._transpose_to_norm_out_format(current)
        
        # Subsequent iterations: Row normalization + Column normalization
        for iter_idx in range(1, num_iters):
            # Row normalization
            current, row_sums = self._row_normalize(current, eps=eps)
            
            if out_flag == 1:
                sum_out[2 * iter_idx] = self._transpose_to_sum_out_format(row_sums)
                norm_out[2 * iter_idx] = self._transpose_to_norm_out_format(current)
            
            # Column normalization
            current, col_sums = self._col_normalize(current, eps=eps)
            
            if out_flag == 1:
                sum_out[2 * iter_idx + 1] = self._transpose_to_sum_out_format(col_sums)
                norm_out[2 * iter_idx + 1] = self._transpose_to_norm_out_format(current)
        
        # Convert back to original dtype
        result = current.to(h_res.dtype)
        
        # Handle optional outputs
        if out_flag == 1:
            return result, norm_out, sum_out
        else:
            return result, None, None
    
    def _row_normalize(self, x, eps=1e-6):
        """
        Normalize rows of the matrix using torch operations.
        
        Args:
            x: Input tensor
            eps: Small constant to prevent division by zero
        
        Returns:
            Tuple of (normalized tensor, row sums)
        """
        # Compute row sums using matrix multiplication with ones
        # This is equivalent to torch.sum(x, dim=-1, keepdims=True)
        # but can be implemented using torch.nn.functional.linear
        if x.ndim == 3:
            # Shape: [batch, n, n]
            # Create ones vector for row sum: [n, 1]
            ones = torch.ones(x.shape[-1], 1, dtype=x.dtype, device=x.device)
            # Reshape x to [batch*n, n] for matrix multiplication
            x_reshaped = x.reshape(-1, x.shape[-1])
            # Compute row sums using linear: [batch*n, n] @ [n, 1] -> [batch*n, 1]
            row_sums = F.linear(x_reshaped, ones.t())
            # Reshape back to [batch, n, 1]
            row_sums = row_sums.reshape(x.shape[0], x.shape[1], 1)
        else:
            # Shape: [B, S, n, n]
            # Create ones vector for row sum: [n, 1]
            ones = torch.ones(x.shape[-1], 1, dtype=x.dtype, device=x.device)
            # Reshape x to [B*S*n, n] for matrix multiplication
            x_reshaped = x.reshape(-1, x.shape[-1])
            # Compute row sums using linear: [B*S*n, n] @ [n, 1] -> [B*S*n, 1]
            row_sums = F.linear(x_reshaped, ones.t())
            # Reshape back to [B, S, n, 1]
            row_sums = row_sums.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
        
        # Add eps to prevent division by zero
        row_sums_eps = row_sums + eps
        
        # Normalize
        x_normalized = x / row_sums_eps
        
        return x_normalized, row_sums
    
    def _col_normalize(self, x, eps=1e-6):
        """
        Normalize columns of the matrix using torch operations.
        
        Args:
            x: Input tensor
            eps: Small constant to prevent division by zero
        
        Returns:
            Tuple of (normalized tensor, column sums)
        """
        # Compute column sums using matrix multiplication with ones
        # This is equivalent to torch.sum(x, dim=-2, keepdims=True)
        # but can be implemented using torch.nn.functional.linear
        if x.ndim == 3:
            # Shape: [batch, n, n]
            # Create ones vector for column sum: [n, 1]
            ones = torch.ones(x.shape[-2], 1, dtype=x.dtype, device=x.device)
            # Reshape x to [batch, n, n] -> transpose to [batch, n, n]
            x_transposed = x.transpose(-1, -2)
            # Reshape to [batch*n, n] for matrix multiplication
            x_reshaped = x_transposed.reshape(-1, x_transposed.shape[-1])
            # Compute column sums using linear: [batch*n, n] @ [n, 1] -> [batch*n, 1]
            col_sums = F.linear(x_reshaped, ones.t())
            # Reshape back to [batch, 1, n]
            col_sums = col_sums.reshape(x.shape[0], 1, x.shape[2])
        else:
            # Shape: [B, S, n, n]
            # Create ones vector for column sum: [n, 1]
            ones = torch.ones(x.shape[-2], 1, dtype=x.dtype, device=x.device)
            # Reshape x to [B, S, n, n] -> transpose to [B, S, n, n]
            x_transposed = x.transpose(-1, -2)
            # Reshape to [B*S*n, n] for matrix multiplication
            x_reshaped = x_transposed.reshape(-1, x_transposed.shape[-1])
            # Compute column sums using linear: [B*S*n, n] @ [n, 1] -> [B*S*n, 1]
            col_sums = F.linear(x_reshaped, ones.t())
            # Reshape back to [B, S, 1, n]
            col_sums = col_sums.reshape(x.shape[0], x.shape[1], 1, x.shape[3])
        
        # Add eps to prevent division by zero
        col_sums_eps = col_sums + eps
        
        # Normalize
        x_normalized = x / col_sums_eps
        
        return x_normalized, col_sums
    
    def _transpose_to_norm_out_format(self, x):
        """
        Transpose input to norm_out format.
        
        Args:
            x: Input tensor with shape [batch_dims, n, n] or [B, S, n, n]
        
        Returns:
            Transposed tensor with shape [n, n, batch_dims]
        """
        if x.ndim == 3:
            # Shape: [batch_dims, n, n] -> [n, n, batch_dims]
            return x.permute(1, 2, 0)
        else:
            # Shape: [B, S, n, n] -> [n, n, B*S]
            batch_dims = x.shape[0] * x.shape[1]
            n = x.shape[2]
            return x.reshape(batch_dims, n, n).permute(1, 2, 0)
    
    def _transpose_to_sum_out_format(self, x):
        """
        Transpose input to sum_out format.
        
        Args:
            x: Input tensor with shape [batch_dims, n, 1] or [B, S, n, 1]
        
        Returns:
            Transposed tensor with shape [n, batch_dims]
        """
        if x.ndim == 3:
            # Shape: [batch_dims, n, 1] -> [n, batch_dims]
            return x.squeeze(-1).permute(1, 0)
        else:
            # Shape: [B, S, n, 1] -> [n, B*S]
            batch_dims = x.shape[0] * x.shape[1]
            n = x.shape[2]
            return x.reshape(batch_dims, n, 1).squeeze(-1).permute(1, 0)
