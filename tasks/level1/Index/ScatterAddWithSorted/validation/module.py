import torch
import torch.nn as nn
import kernel_gen_ops

class Model(nn.Module):
    """
    Model implementing the ScatterAddWithSorted operator logic (Golden).
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, var: torch.Tensor, value: torch.Tensor, sorted_index: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            var: Tensor to be updated [N, D] or [..., N, D]
            value: Updates tensor.
            sorted_index: Indices into var [K]
            pos: Indices into value [K], optional.
        """
        out = var.clone()
        
        inner_dim = var.shape[-1]
        out_flat = out.view(-1, inner_dim)
        value_flat = value.view(-1, inner_dim)
        
        sorted_index_long = sorted_index.long()
        
        if pos is not None:
            pos_long = pos.long()
            updates = value_flat[pos_long]
        else:
            updates = value_flat[:sorted_index.numel()]
            
        out_flat.index_add_(0, sorted_index_long, updates)
        
        return out_flat.view(var.shape)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, var: torch.Tensor, value: torch.Tensor, sorted_index: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        return kernel_gen_ops.scatter_add_with_sorted(var, value, sorted_index, pos)