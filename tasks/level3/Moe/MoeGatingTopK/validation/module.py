import torch
import torch_npu
import numpy as np

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, finished_optional: torch.Tensor, k: int):
        def softmax_func(x_np, axis=None):
            x_np = x_np.astype(np.float32)
            x_max = x_np.max(axis=axis, keepdims=True)
            x_sub = x_np - x_max
            y = np.exp(x_sub)
            x_sum = y.sum(axis=axis, keepdims=True)
            ans = y / x_sum
            return ans, x_max, x_sum

        gating_np = x.to(torch.float32).cpu().numpy()
        num_expert = gating_np.shape[-1]

        softmax, _, _ = softmax_func(gating_np, -1)

        indices = np.argsort(-softmax, axis=-1, kind='stable')
        indices = indices[:, :k]
        out = np.take_along_axis(softmax, indices, axis=-1)

        if finished_optional is not None:
            finished_optional_np = finished_optional.cpu().numpy()
            finished_optional_np = finished_optional_np.reshape(finished_optional_np.shape[0], 1)
            finished_optional_np = np.tile(finished_optional_np, (1, k))
            indices = np.where(finished_optional_np, num_expert, indices)

        source_row_out = np.arange(out.shape[0] * out.shape[1], dtype=np.int32).reshape(
            [out.shape[1], out.shape[0]]).transpose(1, 0)

        return [torch.from_numpy(out).to(x.device, dtype=x.dtype), torch.from_numpy(indices).to(x.device, dtype=torch.int32), torch.from_numpy(source_row_out).to(x.device)]

class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor, finished_optional: torch.Tensor, k: int):
        import kernel_gen_ops
        return list(kernel_gen_ops.moe_gating_top_k(x, finished_optional, k))
