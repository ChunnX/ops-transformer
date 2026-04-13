import torch
import torch_npu
import numpy as np

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, finished_optional: torch.Tensor, k: int, renorm: int):
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

        softmax_out, _, _ = softmax_func(gating_np, -1)

        indices = np.argsort(-softmax_out, axis=-1, kind='stable')
        indices = indices[:, :k]
        out = np.take_along_axis(softmax_out, indices, axis=-1)

        if renorm == 1:
            # 重新归一化
            out_sum = out.sum(axis=-1, keepdims=True) + 1e-10
            out = out / out_sum

        if finished_optional is not None:
            finished_optional_np = finished_optional.cpu().numpy()
            finished_optional_np = finished_optional_np.reshape(finished_optional_np.shape[0], 1)
            finished_optional_np = np.tile(finished_optional_np, (1, k))
            indices = np.where(finished_optional_np, num_expert, indices)

        return [torch.from_numpy(out).to(x.device, dtype=x.dtype), \
                torch.from_numpy(indices).to(x.device, dtype=torch.int32), \
                torch.from_numpy(softmax_out).to(x.device, dtype=torch.float32)]

class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor, finished_optional: torch.Tensor, k: int, renorm: int):
        import kernel_gen_ops
        return list(kernel_gen_ops.moe_gating_top_k_softmax_v2(x, finished_optional, k, renorm))