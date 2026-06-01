import numpy as np
import torch


def softmax_func(x, axis=None):
    if "float16" in x.dtype.name:
        x = x.astype(np.float32)
    x_max = x.max(axis=axis, keepdims=True)
    y = np.exp(x - x_max)
    return y / y.sum(axis=axis, keepdims=True), x_max, y.sum(axis=axis, keepdims=True)


def softplus(x, beta=1.0):
    return np.log(1 + np.exp(beta * x)) / beta


def moe_gating_top_k_hash_golden(x, input_ids, tid2eid, bias, k, k_group=1, group_count=1,
                                 group_select_mode=0, renorm=0, norm_type=0, y2_flag=False,
                                 routed_scaling_factor=1.0, eps=1e-20):
    del k_group, group_count, group_select_mode, renorm, y2_flag
    ori_dtype = x.dtype
    x = x.float().numpy()
    if bias is not None:
        bias = bias.astype("float32")
    if norm_type == 0:
        x, _, _ = softmax_func(x, -1)
    elif norm_type == 1:
        x = 1 / (1 + np.exp(-x))
    else:
        x = np.sqrt(softplus(x))
    original_x = x
    if bias is not None:
        x = x + bias
    if tid2eid is not None and input_ids is not None:
        indices = tid2eid[input_ids]
    else:
        indices = np.argsort(-x, axis=-1, kind="stable")[:, :k]
    y = np.take_along_axis(original_x, indices, axis=1)
    if norm_type != 0:
        y /= (np.sum(y, axis=-1, keepdims=True) + eps)
    y *= routed_scaling_factor
    return torch.from_numpy(y).to(ori_dtype), torch.from_numpy(indices.astype(np.int32))
