import torch


def hc_post_golden(x, residual, post, comb):
    data_type = x.dtype
    x = x.float()
    residual = residual.float()
    post = post.float()
    comb = comb.float()
    out = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=x.dim() - 1)
    return out.to(data_type).reshape(list(residual.shape))
