import torch
import torch.nn as nn


def _batch_idx_for_pt(pt_idx: int, ibc: torch.Tensor) -> int:
    """与 kernel 中 indices_batch_cnt 累计逻辑一致。"""
    b = ibc.numel()
    pt_cnt = int(ibc[0].item())
    bs_idx = 0
    for k in range(1, b):
        if pt_idx >= pt_cnt:
            bs_idx = k
            pt_cnt += int(ibc[k].item())
    return bs_idx


def _feature_range(bs_idx: int, fbc: torch.Tensor):
    """
    fbc 长度 B+1，与 kernel / ops-cv golden 一致：
    batch i 点数为 fbc[i]，循环用 fbc[k+1] 更新 end。
    """
    features_batch_start_idx = 0
    features_batch_end_idx = int(fbc[0].item())
    for k in range(bs_idx):
        features_batch_start_idx += int(fbc[k].item())
        features_batch_end_idx = features_batch_start_idx + int(fbc[k + 1].item())
    return features_batch_start_idx, features_batch_end_idx


def _stack_group_points_reference(
    features: torch.Tensor,
    features_batch_cnt: torch.Tensor,
    indices: torch.Tensor,
    indices_batch_cnt: torch.Tensor,
) -> torch.Tensor:
    """
    features: (N, C)；fbc / ibc: 长度 B 的 int32；
    indices: (M, nsample)；输出 (M, C, nsample)，与 infershape / kernel 索引顺序一致。
    """
    n, c = features.shape
    m, nsample = indices.shape
    b = indices_batch_cnt.numel()
    standard = m * c * nsample
    out = features.new_zeros((m, c, nsample))
    feat_flat = features.reshape(-1).float()

    fbc = features_batch_cnt.cpu()
    ibc = indices_batch_cnt.cpu()
    ind = indices.cpu().long()

    for pt_idx in range(m):
        for c_idx in range(c):
            for sample_idx in range(nsample):
                index = pt_idx * c * nsample + c_idx * nsample + sample_idx
                if index > standard:
                    continue
                bs_idx = _batch_idx_for_pt(pt_idx, ibc)
                fs, fe = _feature_range(bs_idx, fbc)
                tmp_cin = pt_idx * nsample + sample_idx
                if tmp_cin >= m * nsample:
                    continue
                cin = int(ind[pt_idx, sample_idx].item())
                in_idx = cin * c + c_idx
                if in_idx < fe * c and in_idx < n * c - fs * c:
                    fs_idx = in_idx + fs * c
                    if 0 <= fs_idx < n * c:
                        out[pt_idx, c_idx, sample_idx] = feat_flat[fs_idx]

    return out.to(dtype=features.dtype)


class Model(nn.Module):
    """CPU 金标准：与 stack_group_points kernel 中 Gather 逻辑一致。"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        features: torch.Tensor,
        features_batch_cnt: torch.Tensor,
        indices: torch.Tensor,
        indices_batch_cnt: torch.Tensor,
    ) -> torch.Tensor:
        return _stack_group_points_reference(
            features,
            features_batch_cnt,
            indices,
            indices_batch_cnt,
        )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        features: torch.Tensor,
        features_batch_cnt: torch.Tensor,
        indices: torch.Tensor,
        indices_batch_cnt: torch.Tensor,
    ) -> torch.Tensor:
        import kernel_gen_ops

        return kernel_gen_ops.stack_group_points(
            features,
            features_batch_cnt,
            indices,
            indices_batch_cnt,
        )
