import torch
import torch.nn as nn


def stack_ball_query_cpu(
    xyz: torch.Tensor,
    center_xyz: torch.Tensor,
    xyz_batch_cnt: torch.Tensor,
    center_xyz_batch_cnt: torch.Tensor,
    max_radius: float,
    sample_num: int,
) -> torch.Tensor:
    """
    xyz: [3, N] planar；center_xyz: [M, 3]。
    与两条 kernel 分支（FP32 / FP16）在「顺序扫描 + 平方距离与 max_radius^2 比较」语义上一致；
    距离在 float32 中计算（FP16 输入先提升），避免半精度边界与参考不一致。
    无命中：首元素 -1，其余填 0；有命中：输出 batch 内局部下标，不足槽位用第一个命中下标填充。
    """
    m = center_xyz.shape[0]
    max_r2 = max_radius * max_radius

    xb = xyz_batch_cnt.detach().cpu().tolist()
    cb = center_xyz_batch_cnt.detach().cpu().tolist()
    batch_size = len(xb)

    def center_batch(global_idx: int):
        cum = 0
        for b in range(batch_size):
            if global_idx < cum + cb[b]:
                return b, global_idx - cum
            cum += cb[b]
        return batch_size - 1, 0

    def xyz_off(b: int):
        return sum(xb[:b])

    out = torch.empty(m * sample_num, device=xyz.device, dtype=torch.int32)
    cx = center_xyz[:, 0].to(torch.float32)
    cy = center_xyz[:, 1].to(torch.float32)
    cz = center_xyz[:, 2].to(torch.float32)
    px = xyz[0].to(torch.float32)
    py = xyz[1].to(torch.float32)
    pz = xyz[2].to(torch.float32)

    for mi in range(m):
        b, _ = center_batch(mi)
        cxv, cyv, czv = cx[mi].item(), cy[mi].item(), cz[mi].item()
        off = xyz_off(b)
        cnt = xb[b]
        collected = []
        for i in range(cnt):
            dx = px[off + i].item() - cxv
            dy = py[off + i].item() - cyv
            dz = pz[off + i].item() - czv
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < max_r2:
                # kernel 输出为当前 batch 内点的局部下标，非全局堆叠下标
                collected.append(i)
                if len(collected) >= sample_num:
                    break

        if len(collected) == 0:
            out[mi * sample_num] = -1
            for s in range(1, sample_num):
                out[mi * sample_num + s] = 0
        else:
            fr = collected[0]
            for s in range(sample_num):
                if s < len(collected):
                    out[mi * sample_num + s] = collected[s]
                else:
                    out[mi * sample_num + s] = fr

    return out


class Model(nn.Module):
    def __init__(self, max_radius: float, sample_num: int):
        super().__init__()
        self.max_radius = max_radius
        self.sample_num = sample_num

    def forward(self, xyz, center_xyz, xyz_batch_cnt, center_xyz_batch_cnt):
        return stack_ball_query_cpu(
            xyz,
            center_xyz,
            xyz_batch_cnt,
            center_xyz_batch_cnt,
            self.max_radius,
            self.sample_num,
        )


class ModelNew(nn.Module):
    def __init__(self, max_radius: float, sample_num: int):
        super().__init__()
        self.max_radius = max_radius
        self.sample_num = sample_num

    def forward(self, xyz, center_xyz, xyz_batch_cnt, center_xyz_batch_cnt):
        import kernel_gen_ops

        return kernel_gen_ops.stack_ball_query(
            xyz,
            center_xyz,
            xyz_batch_cnt,
            center_xyz_batch_cnt,
            self.max_radius,
            self.sample_num,
        )
