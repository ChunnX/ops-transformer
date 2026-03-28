import torch


def _build_xyz_planar(batch_xyz_counts, dtype, device, base_sep=10.0):
    """xyz: [3, N] planar layout (all x, all y, all z), 与 kernel 一致。"""
    counts = [int(batch_xyz_counts[i]) for i in range(len(batch_xyz_counts))]
    n = sum(counts)
    xyz = torch.zeros(3, n, device=device, dtype=dtype)
    off = 0
    for b, cnt in enumerate(counts):
        for i in range(cnt):
            xyz[0, off + i] = b * base_sep + i * 0.15
            xyz[1, off + i] = 0.02 * i
            xyz[2, off + i] = 0.01 * (b + 1)
        off += cnt
    return xyz


def _build_centers_stacked(batch_center_counts, dtype, device, base_sep=10.0):
    """center_xyz: [M, 3]，与 kernel 交错存储一致。"""
    counts = [int(batch_center_counts[i]) for i in range(len(batch_center_counts))]
    m = sum(counts)
    c = torch.zeros(m, 3, device=device, dtype=dtype)
    off = 0
    for b, cnt in enumerate(counts):
        for i in range(cnt):
            c[off + i, 0] = b * base_sep + 0.05 + i * 0.1
            c[off + i, 1] = 0.0
            c[off + i, 2] = 0.01 * (b + 1)
        off += cnt
    return c


def get_inputs(param, device=None):
    dtype_str = param.get("dtype_xyz", "float32")
    dtype = getattr(torch, dtype_str)

    bx0 = int(param["batch_xyz_0"])
    bx1 = int(param["batch_xyz_1"])
    bc0 = int(param["batch_center_0"])
    bc1 = int(param["batch_center_1"])

    xyz = _build_xyz_planar([bx0, bx1], dtype, device)
    center_xyz = _build_centers_stacked([bc0, bc1], dtype, device)

    xyz_batch_cnt = torch.tensor([bx0, bx1], device=device, dtype=torch.int32)
    center_xyz_batch_cnt = torch.tensor([bc0, bc1], device=device, dtype=torch.int32)

    return [xyz, center_xyz, xyz_batch_cnt, center_xyz_batch_cnt]


def get_init_inputs(param, device=None):
    max_radius = float(param.get("max_radius", 2.0))
    sample_num = int(param.get("sample_num", 4))
    return [max_radius, sample_num]


def custom_check_precision(param, ref_output, test_output):
    ref = ref_output if isinstance(ref_output, torch.Tensor) else ref_output[0]
    tst = test_output if isinstance(test_output, torch.Tensor) else test_output[0]
    ref_i = ref.detach().cpu().to(torch.int64)
    tst_i = tst.detach().cpu().to(torch.int64)
    if not torch.equal(ref_i, tst_i):
        return 0, torch.abs(ref_i.float() - tst_i.float()).reshape(-1), torch.zeros(ref_i.numel())
    return 1, torch.zeros(ref_i.numel(), dtype=torch.float32), torch.zeros(ref_i.numel(), dtype=torch.float32)
