import ast

import torch


def get_init_inputs(param, device=None):
    return []


def _parse_list(s):
    if isinstance(s, list):
        return s
    return ast.literal_eval(str(s).strip())


def get_inputs(row, device=None):
    """列：dtype;T;N;D;compress_block_size;compress_stride;act_seq_len_cumsum（cumsum 列表）"""
    dtype = getattr(torch, str(row["dtype"]).strip())
    t = int(row["T"])
    n = int(row["N"])
    d = int(row["D"])
    cbs = int(row["compress_block_size"])
    cs = int(row["compress_stride"])
    cum = _parse_list(row["act_seq_len_cumsum"])
    cum = [int(x) for x in cum]

    torch.manual_seed(int(row.get("seed", 42)))
    input_tensor = torch.randn(t, n, d, device=device, dtype=dtype)
    weight = torch.randn(cbs, n, device=device, dtype=dtype)
    act_seq = torch.tensor(cum, device=device, dtype=torch.int64)
    cst = int(row.get("act_seq_len_type", 0))
    return (
        input_tensor,
        weight,
        act_seq,
        cbs,
        cs,
        cst,
    )


def custom_check_precision(row, ref_output, test_output, atol=5e-2, rtol=5e-2):
    """树形规约与 cast 后与 NPU 仍可能有微小差异，略放宽阈值。"""
    abs_diff = torch.abs(ref_output.float() - test_output.float()).reshape(-1)
    rel_diff = abs_diff / (torch.abs(ref_output.float().reshape(-1)) + 1e-7)
    ok = torch.all(
        (abs_diff <= atol) | (rel_diff <= rtol)
    ) and torch.isfinite(ref_output).all() and torch.isfinite(test_output).all()
    acc = 1 if bool(ok) else 0
    return acc, abs_diff, rel_diff
