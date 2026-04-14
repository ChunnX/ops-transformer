import ast

import torch


def _compress_out_rows(act_cumsum, compress_block_size: int, compress_stride: int) -> int:
    pre = 0
    total = 0
    for end in act_cumsum:
        cur = end - pre
        if cur >= compress_block_size:
            total += (cur - compress_block_size + compress_stride) // compress_stride
        pre = end
    return total


def get_init_inputs(param, device=None):
    return []


def _parse_list(s):
    if isinstance(s, list):
        return s
    return ast.literal_eval(str(s).strip())


def get_inputs(row, device=None):
    """列与 NsaCompress 一致；额外构造与 forward 输出同形状的 output_grad。"""
    dtype = getattr(torch, str(row["dtype"]).strip())
    t = int(row["T"])
    n = int(row["N"])
    d = int(row["D"])
    cbs = int(row["compress_block_size"])
    cs = int(row["compress_stride"])
    cum = [int(x) for x in _parse_list(row["act_seq_len_cumsum"])]
    cst = int(row.get("act_seq_len_type", 0))

    torch.manual_seed(int(row.get("seed", 42)))
    input_tensor = torch.randn(t, n, d, device=device, dtype=dtype)
    weight = torch.randn(cbs, n, device=device, dtype=dtype)
    act_seq = torch.tensor(cum, device=device, dtype=torch.int64)

    m = _compress_out_rows(cum, cbs, cs)
    output_grad = torch.randn(m, n, d, device=device, dtype=dtype)

    return (
        output_grad,
        input_tensor,
        weight,
        act_seq,
        cbs,
        cs,
        cst,
    )


def custom_check_precision(row, ref_output, test_output, atol=5e-2, rtol=5e-2):
    refs = list(ref_output) if isinstance(ref_output, (list, tuple)) else [ref_output]
    tests = list(test_output) if isinstance(test_output, (list, tuple)) else [test_output]
    all_abs = []
    all_rel = []
    ok = True
    for ref, tst in zip(refs, tests):
        if tst.dtype == torch.bool:
            ref = ref.to(torch.int)
            tst = tst.to(torch.int)
        abs_diff = torch.abs(ref.float() - tst.float()).reshape(-1)
        rel_diff = abs_diff / (torch.abs(ref.float().reshape(-1)) + 1e-7)
        all_abs.append(abs_diff)
        all_rel.append(rel_diff)
        part_ok = torch.all((abs_diff <= atol) | (rel_diff <= rtol)) and torch.isfinite(ref).all() and torch.isfinite(
            tst
        ).all()
        ok = ok and bool(part_ok)
    acc = 1 if ok else 0
    return acc, torch.cat(all_abs), torch.cat(all_rel)
