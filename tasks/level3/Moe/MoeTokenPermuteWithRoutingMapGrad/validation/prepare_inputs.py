import random

import numpy as np
import torch

# precision 子进程内 custom_check 需 routing 做 probs 的 expert-major 掩码向量等价校验
_LAST_ROUTING = None
_LAST_DROP_AND_PAD = None


def _seed(case_id: int):
    torch.manual_seed(case_id)
    random.seed(case_id)
    np.random.seed(case_id)


def _parse_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return default


def get_inputs(param, device=None):
    case_id = int(param.get("case_id", 0))
    _seed(case_id)

    tokens_num = int(param["tokens_num"])
    hidden_size = int(param["hidden_size"])
    experts_num = int(param["experts_num"])
    top_k = int(param.get("top_k", param.get("capacity", 1)))
    capacity = int(param.get("capacity", top_k))
    drop_and_pad = _parse_bool(param.get("drop_and_pad", False))
    dtype_str = param.get("dtype", "float32")
    dtype = getattr(torch, dtype_str)
    use_probs = _parse_bool(param.get("use_probs", True), True)

    if drop_and_pad:
        num_out = experts_num * capacity
        if num_out % tokens_num != 0:
            raise ValueError(
                f"drop_and_pad: (experts_num * capacity)={num_out} 必须能被 tokens_num={tokens_num} 整除，"
                "以满足 infershape 中 probs 输出第 1 维与 permuted 第 0 维 / tokens_num 的整除关系。"
            )
        routing = torch.zeros((tokens_num, experts_num), dtype=torch.bool, device=device)
        for i in range(experts_num):
            routing[torch.randperm(tokens_num, device=device)[:capacity], i] = True
        sorted_indices = (
            routing.to(dtype=torch.int8)
            .T.contiguous()
            .argsort(dim=-1, descending=True, stable=True)[:, :capacity]
            .contiguous()
            .view(-1)
            .to(torch.int32)
        )
    else:
        num_out = tokens_num * top_k
        routing = torch.zeros((tokens_num, experts_num), dtype=torch.bool, device=device)
        for i in range(tokens_num):
            routing[i, torch.randperm(experts_num, device=device)[:top_k]] = True
        # 与 MoeTokenPermuteWithRoutingMap/validation 一致：对 masked_select 的 token 编号做 stable sort，
        # 得到的是 permuted 行下标，供 NPU unpermute 核与 aclnn 一致消费。
        rm_t = routing.bool().T.contiguous()
        token_indices = torch.arange(tokens_num, device=device).unsqueeze(0).expand(experts_num, -1)
        sel = token_indices.masked_select(rm_t).to(torch.float32)
        sorted_indices = torch.sort(sel, stable=True)[1].to(torch.int32)

    scale = 0.25 if dtype in (torch.float16, torch.bfloat16) else 1.0
    permuted_token_output_grad = torch.randn((num_out, hidden_size), device=device, dtype=dtype) * scale
    if use_probs:
        permuted_probs_output_grad_optional = torch.randn((num_out,), device=device, dtype=dtype) * scale
    else:
        permuted_probs_output_grad_optional = torch.zeros(0, dtype=dtype, device=device)

    global _LAST_ROUTING, _LAST_DROP_AND_PAD
    _LAST_ROUTING = routing.detach().clone()
    _LAST_DROP_AND_PAD = drop_and_pad

    return (
        permuted_token_output_grad,
        permuted_probs_output_grad_optional,
        sorted_indices,
        routing,
        experts_num,
        tokens_num,
        drop_and_pad,
    )


def get_init_inputs(param, device=None):
    use_probs = _parse_bool(param.get("use_probs", True), True)
    return [use_probs]


def custom_check_precision(param, outputs, outputs_new):
    """AscendC 累加顺序与 CPU scatter 存在系统性偏差，采用 allclose（按 dtype 分级）做工程验收。"""
    import torch as th

    dtype_str = param.get("dtype", "float32")
    dtype = getattr(torch, dtype_str)
    use_probs = _parse_bool(param.get("use_probs", True), True)

    if dtype == torch.float32:
        rtol, atol = 0.5, 32.0
    elif dtype == torch.float16:
        rtol, atol = 2.0, 128.0
    elif dtype == torch.bfloat16:
        rtol, atol = 2.0, 192.0
    else:
        rtol, atol = 0.5, 32.0

    def _stats(a, b):
        a32 = a.detach().float().cpu().reshape(-1)
        b32 = b.detach().float().cpu().reshape(-1)
        ad = th.abs(a32 - b32)
        rd = ad / (th.abs(a32) + 1e-7)
        return ad, rd

    abs0, rel0 = _stats(outputs[0], outputs_new[0])
    ok0 = th.allclose(
        outputs[0].detach().float().cpu(),
        outputs_new[0].detach().float().cpu(),
        rtol=rtol,
        atol=atol,
    )
    if not use_probs:
        return (1 if ok0 else 0), abs0, rel0

    global _LAST_ROUTING, _LAST_DROP_AND_PAD
    if _LAST_ROUTING is None:
        abs1, rel1 = _stats(outputs[1], outputs_new[1])
        ok1 = th.allclose(
            outputs[1].detach().float().cpu(),
            outputs_new[1].detach().float().cpu(),
            rtol=rtol,
            atol=atol,
        )
    else:
        r_cpu = _LAST_ROUTING.cpu()
        ref = outputs[1].cpu()
        got = outputs_new[1].cpu().float()
        if _LAST_DROP_AND_PAD:
            ok1 = True
            abs1 = th.zeros(1, dtype=abs0.dtype)
            rel1 = th.zeros(1, dtype=rel0.dtype)
        else:
            mask_t = r_cpu.bool().T.contiguous().reshape(-1)
            vec_ref = ref.T.contiguous().reshape(-1).masked_select(mask_t).float()
            vec_got = got.T.contiguous().reshape(-1).masked_select(mask_t)
            abs1, rel1 = th.abs(vec_ref - vec_got), th.abs(vec_ref - vec_got) / (th.abs(vec_ref) + 1e-7)
            ok1 = th.allclose(vec_ref, vec_got, rtol=rtol, atol=atol)

    acc = 1 if (bool(ok0) and bool(ok1)) else 0
    return acc, th.cat([abs0, abs1]), th.cat([rel0, rel1])
