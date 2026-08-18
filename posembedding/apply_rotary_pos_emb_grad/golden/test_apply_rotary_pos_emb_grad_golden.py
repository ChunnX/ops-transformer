import os
import sys
from typing import Optional, Tuple

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TORCH_EXT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(ROOT))), "torch_extension")
if TORCH_EXT not in sys.path:
    sys.path.insert(0, TORCH_EXT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from apply_rotary_pos_emb_grad import _apply_rotary_pos_emb_backward
from cann_ops_transformer.ops.apply_rotary_pos_emb_grad import apply_rotary_pos_emb_grad as npu_apply_rotary_pos_emb_grad

import torch_npu  # noqa: F401  (registers torch.ops.npu.*)


ROTARY_MODE_HALF = "half"
LAYOUT_BSND = 1
LAYOUT_SBND = 2
LAYOUT_BNSD = 3
LAYOUT_TND = 4


CASES = [
    # {
    #     "name": "A_template_bsnd",
    #     "layout": LAYOUT_BSND,
    #     "s": 4,
    #     "b": 40960,
    #     "n_q": 1,
    #     "n_k": 1,
    #     "d": 100,
    #     "cos_b": 40960,
    #     "dtype": torch.float32,
    #     "seed": 0,
    #     "use_optional": True,
    # },
    {
        "name": "AB_template_bsnd",
        "layout": LAYOUT_SBND,
        "s": 4480,
        "b": 1,
        "n_q": 16,
        "n_k": 16,
        "d": 64,
        "cos_b": 1,
        "dtype": torch.float32,
        "seed": 0,
        "use_optional": True,
    },
]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_1, x_2 = x.chunk(2, dim=-1)
    return torch.cat([-x_2, x_1], dim=-1)


def _build_shapes(layout: int, s: int, b: int, n_q: int, n_k: int, d: int, cos_b: int):
    if layout == LAYOUT_BSND:
        return (b, s, n_q, d), (b, s, n_k, d), (cos_b, s, 1, d)
    if layout == LAYOUT_SBND:
        return (s, b, n_q, d), (s, b, n_k, d), (s, cos_b, 1, d)
    if layout == LAYOUT_BNSD:
        return (b, n_q, s, d), (b, n_k, s, d), (cos_b, 1, s, d)
    if layout == LAYOUT_TND:
        return (s, n_q, d), (s, n_k, d), (s, 1, d)
    raise ValueError(f"unsupported layout: {layout}")


def _make_cache(query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    return {
        "query": query,
        "key": key,
        "cos": cos,
        "sin": sin,
        "query_rotate": _rotate_half(query),
        "key_rotate": _rotate_half(key),
    }


def _golden_backward(
    grad_query_embed: torch.Tensor,
    grad_key_embed: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
):
    cache = _make_cache(query, key, cos, sin)
    return _apply_rotary_pos_emb_backward(cache, grad_query_embed, grad_key_embed)


def _golden_backward_required_only(
    grad_query_embed: torch.Tensor,
    grad_key_embed: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    # 与 kernel (BatchHalfGradContiguousVF) 精度模型对齐：中间寄存器 RegTensor<float>，
    # fp16/bf16 输入 upcast 到 fp32 做乘加累加，输出 downcast 回原 dtype。
    # 纯低精度中间计算会引入 kernel 不存在的额外舍入误差（fp16 ~2^-7、bf16 ~2^-4）。
    out_dt = grad_query_embed.dtype
    gq = grad_query_embed.float()
    gk = grad_key_embed.float()
    c = cos.float()
    s = sin.float()

    dq_1, dq_2 = gq.chunk(2, dim=-1)
    dk_1, dk_2 = gk.chunk(2, dim=-1)
    c_1, c_2 = c.chunk(2, dim=-1)
    s_1, s_2 = s.chunk(2, dim=-1)

    grad_query = torch.cat([c_1 * dq_1 + s_2 * dq_2, c_2 * dq_2 - s_1 * dq_1], dim=-1)
    grad_key = torch.cat([c_1 * dk_1 + s_2 * dk_2, c_2 * dk_2 - s_1 * dk_1], dim=-1)
    return grad_query.to(out_dt), grad_key.to(out_dt), None, None


# ============================================================
# 单路算子（rotary_position_embedding_grad）
#   torch.ops.npu.npu_rotary_mul_backward(grad, x, r1, r2, rotary_mode)
#     -> (dx, dcos, dsin)
#   一次只能算一路，要拿到 query 和 key 的梯度需调用两次。
# ============================================================

def _single_path_grad(
    grad_embed: torch.Tensor,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    need_cos_sin_grad: bool,
):
    """调用一次单路算子，返回 (dx, dcos, dsin)。"""
    # x = x.detach()
    if need_cos_sin_grad and not cos.requires_grad:
        cos = cos.detach().requires_grad_(True)
    if need_cos_sin_grad and not sin.requires_grad:
        sin = sin.detach().requires_grad_(True)
    return torch.ops.npu.npu_rotary_mul_backward(grad_embed, x, cos, sin, ROTARY_MODE_HALF)


def _single_path_grad_qk(
    grad_query_embed: torch.Tensor,
    grad_key_embed: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    need_cos_sin_grad: bool,
):
    """两次单路算子调用，等价于一次双路算子。

    返回 (grad_query, grad_key, grad_cos, grad_sin)。
    need_cos_sin_grad=True（四输出场景）时，dcos/dsin 由两路结果相加得到
    （前向中 cos/sin 同时作用于 q 和 k）；否则返回 None。
    """
    dq, dcos_q, dsin_q = _single_path_grad(grad_query_embed, query, cos, sin, need_cos_sin_grad)
    dk, dcos_k, dsin_k = _single_path_grad(grad_key_embed, key, cos, sin, need_cos_sin_grad)

    # 仅在需要 cos/sin 梯度（四输出场景）时累加两路 dcos/dsin。
    # 两输出场景（need_cos_sin_grad=False）op-plugin 仍会返回非 None 的全 0 dcos/dsin，
    # 若按 is not None 判断会触发无意义的 0+0 ADD，故这里用语义开关 gate。
    dcos = None
    dsin = None
    if need_cos_sin_grad:
        if dcos_q is not None and dcos_k is not None:
            dcos = dcos_q + dcos_k
        if dsin_q is not None and dsin_k is not None:
            dsin = dsin_q + dsin_k
    return dq, dk, dcos, dsin


def _dtype_tolerances(dtype: torch.dtype):
    """按 dtype 分档比对容差。

    grad_query/grad_key golden 与 kernel bit-exact，任意档位都通过；
    grad_cos/grad_sin 的 kernel (AccumDcosDsinVF) 在 Q/K 两路累加间把中间结果写回
    fp16/bf16 的 UB buffer（存在跨路低精度截断），而 golden 用 fp32 一次累加（更准），
    逐元素乘加差异落在该 dtype 的若干 ULP 内，故低精度档位放宽 atol。
    """
    if dtype == torch.float16:
        return 1e-2, 1e-3
    if dtype == torch.bfloat16:
        return 1e-1, 1e-2
    return 1e-4, 1e-4


def _compare_tensor(name: str, actual: Optional[torch.Tensor], golden: Optional[torch.Tensor]) -> bool:
    if actual is None and golden is None:
        print(f"{name}: ok=True, actual=None, golden=None")
        return True
    if actual is None or golden is None:
        print(f"{name}: ok=False, actual={actual is None}, golden={golden is None}")
        return False

    atol, rtol = _dtype_tolerances(actual.dtype)
    actual_cpu = actual.detach().cpu().to(torch.float64)
    golden_cpu = golden.detach().cpu().to(torch.float64)
    diff = (actual_cpu - golden_cpu).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = (diff / (golden_cpu.abs() + 1e-7)).max().item()
    ok = torch.allclose(actual_cpu, golden_cpu, atol=atol, rtol=rtol)
    print(f"{name}: ok={ok}, max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}, max_rel={max_rel:.6e} "
          f"(atol={atol:.0e}, rtol={rtol:.0e})")
    return ok

def run_case(case: dict):
    torch.manual_seed(case["seed"])
    q_shape, k_shape, cos_shape = _build_shapes(
        case["layout"], case["s"], case["b"], case["n_q"], case["n_k"], case["d"], case["cos_b"]
    )
    dtype = case["dtype"]

    grad_query_embed = torch.randn(q_shape, dtype=dtype)
    grad_key_embed = torch.randn(k_shape, dtype=dtype)
    cos = torch.randn(cos_shape, dtype=dtype)
    sin = torch.randn(cos_shape, dtype=dtype)
    query = torch.randn(q_shape, dtype=dtype) if case["use_optional"] else None
    key = torch.randn(k_shape, dtype=dtype) if case["use_optional"] else None

    grad_query_embed_npu = grad_query_embed.npu()
    grad_key_embed_npu = grad_key_embed.npu()
    cos_npu = cos.npu()
    sin_npu = sin.npu()
    query_npu = query.npu() if query is not None else None
    key_npu = key.npu() if key is not None else None
    cos_npu_single = cos.npu()
    sin_npu_single = sin.npu()
    if query_npu is not None and key_npu is not None:
        cos_npu_single = cos_npu_single.requires_grad_(True)
        sin_npu_single = sin_npu_single.requires_grad_(True)

    for i in range(3):
        if query_npu is None and key_npu is None:
            actual = npu_apply_rotary_pos_emb_grad(
                grad_query_embed_npu,
                grad_key_embed_npu,
                cos_npu,
                sin_npu,
                rotary_mode=ROTARY_MODE_HALF,
                layout=case["layout"],
            )
        else:
            actual = npu_apply_rotary_pos_emb_grad(
                grad_query_embed_npu,
                grad_key_embed_npu,
                cos_npu,
                sin_npu,
                query=query_npu,
                key=key_npu,
                rotary_mode=ROTARY_MODE_HALF,
                layout=case["layout"],
            )

    # 两次单路算子。不传 query/key 时，x 只用于满足 torch_npu 接口签名；
    # cos/sin 不置 requires_grad，op-plugin 会向底层传空 xOptional，只计算 dx。
    single_query_npu = query_npu if query_npu is not None else torch.empty_like(grad_query_embed_npu)
    single_key_npu = key_npu if key_npu is not None else torch.empty_like(grad_key_embed_npu)
    for i in range(3):
        single = _single_path_grad_qk(
            grad_query_embed_npu,
            grad_key_embed_npu,
            cos_npu_single,
            sin_npu_single,
            single_query_npu,
            single_key_npu,
            need_cos_sin_grad=(query_npu is not None and key_npu is not None),
        )

    if query is not None and key is not None:
        golden = _golden_backward(grad_query_embed, grad_key_embed, cos, sin, query, key)
    else:
        golden = _golden_backward_required_only(grad_query_embed, grad_key_embed, cos, sin)

    print(f"\nCASE: {case['name']}")
    print(f"layout={case['layout']}, dtype={dtype}, s={case['s']}, b={case['b']}, n_q={case['n_q']}, n_k={case['n_k']}, d={case['d']}")
    print(f"cos_b={case['cos_b']}")
    print(f"input grad_query_embed shape={tuple(grad_query_embed.shape)}")
    print(f"input grad_key_embed   shape={tuple(grad_key_embed.shape)}")
    print(f"input cos              shape={tuple(cos.shape)}")
    print(f"input sin              shape={tuple(sin.shape)}")
    print(f"input query            shape={None if query is None else tuple(query.shape)}")
    print(f"input key              shape={None if key is None else tuple(key.shape)}")

    print("actual (dual-path, 1 call):")
    print(f"  grad_query shape={tuple(actual[0].shape)}")
    print(f"  grad_key   shape={tuple(actual[1].shape)}")
    print(f"  grad_cos   shape={None if actual[2] is None else tuple(actual[2].shape)}")
    print(f"  grad_sin   shape={None if actual[3] is None else tuple(actual[3].shape)}")

    single_path_has_dcos = False
    if single is not None:
        print("single (single-path, 2 calls):")
        print(f"  grad_query shape={tuple(single[0].shape)}")
        print(f"  grad_key   shape={tuple(single[1].shape)}")
        print(f"  grad_cos   shape={None if single[2] is None else tuple(single[2].shape)}")
        print(f"  grad_sin   shape={None if single[3] is None else tuple(single[3].shape)}")
        single_path_has_dcos = (
            single[2] is not None and single[2].abs().max().item() > 0
        )

    if golden is not None:
        print("compare dual vs golden:")
        _compare_tensor("grad_query", actual[0], golden[0])
        _compare_tensor("grad_key", actual[1], golden[1])
        if golden[2] is not None and golden[3] is not None:
            _compare_tensor("grad_cos", actual[2], golden[2])
            _compare_tensor("grad_sin", actual[3], golden[3])
        else:
            print("grad_cos/grad_sin: skipped -- required-only path")

    if single is not None and golden is not None:
        print("compare single(x2) vs golden:")
        _compare_tensor("grad_query", single[0], golden[0])
        _compare_tensor("grad_key", single[1], golden[1])
        if golden[2] is not None and golden[3] is not None and single_path_has_dcos:
            _compare_tensor("grad_cos", single[2], golden[2])
            _compare_tensor("grad_sin", single[3], golden[3])
        else:
            print("grad_cos/grad_sin: skipped -- required-only path or 单路算子未产出（全 0）")


def main():
    torch.npu.set_device(0)
    for case in CASES:
        run_case(case)


if __name__ == "__main__":
    main()
