"""golden 数据加载与 L-A/L-B 验收判据（唯一定义点）。

冻结容差口径（任务书《生态算子开源精度标准》双层策略）：
- L-A：rtol = atol = 2^-9 (FP16) / 2^-6 (BF16)；matched_ratio ≥ 0.99；
  max_abs ≤ 0.1 (FP16) / 1.0 (BF16)；LSE ≤ 1e-3；
- L-B：对 FP32 SDPA 的误差比 max/mean/RMSE ≤ (2.0, 1.2, 1.2)，
  分母取 manifest metrics（键缺失或分母 ≤ 0 兜底为 1.0）。

消费方：test_golden_precision.py / test_kernel_twin.py / test_function.py /
test_quant.py。任何口径调整必须先过四者全部用例并同步任务书口径。
"""

import json
import math
from pathlib import Path
from typing import Optional, Tuple, Union

import torch

# ---------------------------------------------------------------------------
# 冻结容差常量（一处定义）
# ---------------------------------------------------------------------------
ATOL_FP16 = 2.0 ** (-9)      # FP16 混合容差 rtol = atol
ATOL_BF16 = 2.0 ** (-6)      # BF16 混合容差 rtol = atol
MATCHED_RATIO_MIN = 0.99     # L-A 匹配比例下限
MAX_ABS_LIMIT_FP16 = 0.1     # L-A 最大绝对误差上限（FP16 输出）
MAX_ABS_LIMIT_BF16 = 1.0     # L-A 最大绝对误差上限（BF16 输出）
LSE_TOL = 1.0e-3             # LSE 最大绝对差
LB_RATIO_LIMITS = (2.0, 1.2, 1.2)  # L-B (max, mean, rmse) 误差比上限
LB_RATIO_EPS = 1.0e-5        # L-B 上限浮点松弛


OP_ROOT = Path(__file__).resolve().parent.parent
# golden 数据集唯一位置：docker/golden（docker/init.sh --golden 幂等置备，绝不重生成）
GOLDEN_DIR = OP_ROOT / "docker" / "golden"
REPO_ROOT = OP_ROOT


def find_golden_dir() -> Path:
    """golden 数据集目录（唯一定义点）。"""
    return GOLDEN_DIR


def find_repo_root() -> Path:
    """算子根目录（唯一定义点）。"""
    return OP_ROOT


def load_manifest_cases() -> dict:
    """读取 golden/manifest.json 的 cases 字典（25 桶）。"""
    with open(GOLDEN_DIR / "manifest.json", encoding="utf-8") as f:
        return json.load(f)["cases"]


def load_tensor(case_id: str, name: str) -> torch.Tensor:
    """加载 golden/<case_id>/<name>.pt（CPU 张量）。"""
    path = GOLDEN_DIR / case_id / f"{name}.pt"
    return torch.load(path, map_location="cpu", weights_only=True)


def load_case_tensors(case_meta: dict) -> dict:
    """按 manifest case 的 tensors 清单加载全部张量 → {name: Tensor}。"""
    res = {}
    for tname, rel_path in case_meta["tensors"].items():
        p = Path(rel_path)
        parts = p.parts
        if parts and parts[0] == ".":
            parts = parts[1:]
        if parts and parts[0] == "golden":
            parts = parts[1:]
        tensor_path = GOLDEN_DIR / Path(*parts)
        res[tname] = torch.load(tensor_path, map_location="cpu", weights_only=True)
    return res


def atol_for(is_bf16: bool) -> float:
    return ATOL_BF16 if is_bf16 else ATOL_FP16


def max_abs_limit_for(is_bf16: bool) -> float:
    return MAX_ABS_LIMIT_BF16 if is_bf16 else MAX_ABS_LIMIT_FP16


def l_a_metrics(out: torch.Tensor, ref_out: torch.Tensor, is_bf16: bool) -> dict:
    """L-A 指标：混合容差 matched_ratio 与 max_abs（对 GPU golden 同超参基准）。"""
    atol = atol_for(is_bf16)
    diff = (out.float() - ref_out.float()).abs()
    tol = atol + atol * ref_out.float().abs()
    return {
        "atol": atol,
        "matched_ratio": (diff <= tol).float().mean().item(),
        "max_abs_error": diff.max().item(),
        "max_abs_limit": max_abs_limit_for(is_bf16),
    }


def l_b_ratios(out: torch.Tensor, sdpa: torch.Tensor, meta: dict) -> dict:
    """L-B 指标：对 FP32 SDPA 误差相对 manifest 基准的比值（max/mean/rmse）。"""
    err = (out.float() - sdpa.float()).abs()
    ref = meta.get("metrics", {})
    den_max = ref.get("max_abs_error_vs_sdpa", 0.0)
    den_mean = ref.get("mean_abs_error_vs_sdpa", 0.0)
    den_rmse = ref.get("rmse_error_vs_sdpa", 0.0)
    rmse = torch.sqrt((err ** 2).mean()).item()

    def _ratio(num: float, den: float) -> float:
        return (num / den) if den > 0 else 1.0

    return {
        "ratio_max": _ratio(err.max().item(), den_max),
        "ratio_mean": _ratio(err.mean().item(), den_mean),
        "ratio_rmse": _ratio(rmse, den_rmse),
    }


def make_qkv(
    b: int = 1,
    hq: int = 4,
    hkv: int = 4,
    sq: int = 64,
    sk: int = 64,
    d: int = 64,
    layout: str = "HND",
    dtype: torch.dtype = torch.float16,
    seed: int = 42,
    device: Union[str, torch.device] = "cpu",
    s: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成测试用的伪随机 Q, K, V 张量（唯一定义点）。
    
    统一支持 (sq, sk) 区分或简写 s (当 s 不为空时 sq = sk = s)。
    """
    if s is not None:
        sq = sk = s
    g = torch.Generator(device="cpu").manual_seed(seed)
    if layout == "HND":
        q = torch.randn(b, hq, sq, d, dtype=dtype, generator=g)
        k = torch.randn(b, hkv, sk, d, dtype=dtype, generator=g)
        v = torch.randn(b, hkv, sk, d, dtype=dtype, generator=g)
    elif layout == "NHD":
        q = torch.randn(b, sq, hq, d, dtype=dtype, generator=g)
        k = torch.randn(b, sk, hkv, d, dtype=dtype, generator=g)
        v = torch.randn(b, sk, hkv, d, dtype=dtype, generator=g)
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    if str(device) != "cpu":
        return q.to(device), k.to(device), v.to(device)
    return q, k, v


def sdpa_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """高精度 FP32 SDPA 基线实现（唯一定义点，L-B 基准计算）。"""
    if tensor_layout == "NHD":
        q_bnsd = q.transpose(1, 2).contiguous().float()
        k_bnsd = k.transpose(1, 2).contiguous().float()
        v_bnsd = v.transpose(1, 2).contiguous().float()
    else:
        q_bnsd = q.float()
        k_bnsd = k.float()
        v_bnsd = v.float()

    b, hq, sq, d = q_bnsd.shape
    _, hkv, sk, _ = k_bnsd.shape

    if hq != hkv:
        rep = hq // hkv
        k_bnsd = k_bnsd.repeat_interleave(rep, dim=1)
        v_bnsd = v_bnsd.repeat_interleave(rep, dim=1)

    scale = (1.0 / math.sqrt(d)) if sm_scale is None else float(sm_scale)
    attn = torch.matmul(q_bnsd, k_bnsd.transpose(-2, -1)) * scale
    if is_causal:
        mask = torch.ones(sq, sk, device=attn.device, dtype=torch.bool).triu(1)
        attn = attn.masked_fill(mask, float("-inf"))
    prob = torch.softmax(attn, dim=-1)
    out = torch.matmul(prob, v_bnsd)

    if tensor_layout == "NHD":
        out = out.transpose(1, 2).contiguous()
    return out
