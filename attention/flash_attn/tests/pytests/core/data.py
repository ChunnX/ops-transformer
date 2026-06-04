"""声明式张量生成规则。一个算子 = 一个 InputSpec 对象，约 10 行声明。"""

import torch
from typing import Any, Callable, Dict, List, Optional


class InputSpec:
    """声明式张量数据生成器。

    用法:
        spec = (InputSpec("flash_attn")
            .tensor("q", layout={"BNSD": "B,N1,S1,D", "BSND": "B,S1,N1,D"})
            .tensor("k", layout={"BNSD": "B,N2,S2,D"})
        )
        inputs = spec.generate(params, device, layout="BNSD")
    """

    def __init__(self, name: str = ""):
        self.name = name
        self._items: List[dict] = []

    # ── 核心 API ──

    def tensor(self, name: str, *,
               shape: str = "",
               layout: dict = None,
               dtype=None,
               method: str = "randn",
               when: Callable = None,
               **kwargs) -> "InputSpec":
        """添加一个张量定义。

        Args:
            name:   张量名，如 "q", "k", "v", "block_table"
            shape:  默认 shape 表达式，如 "B,N1,S1,D"
            layout: 按 layout 的 shape 映射，如 {"BNSD": "B,N1,S1,D", "BSND": "B,S1,N1,D"}
            dtype:  torch.dtype，默认从 params["Dtype"] 推断
            method: "randn" | "uniform" | "fixed" | "zeros" | "custom"
            when:   fn(params) → bool，条件生成（None = 总是生成）
        """
        self._items.append(dict(
            name=name, shape=shape, layout=layout or {},
            dtype=dtype, method=method, when=when, kwargs=kwargs,
        ))
        return self

    # ── 生成入口 ──

    def generate(self, params: dict, device,
                 layout: str = "BNSD", seed: int = 42
                 ) -> Dict[str, torch.Tensor]:
        """在指定 device 上生成所有张量。"""
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        tensors = {}
        for it in self._items:
            if it["when"] and not it["when"](params):
                continue
            shape = self._resolve(it, params, layout)
            if not shape:
                continue
            dt = it["dtype"] or _infer_dtype(params.get("Dtype", "fp16"))
            t = _generate(it["method"], shape, dt, device, rng, **it["kwargs"])
            tensors[it["name"]] = t
        return tensors

    # ── shape 解析 ──

    def _resolve(self, item, params, layout):
        shape_str = item["layout"].get(layout, item["shape"])
        if not shape_str:
            return None
        dims = []
        for token in shape_str.split(","):
            token = token.strip()
            if token in params:
                dims.append(params[token])
            elif token == "_V":
                dims.append(params.get("DV", params["D"]))
            elif token == "total_s1":
                cu = params.get("cu_seqlens_q")
                dims.append(cu[-1] if cu else params.get("S1", 1))
            elif token == "total_s2":
                cu = params.get("cu_seqlens_kv")
                dims.append(cu[-1] if cu else params.get("S2", params.get("S1", 1)))
            elif token == "max_blocks":
                sv = params.get("seqused_kv", [params.get("S2", 1)])
                bs = params.get("block_size", 128)
                sm = max(sv) if isinstance(sv, list) else sv
                dims.append((sm + bs - 1) // bs if sm > 0 else 1)
            elif token == "B+1":
                cu = params.get("cu_seqlens_q")
                dims.append(len(cu) if cu else params.get("B", 1) + 1)
            else:
                dims.append(int(token))
        return tuple(dims)


# ── helpers ──

def _infer_dtype(ds: str) -> torch.dtype:
    return torch.float16 if ds == "fp16" else torch.bfloat16


def _generate(method, shape, dtype, device, rng, **kw):
    if method == "randn":
        return torch.randn(shape, dtype=dtype, device=device, generator=rng)
    if method == "uniform":
        lo, hi = kw.get("low", 0.0), kw.get("high", 1.0)
        t = torch.rand(shape, dtype=dtype, device=device, generator=rng)
        return t * (hi - lo) + lo
    if method == "fixed":
        return torch.full(shape, kw.get("value", 10.0), dtype=dtype, device=device)
    if method == "zeros":
        return torch.zeros(shape, dtype=dtype, device=device)
    if method == "custom":
        return kw["fn"](shape, dtype, device, rng)
    raise ValueError(f"Unknown method: {method}")


# ── 预定义 recipe ──

flash_attn_inputs = (
    InputSpec("flash_attn")
    .tensor("q", layout={"BNSD": "B,N1,S1,D", "BSND": "B,S1,N1,D", "TND": "total_s1,N1,D"})
    .tensor("k", layout={"BNSD": "B,N2,S2,D", "BSND": "B,S2,N2,D", "TND": "total_s2,N2,D"})
    .tensor("v", layout={"BNSD": "B,N2,S2,_V", "BSND": "B,S2,N2,_V", "TND": "total_s2,N2,_V"})
    .tensor("q_rope", layout={"BNSD": "B,N1,S1,DRope"},
            when=lambda p: p.get("DRope", 0) > 0)
    .tensor("k_rope", layout={"BNSD": "B,N2,S2,DRope"},
            when=lambda p: p.get("DRope", 0) > 0)
    .tensor("block_table", shape="B,max_blocks", dtype=torch.int32, method="zeros",
            when=lambda p: str(p.get("layout_kv", "")).startswith("PA_"))
)
