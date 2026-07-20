import torch

from core import CpuBase
from core.utils.flash_attn_golden import FlashAttentionGolden


def _to_tensor(x, device):
    """case 中的 seqused/cu_seqlens 可能是 list/None/tensor, 统一为 cpu tensor。
    None 透传 (由 golden 内部决定如何推导长度), 不在此自行构造。"""
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(device)
    return torch.tensor(x, device=device)


class CpuBackend(CpuBase):
    name = "cpu"

    def __init__(self):
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    def is_available(self) -> bool:
        return True

    def cpu_compute(self, inputs: dict) -> dict:
        """调用 FlashAttentionGolden, 参数严格取自 case (inputs), 不自行构造
        seqused/cu_seqlens/attn_mask。layout 转换由 golden 内部完成。"""
        dev = self._device
        n1 = inputs["N1"]
        n2 = inputs.get("N2", n1)
        d = inputs["D"]
        scale = inputs.get("scale", inputs.get("softmax_scale", 1.0 / (d ** 0.5)))

        layout_q   = inputs.get("layout_q", inputs.get("input_layout", "BNSD"))
        layout_kv  = inputs.get("layout_kv", layout_q)
        layout_out = inputs.get("layout_out", layout_q)

        golden = FlashAttentionGolden(
            num_heads_q=n1, num_heads_kv=n2, head_dim=d, softmax_scale=scale)

        return_lse = int(bool(inputs.get("return_softmax_lse", False)))

        attn_out, softmax_lse = golden.forward_golden(
            inputs["q"], inputs["k"], inputs["v"],
            block_table=_to_tensor(inputs.get("block_table"), dev),
            cu_seqlens_q=_to_tensor(inputs.get("cu_seqlens_q"), dev),
            cu_seqlens_kv=_to_tensor(inputs.get("cu_seqlens_kv"), dev),
            seqused_q=_to_tensor(inputs.get("seqused_q"), dev),
            seqused_kv=_to_tensor(inputs.get("seqused_kv"), dev),
            softmax_scale=scale,
            mask_mode=inputs.get("mask_mode", 0),
            layout_q=layout_q, layout_kv=layout_kv, layout_out=layout_out,
            return_softmax_lse=return_lse,
            device_mode="cpu",
            compute_mode="tiled",
        )
        return {"out": attn_out, "lse": softmax_lse}
