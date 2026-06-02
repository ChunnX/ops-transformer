import argparse
import math
from typing import Optional

import torch


def _torch_e8m0_bytes(scale: torch.Tensor) -> torch.Tensor:
    scale_cpu = scale.detach().to("cpu", dtype=torch.float32).to(torch.float8_e8m0fnu)
    return scale_cpu.view(torch.uint8).to(scale.device)


def _torch_e8m0_roundtrip(scale: torch.Tensor) -> torch.Tensor:
    scale_cpu = scale.detach().to("cpu", dtype=torch.float32).to(torch.float8_e8m0fnu).float()
    return scale_cpu.to(scale.device)


class SwaChunkPrefillMXFP8CatlassGolden:
    def __init__(
        self,
        *,
        q_head_num: int,
        qk_head_dim: int = 128,
        v_head_dim: int = 128,
        global_window_size: int = 4,
        window_size_left: int = 1023,
        gqa_layout: str = "AABB",
        causal: bool = True,
        softmax_scale: Optional[float] = None,
        p_quant_mode: int = 1,
    ):
        if qk_head_dim != 128 or v_head_dim != 128:
            raise ValueError("only qk_head_dim=v_head_dim=128 is covered by this golden")
        if window_size_left not in (1023, 2047):
            raise ValueError("window_size_left must be 1023 or 2047")
        if gqa_layout not in ("AABB", "ABAB"):
            raise ValueError("gqa_layout must be AABB or ABAB")
        if p_quant_mode not in (0, 1):
            raise ValueError("p_quant_mode must be 0(static) or 1(dynamic)")

        self._q_head_num = q_head_num
        self._qk_head_dim = qk_head_dim
        self._v_head_dim = v_head_dim
        self._global_window_size = global_window_size
        self._local_window_size = window_size_left + 1
        self._gqa_layout = gqa_layout
        self._causal = causal
        self._softmax_scale = 1.0 / math.sqrt(qk_head_dim) if softmax_scale is None else softmax_scale
        self._p_quant_mode = p_quant_mode

    @staticmethod
    def _e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
        exponent = scale.to(torch.int16)
        return torch.where(
            exponent == 0,
            torch.zeros_like(exponent, dtype=torch.float32),
            torch.pow(torch.tensor(2.0, dtype=torch.float32, device=scale.device), exponent.to(torch.float32) - 127.0),
        )

    def _generate_mask(self, max_kv_seq_length: int) -> torch.Tensor:
        mask = torch.ones(max_kv_seq_length, max_kv_seq_length, dtype=torch.bool, device="npu")
        inner_size = max_kv_seq_length - self._global_window_size - self._local_window_size
        if inner_size > 0:
            inner_mask = torch.triu(torch.ones(inner_size, inner_size, dtype=torch.bool, device="npu"), 1)
            row_start = self._local_window_size + self._global_window_size
            col_start = self._global_window_size
            mask[row_start : row_start + inner_size, col_start : col_start + inner_size] = inner_mask
        return torch.tril(mask)

    @staticmethod
    def _reconstruct_kv_seq_from_blocks(cache: torch.Tensor, block_table: torch.Tensor, kv_len: int) -> torch.Tensor:
        block_size = cache.shape[2]
        num_blocks = int(math.ceil(kv_len / block_size))
        block_idx = block_table[:num_blocks].to(torch.long)
        blocks = cache.index_select(0, block_idx)
        seq = blocks.permute(0, 2, 1, 3).contiguous().view(num_blocks * block_size, cache.shape[1], cache.shape[3])
        return seq[:kv_len].contiguous()

    def _dequantize_q(self, q: torch.Tensor, q_scale: torch.Tensor) -> torch.Tensor:
        total_q = q.shape[0]
        q_float = q.reshape(total_q, self._q_head_num, self._qk_head_dim).float()
        scale = self._e8m0_to_fp32(q_scale.reshape(total_q, self._q_head_num, self._qk_head_dim // 32))
        return (q_float.reshape(total_q, self._q_head_num, self._qk_head_dim // 32, 32) * scale.unsqueeze(-1)).reshape_as(q_float)

    def _dequantize_k(self, k: torch.Tensor, k_scale: torch.Tensor) -> torch.Tensor:
        scale = self._e8m0_to_fp32(k_scale)
        return (k.float().reshape(*k.shape[:3], self._qk_head_dim // 32, 32) * scale.unsqueeze(-1)).reshape_as(k)

    def _dequantize_v(self, v: torch.Tensor, v_scale: torch.Tensor) -> torch.Tensor:
        block_num, kv_head_num, block_size, v_head_dim = v.shape
        s_group = block_size // 32
        packed = v_scale.reshape(block_num, kv_head_num, (s_group + 1) // 2, v_head_dim, 2)
        scale_bytes = torch.empty((block_num, kv_head_num, s_group, v_head_dim), dtype=torch.uint8, device=v.device)
        scale_bytes[:, :, ::2, :] = packed[..., 0]
        if s_group > 1:
            scale_bytes[:, :, 1::2, :] = packed[..., 1]
        scale = self._e8m0_to_fp32(scale_bytes)
        return (v.float().reshape(block_num, kv_head_num, s_group, 32, v_head_dim) * scale.unsqueeze(3)).reshape_as(v)

    @staticmethod
    def _p_post_quant_scale_from_max_abs(max_abs: torch.Tensor) -> torch.Tensor:
        scale = torch.zeros_like(max_abs, dtype=torch.float32)
        nz = max_abs > 0
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale[nz] = torch.pow(2.0, torch.ceil(torch.log2(max_abs[nz] / fp8_max)))
        scale = torch.where(nz, scale, torch.ones_like(scale))
        return _torch_e8m0_roundtrip(scale)

    def _dynamic_p_mxfp8_bits(self, payload: torch.Tensor, scale_source: torch.Tensor):
        group_size = 32
        column = payload.shape[-1]
        column_round = int(math.ceil(column / group_size) * group_size)
        padded_payload = torch.zeros((*payload.shape[:-1], column_round), dtype=payload.dtype, device=payload.device)
        padded_scale_source = torch.zeros((*scale_source.shape[:-1], column_round), dtype=scale_source.dtype, device=scale_source.device)
        padded_payload[..., :column] = payload
        padded_scale_source[..., :column] = scale_source
        payload_groups = padded_payload.reshape(*padded_payload.shape[:-1], column_round // group_size, group_size)
        scale_groups = padded_scale_source.reshape(*padded_scale_source.shape[:-1], column_round // group_size, group_size)
        block_scale = self._p_post_quant_scale_from_max_abs(scale_groups.abs().amax(dim=-1))
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        p_payload = (payload_groups / block_scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        return p_payload, _torch_e8m0_bytes(block_scale), column

    @staticmethod
    def _quantize_dequantize_p_static(payload: torch.Tensor, p_scale: float = 16.0) -> torch.Tensor:
        return (payload * float(p_scale)).to(torch.float8_e4m3fn).float() / float(p_scale)

    def _quantize_dequantize_p_dynamic(self, payload: torch.Tensor, scale_source: torch.Tensor) -> torch.Tensor:
        p_payload, scale_e8m0, column = self._dynamic_p_mxfp8_bits(payload, scale_source)
        p_dequant = p_payload.float() * self._e8m0_to_fp32(scale_e8m0).unsqueeze(-1)
        return p_dequant.reshape(*payload.shape[:-1], -1)[..., :column]

    def forward_golden(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        block_table: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        softmax_scale: Optional[float] = None,
        p_scale: float = 16.0,
    ) -> torch.Tensor:
        scale = self._softmax_scale if softmax_scale is None else softmax_scale
        q_deq = self._dequantize_q(q, q_scale)
        k_deq = self._dequantize_k(k, k_scale)
        v_deq = self._dequantize_v(v, v_scale)
        hq = self._q_head_num
        hkv = k.shape[1]
        head_ratio = hq // hkv
        if self._gqa_layout == "AABB":
            kv_head_idx = torch.arange(hq, device=q.device) // head_ratio
        else:
            kv_head_idx = torch.arange(hq, device=q.device) % hkv

        out = torch.zeros((q_deq.shape[0], hq, self._v_head_dim), dtype=torch.float32, device=q.device)
        q_start = 0
        batch = int(cu_seqlens_q.numel())
        max_kv_len = int(cu_seqlens_k.max().item()) if batch > 0 else 0
        swa_mask = self._generate_mask(max_kv_len) if self._causal and max_kv_len > 0 else None

        for batch_idx in range(batch):
            q_end = int(cu_seqlens_q[batch_idx].item())
            q_len = q_end - q_start
            kv_len = int(cu_seqlens_k[batch_idx].item())
            if q_len <= 0:
                q_start = q_end
                continue
            if kv_len < q_len:
                raise ValueError("SWA chunk prefill golden expects kv_len >= q_len")

            q_b = q_deq[q_start:q_end]
            k_b = self._reconstruct_kv_seq_from_blocks(k_deq, block_table[batch_idx], kv_len).index_select(1, kv_head_idx)
            v_b = self._reconstruct_kv_seq_from_blocks(v_deq, block_table[batch_idx], kv_len).index_select(1, kv_head_idx)
            scores = torch.bmm(q_b.permute(1, 0, 2).contiguous(), k_b.permute(1, 2, 0).contiguous()) * float(scale)
            if swa_mask is not None:
                real_mask = swa_mask[kv_len - q_len : kv_len, :kv_len]
                scores = scores.masked_fill((~real_mask).unsqueeze(0), float("-inf"))

            row_max = scores.amax(dim=-1, keepdim=True)
            exp_scores = torch.exp(scores - row_max)
            exp_scores = torch.where(torch.isfinite(exp_scores), exp_scores, torch.zeros_like(exp_scores))
            row_sum = exp_scores.sum(dim=-1, keepdim=True)
            if self._p_quant_mode == 0:
                probs = self._quantize_dequantize_p_static(exp_scores, p_scale)
            else:
                probs = self._quantize_dequantize_p_dynamic(exp_scores, exp_scores)
            out_b = torch.bmm(probs, v_b.permute(1, 0, 2).contiguous()) / row_sum
            out[q_start:q_end] = out_b.permute(1, 0, 2).contiguous()
            q_start = q_end

        return out.to(torch.bfloat16)


def _quantize_d_groups(x: torch.Tensor):
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    groups = x.float().reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    amax = groups.abs().amax(dim=-1)
    safe = torch.where(amax > 0, amax / fp8_max, torch.ones_like(amax))
    scale = torch.pow(torch.tensor(2.0, dtype=torch.float32, device=x.device), torch.ceil(torch.log2(safe)))
    scale_e8m0_bytes = _torch_e8m0_bytes(scale)
    scale_for_payload = SwaChunkPrefillMXFP8CatlassGolden._e8m0_to_fp32(scale_e8m0_bytes)
    payload = (groups / scale_for_payload.unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return payload.reshape_as(x), scale_e8m0_bytes


def _quantize_v_s_groups(x: torch.Tensor):
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    block_num, kv_head_num, block_size, v_head_dim = x.shape
    s_group = block_size // 32
    groups = x.float().reshape(block_num, kv_head_num, s_group, 32, v_head_dim)
    amax = groups.abs().amax(dim=3)
    safe = torch.where(amax > 0, amax / fp8_max, torch.ones_like(amax))
    scale = torch.pow(torch.tensor(2.0, dtype=torch.float32, device=x.device), torch.ceil(torch.log2(safe)))
    scale_bytes = _torch_e8m0_bytes(scale)
    scale_for_payload = SwaChunkPrefillMXFP8CatlassGolden._e8m0_to_fp32(scale_bytes)
    payload = (groups / scale_for_payload.unsqueeze(3)).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).reshape_as(x)
    packed = torch.zeros((block_num, kv_head_num, (s_group + 1) // 2, v_head_dim, 2), dtype=torch.uint8, device=x.device)
    packed[..., 0] = scale_bytes[:, :, ::2, :]
    if s_group > 1:
        packed[:, :, : scale_bytes[:, :, 1::2, :].shape[2], :, 1] = scale_bytes[:, :, 1::2, :]
    return payload, packed


def create_forward_inputs(
    *,
    batch: int,
    q_len: int,
    kv_len: int,
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
    block_size: int,
    device: str,
):
    blocks_per_batch = math.ceil(kv_len / block_size)
    block_num = batch * blocks_per_batch
    total_q = batch * q_len
    q_float = torch.randn(total_q, q_head_num, head_dim, dtype=torch.float32, device=device) * 0.2
    k_float = torch.randn(block_num, kv_head_num, block_size, head_dim, dtype=torch.float32, device=device) * 0.2
    v_float = torch.randn(block_num, kv_head_num, block_size, head_dim, dtype=torch.float32, device=device) * 0.2

    q_mxfp8, q_scale = _quantize_d_groups(q_float)
    k_mxfp8, k_scale = _quantize_d_groups(k_float)
    v_mxfp8, v_scale = _quantize_v_s_groups(v_float)

    block_table = torch.arange(block_num, dtype=torch.int32, device=device).reshape(batch, blocks_per_batch)
    cu_seqlens_q = torch.arange(q_len, total_q + 1, q_len, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.full((batch,), kv_len, dtype=torch.int32, device=device)

    return {
        "float_inputs": {"q": q_float, "k": k_float, "v": v_float},
        "mxfp8_inputs": {
            "q": q_mxfp8,
            "k": k_mxfp8,
            "v": v_mxfp8,
            "q_scale": q_scale,
            "k_scale": k_scale,
            "v_scale": v_scale,
            "block_table": block_table,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
        },
    }


def run_case(p_quant_mode: int, inputs, q_head_num: int, window_size_left: int):
    mxfp8 = inputs["mxfp8_inputs"]
    layer = SwaChunkPrefillMXFP8CatlassGolden(
        q_head_num=q_head_num,
        window_size_left=window_size_left,
        causal=True,
        p_quant_mode=p_quant_mode,
    )
    out = layer.forward_golden(
        mxfp8["q"],
        mxfp8["k"],
        mxfp8["v"],
        mxfp8["q_scale"],
        mxfp8["k_scale"],
        mxfp8["v_scale"],
        mxfp8["block_table"],
        mxfp8["cu_seqlens_q"],
        mxfp8["cu_seqlens_k"],
    )
    mode_name = "static" if p_quant_mode == 0 else "dynamic"
    print(f"{mode_name} swa forward_golden out shape", tuple(out.shape), "dtype", out.dtype)
    print(f"{mode_name} swa forward_golden finite", bool(torch.isfinite(out.float()).all().item()))
    return out


def main():
    import torch_npu

    parser = argparse.ArgumentParser()
    parser.add_argument("--p-quant-mode", choices=["static", "dynamic", "both"], default="both")
    parser.add_argument("--window-size-left", type=int, choices=[1023, 2047], default=1023)
    args = parser.parse_args()

    print("torch", torch.__version__)
    print("torch_npu", torch_npu.__version__ if hasattr(torch_npu, "__version__") else "imported")
    torch.manual_seed(11)
    device = "npu:0"
    inputs = create_forward_inputs(
        batch=2,
        q_len=16,
        kv_len=128,
        q_head_num=2,
        kv_head_num=1,
        head_dim=128,
        block_size=64,
        device=device,
    )
    modes = [0, 1] if args.p_quant_mode == "both" else [0 if args.p_quant_mode == "static" else 1]
    outs = [run_case(mode, inputs, 2, args.window_size_left) for mode in modes]
    mxfp8 = inputs["mxfp8_inputs"]
    print("created float inputs", {k: tuple(v.shape) for k, v in inputs["float_inputs"].items()})
    print("created mxfp8 inputs", {k: (tuple(v.shape), str(v.dtype), str(v.device)) for k, v in mxfp8.items()})
    if len(outs) == 2:
        diff = (outs[0].float() - outs[1].float()).abs()
        print("static_dynamic_diff", {"max_abs": float(diff.max().item()), "mean_abs": float(diff.mean().item())})
    print("PASS")


if __name__ == "__main__":
    main()