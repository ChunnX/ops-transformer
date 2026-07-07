import torch

from core.gen_data import GenDataBase
from ops.flash_attn.input_spec import flash_attn_inputs


def normalize_params(raw: dict) -> dict:
    """将 test_cases 原始字段补全为计算所需参数。"""

    c = dict(raw)
    layout_q = c.get("layout_q", "BNSD")
    c["input_layout"] = layout_q
    c.setdefault("layout_kv", layout_q)
    c.setdefault("layout_out", layout_q)
    c.setdefault("N2", c["N1"])
    c.setdefault("S2", c.get("S1"))
    c.setdefault("DV", c.get("D"))
    c.setdefault("DRope", 0)
    c.setdefault("win_left", 2147483647)
    c.setdefault("win_right", 2147483647)
    c.setdefault("mask_mode", 0)

    for key in ("cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv"):
        if c.get(key) == [None] or c.get(key) is None:
            c.pop(key, None)

    if layout_q == "TND":
        c.setdefault("seqused_q", [c["cu_seqlens_q"][i + 1] - c["cu_seqlens_q"][i]
                                   for i in range(len(c["cu_seqlens_q"]) - 1)])
        if c.get("layout_kv", layout_q) not in ("PA_BBND", "PA_BNBD", "PA_NZ"):
            c.setdefault("cu_seqlens_kv", list(c["cu_seqlens_q"]))
            c.setdefault("seqused_kv", [c["cu_seqlens_kv"][i + 1] - c["cu_seqlens_kv"][i]
                                        for i in range(len(c["cu_seqlens_kv"]) - 1)])
        c["B"] = 1

    layout_kv_val = c.get("layout_kv", layout_q)
    if layout_kv_val in ("PA_BBND", "PA_BNBD", "PA_NZ"):
        c.setdefault("block_size", 128)
        if "seqused_kv" not in c:
            c["seqused_kv"] = [c.get("S2", c.get("S1"))] * c.get("B", 1)

        if layout_kv_val == "PA_NZ":
            dtype_str = c.get("dtype", "fp16")
            nz_blk_elem = 16 if dtype_str in ("fp16", "bf16") else 32
            c["nz_blk_elem"] = nz_blk_elem
            c["D_nz_sub"] = c["D"] // nz_blk_elem
            c["DV_nz_sub"] = c.get("DV", c["D"]) // nz_blk_elem

        # 用例提供了 block_table 直接用，不重新生成
        if c.get("block_table") is not None and isinstance(c.get("block_table"), list):
            bt_raw = c["block_table"]
            c["block_table"] = torch.tensor(bt_raw, dtype=torch.int32)
            c["actual_b"] = c["block_table"].shape[0]
            c["num_blocks"] = int(max(max(row) for row in bt_raw)) + 1
            return c

        seqused_kv = c["seqused_kv"]
        block_size = c["block_size"]
        bt_shape = c.get("block_table_shape", [])

        if bt_shape:
            b_val = bt_shape[0]
            max_blk = bt_shape[1]
        else:
            b_val = c.get("B", 1)
            if layout_q == "TND":
                b_val = len(c["cu_seqlens_q"]) - 1
            max_blk = (max(seqused_kv) + block_size - 1) // block_size

        block_table = torch.full((b_val, max_blk), -1, dtype=torch.int32)
        block_idx = 0
        for i in range(b_val):
            b_seq = seqused_kv[i] if i < len(seqused_kv) else seqused_kv[0]
            b_blk = (b_seq + block_size - 1) // block_size
            for j in range(b_blk):
                block_table[i][j] = block_idx
                block_idx += 1
        c["block_table"] = block_table
        c["actual_b"] = b_val
        c["num_blocks"] = block_idx  # PA 时 K/V 需要的总 page 数
    return c


class GenDataBackend(GenDataBase):
    def normalize_params(self, params: dict) -> dict:
        return normalize_params(params)

    def gen_inputs(self, params: dict) -> dict:
        layout = params.get("layout_q", params.get("input_layout", "BNSD"))
        layout_kv = params.get("layout_kv", layout)
        inputs = flash_attn_inputs.generate(
            params, device=torch.device("cpu"),
            layout=layout, layout_kv=layout_kv,
            seed=params.get("seed", 42),
        )
        if params.get("block_table") is not None:
            inputs["block_table"] = params["block_table"]
        for k, v in params.items():
            if k not in inputs:
                inputs[k] = v
        inputs["softmax_scale"] = params.get("scale", 1 / (params["D"] ** 0.5))
        return inputs
