"""用例加载 + 参数规范化。从 test_flash_attn.py 抽取。"""

import itertools
import importlib
import torch
from typing import Dict, List


def load_case_modules(module_names: List[str]) -> Dict[str, dict]:
    """从 test_cases/*.py 加载 TestCases dict，展开所有参数组合。"""
    all_cases = {}
    for mod_name in module_names:
        mod = importlib.import_module(mod_name)
        for case_name, case_dict in mod.TestCases.items():
            keys = list(case_dict.keys())
            values = [case_dict[k] for k in keys]
            combos = list(itertools.product(*values))
            if len(combos) == 1:
                all_cases[f"{mod_name}/{case_name}"] = dict(zip(keys, combos[0]))
            else:
                for i, combo in enumerate(combos):
                    all_cases[f"{mod_name}/{case_name}_{i}"] = dict(zip(keys, combo))
    return all_cases


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
        else:
            if "seqused_kv" not in c and "cu_seqlens_kv" in c:
                cu = c["cu_seqlens_kv"]
                c["seqused_kv"] = [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]
        c["B"] = 1

    layout_kv_val = c.get("layout_kv", layout_q)
    if layout_kv_val in ("PA_BBND", "PA_BNBD", "PA_NZ"):
        c.setdefault("block_size", 128)
        if "seqused_kv" not in c:
            c["seqused_kv"] = [c.get("S2", c.get("S1"))] * c.get("B", 1)

        if layout_kv_val == "PA_NZ":
            dtype_str = c.get("Dtype", "fp16")
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

    # ── quant_mode 参数映射 ──
    quant_mode = c.get("quant_mode", None)

    if quant_mode is None:
        # 向后兼容：从旧的 kv_quant_mode 字符串推导
        old_mode = c.get("kv_quant_mode", "none")
        if old_mode == "per_token_group":
            quant_mode = 1
        else:
            quant_mode = 0

    if quant_mode == 1:
        # 伪量化模式：推导内部参数
        c["kv_quant_mode"] = "per_token_group"
        c["kv_quant_dtype"] = "fp4_e2m1"
        c["scale_quant_dtype"] = "fp8_e8m0"
        c["group_size"] = 32
        assert c["D"] % 32 == 0, f"D={c['D']} must be 32-aligned for fp4"
        c["D_group"] = c["D"] // 32
        dv = c.get("DV", c["D"])
        assert dv % 32 == 0, f"DV={dv} must be 32-aligned for fp4"
        c["DV_group"] = dv // 32
    elif quant_mode == 0:
        # 非量化模式
        c["kv_quant_mode"] = "none"
    elif quant_mode in (2, 3):
        raise NotImplementedError(f"quant_mode={quant_mode} 待拓展")

    c["quant_mode"] = quant_mode

    # ── sinks 参数 ──
    c.setdefault("sinks", None)

    # ── layout_kv 校验（伪量化仅支持 PA） ──
    if quant_mode and quant_mode >= 1:
        layout_kv = c.get("layout_kv", "PA_BBND")
        assert layout_kv in ("PA_BBND", "PA_BNBD", "PA_NZ"), \
            f"layout_kv={layout_kv} must be PA_BBND/PA_BNBD/PA_NZ when quant_mode>=1"

    # ── layout_out 校验（支持 TND 输出） ──
    layout_out = c.get("layout_out", "BNSD")
    assert layout_out in ("BNSD", "BSND", "TND"), \
        f"layout_out={layout_out} must be BNSD|BSND|TND"

    return c


def resolve_case_ids(case_id_arg: str, all_cases: Dict) -> List[str]:
    """解析 --case_id 参数。"""
    if case_id_arg == "all":
        return sorted(all_cases.keys())
    ids = [x.strip() for x in case_id_arg.split(",")]
    result = []
    missing = []
    for cid in ids:
        matches = [k for k in all_cases
                   if k.endswith(f"/{cid}") or k == cid
                   or k.rsplit("/", 1)[-1].startswith(f"{cid}_")]
        if matches:
            result.extend(matches)
        else:
            missing.append(cid)
    if missing:
        print(f"[WARN] 以下 case 不存在: {missing}")
    return result
