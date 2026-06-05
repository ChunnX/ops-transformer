#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
精简版 pytest：从 xlsx 读取参数，分别调用 flash_attn / FIA 算子，支持设定运行轮次。
FIA 模式在本文件内完成数据准备，通过 call_fia_npu() 直接调用
torch_npu.npu_fused_infer_attention_score，不依赖 FIA 原有测试脚本，
也不依赖 flash_attn 的 utils/backends。

用法（直接运行）:
    python test_xlsx_runner.py --xlsx cases.xlsx --operator flash_attn --rounds 10
    python test_xlsx_runner.py --xlsx cases.xlsx --operator fia --rounds 5 --row 2

用法（pytest）:
    pytest test_xlsx_runner.py --xlsx cases.xlsx --operator flash_attn --rounds 3 -v

依赖:
    pip install openpyxl pytest torch torch_npu
"""

import argparse
import os
import sys
import time
import math

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    import openpyxl
except ImportError:
    openpyxl = None

try:
    import torch_npu
except ImportError:
    torch_npu = None

from utils.data import generate_qkv, generate_npu_mask, trans_bnsd_to_layout
from backends.cpu_impl import tforward
from backends.npu_impl import flash_attn_npu
from utils.compare import check_result

# ─────────────────── xlsx 列名 → 内部字段映射 ───────────────────
# None = 跳过；'_' 前缀为中间字段，不直接写入 case
XLSX_COLUMN_MAP = {
    'inputLayout':              'input_layout',
    'out_shape':                None,
    'out_dtype':                'out_dtype',
    'softmaxLseFlag':           'return_softmax_lse',
    'softmaxLse':               None,
    'softmaxLse_dtype':         None,
    'q_shape':                  None,
    'q_dtype':                  'Dtype',
    'q_datarange':              'q_range',
    'q_padding_size':           'q_padding_size',
    'q_rope_shape':             '_q_rope_shape_',
    'q_rope_dtype':             None,
    'q_rope_datarange':         None,
    'k_shape':                  None,
    'k_dtype':                  'k_dtype',
    'k_datarange':              'k_range',
    'k_cache_shape':            '_k_cache_shape_',
    'k_prefix_shape':           None,
    'k_prefix_dtype':           None,
    'k_prefix_datarange':       None,
    'k_rope_shape':             '_k_rope_shape_',
    'k_rope_dtype':             None,
    'k_rope_datarange':         None,
    'k_rope_cache_shape':       None,
    'v_shape':                  None,
    'v_dtype':                  'v_dtype',
    'v_datarange':              'v_range',
    'v_cache_shape':            None,
    'v_prefix_shape':           None,
    'v_prefix_dtype':           None,
    'v_prefix_datarange':       None,
    'kv_padding_size':          'kv_padding_size',
    'q_start_idx':              'q_start_idx',
    'kv_start_idx':             None,
    'pse_type':                 'pse_type',
    'p_shape':                  None,
    'p_dtype':                  None,
    'p_datarange':              None,
    'p_random_nan':             None,
    'm_shape':                  '_m_shape_',
    'm_dtype':                  None,
    'm_random_type':            None,
    'm_random_nan':             None,
    'actual_seq_lengths_q':     'actual_seq_lengths_q',
    'actual_seq_lengths_kv':    'actual_seq_lengths_kv',
    'blockTable_shape':         'block_table_shape',
    'blockTable_dtype':         None,
    'blockSize':                'block_size',
    'numHeads':                 'N1',
    'numKeyValueHeads':         'N2',
    'scaleValue':               'scale',
    'kn_pre':                   'win_left',
    'kn_nxt':                   'win_right',
    'sparse':                   'mask_mode',
    'innerprecise':             'inner_precise',
    'dequantScale1':            None,
    'quantScale1':              None,
    'dequantScale2':            None,
    'enableGpu':                None,
    'fused_flag':               None,
    'B':                        'B',
    'Q_N':                      '_Q_N_',
    'KV_N':                     '_KV_N_',
    'G':                        'G',
    'QS':                       'QS',
    'KVS':                      'KVS',
    'S1':                       'S1',
    'S2':                       'S2',
    'Q_D':                      'D',
    'K_D':                      'K_D',
    'V_D':                      'DV',
    'D不等长':                   'd_unequal',
    'MLA':                      'mla',
    # 兼容旧格式列名
    'Testcase_Name':            '_case_name_',
    'testcase_name':            '_case_name_',
    'layout_q':                 'layout_q',
    'layout_kv':                'layout_kv',
    'layout_out':               'layout_out',
    'batch_size':               'B',
    'num_heads_q':              'N1',
    'num_heads_kv':             'N2',
    'head_dim':                 'D',
    'softmax_scale':            'scale',
    'win_left':                 'win_left',
    'win_right':                'win_right',
    'mask_mode':                'mask_mode',
    'seqused_q':                'seqused_q',
    'seqused_kv':               'seqused_kv',
    'cu_seqlens_q':             'cu_seqlens_q',
    'cu_seqlens_kv':            'cu_seqlens_kv',
    'return_softmax_lse':       'return_softmax_lse',
}

DTYPE_MAP = {
    'BF16': 'bf16', 'FP16': 'fp16', 'FP32': 'fp32', 'INT8': 'int8',
    'bf16': 'bf16', 'fp16': 'fp16', 'fp32': 'fp32', 'int8': 'int8',
    'FLOAT16': 'fp16', 'BFLOAT16': 'bf16', 'float16': 'fp16', 'bfloat16': 'bf16',
}

KV_STORAGE_MAP = {
    'pa_bbh': 'pa_bbh', 'PA_BBND': 'pa_bbh', 'PA_BBH': 'pa_bbh',
    'pa_bnbd': 'pa_bnbd', 'PA_BNBD': 'pa_bnbd',
    'pa_nz': 'pa_nz', 'PA_NZ': 'pa_nz',
}


# ─────────────────── 解析工具 ───────────────────

def is_empty(val):
    if val is None:
        return True
    s = str(val).strip()
    return s in ('', '/', 'None', 'none', 'N/A', 'n/a', '-', 'nan')


def try_int(v):
    f = float(v)
    return int(f) if f == int(f) and abs(f) < 1e15 else f


def parse_int_list(val_str):
    s = str(val_str).strip().strip('[]')
    return [int(float(x.strip())) for x in s.split(',') if x.strip()]


def parse_data_range(val):
    parts = [x.strip() for x in str(val).strip().split(',') if x.strip()]
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    return None


def parse_shape(val):
    if is_empty(val):
        return None
    s = str(val).strip().strip('[]()')
    return tuple(int(float(x.strip())) for x in s.split(',') if x.strip())


def seqlens_to_prefix_sum(lengths):
    result = [0]
    for l in lengths:
        result.append(result[-1] + l)
    return result


def generate_block_table(shape_str):
    parts = parse_shape(shape_str)
    if not parts or len(parts) != 2:
        return None
    b, num_blocks = parts
    table, idx = [], 0
    for _ in range(b):
        table.append(list(range(idx, idx + num_blocks)))
        idx += num_blocks
    return table


def parse_bool(val):
    if is_empty(val):
        return False
    return str(val).strip().lower() in ('true', '1', 'yes')


def parse_scalar_int(val):
    return int(float(val))


def parse_scalar_float(val):
    return float(val)


def _rope_dim_from_shape(shape_val):
    shape = parse_shape(shape_val)
    if shape and len(shape) >= 1:
        return shape[-1]
    return 0


def _detect_kv_storage(k_cache_shape, layout_kv_hint=None):
    if layout_kv_hint and layout_kv_hint in KV_STORAGE_MAP:
        return KV_STORAGE_MAP[layout_kv_hint]
    if is_empty(k_cache_shape):
        return 'continue'
    hint = str(k_cache_shape).upper()
    if 'NZ' in hint:
        return 'pa_nz'
    if 'BNBD' in hint or len(parse_shape(k_cache_shape) or ()) == 4:
        return 'pa_bnbd'
    return 'pa_bbh'


def _resolve_act_seq_lens(raw_val, b, default_s):
    """将 actual_seq_lengths 标量或列表转为 per-batch 长度列表。"""
    if is_empty(raw_val):
        return [default_s] * b
    if isinstance(raw_val, (list, tuple)):
        vals = [int(x) for x in raw_val]
    else:
        s = str(raw_val).strip()
        if ',' in s:
            vals = parse_int_list(s)
        else:
            vals = [int(float(s))]
    if len(vals) == 1:
        return vals * b
    return vals


# ─────────────────── xlsx 行 → case dict ───────────────────

def convert_xlsx_row(headers, row_values, row_idx):
    raw = {h: row_values[i] for i, h in enumerate(headers) if h and i < len(row_values)}
    if all(is_empty(v) for v in raw.values()):
        return None, None

    case_name = None
    for key in ('_case_name_',):
        pass
    for col in ('Testcase_Name', 'testcase_name'):
        if col in raw and not is_empty(raw[col]):
            case_name = str(raw[col]).strip()
            break
    if case_name is None:
        case_name = f"row_{row_idx}"

    case = {'_case_name': case_name}
    specials = {}

    for col, val in raw.items():
        if col not in XLSX_COLUMN_MAP:
            continue
        key = XLSX_COLUMN_MAP[col]
        if key is None or (isinstance(key, str) and key.startswith('_')):
            if key == '_case_name_':
                case_name = str(val).strip()
                case['_case_name'] = case_name
            elif key in ('_q_rope_shape_', '_k_rope_shape_') and not is_empty(val):
                specials['rope_d'] = _rope_dim_from_shape(val)
            elif key == '_m_shape_' and not is_empty(val):
                specials['enable_mask'] = True
                specials['mask_shape'] = parse_shape(val)
            elif key == '_k_cache_shape_' and not is_empty(val):
                specials['k_cache_shape'] = str(val).strip()
            elif key == '_Q_N_' and not is_empty(val):
                case.setdefault('N1', parse_scalar_int(val))
            elif key == '_KV_N_' and not is_empty(val):
                case.setdefault('N2', parse_scalar_int(val))
            continue
        if is_empty(val):
            continue

        val_str = str(val).strip()
        if key in ('B', 'N1', 'N2', 'S1', 'S2', 'D', 'DV', 'K_D', 'mask_mode',
                   'block_size', 'win_left', 'win_right', 'inner_precise',
                   'q_start_idx', 'q_padding_size', 'kv_padding_size', 'G', 'QS', 'KVS'):
            case[key] = parse_scalar_int(val)
        elif key in ('scale',):
            case[key] = parse_scalar_float(val)
        elif key in ('input_layout', 'layout_q', 'layout_kv', 'layout_out'):
            case[key] = val_str
        elif key == 'Dtype':
            case[key] = DTYPE_MAP.get(val_str.upper(), val_str.lower())
        elif key in ('q_range', 'k_range', 'v_range'):
            r = parse_data_range(val)
            if r:
                case[key] = r
        elif key in ('return_softmax_lse', 'd_unequal', 'mla'):
            case[key] = parse_bool(val)
        elif key in ('seqused_q', 'seqused_kv', 'cu_seqlens_q', 'cu_seqlens_kv',
                     'actual_seq_lengths_q', 'actual_seq_lengths_kv'):
            case[key] = parse_int_list(val_str) if ',' in val_str else [parse_scalar_int(val)]
        elif key == 'block_table_shape':
            case[key] = list(parse_shape(val_str) or [])
        else:
            case[key] = val_str

    case.update(specials)

    # inputLayout 同时设置 layout_q/kv/out（若未单独指定）
    layout = case.get('input_layout') or case.get('layout_q')
    if layout:
        base_layout = layout.split('_')[0] if '_' in layout else layout
        case.setdefault('layout_q', base_layout)
        case.setdefault('layout_kv', base_layout)
        case.setdefault('layout_out', base_layout)
        case['input_layout'] = layout

    # B / N1 / N2 / S1 / S2 / D 互补推导
    b = case.get('B', 1)
    case.setdefault('B', b)
    case.setdefault('N1', case.get('N1', 1))
    case.setdefault('N2', case.get('N2', case['N1']))
    case.setdefault('S1', case.get('QS', case.get('S1', 128)))
    case.setdefault('S2', case.get('KVS', case.get('S2', case['S1'])))
    case.setdefault('D', case.get('D', 128))
    case.setdefault('DV', case.get('DV', case['D']))
    case.setdefault('Dtype', case.get('Dtype', 'bf16'))
    case.setdefault('mask_mode', case.get('mask_mode', 0))
    case.setdefault('win_left', case.get('win_left', 2147483647))
    case.setdefault('win_right', case.get('win_right', 2147483647))
    case.setdefault('scale', case.get('scale', 1.0 / math.sqrt(case['D'])))
    case.setdefault('return_softmax_lse', case.get('return_softmax_lse', False))
    case.setdefault('inner_precise', case.get('inner_precise', 0))
    case.setdefault('rope_d', case.get('rope_d', 0))

    # actual_seq_lengths → seqused / cu_seqlens（TND）
    layout_val = case.get('layout_q', 'BNSD')
    if layout_val == 'TND':
        act_q = case.pop('actual_seq_lengths_q', None) or case.get('seqused_q')
        act_kv = case.pop('actual_seq_lengths_kv', None) or case.get('seqused_kv')
        if act_q and 'cu_seqlens_q' not in case:
            seq_q = _resolve_act_seq_lens(act_q, b, case['S1'])
            case['seqused_q'] = seq_q
            case['cu_seqlens_q'] = seqlens_to_prefix_sum(seq_q)
        if act_kv and 'cu_seqlens_kv' not in case:
            seq_kv = _resolve_act_seq_lens(act_kv, b, case['S2'])
            case['seqused_kv'] = seq_kv
            if case.get('layout_kv', layout_val) not in ('PA_BBND', 'PA_BNBD', 'PA_NZ'):
                case['cu_seqlens_kv'] = seqlens_to_prefix_sum(seq_kv)

    # PA 模式
    kv_hint = case.get('layout_kv', '')
    kv_storage = _detect_kv_storage(case.get('k_cache_shape', ''), kv_hint)
    if kv_storage != 'continue':
        pa_map = {'pa_bbh': 'PA_BBND', 'pa_bnbd': 'PA_BNBD', 'pa_nz': 'PA_NZ'}
        case['layout_kv'] = pa_map[kv_storage]
        case['kv_storage_mode'] = kv_storage
        if case.get('block_table_shape'):
            bt = generate_block_table(','.join(str(x) for x in case['block_table_shape']))
            if bt:
                case['block_table'] = bt
    else:
        case['kv_storage_mode'] = 'continue'

    # mask
    if case.get('enable_mask') or (case.get('mask_mode', 0) != 0):
        case['enable_mask'] = True

    return case_name, case


def load_cases_from_xlsx(xlsx_path, sheet_name=None, row_filter=None):
    if openpyxl is None:
        raise RuntimeError("需要安装 openpyxl: pip install openpyxl")
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb[sheet_name] if sheet_name else wb.active
    headers = [str(c.value).strip() if c.value else '' for c in ws[1]]
    cases = {}
    for row_idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        if row_filter is not None and row_idx != row_filter:
            continue
        name, case = convert_xlsx_row(headers, list(row), row_idx)
        if case is None:
            continue
        cases[name] = case
    wb.close()
    return cases


# ─────────────────── flash_attn normalize（复用 test_flash_attn 逻辑） ───────────────────

def normalize_case(raw):
    c = dict(raw)
    layout_q = c.get("layout_q", c.get("input_layout", "BNSD"))
    if '_' in str(layout_q):
        layout_q = str(layout_q).split('_')[0]
    c["input_layout"] = layout_q
    c.setdefault("layout_kv", c.get("layout_kv", layout_q))
    c.setdefault("layout_out", c.get("layout_out", layout_q))
    c.setdefault("N1", c.get("N1", 1))
    c.setdefault("N2", c.get("N2", c["N1"]))
    c.setdefault("S2", c.get("S2", c.get("S1")))
    c.setdefault("DV", c.get("DV", c.get("D")))
    c.setdefault("DRope", c.get("rope_d", 0))
    c.setdefault("q_start_idx", 0)
    c.setdefault("win_left", 2147483647)
    c.setdefault("win_right", 2147483647)
    c.setdefault("prefix", [])
    c.setdefault("q_range", None)
    c.setdefault("k_range", None)
    c.setdefault("v_range", None)
    if isinstance(c.get("return_softmax_lse"), bool):
        c["return_softmax_lse"] = 1 if c["return_softmax_lse"] else 0
    for key in ("cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv"):
        if c.get(key) == [None] or c.get(key) is None:
            c.pop(key, None)
    if layout_q == "TND":
        if "cu_seqlens_q" in c:
            c.setdefault("seqused_q", [c["cu_seqlens_q"][i + 1] - c["cu_seqlens_q"][i]
                                       for i in range(len(c["cu_seqlens_q"]) - 1)])
        if c.get("layout_kv", layout_q) not in ("PA_BBND", "PA_BNBD", "PA_NZ"):
            if "cu_seqlens_kv" in c:
                c.setdefault("cu_seqlens_kv", list(c.get("cu_seqlens_q", [])))
            if "seqused_kv" in c and "cu_seqlens_kv" in c:
                c.setdefault("seqused_kv", [c["cu_seqlens_kv"][i + 1] - c["cu_seqlens_kv"][i]
                                             for i in range(len(c["cu_seqlens_kv"]) - 1)])
        c["B"] = 1
    layout_kv_val = c.get("layout_kv", layout_q)
    if layout_kv_val in ("PA_BBND", "PA_BNBD", "PA_NZ"):
        c.setdefault("cu_seqlens_kv", None)
        c.setdefault("block_size", c.get("block_size", 16))
        block_size = c.get("block_size", 1)
        block_table_shape = c.get("block_table_shape", [])
        if "seqused_kv" not in c:
            c["seqused_kv"] = [c.get("S2", c.get("S1"))] * c.get("B", 1)
        seqused_kv = c.get("seqused_kv")
        if c.get("block_table") is not None:
            c["block_table"] = torch.tensor(c["block_table"], dtype=torch.int32)
        else:
            b_val = c.get("B", 1)
            if block_table_shape:
                b_val = block_table_shape[0]
                max_block_num = block_table_shape[1]
            else:
                if layout_q == "TND":
                    b_val = len(c["cu_seqlens_q"]) - 1
                max_block_num = (max(seqused_kv) + block_size - 1) // block_size
            block_table = torch.full((b_val, max_block_num), -1, dtype=torch.int32)
            block_idx = 0
            for i in range(b_val):
                b_seq = seqused_kv[i] if len(seqused_kv) > i else seqused_kv[0]
                b_block_num = (b_seq + block_size - 1) // block_size
                for j in range(b_block_num):
                    block_table[i][j] = block_idx
                    block_idx += 1
            c["block_table"] = block_table
    return c


# ─────────────────── flash_attn 多轮执行 ───────────────────

def run_flash_attn(case_name, case, rounds, device_id=0, check_cpu=False, warmup=1):
    kwargs = normalize_case(case)
    b = kwargs.get("B", 1)
    n1, n2 = kwargs.get("N1"), kwargs.get("N2", kwargs.get("N1"))
    sq, skv = kwargs.get("S1"), kwargs.get("S2", kwargs.get("S1"))
    d, d_v = kwargs.get("D"), kwargs.get("DV", kwargs.get("D"))
    d_rope = kwargs.get("DRope", 0)
    layout = kwargs.get("input_layout")
    dtype = kwargs.get("Dtype", "bf16")
    pttype = torch.float16 if dtype == "fp16" else torch.bfloat16

    sq_gen, skv_gen = sq, skv
    if layout == "TND":
        cu_q = kwargs["cu_seqlens_q"]
        sq = max(cu_q[i + 1] - cu_q[i] for i in range(len(cu_q) - 1))
        sq_gen = cu_q[-1]
        layout_kv = kwargs.get("layout_kv", layout)
        if layout_kv in ("PA_BBND", "PA_BNBD", "PA_NZ"):
            skv = max(kwargs["seqused_kv"])
            skv_gen = sum(kwargs["seqused_kv"])
        else:
            cu_kv = kwargs["cu_seqlens_kv"]
            skv = max(cu_kv[i + 1] - cu_kv[i] for i in range(len(cu_kv) - 1))
            skv_gen = cu_kv[-1]
        kwargs["S1"], kwargs["S2"] = sq, skv

    q, k, v, q_rope, k_rope, qf, kf = generate_qkv(
        b, n1, n2, sq_gen, skv_gen, d, d_v, d_rope, layout, pttype,
        q_range=kwargs.get("q_range"), k_range=kwargs.get("k_range"),
        v_range=kwargs.get("v_range"))

    mask = generate_npu_mask(b, sq, skv, kwargs.get("mask_mode"),
                             kwargs.get("win_left"), kwargs.get("win_right"))

    if check_cpu:
        out_cpu, _, _ = tforward(qf, kf, v, **kwargs)
        out_cpu = trans_bnsd_to_layout(out_cpu, kwargs.get("layout_out"), **kwargs)

    if torch_npu is None:
        raise RuntimeError("flash_attn 需要 torch_npu")

    torch.npu.set_device(device_id)
    torch.npu.synchronize()

    # warmup
    for _ in range(warmup):
        flash_attn_npu(q, k, v, q_rope, k_rope, mask, **kwargs)

    times = []
    last_out = None
    for i in range(rounds):
        t0 = time.perf_counter()
        last_out, _ = flash_attn_npu(q, k, v, q_rope, k_rope, mask, **kwargs)
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        print(f"  [{case_name}] flash_attn round {i + 1}/{rounds}: {elapsed:.3f} ms")

    result = {
        'operator': 'flash_attn',
        'case': case_name,
        'rounds': rounds,
        'times_ms': times,
        'avg_ms': sum(times) / len(times) if times else 0,
        'min_ms': min(times) if times else 0,
        'max_ms': max(times) if times else 0,
    }

    if check_cpu and last_out is not None:
        passed = check_result(case_name, out_cpu.float(), last_out.float(),
                              except_label="CPU", comp_label="NPU")
        result['cpu_check'] = passed.get('passed', passed) if isinstance(passed, dict) else bool(passed)

    return result


# ─────────────────── FIA：本文件内自包含实现，直接调 torch_npu 接口 ───────────────────

def _fia_torch_dtype(dtype_str):
    if not dtype_str:
        return torch.bfloat16
    s = DTYPE_MAP.get(str(dtype_str).upper(), str(dtype_str).lower())
    return torch.float16 if s == 'fp16' else torch.bfloat16


def _fia_query_layout(input_layout):
    mapping = {
        'BSH': 'BSH', 'BSH_BNSD': 'BSH', 'BSH_NBSD': 'BSH',
        'BSND': 'BSND', 'BSND_BNSD': 'BSND', 'BSND_NBSD': 'BSND',
        'BNSD': 'BNSD', 'BNSD_BSND': 'BNSD', 'BNSD_NBSD': 'BNSD',
        'TND': 'TND', 'TND_NTD': 'TND',
        'NTD': 'NTD', 'NTD_TND': 'NTD',
    }
    return mapping.get(input_layout, input_layout.split('_')[0] if input_layout else 'BNSD')


def _fia_normalize_case(raw):
    """FIA 专用参数归一化（不调用 flash_attn 的 normalize_case）。"""
    c = dict(raw)
    layout_q = c.get('layout_q', c.get('input_layout', 'BNSD'))
    if '_' in str(layout_q):
        layout_q = str(layout_q).split('_')[0]
    c['layout_q'] = layout_q
    c.setdefault('layout_kv', layout_q)
    c.setdefault('B', 1)
    c.setdefault('N1', 1)
    c.setdefault('N2', c['N1'])
    c.setdefault('S1', 128)
    c.setdefault('S2', c['S1'])
    c.setdefault('D', 128)
    c.setdefault('DV', c['D'])
    c.setdefault('rope_d', 0)
    c.setdefault('mask_mode', 0)
    c.setdefault('win_left', 2147483647)
    c.setdefault('win_right', 2147483647)
    c.setdefault('inner_precise', 0)
    c.setdefault('kv_storage_mode', 'continue')
    c.setdefault('scale', 1.0 / math.sqrt(c['D']))

    if layout_q == 'TND':
        if 'cu_seqlens_q' not in c and c.get('seqused_q'):
            c['cu_seqlens_q'] = seqlens_to_prefix_sum(c['seqused_q'])
        if c.get('cu_seqlens_q'):
            c.setdefault('seqused_q', [c['cu_seqlens_q'][i + 1] - c['cu_seqlens_q'][i]
                                         for i in range(len(c['cu_seqlens_q']) - 1)])
        kv_layout = c.get('layout_kv', layout_q)
        if kv_layout not in ('PA_BBND', 'PA_BNBD', 'PA_NZ'):
            if 'cu_seqlens_kv' not in c and c.get('seqused_kv'):
                c['cu_seqlens_kv'] = seqlens_to_prefix_sum(c['seqused_kv'])
            if 'cu_seqlens_kv' not in c and c.get('cu_seqlens_q'):
                c['cu_seqlens_kv'] = list(c['cu_seqlens_q'])
            if c.get('cu_seqlens_kv'):
                c.setdefault('seqused_kv', [c['cu_seqlens_kv'][i + 1] - c['cu_seqlens_kv'][i]
                                             for i in range(len(c['cu_seqlens_kv']) - 1)])

    kv_layout = c.get('layout_kv', layout_q)
    if kv_layout in ('PA_BBND', 'PA_BNBD', 'PA_NZ') or c.get('kv_storage_mode') != 'continue':
        pa_rev = {'PA_BBND': 'pa_bbh', 'PA_BNBD': 'pa_bnbd', 'PA_NZ': 'pa_nz'}
        c['kv_storage_mode'] = pa_rev.get(kv_layout, c.get('kv_storage_mode', 'pa_bnbd'))
        c.setdefault('block_size', 16)
        c.setdefault('seqused_kv', [c['S2']] * max(len(c.get('seqused_q', [c['S2']])), 1))
        seqused_kv = c['seqused_kv']
        block_size = c['block_size']
        block_table_shape = c.get('block_table_shape', [])
        if c.get('block_table') is not None:
            c['block_table'] = torch.tensor(c['block_table'], dtype=torch.int32)
        else:
            b_val = block_table_shape[0] if block_table_shape else (
                len(c['cu_seqlens_q']) - 1 if layout_q == 'TND' and c.get('cu_seqlens_q') else c['B'])
            max_block = block_table_shape[1] if block_table_shape else (
                (max(seqused_kv) + block_size - 1) // block_size)
            block_table = torch.full((b_val, max_block), -1, dtype=torch.int32)
            block_idx = 0
            for i in range(b_val):
                b_seq = seqused_kv[i] if i < len(seqused_kv) else seqused_kv[0]
                for j in range((b_seq + block_size - 1) // block_size):
                    block_table[i][j] = block_idx
                    block_idx += 1
            c['block_table'] = block_table
    return c


def _fia_rand_tensor(shape, data_range, dtype, seed):
    gen = torch.Generator().manual_seed(seed)
    if data_range is not None:
        lo, hi = float(data_range[0]), float(data_range[1])
        return (torch.rand(shape, generator=gen) * (hi - lo) + lo).to(dtype)
    return torch.randint(10, 11, shape, generator=gen).to(dtype)


def _fia_generate_qkv(b, n1, n2, s1, s2, d, d_v, d_rope, dtype, q_range, k_range, v_range):
    q = _fia_rand_tensor((b, n1, s1, d), q_range, dtype, 42)
    k = _fia_rand_tensor((b, n2, s2, d), k_range, dtype, 43)
    v = _fia_rand_tensor((b, n2, s2, d_v), v_range, dtype, 44)
    q_rope = _fia_rand_tensor((b, n1, s1, d_rope), q_range, dtype, 45) if d_rope else None
    k_rope = _fia_rand_tensor((b, n2, s2, d_rope), k_range, dtype, 46) if d_rope else None
    return q, k, v, q_rope, k_rope


def _fia_bnsd_to_bsnd(t):
    return t.permute(0, 2, 1, 3).contiguous()


def _fia_bnsd_to_bsh(t):
    b, n, s, d = t.shape
    return t.permute(0, 2, 1, 3).reshape(b, s, n * d).contiguous()


def _fia_bnsd_to_tnd(t, b, act_seq_lens):
    if not act_seq_lens or len(act_seq_lens) == 1:
        length = act_seq_lens[0] if act_seq_lens else t.shape[2]
        return torch.narrow(t, 2, 0, length).permute(0, 2, 1, 3).reshape(-1, t.shape[1], t.shape[3]).contiguous()
    total = sum(act_seq_lens)
    out = torch.empty(total, t.shape[1], t.shape[3], dtype=t.dtype)
    offset = 0
    for i in range(b):
        length = act_seq_lens[i]
        if length > 0:
            out[offset:offset + length] = t[i, :, :length, :].permute(1, 0, 2)
            offset += length
    return out.contiguous()


def _fia_bnsd_to_ntd(t, b, act_seq_lens):
    tnd = _fia_bnsd_to_tnd(t, b, act_seq_lens)
    return tnd.permute(1, 0, 2).contiguous()


def _fia_to_layout(tensor, layout, b, act_seq_lens=None):
    if tensor is None:
        return None
    if layout == 'BNSD':
        return tensor.contiguous()
    if layout == 'BSND':
        return _fia_bnsd_to_bsnd(tensor)
    if layout == 'BSH':
        return _fia_bnsd_to_bsh(tensor)
    if layout == 'TND':
        return _fia_bnsd_to_tnd(tensor, b, act_seq_lens)
    if layout == 'NTD':
        return _fia_bnsd_to_ntd(tensor, b, act_seq_lens)
    return tensor.contiguous()


def _fia_page_attn_bnsd(bnsd_tensor, b, act_seq_lens_kv, block_table, block_size):
    block_num = int(block_table.max().item()) + 1
    cache = torch.zeros(block_num, bnsd_tensor.shape[1], block_size, bnsd_tensor.shape[3],
                        dtype=bnsd_tensor.dtype)
    cum_offset = 0
    for i in range(block_table.shape[0]):
        b_seq = act_seq_lens_kv[i] if i < len(act_seq_lens_kv) else act_seq_lens_kv[0]
        batch_dim = 0 if b == 1 else i
        seq_base = cum_offset if b == 1 and block_table.shape[0] > 1 else 0
        for j in range((b_seq + block_size - 1) // block_size):
            start = seq_base + j * block_size
            end = min(start + block_size, seq_base + b_seq)
            cache[block_table[i][j].item(), :, :end - start, :] = bnsd_tensor[batch_dim, :, start:end, :]
        cum_offset += b_seq
    return cache


def _fia_to_pa_layout(tensor, pa_mode, block_table, block_size, b, act_seq_lens_kv, dtype_str):
    cache = _fia_page_attn_bnsd(tensor, b, act_seq_lens_kv, block_table, block_size)
    if pa_mode == 'pa_bbh':
        return _fia_bnsd_to_bsh(cache)
    if pa_mode == 'pa_bnbd':
        return cache.contiguous()
    if pa_mode == 'pa_nz':
        elem = 32 // (2 if dtype_str in ('bf16', 'fp16') else 1)
        return cache.reshape(cache.shape[0], cache.shape[1], cache.shape[2],
                             cache.shape[3] // elem, elem).permute(0, 1, 3, 2, 4).contiguous()
    return cache.contiguous()


def _fia_update_act_seq_lens_tnd(layout, act_seq_lens):
    if layout not in ('TND', 'NTD') or not act_seq_lens:
        return act_seq_lens
    result, total = [], 0
    for length in act_seq_lens:
        total += length
        result.append(total)
    return result


def _fia_build_mask(sparse_mode):
    if sparse_mode == 0:
        return None
    if sparse_mode in (2, 3, 4):
        return torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1)
    return None


def _fia_to_npu(tensor, device):
    return tensor.to(device).contiguous() if tensor is not None else None


def _fia_prepare_inputs(case, device_id):
    """在本文件内完成 FIA 全部数据准备，不依赖 flash_attn utils/backends。"""
    c = _fia_normalize_case(case)
    device = torch.device(f"npu:{device_id}")
    dtype_str = c.get('Dtype', 'bf16')
    pttype = _fia_torch_dtype(dtype_str)

    b = c['B']
    n2 = c['N2']
    g = c.get('G', c['N1'] // max(n2, 1))
    n1 = n2 * g
    s1, s2 = c['S1'], c['S2']
    d, d_v = c['D'], c['DV']
    d_rope = c.get('rope_d', 0)
    input_layout = c.get('input_layout', c.get('layout_q', 'BNSD'))
    query_layout = _fia_query_layout(input_layout)
    kv_storage = c.get('kv_storage_mode', 'continue')

    gen_b = 1 if query_layout == 'TND' else b
    sq_gen, skv_gen = s1, s2
    if query_layout == 'TND' and c.get('cu_seqlens_q'):
        cu_q = c['cu_seqlens_q']
        s1 = max(cu_q[i + 1] - cu_q[i] for i in range(len(cu_q) - 1))
        sq_gen = cu_q[-1]
        if kv_storage != 'continue':
            skv_gen = sum(c['seqused_kv'])
            s2 = max(c['seqused_kv'])
        elif c.get('cu_seqlens_kv'):
            cu_kv = c['cu_seqlens_kv']
            s2 = max(cu_kv[i + 1] - cu_kv[i] for i in range(len(cu_kv) - 1))
            skv_gen = cu_kv[-1]

    q, k, v, q_rope, k_rope = _fia_generate_qkv(
        gen_b, n1, n2, sq_gen, skv_gen, d, d_v, d_rope, pttype,
        c.get('q_range'), c.get('k_range'), c.get('v_range'))

    act_q = list(c.get('seqused_q') or [s1] * (len(c['cu_seqlens_q']) - 1 if c.get('cu_seqlens_q') else b))
    act_kv = list(c.get('seqused_kv') or [s2] * len(act_q))

    query = _fia_to_layout(q, query_layout, gen_b, act_q)
    query_rope = _fia_to_layout(q_rope, query_layout, gen_b, act_q) if d_rope else None

    if kv_storage == 'continue':
        key = _fia_to_layout(k, query_layout, gen_b, act_kv)
        value = _fia_to_layout(v, query_layout, gen_b, act_kv)
        key_rope = _fia_to_layout(k_rope, query_layout, gen_b, act_kv) if d_rope else None
        block_table = None
        act_kv_fia = _fia_update_act_seq_lens_tnd(query_layout, act_kv)
    else:
        block_table = c.get('block_table')
        key = _fia_to_pa_layout(k, kv_storage, block_table, c['block_size'], gen_b, act_kv, dtype_str)
        value = _fia_to_pa_layout(v, kv_storage, block_table, c['block_size'], gen_b, act_kv, dtype_str)
        key_rope = _fia_to_pa_layout(k_rope, kv_storage, block_table, c['block_size'], gen_b, act_kv, dtype_str) if d_rope else None
        act_kv_fia = list(act_kv)

    act_q_fia = _fia_update_act_seq_lens_tnd(query_layout, act_q)
    atten_mask = _fia_build_mask(c.get('mask_mode', 0))

    return {
        'device': device,
        'query': _fia_to_npu(query, device),
        'key': _fia_to_npu(key, device),
        'value': _fia_to_npu(value, device),
        'pse_shift': None,
        'atten_mask': _fia_to_npu(atten_mask, device),
        'actual_seq_lengths': act_q_fia,
        'actual_seq_lengths_kv': act_kv_fia,
        'dequant_scale1': None,
        'quant_scale1': None,
        'dequant_scale2': None,
        'quant_scale2': None,
        'quant_offset2': None,
        'block_table': _fia_to_npu(block_table, device),
        'query_padding_size': None,
        'kv_padding_size': None,
        'key_antiquant_scale': None,
        'key_antiquant_offset': None,
        'value_antiquant_scale': None,
        'value_antiquant_offset': None,
        'key_shared_prefix': None,
        'value_shared_prefix': None,
        'actual_shared_prefix_len': None,
        'query_rope': _fia_to_npu(query_rope, device),
        'key_rope': _fia_to_npu(key_rope, device),
        'num_heads': n1,
        'scale': c['scale'],
        'pre_tokens': c['win_left'],
        'next_tokens': c['win_right'],
        'input_layout': input_layout,
        'num_key_value_heads': n2,
        'sparse_mode': c['mask_mode'],
        'inner_precise': c['inner_precise'],
        'block_size': c.get('block_size', 0) or 0,
        'antiquant_mode': 0,
        'softmax_lse_flag': bool(c.get('return_softmax_lse', False)),
        'key_antiquant_mode': 0,
        'value_antiquant_mode': 0,
    }


def call_fia_npu(inputs):
    """直接调用 torch_npu.npu_fused_infer_attention_score，全部参数在本文件显式传入。"""
    return torch_npu.npu_fused_infer_attention_score(
        inputs['query'],
        inputs['key'],
        inputs['value'],
        pse_shift=inputs['pse_shift'],
        atten_mask=inputs['atten_mask'],
        actual_seq_lengths=inputs['actual_seq_lengths'],
        actual_seq_lengths_kv=inputs['actual_seq_lengths_kv'],
        dequant_scale1=inputs['dequant_scale1'],
        quant_scale1=inputs['quant_scale1'],
        dequant_scale2=inputs['dequant_scale2'],
        quant_scale2=inputs['quant_scale2'],
        quant_offset2=inputs['quant_offset2'],
        block_table=inputs['block_table'],
        query_padding_size=inputs['query_padding_size'],
        kv_padding_size=inputs['kv_padding_size'],
        key_antiquant_scale=inputs['key_antiquant_scale'],
        key_antiquant_offset=inputs['key_antiquant_offset'],
        value_antiquant_scale=inputs['value_antiquant_scale'],
        value_antiquant_offset=inputs['value_antiquant_offset'],
        key_shared_prefix=inputs['key_shared_prefix'],
        value_shared_prefix=inputs['value_shared_prefix'],
        actual_shared_prefix_len=inputs['actual_shared_prefix_len'],
        query_rope=inputs['query_rope'],
        key_rope=inputs['key_rope'],
        num_heads=inputs['num_heads'],
        scale=inputs['scale'],
        pre_tokens=inputs['pre_tokens'],
        next_tokens=inputs['next_tokens'],
        input_layout=inputs['input_layout'],
        num_key_value_heads=inputs['num_key_value_heads'],
        sparse_mode=inputs['sparse_mode'],
        inner_precise=inputs['inner_precise'],
        block_size=inputs['block_size'],
        antiquant_mode=inputs['antiquant_mode'],
        softmax_lse_flag=inputs['softmax_lse_flag'],
        key_antiquant_mode=inputs['key_antiquant_mode'],
        value_antiquant_mode=inputs['value_antiquant_mode'],
    )


def run_fia(case_name, case, rounds, device_id=0, check_cpu=False, warmup=1):
    if torch_npu is None:
        raise RuntimeError("FIA 需要 torch_npu")

    torch.npu.set_device(device_id)
    fia_inputs = _fia_prepare_inputs(case, device_id)

    for _ in range(warmup):
        call_fia_npu(fia_inputs)
    torch.npu.synchronize()

    times = []
    for i in range(rounds):
        t0 = time.perf_counter()
        call_fia_npu(fia_inputs)
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        print(f"  [{case_name}] fia round {i + 1}/{rounds}: {elapsed:.3f} ms")

    return {
        'operator': 'fia',
        'case': case_name,
        'rounds': rounds,
        'times_ms': times,
        'avg_ms': sum(times) / len(times) if times else 0,
        'min_ms': min(times) if times else 0,
        'max_ms': max(times) if times else 0,
    }


# ─────────────────── 统一入口 ───────────────────

def run_case(case_name, case, operator, rounds, device_id=0, check_cpu=False, warmup=1):
    print(f"\n{'=' * 60}")
    print(f"Case: {case_name}  operator={operator}  rounds={rounds}")
    print(f"  layout={case.get('layout_q', case.get('input_layout'))}  "
          f"B={case.get('B')} N1={case.get('N1')} N2={case.get('N2')} "
          f"S1={case.get('S1')} S2={case.get('S2')} D={case.get('D')} "
          f"dtype={case.get('Dtype')} sparse={case.get('mask_mode')}")
    print(f"{'=' * 60}")

    if operator == 'flash_attn':
        return run_flash_attn(case_name, case, rounds, device_id, check_cpu, warmup)
    if operator == 'fia':
        return run_fia(case_name, case, rounds, device_id, check_cpu, warmup)
    raise ValueError(f"未知算子: {operator}，可选: flash_attn, fia")


def print_summary(all_results):
    print(f"\n{'=' * 70}")
    print("  运行汇总")
    print(f"{'=' * 70}")
    for r in all_results:
        op = r['operator']
        if op == 'flash_attn':
            print(f"  {r['case']:<30} flash_attn  "
                  f"avg={r['avg_ms']:.3f}ms  min={r['min_ms']:.3f}ms  max={r['max_ms']:.3f}ms")
        else:
            print(f"  {r['case']:<30} fia         "
                  f"avg={r['avg_ms']:.3f}ms  min={r['min_ms']:.3f}ms  max={r['max_ms']:.3f}ms")
    print(f"{'=' * 70}")


# ─────────────────── pytest 集成 ───────────────────

def pytest_addoption(parser):
    parser.addoption("--xlsx", action="store", default=None, help="xlsx 参数表路径")
    parser.addoption("--operator", action="store", default="flash_attn",
                     choices=["flash_attn", "fia"], help="算子选择")
    parser.addoption("--rounds", action="store", type=int, default=1, help="算子运行轮次")
    parser.addoption("--row", action="store", type=int, default=None, help="只运行指定行号")
    parser.addoption("--sheet", action="store", default=None, help="xlsx sheet 名")
    parser.addoption("--device-id", action="store", type=int, default=0, help="NPU device id")
    parser.addoption("--check-cpu", action="store_true", help="flash_attn 模式下做 CPU 精度对比")
    parser.addoption("--warmup", action="store", type=int, default=1, help="flash_attn 预热轮次")


def pytest_generate_tests(metafunc):
    if "case_name" not in metafunc.fixturenames:
        return
    xlsx = metafunc.config.getoption("--xlsx")
    if not xlsx:
        return
    row = metafunc.config.getoption("--row")
    sheet = metafunc.config.getoption("--sheet")
    cases = load_cases_from_xlsx(xlsx, sheet, row)
    metafunc.parametrize("case_name,case_params", list(cases.items()))


def test_xlsx_case(case_name, case_params, request):
    xlsx = request.config.getoption("--xlsx")
    if not xlsx:
        import pytest
        pytest.skip("需要 --xlsx 参数")
    operator = request.config.getoption("--operator")
    rounds = request.config.getoption("--rounds")
    device_id = request.config.getoption("--device-id")
    check_cpu = request.config.getoption("--check-cpu")
    warmup = request.config.getoption("--warmup")
    run_case(case_name, case_params, operator, rounds, device_id, check_cpu, warmup)


# ─────────────────── CLI main ───────────────────

def main():
    parser = argparse.ArgumentParser(description="精简版 xlsx 驱动 flash_attn / FIA 测试")
    parser.add_argument("--xlsx", "-i", required=True, help="xlsx 参数表路径")
    parser.add_argument("--operator", "-o", default="flash_attn",
                        choices=["flash_attn", "fia"], help="算子: flash_attn 或 fia")
    parser.add_argument("--rounds", "-n", type=int, default=1, help="算子运行轮次")
    parser.add_argument("--row", type=int, default=None, help="只运行 xlsx 第 N 行（含表头，第1行为表头）")
    parser.add_argument("--sheet", "-s", default=None, help="xlsx sheet 名称")
    parser.add_argument("--case-id", default=None, help="只运行指定用例名（逗号分隔）")
    parser.add_argument("--device-id", type=int, default=0, help="NPU device id")
    parser.add_argument("--check-cpu", action="store_true", help="flash_attn 模式下对比 CPU golden")
    parser.add_argument("--warmup", type=int, default=1, help="flash_attn 预热轮次（不计入统计）")
    args = parser.parse_args()

    if not os.path.exists(args.xlsx):
        print(f"[ERROR] 文件不存在: {args.xlsx}")
        sys.exit(1)

    cases = load_cases_from_xlsx(args.xlsx, args.sheet, args.row)
    if not cases:
        print("[ERROR] 未从 xlsx 解析到有效用例")
        sys.exit(1)

    if args.case_id:
        ids = [x.strip() for x in args.case_id.split(',')]
        cases = {k: v for k, v in cases.items() if k in ids}
        if not cases:
            print(f"[ERROR] 未找到 case: {args.case_id}")
            sys.exit(1)

    print(f"[INFO] 从 {args.xlsx} 加载 {len(cases)} 个用例, operator={args.operator}, rounds={args.rounds}")

    all_results = []
    failed = False
    for name, case in cases.items():
        try:
            r = run_case(name, case, args.operator, args.rounds,
                         args.device_id, args.check_cpu, args.warmup)
            all_results.append(r)
            if args.check_cpu and r.get('cpu_check') is False:
                failed = True
        except Exception as e:
            import traceback
            print(f"[ERROR] {name} 运行失败: {e}")
            traceback.print_exc()
            failed = True

    print_summary(all_results)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
