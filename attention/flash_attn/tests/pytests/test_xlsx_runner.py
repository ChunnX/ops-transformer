#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
精简版 pytest：从 xlsx 读取参数，分别调用 flash_attn / FIA 算子，支持设定运行轮次。
flash_attn 模式按 Q/KV layout 分别解析 actual_seq_lengths：
  - Q=TND → cu_seqlens_q；Q≠TND → seqused_q
  - KV=PA_* → seqused_kv；KV=TND → cu_seqlens_kv；其他 → seqused_kv

FIA 模式在本文件内完成数据准备，通过 call_fia_npu() 直接调用
torch_npu.npu_fused_infer_attention_score，不依赖 FIA 原有测试脚本，
也不依赖 flash_attn 的 utils/backends。
FA / FIA 共用 _normalize_fa_case() 解析 seqlens，
共用 generate_qkv() + trans_bnsd_to_layout() 保证 Q/K/V shape 与 layout 一致。
TND/NTD 按 seqused 裁剪在本文件 monkey-patch utils.data / npu_impl，不修改其他源文件。

用法（直接运行）:
    python test_xlsx_runner.py --xlsx cases.xlsx --operator flash_attn --rounds 10
    python test_xlsx_runner.py --xlsx cases.xlsx --operator fia --rounds 5 --row 2
    python test_xlsx_runner.py --xlsx cases.xlsx --export-table
    python test_xlsx_runner.py --xlsx cases.xlsx --export-table --table-out summary.xlsx
    python test_xlsx_runner.py --xlsx cases.xlsx --export-py
    python test_xlsx_runner.py --xlsx cases.xlsx --export-py --py-out test_cases/my_cases.py
    python test_xlsx_runner.py --case-xlsx cases.xlsx --compare-perf --fa-perf fa_perf.xlsx --fia-perf fia_perf.xlsx
    python test_xlsx_runner.py --case-xlsx cases.csv --compare-perf --fa-perf fa_perf.csv --fia-perf fia_perf.csv
    # --compare-perf 对比全部数值型性能参数，输出 xlsx 含 duration_summary + all_metrics 两个 sheet
    # --compare-perf / --export-table / --export-py 仅读写表格，不调用算子，不需要 torch_npu

用法（pytest）:
    pytest test_xlsx_runner.py --xlsx cases.xlsx --operator flash_attn --rounds 3 -v

依赖:
    pip install openpyxl pytest torch torch_npu
"""

import argparse
import csv
import os
import re
import sys
import time
import math

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    import openpyxl
except ImportError:
    openpyxl = None

# 算子相关依赖延迟加载：--compare-perf / --export-table 不需要 torch / NPU
torch = None
torch_npu = None
generate_qkv = None
generate_npu_mask = None
trans_bnsd_to_layout = None
tforward = None
flash_attn_npu = None
check_result = None
_OPERATOR_DEPS_LOADED = False
_ORIG_TRANS_BNSD = None
_TRANS_PATCH_INSTALLED = False


def _runner_resolve_act_seq_lens(tensor, act_seq_lens, kwargs):
    """从显式 act_seq_lens 或 seqused_q/kv（结合 head 数）推断 actual seqused。"""
    if act_seq_lens is not None:
        if not isinstance(act_seq_lens, (list, tuple)):
            return [int(act_seq_lens)]
        return [int(x) for x in act_seq_lens]
    n1 = kwargs.get('N1')
    n2 = kwargs.get('N2', n1)
    raw = kwargs.get('seqused_q')
    if tensor is not None and tensor.ndim == 4 and n1 is not None and n2 is not None:
        heads = int(tensor.shape[1])
        if heads == int(n2) and heads != int(n1):
            raw = kwargs.get('seqused_kv', raw)
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        return [int(raw)]
    return [int(x) for x in raw]


def _runner_bnsd_to_tnd_packed(tensor, act_seq_lens):
    """TND 打包：按 actual seqused 裁剪后再展平，与 FIA 官方测试一致。"""
    if not act_seq_lens:
        from einops import rearrange
        return rearrange(tensor.clone(), 'b n s d -> (b s) n d')
    if len(act_seq_lens) == 1:
        length = int(act_seq_lens[0])
        return torch.narrow(tensor.clone(), 2, 0, length).permute(0, 2, 1, 3).reshape(
            -1, tensor.shape[1], tensor.shape[3]).contiguous()
    total = sum(int(x) for x in act_seq_lens)
    out = torch.empty(total, tensor.shape[1], tensor.shape[3], dtype=tensor.dtype)
    t_idx = 0
    batch_b = tensor.shape[0]
    for i, length in enumerate(act_seq_lens):
        length = int(length)
        if length <= 0:
            continue
        bi = 0 if batch_b == 1 else i
        out[t_idx:t_idx + length] = tensor[bi, :, :length, :].permute(1, 0, 2)
        t_idx += length
    return out.contiguous()


def _runner_bnsd_to_ntd_packed(tensor, act_seq_lens):
    tnd = _runner_bnsd_to_tnd_packed(tensor, act_seq_lens)
    return tnd.permute(1, 0, 2).contiguous()


def _runner_trans_bnsd_to_layout(tensor, layout_type, act_seq_lens=None, **kwargs):
    """本文件内 TND/NTD seqused 裁剪；其余 layout 委托 utils.data 原实现。"""
    if tensor is None:
        return None
    if layout_type == 'TND':
        lens = _runner_resolve_act_seq_lens(tensor, act_seq_lens, kwargs)
        return _runner_bnsd_to_tnd_packed(tensor, lens)
    if layout_type == 'NTD':
        lens = _runner_resolve_act_seq_lens(tensor, act_seq_lens, kwargs)
        return _runner_bnsd_to_ntd_packed(tensor, lens)
    return _ORIG_TRANS_BNSD(tensor, layout_type, **kwargs)


def _install_trans_bnsd_patch(data_mod):
    """monkey-patch utils.data / utils / backends.npu_impl，仅改 test_xlsx_runner 即可对齐 FA/FIA QKV。"""
    global _ORIG_TRANS_BNSD, trans_bnsd_to_layout, _TRANS_PATCH_INSTALLED
    if _TRANS_PATCH_INSTALLED:
        return data_mod.trans_bnsd_to_layout
    _ORIG_TRANS_BNSD = data_mod.trans_bnsd_to_layout
    data_mod.trans_bnsd_to_layout = _runner_trans_bnsd_to_layout
    import utils as _utils_mod
    _utils_mod.trans_bnsd_to_layout = _runner_trans_bnsd_to_layout
    from backends import npu_impl as _npu_impl_mod
    _npu_impl_mod.trans_bnsd_to_layout = _runner_trans_bnsd_to_layout
    trans_bnsd_to_layout = _runner_trans_bnsd_to_layout
    _TRANS_PATCH_INSTALLED = True
    return _runner_trans_bnsd_to_layout


def _load_operator_deps():
    """仅在运行算子时加载 torch / backends，对比性能或导出摘要表不触发。"""
    global torch, torch_npu, generate_qkv, generate_npu_mask, trans_bnsd_to_layout
    global tforward, flash_attn_npu, check_result, _OPERATOR_DEPS_LOADED
    if _OPERATOR_DEPS_LOADED:
        return
    import torch as _torch
    torch = _torch
    try:
        import torch_npu as _torch_npu
        torch_npu = _torch_npu
    except ImportError:
        torch_npu = None
    import utils.data as _data_mod
    from utils.data import generate_qkv as _generate_qkv
    from utils.data import generate_npu_mask as _generate_npu_mask
    trans_bnsd_to_layout = _install_trans_bnsd_patch(_data_mod)
    from backends.cpu_impl import tforward as _tforward
    from backends.npu_impl import flash_attn_npu as _flash_attn_npu
    from utils.compare import check_result as _check_result
    generate_qkv = _generate_qkv
    generate_npu_mask = _generate_npu_mask
    tforward = _tforward
    flash_attn_npu = _flash_attn_npu
    check_result = _check_result
    _OPERATOR_DEPS_LOADED = True

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
    """将 actual_seq_lengths 标量或列表转为 per-batch 长度列表（seqused）。"""
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


FA_PA_LAYOUTS = ('PA_BBND', 'PA_BNBD', 'PA_NZ')


def _is_pa_layout(layout):
    return layout in FA_PA_LAYOUTS


def _parse_layout_base(layout):
    if is_empty(layout):
        return 'BNSD'
    s = str(layout).strip()
    return s.split('_')[0] if '_' in s else s


def _to_int_list(raw_val):
    if is_empty(raw_val):
        return None
    if isinstance(raw_val, (list, tuple)):
        return [int(x) for x in raw_val]
    s = str(raw_val).strip()
    if ',' in s:
        return parse_int_list(s)
    return [int(float(s))]


def _fa_actual_as_cu_seqlens(raw_val, b, default_s):
    """flash_attn TND：actual_seq_lengths 视为 cu_seqlens（前缀和，长度 B+1）。"""
    vals = _to_int_list(raw_val)
    if not vals:
        return [0, default_s]
    if len(vals) == 1:
        return [0, vals[0]]
    if vals[0] == 0 and len(vals) >= 2:
        return vals
    if len(vals) == b + 1:
        return vals if vals[0] == 0 else [0] + vals
    if all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)):
        return [0] + vals if vals[0] != 0 else vals
    return seqlens_to_prefix_sum(vals)


def _fa_actual_as_seqused(raw_val, b, default_s):
    """flash_attn 非 TND-Q / PA-KV：actual_seq_lengths 视为 seqused（逐 batch 实际长度）。"""
    return _resolve_act_seq_lens(raw_val, b, default_s)


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

    # inputLayout 仅设置 Q 侧 layout；KV layout 由 layout_kv / k_cache_shape 单独决定
    layout = case.get('input_layout') or case.get('layout_q')
    if layout:
        case['layout_q'] = _parse_layout_base(layout)
        case['input_layout'] = layout
        case.setdefault('layout_out', case['layout_q'])

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

    # actual_seq_lengths 原样保留，由 flash_attn 的 _normalize_fa_case 按 Q/KV layout 分别解析

    # PA 模式（仅影响 KV layout）
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


# ─────────────────── 表格读取（xlsx / csv） ───────────────────

_CSV_EXTENSIONS = ('.csv', '.tsv', '.txt')
_XLSX_EXTENSIONS = ('.xlsx', '.xlsm')


def _is_csv_path(path):
    return os.path.splitext(path)[1].lower() in _CSV_EXTENSIONS


def _table_file_format(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in _CSV_EXTENSIONS:
        return 'csv'
    if ext in _XLSX_EXTENSIONS:
        return 'xlsx'
    return 'xlsx'


def _detect_csv_dialect(sample_text):
    tab_cnt = sample_text.count('\t')
    comma_cnt = sample_text.count(',')
    if tab_cnt > comma_cnt:
        return csv.excel_tab
    try:
        return csv.Sniffer().sniff(sample_text, delimiters=',\t;')
    except csv.Error:
        return csv.excel_tab if tab_cnt > 0 else csv.excel


def _read_csv_rows(file_path):
    last_err = None
    for enc in ('utf-8-sig', 'utf-8', 'gbk', 'gb18030'):
        try:
            with open(file_path, 'r', encoding=enc, newline='') as f:
                sample = f.read(8192)
                f.seek(0)
                dialect = _detect_csv_dialect(sample)
                reader = csv.reader(f, dialect)
                rows = []
                for row in reader:
                    if not row or all(is_empty(c) for c in row):
                        continue
                    rows.append(list(row))
                if rows:
                    print(f"[INFO] 读取 CSV: {file_path}  (encoding={enc}, {len(rows)} rows)")
                    return rows
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise ValueError(f"无法解析 CSV 文件: {file_path} ({last_err})")


def _read_xlsx_rows(file_path, sheet_name=None):
    if openpyxl is None:
        raise RuntimeError("读取 xlsx 需要安装 openpyxl: pip install openpyxl")
    wb = openpyxl.load_workbook(file_path, data_only=True)
    ws = wb[sheet_name] if sheet_name else wb.active
    sheet_title = ws.title
    rows = []
    for row in ws.iter_rows(values_only=True):
        if not row or all(is_empty(c) for c in row):
            continue
        rows.append([c for c in row])
    wb.close()
    if rows:
        print(f"[INFO] 读取 XLSX: {file_path}  (sheet={sheet_title}, {len(rows)} rows)")
    return rows


def _read_table_rows(file_path, sheet_name=None):
    """统一读取 xlsx / csv，返回二维列表（含表头行）。"""
    if _is_csv_path(file_path):
        if sheet_name:
            print(f"[WARN] CSV 文件忽略 sheet 参数: {sheet_name}")
        return _read_csv_rows(file_path)
    try:
        return _read_xlsx_rows(file_path, sheet_name)
    except Exception as e:
        err_name = type(e).__name__
        err_msg = str(e).lower()
        if err_name == 'BadZipFile' or 'not a zip file' in err_msg:
            print(f"[WARN] {file_path} 不是有效 xlsx，尝试按 CSV 读取")
            return _read_csv_rows(file_path)
        raise


def load_cases_from_table(table_path, sheet_name=None, row_filter=None):
    rows = _read_table_rows(table_path, sheet_name)
    if not rows:
        return {}
    headers = [str(c).strip() if c is not None else '' for c in rows[0]]
    cases = {}
    for row_idx, row in enumerate(rows[1:], start=2):
        if row_filter is not None and row_idx != row_filter:
            continue
        name, case = convert_xlsx_row(headers, list(row), row_idx)
        if case is None:
            continue
        cases[name] = case
    return cases


def load_cases_from_xlsx(xlsx_path, sheet_name=None, row_filter=None):
    return load_cases_from_table(xlsx_path, sheet_name, row_filter)


# ─────────────────── 用例参数摘要表 ───────────────────

CASE_SUMMARY_FIELDS = [
    'case_name', 'B', 'N1', 'N2', 'G', 'S1', 'S2',
    'mask_mode', 'D', 'q_layout', 'kv_layout', 'dtype',
]


def case_to_summary_row(case_name, case):
    """从解析后的 case 提取摘要字段。"""
    n1 = case.get('N1', '')
    n2 = case.get('N2', '')
    g = case.get('G')
    if g is None and n2 not in ('', None, 0):
        try:
            g = int(n1) // int(n2) if int(n2) != 0 else ''
        except (TypeError, ValueError):
            g = ''
    q_layout = case.get('layout_q', case.get('input_layout', ''))
    kv_layout = case.get('layout_kv', q_layout)
    if not _is_pa_layout(kv_layout):
        kv_layout = _parse_layout_base(kv_layout)
    return {
        'case_name': case_name,
        'B': case.get('B', ''),
        'N1': n1,
        'N2': n2,
        'G': g if g is not None else '',
        'S1': case.get('S1', case.get('QS', '')),
        'S2': case.get('S2', case.get('KVS', '')),
        'mask_mode': case.get('mask_mode', 0),
        'D': case.get('D', ''),
        'q_layout': q_layout,
        'kv_layout': kv_layout,
        'dtype': case.get('Dtype', ''),
    }


def build_case_summary_table(cases):
    return [case_to_summary_row(name, c) for name, c in cases.items()]


def _format_summary_cell(val):
    if val is None:
        return ''
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    return str(val)


def print_case_summary_table(rows):
    """在终端打印对齐的摘要表。"""
    if not rows:
        print("[WARN] 无可用用例摘要")
        return
    col_widths = {f: len(f) for f in CASE_SUMMARY_FIELDS}
    for row in rows:
        for f in CASE_SUMMARY_FIELDS:
            col_widths[f] = max(col_widths[f], len(_format_summary_cell(row.get(f, ''))))

    def _fmt_row(row_dict):
        return '  '.join(
            _format_summary_cell(row_dict.get(f, '')).ljust(col_widths[f])
            for f in CASE_SUMMARY_FIELDS
        )

    sep = '  '.join('-' * col_widths[f] for f in CASE_SUMMARY_FIELDS)
    print(f"\n{'=' * len(sep)}")
    print(f"  用例参数摘要表  ({len(rows)} cases)")
    print(f"{'=' * len(sep)}")
    print(_fmt_row({f: f for f in CASE_SUMMARY_FIELDS}))
    print(sep)
    for row in rows:
        print(_fmt_row(row))
    print(f"{'=' * len(sep)}\n")


def write_case_summary_csv(rows, output_path):
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CASE_SUMMARY_FIELDS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _format_summary_cell(row.get(k, '')) for k in CASE_SUMMARY_FIELDS})
    print(f"[OK] 摘要表已写入 CSV: {output_path}  ({len(rows)} rows)")


def write_case_summary_xlsx(rows, output_path, sheet_name='case_summary'):
    if openpyxl is None:
        raise RuntimeError("导出 xlsx 需要安装 openpyxl: pip install openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws.append(CASE_SUMMARY_FIELDS)
    for row in rows:
        ws.append([row.get(f, '') for f in CASE_SUMMARY_FIELDS])
    for col_idx, field in enumerate(CASE_SUMMARY_FIELDS, start=1):
        max_len = len(field)
        for row in rows:
            max_len = max(max_len, len(_format_summary_cell(row.get(field, ''))))
        col_letter = openpyxl.utils.get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = min(max_len + 2, 48)
    wb.save(output_path)
    wb.close()
    print(f"[OK] 摘要表已写入 XLSX: {output_path}  ({len(rows)} rows)")


def _default_summary_xlsx_path(input_xlsx):
    base = os.path.splitext(os.path.basename(input_xlsx))[0]
    return os.path.join(os.path.dirname(os.path.abspath(input_xlsx)), f"{base}_summary.xlsx")


def export_case_summary(cases, table_out=None, input_xlsx=None):
    rows = build_case_summary_table(cases)
    print_case_summary_table(rows)

    out_path = table_out
    if not out_path:
        if not input_xlsx:
            raise ValueError("导出摘要表需要指定 --table-out 或提供 input_xlsx 路径")
        out_path = _default_summary_xlsx_path(input_xlsx)

    ext = os.path.splitext(out_path)[1].lower()
    if ext == '.csv':
        write_case_summary_csv(rows, out_path)
    else:
        if ext != '.xlsx':
            out_path = out_path + '.xlsx'
        write_case_summary_xlsx(rows, out_path)
    return rows


# ─────────────────── 导出 test_cases 格式 .py ───────────────────

TESTCASE_PY_KEY_ORDER = [
    'B', 'N1', 'N2', 'S1', 'S2', 'D',
    'layout_q', 'layout_kv', 'layout_out',
    'Dtype',
    'q_range', 'k_range', 'v_range',
    'scale',
    'mask_mode',
    'seqused_q', 'seqused_kv',
    'cu_seqlens_q', 'cu_seqlens_kv',
    'block_table', 'block_size',
    'return_softmax_lse',
]


def _gen_pa_block_table_compact(b_val, seqused_kv, block_size):
    """PA block_table：紧凑行（无 -1 填充），与 functional_redline_infer.py 一致。"""
    table, block_idx = [], 0
    for i in range(b_val):
        b_seq = seqused_kv[i] if i < len(seqused_kv) else seqused_kv[0]
        n_blocks = (int(b_seq) + block_size - 1) // block_size
        table.append(list(range(block_idx, block_idx + n_blocks)))
        block_idx += n_blocks
    return table


def _normalize_export_case(raw):
    """导出 py 用：与 _normalize_fa_case 一致的 seqlens/layout 解析，不依赖 torch。"""
    c = dict(raw)
    layout_q = _parse_layout_base(c.get('layout_q', c.get('input_layout', 'BNSD')))
    layout_kv = c.get('layout_kv', layout_q)
    if not _is_pa_layout(layout_kv):
        layout_kv = _parse_layout_base(layout_kv)

    c['layout_q'] = layout_q
    c['layout_kv'] = layout_kv
    c.setdefault('layout_out', c.get('layout_out', layout_q))
    c.setdefault('N1', c.get('N1', 1))
    c.setdefault('N2', c.get('N2', c['N1']))
    c.setdefault('S2', c.get('S2', c.get('S1')))
    c.setdefault('Dtype', c.get('Dtype', 'bf16'))
    c.setdefault('mask_mode', c.get('mask_mode', 0))
    c.setdefault('scale', c.get('scale', 1.0 / math.sqrt(c.get('D', 128))))
    if isinstance(c.get('return_softmax_lse'), bool):
        c['return_softmax_lse'] = 1 if c['return_softmax_lse'] else 0

    b = c.get('B', 1)
    s1, s2 = c.get('S1', 128), c.get('S2', c.get('S1', 128))
    act_q = c.pop('actual_seq_lengths_q', None)
    act_kv = c.pop('actual_seq_lengths_kv', None)

    if layout_q == 'TND':
        if act_q is not None and 'cu_seqlens_q' not in c:
            c['cu_seqlens_q'] = _fa_actual_as_cu_seqlens(act_q, b, s1)
        if 'cu_seqlens_q' in c:
            c['seqused_q'] = [c['cu_seqlens_q'][i + 1] - c['cu_seqlens_q'][i]
                              for i in range(len(c['cu_seqlens_q']) - 1)]
    else:
        if act_q is not None and 'seqused_q' not in c:
            c['seqused_q'] = _fa_actual_as_seqused(act_q, b, s1)
        c.pop('cu_seqlens_q', None)

    batch_cnt = len(c['seqused_q']) if c.get('seqused_q') else b

    if _is_pa_layout(layout_kv):
        c.pop('cu_seqlens_kv', None)
        if act_kv is not None and 'seqused_kv' not in c:
            c['seqused_kv'] = _fa_actual_as_seqused(act_kv, batch_cnt, s2)
        elif 'seqused_kv' not in c:
            c['seqused_kv'] = [s2] * batch_cnt
    elif layout_kv == 'TND':
        if act_kv is not None and 'cu_seqlens_kv' not in c:
            c['cu_seqlens_kv'] = _fa_actual_as_cu_seqlens(act_kv, batch_cnt, s2)
        if 'cu_seqlens_kv' in c:
            c['seqused_kv'] = [c['cu_seqlens_kv'][i + 1] - c['cu_seqlens_kv'][i]
                               for i in range(len(c['cu_seqlens_kv']) - 1)]
        elif act_kv is not None and 'seqused_kv' not in c:
            c['seqused_kv'] = _fa_actual_as_seqused(act_kv, batch_cnt, s2)
    else:
        c.pop('cu_seqlens_kv', None)
        if act_kv is not None and 'seqused_kv' not in c:
            c['seqused_kv'] = _fa_actual_as_seqused(act_kv, batch_cnt, s2)

    for key in ('cu_seqlens_q', 'cu_seqlens_kv', 'seqused_q', 'seqused_kv'):
        if c.get(key) == [None] or c.get(key) is None:
            c.pop(key, None)

    if layout_q == 'TND' and layout_kv == 'TND':
        if 'cu_seqlens_kv' not in c and 'cu_seqlens_q' in c:
            c.setdefault('cu_seqlens_kv', list(c['cu_seqlens_q']))
            c.setdefault('seqused_kv', [c['cu_seqlens_kv'][i + 1] - c['cu_seqlens_kv'][i]
                                        for i in range(len(c['cu_seqlens_kv']) - 1)])

    if _is_pa_layout(layout_kv):
        c.pop('cu_seqlens_kv', None)
        c.setdefault('block_size', c.get('block_size', 16))
        block_size = int(c['block_size'])
        seqused_kv = c.get('seqused_kv', [s2] * batch_cnt)
        c['seqused_kv'] = seqused_kv
        bt = c.get('block_table')
        if bt is not None and hasattr(bt, 'tolist'):
            bt = bt.tolist()
        if not bt:
            block_table_shape = c.get('block_table_shape', [])
            if block_table_shape:
                bt = generate_block_table(','.join(str(x) for x in block_table_shape))
            else:
                if layout_q == 'TND' and c.get('cu_seqlens_q'):
                    b_val = len(c['cu_seqlens_q']) - 1
                else:
                    b_val = batch_cnt
                bt = _gen_pa_block_table_compact(b_val, seqused_kv, block_size)
        c['block_table'] = bt

    c['_explicit_seq_q'] = bool(
        raw.get('actual_seq_lengths_q') or raw.get('seqused_q') or raw.get('cu_seqlens_q'))
    c['_explicit_seq_kv'] = bool(
        raw.get('actual_seq_lengths_kv') or raw.get('seqused_kv') or raw.get('cu_seqlens_kv'))
    return c


def _should_export_seqused_q(norm):
    if norm.get('layout_q') == 'TND':
        return True
    if norm.get('S1') == 1:
        return True
    return norm.get('_explicit_seq_q', False)


def _should_export_seqused_kv(norm):
    if _is_pa_layout(norm.get('layout_kv')):
        return True
    if norm.get('layout_kv') == 'TND':
        return True
    if norm.get('S1') == 1:
        return True
    return norm.get('_explicit_seq_kv', False)


def _fmt_testcase_val(v):
    """将 Python 值格式化为可直接写入 .py 的字符串。"""
    if isinstance(v, bool):
        return 'True' if v else 'False'
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, tuple):
        parts = []
        for x in v:
            if isinstance(x, float) and x == int(x):
                parts.append(str(int(x)))
            else:
                parts.append(repr(x))
        return f"({', '.join(parts)})"
    if isinstance(v, list):
        inner = ', '.join(_fmt_testcase_val(x) for x in v)
        return f"[{inner}]"
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return repr(v)
    return repr(v)


def _format_testcase_case_lines(case_dict):
    lines, written = [], set()
    for k in TESTCASE_PY_KEY_ORDER:
        if k in case_dict:
            lines.append(f"        '{k}': [{_fmt_testcase_val(case_dict[k][0])}],")
            written.add(k)
    for k, v in case_dict.items():
        if k not in written:
            lines.append(f"        '{k}': [{_fmt_testcase_val(v[0])}],")
    return '\n'.join(lines)


def _case_to_testcases_entry(raw, norm):
    """将归一化 case 转为 functional_redline_infer.py 风格的 TestCases 条目。"""
    layout_q = norm.get('layout_q', 'BNSD')
    layout_kv = norm.get('layout_kv', layout_q)
    layout_out = norm.get('layout_out', layout_q)
    entry = {}

    b = len(norm['seqused_q']) if layout_q == 'TND' and norm.get('seqused_q') else norm.get('B', 1)
    entry['B'] = [int(b)]
    for k in ('N1', 'N2', 'S1', 'S2', 'D'):
        if norm.get(k) is not None:
            entry[k] = [int(norm[k])]
    for k in ('layout_q', 'layout_kv', 'layout_out'):
        entry[k] = [str(locals()[k])]
    if norm.get('Dtype'):
        entry['Dtype'] = [str(norm['Dtype'])]
    for k in ('q_range', 'k_range', 'v_range'):
        if norm.get(k) is not None:
            entry[k] = [norm[k]]
    if norm.get('scale') is not None:
        entry['scale'] = [float(norm['scale'])]
    entry['mask_mode'] = [int(norm.get('mask_mode', 0))]

    if _should_export_seqused_q(norm) and norm.get('seqused_q') is not None:
        entry['seqused_q'] = [list(norm['seqused_q'])]
    if _should_export_seqused_kv(norm) and norm.get('seqused_kv') is not None:
        entry['seqused_kv'] = [list(norm['seqused_kv'])]
    if layout_q == 'TND' and norm.get('cu_seqlens_q') is not None:
        entry['cu_seqlens_q'] = [list(norm['cu_seqlens_q'])]
    if layout_kv == 'TND' and norm.get('cu_seqlens_kv') is not None:
        entry['cu_seqlens_kv'] = [list(norm['cu_seqlens_kv'])]

    if _is_pa_layout(layout_kv):
        bt = norm.get('block_table')
        if bt is not None:
            if hasattr(bt, 'tolist'):
                bt = bt.tolist()
            entry['block_table'] = [bt]
        if norm.get('block_size') is not None:
            entry['block_size'] = [int(norm['block_size'])]

    lse = norm.get('return_softmax_lse', 0)
    if isinstance(lse, bool):
        lse = 1 if lse else 0
    entry['return_softmax_lse'] = [int(lse)]
    return entry


def _default_testcase_py_path(input_xlsx):
    base = os.path.splitext(os.path.basename(input_xlsx))[0]
    safe_base = re.sub(r'[^\w]', '_', base)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'test_cases', f'{safe_base}.py')


def export_testcase_py(cases, py_out=None, input_xlsx=None):
    """将 xlsx/csv 用例导出为 test_cases 风格的 TestCases = {...} .py 文件。"""
    if not cases:
        raise ValueError("无用例可导出")

    out_path = py_out or _default_testcase_py_path(input_xlsx)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    testcase_entries = {}
    for name, raw in cases.items():
        norm = _normalize_export_case(raw)
        testcase_entries[name] = _case_to_testcases_entry(raw, norm)

    src_name = os.path.basename(input_xlsx) if input_xlsx else 'xlsx'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('#!/usr/bin/python\n')
        f.write('# -*- coding: utf-8 -*-\n')
        f.write(f'# Auto-generated from {src_name} by test_xlsx_runner.py\n')
        f.write('# DO NOT EDIT — regenerate from source xlsx instead.\n')
        f.write('#\n')
        f.write(f'# Total cases: {len(testcase_entries)}\n')
        f.write('\n')
        f.write('TestCases = {\n')
        for name, entry in testcase_entries.items():
            safe_name = name.replace("'", "\\'")
            f.write(f"    '{safe_name}': {{\n")
            f.write(_format_testcase_case_lines(entry))
            f.write('\n    },\n')
        f.write('}\n')

    print(f"[OK] 已生成 {len(testcase_entries)} 个用例 → {out_path}")
    mod = os.path.splitext(os.path.basename(out_path))[0]
    print(f"[用法] python test_flash_attn.py --case_files test_cases.{mod}")
    print(f"[用法] python test_flash_attn.py --case_files test_cases.{mod} --case_id <用例名>")
    return out_path


# ─────────────────── FA / FIA 性能对比 ───────────────────

PERF_DURATION_HEADERS = (
    'task duration(us)', 'task duration (us)', 'duration(us)', 'task duration',
)
# Ascend profiler 导出中 Op Name 对所有行相同（如 FusedInferAttentionScore），不能作为 case 分组列
PERF_CASE_NAME_HEADERS = (
    'case_name', 'case', 'testcase_name', 'testcase name', 'test case name',
)

PERF_SUMMARY_FIELDS = CASE_SUMMARY_FIELDS + [
    'fa_avg_us', 'fa_rounds',
    'fia_avg_us', 'fia_rounds',
    'diff_us', 'ratio_pct', 'fa_vs_fia',
]
# 数值越小越好（time / duration / cycles / miss_rate 等）
PERF_LOWER_BETTER_HINTS = (
    'time', 'duration', 'wait', 'cycles', 'miss_rate', 'block num',
)
# 数值越大越好（utilization / ratio 等效率类）
PERF_HIGHER_BETTER_HINTS = (
    'utilization', 'mac_ratio', 'vec_ratio', 'scalar_ratio',
    'mte1_ratio', 'mte2_ratio', 'mte3_ratio', 'fixpipe_ratio',
)
# profiler 导出中不参与数值对比的列（ID / 描述类字符串）
PERF_NON_METRIC_HEADERS = (
    'device_id', 'model id', 'task id', 'stream id', 'context id',
    'task start time', 'op name', 'op type', 'op state', 'task type',
    'hf32 eligible', 'input shapes', 'input data types', 'input formats',
    'output shapes', 'output data types', 'output formats',
)


def _norm_header(h):
    return str(h).strip().lower() if h is not None else ''


def _find_col_index(headers, candidates):
    norm_map = {_norm_header(h): i for i, h in enumerate(headers) if h}
    for cand in candidates:
        key = cand.lower()
        if key in norm_map:
            return norm_map[key]
    for i, h in enumerate(headers):
        nh = _norm_header(h)
        for cand in candidates:
            if cand.lower() in nh:
                return i
    return None


def _parse_duration_val(val):
    if is_empty(val):
        return None
    s = str(val).strip().strip('"').strip("'").replace(',', '')
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _parse_numeric_val(val):
    return _parse_duration_val(val)


def _is_perf_metric_header(header):
    h = _norm_header(header)
    if not h:
        return False
    return not any(skip in h for skip in PERF_NON_METRIC_HEADERS)


def _detect_perf_metric_columns(headers, body_rows):
    """自动识别性能表中可聚合的数值列。"""
    metric_cols = []
    for col_idx, hdr in enumerate(headers):
        if not hdr or not _is_perf_metric_header(hdr):
            continue
        numeric_cnt = 0
        for row in body_rows:
            if col_idx < len(row) and _parse_numeric_val(row[col_idx]) is not None:
                numeric_cnt += 1
        if numeric_cnt > 0:
            metric_cols.append((col_idx, hdr))
    return metric_cols


def _metric_to_field_key(header):
    key = re.sub(r'[^\w%]+', '_', str(header).strip())
    return key.strip('_').lower() or 'metric'


def _extract_row_metrics(row, metric_cols):
    values = {}
    for col_idx, hdr in metric_cols:
        if col_idx < len(row):
            val = _parse_numeric_val(row[col_idx])
            if val is not None:
                values[hdr] = val
    return values


def _new_perf_bucket(display):
    return {'display_name': display, 'row_count': 0, 'metric_sums': {}, 'metric_counts': {}}


def _add_metrics_to_bucket(bucket, metric_values):
    bucket['row_count'] += 1
    for col, val in metric_values.items():
        bucket['metric_sums'].setdefault(col, 0.0)
        bucket['metric_counts'].setdefault(col, 0)
        bucket['metric_sums'][col] += val
        bucket['metric_counts'][col] += 1


def _finalize_perf_bucket(bucket, group_mode, duration_header):
    metrics = {}
    for col, total in bucket['metric_sums'].items():
        cnt = bucket['metric_counts'][col]
        if cnt > 0:
            metrics[col] = total / cnt
    avg_us = metrics.get(duration_header)
    if avg_us is None:
        for col, val in metrics.items():
            if _norm_header(col) in PERF_DURATION_HEADERS or 'task duration' in _norm_header(col):
                avg_us = val
                break
    return {
        'avg_us': avg_us,
        'rounds': bucket['row_count'],
        'display_name': bucket['display_name'],
        'group_mode': group_mode,
        'metrics': metrics,
    }


def _resolve_duration_header(headers, duration_col=None):
    if duration_col:
        idx = _find_col_index(headers, [duration_col])
        return headers[idx] if idx is not None else duration_col
    idx = _find_col_index(headers, list(PERF_DURATION_HEADERS))
    return headers[idx] if idx is not None else 'Task Duration(us)'


def _group_perf_parsed_rows(parsed_rows, header_names, case_idx, perf_path,
                            rounds_per_case=None, num_cases=None):
    """将解析后的性能行按 case 名或顺序分桶聚合。"""
    buckets = {}
    group_mode = ''

    if case_idx is not None and _should_group_perf_by_case_name(
            [(None, r['cname']) for r in parsed_rows]):
        group_mode = 'case_name'
        for item in parsed_rows:
            cname = item['cname']
            if not cname:
                continue
            key = _normalize_case_key(cname)
            if key not in buckets:
                buckets[key] = _new_perf_bucket(cname)
            _add_metrics_to_bucket(buckets[key], item['metrics'])
    else:
        if case_idx is not None:
            print(f"[INFO] {perf_path}: case 列 '{header_names[case_idx]}' 取值相同，"
                  f"改按行顺序分组（配合 --perf-rounds 或用例数自动推断）")
        n = len(parsed_rows)
        rpk = rounds_per_case
        if rpk is None and num_cases and num_cases > 0 and n % num_cases == 0:
            rpk = n // num_cases
            print(f"[INFO] {perf_path}: 自动推断每 case {rpk} 轮 ({n} 行 / {num_cases} cases)")
        if not rpk or rpk <= 0 or n % rpk != 0:
            raise ValueError(
                f"{perf_path}: 共 {n} 行性能数据，无法分组。"
                f"请确保表中有 case 名列，或用 --perf-rounds 指定每 case 轮次"
                f"（当前 num_cases={num_cases}, perf_rounds={rounds_per_case}）")
        group_mode = 'order_block'
        num_groups = n // rpk
        for i in range(num_groups):
            chunk = parsed_rows[i * rpk:(i + 1) * rpk]
            key = f'__order_{i}__'
            display = f'case_order_{i + 1}'
            buckets[key] = _new_perf_bucket(display)
            for item in chunk:
                _add_metrics_to_bucket(buckets[key], item['metrics'])

    return buckets, group_mode


def _find_perf_case_col_index(headers, case_col=None):
    """查找可用于 case 分组的列；profiler 导出中 Op Name 等恒定列不会被自动选用。"""
    if case_col:
        return _find_col_index(headers, [case_col])
    return _find_col_index(headers, list(PERF_CASE_NAME_HEADERS))


def _should_group_perf_by_case_name(data_rows):
    """仅当 case 列取值有多个不同 case 时才按名称分组。"""
    names = [_normalize_case_key(c) for _, c in data_rows if c]
    if not names:
        return False
    if len(set(names)) <= 1:
        return False
    return True


def _normalize_case_key(name):
    return str(name).strip().lower().replace('\\', '/')


def _aggregate_perf_from_rows(rows, perf_path, duration_col=None, case_col=None,
                              rounds_per_case=None, num_cases=None):
    """从二维表格行解析性能数据，聚合全部数值型性能参数的多轮采样。"""
    if not rows:
        return {}

    header_row_idx = 0
    dur_idx = case_idx = None
    header_names = []
    for i, row in enumerate(rows[:20]):
        hdr = [str(c).strip() if c is not None else '' for c in row]
        di = _find_col_index(hdr, [duration_col] if duration_col else PERF_DURATION_HEADERS)
        ci = _find_perf_case_col_index(hdr, case_col)
        if di is not None:
            header_row_idx = i
            dur_idx = di
            case_idx = ci
            header_names = hdr
            break
    if dur_idx is None:
        sample_hdr = [str(c).strip() for c in rows[0]] if rows else []
        raise ValueError(
            f"未在 {perf_path} 中找到 Task Duration(us) 列。"
            f" 表头示例: {sample_hdr[:12]}")

    duration_header = header_names[dur_idx]
    body_rows = [list(row) for row in rows[header_row_idx + 1:]
                 if row and not all(is_empty(c) for c in row)]
    metric_cols = _detect_perf_metric_columns(header_names, body_rows)
    if not metric_cols:
        raise ValueError(f"{perf_path}: 未识别到可聚合的数值型性能列")

    metric_names = [hdr for _, hdr in metric_cols]
    print(f"[INFO] {perf_path}: duration 列 = '{duration_header}' (idx={dur_idx})")
    print(f"[INFO] {perf_path}: 识别 {len(metric_names)} 个性能参数列")
    if case_idx is not None:
        print(f"[INFO] {perf_path}: 候选 case 列 = '{header_names[case_idx]}' (idx={case_idx})")

    parsed_rows = []
    for row in body_rows:
        metrics = _extract_row_metrics(row, metric_cols)
        if duration_header not in metrics:
            continue
        cname = None
        if case_idx is not None and case_idx < len(row) and not is_empty(row[case_idx]):
            cname = str(row[case_idx]).strip().strip('"')
        parsed_rows.append({'metrics': metrics, 'cname': cname})

    if not parsed_rows:
        raise ValueError(
            f"{perf_path}: 找到 duration 列但未解析到有效数据行，请检查 CSV 分隔符/编码")

    buckets, group_mode = _group_perf_parsed_rows(
        parsed_rows, header_names, case_idx, perf_path,
        rounds_per_case=rounds_per_case, num_cases=num_cases)

    result = {}
    for key, bucket in buckets.items():
        result[key] = _finalize_perf_bucket(bucket, group_mode, duration_header)
        result[key]['metric_names'] = metric_names
    return result


def load_perf_table(perf_path, sheet_name=None, duration_col=None, case_col=None,
                    rounds_per_case=None, num_cases=None):
    """读取性能表（xlsx / csv），聚合全部数值型性能参数的多轮平均值。

    轮次 / 分组规则（按优先级）:
      1. 性能表有 case 名列：同名行合并，rounds = 该 case 行数
      2. 无 case 名列 + 指定 rounds_per_case：按顺序每 N 行一组
      3. 无 case 名列 + 给定 num_cases：若 总行数 % num_cases == 0，则自动推断轮次

    返回: {case_key: {'avg_us', 'rounds', 'display_name', 'group_mode', 'metrics'}}
    """
    rows = _read_table_rows(perf_path, sheet_name)
    return _aggregate_perf_from_rows(
        rows, perf_path, duration_col=duration_col, case_col=case_col,
        rounds_per_case=rounds_per_case, num_cases=num_cases)


def load_perf_xlsx(perf_path, sheet_name=None, duration_col=None, case_col=None,
                   rounds_per_case=None, num_cases=None):
    return load_perf_table(perf_path, sheet_name, duration_col, case_col,
                           rounds_per_case, num_cases)


def _print_perf_load_detail(perf_data, label):
    if not perf_data:
        return
    mode = next(iter(perf_data.values())).get('group_mode', '')
    print(f"[INFO] {label} 聚合方式: {mode}")
    for key, v in perf_data.items():
        print(f"       - {v['display_name']}: {v['rounds']} 轮, avg={v['avg_us']:.3f} us")


def _match_perf_for_case(case_name, perf_data, order_idx=None):
    """将用例名与性能表中的 case 匹配。"""
    if order_idx is not None:
        order_key = f'__order_{order_idx}__'
        if order_key in perf_data:
            return perf_data[order_key]
    key = _normalize_case_key(case_name)
    if key in perf_data:
        return perf_data[key]
    base = key.rsplit('/', 1)[-1]
    for k, v in perf_data.items():
        if k.startswith('__order_'):
            continue
        if k == base or k.endswith('/' + base) or base.endswith(k.rsplit('/', 1)[-1]):
            return v
    if case_name.startswith('row_'):
        try:
            row_num = int(case_name.split('_', 1)[1])
            order_key = f'__order_{row_num - 1}__'
            if order_key in perf_data:
                return perf_data[order_key]
        except (ValueError, IndexError):
            pass
    if len(perf_data) == 1:
        return next(iter(perf_data.values()))
    return None


def _metric_lower_is_better(metric_header):
    """判断该性能参数是否越小越好（默认是）。"""
    h = _norm_header(metric_header)
    if any(x in h for x in PERF_LOWER_BETTER_HINTS):
        return True
    if any(x in h for x in PERF_HIGHER_BETTER_HINTS):
        return False
    if 'ratio' in h or '%' in h:
        return False
    return True


def _perf_verdict_fa_vs_fia(ratio_pct, lower_is_better=True):
    """以 FIA 为基准，判断 FA 相对 FIA 是优化 / 劣化 / 持平。

    ratio_pct = (fa - fia) / fia * 100
    - 越小越好：fa > fia → 劣化，fa < fia → 优化
    - 越大越好：fa > fia → 优化，fa < fia → 劣化
    """
    if ratio_pct is None:
        return 'N/A'
    if abs(ratio_pct) < 0.01:
        return '持平'
    if lower_is_better:
        return '劣化' if ratio_pct > 0 else '优化'
    return '优化' if ratio_pct > 0 else '劣化'


def _calc_fa_vs_fia(fa_val, fia_val, lower_is_better=True):
    """返回 diff, ratio_pct, verdict（均以 FIA 为基准看 FA）。"""
    if fa_val is None or fia_val is None or fia_val == 0:
        return '', '', 'N/A'
    diff = fa_val - fia_val
    ratio = diff / fia_val * 100.0
    return diff, ratio, _perf_verdict_fa_vs_fia(ratio, lower_is_better)


def _perf_list_by_order(perf_data, case_count):
    if len(perf_data) != case_count:
        return None
    ordered = []
    for i in range(case_count):
        key = f'__order_{i}__'
        if key not in perf_data:
            return None
        ordered.append(perf_data[key])
    return ordered


def _collect_perf_metric_names(fa_perf, fia_perf):
    names = set()
    for perf in (fa_perf, fia_perf):
        for entry in perf.values():
            names.update(entry.get('metrics', {}).keys())
            names.update(entry.get('metric_names', []))
    duration_names = [n for n in names if 'task duration' in _norm_header(n)]
    other_names = sorted(
        [n for n in names if n not in duration_names],
        key=lambda x: _norm_header(x))
    return duration_names + other_names


def _build_perf_metric_compare_fields(metric_names):
    fields = list(PERF_SUMMARY_FIELDS)
    seen = set(fields)
    for metric in metric_names:
        mkey = _metric_to_field_key(metric)
        if 'task duration' in _norm_header(metric):
            continue
        for suffix in (f'fa_{mkey}', f'fia_{mkey}', f'diff_{mkey}',
                       f'ratio_{mkey}', f'fa_vs_fia_{mkey}'):
            if suffix not in seen:
                fields.append(suffix)
                seen.add(suffix)
    return fields


def _round_metric_val(val, digits=6):
    if val is None or val == '':
        return ''
    return round(float(val), digits)


def _fill_metric_compare_columns(row, fa, fia, metric_names):
    """以 FIA 为基准，填充 FA 相对 FIA 的对比列（含优化/劣化判定）。"""
    for metric in metric_names:
        mkey = _metric_to_field_key(metric)
        lower_better = _metric_lower_is_better(metric)
        fa_val = fa.get('metrics', {}).get(metric) if fa else None
        fia_val = fia.get('metrics', {}).get(metric) if fia else None
        diff, ratio, verdict = _calc_fa_vs_fia(fa_val, fia_val, lower_better)
        if 'task duration' in _norm_header(metric):
            row['fa_avg_us'] = _round_metric_val(fa_val, 3)
            row['fia_avg_us'] = _round_metric_val(fia_val, 3)
            row['fa_rounds'] = fa['rounds'] if fa else ''
            row['fia_rounds'] = fia['rounds'] if fia else ''
            row['diff_us'] = _round_metric_val(diff, 3) if diff != '' else ''
            row['ratio_pct'] = _round_metric_val(ratio, 4) if ratio != '' else ''
            row['fa_vs_fia'] = verdict
            continue
        row[f'fa_{mkey}'] = _round_metric_val(fa_val)
        row[f'fia_{mkey}'] = _round_metric_val(fia_val)
        row[f'diff_{mkey}'] = _round_metric_val(diff) if diff != '' else ''
        row[f'ratio_{mkey}'] = _round_metric_val(ratio, 4) if ratio != '' else ''
        row[f'fa_vs_fia_{mkey}'] = verdict


def build_perf_compare_table(cases, fa_perf, fia_perf):
    """合并用例参数 + 全部性能参数的 FA/FIA 对比。"""
    metric_names = _collect_perf_metric_names(fa_perf, fia_perf)
    rows = []
    fa_order = _perf_list_by_order(fa_perf, len(cases))
    fia_order = _perf_list_by_order(fia_perf, len(cases))
    for idx, (case_name, case) in enumerate(cases.items()):
        row = case_to_summary_row(case_name, case)
        fa = _match_perf_for_case(case_name, fa_perf, order_idx=idx)
        fia = _match_perf_for_case(case_name, fia_perf, order_idx=idx)
        if fa is None and fa_order is not None:
            fa = fa_order[idx]
        if fia is None and fia_order is not None:
            fia = fia_order[idx]
        if fa and fia and fa.get('rounds') != fia.get('rounds'):
            print(f"[WARN] {case_name}: FA {fa['rounds']} 轮 vs FIA {fia['rounds']} 轮，轮次不一致")
        _fill_metric_compare_columns(row, fa, fia, metric_names)
        rows.append(row)
    return rows, metric_names


def _write_xlsx_sheet(ws, rows, fields):
    ws.append(fields)
    for row in rows:
        ws.append([row.get(f, '') for f in fields])
    for col_idx, field in enumerate(fields, start=1):
        max_len = len(field)
        for row in rows:
            max_len = max(max_len, len(_format_summary_cell(row.get(field, ''))))
        ws.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = min(max_len + 2, 52)


def _write_generic_xlsx(rows, fields, output_path, sheet_name='sheet1'):
    if openpyxl is None:
        raise RuntimeError("导出 xlsx 需要 openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name
    _write_xlsx_sheet(ws, rows, fields)
    wb.save(output_path)
    wb.close()


def _write_perf_compare_xlsx(rows, metric_names, output_path):
    if openpyxl is None:
        raise RuntimeError("导出 xlsx 需要 openpyxl")
    full_fields = _build_perf_metric_compare_fields(metric_names)
    wb = openpyxl.Workbook()
    ws_summary = wb.active
    ws_summary.title = 'duration_summary'
    _write_xlsx_sheet(ws_summary, rows, PERF_SUMMARY_FIELDS)
    ws_all = wb.create_sheet('all_metrics')
    _write_xlsx_sheet(ws_all, rows, full_fields)
    wb.save(output_path)
    wb.close()


def print_perf_compare_table(rows):
    if not rows:
        print("[WARN] 无性能对比数据")
        return
    show_fields = ['case_name', 'B', 'N1', 'S1', 'S2',
                   'fa_avg_us', 'fa_rounds', 'fia_avg_us', 'fia_rounds',
                   'diff_us', 'ratio_pct', 'fa_vs_fia']
    col_w = {f: len(f) for f in show_fields}
    for row in rows:
        for f in show_fields:
            col_w[f] = max(col_w[f], len(_format_summary_cell(row.get(f, ''))))
    sep = '  '.join('-' * col_w[f] for f in show_fields)
    print(f"\n{'=' * len(sep)}")
    print(f"  FA vs FIA 性能对比  ({len(rows)} cases)")
    print(f"  以 FIA 为基准，看 FA 是否优化/劣化（fa_vs_fia 列）")
    print(f"  diff_us = fa - fia；ratio_pct = (fa-fia)/fia*100%")
    print(f"  Task Duration: ratio_pct<0 → FA优化(更快)，ratio_pct>0 → FA劣化(更慢)")
    print(f"  全部性能参数及 fa_vs_fia 判定见输出 xlsx 的 all_metrics sheet")
    print(f"{'=' * len(sep)}")
    hdr = '  '.join(f.ljust(col_w[f]) for f in show_fields)
    print(hdr)
    print(sep)
    for row in rows:
        print('  '.join(_format_summary_cell(row.get(f, '')).ljust(col_w[f]) for f in show_fields))
    print(f"{'=' * len(sep)}\n")


def _default_perf_compare_path(case_xlsx):
    base = os.path.splitext(os.path.basename(case_xlsx))[0]
    return os.path.join(os.path.dirname(os.path.abspath(case_xlsx)), f"{base}_fa_fia_perf_compare.xlsx")


def export_perf_compare(case_xlsx, fa_perf_path, fia_perf_path, perf_out=None,
                        case_sheet=None, fa_sheet=None, fia_sheet=None,
                        perf_rounds=None, perf_duration_col=None, perf_case_col=None):
    cases = load_cases_from_xlsx(case_xlsx, case_sheet)
    if not cases:
        raise ValueError(f"未从 {case_xlsx} 解析到用例")
    num_cases = len(cases)
    fa_perf = load_perf_table(
        fa_perf_path, fa_sheet, duration_col=perf_duration_col, case_col=perf_case_col,
        rounds_per_case=perf_rounds, num_cases=num_cases)
    fia_perf = load_perf_table(
        fia_perf_path, fia_sheet, duration_col=perf_duration_col, case_col=perf_case_col,
        rounds_per_case=perf_rounds, num_cases=num_cases)
    if not fa_perf:
        raise ValueError(f"FA 性能表为空: {fa_perf_path}")
    if not fia_perf:
        raise ValueError(f"FIA 性能表为空: {fia_perf_path}")

    print(f"[INFO] 用例参数表: {num_cases} cases")
    print(f"[INFO] FA  性能表: {fa_perf_path}  (聚合后 {len(fa_perf)} cases)")
    _print_perf_load_detail(fa_perf, 'FA')
    print(f"[INFO] FIA 性能表: {fia_perf_path}  (聚合后 {len(fia_perf)} cases)")
    _print_perf_load_detail(fia_perf, 'FIA')
    rows, metric_names = build_perf_compare_table(cases, fa_perf, fia_perf)
    print_perf_compare_table(rows)
    print(f"[INFO] 全量性能参数对比列数: {len(metric_names)} 项 "
          f"(除 Task Duration 外另有 {max(0, len(metric_names) - 1)} 项)")

    out_path = perf_out or _default_perf_compare_path(case_xlsx)
    if not out_path.lower().endswith('.xlsx'):
        out_path += '.xlsx'
    _write_perf_compare_xlsx(rows, metric_names, out_path)
    print(f"[OK] 性能对比表已写入: {out_path}")
    print(f"     - sheet duration_summary: Task Duration 摘要（含 fa_vs_fia）")
    print(f"     - sheet all_metrics: 全部参数 FA/FIA/diff/ratio/fa_vs_fia")
    return rows


# ─────────────────── 用例参数归一化（FA / FIA 共用 seqlens 逻辑） ───────────────────

def _print_case_seqlens(operator, case_name, c):
    """打印 FA/FIA 共用的 seqlens，便于对比是否一致。"""
    print(f"  [{operator}] case={case_name}  layout_q={c.get('layout_q')}  "
          f"layout_kv={c.get('layout_kv')}  B={c.get('B')}")
    if c.get('seqused_q') is not None:
        print(f"  [{operator}] seqused_q={c['seqused_q']}")
    if c.get('cu_seqlens_q') is not None:
        print(f"  [{operator}] cu_seqlens_q={c['cu_seqlens_q']}")
    if c.get('seqused_kv') is not None:
        print(f"  [{operator}] seqused_kv={c['seqused_kv']}")
    if c.get('cu_seqlens_kv') is not None:
        print(f"  [{operator}] cu_seqlens_kv={c['cu_seqlens_kv']}")


def _print_qkv_layout_shapes(operator, c, q, k, v, q_rope=None, k_rope=None):
    """打印 layout 转换后的 Q/K/V shape。"""
    print(f"  [{operator}] Q shape={tuple(q.shape)}  K shape={tuple(k.shape)}  "
          f"V shape={tuple(v.shape)}  (layout_q={c.get('layout_q')}  layout_kv={c.get('layout_kv')})")
    if q_rope is not None:
        print(f"  [{operator}] Q_rope shape={tuple(q_rope.shape)}  "
              f"K_rope shape={tuple(k_rope.shape) if k_rope is not None else None}")


def _resolve_qkv_gen_params(c):
    """FA / FIA 共用的 QKV 生成尺寸参数（与 run_flash_attn 逻辑一致）。"""
    b = c.get('B', 1)
    n1 = c.get('N1')
    n2 = c.get('N2', n1)
    sq = c.get('S1')
    skv = c.get('S2', sq)
    d = c.get('D')
    d_v = c.get('DV', d)
    d_rope = c.get('DRope', c.get('rope_d', 0))
    layout_q = c.get('layout_q', c.get('input_layout', 'BNSD'))
    if isinstance(layout_q, str) and '_' in layout_q:
        layout_q = _parse_layout_base(layout_q)
    dtype = c.get('Dtype', 'bf16')
    pttype = torch.float16 if dtype == 'fp16' else torch.bfloat16

    sq_gen, skv_gen = sq, skv
    if layout_q == 'TND':
        cu_q = c['cu_seqlens_q']
        sq = max(cu_q[i + 1] - cu_q[i] for i in range(len(cu_q) - 1))
        sq_gen = cu_q[-1]
        layout_kv = c.get('layout_kv', layout_q)
        if _is_pa_layout(layout_kv):
            skv = max(c['seqused_kv'])
            skv_gen = sum(c['seqused_kv'])
        elif c.get('cu_seqlens_kv'):
            cu_kv = c['cu_seqlens_kv']
            skv = max(cu_kv[i + 1] - cu_kv[i] for i in range(len(cu_kv) - 1))
            skv_gen = cu_kv[-1]
        c['S1'], c['S2'] = sq, skv

    return {
        'b': b, 'n1': n1, 'n2': n2,
        'sq': sq, 'skv': skv, 'sq_gen': sq_gen, 'skv_gen': skv_gen,
        'd': d, 'd_v': d_v, 'd_rope': d_rope,
        'layout_q': layout_q, 'pttype': pttype,
    }


def _build_shared_qkv(c):
    """FA / FIA 共用：generate_qkv + trans_bnsd_to_layout，保证 Q/K/V 完全一致。"""
    _load_operator_deps()
    params = _resolve_qkv_gen_params(c)
    layout_q = c.get('layout_q', params['layout_q'])
    layout_kv = c.get('layout_kv', layout_q)

    q, k, v, q_rope, k_rope, qf, kf = generate_qkv(
        params['b'], params['n1'], params['n2'],
        params['sq_gen'], params['skv_gen'],
        params['d'], params['d_v'], params['d_rope'],
        params['layout_q'], params['pttype'],
        q_range=c.get('q_range'), k_range=c.get('k_range'), v_range=c.get('v_range'))

    act_q = c.get('seqused_q')
    act_kv = c.get('seqused_kv')
    q_l = trans_bnsd_to_layout(q, layout_q, act_seq_lens=act_q, **c)
    k_l = trans_bnsd_to_layout(k, layout_kv, act_seq_lens=act_kv, **c)
    v_l = trans_bnsd_to_layout(v, layout_kv, act_seq_lens=act_kv, **c)
    q_rope_l = trans_bnsd_to_layout(q_rope, layout_q, act_seq_lens=act_q, **c) if params['d_rope'] else None
    k_rope_l = trans_bnsd_to_layout(k_rope, layout_kv, act_seq_lens=act_kv, **c) if params['d_rope'] else None

    return {
        'params': params,
        'q_bnsd': q, 'k_bnsd': k, 'v_bnsd': v,
        'q_rope_bnsd': q_rope, 'k_rope_bnsd': k_rope,
        'q': q_l, 'k': k_l, 'v': v_l,
        'q_rope': q_rope_l, 'k_rope': k_rope_l,
        'qf': qf, 'kf': kf,
    }


def _normalize_fa_case(raw):
    """FA / FIA 共用：按 Q/KV layout 分别解析 actual_seq_lengths。

    - Q layout = TND：actual_seq_lengths_q → cu_seqlens_q
    - Q layout ≠ TND：actual_seq_lengths_q → seqused_q
    - KV layout = TND：actual_seq_lengths_kv → cu_seqlens_kv
    - KV layout = PA_*：actual_seq_lengths_kv → seqused_kv（无 cu_seqlens_kv）
    - KV layout 其他：actual_seq_lengths_kv → seqused_kv
    """
    c = dict(raw)
    layout_q = _parse_layout_base(c.get('layout_q', c.get('input_layout', 'BNSD')))
    layout_kv = c.get('layout_kv', layout_q)
    if not _is_pa_layout(layout_kv):
        layout_kv = _parse_layout_base(layout_kv)

    c['layout_q'] = layout_q
    c['layout_kv'] = layout_kv
    c['input_layout'] = layout_q
    c.setdefault('layout_out', c.get('layout_out', layout_q))
    c.setdefault('N1', c.get('N1', 1))
    c.setdefault('N2', c.get('N2', c['N1']))
    c.setdefault('S2', c.get('S2', c.get('S1')))
    c.setdefault('DV', c.get('DV', c.get('D')))
    c.setdefault('DRope', c.get('rope_d', 0))
    c.setdefault('q_start_idx', 0)
    c.setdefault('win_left', 2147483647)
    c.setdefault('win_right', 2147483647)
    c.setdefault('prefix', [])
    c.setdefault('q_range', None)
    c.setdefault('k_range', None)
    c.setdefault('v_range', None)
    if isinstance(c.get('return_softmax_lse'), bool):
        c['return_softmax_lse'] = 1 if c['return_softmax_lse'] else 0

    b = c.get('B', 1)
    s1, s2 = c.get('S1', 128), c.get('S2', c.get('S1', 128))

    act_q = c.pop('actual_seq_lengths_q', None)
    act_kv = c.pop('actual_seq_lengths_kv', None)

    # ── Q 侧序列长度 ──
    if layout_q == 'TND':
        if act_q is not None and 'cu_seqlens_q' not in c:
            c['cu_seqlens_q'] = _fa_actual_as_cu_seqlens(act_q, b, s1)
        if 'cu_seqlens_q' in c:
            c['seqused_q'] = [c['cu_seqlens_q'][i + 1] - c['cu_seqlens_q'][i]
                              for i in range(len(c['cu_seqlens_q']) - 1)]
            b = len(c['seqused_q'])
        c['B'] = 1
    else:
        if act_q is not None and 'seqused_q' not in c:
            c['seqused_q'] = _fa_actual_as_seqused(act_q, b, s1)
        c.pop('cu_seqlens_q', None)

    batch_cnt = len(c['seqused_q']) if c.get('seqused_q') else b

    # ── KV 侧序列长度（与 Q layout 独立） ──
    if _is_pa_layout(layout_kv):
        c.pop('cu_seqlens_kv', None)
        if act_kv is not None and 'seqused_kv' not in c:
            c['seqused_kv'] = _fa_actual_as_seqused(act_kv, batch_cnt, s2)
        elif 'seqused_kv' not in c:
            c['seqused_kv'] = [s2] * batch_cnt
    elif layout_kv == 'TND':
        if act_kv is not None and 'cu_seqlens_kv' not in c:
            c['cu_seqlens_kv'] = _fa_actual_as_cu_seqlens(act_kv, batch_cnt, s2)
        if 'cu_seqlens_kv' in c:
            c['seqused_kv'] = [c['cu_seqlens_kv'][i + 1] - c['cu_seqlens_kv'][i]
                               for i in range(len(c['cu_seqlens_kv']) - 1)]
        elif act_kv is not None and 'seqused_kv' not in c:
            c['seqused_kv'] = _fa_actual_as_seqused(act_kv, batch_cnt, s2)
    else:
        c.pop('cu_seqlens_kv', None)
        if act_kv is not None and 'seqused_kv' not in c:
            c['seqused_kv'] = _fa_actual_as_seqused(act_kv, batch_cnt, s2)

    for key in ('cu_seqlens_q', 'cu_seqlens_kv', 'seqused_q', 'seqused_kv'):
        if c.get(key) == [None] or c.get(key) is None:
            c.pop(key, None)

    # TND + TND（非 PA）时 KV 可与 Q 对齐
    if layout_q == 'TND' and layout_kv == 'TND':
        if 'cu_seqlens_kv' not in c and 'cu_seqlens_q' in c:
            c.setdefault('cu_seqlens_kv', list(c['cu_seqlens_q']))
            c.setdefault('seqused_kv', [c['cu_seqlens_kv'][i + 1] - c['cu_seqlens_kv'][i]
                                        for i in range(len(c['cu_seqlens_kv']) - 1)])

    # ── PA block_table ──
    if _is_pa_layout(layout_kv):
        c['cu_seqlens_kv'] = None
        c.setdefault('block_size', c.get('block_size', 16))
        block_size = c['block_size']
        block_table_shape = c.get('block_table_shape', [])
        seqused_kv = c.get('seqused_kv', [s2] * batch_cnt)
        c['seqused_kv'] = seqused_kv
        if c.get('block_table') is not None:
            c['block_table'] = torch.tensor(c['block_table'], dtype=torch.int32)
        else:
            if block_table_shape:
                b_val = block_table_shape[0]
                max_block_num = block_table_shape[1]
            elif layout_q == 'TND' and c.get('cu_seqlens_q'):
                b_val = len(c['cu_seqlens_q']) - 1
            else:
                b_val = batch_cnt
            max_block_num = (block_table_shape[1] if block_table_shape
                             else (max(seqused_kv) + block_size - 1) // block_size)
            block_table = torch.full((b_val, max_block_num), -1, dtype=torch.int32)
            block_idx = 0
            for i in range(b_val):
                b_seq = seqused_kv[i] if i < len(seqused_kv) else seqused_kv[0]
                for j in range((b_seq + block_size - 1) // block_size):
                    block_table[i][j] = block_idx
                    block_idx += 1
            c['block_table'] = block_table
        c['actual_b'] = c['block_table'].shape[0]

    # 无 actual_seq_lengths 时 FA/FIA 统一默认满长 seqused
    if layout_q != 'TND' and not c.get('seqused_q'):
        c['seqused_q'] = [s1] * b
    batch_final = len(c['seqused_q']) if c.get('seqused_q') else b
    if not c.get('seqused_kv'):
        c['seqused_kv'] = [s2] * batch_final

    return c


# ─────────────────── flash_attn 多轮执行 ───────────────────

def run_flash_attn(case_name, case, rounds, device_id=0, check_cpu=False, warmup=1):
    _load_operator_deps()
    kwargs = _normalize_fa_case(case)
    _print_case_seqlens('flash_attn', case_name, kwargs)
    qkv = _build_shared_qkv(kwargs)
    params = qkv['params']
    q, k, v = qkv['q_bnsd'], qkv['k_bnsd'], qkv['v_bnsd']
    q_rope, k_rope = qkv['q_rope_bnsd'], qkv['k_rope_bnsd']
    qf, kf = qkv['qf'], qkv['kf']
    _print_qkv_layout_shapes('flash_attn', kwargs, qkv['q'], qkv['k'], qkv['v'],
                             qkv['q_rope'], qkv['k_rope'])
    win_l, win_r = _aligned_window_tokens(kwargs)
    print(f"  [flash_attn] API cu_seqlens_q={kwargs.get('cu_seqlens_q')}  "
          f"seqused_q={kwargs.get('seqused_q')}")
    print(f"  [flash_attn] API cu_seqlens_kv={kwargs.get('cu_seqlens_kv')}  "
          f"seqused_kv={kwargs.get('seqused_kv')}")
    print(f"  [flash_attn] API mask_mode={kwargs.get('mask_mode')}  "
          f"win_left={win_l}  win_right={win_r}  N1={kwargs.get('N1')}  N2={kwargs.get('N2')}")

    mask = generate_npu_mask(params['b'], params['sq'], params['skv'],
                             kwargs.get('mask_mode'),
                             kwargs.get('win_left'), kwargs.get('win_right'))

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
    """FIA 参数归一化：seqlens / layout / PA 与 FA 共用 _normalize_fa_case，仅补充 FIA 专有字段。"""
    _load_operator_deps()
    c = _normalize_fa_case(raw)
    c['input_layout'] = raw.get('input_layout', c.get('layout_q', 'BNSD'))
    kv_layout = c.get('layout_kv', c.get('layout_q', 'BNSD'))
    if _is_pa_layout(kv_layout) or c.get('kv_storage_mode') not in (None, 'continue'):
        pa_rev = {'PA_BBND': 'pa_bbh', 'PA_BNBD': 'pa_bnbd', 'PA_NZ': 'pa_nz'}
        c['kv_storage_mode'] = pa_rev.get(kv_layout, c.get('kv_storage_mode', 'pa_bnbd'))
    else:
        c.setdefault('kv_storage_mode', 'continue')
    c.setdefault('inner_precise', c.get('inner_precise', 0))
    return c


def _aligned_window_tokens(c):
    """与 flash_attn npu_impl 对齐：mask_mode 0/3 时 win 置 -1。"""
    win_left = c.get('win_left', 2147483647)
    win_right = c.get('win_right', 2147483647)
    if c.get('mask_mode', 0) in (0, 3):
        win_left = -1
        win_right = -1
    return win_left, win_right


def _print_fa_fia_api_diff(case_name, fa_c, fia_inputs):
    """打印 FA / FIA 算子 API 层关键参数差异。"""
    win_l, win_r = _aligned_window_tokens(fa_c)
    print(f"  [API对比] {case_name}")
    print(f"    FA  cu_seqlens_q={fa_c.get('cu_seqlens_q')}  seqused_q={fa_c.get('seqused_q')}")
    print(f"    FA  cu_seqlens_kv={fa_c.get('cu_seqlens_kv')}  seqused_kv={fa_c.get('seqused_kv')}")
    print(f"    FA  mask_mode={fa_c.get('mask_mode')}  win_left={win_l}  win_right={win_r}")
    print(f"    FIA actual_seq_lengths={fia_inputs.get('actual_seq_lengths')}  "
          f"actual_seq_lengths_kv={fia_inputs.get('actual_seq_lengths_kv')}")
    print(f"    FIA sparse_mode={fia_inputs.get('sparse_mode')}  "
          f"pre_tokens={fia_inputs.get('pre_tokens')}  next_tokens={fia_inputs.get('next_tokens')}")
    print(f"    FIA num_heads={fia_inputs.get('num_heads')}  "
          f"num_key_value_heads={fia_inputs.get('num_key_value_heads')}  "
          f"input_layout={fia_inputs.get('input_layout')}")


def _fia_update_act_seq_lens_tnd(layout, act_seq_lens):
    if layout not in ('TND', 'NTD') or not act_seq_lens:
        return act_seq_lens
    result, total = [], 0
    for length in act_seq_lens:
        total += length
        result.append(total)
    return result


def _fia_to_npu(tensor, device):
    return tensor.to(device).contiguous() if tensor is not None else None


def _fia_prepare_inputs(case, device_id, qkv=None):
    """FIA 数据准备：Q/K/V 与 FA 共用 generate_qkv + trans_bnsd_to_layout。"""
    c = _fia_normalize_case(case)
    device = torch.device(f"npu:{device_id}")
    if qkv is None:
        qkv = _build_shared_qkv(c)
    params = qkv['params']
    n1, n2 = params['n1'], params['n2']
    d_rope = params['d_rope']
    input_layout = c.get('input_layout', c.get('layout_q', 'BNSD'))
    query_layout = _fia_query_layout(input_layout)
    layout_kv = c.get('layout_kv', c.get('layout_q'))

    act_q = list(c.get('seqused_q') or [params['sq']] * c.get('B', 1))
    act_kv = list(c.get('seqused_kv') or [params['skv']] * len(act_q))
    act_q_fia = _fia_update_act_seq_lens_tnd(query_layout, act_q)
    if _is_pa_layout(layout_kv):
        act_kv_fia = list(act_kv)
        block_table = c.get('block_table')
    else:
        block_table = None
        act_kv_fia = _fia_update_act_seq_lens_tnd(query_layout, act_kv)

    win_left, win_right = _aligned_window_tokens(c)
    atten_mask = generate_npu_mask(params['b'], params['sq'], params['skv'],
                                   c.get('mask_mode'), win_left, win_right)

    return {
        'device': device,
        'query': _fia_to_npu(qkv['q'], device),
        'key': _fia_to_npu(qkv['k'], device),
        'value': _fia_to_npu(qkv['v'], device),
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
        'query_rope': _fia_to_npu(qkv['q_rope'], device),
        'key_rope': _fia_to_npu(qkv['k_rope'], device),
        'num_heads': n1,
        'scale': c['scale'],
        'pre_tokens': win_left,
        'next_tokens': win_right,
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
    _load_operator_deps()
    if torch_npu is None:
        raise RuntimeError("FIA 需要 torch_npu")

    torch.npu.set_device(device_id)
    norm = _fia_normalize_case(case)
    _print_case_seqlens('fia', case_name, norm)
    qkv = _build_shared_qkv(norm)
    _print_qkv_layout_shapes('fia', norm, qkv['q'], qkv['k'], qkv['v'],
                             qkv['q_rope'], qkv['k_rope'])
    fia_inputs = _fia_prepare_inputs(case, device_id, qkv=qkv)
    _print_fa_fia_api_diff(case_name, norm, fia_inputs)

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
    print(f"  layout_q={case.get('layout_q', case.get('input_layout'))}  "
          f"layout_kv={case.get('layout_kv', case.get('layout_q'))}  "
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
    parser = argparse.ArgumentParser(description="精简版 xlsx/csv 驱动 flash_attn / FIA 测试")
    parser.add_argument("--xlsx", "-i", default=None, help="用例参数表路径（xlsx 或 csv）")
    parser.add_argument("--case-xlsx", default=None,
                        help="用例参数表（--compare-perf 时可用，xlsx/csv，默认等同 --xlsx）")
    parser.add_argument("--operator", "-o", default="flash_attn",
                        choices=["flash_attn", "fia"], help="算子: flash_attn 或 fia")
    parser.add_argument("--rounds", "-n", type=int, default=1, help="算子运行轮次")
    parser.add_argument("--row", type=int, default=None, help="只运行 xlsx 第 N 行（含表头，第1行为表头）")
    parser.add_argument("--sheet", "-s", default=None, help="xlsx sheet 名称")
    parser.add_argument("--case-id", default=None, help="只运行指定用例名（逗号分隔）")
    parser.add_argument("--device-id", type=int, default=0, help="NPU device id")
    parser.add_argument("--check-cpu", action="store_true", help="flash_attn 模式下对比 CPU golden")
    parser.add_argument("--warmup", type=int, default=1, help="flash_attn 预热轮次（不计入统计）")
    parser.add_argument("--export-table", action="store_true",
                        help="仅导出处用例参数摘要表（B/N1/N2/G/S1/S2/mask_mode/D/layout/dtype），不运行算子")
    parser.add_argument("--table-out", default=None,
                        help="摘要表输出路径（默认 <输入xlsx名>_summary.xlsx；.csv 则导出 csv）")
    parser.add_argument("--export-py", action="store_true",
                        help="导出 test_cases 风格 TestCases .py（类似 functional_redline_infer.py），不运行算子")
    parser.add_argument("--py-out", default=None,
                        help="TestCases .py 输出路径（默认 test_cases/<xlsx名>.py）")
    parser.add_argument("--compare-perf", action="store_true",
                        help="仅读取 FA/FIA 性能表与用例参数表做对比（xlsx/csv），不调用算子、不需要 NPU")
    parser.add_argument("--fa-perf", default=None,
                        help="flash_attn 性能表（xlsx/csv，含 Task Duration(us) 列）")
    parser.add_argument("--fia-perf", default=None,
                        help="FIA 性能表（xlsx/csv，含 Task Duration(us) 列）")
    parser.add_argument("--perf-out", default=None,
                        help="性能对比输出 xlsx（默认 <case-xlsx名>_fa_fia_perf_compare.xlsx）")
    parser.add_argument("--fa-perf-sheet", default=None, help="FA 性能表 sheet 名")
    parser.add_argument("--fia-perf-sheet", default=None, help="FIA 性能表 sheet 名")
    parser.add_argument("--perf-rounds", type=int, default=None,
                        help="每 case 性能采样轮次（profiler 导出无 case 列时必填；"
                             "也可自动推断=总行数/用例数）")
    parser.add_argument("--perf-duration-col", default=None,
                        help="性能表 duration 列名（默认自动识别 Task Duration(us)）")
    parser.add_argument("--perf-case-col", default=None,
                        help="性能表 case 分组列名（profiler 导出通常不需要，默认按顺序对齐）")
    args = parser.parse_args()

    if args.compare_perf:
        case_xlsx = args.case_xlsx or args.xlsx
        if not case_xlsx:
            print("[ERROR] --compare-perf 需要 --case-xlsx 或 --xlsx 指定用例参数表")
            sys.exit(1)
        if not args.fa_perf or not args.fia_perf:
            print("[ERROR] --compare-perf 需要同时指定 --fa-perf 和 --fia-perf")
            sys.exit(1)
        for p, label in ((case_xlsx, 'case-xlsx'), (args.fa_perf, 'fa-perf'), (args.fia_perf, 'fia-perf')):
            if not os.path.exists(p):
                print(f"[ERROR] {label} 文件不存在: {p}")
                sys.exit(1)
        print("[INFO] --compare-perf 模式：仅读取表格（xlsx/csv），不调用算子")
        try:
            export_perf_compare(
                case_xlsx, args.fa_perf, args.fia_perf, args.perf_out,
                case_sheet=args.sheet, fa_sheet=args.fa_perf_sheet,
                fia_sheet=args.fia_perf_sheet, perf_rounds=args.perf_rounds,
                perf_duration_col=args.perf_duration_col, perf_case_col=args.perf_case_col)
        except Exception as e:
            import traceback
            print(f"[ERROR] 性能对比失败: {e}")
            traceback.print_exc()
            sys.exit(1)
        sys.exit(0)

    if not args.xlsx:
        print("[ERROR] 运行算子或 --export-table / --export-py 需要 --xlsx 用例参数表")
        sys.exit(1)

    if not os.path.exists(args.xlsx):
        print(f"[ERROR] 文件不存在: {args.xlsx}")
        sys.exit(1)

    cases = load_cases_from_xlsx(args.xlsx, args.sheet, args.row)
    if not cases:
        print("[ERROR] 未从 xlsx 解析到有效用例")
        sys.exit(1)

    if args.export_table:
        print(f"[INFO] 从 {args.xlsx} 加载 {len(cases)} 个用例，导出参数摘要表")
        export_case_summary(cases, args.table_out, input_xlsx=args.xlsx)
        sys.exit(0)

    if args.export_py:
        print(f"[INFO] 从 {args.xlsx} 加载 {len(cases)} 个用例，导出 TestCases .py")
        try:
            export_testcase_py(cases, args.py_out, input_xlsx=args.xlsx)
        except Exception as e:
            import traceback
            print(f"[ERROR] 导出 TestCases .py 失败: {e}")
            traceback.print_exc()
            sys.exit(1)
        sys.exit(0)

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
