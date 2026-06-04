"""Performance runner: build sweep script using InputSpec, run msprof."""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def build_perf_script(cases: List[Dict], runs: int, cold_thr: int,
                      device_id: int = 0, case_names: List[str] = None) -> str:
    """Build a single sweep script that uses flash_attn_inputs.generate()."""
    hot = [(i, c) for i, c in enumerate(cases) if c.get("S1", 9999) > cold_thr]
    cold = [(i, c) for i, c in enumerate(cases) if c.get("S1", 9999) <= cold_thr]
    names = case_names or [f"case_{i}" for i in range(len(cases))]

    case_dicts = _serialize_cases(cases)
    hot_names = repr([names[i] for i, _ in hot])
    cold_names = repr([names[i] for i, _ in cold])

    lines = [
        "import sys, traceback, os",
        f"_pytests_dir = {repr(str(Path(__file__).resolve().parent.parent))}",
        "sys.path.insert(0, _pytests_dir)",
        "import torch, torch_npu",
        f"torch.npu.set_device({device_id})",
        "from core.data import flash_attn_inputs",
        "from npu_ops_transformer.ops import npu_flash_attn, npu_flash_attn_metadata",
        "device = torch.device('npu:0')",
        f"_hot_cases = {case_dicts['hot']}",
        f"_cold_cases = {case_dicts['cold']}",
        f"_hot_names = {hot_names}",
        f"_cold_names = {cold_names}",
        f"print(f'PERF total={len(cases)} hot={len(hot)} cold={len(cold)} runs={runs}')",
        "",
    ]

    for idx, _ in enumerate(hot):
        lines.extend(_gen_hot_v2(idx, runs, len(hot)))
    for idx, _ in enumerate(cold):
        lines.extend(_gen_cold_v2(idx, runs, len(cold)))

    lines.append("print('PERF_DONE')")
    return "\n".join(lines)


def _serialize_cases(cases):
    """Serialize case params to Python literals for the generated script."""
    hot_items = []
    cold_items = []
    thr = 16  # match default cold_thr
    for c in cases:
        s = {}
        for k, v in c.items():
            if hasattr(v, 'shape'):
                s[k] = v.tolist() if v.numel() < 100000 else f"<tensor {tuple(v.shape)}>"
            elif isinstance(v, (list, tuple)):
                s[k] = list(v)
            else:
                s[k] = v
        if c.get("S1", 9999) > thr:
            hot_items.append(s)
        else:
            cold_items.append(s)
    return {"hot": repr(hot_items), "cold": repr(cold_items)}


def _gen_hot_v2(idx, runs, total):
    return [
        f"c = _hot_cases[{idx}]",
        f"name = _hot_names[{idx}]",
        f"print(f'  [hot] {idx+1}/{total}  {{name}}')",
        f"print(f'[CASE_START] hot{idx}')",
        "try:",
        "    layout_q = c.get('layout_q', c.get('input_layout', 'BNSD'))",
        "    layout_kv = c.get('layout_kv', layout_q)",
        f"    inputs = flash_attn_inputs.generate(c, device=device, layout=layout_q, layout_kv=layout_kv)",
        "    bt = c.get('block_table')",
        "    if bt is not None and isinstance(bt, list) and str(layout_kv).startswith('PA_'):",
        "        inputs['block_table'] = torch.tensor(bt, dtype=torch.int32, device=device)",
        "    q, k, v = inputs['q'], inputs['k'], inputs['v']",
        "    bt_dev = inputs.get('block_table')",
        "    _sq = c.get('seqused_q')",
        "    _skv = c.get('seqused_kv')",
        "    meta = npu_flash_attn_metadata(",
        "        cu_seqlens_q=c.get('cu_seqlens_q'), cu_seqlens_kv=c.get('cu_seqlens_kv'),",
        "        seqused_q=_sq, seqused_kv=_skv,",
        "        num_heads_q=c['N1'], num_heads_kv=c.get('N2',c['N1']), head_dim=c['D'],",
        "        batch_size=c.get('B',1), max_seqlen_q=c.get('S1',1), max_seqlen_kv=c.get('S2',c.get('S1',1)),",
        "        mask_mode=c.get('mask_mode',0), win_left=-1, win_right=-1,",
        "        layout_q=layout_q, layout_kv=layout_kv, layout_out=c.get('layout_out',layout_q),",
        "    )",
        "    if c.get('mask_mode', 0) == 3:",
        "        _m = torch.triu(torch.ones(2048, 2048), diagonal=1).to(dtype=torch.int8, device=device)",
        "    else:",
        "        _m = None",
        "    torch.npu.synchronize()",
        f"    for _ in range({runs}):",
        "        torch.npu.synchronize()",
        "        npu_flash_attn(q, k, v, block_table=bt_dev, sinks=None, attn_mask=_m,",
        "            metadata=meta, cu_seqlens_q=c.get('cu_seqlens_q'), cu_seqlens_kv=c.get('cu_seqlens_kv'),",
        "            seqused_q=_sq, seqused_kv=_skv,",
        "            softmax_scale=1/(c['D']**0.5), mask_mode=c.get('mask_mode',0),",
        "            win_left=-1, win_right=-1, max_seqlen_q=-1, max_seqlen_kv=-1,",
        "            layout_q=layout_q, layout_kv=layout_kv, layout_out=c.get('layout_out',layout_q),",
        "            return_softmax_lse=0)",
        "    torch.npu.synchronize()",
        f"    print(f'[CASE_END] hot{idx}  OK')",
        "except Exception as e:",
        f"    print(f'[CASE_END] hot{idx}  CRASH  {{e}}')",
        "    torch.npu.synchronize()",
        "    torch.npu.empty_cache()",
        f"print(f'  [hot] {idx+1}/{total}')",
        "",
    ]


def _gen_cold_v2(idx, runs, total):
    return [
        f"c = _cold_cases[{idx}]",
        f"name = _cold_names[{idx}]",
        f"print(f'  [cold] {idx+1}/{total}  {{name}}')",
        f"print(f'[CASE_START] cold{idx}')",
        "try:",
        "    layout_q = c.get('layout_q', c.get('input_layout', 'BNSD'))",
        "    layout_kv = c.get('layout_kv', layout_q)",
        f"    for _ in range({runs}):",
        f"        inputs = flash_attn_inputs.generate(c, device=device, layout=layout_q, layout_kv=layout_kv)",
        "        bt = c.get('block_table')",
        "        if bt is not None and isinstance(bt, list) and str(layout_kv).startswith('PA_'):",
        "            inputs['block_table'] = torch.tensor(bt, dtype=torch.int32, device=device)",
        "        q, k, v = inputs['q'], inputs['k'], inputs['v']",
        "        bt_dev = inputs.get('block_table')",
        "        _sq = c.get('seqused_q')",
        "        _skv = c.get('seqused_kv')",
        "        meta = npu_flash_attn_metadata(",
        "            cu_seqlens_q=c.get('cu_seqlens_q'), cu_seqlens_kv=c.get('cu_seqlens_kv'),",
        "            seqused_q=_sq, seqused_kv=_skv,",
        "            num_heads_q=c['N1'], num_heads_kv=c.get('N2',c['N1']), head_dim=c['D'],",
        "            batch_size=c.get('B',1), max_seqlen_q=c.get('S1',1), max_seqlen_kv=c.get('S2',c.get('S1',1)),",
        "            mask_mode=c.get('mask_mode',0), win_left=-1, win_right=-1,",
        "            layout_q=layout_q, layout_kv=layout_kv, layout_out=c.get('layout_out',layout_q),",
        "        )",
        "        if c.get('mask_mode', 0) == 3:",
        "            _m = torch.triu(torch.ones(2048, 2048), diagonal=1).to(dtype=torch.int8, device=device)",
        "        else:",
        "            _m = None",
        "        _f1 = torch.randn(32, 32768, 1, 128, dtype=q.dtype, device=device)",
        "        _f2 = torch.randn(32, 32768, 1, 128, dtype=q.dtype, device=device)",
        "        torch.npu.synchronize()",
        "        npu_flash_attn(q, k, v, block_table=bt_dev, sinks=None, attn_mask=_m,",
        "            metadata=meta, cu_seqlens_q=c.get('cu_seqlens_q'), cu_seqlens_kv=c.get('cu_seqlens_kv'),",
        "            seqused_q=_sq, seqused_kv=_skv,",
        "            softmax_scale=1/(c['D']**0.5), mask_mode=c.get('mask_mode',0),",
        "            win_left=-1, win_right=-1, max_seqlen_q=-1, max_seqlen_kv=-1,",
        "            layout_q=layout_q, layout_kv=layout_kv, layout_out=c.get('layout_out',layout_q),",
        "            return_softmax_lse=0)",
        "    torch.npu.synchronize()",
        "    torch.npu.empty_cache()",
        f"    print(f'[CASE_END] cold{idx}  OK')",
        "except Exception as e:",
        f"    print(f'[CASE_END] cold{idx}  CRASH  {{e}}')",
        "    torch.npu.synchronize()",
        "    torch.npu.empty_cache()",
        f"print(f'  [cold] {idx+1}/{total}')",
        "",
    ]


def run_msprof(script: str, output_dir: Path) -> Optional[Path]:
    """Run msprof, return path to op_summary.csv or None on failure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / "_perf_sweep.py"
    script_path.write_text(script)

    msprof_dir = output_dir / "msprof_output"
    if msprof_dir.exists():
        shutil.rmtree(msprof_dir)

    cmd = ["msprof", f"--output={msprof_dir}", "--aic-mode=task-based",
           sys.executable, str(script_path)]
    print(f"[msprof] {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(f"[msprof stderr]\n{result.stderr.strip()}")
    if result.returncode != 0:
        print(f"[msprof] non-zero exit code: {result.returncode}")

    prof_dirs = sorted(msprof_dir.glob("PROF_*"), reverse=True)
    if not prof_dirs:
        print("[msprof] no PROF_* directory found")
        return None

    csv_files = sorted(prof_dirs[0].glob("mindstudio_profiler_output/op_summary_*.csv"))
    if not csv_files:
        print("[msprof] no op_summary CSV found")
        return None

    dst = output_dir / "op_summary.csv"
    shutil.copy(csv_files[0], dst)
    print(f"[msprof] op_summary -> {dst}")
    return dst
