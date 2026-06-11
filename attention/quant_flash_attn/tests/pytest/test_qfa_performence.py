#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一站式参数扫描 / 重复性测试 与绘图工具：
1. 参数扫描模式：根据基础配置生成扫描用例 → msprof 执行 → 解析 CSV → 绘制参数‑耗时曲线
   - 支持1~2个参数扫描（单参数2D折线图，双参数热力图+分组折线图）
   - 支持预设值、显式值、范围步长、随机取值等泛化方式
   - 双参数自动生成笛卡尔积测试集
2. 重复测试模式：仅运行基础用例 N 次 → 绘制耗时曲线 + 统计平均值
"""

import argparse
import subprocess
import time
import sys
import random
import itertools
from pathlib import Path
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ==================== 基础配置 ====================
BASE_CONFIG = {
    'b': [1],
    'n1': [1],
    'n2': [1],
    'g': [1],
    's1': [128],
    's2': [512],
    'qk_d': [128],
    'v_d': [128],
    'input_layout': ['BNSD'],
    'kv_storage_mode': ['continue'],
    'q_dtype': ['fp4_e2m1'],
    'kv_dtype': ['fp4_e2m1'],
    'out_dtype': ['bfloat16'],
    'q_range': [(-1.0, 1.0)],
    'k_range': [(-1.0, 1.0)],
    'v_range': [(-1.0, 1.0)],
    'q_quant_mode': [3],
    'k_quant_mode': [3],
    'v_quant_mode': [3],
    'act_seq_lens_q': [[128]],
    "act_seq_lens_kv": [[512]],
}
BIND_GROUPS = [
    {'len_param': 'b', 'max_param': 's1', 'list_param': 'act_seq_lens_q'},
    {'len_param': 'b', 'max_param': 's2', 'list_param': 'act_seq_lens_kv'},
]

PRESETS = {
    'seq_len': [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
    'batch': [1, 2, 4, 8, 16, 32],
}
# ==================================================


def parse_value(v_str):
    v_str = v_str.strip()
    if v_str.lower().endswith('k'):
        num = float(v_str[:-1])
        return int(num * 1024)
    try:
        return int(v_str)
    except ValueError:
        try:
            return float(v_str)
        except ValueError:
            return v_str


def parse_sweep_spec(spec_str):
    parts = spec_str.split(':', 1)
    if len(parts) != 2:
        raise ValueError(f"sweep 规格格式错误: '{spec_str}', 应为 param:method=spec")
    param_name = parts[0].strip()
    method_spec = parts[1].strip()
    method_parts = method_spec.split('=', 1)
    if len(method_parts) != 2:
        raise ValueError(f"缺少规格定义: '{method_spec}', 应为 method=spec")
    method = method_parts[0].strip()
    spec_val = method_parts[1].strip()

    if method == 'values':
        values = [parse_value(v) for v in spec_val.split(',')]
        return (param_name, 'values', values)
    elif method == 'range':
        range_parts = spec_val.split(',')
        if len(range_parts) != 3:
            raise ValueError(f"range 格式应为 start,step,count: '{spec_val}'")
        start = parse_value(range_parts[0])
        step = parse_value(range_parts[1])
        count = int(range_parts[2])
        return (param_name, 'range', (start, step, count))
    elif method == 'random':
        random_parts = spec_val.split(',')
        if len(random_parts) != 3:
            raise ValueError(f"random 格式应为 count,min,max: '{spec_val}'")
        count = int(random_parts[0])
        min_val = parse_value(random_parts[1])
        max_val = parse_value(random_parts[2])
        return (param_name, 'random', (count, min_val, max_val))
    elif method == 'preset':
        if spec_val not in PRESETS:
            raise ValueError(f"未知预设: '{spec_val}', 可用预设: {list(PRESETS.keys())}")
        return (param_name, 'preset', PRESETS[spec_val])
    else:
        raise ValueError(f"未知方法: '{method}', 可用方法: values, range, random, preset")


def resolve_sweep_values(sweep_specs, rng=None):
    if rng is None:
        rng = random.Random()
    resolved = []
    for param_name, method, spec_data in sweep_specs:
        if method in ('values', 'preset'):
            resolved.append((param_name, list(spec_data)))
        elif method == 'range':
            start, step, count = spec_data
            values = [start + i * step for i in range(count)]
            values = [int(v) if isinstance(v, float) and v.is_integer() else v for v in values]
            resolved.append((param_name, values))
        elif method == 'random':
            count, min_val, max_val = spec_data
            values = sorted(set([rng.randint(int(min_val), int(max_val)) for _ in range(count)]))
            if not values:
                values = [rng.randint(int(min_val), int(max_val))]
            resolved.append((param_name, values))
    return resolved


def apply_bindings(config: dict, random_fill: bool = False, rng=None) -> dict:
    if rng is None:
        rng = random
    for bg in BIND_GROUPS:
        len_key = bg['len_param']
        max_key = bg['max_param']
        list_key = bg['list_param']
        if len_key in config and max_key in config:
            length = config[len_key][0]
            max_val = config[max_key][0]
            if random_fill:
                new_list = [rng.randint(1, max_val) for _ in range(length)]
            else:
                new_list = [max_val] * length
            config[list_key] = [new_list]
    return config


def generate_sweep_cases(base_config: dict, resolved_sweeps: list,
                         random_fill: bool = False, rng=None) -> dict:
    sweep_params = {p: vals for p, vals in resolved_sweeps}
    param_names = list(sweep_params.keys())
    param_values = [sweep_params[p] for p in param_names]
    combinations = list(itertools.product(*param_values))

    cases = {}
    for combo in combinations:
        name_parts = [f"{p}_{v}" for p, v in zip(param_names, combo)]
        case_name = "sweep_" + "_".join(name_parts)
        new_config = deepcopy(base_config)
        for param_name, value in zip(param_names, combo):
            new_config[param_name] = [value]
        new_config = apply_bindings(new_config, random_fill=random_fill, rng=rng)
        cases[case_name] = new_config

    return cases


def write_cases_module(cases: dict, sweep_params: list, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    init_file = output_dir / "__init__.py"
    if not init_file.exists():
        init_file.touch()

    param_suffix = "_".join([p for p, _ in sweep_params])
    module_name = f"sweep_{param_suffix}"
    filepath = output_dir / (module_name + ".py")

    sort_key = lambda item: tuple(item[1][p][0] for p, _ in sweep_params)
    sorted_cases = sorted(cases.items(), key=sort_key)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# 自动生成的性能用例\n")
        f.write("TestCasesPerformance = {\n")
        for name, config in sorted_cases:
            f.write(f"    '{name}': {{\n")
            for key, val in config.items():
                f.write(f"        '{key}': {val},\n")
            f.write("    },\n")
        f.write("}\n\n")
        f.write("TestCases = TestCasesPerformance\n")

    print(f"已生成 {len(cases)} 个测试用例到: {filepath}")
    return module_name


def write_repeat_module(base_name: str, repeat: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    init_file = output_dir / "__init__.py"
    if not init_file.exists():
        init_file.touch()

    module_name = f"repeat_{base_name}"
    filepath = output_dir / (module_name + ".py")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# 自动生成的重复测试用例\n")
        f.write("TestCasesRepeat = {\n")
        for i in range(1, repeat + 1):
            f.write(f"    '{base_name}_run_{i}': {{\n")
            for key, val in BASE_CONFIG.items():
                f.write(f"        '{key}': {val},\n")
            f.write("    },\n")
        f.write("}\n\n")
        f.write("TestCases = TestCasesRepeat\n")

    print(f"已生成 {repeat} 个重复测试用例到: {filepath}")
    return module_name


def find_latest_mtool_dir(base_dir: Path = None, timeout: float = 300) -> Path:
    if base_dir is None:
        base_dir = Path.cwd()
    start_time = time.time()
    while time.time() - start_time < timeout:
        mtool_dirs = sorted(
            [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("PROF")],
            key=lambda d: d.stat().st_ctime,
            reverse=True,
        )
        if mtool_dirs:
            return mtool_dirs[0]
        time.sleep(1)
    raise TimeoutError(f"在 {timeout} 秒内未找到任何 mtool* 文件夹。")


def find_summary_csv(mtool_dir: Path) -> Path:
    csv_files = sorted(
        mtool_dir.glob("mindstudio_profiler_output/op_summary*.csv"),
        key=lambda f: f.stat().st_ctime,
        reverse=True,
    )
    if not csv_files:
        raise FileNotFoundError(f"在 {mtool_dir} 中未找到 op_summary*.csv 文件")
    return csv_files[0]


def parse_quant_rows(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    quant_df = df[df['Op Name'] == 'QuantFlashAttn'].copy()
    if quant_df.empty:
        raise ValueError("CSV 中没有找到 Op Name 为 'QuantFlashAttn' 的行，请确认筛选条件。")
    return quant_df


# ==================== 绘图函数 ====================

REQUIRED_COLS = ['Task Duration(us)', 'aicore_time(us)', 'aiv_time(us)']

def check_required_cols(df):
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise KeyError(f"CSV 中缺少列: {col}，现有列: {list(df.columns)}")


def plot_sweep_1param(param_name: str, param_values: list, quant_df: pd.DataFrame,
                      output_plot: str):
    if len(quant_df) != len(param_values):
        print(f"警告: quant 行数 ({len(quant_df)}) 与参数值个数 ({len(param_values)}) 不一致，将取较小值。")
        n = min(len(quant_df), len(param_values))
        quant_df = quant_df.iloc[:n]
        param_values = param_values[:n]

    sorted_idx = sorted(range(len(param_values)), key=lambda i: param_values[i])
    param_sorted = [param_values[i] for i in sorted_idx]
    quant_sorted = quant_df.iloc[sorted_idx].reset_index(drop=True)

    check_required_cols(quant_sorted)

    task_dur = quant_sorted['Task Duration(us)'].astype(float).values
    aicore   = quant_sorted['aicore_time(us)'].astype(float).values
    aiv      = quant_sorted['aiv_time(us)'].astype(float).values

    plt.figure(figsize=(8, 5))
    plt.plot(param_sorted, task_dur, marker='o', label='Task Duration(us)')
    plt.plot(param_sorted, aicore,   marker='s', label='aicore_time(us)')
    plt.plot(param_sorted, aiv,      marker='^', label='aiv_time(us)')
    plt.xlabel(param_name)
    plt.ylabel('Time (us)')
    plt.title(f'Quant flash attention time vs {param_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    print(f"曲线图已保存: {output_plot}")
    plt.close()


def plot_sweep_2param_heatmap(param1_name, param1_values, param2_name, param2_values,
                               quant_df, output_plot):
    n1 = len(param1_values)
    n2 = len(param2_values)
    expected = n1 * n2

    if len(quant_df) != expected:
        print(f"警告: quant 行数 ({len(quant_df)}) 与笛卡尔积数量 ({expected}) 不一致，将取较小值。")
        quant_df = quant_df.iloc[:min(len(quant_df), expected)]

    check_required_cols(quant_df)

    p1_sorted = sorted(param1_values)
    p2_sorted = sorted(param2_values)

    task_dur = quant_df['Task Duration(us)'].astype(float).values

    heatmap = np.full((len(p2_sorted), len(p1_sorted)), np.nan)
    p1_idx_map = {v: i for i, v in enumerate(p1_sorted)}
    p2_idx_map = {v: i for i, v in enumerate(p2_sorted)}

    for row_idx, row in quant_df.iterrows():
        p1_val = param1_values[row_idx % n1] if row_idx < n1 * n2 else None
        p2_val = param2_values[row_idx // n1] if row_idx < n1 * n2 else None
        if p1_val in p1_idx_map and p2_val in p2_idx_map:
            heatmap[p2_idx_map[p2_val], p1_idx_map[p1_val]] = row['Task Duration(us)']

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap, aspect='auto', origin='lower', cmap='YlOrRd')

    ax.set_xticks(range(len(p1_sorted)))
    ax.set_xticklabels([str(v) for v in p1_sorted])
    ax.set_yticks(range(len(p2_sorted)))
    ax.set_yticklabels([str(v) for v in p2_sorted])
    ax.set_xlabel(param1_name)
    ax.set_ylabel(param2_name)
    ax.set_title(f'Task Duration Heatmap: {param1_name} × {param2_name}')

    for i in range(len(p2_sorted)):
        for j in range(len(p1_sorted)):
            val = heatmap[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                        fontsize=7, color='black' if val < np.nanmax(heatmap) * 0.5 else 'white')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Task Duration (us)')
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    print(f"热力图已保存: {output_plot}")
    plt.close()


def plot_sweep_2param_grouped(param1_name, param1_values, param2_name, param2_values,
                               quant_df, output_plot):
    n1 = len(param1_values)
    n2 = len(param2_values)
    expected = n1 * n2

    if len(quant_df) != expected:
        print(f"警告: quant 行数 ({len(quant_df)}) 与笛卡尔积数量 ({expected}) 不一致，将取较小值。")
        quant_df = quant_df.iloc[:min(len(quant_df), expected)]

    check_required_cols(quant_df)

    p1_sorted = sorted(param1_values)
    p2_sorted = sorted(param2_values)

    task_dur = quant_df['Task Duration(us)'].astype(float).values
    aicore   = quant_df['aicore_time(us)'].astype(float).values
    aiv      = quant_df['aiv_time(us)'].astype(float).values

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    metrics = [
        ('Task Duration(us)', task_dur),
        ('aicore_time(us)', aicore),
        ('aiv_time(us)', aiv),
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, len(p2_sorted)))

    for ax_idx, (metric_name, metric_data) in enumerate(metrics):
        ax = axes[ax_idx]
        for p2_i, p2_val in enumerate(p2_sorted):
            row_start = p2_i * n1
            row_end = row_start + n1
            if row_end <= len(metric_data):
                y_vals = metric_data[row_start:row_end]
                ax.plot(p1_sorted, y_vals, marker='o', linestyle='-',
                        color=colors[p2_i],
                        label=f'{param2_name}={p2_val}')
        ax.set_xlabel(param1_name)
        ax.set_ylabel('Time (us)')
        ax.set_title(metric_name)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True)

    fig.suptitle(f'Grouped Curves: {param1_name} × {param2_name}', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"分组折线图已保存: {output_plot}")
    plt.close()


def plot_repeat_curves(run_indices: list, quant_df: pd.DataFrame, base_name: str,
                       output_plot: str):
    check_required_cols(quant_df)

    task_dur = quant_df['Task Duration(us)'].astype(float).values
    aicore   = quant_df['aicore_time(us)'].astype(float).values
    aiv      = quant_df['aiv_time(us)'].astype(float).values

    total_runs = len(run_indices)
    if total_runs > 1:
        task_dur_excl_first = task_dur[1:]
        aicore_excl_first   = aicore[1:]
        aiv_excl_first      = aiv[1:]
    else:
        task_dur_excl_first = task_dur
        aicore_excl_first   = aicore
        aiv_excl_first      = aiv

    valid_runs = len(task_dur_excl_first)

    def stats(arr, name):
        mean = np.mean(arr)
        std = np.std(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        print(f"{name}: 平均值={mean:.2f} us, 标准差={std:.2f} us, "
              f"最小值={min_val:.2f} us, 最大值={max_val:.2f} us")
        return mean, std, min_val, max_val

    print(f"\n===== {base_name} 重复测试统计 (共{total_runs}次，排除第1次后统计{valid_runs}次) =====")
    mean_task, std_task, min_task, max_task = stats(task_dur_excl_first, "Task Duration(us)")
    mean_aicore, std_aicore, min_aicore, max_aicore = stats(aicore_excl_first, "aicore_time(us)")
    mean_aiv, std_aiv, min_aiv, max_aiv = stats(aiv_excl_first, "aiv_time(us)")

    fluctuation_text = ""
    if total_runs >= 10:
        def calc_fluctuation(mean_val, min_val, max_val, name):
            if mean_val > 1.0:
                ratio = (max_val - min_val) / mean_val
                return f"{name} 波动比例: {ratio:.4f} ({ratio*100:.2f}%)"
            else:
                amplitude = max_val - min_val
                return f"{name} 波动幅度: {amplitude:.4f} us"

        fluct_task = calc_fluctuation(mean_task, min_task, max_task, "Task Duration")
        fluct_aicore = calc_fluctuation(mean_aicore, min_aicore, max_aicore, "aicore_time")
        fluct_aiv = calc_fluctuation(mean_aiv, min_aiv, max_aiv, "aiv_time")

        print("\n--- 波动指标 (排除第1次) ---")
        print(fluct_task)
        print(fluct_aicore)
        print(fluct_aiv)

        fluctuation_text = f"波动指标 (排除第1次):\n{fluct_task}\n{fluct_aicore}\n{fluct_aiv}"

    plt.figure(figsize=(10, 6))
    plt.plot(run_indices, task_dur, marker='o', linestyle='-', label='Task Duration(us)')
    plt.plot(run_indices, aicore,   marker='s', linestyle='-', label='aicore_time(us)')
    plt.plot(run_indices, aiv,      marker='^', linestyle='-', label='aiv_time(us)')

    plt.axhline(y=mean_task, color='C0', linestyle='--', alpha=0.7,
                label=f'Mean Task Duration (excl 1st): {mean_task:.2f}')
    plt.axhline(y=mean_aicore, color='C1', linestyle='--', alpha=0.7,
                label=f'Mean aicore (excl 1st): {mean_aicore:.2f}')
    plt.axhline(y=mean_aiv, color='C2', linestyle='--', alpha=0.7,
                label=f'Mean aiv (excl 1st): {mean_aiv:.2f}')

    plt.xlabel('Run Index')
    plt.ylabel('Time (us)')
    plt.title(f'Repeat Test: {base_name} ({total_runs} runs, stats exclude 1st)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"\n曲线图已保存: {output_plot}")
    plt.close()


# ===================== 主流程 =====================

def main():
    parser = argparse.ArgumentParser(
        description="参数扫描 / 重复性测试 与绘图工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单参数扫描 - 预设值
  python test_qfa_performence.py --sweep s2:preset=seq_len

  # 单参数扫描 - 显式值（支持 k 后缀，1k=1024）
  python test_qfa_performence.py --sweep s2:values=512,1k,4k,8k

  # 单参数扫描 - 范围步长（起始,步长,次数）
  python test_qfa_performence.py --sweep s2:range=128,128,5

  # 单参数扫描 - 随机取值（次数,最小值,最大值）
  python test_qfa_performence.py --sweep s2:random=5,128,8192

  # 双参数笛卡尔积（最多2个参数）
  python test_qfa_performence.py --sweep s2:values=512,1k,4k --sweep b:values=1,2,4

  # 修改基础配置后再扫描
  python test_qfa_performence.py --base-override s1=256 --sweep s2:preset=seq_len

  # 重复测试模式（运行基础用例10次）
  python test_qfa_performence.py --repeat 10

可用预设: seq_len, batch, head_num, head_dim, quant_mode
        """
    )

    parser.add_argument('--base', type=str, default='performence',
                        help='基础用例名称前缀 (默认: performence)')

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--sweep', action='append',
                      help='参数扫描规格，格式: param:method=spec (可指定1~2次，多参数生成笛卡尔积)')
    mode.add_argument('--repeat', type=int,
                      help='重复运行基础用例的次数（不进行参数扫描）')

    parser.add_argument('--base-override', action='append', default=None,
                        help='覆盖基础配置值，格式: key=value (可多次指定)')
    parser.add_argument('--random-fill', action='store_true',
                        help='绑定列表参数使用随机值填充')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')

    parser.add_argument('--device-id', type=int, default=0, help='NPU device id')
    parser.add_argument('--timeout', type=float, default=300, help='等待 mtool 超时时间')
    parser.add_argument('--output-plot', type=str, default=None, help='输出图片文件名前缀')
    parser.add_argument('--keep-cases', action='store_true', help='保留临时用例文件')

    args = parser.parse_args()

    if args.repeat is not None and args.repeat <= 1:
        parser.error("--repeat 必须是一个大于 1 的整数")

    if args.sweep is not None and len(args.sweep) > 2:
        parser.error("最多只能指定 2 个 --sweep 参数（双参数笛卡尔积）")

    rng = random.Random(args.seed) if args.seed is not None else random.Random()

    output_dir = Path("test_cases")
    module_name = None
    sweep_params = None

    if args.sweep is not None:
        sweep_specs = [parse_sweep_spec(s) for s in args.sweep]

        param_names_list = [spec[0] for spec in sweep_specs]
        seen = set()
        duplicates = [p for p in param_names_list if p in seen or seen.add(p)]
        if duplicates:
            parser.error(f"重复的扫描参数: {duplicates}，每个参数只能指定一次")

        resolved_sweeps = resolve_sweep_values(sweep_specs, rng=rng)

        base_config = deepcopy(BASE_CONFIG)
        if args.base_override:
            for ov in args.base_override:
                parts = ov.split('=', 1)
                if len(parts) != 2:
                    raise ValueError(f"覆盖格式错误: '{ov}', 应为 key=value")
                key = parts[0].strip()
                value = parse_value(parts[1])
                if key not in base_config:
                    raise ValueError(f"未知参数: '{key}', 可用参数: {list(base_config.keys())}")
                base_config[key] = [value]

        base_config = apply_bindings(base_config, random_fill=args.random_fill, rng=rng)

        print("扫描参数及取值:")
        for param_name, values in resolved_sweeps:
            print(f"  {param_name}: {values}")

        cases = generate_sweep_cases(base_config, resolved_sweeps,
                                     random_fill=args.random_fill, rng=rng)
        print(f"共生成 {len(cases)} 个测试用例")

        sweep_params = resolved_sweeps
        module_name = write_cases_module(cases, sweep_params, output_dir)

    else:
        repeat = args.repeat
        print(f"重复测试模式: 基础用例 '{args.base}' 运行 {repeat} 次")
        module_name = write_repeat_module(args.base, repeat, output_dir)

    # ==================== 执行 msprof ====================
    cmd = [
        "msprof", "python", "test_quant_flash_attention.py",
        "--case_files", module_name,
        "--case_id", "all",
        "--device_id", str(args.device_id),
    ]
    print("执行命令:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败，返回码: {e.returncode}", file=sys.stderr)
        if not args.keep_cases:
            (output_dir / f"{module_name}.py").unlink(missing_ok=True)
        sys.exit(e.returncode)

    # ==================== 查找结果文件 ====================
    try:
        mtool_dir = find_latest_mtool_dir(timeout=args.timeout)
        csv_file = find_summary_csv(mtool_dir)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        if not args.keep_cases:
            (output_dir / f"{module_name}.py").unlink(missing_ok=True)
        sys.exit(1)

    print(f"找到结果文件: {csv_file}")

    # ==================== 解析并绘图 ====================
    quant_df = parse_quant_rows(csv_file)

    if args.sweep is not None:
        param_suffix = "_".join([p for p, _ in sweep_params])
        base_prefix = args.output_plot or f"sweep_{args.base}_{param_suffix}"

        if len(sweep_params) == 1:
            param_name, param_values = sweep_params[0]
            output_plot = f"{base_prefix}.png"
            plot_sweep_1param(param_name, param_values, quant_df, output_plot)
        elif len(sweep_params) == 2:
            param1_name, param1_values = sweep_params[0]
            param2_name, param2_values = sweep_params[1]
            heatmap_plot = f"{base_prefix}_heatmap.png"
            grouped_plot = f"{base_prefix}_grouped.png"
            plot_sweep_2param_heatmap(param1_name, param1_values,
                                      param2_name, param2_values,
                                      quant_df, heatmap_plot)
            plot_sweep_2param_grouped(param1_name, param1_values,
                                      param2_name, param2_values,
                                      quant_df, grouped_plot)
    else:
        output_plot = args.output_plot or f"repeat_{args.base}.png"
        run_indices = list(range(1, args.repeat + 1))
        plot_repeat_curves(run_indices, quant_df, args.base, output_plot)

    # ==================== 清理 ====================
    if not args.keep_cases:
        (output_dir / f"{module_name}.py").unlink(missing_ok=True)
        print(f"临时用例文件已删除")
    else:
        print(f"用例文件保留在: {output_dir / module_name}.py")

    print("全流程完成！")


if __name__ == "__main__":
    main()