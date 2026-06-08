#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一站式参数扫描 / 重复性测试 与绘图工具：
1. 参数扫描模式：根据基础配置生成扫描用例 → mtool 执行 → 解析 CSV → 绘制参数‑耗时曲线
2. 重复测试模式：仅运行基础用例 N 次 → 绘制耗时曲线 + 统计平均值
"""

import argparse
import subprocess
import time
import sys
import random
from pathlib import Path
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
# ==================================================


def apply_bindings(config: dict, random_fill: bool = False, rng: random.Random = None) -> dict:
    """根据绑定关系自动更新列表参数，列表参数格式为双层列表（如 [[1,2,3]]）。"""
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


def generate_cases(base_name: str, param: str, values: list,
                   random_fill: bool = False, seed: int = None) -> dict:
    """生成所有测试用例字典，返回 {case_name: config_dict}。"""
    rng = random.Random(seed) if seed is not None else random
    cases = {}
    for v in values:
        case_name = f"{base_name}_{param}_{v}"
        new_config = {k: v_list.copy() for k, v_list in BASE_CONFIG.items()}
        new_config[param] = [v]
        new_config = apply_bindings(new_config, random_fill, rng)
        cases[case_name] = new_config
    return cases


def write_cases_module(cases: dict, base_name: str, param: str, output_dir: Path):
    """写入用例文件，返回不带 .py 的模块名（已按参数值升序）。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    init_file = output_dir / "__init__.py"
    if not init_file.exists():
        init_file.touch()

    # 按参数值升序排序
    sorted_cases = sorted(cases.items(), key=lambda item: item[1][param][0])
    module_name = f"sweep_{base_name}_{param}"
    filepath = output_dir / (module_name + ".py")

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
    """
    生成重复测试的用例模块，包含 repeat 个相同的基础用例。
    用例名格式：{base_name}_run_1, {base_name}_run_2, ...
    """
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
    """轮询查找最新创建的 mtool* 文件夹。"""
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
    """在 mtool 文件夹中查找 op_summary 开头的 CSV 文件。"""
    csv_files = sorted(
        mtool_dir.glob("mindstudio_profiler_output/op_summary*.csv"),
        key=lambda f: f.stat().st_ctime,
        reverse=True,
    )
    if not csv_files:
        raise FileNotFoundError(f"在 {mtool_dir} 中未找到 op_summary*.csv 文件")
    return csv_files[0]


def parse_quant_rows(csv_path: Path) -> pd.DataFrame:
    """读取 CSV，返回 Op Name == 'QuantFlashAttn' 的 DataFrame，保留原始顺序。"""
    df = pd.read_csv(csv_path)
    quant_df = df[df['Op Name'] == 'QuantFlashAttn'].copy()
    if quant_df.empty:
        raise ValueError("CSV 中没有找到 Op Name 为 'QuantFlashAttn' 的行，请确认筛选条件。")
    return quant_df


# ==================== 绘图函数 ====================

def plot_sweep_curves(param_name: str, param_values: list, quant_df: pd.DataFrame, 
                       output_plot: str):
    """参数扫描曲线：Task Duration, aicore_time, aiv_time vs 参数值。"""
    if len(quant_df) != len(param_values):
        print(f"警告: quant 行数 ({len(quant_df)}) 与参数值个数 ({len(param_values)}) 不一致，将取较小值。")
        n = min(len(quant_df), len(param_values))
        quant_df = quant_df.iloc[:n]
        param_values = param_values[:n]

    # 按参数值排序
    sorted_idx = sorted(range(len(param_values)), key=lambda i: param_values[i])
    param_sorted = [param_values[i] for i in sorted_idx]
    quant_sorted = quant_df.iloc[sorted_idx].reset_index(drop=True)

    required_cols = ['Task Duration(us)', 'aicore_time(us)', 'aiv_time(us)']
    for col in required_cols:
        if col not in quant_sorted.columns:
            raise KeyError(f"CSV 中缺少列: {col}，现有列: {list(quant_sorted.columns)}")

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


def plot_repeat_curves(run_indices: list, quant_df: pd.DataFrame, base_name: str,
                        output_plot: str):
    """
    重复测试曲线：绘制每次运行的耗时折线图 + 平均值虚线。
    统计信息（平均值、标准差）排除第一次运行的数据，但图中保留第一个点。
    如果 repeat >= 10 且平均耗时 > 1us，计算波动比例 (max-min)/mean；
    否则计算波动幅度 (max-min)，在图旁标注并打印日志。
    """
    required_cols = ['Task Duration(us)', 'aicore_time(us)', 'aiv_time(us)']
    for col in required_cols:
        if col not in quant_df.columns:
            raise KeyError(f"CSV 中缺少列: {col}，现有列: {list(quant_df.columns)}")

    task_dur = quant_df['Task Duration(us)'].astype(float).values
    aicore   = quant_df['aicore_time(us)'].astype(float).values
    aiv      = quant_df['aiv_time(us)'].astype(float).values

    total_runs = len(run_indices)
    # 排除第一次运行的数据（索引0），用于计算统计信息
    if total_runs > 1:
        task_dur_excl_first = task_dur[1:]
        aicore_excl_first   = aicore[1:]
        aiv_excl_first      = aiv[1:]
    else:
        task_dur_excl_first = task_dur
        aicore_excl_first   = aicore
        aiv_excl_first      = aiv

    valid_runs = len(task_dur_excl_first)

    # 计算统计信息（基于排除第一次后的数据）
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

    # 计算波动指标（仅当总次数 >= 10 时）
    fluctuation_text = ""
    if total_runs >= 10:
        def calc_fluctuation(mean_val, min_val, max_val, name):
            """计算波动比例或波动幅度"""
            if mean_val > 1.0:  # 平均耗时 > 1us
                ratio = (max_val - min_val) / mean_val
                return f"{name} 波动比例: {ratio:.4f} ({ratio*100:.2f}%)"
            else:  # 平均耗时 <= 1us
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

    # 绘图（保留所有数据点）
    plt.figure(figsize=(10, 6))
    plt.plot(run_indices, task_dur, marker='o', linestyle='-', label='Task Duration(us)')
    plt.plot(run_indices, aicore,   marker='s', linestyle='-', label='aicore_time(us)')
    plt.plot(run_indices, aiv,      marker='^', linestyle='-', label='aiv_time(us)')

    # 平均值水平线（基于排除第一次后的数据）
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
  # 参数扫描模式
  python test_qfa_performence.py --param s1 --values 1,2,4,8
  python test_qfa_performence.py --param s1 --range 1 2 3

  # 重复测试模式（运行基础用例10次）
  python test_qfa_performence.py --repeat 10

  # 重复测试模式 + 指定基础用例名
  python test_qfa_performence.py --base my_case --repeat 20
        """
    )

    # 基础参数
    parser.add_argument('--base', type=str, default='performence',
                        help='基础用例名称前缀 (默认: performence)')

    # 模式选择（互斥）
    mode = parser.add_mutually_exclusive_group(required=True)
    # 参数扫描模式参数
    mode.add_argument('--param', type=str, help='要扫描的参数名，例如 s1')
    # 重复测试模式参数
    mode.add_argument('--repeat', type=int, help='重复运行基础用例的次数（不进行参数扫描）')

    # 参数扫描所需的值
    value_group = parser.add_argument_group('参数扫描值（与 --param 配合使用）')
    value_spec = value_group.add_mutually_exclusive_group()
    value_spec.add_argument('--values', type=str, help='逗号分隔的参数值列表，如：1,2,4,8')
    value_spec.add_argument('--range', nargs=3, type=float,
                            metavar=('START', 'STEP', 'COUNT'),
                            help='起始值、步长、生成次数，例如：1 2 3 生成 1,3,5')

    # 绑定选项
    parser.add_argument('--random-fill', action='store_true',
                        help='绑定列表参数使用随机值填充')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')

    # 执行选项
    parser.add_argument('--device-id', type=int, default=0, help='NPU device id')
    parser.add_argument('--timeout', type=float, default=300, help='等待 mtool 超时时间')
    parser.add_argument('--output-plot', type=str, default=None, help='输出图片文件名')
    parser.add_argument('--keep-cases', action='store_true', help='保留临时用例文件')

    args = parser.parse_args()

    # ======== 验证参数组合 ========
    if args.repeat is not None and args.repeat <= 1:
        parser.error("--repeat 必须是一个大于 1 的整数")

    if args.param is not None:
        # 参数扫描模式：必须提供 --values 或 --range
        if args.values is None and args.range is None:
            parser.error("参数扫描模式需要 --values 或 --range")
    elif args.repeat is not None:
        # 重复测试模式：不需要 --values/--range
        if args.values is not None or args.range is not None:
            parser.error("重复测试模式不支持 --values/--range")
    # ==============================

    output_dir = Path("test_cases")
    module_name = None
    sorted_case_names = None

    if args.param is not None:
        # ==================== 参数扫描模式 ====================
        if args.values:
            values = []
            for part in args.values.split(','):
                part = part.strip()
                try:
                    values.append(int(part))
                except ValueError:
                    values.append(float(part))
        else:
            start, step, count = args.range
            count = int(count)
            values = [start + i * step for i in range(count)]
            values = [int(v) if isinstance(v, float) and v.is_integer() else v for v in values]

        print(f"参数扫描模式: {args.param}, 取值: {values}")

        cases = generate_cases(args.base, args.param, values,
                               random_fill=args.random_fill, seed=args.seed)
        module_name = write_cases_module(cases, args.base, args.param, output_dir)
        sorted_case_names = [name for name, _ in
                             sorted(cases.items(), key=lambda item: item[1][args.param][0])]
        case_files_arg = module_name
        case_id_arg = ",".join(sorted_case_names) if sorted_case_names else "all"

    else:
        # ==================== 重复测试模式 ====================
        repeat = args.repeat
        print(f"重复测试模式: 基础用例 '{args.base}' 运行 {repeat} 次")

        module_name = write_repeat_module(args.base, repeat, output_dir)
        sorted_case_names = [f"{args.base}_run_{i}" for i in range(1, repeat + 1)]
        case_files_arg = module_name
        case_id_arg = ",".join(sorted_case_names)

    # ==================== 执行 mtool ====================
    cmd = [
        "msprof", "python", "test_quant_flash_attention.py",
        "--case_files", case_files_arg,
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

    if args.param is not None:
        # 参数扫描绘图
        if args.output_plot is None:
            output_plot = f"sweep_{args.base}_{args.param}.png"
        else:
            output_plot = args.output_plot
        plot_sweep_curves(args.param, values, quant_df, output_plot)
    else:
        # 重复测试绘图
        if args.output_plot is None:
            output_plot = f"repeat_{args.base}.png"
        else:
            output_plot = args.output_plot
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