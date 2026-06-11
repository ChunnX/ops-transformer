#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精度泛化测试工具：
基于基础配置，针对一个或多个参数构造泛化用例，
支持预设值、显式值、范围步长、随机取值等方式，
多参数时自动生成笛卡尔积测试集，
然后调用 test_quant_flash_attention.py 执行精度测试。

用法示例:
  # 单参数扫描 - 预设值
  python test_qfa_accuracy.py --sweep s2:preset=seq_len

  # 单参数扫描 - 显式值（支持 k 后缀，1k=1024）
  python test_qfa_accuracy.py --sweep s2:values=512,1k,4k,8k

  # 单参数扫描 - 范围步长（起始,步长,次数）
  python test_qfa_accuracy.py --sweep s2:range=128,128,5

  # 单参数扫描 - 随机取值（次数,最小值,最大值）
  python test_qfa_accuracy.py --sweep s2:random=5,128,8192

  # 多参数笛卡尔积
  python test_qfa_accuracy.py --sweep s2:values=512,1k,4k --sweep b:values=1,2,4

  # 修改基础配置后再扫描
  python test_qfa_accuracy.py --base-override s1=256 --sweep s2:preset=seq_len

可用预设: seq_len, batch, head_num, head_dim, quant_mode
"""

import argparse
import subprocess
import sys
import random
import itertools
from pathlib import Path
from copy import deepcopy


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
    'act_seq_lens_kv': [[512]],
}

BIND_GROUPS = [
    {'len_param': 'b', 'max_param': 's1', 'list_param': 'act_seq_lens_q'},
    {'len_param': 'b', 'max_param': 's2', 'list_param': 'act_seq_lens_kv'},
]

PRESETS = {
    'seq_len': [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
    'batch': [1, 2, 4, 8, 16, 32],
}


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


def apply_bindings(config, random_fill=False, rng=None):
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


def generate_cartesian_cases(base_config, resolved_sweeps, random_fill=False, rng=None):
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


def write_cases_module(cases, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    init_file = output_dir / "__init__.py"
    if not init_file.exists():
        init_file.touch()

    module_name = "sweep_accuracy_cases"
    filepath = output_dir / (module_name + ".py")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# 自动生成的精度泛化测试用例\n")
        f.write("TestCasesAccuracy = {\n")
        for name, config in cases.items():
            f.write(f"    '{name}': {{\n")
            for key, val in config.items():
                f.write(f"        '{key}': {val},\n")
            f.write("    },\n")
        f.write("}\n\n")
        f.write("TestCases = TestCasesAccuracy\n")

    print(f"已生成 {len(cases)} 个测试用例到: {filepath}")
    return module_name


def main():
    parser = argparse.ArgumentParser(
        description="精度泛化测试工具：基于基础配置生成泛化用例并执行精度测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单参数扫描 - 预设值
  python test_qfa_accuracy.py --sweep s2:preset=seq_len

  # 单参数扫描 - 显式值（支持 k 后缀，1k=1024）
  python test_qfa_accuracy.py --sweep s2:values=512,1k,4k,8k

  # 单参数扫描 - 范围步长（起始,步长,次数）
  python test_qfa_accuracy.py --sweep s2:range=128,128,5

  # 单参数扫描 - 随机取值（次数,最小值,最大值）
  python test_qfa_accuracy.py --sweep s2:random=5,128,8192

  # 多参数笛卡尔积
  python test_qfa_accuracy.py --sweep s2:values=512,1k,4k --sweep b:values=1,2,4

  # 修改基础配置后再扫描
  python test_qfa_accuracy.py --base-override s1=256 --sweep s2:preset=seq_len

可用预设: seq_len, batch, head_num, head_dim, quant_mode
        """
    )

    parser.add_argument('--sweep', action='append', required=True,
                        help='参数扫描规格，格式: param:method=spec (可多次指定，多参数生成笛卡尔积)')
    parser.add_argument('--base-override', action='append', default=None,
                        help='覆盖基础配置值，格式: key=value (可多次指定)')
    parser.add_argument('--random-fill', action='store_true',
                        help='绑定列表参数使用随机值填充')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--device-id', type=int, default=0, help='NPU device id')
    parser.add_argument('--keep-cases', action='store_true', help='保留临时用例文件')
    parser.add_argument('--dry-run', action='store_true', help='仅生成用例不执行测试')

    args = parser.parse_args()

    rng = random.Random(args.seed) if args.seed is not None else random.Random()

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

    sweep_specs = [parse_sweep_spec(s) for s in args.sweep]

    param_names_list = [spec[0] for spec in sweep_specs]
    seen = set()
    duplicates = [p for p in param_names_list if p in seen or seen.add(p)]
    if duplicates:
        parser.error(f"重复的扫描参数: {duplicates}，每个参数只能指定一次")

    resolved_sweeps = resolve_sweep_values(sweep_specs, rng=rng)

    print("扫描参数及取值:")
    for param_name, values in resolved_sweeps:
        print(f"  {param_name}: {values}")

    cases = generate_cartesian_cases(base_config, resolved_sweeps,
                                     random_fill=args.random_fill, rng=rng)
    print(f"共生成 {len(cases)} 个测试用例 (笛卡尔积)")

    output_dir = Path(__file__).parent / "test_cases"
    module_name = write_cases_module(cases, output_dir)

    if args.dry_run:
        print("\n[DRY RUN] 用例已生成，跳过测试执行")
        for name, config in cases.items():
            print(f"  {name}: {dict(config)}")
        if not args.keep_cases:
            (output_dir / f"{module_name}.py").unlink(missing_ok=True)
            print("临时用例文件已删除")
        return

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "test_quant_flash_attention.py"),
        "--case_files", module_name,
        "--case_id", "all",
        "--device_id", str(args.device_id),
    ]
    print("\n执行命令:", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败，返回码: {e.returncode}", file=sys.stderr)
        if not args.keep_cases:
            (output_dir / f"{module_name}.py").unlink(missing_ok=True)
        sys.exit(e.returncode)

    if not args.keep_cases:
        (output_dir / f"{module_name}.py").unlink(missing_ok=True)
        print("临时用例文件已删除")
    else:
        print(f"用例文件保留在: {output_dir / module_name}.py")

    print("全流程完成！")


if __name__ == "__main__":
    main()