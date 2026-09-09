"""用明确的设备 Add 标记匹配 FlashKDA profile，再与归档 CGDR/CANN KDA 配对。"""
import argparse
from decimal import Decimal
import json
import math
from pathlib import Path
import shutil
import statistics as st

from flash_kda_rdv_perf import CASES, REPO
from kda_analyze_rdv import digest, read_csv, write_csv

CANN_KDA = REPO / "attention/chunk_gated_delta_rule/docs/kda_perf/comparison.csv"


def summary_group(name, rows):
    if not rows:
        return {"group": name, "cases": 0}
    result = {"group": name, "cases": len(rows)}
    for target in ("cgdr", "cann_kda"):
        pairs = [x for x in rows if x[f"flash_over_{target}_ratio"] != ""]
        ratios = [x[f"flash_over_{target}_ratio"] for x in pairs]
        result.update({f"{target}_paired_cases": len(pairs),
            f"geomean_flash_over_{target}": math.exp(st.mean(math.log(x) for x in ratios)),
            f"median_flash_over_{target}": st.median(ratios),
            f"sum_flash_over_{target}": sum(x["flash_mean_us"] for x in pairs) / sum(float(x[f"{target}_mean_us"]) for x in pairs),
            f"flash_faster_than_{target}_cases": sum(x < 1 for x in ratios)})
    return result


def write_readme(out, summary, paired):
    groups = {x["group"]: x for x in summary["groups"]}
    table = []
    for key, label in [("ALL_MEASURED", "全部可执行映射"), ("NO_SEQUENCE_PADDING", "序列不补齐"),
                       ("NO_PADDING_CONTIGUOUS_SOURCE", "不补齐且原初态连续"),
                       ("NO_PADDING_CONTIGUOUS_FP32", "不补齐、原初态连续且 FP32"), ("SEQUENCE_PADDING", "序列补齐")]:
        x = groups[key]
        table.append(f"| {label} | {x['cases']} | {x['geomean_flash_over_cgdr']:.3f}× | {x['geomean_flash_over_cann_kda']:.3f}× |")
    selected = []
    bn_table = []
    for key, label in [("0<B*Nv<=4", "B×Nv ≤ 4"), ("4<B*Nv<=16", "4 < B×Nv ≤ 16"),
                       ("16<B*Nv<=32", "16 < B×Nv ≤ 32"), ("32<B*Nv<=inf", "B×Nv > 32")]:
        if key in groups:
            x = groups[key]
            bn_table.append(f"| {label} | {x['cases']} | {x['geomean_flash_over_cgdr']:.3f}× | {x['geomean_flash_over_cann_kda']:.3f}× |")
    names = {"ARC-001", "ARC-004", "ARC-009", "ARC-011", "ARC-017", "ARC-018", "ARC-036", "ASC-001"}
    for x in paired:
        if x["test_name"] in names and x["flash_status"] == "PASSED":
            selected.append(f"| {x['test_name']} | {x['B']}/{x['seqlen']}/{x['nk']}/{x['nv']} | "
                f"{float(x['cgdr_mean_us']):.3f} | {float(x['cann_kda_mean_us']):.3f} | {x['flash_mean_us']:.3f} | "
                f"{x['flash_over_cgdr_ratio']:.3f}× | {x['flash_over_cann_kda_ratio']:.3f}× |")
    text = f"""# FlashKDA RDV 性能对比（2026-09-09）

调用 `/home/s00940320/cannbot-dsl/samples/flash_kda/flash_kda.py` 的公开 `flash_kda`，
源仓库版本和文件哈希见 [environment.json](environment.json)。没有修改 DSL 内核或覆盖默认 tiling。
对比 [新版 CGDR](../rdv_perf/new_streaming/comparison.csv) 和 [CANN 内置 KDA](../kda_perf/comparison.csv) 的历史归档主 kernel 耗时。

保留全部 **288** 个 RDV 条目，**{summary['measured_cases']}** 条获得性能，
**{summary['unsupported_cases']}** 条不满足 FlashKDA 的 Dk=Dv=128 要求，时延留空。
其中 **{summary['sequence_padding_cases']}** 条补齐序列，**{summary['contiguous_copy_cases']}** 条将原非连续初态转为连续，均在表中标注。

## 性能数据

每例先独立执行一次 JIT/常量初始化，再记录 1 次预热和 5 次正式测量。
比值为 `FlashKDA / 对照算子`，小于 1 表示 FlashKDA 更快；单位 µs。
只统计 msprof 设备主 kernel 的 `Task Duration(us)`，不含 JIT、输入适配、Python、workspace 分配和 host 等待时间。
用两次独特长度的 Add 标记确定每条用例的边界，标记本身不计时；逐次样本见 [kernel_samples.csv](kernel_samples.csv)。
DSL 直接下发的 kernel 在 msprof 中不提供输入 shape；运行时张量的 shape/dtype/stride 和分块参数
保存在 [execution_results.json](execution_results.json)，并与精度运行的配置核对。

| 范围 | 条目数 | Flash/CGDR 几何平均耗时比 | Flash/CANN KDA 几何平均耗时比 |
|---|---:|---:|---:|
{chr(10).join(table)}

全部可执行条目中，FlashKDA 有 **{groups['ALL_MEASURED']['flash_faster_than_cgdr_cases']}/{summary['measured_cases']}** 条快于新版 CGDR，
**{groups['ALL_MEASURED']['flash_faster_than_cann_kda_cases']}/{summary['measured_cases']}** 条快于 CANN KDA。
最大 CV 为 **{summary['max_cv_pct']:.3f}%**。
全量数据见 [comparison.csv](comparison.csv)，均值、中位数、标准差和五次样本见 [case_performance.csv](case_performance.csv)。

| 用例 | B/S/Nk/Nv | CGDR | CANN KDA | FlashKDA | Flash/CGDR | Flash/CANN KDA |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(selected)}

按 B×Nv 分组（包含已标注的输入适配）：

| B×Nv 范围 | 条目数 | Flash/CGDR | Flash/CANN KDA |
|---|---:|---:|---:|
{chr(10).join(bn_table)}

FlashKDA 默认对较小 B×Nv 使用 Dv 分块：B×Nv≤4/8/16 时，`dv_base` 分别为 16/32/64，
其他为 128。源码还按 B×Nv 选择 chunk group，并在 group 内复用 workspace。
这些设计能增加小 B×Nv 场景的工作划分，但本轮没有固定其他条件做 tiling 消融测试，
因此不把分组性能差异单独归因于某一项设计。

## 输入约束与对比范围

- 原 RDV 的 B/S/Nk/Nv、BF16 Q/K/V、初态值生成类型、seed=20260908 保留。新环境的 PyTorch 版本不同，历史输入没有逐字节快照，因此不声称精确回放历史输入。
- FlashKDA 只支持 **Dk=Dv=128**；其余 52 条维度用例保留为 `UNSUPPORTED_SHAPE`，不填性能或精度通过。
- S 必须为 64 的倍数。ARC-017/018/034/035/036 的四个状态变体共 20 条，按 **6000→6016、5000→5056、65535→65536** 补齐。
  Q/K/V 补 0，raw gate/beta 补 -100，衰减在 FP32 中为 1，零 key 和 value 保证有效输出和最终状态不变。
- 公共接口只接受连续 FP32 初态。原 BF16 初态提升到 FP32，原 NC 初态真实构造后再 `.contiguous()`；转换在计时外。
  NC 条目的时延属于转换后的连续输入，不能用来说明内核原生支持 NC。CSV 保留源 stride 和转换标记。
- 使用 **BSND**、chunk_size=64、默认 group 和 Dv 分块。FlashKDA 内部执行 Q/K L2 normalize（epsilon=1e-6），且将结果量化为 BF16。
  本轮传入与先前 RDV 相同生成方式的归一化 Q/K，因此 FlashKDA 会再次归一化；这部分计算包含在 FlashKDA 主 kernel 中。
- FlashKDA 的 gate 和 beta 都是 **BF16 原始 logits**。per-dim 目标 gate 使用 seed=20260909；有 gate 时将目标 log-gate 逆变换为 safe-sigmoid raw 输入，A_log=0、dt_bias=0、lower_bound=-5。
  无 gate 时 lower_bound=0。beta 由 RDV 的 [0,1) 系数逆 sigmoid 后量化到 BF16，零端点用 -100 避免 -Inf。
  精度参考使用实际量化后的 logits，并重新执行激活，不把目标系数当作实际系数。
- 历史 CANN KDA 接收 FP32 gate、已激活 beta；CGDR 是标量 gate。FlashKDA 还融合 Q/K normalize 和 beta 激活。
  因此这些比值是完整实现之间的比较，不能解释为仅 per-dim 门控或仅 DSL 语言的开销。
- CANN KDA 另需导出 Aqk/Akk，并对部分用例采用 TND；FlashKDA 只返回 attention/final state。两个对照均来自历史测量，没有在本轮交错重测。

## 精度与环境验证

原仓库 `test/flash_kda/test_flash_kda.py` 的 NPU 测试通过，见 [original_pytest.log](original_pytest.log)。
RDV 独立检查 **{summary['precision_checked_cases']}** 条，其中 **{summary['precision_passed_cases']}** 条通过；
失败列表：`{summary['precision_failed_cases']}`。只对实际通过的条目填写 `flash_precision_passed=True`。

RDV 参考先按接口定义执行 Q/K 归一化和 gate/beta 激活，然后使用独立 CPU FP64 逐 token 递推。
原 NC 输入转换为连续后，如果全部实际输入以及两项 NPU 输出均与此前验证的用例逐元素完全相同，
则复用该用例的 FP64 误差；每条用例仍独立调用 FlashKDA，验证方式与复用来源逐条记录。
沿用前一轮 RMS/归一化最大误差判据：`relative_rms<=0.02`、`normalized_max<=0.10`、分母绝对下限 2e-5。
另外保留原样例 `atol=rtol=0.005` 的逐元素指标，未按测试结果改变门限。
该逐元素门限未通过 **{len(summary['precision_allclose_005_failed_cases'])}** 条，名单保存在 `summary.json`。
全部误差、参考复用与补齐等价性状态见 [precision_results.csv](precision_results.csv)，
原始精度执行记录见 [precision_execution_results.json](precision_execution_results.json)；不适用的条目明确留空，不能视为通过。

## 安装与复现

PyPI 发行包名称为 **cannbot-dsl**，导入名称为 `cannbotdsl`，0.0.3 要求 Python 3.12。
来源见 [PyPI](https://pypi.org/project/cannbot-dsl/)，安装记录和源码哈希见 [environment.json](environment.json)。
本轮独立环境为 `/home/s00940320/.venvs/flash-kda-py312`，torch 2.12.0+cpu、torch_npu 2.12.0，仍使用系统 CANN 9.2.0。
前两轮对照使用 Python 3.10、torch/torch_npu 2.13 系列，差异已记录。

安装包名与 Python 导入名不同。新环境可先执行：

```bash
python3.12 -m venv /path/to/flash-kda-py312
/path/to/flash-kda-py312/bin/python -m pip install --index-url https://pypi.org/simple cannbot-dsl==0.0.3 numpy ml_dtypes pytest PyYAML ninja
/path/to/flash-kda-py312/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.0
/path/to/flash-kda-py312/bin/python -m pip install --index-url https://pypi.org/simple torch-npu==2.12.0
```

本机复现命令：

```bash
source /home/s00940320/.venvs/flash-kda-py312/activate_flash_kda.sh
export MAX_JOBS=4
flash_raw=attention/chunk_kda_fwd/tests/pytest/output/flash_kda_rdv_repeat
export CANNBOTDSL_CACHE_DIR="$PWD/$flash_raw/compiler_cache"
python attention/chunk_kda_fwd/tests/pytest/flash_kda_rdv_perf.py --output "$flash_raw/precision" --mode precision
mkdir -p "$flash_raw/full/prof"
msprof --output="$PWD/$flash_raw/full/prof" python attention/chunk_kda_fwd/tests/pytest/flash_kda_rdv_perf.py --output "$flash_raw/full"
printf '%s\\n' "$?" > "$flash_raw/full/tool_exit_code.txt"
python attention/chunk_kda_fwd/tests/pytest/flash_kda_analyze_rdv.py --raw "$flash_raw" --output attention/chunk_gated_delta_rule/docs/flash_kda_perf_repeat
```

[validation.json](validation.json) 保存行数、样本边界、数据配对和精度配置检查；
[metadata.json](metadata.json) 保存原始 profile、基线、脚本和产物 SHA256。
"""
    (out / "README.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw, out = args.raw.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    full = raw / "full"
    assert (full / "runner_exit_code.txt").read_text().strip() == "0"
    assert (full / "tool_exit_code.txt").read_text().strip() == "0"
    cases = read_csv(CASES)
    runtime = json.loads((full / "runtime.json").read_text())
    assert runtime["case_source_sha256"] == digest(CASES)
    manifest = json.loads((full / "case_manifest.json").read_text())
    assert [{k: v for k, v in x.items() if k != "marker_elements"} for x in manifest] == cases
    results = json.loads((full / "results.json").read_text())
    assert len(cases) == len(results) == 288
    assert [x["test_name"] for x in results] == [x["test_name"] for x in cases]
    profiles = list((full / "prof").rglob("op_summary_*.csv"))
    assert len(profiles) == 1
    tasks = sorted(read_csv(profiles[0]), key=lambda x: Decimal(x["Task Start Time(us)"].strip()))
    precision_file = raw / "precision/results.json"
    precision_results = json.loads(precision_file.read_text())
    precision = {x["test_name"]: x for x in precision_results}
    assert len(precision_results) == len(precision) == 288, "精度全量运行尚未完成"
    assert set(precision) == {x["test_name"] for x in cases}
    precision_exit = (raw / "precision/runner_exit_code.txt").read_text().strip()
    assert precision_exit == ("0" if all(x["status"] in ("PASSED", "UNSUPPORTED_SHAPE")
                                       for x in precision_results) else "1")
    precision_runtime = json.loads((raw / "precision/runtime.json").read_text())
    for key in ("flash_source_sha256", "case_source_sha256", "cannbotdsl", "torch", "torch_npu", "layout", "seed", "gate_seed"):
        assert precision_runtime[key] == runtime[key], key
    baseline = {x["test_name"]: x for x in read_csv(CANN_KDA)}
    samples, performance, paired, precision_rows = [], [], [], []
    for case, result in zip(cases, results):
        name = case["test_name"]
        cann = baseline[name]
        item = {k: case[k] for k in ("case_index", "test_name", "family", "B", "seqlen", "total_tokens", "nk", "nv", "dk", "dv", "has_g", "contiguous", "state_dtype")}
        item.update(flash_status=result["status"], flash_error=result["error"], flash_layout=result.get("layout", ""),
            batch_value_heads=int(case["B"]) * int(case["nv"]),
            flash_effective_seqlen=result.get("effective_seqlen", ""), flash_state_dtype="float32",
            flash_sequence_padding=result.get("sequence_padding", False), flash_state_contiguous_copy=result.get("state_contiguous_copy", False),
            flash_source_state_stride=json.dumps(result.get("source_state_stride", [])),
            cgdr_mean_us=case["optimized_mean_us"], cann_kda_mean_us=cann["kda_mean_us"],
            cann_kda_effective_seqlen=cann["kda_effective_seqlen"], cann_kda_layout=cann["kda_layout"],
            cgdr_precision_checked=case["precision_checked"], cann_kda_precision_checked=cann["kda_precision_checked"],
            cann_kda_precision_passed=cann["kda_precision_passed"],
            flash_mean_us="", flash_median_us="", flash_cv_pct="", flash_over_cgdr_ratio="", flash_over_cann_kda_ratio="",
            flash_precision_checked=False, flash_precision_passed="",
            flash_precision_status="UNSUPPORTED_SHAPE" if result["status"] == "UNSUPPORTED_SHAPE" else "NOT_CHECKED")
        if result["status"] == "PASSED":
            marker = str(result["marker_elements"])
            boundaries = [i for i, x in enumerate(tasks) if x["OP Type"] == "Add"
                          and x["Output Shapes"].strip('"') == marker]
            assert len(boundaries) == 2, (name, "marker count", len(boundaries))
            kernels = tasks[boundaries[0] + 1:boundaries[1]]
            assert len(kernels) == 6, (name, [(x["OP Type"], x["Op Name"]) for x in kernels])
            assert len({x["OP Type"] for x in kernels}) == 1
            assert all("kernel_flash_kda_kernel" in x["OP Type"] for x in kernels)
            assert all(int(x["Block Num"]) == result["block_num"] for x in kernels)
            assert all(float(x["Task Duration(us)"]) > 0 for x in kernels)
            for i, task in enumerate(kernels):
                samples.append(dict(test_name=name, sample_index=i, warmup=i == 0,
                    duration_us=float(task["Task Duration(us)"]), op_type=task["OP Type"],
                    op_name=task["Op Name"], task_id=task["Task ID"], start_us=task["Task Start Time(us)"].strip(),
                    aic_blocks=task["Block Num"], aiv_blocks=task["Mix Block Num"]))
            times = [float(x["Task Duration(us)"]) for x in kernels[1:]]
            perf = dict(test_name=name, mean_us=st.mean(times), median_us=st.median(times),
                min_us=min(times), max_us=max(times), stdev_us=st.stdev(times), cv_pct=100 * st.stdev(times) / st.mean(times),
                warmup_us=float(kernels[0]["Task Duration(us)"]), group=result["group"], dv_base=result["dv_base"],
                **{f"run_{i+1}_us": x for i, x in enumerate(times)})
            performance.append(perf)
            item.update(flash_mean_us=perf["mean_us"], flash_median_us=perf["median_us"], flash_cv_pct=perf["cv_pct"],
                flash_over_cgdr_ratio=perf["mean_us"] / float(item["cgdr_mean_us"]),
                flash_over_cann_kda_ratio=perf["mean_us"] / float(item["cann_kda_mean_us"]))
            checked = precision.get(name)
            if checked and checked["precision_checked"]:
                for key in ("input_shapes", "input_strides", "input_dtypes", "layout", "effective_seqlen", "lower_bound"):
                    assert checked[key] == result[key], (name, key)
                metrics = checked["precision_metrics"]
                assert (checked["status"] == "PASSED") == all(x["passed"] for x in metrics)
                assert all(x["relative_rms_limit"] == 0.02 and x["normalized_max_limit"] == 0.10 for x in metrics)
                if checked.get("precision_method") == "bitwise_equal_inputs_and_outputs_to_verified_case":
                    previous = precision[checked["reference_reused_from"]]
                    assert previous["precision_checked"] and previous["precision_metrics"] == metrics
                if checked["sequence_padding"]:
                    assert checked["padding_reference_checked"]
                item.update(flash_precision_checked=True, flash_precision_passed=checked["status"] == "PASSED",
                            flash_precision_status=checked["status"])
                precision_rows.append(dict(test_name=name, status=checked["status"],
                    precision_method=checked.get("precision_method", "cpu_fp64_recurrence"),
                    reference_reused_from=checked["reference_reused_from"], padding_reference_checked=checked["padding_reference_checked"],
                    **{f"{label}_{k}": v for label, metric in zip(("attention", "final_state"), metrics) for k, v in metric.items()}))
        paired.append(item)
    valid = [x for x in paired if x["flash_status"] == "PASSED"]
    subsets = [("ALL_MEASURED", valid), ("NO_SEQUENCE_PADDING", [x for x in valid if not x["flash_sequence_padding"]]),
               ("SEQUENCE_PADDING", [x for x in valid if x["flash_sequence_padding"]]),
               ("NO_PADDING_CONTIGUOUS_SOURCE", [x for x in valid if not x["flash_sequence_padding"] and x["contiguous"] == "True"]),
               ("NO_PADDING_CONTIGUOUS_FP32", [x for x in valid if not x["flash_sequence_padding"] and x["contiguous"] == "True" and x["state_dtype"] == "float32"])]
    for key in ("has_g", "state_dtype", "contiguous", "family"):
        for value in sorted({x[key] for x in valid}):
            subsets.append((f"{key}={value}", [x for x in valid if x[key] == value]))
    for lower, upper in [(0, 4), (4, 16), (16, 32), (32, float("inf"))]:
        rows = [x for x in valid if lower < x["batch_value_heads"] <= upper]
        if rows:
            subsets.append((f"{lower}<B*Nv<={upper}", rows))
    checked_pass = [x for x in valid if x["flash_precision_passed"] is True]
    if checked_pass:
        subsets.append(("PRECISION_PASSED", checked_pass))
    groups = [summary_group(name, rows) for name, rows in subsets]
    summary = dict(rdv_cases=288, measured_cases=len(valid), unsupported_cases=sum(x["flash_status"] == "UNSUPPORTED_SHAPE" for x in paired),
        sequence_padding_cases=sum(x["flash_sequence_padding"] for x in valid), contiguous_copy_cases=sum(x["flash_state_contiguous_copy"] for x in valid),
        kernel_samples=len(samples), measured_kernel_samples=len(valid) * 5, max_cv_pct=max(x["cv_pct"] for x in performance),
        cv_over_5pct=[x["test_name"] for x in performance if x["cv_pct"] > 5],
        precision_checked_cases=len(precision_rows), precision_passed_cases=sum(x["status"] == "PASSED" for x in precision_rows),
        precision_allclose_005_failed_cases=[x["test_name"] for x in precision_rows
            if not x["attention_allclose_005_passed"] or not x["final_state_allclose_005_passed"]],
        precision_failed_cases=[x["test_name"] for x in precision_rows if x["status"] != "PASSED"], groups=groups)
    write_csv(out / "comparison.csv", paired)
    write_csv(out / "case_performance.csv", performance)
    write_csv(out / "kernel_samples.csv", samples)
    write_csv(out / "group_summary.csv", groups)
    if precision_rows:
        write_csv(out / "precision_results.csv", precision_rows)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    shutil.copyfile(full / "runtime.json", out / "runtime.json")
    shutil.copyfile(full / "results.json", out / "execution_results.json")
    shutil.copyfile(precision_file, out / "precision_execution_results.json")
    shutil.copyfile(raw / "precision/runtime.json", out / "precision_runtime.json")
    for name in ("environment.json", "original_pytest.log", "env_check.json", "pip_freeze.txt"):
        source = raw / "environment" / name
        if source.exists():
            (out / name).write_text("\n".join(x.rstrip() for x in source.read_text().splitlines()) + "\n")
    validation = dict(rdv_rows=288, matched_baseline_manifest=True, unique_profile_marker_pairs=len(valid),
        exact_six_tasks_inside_each_marker_pair=True, positive_durations_and_blocks_checked=True,
        performance_precision_input_config_match=True, precision_execution_rows=len(precision_results),
        precision_runner_exit_code=int(precision_exit), performance_rows=len(performance), kernel_samples=len(samples),
        unsupported_durations_blank=all(x["flash_mean_us"] == "" for x in paired if x["flash_status"] != "PASSED"),
        precision_true_only_when_checked=all(x["flash_precision_checked"] for x in paired if x["flash_precision_passed"] is True))
    (out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    write_readme(out, summary, paired)
    sources = [CASES, CANN_KDA, profiles[0], full / "results.json", full / "case_manifest.json"]
    sources.extend([precision_file, raw / "precision/runtime.json", raw / "precision/runner_exit_code.txt"])
    scripts = [Path(__file__), Path(__file__).with_name("flash_kda_rdv_perf.py"),
               Path(__file__).with_name("kda_cpu_reference.cpp"), Path(__file__).with_name("kda_analyze_rdv.py")]
    metadata = dict(date="2026-09-09", metric="msprof Task Duration(us), one FlashKDA kernel per public call",
        sources=[dict(path=str(x), sha256=digest(x)) for x in sources],
        scripts={str(x.relative_to(REPO)): digest(x) for x in scripts},
        artifacts={str(x.relative_to(out)): digest(x) for x in out.rglob("*") if x.is_file() and x.name != "metadata.json"})
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "groups"}, indent=2))


if __name__ == "__main__":
    main()
