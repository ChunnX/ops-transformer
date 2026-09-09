"""严格匹配 KDA 的 msprof 样本并与已归档的新版 CGDR 比较。"""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics as st


REPO = Path(__file__).resolve().parents[4]
CGDR = REPO / "attention/chunk_gated_delta_rule/docs/rdv_perf/new_streaming/comparison.csv"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(label, rows):
    ratios = [float(x["kda_over_cgdr_ratio"]) for x in rows]
    kda = sum(float(x["kda_mean_us"]) for x in rows)
    cgdr = sum(float(x["cgdr_mean_us"]) for x in rows)
    return {"group": label, "cases": len(rows),
        "geomean_kda_over_cgdr": math.exp(st.mean(map(math.log, ratios))),
        "median_kda_over_cgdr": st.median(ratios), "min_ratio": min(ratios), "max_ratio": max(ratios),
        "sum_kda_mean_us": kda, "sum_cgdr_mean_us": cgdr, "sum_duration_ratio": kda / cgdr,
        "kda_faster_cases": sum(x < 1 for x in ratios), "cgdr_faster_cases": sum(x > 1 for x in ratios)}


def write_readme(out, summary, paired):
    groups = {x["group"]: x for x in summary["groups"]}
    same = groups["SAME_SHAPE_NO_SEQUENCE_PADDING"]
    precision_collection = (
        "本轮先完成 288 条精度扫描，再对统一尾块补齐所改变的 20 条 128×128 用例重新校验。\n"
        "最终精度表保留未改变输入的实测结果，并替换这 20 条复测结果；原始失败结果保留在原始输出目录中。"
        if summary["precision_collection"] == "full_scan_plus_changed_tail_retest" else
        "精度结果来自一次完整扫描。")
    table = []
    for key, label in [("SAME_SHAPE_NO_SEQUENCE_PADDING", "保持原始 shape"),
                       ("SAME_SHAPE_FP32_STATE", "原始 shape，双方 FP32 状态"),
                       ("SEQUENCE_PADDING", "中性补齐序列，单列"),
                       ("ALL_MEASURED_INCLUDING_PADDING", "全部可执行映射（含补齐）")]:
        g = groups[key]
        table.append(f"| {label} | {g['cases']} | {g['geomean_kda_over_cgdr']:.3f}× | {g['sum_duration_ratio']:.3f}× |")
    by_name = {x["test_name"]: x for x in paired}
    key_cases = []
    for name in ["ARC-001", "ARC-004", "ARC-009", "ARC-011", "ARC-017", "ARC-036", "ASC-019", "ASC-025", "ASC-028", "ASC-029", "ASC-034"]:
        x = by_name[name]
        key_cases.append(f"| {name} | {x['B']} / {x['seqlen']} / {x['nk']} / {x['nv']} / {x['dk']} / {x['dv']} | "
            f"{x['kda_layout']}，S={x['kda_effective_seqlen']} | {float(x['cgdr_mean_us']):.3f} | "
            f"{x['kda_mean_us']:.3f} | {x['kda_over_cgdr_ratio']:.3f}× |")
    text = f"""# KDA torch_extension 与新版 CGDR：RDV 对比（2026-09-09）

本报告替代此前直接调用 ACLNN 的结果。被测入口为 **`cann_ops_transformer.ops.chunk_kda_fwd`**，
底层仍为 **CANN 9.2.0 内置 KDA**；本轮没有重编译或替换 KDA kernel。
CGDR 使用 [new_streaming/comparison.csv](../rdv_perf/new_streaming/comparison.csv) 的 `optimized_mean_us`，没有重新测量 CGDR。

共保留 **288** 个原始 RDV 条目，**{summary['measured_cases']}** 个获得性能数据，
**{summary['unsupported_shape_cases']}** 个因维度约束不适用。
保持原始 shape 的 **{same['cases']}** 个配对，KDA/CGDR 主 kernel 耗时比几何平均为 **{same['geomean_kda_over_cgdr']:.3f}×**。
另有 **{summary['measured_sequence_padding_cases']}** 个条目使用中性序列补齐，单独统计。
精度实际检查 **{summary['precision_tested_cases']}** 例，通过 **{summary['precision_passed_cases']}** 例；
失败列表：`{summary['precision_failed_cases']}`。通过条目的 `kda_precision_passed` 已填写 `True`。

## 性能结果

比值为 `KDA / 新版 CGDR`，大于 1 表示 KDA 更慢。每例 1 次预热、5 次正式测量，单位 µs。
主表仅比较设备主 kernel 的 `Task Duration(us)`。

| 范围 | 用例数 | 几何平均耗时比 | 各 case 均值之和的比值 |
|---|---:|---:|---:|
{chr(10).join(table)}

最大 KDA CV 为 **{summary['max_kda_cv_pct']:.3f}%**。各 case 均值之和不是业务端到端延迟。
完整数据见 [comparison.csv](comparison.csv)，逐次样本见 [kernel_samples.csv](kernel_samples.csv)。

| 用例 | B / 原 S / Nk / Nv / Dk / Dv | KDA 实际布局和长度 | 新版 CGDR | KDA | 耗时比 |
|---|---|---|---:|---:|---:|
{chr(10).join(key_cases)}

## 输入和调用映射

- 保留 RDV 的 B、序列长度、Nk/Nv、Dk/Dv、BF16 Q/K/V/beta、Q/K 的 L2 归一化、`scale=Dk**-0.5` 和状态连续/非连续变体。
  seed=20260908；历史 CGDR 未保存完整输入快照，因此不声称逐字节回放了历史输入。
- KDA 门控为 `[T,Nv,Dk]`。独立 seed=20260909 生成目标 `-Uniform[0,1)` per-dim log-gate；无 gate 条目使用零 log-gate。
  **128×128 且有 gate** 时，计时外通过 `p=max(-gamma/5,2**-24)`、`raw=log(p)-log(1-p)` 生成 raw 输入，
  `A_log=0`、`dt_bias` 缺省、`lower_bound=-5`，算子内执行 safe-gate 激活。
  零概率端点的微小截断用于避免 -Inf，精度参考使用实际 raw 输入重新计算 gamma。
  **其他维度**直接传入已激活的 log-gate，关闭 `safe_gate` 和 `use_gate_in_kernel`。
- KDA 初态/最终状态接口为 FP32。原 BF16 初态先按原 dtype 生成，再在计时外提升到 FP32；
  NC 初态保持 `Dv+1` 的底层 stride。`state_dtype_matched` 明确标记是否与 CGDR 状态接口类型一致。
- 所有正式调用均保持 **chunk_size=64**、`output_final_state=True`、`state_v_first=True`，
  最少导出 attention、final_state、Aqk、Akk。KDA 的 Aqk/Akk 是额外必需输出，CGDR 不输出这两个矩阵。
- **128×128 且 B×Nv>32**：通过公开 **TND + `cu_seqlens_cpu`** 调用，规避 dense 路径的最终状态异常；
  元素数量、head 和 chunk size 均不变。其他条目用 BSND。实际配置逐行记录在 CSV 中。
- `ASC-025` 的四个状态变体将 S 从 **300 补齐到 320**，`ASC-034` 的四个变体从 **2000 补齐到 2048**。
  128×128 的 `ARC-017/018/034/035/036` 共 20 个条目也统一补齐到 64 的倍数：
  **6000→6016、5000→5056、65535→65536**。补充 token 的 Q/K/V/beta 为 0；
  已激活 log-gate 为 0，raw gate 为 -100（FP32 衰减因子为 1），保持有效输出和最终状态不变。
  CPU FP64 另外比较补齐与未补齐递推，验证两者的有效输出和最终状态相同。
  这些时延对应补齐后的实际 shape，不能作为原始 shape 的时延。
- `ASC-023/024/026/031` 的四种变体共 16 例，Dk/Dv 为 **50/50、5/8、1/1、31/31**，
  不满足 KDA `[16,256]` 内 16 倍数的要求。对应时延和精度通过字段留空。

## 精度验证

使用独立 **CPU FP64 逐 token 递推**，逐元素按 K 维衰减状态，再计算 Delta 更新和 attention。
CPU C++ 实现只并行独立 batch/head，已与 Python FP64 递推在 raw/已激活 gate、GQA、非等长序列上交叉验证。
全量检查每条用例的 attention 和最终状态。NC 变体复用 golden 前，检查全部 CPU 输入逐元素相同，
并分别比较各自的 NPU 输出；复用来源记录在 [precision_results.csv](precision_results.csv)。
{precision_collection}
来源文件、哈希和逐条映射见 [precision_provenance.json](precision_provenance.json)。

沿用 CGDR 泛化测试的 BF16 指标：`relative_rms <= 0.02` 且
`max(abs(error)/max(abs(ref),RMS(ref),2e-5)) <= 0.10`。
这是与数学参考比较的测试门限，不是 CANN 官方精度认证。
此前跨布局一致性测试的 `atol=0.002, rtol=0.02` 不宜直接当作整个长序列递推的判据；
本报告仍保留该旧逐元素门限的通过状态和最大误差比，便于审计，未因本轮结果调整门限。
旧逐元素门限另有 **{len(summary['precision_legacy_allclose_failed_cases'])}** 条未通过，详见
`summary.json` 的 `precision_legacy_allclose_failed_cases` 和精度表的 `legacy_allclose_passed`；
`kda_precision_passed=True` 表示通过上述 RMS/归一化最大误差判据。
CGDR 的历史 `precision_checked` 原样保留为 `cgdr_precision_checked`，未把“未检查”写成“失败”。

## 复测发现的运行问题与规避方式

1. **原始通用尾块仍会等待**：ASC-025、ASC-034 的原始 shape 在扩展调用后停在 NPU 同步，
   见 [tail_probes.json](tail_probes.json) 和 [诊断日志](diagnostics/ASC-025.log)。
   文档声明这些 S 合法，因此这是本机内置实现的运行问题，不能归类为公开 API 不支持。
   正式数据采用上述中性补齐，原始异常记录继续保留，超时不作为 kernel 耗时。
2. **通用维度 raw-gate 路径也观察到等待**：ASC-019 开启 raw safe gate 时复现；
   正式通用维度改用数学等价的预激活输入。不要把测试中使用 raw gate 的 128×128 路径直接推广到所有维度。
3. **dense 多 batch/head 的最终状态不稳定**：ARC-004（B=2、S=2048、Nv=32）重复调用时 attention 通过，
   最终状态出现远大于 BF16 舍入误差的变化，初态未被修改。
   切换 `state_v_first`、预激活 gate、导出 h 或全部中间量均曾复现，见
   [状态诊断](state_diagnostics.json)、[预激活重复诊断](state_pregate_diagnostics.json)。
   TND 路径对同一输入连续 10 次通过，见 [布局诊断](state_layout_diagnostics.json)，因此正式测试对 B×Nv>32 的 128×128 条目使用该路径。
4. **128 维尾块局部误差**：未补齐的 ARC-036 四个状态变体在 attention 上未通过固定门限，
   最大偏差位于最后一个 chunk 的 token 65475；BSND/TND、raw/预激活 gate 均复现，
   见 [尾块诊断](arc036_gate_diagnostics.json)。补齐一个 token 后，该例归一化最大误差从约 0.133 降到约 0.021，
   门限仍为 0.10。正式配置统一补齐所有尾块，共 28 条，并逐例验证补齐的数学等价性。

内核排查建议：重点检查 dense FwdH 从当前 batch/head 的最终状态写回，切换到下一项初态加载时，
MTE3→MTE2 对同一 UB 状态缓冲区的保护；源码 `ProcessAiv` 的最终写回后使用 MTE3→V，
而 `InitializeStateAiv` 随后从 MTE2 写入该缓冲区。这是基于复现与源码的排查线索，尚未通过修改内核证明根因。
通用尾块还应检查前后端共享 TPipe 的事件生命周期，以及 raw gate 与后续阶段的同步。
本轮通过公开接口参数规避，没有宣称修复了 CANN 内核。

## 扩展安装与数据来源

当前分支、更新后的上游 `4638b57da`、本机 torch_npu/op-plugin 均未找到 `chunk_kda_fwd` Python 绑定，
但仓库功能测试明确导入该接口。因此本轮按现有 `OpBuilder` + C++ 扩展规范补齐
[绑定源码](../../../chunk_kda_fwd/torch_extension/chunk_kda_fwd.py)，构建完整 `cann_ops_transformer` wheel，
安装到本轮独立目录并通过 PYTHONPATH 优先加载。仓库原有 **7 项 pytest 全部通过**，见 [日志](extension_pytest.log)。
测试脚本不使用 ctypes 或自行构造 ACLNN 调用；C++ 扩展使用仓库公共桥接层管理 stream、workspace 和张量。

环境为 Ascend950PR_9589（32 AIC/64 AIV）、torch 2.13.0+cpu、torch_npu 2.13.0.rc1、CANN 9.2.0。
CGDR 是历史归档数据，本轮未作交错 A/B 复测。已安装的 custom vendor 只含 CGDR，没有 KDA 覆盖；
库、wheel 和内置 KDA 文件哈希见 [environment.json](environment.json)，实际加载路径见 [loaded_libraries.txt](loaded_libraries.txt)。

| 文件 | 内容 |
|---|---|
| [comparison.csv](comparison.csv) | 288 条完整映射、实际配置、性能和精度状态 |
| [case_performance.csv](case_performance.csv) | 272 条均值/中位数/CV/5 个样本 |
| [kernel_samples.csv](kernel_samples.csv) | 272×6 条主 kernel 样本及同次调用辅助任务耗时 |
| [group_summary.csv](group_summary.csv) | 原始 shape、补齐、状态类型、布局等分组 |
| [precision_results.csv](precision_results.csv) | 独立精度指标、旧逐元素门限和参考复用来源 |
| [metadata.json](metadata.json) | 原始 profile/用例/脚本/产物哈希与测量口径 |
| [validation.json](validation.json) | 行数、样本数、精度和归档完整性检查 |

`kda_api_device_task_sum_mean_us` 统计同次扩展调用的主 kernel、Transpose 和 NC Slice 等设备任务之和，
不含 host 时间和任务间隙。Profile 实测 BSND 每次有 3 个 Transpose 和 1 个主 kernel，
TND 每次有 7 个 Transpose 和 1 个主 kernel；NC 初态各额外增加 1 个 Slice。
其中 TND 增加的 4 次转置将 Q/K/V/g 转为内核的 head-major 布局。
主 kernel 的比值也不能解释成 per-dim 门控单项开销：
门控激活、必需中间输出、状态 dtype、布局和内部流水线均参与差异。
本轮同时调整了 gate 和布局配置，相较旧报告的变化不能单独归因于 Python 调用接口。

复现命令（仓库根目录，使用新的输出目录）：

```bash
cd torch_extension
python3 -m pip install ninja build
python3 setup.py bdist_wheel
cd ..
kda_raw=attention/chunk_kda_fwd/tests/pytest/output/rdv_perf_torch_extension_repeat
python3 -m pip install --no-deps --target "$kda_raw/installed" torch_extension/dist/cann_ops_transformer-1.0.0-py3-none-any.whl
export PYTHONPATH="$PWD/$kda_raw/installed:$PYTHONPATH"
mkdir -p "$kda_raw/build"
python3 -m pytest -q attention/chunk_kda_fwd/tests/pytest/test_chunk_kda_fwd.py > "$kda_raw/build/pytest_binding.log" 2>&1
python3 attention/chunk_kda_fwd/tests/pytest/kda_rdv_perf.py --output "$kda_raw/precision" --mode precision --pad-sequence-tails
mkdir -p "$kda_raw/full/prof"
msprof --output="$PWD/$kda_raw/full/prof" --summary-format=csv --export=on python3 attention/chunk_kda_fwd/tests/pytest/kda_rdv_perf.py --output "$kda_raw/full" --pad-sequence-tails
printf '%s\\n' "$?" > "$kda_raw/full/tool_exit_code.txt"
python3 attention/chunk_kda_fwd/tests/pytest/kda_probe_tail.py --output "$kda_raw/tail_probes" --timeout 20
python3 attention/chunk_kda_fwd/tests/pytest/kda_analyze_rdv.py --raw "$kda_raw" --output attention/chunk_gated_delta_rule/docs/kda_perf_repeat
```

状态问题可用 `kda_diagnose_state.py --case ARC-004 --repeats 10 --output <path.json>` 复现。
旧版直接调用 ACLNN 的原始数据及被替换报告保留在本机 `output/rdv_perf_20260908/`，不混入本轮统计。
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
    for code in ["runner_exit_code.txt", "tool_exit_code.txt"]:
        assert (full / code).read_text().strip() == "0", code
    cases = read_csv(CGDR)
    assert len(cases) == 288 and len({x["test_name"] for x in cases}) == 288
    manifest = json.loads((full / "case_manifest.json").read_text())
    assert manifest == cases, "CGDR archive changed since KDA collection"
    runtime = json.loads((full / "runtime.json").read_text())
    assert runtime["case_source_sha256"] == digest(CGDR)
    assert (runtime["warmup"], runtime["measured"], runtime["layout"]) == (1, 5, "auto")
    results = json.loads((full / "results.json").read_text())
    assert [x["test_name"] for x in results] == [x["test_name"] for x in cases]
    results_by_name = {x["test_name"]: x for x in results}
    passed = [x for x in results if x["status"] == "PASSED"]
    summaries = list((full / "prof").rglob("op_summary_*.csv"))
    assert len(summaries) == 1
    task_rows = read_csv(summaries[0])
    tasks = sorted([x for x in task_rows if x["Op Name"].startswith("aclnnChunkKdaFwd_")],
                   key=lambda x: Decimal(x["Task Start Time(us)"].strip()))
    main_indices = [i for i, x in enumerate(tasks) if x["OP Type"] == "ChunkKdaFwd"]
    assert len(main_indices) == len(passed) * 6
    groups, start = [], 0
    for index in main_indices:
        # 本次 state_v_first=True，主kernel之后由一次Transpose导出最终状态。
        assert tasks[index + 1]["OP Type"] == "Transpose"
        group = tasks[start:index + 2]
        assert sum(x["OP Type"] == "ChunkKdaFwd" for x in group) == 1
        assert all(float(x["Task Duration(us)"]) > 0 for x in group)
        groups.append((tasks[index], group))
        start = index + 2
    assert start == len(tasks), "Unassigned KDA API tasks"
    samples, performance = [], []
    cgdr_by_name = {x["test_name"]: x for x in cases}
    for index, result in enumerate(passed):
        name = result["test_name"]
        case = cgdr_by_name[name]
        b, length, nk, nv, dk, dv = (int(case[k]) for k in ["B", "seqlen", "nk", "nv", "dk", "dv"])
        length = result["effective_lengths"][0]
        group = groups[index * 6:index * 6 + 6]
        for sample_index, (task, api_tasks) in enumerate(group):
            raw_gate = case["has_g"] == "True" and (dk, dv) == (128, 128)
            tnd = result["effective_layout"] == "TND"
            kb, kl = (1, b * length) if tnd else (b, length)
            # TND 公共入口会将 Q/K/V/g 转为 head-major；dense 快路径保留 sequence-major。
            def shape(heads, dim):
                return f"{kb},{heads},{kl},{dim}" if tnd else f"{kb},{kl},{heads},{dim}"
            expected_input_shapes = [shape(nk, dk), shape(nk, dk), shape(nv, dv),
                                     shape(nv, dk), f"{kb},{nv},{kl}"]
            if raw_gate:
                expected_input_shapes.append(str(nv))
            expected_input_shapes.append(f"{b},{nv},{dk},{dv}")
            if tnd:
                expected_input_shapes.append(str(b + 1))
            assert task["Input Shapes"].strip('"').split(";") == expected_input_shapes
            assert task["Input Data Types"].split(";") == [
                "DT_BF16", "DT_BF16", "DT_BF16", "FLOAT", "DT_BF16"] + ["FLOAT"] * (2 if raw_gate else 1) + (["INT64"] if tnd else [])
            shapes = task["Output Shapes"].strip('"').split(";")
            assert shapes[0] == (f"1,{b * length},{nv},{dv}" if tnd else str(b * length * nv * dv))
            assert shapes[1] == f"{b},{nv},{dk},{dv}"
            assert shapes[3:5] == ([f"1,{nv},{b * length},64"] if tnd else [str(b * length * nv * 64)]) * 2
            assert int(task["Block Num"]) == min(32, b * nv)
            assert len(api_tasks) == (8 if tnd else 4) + (case["contiguous"] == "False")
            samples.append({"case_index": case["case_index"], "test_name": name, "sample_index": sample_index,
                "warmup": sample_index == 0, "kernel_duration_us": float(task["Task Duration(us)"]),
                "api_device_task_sum_us": sum(float(x["Task Duration(us)"]) for x in api_tasks),
                "api_task_count": len(api_tasks), "task_id": task["Task ID"],
                "task_start_us": task["Task Start Time(us)"].strip(), "aic_blocks": task["Block Num"],
                "aiv_blocks": task["Mix Block Num"]})
        measured = samples[-5:]
        times = [x["kernel_duration_us"] for x in measured]
        performance.append({"case_index": case["case_index"], "test_name": name,
            "warmup_us": samples[-6]["kernel_duration_us"], "mean_us": st.mean(times),
            "median_us": st.median(times), "min_us": min(times), "max_us": max(times),
            "stdev_us": st.stdev(times), "cv_pct": 100 * st.stdev(times) / st.mean(times),
            "api_device_task_sum_mean_us": st.mean(x["api_device_task_sum_us"] for x in measured),
            "effective_seqlen": length, "effective_layout": result["effective_layout"], "sequence_padding": result["sequence_padding"], "aic_blocks": min(32, b * nv),
            **{f"run_{i + 1}_us": x for i, x in enumerate(times)}})
    performance_by_name = {x["test_name"]: x for x in performance}
    precision_raw = json.loads((raw / "precision/results.json").read_text())
    assert len(precision_raw) == len(cases)
    precision = {x["test_name"]: x for x in precision_raw}
    assert len(precision) == len(cases)
    precision_runtime = json.loads((raw / "precision/runtime.json").read_text())
    assert precision_runtime["case_source_sha256"] == digest(CGDR)
    assert precision_runtime["layout_policy"] == runtime["layout_policy"]
    assert precision_runtime["pad_sequence_tails"] == runtime["pad_sequence_tails"]
    for name, result in results_by_name.items():
        checked = precision[name]
        if result["status"] == "PASSED":
            assert checked["precision_checked"]
            assert all(x["relative_rms_limit"] == 0.02 and x["normalized_max_limit"] == 0.10
                       for x in checked["precision_metrics"])
            assert (checked["status"] == "PASSED") == all(x["passed"] for x in checked["precision_metrics"])
            for field in ["effective_lengths", "effective_layout", "sequence_padding", "input_shapes", "input_strides"]:
                assert checked[field] == result[field], (name, field)
            if result["sequence_padding"]:
                assert checked["padding_reference_checked"]
    probes_raw = json.loads((raw / "tail_probes/results.json").read_text())
    probes = {x["test_name"]: x for x in probes_raw}
    assert len(probes) == 8
    assert {x["test_name"] for x in precision_raw} == {x["test_name"] for x in cases}
    paired = []
    for case in cases:
        name = case["test_name"]
        result, perf, prec = results_by_name[name], performance_by_name.get(name), precision.get(name)
        status = result["status"]
        if status == "DEFERRED_RUNTIME_PROBE":
            assert name in probes
            status = "TIMEOUT"
        row = {k: case[k] for k in ["case_index", "test_name", "family", "B", "seqlen", "total_tokens",
                                    "nk", "nv", "dk", "dv", "has_g", "contiguous", "seed"]}
        row.update(cgdr_state_dtype=case["state_dtype"], kda_state_dtype="float32",
            state_dtype_matched=case["state_dtype"] == "float32", cgdr_layout="TND", kda_layout=result.get("effective_layout", ""),
            kda_gate_mode=("inverse_safe_gate_raw_input" if case["has_g"] == "True" and (case["dk"], case["dv"]) == ("128", "128")
                           else "per_dim_log_gate" if case["has_g"] == "True" else "zero_log_gate"),
            kda_safe_gate=(case["dk"], case["dv"]) == ("128", "128"),
            kda_use_gate_in_kernel=case["has_g"] == "True" and (case["dk"], case["dv"]) == ("128", "128"),
            kda_sequence_padding=result.get("sequence_padding", False),
            kda_effective_seqlen=result.get("effective_lengths", [""])[0],
            original_shape_probe_status=probes[name]["status"] if name in probes else "",
            kda_status=status, cgdr_mean_us=case["optimized_mean_us"], cgdr_median_us=case["optimized_median_us"],
            cgdr_cv_pct=case["optimized_cv_pct"], kda_mean_us=perf["mean_us"] if perf else "",
            kda_median_us=perf["median_us"] if perf else "", kda_cv_pct=perf["cv_pct"] if perf else "",
            kda_over_cgdr_ratio=perf["mean_us"] / float(case["optimized_mean_us"]) if perf else "",
            kda_duration_increase_pct=100 * (perf["mean_us"] / float(case["optimized_mean_us"]) - 1) if perf else "",
            kda_api_device_task_sum_mean_us=perf["api_device_task_sum_mean_us"] if perf else "",

            cgdr_precision_checked=case["precision_checked"],
            kda_precision_checked=prec["precision_checked"] if prec else False,
            kda_precision_status=prec["status"] if prec else "NOT_CHECKED",
            kda_precision_passed=prec["status"] == "PASSED" if prec and prec["precision_checked"] else "",
            reason=("Neutral sequence padding to multiple of 64; tail runtime/numerical workaround"
                    if result.get("sequence_padding") else result["error"]))
        paired.append(row)
    valid = [x for x in paired if x["kda_status"] == "PASSED"]
    groups_summary = [summarize("ALL_MEASURED_INCLUDING_PADDING", valid),
                      summarize("SAME_SHAPE_NO_SEQUENCE_PADDING", [x for x in valid if not x["kda_sequence_padding"]]),
                      summarize("SEQUENCE_PADDING", [x for x in valid if x["kda_sequence_padding"]])]
    groups_summary.append(summarize("SAME_SHAPE_FP32_STATE", [x for x in valid
        if not x["kda_sequence_padding"] and x["cgdr_state_dtype"] == "float32"]))
    for field in ["cgdr_state_dtype", "family", "has_g", "contiguous", "kda_layout"]:
        for value in sorted({x[field] for x in valid}):
            groups_summary.append(summarize(f"{field}={value}", [x for x in valid if x[field] == value]))
    for dk, dv in sorted({(int(x["dk"]), int(x["dv"])) for x in valid}):
        groups_summary.append(summarize(f"Dk={dk},Dv={dv}", [x for x in valid if (int(x["dk"]), int(x["dv"])) == (dk, dv)]))
    groups_summary.append(summarize("KDA_PRECISION_PASSED", [x for x in valid if x["kda_precision_status"] == "PASSED"]))
    precision_rows = []
    for item in precision_raw:
        if not item["precision_checked"]:
            continue
        metrics = item["precision_metrics"]
        precision_rows.append({"test_name": item["test_name"], "status": item["status"],
            "sequence_padding": item["sequence_padding"], "effective_seqlen": item["effective_lengths"][0],
            "reference_reused_from": item["reference_reused_from"],
            "precision_source": item.get("precision_source", "precision/results.json"),
            "padding_reference_checked": item.get("padding_reference_checked", False),
            **{f"{label}_{key}": value for label, metric in zip(["attention", "final_state"], metrics)
               for key, value in metric.items()}})
    write_csv(out / "comparison.csv", paired)
    write_csv(out / "case_performance.csv", performance)
    write_csv(out / "kernel_samples.csv", samples)
    write_csv(out / "group_summary.csv", groups_summary)
    write_csv(out / "precision_results.csv", precision_rows)
    summary = {"rdv_cases": len(cases), "measured_cases": len(valid), "unsupported_shape_cases": sum(x["kda_status"] == "UNSUPPORTED_SHAPE" for x in paired),
        "measured_sequence_padding_cases": sum(x["kda_sequence_padding"] for x in valid),
        "original_shape_timeout_cases": sum(x["status"] == "TIMEOUT" for x in probes.values()),
        "timeout_cases_in_measured_run": sum(x["kda_status"] == "TIMEOUT" for x in paired), "kernel_records": len(samples), "measured_kernel_records": len(valid) * 5,
        "api_device_task_records": len(tasks), "overall": groups_summary[0], "groups": groups_summary,
        "max_kda_cv_pct": max(x["cv_pct"] for x in performance),
        "kda_cv_over_5pct": [x["test_name"] for x in performance if x["cv_pct"] > 5],
        "precision_tested_cases": len(precision_rows),
        "precision_collection": precision_runtime.get("precision_collection", "full_single_scan"),
        "precision_case_wall_seconds_sum": sum(x.get("elapsed_wall_seconds", 0) for x in precision_raw),
        "precision_passed_cases": sum(x["status"] == "PASSED" for x in precision_rows),
        "precision_failed_cases": [x["test_name"] for x in precision_rows if x["status"] != "PASSED"],
        "precision_legacy_allclose_failed_cases": [x["test_name"] for x in precision_rows
            if not x["attention_legacy_allclose_passed"] or not x["final_state_legacy_allclose_passed"]],
        "top_ratios": sorted(valid, key=lambda x: x["kda_over_cgdr_ratio"], reverse=True)[:10],
        "lowest_ratios": sorted(valid, key=lambda x: x["kda_over_cgdr_ratio"])[:10]}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    for source, dest in [(full / "runtime.json", "runtime.json"),
                         (full / "loaded_libraries.txt", "loaded_libraries.txt"),
                         (raw / "tail_probes/results.json", "tail_probes.json")]:
        shutil.copyfile(source, out / dest)
    if (raw / "environment.json").exists():
        shutil.copyfile(raw / "environment.json", out / "environment.json")
    else:
        (out / "environment.json").write_text(json.dumps({"runtime": runtime,
            "note": "This run did not collect a separate package hash snapshot"}, indent=2) + "\n")
    if (raw / "support_probe.json").exists():
        shutil.copyfile(raw / "support_probe.json", out / "support_probe.json")
    diagnostics = out / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    for name in probes:
        shutil.copyfile(raw / "tail_probes" / (name + ".log"), diagnostics / (name + ".log"))
    for name in ["state_diagnostics.json", "state_pregate_diagnostics.json", "state_layout_diagnostics.json",
                 "arc036_gate_diagnostics.json", "precision_provenance.json"]:
        if (raw / name).exists():
            shutil.copyfile(raw / name, out / name)
    for source, dest in [("build/pytest_binding.log", "extension_pytest.log"),
                         ("build/cpu_reference.log", "cpu_reference_crosscheck.log")]:
        if (raw / source).exists():
            # 编译器命令常带行尾空格；归档副本仅清理行尾，原始日志及哈希继续保留。
            (out / dest).write_text("\n".join(x.rstrip() for x in (raw / source).read_text().splitlines()) + "\n")
    if not (out / "precision_provenance.json").exists():
        source = raw / "precision/results.json"
        (out / "precision_provenance.json").write_text(json.dumps({
            "kind": "full_single_scan", "sources": [{"path": str(source), "sha256": digest(source)}],
            "case_sources": [{"test_name": x["test_name"], "source": "precision/results.json"}
                             for x in precision_raw]}, indent=2) + "\n")
    validation = {
        "baseline_manifest_exact_match": True,
        "performance_precision_inputs_and_layouts_match": True,
        "kernel_shapes_dtypes_blocks_and_task_order_checked": True,
        "rdv_rows": len(paired), "performance_rows": len(performance),
        "kernel_samples": len(samples), "samples_per_measured_case": 6,
        "precision_checked_rows": len(precision_rows),
        "precision_passed_rows": summary["precision_passed_cases"],
        "precision_failed_rows": len(summary["precision_failed_cases"]),
        "padding_equivalence_checked_rows": sum(x.get("padding_reference_checked", False) for x in precision_raw),
        "unmeasured_durations_blank": all(x["kda_mean_us"] == "" for x in paired if x["kda_status"] != "PASSED"),
        "unsupported_precision_not_marked_passed": all(x["kda_precision_passed"] == "" for x in paired if x["kda_status"] == "UNSUPPORTED_SHAPE"),
    }
    (out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    write_readme(out, summary, paired)
    sources = [CGDR, summaries[0], full / "case_manifest.json", full / "results.json",
               raw / "precision/results.json", raw / "tail_probes/results.json"]
    sources.extend(raw / name for name in ["build/pytest_binding.log", "build/cpu_reference.log"]
                   if (raw / name).exists())
    if (raw / "support_probe.json").exists():
        sources.append(raw / "support_probe.json")
    metadata = {"date": "2026-09-09", "interface": "cann_ops_transformer.ops.chunk_kda_fwd", "operator": "CANN 9.2.0 built-in ChunkKdaFwd",
        "cgdr_version": "6f599eb3aa317ac1f737def7b1eaaf51c1551d15",
        "metric": "msprof main operator Task Duration(us)", "unit": "microseconds",
        "ratio_definition": "kda_mean_us / cgdr_mean_us; >1 means KDA is slower",
        "sampling": {"warmup_per_case": 1, "measured_per_case": 5, "cv_ddof": 1},
        "log_formatting": "Archived build/pytest log copies have trailing whitespace removed; original logs are hashed in sources",
        "sources": [{"path": str(x), "sha256": digest(x)} for x in sources],
        "scripts": {str(x.relative_to(REPO)): digest(x) for x in Path(__file__).parent.glob("kda_*") if x.suffix in (".py", ".cpp")},
        "comparison_caveats": ["CGDR timings are the historical archived full run, not a same-session retest",
            "KDA uses independent per-dimension gate; its mathematical operator differs from CGDR",
            "28 tail cases measured with neutral sequence padding, tracked separately",
            "raw safe gate activation runs inside KDA for gated 128x128; generic dimensions use preactivated gate",
            "KDA state is FP32; BF16 CGDR initial-state values were promoted outside the measured operator",
            "KDA uses TND+host cu_seqlens for 128x128 with B*Nv>32 to avoid nondeterministic dense final-state errors; BSND otherwise",
            "main-kernel comparison excludes auxiliary device tasks and host overhead",
            "KDA must also write Aqk/Akk, while CGDR returns attention/finalState only",
            "KDA API device-task sum includes Transpose and NC Slice, but excludes host time and task gaps",
            "Precision status is from full CPU FP64 recurrence with CGDR generalization RMS/max criteria"],
        "artifacts": {str(x.relative_to(out)): digest(x) for x in out.rglob("*") if x.is_file() and x.name != "metadata.json"}}
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ["groups", "top_ratios", "lowest_ratios"]}, indent=2))


if __name__ == "__main__":
    main()
