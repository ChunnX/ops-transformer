# FlashAttention 性能测试方案

## 一、目标

在 `pytests/` 内新增 `--perf_mode` 性能测试模式，通过 msprof 采集 NPU 算子耗时，
复用现有 `test_cases/*.py` 用例体系，所有改动限于 `pytests/` 目录内。

---

## 二、整体架构

```
pytests/
├── test_flash_attn.py              # 主入口，新增 --perf_mode 分支
├── utils/
│   ├── perf_runner.py              # [新增] 生成采集脚本 + 调用 msprof
│   └── perf_parser.py              # [新增] 解析 op_summary.csv + 冷热统计
└── test_cases/
    └── performance_redline_infer.py  # 已有，直接复用
```

**原则**：不改动现有精度测试路径，`--perf_mode` 为独立分支。

---

## 三、命令行参数

在 `test_flash_attn.py` 的 argparse 中新增：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--perf_mode` | flag | 关闭 | 启用性能测试 |
| `--perf_runs` | int | `5` | 每个配置重复次数 |
| `--perf_cold_thr` | int | `16` | S1 ≤ 此值为冷启动 |
| `--perf_output` | str | `./perf_output` | 输出目录 |
| `--skip_accuracy` | flag | 关闭 | 跳过精度验证 |

**典型用法**：

```bash
# 全部性能用例
python test_flash_attn.py \
    --case_files performance_redline_infer \
    --case_id all \
    --perf_mode --perf_runs 5 \
    --skip_accuracy

# 指定用例
python test_flash_attn.py \
    --case_files performance_redline_infer \
    --case_id Decode_FP16_FP16_L0_8_8_8_1_8192_128_BNSD_000002 \
    --perf_mode --perf_runs 10
```

---

## 四、冷热分离策略

### 热启动（S1 > cold_thr）

数据生成 1 次，复用 runs 次。统计：丢弃第 1 次，取后 runs-1 次平均。

适用：Prefill、长序列 Decode（S1 > 16）

### 冷启动（S1 ≤ cold_thr）

每次迭代重建 Q/K/V + 两个 `(32,32768,1,128)` randn 刷 HBM（各 ~256MB）。统计：去最大最小值，取中间平均。

适用：Decode 阶段（S1=1），短序列

---

## 五、核心模块设计

### 5.1 `utils/perf_runner.py` — 采集脚本生成 + msprof 执行

生成 Python 脚本 → `msprof --aic-mode=task-based` 执行 → 拷贝 `op_summary.csv`。

```python
"""性能采集：生成 sweep 脚本，调用 msprof，返回 op_summary.csv 路径。"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def build_perf_script(cases: List[Dict], runs: int, cold_thr: int,
                      device_id: int = 0) -> str:
    """生成采集脚本，冷热统一，一次 msprof 采集。"""
    hot = [c for c in cases if c.get("S1", 9999) > cold_thr]
    cold = [c for c in cases if c.get("S1", 9999) <= cold_thr]

    lines = [
        "import torch, torch_npu",
        f"torch.npu.set_device({device_id})",
        "from npu_ops_transformer.ops import npu_flash_attn, npu_flash_attn_metadata",
        "device = torch.device('npu:0')",
        f"print(f'total: {len(cases)}  hot: {len(hot)}  cold: {len(cold)}  runs: {runs}')",
        "",
    ]

    for i, c in enumerate(hot):
        lines.extend(_gen_hot(c, i, runs, len(hot)))
    for i, c in enumerate(cold):
        lines.extend(_gen_cold(c, i, runs, len(cold)))

    lines.append("print('done')")
    return "\n".join(lines)


def _pt_dtype(ds: str) -> str:
    return "torch.float16" if ds == "fp16" else "torch.bfloat16"


def _extract(c: Dict):
    b = c.get("B", 1); n1 = c["N1"]; n2 = c.get("N2", n1)
    s1 = c.get("S1", 1); s2 = c.get("S2", s1); d = c["D"]
    lq = c.get("layout_q", c.get("input_layout", "BNSD"))
    lkv = c.get("layout_kv", lq); lo = c.get("layout_out", lq)
    mm = c.get("mask_mode", 0); dt = _pt_dtype(c.get("Dtype", "fp16"))
    return b, n1, n2, s1, s2, d, lq, lkv, lo, mm, dt


def _meta(n1, n2, d, b, s1, s2, mm, lq, lkv, lo) -> List[str]:
    return [
        "metadata = npu_flash_attn_metadata(",
        f"    num_heads_q={n1}, num_heads_kv={n2}, head_dim={d},",
        f"    batch_size={b}, max_seqlen_q={s1}, max_seqlen_kv={s2},",
        f"    mask_mode={mm}, win_left=-1, win_right=-1,",
        f"    layout_q='{lq}', layout_kv='{lkv}', layout_out='{lo}')",
    ]


def _kernel(d, mm, lq, lkv, lo) -> List[str]:
    return [
        f"    npu_flash_attn(q, k, v, softmax_scale=1/({d}**0.5),",
        f"        mask_mode={mm}, metadata=metadata,",
        f"        layout_q='{lq}', layout_kv='{lkv}', layout_out='{lo}')",
    ]


def _gen_hot(c, idx, runs, total) -> List[str]:
    b, n1, n2, s1, s2, d, lq, lkv, lo, mm, dt = _extract(c)
    return [
        f"# hot {idx}: B={b} N1={n1} N2={n2} S1={s1} S2={s2} D={d}",
        f"q = torch.randn({b}, {n1}, {s1}, {d}, dtype={dt}, device=device)",
        f"k = torch.randn({b}, {n2}, {s2}, {d}, dtype={dt}, device=device)",
        f"v = torch.randn({b}, {n2}, {s2}, {d}, dtype={dt}, device=device)",
        *_meta(n1, n2, d, b, s1, s2, mm, lq, lkv, lo),
        "torch.npu.synchronize()",
        f"for _ in range({runs}):",
        "    torch.npu.synchronize()",
        *_kernel(d, mm, lq, lkv, lo),
        "torch.npu.synchronize()",
        f"print(f'  [hot] {idx+1}/{total}')", "",
    ]


def _gen_cold(c, idx, runs, total) -> List[str]:
    b, n1, n2, s1, s2, d, lq, lkv, lo, mm, dt = _extract(c)
    return [
        f"# cold {idx}: B={b} N1={n1} N2={n2} S1={s1} S2={s2} D={d}",
        *_meta(n1, n2, d, b, s1, s2, mm, lq, lkv, lo),
        "torch.npu.synchronize()",
        f"for _ in range({runs}):",
        f"    q = torch.randn({b}, {n1}, {s1}, {d}, dtype={dt}, device=device)",
        f"    k = torch.randn({b}, {n2}, {s2}, {d}, dtype={dt}, device=device)",
        f"    v = torch.randn({b}, {n2}, {s2}, {d}, dtype={dt}, device=device)",
        f"    _f1 = torch.randn(32, 32768, 1, 128, dtype={dt}, device=device)",
        f"    _f2 = torch.randn(32, 32768, 1, 128, dtype={dt}, device=device)",
        "    torch.npu.synchronize()",
        *_kernel(d, mm, lq, lkv, lo),
        "torch.npu.synchronize()",
        "torch.npu.empty_cache()",
        f"print(f'  [cold] {idx+1}/{total}')", "",
    ]


def run_msprof(script: str, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "_perf_sweep.py").write_text(script)

    msprof_dir = output_dir / "msprof_output"
    if msprof_dir.exists():
        shutil.rmtree(msprof_dir)

    subprocess.run(
        ["msprof", f"--output={msprof_dir}", "--aic-mode=task-based",
         sys.executable, str(output_dir / "_perf_sweep.py")], check=True)

    prof_dirs = sorted(msprof_dir.glob("PROF_*"), reverse=True)
    csv_files = sorted(prof_dirs[0].glob(
        "mindstudio_profiler_output/op_summary_*.csv"))
    dst = output_dir / "op_summary.csv"
    shutil.copy(csv_files[0], dst)
    print(f"[msprof] op_summary -> {dst}")
    return dst
```

### 5.2 `utils/perf_parser.py` — op_summary 解析 + 冷热统计

```python
"""解析 op_summary.csv，按用例分组，冷热分离统计。"""

import csv
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_op_summary(csv_path: str, operator_name: str = "FlashAttn"
                     ) -> List[Tuple[float, float]]:
    """返回 [(start_time_us, duration_us), ...] 按时间升序。"""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    entries = [
        (float(r["Task Start Time(us)"]), float(r["Task Duration(us)"]))
        for r in rows if r["OP Type"] == operator_name
    ]
    entries.sort(key=lambda x: x[0])
    return entries


def compute_stats(entries: List[Tuple[float, float]],
                  cases: List[Dict], runs: int, cold_thr: int
                  ) -> Dict[str, Dict]:
    """
    热启动：丢弃第 1 次，取后 runs-1 次平均
    冷启动：去最大最小值，取中间平均

    Returns: {case_name: {"avg_us", "min_us", "max_us", "mode"}}
    """
    hot_cases = [c for c in cases if c.get("S1", 9999) > cold_thr]
    cold_cases = [c for c in cases if c.get("S1", 9999) <= cold_thr]

    expected = []
    for c in hot_cases:
        for _ in range(runs): expected.append((c, "hot"))
    for c in cold_cases:
        for _ in range(runs): expected.append((c, "cold"))

    if len(entries) > len(expected) and expected:
        r = len(entries) // len(expected)
        if r > 1:
            print(f"[perf] entries={len(entries)} expected={len(expected)} ratio={r}")
            entries = [entries[i * r] for i in range(len(expected))]

    n = min(len(entries), len(expected))
    raw = defaultdict(lambda: {"hot": [], "cold": []})

    for i in range(n):
        c, mode = expected[i]
        raw[_case_key(c)][mode].append(entries[i][1])

    result = {}
    for key, d in raw.items():
        if d["hot"]:
            keep = d["hot"][1:]
            avg = sum(keep) / len(keep) if keep else 0
            mode = "hot"
        elif d["cold"]:
            vals = sorted(d["cold"])
            keep = vals[1:-1] if len(vals) >= 3 else vals
            avg = sum(keep) / len(keep) if keep else 0
            mode = "cold"
        else:
            continue
        all_durs = d.get("hot", []) + d.get("cold", [])
        result[key] = {
            "avg_us": round(avg, 2),
            "min_us": round(min(all_durs), 2) if all_durs else 0,
            "max_us": round(max(all_durs), 2) if all_durs else 0,
            "mode": mode,
        }
    return result


def _case_key(c: Dict) -> str:
    parts = []
    for k in ("B","N1","N2","S1","S2","D","Dtype","layout_q","layout_kv","mask_mode"):
        if k in c: parts.append(f"{k}={c[k]}")
    return "_".join(parts)
```

---

## 六、主流程改造（`test_flash_attn.py`）

### 6.1 argparse 新增

```python
parser.add_argument("--perf_mode",     action="store_true")
parser.add_argument("--perf_runs",     type=int,  default=5)
parser.add_argument("--perf_cold_thr", type=int,  default=16)
parser.add_argument("--perf_output",   type=str,  default="./perf_output")
parser.add_argument("--skip_accuracy", action="store_true")
```

### 6.2 main 中 perf_mode 分支

在精度测试循环之前插入：

```python
if args.perf_mode:
    from pathlib import Path
    from utils.perf_runner import build_perf_script, run_msprof
    from utils.perf_parser import parse_op_summary, compute_stats, _case_key

    cases_list = [normalize_case(all_cases[name]) for name in run_cases]
    out = Path(args.perf_output)

    script = build_perf_script(cases_list, args.perf_runs, args.perf_cold_thr)
    csv_path = run_msprof(script, out)

    entries = parse_op_summary(str(csv_path))
    stats = compute_stats(entries, cases_list, args.perf_runs, args.perf_cold_thr)

    SEP = "─" * 100
    print(f"\n┌{SEP}┐")
    print(f"│  性能测试  ({len(stats)} cases)")
    print(f"├{SEP}┤")
    print(f"│  {'Case':<55}  {'Avg(us)':>9}  {'Min(us)':>9}  {'Max(us)':>9}  {'Mode':>5}  │")
    print(f"├{SEP}┤")

    for name in run_cases:
        c = normalize_case(all_cases[name])
        s = stats.get(_case_key(c))
        if not s: continue
        short = name[:55] if len(name) <= 55 else name[:52] + "..."
        print(f"│  {short:<55}  {s['avg_us']:>9.2f}  {s['min_us']:>9.2f}  "
              f"{s['max_us']:>9.2f}  {s['mode']:>5}  │")

    print(f"└{SEP}┘")

    import csv as csv_mod
    perf_csv = out / "perf.csv"
    with open(perf_csv, "w", newline="") as f:
        w = csv_mod.DictWriter(f, ["case","avg_us","min_us","max_us","mode"])
        w.writeheader()
        for name in run_cases:
            s = stats.get(_case_key(normalize_case(all_cases[name])))
            if not s: continue
            w.writerow(dict(case=name, avg_us=s["avg_us"],
                           min_us=s["min_us"], max_us=s["max_us"], mode=s["mode"]))
    print(f"[perf] {perf_csv}")
    sys.exit(0)
```

---

## 七、输出文件

```
perf_output/
├── op_summary.csv       # msprof 原始数据
├── perf.csv             # case, avg_us, min_us, max_us, mode
└── _perf_sweep.py       # 采集脚本（可复现）
```

---

## 八、实现计划

| 步骤 | 文件 | 行数 | 说明 |
|---|---|---|---|
| 1 | `utils/perf_runner.py` | ~110 | 脚本生成 + msprof 调用 |
| 2 | `utils/perf_parser.py` | ~75 | CSV 解析 + 冷热统计 |
| 3 | `test_flash_attn.py` | ~50 | argparse + perf_mode 分支 |

---

## 九、Kernel 崩溃容错设计

NPU kernel 可能因非法参数、OOM 等原因崩溃，导致整个扫描进程终止。
需要在三层防护：

### 9.1 采集脚本层：逐 case 隔离崩溃

每个 case 用 `try/except` 包裹，一个 case 崩溃不影响后续 case：

```python
# 生成的脚本模板改动（热启动为例）
# _gen_hot 新增 per-case try/except：

def _gen_hot(c, idx, runs, total) -> List[str]:
    b, n1, n2, s1, s2, d, lq, lkv, lo, mm, dt = _extract(c)
    case_tag = f"hot{idx}"  # 用于日志匹配
    return [
        f"# hot {idx}: B={b} N1={n1} N2={n2} S1={s1} S2={s2} D={d}",
        f"q = torch.randn({b}, {n1}, {s1}, {d}, dtype={dt}, device=device)",
        f"k = torch.randn({b}, {n2}, {s2}, {d}, dtype={dt}, device=device)",
        f"v = torch.randn({b}, {n2}, {s2}, {d}, dtype={dt}, device=device)",
        *_meta(n1, n2, d, b, s1, s2, mm, lq, lkv, lo),
        "torch.npu.synchronize()",
        f"print(f'[CASE_START] {case_tag}')",
        f"try:",
        f"    for _ in range({runs}):",
        "        torch.npu.synchronize()",
        *_kernel(d, mm, lq, lkv, lo),
        f"    torch.npu.synchronize()",
        f"    print(f'[CASE_END] {case_tag}  OK')",
        f"except Exception as e:",
        f"    print(f'[CASE_END] {case_tag}  CRASH  '
              f'B={{b}} N1={{n1}} N2={{n2}} S1={{s1}} S2={{s2}} D={{d}}')",
        f"    torch.npu.synchronize()",
        f"    torch.npu.empty_cache()",
        f"    import traceback; traceback.print_exc()",
        f"    print(f'[CASE_END] {case_tag}  DONE_AFTER_CRASH')",
        f"print(f'  [hot] {idx+1}/{total}')", "",
    ]
```

冷启动同理，在 `for _ in range(runs):` 外层加 try/exept。
崩溃后执行 `torch.npu.synchronize()` + `torch.npu.empty_cache()` 清理 NPU 状态。

### 9.2 msprof 层：容忍子进程崩溃

```python
def run_msprof(script: str, output_dir: Path) -> Optional[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "_perf_sweep.py").write_text(script)

    msprof_dir = output_dir / "msprof_output"
    if msprof_dir.exists():
        shutil.rmtree(msprof_dir)

    # 不做 check=True，允许子进程非零退出
    result = subprocess.run(
        ["msprof", f"--output={msprof_dir}", "--aic-mode=task-based",
         sys.executable, str(output_dir / "_perf_sweep.py")],
        capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(f"[msprof] stderr:\n{result.stderr}")
    
    # 即使进程崩溃，msprof 仍可能已写出部分 op_summary
    prof_dirs = sorted(msprof_dir.glob("PROF_*"), reverse=True)
    if not prof_dirs:
        print("[msprof] 无 PROF_* 目录，可能全部崩溃")
        return None

    csv_files = sorted(prof_dirs[0].glob(
        "mindstudio_profiler_output/op_summary_*.csv"))
    if not csv_files:
        print("[msprof] 无 op_summary CSV")
        return None

    dst = output_dir / "op_summary.csv"
    shutil.copy(csv_files[0], dst)
    print(f"[msprof] op_summary -> {dst}")
    return dst
```

### 9.3 解析层：处理缺失的 entry 和崩溃标记

```python
def compute_stats(entries, cases, runs, cold_thr):
    # ... 同前 ...

    # 生成 expected 时同时记录每个 case 的 tag，用于崩溃检测
    # 崩溃 case 的特征：no entries at all
    result = {}
    crashed_cases = []

    for key, d in raw.items():
        if not d["hot"] and not d["cold"]:
            # 完全没有 entry → 崩溃
            result[key] = {"avg_us": 0, "min_us": 0, "max_us": 0, "mode": "crash"}
            crashed_cases.append(key)
            continue
        # ... 同前统计逻辑 ...
        if d["hot"]:
            # 如果 hot entry 数量不足 runs，说明中间崩溃了
            actual_runs = len(d["hot"])
            if actual_runs < runs:
                print(f"[perf] WARN: {key} hot entries={actual_runs} < expected={runs}, "
                      f"可能中途崩溃")
    return result, crashed_cases
```

### 9.4 输出层：标记崩溃 case

终端和 CSV 中崩溃 case 的 `mode` 列为 `CRASH`，duration 为 `-`：

```
│  Case  ...  Avg(us)  Min(us)  Max(us)   Mode  │
│  ...          123.45    120.1    128.9    hot  │
│  ...              -        -        -   CRASH  │
```

同时在 perf.csv 中保留崩溃 case，avg_us/min_us/max_us 为空，便于后续筛选排查。

---

## 十、注意事项

1. **msprof 依赖**：需 CANN 环境已安装 msprof
2. **runs ≥ 5**：冷启动去最大最小后至少 3 个样本
3. **FlashAttn**：op_summary 中算子名为 `FlashAttn`
4. **metadata 耗时**：Task Duration 含 metadata + kernel 全链路
5. **PA 场景**：`performance_redline_infer.py` 有 PA_BBND 用例，脚本生成需处理 `block_table` / `seqused_kv`
6. **刷内存**：冷启动用 `(32, 32768, 1, 128)` randn 刷 HBM（~256MB FP16/张）
7. **现有代码零影响**：`--perf_mode` 是独立分支
8. **崩溃恢复**：崩溃后必须 `torch.npu.synchronize()` + `torch.npu.empty_cache()` 清理 NPU 状态，否则后续 case 可能受污染
