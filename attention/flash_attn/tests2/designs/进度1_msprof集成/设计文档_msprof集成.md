# msprof 性能采集集成设计

## 1. 目标

在 tests2 框架中集成 msprof 性能采集，支持对 flash_attn 算子进行应用级（`msprof <app>`）性能采集，
产物按用例分目录存放，并提供汇总表。

## 2. 架构决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 集成方式 | 方案B：保留 stage 形态 | 复用 `--stages` 体系，CLI 一致；`npuprof` stage 内部 subprocess 包裹 |
| 性能模式 | 应用级 `msprof <app>` | 与 4 个既有框架一致，采集真实图上下文，产物 op_summary.csv |
| L2 缓存 | TODO（暂不实现） | 冷热缓存方案待定，先标记 TODO，接口预留 |
| 精度闸门 | perf 前必须 compare 通过 | 避免在错误结果上采性能数据 |

## 3. 核心矛盾与解法

**矛盾**：msprof 的工作方式是包裹整个 Python 进程（`msprof [opts] <python> [args]`），必须在被测
进程启动前介入。但 `--stages` 体系假设 stage 在 `run_case` 进程内顺序执行。

**解法**：`npuprof` stage 不在当前进程内执行算子，而是 `subprocess.run` 起一个子进程，用 msprof
包裹该子进程。子进程只跑 `npu` 阶段（perf 不需要 gen/cpu/compare 产物）。

## 4. 执行模型

### 4.1 用户调用

```bash
# 完整流程：精度通过后采性能
python test.py --stages gen,cpu,npu,compare,npuprof --case_set perf_rdv

# 仅性能（依赖已有 pt 缓存做精度闸门校验）
python test.py --stages compare,npuprof --case_set perf_rdv
```

### 4.2 单 case 流程（run_case 内）

```
run_case(stages={gen,cpu,npu,compare,npuprof}, params, reporter)
  ├─ _stage_gen          → 生成输入
  ├─ _stage_cpu          → golden
  ├─ _stage_npu          → npu 输出
  ├─ _stage_compare      → 精度校验
  │     └─ 失败 → 跳过 npuprof, 记录 SKIP(reason=precision_fail)
  └─ _stage_npuprof      → 性能采集 (仅 compare 通过时)
        ├─ 构造子进程命令:
        │     msprof --output=result/<case_id>/prof --aic-mode=task-based \
        │            python _perf_runner.py --case_id <id> --case_set <set>
        ├─ subprocess.run(cmd)
        ├─ 解析 result/<case_id>/prof/PROF_*/mindstudio_profiler_output/op_summary_*.csv
        └─ reporter.record_perf(case_id, {op_type, duration_us, ...})
```

### 4.3 子进程（_perf_runner.py）

被 msprof 包裹的子进程，只跑 npu 阶段：

```python
# _perf_runner.py (被采集进程)
# 入参: --case_id, --case_set (单 case)
# 职责: load case → gen → npu_compute × runs → synchronize
# 不做: 精度比对、结果记录 (msprof 负责采集, 解析由主进程做)

def main():
    args = parse_args()
    case = load_single_case(args.case_set, args.case_id)
    inputs = GenDataBackend().gen_inputs(...)
    inputs = to_device(inputs, "npu")
    warmup = 1  # 预热轮数
    runs = args.runs  # 采样轮数, 默认 5

    # warmup
    for _ in range(warmup):
        NpuBackend().npu_compute(inputs, "eager")
    torch.npu.synchronize()

    # 采样 (msprof 采集这里的 task)
    for _ in range(runs):
        NpuBackend().npu_compute(inputs, "eager")
    torch.npu.synchronize()
```

## 5. 模块设计

### 5.1 prof/npuprof/__init__.py（实现）

```python
def run_msprof(case_id: str, cmd: list, prof_dir: str) -> str:
    """subprocess 调 msprof 包裹 cmd, 产物落 prof_dir。

    Args:
        case_id: 用例ID (用于日志)
        cmd: 被包裹命令, 如 ["python", "_perf_runner.py", "--case_id", ...]
        prof_dir: msprof --output 目录
    Returns:
        prof_dir (成功) 或 raise (失败)
    """
    # 实现: subprocess.run(["msprof", f"--output={prof_dir}",
    #                       "--aic-mode=task-based"] + cmd)


def parse_prof_result(prof_dir: str) -> dict:
    """解析 op_summary_*.csv, 提取 flash_attn 算子耗时。

    Returns:
        {"op_type": "FlashAttn", "duration_us": float, "task_id": str, ...}
    """
    # 实现: glob PROF_*/mindstudio_profiler_output/op_summary_*.csv
    #       按 OP Type 过滤, 取 Task Duration(us)
```

### 5.2 orchestrator.py 新增 _stage_npuprof

```python
def _stage_npuprof(stages, params, compare_result, case_set, case_id, result_dir):
    """精度闸门 + subprocess 采性能。"""
    if "npuprof" not in stages:
        return None
    # 精度闸门
    if not compare_result.get("passed", False):
        logger.info("[npuprof] skip | precision not passed")
        return {"skipped": True, "reason": "precision_fail"}
    # 构造子进程命令
    prof_dir = str(Path(result_dir) / "prof")
    cmd = [sys.executable, "_perf_runner.py",
           "--case_id", case_id, "--case_set", case_set]
    run_msprof(case_id, cmd, prof_dir)
    perf = parse_prof_result(prof_dir)
    logger.info("[npuprof] done | %s duration=%.2fus",
                perf.get("op_type", ""), perf.get("duration_us", 0))
    return perf
```

### 5.3 _perf_runner.py（新增子脚本）

独立入口，被 msprof 包裹。只跑 npu，不做精度校验和结果记录。

### 5.4 Reporter 扩展

新增 `record_perf(case_id, perf: dict)`，perf 数据写入独立 CSV（`result/perf_summary_<ts>.csv`），
列: `case_id, op_type, duration_us, task_id`。

## 6. CLI 接口

复用现有 `--stages`，新增 `npuprof` 阶段。新增参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--perf_runs` | 5 | 采样轮数（子进程内跑几次算子） |

无独立 `--perf` 模式（方案B 复用 stages）。

## 7. 产物规范

```
result/
├── result_<ts>.csv              # 精度结果 (既有)
├── perf_summary_<ts>.csv        # 性能汇总 (新增)
└── <case_id>/
    └── prof/                     # msprof 产物
        └── PROF_<digits>_<ts>_<host>/
            └── mindstudio_profiler_output/
                └── op_summary_<ts>.csv
```

## 8. 实现计划

| 步骤 | 文件 | 验证 |
|------|------|------|
| 1. 实现 prof/npuprof | prof/npuprof/__init__.py | 单 case subprocess 跑通, 产出 op_summary.csv |
| 2. 新增 _perf_runner.py | _perf_runner.py | 手动 msprof 包裹跑通, 能采到 FlashAttn 耗时 |
| 3. 集成到 orchestrator | orchestrator.py | --stages npuprof 跑通, 精度闸门生效 |
| 4. Reporter 扩展 | orchestrator.py | perf_summary.csv 写入正确 |
| 5. CLI 参数 | test.py | --perf_runs 生效 |

L2 缓存（`prof/l2_flush`）暂保持 NotImplementedError，TODO 待定。

## 9. TODO

- [ ] L2 缓存刷新方案（冷热缓存区分）
- [ ] gpuprof / npuprofop 实现（当前仅 npuprof）
- [ ] perf_parse 模块（op_summary 解析与 baseline 比对）
