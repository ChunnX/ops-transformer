# QuantLightningIndexerV2 批量测试与性能采集

## 新增功能

### 1. 双模式执行隔离（flag 隔离）

`test_quant_lightning_indexer_v2_batch.py` 支持两种执行模式，通过环境变量 `QLIV2_TESTCASE_PATH` 自动切换：

| 模式 | 触发条件 | Executor | 说明 |
|---|---|---|---|
| **直接 pytest** | `QLIV2_TESTCASE_PATH` 未设置 | `ProcessPoolExecutor` | 每条用例在独立子进程中执行，单条崩溃不影响整体 |
| **批量隔离** | `QLIV2_TESTCASE_PATH` 已设置 | `ThreadPoolExecutor` | shell 层已提供进程隔离，内部使用线程池 |

```python
# flag 判断逻辑（test_quant_lightning_indexer_v2_batch.py:36）
_is_isolated_mode = bool(os.environ.get("QLIV2_TESTCASE_PATH", "").strip())
```

### 2. 批量隔离执行脚本 `batch_isolated_run.sh`

对每条用例单独拉起一个 pytest 进程，实现进程间完全隔离：
- 某条用例 device 越界/崩溃不会影响后续用例
- msprof 挂载在单用例进程上，性能数据采集互不干扰

### 3. 性能数据收集 `collect_perf_data.py`

从 msprof 输出的 `op_summary*.csv` 中提取 `QuantLightningIndexerV2` 算子的完整性能字段（aicore_time, cube_utilization, aiv_time 等），与测试结果合并输出到 `result_perf.xlsx`。

### 4. 运行模式切换（eager / graph）

`test_quant_lightning_indexer_v2_batch.py` 支持两种算子调用模式，通过环境变量 `QLIV2_RUN_MODE` 控制：

| 模式 | QLIV2_RUN_MODE | 说明 |
|---|---|---|
| **eager**（默认） | `eager` | 直接调用 `torch.ops.cann_ops_transformer.quant_lightning_indexer` |
| **graph** | `graph` | 通过 `torch.compile` + `torchair` 后端编译执行 |

```python
# flag 判断逻辑（test_quant_lightning_indexer_v2_batch.py:38）
_run_mode = os.environ.get("QLIV2_RUN_MODE", "eager").strip().lower()
```

---

## 使用方法

### 环境准备

```bash
conda activate <your_env>
source <cann_path>/set_env.sh
cd <repo_root>/attention/quant_lightning_indexer_v2/tests/pytest
```

### 模式1：直接 pytest（原有方式，多进程隔离）

可通过手动步骤或 `test_run.sh batch` 一键执行：

```bash
# 方式A：test_run.sh 一键执行（推荐）
bash test_run.sh batch          # eager 模式（默认）
bash test_run.sh batch graph    # graph 模式

# 方式B：手动分步执行
# 1. 生成 .pt 文件
python3 batch/quant_lightning_indexer_v2_pt_save.py excel/test_cases.xlsx pt_path

# 2. 替换测试脚本路径
python3 batch/replace_path.py test_quant_lightning_indexer_v2_batch.py pt_path

# 3. 执行测试
python3 -m pytest -rA -s test_quant_lightning_indexer_v2_batch.py -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning

# 4. 恢复测试脚本
cp test_quant_lightning_indexer_v2_batch.py.bak test_quant_lightning_indexer_v2_batch.py
```

### 单用例调测（test_run.sh single）

```bash
bash test_run.sh single
```

直接基于 `test_quant_lightning_indexer_v2_paramset.py` 中 `ENABLED_PARAMS` 的参数组合执行单算子精度验证，无需 Excel 用例表和 .pt 文件。

### 模式2：批量隔离执行（shell 层进程隔离）

```bash
# 不采集性能
bash batch_isolated_run.sh ./pt_path 0

# 采集性能（挂载 msprof）
bash batch_isolated_run.sh ./pt_path 1
```

脚本会自动完成：
1. 遍历 `pt_path` 下所有 `.pt` 文件
2. 每条用例通过 `QLIV2_TESTCASE_PATH` 环境变量传递，独立进程执行
3. 汇总 pass/fail 统计到 `batch_summary.log`
4. 若启用 msprof，自动调用 `collect_perf_data.py` 收集性能数据

### 自定义模式组合

可通过环境变量灵活组合各 flag：

```bash
# 批量隔离 + eager 模式（默认）
bash batch_isolated_run.sh ./pt_path 0 eager

# 批量隔离 + graph 模式 + 性能采集
bash batch_isolated_run.sh ./pt_path 1 graph

# 直接 pytest + graph 模式（单条用例，便于调试）
QLIV2_RUN_MODE=graph QLIV2_TESTCASE_PATH="./pt_path/test_case_01.pt" \
  python3 -m pytest -rA -s test_quant_lightning_indexer_v2_batch.py -v -m ci

# 直接 pytest + eager 模式（所有用例，多进程隔离）
QLIV2_RUN_MODE=eager python3 -m pytest -rA -s \
  test_quant_lightning_indexer_v2_batch.py -v -m ci
```

---

## Excel 用例表格式

`excel/test_cases.xlsx` 需包含以下列（Sheet1）：

| 列名 | 类型 | 示例 |
|---|---|---|
| Testcase_Name | str | `test_case_01` |
| batch_size | int | `8` |
| q_seq | int | `15` |
| k_seq | int | `111` |
| q_t_size | int | `8` |
| k_t_size | int | `15` |
| q_head_num | int | `64` |
| k_head_num | int | `1` |
| head_dim | int | `128` |
| block_size | int | `512` |
| block_num | int | `8` |
| qk_dtype | str | `FLOAT8_E4M3FN` / `INT8` / `HIFLOAT8` |
| dequant_dtype | str | `FP32`（仅支持 FP32） |
| actual_seq_dtype | str | `INT32` |
| cu_seqlens_q | None/str | `None` 或 `"[0, 1]"` |
| cu_seqlens_k | None/str | `None` 或 `"[0, 1]"` |
| seqused_q | None/str | `None` 或 `"[3,3,3,3,3,3,3,3]"` |
| seqused_k | str | `"[28,24,80,96,47,76,0,111]"` |
| cmp_residual_k | None/str | `None` 或 `"[0,0,0,0,0,0,0,0]"`（cmp_ratio>1 时必填）|
| max_seqlen_q | int | `-1` |
| quant_mode | int | `1` |
| layout_query | str | `BSND` / `TND` |
| layout_key | str | `PA_BBND` / `BSND` / `TND` |
| sparse_count | int | `512` |
| sparse_mode | int | `0` / `3` |
| query_datarange | str | `"[-448,448]"` |
| key_datarange | str | `"[-20,20]"` |
| weights_datarange | str | `"[-123,123]"` |
| q_scale_datarange | str | `"[0,255]"` |
| k_scale_datarange | str | `"[0,65504]"` |
| cmp_ratio | int | `1` / `4` |
| return_value | int | `0` / `1` |
| output_idx_offset | None/str | `None` 或列表字符串 |

**注意事项**：
- `dequant_dtype` 仅支持 `FP32`（算子不支持 FP16 的 w/qDescale/kDescale）
- `cmp_ratio > 1` 且 `sparse_mode != 0` 时，`cmp_residual_k` 必填（长度=batch_size 的列表）
- `return_value=1` 时，`output_idx_offset` 需提供有效值

---

## 输出文件

| 文件 | 说明 |
|---|---|
| `result.xlsx` | 测试结果（精度、参数等） |
| `result_perf.xlsx` | 测试结果 + 性能数据（仅 msprof 模式） |
| `batch_summary.log` | 批量执行详细日志 |
| `batch_fail_list.log` | 失败用例清单 |
| `PROF_*/` | msprof 性能原始数据目录 |

---

## 已知限制

1. `dequant_dtype` 仅支持 `FP32`
2. JIT 编译残留锁文件（`~/.cache/torch_extensions/py310_cpu/*/lock`）会导致导入卡死，如遇此问题执行：
   ```bash
    find ~/.cache/torch_extensions/py310_cpu -name "lock" -delete
    ```
3. **graph 模式**依赖 `torchair` 编译器后端，部分设备/配置可能不支持

---

## Dual-Mode Execution Isolation (Flag Isolation)

`test_quant_lightning_indexer_v2_batch.py` supports two execution modes, automatically switched via the `QLIV2_TESTCASE_PATH` environment variable:

| Mode | Trigger | Executor | Description |
|---|---|---|---|
| **Direct pytest** | `QLIV2_TESTCASE_PATH` unset | `ProcessPoolExecutor` | Each case runs in a subprocess; single crash doesn't affect others |
| **Batch isolated** | `QLIV2_TESTCASE_PATH` set | `ThreadPoolExecutor` | Shell provides process isolation; thread pool used internally |

## Usage

### Mode 1: Direct pytest (original multi-process isolation)

```bash
# Option A: test_run.sh one-liner (recommended)
bash test_run.sh batch          # eager mode (default)
bash test_run.sh batch graph    # graph mode

# Option B: manual step-by-step
python3 batch/quant_lightning_indexer_v2_pt_save.py excel/test_cases.xlsx pt_path
python3 batch/replace_path.py test_quant_lightning_indexer_v2_batch.py pt_path
python3 -m pytest -rA -s test_quant_lightning_indexer_v2_batch.py -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
cp test_quant_lightning_indexer_v2_batch.py.bak test_quant_lightning_indexer_v2_batch.py
```

### Single Case Test (test_run.sh single)

```bash
bash test_run.sh single
```

Runs single-operator accuracy validation based on `ENABLED_PARAMS` in `test_quant_lightning_indexer_v2_paramset.py`. No Excel or .pt files needed.

### Mode 2: Batch isolated execution (shell-level process isolation)

```bash
bash batch_isolated_run.sh ./pt_path 0         # without msprof
bash batch_isolated_run.sh ./pt_path 1         # with msprof perf collection
bash batch_isolated_run.sh ./pt_path 0 graph   # graph mode, without msprof
bash batch_isolated_run.sh ./pt_path 1 graph   # graph mode + perf collection
```

### Run Mode: eager vs graph

Set via `QLIV2_RUN_MODE` environment variable or `test_run.sh` parameter:

```bash
# eager mode (default, direct operator call)
bash test_run.sh batch
bash test_run.sh batch eager

# graph mode (torch.compile + torchair backend)
bash test_run.sh batch graph

# Direct pytest with mode override
QLIV2_RUN_MODE=graph python3 -m pytest -rA -s \
  test_quant_lightning_indexer_v2_batch.py -v -m ci
```

## Output Files

| File | Description |
|---|---|
| `result.xlsx` | Test results (accuracy, parameters) |
| `result_perf.xlsx` | Test results + performance data (msprof mode only) |
| `batch_summary.log` | Batch execution log |
| `batch_fail_list.log` | Failed case list |
| `PROF_*/` | Raw msprof performance data |

## Known Limitations

1. `dequant_dtype` only supports `FP32`
2. Stale JIT build lock files may cause import hang; fix with:
   ```bash
    find ~/.cache/torch_extensions/py310_cpu -name "lock" -delete
    ```
3. **Graph mode** depends on `torchair` compiler backend; some devices/configurations may not support it.
