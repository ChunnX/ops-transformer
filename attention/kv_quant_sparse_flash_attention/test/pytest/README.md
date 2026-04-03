# kv_quant_sparse_flash_attention pytest 测试框架

## 文件结构

```text
pytest/
├── README.md
├── pytest.ini
├── test_run.sh
├── check_valid_param.py
├── kv_quant_sparse_flash_attention_golden.py
├── kv_quant_sparse_flash_attention_paramset.py
├── result_compare_method.py
├── utils.py
├── test_kv_quant_sparse_flash_attention_single.py
├── test_kv_quant_sparse_flash_attention_batch.py
├── test_kv_quant_sparse_flash_attention_batch_graph.py
└── batch/
    ├── __init__.py
    ├── kv_quant_sparse_flash_attention_process.py
    └── test_kv_quant_sparse_flash_attention_pt_save.py
```

## 概述

该目录用于在 `attention/kv_quant_sparse_flash_attention` 算子目录内直接进行 pytest 调测，保留 single 和 batch 两条主流程：

- `single`：基于 `kv_quant_sparse_flash_attention_paramset.py` 的固定参数直接构造输入并拉起 NPU 单算子执行。
- `batch`：先从 Excel 读取参数生成 `.pt` 用例，再批量回放执行 NPU 算子。
- `graph`：基于 batch 生成的 `.pt` 用例执行图模式编译与回放。

## 当前能力边界

- 保留了结果对比基础设施，例如 `result_compare_method.py` 和结果落盘逻辑。
- 按需求移除了 `run_npu_only` 相关能力，不再提供对应脚本和测试入口。
- `perf_run.sh` 未迁入本目录。
- `generate_and_save_testdata(..., compute_golden=True)` 不再执行 CPU golden 推导；当前会打印提示信息并返回 `cpu_output=None`。
- 因 CPU golden 暂不支持，当前 single/batch/graph 执行结果以 NPU 连通性为主，`result.xlsx` 中会记录为 `SkipCompare`。

## 参数说明

以下参数约束已经下沉到框架校验和本文档中，便于统一维护：

- `layout_query`: 支持 `BSND`、`TND`
- `layout_kv`: 支持 `BSND`、`TND`、`PA_BSND`
- 非 PA 场景要求 `layout_query == layout_kv`
- `q_type`: 仅支持 `torch.float16`、`torch.bfloat16`
- `kv_dtype`: 支持 `hifloat8`、`float8_e4m3fn`，也兼容 `None` 作为 `float8_e4m3fn` 默认生成路径
- `N1`: 仅支持 `1/2/4/8/16/32/48/64`
- `N2`: 仅支持 `1`
- `sparse_mode`: 仅支持 `0` 和 `3`
- `sparse_block_size`: 当前仅支持 `1`
- `key_quant_mode` / `value_quant_mode`: 仅支持 `2`
- `tile_size`: 当前仅支持 `128`
- `quant_scale_repo_mode`: 当前仅支持 `1`
- `attention_mode`: 支持 `0`、`2`；当取 `2` 时，`rope_head_dim` 必须为 `64`
- `block_size`: 仅在 `PA_BSND` 生效，且要求为 16 的倍数
- `actual_seq_q` / `actual_seq_kv`: 如果传入，长度必须等于 `B`

更完整的算子定义和输入约束，请同步参考：

- `attention/kv_quant_sparse_flash_attention/README.md`
- `attention/kv_quant_sparse_flash_attention/op_host/kv_quant_sparse_flash_attention_def.cpp`

## 关键实现说明

### 1. 测试数据生成

`kv_quant_sparse_flash_attention_golden.py` 负责构造以下输入：

- `query`
- `key`
- `value`
- `sparse_indices`
- `block_table`
- `actual_seq_lengths_query`
- `actual_seq_lengths_kv`

其中：

- Q 由 `query_base` 与 `query_rope` 沿最后一维拼接生成。
- K 按算子要求拼接 `key_base + key_rope + antiquant_scale` 存储布局。
- 在 `TND` 布局下，框架会按 `actual_seq_*` 自动裁剪有效 token 并转换为前缀和长度张量。
- 在 `PA_BSND` 布局下，框架会自动生成 `block_table`，若未显式提供 `block_num`，则按 `max(actual_seq_kv)` 推导。

### 2. CPU golden 现状

原始外部框架包含 hifloat8 decode 和 CPU golden 逻辑。本仓落库版本已按需求简化：

- 删除了 `decode_hifloat8` 相关接口和实现。
- 删除了 CPU golden 计算分支。
- 当显式请求 `compute_golden=True` 时，仅打印“当前 CPU golden 暂不支持”的说明，不参与精度对比。

### 3. 执行与结果记录

- `utils.py` 负责参数组合、Excel 读取、测试结果落盘。
- `batch/kv_quant_sparse_flash_attention_process.py` 负责单算子与图模式的 NPU 调用。
- 若 `cpu_output is None`，执行结果记为 `SkipCompare`，表示本次仅校验 NPU 执行链路。

## 运行方式

在 `attention/kv_quant_sparse_flash_attention/test/pytest` 目录下执行：

### single

```bash
bash test_run.sh single
```

### batch

```bash
bash test_run.sh batch
```

batch 流程依赖 `./excel/example.xlsx`，执行时会先生成 `.pt` 用例，再回放执行。

### graph

```bash
bash test_run.sh graph
```

## 结果文件

- `result.xlsx`：记录每个用例的关键信息、执行状态与 `fulfill_percent`
- `../../pt_files/*.pt`：batch 流程生成的中间测试用例
