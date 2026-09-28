# Quant Sparse Lightning Indexer pytest

## Checkout 验证与接口

测试通过 `torch.ops.cann_ops_transformer.ds41` 调用本仓库的 Torch extension。
主算子和 metadata 均使用 `layout_k`，支持 `TND` / `PA_BBND`；默认值保留为 `TND`，PA 用例必须显式传入布局。
调用方必须先生成 metadata 并显式传入；省略时沿用底层错误，不自动生成。
metadata 输出为 `int32[1024]`，`return_value=False` 时 values 为空张量。

先激活匹配的 Torch/NPU/DSL 环境并加载 CANN 的 `set_env.sh`。
强制从 `CANNBOTDSL_TEST_OPS_DIR` 指定的 kernel 仓 `net/ops` 和本仓算子的 `torch_extension` 加载，打印来源路径和 SHA-256。
需要安装 `cann_ops_transformer` 基础包，或将 `OPS_TRANSFORMER_TORCH_EXTENSION_DIR`
设为其源码 `torch_extension` 目录（其中包含 `cann_ops_transformer/op_builder`）。
加载器只加载两个本地算子和真实 OpBuilder，不执行无关算子的自动发现。

公共加载和图测试帮助函数位于 `experimental/attention/indexer_test_runtime.py`。
不会回退到集中开发仓或已安装的旧算子。ACL Graph 显式捕获 metadata→主算子链路。

```bash
export CANNBOTDSL_TEST_OPS_DIR=/path/to/cannbot-arena-ds41/net/ops
export OPS_TRANSFORMER_TORCH_EXTENSION_DIR=/path/to/ops-transformer_dsv41/torch_extension
export QSLI_DEVICE_ID=1
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_ADDOPTS="-p no:cacheprovider"
export TMPDIR=/tmp/ops_transformer_indexer/qsli
mkdir -p "$TMPDIR"
export CANNBOTDSL_CACHE_DIR="$TMPDIR/cache"
python -B -m pytest --collect-only .
bash test_run.sh single
```

脚本默认产物位于上述临时目录；旧布局字段的 PT 请重新生成。

该目录按主线 ASC pytest 的流程组织，但所有数据生成、golden、算子调用和比较均为 QSLI 独立实现，不引用 ASC 测试代码。

## 文件结构

- `test_quant_sparse_lightning_indexer_paramset.py`：single 参数集，含 decode、prefill 和 8 条交叉用例。
- `test_quant_sparse_lightning_indexer_single.py`：paramset 展开、按名称筛选、可选 PT 保存和 NPU 精度比较。
- `test_quant_sparse_lightning_indexer_batch.py`：读取 PT、批量执行并调用 `result_compare_method`。
- `qsli_test_utils.py`：参数生成、连续/axis-0 非连续 PA、NPU 调用和比较公共流程。
- `quant_sparse_lightning_indexer_golden.py`：独立 CPU golden；先按新接口解包 K/scale，数值计算与比较规则保持不变。
- `batch/quant_sparse_lightning_indexer_pt_save.py`：从 Excel 生成 PT。
- `batch/list_pt_from_excel.py`：按 Excel `Testcase_Name` 筛选已有 PT。
- `test_run.sh`：single、batch、batch_exec 统一入口。
- `batch_isolated_run.sh`：每条 PT 独立 pytest 进程。

## 运行

```bash
bash test_run.sh single
bash test_run.sh single -C qsli_decode,qsli_prefill
bash test_run.sh batch -E ./excel/qsli_decode_prefill_20cases.xlsx -P "$TMPDIR/pt_path"
bash test_run.sh batch_exec -E ./excel/qsli_decode_prefill_20cases.xlsx -P "$TMPDIR/pt_path"
bash batch_isolated_run.sh "$TMPDIR/pt_path" 0
```

Excel 默认读取 `TestCases` sheet。输出索引和 value 均使用本目录的 `result_compare_method.py` 判定。

融合 K 的物理 shape 为 `(PA, block_size/8, 544)`。旧 PT 在 CPU 测试适配器中打包，新 PT 直接保存融合 K；算子接口不执行额外打包 kernel。`test_quant_sparse_lightning_indexer_generalization.py` 验证 PA64/128、mask0/3、cmp_ratio1/2、offset 有/无，以及零候选和尾块。

TND 使用独立 K/scale，`descale_k` 为关键字参数；Golden 和参数生成同时覆盖 TND、PA，以及长序列抽样行。
当前 arena 的 TND metadata 实现要求提供 `cu_seqlens_q` 和 `cu_seqlens_k`；测试生成器默认生成这两个累计长度张量。

## Excel 双布局完整行验证

设置 `QSLI_EXCEL_SELECTION` 为用例 JSON 路径，格式为
`{"cases": [{"workbook": "qsli_0910.xlsx", "sheet": "Sheet1", "row": 2, "case_name": "表内名称"}]}`。
`row` 是包含表头的 Excel 物理行号，workbook 相对于本目录的 `excel/`。

```bash
export QSLI_EXCEL_SELECTION="/absolute/path/to/selection.json"
export QSLI_VALIDATION_OUTPUT="/absolute/path/to/results"
export QSLI_DEVICE_ID=1
python -B -m pytest -v -s test_quant_sparse_lightning_indexer_batch.py -k excel_layouts
```

每条源 case 先生成 PA 输入，再按其页表解包为等价的 TND K/scale；Q、权重、候选和长度保持一致。
两种布局分别执行，CPU golden 每块 32 行覆盖全部输出，包括 prefill 未使用的 query 行。
沿用既有并列分数和 value 比较规则，并核对双布局的完整逻辑 score 相同。
结果 JSON 记录源行、参数、输入指纹、校验行数及输出逐位一致性；逐位不同不替代既有并列分数判定。

Excel PT 生成器支持 `--paired-layouts` 生成 `__PA_BBND.pt` 和 `__TND.pt`，
以及 `--defer-golden` 延迟 golden 计算。现有 batch 入口对延迟 golden 的 PT 自动执行完整行分块校验。
未启用这些选项的历史 single/batch 入口保留原有行为。
