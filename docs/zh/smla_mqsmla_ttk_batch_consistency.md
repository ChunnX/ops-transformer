# SMLA/MQSMLA TTK Batch 一致性测试

本文用于执行 `sparse_flash_mla`（SMLA）和
`mixed_quant_sparse_flash_mla`（MQSMLA）的 TTK batch 一致性测试。MQSMLA 不是
`quant_sparse_flash_mla`（QSMLA），不要混用 assets 或 CSV。

测试包含两层校验：

- 同 case：assets custom compare 比较一个 NPU 输出内声明为同一 relation 的 batch 切片；
- 跨 case：分阶段 replay 使用 `--dump out` 保存各 case 的 NPU 输出，assets 内脚本在全部 case
  结束后比较相同 relation 的切片。

## 1. 前置条件

1. 准备包含 batch 实现的 SMLA/MQSMLA 自定义算子包，例如从 `pr_9103` 对应提交编译的包。
   算子包和当前源码必须来自兼容提交。
2. SMLA 可在注册了 `ascend910_93` 的 A3 环境执行；MQSMLA 只注册 `ascend950`，必须在 A5
   环境执行。
3. 准备 CANN 环境脚本、`ops-test-kit`、本算子仓和 `ttk_data` 的实际路径。它们不要求位于同一目录。
4. 每次使用新的空输出目录，避免读取上一次测试留下的同名 output bin。

先设置公共路径：

```bash
source /path/to/env.sh

TTK_FRAMEWORK=/path/to/ops-test-kit
OPS_TRANSFORMER=/path/to/ops-transformer
TTK_DATA=/path/to/ttk_data
CUSTOM_OPP=/path/to/custom_opp

source "$CUSTOM_OPP/vendors/custom_transformer/bin/set_env.bash"
```

SMLA 使用：

```bash
OP=sparse_flash_mla
CSV="$TTK_DATA/Ascend910B3/e2e/test_sparse_flash_mla_batch_consistency.csv"
```

MQSMLA 在 A5 使用：

```bash
OP=mixed_quant_sparse_flash_mla
CSV="$TTK_DATA/Ascend950/e2e/test_mixed_quant_sparse_flash_mla_batch_consistency.csv"
```

## 2. 快速 case：单阶段只 dump NPU output

如果 case 可以一次跑完，跨 case 比较不需要先保存 input/Golden。直接执行 E2E 并只 dump 本轮
NPU output：

```bash
ASSETS="$OPS_TRANSFORMER/attention/$OP/tests/assets"
OUT="$(mktemp -d "${TMPDIR:-/tmp}/ttk_${OP}_batch_quick.XXXXXX")"
export NPU_DUMP_PATH="$OUT/npu_output"
mkdir -p "$NPU_DUMP_PATH"

export PYTHONPATH="$TTK_FRAMEWORK:$OPS_TRANSFORMER/torch_extension:$ASSETS:${PYTHONPATH:-}"

python3 -m ttk e2e \
  -i "$CSV" \
  --plugin "$ASSETS" \
  --dump out \
  -o "$OUT/result.csv"

python3 "$ASSETS/compare_batch_outputs.py" \
  --csv "$CSV" \
  --result "$OUT/result.csv" \
  --dump-dir "$NPU_DUMP_PATH" \
  --report "$OUT/batch_report.json" \
  --require-intra-case \
  --require-cross-case
```

该流程不会创建 manual-data 目录，也不会生成 input 或 Golden 文件。`--dump out` 只把本轮 NPU
输出写入 `NPU_DUMP_PATH`。慢 case、迁移执行或需要复用既有 input/Golden 时，再使用下一节的分阶段流程。

## 3. 分阶段执行

设置本次测试目录。`MANUAL_DATA_DIR` 保存可 replay 的 input/Golden，`NPU_DUMP_PATH` 只保存
replay 产生的 NPU output：

```bash
ASSETS="$OPS_TRANSFORMER/attention/$OP/tests/assets"
OUT="$(mktemp -d "${TMPDIR:-/tmp}/ttk_${OP}_batch.XXXXXX")"
MANUAL_DATA_DIR="$OUT/manual_data"
export NPU_DUMP_PATH="$OUT/npu_output"
mkdir -p "$MANUAL_DATA_DIR" "$NPU_DUMP_PATH"

export PYTHONPATH="$TTK_FRAMEWORK:$OPS_TRANSFORMER/torch_extension:$ASSETS:${PYTHONPATH:-}"
```

### 3.1 Prepare input 和 Golden

```bash
python3 -m ttk e2e \
  -i "$CSV" \
  --plugin "$ASSETS" \
  --no-prof \
  --dump in,golden \
  --dump-format bin \
  --manual-data-dirs "$MANUAL_DATA_DIR" \
  -o "$OUT/prepare_result.csv"
```

该阶段不执行 NPU，不会产生跨 case 比较所需的 `*_output_0.bin`。

### 3.2 Replay 并只 dump NPU output

```bash
python3 -m ttk e2e \
  -i "$CSV" \
  --plugin "$ASSETS" \
  --manual-data-dirs "$MANUAL_DATA_DIR" \
  --dump out \
  -o "$OUT/replay_result.csv"
```

replay 只改变 input/Golden 的来源，仍会执行 NPU 和正常 compare。`--dump out` 将本轮 NPU
结果写为 `$NPU_DUMP_PATH/<testcase_name>_output_<index>.bin`，不会再次生成 input 或 Golden。

### 3.3 比较跨 case NPU output

```bash
python3 "$ASSETS/compare_batch_outputs.py" \
  --csv "$CSV" \
  --result "$OUT/replay_result.csv" \
  --dump-dir "$NPU_DUMP_PATH" \
  --report "$OUT/batch_report.json" \
  --require-intra-case \
  --require-cross-case
```

脚本只使用命令行传入的 CSV、result、dump 和 report 路径，不依赖仓库位置或当前工作目录。

`--require-intra-case` 要求每个 relation 在每个参与 case 中至少有两个切片，并要求 result 的
`eager_precision` 包含 `batch_intra=PASS`。`--require-cross-case` 要求每个 relation 至少出现在两个
不同 case 中。这两个选项不会生成 relation，也不会改变 compare 算法；它们用于拒绝覆盖不完整的测试矩阵。

## 4. 通过标准

以下条件必须同时满足：

1. `replay_result.csv` 中所有参与 case 的 `precision_status` 都是 `PASS`；
2. 每条参与 case 的 `eager_precision` 都包含 `batch_intra=PASS`；
3. 外置脚本退出码为 0，并打印 `batch consistency PASS`；
4. `batch_report.json` 顶层及每个 relation 的 `status` 都是 `PASS`。

若脚本报告 `missing output dump`，确认 replay 命令带有 `--dump out`，且 `NPU_DUMP_PATH` 指向本次
新建目录。`raw output differs` 表示同一 relation 的 NPU 原始输出位不一致；`input context differs`
表示 shape、dtype、range、attributes 或有效长度不同，这些切片不能作为同一跨 case relation 比较。

## 5. 当前边界

当前 assets 支持 `BSND/BSND`、Q 输入索引 0、batch 轴 0、连续切片（`step=1`）和主输出索引 0；
CSV 的 `batch_deterministic_level` 为 3。比较按原始 storage 字节进行，不使用数值容差。扩展 layout、
输入轴或输出索引时，需要同步修改确定性 input、同 case compare、跨 case脚本和白盒测试。

命令不要求指定 `--dev`、`--pc`、`--warmup` 或 `--run`，使用 TTK 默认调度即可。
