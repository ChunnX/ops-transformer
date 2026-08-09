# QBSA MXFP8 TTK 测试

本目录把 `pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_test_cases.py` 中 `enable=True` 的参数集转换为 TTK E2E 用例。TTK 负责用例解析、调度、NPU 执行和结果汇总；`tests/assets` 中的 TestSpec 复用原 pytest 数据生成、CPU Golden 和精度标准。

## 前置条件

1. 已安装与当前源码匹配的 CANN、PyTorch 和 `torch_npu`。
2. 已构建并安装 QBSA PTA 扩展：

   ```bash
   cd ${OPS_TRANSFORMER_ROOT}/experimental/attention/quant_block_sparse_attn/torch_ops_extension
   bash build_and_install.sh
   ```

3. 在 `ops-test-kit` 仓库根目录执行 TTK 命令。

## 校验与执行

```bash
export OPS_TRANSFORMER_ROOT=/path/to/ops-transformer
export QBSA_ASSETS=${OPS_TRANSFORMER_ROOT}/experimental/attention/quant_block_sparse_attn/tests/assets
export QBSA_MXFP8_CASES=${OPS_TRANSFORMER_ROOT}/experimental/attention/quant_block_sparse_attn/tests/ttk/bsa_fullquant_mxfp8_test/qbsa_mxfp8_cases.csv
export PYTHONPATH=${QBSA_ASSETS}:${PYTHONPATH:-}

python3 -m ttk -v
python3 -m ttk info
python3 -m ttk e2e -i "${QBSA_MXFP8_CASES}" --plugin "${QBSA_ASSETS}" --validate
python3 -m ttk list -i "${QBSA_MXFP8_CASES}"
```

本目录的 `qbsa_mxfp8_cases.csv`、`qbsa_mxfp8_test_case_STC.csv` 和
`qbsa_mxfp8_test_case_full.csv` 统一使用以下 12 列 E2E 表头：

```text
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,tensor_formats,
attributes,output_tensor_indexes,golden_api,input_data_ranges,
precision_tolerances,absolute_precision,uncontiguous_dim
```

`attributes` 中的键固定按以下三组排列，便于同时核对算子接口和测试构造逻辑：

1. 算子原生属性：`softmax_scale`、`sparse_q_block_size`、
   `sparse_kv_block_size`、`layout_kv`、`layout_q`、
   `layout_sparse_indices`、`layout_out`、`quant_mode`、`mask_mode`、
   `return_softmax_lse`；顺序与 QBSA 接口一致。
2. 转换为输入 Tensor 的值：`p_scale_value`、`cu_seqlens_q_value`、
   `cu_seqlens_kv_value`、`seqused_q_value`、`seqused_kv_value`。
3. 构造用例的辅助变量：`is_contiguous`、`block_table_pattern`、
   `sparse_count`、`sparse_pattern`、`seed`、`quant_group_size`、
   `s2_base_size`、可选 `blocknum` 以及 Q/K/V 数据范围。

MXFP8 正常用例的序列输入约定如下：

- `cu_seqlens_q_value` 保存 Q 各 batch 有效长度的累加和，长度必须为 `B+1`，首元素为 0；
- `cu_seqlens_kv_value=[]`，执行时转换为 `cu_seqlens_kv=None`；
- `seqused_q_value=[]`，执行时转换为 `seqused_q=None`；
- `seqused_kv_value` 保存每个 batch 的 KV 有效长度，长度必须为 `B`；
- `B` 必须大于 0；单个 batch 的 Q/KV 有效长度允许为 0，但整条用例的 Q/KV 总长度必须大于 0。

`sparse_count` 支持六种形式：

| 写法 | 含义 |
|---|---|
| `"random"` | 在当前有效稀疏块中随机选择 1 到有效块总数个 |
| `10` | 选择 `min(10, 当前有效块总数)` 个 |
| `"10%"` | 按当前有效块总数的 10% 选择，向上取整且至少 1 个 |
| `[n0, n1, ...]` | 按 batch 指定数量；list 长度必须等于 `B`，每项仍取有效块数上限 |
| `"empty_random"` | 交替生成 0 块和随机非 0 块，用于部分空稀疏 |
| `"empty"` | 所有位置均不选择稀疏块，用于全空稀疏 |

`tensor_view_shapes`、`tensor_dtypes`、`tensor_formats` 按 wrapper 的 10 个输入位置一一对应；
`uncontiguous_dim=-1` 表示连续 K/V，`uncontiguous_dim=1` 表示合法非连续 K/V PA view。
pytest Golden 也能直接读取这三张表，保证两套执行框架使用相同参数。

先执行一条小用例冒烟：

```bash
python3 -m ttk e2e \
  -i "${QBSA_MXFP8_CASES}" \
  --plugin "${QBSA_ASSETS}" \
  -t STC_B1_BS128_MIN_PAGE_EQ_MASK3 \
  --pc 1 --single-log --dump-on-fail
```

以 Eager 模式执行全部 71 条启用用例：

```bash
python3 -m ttk e2e \
  -i "${QBSA_MXFP8_CASES}" \
  --plugin "${QBSA_ASSETS}" \
  --pc 1 --proc-no-reuse --proc-timeout 600 \
  --warmup=false --run 1 --single-log
```

以 ACLGraph 模式执行全部 71 条启用用例：

```bash
python3 -m ttk e2e \
  -i "${QBSA_MXFP8_CASES}" \
  --plugin "${QBSA_ASSETS}" \
  --aclgraph \
  --pc 1 --proc-no-reuse --proc-timeout 600 \
  --warmup=false --run 1 --single-log
```

## TTK E2E 参数说明

本节覆盖当前 `python3 -m ttk e2e --help` 中的参数。“CLI 必需”表示 TTK 命令行强制要求；“QBSA 必需”表示虽然 TTK 本身不强制，但本测试缺少它将无法正确生成输入或 Golden；“条件必需”表示只在特定执行方式下必须使用。

### 各场景最小必需参数

| 场景 | 必须使用的参数 |
|---|---|
| Eager 直接执行 | `-i`、`--plugin` |
| ACLGraph 直接执行 | `-i`、`--plugin`、`--aclgraph` |
| 保存可复用 input/golden bin | `-i`、`--plugin`、`--no-prof`、精确的 `--dump in,golden`；本测试建议同时指定 `--manual-data-dirs` |
| 使用已有 bin 执行 Eager | `-i`、`--plugin`、`--manual-data-dirs` |
| 使用已有 bin 执行 ACLGraph | `-i`、`--plugin`、`--manual-data-dirs`、`--aclgraph` |
| 只执行指定用例 | 在对应场景参数上增加 `-t` 或 `--ti` |

无论哪种模式，`PYTHONPATH` 都必须包含 `${QBSA_ASSETS}`，否则无法导入 CSV 中的 `bsa_ttk_ops.quant_block_sparse_attn`。

### 输入、配置和用例筛选

| 参数 | 必需性 | 默认值 | 含义 |
|---|---|---|---|
| `-i FILE`、`--input FILE` | CLI 必需 | 无 | 输入用例 CSV。本测试固定为 `${QBSA_MXFP8_CASES}`。 |
| `--plugin PATH` | QBSA 必需 | 无 | 加载 `${QBSA_ASSETS}` 下的 `spec.py`，提供定制输入、CPU Golden 和精度比较规则。支持逗号分隔多个路径。 |
| `--config FILE` | 可选 | TTK 默认配置链 | 指定 TTK YAML 配置文件，并覆盖默认配置、用户配置和当前目录配置。 |
| `-t NAMES`、`--testcase NAMES` | 条件必需 | 全部用例 | 按 `testcase_name` 筛选；多个名称使用逗号分隔。 |
| `--ti INDEXES`、`--testcase-index INDEXES` | 条件必需 | 全部用例 | 按 CSV 数据行的 0-based 索引筛选，支持 `0,2,4` 或 `0-9`。 |
| `--tc N`、`--testcase-count N` | 可选 | 不随机抽取 | 从筛选后的用例中随机选择 N 条；可配合 `--seed` 固定结果。 |
| `--priority RANGE` | 可选 | 全部优先级 | 按 CSV 的 `priority` 筛选，例如 `--priority=0`、`--priority=1-3`。 |
| `--op NAMES`、`--operator NAMES` | 可选 | 全部算子 | 只运行指定算子，多个名称用逗号分隔。当前 CSV 只有 QBSA。 |
| `--no-op NAMES`、`--exclude-operator NAMES` | 可选 | 不排除 | 排除指定算子。 |
| `--rerun COLUMNS` | 可选 | 关闭 | 根据结果 CSV 的失败状态列重跑，例如 `--rerun=precision_status`；多个列用逗号分隔。 |
| `--seed N`、`--random-seed N` | 可选 | 随机 | 固定 TTK 的随机筛选和默认随机输入种子。QBSA 插件自身还读取 CSV 中每条用例的 `seed`。 |

`-t`、`--ti`、`--tc` 可以用于缩小执行范围；正常情况下选择一种即可。为了保证 prepare 和 replay 对应，回放选择的每条用例必须已经生成 manual-data。

### 输入生成、Golden、比较和 dump

| 参数 | 必需性 | 默认值 | 含义 |
|---|---|---|---|
| `--input-dist uniform/normal` | 可选 | `uniform` | TTK 默认输入生成器的数据分布。QBSA 使用插件定制输入，因此主要由 CSV 属性和插件控制。 |
| `--compare METHOD` | 可选 | TestSpec 路由 | 指定比较方式：`close`、`stat_rel_err`、`cosine`、`binary`、`requant` 或 `cross_check`。本测试默认使用 `spec.py` 的 tolerance/compare。 |
| `--golden-mode Enable/Disable/Promote` | 可选；回放要求 `Enable` | `Enable` | 控制 Golden：正常生成/比较、禁用，或提升 Golden。manual-data replay 必须保持 `Enable`。 |
| `--dump [LEVELS]` | 条件必需 | 关闭；省略值时为 `full` | 保存数据。LEVELS 支持 `full`、`in`、`out`、`golden`，多个级别用逗号分隔。manual-data prepare 必须精确使用 `--dump in,golden`。 |
| `--dump-format FORMAT` | 可选 | `bin` | dump 格式：`bin`、`npy`、`pt` 或 `print`。可复用 manual-data 支持 `bin`、`npy`、`pt`；本文统一使用 `bin`。 |
| `--dump-on-fail` | 可选 | 关闭 | 仅在精度比较失败时保存完整现场数据；不能用于 `--no-prof` prepare。 |
| `--manual-data-dirs DIR [DIR ...]` | prepare 建议；replay 必需 | prepare 时可推导默认目录 | prepare 时指定唯一输出目录；普通执行时从目录加载 input/golden 并比较。回放可传多个搜索目录，但建议只传当前版本目录。 |
| `--no-prof` | 保存可复用 bin 时必需 | 关闭 | E2E 中表示只 prepare input 和 CPU Golden，不调用目标 NPU API。必须与精确的 `--dump in,golden` 搭配。 |
| `--cpu` | 可选 | 关闭 | 强制使用 CPU backend。manual-data replay 是设备执行阶段，不能与 `--cpu` 同时使用。 |

`--no-prof` prepare 不能与 `--validate`、`--dump-on-fail`、动态/静态图选项或 `--fullgraph` 组合。prepare 不区分 Eager 和 ACLGraph，同一份兼容 bin 可以用于两种执行模式。

### Eager、图模式和执行次数

| 参数 | 必需性 | 默认值 | 含义 |
|---|---|---|---|
| `--aclgraph` | ACLGraph 条件必需 | 关闭 | 使用 `torch.compile` 的 reduce-overhead ACLGraph 模式。未指定时为默认 Eager 模式。 |
| `-d[=BOOL]`、`--dynamic[=BOOL]` | 可选 | 关闭 | 开启动态 shape 图测试；可用 `-d=false` 显式关闭。它与 ACLGraph 不是同一个开关。 |
| `-c[=BOOL]`、`--const[=BOOL]` | 可选 | 关闭 | 开启静态 shape 图测试；可用 `-c=false` 显式关闭。 |
| `--fullgraph 0/1` | 可选 | `0` | 控制 `torch.compile` 是否要求完整图捕获，通常与图模式一起使用。 |
| `--warmup=true/false` | 可选 | `true` | 正式计时前是否 warmup。精度回归通常使用 `--warmup=false` 减少额外执行。 |
| `--run N` | 可选 | `3` | 每条用例正式执行次数；当前文档使用 `--run 1`。N 必须为正整数。 |
| `--task-prof=true/false` | 可选 | `true` | 是否统计 E2E 任务耗时。它不会切换 manual-data prepare；prepare 使用的是 `--no-prof`。 |
| `--deterministic-level N`、`--dl N` | 可选 | `0` | 确定性级别：`0` 关闭，`1` 检查同 NPU 多次执行 MD5 一致，`2` 强一致，`3` 跨用例 batch consistency。 |

### 设备、并发、超时和内存

| 参数 | 必需性 | 默认值 | 含义 |
|---|---|---|---|
| `--dev N`、`--device N` | 可选 | 使用全部可用设备 | 使用的设备数量，不是单个 device ID。 |
| `--device-whitelist IDS` | 可选 | 不限制 | 只使用指定设备 ID，例如 `--device-whitelist=0,1`。 |
| `--device-blacklist IDS` | 可选 | 不限制 | 排除指定设备 ID，例如 `--device-blacklist=2,3`。 |
| `--pc N`、`--process-count N` | 可选 | TTK 自动设置 | 每张设备上的 worker 进程数。QBSA/ACLGraph 建议 `--pc 1`，便于隔离显存和卡死问题。 |
| `--proc-no-reuse` | 可选 | 复用 worker | 每条用例新建进程，避免图缓存、异常状态或显存残留影响下一条用例。 |
| `--proc-timeout SECONDS` | 可选 | `0`，不超时 | 单条用例 worker 超时时间，单位秒；本文使用 `600`。 |
| `-l GB`、`--limit GB` | 可选 | `30` | 单条用例允许的 HBM 内存上限，单位 GB。 |
| `--no-memory-check` | 可选 | 开启 host 内存检查 | 跳过 TTK 执行前的主机内存检查。仅在确认风险可控时使用。 |
| `--plat SOC`、`--platform SOC` | 可选 | 自动识别 | 显式指定 SoC，例如 `Ascend910B2`；自动识别失败或离线 prepare 时可使用。 |

`--device-whitelist` 与 `--device-blacklist` 用于指定设备集合，一般选择其中一种，不建议同时使用。

### 结果、日志和校验

| 参数 | 必需性 | 默认值 | 含义 |
|---|---|---|---|
| `-o FILE`、`--output FILE` | 可选但推荐 | 输入 CSV 同目录的 `*_result.csv` | 指定结果 CSV 路径；相同路径再次运行会覆盖原结果。Eager 和 ACLGraph 应使用不同文件名。 |
| `--single-log` | 可选但推荐 | 汇总日志 | 为每条 case 生成独立的 debug/info/error/compare 等日志，便于定位卡死或精度失败。 |
| `--print=true/false` | 可选 | `true` | 是否周期性打印执行汇总。 |
| `--po FILE`、`--progress-output FILE` | 可选 | 无 | 把运行进度写入指定文件。 |
| `--title COLUMNS`、`--titles COLUMNS` | 可选 | TTK 默认结果列 | 指定结果 CSV 输出列，多个列用逗号分隔。 |
| `--csv-preserve` | 可选 | 关闭 | 在结果 CSV 中保留输入 CSV 的原始表头和字段。 |
| `--validate` | 可选 | 关闭 | 只解析并校验 CSV，不执行设备。不能与 manual-data replay 或 prepare 组合。 |
| `--provider NAME` | 可选 | TestSpec 第一个 provider | 多 provider TestSpec 中选择要测试的 backend，例如 `torch`、`tf`。当前 QBSA 通常无需指定。 |

### 文档中使用的环境变量

| 环境变量 | 必需性 | 含义 |
|---|---|---|
| `OPS_TRANSFORMER_ROOT` | 必需 | 当前 `ops-transformer` 仓库根目录。 |
| `QBSA_ASSETS` | 必需 | QBSA TTK 的 wrapper、TestSpec、输入生成和 Golden 所在目录。 |
| `QBSA_MXFP8_CASES` | 必需 | 71 条启用用例的 CSV 路径。 |
| `PYTHONPATH` | 必需 | 必须包含 `${QBSA_ASSETS}`，使 TTK 能导入 `bsa_ttk_ops`。 |
| `QBSA_RESULT_DIR` | 文档推荐 | 结果 CSV 的统一输出目录。 |
| `QBSA_MANUAL_DATA` | 保存/回放时必需 | 可复用 input/golden bin 的目录。CSV tensor 契约变化后应更换新目录或重新 prepare。 |
| `NPU_DUMP_PATH` | runtime dump 时推荐 | `--dump full` 等运行现场 dump 的输出目录；未设置时使用 TTK 默认 dump 目录。 |

## 常用执行方式

下文中的“单算子模式”指默认的 Eager 执行，“图模式”指增加 `--aclgraph` 的 ACLGraph 执行。

以下命令均在 `ops-test-kit` 仓库根目录执行。先定义公共参数、结果目录和 manual-data 目录：

```bash
COMMON_ARGS=(
  -i "${QBSA_MXFP8_CASES}"
  --plugin "${QBSA_ASSETS}"
)
RUN_ARGS=(
  --pc 1
  --proc-no-reuse
  --proc-timeout 600
  --warmup=false
  --run 1
  --single-log
)

export QBSA_RESULT_DIR=/path/to/qbsa_results
export QBSA_MANUAL_DATA=/path/to/qbsa_manual_data
mkdir -p "${QBSA_RESULT_DIR}" "${QBSA_MANUAL_DATA}"
```

`SELECTOR` 决定本次执行哪些用例，下面三种方式任选一种。

### 1. 按用例名执行单个 case

```bash
SELECTOR=(-t "NO_PSCALE_B1_BS128_MASK3_SEQ")
RUN_TAG=single_case
```

### 2. 按 CSV 中第几条执行单个 case

TTK 的 `--ti` 使用 0-based 索引，CSV 第一条数据的索引是 `0`。如果要执行人类习惯中的第 `N` 条，用 `N - 1` 作为索引：

```bash
# 示例：执行 CSV 中第 3 条，即 --ti=2
N=3
SELECTOR=(--ti=$((N - 1)))
RUN_TAG=case_${N}
```

当前 CSV 的第 1、2、3 条分别是：

```text
--ti=0  STC_B1_BS128_MIN_PAGE_EQ_MASK3
--ti=1  STC_B1_BS64_STITCH_8_BLOCKS_MASK3
--ti=2  STC_B1_BS128_TWO_S2_LOOPS_REVERSE
```

### 3. 批量执行

按名称批量执行：

```bash
SELECTOR=(-t "NO_PSCALE_B1_BS128_MASK3_SEQ,NO_PSCALE_B2_BS64_MASK0_CAUSAL")
RUN_TAG=batch_by_name
```

按索引批量执行，例如前 10 条：

```bash
SELECTOR=(--ti=0-9)
RUN_TAG=first_10
```

执行全部 71 条时不设置筛选条件：

```bash
SELECTOR=()
RUN_TAG=all_71
```

选好 `SELECTOR` 和 `RUN_TAG` 后，使用下面对应的执行命令。

### 不保存 bin，直接执行

Eager 模式：

```bash
python3 -m ttk e2e \
  "${COMMON_ARGS[@]}" \
  "${SELECTOR[@]}" \
  "${RUN_ARGS[@]}" \
  -o "${QBSA_RESULT_DIR}/${RUN_TAG}_eager.csv"
```

ACLGraph 模式：

```bash
python3 -m ttk e2e \
  "${COMMON_ARGS[@]}" \
  "${SELECTOR[@]}" \
  "${RUN_ARGS[@]}" \
  --aclgraph \
  -o "${QBSA_RESULT_DIR}/${RUN_TAG}_aclgraph.csv"
```

### 保存可复用的 input/golden bin

TTK 的可复用 manual-data 使用两阶段流程。prepare 阶段只生成 input 和 CPU Golden，不执行目标 NPU 算子；prepare 一次后，同一份 bin 可以分别用于 Eager 和 ACLGraph。

```bash
python3 -m ttk e2e \
  "${COMMON_ARGS[@]}" \
  "${SELECTOR[@]}" \
  --no-prof \
  --dump in,golden \
  --dump-format bin \
  --manual-data-dirs "${QBSA_MANUAL_DATA}"
```

CSV 的 tensor 数量、顺序、shape 或 dtype 发生变化后，旧 bin 不再兼容。此时应换一个新的 `QBSA_MANUAL_DATA` 目录，或者重新 prepare 所有准备执行的用例。

### 使用已有 bin 执行

Eager 模式回放：

```bash
python3 -m ttk e2e \
  "${COMMON_ARGS[@]}" \
  "${SELECTOR[@]}" \
  "${RUN_ARGS[@]}" \
  --manual-data-dirs "${QBSA_MANUAL_DATA}" \
  -o "${QBSA_RESULT_DIR}/${RUN_TAG}_eager_replay.csv"
```

ACLGraph 模式回放：

```bash
python3 -m ttk e2e \
  "${COMMON_ARGS[@]}" \
  "${SELECTOR[@]}" \
  "${RUN_ARGS[@]}" \
  --manual-data-dirs "${QBSA_MANUAL_DATA}" \
  --aclgraph \
  -o "${QBSA_RESULT_DIR}/${RUN_TAG}_aclgraph_replay.csv"
```

prepare 和 replay 必须使用相同的 CSV 和兼容的插件代码，且 replay 选择的每条用例都必须已经 prepare。回放时建议只传入当前数据目录；如果同时传入新旧目录，TTK 可能优先命中不兼容的旧数据。

### 执行 NPU 的同时保存现场 dump

如果需要一次执行中保存 input、device output 和 golden，可以使用 `--dump full`。这种现场 dump 用于问题定位，不等同于可通过 `--manual-data-dirs` 回放的 manual-data。

```bash
export NPU_DUMP_PATH=/path/to/qbsa_runtime_dump
mkdir -p "${NPU_DUMP_PATH}"

# Eager；ACLGraph 模式在命令中再加 --aclgraph
python3 -m ttk e2e \
  "${COMMON_ARGS[@]}" \
  "${SELECTOR[@]}" \
  "${RUN_ARGS[@]}" \
  --dump full \
  --dump-format bin \
  -o "${QBSA_RESULT_DIR}/${RUN_TAG}_runtime_dump.csv"
```

`PYTHONPATH` 必须包含 `tests/assets`，因为 CSV 的 `api_name` 是 `bsa_ttk_ops.quant_block_sparse_attn`；`--plugin` 则用于发现同目录下的 `spec.py`。

`return_softmax_lse=False` 时，PTA 返回的第二个输出是 shape `()` 的无效占位 Tensor。TTK Golden 将该槽位保存为 `None` marker 并跳过数值比较；已有 manual-data 目录需要重新 prepare 才会生成新的 `golden_1_none.bin`。

## 重新生成 CSV

在 `ops-transformer` 仓库根目录执行：

```bash
python3 experimental/attention/quant_block_sparse_attn/tests/ttk/bsa_fullquant_mxfp8_test/generate_ttk_cases.py
python3 experimental/attention/quant_block_sparse_attn/tests/ttk/bsa_fullquant_mxfp8_test/generate_ttk_cases.py --check
```

生成器只读取 `qbsa_mxfp8_test_cases.py`，并只输出其中 `enable=True` 的用例；当前共 71 条。生成结果把 pytest 的 Q/KV 有效长度转换为 `cu_seqlens_q_value` 和 `seqused_kv_value`，并保留 page block size、稀疏模式、数据范围、随机种子和 LSE 开关。

`qbsa_mxfp8_factor_coverage_report.xlsx` 汇总 STC/Full 的单因子、关键交互、有效长度为 0、空稀疏预算和硬约束覆盖情况。
