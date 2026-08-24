# MegaMoe 批跑脚本使用说明

`mega_moe_once.py` 用于从 Excel 读取多条 MegaMoe 用例，生成或复用输入与 Golden，依次执行小算子级联和融合 MegaMoe，并保存精度与性能结果。

## 1. 运行方式

进入批跑目录：

```bash
cd mega_moe/batch_run_once
```

执行指定 Sheet 和 Excel 行范围：

```bash
python3 mega_moe_once.py \
    -excel test_mega.xlsx \
    -sheet test_mega \
    -s 2 \
    -e 10 \
    -m 0 \
    -o result \
    --precision-report
```

Excel 第1行为表头，`-s 2` 表示从第一个 case 开始；开始行和结束行均包含在执行范围内。

只生成输入和 Golden，不执行 NPU：

```bash
python3 mega_moe_once.py \
    -excel test_mega.xlsx \
    -sheet test_mega \
    -s 2 \
    -e 10 \
    -o result \
    -g
```

### 命令行参数

| 参数 | 含义 |
| --- | --- |
| `-excel`, `--excel` | Excel 用例文件路径，必填 |
| `-sheet`, `--sheet` | Sheet 名称，必填 |
| `-s`, `--start_line` | 起始 Excel 行号，必填 |
| `-e`, `--end_line` | 结束 Excel 行号，必填且包含该行 |
| `-m`, `--run_mode` | `0`：相同 die 数的 case 复用通信域；`1`：每条 case 单独创建通信域 |
| `-t`, `--tag` | 兼容性标签参数，默认 `rdv`；当前脚本仅解析，未参与结果路径命名 |
| `-o`, `--output_dir` | 性能、CSV 和编译缓存的输出根目录，默认当前目录 |
| `-g`, `--golden` | 只生成输入与 Golden，跳过 NPU 执行 |
| `--precision-report` | 保存每个 rank 的 Golden、级联输出、MegaMoe 输出，并生成交互式精度 HTML |

如果更希望直接修改脚本而不是传参数，可以编辑 `mega_moe_once.py` 顶部的：

```python
PRECISION_REPORT_ENABLE = True  # 默认生成精度报告
PRECISION_DIFF_THD = 0.001      # 原脚本逐元素展示阈值
PRECISION_PCT_THD = 0.001       # 原矩阵通过比例阈值
```

设置 `PRECISION_REPORT_ENABLE = True` 后，运行命令不再需要添加
`--precision-report`。

启用精度报告后，每个 rank 会生成：

```text
result/precision_reports/<case_name>/rank_<rank>/
├── precision_data.npz     # expected、actual、cascade 原始矩阵
├── summary.json           # 原脚本综合指标与 Top 退化位置
└── precision_report.html  # 无外部依赖，可直接用浏览器打开
```

HTML 顶部直接使用原脚本 `checkResult(...).check_result(...)` 的 PASS/FAIL，
并逐项展示 MegaMoe 指标、Cascade 基线、允许倍率和判定结果。热力图使用原脚本
`data_compare_triangle` 的定位口径，只标识
`abs(MegaMoe-Golden) > abs(Cascade-Golden)` 的退化位置。脚本 PASS 时使用黄色，
只有综合判定 FAIL 时才使用红色。悬浮可查看同一坐标的 Golden、Cascade、
MegaMoe 和 MegaMoe 自身的判定误差。颜色强度不使用
`MegaMoe误差 / Cascade误差`，避免 Cascade 误差为零时把微小噪声放大。
此外，`MegaMoe == 0` 且 Golden、Cascade 都非零的位置会作为“疑似输出清零”
单独统计，并始终以亮青色标出，不受普通误差阈值和色阶影响。

报告还会同时生成一张独立的固定阈值视图，使用 `PRECISION_DIFF_THD` 和
`cal_relative_diff_np` 口径展示 `MegaMoe vs Golden` 的逐元素异常。该视图固定
使用紫色，不参与原脚本的综合 PASS/FAIL。完整矩阵保留在
`precision_data.npz` 中。

两张热力图都支持交互缩放：鼠标滚轮以指针位置为中心缩放，按住鼠标拖拽平移，
双击或点击“重置”恢复全矩阵；工具栏会实时显示当前 token 行和 hidden 列范围。
对于常见的少量 token × 4096 hidden 矩阵，报告保留逐元素网格；只有矩阵过大时
才按区域聚合，以限制单个 HTML 的大小。

也可以对已有 `.npz` 离线重新生成报告：

```bash
python3 precision_report.py precision_data.npz \
    -o regenerated_report \
    --case-name my_case \
    --rank 0 \
    --tolerance 0.001
```

怀疑输出整体或局部变成 Golden 的相反数时，可以离线分析报告目录：

```bash
python3 analyze_sign_flip.py \
    result/precision_reports/<case_name>/rank_0
```

也可以直接传入 `precision_data.npz`。脚本会在同一目录生成：

```text
sign_flip_analysis.json  # 完整统计、自动归因和相反数集中区间
sign_flip_samples.csv    # 最明显的相反数坐标及三方原始值
```

终端中的 `结论代码` 会区分报告映射问题、MegaMoe 算子/Combine 路径问题、
Golden 或共同上游问题，以及非全局相反数的普通误差。

## 2. 可变 BS

Excel 必须包含 `is_variable_bs` 和 `rank_bs` 两列。`rank_bs` 列允许留空。

### 固定 BS

```text
bs=16
is_variable_bs=0
num_max_tokens_per_rank=0
rank_bs=<空>
```

每个 die 都使用 `bs=16`。

### 随机可变 BS

```text
bs=16
is_variable_bs=1
num_max_tokens_per_rank=16
rank_bs=<空>
```

沿用原有随机 mask 路径：

- 每个 die 的真实 BS 在有效范围内随机生成。
- 至少一个 die 会被设置为 `bs`，用于覆盖最大 BS 上界。
- 根据 mask 裁剪 `x`、`expert_ids`、`expert_scales` 和 Golden 输出。

### 指定每个 die 的 BS

两 die 示例：

```text
bs=16
is_variable_bs=1
num_max_tokens_per_rank=16
rank_bs=16,5
```

此时：

```text
die0实际BS=16
die1实际BS=5
```

四 die 示例：

```text
rank_bs=16,5,9,2
```

配置要求：

- `rank_bs` 的元素个数必须等于 `ep_world_size`，当前批跑配置下通常等于 `rank_per_dev`。
- 每个值必须满足 `1 <= rank_bs[i] <= bs`。
- `rank_bs` 只接受整数，使用英文逗号分隔。
- `bs` 仍表示 `num_max_tokens_per_rank` 的上界。
- 小算子和 MegaMoe 在同一个 die 上接收相同的真实 BS。
- Prefill 的 AllGather/ReduceScatter 会在通信内部临时补齐，通信完成后再裁回真实 BS。

当 `is_variable_bs` 或 `rank_bs` 改变时，Golden 缓存元数据会变化，对应用例会重新生成输入和 Golden。

## 3. Excel 参数

除特别注明外，下表字段必须存在且不能为空。

### Shape 与专家配置

| 列名 | 含义 |
| --- | --- |
| `case_name` | 用例名称，同时用于结果和 Profiling 子目录命名 |
| `bs` | 每个 die 的最大 token 数；固定 BS 时也是实际 token 数 |
| `h` | Hidden size |
| `n` | FFN intermediate size |
| `k` | 每个 token 路由的 MoE 专家数 |
| `rank_per_dev` | 当前设备使用的 die 数 |
| `local_moe_expert_num` | 每个 die 部署的 MoE 专家数 |
| `shared_expert_num_per_rank` | 每个 die 的共享专家数，必须大于等于0 |
| `max_recv_token_num` | 通信接收 token 上界 |
| `num_max_tokens_per_rank` | 对称通信缓冲区的单 die 最大 token 数；固定 BS 推荐填 `0`，可变 BS 推荐填 `bs`，批跑以该列实际值为准 |
| `is_variable_bs` | 是否开启可变 BS，`0` 或 `1` |
| `rank_bs` | 各 die 实际 BS 列表；列必须存在，但随机可变 BS 时可以留空 |

### 执行模式

| 列名 | 含义 |
| --- | --- |
| `moe_stage` | `prefill` 或 `decode` |
| `graph_type` | `0` 为单算子；`3` 为 ACLGraph；以当前算子支持范围为准 |
| `is_performance` | `1` 采集性能，`0` 执行精度流程 |
| `loop_acc_time` | 非性能模式连续执行轮数，必须大于等于1 |
| `seed` | 输入、路由和权重的随机种子 |
| `is_topk_balance` | `1` 生成均衡路由；`0` 生成热点专家不均衡路由 |
| `topk_weights_type` | MegaMoe top-k weight 模式开关 |

### 量化与权重

| 列名 | 含义 |
| --- | --- |
| `quant_mode` | Dispatch 量化模式，当前主要使用 MX 量化模式 `4` |
| `quant_mode_type` | Dispatch 输出类型，如 `float8_e4m3fn`、`float8_e5m2`、`float4_e2m1` |
| `swiglu_quant_dtype` | 激活后量化类型 |
| `weight_dtype` | 权重类型 |
| `combine_quant_mode` | Combine 通信量化模式：`0`、`3` 或 `4` |
| `A8W8_is_NZ` | A8W8 权重是否使用 NZ 格式 |
| `weight_is_tensor_list` | MegaMoe 权重是否按专家拆成 TensorList |
| `weight_scale1` | GMM1 scale 配置兼容字段，当前仍要求填写 |
| `weight_scale2` | GMM2 scale 配置兼容字段，当前仍要求填写 |

### 激活函数

| 列名 | 含义 |
| --- | --- |
| `activation_type` | `swiglu`、`swiglustep`、`swigluoai` 或 `situ` |
| `clamp_limit` | SwiGLU clamp 上限；填写 `-1` 表示不裁剪 |
| `situ_beta` | SiTuGLU 的 `beta` |
| `situ_linear_beta` | SiTuGLU 的 `linear_beta` |
| `swiglu_alpha` | 仅 `swigluoai` 使用，典型值 `1.702` |
| `swiglu_beta` | 仅 `swigluoai` 使用，典型值 `1.0` |

`swiglu_alpha` 和 `swiglu_beta` 只在 `activation_type=swigluoai` 时要求填写。`is_interleaved` 是兼容性可选列，缺省按 `0` 处理。

## 4. 执行流程

每批 case 按以下顺序执行：

1. 读取并打印 Excel 参数。
2. 检查 Golden 缓存文件和 `metadata.json`。
3. 缓存不匹配时重新生成输入和 Golden。
4. 按当前 die 重建并搬运对应权重。
5. 执行小算子级联路径。
6. 释放小算子模型和中间显存。
7. 执行融合 MegaMoe。
8. 完成精度比较；性能用例同时保存 Profiling 并更新汇总表。

`run_mode=0` 会将 `rank_per_dev` 相同的 case 放在同一通信域中执行。不同 `rank_per_dev` 的 case 会拆分到不同批次，避免按照错误的 world size 启动进程。

## 5. 输出目录

假设使用：

```bash
-o result
```

主要输出如下：

```text
result/
├── csv_result/
│   ├── mega_moe_result.csv
│   ├── moe_distribute.csv
│   └── profiling.xlsx
├── Profiling_Result_mc2/
│   └── <case_name>/
│       ├── cascade/
│       └── mega_moe/
└── torch_extensions_cache/
```

输入与 Golden 使用相对路径保存：

```text
./input/<sheet>/<case_name>/
├── x.pt
├── expert_ids.pt
├── expert_scales.pt
├── x_active_mask.pt
├── x_final.pt
├── expert_token_nums.pt
├── topk_weights_type.pt
└── metadata.json
```

因此建议始终从 `mega_moe/batch_run_once` 目录启动脚本，避免输入缓存落到其他工作目录。

## 6. 批量调度脚本

`send_case_daily.sh` 用于按 Sheet 和批次调度 `mega_moe_once.py`，支持指定 Excel、Sheet、行范围、输出目录、批大小和批次间隔。

示例：

```bash
bash send_case_daily.sh \
    -e mega_moe_daily_rdv.xlsx \
    -s a8w8_value \
    --start-row 2 \
    --end-row 100 \
    -o daily_result \
    -b 5 \
    -i 30
```

查看完整参数：

```bash
bash send_case_daily.sh --help
```

## 7. 常见问题

### Excel 缺列或单元格为空

脚本会对主参数进行必填校验。缺少列时会报：

```text
Excel missing required column: <column>
```

`rank_bs` 是唯一允许值为空的新增必备列。

### 修改参数后仍命中旧 Golden

Golden 缓存通过 `metadata.json` 比较输入参数。`is_variable_bs`、`rank_bs`、shape、dtype、随机种子或激活参数发生变化时，会重新生成对应缓存。

### 性能模式使用 graph_type=0

当前脚本不支持在单算子模式下采集整图性能。性能用例应使用受支持的图模式。

### rank_bs 长度不匹配

例如 `ep_world_size=4` 时：

```text
rank_bs=16,5
```

会被拒绝，应填写四个整数，例如：

```text
rank_bs=16,5,9,2
```
