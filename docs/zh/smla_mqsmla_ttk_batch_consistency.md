# SMLA、QSMLA、MQSMLA TTK Batch 一致性测试指南

本文面向拿到算子源码、`ops-test-kit`、`ttk_data` 和匹配自定义算子包的测试人员，说明如何设计、执行和判断
`sparse_flash_mla`（SMLA）、`quant_sparse_flash_mla`（QSMLA）与
`mixed_quant_sparse_flash_mla`（MQSMLA）的 batch 一致性测试。文中的路径均为变量，不依赖某台机器的目录。

Batch 一致性不是普通 CPU Golden 精度测试的替代品。一条 case 必须先通过 pytest 同源的 NPU/Golden compare，
再证明被声明为同一 relation 的 NPU output 原始字节一致。

## 1. 验证目标和实现分层

当前交付同时验证两类关系：

1. **同 case 一致性**：一次算子调用的 output 中，两个 batch 切片使用相同 relation 输入。assets custom
   compare 在普通精度通过后，直接比较这两个 NPU output 切片的 shape、dtype 和原始 storage 字节。
2. **跨 case 一致性**：B、切片位置或完整 case 不同，但 relation 的 seed、切片长度和业务上下文相同。
   TTK 用 `--dump out` 保存每条 case 的 NPU output，外置脚本在全部 case 结束后按 relation 分组比较。

完整数据流如下：

```text
CSV 显式参数
  -> TTK 解析 batch_axis / batch_slice_info / batch_seed
  -> TTK 生成当前 case 的 batch_consistency_id
  -> assets input 调用原 pytest input，并让 relation 切片使用确定性随机源
  -> NPU 主算子
  -> 原 pytest compare（普通精度）
  -> assets compare（同 case output 字节一致性）
  -> TTK result + --dump out
  -> compare_batch_outputs.py（跨 case output 字节一致性）
```

三层职责不能互相代替：

| 层次 | 负责内容 | 不负责内容 |
| --- | --- | --- |
| TTK | 解析字段、生成 case 内 ID、调度、保存 result/output | 不自动推导哪些红线 case 应属于同一关系 |
| 算子 assets | pytest input/Golden/compare、确定性 relation input、同 case比较 | 不保存跨进程全局输出，不猜缺失 CSV 参数 |
| 外置脚本 | 读取 CSV/result/bin，校验跨 case relation、覆盖和上下文 | 不运行算子，不生成输入，不放宽数值精度 |

## 2. 前置条件

### 2.1 平台和算子包

| 算子 | 当前真机平台 | 当前 batch CSV | 说明 |
| --- | --- | --- | --- |
| SMLA | A3（`ascend910_93`）/A5 | `Ascend950/e2e/test_sparse_flash_mla_batch_consistency.csv` | 同一正式表可在 A3 完成全流程 |
| QSMLA | A5（`ascend950`） | `Ascend950/e2e/test_quant_sparse_flash_mla_batch_consistency.csv` | A3 不能替代 A5 主算子验收 |
| MQSMLA | A5（`ascend950`） | `Ascend950/e2e/test_mixed_quant_sparse_flash_mla_batch_consistency.csv` | A3 只能做静态/CPU 侧检查 |

自定义包必须与待测源码兼容，并同时包含主算子和 metadata 算子。示例：

```bash
source /path/to/cann_env.sh

OPS_TRANSFORMER=/path/to/ops-transformer
SOC=ascend910_93                 # A5 改为 ascend950
INSTALL_ROOT=/path/to/isolated_custom_opp

cd "$OPS_TRANSFORMER"
bash build.sh --pkg --soc="$SOC" --vendor_name=custom \
  --ops=sparse_flash_mla,sparse_flash_mla_metadata,\
quant_sparse_flash_mla,quant_sparse_flash_mla_metadata,\
mixed_quant_sparse_flash_mla,mixed_quant_sparse_flash_mla_metadata \
  -j16

./build/cann-ops-transformer-custom_linux-aarch64.run \
  --quiet --install-path="$INSTALL_ROOT"
source "$INSTALL_ROOT/vendors/custom_transformer/bin/set_env.bash"
```

构建日志中的 SoC 必须与执行机器一致。不要把源码仓下的 `torch_extension` 加入 `PYTHONPATH` 来冒充已安装
算子包；Python 注册、op-api、metadata 和 kernel 必须来自同一安装产物。

### 2.2 公共路径

```bash
source /path/to/cann_env.sh
source /path/to/custom_opp/vendors/custom_transformer/bin/set_env.bash

TTK_FRAMEWORK=/home/j60100428/ttk/ops-test-kit
OPS_TRANSFORMER=/home/j60100428/ttk/ops-transformer_0728_2
TTK_DATA=/home/j60100428/ttk/ttk_data
OP=quant_sparse_flash_mla
CSV="$TTK_DATA/Ascend950/e2e/test_${OP}_batch_consistency.csv"
ASSETS="$OPS_TRANSFORMER/attention/$OP/tests/assets"

export PYTHONPATH="$TTK_FRAMEWORK:$ASSETS:${PYTHONPATH:-}"
```

QSMLA/MQSMLA 在 A5 分别把 `OP` 和 `CSV` 换成上一节表格中的值。每次执行使用新的输出目录，不能把旧
output bin 混入本轮结果。`Ascend910B3` 下的 SMLA 表可作 A3 历史兼容验证，但不替代上表正式扩展矩阵。

## 3. CSV 字段语义

### 3.1 三个输入字段

TTK 输入 CSV 新增三个顶层字段。三个字段的最外层长度必须等于 `tensor_view_shapes` 的 tensor slot 数量，
同一位置永远描述同一个 slot。

| 字段 | 含义 | 当前三算子约束 |
| --- | --- | --- |
| `batch_axis` | 每个 tensor slot 中哪些轴随 batch 变化 | 所有 batch-shaped slot 都应标出；Q slot 0 为 `(0,)` |
| `batch_slice_info` | 哪些切片被声明为 relation 样本 | 当前只允许 Q slot 0、axis 0；其他 slot 必须为 `None` |
| `batch_seed` | relation 样本使用的确定性 seed | 嵌套结构与 slice 完全一致；其他 slot 必须为 `None` |

以 B=2、Q 位于 tensor slot 0 为例：

```python
batch_axis = (
    (0,),       # q: axis 0 是 batch
    (0,),       # ori_kv: axis 0 也随 batch 变化
    (0,),       # cmp_kv
    None,       # 非 batch tensor 或空 optional slot
    # ...其余 slot 按真实 shape 填 (0,) 或 None
)

batch_slice_info = (
    (((0, 1, 1), (1, 2, 1)),),  # q.axis0 的两个 relation 样本
    None,
    None,
    # ...其余 slot 全为 None
)

batch_seed = (
    ((74123, 74123),),           # 两个样本属于同一 relation
    None,
    None,
    # ...其余 slot 全为 None
)
```

括号层级依次是：

```text
tensor slot -> batch axis -> 该 axis 下的多个 slice/seed
```

`(start, stop, step)` 使用左闭右开切片。当前 assets 只接受 `step=1`、非空、非负且不越过 Q 的 axis 0。
BSND 下该轴就是 B；TND 下该轴是 token 轴 T，`start/stop` 必须同时出现在显式
`cu_seqlens_q_values` 中，即只能选择完整逻辑 batch，不能选择半条 sequence。assets 先把 token 边界映射为
逻辑 batch，再将同 relation seed传给 pytest 随机生成链；PA 下还会通过 block table校验对应物理 KV block。
一个 case 内的 relation 目前只能使用一个 seed；要测试另一 relation，使用另一组 case 或另一 seed。

`batch_axis` 不能只给 Q 填 `(0,)` 后忽略其他实际带 B 轴的 tensor。跨 case 脚本会用它和
`tensor_view_shapes`、attributes 构建去 batch 后的业务上下文；错误标记可能把不等价输入误分组，或直接触发
上下文不一致。

### 3.2 `batch_consistency_id` 不写入输入 CSV

`batch_consistency_id` 是 TTK 根据当前 case 的 `batch_axis`、slice 绝对位置和 seed 自动生成的结果字段，并
作为 kwargs 传给 custom compare。以 B=2 示例为例，内部 ID 等价于：

```python
((('74123_0_0_1_1', '74123_0_1_2_1'),),)
```

assets 会核对该 ID 与输入字段是否一致，以发现字段错位。测试人员不得手填或复制
`batch_consistency_id`。由于它包含当前 case 的绝对 start/stop，B=2 与 B=3 的 TTK ID 可以不同；跨 case
工具使用稳定 relation key：

```text
(output_index, output_batch_axis, seed, slice_length)
```

当前即 `(0, 0, 74123, 1)`。因此跨 case 能比较位于不同 batch 位置、但 seed 和长度相同的输出。

### 3.3 `batch_deterministic_level`

`batch_deterministic_level=3` 是三算子 wrapper 的业务 attribute，写在 `attributes` 字典内：

```python
attributes = {
    # 原红线 case 的完整属性
    'batch_deterministic_level': 3,
}
```

它不是 TTK 顶层 CSV 字段。缺失、写成其他 level，或安装包不支持 level 3，都不能宣称 batch 一致性通过；
不得自动降级到 level 0/1 后报 PASS。

### 3.4 seed、长度和上下文分组规则

跨 case 脚本只有在以下条件全部满足时才比较两个样本：

1. `output_index` 和 output batch axis 相同；当前均为 output 0、axis 0；
2. seed 相同；
3. slice 长度 `stop - start` 相同；
4. output 去 batch 后的 shape 和 dtype 相同；
5. 输入去 batch 后的 shape、dtype、data range 和业务 attributes 相同；
6. 长度为 B/B+1 的 per-batch attributes 按当前 slice 归一化后相同。

相同 seed 但长度分别为 1、2 会形成两个 relation，不能比较。相同 seed/长度但 layout、mask、TopK 或有效长度
不同会报告 `input context differs`，不能靠关闭检查强行归组。seed 是可复现关系标识，不是“任意两个 case
输出应相同”的承诺。

## 4. 从红线 case 构造 B=2/B=3 示例

普通 `ttk_data/csv_tools` Excel 转换器只负责普通 E2E 参数链，当前不会自动设计 batch relation。新增 batch
CSV 时必须先由算子/测试 owner 证明 relation 的输入语义等价，再人工或使用专门脚本构造。

QSMLA 当前示例基于红线 `ORI_CMP_SPARSE` case `000488`。构造过程如下：

1. 复制红线 case 的 dtype、layout、mask、`S1=14`、`S2=8192`、`cmp_ratio=50`、TopK、data range 和
   sparse mode，不改变功能特性；
2. 生成 B=2 与 B=3 两条 case；所有 BSND tensor 的 axis 0、per-batch tensor 和 list attribute 同步扩展；
3. `cu_seqlens_*` 使用 B+1 项，`seqused_*` 使用 B 项，值保持每个 batch 语义相同；
4. 所有实际 batch-shaped tensor 在 `batch_axis` 对应 slot 标 `(0,)`；
5. relation 只声明在 Q slot 0：

```python
# B=2：比较 batch 0 和 batch 1
batch_slice_info = ((((0, 1, 1), (1, 2, 1)),), None, ...)
batch_seed       = (((74123, 74123),), None, ...)

# B=3：比较 batch 0 和 batch 2
batch_slice_info = ((((0, 1, 1), (2, 3, 1)),), None, ...)
batch_seed       = (((74123, 74123),), None, ...)
```

这两条 case 各自提供两个同 case 样本，合计四个 `(0, 0, 74123, 1)` 跨 case样本。B=3 中未参与 relation
的 batch 1 仍是合法完整输入，但不会进入一致性比较。

调整其他红线 case 时逐项检查：

- 所有 B/B+1 shape 和 list 是否同步；
- CSV slot 与 pytest 实际返回 tensor 是否仍双 `None` 或双有值；
- `template_run_mode`、TopK、sparse indices、layout 是否保持源 case 的显式值；
- relation 样本的 Q/KV/mask/有效长度是否真正等价；
- `batch_deterministic_level` 是否为 3；
- 至少两个 case 是否共享同一个 `(seed, slice_length)`，否则无法覆盖跨 case。

当前正式表不是只放两条示例，覆盖如下：

| 算子 | case数 | BSND | TND/TND | TND/PA | 跨case relation组 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SMLA | 13 | 9 | 2 | 2 | 4 |
| QSMLA | 6 | 2（红线 `000488` 派生） | 2 | 2 | 3 |
| MQSMLA | 13 | 9 | 2 | 2 | 4 |

每条 case 都含两个同 seed、同长度 slice，先覆盖同 case；每组至少两条 case共享 seed和 slice长度，再覆盖
跨 case。TND 小矩阵使用显式 prefix和有效长度，PA 小矩阵额外覆盖不同 relation batch指向物理 block 后的
逐字节一致性。QSMLA/MQSMLA 的 TND/PA 只能在 A5完成主算子验收，A3 的 prepare PASS 不能写成 NPU PASS。

TND 的 slice写 token边界，不写逻辑 batch编号。例如 Q shape为`(6, N, D)`且
`cu_seqlens_q_values=[0, 2, 4, 6]`时，要比较逻辑 batch 0和2，应写：

```python
batch_slice_info = ((((0, 2, 1), (4, 6, 1)),), None, ...)
batch_seed = (((76001, 76001),), None, ...)
```

`0/2/4/6`必须都是 prefix边界，两个 slice均含2个token；assets会将它们映射为逻辑batch 0和2，再校验
对应`seqused_*`、KV prefix和residual语义一致。PA的KV axis 0是物理block，不可直接复制同编号block；应保留
pytest生成的block table，并由adapter把relation对应逻辑KV写入各自block table指向的物理block。

## 5. 执行流程

### 5.1 先做静态校验

```bash
OUT=$(mktemp -d "${TMPDIR:-/tmp}/ttk_${OP}_batch.XXXXXX")

python3 -m ttk e2e \
  -i "$CSV" \
  --plugin "$ASSETS" \
  --validate \
  -o "$OUT/validate_result.csv"
```

不仅检查命令退出码，还要检查 `validate_result.csv` 每条 case 都 valid。`API_PARSE_FAIL` 通常表示 assets 未加入
`PYTHONPATH`、wrapper 无法导入或算子 Python 包未注册，不等于 CSV relation 数值错误。

### 5.2 快 case：单阶段只保存 NPU output

跨 case 只需要 NPU output 时，不必生成 input 和 Golden 文件：

```bash
export NPU_DUMP_PATH="./npu_output"
mkdir -p "$NPU_DUMP_PATH"

python3 -m ttk e2e \
  -i "$CSV" \
  --plugin "$ASSETS" \
  --dump out \
  -o "./result.csv"

python3 "$ASSETS/compare_batch_outputs.py" \
  --csv "$CSV" \
  --result "./result.csv" \
  --dump-dir "$NPU_DUMP_PATH" \
  --report "./batch_report.json" \
  --require-intra-case \
  --require-cross-case
```

`--dump out` 只在设备执行后保存 output。需要额外审计 input/Golden 时才使用 `--dump full`；不要为跨 case
比较无条件生成不需要的数据。

### 5.3 慢 case：prepare/replay 分阶段

分阶段适用于 CPU input/Golden 生成与 NPU 执行需要分时、迁移目录或复用准备数据的场景。

```bash
MANUAL_DATA_DIR="$OUT/manual_data"
export NPU_DUMP_PATH="$OUT/replay_output"
mkdir -p "$MANUAL_DATA_DIR" "$NPU_DUMP_PATH"

# prepare：不执行 NPU，只保存 input 和 Golden
python3 -m ttk e2e \
  -i "$CSV" \
  --plugin "$ASSETS" \
  --no-prof \
  --dump in,golden \
  --dump-format bin \
  --manual-data-dirs "$MANUAL_DATA_DIR" \
  -o "$OUT/prepare_result.csv"

# replay：读取准备数据、执行 NPU、正常 compare，只保存本轮 output
python3 -m ttk e2e \
  -i "$CSV" \
  --plugin "$ASSETS" \
  --manual-data-dirs "$MANUAL_DATA_DIR" \
  --dump out \
  -o "$OUT/replay_result.csv"

python3 "$ASSETS/compare_batch_outputs.py" \
  --csv "$CSV" \
  --result "$OUT/replay_result.csv" \
  --dump-dir "$NPU_DUMP_PATH" \
  --report "$OUT/batch_report.json" \
  --require-intra-case \
  --require-cross-case
```

prepare 不产生 NPU output；replay 不重新调用 input/Golden plugin。CSV、assets、算子包或 Golden 逻辑变化后必须
重新 prepare，不能复用旧目录伪造回归。

### 5.4 两个覆盖强制选项

- `--require-intra-case`：每个 relation 在每个参与 case 中至少有两个切片，并要求 result 的
  `eager_precision` 包含 `batch_intra=PASS`。
- `--require-cross-case`：每个 relation 至少出现在两个不同 testcase 中。

它们只拒绝覆盖不足的报告，不创建 relation、不修改 seed、不运行 compare，也不会把普通 case 自动变成 batch
case。交付回归建议同时启用；仅调试一条 case 时可以临时去掉 cross 要求，但该结果不能作为跨 case验收。

## 6. PASS 判据和结果读取

最终 PASS 必须同时满足：

1. TTK result 中每条参与 case 的 `precision_status=PASS`；
2. 每条参与 case 的 `eager_precision` 含 `batch_intra=PASS`；
3. `compare_batch_outputs.py` 退出码为 0；
4. `batch_report.json` 顶层 `status=PASS`；
5. report 中每个 relation 的 `status=PASS`，且 `case_count>=2`、每个 case 至少两个样本；
6. 正式验收使用目标 SoC 和匹配 custom OPP，未发生 level 降级。

report 中每个样本记录 testcase、slice、shape、dtype 和 SHA-256。脚本实际比较原始字节，SHA-256 只用于审计和
快速定位，不代替字节比较。

## 7. 常见失败定位

| 现象 | 所属阶段 | 优先检查 |
| --- | --- | --- |
| `API_PARSE_FAIL` | validate | `PYTHONPATH` 是否含 assets；wrapper/安装包是否可导入 |
| `batch_config=FAIL` | custom compare | 三字段是否成套、嵌套层级和 slot 数是否正确；TTK ID 是否由当前字段生成 |
| `batch_intra=FAIL` | 同 case | 同 seed slice 的输入是否真等价；NPU output 是否逐字节不同 |
| `missing output dump` | 跨 case | replay/单阶段是否带 `--dump out`；`NPU_DUMP_PATH` 是否为本轮目录 |
| `output byte count ...` | 跨 case | CSV output shape/dtype 是否与真实 output 契约一致 |
| `input context differs` | 跨 case | layout、mask、TopK、有效长度、range、去 batch shape 是否不同 |
| `raw output differs` | 跨 case | 同 relation 的 NPU 原始 storage 不一致；不能改成容差 compare 掩盖 |
| relation 少于两个 case | 覆盖 | 是否有另一 case 使用相同 seed 和 slice 长度 |
| SoC/metadata/tiling 拦截 | NPU | 自定义包 SoC、注册范围和 feature 是否匹配，不能归因于 CSV parser |

排错顺序固定为 `validate -> input/Golden -> 普通精度 -> 同 case -> dump 完整性 -> 跨 case上下文 -> 原始输出`。
不要在 inputs.py 内补默认值、跳过 slot 矛盾或放宽 compare 来绕过上游失败。

## 8. 当前支持边界

三算子公共边界：

- `layout_q/layout_kv=BSND/BSND`、`TND/TND`、`TND/PA_BBND`；
- relation 只声明在 Q tensor slot 0、axis 0；
- slice 必须连续且 `step=1`；TND slice必须与`cu_seqlens_q_values`完整边界对齐；
- 同 case 只支持一个 relation seed；
- 只比较主输出 output 0 的 batch axis 0；
- 比较原始 storage 字节，不使用数值容差；
- 普通无 batch 字段的 case 仍完全走原 pytest 路径，不应被 batch adapter 改写。

算子专项边界：

| 算子 | 已支持 | 当前拒绝 |
| --- | --- | --- |
| SMLA | SWA、HCA；BSND/TND/PA，现成交付为 HCA | CSA/ORI sparse类 batch模式 |
| MQSMLA | SWA、HCA，`quant_mode=1`；BSND/TND/PA，现成交付为 HCA | sparse模式、其他 quant mode |
| QSMLA | `quant_mode=1`；SWA/HCA/ORI_SPARSE/ORI_CMP_SPARSE；BSND/TND/PA；full sparse、`fullK/no` TopK、均匀`seqused_q` | CSA、random sparse、random TopK、非均匀`seqused_q` |

QSMLA output 0 的正式 dtype 是 `bfloat16`，即使 Q 输入为 `uint8`；dump 大小必须按 output 契约计算。

扩展 layout、输入轴、多个 seed、非连续 slice 或其他 output 时，必须同步修改 input 确定性逻辑、同 case
compare、跨 case脚本、CSV 样例和正反白盒测试，不能只改 CSV。

## 9. 交付矩阵

每次修改 batch 代码或 CSV 后至少执行：

| 项目 | SMLA/A3 | QSMLA/A5 | MQSMLA/A5 |
| --- | --- | --- | --- |
| input/compare/output-bin 白盒 | 必须 | 必须 | 必须 |
| TTK `--validate` 全表 | 必须 | 必须 | 必须 |
| 普通 pytest 同源精度 | 必须 | 必须 | 必须 |
| 单阶段 `--dump out` | 必须 | 必须 | 必须 |
| 同 case `batch_intra=PASS` | 必须 | 必须 | 必须 |
| 跨 case report PASS | 必须 | 必须 | 必须 |
| prepare/replay + `--dump out` | 至少小矩阵 | 至少小矩阵 | 至少小矩阵 |
| 错误 seed/长度/缺 bin/上下文不等价反例 | 白盒必须 | 白盒必须 | 白盒必须 |

当前交付证据为：三算子正式表静态 validate 分别 `13/13`、`6/6`、`13/13`；新增 TND/TND 和 TND/PA
input/Golden prepare 各 `4/4 PASS`。SMLA 正式 13 条在 A3 的单阶段和 prepare/replay 均为 `13/13 PASS`，
全部包含 `batch_intra=PASS`，跨 case报告为4组、26个样本 PASS。QSMLA/MQSMLA仍必须补 A5主算子结果。

没有目标平台时可以交付白盒和 validate 证据，但必须明确写为“待 A5 NPU 验收”，不能把 A3 静态结果或 CPU
Golden 写成 QSMLA/MQSMLA batch PASS。
