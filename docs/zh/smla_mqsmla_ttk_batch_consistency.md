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
| SMLA | A3（`ascend910_93`） | `Ascend910B3/e2e/test_sparse_flash_mla_batch_consistency.csv` | 可在 A3 完成全流程 |
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

TTK_FRAMEWORK=/path/to/ops-test-kit
OPS_TRANSFORMER=/path/to/ops-transformer
TTK_DATA=/path/to/ttk_data
OP=sparse_flash_mla
CSV="$TTK_DATA/Ascend910B3/e2e/test_sparse_flash_mla_batch_consistency.csv"
ASSETS="$OPS_TRANSFORMER/attention/$OP/tests/assets"

export PYTHONPATH="$TTK_FRAMEWORK:$ASSETS:${PYTHONPATH:-}"
```

QSMLA/MQSMLA 在 A5 分别把 `OP` 和 `CSV` 换成上一节表格中的值。每次执行使用新的输出目录，不能把旧
output bin 混入本轮结果。

## 3. CSV 字段语义

### 3.1 三个输入字段

TTK 输入 CSV 新增三个顶层字段。三个字段的最外层长度必须等于 `tensor_view_shapes` 的 tensor slot 数量，
同一位置永远描述同一个 slot。

| 字段 | 含义 | 当前三算子约束 |
| --- | --- | --- |
| `batch_axis` | 每个 tensor slot 中哪些轴承载 batch相关存储 | Q slot 0固定为 `(0,)`；BSND表示batch轴，TND表示由`cu_seqlens_q`解释的token轴 |
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

`(start, stop, step)` 使用左闭右开切片。当前 assets 只接受 `step=1`、非空、非负且不越过 Q 的 B 维。
一个 case 内的 relation 目前只能使用一个 seed；要测试另一 relation，使用另一组 case 或另一 seed。

`batch_axis` 不能只给 Q 填 `(0,)` 后忽略其他实际带 B/B+1轴的 tensor。TND下Q slot的axis 0虽然是物理
token轴，仍填写`(0,)`；assets会要求slice端点严格落在`cu_seqlens_q_values`边界，再映射成logical batch。
PA下KV slot的axis 0是physical block轴，不等于logical batch；block table和`seqused_*`才定义逻辑映射。
跨case工具会结合layout、prefix和shape构建relation上下文，不能仅凭某个第一维恰好等于B猜语义。

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

### 4.1 TND + PA 示例

TND relation仍写Q的物理token区间，但每个区间必须覆盖完整logical batch。以下A3小case令B=4且每个batch
只有1个query token：

```python
tensor_view_shapes = (
    (4, 64, 512),       # q: T=4
    (4, 128, 1, 512),   # ori_kv physical blocks
    (4, 16, 1, 512),    # cmp_kv physical blocks；A3要求block size为16的倍数
    None, None,
    (4, 1),             # ori_block_table: [B, max_blocks]
    (4, 1),             # cmp_block_table
    (5,),               # cu_seqlens_q: B+1
    None, None,
    (4,), (4,), (4,), (4,),
    None, None,
    (64,),
)

attributes = {
    'layout_q': 'TND',
    'layout_kv': 'PA_BBND',
    'cu_seqlens_q_values': [0, 1, 2, 3, 4],
    'seqused_q_values': [1, 1, 1, 1],
    'seqused_ori_kv_values': [128, 128, 128, 128],
    'seqused_cmp_kv_values': [1, 1, 1, 1],
    'cmp_residual_kv_values': [0, 0, 0, 0],
    'batch_deterministic_level': 3,
    # 其余mask、window、ratio、template等仍来自原功能case
}

# case A比较logical batch 0和2；token边界分别为[0,1)、[2,3)
batch_slice_info = ((((0, 1, 1), (2, 3, 1)),), None, ...)
batch_seed       = (((75123, 75123),), None, ...)

# case B比较logical batch 1和3，形成同一个跨case relation
batch_slice_info = ((((1, 2, 1), (3, 4, 1)),), None, ...)
batch_seed       = (((75123, 75123),), None, ...)
```

input adapter先按`cu_seqlens_q`把token slice映射为logical batch，再让pytest为relation batch生成同一份Q、
ori/cmp KV逻辑内容。PA block id可以不同；pytest写入各自block table后，assets回读对应physical block并验证
内容逐bit一致。不能要求两个batch共享同一个block id，共享id会改变cache别名语义。

A3与A5的PA block约束不同。当前A3 tiling要求cmp block size位于`[16,1024]`且16对齐；A5可接受1。
为A3构造小case时应使用物理block size 16并把`seqused_cmp_kv_values`保持为真实有效长度1，不能把有效长度
也伪改成16。

上述两条关系已作为 `SMLA_BATCH_TND_PA_B4_001_NOLSE` 和
`SMLA_BATCH_TND_PA_B4_002_NOLSE` 收录到 A3 正式 batch CSV。它们使用相同 seed、不同 logical batch位置，既能
覆盖同case比较，也能覆盖跨case比较；不是只能依赖开发机临时文件复现的示例。

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
export NPU_DUMP_PATH="$OUT/npu_output"
mkdir -p "$NPU_DUMP_PATH"

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

## 8. 当前支持边界及原因

### 8.1 “框架支持 batch”不等于“所有算子和 layout 自动支持”

这里列出的边界较多，不是主线功能回退，也不是因为 TTK 无法解析这些 layout。普通算子功能测试只要求一条
case 的 NPU输出满足数值精度；batch一致性还要求不同 batch位置、不同 case中的 relation 在输入语义完全等价后，
NPU输出原始字节仍一致。因此，任何会改变 token来源、KV访问集合、量化编码或输出切片映射的特性，都必须有
对应的 relation建模和反例测试后才能开放。主线某个算子已实现自己的映射，只能证明该算子的 adapter完成，不能
自动覆盖输入合同不同的 MLA算子。

batch 一致性由三个彼此独立的能力层组成：

1. **TTK 通用层**解析 `batch_axis`、`batch_slice_info`、`batch_seed`，生成
   `batch_consistency_id`，把四者传给 assets，并负责保存 result/output bin。TTK 不知道 Q、KV、TopK、block
   table 或某个输出轴的算子语义。
2. **算子 assets 层**必须把 relation 转成可证明等价的输入上下文，并把输入 slice 映射到正确的输出 slice。
   SMLA、QSMLA、MQSMLA 的当前边界主要来自这一层的已实现范围。
3. **算子包和 SoC 层**必须真正支持 deterministic level 3及对应 layout/sparse/quant kernel。CSV validate、CPU
   白盒或 metadata调用成功都不能替代目标 SoC 上的主算子证据。

主线 `compressor` 已适配 batch，表示它为自己的输入和输出实现了第 2层，不表示它提供了一个可直接复用的
通用 attention batch 引擎。其 assets 会区分 TH 与非 TH layout，并结合 `cu_seqlens`、`seqused`、
`start_pos`、`cmp_ratio`、`is_th` 计算压缩输出的起点和长度；PA/cache 场景还会同步 state cache。上述公式来自
compressor 的压缩输出定义，不能复制成 MLA 的 attention 输出映射。

主线实现可在 `attention/compressor/tests/assets/impl/inputs.py` 中直接核对，关键行为不是 TTK 自动提供的：

- `x.dim() == 2` 才进入 compressor 的 TH 分支。该分支要求 `cu_seqlens`，以
  `B=len(cu_seqlens)-1` 恢复逻辑 batch，并以 `seqused[b]` 或
  `cu_seqlens[b+1]-cu_seqlens[b]` 得到每个 batch 的有效长度；非 TH 则直接使用物理 `x.shape[0:2]`。
- `coff=1` 时按 relation 的 `[start,end)` 替换对应 token；`coff=2` 时还必须包含前置
  `cmp_ratio` 个 token，即写入 `[start-cmp_ratio,end)`，`start<cmp_ratio` 的 relation会被拒绝。这个额外上下文
  来自 compressor 算法，不是通用 TND规则。
- cache mode会同步改写 `state_cache` 和 `state_block_table`；block范围还依赖
  `start_pos`、有效长度、`block_size` 与 `cmp_ratio`。只固定 `x` 的随机片段并不能构成 compressor batch等价性。

因此，主线 compressor能够测试 TH，是因为它已经实现并测试了上述专用 token-origin和 cache映射。MLA虽然算子
接口本身可支持 TND，其 Q、ori/cmp KV、sparse index、TopK length及 metadata关系不同，不能调用 compressor
公式得到正确 relation。首版 MLA adapter 曾因缺少这些映射而暂不开放 TND/PA；当前版本已经实现完整 logical
batch映射和PA block回读，但仍只在本节列出的输入合同内开放，不能把“已支持TND/PA”扩大解释为任意 sparse、
量化、零长batch或多block跨case组合都已验收。

同理，deterministic level 3只约束算子内部的确定性调度。它不会自动令两个 case 的输入上下文等价，也不会
替测试框架完成 token-origin、paged KV 或 sparse访问集合的对应关系。只有三层同时完成，才能声明某个场景
batch 一致性 PASS。

### 8.2 三 MLA 当前实际实现的 relation

当前assets支持`layout_q=BSND/TND`以及`layout_kv=BSND/TND/PA_BBND`。共同规则如下：

- relation仍只声明在Q tensor slot 0、axis 0，slice连续且`step=1`；一个case只使用一个relation seed；
- BSND的Q slice直接映射logical batch；TND必须用严格递增的`cu_seqlens_q_values`映射，partial batch直接拒绝；
- relation两侧必须具有相同logical batch数、每batch Q/KV长度、`seqused_*`和`cmp_residual_kv`；
- pytest生成完整输入，assets只控制relation随机源。TND同步Q、ori KV、cmp KV token origin；PA先生成等价
  logical KV，再按各自block table写入physical block；
- 同case和跨case当前都对主输出output 0的Q axis 0切片做原始storage字节比较；普通pytest compare仍检查
  所有启用输出的数值精度；
- 没有batch三字段的普通case完全走原pytest路径。

### 8.3 TND映射和剩余边界

TND的axis 0是token轴。assets先检查每个slice的start/stop均在`cu_seqlens_q_values`中，再把它转换为
logical batch区间；output比较仍切原始token区间，因为SMLA/QSMLA/MQSMLA主输出与Q保持同一T轴。跨case
工具使用同一prefix恢复B，并把B、B+1、Tq、Tori和Tcmp维度归一化后构建上下文。

当前严格拒绝零长logical batch，即prefix必须严格递增。原因不是API必然不支持零长，而是重复prefix会使
token边界到batch编号不再一一映射，现有relation ID无法区分空batch位置。非均匀长度本身可以使用，但relation
两侧对应batch的Q/KV长度必须相同；不满足时应设计另一relation，不能由assets改写长度。

### 8.4 PA映射和剩余边界

PA relation不要求相同block id。pytest为每个logical batch生成KV内容并分配block table；assets确保relation
batch的逻辑内容逐bit相同，再核对block table指向的physical block内容。`seqused_ori_kv`、
`seqused_cmp_kv`、residual及block size共同进入合同，缺任何CSV slot都按`copy_tensor`单边None失败。

当前已覆盖“每个batch一个physical block”的跨case实例。若两个case使用不同B并且每batch需要多个block，
physical block总数和block table宽度会不同；当前跨case上下文仍保守地把这类storage geometry视为不同case。
这不影响同case检查，也不表示PA kernel不支持多block，但在补齐physical-storage归一化和反例前不能强行归组。

### 8.5 sparse、TopK和量化边界

sparse index、TopK length、window、compressed/residual长度决定实际访问的KV集合。当前边界为：

- SMLA：batch模式只开放SWA/HCA；
- MQSMLA：只开放SWA/HCA和`quant_mode=1`；
- QSMLA：开放SWA/HCA/ORI_SPARSE/ORI_CMP_SPARSE及`quant_mode=1`，sparse mode必须为`full`，TopK只允许
  `fullK/no`，`seqused_q`必须均匀；CSA、random sparse和random TopK仍拒绝；
- relation两侧的模式、mask、window、TopK和长度必须一致，不能靠相同seed掩盖业务上下文不同。

QSMLA主输出dtype为`bfloat16`，即使Q输入为`uint8`；外置bin比较按输出合同计算byte count，不能从输入dtype
推断。其他quant mode需同步scale、offset、rounding、输出dtype并补A5证据后再开放。

### 8.6 可选LSE和分阶段边界

pytest在TND下返回的LSE布局为`(T1,N2,G)`并用flatten compare；assets按既定合同原样返回，不推导或转轴。
当前A3包的SMLA设备LSE为`(N2,T1,G)`。单阶段执行因此能按pytest语义完成LSE精度比较，但manual-data replay
会在compare前严格检查保存Golden与设备output shape，并明确报shape不一致。

这不是batch token映射失败，也不能在assets中私自转轴规避。现阶段分阶段batch验收使用
`return_softmax_lse=False`，主输出、同case和跨case均正常；LSE=True使用单阶段验证。若业务要求LSE=True也
支持replay，应先由算子pytest/API owner统一shape合同，或由TTK定义对custom flatten compare可声明的通用
manual-data协议，再同步三算子，不能只改某一个assets。

### 8.7 平台边界如何判定

- `--validate`和CPU白盒只证明字段、shape及relation合同可解析；
- A3真机只能证明当前A3包实际注册且支持的feature；A3 PA压缩block size要求16对齐，A5规则不同；
- A5-only sparse、quant或dtype能力必须在匹配custom OPP的A5验证；
- metadata成功但主算子被SoC/tiling拦截，仍不能写成batch一致性PASS；
- level发生降级或runtime拒绝level 3时，普通精度PASS不能替代level-3证据。

报告中应区分“assets已实现但目标平台未验”“当前包/SoC明确拒绝”和“已验收”，不能笼统写成“不支持”。

### 8.8 当前算子专项矩阵

| 算子 | assets/CPU已实现 | 当前真机证据 | 仍保留边界 |
| --- | --- | --- | --- |
| SMLA | BSND/TND Q；BSND/TND/PA KV；SWA/HCA；正式CSV 11条 | A3正式表11/11精度及同casePASS，跨case`groups=3/samples=22/PASS`；TND+PA LSE单阶段2/2、无LSE prepare/replay 2/2 PASS | sparse模式、零长batch、多block跨B归组、TND LSE分阶段shape合同 |
| MQSMLA | 同上，`quant_mode=1`；真实pytest generator和PA block回读逐bitPASS；正式CSV 9条 | A3 input/Golden prepare通过；主C API被SoC/算子包明确拒绝，目标为A5 | sparse模式、其他quant mode及A5真机 |
| QSMLA | TND/PA真实pytest generator和PA block回读逐bitPASS；既有四类非CSA模式；正式CSV 2条 | A3 input/Golden prepare通过；主C API被SoC/算子包明确拒绝，目标为A5 | CSA、random sparse/TopK、非均匀`seqused_q`及A5真机 |

该表描述TTK assets和验证证据，不是算子host/kernel的完整功能表。

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

没有目标平台时可以交付白盒和 validate 证据，但必须明确写为“待 A5 NPU 验收”，不能把 A3 静态结果或 CPU
Golden 写成 QSMLA/MQSMLA batch PASS。

当前交付基线的静态结果为 SMLA `11/11`、QSMLA `2/2`、MQSMLA `9/9` valid；三算子 input、同case compare和
跨case output-bin 六组白盒共 `55 passed`。SMLA正式11条还完成A3单阶段全量执行和严格跨case检查，结果为
`11/11 PASS`、`groups=3`、`samples=22`。QSMLA/MQSMLA仍须按上表在A5和匹配算子包上完成主算子验收。
