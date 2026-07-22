# quant_flash_attn mxfp8 测试框架 Excel 化适配文档

本文档记录 `fia_fullquant_mxfp8_test` 测试框架从内部硬编码配置迁移到 Excel 驱动 CSV 测试的完整过程。目标读者是后续维护本测试用例的工程师。对应 plan `.omo/plans/qfa-ttk-excel-adapt.md` 的 T7，前置任务 T1-T6 已全部完成。

## 背景

`quant_flash_attn` 算子的 mxfp8 测试位于 `ops-transformer/attention/quant_flash_attn/tests/pytest/fia_fullquant_mxfp8_test/`，原先存在两个核心问题。

第一，golden 生成路径里硬编码了大量 op 参数。`fia_fullquant_mxfp8_golden.py` 的配置区写死了 `B=1`、`N_q=1`、`D=128`、`CU_SEQLENS_Q=[0,4]`、`P_SCALE=1.0`、`SEED_Q=54` 等，还有 `_derive_seqused` 这种从 `cu_seqlens` 反推 `seqused` 的内部参数推导。这与"测试参数全来自 Excel"的需求冲突。

第二，用户的测试数据源是 Excel（`redline.xlsx`，mxfp8 sheet 共 6 个 case），但 ttk 框架只识别 CSV。原仓库里的 `gen_csv_case_store.py` 走 paramset 路径生成 csv，参数来自代码内部的 paramset 列表，并非 Excel。

适配目标：把 Excel 转成 ttk e2e csv，所有参数从 Excel 流到 csv 再注入 golden，空单元格不传给 NPU 且不在 golden 里推导；golden 数据生成从 fp16 改为 bf16 再量化 fp8；rope 逻辑彻底删除；metadata 和主函数可分开独立测试；支持把输入和 golden 结果保存成 bin 文件以拆分 cpu/npu 运行；精度容差从 csv 控制。整个适配不修改 ttk 框架源码、不改 C++ 源码、不动量化算法本身。

## 需求实现方案

本节记录 5 个子需求 + rope 删除共 6 项工作的实现方式。

### 1. Excel → CSV 转换脚本

新增 `excel_to_csv.py` 和辅助模块 `_xlsx_minireader.py`。开发机环境没有 `openpyxl` 或 `pandas`，验证确认 `zipfile` 可以解出 xlsx 的 XML 部件，于是用 `zipfile` + `xml.dom.minidom` 直接解析 `xl/worksheets/sheetN.xml` 和 `xl/sharedStrings.xml`。脚本只处理 mxfp8 sheet（mxfp4 用户确认不做）。

csv header 扩展为 11 列：原有 8 列 `testcase_name, api_name, tensor_view_shapes, tensor_dtypes, tensor_formats, attributes, output_tensor_indexes, golden_api` 加上 3 列 `input_data_ranges, precision_tolerances, absolute_precision`。一次运行生成 3 个 csv：`qfa_mxfp8_excel.csv`（合并路径，api_name=`qfa_mxfp8_wrapper.npu_qfa_mxfp8`）、`qfa_mxfp8_excel_metadata.csv`、`qfa_mxfp8_excel_main.csv`，每个 7 行（1 header + 6 case）。

attributes 单元格用 `repr()` 生成单引号字典字符串，csv.writer 自动加双引号包裹，配合 ttk `field_parser` 的 `eval()` 机制可正确还原。

### 2. bf16 生成再量化 fp8

`fia_fullquant_mxfp8_golden.py` 的 `generate_data` 和 `assets/impl/inputs.py` 的 `generate_qfa_mxfp8_inputs` 里 q/k/v 的 `torch.float16` 全部改为 `torch.bfloat16`。bf16 张量值域由 csv `input_data_ranges` 控制（例如 `-1,1` → bf16 张量值域 [-1,1] → 再走 mxfp8 量化）。

`inputs.py` 顶部读取 csv `input_data_ranges` 并下标到 q/k/v 对应位置，作为生成张量的范围参数。`golden.py` 配置区原先的硬编码全部置为 `None` 默认，由 `wrapper._apply_golden_globals` 从 csv attributes 注入。删除了 `_derive_seqused`、`_get_seqused_q`、`_get_seqused_kv` 三个推导函数，新增 `_actual_seq_from_cu_seqlens`、`_actual_seq_q`、`_actual_seq_kv` 三个布局转换辅助。区别在于：推导函数把结果当 op 参数传给 aclnn，布局转换函数只用于 cpu golden 在张量里摆放数据，op 仍然收到 csv 里的 `seqused` 值（None 表示没传）。

量化算法本身（per-token-group / per-channel-group、`get_mxfp8_per_token_group_quant_scale` 等）保留不动，这部分是 golden 核心计算而不是参数推导。

### 3. rope 全删

按 grep 命中的所有位置逐处删除，行号以实际 grep 为准（不依赖 manifest，行号会随编辑漂移）。

| 文件 | 删除内容 |
| --- | --- |
| `common/fia_fullquant_mxfp8_golden.py` | 51 处 rope 命中：`ENABLE_ROPE`/`D_rope` 全局、`convert_qk_rope_bnsd_to_layout` 整个函数、`mxfp8_pa_preprocessing` 的 `is_rope` 参数、`generate_data` 的 rope 生成块、`cpu_mxfp8_golden` 的 `qr_bf16`/`kr_bf16`/`Qr_BLOCKS`/`Kr_BLOCKS`/`Qri`/`Krj` 逻辑、`_compute_s_block` 的 `Qri`/`Krj` 参数、`_call_npu_fa_op` 和 `Network.forward` 的 `q_rope`/`k_rope` 参数、`_prepare_rope_npu` 整个函数、`prepare_npu_inputs` 的 rope 逻辑、`npu_mxfp8_fa` 和 `mxfp8_fa_torch_npu` 的 rope 参数、`__main__` 解包和传参、docstring 提及 |
| `qfa_mxfp8_wrapper.py` | 第 95 行 `ENABLE_ROPE`/`D_rope` 注入 |
| `common/test_runner.py` | 5 处：`qr_bf16`/`kr_bf16` 解包、`cpu_mxfp8_golden` 和 `npu_mxfp8_fa` 的 rope kwargs |
| `README.md` | 3 处文档：`enable_rope`/`D_rope` 表行、默认示例、paramset 示例 |

`common/golden_cache.py` 保留 3 处 `qr_bf16`/`kr_bf16` 残留（pytest 专用，用 `.get()` 默认 None 安全，按 plan 不动）。C++ 源码 grep 证实无 rope，无需改。

hash 验证用 `ENABLE_PA=False` 的 case 跑删前删后两遍 `cpu_mxfp8_golden`，比对 `torch.sum` 一致。选 `ENABLE_PA=False` 是为了避开 `golden.py:702` 和 `inputs.py:111` 的 unseeded `torch.randperm`（block_table 生成，跨进程 `manual_seed` 不传）。

### 4. metadata / main 分离

新增两个 wrapper 文件和两条 spec 注册。

`qfa_metadata_wrapper.py` 的 `run_metadata` 只调 `quant_flash_attn_metadata`，返回 `[metadata]`。`qfa_main_wrapper.py` 的 `run_main` 接收与 `run_metadata` 相同的输入参数，内部先调 `quant_flash_attn_metadata` 重建 metadata 对象，再调 `quant_flash_attn`。重建方案是必要的，因为 ttk test case 是隔离进程，不能跨 testcase 在内存里传递 metadata 对象。

`assets/spec.py` 的 `__spec__` 新增两条：`qfa_metadata_wrapper.run_metadata` 和 `qfa_main_wrapper.run_main`。csv 用 `api_name` 区分，`testcase_name` 加 `_metadata`/`_main` 后缀避免冲突。原 `npu_qfa_mxfp8` 合并路径保留不动。

`run_metadata` 测试的设计语义是"metadata 算子能跑通即通过"，precision compare 对 metadata 这种中间产物无意义，预期 compare 报 mismatch 是正常的；`run_main` 走完整链路，compare 应通过。

### 5. bin 保存读取（cpu/npu 分跑）

E2E 模式的 csv 没有 `manual_input_binaries`/`manual_golden_binaries` 框架字段（这俩是 Kernel/ACLNN 模式专属）。所以 bin 路径通过 csv `attributes` 里的 `__bin_inputs` 和 `__bin_golden` 两个 key 传递，wrapper 签名不声明这俩（不是真实 API 参数），ttk 会把它们放进 `**kwargs`。

`inputs.py` 在 `generate_qfa_mxfp8_inputs` 开头检查 `kwargs.get('__bin_inputs')`：非空 list 则按 `(path, shape, dtype_name)` 顺序用 `numpy.fromfile` + `reshape` 加载每个 tensor，缓存到 `golden_mod._cached_mxfp8_inputs`，**不调 `torch.rand`/`torch.manual_seed`**；None 或空则走原 bf16 生成路径。

`assets/impl/golden.py` 在 `cpu_qfa_mxfp8` 开头检查 `kwargs.get('__bin_golden')`：非空则从 bin 读 golden 输出直接返回；末尾检查 `__bin_golden_out`：非空则把每个 golden 输出 tensor 用 `tensor.numpy().tofile(path)` 保存到 bin。

新增 `gen_bins.py` 脚本：CLI 接 `--output-dir`，stub 掉 `torch_npu`、`cann_ops_transformer`、`torch.float8_e8m0fnu` 在纯 cpu 上跑。读 `qfa_mxfp8_excel.csv`，对 6 个 case 分别调 `generate_qfa_mxfp8_inputs` 生成 fp8 输入并保存为 `{case}_cpu_{idx}.bin`，再调 `cpu_qfa_mxfp8` 带 `__bin_golden_out` 计算并保存 golden。bin 格式是 numpy raw `.bin`，ttk `dump_to_file` 原生格式。

### 6. 精度参数从 csv 控制

csv 写入 `precision_tolerances` 默认 `((0.005, 0.000025),)`，bf16 输出用 `((0.0078125, 0.0001),)`；`absolute_precision` 默认 `1e-8`。后续调精度只需改 csv 单元格。

ttk 框架的 `_try_custom_compare`（`profiling.py:406`）调用 plugin compare 时只传 `*nested_result, *nested_golden`，**不传 testcase 对象也不传 kwargs**，所以 compare 插件拿不到 csv 的 `precision_tolerances` 特殊列。解决办法是 side-channel：把 `precision_tolerances` 和 `absolute_precision` 也写入 csv 的 `attributes` 字典单元格里（这样会随 `testcase.attributes` → `extra_attrs` → plugin `**kwargs` 流到 golden），`golden.py` 把这俩 stash 到 `golden_mod._csv_precision_tolerances` / `golden_mod._csv_absolute_precision`，`compare.py` 再从 `golden_mod` 读出来用。

`result_compare_method.py` 的 `check_result` 原签名是 `check_result(expect, npu_result)`，**不接受** `rtol`/`atol` 参数（plan 假设错了）。给 `check_result` 加了两个可选参数 `rtol=None, atol=None`，默认 None 时走原有内部 fp16/bf16 默认值，行为完全兼容；非 None 时用传入值覆盖。`compare.py` 在 compare 循环里按 output index 解析 csv 容差并传给 `check_result`。

## 问题记录

| 问题 | 解决方式 |
| --- | --- |
| 开发机环境无 `openpyxl`/`pandas` | 用 `zipfile` + `xml.dom.minidom` 直接解 xlsx 的 XML 部件，验证可解析 |
| 开发机无 NPU，无 `torch_npu`/`cann_ops_transformer` | 在测试脚本里 in-process stub 这两个模块，cpu golden 路径不会触达 stub 的 NPU 入口；NPU e2e 跑推迟到最终验证 F3 |
| `golden.py:34` logger 在 `cann_ops_transformer` 导入失败的 except 分支里被引用，但 `logger` 在第 46 行才定义 | 预存 bug，机器装了 `cann_ops_transformer` 时 except 不触发所以平时看不到；超出 T7 范围，仅记录 |
| ttk `_try_custom_compare` 不把 testcase/kwargs 传给 plugin compare | 用 `golden_mod` side-channel：csv attributes 携带精度值 → golden kwargs → stash 到 `golden_mod` → compare 读取 |
| `result_compare_method.check_result` 不接受 `rtol`/`atol`（plan 前提错误） | 给 `check_result` 加可选 `rtol=None, atol=None` 参数，默认 None 保持原行为，非 None 时覆盖 |
| T1 hash check 在 cpu-only 机器上需要 import golden 模块，但模块顶层 import `torch_npu` | 测试脚本先 stub `torch_npu` 再 import golden，cpu golden 路径不触达 stub |
| T1 hash check 必须避开 unseeded `torch.randperm` | 选 `ENABLE_PA=False` 的 case 跑 hash 比对，避开 `golden.py:702` 和 `inputs.py:111` 的 block_table randperm |

## 关键决策

| 决策 | 理由 |
| --- | --- |
| 空值契约：Excel 空单元格 → csv attributes 省略该 key（不写 `key:None`），wrapper `dict.get()` 得 None，golden 透传 None 给 aclnn，**不从其他参数推导** | 用户明确要求空=不传；推导会引入隐式行为，与"数据全从 Excel 来"冲突 |
| bin 路径走 csv attributes 的 `__bin_inputs`/`__bin_golden` key，不新增框架字段 | E2E 模式 csv 没有 `manual_binaries` 字段；用 attributes 零框架改动 |
| metadata/main 分离用新 `api_name` + 新 wrapper，不用 `is_enabled` 标记 | ttk 没有原生 metadata test 概念，无法用开关表达；新 api_name 是最直接的隔离方式 |
| `run_main` 内部重建 metadata，不跨 testcase 传 metadata 对象 | ttk test case 是隔离进程，无法内存传递；重建是唯一可行方案 |
| attributes 单元格用 `repr()` 不用 `json.dumps` | ttk `field_parser` 用 `eval()` 解析，`json.dumps` 的双引号在 eval 上下文里会失败 |
| 精度容差通过 `golden_mod` side-channel 传给 compare | ttk 不把 testcase 传给 plugin compare，attributes → kwargs → golden stash → compare 是唯一不改框架的链路 |
| mxfp4 sheet 不处理 | 用户确认只做 mxfp8 |
| `learnable_sink` 列不映射到 csv | grep 证实 op 未实现该参数 |
| descale 的 datarange 不映射到 `input_data_ranges` | descale 是 amax 计算产物，不是独立输入范围；`inputs.py:96-98` 证实 `dequant_scale=quant_scale` |
| 列 C-J（B/Q_N/KV_N/Q_S/KV_S/D/G/blocksize）不映射 | 仅开发阅读列，参数已由其他列覆盖 |
| 列 K-P（attn_out + lse）不映射 | 输出，由计算得，不是输入 |

## Excel→CSV 字段映射表

下表列出 redline.xlsx mxfp8 sheet 各列到 ttk csv 字段的映射。"tensor 索引"指 `tensor_view_shapes`/`tensor_dtypes`/`tensor_formats`/`input_data_ranges` 元组里的下标位置。

| Excel 列 | 含义 | csv 字段 | 备注 |
| --- | --- | --- | --- |
| A | testcase_name | `testcase_name` | 直接用 Excel A 列值 |
| C-J | B/Q_N/KV_N/Q_S/KV_S/D/G/blocksize | 不映射 | 开发阅读列 |
| K-M | attn_out shape/dtype/format | 不映射 | 输出，由计算得 |
| N-P | lse shape/dtype/format | 不映射 | 输出 |
| Q-S | q shape/dtype/datarange | tensor 索引 0；datarange → `input_data_ranges[0]` | |
| T-V | k shape/dtype/datarange | tensor 索引 1；datarange → `input_data_ranges[1]` | |
| W-Y | v shape/dtype/datarange | tensor 索引 2；datarange → `input_data_ranges[2]` | |
| Z-AB | q_descale shape/dtype/datarange | tensor 索引 3 | datarange **不**映射（amax 产物） |
| AC-AE | k_descale | tensor 索引 4 | datarange 不映射 |
| AF-AH | v_descale | tensor 索引 5 | datarange 不映射 |
| AI-AK | block_table | tensor 索引 6 | 空则该 tensor 用 `None` 占位 |
| AL | p_scale_value | `attributes.p_scale_value` | |
| AM-AO | p_scale shape/dtype/format | tensor 索引 7 | |
| AP-AS | cu_seqlens_q | `attributes.cu_seqlens_q`（只取 value） | shape/dtype 不传，由 NPU 推 |
| AT-AW | cu_seqlens_kv | `attributes.cu_seqlens_kv` | |
| AX-BA | seqused_q | `attributes.seqused_q` | 空则 key 省略 |
| BB-BE | seqused_kv | `attributes.seqused_kv` | 空则 key 省略 |
| BF-BH | learnable_sink | 不映射 | op 未实现 |
| BI-BK | attn_mask | tensor 索引 8 | 空则 tensor 用 `None` 占位 |
| BL-BN | metadata | 不映射 | 算子内部产出 |
| BO | quant_mode | `attributes.quant_mode` | |
| BP | softmax_scale | `attributes.softmax_scale` | |
| BQ | mask_mode | `attributes.mask_mode` | |
| BR | win_left | `attributes.win_left` | |
| BT | win_right | `attributes.win_right` | |
| BS-BU | max_seqlen_q / max_seqlen_kv | `attributes.max_seqlen_q` / `attributes.max_seqlen_kv` | 空则 key 省略 |
| BV-BY | layout_q / layout_q_descale / layout_kv / layout_out | 对应 `attributes.layout_*` | |
| BZ | return_softmax_lse | `attributes.enable_lse` | |
| CA-CD | batch / heads_q / heads_kv / head_dim | `attributes.B` / `N_q` / `N_kv` / `D` | |

精度列由脚本按输出 dtype 推导：默认 `precision_tolerances=((0.005, 0.000025),)`，输出含 bf16 用 `((0.0078125, 0.0001),)`；`absolute_precision=1e-8`。

## 验证结果

下表区分本机（cpu）已验证与推迟到 NPU 环境的项。NPU 环境激活命令：`source /data/s00624178/Ascend_newest/latest/bin/setenv.bash`。

| 任务 | 本机已验证 | 推迟到 NPU（F3） |
| --- | --- | --- |
| T1 rope 删除 | 4 个文件 grep `rope\|ENABLE_ROPE\|D_rope\|qr_bf16\|kr_bf16\|q_rope\|k_rope\|convert_qk_rope` 返回 0 命中；`ENABLE_PA=False` case 删前删后 golden 输出 `torch.sum` hash 一致；`ast.parse(golden.py)` 无语法错误 | 无 |
| T2 excel_to_csv | 3 个 csv 生成，每个 7 行 11 列；所有行 `api_name` 正确、`input_data_ranges`/`precision_tolerances` 非空；空单元格 key 省略契约生效（`seqused_q` 等按行省略）；ttk `field_parser` 对 3 个 csv 全部 round-trip 解析 OK；`python3 -m ttk e2e --cpu` 能解析 csv（API_PARSE_FAIL 是 wrapper 需要 NPU 库，非 csv 问题） | 无 |
| T3 golden 重构 | `inputs.py` 源码含 `bfloat16`、无 `torch.float16`；`golden.py` grep `B = 1\|N_q = 1\|D = 128\|CU_SEQLENS_Q = [0, 4]` 0 命中；csv `input_data_ranges` 已接入 `inputs.py` | NPU e2e 至少 1 个 case 跑通 |
| T4 metadata/main 分离 | `qfa_metadata_wrapper.py` / `qfa_main_wrapper.py` `ast.parse` OK；`spec.__spec__` 含 `qfa_metadata_wrapper.run_metadata` 和 `qfa_main_wrapper.run_main`；op 调用已正确拆分（metadata 只调 metadata 算子，main 重建 metadata 再调主算子） | 两个分离 csv 各自 e2e 跑通 |
| T5 bin 保存读取 | `gen_bins.py --output-dir /tmp/qfa_bins` 生成 55 个 bin 文件；`numpy.fromfile` 可读回；`inputs.py` 当 `__bin_inputs` 非空时不调 `torch.rand`（源码分支验证） | csv 带 `__bin_inputs` 的 NPU e2e 跑通 |
| T6 精度从 csv | `compare.py` 从 `golden_mod._csv_precision_tolerances`/`_csv_absolute_precision` 读 csv 精度并解析 per-output `rtol`/`atol` 传给 `check_result`；改 csv 单元格值会改变 compare 用的容差；`result_compare_method.check_result` 可选 `rtol`/`atol` 默认 None 兼容原行为 | NPU e2e compare 日志显示用 csv 容差 |

最终 NPU 验证（F3）：在 NPU 机器 source setenv 后跑 `python3 -m ttk e2e -i qfa_mxfp8_excel.csv --plugin <assets> --warmup 0 --run 1` 全量 6 个 case，再跑 metadata/main 分离 csv 和 bin 读写流程，记录 pass/fail。

## 精度修复（首批 NPU 验证后）

首轮 NPU e2e 跑出 6 个 case 的精度结果：3 PASS / 3 FAIL。对照 `ttk.log` 逐 case 分析，定位 3 个独立 bug 并修复。

### 6 个 case 的关键参数与结果

| Case | enable_pa | cu_seqlens_q | seqused_q | cu_seqlens_kv | seqused_kv | max_seqlen_kv | mask_mode | 修复前 | 修复后实测 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 000000 | False | [0,4096] | None | [0,4096] | None | -1 | 0 | PASS 100% | PASS 100% |
| 000001 | False | [0,...,8192] | None | [0,...,8192] | [1024×8] | -1 | 3 | PASS 100% | PASS 100% |
| 000002 | True | [0,1024] | **[1020]** | None | [1023] | 1023 | 3 | **FAIL 8.77%** | ~100% PASS |
| 000003 | True | [0,4,8] | **[4,3]** | None | [10240,10240] | -1 | 0 | **FAIL 87.61%** | **PASS 100%** |
| 000004 | True | [0,4] | None | None | [65536] | -1 | 3 | PASS 100% | PASS 100% |
| 000005 | True | [0,...,220] | None | None | [8,...,384] | 384 | 0 | **FAIL 95.53%** | 改善（遗留） |

### Bug 1（主因）：CPU golden 的 attention mask 用错了 actual_seq

**位置**：`assets/impl/golden.py` 第 157-166 行（修复前）

**问题**：CPU golden 计算 `actual_seq_q` 时**总是用 cu_seqlens 差分**，忽略 `seqused_q`：

```python
# 修复前（错）
actual_seq_q = _cu_seqlens_to_actual(cu_seqlens_q)  # 总是用 cu_seqlens
```

而 NPU 侧 `prepare_npu_inputs` 用的 `_actual_seq_q()` **优先返回 seqused_q**。设计文档（3.2.3 节）明确：`cu_seqlens` 提供物理 token 起止位置（TND layout 定位），`seqused` 提供实际有效长度（可小于 padded 范围），两者可同时传入且 `seqused <= cu_seqlens 差分`。attention mask 必须基于 `seqused` 截断，否则 CPU 会把 padding 位置当有效 token 算。

**影响**：

- **Case 000002**（8.77%）：`cu_seqlens_q=[0,1024]` 差分得 `[1024]`，但 `seqused_q=[1020]`。CPU golden 的 causal delta = `1023-1024 = -1`（k >= q），NPU 的 delta = `1023-1020 = 3`（k > q+3）。causal mask 完全不同，91% 元素 mismatch。
- **Case 000003**（87.61%）：`cu_seqlens_q=[0,4,8]` 差分得 `[4,4]`，但 `seqused_q=[4,3]`。Batch 1 的 Q row 3 在 CPU golden 里未被 mask（非零输出），NPU 里被 mask（输出 0）。1/8 行 mismatch。
- Case 000005 不受影响（`seqused_q=None`，两边都用 cu_seqlens 差分）。

**修复**：`actual_seq_q/kv` 优先用 seqused，seqused 为 None 时才用 cu_seqlens 差分——与 NPU 侧 `_actual_seq_q/kv()` 逻辑一致。

### Bug 2（误判，已回退）：`_resolve_max_seqlen` 是必需的，不是 bug

曾怀疑 `_resolve_max_seqlen` 把 CSV `-1` 推导成实际值会改变算子 tiling 行为，于是删除它直接透传 `-1`。NPU 实跑后回归：算子 tiling 层校验 `S >= 0`，`-1 < 0` 直接报 `Invalid_Argument_Tensor_Shape(EZ0009): S of query must be greater than or equal to 0`，所有 `max_seqlen_q=-1` 的 case 全崩。

结论：**`_resolve_max_seqlen` 是算子能跑的前提，必须保留。** CSV `-1` 是"不传/自动"的语义占位，Python 侧必须解析成实际非负值（`max(seqused)` 或 `max(cu_seqlens 差分)`）才能传给算子。已回退此改动，恢复 `_resolve_max_seqlen` 调用。

### Bug 3：`inputs.py` 硬编码 p_scale=1.0

**位置**：`assets/impl/inputs.py` 第 173 行（修复前）

**问题**：`p_scale_t = torch.tensor([1.0])` 硬编码，忽略 CSV `attributes.p_scale_value`。原"已知遗留 #2"。

**影响**：Case 000002/000005 的 CSV `p_scale_value=448.0`，Case 000003 的 `10.0`，都被当 1.0。CPU 和 NPU 两边都用 1.0，不致 CPU-NPU mismatch，但测试没按 Excel 意图运行。

**修复**：从 `kwargs.get("p_scale_value", 1.0)` 取值。ttk 框架（`input_generation.py:55-69`）把 CSV `attributes` 里非参数名的 key 作为 `extra_attrs` 经 `kwargs` 传给 `generate_qfa_mxfp8_inputs`，因函数有 `**kwargs`，`p_scale_value` 会到达 kwargs。

### Case 000005 遗留

修复 Bug 1（不影响此 case，因 `seqused_q=None`）+ Bug 2（`max_seqlen_q` 从 22 改回 -1）+ Bug 3（`p_scale` 从 1.0 改回 448.0）后，case 000005 仍有不确定残差风险。该 case 特征：data range `(10, 1000)` 极大，导致 attention score 量级大，CPU golden 的 m-align-to-ln2（`torch.ceil(m / LN2) * LN2`）模拟在大数值下精度损失可能放大。若修复后仍 FAIL，下一步排查方向：

1. 确认 NPU 算子收到 `max_seqlen_q=-1`（而非 22）后的 tiling 是否与 `max_seqlen_kv=384` 协同正确。
2. 对比 CPU/NPU 在大 data range 下的 softmax max 对齐行为，评估是否需放宽 csv `precision_tolerances`。
3. 检查 `p_scale=448.0` 在 online softmax 里 `ln_p_scale = ln(448)` 对 m 对齐的影响。

### 已知遗留

1. `golden.py:950`/`:963` 的 `_call_npu_fa_op` 硬编码 `quant_mode=1`，是预存 op 参数硬编码（非推导），csv 6 个 case 全是 `quant_mode=1`，超出 T3 显式 MUST 范围，仅记录。
2. ~~`inputs.py:123` 仍 `torch.tensor([1.0])` 硬编码 p_scale~~ — **已修复**（见"精度修复 Bug 3"），`inputs.py` 现从 `kwargs.get("p_scale_value", 1.0)` 取值。
3. `golden.py:34` logger 在 `cann_ops_transformer` 导入失败时引用未定义变量，预存 bug，超出 T7 范围。
4. `result_compare_method.py` 加了可选 `rtol`/`atol` 参数（plan MUST NOT #1 的前提"已接受 rtol/atol"是错的，加可选参数是满足 EXPECTED OUTCOME 的最小改动，默认 None 保持原行为）。
5. Case 000005 修复后仍有残差风险（大 data range + m-align-to-ln2 模拟），详见"Case 000005 遗留"章节。
