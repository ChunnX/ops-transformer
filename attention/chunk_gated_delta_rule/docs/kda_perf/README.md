# KDA torch_extension 与新版 CGDR：RDV 对比（2026-09-09）

本报告替代此前直接调用 ACLNN 的结果。被测入口为 **`cann_ops_transformer.ops.chunk_kda_fwd`**，
底层仍为 **CANN 9.2.0 内置 KDA**；本轮没有重编译或替换 KDA kernel。
CGDR 使用 [new_streaming/comparison.csv](../rdv_perf/new_streaming/comparison.csv) 的 `optimized_mean_us`，没有重新测量 CGDR。

共保留 **288** 个原始 RDV 条目，**272** 个获得性能数据，
**16** 个因维度约束不适用。
保持原始 shape 的 **244** 个配对，KDA/CGDR 主 kernel 耗时比几何平均为 **5.451×**。
另有 **28** 个条目使用中性序列补齐，单独统计。
精度实际检查 **272** 例，通过 **272** 例；
失败列表：`[]`。通过条目的 `kda_precision_passed` 已填写 `True`。

## 性能结果

比值为 `KDA / 新版 CGDR`，大于 1 表示 KDA 更慢。每例 1 次预热、5 次正式测量，单位 µs。
主表仅比较设备主 kernel 的 `Task Duration(us)`。

| 范围 | 用例数 | 几何平均耗时比 | 各 case 均值之和的比值 |
|---|---:|---:|---:|
| 保持原始 shape | 244 | 5.451× | 6.258× |
| 原始 shape，双方 FP32 状态 | 122 | 5.166× | 5.872× |
| 中性补齐序列，单列 | 28 | 5.765× | 5.902× |
| 全部可执行映射（含补齐） | 272 | 5.483× | 6.207× |

最大 KDA CV 为 **1.857%**。各 case 均值之和不是业务端到端延迟。
完整数据见 [comparison.csv](comparison.csv)，逐次样本见 [kernel_samples.csv](kernel_samples.csv)。

| 用例 | B / 原 S / Nk / Nv / Dk / Dv | KDA 实际布局和长度 | 新版 CGDR | KDA | 耗时比 |
|---|---|---|---:|---:|---:|
| ARC-001 | 1 / 2048 / 32 / 32 / 128 / 128 | BSND，S=2048 | 91.967 | 429.495 | 4.670× |
| ARC-004 | 2 / 2048 / 32 / 32 / 128 / 128 | TND，S=2048 | 174.668 | 1348.759 | 7.722× |
| ARC-009 | 1 / 262144 / 32 / 32 / 128 / 128 | BSND，S=262144 | 9330.127 | 48468.786 | 5.195× |
| ARC-011 | 1 / 2048 / 2 / 2 / 128 / 128 | BSND，S=2048 | 82.730 | 377.986 | 4.569× |
| ARC-017 | 1 / 6000 / 8 / 8 / 128 / 128 | BSND，S=6016 | 276.672 | 1121.687 | 4.054× |
| ARC-036 | 1 / 65535 / 16 / 32 / 128 / 128 | BSND，S=65536 | 2347.418 | 12131.567 | 5.168× |
| ASC-019 | 1 / 128 / 32 / 32 / 64 / 128 | BSND，S=128 | 20.165 | 125.188 | 6.208× |
| ASC-025 | 2 / 300 / 8 / 8 / 64 / 64 | BSND，S=320 | 30.241 | 264.793 | 8.756× |
| ASC-028 | 4 / 512 / 4 / 8 / 64 / 64 | BSND，S=512 | 41.139 | 415.561 | 10.101× |
| ASC-029 | 4 / 512 / 4 / 16 / 64 / 64 | BSND，S=512 | 75.704 | 813.221 | 10.742× |
| ASC-034 | 4 / 2000 / 2 / 2 / 16 / 16 | BSND，S=2048 | 127.688 | 1474.488 | 11.548× |

## 输入和调用映射

- 保留 RDV 的 B、序列长度、Nk/Nv、Dk/Dv、BF16 Q/K/V/beta、Q/K 的 L2 归一化、`scale=Dk**-0.5` 和状态连续/非连续变体。
  seed=20260908；历史 CGDR 未保存完整输入快照，因此不声称逐字节回放了历史输入。
- KDA 门控为 `[T,Nv,Dk]`。独立 seed=20260909 生成目标 `-Uniform[0,1)` per-dim log-gate；无 gate 条目使用零 log-gate。
  **128×128 且有 gate** 时，计时外通过 `p=max(-gamma/5,2**-24)`、`raw=log(p)-log(1-p)` 生成 raw 输入，
  `A_log=0`、`dt_bias` 缺省、`lower_bound=-5`，算子内执行 safe-gate 激活。
  零概率端点的微小截断用于避免 -Inf，精度参考使用实际 raw 输入重新计算 gamma。
  **其他维度**直接传入已激活的 log-gate，关闭 `safe_gate` 和 `use_gate_in_kernel`。
- KDA 初态/最终状态接口为 FP32。原 BF16 初态先按原 dtype 生成，再在计时外提升到 FP32；
  NC 初态保持 `Dv+1` 的底层 stride。`state_dtype_matched` 明确标记是否与 CGDR 状态接口类型一致。
- 所有正式调用均保持 **chunk_size=64**、`output_final_state=True`、`state_v_first=True`，
  最少导出 attention、final_state、Aqk、Akk。KDA 的 Aqk/Akk 是额外必需输出，CGDR 不输出这两个矩阵。
- **128×128 且 B×Nv>32**：通过公开 **TND + `cu_seqlens_cpu`** 调用，规避 dense 路径的最终状态异常；
  元素数量、head 和 chunk size 均不变。其他条目用 BSND。实际配置逐行记录在 CSV 中。
- `ASC-025` 的四个状态变体将 S 从 **300 补齐到 320**，`ASC-034` 的四个变体从 **2000 补齐到 2048**。
  128×128 的 `ARC-017/018/034/035/036` 共 20 个条目也统一补齐到 64 的倍数：
  **6000→6016、5000→5056、65535→65536**。补充 token 的 Q/K/V/beta 为 0；
  已激活 log-gate 为 0，raw gate 为 -100（FP32 衰减因子为 1），保持有效输出和最终状态不变。
  CPU FP64 另外比较补齐与未补齐递推，验证两者的有效输出和最终状态相同。
  这些时延对应补齐后的实际 shape，不能作为原始 shape 的时延。
- `ASC-023/024/026/031` 的四种变体共 16 例，Dk/Dv 为 **50/50、5/8、1/1、31/31**，
  不满足 KDA `[16,256]` 内 16 倍数的要求。对应时延和精度通过字段留空。

## 精度验证

使用独立 **CPU FP64 逐 token 递推**，逐元素按 K 维衰减状态，再计算 Delta 更新和 attention。
CPU C++ 实现只并行独立 batch/head，已与 Python FP64 递推在 raw/已激活 gate、GQA、非等长序列上交叉验证。
全量检查每条用例的 attention 和最终状态。NC 变体复用 golden 前，检查全部 CPU 输入逐元素相同，
并分别比较各自的 NPU 输出；复用来源记录在 [precision_results.csv](precision_results.csv)。
本轮先完成 288 条精度扫描，再对统一尾块补齐所改变的 20 条 128×128 用例重新校验。
最终精度表保留未改变输入的实测结果，并替换这 20 条复测结果；原始失败结果保留在原始输出目录中。
来源文件、哈希和逐条映射见 [precision_provenance.json](precision_provenance.json)。

沿用 CGDR 泛化测试的 BF16 指标：`relative_rms <= 0.02` 且
`max(abs(error)/max(abs(ref),RMS(ref),2e-5)) <= 0.10`。
这是与数学参考比较的测试门限，不是 CANN 官方精度认证。
此前跨布局一致性测试的 `atol=0.002, rtol=0.02` 不宜直接当作整个长序列递推的判据；
本报告仍保留该旧逐元素门限的通过状态和最大误差比，便于审计，未因本轮结果调整门限。
旧逐元素门限另有 **12** 条未通过，详见
`summary.json` 的 `precision_legacy_allclose_failed_cases` 和精度表的 `legacy_allclose_passed`；
`kda_precision_passed=True` 表示通过上述 RMS/归一化最大误差判据。
CGDR 的历史 `precision_checked` 原样保留为 `cgdr_precision_checked`，未把“未检查”写成“失败”。

## 复测发现的运行问题与规避方式

1. **原始通用尾块仍会等待**：ASC-025、ASC-034 的原始 shape 在扩展调用后停在 NPU 同步，
   见 [tail_probes.json](tail_probes.json) 和 [诊断日志](diagnostics/ASC-025.log)。
   文档声明这些 S 合法，因此这是本机内置实现的运行问题，不能归类为公开 API 不支持。
   正式数据采用上述中性补齐，原始异常记录继续保留，超时不作为 kernel 耗时。
2. **通用维度 raw-gate 路径也观察到等待**：ASC-019 开启 raw safe gate 时复现；
   正式通用维度改用数学等价的预激活输入。不要把测试中使用 raw gate 的 128×128 路径直接推广到所有维度。
3. **dense 多 batch/head 的最终状态不稳定**：ARC-004（B=2、S=2048、Nv=32）重复调用时 attention 通过，
   最终状态出现远大于 BF16 舍入误差的变化，初态未被修改。
   切换 `state_v_first`、预激活 gate、导出 h 或全部中间量均曾复现，见
   [状态诊断](state_diagnostics.json)、[预激活重复诊断](state_pregate_diagnostics.json)。
   TND 路径对同一输入连续 10 次通过，见 [布局诊断](state_layout_diagnostics.json)，因此正式测试对 B×Nv>32 的 128×128 条目使用该路径。
4. **128 维尾块局部误差**：未补齐的 ARC-036 四个状态变体在 attention 上未通过固定门限，
   最大偏差位于最后一个 chunk 的 token 65475；BSND/TND、raw/预激活 gate 均复现，
   见 [尾块诊断](arc036_gate_diagnostics.json)。补齐一个 token 后，该例归一化最大误差从约 0.133 降到约 0.021，
   门限仍为 0.10。正式配置统一补齐所有尾块，共 28 条，并逐例验证补齐的数学等价性。

内核排查建议：重点检查 dense FwdH 从当前 batch/head 的最终状态写回，切换到下一项初态加载时，
MTE3→MTE2 对同一 UB 状态缓冲区的保护；源码 `ProcessAiv` 的最终写回后使用 MTE3→V，
而 `InitializeStateAiv` 随后从 MTE2 写入该缓冲区。这是基于复现与源码的排查线索，尚未通过修改内核证明根因。
通用尾块还应检查前后端共享 TPipe 的事件生命周期，以及 raw gate 与后续阶段的同步。
本轮通过公开接口参数规避，没有宣称修复了 CANN 内核。

## 扩展安装与数据来源

当前分支、更新后的上游 `4638b57da`、本机 torch_npu/op-plugin 均未找到 `chunk_kda_fwd` Python 绑定，
但仓库功能测试明确导入该接口。因此本轮按现有 `OpBuilder` + C++ 扩展规范补齐
[绑定源码](../../../chunk_kda_fwd/torch_extension/chunk_kda_fwd.py)，构建完整 `cann_ops_transformer` wheel，
安装到本轮独立目录并通过 PYTHONPATH 优先加载。仓库原有 **7 项 pytest 全部通过**，见 [日志](extension_pytest.log)。
测试脚本不使用 ctypes 或自行构造 ACLNN 调用；C++ 扩展使用仓库公共桥接层管理 stream、workspace 和张量。

环境为 Ascend950PR_9589（32 AIC/64 AIV）、torch 2.13.0+cpu、torch_npu 2.13.0.rc1、CANN 9.2.0。
CGDR 是历史归档数据，本轮未作交错 A/B 复测。已安装的 custom vendor 只含 CGDR，没有 KDA 覆盖；
库、wheel 和内置 KDA 文件哈希见 [environment.json](environment.json)，实际加载路径见 [loaded_libraries.txt](loaded_libraries.txt)。

| 文件 | 内容 |
|---|---|
| [comparison.csv](comparison.csv) | 288 条完整映射、实际配置、性能和精度状态 |
| [case_performance.csv](case_performance.csv) | 272 条均值/中位数/CV/5 个样本 |
| [kernel_samples.csv](kernel_samples.csv) | 272×6 条主 kernel 样本及同次调用辅助任务耗时 |
| [group_summary.csv](group_summary.csv) | 原始 shape、补齐、状态类型、布局等分组 |
| [precision_results.csv](precision_results.csv) | 独立精度指标、旧逐元素门限和参考复用来源 |
| [metadata.json](metadata.json) | 原始 profile/用例/脚本/产物哈希与测量口径 |
| [validation.json](validation.json) | 行数、样本数、精度和归档完整性检查 |

`kda_api_device_task_sum_mean_us` 统计同次扩展调用的主 kernel、Transpose 和 NC Slice 等设备任务之和，
不含 host 时间和任务间隙。Profile 实测 BSND 每次有 3 个 Transpose 和 1 个主 kernel，
TND 每次有 7 个 Transpose 和 1 个主 kernel；NC 初态各额外增加 1 个 Slice。
其中 TND 增加的 4 次转置将 Q/K/V/g 转为内核的 head-major 布局。
主 kernel 的比值也不能解释成 per-dim 门控单项开销：
门控激活、必需中间输出、状态 dtype、布局和内部流水线均参与差异。
本轮同时调整了 gate 和布局配置，相较旧报告的变化不能单独归因于 Python 调用接口。

复现命令（仓库根目录，使用新的输出目录）：

```bash
cd torch_extension
python3 -m pip install ninja build
python3 setup.py bdist_wheel
cd ..
kda_raw=attention/chunk_kda_fwd/tests/pytest/output/rdv_perf_torch_extension_repeat
python3 -m pip install --no-deps --target "$kda_raw/installed" torch_extension/dist/cann_ops_transformer-1.0.0-py3-none-any.whl
export PYTHONPATH="$PWD/$kda_raw/installed:$PYTHONPATH"
mkdir -p "$kda_raw/build"
python3 -m pytest -q attention/chunk_kda_fwd/tests/pytest/test_chunk_kda_fwd.py > "$kda_raw/build/pytest_binding.log" 2>&1
python3 attention/chunk_kda_fwd/tests/pytest/kda_rdv_perf.py --output "$kda_raw/precision" --mode precision --pad-sequence-tails
mkdir -p "$kda_raw/full/prof"
msprof --output="$PWD/$kda_raw/full/prof" --summary-format=csv --export=on python3 attention/chunk_kda_fwd/tests/pytest/kda_rdv_perf.py --output "$kda_raw/full" --pad-sequence-tails
printf '%s\n' "$?" > "$kda_raw/full/tool_exit_code.txt"
python3 attention/chunk_kda_fwd/tests/pytest/kda_probe_tail.py --output "$kda_raw/tail_probes" --timeout 20
python3 attention/chunk_kda_fwd/tests/pytest/kda_analyze_rdv.py --raw "$kda_raw" --output attention/chunk_gated_delta_rule/docs/kda_perf_repeat
```

状态问题可用 `kda_diagnose_state.py --case ARC-004 --repeats 10 --output <path.json>` 复现。
旧版直接调用 ACLNN 的原始数据及被替换报告保留在本机 `output/rdv_perf_20260908/`，不混入本轮统计。
