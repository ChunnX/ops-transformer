# FlashKDA RDV 性能对比（2026-09-09）

调用 `/home/s00940320/cannbot-dsl/samples/flash_kda/flash_kda.py` 的公开 `flash_kda`，
源仓库版本和文件哈希见 [environment.json](environment.json)。没有修改 DSL 内核或覆盖默认 tiling。
对比 [新版 CGDR](../rdv_perf/new_streaming/comparison.csv) 和 [CANN 内置 KDA](../kda_perf/comparison.csv) 的历史归档主 kernel 耗时。

保留全部 **288** 个 RDV 条目，**236** 条获得性能，
**52** 条不满足 FlashKDA 的 Dk=Dv=128 要求，时延留空。
其中 **20** 条补齐序列，**118** 条将原非连续初态转为连续，均在表中标注。

## 性能数据

每例先独立执行一次 JIT/常量初始化，再记录 1 次预热和 5 次正式测量。
比值为 `FlashKDA / 对照算子`，小于 1 表示 FlashKDA 更快；单位 µs。
只统计 msprof 设备主 kernel 的 `Task Duration(us)`，不含 JIT、输入适配、Python、workspace 分配和 host 等待时间。
用两次独特长度的 Add 标记确定每条用例的边界，标记本身不计时；逐次样本见 [kernel_samples.csv](kernel_samples.csv)。
DSL 直接下发的 kernel 在 msprof 中不提供输入 shape；运行时张量的 shape/dtype/stride 和分块参数
保存在 [execution_results.json](execution_results.json)，并与精度运行的配置核对。

| 范围 | 条目数 | Flash/CGDR 几何平均耗时比 | Flash/CANN KDA 几何平均耗时比 |
|---|---:|---:|---:|
| 全部可执行映射 | 236 | 2.214× | 0.437× |
| 序列不补齐 | 216 | 2.306× | 0.451× |
| 不补齐且原初态连续 | 108 | 2.291× | 0.448× |
| 不补齐、原初态连续且 FP32 | 54 | 2.159× | 0.448× |
| 序列补齐 | 20 | 1.426× | 0.308× |

全部可执行条目中，FlashKDA 有 **33/236** 条快于新版 CGDR，
**236/236** 条快于 CANN KDA。
最大 CV 为 **4.258%**。
全量数据见 [comparison.csv](comparison.csv)，均值、中位数、标准差和五次样本见 [case_performance.csv](case_performance.csv)。

| 用例 | B/S/Nk/Nv | CGDR | CANN KDA | FlashKDA | Flash/CGDR | Flash/CANN KDA |
|---|---|---:|---:|---:|---:|---:|
| ARC-001 | 1/2048/32/32 | 91.967 | 429.495 | 266.481 | 2.898× | 0.620× |
| ARC-004 | 2/2048/32/32 | 174.668 | 1348.759 | 541.412 | 3.100× | 0.401× |
| ARC-009 | 1/262144/32/32 | 9330.127 | 48468.786 | 34538.175 | 3.702× | 0.713× |
| ARC-011 | 1/2048/2/2 | 82.730 | 377.986 | 47.088 | 0.569× | 0.125× |
| ARC-017 | 1/6000/8/8 | 276.672 | 1121.687 | 240.147 | 0.868× | 0.214× |
| ARC-018 | 80/5000/16/16 | 7727.104 | 64377.481 | 33463.425 | 4.331× | 0.520× |
| ARC-036 | 1/65535/16/32 | 2347.418 | 12131.567 | 8721.897 | 3.716× | 0.719× |
| ASC-001 | 2/4096/32/32 | 327.974 | 2617.291 | 1098.595 | 3.350× | 0.420× |

按 B×Nv 分组（包含已标注的输入适配）：

| B×Nv 范围 | 条目数 | Flash/CGDR | Flash/CANN KDA |
|---|---:|---:|---:|
| B×Nv ≤ 4 | 20 | 0.639× | 0.170× |
| 4 < B×Nv ≤ 16 | 56 | 1.230× | 0.331× |
| 16 < B×Nv ≤ 32 | 84 | 3.169× | 0.691× |
| B×Nv > 32 | 76 | 3.186× | 0.413× |

FlashKDA 默认对较小 B×Nv 使用 Dv 分块：B×Nv≤4/8/16 时，`dv_base` 分别为 16/32/64，
其他为 128。源码还按 B×Nv 选择 chunk group，并在 group 内复用 workspace。
这些设计能增加小 B×Nv 场景的工作划分，但本轮没有固定其他条件做 tiling 消融测试，
因此不把分组性能差异单独归因于某一项设计。

## 输入约束与对比范围

- 原 RDV 的 B/S/Nk/Nv、BF16 Q/K/V、初态值生成类型、seed=20260908 保留。新环境的 PyTorch 版本不同，历史输入没有逐字节快照，因此不声称精确回放历史输入。
- FlashKDA 只支持 **Dk=Dv=128**；其余 52 条维度用例保留为 `UNSUPPORTED_SHAPE`，不填性能或精度通过。
- S 必须为 64 的倍数。ARC-017/018/034/035/036 的四个状态变体共 20 条，按 **6000→6016、5000→5056、65535→65536** 补齐。
  Q/K/V 补 0，raw gate/beta 补 -100，衰减在 FP32 中为 1，零 key 和 value 保证有效输出和最终状态不变。
- 公共接口只接受连续 FP32 初态。原 BF16 初态提升到 FP32，原 NC 初态真实构造后再 `.contiguous()`；转换在计时外。
  NC 条目的时延属于转换后的连续输入，不能用来说明内核原生支持 NC。CSV 保留源 stride 和转换标记。
- 使用 **BSND**、chunk_size=64、默认 group 和 Dv 分块。FlashKDA 内部执行 Q/K L2 normalize（epsilon=1e-6），且将结果量化为 BF16。
  本轮传入与先前 RDV 相同生成方式的归一化 Q/K，因此 FlashKDA 会再次归一化；这部分计算包含在 FlashKDA 主 kernel 中。
- FlashKDA 的 gate 和 beta 都是 **BF16 原始 logits**。per-dim 目标 gate 使用 seed=20260909；有 gate 时将目标 log-gate 逆变换为 safe-sigmoid raw 输入，A_log=0、dt_bias=0、lower_bound=-5。
  无 gate 时 lower_bound=0。beta 由 RDV 的 [0,1) 系数逆 sigmoid 后量化到 BF16，零端点用 -100 避免 -Inf。
  精度参考使用实际量化后的 logits，并重新执行激活，不把目标系数当作实际系数。
- 历史 CANN KDA 接收 FP32 gate、已激活 beta；CGDR 是标量 gate。FlashKDA 还融合 Q/K normalize 和 beta 激活。
  因此这些比值是完整实现之间的比较，不能解释为仅 per-dim 门控或仅 DSL 语言的开销。
- CANN KDA 另需导出 Aqk/Akk，并对部分用例采用 TND；FlashKDA 只返回 attention/final state。两个对照均来自历史测量，没有在本轮交错重测。

## 精度与环境验证

原仓库 `test/flash_kda/test_flash_kda.py` 的 NPU 测试通过，见 [original_pytest.log](original_pytest.log)。
RDV 独立检查 **236** 条，其中 **236** 条通过；
失败列表：`[]`。只对实际通过的条目填写 `flash_precision_passed=True`。

RDV 参考先按接口定义执行 Q/K 归一化和 gate/beta 激活，然后使用独立 CPU FP64 逐 token 递推。
原 NC 输入转换为连续后，如果全部实际输入以及两项 NPU 输出均与此前验证的用例逐元素完全相同，
则复用该用例的 FP64 误差；每条用例仍独立调用 FlashKDA，验证方式与复用来源逐条记录。
沿用前一轮 RMS/归一化最大误差判据：`relative_rms<=0.02`、`normalized_max<=0.10`、分母绝对下限 2e-5。
另外保留原样例 `atol=rtol=0.005` 的逐元素指标，未按测试结果改变门限。
该逐元素门限未通过 **0** 条，名单保存在 `summary.json`。
全部误差、参考复用与补齐等价性状态见 [precision_results.csv](precision_results.csv)，
原始精度执行记录见 [precision_execution_results.json](precision_execution_results.json)；不适用的条目明确留空，不能视为通过。

## 安装与复现

PyPI 发行包名称为 **cannbot-dsl**，导入名称为 `cannbotdsl`，0.0.3 要求 Python 3.12。
来源见 [PyPI](https://pypi.org/project/cannbot-dsl/)，安装记录和源码哈希见 [environment.json](environment.json)。
本轮独立环境为 `/home/s00940320/.venvs/flash-kda-py312`，torch 2.12.0+cpu、torch_npu 2.12.0，仍使用系统 CANN 9.2.0。
前两轮对照使用 Python 3.10、torch/torch_npu 2.13 系列，差异已记录。

安装包名与 Python 导入名不同。新环境可先执行：

```bash
python3.12 -m venv /path/to/flash-kda-py312
/path/to/flash-kda-py312/bin/python -m pip install --index-url https://pypi.org/simple cannbot-dsl==0.0.3 numpy ml_dtypes pytest PyYAML ninja
/path/to/flash-kda-py312/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.0
/path/to/flash-kda-py312/bin/python -m pip install --index-url https://pypi.org/simple torch-npu==2.12.0
```

本机复现命令：

```bash
source /home/s00940320/.venvs/flash-kda-py312/activate_flash_kda.sh
export MAX_JOBS=4
flash_raw=attention/chunk_kda_fwd/tests/pytest/output/flash_kda_rdv_repeat
export CANNBOTDSL_CACHE_DIR="$PWD/$flash_raw/compiler_cache"
python attention/chunk_kda_fwd/tests/pytest/flash_kda_rdv_perf.py --output "$flash_raw/precision" --mode precision
mkdir -p "$flash_raw/full/prof"
msprof --output="$PWD/$flash_raw/full/prof" python attention/chunk_kda_fwd/tests/pytest/flash_kda_rdv_perf.py --output "$flash_raw/full"
printf '%s\n' "$?" > "$flash_raw/full/tool_exit_code.txt"
python attention/chunk_kda_fwd/tests/pytest/flash_kda_analyze_rdv.py --raw "$flash_raw" --output attention/chunk_gated_delta_rule/docs/flash_kda_perf_repeat
```

[validation.json](validation.json) 保存行数、样本边界、数据配对和精度配置检查；
[metadata.json](metadata.json) 保存原始 profile、基线、脚本和产物 SHA256。
