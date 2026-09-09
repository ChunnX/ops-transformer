# CGDR 泛化与精度验证（2026-09-08）

本轮新增 **616 个 pytest 用例，实测 614 通过、2 失败、0 跳过**。544 个合法输入、精度及语义用例全部通过；失败均为非法维度未被第一段接口拒绝，CANN 原版也有相同行为。本轮增加测试和报告，未修改算子实现，也未将失败标为 xfail。

输入契约来自 [aclnnChunkGatedDeltaRule.md](../aclnnChunkGatedDeltaRule.md)。完整覆盖矩阵、阈值定义和运行环境设置见 [pytest README](../../tests/pytest/README.md#泛化与精度验证新增)。

## 覆盖与结果

| 类别 | 用例数 | 通过 | 失败 |
|---|---:|---:|---:|
| 通用精度：维度边界、尾块、head、batch、gate、scale、stride、多 seed | 356 | 356 | 0 |
| 8 个核心网络形状 × BF16/FP32 state | 16 | 16 | 0 |
| 确定性、gate 等价、scale、batch/head/布局关系 | 28 | 28 | 0 |
| 单 token 闭式解、因果性、零值、输出边界、广播、分流边界 | 144 | 144 | 0 |
| 非法输入：nullptr、dtype、rank、空 tensor、关联 shape、维度范围 | 72 | 70 | 2 |
| 合计 | 616 | 614 | 2 |

网络形状均为 B=1、Dk=Dv=128、含 gate，T∈{3500,7000,8000,11264}，Nk/Nv∈{16/32,14/28}。通用用例覆盖 Dk/Dv=1～128 的代表性边界与非对称组合、chunk 两侧长度、Nv/Nk 分组、零长度子序列、非连续输入和 BF16/FP32 state；16 个额外 seed 为 1000～1015，其他用例固定为 20260908。

合法输入保持文档约束：Q/K 包含正负数，先归一化再转 BF16；检查量化后的范围及范数误差。beta 的边界取 BF16 可表示的严格内点，不使用 0 或 1。gate 包括 nullptr、0、-1、接近 0、交替和范围内随机值。

## 精度方法与实测误差

新参考实现在 CPU FP64 上逐 token 递推，使用输入量化后的实际值、调用者传入的 FP32 scale 和真实 state dtype。参考不依赖当前算子的 Cube/Streaming 实现，不舍入 BF16 中间量；额外使用单 token 闭式解核验参考函数。旧 golden 会重新设置默认 scale，旧 benchmark 会按 BF16 处理 state，因此新测试不依赖它们。

固定判定标准为相对 RMS ≤2%，局部归一化最大误差 ≤10%，并同时检查有限性、shape、dtype。局部误差分母为 `max(abs(reference_i), RMS(reference), 2e-5)`，因此下表的“局部最大”不是直接逐元素相对误差。该标准是测试标准，不是接口文档承诺的精度上限。

442 个用例保存了两项输出的完整误差指标；其他用例的精度断言、闭式解或逐元素相等断言仍实际执行，CSV 中无误差指标不代表跳过测试。

| 已记录指标的范围 | 最大相对 RMS | 最大局部归一化误差 |
|---|---:|---:|
| 全部已记录精度用例 | 1.377233% | 7.685493% |
| 16 个核心网络用例 | 0.370209% | 6.052319% |

全局两项最大值均来自 `beta_near_one_one_hot_128_128_float32` 的 attention。网络最大 RMS 来自 T=8000、Nk=14、FP32 state；网络局部最大来自 T=11264、Nk=16、FP32 state。具体输出和原始数值见 [summary.json](summary.json) 与 [case_results.csv](case_results.csv)。FP32 state 不改变 Q/K/V/beta 和 attention 为 BF16 的接口约束。

## 发现的契约问题

| 失败用例 | 期望 | 优化包实测 | CANN 内置接口实测 |
|---|---|---:|---:|
| `dimension_nv_0` | 拒绝 Nv=0 | 0（成功） | 0（成功） |
| `dimension_dv_0` | 拒绝 Dv=0 | 0（成功） | 0（成功） |

两者都只调用 `aclnnChunkGatedDeltaRuleGetWorkspaceSize`，没有发射非法 kernel。构造时同步调整 value、beta、state、gate 和输出 shape，避免被其他 shape 不匹配错误提前遮盖。

文档要求 `0<Nv<=64`、`0<Dv<=128`。当前 [ACLNN 实现](../../op_host/op_api/aclnn_chunk_gated_delta_rule.cpp)在检查空指针和 dtype 后，如果 `out->IsEmpty() && finalState->IsEmpty()`，就直接成功返回，未进入后续维度检查。这解释了两种输入被接受的原因。内置接口的函数地址通过 `dladdr` 确认为 CANN `libopapi.so`，其第一段也返回成功，结果见 [baseline_preflight.json](baseline_preflight.json)。该核对未执行内置 kernel，不是原版全套精度验证。

另外，错误码与文档存在差异：8 个 required nullptr 用例实际返回 161002，文档为 161001；部分 shape/rank 校验返回 561103 或 561101，文档列出的 shape 错误码为 561002。72 个非法用例的实测返回码分布为 161002×25、561103×44、561101×1、0×2。当前测试断言“必须拒绝”，并记录具体错误码；70 个通过不能解释为错误码完全符合文档。

建议后续修复时将必要的 shape/正维度检查移到空输出成功返回之前，并统一错误码与文档；这两个普通失败用例可直接作为回归验证。

## 环境与版本核验

- Ascend950PR_9589，device 0；CANN 9.2.0；torch 2.13.0+cpu、torch_npu 2.13.0.rc1。
- 当前源码 HEAD 为 `f6d8cd6ff2d5b621b33ca938b08e8ca62aaaf439`。使用已构建的 `cgdr_stream_update_transformer` 隔离 vendor，直接绑定其 `libcust_opapi.so`；未替换系统安装目录。
- 核对包内发布的 kernel 源码与工作区对应文件一致；运行库路径见 [runtime.json](runtime.json) 和 [network_runtime.json](network_runtime.json)。
- msopprof BasicInfo 抽样执行 `diagonal_5_bfloat16`，实际 kernel 二进制 SHA-256 为 `ff88b552dc9cd8bbacbd622d017bfba7ff8d58ba98cd54d931cb4c84fce8aba4`，与当前优化包对应对象完全一致。采集只在子进程中桥接 CANN 9.2 的 ACL API 拦截函数，未改动算子或系统文件。该证据是一次抽样审计，不代表对每个用例都采集了二进制。

源码及包校验记录见 [metadata.json](metadata.json)，采集记录见 [kernel_basic_info.csv](kernel_basic_info.csv)。旧 single 入口另跑 1 个用例通过，并核对原 CSV 列格式保持兼容，见 [legacy_smoke.csv](legacy_smoke.csv)；它不计入 616 个新用例。

## 复现与结果文件

按 pytest README 配置 CANN 和 `CGDR_TEST_VENDOR` 后，在 `tests/pytest` 下运行：

```bash
python3 -m pytest -q \
  test_chunk_gated_delta_rule_generalization.py \
  test_chunk_gated_delta_rule_invariants.py \
  test_chunk_gated_delta_rule_validation.py \
  --junitxml=output/generalization/results.xml
```

运行前创建 `output/generalization`。当前算子版本预期报告 2 个上述失败，退出码为 1。只运行合法输入可去掉 validation 文件；仅快速用例追加 `-k 'not network_precision'`。

本轮完整统计由互不重复的 `quick_final.xml`（600 个）和 `network.xml`（16 个）合并。对应 pytest 会话耗时分别约 23.5 秒和 572 秒，主要长耗时来自 CPU FP64 参考计算，不是算子性能数据。原始 XML、日志、复现脚本和 profiling dump 位于本机 `tests/pytest/output/generalization_20260908/`，该目录不纳入 Git；可共享的逐用例结果及摘要已归档在本文同目录。

这是代表性输入组合验证，不是所有形状和数值分布的穷举，也不覆盖未承诺的 in-place、输出别名、图模式、并发 stream 或其他芯片。文档要求调用者保证的非法数值、负序列长度和不满足 sum(lengths)=T 的输入未送入 kernel。
