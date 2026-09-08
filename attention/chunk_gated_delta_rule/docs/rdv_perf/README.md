# CGDR RDV 性能结果（2026-09-08）

三个版本分别保存与 **CANN 9.2.0 内置实现** 的对比。数据来自已完成的测量，本次整理没有重新测试。CSV使用UTF-8 BOM编码和LF换行，保留原始性能数值，并合并独立精度测试的通过记录。

| 版本目录及主对比表 | 性能覆盖 | 版本及采样口径 | 已确认精度通过 |
|---|---:|---|---:|
| [old_streaming/comparison.csv](old_streaming/comparison.csv) | 288例 | `8432b2351`；CANN和旧版均为每例1次预热+5次测量 | 0例有对应记录，288例未确认 |
| [forced_cube/comparison.csv](forced_cube/comparison.csv) | 8例 | `8432b2351`加[强制Cube分流补丁](forced_cube/force_cube.patch)；CANN为1+5，Cube为4+20 | 8/8例 |
| [new_streaming/comparison.csv](new_streaming/comparison.csv) | 288例 | `6f599eb3a`，Streaming逻辑来自`121832027`；两版均为1+5 | 8/288例，其余280例未确认 |

旧、新Streaming目录均代表对应的完整算子版本，仍按参数自动选择Streaming或Cube路径。强制Cube目录仅含实际测量的ASC-024、ASC-026及FP32、NC变体共8例，没有借用其他版本的数据补成288例。三份主表具有相同列结构，`baseline_*`均指CANN原版，`optimized_*`指所在目录的版本。

`speedup = baseline_mean_us / optimized_mean_us`，大于1代表优化版本更快；耗时单位均为µs，指标为msprof的CGDR kernel `Task Duration(us)`。CV使用样本标准差（ddof=1），百分数列不需要再乘100。旧版几何平均加速为5.437×；新版288例均快于CANN，几何平均加速5.724×，各case均值之和的加速比为7.084×，后者不等同业务端到端加速。

`precision_checked=True`表示**该优化版本、该RDV用例已通过独立精度测试**：`ENABLE_PROF=false`，检查attention和finalState。匹配用例名、shape、状态dtype、布局、has_g和seed后合并到性能表，不要求精度和性能在同一次运行中采集，也不表示CANN基线另行通过了精度验证。`False`表示尚无对应精度通过记录，不能解释为精度失败；性能模式的`execution_status=PASSED`不作为精度通过依据。

强制Cube和新版各8例通过记录分别保存在 [Cube精度结果](forced_cube/precision/pytest_results.csv) 和 [新版精度结果](new_streaming/precision/pytest_results.csv)，各自的`precision/`目录还保留运行环境与退出码。新版另外通过了[24例补充边界验证](new_streaming/boundary_results.json)，这些不是288例RDV中的对应行，因此不据此扩大`True`的范围。现有记录没有288例全量独立精度回归。

各版本的补充结果如下，复测与首轮数据分开保存：

- 旧Streaming：[8例重点复测与CANN对比](old_streaming/selected_comparison.csv)（旧版4+20）、[4例退化结果](old_streaming/regressions.csv)、[全量统计](old_streaming/summary.json)。
- 强制Cube：[本轮复测与CANN对比](forced_cube/recheck_comparison.csv)（Cube4+20）、[旧Streaming/Cube消融对比](forced_cube/ablation_comparison.csv)、[消融统计](forced_cube/ablation_summary.json)、[与CANN对比统计](forced_cube/summary.json)。主表使用前轮消融结果，复测表使用同一消融包的本轮结果，不拼接两轮样本。
- 新Streaming：[8例重点复测与CANN对比](new_streaming/selected_comparison.csv)（新版4+20）、[本轮综合统计](new_streaming/summary.json)。全量批跑18例CV超过5%，保留原始均值，不用补测样本替换。

涉及多个版本的汇总统一保存在`cross_version/`：

| 文件 | 内容 | 样本口径 |
|---|---|---|
| [version_comparison.csv](cross_version/version_comparison.csv) | 288例CANN、旧版、新版三方对比 | 各版1+5 |
| [selected_comparison.csv](cross_version/selected_comparison.csv) | 8例旧/新Streaming及两轮强制Cube结果 | CANN为1+5，其他为4+20 |
| [control_comparison.csv](cross_version/control_comparison.csv) | 10个疑似退化Cube用例的旧/新版复测 | 每版4+20 |
| [confirmation.csv](cross_version/confirmation.csv) | ARC-031-FP32、ARC-031-NC的100次确认 | 每版20+100，新版先测 |
| [msopprof_comparison.csv](cross_version/msopprof_comparison.csv) | ASC-024旧/新Streaming按核角色聚合的指标 | 每版1个kernel，同角色32核平均 |

跨版本8例表中的`historical_cube_us`对应强制Cube主表，`fresh_cube_us`对应其复测表。msopprof表的`compute_active_pct`对cube0表示Cube管线，对vector0/vector1表示Vector管线；空单元格代表不适用。6行是两个版本各3种核角色，不能当作6次独立测量；其探针/重放耗时不能与msprof的重复测量混算，各管线占比也不能相加为墙钟时间分解。

完整分析见[性能分析报告](../rdv_regression_analysis_20260908.md)。[metadata.json](metadata.json)记录版本、采样、精度口径、归档文件及来源的SHA-256；原始来源路径以算子目录为基准。大型profile、编译包和完整日志仍在`tests/pytest/output/rdv_perf_20260908/`。
