# 阶段 1 性能验证实验日志（round 001）

2026-04-30 进行。配套 `summary.md`（结论）、`*.csv`（数据）、`profiling_id0007/`（流水）。

## 关键过程

1. **去重 22/48 用例**：按 `(logical_shape × quant_mode × outdtype)` 去重，2D/3D 等价（kernel 总元素数相同）；input_dtype 都 1 字节可合并；T-G vs MX 路径不同必须分开。改 xlsx 紧凑排（mc2_test 按 row index 跑 `--num 1 N`），测完恢复。
2. **构建 baseline**：`git checkout fdbcd206 -- mc2/`（不动 HEAD），build 装到 `/home/m00947052/SUB_baseline`。验证：optimized binary 比 baseline 大约 1KB。
3. **跑两版**：先 OPT 后 BASELINE，新 shell 重新 source 避免 LD_LIBRARY_PATH 污染。每版 22 用例。
4. **对比**：22/22 中 20 个变慢，2 个变快。AR 平均 +5.27%，RS 平均 +2.14%。
5. **profiling 找根因**：跑 ID0007 各一次（`Profiling_Result_*/` 会被覆盖，跑前先 `cp -r` 备份）。取 iter 3-10 稳态（iter 1 冷启动、iter 2 受 Exp 重算影响）。
6. **结论**：算子是 MTE2-bound（aiv_mte2_ratio ≈ 87%），Vec/Scalar 早就被 DoubleBuffer 掩盖。阶段 1 让它们提前结束反而让 MTE2 在尾段暴露。

## 踩过的坑

- **跨用户 NPU 冲突**：跑前必查 `npu-smi info` Process id 列；同用户残留 `sudo kill`，**别人进程不能 kill**，等他跑完
- **端口 50001 EADDRINUSE**：上一次失败遗留的 TCP store。`ss -tlnp | grep :50001` 确认空再跑，**不要换端口绕过**
- **rank 0 缺席的伪精度 FAIL**：HCCL init_process_group 失败 → 单条用例 result FAIL with 1.6% 通过率，看似精度问题实是 rank 0 没参与
- **awk 解析 CSV 列错位**：mc2_test CSV 含 `"[96, 7168]"` 带逗号字段，必须 `csv.DictReader`，不能 `awk -F','`（awk 错位时不报错，只能肉眼发现）
- **CSV 漏行**：第一次 OPT 跑因 HCCL 挂掉在中间，CSV 只落 ID0001。对比时发现 base 10 行 vs opt 9 行才察觉。**对比脚本必须先验行数**

## 工具沉淀：mc2_test 已等价 msprof 10 轮稳态

| 项 | mc2_test 做法 |
|---|---|
| 多轮循环 | `for i in range(10):` 见 `op_class/aclnnQuantAllReduce.py:269` |
| 防 host bound | iter 1 插 `for _ in range(50): torch.exp(10000×10000 fp16)` |
| 输出 | `Profiling_Result_*/ASCEND_PROFILER_OUTPUT/kernel_details.csv` 含 `aiv_vec/scalar/mte2/mte3_ratio` 等 |

→ **不需要再写 msprof 包装**，直接用 mc2_test 输出。

## (a)(b)(c) 耦合关系

(c) 是修 (a)(b) 引入的"CopyDataToWin 1024B 与 Execute xPerBlock 不一致"bug，不是修原始代码 bug。原始 fdbcd206 两侧都 1024B 一致分块没问题。三件改动是耦合包，要么全保留要么全回退。

## 给阶段 2 的话

- 推荐 `git branch stage2 fdbcd206 && git checkout stage2`，从干净 baseline 起步（详见 plan §5.2.0）
- 主要看 `aiv_mte2_time/ratio` 是否下降，**不要看 vec 提速**（已在 DoubleBuffer 下被掩盖）
- profiling 直接用 mc2_test，归档到 `round_002/`
