# Batch-Invariance (BI) 测试 — all_gather_matmul_v2

本目录提供 `npu_all_gather_quant_mm`（AllGatherMatmulV2）算子的 batch-invariance 测试套件，作为 issue #2956 提出的「MC2 算子官方测试套件增加 BI 测试维度」需求的 **参考实现**。

## 1. 为什么需要 BI 测试

现有 UT/ST 测试覆盖：
- `acc`：与 reference 实现的数值精度对比（tolerance 内）
- `perf`：性能阈值（吞吐 / 延迟）

但不覆盖以下两个 BI 维度：

### 维度 A：Row-BI（literature batch-invariance）

```
y[i] = f(x[i], W)

输出第 i 行仅依赖于输入第 i 行 + 共享权重 W，
与同一 batch 中其他行的内容无关。
```

文献定义参考：<https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference>

### 维度 B：Cross-rank BI（issue #2872）

```
同一 op 调用中，所有 rank 的 y 输出必须 byte-exact 一致。
```

`AllGatherMatmulV2` 所有 rank 共享相同的 all-gathered x1 和 x2，因此输出 y 必须完全相同。PR #6373 修复前，LOCAL/GATHER 双路径分裂导致 WS=2 + varying x2_scale 时跨 rank 输出不一致（567/2048 bytes differ）。

## 2. Methodology Note —— 流程缺一不可

> **本节为重点**。每个 BI cell 走以下 phase：

### Phase 0：HCCL warm-up

Worker 启动后 HCCL state 在前几次调用容易处于 cold-start 状态，第一次 op 调用经常 `HcclAllocComResourceByTiling failed`（error code 561000），即便成功也可能选择跑非确定的 ring 拓扑。

**做法**：测试矩阵正式开始前，先做 `N_WARMUP=2` 次 dummy 调用（用任意合法 input），让 HCCL state 稳定。warm-up 中的 error 不计入测试结果。

### Phase 1：Multi-trial determinism baseline

每个 cell 的 variant A 用同一个 input 跑 `N_TRIALS=3` 次，所有 SHA 必须完全一致；variant B 同样。

```python
for _ in range(N_TRIALS):
    sha_list_a.append(run(seed_a))
assert len(set(sha_list_a)) == 1  # variant A 必须 deterministic
```

若 sha_list_a 不一致 → 标记为 `NON_DETERMINISTIC`，该 cell **不进入 Phase 2**（因为 A vs B 比较已无意义）。

### Phase 2：Variant A vs Variant B Row-BI 检查

只有 Phase 1 通过，才把 sha_a vs sha_b 比较：
- 同 → `BI_PASS`（行间独立）
- 不同 → `BI_FAIL`（真 BI 违例）

### Phase 3：Cross-rank BI 检查（每次 trial 同步执行）

每次 op 调用后，通过 `dist.all_gather` 收集所有 rank 的 full-y tensor，逐字节比较：
- 所有 rank 输出 byte-exact 一致 → 该 trial cross-rank agree
- 多次 trial 均 agree → `X_PASS`
- 多次 trial 均 disagree 且 byte diff 模式稳定 → `X_FAIL`（真 cross-rank BI 违例）
- trial 间 cross-rank 结果不稳定 → `X_NON_DET`

### Row-BI 状态输出

| 状态 | 含义 | 含义解读 |
|---|---|---|
| `BI_PASS` | det baseline 通过 + A == B | row 内容跨 batch 独立 ✓ |
| `BI_FAIL` | det baseline 通过 + A ≠ B | **真 BI 违例**，需修复 |
| `NON_DETERMINISTIC` | 同 input 多次跑 SHA 不一致 | 测试系统问题（HCCL state / 共享 NPU / 硬件），需排查；**不能简单判 BI fail** |
| `ERROR` | op 调用抛异常 | op-plugin shape 拒绝或基础设施故障 |

### Cross-rank BI 状态输出

| 状态 | 含义 | 含义解读 |
|---|---|---|
| `X_PASS` | 所有 trial 所有 rank SHA 一致 | 跨 rank 输出一致 ✓ |
| `X_FAIL` | trial 稳定但跨 rank byte diff > 0 | **真 cross-rank BI 违例**（issue #2872 类型） |
| `X_NON_DET` | cross-rank byte diff 在 trial 间不稳定 | 测试系统问题，需排查 |

**关键教训**：把 `BI_FAIL` 与 `NON_DETERMINISTIC` 混淆会产生大量假阳性，淹没真 bug。

## 3. 算子说明

`npu_all_gather_quant_mm` 是 AllGatherMatmulV2 的 PTA 接口。每个 rank 持有 x1 的一个分片（M_per_rank × K），通过 all-gather 拼接为 M_total × K（M_total = M_per_rank × world_size），再与共享权重 x2（K × N）做量化矩阵乘，输出 M_total × N。

**所有 rank 得到相同的完整输出**（M_total × N），这使得 cross-rank BI 检查成为可能：直接比较各 rank 的 y 是否 byte-exact。

与 `matmul_reduce_scatter_v2`（natively BI-clean，每 rank 输出不同行）不同，`AllGatherMatmulV2` 在 PR #6373 修复前同时存在：
- Row-BI 违例：gather 路径与 local 路径的分裂导致行间相互影响
- Cross-rank BI 违例：WS=2 + varying x2_scale 时 LOCAL/GATHER 双路径 cube 状态 diverge

## 4. 矩阵参数

| 维度 | 取值 |
|---|---|
| WorldSize | {2, 4, 8} |
| M_per_rank | {4, 8, 16}（M_total = M_per_rank × WS） |
| K | {256, 512, 1024}（多次跨 cube baseK=128 边界） |
| N | {64, 128}（含 baseN=64 边界） |
| 量化 scale | (x1=ID, x2=VAR) — K-axis varying，已知 BI 触发条件 |
| 目标 row 位置 | 0 |
| N_WARMUP | 2 |
| N_TRIALS | 3 |

约 27 cells × WS={2,4,8} = 81 cells，每个 cell 6 次 op 调用（3 trials × 2 variants），每次调用同时做 Row-BI + Cross-rank 检查，CI wall-clock 估算 < 15 min。

## 5. 如何运行

需要 V300 (Ascend 950PR/DT) 硬件 + CANN 9.1+ + torch_npu 2.9+。

```bash
cd mc2/all_gather_matmul_v2/tests/bi/
bash run_bi_test.sh
```

`run_bi_test.sh` 会以 WS=2/4/8 三次启动 `bi_test_driver.py`，每次打印：

```
=== all_gather_matmul_v2 BI rigorous WS=2 N_WARMUP=2 N_TRIALS=3 ===
  Phase 0: HCCL warm-up...
  [Row-BI]   BI_PASS=27 BI_FAIL=0 NON_DET=0 ERROR=0 SKIP=0
  [XRank-BI] X_PASS=27 X_FAIL=0 X_NON_DET=0
```

- `BI_FAIL > 0`：row-BI 违例，需修复算子
- `X_FAIL > 0`：cross-rank BI 违例（issue #2872 类型），需修复算子
- `NON_DET / X_NON_DET > 0`：需排查测试系统（HCCL state / 共享 NPU / 硬件）
- `X_FAIL` 输出中 `byte_diffs` 显示各 rank 与本 rank 的字节差异数，便于定位 divergence 范围
- `ERROR`：通常是 op-plugin shape rejection

## 6. 与 ATK 框架的集成（待讨论）

当前实现作为 standalone Python driver，不依赖 ATK。集成时需保留三个关键点：

1. `n_warmup` 和 `n_trials` 必须可配置（这是 BI 方法学的核心，不是 boilerplate）
2. Row-BI 和 Cross-rank BI 结果需独立汇总，不能合并
3. `NON_DETERMINISTIC > 0` 不应该自动判 op 失败，但要写入测试报告引起关注
