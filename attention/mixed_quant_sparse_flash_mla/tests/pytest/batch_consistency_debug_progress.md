# MQSMLA Batch 一致性问题定位进展

> 最后更新：2026-08-08（第三轮，关闭 batch consistency 后全量通过 + 规约类型分析）
> 调试树：`attention/mixed_quant_sparse_flash_mla`
> 方法论：ascendc-batch-consistency-debug skill（路径确认 → 复现收敛 → 逻辑行定位 → metadata 映射 → 数据链首个分歧点）

---

## 1. 问题描述

Batch 换序（mode1）或拆分（mode2）后，同一逻辑样本的 attention 输出 MD5 与 baseline 不一致。
测试入口：`test_mixed_quant_sparse_flash_mla_batch_consistency.py`，通过 `test_run.sh batch_consistency` 执行。

判定方式：`_assert_same_md5` 对每个逻辑样本的 attention output（和 softmax LSE）分别做 MD5 比对。
失败时输出 MD5、shape、mismatch_count、first_mismatch 坐标和 max_abs_diff。

---

## 2. 测试执行结果

### 2.1 执行命令

```bash
bash test_run.sh batch_consistency \
    --pt-dir /tmp/mqsmla-pr8737-verify/attention/mixed_quant_sparse_flash_mla/tests/pytest/mqsmla_testcase \
    --consistency-mode all --consistency-seed 20260806 --device-id 0
```

### 2.2 总览

| 指标 | 数量 |
|------|------|
| PT 用例总数 | 19 |
| 跳过（B<2） | 10 |
| 执行 batch 一致性用例（B>=2） | 8 |
| 单元测试 | 5 |
| **FAILED** | **4** |
| PASSED | 20 |
| SKIPPED | 20 |
| 执行耗时 | 32.50s |

### 2.3 失败用例明细

#### 用例 1：`qsmla_case_1_core_has_2_reduce`（B=33, BSND, HCA）

| 字段 | 值 |
|------|-----|
| PT 路径 | `/tmp/mqsmla-pr8737-verify/attention/mixed_quant_sparse_flash_mla/tests/pytest/mqsmla_testcase/qsmla_case_1_core_has_2_reduce.pt` |
| B | 33 |
| layout_q | BSND |
| layout_kv | BSND |
| template_run_mode | HCA |
| return_softmax_lse | False |

**失败 1 — mode1 换序**

| 字段 | 值 |
|------|-----|
| label | `mode=mode1-reorder, sample=27, output=attention` |
| expected_md5 | `249dec422ae036b2826ed8b1c26f8be3` |
| actual_md5 | `ae27c9c256ab5f25444411f01320fd78` |
| shape | (1, 64, 512) |
| mismatch_count | 3/32768 |
| first_mismatch | (0, 4, 211) |
| max_abs_diff | 0.000244140625 |

**失败 2 — mode2 拆分**

| 字段 | 值 |
|------|-----|
| label | `mode=mode2-split-0, sample=2, output=attention` |
| expected_md5 | `ae70c66b16f3c1b8fb3f01db3bfa9e79` |
| actual_md5 | `7966237c84521e8bd0d4b9d66757c112` |
| shape | (1, 64, 512) |
| mismatch_count | 2/32768 |
| first_mismatch | (0, 16, 255) |
| max_abs_diff | 0.03125 |

#### 用例 2：`qsmla_case_1_core_has_2_reduce_with_no_reduce_insight`（B=33, BSND, HCA）

| 字段 | 值 |
|------|-----|
| PT 路径 | `/tmp/mqsmla-pr8737-verify/attention/mixed_quant_sparse_flash_mla/tests/pytest/mqsmla_testcase/qsmla_case_1_core_has_2_reduce_with_no_reduce_insight.pt` |
| B | 33 |
| layout_q | BSND |
| layout_kv | BSND |
| template_run_mode | HCA |
| return_softmax_lse | False |

**失败 3 — mode1 换序**

| 字段 | 值 |
|------|-----|
| label | `mode=mode1-reorder, sample=30, output=attention` |
| expected_md5 | `3a612d34da7edcd2e0d03c0e8a58d960` |
| actual_md5 | `17f477bbe1d90b3fca7ab92eddb8f637` |
| shape | (1, 64, 512) |
| mismatch_count | 1/32768 |
| first_mismatch | (0, 0, 41) |
| max_abs_diff | 0.0001220703125 |

**失败 4 — mode2 拆分**

| 字段 | 值 |
|------|-----|
| label | `mode=mode2-split-0, sample=2, output=attention` |
| expected_md5 | `7966237c84521e8bd0d4b9d66757c112` |
| actual_md5 | `ae70c66b16f3c1b8fb3f01db3bfa9e79` |
| shape | (1, 64, 512) |
| mismatch_count | 2/32768 |
| first_mismatch | (0, 16, 255) |
| max_abs_diff | 0.03125 |

> 注意：失败 2 和失败 4 的 expected/actual MD5 互换（`ae70c66b...` ↔ `7966237c...`），说明 sample=2 在两个用例中是相同数据，mode2 拆分后结果一致地偏移。

### 2.4 通过用例

| 用例名 | B | layout_q | layout_kv | mode | 结果 |
|--------|---|----------|-----------|------|------|
| `multi_batch_full_cores` | 8 | BSND | BSND | reorder + split | PASSED |
| `residual` | 5 | TND | TND | reorder + split | PASSED |
| `tnd_padding_kv` | 2 | TND | TND | reorder + split | PASSED |
| `with_kvs_0` | 9 | BSND | BSND | reorder + split | PASSED |
| `with_qs_0` | 9 | BSND | BSND | reorder + split | PASSED |
| `with_tail` | 3 | TND | TND | reorder + split | PASSED |

### 2.5 跳过用例（B<2）

`K_1024`、`N128`、`N2`、`N32_single_batch_3_cores`、`single_batch_2_cores`、
`single_batch_3_cores`、`single_batch_full_cores`、`single_batch_full_cores_with_blocks`、
`testest`、`tnd_padding_1`

---

## 3. 失败模式分析

### 3.1 共性特征

1. **B=33 边界**：两个失败用例均为 B=33，恰好等于 `BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM = 33`（`mixed_quant_sparse_flash_mla_common_arch35.h:33`）。
2. **1 core has 2 reduce**：用例名暗示单个核内有多个 reduce block，涉及核内归约路径。
3. **mode2 sample=2 一致**：两个用例的 mode2-split-0 失败均出现在 sample=2，MD5 互为 expected/actual，说明是同一数据在同一 split 分组下的分歧。
4. **mismatch 极少**：mode1 仅 1-3 个元素不一致（32768 中），mode2 仅 2 个元素不一致。max_abs_diff 从 0.000122 到 0.03125。
5. **仅 attention output 失败**：`return_softmax_lse = False`，无 LSE 比对。

### 3.2 初步假设

- **B=33 触发 reduce block 边界**：`BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM = 33` 可能导致 workspace 上界或 reduce block 分配的 off-by-one。
- **核内归约状态残留**：单个核内多个 reduce block 时，online-softmax 状态（max/sum）可能在新 reduce block 开始时未正确重置。
- **mode2 更大误差**：mode2-split 的 max_abs_diff（0.03125）远大于 mode1-reorder（0.000122-0.000244），暗示拆分改变了 reduce block 数量或分布，放大了状态残留效应。

### 3.3 已排除

- **路由错误**：日志确认所有运行均进入 `BE IS_BATCH_CONSISTENCY` 路径（printf 标记可见）。
- **输入不一致**：`assert_sample_inputs_unchanged` 在所有用例中均通过，排除样本选择错误。
- **B<33 的用例**：B=2/3/5/8/9 的用例全部通过，排除通用的 batch 一致性逻辑错误。

---

## 4. 当前源码状态

### 4.1 Git 基线

- 最新 commit：`bffafc75b`（"111"）
- 上一相关 commit：`a440268ec`（"fix batch一致性 coff=2时刷新state_cache错误的bug"，2026-08-06）
  - 关联 issue：https://gitcode.com/cann/ops-transformer/issues/4158

### 4.2 未提交改动清单

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `op_host/mixed_quant_sparse_flash_mla_tiling.cpp` | 强制路径 | `batchConsistency_ = true`；添加 `OP_LOGE` 打印 DeterministicLevel |
| `op_host/aclnn_mixed_quant_sparse_flash_mla_metadata.cpp`（metadata 算子包） | 强制路径 | `isBatchConsistency = true` |
| `op_kernel/arch35/mixed_quant_sparse_flash_mla_csa_kernel.h` | 路径标记 | 添加 `printf("BE/NOT BE IS_BATCH_CONSISTENCY\n")` |
| `op_kernel/arch35/mixed_quant_sparse_flash_mla_csa_block_vector.h` | LSE 修复 | `ReduceIntraBlockAndStage` 中 `copyOutMergedLse` 分支：调用 `ComputeLogSumExp_VF` 计算 LSE，改从 `sumBroadcastUb` 写 GM |
| `attention/sparse_flash_mla/.../sparse_flash_mla_csa_block_vector.h` | 同步 LSE 修复 | 与 mqsmla 相同改动 |
| `attention/sparse_flash_mla/.../common/flash_decode.h` | LSE 恢复修复 | `MergeStagedAndCurrentChunk`：用 `RecoverMaxSumFromLse` 替换 `Duplicate(mergedSumBroadcastUb, 1.0, ...)` |
| `attention/common/.../vf/vf_flash_decode.h` | 新增工具函数 | 添加 `RecoverMaxSumFromLseVF` 和 `RecoverMaxSumFromLse`（从 LSE 反推 max/sum） |
| `tests/pytest/test_mixed_quant_sparse_flash_mla_batch_consistency.py` | 新文件 | batch 一致性测试框架 |
| `tests/pytest/conftest.py` | 新增选项 | `--consistency-mode/seed/split-count/batch-groups` |
| `tests/pytest/test_run.sh` | 新增命令 | `batch_consistency` 命令和 single `--batch-consistency` |
| `tests/pytest/mixed_quant_sparse_flash_mla_paramset.py` | 新增用例 | `zhengwang` / `zhengwang4` |
| `tests/pytest/test_mixed_quant_sparse_flash_mla_single.py` | 集成 | 导入 `batch_consistency_level` 和 `run_batch_consistency_case` |

### 4.3 强制路径注意事项

当前工作树强制 `batchConsistency_ = true`，设置 `MQSMLA_DET_LEVEL=0` 不会走普通路径。
测试 fixture `batch_consistency_level` 会 `set_deterministic_level(1)`，但 tiling 已忽略该值。

---

## 5. 关键代码位置速查

| 功能 | 文件 | 行号 |
|------|------|------|
| batchConsistency_ 强制 | `op_host/mixed_quant_sparse_flash_mla_tiling.cpp` | :84 |
| BATCH_CONSISTENCY_LEVEL 常量 | `op_host/mixed_quant_sparse_flash_mla_tiling.cpp` | :27 |
| BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM | `op_kernel/arch35/mixed_quant_sparse_flash_mla_common_arch35.h` | :33 |
| workspace 上界计算 | `op_host/mixed_quant_sparse_flash_mla_tiling.cpp` | :652 |
| 路径 printf 标记 | `op_kernel/arch35/mixed_quant_sparse_flash_mla_csa_kernel.h` | :193/:196 |
| LSE 输出修复 | `op_kernel/arch35/mixed_quant_sparse_flash_mla_csa_block_vector.h` | :1408 |
| 跨核 FD 入口修复 | `attention/sparse_flash_mla/op_kernel/arch35/common/flash_decode.h` | :411 |
| RecoverMaxSumFromLse 实现 | `attention/common/op_kernel/arch35/vf/vf_flash_decode.h` | :651 |
| IS_BATCH_CONSISTENCY 分支集合 | `op_kernel/arch35/mixed_quant_sparse_flash_mla_csa_block_vector.h` | :1191+:1270+:1333+:1375+:1462+:1472+:1479+:1505+:1768+:1822+:1826 |
| batch consistency 测试入口 | `tests/pytest/test_mixed_quant_sparse_flash_mla_batch_consistency.py` | :572 |
| det_level fixture | `tests/pytest/test_mixed_quant_sparse_flash_mla_batch_consistency.py` | :600 |
| 样本选择 | `tests/pytest/test_mixed_quant_sparse_flash_mla_batch_consistency.py` | :162 |
| MD5 比对 | `tests/pytest/test_mixed_quant_sparse_flash_mla_batch_consistency.py` | :426 |

---

## 6. 下一步计划

1. **审计 B=33 reduce block 边界**：检查 `BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM = 33` 的 workspace 上界是否被 B=33 用例触及，以及 `baseBlockNumInTile` 的 128 对齐和 32 份均衡策略在 B=33 时的 off-by-one。
2. **定位失败逻辑行**：对 `1_core_has_2_reduce` 用例实现逐逻辑行/逐 head 诊断，从 `first_mismatch` 坐标反推 FA owner 和 reduce block。
3. **审计核内归约状态协议**：检查每个新 reduce block 是否正确重置 online-softmax 状态（max/sum），特别是 `s2LoopCount == 0` 与 `isFirstBase` 的关系。
4. **构造 hierarchical-FD 双参考**：排除 benchmark 计算顺序误差。
5. **恢复 `batchConsistency_` 条件赋值后验证普通路径回归**。

---

## 7. 变更日志

| 日期 | 变更 |
|------|------|
| 2026-08-06 | commit `a440268ec`：fix batch一致性 coff=2时刷新state_cache错误的bug |
| 2026-08-08 | 第一轮测试（旧包）：执行全量 batch 一致性测试（19 PT 用例）；定位失败用例为 B=33 的 `1_core_has_2_reduce` 系列（4 个失败）；记录失败明细和分析 |
| 2026-08-08 | 第二轮测试（换包后）：相同用例全量通过，**24 passed, 20 skipped, 0 failed**；之前失败的 `1_core_has_2_reduce` 和 `1_core_has_2_reduce_with_no_reduce_insight`（B=33）的 mode1 换序和 mode2 拆分均 PASSED |
| 2026-08-08 | 第三轮确认：第二轮通过是因为关闭了 batch consistency（走普通路径），batch consistency 路径仍有 bug |
| 2026-08-08 | 规约类型分析：模拟 AICPU 调度，确认所有 4 个失败样本都涉及核内规约；仅 sample=27 同时涉及核内+核间规约；3/4 失败仅为核内规约。根因方向：核内归约路径 |

---

## 8. 第二轮测试结果（换包后）

### 8.1 执行命令

```bash
bash test_run.sh batch_consistency \
    --pt-dir /tmp/mqsmla-pr8737-verify/attention/mixed_quant_sparse_flash_mla/tests/pytest/mqsmla_testcase \
    --consistency-mode all --consistency-seed 20260806 --device-id 0
```

### 8.2 总览

| 指标 | 第一轮（旧包） | 第二轮（新包） |
|------|---------------|---------------|
| FAILED | 4 | **0** |
| PASSED | 20 | **24** |
| SKIPPED | 20 | 20 |
| 耗时 | 32.50s | 33.61s |

### 8.3 之前失败用例的状态

| 用例 | B | mode1-reorder | mode2-split |
|------|---|---------------|-------------|
| `1_core_has_2_reduce` | 33 | FAILED → **PASSED** | FAILED → **PASSED** |
| `1_core_has_2_reduce_with_no_reduce_insight` | 33 | FAILED → **PASSED** | FAILED → **PASSED** |

### 8.4 结论

换包后 batch 一致性问题已修复。所有 B>=2 用例（含 B=33 边界用例）在 mode1 换序和 mode2 拆分下均通过 MD5 一致性验证。

> **修正**：第二轮通过的原因是用户关闭了 batch consistency（走普通路径），而非修复了 batch consistency 路径的 bug。
> batch consistency ON 时仍有 4 个失败，OFF 时全量通过，说明 **batch consistency 路径本身存在 bug**。

后续建议：
1. 恢复 `batchConsistency_` 条件赋值，验证普通路径回归
2. 删除 `csa_kernel.h` 中的 `printf` 路径标记
3. 增加 20 次重复运行验证稳定性
4. 扩展 B=33 附近的边界用例（B=32, B=34）

---

## 9. 第三轮：关闭 batch consistency 后全量通过

### 9.1 操作

用户安装了关闭 batch consistency 的包（`batchConsistency_` 不再强制 `true`，走普通 FD 路径）。

### 9.2 测试结果

```bash
bash test_run.sh batch_consistency \
    --pt-dir /tmp/mqsmla-pr8737-verify/attention/mixed_quant_sparse_flash_mla/tests/pytest/mqsmla_testcase \
    --consistency-mode all --consistency-seed 20260806 --device-id 0
```

**24 passed, 20 skipped, 0 failed**（33.61s）

### 9.3 结论

| 路径 | 结果 | 说明 |
|------|------|------|
| batch consistency ON（第一轮） | 4 failed | B=33 的 `1_core_has_2_reduce` 系列失败 |
| batch consistency OFF（第三轮） | 0 failed | 全量通过 |

**batch consistency 路径本身存在 bug**，普通路径无问题。

---

## 10. 失败用例规约类型分析

### 10.1 分析方法

由于当前包 batch consistency OFF 无法获取 S2 切分 metadata，使用 Python 模拟 AICPU 调度逻辑
（`mixed_quant_sparse_flash_mla_metadata_aicpu.cpp` 的 `CalcSplitPlan` → `AssignBlocksToCore`）。

关键参数：
- `BATCH_CONSISTENCY_MAX_REDUCTION_PARTS = 32`
- `s2BaseSize = 128`
- `reductionBlockSize = (actTotalS2 / 32 + 127) / 128 * 128`
- 32 AIC cores, 33 batches → 负载均衡导致部分 batch 的 S2 被切分到多个核

### 10.2 用例参数

两个失败用例参数相同：

| batch | ori_kv | cmp_kv | total_s2 | reduction_blocks | s2_loop |
|-------|--------|--------|----------|-----------------|---------|
| 0 | 256 | 64 | 320 | 3 | 3 |
| 1-31 | 1024 | 256 | 1280 | 10 | 10 |
| 32 | 512 | 128 | 640 | 5 | 5 |

- `reductionBlockSize = 128`（对所有 batch 均如此）
- 总 reduction blocks = 3 + 31×10 + 5 = 318
- 平均每核 = 318/32 ≈ 9.94

### 10.3 模拟调度结果

32 核分配 33 个 batch 时，奇数 batch（1, 3, 5, ..., 31）的 S2 被切分到 2 个核：

| 核 | 分配 | s2_loop | 核内规约 | 跨核切分 |
|----|------|---------|----------|----------|
| 0 | b0(全部,3blk) + b1(前6blk) | 3 | YES | YES (b1) |
| 1 | b1(后4blk) | 0 | NO | YES (b1) |
| 2 | b2(全部,10blk) + b3(前1blk) | 10 | YES | YES (b3) |
| 3 | b3(后9blk) | 0 | NO | YES (b3) |
| ... | 偶数核: 完整batch + 下一个batch的1块 | 10 | YES | YES |
| ... | 奇数核: 剩余9块 | 0 | NO | YES |
| 30 | b30(全部,10blk) + b31(前2blk) | 10 | YES | YES (b31) |
| 31 | b31(后8blk) | 0 | NO | YES (b31) |

### 10.4 失败样本规约类型

| 用例 | 失败样本 | 模式 | 分配核 | 核内规约 | 跨核规约 | 类型 | mismatch | max_abs_diff |
|------|----------|------|--------|----------|----------|------|----------|-------------|
| `1_core_has_2_reduce` | 27 | mode1-reorder | core26(1blk) + core27(9blk) | YES (core26: s2_loop=10) | YES (b27 split) | **BOTH** | 3/32768 | 0.000244 |
| `1_core_has_2_reduce` | 2 | mode2-split-0 | core2(10blk, 全部) | YES (s2_loop=10) | NO | **intra-only** | 2/32768 | 0.03125 |
| `1_core_has_2_reduce_with_no_reduce_insight` | 30 | mode1-reorder | core30(10blk, 全部) | YES (s2_loop=10) | NO | **intra-only** | 1/32768 | 0.000122 |
| `1_core_has_2_reduce_with_no_reduce_insight` | 2 | mode2-split-0 | core2(10blk, 全部) | YES (s2_loop=10) | NO | **intra-only** | 2/32768 | 0.03125 |

### 10.5 关键发现

1. **所有 4 个失败样本都涉及核内规约**（s2_loop=10，即每个核有 10 个 128-token 基本块需要核内合并）。
2. **只有 sample=27 同时涉及核内规约和核间规约**（BOTH），其余 3 个失败仅涉及核内规约（intra-only）。
3. **没有失败样本仅涉及核间规约**（cross-only）。
4. **intra-only 的失败也存在**（sample 2, 30），说明 bug 不只在核间 FD，**核内归约路径本身也有问题**。
5. **BOTH 的误差不一定更大**：sample 27 (BOTH) 的 max_abs_diff=0.000244，而 sample 2 (intra-only) 的 max_abs_diff=0.03125（更大）。

### 10.6 分析结论

- **核内规约是必要条件**：所有失败样本都有核内规约。没有核内规约的核（s2_loop=0，如 core 1/3/5/...）未出现失败。
- **核间规约非必要条件**：3/4 的失败不涉及核间规约。
- **核间规约非充分条件**：涉及核间规约但无核内规约的核未出现失败。
- **根因方向**：优先排查核内归约路径（`ReduceIntraBlockAndStage`、`MergeStagedAndCurrentChunk`），特别是 `RecoverMaxSumFromLse` 替换 `Duplicate(sum, 1.0)` 后的数值协议正确性。

### 10.7 注意事项

- 以上调度为 Python 模拟结果，实际 AICPU 调度器的 `AssignByBatch → AssignByRow → AssignByBlock → ForceAssign` 逻辑更复杂，具体核分配可能有差异。
- `s2_loop` 在模拟中取当前核处理的最大 batch 的 s2_loop，实际 kernel 中 `FA_S2_MAX_NUM` 是所有核的最大值。
- 需要用 batch consistency ON 的包运行 metadata 算子获取精确调度来确认。
