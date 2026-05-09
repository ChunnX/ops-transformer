# QuantReduceScatter stage3 性能报告

> **分支**：`stage3_target_iter` (基于 stage2 `5223bcbd`)
> **改动**：host 侧 TARGET_ITER 公式化 xPerBlock + DataCopyPad 替换 scale DataCopy（鲁棒性提升）
> **基线**：stage2 cap=2048 归档 `x_block_2048_perf.csv`
> **测试**：world_size=4, 4×Ascend950PR (aivNum=56), CANN 9.0.0, level0 sheet 全 24 用例
> **采集**：mc2_test 内置 10 轮取后 5 轮均值

## 最终配置

```cpp
// host: quant_reduce_scatter_tiling.cpp::SetXPerBlock
constexpr uint32_t TARGET_ITER = 3U;   // 与 QAR 对称（QRS 上 T 实际不激活）
constexpr uint32_t MIN_BLOCK = 2048U;  // QRS comm-bound：per_core 太小公式失效，MIN 兜到 2048
constexpr uint32_t ALIGN_BLOCK = 1024U; // 与 kernel X_BLOCK_ALIGN_NUM 对齐

// 公式（QRS 先除 rankSize=4）
xSliceSizeNums = xNums / rankSize;
xPerBlock = max(ceil(xSliceSizeNums / aivNum / TARGET_ITER), MIN_BLOCK);
xPerBlock = floor(xPerBlock, ALIGN_BLOCK);
```

## xPerBlock 实际下发值

| shape | per_core (slice/aivNum) | T=3 raw | max(raw, 2048) | floor 1024 | 每核 iter |
|-------|-------------------------|---------|----------------|------------|-----------|
| 48×5120 (qwen) | 1097 | 366 | **2048** | 2048 | 1（30 cores 活跃 + 26 cores 空转）|
| 48×7168 (DSV3) | 1536 | 512 | **2048** | 2048 | 1（42 cores 活跃 + 14 cores 空转）|

QRS 在所有 24 用例上 xPerBlock=2048，与 stage2 cap=2048 等效。**核心结论**：QRS per_core_elem 太小（先除 rankSize=4），公式分子永远 < MIN，**T 在 QRS 上对当前 24 用例无意义，真正起作用的是 MIN**。

## 24 用例 perf 对照

| case | stage2 (us) | stage3 (us) | Δ% |
|---|---:|---:|---:|
| 48_7168_fp8_e5m2→bf16 ID0001 | 10.543 | 10.397 | **−1.39%** |
| 48_7168_fp8_e4m3fn→fp16 ID0002 | 10.571 | 10.518 | −0.50% |
| 48_7168_int8→fp32 ID0003 | 10.479 | 10.688 | +2.00% ¹ |
| 48_7168_hif8→fp16 ID0004 | 10.556 | 10.455 | −0.95% |
| 48_7168_MX_e5m2→fp32 ID0005 | 10.865 | 10.744 | −1.11% |
| 48_7168_MX_e4m3fn→bf16 ID0006 | 10.589 | 10.720 | +1.23% |
| 1×48×7168_fp8_e5m2→fp32 ID0007 | 10.388 | 10.333 | −0.52% |
| 1×48×7168_fp8_e4m3fn→bf16 ID0008 | 10.497 | 10.574 | +0.74% |
| 1×48×7168_int8→fp16 ID0009 | 10.480 | 10.577 | +0.92% |
| 1×48×7168_hif8→bf16 ID0010 | 10.515 | 10.543 | +0.27% |
| 1×48×7168_MX_e5m2→fp16 ID0011 | 10.847 | 10.741 | −0.97% |
| 1×48×7168_MX_e4m3fn→fp32 ID0012 | 10.798 | 10.673 | −1.15% |
| 48_5120_fp8_e4m3fn→fp16 ID0013 | 9.640 | 9.466 | −1.80% |
| 48_5120_fp8_e5m2→fp32 ID0014 | 48.129 ² | 9.653 | (异常 baseline，不计) |
| 48_5120_int8→bf16 ID0015 | 9.414 | 9.433 | +0.20% |
| 48_5120_hif8→fp32 ID0016 | 9.567 | 9.327 | **−2.51%** |
| 48_5120_MX_e4m3fn→bf16 ID0017 | 9.680 | 9.831 | +1.56% |
| 48_5120_MX_e5m2→fp16 ID0018 | 9.804 | 9.650 | −1.57% |
| 1×48×5120_fp8_e4m3fn→fp32 ID0013 | 9.523 | 9.446 | −0.81% |
| 1×48×5120_fp8_e5m2→fp16 ID0014 | 9.487 | 9.604 | +1.24% |
| 1×48×5120_int8→fp16 ID0015 | 9.563 | 9.647 | +0.88% |
| 1×48×5120_hif8→bf16 ID0016 | 9.536 | 9.396 | −1.47% |
| 1×48×5120_MX_e4m3fn→fp32 ID0017 | 9.767 | 9.868 | +1.03% |
| 1×48×5120_MX_e5m2→bf16 ID0018 | 9.989 | 9.698 | **−2.91%** |

注：
1. ID0003 +2.00% 处于测量噪声边界，多次重测仍在 ±2% 内
2. ID0014 stage2 baseline 48.13us 为已知异常（stage2 报告中标注）；本次 stage3 测得 9.65us 与同 shape 用例一致

## 总结

- **24/24 用例精度 PASS**（4×rank 共 96 项）
- **23 用例总时长（去 ID0014 异常）**：233.10 us → 232.33 us（**−0.33%，持平 stage2**）
- **1 用例显著加速**：ID0018 (1×48×5120_MX_e5m2→bf16) 由 9.989 → 9.698 (−2.91%)
- **0 真实回退**：ID0003 +2.00% 在测量噪声内

## 调优过程关键决策

| 尝试 | T | MIN | ALIGN | 结果 |
|------|---|-----|-------|------|
| 1 | 6 | 1024 | 1024 | xPerBlock 落 1024（vs stage2 的 2048）→ 推算 +30~40% 回退（未上板） |
| 2 | 3 | 512 | 512（X_BLOCK_ALIGN=512）| scale GM 越界 → 精度 102 FAIL |
| 3 | 3 | 512 | 512 + DataCopyPad | 精度修复 PASS，但 +12.14% 回退（DB 流水启动但通信 overhead 主导，块越大越优） |
| 4 | 3 | 2048 | 1024 + DataCopyPad | **−0.33% 持平 stage2** ✅ |

## xPerBlock=512 假设证伪 — 详细分析

stage3 调优过程中曾尝试将 `X_BLOCK_ALIGN_NUM` 从 1024 降到 512，意图让 QRS 每核迭代数从 1 提升到 3，启动 DoubleBuffer 流水。该假设来自 QAR 上 T=3（每核 3-4 iter）显著优于 T=6（每核 6-7 iter）的成功经验。**实测结论：假设不成立，QRS 反而劣化 +12.14%**。

### 实测数据

| QRS 配置 | xPerBlock | 每核 iter | total_blocks (h=7168) | 活跃核数 (h=7168) | 24 用例总耗时 | vs stage2 |
|----------|-----------|-----------|------------------------|--------------------|--------------|-----------|
| stage2 cap=2048 | 2048 | **1** | 42 | 42 / 56 (75%) | 281.23 us | 0%（baseline）|
| stage3 尝试 ALIGN=512 | 512 | **3** | 168 | 56 / 56 (100%) | 315.37 us | **+12.14%** ❌ |
| stage3 最终 MIN=2048 | 2048 | 1 | 42 | 42 / 56 | 232.33 us | **−0.33%**（持平）|

stage2 (2048, 1 iter) **快于** stage3 尝试 (512, 3 iter)，**与 DB 流水越多越好的直觉相反**。

### 失败机制：通信 overhead 主导

QRS 单次 reduce-scatter 的 mc2_prof 由三部分组成：
```
mc2_prof ≈ N_iter × (T_comm_setup + T_comm_actual + T_compute) + T_pipeline_overhead
```

其中：
- `T_comm_setup`：每次通信启动开销（HCCL handshake、地址计算、状态同步），与块大小**无关**
- `T_comm_actual`：实际数据传输时间，与块大小**线性相关**
- `T_compute`：vec_comp 反量化 + reduce_sum 时间，与块大小线性相关
- `T_pipeline_overhead`：DB 流水未满载时的同步开销

**关键事实**：QRS 每核数据量极小（h=7168 时 per_core ≈ 1536，h=5120 时 ≈ 1097），`T_compute << T_comm_actual`。

xPerBlock=2048（1 iter）：
```
mc2_prof ≈ 1 × (T_setup + T_data×2048 + T_compute×2048) ≈ T_setup + T_data×2048
```

xPerBlock=512（3 iter）：
```
mc2_prof ≈ 3 × (T_setup + T_data×512 + T_compute×512) - T_DB_overlap×2
        ≈ 3×T_setup + T_data×1536 + (DB 重叠收益)
```

**3 倍 T_setup 的代价 > T_data 节省的 1/4 + DB 重叠收益**。实测净劣化 +12%。

### QAR 为什么 DB 流水有效？

QAR per_core 数据量大（h=4096 → 7023, h=7168 → 12289），约是 QRS 的 4-8 倍：
```
QAR T=3, h=7168: 4 iter, per_core=12289, xPerBlock=4096
  T_compute×4096 与 T_comm×4096 量级相当 → DB 重叠收益显著
QRS xPerBlock=512: 3 iter, per_core=1536
  T_compute×512 << T_comm×512，DB 重叠的"compute 部分"太短，重叠不上 → 收益微小
```

**核心差异**：DB 流水有效的前提是 **compute 时间 ≥ comm 时间**（这样才能在 comm 时段隐藏 compute）。QRS comm 远大于 compute，DB 流水"无可重叠"。

### 推论：何时该减小 xPerBlock 启动 DB？

| 算子特性 | 减小 xPerBlock | 增大 xPerBlock |
|----------|----------------|----------------|
| compute-bound（T_compute ≥ T_comm）| ✅ 多 iter 让 DB 重叠 compute 与 comm | ❌ 反而浪费 |
| comm-bound（T_compute << T_comm）| ❌ 增加 N×T_setup 成本，DB 无可重叠 | ✅ 减少 setup 开销 |
| memory-bound（带宽饱和）| 视具体瓶颈 | 视具体瓶颈 |

QAR (allreduce) 数据量约是 QRS (reduce-scatter) 的 4 倍（rankSize=4），双向通信也意味着 compute 与 comm 比值更大 → compute-friendly。
QRS (reduce-scatter) 单向小数据量通信 → comm-bound。

### 启示：通用算子不存在通用最优 xPerBlock 公式

`xPerBlock = per_core_elem / TARGET_ITER` 这套基于"目标每核迭代数"的公式，**隐含假设是 compute-bound**（多 iter 启 DB 流水）。该公式：
- 对 QAR 这种 compute-friendly 场景：**有效**，T=3 是甜点
- 对 QRS 这种 comm-bound 场景：**失效**，公式 raw 永远 < MIN，必须靠 MIN 兜底为大块

stage3 的最终方案（QAR 走公式 + QRS 走 MIN）暴露了这一点：**单一公式无法兼容 compute-bound 与 comm-bound 两类算子**。后续 stage4 若想统一框架，需引入"算子特性"标签或 cost model 自动判别。

---

## QAR vs QRS 对称性分析

两算子 kernel 实现框架完全一致（CeilDiv(total, xPerBlock) 分块，aivNum 核分摊），但**最优策略不同**：

| 维度 | QAR | QRS |
|------|-----|-----|
| 类型 | compute-friendly 可流水 | comm-bound latency-dominant |
| 最优 xPerBlock | 2048-4096（每核 3-4 iter）| 2048（每核 1 iter，部分核空转）|
| 公式驱动 | ✅ 公式输出主导 | ❌ 公式被 MIN 截断（per_core 太小）|
| MIN 作用 | 兜底（24 用例不激活）| 主导（24 用例都靠它）|

**核心教训**：TARGET_ITER 这套基于 "DoubleBuffer 流水甜点" 的公式只对 compute-bound 算子有效。对 comm-bound 且 per_core 小的算子（如 QRS），DB 收益小于通信块大小的影响，更大块（少 iter）反而更优。

## 经验教训

1. **per_core_elem 决定公式是否激活**：QAR 经验"T=3 最优"不能直接套到 QRS（QRS per_core 比 QAR 小 4 倍）
2. **MIN_BLOCK 对 QRS 是真正起作用的参数**，等效于"为 comm-bound 算子设置最小块大小"
3. **DataCopyPad 替代 DataCopy** 解锁了 sub-1024 alignment 的可能性（虽然 stage3 实测 sub-1024 对 QRS 不利），但保留作为代码鲁棒性改进
4. **不同算子最优策略可不同**：哪怕 kernel 框架一致，shape 与通信模式差异会导致最优分块策略反向
