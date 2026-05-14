# QAR 算子性能验收标准

> 版本 1.0 | 2026-05-14 | Server 8p 正式验收

---

## 1. 适用范围

| 算子 | 数据类型 | 卡数 | 验收环境 |
|------|---------|------|---------|
| `npu_quant_all_reduce` (QAR) | int8 / hif8 / fp8（sizeof=1B） | 2 / 4 / 8 | Server 8p（正式）；POD 8p / 标卡 4p / EVB 2p（开发自测） |

---

## 2. 性能模型

QAR kernel 时间由三部分组成：

```
T_kernel = T_sync  +  T_mte  +  T_compute
           └同步┘    └数据搬运┘  └反量化+规约┘
```

- **T_sync**：ReadStatus 软同步，等待 N 个 rank 完成 Win 区数据写入。开销 ∝ N
- **T_mte**：MTE 引擎从各 rank Win 区读取量化数据（x + scale）到 UB。数据量 ∝ V × N
- **T_compute**：AIV 核执行反量化（dequant）+ ReduceSum 累加。计算量 ∝ V × N

三项均为 workload 参数 V（有效载荷）和 N（卡数）的线性函数，合并为：

```
T_kernel(V, N) = T₀(N) + V / BW_eff(N)

其中：
  V       = bs × h × 1B       有效载荷（MB）
  N       = world_size         卡数
  T₀      = 固定开销（us）：kernel 启动 + ReadStatus × N
  BW_eff  = 等效带宽（GB/s）：MTE 搬运 + 反量化计算 + 流水线开销
```

### 2.1 参数定义依据

| 参数 | 物理来源 | 估算方法 |
|------|---------|---------|
| T₀ | kernel launch（~5 us）+ ReadStatus 忙等 N 个 rank 状态位（~2-3 us/rank） | 理论：5 + 3×N |
| BW_eff | MTE 读带宽受 AIV 核数、反量化计算吞吐、HCCL 传输层延迟共同限制 | 上限 ≤ min(MTE_peak, AIV_compute_throughput) = 56 GB/s |

---

## 3. 验收参考参数

### 3.1 Server 8p（正式验收）

```
T₀_ref     = 20 us
BW_eff_ref = 21.0 GB/s
```

| 项 | 值 | 依据 |
|----|----|------|
| T₀ | 20 us | kernel launch ~5 us + ReadStatus(8 rank) ~15 us |
| BW_eff | 21.0 GB/s | Server HCCL 传输层延迟较高，等效带宽约为 MTE 峰值的 37% |

### 3.2 开发自测环境

| 环境 | N | T₀_ref (us) | BW_eff_ref (GB/s) |
|------|---|------------|-------------------|
| EVB 2p | 2 | 8 | 29.0 |
| 标卡 4p | 4 | 10 | 25.0 |
| POD 8p | 8 | 23 | 26.4 |

> 注：开发环境参数作为自测参考，仅 Server 8p 参数用于正式验收判定。

---

## 4. 验收判定

### 4.1 时间预算

```
T_budget(V, N) = T₀_ref(N) + V / BW_eff_ref(N)

Server 8p:  T_budget(V) = 20 + V / 21.0      (V 单位 MB，T 单位 us)
```

### 4.2 判定规则

```
T_actual ≤ T_budget × 1.10   →  ACCEPT
T_actual  > T_budget × 1.10   →  REJECT（性能退化，需定位根因）
```

10% 容差覆盖测量噪声和 tiling 块对齐微扰。

### 4.3 等价速率表示

```
BW_actual = V / T_actual
BW_min    = V / (T_budget × 1.10)

BW_actual ≥ BW_min  →  ACCEPT
```

---

## 5. 测试条件

| 项目 | 要求 |
|------|------|
| **数据类型** | int8（sizeof=1B）；hif8 / fp8 可复用同标准 |
| **V 覆盖范围** | 0.01 ~ 12 MB，至少覆盖 L0-L5 各档一个点（见附录 A） |
| **采集方式** | `torch_npu.profiler.profile`，10 轮迭代，warmup=0，active=1 |
| **T_kernel 取值** | 从 `op_summary.csv` 取非 AI_CPU 任务 Task Duration 平均值 |
| **环境变量** | `HCCL_BUFFSIZE=2048` |
| **前置条件** | 被测 NPU 无其他进程占用 |
| **Scale 类型** | fp32（per-128-group）；fp8_e8m0（MXFP，per-32-group）需单独验证 |

---

## 6. 自检：Server 8p 实测 26 shape 全量 PASS

| V (MB) | T_budget (us) | T_actual (us) | 比值 | 判定 |
|--------|--------------|--------------|------|------|
| 0.031 | 21.5 | 15.5 | 0.72 | ACCEPT |
| 0.063 | 23.0 | 16.0 | 0.70 | ACCEPT |
| 0.125 | 26.0 | 17.8 | 0.68 | ACCEPT |
| 0.219 | 30.4 | 23.2 | 0.76 | ACCEPT |
| 0.406 | 39.4 | 34.2 | 0.87 | ACCEPT |
| 0.711 | 53.9 | 41.2 | 0.77 | ACCEPT |
| 0.766 | 56.5 | 47.1 | 0.83 | ACCEPT |
| 1.500 | 91.4 | 90.4 | 0.99 | ACCEPT |
| 3.000 | 162.9 | 171.6 | 1.05 | ACCEPT |
| 4.500 | 234.3 | 247.1 | 1.05 | ACCEPT |
| 6.000 | 305.7 | 306.9 | 1.00 | ACCEPT |
| 9.000 | 448.6 | 448.7 | 1.00 | ACCEPT |
| 12.00 | 591.4 | 582.8 | 0.99 | ACCEPT |

> 小 V 段 12 个 shape 省略，比值均 < 0.90。全 26 shape max(ratio) = 1.05，全部 ACCEPT。

---

## 附录 A：推荐测试 shape

| 档位 | V 范围 (MB) | 推荐测试 shape（4 卡，h=4096~8192） |
|------|------------|----------------------------------|
| L0 | < 0.1 | [4, 4096], [16, 4096] |
| L1 | 0.1 ~ 0.3 | [28, 4096], [52, 4096] |
| L2 | 0.3 ~ 0.7 | [100, 4096], [100, 7168] |
| L3 | 0.7 ~ 2.0 | [112, 7168], [192, 8192] |
| L4 | 2.0 ~ 6.0 | [384, 8192], [768, 8192] |
| L5 | > 6.0 | [1152, 8192], [1536, 8192] |

> 不同卡数用 `perf_bandwidth.py --world_size N` 自动推导 shape。

## 附录 B：参考数据

完整实测数据见 `QAR_performance_metric_analysis.md`，覆盖 EVB 2p / 标卡 4p / POD 8p / Server 8p 四环境共 104 shape。
