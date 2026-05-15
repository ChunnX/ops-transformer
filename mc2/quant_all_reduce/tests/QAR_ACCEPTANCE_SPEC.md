# QAR 算子性能验收标准

> 版本 2.0 | 2026-05-14 | Server 8p 正式验收

---

## 1. 适用范围

| 算子 | 数据类型 | 卡数 | 验收环境 |
|------|---------|------|---------|
| `npu_quant_all_reduce` (QAR) | int8 / hif8 / fp8（sizeof=1B） | 2 / 4 / 8 | Server 8p（正式）；POD 8p / 标卡 4p / EVB 2p（开发自测） |
| Scale 类型 | fp32（per-128-group）或 fp8_e8m0（MXFP，per-32-group） | — | 实测验证两者性能等同，共用同标准 |

---

## 2. 性能模型

QAR 算子基于 MTE 引擎直读 Win 区（`SetCommEngine(3)`），kernel 时间由固定开销 + 数据搬运时间组成：

```
T_kernel = T_sync + T_data

T_sync  = 5 + 2N                 (us)    kernel 启动 + 等待 N 个 rank 完成
T_data  = V × N / (56 × η)       (us)    MTE 跨卡搬运
```

参数定义：

| 符号 | 含义 | 确定方式 |
|------|------|---------|
| V | 有效载荷 `bs × h × 1B`（MB） | 测试 shape |
| N | 卡数（world_size） | 部署配置 |
| 56 GB/s | MTE 单口峰值带宽 | 硬件规格 |
| η(N) | 流水线等效深度 | N=2→1, N=4→2, N=8→3 |

η 的物理含义：MTE 读和 AIV 反量化/规约计算通过 double/triple buffer 流水重叠。N 越大，Win 区越多，计算窗口越大，掩盖 MTE 读延迟的能力越强。

---

## 3. 预算带宽

合并后：

```
T_budget(V, N) = (5 + 2N) + V × N / (56 × η(N))
               = (5 + 2N) + V / BW(N)

BW(N) = 56 × η(N) / N
```

| N | η | BW(N) | T_budget |
|---|----|--------|----------|
| 2 | 1 | **28 GB/s** | `9 + V / 28` |
| 4 | 2 | **28 GB/s** | `13 + V / 28` |
| 8 | 3 | **21 GB/s** | `21 + V / 21` |

> N=2 和 N=4 时 BW 同为 28 GB/s，因为 η 和 N 同步增长（2/1 = 4/2），流量放大被流水线深度抵消。

---

## 4. 验收判定

```
T_actual ≤ T_budget(V, N) × 1.10  →  ACCEPT
T_actual  > T_budget(V, N) × 1.10  →  REJECT（性能退化）
```

10% 容差覆盖测量噪声和 tiling 块对齐微扰。

### Server 8p（正式验收）

```
T_budget(V) = 21 + V / 21      (V: MB, T: us)

判定: T_actual ≤ T_budget × 1.10 → ACCEPT
```

---

## 5. 实测验证（全环境 PASS）

| 环境 | N | η | BW | max(T_act/T_budget) | 结果 |
|------|---|---|----|-------------------|------|
| EVB 2p | 2 | 1 | 28 GB/s | 0.94 | ✓ |
| 标卡 4p | 4 | 2 | 28 GB/s | 1.08 | ✓ |
| MXFP 4p | 4 | 2 | 28 GB/s | 1.09 | ✓ |
| POD 8p | 8 | 3 | 21 GB/s | 0.88 | ✓ |
| **Server 8p** | **8** | **3** | **21 GB/s** | **1.05** | **✓** |

**5 环境 × 13 shape = 65 数据点，全部 ACCEPT。**

### Server 8p 明细

| V (MB) | T_act (us) | T_budget (us) | ratio | 判定 |
|--------|-----------|--------------|-------|------|
| 0.031 | 15.5 | 22.5 | 0.69 | ACCEPT |
| 0.063 | 16.0 | 24.0 | 0.67 | ACCEPT |
| 0.125 | 17.8 | 27.0 | 0.66 | ACCEPT |
| 0.219 | 23.2 | 31.4 | 0.74 | ACCEPT |
| 0.406 | 34.2 | 40.4 | 0.85 | ACCEPT |
| 0.711 | 41.2 | 54.9 | 0.75 | ACCEPT |
| 0.766 | 47.1 | 57.5 | 0.82 | ACCEPT |
| 1.500 | 90.4 | 92.4 | 0.98 | ACCEPT |
| 3.000 | 171.7 | 163.9 | 1.05 | ACCEPT |
| 4.500 | 247.1 | 235.4 | 1.05 | ACCEPT |
| 6.000 | 306.9 | 306.8 | 1.00 | ACCEPT |
| 9.000 | 448.7 | 449.7 | 1.00 | ACCEPT |
| 12.00 | 582.8 | 592.6 | 0.98 | ACCEPT |

---

## 6. 测试条件

| 项目 | 要求 |
|------|------|
| **数据类型** | int8 / hif8 / fp8_e4m3fn / fp8_e5m2（1B 类型均复用） |
| **Scale** | fp32 或 fp8_e8m0（MXFP），实测等同 |
| **V 覆盖** | 0.01 ~ 12 MB，至少 6 个数量级 |
| **采集方式** | `torch_npu.profiler.profile`，10 轮迭代 |
| **环境变量** | `HCCL_BUFFSIZE=2048` |
| **前置条件** | 被测 NPU 空闲 |

### 使用方式

```bash
source ~/env.sh
source ~/SUB/vendors/custom_transformer/bin/set_env.bash
export HCCL_BUFFSIZE=2048
cd ~/mc2_test_0416

python3 perf_bandwidth.py --op qar --world_size 4          # int8
python3 perf_bandwidth.py --op qar --world_size 8 --dtype mxfp  # MXFP
```

对照 `T_budget = 21 + V/21`，检查 `T_actual / T_budget ≤ 1.10`。
