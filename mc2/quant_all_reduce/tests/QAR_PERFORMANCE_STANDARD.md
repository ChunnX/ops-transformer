# QAR 算子性能标准

---

## 1. 性能模型

QAR 算子使用 UB-MEM 进行数据搬运，通过 MTE 单口通信读写各 rank 的 Win 区完成跨卡数据交换。kernel 时间由固定开销 + 数据搬运时间组成：

```
T_kernel = T_overhead  +  T_data

T_overhead = 5 + 2N               (us)   固定开销 = 头开销 + 与卡数 N 相关的开销
T_data     = V × N / (56 × log₂N)  (us)   数据搬运时间
```

| 符号 | 含义 | 来源 |
|------|------|------|
| V | 有效载荷 `bs × h × 1B`（MB） | 测试 shape |
| N | 卡数（world_size） | 部署配置 |
| 56 GB/s | MTE 单口峰值带宽 | 硬件规格 |
| log₂N | 流水线等效深度 | 代码 double-buffer 结构，N 个 Win 区的读/算可重叠 log₂N 层 |

### 1.1 推导

MTE 跨卡 Win 区读为所有 AIV 核共享单口，峰值 56 GB/s。

算子每处理 1B 有效载荷，需从 N 个 Win 区各读 1B，流量放大 N 倍。代码通过 double/triple buffer 将 MTE 读与 AIV 反量化/规约计算重叠，等效流水线深度为 log₂N。

```
T_data = V × N / (56 × log₂N)
```

### 1.2 预算公式

```
T_budget(V, N) = (5 + 2N) + V × N / (56 × log₂N)
```

其中 `5 + 2N` 为固定开销（us），`V × N / (56 × log₂N)` 为数据搬运时间（us）。

---

## 2. 验收判定

```
T_actual ≤ T_budget(V, N) × 1.10  →  ACCEPT
T_actual  > T_budget(V, N) × 1.10  →  REJECT
```

10% 容差覆盖测量噪声和 tiling 块对齐微扰。

---

## 3. 测试条件

| 项目 | 要求 |
|------|------|
| 数据类型 | int8 / hif8 / fp8_e4m3fn / fp8_e5m2（sizeof=1B） |
| Scale 类型 | fp32（per-128-group）或 fp8_e8m0（MXFP，per-32-group），两者性能等同 |
| V 覆盖范围 | 0.01 ~ 12 MB，至少覆盖 6 个数量级 |
| 采集方式 | `torch_npu.profiler.profile`，10 轮迭代，warmup=0，active=1 |
| T_kernel 取值 | 从 `op_summary.csv` 取非 AI_CPU 任务 Task Duration 平均值 |
| Win 区 buffer | 需确保 Win 区大小满足大 shape 数据量要求 |
| 前置条件 | 被测 NPU 空闲 |
