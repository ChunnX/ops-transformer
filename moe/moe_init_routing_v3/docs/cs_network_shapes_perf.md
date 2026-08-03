# A5 CountingSort 网络典型 Shape 性能验证报告

> **测试日期**：2026-08-10
> **数据来源**：需求文档 §1.2.4 网络 shape（RL 35B + 推理 92B/505B 的 decode/prefill）
> **对比方式**：custom 包（CS 模板）vs 卸载后 builtin（源代码），双侧 DYN_PERF
> **测试设备**：device 4（Ascend950DT），`--pc=1` 串行，`TTK_DISABLE_BUILTIN=0`
> **CSV**：`transpose/moe_v3_test_network_shapes.csv`（10 条）
> **固定参数**：`quant_mode=-1, expert_tokens_num_type=1(COUNT), expert_tokens_num_flag=True, drop_pad_mode=0, row_idx_type=0`，dtype=fp16
> **EP 设置**：`expert_num=E`(total), `active_expert_range=[0,LE]`(local)，expertIdx 值域 `[0,LE)` 模拟单卡本地专家路由

---

## 1. 用例设计（10 条）

| ti | 场景 | T | H | K | E | LE | AER | x内存 | expX内存 |
|----|------|---|---|---|---|----|-----|-------|---------|
| 0 | RL decode | 256 | 2560 | 10 | 192 | 24 | [0,24] | 1.3MB | 12.5MB |
| 1 | RL decode | 128 | 2560 | 10 | 192 | 24 | [0,24] | 0.7MB | 6.3MB |
| 2 | RL decode | 64 | 2560 | 10 | 192 | 24 | [0,24] | 0.3MB | 3.1MB |
| 3 | RL prefill | 131072 | 2560 | 10 | 192 | 24 | [0,24] | 0.67GB | 6.71GB |
| 4 | RL prefill | 65536 | 2560 | 10 | 192 | 24 | [0,24] | 0.34GB | 3.36GB |
| 5 | RL prefill | 16384 | 2560 | 10 | 192 | 24 | [0,24] | 0.08GB | 0.84GB |
| 6 | 92B decode | 1792 | 2560 | 8 | 256 | 32 | [0,32] | 9.2MB | 73.4MB |
| 7 | 505B decode | 2560 | 5120 | 8 | 384 | 6 | [0,6] | 26.2MB | 209.7MB |
| 8 | 92B prefill | 131088 | 2560 | 8 | 256 | 32 | [0,32] | 0.67GB | 5.37GB |
| 9 | 505B prefill | 131088 | 5120 | 8 | 384 | 24 | [0,24] | 1.34GB | 10.74GB |

---

## 2. 完整对比表

| ti | 场景 | CS TilingKey | CS (µs) | 源代码 TilingKey | 源代码 (µs) | Δ% | 判定 |
|----|------|-------------|---------|------------------|-------------|-----|------|
| 0 | RL decode T=256 | 200000 | 97.047 | 200000 | 97.438 | -0.4% | ⚪ 同路径 |
| 1 | RL decode T=128 | 200000 | 49.662 | 200000 | 50.276 | -1.2% | ⚪ 同路径 |
| 2 | RL decode T=64 | 200000 | 28.432 | 200000 | 28.425 | +0.0% | ⚪ 同路径 |
| 3 | RL prefill T=131072 | 11000000 | 8009.498 | 11000000 | 同路径 | — | ⚪ CS不适用 |
| 4 | RL prefill T=65536 | 11000000 | 3759.554 | 11000000 | 同路径 | — | ⚪ CS不适用 |
| 5 | RL prefill T=16384 | 500000 | 838.104 | 11000000 | 870.414 | **-3.7%** | ✅ 优化 |
| 6 | 92B decode T=1792 | 400000 | 98.792 | 11000000 | 73.782 | **+33.9%** | ❌ 劣化 |
| 7 | 505B decode T=2560 | 500000 | 126.434 | 11000000 | 136.095 | **-7.1%** | ✅ 优化 |
| 8 | 92B prefill T=131088 | 11000000 | 6145.654 | 11000000 | 同路径 | — | ⚪ CS不适用 |
| 9 | 505B prefill T=131088 | 11000000 | 8562.742 | 11000000 | 同路径 | — | ⚪ CS不适用 |

> ti=3/4/8/9 两侧均走 TK=11000000（同一 builtin 旧路径），CS 因 T 过大 UB 无法容纳而回落，builtin perf 与 CS 侧相同代码路径，未单独复测。

---

## 3. CS 适用性分布

| TilingKey | 用例 | 含义 |
|-----------|------|------|
| **200000** | ti=0,1,2（RL decode 小 T + 窄 AER） | builtin 特殊优化路径（源代码专属），CS 不介入。窄 AER `[0,24]` < expertNum=192 触发 builtin 的 N 较小特殊分支 |
| **11000000** | ti=3,4,8,9（大 prefill T≥65536） | builtin 旧分阶段路径。T 过大 CS UB 无法全载，两侧同路径 |
| **400000** (CS FullLoad) | ti=6（92B decode T=1792） | CS FullLoad 模板，expertIdx 可全载 |
| **500000** (CS CutOrigin) | ti=5,7（RL prefill T=16384 / 505B decode T=2560） | CS CutOrigin 模板，expertIdx 部分载入 |

**CS 生效仅 3/10 用例**（ti=5,6,7）。其余 7 条 CS 不适用（小 T 走 200000，大 T 回落 11000000）。

---

## 4. 关键发现

### 4.1 92B decode（ti=6）CS 严重劣化 +33.9%

这是需求文档 §2.9 的**首要验收 shape**（T=1792, H=2560, K=8, E=256, LE=32）。CS FullLoad（400000）perf=98.792µs vs builtin 11000000 perf=73.782µs，**劣化 +33.9%**。

根因（与 `cs_vs_builtin_perf.md` FullLoad 小列劣化一致）：FullLoad 模板的 Phase A/B/C 三相流水线 + SyncAll/workspace 读写固定开销，在 decode 中等 T 场景下无法摊薄，反而拖累。

### 4.2 505B decode（ti=7）CS 优化 -7.1%

505B decode（T=2560, H=5120, K=8, E=384, LE=6）CS CutOrigin（500000）perf=126.434µs vs builtin 136.095µs，**优化 -7.1%**。

> **v1.1 归因修正（2026-08-11）**：v1.0 原归因为「LE=6 极窄 active range，向量过滤 + 计数排序有正收益」。经 `cs_le_boundary_test.md` 补测推翻：相同 LE=6/AER=1.6% 在 active_num=9000 时劣化 +10%。**真正驱动因子是 active_num=20480（T×K=2560×8）**，大 active_num 摊薄了 CS 固定开销，与 LE/AER 宽窄无关。

### 4.3 RL prefill T=16384（ti=5）CS 优化 -3.7%

CS CutOrigin（500000）vs builtin（11000000），**-3.7% 轻微优化**。

> **v1.1 归因修正（2026-08-11）**：优化来自 **active_num=163840（T×K=16384×10）** 极大，CS 的 O(active_num) 算法优势显著，与 LE=24/AER=12.5% 无关。T=16384 是 CS 适用的上限边界（T=65536 即回落 11000000）。

### 4.4 RL decode（ti=0,1,2）走 builtin 200000 特殊路径

窄 AER `[0,24]` < expertNum=192 且小 T（64~256）→ builtin 有专属优化路径 TK=200000，CS 完全不介入。两侧同路径，perf 一致。

### 4.5 大 prefill（ti=3,4,8,9）CS 不适用

T≥65536 时 expertIdx 全载 UB 超限 → CS 回落 11000000（与 builtin 同路径）。CS 当前仅支持 expertIdx 全载 / chunk 载入，未覆盖超大 T 的分阶段路径。

---

## 5. 结论与建议

### 5.1 CS 在网络典型 shape 上的覆盖不足

10 条网络 shape 中 CS 仅生效 3 条（ti=5,6,7），其中：
- ✅ 优化 2 条：ti=5（-3.7%）、ti=7（-7.1%）
- ❌ 劣化 1 条：ti=6（+33.9%，**92B decode 首要验收 shape**）

### 5.2 关键问题：92B decode 验收 shape 劣化

§2.9 要求 92B decode shape「算子耗时优于 A3」。当前 CS FullLoad 在此 shape 劣化 +33.9%，是性能验收的**阻断点**。建议：
- 调整 `IsCountingSortApplicable` 启用门槛，对 92B decode 类 shape（中等 T + FullLoad）禁用 CS 或优化 FullLoad 常数开销
- 或对该 shape 走 builtin 11000000 路径（当前 builtin perf=73.782µs 已较优）

### 5.3 CS 未覆盖的场景

- **小 T + 窄 AER**（RL decode）：builtin 200000 专属路径，CS 无对应模板
- **超大 T prefill**（T≥65536）：CS UB 无法容纳，回落 11000000。若需优化 prefill，需扩展 CS 支持超大 T 的分阶段路径

---

**文档版本**：v1.1（2026-08-11 修正 ti=5/7 归因：active_num 替代 LE/AER）
**生成日期**：2026-08-10（v1.1：2026-08-11）
**关联文档**：
- 需求文档 §1.2.4（网络 shape）、§2.9（验收标准）
- `cs_vs_builtin_perf.md`（19 条混合场景对比）
- `cs_droppad_optimization_validation.md`（dropPad 稳定优化验证）
- `cs_le_boundary_test.md`（**LE 边界探索，确认 active_num 为真正驱动因子**）
