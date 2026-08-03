# A5 CountingSort 新模板优化条件总览

> **生成日期**：2026-08-11（v1.0 综合版）
> **测试设备**：device 4 / 6（Ascend950DT, arch35）
> **安装包**：2026-08-10 17:19 编译（含 Phase C 双缓冲重建）
> **数据来源**：
> - `cs_droppad_optimization_validation.md`（dropPad 13 条同类验证，2026-08-08）
> - `cs_cutorigin_optimization_conditions_v2.md` v2.3（dropless CutOrigin 28 条边界测试，2026-08-11）
> **指标**：DYN_PERF（µs，越小越好），Δ% = (CS - Builtin) / Builtin × 100%
> **判定**：|Δ%| > 1% 优化/劣化，否则持平

---

## 0. 一图速览：新模板决策矩阵

```
                          drop_pad_mode=1 (dropPad)        drop_pad_mode=0 (dropless)
                         ┌─────────────────────────────┐  ┌──────────────────────────────────┐
   row_idx_type=0 GATHER │  ✅ 全场景稳定优化            │  │  依赖 K / H / T 组合（见 §2）       │
                         │  FullLoad: -24% ~ -33%       │  │  ✅ K≥3 且 H≥5120 且 T≥2048       │
                         │  CutOrigin: -5% ~ -15%       │  │     CutOrigin: -3% ~ -15%         │
                         │  13/13 复现（100%）           │  │  ❌ K≤2 / H≤2560 / T<2048         │
                         └─────────────────────────────┘  └──────────────────────────────────┘
                         ┌─────────────────────────────┐  ┌──────────────────────────────────┐
   row_idx_type=1 SCATTER│  未单独测试（算法级瓶颈）     │  │  ❌ 必劣化 +22% ~ +43%（4/4）      │
                         │  建议沿用 dropless 结论：禁用 │  │     算法级瓶颈，与 K/H/T 无关      │
                         └─────────────────────────────┘  └──────────────────────────────────┘
```

---

## 1. dropPad 场景（drop_pad_mode=1）

### 1.1 优化条件：全场景稳定优化

**无条件优化**——dtype / cols / N / expertCapacity / K 在测试区间内全部产生稳定正收益，**无需任何启用门槛**。

| 模板 | 优化区间 | 复现率 | 测试维度 |
|------|---------|--------|----------|
| **FullLoad dropPad**（TilingKey 400100）| **-24% ~ -33%** | 7/7 | fp16/bf16/fp32, cols∈{128,512}, N∈{4096,9000}, cap∈{32,64,128} |
| **CutOrigin dropPad**（TilingKey 500100）| **-5% ~ -15%** | 6/6 | fp16/bf16/fp32, cols∈{4096,7168}, N∈{4096,9000}, cap∈{32,64} |

### 1.2 关键特性

- **dtype 无关**：fp16/bf16/fp32 表现完全一致（`EstimateArch35CountingSortFullLoadUB` 使用统一 `inputXDtypeSize_`，bf16=fp16=2 字节）
- **expertCapacity 无关**：cap∈{32,64,128} 优化幅度差 < 5%
- **cols 无关**：cols∈{128,512,4096,7168} 全部优化
- **N 切换模板**：N 大走 FullLoad，N 小（如 4096）切到 CutOrigin，两者都优化
- **quant 约束**：dropPad 强制 `quant_mode=-1`（见 op 层 ValidateQuantMode），故仅验证 unquant

### 1.3 根因

dropPad 模式下：
- Phase A 批量 staging + ping-pong 载入
- Phase C dropPad seed 预计算（避免逐 pair 判定）
- 旧 builtin 11000100 路径是分阶段逐 pair 处理，固定开销高

→ 新模板的算法+pipeline 优势在所有 dropPad 子场景中都能覆盖 Phase C 流式载入的开销。

---

## 2. dropless CutOrigin 场景（drop_pad_mode=0, CutOrigin 模板）

### 2.1 三因子启用条件

**CutOrigin（500000）启用并优化的完整条件**：

```
┌──────────────────────────────────────────────────────────────┐
│  K ≥ 3  且  H ≥ 5120  且  T ≥ 2048  →  CS CutOrigin 优化     │
│  （fp16/bf16, COUNT/CUMSUM, unquant/static, LE∈[6,256]）   │
└──────────────────────────────────────────────────────────────┘
```

任一条件不满足 → 劣化或切换到非 CutOrigin 路径。

### 2.2 K 因子（主方向决定者）

| K | 数据点 | ✅优化 | ⚪持平 | ❌劣化 | Δ% 区间 |
|---|--------|-------|--------|--------|---------|
| 1 | 5 | 0 | 0 | **5** | +1.9%~+10.0% |
| 2 | 1 | 0 | 0 | **1** | +7.4% |
| **3** | 1 | **1** | 0 | 0 | -7.9% |
| **4** | 4 | **2** | 0 | **2** | -7.7%~+13.8%（H 敏感）|
| **5~16** | 12 | **12** | 0 | 0 | -3.1%~-13.3% |

- **K ≤ 2 → 必劣化**，与 H/T/LE 无关
- **K ≥ 3 → 进入优化区**（K=3 在 H=7168 已 -7.9%）
- **K 是方向决定者**

### 2.3 H 因子（幅度调节者）

| H | 数据点 | Δ% 区间 | 平均 | 备注 |
|---|--------|---------|------|------|
| **7168** | 6 | -13.3%~-7.7% | **-11.1%** | 最优 |
| **5120** | 6 | -14.5%~-3.1% | **-8.0%** | 中等（受 T 放大）|
| **4096** | 2 | +0.7%~+2.8% | +1.8% | ⚪ 持平/轻微劣化 |
| **2560** | 1 | +13.8% | +13.8% | ❌ 严重劣化 |

- **H ≥ 5120 → 优化**
- **H ≤ 4096 → 持平或劣化**
- **大列放大优化**：H 大 → 单行搬运延迟长 → Phase C 双缓冲重叠收益大

### 2.4 T 因子（放大者 + 路径切换者）

**T 的双重作用**：

**(a) 路径切换**（K=8, H=5120 固定，决定 CS 是否启用及启用哪个模板）：

| T 区间 | CS TilingKey | 行为 | Δ% |
|--------|--------------|------|------|
| [256, 512] | **200000** | ⚠ CS 不启用（双侧走 builtin 旧路径）| 0% |
| [1024, 2047] | **400000** | ◆ CS **FullLoad**（dropless 场景下劣化）| **+10%** |
| **≥ 2048** | **500000** | ✅ CS **CutOrigin**（优化）| **-3% ~ -15%** |

**(b) 优化放大**（K=8, H=5120, T≥2048 时）：

| T | CS_md | BI_md | Δ% |
|------|-------|-------|------|
| 2048 | 102.4 | 107.9 | -5.1%（启用下界）|
| 2560 | 126.8 | 130.9 | -3.1%（局部最弱点）|
| 5120 | 249.4 | 281.2 | -11.3% |
| 10240 | 497.4 | 582.0 | **-14.5%**（趋饱和）|

- **T 翻倍边际递减**：2560→5120 = +8.2pp；5120→10240 = +3.2pp
- **非严格单调**：T=2560 是局部最弱点（perCoreTokens 量化效应）
- **K 决定方向，T 决定幅度**

### 2.5 其他维度（v2.1/v2.2 补测）

| 维度 | 测试条件 | Δ% | 判定 | 结论 |
|------|---------|-----|------|------|
| expert_tokens_num_type=CUMSUM | K=8, H=5120, fp16, unquant | -6.0% | ✅ | 与 COUNT 一致优化 |
| dtype=bf16 | K=8, H=5120, COUNT, unquant | -4.9% | ✅ | 与 fp16 一致优化 |
| quant_mode=0 (static) | K=8, H=5120, fp16, COUNT | -0.4% | ⚪ | 边界持平（量化开销压缩 CS 优势）|
| LE | 6 ~ 256 | <10% 差异 | — | 不影响方向 |
| AER | 1.6% ~ 100% | — | — | 不影响方向 |

### 2.6 根因

CutOrigin Phase C 双缓冲 ping-pong 流水要求 active 行数足够多才能填满 DMA pipeline：
- **K 小** → 每核 active 行少 → pipeline 填不满 → 双缓冲 stall
- **K ≥ 3** → 充分行数 → MTE2-MTE3 重叠隐藏延迟
- **H 大** → 单行搬运延迟长 → 双缓冲重叠收益大
- **T 大** → perCoreTokens 大 → pipeline 重叠深度更深 → CS 相对 builtin 优势充分发挥

---

## 3. dropless FullLoad 场景（drop_pad_mode=0, FullLoad 模板）

### 3.1 全场景劣化或持平（不建议启用）

| 子场景 | Δ% | 备注 |
|--------|------|------|
| **小列（cols=128）+ dropless** | +15% ~ +30% | 固定开销无法摊薄 |
| **中等 T decode（92B decode T=1792）** | **+33.9%** | §2.9 验收阻断点 |
| **T ∈ [1024, 2047] + 大列** | +10% | 小 T 陷阱（v2.3 发现）|

**结论**：dropless FullLoad 在所有已知测试场景中无优化信号，建议禁用走 builtin 11000000。

---

## 4. SCATTER 场景（row_idx_type=1）

### 4.1 算法级瓶颈，必劣化

- 4/4 用例复现 **+22% ~ +43%** 劣化
- 与 dropPad / dropless、K、H、T 均无关
- 根因：SCATTER 行索引模式与 CS Phase C 批量写出语义冲突
- **建议**：row_idx_type=1 时直接禁用 CS

---

## 5. 启用决策表（生产应用指南）

| 场景组合 | 启用 CS？ | 模板 | 预期 Δ% |
|----------|----------|------|---------|
| dropPad + GATHER | ✅ 启用 | FullLoad 优先，UB 超则 CutOrigin | **-5% ~ -33%** |
| dropPad + SCATTER | ❌ 禁用 | — | 算法级瓶颈 |
| dropless + GATHER + **K≥3 + H≥5120 + T≥2048** | ✅ 启用 | CutOrigin | **-3% ~ -15%** |
| dropless + GATHER + K≤2 | ❌ 禁用 | — | +2% ~ +10% |
| dropless + GATHER + H≤4096 | ❌ 禁用 | — | +1% ~ +14% |
| dropless + GATHER + T<2048 | ❌ 禁用 | （CS 不启用或切 FullLoad 劣化）| 0% 或 +10% |
| dropless + SCATTER | ❌ 禁用 | — | +22% ~ +43% |
| 动态量化（quant_mode=1）| ❌ 不适用 | — | IsCountingSortApplicable 拦截 |
| 92B decode（T=1792）| ❌ 禁用 | — | FullLoad +33.9% 验收阻断点 |

---

## 6. TilingKey 编码对照

| TilingKey | 路径含义 |
|-----------|----------|
| **200000** | builtin 旧路径（CS 未启用，CS_MIN_N 通过但 csKind=0）|
| **400000** | CS FullLoad dropless unquant+GATHER+COUNT |
| **400100** | CS FullLoad **dropPad** unquant+GATHER+COUNT |
| **500000** | CS CutOrigin dropless unquant+GATHER+COUNT |
| **500100** | CS CutOrigin **dropPad** unquant+GATHER+COUNT |
| **510000** | CS CutOrigin dropless static_quant+GATHER+COUNT |
| **11000000** | 源代码（builtin）dropless unquant+GATHER+COUNT |
| **11000100** | 源代码（builtin）dropPad unquant+GATHER+COUNT |

---

## 7. 不适用区（IsCountingSortApplicable 拦截）

| 拦截条件 | 行为 |
|----------|------|
| T < 256（CS_MIN_N）| 走 builtin |
| T > 1048576（CS_MAX_N = 1024×1024）| 走 builtin |
| expertNum > 1024 | 走 builtin |
| **quant_mode=1（动态量化）** | 走 builtin（用户决策：动态量化场景禁用 CS，详见 `project_moe_v3_a3_vs_a5_rootcause.md`）|
| actualExpertNum > 256 | 走 builtin |
| dropless AER 约束不满足（`actualExpertNum × 2 > expertNum` 且 `actualExpertNum != expertNum`）| 走 builtin |
| T ≥ 65536 且 UB 无法全载 | 走 builtin 11000000 |

---

## 8. 数据可靠性声明

| 数据集 | 数据点 | CS CV | BI CV | 测试方式 |
|--------|--------|-------|-------|----------|
| dropPad 同类验证 | 13 | <2% | <2% | `--pc=1` 串行 |
| 稳定性测试 | 11 | <2% | <2% | `--ti=N` 串行 3 轮 |
| LE 边界 | 5 | <1% | <1% | `--ti=N` 串行 |
| K 边界补测 | 6 | <1% | <4% | `--ti=N` 串行 3~5 轮 |
| T 放大补测 | 2 | <0.3% | <0.3% | `--ti=N` 串行 3 轮 |
| T 下界补测 | 4 | <1% | <4% | `--ti=N` 串行 3 轮 |
| **总计** | **41** | — | — | 全部 CV < 4% |

---

## 9. 关键修正历史

| 版本 | 修正内容 | 推翻的旧结论 |
|------|----------|--------------|
| v1.0（2026-08-05）| dropPad 是 CS 唯一稳定优化信号 | — |
| v2.0（2026-08-10）| K 是主驱动因子；n_ = T（不是 active_num）| v1.x 的 active_num ≥ 20000 假设 |
| v2.1（2026-08-11）| K 阈值从 K≥4 下修为 K≥3；CUMSUM/bf16 一致优化 | — |
| v2.2（2026-08-11）| T 是放大因子（非"非因子"）| v2.1 的 T 非因子结论 |
| v2.3（2026-08-11）| T 三段式路径切换；CutOrigin 启用下界 T≥2048 | — |
| **v1.0 综合版（本文档）** | 整合 dropPad + dropless + SCATTER + FullLoad 全场景决策矩阵 | — |

---

**文档版本**：v1.0（综合版，2026-08-11）
**生成日期**：2026-08-11
**关联文档**：
- `cs_droppad_optimization_validation.md`（dropPad 13 条同类验证原始数据）
- `cs_cutorigin_optimization_conditions_v2.md` v2.3（dropless CutOrigin 28 条边界测试原始数据）
- `cs_optimization_patterns.md`（早期优化规律汇总）
- `cs_vs_builtin_perf.md`（19 条混合场景首次发现 dropPad 优化）
- `perf_baseline_vs_new_template.md`（Phase C 双缓冲落地后的性能基线）
