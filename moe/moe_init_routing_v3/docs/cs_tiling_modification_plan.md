# CS 模板 Tiling 修改方案（保证走入新模板的样例不劣化）

> **生成日期**：2026-08-12
> **状态**：待实施（pending）
> **依据**：`cs_optimization_conditions_summary.md` v1.0（41 条数据点）
> **目标**：通过 tiling 层拦截，保证走入新模板（CutOrigin 500000 / FullLoad 400100/400000）的样例不劣化
> **范围**：仅 tiling 层，不动 kernel

---

## 0. 问题陈述

当前 tiling 选择逻辑（line 559-576）存在两个问题：

1. **dropless 场景**：CS 启用门槛过宽，导致 K≤2 / H≤4096 / T<2048 / SCATTER / static_quant 等已知劣化场景仍进入 CS 路径
2. **dropless FullLoad（400000）**：先尝试 FullLoad → UB 够则启用，但 dropless FullLoad 在所有已测场景劣化（+10% ~ +33.9%），属于"必须跳过"的路径

数据支撑（41 条测试，详见 `cs_optimization_conditions_summary.md`）：
- dropPad 13/13 全优化（-5% ~ -33%）→ 无需改动
- dropless CutOrigin：仅 K≥3 + H≥5120 + T≥2048 + unquant + GATHER 才优化
- dropless FullLoad：全场景劣化
- SCATTER：必劣化（+22% ~ +43%）

---

## 1. 修改范围

**只动一个文件**：`op_host/moe_init_routing_v3_tiling_arch35.cpp`

**三处改动**（互相独立，可单独 revert）：
1. 新增 3 个边界常量（dropless 三因子阈值）
2. `IsCountingSortApplicable()` 加 4 个拦截条件
3. csKind 选择逻辑（line 559-576）按 dropPad/dropless 分流

---

## 2. 改动 1：新增边界常量

**位置**：文件顶部现有 CS 常量区（line 142 附近，紧邻 `CS_DB_BATCH_CAP_BYTES`）

```cpp
// ============== CS dropless 优化边界（基于 41 条数据点，详见 docs/cs_optimization_conditions_summary.md）==============
// dropless 场景下 CS CutOrigin 启用的三因子阈值
const static int64_t CS_DROPLESS_MIN_K = 3LL;        // K(topk) ≥ 3，每核 Phase C 双缓冲 pipeline 行数需充分
const static int64_t CS_DROPLESS_MIN_COLS = 5120LL;  // H(cols) ≥ 5120，单行搬运延迟需覆盖 pipeline 切换开销
const static int64_t CS_DROPLESS_MIN_N = 2048LL;     // T(n_) ≥ 2048，CutOrigin 启用下界（T<2048 会切 FullLoad +10% 劣化）
```

---

## 3. 改动 2：`IsCountingSortApplicable()` 加拦截

**位置**：line 2284-2324

**当前已有逻辑**：n 范围、expertNum≤1024、非动态量化、actualExpertNum≤256、dropless AER 约束、dropPad n≥512。

**新增 4 个拦截**（标注 `【新增】`）：

```cpp
bool MoeInitRoutingV3Arch35TilingClass::IsCountingSortApplicable()
{
    // n 范围（保持不变）
    if (n_ < CS_MIN_N || n_ > CS_MAX_N) {
        return false;
    }
    if (expertNum_ > 1024LL) {
        return false;
    }
    if (IsAnyDynamicQuantCase()) {
        return false;
    }

    // 【新增 1】SCATTER（row_idx_type=1）必劣化 +22%~+43%（4/4 复现，算法级瓶颈）
    // 根因：SCATTER 行索引语义与 CS Phase C 批量写出冲突，与 K/H/T/dropPad 无关
    if (rowIdxType_ != ROW_IDX_GATHER) {
        return false;
    }

    int64_t actualExpertNum = expertEnd_ - expertStart_;
    if (actualExpertNum <= 0 || actualExpertNum > CS_MAX_ACTUAL_EXPERT_NUM) {
        return false;
    }

    if (dropPadMode_ == DROP_PAD_MODE_DROPLESS) {
        if (actualExpertNum != expertNum_ && actualExpertNum * NUM_TWO > expertNum_) {
            return false;
        }

        // 【新增 2】dropless CutOrigin 三因子门槛（详见 docs/cs_cutorigin_optimization_conditions_v2.md v2.3）
        // 数据支撑：28 条 dropless CutOrigin 边界测试，CV<4%
        //   - K ≤ 2：6/6 劣化 +1.9%~+10.0%（含 H=7168/T=9000 也救不了）
        //   - H ≤ 4096：H=4096 K=4 +2.8% / H=2560 K=4 +13.8%
        //   - T < 2048：T∈[256,512] 走 builtin(200000) Δ=0%；T∈[1024,2047] 切 FullLoad(400000) +10%
        if (k_ < CS_DROPLESS_MIN_K) {
            return false;
        }
        if (cols_ < CS_DROPLESS_MIN_COLS) {
            return false;
        }
        if (n_ < CS_DROPLESS_MIN_N) {
            return false;
        }

        // 【新增 3】static_quant 在 dropless 边界持平（-0.4%），保守禁用
        // 根因：量化开销压缩 CS 优势，无正收益
        if (quantMode_ == QUANT_MODE_STATIC) {
            return false;
        }
    } else if (dropPadMode_ == DROP_PAD_MODE_DROPPAD) {
        if (n_ < 512LL) {
            return false;
        }
        // dropPad 全场景优化（13/13 复现 -5%~-33%），无需额外门槛
    }

    if (actualExpertNum > 256LL) {
        return false;
    }
    return true;
}
```

---

## 4. 改动 3：csKind 选择逻辑按 dropPad/dropless 分流

**位置**：line 559-576

**当前问题**：dropless 时先尝试 FullLoad（400000），但 **dropless FullLoad 全场景劣化**（小列 +15~30%，中等 T decode +33.9%，T∈[1024,2047] +10%）。

**修改方案**：dropless 跳过 FullLoad，直接尝试 CutOrigin。

```cpp
countingSortMode_ = 0;
tilingDataPtr_->countingSortParamsOp.countingSortMode = 0;
if (!isFullload_ && IsCountingSortApplicable()) {
    if (dropPadMode_ == DROP_PAD_MODE_DROPPAD) {
        // ============ dropPad 路径：保留现有 FullLoad 优先 → CutOrigin 回退（13/13 全优化 -5%~-33%）============
        int64_t perCoreTokensEst = Ops::Base::CeilDiv(n_, aivCoreNum_);
        int64_t estUb = EstimateArch35CountingSortFullLoadUB(perCoreTokensEst);
        if (estUb > 0 && estUb <= availUbSize_ - LENGTH_1024) {
            countingSortMode_ = 1;
            ComputeArch35CountingSortFullLoadTiling();
        } else {
            countingSortMode_ = 2;
            ComputeArch35CountingSortCutOriginTiling();
            if (tilingDataPtr_->countingSortParamsOp.countingSortMode == 0) {
                countingSortMode_ = 0;
            }
        }
    } else {
        // ============ dropless 路径：跳过 FullLoad，直接 CutOrigin ============
        // 原因：dropless FullLoad（400000）在所有已测场景劣化
        //   - 小列(cols=128) +15%~+30%：固定开销无法摊薄
        //   - 中等 T decode (92B T=1792) +33.9%：§2.9 验收阻断点
        //   - T∈[1024,2047] +10%：小 T 陷阱
        // IsCountingSortApplicable 已保证 K≥3 + H≥5120 + T≥2048 + 非静态量化
        countingSortMode_ = 2;
        ComputeArch35CountingSortCutOriginTiling();
        // CutOrigin tiling 内部若判定不适用（colsLoops>1 且非 UN_QUANT）仍会回退到 mode=0
        if (tilingDataPtr_->countingSortParamsOp.countingSortMode == 0) {
            countingSortMode_ = 0;
        }
    }
}
tilingDataPtr_->countingSortParamsOp.countingSortMode = countingSortMode_;
```

---

## 5. 改后效果对照（按场景）

| 场景 | 改前 TilingKey | 改前 Δ% | 改后 TilingKey | 改后 Δ% | 处理 |
|------|----------------|---------|----------------|---------|------|
| **dropPad + GATHER**（任意 K/H/T/dtype/cap）| 400100/500100 | -5%~-33% | 400100/500100 | -5%~-33% | **不变**（保持优化）|
| **dropPad + SCATTER** | 400110/500110 | 未测（理论劣化）| **11000000** | 0% | 改动 2 拦截，回退 builtin |
| **dropless + GATHER + K≥3 + H≥5120 + T≥2048 + unquant** | 500000 | -3%~-15% | 500000 | -3%~-15% | **不变**（保持优化）|
| dropless + GATHER + **K≤2** | 500000 | +2%~+10% | **11000000** | 0% | 改动 2 拦截 |
| dropless + GATHER + **H≤4096** | 500000 | +1%~+14% | **11000000** | 0% | 改动 2 拦截 |
| dropless + GATHER + **T∈[1024,2047]** | **400000** | **+10%** | **11000000** | 0% | 改动 2 拦截（改动 3 双保险）|
| dropless + GATHER + **T∈[256,512]** | 200000 | 0% | 200000 | 0% | 不变（本就 builtin）|
| dropless + GATHER + **static_quant** | 510000 | -0.4% ⚪ | **11010000** | 0% | 改动 2 拦截（保守禁用）|
| **dropless + SCATTER** | 500010 | +22%~+43% | **11000000** | 0% | 改动 2 拦截 |
| **dropless + FullLoad UB 命中** | **400000** | **+10%~+33.9%** | **500000** 或 **11000000** | -3%~-15% 或 0% | 改动 3 跳过 FullLoad |
| **92B decode**（T=1792, K=8, H=5120）| 400000 | **+33.9%** ❌ 阻断点 | **11000000** | 0% | 改动 2 拦截（T<2048），**消除阻断点** |

---

## 6. 验证清单

改后需 A/B 验证的用例集（建议跑 `moe_v3_test.csv` 全 40 条 + 边界测试集 28 条）：

### 6.1 保持优化（不应回退）

- **dropPad 13 条** → 期望 TilingKey 400100/500100，Δ% 仍 -5%~-33%
- **dropless CutOrigin 优化区 12 条**（K≥3+H≥5120+T≥2048+unquant+GATHER）→ 期望 TilingKey 500000，Δ% -3%~-15%

### 6.2 正确回退（应切到 builtin）

- dropless K=1（LE 边界 5 条）→ 期望 11000000
- dropless K=2（稳定性 ti=10）→ 期望 11000000
- dropless K=4 + H=4096/2560（K 边界 ti=1/2）→ 期望 11000000
- dropless T∈[256,2047]（T 下界 4 条）→ 期望 200000 / 11000000
- dropless static_quant（K 边界 ti=5）→ 期望 11010000
- 92B decode（T=1792）→ 期望 11000000，**消除 +33.9% 阻断点**

### 6.3 精度回归

全部 PRECISION PASS 不退化（tiling 改动只影响路径选择，不影响 kernel 计算）。

---

## 7. 风险与回退

| 风险 | 缓解 |
|------|------|
| 拦截过严，丢失边界优化场景（如 K=3 + H=4096 未测）| 边界常量集中在文件顶部，发现新优化场景时下修阈值即可（无需大改）|
| static_quant 禁用可能误伤后续优化 | 数据点仅 1 条（K=8/H=5120/-0.4%），若补测发现显著优化可放开 |
| dropless FullLoad 跳过，丢失潜在 UB 充足时的 FullLoad 优化 | 数据支撑：dropless FullLoad 在所有已测场景劣化（含 UB 充足场景），无正收益信号 |
| 回退到 builtin 后某些场景变慢 | builtin 是原代码基线，Δ%=0%（相对原代码无变化）|

**整体回退**：三处改动互相独立，可单独 revert。
- 改动 1 仅加常量，不影响逻辑
- 改动 2/3 加 `if` 分支，旧路径代码逐字保留在 `else` 或被新条件短路

---

## 8. 实施步骤（落地时遵循）

1. **备份当前文件**：`cp moe_init_routing_v3_tiling_arch35.cpp moe_init_routing_v3_tiling_arch35.cpp.bak`
2. **应用 3 处改动**（按本文档顺序）
3. **编包安装**：
   ```bash
   cd /home1/l00939308/ops-transformer0716
   source /home1/l00939308/myCann/v100_0803/Ascend-v100-0803/cann/set_env.sh
   bash build.sh --pkg --soc=ascend950 --vendor_name=custom --ops=moe_init_routing_v3
   bash /home1/l00939308/myCann/v100_0803/Ascend-v100-0803/cann/opp/vendors/custom_transformer/scripts/uninstall.sh
   bash /home1/l00939308/ops-transformer0716/build/cann-ops-transformer-custom_linux-aarch64.run
   ```
4. **按 §6 验证清单跑 A/B**（TTK `--ti=N` 串行，device 4，各 3 轮）
5. **判定**：
   - 保持优化组：TilingKey 不变 + Δ% 仍为负
   - 正确回退组：TilingKey 切到 11000000 / 11010000 / 200000
   - 全部 PRECISION PASS
6. **结论写入** `perf_baseline_vs_new_template.md` 新增章节

---

**文档版本**：v1.0（待实施）
**生成日期**：2026-08-12
**关联文档**：
- `cs_optimization_conditions_summary.md`（41 条数据点支撑）
- `cs_cutorigin_optimization_conditions_v2.md` v2.3（dropless CutOrigin 边界测试原始数据）
- `cs_droppad_optimization_validation.md`（dropPad 13 条同类验证原始数据）
- `perf_baseline_vs_new_template.md`（Phase C 双缓冲落地后的性能基线，验收时追加 §7）
