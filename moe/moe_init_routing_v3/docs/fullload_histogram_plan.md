# CountingSort FullLoad 模板 Histograms 改造方案

> **创建日期**：2026-08-08
> **目标**：在 FullLoad 模板（模板1）的 Phase A 计数段引入 `MOE_CS_USE_HISTOGRAM`，测出 cs_vs_builtin_perf.md 中 FullLoad 11 条用例（md ti=0-10）的 Histograms 性能数据，补全"宏关 baseline / Histograms / Hist Δ%"三列。
> **关联文档**：`cs_vs_builtin_perf.md`（性能对比主表）、`histogram_counting_optim_plan.md`（CutOrigin Histograms 方案，已落地）、`project_moe_v3_histogram_ab.md`（CutOrigin A/B v1/v2 结论）

---

## 1. 核心洞察

FullLoad 的 `FilterAndCountVector`（`full_load_base.h` L309-372）与 CutOrigin 的向量过滤段**完全同构**（Cast→CompareScalar→And→GatherMask），唯一差异是末尾用**标量 count 循环**（L360-371）而非 Histograms。因此 CutOrigin 的 Histograms 后段可**几乎逐字平移**到 FullLoad。

FullLoad base.h L18-19 已有 PERF TODO 明确预留此改造点：
> `FilterAndCountVector 的计数段（CompareScalar + And + 标量累加）后续可替换为 SIMD VF Histograms API`

---

## 2. 改动点（3 个文件）

### 2.1 `op_kernel/arch35/moe_v3_counting_sort_full_load_base.h`

**(a) 宏开关**：复用 CutOrigin 已有的 `MOE_CS_USE_HISTOGRAM`（编译期全局 define，FullLoad/CutOrigin 同包构建，开一次同时生效）。因 FullLoad 用例（cols=128 小列 → TK=400xxx）与 CutOrigin 用例（cols=7168 大列 → TK=500xxx）按 TilingKey 分流到不同模板，互不污染，共享宏可接受。若需独立 A/B，可改用 `MOE_CS_FULLLOAD_USE_HISTOGRAM`（见 §5 可选项）。

**(b) UB offset 成员**（参照 cut_origin_base.h L153-155，在 full_load_base.h L119 `gatheredIdxLocalOffset_` 之后追加）：

```cpp
#ifdef MOE_CS_USE_HISTOGRAM
    int64_t expertU8LocalOffset_;   // uint8 源（filteredInBatch 个 expertOffset，值域 [0,256)）
    int64_t histBinOffset_;         // uint16 bins（256 个 = BIN0 128 + BIN1 128）
#endif
```

**(c) UB 布局**（`ComputeCommonUbLayout()` L263 `gatheredIdxLocalOffset_` offset 累加之后、L267 `commonBufSize_ = offset` 之前追加）：

```cpp
#ifdef MOE_CS_USE_HISTOGRAM
    expertU8LocalOffset_ = offset;
    offset += AlignBytes(Ceil(entriesAligned_, static_cast<int64_t>(256)) * 256,
                         static_cast<int64_t>(sizeof(uint8_t)));
    histBinOffset_ = offset;
    offset += AlignBytes(256, static_cast<int64_t>(sizeof(uint16_t)));
#endif
```

**(d) Histograms VF 函数放置（两选一，见 §5）**：

CutOrigin 的 `MoeV3CutOriginHistogramVF`（cut_origin_base.h L379-399，21 行 `__simd_vf__ inline`）定义在 cut_origin_base.h 内。FullLoad 是独立模板类，不 include 它。两个选择：

- **方案A（推荐，去重）**：把 `MoeV3CutOriginHistogramVF` 从 cut_origin_base.h **移到 `moe_v3_common.h`** 并重命名为 `MoeV3CountingSortHistogramVF`，两个模板共用；cut_origin_base.h 的调用点（L468）改新名。
- **方案B（最低风险）**：在 full_load_base.h **复制一份**同名函数（21 行），不动已验证的 CutOrigin。

**(e) `FilterAndCountHistogram()`**（新增，逐字平移 cut_origin_base.h L404-488）：

- 向量过滤段（Cast int32→fp32 → CompareScalar GE/LT → And → GatherMask 压缩 expert+flatIdx）与 FullLoad 现有 `FilterAndCountVector` L312-354 完全相同，直接复用布局。
- 新增后段（cut_origin L453-487）：
  1. `Adds(gatheredExpert, gatheredExpert, -expertStart_, filteredInBatch)` 把 expertId 平移到 [0, actualExpertNum)
  2. 两步窄化 `Cast int32→int16→uint8`（值域已保证 [0,256)）
  3. `MoeV3CountingSortHistogramVF(uint8 源, uint16 bins, filteredInBatch)` 计数
  4. uint16 bins → int32 expertCountLocal 标量回写（actualExpertNum_ ≤ 256）
  5. pairs 在压缩结果上标量填充（不再做 count 累加）

**(f) 派发**（`FilterAndCount()` L299-307）：

```cpp
template <typename T>
__aicore__ inline void MoeV3CountingSortFullLoadBase<T>::FilterAndCount()
{
    LocalTensor<int32_t> expertCountLocal = buf_.Get<int32_t>()[expertCountLocalOffset_ / sizeof(int32_t)];
    Duplicate(expertCountLocal, static_cast<int32_t>(0), static_cast<int32_t>(expertCountStride_));
    PipeBarrier<PIPE_V>();
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

#ifdef MOE_CS_USE_HISTOGRAM
    FilterAndCountHistogram();
#else
    FilterAndCountVector();
#endif
}
```

> **注意**：FullLoad `FilterAndCount()` 入口已含 `Duplicate(expertCountLocal,0)` + PipeBarrier + SetWaitFlag，Histograms 函数体**必须删掉开头那段重复的 Duplicate/PipeBarrier**（cut_origin L410-411），避免重复清零。CutOrigin 入口 FilterAndCount 不做清零、由 Histograms 函数自带清零；FullLoad 相反——入口做清零、Histograms 函数不带清零。两边职责划分不同，平移时需对齐。

### 2.2 `op_host/moe_init_routing_v3_tiling_arch35.cpp`

`EstimateArch35CountingSortFullLoadUB()`（L2306-2356）在 gatheredExpert/flatIdx/gatheredIdx 三项（L2344）之后、量化临时区（L2347）之前追加（镜像 cut_origin tiling L2430-2437）：

```cpp
#ifdef MOE_CS_USE_HISTOGRAM
    // Histograms 专属：uint8 源（coreEntries 对齐 256）+ uint16 bins（256 个）。
    // 镜像 kernel ComputeCommonUbLayout 的 #ifdef MOE_CS_USE_HISTOGRAM 分支。
    total += Ops::Base::CeilAlign(
        Ops::Base::CeilDiv(coreEntries, static_cast<int64_t>(256)) * 256 * static_cast<int64_t>(sizeof(uint8_t)),
        UB_BLOCK_SIZE);
    total += Ops::Base::CeilAlign(256 * static_cast<int64_t>(sizeof(uint16_t)), UB_BLOCK_SIZE);
#endif
```

FullLoad UB 预算宽裕（cols=128 小列，coreEntries = perCoreTokens × k 较小），追加这两项不会触发 mode-1→mode-2 降级。

### 2.3 无 tilingdata 字段改动

不新增 tilingdata 字段（与 CutOrigin 一致，纯编译期宏控制），无需改 `*_tiling.h` / `*_tiling_base.cpp`。

---

## 3. 数据一致性保证（双向镜像）

| kernel ComputeCommonUbLayout（full_load_base.h） | tiling EstimateArch35CountingSortFullLoadUB |
|---------------------------------------------------|---------------------------------------------|
| `expertU8LocalOffset_`: Ceil(entriesAligned,256)×256 × 1B | `CeilDiv(coreEntries,256)×256 × sizeof(uint8_t)` |
| `histBinOffset_`: 256 × 2B | `256 × sizeof(uint16_t)` |

> kernel 用 `entriesAligned_`（=Ceil(coreEntries,64)×64），tiling 用 `coreEntries` CeilDiv 256。因 256 是 64 的整数倍，Ceil(coreEntries,64) 再 CeilDiv 256 与直接 CeilDiv(coreEntries,256) 结果一致。两侧 UB 字节数严格相等。

---

## 4. 测试范围（cs_vs_builtin_perf.md FullLoad 11 条）

| md ti | csv 位置 | 用例 | CS TilingKey | 预期 Histograms 是否生效 |
|-------|---------|------|-------------|------------------------|
| 0 | cs_coverage.csv#ti=0 | FL unquant fp16 GATHER COUNT drop | 400000 | ✅ 生效 |
| 1 | cs_coverage.csv#ti=1 | FL unquant bf16 GATHER COUNT drop | 400000 | ✅ 生效 |
| 2 | cs_coverage.csv#ti=2 | FL unquant fp32 GATHER COUNT drop | 400000 | ✅ 生效 |
| 3 | cs_coverage.csv#ti=3 | FL unquant fp16 SCATTER count drop | 401000 | ✅ 生效 |
| 4 | cs_coverage.csv#ti=4 | FL unquant fp16 GATHER CUMSUM drop | 400000 | ✅ 生效 |
| 5 | cs_coverage.csv#ti=5 | FL unquant fp16 GATHER KEY_VALUE drop | 400000 | ✅ 生效 |
| 6 | cs_coverage.csv#ti=6 | FL static fp16 GATHER COUNT drop | 410000 | ✅ 生效 |
| 7 | cs_coverage.csv#ti=7 | FL static fp16 SCATTER count drop | 411000 | ✅ 生效 |
| 8 | cs_coverage.csv#ti=8 | FL static fp16 GATHER CUMSUM drop | 410000 | ✅ 生效 |
| 9 | cs_coverage.csv#ti=9 | FL unquant fp16 GATHER COUNT **pad** | 400100 | ✅ 生效（dropPad 唯一可能正收益） |
| 10 | cs_coverage.csv#ti=10 | FL dynamic fp16 GATHER COUNT drop | 11020000 | ❌ **CS 关闭回落**，Histograms 不生效 → N/A |

### 测试流程（同 CutOrigin v2）

1. 编包安装：
   ```bash
   source /home1/l00939308/myCann/v100_0803/Ascend-v100-0803/cann/set_env.sh
   bash /home1/l00939308/moeInitrouteV3/ttk精度调试.sh   # 含卸载+装包（TMPDIR=/home1/l00939308）
   ```
2. 宏关测 baseline（每条 3 次）：
   ```bash
   TTK_DISABLE_BUILTIN=0 python3 -m ttk kernel -i transpose/moe_v3_test_cs_coverage.csv \
     --ti=N --pc=1 --device-whitelist=4 > transpose/moe_fl_tiN_off.log 2>&1
   ```
   验证 TilingKey=400xxx（FullLoad 模板），记录 DYN_PERF。
3. 开宏 `MOE_CS_USE_HISTOGRAM` → 重编重装 → 同命令测 Histograms perf。
4. 每条算 Hist Δ% = (宏开 − 宏关) / 宏关 × 100%，填入 cs_vs_builtin_perf.md 三列。
5. 判定阈值 |Δ%| > 1%。

---

## 5. 决策（已确认 2026-08-08）

### 决策1：Histograms VF 函数放置 → **方案B（FullLoad 内复制）**

在 `full_load_base.h` 内复制 21 行，重命名为 `MoeV3FullLoadHistogramVF`，**不动 CutOrigin**，避免交叉影响已验证代码。

### 决策2：宏策略 → **新增 `MOE_CS_FULLLOAD_USE_HISTOGRAM`**

与 CutOrigin 的 `MOE_CS_USE_HISTOGRAM` 独立，两个模板按 TilingKey 分流，可单独开/关做 A/B。kernel 与 tiling 的 `#ifdef` 均使用 `MOE_CS_FULLLOAD_USE_HISTOGRAM`。

---

## 6. 实测结果（2026-08-08，device 6）

| ti | 用例 | TK | 宏关 baseline (µs) | Histograms (µs) | Δ% | 判定 |
|----|------|----|-------------------|-----------------|-----|------|
| 0 | FL unquant fp16 GATHER COUNT drop | 400000 | 65.417 | 67.353 | +3.0% | ❌劣化 |
| 1 | FL unquant bf16 GATHER COUNT drop | 400000 | 66.764 | 67.260 | +0.7% | ⚪持平 |
| 2 | FL unquant fp32 GATHER COUNT drop | 400000 | 65.963 | 67.696 | +2.6% | ❌劣化 |
| 3 | FL unquant fp16 SCATTER count drop | 401000 | 71.460 | 72.733 | +1.8% | ❌劣化 |
| 4 | FL unquant fp16 GATHER CUMSUM drop | 400000 | 66.005 | 67.120 | +1.7% | ❌劣化 |
| 5 | FL unquant fp16 GATHER KEY_VALUE drop | 400000 | 65.067 | 66.364 | +2.0% | ❌劣化 |
| 6 | FL static fp16 GATHER COUNT drop | 410000 | 69.828 | 72.410 | +3.7% | ❌劣化 |
| 7 | FL static fp16 SCATTER count drop | 411000 | 75.191 | 76.845 | +2.2% | ❌劣化 |
| 8 | FL static fp16 GATHER CUMSUM drop | 410000 | 70.147 | 71.004 | +1.2% | ❌劣化 |
| 9 | FL unquant fp16 GATHER COUNT **pad** | 400100 | 63.918 | 66.622 | +4.2% | ❌劣化 |

**汇总**：10/10 PRECISION PASS，平均 Δ% ≈ **+2.3%（一致劣化）**。✅优化 0 / ⚪持平 1（ti=1）/ ❌劣化 9（+1.2% ~ +4.2%）。

**结论：FullLoad Histograms 为纯负收益**，劣化幅度（+2.3%）比 CutOrigin（+0.3%）更明显。根因：
- FullLoad cols=128 小列、coreEntries 小、baseline 耗时本就短（~65-76µs）
- Histograms 固定开销（Adds + 两步 Cast + Histograms VF + uint16→int32 回写）的 ~1-3µs 增量在小基数上占比更高
- ti=9 dropPad 劣化最严重（+4.2%），说明 dropPad 优化由批量 staging 承担、非计数段，Histograms 无加成

**与 CutOrigin 对比**：两个模板 Histograms 均为负收益，方向一致。FullLoad 劣化更严重（+2.3% vs +0.3%），因 FullLoad baseline 更短、固定开销占比更高。两宏（`MOE_CS_USE_HISTOGRAM` / `MOE_CS_FULLLOAD_USE_HISTOGRAM`）均默认关闭，代码保留。

## 7. 风险与预判（实测前，归档参考）

- **预判负收益（与 CutOrigin v1/v2 一致）**：FullLoad 用例 AER=[0,256] 全通过，向量过滤压缩不出淘汰元素，Histograms 整链（Adds + 两步 Cast + Histograms + uint16→int32 回写）是**纯开销**。且 FullLoad cols=128 小列、coreEntries 小（perCoreTokens ≈ N/aivCoreNum × k），计数段在总耗时中占比本就低，劣化幅度预计很小（< +1%，落在 ⚪持平区间）。
- **唯一正收益可能**：ti=9（dropPad）—— dropPad 路径 filter 后 count 累加占比略高。但 FullLoad dropPad 的 -31.2% 优化主要由批量 staging + ping-pong 承担（非计数段），Histograms 未必额外加成。
- **精度**：Histograms 只改 count 计算（uint8 窄化值域 [0,256) 安全，actualExpertNum_ ≤ 256），不改 pairs/GatherAndWrite，预期 11/11 PRECISION PASS。
- **回退**：宏默认关闭，旧 `FilterAndCountVector` 标量路径逐字保留，UB 占用与行为不变。

---

## 8. 收敛判据（实测结论）

- FullLoad 10 条有效用例（ti=0-9）已测出 Hist Δ%，已补全 cs_vs_builtin_perf.md 三列。
- **9/10 劣化 + 1/10 持平，无优化**，与 CutOrigin 结论方向一致：Histograms 在 AER 全通过用例为非正收益，宏默认关闭，代码保留。
- 精度 10/10 PRECISION PASS，无退化。

---

**文档版本**：v1.2（A/B 测试完成，结论：纯负收益）
**状态**：✅ 实现完成 + A/B 测试完成（宏默认关闭，代码保留）
