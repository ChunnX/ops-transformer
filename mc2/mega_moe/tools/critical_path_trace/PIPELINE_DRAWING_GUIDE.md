# MegaMoe A8W4 流水示意图绘制规范（PIPELINE DRAWING GUIDE）

> 目的：把 A8W4（及 MegaMoe 相关）流水示意图的**绘制约定、时序数据、依赖规则**固化下来。下次需要画流水图时直接照此执行，无需重新说明。
> 适用范围：`tools/critical_path_trace/artifacts/` 下的 `a8w4_*.svg` 流水示意图。
> 相关设计文档：`docs/mega_moe_a8w4_wave_design.md`、`docs/mega_moe_a8w4_critical_path_trace_design.md`。

---

## 1. 图的定位（必读）

- 一律是 **示意图 / 单核代表视图**：专家按 tile 分布到各核，依赖是**数据级**（如"G2(0) 依赖 S0"指依赖**对应 tile 的 swiglu 完成**，可能由其他核完成）；各核流水相近，单泳道近似成立。
- 标题写 **"示意图（单核代表视图）"**，**不要写"真实 trace 数据"**。
- 白底画布 + 高对比文字（浅色/深色主题都可读）；白色背景 `<rect>` 放在第一层。

## 2. 泳道与图元（三泳道固定顺序）

| 泳道 | 图元 | 填充 | 文字色 | 备注 |
| --- | --- | --- | --- | --- |
| AIC | GMM1 | `#1d4ed8`（蓝） | 白 | 73us |
| | GMM2 | `#0891b2`（青） | 白 | 38us |
| AIV0 | G1/G2 权重 prologue（W4→W8） | `#ccfbf1` + 描边 `#0d9488` | `#134e4a` | P=G1、Q=G2；浅色描边表示"支持性任务、不在关键路径" |
| AIV1 | dispatch（跨卡） | `#f59e0b`（琥珀） | `#1f2937` | 30us |
| | swiglu | `#10b981`（翠绿） | `#064e3b` | 12us |
| | combine | `#8b5cf6`（紫） | 白 | 48us |
| 空泡（等 GMM1 等） | `#e5e7eb` 虚线框 | `#6b7280` | 标注数值，如"首空泡 43 / 空泡 31 / 尾空泡 61" |
| 起止标记 | 开始=绿 `#059669`、结束=红 `#dc2626` 竖虚线 | — | 标签"开始 t=0 / 结束 t=xxx" |
| 对齐/依赖虚线 | 红 `#dc2626` 虚线（跨泳道时 opacity ≤ 0.55） | — | 如对齐律、C 晚启 |

泳道顺序固定：**AIC（上）→ AIV0（中）→ AIV1（下）**；跨三泳道的起止/对齐线会穿过 AIV0。

## 3. 时间轴与坐标

- **1us = 2px**；`x = 100 + 2 * t`。
- `t=0` = dispatch(0) / P(0) 启动。
- 每张图必须包含：
  1. 绿竖线 `开始 t=0`、红竖线 `结束 t=xxx`（跨三条泳道）；
  2. 加粗标注 **`算子总时长 ≈ xxxus（t=0：D(0)/P(0) 启动 → t=xxx：C(3) 完成）`**；
  3. 时间轴末端刻度（= 结束 t 值）。

## 4. 单专家实测时长（固定值，直接引用）

| 项 | 时长 |
| --- | --- |
| D dispatch（跨卡） | 30us |
| G1 GMM1 | 73us |
| S swiglu | 12us |
| G2 GMM2 | 38us |
| C combine | 48us |
| P/Q prologue | 与对应 GMM 重叠、比其早结束 **2~5us（默认取 4us）** |

## 5. 依赖与衔接规则（画图必须遵守）

1. **逐专家链**：`D → G1 → S → G2 → C`。
2. **衔接靠逐 wave / 逐 tile 的 flag**：`dispatchToGmm1Flag`、`swigluToGmm2Flag`、`gmmToEpilogueFlag`。
3. **没有任务型 EndSync、没有 SyncAll**（SyncAll 属 topkWeightsPrefetch 路径）；**没有全局屏障**。
4. **对齐律**（分阶段图）：`G1(e+1).start = S(e).start`（红虚线）。
5. **G2(0) 紧贴 G1(3)**：G2(e) 只依赖 S(e)，而 S0/S1/S2 在 GMM1 阶段内早已完成 → 无空隙。
6. **combine 比 G2 晚启 14us**（首 tile 等待；tile 级：G2 启动后 14us 即可开始 combine）。
7. **prologue 约束**：比对应 GMM 早结束 2~5us（取 4us）+ **尽可能早启动**（紧密排布：前一个结束、下一个马上开始）。
8. **wave 结构（模拟图）**：
   - **1 专家/wave**：AIC 按 `G1(e)→G2(e)`；G2(e) 等 S(e) 12us → 每专家 12us 空泡，周期 **123us/专家**。
   - **2 专家/wave**：AIC 按 `G1(e0)→G1(e1)→G2(e0)→G2(e1)`；S(e) 藏在同 wave 的 G1(e1)/G2(e0) 后 → 无空泡，周期 **111us/专家**。
   - 循环参考 A8W8 `MegaMoeWave`：AIC `G1(B)→G2(B)→G1(B+1)`；AIV1 预取 `D(B+1)` + `C(B)`；AIV0 prologue 跨 wave 预取。

## 6. 典型形态与总时长（可直接引用）

| 形态 | 总时长 | 备注 |
| --- | --- | --- |
| 当前分阶段 | ≈ 528us | Phase1 AIC-bound（73us/专家）；AIV1 空泡 ~31us/专家（首 43 / 尾 61）；Phase2 AIV1-bound（C 48 > G2 38）；AIC 富余 ~10us/专家 |
| wave-1（1 专家/wave） | ≈ 570us | 每专家 +12us 等 S 空泡（**负面案例**，只验证结构） |
| **wave-2（2 专家/wave）** | **≈ 508us** | **最优，模型下限** |
| 508 下限推导 | — | AIC 连续：G2(3) ≥ `30 + 4×73 + 3×38` = 436 起；combine 串行：C(3) ≥ C(2) ≥ G2(2)tile(412)+48 = 460 → wall ≥ 460+48 = 508 |

## 7. 绘制 checklist

- [ ] 白色背景 `<rect>` 第一层 + 高对比文字（深色主题可读）
- [ ] 三泳道（AIC/AIV0/AIV1）+ 起止竖线（绿/红）+ `算子总时长 ≈ xxxus` 标注 + 末端刻度
- [ ] prologue 校验：每个比对应 GMM 早结束 2~5us、且紧密排布（可用脚本核对）
- [ ] XML 合法、元素不出画布
- [ ] 标题为"示意图（单核代表视图）"
- [ ] 文件名：`artifacts/a8w4_*.svg`

## 8. 现有文件

- `artifacts/a8w4_pipeline_overview.svg` —— 当前分阶段流水（528us）
- `artifacts/a8w4_wave1_pipeline_overview.svg` —— wave-1 模拟（570us，负面案例）
- `artifacts/a8w4_wave2_pipeline_overview.svg` —— wave-2 模拟（508us，最优）
- `docs/mega_moe_a8w4_wave_design.md` —— 由上述数据推导的 wave 改造设计文档
