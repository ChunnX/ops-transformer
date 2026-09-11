# MEMORY — pdist（PyPTO-Pro Stage 1 规划）

## 任务摘要

- 任务：pdist（pair-wise p-范数距离，reduction 类复杂算子）Stage 1 需求规划
- 主需求输入：`/data/z00954450/zzz/pdist_analysis_report.md`（✓高置信度：PyTorch 2.10 本地实测 + 源码实读）
- 确认状态：无人值守模式，基于高置信度报告裁定；非阻塞默认全部披露于 SPEC §12
- 目标设备：A5（Ascend 950PR，CANN 9.2.0，TILE_FWK_DEVICE_ID=0）——运行时已确认，非默认假设

## Stage 1 产物（全部完成）

| 产物 | 路径 | 状态 |
|------|------|------|
| SPEC.md | `custom/pdist/SPEC.md` | `validate_spec.py` PASS（2026-09-01） |
| PRO_MATERIAL_INDEX.md | `custom/pdist/PRO_MATERIAL_INDEX.md` | §A 314 API 文档（48 组）/ §B 13 清单项（12 在场）/ §C 40 教程 |
| EXPLORE_REPORT.md | `custom/pdist/EXPLORE_REPORT.md` | 10 章节完整；可行性=可行；无 unsupported |
| MEMORY.md | `custom/pdist/MEMORY.md` | 本文件 |
| KB_SELECTION.json | `custom/pdist/KB_SELECTION.json` | plan skill 预检 OK（schema_version=3） |

## kernel 契约补充裁定（详见 SPEC 末尾表格）

- 无辅助 tensor；`p` 为标量属性（host 侧编译期常量，TilingData 传递；p<0 host 预校验）
- cast 链：IO fp32/fp16 → kernel 内全程 fp32（diff/pow/累加/inv-pow）→ 出口单点 narrow（仅 fp16 case，非饱和保 NaN/Inf）
- 写回语义：逐对独立、覆盖写，idx = i*(2N-i-1)/2 + (j-i-1) 唯一；无原子/无 GM 累加
- topology/tile/同步：Stage 3 设计，Stage 1 不猜测

## 字段裁定记录（SPEC §12 自动决策）

| # | 裁定 | 依据 |
|---|------|------|
| 1 | 单 class flat（class_id="."）；machine-contract 冻结 fp32 组合，fp16 为同接口第二 IO dtype（正文 + cast 链 + KB properties.dtypes + P0 case 注明） | 拆 class 规则的保护目标（辅助 tensor dtype 丢失）不存在；下游 Stage 2–4 单文件链无多 class 协议；校验器 dtype 字段单值 |
| 2 | x.value_range=[-65504, 65504]（fp16 全有限值域） | IEEE-754 half 范围 + 报告 §4/§6 |
| 3 | M 上限 65535（数学无上限，合同需有限区间） | 报告 §7 + 合同约束 |
| 4 | p=+inf 在 p0_cases 用字符串 "inf"（JSON 拒绝非有限数） | 校验器 parse_constant 行为 |
| 5 | perf_target=null；Stage 5 默认 golden_reference_ratio>=1.0 | 用户无数值目标 |
| 6 | y.value_range=[0, 65504] | 距离非负 + fp16 max；P0 期望输出全在界内 |

## 发现的上游报告笔误（已按公式修正，SPEC §10 已注明）

- 报告 §9 `性能_P0_fp16` 输出长度 33,558,528 → **正确 33,550,336**（8192*8191/2）
- 报告 §9 `功能_P0_部分块` 输出长度 301,086 → **正确 301,476**（777*776/2）
- 后续 Stage 引用报告 §9 时以 SPEC 公式 N*(N-1)/2 为准

## KB 路由结果（公式拓扑路由，非算子名）

- **topologies: ["long-axis"]**——M∈[1,65535] 无上界 + P0 func_p0_large_m（M=8192）强制多 tile 遍历携带部分和
- **评估过不匹配**（供 verifier 复核）：row-reduction（pdist 是**行对**规约、结果离开 kernel 不广播回行——pattern 文件明确排除跨行共享规约）；elementwise（输出元素数≠输入）；multi-phase-fusion（无多语义 phase 状态携带，M 分块部分和属规约循环常态）；cube-matmul（SPEC 冻结公式为 abs-diff 直接规约，非 Gram 矩阵法——p=2 的等价改写属 Stage 3 设计空间，不改变 Stage 1 路由）；scan/gather-scatter/paired-lane-rotation/row-wise-dynamic-quant（语义不符）
- properties：dtypes=[fp32,fp16]、ranks=[2,1]、unaligned_shapes/tail_blocks/long_axis/mixed_precision/dynamic_dims=true、index_dtypes/is_list=false
- required_constraints ×7：tiling、memory-layout、tail-validshape、precision、vec-mask-width、arch-a5（target）、wrapper-boundary（mandatory）
- optional_patterns ×1：buffer-reuse-lifetime（M 分块轮转槽位数法则 + persistent 部分和 + scratch 输出）；no_matching_pattern=false

## 公式步骤 → API 候选速查（详见 EXPLORE_REPORT §2/§3）

- P0 五个 p 值**全部精确 API 链**（无近似）：
  - p=2：`vf.abs_sub` → `vf.mul(d,d)` → `vf.reduce_sum` → `vf.sqrt`
  - p=1：`vf.abs_sub` → `vf.reduce_sum`
  - p=0.5：`vf.abs_sub` → `vf.sqrt` → `vf.reduce_sum` → `vf.mul(s,s)`
  - p=0：`vf.abs_sub` → `vf.ne(d,0)` → `vf.select(1,0)` → `vf.reduce_sum`
  - p=inf：`vf.abs_sub` → `vf.reduce_max`（NaN 传播语义文档确认）
- 通用结构：`pl.load`（[i,col]/[j,col] 行取数）→ `vf.load_align` →（fp16 widen `vf.astype`）→ 分支 → 多寄存器规约（`vf.update_mask` 尾 lane + `vf.add` lane0 合并，layernorm 样例模式）→（fp16 narrow）→ `vf.store_align` + `pl.store`
- 一般 p（非 P0）：`vf.ln`→`vf.muls`→`vf.exp`（近似；ln 正数约束，d=0 域外——Stage 3 裁定）
- 最高参考样例：`pro_ops/vf_api/test_layernorm_tile_group_vf.py`（动态双轴+行规约+尾 lane+多核，case [1000,200]/[777,300] 与 pdist P0 同 shape）

## 关键风险 / 事实（详见 EXPLORE_REPORT §7/§8）

1. **fp16 规约必须先 widen**：`vf.reduce_sum` FP16 src = FP16 累加（违反 SPEC）——Stage 4/verifier 检查点
2. `vf.ln` 正数定义域：d=0 在域外（仅影响非 P0 一般 p）
3. 输出散布写：同 i 连续 j 段 idx 连续可攒批（memory-bound 关键，Stage 3）
4. **idx 整数溢出**：N=65535 时中间量 i*(2N-i-1)≈8.59e9 超 int32——索引算术须 int64（kernel + golden）
5. 寄存器 32 上限（溢出 spill 8KB UB）；FP32 VL=64
6. UB 248KB（教程直接陈述，与平台文件 253952 一致）；tile 32B 对齐
7. 官方样例清单 #2 缺失（上游版本漂移，非装配失败；matmul 入门样例与 pdist 无参考关系，非阻断，已如实披露于 INDEX §B / EXPLORE_REPORT §8.1）

## 尝试历史 / 阻塞

- 无阻塞。报告"Ascend 生态缺口"节（C++ 算子包未适配 950、op-plugin 无映射）为背景信息——本任务走 PyPTO-Pro DSL 路径，不依赖该包可运行性
- 环境备注：`.devkit/pro_ops` 与 `.devkit/docs` 为 symlink，`find` 枚举须 `-L`（否则误报空目录）
