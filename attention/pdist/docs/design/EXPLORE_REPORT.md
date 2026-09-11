---
schema_version: 2
op_name: pdist
feasibility: feasible
---

# PyPTO-Pro 资料探索报告

> **生成时间**: 2026-09-01
> **全量资料索引**: `custom/pdist/PRO_MATERIAL_INDEX.md`
> **输入 SPEC**: `custom/pdist/SPEC.md`（machine-contract 校验 PASS；fp32 规范组合 + fp16 第二 IO dtype；p 三分支；N∈[2,65535]、M∈[1,65535]）
> **目标设备**: A5（Ascend 950PR，运行时确认）

---

## 1. 概述

### 1.1 输入摘要

pdist：输入 `x (N, M)`（fp32/fp16），标量属性 `p ∈ {0} ∪ (0,+∞) ∪ {+inf}`，输出 `y (N(N-1)/2,)`（与 x 同 dtype），上三角行优先索引 `idx = i*(2N-i-1)/2 + (j-i-1)`。逐对沿 M 规约，中间 fp32 累加，NaN/Inf 逐对传播。P0 覆盖 p ∈ {0, 1, 2, 0.5, inf} 全部分支 + 最小 shape + M 大分块 + 非整块尾部 + fp16 大规模。

### 1.2 算子分类

- **计算引擎类型**: 纯 **Vector（VF 寄存器路径）**
- **判断依据**: 全部原子操作为逐元素算术（差/绝对值/幂/比较）+ 沿 M 的规约（sum/max）+ 标量后处理（1/p 幂）+ 数据搬运；**无 matmul/收缩**（Cube 不参与）；输出元素数 ≠ 输入元素数（非 elementwise 拓扑）

### 1.3 可能会涉及的 API 类别

> 实际 API 以本次 PRO_MATERIAL_INDEX §A 扫描结果为准。Vector 规则见 [Vector 选择规范](../../../references/performance-constraints.md)；本阶段记录默认 VF 映射，KB 模板例外交 Stage 3 裁定。

| 类别 | 是否涉及 | 关键 API | 指定算子是否覆盖 |
|------|---------|----------|------------------|
| 数据搬运 | 是 | `pl.load` / `pl.store` / `vf.load_align` / `vf.store_align` | 覆盖（add/layernorm/softmax 样例） |
| VF 实现 | 是 | `vf.abs_sub` / `vf.sub` / `vf.abs` / `vf.mul` / `vf.muls` / `vf.sqrt` / `vf.ln` / `vf.exp` / `vf.ne` / `vf.select` / `vf.astype` / `vf.reduce_sum` / `vf.reduce_max` / `vf.update_mask` / `vf.full` / `vf.add` | 覆盖（vf_api 类样例） |
| 矩阵计算（Cube） | **否** | — | 不适用（无收缩） |
| 系统访问 | 是 | `get_block_idx` / `get_block_num` | 覆盖（全部样例） |
| 控制流 | 是 | `section_vector` / `pl.range` / kernel 内 `if`（或 TilingKey 编译期分支） | 覆盖（fa tilingkey 样例） |
| 工具 | 是 | `TileType`(valid_shape) / `make_tile_group`(N-buffer) / `set_validshape` / `TilingData`(@dataclass) / `TilingKeyField` | 覆盖 |

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | vf.* 调用链（vec）/ pl.* 调用链（cube） | 说明 |
|------|----------|----------|--------------------------------------|------|
| 1 | 搬运 GM→UB | `t_i = x[i, :], t_j = x[j, :]`（M 分块时按列偏移） | `pl.load(tile_i, x, [i, col_off])` → `pl.load(tile_j, x, [j, col_off])` | offsets 为绝对元素坐标；M 超 tile 宽时列分块循环 |
| 2 | 搬运 UB→Reg | 载入两行 | `vf.load_align(tile_i, off)` → `vf.load_align(tile_j, off)` | fp32 时每寄存器 64 元素（VL=64） |
| 3 | cast（仅 fp16 输入） | `d32 = fp32(x16)` | `vf.astype(reg, preg, dtype=pl.DT_FP32)` | FP16→FP32 widen（astype 表3：layout=UNKNOWN 可省） |
| 4 | 逐维差+绝对值 | `d_k = |x_ik - x_jk|` | `vf.abs_sub(reg_i, reg_j, preg)`（单指令） | 等价组合：`vf.sub` → `vf.abs`；abs_sub 为文档化复合 API |
| 5 | p 分支预处理 | 见 5a/5b/5c | 见下 | 分支选择与 p 值绑定（编译期或运行时，Stage 3 裁定） |
| 5a | 幂（p=2） | `d²` | `vf.mul(d, d, preg)` | 精确（无超越函数） |
| 5b | 幂（p=0.5） | `d^{1/2}` | `vf.sqrt(d, preg)` | 精确；src 非负约束天然满足（d=|·|≥0） |
| 5c | 幂（p=1） | 恒等 | `d` 直通 | — |
| 5d | 幂（整数 p≥3） | `d^p` | `vf.mul` 链 ×(p-1) | 精确；寄存器代价随 p 增长 |
| 5e | 幂（一般 p） | `d^p = exp(p·ln d)` | `vf.ln(d)` → `vf.muls(ln_d, p)` → `vf.exp(...)` | **近似路径**（见 §2.1 sanity check 与 §8 约束） |
| 5f | 比较（p=0） | `𝟙(d≠0)` | `vf.ne(d, 0.0, preg)` → `vf.select(ones, zeros, mask)` | ne 返回 mask_reg；select 产出 0/1 reg_tensor |
| 6 | 沿 M 规约 | `s = Σ_k d'_k`（p=inf: `max_k d_k`） | `vf.reduce_sum(reg, mreg)`（或 `vf.reduce_max`）；多寄存器时逐 reg 归约 + `vf.add` 合并 lane0 累加器 | 尾寄存器用 `vf.update_mask(valid)` 掩码（layernorm 样例模式）；reduce_sum FP32→FP32 累加 |
| 7 | inv-pow（p=2） | `s^{1/2}` | `vf.sqrt(s, preg)` | 精确 |
| 8 | inv-pow（p=0.5） | `s^{2}` | `vf.mul(s, s, preg)` | 精确 |
| 9 | inv-pow（p=1） | 恒等 | 直通 | — |
| 10 | inv-pow（一般 p） | `s^{1/p} = exp(ln s / p)` | `vf.ln(s)` → `vf.muls(ln_s, 1.0/p)` → `vf.exp(...)` | 近似路径 |
| 11 | cast（仅 fp16 输出） | `y16 = fp16(s32)` | `vf.astype(reg, preg, dtype=pl.DT_FP16, layout=pl.CastLayout.ZERO, round_mode=pl.VFRoundMode.CAST_RINT)` | FP32→FP16 narrow；非饱和模式 NaN→NaN、±inf→±inf（NaN/Inf 传播语义成立） |
| 12 | 写回 UB→GM | `y[idx] = y_pair` | `vf.store_align(out_tile, reg, preg, off)` → `pl.store(y, out_tile, [idx_off])` | 批量攒批写回或逐对标量写（Stage 3 裁定；`pl.store` offsets 长度 = tensor 维数，一维输出合法） |

> **P0 关键结论**：P0 覆盖的 5 个 p 值（0 / 1 / 2 / 0.5 / inf）**全部存在精确 API 链**（ne+select / 恒等 / mul+sqrt / sqrt+mul / reduce_max），不依赖 exp/ln 近似；exp/ln 仅服务非 P0 的一般 p（P1/P2 范围）。

### 2.1 近似步骤 sanity check（仅一般 p 的 exp/ln 路径）

- `d^p = exp(p·ln d)`：d=1, p=2.5 → exp(0)=1=1^2.5 ✓；d=4, p=2 → exp(2·ln4)=exp(ln16)=16=4² ✓
- `s^{1/p} = exp(ln s / p)`：s=16, p=2 → exp(ln16/2)=exp(ln4)=4=√16 ✓；s=27, p=0.5 → exp(ln27/0.5)=exp(2·ln27)=729=27² ✓
- 适用区间：d>0、s>0（ln 定义域）；d=0 / s=0 行为见 §8.2 风险 R2
- 精度：vf.exp/vf.ln 提供 `precision=True` 高精度模式（文档确认），误差量级与 round_mode 由 Stage 3 按容差 1e-3 裁定

---

## 3. API 文档探索

> **来源**: 探索方向 1 — 基于 `PRO_MATERIAL_INDEX.md` §A（314 文档；本报告详读与公式相关的 20+ 个，全部路径见 §9）

### 3.1 API 映射结果

| 步骤 | 数学表达 | vf.* 调用链（vec）/ pl.* 调用链（cube） | 状态 | 约束满足 |
|------|----------|--------------------------------------|------|----------|
| 1 | 行对取数 | `pl.load` ×2（offsets=[行,列]） | 直接可用 | ✓ |
| 2 | UB→Reg | `vf.load_align` ×2 | 直接可用 | ✓ |
| 3 | fp16 widen | `vf.astype(..., DT_FP32)` | 直接可用 | ✓ |
| 4 | `\|x_i−x_j\|` | `vf.abs_sub` | 直接可用 | ✓（FP16/FP32 均支持） |
| 5a–5d | 特殊 p 幂 | `vf.mul` / `vf.sqrt` / 直通 | 直接可用 | ✓（精确） |
| 5e | 一般 p 幂 | `vf.ln`→`vf.muls`→`vf.exp` | 需组合 | ⚠（ln 定义域与精度，非 P0） |
| 5f | p=0 计数 | `vf.ne`→`vf.select` | 需组合 | ✓ |
| 6 | 沿 M 规约 | `vf.reduce_sum` / `vf.reduce_max`（+`vf.update_mask` 尾寄存器、`vf.add` 跨寄存器合并） | 直接可用 | ✓ |
| 7–9 | 特殊 p inv-pow | `vf.sqrt` / `vf.mul` / 直通 | 直接可用 | ✓（精确） |
| 10 | 一般 p inv-pow | `vf.ln`→`vf.muls`→`vf.exp` | 需组合 | ⚠（同 5e） |
| 11 | fp16 narrow | `vf.astype(..., DT_FP16)` | 直接可用 | ✓（NaN/Inf 非饱和传播） |
| 12 | 写回 | `vf.store_align` + `pl.store` | 直接可用 | ✓ |
| — | 多核分片 | `pl.get_block_idx/num` + `pl.range(core_id, total, num_cores)` | 直接可用 | ✓ |
| — | p 分支机制 | `TilingKeyField`（编译期折叠）或 kernel 内 `if` | 两种均文档化 | Stage 3 裁定 |
| — | host 参数传递 | `TilingData`（@dataclass，kernel 签名传入） | 直接可用 | ✓（可传 N/M/p） |

**结论：无 `unsupported` 步骤**；全部 P0 语义有可行调用链。

### 3.2 替代方案

| 步骤 | 默认 VF 映射 | API 依据 | 目标版本 | 适用条件 | 交接状态 |
|------|---------------|----------|----------|----------|----------|
| 4 | `vf.abs_sub` | `api/SIMD-API/operation/vf_computation/composite_computation/abs_sub.md`：`dst_i = \|src0_i − src1_i\|`，950PR 支持 | 当前缓存（d6e9b83） | FP16/FP32 reg_tensor | 待 Stage 3 核对已选 KB 模板 |
| 4' | `vf.sub` + `vf.abs` 两步 | `basic_arithmetic/sub.md` + `basic_arithmetic/abs.md` | 当前缓存 | 同上（指令数+1，语义等价） | 备选 |
| 6 | `vf.reduce_sum`（FP32 src 全寄存器归约到 dst[0]，二叉树、确定性） | `vf_computation/reduction/reduce_sum.md` | 当前缓存 | src=DT_FP32 时 FP32 累加（满足 SPEC cast 链） | 待 Stage 3 核对 |
| 6' | `vf.reduce_max`（NaN 写入 dst[0]） | `vf_computation/reduction/reduce_max.md` | 当前缓存 | p=inf 分支；NaN 传播语义与 SPEC 一致 | 待 Stage 3 核对 |
| 5f | `vf.ne`→`vf.select` | `comparison_and_selection/ne.md`（返回 mask_reg）+ `select.md`（支持 mask_reg 源） | 当前缓存 | p=0 分支 | 待 Stage 3 核对 |
| 尾 lane | `vf.update_mask(valid)` | `mask_operations/update_mask.md`（layernorm/softmax 样例内实证用法） | 当前缓存 | M 非 64 倍数的尾寄存器 | 待 Stage 3 核对 |
| lane0 合并 | `vf.add(acc, part)` + `vf.full(acc, preg)` 广播 | `basic_arithmetic/add.md` + `data_movement/full.md`（标量/lane0 广播） | 当前缓存 | 跨寄存器部分和合并 | 待 Stage 3 核对 |

### 3.3 API 约束

| API | 约束项 | 要求 | 结果 | 参数语义/寄存器级行为/同名差异 |
|-----|--------|------|------|----------------------|
| `vf.ln` | src 定义域 | "必须为正数"（文档原文） | ⚠ | d=0（两行相同维度，常见输入）在文档域外；见风险 R2，仅影响非 P0 一般 p |
| `vf.sqrt` | src 非负 | "必须为非负数" | ✓ | d=|·|≥0、s=Σd^p≥0 天然满足 |
| `vf.reduce_sum` | 累加精度 | FP16 src → **FP16 精度累加**；FP32 src → FP32 | ✓ | 满足 SPEC "中间 fp32" 要求的前提：**fp16 输入必须先 astype 到 FP32 再规约**（禁止 FP16 直接规约） |
| `vf.reduce_sum/max` | 输出形态 | 结果写 dst[0]，其余置零；reduce_max 额外把首个最大值索引写 dst[1] | ✓ | 后续按 lane0 取用（`vf.full` 广播或 store 后取首元素） |
| `vf.reduce_max` | NaN | 输入含 NaN → dst[0]=NaN（首个 NaN 索引入 dst[1]）；max(−0,+0)=+0 | ✓ | p=inf 分支 NaN 传播语义成立 |
| `vf.ne` | 返回类型 | 返回 **mask_reg**（比特位 1/0），非 reg_tensor | ✓ | 计数需经 `vf.select`（或等价）转 reg_tensor 再 reduce_sum |
| `vf.astype` FP16↔FP32 | 转换参数 | FP16→FP32（widen，layout 可省）；FP32→FP16（narrow：`layout`、`round_mode` 默认 CAST_RINT、`saturate`） | ✓ | 浮点转浮点**非饱和模式**：NaN→NaN、±inf→±inf（NaN/Inf 传播）；饱和模式 NaN→0——**必须用非饱和**（默认 OFF） |
| `vf.exp` / `vf.ln` | 精度模式 | `precision=True` 高精度开关 | ✓ | 一般 p 路径如需提精度可启用（Stage 3 裁定） |
| `pl.load` / `pl.store` | offsets | 绝对元素坐标，长度=tensor 维数（一维输出 y 用 `[idx]` 合法）；不支持负索引 | ✓ | 尾块 Tile shape 须 32B 对齐、valid_shape 可不对齐 |
| `pl.load`/`pl.store`/`vf.load_align` | 地址对齐 | UB/L1 首地址 32 字节对齐；`vf.load_align` tile 地址 32B 对齐 | ✓ | tile 布局（fp32 列数×4B 须 32B 倍数）由 Stage 3 定 |
| `vf.reg_tensor` | 数量上限 | 32 个寄存器；溢出 spill 到预留 8KB UB（性能劣化） | ⚠ | 双行载入 + 幂中间量 + 累加器并存时的寄存器预算是 Stage 3 要点（风险 R5） |
| `vf.reg_tensor` | VL | FP32 64 元素/寄存器（256B）；FP16 128 元素 | ✓ | M 分块的 n_regs = ceil(M/64)（fp32） |
| `make_tile_group` | mutex/event | mutex_ids ∈ [0,31]；fwd/bwd_ids event_id ∈ [0,15] | ✓ | pdist 无跨核同步需求（逐对独立），仅需双缓冲 mutex |
| `TilingKey` | 适用性 | 编译期折叠为常量、每 Key 独立 Kernel 实例；"应仅描述有限且会影响代码生成的模式" | ✓ | p 三分支（0/(0,∞)/inf）与幂策略（精确/超越）是有限枚举，符合适用条件 |
| `TilingData` | 传递 | @dataclass 字段全标量/定长数组；host 构造后经方括号启动语法传入 | ✓ | 可传 N、M、p 及派生 tiling 量；`p<0` 可在 host 侧预校验 |

### 3.4 MemorySpace 约束

| 操作 | MemorySpace | 约束 |
|------|-------------|------|
| 输入 x / 输出 y | GM（`pl.Tensor`，默认 ND layout） | load 支持 ND/DN/NZ；pdist 用 ND（行主序自然布局） |
| 输入 tile（两行缓冲） | `MemorySpace.Vec`（UB） | 单核 UB 预算 248KB；tile 32B 对齐；双缓冲×2 tile |
| 输出 tile（攒批） | `MemorySpace.Vec`（UB） | `pl.store` 源须 Vec（UB）或 Acc；pdist 用 Vec |
| 计算中间量 | VF 寄存器（reg_tensor） | 32 寄存器上限；FP32 VL=64 |
| 无 Cube 路径 | — | 不涉及 Mat/Left/Right/Acc 空间 |

---

## 4. 算子样例探索

> **来源**: 探索方向 2 — 基于 `PRO_MATERIAL_INDEX.md` §B（官方指定算子；12/13 在缓存，见 §B 核对说明）
> **注意**: 官方指定算子为精选实现参考。a5 目录下其余文件不得参考。

### 4.1 全量样例参考（按 cube/vec 组成分类）

**纯 Cube 样例**（matmul 类，pdist 无 Cube 计算，仅提取通用工程模式）：

| # | 示例路径 | 可复用点 |
|---|----------|----------|
| 1 | `pro_ops/matmul/test_matmul_perf_asw_4k_dn_move_offset.py` | 动态 shape `pl.DYNAMIC` + `a.shape[0]` 运行时读维 + TilingData host 传参的工程骨架 |
| 2 | `pro_ops/matmul/test_matmul_perf_asw_8k_k128_dn_move_offset.py` | 同上（K=128 变体） |
| 3 | `pro_ops/matmul/test_matmul_perf_asw_4k_dn_move_offset_dynamic.py` | 全动态 M/N/K 单一编译变体复用（pdist 的 N/M 动态同构需求） |
| 4 | `pro_ops/matmul/test_matmul_perf_asw_8k_k128_dn_move_offset_dynamic.py` | 同上 |

**纯 Vec 样例**（elementwise / vf_api 类——pdist 的直接同类）：

| # | 示例路径 | 可复用点 |
|---|----------|----------|
| 5 | `pro_ops/element_wise/test_add.py` | `@pl.jit(auto_mutex=True)` + `make_tile_group` 双缓冲（`addrs` + `mutex_ids=[0,1]`）+ `pl.load_tile` 按块坐标 + `pl.range(core_id, m_tiles, num_cores)` 跨步多核——纯 Vec 多核标准骨架 |
| 6 | `pro_ops/vf_api/test_layernorm_tile_group_vf.py` | **最高参考价值**：全动态双轴（`[[DYNAMIC, DYNAMIC], DT_FP32]`）；行跨 `n_regs=ceil(N/64)` 寄存器规约 + `vf.update_mask(valid)` 尾 lane 掩码 + `vf.reduce_sum`→`vf.add` lane0 累加合并；`pl.set_validshape` 部分行尾块；`kernel[None, num_cores]` 启动；pytest + torch golden（rtol/atol 1e-3）验收形态 |
| 7 | `pro_ops/vf_api/test_softmax_tile_group_vf.py` | 两遍规约（`vf.reduce_max` + `vf.reduce_sum`）+ `NEG_INF=-1e30` 归约初值模式（p=inf 分支的 max 规约直接对应）；尾 lane 掩码同上 |

**VC 融合样例**（FA 类、lightning_indexer 类——pdist 无融合，仅提取通用模式）：

| # | 示例路径 | 可复用点 |
|---|----------|----------|
| 8 | `pro_ops/fa/test_fa_perf_tkv_preload_dn_vf_bufid_dynrank.py` | 生产级动态 rank（合轴 `pl.make_tensor` 视图重建）+ NBuffer 预取 |
| 9 | `pro_ops/fa/test_fa_tilingkey_attn_mask.py` | **TilingKey 分支模式**：`TilingKeyField(bits=1, values=[0,1])` + kernel 内 `if NeedAttnMask == 1`——pdist p 三分支的机制参考 |
| 10 | `pro_ops/fa/test_fa_with_mask.py` | NBuffer + auto_mutex 深流水组织；大文件工程化组织 |
| 11 | `pro_ops/fa/test_flex_attention.py` | 复杂分支组合与分段循环组织 |
| 12 | `pro_ops/lightning_indexer/test_quant_lightning_indexer_vf.py` | VF TopK/排序类（pdist 不涉及，仅枚举证明已遍历） |

（清单 #2 `test_matmul_8K_example.py` 缺失：上游仓库已无此文件，见 §B 核对说明与 §8.1。）

### 4.2 可复用模式

**直接可复用**（来自当前算子对应分类的样例，标注来源）：
- **API 调用链**：`pl.load`（GM→UB，行偏移）→ `vf.load_align` → `vf.abs_sub`/`vf.sub` → `vf.reduce_sum`（逐 reg）→ `vf.add` lane0 合并 → `vf.store_align` → `pl.store`（来源：`pro_ops/vf_api/test_layernorm_tile_group_vf.py` 的 pass-1 行求和链）
- **Tile 配置**：`TileType(shape=[TILE_ROWS, MAX_N], DT_FP32, Vec, valid_shape=[-1,-1])` + 双缓冲 `addrs=[A0, A1], mutex_ids=[0,1]`（来源：layernorm 样例）
- **同步策略**：`@pl.jit(auto_mutex=True)` 全自动（make_tile_group mutex 元数据驱动，无手写 sync_src/sync_dst）（来源：`pro_ops/element_wise/test_add.py`）
- **循环结构**：核间 `pl.range(core_id, num_tiles, num_cores)` 跨步 + 核内 `pl.range` 寄存器分块双层循环（来源：layernorm 样例）

**通用写法参考**（来自所有样例）：
- **API 用法**：`vf.update_mask(valid, dtype=pl.DT_FP32)` 构造尾 lane 掩码；`vf.full(scalar/lane0, preg)` 广播；`from pypto_pro.language import Vf as vf` 的 vf 命名空间导入
- **分核策略**：跨步循环天然负载均衡（最多差 1 tile，无首尾核分支）；`num_cores = min(物理核数, 任务数)` 启动
- **同步事件管理**：纯 Vec + auto_mutex 场景无需显式 event_id 分配（样例均未手写）；仅同一 tile 地址连续写入且未读取时需 `pl.system.bar_mte2()`（load.md 高级用法节）
- **尾块处理**：`pl.set_validshape(tile, [valid_rows, cols])` 须在 load 之前设置；规约场景 padding 不参与（update_mask 掩码）
- **测试形态**：`@pytest.mark.soc("950")` + `@pypto.options(pass_options={"enable_slice": False})` + `torch.testing.assert_close(rtol=1e-3, atol=1e-3)` + `TILE_FWK_DEVICE_ID` 环境变量取卡

### 4.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| 规约单元 | layernorm/softmax：**单行**沿列规约后广播回行（输出 shape 同输入） | **行对 (i,j)** 联合规约，每对输出独立标量到 y[idx] | 同核内加载两行做 `abs_sub`；规约链本身（update_mask/reduce/add）原样复用 |
| 输出写回 | 样例输出 tile 与输入 tile 同形，按 `[row_off, 0]` 整块 store | 每对标量按上三角 idx 散布写（或攒批连续段写） | Stage 3 裁定：pair 枚举按行对分块（同一 i 的连续 j 段 idx 连续，天然可攒批）；`pl.store` 一维 offsets |
| 分支机制 | fa：TilingKeyField 编译期折叠 + `if` 混合 | p 三分支（0/(0,∞)/inf）+ 幂策略（精确/超越） | p 是连续标量——**p 三分支建议 TilingKey 或 host 侧归一化分支标签**；一般 p 的超越路径仅 P1/P2（Stage 3 裁定） |
| 输入 dtype | 样例均为 FP32（vf_api）或 FP16（add）单一 dtype | fp32 与 fp16 两 IO dtype + **fp32 中间累加** | fp16 路径：load 后立即 `astype` widen，规约前完成；出口单点 narrow（astype 约束已核） |

### 4.4 高参考价值样例推荐

| 样例路径 | 推荐理由 |
|----------|----------|
| `pro_ops/vf_api/test_layernorm_tile_group_vf.py` | 与 pdist 的计算形状最接近（动态双轴 + 沿列规约 + 尾 lane + 多核 + 尾行块），规约链与工程骨架可直接移植；其测试 case `(1000,200)`/`(777,300)` 与 pdist 功能 P0 case 同 shape |
| `pro_ops/vf_api/test_softmax_tile_group_vf.py` | 补充 `vf.reduce_max`（p=inf 分支）与两遍规约组织 |
| `pro_ops/element_wise/test_add.py` | 纯 Vec 多核 + 双缓冲 + auto_mutex 的最小完整骨架 |
| `pro_ops/fa/test_fa_tilingkey_attn_mask.py` | TilingKey 分支声明与使用（p 三分支机制参考） |
| `pro_ops/matmul/test_matmul_perf_asw_4k_dn_move_offset_dynamic.py` | 全动态维度的 TilingData/运行时 shape 读取组织（虽为 Cube 样例，模式通用） |

---

## 5. 教程与设计指南探索

> **来源**: 探索方向 3 — 基于 `PRO_MATERIAL_INDEX.md` §C（40 文档全量遍历；`tutorials/` 非空，无阻断）

### 5.1 适用的设计模式

| 指南/教程文档（索引 §C） | 来源目录 | 设计模式 | 适用性 |
|---------------------------|----------|----------|--------|
| `tutorials/operator_development/tile_based_python_programming/Python_programming_overview.md` | tile_based_python_programming | tile 编程总览 | **适用**：Vec/数据流编程模型的入口框架 |
| `tutorials/operator_development/tile_based_python_programming/Reg_vector_computation.md` | 同上 | Reg（VF）矢量计算编程 | **高度适用**：pdist 的主实现范式（VF 函数/执行域/同步与依赖/使用建议） |
| `tutorials/operator_development/tile_based_python_programming/Tile_vector_computation.md` | 同上 | Tile 级 vector 计算 | **适用**：tile-op 与 vf 两种表面的取舍背景（pdist 倾向 vf，最终 Stage 3 裁定） |
| `tutorials/operator_development/tile_based_python_programming/multi_core_partitioning_and_Tiling.md` | 同上 | 跨步多核切分 + UB 容量预算（248KB 直接陈述）+ Host↔Device TilingData | **高度适用**：pair 域多核切分与 UB 预算方法论 |
| `tutorials/operator_development/tile_based_python_programming/tail_block_handling.md` | 同上 | valid_shape/set_validshape/compact/pad+fillpad 四类尾块 | **高度适用**：M 非 tile 倍数、N 非整核均分的尾块（P0 case [777,300]） |
| `tutorials/operator_development/tile_based_python_programming/tiling_key.md` | 同上 | TilingKey 编译期分支 | **适用**：p 三分支/幂策略分支机制 |
| `tutorials/operator_development/tile_based_python_programming/TilingData.md` | 同上 | @dataclass TilingData + 动态 shape/尾块完整示例 | **适用**：N/M/p 传参与 host 侧 `p<0` 预校验 |
| `tutorials/operator_development/tile_based_python_programming/Cube_matrix_computation.md` | 同上 | Cube matmul | 不适用（pdist 无 Cube；仅证明遍历完整） |
| `tutorials/operator_development/kernel_function.md` | operator_development | kernel 函数结构（签名/jit/启动） | **适用**：kernel 骨架基础 |
| `tutorials/operator_development/compilation_and_execution/JIT_compilation.md` | compilation_and_execution | JIT 编译 | 适用：运行/调试与编译缓存（DYNAMIC 复用变体）认知 |
| `tutorials/operator_development/compilation_and_execution/offline_binary_compilation.md` | 同上 | 离线编译 | 不适用（本流程 JIT 路径） |
| `tutorials/quick_start/SIMD/Add_operator.md` | quick_start/SIMD | Vec 入门 | 适用：最小 Vec kernel 形态参考 |
| `tutorials/quick_start/SIMD/CV_fused_operator.md` | 同上 | Cube+Vec 融合入门 | 不适用（无融合） |
| `tutorials/quick_start/SIMD/HelloWorld.md` | 同上 | 环境验证 | 适用：环境冒烟 |
| `tutorials/quick_start/SIMD/Matmul_operator.md` | 同上 | matmul 入门 | 不适用 |
| `tutorials/quick_start/SIMT/Add_operator.md` 及 `tutorials/programming_paradigm/simt_programming.md` | quick_start/SIMT、programming_paradigm | SIMT 线程级编程 | 不适用（pdist 用 SIMD/VF 范式；证明遍历完整） |
| `tutorials/programming_paradigm/abstract_hardware_architecture.md` | programming_paradigm | 硬件架构（UB/寄存器/流水） | 适用：容量与流水背景 |
| `tutorials/programming_paradigm/programming_paradigm_overview.md` | 同上 | 范式总览 | 适用 |
| `tutorials/advanced_programming/aclnn_operator_project_development.md` | advanced_programming | aclnn 算子工程 | 不适用（非本任务交付形态；背景认知） |
| `tutorials/advanced_programming/advanced_ai_core_programming_model.md` | 同上 | 高级 AI Core 编程模型 | 部分适用（多核/执行域细节） |
| `tutorials/advanced_programming/ai_framework_operator_adaptation.md` | 同上 | 框架适配 | 不适用 |
| `tutorials/advanced_programming/aot_compilation_optimization.md` | 同上 | AOT 编译优化 | 不适用（JIT 路径；Stage 5 性能阶段如涉及再取） |
| `tutorials/advanced_programming/auto_parallel_pipeline.md` | 同上 | 自动流水（pipeline 变换） | 不适用（pdist 逐对独立无跨核流水；保留认知） |
| `tutorials/advanced_programming/hardware_implementation.md` | 同上 | 硬件实现细节 | 部分适用（性能阶段） |
| `tutorials/advanced_programming/operator_graph_integration_development.md` | 同上 | 算子图集成 | 不适用 |
| `tutorials/advanced_programming/superkernel.md` | 同上 | superkernel | 不适用 |
| `tutorials/debugging_and_optimization/functional_debugging.md` | debugging_and_optimization | 功能调试 | 适用：Stage 4 调试方法论 |
| `tutorials/debugging_and_optimization/performance_optimization.md` | 同上 | 性能优化 | 适用：Stage 5 前置认知 |
| `tutorials/introduction.md`、各 `index.md`（10 个） | — | 总览/目录页 | 背景导航，无直接设计输入 |

### 5.2 来自教程的关键约束与建议

| 来源 | 约束/建议 | 影响 |
|------|----------|------|
| `multi_core_partitioning_and_Tiling.md` §"UB切分的要点" | **UB 容量 Ascend 950PR/DT = 248KB**；"所有并存 Tile 的总字节数（包括双缓冲副本）不能超过 UB 容量"；文中给出 FP32 缓冲超限时的缩 tile 实例 | Stage 3 双行输入 tile + 输出 tile + 双缓冲副本的 UB 预算硬约束 |
| 同上 §"跨步循环" | `pl.range(core_id, total, num_cores)` 各核任务量最多差 1，无首尾分支；可二维切分 | pair 域切分的标准做法与负载均衡证明 |
| `tail_block_handling.md` | valid_shape 须在 load 之前 set；"测试只使用满块 shape"是常见错误；给输入和输出 tile 都要考虑 valid_shape | P0 case [777,300]、M 尾块、输出尾段的验收覆盖依据 |
| `tiling_key.md` | "TilingKey 应仅描述有限且会影响代码生成的模式"；字段编译期折叠、每 Key 独立 Kernel | p 分支若用 TilingKey 须枚举有限值（三分支/幂策略），不可枚举任意 p |
| `TilingData.md` | 字段全标量/定长数组；未显式传入用 dataclass 默认值；host 方括号语法传实例 | N/M/p 与派生量的规范传递通道 |
| `Reg_vector_computation.md` §"VF函数与执行域" | `@pl.vector_function` 内创建/使用寄存器、函数结束自动释放 | pdist 的 pair 内计算封装为 VF 函数的组织方式 |

---

## 6. Stage 3 设计事实输入

> **综合来源**: §3 API 约束 + §4 样例参考 + §5 教程指导。本节是 Stage 3 的消费入口，只汇总事实与待裁定项，不冻结 tile、Module、同步事件或 topology；这些由 Stage 3 决定。

### 6.1 Tile / 同步约束证据

| 设计问题 | 已证实约束 | 证据路径 | Stage 3 待裁定项 |
|----------|------------|----------|------------------|
| 输入双行 tile 布局 | `pl.load` offsets 支持任意 [row, col]；tile shape 32B 对齐（fp32 列数 8 倍数）；UB 248KB 总预算 | `api/.../memory_data_movement/load.md`；`tutorials/.../multi_core_partitioning_and_Tiling.md` §UB 切分 | tile 宽度（MAX_M 常量，64 的倍数）、每 tile 行数、双缓冲深度 |
| M 分块规约 | `vf.reduce_sum` 全寄存器归约 + `vf.update_mask` 尾 lane + `vf.add` lane0 合并（layernorm 样例完整链） | `pro_ops/vf_api/test_layernorm_tile_group_vf.py` L88–108 | n_regs 上限（寄存器 32 上限内的双行预算）；M=65535 极值的分块轮数 |
| p 分支机制 | TilingKey 编译期折叠（有限枚举）与 kernel 内 `if` 两种文档化机制 | `tutorials/.../tiling_key.md`；`pro_ops/fa/test_fa_tilingkey_attn_mask.py` | p=0/(0,∞)/inf 三分支 + 幂策略（精确 mul/sqrt vs 超越 exp/ln）的分支载体 |
| 输出散布写 | `pl.store` 一维 offsets 合法（长度=维数）；`vf.store_align` 支持列偏移；`pl.getval`/下标语法可读写 tile 单元素（标量 pipe S） | `api/.../memory_data_movement/store.md`；`api/.../transpose_and_element_access/getval.md` | 攒批策略（同 i 连续 j 段 idx 连续）vs 逐对 setval；输出 tile 形状 |
| pair→(i,j) 映射 | idx = i*(2N-i-1)/2 + (j-i-1) 为整数运算；**N=65535 时中间量 i*(2N-i-1) 最大 ≈ 8.59e9 超 int32**（输出总数 2.147e9 贴近 2^31） | SPEC 公式 + 整数范围核算 | kernel 内索引算术的位宽（int64/TilingData 传递策略）与映射实现位置（kernel 标量循环 vs host 预展开） |
| 多核切分 | 跨步循环负载均衡差 1；`kernel[None, num_cores]` 启动；纯 Vec 执行域 block_idx∈[0, num_cores) | `tutorials/.../multi_core_partitioning_and_Tiling.md`；`api/.../system_variables/get_block_idx.md` | pair 域切分粒度（按对/按行对块）与启动核数 |
| fp16 路径 | FP16 reg 直接 `reduce_sum` 为 **FP16 累加**（不满足 SPEC）；须先 `astype` widen FP32；出口 narrow 非饱和保 NaN/Inf | `api/.../reduction/reduce_sum.md` 表1；`api/.../type_conversion/astype.md` 表6 | widen 位置（load 后立即）与出口单点 narrow 的实现点 |
| 同步 | 纯 Vec + auto_mutex + make_tile_group 免手写同步；同 tile 地址连续写未读需 `bar_mte2` | `api/.../load.md` §Tile地址复用 | 输入双行是否各自独立 tile（免 WAW 顾虑）或复用同 tile（需 bar） |
| 一般 p 幂 | `vf.ln` src 必须为正（d=0 域外）；`vf.exp/ln` 有 `precision=True` 高精度模式 | `api/.../basic_arithmetic/ln.md`；`exp.md` | 非 P0 一般 p 的 d=0 边界策略与精度模式（P1/P2 范围，可延后） |

---

## 7. 环境常量快照

> **来源**: 探索方向 1（API 文档约束）+ 探索方向 2（官方指定算例 assert）
> 下游 Stage 3/4 全部引用本节，不再各自写死。无法确认时写 `unknown` 并列入风险。

| 常量 | 探测值 | 来源路径 | 备注 |
|------|--------|----------|------|
| UB 容量 | **248 KB**（Ascend 950PR/950DT） | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/multi_core_partitioning_and_Tiling.md`（"Ascend 950PR/Ascend 950DT为248KB"，§UB切分要点） | 与 KB `constraints/arch-a5.md` 引用的平台文件 `950PR_957x.ini` `ub_size=253952`（=248KB）一致；含双缓冲副本的总预算 |
| 地址对齐要求 | **32 字节** | `api/SIMD-API/operation/memory_data_movement/load.md`/`store.md`（UB/L1 首地址）、`api/SIMD-API/operation/vf_computation/data_movement/load_align.md`（tile 地址） | tile shape 须 32B 对齐（fp32 列数 8 倍数 / fp16 16 倍数）；valid_shape 可不对齐 |
| VF 寄存器 | FP32 **VL=64 元素**（256B/寄存器）；上限 **32 寄存器**，溢出 spill 预留 8KB UB | `api/SIMD-API/operation/vf_computation/reg_tensor.md` | 双行载入 + 幂中间量的寄存器预算依据 |
| cross_core event_id 上限 | **0~15**（fwd/bwd_ids）；mutex_ids **[0,31]** | `api/SIMD-API/operation/resource_management/make_tile_group.md` | pdist 无跨核同步需求（仅双缓冲 mutex） |
| 逻辑核数 | 由启动 `block_dim` 决定（`get_block_num()` 返回启动值；样例实践 `min(32, tiles)`） | `api/SIMD-API/operation/system_variables/get_block_num.md`；layernorm/add 样例 | 物理 AIV 核数以 Stage 3/5 从运行时/平台文件实测为准（本节不预填） |
| 数值安全域（fp16） | ±65504（IEEE half 最大有限值）——SPEC value_range 的依据 | IEEE-754（SPEC §12 自动决策 2/6 引用） | fp16 输出超界按 IEEE 上溢 inf（P2 不保护） |
| 索引位宽警示 | 输出总数 N(N-1)/2 在 N=65535 时 = 2,147,385,345 < 2^31−1（贴近）；**中间量 i*(2N-i-1) 最大 ≈ 8.59e9 > 2^31** | SPEC 公式 + 本报告核算 | idx 计算须用 64 位整型（Stage 3/4 实现 + golden 同步注意） |

---

## 8. 风险评估

### 8.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| （无 unsupported 阻断——全部公式步骤有可行 API 链） | — | — |
| 官方样例清单 #2 缺失（`pro_ops/matmul/test_matmul_8K_example.py`） | 清单与上游版本漂移：缓存内完整上游快照 `.devkit/.repo-pypto/python/tests/st/pypto_pro/frontend/matmul/` 仅有 4 个 perf 样例，无此文件（无大小写变体）——非装配失败，重新装配无法恢复 | **对 pdist 无实质影响**：缺失项为 matmul 入门教程样例，与本算子（纯 Vector 规约）无参考关系；pdist 所需的全部参考（vf_api ×2、element_wise ×1 等 12 项）完整在场。建议 orchestrator 核对清单版本（更新 `official_samples.md` 或向上游求证）；本报告不静默跳过，如实披露 |

### 8.2 注意事项

| 注意点 | 说明 |
|--------|------|
| R1 fp16 规约精度 | `vf.reduce_sum` 对 FP16 src 按 **FP16 精度**累加（文档表1）——违反 SPEC "中间 fp32" 契约。fp16 路径必须在规约前 `astype` widen 到 FP32（禁止 FP16 直接规约）；Stage 4 自验与 verifier 检查点 |
| R2 `vf.ln` 定义域（仅非 P0 一般 p） | ln src "必须为正数"；d=0（两行相同维度）在域外。P0 的 5 个 p 值全部走精确链（mul/sqrt/ne/max）不受影响；一般 p（P1/P2）Stage 3 需裁定 d=0 策略（特判/下限钳位）或声明该路径精度语义 |
| R3 输出散布写性能 | 每对输出 1 个标量，逐对 `pl.store` 会产生 O(N²) 次碎写；同一 i 的连续 j 段 idx 连续，天然可攒批（`vf.store_align` 到输出 tile 后整段 `pl.store`）。Stage 3 tile 设计决定写回效率（memory-bound 算子的关键） |
| R4 idx 整数溢出 | N=65535 时中间量 i*(2N−i−1) ≈ 8.59e9 超 int32；输出总数 2.147e9 贴近 2^31。索引算术须 int64；golden/test 同步注意（详见 §7 末行） |
| R5 寄存器压力 | 双行载入（2×n_regs）+ 幂中间量 + 累加器 + mask 并存可能超 32 寄存器上限（溢出 spill 8KB UB 劣化性能）。M 分块粒度与寄存器分配是 Stage 3 要点 |
| R6 tile 32B 对齐 | tile 物理列宽须满足 32B 倍数（fp32 8 列 / fp16 16 列倍数）；M=1 等极窄输入的 tile 声明需 Stage 3 处理（KB vec-row-reduce-broadcast 记录了窄轴退化与 over-declare 手法，已入 KB_SELECTION） |
| R7 大 N 内存约束 | N=65535 输出 fp32 ≈ 8.59GB（SPEC 已限 N≤65535，8GB 级为设备内存边界）；P0 最大 case（8192，fp16）输出 67MB 无压力。golden 构造与 Stage 5 采集时避免超大 N |
| R8 同一 tile 地址复用 | 若输入两行复用同一 UB tile（先后 load），auto_mutex 不覆盖同 pipe WAW——须 `pl.system.bar_mte2()` 或用独立 tile/双缓冲（load.md 高级用法节） |
| R9 `p` 属性传递 | p 为 host 标量（TilingData 或 kernel 参数）；`p<0` 参数错误在 host 侧预校验（SPEC §9），不进 kernel；p=inf 以 `float('inf')` 传入（SPEC 自动决策 4 的 `"inf"` 约定） |

---

## 9. 证据索引

> **全量资料索引**: `custom/pdist/PRO_MATERIAL_INDEX.md`

### 9.1 API 文档证据

| 信息 | 路径 |
|------|------|
| `vf.abs_sub`（一步 \|a−b\|）文档 | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/composite_computation/abs_sub.md` |
| `vf.sub` / `vf.mul` / `vf.muls` / `vf.add` / `vf.abs` 文档 | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/{sub,mul,muls,add,abs}.md` |
| `vf.sqrt` / `vf.exp` / `vf.ln` 文档（含 precision 开关与定义域） | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/{sqrt,exp,ln}.md` |
| `vf.reduce_sum` / `vf.reduce_max` 文档（累加精度/NaN 语义） | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/reduction/{reduce_sum,reduce_max}.md` |
| `vf.ne` / `vf.select` 文档 | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/comparison_and_selection/{ne,select}.md` |
| `vf.astype` 文档（FP16↔FP32、round/saturate、NaN/Inf 行为） | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/type_conversion/astype.md` |
| `vf.load_align` / `vf.store_align` / `vf.full` 文档 | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/{data_movement/load_align,data_movement/store_align,data_movement/full}.md` |
| `vf.update_mask` / `vf.create_mask` / `mask_reg` 文档 | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/{mask_operations/update_mask,mask_operations/create_mask,mask_reg}.md` |
| `vf.reg_tensor` 文档（VL/寄存器上限） | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/reg_tensor.md` |
| `pl.load` / `pl.store` 文档（offsets/对齐/bar_mte2/order） | `docs/pypto_pro/api/SIMD-API/operation/memory_data_movement/{load,store}.md` |
| `TileType` / `make_tile_group` / `set_validshape` 文档 | `docs/pypto_pro/api/SIMD-API/{basic_data_structures/TileType,operation/resource_management/make_tile_group,operation/memory_vector_computation/transpose_and_element_access/set_validshape}.md` |
| `get_block_idx` / `get_block_num` 文档 | `docs/pypto_pro/api/SIMD-API/operation/system_variables/{get_block_idx,get_block_num}.md` |
| `getval`/`setval`（tile/tensor 单元素读写）文档 | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/transpose_and_element_access/getval.md` |
| `Tensor`（shape 策略/DYNAMIC/一维合法）文档 | `docs/pypto_pro/api/SIMD-API/basic_data_structures/Tensor.md` |
| `section_vector` / `range` 控制流文档 | `docs/pypto_pro/api/SIMD-API/operation/controlflow/{section_vector_section_cube,range}.md` |

### 9.2 算子样例证据

| 信息 | 路径 |
|------|------|
| 行规约完整链（动态双轴+尾 lane+多核）——最高参考 | `pro_ops/vf_api/test_layernorm_tile_group_vf.py` |
| 两遍规约（reduce_max+reduce_sum） | `pro_ops/vf_api/test_softmax_tile_group_vf.py` |
| 纯 Vec 多核 + 双缓冲 + auto_mutex 最小骨架 | `pro_ops/element_wise/test_add.py` |
| TilingKey 分支模式 | `pro_ops/fa/test_fa_tilingkey_attn_mask.py` |
| 全动态维度 + TilingData 组织 | `pro_ops/matmul/test_matmul_perf_asw_4k_dn_move_offset_dynamic.py` |
| 其余官方样例（fa ×3、lightning_indexer、matmul ×2） | 见 `PRO_MATERIAL_INDEX.md` §B（12 项全部在缓存） |

### 9.3 指南与教程文档证据

| 信息 | 来源目录 | 路径 |
|------|----------|------|
| UB 248KB + 多核切分方法论 | tutorials | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/multi_core_partitioning_and_Tiling.md` |
| Reg 矢量计算编程（VF 范式） | tutorials | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/Reg_vector_computation.md` |
| 尾块处理（valid_shape/compact/pad） | tutorials | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/tail_block_handling.md` |
| TilingKey 机制 | tutorials | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/tiling_key.md` |
| TilingData 传递 | tutorials | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/TilingData.md` |
| Tile 向量计算 / Python 编程总览 | tutorials | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/{Tile_vector_computation,Python_programming_overview}.md` |
| 其余 33 个教程的适用性记录 | tutorials | 见 §5.1（全量 40 文档逐一评估） |

---

## 10. 结论

- **可行性**: **可行**。pdist 的全部公式步骤（含 P0 覆盖的 p ∈ {0, 1, 2, 0.5, inf} 五个分支）在当前目标版本（gitcode.com/cann/pypto @ d6e9b83 文档快照，Ascend 950PR/A5）均有文档化的可行 VF 调用链，**无 unsupported 阻断项**；P0 分支全部存在精确（非近似）API 链。
- **主要问题**: ① 官方样例清单与上游版本漂移（1 个 matmul 入门样例缺失，对 pdist 无参考关系，非阻断，已如实披露）；② 非 P0 一般 p 的 `vf.ln` 定义域（d=0）与 exp/ln 近似精度（P1/P2 范围，Stage 3 裁定）；③ 输出散布写回与寄存器/UB 预算是 Stage 3 设计的三个核心待裁定项（§6.1）。
