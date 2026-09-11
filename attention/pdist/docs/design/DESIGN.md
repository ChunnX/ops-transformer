# pdist 算子架构设计（DESIGN.md）

- **op_name**: pdist
- **设计时间**: 2026-09-01
- **基于 SPEC**: `custom/pdist/SPEC.md`
- **基于 EXPLORE_REPORT**: `custom/pdist/EXPLORE_REPORT.md`
- **目标平台**: Ascend 950PR_9579（A5，CANN 9.2.0，`TILE_FWK_DEVICE_ID=0`）
- **平台常量来源**: `/usr/local/Ascend/cann-9.2.0-beta.1/python/site-packages/pypto/lib/configs/950PR_957x.ini`（`ub_size=253952`、`vector_core_cnt=56`、`ubblock_size=32`）；文档快照 gitcode.com/cann/pypto @ d6e9b83

---

## Knowledge Bindings

> 完整、机器可读的 requirement 清单见 [`DESIGN_BINDINGS.json`](DESIGN_BINDINGS.json)。
> 本文只记录 R0–R8 的实际设计决策；每条活动 requirement 的 `planned_location` 必须准确指向
> 对应决策及最终 `test_pdist.py` file/symbol，不复制 Binding 分组或 requirement 表格。

---

## §0 Module 划分（R0 输出）

> 依据 [Module划分](../references/module_partitioning.md)，先按数据依赖确定Module边界，再按Section边界继续拆分。R0记录每个Module的Section归属，具体Section代码结构在§4确定。

### 维度契约

- **kernel 接收与处理维度**：x 为 rank-2 `[N, M]`（`pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], DT_FP32/DT_FP16]`），y 为 rank-1 `[N*(N-1)//2]`（`pl.Tensor[[pl.DYNAMIC], 同 dtype]`）。SPEC 仅定义 2D 输入（shape_constraints 第 1 条），kernel 按原始 rank/stride 访问（ND 行主序），无需多维适配；输出一维 `pl.store` 的 `offsets=[idx]` 合法（`store.md`：offsets 长度等于 tensor 维数）。

| SPEC 输入 shape | kernel 接收 | kernel 内索引/stride 映射 | 输出形态 |
|-----------------|-------------|---------------------------|----------|
| 2D `[N, M]` fp32/fp16 | 原始 `[N, M]`（DYNAMIC 双轴） | `pl.load(tile, x, [row, col])` 按元素绝对坐标；GM 行主序 ND | 1D `[N(N-1)/2]`，`pl.store(y, tile, [idx])` |

### Wrapper 边界外操作

空

### 划分依据

**单 Module**。理由（依据 module_partitioning.md 的数据依赖判据）：

1. **无跨遍历的依赖边界**：每个行对 (i, j) 的距离是独立的一次沿 M 规约。M 分块的部分和由同一层 M-块循环内的 acc 状态即时累加（"A 处理完当前 Tile 后，B 能在同一层循环内立即消费"），不存在"基于最终结果重新遍历完整 M 轴"的第二遍（对比 Softmax 的三遍结构）。p 三分支（0 / (0,∞) / ∞）是与逐对无关的全局常量分支（SPEC §3 算法 1.2 条），选择子由 TilingKey 编译期折叠，不构成 Module 边界。
2. **无 Section 边界**：全部计算为逐元素算术 + 沿 M 归约 + 数据搬运（EXPLORE §1.2：纯 Vector VF 寄存器路径，无 matmul/收缩），整个数据流 GM→UB→Reg→UB→GM 位于单一 `section_vector`。无 Cube↔Vector 数据交接。
3. **UB 容量不构成拆分理由**：单任务（64 对 × M 分块）的全部并存 tile 共 ~131KB（§3，两实例 52.6%/52.9%），远小于 248KB。

### Module 列表

| Module | 目的 | 输入依赖 | 输出 | Section |
|-------|------|---------|------|---------|
| Module 1 | 对每个 64-pair 任务块：载入锚行与行块的 M 分块 → 转置布局 → 逐 m 计算 p-范数距离 → 打包写回 y 的连续段 | x（GM）、p（host 标量→TilingKey/TilingData） | y（GM，上三角行优先） | section_vector |

### 归约轴容量结论

- **归约轴是否可能超单 tile**：是。M ∈ [1, 65535]（SPEC 动态轴），fp32/fp16 实例统一 M_BLK=128，M > 128 时必须多块遍历（P0 case `func_p0_large_m` M=8192 强制 64 块）。
- **多Tile归约方案**：**M 分块 + 跨块部分和状态**。遍历范围：每个任务块内 `for mblk in range(0, ceil(M/M_BLK))`；状态：64 个 pair 的部分和 acc（fp32，寄存器内累加，跨 M-块经 acc ping-pong tile 暂存）；依据：M 维分块只改变求和顺序的浮点舍入、不改语义（SPEC §3 要点），acc 状态在同一任务循环内存活、任务结束即消费（inv-pow）后失效。

### Module 级数据流

```
GM x[N,M] ──pl.load(anchor)──→ anchor_tile[1,M_BLK] ──┐
GM x[N,M] ──pl.load(rows)───→ rows_tile[64,M_BLK] ──pl.transpose──→ trans_tile[M_BLK,64]
                                                        │            │
        acc_tile[1,64] (ping-pong, fp32) ───────────────┤            ↓
                                                        └→ @pl.vector_function:
                                                             per m: BRC(anchor,m) ⊕ load_align(trans,m*64)
                                                             → abs_sub → p-branch → add/max acc
                                                        ↓
                                            finalize: inv-pow → out32[1,64]
        (fp16) pl.cast(out16 ← out32) ──pl.store(y[idx])──→ GM y[N(N-1)/2]
```

---

## §1 API 映射（R1 输出）

> 按 `.opencode/references/performance-constraints.md` 的"强制 2：Vector 默认使用 VF"填写。每个 Vector 步骤冻结唯一实现。已选 KB（`patterns/buffer-reuse-lifetime.md` 等）均不要求任何具体数值步骤使用 tile-op `pl.*`，故全部数值步骤为 `vf`（default_vf）。`pl.load`/`pl.store`/`pl.transpose`/`pl.cast` 属数据搬运与布局转换（强制 2 明示"不受此限"）。

### 关键结构决策：pair-lane 转置布局

64 个行对 (i, j0..j0+63) 的距离计算组织为**一个 fp32 寄存器的 64 个 lane**（lane k ↔ pair k ↔ 行 j0+k）：

- `pl.load(rows_tile[64, M_BLK], x, [j0, m0])`（ND 行主序，GM 行连续搬运）→ `pl.transpose(trans_tile[M_BLK, 64], rows_tile)`（V 流水，950 支持，源/目标主维 × dtype 字节 32B 对齐满足）→ `vf.load_align(trans_tile, m*64)` 一次装载"同一 m 下 64 个 pair 的元素"。
- 锚行值 x[i, m] 用 `vf.load_align(anchor_tile, m, dist=pl.LoadDist.BRC_B32)` 单指令广播到 64 lanes（对齐约束 4B，元素粒度）。
- 归约方向沿 m（M 维）：每 m 一条 `add`（sum 分支）或 `max`（inf 分支）更新 64 个 pair 的并行状态——**归约不使用 `vf.reduce_sum`**（其归约方向是寄存器内 lanes，与本算子"沿 M 归约、pair 并行"的方向正交）。
- 任务尾 inv-pow 一条指令处理 64 对（lane 级并行），`store_align` 一次写 64 个连续距离，`pl.store` 一次写回 GM 连续段。

选择依据 vs 备选（逐 pair 行主序 + 每对独立 `reduce_sum`）：两者每元素 V 指令数同为 ~5/64，但 pair-lane 布局的 acc 仅 1~2 个寄存器（备选需 64 个 acc 寄存器，超 32 上限）、inv-pow/store 为 lane 级并行（备选逐对标量 32 条）、p=inf 的 max 链最短（3 条/m）。转置开销为每 [64, M_BLK] tile 一次 V 流水转置（~1 条/m 等效），被上述节省抵消。

### vector_selection（每个 Vector 步骤冻结唯一实现）

> 已选 KB（`patterns/buffer-reuse-lifetime.md` 及全部 constraints）均不含"当前具体步骤必须使用 tile-op `pl.*`"的模板要求，故全部**数值计算**步骤为 `vf`（default_vf）；`pl.load`/`pl.store`/`pl.transpose` 为数据搬运（强制 2 明示不受此限）；两处 `pl.cast` 为 dtype 转换（非数值计算，wrapper-boundary KB "What to do instead" 表明确许可 tile 级 cast；选 tile 级的数值理由见 §1 fp16 差异节——寄存器级 widen/narrow 的 layout 交错陷阱）。

| step | implementation | api_sequence | decision_reason | kb_template_evidence | target_version | applicable_conditions |
|------|----------------|--------------|-----------------|----------------------|---------------|----------------------|
| fp16 入口 widen（转置后/锚） | tile_op | `pl.cast(trans32, trans16, mode=CAST_NONE)`; `pl.cast(anchor32, anchor16, mode=CAST_NONE)` | dtype 转换非数值计算；寄存器级 `vf.astype` widen 的 ZERO/ONE 为全局偶/奇交错 lanes（precision.md 实测），连续行布局下拆出行混合，需 interleave 重组 | n/a（非数值步骤；cast.md/cast 许可 + precision.md 交错知识） | A5 / CANN 9.2.0 / d6e9b83 | 仅 fp16 实例；CAST_NONE（16→32 扩展无舍入） |
| 锚值广播（每 m） | vf | `vf.load_align(anchor_tile, m, dist=pl.LoadDist.BRC_B32)` | default_vf；单指令广播 x[i,m] 到 64 lanes（对齐 4B） | n/a | 同上 | 全分支/全 dtype（fp16 读 anchor32） |
| pair 行载入寄存器（每 m） | vf | `vf.load_align(trans_tile, m*64)` | default_vf；一次装载同一 m 的 64 个 pair | n/a | 同上 | 全分支/全 dtype（fp16 读 trans32） |
| 逐维绝对差 | vf | `vf.abs_sub(bcast, rv, preg)` | default_vf；文档化复合指令（FP16/FP32 支持，本设计在 fp32 域） | n/a | 同上 | 全分支 |
| 幂/比较预处理（p 分支） | vf | PBranch 2: `vf.mul(d,d,preg)`；4: `vf.sqrt`；0: `vf.ne`+`vf.full`+`vf.select`；1: 无（直通）；5: `vf.ln`+`vf.muls`+`vf.exp`+`vf.select` | default_vf；P0 五个 p 值全精确链（EXPLORE §2 结论），一般 p 走 exp/ln 近似（P1/P2） | n/a | 同上 | PBranch 由 TilingKey 编译期折叠，每变体唯一链 |
| 沿 M 归约（每 m） | vf | PBranch 0/2/3/4/5: `vf.add(acc, dp, preg)`；1: `vf.max(acc, d, preg)` | default_vf；pair-lane 布局下沿 M 归约为逐 m 的 add/max 链（非寄存器内 lanes 归约，不用 vf.reduce_*） | n/a | 同上 | acc 状态跨 M 块 ping-pong |
| NaN 防护累积（PBranch 1/5） | vf | `vf.ne(d,d,preg)` → `vf.select(ones,zeros,nm)` → `vf.add(nan_cnt,nf,preg)` | default_vf；`vf.max` NaN 语义未文档化，reg 级链全文档化（mask 级 `vf.or_` 三参签名未实证，弃用） | n/a | 同上 | 仅 PBranch 1/5；nan_cnt 驻 acc tile 行 1 |
| inv-pow（任务尾） | vf | 2: `vf.sqrt(acc)`；4: `vf.mul(acc,acc)`；0/1/3: 恒等/NaN 覆写 `vf.select`；5: `vf.ln`+`vf.muls`+`vf.exp`+`vf.select` | default_vf；一条指令处理 64 pair（lane 级并行） | n/a | 同上 | ln 前钳位（s=0→1）+贡献归零+NaN 覆写 |
| pair 尾裁剪（任务尾） | vf | `vf.update_mask(VP, dtype=DT_FP32)` | default_vf；掩码控制 store_align 只写 VP 个 | n/a | 同上 | VP<64 的尾块任务 |
| 结果写 UB（任务尾） | vf | `vf.store_align(out32, y, mreg)` | default_vf；一次写 64 连续距离 | n/a | 同上 | 全 dtype（out32 恒 fp32） |
| fp16 出口窄化 | tile_op | `pl.cast(out16, out32, mode=CAST_RINT)` | dtype 转换非数值计算；寄存器级 narrow 结果位于偶数交错 lanes，store NORM 会交错错位（precision.md），tile 级无 lane 陷阱；NaN 语义为 Stage 4 验证点 + vf.astype 非饱和 fallback | n/a（非数值步骤） | 同上 | 仅 fp16 实例 |
| acc 跨块中转 | vf | `vf.load_align(acc,0/64)` / `vf.store_align(acc,...,0/64)` / `vf.mem_bar(VST_VLD)` | default_vf；VF 寄存器跨调用不存续，经 resident ping-pong tile 中转 | n/a | 同上 | 每 M 块；mem_bar 保证 VST→VLD 顺序 |

### Module 1 API 调用序列（fp32 实例，PBranch=p=2 主链）

```
Module 1 — pdist pair-lane 距离计算（单 section_vector）:
  # 每核定位起始任务块 (i, jseg)（§5），任务循环 for t in range(0, hi-lo):
  #   任务 (i, jseg)：j0 = i+1+jseg*64；VP = min(64, N-1-i-jseg*64)；base = i*(2N-i-1)//2 + jseg*64 (int64)
  for mblk in pl.range(0, ceil(M/M_BLK)):            # M_BLK=128（fp32/fp16 统一）
      m0 = mblk*M_BLK; vc = pl.min(M_BLK, M - m0)
      rows  = rows_group.next();  pl.set_validshape(rows,  [VP, vc])
      anch  = anchor_group.next();pl.set_validshape(anch,  [1, vc])
      trans = trans_group.next(); pl.set_validshape(trans, [vc, VP])
      pl.load(rows, x, [j0, m0])                      # MTE2, GM 2D 块 (行 stride=M)
      pl.load(anch, x, [i, m0])                       # MTE2, 单行
      pl.transpose(trans, rows)                        # V 流水: [64,M_BLK] → [M_BLK,64]
      (fp16 实例追加, 在 transpose 后、VF 前)
        trans32 = trans32_group.next(); anchor32 = anchor32_group.next()
        pl.set_validshape(trans32, [vc, VP]); pl.set_validshape(anchor32, [1, vc])
        pl.cast(trans32, trans,  mode=pl.RoundMode.CAST_NONE)   # fp16→fp32 widen (tile 级)
        pl.cast(anchor32, anch, mode=pl.RoundMode.CAST_NONE)
        # fp16 实例的 rows/trans/anchor 为 FP16 tile，VF 消费 trans32/anchor32 (FP32)
      pdist_accum_vf(trans32|trans, anchor32|anch, acc_group[mblk%2], acc_group[(mblk+1)%2], vc, ...)
        # @pl.vector_function 内（全 fp32 域，fp32/fp16 实例同构）:
        #   preg = vf.create_mask(ALL, DT_FP32)
        #   acc  = vf.load_align(acc_in, 0)      # 上一 M-块部分和（首块: vf.full(0.0, preg) 初始化）
        #   (PBranch 1/5) nan_cnt = vf.load_align(acc_in, 64)   # 行 1 状态（首块 full(0)）
        #   for m in pl.range(0, vc):
        #       bcast = vf.load_align(anch, m, dist=BRC_B32)   # x[i, m0+m] → 64 lanes
        #       rv    = vf.load_align(trans, m*64)             # lane k = x[j0+k, m0+m]
        #       d     = vf.abs_sub(bcast, rv, preg)            # |x_i - x_jk|
        #       [PBranch] dp = vf.mul(d, d, preg)              # p=2 幂（其余分支见下表）
        #       acc = vf.add(acc, dp, preg)                    # 64 pair 并行累加
        #   vf.store_align(acc_out, acc, preg, 0)   (+ nan_cnt → 偏移 64)
        #   vf.mem_bar(mode=pl.MemBarMode.VST_VLD)             # acc 跨块 VST→VLD 顺序
  out = out32_group.next()
  pdist_finalize_vf(acc_group[n_mblk % 2], out, VP, ...)        # 最后写入槽 = n_mblk%2
        #   acc = vf.load_align(acc_last); y = vf.sqrt(acc, preg)   # p=2 inv-pow
        #   mreg = vf.update_mask(VP, dtype=DT_FP32)                # pair 尾裁剪
        #   vf.store_align(out, y, mreg)
  (fp16 实例追加) out16 = out16_group.next(); pl.cast(out16, out, mode=pl.RoundMode.CAST_RINT)
  pl.store(y, out, [base])                            # MTE3, 一维 offsets, 覆盖写
```

API 文档来源：`load.md`/`store.md`（memory_data_movement，offsets 绝对元素坐标）、`transpose.md`（V 流水，950PR 支持）、`load_align.md`（BRC_B32 广播，对齐 4B；NORM 32B）、`store_align.md`（NORM 32B；preg 掩码控制写入）、`abs_sub.md`（FP16/FP32）、`reg_tensor.md`（FP32 VL=64 / FP16 VL=128，32 寄存器上限）、`cast.md`（tile 级 FP32→FP16，shape/valid 一致）、`astype.md`（widen/narrow 参数表）、`update_mask.md`、`full.md`、`ne.md`/`select.md`、`max.md`、`ln.md`/`exp.md`（general 分支）、`mem_bar`（store_align.md 示例 L399 同款 VST_VLD）。

### p 分支实现表（TilingKey `PBranch` 编译期折叠，kernel 内 `if` 死代码消除）

| PBranch | p 值 | 幂步骤（每 m，替换 `mul`） | 归约 | inv-pow（finalize，每任务 1 条） | 精度 |
|---|---|---|---|---|---|
| 0 | p=0（Hamming） | `m= vf.ne(d, 0.0, preg)` → `dp = vf.select(vf.full(1.0,preg), vf.full(0.0,preg), m)` | `vf.add(acc, dp)` | 恒等（acc 即计数，≤ M < 2²⁴ fp32 精确） | 精确；NaN 计为不等（`ne(NaN,0)=1`，golden 语义） |
| 1 | p=+∞（Chebyshev） | 无幂，`d` 直通 | `acc = vf.max(acc, d, preg)`；**NaN 防护（reg 级，全文档化链）**：`nm = vf.ne(d, d, preg)`（NaN lane → mask 1）→ `nf = vf.select(ones, zeros, nm)`（0/1 reg）→ `nan_cnt = vf.add(nan_cnt, nf, preg)` | `bad = vf.ne(nan_cnt, 0.0, preg)` → `y = vf.select(vf.full(NaN,preg), acc, bad)` | 精确；`vf.max` 的 NaN 语义未文档化（max.md 仅记录 ±0），防护链独立检测并显式覆写 NaN（曾评估 mask 级 `vf.or_` 累积，其 mask 源三参签名未实证，弃用） |
| 2 | p=2 | `dp = vf.mul(d, d, preg)` | `vf.add` | `y = vf.sqrt(acc, preg)` | 精确（无超越函数） |
| 3 | p=1 | 恒等 | `vf.add` | 恒等 | 精确 |
| 4 | p=0.5 | `dp = vf.sqrt(d, preg)` | `vf.add` | `y = vf.mul(acc, acc, preg)` | 精确；`sqrt(d)` 输入非负天然满足（d=\|·\|≥0） |
| 5 | 其他 0<p<∞ | `safe = vf.select(nz_mask, d, vf.full(1.0,preg))` → `dp = vf.exp(vf.muls(vf.ln(safe, preg), p_val), preg)` → `dp = vf.select(nz_mask, dp, zero)`；NaN 维经 PBranch 1 同款 nan_cnt 链累积并在 finalize 覆写 | `vf.add` | `s_safe = select(nz, acc, 1.0)` → `y = exp(muls(ln(s_safe), inv_p))` → `y = select(nz, y, 0.0)`；NaN 覆写同 PBranch 1 | 近似（exp/ln ~2-3 ulp ≪ 容差 1e-3）；`ln` 定义域 d>0（ln.md"必须为正数"），d=0/s=0 经 select 钳位到 1 再归零贡献（0^p = 0，p>0）；仅 P1/P2 路径，P0 五个 p 值全走精确链 |

`p_val`/`inv_p` 经 TilingData `float` 字段传入（FP32）；`nz_mask = vf.ne(d, 0.0, preg)`（复用 branch 0 的比较）。**跨 M-块状态**：PBranch 1/5 的 `nan_cnt` 与 `acc` 同为 64-lane fp32 状态，共同驻留 `acc_group` tile（`[2, 64]`：行 0 = acc、行 1 = nan_cnt；`vf.load_align/store_align` 偏移 0/64 读写，与 acc 同 ping-pong 同初始化）。

### fp16 实例差异（同一源码、`DT_FP16` 签名的第二个 JIT 实例）

- **数值链与 fp32 实例完全同构**：M_BLK=128（与 fp32 统一）、VF 内全部计算在 fp32 域（`preg` 统一 `DT_FP32`）、m 循环步进 1——**不在寄存器域做任何 fp16↔fp32 转换**。
- **widen 位置（precision.md "widen before the operation that overflows"）**：`pl.transpose` 之后、VF 消费之前，用 **tile 级 `pl.cast`（`mode=pl.RoundMode.CAST_NONE`，扩展无舍入）** 完成两处 fp16→fp32：`pl.cast(trans32, trans16)`（[128,64]）与 `pl.cast(anchor32, anchor16)`（[1,128]）。此后 abs_sub/pow/累加/inv-pow 全链 fp32。
- **为什么不用寄存器级 `vf.astype` widen**：fp16 reg 128 lanes 一次 astype 仅产 64 fp32 且 `layout=ZERO/ONE` 取的是**全局偶/奇交错 lanes**（precision.md 设备实测纠正；astype.md L21）——trans 的连续两行（lane 0-63 = 行 m、64-127 = 行 m+1）经 ZERO 拆出的是**两行的偶位混合**而非单行，需要 DINTLV/interleave 家族额外重组；tile 级 `pl.cast` 要求仅"物理 shape/valid 一致"（cast.md），无 lane 陷阱。
- **出口窄化**：finalize 将 y（fp32 reg）`store_align` 到 `out32`（fp32 tile）→ **tile 级 `pl.cast(out16, out32, mode=CAST_RINT)`** → `pl.store(y, out16, [base])`。理由同上（寄存器级 narrow 的结果位于 128-lane 目的寄存器的偶数交错 lanes，`store_align NORM` 会写出交错数据）。
- **NaN 语义验证义务**：`pl.cast` 文档未声明非饱和 NaN→NaN（对照 `vf.astype` 表 6 明确浮点转浮点非饱和 NaN→NaN、±inf→±inf）。入口 widen 为 16→32 位扩展（位宽扩展、`CAST_NONE` 无舍入，语义上无损，风险低）；出口 32→16 窄化为 Stage 4 必验点——以 fp16 NaN 输入的语义 case 验证；若 `pl.cast` 破坏 NaN，fallback 为 `vf.astype(..., DT_FP16, saturate=OFF, round_mode=CAST_RINT)` + interleave 重组（KB vec-mask-width 的 interleave 家族）。

### 超越函数数值安全边界

| API | 输入理论范围 | 目标 dtype 上限 | 是否溢出 | 防护措施 |
|-----|-------------|----------------|----------|----------|
| `vf.sqrt`（PBranch 2/4 幂与 inv-pow） | d=\|·\|≥0；s=Σd^p≥0（mul 非负累加） | fp32 ≈ 3.4e38 | 否（域内） | 无需防护；sqrt(NaN)=NaN（IEEE）满足逐对传播 |
| `vf.ln`（仅 PBranch 5） | 定义域 src>0（ln.md 原文"必须为正数"）；d=0（两行同维）、s=0（全同行对）在域外 | fp32 | 否（钳位后 ln(1)=0） | `select(ne(d,0), d, 1.0)` 钳位 + 贡献 `select(ne(d,0), dp, 0.0)` 归零（数学等价 0^p=0, p>0）；NaN 另行防护链 |
| `vf.exp`（仅 PBranch 5） | p·ln d ≤ p·ln(2·65504)≈p·11.8；SPEC value_range ±65504 且大 p 上溢不保护（P2，与 PyTorch 对齐） | fp32 | 大 p 可能 +inf | 按 SPEC §9 不做保护（IEEE 上溢 inf） |
| fp16 幂中间量 \|d\|^p | widen 后全 fp32 计算，中间量 ≤ (2·65504)^p 上限由 fp32 承载 | fp32 ≈ 3.4e38 | p ≤ ~24 时有限；更大按 IEEE inf | **widen 在产生量级增长的 mul/pow 之前**（abs_sub 前），非归约前（precision.md 顺序要求）；fp16 通路无任何 fp16 精度中间量 |
| fp16 出口窄化 | y ∈ [0, 65504]（合同值域）；超界 IEEE inf（P2 不保护） | fp16 65504 | 可能（P2 范围） | `CAST_RINT` 舍入；NaN/Inf 传播见上文验证义务 |

### TilingData / TilingKey 合同

```python
@dataclass
class PdistTiling:
    N: int        # INDEX → int64_t（idx 算术安全：N=65535 时 i*(2N-i-1)≈8.59e9 > int32）
    M: int
    B: int        # 总 64-pair 任务块数（host 闭式，见 §5）
    p_val: float  # FP32，PBranch 5 使用
    inv_p: float  # FP32，PBranch 5 使用

class PdistTilingKey:
    PBranch = TilingKeyField(bits=3, values=[0, 1, 2, 3, 4, 5])   # §1 p 分支表
    def is_valid(self, key): return True
```

TilingKey 适用性：p 三分支 + 幂策略是**有限枚举**（6 值）且**显著改变生成代码**（每 m 指令序列不同），符合 `tiling_key.md`"应仅描述有限且会影响代码生成的模式"；编译期折叠使每变体内层 m 循环无分支（若用运行时 if，分支须置于任务循环外、m 循环外，代码膨胀 6×——两机制均文档化，选 TilingKey）。dtype 分支（fp32/fp16）由两个 JIT 实例的签名区分，不占 TilingKey 位。

---

## §2 Tile 规划（R2 输出）

### 2.1 关键常量定义

```python
# ── 算子关键常量（coder 必须逐字使用）──
TP      = 64            # 每任务块的 pair 数 = fp32 寄存器 lane 数（lane k ↔ 行 j0+k）
M_BLK   = 128           # M 分块宽度，fp32/fp16 实例统一（列 128×4B=512B / fp16 tile 128×2B=256B，均 32B 对齐 ✓）
LANES   = 64            # fp32 VL；fp16 数据经 tile 级 pl.cast 后全部在 fp32 域消费（寄存器域无 fp16）
VLMIN   = 8             # 32B 对齐的最小 fp32 列数（M_BLK 远大于此，M=1 尾块由 valid_shape 处理）
```

> M 尾块（M % M_BLK ≠ 0）由 `set_validshape` + VF m 循环上界 `vc` 处理；**M 维不产生残 lane**（m 按整行推进，每行恰 64 pair lanes），无需 per-lane 掩码；pair 尾（VP < 64）由 finalize 的 `update_mask(VP)` 裁剪。

### Tile 属性表（fp32 实例；fp16 实例差异见表后注）

| 用途 | 变量名 | shape | dtype | 内存空间 | layout | 大小/槽 | 槽数 | valid_shape | 生命周期 |
|------|--------|-------|-------|---------|--------|---------|------|-------------|----------|
| 行块暂存（64 pair 行 × M 块） | `rows_group` | `[64, 128]` | FP32 | UB(Vec) | `—` | 32768B | 2 | `[-1,-1]`（逐块 `[VP, vc]`） | rotating（M 块轮转） |
| 转置布局（M 块行 × 64 pair 列） | `trans_group` | `[128, 64]` | FP32 | UB(Vec) | `—` | 32768B | 2 | `[-1,-1]`（逐块 `[vc, VP]`） | rotating |
| 锚行暂存 | `anchor_group` | `[1, 128]` | FP32 | UB(Vec) | `—` | 512B | 2 | `[-1,-1]`（`[1, vc]`） | rotating |
| 跨 M 块归约状态（行 0 = acc 部分和；行 1 = PBranch 1/5 的 nan_cnt） | `acc_group` | `[2, 64]` | FP32 | UB(Vec) | `—` | 512B | 2 | `—`（固定满） | **resident**（ping-pong：`group[mblk%2]` 读 / `group[(mblk+1)%2]` 写） |
| 输出攒批（fp32 距离段） | `out32_group` | `[1, 64]` | FP32 | UB(Vec) | `—` | 256B | 2 | `—` | rotating（任务轮转） |

fp16 实例：`rows_group [64,128]` FP16 16384B×2、`trans16`（转置布局）`[128,64]` FP16 16384B×2、`anchor16 [1,128]` FP16 256B×2 为 GM 入口 tile（原始 dtype）；追加 `trans32_group [128,64]` FP32 32768B×2 与 `anchor32_group [1,128]` FP32 512B×2（tile 级 `pl.cast(CAST_NONE)` widen 的目的 tile，VF 消费）；`acc_group [2,64]` FP32 与 `out32_group [1,64]` FP32 同 fp32 实例（中间精度契约）；`out16_group [1,64]` FP16 128B×2（出口窄化段）。

### TileGroup槽位访问表

| TileGroup | depth | 每个Tile的mutex ID | 访问方式 | 槽位表达式 | 同步及有界/游标说明 |
|---|---:|---|---|---|---|
| `rows_group` | 2 | `[0]` / `[1]` | `next()` | —（固定顺序轮转） | MTE2 写 ↔ V（transpose）读跨 pipe 依赖由 auto_mutex 管理；每 M 块 `next()` 一次 |
| `trans_group`（fp32）/ `trans16`（fp16） | 2 | `[2]` / `[3]` | `next()` | — | V（transpose）写 ↔ V（pl.cast，fp16）/VF（fp32）读；auto_mutex |
| `anchor_group`（fp32）/ `anchor16`（fp16） | 2 | `[4]` / `[5]` | `next()` | — | MTE2 写 ↔ V（cast，fp16）/VF BRC（fp32）读；auto_mutex |
| （fp16）`trans32_group` | 2 | `[8]` / `[9]` | `next()` | — | V（pl.cast）写 ↔ VF load_align 读；auto_mutex |
| （fp16）`anchor32_group` | 2 | `[10]` / `[11]` | `next()` | — | V（pl.cast）写 ↔ VF BRC 读；auto_mutex |
| `acc_group` | 2 | **不配置**（`depth=2`） | `group[i]` 显式下标 | `i = mblk % 2`（读）/ `(mblk+1) % 2`（写） | 纯 V 域（VF load_align/store_align，偏移 0=acc、64=nan_cnt）无跨 pipe 依赖，mutex 机制不适用；VST→VLD 顺序由 VF 内 `vf.mem_bar(VST_VLD)` 保证（store_align.md 示例 L399 模式）；下标 ∈ [0,2) 恒成立（mblk ≥ 0） |
| `out32_group` | 2 | `[6]` / `[7]` | `next()` | — | VF store_align 写 ↔ MTE3（pl.store）读；auto_mutex |
| `out16_group`（fp16） | 2 | `[12]` / `[13]` | `next()` | — | V（pl.cast）写 ↔ MTE3 读；auto_mutex |

- slot 计数法则核算（KB buffer-reuse-lifetime）：overlapped beats × simultaneously live roles——每 tile 的角色数均为 1，M-块循环内在飞 beat 数 1（load 块 b+1 与 VF 块 b 重叠由双缓冲承载：2 ≥ 1×1，参照官方 layernorm/add 样例的双缓冲配置）；`acc` 状态 live roles = 1（acc 与 nan_cnt 为同一逻辑状态的两行，非独立角色），2 slots 为读写分离的保守配置。同时轮转槽合计 8（fp32）/ 16（fp16）≤ 32 上限；mutex ID 8 个（fp32，[0,7]）/ 14 个（fp16，[0,13]）互异且 ∈ [0,31] ✓。
- 跨 MTE2 同地址复用：各 tile 双缓冲轮转不同地址，不触发 load.md 的 `bar_mte2` 场景（"若目标地址不同…则不适用本节规则"）。

### tile_dims stride 注意事项

不涉及（未使用 `load_tile`/`tile_dims`；`pl.load` 按绝对元素坐标 `[row, col]` 直接定位）。

---

## §3 片上空间布局（R3 输出）

> 纯 Vector 算子，仅 UB(Vec) 一节。地址从 `0x00000` 起算，32B 对齐（`ubblock_size=32`）。

### 片上地址映射表（UB(Vec)）

**fp32 实例**（M_BLK=128）：

| 内存空间 | 用途 | 变量名 | shape | dtype | layout | 地址 | 大小/槽 | slots | lifetime | end |
|---------|------|--------|-------|-------|--------|------|---------|-------|----------|-----|
| UB(Vec) | 行块暂存 | `rows_group` | `[64,128]` | FP32 | `—` | `0x00000` | 32768 | 2 | rotating | 0x10000 |
| UB(Vec) | 转置布局 | `trans_group` | `[128,64]` | FP32 | `—` | `0x10000` | 32768 | 2 | rotating | 0x20000 |
| UB(Vec) | 锚行 | `anchor_group` | `[1,128]` | FP32 | `—` | `0x20000` | 512 | 2 | rotating | 0x20400 |
| UB(Vec) | 归约状态（acc+nan_cnt） | `acc_group` | `[2,64]` | FP32 | `—` | `0x20400` | 512 | 2 | resident | 0x20800 |
| UB(Vec) | 输出段 | `out32_group` | `[1,64]` | FP32 | `—` | `0x20800` | 256 | 2 | rotating | 0x20A00 |

**fp16 实例**（M_BLK=128，GM 入口 tile 为 FP16 + tile 级 cast 的 FP32 目的 tile）：

| 内存空间 | 用途 | 变量名 | shape | dtype | layout | 地址 | 大小/槽 | slots | lifetime | end |
|---------|------|--------|-------|-------|--------|------|---------|-------|----------|-----|
| UB(Vec) | 行块暂存（GM 入口） | `rows16_group` | `[64,128]` | FP16 | `—` | `0x00000` | 16384 | 2 | rotating | 0x08000 |
| UB(Vec) | 转置布局（GM 入口） | `trans16_group` | `[128,64]` | FP16 | `—` | `0x08000` | 16384 | 2 | rotating | 0x10000 |
| UB(Vec) | 转置布局（widen 后，VF 消费） | `trans32_group` | `[128,64]` | FP32 | `—` | `0x10000` | 32768 | 2 | rotating | 0x20000 |
| UB(Vec) | 锚行（GM 入口） | `anchor16_group` | `[1,128]` | FP16 | `—` | `0x20000` | 256 | 2 | rotating | 0x20200 |
| UB(Vec) | 锚行（widen 后） | `anchor32_group` | `[1,128]` | FP32 | `—` | `0x20200` | 512 | 2 | rotating | 0x20600 |
| UB(Vec) | 归约状态（acc+nan_cnt） | `acc_group` | `[2,64]` | FP32 | `—` | `0x20600` | 512 | 2 | resident | 0x20A00 |
| UB(Vec) | 输出段（fp32） | `out32_group` | `[1,64]` | FP32 | `—` | `0x20A00` | 256 | 2 | rotating | 0x20C00 |
| UB(Vec) | 输出段（fp16） | `out16_group` | `[1,64]` | FP16 | `—` | `0x20C00` | 128 | 2 | rotating | 0x20D00 |

**各空间总用量**：

- fp32 实例 UB(Vec): 0x20A00 = 133632B / 253952B（`950PR_957x.ini ub_size`）= **52.6%**
- fp16 实例 UB(Vec): 0x20D00 = 134400B / 253952B = **52.9%**
- 其余空间（Mat/L1/Left/L0A/Right/L0B/Acc/L0C）：无 tile，不涉及。

**GM 侧越界核算（linear extent，KB memory-layout）**：rows 的 GM 搬运 extent = `(VP−1)·M + vc`（不是 `VP×vc`），因 `j0+VP−1 ≤ N−1` 且 `m0+vc ≤ M`，不越界；anchor extent = `vc`；输出 `base+VP ≤ N(N−1)/2`（上三角 idx 双射）。所有 store 为覆盖写（`AtomicNone`），无累加语义（SPEC kernel 契约"一次性覆盖写"）。

### 分配方式选择

> **性能强制**：所有需要 buffer 切换/轮转的 tile 一律用 `make_tile_group` + `auto_mutex`；`make_tile` 仅限单次使用 scratch tile。

- **方案**: 全部 tile 用 `make_tile_group` + `@pl.jit(auto_mutex=True)`。fp32 实例的 rows/trans/anchor/out32 与 fp16 实例的 rows16/trans16/anchor16/trans32/anchor32/out32/out16 均配非空 `mutex_ids`：跨 pipe 依赖（MTE2 写 rows/anchor ↔ V 读；VF 写 out32 ↔ MTE3 读）由 auto_mutex 按 slot 元数据管理；fp16 的 trans16→trans32（transpose(V)→cast(V)）与 anchor16 类似链路中同 pipe 顺序部分，mutex 非严格必需，为 rotating 组态统一仍配置（auto_mutex 的冗余同步正确无害）。`acc_group` 不配 mutex（`depth=2`），因其只被 VF 的 load_align/store_align 访问（纯 V 域，无跨 pipe 依赖——mutex 机制覆盖的是 pipe 间依赖，不覆盖 V 域内 VST→VLD 顺序），该顺序由 VF 内 `vf.mem_bar(VST_VLD)` 显式保证并在此记录（api_mapping_and_tile_planning.md："不配置mutex_ids时，相关核内跨Pipe依赖必须按实际数据路径手工插入同步"——本 tile 无跨 pipe 依赖，V 域内用 vf.mem_bar）。无 `make_tile` scratch tile。
- **依据算子**: `pro_ops/vf_api/test_layernorm_tile_group_vf.py`（双缓冲 in/out + 单槽向量 group + auto_mutex 全自动同步）、`pro_ops/element_wise/test_add.py`（`mutex_ids=[0,1]` 双缓冲最小骨架）。
- **理由**: M-块循环内 `pl.load`（MTE2）写 rows/anchor ↔ `pl.transpose`/VF（V）读，VF 写 out32 ↔ `pl.store`（MTE3）读——全部为跨 pipe 依赖，由各 tile 槽位携带的 mutex ID 经 auto_mutex 管理；槽位以 `next()` 固定顺序轮转（每 M 块/每任务一次），无动态下标（acc 的显式下标 `mblk%2 ∈ [0,2)` 有界）。

### double buffer 地址规划

| 项目 | 结论 | 受影响 tile/地址 |
|------|------|-----------------|
| buffer 数 | rows/trans/anchor/out32 = 2；out16（fp16 实例）= 2；acc = 2（ping-pong 状态，非流水用途） | 全部 tile |
| PONG 地址 | 单基址 + 槽间距自动展开（`addrs=0x…` 单值，第 i 块 = base + i×slot_size） | 见上方地址表（每 tile 第二槽 = 基址 + 单槽大小） |
| 槽位访问 | `next()`（rows/trans/anchor/out32/out16）；`group[mblk%2]`（acc，下标有界 [0,2)） | 见 §2 TileGroup 槽位访问表 |

buffer 数依据：官方指定算子中同类行规约（layernorm）与逐元素（add）样例均为双缓冲（搬运与计算 2 级重叠）；本算子 M-块循环同构（load 块 b+1 ∥ VF 块 b），2 槽满足 1 beat × 1 role 的计数法则下限，UB 余量 48% 亦不允许更深的无谓占用。

---

## §4 循环与 Section 结构（R4 输出）

> 依据 [循环与Section结构设计](../references/loop_design.md)，把§0确定的Module放入具体Section代码结构，再组织循环和跨Tile状态，并确定分核信息的获取位置。

### 参考样例

- **主要参考样例**: `pro_ops/vf_api/test_layernorm_tile_group_vf.py`（EXPLORE_REPORT §4.1 #6，最高参考价值：全动态双轴 + 沿列规约 + 尾 lane 掩码 + `pl.set_validshape` + `kernel[None, num_cores]` 启动 + pytest/torch golden 验收形态）
- **可复用结构点**: ①`pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], DT_FP32]` 动态双轴签名与 `x.shape[0]/[1]` 运行时读取；②`make_tile_group` 双缓冲 + `@pl.jit(auto_mutex=True)` 免手写同步；③`pl.range(core_id, num_tiles, num_cores)` 多核分工形态（本设计改为等价均衡的连续均分，见 §5）；④`@pl.vector_function` 封装寄存器级规约链、kernel 主体负责 GM↔UB 搬运的分层。
- **补充参考**:
  - `pro_ops/element_wise/test_add.py`：双缓冲 + auto_mutex 最小骨架。
  - `pro_ops/vf_api/test_softmax_tile_group_vf.py`：max 归约组织（本设计 p=inf 用 `vf.max` 沿 m 链而非 `reduce_max`——归约方向不同：softmax 是寄存器内 lanes 归约，pdist 是沿 M 逐行更新 64 个 pair 并行状态，已在 §1 记录差异依据）。
  - `pro_ops/fa/test_fa_tilingkey_attn_mask.py`：`TilingKeyField(bits, values)` 声明 + 启动第三位 Key dict + kernel 内 `if` 折叠的分支模式（PBranch 复用）。
  - `pro_ops/matmul/test_matmul_perf_asw_4k_dn_move_offset_dynamic.py`：TilingData 传递动态维度与派生量的组织。

### 本算子结构说明

- **Module / Section对应关系**: Module 1 → 唯一 `pl.section_vector()`；无其他 Section。
- **结果单元**: 一个 64-pair 任务块 (i, jseg) 的输出段 `y[base : base+VP)`（连续、覆盖写、唯一写入者）。M-块循环在结果单元内部持续更新 acc 状态。
- **跨Tile状态生命周期**: `acc_group`（`[2,64]` fp32：行 0 = 64 lane 部分和 acc、行 1 = PBranch 1/5 的 64 lane NaN 计数 nan_cnt）——初始化：任务首个 M 块的 VF 内 `vf.full(0.0, preg)`（PBranch 1 的 max 初值 0 安全：d=|·|≥0，max(0,d)=d；NaN 由防护链独立传播）；更新：每 M 块 VF 调用内 `load_align(acc_in, 偏移0/64) → add/max → store_align(acc_out, 偏移0/64)`（ping-pong 槽 `mblk%2`）；使用/失效：任务尾 finalize `load_align` → inv-pow/NaN 覆写 → 写 out32，任务结束后槽位可被下一任务复用（新任务重新初始化）。
- **Section结构**:
  ```
  @pl.jit(auto_mutex=True, tiling_key=PdistTilingKey)
  def pdist_kernel_f32(x, y: pl.Tensor[[pl.DYNAMIC], DT_FP32], tiling: PdistTiling):
      # tile groups 声明（§3 地址；fp16 实例另有 rows16/trans16/anchor16 + trans32/anchor32）
      with pl.section_vector():
          core_id = pl.get_block_idx(); num_cores = pl.get_block_num()
          lo = core_id * tiling.B // num_cores; hi = (core_id + 1) * tiling.B // num_cores
          i, jseg = _locate_block(lo, tiling.N)          # 闭式二分（§5），每核一次
          for t in pl.range(0, hi - lo):                  # 任务块循环（连续段）
              j0 = i + 1 + jseg * TP; vp = pl.min(TP, N - 1 - i - jseg*TP)
              base = (i * (2*N - i - 1)) // 2 + jseg * TP          # int64（TilingData int=INDEX）
              n_mblk = (M + M_BLK - 1) // M_BLK
              for mblk in pl.range(0, n_mblk):            # M-块循环（内层，持续更新 acc）
                  ...load/transpose/(fp16: pl.cast widen)/set_validshape + pdist_accum_vf（§1 序列）...
              pdist_finalize_vf(acc_group[n_mblk % 2], out32_group.next(), vp, ...)
              pl.store(y, out32_slot, [base])             # 一维 offsets
              jseg = jseg + 1                             # 顺序推进（每步至多换 1 行）
              if jseg == _segs(i, N): i = i + 1; jseg = 0
  ```
- **循环嵌套**: 任务块循环（外，每核连续段 `[lo, hi)`）→ M-块循环（中，`ceil(M/M_BLK)` 次）→ VF 内 m 循环（内，`vc` 次，fp32/fp16 统一步进 1）。
- **动态循环上界**: `hi-lo`（块域均分）、`n_mblk = (M + M_BLK - 1) // M_BLK`、`vc = min(M_BLK, M - m0)`、`_segs(i, N) = ceil((N-1-i)/TP)`——全部 ceiling division + 实际维度 clamp；`N=2` 时 B=1、单核单任务；`M ≥ 1` 保证 `n_mblk ≥ 1`（规格不允许 M=0）。
- **分核信息**: 仅 Vector Section：`pl.get_block_idx()` / `pl.get_block_num()`（`section_vector` 内读取；loop_design.md 表：仅 Vector Kernel 直接用 block_idx/block_num）。

### CV 手动预加载流水设计（`is_fusion=true`时必填）

不适用（`is_fusion=false`，单 Vector Module，无 Cube↔Vector 数据交接）。

---

## §5 分核策略（R5 输出）

> 📌 权威依据：`$PYPTO_DEVKIT_DIR/docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/multi_core_partitioning_and_Tiling.md`。

### 分核方式

- **方案**: **块域连续均分**（非跨步）：把上三角全部 64-pair 任务块按字典序 (i, jseg) 编号 `0..B-1`，核 c 分配连续段 `[lo_c, hi_c)`，`lo_c = c·B // K`、`hi_c = (c+1)·B // K`（K = num_cores）。
  - **均衡性**：整除均分公式下各核块数差 ≤ 1，与跨步 `pl.range(core_id, B, K)` 的"最多差 1"等价（tutorial §跨步循环的均衡证明同构适用）；tutorial 警告的"连续分段不均衡"针对**不取余的固定长度 chunk**，本设计的整除均分无此缺陷。
  - **选连续均分而非跨步的理由**：任务块在 (i, jseg) 字典序上连续 → 同核的 rows GM 搬运地址连续推进（同 i 行内 j0 单调 +64），MTE2 局部性优于跨步的地址跳跃；核内 `(i, jseg)` 状态可顺序推进（每步至多换 1 行，单分支 `if jseg == segs(i)`），而跨步需按步长 K 换行（最多 K 行的循环修正）。
  - **核内定位**：`lo_c → (i₀, jseg₀)` 用闭式二分（每核一次，固定 17 次迭代，`pl.range(0, 17)` + `if` 更新）：`cumsegs(i) = i + F(N-2) - F(N-2-i)`，其中 `F(x) = Σ_{v=0}^{x} floor(v/64) = 64·q(q-1)/2 + q·r`（`x+1 = 64q+r, 0≤r<64`）。闭式由 `segs(i) = ceil((N-1-i)/64) = floor((N-2-i)/64) + 1`（i ≤ N-2）求和导出，`B = cumsegs(N-1) = (N-1) + F(N-2)` 与 host 侧同一公式（Python `divmod` 实现），host/kernel 单一来源（TilingData 传 B 供核数计算与核内均分）。
  - **块唯一写入者**：块 (i, jseg) 的输出段 `[base, base+VP)` 与任何其他块不重叠（上三角 idx 双射），无跨核写冲突，无需原子/同步（SPEC kernel 契约"无跨块累加到 GM、无原子累加"）。
- **host 侧 `num_cores` 计算式**:
  ```python
  from pypto_pro.runtime.platform import get_platform_info
  B = (N - 1) + _floor_sum(N - 2)              # host 闭式，与 kernel 同源公式
  num_cores = min(get_platform_info().vector_core_num, B)   # 仅 Vector Kernel → vector_core_num
  pdist_kernel_f32[None, num_cores, {"PBranch": code}](x, y, tiling)
  ```
  `vector_core_num` 从运行时平台信息读取（A5: `950PR_957x.ini vector_core_cnt=56`；tutorial："仅Vector Kernel使用 vector_core_num 作为上限，Host侧必须根据Kernel类型和任务Tile数计算合法的 block_dim"）。JIT 不自动截断超限 block_dim——min 保证。
- **launch 次数**: **1**。单 kernel 单启动；`is_list=false`（非 TensorList 输入），fp32/fp16 两实例按 `x.dtype` 选择其一启动（host 读 dtype 元数据选择 JIT 实例，属 wrapper 允许的只读元数据 + 一次启动）。

---

## §6 核间同步（R6 输出）

> 依据 [跨核同步](../references/cross_core_synchronization.md)。

### cross_core 涉及判定

- **是否涉及 cross_core**: **不涉及**
- **依据**: ①纯 Vector 单 Section，无 Cube↔Vector 执行域数据交接（R0：无 Cube 步骤）；②核间任务块经 §5 连续均分互不相交，每块输出段唯一写入者，无跨 Block 共享状态、无 `INTER_BLOCK`/`INTER_SUBBLOCK`/`UNICAST_BLOCK` 依赖；③核内依赖全部为 pipe 间（MTE2↔V↔MTE3）由带 mutex 元数据的 tile group + `auto_mutex=True` 管理（§3），V 域内 acc 的 VST→VLD 由 `vf.mem_bar(VST_VLD)` 管理（§2）——均属"普通核内数据依赖"，按 cross_core_synchronization.md 判定流程落入"不设置 cross-core 事件"分支。Section 数量（1 个 section_vector）不是判断依据，已按依赖实际核查。
- **流水实现**: 不涉及（无手动预加载流水）。

### 共享数据与槽位映射 / 同步点 / event_id 分配表

不涉及（无跨核共享 TileGroup / workspace；无 event_id 分配；`make_tile_group` 的 `fwd_ids`/`bwd_ids` 未使用——其仅绑定自动 Pipeline 变换，本设计未启用 `pipeline`）。

---

## §7 尾块处理（R7 输出）

> 📌 权威依据：`$PYPTO_DEVKIT_DIR/docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/tail_block_handling.md`。

### 尾块处理方案

```python
# ceiling division 计算（必须向上取整）
n_mblk = (M + M_BLK - 1) // M_BLK           # M-块数
segs_i = (N - 1 - i + TP - 1) // TP          # 行 i 的任务块数（i ≤ N-2）
B      = (N - 1) + F(N - 2)                  # 总块数（host 闭式，见 §5）

# 循环内计算尾块有效尺寸并先 set 再 load（set_validshape 有状态，每块重设）
vc = pl.min(M - m0, M_BLK)                   # 满 = M_BLK，M 尾 = 余数
vp = pl.min(TP, N - 1 - i - jseg * TP)       # pair 尾（上三角收缩的自然尾块）
pl.set_validshape(rows_slot,  [vp, vc])      # load 前：GM 只搬 VP 行 × vc 列，不越界读
pl.set_validshape(anchor_slot,[1, vc])
pl.set_validshape(trans_slot, [vc, vp])      # 转置后的有效区（行列互换）
pl.load(rows_slot, x, [j0, m0]); pl.load(anchor_slot, x, [i, m0])
pl.transpose(trans_slot, rows_slot)
# VF 内 m 循环上界 = vc（M 尾按整行跳过，fp32/fp16 统一步进 1，无残 lane）
# finalize 的写回裁剪：mreg = vf.update_mask(vp, dtype=pl.DT_FP32) → store_align 只写 vp 个
```

- **compact**: `None`（Vec ND `load`/`store` 路径用 valid_shape 控制搬运、物理 shape 作 UB 跨度，tail_block_handling.md："当前A5的Vec ND load/store使用valid_shape控制实际搬运行列…不需要配置compact"；本设计无 Mat→Left/Right、Acc 分形路径，`compact=1` 的布局耦合陷阱不适用）。
- **无效区域是否会被读取**: 部分会，均已中和：①trans tile 的垃圾列（pair k ≥ vp，来自 rows 未搬入的行）会被 VF load_align 读入，但各 lane 计算独立、垃圾 lane 的结果在 finalize 被 `update_mask(vp)` 裁剪，不污染有效 lane、不写出；②M 尾不产生残 lane（m 整行推进、循环上界 vc）；③`pl.store(y, out_slot, [base])` 的写出量由 out tile 的有效数据（vp 个，store_align 掩码已保证其余为旧值/零且不越界 GM——base+vp ≤ L）。无 fillpad 需求（不存在"归约读取 padding 区域且 padding 影响结果"的场景：sum/max 的无效输入根本不进入计算）。
- **transpose × valid_shape 交互（Stage 4 验证点）**: `pl.transpose` 文档未声明 valid_shape 传播语义。两种可能行为（只转置有效区 / 按物理 shape 全量转置）下本设计均正确（m 循环上界 vc + 垃圾 lane 裁剪对两者鲁棒）；Stage 4 以 §8 双尾 case 验证实际行为并在异常时回到本节修订。

---

## §8 目标测试 case（R7.5 输出 → 交付 develop）

> SPEC 已给出 10 个 P0 case（机器合同 `p0_cases`），在其基础上追加 4 个基础 case（按 R2/R7 的 tile 切分推导）。develop 直接实现，不再自行重算。全部 case 与 `pdist_golden_cpu.py`（fp32 参考；fp16 case 由 fp32 期望单点窄化派生，SPEC kernel 契约 cast 边界链）比对，容差 atol=rtol=1e-3。

| case 名 | 具体 shape / p / dtype | 覆盖场景 | design 适配确认 |
|---------|-----------|---------|----------------|
| `test_pdist_aligned` | `[130, 128]`, p=2, fp32 | M 恰为 1×M_BLK 满块（vc=128 无 M 尾）；行 0 有 3 块（64+64+1，跨 3 jseg） | ✅ segs(0)=ceil(129/64)=3，jseg 尾块 vp=1 走 update_mask 裁剪 |
| `test_pdist_m_tail` | `[200, 150]`, p=2, fp32 | M 尾 vc=22（非 64 倍数的短尾，M-块循环上界裁剪） | ✅ n_mblk=2，块 2 vc=22；M 维无残 lane 由整行推进保证 |
| `test_pdist_tail2d` | `[77, 65]`, p=1, fp32 | 双轴尾块：N 尾（多行 segs=1 的短块）+ M 尾 vc=65 | ✅ N=77 → B=76+F(75)=76+12=88 块（行 0 两块 64+12、行 1 两块 64+11，尾部行单块递减）；M=65 → 1 块 vc=65 |
| `test_pdist_multitile` | `[600, 300]`, p=2, fp32 | 跨多 tile：B=3110 > 56 核（多核均分）；M=300 → 3 M-块（128+128+44，acc 跨 3 块 ping-pong）；行内多 jseg | ✅ B=(599)+F(598)=599+2511=3110；n_mblk=3 验证跨块状态持久化 |
| `func_p0_hamming_p0` | `[1000, 200]`, p=0, fp32 | P0：Hamming 分支（PBranch 0 全链） | ✅ §1 分支表 |
| `func_p0_manhattan_p1` | `[1000, 200]`, p=1, fp32 | P0：Manhattan（恒等链） | ✅ |
| `func_p0_euclidean_p2` | `[1000, 200]`, p=2, fp32 | P0：Euclidean 主链 | ✅ |
| `func_p0_fractional_p05` | `[1000, 200]`, p=0.5, fp32 | P0：分数 p（sqrt+mul 精确链） | ✅ |
| `func_p0_chebyshev_pinf` | `[1000, 200]`, p=+inf, fp32 | P0：Chebyshev（max 链 + NaN 防护链） | ✅ |
| `func_p0_min_shape` | `[2, 1]`, p=2, fp32 | P0：最小 shape（B=1 单核单块、vp=1、vc=1） | ✅ num_cores=min(56,1)=1 |
| `func_p0_large_m` | `[64, 8192]`, p=2, fp32 | P0：M 超片上（64 M-块，强制分块轮转） | ✅ n_mblk=64 |
| `func_p0_partial_block` | `[777, 300]`, p=1, fp32 | P0：非整块尾部 + 多核不均（B=5196） | ✅ |
| `perf_p0_euclidean` | `[4096, 256]`, p=2, fp32 | P0 性能：核心场景（B=133056） | ✅ B=4095+F(4094)=4095+128961=133056 |
| `perf_p0_fp16_large` | `[8192, 512]`, p=2, **fp16** | P0 性能：fp16 大规模（tile 级 cast widen/narrow 全链，M_BLK=128 与 fp32 统一） | ✅ B=8191+F(8190)=8191+520065=528256；M=512 → 4 M-块；入口 2 处 + 出口 1 处 pl.cast |
| （语义追加）`test_pdist_semantics` | NaN/Inf 输入构造（同 golden [5] 组）：NaN 行、inf 元素、重复行、p 各分支 × fp32/fp16 | NaN/Inf 逐对传播、p=0 NaN 计数、fp16 出口窄化 NaN 语义（§1 验证义务） | ✅ §1 NaN 语义表逐条对应 |
| （general 追加）`test_pdist_general_p` | `[40, 13]`, p=3.0 与 p=1.5, fp32 | PBranch 5 近似链（exp/ln + d=0/s=0 钳位 + NaN 防护）；golden 按 SPEC 公式 pow 计算 | ✅ §1 分支表 PBranch 5；d=0 由重复行构造触发 |

单轴说明：本算子输出为一维（上三角展平），"单轴尾块"由 pair 尾（vp）承担、"双轴尾块"由 N+M 尾共同承担，4 个基础 case 已覆盖整除/单尾/双尾/跨多 tile 四类，不适用 R7.5 单轴算子例外（仍给足 4 个）。

---

## §9 综合评估（R8 输出）

### 准确性检查

| 检查项 | 结果 | 证据 |
|--------|------|------|
| API 调用链完整覆盖数学公式的每一步 | ✅ | §1 分支表逐 p 覆盖公式 (2)(3)(4)：abs_sub→幂/比较→沿 M 累加/max→inv-pow→上三角写回；idx 公式 (1) 在任务定位与 base 计算；每个原始输入 x 沿调用链可追踪到 y |
| §4 已参照官方样例确定循环与Section结构（含参考样例路径与结构说明） | ✅ | §4 参考样例节：layernorm（主参考：动态双轴/双缓冲/VF 归约链/多核）、add、softmax、fa-tilingkey、matmul-dynamic，逐项列出可复用结构点与差异 |
| 数据依赖正确（Module 顺序 + sync 位置） | ✅ | 单 Module 单 Section；同步点全集 = tile mutex（auto_mutex，MTE2↔V↔MTE3）+ acc 的 vf.mem_bar(VST_VLD)（§2/§3），无遗漏场景（load.md 同地址 WAW 场景因双缓冲不触发） |
| dtype 精度满足要求 | ✅ | 中间全 fp32（SPEC cast 边界链）：fp16 在 load 后 abs_sub 前 widen（§1），reduce 路径无 FP16 累加（本设计归约用 add/max 链于 fp32 域，规避 reduce_sum FP16-src-FP16-累加限制）；fp32 实例全程 FP32 |
| 归约类 API 的 `[M,1]`/`[1,N]` 输出已设 `layout` | ✅（不涉及） | 未使用 vf.reduce_*（归约方向为沿 M 的 add/max 链，非寄存器内 lanes 归约，§1 已记录依据）；无归约输出 tile 需设 layout |
| Acc tile 物理 M×N×dtype_bytes ≥ fractal | ✅（不涉及） | 无 Cube/Acc tile（纯 Vector，仅 UB） |

### 泛化性检查

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 目标测试 case（≥4）已按 tile 切分确定具体 shape，且逐个验证 design 可适配 | ✅ | §8 表 16 个 case（4 基础 + 10 P0 + 2 语义/ general），逐个给出 tile 适配推导（B/vc/vp/n_mblk 数值核算） |
| 是否正确处理了尾块（M/N 尾块） | ✅ | §7：M 尾 vc 循环上界（无残 lane）、pair 尾 update_mask 裁剪、N 尾 segs 递减 + 顺序推进、set_validshape 先于 load、GM linear extent 越界核算 |
| 归约轴可能超单Tile时，已给出覆盖完整归约轴的方案 | ✅ | §0 归约轴容量：M 分块 + acc ping-pong 跨块状态（[1,65535] 全区间，func_p0_large_m M=8192 → 64 块验证） |
| 循环边界正确 | ✅ | n_mblk/segs/B 全 ceiling division；vc/vp clamp 到实际维度；hi-lo 均分段；N=2/M=1 退化 case 在 §8 |
| 超越函数在 dtype 范围内无溢出 | ✅ | §1 数值安全边界表：sqrt 域内；ln 域外（d=0/s=0）钳位防护；exp 大 p 按 SPEC P2 不保护（与 PyTorch 对齐）；fp16 中间量全在 fp32 域 |
| 窄 dtype（fp16）下平方/同量级相乘/长轴累加的中间值量级上界在目标 dtype 范围内 / 升位宽 cast 位置正确 | ✅ | \|d\|^p ≤ (2·65504)^p：fp16 域必然溢出（(2·65504)²≈1.7e10 > 65504），widen 在 mul 之前（abs_sub 前，非归约前）——precision.md 顺序要求满足；长轴累加（M ≤ 65535 项）在 fp32（误差 ~√M·2⁻²⁴ 相对量级 ≪ rtol 1e-3） |
| 跨 tile 状态是否正确初始化和持久化 | ✅ | acc：任务首块 full(0.0)（PBranch1 初值 0 的安全性论证 §4）、跨块 ping-pong + mem_bar、任务尾消费失效；acc_group 显式下标有界 [0,2) |
| cross_core 同步方案正确（存在跨执行域或跨Block/subblock依赖时） | ✅（不涉及） | §6：单 Vector Section + 块唯一写入者，按判定流程核查无跨核依赖；条件性检查确认无 Cube↔Vector / 跨 Block 依赖 |

### 一致性检查

| 检查项 | 结果 | 说明 |
|--------|------|------|
| R0-R7 输出无矛盾 | ✅ | §1 调用链的每个 tile（rows/trans/anchor/acc/out32/out16）在 §2 属性表与 §3 地址表有条目；§4 循环引用 §2/§3/§5 的量；§7 的 set_validshape 与 §2 valid_shape=[-1,-1] 声明一致 |
| 所有决策有证据支撑 | ✅ | 每项决策标注 API 文档路径 / 官方样例 / KB 引用 / 平台文件键（§3 容量、§5 核数、§1 各 API） |
| 各内存空间 tile 总用量分别不超容量上限 | ✅ | UB: fp32 133632B/253952B=52.6%、fp16 134400B/253952B=52.9%（§3，`950PR_957x.ini ub_size`）；其余空间无 tile |
| `tile_dims` 使用时已关注大 stride 影响 | ✅（不涉及） | 未使用 tile_dims（§2 说明） |
| 条件性检查（§6 "不涉及cross_core"时确认无跨域依赖） | ✅ | §6 依据节逐条核查（执行域/跨 Block/写冲突） |

### 评估结论

- **整体**: 通过（R8 全部检查项 ✅，无未通过标记）
- **限制条件**: ①`pl.transpose` 与 valid_shape 的交互语义文档未明示，已设计为两种解释下均正确 + §7 验证点；②fp16 出口 `pl.cast` 的 NaN 传播未文档化，§1 已列 fallback（vf.astype 非饱和 + interleave 重组）与语义验证 case；③PBranch 5（一般 p）为 exp/ln 近似路径（非 P0），d=0/s=0/NaN 均有防护链但精度语义依赖 Stage 4 语义 case 验收；④`vf.load_align` 的 `dist=BRC_B32` 锚行广播与 `store_align` 的 preg 掩码裁剪均为文档化能力（对齐约束 4B / mask 写入语义）但无官方样例直接组合先例，Stage 4 首个冒烟 case 覆盖。
- **修改记录**: R2→R8 迭代 1 轮：R1 初版曾采用"逐 pair 行主序 + 每对独立 reduce_sum"布局，R2 寄存器核算发现 64 个 acc 寄存器超 32 上限（TP 退到 16 后输出攒批退化），回退 R1 改为 pair-lane 转置布局（现设计）；R4 初版跨步分配需动态换行循环，回退改连续均分 + 闭式二分定位（现设计）。无未闭合修改。

---

## §10 Tile 数据流全景图

```
GM x[N,M] (fp32/fp16, ND 行主序)
   │
   │ [pl.load offsets=[i, m0], valid [1,vc]]                     (MTE2)
   ▼
anchor_group[1,M_BLK] (UB, ×2 slots, mutex[4,5], rotating)        (fp16: anchor16 fp16 tile)
   │                                                              │ (fp16: [pl.cast CAST_NONE] → anchor32 fp32)
   │ ── vf.load_align(offset=m, dist=BRC_B32) → 64-lane 广播 ──┐  (fp16 的 BRC 读 anchor32，与 fp32 同构)
   │                                                             ▼
GM x[N,M] ──[pl.load offsets=[j0,m0], valid [VP,vc]]──→ rows_group[64,M_BLK] (UB, ×2, mutex[0,1])
   (MTE2)                                                        │
                                                                 │ [pl.transpose]  (V 流水)
                                                                 ▼
                                              trans_group[M_BLK,64] (UB, ×2, mutex[2,3])   (fp16: trans16 fp16 tile)
                                                                 │ (fp16: [pl.cast CAST_NONE] → trans32 [M_BLK,64] fp32, mutex[8,9])
                                                                 ▼
   acc_group[2,64] fp32 (UB, ×2, 无mutex, resident ping-pong;    │
                          行0=acc / 行1=nan_cnt)                 │
   ◄──[vf.load_align acc_in, 偏移0/64]──┐                        │
                                        │  ┌── vf.load_align(trans32|trans, m*64) ─┘ (每 m 一行 64 pair, 步进 1)
                                        ▼  ▼
                        @pl.vector_function pdist_accum_vf (V, 全 fp32 域)
                        d  = vf.abs_sub(bcast, rv)            ←─ p 分支 (TilingKey PBranch):
                        dp = mul(d,d) / sqrt(d) / ne+select / max / exp(ln·p)   §1 分支表
                        acc = vf.add(acc, dp)   (或 max 链 + nan_cnt 累积链)
                              │
                              └──[vf.store_align acc_out, 偏移0/64]──→ acc_group 另一槽
                                  [vf.mem_bar VST_VLD]           (M-块循环内轮转, mblk%2)
                                                                  │  (最后一块后)
                                                                  ▼
                        @pl.vector_function pdist_finalize_vf (V)
                        y = sqrt(acc)/mul(acc,acc)/恒等/select(NaN 覆写)/exp(ln·inv_p)
                        [vf.update_mask(VP) 裁剪 pair 尾]
                              │
                              └──[vf.store_align]──→ out32_group[1,64] fp32 (UB, ×2, mutex[6,7])
                                                              │ (fp16 实例)
                                              [pl.cast CAST_RINT] (V)
                                                              ▼
                                              out16_group[1,64] fp16 (UB, ×2, mutex[12,13])
                                                              │
GM y[N(N-1)/2] ◄──[pl.store offsets=[base]]───────────────────┘ (MTE3, 覆盖写, 一维)
   base = i*(2N-i-1)//2 + jseg*64   (int64, 上三角行优先)

多核: 块域连续均分 lo=c·B//K .. hi=(c+1)·B//K；每块唯一写者；无 cross_core 事件
同步: auto_mutex (tile mutex, MTE2↔V↔MTE3) + vf.mem_bar(VST_VLD) (acc) —— 全部核内
```

### 图例

| 标记 | 含义 |
|------|------|
| `[pl.load]` / `[pl.store]` | MTE2 / MTE3 搬运：GM ↔ UB |
| `[pl.transpose]` / `[pl.cast]` | V 流水 tile 级布局/dtype 转换 |
| `→` | 操作，标注 API 名 |
| `---` | 跨阶段持久化（acc 跨 M-块、跨 VF 调用存活，ping-pong 槽） |
| `├─` / `◄──` | 同一数据的多条消费路径（anchor 广播 / acc 读写） |

> 全景图每个 tile 与操作均可在 §3 地址表（anchor/rows/trans/acc/out32/out16）与 §1 API 序列（load/transpose/BRC/load_align/abs_sub/分支/add/store_align/mem_bar/cast/store）找到对应条目，已逐项核对无缺漏。
