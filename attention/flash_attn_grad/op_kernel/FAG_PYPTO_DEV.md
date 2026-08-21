# FlashAttentionScoreGrad pypto-pro 开发文档

对标 `arch35/flash_attention_score_grad_kernel.h`（AscendC BN2GS1S2 模板）的
pypto-pro 重写。变量名与代码结构尽量跟随 AscendC 原版。

## 1. 范围与配置

| 项 | 取值 |
|---|---|
| 布局 | BSND / BNSD（均为非 TND），见 3 节 |
| head 数 | 支持 GQA：N1 = N2*G（Q 侧 N1，KV 侧 N2），G=1 即 MHA |
| headDim | 128 |
| 数据类型 | fp16 输入 / fp32 累加 / fp16 输出 |
| 不支持 | sparse、pse、dropout、attenMask、确定性计算、rope |
| 必开 | L1_PRELOAD、L0C 驻留累加 |
| 硬限 | UB ≤ 248 KB（实测占 241.5 KB）、L1 512 KB |
| 基准平台 | 910D，36 cube / 72 vector 核，L2 = 128 MB |

参考文档 `/root/z00675317/pypto/docs/zh/pypto_pro/`。

**运行环境**：必须用 CANN 9.2.0 路径，conda 里的 `pypto_pro` 副本是旧版、
会让 `order=` 参数失效：

```bash
PYTHONPATH=/usr/local/Ascend/cann-9.2.0/python/site-packages \
  /root/.conda/envs/zyj/bin/python3 <script>
```

## 2. 性能结果

S1=S2=8192，D=128，36 核。口径为 msprof `task_duration`（**这才是端到端耗时**，
不要用 wall-clock：pypto 有约 3 ms 固定 launch 开销）。AscendC 基准 206.6 us。

| 阶段 | 耗时 | vs ASC |
|---|---|---|
| 起点 | 6011 us | 29.1x |
| NZ/ZN 双视图消除 `pl.transpose` | 364.3 us | 1.76x |
| 首轮连发 4 次 matmul | 341.5 us | 1.65x |
| K/V 按 s2 组条件载入 | 332.9 us | 1.61x |
| Q/dO 复用（4 槽双游标） | 330.4 us | 1.60x |
| 融合 cast+ND2NZ VF | 319.1 us | 1.54x |
| **L0A/L0B 双缓冲** | 244.0 us | 1.18x |
| **L0C acc 双缓冲** | 183.9 us | 0.89x |
| 补齐 kernel 内 pre/post（口径对齐） | 193.7 us | 0.94x |
| 改用 LSE 输入 | **192.5 us** | **0.93x** |

等效算力 467 TFLOPS（峰值约 500），AscendC 416 TFLOPS。

### 流水占用对比（193.7 us 版，绝对时间 us）

| 流水 | ASC | pypto | 差 |
|---|---|---|---|
| cube mac | 185.8 | 182.4 | −3.5 |
| cube mte1 | 120.7 | 109.9 | −10.8 |
| cube mte2 | 69.6 | 94.0 | +24.4 |
| cube fixpipe | 190.0 | 142.3 | −47.6 |
| cube scalar | 94.8 | 73.0 | −21.8 |
| vec 计算 | 85.3 | 127.9 | +42.6 |
| vec mte2 | 24.6 | 66.3 | +41.7 |
| vec mte3 | 39.3 | 28.5 | −8.9 |
| vec scalar | 63.6 | 30.0 | −33.6 |
| **流水停顿** | **16.7** | **4.7** | |

两边瓶颈性质不同：ASC 是 fixpipe(0.919) + mac(0.899) 双限；pypto 是 mac 单限
（0.941），停顿已压到 4.7 us。

### 多 head 扩展性（S=8192）

接口对齐 `flash_attn_grad`（13 入 / 3 出 / 单 workspace）后重测，单位 us：

| N | 线性切分 | swizzle | 工作集 | 判据 | 判据是否选中更快项 |
|---|---|---|---|---|---|
| 1 | **193.4** | 221.7 | 22 MB | 0 | ✅ |
| 8 | **1556.5** | 1760.6 | 176 MB | 0 | ✅ |
| 10 | **1904.3** | 2007.4 | 220 MB | 0 | ✅ |
| 12 | 2833.4 | **2676.8** | 264 MB | 1 | ✅ |
| 16 | 4120.5 | **3549.5** | 352 MB | 1 | ✅ |
| 24 | 6482.9 | **4982.3** | 528 MB | 1 | ✅ |

实测拐点在 N=10 与 N=12 之间，即工作集约 2×L2（264 MB），**不是**刚超 1×L2
的 N=6。原因是刚超 L2 时仍有可观的部分命中，此时 swizzle 更粗的分配粒度
（整列 s1Outer 块）带来的负载不均反而占上风。判据阈值因此取 `2 * L2`，六个
测点全部选中更快的那条路径。

早期（接口改造前）以 ASC 为基线的对比：N=1 ASC 206.7 / pypto 193.5（0.94x）、
N=8 1696.8 / 1602.6（1.04x）、N=32 6859.0 / 6555.6（0.96x）、
N=64 14294.8 / 13395.3（0.94x）。线性路径在接口改造后进一步变快
（N=8 1630→1556），拐点随之右移。

### 精度

对 AscendC 结果直接比对（**不是 CPU golden**）：max/sum 版 dQ/dK 3.05e-05、
dV 6.1e-05；LSE 版三项均 6.1e-05。差异来自舍入次序，在 fp16 精度内。

## 3. 接口定义

kernel 入口 `flash_attn_grad`：**13 输入 / 3 输出 / 1 workspace / 1 tiling**，
形参顺序严格对齐 `op_host/flash_attn_grad_def.cpp` 的 `Input()` 声明。

```
输入  q, k, v, dout, attn_out, softmax_lse,
      cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, sinks, attn_mask, metadata
输出  dq, dk, dv
其余  workspace, tiling
```

后 7 个输入本实现不读取（变长 TND、sinks、attn_mask 不在支持范围，metadata
预留）。**形参必须原样保留** —— 少一个就会与 host 的 13 输入按位错位，症状是
读到相邻张量的数据而非报错。

`FlashAttnGradTilingData` 的字段名与顺序必须与 host 的
`TILING_DATA_FIELD_DEF` 逐一对应，codegen 出的
`FlashAttnGradTilingData_tiling.h` 按此声明顺序排布内存：

```python
b, s1, s2, n1, n2, d, dv: int
scaleValue: float
gSize: int                                   # GQA 组大小 G
viewD0Q, viewD2Q, coefB0Q: int               # Q 侧(N1 个 head)视图参数
viewD0KV, viewD2KV, coefB0KV: int            # KV 侧(N2 个 head)视图参数
coefN0, coefN2: int                          # 只取决于 layout，两侧共用
```

注意其中**没有 coreNum 字段**，kernel 里改用 `pl.get_block_num()`（实测
cube/vector 两侧都返回物理核数 36，故 CV 双侧算出的 `numBlocks` 一致，
set/wait 能正确配对）。

### GQA：N1 = N2 * G

Q 侧（`q`/`dout`/`attn_out`/`dq`）有 N1 个 head，KV 侧（`k`/`v`/`dk`/`dv`）有
N2 个，`G = N1 / N2` 由 host 相除得出（对齐 ASC 的
`tiling_normal_regbase.cpp`，各 layout 分支都是 Q 的 head 维除以 KV 的）。
host 校验 `N1 % N2 == 0`，不满足则返回 `GRAPH_PARAM_INVALID`。

块网格为 **b → n2 → g → s2o → s1o**（g 在 n2 之内、s2o 之外，与 ASC 的
`GetNextValidIdx` 一致），共 `b*n2*G*s1Outer*s2Outer` 块。索引：

```
idx0/idx2 的 head：Q 侧用 n2oIdx*G + goIdx，KV 侧用 n2oIdx
lse   ：[B, N1, S1]，offset = ((b*N2 + n2o)*G + g)*S1
```

`softmax_lse` 必须按 **N1** 存：每个 Q head 的 softmax 归一化因子不同，G 个
Q head 不能共享一份 lse。

**dK/dV 跨 g 靠 GM 的 atomicAdd 合并**。组号 = `index // s1Outer`，已融合
`(b,n2,g,s2o)` 四维，跨 g 即换组 → 先 flush 再重新在 L0C 上累加，G 个 Q head
对同一 KV head 的贡献在 GM 上相加。这与 ASC 的 BN2GS1S2 模板一致（该模板
`needAtomic` 恒真，`keyOffset` 不含 g）。**故 GQA 是纯索引改动，不动片上存储。**

三处容易漏掉 G 的地方（都会静默算错）：

| 位置 | 症状 |
|---|---|
| `wsRowsQ`（workspace 分区） | dq 段须按 N1，误用 N2 会让 dk/dv 起始偏移错位 |
| `numColTasks`（swizzle 列数） | 须含 G，否则 swizzle 下漏掉 `(G-1)/G` 的列 |
| `s2_group_of`（dKV 累加组号） | 旧的逐维还原式在 G>1 下失效，已化简为 `index // s1Outer` |

swizzle 本身无需改：它只依赖「融合 batch 维 × s2Outer」的列数，把 G 并入
`b*n2*G` 即可，列内 s2o 不变的性质保持，K/V 单载入与 L0C 累加都不受影响。
（已用 Python 全量验证：索引分解 240 个 index 全检自洽；swizzle 在 4 组 GQA
配置下无重复、无遗漏、无组内断裂。）

### 布局泛化：BSND / BNSD 统一四维视图

跟随布局的共 8 个张量：输入 `q/k/v/dout/attn_out` 与输出 `dq/dk/dv`
（输出形状由 infershape 的 `*dqShape = *qShape` 强制跟随输入）。
`softmax_lse` 恒为 `[B,N,S1]`，其余输入布局不变。

**layout 不占 tilingkey 位**（组合数保持 32，不撑大算子二进制）。做法是把两种
布局收敛成同一种四维紧凑视图 `[viewD0, S, viewD2, D]`，于是 kernel 里
`make_tensor` 与 `pl.load(order=...)` 只有一份代码，布局差异全部退化为数值：

| layout | 视图 shape | 索引 `[idx0, s, idx2, 0]` | 系数 (B0,N0,N2) |
|---|---|---|---|
| BSND `[B,S,N,D]` | `[B, S, N, D]` | `[b, s, n, 0]` | (1, 0, 1) |
| BNSD `[B,N,S,D]` | `[B*N, S, 1, D]` | `[b*N+n, s, 0, 0]` | (N, 1, 0) |

```
idx0 = boIdx * coefB0 + head * coefN0
idx2 = head * coefN2
```

GQA 下 Q 侧与 KV 侧的 head 数不同（N1 vs N2），故 `viewD0/viewD2/coefB0`
各有一套（表中的 N 分别取 N1 / N2），`head` 分别取 `n2oIdx*G+goIdx` / `n2oIdx`；
`coefN0`/`coefN2` 只取决于 layout，两套共用。

两者的**最内轴都是 D**，所以已验证的 `order=[1,3]`（转置读 `[3,1]`）语义不变；
索引是纯算术、无分支，因此不触碰 pypto「分支赋值无法 phi 合并」的限制。
BNSD 的 `viewD2 = 1` 是退化轴（实测支持）。

**不能改用「倒置 stride」的办法**：`make_tensor` 的 `stride` 参数不支持任意
跨距，给出中间轴跨距倒置的 stride 时被**静默忽略**、按 shape 的紧凑跨距去读，
不报错。实测 BNSD 这样读出的数据精确等于「按 BSND 语义（stride `N*D`）误读」
的那一段。探针见 `/root/.claude/jobs/c66577db/tmp/probe_layout.py`（证伪）与
`probe_bnsd4d.py`（验证本方案的 plain 读 / 转置读 / atomicAdd 三项）。

pre/post 与 workspace **无需改动**：workspace 与输出用同一布局，而 pre 是整体
清零（顺序无关）、post 是 `[rows,D]` 展平后逐行 cast 搬运，两者逐元素对应 ——
无论哪种布局都是同一个 `B*S*N*D` 紧凑块的同一种排列。

host 侧维度提取对齐 ASC 参考实现
（`attention/flash_attention_score_grad/op_host/flash_attention_score_grad_infershape.cpp`
的 BNSD 分支）：BNSD 取 `b,n,s1,d = dim(0..3)`，k/v 的 `s2 = dim(2)`。
swizzle 判据与 workspace 尺寸只依赖 B/S/N/D 的数值，与布局无关。

单块 workspace 内部再切三份 fp32 累加区（主循环用 atomicAdd 累加，pre 阶段清零）。
dq/dk 段按 D 宽、dv 段按 Dv 宽（见下一节，D 与 Dv 可以不等长）：

```
dq_workspace : b*s1*n1 * D  fp32
dk_workspace : b*s2*n2 * D  fp32
dv_workspace : b*s2*n2 * Dv fp32
```

host 侧尺寸为 `libapiSize + (dqElems + dkElems + dvElems) * 4`。

### 不等长 headdim：D=192 / Dv=128，且 D 轴分块

Q/K/dq/dk 用 `D`，V/dO/attn_out/dv 用 `Dv`（`Dv <= D`，host 侧 CheckParams
显式拒绝 `Dv > D` —— ASC 参考实现缺这个校验，此处不继承该缺陷）。

**D 与 Dv 都必须是编译期分档，不能像 layout 那样纯运行期判断。** layout 只改
索引算术（运行期数值），而 D/Dv 决定的是**片上 tile 的分配宽度**，而 pypto 的
tile shape 是 trace 期常量。ASC 同样把 D 做成模板参数
`DTemplateType::Aligned128/192`。分两档：`d_align/dv_align ∈ {128, 192}`。

按最大值无条件分配不只是慢，而是**放不下**：L0C 需要 288KB > 256KB 预算。

`d_align` 占 tilingkey bit5、`dv_align` 占 bit6。加两位本会把组合数从 32 变成
128，但 `is_valid` 剪掉了不可能与未实现的组合（`dv_align > d_align`，以及
D=192 只编本实现真正走的那条路径），**实际有效组合 36**（+12.5%）。

#### D 轴分块：保住 L0A/L0B 双缓冲

直接按 192 单发会让 L0 tile 变成 48KB，双缓冲（2×48 > 64KB）放不下 —— 那是
收益最大的一项优化（5.1 节，135us）。解法是把 D 轴按 **trace 期常量**分块：
`D=128 -> [(0,128)]`，`D=192 -> [(0,96),(96,96)]`。每块 `[128,96]` fp16 = 24KB，
双缓冲 48KB ≤ 64KB。**D=128 档分块表只有一项，展开后与改动前生成同样的代码，
故该档天然零回归。**

分块宽度取 96+96 而非 128+64（实测，单位 us/次完整 D 遍历）：

| 方案 | MM2（D 在 K 轴） | dQ/dK（D 在 N 轴） |
|---|---|---|
| 96+96 (DB) | **0.466** | **0.738** |
| 128+64 (DB) | 0.549 (+15.2%) | 0.818 (+9.8%) |
| 192 单发（无 DB） | 0.781 | 1.123 |

两段等宽让 mac 与 MTE1 节拍整齐；128+64 的第二段只有一半宽，盖不住前段。
96 = 6×16 满足 fp16 的 C0=16 分形对齐。

D 落在哪根轴上决定分块怎么合并：

| matmul | D 所在轴 | 分块处理 |
|---|---|---|
| MM2 `S=Q@Kᵀ` | K 轴（D） | `matmul_acc` 跨块累加到同一 acc |
| MM1 `dP=dO@Vᵀ` | K 轴（**Dv**） | 同上，宽度按 Dv 截断，整块无效则跳过 |
| MM3 `dQ=dS@K` | N 轴（D） | 各块写不相交的输出列，**不累加** |
| MM4 `dK=dSᵀ@Q` | N 轴（D） | 每块各有一块常驻 L0C，跨 s1 累加 |
| MM5 `dV=Pᵀ@dO` | N 轴（**Dv**） | 同上，宽度按 Dv 截断 |

L0C 预算：192 档 acc×1（48KB 声明宽 128 列 → 64KB）+ dk×2 + dv×2，正好 256KB；
代价是 acc 失去双缓冲。对应 ASC 的 `isDkvL0CResidentForD192Dv128`。
L1 预算：192 档每块 48KB，512KB 装不下 14 块，故 Q/dO 各留 **3** 槽（mm4/mm5
复用 mm2/mm1 数据所需的最小值，`N%3 != (N-2)%3` 仍成立），合计 496KB。

**双缓冲槽位间距必须按 tile 的「声明宽度」算，不是分块宽度**：L0A/L0B 还要装
MM3/MM4/MM5 的 `[M,N]` 左矩阵，声明成 `ACC_COLS = max(128, 分块宽)` 列，若按
96 算间距（24KB）两个槽位会重叠。

性能参考（S1=S2=4096，36 核）：D=192/Dv=128 **115.9us**，D=Dv=192 **119.7us**。

不等长时的实际配置恒为 **D=192 / Dv=128**，等长时为 D=Dv=192；只覆盖这两种
组合，不对 Dv 做泛化（Dv=96/160/64 并非真实配置）。

实测通过：JIT 12 个 DV 用例（等长 192 与 D=192/Dv=128 各含单块/多块、S1/S2
尾块、双尾块、多 batch/head、GQA G=3/4、多核、4096 满核）36/36 + BNSD 交叉
+ D=128 基线回归 13/13 + GQA 11/11 + aclnn 端到端。

## 4. 数据流与内存布局

### 五个 matmul

| 编号 | 计算 | 左矩阵来源 |
|---|---|---|
| mm2 | S = Q @ K^T | L1_Q / L1_K（NZ） |
| mm1 | dP = dO @ V^T | L1_DO / L1_V |
| mm3 | dQ = dS @ K | L1_DS（NZ）/ L1_K_ND |
| mm4 | dK = dS^T @ Q | L1_DS 的 **ZN 视图** / L1_Q 延后游标 |
| mm5 | dV = P^T @ dO | L1_P 的 **ZN 视图** / L1_DO 延后游标 |

### L1（512 KB）

`L1_Q/K/V/DO` 各 32 KB，`L1_DS`、`L1_P` 为 CV 共享缓冲，`L1_K_ND` 供 mm3 用
非转置 K。`L1_Q2..Q4`、`L1_DO2..DO4` 构成 Q/dO 的 4 槽环。

### L0（各 64 KB / 256 KB）

```
L0A: 0x0000, 0x8000        两槽双缓冲，mutex 12/26
L0B: 0x0000, 0x8000        两槽双缓冲，mutex 13/27
L0C: acc  0x00000/0x30000  两槽双缓冲，mutex 14/28
     dk   0x10000          L0C 驻留累加器，不轮转
     dv   0x20000          同上
```

L0C 正好用满 256 KB（4 × 64 KB）。

### UB（248 KB 硬限，实测 241.5 KB）

主循环：`UB_MM1_0/1`、`UB_MM2_0/1` 各 32 KB（mm 结果双缓冲），
`UB_Y`/`UB_DY`/`UB_PROD`/`UB_TMP`（V1 分段），`UB_NZP` 16.25 KB（融合 VF 输出，
65 行），`UB_SFMG`、`UB_LSE_0/1`。

pre/post 与主循环靠 `sync_all` 完全隔开，**直接从 0 地址复用主循环的 UB 与
mutex_id**，不额外占空间。

## 5. 核心优化

### 5.1 L0A/L0B/L0C 双缓冲（收益最大，135 us）

五个 matmul 串行复用同一块 L0A/L0B 时，`pl.move` 填充期间 mac 只能干等。
每个缓冲 64 KB、`[128,128]` fp16 占 32 KB，正好放两槽；把所有 matmul 的
`.current()` 改成 `.next()` 即可。

acc（L0C）同理：fixpipe 写出期间下一个 matmul 被阻塞。

> **只让 mm1/mm2/mm3 轮转 acc。** mm4/mm5 拿的是 `dk_acc`/`dv_acc` 这两个
> L0C 驻留累加器，轮转会丢累加值。

**诊断方法**：算 `kernel_time − max(各流水忙时)`。若缺口大且无流水接近 1.0，
就是等待而非工作量 —— 优先查双缓冲，而不是试图减少计算。

### 5.2 融合 cast + ND2NZ VF

独立的 `pl.move` 做 ND→NZ 转换跑在 **vector 计算流水**（`aiv_vec_ratio`），
不是 MTE3。它曾是与 ASC 差距最大的单项（+127 us）。折进产生数据的 VF：

```python
vf.store_align(nzp_ub, vregCastRes, pregB16,
               data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
               block_stride=(VECTOR_BASEM * 16) * 2 // 32 + 1,   # 65
               repeat_stride=1, post_update=True)
```

一条 `vsstb` 同时完成 cast 与重排，对应 AscendC 的
`CastTransdataDeconflict`。`block_stride` 末尾 **+1** 是 bank 错开填充，
UB tile 声明 65 行并配 `compact=2`（RowPlusOne）。

**尾块的正解**（此前两次失败都在这里）：

```python
pl.set_validshape(nzp_ub, [VECTOR_BASEM, N])   # 固定 64，源跨距 65-64=1
pl.insert(l1_tile, nzp_ub, [row_offset, 0])
pl.set_validshape(nzp_ub, [NZP_ROWS, N])       # 必须恢复 65
```

不能按 `halfS1RealSize` 收窄：源跨距会变成 `65-rows`，与 VF 固定按 65 写入
错位。两个子核 insert 到 `rowOff` 0/64 各占 64 行互不重叠，超出 `s1Real` 的
脏行只落在本子核半区内，由 matmul `valid_shape` 与 store 裁掉。

### 5.3 kernel 内 pre/post

AscendC 把三段融进**一个** kernel（`entry_regbase.h`:
`opPre.Process()` → `op.Process()` → `opPost.Process()`，msprof 也只见一个
kernel），所以 206.6 us 本就含 pre/post。把清零丢给 host `torch.zeros`、
cast 丢给 `.half()` 是在测更小的活。内联后开销 +9.6 us。

- **pre**：清零 dq/dk/dv 三个 fp32 workspace（主循环用 atomicAdd 累加）
- **post**：fp32 → `muls(scale)` → cast fp16。**只有 dQ/dK 乘 scale，dV 不乘**
  （对应 ASC 的 `qkvIdx < 2`）

> **scale 必须在矩阵乘之后施加。** 虽然
> `scale*(dS@K) == (scale*dS)@K` 数学等价，但提前乘进 dS 会让 fp16 舍入发生在
> 不同量级上，训练中误差会累积。改到 post 后精度实测从 6.1e-05 提升到 3.05e-05。

**chunk 取值：不要把 UB 吃满。** 并行度比单次搬运大小重要：

| pre chunk | UB | 块数 | N=1 耗时 |
|---|---|---|---|
| 496 行 | 248 KB | 17 | 201.3 us |
| 128 行 | 64 KB | 64 | **193.6 us** |
| 48 行 | 24 KB | 171 | 192.8 us |

pre=496 时 8192 行只切出 17 块，72 个核有 55 个空跑。post chunk 影响很小
（165→32 仅差 1.3 us），因为它每轮都有 load/cast/store 三段可重叠。
定稿 `PRE_CHUNK_ROWS=128` / `POST_CHUNK_ROWS=64`（depth 2 双槽，96 KB）。

### 5.4 Swizzle 分核（大 N 的 L2 复用）

线性切分让 36 核同时落在 32+ 个不同 head 上，工作集撑爆 L2（128 MB）：

| N | 同时活跃 head | K+V 工作集 |
|---|---|---|
| 1 | 1 | 4 MB |
| 32 | 32 | 128 MB |
| 64 | 36 | 144 MB |

症状：归一化 mte2 从 93.9 涨到 227.5 us，cube/vec 的 `mte2_ratio` 双双 0.72，
而 mac 反而从 0.938 掉到 0.733 —— 退化成 HBM 带宽受限。

AscendC 用 swizzle 应对（`M/N_SWIZZLE_SIZE = 32768`，`common.h:67`，
条件 `enableSwizzle = (isExceedL2Cache || isLargeInvalidBlk) && blockOuter == aicNum`）。

本实现按 **s2 列跨核发放**：核 c 领第 c, c+coreNum, … 列（一列 = 一个
`(b,n2,s2o)` 组 = `s1Outer` 个连续块）。同核在列内连续处理同一 s2 块，故
**K/V 单次载入与 dK/dV 的 L0C 驻留累加都保住**，而活跃核锁在同一 head 内。

副产品：`need_load_kv` / `dkv_acc_flag` / `next_s2_same` 从"比较相邻块组号"
简化成"看列内位置 `localIdx % s1Outer`"，且天然覆盖"本核首块必须覆写"
（列首即 `localIdx==0`）—— 漏掉这条时 L0C 残留值污染过 dK/dV（max_diff 57）。

**判据必须算全部张量**，不能只算 K+V：只算 K+V 时 N=32 恰好等于 128 MB 被判
为不超，但实测 swizzle 快 34%。每 head 约 22 MB（q+dy+y 读 fp16、dq 写 fp32、
k+v 读 fp16、dk+dv 写 fp32）。另外列数 < `2*coreNum` 时不开（N=1 只有 64 列，
核 0 拿 2 列而核 35 拿 1 列，2 倍失衡反而慢 14%）。

**阈值取 `2 * L2` 而非 `1 * L2`**：实测拐点是工作集 264 MB（N=12），不是刚超
128 MB 的 N=6，详见 2 节的表。

### 判据落点：tilingkey bit4，不是 tiling data 字段

host 在 `TilingFunc` 里按输入 shape 算出 `swizzle`，OR 进 tilingkey：

```
bit0-1 = template, bit2 = layout, bit3 = hasAttenMask, bit4 = swizzle
```

kernel 侧 `swizzle = TilingKeyField(bits=1, values=[0, 1])`，函数体内当**编译期
常量**用（`set_const_info(tiling, coreNum, swizzle)`），两条分核路径各自编译成
独立分支，运行期无判断开销。tilingkey 组合数从 16 升到 32，可在生成的
`FlashAttnGradTilingKey_tilingkey.h` 里查到 `// bit:4-4 ASCENDC_TPL_UINT_DECL(swizzle, ...)`。

host 计数活跃 head 时**必须用 `aicNum`，不能用 `blockDim`**：后者是
`CalcTschBlockDim(aivNum, aicNum, aivNum)` 的混合调度值，而 kernel 里
`pl.get_block_num()` 返回物理 cube 核数（36）。

### 判据依赖 S1/S2 的方向（S1 != S2 泛化）

**功能上 S1 与 S2 本就完全独立**：块网格是 `s1Outer x s2Outer` 两个独立维度，
尾块由 `s1CvTail`/`s2CvTail` 各自处理，host 也分别校验 q.S 与 k.S（不要求相等）。
实测 30 个不对称 shape（含 1:64 与 64:1 极端比、S1/S2 < 128 不足一块、
S1=1、S2=1、双非对齐、以及与 D=192/GQA/多核的组合）全部 PASS，无需改动。

**需要改的是 swizzle 判据** —— 它原先只按工作集大小判断、与方向无关，而在
S1 != S2 时会明显误判。swizzle 换来的是「同核连续处理一整列 `s1Outer` 个块」
（K/V 只载一次、dK/dV 在 L0C 连续累加），付出的是分配粒度变粗（一列 =
`s1Outer` 块）。这笔收支**取决于列有多长**，所以判据必须看方向：

| 方向 | 阈值 | 依据 |
|---|---|---|
| `s2Outer > s1Outer`（S2 更长） | 恒不开 | 实测 5/5 全负；318MB / 404MB 工作集仍是线性胜 |
| `s1Outer >= 2*s2Outer`（S1 更长） | 1.25x L2 | 分界落在 138MB（负）与 172MB（正）之间 |
| `s1Outer == s2Outer`（对称） | 2x L2（原值） | N=10 在 220MB 仍偏好线性，不能降 |

S2 更长时必须是一条**独立的方向判据**，不是调阈值能解决的：那几个点工作集
已达 318/404MB，远超任何阈值却仍然线性更快。原因是列长只有 `s1Outer` 块、
复用窗口短，而列数 `b*n*s2Outer` 很大，线性切分本身就已经落在同一 head 内。

实测数据（36 核 D=128 B=1，linear/swizzle us）：

| 形状 | 工作集 | linear | swizzle | 最优 |
|---|---|---|---|---|
| N8 1024/8192 | 106MB | 237.2 | 250.3 | linear |
| N24 1024/8192 | 318MB | 723.1 | 728.0 | linear |
| N32 512/8192 | 404MB | 629.3 | 632.1 | linear |
| N12 8192/1024 | 138MB | 330.0 | 353.0 | linear |
| N16 8192/512 | 172MB | 277.4 | 263.7 | **swizzle** |
| N20 8192/1024 | 230MB | 781.8 | 583.1 | **swizzle** |
| N24 8192/1024 | 276MB | 994.6 | 698.5 | **swizzle** |
| N32 8192/512 | 344MB | 725.9 | 523.4 | **swizzle** |

全部 29 个实测点上：原判据（方向无关 2xL2）误判 7 个、累计代价 67.7%；
新判据误判 2 个（均为 ±5~13% 的近平局）、累计代价 18.0%。
1.1x~1.4x 是同一个平台（误判数都为 2），取中点 1.25x。

**`decide_swizzle()` 与 host 必须逐位一致**，否则 host 派发的二进制与测试
验证的那份不是同一个（不会报错，只会静默测错对象）。已用 21632 + 28800 个
组合（含 GQA）交叉校验，零分歧。顺带修掉一个既有缺陷：kernel 侧
`decide_swizzle` 原先不按 G 摊薄 KV 字节，而 host 有 `/ gSize`，GQA 下
19 个 shape 判据不一致 —— 现已加上 `g_size` 形参。

大 S 下 `S2 = 1` 时 dQ/dK **数学上恒为 0**（单个 key 的 softmax 给出 P == 1，
故 `sfmg == dP`、`dS = P*(dP - sfmg) == 0`）。aclnn 用例原先的「全零即认为
kernel 没写」守护会把这个正确结果误报为失败，已改成「输出为零但 golden 非零」
才告警。

判据在 kernel 与 host 两处各有一份实现，**必须同步改**：
`op_kernel/flash_attn_grad.py` 的 `decide_swizzle()`（JIT 路径与测试用）
和 `op_host/flash_attn_grad_tiling.cpp` 的 C++ 门控（AOT 路径用）。
调试时可用 `FAG_FORCE_SWIZZLE=0/1` 绕过判据强制指定分支。

## 6. LSE 接口变更

不再输入 `softmax_max` / `softmax_sum` 两个张量，改为单个 `softmax_lse`，
形状 **`[B, N, S1]`** fp32。约定对标
[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)：

```
lse = max + log(sum(exp(s - max)))      # 自然对数
P   = exp(S * scale - lse)              # 一步到位，行和恒为 1
```

与原式 `exp(s-max)/sum` 数值差异 6e-08。反向少一条 `vf.div`、少读一个 GM
张量，UB 也从四块降到两块（243 KB → 241.5 KB），实测略快 1 us。

```python
pl.load(lse_ub, tensor_lse,
        [runInfo.boIdx, runInfo.n2oIdx, loadBase], order=[2, 0])
```

## 7. 踩过的坑

| 现象 | 根因 |
|---|---|
| dK/dV 错 0.61/0.57，dQ 正确 | 别名视图用了不同 `mutex_ids`，框架不知它们是同一块内存，互锁失效。**同地址视图必须共用 mutex_id**（对应 AscendC 的 `Get()`/`GetPre()`） |
| 尾块 dQ/dK/dV 全错（0.2~nan） | `set_validshape` 收窄到 `rows` 改变了源跨距，与 VF 固定按 65 写入错位；且 insert 后忘记恢复 65 |
| workspace 残留 NaN 在高位列 | **一个 vreg 只装 64 个 fp32**，`[rows,128]` 每行需要**两条** store（offset 0 和 +64）。按行判定 NaN 会看不见（缺口在列方向），要逐元素并同时报行和列 |
| B2N2 / S1=384 的 dQ 错 1.3/4.2 | 手工软流水预取是条件执行的，`next()` 调用次数随分支变化导致 `current()` 槽位错位。改成**每轮无条件轮转**，重叠交给框架 |
| BNSD 用「倒置 stride」读出错位数据且不报错 | `make_tensor` 的 `stride` 不支持任意跨距，非紧凑轴序被**静默忽略**、按紧凑跨距读。改用该布局自身的紧凑视图（见 3 节） |
| 3 槽 L1 复用比不复用还慢 | writer 与 reader 在环上挨着，共享 mutex 的阻塞超过省下的 GM 搬运。加到 4 槽才转正 |
| dS/P 共享 L1 加双缓冲零收益 | 假设错了：停顿不在 CV 反向等待，而在 L0 层。已回退 |
| **D=192 时 dV 全错（1.12）而 dQ/dK 正确** | `ds_l1/p_l1/ds_t_l1/p_t_l1` 四处仍写死 `L1_DS/L1_P`（0x20000/0x28000），没走分档地址表 `L1_DS_ADDR[d_align]`。192 档每块 48KB，这两个常量正好落在 **Q 槽 2 / dO 槽 0** 内部，P 被 dO 的载入覆写。P 只喂 MM5，故只有 dV 错 —— 症状把人往 D 分块方向引，实际是**地址表漏改**。新增分档表时要 grep 所有旧常量名的残留引用 |
| **Dv != 128 时 dQ/dK/dV 全错，恰好 Dv=128 才对** | sfmg 的 VF 里 `astype(ZERO/ONE)` 的去交错是**跨整段 128 个元素**的，而一段不总等于一行：`dv_align=128` 时一段恰好一行，打乱只发生在行内，行和不受影响（故 Dv=128 一直是对的）；`dv_align=192` 时一行是 1.5 段，去交错把元素搬到**另一行**，行和全错。修法是写回时用 `dist=pl.StoreDist.INTLV_B32` 一条指令交错还原（顺带比原来两条 store 更快）|

**调试方法**：workspace 初值填 **NaN 而非 0**（否则"清成 0"与"从未触碰"无法
区分），并分段验证（只跑 pre，注释掉主循环与 post）。探针见
`/root/.claude/jobs/c66577db/tmp/check_pre_only.py`、`dump_nd2nz.py`。

**pypto 语法约束**：内联函数只能在末尾顶层语句 `return`；分支赋值无法 phi
合并；bool 不能参与算术。需要运行期二选一时用算术选择：

```python
def sel(flag, whenOne, whenZero):
    return whenZero + flag * (whenOne - whenZero)
```

## 8. 文件与用法

代码位于 `attention/flash_attn_grad/`：

| 文件 | 说明 |
|---|---|
| `op_kernel/flash_attn_grad.py` | kernel 实现（~1530 行），含 `FlashAttnGradTilingData` / `FlashAttnGradTilingKey` / `decide_swizzle()` |
| `op_kernel/test_flash_attention_score_grad_pypto.py` | 13 个 shape 用例（含 S1/S2 尾块）+ `CASE_FILTER=GQA` 的 11 个 GQA 用例；带 2 分钟执行看护 |
| `op_kernel/prof_pypto.py` | msprof 各流水占用率，支持 `B/N/G/S1/S2/CORES/FAG_LAYOUT` 环境变量（N 为 N2） |
| `op_kernel/bench_fag_asc_vs_pypto.py` | 与 AscendC 端到端对比 + 正确性交叉验证 |
| `op_host/flash_attn_grad_tiling.cpp` | tiling：workspace 尺寸、swizzle 门控、tilingkey 拼装 |
| `op_host/flash_attn_grad_tiling_check.cpp` | 参数校验。**文件名必须含 `_tiling`**，tiling 对象库按 `*_tiling*.cpp` glob 收源文件，原名 `_check.cpp` 会漏编，链接期报 `undefined symbol CheckParams` |
| `examples/test_aclnn_flash_attn_grad.cpp` | aclnn 单算子调用 + host fp32 golden |

### JIT 路径（快速迭代）

```bash
export PYPTO=/usr/local/Ascend/cann-9.2.0/python/site-packages
export PY=/root/.conda/envs/zyj/bin/python3
cd op_kernel

PYTHONPATH=$PYPTO $PY test_flash_attention_score_grad_pypto.py   # 全部用例
CASE_FILTER=11 PYTHONPATH=$PYPTO $PY test_...py                  # 单个用例
FAG_FORCE_SWIZZLE=1 PYTHONPATH=$PYPTO $PY test_...py             # 强制分核方式
FAG_LAYOUT=BNSD PYTHONPATH=$PYPTO $PY test_...py                 # 切布局(BSND/BNSD)
CASE_FILTER=GQA PYTHONPATH=$PYPTO $PY test_...py                 # 11 个 GQA 用例
B=1 N=32 S1=8192 S2=8192 PROF_OUT=/tmp/p PYTHONPATH=$PYPTO $PY prof_pypto.py
PYTHONPATH=$PYPTO $PY bench_fag_asc_vs_pypto.py                  # 对比 ASC
```

### AOT 路径（对标 ASC 交付件）

```bash
cd <repo root>
export PATH=/root/.conda/envs/zyj/bin:$PATH          # 需 cmake >= 3.18
export Python3_ROOT_DIR=/root/.conda/envs/zyj \
       Python3_EXECUTABLE=/root/.conda/envs/zyj/bin/python3
bash build.sh --op_debug_config dump_cce --soc=ascend950 --ops=flash_attn_grad --pkg
cd build && ./cann-ops-transformer-custom_linux-x86_64.run
```

两个环境变量不是可选的：默认会挑到 `/root/miniconda3/bin/python3.13`（无
`pypto_pro`），于是 `require_pypto_pro` 提前返回、整个算子被跳过，最终报成
`CMake Error: No SOURCES given to target: cust_proto` —— 报错位置和真因毫无关系。
`--ops` 的取值是**目录名** `flash_attn_grad`；写错（如 `flash_atten_grad`）会因
匹配不到算子而报出同一条 `cust_proto` 错误。

运行 aclnn 用例：

```bash
CANN=/home/kjc/kjc_cann_0701/cann-9.1.0
VEND=$CANN/opp/vendors/custom_transformer
g++ -std=c++17 -O2 examples/test_aclnn_flash_attn_grad.cpp -o /tmp/fag_aclnn \
  -I$CANN/include -I$VEND/op_api/include \
  -L$CANN/lib64 -L$VEND/op_api/lib -lascendcl -lnnopbase -lcust_opapi
export LD_LIBRARY_PATH=$VEND/op_api/lib:$CANN/lib64:$LD_LIBRARY_PATH
export ASCEND_CUSTOM_OPP_PATH=$VEND
FAG_LAYOUT=BNSD FAG_B=1 FAG_N=4 FAG_G=2 FAG_S1=256 FAG_S2=256 FAG_CHECK=1 /tmp/fag_aclnn
```

`FAG_N` 是 N1，`FAG_G` 是 G（N2 = N1/G）。`FAG_CHECK=0` 跳过 host golden（`O(B*N*S1*S2*D)` 三重循环，大 shape 太慢），
只校验执行成功与输出量级非零。加
`ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=1` 可看到 tiling 日志里
实际选中的 `swizzle=`。

性能测试固定用 S=8192。

## 9. 状态与后续

已完成并验证：

- **JIT 路径**：13 个用例全 PASS（含 S1/S2 尾块），`FAG_FORCE_SWIZZLE=0` 与 `=1`
  两条分核路径各自跑满 13 例。对 AscendC 交叉验证通过。
- **AOT 路径**：按 ASC 交付件形态编出 bf16/fp16 两个 MIX 核 ELF（各 316 KB），
  导出 `aclnnFlashAttnGrad` / `aclnnFlashAttnGradGetWorkspaceSize`；aclnn 用例
  PASS，dQ 6.14e-05 / dK 6.12e-05 / dV 6.61e-05，rms 与 golden 前 5 位有效数字
  一致（dQ 1.503249e-03 vs 1.503265e-03）。workspace 随 shape 正确缩放
  （N=1 16.38 MB，N=8 112 MB）。
- **swizzle 判据**：已改为 tilingkey bit4，阈值校准到 `2 * L2`；N=1/8/10/12/16/24
  六个测点，host 判据全部选中实测更快的那条路径。
- **布局泛化**：BSND / BNSD 均通过。JIT 侧 2 布局 x 2 分核 = 4 种组合各跑满
  13 个 shape 用例，三项 max_diff 恒为 6.10e-05（与布局无关）；AOT 侧
  `FAG_LAYOUT=BSND/BNSD` 各跑 aclnn 用例（N=2，真实走到 head 轴）均 PASS。
  host 日志确认 BSND 得 `view=[1,S,4,D] coef=(1,0,1)`、BNSD 得
  `view=[4,S,1,D] coef=(4,1,0)`，且两者 `tilingKey` 相同 —— 布局未占 tilingkey。
  性能无回归（S=8192，36 核，task_duration）：

  | N | BSND | BNSD | 差异 |
  |---|---|---|---|
  | 1 | 192.96 | 193.32 | +0.2% |
  | 12 | 2339.10 | 2385.05 | +2.0% |

  索引由编译期常量变成一次乘加，开销在噪声内。

- **GQA（N1=N2*G）**：JIT 侧 11 个 GQA 用例（G 取 2/3/4/8/12，含 N2=1、
  奇数 G、S1/S2 尾块、多核大 shape）在 2 布局 x 2 分核 = 4 种组合下全 PASS；
  AOT 侧 aclnn 用例 G=1/2/4 均 PASS，且 dK 的 rms 随 G 线性增长
  （1.50e-03 / 2.12e-03 / 3.00e-03），印证 G 份贡献确实累加到了同一 KV head。
  host 日志确认 G=4 时 `viewKV` 的 head 数由 8 降到 2、workspace 相应缩小。

  GQA 用例的 max_diff 比 MHA 大（G=8 时 dK 达 9.77e-04 vs MHA 6.10e-05），
  但全部**精确等于某个 fp16 ULP**（2^-10 ~ 2^-14），即输出以 fp16 存储的量化
  下限，非算法误差 —— 算法错误不会精确落在 ULP 边界上。G 越大 dK/dV 量级越大、
  落入更高的 fp16 指数区间，ULP 随之变大。容差 5e-2 仍有约 51 倍余量。

性能：N=1 0.93x、N=32 0.96x、N=64 0.94x（ASC 为 1.0 基线）。

> GQA 的性能数未采集：本机 msprof 的解析任务全部失败且**MHA 基线同样失败**，
> 属环境问题（`set_env.sh` 会把 PATH 覆盖成不含 `/usr/bin` 的值，parser 调不到
> 系统工具；原 PATH 里还残留已删除的 cann-9.1.0 路径）。跑 profiling 前需
> `export PATH=$PATH:/usr/bin:/bin`。GQA 是纯索引改动、不动片上存储，预期无
> 回归，但**尚未实测**。

**推送受阻**：token 认证正常（Julius47，全 scope，对 fork 是 Owner），但 gitcode
返回 `<CH.00905403> You are not allowed to push code to this project` 403，
新分支同样如此 —— 需在平台侧解除推送限制。

可继续的方向（当前无明显瓶颈，mac 已 0.941 近饱和）：

- vec 计算 127.9 vs ASC 85.3 us、vec mte2 66.3 vs 24.6 us 仍偏高；后者部分来自
  post 阶段读回 fp32 workspace
- cube mte2 94.0 vs 69.6 us
- 拐点邻域（N=11 附近）未测，阈值可再细调
- `flash_attn_grad_binary.json` 目前只产 bf16/fp16 两个变体，两个 swizzle 分支
  是各自出核还是派发期选择，尚未确认

进一步优化 vector 侧收益有限，除非同时降低 cube 侧工作量。
