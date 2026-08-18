# FlashAttentionScoreGrad pypto-pro 开发文档

对标 `arch35/flash_attention_score_grad_kernel.h`（AscendC BN2GS1S2 模板）的
pypto-pro 重写。变量名与代码结构尽量跟随 AscendC 原版。

## 1. 范围与配置

| 项 | 取值 |
|---|---|
| 布局 | BSND（非 TND） |
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

| N | ASC | 线性切分 | swizzle | 判据 | 最终 |
|---|---|---|---|---|---|
| 1 | 206.7 | **193.5** | 220.8 | 0 | 0.94x |
| 2 | — | **378.0** | 427.3 | 0 | |
| 4 | — | **747.7** | 857.6 | 0 | |
| 8 | 1696.8 | 1630.5 | **1602.6** | 1 | 1.04x |
| 16 | — | 4103.6 | **3438.2** | 1 | |
| 32 | 6859.0 | 10106.1 | **6555.6** | 1 | 0.96x |
| 64 | 14294.8 | 22215.7 | **13395.3** | 1 | 0.94x |

判据切换点（N=6）与实测拐点（N=8）之间无反例。

### 精度

对 AscendC 结果直接比对（**不是 CPU golden**）：max/sum 版 dQ/dK 3.05e-05、
dV 6.1e-05；LSE 版三项均 6.1e-05。差异来自舍入次序，在 fp16 精度内。

## 3. 数据流与内存布局

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

## 4. 核心优化

### 4.1 L0A/L0B/L0C 双缓冲（收益最大，135 us）

五个 matmul 串行复用同一块 L0A/L0B 时，`pl.move` 填充期间 mac 只能干等。
每个缓冲 64 KB、`[128,128]` fp16 占 32 KB，正好放两槽；把所有 matmul 的
`.current()` 改成 `.next()` 即可。

acc（L0C）同理：fixpipe 写出期间下一个 matmul 被阻塞。

> **只让 mm1/mm2/mm3 轮转 acc。** mm4/mm5 拿的是 `dk_acc`/`dv_acc` 这两个
> L0C 驻留累加器，轮转会丢累加值。

**诊断方法**：算 `kernel_time − max(各流水忙时)`。若缺口大且无流水接近 1.0，
就是等待而非工作量 —— 优先查双缓冲，而不是试图减少计算。

### 4.2 融合 cast + ND2NZ VF

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

### 4.3 kernel 内 pre/post

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

### 4.4 Swizzle 分核（大 N 的 L2 复用）

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

## 5. LSE 接口变更

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

## 6. 踩过的坑

| 现象 | 根因 |
|---|---|
| dK/dV 错 0.61/0.57，dQ 正确 | 别名视图用了不同 `mutex_ids`，框架不知它们是同一块内存，互锁失效。**同地址视图必须共用 mutex_id**（对应 AscendC 的 `Get()`/`GetPre()`） |
| 尾块 dQ/dK/dV 全错（0.2~nan） | `set_validshape` 收窄到 `rows` 改变了源跨距，与 VF 固定按 65 写入错位；且 insert 后忘记恢复 65 |
| workspace 残留 NaN 在高位列 | **一个 vreg 只装 64 个 fp32**，`[rows,128]` 每行需要**两条** store（offset 0 和 +64）。按行判定 NaN 会看不见（缺口在列方向），要逐元素并同时报行和列 |
| B2N2 / S1=384 的 dQ 错 1.3/4.2 | 手工软流水预取是条件执行的，`next()` 调用次数随分支变化导致 `current()` 槽位错位。改成**每轮无条件轮转**，重叠交给框架 |
| 3 槽 L1 复用比不复用还慢 | writer 与 reader 在环上挨着，共享 mutex 的阻塞超过省下的 GM 搬运。加到 4 槽才转正 |
| dS/P 共享 L1 加双缓冲零收益 | 假设错了：停顿不在 CV 反向等待，而在 L0 层。已回退 |

**调试方法**：workspace 初值填 **NaN 而非 0**（否则"清成 0"与"从未触碰"无法
区分），并分段验证（只跑 pre，注释掉主循环与 post）。探针见
`/root/.claude/jobs/c66577db/tmp/check_pre_only.py`、`dump_nd2nz.py`。

**pypto 语法约束**：内联函数只能在末尾顶层语句 `return`；分支赋值无法 phi
合并；bool 不能参与算术。需要运行期二选一时用算术选择：

```python
def sel(flag, whenOne, whenZero):
    return whenZero + flag * (whenOne - whenZero)
```

## 7. 文件与用法

| 文件 | 说明 |
|---|---|
| `flash_attention_score_grad_pypto.py` | kernel 实现（1528 行） |
| `test_flash_attention_score_grad_pypto.py` | 13 个 shape 用例，含 S1/S2 尾块；带 2 分钟执行看护 |
| `prof_pypto.py` | msprof 各流水占用率，支持 `B/N/S1/S2/CORES` 环境变量 |
| `bench_fag_asc_vs_pypto.py` | 与 AscendC 端到端对比 + 正确性交叉验证 |

```bash
export PYPTO=/usr/local/Ascend/cann-9.2.0/python/site-packages
export PY=/root/.conda/envs/zyj/bin/python3

PYTHONPATH=$PYPTO $PY test_flash_attention_score_grad_pypto.py      # 全部用例
CASE_FILTER=11 PYTHONPATH=$PYPTO $PY test_...py                     # 单个用例
B=1 N=32 S1=8192 S2=8192 PROF_OUT=/tmp/p PYTHONPATH=$PYPTO $PY prof_pypto.py
PYTHONPATH=$PYPTO $PY bench_fag_asc_vs_pypto.py                     # 对比 ASC
```

性能测试固定用 S=8192。

## 8. 状态与后续

13 个用例全 PASS（含尾块），对 AscendC 交叉验证通过。N=1 0.93x、N=32 0.96x、
N=64 0.94x。

已提交在 `FAG_pro` 分支。**推送受阻**：token 认证正常（Julius47，全 scope，
对 fork 是 Owner），但 gitcode 返回
`<CH.00905403> You are not allowed to push code to this project` 403，
新分支同样如此 —— 需在平台侧解除推送限制。

可继续的方向（当前无明显瓶颈，mac 已 0.941 近饱和）：

- vec 计算 127.9 vs ASC 85.3 us、vec mte2 66.3 vs 24.6 us 仍偏高；后者部分来自
  post 阶段读回 fp32 workspace
- cube mte2 94.0 vs 69.6 us
- N=8 是 1.04x，为判据切换点附近的唯一劣势档位，可细调阈值

进一步优化 vector 侧收益有限，除非同时降低 cube 侧工作量。
