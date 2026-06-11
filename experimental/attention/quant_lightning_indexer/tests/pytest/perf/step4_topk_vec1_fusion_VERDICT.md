# QLI · TopK→Vec1 HistHigh 融合 — 真机单变量拆解 · 最终裁决

> 日期:2026-06-11。方法:**真实完整 prefill op** 上的单变量隔离实验(full cube matmul,
> torch.npu.Event,repeat=20/warmup=3~5)。取代了 `step4v_results.md` 的纯 V 微核路线
> (那条路被 camodel 第二条 unknown 指令 `pc=0x9000d16ba0` 堵死,致命、内核不执行)。
> 也取代了 `step4_topk_vec1_fusion_analysis.md` 的"饱和度手挥"结论 —— 本文用**拆解后的代价**
> 给出为什么融合回归,精确到 us。

---

## 0. 一句话结论

**把 TopK 的 HistHigh(高字节直方图)融进 Vec1 主循环 = 净回归 +1.9%~3.0%,无论用攒批拷贝还是零拷贝实现。**
节省侧只有 **−143us**,但把同等直方图工作落到 99.8% 饱和的 Vec1 binder 上要 **+277~349us**。
**根因是数据布局**:Vec1 每个 s2-block 只产出 128 个有效 key、行间隔 256 元素(bank gap),
逼你二选一 —— 攒批拷贝(+281us)或 128 宽粒度惩罚(+277us);二者都远超 143us 的节省。

---

## 1. 实验设计(单变量隔离)

把"融合净收益 = 节省 − 代价"拆成两个**可独立测量、互不耦合**的真机实验,各只动一个变量,
都跑在真实 op 上(饱和态 = 真实 binder 条件),用 `time_bench.py` 测 wall。
宏脚手架在 `quant_lightning_indexer_kernel.h` 顶部(`EXP_R_SKIP_TOPK_HISTHIGH` /
`EXP_C_VEC1_HISTHIGH` / `EXP_C_COPYONLY` / `EXP_C2_NOCOPY`),默认全关。

| 实验 | 改动 | 测的是 | 输出 |
|---|---|---|---|
| **Exp-R** | TopK 跳过 HistHigh,Vec1 不动,直方图留脏 | 把 HistHigh 从 TopK 尾巴搬走能**省**多少 wall | 错(仅测时序) |
| **Exp-C** | Vec1 跑攒批 HistHigh(废work),TopK 仍做自己的 | 把 HistHigh 加到 Vec1 的**代价**(攒批+拷贝实现) | 正确 |
| **Exp-C-copyonly** | Exp-C 里只留 UB→UB 攒批拷贝,去掉直方图计算 | 攒批**拷贝**单独的代价 | 正确 |
| **Exp-C2** | Vec1 逐块直接 histogram outBase(VL128 掩码),uint16 running hist,**零拷贝** | HistHigh-in-Vec1 的**最精简(零拷贝)**代价 | 正确 |

---

## 2. 数据(全部 std ≤ 18us,机器同会话无漂移)

| 配置 | 中位数 us | Δ vs 基线 | std |
|---|---|---|---|
| **基线**(clean) | **6969** | — | 17 |
| Exp-R(TopK 去 HistHigh) | 6826 | **−143** | 14 |
| Exp-C(攒批+拷贝) | 7318 | **+349** | 12 |
| Exp-C-copyonly(仅拷贝) | 7250 | +281 | 15 |
| Exp-C2(零拷贝逐块) | 7246 | **+277** | 18 |

基线在本会话测过 6971 / 6969 / 6917(后者 cjl 并发 build 争用),取一致早读 **6969** 为基准。

---

## 3. 拆解与裁决

### 3.1 关键拆解(Exp-C 的 +349 拆开)
```
ΔC(攒批) = +349us
         = 攒批拷贝 +281us (copy-only 实测)  +  纯直方图 +68us (349−281)
```
**"纯直方图只要 +68us" 是攒批摊薄的假象** —— 它便宜,是因为攒到 4096 宽一次性算;
而**凑出这个批的代价就是那 +281us 的 UB→UB 拷贝**。两者绑定,不可分。

### 3.2 零拷贝也救不了(Exp-C2)
绕开拷贝、逐块 128 宽直接 histogram = **+277us ≈ copy-only 的 +281us**。
原因:128 宽 = 2× 直方图迭代 + 逐块 uint16 hist 的 UB load/store + 每块 `__VEC_SCOPE__` 进出开销。
**所以无论攒批(+349)还是零拷贝(+277),把 HistHigh 塞进 Vec1 都要 ~+277~349us。**

### 3.3 净收益
```
净融合 = −ΔR(节省) + ΔC(代价)
       = −143 + 277  = +134us / +1.9%   (零拷贝,最优实现)
       = −143 + 349  = +206us / +3.0%   (攒批实现)
```
**两个数刚好夹住历史脏测 +159us / +2.3%(STATUS.md:43)** —— 同向、同量级,
证明那次回归是**结构性必然**,不是噪声或实现 bug。

### 3.4 为什么同样的直方图在 TopK 里只占 143us、在 Vec1 里要 277us?
- TopK 的 HistHigh 跑在 **684us 的暴露 bubble 尾巴**里 —— 那里有 slack,直方图的一部分
  与其它尾部工作/MTE 重叠,所以移走只省了 143us(≈ 它真正暴露的部分)。
- Vec1 是 **99.8% 饱和的 binder** —— 加的每条 vec 指令几乎 1:1 转成 wall,无可藏;
  再叠加零拷贝的 128 宽粒度惩罚 ⇒ 277us。
- **数据布局是物理约束**:Vec1 每块产 128 有效 key(s2BaseSize=128),
  行距 `UB_BANK_DEPTH_STRIDE=512B=256 元素`(中间 128 是 bank gap)。
  要 256 宽连续就得拷贝;不拷贝就吃 128 宽惩罚。`+68us 的纯批直方图`无法不付拷贝地落到 binder 上。

---

## 4. 还能不能救?(给后续的明确边界)

要让融合转正,唯一的路是**消除数据布局这道约束** —— 让 Vec1 以 256 连续布局产出 key
(去掉 bank-depth gap,或 s2BaseSize 128→256)。这会动 Vec1 输出 tiling + scoreGm 布局 + 可能牵连 cube,
且 **L0C 已在上限**(buffer_analysis:M_BASE=256/S2_BASE=128 已顶满 L0C,不能再涨)。
风险/工作量远大于 1.9% 的潜在收益(还得先证明 256 连续布局本身不引入新代价),**不建议在本 op 上追**。

**Decision-tree Step-2(融合)= NO,现由真机拆解定论(不再是饱和度手挥)。**
TopK 尾部 pass-reduction 仍是唯一杠杆,但两条 radix lever 已 break-even(STATUS.md)⇒ TopK 近地板。

---

## 5. cannsim 路线说明

用户路线 2(只跑 TopK 部分的 log_ca 看 cycle 变化)**已被本真机拆解取代且更优**:
(a) camodel 对本 op 不保真(prefill 含 cube/FFTS,decode 1.85× 偏快;纯 V 微核又撞第二条 unknown 指令致命);
(b) 即便仿真能跑,它只给 HistHigh 的孤立 cycle,**回答不了"这些 cycle 落到饱和 Vec1 上能否藏住"**——
而这正是决定性问题,只有真机饱和态能答。Exp-R/C/C2 就是这一问题的**保真**实证版本。

---

## 6. 复现

```bash
source /tmp/qli_env.sh   # 见 STATUS:修复 harness grep-函数破坏 CANN set_env PATH
# 在 quant_lightning_indexer_kernel.h 顶部取消注释对应 EXP_* 宏(一次一个),然后:
cd ~/ops-transformer_scfa
bash build.sh --pkg --experimental --soc=ascend950 --ops=quant_lightning_indexer,quant_lightning_indexer_metadata -j16
./build_out/cann-ops-transformer-custom_linux-x86_64.run --quiet --force --install-path=/home/w00927905/qli_custom_opp
cp -f build/libopapi_math.so /home/w00927905/qli_custom_opp/vendors/custom_transformer/op_api/lib/  # 装后必补:.run 不带此依赖
cd experimental/attention/quant_lightning_indexer/tests/pytest
python3 perf/time_bench.py --case prefill --repeat 20 --warmup 3
```
各实验 .o 已存档:`perf/step4_fusion_verdict/{baseline,expR,expC,expC2}/`(树内,持久)。
源码脚手架(全部 `#ifdef`,默认关)留在 `op_kernel/arch35/{quant_lightning_indexer_kernel.h,
quant_lightning_indexer_service_vector.h, vf/vf_topk_16_gather.h}`。

---

## 7. 问题2:能否用 s2base=256 / s1base 减半 解决数据布局约束?(2026-06-11 实测)

**用户思路(很准):** M↔S2 互换 —— `mBaseSize 256→128`(s1base 4→2)、`s2base 128→256`,
保持 L0C tile 面积不变([128,256]=[256,128]),绕开"S2_BASE 已顶满 L0C"。
若成立,Vec1 每行就有 256 个连续 key(填满 256 元素行距、无 bank gap)⇒ 零拷贝零粒度惩罚的 256 宽 HistHigh。

### 7.1 实测结论:**不可行 —— 被 Fixpipe 架构硬卡死(非 tiling 旋钮)**
翻 6 个常量(kernel.h 的 `M_BASE_SIZE/S2_BASE_SIZE` + cube 的 `M_BASIC_BLOCK{,_L0}/S2_BASIC_BLOCK{,_L0}`)后
编译**直接失败**,撞上开发者预设的护栏:
```
service_cube.h:362: static_assert(S2_BASIC_BLOCK == S2_BASIC_BLOCK_L0 && S2_BASIC_BLOCK_L0 == 128);
```
读 `Fixp()`(L0C→UB 搬运,fp32 分支 = 本 case)证实根因,代码注释直说
**"S2_BASIC_BLOCK_L0 不为 128 会有问题"**(service_cube.h:384):
- 一行 matmul 结果 = 128 fp32 = **512B = 正好一个 `UB_BANK_DEPTH_STRIDE`**,
  靠 `dualDstCtl=1` + `ndNum=2` 把结果按 M/2 拆到双 bank 搬运(`dstStride=UB_BANK_DEPTH_STRIDE/sizeof`)。
- S2=256 → 一行 1024B = **跨 4 个 bank**,这套 NZ2ND 描述符(`ndNum/srcNdStride/dstNdStride/dualDstCtl`)
  彻底失效;下游 `BatchMulWeightAndReduceSum` 读 `resMm1UB` 的 `qkVLstride`、vec1Out 布局、scoreGm copy-out
  全部依赖这个双 bank 假设。
- 另:host tiling(`quant_lightning_indexer_tiling.cpp:879`)硬编码 `s2BaseSize=128` 算 scoreGm workspace,
  内核改 256 还有 GM 对齐/溢出风险。

⇒ **s2base=256 不是改常量,而是要重写 Fixpipe 的 L0C→UB NZ2ND 搬运(支持 1024B/4-bank 行)
+ 整条 cube→Vec1 UB marshaling(resMm1 stride、MulWeightAndReduceSum、vec1Out、scoreGm)+ host workspace。**
属多百行、跨 cube/vec/host 的深度改造,且 L0B 会涨到正好 64KB(顶 L0B 上限)。

### 7.2 就算硬把它重写完,收益也只是 break-even(不值)
对 Exp-C2(零拷贝 128 宽 = +277us)做结构外推:s2base=256 让块数从 ~136 减半到 ~68,
直方图迭代/key-load/逐块 hist 读写/scope 开销**几乎全部减半** ⇒ 代价约 **+277/2 ≈ +140us**。
净融合 = −143(节省)+ 140 ≈ **−3us ≈ break-even(噪声内)**。
**且这还没算 tiling 改造自身的反向开销**:s1base 减半 ⇒ 外层 tile 翻倍 ⇒ prologue/cross-core sync 开销翻倍;
cube matmul 的 M 维 256→128 可能降 cube 利用率(cube 与 vec 锁步饱和,cube 变慢直接抬 wall)。
这两项很可能把 break-even 推回**轻微回归**。

### 7.3 最终判定
**s2base=256/s1base=2 救不了融合:** 既被 Fixpipe 的 128-only 架构护栏挡死(深度重写、非旋钮),
**且**即使重写,vec 侧最好 break-even、叠加 tiling 自身开销后大概率轻微回归。**期望收益 ≪ 改造成本与正确性风险,不建议追。**
失败的 tiling 改动已回退(源码、装回 baseline .o);备份在 `/tmp/qli_exp/tiling_backup/`。

---

## 8. 用户三问(2026-06-11,代码级回答)

> 背景:本 op 的 TopK 是 **radix-select(基数选择)**,不是排序。对每行 uint16 sortable key
> (`FloatToSortableKey<bf16>`)做:HistHigh(高字节直方图,256 桶)→ FindHighTargetBin(定位第 k 大
> 落在哪个高字节桶)→ HistLow(条件直方图,只统计高字节==目标桶的 key 的低字节)→ FindKth(定出
> 精确第 k 大值)→ FindIdxGT(收 value>kth 的 index)+ FindIdxEQ(补 value==kth 的 index 到正好 K)。
> 全程 O(N) 线性扫描 + 几个 256 桶扫描,**不排序**。代码:`vf/vf_topk_16_gather.h`、`vf/quant_lightning_indexer_topk.h`。

### Q1:为什么不能每 1024 个流式比较(攒 2048 选 1024,再与上一批 1024 merge)?

**答:代码本来就在做"流式 merge top-K",只是 chunk = trunkLen=16384,不是 1024;取大 chunk 是为性能。**
`LITopk<uint16_t>::TopK`(topk.h:129)对 validS2>16K 的情形走多-trunk:每个 trunk 用 `LiTopKVF` 对
**[上一轮 top-K ⊕ 新 trunk]** 做 radix-select 出新的 top-K,再用 `LiTopKGatherVF`(topk.h:150)把选中的
局部 index gather 回全局 index——**这正是你说的"挑 1024,与之前的 1024 比较"**,只是每批 16384。

**为什么不缩到 1024:**
1. **radix-select 已是 O(N) 线性**:把 chunk 缩小**不减少**总扫描量(还是要把每个 key 都 histogram 一遍),
   只会**成倍放大每个 trunk 的固定开销**。
2. **每个 trunk 的固定开销与 chunk 大小无关**:FindHighTargetBin、FindKth 都是**扫 256 桶**的定值开销
   (topk.h 每 trunk 一整套 6-pass),外加一次 `LiTopKGatherVF` 的 index 重映射 gather。
   validS2≈17408 时:trunk=16384 → **2 个 trunk**(2 套固定开销);trunk=1024 → **17 个 trunk**(17 套)。
   ⇒ 1024 流式把固定开销 ×8~17,线性扫描一点没省 ⇒ **更慢**。
3. **若理解为"排序式 merge"(对 2048 partial-sort 选 1024)更糟**:AIV 上排序要 bitonic,
   O(M·log²M) 比较(2048 约 121 阶),×17 批,远超 radix 的线性 histogram;且要带着 index 一起换位。
   radix 用 2 趟字节直方图就定出第 k 大、与 K 无关,本质上比"排序 top-K"省得多。
4. **chunk 上限由 UB 决定**:trunkLen=16384 uint16 ≈ 32KB(`GetSharedTmpBufferSize`),已接近能放下的最大,
   选它就是为了把 merge 次数压到最少。
**结论:流式 1024 既不省线性扫描、又把 per-trunk 固定开销和 gather 次数 ×8~17,是更慢的方向。**

### Q2:融合为什么只能融 HistHigh,其余 pass 为什么不能融?

**答:因为只有 HistHigh 的输入"仅仅是 key 本身";其余每个 pass 都依赖一个"必须先看完所有 key 才能算出的中间量",无法随 Vec1 边产边算。** 看依赖链(`LiTopKVF`,vf_topk_16_gather.h:378-395):
- **HistHigh(可融)**:对每个 key 的高字节做 `Histograms ACCUMULATE`——纯逐元素累加,key 之间独立。
  Vec1 每个 s2-block 产出一批 key 时就能顺手累进 256 桶 ⇒ **可流式融入主循环**。
- **FindHighTargetBin(不可融)**:要扫**完整**的 HistHigh(256 桶累计跨过 bottomK 才知目标桶),
  必须等所有 key 都 histogram 完 ⇒ 只能在尾部跑。**(深入,见下方 Q2-补)**
- **HistLow(不可融)**:`HistogramsLowVFImpl`(:156)先 `Compare EQ(高字节, idxHigh)` 再只统计命中的低字节——
  它**条件依赖 idxHigh = FindHighTargetBin 的输出**,而那个又依赖完整 HistHigh。Vec1 边产 key 时
  **还不知道目标桶是谁**,无法决定统计哪些 ⇒ 不可融。
- **FindKth(不可融)**:依赖完整 HistLow。
- **FindIdxGT / FindIdxEQ(不可融)**:要拿 kth 值去**重扫所有 key**挑 index,kth 在管线最末才有。
**一句话:依赖链 HistHigh→targetHigh→HistLow→kth→FindGT/EQ 是严格串行的,只有第一环 HistHigh
只吃原始 key、可与 Vec1 产 key 同步;后面每一环都要"先有全量 key 派生出的标量"才能开工,天然不可流式。**
(这也是 §3 实测里唯一被测的可融部分;而 §3 证明:即便只融这一环,代价也 > 收益。)

#### Q2-补:为什么 FindHighTargetBin 必须等"该行所有 s2 方向"算完?(归约 vs 查询)
**核心:HistHigh 是"归约(reduction)",FindHighTargetBin 是"对归约最终结果的查询(query)"。**
- **归约可增量、可融**:计数是可交换可结合的累加,分块/乱序/边产边累都行 ⇒ HistHigh 能摊进 Vec1 产 key 流,
  最后一个 s2-block 产完直方图正好就绪,无额外尾扫。
- **查询不可增量**:FindHighTargetBin 要回答的是**全局序统计量**——第 k 大落在哪个高字节桶(`bottomK=validLen−topK+1`
  累计跨过的那个桶)+ `nextK`(桶内剩余排名)。**序统计量按定义是全部 N 个 key 的函数。**
- **为什么半数据会错**:某桶在只数完前一半 s2-block 时计数=800,后一半可能再来 900 落进同桶→变 1700;
  "第 k 大边界落在哪个桶"会随后续 key 左右移动(多来高分→边界上移,多来低分→下移)。
  **任何基于前缀的目标桶都可能被尚未到来的 key 推翻**,直到最后一个 key 计入才定。等同于"求全局第 1024 大,
  不可能只看前一半就断言"。
- **所以这是一道硬屏障**:`边产key边累(HistHigh,可融) │ 屏障=所有s2计入 │ 查询目标桶(FindHighTargetBin) →
  条件归约HistLow(依赖目标桶) → 查kth → 过滤GT/EQ(依赖kth)`。HistHigh 之后每步要么是"对完整归约的便宜查询"
  (FindHighTargetBin/FindKth,定值256桶),要么是"先拿全局标量才能开工的O(N)重扫"(HistLow/GT/EQ)。
- **能否绕开这个全局统计?** 能——就是 Q1 的流式 merge top-K(维护 running top-K,永不求全局第k大),
  但那是用"很多趟 per-trunk merge"换"一趟全局选择",per-trunk 固定开销 ×8~17 ⇒ 更慢。
  **radix 的本质权衡 = 用"必须等全量、只做一趟全局选择"换掉"不用等、但要很多趟增量合并"。**

### Q3:256 的限制从何而来?与 S2Base=128 不友好,有办法解决吗?

**答:256 = AIV 向量寄存器宽度(256B = 256 个 uint8 lane),是字节直方图指令的天然处理宽度,纯硬件常量。**
代码里直接写死(vf_topk_16_gather.h:368-370):
```
const uint16_t repeatSize8 = 256;   // uint8  每条向量指令 256 lane
const uint16_t repeatSize16 = 128;  // uint16 128 lane
const uint16_t repeatSize32 = 64;   // uint32 64 lane
```
HistHigh 的 `LoadAlign DIST_DINTLV_B8`(:49)一次吃 **256 个 uint16**(512B)、拆出 256 个高字节喂给
`Histograms`(256-lane uint8 op);`vfLoop = validLen/256`。**S2Base=128 个 key/block = 256B = 只够半条
256-uint16 直方图迭代** ⇒ 一个 s2-block 填不满一次 histogram 发射 ⇒ 这就是"256 与 128 不友好"。

**能否解决(三条路,均已在 §3/§7 实测或定性):**
- **(a) 攒 2 个 block 到 256 再 histogram**:要一次 UB→UB 拷贝 ⇒ §3 实测 **+281us**(拷贝即杀手)。
- **(b) 128 宽掩码直方图(VL128)**:不拷贝,但每次发射只用半宽、迭代数翻倍 ⇒ §3 Exp-C2 实测 **+277us**。
- **(c) 让 S2Base=256**(一个 block 正好 256 key,完美对齐):§7 证明被 Fixpipe 的 128-only 架构焊死,
  且需 kernel cube/vec/Fixpipe + host tiling + metadata **5 组件协同重写**(naive 改单组件 → 18× 爆炸),
  最好也只 break-even。
**256 本身改不了(硬件 SIMD 宽度);S2Base=128 是它的一半,这个错位正是融合代价的来源。三条路都"要么实测回归、
要么被架构挡死且收益 break-even" ⇒ 数据布局这道坎对本 op 没有划算的解。**

**三问汇总:**TopK 是 radix-select(非排序),已在做大-chunk 流式 merge(缩到 1024 只会更慢);
依赖链决定只有 HistHigh 可融(其余都要全量 key 的派生标量);256 是硬件向量宽度、与 S2Base=128 错位,
而对齐它的唯一干净办法(S2Base=256)被 Fixpipe 架构挡死。**整体结论不变:本 op 上 TopK→Vec1 融合无划算解。**
