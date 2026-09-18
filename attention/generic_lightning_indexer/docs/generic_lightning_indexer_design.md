# GenericLightningIndexer 算子需求设计

- **算子名**：`GenericLightningIndexer` / `aclnnGenericLightningIndexer`
- **路径**：`attention/generic_lightning_indexer`
- **开发分支**：`generic-lightning-indexer`（基于 `cann/master` `12a3dd661`）
- **阶段**：Step 1（收编前 5 个 indexer + Qwen CLI；为 MSA 预留接口）
- **日期**：2026-09-18

配套分析：

- 模型侧变体：`AR/indexer_analysis/LI_variants_model_impl.md`
- 仓内实现：`AR/indexer_analysis/indexer_fusion_template_strategy.md`
- 变体清单：`AR/indexer_analysis/lightning_indexer_variants.md`

---

## 1. 开发背景

仓内 Indexer 已有 6 个前向算子，骨架高度同构，但各自维护一份 Cube / Vector / TopK：

| 算子 | 服务模型 | 打分流 |
|---|---|---|
| LightningIndexer | DeepSeek V3.2 | ReLU + 加权和 + token TopK |
| LightningIndexerV2 | DeepSeek V4 CSA Indexer | 同上 + 压缩 Key 的 causal 映射 |
| QuantLightningIndexer | V3.2 量化 | 量化 QK + ReLU + 加权和 + TopK |
| QuantLightningIndexerV2 | V4 量化 | 同上 + 压缩 Key + MXFP8/MXFP4 |
| PoolKeyIndexer | GLM 5.3 Flash | ReLU + 加权和 + pool TopK + Expand + 追加尾 token |
| MsaIndexScore | MiniMax M3 | 无激活、无加权和、block Maxpool、**无融合 TopK** |

逐算子 fork 的代价已经量化：仅 LI 家族 + PKI 合计约 77k 行，TopK 文件在 LI 与 PKI 之间几乎逐字复制。Qwen3.8-Flash-Next 的 Compressed Lightweight Indexer 即将接入，若再 fork 第六份，横向改动（950 新 dtype、C2UB、MicroAPI）会继续按算子数付费。

本算子的目标不是「一个 IR 塞进所有开关」，而是：**一份调度骨架 + 编译期 Policy 组装「matmul + 融合 + 后处理（含 TopK）」**，用属性选择 Policy，而不是复制目录。

---

## 2. 开发目标与非目标

### 2.1 目标

1. 用 `GenericLightningIndexer` 覆盖下列功能组合（Step 1）：
   - DeepSeek V3.2 Lightning Indexer（LI / QLI）
   - DeepSeek V4 CSA Indexer（LIv2 / QLIv2，Key **已压缩**）
   - GLM 5.3 Flash Index Pool（PKI：打分后 Expand + 追加尾）
   - Qwen3.8-Flash-Next Compressed Lightweight Indexer（ReLU + **累加和** + 块 TopK + Expand + 追加不完整块）
2. 接口上预留 MiniMax M3 / `MsaIndexScore` 的扩展点（Step 2 再实现 kernel）：无激活、沿 Key 分段 Maxpool、可选不融合 TopK。
3. Q/K **前处理不做进本算子**（RoPE、RMSNorm、AvgPool、KeyPool、CSA 压缩器均在上游）。本算子必须能消费**已经压缩**的 Key，并用 `cmp_ratio` 做 mask / Expand。
4. 后处理**必须包含 TopK**（Step 1）。MSA 的「只出 score」通过 `select_mode=none` 预留，Step 1 tiling 拒绝该取值。

### 2.2 非目标

- 不实现 CSA/HCA 主注意力（`SparseFlashMla`）。
- 不实现 Key 的 softmax 加权池化 / AvgPool / RMSNorm / RoPE。
- Step 1 不把 MSA 的 catlass 私有分支收编进本算子 kernel。
- 不把 6 个旧算子在本阶段从仓中删除；旧 IR 保持，本算子并行落地，后续再做兼容层。

---

## 3. 统一数学语义

去掉各家前处理后，Indexer 计算路径为：

$$
S = \phi\!\left( \alpha \cdot Q K^\top \odot \mathrm{Dequant} \right)
$$

$$
s = \mathrm{Reduce}_h(S),\qquad
idx = \mathrm{Post}(\mathrm{TopK}(s \odot M))
$$

| 符号 | 含义 | Step 1 取值 |
|---|---|---|
| $Q$ | Indexer query，$[S_1, H_I, D]$，$D=128$，$H_I\le 64$ | 必选 |
| $K$ | **已压缩或未压缩**的 Indexer key，$K_N=1$ | 必选；压缩发生在算子外 |
| $\alpha$ | score 缩放 | 默认 $1$；PKI 映射为 $1/\sqrt{D}$ |
| $\phi$ | 激活 | `relu` / `identity` |
| $\mathrm{Reduce}_h$ | 沿 indexer head | `weighted_sum`（要 $W$）或 `sum`（Qwen） |
| $M$ | 可见性掩码 | 见 §3.2 |
| $\mathrm{Post}$ | TopK 之后 | identity / Expand / Expand+AppendTail |

### 3.1 `cmp_ratio` 与 mask 斜率

`cmp_ratio = r`（$\ge 1$ 的整数）：

- $r=1$：Key 未压缩，打分轴 = token 轴。
- $r>1$：Key 的 $S_2$ 是压缩后长度；算子**不再压缩**，只把「原始 token 位置 $\leftrightarrow$ 压缩轴」用 $r$ 做整除映射。

从 causal mask 看，$r$ 就是可见区域边界在 $(q, k_{\mathrm{cmp}})$ 平面上的**斜率倒数**：

$$
k_{\mathrm{cmp}}^{\mathrm{visible}}(q) = \left\lfloor \frac{q_{\mathrm{orig}} + c}{r} \right\rfloor
$$

$c$ 由 `sparse_mode` 决定（§3.2）。这与 LIv2 的 `cmp_ratio`、PKI 的 `pool_size`、Qwen 的 `indexer_compress_ratio` 是同一参数。

凑不满一组的残余 token 数记为 `cmp_residual_k[b]`，满足 $0 \le \mathrm{residual} < r$。它们**不进入打分**（与 LIv2 / PKI / Qwen 一致）。

### 3.2 `sparse_mode`

| 值 | 名称 | 可见压缩 Key 数（第 $i$ 个 query，0-based） |
|---|---|---|
| 0 | none | 全部有效压缩 Key |
| 3 | rightDownCausal | $\lfloor (L_k^{\mathrm{orig}} - S_1 + i + 1) / r \rfloor$（LI / LIv2 / PKI） |
| 4 | blockCompleteCausal | $\lfloor (i+1)/r \rfloor$，即只给**已完整观测**的块打分（Qwen） |

`sparse_mode=4` 是 Step 1 为 Qwen 新增的取值；`3` 覆盖现有 5 个算子。

### 3.3 Expand 与尾 token

| `expand_indices` | `append_tail` | 输出 indices |
|---|---|---|
| 0 | 0 | 压缩轴（或 token 轴，$r=1$）上的 TopK 下标 |
| 1 | 0 | 每个选中块展开为 $r$ 个连续 token |
| 1 | 1 | 展开后再追加本序列残余 token（最多 $r-1$ 个） |

输出最后一维：

$$
L_{\mathrm{out}} =
\begin{cases}
\mathrm{topk}, & \text{expand}=0 \\
\mathrm{topk}\cdot r + [ \text{append} ]\cdot(r-1), & \text{expand}=1
\end{cases}
$$

无效位置填 `-1`；对应 values 填 `-inf`。

**注意**：本算子的 `topk` 一律表示**打分轴上选出的个数**。PKI 旧接口的 `topk` 是展开后的 token 预算，映射时要除以 `pool_size`。

---

## 4. 收编映射

### 4.1 Step 1 必达

| 来源 | `cmp_ratio` | `act_type` | `has_weighted_sum` | `sparse_mode` | `expand_indices` | `append_tail` | `quant_mode` | `score_scale` |
|---|---|---|---|---|---|---|---|---|
| LightningIndexer | 1 | relu | true | 0 或 3 | 0 | 0 | none | 1 |
| LightningIndexerV2 | $r$ | relu | true | 0 或 3 | 0 | 0 | none | 1 |
| QuantLightningIndexer | 1 | relu | true | 0 或 3 | 0 | 0 | INT8/FP8/HiFP8 | 1 |
| QuantLightningIndexerV2 | $r$ | relu | true | 0 或 3 | 0 | 0 | +MXFP8/MXFP4 | 1 |
| PoolKeyIndexer | `pool_size` | relu | true | 0 或 3 | 1 | 1 | none/FP8/MXFP8 | $1/\sqrt{D}$ |
| Qwen CLI | 4（可配） | relu | **false（累加和）** | **4** | 1 | 1 | none（后续可量化） | 1 |

### 4.2 Step 2 预留（本阶段只占属性，tiling 拒绝 kernel 路径）

| 来源 | `act_type` | `has_weighted_sum` | `key_reduce_mode` | `select_mode` | 说明 |
|---|---|---|---|---|---|
| MsaIndexScore | identity | false | **maxpool** | **none** | 输出 block score，TopK 在算子外 |
| MSA + 融合 TopK（可选） | identity | false | maxpool | topk | 若后续 MiniMax 要把 TopK 融进来 |

`key_reduce_mode` 与 `select_mode` 在 Step 1 的合法取值仅为 `none` / `topk`。

---

## 5. 算子接口

### 5.1 输入

| 序号 | 名称 | 必选 | dtype | 说明 |
|---|---|---|---|---|
| 0 | `query` | 是 | FP16/BF16；950 另支持 FP8_e4m3fn / HiFP8 / FP4_e2m1 / INT8 | Indexer Q |
| 1 | `key` | 是 | 与 Q 同计算族 | **压缩后或未压缩**的 Indexer K；PA 时 0 轴可非连续 |
| 2 | `weights` | 否 | FP16/BF16/FP32 | `has_weighted_sum=true` 时必选；false 时必须为空 |
| 3 | `actual_seq_q` | 否 | INT32/INT64 | TND：Q 前缀和；BSND：可选 per-batch 真实长度（`seqused_q`） |
| 4 | `actual_seq_k` | 否 | INT32/INT64 | TND：压缩轴前缀和；PA/BSND：压缩轴真实长度 |
| 5 | `cmp_residual_k` | 否 | INT32/INT64 | 压缩残余；$r>1$ 且 `sparse_mode!=0` 时必选 |
| 6 | `block_table` | 否 | INT32 | `layout_k=PA_BBND` 时必选 |
| 7 | `q_descale` | 否 | FP16/FP32/FP8_e8m0 | 量化时必选 |
| 8 | `k_descale` | 否 | 同 Q scale | 量化时必选 |
| 9 | `metadata` | 否 | INT32 `[1024]` | 预留 LIv2 式负载均衡；Step 1 可不传 |

### 5.2 输出

| 序号 | 名称 | dtype | 说明 |
|---|---|---|---|
| 0 | `sparse_indices` | INT32 | 选中下标；无效 `-1` |
| 1 | `sparse_values` | FP32（量化路径可 BF16） | `return_value=true` 时为 TopK 分数；否则 shape `(0,)` |

`select_mode=none`（Step 2）时：`sparse_indices` 空，`sparse_values` 改为 score 张量（MSA 的 `score`）。

### 5.3 属性

| 序号 | 名称 | 类型 | 默认 | 取值 |
|---|---|---|---|---|
| 0 | `topk` | int | 2048 | 打分轴选出的个数；`[1,2048]` ∪ `{3072,4096,5120,6144,7168,8192}` |
| 1 | `cmp_ratio` | int | 1 | $[1,128]$；1 = 未压缩 |
| 2 | `layout_q` | string | `"BSND"` | `BSND` / `TND` |
| 3 | `layout_k` | string | `"BSND"` | `BSND` / `TND` / `PA_BBND` |
| 4 | `sparse_mode` | int | 0 | 0 / 3 / 4 |
| 5 | `act_type` | int | 0 | 0=`relu`，1=`identity`（Step 2 / MSA） |
| 6 | `has_weighted_sum` | bool | true | true=乘 $W$ 再沿 head 求和；false=沿 head 累加和 |
| 7 | `expand_indices` | bool | false | TopK 后是否展开回 token |
| 8 | `append_tail` | bool | false | 展开后是否追加残余 token；仅 `expand_indices=true` |
| 9 | `quant_mode` | int | -1 | -1 none；对齐 QLIv2：1 FP8 PTH / 2 INT8 / 3 MXFP8 / 4 HiFP8 / 5 MXFP4 |
| 10 | `return_value` | bool | false | 是否输出分数 |
| 11 | `select_mode` | int | 0 | 0=`topk`；1=`none`（Step 2 预留，Step 1 拒绝） |
| 12 | `key_reduce_mode` | int | 0 | 0=`none`；1=`maxpool`（Step 2 预留，Step 1 拒绝） |
| 13 | `score_scale` | float | 1.0 | QK 缩放；PKI 传 `1/sqrt(128)` |
| 14 | `key_stride0` | int | -1 | PA 下 K 的 0 轴 stride；-1 由 shape 推导 |

约束：

- HeadDim $=128$，$K_N=1$，$H_I\in[1,64]$。
- `append_tail=true` ⇒ `expand_indices=true`。
- `has_weighted_sum=true` ⇒ `weights` 非空；反之 `weights` 必须为空。
- `act_type=identity` 在 Step 1 仅允许与 `select_mode=topk` 且 `key_reduce_mode=none` 组合（给 MSA 预留 identity+maxpool，但 Step 1 不跑）。
- `sparse_mode=4` 建议与 `expand_indices=true` 一起使用（Qwen）；允许单独使用以便单测 mask。

### 5.4 Shape

`layout_q=BSND`：

- Q: `[B, S1, N1, D]`；K: `[B, S2, 1, D]` 或 PA `[BN, BS, 1, D]`
- weights: `[B, S1, N1]`
- indices: `[B, S1, L_out]`

`layout_q=TND`：

- Q: `[T1, N1, D]`；indices: `[T1, L_out]`

$S_2$ / $T_2$ 始终是**压缩后**长度。

---

## 6. 实现方案

### 6.1 分层

```
Layer 4  算子壳     def / infershape / tiling / aclnn / torch
Layer 3  Policy     GliPolicy<Act, HeadReduce, KeyReduce, Select, Expand, Mask>
Layer 2  Stage      CubeQK | Act | HeadReduce | KeyReduce | Mask | TopK | ExpandAppend
Layer 1  调度       GliKernel：分核、CV 同步、PA、actual_seq、尾块
Layer 0  硬件       复用 common/include/op_kernel；禁止再 vendor 一套 catlass
```

Policy 全部是 **编译期** `if constexpr` / 模板参数，不把激活、是否加权做成运行期分支。这与 Catlass EVG「声明节点、由 Block 统一调度 LOAD/COMPUTE/STORE」同一意图。

### 6.2 为什么不完全用 Catlass EVG 做整算子

参考 `catlass/docs/zh/2_Design/03_evg/` 与 `examples/64_ascend950_matmul_evg`：

EVG 适合 **逐元素** 尾处理（Add / Relu / Silu / Bias broadcast）。Indexer 的 Vector 侧还有三类 **非逐元素** 图：

| 阶段 | 是否 EVG 能直接表达 | 处理 |
|---|---|---|
| $\phi$（ReLU / Identity） | 能。对应 `VisitorCompute<Relu>` / 旁路 | Step 1 用 Policy 对齐 EVG 节点语义；后续可换成 `TreeVisitor` |
| 沿 head 加权和 / 累加和 | 部分。是 $H_I$ 轴规约，不是 elementwise | 自定义 `HeadReduce` Stage；接口按 EVG 三阶段（LOAD W → COMPUTE → 产出 $s$） |
| 沿 Key 分段 Maxpool | 否（MSA Step 2） | `KeyReduce` Stage，预留 |
| TopK + Expand + Append | 否。排序 / gather / 标量写尾 | 独立 `Select` + `IndexPost` Stage，复用 LI/PKI 已验证的 `vf_topk_gather` |

Cube 主循环：**不新建 catlass 私有分支**（MSA 那条路已经分叉）。Step 1 沿用 LI/PKI arch35 的 Fixpipe L0C→UB（`CO2Layout::ROW_MAJOR` 直出 UB），这是该家族 950 的默认路径，避免再实现一遍 C2UB。

Catlass 组件选型（供后续把 Cube 迁到 BlockMmad 时使用，**不是 Step 1 必交付**）：

| 组件 | 选型 | 说明 |
|---|---|---|
| ArchTag | `Arch::Ascend950` / A2 `AtlasA2` | 代际隔离 `arch35` / `arch22` |
| 参考 example | `64_ascend950_matmul_evg`（LeakyReLU / UB visitor） | 只借鉴 Act 节点，不借鉴整块 HBM C |
| DispatchPolicy | 950：与 LI cube 等价的 pingpong + Fixpipe-to-UB | 禁止用 `MatmulActivation` 把整块 S 写 GM 再读回 |
| BlockEpilogue | 自定义 `GliEpilogue`，槽：Act → HeadReduce → Mask | TopK 不进 Epilogue |
| Kernel | 自有 `GliKernel`，不是 `BasicMatmulTlaVisitor` | 多 query 行 × 长 S2 滑窗 + 在线 TopK，不是一次 GEMM |

### 6.3 Policy 类型（kernel）

```cpp
enum class GliAct : uint32_t { Relu = 0, Identity = 1 };
enum class GliHeadReduce : uint32_t { WeightedSum = 0, Sum = 1, None = 2 };
enum class GliKeyReduce : uint32_t { None = 0, MaxPool = 1 };
enum class GliSelect : uint32_t { TopK = 0, None = 1 };
enum class GliExpand : uint32_t { Off = 0, Expand = 1, ExpandAppend = 2 };
enum class GliMask : uint32_t { None = 0, Causal = 3, BlockComplete = 4 };

struct GliPolicy {
    static constexpr GliAct act = ...;
    static constexpr GliHeadReduce headReduce = ...;
    static constexpr GliKeyReduce keyReduce = ...;
    static constexpr GliSelect select = ...;
    static constexpr GliExpand expand = ...;
    static constexpr GliMask mask = ...;
};
```

属性到 Policy 的映射由 host 写入 TilingKey，kernel `if constexpr` 实例化。Step 1 合法组合见 §7。

### 6.4 调度骨架

与 `PoolKeyIndexerKernel` / `LightningIndexerV2Kernel` 同构（方法名对齐，便于对照回归）：

`InitTilingData` → `SplitCore` → `ProcessMain` → `ProcessBaseBlock`：

1. Cube：`Q_tile @ K_tile^T`，Fixpipe 到 UB（可选 dequant）。
2. Vector1：$\phi$ + `HeadReduce` + 写 score workspace（或在线喂 TopK）。
3. 最后一个 S2 tile：`TopK`；若 `expand!=Off`，`ExpandAndAppendIndices`。

`cmp_ratio` 只出现在：

- 计算 `actS2SizeOrig = actS2Cmp * r + residual` 做 mask；
- Expand 时 `token = pool * r + offset`；
- Append 时写 `L_orig - residual + t`。

### 6.5 与旧算子 TopK 去重

Step 1 kernel 将 TopK gather 放到本算子 `op_kernel/arch35/vf/` 下的 **一份** 实现（从 LI/PKI 收敛，namespace `gli::topk`）。后续 P0 若抽 `indexer_common`，本算子直接改 include。禁止再复制第四份 `vf_topk_gather.h`。

---

## 7. TilingKey

模板字段（`ASCENDC_TPL_*`）：

| 字段 | 编码 | 说明 |
|---|---|---|
| DT_Q / DT_K | FP16=1, BF16=27, FP8=2, INT8=… | 与 LI 家族一致 |
| LAYOUT_Q | 0 BSND / 1 TND | |
| LAYOUT_K | 0 BSND / 1 TND / 2 PA_BBND | |
| SPARSE_MODE | 0 / 3 / 4 | |
| ACT_TYPE | 0 relu / 1 identity | |
| HEAD_REDUCE | 0 weighted / 1 sum | |
| EXPAND | 0 off / 1 expand / 2 expand+append | |
| QUANT_MODE | host+1 偏移 | |
| RETURN_VALUE | 0/1 | |

不进 TilingKey、走 tiling data：`topk`、`cmp_ratio`、`score_scale`、`block_size`、seqlen、workspace 偏移。

Step 1 `ASCENDC_TPL_SEL` 只枚举需要的组合，避免笛卡尔积：

- 非量化 FP16/BF16 × 三种合法布局 × sparse{0,3,4} × act=relu × headReduce{weighted,sum} × expand{0,1,2} × return{0,1}
- 950 FP8：先覆盖 `quant_mode` 与 LI/PKI 已有档，expand 以 off / expand+append 为主

`act=identity`、`key_reduce=maxpool`、`select=none` **不进 Step 1 SEL**。

---

## 8. 测试策略

测试必须按 **收编剖面** 组 case，而不是只测默认属性。

### 8.1 Host UT（InferShape）

每个剖面对应一条成功 case + 若干非法属性失败 case：

| ID | 剖面 | 关键属性 | 预期 indices 末维 |
|---|---|---|---|
| P-LI | LI | r=1, relu, weighted, sm=3, expand=0 | topk |
| P-LIv2 | LIv2 | r=4, relu, weighted, sm=3, expand=0 | topk |
| P-QLI | QLI | r=1, quant INT8/FP8 | topk |
| P-QLIv2 | QLIv2 | r=4, quant MXFP8 | topk |
| P-PKI | PKI | r=16, relu, weighted, sm=3, expand+append | topk*r+(r-1) |
| P-QWEN | Qwen CLI | r=4, relu, **sum**, sm=**4**, expand+append | topk*r+(r-1) |
| P-MSA-REJ | MSA 预留 | select_mode=1 或 key_reduce=1 | InferShape/Tiling **失败**（Step 1） |

另覆盖：TND、PA_BBND、`return_value`、`weights` 与 `has_weighted_sum` 不一致、`append_tail` 无 expand、`cmp_ratio` 越界。

### 8.2 Host UT（Tiling）

合法剖面 tilingKey 可解码；非法组合 `GRAPH_FAILED`。

### 8.3 Kernel / ST

Golden（NumPy，与 §3 公式一致）：

```text
S = score_scale * (Q @ K^T)
S = relu(S) or S
s = einsum("qhd,qh->qd", S, W)  or  S.sum(head)
s[:, visible:) = -inf
idx = topk(s, k)
if expand: idx = idx * r + arange(r); if append: concat residual tokens
```

精度：indices 必须与 golden 一致（排序稳定：分数相等取更小下标）；values 按 dtype 走仓内 mixed tolerance。

性能：Step 1 门禁为不低于对应旧算子基线的 95%（同 shape、同平台）。未达则记为已知差距，不阻塞 IR 合入，但不得宣称「性能归一」。

---

## 9. 开发迭代（本分支）

```
Commit A  需求设计文档（本文件）
Commit B  Host：def / infershape / tiling / proto / CMake / InferShape UT
Commit C  Kernel：Policy + arch35 调度 + Act/Reduce/TopK/Expand（FP16/BF16）
Commit D  量化 / PA 路径补齐 + 剖面 ST
Commit E  （Step 2，另开迭代）MSA Maxpool + select_mode=none
```

每步可运行对应 UT；kernel 未就绪时 Host UT 仍可回归接口。

---

## 10. 风险

| 风险 | 缓解 |
|---|---|
| TilingKey 组合膨胀 | SEL 表只列收编剖面；新模型先加一行 SEL |
| EVG 不能表达 TopK | TopK 永不塞进 EVG；只把 Act 做成可替换节点 |
| Qwen mask 与 LI causal 混淆 | `sparse_mode=4` 独立编码，golden 单测 |
| PKI `topk` 语义不同 | 文档 + 映射表强制「打分轴 k」；旧 PKI 调用方除以 r |
| MSA 过早实现拖垮 Step 1 | tiling 硬拒绝 `key_reduce_mode=maxpool` |
| 再 vendor catlass | Layer 0 禁止；Cube 用现有 LI Fixpipe-to-UB |
