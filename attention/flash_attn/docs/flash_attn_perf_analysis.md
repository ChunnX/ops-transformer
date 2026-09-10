# Flash Attention 性能 (Cube Bound) 分析

基于 Ascend 硬件规格与 FA (Flash Attention) kernel 的 tiling 模型，推导 FA 达到 **cube bound**（cube 为性能瓶颈）所需满足的计算/访存比与空间约束。

## 1. 硬件规格 (单核)

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| CoreNum | 32 | 核数 |
| Frequency | 1.65 GHz | 主频 |
| L1 | 512 KB | |
| L0A | 64 KB | 左矩阵 A 缓冲 |
| L0B | 64 KB | 右矩阵 B 缓冲 |
| L0C | 256 KB | cube 输出 / 累加缓冲 |
| UB | 256 KB | 统一缓冲（可用 248 KB） |
| L2Capacity | 128 MB | |
| L2BandWidth | 5.2 TB/s | 折合单核 ~101 B/cycle |
| HBM Size | 128 GB | |
| HBM BandWidth | 1.6 TB/s | 折合单核 ~31 B/cycle |

> 单核带宽换算：总带宽 ÷ 核数 ÷ 主频。
> L2：5.2×10¹² ÷ (32 × 1.65×10⁹) ≈ 101 B/cycle；HBM：1.6×10¹² ÷ (32 × 1.65×10⁹) ≈ 31 B/cycle。

## 2. Tiling 变量定义

FA 单次外层迭代（处理一个 KV 块）涉及的 tiling 维度：

| 变量 | 含义 | 取值（按 config） |
| --- | --- | --- |
| M = sInner | Q 序列方向 tiling 块大小（bmm 的 M 维） | 128 / 256 |
| N = sOuter | KV 序列方向 tiling 块大小（bmm 的 N 维） | 32 / 64 |
| D = qkHeadDim | Q/K 头维度 | 64 / 128 / 256 |
| dV = vHeadDim | V 头维度（非量化时 = D） | = D |
| S2 | KV 序列总长 | 运行时 |

config 与 `(N, M, D)` 的对应（见 `flash_attn_template_tiling_key.h`）：

| config | N (sOuter) | M (sInner) | D (qk) | dV |
| --- | --- | --- | --- | --- |
| 0 | 64 | 128 | 64 | 64 |
| 1 | 32 | 256 | 64 | 64 |
| 2 | 64 | 128 | 128 | 128 |
| 3 | 32 | 256 | 128 | 128 |
| 4 | 32 | 256 | 256 | 256 |

dtype：非量化 FA 的 Q/K/V 均为 fp16 / bf16，`sizeof = 2 B`。

## 3. 时间模型

Cube 单元单周期完成一次 `16×16×16` mma，吞吐：

$$ FMA = 16 \times 16 \times 16 = 4096 \text{ FMA/cycle} $$

以下时间均按**单次迭代总量**计（含 N 维），便于跨阶段一致比较。

### 3.1 Cube 计算时间

单次迭代两个 matmul（bmm1 = Q@Kᵀ、bmm2 = P@V），总量纲 `2·M·N·D`：

$$ T_{cube} = \frac{2 \cdot M \cdot N \cdot D}{FMA} $$

> 按 N 归一即原文①式 $T_{cube}/N = 2MD/FMA$（“2 为 2 个 matmul”）。

### 3.2 MTE2 搬数时间

Q 矩阵仅在首次搬入一次（`M·D`），K/V 每次迭代搬入（`2·N·D`）：

$$ T_{mte2} = \frac{M \cdot D \cdot \text{sizeof}(qDtype)}{HBM_{bw}} + \frac{2 \cdot N \cdot D \cdot \text{sizeof}(kvDtype)}{Mix_{bw}} $$

其中 $Mix_{bw} \in [HBM_{bw},\ L2_{bw})$ 为 L2 命中与 HBM 缺失混合后的有效带宽。

当 S2 足够大时，Q 的一次性搬入可被多次迭代摊薄，忽略左矩阵（Q）搬入对总时长的影响：

$$ T_{mte2} \approx \frac{2 \cdot N \cdot D \cdot \text{sizeof}(kvDtype)}{Mix_{bw}} $$

### 3.3 Fixpipe 时间

bmm1 / bmm2 结果由 L0C fixpipe 到 UB（fp32）：

$$ T_{fixpipe} = \frac{M \cdot N \cdot \text{sizeof}(fp32)}{UB_{bw}} + \frac{M \cdot D \cdot \text{sizeof}(fp32)}{UB_{bw}} = \frac{(M N + M D)\cdot \text{sizeof}(fp32)}{UB_{bw}} $$

## 4. Cube Bound 条件与推导

FA 目标做到 cube bound，即 cube 为瓶颈：

$$ T_{cube} > T_{mte2}, \qquad T_{cube} > T_{fixpipe} $$

### 4.1 由 $T_{cube} > T_{mte2}$ 推 M 的范围（大 S2）

$$ \frac{2 M N D}{FMA} > \frac{2 N D \cdot \text{sizeof}(kvDtype)}{Mix_{bw}} $$

两侧 `N`、`D` 同时约去：

$$ M > \frac{FMA \cdot \text{sizeof}(kvDtype)}{Mix_{bw}} $$

代入 `FMA = 4096`、`sizeof(kvDtype) = 2 B`（fp16/bf16）：

- $Mix_{bw} = L2_{bw} = 101$ B/cycle（L2 全命中，带宽上界）：
  $$ M > \frac{4096 \times 2}{101} = 81.1 \;\Rightarrow\; M > 82 $$
- $Mix_{bw} = HBM_{bw} = 31$ B/cycle（全 HBM 缺失，带宽下界）：
  $$ M > \frac{4096 \times 2}{31} = 264.3 \;\Rightarrow\; M > 265 $$

故 $82 < M < 265$ 的含义为：

| M 取值 | 结论 |
| --- | --- |
| $M \ge 265$ | 任意 $Mix_{bw}$ 下均为 cube bound |
| $82 < M < 265$ | 仅当 L2 命中较好（$Mix_{bw}$ 偏大）时为 cube bound |
| $M \le 82$ | 始终为 mte2 bound |

当前 tiling 的 `M ∈ {128, 256}` 均落在 $(82, 265)$，故在 L2 命中较好时可达到 cube bound。

### 4.2 由 $T_{cube} > T_{fixpipe}$ 推 N 的约束

$$ \frac{2 M N D}{FMA} > \frac{(M N + M D)\cdot \text{sizeof}(fp32)}{UB_{bw}} $$

约去 `M` 并整理：

$$ N > \frac{FMA \cdot D \cdot \text{sizeof}(fp32)}{UB_{bw} \cdot 2D - FMA \cdot \text{sizeof}(fp32)} $$

> 注：上式分子分母均含 `D`，需求 `UB_bw`（UB 带宽）方可数值求解；当前硬件表未列 UB 带宽，待补充。

## 5. 空间约束

### 5.1 L0C

bmm 输出在 L0C 双缓冲：

$$ M \cdot N \cdot \text{sizeof}(fp32) < \frac{L0CSize}{2} = 128\text{ KB} $$

即 $M \cdot N < 32768$。当前各 config 的 $M \cdot N$ 均为 $8192$（$128\times64$ 或 $256\times32$），余量 4×，全部满足。

### 5.2 UB buffer

各 buffer 占用（双缓冲，M 维取 `M/2`）：

| buffer | 占用 (B) |
| --- | --- |
| bmm1 | $2 \cdot \tfrac{M}{2} \cdot N \cdot \text{sizeof}(fp32) = 4MN$ |
| bmm2 | $2 \cdot \tfrac{M}{2} \cdot D \cdot \text{sizeof}(fp32) = 4MD$ |
| vec1 | $2 \cdot (\tfrac{M}{2}\cdot N \cdot \text{sizeof}(fp16) + \tfrac{M}{2}\cdot 4) = 2MN + 4M$ |
| vec2 | $\tfrac{M}{2}\cdot D \cdot \text{sizeof}(fp32) = 2MD$ |
| Mask | $2 \cdot \tfrac{M}{2}\cdot N \cdot \text{sizeof}(bool) = MN$ |
| lseout | $2 \cdot \tfrac{M}{2}\cdot 8 \cdot \text{sizeof}(fp32) = 32M$ |
| softmaxMax | $\text{preload\_num}\cdot \tfrac{M}{2}\cdot \text{sizeof}(fp32) = 6M$（preload_num=3） |
| softmaxSum | $6M$ |
| softmaxExp | $6M$ |
| softmaxTmp | $512$ |

若各 buffer 同时驻留（最坏情况、无分阶段复用），合计：

$$ Total_{UB} = 7MN + 6MD + 54M + 512 $$

约束：$Total_{UB} \le 248\text{ KB} = 253952\text{ B}$。

- 以 config 0（$M=128,N=64,D=64$）为例：$Total_{UB} = 113920\text{ B} \approx 111\text{ KB} \le 248\text{ KB}$，满足。
- 其中 bmm2 ($4MD$) 与 vec1 ($2MN$) 在大 D / 大 M 下为主导项；当 $D \ge 128$ 且 $M=256$ 时最坏合计超过 248 KB，实际通过 **D 维分块** 与 **buffer 分阶段复用**（softmaxMax/Sum/Exp 等中间量复用同一物理空间）予以缓解。

## 6. 结论

- **计算/访存比**：当前 tiling 的 $M \in \{128, 256\}$ 落在 $(82, 265)$ 区间，L2 命中较好时可达到 cube bound；L2 命中差（趋近 HBM 带宽）时退化为 mte2 bound，需尽量提高 KV 的 L2 复用。
- **空间**：L0C 约束（$MN < 32768$）所有 config 均满足；UB 约束在小 D config 满足，大 D config 依赖 D 维分块与 buffer 复用。
- **待补充**：UB 带宽 `UB_bw`，用于 4.2 节 N 约束的数值求解。
