# QkvRmsNormRopeCacheWithKScale 算子入门讲解

## 1. 算子做什么

`QkvRmsNormRopeCacheWithKScale` 是一个 Attention 前处理融合算子。它接收已经投影好的融合 Q/K/V，完成以下计算：

```text
qkv
 ├─ Q → RMSNorm → RoPE → × rotation → 动态量化 → q_out + q_scale
 ├─ K → RMSNorm → RoPE → × rotation → 动态量化 → k_cache + k_scale_cache
 └─ V → × v_scale → FP8 量化 → v_cache
```

如果不用融合算子，上述步骤可能分别启动多个 kernel，并在 GM（Global Memory）中产生多个中间张量。融合后，很多中间结果可以保留在片上存储中，并让 Vector 和 Cube 流水执行，从而减少 kernel 启动和 GM 读写。

需要特别区分两个概念：

- `RoPE` 使用 `cos_sin` 给 Q/K 加入位置信息；
- `rotation` 是 RoPE 之后额外右乘的一个 `[D,D]` 稠密矩阵。

`rotation` 不属于 RoPE 公式。

本文首先讲当前最常见的 `ROPE_FP8_QK` 场景。掌握这条主链路后，可继续阅读
`18_mrope_math_beginner_guide.md` 的 M-RoPE 数学逻辑与推导；完整接口和实现合同见第 11 篇设计文档。

## 2. Shape 和布局

常用符号如下：

| 符号 | 含义 |
|---|---|
| `T` | 本次调用处理的 token 总数 |
| `Nq` | Q head 数 |
| `Nk` | K head 数 |
| `Nv` | V head 数 |
| `D` | 每个 head 的维度，当前固定为 128 |
| `Batch` | 本次调用中的 batch 数 |
| `BlockNum` | KV Cache 的 block 数 |
| `BlockSize` | 每个 cache block 容纳的 token 数 |

当前 arch35 实现要求：

```text
0 < Nq <= 64
Nq = 8 × Nk
Nk = Nv
D = 128
```

例如：

```text
Nq = 16
Nk = 2
Nv = 2
D  = 128
```

融合 QKV 的 head 总数为：

```text
qkvN = Nq + Nk + Nv = 20
```

### 2.1 TND 和 NTD

`qkv` 支持两种布局：

```text
TND: qkv.shape = [T, Nq + Nk + Nv, D]
NTD: qkv.shape = [Nq + Nk + Nv, T, D]
```

三个字母分别表示：

- `T`：token；
- `N`：head；
- `D`：head dimension。

例如，TND 布局下 `[3,20,128]` 表示：

- 本次处理 3 个 token；
- 每个 token 有 20 个融合 QKV head；
- 每个 head 包含 128 个元素。

数学公式统一使用逻辑视图：

```text
qkv_value[t,n,d]
```

kernel 再根据实际布局选择不同的 GM stride：

```text
layout_qkv == NTD:
    qkvGmStrideT = D
    qkvGmStrideN = T * D

layout_qkv == TND:
    qkvGmStrideT = qkvN * D
    qkvGmStrideN = D
```

因此两种布局可以统一寻址为：

```text
qkv_value(t,n,d)
  = qkv_gm[n * qkvGmStrideN + t * qkvGmStrideT + d]
```

## 3. 从融合 QKV 中拆出 Q、K、V

QKV 在 N 轴上的排列固定为：

```text
[全部 Q heads | 全部 K heads | 全部 V heads]
```

拆分公式为：

```text
Q[t,nq,d] = qkv_value(t,nq,d)

K[t,nk,d] = qkv_value(t,Nq + nk,d)

V[t,nv,d] = qkv_value(t,Nq + Nk + nv,d)
```

如果：

```text
Nq=16, Nk=2, Nv=2
```

那么融合 N 轴的范围为：

```text
0  ... 15 → Q
16 ... 17 → K
18 ... 19 → V
```

拆分后的逻辑 shape 为：

```text
Q: [T,Nq,D]
K: [T,Nk,D]
V: [T,Nv,D]
```

这里的“拆分”主要是逻辑概念。kernel 不一定创建三块完整的新内存，而是根据相应 offset 从融合 `qkv` 中直接读取。

## 4. Q/K 的 RMSNorm

只有 Q 和 K 做 RMSNorm，V 不做。

对于某个 token 的某个 head，取长度为 `D` 的向量：

```text
x = X[t,h,:]
```

首先计算平方和：

$$
S_{t,h}=\sum_{j=0}^{D-1}X_{t,h,j}^2
$$

然后计算 RMS：

$$
R_{t,h}=\sqrt{\frac{S_{t,h}}{D}+\epsilon}
$$

最后归一化并乘 gamma：

$$
Y_{t,h,j}=\frac{X_{t,h,j}}{R_{t,h}}G_{h,j}
$$

Q 和 K 使用不同的 gamma：

$$
Q^{\mathrm{norm}}_{t,n,j}
=
\frac{Q_{t,n,j}}
{\sqrt{\frac{1}{D}\sum_r Q_{t,n,r}^2+\epsilon}}
\cdot q\_gamma_j
$$

$$
K^{\mathrm{norm}}_{t,n,j}
=
\frac{K_{t,n,j}}
{\sqrt{\frac{1}{D}\sum_r K_{t,n,r}^2+\epsilon}}
\cdot k\_gamma_j
$$

RMSNorm 和 LayerNorm 的重要区别是：RMSNorm 不减均值。

```text
LayerNorm ≈ (x - mean) / sqrt(variance + epsilon)

RMSNorm   = x / sqrt(mean(x²) + epsilon)
```

本算子的 RMSNorm 使用 FP32 计算，以减小平方和、开根和归一化过程的数值误差。

## 5. 确定每个 token 的 RoPE 位置

RoPE 场景使用两个位置相关输入：

```text
query_start_loc: [Batch+1]
seq_lens:        [Batch]
```

- `query_start_loc` 描述本次调用中各 batch 的 token 分界；
- `seq_lens` 描述追加当前 token 后，各 batch 的完整序列长度。

这个算子的输入没有显式的 Batch 轴。不同 batch 在本次调用中的有效 token 会被紧凑拼接到同一个 T 轴：

$$
T=\sum_{b=0}^{Batch-1}local\_len(b)
$$

例如，batch 0 本次输入 2 个 token、batch 1 本次输入 3 个 token，拼接结果为：

```text
融合 T 轴：[batch0_token0, batch0_token1,
            batch1_token0, batch1_token1, batch1_token2]
全局索引：       0,             1,
                 2,             3,             4
```

这不是把 batch 编号和 token 编号做数值相加，而是把各 batch 的有效 token 依次排列。`query_start_loc` 保存它们在融合 T 轴上的边界。

例如：

```text
query_start_loc = [0, 2, 5]
seq_lens        = [10, 20]
```

本次调用有两个 batch：

```text
batch 0: 全局 token 0、1，local_len=2
batch 1: 全局 token 2、3、4，local_len=3
```

对于 batch 0：

```text
history_offset = 10 - 2 = 8
```

两个 token 的 RoPE position 是 `8,9`。

原因是 `seq_lens[0]=10` 表示追加本次两个 token 后，batch 0 的完整长度为 10。本次调用前已经存在：

```text
10 - 2 = 8 个历史 token
```

历史 token 占据位置 `0...7`，所以两个新 token 接着使用位置 `8,9`。

对于 batch 1：

```text
history_offset = 20 - 3 = 17
```

三个 token 的 RoPE position 是 `17,18,19`。

同理，`seq_lens[1]=20` 表示追加本次三个 token 后完整长度为 20，因此本次调用前有 `20-3=17` 个历史 token，占据位置 `0...16`；三个新 token 接着使用位置 `17,18,19`。

完整公式为：

$$
local\_len(b)
=query\_start\_loc[b+1]-query\_start\_loc[b]
$$

$$
history\_offset(b)=seq\_lens[b]-local\_len(b)
$$

对于属于 batch $b$ 的全局 token $t$：

$$
local\_pos(t)=t-query\_start\_loc[b]
$$

$$
s(t)=history\_offset(b)+local\_pos(t)
$$

这两个公式分别完成两次坐标转换。

第一个公式：

$$
local\_pos(t)=t-query\_start\_loc[b]
$$

- local_pos：可以表示为当前batch轴内token的索引
- history_offset：可以表示为当前batch轴的起始位置在全局token中的偏移位置
- 把融合 T 轴上的全局编号转换为“该 token 在本 batch 本次输入中的局部编号”。例如 batch 1 的起点是 `query_start_loc[1]=2`：

```text
全局 t：      2  3  4
local_pos：  0  1  2
```

第二个公式：

$$
s(t)=history\_offset(b)+local\_pos(t)
$$

在局部编号前加上本次调用之前已有的历史 token 数，从而得到该 token 在完整序列中的绝对位置。两个公式合并后是：

$$
s(t)
=\underbrace{seq\_lens[b]-local\_len(b)}_{\text{本次调用前的历史长度}}
+\underbrace{t-query\_start\_loc[b]}_{\text{本次输入中的局部位置}}
$$

以前面的输入为例：

| 全局 `t` | 所属 batch | `local_pos` | `history_offset` | RoPE 位置 `s(t)` |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 8 | 8 |
| 1 | 0 | 1 | 8 | 9 |
| 2 | 1 | 0 | 17 | 17 |
| 3 | 1 | 1 | 17 | 18 |
| 4 | 1 | 2 | 17 | 19 |

最终用 `s(t)` 读取：

```text
cos_sin[s(t),:]
```

不能直接把当前调用内的全局编号 `t` 当作位置，因为 `t` 不包含该序列过去已经缓存的 token 数。

## 6. RoPE 公式

当前 `D=128`。把归一化后的向量分为前后两半：

```text
low  = y[0:64]
high = y[64:128]
```

`cos_sin` 的对应行也分成两半：

```text
cos = cos_sin[position,0:64]
sin = cos_sin[position,64:128]
```

计算：

$$
out\_low=low\cdot cos-high\cdot sin
$$

$$
out\_high=high\cdot cos+low\cdot sin
$$

最后将两部分拼接：

```text
rope_out = concat(out_low,out_high)
```

对于第 `i` 对元素：

$$
out_i=y_i\cos_i-y_{i+64}\sin_i
$$

$$
out_{i+64}=y_{i+64}\cos_i+y_i\sin_i
$$

本质上，这是对每一对元素应用二维旋转矩阵：

$$
\begin{bmatrix}
x'\\y'
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}
\begin{bmatrix}
x\\y
\end{bmatrix}
$$

Q 和 K 使用同一个 token position，但分别处理各自的每个 head。RoPE 使用 FP32 计算，结果进入 Cube 前转换为 BF16。

### 6.1 二维旋转公式的作用

从几何意义看，二维旋转矩阵把向量 $(x,y)$ 旋转 $\theta$ 角度。理想数学条件下，旋转前后的长度保持不变：

$$
x'^2+y'^2=x^2+y^2
$$

因此它主要改变向量的方向，而不是简单放大或缩小向量。

RoPE 让位置 $p$ 使用与位置相关的旋转角度：

$$
\theta_{p,i}=p\cdot\omega_i
$$

其中 $i$ 表示第几个二维维度对，$\omega_i$ 是该维度对的频率。于是同一个 Q/K 内容位于不同位置时，会被旋转不同的角度，从而把位置信息编码进向量方向。

二维旋转还有一个重要性质。令 $R(\theta)$ 表示旋转矩阵，位置 $p$ 的 Q 和位置 $q$ 的 K 做点积时：

$$
(R(p\omega)Q)^T(R(q\omega)K)
=Q^TR((q-p)\omega)K
$$

因为：

$$
R(p\omega)^TR(q\omega)=R((q-p)\omega)
$$

最终旋转关系依赖 $q-p$，也就是两个 token 的相对位置。这使后续 Attention 的 Q·K 点积能够自然感知 token 之间的相对距离。

当前 `D=128`，算子把它看成 64 个二维向量，采用 half-split 配对：

```text
第 0 维  ↔ 第 64 维
第 1 维  ↔ 第 65 维
...
第 63 维 ↔ 第 127 维
```

每一对维度按照当前位置及该维度对的频率独立旋转。

## 7. RoPE 后的 rotation 矩阵乘

完成 RoPE 后，Q/K 分别右乘同一个外部 `rotation` 矩阵：

$$
Q^{\mathrm{cube}}_{t,n,j}
=\sum_{r=0}^{D-1}Q^{\mathrm{rope}}_{t,n,r}\,rotation_{r,j}
$$

$$
K^{\mathrm{cube}}_{t,n,j}
=\sum_{r=0}^{D-1}K^{\mathrm{rope}}_{t,n,r}\,rotation_{r,j}
$$

从矩阵 shape 看：

```text
[T × Nq,D] @ [D,D] → [T × Nq,D]
[T × Nk,D] @ [D,D] → [T × Nk,D]
```

数学上可以把 Q/K 看成拼接到一起，但当前实现分别发起 Q 和 K 的矩阵乘，共用右矩阵 `rotation`。该部分由 AIC/Cube 完成。

## 8. Q/K 的动态量化

普通 RoPE 场景中，Q 和 K 最终量化为 FP8 E4M3FN。

量化粒度是逐 token、逐 head。因此，每个长度为 128 的 `[t,h,:]` 向量都有自己的 scale。

对于 Cube 输出 $C$，第一步计算该行的最大绝对值：

$$
m_{t,h}=\max_{0\le d<D}|C_{t,h,d}|
$$

FP8 E4M3FN 最大有限正值为 448，因此：

$$
scale_{t,h}=\frac{m_{t,h}}{448}
$$

然后除以 scale 并转换为 FP8：

$$
\widehat C_{t,h,d}
=\operatorname{FP8}\left(\frac{C_{t,h,d}}{scale_{t,h}}\right)
$$

转换包含 RINT 舍入和饱和处理。

例如，一行的最大绝对值为 `896`：

```text
scale = 896 / 448 = 2
```

该行所有元素先除以 2，再转换为 FP8。后续使用时可以近似反量化：

```text
原始值 ≈ FP8值 × scale
```

因此：

- `q_out` 保存量化后的 Q；
- `q_scale` 保存 Q 的动态 scale；
- `k_cache` 保存量化后的 K；
- `k_scale_cache` 保存 K 的动态 scale。

## 9. V 分支

V 不参与以下计算：

```text
RMSNorm
RoPE
rotation matmul
基于 max_abs 的动态 scale 计算
```

普通 RoPE 场景中，V 只做：

$$
V^{\mathrm{scaled}}_{t,n,d}
=fp32(V_{t,n,d})\cdot v\_scale_n
$$

然后转换为 FP8：

$$
V^{\mathrm{out}}_{t,n,d}
=\operatorname{FP8}(V^{\mathrm{scaled}}_{t,n,d})
$$

这里的 `v_scale` 是调用者传入的乘数，不是 Q/K 动态量化过程中输出的 scale：

```text
Q/K scale：kernel 根据每行 max_abs 动态计算
V scale：调用者传入，直接与 V 相乘
```

普通 RoPE 场景要求：

```text
v_scale.shape = [Nv]
```

即一个 V head 共用一个乘数。

## 10. KV Cache 地址与写回

### 10.1 BlockSize 的物理意义

`BlockSize` 表示一个 KV Cache block 能够容纳多少个 token，而不是这个 block 占多少字节。Cache 的 shape 为：

```text
k_cache:       [BlockNum,Nk,BlockSize,D]
v_cache:       [BlockNum,Nv,BlockSize,D]
k_scale_cache: [BlockNum,Nk,BlockSize,1]
```

例如：

```text
BlockNum  = 100
BlockSize = 16
Nk = Nv   = 2
D         = 128
```

表示共有 100 个 KV Cache block，每个 block 可以存放 16 个 token；每个 token 保存 2 个 K head、2 个 V head，每个 head 包含 128 个元素。

把所有逻辑 slot 连续编号后，每 `BlockSize` 个 slot 属于同一个 block。`BlockSize=16` 时：

```text
block 0 → slot 0 ... 15
block 1 → slot 16 ... 31
block 2 → slot 32 ... 47
```

一个 block 实际占用的字节数还取决于 head 数、`D` 和数据类型。普通 RoPE 场景的 K/V 为 FP8，每个元素占 1 字节。以上例子中：

```text
一个 K block：2 × 16 × 128 × 1 = 4096 bytes
一个 V block：2 × 16 × 128 × 1 = 4096 bytes
一个 K scale block：2 × 16 × 1 × 4 = 128 bytes
```

这里的 block 是 Paged Attention 使用的逻辑/存储块，不等同于操作系统内存页或硬件 Cache Line。

每个 token 都有一个逻辑 cache slot：

```text
slot = slot_mapping[t]
```

算子将它转换为 block 地址：

$$
block\_id=\left\lfloor\frac{slot}{BlockSize}\right\rfloor
$$

$$
block\_offset=slot\bmod BlockSize
$$

例如：

```text
BlockSize = 16
slot      = 35
```

得到：

```text
block_id     = 2
block_offset = 3
```

对应写入位置为：

```text
k_cache[2,nk,3,:]
v_cache[2,nv,3,:]
k_scale_cache[2,nk,3,0]
```

完整写回公式为：

$$
KCache[block,nk,offset,d]=K^{\mathrm{quant}}[t,nk,d]
$$

$$
KScaleCache[block,nk,offset,0]=K^{\mathrm{scale}}[t,nk]
$$

$$
VCache[block,nv,offset,d]=V^{\mathrm{out}}[t,nv,d]
$$

所以，该算子不仅产生 K/V，还根据 `slot_mapping` 直接把结果散写到分页 KV Cache 中。

同一次调用中，如果多个 token 指向同一个 slot，就会发生重复写，最终结果未定义。正常调用应保证本次 `slot_mapping` 中的 slot 唯一。

## 11. 输出

普通 RoPE 模式有五个逻辑输出：

```text
q_out
q_scale
k_cache
v_cache
k_scale_cache
```

后三个是原地更新的 Cache。

当 `layout_q_out=TND`：

```text
q_out:   [T,Nq,D]
q_scale: [T,Nq]
```

当 `layout_q_out=NTD`：

```text
q_out:   [Nq,T,D]
q_scale: [Nq,T]
```

以 TND 为例：

```text
q_out[t,nq,:] = 量化后的 Q
q_scale[t,nq] = 这条 Q 向量的动态 scale
```

## 12. 数学语义和 kernel 实现的对应关系

数学公式描述的是数据依赖关系。实际 kernel 不会简单地逐步骤串行执行，而是让 AIV 和 AIC 并行流水。

### 12.1 AIV/Vector 主要负责

```text
读取 qkv、cos_sin、slot_mapping
计算 cache offset
V × v_scale 和 FP8 转换
Q/K RMSNorm
Q/K RoPE
Q/K FP32 → BF16
Q/K 动态量化
Q/K/V scatter 写回
```

### 12.2 AIC/Cube 主要负责

```text
Q_rope @ rotation
K_rope @ rotation
```

一个 tile 的主要执行顺序是：

```text
1. 搬入 qkv、cos_sin 和 slot_mapping
2. 根据 slot_mapping 生成 cache offset
3. 启动 V 量化
4. 计算 Q/K RMSNorm 和 RoPE
5. 写入 V Cache
6. 把预处理后的 Q/K 搬给 Cube
7. Cube 执行 rotation matmul
8. AIV 对 Cube 输出执行动态量化
9. 输出 Q，并散写 K Cache 和 K Scale Cache
```

这里“先启动 V”是流水优化，不会改变公式语义。V 分支不依赖 Q/K 的 RMSNorm、RoPE 和 Cube 输出，因此可以与 Q/K 路径交叠执行。

## 13. 用一组公式总结完整流程

忽略 layout 后，普通 RoPE 场景可以概括为：

$$
Q,K,V=\operatorname{split}(qkv)
$$

$$
Q_n=\operatorname{RMSNorm}(Q,q\_gamma)
$$

$$
K_n=\operatorname{RMSNorm}(K,k\_gamma)
$$

$$
Q_r=\operatorname{RoPE}(Q_n,position,cos\_sin)
$$

$$
K_r=\operatorname{RoPE}(K_n,position,cos\_sin)
$$

$$
Q_c=Q_r\cdot rotation
$$

$$
K_c=K_r\cdot rotation
$$

$$
q\_out,q\_scale=\operatorname{DynamicQuantFP8}(Q_c)
$$

$$
k\_quant,k\_scale=\operatorname{DynamicQuantFP8}(K_c)
$$

$$
v\_quant=\operatorname{FP8}(V\cdot v\_scale)
$$

最后：

```text
q_out、q_scale → 普通输出
k_quant、k_scale → 根据 slot_mapping 写入 K Cache
v_quant → 根据 slot_mapping 写入 V Cache
```

## 14. 当前 RoPE 代码实现伪代码

下面的伪代码按照当前普通 RoPE 模式整理。它表达实际代码的数据组织和计算顺序，但省略了 tile 双缓冲、
AIV/AIC 事件同步以及 TND/NTD 的具体地址模板。

```text
常量：
    D        = 128
    HALF_D   = 64
    FP8_MAX  = 448

输入：
    qkv
    q_gamma[128]
    k_gamma[128]
    cos_sin[MaxSeqLen,128]
    query_start_loc[Batch+1]
    seq_lens[Batch]
    rotation[128,128]
    v_scale[Nv]
    slot_mapping[T]


阶段一：推导每个token的RoPE位置，并把对应cos/sin行搬入UB

    for batch in 0 .. Batch-1:
        token_begin = query_start_loc[batch]
        token_end   = query_start_loc[batch+1]
        local_len   = token_end - token_begin

        # seq_lens表示追加当前token后的完整序列长度
        history_offset = max(seq_lens[batch] - local_len, 0)

        for token in token_begin .. token_end-1:
            local_pos = token - token_begin
            position  = history_offset + local_pos

            # 当前tile中每个token只保存一行长度为128的cos/sin数据
            cos_sin_ub[token,0:128] = cos_sin[position,0:128]

    # 实际代码会把同一batch内位置连续的多行合并为一次二维搬运


阶段二：根据slot_mapping提前计算K/V Cache目标地址

    for token in 0 .. T-1:
        slot = slot_mapping[token]

        block_id     = slot // BlockSize
        block_offset = slot % BlockSize

        cache_offset[token] =
            block_id * cache_block_stride
            + block_offset * cache_token_stride

        scale_cache_offset[token] =
            block_id * scale_cache_block_stride
            + block_offset * scale_cache_token_stride


阶段三：V分支使用两组64维Vector完成静态缩放和FP8转换

    v_input_base = qkv_ub中第(Nq+Nk)个head的起点

    # 真实代码是head外循环、token内循环；同一head的所有token复用一次scale广播
    for v_head in 0 .. Nv-1:
        scale_reg[0:64] = VectorBroadcastFP32(v_scale[v_head])

        for token in 0 .. tileTokenSize-1:
            input_addr =
                v_input_base
                + v_head * inputHeadStride
                + token * inputTokenStride

            # 两次Vector Load覆盖完整D=128，不存在逐d标量循环
            v_low_bf16[0:64]  = VectorLoad64BF16(input_addr)
            v_high_bf16[0:64] = VectorLoad64BF16(input_addr + 64)

            v_low_fp32[0:64]  = VectorCastBF16ToFP32(v_low_bf16)
            v_high_fp32[0:64] = VectorCastBF16ToFP32(v_high_bf16)

            scaled_low[0:64]  = VectorMul(v_low_fp32,  scale_reg)
            scaled_high[0:64] = VectorMul(v_high_fp32, scale_reg)

            v_low_fp8[0:64] = VectorCastFP32ToFP8(
                scaled_low,
                round=RINT,
                saturate=true
            )

            v_high_fp8[0:64] = VectorCastFP32ToFP8(
                scaled_high,
                round=RINT,
                saturate=true
            )

            # V的UB临时输出固定使用[Nv,tileTokenSize,128]的head-major布局
            output_addr =
                v_out_ub
                + v_head * tileTokenSize * 128
                + token * 128

            VectorStore64FP8(output_addr,      v_low_fp8)
            VectorStore64FP8(output_addr + 64, v_high_fp8)

    # 再按token遍历，通过二维搬运把同一token的所有V heads写入目标Cache slot
    for token in 0 .. tileTokenSize-1:
        DataCopyUbToGm2D(
            dst = v_cache + cache_offset[token],
            src = v_out_ub + token * 128,
            blockCount = Nv,
            blockLen = 128个FP8,
            srcStride = tileTokenSize * 128,
            dstStride = kvCacheStrideHead
        )


阶段四：Q/K执行RMSNorm和half-split RoPE

    # Q、K分别调用一次VF；每个分支只加载一次自己的gamma
    for branch in [Q,K]:
        gamma     = q_gamma if branch == Q else k_gamma
        head_num  = Nq      if branch == Q else Nk
        head_base = 0       if branch == Q else Nq

        gamma_low[0:64]  = VectorLoad64FP32(gamma[0:64])
        gamma_high[0:64] = VectorLoad64FP32(gamma[64:128])

        for token in 0 .. tileTokenSize-1:
            cos[0:64] = VectorLoad64FP32(cos_sin_ub[token,0:64])
            sin[0:64] = VectorLoad64FP32(cos_sin_ub[token,64:128])

            for head in 0 .. head_num-1:
                input_addr = 当前token、head_base+head在qkv_ub中的起点

                x_low_bf16[0:64]  = VectorLoad64BF16(input_addr)
                x_high_bf16[0:64] = VectorLoad64BF16(input_addr + 64)

                x_low[0:64]  = VectorCastBF16ToFP32(x_low_bf16)
                x_high[0:64] = VectorCastBF16ToFP32(x_high_bf16)

                square_sum_vector[0:64] =
                    VectorMul(x_low,x_low)
                    + VectorMul(x_high,x_high)

                square_sum = VectorReduceSum(square_sum_vector)
                reciprocal_rms = 1 / sqrt(square_sum / 128 + epsilon)

                norm_low[0:64] =
                    VectorMul(x_low, reciprocal_rms)
                    * gamma_low

                norm_high[0:64] =
                    VectorMul(x_high, reciprocal_rms)
                    * gamma_high

                rope_low[0:64] =
                    VectorMul(norm_low,cos)
                    - VectorMul(norm_high,sin)

                rope_high[0:64] =
                    VectorMul(norm_high,cos)
                    + VectorMul(norm_low,sin)

                rope_low_bf16[0:64]  = VectorCastFP32ToBF16(rope_low)
                rope_high_bf16[0:64] = VectorCastFP32ToBF16(rope_high)

                # 不先生成完整ND输出，而是直接按NZ目标地址Scatter到UB
                ScatterNzBf16(
                    rope_nz_ub,
                    rope_low_bf16,
                    rope_high_bf16,
                    token,
                    head
                )

    # 搬到L1时跳过UB中16k+1步长产生的padding，形成标准NZ布局
    CopyNzUbToL1(q_rope_nz_ub, q_rope_nz_l1)
    CopyNzUbToL1(k_rope_nz_ub, k_rope_nz_l1)


阶段五：Cube执行RoPE后的rotation矩阵乘

    q_cube = MatMul(
        q_rope_bf16,
        rotation_bf16,
        accumulate_dtype=FP32
    )

    k_cube = MatMul(
        k_rope_bf16,
        rotation_bf16,
        accumulate_dtype=FP32
    )


阶段六：Q/K使用两组64维Vector执行逐token、逐head动态FP8量化

    # TND与NTD会选择不同的token/head遍历顺序和地址stride；每一行的计算相同
    for each (token,head) row in q_cube:
        q_low[0:64]  = VectorLoad64FP32(q_row_addr)
        q_high[0:64] = VectorLoad64FP32(q_row_addr + 64)

        abs_max_vector[0:64] =
            VectorMax(VectorAbs(q_low), VectorAbs(q_high))

        max_abs = VectorReduceMax(abs_max_vector)
        scale   = max_abs / FP8_MAX
        scale_reg[0:64] = VectorBroadcastFP32(scale)

        q_low_scaled[0:64]  = VectorDiv(q_low,  scale_reg)
        q_high_scaled[0:64] = VectorDiv(q_high, scale_reg)

        q_low_fp8[0:64] = VectorCastFP32ToFP8(
            q_low_scaled,
            round=RINT,
            saturate=true
        )

        q_high_fp8[0:64] = VectorCastFP32ToFP8(
            q_high_scaled,
            round=RINT,
            saturate=true
        )

        VectorStore64FP8(q_out_ub_row,      q_low_fp8)
        VectorStore64FP8(q_out_ub_row + 64, q_high_fp8)
        StoreFirstElement(q_scale_ub[token,head], scale)

    for each (token,head) row in k_cube:
        k_low[0:64]  = VectorLoad64FP32(k_row_addr)
        k_high[0:64] = VectorLoad64FP32(k_row_addr + 64)

        abs_max_vector[0:64] =
            VectorMax(VectorAbs(k_low), VectorAbs(k_high))

        max_abs = VectorReduceMax(abs_max_vector)
        scale   = max_abs / FP8_MAX
        scale_reg[0:64] = VectorBroadcastFP32(scale)

        k_low_fp8[0:64] = VectorCastFP32ToFP8(
            VectorDiv(k_low,scale_reg),
            round=RINT,
            saturate=true
        )

        k_high_fp8[0:64] = VectorCastFP32ToFP8(
            VectorDiv(k_high,scale_reg),
            round=RINT,
            saturate=true
        )

        VectorStore64FP8(k_quant_ub_row,      k_low_fp8)
        VectorStore64FP8(k_quant_ub_row + 64, k_high_fp8)
        StoreFirstElement(k_scale_ub[token,head], scale)

    按cache_offset把k_quant_ub写入k_cache
    按scale_cache_offset把k_scale_ub写入k_scale_cache


阶段七：按照输出布局搬出Q

    if layout_q_out == TND:
        保存q_out为[T,Nq,128]
        保存q_scale为[T,Nq]

    else if layout_q_out == NTD:
        保存q_out为[Nq,T,128]
        保存q_scale为[Nq,T]
```

这段伪代码体现了普通 RoPE 和 M-RoPE 实现上的三个主要区别：

1. 普通 RoPE 每个 token 只搬入一行 `cos_sin[position,:]`，不需要 T/H/W 三行和 `gather_index`；
2. 同一个 token 的所有 Q/K head 共用这一行中的 64 个 cos 和 64 个 sin；
3. 普通 RoPE 的 Q/K 都在 Cube 后动态量化为 FP8，并分别产生 `q_scale` 和 `k_scale_cache`。

实际流水中，V 分支会提前启动并与 Q/K 预处理交叠；这只改变执行时序，不改变上面的数学依赖关系。

## 15. 推荐源码阅读顺序

建议按照以下顺序阅读：

1. `01_operator_proto_and_formula.md`：理解输入、shape 和数学公式；
2. `assets/golden.py`：查看最接近公式的参考实现；
3. `op_kernel/qkv_rms_norm_rope_cache_with_k_scale_apt.cpp`：查看 kernel 入口和模板模式；
4. `op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_controller.h`：查看核间任务和 token 分块；
5. `op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_basic_block.h`：查看 AIV/AIC 流水组织；
6. `op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_vec.h`：查看 RMSNorm、RoPE、量化和 Cache 写回；
7. `op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_cube.h`：查看 `rotation` 矩阵乘；
8. 最后阅读 tiling、核间同步和核内同步设计。

Golden 最适合用于建立正确性认识；Vec/Cube 代码则是在相同数学语义上加入数据搬运、切块、双缓冲和事件同步。
