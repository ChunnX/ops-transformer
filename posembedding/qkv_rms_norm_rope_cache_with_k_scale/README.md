# QkvRmsNormRopeCacheWithKScale

## 产品支持情况

| 产品 | 是否支持 |
|:---|:---:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

### 算子概述

算子在一次调用中完成融合QKV处理、Q/K归一化与位置编码、共享矩阵乘、Q/K/V量化以及PagedAttention KV Cache更新。输入`qkv`沿逻辑head轴拆分为Q、K、V，Q结果写入`q_out`，K/V结果按照`slot_mapping`原地写入Cache，K的动态量化scale写入`k_scale_cache`。

算子包含两条位置编码路径：

| 路径 | Q分支 | K分支 | V分支 | Cache更新 |
|---|---|---|---|---|
| RoPE | RMSNorm、RoPE、共享矩阵乘、FP8动态量化 | RMSNorm、RoPE、共享矩阵乘、FP8动态量化 | 按`v_scale`缩放并转换为FP8 | 更新FP8 K Cache、FP8 V Cache和K scale Cache |
| M-RoPE | RMSNorm、M-RoPE、共享矩阵乘，输出BF16 | RMSNorm、M-RoPE、共享矩阵乘、INT8动态量化 | 按`v_scale`缩放并转换为FP8 | 更新INT8 K Cache、FP8 V Cache和K scale Cache |

该融合流程面向Transformer推理中的PagedAttention Cache更新：Q供当前注意力计算使用，K/V在对应slot写入Cache，后续调用可以直接复用更新后的Cache内容。

### 计算流程

#### 1. 融合输入拆分

设`head_nums=[Nq, Nk, Nv]`，将`qkv`的逻辑视图记为$X$，沿head轴拆分为

$$
Q=X[...,0:N_q,:],\quad
K=X[...,N_q:N_q+N_k,:],\quad
V=X[...,N_q+N_k:N_q+N_k+N_v,:].
$$

`layout_qkv`只改变T轴和N轴在内存中的排列，不改变上述逻辑拆分。TND布局的逻辑形状为`[T,N,D]`，NTD布局的逻辑形状为`[N,T,D]`。

#### 2. Q/K RMSNorm

Q和K使用独立的权重`q_gamma`、`k_gamma`进行归一化，V分支不参与RMSNorm。对输入向量$x$和权重$\gamma$，输出为

$$
y_d=\frac{x_d\gamma_d}{\sqrt{\epsilon+\frac{1}{D}\sum_{j=0}^{D-1}x_j^2}}.
$$

归一化结果以BF16参与后续位置编码和共享矩阵乘。

#### 3. 位置编码

RoPE路径使用`query_start_loc`和`seq_lens`恢复每个token在当前序列中的位置。设batch $b$在本次调用中包含$L_b$个token，则

$$
L_b=query\_start\_loc[b+1]-query\_start\_loc[b],\qquad
p_{b,i}=seq\_lens[b]-L_b+i.
$$

token $(b,i)$从`cos_sin[p_{b,i}, :]`读取位置编码；前半段为cos，后半段为sin。将归一化结果按head维拆成low/high两半，对每个half lane执行

$$
\begin{aligned}
z_{low}&=y_{low}\cdot cos-y_{high}\cdot sin,\\
z_{high}&=y_{high}\cdot cos+y_{low}\cdot sin.
\end{aligned}
$$

M-RoPE路径通过`mrope_position[u,0/1/2]`为每个token选择T/H/W三路位置编码行，并由`mrope_section=[t,h,w]`控制H/W lane在half维中的交织位置。T路作为默认来源，H/W路在各自section覆盖的lane中替换T路；Q和K使用相同的位置选择规则。

#### 4. Q/K共享矩阵乘

位置编码后的Q和K使用同一个`rotation`矩阵。对每个token和head，计算为

$$
q_{rot}=z_q\ @\ rotation,\qquad k_{rot}=z_k\ @\ rotation.
$$

矩阵乘结果以FP32进行量化统计，再按目标输出类型转换。

#### 5. Q/K动态量化与输出

量化按token、head独立计算最大绝对值。设$M$为目标类型的量化常数，则

$$
s=\frac{\max_d|x_d|}{M},\qquad x_{quant}=cast\left(\frac{x}{s}\right).
$$

FP8 E4M3FN使用$M=448$，INT8使用$M=127$。RoPE路径将量化后的Q写入`q_out`，对应scale写入`q_scale`；M-RoPE路径将BF16 Q结果写入`q_out`，Q不执行动态量化。

#### 6. V量化

V不执行位置编码和共享矩阵乘，按head对应的`v_scale`缩放后转换为FP8：

$$
v_{fp8}=cast(v\cdot v\_scale).
$$

#### 7. KV Cache写回

对token $t$，`slot_mapping[t]`给出Cache中的逻辑slot。令$B$为Cache的BlockSize，则

$$
blockId=\left\lfloor\frac{slot\_mapping[t]}{B}\right\rfloor,\qquad
blockOffset=slot\_mapping[t]\bmod B.
$$

算子将量化后的K、V以及K scale写入对应位置：

$$
\begin{aligned}
k\_cache[blockId,n,blockOffset,:]&=k_{quant}[t,n,:],\\
v\_cache[blockId,n,blockOffset,:]&=v_{fp8}[t,n,:],\\
k\_scale\_cache[blockId,n,blockOffset,0]&=k\_scale[t,n].
\end{aligned}
$$

K Cache、V Cache和K scale Cache均采用输入输出同地址复用方式；未被本次`slot_mapping`选中的Cache位置保持不变。

### 布局与输出语义

`layout_q_out`控制`q_out`及`q_scale`的T/N轴排列：TND输出按`[T,Nq,D]`组织，NTD输出按`[Nq,T,D]`组织。QKV输入布局和Q输出布局可以独立表达，算子的逻辑计算顺序不随物理布局改变。

RoPE路径提供量化Q和Q scale；M-RoPE路径提供BF16 Q，公开接口不返回Q scale。两条路径都返回原地更新后的K/V Cache和K scale Cache，Cache的head、block和token位置由输入shape、stride以及`slot_mapping`共同确定。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|---|---|---|---|---|
| qkv | 输入 | Q/K/V融合输入。`layout_qkv="TND"`时shape为`[T, Nq+Nk+Nv, D]`，`layout_qkv="NTD"`时shape为`[Nq+Nk+Nv, T, D]`。 | BFLOAT16 | ND |
| q_gamma | 输入 | Q分支RMSNorm权重，shape为`[D]`。 | FLOAT | ND |
| k_gamma | 输入 | K分支RMSNorm权重，shape为`[D]`。 | FLOAT | ND |
| cos_sin | 输入 | RoPE/M-RoPE位置编码表，shape为`[MaxSeqLen, D]`。前`D/2`列为cos，后`D/2`列为sin。 | FLOAT | ND |
| slot_mapping | 输入 | 每个token写入cache的slot索引，shape为`[T]`。 | INT32 | ND |
| k_cache | 输入/输出 | K Cache，输入输出同地址复用，shape为`[BlockNum, Nk, BlockSize, D]`，支持非连续Tensor。RoPE为FP8 E4M3FN，M-RoPE为INT8。 | FLOAT8_E4M3FN / INT8 | ND |
| v_cache | 输入/输出 | V Cache，输入输出同地址复用，shape为`[BlockNum, Nv, BlockSize, D]`，支持非连续Tensor。 | FLOAT8_E4M3FN | ND |
| k_scale_cache | 输入/输出 | K动态量化scale cache，输入输出同地址复用，shape为`[BlockNum, Nk, BlockSize, 1]`，支持非连续Tensor。 | FLOAT | ND |
| query_start_loc | 可选输入 | RoPE场景当前调用内各batch token数的前缀和，shape为`[Batch+1]`；M-RoPE场景不传。 | INT32 | ND |
| seq_lens | 可选输入 | RoPE场景每个batch追加当前token后的实际序列长度，shape为`[Batch]`；M-RoPE场景不传。 | INT32 | ND |
| rotation | 可选输入 | Q/K共享矩阵乘权重，shape为`[D, D]`。 | BFLOAT16 | ND |
| v_scale | 可选输入 | V分支量化缩放因子，shape为`[Nv]`。 | FLOAT | ND |
| mrope_position | 可选输入 | M-RoPE场景token-major位置索引，shape为`[T,3]`，三列依次为T/H/W；RoPE场景不传。 | INT32 | ND |
| q_out | 输出 | Q输出，数据类型由位置编码路径和Q输出配置决定，T/N轴排列由`layout_q_out`决定。 | FLOAT8_E4M3FN / BFLOAT16 | ND |
| q_scale | 输出 | Q动态量化scale。RoPE场景返回FP32 scale；M-RoPE场景在ACLNN/Torch接口中分别为`nullptr`/`None`。 | FLOAT | ND |
| head_nums | 属性 | Q/K/V头数，按`[Nq, Nk, Nv]`顺序传入。 | INT64数组 | - |
| layout_qkv | 可选属性 | 控制`qkv`的N/T轴布局，默认值为`"TND"`。 | STRING | - |
| layout_q_out | 可选属性 | 控制`q_out`和`q_scale`的N/T轴布局，默认值为`"NTD"`。 | STRING | - |
| epsilon | 可选属性 | RMSNorm防除零参数，默认值为`1e-6`。 | FLOAT | - |
| mrope_section | 可选属性 | M-RoPE的T/H/W section 参数，按`[t,h,w]`顺序传入。 | INT64数组 | - |
| q_quant_mode | 可选属性 | Q量化模式配置。 | STRING | - |
| q_out_dtype | 可选属性 | Q输出类型配置。 | INT64 | - |

## 约束说明

- 输入shape限制：
  - 当前实现仅支持`D=128`，RoPE和M-RoPE均要求`0<Nq<=64`、`Nq=8*Nk`、`Nk=Nv`。
  - RoPE场景必须同时传`query_start_loc/seq_lens`，不传`mrope_position`且`mrope_section`为空或未传；
    `q_quant_mode=PerTokenPerHead`、`q_out_dtype=FP8 E4M3FN`、`q_scale`非空，K Cache为FP8。
  - M-RoPE场景必须同时传`mrope_position[T,3]`和非空`mrope_section=[t,h,w]`，不传`query_start_loc/seq_lens`；
    `q_quant_mode=NoQuant`、`q_out_dtype=BF16`、公开`q_scale=None`，K Cache为INT8，且仅支持
    `layout_qkv="TND"`、`layout_q_out="TND"`。两组位置输入不能同时存在或同时缺失。
  - M-RoPE的`mrope_section`长度必须为3。D128时三项分别受`[22,21,21]`上限约束；这些互不重叠的lane容量使三项总量自然不超过`64`，第0项
    只参与值域校验，不参与lane选源。`mrope_position`必须为INT32、rank 2、shape `[T,3]`。
  - `layout_qkv`控制`qkv`的N/T轴布局，默认值为`"TND"`；`layout_q_out`控制`q_out`和`q_scale`的N/T轴布局，默认值为`"NTD"`：
    - `layout_qkv="TND"`，`layout_q_out="TND"`：`qkv=[T, Nq+Nk+Nv, D]`，`q_out=[T, Nq, D]`，`q_scale=[T, Nq]`。
    - `layout_qkv="TND"`，`layout_q_out="NTD"`：`qkv=[T, Nq+Nk+Nv, D]`，`q_out=[Nq, T, D]`，`q_scale=[Nq, T]`。
    - `layout_qkv="NTD"`，`layout_q_out="NTD"`：`qkv=[Nq+Nk+Nv, T, D]`，`q_out=[Nq, T, D]`，`q_scale=[Nq, T]`。
  - `cos_sin`第二维必须为`D`，第一维必须覆盖本次调用会访问的RoPE/M-RoPE位置。
  - `k_cache`、`v_cache`和`k_scale_cache`支持非连续Tensor，需符合以下约束：
    - `k_cache`和`v_cache`均为4维正stride，最后一维stride为1，head维和token维stride均不小于`D=128`；
      K Cache dtype 按场景分别为FP8或INT8，V Cache始终为FP8。
    - `k_scale_cache`为4维正stride，最后一维stride为1。
    - `k_cache`和`v_cache`前三维stride必须一致。
- 输入值域限制：
  - RoPE场景中，`query_start_loc`表示当前调用内token的batch前缀和，`query_start_loc[0]`应为0，`query_start_loc[Batch]`应等于`T`。
  - RoPE场景中，`seq_lens[b]`表示第`b`个batch追加本次token后的实际序列长度。对该batch内第`i`个token，RoPE位置由`seq_lens[b] - (query_start_loc[b+1] - query_start_loc[b]) + i`得到；调用方需保证`seq_lens[b] >= query_start_loc[b+1] - query_start_loc[b]`。若`seq_lens[b]`小于该batch本次调用的token数，行为未定义。
  - `slot_mapping`取值范围应为`[0, BlockNum * BlockSize - 1]`。同一次调用内多个token写入同一slot会对同一cache地址产生重复写，最终写入顺序和结果未定义。
  - 资源边界约束：`Nq+Nk <= 128`，`Nq+Nk+Nv <= 160`，`Nv <= 80`，`ceil_align(Nq,16)+ceil_align(Nk,16) <= 256`。
- 输入属性限制：
  - `head_nums`必须包含3个正整数，顺序为`[Nq, Nk, Nv]`，并满足`Nq<=64`、`Nq=8*Nk`、
    `Nk=Nv`。
  - `layout_qkv`和`layout_q_out`大小写敏感，仅支持`"TND"`和`"NTD"`，且当前不支持`layout_qkv="NTD"`、`layout_q_out="TND"`。
- 输入数据类型限制：
  - 各输入的数据类型和数据格式需符合参数说明，不支持隐式类型转换。
  - 输入均为ND格式，不支持私有格式。
- 其他限制：
  - `rotation`和`v_scale`为可选输入，当前不支持不传或空Tensor。
  - 非可选输入和输出不支持空Tensor；可选位置输入按场景合同使用`None`，空`mrope_section=[]`等价于未传。

## 调用说明

<term>Ascend 950PR/Ascend 950DT</term>

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| aclnn API | [test_aclnn_qkv_rms_norm_rope_cache_with_k_scale](examples/test_aclnn_qkv_rms_norm_rope_cache_with_k_scale.cpp) | 通过[aclnnQkvRmsNormRopeCacheWithKScale](docs/aclnnQkvRmsNormRopeCacheWithKScale.md)接口方式调用QkvRmsNormRopeCacheWithKScale算子。 |
| PyTorch API | - | 通过[qkv_rms_norm_rope_cache_with_k_scale](../../torch_extension/cann_ops_transformer/docs/zh/qkv_rms_norm_rope_cache_with_k_scale.md)接口方式调用QkvRmsNormRopeCacheWithKScale算子。 |
