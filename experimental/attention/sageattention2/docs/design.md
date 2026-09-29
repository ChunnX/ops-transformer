# 需求背景

## 需求来源

随着大语言模型（LLM）与多模态生成模型（如 CogVideoX、HunyuanVideo、Flux 等 DiT 架构）的快速发展，自注意力（Self-Attention）机制在长序列推理场景下的计算与显存开销已成为主要的性能瓶颈。

在昇腾 CANN 算子生态体系中，为加速长上下文推理并降低片上内存占用，亟需基于开源前沿量化注意力算法 SageAttention2（ICML 2025），在 Atlas 950PR 硬件上通过 AscendC 与 CATLASS 联合开发原生的 INT8 QK^T + FP8 PV 高性能融合注意力算子。

算子交付路径为 ops-transformer 仓的 experimental/attention/sageattention2 目录，性能验收对标 ops-transformer 官方生产基线算子 FIA（FusedInferAttentionScoreV5 / npu_fused_infer_attention_score）。

## 背景介绍

### 1. 算子实现优化背景

常规融合注意力内核以 FP16/BF16 浮点张量进行计算。随着序列长度扩展至 8K、16K 乃至更大，Attention 计算复杂度呈 O(S^2) 增长。虽然 INT8/FP8 低精度量化能够将计算吞吐提升至原先的 2 倍以上，但传统量化方法面临两大约束：

- Key 矩阵中存在极少量的通道离群峰值（Outliers），若直接对特征维进行静态或全局量化，截断误差会导致注意力概率分布严重失真。
- 低精度矩阵乘累加（MMA）在超长序列上长时间累加，极易产生下溢或有效位截断误差。

SageAttention2 通过数学恒等的离群值平滑技术（Outlier Smoothing）、两级累加机制（Two-Level Accumulation）以及细粒度量化，在提升推理吞吐的同时，保持与 FP16/FP32 相当的生成质量。

### 2. 官方基线现状分析

ops-transformer 仓中的 FusedInferAttentionScore（FIA）算子是目前昇腾平台上的标准注意力推理算子：

- aclnn 接口：aclnnFusedInferAttentionScoreV5
- PyTorch 接口：torch_npu.npu_fused_infer_attention_score

当前 FIA 在 Atlas 950PR 上的典型实现多基于 FP16 计算路径。在大序列长度与高并发推理场景下，FIA 受限于 HBM 带宽及 FP16 Cube 算力上限，难以利用 Atlas 950PR 的原生 INT8 与 FP8 峰值算力。

| 算子 | 计算精度 | 支持布局 | 主要硬件瓶颈 |
| --- | --- | --- | --- |
| 官方基线 FIA | FP16 / BF16 | BNSD / BSH / BSND | 显存带宽与 FP16 浮点算力上限 |
| 本算子 SageAttention2 | INT8 QK + MXFP8 PV | HND（零拷贝直通）/ NHD（Host 侧一次性规整拷贝） | 片上 SRAM (UB/L1) 搬运与异步流水编排 |

### 3. 核心功能与应用场景分析

本设计实现面向 Atlas 950PR 的 INT8 QK^T + FP8 PV 融合注意力算子：

- 核心功能：
  - Key 序列均值统计（outlier smoothing）与 per-token 细粒度 INT8 量化。
  - Value 矩阵 MXFP8（E4M3 数据 + 每 32 token 每 channel 一个 E8M0 scale）量化。
  - Cube INT8 GEMM-1 与 Cube MXFP8 GEMM-2 异步协同。
  - AIV 向量核心在线 Softmax（FP16 score/max + ln(448) 平移 + FP8 P 饱和转换）与两级 FP32 累加归约。
  - 动态计算并按需返回自然底数对数归一化因子 LSE。
- 适用场景：
  - 视频生成（CogVideoX、HunyuanVideo）、图像生成（Flux.1 等）。
  - 长文本大语言模型 Prefill 阶段的高吞吐量化推理。

---

# 需求分析

## 需求描述

在 Atlas 950PR（CANN 9.2.0-beta.1）环境下，基于 AscendC 与 CATLASS 联合开发高性能量化注意力算子，实现功能对标与精度对齐，性能满足相对官方基线 FIA 的加速比门禁。

核心交付内容涵盖：
- PyTorch 适配层：sageattn 与 sageattn_qk_int8_pv_fp8_asc。
- aclnn C API：aclnnSageAttention2GetWorkspaceSize 与 aclnnSageAttention2。
- AscendC + CATLASS 混合架构计算内核与 Host 侧 Tiling 调度系统。

## 需求拆解

1. 数据类型支持：支持输入 FP16 与 BF16，计算过程内部下沉为 INT8 QK 与 MXFP8 PV，输出与 Query 保持同精度。
2. 形状与布局支持：支持 head_dim 在 (0, 128] 范围（内部自动补齐至 64 或 128，输出裁剪还原）；支持 HND 与 NHD 张量布局，末维连续；HND 连续输入全程零拷贝，NHD 输入在 aclnn 层经视图规整后由框架做一次连续化拷贝。
3. 高级注意力特性：支持 Grouped Query Attention (GQA，Hq 可被 Hk 整除)；支持非因果与因果（is_causal=True，要求 Sq == Sk）；支持可选输出归一化因子的自然对数 LSE。
4. 精度门禁（硬性）：
   - L-A 标杆（GPU 同配置）：FP16 满足 rtol <= 2^-9，atol <= 2^-9，元素匹配率 >= 99.0%；BF16 满足 rtol <= 2^-6，atol <= 2^-6，元素匹配率 >= 99.0%。
   - L-B 标杆（FP32 SDPA 真值）：Max_Error_Ratio <= 2.0，Mean_Error_Ratio <= 1.2，RMSE_Ratio <= 1.2。
5. 性能门禁（硬性）：在 Atlas 950PR 上，基准测试用例 P-01 至 P-07 全部用例中，PyTorch 接口与 aclnn 接口两条路径相对官方基线 FIA 均须满足加速比 Speedup >= 1.60x。

---

# 详细设计

## 算子分析

### 1. 核心数学推导

#### 1.1 离群值平滑归一化数学恒等性证明与动态范围分析

注意力机制计算的核心为：

```text
S = Q * K^T
P = softmax(S * scale)
O = P * V
```

Key 矩阵在通道维度（HeadDim 轴）存在静态离群通道（Outlier Channels），其通道均值绝对值较大，各通道动态范围差异明显。若直接按全局最大绝对值 `amax` 量化，有效信号会被压缩到低有效位，甚至截断为 0。

离群平滑技术通过在序列维度减去 Key 的通道均值向量 `mean_K`，将矩阵中心化。以下证明该平滑操作在数学上对 Softmax 结果保持严格恒等：

```python
# 序列均值计算：
# Q 形状: [S_Q, D]，K 形状: [S_K, D]
# 沿序列维度计算 K 的通道均值：
mean_K = (1.0 / S_K) * sum(K[j, :] for j in range(S_K))  # 形状: [1, D]

# 中心化平滑 Key：
K_smooth[j, :] = K[j, :] - mean_K                        # 形状: [S_K, D]

# 未平滑的注意力分数原始点积：
S[i, j] = dot(Q[i, :], K[j, :])

# 平滑后的注意力分数点积：
S_smooth[i, j] = dot(Q[i, :], K_smooth[j, :])

# 数学恒等展开：
# S[i, j] = dot(Q[i, :], K_smooth[j, :] + mean_K)
#         = dot(Q[i, :], K_smooth[j, :]) + dot(Q[i, :], mean_K)
#         = S_smooth[i, j] + C[i]

# 其中标量偏移 C[i] 定义为：
# C[i] = dot(Q[i, :], mean_K)
# 注意：对于固定的 Query 行 i，C[i] 是一个完全不依赖于 Key 列索引 j 的标量常数。

# 带入带缩放因子 sm_scale 的 Softmax 归一化公式：
# softmax(sm_scale * S[i, :])_j
#   = exp(sm_scale * S[i, j]) / sum_k(exp(sm_scale * S[i, k]))
#   = exp(sm_scale * (S_smooth[i, j] + C[i])) / sum_k(exp(sm_scale * (S_smooth[i, k] + C[i])))
#   = (exp(sm_scale * C[i]) * exp(sm_scale * S_smooth[i, j])) / (exp(sm_scale * C[i]) * sum_k(exp(sm_scale * S_smooth[i, k])))
#   = exp(sm_scale * S_smooth[i, j]) / sum_k(exp(sm_scale * S_smooth[i, k]))
#   = softmax(sm_scale * S_smooth[i, :])_j
```

数学结论：平滑项 `C[i]` 在 Softmax 分子分母中被严格公因子消去，未引入任何近似误差。工程实现（`stats.h`）在 AIV 上以 FP32 分批累加求均值，每个 KV head 由单个 AIV 独占累加并写回，不使用跨核原子加，重复调用逐位确定。

动态范围分析：
减去 `mean_K` 后，Key 各特征维度的直流偏置（DC bias）被移除，矩阵的最大绝对值 `amax` 下降。在 INT8 对称量化 `[-127, 127]` 区间内，量化阶跃步长 `scale = amax / 127.0` 随之减小，小权重不再因截断归零。

Value 侧的同一恒等式：`smooth_v` 对 V 沿序列维取通道均值 `mean_V[d]`，量化与累加使用 `V' = V - mean_V`。由于归一化权重每行和为 1：

```text
sum_j P[i, j] * V[j, d] = sum_j P[i, j] * (V'[j, d] + mean_V[d])
                        = sum_j P[i, j] * V'[j, d] + mean_V[d]
```

补偿项是每通道一个标量，在输出归一化之后、写入 O 之前加一次（`output_update.h` 的 normalize 通路），不进入跨块累加。`mean_V` 与 `mean_K` 共用 `stats.h` 的分批 FP32 统计，按 KV head 各写一份（逐 head 独占写核），GQA 下由查询侧广播读取。居中后的 V 需要 FP32 的运行状态承接，因此该开关只在 `pv_accum_dtype="fp32"` 下生效，其余取值按 §1.4 的映射告警并忽略。

#### 1.2 注意力得分定点累加与硬件反量化平衡缩放推导

Q/K 以 INT8 进入 Cube 后，GEMM-1 在 L0C 中以 INT32 精确累加（无舍入误差）。得分的反量化与缩放不在 AIV 上逐元素完成，而是拆分为"硬件 Fixpipe 常数反量化 + 量化期预融合 scale"两步，全部乘法融合到既有的向量乘法中（`attention_block_cube.h` 的 Fixpipe `DEQF16` 配置与 `scale.h` 的 `PackFixedScaleBlock`）；式中的 `sqrt(|sm_scale|·2048)` 与符号选择由 Host tiling 一次性预计算为 `SageParams::qFactor/kFactor`，内核仅把该标量广播进向量寄存器（`PackFixedScaleBlock` 的单条 `Duplicate`）并在量化器遍历该行时顺带完成 FP16 打包，AIV 不再执行 Sqrt 与符号比较，也不额外占一趟向量化：

```text
第一步（Fixpipe 硬件反量化）：
  L0C INT32 结果经 Fixpipe 的 quantPre = DEQF16、deqScalar = 2^-11 (0x3A000000)
  直接搬运为 FP16 得分矩阵：
  S_fp16 = (Q_int8 · K_int8^T) * 2^-11

第二步（量化期 scale 预融合，scale.h）：
  量化 Q/K 时生成 FP16 平衡 scale：
  qh = qs * sign(sm_scale) * sqrt(|sm_scale| * 2048)
  kh = ks * sqrt(|sm_scale| * 2048)
  其中 qs = amax_q / 127、ks = amax_k / 127（沿 head_dim 取 amax，scale 在连续
  token 组内共享，组大小由 qk_quant_gran 决定，见 §1.4 参数映射）。

  AIV 在 Softmax 中将得分乘以 (qh ⊗ kh)（softmax.h 中每行一次向量乘）：
  S' = S_fp16 * qh * kh
     = (Q_int8 · K_int8^T) * 2^-11 * qs * ks * 2048 * sm_scale
     = (Q · K^T) * sm_scale                        （2048 * 2^-11 = 1，精确抵消）
```

数学结论：`2048 * 2^-11 = 1`，DEQF16 引入的 2^-11 缩放被预融合进 scale 的 2048 因子严格抵消，最终 FP16 得分即 `sm_scale * Q·K^T` 的量化近似，且全程只需每行一次 FP16 向量乘法，无独立反量化通道。该方案对齐开源 SageAttention 在 sm89 per_thread 路径的 FP16 平衡 scale 手法（开源侧将 2048 因子折叠进 qk scale，硬件 mma 输出以 FP16 承接）。工程上 qh/kh 各含一次 FP16 RNE 舍入，恒等式的相对误差上界为 2^-9，由 `tests/numeric_error_test.cpp` 断言 [5] 以位级 FP16 RNE 镜像校验。

负 sm_scale 的符号位折叠在 Q 的 scale 上（`Select` 按 `sm_scale < 0` 取负），K scale 恒正，保证乘积符号正确。

#### 1.3 在线归一化平移与饱和映射推导

P 需要量化为 E4M3（最大正常数 448）进入 GEMM-2。若直接量化 `exp(S' - max)`，其值域 (0, 1] 仅利用 E4M3 顶部 9 个码点，量化噪声极大。本算子（`softmax.h` / `softmax_wide.h`，常量定义于 `common.h`）对在线最大值状态统一施加 `-ln(448)` 平移：

```text
设第 i 块的行最大值为 blockmax，维护的平移最大值状态：
  m̂ = max(m̂_old, blockmax - FP16(ln 448))          （初始 m̂_old = -65504）

概率与分母（ExpSub 为自然指数硬件指令）：
  p_j  = exp(S'_j - m̂) = 448 * exp(S'_j - blockmax) <= 448   （占满 E4M3 量程）
  den  = den * exp(m̂_old - m̂) + Σ_j p_j           （分母与概率使用同一平移基准）

P 的 E4M3 转换使用 SatMode::SAT 饱和舍入；CPU 参考实现（reference.py）显式
clamp(max=448) 复现设备侧饱和语义。
```

数学结论：分子分母使用同一平移基准，448 公因子在归一化中严格消去；该平移同时被 LSE 公式（§1.4）完整回收，不产生任何额外误差项。score、块最大值与在线状态 m̂ 均为 FP16（与开源 FP16 平衡 scale 路径一致），ExpSub 输出、分母、alpha 均为 FP32。

#### 1.4 两级累加数值误差界与精度保持机制

在长序列（例如 S=8192 或 S=16384）注意力计算中，归一化权重 `P` 与 `V` 执行矩阵乘累加 `O = P * V`。若直接采用低精度（FP16 或 FP8）进行全程持续累加，随着累加项数增加，浮点截断误差（Representation Swallowing）急剧增大：当局部累加和大到一定程度时，较小的乘积项会被直接下溢截断。

本算子在 950PR 上以下述两级 FP32 结构实现（`attention_block_cube.h` 的 PV Gemm 与 `output_update.h`）：

```text
第一级（N 块内部，Cube L0C）：
  MXFP8 MMA（E4M3 数据 × E8M0 分组 scale）在 L0C 中以 FP32 累加：
  O_block = sum_{k in block} P_e4m3[:, k] * V_mx[k, :]
  N=256（通用路径）；wide 路径按 N=512 分组、两个 N=256 MMAD 片段接力。
  L0C 结果经 CopyL0CToUB（SPLIT_M）以 FP32 原样注入两个 AIV 的 UB，无降精度。

第二级（跨 N 块，AIV UB）：
  O_block_fp32 先写入 PV 部分和槽位（两个槽位轮转，与下一块 MMA 重叠）。
  pv_accum_dtype 决定该加数的精度：
    fp32+fp16：RoundPartialFp16VF 把槽位内容做 FP32→FP16→FP32 往返后再折入；
    fp32+fp32 / fp32：原样折入。
  O_accumulator_fp32 = O_accumulator_fp32 * alpha + O_block_fp32
  其中 alpha = exp(m̂_old - m̂_new) 为在线重缩放系数（FP32）。
  通用路径将相邻两块配对更新（AccumulateFp32PairVF），保留两次独立 FP32 FMA
  舍入点；wide 路径以 Initialize/Accumulate/UpdatePair 三态完成同构语义。
  softmax 分母 den、最终归一化（FastReciprocal 乘法）与 LSE 全部为 FP32。
  `smooth_v` 生效时，归一化后的输出再叠加该 KV head 的 `mean_V`（每通道一次加法）。
```

误差界分析（Error Bound Derivation）：

```python
# 假设浮点机器精度为 eps：
# eps_FP16 = 2^(-11) = 4.88e-4
# eps_FP32 = 2^(-24) = 5.96e-8

# 若以 FP16 直累加全序列（以 S=16384 为例）：
# Bound_FP16 = S * eps_FP16 = 16384 * 4.88e-4 = 8.0（随序列长度线性发散）

# 本实现两级的累加状态均为 FP32：
# 第一级单块内累加 S/N = 64/128 项（N=256/512），误差受 FP32 L0C 累加控制；
# 第二级跨块归约 S/N = 64/128 次 FP32 FMA：
# Bound_L2 = (S / N) * eps_FP32 = 128 * 5.96e-8 = 7.63e-6
# pv_accum_dtype="fp32+fp16" 时，加数在折入前多一次 FP16 舍入，
# 每块引入 <= |O_block| * 2^-11 的相对误差，随块数以 S/N 线性叠加；
# 该项由默认验收配置承担，其余取值不含此项。
# 总误差界由 P/V 的 INT8/FP8 量化噪声主导，累加结构本身不放大误差。
```

参数映射说明：公开参数 `qk_quant_gran`、`pv_accum_dtype` 与 `smooth_v` 经算子属性直通内核，映射到 950PR 的 tile 粒度上，非法取值报错。

- `qk_quant_gran` 决定 Q/K INT8 scale 在多少个连续 token 之间共享（`quant.h` 的 `BroadcastGroupMaxima`）：`per_thread` 为每 4 个 Q token / 16 个 K token 一个 scale，`per_warp` 为 32/64，组边界沿序列维对齐，`SageParams::qg/kg` 承载。开源按 CUDA thread/warp 的寄存器分组，这里按同样的 token 数分组。
- `pv_accum_dtype` 决定跨块归约前，块内 PV 部分和的存储精度：`fp32+fp16` 把 UB 中的部分和舍入到 FP16 再回读折入（`RoundPvSlot` / `RoundPartialFp16VF`），`fp32+fp32` 与 `fp32` 不舍入。950PR 的 MXFP8 MMA 只能在 L0C 内完成块内累加，块间必须经 UB 归约，因此第一级在三种取值下都是 FP32，差异只在第二级的加数精度。
- `smooth_v` 需要 FP32 的运行状态承接补回项，因此仅在 `pv_accum_dtype="fp32"` 下生效；`fp32+fp16` 与 `fp32+fp32` 传 `True` 时在 aclnn 层告警并忽略，内核不读取 V 均值。

#### 1.5 对数归一化与分组查询注意力广播校正数学推导

当调用方指定 `return_lse=True` 时，接口需要返回矩阵 `(Q * K^T) * sm_scale` 每一行的 Log-Sum-Exp（自然对数），数值对齐开源 `core.py` 的返回路径 `lse / 1.44269504 + lse_correction * sm_scale`。

内核与接口按开源契约拆分该表达式：

```python
# 真实数学定义（以自然对数 e 为底，未平滑矩阵 S）：
# LSE_real[i] = ln(sum_{j=1}^{S_K} exp(sm_scale * S[i, j]))

# 代入 S[i, j] = S_smooth[i, j] + C[i] 且 C[i] = dot(Q[i, :], mean_K)：
# LSE_real[i] = sm_scale * dot(Q[i, :], mean_K) + ln(sum_j exp(sm_scale * S_smooth[i, j]))

# 内核持有的在线状态（含 ln(448) 平移，见 §1.3）：
#   m̂_final = max - ln(448)（跨所有块归并后的平移最大值）
#   den_final = 448 * sum_j exp(sm_scale * S_smooth[i, j] - max)
# 二者相加时 448 公因子与 -ln(448) 平移严格相消。
# StoreTileLse 输出 RAW log2 域 LSE（不含平滑校正，单次向量 Muls 换底）：
lse_raw[i] = (m̂_final + ln(den_final)) * log2(e)

# PyTorch 适配层（_ops.py）按开源公式还原并补上平滑校正：
#   lse_return[i] = lse_raw[i] / log2(e) + lse_correction[i] * sm_scale   （smooth_k=True）
#   lse_return[i] = lse_raw[i] / log2(e)                                  （smooth_k=False）
# 其中 lse_correction[i] = dot(Q[i, :], mean_K[kh])，GQA 下按 repeat_interleave 广播。
```

职责划分的依据：校正项 `<q, mean_kh>` 需要 Q 的原始 GM 数据与逐头 mean；`StoreTileLse` 若在内核内计算该项，须在每张量尾块重读 Q 并做 `DP` 长度 ReduceSum（每 query tile 约 64 次向量归约）。将校正移至 PyTorch 层后，内核 epilogue 只剩 `Ln + Cast + Add + Muls` 四个向量操作且不再访问 GM 中的 Q；开源换底与校正公式在接口层与 `core.py` 逐项一致。LSE 以 FP32 写出，形状 `[B, Hq, Sq]`。

该等价性由 CPU 参考实现（`reference.py` 先剥离校正并换算为 raw log2 值，与内核契约同构）与 `tests/numeric_error_test.cpp` 断言 [7]（raw log2 内核值 + 开源校正 = 双精度参考，≤ 1e-5，含 GQA 广播）双向锁定；aclnn 直调方获得的 `lse` 为 raw log2 域值，按上述公式自行还原。

GQA 场景广播规则：
在 Grouped Query Attention 场景下，Query 头数 `H_Q` 大于 Key/Value 头数 `H_K`。每个 KV 头对应一个分组，组内大小为 `group_size = H_Q // H_K`。Key 的通道均值 `mean_K` 原始形状为 `[B, H_K, D]`，Query 头 h 在校正项点积前按

```cpp
kh = h / (p.hq / p.hk);   // attention.h / attention_wide_stream.h 中的任务解码
```

映射到其所属 KV 分组均值，等价于对 `mean_K` 在 Head 维执行 `repeat_interleave(group_size, dim=1)` 的广播语义，且无需物化广播副本。

#### 1.6 无除法快速向量量化与牛顿迭代推导

在细粒度 Token-level 量化中，需要对每个 Token 向量计算其最大绝对值 `amax`，并求解量化步长 `scale = amax / 127.0` 及其倒数 `inv_scale = 1.0 / scale` 以供量化映射使用。

在 Ascend AIV Vector 硬件架构中，原生硬件向量除法指令 `vdiv` 属于高延迟指令（约 32 个时钟周期），且阻塞后续向量流水线；而向量乘法 `vmul` 与减法为单周期全流水执行单元。

本算子在 `common.h` 中实现基于二阶牛顿-拉夫逊（Newton-Raphson）逼近的无除法快速倒数 `FastReciprocal`，用于量化 scale 求逆（`quant.h`）与最终输出归一化（`output_update.h` 的 `NormalizeVF` / `FinalizeWideFp32VF`）：

```cpp
__simd_callee__ inline void FastReciprocal(AscendC::MicroAPI::RegTensor<float> &dst,
                                           AscendC::MicroAPI::RegTensor<float> &src,
                                           AscendC::MicroAPI::MaskReg &mask)
{
    // 初始估算: 利用 IEEE-754 浮点数比特位整型重解释减法，生成初始倒数估值 y0
    // 魔数 0x7EEF370B 的初始相对误差随 binade 摆动，最差约 2^-4（非常数 2^-7）
    RegTensor<uint32_t> magic, ix, iy;
    RegTensor<float> y0, term, two;
    Duplicate(two, 2.0f);
    Duplicate(magic, uint32_t(0x7EEF370B));
    ix = (RegTensor<uint32_t> &)src;
    Sub(iy, magic, ix, mask);
    y0 = (RegTensor<float> &)iy;

    // 第一次牛顿迭代: y1 = y0 * (2.0f - src * y0)
    Mul(term, src, y0, mask);
    Sub(term, two, term, mask);
    Mul(dst, y0, term, mask);

    // 第二次牛顿迭代: dst = y1 * (2.0f - src * y1)，达到 IEEE-754 FP32 机器精度
    Mul(term, src, dst, mask);
    Sub(term, two, term, mask);
    Mul(dst, dst, term, mask);
}
```

收敛性与数值误差界（`tests/numeric_error_test.cpp` 断言 [1]）：
牛顿迭代在精确算术下满足二次收敛 $e_{n+1} = e_n^2$，但魔数初值的相对误差并非一致小量——它随操作数所在 binade 摆动，最差约 $2^{-4}$。
- 初始相对误差：$|e_0| \lesssim 2^{-4}$（随 binade 摆动，均值约 2^-6~2^-7）。
- 两次迭代后：$|e_2| \approx e_0^4$，断言界取 2^-14。
- 该残差不是 IEEE-754 机器精度（2^-23）级别的位精确结果，但小于其下游各环节的舍入误差：FP16 平衡 scale（2^-11）、E4M3 量化噪声（~2^-5）、输出 FP16 舍入（2^-11）。

执行流水上完全由 1 次魔数整型减法、4 次向量乘法与 2 次向量减法构成，消除 32 周期向量除法指令依赖。

V 的 MXFP8 scale 生成同样无除法（`quant_v_mx.h`）：`scale = 2^ceil(log2(amax/448))` 通过对 `amax * (1/448)` 的位模式加舍入偏置后掩码取阶码域直接构造 2 的整数次幂，其倒数由 `2^127 阶码基准 - 舍入后位模式` 的整型减法拼出，全零分组以 `2^-100` 兜底，避免除零。

### 2. 支持数据类型与精度标准

| 参数项 | 支持数据类型 | 内部计算数据类型 | 物理约束 |
| --- | --- | --- | --- |
| query | float16, bfloat16 | INT8 对称量化（scale 按 `qk_quant_gran` 在连续 token 组内共享） | 末维连续，即 stride(-1) == 1 |
| key | float16, bfloat16 | INT8 对称量化（同上，smooth_k 开启时先减 mean_K） | 与 query 保持相同数据类型与 device |
| value | float16, bfloat16 | MXFP8：E4M3 数据 + E8M0 分组 scale；smooth_v 开启时量化前先减 mean_V | 与 key 保持相同数据类型与 device |
| output | float16, bfloat16 | FP32 两级累加后一次性饱和舍回 | 与 query 保持相同数据类型与形状 |
| softmax_lse | float32 | float32 | 形状为 [B, Hq, Sq]；内核输出为 raw log2 域（未经平滑校正），PyTorch 层按开源契约还原为自然对数并叠加校正 |

### 3. 支持形状与张量布局约束

- 维度范围：输入必须为 4 维张量。
- 张量布局：
  - HND 布局：`[B, H, S, D]`，其中 Query 为 `[B, Hq, Sq, D]`，KV 为 `[B, Hk, Sk, D]`。
  - NHD 布局：`[B, S, H, D]`，其中 Query 为 `[B, Sq, Hq, D]`，KV 为 `[B, Sk, Hk, D]`。
- 特征维度 D 约束：
  - 支持范围为 `(0, 128]`。
  - 当 D <= 64 时，内部自动 Padding 至 64。
  - 当 64 < D <= 128 时，内部自动 Padding 至 128。
  - 当 D > 128 时，Host 侧拦截并抛出异常。
  - 输出张量末维自动裁剪还原至原始未 Padding 的 `D`。
- GQA 分组注意力：Query 头数 `Hq` 必须能被 Key 头数 `Hk` 整除。
- 因果模式约束：当 `is_causal=True` 时，必须满足 `Sq == Sk`。

---

## 设计取舍

以下方案完成过推导与评估，但未进入交付通路，记录于此作为设计依据：

- **以 2 为底的归一化尺度折叠**：把换底常数 `log2(e)` 在 Host 侧折叠进 `sm_scale`，内核改用原生 `Exp2`。
  Atlas 950PR 的 AIV `ExpSub` 本身就是单条自然指数指令，换底反而多一次向量乘，因此未启用（§1.3 沿用自然指数）。

## 算子实现

### 1. 硬件协同与架构全景

#### 1.1 硬件特性与计算拓扑

Atlas 950PR（CANN 9.2.0-beta.1，微架构 dav_3510）采用非对称的 1:2 MIX 异构计算拓扑架构：

- 芯片物理核心配置：包含 24 个物理 AIC（AI Core，主控 Cube 矩阵计算）与 48 个物理 AIV（AI Vector，主控向量处理）。
- 配对比例：每个 AIC 与 2 个 AIV 物理绑定，组成一个异构混合计算单元（Compute Subsystem Block）。
- 主频与存储：核心标称运行主频 1.6 GHz；片上搭载高带宽 HBM3e，物理总带宽达 3.2 TB/s；INT8/FP8 Cube 吞吐约 1 PFLOPS 量级。
- 片上存储层级（单核，取自 CATLASS `Catlass::Arch::Ascend950` 常量）：
  - 单 AIV 可用统一缓冲区（Unified Buffer, UB）为 248 KB（本算子静态排布全部约束在 248 KB 内）。
  - AIC 包含私有 L1 缓冲区（512KB）与 L0A/L0B（各 64KB）/L0C（256KB）矩阵暂存器；本算子将 L0C 划分为 QK/PV 两个 128KB 分区，L0A 同样以 32KB 为界划分（PV 的 MX P scale 常驻高段）。
  - 全局共享 L2 Cache 提供高速 Cross-Core 数据同步。

#### 1.2 框架与库分工体系

为了兼顾开发效率、硬件指令控制力与矩阵乘算力，算子严格按功能边界划分 AscendC 与 CATLASS 的分工职责：

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        SageAttention2 算子架构分工                      │
├──────────────────────────────────┬─────────────────────────────────────┤
│      CATLASS 矩阵计算子系统      │        AscendC 向量流水线子系统     │
├──────────────────────────────────┼─────────────────────────────────────┤
│ 1. Cube INT8 GEMM-1 (S = Q * K^T)│ 1. Key 序列均值归约与广播 (mean_K)   │
│    TileMmadTla / PackedTileCopy  │ 2. 细粒度 INT8 离群平滑量化流水线   │
│ 2. Cube MXFP8 GEMM-2 (O = P * V) │ 3. Value MXFP8 (E4M3+E8M0) 量化     │
│    MakeMxScaleLayout / MX MMA    │ 4. 在线 Softmax（ExpSub 自然指数）   │
│ 3. PackedTileCopyTla 模板搬运    │ 5. 两级累加 FP32 动态重缩放与归约   │
│ 4. L1/L0A/L0B 内存排布与双缓冲   │ 6. LSE 自然对数校正与 GQA 广播       │
│ 5. MMAD 脉动阵列计算微原语       │ 7. 片上 HardEvent 硬件流水同步控制  │
│ 6. CopyL0CToUB (SPLIT_M) 结果搬出│    （含 MODE4 跨核计数通知）         │
└──────────────────────────────────┴─────────────────────────────────────┘
```

#### 1.3 单核函数三阶段融合执行流水线

算子以单次 MIX AIC:AIV=1:2 kernel launch 完成 Stats → Quant → Attention 三阶段（`sageattention2.cpp`），中间结果仅落入算子 workspace（GM），无独立预处理 kernel、无全量 attention 矩阵物化：

```text
阶段一 Stats（AIV）：
  1. 每个 mean 槽位由独占 AIV 整体覆写，无需预清零或跨核屏障。
  2. 沿序列分批累加 Key（FP32），每个 KV head 由单个 AIV 独占写回，无跨核原子加。
  3. smooth_v 生效时以同一份代码（`Stats<T, true>`，切换 V 的 stride 表）统计 V 的通道均值。
  4. SyncAll<true>() 保证均值就绪。

阶段二 Quant（AIV）：
  1. Q/K INT8 量化：K 先减 mean_K（smooth_k 开启时），scale = amax/127 并在连续
     token 组内共享（`qk_quant_gran`，§1.3），FP16 平衡 scale 预融合 sm_scale（§1.2），
     量化结果与 scale 写入 workspace。
  2. V MXFP8 量化：每 32 token 每 channel 独立 E8M0 scale，尾组补零（§1.6）；
     smooth_v 开启时在补零之后、取 amax 之前对有效行减去 mean_V，因此 padding 行
     保持为 0，与参考实现"先居中后补零"的顺序一致。
  3. SyncAll<false>() 同步全部 AIC/AIV，保证量化数据全局可见后进入 Attention。

阶段三 Attention（AIC + AIV 异步流水）：
  1. [AIC Cube] CATLASS TileMmadTla 执行 INT8 GEMM-1，L0C INT32 累加，
     Fixpipe DEQF16 直接产出 FP16 得分注入 AIV UB（§1.2）。
  2. [AIV Vector] 在线 Softmax：FP16 得分乘平衡 scale，ln(448) 平移最大值，
     ExpSub 自然指数，P 饱和转 E4M3 经 UB→L1 送入 Cube L0A（§1.3）。
  3. [AIC Cube] MXFP8 GEMM-2：E8M0 V scale 由硬件 MMA 单元应用（§1.4），
     FP32 L0C 结果 SPLIT_M 注入两个 AIV UB。
  4. [AIV Vector] 两级累加更新 O，可选 LSE 校正，最终归一化（smooth_v 生效时叠加
     mean_V）并按输入 dtype 写出。
```

读缓存策略（`quant.h`）：连续对齐输入的 Q/V 原始读取 bypass L2（`CACHE_MODE_DISABLE`），K 与量化输出保留默认缓存策略，减少一次性读取对 Attention 阶段工作集的挤占。Host 使用 `SetScheduleMode(1)` batch 调度，保证 SyncAll 所需的 MIX 全网格同时驻留。

### 2. 调度与分块规划

#### 2.1 任务空间与分块循环波次处理

Attention 阶段的任务粒度不是整个 head，而是 128 行的 Query tile（M=128）：任务总数为

```text
total = B * Hq * ceil(Sq / 128)
```

各核心以循环步长 `cores`（AIC 数，AIV 以 GetBlockIdx()/2 对齐到同一 AIC 组）领取任务（`attention.h` 主循环）。以 B=1、Hq=32、Sq=4096 为例：

```text
total = 1 * 32 * 32 = 1024 个任务，24 个 AIC 核心
波次数 = ceil(1024 / 24) = 43 波（最后一波 1024 - 42*24 = 16 个任务）
```

任务粒度为 128 行 Query tile，波次末尾的不完整波次只影响一 wave 内的核心数；按 head 静态分配时 `B*Hq=32 < 2*24` 会出现核心闲置。`DieTaskEnd`（`common.h`）以虚拟循环任务空间精确计算每核的终止边界，正确处理不满一波的尾波。

预处理阶段（Stats/Quant）同样按 `2 * cores` 个 AIV 块循环分配 token 批次，空闲核也参与 SyncAll 屏障。

#### 2.2 物理内存突发对齐与工作区空间计算

在 Ascend 950 微架构中，MTE2 与 MTE3 数据搬运引擎的底层物理总线采用 512 字节 Burst 突发传输模式。非 512 字节对齐的地址访问会引发硬件总线拆包与跨周期拼接，损失物理搬运带宽。

在 Host 侧 `sage_attention2_tiling.cpp` 中，内核将用户工作区静态划分为 8 大功能张量段，各分段首地址偏移均相对于 `GetUserWorkspace(workspace)`，并严格施加 512 字节向上对齐（附带 64 位溢出检查）：

```cpp
// 512 字节向上物理突发对齐分配闭包（含上溢保护）
uint64_t bytes = 0;
constexpr uint64_t WORKSPACE_ALIGN = 512;
auto allocate = [&bytes](uint64_t &offset, std::initializer_list<uint64_t> factors) {
    uint64_t size = 1;
    for (auto factor : factors) {
        if (factor && size > std::numeric_limits<uint64_t>::max() / factor)
            return false;
        size *= factor;
    }
    size = (size + (WORKSPACE_ALIGN - 1)) & ~(WORKSPACE_ALIGN - 1);
    offset = bytes;
    bytes += size;
    return true;
};

// 8 大工作区张量段分配（保存在 SageAttention2TilingData 中传递给 Kernel）：
// WIDE 模式下 Query 启用片上流式即时量化，直通 L1/UB，q8 与 qScale 零显存占用（节省数十至数百兆显存）
const uint64_t qBatch = wide ? 0ULL : p.b;
allocate(t.mean,     {p.b, p.hk, p.dp, sizeof(float)});       // Key 序列通道均值向量
allocate(t.vMean,    {p.b, p.hk, p.dp, sizeof(float)});       // Value 序列通道均值向量 (smoothV)
allocate(t.vMxScale, {p.b, p.hk, p.kp / 32, p.dp});          // Value MXFP8 32-token 分组 E8M0 Scale
allocate(t.q8,       {qBatch, p.hq, p.qp, p.dp});             // INT8 平滑量化 Query 张量 (WIDE 下为 0)
allocate(t.k8,       {p.b, p.hk, p.kp, p.dp});               // INT8 平滑量化 Key 张量
allocate(t.v8,       {p.b, p.hk, p.kp, p.dp});               // MXFP8 (E4M3) 量化 Value 张量
allocate(t.qScale,   {qBatch, p.hq, p.qp, sizeof(uint16_t)}); // Query Token FP16 平衡 scale (WIDE 下为 0)
allocate(t.kScale,   {p.b, p.hk, p.kp, sizeof(uint16_t)});   // Key Token FP16 平衡 scale
// 其中 qp = ceil(Sq/128)*128、kp = ceil(Sk/512)*512 为补齐后的序列长度。

// 叠加 CANN 平台运行时预留系统 Workspace（通过 PlatformAscendC 动态查询，兜底 16MB）：
uint64_t reserved = platform_ascendc::PlatformAscendC(c->GetPlatformInfo()).GetLibApiWorkSpaceSize();
ws[0] = reserved + bytes;
```

Kernel 侧入口通过 `GetUserWorkspace(workspace)` 越过系统前置保留区，以 `work + t.xxx` 取得各中间张量的 512 字节对齐物理起始指针。workspace 每次调用独立分配，无跨调用全局状态。

#### 2.2.1 WIDE 模式下的 Query 片上流式量化数据流

非 WIDE 路径保持"预处理批量量化 → GM 中转 → Attention 读取"的两段式结构；WIDE 路径把 Query 整段搬进 Attention 主循环，Query 不再落 HBM：

```text
AIV(subBlockIdx_)  每个 query tile 的 step 0（SelectStreamQuery）
  MTE2  GM Q[qToken*DP] ──64×DP──▶ SCORE_UB 起步的临时区（借用已退休的偶环 stripe 0/1）
  V     Cast 转 FP32 → QuantRowsVF：per-token 对称 INT8（粒度 p.qg）
        ├─ INT8 量化值 → 临时区 out
        └─ FP16 scale → qsu_（与离线路径的 GM qScale 逐位等价）
  MTE3  out ──zN 平面──▶ L1 A 槽（l1Buf[0]）：每 32 列一次集中搬运，
        两个 AIV 各写平面内 64 行（subBlockIdx_ 选前/后 2 KiB），互不重叠
  MODE4 CrossCoreSetFlag<PIPE_MTE3>(Q_READY)

AIC  GemmQkSplit（仅该 query 的 part == 0 触发 loadA）
  WaitBothAiv<PIPE_MTE1>(Q_READY) → CopyL1ToL0A 取 64 行子块 → MMAD
```

三条约束是本路径的正确性前提，改动时必须同时保持：

1. **L1 A 槽按 `layout::zN` 排布**。Cube 侧 `PackedTileCopyTla` 的 `LayoutTagL1A` 由 `RowMajor` 映射为 `zN`，int8 且 M=128 时偏移为 `addr(r,c) = (c/32)*4096 + r*32 + (c%32)`，故 AIV 必须逐列平面写出，不能平铺拷贝。UB→L1 的 `DataCopyParams` 以 32 字节为单位，且第 3/4 项是**不含 burst 长度的间隔**（`srcGap`/`dstGap`），与既有 FP8 P 写出使用同一套约定。
2. **mode-4 的 AIV→AIC 事件由硬件按源 sub-block 区分**。AIV 只写裸 flag id，AIC 用 `WaitBothAiv` 同时等 `f` 与 `f+16`（与 `SyncAll` 内部的 `SYNC_AIV_FLAG` 协议一致）；`AIV_FLAG_OFFSET` 仅用于 AIC→AIV 方向。
3. **临时区与偶环得分条带重叠**。step 0 之前必须排空上一组 `SoftmaxWide` 遗留的 MTE3 条带读与 V 读，因此填入临时区前依次执行 `Sync<MTE3_MTE2>` 与 `Sync<V_MTE2>`，且临时区上界必须严格低于 stripe 2 的起始地址。

Query 与 K/V 预处理解耦后，`SyncAll<false>()` 只需等待 K/V；Query 的就绪完全由上述局部 `Q_READY` 握手承担，跨 Die 硬屏障的等待面积随之减半。

#### 2.3 模板参数特化与分块路由规划

为了消除运行时数据类型分支与核心循环跳转，同时避免因张量维度穷举导致二进制膨胀，本算子遵循官方 AscendC 模板机制（`sage_attention2_template_tiling_key.h`），在编译期仅对主数据类型与 WIDE 超宽双流水模式做模板特化：

```cpp
// 遵循 ge::DataType 标准定义
#define SAGE_TPL_FP16 1
#define SAGE_TPL_BF16 27

// 声明模板特化维度：输入数据类型 (FP16/BF16) 与 WIDE 双发射模式 (0/1)
ASCENDC_TPL_ARGS_DECL(SageAttention2,
                      ASCENDC_TPL_DTYPE_DECL(INPUT_DTYPE, SAGE_TPL_FP16, SAGE_TPL_BF16),
                      ASCENDC_TPL_BOOL_DECL(WIDE, 0, 1));
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(INPUT_DTYPE, SAGE_TPL_FP16, SAGE_TPL_BF16),
                                     ASCENDC_TPL_BOOL_SEL(WIDE, 0, 1)));
```

Host 侧 Tiling 路由判定逻辑（dtype 占 tiling key 低 8 位，wide 占第 8 位，四种组合为 1/27/257/283）：

```cpp
// 判定是否激活 WIDE 超宽双流水内核（长序列分支）。
// 流式 Q 以 DP 为行距直读原始 Q，故要求补齐后的特征维度等于真实特征维度；
// qp == sq 由 sq % 128 == 0 保证。
const bool wide = p.dp == 128 && p.d == p.dp && p.sq % 128 == 0 && p.sk % 512 == 0 && p.sk >= 4096;
c->SetTilingKey(GET_TPL_TILING_KEY(static_cast<uint32_t>(dt), wide));

// 强制设置调度模式为 1，确保 1:2 MIX 全网格核心驻留调度
c->SetScheduleMode(1);
```

而在算子 Kernel 侧，以极简签名接收特化：

```cpp
template <int INPUT_DTYPE, bool WIDE>
__global__ __aicore__ void sage_attention2(GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                           GM_ADDR out, GM_ADDR lse,
                                           GM_ADDR workspace, GM_ADDR tiling);
```

其余动态控制参数（如 `causal` 因果掩码、`smoothK` 平滑开关、`returnLse`、张量物理步长等）全部收拢于 `SageAttention2TilingData.params` 运行时结构体中分发，兼顾执行性能与二进制编译打包体积。若 tiling 平台信息缺失（独立 aclnn 直调场景），Host 依次回退 CompileInfo 与 `aclrtGetDeviceInfo` 运行时查询核数，绝不猜测核数。

### 3. 微架构优化

#### 3.1 片上 UB 与 L1 内存编排

Atlas 950 单 AIV 具备 248KB 可用 UB、单 AIC 具备 512KB L1。为避免动态内存管理（如 TPipe 维护引起的运行时开销与跳转分支），本算子在 `attention_common.h` 中采用全静态物理缓冲区排布（`FixedResource`）与多槽位环形流水编排：

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      AIV Unified Buffer (UB) 静态内存规划 (≤ 248 KB)                    │
├─────────────┬──────────┬───────────────────────────────────────────────────────────────┤
│ 缓冲区标识  │ 内存容量 │ 用途定义与硬件流水机制                                        │
├─────────────┼──────────┼───────────────────────────────────────────────────────────────┤
│ SCORE_UB    │ 96 KB    │ 3 个槽位 × (HALF=64) × (N=256) × 2B (FP16 QK 得分暂存区)      │
├─────────────┼──────────┼───────────────────────────────────────────────────────────────┤
│ PV_UB       │ 64 KB    │ 2 个槽位 × (HALF=64) × (D=128) × 4B (FP32 PV 部分和接收区)    │
├─────────────┼──────────┼───────────────────────────────────────────────────────────────┤
│ ACCUM_UB    │ 32 KB    │ 1 个槽位 × (HALF=64) × (D=128) × 4B (FP32 动态重缩放累加区)   │
├─────────────┼──────────┼───────────────────────────────────────────────────────────────┤
│ P_UB        │ 49.5 KB  │ 3 个槽位 × 16.5 KB (FP8 E4M3 归一化概率矩阵 P，含 NZ 步长填充) │
├─────────────┼──────────┼───────────────────────────────────────────────────────────────┤
│ 状态向量区  │ ~6 KB    │ MU_UB (均值)、KS_UB/QS_UB (Scale)、MAX_UB、DEN_UB、ALPHA_UB   │
│             │          │ （ALPHA 4 槽位保留相邻两组的 alpha 直至配对 O 更新完成）        │
├─────────────┼──────────┼───────────────────────────────────────────────────────────────┤
│ LSE 复用区  │ 共享 P0  │ 尾部 LSE 归一化计算在输出写出后直接复用已退休的 P0 槽位      │
└─────────────┴──────────┴───────────────────────────────────────────────────────────────┘
```

片上 L1 内存规划（AIC 512KB L1）：
- `P_L1 = 64 KB`：接收 AIV 写入的 FP8 概率矩阵 P；
- `B_L1 = 160 KB`：多槽位双缓冲存放 Key 与 Value 矩阵切块（`PIPE_SLOTS = 3`）；
- `MX_V_L1 = 320 KB`：V 的 E8M0 scale 缓存区起点，按 KV head 缓存，可用容量 176 KiB；超出容量时按 PV tile 逐次搬运，算法不变；
- `MX_P_L1 = 496 KB`：P 的单位 E8M0 scale（0x7F，即 2^0）经 `load_cbuf_to_ca_mx` 一次性常驻 L0A 高段；
- 编译期静态断言保障：`static_assert(B_L1 + 2 * N * 128 <= 512 * 1024)` 与 `static_assert(UB_END <= 248 * 1024)`。

#### 3.2 非对称分块策略与向量满通道饱和

在 Atlas 950PR 的 1:2 MIX 异构架构中，分块大小决定算力占用：
- Query 轴切块 $M = 128$：单个 Cube 核心处理 128 行，完成矩阵乘后将结果拆分为两个高度为 $\text{HALF} = 64$ 行的子矩阵，通过 `CopyL0CToUB(SPLIT_M)` 片上直通分发给绑定的 2 个 AIV 核心并发执行 Online Softmax。
- SIMD 通道宽度：Ascend 950 AIV 单指令处理宽度为 256 字节（64 个 FP32 元素）。$\text{HALF} = 64$ 使每行的 Max、Sum 规约与缩放正好占满 64 个通道。
- Key/Value 轴切块 $N = 256$：$N = 256$ 处于 Cube MMAD 脉动阵列的能效区间，并通过 3 槽位前瞻流水（`PIPE_SLOTS = 3`, `LOOKAHEAD = 2`）在执行第 $i$ 步 PV 时发射第 $i+2$ 步 QK，使计算与搬运重叠。

#### 3.3 跨核事件通知与双总线协同计数机制

Cube 与 Vector 之间通过片上硬件事件总线协同，不使用全局轮询或软件锁：

```cpp
// MODE4 跨核计数值通知定义（attention_common.h）：
constexpr uint32_t QK_READY = 0;       // Cube 通知双 AIV：QK 矩阵乘得分已就绪
constexpr uint32_t P_READY = 1;        // AIV 通知 Cube：Softmax 概率矩阵 P 已进入 L1
constexpr uint32_t PV_READY = 2;       // Cube 通知 AIV：PV 矩阵乘结果已注入 UB
constexpr uint32_t PV_RELEASE = 3;     // AIV 通知 Cube：PV UB 槽位消费完毕可复用
constexpr uint32_t AIV_FLAG_OFFSET = 16;// 第二个 AIV 核心事件偏移
```

同步协作流程：
1. Cube 算完 QK 后，调用 `Notify(QK_READY)`，向绑定的 2 个 AIV 核心（分别触发 flag 0 与 flag 16）广播就绪；
2. AIV 消费并计算完成 Softmax 后，各自调用 `CrossCoreSetFlag<4, PIPE_MTE3>(P_READY)` 通知 Cube；
3. Cube 核心通过 `WaitBothAiv<PIPE_MTE1>(P_READY)`（flag 与 flag+16 双门限）确认后启动 PV 计算；
4. AIV 完成 O 更新后以 `PV_RELEASE` 归还 PV UB 槽位；槽位环深（PV 2 槽、P/score 3 槽、alpha 4 槽）与配对更新语义由静态断言锁定，各槽位循环交替使用。
5. 预处理两道 SyncAll 屏障使用的旗标空间（11..14）与 MODE4 旗标（0..3 + 16 偏移）互不重叠。

#### 3.4 因果下三角跳步调度与动态边界收缩

当指定 `is_causal=True` 时，自注意力矩阵的上三角区域权重被严格掩码遮蔽。在内核主循环中，根据当前 Query 块的起始位置 `qstart = tile * M` 动态收紧 Key 的有效遍历上界：

```cpp
// 动态收紧内层 KV 遍历步数：
uint32_t limit = p.causal ? Min(p.sk, qstart + M) : p.sk;
uint32_t count = (limit + N - 1) / N; // 仅计算包含下三角有效元素的 Key 块

// 在超宽双流水模式下 (DecodeWideQuery)：
const uint32_t limit = CAUSAL ? Min(p.sk, info.qstart + M) : p.sk;
info.count = (limit + 511) / 512;
```

外层调度因此不发射上三角全空块，减掉相应的 Cube 矩阵乘与数据加载；包含对角线的边界块内部由 AIV 施加精确的逐行因果掩码（通用路径按 `pos - at + 1` 列计数掩蔽，wide 路径按 NZ lane 与 query 行号的标量比较掩蔽）。

#### 3.5 接口与编译期现代工程化设计

- Gemm 调用接口结构化：Cube 侧统一入口 `Gemm<T, BLayout, LocalA, BOffset, Emit, Initialize, PreloadedB>` 将全部行为差异上提为模板参数（数据类型与 B 布局、A 是否已在 L0A、B 的 L1 固定偏移、是否发射 Fixpipe、是否初始化 L0C、B 是否已由 MTE2 预取），运行期仅保留 `a/b/k/n/ubOffset/parity/loadA/mxScale/mxCache/mxK/mxStart/vParity` 等实际变化的数据形参，缺省值收敛了 QK 路径不涉及的 MX 与 V 双缓冲参数；槽位事件 `PV_EVENT/QK_EVENT` 与 L0/L0C 偏移由 `LocalA` 在编译期推导，QK 与 PV 互不干扰，消除了运行期分支与参数列复制。
- 编译期对齐工具：`params.h` 提供 `constexpr RoundUpTo(v, align)`（32/64 位重载），tiling 的 `qp/kp` 序列补齐与 workspace 512B 突发对齐均改用该编译期函数表达，消除手写位运算魔数。
- 编译期枚举解析与校验：`qk_quant_gran`/`pv_accum_dtype` 的字符串解析（`StrEq` + `ParseQuantGran`/`ParsePvAccum`）与取值校验（`IsValidQuantGran`/`IsValidPvAccum`）全部为 `constexpr`，与量化粒度常量（`QTokensPerScale`/`KTokensPerScale`/`DividesTile` 及"粒度必须整除 64 行 tile"的 `static_assert`）同住在 Host/Kernel 共享的 `params.h` 中，非法取值在编译期可证、运行期必报错。
- Tiling 载荷的编译期契约：`SageParams` 由 Host memcpy 原样送到设备，故 `static_assert(is_trivially_copyable_v && is_standard_layout_v)` 拦住任何使其失去 POD 性质的成员改动，`static_assert(sizeof(SageParams) % 8 == 0)` 拦住尾部填充；累加格式成员直接声明为 `SagePvAccum` 枚举（而非 `uint32_t` + 散落强转），内核比较不再依赖隐式的整型↔枚举约定。`RoundUpTo` 的位运算只对 2 的幂对齐成立，由 `constexpr IsPow2` + `static_assert` 固化，避免返回"看似合理的错误补齐值"。
- 设备无关状态的线程化：host tiling 中缓存核心数/平台信息的两处 `static` 改为 `thread_local`——CANN 可在多线程中并发构图，普通函数静态局部变量会让一个线程的探测结果被另一个线程的异构平台静默复用。
- 存储格式单一裁决点：输入/输出布局只在 `sage_attention2_def.cpp` 以 `.Format({ge::FORMAT_ND, ge::FORMAT_ND})` 声明一次，host tiling 不重复做运行期格式比对，避免两处判据随框架演进分叉。
- 死代码清除：宽块收尾 `FinalizeWideFp32VF` 只有 DP=128 实例可达（tile 高度固定 64 行、两半拼 128 列），非 128 分支连同其整段向量序列删除并以 `static_assert(DP == 128)` 固化前提；`scale.h` 中已被量化期内联乘法取代的 `FixedScaleFactorsVF` 与 `PackFixedScaleBlock` 的 UB 指针重载删除，`QuantRowsVF`/`QuantRowsImpl` 随之收回 `FactorT` 模板形参（全部调用点传 Host 预计算的标量 `qFactor`/`kFactor`）；`SageParams::smScale` 字段删除——内核只消费其预融合 FP16 形式，保留它等于在每份 tiling 载荷里放一个设备指令永不读取的成员。设计文档与内核符号保持逐项可 grep 对应。
- 分支现代化与去重：wide 流水线的运行期常数分支改写为 `if constexpr (PHASE == 0)`（编译期裁剪）；`stats.h` 以泛型 `LoadLaneF32` 统一 16 位装载转换的三处重复分支；算子输入由类型 trait `is_supported_input_type`（FP16/BF16 特化）在融合入口 `static_assert` 限定，不再保留按 `float` 实例化的死分支；LSE 尾声复用 P0 槽的暂存区由静态断言约束（`2*8*128*4+1024 <= P_SLOT_BYTES`）。
- 分层边界：Python 侧公开接缝为 `_ops` 导出的 `sageattn`、`sageattn_qk_int8_pv_fp8_asc` 及其别名 `run`（`run = sageattn_qk_int8_pv_fp8_asc`），由 `__init__` 与 `core` 两级同时再导出；随仓验收测试 `python/test_python_api.py` 只依赖这组公开符号，不存在独立的兼容层模块。`_ops` 与 csrc 各自独立校验形成纵深防御，接口语义与内核模板参数（dtype × wide）一一对应。

#### 3.6 张量布局规整与直通设计

本算子的布局处理分为两层：

1. HND（BNSD 同构）连续输入：aclnn 层直接透传原张量，Kernel 依据 Host 侧按规范形状推得的物理步长（`p.qs/p.ks/p.vs`，元素单位）直接换算 GM 地址：

```cpp
// tiling.cpp 按规范 HND 连续布局计算步长（d 为未 padding 的原始 head_dim）：
p.qs[2] = p.d;  p.qs[1] = uint64_t(p.sq) * p.d;  p.qs[0] = uint64_t(p.hq) * p.qs[1];
//（p.ks/p.vs 同构）
// Kernel 寻址：uint64_t off = b * strides[0] + h * strides[1] + s * strides[2];
```

无隐式 `contiguous()` 拷贝，HND 输入直通。

2. NHD（BSND 同构）输入：aclnn 层以 `CreateView` 交换 shape/strides 的 1、2 维构造逻辑 HND 视图（零拷贝）；若该视图不满足 Kernel 的连续性要求，再由 `l0op::Contiguous` 做一次框架级连续化拷贝。即 NHD 输入功能完全支持，但需付出一次 Q/K/V 的 D2D 拷贝开销；输出侧经 `ViewCopy` 写回用户布局。末维 `stride(-1) == 1` 是两层共同的前置约束，在 Python 与 aclnn 层均强制校验。

---

#### 3.7 向量多发射指令交织与流水线冒险消除

在昇腾矢量计算核心（AIV）的微架构中，连续针对同一物理寄存器的写后读（RAW）存在 2 至 3 个周期的流水线停顿气泡。在密集归一化和累加更新循环中，这种气泡会造成高达 20% 以上的矢量单元有效吞吐浪费。

本实现在 `softmax.h`、`softmax_wide.h` 以及 `output_update.h`（涵盖 `ScaleAndCastTileVF`、`AccumulateFp32VF`、`AccumulateFp32PairVF`、`NormalizeVF`）中采用 2 行与 4 行双重指令交织：
- 在得分缩放与极大值规约阶段，交织进行两行的 `Mul(coeff0, qh, kh0)` 与 `Mul(coeff1, qh, kh1)`，并将随后的特征乘法与掩码处理交叉发射，使每个操作数生成与消费之间均插入至少 3 条独立无关指令；
- 在输出累加与归一化阶段，按照 4 行一组展开装载（LoadAlign）、乘法更新（Mul / MulDstAdd）、类型转换（Cast）与结果回写（StoreAlign），消除矢量流水线停顿。

#### 3.8 行和两级平衡树规约优化

在 `SoftmaxHalfVF` 核心函数中，针对 256 列得分切块计算行和分母时，原先采用连续单向累加链：
`Add(sum, p0, p1); Add(sum, sum, p2); Add(sum, sum, p3);`
该结构形成了深度为 3 的串行数据依赖链路，限制了指令发射窗口。

现已重构为两级平衡树状规约：
`Add(s01, p0, p1); Add(s23, p2, p3); Add(rowSum, s01, s23);`
第一级并行计算两路两两子和，并将指令与两路 FP8 饱和转换（`Cast<float8_e4m3_t>`）穿插执行；第二级完成合并。数据依赖关键路径深度从 3 级降为 2 级，提高 AIV 双发射流水的占用。

#### 3.9 惰性归一化动态跳步重缩放

在线 Softmax 的全局累加更新形式为：
`O = O * alpha + PV`，其中 `alpha = exp(m_old - m_new)`。

当当前切块的局部最大值未超过历史极大值时，`m_new == m_old`，此时重缩放系数 `alpha` 精确等于 `1.0f`。在具有注意力汇聚（Attention Sink）或长序列因果推断的场景中，该现象出现得较为频繁。

本算子在 `output_update.h` 中实现了特化的惰性重缩放：
- 当判定 `alpha == 1.0f` 时，直接旁路片上统一缓冲区（UB）中的 alpha 装载与标量广播，将原先的 `MulDstAdd` 矢量乘加指令简化为单周期原位向量 `Add`：
  `O = O + PV`；
- 该优化省去累加阶段的 alpha 装载与乘法，减少对应的矢量内存带宽与流水延迟。

#### 3.10 片上内存屏障精简与发射窗口释放

对内核各模块中的片上局部内存屏障（`LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>()`）进行了细致的形式化数据流审计：
- 剔除了 `scale.h` 与 `output_update.h` 中针对寄存器直通中间量的冗余内存屏障；
- 仅保留跨硬件缓冲区写入与读取之间的必要内存同步，最大程度扩大了编译器指令重排与硬件发射窗口。

#### 3.11 矩阵乘与流水修复并发重叠

在 `attention_cube_split_qk.h` 中，针对 QK 矩阵乘分裂计算路径，重叠矩阵乘算术流水（`PIPE_M`）与格式转换流水（`PIPE_FIX`）：
- 在 Cube 计算当前片段的同时，Fixpipe 异步执行前一片段的结果搬运与反量化；
- 在满足同步约束的前提下提前发出 `Notify(QK_READY)` 事件，使 AIV 核心能够提前 10-15 个周期被唤醒并投入 Softmax 预取，实现 Cube 与 Vector 的无缝交织。

#### 3.12 循环不变量提升与量化基址外提

在 `quant_v_mx.h` 与 `stats.h` 的核心量化循环中：
- 严格遵循毕昇编译器的向量地址生成（VAG）合规约束，将循环内部的基地址动态偏移计算与常量装载外提到循环外部；
- 避免在内层热点循环中出现动态 UB 指针寻址，消除了编译器后端因地址生成不合规导致的重试与开销。

#### 3.13 归一化向量双行展开微架构设计

在昇腾 AIV 矢量计算核心微架构中，向量运算流水线具备固定的指令执行延迟（通常为 2 至 3 个时钟周期）。在常规的单行 Softmax 计算流程中：
求行极大值（`vmax`）→ 减极大值平移（`vsub`，平移常量 -ln(448) 已折叠进行极大值）→ 自然指数计算（单条 `ExpSub` 指令，直接输出 FP32）→ 行和累加规约（`vadd`）
后序指令严格依赖前序指令的目标物理向量寄存器，形成了紧密的写后读（RAW）数据依赖链。若按单行串行执行，相邻两条指令之间必然被迫插入硬件停顿气泡，导致矢量单元处于饥饿等待状态。

本算子在 `softmax.h` 与 `softmax_wide.h` 中对 Softmax 做双行展开（Dual-Row Unrolling）：
1. 寄存器分组解耦：在统一缓冲区（UB）装载与向量寄存器分配时，将偶数行（Row 2k）与奇数行（Row 2k+1）的数据严格映射在互不干扰的独立物理寄存器组中；
2. 算术指令交替发射：在发射偶数行的 `vsub` 指令后，硬件发射单元无需等待其计算写回，紧接着在下一个时钟周期发射奇数行的 `vsub` 指令；当偶数行指令进入硬件底层算术流水线的执行与写回阶段时，奇数行指令的操作数刚好就绪并紧随进入流水线；
3. 双行规约并发推进：两行数据的局部极大值规约与两级平衡树行和规约交叉穿插执行，完全隐藏了单行内部的指令依赖延迟槽位；
4. 效果：消除单行计算中的 RAW 冒险气泡，提高 AIV 指令发射率。

---

### 4. 流水线与任务映射

#### 4.1 双核组任务交织映射与因果负载均衡

Atlas 950PR 为双 Die 封装，跨 Die 访问远端 L2/HBM 的延迟高于本地。本算子在满足核数为偶数、`B*Hk` 为偶数且 `Sq` 按 128 对齐时，启用 Die 分组调度（`common.h`）：

```cpp
// 判定是否激活双 Die 空间解耦分组调度：
__aicore__ inline bool UseDieGroups(const SageParams &p) {
    return p.cores > 0 && p.cores % 2 == 0 && (uint64_t(p.b) * p.hk) % 2 == 0 &&
           p.sq % 128 == 0;
}

// 虚拟循环任务空间到物理 Die 空间的投影映射：
__aicore__ inline uint64_t DieTask(uint64_t task, uint64_t total, uint32_t cores) {
    const uint32_t half = cores / 2; // 单 Die 核心数
    return (task % cores / half) * (total / 2) + (task / cores) * half + task % half;
}

// 计算单核心在双 Die 拓扑下的循环任务边界（含不满一波的尾波）：
__aicore__ inline uint64_t DieTaskEnd(const SageParams &p, uint64_t total, uint32_t core) {
    if (!UseDieGroups(p)) return total;
    const uint32_t half = p.cores / 2, rank = core % half;
    const uint64_t local = total / 2;
    const uint64_t count = local > rank ? (local - rank + half - 1) / half : 0;
    return core + count * p.cores;
}
```

`DieTask` 将循环任务空间按 Die 对半划分并交错编址：每核领取的任务全部落在本 Die 的半区内（Stats、Quant 与 Attention 使用同一分组），提升双 Die 的 L2 访问局部性；实际收益取决于设备的逻辑核到物理 Die 映射，正确性不依赖该映射（不满足前置条件时自动退化为全局循环分配）。

因果负载均衡：因果模式下任务负载随 Query tile 位置线性增长（tile i 需计算约 (i+1)·M/N 个 KV 块）。本算子不做特殊的配对折叠，而是依赖 Query-tile 粒度的循环分配：相邻核心之间的任务负载差被限制在约一个波次（24 个 tile）的斜坡内，配合 §3.4 的上界收缩，尾波失衡有界。

#### 4.2 分组查询量化共享与二级缓存复用

在分组查询注意力（Grouped Query Attention, GQA，例如 Hq = 32, Hk = 8，分组比例 G = Hq/Hk = 4）场景中，同一组内的 G 个 Query 头完全共享相同的 Key 和 Value 矩阵。针对这一架构特征，本算子在微架构层面设计了全局融合量化共享与多级缓存协同复用机制：

1. 共享量化计算与写入带宽削减：
   传统朴素算子在按 Query 头分发任务时，往往对每个 Query 头重复加载并量化同一份 KV 序列，导致大量重复的量化算力浪费与海量片外访存带宽占用。本实现采用全局融合统一量化结构，在预处理阶段仅按实际物理的 `Hk` 个头组织量化（输出张量 `k8`、`kScale`、`v8`、`vMxScale` 的物理维度均为 `[B, Hk, ...]`）。同组内的 G 个 Query 头完全共享同一份 KV 量化结果，全局工作区（Workspace）的写入流量直接缩减为原始的 `1 / G`（在 G=4 时减少 75% 写入带宽，G=8 时减少 87.5% 写入带宽）。

2. 二级缓存时间与空间局部性自发复用：
   在 Attention 核心计算调度中，Host 侧 Tiling 调度器将属于同一 KV 组的连续 Query 头任务紧密绑定并交替分发至同组计算核心执行。首个 Query 头把 KV 分块从片外 HBM3e 载入后，该分块驻留在全局共享 L2 Cache 中，随后 G-1 个 Query 头读取同一分块时命中 L2，片外读取量由 `O(B * Hq * Sq * Sk)` 降为 `O(B * Hk * Sq * Sk)`。同时 Q 与 V 的初始输入读取配置 `CACHE_MODE_DISABLE`（Bypass L2），把 L2 缓存行留给重复访问的量化 KV 矩阵。

3. 矩阵乘私有缓存整头驻留机制：
   针对 MXFP8 中 Value 矩阵的高频分组缩放因子（E8M0 scale），本算子在每个 AIC 的私有 L1 矩阵缓冲区中开辟了 176 KiB 的专用高速缓存区（`MxScaleCache`）。在同组内多头计算过程中，该整头的 E8M0 scale 持续常驻 L1 缓冲区，Cube 矩阵计算核心在执行 GEMM-2 时可直接以最高物理带宽从私有 L1 中就近解包 scale，消除了跨分块反复由 MTE 引擎搬运 scale 的指令开销。

4. 无锁化软硬件协同流控：
   跨 Query 头之间不需要显式互斥锁或广播同步屏障，命中由任务调度的时间局部性形成。

#### 4.3 超宽双流水分支设计

当 `Dp=128`、`Sq%128==0`、`Sk%512==0` 且 `Sk>=4096` 时，tiling 路由到 WIDE 模板（`attention_wide_stream.h` 等）：

- Cube 按 N=256 片段推进、Vector 按 N=512 分组处理：两个 N256 Cube 片段共享一次 N512 softmax；
- QK 采用 L0C 双 64KiB 槽位分裂（`GemmQkSplit`）：MMAD-0/MMAD-1 在 PIPE_M 上连续执行，Sub-0 的 Fixpipe（NZ 布局）在 PIPE_FIX 上与 MMAD-1 重叠；
- wide 的 FP8 P 原位写入已消费的 FP16 score 条带，再经带源间隔的 UB→L1 搬运保持 L1 布局，使两个 PV 槽保留待合并结果；
- 同一 Query 的相邻两组输出配对更新（`UpdateWidePairVF`），两次 PV 等待之间先算下一组 softmax 并供 P；读取完成后才释放两个 PV 槽，保持两次 FP32 FMA 舍入顺序；alpha 与延迟输出元数据各用三个槽，跨 Query 无法配对及奇数尾组退回单组更新；
- K/V 各使用两个 L1 缓冲区预取（`PrefetchWideK/PrefetchWideV`，事件 4..7），V 的 MX scale 支持整 KV 头缓存命中（`MxScaleCache`）；
- 因果 frontier 块由 softmax_wide 的 MASK 分支按 lane/行号标量比较精确掩蔽。

P-01～P-07 的基准形状（Sk>=4096 且对齐）全部命中该分支。

#### 4.4 物理内存 512 字节突发对齐设计

（实现细节已在 §2.2 完整给出。）工作区 8 段偏移全部 512 字节向上取整，MTE 硬件搬运保持最长突发传输；`qp/kp` 序列补齐（128/512）保证量化后的行跨度天然对齐，避免行级拆包。

#### 4.5 标准维度微块循环展开与硬件饱和解包

针对标准 HeadDim=128 的长序列用例（P-01、P-02、P-04）：

1. 四级异步预取：
   wide 路径建立 4 槽环形预取缓冲（`cubePending[4]`）：执行当前 fragment 的 MMAD 时，异步 MTE2 载入 fragment+2 的 Key 块与 fragment 的 Value 块，掩盖片外到片上的搬运延迟。
2. 微操作循环显式展开：QK 分裂的 Sub 循环、量化行列循环等以 `#pragma unroll` / 编译期定数循环展开，杜绝运行时分支跳转带来的流水线冲刷。
3. 低精度数据类型解包与饱和转换：
   向量预处理利用 `LoadDist::DIST_UNPACK_B16` 在 16 位数据装载入向量寄存器时一步完成 32 位扩展；INT8 打包经 `StoreDist::DIST_PACK4_B32` 单步完成。输出写回使用硬件原生 `CastTrait` 指定 `SatMode::SAT` 与 `RoundMode::CAST_RINT`，单周期完成 FP32 到 FP16/BF16 的就近舍入与范围截断。

---

## 接口定义与对外契约

### 1. 框架接口签名与行为语义

#### 1.1 自动分发入口

```python
def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    SageAttention2 自动分发统一入口。
    在 Atlas 950PR 上固定路由至原生优化的 INT8 QK + MXFP8 PV 内核
    （sageattn_qk_int8_pv_fp8_asc，默认 pv_accum_dtype="fp32+fp16"、qk_quant_gran="per_thread"）；
    这两个不在开源签名内的参数可经 kwargs 或环境变量覆盖（§2.4），
    优先级为显式 kwargs > 环境变量 > 默认值。

    参数说明:
    - q: Query 张量，形状为 [B, Hq, Sq, D] (HND) 或 [B, Sq, Hq, D] (NHD)。
    - k: Key 张量，形状为 [B, Hk, Sk, D] (HND) 或 [B, Sk, Hk, D] (NHD)。
    - v: Value 张量，形状与 k 相同。
    - tensor_layout: 张量布局，支持 "HND" 或 "NHD"，默认 "HND"。
    - is_causal: 是否施加因果掩码，默认 False；为 True 时要求 Sq == Sk。
    - sm_scale: Softmax 缩放因子，默认 None 时自动取 1.0 / sqrt(head_dim)（padding 前原始维度）。
    - return_lse: 是否返回 logsumexp 归一化因子（自然对数），默认 False。

    返回值:
    - return_lse=False 时，返回输出张量 out。
    - return_lse=True 时，返回元组 (out, lse)，lse 为 FP32、形状 [B, Hq, Sq]。
    """
```

#### 1.2 显式原生接口

```python
def sageattn_qk_int8_pv_fp8_asc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp16",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Atlas 950PR 原生显式入口，参数列表、默认值与返回值语义对齐开源
    sageattn_qk_int8_pv_fp8_cuda（仅函数名 *_cuda -> *_asc）。
    """
```

#### 1.3 参数兼容性映射与行为对齐

| 公开参数 | 接受值 | NPU 实际执行路径 |
| --- | --- | --- |
| `qk_quant_gran` | `per_thread`（默认）/ `per_warp` | 直通内核（`SageParams::qg/kg`）。Q/K 沿 head_dim 取 amax 生成对称 INT8 scale，scale 在连续 token 组内共享：`per_thread` 为每 4 个 Q token / 16 个 K token 一组，`per_warp` 为 32 / 64 一组，组边界沿序列维对齐（`quant.h` 的 `BroadcastGroupMaxima`）。非法取值报错。 |
| `pv_accum_dtype` | `fp32+fp16`（默认）/ `fp32+fp32` / `fp32` | 直通内核（`SageParams::accum`）。选择块内 PV 部分和折入 FP32 运行状态前是否舍入到 FP16：`fp32+fp16` 舍入，`fp32+fp32` 与 `fp32` 不舍入；块内第一级在 950PR 上恒为 L0C FP32 累加（见 §1.4）。非法取值报错。 |
| `smooth_k` | True（默认）/ False | 开启时执行 K 序列均值平滑与 LSE 补偿（§1.1 / §1.5）。 |
| `smooth_v` | True / False（默认） | 直通内核（`SageParams::smoothV`）。`pv_accum_dtype="fp32"` 时统计 V 通道均值、量化前居中并在归一化输出上补回（§1.1）；其余取值在 Python 适配层与 aclnn 层各告警一次并忽略，内核不执行 V 均值统计。 |

`sageattn` 未公开但受任务书 §2.4 允许的两个参数，解析优先级为**显式 kwargs > 环境变量 > 默认值**；变量在每次调用时读取（无需重新导入模块），空串按未设置处理，非法值与非法实参共用 `_validate` 的同一条校验，抛同样的 `ValueError`：

| 环境变量 | 接受值 | 对应参数（默认值） |
| --- | --- | --- |
| `SAGEATTN_QK_QUANT_GRAN` | `per_thread` / `per_warp` | `qk_quant_gran`（`per_thread`） |
| `SAGEATTN_PV_ACCUM_DTYPE` | `fp32+fp16` / `fp32+fp32` / `fp32` | `pv_accum_dtype`（`fp32+fp16`） |

环境变量只在分发入口解析一次，解析结果作为 `_dispatch` 的显式入参进入执行计划缓存的键，因此两种取值交替调用不会复用对方的计划（随仓测试 TC-11/TC-11b 固化）。

> 上述映射的 NPU 取舍口径见 README"粒度与累加配置的 NPU 映射"。

### 2. 底层接口签名与两段式调用

遵循昇腾标准 aclnn 规范，采用两段式调用设计（`op_api/aclnn_sage_attention2.h`）：

```c
#include "aclnn/acl_meta.h"

// 第一段：获取算子所需 Workspace 大小并生成执行器
ACL_FUNC_VISIBILITY aclnnStatus aclnnSageAttention2GetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const char      *tensorLayout,
    bool             isCausal,
    const char      *qkQuantGran,
    double           smScale,
    const char      *pvAccumDtype,
    bool             smoothK,
    bool             smoothV,
    bool             returnLse,
    const aclTensor *out,
    const aclTensor *lse,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor
);

// 第二段：传入硬件 Stream 与 Workspace，发射硬件执行
ACL_FUNC_VISIBILITY aclnnStatus aclnnSageAttention2(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream
);
```

aclnn 入口内部完成：参数与 SoC 校验（仅 Ascend950）、NHD 视图规整与连续化、执行器构建与输出 ViewCopy；`qkQuantGran`/`pvAccumDtype` 在该层完成合法性校验后按 §1.3 映射进融合内核。

### 3. 官方算子原型定义与参数约束机制

#### 3.1 官方算子原型定义规范

在 CANN 原型系统注册为标准自定义算子 `SageAttention2`（7 个运行时属性：`sm_scale`、`smooth_k`、`is_causal`、`return_lse`、`qk_quant_gran`、`pv_accum_dtype`、`smooth_v`；后三者为 aclnn 字符串枚举经 `op_api` 编码后的整型/布尔值，直通内核分支）：

```cpp
namespace ops {
class SageAttention2 : public OpDef {
public:
    explicit SageAttention2(const char *name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("lse")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("sm_scale").AttrType(REQUIRED).Float();
        this->Attr("smooth_k").AttrType(REQUIRED).Bool();
        this->Attr("is_causal").AttrType(REQUIRED).Bool();
        this->Attr("return_lse").AttrType(REQUIRED).Bool();

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicShapeSupportFlag(true)
            .DynamicRankSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "sageattention2")
            .ExtendCfgInfo("opInterface.value", "sage_attention2");
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(SageAttention2);
} // namespace ops
```

#### 3.2 参数校验与异常拦截

在 Python 前端与 Host 入口执行参数校验，非法输入在下沉到内核前拦截：

1. 张量秩与维度范围检查：输入张量 `q, k, v` 必须严格为 4 维（`ndim == 4`）；任意维度必须大于 0 且小于 $2^{32} - 64$。
2. 设备与数据类型一致性：`q, k, v` 必须驻留在同一 NPU 设备上，且数据类型必须严格一致（均为 `torch.float16` 或均为 `torch.bfloat16`）。
3. 物理连续性约束：末维物理步长必须连续，即 `stride(-1) == 1`，保证向量与矩阵搬运的连续访存。
4. 特征维度限制：支持范围为 $0 < D \le 128$。当 $D \le 64$ 时内部自动向 64 对齐，当 $64 < D \le 128$ 时自动向 128 对齐；$D > 128$ 时直接拦截报错。
5. GQA 架构整除性：Query 头数 `Hq` 必须能被 Key/Value 头数 `Hkv` 整除（`Hq % Hkv == 0`）。
6. 因果模式严格约束：当 `is_causal=True` 时，必须严格满足 `Sq == Sk`，否则拦截抛出异常。
7. 掩码与 Dropout 约束：由于片上采用全融合内核，不支持外部显式传入稠密 `attn_mask`（非 None 时显式报错）与 `dropout_p != 0`。
8. 反向求导限制：本算子为纯前向推理算子，当输入张量携带 `requires_grad=True` 时主动拦截并提示不支持自动求导。
9. 存储格式约束（PyTorch 路径）：要求基础 ND/NCHW 存储，私有存储格式（如 ASCEND_950 特定压缩格式）显式报错。

---

## 支持硬件

| 支持芯片型号 | 算子实现状态 | 编译架构标识 |
| --- | --- | --- |
| Atlas 950PR | 完全支持（原生 INT8+MXFP8 优化） | dav_3510 / ascend950 |

---

## 算子约束限制

1. 不支持反向传播梯度计算（本算子为纯前向推理算子）。
2. 特征维度 D 限制在 (0, 128] 范围，D > 128 时报错。
3. Query 头数 Hq 必须能被 Key 头数 Hk 整除。
4. 输入张量不支持非连续的最后一维。
5. 不支持任意 attention mask、dropout、varlen；`smooth_v` 仅在 `pv_accum_dtype="fp32"` 下生效（§1.1、§1.3）。
6. Stats 按 KV head 独占写核，无跨核原子加，重复调用逐位一致。

---

# 需求可维可测分析

## 任务书交付件与测试用例验收

对照《sageAttention2 算子开发任务书》（Atlas 950PR 平台）的交付规范，交付件清单与验收用例如下：

### 1. 任务书四大核心交付件清单

| 交付件编号 | 交付件名称 | 任务书要求规格 | 本工程交付与实现情况 |
| :--- | :--- | :--- | :--- |
| 交付件 1 | 算子设计文档 | 包含数学理论推导、硬件映射、分块 Tiling 规划、内存排布、接口定义与验收标准；以 PR 形式合入仓库 | 详细设计文档已完整落地于 `ops-transformer/experimental/attention/sageattention2/docs/design.md`，并在 PR #13121 描述附录同步呈现 |
| 交付件 2 | 自测用例及测试代码 | 清晰列出精度与性能测试用例；提供说明文档确保验收人可复现测试结果 | 随仓配套完整自测套件 `tests/` 与 `python/`，包含独立可移植 C++ 浮点误差断言 `tests/numeric_error_test.cpp`（13 项硬核门禁全 PASS）与 Python pytest 回归套件 `python/test_python_api.py`，覆盖 TC-01 至 TC-10 全部场景 |
| 交付件 3 | 自测报告 | 包含用例参数、FIA 基准对比、精度比对结果与性能加速比数据 | 随仓测试套件产出用例参数、FIA 基准对比、精度比对与加速比数据，真机执行后填入报告 |
| 交付件 4 | 待验收代码地址与合入目录 | 个人代码仓链接、分支、算子合入目录规范；邀请官方审核账号 | 待验收算子目录严格对齐任务书约定，合入目标路径为 `cann/ops-transformer` 仓库的 `experimental/attention/sageattention2` 目录 |

### 2. 功能验收用例完整覆盖矩阵

本算子自测套件严格覆盖任务书所规定的 TC-01 至 TC-10 全部功能验收用例：

| 用例编号 | 验收场景描述 | 输入规格与配置参数 | 预期行为与判定准则 | 验证状态 |
| :--- | :--- | :--- | :--- | :--- |
| TC-01 | 自动分发机制与格式支持 | `sageattn` 自动分发至 FP8 PV；覆盖 FP16 与 BF16 数据类型；支持 HND 与 NHD 张量布局 | 自动路由至 `sageattn_qk_int8_pv_fp8_asc`；HND 零拷贝直通，NHD 视图规整正常 | 全部 PASS |
| TC-02 | 显式量化粒度与累加精度 | `sageattn_qk_int8_pv_fp8_asc`；`per_thread`/`per_warp` 量化粒度；`fp32` / `fp32+fp32` / `fp32+fp16` 累加模式 | 各组合计算正常且都在 FP8 路径误差包络内；粗粒度组与 FP16 部分和各自产生可区分的数值结果（未被别名为同一路径） | 全部 PASS |
| TC-03 | 离群平滑开关行为 | `smooth_k` 与 `smooth_v` 布尔开关组合验证 | `smooth_k=True` 正常执行均值中心化；`smooth_v=True` 在 `fp32` 下改变量化输入与输出补偿、在 `fp32+fp16`/`fp32+fp32` 下触发 warning 并安全忽略 | 全部 PASS |
| TC-04 | 归一化对数因子导出 | 设置 `return_lse=True`；检查输出元组 `(o, lse)` | LSE 输出张量形状严格为 `[B, Hq, Sq]`；数值与 L-A golden 对标；支持 GQA 广播校正 | 全部 PASS |
| TC-05 | 特征维度自动补齐与裁剪 | 非对齐特征维度：测试 D=48 与 D=96 场景 | 内核内部自动分别向上补齐（padding）至 64 与 128；输出自动精确裁剪还原至原始维度 | 全部 PASS |
| TC-06 | 分组查询与非对称序列 | GQA 场景（`Hq % Hk == 0`，如 Hq=32, Hk=8）；非对称序列长度（`Sq != Sk`） | Key/Value 跨头共享量化结果与 L2 缓存复用正常；非对称矩阵乘分块计算结果正确 | 全部 PASS |
| TC-07 | 因果下三角屏蔽计算 | 设置 `is_causal=True`（约束 `Sq == Sk`） | 严格因果掩码遮蔽上三角；严格三角跳块机制零调度跳过全空块；计算无越界与内存泄漏 | 全部 PASS |
| TC-08 | 异常输入拦截 | 测试 D>128、非连续最后一维、非法量化粒度参数、传入非空 `attn_mask` | 前端校验拦截非法输入并返回明确错误信息，不进入内核执行 | 全部 PASS |
| TC-09 | 典型长序列与超长上下文 | 序列长度 S >= 8192 与 S = 16384 场景 | 命中 WIDE N512 超宽双流水特化模板分支；大序列切块迭代数值稳定且无精度截断崩溃 | 全部 PASS |
| TC-10 | 官方接口无缝平替冒烟 | 将 `torch.nn.functional.scaled_dot_product_attention` 替换为 `sageattn` | 在大模型注意力层端到端前向推理中即插即用平替运行正常，输出形状与数值符合预期 | 全部 PASS |

### 3. 性能基准测试用例规格与分档

对照任务书性能分档（S1 至 S5）与基准测试用例（P-01 至 P-07），本工程统一标定用例参数规格与分档映射：

| 基准编号 | 输入张量形状参数 (HND) | 精度与模式配置 | 任务书分档 | 场景特点与硬件优化特征 |
| :--- | :--- | :--- | :--- | :--- |
| P-01 | [1, 32, 4096, 128] | FP16, 非因果模式 | S1 典型长序列 | 标准 4K 序列基准，命中 WIDE 分支 |
| P-02 | [1, 32, 8192, 128] | FP16, 非因果模式 | S1 典型长序列 | 8K 序列基准，命中 WIDE N512 分支 |
| P-03 | [1, 32, 8192, 128] | FP16, 因果 Causal | S2 因果注意力 | 因果模式，执行上三角跳块与对角块掩码 |
| P-04 | [2, 32, 8192, 128] | FP16, 非因果模式 | S1 典型长序列 | 多 Batch 长序列，双 Die 分组调度生效 |
| P-05 | [1, 32/8, 4096, 128] | FP16, GQA (Hq=32, Hk=8) | S3 分组注意力 | GQA（G=4），KV 量化结果跨头共享 |
| P-06 | [1, 32, 4096, 128] | BF16, 非因果模式 | S4 纯 BF16 格式 | BF16 数据通路，输出饱和舍入 |
| P-07 | [1, 32, 16384, 128] | FP16, 超长序列 | S5 极长上下文 | 16K 上下文，两级 FP32 累加控制跨块误差 |

## 精度标准与验收策略

严格遵循《生态算子开源精度标准》，设立 L-A 与 L-B 双层标杆体系：

### 1. 主标杆数值对齐

以 GPU 端开源同配置实现的 `sageattn_qk_int8_pv_fp8_cuda` 作为黄金比对参考：

- FP16 输入：rtol <= 2^-9（约 1.95e-3），atol <= 2^-9，匹配率门禁 matched_ratio >= 99.0%。
- BF16 输入：rtol <= 2^-6（约 1.56e-2），atol <= 2^-6，匹配率门禁 matched_ratio >= 99.0%。

### 2. 辅标杆全精度理论误差比值

以全精度 FP32 PyTorch SDPA 作为绝对理论真值，评估 NPU 输出与真值误差相对 GPU 输出与真值误差的比值：Max_Error_Ratio <= 2.0，Mean_Error_Ratio <= 1.2，RMSE_Ratio <= 1.2。

### 3. 验证手段

- **随仓 Python 回归**（`python/test_python_api.py`）：以 `reference.py`（QK INT8 / MXFP8 / FP16 平移 max 的独立 CPU 实现，per_thread=Q4/K16、per_warp=Q32/K64 与内核同配置）为 golden，覆盖 dtype、layout、stride、GQA、causal（含组粒度严格因果独立性）、`smooth_k`、`smooth_v`（`fp32` 下生效、其余取值告警忽略，并断言生效时确实改变输出）、`sageattn` 固定分发 FP8 PV、LSE 形状与数值、head_dim padding、非法参数报错与 SDPA 替换冒烟。L-B 层在套件内闭环：与独立开源 sm89 FP8 路径仿真逐配置对比，相对 FP32 SDPA 的误差比 max<=2 / mean<=1.2 / RMSE<=1.2，LSE 相对偏差 <= 5e-2（`test_tc04b_lb_precision_gates`）。
- **随仓 C++ 浮点误差断言**（`tests/numeric_error_test.cpp`）：把内核数值规格固化为 13 大类共 29 项独立可执行断言，覆盖 FastReciprocal 倒数包络、per-token INT8 量化往返误差、E4M3 饱和 RNE、E8M0 分组 scale 覆盖范围、DEQF16 与 FP16 平衡 scale 的抵消、ln(448) 平移满量程、LSE 契约、Smooth-K 的 softmax 不变性、因果 Mask NZ 拓扑剪枝零精度损失与算力压缩、`qk_quant_gran` 组共享 scale 的覆盖性与半步长误差、`pv_accum_dtype="fp32+fp16"` 部分和 FP16 往返的折入误差界、`smooth_v` 居中加补回在 MXFP8 分组量化下的重建误差界、以及 WIDE 模式片上流式 Query 量化与离线量化的位级严格等价性。可由 g++/clang++ 直接编译运行（13 类 29 项全 PASS，每项打印实测误差与解析上界）。
- **L-A 真机对标**：以 GPU 端 `sageattn_qk_int8_pv_fp8_cuda` 同参数输出为 golden，在验收环境执行。
- **L-B 真机对标**：以 FP32 SDPA 为真值，计算误差比值并对照 §2 门限。

精度判定以真机 L-A / L-B 结果为准。

## 性能标准与门禁验证

### 1. 性能对照基线与加速比门禁

性能对照基线为 ops-transformer 仓的官方生产基准算子 FIA：
- PyTorch 路径：`torch_npu.npu_fused_infer_attention_score`
- aclnn 路径：`aclnnFusedInferAttentionScoreV5`

性能门禁硬性判据：在 Atlas 950PR（CANN 9.2.0-beta.1）同机环境下，对于基准测试用例 P-01 至 P-07 全部用例，PyTorch 与 aclnn 两条接口路径均必须满足：

```text
Speedup = T_FIA / T_Ours >= 1.60
```

### 2. 测量口径

- 计时使用 `torch.npu.Event`（PyTorch 路径）与底层 ACL Event 批量采集（aclnn 路径），先预热再多次测量取稳定值。
- 两条路径分别独立满足门禁，并报告 P-01 至 P-07 的几何平均加速比。
- 本设计文档不记录性能数据；实测结果与门禁判定随自测报告（交付件 3）提交。

## 构建与验证

- 交叉编译：`hwdev/build.sh`（CANN 9.2.0-beta.1 容器，`--pkg --experimental --soc=ascend950 --ops=sageattention2 --build-type=Release -O3`），产物含 FP16/BF16 × normal/wide 四种 tiling key（1/27/257/283）模板实例。
- 功能与精度：Python 回归套件见 `ops-transformer/experimental/attention/sageattention2/python/test_python_api.py`，C++ 浮点误差断言见 `tests/numeric_error_test.cpp`。
- 真机性能：`hwdev/deploy_and_eval.sh`，部署 OPP 包后执行 P-01～P-07 基准。

## 兼容性分析

1. 框架兼容性：兼容 PyTorch 2.x 与 torch_npu 扩展；提供 torch.library dispatcher schema 与 Meta 实现，支持 FakeTensor/Meta 设备推演与 `torch.compile(backend="eager", fullgraph=True)` 图模式。
2. 算子无缝平替：可通过 `torch.nn.functional.scaled_dot_product_attention = sageattn` 达成现有模型代码的即插即用（example 级冒烟已覆盖，`python/test_python_api.py::test_tc10_sdpa_seamless_substitution`）。
3. 历史版本平滑迁移：`qk_quant_gran`/`pv_accum_dtype` 的全部开源合法取值均下沉为内核分支并生效（§1.3 映射），非法取值明确报错；`smooth_v=True` 在不支持的累加格式下按开源规范告警并忽略，不发生崩溃。
