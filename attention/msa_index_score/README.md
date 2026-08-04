# MsaIndexScore

## 产品支持情况

| 产品 | 是否支持 |
|------|:--------:|
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A5 训练系列产品/Atlas A5 推理系列产品</term> | × |

## 功能说明

- **算子功能**：MiniMax-M3 / DeepSeek 类稀疏注意力（MSA）中 Lightning Indexer 的 block 打分算子。
  用一路"廉价"的 index Q 与分页 index K cache 做点乘，并在每个 sparse block 内取最大值，
  产出供下游 `topk` 选块使用的 block score。Prefill 与 Decode 归一到同一个算子。

- **核心运算**：矩阵乘（`Mmad`）+ 段长为 `block_size` 的行方向分段最大值（`ReduceMax`）。
  三步顺序为 **点乘 → mask → max**，mask 必须先于 max 施加。

- **计算公式**：对请求 $b$、全局 token $t$、head $h$、逻辑 block $\beta$，

$$
q_{\text{pos}}(t) = \texttt{start\_loc}[b] + \bigl(t - \texttt{cu\_seq\_q}[b]\bigr)
$$

$$
\texttt{score}[h, t, \beta] =
\max_{\substack{j \in [\beta P,\ \beta P + P) \\ j \le q_{\text{pos}}(t),\ j < \texttt{kv\_len}[b]}}
\bigl\langle \mathbf{q}_{t,h},\ \tilde{\mathbf{k}}_{j} \bigr\rangle
$$

  其中 $P$ = `block_size`。非量化：$\tilde{\mathbf{k}}_j = \texttt{key\_cache}[\texttt{block\_table}[b,\lfloor j/P\rfloor], j\bmod P, 0, :]$。
  int8 前融合：$\tilde{\mathbf{k}}_j = \mathrm{cast}(\texttt{key\_cache}[\ldots])\cdot
  \texttt{scale}[\texttt{page}, 0, j\bmod P]$，`scale` shape 为 $[NP, N_{kv}, P]$。
  窗口内无可见 $j$ 时输出 $-\infty$。**非量化路径不在 score 上再乘 scale。**

## 参数说明

> **说明：**
> $T$ = `total_q`（varlen 扁平后的 query token 总数），$B$ = batch，
> $H_q$ = `num_q_heads`（index heads），$H_{kv}$ = `num_kv_heads`，$D$ = `head_dim`，
> $P$ = `block_size`，$NP$ = KV cache 的物理 page 数，$MB$ = 每 batch 最大 block 数。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|----------------|------|----------|----------|
| query | 输入 | index query，shape $[T, H_q, D]$ | BFLOAT16, FLOAT16 | ND (TND) |
| key_cache | 输入 | 分页 index key cache，shape $[NP, P, H_{kv}, D]$ | 同 query，或 INT8（前融合量化） | ND (BSND) |
| block_table | 输入 | 逻辑 block → 物理 page 映射，shape $[B, MB]$ | INT32 | ND |
| scale | 可选输入 | **仅 int8 路径**：反量化系数 $[NP, N_{kv}, P]$；非量化必须为空 | FLOAT | ND |
| actual_seq_lengths_query | 输入 | query 长度前缀和（cumsum），shape $[B+1]$，首元素为 0 | INT32 | ND |
| actual_seq_lengths_key | 输入 | 每个请求可见的 kv 长度，shape $[B]$ | INT32 | ND |
| start_loc | 输入 | 每个请求首个 query token 在自身 kv 序列中的位置，shape $[B]$ | INT32 | ND |
| block_size | 属性 | 一个 sparse block / page 的 token 数，默认 128 | INT | - |
| fill_invalid_tail | 属性 | 是否把不可见 block 写成 $-\infty$，默认 true | BOOL | - |
| score | 输出 | block score，shape $[H_q, T, \text{RoundUp}(MB, 16)]$ | FLOAT | ND |

**Prefill / Decode 的参数映射**

| 语义量 | Prefill | Decode |
|--------|---------|--------|
| `actual_seq_lengths_query` | 真实 varlen 前缀和 | $\texttt{cu}[i] = i \cdot L_q$（$L_q$ 常为 1） |
| `start_loc[b]` | 已缓存的 prefix 长度（chunked prefill） | `global_seq_len[b] - `$L_q$ |
| `actual_seq_lengths_key[b]` | 该请求当前 kv 总长 | 同左 |

## 约束说明

- `block_size` 当前仅支持 **128**，且必须与 `key_cache` 的第 1 维一致。
- `num_kv_heads` 当前仅支持 **1**（MSA 的 index K 为全 head 共享单头）。
- `head_dim` 必须是 16 的正整数倍，且当前不超过 128。`head_dim == 1` 的标量特化不支持。
- `num_q_heads` 必须能被 `num_kv_heads` 整除。
- **非量化**：`key_cache` dtype 与 `query` 相同，**不得**传入 `scale`。
- **int8 前融合**：`key_cache` 为 INT8，必须传入 `scale`，shape `$[NP, N_{kv}, P]$`（与 key 物理 page 对齐）。
- 输出末维是 **block 数**而非 kv token 数。
- 本算子**不包含** `topk`，输出止于 block score。

## 调用示例

| 调用方式 | 样例代码 | 说明 |
|----------|----------|------|
| aclnn 单算子调用 | [test_aclnn_msa_index_score.cpp](./examples/test_aclnn_msa_index_score.cpp) | 内置 CPU golden 的端到端精度自验证 |
| 接口文档 | [aclnnMsaIndexScore.md](./docs/aclnnMsaIndexScore.md) | 两段式接口说明 |
| 测试说明 | [tests/README.md](./tests/README.md) | 用例矩阵与运行方式 |

编译与运行：

```bash
bash build.sh --pkg --soc=ascend910b --ops=msa_index_score -j32
./build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/tmp/msa_opp
export ASCEND_CUSTOM_OPP_PATH=/tmp/msa_opp/vendors/custom_transformer
bash build.sh --run_example msa_index_score eager cust --vendor_name=custom
```

## 实现说明

本算子的 device 侧基于 **CATLASS 模板库**（`3rdparty/catlass`，header-only）实现：

| 层 | 选型 |
|----|------|
| ArchTag | `Catlass::Arch::AtlasA2` |
| DispatchPolicy | `Catlass::Gemm::MmadAtlasA2Pingpong<false>` |
| L1TileShape / L0TileShape | `GemmShape<128, 128, 128>` |
| AType / BType / CType | `GemmType<bf16\|fp16, RowMajor>` / `GemmType<bf16\|fp16, ColumnMajor>` / `GemmType<float, RowMajor>` |
| BlockMmad | `Catlass::Gemm::Block::BlockMmad<...>`，逐 page 经 `block_table` 间接寻址 |
| BlockEpilogue | 自定义 `MsaSegRowMaxEpilogue`（catlass 无 reduce 类 Tile，属粒度 B 自定义） |
| Scheduler | 自定义 `MsaTaskScheduler`，varlen + 因果裁剪的三元任务枚举 |

catlass 通过 `op_host/CMakeLists.txt` 的 `add_ops_compile_options(... OPTIONS -I<repo>/3rdparty/catlass/include)`
注入 kernel 编译选项，不拷贝到 `op_kernel/` 下。

## Device 侧调试打印

`op_host/CMakeLists.txt` 默认打开：

```
-DMSA_INDEX_SCORE_DEBUG=1 -DENABLE_ASCENDC_DUMP -DASCENDC_DUMP=1 -DASCENDC_DEBUG
```

会在 **core0 / 首个 task / 首个 S-tile / 首个 pass** 打印：

| 阶段 | 位置 | 内容 |
|------|------|------|
| tiling / task 参数 | AIC+AIV | batch、Hq、D、fullEndBlk、visibleEndBlk 等 |
| 输入 Q / K | AIC | `DumpTensor` 各取前 16 个元素 |
| 点乘后 S | AIC workspace + AIV UB | fp32，前 16 个 |
| mask 后 S | AIV UB | 同窗口；边界 block 另打尾部看 `-inf` |
| max 后 score | AIV UB | 一行 × 8 个 sparse block |
| scale/fill 后最终 score | AIV UB | 写回前的结果 |

example 默认只跑小尺寸 `L0-debug-trace`（`Hq=2, D=16, qLen=2, kvLen=5, startLoc=3`）：
host 侧会逐步打印 Q/K → DOT → MASK → MAX → scale → device 输出；device 侧 dump
也会把完整 Q 行、可见 K、以及 mask 切分点附近的 S 打出来。完整矩阵：

```bash
MSA_INDEX_SCORE_RUN_ALL=1 bash build.sh --run_example msa_index_score eager cust --vendor_name=custom
```

正式交付前把 CMake 里的 `-DMSA_INDEX_SCORE_DEBUG=1` 改为 `0`（或删掉 dump 相关 `-D`）即可关闭。

**关键实现要点**

- **M 维定义为"请求内 `(token, head)` 扁平行、head 为低位"**。这是 Prefill / Decode 能归一到
  同一算子的基础：Decode 下所有 head 共享同一次 KV 读取，相比"固定 head + 连续 token"的分块
  可节省 $H_q$ 倍 KV cache 带宽，而 Prefill 的流量完全相同。
- **归约段长恰等于 page 宽度**，因此不存在跨 N-tile 的归约依赖，block 轴可自由分核，
  无需 atomic、两遍归约或 AIV 域 barrier。
- **Atlas A2 没有 L0C→UB 直通路径**，Cube 产出的 $S$ 必须经 GM workspace 中转给 Vector。
  实现采用每 AIC 私有的三级轮转 workspace（L2 驻留），配 `CrossCoreFlagWithReverse` 同步。
- **block 三态分流**：全可见 block 走无 mask 快通道；边界 block（恒 1~2 个）逐行置 $-\infty$；
  全不可见 block 跳过 GEMM 并直接填充。

## 后续规划

当前为首版，聚焦编译与精度正确性，性能调优尚未开展。已识别的优化项：

1. **S 中间结果落盘改 fp16**：$S$ 的 GM 往返流量固定为 $2 M_{tot}\cdot kv\_len\cdot \text{sizeof}(S)$，
   与分块无关。改 fp16 可使算术强度从 16 提升到 32 MAC/byte，跨过 A2 的机器平衡点。
2. **Q 常驻 L1**：当前每个 S-tile 重新搬一次 Q，可改为一个任务内只搬一次。
3. **block 轴分核**：Decode 小 batch 下任务数少于核数，需按 block 轴再切分填满 AIC。
4. **UB 双缓冲 + `ENABLE_UNIT_FLAG`**：隐藏 L2 读延迟并让 FIXPIPE drain 与 MAD 重叠。
5. **`num_kv_heads > 1`**：需改用 FAI 风格的多段 Nd2Nz（`ndNum`）搬运 Q。
6. **JIT 模式支持**：当前 catlass 头文件未随算子包安装，`--jit` 构建的运行期编译会失败，
   需要额外的安装规则或改为 vendoring。
