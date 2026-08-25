# FlashAttn

## 功能说明

- 算子功能：基于FlashAttention算法实现self-attention（自注意力）计算，训练推理归一化，仅支持非量化场景。
- 关键特性：

    - **GQA（Grouped Query Attention）**：Q的head数（Q_N）为KV head数（KV_N）的整数倍，同一组Q head共享一份KV，减少KV显存与计算量。
    - **多布局支持**：Q/输出支持BSND、BNSD、TND三种布局；KV额外支持Paged Attention（PA_BBND/PA_BNBD/PA_NZ）分页KV cache。
    - **Paged Attention**：KV cache按block分页存储，通过block_table索引物理block，支持推理场景KV cache复用与扩容。
    - **变长序列**：通过cu_seqlens（TND packing）与seqused_q/seqused_kv（序列截断）支持batch内变长序列。
    - **FlashDecode**：decode场景（S1小、S2长）将KV方向split-K切分到多核并行，自动启用，无需显式配置。
    - **稀疏掩码**：mask_mode支持全计算（0）、causal（3）、window（4）三种模式，配合固定(2048, 2048)的attn_mask模板。
    - **softmax_lse输出**：通过return_softmax_lse开关输出每行softmax的Log-Sum-Exp结果，用于KV cache拼接等跨batch融合场景。

- 计算公式：

  自注意力的正向计算公式如下：

  $$
  Attention(Q,K,V)=Softmax(SoftmaxScale * (Q \cdot K^T))V
  $$

  其中$Q \in (Q_S, Q_N, D)$，$K, V \in (KV_S, KV_N, D)$，$D$为head dim。默认$SoftmaxScale=1.0$，标准实现中通常取$1/\sqrt{D}$作为缩放因子。

  FlashAttention算法将$Q_S \times KV_S$的注意力矩阵按块（tile）分块计算，避免实例化完整注意力矩阵，显存复杂度由$O(Q_S \cdot KV_S)$降至$O(tile_M \cdot tile_N)$。

## Quick Start

### 1. 编译

**① 全量编译（默认推荐，flash_attn 与 flash_attn_metadata 需同时编译，生成所有 tiling key 的 kernel 变体）：**

```bash
bash build.sh --pkg --soc=ascend950 --ops=flash_attn,flash_attn_metadata
```

**② 限制并行线程数（`-j` 可选参数）**：指定并行编译线程数，默认按机器核数并行。当机器内存不足、或 cgroup 实际限制核数小于 `/proc/cpuinfo` 报告值（如本机报告 128 核但 cgroup 限制 32 核）导致编译 OOM 或失败时，需显式指定较小值，例如：

```bash
bash build.sh --pkg --soc=ascend950 --ops=flash_attn,flash_attn_metadata -j16
```

**③ 只编译指定 tiling key 变体（`--tiling_key` 可选参数）**：默认全量编译会生成所有 tiling key 的 kernel 变体，耗时较长。当只需要调试/验证特定布局与配置组合（如仅 BSND/BNSD/TND × 无 mask × D=128）时，可只编译对应变体以加速，例如：

```bash
bash build.sh --pkg --soc=ascend950 --ops=flash_attn,flash_attn_metadata --tiling_key="2279866368;2279866369;2279866370"
```

- 各 tiling key 的编码含义见[开发者指南 §5](#5-tilingkey)。

编译产物：`build_out/cann-ops-transformer-custom_linux-x86_64.run`

### 2. 运行 tiling 单元测试

```bash
bash build.sh --ops=flash_attn,flash_attn_metadata --soc=ascend950
cd build && ctest -R flash_attn_tiling --output-on-failure
```

用例定义：`tests/ut/op_host/arch35/test_flash_attn_tiling.csv`。另有 inferShape 与 inferDtype 单测：`tests/ut/op_host/test_flash_attn_shape_infershape.cpp`、`test_flash_attn_dtype_infershape.cpp`。

### 3. 运行 e2e 精度测试（NPU真机）

```bash
python tests/pytests/test_flash_attn.py --device_id 0            # 跑全部functional用例
python tests/pytests/test_flash_attn.py --device_id 0 --case_id <id>   # 跑单个用例
python tests/pytests/test_flash_attn.py --device_id 0 --use_gpu --gpu_device 0   # 用GPU端flash_attn库对照
```

测试框架说明见 [tests/pytests/readme.md](./tests/pytests/readme.md)（npu/gpu/cpu 三后端、functional/redline/perf 用例矩阵、结果CSV输出）。

### 4. 接口调用

- **torch 接口**（`flash_attn_metadata` + `flash_attn` 的函数原型、参数说明、返回值说明）：[flash_attn.md](../../torch_extension/cann_ops_transformer/docs/zh/flash_attn.md)

调用分两步：先用`flash_attn_metadata`生成负载均衡metadata，再调用`flash_attn`主算子。完整调用示例（含代码）见接口文档的[调用示例](../../torch_extension/cann_ops_transformer/docs/zh/flash_attn.md#调用示例)章节。

## 开发者指南

### 1. 基础概念

#### 1.1 MHA 与 GQA

- **MHA（Multi-Head Attention）**：Q、K、V 各有一组 head，Q_N = KV_N，每个 Q head 各自与对应的 KV head 做注意力。
- **GQA（Grouped Query Attention）**：Q_N 是 KV_N 的整数倍，即 g = Q_N / KV_N，同一组 g 个 Q head 共享一份 KV。用于减少 KV 的显存与计算量（Q_N 远大 KV_N 时，KV cache 显著缩小）。
- 本算子要求 Q_N % KV_N == 0（GQA 场景）。kernel 中 `gSize = g`，任务按 (bN2, gS1, s2) 三层循环展开，其中 bN2 = batch × KV_N 扁平索引，gS1 = Q 序列 × g 的扁平索引。

#### 1.2 PagedAttention 场景及 layout_kv 取值

- **PagedAttention**：将 KV cache 按 block（block_size 个 token）分页存储，通过 `block_table` 索引物理 block，避免 KV cache 预留过大的连续显存，支持动态分配与复用。推理场景（长序列 decode 且 KV cache 会增长）通常使用。
- 启用方式：`layout_kv` 取 PA 布局，同时传入 `block_table`（INT32，shape (B, max_blocks_per_seq)）。block_size 由 KV 的 PA 布局推断，取值需 ∈ [16, 1024] 且为 16 的整数倍。
- `layout_kv` 取值及对应 KV shape：

| layout_kv | KV shape | 说明 |
|---|---|---|
| BSND / BNSD / TND | (B, S, N, D) / (B, N, S, D) / (T, N, D) | 连续 KV，非分页 |
| PA_BBND | (num_blocks, block_size, KV_N, D) | block 内按 BSND 排布 |
| PA_BNBD | (num_blocks, KV_N, block_size, D) | block 内按 BNSD 排布 |
| PA_NZ | (num_blocks, KV_N, D/16, block_size, 16) | NZ 分块格式 |

- PA 场景约束：PA 时 `seqused_kv` 必填；非 PA 场景传入 `block_table` 会直接报错。`layout_kv` 为 PA 时，`layout_q`/`layout_out` 维持 BSND/BNSD/TND 不变。

#### 1.3 变长序列（TND）

- **TND 布局**：将 batch 内所有序列的 token 拼接为一个长序列（总长度 Q_T / KV_T），shape 记为 (Q_T, Q_N, D) 与 (KV_T, KV_N, D)，消除 padding 浪费。用于 batch 内序列长度不等的场景。
- 启用方式：`layout_q`（及 `layout_kv`）取 TND，同时传入 `cu_seqlens_q`/`cu_seqlens_kv` 描述每个序列的起始位置。
- TND 场景 `max_seqlen` 属性不参与实际切分（以 cu_seqlens 为准），`layout_out` 必须为 TND。
- 功能说明中的"Packed Sequence"即 TND 场景。

#### 1.4 cu_seqlens 与 seqused

- **cu_seqlens_q / cu_seqlens_kv**：INT32 一维，shape (B+1,)，单调不减且首元素为 0，表示每个序列的累积起点/终点（TND 布局用于恢复各序列的边界）。仅 TND 布局允许传入。
- **seqused_q / seqused_kv**：INT32 一维，shape (B,)，表示每个 batch 实际生效的序列长度（截断冗余 padding）。PA 场景下 `seqused_kv` 必填，用于确定每个 batch 实际 KV 长度。
- 这两个字段的意义：cu_seqlens 定位序列边界（TND），seqused 做长度截断（PA/变长）；两者同时传入时以实际较短者生效。

#### 1.5 Mask（mask_mode 与 attn_mask）

- **mask_mode** 属性选择掩码模式：0 = 全计算（无掩码）、3 = causal（下三角）、4 = window（band，前后各 win_left/win_right 个 token）。
- **attn_mask 输入**：INT8 模板矩阵，固定 shape (2048, 2048)，kernel 按 tile 从中扣取所需子块，适配任意 S1/S2。mask_mode=3/4 时必须传入 attn_mask；mask_mode=0 时禁止传入 attn_mask 且 win_left/win_right 必须为 -1。
- mask_mode=4（window）时 win_left/win_right 定义窗口宽度，kernel 据此跳过无效 tile。

#### 1.6 LSE（softmax_lse）

- **LSE（Log-Sum-Exp）**：`return_softmax_lse=True` 时输出每行 softmax 的 log 分母：lse = log(Σ e^{s-m}) + m，其中 m 为行内 max 值。
- 输出 shape：BSND/BNSD → (B, Q_N, Q_S)；TND → (Q_T, Q_N)；dtype 为 FLOAT32。
- 典型用途：跨 batch 的 KV cache 拼接（如 speculative decoding 的概率融合）、断点续训时的 softmax 状态重建。

#### 1.7 行无效（RowInvalid）

- **概念**：当某一行（一个 Q head 的一行 Q token）的所有 token 均被掩码排除（如 window 窗口外、seqused 截断、mask 全 1），该行没有任何有效参与计算的元素时，称为**无效行**。kernel 以该行 softmax max 是否仍为 `-inf`（softmaxMax == -inf，即没有任何有效 s 值）作为判定依据。
- **行无效时输出**：
  - **attn_out**：无效行输出**清零**（写 0）。`RowInvalidUpdateVF` 用 `Select` 将 max == -inf 的行整体写 0（`vf_flashupdate_new.h`），FD 阶段由 `InvalidRows`/`InvalidMaskRows` 清零后写出。
  - **softmax_lse**：无效行输出 **+inf**（3e+99）。代码不依赖数学推导（log(sum) + max 会得 -inf），而是显式处理：`ComputeLseOutputVF` 先算 `lse = log(sum) + max`，再用 `Select` 将 max == -inf 的行置为 `infValue = 3e+99`（`vf_flashupdate_new.h:522-523`）；FD 阶段 `ComputeScaleValue` 同样处理（`vf_flash_decode_arch35.h:310-311`）。

#### 1.8 Sink（learnable sink）

- **Sink**：在 softmax 分母中额外加入一项 `e^{sink - m}`（sink 同时参与 m 的 max 计算），用于提升部分模型（如 Qwen/GLM 系列）的数值稳定性。
- 本算子接口已预留 `sinks` 输入（FLOAT32，shape (Q_N,)），但**当前版本尚未支持**：传入会因 checker 拦截报错（"sinks is currently not supported"），需要等后续版本开放。

#### 1.9 FlashDecode

- **FlashDecode**：decode 阶段 S1 很小（如 1）、S2 很长时的优化——将 KV 方向 split-K 切分到多个 AIV 核并行计算，再跨核归约，避免单核串行扫描全部 KV。
- 分两个阶段：FA 阶段每个 split 独立算局部 softmax max/sum 与部分输出（写入 workspace）；FD 阶段跨 split 归约全局 max/sum、重算 scale、加权求和得最终输出。
- 自动启用（由 metadata 的 `enableFlashDecode` 决定），无需用户配置；`return_softmax_lse` 与 FlashDecode 可同时使用（LSE 从 FD 归约结果直接产出）。

### 2. 代码分层

算子从调用入口到执行引擎由四部分构成：**torch API 层**（用户 Python 入口）、**AICPU 侧**（配套 `flash_attn_metadata` 负载均衡）、**Host 侧**（主算子的接口/定义/tiling）、**Kernel 侧**（AIC+AIV 异构执行），共享 `attention/common/` 公共设施。

#### 2.1 torch API 层（用户入口）

| 文件 | 重点内容 |
|---|---|
| `torch_extension/flash_attn.py` | PyTorch 算子 schema（`flash_attn_metadata` + `flash_attn` 两个接口）、FakeTensor meta 实现（shape/dtype 推导）、`PrivateUse1` 分发的 NPU 实现；内含 metadata 尺寸公式 `((36+72)*B*KV_N+1)*16`（4096B 对齐）与 batch 推导 |
| `torch_extension/graph_convert_flash_attn.py` | torchair 图模式（graph mode）下发转换 |
| `torch_extension/csrc/` | C++ 绑定层，承载 `OpBuilder` 编译与底层调用 |

对外函数原型、参数/返回值说明及调用示例见接口文档 `flash_attn.md`（不在此重复）。

#### 2.2 AICPU 侧（flash_attn_metadata）

| 位置 | 重点内容 |
|---|---|
| `attention/flash_attn_metadata/op_kernel_aicpu/` | AICPU 算子实体：读 cu_seqlens/seqused/mask 等信息，调 SectionStreamK 生成任务切分 metadata |
| `attention/common/op_kernel/load_balance/section_stream_k/` | SectionStreamK 算法库（与 quant_flash_attn 共用）：按 96MB L2 预算切 section、成本模型 `6*ceil(m/16)+10*ceil(s2/64)`、FA 段（按核分配 bN2/gS1/s2 区间）与 FD 段（跨核 split-K 行分配） |
| 输入/输出 | 输入 shape/头数/mask 等参数；输出 metadata 张量（header + FA 段 + FD 段，由主算子 kernel 消费） |

AICPU 侧与 Host 侧 tiling 共享 `fa_adjust_sinner_souter.h` 的档位选择，保证两侧切分口径一致。

#### 2.3 Host 侧（3 层）

**① 接口层（op_api）**

| 文件 | 重点内容 |
|---|---|
| `op_api/aclnn_flash_attn.{h,cpp}` | 对外 `aclnnFlashAttn` 两段式接口（GetWorkspaceSize + 执行），薄壳转发 |
| `op_api/aclnn_flash_attn_inner.{h,cpp}` | inner 实现：对可选输入（sinks/softmax_lse）造占位 tensor |
| `op_api/flash_attn.{h,cpp}` | L0 API 层（`namespace l0op`）：注册 launcher-list，由框架据此自动生成 inner workspace 计算 |

**② 算子定义与 shape 推导**

| 文件 | 重点内容 |
|---|---|
| `op_host/flash_attn_def.cpp` | 算子定义：9 输入（q/k/v/block_table/cu_seqlens/seqused/sinks/attn_mask/metadata）+ 2 输出（attn_out/softmax_lse）+ 10 属性，含 dtype/格式/默认值与 ascend950 配置 |
| `op_host/flash_attn_infershape.cpp` | 输出 shape 推导：attn_out 按 layout；softmax_lse 按 (B, Q_N, Q_S) 或 TND 的 (Q_T, Q_N)；**不查值域**（留待 tiling 的 checker） |

**③ tiling 层**（重点：入口三层调用）

**入口**：`TilingFlashAttn`（`op_host/flash_attn_tiling.cpp:40-66`，经 `IMPL_OP_OPTILING(FlashAttn).Tiling(TilingFlashAttn)` 注册）。其函数体就是 **parser → check → doTiling 三层调用的骨架**：

```cpp
ge::graphStatus TilingFlashAttn(gert::TilingContext *context)
{
    ...
    FaTilingInfo faInfo;
    FaInfoParser faInfoParser(context);
    if (faInfoParser.Parse(faInfo) != ge::GRAPH_SUCCESS) {      // 第1层：parser
        return ge::GRAPH_FAILED;
    }
    FAChecker faChecker;
    faChecker.Init(faInfo);
    if (faChecker.Process(faInfo) != ge::GRAPH_SUCCESS) {       // 第2层：check
        return ge::GRAPH_FAILED;
    }
    return FiaTilingRegistry::GetInstance().DoTilingImpl(context, &faInfo);  // 第3层：doTiling
}
```

**第 1 层：parser（参数解析）** — `FaInfoParser::Parse`（`op_host/flash_attn_tiling_info_parser.cpp`）

```
GetOpParaInfo     取输入/属性（q/k/v/block_table/cu_seqlens/seqused/…）
布局解析           决定 BSND/BNSD/TND/PA 与 stride；GraphStatus 失败即报错
GetKvStorageMode   KV 存储模式（连续/PA_BBND/PA_BNBD/PA_NZ）
ParseAxisInfo      GetN1/N2 → GetQueryTSize → GetQkHeadDim/ValueHeadDim
                   → GetBSize → GetS1Size → GetS2Size → GetGSize
ParseFeatureInfo   mask/sinks/lse/maxSeq
产出：faInfo（统一信息结构体），后续 tiling 只读它
```

**第 2 层：check（合法性校验）** — `FAChecker::Process`（`op_host/checkers/fa_checker.cpp:85-108`）

```
对 faInfo 执行 9 组校验器 × 四轮：单参数 → 存在性 → 特性 → 组合
任一校验失败即返回报错，阻断后续 tiling
（base/common/mask/metadata/paged_attention/seq_len/sinks/softmax_lse 各自负责一组约束）
```

**第 3 层：doTiling（切分计算）** — `FiaTilingRegistry::DoTilingImpl`（common）→ `FlashAttnTilingImpl`（arch35）

```
FiaTilingRegistry::DoTilingImpl（common/op_host/fia_tiling_templates_registry.h:94-126）
  按 NpuArch 查表分发 → FlashAttnTilingImpl::DoOpTiling（arch35/flash_attn_tiling.cpp:51-69，DAV_3510，优先级1）
    SetPlatMemoryInfo   平台信息：aiv/aic 核数、CV_RATIO=2、UB/L0/L1 容量、sys workspace
    InitImplParam       cu_seqlens/seqused 存在性判断
    SplitPolicy         win处理 + AdjustSinnerAndSouter + CalcNumBlocks（coreNum）
    FillTiling          ComputeTilingData：tiling data 与 metadata 所需尺寸
    CalcWorkspaceSize   sys区 + D>128额外BMM2暂存 + PA对齐 + FD区（见§7.5）
    GenTilingKey        UpdateTilingKeyInfo → GET_TPL_TILING_KEY
产出：tiling data / workspace size / tiling key / 核数
```

**产出下发**（运行期，经 TilingContext 交给 kernel）：`SetBlockDim`（核数）、`SetTilingKey`（kernel 变体选择）、`SetWorkspaceSize`、`SetTilingData`（raw 结构）。

三层的职责边界：**parser 只做参数解析**（把 GE/Ascend 参数映射为算子内部 `faInfo`），**check 只做合法性校验**（不产出 tiling 结果，用检测非输入组合），**doTiling 只做切分计算**（信任前两层的结果）。三层先后严格有序、缺一不可——解析完成后、计算前由 check 拦截非法输入。

#### 2.4 Kernel 侧：入口 + 4 层

| 层 | 位置 | 重点内容 |
|---|---|---|
| ① 入口层 | `op_kernel/flash_attn.cpp` | 唯一的 `__global__` 入口：按 tiling key 路由 Dn/Nd（`EnableSoftmaxDn`），并用 `__DAV_C310_CUBE__` 宏区分 AIC/AIV 双编译（`KERNEL_TYPE_MIX_AIC_1_2`），同一份源码编译出 Cube 侧与 Vector 侧两个变体 |
| ② 调度框架层 | `arch35/flash_attn_kernel_dn.h` / `_nd.h` | 任务级流水框架：`Process()` 按 metadata section 循环 `FlashAttention`（(bN2, gS1, s2) 三重循环 + `CreateTask/ExecuteTask`，PRELOAD_N=2 预取）与 `FlashDecode`（AIV 做跨核归约，两端 `SyncAll`） |
| ③ 计算 block 层（AIC/AIV 分离） | `arch35/flash_attn_block_cube_dn.h / _nd.h`（AIC 侧）、`flash_attn_block_vec_dn.h / _nd.h`（AIV 侧）、`flash_attn_block_vec_flashdecode.h`（FD 归约） | AIC：BMM1/BMM2（L0A/L0B/L0C 多级 buffer、MTE2→MTE1→M→FIX 四级流水）；AIV：softmax VF（ProcessVec1）与 output 累加（ProcessVec2）、FD 用的跨 split 归约 |
| ④ 公共 API 层 | `attention/common/op_kernel/arch35/`（`flash_attention_score_common_regbase_arch35.h` 等）、`utils/`（`flash_attn_type.h`、`flash_attn_common_def.h`、`attenmask_gs1.h`） | 指令级 VF 算子库（`ProcessVec1VfDn`/`FusedExpSub`/`FlashUpdateNew` 等）与类型/布局/掩码工具；与 flash_attention_score 等算子共享 |

#### 2.5 层间关系

```
torch API 层（flash_attn.py，用户入口）
  → AICPU 侧（flash_attn_metadata：SectionStreamK 生成 metadata）
  → Host 侧：接口层（aclnn/L0）
      → def/infershape（定义与 shape）
      → tiling 层（parser → check → doTiling）
          ↓ TilingContext 下发：tiling data / workspace size / tiling key / 核数
      → Kernel 侧：flash_attn.cpp 入口 → kernel_dn/nd 调度框架 → block_cube + block_vec
          → attention/common VF 指令库（指令级）
```

关键点：kernel 不感知全局 shape，只按 **metadata 任务区间**执行；编译期由 **tiling key**（模板参数）决定变体，运行期由 **tiling data** 决定切分，两者分别来自 doTiling 的编译清单与切分计算。

### 3. 基本块

FA 计算的核心是把注意力矩阵按块切分、逐块 online 计算。基本块参数由编译期 config 与运行时 shape 共同决定。

**config 映射**（编译期模板参数之一，见 §5 TilingKey）：

| config | sOuter | sInner | D | DV |
|---|---|---|---|---|
| 0 | 64 | 128 | 64 | 64 |
| 1 | 32 | 256 | 64 | 64 |
| 2 | 64 | 128 | 128 | 128 |
| 3 | 32 | 256 | 128 | 128 |
| 4 | 32 | 256 | 256 | 256 |

- **sOuter**：M 方向（Q 序列）每核每次迭代的块行数。
- **sInner**：N 方向（KV 序列）的块大小，即 s2BaseSize。

**运行时基本块参数**：

- **mBaseSize = sOuter × CV_RATIO**（CV_RATIO=2，AIC:AIV=1:2）：一个 AIC 核承担 mBaseSize 行 Q，对应 2 个 AIV 核各处理 mBaseSize/2 行。
- **s2BaseSize = sInner**：KV 序列方向的块大小。
- host 侧 `AdjustSinnerAndSouter` 按 D（及 maxSeq/window 条件）选择 sOuter/sInner，再映射到 config（D=64→config0/1，D=128→config2/3，D=256→config4）。

**任务循环结构**（kernel 内三重循环，详见 §7.1）：

```
for bN2 in [bN2Start, bN2End):        # batch × KV head 维度
    for gS1 in [gS1Start, gS1End):    # Q 序列 × GQA group 维度
        for s2 in [s2Start, s2End):   # KV 序列维度
            CreateTask / ExecuteTask  # BMM1 + Softmax + BMM2
```

每个 tile 做一次 BMM1(Q×K^T) → Softmax → BMM2(attn×V)，跨 s2 份在线累加 softmax 与输出。

### 4. 负载均衡（SectionStreamK）

**算法位置**：`attention/common/op_kernel/load_balance/section_stream_k/`，由 `flash_attn_metadata`（AICPU）调用，输出 FA段 + FD段 任务切分 metadata。

**SectionStreamK 切分要点**：

- **section 划分**：按 L2 预算（96MB token 成本）逐 batch×head 累加切 section；`maxGS1Size ≤ mBaseSize` 或单 head 成本 ≤ 预算/AIC数 时不切。
- **成本模型**（非 token 加权）：块成本 = `6*ceil(m/16) + 10*ceil(s2/64)`。
- **每 section 核数**：`minCore = sqrt(块数)` 下界启发，maxCore = min(36, 块数)。
- **FA 任务分配**：三档贪心 —— 整 batch → 整行（gS1G）→ 行内按 s2 块，逐核按 `成本/剩余核数` 配额分配。
- **FD 任务分配**（跨核行）：行被多个核切分时记 FD 任务，`s2SplitNum` = 切分段数，按 `Σ s2Split×m` 平摊到 ≤72 个 AIV。
- **FD 启用判定**（`CheckChooseWithFd`）：sectionNum>1 必用 FD；否则无 FD 方案收益优于 10 块成本时不用 FD。

**metadata 输出**：`16 + sectionNum×(36 AIC + 72 AIV)×16` 个 uint32（header + FA 段 + FD 段，见 §7.5）。

**对 NumBlock（核数）设置的影响**：

- host tiling 中 `CalcNumBlocks` 决定 `SetBlockDim`（NumBlock）：`numBlocks_ = CalcTschBlockDim(aivNum, aicNum, aivNum)`，aivNum = aicNum × CV_RATIO（950 上 aic=36 → aiv=72 → NumBlock=36）。
- metadata 的切分基于平台的 aicCoreNum/aivCoreNum（36/72）：FA 段按 AIC 核数分配，FD 段按 AIV 核数分配。
- **两者必须一致**：NumBlock 决定启动多少个 AIC 核，metadata 决定每个核分到哪些任务；NumBlock 小于 metadata 需要的核数时部分任务无核执行（丢任务/结果错误），大于时多余核空转。二者均取自同一平台核数接口，默认一致；人为调整 NumBlock 时必须保证 ≥ metadata 实际分配核数。
- kernel 侧有兜底：`aicIdx >= coreNum` 的核直接 return（核号越界保护）。

### 5. TilingKey

**作用**：编译期决定生成哪些 kernel 变体，运行期选择变体。4 个模板参数按 8+8+1+3 位从 bit0 起拼成 tiling key（`flash_attn_template_tiling_key.h`）：

| bits | 参数 | 取值 |
|---|---|---|
| 0-7 | InOutLayoutType | 0=BSND, 1=BNSD, 2=TND, 3=BNSD_BSND |
| 8-15 | KvLayoutType | 0=连续, 1=PA_BBND, 2=PA_BNBD, 3=PA_NZ |
| 16 | HasAttenMask | false/true |
| 17-19 | Config | 0~4（sOuter×sInner×D×DV 组合，见 §3 基本块） |

**生成**：host tiling 的 `GenTilingKey` 一步 —— `UpdateTilingKeyInfo`（按布局/mask/config 填 tilingKeyInfo）→ `GET_TPL_TILING_KEY`（`arch35/flash_attn_tiling.cpp:193-203`）。

**下发与选择**：tiling 期 `SetTilingKey` 写入 TilingContext；运行时 kernel 按 key 选择变体执行。编译期可用 `--tiling_key` 指定只生成部分变体（见 Quick Start 编译）。

**示例解码**：tiling key `2279866368` = 0x87E40000 → InOutLayoutType=0(BSND)、KvLayout=0(连续)、无mask、config=2(D=128, sOuter=64, sInner=128)。

**与 Dn/Nd 的关系**：`useDn = !hasAttenMask && (config==0||config==2)`，见 §6。

### 6. Kernel 模板范围与 Dn/Nd 路由

kernel 按 tiling key 的 4 个模板参数编译出不同变体；运行时入口 `op_kernel/flash_attn.cpp` 同时编译 AIC 与 AIV（`KERNEL_TYPE_MIX_AIC_1_2`，`__DAV_C310_CUBE__` 区分 Cube 编译/Vector 编译），每个 AIC 配 2 个 AIV（CV_RATIO=2）。

Dn/Nd 路由（`EnableSoftmaxDn`）：

```cpp
useDn = !hasAttenMask && (config == 0 || config == 2)
```

| 路径 | 条件 | 调度框架 | softmax VF | 适用 |
|---|---|---|---|---|
| Dn | 无mask 且 config∈{0,2} | kernel_dn.h | ProcessVec1VfDn（无mask专用优化） | 无mask、D≤128 |
| Nd | 其余（有mask 或 config∈{1,3,4}） | kernel_nd.h | ProcessVec1Vf（按 actS2 四档通用） | 有mask或 D=256 |

加新模板组合（如新 config、新布局）的改动点：`flash_attn_template_tiling_key.h`（参数声明）→ host `UpdateTilingKeyConfig`（映射）→ `flash_attn.cpp` 的路由与 kernel 模板实例化处。

### 7. 核心代码路径

#### 7.1 任务调度（kernel_dn.h / kernel_nd.h）

```text
Process(): for sectionIdx in [0, sectionNum)
  ├─ FlashAttention: 三重循环 (bN2, gS1, s2)，按 metadata 指定的
  │                  (bN2Start, gS1Start, s2Start)~(bN2End, gS1End, s2End) 执行
  │                  CreateTask → ExecuteTask（PRELOAD_N=2 流水缓冲）
  └─ FlashDecode（仅AIV，两侧 SyncAll 包裹）
```

- bN2 = batch × KV head 扁平索引；gS1 = Q序列×GQA group 分块；s2 = KV 序列分块。
- `ExecuteTask`：本轮执行 BMM1(Vec1)，同时执行 `PRELOAD_N=2` 轮前的 BMM2(Vec2)；`PRELOAD_TASK_CACHE_SIZE=4` 用 `loop & 3` 代替取模。

#### 7.2 AIC 侧（block_cube_dn.h / _nd.h）

- BMM1：Q×K^T → L0C（Dn 时输出 S2×M 转置布局，dualDstCtl=2 双目标写 UB）；Nd 时 L0C=M×S2，D=256 走 `MatmulN`（256 拆 2×128）。
- BMM2：attn×V，DV>128 走 `MatmulN`，否则 `MatmulFull/MatmulBase`。
- 四级流水：PIPE_MTE2（GM→L1）→ PIPE_MTE1（L1→L0A/L0B）→ PIPE_M（Matmul）→ PIPE_FIX（L0C→UB），经 `Mutex::Lock/Unlock` 级联（每 buffer 独立 eventID，多 buffer 轮转）。
- buffer：L1 上 Q 2×、P（softmax结果）3×、KV 4×64KB 共享（Nd 的 D=256 用 2×128KB 大缓冲）；L0A/L0B 2×32KB，L0C 4×64KB。
- Q 仅在 `isFirstS2Loop` 加载、`isLastS2Loop` 释放（一个 gS1 行内多个 s2 循环复用 Q）。

#### 7.3 AIV 侧（block_vec_dn.h / _nd.h / flashdecode）

- **ProcessVec1（softmax）**：Cast(f16→f32) → 4路Max→二级归约→Muls(scale) → `FusedExpSub` → 累加 sum → Cast 回 f16 → DataCopy 写 L1(P) 供 BMM2；mask 场景用 `Select` 置 -inf。软掩码流程见 utils/attenmask_gs1.h（MergeMask/ComputeAttenMaskOffset）。
- **ProcessVec2（output 累加）**：非末轮 `FlashUpdateNew`（expUb*pre + cur）；末轮 `FlashUpdateLastNew` + `LastDivNew`（除以 sum）；`isFirstS2Loop` 时用 `DataCopy` 跳过 `expUb*pre`（避免脏值泄漏）。
- **FlashDecode（block_vec_flashdecode.h）**：`ComputeScaleValue` 跨 split-K 归约全局 max/sum 并重算 scale → `ReduceFinalRes` 加权求和；核内按 `fdBalanceMBaseSize=8` 细分 M 方向。

#### 7.4 CV 核间同步

| 同步类型 | 机制 |
|---|---|
| AIC↔AIV 数据传递 | `CrossCoreSetFlag/WaitFlag`：CC_BMM1_0/1=0/1（UB 双buffer）、CC_BMM2_0/1=2/3、CC_L1P_0~2=5/6/7（flag 4 空置） |
| AIV0↔AIV1 独立 | flag + `AIV0_AIV1_OFFSET=16` |
| FD 阶段 | `SyncAll()`：FA 全完成后再 FD，FD 全完成后再下一 section |
| 首次 WaitFlag 不阻塞 | AIV 初始化时预 set 4 个 BMM flag |

#### 7.5 metadata 与 workspace

- **metadata 生成**（flash_attn_metadata AICPU，算法在 `common/op_kernel/load_balance/section_stream_k/`）：按 L2 预算（96MB token 成本）切 section；成本模型 `6*ceil(m/16)+10*ceil(s2/64)` 非 token 加权；核数下界 `sqrt(块数)`；FA 三档贪心分配（batch→行→块），FD 按 `Σ s2Split×m` 平摊到 72 AIV。是否启用 FD 由 `CheckChooseWithFd` 决策（sectionNum>1 或收益>10 块）。输出 = `16 + sectionNum*(36个AIC+72个AIV)*16` 个 uint32。
- **workspace 布局**（kernel 侧 `block_vec_dn.h` 三段）：`[0, accumOutSize)` = accumOut（FA 中间输出）、`+logSumExpSize` = LSE sum、其后 = LSE max；尺寸计算在 arch35 tiling 的 `CalcWorkspaceSize`（sys区 + D>128 暂存 + PA cacheline 对齐 + FD 区）。

### 8. 给贡献者的建议

- **从哪读起**：`flash_attn_def.cpp` → `checkers/` → `arch35/flash_attn_tiling.cpp` → `op_kernel/flash_attn.cpp` → `kernel_dn.h`/`kernel_nd.h`。
- **改接口**：def.cpp 增删参数 → 同步 checkers → infershape → 接口文档（torch_extension 侧）。
- **改切分策略**：`fa_adjust_sinner_souter.h`（主算子与 metadata 共用，必须保持一致）+ `common/op_kernel/load_balance/section_stream_k/`。
- **改 kernel 计算**：先在 `flash_attn_template_tiling_key.h` 确认模板范围，AIC 侧重 `block_cube_*`，AIV 侧重 `block_vec_*`；共享 VF API 在 `common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h`。
- **新增测试**：tiling 用例加进 `tests/ut/op_host/arch35/test_flash_attn_tiling.csv`；e2e 用例加进 `tests/pytests/test_cases/`。
- **注意事项**：kernel 侧无独立单测（依赖 pytests 真机验证）；tiling CSV 中 TND/BF16 部分正例当前标记 FAIL（按实际结果反填），新增用例时以 `expectResult` 表述期望。

## 参考资源

- 接口文档：`../../torch_extension/cann_ops_transformer/docs/zh/flash_attn.md`
- e2e 测试框架：`tests/pytests/readme.md`
- SectionStreamK 算法：`../common/op_kernel/load_balance/section_stream_k/`
- 共享 VF 公共 API：`../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h`
