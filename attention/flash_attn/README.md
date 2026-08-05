# flash_attn 算子技术报告

> 仓库: `ops-transformer` (`attention/flash_attn/`)
> 硬件: Ascend 950 (arch35)
> 算子类型: 非量化 GQA Flash Attention, 训练推理归一化
> 配套算子: `flash_attn_metadata` (AICPU 负载均衡元数据生成)

---

## 一、公式与原理

### 1.1 标准 Flash Attention

自注意力的核心计算:

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V
$$

其中 $Q \in \mathbb{R}^{S_1 \times D}$, $K, V \in \mathbb{R}^{S_2 \times D}$, $d$ 为 head dim, $\sqrt{d}$ 为缩放因子。

flash_attn 采用 **Flash Attention** 算法, 将 $S_1 \times S_2$ 的注意力矩阵分块 (tiling) 计算, 避免实例化完整 $S_1 \times S_2$ 矩阵, 将显存复杂度从 $O(S_1 \cdot S_2)$ 降至 $O(\text{tile}_M \cdot \text{tile}_N)$。

### 1.2 分块 Softmax 数学推导

标准 Softmax 对每行 $i$:

$$
o_i = \frac{\sum_{j} e^{s_{ij}} \cdot v_j}{\sum_{j} e^{s_{ij}}}
$$

分块计算时, 将 $S_2$ 维度切成多份, 每份计算局部 max 和 sum, 再跨份归约:

**局部计算 (第 $k$ 份)**:
$$
m_i^{(k)} = \max(s_{i, j \in \text{tile}_k})
$$
$$
\ell_i^{(k)} = \sum_{j \in \text{tile}_k} e^{s_{ij} - m_i^{(k)}}
$$
$$
o_i^{(k)} = \sum_{j \in \text{tile}_k} e^{s_{ij} - m_i^{(k)}} \cdot v_j
$$

**跨份归约**:
$$
m_i = \max_k(m_i^{(k)})
$$
$$
\ell_i = \sum_k e^{m_i^{(k)} - m_i} \cdot \ell_i^{(k)}
$$
$$
o_i = \frac{\sum_k e^{m_i^{(k)} - m_i} \cdot o_i^{(k)}}{\ell_i}
$$

kernel 中 `softmaxMax` 对应 $m_i$, `softmaxSum` 对应 $\ell_i$, `FlashUpdateNew` 对应增量 $o_i^{(k)}$ 累加, `LastDivNew` 对应最终除以 $\ell_i$。

### 1.3 Learnable Sink

部分模型 (如 Qwen/GLM 系列) 在 softmax 分母中引入 learnable sink 改善数值稳定性:

$$
m = \max(\text{sink}, \max(S))
$$
$$
\text{Attention} = \frac{e^{S - m} \cdot V}{\sum e^{S - m} + e^{\text{sink} - m}}
$$

sink 参与 max 计算 (使 max 不会过小) 并加到分母 (防止分母为 0)。每个 head 一个 sink 标量值, shape 为 `(Q_N,)`。

> 当前 sinks 输入参数已声明但 checker 硬编码拦截, kernel 计算逻辑已实现但未打通。打通方案详见第八节。

### 1.4 Softmax LSE (Log-Sum-Exp)

开启 `return_softmax_lse` 时输出:

$$
\text{softmax\_lse} = \log\left(\sum_j e^{s_{ij} - m_i}\right) + m_i
$$

用于推理时跨 batch 的 KV cache 拼接 (如 speculative decoding 的概率融合)。

### 1.5 GQA (Grouped Query Attention)

Q 的 head 数 $Q_N$ 是 KV head 数 $KV_N$ 的整数倍, $g = Q_N / KV_N$。同一组 $g$ 个 Q head 共享一份 KV, 减少 KV 的显存和计算量。

kernel 中 `gSize = g`, Q 按 `gS1 = actSeqLenQ × gSize` 维度分块, 每个 bN2 (batch × KV head) 内循环 $g$ 个 Q head。

### 1.6 Paged Attention

KV cache 按 block (block_size) 分页存储, 通过 `block_table` 索引物理 block。支持三种布局:
- **PA_BBND**: `(num_blocks, block_size, KV_N, D)` — block 内 BSND
- **PA_BNBD**: `(num_blocks, KV_N, block_size, D)` — block 内 BNSD
- **PA_NZ**: `(num_blocks, KV_N, D//16, block_size, 16)` — NZ 格式

### 1.7 FlashDecode

推理 decode 阶段 ($S_1$ 很小, 如 1, $S_2$ 很长), 将 KV 方向 split-K 切分到多个 AIV 核并行计算, 再跨核归约:

1. **FA 阶段**: 每个 split 份独立计算局部 softmax max/sum 和 accumOut, 写入 workspace
2. **FD 阶段**: 跨 split 份归约全局 max/sum, 重算 scale, 加权求和得到最终 output

---

## 二、目录结构

```
attention/flash_attn/
├── CMakeLists.txt
├── op_api/                          # ACLNN 接口层
│   ├── aclnn_flash_attn.cpp         # ACLNN 入口
│   ├── aclnn_flash_attn_inner.cpp   # ACLNN 内部实现
│   └── flash_attn.cpp               # Python 扩展入口
├── op_graph/
│   └── flash_attn_proto.h           # 算子 proto 定义
├── op_host/                         # Host 侧 (tiling/checker/infershape)
│   ├── arch35/
│   │   └── flash_attn_tiling.cpp    # arch35 tiling 实现
│   ├── checkers/                    # 参数校验器
│   │   ├── base_checker.{h,cpp}
│   │   ├── common_checker.{h,cpp}
│   │   ├── fa_checker.{h,cpp}       # checker 调度入口
│   │   ├── mask_checker.{h,cpp}
│   │   ├── metadata_checker.{h,cpp}
│   │   ├── paged_attention_checker.{h,cpp}
│   │   ├── seq_len_checker.{h,cpp}
│   │   ├── sinks_checker.{h,cpp}    # sinks 校验 (当前拦截)
│   │   └── softmax_lse_checker.{h,cpp}
│   ├── flash_attn_def.cpp           # 算子定义 (Input/Output/Attr)
│   ├── flash_attn_infershape.cpp    # shape 推导
│   ├── flash_attn_tiling.cpp        # 通用 tiling 入口
│   ├── fa_tiling_info.{h,cpp}       # tiling 信息结构
│   ├── fa_tiling_shape.{h,cpp}      # shape 计算
│   ├── fa_adjust_sinner_souter.h    # sInner/sOuter 档位选择
│   └── flash_attn_tiling_info_parser.{h,cpp}  # 参数解析
├── op_kernel/                       # Kernel 侧
│   ├── flash_attn.cpp               # __global__ 入口 (Dn/Nd 路由)
│   ├── arch35/
│   │   ├── flash_attn_kernel_dn.h   # Dn 路径 kernel (无 mask, D≤128)
│   │   ├── flash_attn_kernel_nd.h   # Nd 路径 kernel (有 mask 或 D=256)
│   │   ├── flash_attn_block_cube_dn.h  # Dn cube block (BMM1/BMM2)
│   │   ├── flash_attn_block_cube_nd.h  # Nd cube block
│   │   ├── flash_attn_block_vec_dn.h   # Dn vec block (softmax/BMM2 update)
│   │   ├── flash_attn_block_vec_nd.h   # Nd vec block
│   │   ├── flash_attn_block_vec_flashdecode.h  # FlashDecode 归约 block
│   │   ├── flash_attn_template_tiling_key.h    # 编译期 tiling key 定义
│   │   ├── flash_attn_tiling_data.h            # tiling data 结构
│   │   └── memory_copy_arch35.h                # LSE 拷出辅助函数
│   └── utils/
│       ├── flash_attn_type.h        # FAType/SeqLensTool/布局映射
│       ├── flash_attn_common_def.h  # ConstInfo_t/枚举定义
│       ├── flash_attn_utils.h       # 工具函数
│       └── attenmask_gs1.h          # mask 处理
└── tests/
    └── ut/op_host/
        ├── arch35/test_flash_attn_tiling.cpp   # tiling 单测
        ├── arch35/test_flash_attn_tiling.csv   # 测试用例
        ├── test_flash_attn_dtype_infershape.cpp
        └── test_flash_attn_shape_infershape.cpp

attention/flash_attn_metadata/       # 配套 metadata 算子
├── op_host/op_api/                  # ACLNN 接口
├── op_kernel_aicpu/                 # AICPU kernel (负载均衡计算)
│   ├── flash_attn_metadata_aicpu.cpp
│   ├── flash_attn_metadata_aicpu.h
│   └── flash_attn_metadata.h
└── examples/test_aclnn_flash_attn_metadata.cpp
```

---

## 三、基本块切分

### 3.1 编译期模板参数

`flash_attn_template_tiling_key.h` 定义 4 个编译期模板参数, 编译时生成不同 tiling key 的 kernel 变体:

| 参数 | 位宽 | 取值 | 含义 |
|---|---|---|---|
| `InOutLayoutType` | 8-bit | 0=BSND, 1=BNSD, 2=TND, 3=BNSD_BSND | Q/Output 布局 |
| `KvLayoutType` | 8-bit | 0=连续, 1=PA_BBND, 2=PA_BNBD, 3=PA_NZ | KV 布局 |
| `HasAttenMask` | 1-bit | false / true | 是否有 mask |
| `Config` | 3-bit | 0~4 | sOuter×sInner×D×DV 组合 |

`Config` 映射:

```
config=0: sOuter=64,  sInner=128, D=64,  DV=64
config=1: sOuter=32,  sInner=256, D=64,  DV=64
config=2: sOuter=64,  sInner=128, D=128, DV=128
config=3: sOuter=32,  sInner=256, D=128, DV=128
config=4: sOuter=32,  sInner=256, D=256, DV=256
```

host 侧 `AdjustSinnerAndSouter` 根据 D (head_dim) 选择 sOuter/sInner, 再映射到 config。

### 3.2 运行时 tile 参数

- **mBaseSize** = sOuter × CV_RATIO (CV_RATIO=2, AIC:AIV=1:2)。每个 AIC 核处理 mBaseSize 行 Q, 对应 2 个 AIV 核各处理 mBaseSize/2 行。
- **s2BaseSize** = sInner, KV 序列方向的 tile 大小。

### 3.3 任务循环结构

kernel 内三重循环遍历 `(bN2, gS1, s2)`:

```
for bN2 in [bN2Start, bN2End):        # batch × KV head 维度
    for gS1 in [gS1Start, gS1End):    # Q 序列 × GQA group 维度
        for s2 in [s2Start, s2End):   # KV 序列维度
            CreateTask / ExecuteTask
```

- `bN2`: batch × n2 (n2 = KV head 数) 的扁平索引
- `gS1`: Q 序列方向的 g×s1 分块索引 (g = Q_N / KV_N, GQA group size)
- `s2`: KV 序列方向的分块索引

每个 tile 计算一次 BMM1 (Q×K^T) + Softmax + BMM2 (attn×V), 跨 s2 份累加 softmax 和 output。

---

## 四、分核算法

### 4.1 两级分核架构

#### 第一级: metadata AICPU 负载均衡

`flash_attn_metadata` 算子在 AICPU 上运行, 使用 `SectionStreamK` 算法进行负载均衡:

```
输入: batch, seqLen, headNum, headDim, layout, maskMode, aicCoreNum, aivCoreNum
算法: SectionStreamK::Compute (Stream-K 风格任务切分)
输出: metadata (FA段 + FD段)
```

metadata 格式 (`flash_attn_tiling_data.h`):

```
Header: sectionNum, mBaseSize, s2BaseSize, enableFlashDecode
FA段 [section][aicCore][16 fields]:
    bN2Start, mStart, s2Start,    ← 任务起始
    bN2End, mEnd, s2End,          ← 任务结束
    firstFdDataWorkspaceIdx       ← FD workspace 偏移
FD段 [section][aivCore][16 fields]:
    bN2Idx, mIdx,                 ← 处理哪个 batch×head, Q 位置
    workspaceIdx, s2SplitNum,     ← workspace 偏移, KV split 份数
    mStart, mLen                  ← M 方向起止
```

#### 第二级: kernel 按 metadata 执行

```cpp
// kernel_dn.h:Process()
for (sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
    FlashAttention(sectionIdx);  // AIC+AIV 协同计算
    FlashDecode(sectionIdx);     // AIV FD 归约
}
```

每个 AIC 核从 metadata 读取自己的 `(bN2Start, gS1Start, s2Start, bN2End, gS1End, s2End)`, 只处理分配的任务范围。

### 4.2 Dn vs Nd 路径选择

```cpp
useDn = EnableSoftmaxDn<hasAttenMask, config>()
     = !hasAttenMask && (config == 0 || config == 2)
```

| 条件 | 路径 | softmax VF 指令 | 适用场景 |
|---|---|---|---|
| 无 mask 且 D≤128 | Dn | `ProcessVec1VfDn` (专用优化) | 性能更优 |
| 有 mask 或 D=256 | Nd | `ProcessVec1VfNd` (通用) | 功能完整 |

### 4.3 FlashDecode 分核

FD 阶段由 AIV 核执行, 每个 AIV 核从 metadata FD 段读取:
- `fdBN2Idx`: 处理哪个 batch × head
- `fdMIdx`: Q 方向起始位置
- `fdWorkspaceIdx`: workspace 偏移 (指向 FA 阶段输出的中间结果)
- `fdS2SplitNum`: KV 方向 split-K 份数
- `mLen`: 该核处理的 M 行数

FD 核内再按 `fdBalanceMBaseSize=8` 细分 M 方向。

---

## 五、CV 并行流水

### 5.1 Mix Kernel 架构

```cpp
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);  // 1 AIC + 2 AIV
```

CV_RATIO = 2 (AIC:AIV = 1:2)。编译时通过 `__DAV_C310_CUBE__` 宏分离 AIC/AIV 代码:

```cpp
#ifdef __DAV_C310_CUBE__
    // AIC 编译: 只有 Cube block, Vec block 为 Dummy
    using Kernel = ...KernelDn<FA_T, CubeBlock, VecDummy, VecDummy>;
#else
    // AIV 编译: 只有 Vec block, Cube block 为 Dummy
    using Kernel = ...KernelDn<FA_T, CubeDummy, VecFaBlock, VecFdBlock>;
#endif
```

### 5.2 PRELOAD_N=2 三级流水

`ExecuteTask` 实现预取流水:

```cpp
void ExecuteTask(loop, taskRunInfo) {
    // 本轮任务: BMM1 (AIC) + Vec1 (AIV)
    RunInfoX &runInfo0 = taskRunInfo[loop & MASK];
    ComputeMm1(runInfo0);    // AIC: Q×K^T → L0C → UB
    ComputeVec1(runInfo0);   // AIV: softmax + cast → L1

    // PRELOAD_N 轮前的任务: BMM2 (AIC) + Vec2 (AIV)
    RunInfoX &runInfoNegN = taskRunInfo[(loop - PRELOAD_N) & MASK];
    ComputeMm2(runInfoNegN);  // AIC: attn×V → L0C → UB
    ComputeVec2(runInfoNegN); // AIV: FlashUpdate + copyout
}
```

流水时序图:

```
时间 →    t0          t1          t2          t3          t4
AIC:   BMM1(task0)  BMM1(task1)  BMM1(task2)  BMM2(task0)  BMM2(task1)
AIV:   Vec1(task0)  Vec1(task1)  Vec1(task2)  Vec2(task0)  Vec2(task1)
                                   ↑
                           PRELOAD_N=2: task0 的 BMM2 延后 2 轮
```

`PRELOAD_TASK_CACHE_SIZE=4` (向上取整到 2 的幂), 实际预取 2 轮, 用位掩码 `loop & 3` 替代取模。

### 5.3 CV 核间数据传递

AIC 和 AIV 通过共享 UB/L1 传递数据, 用 `CrossCoreSetFlag/WaitFlag` 同步:

```
BMM1 数据流:
  AIC: IterateBmm1 → Fixpipe 到 UB[bufId]
       SetFlag(CC_BMM1_0 + bufId)              ← 通知 AIV UB 可读
  AIV: WaitFlag(CC_BMM1_0 + bufId)             ← 等 AIC 写完
       ProcessVec1 (读 UB 做 softmax)
       SetFlag(CC_BMM1_0 + bufId)              ← 通知 AIC UB 可覆写

BMM2 数据流:
  AIV: ProcessVec1 → cast 结果写 L1[bufId]
       SetFlag(CC_L1P_0 + bufId)               ← 通知 AIC L1 可读
  AIC: WaitFlag(CC_L1P_0 + bufId)              ← 等 AIV 写完
       IterateBmm2 (读 L1 做 attn×V)
       Fixpipe 到 UB[bufId]
       SetFlag(CC_BMM2_0 + bufId)              ← 通知 AIV UB 可读
  AIV: WaitFlag(CC_BMM2_0 + bufId)             ← 等 AIC 写完
       ProcessVec2 (读 UB 做 FlashUpdate + copyout)
```

`AIV0_AIV1_OFFSET=16`: 同一 AIC 对应的 2 个 AIV 核各自有独立 flag (`flag + 0` 和 `flag + 16`), 实现双 AIV 并行。

---

## 六、核内流水与同步

### 6.1 Cube 侧 (AIC) 多级 Buffer

| 存储层 | Buffer | 数量 | 大小 | 用途 |
|---|---|---|---|---|
| L1 | l1PBuffers_ | 3 | mBaseSize×s2BaseSize×sizeof(T) | BMM2 的 P 矩阵 (softmax 结果) |
| L1 | l1QBuffers_ | 2 | mBaseSize×dBaseSize×sizeof(T) | Q 矩阵 (isFirstS2Loop 加载, isLastS2Loop 释放) |
| L1 | l1KvBuffers_ | 4 | 64KB | K/V 矩阵 (BMM1 用 K, BMM2 用 V, 共享 4 buffer) |
| L0A | BufferManager | 2 | 32KB | matmul A 矩阵 |
| L0B | BufferManager | 2 | 32KB | matmul B 矩阵 |
| L0C | l0CBuffers_ | 4 | 64KB | matmul 结果 |
| UB | ubMm1ResBuffers_ | 2 | mBaseSize/2×s2BaseSize×4 | BMM1 fixpipe 结果 |
| UB | ubMm2ResBuffers_ | 2 | mBaseSize/2×dVBaseSize×4 | BMM2 fixpipe 结果 |

### 6.2 Cube 侧四级流水

以 BMM1 为例 (`IterateBmm1Dn`):

```
PIPE_MTE2: GM → L1 (CopyKeySlice)         ← K 数据搬入 L1
PIPE_MTE1: L1 → L0A/L0B (LoadDataToL0)    ← K→L0A, Q→L0B
PIPE_M:    Matmul (L0A × L0B → L0C)        ← Q×K^T
PIPE_FIX:  Fixpipe (L0C → UB)              ← 结果搬出到 UB, 供 AIV 读取
```

通过 `Mutex::Lock/Unlock` 管理各级 pipe 间依赖:

```cpp
Mutex::Lock<PIPE_MTE2>(KV_L1_EVENT0 + kvBufId_);    // MTE2 写 L1
CopyKeySlice(kL1Tensor, ...);
Mutex::Unlock<PIPE_MTE2>(KV_L1_EVENT0 + kvBufId_);

Mutex::Lock<PIPE_MTE1>(KV_L1_EVENT0 + kvBufId_);    // MTE1 等 MTE2 完成
// LoadDataToL0A / LoadDataToL0B ...

Mutex::Lock<PIPE_M>(L0C_EVENT0 + l0cBufId_);        // M 等 MTE1 完成
MatmulBase<K, Q, float, ...>(kL1Tensor, qL1Tensor, ...);
Mutex::Unlock<PIPE_M>(L0C_EVENT0 + l0cBufId_);

Mutex::Lock<PIPE_FIX>(L0C_EVENT0 + l0cBufId_);      // FIX 等 M 完成
FixpipeMm1Dn(mm1ResUbTensor, l0CSubTensor, ...);
Mutex::Unlock<PIPE_FIX>(L0C_EVENT0 + l0cBufId_);
```

每个 buffer 有独立 eventID (如 `KV_L1_EVENT0~3` 对应 4 个 KV buffer), 实现多 buffer 并行。

Q buffer 特殊处理: `isFirstS2Loop` 时加载 Q (Q 在一个 gS1 行的多个 s2 循环中复用), `isLastS2Loop` 时释放。

### 6.3 Vector 侧 (AIV) 多级 Buffer

| Buffer | 数量 | 大小 | 用途 |
|---|---|---|---|
| ubMm1ResBuffers_ | 2 | 32KB | BMM1 结果 (从 AIC fixpipe 来) |
| ubMm2ResBuffers_ | 2 | 32KB | BMM2 结果 (从 AIC fixpipe 来) |
| ubVec1ResBuffers_ | 2 | 32KB | Vec1 cast 结果 (写 L1 供 BMM2) |
| ubVec2Res_ | 1 | 32KB | Vec2 最终输出 buffer |
| softmaxSumBuf_ | 3 | 256B | softmax 分母累加 (per mloop) |
| softmaxMaxBuf_ | 3 | 256B | softmax max 累加 (per mloop) |
| softmaxExpBuf_ | 3 | 256B | softmax exp 中间结果 (per loop) |
| ubLseOutBuffers_ | 2 | 2KB | FD 中间结果 (sum/max) 拷出 GM |

softmaxSum/Max 用 3 buffer 对应 `mloop % 3` — 每个 gS1 行独立维护一组 sum/max。

### 6.4 同步机制汇总

| 同步类型 | 机制 | 用途 |
|---|---|---|
| 核间 (AIC↔AIV) | `CrossCoreSetFlag/WaitFlag` | UB/L1 数据传递同步 |
| 核内 (pipe 间) | `Mutex::Lock/Unlock` | MTE2→MTE1→M→FIX 级联依赖 |
| AIV0↔AIV1 | flag + `AIV0_AIV1_OFFSET(16)` | 双 AIV 核独立同步 |
| FD 阶段 | `SyncAll()` | 全 AIV 同步, 等 FA 完成/等 FD 完成 |
| L1 buffer | `CC_L1P_0~2` (3 个 flag) | P 矩阵三 buffer 与 PRELOAD_N=2 配合 |

核间 flag ID 分配:

```
CC_BMM1_0 = 0, CC_BMM1_1 = 1    ← BMM1 UB 双 buffer
CC_BMM2_0 = 2, CC_BMM2_1 = 3    ← BMM2 UB 双 buffer
CC_L1P_0 = 5, CC_L1P_1 = 6, CC_L1P_2 = 7  ← L1 P 三 buffer
```

AIV 侧初始化时预 set BMM flag (让 AIC 第一次 WaitFlag 不阻塞):

```cpp
void InitCrossCoreSync() {
    CrossCoreSetFlag<..., PIPE_V>(CC_BMM2_0);
    CrossCoreSetFlag<..., PIPE_V>(CC_BMM2_1);
    CrossCoreSetFlag<..., PIPE_V>(CC_BMM1_0);
    CrossCoreSetFlag<..., PIPE_V>(CC_BMM1_1);
}
```

---

## 七、拷贝指令与计算指令

### 7.1 数据搬运指令

| 路径 | 指令 | 说明 |
|---|---|---|
| GM → L1 | `copyQueryGmToL1_` / `copyKvGmToL1_` | 按 GmCoord/KvGmCoord 计算 offset, 支持 PA block_table 查址, 输出 NZ 格式 |
| L1 → L0A/L0B | `LoadDataToL0A` / `LoadDataToL0B` (matmul.h) | 由 `BuffersPolicyDB` 管理双 buffer Lock/Unlock |
| L0C → UB | `Fixpipe<..., FIXPIPE_ROW_MAJOR_UB>` | BMM1: `dualDstCtl=2` (M/2×N 双目标); BMM2: `dualDstCtl=1` |
| UB → L1 | `DataCopy` (带 stride) | Vec1 softmax 结果 cast 后写回 L1 供 BMM2, 适配 NZ 格式 |
| UB → GM | `copyAttenOutUbToGm_` / `DataCopyPad` | 非 FD: Cast+copyout; FD: 写 workspace accumOut |
| GM → UB (FD) | `DataCopy` / `DataCopyPad` | FD 阶段加载 lseSum/lseMax/accumOut |

### 7.2 Cube 侧计算指令

**BMM1 (Q × K^T)**:

```cpp
// D≤128: 一次 LoadDataToL0 + Mmad
MatmulBase<KV_T, Q_T, MM_T, 128, 128, dBaseSize, ABLayout::MK, ABLayout::KN>(...)
// D>128: K 维度分 128 块, 多次 LoadDataToL0A + Mmad
MatmulK<KV_T, Q_T, MM_T, 128, 128, 128, ABLayout::MK, ABLayout::KN>(...)
```

- A=K (L1→L0A, 转置), B=Q (L1→L0B), 输出 L0C (S2×M)
- `ABLayout::MK` / `ABLayout::KN`: A 按 MK 排布, B 按 KN 排布

**BMM2 (attn × V)**:

```cpp
// DV≤128 且 S2=128: 全量一次算完
MatmulFull<Q_T, KV_T, MM_T, 128, dV, 128, ...>(...)
// DV≤128 且 S2≠128: Base (K 维固定 128)
MatmulBase<Q_T, KV_T, MM_T, 128, dV, 128, ...>(...)
// DV>128: N 维度切分
MatmulN<Q_T, KV_T, MM_T, mBaseSize, 128, s2BaseSize, ...>(...)
```

- A=P (softmax 结果, L1→L0A), B=V (L1→L0B), 输出 L0C (M×DV)

### 7.3 Vector 侧计算指令

**ProcessVec1 (Softmax)** — `ProcessVec1VfDn`:

1. **Cast**: bmm1Res (fp16/bf16) → fp32
2. **Max reduce**: 跨 s2 方向求每行 max
   - 4 路并行 `Max(max0, max1, max2, max3)`
   - 二级归约 `Max(max0, max0, max2)` → `Max(max0, max0, max1)`
   - `Muls(max0, dScale)` 缩放
3. **Sink (hasSink=true 时)**: `Max(max0, max0, vreg_sink_input)` — sink 参与 max
4. **Exp-Sub**: `FusedExpSub(x, max0)` — 融合指令, 计算 $e^{x - \text{max}}$
5. **Sum**: `Mul(sum, exp)` + `Add(totalSum, sum)` 累加
6. **Cast back**: fp32 → fp16/bf16 (写 L1 供 BMM2)
7. **Store**: softmaxMax/softmaxSum 写 UB buffer (per mloop)

> **首次/非首次 softmax VF 统一方案 (计划中)**: 当前区分 `isFirstS2Loop` 调用 NoUpdate/Update 两套 VF。若在首次 tile 前显式初始化 `max=-inf, sum=0, ubVec2Res=0`, 则统一调用 Update 版即可——首次时 `exp(old_max - new_max) = exp(-inf) = 0`, `old_sum * 0 + new_sum = new_sum`, `FlashUpdateNew: 0 * 0 + mm2Res = mm2Res` (等价 NoUpdate 的 DataCopy)。消除 NoUpdate VF (~200 行), 简化代码路径, 且 Sink 集成时 sum 初始化可直接设为 `exp(sink-max)`。前置条件: 验证 950 `FusedExpSub(-inf, finite)` 产生 0 而非 NaN。

**ProcessVec2 (Output 累加)**:

```cpp
// 非最后轮: o = o * expMax + bmm2_res  (expMax = exp(old_max - new_max), rescale factor)
FlashUpdateNew<..., false, false>(ubVec2Res_, mm2Res, ubVec2Res_, expUb, ...);

// 最后轮: o = (o * expMax + bmm2_res) / sum
FlashUpdateLastNew<..., false, false>(ubVec2Res_, mm2Res, ubVec2Res_, expUb, ..., sumUb, ...);
LastDivNew<..., false>(ubVec2Res_, ubVec2Res_, sumUb, ...);
```

> `expUb` 存储 rescale factor `exp(old_max - new_max)`, 供 `FlashUpdateNew` 对之前的 output 做 rescale。NoUpdate 版不写入 expUb; Update 版写入。统一方案下首次 tile 的 expUb = `exp(-inf - tile_max) = 0`, 使 `FlashUpdateNew` 退化为直接赋值。

**FlashDecode 归约**:

`ComputeScaleValue` (跨 split-K 份归约 softmax):
1. **Max 归约**: 跨 `actualCombineLoopSize` (K) 份 lseMax, 求全局 max
2. **Exp+Sum 重算**: `Sub(max_i, globalMax)` → `Exp` → `Mul(sum_i, exp)` → `Add(totalSum, ...)`
3. **Sink (learnableSinkFlag=true 时)**: 见第八章方案
4. **LSE 输出**: `Log(sum) + max` → lse

`ReduceFinalRes` (跨 split-K 份归约 output):
```cpp
ReduceFinalRes_VF<T>(reduceOut, lseLocal, mm2Res, dealRowCount, dSizeV, cntKV);
```
用归约后的 scale 重算每份 accumOut 的权重, 加权求和。

### 7.4 Mask 处理

**无 mask (Dn 路径)**: `hasAtten=false`, softmax 不做 mask select。

**有 mask (Nd 路径)**:
- `attenmask_gs1.h` 提供 `AttentionmaskCopyIn` (mask GM→UB)、`MergeMask`、`ComputeAttenMaskOffset`
- `ProcessVec1VfDnCausal/ProcessVec1Nd` 中 `hasAtten=true` 时用 `Select` 指令将 mask 位置置为 `-inf`
- mask_mode 映射到 `sparseMode`:
  - 0 (NO_MASK): 全计算
  - 3 (CAUSAL): 下三角
  - 4 (BAND): window (pre/next tokens)
- `CalcCurS2StartEndWithSparse` 按 sparseMode 计算每行有效 S2 区间 `[curS2Start, curS2End)`, 跳过无效 tile
- FD 阶段 `DealInvalidRows`/`DealInvalidMaskRows` 处理超出 seqused/pre/next tokens 范围的无效行

mask 矩阵固定为 `(2048, 2048)` 的下三角模板, kernel 按 flash attention 分块从中扣取所需子块, 适配任意 S1/S2 长度。

---

## 八、Sink 特性方案

### 8.1 当前状态

| 层 | 状态 | 说明 |
|---|---|---|
| 接口 (def.cpp) | ⚠️ dtype 不一致 | `sinks` 声明 DT_FLOAT, 但 kernel `SINK_T = INPUT_T` (fp16/bf16) |
| 文档 (flash_attn.md) | ⚠️ "暂不支持" | 需更新 |
| Checker | ❌ 硬编码拦截 | `if (faInfo.sinksFlag) return FAILED` |
| Tiling parser | ✅ 已解析 | `sinksFlag_` 正确设置 |
| Kernel constInfo | ✅ 已设置 | `constInfo_.learnableSinkFlag = (learnableSink != nullptr)` |
| **位置 A: FA ProcessVec1** | ❌ 未打通 | `hasSink` 缺省=false, `sinkValue` 用默认 0.0, sum 未含 sink |
| **位置 B: FD ComputeScaleValue** | ❌ 未打通 | `learnableSinkFlag` 硬编码 false, `sinkExpBuf` 空 |
| VF 计算逻辑 | ✅ 已实现 | `hasSink` 的 `Max(sink)` 分支 + `learnableSinkFlag` 的 `Add(sink)` 分支均完整 |

### 8.2 Sink 参与计算的两个位置

**位置 A (FA 阶段 ProcessVec1)**: sink 参与 max 和 sum

- `ProcessVec1VfDn` 的 `hasSink=true` 分支: `Duplicate(vreg_sink_input, sinkValue)` → `Max(max0, max0, vreg_sink_input)`
- sinkValue 是标量 (float), 通过 `sinkGm.GetValue(n2Idx * gSize + gIdx)` 取单值
- 对应公式: $m = \max(\text{sink}, \max(S))$
- **当前 `hasSink` 只做 Max, 不在 sum 中加 `exp(sink-m)`** — 需在 sum 初始化时设为 `exp(sink-max)`

**位置 B (FD 阶段 ComputeScaleValue)**: sink 修正分母

- `ComputeScaleValue_8_VF_FD` 的 `learnableSinkFlag` 分支
- sink 是 UB tensor (dealCount 个元素, 按 gIdx 映射每行到对应 head 的 sink 值)
- 跨 K 份 split-K 归约时需修正多加的 sink 份数

### 8.3 Sink 方案: 首次初始化 + FD 减去多加份数

#### 核心思路

利用"首次 softmax 统一方案"的初始化机制, 在首次 tile 的 sum 初始化时直接设为 `exp(sink - max)`, 使 sink 自然参与分母; FD 阶段减去多加的 (K-1) 份 sink。

#### 数学推导

**非 FD 路径 (FA 阶段, 单核多 S2 tile)**:

首次 tile (NoUpdate, hasSink=true):
$$
m = \max(\text{sink}, \max(S_{\text{tile}_0}))
$$
$$
\ell = e^{\text{sink} - m} + \sum_{j \in \text{tile}_0} e^{s_j - m}
$$

后续 tile (Update, hasSink=true):
$$
m' = \max(m, \max(S_{\text{tile}_k}))
$$
$$
\ell' = \ell \cdot e^{m - m'} + \sum_{j \in \text{tile}_k} e^{s_j - m'}
$$

rescale 后 sink 项: $e^{\text{sink}-m} \cdot e^{m-m'} = e^{\text{sink}-m'}$ ✅ (自然正确, 无需额外处理)

最终: $\text{output} = \frac{\sum e^{s_j - m'} \cdot v_j}{\ell'}$, 分母含 1 份 sink ✅

**FD 路径 (K 份 split-K, 每份独立计算)**:

每份 i 的 local_max_i ≥ sink, local_sum_i 含 1 份 `exp(sink - local_max_i)`。

FD 归约:
$$
m_{\text{global}} = \max_i(\text{local\_max}_i)
$$
$$
\ell_{\text{total}} = \sum_i \ell_i \cdot e^{\text{local\_max}_i - m_{\text{global}}}
= \sum_j e^{s_j - m_{\text{global}}} + K \cdot e^{\text{sink} - m_{\text{global}}}
$$

正确结果应为 $\sum_j e^{s_j - m_{\text{global}}} + 1 \cdot e^{\text{sink} - m_{\text{global}}}$

**需减去**: $(K-1) \cdot e^{\text{sink} - m_{\text{global}}}$

#### 逐层修改清单

| 层 | 文件 | 修改 |
|---|---|---|
| **common (VF)** | `vf_mul_sel_softmaxflashv2_cast_nz_dn.h` | NoUpdate 的 `hasSink=true` 分支: sum 初始化从 0 改为 `exp(sink - max)` (`Sub` → `Exp` → `Mov` 到 sum 寄存器) |
| **common (VF)** | `vf_mul_sel_softmaxflashv2_cast_nz.h` | Nd 版 `ProcessVec1Vf` 的 `hasSink=true` 分支: 同上 |
| **common (VF)** | `vf_flash_decode.h` | `ComputeScaleValue_8_VF_FD` 的 `learnableSinkFlag` 分支: `Add(sum, sinkExp)` 改为 `Muls(sinkExp, K-1)` + `Sub(sum, sinkExp)` |
| **block_vec** | `block_vec_dn.h` / `block_vec_nd.h` | InitBlock 保存 `sinkGm_`; ProcessVec1 中 `GetValue(n2Idx * gSize + gIdx)` 取 sinkValue; if 分支传 `hasSink=true` + `sinkValue` (两套编译路径) |
| **fd** | `block_vec_flashdecode.h` | InitBlock 保存 `sinkGm_`; 新增 `CopySinkIn` (DataCopy + Cast + Brcb) + `SinkMax` (按 gIdx 映射每行 sink 值); ComputeScaleValue 传真实 flag + sinkExpBuf; InitBuffers 新增 sink UB buffer |
| **kernel** | `kernel_dn.h` / `kernel_nd.h` | CalcParams 中赋值 `info.gIdx` (当前 RunInfoX 有 gIdx 字段但未赋值) |
| **block_cube** | `block_cube_dn.h` / `block_cube_nd.h` | 无需修改 (sink 是纯 Vector 概念) |
| **host** | `def.cpp` | sinks dtype: DT_FLOAT → DT_FLOAT16/DT_BF16 |
| **host** | `sinks_checker.cpp` | 解除拦截, 改为校验 dtype/shape |
| **host** | `flash_attn.md` | Sinks 参数组: "暂不支持" → 正式约束 |

#### FD 阶段 sink 数据加载

`ComputeScaleValue_8_VF_FD` 从 `lseSink` (UB) 加载 `dealCount = dealRowCount × 8` 个 float 元素。每行的 sink 值取决于该行所属的 g group, 需通过 `CopySinkIn` + `SinkMax` 映射:

```cpp
// CopySinkIn: 从 sinkGm 加载 gSize 个 sink 值 (per n2 head), Cast 到 float, Brcb 广播
DataCopy(sinkCopyInBuf, sinkGm[n2Idx * gSize], copySize);
Cast(sinkCastBuf, sinkCopyInBuf, CAST_NONE, gSize);
Brcb(sinkBrcbBuf, sinkCastBuf, ...);

// SinkMax: 按每行 gIdx 从广播结果取对应 sink 值到 sinkExpBuf
for (row = 0; row < dealRowCount; ++row) {
    gIdx = (BSND/TND) ? (gS1Idx + row) % gSize : (gS1Idx + row) / actS1Size;
    DataCopy(sinkExpBuf[row * 8], sinkBrcbBuf[gIdx * 8], 8);
}
```

参考 FIA fullquant: `fia_block_vec_flashdecode_fullquant.h:420-465`。

#### 关键风险

1. **hasSink 编译期 vs 运行时**: `hasSink` 是模板参数, `learnableSinkFlag` 是运行时值。需用 `if` 分支编译两套调用路径 (hasSink=true/false), 增加代码量但无运行时开销。
2. **UB 空间**: FD 阶段需新增 4 个 sink buffer (copyIn 1K / cast / brcb / exp), 需确认 UB 剩余空间。flashdecode.h 的 InitBuffers 与 vec_dn.h 共享 UB 布局 (FD 复用 FA 的 buffer 区域)。
3. **expUb 统一方案的前置条件**: `FusedExpSub(-inf, finite)` 需产生 0 而非 NaN, 需上板验证。
4. **与 FIA 的差异**: FIA noquant 路径只做 `Max(sink)` 不在 sum 加 sink; 本方案在 sum 初始化时加 sink, 比FIA 更完整, 但偏离 FIA 实现。

---

## 九、接口规格

### 9.1 函数原型

```python
# 前置: 生成 metadata
cann_ops_transformer.flash_attn_metadata(
    num_heads_q, num_heads_kv, head_dim, *,
    cu_seqlens_q=None, cu_seqlens_kv=None,
    seqused_q=None, seqused_kv=None,
    batch_size=None, max_seqlen_q=None, max_seqlen_kv=None,
    mask_mode=None, win_left=None, win_right=None,
    layout_q=None, layout_kv=None, layout_out=None
) -> Tensor

# 主算子
cann_ops_transformer.flash_attn(
    q, k, v,
    block_table=None,
    cu_seqlens_q=None, cu_seqlens_kv=None,
    seqused_q=None, seqused_kv=None,
    sinks=None,
    metadata=None,
    softmax_scale=1.0,
    mask_mode=0,
    attn_mask=None,
    win_left=-1, win_right=-1,
    max_seqlen_q=-1, max_seqlen_kv=-1,
    layout_q="BSND", layout_kv="BSND", layout_out="BSND",
    return_softmax_lse=False
) -> (Tensor, Tensor)
```

### 9.2 支持的配置

| 特性 | 支持范围 |
|---|---|
| dtype | float16 / bfloat16 |
| D (head_dim) | 64 / 128 / 256 |
| layout_q/out | BSND / BNSD / TND |
| layout_kv | BSND / BNSD / TND / PA_BBND / PA_BNBD / PA_NZ |
| mask_mode | 0 (全计算) / 3 (causal) / 4 (window) |
| attn_mask | int8, (2048, 2048) 下三角模板 |
| GQA | Q_N % KV_N == 0 |
| Paged Attention | block_size ∈ [16, 1024], % 16 == 0 |
| 变长序列 | cu_seqlens (TND packing) + seqused (截断) |
| FlashDecode | 自动启用 (enableFlashDecode) |
| softmax_lse | return_softmax_lse=True 时输出 |
| sinks | ⚠️ 暂不支持 (参数已声明, 待打通) |

### 9.3 layout 匹配关系

| layout_q | layout_kv | layout_out | softmax_lse |
|---|---|---|---|
| BNSD | BNSD/PA_BBND/PA_BNBD/PA_NZ | BNSD/BSND | (B, Q_N, Q_S) |
| BSND | BSND/PA_BBND/PA_BNBD/PA_NZ | BSND | (B, Q_N, Q_S) |
| TND | TND/PA_BBND/PA_BNBD/PA_NZ | TND | (Q_N, Q_T) |

---

## 十、编译与测试

### 10.1 编译命令

工作目录: `/workspace/ops-transformer`

```bash
# 全量编译 (生成所有 tiling key 的 kernel)
bash build.sh --pkg --soc=ascend950 --ops=flash_attn -j16

# 快速编译 (仅 3 个代表性 tiling key)
bash build.sh --pkg --soc=ascend950 --ops=flash_attn -j16 \
    --tiling_key="2279866368;2279866369;2279866370"
```

> **必须指定 `-j16`**: 本机 `/proc/cpuinfo` 返回 128 核但 cgroup 限制为 32 核, 不指定会触发 OOM。

编译产物:
```
build_out/cann-ops-transformer-custom_linux-x86_64.run
```

编译成功标志:
```
Self-extractable archive "cann-ops-transformer-custom_linux-x86_64.run" successfully created.
```

编译中断后恢复:
```bash
rm -rf build/binary/ascend950/gen   # 清理残留
bash build.sh --pkg --soc=ascend950 --ops=flash_attn -j16
```

### 10.2 编译验证

```bash
# 错误计数 (应为 0)
grep -c "error:" /tmp/opencode/fa_build.log

# 成功标记
grep "successfully created" /tmp/opencode/fa_build.log

# kernel 产物 (应生成 2 个 .o 文件)
find build/binary/ascend950/bin -name "FlashAttn_*.o"
```

### 10.3 单元测试

```bash
# Tiling 单测
cd /workspace/ops-transformer
bash build.sh --ops=flash_attn --soc=ascend950 -j16
# 运行 tiling UT
cd build && ctest -R flash_attn_tiling --output-on-failure
```

测试用例 CSV: `attention/flash_attn/tests/ut/op_host/arch35/test_flash_attn_tiling.csv`

典型测试用例:

| 用例名 | layout | dtype | mask_mode | 说明 |
|---|---|---|---|---|
| BSND_FP16_Normal | BSND | FP16 | 0 | 标准 BSND 无 mask |
| BSND_FP16_S1_64 | BSND | FP16 | 0 | 短序列 S1=64 |
| BSND_BF16_Causal | BSND | BF16 | 3 | causal mask |
| BNSD_FP16_Window | BNSD | FP16 | 4 | window mask |
| TND_FP16_Varlen | TND | FP16 | 0 | 变长序列 packing |

### 10.4 调用示例

```python
import torch
import torch_npu
import cann_ops_transformer

torch_npu.npu.set_device(0)

dtype = torch.bfloat16
B, S, Q_N, KV_N, D = 2, 16, 8, 2, 128

q = torch.randn(B, S, Q_N, D, dtype=dtype, device="npu")
k = torch.randn(B, S, KV_N, D, dtype=dtype, device="npu")
v = torch.randn(B, S, KV_N, D, dtype=dtype, device="npu")

# 1. 生成 metadata
metadata = cann_ops_transformer.ops.flash_attn_metadata(
    Q_N, KV_N, D,
    batch_size=B, max_seqlen_q=S, max_seqlen_kv=S,
    mask_mode=0, win_left=-1, win_right=-1,
    layout_q="BSND", layout_kv="BSND", layout_out="BSND",
)

# 2. 执行 flash_attn
attn_out, softmax_lse = cann_ops_transformer.ops.flash_attn(
    q, k, v,
    metadata=metadata,
    softmax_scale=1.0 / (D ** 0.5),
    mask_mode=0, win_left=-1, win_right=-1,
    layout_q="BSND", layout_kv="BSND", layout_out="BSND",
    return_softmax_lse=False,
)
torch_npu.npu.synchronize()
assert attn_out.shape == q.shape
assert attn_out.dtype == q.dtype
```

---

## 十一、关键设计决策

### 11.1 为什么用 Mix Kernel (AIC+AIV 分离)?

BMM1/BMM2 是 Cube 计算密集型, Softmax 是 Vector 计算密集型。Mix kernel 让 AIC 做 matmul、AIV 做 softmax, 两者并行流水, 充分利用 950 的 Cube+Vector 双引擎。

### 11.2 为什么 PRELOAD_N=2?

BMM1 结果 (UB) → AIV ProcessVec1 → cast 结果 (L1) → AIC BMM2。数据从 AIC→AIV→AIC 走一个来回。PRELOAD_N=2 让当前轮 BMM1 和 2 轮前的 BMM2 并行, 隐藏 CV 通信延迟。PRELOAD_TASK_CACHE_SIZE=4 (2 的幂) 配合位掩码 `loop & 3` 避免取模。

### 11.3 为什么 Q buffer 只在 isFirstS2Loop 加载?

一个 gS1 行内 (固定 Q), s2 循环多次 (不同 KV 分块), Q 矩阵不变。首次加载后复用, 末次释放, 节省 MTE2 带宽。Q 用 2 buffer (与 gS1 行切换配合)。

### 11.4 为什么 KV 共享 4 buffer?

BMM1 用 K, BMM2 用 V, 两者时间错开 (PRELOAD_N=2)。4 buffer 让 K 和 V 的加载交替进行, 充分利用 L1 带宽, 避免冲突。

### 11.5 为什么 mask 用固定 (2048, 2048) 矩阵?

kernel 按 flash attention 分块, 每个 tile 从模板矩阵中扣取所需子块, 适配任意 S1/S2 长度。固定矩阵避免动态 shape 的 tiling 复杂性, 且 mask 矩阵只占 4MB (int8), 可常驻 GM。

### 11.6 为什么 FlashDecode 用 SyncAll?

FD 阶段所有 AIV 核需要读取 FA 阶段所有核写入 workspace 的中间结果 (lseSum/lseMax/accumOut)。`SyncAll` 确保 FA 阶段全部完成后再开始 FD, FD 全部完成后再进入下一 section。

### 11.7 为什么计划消除首次/非首次 softmax VF 区分?

当前 `ProcessVec1Dn` 按 `isFirstS2Loop` 调用两套 VF:
- **NoUpdate** (首次): max 初始化为 `-inf`, sum 初始化为 0, expUb 不写入, ProcessVec2 用 `DataCopy` 直接赋值
- **Update** (非首次): 加载 old_max/old_sum, rescale 后累加, expUb 写入 `exp(old_max - new_max)`, ProcessVec2 用 `FlashUpdateNew` 做 `pre * expMax + cur`

若首次 tile 前显式初始化 `max=-inf, sum=0, ubVec2Res=0`, 统一调用 Update 版:
- `exp(-inf - tile_max) = 0` → `old_sum * 0 + new_sum = new_sum` ✅
- `FlashUpdateNew: 0 * 0 + mm2Res = mm2Res` (等价 DataCopy) ✅

**收益**: 消除 NoUpdate VF (~200 行), 减少 kernel binary, 改善 I-cache, 简化 Sink 集成 (sum 初始化时可直接设 `exp(sink-max)`)。

**前置条件**: 950 的 `FusedExpSub(-inf, finite)` 需产生 0 而非 NaN (IEEE 754 规定 `exp(-inf)=0`, 但融合指令需验证)。

### 11.8 为什么 Sink 方案用"首次初始化 + FD 减去多加份数"?

**问题**: Sink 需参与 softmax 分母 ($\sum e^{S-m} + e^{\text{sink}-m}$), 但 FA 阶段的 `hasSink` 分支只做 `Max(max, sink)`, 不在 sum 中加 `exp(sink-max)`。

**方案**: 利用首次 softmax 统一方案的初始化机制:
- **FA 首次 tile**: sum 初始化为 `exp(sink - max)` (而非 0), 后续 Update 的 rescale 自然保持 sink 项正确
- **FD 阶段**: K 份 split-K 每份含 1 份 sink, 归约后多 (K-1) 份, 需 `Sub(sum, (K-1) * exp(sink - global_max))`

**优势**:
- 非 FD 路径 (prefill) sink 完整参与 max + sum (比 FIA noquant 更完整)
- 无需在 Update VF 中添加 sink 逻辑 (rescale 自然正确)
- FD 阶段只需改 `Add` → `Sub(K-1)`, 逻辑简单

**与 FIA 差异**: FIA noquant 只做 `Max(sink)` 不在 sum 加 sink; 本方案 sum 含 sink, 更数学完整, 但偏离 FIA。
