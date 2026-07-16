# 需求背景

* 目标：支持token粒度计算结果一致性；

    * batch间拆分的一致性：例如B = 8，拆分成两次（B = 3和B = 5）算子调用，或调换batch顺序，对应batch的计算结果一致。

    * 单batch内拆分的一致性：例如B = 1，Token = 16K，拆分成两次（B = 1，Token = 4K和B = 1，Token = 12K）的算子调用，拆分前后对应token的计算结果一致。

    * 相同token，不同batch shape的计算一致性：例如Shape1，B = 1，Token = 300 + 100；Shape2，B = 1，Token = 300 + 200。两个Shape前300个Token完全一样，则前300个Token对应输出的计算结果一致。

* 影响batch一致性的关键是在FlashDecode场景，S2开多核，不同Shape可能导致分核结果不同。规约计算的顺序，会影响结果的一致性。

    例如，在32核环境下，计算下面两个Shape。

    * Shape1：B = 1，S1 = 1，S2 = 16384。S2开多核平分到32核参与计算。16384/128=512，每个核计算4轮循环，最后整体规约。

    * Shape2：B = 32，S1 = 1，S2 = 16384。每个核计算一个batch，S2不开多核。每个核计算128轮循环，直接输出结果。

> 说明：对于NPU环境，浮点数计算顺序变化会影响结果一致性。例如：\(a + b\) + c和a + \(b + c\)的结果不一定完全一致。

* 思路：

    * 需要保证每个token的切分，只跟自身长度有关，不受其他token影响

    * 为了兼顾性能，默认S2开多核

|符号|含义|
|---|---|
|B|batch数|
|N1|Query的head数|
|N2|KV的head数|
|S1|Q的序列长度|
|S2|KV的最大序列长度（规约维度）|
|S2\_*b = seqused\_*kv\[b\]|第 b 个 batch，实际参与attention计算的token数量|
|D|head维度 \(head dim\)|
|G = N1/N2|每个KV head对应的Q head数（group size），作为矩阵乘的m轴|

# 功能开关

## 1、一致性等级

|level取值|含义|说明|
|---|---|---|
|0|不开启确定性|SMLA&amp;MQSMLA&amp;QSMLA算子**默认是确定性的**，配置0/1对算子行为没有影响|
|1|开启确定性||
|2|强一致性|其他算子使用，不涉及，不处理|
|3|batch一致性|**本次迭代需求，开启batch一致性**|

## 2、用户如何配置功能开关

（1）在torch调用场景（torch直调、acl\_graph等）

```python
torch_npu.npu.set_deterministic_level(level)
```

[https://www.hiascend.com/document/detail/zh/Pytorch/2600/apiref/torchnpuCustomsapi/docs/zh/custom\_APIs/torch\_npu\-npu/torch\_npu\-npu\-set\_deterministic\_level.md](https://www.hiascend.com/document/detail/zh/Pytorch/2600/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu-npu/torch_npu-npu-set_deterministic_level.md)

（2）在aclnn直调场景

```cpp
// 函数原型，返回值ACL_SUCCES（0）表示调用成功
aclError aclrtSetSysParamOpt(aclSysParamOpt opt, int64_t value);

// 调用示例
aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 3);
```

AscendC接口文档：

[https://www.hiascend.com/document/detail/zh/canncommercial/900/API/runtimeapi/aclcppdevg\_03\_1776.html](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/runtimeapi/aclcppdevg_03_1776.html)

aclSysParamOpt枚举值定义文档：

[https://www.hiascend.com/document/detail/zh/canncommercial/900/API/runtimeapi/aclcppdevg\_03\_1393.html](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/runtimeapi/aclcppdevg_03_1393.html)

## 3、算子如何获取功能开关

（1）在opapi

```cpp
// 函数原型，返回值ACL_SUCCES（0）表示调用成功
aclError aclrtGetSysParamOpt(aclSysParamOpt opt, int64_t *value);
```

（2）在tiling侧

AscendC提供GetDeterministic接口，通过TilingContext指针获取配置。

```cpp
int32_t GetDeterministicLevel() const;
```

[https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/basicdataapi/atlasopapi\_07\_00792.html](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/basicdataapi/atlasopapi_07_00792.html)

算子适配如下：

（1）tiling类新增类成员变量，保存功能开关，默认关闭。

```cpp
bool isBatchConsistent = false;
```

（2）调用接口，获取功能开关，区分是否开启batch一致性，其他取值不需要处理。

```cpp
ge::graphStatus Tiling4XXX(TilingContext* context) {
    int32_t ret = context->GetDeterministicLevel();
    if (ret == 3) {
        // 开启batch一致性
        isBatchConsistent = true;
    }
}
```

（3）新增Tiling Key

模板参数新增字段，区分是否开启batch一致性，相应的修改tiling key计算的位置。

```cpp
sparse_flash_mla_template_tiling_key.h
ASCENDC_TPL_ARGS_DECL(SparseFlashMla,
...
ASCENDC_TPL_BOOL_DECL(BATCH_CONSISTENCY, 0, 1),
...
);
```

# Tiling层级划分

> 思路：**保证每个token的切分，只跟自身长度有关，不受其他token影响**

## （1）全局视图

```cpp
Batch 维 [B]  (每个batch的S2_b可不同)
└── KV Head 维 [N2] (每个KV head独立计算)
    └── Q序列维 [S1] (每个query position独立计算)
        │
        │  每个计算单元: q[G, D] (G = N1/N2 个Q head作为m轴)
        │
        └── KV序列维 S2_b (规约维，per batch不同，做多级tiling)
            ├── Level-1: 规约级 (Reduction Tile)
            │   大小 = R_b = align_up(S2_b/32, 128)
            │   ┌────────────────────────────────┐
            │   │  Level-2   Level-2   Level-2   │
            │   │  [128]     [128]     [128]     │
            │   └────────────────────────────────┘
            ├── Level-1: 规约级
            │   ...
            └── Level-1: 规约级
```

## （2）PerBatch 切分公式

```cpp
S2_b = seqused_kv[b]                    // 第 b 个 batch 的实际 KV 长度
R_b  = align_up(S2_b / 32, 128)         // 第 b 个 batch 的规约块大小
num_tiles_b = S2_b / R_b                // 第 b 个 batch 的规约块数量
```

|层级|切分维度|切分大小|约束|
|---|---|---|---|
|Level\-1|S2\_b|`R_b = align_up(S2_b / 32, 128)`|多核调度的最小单位，per batch|
|Level\-2|S2\_b|`128` \(固定\)|片上计算的基本块|

**示例：**

|batch|S2\_b\(seqused\_kv\)|S2\_b / 32|R\_b \(对齐128\)|numtiles\_b|
|---|---|---|---|---|
|0|4096|128|128|32|
|1|8192|256|256|32|
|2|2048|64|128|16|
|3|5120|160|256|20|

Batch0，每个核内计算128（1轮循环），生成32个规约块，规约计算后输出最终结果。

Batch1，每个核内计算256（2轮循环），生成32个规约块，规约计算后输出最终结果。

Batch2，每个核内计算128（1轮循环），生成16个规约块，规约计算后输出最终结果。

Batch3，每个核内计算256（2轮循环），生成20个规约块，规约计算后输出最终结果。

## （3）总负载与任务总量

**总负载不是规约块的数量，而是所有 batch 的 负载 累加和，可沿用当前模板的总负载计算；**

```auth
# 示例：
total_workload = sum(S2_b for b in range(B))      // 真实计算量的度量
total_tasks    = sum(num_tiles_b for b in range(B)) × N2 × S1
```

每个 `(b, n2, s1, tile)` 四元组对应一个独立的计算任务，产出一个 partial result（形状为 `[G, D]`）。不同 batch 的任务计算量可能不同（R\_b 不同）。

> 之前的FD计算，根据整体Shape的负载均衡计算分核与规约任务分配。开启batch一致性之后，分配方案只根据S2\_b计算生成。
> 在整体计算负载之后，有可能当前batch还是分在一个核内。**仍然要求按照规约块大小完成核内循环和规约计算**，以保证结果一致。

# 计算流程

## 单规约块内部（Level\-2 循环）

一个规约块包含 `R / 128` 个基本块，在单核内串行执行：

```cpp
输入: q[G, D], K_tile[R, D], V_tile[R, D]    (G = N1/N2)
输出: partial_O[G, D], partial_m[G], partial_l[G]

m = -inf, l = 0, O = 0          # 初始化，形状均含G维

for j in range(R // 128):       # Level-2 循环
    Kj = K_tile[j*128 : (j+1)*128, :]    # [128, D]
    Vj = V_tile[j*128 : (j+1)*128, :]    # [128, D]

    Sj = q @ Kj^T / sqrt(D)              # [G, 128]  mm, m轴=G
    m_new = max(m, rowmax(Sj))            # online softmax [G]
    Pj = exp(Sj - m_new)                 # [G, 128]
    l = exp(m - m_new) * l + rowsum(Pj)
    O = exp(m - m_new) * O + Pj @ Vj     # 累加 [G, D]  mm, m轴=G
    m = m_new

partial_O = O      # [G, D]
partial_m = m      # [G]
partial_l = l      # [G]
```

## 跨规约级合并（Combine）

所有规约级完成后，合并各 partial result：

```cpp
输入: {partial_O_i[G, D], partial_m_i[G], partial_l_i[G]}, i = 0..num_tiles-1
输出: final_O[G, D]

# 全局最大值
M = max(partial_m_0, partial_m_1, ..., partial_m_{n-1})   # [G]

# 重新缩放并合并
L = 0, O = 0
for i in range(num_tiles):
    scale_i = exp(partial_m_i - M)        # [G]
    L += scale_i * partial_l_i
    O += scale_i * partial_O_i            # [G, D]

final_O = O / L                           # [G, D]
```

# 多核切分策略

**约束：多核只能以规约块（Level\-1 tile）为调度单位，一个核可承载多个规约块。**

## 分配原则

**负载度量：以 S2 累加和为基准，而非规约块数量。**

不同 batch 的规约块大小 R\_b 不同，因此不能简单按规约级个数均分。以实际计算量（S2 累加和）均衡分配：

```cpp
total_workload = sum(S2_b for b in range(B))   // 所有batch的KV长度之和
workload_per_core = total_workload / C          // C = 32

// 分配时，按规约级粒度贪心填充，使每核负载接近 workload_per_core
// 每个规约级的负载 = R_b (该规约级覆盖的KV长度)
```

每个核串行处理分配到的多个规约级，每个规约级独立输出一个 partial result，核内根据情况决定是否就地 combine（见 4.5 节）。

## 示例：变长 batch 的负载分配

```yaml
假设 B=3, N2=1, S1=1, C=4 核 (简化为单head单token):
  seqused_kv = [4096, 2048, 5120]
  R_0 = align_up(4096/32, 128) = 128,  num_tiles_0 = 32, 负载 = 4096
  R_1 = align_up(2048/32, 128) = 128,  num_tiles_1 = 16, 负载 = 2048
  R_2 = align_up(5120/32, 128) = 256,  num_tiles_2 = 20, 负载 = 5120

  total_workload = 4096 + 2048 + 5120 = 11264
  workload_per_core = 11264 / 4 = 2816

按负载贪心分配:
  Core 0: batch0 tile[0..21]   负载 = 22×128 = 2816
  Core 1: batch0 tile[22..31] + batch1 tile[0..11]   负载 = 10×128 + 12×128 = 2816
  Core 2: batch1 tile[12..15] + batch2 tile[0..8]    负载 = 4×128 + 9×256 = 2816
  Core 3: batch2 tile[9..19]  负载 = 11×256 = 2816
```

## 示例：单 batch 场景

```yaml
S2_0 = 2048, R_0 = 128, 16个规约块, C = 4核:

workload_per_core = 2048 / 4 = 512 (即每核 4 个规约块)

Core 0: batch0 tile[0..3]   [K[0:512],    V[0:512]]    → partial_0..3
Core 1: batch0 tile[4..7]   [K[512:1024], V[512:1024]]  → partial_4..7
Core 2: batch0 tile[8..11]  [K[1024:1536],V[1024:1536]] → partial_8..11
Core 3: batch0 tile[12..15] [K[1536:2048],V[1536:2048]] → partial_12..15
                              ↓
               Combine: 合并 partial_0..15 → final_O
```

## 核内执行流程（一核多规约块）

```python
# Core i 负责规约级 [start_tile, end_tile)，处理某个 (b, n2, s1) 的计算

q = Q[b, n2*G:(n2+1)*G, s1, :]           # [G, D]  (G = N1/N2 个Q head)

for t in range(start_tile, end_tile):       # 遍历本核的各规约块
    # --- Level-2 循环：计算当前规约块 ---
    m = -inf, l = 0, O = 0
    for j in range(R // 128):
        Kj, Vj = load_block(b, n2, t, j)   # [128, D]
        Sj = q @ Kj^T / sqrt(D)            # [G, 128]
        m_new = max(m, rowmax(Sj))
        Pj = exp(Sj - m_new)
        l = exp(m - m_new) * l + rowsum(Pj)
        O = exp(m - m_new) * O + Pj @ Vj   # [G, D]
        m = m_new

    # --- 直接写回当前规约块的 partial result ---
    write_back(partial_O[t] = O, partial_m[t] = m, partial_l[t] = l)
```

## Combine 策略（基于 Workspace 优化）

根据某个 token 的规约级是否全部落在同一个核，分两种 combine 路径：

**假定核数 C = 32。**

#### 情况一：同核 Combine（规约块全在同一核内）

当某个 `(b, n2, s1)` 的所有规约块均分配在同一个核时，该核在计算完所有规约块后，**就地完成 combine**，直接输出最终结果，无需跨核通信。

```python
# Core i 负责某 (b, n2, s1) 的全部规约块
# 计算完所有 partial result 后，立即在核内 combine

M = max(partial_m[0], ..., partial_m[num_tiles-1])  # [G]

L = 0, O = 0
for t in range(num_tiles):
    scale_t = exp(partial_m[t] - M)
    L += scale_t * partial_l[t]
    O += scale_t * partial_O[t]

O_final[b, n2*G:(n2+1)*G, s1, :] = O / L   # 直接写回最终结果
```

#### 情况二：跨核 Combine（规约块分布在多个核上）

当某个 `(b, n2, s1)` 的规约块分布在多个核时，各核将 partial result 写入共享 workspace，待所有核执行完成后再统一 combine。

```python
# 各核将 partial result 写入 workspace
workspace_O[core_id, tile_local_idx] = O     # [G, D]
workspace_m[core_id, tile_local_idx] = m     # [G]
workspace_l[core_id, tile_local_idx] = l     # [G]

# --- 同步屏障：等待所有核完成 ---

# Combine 阶段：汇总 workspace 中所有 partial result
M = max(所有相关 partial_m)                   # [G]
L = 0, O = 0
for each partial in workspace:
    scale = exp(partial_m - M)
    L += scale * partial_l
    O += scale * partial_O

O_final[b, n2*G:(n2+1)*G, s1, :] = O / L
```

## Workspace 空间设计

```sql
┌─────────────────────────────────────────────────────────────┐
│  核内 Workspace (per core)                                  │
│                                                             │
│  大小: C × 1 × partial_result_size                          │
│      = 32 × 1 × (G×D + G + G)                               │
│                                                             │
│  用途: 存放同核 token 的各规约块 partial result,             │
│        计算完即刻 combine, 空间可复用                        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  跨核 Workspace (共享)                                      │
│                                                             │
│  大小: C × 32 × partial_result_size                         │
│      = 32 × 32 × (G×D + G + G)                              │
│                                                             │
│  用途: 跨核 token 的 partial result 暂存,                    │
│        每个跨核token最多贡献 C 个 partial result,            │
│        共 C 个核 → C×C 个槽位                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

|Workspace 类型|大小|说明|
|---|---|---|
|核内 workspace \(per core\)|`C × (G×D + 2G)`|同核 combine 的中间存储，可复用|
|跨核 workspace \(全局共享\)|`C × 32 × (G×D + 2G)`|跨核 pa|

简化计算（以 G=8, D=128 为例）：

```auth
partial_result_size = G×D + G + G = 8×128 + 8 + 8 = 1040 元素

核内 workspace (per core): 32 × 1040 = 33,280 元素
跨核 workspace (全局):     32 × 32 × 1040 = 1,064,960 元素
```

## 核内执行流程（含 Workspace 优化）

```python
# Core i 的执行流程，假定核数 C = 32

for each (b, n2, s1, tile) assigned to this core:
    # --- Level-2 循环：计算当前规约块 ---
    q = Q[b, n2*G:(n2+1)*G, s1, :]         # [G, D]
    m = -inf, l = 0, O = 0
    for j in range(R // 128):
        Kj, Vj = load_block(b, n2, tile, j) # [128, D]
        Sj = q @ Kj^T / sqrt(D)             # [G, 128]
        m_new = max(m, rowmax(Sj))
        Pj = exp(Sj - m_new)
        l = exp(m - m_new) * l + rowsum(Pj)
        O = exp(m - m_new) * O + Pj @ Vj    # [G, D]
        m = m_new

    # --- 根据 token 的规约块分布选择路径 ---
    if token_all_on_this_core(b, n2, s1):
        # 情况一: 存入核内 workspace，累积完后就地 combine
        local_buf[local_idx] = (O, m, l)
        if is_last_tile_of_token(b, n2, s1):
            combine_and_write_final(local_buf, b, n2, s1)
    else:
        # 情况二: 存入跨核 workspace，等待全局 combine
        cross_workspace[core_id, slot_idx] = (O, m, l)

# --- 同步屏障 ---
# 对跨核 token 执行全局 combine
```

## 调度规则

1. **分配粒度**：以完整规约块为最小调度单位，不可拆分 Level\-1 tile 跨核

2. **负载均衡基准**：以`sum(S2_b) / C`为每核目标负载，按规约块粒度做负载均衡

3. **一核多级**：每核可承载来自不同 batch 的多个规约块

4. **同核 combine 优化**：若某 token`(b, n2, s1)`的全部规约块落在同一核，核内即刻 combine，节省跨核通信与 workspace

5. **跨核 combine**：规约块跨核的 token，partial result 写入共享 workspace，待所有核完成后统一 combine

# 内存排布

```yaml
片上 (per core, 处理单个 (b, n2, s1) 的单个规约块):
  Q_buf:       [G, D]         — G = N1/N2 个Q head，常驻 (m轴)
  KV_buf:      [128, D] × 2   — Level-2 ping-pong
  Acc_O:       [G, D]         — 累加器
  Acc_m/l:     [G]            — softmax状态

Workspace (C = 32):
  核内 workspace (per core):
    local_O:   [C, G, D]      — 同核 token 的 partial result 暂存
    local_m:   [C, G]
    local_l:   [C, G]

  跨核 workspace (全局共享, HBM):
    cross_O:   [C, C, G, D]   — 跨核 token 的 partial result
    cross_m:   [C, C, G]      — 32个核 × 每核最多32个 partial
    cross_l:   [C, C, G]
```

# 修改汇总

## 1、METADATA算子修改

### OP\_API修改

AICPU算子没有实现Tiling部分，可以在op\_api中获取（aclrtGetSysParamOpt），通过L0接口传入。

**不修改aclnn接口和torch接口。**

```cpp
aclnnStatus aclnnSparseFlashMlaMetadataGetWorkspaceSize(xxxxxx)
{
        ...
        int64_t value;
        aclError ret = aclrtGetSysParamOpt(ACL_OPT_DETERMINISTIC, &value);
        CHECK_RET(ret != ACLNN_SUCCESS, ret);

        auto output = l0op::SparseFlashMlaMetadata(xxxxxx, value, metaData, uniqueExecutor.get());
        CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);
        ...
}
```

### OP\_KERNEL修改

1、解析batch一致性开关。

```cpp
// AICPU计算入口
uint32_t SparseFlashMlaMetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    /* 
        参数解析，补充解析batch一致性参数，
        判断是否等于3，修改isBatchConsistent，其他值不处理
     */
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    SplitResult splitRes {aicCoreNum_, aivCoreNum_};
    /*
        分核计算阶段，区分判断
     */
    success = BalanceSchedule(splitRes) && GenMetadata(splitRes);
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}
```

```cpp
class SparseFlashMlaMetadataCpuKernel : public CpuKernel {
public:
    ...
    ...
private:
    ...
    ...
    ...
    bool isBatchConsistent_ = false;
}
```

2、BalanceSchedule计算分核

（1）总负载计算，同步计算规约块大小

修改路径：BalanceSchedule \-> CalcCostInfo

（2）重点修改分块逻辑，划分时要按照规约块来计算

修改路径：BalanceSchedule \-> CalcSplitPlan \-> AssignBlocksToCore

3、其他逻辑同步适配

SplitFD计算规约任务时，注意不要打破规约块

GenMetadata生成metadata信息，注意不要打破规约块

## SMLA算子修改

### OP\_TILING修改

1、在Tiling侧获取batch一致性开关，参考【功能开关】章节。

2、workspace计算，参考【多核切分策略】和【内存排布】章节。

3、新增TilingKey区分计算逻辑。

### OP\_KERNEL修改

参考【计算流程】、【多核切分策略】、【内存排布】章节适配
