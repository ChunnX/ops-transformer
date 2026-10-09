# Arch35 MTE MegaMoe 循环 GM buffer 与跨窗口去重容量推导

- 日期：2026-10-09
- 分析基线：`7bad535f3`（叠加 PR #13256 后的代码）
- 分支：`feat/mega-moe-ccl-workspace-pr13256`
- 状态：静态设计与容量推导，尚未实现或进行 NPU 验证。

## 1. 需求与适用范围

Dispatch 接收数据、Activation 输出、实际落 GM 的 GMM1 结果、GMM2 结果各有独立的固定容量循环 buffer。这些 buffer 跨 wave 复用，同一 wave 内不同阶段不共用地址空间。

物理槽位单位为 256 行 M group。专家或专家段的有效行数不足 256 时，仍占一个完整槽位，下一段从下一槽位开始。全局 token 行号用于路由、metadata 和 rowDesc，不能直接作为循环数据区的物理行偏移。

本轮仅讨论 Arch35 MTE、非量化 Combine。量化 Combine 暂不纳入；Dispatch 去重、Combine 去重及两者同时启用均需考虑。共享专家区、通信区、metadata、rowDesc 和同步区不计入本文的数据主体比较，最终申请时仍需计入。

GMM1 并非所有模板都落 GM：A8W4 或 topkWeightsPrefetch 开启时分配 GMM1 GM 结果；其他适用路径走 UB，不新增对应循环区。A4W4 的 Activation 输出为 FP8，不能按 FP4 压缩该区域。

## 2. 当前代码依据

以下路径均相对仓库根目录：

- [workspace 布局](../op_kernel/arch35/common/mega_moe_workspace.h)：当前各数据区按 maxOutputSize 预留。
- [tiling 与 wave 预算](../op_host/op_tiling/arch35/mega_moe_tiling.cpp)：CalcMGroupsPerWave、SetPrefetchAndWaveParams、CheckAndSetMaxRecvTokenNumAttr。
- [去重建表](../op_kernel/arch35/stage/mega_moe_token_dedup.h)：按源卡和源 token 建立成员组，成员按专家顺序排列；接收截断后仅保留有效成员。
- [Dispatch 去重](../op_kernel/arch35/stage/mega_moe_token_dispatch_dedup.h)：FIRST 读取一次后扇出写入全部成员，可能提前写入后续 wave。
- [Combine 去重](../op_kernel/arch35/stage/mega_moe_gmm2_combine_dedup.h)：LAST 按全局成员行号回读较早结果，在 FP32 中顺序合并后发送。

注意：去重组是同一个源 token 在本卡的专家成员集合，与 256 行 M group 是两个不同概念。去重减少通信，不取消各成员的专家计算。

## 3. 符号和动态 wave 预算

| 符号 | 含义 |
| --- | --- |
| B | 归一化后的 numMaxTokensPerRank；属性为 0 时，当前 tiling 以实际 bs 替代 |
| E | epWorldSize |
| S = E × B | 所有源卡可能提供的唯一 token 数上界；token 身份为 (srcRank, tokenIndex) |
| L = min(topK, moeExpertPerRank) | 同一个 token 在本卡最多有多少个专家成员 |
| Rcap | maxOutputSize，即原 workspace 的路由行预留容量 |
| Rbar = min(Rcap, S × L) | 有效接收路由行数上界；允许 Rcap 大于实际路由上界的 MTE 配置 |
| H | h，输入和最终输出维度 |
| I | hiddenDim / 2，Activation 后的中间维度 |
| W | 本次 tiling 的 mGroupsPerWave |
| C_D、C_A、C_1、C_2 | Dispatch、Activation、GMM1、GMM2 各自的循环槽位数 |
| delta | GMM1 落 GM 时为 1，否则为 0 |

当前 wave 预算在 tiling 中计算，并在一次 kernel 调用中保持不变：

```text
W = max(
    ceil(aicNum × 4 / ceil(2I / 256)),
    ceil(aicNum     / ceil(H  / 256))
)
```

公式中的 4 是每核 GMM1 的逻辑 N tile 预算，不是固定的 M group 数。

每个 wave 最多占 W 个 256 行槽位。C_* 应由 W 和各阶段在途深度推导，启动预填、提前 Dispatch、GMM2 滞后及消费者完成等待均需纳入，本文不固定倍数。若采用容纳 n_* 个完整 wave 的预算，则 C_* = n_* × W。

循环 buffer 的物理行容量为 256 × C_*。有效行数不等于物理容量，专家尾 group 的空余行不能分配给下一专家段。

## 4. 每行数据字节数和对齐口径

```text
d(H) = H       // FP8 Dispatch 输入
     = H / 2   // FP4 Dispatch 输入

s(X) = 2 × ceil(X / 64)   // MX scale，当前常量为 64 个元素一组、每组 2 字节

q(K) = align32(4 × K)    // 需要保留整份 topK FP32 权重时的记录大小
A512(x) = ceil(x / 512) × 512
```

数据主体比较忽略对齐。实际布局需对每个独立区域执行对齐，并检查乘法、求和溢出；例如：

```text
Dispatch ring  = A512(256 C_D d(H)) + A512(256 C_D s(H))
Activation ring = A512(256 C_A I) + A512(256 C_A s(I))
GMM1 ring      = delta × A512(256 C_1 × 4I)
GMM2 ring      = A512(256 C_2 × 2H)
```

缓存可以将 data、scale、weights 分区存放，也可以按记录打包。若打包，则须用实际记录对齐后的 stride 计算总容量，不能把未对齐的逐项和直接作为记录跨度。

## 5. Dispatch 跨窗口去重缓存

### 5.1 生命周期与精确需求

令 token t 截断后的有效成员数为 m_t。为了保留一次远端读取的通信收益，需要保存一份输入和 scale，后续成员到达执行窗口时再从缓存复制到 Dispatch ring。

```text
开始：该 token 首次远端读取
结束：最后一个有效成员的数据和 scale 已复制完成

U_D(p) = 时刻 p 已开始、尚未结束的多成员 token 数
U_D_peak = max_p U_D(p)
```

这里的结束是异步复制实际完成，不只是提交完成。缓存读取涉及的源数据不能提前覆盖。

m_t = 1 不需要跨窗口长期缓存。即使原路由有多个成员、截断后 rowDesc 仍标记 GROUP，也可只处理唯一幸存成员；仍须保留原 canonical 发送槽和权重语义。

### 5.2 安全条目上界

每个长期缓存的 token 至少占两条有效路由行，所有成员行总数不超过 Rbar，因此：

```text
U_bound = 0                              当 L < 2
U_bound = min(S, floor(Rbar / 2))          当 L >= 2

U_D_peak <= U_bound
```

该上界不依赖 wave 大小。最坏路由可把所有 token 的 FIRST 放在前部专家、LAST 放在后部专家，使大量缓存同时存活。

### 5.3 容量与收益

忽略对齐、管理结构：

```text
M_Dcache <= U_bound × [d(H) + s(H) + q_optional]

M_D_old = Rcap × [d(H) + s(H)]
M_D_new = 256 C_D × [d(H) + s(H)] + M_Dcache
```

q_optional 仅在权重不能从其他长期有效位置取得时增加。现有权重区是否保留、是否改为 token 记录，需单独决定，不能重复计算节省或存储。

若 Rcap = S × L，且按最坏情况 U_bound = S，则输入缓存主体相对原全量 Dispatch 区约为 1/L，再加循环区和必要权重。

Dispatch 去重单项节省条件（忽略权重、对齐）：

```text
256 C_D + U_bound < Rcap
```

## 6. Combine 跨窗口待合并区

### 6.1 方案假设

本节选择每个待合并 token 保存一个 FP32 部分和。当前 wave 的 GMM2 输出就绪后，及时读取成员贡献并转入部分和，然后允许释放 GMM2 ring 的对应数据。

```text
首成员：BF16 输出转 FP32，按当前模式处理权重，建立部分和
中间成员：按原成员顺序处理权重、加入部分和
LAST：完成累加，最终转换成发送格式并发送，释放记录
```

必须保持原成员顺序、加权位置及最后一次转换语义。不能每轮把部分和转回 BF16，也不能直接用无序原子加代替顺序合并。FP32 部分和在 GM 中保存不意味着上述顺序和同步自动成立，仍需指定每个 token 的合并执行者。

当前去重 Combine 按完整专家完成后消费，需要改成能及时吸收已完成 wave 的成员贡献；只新增部分和而不调整消费时机，无法回收跨 wave 专家的早期 GMM2 输出。

### 6.2 条目数和容量上界

```text
U_C(p) = 已吸收首成员、尚未完成最后成员合并的 token 数
U_C_peak = max_p U_C(p)
U_C_peak <= U_bound

M_Caccum <= U_bound × 4H
```

单成员无需长期部分和，可沿用正确加权和 canonical 槽语义立即完成发送。

若最后一次发送直接读取待合并区，需等发送读取完成再释放；若先复制到独立发送缓冲区，则等该复制读取完成即可释放。本上界假设记录在该读取结束后及时回收；另行保留已闭合组或扩大发送队列时，须追加在途容量。

### 6.3 与原 GMM2 全量结果区比较

```text
M_2_old = Rcap × 2H
M_2_new = 256 C_2 × 2H + U_bound × 4H

单项节省条件：256 C_2 + 2 U_bound < Rcap
```

当 Rcap = S × L、U_bound = S 时，仅比较 FP32 部分和与原 BF16 全量输出：

| L | 部分和 / 原 GMM2 输出 |
| --- | --- |
| 2 | 100% |
| 4 | 50% |
| 8 | 25% |

L = 2 时，两个 BF16 结果与一个 FP32 部分和都占 4H 字节，单项没有容量优势，加上 GMM2 ring 还可能更大。总体仍可能因 Activation 和 GMM1 循环化而节省空间。

替代方案是缓存未合并的 BF16 成员结果，保留现有 LAST 回读方式。它更接近现有数值流程，但每个组可能保留 m_t-1 行，不能按一条记录容量计算；该方案未计入上述部分和模型。

## 7. 总数据主体比较

忽略对齐、权重、映射和同步结构：

```text
M_old =
    Rcap × [d(H) + s(H)]
  + Rcap × [I + s(I)]
  + delta × Rcap × 4I
  + Rcap × 2H

M_ring =
    256 C_D × [d(H) + s(H)]
  + 256 C_A × [I + s(I)]
  + delta × 256 C_1 × 4I
  + 256 C_2 × 2H

M_new = M_ring
      + D_on × U_bound × [d(H) + s(H)]
      + C_on × U_bound × 4H
```

D_on 和 C_on 表示最终开启的 Dispatch/Combine 去重，分别取 0 或 1。若开关使用不同缓存容量上界，应分别替换。

同时启用两种去重时，两类保留数据可能同时存活，不能默认共用地址区。该公式采用独立保留区的保守求和。

## 8. 数值示例

仅用于容量比较，不固定 wave 或循环区参数：

```text
S = 65,536
L = 4
Rcap = Rbar = 262,144
H = 4096
I = 2048
输入 FP8，GMM1 落 GM，两种去重均开启
U_bound = 65,536
1 MiB = 1,048,576 字节
```

| 区域 | 原主体容量 | 去重保留区上界 |
| --- | ---: | ---: |
| Dispatch data + scale | 1056 MiB | 264 MiB |
| Activation data + scale | 528 MiB | 0 |
| GMM1 BF16 结果 | 2048 MiB | 0 |
| GMM2 BF16 结果 | 2048 MiB | FP32 部分和 1024 MiB |
| 合计 | 5680 MiB | 1288 MiB + 四套循环区 |

该例各循环区每个 256 行槽位的主体容量为：

```text
Dispatch：1.03125 MiB
Activation：0.515625 MiB
GMM1：2 MiB
GMM2：2 MiB

M_new主体(MiB) = 1288
              + 1.03125 C_D + 0.515625 C_A + 2 C_1 + 2 C_2
```

这不是整个算子的显存量或最终申请值，尚未包含下节列出的其他区域。

## 9. 实际申请与管理开销

### 9.1 紧凑槽位与直接索引

按 U_bound 申请缓存，需要 token 到保留槽位的映射和槽位回收。映射可复用/扩展 rowDesc，也可使用单独索引表；其内存、初始化和访问成本要明确。

若简单使用 srcRank × B + tokenIndex 直接索引，每个保留区实际需申请 S 条记录，而不是 U_bound 条。Rcap 较小或接收截断较强时，两者差异可能显著。

例如使用 S 个 int32 索引项的直接映射表，仅索引就占 4S 字节；槽位描述、空闲列表和有效性标志还需另外计算。具体是否可以共享索引，取决于两个保留区的生命周期，不在本推导中假定。

### 9.2 不能从主体比较中省略的区域

最终 workspace 必须加入：

- 必要的权重记录和缓存管理结构。
- metadata、rowDesc、专家计数、cumsum、tile 状态和同步标志。
- 共享专家输入、中间结果、输出等仍然保留的区域。
- 各分区对齐及布局间填充。

通信窗口前部和 LibAPI 系统 workspace 的边界沿用 PR #13256。本文缩小的是后部业务 workspace，不能把通信前部容量作为已节省量。

### 9.3 外部申请上界与实际用量

SymmBuffer 在真实路由进入 kernel 前创建，不能按实测 U_D_peak/U_C_peak 临时缩小申请。外部 helper 需用已知模型参数和已核实的平台核数上界，计算 W_upper、C_*_upper 以及安全去重容量。

Tiling 根据实际配置生成布局、校验窗口容量；kernel/prepass 根据真实路由使用缓存条目。减少实际使用条目数不自动减少已申请的 CCL 空间，必须同步修改 Torch size helper 才能得到申请量收益。

## 10. 实现前需要闭合的事项

1. 四类 C_* 与启动双轮 Dispatch、提前生产和 GMM2 滞后的容量关系。
2. wave 内物理 group 编号、专家尾块填充及回绕拆段。
3. 覆盖前对旧数据最后一次读取完成的同步证明，避免等待环。
4. Dispatch 从一次全组提前扇出改为按有效执行窗口复制，并正确发布 ready。
5. Combine 及时吸收各 wave 输出、顺序累加及部分和释放时机。
6. 截断后的单幸存成员、canonical 发送槽与不同 topkWeightsPrefetch 模式。
7. 紧凑缓存映射的容量上界和全部管理开销。
8. host/device 布局与 Torch 保守申请公式一致。

这些公式说明容量可行性，不替代同步正确性和数值验证。后续需覆盖跨 wave 成员、专家尾 group、强接收截断、连续多次回绕及两种去重开关组合。
